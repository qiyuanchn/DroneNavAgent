"""Mixed mission schema + executor (goal.md M2).

These tests pin the contract that matters:

* steps run strictly in order and a failure stops the mission with the
  remaining steps still ``pending``;
* a semantic step is done only on candidate_id/arrived/accepted/verified - a
  service that says ``ok=True`` but ``accepted=False`` must not complete;
* motion steps run through the M1 executor, store their ``motion_state``, and
  a cancelled/resumed motion step continues its measured distance/time;
* pause is cooperative between steps, cancel keeps progress, and completed
  steps are never replayed;
* state (and its progress) is JSON-serializable and persisted after each step.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vlm_nav.actions import action_to_body_delta, body_delta_to_world
from vlm_nav.mixed_mission import (
    MIXED_BLOCKED,
    MIXED_CANCELLED,
    MIXED_COMPLETED,
    MIXED_FAILED,
    MIXED_PAUSED,
    MIXED_STEP_BLOCKED,
    MIXED_STEP_DONE,
    MIXED_STEP_FAILED,
    MIXED_STEP_PENDING,
    MixedMissionError,
    MixedMissionExecutor,
    MixedMissionSpec,
    MixedMissionState,
    format_subgoal,
    load_state,
    save_state,
)
from vlm_nav.motion import MotionExecutor, VirtualClock
from vlm_nav.tools import ToolContext


class FakeWorld:
    """Kinematic world with optional circular bounds, mirroring apply_action."""

    def __init__(self, bound_radius: float | None = None) -> None:
        self.position = np.array([0.0, 0.0, 1.0])
        self.yaw = 0.0
        self.bound_radius = bound_radius
        self.calls = 0
        self.renders = 0
        self.trajectory = [tuple(self.position)]

    @property
    def position_flu(self):
        return self.position

    @property
    def yaw_rad(self):
        return self.yaw

    def apply_action(self, action):
        self.calls += 1
        dx, dy, dz, dyaw = action_to_body_delta(action)
        self.yaw += dyaw
        wx, wy, wz = body_delta_to_world(dx, dy, dz, self.yaw)
        new = self.position + np.array([wx, wy, wz])
        clamped = False
        if self.bound_radius is not None:
            radius = math.hypot(new[0], new[1])
            if radius > self.bound_radius:
                scale = self.bound_radius / radius
                new[0] *= scale
                new[1] *= scale
                clamped = True
        self.position = new
        self.trajectory.append(tuple(new))
        return "clamped" if clamped else "ok"

    def render(self):
        self.renders += 1
        return np.zeros((3, 8, 8), dtype=np.float32)


class CapableWorld(FakeWorld):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.took_off = False

    def takeoff(self):
        self.took_off = True
        return "airborne"


class FakeSemanticService:
    def __init__(
        self,
        *,
        candidate_id: str = "cand_1",
        accepted: bool = True,
        verified: bool = True,
        arrived: bool = True,
        move_m: float = 0.0,
        status: str = "ok",
        confirmation_required: bool = False,
    ) -> None:
        self.candidate_id = candidate_id
        self.accepted = accepted
        self.verified = verified
        self.arrived = arrived
        self.move_m = move_m
        self.status = status
        self.confirmation_required = confirmation_required
        self.calls: list[tuple[str, str]] = []
        self.revisit_args: list[dict] = []

    def _reply(self, tool: str, **fields) -> dict:
        data = {
            "ok": True,
            "tool": tool,
            "status": self.status,
            "candidate_id": self.candidate_id,
            "task_id": "task_1",
            "arrived": self.arrived,
            "accepted": self.accepted,
            "verified": self.verified,
            "calls": 1,
            "error": "",
            "data": {},
        }
        data.update(fields)
        return data

    def _move(self, ctx) -> None:
        if self.move_m and ctx.world is not None:
            ctx.world.position[0] += self.move_m

    def search(self, ctx, target, *, relation="", instruction=None,
               max_steps=8, max_turns=1):
        self.calls.append(("search", str(target)))
        return self._reply("search_target")

    def approach(self, ctx, candidate_id=None, *, standoff_m=1.2, max_legs=8):
        self.calls.append(("approach", str(candidate_id or "")))
        self._move(ctx)
        return self._reply(
            "approach_candidate",
            arrived=self.arrived,
            confirmation_required=self.confirmation_required,
        )

    def confirm(self, ctx, candidate_id=None, *, instruction=None,
                relation=None, max_views=2):
        self.calls.append(("confirm", str(candidate_id or "")))
        return self._reply("confirm_candidate", accepted=self.accepted)

    def verify(self, ctx, candidate_id=None, *, success_dist_m=None):
        self.calls.append(("verify", str(candidate_id or "")))
        return self._reply("verify_goal", verified=self.verified)

    def revisit(self, ctx, *, position=None, observation_id=None,
                viewpoint_id=None, standoff_m=0.6):
        self.calls.append(("revisit", str(position or observation_id or viewpoint_id)))
        self.revisit_args.append({
            "position": position,
            "observation_id": observation_id,
            "viewpoint_id": viewpoint_id,
            "standoff_m": standoff_m,
        })
        self._move(ctx)
        return self._reply("return_to_observation", arrived=self.arrived)

    def goto(self, ctx, position, *, standoff_m=0.6):
        self.calls.append(("goto", str(position)))
        self.revisit_args.append({"position": position, "standoff_m": standoff_m})
        self._move(ctx)
        return self._reply("go_to_viewpoint", arrived=self.arrived)


def _executor(world, tmp, **kwargs):
    ctx = kwargs.pop("ctx", None) or ToolContext(world=world, extra={})
    return MixedMissionExecutor(
        world, ctx=ctx, artifacts_dir=tmp, **kwargs
    ), ctx


class SchemaTest(unittest.TestCase):
    def test_roundtrip_motion_and_semantic_steps(self):
        payload = {
            "mission_id": "mixed_roundtrip",
            "instruction": "up, forward, search, approach, confirm, verify",
            "steps": [
                {"kind": "move", "direction": "up", "meters": 1.0},
                {"kind": "turn", "direction": "right", "degrees": 90},
                {"kind": "hover", "seconds": 3},
                {"kind": "photo", "label": "window chair"},
                {"kind": "search", "target": "窗边的木椅", "relation": "窗边",
                 "max_steps": 4},
                {"kind": "approach", "target": "last", "standoff_m": 1.2},
                {"kind": "confirm", "target": "last", "max_views": 2},
                {"kind": "verify", "target": "last", "success_dist_m": 1.2},
                {"kind": "revisit", "place": "kitchen"},
                {"kind": "goto", "position": [1, 2, 1]},
                {"kind": "takeoff"},
            ],
        }
        spec = MixedMissionSpec.from_dict(payload)
        self.assertEqual(len(spec.steps), 11)
        self.assertEqual(spec.steps[4]["relation"], "窗边")
        self.assertEqual(spec.steps[9]["position"], [1.0, 2.0, 1.0])
        restored = MixedMissionSpec.from_dict(spec.to_dict())
        self.assertEqual(restored.to_dict(), spec.to_dict())

    def test_bare_step_list_and_json(self):
        spec = MixedMissionSpec.from_dict([
            {"kind": "move", "direction": "forward", "meters": 2},
        ])
        self.assertTrue(spec.mission_id.startswith("mission_"))
        restored = MixedMissionSpec.from_json(json.dumps(spec.to_dict()))
        self.assertEqual(restored.to_dict(), spec.to_dict())

    def test_invalid_steps_rejected(self):
        bad = [
            [{"kind": "walk", "meters": 1}],
            [{"kind": "search"}],
            [{"kind": "search", "target": "  "}],
            [{"kind": "goto"}],
            [{"kind": "goto", "position": [1, 2]}],
            [{"kind": "revisit"}],
            [{"kind": "move", "direction": "forward"}],
            [{"kind": "move", "direction": "sideways", "meters": 1}],
        ]
        for steps in bad:
            with self.subTest(steps=steps):
                with self.assertRaises(MixedMissionError):
                    MixedMissionSpec.from_dict(steps)
        with self.assertRaises(MixedMissionError):
            MixedMissionSpec.from_dict({
                "schema": "drone.mixed_mission/999",
                "steps": [{"kind": "takeoff"}],
            })

    def test_state_json_roundtrip(self):
        spec = MixedMissionSpec.from_dict([
            {"kind": "move", "direction": "forward", "meters": 2},
            {"kind": "search", "target": "chair"},
        ])
        state = MixedMissionState.new(spec, clock_s=1.5)
        state.steps[0].status = MIXED_STEP_DONE
        state.steps[0].attempts = 1
        state.steps[0].motion_state = {"schema": "x"}
        state.status = MIXED_PAUSED
        state.current_step = 1
        state.subgoal = format_subgoal(state)
        restored = MixedMissionState.from_dict(state.to_dict())
        self.assertEqual(restored.to_dict(), state.to_dict())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            save_state(state, path)
            self.assertEqual(load_state(path).to_dict(), state.to_dict())
        with self.assertRaises(MixedMissionError):
            MixedMissionState.from_dict({
                "schema": "drone.mixed_mission.state/999",
                "mission_id": "x",
                "steps": [{"kind": "takeoff", "params": {}}],
            })

    def test_subgoal_is_bounded_and_readable(self):
        spec = MixedMissionSpec.from_dict([
            {"kind": "move", "direction": "forward", "meters": 5},
            {"kind": "search", "target": "木椅", "relation": "窗边"},
        ])
        state = MixedMissionState.new(spec)
        line = format_subgoal(state)
        self.assertRegex(line, r"^mission .+ step 0/2: move forward 5m \(2 left\)$")
        self.assertLessEqual(len(line), 160)
        state.steps[0].status = MIXED_STEP_DONE
        line = format_subgoal(state)
        self.assertIn("search 窗边 木椅", line)
        self.assertIn("(1 left)", line)

    def test_subgoal_bounds_a_long_mission_id(self):
        spec = MixedMissionSpec(
            steps=({"kind": "move", "direction": "forward", "meters": 1},),
            mission_id="m" * 400,
        )
        state = MixedMissionState.new(spec)
        line = format_subgoal(state)
        self.assertLessEqual(len(line), 160)
        self.assertIn("step 0/1:", line)
        self.assertIn("(1 left)", line)


class ExecutionTest(unittest.TestCase):
    def test_ordered_execution_and_context_progress(self):
        world = FakeWorld()
        service = FakeSemanticService(move_m=0.5)
        steps = [
            {"kind": "move", "direction": "forward", "meters": 2},
            {"kind": "search", "target": "chair", "relation": "window"},
            {"kind": "approach", "target": "last"},
            {"kind": "confirm", "target": "last"},
            {"kind": "verify", "target": "last"},
            {"kind": "photo", "label": "end"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            executor, ctx = _executor(
                world, tmp, semantic_service=service,
            )
            state = executor.run(MixedMissionSpec.from_dict({"steps": steps}))
            self.assertEqual(state.status, MIXED_COMPLETED)
            self.assertEqual(
                [record.status for record in state.steps],
                [MIXED_STEP_DONE] * len(steps),
            )
            self.assertEqual(service.calls, [
                ("search", "chair"),
                ("approach", "cand_1"),
                ("confirm", "cand_1"),
                ("verify", "cand_1"),
            ])
            self.assertAlmostEqual(world.position[0], 2.5)
            self.assertTrue(Path(state.steps[-1].evidence["artifact"]).exists())
            self.assertTrue(ctx.extra["subgoal"].startswith("mission "))
            self.assertLessEqual(len(ctx.extra["subgoal"]), 160)
            progress = ctx.extra["mission.progress"]
            self.assertEqual(progress["status"], MIXED_COMPLETED)
            self.assertEqual(progress["steps_done"], len(steps))
            self.assertEqual(progress["steps_remaining"], 0)

    def test_false_success_is_not_completion(self):
        cases = [
            ("approach", {"arrived": False, "accepted": True, "verified": True}),
            ("confirm", {"arrived": True, "accepted": False, "verified": True}),
            ("verify", {"arrived": True, "accepted": True, "verified": False}),
        ]
        for kind, flags in cases:
            with self.subTest(kind=kind):
                world = FakeWorld()
                service = FakeSemanticService(
                    accepted=flags["accepted"],
                    verified=flags["verified"],
                    arrived=flags["arrived"],
                )
                steps = [
                    {"kind": "search", "target": "chair"},
                    {"kind": kind, "target": "last"},
                    {"kind": "move", "direction": "forward", "meters": 1},
                ]
                with tempfile.TemporaryDirectory() as tmp:
                    executor, _ = _executor(world, tmp, semantic_service=service)
                    state = executor.run(steps)
                self.assertIn(state.status, (MIXED_FAILED, MIXED_BLOCKED))
                self.assertEqual(state.steps[1].status in (
                    "failed", "blocked"
                ), True)
                self.assertNotEqual(state.steps[1].status, MIXED_STEP_DONE)
                self.assertEqual(state.steps[2].status, MIXED_STEP_PENDING)
                self.assertEqual(world.calls, 0)

    def test_approach_auto_confirms_accepted_instance(self):
        service = FakeSemanticService(confirmation_required=True)
        with tempfile.TemporaryDirectory() as tmp:
            executor, _ = _executor(
                FakeWorld(), tmp, semantic_service=service
            )
            state = executor.run([
                {"kind": "search", "target": "chair"},
                {"kind": "approach", "target": "last"},
                {"kind": "photo", "label": "arrived"},
            ])
        self.assertEqual(state.status, MIXED_COMPLETED)
        self.assertEqual(state.steps[1].status, MIXED_STEP_DONE)
        self.assertTrue(state.steps[1].evidence["confirmation_accepted"])
        self.assertEqual(service.calls, [
            ("search", "chair"),
            ("approach", "cand_1"),
            ("confirm", "cand_1"),
        ])

    def test_approach_requires_accepted_confirmation_when_flagged(self):
        service = FakeSemanticService(
            accepted=False, confirmation_required=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            executor, _ = _executor(
                FakeWorld(), tmp, semantic_service=service
            )
            state = executor.run([
                {"kind": "search", "target": "chair"},
                {"kind": "approach", "target": "last"},
                {"kind": "photo", "label": "after"},
            ])
        self.assertEqual(state.status, MIXED_FAILED)
        self.assertEqual(state.steps[1].status, MIXED_STEP_FAILED)
        self.assertIn("confirmation_required", state.steps[1].error)
        self.assertFalse(state.steps[1].evidence["confirmation_accepted"])
        self.assertEqual(state.steps[2].status, MIXED_STEP_PENDING)
        self.assertEqual(service.calls, [
            ("search", "chair"),
            ("approach", "cand_1"),
            ("confirm", "cand_1"),
        ])

    def test_search_without_candidate_is_not_done(self):
        service = FakeSemanticService(candidate_id="")
        with tempfile.TemporaryDirectory() as tmp:
            executor, _ = _executor(
                FakeWorld(), tmp, semantic_service=service
            )
            state = executor.run([
                {"kind": "search", "target": "chair"},
                {"kind": "photo"},
            ])
        self.assertEqual(state.status, MIXED_FAILED)
        self.assertIn("no candidate_id", state.steps[0].error)
        self.assertEqual(state.steps[1].status, MIXED_STEP_PENDING)

    def test_missing_semantic_service_is_structured_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor, _ = _executor(FakeWorld(), tmp)
            state = executor.run([
                {"kind": "search", "target": "chair"},
                {"kind": "move", "direction": "forward", "meters": 1},
            ])
        self.assertEqual(state.status, MIXED_FAILED)
        self.assertIn("semantic_service_unavailable", state.steps[0].error)

    def test_mid_motion_cancel_resumes_remaining_distance(self):
        world = FakeWorld()
        with tempfile.TemporaryDirectory() as tmp:
            executor, _ = _executor(world, tmp)
            chunks: list[int] = []

            def on_chunk(event):
                if event["event"] == "step.chunk" and len(chunks) == 0:
                    chunks.append(1)
                    executor.cancel("mid-motion stop")

            executor.motion.on_progress = on_chunk
            state = executor.run([
                {"kind": "move", "direction": "forward", "meters": 5},
                {"kind": "photo", "label": "after"},
            ])
            self.assertEqual(state.status, MIXED_CANCELLED)
            self.assertEqual(state.steps[0].status, "cancelled")
            self.assertEqual(state.steps[1].status, MIXED_STEP_PENDING)
            partial = state.steps[0].motion_state["steps"][0]["progress"]
            self.assertAlmostEqual(partial["moved_m"], 2.0)
            self.assertAlmostEqual(partial["remaining_m"], 3.0)
            executor.motion.on_progress = None
            resumed = executor.resume(state)
        self.assertEqual(resumed.status, MIXED_COMPLETED)
        self.assertAlmostEqual(world.position[0], 5.0)
        progress = resumed.steps[0].motion_state["steps"][0]["progress"]
        self.assertAlmostEqual(progress["moved_m"], 5.0)
        self.assertEqual(progress["remaining_m"], 0.0)
        self.assertEqual(resumed.steps[0].attempts, 2)

    def test_motion_chunk_progress_is_kept_when_blocked(self):
        world = FakeWorld(bound_radius=3.0)
        with tempfile.TemporaryDirectory() as tmp:
            executor, _ = _executor(world, tmp)
            state = executor.run([
                {"kind": "move", "direction": "forward", "meters": 5},
                {"kind": "hover", "seconds": 1},
            ])
        self.assertEqual(state.status, MIXED_BLOCKED)
        self.assertEqual(state.steps[0].status, MIXED_STEP_BLOCKED)
        motion = state.steps[0].motion_state
        self.assertIsNotNone(motion)
        self.assertEqual(motion["steps"][0]["status"], "blocked")
        self.assertAlmostEqual(
            motion["steps"][0]["progress"]["remaining_m"], 2.0, places=3
        )
        self.assertEqual(state.steps[1].status, MIXED_STEP_PENDING)
        self.assertEqual(world.position[0], 3.0)

    def test_pause_between_steps_and_resume_skips_completed_step(self):
        world = FakeWorld()
        clock = VirtualClock()
        paused = []
        holder = {}

        def on_progress(payload):
            if (
                payload["event"] == "step.finished"
                and payload["step"]["index"] == 0
                and not paused
            ):
                paused.append(True)
                holder["executor"].pause("test pause")

        with tempfile.TemporaryDirectory() as tmp:
            executor, ctx = _executor(
                world, tmp, clock=clock, on_progress=on_progress
            )
            holder["executor"] = executor
            state = executor.run([
                {"kind": "move", "direction": "forward", "meters": 1},
                {"kind": "turn", "direction": "right", "degrees": 90},
                {"kind": "hover", "seconds": 1},
            ])
            self.assertEqual(state.status, MIXED_PAUSED)
            self.assertEqual(state.current_step, 1)
            self.assertEqual(state.steps[0].status, MIXED_STEP_DONE)
            self.assertEqual(world.calls, 1)  # one 1 m move chunk
            state = executor.resume()
        self.assertEqual(state.status, MIXED_COMPLETED)
        self.assertEqual(state.steps[0].attempts, 1)
        # 2 turn chunks (45 deg each) and no replayed move chunk.
        self.assertEqual(world.calls, 3)
        self.assertAlmostEqual(clock.now_s(), 1.0)

    def test_cancel_between_steps_then_resume(self):
        paused = []
        holder = {}

        def on_progress(payload):
            if (
                payload["event"] == "step.finished"
                and payload["step"]["index"] == 0
                and not paused
            ):
                paused.append(True)
                holder["executor"].cancel("interrupt")

        with tempfile.TemporaryDirectory() as tmp:
            executor, _ = _executor(FakeWorld(), tmp, on_progress=on_progress)
            holder["executor"] = executor
            state = executor.run([
                {"kind": "move", "direction": "forward", "meters": 1},
                {"kind": "move", "direction": "forward", "meters": 1},
            ])
            self.assertEqual(state.status, MIXED_CANCELLED)
            self.assertEqual(state.steps[0].status, MIXED_STEP_DONE)
            self.assertEqual(state.steps[1].status, MIXED_STEP_PENDING)
            state = executor.resume()
        self.assertEqual(state.status, MIXED_COMPLETED)

    def test_cancel_mid_motion_and_resume_hover_remaining_time(self):
        world = FakeWorld()
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp:
            executor, _ = _executor(world, tmp, clock=clock)
            original_advance = clock.advance
            fired = []

            def advance(dt):
                now = original_advance(dt)
                if now >= 1.0 and not fired:
                    fired.append(now)
                    executor.cancel("mid-hover")
                return now

            clock.advance = advance
            state = executor.run([{"kind": "hover", "seconds": 3}])
            self.assertEqual(state.status, MIXED_CANCELLED)
            self.assertEqual(state.steps[0].status, "cancelled")
            motion = state.steps[0].motion_state
            elapsed = motion["steps"][0]["progress"]["elapsed_s"]
            self.assertGreaterEqual(elapsed, 1.0)
            self.assertLess(elapsed, 1.5)
            clock.advance = original_advance
            state = executor.resume()
        self.assertEqual(state.status, MIXED_COMPLETED)
        self.assertEqual(state.steps[0].attempts, 2)
        self.assertAlmostEqual(clock.now_s(), 3.0, places=6)
        self.assertAlmostEqual(
            state.steps[0].motion_state["steps"][0]["progress"]["elapsed_s"],
            3.0,
            places=6,
        )

    def test_cancel_mid_move_and_resume_remaining_distance(self):
        world = FakeWorld()

        # Patch after construction so the first chunk can trigger the cancel.
        with tempfile.TemporaryDirectory() as tmp:
            executor, _ = _executor(world, tmp)
            hooked = []

            def apply(action):
                result = FakeWorld.apply_action(world, action)
                if world.calls == 1 and not hooked:
                    hooked.append(True)
                    executor.cancel("stop-move")
                return result

            world.apply_action = apply
            state = executor.run([
                {"kind": "move", "direction": "forward", "meters": 5},
            ])
            self.assertEqual(state.status, MIXED_CANCELLED)
            progress = state.steps[0].motion_state["steps"][0]["progress"]
            self.assertAlmostEqual(progress["moved_m"], 2.0, places=3)
            self.assertEqual(progress["chunks"], 1)
            calls_after_cancel = world.calls

            world.apply_action = lambda action: FakeWorld.apply_action(world, action)
            state = executor.resume()
        self.assertEqual(state.status, MIXED_COMPLETED)
        self.assertEqual(state.steps[0].attempts, 2)
        self.assertAlmostEqual(world.position[0], 5.0, places=6)
        self.assertEqual(
            state.steps[0].motion_state["steps"][0]["progress"]["remaining_m"], 0.0
        )
        # 2 m in the interrupted attempt, then 2 m + 1 m on resume: the first
        # chunk is preserved, not replayed.
        self.assertEqual(world.calls, calls_after_cancel + 2)

    def test_state_path_persisted_after_every_step(self):
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mission.state.json"

            def on_progress(payload):
                if payload["event"] == "step.finished":
                    seen.append(json.loads(path.read_text(encoding="utf-8")))

            executor, _ = _executor(
                FakeWorld(), tmp, on_progress=on_progress, state_path=path
            )
            state = executor.run([
                {"kind": "move", "direction": "forward", "meters": 1},
                {"kind": "hover", "seconds": 0.2},
                {"kind": "photo", "label": "persist"},
            ])
            self.assertEqual(len(seen), 3)
            self.assertTrue(all(item["status"] == "running" for item in seen[:2]))
            persisted = load_state(path)
        self.assertEqual(state.status, MIXED_COMPLETED)
        self.assertEqual(persisted.to_dict(), state.to_dict())

    def test_capability_gating(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor, _ = _executor(FakeWorld(), tmp)
            state = executor.run([{"kind": "takeoff"}])
            self.assertEqual(state.status, MIXED_FAILED)
            self.assertIn(
                "unsupported_capability:takeoff", state.steps[0].error
            )

        world = CapableWorld()
        with tempfile.TemporaryDirectory() as tmp:
            executor, _ = _executor(world, tmp)
            state = executor.run([{"kind": "takeoff"}, {"kind": "land"}])
        # land() does not exist on the fake world: gated separately.
        self.assertEqual(state.status, MIXED_FAILED)
        self.assertEqual(state.steps[0].status, MIXED_STEP_DONE)
        self.assertTrue(world.took_off)
        self.assertIn("unsupported_capability:land", state.steps[1].error)

    def test_revisit_resolves_place_from_ctx_extra(self):
        service = FakeSemanticService()
        with tempfile.TemporaryDirectory() as tmp:
            ctx = ToolContext(
                world=FakeWorld(),
                extra={"places": {"kitchen": [1.0, 2.0, 1.0]}},
            )
            executor = MixedMissionExecutor(
                ctx.world, ctx=ctx, semantic_service=service,
                artifacts_dir=tmp,
            )
            state = executor.run([
                {"kind": "revisit", "place": "kitchen"},
                {"kind": "goto", "position": [3, 4, 1]},
            ])
        self.assertEqual(state.status, MIXED_COMPLETED)
        self.assertEqual(service.revisit_args[0]["position"], [1.0, 2.0, 1.0])
        self.assertEqual(service.revisit_args[1]["position"], [3.0, 4.0, 1.0])

    def test_resume_from_persisted_state_with_new_executor(self):
        world = FakeWorld()
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            first = FakeSemanticService(arrived=False)
            executor = MixedMissionExecutor(
                world, clock=clock, semantic_service=first,
                artifacts_dir=tmp, state_path=path,
            )
            state = executor.run([
                {"kind": "move", "direction": "forward", "meters": 2},
                {"kind": "search", "target": "chair"},
                {"kind": "approach", "target": "last"},
            ])
            self.assertEqual(state.status, MIXED_FAILED)
            calls_after_failure = world.calls
            loaded = load_state(path)
            self.assertEqual(loaded.status, MIXED_FAILED)

            second = FakeSemanticService(arrived=True)
            resumed = MixedMissionExecutor(
                world, clock=clock, semantic_service=second,
                artifacts_dir=tmp, state_path=path,
            ).resume(loaded)
        self.assertEqual(resumed.status, MIXED_COMPLETED)
        self.assertEqual(world.calls, calls_after_failure)  # no motion replay
        self.assertEqual(second.calls, [("approach", "cand_1")])
        self.assertEqual(resumed.steps[0].attempts, 1)
        self.assertEqual(resumed.steps[1].attempts, 1)
        self.assertEqual(resumed.steps[2].attempts, 2)

    def test_provided_motion_executor_shares_simulation_clock(self):
        world = FakeWorld()
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp:
            motion = MotionExecutor(world, clock=clock, artifacts_dir=tmp)
            executor = MixedMissionExecutor(
                world, motion_executor=motion, artifacts_dir=tmp,
            )
            state = executor.run([
                {"kind": "hover", "seconds": 0.5},
                {"kind": "move", "direction": "forward", "meters": 1},
            ])
        self.assertEqual(state.status, MIXED_COMPLETED)
        self.assertAlmostEqual(clock.now_s(), 0.5)
        self.assertAlmostEqual(state.clock_s, 0.5)
        self.assertIs(executor.motion.clock, clock)

    def test_revisit_resolves_place_from_memory(self):
        service = FakeSemanticService()

        class Memory:
            def place_position(self, name):
                if name == "kitchen":
                    return {"position": [2.0, 1.0, 1.0]}
                return None

        with tempfile.TemporaryDirectory() as tmp:
            ctx = ToolContext(world=FakeWorld(), memory=Memory(), extra={})
            executor = MixedMissionExecutor(
                ctx.world, ctx=ctx, semantic_service=service,
                artifacts_dir=tmp,
            )
            state = executor.run([{"kind": "revisit", "place": "kitchen"}])
        self.assertEqual(state.status, MIXED_COMPLETED)
        self.assertEqual(service.revisit_args[0]["position"], [2.0, 1.0, 1.0])

    def test_run_requires_at_least_one_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor, _ = _executor(FakeWorld(), tmp)
            with self.assertRaises(MixedMissionError):
                executor.run([])


if __name__ == "__main__":
    unittest.main()
