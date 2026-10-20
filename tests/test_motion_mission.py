"""Task-level motion execution (goal.md M1).

These tests pin the semantics the plugin must not lose:

* a 5 m request is segmented into <=2 m chunks but completed only when the
  *cumulative target displacement* is reached;
* a world that clamps the drone reports ``blocked`` with the remaining distance
  instead of claiming the truncated move succeeded;
* "forward" is the heading at the step's start, even if the body is rotated
  while the step is running;
* hover advances simulation time and cancellation/resume continues the
  remaining seconds, not the original request;
* cancel/resume keeps measured progress, skips completed steps, and never
  retakes an existing photo artifact.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from vlm_nav.actions import action_to_body_delta, body_delta_to_world
from vlm_nav.motion import (
    MISSION_COMPLETED,
    MISSION_BLOCKED,
    MISSION_CANCELLED,
    STEP_DONE,
    MissionError,
    MotionExecutor,
    MissionSpec,
    MissionState,
    MoveStep,
    VirtualClock,
    load_state,
    save_state,
    step_from_dict,
)
from vlm_nav.plugins import PluginHost
from vlm_nav.plugins.motion import MOTION_SERVICE, MotionPlugin, MotionService
from vlm_nav.tools import ToolContext, ToolRegistry
from vlm_nav.world import SplatWorldCPU, TargetSpec

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "phase1" / "scenes" / "scenario_scene.ply"
SCRIPT = ROOT / "scripts" / "run_motion_mission.py"


def _load_script():
    name = "drone_script_run_motion_mission"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeWorld:
    """Kinematic world with optional circular bounds, mirroring world.apply_action."""

    def __init__(self, bound_radius: float | None = None) -> None:
        self.position = np.array([0.0, 0.0, 1.0])
        self.yaw = 0.0
        self.bound_radius = bound_radius
        self.calls = 0
        self.renders = 0
        self.turns = 0
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
        if dyaw:
            self.turns += 1
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


class SchemaTest(unittest.TestCase):
    def test_mission_roundtrip(self):
        spec = MissionSpec.from_dict({
            "mission_id": "m1",
            "instruction": "forward, hover, turn",
            "steps": [
                {"kind": "move", "direction": "forward", "meters": 5.0},
                {"kind": "hover", "seconds": 3.0},
                {"kind": "turn", "direction": "left", "degrees": 90},
                {"kind": "photo", "label": "window chair"},
            ],
        })
        self.assertEqual(spec.mission_id, "m1")
        self.assertEqual(len(spec.steps), 4)
        self.assertEqual(spec.steps[2].degrees, 90.0)
        self.assertEqual(spec.steps[3].safe_label, "window_chair")
        restored = MissionSpec.from_dict(spec.to_dict())
        self.assertEqual(restored.to_dict(), spec.to_dict())

    def test_turn_direction_sign(self):
        left = step_from_dict({"kind": "turn", "direction": "left", "degrees": 45})
        right = step_from_dict({"kind": "turn", "direction": "right", "degrees": 45})
        self.assertEqual(left.degrees, 45.0)
        self.assertEqual(right.degrees, -45.0)

    def test_invalid_missions_rejected(self):
        bad = [
            {"steps": []},
            {"steps": [{"kind": "walk", "meters": 1}]},
            {"steps": [{"kind": "move", "direction": "sideways", "meters": 1}]},
            {"steps": [{"kind": "move", "direction": "forward", "meters": 0}]},
            {"steps": [{"kind": "hover", "seconds": -1}]},
            {"steps": [{"kind": "turn", "degrees": 0}]},
        ]
        for data in bad:
            with self.subTest(data=data):
                with self.assertRaises(MissionError):
                    MissionSpec.from_dict(data)

    def test_state_roundtrip_and_clock(self):
        clock = VirtualClock(2.5)
        clock.advance(0.5)
        state = MissionState.new(
            MissionSpec(steps=(MoveStep("forward", 1.0),)), clock_s=clock.now_s()
        )
        state.steps[0].status = STEP_DONE
        state.status = MISSION_COMPLETED
        with tempfile.TemporaryDirectory() as tmp:
            path = save_state(state, Path(tmp) / "state.json")
            loaded = load_state(path)
        self.assertEqual(loaded.to_dict(), state.to_dict())
        self.assertAlmostEqual(loaded.clock_s, 3.0)


class ExecutionTest(unittest.TestCase):
    def _executor(self, world, tmp, **kwargs):
        return MotionExecutor(world, artifacts_dir=tmp, **kwargs)

    def test_full_mission_uses_world_and_sim_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            world = FakeWorld()
            executor = self._executor(world, tmp)
            state = executor.run(MissionSpec.from_dict({
                "mission_id": "full",
                "steps": [
                    {"kind": "move", "direction": "forward", "meters": 5.0},
                    {"kind": "hover", "seconds": 3.0},
                    {"kind": "turn", "direction": "left", "degrees": 90},
                    {"kind": "move", "direction": "forward", "meters": 1.0},
                    {"kind": "photo", "label": "end"},
                ],
            }))
            photo = Path(state.steps[4].artifact)
            self.assertTrue(photo.exists())
            self.assertGreater(photo.stat().st_size, 0)
        self.assertEqual(state.status, MISSION_COMPLETED)
        self.assertEqual(state.completed_steps(), 5)
        self.assertAlmostEqual(state.clock_s, 3.0)
        # three chunks for 5 m (2 + 2 + 1), not one truncated 2 m action
        self.assertEqual(state.steps[0].progress["chunks"], 3)
        self.assertAlmostEqual(state.steps[0].progress["moved_m"], 5.0)
        np.testing.assert_allclose(world.position, [5.0, 1.0, 1.0], atol=1e-6)
        self.assertAlmostEqual(math.degrees(world.yaw), 90.0)

    def test_clamped_world_reports_blocked_with_remaining_distance(self):
        with tempfile.TemporaryDirectory() as tmp:
            world = FakeWorld(bound_radius=3.0)
            executor = self._executor(world, tmp)
            state = executor.run(MissionSpec.from_dict({
                "mission_id": "clamped",
                "steps": [{"kind": "move", "direction": "forward", "meters": 5.0}],
            }))
        self.assertEqual(state.status, MISSION_BLOCKED)
        record = state.steps[0]
        self.assertEqual(record.status, "blocked")
        self.assertAlmostEqual(record.progress["moved_m"], 3.0, places=6)
        self.assertAlmostEqual(record.progress["remaining_m"], 2.0, places=6)
        self.assertIn("clamped", record.error)

    def test_forward_target_is_fixed_to_step_start_heading(self):
        with tempfile.TemporaryDirectory() as tmp:
            world = FakeWorld()
            executor = self._executor(world, tmp)
            rotated = {"done": False}

            def on_progress(event):
                if event["event"] == "step.chunk" and not rotated["done"]:
                    # Something else rotates the body after the first chunk.
                    world.yaw += math.radians(90.0)
                    rotated["done"] = True

            executor.on_progress = on_progress
            state = executor.run(MissionSpec.from_dict({
                "mission_id": "fixed-heading",
                "steps": [{"kind": "move", "direction": "forward", "meters": 5.0}],
            }))
        self.assertEqual(state.status, MISSION_COMPLETED)
        # The target stays on the initial +x axis; the executor converts the
        # remaining world vector into the new body frame each chunk.
        np.testing.assert_allclose(world.position, [5.0, 0.0, 1.0], atol=1e-6)
        self.assertAlmostEqual(state.steps[0].progress["moved_m"], 5.0)

    def test_cancel_hover_and_resume_continues_remaining_sim_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            world = FakeWorld()
            executor = self._executor(world, tmp, hover_dt_s=0.5)
            cancels = {"done": False}

            def on_progress(event):
                if (
                    not cancels["done"]
                    and event["event"] == "step.chunk"
                    and event["kind"] == "hover"
                ):
                    cancels["done"] = True
                    executor.cancel("operator stop")

            executor.on_progress = on_progress
            state = executor.run(MissionSpec.from_dict({
                "mission_id": "hover-cancel",
                "steps": [
                    {"kind": "move", "direction": "forward", "meters": 2.0},
                    {"kind": "hover", "seconds": 3.0},
                ],
            }))
            self.assertEqual(state.status, MISSION_CANCELLED)
            self.assertEqual(state.steps[0].status, STEP_DONE)
            self.assertAlmostEqual(state.clock_s, 0.5)
            self.assertAlmostEqual(state.steps[1].progress["elapsed_s"], 0.5)
            executor.on_progress = None
            resumed = executor.resume()
        self.assertEqual(resumed.status, MISSION_COMPLETED)
        # The hover continued from 0.5 s; it did not restart the full 3 s.
        self.assertAlmostEqual(resumed.clock_s, 3.0)
        self.assertAlmostEqual(resumed.steps[1].progress["elapsed_s"], 3.0)

    def test_resume_does_not_repeat_completed_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            world = FakeWorld()
            executor = self._executor(world, tmp)
            steps = [
                {"kind": "move", "direction": "forward", "meters": 2.0},
                {"kind": "photo", "label": "evidence"},
                {"kind": "turn", "degrees": 45},
                {"kind": "move", "direction": "forward", "meters": 1.0},
            ]
            missions = {"count": 0}

            def on_progress(event):
                if (
                    event["event"] == "step.finished"
                    and event["index"] == 1
                    and missions["count"] == 0
                ):
                    missions["count"] = 1
                    executor.cancel("stop after photo")

            executor.on_progress = on_progress
            state = executor.run(MissionSpec.from_dict({
                "mission_id": "photo-resume", "steps": steps,
            }))
            self.assertEqual(state.status, MISSION_CANCELLED)
            self.assertEqual(state.steps[0].status, STEP_DONE)
            self.assertEqual(state.steps[1].status, STEP_DONE)
            photo = Path(state.steps[1].artifact)
            first_size = photo.stat().st_size
            renders = world.renders
            turns = world.turns
            executor.on_progress = None
            resumed = executor.resume()
            self.assertEqual(world.renders, renders, "photo artifact was retaken")
            self.assertGreater(world.turns, turns)
            self.assertEqual(Path(resumed.steps[1].artifact), photo)
            self.assertEqual(photo.stat().st_size, first_size)
        self.assertEqual(resumed.status, MISSION_COMPLETED)

    def test_resume_reuses_artifact_after_interruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            world = FakeWorld()
            executor = self._executor(world, tmp)
            state = executor.run(MissionSpec.from_dict({
                "mission_id": "photo-interrupted",
                "steps": [{"kind": "photo", "label": "evidence"}],
            }))
            photo = Path(state.steps[0].artifact)
            renders = world.renders
            # Simulate the process dying after the PNG was written but before
            # the step was recorded as done.
            state.status = MISSION_CANCELLED
            state.current_step = 0
            state.steps[0].status = "cancelled"
            resumed = executor.resume(state)
            self.assertEqual(resumed.status, MISSION_COMPLETED)
            self.assertEqual(world.renders, renders, "existing artifact was retaken")
            self.assertTrue(resumed.steps[0].progress["reused"])
            self.assertTrue(photo.exists())

    def test_resume_completed_mission_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            world = FakeWorld()
            executor = self._executor(world, tmp)
            state = executor.run(MissionSpec.from_dict({
                "mission_id": "idempotent",
                "steps": [{"kind": "move", "direction": "forward", "meters": 1.0}],
            }))
            calls = world.calls
            again = executor.resume(state)
        self.assertEqual(again.status, MISSION_COMPLETED)
        self.assertEqual(world.calls, calls)


class MotionPluginToolTest(unittest.TestCase):
    """The tool surface must carry the executor semantics end to end."""

    def test_mission_tools_lifecycle_and_resume_after_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            world = FakeWorld(bound_radius=3.0)
            registry = ToolRegistry()
            host = PluginHost(root_dir=root / "plugins", tool_registry=registry)
            host.install(MotionPlugin(artifacts_dir=root / "artifacts"))
            host.activate("motion.base")
            self.assertIsInstance(host.services.get(MOTION_SERVICE), MotionService)
            ctx = ToolContext(world=world, budget={"tool_calls_left": 20})
            blocked = registry.call("motion_mission", {
                "mission_id": "tool-clamped",
                "steps": [
                    {"kind": "move", "direction": "forward", "meters": 5.0},
                ],
                "artifacts_dir": str(root / "artifacts"),
            }, ctx)
            self.assertEqual(blocked.status, "blocked")
            self.assertEqual(blocked.data["steps"][0]["status"], "blocked")
            self.assertAlmostEqual(
                blocked.data["steps"][0]["progress"]["remaining_m"], 2.0, places=6
            )

            # The obstruction is gone: resume only the remaining 2 m.
            world.bound_radius = None
            resumed = registry.call("motion_resume", {}, ctx)
            self.assertEqual(resumed.status, "ok")
            self.assertEqual(resumed.data["status"], MISSION_COMPLETED)
            self.assertAlmostEqual(
                resumed.data["steps"][0]["progress"]["moved_m"], 5.0, places=6
            )
            self.assertGreaterEqual(resumed.data["steps"][0]["progress"]["chunks"], 3)

            reported = registry.call("motion_state", {}, ctx)
            self.assertEqual(reported.status, "ok")
            self.assertEqual(reported.data["mission_id"], "tool-clamped")

            moved = registry.call(
                "motion_move", {"direction": "right", "meters": 2.0}, ctx
            )
            self.assertEqual(moved.status, "ok")
            self.assertEqual(moved.data["status"], MISSION_COMPLETED)
            self.assertAlmostEqual(world.position[0], 5.0, places=6)
            # FLU: +y is left, so "right 2 m" at yaw 0 lands on -2 m.
            self.assertAlmostEqual(world.position[1], -2.0, places=6)

            photo = registry.call(
                "motion_photo",
                {"label": "evidence", "artifacts_dir": str(root / "artifacts")},
                ctx,
            )
            self.assertEqual(photo.status, "ok")
            artifact = Path(photo.data["artifacts"][0])
            self.assertTrue(artifact.exists())

            self.assertEqual(
                registry.names(),
                [
                    "motion_cancel", "motion_hover", "motion_mission",
                    "motion_move", "motion_photo", "motion_resume",
                    "motion_state", "motion_turn",
                ],
            )
            host.deactivate("motion.base")
            self.assertEqual(registry.names(), [])
            report = host.uninstall("motion.base")
            self.assertEqual(report.status, "uninstalled")
            self.assertTrue(artifact.exists(), "mission evidence must survive uninstall")

    def test_single_step_validation_is_a_structured_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            world = FakeWorld()
            registry = ToolRegistry()
            host = PluginHost(root_dir=Path(tmp) / "plugins", tool_registry=registry)
            host.install(MotionPlugin(artifacts_dir=Path(tmp) / "artifacts"))
            host.activate("motion.base")
            ctx = ToolContext(world=world, budget={"tool_calls_left": 5})
            result = registry.call("motion_move", {"meters": 0}, ctx)
        self.assertEqual(result.status, "failed")
        self.assertIn("meters", result.data["error"])


class AcceptanceScriptTest(unittest.TestCase):
    def test_script_runs_full_lifecycle(self):
        module = _load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc = module.main([
                "--width", "24", "--height", "16",
                "--out", str(root / "out"),
                "--artifacts-dir", str(root / "artifacts"),
                "--plugin-root", str(root / "plugins"),
            ])
            summary = json.loads((root / "out" / "summary.json").read_text())
            self.assertEqual(rc, 0)
            self.assertEqual(summary["result"], "ok")
            self.assertEqual(summary["mission_status"], "completed")
            np.testing.assert_allclose(summary["position"], [5.0, 1.0, 1.0], atol=1e-6)
            self.assertAlmostEqual(summary["yaw_deg"], 90.0)
            self.assertEqual(summary["reports"]["deactivate"]["status"], "inactive")
            self.assertEqual(summary["reports"]["uninstall"]["status"], "uninstalled")
            self.assertEqual(summary["reports"]["uninstall"]["residual"], [])
            preserved = summary["reports"]["artifacts_preserved"]
            self.assertTrue(any(path.endswith(".png") for path in preserved))
            self.assertTrue(any(path.endswith(".state.json") for path in preserved))
            self.assertTrue(all(Path(path).exists() for path in preserved))


class CpuWorldIntegrationTest(unittest.TestCase):
    def test_mission_on_real_cpu_world(self):
        with tempfile.TemporaryDirectory() as tmp:
            world = SplatWorldCPU(
                FIXTURE,
                TargetSpec(name="target", position_flu=(0.0, 0.0, 0.0), radius_m=0.2),
                start_flu=(0.0, 0.0, 1.0),
                start_yaw_deg=0.0,
                width=48,
                height=27,
                bound_radius=20.0,
            )
            executor = MotionExecutor(world, artifacts_dir=tmp)
            state = executor.run(MissionSpec.from_dict({
                "mission_id": "cpu-real",
                "steps": [
                    {"kind": "move", "direction": "forward", "meters": 5.0},
                    {"kind": "hover", "seconds": 3.0},
                    {"kind": "turn", "direction": "left", "degrees": 90},
                    {"kind": "move", "direction": "forward", "meters": 1.0},
                    {"kind": "photo", "label": "end"},
                ],
            }))
            self.assertTrue(Path(state.steps[4].artifact).exists())
        self.assertEqual(state.status, MISSION_COMPLETED)
        np.testing.assert_allclose(world.position_flu, [5.0, 1.0, 1.0], atol=1e-6)
        self.assertAlmostEqual(math.degrees(world.yaw_rad), 90.0)
        self.assertAlmostEqual(state.clock_s, 3.0)

    def test_real_world_bounds_block_instead_of_truncating(self):
        with tempfile.TemporaryDirectory() as tmp:
            world = SplatWorldCPU(
                FIXTURE,
                TargetSpec(name="target", position_flu=(0.0, 0.0, 0.0), radius_m=0.2),
                start_flu=(0.0, 0.0, 1.0),
                width=32,
                height=18,
                bound_radius=3.0,
            )
            executor = MotionExecutor(world, artifacts_dir=tmp)
            state = executor.run(MissionSpec.from_dict({
                "mission_id": "cpu-clamped",
                "steps": [{"kind": "move", "direction": "forward", "meters": 5.0}],
            }))
        self.assertEqual(state.status, MISSION_BLOCKED)
        self.assertAlmostEqual(state.steps[0].progress["remaining_m"], 2.0, places=3)
        self.assertIn("clamped", state.steps[0].error)


if __name__ == "__main__":
    unittest.main()
