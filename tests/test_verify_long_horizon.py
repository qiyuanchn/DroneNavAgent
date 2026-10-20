"""Tests for the long-horizon driver (``scripts/verify_long_horizon.py``, P4).

The driver answers three questions about a *chained* task ("find the landmark,
then the object beside it, then the target next to that object"):

  stages      a stage is complete only when the executor really carried the drone
              there (reach) *and* a tool result located the stage's place
              (locate) - a plan step, a finish reason or a recovery note cannot
              complete one. These tests pin that with synthetic ledgers, and pin
              that the drone's own pose is not a located reference.
  interruption  an interrupted leg hands over pose, plan, subgoal and a brief
              built *only* from its own traced places; the plan record is written
              through the task module's ``update_plan`` (so the resumed leg's
              executor can keep running it) and adopted through ``resume_plan``.
              The no-carry control legs carry nothing but the pose.
  decisions vs executor  decision turns, the runtime's own decision nodes,
              sub-task transitions, model calls, executor legs and control steps
              are reported together, so "the VLM is consulted at decision nodes,
              not per executor step" is a ratio and not a claim.

The end-to-end test runs the *real* tool runner on a tiny CPU 3DGS scene with
``--backend mock`` (no weights, no GPU): two seeds, one interrupted mid-task and
resumed, one run straight through - what the driver writes and what it measures.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = ROOT / "scripts" / "verify_long_horizon.py"
TOOL_RUNNER_PATH = ROOT / "scripts" / "run_vlm_nav_tools.py"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve string annotations via sys.modules
    spec.loader.exec_module(module)
    return module


DRIVER = _load("diffaero_script_verify_long_horizon", DRIVER_PATH)
Runner = _load("diffaero_script_run_vlm_nav_tools_for_long_horizon", TOOL_RUNNER_PATH)

from vlm_nav import tool_impl_task as task_module  # noqa: E402
from vlm_nav.tools import (  # noqa: E402
    STATUS_OK,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)

MANIFEST = {
    "object_positions_flu": [
        [-1.0, 2.0, 0.5],
        [-1.5, 3.0, 0.5],
        [-2.0, 4.0, 0.5],
    ],
    "source_objects": [
        {"name": "closed door wood", "key": "door_wood_closed"},
        {"name": "cabinet", "key": "cabinet"},
        {"name": "cardboard box", "key": "cardboard_box"},
    ],
}

TASK_SCENE = {
    "world": "cpu",
    "start_pos": [0.0, 0.0, 1.0],
    "target_pos": [-2.0, 4.0, 0.5],
    "target_name": "cardboard box",
    "target_radius": 0.4,
    "success_dist": 1.5,
}

TASK_STAGES = [
    {
        "id": "landmark",
        "kind": "landmark",
        "label": "走廊里的门",
        "object_key": "door_wood_closed",
        "reach_m": 1.5,
        "locate_m": 0.6,
        "step": "找到走廊里的门",
    },
    {
        "id": "object",
        "kind": "object",
        "label": "门旁边的柜子",
        "object_key": "cabinet",
        "anchor": "landmark",
        "within_m": 1.5,
        "reach_m": 1.5,
        "locate_m": 0.6,
        "step": "找到门旁边的柜子",
    },
    {
        "id": "target",
        "kind": "target",
        "label": "柜子旁边的目标",
        "target": True,
        "anchor": "object",
        "within_m": 1.5,
        "reach_m": 1.2,
        "locate_m": 0.6,
        "step": "确认柜子旁边的目标",
    },
]


def write_task(directory: Path, *, stages=None, scene=None, instruction=None) -> Path:
    payload = {
        "name": "unit_chain",
        "instruction": instruction or "先找到门，再找到柜子，最后确认目标",
        "scene": dict(scene or TASK_SCENE),
        "stages": list(stages if stages is not None else TASK_STAGES),
    }
    payload["scene"].setdefault("semantic_manifest", str(directory / "manifest.json"))
    path = directory / "task.json"
    path.write_text(json.dumps(payload))
    (directory / "manifest.json").write_text(json.dumps(MANIFEST))
    return path


def make_task(directory: Path, **kwargs) -> object:
    return DRIVER.load_task(write_task(directory, **kwargs))


def entry(
    seq: int,
    tool: str,
    *,
    data: dict | None = None,
    evidence: list | None = None,
    pose: list | None = None,
    yaw: float = 0.0,
    outer: bool = True,
    status: str = STATUS_OK,
    leg: str | None = None,
    global_seq: int | None = None,
) -> dict:
    row = {
        "seq": seq,
        "depth": 0 if outer else 1,
        "outer": outer,
        "tool": tool,
        "args": {},
        "status": status,
        "data": data or {},
        "evidence": evidence or [],
        "pose_before": list(pose or []),
        "pose_after": list(pose or []),
        "yaw_after": yaw,
        "executor_before": {},
        "executor_after": {},
        "ms": 1.0,
    }
    if leg:
        row["leg"] = leg
    if global_seq is not None:
        row["global_seq"] = global_seq
    return row


def ref(kind: str, name: str) -> dict:
    return {"kind": kind, "ref": name, "note": ""}


def make_leg(entries, *, label="seed7_leg1", summary=None, executor=None, interrupted=False,
             reason="", plan=None, prompt_id="ep_007", instruction="do the chain",
             carry=None, vlm_calls=1, log=None, out_dir=None, start_pose=(0.0, 0.0, 1.0),
             role="fresh", checkpoint=None, final_plan=None):
    leg_plan = DRIVER.LegPlan(
        label=label,
        instruction=instruction,
        prompt_episode_id=prompt_id,
        tool_budget=8,
        max_turns=8,
        carry=dict(carry or {}),
        role=role,
    )
    if plan is not None:
        leg_plan = plan
    return DRIVER.LegResult(
        plan=leg_plan,
        out_dir=Path(out_dir or tempfile.mkdtemp()),
        summary=summary,
        interrupted=interrupted,
        interrupt_reason=reason,
        error=None,
        entries=list(entries),
        executor=dict(executor or {"legs": 2, "ticks": 4, "renders": 3, "trajectory_growth": 2,
                                   "total_legs": 2}),
        vlm_calls=int(vlm_calls),
        wall_time_s=1.0,
        start_pose=list(start_pose),
        log=dict(log or {}),
        checkpoint=dict(checkpoint or {}),
        final_plan=final_plan,
    )


class _StubWorld:
    """Minimal world: an ``apply_action`` that advances the trajectory."""

    def __init__(self, ticks_per_leg: int = 3):
        self.state = types.SimpleNamespace(
            position_flu=np.zeros(3), yaw_rad=0.0
        )
        self.trajectory: list = [(0.0, 0.0, 0.0)]
        self.actions: list = []
        self.renders = 0
        self._ticks = int(ticks_per_leg)

    def apply_action(self, action):
        self.actions.append(action)
        for _ in range(self._ticks):
            self.trajectory.append(tuple(self.state.position_flu))
        return "moved"

    def render(self):
        self.renders += 1
        return "frame"

    def position_flu(self):
        return self.state.position_flu

    @property
    def yaw_rad(self):
        return self.state.yaw_rad


# ---------------------------------------------------------------------------
# the task file: stage places and chain relations
# ---------------------------------------------------------------------------


class TaskSpecTest(unittest.TestCase):
    def test_stage_places_resolve_from_manifest_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = make_task(Path(tmp))
        self.assertEqual(task.stages[0].position, (-1.0, 2.0, 0.5))
        self.assertEqual(task.stages[0].source, "manifest:door_wood_closed")
        self.assertEqual(task.stages[1].position, (-1.5, 3.0, 0.5))
        self.assertEqual(task.stages[1].source, "manifest:cabinet")

    def test_target_stage_resolves_from_scene_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = make_task(Path(tmp))
        self.assertTrue(task.stages[2].target)
        self.assertEqual(task.stages[2].position, (-2.0, 4.0, 0.5))
        self.assertEqual(task.stages[2].source, "scene.target_pos")
        # an explicit reach_m wins; without one the target stage defaults to the
        # episode's success gate (the same number the runner judges with)
        self.assertAlmostEqual(task.stages[2].reach_m, 1.2)
        stages = [dict(stage) for stage in TASK_STAGES]
        stages[2].pop("reach_m")
        with tempfile.TemporaryDirectory() as tmp:
            defaulted = make_task(Path(tmp), stages=stages)
        self.assertAlmostEqual(defaulted.stages[2].reach_m, 1.5)

    def test_target_stage_can_fall_back_to_manifest_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            stages = [dict(stage) for stage in TASK_STAGES]
            stages[2] = {
                "id": "target", "kind": "target", "label": "目标", "target": True,
                "locate_m": 0.6, "step": "确认目标",
            }
            scene = dict(TASK_SCENE)
            scene.pop("target_pos")
            scene["target_index"] = 2
            task = make_task(Path(tmp), stages=stages, scene=scene)
        self.assertEqual(task.stages[2].position, (-2.0, 4.0, 0.5))
        self.assertEqual(task.stages[2].source, "manifest_target:2")

    def test_explicit_position_wins_over_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            stages = [dict(stage) for stage in TASK_STAGES]
            stages[0]["position_flu"] = [9.0, 9.0, 9.0]
            task = make_task(Path(tmp), stages=stages)
        self.assertEqual(task.stages[0].position, (9.0, 9.0, 9.0))
        self.assertEqual(task.stages[0].source, "position_flu")

    def test_unknown_object_key_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            stages = [dict(stage) for stage in TASK_STAGES]
            stages[0]["object_key"] = "not_in_the_scene"
            with self.assertRaises(SystemExit):
                make_task(Path(tmp), stages=stages)

    def test_unknown_anchor_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            stages = [dict(stage) for stage in TASK_STAGES]
            stages[1]["anchor"] = "nowhere"
            with self.assertRaises(SystemExit):
                make_task(Path(tmp), stages=stages)

    def test_task_without_stages_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                make_task(Path(tmp), stages=[])

    def test_chain_check_reports_relation_distances(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = make_task(Path(tmp))
        check = DRIVER.chain_check(task)
        rows = {row["stage"]: row for row in check["rows"]}
        self.assertTrue(check["ok"], check["problems"])
        self.assertEqual(rows["object"]["status"], "ok")
        self.assertAlmostEqual(rows["object"]["distance_m"], math.dist(
            MANIFEST["object_positions_flu"][1], MANIFEST["object_positions_flu"][0]), places=3)

    def test_chain_check_flags_a_violated_relation(self):
        with tempfile.TemporaryDirectory() as tmp:
            stages = [dict(stage) for stage in TASK_STAGES]
            stages[2]["within_m"] = 0.2
            task = make_task(Path(tmp), stages=stages)
        check = DRIVER.chain_check(task)
        self.assertFalse(check["ok"])
        self.assertTrue(any("within_m" in problem for problem in check["problems"]))

    def test_chain_check_flags_unevaluable_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            stages = [dict(stage) for stage in TASK_STAGES]
            stages[1].pop("object_key")
            stages[1].pop("anchor")
            task = make_task(Path(tmp), stages=stages)
        check = DRIVER.chain_check(task)
        self.assertFalse(check["ok"])
        rows = {row["stage"]: row for row in check["rows"]}
        self.assertEqual(rows["object"]["status"], "unevaluable")


# ---------------------------------------------------------------------------
# interrupt specs and seeds
# ---------------------------------------------------------------------------


class InterruptSpecTest(unittest.TestCase):
    def test_parse_forms(self):
        self.assertEqual(DRIVER.parse_interrupt_spec("7:budget:6"), (7, "budget", 6.0))
        self.assertEqual(DRIVER.parse_interrupt_spec("11:deadline:45"), (11, "deadline", 45.0))
        self.assertEqual(DRIVER.parse_interrupt_spec("3:budget"), (3, "budget", None))
        self.assertEqual(DRIVER.parse_interrupt_spec(" 9 : DEADLINE : 12.5 "),
                         (9, "deadline", 12.5))

    def test_parse_errors(self):
        for value in ("7", "x:budget:1", "7:oops:1", "7:budget:0", "7:budget:-2"):
            with self.assertRaises(SystemExit, msg=value):
                DRIVER.parse_interrupt_spec(value)

    def test_seeds_and_interrupt_map(self):
        self.assertEqual(DRIVER.parse_seeds("7, 11 ,13"), [7, 11, 13])
        self.assertEqual(DRIVER.parse_seeds([7, 7, 9]), [7, 9])
        with self.assertRaises(SystemExit):
            DRIVER.parse_seeds("7,x")
        table = DRIVER.interrupt_map(["7:budget:6", "11:deadline:30"])
        self.assertEqual(table[7], ("budget", 6.0))
        self.assertEqual(table[11], ("deadline", 30.0))


# ---------------------------------------------------------------------------
# executor probe and call ledger
# ---------------------------------------------------------------------------


class ProbeAndLedgerTest(unittest.TestCase):
    def test_probe_counts_legs_ticks_and_renders(self):
        world = _StubWorld(ticks_per_leg=3)
        probe = DRIVER.ExecutorProbe(world)
        self.assertEqual(probe.installed, ["apply_action", "render"])
        world.apply_action(types.SimpleNamespace(kind="move"))
        world.apply_action(types.SimpleNamespace(kind="move"))
        world.render()
        snapshot = probe.snapshot()
        self.assertEqual(snapshot["legs"], 2)
        self.assertEqual(snapshot["renders"], 1)
        self.assertGreaterEqual(snapshot["points"], 6)

    def test_world_pose_reads_the_state(self):
        world = _StubWorld()
        world.state.position_flu = np.array([1.0, 2.0, 3.0])
        world.state.yaw_rad = 0.5
        position, yaw = DRIVER.world_pose(world)
        self.assertEqual(position, [1.0, 2.0, 3.0])
        self.assertAlmostEqual(yaw, 0.5)

    def _registry_with_nested_call(self):
        registry = ToolRegistry()

        def inner(ctx, args):
            return ToolResult(STATUS_OK, {"observation_id": "obs_1"}, [])

        def outer(ctx, args):
            nested = ctx.extra["wrapped"].call("inner", {}, ctx)
            return ToolResult(STATUS_OK, {"candidate_id": "cand_1",
                                          "position_flu": [1.0, 2.0, 3.0]},
                              [], )

        registry.register(ToolSpec(name="inner", description="", input_schema={}, handler=inner))
        registry.register(ToolSpec(name="outer", description="", input_schema={}, handler=outer))
        return registry

    def test_ledger_records_outer_and_nested_calls(self):
        world = _StubWorld()
        probe = DRIVER.ExecutorProbe(world)
        ledger = DRIVER.Ledger(world, probe)
        wrapped = ledger.wrap(self._registry_with_nested_call())
        ctx = ToolContext(world=world, memory=None, episode_id="ep_1",
                          budget={"tool_calls_left": 5}, extra={"wrapped": wrapped})
        result = wrapped.call("outer", {"a": 1}, ctx)
        self.assertEqual(result.data["candidate_id"], "cand_1")
        # the nested call closes first, so it lands in the ledger first: what
        # decides "decision call vs routine call" is ``depth``/``outer``, never
        # the position in the list
        self.assertEqual([row["tool"] for row in ledger.entries], ["inner", "outer"])
        self.assertEqual([row["depth"] for row in ledger.entries], [1, 0])
        self.assertEqual([row["outer"] for row in ledger.entries], [False, True])
        self.assertIn("position_flu", ledger.entries[1]["data"])

    def test_ledger_records_the_error_and_reraises(self):
        world = _StubWorld()
        ledger = DRIVER.Ledger(world, DRIVER.ExecutorProbe(world))

        class _BrokenRegistry:
            def names(self):
                return ["boom"]

            def get(self, name):
                return None

            def call(self, name, args, ctx):
                raise RuntimeError("kaboom")

        wrapped = ledger.wrap(_BrokenRegistry())
        ctx = ToolContext(world=world, memory=None, episode_id="ep_1",
                          budget={"tool_calls_left": 5}, extra={})
        with self.assertRaises(RuntimeError):
            wrapped.call("boom", {}, ctx)
        self.assertEqual(len(ledger.entries), 1)
        self.assertIn("RuntimeError", ledger.entries[0]["error"])

    def test_ledger_delegates_the_mock_result_index(self):
        world = _StubWorld()
        ledger = DRIVER.Ledger(world, DRIVER.ExecutorProbe(world))
        registry = ToolRegistry()
        registry.result_index = {"observation_id": "obs_9"}
        wrapped = ledger.wrap(registry)
        self.assertIs(wrapped.result_index, registry.result_index)


# ---------------------------------------------------------------------------
# places, stages and completion evidence
# ---------------------------------------------------------------------------


class StageReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.task = make_task(Path(self.tmp.name))

    def test_drone_pose_is_not_a_located_place(self):
        rows = DRIVER.positions_in({"pose": [1.0, 2.0, 3.0], "yaw_deg": 0.0})
        self.assertEqual(rows, [])
        rows = DRIVER.positions_in({"candidates": [{"position_flu": [1.0, 2.0, 3.0]}]})
        self.assertEqual(rows, [(1.0, 2.0, 3.0)])
        rows = DRIVER.positions_in({"target_position": [4.0, 5.0, 6.0]})
        self.assertEqual(rows, [(4.0, 5.0, 6.0)])

    def test_reached_without_a_located_reference_is_not_complete(self):
        place = self.task.stages[0].position
        entries = [entry(1, "go_to_viewpoint", data={},
                         pose=list(place), evidence=[ref("observation", "obs_1")])]
        report = DRIVER.stage_report(entries, self.task)
        row = report["stages"]["landmark"]
        self.assertTrue(row["reach"])
        self.assertFalse(row["located"])
        self.assertFalse(row["complete"])
        self.assertEqual(report["complete_count"], 0)

    def test_located_without_reaching_is_not_complete(self):
        place = self.task.stages[0].position
        entries = [entry(1, "ground_target",
                         data={"candidates": [{"position_flu": list(place)}]},
                         pose=[5.0, 5.0, 1.0], evidence=[ref("observation", "obs_1")])]
        report = DRIVER.stage_report(entries, self.task)
        row = report["stages"]["landmark"]
        self.assertTrue(row["located"])
        self.assertFalse(row["reach"])
        self.assertFalse(row["complete"])

    def test_complete_stage_needs_reach_and_locate(self):
        place = self.task.stages[0].position
        entries = [entry(1, "approach_candidate",
                         data={"candidates": [{"position_flu": list(place)}]},
                         pose=list(place), evidence=[ref("memory", "cand_1")])]
        report = DRIVER.stage_report(entries, self.task)
        row = report["stages"]["landmark"]
        self.assertTrue(row["complete"])
        self.assertEqual(row["evidence_count"], 1)
        self.assertEqual(row["locate_tool"], "approach_candidate")

    def test_stage_without_a_place_is_unevaluable(self):
        stages = [dict(stage) for stage in TASK_STAGES]
        stages[0].pop("object_key")
        task = make_task(Path(self.tmp.name), stages=stages)
        report = DRIVER.stage_report([entry(1, "observe")], task)
        row = report["stages"]["landmark"]
        self.assertTrue(row["unevaluable"])
        self.assertFalse(row["complete"])

    def test_order_uses_the_episode_wide_order_key(self):
        places = [stage.position for stage in self.task.stages]
        leg1 = [entry(5, "ground_target",
                      data={"candidates": [{"position_flu": list(places[0])}]},
                      pose=list(places[0]), evidence=[ref("observation", "obs_1")],
                      global_seq=1)]
        leg2 = [entry(1, "ground_target",
                      data={"candidates": [{"position_flu": list(places[1])}]},
                      pose=list(places[1]), evidence=[ref("observation", "obs_2")],
                      global_seq=2)]
        report = DRIVER.stage_report(leg1 + leg2, self.task)
        self.assertEqual(report["complete_ids"], ["landmark", "object"])
        self.assertTrue(report["order_ok"])
        # without the episode-wide key the second leg's own numbering (1) would
        # order the object before the landmark it depends on
        backwards = DRIVER.stage_report(
            [dict(row, **{}) for row in leg1] + [dict(row, **{}) for row in leg2],
            self.task,
        )
        self.assertTrue(backwards["order_ok"])
        without_key = DRIVER.stage_report(
            [{key: value for key, value in row.items() if key != "global_seq"}
             for row in (leg1 + leg2)],
            self.task,
        )
        self.assertFalse(without_key["order_ok"])

    def test_relation_uses_the_agents_own_places(self):
        a = np.asarray(self.task.stages[0].position, dtype=float)
        b = np.asarray(self.task.stages[1].position, dtype=float)
        axis = (b - a) / float(np.linalg.norm(b - a))
        # both places sit inside their own locate_m window (0.6 m), but on the
        # far side of it: the agent's chain is 2.3 m wide where the scene's is
        # 1.1 m, so the declared relation does not hold for *its* places
        placed_a = a - 0.59 * axis
        placed_b = b + 0.59 * axis
        entries = [
            entry(1, "ground_target",
                  data={"candidates": [{"position_flu": [float(v) for v in placed_a]}]},
                  pose=list(a), evidence=[ref("observation", "obs_1")]),
            entry(2, "ground_target",
                  data={"candidates": [{"position_flu": [float(v) for v in placed_b]}]},
                  pose=list(b), evidence=[ref("observation", "obs_2")]),
        ]
        report = DRIVER.stage_report(entries, self.task)
        self.assertTrue(report["stages"]["object"]["located"])
        self.assertFalse(report["stages"]["object"]["relation_ok"])
        self.assertGreater(report["stages"]["object"]["relation_distance_m"], 2.1)
        self.assertFalse(report["relation_ok"])

    def test_relation_holds_when_the_places_agree(self):
        places = [stage.position for stage in self.task.stages]
        entries = [
            entry(1, "ground_target",
                  data={"candidates": [{"position_flu": list(places[0])}]},
                  pose=list(places[0]), evidence=[ref("observation", "obs_1")]),
            entry(2, "ground_target",
                  data={"candidates": [{"position_flu": list(places[1])}]},
                  pose=list(places[1]), evidence=[ref("observation", "obs_2")]),
        ]
        report = DRIVER.stage_report(entries, self.task)
        self.assertTrue(report["stages"]["object"]["relation_ok"])
        self.assertTrue(report["relation_ok"])

    def test_completion_chain_is_the_runtime_judge(self):
        target = self.task.stages[2].position
        verified_entries = [
            entry(1, "ground_target",
                  data={"candidates": [{"position_flu": list(target),
                                        "id": "cand_1"}]},
                  evidence=[ref("observation", "obs_1")]),
            entry(2, "verify_goal", data={"verified": True, "distance_m": 0.4},
                  evidence=[ref("observation", "obs_2")]),
        ]
        verdict = DRIVER.completion_chain(verified_entries)
        self.assertTrue(verdict["supported"])
        self.assertTrue(verdict["verified"])
        self.assertEqual(verdict["kind"], "verification")

        unverified = [
            entry(1, "verify_goal", data={"verified": False}, evidence=[ref("observation", "obs_1")]),
        ]
        self.assertFalse(DRIVER.completion_chain(unverified)["verified"])

    def test_verify_evidence_reads_the_gate(self):
        target = self.task.stages[2].position
        inside = [entry(1, "approach_candidate",
                        data={"target_position": [target[0] + 0.2, target[1], target[2]]},
                        evidence=[ref("memory", "cand_1")])]
        outside = [entry(1, "approach_candidate",
                         data={"target_position": [target[0] + 5.0, target[1], target[2]]},
                         evidence=[ref("memory", "cand_1")])]
        self.assertTrue(DRIVER.verify_evidence(inside, self.task.stages[2], 1.5)["ok"])
        self.assertFalse(DRIVER.verify_evidence(outside, self.task.stages[2], 1.5)["ok"])


# ---------------------------------------------------------------------------
# resume: the plan record, the brief and the control legs
# ---------------------------------------------------------------------------


def complete_stage_entry(seq, place, ref_name, *, tool="ground_target", pose=None,
                         global_seq=None):
    row = entry(
        seq, tool,
        data={"candidates": [{"position_flu": list(place), "id": f"cand_{seq}"}]},
        pose=list(pose or place),
        evidence=[ref("observation", ref_name), ref("memory", f"cand_{seq}")],
        global_seq=global_seq,
    )
    return row


def checkpoint_plan(task, statuses, *, label="seed7_leg1",
                    instruction="do the chain", evidence=None):
    """The plan record an interrupted leg's checkpoint carries (agent state)."""
    subtasks = []
    for stage, status in zip(task.stages, statuses):
        subtasks.append(
            {
                "id": f"st_{stage.index}",
                "goal": stage.step,
                "status": status,
                "evidence": list((evidence or {}).get(stage.index) or []),
            }
        )
    done = [stage.index for stage, status in zip(task.stages, statuses) if status == "done"]
    current = min(
        [stage.index for stage, status in zip(task.stages, statuses) if status != "done"]
        or [len(task.stages)]
    )
    return {
        "plan_id": "plan_1",
        "revision": 1,
        "steps": [stage.step for stage in task.stages],
        "subtasks": subtasks,
        "current_step": current,
        "done": done,
        "episode_id": label,
        "instruction": instruction,
    }


def make_checkpoint(*, plan=None, pose=(1.0, 2.0, 3.0), yaw=0.5,
                    ledger=None, candidates=None, observations=None,
                    open_questions=None, source_leg="seed7_leg1"):
    return {
        "version": 3,
        "source": "runtime.ctx.extra",
        "source_leg": source_leg,
        "source_episode_id": "ep_007",
        "instruction": "do the chain",
        "pose": {"position_flu": [float(v) for v in pose], "yaw_rad": float(yaw)},
        "plan": plan,
        "confirmation_ledger": dict(ledger or {}),
        "candidates": dict(candidates or {}),
        "observations": dict(observations or {}),
        "viewpoints_by_id": {},
        "hypotheses": {},
        "task_scope": {"task_id": "plan_1", "subtask_id": "st_1",
                       "candidate_id": "", "evidence_revision": 0},
        "beliefs": {},
        "open_questions": open_questions,
        "evidence_pack": {},
        "last_observation_id": "",
    }


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.task = make_task(self.directory)

    def _partial_leg(self, *, only_first=True):
        statuses = ["done", "active", "pending"]
        if not only_first:
            statuses = ["done", "done", "pending"]
        plan = checkpoint_plan(
            self.task, statuses, evidence={1: ["obs_1"], 2: ["obs_2"]}
        )
        checkpoint = make_checkpoint(plan=plan)
        leg = make_leg(
            [], interrupted=True, reason="budget_exhausted", checkpoint=checkpoint
        )
        return leg

    def test_resume_state_is_the_checkpoint_verbatim(self):
        leg = self._partial_leg()
        state = DRIVER.build_resume_state(leg, self.task, prompt_episode_id="ep_007",
                                          world=_StubWorld())
        # the whole checkpoint is carried untouched, including the agent's own
        # plan statuses and evidence - nothing is recomputed from the ledger
        self.assertEqual(state["checkpoint"], leg.checkpoint)
        self.assertEqual(state["plan"], leg.checkpoint["plan"])
        self.assertEqual(state["done"], [1])
        plan = state["plan"]
        self.assertEqual(plan["current_step"], 2)
        self.assertEqual([item["status"] for item in plan["subtasks"]],
                         [task_module.SUBTASK_DONE, task_module.SUBTASK_ACTIVE,
                          task_module.SUBTASK_PENDING])
        self.assertIn("obs_1", plan["subtasks"][0]["evidence"])
        self.assertEqual(state["complete_count"], 1)

    def test_a_wrong_done_belief_survives_restart_and_is_scored_offline(self):
        """The checkpoint says st_1 is done; the scene says it never was.

        The resumed leg must inherit the wrong belief (that is what makes the
        restart a memory test), and only the offline report may call it wrong.
        """
        plan = checkpoint_plan(self.task, ["done", "pending", "pending"])
        checkpoint = make_checkpoint(plan=plan)
        leg = make_leg([], interrupted=True, reason="budget_exhausted",
                       checkpoint=checkpoint)

        state = DRIVER.build_resume_state(leg, self.task, prompt_episode_id="ep_007",
                                          world=_StubWorld())
        self.assertEqual(state["done"], [1])
        self.assertEqual(state["plan"]["subtasks"][0]["status"],
                         task_module.SUBTASK_DONE)
        self.assertIn("st_1", state["brief"])
        self.assertIn("done", state["brief"])

        # the offline channel is the only one that reads the scene: no entry
        # supports stage 1, so the agent's 'done' is a wrong belief
        report = DRIVER.checkpoint_report(self.task, [leg])
        self.assertTrue(report["available"])
        self.assertEqual(report["claimed_done"], ["st_1"])
        self.assertEqual(report["gt_done"], [])
        self.assertEqual(report["wrong_done"], ["st_1"])
        self.assertEqual(report["wrong_belief_count"], 1)
        self.assertEqual(report["resume_corrections"], 0)

    def test_a_resumed_leg_can_correct_the_checkpoint_belief_itself(self):
        plan = checkpoint_plan(self.task, ["done", "pending", "pending"])
        leg = make_leg([], interrupted=True, reason="budget_exhausted",
                       checkpoint=make_checkpoint(plan=plan))
        # the resumed leg's own plan now marks the stage blocked: it noticed
        corrected_plan = checkpoint_plan(self.task, ["blocked", "pending", "pending"])
        follow = make_leg(
            [], label="seed7_leg2_resume", role="resume",
            checkpoint=make_checkpoint(plan=corrected_plan, source_leg="seed7_leg2_resume"),
            final_plan=corrected_plan,
        )
        report = DRIVER.checkpoint_report(self.task, [leg, follow])
        self.assertEqual(report["wrong_done"], ["st_1"])
        self.assertEqual(report["resume_corrections"], 1)
        self.assertEqual(report["corrected"], ["st_1"])

    def test_the_resumed_leg_recheck_shows_up_as_a_checkpoint_correction(self):
        """The agent-side recheck overturns a wrong done; the offline score sees it.

        No ground truth enters the recheck: the first evidence turn of the
        resumed leg re-judges the checkpoint's confirmation (here: accepted
        without an independent position anchor) and demotes the stage. The
        offline checkpoint report then counts ``st_1`` as corrected because the
        follow leg's own final plan no longer says done.
        """
        plan = checkpoint_plan(self.task, ["done", "pending", "pending"])
        first = plan["subtasks"][0]
        first.update(
            status="done",
            _evidence_marks=["ref", "identity", "confirmed"],
            _evidence_confirmed={
                "candidate_id": "cand_1",
                "evidence_revision": 1,
                "observation_ids": ["obs_1"],
                "position_delta_m": None,
            },
        )
        checkpoint = make_checkpoint(plan=plan)
        leg = make_leg([], interrupted=True, reason="budget_exhausted", checkpoint=checkpoint)

        state = DRIVER.build_resume_state(
            leg, self.task, prompt_episode_id="ep_007", world=_StubWorld()
        )
        ctx = ToolContext(
            world=_StubWorld(), memory=None, episode_id="ep_007",
            budget={"tool_calls_left": 8},
            extra={
                "plan": state["resume_plan"],
                "instruction": state["instruction"],
                "confirmation_ledger": {"cand_1": {"last_verdict": "accepted"}},
                "perception.candidates": {
                    "cand_1": {"id": "cand_1", "status": "accepted"}
                },
            },
        )
        adopted = task_module.resume_plan(ctx, "ep_007", state["instruction"])
        self.assertIsNotNone(adopted)
        outcome = task_module.executor_step(
            ctx,
            {
                "tool": "observe",
                "args": {},
                "status": STATUS_OK,
                "data": {"observation_id": "obs_9"},
                "evidence": [ref("observation", "obs_9")],
                "turn": 1,
            },
        )
        self.assertEqual(
            outcome["resume_recheck"]["corrected"], ["st_1"], outcome
        )
        self.assertEqual(adopted["subtasks"][0]["status"], task_module.SUBTASK_CLAIMED_DONE)
        self.assertEqual(
            ctx.extra["confirmation_ledger"]["cand_1"]["last_verdict"], "rejected"
        )

        follow = make_leg(
            [], label="seed7_leg2_resume", role="resume",
            checkpoint=make_checkpoint(
                plan=adopted,
                ledger=ctx.extra["confirmation_ledger"],
                source_leg="seed7_leg2_resume",
            ),
            final_plan=adopted,
        )
        report = DRIVER.checkpoint_report(self.task, [leg, follow])
        self.assertEqual(report["wrong_done"], ["st_1"])
        self.assertIn("st_1", report["corrected"])
        self.assertGreaterEqual(report["resume_corrections"], 1)

    def test_wrong_confirmation_crosses_the_stage_scope(self):
        """An accepted identity is scored from the leg's whole candidate table.

        v10 seed23 accepted (again) a candidate that had been proposed under the
        *landmark* sub-task while the plan was executing the *object* stage.
        The stage-scoped confirmation cannot resolve that candidate's place, so
        its distance/correctness stayed ``None`` and ``wrong_confirmed`` missed
        it. The checkpoint score must fall back to the candidate table the whole
        leg built, exactly like the offline instance audit.
        """
        object_place = self.task.stages[1].position
        stale_place = (object_place[0], object_place[1] + 2.5, object_place[2])
        entries = [
            scoped_entry(
                1, "ground_target", subtask="st_1", candidate="cand_1",
                data={"candidates": [{"id": "cand_1", "position_flu": list(stale_place)}]},
                evidence=[ref("observation", "obs_1")],
                pose=list(self.task.stages[0].position),
            ),
            scoped_entry(
                2, "confirm_candidate", subtask="st_2", candidate="cand_1", revision=1,
                data={"verdict": "accepted"},
                evidence=[ref("memory", "cand_1")],
                pose=list(object_place),
            ),
        ]
        plan = checkpoint_plan(self.task, ["active", "active", "pending"])
        ledger = {"cand_1": {"last_verdict": "accepted", "accepted_revision": 1}}
        leg = make_leg(entries, interrupted=True, reason="budget_exhausted",
                       checkpoint=make_checkpoint(plan=plan, ledger=ledger))
        truth = DRIVER.stage_report(
            entries, self.task, subtask_map=DRIVER.stage_subtask_map(self.task, [leg])
        )
        object_row = truth["stages"]["object"]
        self.assertTrue(object_row["confirmed"])
        self.assertIsNone(object_row["confirmation_correct"])
        self.assertIsNone(object_row["confirmation_distance_m"])
        report = DRIVER.checkpoint_report(self.task, [leg])
        self.assertEqual(report["wrong_confirmed"], ["cand_1"])
        self.assertFalse(report["checkpoint_correct"])

    def test_a_claimed_done_without_the_scene_completion_is_a_wrong_belief(self):
        """``claimed_done`` is a completion claim even before it says ``done``."""
        plan = checkpoint_plan(self.task, ["claimed_done", "pending", "pending"])
        leg = make_leg([], interrupted=True, reason="budget_exhausted",
                       checkpoint=make_checkpoint(plan=plan))
        report = DRIVER.checkpoint_report(self.task, [leg])
        self.assertEqual(report["claimed_done"], [])
        self.assertEqual(report["wrong_claimed"], ["st_1"])
        self.assertEqual(report["wrong_done"], [])
        self.assertEqual(report["wrong_belief_count"], 1)
        self.assertFalse(report["checkpoint_correct"])

    def test_the_brief_reads_candidates_from_the_perception_table(self):
        """The runtime stores candidates under ``perception.candidates``."""
        plan = checkpoint_plan(self.task, ["active", "pending", "pending"])
        checkpoint = make_checkpoint(plan=plan)
        checkpoint["perception.candidates"] = {
            "cand_1": {"id": "cand_1", "position_flu": [-0.70, 0.388, -0.698]},
        }
        leg = make_leg([], interrupted=True, reason="budget_exhausted",
                       checkpoint=checkpoint)
        brief = DRIVER.resume_brief(leg, self.task)
        self.assertIn("cand_1 at (-0.70, 0.39, -0.70)", brief["text"])
        self.assertIn([-0.70, 0.388, -0.698], brief["places"])

    def test_resume_chain_never_calls_stage_report(self):
        """The resume path consumes the checkpoint, not the evaluator's verdict."""
        import ast
        import inspect
        import textwrap

        for function in (DRIVER.resume_brief, DRIVER.build_resume_state,
                         DRIVER._leg_checkpoint, DRIVER.plan_legs):
            tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
            called = {
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            } | {
                node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            }
            self.assertNotIn("stage_report", called)
        leg = self._partial_leg()
        original = DRIVER.stage_report

        def forbidden(*args, **kwargs):
            raise AssertionError("stage_report called on the resume path")

        DRIVER.stage_report = forbidden
        try:
            state = DRIVER.build_resume_state(
                leg, self.task, prompt_episode_id="ep_007", world=_StubWorld()
            )
            self.assertEqual(state["done"], [1])
            DRIVER.plan_legs(
                7, task=self.task, args=self._args(), prompt_episode_id="ep_007",
                interrupt=("budget", 3.0), base_budget=8, base_turns=8,
                resume_leg=leg, world=_StubWorld(), control=True,
            )
            DRIVER.plan_legs(
                7, task=self.task, args=self._args(), prompt_episode_id="ep_007",
                interrupt=("budget", 3.0), base_budget=8, base_turns=8,
                resume_leg=leg, world=_StubWorld(),
            )
        finally:
            DRIVER.stage_report = original

    def test_resume_leg_adopts_the_carried_plan_through_the_module(self):
        leg = self._partial_leg()
        state = DRIVER.build_resume_state(leg, self.task, prompt_episode_id="ep_007",
                                          world=_StubWorld())
        ctx = ToolContext(world=_StubWorld(), memory=None, episode_id="ep_007",
                          budget={"tool_calls_left": 8},
                          extra={"plan": state["resume_plan"],
                                 "instruction": state["instruction"]})
        adopted = task_module.resume_plan(ctx, "ep_007", state["instruction"])
        self.assertIsNotNone(adopted)
        self.assertEqual(adopted["episode_id"], "ep_007")
        self.assertTrue(adopted.get("resumed_from"))
        self.assertEqual(adopted.get("resume_count"), 1)
        self.assertEqual(task_module.plan_progress(ctx).split(":")[0].strip().split()[0],
                         adopted["plan_id"])
        self.assertIn("step 2/3", task_module.plan_progress(ctx))
        # the statuses/evidence stay the checkpoint's own, not a re-derivation
        self.assertEqual([item["status"] for item in adopted["subtasks"]],
                         [task_module.SUBTASK_DONE, task_module.SUBTASK_ACTIVE,
                          task_module.SUBTASK_PENDING])

    def test_resume_is_refused_for_another_instruction(self):
        leg = self._partial_leg()
        state = DRIVER.build_resume_state(leg, self.task, prompt_episode_id="ep_007",
                                          world=_StubWorld())
        ctx = ToolContext(world=_StubWorld(), memory=None, episode_id="ep_007",
                          budget={"tool_calls_left": 8}, extra={"plan": state["plan"]})
        self.assertIsNone(task_module.resume_plan(ctx, "ep_007", "a completely other task"))

    def test_subtask_specs_chain_on_the_previous_stage(self):
        leg = self._partial_leg()
        report = DRIVER.stage_report(leg.entries, self.task)
        specs = DRIVER.subtask_specs(self.task, report)
        self.assertEqual([spec["id"] for spec in specs], ["st_1", "st_2", "st_3"])
        self.assertEqual(specs[1]["depends_on"], ["st_1"])
        self.assertEqual(specs[2]["depends_on"], ["st_2"])
        self.assertTrue(all(spec["failure_branch"] for spec in specs))
        # every stage declares its reach tolerance and one concrete continuation;
        # neither carries the stage's true place
        for spec, stage in zip(specs, self.task.stages):
            self.assertEqual(spec["reach_m"], stage.reach_m)
            call = spec["next_step"]
            self.assertEqual(call["tool"], DRIVER.STAGE_NEXT_TOOL)
            self.assertTrue(call["args"].get("instruction"))
            self.assertNotIn("goal_position", spec)
            self.assertNotIn("position", call["args"])

    def test_fresh_leg_writes_a_structured_plan_without_ground_truth(self):
        plans = DRIVER.plan_legs(
            7,
            task=self.task,
            args=self._args(),
            prompt_episode_id="ep_007",
            interrupt=None,
            base_budget=8,
            base_turns=8,
            world=_StubWorld(),
        )
        first = plans[0]
        record = first.ctx_extra.get("plan")
        self.assertIsNotNone(record)
        self.assertEqual(record["episode_id"], first.label)
        subtasks = record["subtasks"]
        self.assertEqual(
            [item["status"] for item in subtasks],
            ["active", "pending", "pending"],
        )
        self.assertEqual([item["depends_on"] for item in subtasks],
                         [[], ["st_1"], ["st_2"]])
        self.assertTrue(all(item["done_when"] for item in subtasks))
        for item in subtasks:
            self.assertFalse(item.get("goal_position"))
            call = item["next_step"]
            self.assertEqual(call["tool"], DRIVER.STAGE_NEXT_TOOL)
            self.assertTrue(call["args"].get("instruction"))
        text = json.dumps(record, default=str)
        for stage in self.task.stages:
            self.assertNotIn(json.dumps(list(stage.position)), text)

    def test_brief_names_only_places_in_the_checkpoint(self):
        truth = self.task.stages[0].position
        self.assertIsNotNone(truth)
        offset = (truth[0] + 0.4, truth[1], truth[2])
        plan = checkpoint_plan(self.task, ["done", "active", "pending"])
        checkpoint = make_checkpoint(
            plan=plan,
            candidates={"cand_1": {"position_flu": list(offset), "status": "confirmed"}},
            observations={"obs_1": {"position_flu": list(offset)}},
        )
        leg = make_leg([], interrupted=True, reason="budget_exhausted",
                       checkpoint=checkpoint)
        state = DRIVER.build_resume_state(leg, self.task, prompt_episode_id="ep_007",
                                          world=_StubWorld())
        brief = state["brief"]
        self.assertIn("SESSION RESUME", brief)
        self.assertIn("(-0.60, 2.00, 0.50)", brief)
        self.assertNotIn("(-1.00, 2.00, 0.50)", brief)  # the ground truth never leaks
        # every place the brief names is one the checkpoint itself carries
        traced = DRIVER.positions_in(checkpoint)
        for place in state["brief_places"]:
            self.assertTrue(
                any(DRIVER._distance(place, candidate) < 1e-9 for candidate in traced),
                place,
            )

    def test_brief_lists_the_unfinished_stages(self):
        leg = self._partial_leg()
        state = DRIVER.build_resume_state(leg, self.task, prompt_episode_id="ep_007",
                                          world=_StubWorld())
        self.assertIn("NOT done", state["brief"])
        self.assertIn("stage 2", state["brief"])

    def test_resumed_instruction_is_the_base_plus_the_brief(self):
        leg = self._partial_leg()
        state = DRIVER.build_resume_state(leg, self.task, prompt_episode_id="ep_007",
                                          world=_StubWorld())
        self.assertTrue(state["instruction"].startswith(leg.plan.instruction))
        self.assertIn("SESSION RESUME", state["instruction"])
        # the verbatim record keeps the interrupted leg's own instruction ...
        self.assertEqual(state["plan"]["instruction"], leg.checkpoint["plan"]["instruction"])
        # ... while the run copy carries the module's own truncation of the
        # resumed instruction, the string resume_plan compares against
        limit = getattr(task_module, "MAX_INSTRUCTION_CHARS", 160)
        self.assertEqual(state["resume_plan"]["instruction"], state["instruction"][:limit])

    def _args(self, *extra):
        return DRIVER.parse_args([
            "--task-file", str(self.directory / "task.json"),
            "--out", str(self.directory / "out"), *extra,
        ])

    def test_plan_legs_carry_and_control_variants(self):
        leg = self._partial_leg()
        args = self._args()
        common = dict(task=self.task, args=args, prompt_episode_id="ep_007",
                      interrupt=None, base_budget=8, base_turns=8, resume_leg=leg,
                      world=_StubWorld())
        resumed = DRIVER.plan_legs(7, **common)[1]
        self.assertEqual(resumed.label, "seed7_leg2_resume")
        self.assertEqual(resumed.carry["mode"], "resume")
        self.assertEqual(resumed.carry["done"], [1])
        self.assertEqual(resumed.carry["plan_subtasks"],
                         [{"id": "st_1", "status": "done"},
                          {"id": "st_2", "status": "active"},
                          {"id": "st_3", "status": "pending"}])
        self.assertIsNotNone(resumed.teleport)
        self.assertIn("plan", resumed.ctx_extra)
        self.assertIn("subgoal", resumed.ctx_extra)

        control = DRIVER.plan_legs(7, control=True, **common)[1]
        self.assertEqual(control.carry["mode"], "control")
        self.assertEqual(control.ctx_extra, {})
        self.assertEqual(control.carry["done"], [])
        self.assertIsNotNone(control.teleport)
        self.assertNotIn("SESSION RESUME", control.instruction)

        restart = DRIVER.plan_legs(7, restart_control=True, **common)[1]
        self.assertEqual(restart.carry["mode"], "restart")
        # the restart arm goes back to the fresh leg's own start pose, not to
        # wherever the interrupted leg stopped
        self.assertEqual(restart.teleport[0], [0.0, 0.0, 1.0])
        self.assertEqual(restart.ctx_extra, {})

    def test_restart_arm_returns_to_the_fresh_leg_start_pose(self):
        leg = self._partial_leg()
        args = self._args()
        common = dict(task=self.task, args=args, prompt_episode_id="ep_007",
                      interrupt=None, base_budget=8, base_turns=8, resume_leg=leg,
                      world=_StubWorld())
        control = DRIVER.plan_legs(7, control=True, **common)[1]
        restart = DRIVER.plan_legs(7, restart_control=True, **common)[1]
        # control keeps the pose the interrupted leg stopped at
        self.assertEqual(control.teleport[0], [1.0, 2.0, 3.0])
        # restart goes back to the fresh leg's own start pose (its recorded one)
        self.assertEqual(restart.teleport[0], [0.0, 0.0, 1.0])
    def test_interrupt_budget_shrinks_the_first_leg(self):
        args = self._args()
        plans = DRIVER.plan_legs(7, task=self.task, args=args, prompt_episode_id="ep_007",
                                 interrupt=("budget", 3.0), base_budget=8, base_turns=8)
        self.assertEqual(plans[0].tool_budget, 3)
        self.assertEqual(plans[0].max_turns, 3)
        self.assertEqual(plans[0].interrupt_mode, "budget")
        plans = DRIVER.plan_legs(7, task=self.task, args=args, prompt_episode_id="ep_007",
                                 interrupt=("deadline", 45.0), base_budget=8, base_turns=8)
        self.assertEqual(plans[0].deadline_s, 45.0)
        self.assertEqual(plans[0].tool_budget, 8)


# ---------------------------------------------------------------------------
# metrics: decisions vs executor, repeats, recovery
# ---------------------------------------------------------------------------


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.task = make_task(self.directory)

    def test_repeat_search_counts_cached_and_evidence_free_checks(self):
        entries = [
            complete_stage_entry(1, self.task.stages[0].position, "obs_1", tool="observe"),
            entry(2, "observe", data={}, pose=[0.0, 0.0, 1.0], status="cached",
                  evidence=[]),
            entry(3, "observe", data={}, pose=[0.0, 0.0, 1.0], evidence=[]),
            entry(4, "observe", data={}, pose=[3.0, 3.0, 1.0],
                  evidence=[ref("observation", "obs_9")]),
        ]
        leg = make_leg(entries, summary={"turns": 4})
        metrics = DRIVER.leg_metrics(leg)
        self.assertEqual(metrics["info_calls"], 4)
        self.assertEqual(metrics["repeat_checks"], 2)  # the cached one, the empty one
        self.assertEqual(metrics["repeat_views"], 1)  # calls 2 and 3 from one view
        self.assertAlmostEqual(metrics["repeat_search_ratio"], 0.5)

    def test_decision_and_executor_anatomy(self):
        entries = [complete_stage_entry(1, self.task.stages[0].position, "obs_1")]
        leg = make_leg(
            entries,
            summary={"turns": 3},
            executor={"legs": 6, "ticks": 30, "renders": 4, "trajectory_growth": 6,
                      "total_legs": 6},
            vlm_calls=3,
            log={"turns_count": 3,
                 "decision_nodes": [{"turn": 1, "node": "new_candidate", "subtask": "st_1"}],
                 "plan_transitions": [
                     {"turn": 1, "id": "st_1", "from": "active", "to": "done", "reason": "done"}],
                 "plan_resumed": {}},
        )
        metrics = DRIVER.leg_metrics(leg)
        self.assertEqual(metrics["decision_turns"], 3)
        self.assertEqual(metrics["decision_nodes"], 1)
        self.assertEqual(metrics["decision_node_kinds"], {"new_candidate": 1})
        self.assertEqual(metrics["routine_turns"], 2)
        self.assertEqual(metrics["subtask_switches"], 1)
        self.assertEqual(metrics["n_plan_transitions"], 1)
        self.assertEqual(metrics["executor_legs_per_decision"], 2.0)
        self.assertEqual(metrics["ticks_per_model_call"], 10.0)

    def test_leg_metrics_reports_executor_auto_steps(self):
        # P4-B: an auto step is an executed call that costs no model turn; the
        # summary carries the runtime's count, the log is the fallback.
        entries = [complete_stage_entry(1, self.task.stages[0].position, "obs_1")]
        leg = make_leg(
            entries,
            summary={"turns": 2, "model_turns": 2,
                     "executor": {"auto_steps": 3, "saved_vlm_turns": 3, "limit": 3}},
            vlm_calls=2,
            log={"executor_auto_steps": [
                {"turn": 1, "tool": "observe", "origin": "plan_step"},
                {"turn": 1, "tool": "ground_target", "origin": "next_action"},
                {"turn": 2, "tool": "observe", "origin": "failure_branch"},
            ]},
        )
        metrics = DRIVER.leg_metrics(leg)
        self.assertEqual(metrics["executor_auto_steps"], 3)
        self.assertEqual(metrics["vlm_turns_saved"], 3)
        self.assertEqual(metrics["model_turns"], 2)

        # an old artifact has neither field: the log count is used, and a
        # missing summary block counts zero saved turns rather than inventing one
        old = make_leg(entries, summary={"turns": 4}, vlm_calls=4,
                       log={"executor_auto_steps": [{"turn": 1, "tool": "observe"}]})
        metrics = DRIVER.leg_metrics(old)
        self.assertEqual(metrics["executor_auto_steps"], 1)
        self.assertEqual(metrics["vlm_turns_saved"], 1)
        self.assertEqual(metrics["model_turns"], 4)

    def test_episode_metrics_counts_recovery_on_resume(self):
        places = [stage.position for stage in self.task.stages]
        leg1 = make_leg([complete_stage_entry(1, places[0], "obs_1")],
                        interrupted=True, reason="budget_exhausted",
                        carry={"mode": "resume", "done": [1]})
        leg2_entries = [
            complete_stage_entry(2, places[1], "obs_2", global_seq=2),
            complete_stage_entry(3, places[2], "obs_3", global_seq=3),
        ]
        leg2_entries.append(entry(4, "verify_goal", data={"verified": True, "distance_m": 0.3},
                                  pose=list(places[2]), evidence=[ref("observation", "obs_4")],
                                  global_seq=4))
        leg2 = make_leg(leg2_entries, label="seed7_leg2_resume", role="resume",
                        summary={"success": True, "finish_reason": "finish", "turns": 2},
                        carry={"mode": "resume", "done": [1]})
        metrics = DRIVER.episode_metrics(self.task, [leg1, leg2], gate_m=2.0)
        self.assertTrue(metrics["interrupted"])
        self.assertEqual(metrics["interrupt_at_stages"], 1)
        self.assertEqual(metrics["stages_gained_on_resume"], 2)
        self.assertTrue(metrics["interrupt_recovered"])
        self.assertTrue(metrics["resume_used_carry"])
        self.assertEqual(metrics["stages"]["complete_count"], 3)
        self.assertTrue(metrics["success"])
        self.assertTrue(metrics["verify"]["ok"])
        self.assertTrue(metrics["completion_chain"]["verified"])
        self.assertTrue(metrics["traceable_success"])
        self.assertEqual(metrics["done_stage_recheck"], 0)

    def test_traceable_success_requires_the_whole_chain(self):
        places = [stage.position for stage in self.task.stages]
        entries = [complete_stage_entry(i + 1, place, f"obs_{i + 1}")
                   for i, place in enumerate(places)]
        verified = entries + [entry(9, "verify_goal", data={"verified": True},
                                    pose=list(places[2]),
                                    evidence=[ref("observation", "obs_9")])]
        leg = make_leg(verified, summary={"success": True, "finish_reason": "finish"})
        metrics = DRIVER.episode_metrics(self.task, [leg], gate_m=2.0)
        self.assertTrue(metrics["stages"]["chain_complete"])
        self.assertTrue(metrics["traceable_success"])

        # without the verification, the same chain is *not* traceable
        unverified = make_leg(entries, summary={"success": True, "finish_reason": "finish"})
        metrics = DRIVER.episode_metrics(self.task, [unverified], gate_m=2.0)
        self.assertFalse(metrics["completion_chain"]["verified"])
        self.assertFalse(metrics["traceable_success"])

        # a stage without any evidence reference breaks the chain too
        stripped = [{**row, "evidence": []} for row in verified]
        metrics = DRIVER.episode_metrics(
            self.task, [make_leg(stripped, summary={"success": True,
                                                    "finish_reason": "finish"})],
            gate_m=2.0,
        )
        self.assertFalse(metrics["evidence_complete"])
        self.assertFalse(metrics["traceable_success"])

        # reaching the gate without a terminating success is not traceable either
        metrics = DRIVER.episode_metrics(
            self.task, [make_leg(verified, summary={"success": False,
                                                    "finish_reason": "max_turns"})],
            gate_m=2.0,
        )
        self.assertFalse(metrics["traceable_success"])

    def test_annotations_say_why_a_stage_switched(self):
        places = [stage.position for stage in self.task.stages]
        entries = [
            complete_stage_entry(1, places[0], "obs_1"),
            entry(2, "approach_candidate", data={"outcome": "stalled"}, status="failed",
                  pose=[0.0, 0.0, 1.0], evidence=[ref("observation", "obs_2")]),
        ]
        leg = make_leg(entries, summary={"success": False, "finish_reason": "max_turns"})
        metrics = DRIVER.episode_metrics(self.task, [leg], gate_m=2.0)
        text = " | ".join(metrics["annotations"])
        self.assertIn("stage 1", text)
        self.assertIn("failure branch walked", text)
        self.assertIn("approach_candidate:failed", text)

    def test_annotation_uses_the_real_traceable_verdict(self):
        # all three stages complete, but the object was completed before the
        # landmark: the chain holds, the order does not, so the episode is not a
        # traceable success - and the annotation has to say exactly that
        places = [stage.position for stage in self.task.stages]
        entries = [
            complete_stage_entry(1, places[1], "obs_1"),
            complete_stage_entry(2, places[0], "obs_2", global_seq=2),
            complete_stage_entry(3, places[2], "obs_3", global_seq=3),
            entry(4, "verify_goal", data={"verified": True},
                  pose=list(places[2]), evidence=[ref("observation", "obs_4")],
                  global_seq=4),
        ]
        leg = make_leg(entries, summary={"success": True, "finish_reason": "finish"})
        verdict = DRIVER.combined_verdict(self.task, [leg], gate_m=2.0)
        self.assertTrue(verdict["stages"]["chain_complete"])
        self.assertFalse(verdict["stages"]["order_ok"])
        self.assertFalse(verdict["traceable_success"])
        self.assertIn("traceable success: False", " | ".join(verdict["annotations"]))

    def test_aggregate_rates_and_checks(self):
        places = [stage.position for stage in self.task.stages]
        good_entries = [complete_stage_entry(i + 1, place, f"obs_{i + 1}")
                        for i, place in enumerate(places)]
        good_entries.append(entry(9, "verify_goal", data={"verified": True},
                                  pose=list(places[2]),
                                  evidence=[ref("observation", "obs_9")]))
        good = make_leg(good_entries, summary={"success": True, "finish_reason": "finish"})
        leg1 = make_leg([complete_stage_entry(1, places[0], "obs_1", global_seq=1)],
                        interrupted=True, reason="deadline reached")
        leg2 = make_leg([complete_stage_entry(2, places[1], "obs_2", global_seq=2)],
                        label="seed7_leg2_control", role="control",
                        summary={"success": False, "finish_reason": "max_turns"},
                        carry={"mode": "control", "done": []})
        episodes = [
            DRIVER.episode_metrics(self.task, [good], gate_m=2.0),
            DRIVER.episode_metrics(self.task, [leg1, leg2], gate_m=2.0),
        ]
        agg = DRIVER.aggregate(episodes, self.task, {"x": 1})
        self.assertEqual(agg["n_episodes"], 2)
        self.assertEqual(agg["traceable_success_rate"], 0.5)
        self.assertEqual(agg["interrupt"]["n_episodes"], 1)
        # the control leg gains a stage too (the scripted mock replays its play):
        # what the paired arms measure is the carry, not whether *any* leg runs
        self.assertEqual(agg["interrupt"]["recovered"], 1)
        self.assertEqual(agg["interrupt"]["recovery_rate"], 1.0)
        self.assertEqual(agg["stage_complete_rate"], 0.833)
        checks = DRIVER.protocol_checks(
            episodes, self.task,
            ledger_vs_trace=[{"label": "seed7_leg1", "ledger_outer": 3,
                              "trace_calls": 3, "trace_turns": 4, "summary_turns": 4}],
        )
        self.assertTrue(checks["ledger_vs_trace"]["all_match"])
        self.assertTrue(checks["progress_kinds_known"]["all_known"])

    def test_carry_plan_steps_check(self):
        leg = make_leg([complete_stage_entry(1, self.task.stages[0].position, "obs_1")],
                       interrupted=True, reason="budget_exhausted")
        state = DRIVER.build_resume_state(leg, self.task, prompt_episode_id="ep_007",
                                          world=_StubWorld())
        resumed_plan = DRIVER.LegPlan(
            label="seed7_leg2_resume", instruction=state["instruction"],
            prompt_episode_id="ep_007", tool_budget=8, max_turns=8,
            carry={"mode": "resume", "plan_steps": [stage.step for stage in self.task.stages],
                   "brief_places": state["brief_places"]},
        )
        episode = {"label": "ep_007", "legs": [
            {"label": "seed7_leg2_resume", "carry": resumed_plan.carry, "reported_places": []},
        ]}
        checks = DRIVER.protocol_checks([episode], self.task, ledger_vs_trace=[])
        self.assertTrue(checks["carry_plan_steps"]["match_task"])

    def test_report_markdown_renders_the_tables(self):
        places = [stage.position for stage in self.task.stages]
        entries = [complete_stage_entry(i + 1, place, f"obs_{i + 1}")
                   for i, place in enumerate(places)]
        leg = make_leg(entries, summary={"success": True, "finish_reason": "finish"})
        episodes = [DRIVER.episode_metrics(self.task, [leg], gate_m=2.0)]
        report = {
            "meta": {"task_name": self.task.name, "task_file": str(self.task.path),
                     "instruction": self.task.instruction, "n_episodes": 1,
                     "backend": "mock", "world": "cpu",
                     "chain_check": DRIVER.chain_check(self.task)},
            "task": self.task.to_dict(),
            "aggregate": DRIVER.aggregate(episodes, self.task, {}),
            "checks": {"ledger_vs_trace": {"all_match": True}},
            "episodes": [{**episodes[0], "label": "ep_007"}],
        }
        text = DRIVER.report_markdown(report)
        self.assertIn("Long-horizon verification", text)
        self.assertIn("## stages (per episode share)", text)
        self.assertIn("## decision nodes vs executor", text)
        self.assertIn("causal annotations", text)


# ---------------------------------------------------------------------------
# end to end: the real runner on a tiny CPU scene with the mock backend
# ---------------------------------------------------------------------------


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _make_scene(path: Path, target=(2.0, 0.5, 0.0)):
    """Tiny FLU scene; returns the FLU origin the world frame is shifted by.

    ``CpuGaussianScene`` reads PLY vertices as absolute 3DGS coordinates and
    puts the FLU origin at their centroid, so geometry authored in FLU has to
    be rotated into the 3DGS frame and the world poses shifted by that centroid
    (otherwise a rendered target and a FLU target are two different points).
    """
    points: list[np.ndarray] = []
    rng = np.random.default_rng(0)
    for _ in range(120):
        point = rng.uniform(-6, 6, size=3)
        point[2] = -1.5
        points.append(point)
    for offset in ((0.0, 0.0, 0.0), (0.15, 0.1, 0.0), (0.0, 0.2, 0.1)):
        points.append(np.asarray(target, dtype=np.float64) + np.asarray(offset))
    origin = np.mean(points, axis=0)
    header = "\n".join([
        "ply", "format binary_little_endian 1.0",
        f"element vertex {len(points)}",
        "property float x", "property float y", "property float z",
        "property float f_dc_0", "property float f_dc_1", "property float f_dc_2",
        "property float opacity", "property float scale_0", "property float scale_1",
        "property float scale_2", "property float rot_0", "property float rot_1",
        "property float rot_2", "property float rot_3", "end_header",
    ])
    rows = []
    for index, point in enumerate(points):
        local = np.asarray(point, dtype=np.float64) - origin
        # FLU -> the 3DGS frame CpuGaussianScene renders in
        x, y, z = float(-local[1]), float(local[2]), float(-local[0])
        if index < 120:
            sh, opacity = (0.0, 0.0, 0.0), _logit(0.8)
            scale = math.log(0.2)
        else:
            sh, opacity = (2.0, -1.0, -1.0), _logit(0.9)
            scale = math.log(0.25)
        rows.append([x, y, z, *sh, opacity, scale, scale, scale,
                     1.0, 0.0, 0.0, 0.0])
    with open(path, "wb") as handle:
        handle.write((header + "\n").encode("ascii"))
        for row in rows:
            handle.write(struct.pack("<14f", *row))
    return origin


TARGET = (2.0, 0.5, 0.0)
E2E_INSTRUCTION = "先找到走廊，再找到门旁边的柜子，最后确认柜子旁边的目标并飞过去"
E2E_STAGES = [
    {"id": "landmark", "kind": "landmark", "label": "走廊里的门",
     "position_flu": [2.0, 0.5, 0.0], "reach_m": 1.8, "locate_m": 2.0,
     "step": "找到走廊里的门"},
    {"id": "object", "kind": "object", "label": "门旁边的柜子",
     "position_flu": [2.0, 0.5, 0.0], "anchor": "landmark", "within_m": 1.0,
     "reach_m": 1.8, "locate_m": 2.0, "step": "找到门旁边的柜子"},
    {"id": "target", "kind": "target", "label": "柜子旁边的目标", "target": True,
     "position_flu": [2.0, 0.5, 0.0], "anchor": "object", "within_m": 0.5,
     "reach_m": 1.5, "locate_m": 2.0, "step": "确认柜子旁边的目标"},
]
#: One ground+approach cycle per stage: the v1 scope contract judges each stage
#: on the calls made while that stage's sub-task was executing, so the scripted
#: model has to establish the chain stage by stage (all three stage places are
#: the same point in this tiny scene).
E2E_MOCK_SCRIPT = [
    {"tool": "observe", "args": {"mode": "forward", "reason": "look ahead"}},
    {"tool": "ground_target", "args": {"observation_id": "$observation_id",
                                       "instruction": "$target_name",
                                       "reason": "locate the landmark"}},
    {"tool": "approach_candidate", "args": {"candidate_id": "$candidate_id",
                                            "standoff_m": 1.0}},
    # version 2: every movement stage needs the accepted identity, not only the
    # terminal one (a reached candidate is not yet a confirmed instance)
    {"tool": "confirm_candidate", "args": {"candidate_id": "$candidate_id"}},
    {"tool": "observe", "args": {"mode": "forward", "reason": "look at stage 2"}},
    {"tool": "ground_target", "args": {"observation_id": "$observation_id",
                                       "instruction": "$target_name",
                                       "reason": "locate the object of stage 2"}},
    {"tool": "approach_candidate", "args": {"candidate_id": "$candidate_id",
                                            "standoff_m": 1.5}},
    {"tool": "confirm_candidate", "args": {"candidate_id": "$candidate_id"}},
    {"tool": "observe", "args": {"mode": "forward", "reason": "look at stage 3"}},
    {"tool": "ground_target", "args": {"observation_id": "$observation_id",
                                       "instruction": "$target_name",
                                       "reason": "locate the target of stage 3"}},
    {"tool": "approach_candidate", "args": {"candidate_id": "$candidate_id",
                                            "standoff_m": 1.0}},
    {"tool": "confirm_candidate", "args": {"candidate_id": "$candidate_id"}},
    {"tool": "verify_goal", "args": {"reason": "the reached instance is verified"}},
    {"tool": "finish", "args": {"reason": "target reached and confirmed", "success": True}},
]


class EndToEndTest(unittest.TestCase):
    def _run(self, out_dir: Path, scene: Path, task_file: Path, script: Path, *extra: str):
        cmd = [
            sys.executable, "-m", "scripts.verify_long_horizon",
            "--task-file", str(task_file),
            "--out", str(out_dir),
            "--world", "cpu",
            "--ply", str(scene),
            "--backend", "mock",
            "--mock-script", str(script),
            "--tool-budget", "20",
            "--max-turns", "20",
            *extra,
        ]
        env = dict(os.environ)
        env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
        return subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True,
                              text=True, timeout=1800)

    def _fixtures(self, directory: Path):
        scene = directory / "scene.ply"
        origin = _make_scene(scene)
        # the world's FLU frame is the authored one shifted by the scene origin
        def shift(point):
            return [float(v) for v in (np.asarray(point, dtype=np.float64) - origin)]

        stages = [
            {**stage, "position_flu": shift(stage["position_flu"])}
            for stage in E2E_STAGES
        ]
        task_file = directory / "task.json"
        task_file.write_text(json.dumps({
            "name": "e2e_chain",
            "instruction": E2E_INSTRUCTION,
            "scene": {
                "world": "cpu",
                "target_name": "target",
                "target_pos": shift(TARGET),
                "target_radius": 0.5,
                "start_pos": shift([-3.0, 0.0, 1.0]),
                "success_dist": 1.5,
            },
            "stages": stages,
        }))
        script = directory / "mock_script.json"
        script.write_text(json.dumps(E2E_MOCK_SCRIPT))
        return scene, task_file, script

    def test_report_and_stage_verdicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            scene, task_file, script = self._fixtures(directory)
            out_dir = directory / "out"
            proc = self._run(out_dir, scene, task_file, script, "--seeds", "7")
            diagnostics = proc.stdout[-4000:] + "\n--- stderr ---\n" + proc.stderr[-3000:]
            self.assertEqual(proc.returncode, 0, diagnostics)
            report = json.loads((out_dir / DRIVER.REPORT_JSON).read_text())
            self.assertEqual(report["meta"]["n_episodes"], 1)
            self.assertEqual(report["meta"]["chain_check"]["ok"], True)
            episode = report["episodes"][0]
            self.assertFalse(episode["interrupted"])
            self.assertEqual(episode["n_legs"], 1)
            self.assertEqual(episode["stages"]["complete_count"], 3, diagnostics)
            self.assertTrue(episode["success"], diagnostics)
            self.assertTrue(episode["traceable_success"], diagnostics)
            self.assertTrue(episode["completion_chain"]["verified"])
            self.assertTrue(episode["evidence_complete"])
            legs = episode["legs"]
            self.assertGreater(legs[0]["executor"]["legs"], 0)
            self.assertGreater(legs[0]["model_calls"], 0)
            self.assertTrue((out_dir / DRIVER.REPORT_MD).exists())
            self.assertTrue((out_dir / "task.json").exists())
            leg_dir = out_dir / "seed7" / "seed7_leg1"
            self.assertTrue((leg_dir / DRIVER.EPISODES_JSONL).name)
            self.assertTrue((leg_dir / "ledger.jsonl").exists())
            self.assertTrue((leg_dir / "summary.json").exists())
            self.assertTrue((leg_dir / "tool_trace.jsonl").exists())
            self.assertTrue((leg_dir / "run_config.json").exists())
            ledger = [json.loads(line) for line in
                      (leg_dir / "ledger.jsonl").read_text().splitlines() if line.strip()]
            self.assertTrue(ledger)
            self.assertTrue(all("executor_before" in row for row in ledger))
            self.assertTrue(any(row["outer"] for row in ledger))
            checks = report["checks"]
            self.assertTrue(checks["ledger_vs_trace"]["all_match"], checks["ledger_vs_trace"])
            aggregate = report["aggregate"]
            self.assertEqual(aggregate["decision_vs_executor"]["executor_legs"], 1)
            self.assertIsNotNone(aggregate["cost"]["wall_time_s"])

    def test_interrupt_resume_and_control_legs(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            scene, task_file, script = self._fixtures(directory)
            out_dir = directory / "out"
            proc = self._run(out_dir, scene, task_file, script, "--seeds", "7,11",
                             "--interrupt", "7:budget:2", "--interrupt", "11:deadline:0.01",
                             "--restart-control")
            diagnostics = proc.stdout[-4000:] + "\n--- stderr ---\n" + proc.stderr[-3000:]
            self.assertEqual(proc.returncode, 0, diagnostics)
            report = json.loads((out_dir / DRIVER.REPORT_JSON).read_text())
            by_seed = {row["seed"]: row for row in report["episodes"]}
            self.assertEqual(len(by_seed), 2)

            budget_row = by_seed[7]
            self.assertTrue(budget_row["interrupted"], diagnostics)
            self.assertEqual(budget_row["legs"][0]["label"], "seed7_leg1")
            labels = [leg["label"] for leg in budget_row["legs"]]
            self.assertIn("seed7_leg2_resume", labels)
            self.assertIn("seed7_leg2_restart", labels)
            self.assertTrue(budget_row["resume_used_carry"])
            self.assertGreaterEqual(budget_row["stages_gained_on_resume"], 1, diagnostics)
            self.assertTrue(budget_row["interrupt_recovered"], diagnostics)
            resume_leg = next(leg for leg in budget_row["legs"]
                              if leg["label"] == "seed7_leg2_resume")
            self.assertEqual(resume_leg["carry"]["mode"], "resume")
            self.assertEqual(resume_leg["carry"]["done"], [])
            self.assertIn("SESSION RESUME", resume_leg["plan"]["instruction"])
            # version 3 wiring: the interrupted leg wrote its runtime checkpoint
            # and the offline report scored it against the scene
            self.assertTrue(
                (out_dir / "seed7" / "seed7_leg1" / "checkpoint.json").exists()
            )
            self.assertTrue(budget_row["checkpoint"]["available"])
            self.assertIn("plan", budget_row["checkpoint"]["fields"])
            restart_leg = next(leg for leg in budget_row["legs"]
                               if leg["label"] == "seed7_leg2_restart")
            self.assertEqual(restart_leg["carry"]["mode"], "restart")

            deadline_row = by_seed[11]
            self.assertTrue(deadline_row["interrupted"], diagnostics)
            self.assertGreater(len(deadline_row["legs"]), 1)
            self.assertIn("deadline", deadline_row["legs"][0]["interrupt_reason"])
            self.assertEqual(deadline_row["legs"][0]["model_calls"], 0,
                             "the hard interrupt lands before the first decision node")

            # the plan the resumed leg adopts is written by the task module and
            # adopted through resume_plan: the runtime logs the adoption
            episode_log = json.loads(
                (out_dir / "seed7" / "seed7_leg2_resume" / "episode_log.json").read_text()
            )
            self.assertTrue(episode_log.get("plan_resumed"), episode_log.keys())
            self.assertGreaterEqual(int(episode_log["plan_resumed"].get("resume_count", 0)), 1)
            self.assertTrue(episode_log.get("plan_transitions"))
            self.assertIn("plan_resumed", [event.get("name") for event in episode_log["events"]])
            checks = report["checks"]
            self.assertTrue(checks["brief_places_are_traced"]["ok"], checks["brief_places_are_traced"])
            self.assertTrue(checks["carry_plan_steps"]["match_task"])

            aggregate = report["aggregate"]
            self.assertGreaterEqual(
                aggregate["checkpoint_correctness"]["episodes"], 1
            )
            self.assertEqual(aggregate["interrupt"]["n_episodes"], 2)
            self.assertEqual(aggregate["interrupt"]["recovered"], 2, diagnostics)
            self.assertEqual(aggregate["interrupt"]["recovery_rate"], 1.0)
            self.assertIsNotNone(aggregate["decision_vs_executor"]["executor_legs_per_decision"])
            self.assertGreaterEqual(aggregate["decision_vs_executor"]["executor_legs"], 2)
            self.assertGreaterEqual(aggregate["n_legs"], 4)
            episodes_file = (out_dir / DRIVER.EPISODES_JSONL).read_text().strip().splitlines()
            self.assertEqual(len(episodes_file), 2)

    def test_resume_off_runs_the_control_leg_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            scene, task_file, script = self._fixtures(directory)
            out_dir = directory / "out"
            proc = self._run(out_dir, scene, task_file, script, "--seeds", "7",
                             "--interrupt", "7:budget:2", "--resume", "off")
            diagnostics = proc.stdout[-4000:] + "\n--- stderr ---\n" + proc.stderr[-3000:]
            self.assertEqual(proc.returncode, 0, diagnostics)
            report = json.loads((out_dir / DRIVER.REPORT_JSON).read_text())
            row = report["episodes"][0]
            labels = [leg["label"] for leg in row["legs"]]
            self.assertIn("seed7_leg2_control", labels)
            self.assertNotIn("seed7_leg2_resume", labels)
            self.assertFalse(row["resume_used_carry"])
            control = next(leg for leg in row["legs"] if leg["label"] == "seed7_leg2_control")
            self.assertEqual(control["carry"]["channels"], ["pose"])
            self.assertEqual(control["plan_resumed"], {})

    def test_dry_run_lists_the_legs_without_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            scene, task_file, script = self._fixtures(directory)
            out_dir = directory / "out"
            proc = self._run(out_dir, scene, task_file, script, "--seeds", "7,11",
                             "--interrupt", "7:budget:3", "--dry-run")
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            payload = json.loads(proc.stdout[proc.stdout.index("{"):])
            self.assertEqual(payload["seeds"], [7, 11])
            self.assertEqual(payload["chain_check"]["ok"], True)
            self.assertEqual(payload["legs_per_seed"]["7"][0]["interrupt_mode"], "budget")
            self.assertFalse(out_dir.exists())


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# the plan the executor keeps vs. the stages the evidence supports
# ---------------------------------------------------------------------------


class PlanVsEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.task = make_task(Path(self.tmp.name))

    def test_agreement_when_the_executor_and_the_evidence_match(self):
        places = [stage.position for stage in self.task.stages]
        entries = [complete_stage_entry(1, places[0], "obs_1"),
                   complete_stage_entry(2, places[1], "obs_2")]
        report = DRIVER.stage_report(entries, self.task)
        verdict = DRIVER.plan_vs_evidence(
            self.task, report,
            carry_subtasks=[{"id": "st_1", "status": "done"},
                            {"id": "st_2", "status": "active"},
                            {"id": "st_3", "status": "pending"}],
            transitions=[{"id": "st_2", "from": "active", "to": "done", "reason": "done_when_met"},
                         {"id": "st_3", "from": "pending", "to": "active",
                          "reason": "auto_advanced"}],
        )
        self.assertTrue(verdict["agree"], verdict["disagreements"])
        self.assertEqual(verdict["disagreements"], [])

    def test_disagreement_when_a_subtask_closed_without_evidence(self):
        # the executor closed stage 2 (a search found the *next* stage's object)
        # while no tool result ever located the cabinet
        places = [stage.position for stage in self.task.stages]
        entries = [complete_stage_entry(1, places[0], "obs_1")]
        report = DRIVER.stage_report(entries, self.task)
        verdict = DRIVER.plan_vs_evidence(
            self.task, report,
            carry_subtasks=[{"id": "st_1", "status": "done"},
                            {"id": "st_2", "status": "active"},
                            {"id": "st_3", "status": "pending"}],
            transitions=[{"id": "st_2", "from": "active", "to": "done", "reason": "done_when_met"}],
        )
        self.assertFalse(verdict["agree"])
        self.assertEqual([row["stage"] for row in verdict["disagreements"]], ["object"])
        self.assertFalse(verdict["disagreements"][0]["evidence_complete"])

    def test_per_leg_check_covers_every_leg(self):
        places = [stage.position for stage in self.task.stages]
        fresh = make_leg(
            [complete_stage_entry(1, places[0], "obs_1")],
            summary={"turns": 1},
            log={"plan_transitions": [
                {"id": "st_1", "from": "active", "to": "done", "reason": "done_when_met"}]},
        )
        resume = make_leg(
            [complete_stage_entry(2, places[1], "obs_2", global_seq=2)],
            label="seed7_leg2_resume", role="resume",
            summary={"turns": 1},
            carry={"mode": "resume", "plan_subtasks": [
                {"id": "st_1", "status": "done"},
                {"id": "st_2", "status": "active"},
                {"id": "st_3", "status": "pending"},
            ]},
            log={"plan_transitions": [
                {"id": "st_2", "from": "active", "to": "done", "reason": "done_when_met"}]},
        )
        rows = DRIVER.plan_vs_evidence_by_leg(self.task, [fresh, resume])
        self.assertEqual(rows["checked"], 2)  # one check per leg, not per episode
        self.assertEqual(rows["agreed_legs"], 2)
        self.assertTrue(rows["ok"])
        self.assertEqual(rows["disagreements"], [])
        self.assertEqual(rows["unsafe_done_claims"], [])
        self.assertEqual(rows["lagging_statuses"], [])

        # a status that disagrees with the evidence is caught on the leg that
        # claims it, even though the earlier leg looked consistent
        resume.log["plan_transitions"] = []
        rows = DRIVER.plan_vs_evidence_by_leg(self.task, [fresh, resume])
        self.assertEqual(rows["agreed_legs"], 1)
        self.assertFalse(rows["ok"])
        self.assertEqual(
            [(row["leg"], row["stage"]) for row in rows["disagreements"]],
            [("seed7_leg2_resume", "object")],
        )
        # the carried status is ``active`` (not a done claim): conservative lag,
        # not the unsafe direction (a done claim without evidence)
        self.assertEqual(rows["unsafe_done_claims"], [])
        self.assertEqual(
            [(row["leg"], row["stage"]) for row in rows["lagging_statuses"]],
            [("seed7_leg2_resume", "object")],
        )

    def test_closest_place_says_how_close_the_agent_looked(self):
        place = self.task.stages[0].position
        offset = (place[0] + 3.0, place[1], place[2])
        entries = [entry(1, "ground_target",
                         data={"candidates": [{"position_flu": list(offset)}]},
                         pose=list(place), evidence=[ref("observation", "obs_1")])]
        report = DRIVER.stage_report(entries, self.task)
        row = report["stages"]["landmark"]
        self.assertFalse(row["located"])
        self.assertAlmostEqual(row["closest_place_distance_m"], 3.0)
        self.assertEqual(row["closest_place_tool"], "ground_target")


class AnalyzeRunTest(unittest.TestCase):
    """The offline re-analysis reads a finished run's artifacts (no world, no GPU)."""

    def test_load_leg_round_trips_the_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            leg_dir = Path(tmp) / "seed7_leg1"
            leg_dir.mkdir(parents=True)
            entries = [complete_stage_entry(1, (-1.0, 2.0, 0.5), "obs_1"),
                       entry(2, "finish", data={"success": False})]
            (leg_dir / "ledger.jsonl").write_text(
                "\n".join(json.dumps(row) for row in entries)
            )
            (leg_dir / "run_config.json").write_text(json.dumps({
                "leg": {"label": "seed7_leg1", "instruction": "do it",
                        "prompt_episode_id": "ep_007", "tool_budget": 8, "max_turns": 8,
                        "role": "fresh", "carry": {"mode": "resume", "plan_subtasks": []}},
            }))
            (leg_dir / "leg_summary.json").write_text(json.dumps({
                "interrupted": True, "interrupt_reason": "budget",
                "executor": {"legs": 3, "ticks": 9}, "vlm_calls_wrapper": 4,
                "wall_time_s": 2.5, "start_pose": [0.0, 0.0, 1.0],
            }))
            (leg_dir / "episode_log.json").write_text(json.dumps({
                "turn_decisions": [{"turn": 1, "tool": "observe", "decision": "new_candidate"}],
                "decision_nodes": [{"turn": 1, "node": "new_candidate"}],
                "plan_transitions": [{"id": "st_1", "to": "done", "reason": "done_when_met"}],
            }))
            leg = DRIVER.load_leg(leg_dir)
        self.assertIsNotNone(leg)
        self.assertTrue(leg.interrupted)
        self.assertEqual(leg.plan.role, "fresh")
        self.assertEqual(leg.executor["legs"], 3)
        self.assertEqual(len(leg.entries), 2)
        metrics = DRIVER.leg_metrics(leg)
        self.assertEqual(metrics["model_calls"], 4)
        self.assertEqual(metrics["decision_nodes"], 1)
        self.assertEqual(metrics["n_plan_transitions"], 1)

    def test_load_leg_restores_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            leg_dir = Path(tmp) / "seed7_leg1"
            leg_dir.mkdir(parents=True)
            (leg_dir / "ledger.jsonl").write_text("")
            (leg_dir / "run_config.json").write_text(json.dumps({
                "leg": {"label": "seed7_leg1", "instruction": "do it",
                        "prompt_episode_id": "ep_007", "tool_budget": 8, "max_turns": 8,
                        "role": "fresh", "carry": {}},
            }))
            plan = {"plan_id": "plan_1", "subtasks": [
                {"id": "st_1", "status": "done"},
                {"id": "st_2", "status": "pending"},
                {"id": "st_3", "status": "pending"},
            ]}
            checkpoint = {
                "version": 3, "source": "runtime.ctx.extra",
                "source_leg": "seed7_leg1",
                "pose": {"position_flu": [0.0, 0.0, 1.0], "yaw_rad": 0.0},
                "plan": plan, "confirmation_ledger": {}, "candidates": {},
                "observations": {}, "viewpoints_by_id": {}, "hypotheses": {},
                "task_scope": {}, "beliefs": {}, "open_questions": None,
                "evidence_pack": {}, "last_observation_id": "",
            }
            (leg_dir / "checkpoint.json").write_text(json.dumps(checkpoint))
            leg = DRIVER.load_leg(leg_dir)
        self.assertIsNotNone(leg)
        self.assertEqual(leg.checkpoint, checkpoint)
        self.assertEqual(leg.final_plan, plan)

    def test_analyze_run_scores_the_checkpoint(self):
        """Re-analysis must load ``checkpoint.json`` and score it, not 0/0."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            task = make_task(run_dir)
            (run_dir / "task.json").write_text(json.dumps({
                "name": task.name, "instruction": task.instruction,
                "scene": task.scene, "stages": [stage.to_dict() for stage in task.stages],
            }))
            leg_dir = run_dir / "seed7" / "seed7_leg1"
            leg_dir.mkdir(parents=True)
            (leg_dir / "ledger.jsonl").write_text("")
            (leg_dir / "run_config.json").write_text(json.dumps({
                "leg": {"label": "seed7_leg1", "instruction": task.instruction,
                        "prompt_episode_id": "ep_007", "tool_budget": 8, "max_turns": 8,
                        "role": "fresh", "carry": {}},
            }))
            (leg_dir / "leg_summary.json").write_text(json.dumps({
                "interrupted": True, "interrupt_reason": "budget_exhausted",
                "executor": {"legs": 0, "ticks": 0}, "vlm_calls_wrapper": 1,
                "wall_time_s": 1.0, "start_pose": [0.0, 0.0, 1.0],
            }))
            plan = {"plan_id": "plan_1", "subtasks": [
                {"id": "st_1", "status": "done"},
                {"id": "st_2", "status": "pending"},
                {"id": "st_3", "status": "pending"},
            ]}
            checkpoint = {
                "version": 3, "source": "runtime.ctx.extra",
                "source_leg": "seed7_leg1",
                "pose": {"position_flu": [0.0, 0.0, 1.0], "yaw_rad": 0.0},
                "plan": plan, "confirmation_ledger": {}, "candidates": {},
                "observations": {}, "viewpoints_by_id": {}, "hypotheses": {},
                "task_scope": {}, "beliefs": {}, "open_questions": None,
                "evidence_pack": {}, "last_observation_id": "",
            }
            (leg_dir / "checkpoint.json").write_text(json.dumps(checkpoint))
            report = DRIVER.analyze_run(run_dir, write=False)
        episode = report["episodes"][0]
        self.assertTrue(episode["checkpoint"]["available"])
        self.assertEqual(episode["checkpoint"]["wrong_done"], ["st_1"])
        self.assertEqual(report["aggregate"]["checkpoint_correctness"]["episodes"], 1)
        self.assertEqual(
            report["aggregate"]["checkpoint_correctness"]["wrong_done_beliefs"], 1
        )

    def test_analyze_run_rebuilds_the_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            task = make_task(run_dir)
            (run_dir / "task.json").write_text(json.dumps({
                "name": task.name, "instruction": task.instruction,
                "scene": task.scene, "stages": [stage.to_dict() for stage in task.stages],
            }))
            seed_dir = run_dir / "seed7"
            leg_dir = seed_dir / "seed7_leg1"
            leg_dir.mkdir(parents=True)
            places = [stage.position for stage in task.stages]
            rows = [complete_stage_entry(i + 1, place, f"obs_{i + 1}")
                    for i, place in enumerate(places)]
            rows.append(entry(9, "verify_goal", data={"verified": True},
                              pose=list(places[2]),
                              evidence=[ref("observation", "obs_9")]))
            (leg_dir / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            (leg_dir / "run_config.json").write_text(json.dumps({
                "leg": {"label": "seed7_leg1", "instruction": task.instruction,
                        "prompt_episode_id": "ep_007", "tool_budget": 8, "max_turns": 8,
                        "role": "fresh", "carry": {}},
            }))
            (leg_dir / "leg_summary.json").write_text(json.dumps({
                "interrupted": False, "interrupt_reason": "",
                "executor": {"legs": 2, "ticks": 5}, "vlm_calls_wrapper": 4,
                "wall_time_s": 1.0, "start_pose": [0.0, 0.0, 1.0],
            }))
            (leg_dir / "summary.json").write_text(json.dumps({
                "success": True, "finish_reason": "finish", "success_dist_m": 2.0,
            }))
            report = DRIVER.analyze_run(run_dir, write=False)
            self.assertEqual(report["meta"]["n_episodes"], 1)
            self.assertEqual(report["episodes"][0]["stages"]["complete_count"], 3)
            self.assertTrue(report["episodes"][0]["traceable_success"])
            text = DRIVER.report_markdown(report)
            self.assertIn("Long-horizon verification", text)



# ---------------------------------------------------------------------------
# goal.md v1 offline contract: scope-filtered stages/completion and main metrics
# ---------------------------------------------------------------------------


def scoped_entry(
    seq: int,
    tool: str,
    *,
    subtask: str,
    candidate: str = "",
    revision: int = 0,
    data: dict | None = None,
    task_id: str = "plan_1",
    **kwargs,
) -> dict:
    """One ledger entry whose result carries the v1 task scope."""
    payload = dict(data or {})
    payload["scope"] = {
        "task_id": task_id,
        "subtask_id": subtask,
        "candidate_id": candidate,
        "evidence_revision": int(revision),
    }
    return entry(seq, tool, data=payload, **kwargs)


class ScopeContractTest(unittest.TestCase):
    """The offline evaluator applies the same scope filter as ``finish``."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.task = make_task(Path(self.tmp.name))
        self.places = [stage.position for stage in self.task.stages]
        self.mapping = DRIVER.stage_subtask_map(self.task, [])

    def test_stage_report_ignores_a_result_of_another_subtask(self):
        # the object place is located by a call made while the *landmark*
        # sub-task was executing: legacy evaluation would count it for stage 2
        entries = [
            scoped_entry(1, "ground_target", subtask="st_1",
                         data={"candidates": [{"position_flu": list(self.places[1])}]},
                         pose=list(self.places[1]), evidence=[ref("observation", "obs_1")]),
        ]
        legacy = DRIVER.stage_report(entries, self.task, scoped=False)
        self.assertTrue(legacy["stages"]["object"]["located"])
        scoped = DRIVER.stage_report(
            entries, self.task, subtask_map=self.mapping, scoped=True
        )
        self.assertFalse(scoped["stages"]["object"]["located"])
        self.assertTrue(scoped["scope"]["active"])
        self.assertEqual(scoped["stages"]["object"]["scope_excluded_entries"], 1)
        # the landmark row is still judged on its own scoped call
        self.assertFalse(scoped["stages"]["landmark"]["located"])

    def test_scoped_stage_report_counts_the_subtask_own_call(self):
        entries = [
            scoped_entry(1, "ground_target", subtask="st_2", candidate="cand_1",
                         revision=1,
                         data={"candidate_id": "cand_1",
                               "position_flu": list(self.places[1])},
                         pose=list(self.places[1]), evidence=[ref("observation", "obs_2")]),
            scoped_entry(2, "confirm_candidate", subtask="st_2", candidate="cand_1",
                         revision=1,
                         data={"candidate_id": "cand_1", "verdict": "accepted"},
                         evidence=[ref("observation", "obs_9")]),
        ]
        report = DRIVER.stage_report(
            entries, self.task, subtask_map=self.mapping, scoped=True
        )
        self.assertTrue(report["stages"]["object"]["located"])
        self.assertTrue(report["stages"]["object"]["reach"])
        self.assertTrue(report["stages"]["object"]["confirmed"])
        self.assertTrue(report["stages"]["object"]["complete"])
        self.assertFalse(report["stages"]["landmark"]["complete"])

    def test_a_scoped_stage_without_a_confirmation_is_not_identity_complete(self):
        """Reach + locate is only the version-1 half: identity needs a confirm."""
        entries = [
            scoped_entry(1, "ground_target", subtask="st_2", candidate="cand_1",
                         revision=1,
                         data={"candidate_id": "cand_1",
                               "position_flu": list(self.places[1])},
                         pose=list(self.places[1]), evidence=[ref("observation", "obs_2")]),
        ]
        report = DRIVER.stage_report(
            entries, self.task, subtask_map=self.mapping, scoped=True
        )
        row = report["stages"]["object"]
        self.assertTrue(row["reach"])
        self.assertTrue(row["located"])
        self.assertIs(row["confirmation_required"], True)
        self.assertIs(row["confirmed"], False)
        self.assertIs(row["complete"], False)
        self.assertEqual(report["identity_complete_count"], 0)

    def test_a_stale_confirmation_does_not_complete_the_stage(self):
        """A confirmation of the previous candidate/revision is not identity."""
        far = [self.places[1][0] + 4.0, self.places[1][1], self.places[1][2]]
        entries = [
            scoped_entry(1, "ground_target", subtask="st_2", candidate="cand_1",
                         revision=1,
                         data={"candidate_id": "cand_1",
                               "position_flu": list(self.places[1])},
                         pose=list(self.places[1]), evidence=[ref("observation", "obs_2")]),
            scoped_entry(2, "confirm_candidate", subtask="st_2", candidate="cand_1",
                         revision=1,
                         data={"candidate_id": "cand_1", "verdict": "accepted"},
                         evidence=[ref("observation", "obs_9")]),
            scoped_entry(3, "ground_target", subtask="st_2", candidate="cand_2",
                         revision=2,
                         data={"candidate_id": "cand_2", "position_flu": far},
                         pose=list(self.places[1]), evidence=[ref("observation", "obs_10")]),
        ]
        report = DRIVER.stage_report(
            entries, self.task, subtask_map=self.mapping, scoped=True
        )
        row = report["stages"]["object"]
        self.assertIs(row["confirmed"], False)
        self.assertIs(row["complete"], False)
        self.assertEqual(row["confirmation_inherited_from"], [])

    def test_a_nearby_new_candidate_inherits_the_confirmation(self):
        """A re-grounding 0.4 m from the confirmed candidate is the same
        instance: the acceptance is inherited instead of being made stale."""
        nearby = [self.places[1][0] + 0.4, self.places[1][1], self.places[1][2]]
        entries = [
            scoped_entry(1, "ground_target", subtask="st_2", candidate="cand_1",
                         revision=1,
                         data={"candidate_id": "cand_1",
                               "position_flu": list(self.places[1])},
                         pose=list(self.places[1]), evidence=[ref("observation", "obs_2")]),
            scoped_entry(2, "confirm_candidate", subtask="st_2", candidate="cand_1",
                         revision=1,
                         data={"candidate_id": "cand_1", "verdict": "accepted"},
                         evidence=[ref("observation", "obs_9")]),
            scoped_entry(3, "ground_target", subtask="st_2", candidate="cand_2",
                         revision=2,
                         data={"candidate_id": "cand_2", "position_flu": nearby},
                         pose=list(self.places[1]), evidence=[ref("observation", "obs_10")]),
        ]
        report = DRIVER.stage_report(
            entries, self.task, subtask_map=self.mapping, scoped=True
        )
        row = report["stages"]["object"]
        self.assertIs(row["confirmed"], True)
        self.assertIs(row["identity_complete"], True)
        self.assertIs(row["complete"], True)
        self.assertEqual(row["confirmation_inherited_from"], ["cand_1"])

    def test_reach_accepts_the_candidate_standoff_shell(self):
        """The stage centre can stay outside ``reach_m`` while the arrival is on
        the confirmed candidate's standoff shell: that is a reach, recorded
        with its own basis."""
        candidate = [self.places[1][0] + 0.59, self.places[1][1], self.places[1][2]]
        pose = [candidate[0] + 1.2, candidate[1], candidate[2]]
        entries = [
            scoped_entry(1, "ground_target", subtask="st_2", candidate="cand_1",
                         revision=1,
                         data={"candidate_id": "cand_1", "position_flu": candidate},
                         pose=list(self.places[1]), evidence=[ref("observation", "obs_2")]),
            scoped_entry(2, "approach_candidate", subtask="st_2", candidate="cand_1",
                         revision=1,
                         data={"candidate_id": "cand_1", "outcome": "arrived",
                               "standoff_m": 1.2},
                         pose=pose, evidence=[ref("observation", "obs_3")]),
        ]
        report = DRIVER.stage_report(
            entries, self.task, subtask_map=self.mapping, scoped=True
        )
        row = report["stages"]["object"]
        # the stage centre is 1.79 m away, outside reach_m=1.5; the arrival is
        # on the candidate's 1.2 m shell, which is a reach on its own
        self.assertGreater(
            DRIVER._distance(pose, list(self.places[1])), row["reach_m"]
        )
        self.assertTrue(row["reach"])
        self.assertEqual(row["reach_basis"], "candidate")
        self.assertAlmostEqual(row["reach_distance_m"], 1.2, places=6)
        self.assertAlmostEqual(row["reach_candidate_distance_m"], 1.2, places=6)
        self.assertAlmostEqual(row["reach_standoff_m"], 1.2, places=6)

    def test_a_wrong_instance_confirmation_is_not_identity_complete(self):
        """An accepted instance 3 m from the stage place is not this stage's object."""
        wrong = [self.places[1][0] + 3.0, self.places[1][1], self.places[1][2]]
        entries = [
            scoped_entry(1, "ground_target", subtask="st_2", candidate="cand_1",
                         revision=1,
                         data={"candidate_id": "cand_1", "position_flu": wrong},
                         pose=list(self.places[1]), evidence=[ref("observation", "obs_2")]),
            scoped_entry(2, "confirm_candidate", subtask="st_2", candidate="cand_1",
                         revision=1,
                         data={"candidate_id": "cand_1", "verdict": "accepted"},
                         evidence=[ref("observation", "obs_9")]),
        ]
        report = DRIVER.stage_report(
            entries, self.task, subtask_map=self.mapping, scoped=True
        )
        row = report["stages"]["object"]
        self.assertIs(row["confirmed"], True)
        self.assertIs(row["confirmation_correct"], False)
        self.assertIs(row["identity_complete"], False)
        self.assertIs(row["complete"], False)

    def test_scoped_completion_rejects_a_stale_candidate_confirmation(self):
        target = self.places[2]
        entries = [
            scoped_entry(1, "ground_target", subtask="st_3", candidate="cand_1",
                         revision=1,
                         data={"candidate_id": "cand_1",
                               "position_flu": list(target)},
                         evidence=[ref("memory", "cand_1")]),
            scoped_entry(2, "verify_goal", subtask="st_3", candidate="cand_1",
                         revision=1, data={"verified": True},
                         evidence=[ref("observation", "obs_1")]),
            scoped_entry(3, "ground_target", subtask="st_3", candidate="cand_2",
                         revision=2,
                         data={"candidate_id": "cand_2",
                               "position_flu": list(target)},
                         evidence=[ref("memory", "cand_2")]),
        ]
        verdict = DRIVER.completion_chain(entries)
        self.assertTrue(verdict["raw_verified"])
        self.assertFalse(verdict["verified"])
        self.assertTrue(verdict["scoped"])
        self.assertGreaterEqual(verdict["rejected_records"], 1)
        # the current candidate's arrival plus its accepted confirmation closes
        # the scoped chain (version 2: arrival alone shares the grounding place)
        entries.append(
            scoped_entry(4, "approach_candidate", subtask="st_3",
                         candidate="cand_2", revision=2,
                         data={"candidate_id": "cand_2", "outcome": "arrived",
                               "target_position": list(target)},
                         evidence=[ref("memory", "cand_2")])
        )
        without = DRIVER.completion_chain(entries)
        self.assertFalse(without["verified"])
        entries.append(
            scoped_entry(5, "confirm_candidate", subtask="st_3",
                         candidate="cand_2", revision=2,
                         data={"candidate_id": "cand_2", "verdict": "accepted"},
                         evidence=[ref("observation", "obs_9")])
        )
        verdict = DRIVER.completion_chain(entries)
        self.assertTrue(verdict["verified"])

    def test_offline_plan_completion_uses_the_task_module(self):
        stages = self.task.stages
        leg = make_leg(
            [scoped_entry(1, "observe", subtask="st_1",
                          data={}, evidence=[ref("observation", "obs_1")])],
            carry={"plan_subtasks": [
                {"id": f"st_{s.index}", "status": "done" if s.index == 1 else "pending"}
                for s in stages
            ]},
            log={"plan_transitions": [
                {"id": "st_1", "from": "active", "to": "done", "reason": "done_when_met"},
                {"id": "st_2", "from": "pending", "to": "active", "reason": "auto_advanced"},
            ]},
        )
        verdict = DRIVER.offline_plan_completion([leg], self.task)
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict["source"], "vlm_nav.tool_impl_task.plan_completion")
        self.assertFalse(verdict["complete"])
        self.assertEqual(
            [item["subtask_id"] for item in verdict["open_subtasks"]], ["st_2", "st_3"]
        )
        self.assertEqual(verdict["done_subtasks"], ["st_1"])

        done = make_leg(
            [], carry={"plan_subtasks": [
                {"id": f"st_{s.index}", "status": "done"} for s in stages
            ]}
        )
        self.assertTrue(DRIVER.offline_plan_completion([done], self.task)["complete"])

    def test_wrong_instance_acceptance_flags_an_arrival_at_another_object(self):
        true_place = self.places[0]
        wrong_place = [true_place[0] + 6.0, true_place[1], true_place[2]]
        entries = [
            scoped_entry(1, "ground_target", subtask="st_1", candidate="cand_9",
                         revision=1,
                         data={"candidate_id": "cand_9",
                               "position_flu": list(wrong_place)},
                         evidence=[ref("memory", "cand_9")]),
            scoped_entry(2, "approach_candidate", subtask="st_1", candidate="cand_9",
                         revision=1,
                         data={"candidate_id": "cand_9", "outcome": "arrived",
                               "target_position": list(wrong_place)},
                         evidence=[ref("memory", "cand_9")]),
        ]
        verdict = DRIVER.instance_acceptance(entries, self.task, self.mapping)
        self.assertEqual(verdict["events"], 1)
        self.assertEqual(verdict["resolved"], 1)
        self.assertEqual(verdict["wrong"], 1)
        self.assertEqual(verdict["wrong_rate"], 1.0)
        self.assertEqual(verdict["rows"][0]["stage"], "landmark")

        good = [
            scoped_entry(1, "ground_target", subtask="st_1", candidate="cand_1",
                         revision=1,
                         data={"candidate_id": "cand_1",
                               "position_flu": list(true_place)},
                         evidence=[ref("memory", "cand_1")]),
            scoped_entry(2, "approach_candidate", subtask="st_1", candidate="cand_1",
                         revision=1,
                         data={"candidate_id": "cand_1", "outcome": "arrived",
                               "target_position": list(true_place)},
                         evidence=[ref("memory", "cand_1")]),
        ]
        self.assertEqual(DRIVER.instance_acceptance(good, self.task, self.mapping)["wrong"], 0)

    def test_wrong_completion_is_a_claim_without_the_ordered_chain(self):
        first = scoped_entry(1, "ground_target", subtask="st_1",
                             data={"candidates": [{"position_flu": list(self.places[0])}]},
                             pose=list(self.places[0]),
                             evidence=[ref("observation", "obs_1")])
        finish = scoped_entry(10, "finish", subtask="st_2",
                              data={"claimed": True, "verified": False,
                                    "plan_complete": False,
                                    "missing_subtasks": [
                                        {"subtask_id": "st_2", "status": "active"},
                                        {"subtask_id": "st_3", "status": "pending"},
                                    ]},
                              status="blocked", evidence=[])
        finish["args"] = {"success": True, "reason": "done"}
        leg = make_leg([first, finish], summary={"success": False, "finish_reason": "max_turns"})
        row = DRIVER.episode_metrics(self.task, [leg], gate_m=2.0)
        self.assertFalse(row["ordered_success"])
        self.assertTrue(row["completion_claim"]["wrong"])
        self.assertEqual(row["finish_contract"]["blocked"], 1)
        self.assertEqual(row["finish_contract"]["missing_subtasks"][0]["subtask_id"], "st_2")
        self.assertEqual(row["finish_contract"]["rows"][0]["calls_after"], 0)

    def test_recovery_regrounding_classifies_subgoal_and_global(self):
        trace = [
            {"turn": 2, "tool": "approach_candidate", "status": "failed",
             "data": {"scope": {"task_id": "plan_1", "subtask_id": "st_2",
                                "candidate_id": "cand_1", "evidence_revision": 1}},
             "evidence": []},
            {"turn": 3, "tool": "ground_target", "status": "ok",
             "args": {"instruction": "门旁边的柜子 - 木门附近的柜子"},
             "evidence": [ref("memory", "cand_2")]},
        ]
        leg = make_leg([], log={"recoveries": [
            {"turn": 2, "stalled_tool": "approach_candidate",
             "advice": ["re-ground the current sub-task"]},
        ]})
        (leg.out_dir / "tool_trace.jsonl").write_text(
            "\n".join(json.dumps(row) for row in trace) + "\n"
        )
        report = DRIVER.recovery_regrounding([leg], self.task)
        self.assertEqual(report["events"], 1)
        self.assertTrue(report["rows"][0]["executed"])
        self.assertEqual(report["rows"][0]["classification"], "subgoal+anchor")
        self.assertEqual(report["subgoal_scoped"], 1)

        global_trace = [
            trace[0],
            {"turn": 3, "tool": "ground_target", "status": "ok",
             "args": {"instruction": self.task.instruction}, "evidence": []},
        ]
        leg2 = make_leg([], log={"recoveries": [
            {"turn": 2, "stalled_tool": "approach_candidate", "advice": []},
        ]})
        (leg2.out_dir / "tool_trace.jsonl").write_text(
            "\n".join(json.dumps(row) for row in global_trace) + "\n"
        )
        report = DRIVER.recovery_regrounding([leg2], self.task)
        self.assertEqual(report["rows"][0]["classification"], "global_instruction")
        self.assertEqual(report["global_instruction"], 1)

    def test_aggregate_exposes_the_goal_md_main_metrics(self):
        first = scoped_entry(1, "ground_target", subtask="st_1",
                             data={"candidates": [{"position_flu": list(self.places[0])}]},
                             pose=list(self.places[0]),
                             evidence=[ref("observation", "obs_1")])
        finish = scoped_entry(10, "finish", subtask="st_2",
                              data={"claimed": True, "verified": False,
                                    "plan_complete": False},
                              status="blocked", evidence=[])
        finish["args"] = {"success": True}
        bad = make_leg([first, finish], summary={"success": False,
                                                 "finish_reason": "max_turns"})
        good_entries = []
        for i, place in enumerate(self.places):
            candidate = f"cand_{i + 1}"
            subtask = f"st_{i + 1}"
            good_entries.append(
                scoped_entry(2 * i + 1, "ground_target", subtask=subtask,
                             candidate=candidate, revision=i + 1,
                             data={"candidate_id": candidate,
                                   "position_flu": list(place)},
                             pose=list(place),
                             evidence=[ref("observation", f"obs_{i + 1}")])
            )
            good_entries.append(
                scoped_entry(2 * i + 2, "confirm_candidate", subtask=subtask,
                             candidate=candidate, revision=i + 1,
                             data={"candidate_id": candidate, "verdict": "accepted"},
                             pose=list(place),
                             evidence=[ref("observation", f"obs_c{i + 1}")])
            )
        good = make_leg(good_entries,
                        summary={"success": True, "finish_reason": "finish"})
        episodes = [
            DRIVER.episode_metrics(self.task, [bad], gate_m=2.0),
            DRIVER.episode_metrics(self.task, [good], gate_m=2.0),
        ]
        agg = DRIVER.aggregate(episodes, self.task, {})
        self.assertEqual(agg["ordered_success_rate"], 0.5)
        self.assertEqual(agg["complete_ordered_task_success"]["count"], 1)
        self.assertEqual(agg["wrong_completion"]["wrong"], 1)
        self.assertEqual(agg["wrong_completion"]["episodes_claiming"], 2)
        self.assertEqual(agg["wrong_completion"]["rate"], 0.5)
        self.assertIn("wrong_instance_acceptance", agg)
        self.assertIn("finish_contract", agg)
        self.assertEqual(agg["finish_contract"]["blocked"], 1)


class SemanticCompletionTest(unittest.TestCase):
    """goal.md section 4: the stage's done_when type decides what proves it.

    A ``sighted`` landmark completes on a located sighting (no reach and no
    confirmation); an ``arrived`` target keeps the reach + locate (+ identity)
    contract. The offline verdict and the executor share the same vocabulary.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        stages = [dict(stage) for stage in TASK_STAGES]
        stages[0]["done_when_type"] = "sighted"
        stages[1]["done_when_type"] = "sighted"
        stages[2]["done_when_type"] = "arrived"
        self.task = make_task(self.directory, stages=stages)

    def test_stage_types_are_read_from_the_task_file(self):
        self.assertEqual(
            [stage.effective_done_when_type() for stage in self.task.stages],
            ["sighted", "sighted", "arrived"],
        )

    def test_an_untyped_stage_that_declares_reach_stays_arrived(self):
        # the fixture's original stages declare reach_m and no type
        task = make_task(self.directory)
        self.assertEqual(
            [stage.effective_done_when_type() for stage in task.stages],
            ["arrived", "arrived", "arrived"],
        )

    def test_a_sighted_stage_is_complete_on_a_located_sighting_alone(self):
        place = self.task.stages[0].position
        entries = [entry(
            1, "ground_target",
            data={"candidates": [{"position_flu": list(place)}]},
            pose=[5.0, 5.0, 1.0], evidence=[ref("observation", "obs_1")],
        )]
        report = DRIVER.stage_report(entries, self.task)
        row = report["stages"]["landmark"]
        self.assertTrue(row["located"])
        self.assertFalse(row["reach"])
        self.assertFalse(row["reach_required"])
        self.assertTrue(row["complete"])
        # the flight was never attempted: the semantic criterion saved it
        self.assertEqual(row["reach_attempts"], 0)
        self.assertEqual(report["complete_count"], 1)

    def test_an_arrived_stage_is_not_complete_on_a_located_sighting(self):
        place = self.task.stages[2].position
        entries = [entry(
            1, "ground_target",
            data={"candidates": [{"position_flu": list(place)}]},
            pose=[5.0, 5.0, 1.0], evidence=[ref("observation", "obs_1")],
        )]
        report = DRIVER.stage_report(entries, self.task)
        row = report["stages"]["target"]
        self.assertTrue(row["located"])
        self.assertFalse(row["reach"])
        self.assertTrue(row["reach_required"])
        self.assertFalse(row["complete"])

    def test_a_flown_arrived_stage_completes_and_counts_its_reach_attempt(self):
        place = self.task.stages[2].position
        entries = [entry(
            1, "approach_candidate",
            data={"candidates": [{"position_flu": list(place)}],
                  "outcome": "arrived", "target_position": list(place)},
            pose=list(place), evidence=[ref("memory", "cand_1")],
        )]
        report = DRIVER.stage_report(entries, self.task)
        row = report["stages"]["target"]
        self.assertTrue(row["reach"])
        self.assertTrue(row["complete"])
        self.assertEqual(row["reach_attempts"], 1)

    def test_self_mode_writes_no_plan_and_records_the_mode(self):
        args = DRIVER.parse_args([
            "--task-file", str(self.directory / "task.json"),
            "--out", str(self.directory / "out"),
            "--plan-mode", "self",
        ])
        plans = DRIVER.plan_legs(
            7, task=self.task, args=args, prompt_episode_id="ep_007",
            interrupt=None, base_budget=8, base_turns=8, world=_StubWorld(),
        )
        first = plans[0]
        self.assertNotIn("plan", first.ctx_extra)
        self.assertEqual(first.ctx_extra.get("plan_mode"), "self")
        self.assertEqual(first.carry["channels"], [])
        self.assertEqual(first.carry["plan_steps"], [])
        self.assertIn("--plan-mode", DRIVER.runner_argv(
            self.task, args, out_dir=self.directory / "out",
            instruction="do it", tool_budget=8, max_turns=8, seed=7,
            episode_index=None,
        ))

    def test_plan_quality_audit_maps_the_models_own_plan(self):
        record = {
            "plan_id": "plan_1",
            "revision": 1,
            "steps": [],
            "subtasks": [
                {"id": "st_1", "goal": "找到走廊里的门（地标）",
                 "done_when": {"type": "sighted", "description": "door seen"}},
                {"id": "st_2", "goal": "找到门旁边的柜子", "depends_on": ["st_1"],
                 "done_when": {"type": "sighted", "description": "cabinet seen"}},
                {"id": "st_3", "goal": "确认柜子旁边的目标并飞过去", "depends_on": ["st_2"],
                 "done_when": {"type": "arrived", "description": "box reached"}},
            ],
        }
        leg = make_leg([], final_plan=record)
        audit = DRIVER.plan_quality_audit([leg], self.task, plan_mode="self")
        self.assertTrue(audit["available"])
        self.assertTrue(audit["count_match"])
        self.assertTrue(audit["order_ok"])
        self.assertEqual(audit["coverage_rate"], 1.0)
        self.assertEqual(audit["done_when_type_accuracy"], 1.0)
        self.assertTrue(audit["structure_ok"])

    def test_plan_quality_audit_flags_a_missing_stage(self):
        record = {
            "plan_id": "plan_1",
            "subtasks": [
                {"id": "st_1", "goal": "找到走廊里的门",
                 "done_when": {"type": "sighted", "description": "door seen"}},
            ],
        }
        audit = DRIVER.plan_quality_audit([make_leg([], final_plan=record)], self.task,
                                          plan_mode="self")
        self.assertFalse(audit["count_match"])
        self.assertEqual(audit["coverage"], 1)

    def test_relation_goals_are_mapped_in_plan_order(self):
        """A goal that names its anchor too ("木椅旁边的灯") goes to the later
        stage: order, not the longest keyword, disambiguates."""
        stages = [dict(stage) for stage in TASK_STAGES]
        stages[0]["keywords"] = ["木门", "door"]
        stages[1]["keywords"] = ["柜子", "cabinet"]
        stages[2]["keywords"] = ["目标", "纸箱", "box"]
        task = make_task(self.directory, stages=stages)
        indices = DRIVER.map_goals_to_stages(
            [
                "找到走廊里的木门（地标）",
                "找到木门旁边的柜子",  # names the anchor too
                "找到柜子旁边的纸箱并飞过去",
            ],
            task,
        )
        self.assertEqual(indices, [1, 2, 3])
