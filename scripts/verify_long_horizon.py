#!/usr/bin/env python
"""Long-horizon verification driver: multi-stage tasks, interruption, resume (P4).

The P4 acceptance play is a *chained* task - "first find the corridor (or its
door), then the cabinet next to it, then confirm the target beside that cabinet"
- run by the tool-calling runtime of ``scripts/run_vlm_nav_tools.py``. Three
properties are on trial, and each one is a number the driver computes from
evidence, never from the model's prose:

  stages advance in order      every stage is judged on (a) the *pose* the
                               executor really carried the drone to (reach) and
                               (b) a *located* reference for that stage's object
                               in a tool result (a candidate / target position
                               inside ``locate_m`` of the stage's declared
                               place). A stage that was only announced - in a
                               plan, in prose or in a ``finish`` reason - is not
                               complete. ``stage_reach_rate`` and
                               ``stage_complete_rate`` keep the two apart.
  an interrupt does not lose   a leg that runs out of budget (or hits the
  progress                     ``--interrupt`` deadline at a decision node) ends
                               mid-task; the resumed leg inherits what the
                               interrupted leg *established*, through the system
                               module's own interfaces:
                                 plan         the record is written by
                                              ``tool_impl_task.update_plan`` (the
                                              sub-task schema: goals, done_when,
                                              failure branches, dependencies,
                                              budgets) and adopted by the runtime
                                              through ``resume_plan``, so the
                                              executor keeps advancing the very
                                              sub-tasks the interrupted leg had
                                              open; only stages whose evidence
                                              chain was complete are marked done,
                                 pose         the world keeps the drone where the
                                              leg ended (teleport + trajectory),
                                 subgoal      ``ctx.extra["subgoal"]`` (the STATE
                                              block's ``subgoal`` label),
                                 brief        an instruction suffix built *only*
                                              from the interrupted leg's own
                                              trace (its candidates, ids, poses,
                                              end pose) - never from the scene's
                                              ground truth, so no leak
                               The runtime's log is read back per leg
                               (``decision_nodes``, ``plan_transitions``,
                               ``plan_resumed``): the report can say *why* a
                               sub-task switched, not only that it did.
                               ``interrupt_recovery_rate`` counts interrupted
                               episodes that gained at least one more stage after
                               the resume; ``no_progress_on_resume`` counts the
                               ones that did not, and ``done_stage_recheck`` says
                               whether the resumed leg re-inspected places it had
                               already established (it is allowed to - what is
                               measured is the stage progress it bought). The
                               ``--resume off`` control runs the same second leg
                               without the plan/subgoal/brief carry, and
                               ``--restart-control`` adds a leg from the original
                               pose - the arms the carry is priced against.
  the VLM is consulted at      ``decision_turns`` (model turns), the runtime's own
  decision nodes, not per      ``decision_nodes`` (``DECISION_NODES``) and
  executor step                ``routine_turns``, ``inner_tool_calls`` (calls a
                               routine made inside one decision), ``model_calls``,
                               ``executor_legs`` (world legs the tools flew) and
                               ``executor_ticks`` (the world's own control steps)
                               are reported per leg with
                               ``executor_legs_per_decision`` and
                               ``ticks_per_model_call`` - a scheduler that asked
                               the model per executor step would show ~1 there.

A *success* is traceable only when the whole chain is: every stage reached AND
located with at least one evidence reference, the scene's stage relations held by
the agent's own reported positions, and the completion chain
``vlm_nav.tools.completion_evidence`` accepts (a verification that passed, or a
grounded candidate plus an arrival, with evidence references) together with the
runner's own geometric verdict (gate + terminating finish). ``traceable_success``
is the conjunction; ``success`` (the runner's verdict) is reported next to it, so
the gap between "got there" and "can prove it" is visible.

Task file (``--task-file``): a small JSON document naming the stages, the scene
and the stage places. Places come from the scene manifest (``object_key`` /
``object_index`` into ``object_positions_flu``), from the scene target
(``"target": true``) or from an explicit ``position_flu`` (the CPU/synthetic
path):

    {
      "name": "corridor_door_cabinet_box",
      "instruction": "先找到走廊里的门，再找到门旁边的柜子，最后确认柜子旁边的目标并飞到它旁边停下",
      "scene": {"world": "env",
                "combined_ply": ".../combo.spz",
                "background_ply": ".../combo_background.spz",
                "semantic_manifest": ".../combo.json",
                "target_index": 2, "target_name": "cardboard box",
                "target_radius": 0.35, "success_dist": 1.5,
                "env_overrides": ["env.gaussian_splatting.start_goal.min_dist=6.0",
                                  "env.gaussian_splatting.start_goal.max_dist=12.0"]},
      "stages": [
        {"id": "landmark", "kind": "landmark", "label": "走廊里的门",
         "object_key": "door_wood_closed", "reach_m": 2.5, "locate_m": 2.0,
         "step": "找到走廊里的门（地标）"},
        {"id": "object", "kind": "object", "label": "门旁边的柜子",
         "object_key": "cabinet", "anchor": "landmark", "within_m": 5.0,
         "reach_m": 2.0, "locate_m": 2.0, "step": "找到门旁边的柜子"},
        {"id": "target", "kind": "target", "label": "柜子旁边的目标",
         "target": true, "anchor": "object", "within_m": 3.0,
         "reach_m": null, "locate_m": 2.0, "step": "确认柜子旁边的目标并飞到它旁边"}
      ]
    }

``reach_m`` defaults to the episode's success gate for the target stage and to
1.5 m elsewhere; ``within_m`` is the relation distance the anchor pair must
satisfy (a *scene* sanity property: the driver refuses to call the chain
traceable when the declared relation does not hold, and reports the check in
``meta.chain_check``). Stage positions are ground truth used for *evaluation and
scene validation only* - they never reach the prompt: the instruction names
objects, the resume brief names the agent's own traced positions.

Interrupt specs are per seed (``--interrupt SEED:MODE:VALUE``, repeatable):

    --interrupt 7:budget:6      leg 1 gets tool_budget=6 / max_turns=6; the leg
                                ends mid-task on the runtime's own budget rule
    --interrupt 11:deadline:45  leg 1 runs to the wall-clock deadline; the
                                wrapper raises :class:`Interrupted` at the first
                                decision node past it (a hard executor-level
                                interrupt, not a runtime rule)

``--resume on|off`` (default ``on``) runs the second leg with the inherited
state; ``--resume off`` runs the same second leg *without* the inherited state
(no pose carry, no plan, no subgoal, no brief) - the paired control that says
what the carry is worth. Both legs of one episode share the model, scene, seeds
and budgets; only the carry differs.

Mock run on a tiny CPU scene (no weights, no GPU - what the tests drive):

    python scripts/verify_long_horizon.py --task-file task.json \
        --backend mock --mock-script play.json --seeds 7,11 --out out/mock

Real run (inside the GPU container, Qwen3-VL-8B on GPU 0):

    CUDA_VISIBLE_DEVICES=0 TORCH_EXTENSIONS_DIR=$PWD/outputs/vlm_nav/torch_extensions \
    python scripts/verify_long_horizon.py --task-file task.json --backend hf \
        --model-path /models_ro/Qwen3-VL-8B --device cuda --seeds 7,11,13,17,19,23 \
        --interrupt 7:budget:6 --interrupt 11:deadline:45 --restart-control \
        --out outputs/vlm_nav/verify_long_horizon

Re-analysis of a finished run (no world, no model, no GPU): every metric and check
is computed from the artifacts, so a check added after an expensive run can be
applied to the *same* episodes without re-running anything:

    python scripts/verify_long_horizon.py --analyze out/real
    # -> out/real/report_reanalyzed.json / .md

Artifacts: ``<out>/report.json``, ``<out>/report.md``, ``<out>/episodes.jsonl``,
``<out>/task.json`` and per leg ``<out>/seed<k>/<leg>/{tool_trace.jsonl,
summary.json,leg_summary.json,run_config.json,ledger.jsonl,episode_log.json}``.
The ledger is the driver's own record of every tool call (outer and
routine-inner) with the drone pose and the executor counters before and after it:
the runner's trace is written at the end of a leg, an interrupted leg has none,
so the ledger is what the resume state is built from; ``episode_log.json`` is the
runtime's own log (decision nodes, sub-task transitions, the adopted plan).

Two diagnostics keep the verdicts honest: ``closest_place_distance_m`` says how
close *any* located reference got to a stage's place (a stage the agent never
established still shows whether it looked at the right object - an ambiguous
landmark shows up here as a candidate a couple of metres off), and
``plan_vs_evidence`` compares the runtime's sub-task statuses with the
evidence-based stage verdicts, so a plan bookkeeping that closed a sub-task
without the evidence says so instead of being read as agreement.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
TOOL_RUNNER_PATH = ROOT / "scripts" / "run_vlm_nav_tools.py"

#: progress kinds the runtime classifies a decision turn with (P3 contract).
PROGRESS_KINDS = ("information", "action", "task", "none")

#: stage kinds of a chained task.
STAGE_LANDMARK = "landmark"
STAGE_OBJECT = "object"
STAGE_TARGET = "target"
STAGE_KINDS = (STAGE_LANDMARK, STAGE_OBJECT, STAGE_TARGET)

#: tools that gather information (a repeat of one is a repeated check).
INFO_TOOLS = (
    "observe",
    "ground_target",
    "inspect_region",
    "compare_candidates",
    "query_memory",
    "check_path",
    "propose_viewpoints",
    "search_target",
    "task_status",
    "get_plan",
)
#: tools that move the drone (the executor's own work).
ACTION_TOOLS = (
    "approach_candidate",
    "go_to_viewpoint",
    "return_to_observation",
    "navigate",
)

DEFAULT_STAGE_REACH_M = 1.5
DEFAULT_STAGE_LOCATE_M = 2.0
#: The concrete continuation every stage plan declares (no place, no truth):
#: the executor can ground the stage's name without a model turn.
STAGE_NEXT_TOOL = "ground_target"
#: repeated-view detection granularity (a re-observation from the same place).
VIEW_POS_ROUND = 0.25
VIEW_YAW_ROUND_DEG = 15.0

LEG_FIRST = "leg1"
LEG_RESUME = "leg2_resume"
LEG_CONTROL = "leg2_control"

REPORT_JSON = "report.json"
REPORT_MD = "report.md"
EPISODES_JSONL = "episodes.jsonl"


# ---------------------------------------------------------------------------
# the tool runner, loaded as a module (read-only reuse)
# ---------------------------------------------------------------------------


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve string annotations via sys.modules
    spec.loader.exec_module(module)
    return module


RUNNER = _load("diffaero_script_run_vlm_nav_tools_for_long_horizon", TOOL_RUNNER_PATH)


# ---------------------------------------------------------------------------
# task specification: stages, places, chain relations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageSpec:
    """One stage of the chained task, with its evaluated place resolved."""

    stage_id: str
    label: str
    kind: str
    index: int
    position: tuple[float, float, float] | None
    reach_m: float
    locate_m: float
    step: str
    anchor: str | None = None
    within_m: float | None = None
    target: bool = False
    source: str = ""
    done_when: str = ""
    expected_observation: str = ""
    failure_branch: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "id": self.stage_id,
            "index": self.index,
            "label": self.label,
            "kind": self.kind,
            "position_flu": None if self.position is None else [float(v) for v in self.position],
            "reach_m": self.reach_m,
            "locate_m": self.locate_m,
            "step": self.step,
            "anchor": self.anchor,
            "within_m": self.within_m,
            "target": self.target,
            "position_source": self.source,
            "done_when": self.done_when,
            "expected_observation": self.expected_observation,
            "failure_branch": list(self.failure_branch),
        }


@dataclass
class TaskSpec:
    name: str
    instruction: str
    stages: list[StageSpec]
    scene: dict
    path: Path | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "instruction": self.instruction,
            "scene": dict(self.scene),
            "stages": [stage.to_dict() for stage in self.stages],
            "path": None if self.path is None else str(self.path),
            "meta": dict(self.meta),
        }


def _as_position(value: Any) -> tuple[float, float, float] | None:
    if value is None or isinstance(value, (str, bytes, bool)):
        return None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size < 3 or not np.all(np.isfinite(array[:3])):
        return None
    return tuple(float(v) for v in array[:3])


def manifest_positions(manifest: Mapping | None) -> list[tuple[float, float, float]]:
    """Object places of a combination manifest (``object_positions_flu``)."""
    raw = (manifest or {}).get("object_positions_flu")
    positions: list[tuple[float, float, float]] = []
    for item in raw or []:
        point = _as_position(item)
        if point is not None:
            positions.append(point)
    return positions


def manifest_keys(manifest: Mapping | None) -> list[str]:
    """Object keys of a combination manifest, in position order."""
    keys: list[str] = []
    for entry in (manifest or {}).get("source_objects") or []:
        if isinstance(entry, Mapping):
            keys.append(str(entry.get("key") or entry.get("name") or ""))
        else:
            keys.append(str(entry))
    return keys


def resolve_stage_position(
    stage: Mapping, manifest: Mapping | None, scene: Mapping
) -> tuple[tuple[float, float, float] | None, str]:
    """``(position, source)`` of one stage declaration.

    Order: an explicit ``position_flu``; an ``object_key``/``object_index`` into
    the manifest's ``object_positions_flu``; ``"target": true`` -> the scene
    target (``scene.target_pos`` or the manifest index of ``scene.target_index``).
    """
    explicit = _as_position(stage.get("position_flu"))
    if explicit is not None:
        return explicit, "position_flu"
    positions = manifest_positions(manifest)
    keys = manifest_keys(manifest)
    object_key = stage.get("object_key")
    if object_key not in (None, ""):
        wanted = str(object_key)
        for index, key in enumerate(keys):
            if key == wanted:
                if index < len(positions):
                    return positions[index], f"manifest:{wanted}"
                break
        raise ValueError(f"stage object_key {wanted!r} is not in the manifest ({keys})")
    if stage.get("object_index") is not None:
        index = int(stage["object_index"])
        if 0 <= index < len(positions):
            return positions[index], f"manifest_index:{index}"
        raise ValueError(f"stage object_index {index} outside the manifest's {len(positions)} objects")
    if stage.get("target"):
        explicit_target = _as_position(scene.get("target_pos"))
        if explicit_target is not None:
            return explicit_target, "scene.target_pos"
        index = scene.get("target_index")
        if index is not None and 0 <= int(index) < len(positions):
            return positions[int(index)], f"manifest_target:{int(index)}"
        if positions:
            return positions[-1], "manifest_last"
    return None, ""


def load_task(path: str | Path, *, instruction: str | None = None) -> TaskSpec:
    """Read a task file and resolve every stage place and relation."""
    path = Path(path)
    with open(path) as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        raise SystemExit(f"--task-file {path} must hold a JSON object")
    scene = dict(data.get("scene") or {})
    manifest = None
    manifest_path = scene.get("semantic_manifest")
    if manifest_path:
        candidate = Path(str(manifest_path))
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if candidate.exists():
            with open(candidate) as handle:
                manifest = json.load(handle)
    if scene.get("target_index") is not None:
        scene["target_index"] = int(scene["target_index"])
    success_dist = scene.get("success_dist")
    stages: list[StageSpec] = []
    raw_stages = data.get("stages") or []
    if not raw_stages:
        raise SystemExit(f"--task-file {path} declares no stages")
    for position, raw in enumerate(raw_stages):
        if not isinstance(raw, Mapping):
            raise SystemExit(f"stage {position} of {path} is not an object")
        kind = str(raw.get("kind") or (STAGE_TARGET if raw.get("target") else STAGE_OBJECT))
        if kind not in STAGE_KINDS:
            raise SystemExit(f"stage {position} has unknown kind {kind!r}")
        try:
            place, source = resolve_stage_position(raw, manifest, scene)
        except ValueError as exc:
            raise SystemExit(f"--task-file {path}: {exc}")
        stage_id = str(raw.get("id") or f"stage_{position + 1}")
        label = str(raw.get("label") or raw.get("step") or stage_id)
        target = bool(raw.get("target")) or kind == STAGE_TARGET
        reach_default = (
            float(success_dist) if (target and success_dist is not None) else DEFAULT_STAGE_REACH_M
        )
        stages.append(
            StageSpec(
                stage_id=stage_id,
                label=label,
                kind=kind,
                index=position + 1,
                position=place,
                reach_m=float(raw.get("reach_m", reach_default)),
                locate_m=float(raw.get("locate_m", DEFAULT_STAGE_LOCATE_M)),
                step=str(raw.get("step") or label),
                anchor=None if raw.get("anchor") in (None, "") else str(raw.get("anchor")),
                within_m=None if raw.get("within_m") is None else float(raw["within_m"]),
                target=target,
                source=source,
                done_when=str(raw.get("done_when") or ""),
                expected_observation=str(raw.get("expected_observation") or ""),
                failure_branch=tuple(
                    str(item) for item in (raw.get("failure_branch") or [])
                ),
            )
        )
    by_id = {stage.stage_id: stage for stage in stages}
    for stage in stages:
        if stage.anchor and stage.anchor not in by_id:
            raise SystemExit(
                f"stage {stage.stage_id!r} anchors on unknown stage {stage.anchor!r}"
            )
    task_instruction = instruction or data.get("instruction")
    if not task_instruction:
        raise SystemExit(f"--task-file {path} has no instruction (and none was given)")
    return TaskSpec(
        name=str(data.get("name") or path.stem),
        instruction=str(task_instruction),
        stages=stages,
        scene=scene,
        path=path,
        meta={
            "manifest": None if manifest is None else str(scene.get("semantic_manifest")),
            "manifest_keys": manifest_keys(manifest),
            "manifest_positions_flu": [list(p) for p in manifest_positions(manifest)],
            "description": str(data.get("description") or ""),
        },
    )


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.dist([float(v) for v in a[:3]], [float(v) for v in b[:3]]))


def chain_check(task: TaskSpec) -> dict:
    """Scene-level sanity of the declared chain (evaluation-side ground truth).

    A stage with an ``anchor`` must really sit inside ``within_m`` of that stage's
    place; a stage without a resolved place cannot be evaluated and is reported
    as ``unevaluable`` instead of being counted as a pass.
    """
    rows: list[dict] = []
    problems: list[str] = []
    for stage in task.stages:
        row: dict[str, Any] = {"stage": stage.stage_id, "anchor": stage.anchor}
        if stage.position is None:
            row["status"] = "unevaluable"
            problems.append(f"stage {stage.stage_id} has no resolvable place")
            rows.append(row)
            continue
        if stage.anchor:
            anchor = next(s for s in task.stages if s.stage_id == stage.anchor)
            if anchor.position is None:
                row["status"] = "unevaluable"
                problems.append(f"stage {stage.stage_id} anchors on an unevaluable stage")
            else:
                gap = _distance(stage.position, anchor.position)
                limit = stage.within_m if stage.within_m is not None else float("inf")
                row.update(
                    {
                        "distance_m": round(gap, 3),
                        "within_m": None if stage.within_m is None else float(stage.within_m),
                        "status": "ok" if gap <= limit else "relation_violated",
                    }
                )
                if gap > limit:
                    problems.append(
                        f"{stage.stage_id} is {gap:.2f} m from {stage.anchor} "
                        f"(> within_m={limit:.2f})"
                    )
        else:
            row["status"] = "ok"
        rows.append(row)
    return {"rows": rows, "problems": problems, "ok": not problems}


# ---------------------------------------------------------------------------
# executor probe and call ledger
# ---------------------------------------------------------------------------


class ExecutorProbe:
    """Counts what the *executor* did, independent of the model's decisions.

    Installed on the world instance (never on the module source): ``legs`` counts
    ``apply_action`` calls - one per executor segment a tool flew - and ``renders``
    the frames the world produced. ``ticks`` is the world's own control-step
    count, read from how many trajectory samples each leg appended: a world that
    interpolates a leg into ``move_dt_s`` steps (``SplatWorldCPU``/``HiGS``)
    reports more ticks than legs, a world that writes one pose per action (the
    ``env`` world) reports ``ticks == legs``. Both numbers are published, so a
    report says which world measured what.
    """

    def __init__(self, world: Any):
        self.world = world
        self.legs = 0
        self.renders = 0
        self.points = 0
        self.installed: list[str] = []
        self._install()

    def _trajectory_len(self) -> int:
        trajectory = getattr(self.world, "trajectory", None)
        try:
            return len(trajectory) if trajectory is not None else 0
        except TypeError:  # pragma: no cover - a foreign duck-typed world
            return 0

    def _install(self) -> None:
        original_apply = getattr(self.world, "apply_action", None)
        if callable(original_apply):
            def counted_apply(action, _original=original_apply):
                points_before = self._trajectory_len()
                text = _original(action)
                self.legs += 1
                self.points += max(1, self._trajectory_len() - points_before)
                return text

            self.world.apply_action = counted_apply
            self.installed.append("apply_action")
        original_render = getattr(self.world, "render", None)
        if callable(original_render):
            def counted_render(*call_args, **call_kwargs):
                self.renders += 1
                return original_render(*call_args, **call_kwargs)

            self.world.render = counted_render
            self.installed.append("render")

    def snapshot(self) -> dict:
        return {
            "legs": int(self.legs),
            "renders": int(self.renders),
            "points": int(self.points),
            "trajectory": self._trajectory_len(),
        }


def world_pose(world: Any) -> tuple[list[float], float]:
    """The drone pose ``([x, y, z], yaw_rad)`` a world reports, or ``([], 0.0)``."""
    state = getattr(world, "state", None)
    position = getattr(world, "position_flu", None)
    if callable(position):
        position = position()
    if position is None:
        position = getattr(state, "position_flu", None)
    yaw = getattr(world, "yaw_rad", None)
    if callable(yaw):
        yaw = yaw()
    if yaw is None:
        yaw = getattr(state, "yaw_rad", 0.0)
    try:
        values = [float(v) for v in np.asarray(position, dtype=float).reshape(-1)[:3]]
    except (TypeError, ValueError):
        return [], float(yaw or 0.0)
    return values, float(yaw or 0.0)


class Ledger:
    """The driver's own record of the calls a leg made (the resume source).

    ``registry.call`` is the single funnel both runtimes and the routines inside
    them go through, so wrapping it captures every outer *decision* call
    (``depth == 0``) and every call a routine made on its own allowance
    (``depth > 0``) with the pose and the executor counters before and after it.
    The runner's own ``tool_trace.jsonl`` is written at the end of a leg; an
    interrupted leg has none, which is why the resume state is built from here.
    """

    def __init__(self, world: Any, probe: ExecutorProbe):
        self.world = world
        self.probe = probe
        self.entries: list[dict] = []
        self._depth = 0

    def snapshot(self) -> dict:
        """The pose and the executor counters at one instant."""
        position, yaw = world_pose(self.world)
        return {"pose": position, "yaw": yaw, "executor": self.probe.snapshot()}

    def record(self, name: str, args: Mapping, result: Any, before: dict, after: dict,
               started: float, depth: int, error: str | None = None) -> dict:
        entry = {
            "seq": len(self.entries) + 1,
            "depth": int(depth),
            "outer": int(depth) == 0,
            "tool": str(name),
            "args": jsonable(dict(args or {})),
            "status": None if result is None else str(getattr(result, "status", "")),
            "data": {} if result is None else jsonable(dict(getattr(result, "data", {}) or {})),
            "evidence": [
                ref.to_dict() if hasattr(ref, "to_dict") else jsonable(ref)
                for ref in getattr(result, "evidence", []) or []
            ],
            "pose_before": before["pose"],
            "pose_after": after["pose"],
            "yaw_after": after["yaw"],
            "executor_before": before["executor"],
            "executor_after": after["executor"],
            "ms": round((time.monotonic() - started) * 1000.0, 1),
        }
        if error:
            entry["error"] = error
        self.entries.append(entry)
        return entry

    def wrap(self, registry: Any) -> "LedgerRegistry":
        return LedgerRegistry(registry, self)

    @property
    def outer_calls(self) -> list[dict]:
        return [entry for entry in self.entries if entry["outer"]]

    def entries_of(self, *tools: str) -> list[dict]:
        wanted = set(tools)
        return [entry for entry in self.entries if entry["tool"] in wanted]

    def poses(self) -> list[tuple[float, float, float]]:
        points: list[tuple[float, float, float]] = []
        for entry in self.entries:
            for value in (entry.get("pose_before"), entry.get("pose_after")):
                point = _as_position(value)
                if point is not None and (not points or points[-1] != point):
                    points.append(point)
        return points

    def dump(self, path: Path) -> None:
        with open(path, "w") as handle:
            for entry in self.entries:
                handle.write(json.dumps(entry, default=str) + "\n")


class LedgerRegistry:
    """A ``ToolRegistry`` proxy that records every call into a :class:`Ledger`."""

    def __init__(self, registry: Any, ledger: Ledger):
        self._registry = registry
        self._ledger = ledger

    def call(self, name: str, args: dict, ctx: Any) -> Any:
        ledger = self._ledger
        depth = ledger._depth
        ledger._depth += 1
        started = time.monotonic()
        before = ledger.snapshot()
        result = None
        error = None
        try:
            result = self._registry.call(name, dict(args or {}), ctx)
            return result
        except BaseException as exc:  # noqa: BLE001 - the ledger keeps the interrupt path
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            ledger._depth -= 1
            ledger.record(name, args, result, before, ledger.snapshot(), started, depth, error)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._registry, name)

    @property
    def registry(self) -> Any:
        return self._registry


def jsonable(value: Any) -> Any:
    """Payloads stay JSON-plain (numpy scalars, tensors and paths unwrapped)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, np.floating):
        return jsonable(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        try:
            return jsonable(value.to_dict())
        except Exception:  # noqa: BLE001 - a foreign payload keeps its repr
            return repr(value)
    return str(value)


# ---------------------------------------------------------------------------
# the model call path: an interrupt the driver controls
# ---------------------------------------------------------------------------


class Interrupted(RuntimeError):
    """A leg stopped by the driver at a decision node (the deadline interrupt).

    Raised by :class:`DeadlineVLM` before the model call that would have crossed
    the deadline; the leg's artifacts are whatever the ledger recorded up to it.
    """


class DeadlineVLM:
    """VLM proxy counting model calls and enforcing the driver's deadline.

    ``decide`` and ``decide_with_tools`` are the two entry points a runtime and a
    routine use; both check the clock first, so an interrupt lands *between*
    decision nodes and never inside the world. Every other attribute is forwarded
    to the wrapped backend.
    """

    def __init__(self, vlm: Any, deadline_s: float | None = None, started: float | None = None):
        self._vlm = vlm
        self.deadline_s = None if deadline_s is None else float(deadline_s)
        self.started = time.monotonic() if started is None else float(started)
        self.calls = 0
        self.elapsed_at_stop: float | None = None

    def _check(self) -> None:
        if self.deadline_s is None:
            return
        elapsed = time.monotonic() - self.started
        if elapsed > self.deadline_s:
            self.elapsed_at_stop = elapsed
            raise Interrupted(
                f"deadline {self.deadline_s:.1f}s reached at a decision node "
                f"(elapsed {elapsed:.1f}s)"
            )

    def decide(self, *call_args: Any, **call_kwargs: Any) -> Any:
        self._check()
        self.calls += 1
        return self._vlm.decide(*call_args, **call_kwargs)

    def decide_with_tools(self, *call_args: Any, **call_kwargs: Any) -> Any:
        self._check()
        self.calls += 1
        return self._vlm.decide_with_tools(*call_args, **call_kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._vlm, name)


# ---------------------------------------------------------------------------
# runner glue: one leg of an episode
# ---------------------------------------------------------------------------


#: driver argument -> (runner flag, nargs-multi-value)
RUNNER_SCENE_FLAGS: tuple[tuple[str, str, bool], ...] = (
    ("ply", "--ply", False),
    ("combined_ply", "--combined-ply", False),
    ("background_ply", "--background-ply", False),
    ("semantic_manifest", "--semantic-manifest", False),
    ("scene_scale", "--scene-scale", False),
    ("min_opacity", "--min-opacity", False),
    ("voxel_size", "--voxel-size", False),
    ("map_mode", "--map-mode", False),
    ("sensing", "--sensing", False),
    ("moge_model_path", "--moge-model-path", False),
    ("worker_python", "--worker-python", False),
    ("target_name", "--target-name", False),
    ("target_pos", "--target-pos", True),
    ("target_radius", "--target-radius", False),
    ("start_pos", "--start-pos", True),
    ("start_yaw_deg", "--start-yaw-deg", False),
    ("target_index", "--target-index", False),
    ("initial_yaw_mode", "--initial-yaw-mode", False),
    ("max_init_attempts", "--max-init-attempts", False),
    ("render_size", "--render-size", True),
    ("env_overrides", "--env-overrides", True),
)

#: driver argument -> runner flag, always forwarded (mechanism switches included)
RUNNER_RUN_FLAGS: tuple[tuple[str, str], ...] = (
    ("backend", "--backend"),
    ("model_path", "--model-path"),
    ("model", "--model"),
    ("base_url", "--base-url"),
    ("device", "--device"),
    ("max_new_tokens", "--max-new-tokens"),
    ("image_size", "--image-size"),
    ("num_threads", "--num-threads"),
    ("mock_script", "--mock-script"),
    ("toolset", "--toolset"),
    ("context_frames", "--context-frames"),
    ("recovery", "--recovery"),
    ("state_block", "--state-block"),
    ("search_skill", "--search-skill"),
    ("memory_spatial", "--memory-spatial"),
    ("memory_episodic", "--memory-episodic"),
    ("scan_views", "--scan-views"),
    ("scan_pitch_deg", "--scan-pitch-deg"),
    ("success_dist", "--success-dist"),
)


def _flag_tokens(flag: str, value: Any, multi: bool) -> list[str]:
    if value is None or value is False:
        return []
    if multi:
        values = value if isinstance(value, (list, tuple)) else [value]
        return [flag] + [str(item) for item in values]
    if value is True:
        return [flag]
    return [flag, str(value)]


def runner_argv(
    task: TaskSpec,
    args: argparse.Namespace,
    *,
    out_dir: Path,
    instruction: str,
    tool_budget: int | None,
    max_turns: int | None,
    seed: int,
    episode_index: int | None,
) -> list[str]:
    """The tool runner's argv for one leg: the task's scene plus the CLI knobs.

    A knob the caller did not set is left out, so the runner's own default
    applies; the scene block of the task file is the fallback for both the scene
    flags and the run-level ones it may carry (``success_dist``, ``target_name``).
    """
    argv: list[str] = [
        "--world", str(args.world or task.scene.get("world") or "cpu"),
        "--instruction", instruction,
        "--out", str(out_dir),
        "--seed", str(int(seed)),
    ]
    argv += _flag_tokens("--tool-budget", tool_budget, False)
    argv += _flag_tokens("--max-turns", max_turns, False)
    for name, flag, multi in RUNNER_SCENE_FLAGS:
        value = getattr(args, name, None)
        if value is None:
            value = task.scene.get(name)
        argv += _flag_tokens(flag, value, multi)
    for name, flag in RUNNER_RUN_FLAGS:
        value = getattr(args, name, None)
        if value is None:
            value = task.scene.get(name)
        multi = flag in ("--image-size",)
        argv += _flag_tokens(flag, value, multi)
    if getattr(args, "no_think", False):
        argv.append("--no-think")
    if episode_index is not None:
        setattr(args, "episode_index", int(episode_index))
    return argv


def make_registry_factory(ledger: Ledger, toolset: str, exclude: Sequence[str]):
    """The runner's registry, wrapped by the ledger (read-only reuse)."""

    def factory(modules: dict, vlm: Any):
        registry, backend, source = RUNNER.build_registry(
            modules, vlm=vlm, toolset=toolset, exclude=exclude
        )
        return ledger.wrap(registry), backend, source

    return factory


@dataclass
class LegPlan:
    """Everything one leg needs: instruction, budgets, carry and label."""

    label: str
    instruction: str
    prompt_episode_id: str
    tool_budget: int
    max_turns: int
    deadline_s: float | None = None
    interrupt_mode: str = ""
    #: ``fresh`` (the episode's first leg), ``resume`` (the carried continuation),
    #: ``control`` (same pose, no carry) or ``restart`` (original pose, no carry).
    role: str = "fresh"
    teleport: tuple[list[float], float] | None = None
    ctx_extra: dict = field(default_factory=dict)
    carry: dict = field(default_factory=dict)
    mock_script: str | None = None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "instruction": self.instruction,
            "prompt_episode_id": self.prompt_episode_id,
            "tool_budget": int(self.tool_budget),
            "max_turns": int(self.max_turns),
            "deadline_s": self.deadline_s,
            "interrupt_mode": self.interrupt_mode,
            "role": self.role,
            "mock_script": self.mock_script,
            "teleport": None if self.teleport is None else {
                "position_flu": [float(v) for v in self.teleport[0]],
                "yaw_rad": float(self.teleport[1]),
            },
            "carry": dict(self.carry),
        }


@dataclass
class LegResult:
    plan: LegPlan
    out_dir: Path
    summary: dict | None
    interrupted: bool
    interrupt_reason: str
    error: str | None
    entries: list[dict]
    executor: dict
    vlm_calls: int
    wall_time_s: float
    start_pose: list[float]
    start_yaw_rad: float = 0.0
    log: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        anatomy = decision_anatomy(self.log, fallback_turns=self.vlm_calls)
        return {
            "label": self.plan.label,
            "out_dir": str(self.out_dir),
            "interrupted": bool(self.interrupted),
            "interrupt_reason": self.interrupt_reason,
            "error": self.error,
            "finish_reason": None if self.summary is None else self.summary.get("finish_reason"),
            "success": None if self.summary is None else bool(self.summary.get("success")),
            "turns": None if self.summary is None else self.summary.get("turns"),
            "tool_calls": None if self.summary is None else self.summary.get("tool_calls"),
            "vlm_calls": int(self.vlm_calls),
            "wall_time_s": round(float(self.wall_time_s), 3),
            "executor": dict(self.executor),
            "final_distance_m": None if self.summary is None else self.summary.get("final_distance_m"),
            "plan": self.plan.to_dict(),
            "n_ledger_entries": len(self.entries),
            "start_pose": list(self.start_pose),
            "start_yaw_rad": float(self.start_yaw_rad),
            "decision_nodes": anatomy["decision_nodes"],
            "decision_node_kinds": anatomy["decision_node_kinds"],
            "plan_transitions": anatomy["plan_transitions"],
            "plan_resumed": anatomy["plan_resumed"],
        }


def teleport_world(world: Any, position: Sequence[float], yaw_rad: float) -> bool:
    """Put the drone at ``position``/``yaw_rad`` and re-render the sensors.

    The pose the resumed leg starts from is the pose the interrupted leg ended
    at: the interrupted leg's run is the experiment's only authority on where the
    drone was, so the world is written the way its own ``apply_action`` writes it
    (dynamics state + ``_write_pose`` when the world has one).
    """
    point = _as_position(position)
    if point is None:
        return False
    state = getattr(world, "state", None)
    if state is None:
        return False
    try:
        state.position_flu = np.asarray(point, dtype=float)
        state.yaw_rad = float(yaw_rad)
    except Exception:  # noqa: BLE001 - a frozen state object keeps its pose
        return False
    write_pose = getattr(world, "_write_pose", None)
    if callable(write_pose):
        try:
            write_pose()
        except Exception:  # noqa: BLE001 - the pose is set even if a sensor fails
            pass
    trajectory = getattr(world, "trajectory", None)
    if isinstance(trajectory, list):
        trajectory.append(tuple(float(v) for v in point))
    return True


def force_env_target(world: Any, target_index: int, seed: int) -> dict:
    """Pin the semantic target of an ``env``/``super`` world to one object index.

    The production env samples the target object on every reset
    (``reset_idx`` -> ``_assign_semantic_targets``), so a multi-object scene has
    no fixed target without pinning it: the wrapper replaces the sampled index
    with the task's, then the world is reset again so the sampled start pose
    belongs to that target (the start band is referenced to the target's own
    height). Worlds without semantic targets are left alone.
    """
    env = getattr(world, "env", None)
    original = getattr(env, "_assign_semantic_targets", None)
    if not callable(original):
        return {"applied": False, "reason": "world has no semantic targets"}
    index = int(target_index)

    def forced(env_indices, target_indices, _original=original, _index=index):
        forced_indices = torch.full_like(target_indices, _index)
        return _original(env_indices, forced_indices)

    env._assign_semantic_targets = forced
    world.reset(seed=int(seed))
    position, yaw = world_pose(world)
    return {
        "applied": True,
        "target_index": index,
        "reset": True,
        "start_pose": position,
        "start_yaw_rad": yaw,
    }


#: finish reasons the runner counts as a terminating completion.
FINISH_REASONS = ("finish", "finished", "vlm_finish", "target_verified")


def read_trace(out_dir: Path) -> list[dict]:
    """The runner's own ``tool_trace.jsonl`` of a leg (empty when interrupted)."""
    path = Path(out_dir) / "tool_trace.jsonl"
    if not path.exists():
        return []
    entries: list[dict] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                entries.append(item)
    return entries


# ---------------------------------------------------------------------------
# the P4 interface: the runtime's own plan / sub-task / evidence modules
# ---------------------------------------------------------------------------

#: Dotted modules of the P4 wiring (the same ones the tool runner imports).
M1_MODULES = {
    "task": "vlm_nav.tool_impl_task",
    "tools": "vlm_nav.tools",
    "runtime": "vlm_nav.runtime",
}


def m1_modules() -> dict[str, Any]:
    """The P4-capable modules, imported once (absent modules stay None)."""
    modules: dict[str, Any] = {}
    for key, dotted in M1_MODULES.items():
        try:
            modules[key] = importlib.import_module(dotted)
        except ImportError:  # pragma: no cover - a partial checkout
            modules[key] = None
    return modules


def carry_support() -> dict:
    """What of the P4 sub-task / resume interface is really there.

    The driver reads the interface instead of assuming it: which plan functions
    the task module exposes, whether the runtime classifies decision nodes, and
    whether the tools module judges completion chains. A missing piece degrades
    one metric (reported as ``fallback``), never the run.
    """
    modules = m1_modules()
    task_module = modules.get("task")
    tools_module = modules.get("tools")
    runtime_module = modules.get("runtime")
    return {
        "subtask_schema": all(
            hasattr(task_module, name)
            for name in ("executor_step", "resume_plan", "plan_subtasks", "SUBTASK_DONE")
        ),
        "plan_tool": all(
            hasattr(task_module, name) for name in ("update_plan", "plan_state", "plan_progress")
        ),
        "decision_nodes": bool(
            runtime_module is not None and hasattr(runtime_module, "DECISION_NODES")
        ),
        "completion_evidence": bool(
            tools_module is not None and hasattr(tools_module, "completion_evidence")
        ),
        "carry_backend": (
            "tool_impl_task.update_plan+resume_plan"
            if all(hasattr(task_module, name) for name in ("update_plan", "resume_plan"))
            else "ctx_extra.plan (legacy record)"
        ),
    }


def _carry_context(
    modules: Mapping[str, Any], world: Any, instruction: str, episode_id: str = "carry_source"
) -> Any:
    """A throwaway ``ToolContext`` used to write the plan record."""
    tools_module = modules.get("tools")
    return tools_module.ToolContext(
        world=world,
        memory=None,
        episode_id=episode_id,
        observation_seq=0,
        budget={"tool_calls_left": 0},
        extra={"instruction": instruction},
    )


def subtask_specs(
    task: TaskSpec, report: Mapping, *, include_budget: bool = True
) -> list[dict]:
    """The stage chain as sub-task specs (the schema ``update_plan`` documents).

    One sub-task per stage: its goal is the stage step, ``done_when`` says what
    finishes it, ``expected_observation`` what the model should see, the
    ``failure_branch`` what to do when the stage's leg fails, and the dependency
    on the previous stage is what makes the chain a chain (the executor cannot
    activate a sub-task whose dependency is not done).

    Every stage carries its ``reach_m`` (the movement-stage completion
    tolerance) and one concrete ``next_step`` call - the stage's own label, never
    its true place - so the executor has a deterministic step source from the
    first turn. ``include_budget=False`` is the fresh leg's plan: its per-stage
    budget is left to the episode's own tool budget.
    """
    specs: list[dict] = []
    for stage in task.stages:
        row = (report.get("stages") or {}).get(stage.stage_id) or {}
        spec: dict[str, Any] = {
            "id": f"st_{stage.index}",
            "goal": stage.step,
            "done_when": stage.done_when or (
                f"stage {stage.index} ({stage.label}) is reached and located "
                f"(a tool result within {stage.locate_m:.1f} m of it)"
            ),
            "expected_observation": stage.expected_observation or (
                f"stage {stage.index}: {stage.label}"
            ),
            "reach_m": float(stage.reach_m),
            "next_step": {
                "tool": STAGE_NEXT_TOOL,
                "args": {"instruction": stage.label},
            },
            "failure_branch": list(stage.failure_branch) or [
                "check task_status and the failure branch, re-observe from another "
                "viewpoint, then re-plan this stage with update_plan"
            ],
        }
        if include_budget:
            spec["budget"] = {"tool_calls": max(2, int(row.get("evidence_count") or 0) + 3)}
        if stage.anchor:
            anchor = next(s for s in task.stages if s.stage_id == stage.anchor)
            spec["depends_on"] = [f"st_{anchor.index}"]
        specs.append(spec)
    return specs


def _mark_done_subtasks(
    modules: Mapping[str, Any], record: dict, report: Mapping, task: TaskSpec
) -> list[dict]:
    """Mark the stages whose evidence chain closed in the interrupted leg.

    The status constant, the evidence list shape and the ``reason`` field are the
    task module's; the driver only decides *which* sub-tasks are done, from the
    stage report's own verdicts.
    """
    task_module = modules.get("task")
    done_status = getattr(task_module, "SUBTASK_DONE", "done")
    marked: list[dict] = []
    subtasks = record.get("subtasks") or []
    for position, stage in enumerate(task.stages):
        row = (report.get("stages") or {}).get(stage.stage_id) or {}
        if position >= len(subtasks) or not row.get("complete"):
            continue
        subtask = subtasks[position]
        subtask["status"] = done_status
        tokens = [
            str(ref.get("ref") or "") for ref in (row.get("evidence_refs") or [])
            if isinstance(ref, Mapping) and ref.get("ref")
        ]
        if tokens:
            subtask["evidence"] = tokens[:8]
        subtask["reason"] = "confirmed in the interrupted leg (evidence chain complete)"
        subtask["updated_turn"] = int(row.get("complete_order") or 0)
        marked.append({"id": subtask.get("id"), "stage": stage.stage_id, "evidence": tokens[:8]})
    return marked


def build_fresh_plan(
    task: TaskSpec, *, world: Any, instruction: str, episode_id: str
) -> tuple[dict | None, dict]:
    """Write the fresh leg's structured plan through ``update_plan``.

    The plan is the executor's deterministic step source: every stage carries
    ``goal``/``depends_on``/``done_when``/``reach_m`` and one concrete
    ``next_step`` (the stage's label as a ``ground_target`` call - never a truth
    coordinate). Returns ``(record, detail)``; with the plan module missing the
    leg simply starts without a plan (the pre-P4 behavior).
    """
    modules = m1_modules()
    task_module = modules.get("task")
    support = all(
        hasattr(task_module, name) for name in ("update_plan", "plan_state")
    ) and hasattr(modules.get("tools"), "ToolContext")
    if not support:
        return None, {"plan_backend": "missing", "subtasks": False}
    context = _carry_context(modules, world, instruction, episode_id=episode_id)
    note = "fresh leg: one sub-task per stage, written before the first model turn"
    result = task_module.update_plan(
        context,
        {"steps": subtask_specs(task, {}, include_budget=False), "note": note},
    )
    record = task_module.plan_state(context)
    if not isinstance(record, dict):
        return None, {
            "plan_backend": "tool_impl_task.update_plan",
            "subtasks": False,
            "error": getattr(result, "data", {}),
        }
    record["note"] = note
    return record, {
        "plan_backend": "tool_impl_task.update_plan",
        "subtasks": True,
        "n_subtasks": len(record.get("subtasks") or []),
        "statuses": [
            {"id": item.get("id"), "status": item.get("status")}
            for item in record.get("subtasks") or []
        ],
    }


def build_carry_plan(
    task: TaskSpec,
    report: Mapping,
    *,
    world: Any,
    instruction: str,
    note: str,
) -> tuple[dict, dict]:
    """Build the plan record a resumed leg adopts (the P4 interface when present).

    Returns ``(record, detail)``. With the sub-task schema in place the record is
    written by the task module's own ``update_plan`` (so the executor reads
    exactly the schema it writes), the stages whose evidence chain closed are
    marked done with their evidence tokens, and the record's ``instruction`` is
    the task module's own truncation of the resumed instruction - the same string
    ``resume_plan`` compares against. Without the module the driver falls back to
    the documented ``ctx.extra["plan"]`` record.
    """
    modules = m1_modules()
    task_module = modules.get("task")
    steps = [stage.step for stage in task.stages]
    done = [
        stage.index for stage in task.stages
        if ((report.get("stages") or {}).get(stage.stage_id) or {}).get("complete")
    ]
    total = len(task.stages)
    current = min([index for index in range(1, total + 1) if index not in done] or [total])
    spec_support = all(
        hasattr(task_module, name) for name in ("update_plan", "SUBTASK_DONE")
    ) and hasattr(modules.get("tools"), "ToolContext")
    if not spec_support:
        record = {
            "plan_id": "plan_1",
            "revision": 1,
            "steps": steps,
            "current_step": int(current),
            "done": [int(index) for index in done],
            "done_when": f"all {total} stages are confirmed and the final target is verified",
            "note": note,
            "instruction": str(instruction)[:160],
            "episode_id": "carry_source",
            "updated_t": round(time.time(), 3),
        }
        return record, {"carry_backend": "ctx_extra.plan (legacy record)", "subtasks": False, "done": done}
    context = _carry_context(modules, world, instruction)
    args: dict[str, Any] = {"steps": subtask_specs(task, report), "note": note}
    if done:
        args["done"] = [int(index) for index in done]
    args["done_when"] = f"all {total} stages are confirmed and the final target is verified"
    result = task_module.update_plan(context, args)
    record = task_module.plan_state(context)
    if not isinstance(record, dict):
        raise RuntimeError(f"the plan module refused the carried plan: {result.data}")
    marked = _mark_done_subtasks(modules, record, report, task)
    record["note"] = note
    record["updated_t"] = round(time.time(), 3)
    return record, {
        "carry_backend": "tool_impl_task.update_plan+resume_plan",
        "subtasks": True,
        "done": done,
        "marked_done": marked,
        "n_subtasks": len(record.get("subtasks") or []),
        "statuses": [
            {"id": item.get("id"), "status": item.get("status")}
            for item in record.get("subtasks") or []
        ],
    }


def completion_chain(entries: Sequence[Mapping]) -> dict:
    """The runtime's own completion-chain verdict over a leg's turns.

    ``vlm_nav.tools.completion_evidence`` is the judge of a ``success: true``
    claim: a verification that passed, or a grounded candidate plus an arrival,
    and at least one evidence reference. The driver reads the *same* chain from
    its ledger (``call`` = the ledger seq) so the report's traceability verdict
    is the runtime's, not a second opinion.
    """
    modules = m1_modules()
    tools_module = modules.get("tools")
    if tools_module is None or not hasattr(tools_module, "completion_evidence"):
        return {"supported": False}
    records = [
        {
            "tool": entry.get("tool"),
            "status": entry.get("status"),
            "data": entry.get("data") or {},
            "evidence": entry.get("evidence") or [],
            "call": entry.get("seq"),
        }
        for entry in entries
        if entry.get("tool")
    ]
    verdict = dict(tools_module.completion_evidence(records))
    verdict["supported"] = True
    return verdict


def decision_anatomy(log: Mapping | None, *, fallback_turns: int) -> dict:
    """Decision nodes, sub-task transitions and recoveries of one leg's log.

    ``EpisodeLog`` (the runtime's own report) classifies every turn: a *decision
    node* (``DECISION_NODES``) is a turn the model has to look at the plan again,
    a routine turn was absorbed by the sub-task executor. ``plan_transitions`` is
    the sub-task state machine itself - why a stage switched.
    """
    log = log or {}
    nodes = [dict(entry) for entry in (log.get("decision_nodes") or []) if isinstance(entry, Mapping)]
    transitions = [
        dict(entry) for entry in (log.get("plan_transitions") or []) if isinstance(entry, Mapping)
    ]
    kinds: dict[str, int] = {}
    for entry in nodes:
        node = str(entry.get("node") or "")
        kinds[node] = kinds.get(node, 0) + 1
    return {
        "turns": int(log.get("turns_count") or fallback_turns),
        "decision_nodes": len(nodes),
        "decision_node_kinds": kinds,
        "decision_node_turns": [entry.get("turn") for entry in nodes],
        "routine_turns": max(0, int(fallback_turns) - len(nodes)),
        "plan_transitions": transitions,
        "plan_resumed": dict(log.get("plan_resumed") or {}),
        "recoveries": [dict(entry) for entry in (log.get("recoveries") or []) if isinstance(entry, Mapping)],
    }


def _capture_runtime(holder: list) -> Any:
    """Capture the runtime instance ``run_episode`` builds (its own episode log).

    The runner returns a summary, not the runtime; the P4 report wants the
    runtime's *own* classification (decision nodes, sub-task transitions, the
    adopted resume plan), so ``build_runtime`` is wrapped for the duration of one
    leg and the instance is kept in ``holder``. Returns the restore callable.
    """
    original = RUNNER.build_runtime

    def wrapper(modules: dict, **kwargs):
        built = original(modules, **kwargs)
        runtime = built[0] if isinstance(built, tuple) and built else None
        if runtime is not None:
            holder.append(runtime)
        return built

    RUNNER.build_runtime = wrapper

    def restore() -> None:
        RUNNER.build_runtime = original

    return restore


def _episode_log_dict(captured: Sequence[Any], *, full: bool = False) -> dict:
    """The captured runtime's episode log as a mapping (empty when unavailable).

    ``full`` keeps every turn payload (the on-disk artifact); the in-memory form
    keeps the per-turn classification only, so a report does not carry whole tool
    results again.
    """
    if not captured:
        return {}
    log = getattr(captured[-1], "log", None)
    if log is None:
        return {}
    if hasattr(log, "to_dict"):
        data = dict(log.to_dict())
        turns = data.get("turns") or []
        data["turns_count"] = len(turns)
        data["turn_decisions"] = [
            {
                "turn": turn.get("turn"),
                "tool": turn.get("tool"),
                "decision": turn.get("decision"),
                "progress_kind": turn.get("progress_kind"),
                "status": turn.get("status"),
                "recovery": turn.get("recovery"),
            }
            for turn in turns
            if isinstance(turn, Mapping)
        ]
        if not full:
            data.pop("turns", None)
            data.pop("vlm_log", None)
        return data
    return dict(log) if isinstance(log, Mapping) else {}


def run_leg(
    plan: LegPlan,
    *,
    task: TaskSpec,
    args: argparse.Namespace,
    world: Any,
    probe: ExecutorProbe,
    vlm_backend: Any,
    seed: int,
    episode_index: int | None,
    out_dir: Path,
) -> LegResult:
    """Build the stack for one leg and run the tool loop (the runner's own path)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(world, probe)
    started_pose, started_yaw = world_pose(world)
    if plan.teleport is not None:
        teleport_world(world, plan.teleport[0], plan.teleport[1])
        started_pose, started_yaw = world_pose(world)
    argv = runner_argv(
        task, args,
        out_dir=out_dir,
        instruction=plan.instruction,
        tool_budget=plan.tool_budget,
        max_turns=plan.max_turns,
        seed=seed,
        episode_index=episode_index,
    )
    leg_args = RUNNER.parse_args(argv)
    leg_args.out = str(out_dir)
    if plan.mock_script:
        leg_args.mock_script = plan.mock_script
        if "--mock-script" in argv:
            argv[argv.index("--mock-script") + 1] = str(plan.mock_script)
        else:
            argv += ["--mock-script", str(plan.mock_script)]
    toolset = str(getattr(args, "toolset", RUNNER.DEFAULT_TOOLSET) or RUNNER.DEFAULT_TOOLSET)
    switches = RUNNER.switch_report(leg_args)
    exclude = () if switches["search_skill"] == "on" else (RUNNER.SEARCH_SKILL_TOOL,)
    with open(out_dir / "run_config.json", "w") as handle:
        json.dump(
            {
                "leg": plan.to_dict(),
                "runner_argv": argv,
                "switches": switches,
                "task": task.to_dict(),
                "episode_index": episode_index,
                "seed": int(seed),
                "world_flags": {
                    name: jsonable(getattr(args, name, None))
                    for name, _flag, _multi in RUNNER_SCENE_FLAGS
                },
            },
            handle,
            indent=2,
            default=str,
        )
    if vlm_backend is None:
        # the mock backend keeps a script cursor per episode: a leg gets a fresh
        # scripted model (a real backend was loaded once and is reused)
        vlm_backend = RUNNER.build_vlm(leg_args)
    vlm = DeadlineVLM(vlm_backend, plan.deadline_s)
    executor_before = probe.snapshot()
    started = time.monotonic()
    summary: dict | None = None
    error: str | None = None
    interrupted = False
    reason = ""
    captured: list[Any] = []
    restore = _capture_runtime(captured)
    try:
        summary = RUNNER.run_episode(
            world, vlm, leg_args, out_dir,
            episode_id=plan.label,
            registry_factory=make_registry_factory(ledger, toolset, exclude),
            ctx_extra=dict(plan.ctx_extra) or None,
            prompt_episode_id=plan.prompt_episode_id,
        )
    except Interrupted as exc:
        interrupted = True
        reason = str(exc)
    except Exception as exc:  # noqa: BLE001 - one bad leg must not kill the run
        error = f"{type(exc).__name__}: {exc}"
    finally:
        restore()
    wall_time = time.monotonic() - started
    executor_after = probe.snapshot()
    log = _episode_log_dict(captured)

    finish_reason = ""
    if isinstance(summary, dict):
        finish_reason = str(summary.get("finish_reason") or "")
    ended_early = bool(interrupted) or not (
        finish_reason in FINISH_REASONS and summary is not None and summary.get("success")
    )
    if plan.interrupt_mode and ended_early and not interrupted:
        interrupted = True
        reason = f"leg ended mid-task: {finish_reason or error or 'no terminating finish'}"
    executor = {
        "legs": int(executor_after["legs"] - executor_before["legs"]),
        "renders": int(executor_after["renders"] - executor_before["renders"]),
        "ticks": int(executor_after["points"] - executor_before["points"]),
        "trajectory_growth": int(executor_after["trajectory"] - executor_before["trajectory"]),
        "total_legs": int(executor_after["legs"]),
    }
    ledger.dump(out_dir / "ledger.jsonl")
    if log:
        with open(out_dir / "episode_log.json", "w") as handle:
            json.dump(jsonable(_episode_log_dict(captured, full=True) or log), handle,
                      indent=2, default=str)
    result = LegResult(
        plan=plan,
        out_dir=out_dir,
        summary=summary,
        interrupted=interrupted,
        interrupt_reason=reason,
        error=error,
        entries=list(ledger.entries),
        executor=executor,
        vlm_calls=int(vlm.calls),
        wall_time_s=wall_time,
        start_pose=[float(v) for v in started_pose],
        start_yaw_rad=float(started_yaw or 0.0),
        log=log,
    )
    partial = {
        **result.to_dict(),
        "interrupted": bool(interrupted),
        "ended_early": bool(ended_early),
        "switches": {"requested": switches, "applied": None if summary is None else summary.get("switches", {}).get("applied")},
        "outer_calls": len(ledger.outer_calls),
        "inner_calls": len(ledger.entries) - len(ledger.outer_calls),
        "vlm_calls_wrapper": int(vlm.calls),
        "deadline_elapsed_s": vlm.elapsed_at_stop,
    }
    with open(out_dir / "leg_summary.json", "w") as handle:
        json.dump(jsonable(partial), handle, indent=2, default=str)
    print(
        f"[long_horizon] {plan.label} calls={len(ledger.entries)} "
        f"(outer={len(ledger.outer_calls)}) legs={executor['legs']} "
        f"ticks={executor['ticks']} vlm={vlm.calls} "
        f"finish={finish_reason or error or 'interrupted'} "
        f"success={None if summary is None else summary.get('success')}",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# stage evaluation: what the episode really established
# ---------------------------------------------------------------------------

#: payload keys whose 3-vector value is a *place* (stage matching uses them).
_POSITION_KEYS = (
    "position_flu",
    "target_position",
    "candidate_position",
    "position",
    "place",
    "waypoint",
    "point",
    "center",
    "centre",
    "xyz",
)

#: payload keys that carry the *drone's* own pose, never a scene place: a stage is
#: located by a place a tool reported, not by the drone having been somewhere.
_POSE_KEYS = ("pose", "start_pose", "end_pose", "drone_pose", "current_pose", "last_pose")


def positions_in(payload: Any, _depth: int = 0) -> list[tuple[float, float, float]]:
    """Every *scene place* a tool result reports (candidates, targets, waypoints).

    Keys naming the drone's own pose are skipped: ``observe`` answering with the
    camera pose says where the drone is, not where the object is.
    """
    if _depth > 6:
        return []
    found: list[tuple[float, float, float]] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            lowered = str(key).lower()
            if lowered in _POSE_KEYS:
                continue
            if any(token in lowered for token in _POSITION_KEYS):
                point = _as_position(value)
                if point is not None:
                    found.append(point)
                    continue
                if isinstance(value, Mapping):  # e.g. relocalization: {position: [...]}
                    nested = _as_position(value.get("position_flu", value.get("position")))
                    if nested is not None:
                        found.append(nested)
                        continue
            found += positions_in(value, _depth + 1)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            found += positions_in(item, _depth + 1)
    return found


def order_of(entry: Mapping) -> int:
    """The episode-wide order key of a ledger entry (legs restart their ``seq``)."""
    value = entry.get("global_seq", entry.get("seq", 0))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _scan_entries(entries: Sequence[dict], stage: StageSpec) -> dict | None:
    """First call that reported a place within ``locate_m`` of the stage's place."""
    if stage.position is None:
        return None
    located: dict | None = None
    for entry in entries:
        best: tuple[float, tuple[float, float, float]] | None = None
        for point in positions_in(entry.get("data") or {}):
            gap = _distance(point, stage.position)
            if gap <= stage.locate_m and (best is None or gap < best[0]):
                best = (gap, point)
        if best is not None:
            located = {
                "seq": entry["seq"],
                "order": order_of(entry),
                "tool": entry["tool"],
                "distance_m": round(best[0], 3),
                "position_flu": [float(v) for v in best[1]],
                "evidence": list(entry.get("evidence") or []),
                "outer": bool(entry.get("outer")),
            }
            break
    return located


def stage_report(
    entries: Sequence[dict],
    task: TaskSpec,
    *,
    poses: Sequence[Sequence[float]] = (),
) -> dict:
    """Per-stage reach/locate/complete verdicts, from evidence only.

    A stage is *reached* when the executor carried the drone inside ``reach_m``
    of the stage's place (any pose the ledger or the world's trajectory shows) and
    *located* when a tool result carried a place inside ``locate_m`` of it; the
    stage is complete when both hold. Nothing here reads the model's prose: a
    plan step, a ``finish`` reason or a recovery note cannot complete a stage.
    """
    pose_points: list[tuple[float, float, float]] = []
    for item in poses:
        point = _as_position(item)
        if point is not None:
            pose_points.append(point)
    stages: dict[str, dict] = {}
    for stage in task.stages:
        row: dict[str, Any] = {
            "index": stage.index,
            "label": stage.label,
            "kind": stage.kind,
            "position_flu": None if stage.position is None else [float(v) for v in stage.position],
            "reach_m": float(stage.reach_m),
            "locate_m": float(stage.locate_m),
            "anchor": stage.anchor,
            "within_m": stage.within_m,
            "target": bool(stage.target),
            "position_source": stage.source,
            "reach": False,
            "reach_order": None,
            "reach_distance_m": None,
            "located": False,
            "located_order": None,
            "located_seq": None,
            "locate_tool": "",
            "locate_distance_m": None,
            "located_position_flu": None,
            "evidence_refs": [],
            "evidence_count": 0,
            "complete": False,
            "complete_order": None,
            "relation_ok": None,
            "relation_distance_m": None,
            "closest_place_distance_m": None,
            "closest_place_flu": None,
            "closest_place_tool": "",
            "unevaluable": stage.position is None,
        }
        if stage.position is not None:
            best_reach: tuple[float, int | None] | None = None
            for point in pose_points:
                gap = _distance(point, stage.position)
                if gap <= stage.reach_m and (best_reach is None or gap < best_reach[0]):
                    best_reach = (gap, None)
            for entry in entries:
                # both ends of every call: a leg that carried the drone *through*
                # the stage's place and on to somewhere else still reached it
                for value in (entry.get("pose_before"), entry.get("pose_after")):
                    pose_here = _as_position(value)
                    if pose_here is None:
                        continue
                    gap = _distance(pose_here, stage.position)
                    if gap <= stage.reach_m and (best_reach is None or gap < best_reach[0]):
                        best_reach = (gap, order_of(entry))
            if best_reach is not None:
                row["reach"] = True
                row["reach_distance_m"] = round(best_reach[0], 3)
                row["reach_order"] = best_reach[1]
            # how close *any* located reference got, inside the radius or not: a
            # stage the agent never established still says whether it looked at
            # the right place at all (a candidate 3 m away is a different object)
            closest: tuple[float, tuple[float, float, float], str] | None = None
            for entry in entries:
                for point in positions_in(entry.get("data") or {}):
                    gap = _distance(point, stage.position)
                    if closest is None or gap < closest[0]:
                        closest = (gap, point, str(entry.get("tool") or ""))
            if closest is not None:
                row["closest_place_distance_m"] = round(closest[0], 3)
                row["closest_place_flu"] = [float(v) for v in closest[1]]
                row["closest_place_tool"] = closest[2]
            located = _scan_entries(entries, stage)
            if located is not None:
                row.update(
                    {
                        "located": True,
                        "located_order": located["order"],
                        "located_seq": located["seq"],
                        "locate_tool": located["tool"],
                        "locate_distance_m": located["distance_m"],
                        "located_position_flu": located["position_flu"],
                        "evidence_refs": located["evidence"],
                        "evidence_count": len(located["evidence"]),
                    }
                )
        row["complete"] = bool(row["reach"] and row["located"])
        if row["complete"]:
            row["complete_order"] = max(
                [value for value in (row["reach_order"], row["located_order"]) if value]
                or [0]
            )
        stages[stage.stage_id] = row
    # the agent's own places must satisfy the declared relations (with both
    # places' localization slop on top: a chain is traceable when the places the
    # agent reported are consistent with the chain, not when they are exact)
    for stage in task.stages:
        row = stages[stage.stage_id]
        if not stage.anchor:
            continue
        anchor = stages.get(stage.anchor)
        if anchor is None or not (row.get("located") and anchor.get("located")):
            row["relation_ok"] = None
            continue
        gap = _distance(row["located_position_flu"], anchor["located_position_flu"])
        limit = (stage.within_m if stage.within_m is not None else float("inf")) + stage.locate_m
        row["relation_distance_m"] = round(gap, 3)
        row["relation_ok"] = bool(gap <= limit)
    complete = [row["index"] for row in stages.values() if row["complete"]]
    complete_orders = [
        row["complete_order"] for row in stages.values() if row["complete"]
    ]
    order_ok = complete_orders == sorted(complete_orders) and len(complete) == len(complete_orders)
    return {
        "stages": stages,
        "order": [stage.index for stage in task.stages],
        "n_stages": len(task.stages),
        "reach_count": sum(1 for row in stages.values() if row["reach"]),
        "locate_count": sum(1 for row in stages.values() if row["located"]),
        "complete_count": len(complete),
        "complete_ids": [task.stages[i - 1].stage_id for i in complete],
        "complete_orders": complete_orders,
        "order_ok": bool(order_ok),
        "chain_complete": len(complete) == len(task.stages),
        "relation_ok": all(
            row["relation_ok"] is not False for row in stages.values()
        ),
    }


def verify_evidence(entries: Sequence[dict], target: StageSpec | None, gate_m: float | None) -> dict:
    """Whether the episode carries its own verification of the final place.

    Two admissible forms, both from tool results: a ``verify_goal`` the runtime
    answered ``verified: true``, or a grounded/relocalized place inside the
    episode's own success gate around the target.
    """
    verified: list[dict] = []
    in_gate: list[dict] = []
    for entry in entries:
        data = entry.get("data") or {}
        if entry.get("tool") == "verify_goal":
            value = data.get("verified", data.get("success"))
            if value is True:
                verified.append(
                    {
                        "seq": entry["seq"],
                        "distance_m": data.get("distance_m"),
                        "limit_m": data.get("limit_m", data.get("done_dist_m")),
                        "evidence": list(entry.get("evidence") or []),
                    }
                )
        if target is None or target.position is None or gate_m is None:
            continue
        for point in positions_in(data):
            gap = _distance(point, target.position)
            if gap <= float(gate_m):
                in_gate.append(
                    {
                        "seq": entry["seq"],
                        "tool": entry["tool"],
                        "distance_m": round(gap, 3),
                        "position_flu": [float(v) for v in point],
                    }
                )
                break
    return {
        "verify_goal_verified": bool(verified),
        "verified": verified,
        "in_gate": in_gate,
        "ok": bool(verified or in_gate),
    }


# ---------------------------------------------------------------------------
# resume: what a leg hands to the next one
# ---------------------------------------------------------------------------

#: keys of the inherited state (the documented channels; see the module docstring)
RESUME_CHANNELS = ("pose", "plan", "subgoal", "brief", "stages")


def resume_brief(leg: LegResult, task: TaskSpec, report: Mapping, *, limit: int = 1200) -> dict:
    """The instruction suffix of a resumed leg, and the places it names.

    Every *stage* place named here comes from a tool result the interrupted leg
    received (a candidate, a relocalized target) - the scene's ground truth is
    never read, so a resumed episode cannot be handed a place the model did not
    establish itself. The drone's own pose is carried too (it is the agent's own
    state, the same number every prompt shows it), but the two are reported
    separately: ``places`` is what the report audits against the ledger,
    ``pose`` is where the drone is.
    """
    stages = report.get("stages") or {}
    lines = [
        "[SESSION RESUME] This episode was interrupted before the task was done "
        f"({leg.plan.interrupt_mode or 'interrupt'}: {leg.interrupt_reason or 'leg ended early'}).",
        "You already established the following in the interrupted leg, from your own tool results:",
    ]
    established: list[str] = []
    places: list[list[float]] = []
    for stage in task.stages:
        row = stages.get(stage.stage_id) or {}
        if row.get("complete"):
            place = row.get("located_position_flu")
            refs = [f"{ref.get('kind')}:{ref.get('ref')}" for ref in row.get("evidence_refs") or []]
            detail = f"stage {row['index']} ({stage.label})"
            if place:
                detail += " at (" + ", ".join(f"{float(v):.2f}" for v in place) + ")"
                places.append([float(v) for v in place])
            if refs:
                detail += " — evidence " + ", ".join(refs[:3])
            established.append(detail)
        else:
            missing = []
            if not row.get("reach"):
                missing.append("not reached")
            if not row.get("located"):
                missing.append("not located")
            established.append(
                f"stage {row.get('index', stage.index)} ({stage.label}) NOT done ({', '.join(missing) or 'unknown'})"
            )
    lines.extend(f"  - {item}" for item in established)
    pose = _as_position(leg.entries[-1].get("pose_after")) if leg.entries else None
    pose = pose or _as_position(leg.start_pose)
    pose_out: list[float] | None = None
    if pose is not None:
        yaw_deg = math.degrees(float(leg.entries[-1].get("yaw_after") or 0.0)) if leg.entries else 0.0
        pose_out = [float(v) for v in pose]
        lines.append(
            "Your drone is at (" + ", ".join(f"{float(v):.2f}" for v in pose)
            + f"), heading {yaw_deg:.0f} deg. The plan progress and the STATE block carry "
            "what is done; continue with the remaining stages from this pose and do not "
            "redo what is already established."
        )
    return {"text": "\n".join(lines)[:limit], "places": places, "pose": pose_out}


def build_resume_state(
    leg: LegResult,
    task: TaskSpec,
    *,
    prompt_episode_id: str,
    world: Any = None,
    base_instruction: str | None = None,
    report: Mapping | None = None,
) -> dict:
    """The inherited state of a resumed leg (pose, plan record, subgoal, brief).

    The plan record is built by :func:`build_carry_plan` - through the task
    module's own ``update_plan`` when its sub-task schema is there, so the resumed
    leg adopts a plan the executor can keep running rather than a hand-written
    look-alike. ``base_instruction`` is the instruction without the brief; the
    record's ``instruction`` is the *resumed* instruction (brief included), which
    is what ``resume_plan``'s guard compares against.
    """
    report = report or stage_report(leg.entries, task)
    stages = report.get("stages") or {}
    done = [stage.index for stage in task.stages if (stages.get(stage.stage_id) or {}).get("complete")]
    total = len(task.stages)
    current = min([index for index in range(1, total + 1) if index not in done] or [total])
    brief = resume_brief(leg, task, report)
    base = str(base_instruction if base_instruction is not None else leg.plan.instruction)
    resumed_instruction = f"{base}\n\n{brief['text']}" if brief["text"] else base
    note = (
        f"carried over from the interrupted leg "
        f"({leg.plan.label}: {leg.interrupt_reason or 'early end'})"
    )
    plan, carry_detail = build_carry_plan(
        task, report, world=world, instruction=resumed_instruction, note=note
    )
    subgoal = (
        f"resumed after an interruption; stages done: {done}; next: stage {current} "
        f"({task.stages[current - 1].label if 1 <= current <= total else 'done'})"
    )
    end_pose = _as_position(leg.entries[-1].get("pose_after")) if leg.entries else None
    yaw = float(leg.entries[-1].get("yaw_after") or 0.0) if leg.entries else 0.0
    return {
        "pose": {"position_flu": list(end_pose or leg.start_pose), "yaw_rad": yaw},
        "plan": plan,
        "plan_carry_detail": carry_detail,
        "subgoal": subgoal,
        "brief": brief["text"],
        "brief_places": brief["places"],
        "brief_pose": brief["pose"],
        "instruction": resumed_instruction,
        "stages": report,
        "done": done,
        "complete_count": int(report.get("complete_count", 0)),
    }


# ---------------------------------------------------------------------------
# metrics: decisions vs executor, repeats, cost
# ---------------------------------------------------------------------------


def leg_metrics(
    leg: LegResult,
    *,
    trace: Sequence[dict] | None = None,
) -> dict:
    """Decision/executor anatomy, repeat searches and cost of one leg."""
    entries = list(leg.entries)
    trace = list(trace or [])
    outer = [entry for entry in entries if entry["outer"]]
    inner = [entry for entry in entries if not entry["outer"]]
    info = [entry for entry in outer if entry["tool"] in INFO_TOOLS]
    actions = [entry for entry in outer if entry["tool"] in ACTION_TOOLS]
    seen_evidence: set[str] = set()
    no_new: list[dict] = []
    new_total = 0
    for entry in entries:
        fresh = 0
        for ref in entry.get("evidence") or []:
            key = f"{ref.get('kind')}:{ref.get('ref')}"
            if key not in seen_evidence:
                seen_evidence.add(key)
                fresh += 1
        new_total += fresh
        if entry["outer"] and entry["tool"] in INFO_TOOLS and fresh == 0:
            no_new.append(entry)
    cached = [entry for entry in info if entry["status"] == "cached"]
    view_keys = [view_key(entry) for entry in info]
    repeated_views = len(view_keys) - len(set(view_keys))
    repeats = len({entry["seq"] for entry in no_new} | {entry["seq"] for entry in cached})
    info_count = len(info)
    decision_turns = None
    if isinstance(leg.summary, dict) and isinstance(leg.summary.get("turns"), int):
        decision_turns = int(leg.summary["turns"])
    trace_turns = [entry.get("turn") for entry in trace if entry.get("turn")]
    if decision_turns is None and trace_turns:
        decision_turns = int(max(trace_turns))
    if decision_turns is None:
        decision_turns = len(outer)
    empty = [entry for entry in outer if entry["status"] in ("empty", "failed", "blocked")]
    kinds: dict[str, int] = {}
    for entry in trace:
        kind = str(entry.get("progress_kind") or "")
        if kind:
            kinds[kind] = kinds.get(kind, 0) + 1
    executor = dict(leg.executor)
    model_calls = int(leg.vlm_calls)
    anatomy = decision_anatomy(leg.log, fallback_turns=int(decision_turns))
    summary = leg.summary if isinstance(leg.summary, dict) else {}
    executor_summary = (
        summary.get("executor") if isinstance(summary.get("executor"), Mapping) else {}
    )
    auto_steps = executor_summary.get("auto_steps")
    if not isinstance(auto_steps, int):
        auto_steps = len(leg.log.get("executor_auto_steps") or [])
    saved_turns = executor_summary.get("saved_vlm_turns")
    if not isinstance(saved_turns, int):
        saved_turns = auto_steps
    logged_model_turns = summary.get("model_turns")
    if not isinstance(logged_model_turns, int):
        logged_model_turns = int(decision_turns)
    return {
        "decision_turns": int(decision_turns),
        # P4-B: deterministic continuation steps the executor ran itself (no
        # model turn) and the model turns they saved
        "model_turns": logged_model_turns,
        "executor_auto_steps": int(auto_steps),
        "vlm_turns_saved": int(saved_turns),
        "decision_nodes": anatomy["decision_nodes"],
        "decision_node_kinds": anatomy["decision_node_kinds"],
        "routine_turns": anatomy["routine_turns"],
        "plan_transitions": anatomy["plan_transitions"],
        "n_plan_transitions": len(anatomy["plan_transitions"]),
        "plan_resumed": anatomy["plan_resumed"],
        "recoveries": anatomy["recoveries"],
        "outer_tool_calls": len(outer),
        "inner_tool_calls": len(inner),
        "info_calls": info_count,
        "action_calls": len(actions),
        "empty_or_failed": len(empty),
        "model_calls": model_calls,
        "model_calls_outside_decisions": max(0, model_calls - int(decision_turns)),
        "switches": None if leg.summary is None else leg.summary.get("switches"),
        "executor": executor,
        "executor_legs_per_decision": round(executor["legs"] / decision_turns, 3) if decision_turns else None,
        "ticks_per_model_call": round(executor["ticks"] / model_calls, 3) if model_calls else None,
        "subtask_switches": sum(
            1 for entry in anatomy["plan_transitions"]
            if str(entry.get("reason") or "") == "done" or str(entry.get("to") or "") == "done"
        ),
        "repeat_checks": repeats,
        "repeat_views": repeated_views,
        "repeat_search_ratio": round(repeats / info_count, 3) if info_count else None,
        "new_evidence_total": new_total,
        "evidence_refs": sum(len(entry.get("evidence") or []) for entry in entries),
        "reported_places": reported_places(entries),
        # the runner judges the same completion chain on its own trace
        # (``summary.completion``) - kept side by side with the driver's verdict
        "runner_completion": (
            None if leg.summary is None else leg.summary.get("completion")
        ),
        "driver_completion_verified": bool(completion_chain(entries).get("verified")),
        "progress_kinds": kinds,
        "tools": tool_histogram(entries),
        "wall_time_s": round(float(leg.wall_time_s), 3),
    }


def reported_places(entries: Iterable[Mapping]) -> list[list[float]]:
    """Every place the leg's tool results reported (deduplicated, in order).

    This is the audit surface of the resume brief: a place the brief names has to
    appear here, so a brief that leaked a ground-truth coordinate is detectable.
    """
    seen: set[tuple[float, float, float]] = set()
    places: list[list[float]] = []
    for entry in entries:
        points = list(positions_in(entry.get("data") or {}))
        for value in (entry.get("pose_before"), entry.get("pose_after")):
            if not value:
                continue
            point = _as_position(value)
            if point is not None:
                points.append(point)
        for point in points:
            if point in seen:
                continue
            seen.add(point)
            places.append([float(v) for v in point])
    return places


def view_key(entry: Mapping) -> str:
    """Place a *decision* was made from - the re-observation key of the episode.

    Rounded pose + heading: two information calls from the same spot with the
    same heading are the same *view*, whatever ids the tools minted for them.
    """
    pose = _as_position(entry.get("pose_before")) or _as_position(entry.get("pose_after"))
    if pose is None:
        return f"{entry.get('seq')}"
    yaw = float(entry.get("yaw_after") or 0.0)
    quantised = tuple(round(float(v) / VIEW_POS_ROUND) for v in pose)
    sector = int(round(math.degrees(yaw) / VIEW_YAW_ROUND_DEG))
    return f"{quantised}:{sector}"


def tool_histogram(entries: Iterable[Mapping]) -> dict:
    """Calls per tool, outer and inner kept apart (finish included)."""
    rows: dict[str, dict] = {}
    for entry in entries:
        row = rows.setdefault(entry.get("tool") or "", {"outer": 0, "inner": 0})
        row["outer" if entry.get("outer") else "inner"] += 1
    return rows


def combined_verdict(
    task: TaskSpec,
    legs: Sequence[LegResult],
    *,
    gate_m: float | None,
    annotate: bool = True,
) -> dict:
    """The stage/evidence verdict of one chain of legs (in order).

    A *chain* is the fresh leg plus the leg that continues it; the control arms
    (``control``/``restart``) are judged by their own call, so an extra arm can
    never decide what the episode under test achieved.
    """
    entries: list[dict] = []
    for leg in legs:
        for entry in leg.entries:
            entries.append({**entry, "leg": leg.plan.label, "global_seq": len(entries) + 1})
    poses: list[tuple[float, float, float]] = []
    for entry in entries:
        for value in (entry.get("pose_before"), entry.get("pose_after")):
            point = _as_position(value)
            if point is not None and (not poses or poses[-1] != point):
                poses.append(point)
    report = stage_report(entries, task, poses=poses)
    target_stage = next((stage for stage in task.stages if stage.target), None)
    verify = verify_evidence(entries, target_stage, gate_m)
    chain_completion = completion_chain(entries)
    final = legs[-1] if legs else None
    success = bool(final is not None and isinstance(final.summary, dict) and final.summary.get("success"))
    evidence_complete = all(
        row["complete"] and row["evidence_count"] > 0 for row in report["stages"].values()
    )
    chain_ok = bool(
        chain_completion.get("verified")
        if chain_completion.get("supported")
        else verify["ok"]
    )
    traceable = bool(
        report["chain_complete"]
        and report["order_ok"]
        and report["relation_ok"]
        and evidence_complete
        and chain_ok
        and success
    )
    interrupted_leg = next((leg for leg in legs if leg.interrupted), None)
    before = (
        stage_report(
            [entry for entry in entries if entry["leg"] == interrupted_leg.plan.label], task
        ).get("complete_count", 0)
        if interrupted_leg is not None
        else 0
    )
    interrupted = interrupted_leg is not None
    return {
        "label": legs[0].plan.prompt_episode_id if legs else "",
        "roles": [leg.plan.role for leg in legs],
        "n_legs": len(legs),
        "success": success,
        "success_finish_reason": None if final is None or final.summary is None else final.summary.get("finish_reason"),
        "final_distance_m": None if final is None or final.summary is None else final.summary.get("final_distance_m"),
        "gate_m": gate_m,
        "stages": report,
        "verify": verify,
        "completion_chain": chain_completion,
        "evidence_complete": evidence_complete,
        "traceable_success": traceable,
        "interrupted": interrupted,
        "interrupt_at_stages": int(before),
        "stages_gained_on_resume": int(report["complete_count"] - before) if interrupted else 0,
        "interrupt_recovered": bool(interrupted and report["complete_count"] > before),
        "resume_used_carry": bool(
            interrupted and any(leg.plan.carry.get("mode") == "resume" for leg in legs[1:])
        ),
        "done_stage_recheck": done_stage_recheck(entries, task, interrupted_leg),
        "entries": entries,
        "annotations": (
            annotations(task, legs, report, verify, traceable=traceable) if annotate else []
        ),
    }


def plan_vs_evidence(
    task: TaskSpec,
    report: Mapping,
    *,
    carry_subtasks: Sequence[Mapping] = (),
    transitions: Sequence[Mapping] = (),
) -> dict:
    """The runtime's sub-task statuses against the evidence-based stage verdicts.

    The executor advances the plan from the *signals* a tool result declares
    (``progress_kind == task``, a satisfied/arrived/verified flag) and charges the
    call to the sub-task that is currently active; a stage is complete when the
    drone reached its place *and* a tool result located it. The two can disagree -
    a search that found the *next* stage's object still closes the open sub-task -
    and this is where the report says so instead of assuming either one.
    """
    statuses = {
        str(item.get("id")): str(item.get("status"))
        for item in carry_subtasks or [] if isinstance(item, Mapping)
    }
    for entry in transitions or ():
        if not isinstance(entry, Mapping):
            continue
        stage_id = str(entry.get("id") or "")
        if stage_id:
            statuses[stage_id] = str(entry.get("to") or statuses.get(stage_id, ""))
    rows: list[dict] = []
    for stage in task.stages:
        subtask_id = f"st_{stage.index}"
        status = statuses.get(subtask_id)
        complete = bool((report.get("stages") or {}).get(stage.stage_id, {}).get("complete"))
        rows.append(
            {
                "stage": stage.stage_id,
                "subtask": subtask_id,
                "status": status,
                "evidence_complete": complete,
                "agree": None if not status else ((status == "done") == complete),
            }
        )
    disagreements = [row for row in rows if row["agree"] is False]
    return {"rows": rows, "disagreements": disagreements, "agree": not disagreements}


def plan_vs_evidence_by_leg(
    task: TaskSpec,
    legs: Sequence[LegResult],
) -> dict:
    """One plan-vs-evidence verdict per leg, not only per interrupted episode.

    A leg's sub-task statuses are the ones it inherited (``plan_subtasks`` on a
    resumed leg's carry) updated by its own ``plan_transitions``; the evidence is
    the cumulative tool-call ledger of the chain that leg continues (a ``restart``
    arm is judged on its own). A stage whose status the log never reports is left
    ``None`` and is not counted as either agreement or disagreement - the check
    says how many legs it really compared.

    This is the P4 ``10/10`` check: every leg the run produced is checked, so an
    intermittent bookkeeping error cannot hide in a leg that was not interrupted.
    """
    rows: list[dict] = []
    chain_entries: list[dict] = []
    leg_rows: list[dict] = []
    for leg in legs:
        leg_entries: list[dict] = []
        for entry in leg.entries:
            item = {**entry, "leg": leg.plan.label, "global_seq": len(chain_entries) + 1}
            chain_entries.append(item)
            leg_entries.append(item)
        restart = leg.plan.role == "restart"
        evidence_entries = leg_entries if restart else chain_entries
        statuses = {
            str(item.get("id")): str(item.get("status"))
            for item in (leg.plan.carry.get("plan_subtasks") or [])
            if isinstance(item, Mapping) and item.get("id")
        }
        for transition in leg.log.get("plan_transitions") or []:
            if not isinstance(transition, Mapping):
                continue
            subtask_id = str(transition.get("id") or "")
            if subtask_id:
                statuses[subtask_id] = str(
                    transition.get("to") or statuses.get(subtask_id, "")
                )
        report = stage_report(evidence_entries, task)
        leg_disagreements: list[dict] = []
        compared = 0
        for stage in task.stages:
            subtask_id = f"st_{stage.index}"
            status = statuses.get(subtask_id)
            complete = bool(
                (report.get("stages") or {}).get(stage.stage_id, {}).get("complete")
            )
            agree = None if not status else ((status == "done") == complete)
            if agree is not None:
                compared += 1
            row = {
                "leg": leg.plan.label,
                "role": leg.plan.role,
                "stage": stage.stage_id,
                "subtask": subtask_id,
                "status": status,
                "evidence_complete": complete,
                "agree": agree,
            }
            rows.append(row)
            if agree is False:
                leg_disagreements.append(row)
        leg_rows.append(
            {
                "leg": leg.plan.label,
                "role": leg.plan.role,
                "compared": compared,
                "disagreements": len(leg_disagreements),
                "agree": not leg_disagreements,
            }
        )
    disagreements = [row for row in rows if row["agree"] is False]
    # not every disagreement is equally bad: a ``done`` whose evidence is not
    # complete is a false completion claim (the failure P4/A fixed); a
    # ``claimed_done``/``pending`` whose evidence *is* complete is only a
    # conservative lag - the plan has not been re-marked yet.
    unsafe_done = [
        row for row in rows if row["status"] == "done" and not row["evidence_complete"]
    ]
    lagging = [
        row for row in rows
        if row["status"] in ("claimed_done", "active", "pending")
        and row["evidence_complete"]
    ]
    return {
        "checked": len(legs),
        "compared_legs": sum(1 for row in leg_rows if row["compared"]),
        "agreed_legs": sum(1 for row in leg_rows if row["agree"]),
        "legs": leg_rows,
        "rows": rows,
        "disagreements": disagreements,
        "unsafe_done_claims": unsafe_done,
        "lagging_statuses": lagging,
        "ok": not disagreements,
    }


def episode_metrics(
    task: TaskSpec,
    legs: Sequence[LegResult],
    *,
    gate_m: float | None,
) -> dict:
    """One episode: the chain under test plus the control arms it ran.

    The first leg is the fresh start. The follow-up legs are the arms of the
    interrupted episode: the ``resume`` arm (or the ``control`` arm when the run
    is the no-carry one) is what the episode *does*, and any further arm
    (``restart``) is judged on its own - a leg that replays the task from the
    original pose is a comparison, not a continuation.
    """
    first = list(legs[:1])
    primary = list(first)
    arms: list[dict] = []
    primary_taken = False
    for leg in legs[1:]:
        role = leg.plan.role
        if role == "restart":
            arm_legs = [leg]
        else:
            arm_legs = first + [leg]
        verdict = combined_verdict(task, arm_legs, gate_m=gate_m, annotate=False)
        arms.append(
            {
                "role": role,
                "label": leg.plan.label,
                "carry": dict(leg.plan.carry),
                "stages_complete": verdict["stages"]["complete_count"],
                "chain_complete": verdict["stages"]["chain_complete"],
                "success": verdict["success"],
                "traceable_success": verdict["traceable_success"],
                "verify_ok": verdict["verify"]["ok"],
                "final_distance_m": verdict["final_distance_m"],
            }
        )
        if role in ("resume", "control") and not primary_taken:
            primary = arm_legs
            primary_taken = True
    verdict = combined_verdict(task, primary, gate_m=gate_m)
    entries = verdict.pop("entries")
    leg_rows = []
    rows_by_label: dict[str, dict] = {}
    for leg in legs:
        metrics = leg_metrics(leg, trace=read_trace(leg.out_dir))
        metrics["label"] = leg.plan.label
        metrics["leg_label"] = leg.plan.label
        metrics["role"] = leg.plan.role
        metrics["plan"] = leg.plan.to_dict()
        metrics["carry"] = leg.plan.carry
        metrics["interrupted"] = bool(leg.interrupted)
        metrics["interrupt_reason"] = leg.interrupt_reason
        leg_rows.append(metrics)
        rows_by_label[leg.plan.label] = metrics
    verdict["legs"] = leg_rows
    verdict["arms"] = arms
    verdict["n_arms"] = len(arms)
    follow_ups = list(primary[1:])
    if follow_ups:
        follow = follow_ups[-1]
        verdict["plan_vs_evidence"] = plan_vs_evidence(
            task,
            verdict["stages"],
            carry_subtasks=follow.plan.carry.get("plan_subtasks") or [],
            transitions=rows_by_label.get(follow.plan.label, {}).get("plan_transitions") or [],
        )
    # the same question for *every* leg, so an uninterrupted leg cannot hide an
    # executor/evidence disagreement behind a report that only checked resumes
    verdict["plan_vs_evidence_by_leg"] = plan_vs_evidence_by_leg(task, legs)
    return verdict


def done_stage_recheck(
    entries: Sequence[dict], task: TaskSpec, interrupted_leg: LegResult | None
) -> int:
    """Information calls of a follow-up leg inside a stage it already established.

    The resumed episode legitimately sits inside the stage it confirmed, so a
    re-observation there is not by itself a repeat - it becomes one when an
    information call is made from inside a stage that was already complete at the
    interrupt *and* the call bought no evidence the interrupted leg had not
    already produced.
    """
    if interrupted_leg is None:
        return 0
    label = interrupted_leg.plan.label
    prefix = [entry for entry in entries if entry.get("leg") == label]
    done = [
        stage for stage in task.stages
        if (stage_report(prefix, task)["stages"][stage.stage_id])["complete"]
    ]
    if not done:
        return 0
    seen: set[str] = set()
    for entry in prefix:
        for ref in entry.get("evidence") or []:
            seen.add(f"{ref.get('kind')}:{ref.get('ref')}")
    count = 0
    for entry in entries:
        if entry.get("leg") == label or not entry.get("outer"):
            continue
        if entry.get("tool") not in INFO_TOOLS:
            continue
        pose = _as_position(entry.get("pose_before")) or _as_position(entry.get("pose_after"))
        if pose is None:
            continue
        inside = any(stage.position is not None and _distance(pose, stage.position) <= stage.reach_m for stage in done)
        if not inside:
            continue
        fresh = [
            ref for ref in entry.get("evidence") or []
            if f"{ref.get('kind')}:{ref.get('ref')}" not in seen
        ]
        if not fresh:
            count += 1
    return count


def annotations(
    task: TaskSpec,
    legs: Sequence[LegResult],
    report: Mapping,
    verify: Mapping,
    *,
    traceable: bool | None = None,
) -> list[str]:
    """The causal chain of the episode, in plain statements of what happened.

    Each line is anchored on a tool result (which call, which evidence, which
    place); the point is to say *why* the episode moved between stages and whether
    the failure branches of the loop were walked at all.
    """
    lines: list[str] = []
    stages = report.get("stages") or {}
    current_leg = legs[0].plan.label if legs else ""
    for stage in task.stages:
        row = stages.get(stage.stage_id) or {}
        if row.get("complete"):
            lines.append(
                f"stage {stage.index} ({stage.label}) complete: reached {row.get('reach_distance_m')} m "
                f"from its place and located it {row.get('locate_distance_m')} m off "
                f"via {row.get('locate_tool') or 'a tool result'} at call {row.get('located_seq')} "
                f"with {row.get('evidence_count')} evidence ref(s)"
            )
        else:
            missing = ", ".join(
                name for name, flag in (("reach", row.get("reach")), ("locate", row.get("located")))
                if not flag
            )
            lines.append(f"stage {stage.index} ({stage.label}) NOT complete ({missing or 'unevaluable'})")
        if row.get("relation_ok") is not None:
            lines.append(
                f"  stage {stage.index} relation to {stage.anchor}: "
                f"{row.get('relation_distance_m')} m "
                f"({'consistent' if row['relation_ok'] else 'inconsistent'} with within_m="
                f"{stage.within_m})"
            )
    for leg in legs:
        trace = read_trace(leg.out_dir)
        counts: dict[str, int] = {}
        for entry in trace:
            kind = str(entry.get("progress_kind") or "")
            if kind:
                counts[kind] = counts.get(kind, 0) + 1
        failed = [
            entry for entry in leg.entries
            if entry["outer"] and entry["status"] in ("failed", "blocked", "empty")
        ]
        recoveries = [entry for entry in trace if entry.get("recovery")]
        lines.append(
            f"{leg.plan.label}: {len(leg.entries)} tool calls (outer "
            f"{sum(1 for e in leg.entries if e['outer'])}, inner "
            f"{sum(1 for e in leg.entries if not e['outer'])}), "
            f"{leg.executor['legs']} executor legs / {leg.executor['ticks']} control steps, "
            f"{leg.vlm_calls} model calls, progress kinds {counts or 'n/a'}"
        )
        if failed:
            worst = ", ".join(f"{entry['tool']}:{entry['status']}" for entry in failed[:4])
            lines.append(f"  failure branch walked: {len(failed)} call(s) ({worst}) - the loop had to re-plan")
        else:
            lines.append("  failure branch not walked: every call returned ok/empty-free")
        if recoveries:
            lines.append(f"  recovery advice fired on {len(recoveries)} turn(s): {recoveries[0].get('recovery')}")
        if leg.interrupted:
            lines.append(
                f"  interrupted ({leg.interrupt_reason}); the next leg inherits pose + plan + subgoal + brief"
            )
        if leg.plan.carry:
            lines.append(
                f"  inherited carry: {leg.plan.carry.get('mode')} "
                f"(stages done at interrupt: {leg.plan.carry.get('done')})"
            )
    target = next((stage for stage in task.stages if stage.target), None)
    if target is not None:
        ok = "verified" if verify.get("ok") else "NOT verified"
        lines.append(
            f"final target ({target.label}) {ok}: verify_goal_verified="
            f"{bool(verify.get('verify_goal_verified'))}, in-gate references={len(verify.get('in_gate') or [])}"
        )
    if traceable is None:
        traceable = bool(report.get("chain_complete")) and bool(verify.get("ok"))
    # the annotation must use the driver's own verdict (order and evidence
    # included), not a weaker chain+verify shortcut: seed 7 completes all three
    # stages out of order and is a chain but *not* a traceable success
    lines.append(f"traceable success: {bool(traceable)}")
    return lines


# ---------------------------------------------------------------------------
# aggregation and report
# ---------------------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 3) if denominator else None


def _kind_totals(leg_rows: Sequence[Mapping], key: str) -> dict:
    totals: dict[str, int] = {}
    for leg in leg_rows:
        for name, count in (leg.get(key) or {}).items():
            totals[str(name)] = totals.get(str(name), 0) + int(count)
    return totals


def aggregate(episodes: Sequence[Mapping], task: TaskSpec, meta: Mapping) -> dict:
    """Rates over episodes: stages, interrupts, decisions/executor, repeats, cost."""
    n = len(episodes)
    stage_rows: list[dict] = []
    for stage in task.stages:
        reach = sum(1 for row in episodes if (row["stages"]["stages"].get(stage.stage_id) or {}).get("reach"))
        located = sum(1 for row in episodes if (row["stages"]["stages"].get(stage.stage_id) or {}).get("located"))
        complete = sum(1 for row in episodes if (row["stages"]["stages"].get(stage.stage_id) or {}).get("complete"))
        stage_rows.append(
            {
                "id": stage.stage_id,
                "index": stage.index,
                "label": stage.label,
                "n": n,
                "reach": reach,
                "located": located,
                "complete": complete,
                "reach_rate": _rate(reach, n),
                "locate_rate": _rate(located, n),
                "complete_rate": _rate(complete, n),
            }
        )
    interrupted = [row for row in episodes if row["interrupted"]]
    recovered = [row for row in interrupted if row["interrupt_recovered"]]
    gains = [row["stages_gained_on_resume"] for row in interrupted]
    leg_rows = [leg for row in episodes for leg in row["legs"]]
    decision_turns = sum(leg["decision_turns"] for leg in leg_rows)
    model_calls = sum(leg["model_calls"] for leg in leg_rows)
    executor_legs = sum(leg["executor"]["legs"] for leg in leg_rows)
    executor_ticks = sum(leg["executor"]["ticks"] for leg in leg_rows)
    info_calls = sum(leg["info_calls"] for leg in leg_rows)
    repeat_checks = sum(leg["repeat_checks"] for leg in leg_rows)
    repeat_views = sum(leg["repeat_views"] for leg in leg_rows)
    wall_time = sum(leg["wall_time_s"] for leg in leg_rows)
    by_role: dict[str, dict] = {}
    for leg in leg_rows:
        row = by_role.setdefault(
            str(leg.get("role") or "fresh"),
            {"legs": 0, "model_calls": 0, "tool_calls": 0, "executor_legs": 0,
             "executor_ticks": 0, "wall_time_s": 0.0, "decision_nodes": 0},
        )
        row["legs"] += 1
        row["model_calls"] += int(leg["model_calls"])
        row["tool_calls"] += int(leg["outer_tool_calls"]) + int(leg["inner_tool_calls"])
        row["executor_legs"] += int(leg["executor"]["legs"])
        row["executor_ticks"] += int(leg["executor"]["ticks"])
        row["wall_time_s"] = round(row["wall_time_s"] + float(leg["wall_time_s"]), 3)
        row["decision_nodes"] += int(leg["decision_nodes"])
    return {
        "n_episodes": n,
        "n_legs": len(leg_rows),
        "by_role": by_role,
        "stages": stage_rows,
        "stage_reach_rate": _rate(
            sum(row["reach"] for row in stage_rows), n * len(task.stages)
        ),
        "stage_complete_rate": _rate(
            sum(row["complete"] for row in stage_rows), n * len(task.stages)
        ),
        "chain_complete_rate": _rate(sum(1 for row in episodes if row["stages"]["chain_complete"]), n),
        "order_ok_rate": _rate(sum(1 for row in episodes if row["stages"]["order_ok"]), n),
        "success_rate": _rate(sum(1 for row in episodes if row["success"]), n),
        "traceable_success_rate": _rate(sum(1 for row in episodes if row["traceable_success"]), n),
        "evidence_complete_rate": _rate(sum(1 for row in episodes if row["evidence_complete"]), n),
        "verify_rate": _rate(sum(1 for row in episodes if row["verify"]["ok"]), n),
        "interrupt": {
            "n_episodes": len(interrupted),
            "recovered": len(recovered),
            "recovery_rate": _rate(len(recovered), len(interrupted)),
            "no_progress_on_resume": len(interrupted) - len(recovered),
            "mean_stages_gained_on_resume": round(sum(gains) / len(gains), 3) if gains else None,
            "done_stage_recheck_total": sum(row["done_stage_recheck"] for row in interrupted),
            "reasons": [
                {"label": row["legs"][0]["label"], "reason": row["legs"][0]["interrupt_reason"]}
                for row in interrupted
            ],
        },
        "decision_vs_executor": {
            "decision_turns": decision_turns,
            "model_calls": model_calls,
            "decision_nodes": sum(leg["decision_nodes"] for leg in leg_rows),
            "routine_turns": sum(leg["routine_turns"] for leg in leg_rows),
            "decision_node_kinds": _kind_totals(leg_rows, "decision_node_kinds"),
            "subtask_switches": sum(leg["subtask_switches"] for leg in leg_rows),
            "plan_transitions": sum(leg["n_plan_transitions"] for leg in leg_rows),
            "inner_tool_calls": sum(leg["inner_tool_calls"] for leg in leg_rows),
            "outer_tool_calls": sum(leg["outer_tool_calls"] for leg in leg_rows),
            "executor_legs": executor_legs,
            "executor_ticks": executor_ticks,
            "executor_legs_per_decision": round(executor_legs / decision_turns, 3) if decision_turns else None,
            "ticks_per_model_call": round(executor_ticks / model_calls, 3) if model_calls else None,
            # P4-B: deterministic steps the executor ran itself (they cost a tool
            # call but no model turn), and the turns they saved
            "executor_auto_steps": sum(int(leg.get("executor_auto_steps") or 0) for leg in leg_rows),
            "vlm_turns_saved": sum(int(leg.get("vlm_turns_saved") or 0) for leg in leg_rows),
            "model_turns": sum(int(leg.get("model_turns") or 0) for leg in leg_rows),
        },
        "repeat_search": {
            "info_calls": info_calls,
            "repeat_checks": repeat_checks,
            "repeat_views": repeat_views,
            "repeat_search_ratio": round(repeat_checks / info_calls, 3) if info_calls else None,
        },
        "cost": {
            "wall_time_s": round(wall_time, 3),
            "wall_time_per_episode_s": round(wall_time / n, 3) if n else None,
            "wall_time_per_leg_s": round(wall_time / len(leg_rows), 3) if leg_rows else None,
            "model_calls": model_calls,
            "model_calls_per_episode": round(model_calls / n, 3) if n else None,
            "tool_calls": sum(leg["outer_tool_calls"] + leg["inner_tool_calls"] for leg in leg_rows),
            "evidence_refs": sum(leg["evidence_refs"] for leg in leg_rows),
        },
        "meta": dict(meta),
    }


def protocol_checks(
    episodes: Sequence[Mapping], task: TaskSpec, *, ledger_vs_trace: Sequence[Mapping]
) -> dict:
    """Cross-checks that keep the report honest (each one is pass/fail)."""
    checks: dict[str, Any] = {}
    steps = [stage.step for stage in task.stages]
    plan_rows = [
        leg["carry"].get("plan_steps")
        for row in episodes
        for leg in row["legs"]
        if leg["carry"].get("plan_steps")
    ]
    checks["carry_plan_steps"] = {
        "checked": len(plan_rows),
        "match_task": all(list(row) == steps for row in plan_rows),
    }
    # Every place the brief names must be one the interrupted leg's own tool
    # results (or the ledger's poses) reported: the leak would be a place the
    # experiment knows but the episode never established.
    brief_rows: list[tuple[str, str]] = []
    untraced: list[dict] = []
    for row in episodes:
        for leg in row["legs"]:
            if leg["carry"].get("mode") != "resume":
                continue
            brief_rows.append((str(row.get("label")), str(leg["label"])))
            traced: list[list[float]] = []
            for other in row["legs"]:
                if other["label"] == leg["label"]:
                    break
                traced += [list(map(float, p)) for p in (other.get("reported_places") or [])]
            for place in leg["carry"].get("brief_places") or []:
                point = _as_position(place)
                if point is None:
                    continue
                if not any(_distance(point, candidate) < 1e-6 for candidate in traced):
                    untraced.append({"leg": leg["label"], "place": list(point)})
    checks["brief_places_are_traced"] = {
        "checked_legs": len(brief_rows),
        "untraced": untraced,
        "ok": not untraced,
    }
    matched = []
    for row in ledger_vs_trace:
        matched.append(
            {
                "label": row["label"],
                "ledger_outer": row["ledger_outer"],
                "trace_calls": row["trace_calls"],
                "match": None if row["trace_calls"] is None else row["ledger_outer"] == row["trace_calls"],
            }
        )
    checks["ledger_vs_trace"] = {"rows": matched, "all_match": all(r["match"] is not False for r in matched)}
    disagreements = [
        {"episode": row.get("label"), "stage": item["stage"], "status": item["status"]}
        for row in episodes
        for item in ((row.get("plan_vs_evidence") or {}).get("disagreements") or [])
    ]
    checks["plan_vs_evidence"] = {
        "episodes_checked": sum(1 for row in episodes if row.get("plan_vs_evidence")),
        "disagreements": disagreements,
    }
    leg_checks = [
        (row.get("label"), leg)
        for row in episodes
        for leg in ((row.get("plan_vs_evidence_by_leg") or {}).get("legs") or [])
    ]
    checks["plan_vs_evidence_legs"] = {
        "legs_checked": len(leg_checks),
        "legs_compared": sum(1 for _label, leg in leg_checks if leg.get("compared")),
        "agreed_legs": sum(1 for _label, leg in leg_checks if leg.get("agree")),
        "disagreements": [
            {"episode": label, **item}
            for row in episodes
            for item in ((row.get("plan_vs_evidence_by_leg") or {}).get("disagreements") or [])
            for label in [row.get("label")]
        ],
        "unsafe_done_claims": [
            {"episode": label, **item}
            for row in episodes
            for item in ((row.get("plan_vs_evidence_by_leg") or {}).get("unsafe_done_claims") or [])
            for label in [row.get("label")]
        ],
        "lagging_statuses": [
            {"episode": label, **item}
            for row in episodes
            for item in ((row.get("plan_vs_evidence_by_leg") or {}).get("lagging_statuses") or [])
            for label in [row.get("label")]
        ],
        "all_agree": all(leg.get("agree") for _label, leg in leg_checks),
    }
    completion_rows = [
        {
            "label": leg.get("label"),
            "runner": (leg.get("runner_completion") or {}).get("verified"),
            "driver": leg.get("driver_completion_verified"),
        }
        for row in episodes
        for leg in row.get("legs", [])
        if leg.get("runner_completion")
    ]
    checks["completion_chain_agreement"] = {
        "checked": len(completion_rows),
        "rows": [
            item for item in completion_rows if item["runner"] != item["driver"]
        ],
        "all_match": all(item["runner"] == item["driver"] for item in completion_rows),
    }
    kinds = [
        kind
        for row in episodes
        for leg in row["legs"]
        for kind in (leg.get("progress_kinds") or {})
    ]
    checks["progress_kinds_known"] = {
        "seen": sorted(set(kinds)),
        "all_known": all(kind in PROGRESS_KINDS for kind in kinds),
    }
    return checks


def report_markdown(report: Mapping) -> str:
    """The human-readable report: metrics tables plus the per-episode chain."""
    meta = report.get("meta") or {}
    agg = report.get("aggregate") or {}
    lines = [
        f"# Long-horizon verification: {meta.get('task_name')}",
        "",
        f"task file: `{meta.get('task_file')}` - {meta.get('n_episodes')} episode(s), "
        f"{agg.get('n_legs')} leg(s), backend `{meta.get('backend')}`, world `{meta.get('world')}`",
        "",
        "## task",
        "",
        f"instruction: {meta.get('instruction')}",
        "",
        "| # | stage | kind | place (flu) | reach<=m | locate<=m | anchor<=m |",
        "|---|---|---|---|---|---|---|",
    ]
    for stage in report.get("task", {}).get("stages", []):
        place = stage.get("position_flu")
        lines.append(
            "| {index} | {label} | {kind} | {place} | {reach} | {locate} | {anchor} |".format(
                index=stage.get("index"),
                label=stage.get("label"),
                kind=stage.get("kind"),
                place="n/a" if place is None else ", ".join(f"{v:.2f}" for v in place),
                reach=stage.get("reach_m"),
                locate=stage.get("locate_m"),
                anchor="-" if stage.get("within_m") is None else f"{stage['anchor']} <= {stage['within_m']}",
            )
        )
    chain = (report.get("meta") or {}).get("chain_check") or {}
    if not chain:
        chain_status = "n/a (re-analysis without a scene check)"
    elif chain.get("ok"):
        chain_status = "ok"
    else:
        chain_status = "PROBLEMS (" + "; ".join(chain.get("problems") or []) + ")"
    lines += [
        "",
        f"scene chain check: {chain_status}",
        "",
        "## stages (per episode share)",
        "",
        "| # | stage | reach | located | complete |",
        "|---|---|---|---|---|",
    ]
    for row in agg.get("stages", []):
        lines.append(
            f"| {row['index']} | {row['label']} | {row['reach']}/{row['n']} "
            f"({row['reach_rate']}) | {row['located']}/{row['n']} ({row['locate_rate']}) | "
            f"{row['complete']}/{row['n']} ({row['complete_rate']}) |"
        )
    intr = agg.get("interrupt") or {}
    lines += [
        "",
        "## episodes",
        "",
        "| episode | seed | legs (roles) | stages (reach/locate/complete) | chain | success | traceable | interrupted | recovered |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in report.get("episodes", []):
        stages = row.get("stages", {})
        legs = row.get("legs", [])
        roles = "+".join(str(leg.get("role")) for leg in legs) or "n/a"
        lines.append(
            "| {label} | {seed} | {legs} ({roles}) | {reach}/{locate}/{complete} | {chain} | {success} | {traceable} | {intr} | {rec} |".format(
                label=row.get("label") or (legs[0]["label"] if legs else "n/a"),
                seed=row.get("seed", "-"),
                legs=row.get("n_legs"),
                roles=roles,
                reach=stages.get("reach_count"),
                locate=stages.get("locate_count"),
                complete=stages.get("complete_count"),
                chain=stages.get("chain_complete"),
                success=row.get("success"),
                traceable=row.get("traceable_success"),
                intr=row.get("interrupted"),
                rec=row.get("interrupt_recovered"),
            )
        )
        for arm in row.get("arms") or []:
            lines.append(
                f"|   arm {arm.get('role')} | {arm.get('label')} | "
                f"{arm.get('stages_complete')} complete | {arm.get('chain_complete')} | "
                f"{arm.get('success')} | {arm.get('traceable_success')} | - | - |"
            )
    lines += [
        "",
        "## interrupt / resume",
        "",
        f"- interrupted episodes: {intr.get('n_episodes', 0)} - recovered (gained a stage on resume): "
        f"{intr.get('recovered', 0)} ({intr.get('recovery_rate')})",
        f"- mean stages gained on resume: {intr.get('mean_stages_gained_on_resume')}",
        f"- information calls in the resumed leg from an already-established stage with no new evidence: "
        f"{intr.get('done_stage_recheck_total')}",
        "",
        "## decision nodes vs executor",
        "",
        "| decision turns | decision nodes | routine turns | sub-task switches | model calls | executor legs | "
        "executor auto steps | saved VLM turns | control steps | legs/decision | steps/model call |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        "| {decision_turns} | {decision_nodes} | {routine_turns} | {subtask_switches} | {model_calls} | "
        "{executor_legs} | {executor_auto_steps} | {vlm_turns_saved} | {executor_ticks} | "
        "{executor_legs_per_decision} | "
        "{ticks_per_model_call} |".format(**(agg.get("decision_vs_executor") or {})),
        "",
        "decision node kinds: "
        + json.dumps((agg.get("decision_vs_executor") or {}).get("decision_node_kinds") or {}),
        "",
        "_executor auto steps_ are calls the executor issued itself from the plan "
        "(no model turn); _saved VLM turns_ is what a per-step scheduler would "
        "have spent on them.",
        "",
        "## repeat search",
        "",
        f"information calls {agg.get('repeat_search', {}).get('info_calls')}, repeated checks "
        f"{agg.get('repeat_search', {}).get('repeat_checks')} (ratio "
        f"{agg.get('repeat_search', {}).get('repeat_search_ratio')}), repeated views "
        f"{agg.get('repeat_search', {}).get('repeat_views')}",
        "",
        "## cost",
        "",
        f"wall time {agg.get('cost', {}).get('wall_time_s')} s "
        f"({agg.get('cost', {}).get('wall_time_per_episode_s')} s/episode), model calls "
        f"{agg.get('cost', {}).get('model_calls')}, tool calls {agg.get('cost', {}).get('tool_calls')}, "
        f"evidence refs {agg.get('cost', {}).get('evidence_refs')}",
        "",
        "by leg role (the control arms are real GPU work - this is where it went):",
        "",
        "| role | legs | model calls | tool calls | executor legs | control steps | wall time (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for role, row in sorted((agg.get("by_role") or {}).items()):
        lines.append(
            f"| {role} | {row['legs']} | {row['model_calls']} | {row['tool_calls']} | "
            f"{row['executor_legs']} | {row['executor_ticks']} | {row['wall_time_s']} |"
        )
    lines += [
        "",
        "## checks",
        "",
    ]
    for name, value in (report.get("checks") or {}).items():
        lines.append(f"- `{name}`: {json.dumps(value, default=str)}")
    lines += ["", "## causal annotations", ""]
    for row in report.get("episodes", []):
        legs = row.get("legs", [])
        label = legs[0]["label"] if legs else row.get("label")
        lines.append(f"### {row.get('label')}")
        plan_check = row.get("plan_vs_evidence") or {}
        if plan_check:
            lines.append(
                "- plan vs evidence: "
                + ("sub-task statuses agree with the stage verdicts"
                   if plan_check.get("agree")
                   else "DISAGREEMENT "
                        + json.dumps(plan_check.get("disagreements"), default=str))
            )
        for line in row.get("annotations") or []:
            lines.append(f"- {line}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# offline re-analysis of a finished run
# ---------------------------------------------------------------------------


def load_leg(out_dir: Path) -> LegResult | None:
    """Rebuild one leg's :class:`LegResult` from its artifacts on disk.

    Everything the metrics need is in the leg's own files: the ledger (calls,
    poses, executor counters), ``leg_summary.json`` (interrupt/executor/summary),
    ``run_config.json`` (the leg plan), ``summary.json`` (the runner's verdict)
    and ``episode_log.json`` (the runtime's decision nodes and plan transitions).
    Re-analysing a finished run therefore needs no world, no model and no GPU.
    """
    out_dir = Path(out_dir)
    ledger_path = out_dir / "ledger.jsonl"
    config_path = out_dir / "run_config.json"
    if not ledger_path.exists() or not config_path.exists():
        return None
    entries = [
        json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()
    ]
    config = json.loads(config_path.read_text())
    summary = None
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    leg_summary: dict = {}
    leg_summary_path = out_dir / "leg_summary.json"
    if leg_summary_path.exists():
        leg_summary = json.loads(leg_summary_path.read_text())
    log: dict = {}
    log_path = out_dir / "episode_log.json"
    if log_path.exists():
        log = json.loads(log_path.read_text())
        log.setdefault("turns_count", len(log.get("turn_decisions") or []))
    plan_data = dict(config.get("leg") or {})
    plan = LegPlan(
        label=str(plan_data.get("label") or out_dir.name),
        instruction=str(plan_data.get("instruction") or ""),
        prompt_episode_id=str(plan_data.get("prompt_episode_id") or ""),
        tool_budget=int(plan_data.get("tool_budget") or 0),
        max_turns=int(plan_data.get("max_turns") or 0),
        deadline_s=plan_data.get("deadline_s"),
        interrupt_mode=str(plan_data.get("interrupt_mode") or ""),
        role=str(plan_data.get("role") or "fresh"),
        teleport=None,
        ctx_extra={},
        carry=dict(plan_data.get("carry") or {}),
        mock_script=plan_data.get("mock_script"),
    )
    executor = dict(leg_summary.get("executor") or {})
    return LegResult(
        plan=plan,
        out_dir=out_dir,
        summary=summary,
        interrupted=bool(leg_summary.get("interrupted")),
        interrupt_reason=str(leg_summary.get("interrupt_reason") or ""),
        error=leg_summary.get("error"),
        entries=entries,
        executor=executor,
        vlm_calls=int(leg_summary.get("vlm_calls") or leg_summary.get("vlm_calls_wrapper") or 0),
        wall_time_s=float(leg_summary.get("wall_time_s") or 0.0),
        start_pose=[float(v) for v in (leg_summary.get("start_pose") or [])],
        log=log,
    )


def analyze_run(run_dir: Path, *, write: bool = True, gate_m: float | None = None) -> dict:
    """Re-judge a finished run from its artifacts (the same driver functions).

    The point is that a metric or a check added after a GPU run can be applied to
    the *same* episodes without re-running the model: the artifacts carry every
    call, every pose and the runtime's own log. Writes
    ``report_reanalyzed.json`` / ``report_reanalyzed.md`` next to the run.
    """
    run_dir = Path(run_dir)
    task_path = run_dir / "task.json"
    if not task_path.exists():
        raise SystemExit(f"--analyze {run_dir}: no task.json in the run directory")
    task = TaskSpec(
        name=str(json.loads(task_path.read_text()).get("name") or run_dir.name),
        instruction=str(json.loads(task_path.read_text()).get("instruction") or ""),
        stages=load_task(task_path).stages,
        scene=dict(json.loads(task_path.read_text()).get("scene") or {}),
        path=task_path,
    )
    gate = gate_m
    backend = world_name = model_path = None
    first_config = next(iter(sorted(run_dir.glob("seed*/seed*_leg*/run_config.json"))), None)
    if first_config is not None:
        argv = json.loads(first_config.read_text()).get("runner_argv") or []
        for flag, key in (("--backend", "backend"), ("--world", "world"),
                          ("--model-path", "model_path")):
            if flag in argv and argv.index(flag) + 1 < len(argv):
                value = str(argv[argv.index(flag) + 1])
                if key == "backend":
                    backend = value
                elif key == "world":
                    world_name = value
                else:
                    model_path = value
    episodes = []
    checks_rows: list[dict] = []
    for seed_dir in sorted(run_dir.glob("seed*")):
        legs = [
            leg for leg in (load_leg(path) for path in sorted(seed_dir.glob("seed*_leg*")))
            if leg is not None
        ]
        if not legs:
            continue
        if gate is None:
            gate = legs[0].summary.get("success_dist_m") if isinstance(legs[0].summary, dict) else None
        checks_rows += [ledger_vs_trace_row(leg) for leg in legs]
        metrics = episode_metrics(task, legs, gate_m=gate)
        metrics["seed"] = int(str(seed_dir.name).replace("seed", "") or 0)
        episodes.append(metrics)
    meta = {
        "task_name": task.name,
        "task_file": str(task_path),
        "instruction": task.instruction,
        "n_episodes": len(episodes),
        "success_gate_m": gate,
        "backend": backend,
        "world": world_name,
        "model_path": model_path,
        "chain_check": chain_check(task),
        "reanalyzed": True,
        "source_run": str(run_dir),
        "carry_support": carry_support(),
    }
    report = {
        "meta": meta,
        "task": task.to_dict(),
        "aggregate": aggregate(episodes, task, meta),
        "checks": protocol_checks(episodes, task, ledger_vs_trace=checks_rows),
        "episodes": episodes,
    }
    if write:
        with open(run_dir / "report_reanalyzed.json", "w") as handle:
            json.dump(jsonable(report), handle, indent=2)
        with open(run_dir / "report_reanalyzed.md", "w") as handle:
            handle.write(report_markdown(report))
        print(f"[long_horizon] re-analyzed: {run_dir / 'report_reanalyzed.json'}", flush=True)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_interrupt_spec(value: str) -> tuple[int, str, float | None]:
    """``SEED:MODE[:VALUE]`` -> ``(seed, mode, value)`` (see the module docstring)."""
    parts = [part.strip() for part in str(value).split(":")]
    if len(parts) < 2:
        raise SystemExit(f"--interrupt {value!r} must be SEED:MODE[:VALUE]")
    try:
        seed = int(parts[0])
    except ValueError:
        raise SystemExit(f"--interrupt {value!r}: seed must be an integer")
    mode = parts[1].lower()
    if mode not in ("budget", "deadline"):
        raise SystemExit(f"--interrupt {value!r}: mode must be budget or deadline")
    raw = parts[2] if len(parts) > 2 and parts[2] != "" else None
    if raw is None:
        return seed, mode, None
    try:
        number = float(raw)
    except ValueError:
        raise SystemExit(f"--interrupt {value!r}: value must be a number")
    if number <= 0:
        raise SystemExit(f"--interrupt {value!r}: value must be positive")
    return seed, mode, number


def interrupt_map(values: Sequence[str]) -> dict[int, tuple[str, float | None]]:
    specs: dict[int, tuple[str, float | None]] = {}
    for value in values or ():
        seed, mode, number = parse_interrupt_spec(value)
        specs[seed] = (mode, number)
    return specs


def parse_seeds(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        items: Iterable[Any] = value
    else:
        items = str(value).replace(",", " ").split()
    seeds: list[int] = []
    for item in items:
        try:
            seed = int(item)
        except (TypeError, ValueError):
            raise SystemExit(f"--seeds {value!r}: {item!r} is not an integer")
        if seed not in seeds:
            seeds.append(seed)
    return seeds or [0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Long-horizon multi-stage verification driver (P4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--analyze", type=str, default=None,
                        help="re-judge a finished run directory from its artifacts "
                             "(no world, no model, no GPU) and exit")
    parser.add_argument("--task-file", type=str, default=None,
                        help="multi-stage task JSON (see the module docstring)")
    parser.add_argument("--out", type=str, default=None,
                        help="output directory of the run (with --analyze: the run to read)")
    parser.add_argument("--seeds", type=str, default="7")
    parser.add_argument("--instruction", type=str, default=None,
                        help="override the task file's instruction")
    parser.add_argument("--interrupt", type=str, action="append", default=[],
                        help="SEED:MODE[:VALUE] per seed; MODE=budget|deadline; repeatable")
    parser.add_argument("--resume", type=str, default="on", choices=["on", "off"],
                        help="carry the plan/subgoal/brief into the second leg")
    parser.add_argument("--restart-control", action="store_true",
                        help="also run a no-carry leg from the original start pose")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the legs and the stage layout, run nothing")
    parser.add_argument("--mock-script-leg2", type=str, default=None,
                        help="mock play script for resumed legs (default: --mock-script)")
    # runner knobs, forwarded only when given (the runner's own defaults apply)
    parser.add_argument("--world", type=str, default=None)
    parser.add_argument("--ply", type=str, default=None)
    parser.add_argument("--combined-ply", type=str, default=None, dest="combined_ply")
    parser.add_argument("--background-ply", type=str, default=None, dest="background_ply")
    parser.add_argument("--semantic-manifest", type=str, default=None, dest="semantic_manifest")
    parser.add_argument("--scene-scale", type=float, default=None, dest="scene_scale")
    parser.add_argument("--min-opacity", type=float, default=None, dest="min_opacity")
    parser.add_argument("--voxel-size", type=float, default=None, dest="voxel_size")
    parser.add_argument("--map-mode", type=str, default=None, dest="map_mode")
    parser.add_argument("--sensing", type=str, default=None)
    parser.add_argument("--moge-model-path", type=str, default=None, dest="moge_model_path")
    parser.add_argument("--worker-python", type=str, default=None, dest="worker_python")
    parser.add_argument("--target-name", type=str, default=None, dest="target_name")
    parser.add_argument("--target-pos", type=float, nargs=3, default=None, dest="target_pos")
    parser.add_argument("--target-radius", type=float, default=None, dest="target_radius")
    parser.add_argument("--start-pos", type=float, nargs=3, default=None, dest="start_pos")
    parser.add_argument("--start-yaw-deg", type=float, default=None, dest="start_yaw_deg")
    parser.add_argument("--target-index", type=int, default=None, dest="target_index")
    parser.add_argument("--initial-yaw-mode", type=str, default=None, dest="initial_yaw_mode")
    parser.add_argument("--max-init-attempts", type=int, default=None, dest="max_init_attempts")
    parser.add_argument("--render-size", type=int, nargs=2, default=None, dest="render_size")
    parser.add_argument("--env-overrides", type=str, nargs="*", default=None, dest="env_overrides")
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None, dest="model_path")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None, dest="base_url")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None, dest="max_new_tokens")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, dest="image_size")
    parser.add_argument("--num-threads", type=int, default=None, dest="num_threads")
    parser.add_argument("--no-think", action="store_true", dest="no_think")
    parser.add_argument("--mock-script", type=str, default=None, dest="mock_script")
    parser.add_argument("--toolset", type=str, default=None)
    parser.add_argument("--max-turns", type=int, default=None, dest="max_turns")
    parser.add_argument("--tool-budget", type=int, default=None, dest="tool_budget")
    parser.add_argument("--context-frames", type=int, default=None, dest="context_frames")
    parser.add_argument("--recovery", type=str, default=None)
    parser.add_argument("--state-block", type=str, default=None, dest="state_block")
    parser.add_argument("--search-skill", type=str, default=None, dest="search_skill")
    parser.add_argument("--memory-spatial", type=str, default=None, dest="memory_spatial")
    parser.add_argument("--memory-episodic", type=str, default=None, dest="memory_episodic")
    parser.add_argument("--scan-views", type=int, default=None, dest="scan_views")
    parser.add_argument("--scan-pitch-deg", type=float, default=None, dest="scan_pitch_deg")
    parser.add_argument("--success-dist", type=float, default=None, dest="success_dist")
    return parser.parse_args(argv)


def plan_legs(
    seed: int,
    *,
    task: TaskSpec,
    args: argparse.Namespace,
    prompt_episode_id: str,
    interrupt: tuple[str, float | None] | None,
    base_budget: int,
    base_turns: int,
    resume_leg: LegResult | None = None,
    world: Any = None,
    control: bool = False,
    restart_control: bool = False,
) -> list[LegPlan]:
    """The legs one episode runs: a fresh leg, then (for interrupts) a second one.

    ``resume_leg`` is the interrupted leg the second leg continues; ``control``
    builds that second leg *without* the plan/subgoal/brief carry (the pose is
    kept: physically the drone is where it stopped), and ``restart_control``
    builds a leg from the original start pose with no carry at all - the paired
    arms that say what each channel is worth.
    """
    plans: list[LegPlan] = []
    first = LegPlan(
        label=f"seed{seed}_leg1",
        instruction=args.instruction or task.instruction,
        prompt_episode_id=prompt_episode_id,
        tool_budget=int(base_budget),
        max_turns=int(base_turns),
        mock_script=getattr(args, "mock_script", None),
    )
    if interrupt is not None:
        mode, value = interrupt
        if mode == "budget":
            budget = int(value) if value else max(1, int(base_budget) // 2)
            first.tool_budget = min(int(base_budget), budget)
            first.max_turns = min(int(base_turns), max(1, first.tool_budget))
        else:
            first.deadline_s = float(value) if value else 60.0
        first.interrupt_mode = mode
    fresh_plan, fresh_detail = build_fresh_plan(
        task,
        world=world,
        instruction=first.instruction,
        episode_id=first.label,
    )
    if fresh_plan is not None:
        first.ctx_extra = {"plan": fresh_plan}
        first.carry = {
            "mode": "fresh",
            "channels": ["plan"],
            "done": [],
            "plan_steps": [stage.step for stage in task.stages],
            "plan_current_step": fresh_plan.get("current_step"),
            "plan_subtasks": [
                {"id": item.get("id"), "status": item.get("status")}
                for item in fresh_plan.get("subtasks") or []
            ],
            "plan_carry_backend": fresh_detail.get("plan_backend"),
            "note": "fresh leg starts with a structured plan (no truth places)",
        }
    plans.append(first)
    if resume_leg is None:
        return plans
    state = build_resume_state(
        resume_leg, task, prompt_episode_id=prompt_episode_id, world=world,
        base_instruction=first.instruction,
    )
    carry = str(state["brief"])
    ctx_extra = {"plan": state["plan"], "subgoal": state["subgoal"]}
    teleport = (
        [float(v) for v in state["pose"]["position_flu"]],
        float(state["pose"]["yaw_rad"]),
    )
    follow_mock = getattr(args, "mock_script_leg2", None) or getattr(args, "mock_script", None)
    if control:
        plans.append(
            LegPlan(
                label=f"seed{seed}_leg2_control",
        role="control",
                instruction=first.instruction,
                prompt_episode_id=prompt_episode_id,
                tool_budget=int(base_budget),
                max_turns=int(base_turns),
                teleport=teleport,
                ctx_extra={},
                mock_script=follow_mock,
                carry={
                    "mode": "control",
                    "channels": ["pose"],
                    "done": [],
                    "note": "same pose as the resumed leg, no plan/subgoal/brief carry",
                },
            )
        )
        return plans
    if restart_control:
        # the original pose is the fresh leg's own recorded start pose (the
        # sampled one, not the task file's request): a restart is only a restart
        # if the drone really goes back to where the episode began
        restart_pose = (
            ([float(v) for v in resume_leg.start_pose], float(resume_leg.start_yaw_rad))
            if resume_leg.start_pose
            else teleport
        )
        plans.append(
            LegPlan(
                label=f"seed{seed}_leg2_restart",
                role="restart",
                instruction=first.instruction,
                prompt_episode_id=prompt_episode_id,
                tool_budget=int(base_budget),
                max_turns=int(base_turns),
                teleport=restart_pose,
                ctx_extra={},
                mock_script=follow_mock,
                carry={
                    "mode": "restart",
                    "channels": [],
                    "done": [],
                    "note": "original start pose, no carry at all",
                },
            )
        )
        return plans
    plans.append(
        LegPlan(
            label=f"seed{seed}_leg2_resume",
            role="resume",
            instruction=str(state["instruction"]),
            prompt_episode_id=prompt_episode_id,
            tool_budget=int(base_budget),
            max_turns=int(base_turns),
            teleport=teleport,
            ctx_extra=ctx_extra,
            mock_script=follow_mock,
            carry={
                "mode": "resume",
                "channels": list(RESUME_CHANNELS),
                "done": list(state["done"]),
                "plan_steps": [stage.step for stage in task.stages],
                "plan_current_step": ctx_extra["plan"].get("current_step"),
                "plan_subtasks": [
                    {"id": item.get("id"), "status": item.get("status")}
                    for item in ctx_extra["plan"].get("subtasks") or []
                ],
                "plan_carry_backend": (state.get("plan_carry_detail") or {}).get("carry_backend"),
                "subgoal": ctx_extra.get("subgoal"),
                "brief": carry,
                "brief_places": list(state["brief_places"]),
                "brief_pose": state.get("brief_pose"),
                "stages_at_interrupt": int(state["complete_count"]),
            },
        )
    )
    return plans


def gate_for(world: Any, args: argparse.Namespace) -> float | None:
    try:
        return float(RUNNER.success_gate_m(world, args))
    except Exception:  # noqa: BLE001 - a world without a target has no gate
        return None


def run_episode_legs(
    seed: int,
    *,
    task: TaskSpec,
    args: argparse.Namespace,
    world: Any,
    probe: ExecutorProbe,
    vlm_backend: Any,
    out_dir: Path,
    interrupt: tuple[str, float | None] | None,
    base_budget: int,
    base_turns: int,
) -> tuple[list[LegResult], list[dict]]:
    """Run one episode's legs on the shared world and return them in order.

    The first leg always runs; when it ends mid-task on the driver's interrupt it
    is followed by the resumed leg (``--resume on``), by the no-carry control
    (``--resume off``) and, with ``--restart-control``, by a leg from the original
    start pose. A leg that finishes on its own ends the episode: there is nothing
    to resume.
    """
    prompt_id = RUNNER.neutral_episode_id(seed)
    legs: list[LegResult] = []
    trace_rows: list[dict] = []
    first_plan = plan_legs(
        seed, task=task, args=args, prompt_episode_id=prompt_id,
        interrupt=interrupt, base_budget=base_budget, base_turns=base_turns,
        world=world,
    )[0]
    result = run_leg(
        first_plan, task=task, args=args, world=world, probe=probe,
        vlm_backend=vlm_backend, seed=seed, episode_index=None,
        out_dir=out_dir / first_plan.label,
    )
    legs.append(result)
    trace_rows.append(ledger_vs_trace_row(result))
    if not result.interrupted:
        if first_plan.interrupt_mode:
            print(
                f"[long_horizon] seed {seed}: leg 1 finished on its own "
                f"({result.summary.get('finish_reason') if result.summary else result.error})"
                " - no resume leg",
                flush=True,
            )
        return legs, trace_rows
    follows: list[LegPlan] = []
    common = dict(
        task=task, args=args, prompt_episode_id=prompt_id, interrupt=None,
        base_budget=base_budget, base_turns=base_turns, resume_leg=result, world=world,
    )
    if args.resume == "off":
        follows += plan_legs(seed, control=True, **common)[1:]
    else:
        follows += plan_legs(seed, **common)[1:]
        if args.restart_control:
            follows += plan_legs(seed, restart_control=True, **common)[1:]
    for follow in follows:
        follow_result = run_leg(
            follow, task=task, args=args, world=world, probe=probe,
            vlm_backend=vlm_backend, seed=seed, episode_index=None,
            out_dir=out_dir / follow.label,
        )
        legs.append(follow_result)
        trace_rows.append(ledger_vs_trace_row(follow_result))
    return legs, trace_rows


def ledger_vs_trace_row(leg: LegResult) -> dict:
    """The ledger's own calls against the runner's trace of the same leg.

    Both sides count the *tool* calls (``finish`` is a decision, not a call, and
    the trace counts it separately): a mismatch means the ledger missed a call or
    invented one, which would make every number built on it wrong.
    """
    trace = read_trace(leg.out_dir)
    calls = None
    if trace:
        calls = sum(
            1 for entry in trace
            if entry.get("tool") and not entry.get("parse_failure") and entry.get("tool") != "finish"
        )
    ledger_outer = sum(
        1 for entry in leg.entries if entry["outer"] and entry["tool"] != "finish"
    )
    return {
        "label": leg.plan.label,
        "ledger_outer": ledger_outer,
        "ledger_entries": len(leg.entries),
        "trace_calls": calls,
        "trace_turns": len(trace),
        "summary_turns": None if leg.summary is None else leg.summary.get("turns"),
    }


def _reset_for_seed(
    world: Any, args: argparse.Namespace, task: TaskSpec, seed: int, base_args: argparse.Namespace
) -> dict:
    """Put the shared world into this seed's start state before the fresh leg.

    The scene is loaded once for the whole run; what changes per episode is the
    start pose (sampled by the world's own reset, or the task's ``start_pos``) and
    the semantic target index, which the production env samples per reset and the
    driver pins (see :func:`force_env_target`).
    """
    info: dict[str, Any] = {"seed": int(seed)}
    scene = task.scene
    target_index = getattr(args, "target_index", None)
    if target_index is None:
        target_index = scene.get("target_index")
    if target_index is not None and hasattr(world, "env"):
        info.update(force_env_target(world, int(target_index), seed))
    elif callable(getattr(world, "reset", None)):
        try:
            world.reset(seed=int(seed))
            info["reset"] = True
        except Exception as exc:  # noqa: BLE001 - a world that cannot reset keeps its pose
            info["reset_error"] = f"{type(exc).__name__}: {exc}"
    start = getattr(args, "start_pos", None) or scene.get("start_pos")
    yaw_deg = getattr(args, "start_yaw_deg", None)
    if yaw_deg is None:
        yaw_deg = scene.get("start_yaw_deg")
    if start is not None and not info.get("applied"):
        info["teleport"] = [float(v) for v in start]
        teleport_world(world, start, math.radians(float(yaw_deg or 0.0)))
    info["target_index"] = None if target_index is None else int(target_index)
    info["pose"] = world_pose(world)[0]
    info["yaw_rad"] = world_pose(world)[1]
    info["gate_m"] = gate_for(world, base_args)
    return info


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    if args.analyze:
        return analyze_run(Path(args.analyze))
    if not args.task_file:
        raise SystemExit("--task-file is required (or use --analyze RUN_DIR)")
    if not args.out:
        raise SystemExit("--out is required")
    seeds = parse_seeds(args.seeds)
    interrupts = interrupt_map(args.interrupt)
    task = load_task(args.task_file, instruction=args.instruction)
    chain = chain_check(task)
    if not chain["ok"]:
        print(f"[long_horizon] WARNING: chain check problems: {chain['problems']}", flush=True)
    out_dir = Path(args.out)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "task.json", "w") as handle:
            json.dump(jsonable(task.to_dict()), handle, indent=2)

    base_args = RUNNER.parse_args(
        runner_argv(
            task, args, out_dir=out_dir / "probe", instruction=task.instruction,
            tool_budget=args.tool_budget, max_turns=args.max_turns,
            seed=seeds[0], episode_index=None,
        )
    )
    base_budget = int(base_args.tool_budget)
    base_turns = int(base_args.max_turns)
    backend_name = str(getattr(args, "backend", None) or base_args.backend)

    if args.dry_run:
        print(json.dumps(
            {
                "task": task.to_dict(),
                "chain_check": chain,
                "seeds": seeds,
                "interrupts": {str(k): list(v) for k, v in interrupts.items()},
                "base_budget": base_budget,
                "base_turns": base_turns,
                "backend": backend_name,
                "legs_per_seed": {
                    str(seed): [
                        plan.to_dict()
                        for plan in plan_legs(
                            seed, task=task, args=args,
                            prompt_episode_id=RUNNER.neutral_episode_id(seed),
                            interrupt=interrupts.get(seed),
                            base_budget=base_budget, base_turns=base_turns,
                            resume_leg=None,
                        )
                    ]
                    for seed in seeds
                },
            },
            indent=2, default=str,
        ))
        return {}

    world = RUNNER.build_world(base_args)
    probe = ExecutorProbe(world)
    backend_is_mock = backend_name.startswith("mock")
    vlm_backend = None if backend_is_mock else RUNNER.build_vlm(base_args)
    gate = gate_for(world, base_args)

    episodes: list[dict] = []
    checks_rows: list[dict] = []
    for seed in seeds:
        start = f"seed{seed}"
        episode_dir = out_dir / start
        episode_dir.mkdir(parents=True, exist_ok=True)
        reset_info = _reset_for_seed(world, args, task, seed, base_args)
        print(
            f"[long_horizon] {start}: pose={[round(v, 2) for v in world_pose(world)[0]]} "
            f"target={reset_info.get('target_index')} budget={base_budget} turns={base_turns}",
            flush=True,
        )
        legs, rows = run_episode_legs(
            seed, task=task, args=args, world=world, probe=probe, vlm_backend=vlm_backend,
            out_dir=episode_dir, interrupt=interrupts.get(seed),
            base_budget=base_budget, base_turns=base_turns,
        )
        checks_rows += rows
        metrics = episode_metrics(task, legs, gate_m=gate)
        metrics["seed"] = seed
        metrics["reset"] = reset_info
        episodes.append(metrics)
        with open(episode_dir / "episode.json", "w") as handle:
            json.dump(jsonable(metrics), handle, indent=2)
        print(
            f"[long_horizon] {start}: stages {metrics['stages']['complete_count']}/"
            f"{metrics['stages']['n_stages']} success={metrics['success']} "
            f"traceable={metrics['traceable_success']} interrupted={metrics['interrupted']} "
            f"recovered={metrics['interrupt_recovered']}",
            flush=True,
        )
        with open(out_dir / EPISODES_JSONL, "a") as handle:
            handle.write(json.dumps(jsonable({
                "seed": seed,
                "label": metrics["label"],
                "n_legs": metrics["n_legs"],
                "success": metrics["success"],
                "traceable_success": metrics["traceable_success"],
                "chain_complete": metrics["stages"]["chain_complete"],
                "complete_count": metrics["stages"]["complete_count"],
                "interrupted": metrics["interrupted"],
                "interrupt_recovered": metrics["interrupt_recovered"],
                "verify_ok": metrics["verify"]["ok"],
                "evidence_complete": metrics["evidence_complete"],
            }), default=str) + "\n")

    meta = {
        "task_name": task.name,
        "task_file": None if task.path is None else str(task.path),
        "instruction": task.instruction,
        "seeds": seeds,
        "n_episodes": len(episodes),
        "world": str(args.world or task.scene.get("world") or base_args.world),
        "backend": backend_name,
        "model_path": getattr(args, "model_path", None) or base_args.model_path,
        "base_tool_budget": base_budget,
        "base_max_turns": base_turns,
        "success_gate_m": gate,
        "interrupts": {str(k): list(v) for k, v in interrupts.items()},
        "resume": args.resume,
        "restart_control": bool(args.restart_control),
        "switches": RUNNER.switch_report(base_args),
        "switches_applied": {
            str(seed): [
                [leg["label"], leg.get("switches")]
                for leg in row["legs"]
            ]
            for seed, row in zip(seeds, episodes)
        },
        "chain_check": chain,
        "probe_installed": probe.installed,
        "resume_channels": list(RESUME_CHANNELS),
        "resume_backend": "ctx_extra.plan+subgoal+instruction_brief+pose",
        "memory_carry": (
            "unavailable with the current interface: the memory module is built per "
            "leg (build_memory(out_dir, episode_id)) and exposes no cross-episode load; "
            "the resumed leg's knowledge travels through the plan record, the subgoal, "
            "the brief and the carried pose only"
        ),
    }
    report = {
        "meta": meta,
        "task": task.to_dict(),
        "aggregate": aggregate(episodes, task, meta),
        "checks": protocol_checks(episodes, task, ledger_vs_trace=checks_rows),
        "episodes": episodes,
    }
    with open(out_dir / REPORT_JSON, "w") as handle:
        json.dump(jsonable(report), handle, indent=2)
    with open(out_dir / REPORT_MD, "w") as handle:
        handle.write(report_markdown(report))
    print(
        f"[long_horizon] report: {out_dir / REPORT_JSON} | {out_dir / REPORT_MD}",
        flush=True,
    )
    close = getattr(world, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - a world that refuses to close is not fatal
            pass
    return report


if __name__ == "__main__":
    main()
