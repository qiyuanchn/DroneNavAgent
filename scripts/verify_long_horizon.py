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
                               mid-task; the resumed leg inherits a *runtime
                               checkpoint* saved verbatim at the interrupt.
                               Version 3 removes the evaluator from this path:
                                 checkpoint   the live ``ctx.extra`` state (plan
                                              record and sub-task statuses,
                                              confirmation ledger, candidate /
                                              observation / viewpoint memory
                                              references, task scope, beliefs,
                                              open questions, evidence pack) is
                                              written to ``checkpoint.json`` as
                                              JSON and loaded as JSON; a wrong
                                              "done" survives the restart,
                                 plan         the checkpoint's own record is
                                              adopted (the module's ``resume_plan``
                                              logs the adoption) - it is not
                                              re-written from ``stage_report``,
                                 pose         the world keeps the drone where the
                                              leg ended (teleport + trajectory),
                                 subgoal      ``ctx.extra["subgoal"]`` (the STATE
                                              block's ``subgoal`` label),
                                 brief        an instruction suffix listing the
                                              checkpoint's own statuses/evidence
                                              and places - never the scene's
                                              ground truth, so no leak
                               The offline score of that checkpoint
                               (``checkpoint_correctness``) compares the claimed
                               done/confirmed stages with ``stage_report`` and
                               counts the beliefs the resumed leg corrected.
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

``--plan-mode given|self`` separates *executing a plan* from *making one*
(goal.md section 4). ``given`` (default) is the P4 behavior above: the driver
writes one sub-task per stage through ``update_plan`` before the first turn.
``self`` writes no plan at all; the runtime gives the model the natural-language
instruction, asks for a plan on the first decision turn, validates its structure
(``tool_impl_task.plan_structure_errors``), asks for one correction when invalid
and then executes the accepted plan. Stage ``done_when`` criteria are typed
(``sighted``/``located``/``arrived``/``verified``): a landmark that only has to
be found is judged on a located sighting, not on a flight; an ``arrived``
destination keeps the reach + accepted-confirmation contract. The report's
``plan_quality`` block is an offline audit of the plan the model wrote against
the truth stages (count, order, coverage, type accuracy) and never enters a
prompt; ``reach_attempts_by_type`` prices the semantic criterion in motion.

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
summary.json,leg_summary.json,run_config.json,ledger.jsonl,episode_log.json,
checkpoint.json}``. The ledger is the driver's own record of every tool call
(outer and routine-inner) with the drone pose and the executor counters before
and after it; ``checkpoint.json`` is the runtime's own state at the end of the
leg (the version-3 resume source); ``episode_log.json`` is the runtime's own log
(decision nodes, sub-task transitions, the adopted plan).

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
import copy
import importlib.util
import json
import math
import os
import re
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
#: Typed completion criteria of a stage (goal.md section 4): a landmark is
#: ``sighted`` when a located tool result is enough, a destination is
#: ``arrived`` when the drone has to fly there (reach + accepted confirmation),
#: and ``verified`` asks for the accepted confirmation alone.
DONE_WHEN_SIGHTED = "sighted"
DONE_WHEN_LOCATED = "located"
DONE_WHEN_ARRIVED = "arrived"
DONE_WHEN_VERIFIED = "verified"
DONE_WHEN_TYPES = (
    DONE_WHEN_SIGHTED,
    DONE_WHEN_LOCATED,
    DONE_WHEN_ARRIVED,
    DONE_WHEN_VERIFIED,
)
#: The task files' own reach wording (evaluation-side inference only).
REACH_WORDS = (
    "reach",
    "arriv",
    "fly",
    "approach",
    "go to",
    "到达",
    "抵达",
    "飞到",
    "接近",
    "停在",
)
#: An arrival on the candidate's standoff shell is a reach even when the stage
#: centre is farther than ``reach_m`` (the two geometries are alternatives).
DEFAULT_APPROACH_STANDOFF_M = 1.2
CANDIDATE_REACH_TOLERANCE_M = 0.35
#: A new candidate this close to the confirmed one in the same sub-task is the
#: same instance: the accepted confirmation is inherited, not stale.
CONFIRMATION_INHERIT_RADIUS_M = 1.5
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
    #: the typed completion criterion (goal.md section 4): ``sighted`` /
    #: ``located`` / ``arrived`` / ``verified``. A task file may declare it;
    #: otherwise ``arrived`` when the done_when says reach, ``sighted`` for a
    #: landmark/object stage and ``arrived`` for the target stage.
    done_when_type: str = ""
    #: whether the task file itself declared ``reach_m`` (a pre-typed file's
    #: reach declaration is a movement condition unless a type says otherwise)
    reach_declared: bool = False
    expected_observation: str = ""
    failure_branch: tuple[str, ...] = ()
    #: evaluation-side words a self-written plan goal may use for this stage
    #: (the model rarely repeats the full label); never enters the prompt.
    keywords: tuple[str, ...] = ()

    def effective_done_when_type(self) -> str:
        """The criterion the offline stage check uses.

        An explicit ``done_when_type`` wins. Otherwise the stage file's own
        wording decides: a reach phrase means the stage has to be flown to;
        a landmark/object that only has to be found is ``sighted``; the target
        is ``arrived``. This keeps every pre-typed task file on its historical
        reach semantics (their ``done_when`` spells "reached") while a task
        that only says "find the landmark" stops checking a flight it never
        asked for.
        """
        declared = str(self.done_when_type or "").strip().lower()
        if declared in DONE_WHEN_TYPES:
            return declared
        text = str(self.done_when or "").lower()
        if self.reach_declared:
            return DONE_WHEN_ARRIVED
        if any(word in text for word in REACH_WORDS):
            return DONE_WHEN_ARRIVED
        if self.target or self.kind == STAGE_TARGET:
            return DONE_WHEN_ARRIVED
        return DONE_WHEN_SIGHTED

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
            "done_when_type": self.effective_done_when_type(),
            "expected_observation": self.expected_observation,
            "failure_branch": list(self.failure_branch),
            "keywords": list(self.keywords),
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
                done_when_type=str(raw.get("done_when_type") or ""),
                reach_declared=raw.get("reach_m") is not None,
                expected_observation=str(raw.get("expected_observation") or ""),
                failure_branch=tuple(
                    str(item) for item in (raw.get("failure_branch") or [])
                ),
                keywords=tuple(str(item) for item in (raw.get("keywords") or []) if str(item)),
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
    argv += _flag_tokens("--plan-mode", getattr(args, "plan_mode", None), False)
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
    #: the runtime state exactly as the interrupted leg left it (version 3).
    #: Resume consumes this checkpoint verbatim; nothing from ``stage_report``
    #: is mixed into it.
    checkpoint: dict = field(default_factory=dict)
    #: the plan record at the end of the leg (the agent's own claim, used by
    #: the offline checkpoint-correctness metric to see what it corrected).
    final_plan: dict | None = None

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
            "checkpoint": bool(self.checkpoint),
            "checkpoint_fields": sorted(self.checkpoint),
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
    "navigation": "vlm_nav.tool_impl_navigation",
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

    Every stage carries its typed ``done_when`` (``done_when_type``) and, for an
    ``arrived`` stage only, its ``reach_m`` tolerance; a ``sighted`` landmark
    keeps no reach requirement. One concrete ``next_step`` call - the stage's
    own label, never its true place - gives the executor a deterministic step
    source from the first turn. ``include_budget=False`` is the fresh leg's
    plan: its per-stage budget is left to the episode's own tool budget.
    """
    specs: list[dict] = []
    for stage in task.stages:
        row = (report.get("stages") or {}).get(stage.stage_id) or {}
        stage_type = stage.effective_done_when_type()
        if stage.done_when:
            done_when = stage.done_when
        elif stage_type in (DONE_WHEN_SIGHTED, DONE_WHEN_LOCATED):
            done_when = (
                f"stage {stage.index} ({stage.label}) is sighted and located "
                f"(a tool result within {stage.locate_m:.1f} m of it; no flight)"
            )
        elif stage_type == DONE_WHEN_VERIFIED:
            done_when = f"stage {stage.index} ({stage.label}) is confirmed (identity verified)"
        else:
            done_when = (
                f"stage {stage.index} ({stage.label}) is reached and located "
                f"(a tool result within {stage.locate_m:.1f} m of it)"
            )
        spec: dict[str, Any] = {
            "id": f"st_{stage.index}",
            "goal": stage.step,
            "done_when": done_when,
            "done_when_type": stage_type,
            "expected_observation": stage.expected_observation or (
                f"stage {stage.index}: {stage.label}"
            ),
            "next_step": {
                "tool": STAGE_NEXT_TOOL,
                "args": {"instruction": stage.label},
            },
            "failure_branch": list(stage.failure_branch) or [
                "check task_status and the failure branch, re-observe from another "
                "viewpoint, then re-plan this stage with update_plan"
            ],
        }
        if stage_type == DONE_WHEN_ARRIVED:
            spec["reach_m"] = float(stage.reach_m)
        if include_budget:
            spec["budget"] = {"tool_calls": max(2, int(row.get("evidence_count") or 0) + 3)}
        if stage.anchor:
            anchor = next(s for s in task.stages if s.stage_id == stage.anchor)
            spec["depends_on"] = [f"st_{anchor.index}"]
        specs.append(spec)
    return specs


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


# ---------------------------------------------------------------------------
# task scope: offline verdicts under the same contract as ``finish`` (goal.md v1)
# ---------------------------------------------------------------------------

#: The four scope fields of the v1 contract (``vlm_nav.tools.SCOPE_FIELDS``).
SCOPE_FIELDS = ("task_id", "subtask_id", "candidate_id", "evidence_revision")
#: Tools whose answer grounds a candidate identity (``vlm_nav.tools.IDENTITY_TOOLS``).
IDENTITY_TOOLS = ("ground_target", "search_target")
#: Tools whose answer flies to a place; ``outcome == arrived`` is the arrival.
ARRIVAL_TOOLS = ("approach_candidate", "go_to_viewpoint", "return_to_observation")
ARRIVAL_OUTCOMES = ("arrived",)


def entry_scope(entry: Mapping) -> dict | None:
    """The task scope a ledger entry's result carries, or None (pre-v1 trace).

    ``ToolRegistry.call`` stamps every result with ``data["scope"]``; the four
    fields are the ones ``finish`` filters its evidence pack by. A ledger written
    before the scope contract has no such block and is evaluated the legacy,
    unscoped way (``stage_report`` and ``completion_chain`` say which one ran).
    """
    data = entry.get("data")
    if not isinstance(data, Mapping):
        return None
    scope = data.get("scope")
    if not isinstance(scope, Mapping):
        return None
    values = {field: scope.get(field) for field in SCOPE_FIELDS}
    if not any(
        values[field] not in (None, "")
        for field in ("task_id", "subtask_id", "candidate_id")
    ):
        return None
    return values


def final_scope(entries: Sequence[Mapping]) -> dict | None:
    """The scope in force at the end of a chain (last result that carries one)."""
    for entry in reversed(list(entries)):
        scope = entry_scope(entry)
        if scope is not None:
            return scope
    return None


def _stage_index_of_text(text: Any, task: TaskSpec) -> int | None:
    """The stage a free-text goal names (exact, containment, then keywords).

    The keyword pass is what maps a self-written plan to the truth stages: the
    model rarely repeats the whole label ("卧室窗边的木椅") but almost always
    names the object ("木椅"). ``keywords`` is evaluation-side data from the task
    file and never reaches a prompt. A relation goal names two objects ("木椅
    旁边的灯"); :func:`map_goals_to_stages` resolves that by order, this
    single-text lookup returns the best (longest) match.
    """
    value = str(text or "").strip().casefold()
    if not value:
        return None
    for stage in task.stages:
        if value in (stage.step.casefold(), stage.label.casefold()):
            return stage.index
    best: tuple[int, int] | None = None
    for stage in task.stages:
        for candidate in (stage.step, stage.label):
            needle = str(candidate or "").strip().casefold()
            if needle and needle in value and (best is None or len(needle) > best[0]):
                best = (len(needle), stage.index)
    if best is not None:
        return best[1]
    for stage in task.stages:
        for needle in stage.keywords:
            token = str(needle or "").strip().casefold()
            if token and token in value and (best is None or len(token) > best[0]):
                best = (len(token), stage.index)
    return None if best is None else best[1]


def _stage_match_score(text: Any, stage: StageSpec) -> int:
    """How strongly a goal text names one stage (0 = no match).

    The score is the length of the longest label/step substring or declared
    keyword the goal contains. It only ranks candidate stages; order breaks
    ties (:func:`map_goals_to_stages`).
    """
    value = str(text or "").strip().casefold()
    if not value:
        return 0
    best = 0
    for candidate in (stage.step, stage.label):
        needle = str(candidate or "").strip().casefold()
        if needle and needle in value:
            best = max(best, len(needle))
    for needle in stage.keywords:
        token = str(needle or "").strip().casefold()
        if token and token in value:
            best = max(best, len(token))
    return best


def map_goals_to_stages(texts: Sequence[Any], task: TaskSpec) -> list[int | None]:
    """Map an ordered plan's goals to the truth stages, order-aware.

    A relation goal names its anchor too ("找到木椅旁边的灯"), so a plain
    longest-keyword match would assign it to the chair stage. The plan is
    ordered, which disambiguates: each goal is assigned to a not-yet-used stage
    *after* the previous assignment when one matches. The result is what the
    audit's coverage/order metrics count.
    """
    used: set[int] = set()
    last = 0
    out: list[int | None] = []
    for text in texts:
        candidates = [
            stage.index
            for stage in task.stages
            if stage.index not in used and _stage_match_score(text, stage) > 0
        ]
        forward = [index for index in candidates if index > last]
        if forward:
            chosen: int | None = forward[0]
        elif candidates:
            chosen = candidates[0]
        else:
            chosen = _stage_index_of_text(text, task)
        out.append(chosen)
        if chosen is not None:
            used.add(chosen)
            last = chosen
    return out


def offline_plan_subtasks(legs: Sequence[LegResult], task: TaskSpec) -> list[dict]:
    """The final sub-task list reconstructed from the run's own records.

    The statuses a leg's plan started with (``carry.plan_subtasks``, written by
    ``tool_impl_task.update_plan``) are updated by every ``plan_transitions``
    entry the runtime logged. A plan the model restated with new ids adds those
    ids; the driver's own positional convention (``st_1..st_n``) is the floor.
    """
    subtasks: dict[str, dict] = {}
    order: list[str] = []

    def put(subtask_id: str, goal: str, status: str) -> None:
        if not subtask_id:
            return
        if subtask_id not in subtasks:
            subtasks[subtask_id] = {
                "id": subtask_id,
                "goal": goal or "",
                "status": status or "pending",
            }
            order.append(subtask_id)
            return
        if goal:
            subtasks[subtask_id]["goal"] = goal
        if status:
            subtasks[subtask_id]["status"] = status

    for leg in legs:
        rows = leg.plan.carry.get("plan_subtasks") or []
        for position, row in enumerate(rows):
            if not isinstance(row, Mapping):
                continue
            goal = str(row.get("goal") or "")
            if not goal and position < len(task.stages):
                goal = task.stages[position].step
            put(str(row.get("id") or f"st_{position + 1}"),
                goal, str(row.get("status") or "pending"))
        for position, step in enumerate(leg.plan.carry.get("plan_steps") or []):
            if position < len(task.stages):
                put(f"st_{position + 1}", str(step), "")
        for transition in leg.log.get("plan_transitions") or []:
            if isinstance(transition, Mapping):
                put(str(transition.get("id") or ""), "", str(transition.get("to") or ""))
    if not order:
        for stage in task.stages:
            put(f"st_{stage.index}", stage.step, "pending")
    return [subtasks[subtask_id] for subtask_id in order]


def stage_subtask_map(task: TaskSpec, legs: Sequence[LegResult] = ()) -> dict[str, int]:
    """``subtask_id -> stage index`` for the plan the episode really ran.

    The driver's plan names the stages ``st_1..st_n``; a plan the model rewrote
    keeps those ids when it patches in place and is mapped by goal text
    otherwise. Without any plan record the positional convention is all there is.
    """
    mapping = {f"st_{stage.index}": stage.index for stage in task.stages}
    for leg in legs:
        rows = [
            row for row in leg.plan.carry.get("plan_subtasks") or []
            if isinstance(row, Mapping)
        ]
        indices = map_goals_to_stages(
            [str(row.get("goal") or "") for row in rows], task
        )
        for row, index in zip(rows, indices):
            subtask_id = str(row.get("id") or "")
            if subtask_id and index:
                mapping[subtask_id] = index
    for leg in legs:
        plan = leg.final_plan
        if not isinstance(plan, Mapping) and isinstance(leg.checkpoint, Mapping):
            plan = leg.checkpoint.get("plan")
        rows = [
            row for row in (plan or {}).get("subtasks") or []
            if isinstance(row, Mapping)
        ]
        if not rows:
            continue
        indices = map_goals_to_stages(
            [str(row.get("goal") or "") for row in rows], task
        )
        for row, index in zip(rows, indices):
            subtask_id = str(row.get("id") or "")
            if subtask_id and index:
                mapping[subtask_id] = index
    pending = [
        subtask for subtask in offline_plan_subtasks(legs, task)
        if str(subtask.get("id") or "") not in mapping
        or mapping.get(str(subtask.get("id") or "")) is None
    ]
    indices = map_goals_to_stages(
        [str(subtask.get("goal") or "") for subtask in pending], task
    )
    for subtask, index in zip(pending, indices):
        if index:
            mapping[str(subtask.get("id"))] = index
    return mapping


def plan_quality_audit(
    legs: Sequence[LegResult], task: TaskSpec, *, plan_mode: str = "given"
) -> dict:
    """Offline audit of the plan the episode *ran* against the truth stages.

    Evaluation only: nothing here reaches a prompt. The audit says whether the
    plan has the stage count the mission needs, whether its goals map to the
    stages in the declared order (goal-text coverage via label/keyword
    matching), and whether each stage's typed ``done_when`` criterion is the
    semantic one the mission implies (a landmark ``sighted``, the target
    ``arrived``). ``structure_errors`` comes from the task module's own
    ``plan_structure_errors`` - the same check the self-plan runtime uses.
    """
    record: Mapping | None = None
    source_leg = ""
    for leg in legs:
        candidate = leg.final_plan
        if not isinstance(candidate, Mapping) and isinstance(leg.checkpoint, Mapping):
            candidate = leg.checkpoint.get("plan")
        if isinstance(candidate, Mapping) and candidate.get("subtasks"):
            record = candidate
            source_leg = leg.plan.label
            break
    if record is None:
        return {
            "available": False,
            "plan_mode": plan_mode,
            "n_subtasks": 0,
            "n_stages": len(task.stages),
            "count_match": False,
            "coverage": 0,
            "coverage_rate": 0.0,
            "order_ok": False,
        }
    modules = m1_modules()
    task_module = modules.get("task")
    read_type = getattr(task_module, "done_when_type", None) if task_module else None
    structure_errors = (
        task_module.plan_structure_errors(record)
        if task_module is not None and hasattr(task_module, "plan_structure_errors")
        else []
    )
    subtasks = [
        subtask for subtask in record.get("subtasks") or [] if isinstance(subtask, Mapping)
    ]
    rows: list[dict] = []
    goals = [str(subtask.get("goal") or "") for subtask in subtasks]
    indices: list[int | None] = map_goals_to_stages(goals, task)
    covered: set[int] = set()
    type_hits = 0
    type_checked = 0
    done_when_present = 0
    for position, (subtask, index) in enumerate(zip(subtasks, indices), 1):
        goal = goals[position - 1]
        stage = task.stages[index - 1] if index else None
        got_type = ""
        if callable(read_type):
            try:
                got_type = str(read_type(subtask) or "")
            except Exception:  # noqa: BLE001 - an audit never breaks the report
                got_type = ""
        if not got_type:
            got_type = str(subtask.get("done_when_type") or "")
        if str(subtask.get("done_when") or "").strip():
            done_when_present += 1
        expected_type = stage.effective_done_when_type() if stage else ""
        type_ok: bool | None = None
        if stage is not None and expected_type:
            type_checked += 1
            type_ok = got_type == expected_type or (
                expected_type == DONE_WHEN_SIGHTED and got_type == DONE_WHEN_LOCATED
            )
            if type_ok:
                type_hits += 1
        if index is not None:
            covered.add(index)
        rows.append(
            {
                "position": position,
                "id": str(subtask.get("id") or ""),
                "goal": goal[:120],
                "stage": None if stage is None else stage.stage_id,
                "stage_index": index,
                "done_when_type": got_type,
                "expected_type": expected_type,
                "type_ok": type_ok,
                "depends_on": [str(dep) for dep in subtask.get("depends_on") or []],
            }
        )
    mapped = [index for index in indices if index is not None]
    order_ok = bool(mapped) and mapped == sorted(mapped) and len(set(mapped)) == len(mapped)
    return {
        "available": True,
        "plan_mode": plan_mode,
        "plan_id": str(record.get("plan_id") or ""),
        "source_leg": source_leg,
        "n_subtasks": len(subtasks),
        "n_stages": len(task.stages),
        "count_match": len(subtasks) == len(task.stages),
        "order_indices": indices,
        "order_ok": order_ok,
        "coverage": len(covered),
        "coverage_rate": _rate(len(covered), len(task.stages)),
        "done_when_present": done_when_present,
        "done_when_type_checked": type_checked,
        "done_when_type_hits": type_hits,
        "done_when_type_accuracy": _rate(type_hits, type_checked),
        "structure_errors": list(structure_errors),
        "structure_ok": not structure_errors,
        "rows": rows,
    }


def entry_stage_index(entry: Mapping, mapping: Mapping[str, int]) -> int | None:
    """The stage a result belongs to, by its scope's ``subtask_id``."""
    scope = entry_scope(entry)
    if scope is None:
        return None
    subtask_id = str(scope.get("subtask_id") or "")
    if not subtask_id:
        return None
    if subtask_id in mapping:
        return int(mapping[subtask_id])
    match = re.fullmatch(r"st_(\d+)", subtask_id)
    return int(match.group(1)) if match else None


def offline_plan_completion(legs: Sequence[LegResult], task: TaskSpec) -> dict | None:
    """The plan's completion verdict, through the task module's own function.

    ``finish`` refuses a success claim on ``plan_completion(ctx)``; the driver
    rebuilds that call from the leg artifacts (statuses from the plan record the
    legs carried and the runtime's transitions) so the offline verdict uses the
    same function, not a look-alike. ``evidence_missing`` in the returned rows is
    not read by the report: the reconstructed records carry no executor evidence
    marks; the scoped completion chain below is the evidence judge.
    """
    subtasks = offline_plan_subtasks(legs, task)
    modules = m1_modules()
    task_module = modules.get("task")
    tools_module = modules.get("tools")
    if not subtasks or task_module is None or tools_module is None:
        return None
    plan_id = ""
    for leg in legs:
        for entry in leg.entries:
            scope = entry_scope(entry)
            if scope and scope.get("task_id"):
                plan_id = str(scope["task_id"])
    record = {
        "plan_id": plan_id or "plan_1",
        "revision": sum(len(leg.log.get("plan_transitions") or []) for leg in legs),
        "steps": [str(item.get("goal") or "") for item in subtasks],
        "subtasks": subtasks,
        "current_step": 1,
        "done": [],
        "done_when": "",
        "note": "reconstructed from leg artifacts for offline evaluation",
        "instruction": str(legs[0].plan.instruction if legs else "")[:160],
        "episode_id": "offline_plan",
    }
    context = tools_module.ToolContext(
        world=None,
        memory=None,
        episode_id="offline_plan",
        observation_seq=0,
        budget={"tool_calls_left": 0},
        extra={"plan": record, "instruction": record["instruction"]},
    )
    try:
        verdict = task_module.plan_completion(context)
    except Exception:  # noqa: BLE001 - a reconstructed plan must not break a report
        return None
    if not isinstance(verdict, Mapping):
        return None
    verdict = dict(verdict)
    verdict["source"] = "vlm_nav.tool_impl_task.plan_completion"
    verdict["statuses"] = [
        {"id": item["id"], "goal": item["goal"], "status": item["status"]}
        for item in subtasks
    ]
    return verdict


def _scope_fields(entry: Mapping) -> dict:
    scope = entry_scope(entry) or {}
    try:
        revision = int(scope.get("evidence_revision") or 0)
    except (TypeError, ValueError):
        revision = 0
    return {
        "task_id": str(scope.get("task_id") or ""),
        "subtask_id": str(scope.get("subtask_id") or ""),
        "candidate_id": str(scope.get("candidate_id") or ""),
        "evidence_revision": revision,
    }


def completion_records(entries: Sequence[Mapping]) -> list[dict]:
    """Ledger entries in the evidence-pack shape ``_record_in_scope`` reads.

    The pack stores one record per evidence event (verification, grounded
    candidate, arrival) with the scope fields lifted to its ``data``; the ledger
    keeps one entry per call with the scope nested. This adapter rebuilds the
    pack shape from the calls, so the offline filter is the runtime's own.
    """
    records: list[dict] = []
    for entry in entries:
        tool = str(entry.get("tool") or "")
        if not tool:
            continue
        data = entry.get("data") if isinstance(entry.get("data"), Mapping) else {}
        fields = _scope_fields(entry)
        refs = list(entry.get("evidence") or [])
        status = str(entry.get("status") or "")
        call = order_of(entry) or entry.get("seq")
        if tool == "verify_goal":
            records.append({
                "tool": tool,
                "status": status,
                "data": {
                    "verified": data.get("verified"),
                    "checks": dict(data.get("checks") or {}),
                    "observation_id": str(data.get("observation_id") or ""),
                    **fields,
                },
                "evidence": refs,
                "call": call,
            })
        elif tool in IDENTITY_TOOLS:
            candidate_id = str(data.get("candidate_id") or "")
            if not candidate_id:
                candidates = data.get("candidates")
                if isinstance(candidates, (list, tuple)) and candidates:
                    top = candidates[0] if isinstance(candidates[0], Mapping) else {}
                    candidate_id = str(top.get("id") or top.get("candidate_id") or "")
            point = _as_position(data.get("position_flu"))
            if point is None:
                candidates = data.get("candidates")
                if isinstance(candidates, (list, tuple)) and candidates:
                    top = candidates[0] if isinstance(candidates[0], Mapping) else {}
                    point = _as_position(top.get("position_flu"))
            records.append({
                "tool": tool,
                "status": status,
                "data": {
                    "candidate_id": candidate_id,
                    "observation_id": str(data.get("observation_id") or ""),
                    "position_flu": list(point) if point is not None else None,
                    **fields,
                },
                "evidence": refs,
                "call": call,
            })
        elif tool == "confirm_candidate":
            records.append({
                "tool": tool,
                "status": status,
                "data": {
                    "verdict": str(data.get("verdict") or ""),
                    "candidate_id": str(data.get("candidate_id") or ""),
                    "observation_ids": list(data.get("observation_ids") or []),
                    "position_delta_m": data.get("position_delta_m"),
                    "missing": list(data.get("missing") or []),
                    **fields,
                },
                "evidence": refs,
                "call": call,
            })
        elif tool in ARRIVAL_TOOLS and (
            str(data.get("outcome") or "") in ARRIVAL_OUTCOMES
            or data.get("arrived") is True
        ):
            records.append({
                "tool": tool,
                "status": status,
                "data": {
                    "outcome": str(data.get("outcome") or ""),
                    "task_status": "arrived" if data.get("arrived") is True
                    else str(data.get("task_status") or ""),
                    "candidate_id": str(data.get("candidate_id") or data.get("target_id") or ""),
                    "scope_task_id": fields["task_id"],
                    **fields,
                },
                "evidence": refs,
                "call": call,
            })
        elif tool == "task_status" and data.get("completed") and data.get("arrived"):
            records.append({
                "tool": tool,
                "status": status,
                "data": {
                    "completed": True,
                    "arrived": True,
                    "candidate_id": str(data.get("candidate_id") or ""),
                    "scope_task_id": fields["task_id"],
                    **fields,
                },
                "evidence": refs,
                "call": call,
            })
    return records


def completion_chain(
    entries: Sequence[Mapping],
    *,
    scope: Mapping | None = None,
    task_scoped: bool | None = None,
) -> dict:
    """The runtime's own completion-chain verdict, filtered by the task scope.

    ``vlm_nav.tools.completion_evidence`` is the judge of a ``success: true``
    claim: a verification that passed, or a grounded candidate plus an arrival,
    and at least one evidence reference. The records are first filtered through
    ``vlm_nav.tools._record_in_scope`` with the scope in force at the end of the
    chain - the same filter ``finish`` applies - so a confirmation of an earlier
    candidate instance, revision or sub-task cannot back the claim. Without a
    scope-carrying entry (a pre-v1 artifact) the raw chain is reported and
    ``scoped`` says false.
    """
    modules = m1_modules()
    tools_module = modules.get("tools")
    if tools_module is None or not hasattr(tools_module, "completion_evidence"):
        return {"supported": False}
    records = completion_records(entries)
    verdict = dict(tools_module.completion_evidence(records))
    verdict["supported"] = True
    current = dict(scope) if isinstance(scope, Mapping) else final_scope(entries)
    # ``_record_in_scope`` lives with the finish/evidence-pack contract in the
    # navigation tool module (the same function ``completion_claim`` calls).
    filter_fn = getattr(tools_module, "_record_in_scope", None)
    if not callable(filter_fn):
        navigation = modules.get("navigation")
        filter_fn = getattr(navigation, "_record_in_scope", None)
    if current is None or not callable(filter_fn):
        verdict.update(scoped=False, raw_verified=bool(verdict["verified"]),
                       rejected_records=0, scoped_records=len(records), scope=None)
        return verdict
    scoped_task = bool(current.get("subtask_id")) if task_scoped is None else bool(task_scoped)
    kept = [record for record in records if filter_fn(record, current, scoped_task)]
    scoped_verdict = dict(tools_module.completion_evidence(kept))
    scoped_verdict.update(
        supported=True,
        scoped=True,
        scope={field: current.get(field) for field in SCOPE_FIELDS},
        raw_verified=bool(verdict["verified"]),
        rejected_records=len(records) - len(kept),
        scoped_records=len(kept),
        task_scoped=scoped_task,
    )
    return scoped_verdict


def finish_contract_rows(entries: Sequence[Mapping]) -> dict:
    """Every ``finish`` call in the chain with the contract fields it answered.

    The v1 contract makes ``finish`` answer ``blocked`` (with ``missing`` and
    ``missing_subtasks``) instead of terminating when the plan is incomplete or
    the confirmation is stale. The rows say what the call declared, what the
    scope was, and whether the episode really continued afterwards.
    """
    ordered = list(entries)
    rows: list[dict] = []
    for position, entry in enumerate(ordered):
        if str(entry.get("tool") or "") != "finish":
            continue
        data = entry.get("data") if isinstance(entry.get("data"), Mapping) else {}
        args = entry.get("args") if isinstance(entry.get("args"), Mapping) else {}
        after = ordered[position + 1:]
        after_tools: dict[str, int] = {}
        for item in after:
            name = str(item.get("tool") or "")
            if name:
                after_tools[name] = after_tools.get(name, 0) + 1
        declared = data.get("claimed")
        if declared is None:
            declared = args.get("success") is True
        rows.append({
            "seq": entry.get("seq"),
            "leg": entry.get("leg"),
            "status": str(entry.get("status") or ""),
            "declared_success": args.get("success"),
            "claimed": bool(declared),
            "claim": str(data.get("claim") or ""),
            "verified": data.get("verified"),
            "success": data.get("success"),
            "plan_complete": data.get("plan_complete"),
            "missing": list(data.get("missing") or []),
            "missing_subtasks": [
                dict(item) for item in data.get("missing_subtasks") or []
                if isinstance(item, Mapping)
            ],
            "scope": _scope_fields(entry),
            "blocked": str(entry.get("status") or "") == "blocked",
            "calls_after": len(after),
            "tools_after": after_tools,
            "continued_after_blocked": bool(
                str(entry.get("status") or "") == "blocked" and after
            ),
        })
    first_missing: list[dict] = []
    for row in rows:
        if row["missing_subtasks"]:
            first_missing = row["missing_subtasks"]
            break
    return {
        "rows": rows,
        "n": len(rows),
        "blocked": sum(1 for row in rows if row["blocked"]),
        "blocked_continued": sum(1 for row in rows if row["continued_after_blocked"]),
        "missing_subtasks": first_missing,
        "claims": [row for row in rows if row["claimed"]],
    }


def completion_claims(
    legs: Sequence[LegResult], entries: Sequence[Mapping], *, ordered_success: bool
) -> dict:
    """The episode's completion claim and whether the evaluation agrees.

    A claim is a ``finish`` that declared ``success: true`` (or the runner's
    final verdict). ``wrong`` is the goal.md "claimed done, not done" failure:
    the claim exists and the ordered full chain was not established.
    """
    claims: list[dict] = []
    for entry in entries:
        if str(entry.get("tool") or "") != "finish":
            continue
        data = entry.get("data") if isinstance(entry.get("data"), Mapping) else {}
        args = entry.get("args") if isinstance(entry.get("args"), Mapping) else {}
        declared = data.get("claimed")
        if declared is None:
            declared = args.get("success") is True
        if not declared:
            continue
        claims.append({
            "leg": entry.get("leg"),
            "seq": entry.get("seq"),
            "status": str(entry.get("status") or ""),
            "verified": data.get("verified"),
            "plan_complete": data.get("plan_complete"),
            "missing_subtasks": len(data.get("missing_subtasks") or []),
        })
    summary_claim = any(
        isinstance(leg.summary, Mapping) and leg.summary.get("success") is True
        for leg in legs
    )
    claimed = bool(claims) or summary_claim
    return {
        "claimed": claimed,
        "claims": claims,
        "n_claims": len(claims),
        "runner_claimed": summary_claim,
        "evaluated_complete": bool(ordered_success),
        "wrong": bool(claimed and not ordered_success),
    }


def _collect_candidates(payload: Any, table: dict[str, list[float]], _depth: int = 0) -> None:
    """``candidate_id -> place`` from every candidate the episode's results name."""
    if _depth > 6:
        return
    if isinstance(payload, Mapping):
        candidate_id = (
            payload.get("candidate_id")
            or payload.get("target_id")
            or payload.get("id")
        )
        point = None
        for key in ("position_flu", "position", "target_position", "candidate_position"):
            point = _as_position(payload.get(key))
            if point is not None:
                break
        if candidate_id not in (None, "") and point is not None:
            table.setdefault(str(candidate_id), [float(v) for v in point])
        for value in payload.values():
            _collect_candidates(value, table, _depth + 1)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            _collect_candidates(item, table, _depth + 1)


def instance_acceptance(
    entries: Sequence[Mapping], task: TaskSpec, mapping: Mapping[str, int]
) -> dict:
    """Arrivals/confirmations that accepted an object which is not the truth one.

    Every acceptance event is an arrival (``outcome=arrived`` on an object
    motion) or a passing ``verify_goal``. Its object instance is resolved from
    the candidate table the episode's own groundings built (the scope's
    ``candidate_id`` first, the result's place second) and compared with the
    stage place the call was scoped to. A gap beyond that stage's ``locate_m``
    is a wrong-instance acceptance; an event whose instance cannot be resolved
    is reported as unresolved and kept out of the rate.
    """
    table: dict[str, list[float]] = {}
    for entry in entries:
        _collect_candidates(entry.get("data") or {}, table)
    rows: list[dict] = []
    resolved = wrong = 0
    for entry in entries:
        tool = str(entry.get("tool") or "")
        data = entry.get("data") if isinstance(entry.get("data"), Mapping) else {}
        kind = ""
        if tool == "approach_candidate" and (
            str(data.get("outcome") or "") in ARRIVAL_OUTCOMES
            or data.get("arrived") is True
        ):
            kind = "arrival"
        elif tool == "verify_goal" and data.get("verified") is True:
            kind = "confirmation"
        elif tool == "confirm_candidate" and str(
            data.get("verdict") or ""
        ) == "accepted":
            # version 2: the acceptance event is the new-view confirmation,
            # not the arrival that shares its place with the grounding
            kind = "instance_confirmation"
        if not kind:
            continue
        scope = _scope_fields(entry)
        candidate_id = str(
            scope.get("candidate_id")
            or data.get("candidate_id")
            or data.get("target_id")
            or ""
        )
        point = table.get(candidate_id)
        if point is None:
            for key in ("target_position", "candidate_position", "position_flu"):
                point = _as_position(data.get(key))
                if point is not None:
                    break
        stage_index = entry_stage_index(entry, mapping)
        if point is None and stage_index is not None:
            stage = task.stages[stage_index - 1]
            if stage.position is not None:
                point = [float(v) for v in stage.position]
        stage = task.stages[stage_index - 1] if stage_index else None
        if point is None or stage is None or stage.position is None:
            rows.append({
                "kind": kind, "leg": entry.get("leg"), "seq": entry.get("seq"),
                "tool": tool, "candidate_id": candidate_id, "stage": None,
                "distance_m": None, "wrong": None, "resolved": False,
            })
            continue
        gap = _distance(point, stage.position)
        is_wrong = gap > float(stage.locate_m)
        resolved += 1
        wrong += int(is_wrong)
        rows.append({
            "kind": kind, "leg": entry.get("leg"), "seq": entry.get("seq"),
            "tool": tool, "candidate_id": candidate_id, "stage": stage.stage_id,
            "distance_m": round(gap, 3), "locate_m": float(stage.locate_m),
            "wrong": bool(is_wrong), "resolved": True,
        })
    return {
        "events": len(rows),
        "resolved": resolved,
        "wrong": wrong,
        "wrong_rate": _rate(wrong, resolved),
        "unresolved": len(rows) - resolved,
        "rows": rows,
    }


#: Tokens of a stage label that name its object/relation (the anchor check).
_TOKEN_SPLIT_RE = re.compile(r"[的旁里边附近在是与和，,。;；、\s]+")


def _label_tokens(*labels: str) -> list[str]:
    tokens: list[str] = []
    for label in labels:
        for token in _TOKEN_SPLIT_RE.split(str(label or "")):
            token = token.strip()
            if token and token not in tokens:
                tokens.append(token)
    return tokens


def _names_stage(text: str, stage: StageSpec) -> bool:
    """True when the text contains the stage's full label or step."""
    folded = text.casefold()
    return any(
        candidate and str(candidate).strip().casefold() in folded
        for candidate in (stage.label, stage.step)
    )


def _classify_regrounding(instruction: str, stage: StageSpec | None, task: TaskSpec) -> str:
    """Whether an automatic re-grounding asked for the current sub-goal or the task."""
    text = str(instruction or "").strip()
    if not text:
        return "empty"
    if stage is None:
        return "unknown_stage"
    global_text = str(task.instruction or "")
    if global_text and (
        text == global_text or text in global_text
        or (global_text in text and len(text) - len(global_text) < 40)
    ):
        return "global_instruction"
    if any(marker in text for marker in ("第一步", "第二步", "第三步")):
        return "global_instruction"
    named = [item for item in task.stages if _names_stage(text, item)]
    if len(named) >= 2 and len(text) >= len(global_text) * 0.5:
        return "global_instruction"
    has_stage = _names_stage(text, stage) or any(
        token in text for token in _label_tokens(stage.label, stage.step)
    )
    anchor = next((s for s in task.stages if s.stage_id == stage.anchor), None)
    anchor_tokens = _label_tokens(anchor.label, anchor.step) if anchor else []
    has_anchor = any(token in text for token in anchor_tokens) if anchor_tokens else False
    if has_stage and (has_anchor or not anchor_tokens):
        return "subgoal+anchor"
    if has_stage:
        return "subgoal"
    return "other"


#: Tools that turn a visible/remembered object into a candidate (a re-grounding).
REGROUND_TOOLS = ("ground_target", "search_target")


def _call_rows_for_regrounding(leg: LegResult) -> list[dict]:
    """The leg's calls in order, with a turn when the trace is available.

    A finished leg has ``tool_trace.jsonl`` (one row per decision turn); an
    interrupted leg has none, and its ledger's outer entries are the same calls.
    """
    trace = read_trace(leg.out_dir)
    if trace:
        return [
            {
                "turn": int(item.get("turn") or 0),
                "tool": str(item.get("tool") or ""),
                "status": str(item.get("status") or ""),
                "args": dict(item.get("args") or {}),
                "data": dict(item.get("data") or {}),
                "evidence": list(item.get("evidence") or []),
            }
            for item in trace
        ]
    return [
        {
            "turn": int(entry.get("seq") or 0),
            "tool": str(entry.get("tool") or ""),
            "status": str(entry.get("status") or ""),
            "args": dict(entry.get("args") or {}),
            "data": dict(entry.get("data") or {}),
            "evidence": list(entry.get("evidence") or []),
        }
        for entry in leg.entries
        if entry.get("outer")
    ]


def recovery_regrounding(legs: Sequence[LegResult], task: TaskSpec) -> dict:
    """What a loop-protected recovery asked the next re-grounding to look for.

    goal.md v1 (seed23): a recovery must re-locate the *current sub-task* and its
    relation anchor, not restate the three-stage instruction. Two sources are
    paired with the ``ground_target``/``search_target`` the episode really
    executed:

    - ``recovery``: the runtime's stagnation advice (``EpisodeLog.recoveries``);
    - ``next_action``: a failed/blocked movement whose own answer names a
      re-grounding call (``_ground_again_action``), which is the path the
      seed23 loop protection takes.

    The instruction of the executed call is classified as ``subgoal+anchor``
    (current stage plus its relation anchor), ``subgoal``, ``global_instruction``
    (the whole task restated) or ``other``.
    """
    rows: list[dict] = []
    mapping = stage_subtask_map(task, legs)
    for leg in legs:
        calls = _call_rows_for_regrounding(leg)
        by_turn = {call["turn"]: call for call in calls}
        seen: set[tuple] = set()

        def follow_for(index: int, wanted: str) -> dict | None:
            for item in calls[index + 1:index + 4]:
                if item["tool"] == wanted:
                    return item
            for item in calls[index + 1:]:
                if item["tool"] in REGROUND_TOOLS:
                    return item
            return None

        def add(source: str, index: int, stalled: dict, advice: list[str]) -> None:
            scope = _scope_fields(stalled)
            stage_index = (
                entry_stage_index(stalled, mapping)
                if scope.get("subtask_id") else None
            )
            stage = task.stages[stage_index - 1] if stage_index else None
            next_action = stalled["data"].get("next_action")
            wanted = (
                str(next_action.get("tool") or "")
                if isinstance(next_action, Mapping) else ""
            )
            if wanted in REGROUND_TOOLS:
                follow = follow_for(index, wanted)
            else:
                follow = follow_for(index, "ground_target") or follow_for(index, "search_target")
            instruction = ""
            if follow is not None:
                instruction = str(follow["args"].get("instruction") or "")
            key = (
                int(stalled["turn"]), follow["turn"] if follow is not None else None,
                source,
            )
            if key in seen:
                return
            seen.add(key)
            rows.append({
                "leg": leg.plan.label,
                "source": source,
                "turn": int(stalled["turn"]),
                "stalled_tool": str(stalled["tool"] or ""),
                "stalled_status": str(stalled["status"] or ""),
                "stage": None if stage is None else stage.stage_id,
                "advice": [str(item) for item in advice],
                "executed": follow is not None,
                "follow_tool": "" if follow is None else follow["tool"],
                "follow_turn": None if follow is None else follow["turn"],
                "instruction": instruction,
                "classification": _classify_regrounding(instruction, stage, task),
                "new_evidence": bool(follow["evidence"]) if follow is not None else False,
            })

        recovery_turns: dict[int, dict] = {}
        for record in leg.log.get("recoveries") or []:
            if not isinstance(record, Mapping):
                continue
            turn = int(record.get("turn") or 0)
            recovery_turns[turn] = dict(record)
            stalled = by_turn.get(turn) or {"turn": turn, "tool": "", "status": "",
                                           "args": {}, "data": {}, "evidence": []}
            index = next(
                (position for position, call in enumerate(calls) if call["turn"] == turn),
                -1,
            )
            if index >= 0:
                add("recovery", index, stalled, list(record.get("advice") or []))
        for index, call in enumerate(calls):
            if call["turn"] in recovery_turns:
                continue  # already reported by the runtime's own recovery event
            next_action = call["data"].get("next_action")
            if not isinstance(next_action, Mapping):
                continue
            if str(next_action.get("tool") or "") not in REGROUND_TOOLS:
                continue
            add("next_action", index, call, [])
    subgoal = sum(
        1 for row in rows if row["classification"] in ("subgoal", "subgoal+anchor")
    )
    global_rows = sum(1 for row in rows if row["classification"] == "global_instruction")
    return {
        "events": len(rows),
        "executed": sum(1 for row in rows if row["executed"]),
        "subgoal_scoped": subgoal,
        "global_instruction": global_rows,
        "with_anchor": sum(1 for row in rows if row["classification"] == "subgoal+anchor"),
        "by_source": {
            source: sum(1 for row in rows if row["source"] == source)
            for source in ("recovery", "next_action")
        },
        "rows": rows,
    }


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
    # the runtime's live context is the checkpoint source: its plan/sub-task
    # statuses, confirmation ledger, memory references, scope and beliefs are
    # saved exactly as the leg left them (goal.md version 3)
    runtime_ctx = getattr(captured[-1], "ctx", None) if captured else None
    checkpoint = build_runtime_checkpoint(plan, world, ctx=runtime_ctx)
    with open(out_dir / "checkpoint.json", "w") as handle:
        json.dump(jsonable(checkpoint), handle, indent=2, default=str)

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
        checkpoint=checkpoint,
        final_plan=copy.deepcopy(checkpoint.get("plan"))
        if isinstance(checkpoint.get("plan"), Mapping) else None,
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
        "checkpoint_path": str(out_dir / "checkpoint.json"),
        "checkpoint_fields": sorted(checkpoint),
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


def _entries_for_stage(
    entries: Sequence[dict],
    stage: StageSpec,
    mapping: Mapping[str, int],
) -> tuple[list[dict], int, int]:
    """``(entries, excluded, unscoped)`` of one stage under its task scope.

    A tool result counts for a stage only when the scope it was produced under
    names that stage's sub-task. This is the offline half of the v1 contract:
    a grounding made while another sub-task was executing cannot complete this
    stage (goal.md: "candidate, arrival and confirmation must belong to the same
    sub-task"). Entries without any scope (pre-v1 artifacts) are excluded from a
    scoped verdict and counted, never silently reused.
    """
    if not mapping:
        return list(entries), 0, 0
    selected: list[dict] = []
    excluded = 0
    unscoped = 0
    for entry in entries:
        scope = entry_scope(entry)
        if scope is None:
            unscoped += 1
            continue
        if entry_stage_index(entry, mapping) == stage.index:
            selected.append(entry)
        else:
            excluded += 1
    return selected, excluded, unscoped


def _stage_confirmation(
    stage_entries: Sequence[Mapping], stage: StageSpec,
) -> dict:
    """The stage's accepted identity, under its current candidate/revision.

    Version 2: reach + locate prove *where* the drone went, not *what* it
    found. The stage is identity-complete only with an ``accepted``
    ``confirm_candidate`` for the current scope, and - when the candidate's
    position is resolvable - that instance within the stage's ``locate_m``.
    A confirmation of an earlier candidate/revision is stale and does not
    count, unless the new candidate is within
    :data:`CONFIRMATION_INHERIT_RADIUS_M` of the confirmed one (the same
    instance re-grounded under a fresh id; reported in
    ``confirmation_inherited_from``).
    """
    current_candidate = ""
    current_revision: int | None = None
    for entry in stage_entries:
        scope = entry_scope(entry)
        if scope is None:
            continue
        candidate = str(scope.get("candidate_id") or "")
        if candidate:
            current_candidate = candidate
        try:
            current_revision = int(scope.get("evidence_revision"))
        except (TypeError, ValueError):
            current_revision = None
    accepted: list[Mapping] = []
    table: dict[str, list[float]] = {}
    for entry in stage_entries:
        _collect_candidates(entry.get("data") or {}, table)
    inherited_from: list[str] = []
    for entry in stage_entries:
        data = entry.get("data") if isinstance(entry.get("data"), Mapping) else {}
        if str(entry.get("tool") or "") != "confirm_candidate":
            continue
        if str(data.get("verdict") or "") != "accepted":
            continue
        scope = entry_scope(entry) or {}
        candidate = str(scope.get("candidate_id") or "")
        inherited = False
        if current_candidate and candidate and candidate != current_candidate:
            old_point = table.get(candidate)
            new_point = table.get(current_candidate)
            if old_point is not None and new_point is not None:
                inherited = (
                    _distance(old_point, new_point) <= CONFIRMATION_INHERIT_RADIUS_M
                )
            if not inherited:
                # an acceptance of another, unrelated instance is stale
                continue
            if candidate not in inherited_from:
                inherited_from.append(candidate)
        if not inherited:
            try:
                revision = int(scope.get("evidence_revision"))
            except (TypeError, ValueError):
                revision = None
            if (
                current_revision is not None
                and revision is not None
                and revision != current_revision
            ):
                continue
        accepted.append(entry)
    point = table.get(current_candidate) if current_candidate else None
    distance = None
    if point is not None and stage.position is not None:
        distance = _distance(point, stage.position)
    confirmed = bool(accepted)
    correct = None
    if distance is not None:
        correct = bool(distance <= float(stage.locate_m))
    latest = accepted[-1] if accepted else None
    return {
        "confirmation_required": True,
        "confirmed": confirmed,
        "confirmation_correct": correct,
        "identity_complete": bool(confirmed and correct is True),
        "confirmation_candidate_id": current_candidate or None,
        "confirmation_distance_m": None if distance is None else round(distance, 3),
        "confirmation_inherited_from": inherited_from,
        "confirmed_seq": None if latest is None else latest.get("seq"),
        "confirmed_order": None if latest is None else order_of(latest),
        "confirmation_count": len(accepted),
    }


def stage_report(
    entries: Sequence[dict],
    task: TaskSpec,
    *,
    poses: Sequence[Sequence[float]] = (),
    subtask_map: Mapping[str, int] | None = None,
    scoped: bool | None = None,
) -> dict:
    """Per-stage reach/locate/complete verdicts, from scoped evidence only.

    A stage is *reached* when the executor carried the drone inside ``reach_m``
    of the stage's place **or** on the current candidate's standoff shell
    (``standoff_m + CANDIDATE_REACH_TOLERANCE_M``); ``reach_basis`` says which
    geometry held, and *located* when a tool result carried a place inside
    ``locate_m`` of it. Version 2 adds the identity half: a scoped stage is
    complete only when it also carries an ``accepted`` ``confirm_candidate`` for
    the *current* candidate/revision, and the confirmed candidate resolves
    inside that stage's ``locate_m`` (``identity_complete``). Reach + locate
    alone is the version-1 verdict and is reported as ``complete=False`` for a
    scoped report. Nothing here reads the model's prose: a plan step, a
    ``finish`` reason or a recovery note cannot complete a stage.

    The typed criterion (:meth:`StageSpec.effective_done_when_type`) decides
    which halves are required: ``sighted``/``located`` stages complete on the
    located half alone (a landmark the mission only asks to find must not be
    charged for a flight), ``verified`` needs the accepted confirmation, and
    ``arrived`` keeps the reach + locate + identity contract. ``reach_attempts``
    counts the movement calls the stage's own sub-task made, whether or not the
    stage needed them - the number that says whether a semantic criterion saved
    motion.

    The v1 scope contract narrows both halves to the sub-task that stage was
    executing: an observation, candidate or arrival produced while another
    sub-task was current does not count for this one. The mode is automatic -
    a ledger whose results carry ``data["scope"]`` is judged scoped, a pre-v1
    ledger (and every synthetic fixture that predates the contract) keeps the
    historical unscoped verdict, reported under ``scope.active = false``.
    """
    scope_entries = sum(1 for entry in entries if entry_scope(entry) is not None)
    if scoped is None:
        scoped = bool(subtask_map) and scope_entries > 0
    mapping = dict(subtask_map or {f"st_{stage.index}": stage.index for stage in task.stages})
    external_poses: list[tuple[float, float, float]] = []
    if not scoped:
        for item in poses:
            point = _as_position(item)
            if point is not None:
                external_poses.append(point)
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
            "done_when_type": stage.effective_done_when_type(),
            "reach_required": stage.effective_done_when_type() == DONE_WHEN_ARRIVED,
            "reach": False,
            "reach_order": None,
            "reach_distance_m": None,
            "reach_basis": None,
            "reach_candidate_id": None,
            "reach_standoff_m": None,
            "reach_candidate_distance_m": None,
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
            "scoped_entries": 0,
            "scope_excluded_entries": 0,
        }
        if scoped:
            stage_entries, excluded, _unscoped = _entries_for_stage(entries, stage, mapping)
        else:
            stage_entries, excluded = list(entries), 0
        row["scoped_entries"] = len(stage_entries)
        row["scope_excluded_entries"] = excluded
        movement_entries = [
            entry for entry in stage_entries if str(entry.get("tool") or "") in ACTION_TOOLS
        ]
        row["reach_attempts"] = len(movement_entries)
        row["reach_tools"] = sorted({str(entry.get("tool") or "") for entry in movement_entries})
        if scoped:
            row.update(_stage_confirmation(stage_entries, stage))
        else:
            row.update(
                confirmation_required=False,
                confirmed=None,
                confirmation_correct=None,
                identity_complete=None,
                confirmation_candidate_id=None,
                confirmation_distance_m=None,
                confirmation_inherited_from=[],
                confirmed_seq=None,
                confirmed_order=None,
                confirmation_count=0,
            )
        if stage.position is not None:
            best_reach: tuple[float, int | None, str, str, float | None] | None = None
            for point in external_poses:
                gap = _distance(point, stage.position)
                if gap <= stage.reach_m and (best_reach is None or gap < best_reach[0]):
                    best_reach = (gap, None, "center", "", None)
            for entry in stage_entries:
                # both ends of every call: a leg that carried the drone *through*
                # the stage's place and on to somewhere else still reached it
                for value in (entry.get("pose_before"), entry.get("pose_after")):
                    pose_here = _as_position(value)
                    if pose_here is None:
                        continue
                    gap = _distance(pose_here, stage.position)
                    if gap <= stage.reach_m and (best_reach is None or gap < best_reach[0]):
                        best_reach = (gap, order_of(entry), "center", "", None)
            if scoped:
                # the second, wider basis: the arrival is on the current
                # candidate's standoff shell even when the stage centre is
                # outside reach_m (the approach stops standoff_m short of the
                # object, not at the stage's declared place)
                candidate_table: dict[str, list[float]] = {}
                for entry in stage_entries:
                    _collect_candidates(entry.get("data") or {}, candidate_table)
                candidate_reach: (
                    tuple[float, int, str, str, float] | None
                ) = None
                for entry in stage_entries:
                    scope = entry_scope(entry) or {}
                    candidate = str(scope.get("candidate_id") or "")
                    position = candidate_table.get(candidate) if candidate else None
                    if position is None:
                        continue
                    data = entry.get("data") if isinstance(entry.get("data"), Mapping) else {}
                    standoff = data.get("standoff_m")
                    if standoff is None and str(entry.get("tool") or "") == "approach_candidate":
                        standoff = DEFAULT_APPROACH_STANDOFF_M
                    try:
                        standoff = float(standoff)
                    except (TypeError, ValueError):
                        continue
                    limit = max(0.0, standoff) + CANDIDATE_REACH_TOLERANCE_M
                    for value in (entry.get("pose_before"), entry.get("pose_after")):
                        pose_here = _as_position(value)
                        if pose_here is None:
                            continue
                        gap = _distance(pose_here, position)
                        if gap <= limit and (
                            candidate_reach is None or gap < candidate_reach[0]
                        ):
                            candidate_reach = (
                                gap, order_of(entry), candidate, standoff, gap
                            )
                if candidate_reach is not None:
                    # prefer the candidate basis when both hold: it is the
                    # geometry the approach actually flew to
                    best_reach = (
                        candidate_reach[0],
                        candidate_reach[1],
                        "candidate",
                        candidate_reach[2],
                        candidate_reach[3],
                    )
                    row["reach_candidate_distance_m"] = round(candidate_reach[4], 3)
            if best_reach is not None:
                row["reach"] = True
                row["reach_distance_m"] = round(best_reach[0], 3)
                row["reach_order"] = best_reach[1]
                row["reach_basis"] = best_reach[2]
                row["reach_candidate_id"] = best_reach[3] or None
                row["reach_standoff_m"] = best_reach[4]
            # how close *any* located reference got, inside the radius or not: a
            # stage the agent never established still says whether it looked at
            # the right place at all (a candidate 3 m away is a different object)
            closest: tuple[float, tuple[float, float, float], str] | None = None
            for entry in stage_entries:
                for point in positions_in(entry.get("data") or {}):
                    gap = _distance(point, stage.position)
                    if closest is None or gap < closest[0]:
                        closest = (gap, point, str(entry.get("tool") or ""))
            if closest is not None:
                row["closest_place_distance_m"] = round(closest[0], 3)
                row["closest_place_flu"] = [float(v) for v in closest[1]]
                row["closest_place_tool"] = closest[2]
            located = _scan_entries(stage_entries, stage)
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
        stage_type = stage.effective_done_when_type()
        if stage_type in (DONE_WHEN_SIGHTED, DONE_WHEN_LOCATED):
            # a landmark that only has to be found is complete on a located
            # sighting: no arrival and no identity confirmation are asked for
            row["complete"] = bool(row["located"])
        elif stage_type == DONE_WHEN_VERIFIED:
            row["complete"] = bool(
                row["located"] and row["identity_complete"]
            ) if scoped else bool(row["located"])
        elif scoped:
            # identity-complete: an arrival at the candidate and a located place
            # are only the version-1 half; the accepted confirmation of the
            # *current* instance is the second half.
            row["complete"] = bool(
                row["reach"] and row["located"] and row["identity_complete"]
            )
        else:
            row["complete"] = bool(row["reach"] and row["located"])
        if row["complete"]:
            row["complete_order"] = max(
                [
                    value for value in (
                        row["reach_order"], row["located_order"],
                        row.get("confirmed_order"),
                    )
                    if value
                ]
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
        "confirmation_required": bool(scoped),
        "confirmed_count": sum(
            1 for row in stages.values() if row.get("confirmed") is True
        ),
        "identity_complete_count": sum(
            1 for row in stages.values() if row.get("identity_complete") is True
        ),
        "identity_complete_ids": [
            task.stages[row["index"] - 1].stage_id
            for row in stages.values()
            if row.get("identity_complete") is True
        ],
        "complete_count": len(complete),
        "complete_ids": [task.stages[i - 1].stage_id for i in complete],
        "complete_orders": complete_orders,
        "complete_by_type": {
            kind: sum(
                1
                for row in stages.values()
                if row.get("done_when_type") == kind and row["complete"]
            )
            for kind in DONE_WHEN_TYPES
        },
        "stages_by_type": {
            kind: sum(
                1 for row in stages.values() if row.get("done_when_type") == kind
            )
            for kind in DONE_WHEN_TYPES
        },
        "reach_attempts": sum(int(row.get("reach_attempts") or 0) for row in stages.values()),
        "reach_attempts_by_type": {
            kind: sum(
                int(row.get("reach_attempts") or 0)
                for row in stages.values()
                if row.get("done_when_type") == kind
            )
            for kind in DONE_WHEN_TYPES
        },
        "order_ok": bool(order_ok),
        "chain_complete": len(complete) == len(task.stages),
        "relation_ok": all(
            row["relation_ok"] is not False for row in stages.values()
        ),
        "scope": {
            "active": bool(scoped),
            "entries": len(entries),
            "scoped_entries": scope_entries,
            "unscoped_entries": len(entries) - scope_entries,
            "mapping": {str(key): int(value) for key, value in mapping.items()},
            "unmapped_stages": [
                stage.stage_id for stage in task.stages
                if not any(index == stage.index for index in mapping.values())
            ],
        },
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
RESUME_CHANNELS = ("pose", "plan", "subgoal", "brief", "checkpoint")

#: Version of the runtime checkpoint (the agent-side state, saved verbatim).
CHECKPOINT_VERSION = 3
#: ``ctx.extra`` keys the checkpoint carries and the resumed leg restores.
CHECKPOINT_SOURCES = (
    "confirmation_ledger",
    "perception.candidates",
    "observations",
    "viewpoints_by_id",
    "hypotheses",
    "task_scope",
    "beliefs",
    "open_questions",
    "evidence_pack",
    "last_observation_id",
)


def _checkpoint_extra(ctx: Any, plan: LegPlan | None) -> Mapping:
    """The live ``ctx.extra`` of a finished/interrupted leg (or a safe empty)."""
    extra = getattr(ctx, "extra", None) if ctx is not None else None
    if isinstance(extra, Mapping):
        return extra
    if plan is not None and isinstance(plan.ctx_extra, Mapping):
        return plan.ctx_extra
    return {}


def build_runtime_checkpoint(
    plan: LegPlan | None,
    world: Any,
    *,
    ctx: Any = None,
) -> dict:
    """Save the runtime state **verbatim** (goal.md version 3).

    Only the agent's own state travels: the plan record with its sub-task
    statuses, the confirmation ledger, candidate/observation/viewpoint memory
    references, the task scope, beliefs/hypotheses, open questions and the
    evidence pack. The scene's ground truth is never read here - a wrong belief
    (a sub-task marked done that the evaluator would not accept) is saved as the
    agent left it, so the resumed leg has to discover and correct it itself.
    """
    extra = _checkpoint_extra(ctx, plan)
    position, yaw = world_pose(world)
    raw_plan = extra.get("plan")
    if not isinstance(raw_plan, Mapping) and plan is not None:
        candidate_plan = (plan.ctx_extra or {}).get("plan")
        raw_plan = candidate_plan if isinstance(candidate_plan, Mapping) else None
    raw_plan = jsonable(copy.deepcopy(dict(raw_plan))) if isinstance(raw_plan, Mapping) else None
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "source": "runtime.ctx.extra",
        "source_leg": "" if plan is None else str(plan.label),
        "source_episode_id": str(
            getattr(ctx, "episode_id", "") if ctx is not None else ""
        ),
        "saved_at_s": round(time.time(), 3),
        "instruction": str(
            extra.get("instruction")
            or ("" if plan is None else plan.instruction)
            or ""
        ),
        "pose": {
            "position_flu": [float(v) for v in position],
            "yaw_rad": float(yaw or 0.0),
        },
        "plan": raw_plan,
        "confirmation_ledger": {},
        "candidates": {},
        "observations": {},
        "viewpoints_by_id": {},
        "hypotheses": {},
        "task_scope": {},
        "beliefs": {},
        "open_questions": None,
        "evidence_pack": {},
        "last_observation_id": "",
    }
    for key in CHECKPOINT_SOURCES:
        value = extra.get(key)
        if value is None:
            continue
        checkpoint[key] = copy.deepcopy(jsonable(value))
    if not checkpoint["beliefs"] and checkpoint["hypotheses"]:
        checkpoint["beliefs"] = copy.deepcopy(checkpoint["hypotheses"])
    return jsonable(checkpoint)


def _leg_checkpoint(leg: LegResult) -> dict:
    """The checkpoint the resume path consumes; never a stage verdict.

    A leg that carries a runtime checkpoint uses it verbatim. For an artifact
    written before version 3 the fallback carries the leg's *starting* plan and
    pose only - it makes no completion claim and reads no ground truth, so the
    old artifact degrades to "nothing established" instead of being handed an
    evaluator-judged resume state.
    """
    checkpoint = getattr(leg, "checkpoint", None)
    if isinstance(checkpoint, Mapping) and checkpoint:
        return dict(checkpoint)
    start_plan = (leg.plan.ctx_extra or {}).get("plan")
    pose = _as_position(leg.entries[-1].get("pose_after")) if leg.entries else None
    pose = pose or _as_position(leg.start_pose) or [0.0, 0.0, 0.0]
    yaw = float(leg.entries[-1].get("yaw_after") or 0.0) if leg.entries else 0.0
    return {
        "version": CHECKPOINT_VERSION,
        "source": "leg_metadata",
        "source_leg": leg.plan.label,
        "source_episode_id": leg.plan.prompt_episode_id,
        "instruction": leg.plan.instruction,
        "pose": {"position_flu": [float(v) for v in pose], "yaw_rad": yaw},
        "plan": copy.deepcopy(jsonable(start_plan)) if isinstance(start_plan, Mapping) else None,
        "confirmation_ledger": {},
        "candidates": {},
        "observations": {},
        "viewpoints_by_id": {},
        "hypotheses": {},
        "task_scope": {},
        "beliefs": {},
        "open_questions": None,
        "evidence_pack": {},
        "last_observation_id": "",
    }


def _subtask_number(subtask: Mapping, position: int) -> int:
    """The stage index a sub-task id names (``st_2`` -> 2), else its position."""
    text = str(subtask.get("id") or "")
    match = re.search(r"(\d+)\s*$", text)
    return int(match.group(1)) if match else int(position)


def _plan_belief(plan: Mapping | None) -> tuple[list[int], list[int]]:
    """``(done, claimed)`` stage indexes the checkpoint's own plan records."""
    done: list[int] = []
    claimed: list[int] = []
    subtasks = (plan or {}).get("subtasks") or []
    for position, subtask in enumerate(subtasks, 1):
        if not isinstance(subtask, Mapping):
            continue
        index = _subtask_number(subtask, position)
        status = str(subtask.get("status") or "")
        if status == "done":
            done.append(index)
        elif status in ("claimed_done", "blocked", "failed"):
            claimed.append(index)
    if not subtasks:
        for value in (plan or {}).get("done") or []:
            try:
                done.append(int(value))
            except (TypeError, ValueError):
                continue
    return sorted(set(done)), sorted(set(claimed))


def _checkpoint_candidates(checkpoint: Mapping) -> dict:
    """Candidate records the checkpoint itself carries.

    The runtime keeps its candidate table under ``perception.candidates`` (a
    literal dotted ``ctx.extra`` key) while older fixtures and the fallback
    checkpoint use ``candidates``. The brief and the checkpoint score read both
    and prefer the runtime table when an id appears in both.
    """
    merged: dict[str, Mapping] = {}
    for key in ("candidates", "perception.candidates"):
        table = checkpoint.get(key)
        if isinstance(table, Mapping):
            for candidate_id, record in table.items():
                if isinstance(record, Mapping):
                    merged[str(candidate_id)] = record
    perception = checkpoint.get("perception")
    if isinstance(perception, Mapping) and isinstance(perception.get("candidates"), Mapping):
        for candidate_id, record in perception["candidates"].items():
            if isinstance(record, Mapping):
                merged.setdefault(str(candidate_id), record)
    return merged


def _checkpoint_candidate_table(checkpoint: Mapping, entries: Sequence[Mapping]) -> dict[str, list[float]]:
    """``candidate_id -> place`` from the whole leg plus the checkpoint.

    Stage-scoped confirmation rows can only resolve candidates proposed while
    that stage was current. A candidate proposed under another stage (v10
    seed23 accepted the landmark candidate while executing the object stage)
    is still the agent's own record, so the checkpoint score resolves it from
    every entry of the leg - the same table ``instance_acceptance`` builds.
    """
    table: dict[str, list[float]] = {}
    for entry in entries:
        _collect_candidates(entry.get("data") or {}, table)
    for candidate_id, record in _checkpoint_candidates(checkpoint).items():
        point = _as_position(record.get("position_flu"))
        if point is not None:
            table.setdefault(candidate_id, [float(v) for v in point])
    return table


def _plan_claimed_done(plan: Mapping | None) -> list[int]:
    """Stage indexes the plan marks ``claimed_done`` (a completion claim)."""
    claimed: list[int] = []
    for position, subtask in enumerate((plan or {}).get("subtasks") or [], 1):
        if isinstance(subtask, Mapping) and str(subtask.get("status") or "") == "claimed_done":
            claimed.append(_subtask_number(subtask, position))
    return sorted(set(claimed))


def _checkpoint_payload(checkpoint: Mapping) -> dict:
    return {
        key: checkpoint.get(key)
        for key in (
            "candidates",
            "perception.candidates",
            "observations",
            "viewpoints_by_id",
            "hypotheses",
            "evidence_pack",
            "plan",
        )
        if checkpoint.get(key)
    }


def resume_brief(leg: LegResult, task: TaskSpec | None = None, *, limit: int = 1200) -> dict:
    """The instruction suffix of a resumed leg, built from its checkpoint only.

    Nothing here compares the agent's records with the scene: the plan statuses,
    confirmation ledger and places are reproduced exactly as the interrupted leg
    left them. A wrong "done" therefore survives the restart - the resumed leg is
    the one that has to notice and correct it (the offline evaluator scores that
    separately).
    """
    checkpoint = _leg_checkpoint(leg)
    plan = checkpoint.get("plan") if isinstance(checkpoint.get("plan"), Mapping) else {}
    subtasks = plan.get("subtasks") or []
    lines = [
        "[SESSION RESUME] This episode was interrupted before the task was done "
        f"({leg.plan.interrupt_mode or 'interrupt'}: {leg.interrupt_reason or 'leg ended early'}).",
        "These are YOUR OWN records from the interrupted leg, loaded from its "
        "checkpoint without any evaluator correction:",
    ]
    if subtasks:
        for position, subtask in enumerate(subtasks, 1):
            if not isinstance(subtask, Mapping):
                continue
            index = _subtask_number(subtask, position)
            status = str(subtask.get("status") or "pending")
            goal = str(subtask.get("goal") or "")
            detail = f"sub-task {subtask.get('id') or f'st_{index}'} (stage {index}): {status}"
            if goal:
                detail += f" — {goal}"
            evidence = [str(item) for item in subtask.get("evidence") or () if item]
            if evidence:
                detail += "; evidence: " + ", ".join(evidence[:4])
            if status != "done":
                detail += " (NOT done)"
            lines.append(f"  - {detail}")
    elif plan.get("steps"):
        for index, step in enumerate(plan.get("steps") or [], 1):
            mark = "done" if index in (plan.get("done") or []) else "open"
            lines.append(f"  - step {index} (stage {index}): {mark} — {step}")
    else:
        lines.append("  - no plan was recorded before the interruption")
    ledger = checkpoint.get("confirmation_ledger") or {}
    if isinstance(ledger, Mapping) and ledger:
        lines.append("Your confirmation ledger at interruption:")
        for candidate_id, entry in ledger.items():
            if not isinstance(entry, Mapping):
                continue
            status = str(entry.get("last_verdict") or "unconfirmed")
            revision = entry.get("accepted_revision", entry.get("revision"))
            detail = f"  - {candidate_id}: {status}"
            if revision is not None:
                detail += f" (revision {revision})"
            missing = [str(item) for item in entry.get("missing") or () if item]
            if missing:
                detail += "; missing: " + ", ".join(missing)
            lines.append(detail)
    candidates = _checkpoint_candidates(checkpoint)
    places: list[list[float]] = []
    if candidates:
        lines.append("Candidates you recorded yourself:")
        for candidate_id, record in candidates.items():
            place = _as_position(record.get("position_flu"))
            detail = f"  - {candidate_id}"
            if place is not None:
                detail += " at (" + ", ".join(f"{float(v):.2f}" for v in place) + ")"
                places.append([float(v) for v in place])
            status = str(record.get("status") or record.get("confirmation_status") or "")
            if status:
                detail += f" [{status}]"
            lines.append(detail)
    for key in ("observations", "viewpoints_by_id"):
        table = checkpoint.get(key) or {}
        if not isinstance(table, Mapping):
            continue
        for item_id, record in table.items():
            if not isinstance(record, Mapping):
                continue
            place = _as_position(record.get("position_flu"))
            if place is not None:
                places.append([float(place[0]), float(place[1]), float(place[2])])
    open_questions = checkpoint.get("open_questions")
    if open_questions:
        lines.append(f"Your open questions: {open_questions}")
    pose = _as_position((checkpoint.get("pose") or {}).get("position_flu"))
    if pose is None:
        pose = _as_position(leg.entries[-1].get("pose_after")) if leg.entries else None
    pose = pose or _as_position(leg.start_pose)
    pose_out: list[float] | None = None
    if pose is not None:
        yaw_deg = math.degrees(float((checkpoint.get("pose") or {}).get("yaw_rad") or 0.0)) \
            if checkpoint.get("pose") else 0.0
        pose_out = [float(v) for v in pose]
        lines.append(
            "Your drone is at (" + ", ".join(f"{float(v):.2f}" for v in pose)
            + f"), heading {yaw_deg:.0f} deg. Continue from these records. If you now "
            "doubt a record (a 'done' you cannot back with evidence, a weak "
            "confirmation), re-check it yourself with observe / compare_candidates / "
            "confirm_candidate instead of assuming it was corrected."
        )
    # every place named in the brief is one the agent itself recorded
    for item in positions_in(_checkpoint_payload(checkpoint)):
        place = [float(item[0]), float(item[1]), float(item[2])]
        if place not in places:
            places.append(place)
    return {"text": "\n".join(lines)[:limit], "places": places, "pose": pose_out}


def _restore_extras(checkpoint: Mapping) -> dict:
    """``ctx.extra`` entries the resumed leg inherits verbatim."""
    restore: dict[str, Any] = {}
    for key in CHECKPOINT_SOURCES:
        value = checkpoint.get(key)
        if value in (None, {}, [], ""):
            continue
        restore[key] = copy.deepcopy(value)
    return restore


def build_resume_state(
    leg: LegResult,
    task: TaskSpec,
    *,
    prompt_episode_id: str,
    world: Any = None,
    base_instruction: str | None = None,
) -> dict:
    """The inherited state of a resumed leg, consumed from its checkpoint.

    Version 3: the plan record (with its sub-task statuses and evidence), the
    confirmation ledger, the memory references, the scope and the beliefs are
    copied **verbatim** from ``leg.checkpoint``. No ``stage_report`` (or any
    other ground-truth verdict) is consulted, so a wrong belief - a sub-task the
    checkpoint marks done that the scene would not accept - is restored as the
    agent's own belief, not silently corrected.

    ``base_instruction`` is the instruction without the brief; the plan record's
    own ``instruction`` is left untouched (the checkpoint is the authority).
    """
    checkpoint = _leg_checkpoint(leg)
    plan = checkpoint.get("plan")
    plan = copy.deepcopy(dict(plan)) if isinstance(plan, Mapping) else None
    done, claimed = _plan_belief(plan)
    total = len(task.stages) if task is not None else len((plan or {}).get("subtasks") or [])
    current = min([index for index in range(1, total + 1) if index not in done] or [total or 1])
    brief = resume_brief(leg, task)
    base = str(base_instruction if base_instruction is not None else leg.plan.instruction)
    resumed_instruction = f"{base}\n\n{brief['text']}" if brief["text"] else base
    # the record the runtime *runs*: identical to the checkpoint except for the
    # instruction the module's resume guard compares (``state["plan"]`` stays
    # verbatim; this copy is bookkeeping, not a verdict)
    run_plan = copy.deepcopy(plan) if isinstance(plan, Mapping) else None
    if isinstance(run_plan, dict):
        statuses = [
            str(item.get("status") or "")
            for item in run_plan.get("subtasks") or []
            if isinstance(item, Mapping)
        ]
        if not statuses or all(status == "done" for status in statuses):
            # a legacy plan without sub-tasks, or every stage (wrongly) marked
            # done: resume_plan would refuse it, so make it the current
            # episode's record - the belief must survive, and the agent has to
            # overturn it
            if prompt_episode_id:
                run_plan["episode_id"] = str(prompt_episode_id)
        else:
            # a partial plan is adopted through the module's own resume_plan
            # (it logs plan_resumed and keeps the statuses/evidence unchanged)
            source_leg = str(checkpoint.get("source_leg") or "")
            if source_leg and prompt_episode_id:
                run_plan["episode_id"] = source_leg
        if str(prompt_episode_id or "") and str(
            run_plan.get("episode_id") or ""
        ) not in ("", str(prompt_episode_id or "")):
            run_plan["instruction"] = resumed_instruction[:160]
    subtasks = (plan or {}).get("subtasks") or []
    next_goal = ""
    if 1 <= current <= len(subtasks) and isinstance(subtasks[current - 1], Mapping):
        next_goal = str(subtasks[current - 1].get("goal") or "")
    subgoal = (
        f"resumed from your own checkpoint; done: {done}; next: stage {current}"
        + (f" ({next_goal})" if next_goal else "")
    )
    checkpoint_pose = checkpoint.get("pose") if isinstance(checkpoint.get("pose"), Mapping) else {}
    position = _as_position(checkpoint_pose.get("position_flu"))
    if position is None:
        position = _as_position(leg.entries[-1].get("pose_after")) if leg.entries else None
    position = position or _as_position(leg.start_pose) or [0.0, 0.0, 0.0]
    yaw = float(checkpoint_pose.get("yaw_rad") or 0.0)
    return {
        "pose": {"position_flu": [float(v) for v in position], "yaw_rad": yaw},
        "plan": plan,
        "resume_plan": run_plan,
        "plan_carry_detail": {
            "carry_backend": "checkpoint (verbatim)",
            "subtasks": bool(subtasks),
            "done": done,
            "claimed": claimed,
        },
        "subgoal": subgoal,
        "brief": brief["text"],
        "brief_places": brief["places"],
        "brief_pose": brief["pose"],
        "instruction": resumed_instruction,
        "checkpoint": copy.deepcopy(checkpoint),
        "restore": _restore_extras(checkpoint),
        "done": done,
        "claimed": claimed,
        "complete_count": len(done),
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
    completion = completion_chain(entries)
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
        "driver_completion_verified": bool(completion.get("verified")),
        "completion_scope": {
            key: completion.get(key)
            for key in ("scoped", "scope", "rejected_records", "scoped_records",
                        "raw_verified", "task_scoped")
        },
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
    mapping = stage_subtask_map(task, legs)
    report = stage_report(entries, task, poses=poses, subtask_map=mapping)
    target_stage = next((stage for stage in task.stages if stage.target), None)
    verify = verify_evidence(entries, target_stage, gate_m)
    chain_completion = completion_chain(entries)
    plan_completion = offline_plan_completion(legs, task)
    finish_contract = finish_contract_rows(entries)
    acceptance = instance_acceptance(entries, task, mapping)
    recovery_scope = recovery_regrounding(legs, task)
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
    ordered_success = bool(report["chain_complete"] and report["order_ok"])
    traceable = bool(
        ordered_success
        and report["relation_ok"]
        and evidence_complete
        and chain_ok
        and success
    )
    claim = completion_claims(legs, entries, ordered_success=ordered_success)
    interrupted_leg = next((leg for leg in legs if leg.interrupted), None)
    before = (
        stage_report(
            [entry for entry in entries if entry["leg"] == interrupted_leg.plan.label],
            task,
            subtask_map=mapping,
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
        "plan_completion": plan_completion,
        "finish_contract": finish_contract,
        "instance_acceptance": acceptance,
        "completion_claim": claim,
        "recovery_regrounding": recovery_scope,
        "ordered_success": ordered_success,
        "evidence_complete": evidence_complete,
        "traceable_success": traceable,
        "interrupted": interrupted,
        "interrupt_at_stages": int(before),
        "stages_gained_on_resume": int(report["complete_count"] - before) if interrupted else 0,
        "interrupt_recovered": bool(interrupted and report["complete_count"] > before),
        "resume_used_carry": bool(
            interrupted and any(leg.plan.carry.get("mode") == "resume" for leg in legs[1:])
        ),
        "done_stage_recheck": done_stage_recheck(entries, task, interrupted_leg, mapping),
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
        report = stage_report(
            evidence_entries, task, subtask_map=stage_subtask_map(task, legs)
        )
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


def checkpoint_report(
    task: TaskSpec,
    legs: Sequence[LegResult],
    *,
    subtask_map: Mapping[str, int] | None = None,
) -> dict:
    """Offline score of the checkpoint against the scene's ground truth.

    This is the only place the two channels meet. The checkpoint is what the
    *agent* believed at the interrupt (its own plan statuses and confirmation
    ledger); ``stage_report`` is what the scene really accepted. The report
    counts the agent's wrong beliefs and whether the resumed leg overturned any
    of them by itself (a sub-task it no longer marks done, or a confirmation it
    now rejects).

    Two blind spots of the stage-scoped verdict are filled from the checkpoint's
    own records: a confirmation accepted under a *different* sub-task still has
    a candidate place in the leg's whole candidate table (v10 seed23 accepted
    the landmark candidate while executing the object stage), and a plan status
    ``claimed_done`` is a completion claim even though the runtime has not
    promoted it to ``done`` yet.
    """
    interrupted = next((leg for leg in legs if leg.interrupted), None)
    checkpoint = interrupted.checkpoint if interrupted is not None else {}
    if not isinstance(checkpoint, Mapping) or not checkpoint:
        return {"available": False}
    mapping = subtask_map or stage_subtask_map(task, [interrupted])
    truth = stage_report(interrupted.entries, task, subtask_map=mapping)
    gt_done = {
        f"st_{stage.index}"
        for stage in task.stages
        if stage.stage_id in set(truth.get("complete_ids") or ())
    }
    plan = checkpoint.get("plan") if isinstance(checkpoint.get("plan"), Mapping) else None
    claimed_indexes, claimed_open = _plan_belief(plan)
    claimed_done = {f"st_{index}" for index in claimed_indexes}
    wrong_done = sorted(claimed_done - gt_done)
    missed_done = sorted(gt_done - claimed_done)
    claimed_claims = {f"st_{index}" for index in _plan_claimed_done(plan)}
    wrong_claimed = sorted(claimed_claims - gt_done)

    ledger = checkpoint.get("confirmation_ledger") or {}
    claimed_confirmed = sorted(
        str(candidate_id)
        for candidate_id, entry in (ledger.items() if isinstance(ledger, Mapping) else ())
        if isinstance(entry, Mapping) and str(entry.get("last_verdict") or "") == "accepted"
    )
    # identity claims the offline report can adjudicate: a confirmed candidate
    # whose own place is outside the stage's locate radius is a wrong instance.
    # The stage-scoped confirmation can only resolve candidates proposed under
    # that stage; fall back to the whole leg's table (and the checkpoint's own
    # candidate records) for a candidate proposed under another sub-task.
    candidate_table = _checkpoint_candidate_table(checkpoint, interrupted.entries)
    wrong_confirmed: set[str] = set()
    resolved_confirmations: list[dict] = []
    for stage in task.stages:
        row = (truth.get("stages") or {}).get(stage.stage_id) or {}
        if not row.get("confirmed"):
            continue
        candidate_id = str(row.get("confirmation_candidate_id") or "")
        correct = row.get("confirmation_correct")
        distance = row.get("confirmation_distance_m")
        source = "stage_scope"
        if correct is None and candidate_id:
            point = candidate_table.get(candidate_id)
            if point is not None and stage.position is not None:
                distance = _distance(point, stage.position)
                correct = bool(distance <= float(stage.locate_m))
                source = "leg_candidate_table"
                resolved_confirmations.append(
                    {
                        "stage": stage.stage_id,
                        "candidate_id": candidate_id,
                        "distance_m": round(float(distance), 3),
                        "locate_m": float(stage.locate_m),
                        "correct": correct,
                        "source": source,
                    }
                )
        if correct is False and candidate_id:
            wrong_confirmed.add(candidate_id)
    wrong_confirmed_ids = sorted(wrong_confirmed)

    follow = next(
        (leg for leg in legs[1:] if leg.plan.role in ("resume", "control")), None
    )
    final_plan = follow.final_plan if follow is not None else None
    final_status: dict[str, str] = {}
    for position, subtask in enumerate((final_plan or {}).get("subtasks") or [], 1):
        if isinstance(subtask, Mapping):
            final_status[f"st_{_subtask_number(subtask, position)}"] = str(
                subtask.get("status") or ""
            )
    corrected = [
        stage_id for stage_id in wrong_done
        if final_status and final_status.get(stage_id) != "done"
    ]
    corrected_claimed = [
        stage_id for stage_id in wrong_claimed
        if final_status and final_status.get(stage_id) not in ("done", "claimed_done")
    ]
    final_ledger = (
        follow.checkpoint.get("confirmation_ledger")
        if follow is not None and isinstance(follow.checkpoint, Mapping)
        else {}
    )
    corrected_confirmed = [
        candidate_id for candidate_id in wrong_confirmed_ids
        if isinstance(final_ledger, Mapping)
        and str((final_ledger.get(candidate_id) or {}).get("last_verdict") or "")
        == "rejected"
    ]
    corrections = sorted(set(corrected) | set(corrected_claimed) | set(corrected_confirmed))
    return {
        "available": True,
        "source": str(checkpoint.get("source") or ""),
        "fields": sorted(str(key) for key in checkpoint),
        "claimed_done": sorted(claimed_done),
        "claimed_open": [f"st_{index}" for index in claimed_open],
        "claimed_claimed_done": sorted(claimed_claims),
        "gt_done": sorted(gt_done),
        "wrong_done": wrong_done,
        "wrong_claimed": wrong_claimed,
        "missed_done": missed_done,
        "claimed_confirmed": claimed_confirmed,
        "wrong_confirmed": wrong_confirmed_ids,
        "resolved_confirmations": resolved_confirmations,
        "wrong_belief_count": len(wrong_done) + len(wrong_claimed) + len(wrong_confirmed_ids),
        "resume_corrections": len(corrections),
        "corrected": corrections,
        "checkpoint_correct": not wrong_done and not wrong_claimed and not wrong_confirmed_ids,
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
    verdict["plan_quality"] = plan_quality_audit(
        list(legs), task,
        plan_mode=str((legs[0].plan.ctx_extra or {}).get("plan_mode") or "given"),
    )
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
    # the two channels meet only here: the checkpoint is judged against the
    # offline stage verdict after it has been restored and acted on
    verdict["checkpoint"] = checkpoint_report(task, legs)
    return verdict


def done_stage_recheck(
    entries: Sequence[dict],
    task: TaskSpec,
    interrupted_leg: LegResult | None,
    subtask_map: Mapping[str, int] | None = None,
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
        if (stage_report(
            prefix, task, subtask_map=subtask_map or stage_subtask_map(task, [interrupted_leg])
        )["stages"][stage.stage_id])["complete"]
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
        stage_type = str(row.get("done_when_type") or stage.effective_done_when_type())
        if row.get("complete"):
            if stage_type in (DONE_WHEN_SIGHTED, DONE_WHEN_LOCATED):
                lines.append(
                    f"stage {stage.index} ({stage.label}) complete as {stage_type}: "
                    f"located {row.get('locate_distance_m')} m off via "
                    f"{row.get('locate_tool') or 'a tool result'} at call "
                    f"{row.get('located_seq')} with {row.get('evidence_count')} "
                    f"evidence ref(s); no flight required"
                )
            else:
                lines.append(
                    f"stage {stage.index} ({stage.label}) complete as {stage_type}: "
                    f"reached {row.get('reach_distance_m')} m from its place and located it "
                    f"{row.get('locate_distance_m')} m off via "
                    f"{row.get('locate_tool') or 'a tool result'} at call "
                    f"{row.get('located_seq')} with {row.get('evidence_count')} evidence ref(s)"
                )
        else:
            checks = [("locate", row.get("located"))]
            if row.get("reach_required"):
                checks.append(("reach", row.get("reach")))
            missing = ", ".join(name for name, flag in checks if not flag)
            lines.append(
                f"stage {stage.index} ({stage.label}) NOT complete "
                f"(done_when={stage_type}; missing {missing or 'identification'})"
            )
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
                "done_when_type": stage.effective_done_when_type(),
                "n": n,
                "reach": reach,
                "located": located,
                "complete": complete,
                "reach_attempts": sum(
                    int(
                        ((row["stages"]["stages"].get(stage.stage_id) or {}).get(
                            "reach_attempts"
                        ))
                        or 0
                    )
                    for row in episodes
                ),
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
    ordered = [row for row in episodes if row.get("ordered_success")]
    acceptance_rows = [
        row.get("instance_acceptance") or {} for row in episodes
    ]
    acceptance_events = sum(int(row.get("events") or 0) for row in acceptance_rows)
    acceptance_resolved = sum(int(row.get("resolved") or 0) for row in acceptance_rows)
    acceptance_wrong = sum(int(row.get("wrong") or 0) for row in acceptance_rows)
    claim_rows = [row.get("completion_claim") or {} for row in episodes]
    episodes_claiming = sum(1 for row in claim_rows if row.get("claimed"))
    wrong_completions = sum(1 for row in claim_rows if row.get("wrong"))
    checkpoint_rows = [
        row.get("checkpoint") or {} for row in episodes
    ]
    checkpoint_scored = [row for row in checkpoint_rows if row.get("available")]
    unsafe_done = [
        item
        for row in episodes
        for item in ((row.get("plan_vs_evidence_by_leg") or {}).get("unsafe_done_claims") or [])
    ]
    compared_stage_rows = sum(
        len(
            [
                item
                for item in ((row.get("plan_vs_evidence_by_leg") or {}).get("rows") or [])
                if item.get("agree") is not None
            ]
        )
        for row in episodes
    )
    finish_rows = [
        item for row in episodes
        for item in ((row.get("finish_contract") or {}).get("rows") or [])
    ]
    recovery_rows = [
        item for row in episodes
        for item in ((row.get("recovery_regrounding") or {}).get("rows") or [])
    ]
    quality_rows = [row.get("plan_quality") or {} for row in episodes]
    quality_available = [row for row in quality_rows if row.get("available")]
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
        "stage_complete_by_type": {
            kind: {
                "stages": sum(1 for row in stage_rows if row["done_when_type"] == kind),
                "complete": sum(
                    row["complete"] for row in stage_rows if row["done_when_type"] == kind
                ),
                "reach_attempts": sum(
                    row["reach_attempts"]
                    for row in stage_rows
                    if row["done_when_type"] == kind
                ),
            }
            for kind in DONE_WHEN_TYPES
        },
        "reach_attempts": sum(int(row["reach_attempts"]) for row in stage_rows),
        "chain_complete_rate": _rate(sum(1 for row in episodes if row["stages"]["chain_complete"]), n),
        "order_ok_rate": _rate(sum(1 for row in episodes if row["stages"]["order_ok"]), n),
        # goal.md v1 main metric: the whole chain in the declared stage order
        "ordered_success_rate": _rate(len(ordered), n),
        "complete_ordered_task_success": {
            "count": len(ordered), "n": n, "rate": _rate(len(ordered), n)
        },
        "plan_quality": {
            "plans": len(quality_available),
            "count_match": sum(1 for row in quality_available if row.get("count_match")),
            "count_match_rate": _rate(
                sum(1 for row in quality_available if row.get("count_match")),
                len(quality_available),
            ),
            "order_ok": sum(1 for row in quality_available if row.get("order_ok")),
            "order_ok_rate": _rate(
                sum(1 for row in quality_available if row.get("order_ok")),
                len(quality_available),
            ),
            "mean_coverage": (
                round(
                    sum(float(row.get("coverage_rate") or 0.0) for row in quality_available)
                    / len(quality_available),
                    3,
                )
                if quality_available
                else None
            ),
            "done_when_type_checked": sum(
                int(row.get("done_when_type_checked") or 0) for row in quality_available
            ),
            "done_when_type_hits": sum(
                int(row.get("done_when_type_hits") or 0) for row in quality_available
            ),
            "done_when_type_accuracy": _rate(
                sum(int(row.get("done_when_type_hits") or 0) for row in quality_available),
                sum(int(row.get("done_when_type_checked") or 0) for row in quality_available),
            ),
            "structure_ok": sum(
                1 for row in quality_available if row.get("structure_ok")
            ),
            "by_episode": [
                {
                    "label": row.get("label"),
                    "seed": row.get("seed"),
                    "plan_mode": row.get("plan_mode"),
                    **{
                        key: (quality_rows[index] or {}).get(key)
                        for key in (
                            "available", "n_subtasks", "n_stages", "count_match",
                            "order_ok", "coverage_rate", "done_when_type_accuracy",
                            "structure_ok",
                        )
                    },
                }
                for index, row in enumerate(episodes)
            ],
        },
        "wrong_instance_acceptance": {
            "events": acceptance_events,
            "resolved": acceptance_resolved,
            "wrong": acceptance_wrong,
            "unresolved": acceptance_events - acceptance_resolved,
            "rate": _rate(acceptance_wrong, acceptance_resolved),
            "episodes_with_wrong": sum(
                1 for row in acceptance_rows if int(row.get("wrong") or 0)
            ),
        },
        "wrong_completion": {
            "episodes_claiming": episodes_claiming,
            "wrong": wrong_completions,
            "rate": _rate(wrong_completions, episodes_claiming),
            "claims": sum(int(row.get("n_claims") or 0) for row in claim_rows),
            # the stage-level form of the same failure (r5's "unsafe done"):
            # the plan closed a sub-task whose stage the evidence does not support
            "unsafe_done_stage_claims": len(unsafe_done),
            "unsafe_done_stage_rate": _rate(len(unsafe_done), compared_stage_rows),
            "unsafe_done_episodes": sum(
                1 for row in episodes
                if (row.get("plan_vs_evidence_by_leg") or {}).get("unsafe_done_claims")
            ),
            "by_episode": [
                {
                    "label": row.get("label"),
                    "seed": row.get("seed"),
                    "claimed": bool((row.get("completion_claim") or {}).get("claimed")),
                    "evaluated_complete": bool(
                        (row.get("completion_claim") or {}).get("evaluated_complete")
                    ),
                    "wrong": bool((row.get("completion_claim") or {}).get("wrong")),
                }
                for row in episodes
            ],
        },
        "checkpoint_correctness": {
            "episodes": len(checkpoint_scored),
            "checkpoint_correct": sum(
                1 for row in checkpoint_scored if row.get("checkpoint_correct")
            ),
            "checkpoint_correct_rate": _rate(
                sum(1 for row in checkpoint_scored if row.get("checkpoint_correct")),
                len(checkpoint_scored),
            ),
            "wrong_beliefs": sum(
                int(row.get("wrong_belief_count") or 0) for row in checkpoint_scored
            ),
            "episodes_with_wrong_belief": sum(
                1 for row in checkpoint_scored if int(row.get("wrong_belief_count") or 0)
            ),
            "wrong_done_beliefs": sum(
                len(row.get("wrong_done") or []) for row in checkpoint_scored
            ),
            "wrong_claimed_beliefs": sum(
                len(row.get("wrong_claimed") or []) for row in checkpoint_scored
            ),
            "wrong_confirmation_beliefs": sum(
                len(row.get("wrong_confirmed") or []) for row in checkpoint_scored
            ),
            "resume_corrections": sum(
                int(row.get("resume_corrections") or 0) for row in checkpoint_scored
            ),
            "episodes_with_correction": sum(
                1 for row in checkpoint_scored if int(row.get("resume_corrections") or 0)
            ),
            "by_episode": [
                {
                    "label": row.get("label"),
                    "wrong_done": list((row.get("checkpoint") or {}).get("wrong_done") or []),
                    "wrong_claimed": list(
                        (row.get("checkpoint") or {}).get("wrong_claimed") or []
                    ),
                    "wrong_confirmed": list(
                        (row.get("checkpoint") or {}).get("wrong_confirmed") or []
                    ),
                    "corrections": int(
                        (row.get("checkpoint") or {}).get("resume_corrections") or 0
                    ),
                }
                for row in episodes
                if (row.get("checkpoint") or {}).get("available")
            ],
        },
        "finish_contract": {
            "calls": len(finish_rows),
            "blocked": sum(1 for row in finish_rows if row.get("blocked")),
            "blocked_then_continued": sum(
                1 for row in finish_rows if row.get("continued_after_blocked")
            ),
            "episodes_with_missing_subtasks": sum(
                1 for row in episodes
                if (row.get("finish_contract") or {}).get("missing_subtasks")
            ),
        },
        "recovery_scope": {
            "events": len(recovery_rows),
            "executed": sum(1 for row in recovery_rows if row.get("executed")),
            "subgoal_scoped": sum(
                1 for row in recovery_rows
                if row.get("classification") in ("subgoal", "subgoal+anchor")
            ),
            "with_anchor": sum(
                1 for row in recovery_rows
                if row.get("classification") == "subgoal+anchor"
            ),
            "global_instruction": sum(
                1 for row in recovery_rows
                if row.get("classification") == "global_instruction"
            ),
        },
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
            "outer_tool_calls": sum(leg["outer_tool_calls"] for leg in leg_rows),
            "inner_tool_calls": sum(leg["inner_tool_calls"] for leg in leg_rows),
            "tool_calls": sum(leg["outer_tool_calls"] + leg["inner_tool_calls"] for leg in leg_rows),
            "executor_auto_steps": sum(
                int(leg.get("executor_auto_steps") or 0) for leg in leg_rows
            ),
            "executor_legs": executor_legs,
            "executor_ticks": executor_ticks,
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
    scope_rows = [
        {
            "episode": row.get("label"),
            "leg": leg.get("label"),
            "entries": int(leg.get("outer_tool_calls") or 0) + int(leg.get("inner_tool_calls") or 0),
            "scope_active": bool((row.get("stages") or {}).get("scope", {}).get("active")),
        }
        for row in episodes
        for leg in row.get("legs", [])
    ]
    checks["scope_contract"] = {
        "episodes_scoped": sum(
            1 for row in episodes
            if (row.get("stages") or {}).get("scope", {}).get("active")
        ),
        "episodes": len(episodes),
        "rows": scope_rows,
    }
    finish_rows = [
        item for row in episodes
        for item in ((row.get("finish_contract") or {}).get("rows") or [])
    ]
    checks["finish_contract"] = {
        "calls": len(finish_rows),
        "blocked": sum(1 for item in finish_rows if item.get("blocked")),
        "blocked_then_continued": sum(
            1 for item in finish_rows if item.get("continued_after_blocked")
        ),
        "episodes_with_missing_subtasks": sum(
            1 for row in episodes
            if (row.get("finish_contract") or {}).get("missing_subtasks")
        ),
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
    ordered = agg.get("complete_ordered_task_success") or {}
    wrong_instance = agg.get("wrong_instance_acceptance") or {}
    wrong_completion = agg.get("wrong_completion") or {}
    cost = agg.get("cost") or {}
    lines += [
        "",
        f"scene chain check: {chain_status}",
        "",
        "## goal.md v1 main metrics",
        "",
        "| metric | value | denominator |",
        "|---|---|---|",
        f"| complete ordered task success | {ordered.get('count')}/{ordered.get('n')} "
        f"({ordered.get('rate')}) | episodes (full chain + stage order) |",
        f"| wrong-instance acceptance | {wrong_instance.get('wrong')}/{wrong_instance.get('resolved')} "
        f"({wrong_instance.get('rate')}) | resolved arrival/confirmation events "
        f"({wrong_instance.get('unresolved')} unresolved) |",
        f"| wrong completion | {wrong_completion.get('wrong')}/{wrong_completion.get('episodes_claiming')} "
        f"({wrong_completion.get('rate')}) | episodes with a success claim |",
        f"| unsafe `done` stages | {wrong_completion.get('unsafe_done_stage_claims')} "
        f"({wrong_completion.get('unsafe_done_stage_rate')}) | stage rows compared |",
        f"| total model calls | {cost.get('model_calls')} | incl. routine/inner calls |",
        f"| total tool calls | {cost.get('tool_calls')} | outer {cost.get('outer_tool_calls')} + "
        f"inner {cost.get('inner_tool_calls')} |",
        f"| executor auto steps | {cost.get('executor_auto_steps')} | calls without a model turn |",
        "",
        "_complete ordered task success_ = every stage reached and located, and the "
        "stage completion order is the declared one (the offline evaluator's verdict, "
        "not the model's). _wrong-instance acceptance_ = an arrival/verification whose "
        "object is farther than the stage's `locate_m` from the true instance. "
        "_wrong completion_ = a `finish` claiming success while that ordered chain "
        "was not established.",
        "",
        "## stages (per episode share)",
        "",
        "| # | stage | done_when | reach | located | complete | reach attempts |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in agg.get("stages", []):
        lines.append(
            f"| {row['index']} | {row['label']} | {row.get('done_when_type', '')} | "
            f"{row['reach']}/{row['n']} "
            f"({row['reach_rate']}) | {row['located']}/{row['n']} ({row['locate_rate']}) | "
            f"{row['complete']}/{row['n']} ({row['complete_rate']}) | "
            f"{row.get('reach_attempts', 0)} |"
        )
    quality = agg.get("plan_quality") or {}
    lines += [
        "",
        "## plan quality (offline audit, not in the prompt)",
        "",
        f"- plans audited: {quality.get('plans', 0)}; stage-count match: "
        f"{quality.get('count_match', 0)} ({quality.get('count_match_rate')}); "
        f"order ok: {quality.get('order_ok', 0)} ({quality.get('order_ok_rate')})",
        f"- mean stage coverage: {quality.get('mean_coverage')}; "
        f"done_when type accuracy: {quality.get('done_when_type_hits', 0)}/"
        f"{quality.get('done_when_type_checked', 0)} "
        f"({quality.get('done_when_type_accuracy')}); structurally valid plans: "
        f"{quality.get('structure_ok', 0)}/{quality.get('plans', 0)}",
        "",
        "## semantic completion (reach attempts by stage type)",
        "",
        "| done_when type | stages | complete | reach attempts |",
        "|---|---|---|---|",
    ]
    for kind, row in (agg.get("stage_complete_by_type") or {}).items():
        lines.append(
            f"| {kind} | {row.get('stages', 0)} | {row.get('complete', 0)} | "
            f"{row.get('reach_attempts', 0)} |"
        )
    intr = agg.get("interrupt") or {}
    lines += [
        "",
        "## episodes",
        "",
        "| episode | seed | legs (roles) | stages (reach/locate/complete) | ordered | chain | success | traceable | wrong inst. | wrong completion | interrupted | recovered |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in report.get("episodes", []):
        stages = row.get("stages", {})
        legs = row.get("legs", [])
        roles = "+".join(str(leg.get("role")) for leg in legs) or "n/a"
        acceptance = row.get("instance_acceptance") or {}
        claim = row.get("completion_claim") or {}
        lines.append(
            "| {label} | {seed} | {legs} ({roles}) | {reach}/{locate}/{complete} | {ordered} | {chain} | {success} | {traceable} | {wrong_i} | {wrong_c} | {intr} | {rec} |".format(
                label=row.get("label") or (legs[0]["label"] if legs else "n/a"),
                seed=row.get("seed", "-"),
                legs=row.get("n_legs"),
                roles=roles,
                reach=stages.get("reach_count"),
                locate=stages.get("locate_count"),
                complete=stages.get("complete_count"),
                ordered=row.get("ordered_success"),
                chain=stages.get("chain_complete"),
                success=row.get("success"),
                traceable=row.get("traceable_success"),
                wrong_i=f"{acceptance.get('wrong')}/{acceptance.get('resolved')}",
                wrong_c=claim.get("wrong"),
                intr=row.get("interrupted"),
                rec=row.get("interrupt_recovered"),
            )
        )
        for arm in row.get("arms") or []:
            lines.append(
                f"|   arm {arm.get('role')} | {arm.get('label')} | "
                f"{arm.get('stages_complete')} complete | {arm.get('chain_complete')} | "
                f"- | {arm.get('chain_complete')} | {arm.get('success')} | "
                f"{arm.get('traceable_success')} | - | - | - | - |"
            )
    lines += [
        "",
        "## finish contract (goal.md v1)",
        "",
        "| episode | leg | status | claimed | verified | plan complete | missing sub-tasks | calls after |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in report.get("episodes", []):
        for item in ((row.get("finish_contract") or {}).get("rows") or []):
            missing = ", ".join(
                f"{entry.get('subtask_id')}[{entry.get('status')}]"
                for entry in item.get("missing_subtasks") or []
            )
            lines.append(
                f"| {row.get('label')} | {item.get('leg')} | {item.get('status')} | "
                f"{item.get('claimed')} | {item.get('verified')} | {item.get('plan_complete')} | "
                f"{missing or '-'} | {item.get('calls_after')} |"
            )
    lines += [
        "",
        "## recovery re-grounding scope (goal.md v1, seed23)",
        "",
        "| episode / leg | turn | stalled tool | stage | executed | instruction | classification |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report.get("episodes", []):
        for item in ((row.get("recovery_regrounding") or {}).get("rows") or []):
            instruction = str(item.get("instruction") or "").replace("|", "\\|")
            lines.append(
                f"| {row.get('label')} / {item.get('leg')} | {item.get('turn')} | "
                f"{item.get('stalled_tool')} | {item.get('stage')} | {item.get('executed')} | "
                f"{instruction or '-'} | {item.get('classification')} |"
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
        "## checkpoint (agent beliefs vs offline ground truth)",
        "",
        f"- scored checkpoints: {(agg.get('checkpoint_correctness') or {}).get('episodes', 0)}; "
        f"correct: {(agg.get('checkpoint_correctness') or {}).get('checkpoint_correct', 0)} "
        f"({(agg.get('checkpoint_correctness') or {}).get('checkpoint_correct_rate')})",
        f"- wrong beliefs (done/confirmed the scene did not accept): "
        f"{(agg.get('checkpoint_correctness') or {}).get('wrong_beliefs', 0)} "
        f"in {(agg.get('checkpoint_correctness') or {}).get('episodes_with_wrong_belief', 0)} episode(s)",
        f"- wrong beliefs corrected by the resumed leg itself: "
        f"{(agg.get('checkpoint_correctness') or {}).get('resume_corrections', 0)} "
        f"in {(agg.get('checkpoint_correctness') or {}).get('episodes_with_correction', 0)} episode(s)",
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
    ``run_config.json`` (the leg plan), ``summary.json`` (the runner's verdict),
    ``episode_log.json`` (the runtime's decision nodes and plan transitions) and
    ``checkpoint.json`` (the runtime state the offline checkpoint score reads -
    without it a re-analysis reports ``checkpoint_correctness.episodes=0``).
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
    checkpoint: dict = {}
    checkpoint_path = out_dir / "checkpoint.json"
    if checkpoint_path.exists():
        raw_checkpoint = json.loads(checkpoint_path.read_text())
        if isinstance(raw_checkpoint, Mapping):
            checkpoint = dict(raw_checkpoint)
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
        checkpoint=checkpoint,
        final_plan=copy.deepcopy(checkpoint.get("plan"))
        if isinstance(checkpoint.get("plan"), Mapping) else None,
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
    source_meta: dict = {}
    source_report = run_dir / REPORT_JSON
    if source_report.exists():
        try:
            loaded = json.loads(source_report.read_text()).get("meta")
            source_meta = dict(loaded) if isinstance(loaded, Mapping) else {}
        except (OSError, ValueError):
            source_meta = {}
    meta = {
        "task_name": task.name,
        "task_file": str(task_path),
        "instruction": task.instruction,
        # run-level knobs the raw report carries (the re-analysis must not
        # silently drop the arm label it is scoring)
        "plan_mode": source_meta.get("plan_mode"),
        "seeds": source_meta.get("seeds"),
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
    parser.add_argument("--plan-mode", type=str, default="given",
                        dest="plan_mode", choices=["given", "self"],
                        help="given (default): the driver writes the structured stage "
                             "plan before the first model turn (the execution arm); "
                             "self: the driver writes no plan - the model receives "
                             "only the natural-language instruction and must write "
                             "the sub-task plan itself, which the runtime validates "
                             "(one correction) and executes")
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
    plan_mode = str(getattr(args, "plan_mode", "given") or "given")
    if plan_mode == "self":
        # the self-plan arm gives the model the natural-language mission only:
        # no carry plan, no stage list, no next_step bootstrap. The runtime asks
        # for the plan on the first decision turn and validates it.
        first.ctx_extra = {"plan_mode": "self"}
        first.carry = {
            "mode": "fresh",
            "channels": [],
            "plan_mode": "self",
            "done": [],
            "plan_steps": [],
            "note": "natural-language instruction only; the model writes the plan",
        }
    else:
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
    ctx_extra: dict[str, Any] = {"subgoal": state["subgoal"]}
    run_plan = state.get("resume_plan", state.get("plan"))
    if run_plan is not None:
        ctx_extra["plan"] = run_plan
    # the checkpoint's own stores travel with the plan: confirmation ledger,
    # candidate/observation/viewpoint references, scope, beliefs and the
    # evidence pack. This is a verbatim load - no verdict is recomputed here.
    ctx_extra.update(state.get("restore") or {})
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
                "claimed": list(state.get("claimed") or []),
                "plan_steps": [stage.step for stage in task.stages],
                "plan_current_step": (state.get("plan") or {}).get("current_step"),
                "plan_subtasks": [
                    {"id": item.get("id"), "status": item.get("status")}
                    for item in (state.get("plan") or {}).get("subtasks") or []
                ],
                "plan_carry_backend": (state.get("plan_carry_detail") or {}).get("carry_backend"),
                "checkpoint_fields": sorted(state.get("checkpoint") or {}),
                "checkpoint_source": (state.get("checkpoint") or {}).get("source"),
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
                "ordered_success": metrics["ordered_success"],
                "chain_complete": metrics["stages"]["chain_complete"],
                "complete_count": metrics["stages"]["complete_count"],
                "wrong_instance": (metrics["instance_acceptance"] or {}).get("wrong"),
                "wrong_completion": (metrics["completion_claim"] or {}).get("wrong"),
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
        "plan_mode": str(getattr(args, "plan_mode", "given") or "given"),
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
        "resume_backend": (
            "runtime checkpoint (ctx.extra) saved verbatim to checkpoint.json and "
            "restored verbatim: plan/sub-task statuses, confirmation ledger, "
            "perception.candidates, observations, viewpoints, hypotheses, task scope, "
            "beliefs, open questions, evidence pack; plus instruction brief and pose"
        ),
        "memory_carry": (
            "the memory module itself is built per leg (build_memory(out_dir, "
            "episode_id)) and exposes no cross-episode load, but the runtime state it "
            "shares through ctx.extra is saved verbatim as checkpoint.json and restored "
            "verbatim on resume (including perception.candidates); the resumed leg's "
            "knowledge therefore travels through the checkpoint, the instruction brief "
            "and the carried pose - a wrong belief survives the restart and is scored "
            "offline, not corrected by the driver"
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
