#!/usr/bin/env python
"""M4 mixed-mission matrix driver (action / semantic / mixed x 2,4,8 stages).

The driver executes the JSON missions under ``data/mixed_missions`` through
:class:`vlm_nav.mixed_mission.MixedMissionExecutor` with the real
:class:`vlm_nav.motion.MotionExecutor` for motion steps.  Semantic steps are
served by a deterministic, CPU-only ``MockSemanticService`` (default) or by the
installed semantic-navigation plugin stack (``--semantic real``).  The mock
speaks the same contract as the real service and issues its calls through the
tool registry, so ``tool_calls`` in the report is a real registry count.

Every cell installs, activates, runs and then deactivates/uninstalls its plugin
stack (motion-only for the action group; motion + planner + mixed for the
semantic/mixed groups; the legacy perception/navigation/semantic stack for
``--semantic real``).  ``resources_after_uninstall`` records the plugin
catalogue count, residual cleanup entries and any live worker pids.

``--interrupt-step N`` cancels each episode after N completed steps through the
executor's ``step.finished`` progress callback, loads the persisted state and
resumes it.  The report records the steps completed before the interruption,
the number of replayed steps (must be zero), and whether the measured motion
progress survived the resume.

No VLM, GPU or network is required:

    python scripts/run_mixed_missions.py --stages 2 --seeds 2 \
        --groups action,semantic,mixed --out outputs/mixed_missions/matrix

The report is ``<out>/report.json`` plus a human-readable ``report.md`` with
schema ``vlm_nav.mixed_missions.v1``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from vlm_nav.hardware import reset_peak, snapshot  # noqa: E402
from vlm_nav.mixed_mission import (  # noqa: E402
    MIXED_CANCELLED,
    MIXED_COMPLETED,
    MIXED_STEP_DONE,
    MOTION_STEP_KINDS,
    SEMANTIC_STEP_KINDS,
    MixedMissionExecutor,
    MixedMissionSpec,
    MixedMissionState,
    load_state,
)
from vlm_nav.motion import MotionExecutor, clock_for_world  # noqa: E402
from vlm_nav.plugins import PluginHost  # noqa: E402
from vlm_nav.plugins.mixed import MixedMissionPlugin  # noqa: E402
from vlm_nav.plugins.motion import MotionPlugin  # noqa: E402
from vlm_nav.plugins.task_planner import TaskPlannerPlugin  # noqa: E402
from vlm_nav.semantic import SEMANTIC_SERVICE, compose_instruction  # noqa: E402
from vlm_nav.tools import (  # noqa: E402
    STATUS_FAILED,
    STATUS_OK,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from vlm_nav.world import SplatWorldCPU, TargetSpec  # noqa: E402

SCHEMA = "vlm_nav.mixed_missions.v1"
TASKS_SCHEMA = "vlm_nav.mixed_missions.tasks/1"
GROUPS = ("action", "semantic", "mixed")
STAGE_COUNTS = (2, 4, 8)
SEMANTIC_GROUP_KINDS = (
    "search", "approach", "confirm", "verify", "revisit", "goto",
)

DEFAULT_PLY = REPO / "data" / "phase1" / "scenes" / "scenario_scene.ply"
DEFAULT_TASKS_DIR = REPO / "data" / "mixed_missions"
DEFAULT_OUT = REPO / "outputs" / "mixed_missions" / "matrix"
DEFAULT_TARGET_POS = (4.0, 0.0, 1.0)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


# ---------------------------------------------------------------------------
# deterministic mock semantic service
# ---------------------------------------------------------------------------


def _point(value: Any) -> list[float] | None:
    try:
        if isinstance(value, Mapping):
            value = [value["x"], value["y"], value["z"]]
        point = [float(item) for item in value]
    except (TypeError, ValueError, KeyError):
        return None
    if len(point) != 3 or not all(math.isfinite(item) for item in point):
        return None
    return point


def _set_world_position(world: Any, new: Sequence[float]) -> bool:
    """Move a duck-typed world to ``new`` in place when it is writable."""
    point = [float(item) for item in new]
    for holder in (world, getattr(world, "state", None)):
        if holder is None:
            continue
        try:
            current = holder.position_flu
        except (AttributeError, TypeError):
            continue
        try:
            array = np.asarray(current, dtype=float)
            if array.size != 3:
                continue
            array.reshape(-1)[:3] = point
            if np.allclose(
                np.asarray(holder.position_flu, dtype=float).reshape(-1)[:3],
                point,
            ):
                return True
        except (AttributeError, TypeError, ValueError):
            pass
    for attr in ("position_flu", "position"):
        try:
            setattr(world, attr, np.asarray(point, dtype=float))
            current = np.asarray(getattr(world, attr), dtype=float).reshape(-1)
            if current.size == 3 and np.allclose(current[:3], point):
                return True
        except (AttributeError, TypeError, ValueError):
            continue
    return False


class MockSemanticService:
    """Deterministic ``SemanticNavigationService`` substitute for CPU runs.

    It answers the documented contract (``ok``, ``tool``, ``status``,
    ``candidate_id``, ``task_id``, ``arrived``, ``accepted``, ``verified``,
    ``calls``, ``error``, ``data``) and performs its work through the tool
    registry, so every semantic stage is a real registry call.  Approach and
    travel methods move the world toward ``ctx.extra["mission.target_pose"]``
    (falling back to the target passed by the driver), stopping at the
    requested stand-off distance.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        target_pose: Sequence[float] | None = None,
        step_m: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.registry = registry
        self.target_pose = _point(target_pose) or [0.0, 0.0, 0.0]
        self.step_m = max(1e-6, float(step_m))
        self.seed = int(seed)
        self.calls = 0
        self.moved_m = 0.0
        self.candidate_id = ""
        self._search_seq = 0
        self._task_seq = 0
        self._registered: list[str] = []
        self._specs = self._tool_specs()

    # -- lifecycle -------------------------------------------------------- #

    def register(self) -> None:
        names = set(self.registry.names())
        for spec in self._specs:
            if spec.name in names:
                raise RuntimeError(
                    f"mock semantic tool {spec.name!r} is already registered"
                )
        for spec in self._specs:
            self.registry.register(spec)
            self._registered.append(spec.name)

    def close(self) -> None:
        for name in reversed(self._registered):
            try:
                self.registry.unregister(name)
            except KeyError:
                pass
        self._registered.clear()

    def reset(
        self,
        *,
        seed: int | None = None,
        target_pose: Sequence[float] | None = None,
    ) -> None:
        if seed is not None:
            self.seed = int(seed)
        if target_pose is not None:
            self.target_pose = _point(target_pose) or self.target_pose
        self.calls = 0
        self.moved_m = 0.0
        self.candidate_id = ""
        self._search_seq = 0
        self._task_seq = 0

    # -- service contract -------------------------------------------------- #

    def search(
        self,
        ctx: Any,
        target: str,
        *,
        relation: str = "",
        instruction: str | None = None,
        max_steps: int = 8,
        max_turns: int = 1,
    ) -> dict:
        text = str(instruction or "").strip() or compose_instruction(
            str(target or ""), str(relation or "")
        )
        before = self.calls
        status, data = self._invoke(
            "search_target",
            {
                "instruction": text,
                "max_steps": max(1, int(max_steps or 8)),
                "budget_turns": max(1, int(max_turns or 1)),
            },
            ctx,
        )
        candidate = str(data.get("candidate_id") or "")
        if candidate:
            self.candidate_id = candidate
        return self._reply(
            "search_target",
            ok=bool(candidate),
            status=status or ("ok" if candidate else STATUS_FAILED),
            candidate_id=candidate,
            calls=self.calls - before,
            error="" if candidate else "search_target returned no candidate_id",
            data=data,
        )

    def approach(
        self,
        ctx: Any,
        candidate_id: str | None = None,
        *,
        standoff_m: float = 1.2,
        max_legs: int = 8,
    ) -> dict:
        candidate = str(candidate_id or self.candidate_id or "")
        if not candidate:
            return self._blank("approach_candidate", "approach requires a candidate id")
        before = self.calls
        status, data = self._invoke(
            "approach_candidate",
            {"candidate_id": candidate, "standoff_m": float(standoff_m)},
            ctx,
        )
        arrived = data.get("arrived") is True
        return self._reply(
            "approach_candidate",
            ok=arrived,
            status=status or ("arrived" if arrived else STATUS_FAILED),
            candidate_id=str(data.get("candidate_id") or candidate),
            task_id=str(data.get("task_id") or ""),
            arrived=arrived,
            calls=self.calls - before,
            error="" if arrived else str(data.get("error") or "arrived=false"),
            data=data,
        )

    def confirm(
        self,
        ctx: Any,
        candidate_id: str | None = None,
        *,
        instruction: str | None = None,
        relation: str | None = None,
        max_views: int = 2,
    ) -> dict:
        candidate = str(candidate_id or self.candidate_id or "")
        if not candidate:
            return self._blank("confirm_candidate", "confirm requires a candidate id")
        args: dict[str, Any] = {
            "candidate_id": candidate,
            "max_views": max(1, int(max_views or 2)),
        }
        if instruction:
            args["instruction"] = str(instruction)
        if relation:
            args["relation"] = str(relation)
        before = self.calls
        status, data = self._invoke("confirm_candidate", args, ctx)
        accepted = str(data.get("verdict") or "").lower() == "accepted"
        return self._reply(
            "confirm_candidate",
            ok=accepted,
            status=str(data.get("verdict") or status or STATUS_FAILED),
            candidate_id=str(data.get("candidate_id") or candidate),
            task_id=str(data.get("task_id") or ""),
            accepted=accepted,
            calls=self.calls - before,
            error="" if accepted else str(data.get("error") or "verdict is not accepted"),
            data=data,
        )

    def verify(
        self,
        ctx: Any,
        candidate_id: str | None = None,
        *,
        success_dist_m: float | None = None,
    ) -> dict:
        candidate = str(candidate_id or self.candidate_id or "")
        if not candidate:
            return self._blank("verify_goal", "verify requires a candidate id")
        args: dict[str, Any] = {"candidate_id": candidate}
        if success_dist_m is not None:
            args["success_dist_m"] = float(success_dist_m)
        before = self.calls
        status, data = self._invoke("verify_goal", args, ctx)
        verified = data.get("verified") is True
        return self._reply(
            "verify_goal",
            ok=verified,
            status=status or ("verified" if verified else STATUS_FAILED),
            candidate_id=str(data.get("candidate_id") or candidate),
            task_id=str(data.get("task_id") or ""),
            verified=verified,
            calls=self.calls - before,
            error="" if verified else str(data.get("error") or "verified=false"),
            data=data,
        )

    def revisit(
        self,
        ctx: Any,
        *,
        position: Any = None,
        observation_id: str | None = None,
        viewpoint_id: str | None = None,
        standoff_m: float = 0.6,
    ) -> dict:
        observation = str(observation_id or "").strip()
        viewpoint = str(viewpoint_id or "").strip()
        point = _point(position) if position is not None else None
        before = self.calls
        if observation:
            status, data = self._invoke(
                "return_to_observation",
                {"observation_id": observation, "standoff_m": float(standoff_m)},
                ctx,
            )
        elif viewpoint:
            status, data = self._invoke(
                "go_to_viewpoint",
                {"viewpoint_id": viewpoint, "standoff_m": float(standoff_m)},
                ctx,
            )
        elif point is not None:
            status, proposal = self._invoke(
                "propose_viewpoints",
                {"target_hypothesis": point, "count": 3},
                ctx,
            )
            viewpoint = str(proposal.get("viewpoint_id") or "")
            if not viewpoint:
                return self._blank(
                    "propose_viewpoints",
                    "propose_viewpoints returned no viewpoint",
                    calls=self.calls - before,
                )
            status, data = self._invoke(
                "go_to_viewpoint",
                {"viewpoint_id": viewpoint, "standoff_m": float(standoff_m)},
                ctx,
            )
        else:
            return self._blank(
                "revisit", "revisit requires position, observation_id or viewpoint_id"
            )
        arrived = data.get("arrived") is True
        return self._reply(
            "go_to_viewpoint",
            ok=arrived,
            status=status or ("arrived" if arrived else STATUS_FAILED),
            task_id=str(data.get("task_id") or ""),
            arrived=arrived,
            calls=self.calls - before,
            error="" if arrived else str(data.get("error") or "arrived=false"),
            data=data,
        )

    def goto(
        self,
        ctx: Any,
        position: Any,
        *,
        standoff_m: float = 0.6,
    ) -> dict:
        return self.revisit(ctx, position=position, standoff_m=standoff_m)

    # -- tool handlers ----------------------------------------------------- #

    def _tool_specs(self) -> list[ToolSpec]:
        object_schema = {"type": "object", "properties": {}}

        def spec(name: str, handler, *, side_effects: bool) -> ToolSpec:
            return ToolSpec(
                name=name,
                description=f"Deterministic M4 mock of {name}.",
                input_schema=dict(object_schema),
                handler=handler,
                side_effects=side_effects,
                cacheable=False,
            )

        return [
            spec("search_target", self._handle_search, side_effects=False),
            spec("approach_candidate", self._handle_approach, side_effects=True),
            spec("confirm_candidate", self._handle_confirm, side_effects=True),
            spec("verify_goal", self._handle_verify, side_effects=True),
            spec("propose_viewpoints", self._handle_propose, side_effects=False),
            spec("go_to_viewpoint", self._handle_travel, side_effects=True),
            spec("return_to_observation", self._handle_travel, side_effects=True),
        ]

    def _handle_search(self, ctx: ToolContext, args: Mapping) -> ToolResult:
        self._search_seq += 1
        self.candidate_id = f"cand_mock_{self.seed}_{self._search_seq:02d}"
        return ToolResult(
            STATUS_OK,
            {
                "candidate_id": self.candidate_id,
                "confirmed": False,
                "instruction": str(args.get("instruction", "") or ""),
                "max_steps": int(args.get("max_steps", 8) or 8),
                "budget_turns": int(args.get("budget_turns", 1) or 1),
            },
        )

    def _handle_approach(self, ctx: ToolContext, args: Mapping) -> ToolResult:
        candidate = str(args.get("candidate_id") or self.candidate_id)
        moved = self._move_toward(
            ctx, float(args.get("standoff_m", 1.2) or 0.0)
        )
        self._task_seq += 1
        return ToolResult(
            STATUS_OK,
            {
                "candidate_id": candidate,
                "task_id": f"task_mock_{self.seed}_{self._task_seq:02d}",
                "task_status": "arrived",
                "arrived": True,
                "standoff_m": float(args.get("standoff_m", 1.2) or 0.0),
                "moved_m": round(moved, 3),
            },
        )

    def _handle_confirm(self, ctx: ToolContext, args: Mapping) -> ToolResult:
        self._task_seq += 1
        return ToolResult(
            STATUS_OK,
            {
                "candidate_id": str(args.get("candidate_id") or self.candidate_id),
                "task_id": f"task_mock_{self.seed}_{self._task_seq:02d}",
                "verdict": "accepted",
                "accepted": True,
                "max_views": int(args.get("max_views", 2) or 2),
            },
        )

    def _handle_verify(self, ctx: ToolContext, args: Mapping) -> ToolResult:
        self._task_seq += 1
        return ToolResult(
            STATUS_OK,
            {
                "candidate_id": str(args.get("candidate_id") or self.candidate_id),
                "task_id": f"task_mock_{self.seed}_{self._task_seq:02d}",
                "verified": True,
                "distance_m": round(self._distance_to_target(ctx), 3),
                "success_dist_m": args.get("success_dist_m"),
            },
        )

    def _handle_propose(self, ctx: ToolContext, args: Mapping) -> ToolResult:
        point = _point(args.get("target_hypothesis")) or self._target(ctx)
        return ToolResult(
            STATUS_OK,
            {
                "viewpoint_id": f"vp_mock_{self.seed}_{self._task_seq + 1:02d}",
                "target_hypothesis": point,
                "viewpoints": [
                    {
                        "viewpoint_id": f"vp_mock_{self.seed}_{self._task_seq + 1:02d}",
                        "position": point,
                    }
                ],
            },
        )

    def _handle_travel(self, ctx: ToolContext, args: Mapping) -> ToolResult:
        moved = self._move_toward(
            ctx, float(args.get("standoff_m", 0.6) or 0.0)
        )
        self._task_seq += 1
        data: dict[str, Any] = {
            "task_id": f"task_mock_{self.seed}_{self._task_seq:02d}",
            "task_status": "arrived",
            "arrived": True,
            "moved_m": round(moved, 3),
        }
        if args.get("viewpoint_id"):
            data["viewpoint_id"] = str(args["viewpoint_id"])
        if args.get("observation_id"):
            data["observation_id"] = str(args["observation_id"])
        return ToolResult(STATUS_OK, data)

    # -- internals --------------------------------------------------------- #

    def _invoke(
        self, name: str, args: Mapping, ctx: ToolContext
    ) -> tuple[str, dict]:
        self.calls += 1
        try:
            result = self.registry.call(name, dict(args), ctx)
        except Exception as exc:  # noqa: BLE001 - the contract reports, never raises
            return STATUS_FAILED, {"error": f"{type(exc).__name__}: {exc}"}
        status = str(getattr(result, "status", "") or "")
        data = getattr(result, "data", None)
        return status, dict(data) if isinstance(data, Mapping) else {}

    def _target(self, ctx: Any) -> list[float]:
        extra = getattr(ctx, "extra", None)
        if isinstance(extra, Mapping):
            value = _point(extra.get("mission.target_pose"))
            if value is not None:
                return value
        return list(self.target_pose)

    def _distance_to_target(self, ctx: Any) -> float:
        world = getattr(ctx, "world", None)
        try:
            position = np.asarray(world.position_flu, dtype=float).reshape(-1)[:3]
        except (AttributeError, TypeError, ValueError):
            return 0.0
        return float(np.linalg.norm(np.asarray(self._target(ctx)) - position))

    def _move_toward(self, ctx: Any, standoff_m: float) -> float:
        world = getattr(ctx, "world", None)
        if world is None:
            return 0.0
        try:
            position = np.asarray(world.position_flu, dtype=float).reshape(-1)[:3]
        except (AttributeError, TypeError, ValueError):
            return 0.0
        target = np.asarray(self._target(ctx), dtype=float)
        delta = target - position
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-9:
            return 0.0
        stop = max(0.0, min(float(standoff_m or 0.0), distance))
        travel = min(self.step_m, distance - stop)
        if travel <= 1e-9:
            return 0.0
        new = position + delta / distance * travel
        if not _set_world_position(world, new):
            return 0.0
        self.moved_m += travel
        return travel

    @staticmethod
    def _blank(tool: str, error: str, *, calls: int = 0) -> dict:
        return {
            "ok": False,
            "tool": tool,
            "status": STATUS_FAILED,
            "candidate_id": "",
            "task_id": "",
            "arrived": False,
            "accepted": False,
            "verified": False,
            "calls": max(0, int(calls)),
            "error": str(error),
            "data": {},
        }

    @staticmethod
    def _reply(
        tool: str,
        *,
        ok: bool,
        status: str,
        candidate_id: str = "",
        task_id: str = "",
        arrived: bool = False,
        accepted: bool = False,
        verified: bool = False,
        calls: int = 0,
        error: str = "",
        data: Mapping | None = None,
    ) -> dict:
        return {
            "ok": bool(ok),
            "tool": tool,
            "status": str(status or ""),
            "candidate_id": str(candidate_id or ""),
            "task_id": str(task_id or ""),
            "arrived": bool(arrived),
            "accepted": bool(accepted),
            "verified": bool(verified),
            "calls": max(0, int(calls)),
            "error": str(error or ""),
            "data": dict(data or {}),
        }


# ---------------------------------------------------------------------------
# task files
# ---------------------------------------------------------------------------


class TaskFileError(ValueError):
    """A ``data/mixed_missions/tasks_<n>.json`` file is invalid."""


def _mission_categories(mission: Mapping) -> list[str]:
    return [
        "motion" if str(step.get("kind")) in MOTION_STEP_KINDS else "semantic"
        for step in mission.get("steps") or []
        if isinstance(step, Mapping)
    ]


def validate_mission(group: str, mission: Mapping, stages: int) -> None:
    name = str(mission.get("name") or "").strip()
    if not name:
        raise TaskFileError(f"{group}/{stages}: mission is missing 'name'")
    steps = mission.get("steps")
    if isinstance(steps, (str, bytes)) or not isinstance(steps, Sequence):
        raise TaskFileError(f"{group}/{stages}/{name}: missing 'steps' array")
    if len(steps) != stages:
        raise TaskFileError(
            f"{group}/{stages}/{name}: {len(steps)} steps, expected {stages}"
        )
    instruction = str(mission.get("instruction") or "").strip()
    if not instruction:
        raise TaskFileError(f"{group}/{stages}/{name}: instruction is required")
    try:
        MixedMissionSpec.from_dict(
            {"steps": list(steps), "instruction": instruction}
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as a task error
        raise TaskFileError(
            f"{group}/{stages}/{name}: invalid mixed mission: {exc}"
        ) from exc

    kinds = [str(step.get("kind", "")) for step in steps if isinstance(step, Mapping)]
    if group == "action":
        bad = [kind for kind in kinds if kind not in MOTION_STEP_KINDS]
        if bad:
            raise TaskFileError(
                f"{group}/{stages}/{name}: non-motion steps in the action group: {bad}"
            )
    elif group == "semantic":
        bad = [kind for kind in kinds if kind not in SEMANTIC_GROUP_KINDS]
        if bad:
            raise TaskFileError(
                f"{group}/{stages}/{name}: unsupported semantic steps: {bad}"
            )
    elif group == "mixed":
        categories = _mission_categories(mission)
        if "motion" not in categories or "semantic" not in categories:
            raise TaskFileError(
                f"{group}/{stages}/{name}: mixed mission needs motion and semantic steps"
            )
        for index in range(1, len(categories)):
            if categories[index] == categories[index - 1]:
                raise TaskFileError(
                    f"{group}/{stages}/{name}: steps do not alternate at index {index}"
                )
    else:
        raise TaskFileError(f"unknown group {group!r}")


def load_tasks(tasks_dir: str | Path, stages: int) -> dict[str, list[dict]]:
    """Load and validate one ``tasks_<stages>.json`` file."""
    path = Path(tasks_dir) / f"tasks_{int(stages)}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TaskFileError(f"cannot read task file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TaskFileError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise TaskFileError(f"{path}: task file must be an object")
    schema = str(data.get("schema") or TASKS_SCHEMA)
    if schema != TASKS_SCHEMA:
        raise TaskFileError(f"{path}: unknown tasks schema {schema!r}")
    if int(data.get("stages", 0) or 0) != int(stages):
        raise TaskFileError(
            f"{path}: 'stages' is {data.get('stages')!r}, expected {stages}"
        )
    groups_data = data.get("groups")
    if not isinstance(groups_data, Mapping):
        raise TaskFileError(f"{path}: missing 'groups' object")
    result: dict[str, list[dict]] = {}
    for group in GROUPS:
        missions = groups_data.get(group)
        if missions is None:
            raise TaskFileError(f"{path}: missing group {group!r}")
        if isinstance(missions, (str, bytes)) or not isinstance(missions, Sequence):
            raise TaskFileError(f"{path}: group {group!r} must be an array")
        if not missions:
            raise TaskFileError(f"{path}: group {group!r} is empty")
        result[group] = []
        for mission in missions:
            if not isinstance(mission, Mapping):
                raise TaskFileError(
                    f"{path}: {group} entry must be an object"
                )
            validate_mission(group, mission, int(stages))
            result[group].append(dict(mission))
    return result


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class MatrixConfig:
    groups: tuple[str, ...] = GROUPS
    stages: tuple[int, ...] = STAGE_COUNTS
    seeds: int = 2
    world: str = "cpu"
    ply: Path = DEFAULT_PLY
    target_pos: tuple[float, float, float] = DEFAULT_TARGET_POS
    start_pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
    start_yaw_deg: float = 0.0
    target_radius: float = 0.2
    width: int = 64
    height: int = 36
    bound_radius: float = 20.0
    out_dir: Path = DEFAULT_OUT
    plugin_root: Path | None = None
    artifacts_dir: Path | None = None
    tasks_dir: Path = DEFAULT_TASKS_DIR
    interrupt_step: int = 0
    max_tool_calls: int = 256
    semantic: str = "mock"
    resume: bool = False
    device: str = "cuda"

    def __post_init__(self) -> None:
        self.groups = tuple(self.groups)
        self.stages = tuple(int(stage) for stage in self.stages)
        bad_groups = [group for group in self.groups if group not in GROUPS]
        if bad_groups:
            raise ValueError(f"unknown groups: {bad_groups}")
        bad_stages = [stage for stage in self.stages if stage not in STAGE_COUNTS]
        if bad_stages:
            raise ValueError(f"unknown stage counts: {bad_stages}")
        if self.seeds < 1:
            raise ValueError("seeds must be >= 1")
        if self.interrupt_step < 0:
            raise ValueError("interrupt_step must be >= 0")
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be >= 1")
        if self.semantic not in ("mock", "real"):
            raise ValueError(f"unknown semantic backend {self.semantic!r}")
        if self.world != "cpu":
            raise ValueError("this driver supports --world cpu only")
        self.ply = Path(self.ply)
        self.out_dir = Path(self.out_dir)
        self.tasks_dir = Path(self.tasks_dir)
        self.plugin_root = (
            Path(self.plugin_root) if self.plugin_root else self.out_dir / "_plugins"
        )
        self.artifacts_dir = (
            Path(self.artifacts_dir) if self.artifacts_dir else self.out_dir / "artifacts"
        )


def _int_list(text: str) -> tuple[int, ...]:
    values = tuple(
        int(part.strip()) for part in str(text or "").split(",") if part.strip()
    )
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _group_list(text: str) -> tuple[str, ...]:
    values = tuple(
        part.strip() for part in str(text or "").split(",") if part.strip()
    )
    unknown = [value for value in values if value not in GROUPS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown groups {unknown}; choose from {list(GROUPS)}"
        )
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 2/4/8-stage mixed-mission matrix on the CPU fixture and "
            "write report.json/report.md (schema vlm_nav.mixed_missions.v1)."
        )
    )
    parser.add_argument(
        "--groups", type=_group_list, default=GROUPS,
        help="comma separated: action,semantic,mixed",
    )
    parser.add_argument(
        "--stages", type=_int_list, default=STAGE_COUNTS,
        help="comma separated stage counts loaded from data/mixed_missions",
    )
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--world", choices=("cpu",), default="cpu")
    parser.add_argument("--ply", default=str(DEFAULT_PLY))
    parser.add_argument(
        "--target-pos", nargs=3, type=float, default=DEFAULT_TARGET_POS,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--start-pos", nargs=3, type=float, default=(0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--start-yaw-deg", type=float, default=0.0)
    parser.add_argument("--target-radius", type=float, default=0.2)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=36)
    parser.add_argument("--bound-radius", type=float, default=20.0)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--plugin-root",
        help="plugin lifecycle root (default: <out>/_plugins)",
    )
    parser.add_argument(
        "--artifacts-dir",
        help="mission artifacts/states (default: <out>/artifacts)",
    )
    parser.add_argument("--tasks-dir", default=str(DEFAULT_TASKS_DIR))
    parser.add_argument(
        "--interrupt-step", type=int, default=0,
        help="cancel each episode after N completed steps, then resume",
    )
    parser.add_argument(
        "--max-tool-calls", type=int, default=256,
        help="per-episode registry tool-call budget",
    )
    parser.add_argument(
        "--semantic", choices=("mock", "real"), default="mock",
        help="deterministic mock (default) or the installed semantic plugin stack",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume interrupted episodes (implied by --interrupt-step)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="hardware snapshot device; CPU-only hosts report available=false",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> MatrixConfig:
    return MatrixConfig(
        groups=tuple(args.groups),
        stages=tuple(args.stages),
        seeds=int(args.seeds),
        world=str(args.world),
        ply=Path(args.ply),
        target_pos=tuple(float(value) for value in args.target_pos),
        start_pos=tuple(float(value) for value in args.start_pos),
        start_yaw_deg=float(args.start_yaw_deg),
        target_radius=float(args.target_radius),
        width=int(args.width),
        height=int(args.height),
        bound_radius=float(args.bound_radius),
        out_dir=Path(args.out),
        plugin_root=Path(args.plugin_root) if args.plugin_root else None,
        artifacts_dir=Path(args.artifacts_dir) if args.artifacts_dir else None,
        tasks_dir=Path(args.tasks_dir),
        interrupt_step=int(args.interrupt_step),
        max_tool_calls=int(args.max_tool_calls),
        semantic=str(args.semantic),
        resume=bool(args.resume),
        device=str(args.device),
    )


# ---------------------------------------------------------------------------
# world / plugin stack
# ---------------------------------------------------------------------------


def _make_world(config: MatrixConfig) -> SplatWorldCPU:
    return SplatWorldCPU(
        config.ply,
        TargetSpec(
            name="semantic_target",
            position_flu=tuple(float(value) for value in config.target_pos),
            radius_m=float(config.target_radius),
        ),
        start_flu=tuple(float(value) for value in config.start_pos),
        start_yaw_deg=float(config.start_yaw_deg),
        width=int(config.width),
        height=int(config.height),
        bound_radius=float(config.bound_radius),
    )


def _real_plugin_classes() -> dict[str, Any]:
    """Import the legacy/semantic stack; ImportError is reported by the caller."""
    from vlm_nav.plugins.legacy import (  # noqa: PLC0415 - optional workstream
        NavigationPlugin,
        PerceptionPlugin,
        TaskToolsPlugin,
    )
    from vlm_nav.plugins.semantic import (  # noqa: PLC0415
        SemanticNavigationPlugin,
    )

    return {
        "perception": PerceptionPlugin,
        "navigation": NavigationPlugin,
        "task_tools": TaskToolsPlugin,
        "semantic": SemanticNavigationPlugin,
    }


def _plugins_for(
    group: str,
    mode: str,
    artifacts_dir: Path,
    real_classes: Mapping[str, Any] | None,
) -> list[Any]:
    motion = MotionPlugin(artifacts_dir=artifacts_dir)
    if group == "action":
        return [motion]
    if mode == "mock":
        return [
            motion,
            TaskPlannerPlugin(),
            MixedMissionPlugin(artifacts_dir=artifacts_dir),
        ]
    if real_classes is None:
        raise RuntimeError("semantic backend is not available")
    return [
        motion,
        TaskPlannerPlugin(),
        real_classes["perception"](),
        real_classes["navigation"](),
        real_classes["task_tools"](),
        real_classes["semantic"](),
        MixedMissionPlugin(artifacts_dir=artifacts_dir),
    ]


def _probe_lifecycle(
    registry: ToolRegistry,
    config: MatrixConfig,
    artifacts_dir: Path,
) -> dict:
    """Call one tool of each activated plugin family, without touching episodes."""
    world = _make_world(config)
    ctx = ToolContext(
        world=world,
        episode_id="lifecycle_probe",
        budget={"tool_calls_left": 16},
        extra={"mission.artifacts_dir": str(artifacts_dir)},
    )
    result: dict[str, Any] = {}
    names = set(registry.names())
    if "plan_mission" in names:
        try:
            call = registry.call(
                "plan_mission", {"instruction": "悬停 0.1 秒"}, ctx
            )
            result["plan_mission"] = {
                "status": call.status,
                "steps": len(call.data.get("steps") or []),
                "error": str(call.data.get("error") or ""),
            }
        except Exception as exc:  # noqa: BLE001 - reported, not hidden
            result["plan_mission"] = {
                "status": STATUS_FAILED,
                "error": f"{type(exc).__name__}: {exc}",
            }
    if "motion_mission" in names:
        probe_dir = artifacts_dir / "probe"
        try:
            call = registry.call(
                "motion_mission",
                {
                    "steps": [{"kind": "hover", "seconds": 0.1}],
                    "artifacts_dir": str(probe_dir),
                    "state_path": str(probe_dir / "probe.state.json"),
                },
                ctx,
            )
            result["motion_mission"] = {
                "status": call.status,
                "mission_status": str(call.data.get("status") or ""),
                "error": str(call.data.get("error") or ""),
            }
        except Exception as exc:  # noqa: BLE001
            result["motion_mission"] = {
                "status": STATUS_FAILED,
                "error": f"{type(exc).__name__}: {exc}",
            }
    if "semantic_status" in names:
        try:
            call = registry.call("semantic_status", {}, ctx)
            result["semantic_status"] = {
                "status": call.status,
                "ready": bool(call.data.get("ready")),
                "missing": list(call.data.get("missing") or []),
            }
        except Exception as exc:  # noqa: BLE001
            result["semantic_status"] = {
                "status": STATUS_FAILED,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return result


# ---------------------------------------------------------------------------
# episode execution
# ---------------------------------------------------------------------------


def _safe_name(value: str) -> str:
    cleaned = _SAFE_NAME.sub("_", str(value or "")).strip("._")
    return cleaned or "mission"


def _motion_summary(record: Any) -> dict | None:
    state = getattr(record, "motion_state", None)
    if not isinstance(state, Mapping):
        return None
    steps = state.get("steps")
    if not isinstance(steps, Sequence) or not steps:
        return None
    progress = steps[0].get("progress") if isinstance(steps[0], Mapping) else None
    if not isinstance(progress, Mapping):
        return None
    keys = (
        "target",
        "requested_m",
        "moved_m",
        "remaining_m",
        "chunks",
        "requested_deg",
        "turned_deg",
        "remaining_deg",
        "requested_s",
        "elapsed_s",
        "remaining_s",
    )
    summary = {key: progress[key] for key in keys if key in progress}
    if not summary:
        return None
    return summary


def _step_progress(state: MixedMissionState) -> list[dict]:
    rows: list[dict] = []
    for index, record in enumerate(state.steps):
        rows.append(
            {
                "index": index,
                "kind": record.kind,
                "status": record.status,
                "attempts": record.attempts,
                "motion": _motion_summary(record),
                "evidence": dict(record.evidence),
            }
        )
    return rows


def _count_replayed(
    before: MixedMissionState, after: MixedMissionState
) -> int:
    replayed = 0
    for index, record in enumerate(before.steps):
        if record.status != MIXED_STEP_DONE:
            continue
        current = after.steps[index] if index < len(after.steps) else None
        if (
            current is None
            or current.status != MIXED_STEP_DONE
            or current.attempts != record.attempts
        ):
            replayed += 1
    return replayed


def _motion_progress_preserved(
    before: MixedMissionState, after: MixedMissionState
) -> bool:
    for index, record in enumerate(before.steps):
        saved = _motion_summary(record)
        if saved is None:
            continue
        current = _motion_summary(after.steps[index]) if index < len(after.steps) else None
        if current is None:
            return False
        if saved.get("target") is not None:
            try:
                if not np.allclose(
                    np.asarray(saved["target"], dtype=float),
                    np.asarray(current["target"], dtype=float),
                    atol=1e-6,
                ):
                    return False
            except (TypeError, ValueError):
                return False
        for key in ("requested_m", "requested_deg", "requested_s"):
            if key in saved and not _close(saved[key], current.get(key)):
                return False
        for key in ("remaining_m", "remaining_deg", "remaining_s"):
            if key in saved:
                if key not in current:
                    return False
                if float(current[key]) > float(saved[key]) + 1e-6:
                    return False
        if "chunks" in saved:
            if int(current.get("chunks") or 0) < int(saved["chunks"]):
                return False
    return True


def _close(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-6
    except (TypeError, ValueError):
        return False


def _semantic_evidence_ok(record: Any) -> bool:
    evidence = record.evidence if isinstance(record.evidence, Mapping) else {}
    if record.kind == "search":
        return bool(evidence.get("candidate_id"))
    if record.kind in ("approach", "revisit", "goto"):
        return evidence.get("arrived") is True
    if record.kind == "confirm":
        return evidence.get("accepted") is True
    if record.kind == "verify":
        return evidence.get("verified") is True
    return False


def false_completion(state: MixedMissionState | None) -> bool:
    """True when a completed state lacks the required semantic evidence."""
    if state is None or state.status != MIXED_COMPLETED:
        return False
    for record in state.steps:
        if record.status != MIXED_STEP_DONE:
            return True
        if record.kind in SEMANTIC_STEP_KINDS and not _semantic_evidence_ok(record):
            return True
    return False


def _episode_error(
    group: str,
    stages: int,
    seed: int,
    mission: Mapping,
    error: str,
    *,
    wall_time_s: float,
) -> dict:
    return {
        "group": group,
        "stages": stages,
        "seed": seed,
        "mission": str(mission.get("name") or ""),
        "instruction": str(mission.get("instruction") or ""),
        "steps_total": len(mission.get("steps") or []),
        "status": "error",
        "completed": False,
        "success": False,
        "false_completion": False,
        "error": error,
        "tool_calls": 0,
        "vlm_calls": 0,
        "wall_time_s": round(wall_time_s, 3),
        "interrupted": False,
        "resume": {
            "attempted": False,
            "recovered": False,
            "replayed_steps": 0,
            "motion_progress_preserved": True,
            "motion_remaining_preserved": True,
        },
        "state": None,
    }


def run_episode(
    *,
    group: str,
    stages: int,
    seed: int,
    mission: Mapping,
    config: MatrixConfig,
    registry: ToolRegistry,
    semantic_service: Any,
    artifacts_dir: Path,
) -> dict:
    """Run one mission, with optional interrupt/resume, and return its record."""
    started = time.perf_counter()
    name = str(mission.get("name") or "")
    steps = list(mission.get("steps") or [])
    interrupt_step = (
        int(config.interrupt_step)
        if 0 < int(config.interrupt_step) < len(steps)
        else 0
    )
    world = _make_world(config)
    initial = [float(value) for value in world.position_flu]
    state_path = (
        artifacts_dir
        / "states"
        / f"{group}_{stages}_seed{seed}_{_safe_name(name)}.state.json"
    )
    ctx = ToolContext(
        world=world,
        episode_id=f"{group}_{stages}_seed{seed}_{_safe_name(name)}",
        budget={"tool_calls_left": int(config.max_tool_calls)},
        extra={
            "mission.target_pose": [float(value) for value in config.target_pos],
            "mission.artifacts_dir": str(artifacts_dir),
            "mission.name": name,
            "mission.group": group,
        },
    )
    spec = MixedMissionSpec.from_dict(
        {"steps": steps, "instruction": str(mission.get("instruction") or "")}
    )
    mission_id = f"{group}_{stages}_{_safe_name(name)}"

    fired = {"value": False}
    done = {"value": 0}
    holder: dict[str, Any] = {}

    def on_progress(payload: Mapping) -> None:
        if payload.get("event") != "step.finished":
            return
        step = payload.get("step") or {}
        if step.get("status") != MIXED_STEP_DONE:
            return
        done["value"] += 1
        if done["value"] >= interrupt_step and not fired["value"]:
            fired["value"] = True
            executor = holder.get("executor")
            if executor is not None:
                executor.cancel(f"interrupt_after_{interrupt_step}_steps")

    executor = MixedMissionExecutor(
        world,
        registry=registry,
        ctx=ctx,
        motion_executor=MotionExecutor(
            world,
            clock=clock_for_world(world),
            artifacts_dir=artifacts_dir,
        ),
        semantic_service=semantic_service,
        artifacts_dir=artifacts_dir,
        state_path=state_path,
        on_progress=on_progress if interrupt_step else None,
    )
    holder["executor"] = executor
    state = executor.run(spec, mission_id=mission_id, instruction=spec.instruction)
    interrupted = state.status == MIXED_CANCELLED and fired["value"]
    resume_info: dict[str, Any] = {
        "attempted": False,
        "requested": bool(interrupt_step),
        "interrupt_step": interrupt_step,
        "recovered": False,
        "replayed_steps": 0,
        "motion_progress_preserved": True,
        "motion_remaining_preserved": True,
    }
    if interrupted:
        try:
            saved = load_state(state_path)
            resume_info.update(
                {
                    "attempted": True,
                    "status_before": saved.status,
                    "completed_before": saved.completed_steps(),
                    "before": _step_progress(saved),
                }
            )
            resumed = MixedMissionExecutor(
                world,
                registry=registry,
                ctx=ctx,
                motion_executor=MotionExecutor(
                    world,
                    clock=executor.clock,
                    artifacts_dir=artifacts_dir,
                ),
                semantic_service=semantic_service,
                artifacts_dir=artifacts_dir,
                state_path=state_path,
            )
            final = resumed.resume(saved)
            replayed = _count_replayed(saved, final)
            preserved = _motion_progress_preserved(saved, final)
            resume_info.update(
                {
                    "recovered": (
                        final.status == MIXED_COMPLETED and replayed == 0
                    ),
                    "replayed_steps": replayed,
                    "motion_progress_preserved": preserved,
                    "motion_remaining_preserved": preserved,
                    "status_after": final.status,
                    "completed_after": final.completed_steps(),
                    "after": _step_progress(final),
                }
            )
            state = final
        except Exception as exc:  # noqa: BLE001 - the episode reports the failure
            resume_info["error"] = f"{type(exc).__name__}: {exc}"
            resume_info["recovered"] = False
            try:
                state = load_state(state_path)
            except Exception:  # noqa: BLE001 - keep the cancelled state
                pass

    fc = false_completion(state)
    final_position = [float(value) for value in world.position_flu]
    target = np.asarray(config.target_pos, dtype=float)
    distance = float(np.linalg.norm(target - np.asarray(final_position)))
    return {
        "group": group,
        "stages": stages,
        "seed": seed,
        "mission": name,
        "instruction": spec.instruction,
        "steps_total": len(steps),
        "status": state.status,
        "completed": state.status == MIXED_COMPLETED,
        "success": state.status == MIXED_COMPLETED and not fc,
        "false_completion": fc,
        "error": str(state.error or resume_info.get("error") or ""),
        "tool_calls": len(ctx.log),
        "vlm_calls": int(ctx.extra.get("vlm_calls", 0) or 0),
        "wall_time_s": round(time.perf_counter() - started, 3),
        "interrupted": interrupted,
        "resume": resume_info,
        "initial_position": initial,
        "final_position": final_position,
        "target_position": [float(value) for value in config.target_pos],
        "distance_to_target_m": round(distance, 3),
        "state_path": str(state_path),
        "state_persisted": state_path.exists(),
        "state": state.to_dict(),
    }


# ---------------------------------------------------------------------------
# aggregation / report
# ---------------------------------------------------------------------------


def _zero_metrics() -> dict:
    return {
        "episodes": 0,
        "completed": 0,
        "success_rate": 0.0,
        "false_completion": 0,
        "tool_calls": 0,
        "vlm_calls": 0,
        "wall_time_s": 0.0,
        "peak_allocated_mib": None,
        "peak_reserved_mib": None,
        "resume_recovered": 0,
        "replayed_steps": 0,
        "interrupted": 0,
        "motion_progress_preserved": True,
        "motion_remaining_preserved": True,
        "clean_uninstall": False,
    }


def aggregate_metrics(episodes: Sequence[Mapping], hardware: Mapping) -> dict:
    metrics = _zero_metrics()
    total = len(episodes)
    successes = sum(1 for episode in episodes if episode.get("success"))
    interrupted = [episode for episode in episodes if episode.get("interrupted")]
    metrics.update(
        {
            "episodes": total,
            "completed": sum(1 for episode in episodes if episode.get("completed")),
            "success_rate": round(successes / total, 4) if total else 0.0,
            "false_completion": sum(
                1 for episode in episodes if episode.get("false_completion")
            ),
            "tool_calls": sum(int(episode.get("tool_calls", 0) or 0) for episode in episodes),
            "vlm_calls": sum(int(episode.get("vlm_calls", 0) or 0) for episode in episodes),
            "wall_time_s": round(
                sum(float(episode.get("wall_time_s", 0.0) or 0.0) for episode in episodes),
                3,
            ),
            "resume_recovered": sum(
                1 for episode in interrupted
                if (episode.get("resume") or {}).get("recovered")
            ),
            "replayed_steps": sum(
                int((episode.get("resume") or {}).get("replayed_steps", 0) or 0)
                for episode in interrupted
            ),
            "interrupted": len(interrupted),
            "motion_progress_preserved": all(
                (episode.get("resume") or {}).get("motion_progress_preserved", True)
                for episode in interrupted
            ),
            "motion_remaining_preserved": all(
                (episode.get("resume") or {}).get("motion_remaining_preserved", True)
                for episode in interrupted
            ),
        }
    )
    if hardware.get("available"):
        metrics["peak_allocated_mib"] = hardware.get("peak_allocated_mib")
        metrics["peak_reserved_mib"] = hardware.get("peak_reserved_mib")
    return metrics


def _leftover_files(root: Path) -> list[str]:
    if not root.exists():
        return []
    try:
        return [
            str(path) for path in sorted(root.rglob("*")) if path.is_file()
        ]
    except OSError:
        return []


def _collect_resources(
    host: PluginHost,
    plugins: Sequence[Any],
    plugin_root: Path,
    deactivate_reports: Sequence[Mapping],
    uninstall_reports: Sequence[Mapping],
) -> dict:
    catalog = host.catalog()
    residual: list[dict] = []
    for report in list(deactivate_reports) + list(uninstall_reports):
        for entry in report.get("residual") or []:
            residual.append(dict(entry) if isinstance(entry, Mapping) else {"error": str(entry)})
    worker_pids: list[int] = []
    for plugin in plugins:
        worker = getattr(plugin, "worker", None)
        pid = getattr(worker, "pid", None)
        alive = bool(getattr(worker, "alive", False))
        if pid and alive:
            worker_pids.append(int(pid))
    leftovers = _leftover_files(plugin_root)
    plugin_count = len(catalog.get("plugins") or [])
    service_count = len(catalog.get("services") or {})
    tool_count = len(catalog.get("tools") or [])
    prompt_count = len(catalog.get("prompts") or [])
    clean = (
        plugin_count == 0
        and service_count == 0
        and tool_count == 0
        and prompt_count == 0
        and not residual
        and not worker_pids
        and not leftovers
    )
    return {
        "clean": clean,
        "plugin_catalog_count": plugin_count,
        "service_count": service_count,
        "tool_count": tool_count,
        "prompt_count": prompt_count,
        "residual": residual,
        "worker_pids": worker_pids,
        "leftover_paths": leftovers,
    }


def _skipped_cell(
    group: str, stages: int, config: MatrixConfig, reason: str
) -> dict:
    metrics = _zero_metrics()
    metrics["skipped"] = True
    resources = {
        "clean": False,
        "skipped": True,
        "plugin_catalog_count": 0,
        "service_count": 0,
        "tool_count": 0,
        "prompt_count": 0,
        "residual": [],
        "worker_pids": [],
        "leftover_paths": [],
        "reason": reason,
    }
    metrics["resources_after_uninstall"] = resources
    return {
        "group": group,
        "stages": stages,
        "skipped": True,
        "error": reason,
        "episodes": [],
        "metrics": metrics,
        "hardware": snapshot(config.device),
        "lifecycle": {
            "install": [],
            "activate": [],
            "deactivate": [],
            "uninstall": [],
            "probe": {},
            "active": [],
        },
        "resources_after_uninstall": resources,
    }


def run_cell(
    group: str,
    stages: int,
    missions: Sequence[Mapping],
    config: MatrixConfig,
    real_classes: Mapping[str, Any] | None,
) -> dict:
    started = time.perf_counter()
    cell_artifacts = config.artifacts_dir / f"{group}_{stages}"
    cell_artifacts.mkdir(parents=True, exist_ok=True)
    plugin_root = config.plugin_root / f"{group}_{stages}"
    registry = ToolRegistry()
    host = PluginHost(root_dir=plugin_root, tool_registry=registry)
    plugins: list[Any] = []
    mock: MockSemanticService | None = None
    lifecycle: dict[str, Any] = {
        "install": [],
        "activate": [],
        "deactivate": [],
        "uninstall": [],
        "probe": {},
        "active": [],
    }
    episodes: list[dict] = []
    error = ""
    hardware = snapshot(config.device)
    try:
        plugins = _plugins_for(
            group, config.semantic, cell_artifacts, real_classes
        )
        lifecycle["install"] = [
            report.to_dict() for report in host.install_bundle(plugins)
        ]
        for plugin in plugins:
            lifecycle["activate"].append(
                host.activate(plugin.manifest.id).to_dict()
            )
        lifecycle["active"] = [
            plugin.manifest.id
            for plugin in plugins
            if host.status(plugin.manifest.id) == "active"
        ]
        lifecycle["probe"] = _probe_lifecycle(
            registry, config, cell_artifacts
        )

        semantic_service: Any = None
        if config.semantic == "mock" and group != "action":
            mock = MockSemanticService(
                registry, target_pose=config.target_pos
            )
            mock.register()
            semantic_service = mock
        elif config.semantic == "real" and group != "action":
            semantic_service = host.services.get(SEMANTIC_SERVICE)

        reset_peak(config.device)
        # Every task fixture in the cell runs; --seeds repeats each task. The
        # old mission[seed % n] round-robin silently skipped tasks when seeds
        # was smaller than the task list (e.g. the 2-stage revisit task was
        # never run with the default seed count).
        for mission_template in missions:
            mission = dict(mission_template)
            for seed in range(int(config.seeds)):
                if mock is not None:
                    mock.reset(seed=seed, target_pose=config.target_pos)
                try:
                    episodes.append(
                        run_episode(
                            group=group,
                            stages=stages,
                            seed=seed,
                            mission=mission,
                            config=config,
                            registry=registry,
                            semantic_service=semantic_service,
                            artifacts_dir=cell_artifacts,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - one episode cannot stop the matrix
                    episodes.append(
                        _episode_error(
                            group,
                            stages,
                            seed,
                            mission,
                            f"{type(exc).__name__}: {exc}",
                            wall_time_s=0.0,
                        )
                    )
        hardware = snapshot(config.device)
    except Exception as exc:  # noqa: BLE001 - reported in the cell, teardown still runs
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if mock is not None:
            try:
                mock.close()
            except Exception as exc:  # noqa: BLE001
                error = error or f"mock cleanup failed: {type(exc).__name__}: {exc}"
        try:
            installed_ids = {
                str(item.get("id")) for item in host.plugins()
            }
        except Exception:  # noqa: BLE001 - fall back to the planned list
            installed_ids = {plugin.manifest.id for plugin in plugins}
        teardown = [
            plugin for plugin in plugins
            if plugin.manifest.id in installed_ids
        ]
        for plugin in reversed(teardown):
            plugin_id = plugin.manifest.id
            try:
                lifecycle["deactivate"].append(
                    host.deactivate(plugin_id, force=True).to_dict()
                )
            except Exception as exc:  # noqa: BLE001 - teardown reports, never hides
                lifecycle["deactivate"].append(
                    {
                        "plugin_id": plugin_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        for plugin in reversed(teardown):
            plugin_id = plugin.manifest.id
            try:
                lifecycle["uninstall"].append(
                    host.uninstall(plugin_id, purge_data=True).to_dict()
                )
            except Exception as exc:  # noqa: BLE001
                lifecycle["uninstall"].append(
                    {
                        "plugin_id": plugin_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    resources = _collect_resources(
        host,
        plugins,
        plugin_root,
        lifecycle["deactivate"],
        lifecycle["uninstall"],
    )
    metrics = aggregate_metrics(episodes, hardware)
    metrics["clean_uninstall"] = bool(resources["clean"])
    metrics["resources_after_uninstall"] = resources
    for directory in (plugin_root, config.plugin_root):
        try:
            directory.rmdir()
        except OSError:
            pass
    cell = {
        "group": group,
        "stages": stages,
        "skipped": False,
        "error": error,
        "episodes": episodes,
        "metrics": metrics,
        "hardware": hardware,
        "lifecycle": lifecycle,
        "resources_after_uninstall": resources,
        "wall_time_s": round(time.perf_counter() - started, 3),
    }
    return cell


def _aggregate_resources(cells: Sequence[Mapping]) -> dict:
    residual: list[dict] = []
    worker_pids: list[int] = []
    leftover_paths: list[str] = []
    plugin_count = service_count = tool_count = prompt_count = 0
    considered = 0
    for cell in cells:
        if cell.get("skipped"):
            continue
        resources = cell.get("resources_after_uninstall") or {}
        considered += 1
        plugin_count += int(resources.get("plugin_catalog_count", 0) or 0)
        service_count += int(resources.get("service_count", 0) or 0)
        tool_count += int(resources.get("tool_count", 0) or 0)
        prompt_count += int(resources.get("prompt_count", 0) or 0)
        for entry in resources.get("residual") or []:
            residual.append(
                {
                    "group": cell.get("group"),
                    "stages": cell.get("stages"),
                    **dict(entry),
                }
            )
        worker_pids.extend(int(pid) for pid in resources.get("worker_pids") or [])
        leftover_paths.extend(
            str(path) for path in resources.get("leftover_paths") or []
        )
    clean = (
        plugin_count == 0
        and service_count == 0
        and tool_count == 0
        and prompt_count == 0
        and not residual
        and not worker_pids
        and not leftover_paths
    )
    return {
        "clean": clean,
        "cells_checked": considered,
        "plugin_catalog_count": plugin_count,
        "service_count": service_count,
        "tool_count": tool_count,
        "prompt_count": prompt_count,
        "residual": residual,
        "worker_pids": sorted(set(worker_pids)),
        "leftover_paths": leftover_paths,
        "cells": [
            {
                "group": cell.get("group"),
                "stages": cell.get("stages"),
                "clean": bool(
                    (cell.get("resources_after_uninstall") or {}).get("clean")
                ),
            }
            for cell in cells
        ],
    }


def _summary(cells: Sequence[Mapping]) -> dict:
    episodes = [episode for cell in cells for episode in cell.get("episodes") or []]
    total = len(episodes)
    successes = sum(1 for episode in episodes if episode.get("success"))
    return {
        "cells": len(cells),
        "episodes": total,
        "completed": sum(1 for episode in episodes if episode.get("completed")),
        "success_rate": round(successes / total, 4) if total else 0.0,
        "false_completion": sum(
            1 for episode in episodes if episode.get("false_completion")
        ),
        "skipped_cells": sum(1 for cell in cells if cell.get("skipped")),
    }


def _simple_table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def _markdown(report: Mapping) -> str:
    config = report.get("config") or {}
    backend = report.get("semantic_backend") or {}
    summary = report.get("summary") or {}
    lines = [
        f"# Mixed mission matrix ({report.get('schema')})",
        "",
        f"- generated: {report.get('generated_at')}",
        f"- groups: {', '.join(config.get('groups') or [])}",
        f"- stages: {', '.join(str(stage) for stage in config.get('stages') or [])}",
        f"- seeds per cell: {config.get('seeds')}",
        f"- semantic backend: {backend.get('mode')} "
        f"(available={backend.get('available')})",
        f"- interrupt step: {config.get('interrupt_step') or 'none'}",
        f"- episodes: {summary.get('episodes')}, "
        f"success_rate={summary.get('success_rate')}, "
        f"false_completion={summary.get('false_completion')}",
        "",
        "## Metrics",
        "",
    ]
    rows = []
    for cell in report.get("cells") or []:
        metrics = cell.get("metrics") or {}
        rows.append(
            [
                cell.get("group"),
                cell.get("stages"),
                metrics.get("episodes"),
                metrics.get("completed"),
                metrics.get("success_rate"),
                metrics.get("false_completion"),
                metrics.get("tool_calls"),
                metrics.get("vlm_calls"),
                metrics.get("wall_time_s"),
                metrics.get("peak_allocated_mib"),
                metrics.get("peak_reserved_mib"),
                metrics.get("resume_recovered"),
                metrics.get("replayed_steps"),
                metrics.get("clean_uninstall"),
                cell.get("error") or "ok",
            ]
        )
    lines.append(
        _simple_table(
            rows,
            [
                "group",
                "stages",
                "episodes",
                "completed",
                "success_rate",
                "false_completion",
                "tool_calls",
                "vlm_calls",
                "wall_time_s",
                "peak_allocated_mib",
                "peak_reserved_mib",
                "resume_recovered",
                "replayed_steps",
                "clean_uninstall",
                "note",
            ],
        )
    )
    lines.extend(["", "## Episodes", ""])
    episode_rows = []
    for episode in report.get("episodes") or []:
        resume = episode.get("resume") or {}
        episode_rows.append(
            [
                episode.get("group"),
                episode.get("stages"),
                episode.get("seed"),
                episode.get("mission"),
                episode.get("status"),
                episode.get("success"),
                episode.get("false_completion"),
                episode.get("tool_calls"),
                episode.get("interrupted"),
                resume.get("replayed_steps"),
                resume.get("motion_progress_preserved"),
                episode.get("error") or "ok",
            ]
        )
    lines.append(
        _simple_table(
            episode_rows,
            [
                "group",
                "stages",
                "seed",
                "mission",
                "status",
                "success",
                "false_completion",
                "tool_calls",
                "interrupted",
                "replayed_steps",
                "motion_preserved",
                "note",
            ],
        )
    )
    resources = report.get("resources_after_uninstall") or {}
    lines.extend(["", "## Resources after uninstall", ""])
    lines.append(
        _simple_table(
            [
                ["plugin_catalog_count", resources.get("plugin_catalog_count")],
                ["service_count", resources.get("service_count")],
                ["tool_count", resources.get("tool_count")],
                ["prompt_count", resources.get("prompt_count")],
                ["residual_entries", len(resources.get("residual") or [])],
                ["worker_pids", resources.get("worker_pids")],
                ["leftover_paths", len(resources.get("leftover_paths") or [])],
                ["clean", resources.get("clean")],
            ],
            ["metric", "value"],
        )
    )
    if backend.get("error"):
        lines.extend(
            [
                "",
                "## Semantic backend error",
                "",
                f"```\n{backend.get('error')}\n```",
            ]
        )
    return "\n".join(lines) + "\n"


def write_report(report: Mapping, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "report.json"
    markdown_path = out / "report.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def run_matrix(config: MatrixConfig) -> dict:
    """Run every requested (group, stages) cell and write report.json/report.md."""
    started = time.perf_counter()
    backend: dict[str, Any] = {
        "mode": config.semantic,
        "available": config.semantic == "mock",
        "error": "",
        "vlm": "none",
        "plugins": [],
    }
    real_classes: Mapping[str, Any] | None = None
    if config.semantic == "real":
        try:
            real_classes = _real_plugin_classes()
            backend["plugins"] = sorted(real_classes)
            backend["available"] = True
        except ImportError as exc:
            backend["available"] = False
            backend["error"] = f"ImportError: {exc}"
        except Exception as exc:  # noqa: BLE001 - reported, never silently mocked
            backend["available"] = False
            backend["error"] = f"{type(exc).__name__}: {exc}"

    tasks_cache: dict[int, dict[str, list[dict]]] = {}
    cells: list[dict] = []
    for group in config.groups:
        for stages in config.stages:
            if group != "action" and not backend["available"]:
                cells.append(
                    _skipped_cell(
                        group,
                        stages,
                        config,
                        "semantic backend unavailable: "
                        + (backend["error"] or "not installed"),
                    )
                )
                continue
            try:
                if stages not in tasks_cache:
                    tasks_cache[stages] = load_tasks(config.tasks_dir, stages)
                missions = tasks_cache[stages][group]
            except Exception as exc:  # noqa: BLE001 - task errors are cells too
                cells.append(
                    _skipped_cell(
                        group,
                        stages,
                        config,
                        f"task loading failed: {type(exc).__name__}: {exc}",
                    )
                )
                continue
            cells.append(
                run_cell(group, stages, missions, config, real_classes)
            )

    summary = _summary(cells)
    resources = _aggregate_resources(cells)
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "groups": list(config.groups),
            "stages": list(config.stages),
            "seeds": int(config.seeds),
            "world": config.world,
            "ply": str(config.ply),
            "target_pos": [float(value) for value in config.target_pos],
            "start_pos": [float(value) for value in config.start_pos],
            "start_yaw_deg": float(config.start_yaw_deg),
            "interrupt_step": int(config.interrupt_step),
            "max_tool_calls": int(config.max_tool_calls),
            "semantic": config.semantic,
            "resume": bool(config.resume),
            "device": config.device,
            "out_dir": str(config.out_dir),
            "tasks_dir": str(config.tasks_dir),
        },
        "semantic_backend": backend,
        "summary": summary,
        "cells": cells,
        "episodes": [
            episode for cell in cells for episode in cell.get("episodes") or []
        ],
        "metrics": {
            group: {
                str(cell["stages"]): cell["metrics"]
                for cell in cells
                if cell["group"] == group
            }
            for group in GROUPS
            if any(cell["group"] == group for cell in cells)
        },
        "resources_after_uninstall": resources,
        "wall_time_s": round(time.perf_counter() - started, 3),
    }
    write_report(report, config.out_dir)
    return report


def exit_code(report: Mapping) -> int:
    backend = report.get("semantic_backend") or {}
    if backend.get("mode") == "real" and not backend.get("available"):
        return 2
    summary = report.get("summary") or {}
    if summary.get("skipped_cells"):
        return 1
    if summary.get("episodes", 0) == 0:
        return 1
    if summary.get("false_completion"):
        return 1
    if float(summary.get("success_rate", 0.0)) < 1.0:
        return 1
    resources = report.get("resources_after_uninstall") or {}
    if not resources.get("clean"):
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    report = run_matrix(config)
    summary = report["summary"]
    resources = report["resources_after_uninstall"]
    print(
        "mixed mission matrix: "
        f"{summary['episodes']} episodes, "
        f"success_rate={summary['success_rate']}, "
        f"false_completion={summary['false_completion']}, "
        f"clean_uninstall={resources['clean']}"
    )
    backend = report["semantic_backend"]
    if backend.get("mode") == "real":
        print(
            f"semantic backend: available={backend.get('available')} "
            f"{backend.get('error') or ''}".rstrip()
        )
    for cell in report["cells"]:
        metrics = cell["metrics"]
        note = cell.get("error") or "ok"
        print(
            f"  {cell['group']:<8} stages={cell['stages']}: "
            f"{metrics['completed']}/{metrics['episodes']} completed, "
            f"false_completion={metrics['false_completion']}, "
            f"replayed={metrics['replayed_steps']}, "
            f"clean={metrics['clean_uninstall']} ({note})"
        )
    print(f"report: {config.out_dir / 'report.json'}")
    print(f"markdown: {config.out_dir / 'report.md'}")
    return exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
