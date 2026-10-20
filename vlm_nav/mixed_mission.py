"""Mixed missions: motion steps plus semantic/control steps (goal.md M2).

A mixed mission is an ordered chain.  The *schema* is deliberately a plain
list of step dictionaries so a planner (deterministic grammar or a VLM) can
emit it without importing classes:

``move`` / ``turn`` / ``hover`` / ``photo``
    Exactly the M1 task-level motion schema from :mod:`vlm_nav.motion`.
``search`` / ``approach`` / ``confirm`` / ``verify`` / ``revisit`` / ``goto``
    Semantic navigation steps, executed through a
    ``SemanticNavigationService``-like object.
``takeoff`` / ``land`` / ``orbit`` / ``patrol``
    Capability-gated backend operations.  They call the world method of the
    same name when it exists and fail with ``unsupported_capability:<kind>``
    otherwise - the executor never silently reinterprets them.

The important semantics this module pins:

* execution is strictly ordered; a failed/blocked step stops the mission and
  the remaining steps stay ``pending``;
* a semantic step is ``done`` only on its own completion signal
  (``candidate_id`` / ``arrived`` / ``accepted`` / ``verified``), never on
  ``ok=True`` or a free-text progress claim;
* a motion step is one ``MissionSpec`` on a :class:`MotionExecutor`; its
  ``MissionState`` is stored in the mixed step record, so a partially completed
  motion step resumes from its measured distance/angle/time instead of being
  replayed;
* pause is cooperative between steps (cancel is the mid-motion stop), and
  cancel/resume preserve every completed step and the in-flight progress;
* the state is JSON-serializable and can be persisted after every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import uuid
from typing import Any, Callable, Mapping, Sequence

from .motion import (
    DEFAULT_CHUNK_M,
    DEFAULT_CHUNK_TURN_DEG,
    DEFAULT_HOVER_DT_S,
    MISSION_BLOCKED as MOTION_BLOCKED,
    MISSION_CANCELLED as MOTION_CANCELLED,
    MISSION_COMPLETED as MOTION_COMPLETED,
    MissionError,
    MissionSpec,
    MissionState,
    MotionExecutor,
    clock_for_world,
    step_from_dict,
)
from .tools import ToolContext

MIXED_MISSION_SCHEMA = "drone.mixed_mission/1"
MIXED_STATE_SCHEMA = "drone.mixed_mission.state/1"

MOTION_STEP_KINDS = ("move", "turn", "hover", "photo")
SEMANTIC_STEP_KINDS = ("search", "approach", "confirm", "verify", "revisit", "goto")
CAPABILITY_STEP_KINDS = ("takeoff", "land", "orbit", "patrol")
STEP_KINDS = MOTION_STEP_KINDS + SEMANTIC_STEP_KINDS + CAPABILITY_STEP_KINDS

MIXED_STEP_PENDING = "pending"
MIXED_STEP_RUNNING = "running"
MIXED_STEP_DONE = "done"
MIXED_STEP_CANCELLED = "cancelled"
MIXED_STEP_BLOCKED = "blocked"
MIXED_STEP_FAILED = "failed"
MIXED_STEP_STATUSES = (
    MIXED_STEP_PENDING,
    MIXED_STEP_RUNNING,
    MIXED_STEP_DONE,
    MIXED_STEP_CANCELLED,
    MIXED_STEP_BLOCKED,
    MIXED_STEP_FAILED,
)

MIXED_PENDING = "pending"
MIXED_RUNNING = "running"
MIXED_PAUSED = "paused"
MIXED_COMPLETED = "completed"
MIXED_CANCELLED = "cancelled"
MIXED_BLOCKED = "blocked"
MIXED_FAILED = "failed"
MIXED_STATUSES = (
    MIXED_PENDING,
    MIXED_RUNNING,
    MIXED_PAUSED,
    MIXED_COMPLETED,
    MIXED_CANCELLED,
    MIXED_BLOCKED,
    MIXED_FAILED,
)

#: ``ctx.extra`` keys owned by this module.
MISSION_STATE_KEY = "mission.state"
MISSION_EXECUTOR_KEY = "mission.executor"
MISSION_STATE_PATH_KEY = "mission.state_path"
MISSION_ARTIFACTS_KEY = "mission.artifacts_dir"
MISSION_SEMANTIC_KEY = "mission.semantic_service"
MISSION_PROGRESS_KEY = "mission.progress"
SUBGOAL_KEY = "subgoal"

DEFAULT_ARTIFACTS = Path("outputs") / "mixed_missions"

_LAST_TARGETS = ("last", "it", "target", "它", "目标", "该目标", "那个")


class MixedMissionError(ValueError):
    """Invalid mixed mission specification or state."""


# ---------------------------------------------------------------------------
# step validation / schema
# ---------------------------------------------------------------------------


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MixedMissionError(f"{label} must be a number, got {value!r}") from exc
    if not math.isfinite(number):
        raise MixedMissionError(f"{label} must be finite, got {value!r}")
    return number


def _text(value: Any, label: str) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        raise MixedMissionError(f"{label} is required")
    return text


def _positive_int(value: Any, label: str, default: int) -> int:
    if value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise MixedMissionError(f"{label} must be an integer, got {value!r}") from exc
    if number < 1:
        raise MixedMissionError(f"{label} must be >= 1")
    return number


def _position(value: Any, label: str = "position") -> list[float]:
    if isinstance(value, Mapping):
        try:
            value = [value["x"], value["y"], value["z"]]
        except KeyError as exc:
            raise MixedMissionError(
                f"{label} mapping must contain x, y and z"
            ) from exc
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MixedMissionError(f"{label} must be a sequence of three numbers")
    if len(value) != 3:
        raise MixedMissionError(f"{label} must contain exactly three numbers")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def validate_step(data: Mapping) -> dict:
    """Return a validated copy of one step dictionary.

    Unknown kinds and missing required fields raise :class:`MixedMissionError`.
    Motion steps are validated by the M1 schema itself, so the two schemas
    cannot drift apart.
    """
    if isinstance(data, str) or not isinstance(data, Mapping):
        raise MixedMissionError(
            f"mission step must be an object, got {type(data).__name__}"
        )
    step = _json_safe(dict(data))
    kind = str(step.get("kind", "") or "").strip().lower()
    if not kind:
        raise MixedMissionError("mission step is missing its 'kind'")
    step["kind"] = kind

    if kind in MOTION_STEP_KINDS:
        try:
            step_from_dict(step)
        except MissionError as exc:
            raise MixedMissionError(f"{kind} step is invalid: {exc}") from exc
        return step

    if kind == "search":
        step["target"] = _text(step.get("target"), "search target")
        if "relation" in step and step["relation"] is not None:
            step["relation"] = str(step["relation"]).strip()
        if "max_steps" in step:
            step["max_steps"] = _positive_int(step["max_steps"], "search max_steps", 8)
        if "max_turns" in step:
            step["max_turns"] = _positive_int(step["max_turns"], "search max_turns", 1)
        return step

    if kind == "approach":
        step["target"] = str(step.get("target", "last") or "last").strip()
        if "standoff_m" in step:
            value = _finite(step["standoff_m"], "approach standoff_m")
            if value < 0:
                raise MixedMissionError("approach standoff_m must be >= 0")
            step["standoff_m"] = value
        if "max_legs" in step:
            step["max_legs"] = _positive_int(step["max_legs"], "approach max_legs", 8)
        return step

    if kind == "confirm":
        step["target"] = str(step.get("target", "last") or "last").strip()
        if "max_views" in step:
            step["max_views"] = _positive_int(step["max_views"], "confirm max_views", 2)
        return step

    if kind == "verify":
        step["target"] = str(step.get("target", "last") or "last").strip()
        if step.get("success_dist_m") is not None:
            value = _finite(step["success_dist_m"], "verify success_dist_m")
            if value < 0:
                raise MixedMissionError("verify success_dist_m must be >= 0")
            step["success_dist_m"] = value
        return step

    if kind == "revisit":
        if step.get("position") is not None:
            step["position"] = _position(step["position"], "revisit position")
        has_target = any(
            step.get(key) not in (None, "", [], ())
            for key in ("position", "place", "observation_id", "viewpoint_id")
        )
        if not has_target:
            raise MixedMissionError(
                "revisit requires position, place, observation_id or viewpoint_id"
            )
        for key in ("place", "observation_id", "viewpoint_id"):
            if step.get(key) is not None:
                step[key] = str(step[key]).strip()
        if "standoff_m" in step:
            value = _finite(step["standoff_m"], "revisit standoff_m")
            if value < 0:
                raise MixedMissionError("revisit standoff_m must be >= 0")
            step["standoff_m"] = value
        return step

    if kind == "goto":
        if step.get("position") is None:
            raise MixedMissionError("goto requires a 'position'")
        step["position"] = _position(step["position"], "goto position")
        if "standoff_m" in step:
            value = _finite(step["standoff_m"], "goto standoff_m")
            if value < 0:
                raise MixedMissionError("goto standoff_m must be >= 0")
            step["standoff_m"] = value
        return step

    if kind in CAPABILITY_STEP_KINDS:
        return step

    raise MixedMissionError(
        f"unknown step kind {kind!r}; expected one of {', '.join(STEP_KINDS)}"
    )


@dataclass(frozen=True)
class MixedMissionSpec:
    """An ordered mixed mission.  Steps are validated dictionaries."""

    steps: tuple[dict, ...]
    mission_id: str = ""
    instruction: str = ""
    schema: str = MIXED_MISSION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != MIXED_MISSION_SCHEMA:
            raise MixedMissionError(
                f"unknown mixed mission schema {self.schema!r}; "
                f"expected {MIXED_MISSION_SCHEMA!r}"
            )
        steps = tuple(validate_step(step) for step in self.steps)
        if not steps:
            raise MixedMissionError("mixed mission must contain at least one step")
        object.__setattr__(self, "steps", steps)
        mission_id = str(self.mission_id or "").strip()
        object.__setattr__(
            self, "mission_id", mission_id or f"mission_{uuid.uuid4().hex[:8]}"
        )
        object.__setattr__(self, "instruction", str(self.instruction or ""))

    @classmethod
    def from_dict(
        cls, data: "MixedMissionSpec | Mapping | Sequence | str"
    ) -> "MixedMissionSpec":
        if isinstance(data, MixedMissionSpec):
            return data
        if isinstance(data, str):
            return cls.from_json(data)
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes, Mapping)):
            data = {"steps": list(data)}
        if not isinstance(data, Mapping):
            raise MixedMissionError(
                f"mixed mission must be an object, got {type(data).__name__}"
            )
        if "mission" in data and isinstance(data["mission"], Mapping):
            merged = dict(data["mission"])
            for key in ("mission_id", "instruction"):
                if key in data and key not in merged:
                    merged[key] = data[key]
            data = merged
        steps = data.get("steps")
        if isinstance(steps, (str, bytes)) or not isinstance(steps, Sequence):
            raise MixedMissionError("mixed mission requires a 'steps' array")
        return cls(
            steps=tuple(steps),
            mission_id=str(data.get("mission_id", "") or ""),
            instruction=str(data.get("instruction", "") or ""),
            schema=str(data.get("schema", MIXED_MISSION_SCHEMA) or MIXED_MISSION_SCHEMA),
        )

    @classmethod
    def from_json(cls, text: str) -> "MixedMissionSpec":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MixedMissionError(f"mixed mission JSON is invalid: {exc}") from exc
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "mission_id": self.mission_id,
            "instruction": self.instruction,
            "steps": [dict(step) for step in self.steps],
        }


# ---------------------------------------------------------------------------
# execution state
# ---------------------------------------------------------------------------


@dataclass
class MixedStepRecord:
    """One step's serialized parameters plus its measured outcome."""

    kind: str
    params: dict = field(default_factory=dict)
    status: str = MIXED_STEP_PENDING
    attempts: int = 0
    motion_state: dict | None = None
    evidence: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    error: str = ""

    def __post_init__(self) -> None:
        self.kind = str(self.kind or "").strip().lower()
        if self.kind not in STEP_KINDS:
            raise MixedMissionError(f"unknown step kind in state: {self.kind!r}")
        if self.status not in MIXED_STEP_STATUSES:
            raise MixedMissionError(f"unknown step status: {self.status!r}")
        if not isinstance(self.params, Mapping):
            raise MixedMissionError("step params must be an object")
        # Validate eagerly so a resumed state cannot hide a corrupt step, and
        # keep the JSON-safe validated copy so state serialization cannot fail.
        validated = validate_step(self.step)
        self.params = {
            key: value for key, value in validated.items() if key != "kind"
        }
        self.attempts = int(self.attempts or 0)
        self.evidence = dict(self.evidence or {})
        self.result = dict(self.result or {})
        self.error = str(self.error or "")
        if self.motion_state is not None and not isinstance(self.motion_state, Mapping):
            raise MixedMissionError("motion_state must be an object or null")
        if self.motion_state is not None:
            self.motion_state = dict(self.motion_state)

    @property
    def step(self) -> dict:
        return {"kind": self.kind, **dict(self.params)}

    @property
    def artifact(self) -> str:
        value = self.evidence.get("artifact") or self.result.get("artifact")
        return str(value) if value else ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "params": dict(self.params),
            "status": self.status,
            "attempts": self.attempts,
            "motion_state": dict(self.motion_state) if self.motion_state else None,
            "evidence": dict(self.evidence),
            "result": dict(self.result),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "MixedStepRecord":
        if not isinstance(data, Mapping):
            raise MixedMissionError("step record must be an object")
        kind = str(data.get("kind", "") or "")
        params = data.get("params")
        if params is None and isinstance(data.get("step"), Mapping):
            # tolerate a full step dict stored under "step"
            step = dict(data["step"])
            kind = kind or str(step.get("kind", ""))
            params = {key: value for key, value in step.items() if key != "kind"}
        if not isinstance(params, Mapping):
            raise MixedMissionError("step record is missing its params")
        return cls(
            kind=kind,
            params=dict(params),
            status=str(data.get("status", MIXED_STEP_PENDING) or MIXED_STEP_PENDING),
            attempts=int(data.get("attempts", 0) or 0),
            motion_state=data.get("motion_state"),
            evidence=dict(data.get("evidence") or {}),
            result=dict(data.get("result") or {}),
            error=str(data.get("error", "") or ""),
        )


def _step_label(record: MixedStepRecord) -> str:
    """A short, human-readable label used in the runtime ``subgoal`` line."""
    params = record.params
    kind = record.kind
    if kind == "move":
        return f"move {params.get('direction', 'forward')} {float(params.get('meters', 0) or 0):g}m"
    if kind == "turn":
        degrees = params.get("degrees", 0)
        direction = params.get("direction", "")
        if direction:
            return f"turn {direction} {float(degrees or 0):g}deg"
        return f"turn {float(degrees or 0):g}deg"
    if kind == "hover":
        return f"hover {float(params.get('seconds', 0) or 0):g}s"
    if kind == "photo":
        return f"photo {params.get('label', 'photo')}"
    if kind == "search":
        relation = params.get("relation", "")
        target = params.get("target", "")
        return f"search {relation + ' ' if relation else ''}{target}".strip()
    if kind in ("approach", "confirm", "verify"):
        return f"{kind} {params.get('target', 'last')}"
    if kind == "revisit":
        place = params.get("place") or params.get("observation_id") or params.get("viewpoint_id")
        if place:
            return f"revisit {place}"
        return "revisit position"
    if kind == "goto":
        position = params.get("position") or []
        return "goto " + ",".join(f"{float(value):g}" for value in position)
    return kind


def format_subgoal(state: "MixedMissionState") -> str:
    """``mission <id> step <done>/<n>: <label> (<remaining> left)`` (<=160)."""
    total = len(state.steps)
    done = state.completed_steps()
    remaining = max(0, total - done)
    label = "complete"
    for record in state.steps:
        if record.status != MIXED_STEP_DONE:
            label = _step_label(record)
            break
    suffix = f" ({remaining} left)"
    mission_id = str(state.mission_id)
    prefix = f"mission {mission_id} step {done}/{total}: "
    limit = 160
    if len(prefix) + len(suffix) >= limit:
        # A pathologically long mission id must not push the line over the
        # runtime's subgoal budget; keep the step counter and label readable.
        fixed = len(f"mission  step {done}/{total}: ") + len(suffix)
        keep = max(1, limit - fixed - 10)
        mission_id = mission_id[:keep]
        prefix = f"mission {mission_id} step {done}/{total}: "
    room = max(1, limit - len(prefix) - len(suffix))
    if len(label) > room:
        label = label[: room - 3] + "..." if room > 3 else label[:room]
    return (prefix + label + suffix)[:limit]


@dataclass
class MixedMissionState:
    mission_id: str
    steps: list[MixedStepRecord]
    status: str = MIXED_PENDING
    current_step: int = 0
    instruction: str = ""
    schema: str = MIXED_STATE_SCHEMA
    clock_s: float = 0.0
    subgoal: str = ""
    error: str = ""
    cancel_reason: str = ""
    runs: int = 0

    def __post_init__(self) -> None:
        if self.schema != MIXED_STATE_SCHEMA:
            raise MixedMissionError(f"unknown mixed state schema: {self.schema!r}")
        if self.status not in MIXED_STATUSES:
            raise MixedMissionError(f"unknown mixed mission status: {self.status!r}")
        if not self.steps:
            raise MixedMissionError("mixed mission state has no steps")
        if not 0 <= int(self.current_step) <= len(self.steps):
            raise MixedMissionError(
                f"current_step {self.current_step} is outside 0..{len(self.steps)}"
            )
        self.current_step = int(self.current_step)
        self.clock_s = float(self.clock_s or 0.0)
        self.instruction = str(self.instruction or "")
        self.subgoal = str(self.subgoal or "")
        self.error = str(self.error or "")
        self.cancel_reason = str(self.cancel_reason or "")

    @classmethod
    def new(cls, spec: MixedMissionSpec, clock_s: float = 0.0) -> "MixedMissionState":
        return cls(
            mission_id=spec.mission_id,
            steps=[
                MixedStepRecord(
                    kind=str(step["kind"]),
                    params={key: value for key, value in step.items() if key != "kind"},
                )
                for step in spec.steps
            ],
            status=MIXED_PENDING,
            current_step=0,
            instruction=spec.instruction,
            clock_s=float(clock_s),
        )

    @property
    def spec(self) -> MixedMissionSpec:
        return MixedMissionSpec(
            steps=tuple(record.step for record in self.steps),
            mission_id=self.mission_id,
            instruction=self.instruction,
        )

    @property
    def done(self) -> bool:
        return self.status == MIXED_COMPLETED

    def completed_steps(self) -> int:
        return sum(1 for record in self.steps if record.status == MIXED_STEP_DONE)

    def summary(self) -> dict:
        total = len(self.steps)
        done = self.completed_steps()
        return {
            "schema": self.schema,
            "mission_id": self.mission_id,
            "status": self.status,
            "current_step": self.current_step,
            "steps_total": total,
            "steps_done": done,
            "steps_remaining": max(0, total - done),
            "clock_s": self.clock_s,
            "instruction": self.instruction,
            "subgoal": self.subgoal,
            "error": self.error,
            "cancel_reason": self.cancel_reason,
            "steps": [
                {
                    "index": index,
                    "kind": record.kind,
                    "status": record.status,
                    "attempts": record.attempts,
                    "error": record.error,
                }
                for index, record in enumerate(self.steps)
            ],
        }

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "mission_id": self.mission_id,
            "status": self.status,
            "current_step": self.current_step,
            "instruction": self.instruction,
            "clock_s": self.clock_s,
            "subgoal": self.subgoal,
            "error": self.error,
            "cancel_reason": self.cancel_reason,
            "runs": self.runs,
            "steps": [record.to_dict() for record in self.steps],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "MixedMissionState":
        if not isinstance(data, Mapping):
            raise MixedMissionError("mixed mission state must be an object")
        steps = data.get("steps")
        if isinstance(steps, (str, bytes)) or not isinstance(steps, Sequence):
            raise MixedMissionError("mixed mission state requires a 'steps' array")
        return cls(
            mission_id=str(data.get("mission_id", "") or ""),
            steps=[MixedStepRecord.from_dict(item) for item in steps],
            status=str(data.get("status", MIXED_PENDING) or MIXED_PENDING),
            current_step=int(data.get("current_step", 0) or 0),
            instruction=str(data.get("instruction", "") or ""),
            schema=str(data.get("schema", MIXED_STATE_SCHEMA) or MIXED_STATE_SCHEMA),
            clock_s=float(data.get("clock_s", 0.0) or 0.0),
            subgoal=str(data.get("subgoal", "") or ""),
            error=str(data.get("error", "") or ""),
            cancel_reason=str(data.get("cancel_reason", "") or ""),
            runs=int(data.get("runs", 0) or 0),
        )

    @classmethod
    def from_json(cls, text: str) -> "MixedMissionState":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MixedMissionError(
                f"mixed mission state JSON is invalid: {exc}"
            ) from exc
        return cls.from_dict(data)


def save_state(state: MixedMissionState, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(tmp, path)
    return path


def load_state(path: str | Path) -> MixedMissionState:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MixedMissionError(f"cannot read mixed mission state {path}: {exc}") from exc
    return MixedMissionState.from_json(text)


# ---------------------------------------------------------------------------
# executor
# ---------------------------------------------------------------------------


def _blocked_result(data: Mapping) -> bool:
    status = str(data.get("status", "") or "").strip().lower()
    return status in ("blocked", "unreachable", "not_found", "missing")


def _needs_confirmation(data: Mapping) -> bool:
    """True when an arrival still needs an accepted identity confirmation."""
    details = data.get("data")
    if not isinstance(details, Mapping):
        details = {}
    return bool(
        data.get("confirmation_required")
        or data.get("confirmation_acceptance_required")
        or details.get("confirmation_required")
        or details.get("confirmation_acceptance_required")
    )


def _failure_text(kind: str, data: Mapping) -> str:
    error = str(data.get("error", "") or "").strip()
    reason = str(data.get("reason", "") or "").strip()
    status = str(data.get("status", "") or "").strip()
    if error or reason:
        detail = error or reason
    elif kind == "search":
        detail = "no candidate_id"
    elif kind in ("approach", "revisit", "goto"):
        detail = "arrived=false"
    elif kind == "confirm":
        detail = f"accepted=false (status={status or 'unknown'})"
    elif kind == "verify":
        detail = f"verified=false (status={status or 'unknown'})"
    else:
        detail = status or "no completion signal"
    return f"{kind} not completed: {detail}"


def _semantic_evidence(kind: str, data: Mapping) -> dict:
    evidence = {
        "tool": str(data.get("tool", "") or ""),
        "status": str(data.get("status", "") or ""),
        "candidate_id": str(data.get("candidate_id", "") or ""),
        "task_id": str(data.get("task_id", "") or ""),
        "calls": int(data.get("calls", 0) or 0),
    }
    for key in ("arrived", "accepted", "verified", "confirmation_accepted"):
        if key in data:
            evidence[key] = bool(data.get(key))
    confirmation = data.get("confirmation")
    if isinstance(confirmation, Mapping):
        evidence["confirmation_status"] = str(
            confirmation.get("status", "") or ""
        )
        evidence["confirmation_candidate_id"] = str(
            confirmation.get("candidate_id", "") or ""
        )
    return evidence


class MixedMissionExecutor:
    """Execute a :class:`MixedMissionSpec` step by step against a world."""

    def __init__(
        self,
        world: Any,
        *,
        registry: Any = None,
        ctx: ToolContext | None = None,
        motion_executor: MotionExecutor | None = None,
        semantic_service: Any = None,
        clock: Any = None,
        artifacts_dir: str | Path | None = None,
        max_semantic_retries: int = 2,
        on_progress: Callable[[dict], Any] | None = None,
        state_path: str | Path | None = None,
        chunk_m: float = DEFAULT_CHUNK_M,
        chunk_turn_deg: float = DEFAULT_CHUNK_TURN_DEG,
        hover_dt_s: float = DEFAULT_HOVER_DT_S,
    ) -> None:
        if world is None:
            raise MixedMissionError("mixed mission requires a world backend")
        self.world = world
        self.ctx = ctx if ctx is not None else ToolContext(world=world)
        self.registry = registry
        self.artifacts_dir = Path(artifacts_dir or DEFAULT_ARTIFACTS)
        self.state_path = Path(state_path) if state_path else None
        self.max_semantic_retries = int(max_semantic_retries)
        self.on_progress = on_progress
        self.clock = clock if clock is not None else clock_for_world(world)
        self._semantic = semantic_service
        if self._semantic is None and isinstance(self.ctx.extra, dict):
            self._semantic = self.ctx.extra.get(MISSION_SEMANTIC_KEY)
        self._semantic_error = ""
        self._last_candidate = ""
        self._last_target_phrase = ""

        if motion_executor is None:
            # Reuse the motion plugin's executor (same world) so chunk sizes and
            # the simulation clock stay consistent across motion and mixed tools.
            existing = (
                self.ctx.extra.get("motion.executor")
                if isinstance(self.ctx.extra, dict)
                else None
            )
            if existing is not None and getattr(existing, "world", None) is world:
                motion_executor = existing
                if clock is not None:
                    motion_executor.clock = clock
                else:
                    # One simulation clock for hover/turn/move and the mixed
                    # state, even when the motion plugin created it first.
                    self.clock = motion_executor.clock
            else:
                motion_executor = MotionExecutor(
                    world,
                    clock=self.clock,
                    artifacts_dir=self.artifacts_dir,
                    chunk_m=chunk_m,
                    chunk_turn_deg=chunk_turn_deg,
                    hover_dt_s=hover_dt_s,
                )
        elif clock is not None:
            motion_executor.clock = clock
        else:
            self.clock = motion_executor.clock
        self.motion = motion_executor
        self.motion.artifacts_dir = self.artifacts_dir
        # The mixed state is the single source of truth for resume.
        self.motion.state_path = None

        self.state: MixedMissionState | None = None
        self.persist_error = ""
        self._running = False
        self._pause_requested = False
        self._pause_reason = ""
        self._cancel_requested = False
        self._cancel_reason = ""

    # -- public API ------------------------------------------------------- #

    @property
    def running(self) -> bool:
        return self._running

    @property
    def state(self) -> MixedMissionState | None:  # noqa: F811 - documented property
        return self._state

    @state.setter
    def state(self, value: MixedMissionState | None) -> None:
        self._state = value

    def run(
        self,
        mission: MixedMissionSpec | Mapping | Sequence,
        *,
        mission_id: str = "",
        instruction: str = "",
    ) -> MixedMissionState:
        if self._running:
            raise MixedMissionError("mixed mission executor is already running")
        spec = (
            mission
            if isinstance(mission, MixedMissionSpec)
            else MixedMissionSpec.from_dict(mission)
        )
        if mission_id or instruction:
            spec = MixedMissionSpec(
                steps=spec.steps,
                mission_id=mission_id or spec.mission_id,
                instruction=instruction or spec.instruction,
            )
        state = MixedMissionState.new(spec, clock_s=self.clock.now_s())
        self.state = state
        self._reset_requests()
        return self._run_state(state)

    def resume(self, state: MixedMissionState | None = None) -> MixedMissionState:
        if self._running:
            raise MixedMissionError("mixed mission executor is already running")
        if isinstance(state, Mapping):
            state = MixedMissionState.from_dict(state)
        state = state or self.state
        if state is None:
            raise MixedMissionError("no mixed mission state to resume")
        self.state = state
        if state.status == MIXED_COMPLETED:
            # Idempotent: a completed mission is never replayed.
            self._sync_context(state)
            return state
        first_open = next(
            (
                index
                for index, record in enumerate(state.steps)
                if record.status != MIXED_STEP_DONE
            ),
            len(state.steps),
        )
        state.current_step = min(state.current_step, first_open)
        self._reset_requests()
        state.cancel_reason = ""
        state.error = ""
        for index in range(state.current_step, len(state.steps)):
            record = state.steps[index]
            if record.status != MIXED_STEP_DONE:
                # Reopen the interrupted step but keep motion_state/attempts.
                record.status = MIXED_STEP_PENDING
                record.error = ""
        return self._run_state(state)

    def pause(self, reason: str = "") -> MixedMissionState:
        state = self.state
        if state is None:
            raise MixedMissionError("no mixed mission state to pause")
        self._pause_requested = True
        self._pause_reason = str(reason or "")
        if not self._running and state.status in (MIXED_PENDING, MIXED_RUNNING):
            state.status = MIXED_PAUSED
            state.error = self._pause_reason
            self._sync_context(state)
            self._persist()
        return state

    def cancel(self, reason: str = "cancelled") -> MixedMissionState:
        state = self.state
        if state is None:
            raise MixedMissionError("no mixed mission state to cancel")
        self._cancel_requested = True
        self._cancel_reason = str(reason or "cancelled")
        # A mid-motion cancel stops the motion executor at the next chunk; the
        # mixed executor then records the measured partial progress.
        if self.motion is not None and getattr(self.motion, "running", False):
            try:
                self.motion.cancel(self._cancel_reason)
            except Exception:  # noqa: BLE001 - cancellation must not raise
                pass
        if not self._running and state.status in (
            MIXED_PENDING,
            MIXED_RUNNING,
            MIXED_PAUSED,
        ):
            state.status = MIXED_CANCELLED
            state.cancel_reason = self._cancel_reason
            if state.current_step < len(state.steps):
                record = state.steps[state.current_step]
                if record.status in (MIXED_STEP_PENDING, MIXED_STEP_RUNNING):
                    record.status = MIXED_STEP_CANCELLED
                    record.error = self._cancel_reason
            self._sync_context(state)
            self._persist()
        return state

    # -- state machine ---------------------------------------------------- #

    def _reset_requests(self) -> None:
        self._pause_requested = False
        self._pause_reason = ""
        self._cancel_requested = False
        self._cancel_reason = ""

    def _run_state(self, state: MixedMissionState) -> MixedMissionState:
        if self._running:
            raise MixedMissionError("mixed mission executor is already running")
        self.state = state
        self._running = True
        state.runs = int(state.runs or 0) + 1
        state.status = MIXED_RUNNING
        state.error = ""
        state.clock_s = self.clock.now_s()
        self._sync_context(state)
        self._publish("mission.started")
        try:
            while state.current_step < len(state.steps):
                if self._pause_requested:
                    state.status = MIXED_PAUSED
                    state.error = self._pause_reason
                    break
                if self._cancel_requested:
                    state.status = MIXED_CANCELLED
                    state.cancel_reason = self._cancel_reason or "cancelled"
                    break
                record = state.steps[state.current_step]
                if record.status == MIXED_STEP_DONE:
                    index = state.current_step
                    state.current_step += 1
                    self._after_step(state, record, index)
                    continue
                index = state.current_step
                self._execute_step(record, state)
                if record.status == MIXED_STEP_DONE:
                    state.current_step += 1
                    self._after_step(state, record, index)
                    continue
                if record.status == MIXED_STEP_CANCELLED:
                    state.status = MIXED_CANCELLED
                    state.cancel_reason = (
                        self._cancel_reason or record.error or "cancelled"
                    )
                elif record.status == MIXED_STEP_BLOCKED:
                    state.status = MIXED_BLOCKED
                    state.error = record.error
                else:
                    state.status = MIXED_FAILED
                    state.error = record.error or (
                        f"step {state.current_step} ended as {record.status}"
                    )
                state.clock_s = self.clock.now_s()
                self._sync_context(state)
                self._persist()
                self._publish(
                    "step.finished",
                    {
                        "step": {
                            "index": index,
                            "kind": record.kind,
                            "status": record.status,
                            "attempts": record.attempts,
                            "error": record.error,
                        }
                    },
                )
                break
            else:
                state.status = MIXED_COMPLETED
                state.current_step = len(state.steps)
                state.error = ""
                state.clock_s = self.clock.now_s()
                self._sync_context(state)
                self._persist()
            state.clock_s = self.clock.now_s()
            self._sync_context(state)
            return state
        finally:
            self._running = False
            self._persist()
            self._publish("mission.finished")

    def _execute_step(self, record: MixedStepRecord, state: MixedMissionState) -> None:
        record.status = MIXED_STEP_RUNNING
        record.attempts += 1
        record.error = ""
        try:
            if record.kind in MOTION_STEP_KINDS:
                self._run_motion_step(record, state)
            elif record.kind in SEMANTIC_STEP_KINDS:
                self._run_semantic_step(record, state)
            elif record.kind in CAPABILITY_STEP_KINDS:
                self._run_capability_step(record)
        except Exception as exc:  # noqa: BLE001 - a step failure is state
            record.status = MIXED_STEP_FAILED
            record.error = f"{type(exc).__name__}: {exc}"

    # -- motion steps ----------------------------------------------------- #

    def _run_motion_step(
        self, record: MixedStepRecord, state: MixedMissionState
    ) -> None:
        step = step_from_dict(record.step)
        spec = MissionSpec(
            steps=(step,),
            mission_id=f"{state.mission_id}.s{state.current_step}",
            instruction=_step_label(record),
        )
        self.motion.artifacts_dir = self.artifacts_dir
        self.motion.state_path = None

        prior: MissionState | None = None
        if record.motion_state:
            try:
                prior = MissionState.from_dict(record.motion_state)
            except MissionError:
                prior = None
            if prior is not None and prior.mission_id != spec.mission_id:
                # A state for another mission must not make this step look done.
                prior = None
        if prior is not None and prior.status == MOTION_COMPLETED:
            motion_state = prior
        elif prior is not None:
            self.motion.state = prior
            motion_state = self.motion.resume(prior)
        else:
            motion_state = self.motion.run(spec)

        record.motion_state = motion_state.to_dict()
        record.result = {
            "motion_status": motion_state.status,
            "steps_done": motion_state.completed_steps(),
            "steps_total": len(motion_state.steps),
            "clock_s": motion_state.clock_s,
        }
        artifacts = [
            item.artifact for item in motion_state.steps if item.artifact
        ]
        if artifacts:
            record.evidence = {
                "artifact": artifacts[-1],
                "artifacts": artifacts,
                "motion_status": motion_state.status,
            }
        else:
            record.evidence = {"motion_status": motion_state.status}

        if motion_state.status == MOTION_COMPLETED:
            record.status = MIXED_STEP_DONE
        elif motion_state.status == MOTION_CANCELLED:
            record.status = MIXED_STEP_CANCELLED
            record.error = motion_state.cancel_reason or "motion cancelled"
        elif motion_state.status == MOTION_BLOCKED:
            record.status = MIXED_STEP_BLOCKED
            record.error = motion_state.error or "motion blocked"
        else:
            record.status = MIXED_STEP_FAILED
            record.error = motion_state.error or (
                f"motion ended as {motion_state.status}"
            )

    # -- semantic steps --------------------------------------------------- #

    def _semantic_service(self) -> Any:
        if self._semantic is not None:
            return self._semantic
        if self.registry is None:
            self._semantic_error = "no tool registry is configured"
            return None
        try:
            from .semantic import SemanticNavigationService
        except Exception as exc:  # noqa: BLE001 - workstream C is optional
            self._semantic_error = (
                f"semantic module unavailable: {type(exc).__name__}: {exc}"
            )
            return None
        try:
            self._semantic = SemanticNavigationService(
                self.registry, max_retries=self.max_semantic_retries
            )
        except Exception as exc:  # noqa: BLE001
            self._semantic_error = (
                f"semantic service could not start: {type(exc).__name__}: {exc}"
            )
            return None
        return self._semantic

    def _resolve_candidate(
        self, params: Mapping, state: MixedMissionState
    ) -> str | None:
        target = str(params.get("target", "last") or "last").strip()
        if target.lower() not in _LAST_TARGETS and target not in _LAST_TARGETS:
            return target
        if self._last_candidate:
            return self._last_candidate
        for record in reversed(state.steps):
            candidate = (
                record.evidence.get("candidate_id")
                or record.result.get("candidate_id")
                or ""
            )
            if candidate:
                self._last_candidate = str(candidate)
                return self._last_candidate
        return None

    def _run_semantic_step(
        self, record: MixedStepRecord, state: MixedMissionState
    ) -> None:
        service = self._semantic_service()
        if service is None:
            record.status = MIXED_STEP_FAILED
            record.error = (
                "semantic_service_unavailable: "
                + (self._semantic_error or "no semantic navigation service")
            )
            return
        params = record.params
        kind = record.kind
        instruction = params.get("instruction") or None
        data: dict
        try:
            if kind == "search":
                # The step's target/relation is the search phrase; the whole
                # mission sentence must not override it (a multi-stage mission
                # would otherwise search for every clause at once).
                self._last_target_phrase = str(params["target"])
                data = service.search(
                    self.ctx,
                    str(params["target"]),
                    relation=str(params.get("relation", "") or ""),
                    instruction=instruction,
                    max_steps=int(params.get("max_steps", 8) or 8),
                    max_turns=int(params.get("max_turns", 1) or 1),
                )
            elif kind == "approach":
                candidate_id = self._resolve_candidate(params, state)
                data = service.approach(
                    self.ctx,
                    candidate_id,
                    standoff_m=float(params.get("standoff_m", 1.2)),
                    max_legs=int(params.get("max_legs", 8) or 8),
                )
                # goal.md M2: "approach the chair" is complete only when the
                # correct instance is confirmed *and* the distance condition
                # holds. approach_candidate reports confirmation_required on a
                # candidate arrival; the mission step must not ignore it.
                if data.get("arrived") is True and _needs_confirmation(data):
                    confirmation = service.confirm(
                        self.ctx,
                        data.get("candidate_id") or candidate_id,
                        instruction=(
                            instruction
                            or self._last_target_phrase
                            or (state.instruction or None)
                        ),
                        max_views=max(1, int(params.get("confirm_views", 2) or 2)),
                    )
                    data = dict(data)
                    data["confirmation"] = confirmation
                    accepted = (
                        confirmation.get("ok") is True
                        and confirmation.get("accepted") is True
                    )
                    if not accepted:
                        data["ok"] = False
                        data["confirmation_accepted"] = False
                        data["error"] = (
                            "confirmation_required: approach arrived but the "
                            "instance was not accepted ("
                            f"{confirmation.get('status') or confirmation.get('error') or 'unknown'}"
                            ")"
                        )
                    else:
                        data["confirmation_accepted"] = True
            elif kind == "confirm":
                data = service.confirm(
                    self.ctx,
                    self._resolve_candidate(params, state),
                    instruction=(
                        instruction
                        or self._last_target_phrase
                        or (state.instruction or None)
                    ),
                    relation=params.get("relation"),
                    max_views=int(params.get("max_views", 2) or 2),
                )
            elif kind == "verify":
                distance = params.get("success_dist_m")
                data = service.verify(
                    self.ctx,
                    self._resolve_candidate(params, state),
                    success_dist_m=None if distance is None else float(distance),
                )
            elif kind == "revisit":
                placement = self._resolve_revisit(params)
                if placement is None:
                    record.status = MIXED_STEP_BLOCKED
                    record.error = (
                        "revisit place "
                        f"{params.get('place', '')!r} is not resolvable from "
                        "ctx.extra['places'] or memory"
                    )
                    return
                data = service.revisit(
                    self.ctx,
                    standoff_m=float(params.get("standoff_m", 0.6)),
                    **placement,
                )
            elif kind == "goto":
                data = service.goto(
                    self.ctx,
                    list(params["position"]),
                    standoff_m=float(params.get("standoff_m", 0.6)),
                )
            else:  # pragma: no cover - validated earlier
                record.status = MIXED_STEP_FAILED
                record.error = f"unknown semantic step: {kind}"
                return
        except Exception as exc:  # noqa: BLE001 - report the tool failure
            record.status = MIXED_STEP_FAILED
            record.error = f"{type(exc).__name__}: {exc}"
            return
        if not isinstance(data, Mapping):
            record.status = MIXED_STEP_FAILED
            record.error = (
                f"semantic service returned {type(data).__name__}, expected an object"
            )
            return

        data = dict(data)
        record.result = _json_safe(data)
        record.evidence = _semantic_evidence(kind, data)
        candidate = str(data.get("candidate_id", "") or "")
        if candidate:
            self._last_candidate = candidate

        done = False
        if kind == "search":
            done = bool(candidate)
        elif kind == "approach":
            done = (
                data.get("arrived") is True
                and data.get("confirmation_accepted", True) is not False
            )
        elif kind == "confirm":
            done = data.get("accepted") is True
        elif kind == "verify":
            done = data.get("verified") is True
        elif kind in ("revisit", "goto"):
            done = data.get("arrived") is True

        if done:
            record.status = MIXED_STEP_DONE
            return
        record.status = MIXED_STEP_BLOCKED if _blocked_result(data) else MIXED_STEP_FAILED
        record.error = _failure_text(kind, data)

    def _resolve_revisit(self, params: Mapping) -> dict | None:
        if params.get("position") is not None:
            return {"position": list(params["position"])}
        observation_id = params.get("observation_id")
        viewpoint_id = params.get("viewpoint_id")
        if observation_id or viewpoint_id:
            placement: dict[str, Any] = {}
            if observation_id:
                placement["observation_id"] = str(observation_id)
            if viewpoint_id:
                placement["viewpoint_id"] = str(viewpoint_id)
            return placement
        place = params.get("place")
        if not place:
            return None
        return self._lookup_place(str(place))

    def _lookup_place(self, name: str) -> dict | None:
        extra = self.ctx.extra if isinstance(self.ctx.extra, dict) else {}
        places = extra.get("places")
        if isinstance(places, Mapping):
            value = places.get(name)
            if value is None:
                lowered = name.lower()
                for key, item in places.items():
                    if str(key).lower() == lowered:
                        value = item
                        break
            resolved = _placement_from_value(value)
            if resolved is not None:
                return resolved
        memory = getattr(self.ctx, "memory", None)
        for attr in (
            "place_position",
            "resolve_place",
            "get_place",
            "find_place",
            "lookup_place",
        ):
            method = getattr(memory, attr, None)
            if not callable(method):
                continue
            try:
                value = method(name)
            except Exception:  # noqa: BLE001 - a broken lookup is not a crash
                continue
            resolved = _placement_from_value(value)
            if resolved is not None:
                return resolved
        return None

    # -- capability-gated steps ------------------------------------------- #

    def _run_capability_step(self, record: MixedStepRecord) -> None:
        kind = record.kind
        method = getattr(self.world, kind, None)
        if not callable(method):
            record.status = MIXED_STEP_FAILED
            record.error = f"unsupported_capability:{kind}"
            return
        try:
            outcome = method(**record.params) if record.params else method()
        except Exception as exc:  # noqa: BLE001
            record.status = MIXED_STEP_FAILED
            record.error = f"{type(exc).__name__}: {exc}"
            return
        record.result = {"capability": kind, "outcome": _json_safe(outcome)}
        record.evidence = {"capability": kind}
        if outcome is False:
            record.status = MIXED_STEP_FAILED
            record.error = f"unsupported_capability:{kind} (backend returned False)"
            return
        if isinstance(outcome, Mapping):
            status = str(outcome.get("status", "") or "").lower()
            if outcome.get("ok") is False or status in ("failed", "blocked"):
                record.status = (
                    MIXED_STEP_BLOCKED if status == "blocked" else MIXED_STEP_FAILED
                )
                record.error = (
                    str(outcome.get("error", "") or "")
                    or f"capability {kind} reported {status or 'failure'}"
                )
                return
        record.status = MIXED_STEP_DONE

    # -- context / persistence / progress --------------------------------- #

    def _sync_context(self, state: MixedMissionState) -> None:
        state.subgoal = format_subgoal(state)
        if isinstance(self.ctx.extra, dict):
            self.ctx.extra[SUBGOAL_KEY] = state.subgoal
            self.ctx.extra[MISSION_PROGRESS_KEY] = state.summary()
            self.ctx.extra[MISSION_STATE_KEY] = state

    def _after_step(
        self,
        state: MixedMissionState,
        record: MixedStepRecord | None = None,
        index: int = -1,
    ) -> None:
        state.clock_s = self.clock.now_s()
        self._sync_context(state)
        self._persist()
        payload: dict[str, Any] = {"event": "step.finished"}
        if record is not None:
            payload["step"] = {
                "index": index,
                "kind": record.kind,
                "status": record.status,
                "attempts": record.attempts,
                "error": record.error,
            }
        self._publish("step.finished", payload)

    def _publish(self, event: str, payload: Mapping | None = None) -> None:
        if self.on_progress is None or self.state is None:
            return
        data = {
            "event": event,
            "mission_id": self.state.mission_id,
            "mission_status": self.state.status,
            "current_step": self.state.current_step,
            "clock_s": self.clock.now_s(),
            "state": self.state.summary(),
        }
        if payload:
            data.update(payload)
        try:
            self.on_progress(data)
        except Exception:  # noqa: BLE001 - observers must not break execution
            pass

    def _persist(self) -> None:
        if self.state is None or self.state_path is None:
            return
        try:
            save_state(self.state, self.state_path)
            self.persist_error = ""
        except OSError as exc:
            self.persist_error = f"{type(exc).__name__}: {exc}"


def _placement_from_value(value: Any) -> dict | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if value.get("position") is not None:
            position = value.get("position")
            if isinstance(position, Mapping):
                position = [position.get("x"), position.get("y"), position.get("z")]
            if isinstance(position, Sequence) and not isinstance(position, (str, bytes)):
                if len(position) == 3 and all(
                    isinstance(item, (int, float)) for item in position
                ):
                    return {"position": [float(item) for item in position]}
        if value.get("observation_id") or value.get("viewpoint_id"):
            placement: dict[str, Any] = {}
            if value.get("observation_id"):
                placement["observation_id"] = str(value["observation_id"])
            if value.get("viewpoint_id"):
                placement["viewpoint_id"] = str(value["viewpoint_id"])
            return placement
        if all(key in value for key in ("x", "y", "z")):
            return {
                "position": [float(value["x"]), float(value["y"]), float(value["z"])]
            }
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 3 and all(isinstance(item, (int, float)) for item in value):
            return {"position": [float(item) for item in value]}
        return None
    text = str(value).strip()
    if text.startswith("obs_"):
        return {"observation_id": text}
    if text.startswith("vp_"):
        return {"viewpoint_id": text}
    return None
