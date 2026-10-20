"""Task-level motion schema and execution (goal.md M1).

The existing ``actions`` module answers "what is one primitive action worth" and
clamps it to a per-decision 2 m / 45 deg envelope. A user instruction is not a
per-decision action: "forward 5 metres, hover 3 seconds, turn, keep moving" must
be executed as a *mission* whose steps have completion conditions tied to pose
and simulation time.

This module provides:

* a serializable mission schema (:class:`MissionSpec` and the four step types
  ``move`` / ``turn`` / ``hover`` / ``photo``);
* a state machine (:class:`MissionState`, :class:`StepRecord`) that can be
  written to disk and resumed without replaying finished steps;
* :class:`MotionExecutor`, which splits a long move into chunks the world can
  execute, but keeps the *target position fixed in the frame of the step's
  starting heading* and tracks the remaining distance - so a 5 m request can
  never be truncated to 2 m and reported as done;
* a simulation-time clock. Hover advances that clock; a wall-clock sleep is
  never evidence that a timed hover happened.

Deliberate semantics (goal.md section 1):

``forward``
    The direction is the body heading at the *start* of that move step. The
    world target is computed once and stored; later chunks fly to that target
    even if the body is rotated by something else meanwhile.
``meters``
    Target displacement, not accumulated path length.
``hover``
    ``seconds`` of simulation/flight-controller time, never model-call latency.
``cancel``
    The current step keeps its measured progress and is marked ``cancelled``;
    completed steps stay done.
``resume``
    Recomputes the remaining motion from the stored target and the current
    pose. It does not re-run completed steps, and a photo whose artifact already
    exists is not retaken.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import uuid
from typing import Any, Callable, ClassVar, Mapping, Sequence

import numpy as np
from PIL import Image

from .actions import NavAction, body_delta_to_world

MISSION_SCHEMA = "drone.motion.mission/1"
STATE_SCHEMA = "drone.motion.state/1"

STEP_MOVE = "move"
STEP_TURN = "turn"
STEP_HOVER = "hover"
STEP_PHOTO = "photo"
STEP_KINDS = (STEP_MOVE, STEP_TURN, STEP_HOVER, STEP_PHOTO)

STEP_PENDING = "pending"
STEP_RUNNING = "running"
STEP_DONE = "done"
STEP_CANCELLED = "cancelled"
STEP_BLOCKED = "blocked"
STEP_FAILED = "failed"
STEP_STATUSES = (
    STEP_PENDING, STEP_RUNNING, STEP_DONE, STEP_CANCELLED, STEP_BLOCKED,
    STEP_FAILED,
)

MISSION_PENDING = "pending"
MISSION_RUNNING = "running"
MISSION_COMPLETED = "completed"
MISSION_CANCELLED = "cancelled"
MISSION_BLOCKED = "blocked"
MISSION_FAILED = "failed"
MISSION_STATUSES = (
    MISSION_PENDING, MISSION_RUNNING, MISSION_COMPLETED, MISSION_CANCELLED,
    MISSION_BLOCKED, MISSION_FAILED,
)

DEFAULT_CHUNK_M = 2.0
DEFAULT_CHUNK_TURN_DEG = 45.0
DEFAULT_HOVER_DT_S = 0.1
DEFAULT_POSITION_TOL_M = 0.05
DEFAULT_ANGLE_TOL_DEG = 0.5
MAX_CHUNKS = 10000

#: Body-frame unit vectors: (forward, right, up). ``actions`` maps ``right`` to
#: body ``-y`` (FLU), so no sign conversion happens here.
_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "forward": (1.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0),
    "left": (0.0, -1.0, 0.0),
    "right": (0.0, 1.0, 0.0),
    "up": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
}


class MissionError(ValueError):
    """Invalid mission specification or state."""


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MissionError(f"{label} must be a number, got {value!r}") from exc
    if not math.isfinite(number):
        raise MissionError(f"{label} must be finite, got {value!r}")
    return number


def _wrap_deg(degrees: float) -> float:
    return (float(degrees) + 180.0) % 360.0 - 180.0


def _wrap_rad(radians: float) -> float:
    return (float(radians) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class MoveStep:
    direction: str = "forward"
    meters: float = 0.0
    kind: ClassVar[str] = STEP_MOVE

    def __post_init__(self) -> None:
        direction = str(self.direction or "").strip().lower()
        if direction not in _DIRECTIONS:
            raise MissionError(
                f"move direction must be one of {sorted(_DIRECTIONS)}, got {self.direction!r}"
            )
        object.__setattr__(self, "direction", direction)
        meters = _finite(self.meters, "move meters")
        if meters <= 0:
            raise MissionError("move meters must be > 0")
        object.__setattr__(self, "meters", meters)

    def to_dict(self) -> dict:
        return {"kind": STEP_MOVE, "direction": self.direction, "meters": self.meters}


@dataclass(frozen=True)
class TurnStep:
    """A relative turn. Positive ``degrees`` is left/CCW in FLU yaw."""

    degrees: float = 0.0
    kind: ClassVar[str] = STEP_TURN

    def __post_init__(self) -> None:
        degrees = _finite(self.degrees, "turn degrees")
        if abs(degrees) < 1e-9:
            raise MissionError("turn degrees must be non-zero")
        if abs(degrees) > 3600.0:
            raise MissionError("turn degrees must be within 3600")
        object.__setattr__(self, "degrees", degrees)

    @classmethod
    def from_dict(cls, data: Mapping) -> "TurnStep":
        degrees = _finite(data.get("degrees", 0.0), "turn degrees")
        direction = str(data.get("direction", "") or "").strip().lower()
        if direction in ("left", "ccw", "counterclockwise"):
            degrees = abs(degrees)
        elif direction in ("right", "cw", "clockwise"):
            degrees = -abs(degrees)
        elif direction:
            raise MissionError(f"unknown turn direction: {direction!r}")
        return cls(degrees=degrees)

    def to_dict(self) -> dict:
        return {"kind": STEP_TURN, "degrees": self.degrees}


@dataclass(frozen=True)
class HoverStep:
    seconds: float = 0.0
    kind: ClassVar[str] = STEP_HOVER

    def __post_init__(self) -> None:
        seconds = _finite(self.seconds, "hover seconds")
        if seconds <= 0:
            raise MissionError("hover seconds must be > 0")
        object.__setattr__(self, "seconds", seconds)

    def to_dict(self) -> dict:
        return {"kind": STEP_HOVER, "seconds": self.seconds}


_LABEL_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class PhotoStep:
    label: str = "photo"
    kind: ClassVar[str] = STEP_PHOTO

    def __post_init__(self) -> None:
        label = str(self.label or "photo").strip() or "photo"
        object.__setattr__(self, "label", label[:80])

    @property
    def safe_label(self) -> str:
        safe = _LABEL_RE.sub("_", self.label).strip("._")
        return safe or "photo"

    def to_dict(self) -> dict:
        return {"kind": STEP_PHOTO, "label": self.label}


Step = MoveStep | TurnStep | HoverStep | PhotoStep


def step_from_dict(data: Mapping | Step) -> Step:
    if isinstance(data, (MoveStep, TurnStep, HoverStep, PhotoStep)):
        return data
    if not isinstance(data, Mapping):
        raise MissionError(f"step must be an object, got {type(data).__name__}")
    kind = str(data.get("kind", "") or "").strip().lower()
    if kind == STEP_MOVE:
        return MoveStep(
            direction=str(data.get("direction", "forward")),
            meters=data.get("meters", 0.0),
        )
    if kind == STEP_TURN:
        return TurnStep.from_dict(data)
    if kind == STEP_HOVER:
        return HoverStep(seconds=data.get("seconds", data.get("duration_s", 0.0)))
    if kind == STEP_PHOTO:
        return PhotoStep(label=str(data.get("label", "photo")))
    raise MissionError(f"unknown step kind: {kind!r}")


@dataclass(frozen=True)
class MissionSpec:
    steps: tuple[Step, ...]
    mission_id: str = ""
    instruction: str = ""
    schema: str = MISSION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != MISSION_SCHEMA:
            raise MissionError(
                f"unknown mission schema {self.schema!r}; expected {MISSION_SCHEMA!r}"
            )
        steps = tuple(step_from_dict(step) for step in self.steps)
        if not steps:
            raise MissionError("mission must contain at least one step")
        object.__setattr__(self, "steps", steps)
        mission_id = str(self.mission_id or "").strip()
        object.__setattr__(
            self, "mission_id", mission_id or f"mission_{uuid.uuid4().hex[:8]}"
        )

    @classmethod
    def from_dict(cls, data: Mapping | Sequence | str) -> "MissionSpec":
        if isinstance(data, str):
            return cls.from_json(data)
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes, Mapping)):
            data = {"steps": list(data)}
        if not isinstance(data, Mapping):
            raise MissionError(f"mission must be an object, got {type(data).__name__}")
        if "mission" in data and isinstance(data["mission"], Mapping):
            merged = dict(data["mission"])
            for key in ("mission_id", "instruction"):
                if key in data and key not in merged:
                    merged[key] = data[key]
            data = merged
        steps = data.get("steps")
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            raise MissionError("mission requires a 'steps' array")
        return cls(
            steps=tuple(step_from_dict(step) for step in steps),
            mission_id=str(data.get("mission_id", "") or ""),
            instruction=str(data.get("instruction", "") or ""),
            schema=str(data.get("schema", MISSION_SCHEMA)),
        )

    @classmethod
    def from_json(cls, text: str) -> "MissionSpec":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MissionError(f"mission JSON is invalid: {exc}") from exc
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "mission_id": self.mission_id,
            "instruction": self.instruction,
            "steps": [step.to_dict() for step in self.steps],
        }


# ---------------------------------------------------------------------------
# simulation clock
# ---------------------------------------------------------------------------


class SimClock:
    """Simulation / flight-controller time. Never a wall-clock substitute."""

    def now_s(self) -> float:
        raise NotImplementedError

    def advance(self, seconds: float) -> float:
        raise NotImplementedError


class VirtualClock(SimClock):
    """Deterministic simulation clock for CPU worlds and tests."""

    def __init__(self, start_s: float = 0.0) -> None:
        self._t = float(start_s)

    def now_s(self) -> float:
        return self._t

    def advance(self, seconds: float) -> float:
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("clock advance must be finite and >= 0")
        self._t += seconds
        return self._t


class WorldClock(SimClock):
    """Clock backed by ``world.advance_time`` / ``world.sim_time`` when exposed."""

    def __init__(self, world: Any, start_s: float = 0.0) -> None:
        self._world = world
        self._fallback = float(start_s)

    def now_s(self) -> float:
        for attr in ("sim_time", "time_s", "sim_time_s"):
            value = getattr(self._world, attr, None)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)
        return self._fallback

    def advance(self, seconds: float) -> float:
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("clock advance must be finite and >= 0")
        advance = getattr(self._world, "advance_time", None)
        if callable(advance):
            advance(seconds)
        else:
            self._fallback = self.now_s() + seconds
            if hasattr(self._world, "sim_time") and not isinstance(
                getattr(type(self._world), "sim_time", None), property
            ):
                try:
                    setattr(self._world, "sim_time", self._fallback)
                except (AttributeError, TypeError):
                    pass
        return self.now_s()


def clock_for_world(world: Any, start_s: float = 0.0) -> SimClock:
    """Use the world's own clock when it has one, else a virtual sim clock.

    A virtual clock is still *simulation* time (an explicit counter advanced by
    the executor); it is never read from the wall clock.
    """
    if callable(getattr(world, "advance_time", None)):
        return WorldClock(world, start_s=start_s)
    for attr in ("sim_time", "time_s", "sim_time_s"):
        if isinstance(getattr(world, attr, None), (int, float)):
            return WorldClock(world, start_s=start_s)
    return VirtualClock(start_s=start_s)


# ---------------------------------------------------------------------------
# execution state
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    step: dict
    status: str = STEP_PENDING
    attempts: int = 0
    start_sim_s: float | None = None
    end_sim_s: float | None = None
    start_pose: list[float] | None = None
    end_pose: list[float] | None = None
    progress: dict = field(default_factory=dict)
    artifact: str = ""
    error: str = ""

    def __post_init__(self) -> None:
        if self.status not in STEP_STATUSES:
            raise MissionError(f"unknown step status: {self.status!r}")
        # Validate the serialized step eagerly so a resumed state cannot hide a
        # corrupt mission until mid-flight.
        step_from_dict(self.step)

    @property
    def kind(self) -> str:
        return str(self.step.get("kind", ""))

    def to_dict(self) -> dict:
        return {
            "step": dict(self.step),
            "status": self.status,
            "attempts": self.attempts,
            "start_sim_s": self.start_sim_s,
            "end_sim_s": self.end_sim_s,
            "start_pose": self.start_pose,
            "end_pose": self.end_pose,
            "progress": dict(self.progress),
            "artifact": self.artifact,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "StepRecord":
        if not isinstance(data, Mapping) or not isinstance(data.get("step"), Mapping):
            raise MissionError("step record is missing its step")
        return cls(
            step=dict(data["step"]),
            status=str(data.get("status", STEP_PENDING)),
            attempts=int(data.get("attempts", 0) or 0),
            start_sim_s=data.get("start_sim_s"),
            end_sim_s=data.get("end_sim_s"),
            start_pose=_pose_list(data.get("start_pose")),
            end_pose=_pose_list(data.get("end_pose")),
            progress=dict(data.get("progress") or {}),
            artifact=str(data.get("artifact", "") or ""),
            error=str(data.get("error", "") or ""),
        )


def _pose_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


@dataclass
class MissionState:
    mission_id: str
    steps: list[StepRecord]
    status: str = MISSION_PENDING
    current_step: int = 0
    instruction: str = ""
    schema: str = STATE_SCHEMA
    clock_s: float = 0.0
    started_sim_s: float | None = None
    finished_sim_s: float | None = None
    error: str = ""
    cancel_reason: str = ""
    runs: int = 0

    def __post_init__(self) -> None:
        if self.schema != STATE_SCHEMA:
            raise MissionError(f"unknown state schema: {self.schema!r}")
        if self.status not in MISSION_STATUSES:
            raise MissionError(f"unknown mission status: {self.status!r}")
        if not self.steps:
            raise MissionError("mission state has no steps")
        if not 0 <= int(self.current_step) <= len(self.steps):
            raise MissionError(
                f"current_step {self.current_step} is outside 0..{len(self.steps)}"
            )

    @classmethod
    def new(cls, spec: MissionSpec, clock_s: float = 0.0) -> "MissionState":
        return cls(
            mission_id=spec.mission_id,
            steps=[StepRecord(step=step.to_dict()) for step in spec.steps],
            status=MISSION_PENDING,
            current_step=0,
            instruction=spec.instruction,
            clock_s=float(clock_s),
        )

    @property
    def spec(self) -> MissionSpec:
        return MissionSpec(
            steps=tuple(step_from_dict(record.step) for record in self.steps),
            mission_id=self.mission_id,
            instruction=self.instruction,
        )

    @property
    def done(self) -> bool:
        return self.status == MISSION_COMPLETED

    def completed_steps(self) -> int:
        return sum(1 for record in self.steps if record.status == STEP_DONE)

    def summary(self) -> dict:
        return {
            "mission_id": self.mission_id,
            "schema": self.schema,
            "status": self.status,
            "current_step": self.current_step,
            "steps_total": len(self.steps),
            "steps_done": self.completed_steps(),
            "clock_s": self.clock_s,
            "started_sim_s": self.started_sim_s,
            "finished_sim_s": self.finished_sim_s,
            "instruction": self.instruction,
            "error": self.error,
            "cancel_reason": self.cancel_reason,
        }

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "mission_id": self.mission_id,
            "status": self.status,
            "current_step": self.current_step,
            "instruction": self.instruction,
            "clock_s": self.clock_s,
            "started_sim_s": self.started_sim_s,
            "finished_sim_s": self.finished_sim_s,
            "error": self.error,
            "cancel_reason": self.cancel_reason,
            "runs": self.runs,
            "steps": [record.to_dict() for record in self.steps],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "MissionState":
        if not isinstance(data, Mapping):
            raise MissionError("mission state must be an object")
        steps = data.get("steps")
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            raise MissionError("mission state requires a 'steps' array")
        return cls(
            mission_id=str(data.get("mission_id", "") or ""),
            steps=[StepRecord.from_dict(item) for item in steps],
            status=str(data.get("status", MISSION_PENDING)),
            current_step=int(data.get("current_step", 0) or 0),
            instruction=str(data.get("instruction", "") or ""),
            schema=str(data.get("schema", STATE_SCHEMA)),
            clock_s=float(data.get("clock_s", 0.0) or 0.0),
            started_sim_s=data.get("started_sim_s"),
            finished_sim_s=data.get("finished_sim_s"),
            error=str(data.get("error", "") or ""),
            cancel_reason=str(data.get("cancel_reason", "") or ""),
            runs=int(data.get("runs", 0) or 0),
        )


def save_state(state: MissionState, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_state(path: str | Path) -> MissionState:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MissionError(f"cannot read mission state {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MissionError(f"mission state {path} is not valid JSON: {exc}") from exc
    return MissionState.from_dict(data)


# ---------------------------------------------------------------------------
# executor
# ---------------------------------------------------------------------------


def _frame_to_pil(frame: Any) -> Image.Image:
    if hasattr(frame, "detach"):
        array = frame.detach().clamp(0, 1).cpu().numpy()
    else:
        array = np.asarray(frame)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.dtype != np.uint8:
        array = (np.clip(array.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(array)


class MotionExecutor:
    """Execute a mission step by step against a world backend."""

    def __init__(
        self,
        world: Any,
        *,
        clock: SimClock | None = None,
        artifacts_dir: str | Path | None = None,
        chunk_m: float = DEFAULT_CHUNK_M,
        chunk_turn_deg: float = DEFAULT_CHUNK_TURN_DEG,
        hover_dt_s: float = DEFAULT_HOVER_DT_S,
        position_tol_m: float = DEFAULT_POSITION_TOL_M,
        angle_tol_deg: float = DEFAULT_ANGLE_TOL_DEG,
        max_chunks: int = MAX_CHUNKS,
        on_progress: Callable[[dict], Any] | None = None,
        state_path: str | Path | None = None,
    ) -> None:
        for attr in ("position_flu", "yaw_rad", "apply_action"):
            if not hasattr(world, attr):
                raise MissionError(f"world does not expose {attr!r}")
        self.world = world
        self.clock = clock if clock is not None else clock_for_world(world)
        self.artifacts_dir = Path(artifacts_dir or "outputs/motion_missions")
        self.chunk_m = float(chunk_m)
        self.chunk_turn_deg = float(chunk_turn_deg)
        self.hover_dt_s = float(hover_dt_s)
        self.position_tol_m = float(position_tol_m)
        self.angle_tol_deg = float(angle_tol_deg)
        self.max_chunks = int(max_chunks)
        self.on_progress = on_progress
        self.state_path = Path(state_path) if state_path else None
        if self.chunk_m <= 0 or self.chunk_turn_deg <= 0 or self.hover_dt_s <= 0:
            raise MissionError("chunk sizes and hover step must be > 0")
        self.state: MissionState | None = None
        self.persist_error = ""
        self._cancel_requested = False
        self._cancel_reason = ""
        self._running = False
        self.step_callbacks = 0

    # -- public API ------------------------------------------------------- #

    @property
    def running(self) -> bool:
        return self._running

    @property
    def cancelled(self) -> bool:
        return self._cancel_requested

    def run(
        self,
        mission: MissionSpec | Mapping | Sequence | None = None,
        *,
        mission_id: str = "",
        instruction: str = "",
    ) -> MissionState:
        if mission is None:
            if self.state is None:
                raise MissionError("no mission to run")
            state = self.state
        else:
            spec = mission if isinstance(mission, MissionSpec) else MissionSpec.from_dict(mission)
            if mission_id or instruction:
                spec = MissionSpec(
                    steps=spec.steps,
                    mission_id=mission_id or spec.mission_id,
                    instruction=instruction or spec.instruction,
                )
            state = MissionState.new(spec, clock_s=self.clock.now_s())
        return self._run_state(state)

    def resume(self, state: MissionState | None = None) -> MissionState:
        state = state or self.state
        if state is None:
            raise MissionError("no mission state to resume")
        self.state = state
        if state.status == MISSION_COMPLETED:
            # Idempotent: a completed mission is never replayed on resume.
            return state
        self._cancel_requested = False
        self._cancel_reason = ""
        state.cancel_reason = ""
        state.status = MISSION_PENDING
        state.error = ""
        for index in range(state.current_step, len(state.steps)):
            record = state.steps[index]
            if record.status in (STEP_CANCELLED, STEP_BLOCKED, STEP_FAILED):
                record.status = STEP_PENDING
                record.error = ""
        return self._run_state(state)

    def cancel(self, reason: str = "cancelled") -> MissionState | None:
        self._cancel_requested = True
        self._cancel_reason = str(reason or "cancelled")
        state = self.state
        if state is None:
            return None
        if state.status in (MISSION_PENDING, MISSION_RUNNING):
            state.status = MISSION_CANCELLED
            state.cancel_reason = self._cancel_reason
            if state.current_step < len(state.steps):
                record = state.steps[state.current_step]
                if record.status in (STEP_PENDING, STEP_RUNNING):
                    record.status = STEP_CANCELLED
                    record.error = self._cancel_reason
                    record.end_sim_s = self.clock.now_s()
                    record.end_pose = self._pose()
                    self._refresh_cancel_progress(record)
            self._persist()
        return state

    def _refresh_cancel_progress(self, record: StepRecord) -> None:
        """Recompute the cancelled step's remaining work from the actual pose.

        A chunk updates ``moved_m`` before the loop sees the cancel flag, so
        without this the state would report the *requested* remaining distance
        instead of the distance actually left to fly.
        """
        target = record.progress.get("target")
        start = record.progress.get("start_pose") or record.start_pose
        if record.kind == STEP_MOVE and target is not None and start is not None:
            current = np.asarray(self._pose(), dtype=np.float64)
            target_vec = np.asarray(target, dtype=np.float64)
            record.progress["remaining_m"] = round(
                float(np.linalg.norm(target_vec - current)), 6
            )
            record.progress["moved_m"] = round(
                float(np.linalg.norm(current - np.asarray(start, dtype=np.float64))),
                6,
            )

    # -- state machine ---------------------------------------------------- #

    def _run_state(self, state: MissionState) -> MissionState:
        if self._running:
            raise MissionError("executor is already running")
        self.state = state
        self._running = True
        state.runs += 1
        state.status = MISSION_RUNNING
        state.started_sim_s = (
            state.started_sim_s if state.started_sim_s is not None else self.clock.now_s()
        )
        state.finished_sim_s = None
        state.clock_s = self.clock.now_s()
        self._notify("mission.started")
        try:
            while state.current_step < len(state.steps):
                index = state.current_step
                record = state.steps[index]
                if self._cancel_requested:
                    self._cancel_current(record, state)
                    break
                if record.status == STEP_DONE:
                    state.current_step += 1
                    continue
                if record.status in (STEP_BLOCKED, STEP_FAILED):
                    # A resumed/blocked step is reopened explicitly by resume().
                    state.status = (
                        MISSION_BLOCKED
                        if record.status == STEP_BLOCKED else MISSION_FAILED
                    )
                    state.error = record.error
                    break
                self._execute_step(record, index, state)
                if record.status == STEP_DONE:
                    state.current_step += 1
                    self._maybe_checkpoint()
                    continue
                if record.status == STEP_CANCELLED:
                    state.status = MISSION_CANCELLED
                    state.cancel_reason = record.error or self._cancel_reason or "cancelled"
                    break
                if record.status == STEP_BLOCKED:
                    state.status = MISSION_BLOCKED
                    state.error = record.error
                    break
                state.status = MISSION_FAILED
                state.error = record.error or f"step {index} ended as {record.status}"
                break
            else:
                state.status = MISSION_COMPLETED
                state.current_step = len(state.steps)
                state.finished_sim_s = self.clock.now_s()
            if state.status in (
                MISSION_COMPLETED, MISSION_CANCELLED, MISSION_BLOCKED, MISSION_FAILED
            ):
                state.finished_sim_s = state.finished_sim_s or self.clock.now_s()
            state.clock_s = self.clock.now_s()
            return state
        finally:
            self._running = False
            self._persist()
            self._notify("mission.finished")

    def _cancel_current(self, record: StepRecord, state: MissionState) -> None:
        if record.status in (STEP_PENDING, STEP_RUNNING):
            record.status = STEP_CANCELLED
            record.error = self._cancel_reason or "cancelled"
            record.end_sim_s = self.clock.now_s()
            record.end_pose = self._pose()
        state.status = MISSION_CANCELLED
        state.cancel_reason = self._cancel_reason or "cancelled"

    def _execute_step(self, record: StepRecord, index: int, state: MissionState) -> str:
        step = step_from_dict(record.step)
        record.attempts += 1
        record.status = STEP_RUNNING
        record.error = ""
        if record.start_sim_s is None:
            record.start_sim_s = self.clock.now_s()
        if record.start_pose is None:
            record.start_pose = self._pose()
        state.current_step = index
        self._notify("step.started", index=index, kind=step.kind)
        try:
            if isinstance(step, MoveStep):
                self._step_move(record, step, index)
            elif isinstance(step, TurnStep):
                self._step_turn(record, step, index)
            elif isinstance(step, HoverStep):
                self._step_hover(record, step, index)
            elif isinstance(step, PhotoStep):
                self._step_photo(record, step, index)
        except Exception as exc:  # noqa: BLE001 - a step failure is state, not a crash
            record.status = STEP_FAILED
            record.error = f"{type(exc).__name__}: {exc}"
        if record.status in (STEP_DONE, STEP_CANCELLED, STEP_BLOCKED, STEP_FAILED):
            record.end_sim_s = self.clock.now_s()
            record.end_pose = self._pose()
        self._notify("step.finished", index=index, kind=step.kind)
        return record.status

    # -- steps ------------------------------------------------------------ #

    def _step_move(self, record: StepRecord, step: MoveStep, index: int) -> None:
        target = record.progress.get("target")
        start_yaw = record.progress.get("start_yaw_rad")
        if target is None or start_yaw is None:
            start_pose = np.asarray(self._pose(), dtype=np.float64)
            start_yaw = float(self.world.yaw_rad)
            unit = _DIRECTIONS[step.direction]
            forward_m = unit[0] * step.meters
            right_m = unit[1] * step.meters
            up_m = unit[2] * step.meters
            delta = body_delta_to_world(forward_m, -right_m, up_m, start_yaw)
            target = (start_pose + np.asarray(delta, dtype=np.float64)).tolist()
            record.progress.update({
                "target": target,
                "start_yaw_rad": start_yaw,
                "requested_m": step.meters,
                "direction": step.direction,
                "moved_m": 0.0,
                "remaining_m": step.meters,
                "chunks": 0,
            })
        target_vec = np.asarray(target, dtype=np.float64)
        tolerance = min(
            self.position_tol_m,
            max(1e-3, step.meters * 0.05),
        )
        record.progress["tolerance_m"] = tolerance
        no_progress = int(record.progress.get("no_progress", 0) or 0)
        chunks = int(record.progress.get("chunks", 0) or 0)
        last_feedback = str(record.progress.get("last_feedback", "") or "")
        start = np.asarray(
            record.progress.get("start_pose") or record.start_pose or self._pose(),
            dtype=np.float64,
        )
        while True:
            if self._cancel_requested:
                record.status = STEP_CANCELLED
                record.error = self._cancel_reason or "cancelled"
                return
            current = np.asarray(self._pose(), dtype=np.float64)
            delta = target_vec - current
            remaining = float(np.linalg.norm(delta))
            record.progress["remaining_m"] = round(remaining, 6)
            if remaining <= tolerance:
                record.progress["remaining_m"] = 0.0
                record.progress["moved_m"] = round(
                    float(np.linalg.norm(current - start)), 6
                )
                record.progress["no_progress"] = 0
                record.status = STEP_DONE
                return
            if chunks >= self.max_chunks:
                record.status = STEP_BLOCKED
                record.error = f"max_chunks_exceeded remaining={remaining:.3f}m"
                return
            step_len = min(remaining, self.chunk_m)
            unit = delta / remaining
            yaw = float(self.world.yaw_rad)
            # world -> body (inverse of actions.body_delta_to_world)
            body_x = math.cos(yaw) * unit[0] + math.sin(yaw) * unit[1]
            body_y = -math.sin(yaw) * unit[0] + math.cos(yaw) * unit[1]
            body_z = unit[2]
            action = NavAction(
                kind="move",
                forward_m=body_x * step_len,
                right_m=-body_y * step_len,
                up_m=body_z * step_len,
                reason=f"motion step {index}: {step.direction} {step.meters:g}m",
            )
            before = current
            last_feedback = str(self.world.apply_action(action))
            after = np.asarray(self._pose(), dtype=np.float64)
            measured = float(np.linalg.norm(after - before))
            chunks += 1
            record.progress["chunks"] = chunks
            record.progress["moved_m"] = round(
                float(np.linalg.norm(after - start)), 6
            )
            record.progress["last_feedback"] = last_feedback
            record.progress["last_chunk_m"] = round(measured, 6)
            record.progress["remaining_m"] = round(
                float(np.linalg.norm(target_vec - after)), 6
            )
            self._notify("step.chunk", index=index, kind=STEP_MOVE)
            if measured <= 1e-6 or float(np.linalg.norm(target_vec - after)) >= remaining - 1e-6:
                no_progress += 1
                record.progress["no_progress"] = no_progress
                if no_progress >= 2:
                    record.status = STEP_BLOCKED
                    record.progress["remaining_m"] = round(
                        float(np.linalg.norm(target_vec - after)), 6
                    )
                    record.error = (
                        "movement made no progress (world bounds or obstacle): "
                        f"{last_feedback}"
                    )
                    return
            else:
                no_progress = 0
                record.progress["no_progress"] = 0

    def _step_turn(self, record: StepRecord, step: TurnStep, index: int) -> None:
        start_deg = record.progress.get("start_yaw_deg")
        if start_deg is None:
            start_deg = math.degrees(float(self.world.yaw_rad))
            record.progress.update({
                "start_yaw_deg": start_deg,
                "target_yaw_deg": _wrap_deg(start_deg + step.degrees),
                "requested_deg": step.degrees,
                "turned_deg": 0.0,
            })
        target_deg = float(record.progress["target_yaw_deg"])
        no_progress = int(record.progress.get("no_progress", 0) or 0)
        chunks = int(record.progress.get("chunks", 0) or 0)
        while True:
            if self._cancel_requested:
                record.status = STEP_CANCELLED
                record.error = self._cancel_reason or "cancelled"
                return
            yaw_deg = math.degrees(float(self.world.yaw_rad))
            delta = _wrap_deg(target_deg - yaw_deg)
            record.progress["turned_deg"] = round(_wrap_deg(yaw_deg - float(start_deg)), 6)
            if abs(delta) <= self.angle_tol_deg:
                record.progress["remaining_deg"] = 0.0
                record.status = STEP_DONE
                return
            if chunks >= self.max_chunks:
                record.status = STEP_BLOCKED
                record.error = f"max_chunks_exceeded remaining={delta:.3f}deg"
                return
            chunk = max(-self.chunk_turn_deg, min(self.chunk_turn_deg, delta))
            before = yaw_deg
            feedback = self.world.apply_action(
                NavAction(
                    kind="turn",
                    yaw_deg=chunk,
                    reason=f"motion step {index}: turn {step.degrees:g}deg",
                )
            )
            after = math.degrees(float(self.world.yaw_rad))
            progressed = abs(_wrap_deg(after - before))
            chunks += 1
            record.progress["chunks"] = chunks
            record.progress["turned_deg"] = round(
                _wrap_deg(after - float(start_deg)), 6
            )
            record.progress["remaining_deg"] = round(
                _wrap_deg(target_deg - after), 6
            )
            record.progress["last_feedback"] = str(feedback)
            self._notify("step.chunk", index=index, kind=STEP_TURN)
            if progressed <= 1e-6:
                no_progress += 1
                record.progress["no_progress"] = no_progress
                if no_progress >= 2:
                    record.status = STEP_BLOCKED
                    record.error = f"turn made no progress: {feedback}"
                    return
            else:
                no_progress = 0
                record.progress["no_progress"] = 0

    def _step_hover(self, record: StepRecord, step: HoverStep, index: int) -> None:
        elapsed = float(record.progress.get("elapsed_s", 0.0) or 0.0)
        start = record.progress.get("start_sim_s")
        if start is None:
            start = self.clock.now_s()
            record.progress.update({
                "start_sim_s": start,
                "requested_s": step.seconds,
                "elapsed_s": elapsed,
            })
        while step.seconds - elapsed > 1e-9:
            if self._cancel_requested:
                record.status = STEP_CANCELLED
                record.error = self._cancel_reason or "cancelled"
                return
            before = self.clock.now_s()
            dt = min(self.hover_dt_s, step.seconds - elapsed)
            self.clock.advance(dt)
            after = self.clock.now_s()
            advanced = after - before
            if advanced <= 1e-9:
                record.status = STEP_FAILED
                record.error = "simulation clock did not advance during hover"
                return
            elapsed += advanced
            record.progress["elapsed_s"] = round(elapsed, 9)
            record.progress["remaining_s"] = round(
                max(0.0, step.seconds - elapsed), 9
            )
            record.progress["clock_s"] = after
            self._notify("step.chunk", index=index, kind=STEP_HOVER)
        if self.clock.now_s() - float(start) + 1e-9 < step.seconds:
            record.status = STEP_FAILED
            record.error = "hover ended before the requested simulation time"
            return
        record.progress["elapsed_s"] = round(elapsed, 9)
        record.progress["remaining_s"] = 0.0
        record.status = STEP_DONE

    def _step_photo(self, record: StepRecord, step: PhotoStep, index: int) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        safe_mission = _LABEL_RE.sub("_", self.state.mission_id if self.state else "mission")
        path = self.artifacts_dir / (
            f"{safe_mission}_{index:02d}_{step.safe_label}.png"
        )
        if path.exists() and path.stat().st_size > 0:
            # Idempotent resume: the evidence already exists, do not retake it.
            record.artifact = str(path)
            record.progress["artifact"] = str(path)
            record.progress["reused"] = True
            record.progress.setdefault("pose_at_photo", self._pose())
            record.status = STEP_DONE
            return
        render = getattr(self.world, "render", None)
        if not callable(render):
            record.status = STEP_FAILED
            record.error = "world cannot render a photo"
            return
        frame = render()
        image = _frame_to_pil(frame)
        tmp = path.with_name(path.name + ".tmp")
        image.save(tmp, format="PNG")
        os.replace(tmp, path)
        if not path.exists() or path.stat().st_size <= 0:
            record.status = STEP_FAILED
            record.error = f"photo artifact was not written: {path}"
            return
        record.artifact = str(path)
        record.progress.update({
            "artifact": str(path),
            "reused": False,
            "pose_at_photo": self._pose(),
            "size_bytes": int(path.stat().st_size),
        })
        self._notify("step.chunk", index=index, kind=STEP_PHOTO)
        record.status = STEP_DONE

    # -- helpers ---------------------------------------------------------- #

    def _pose(self) -> list[float]:
        position = getattr(self.world, "position_flu")
        try:
            return [float(item) for item in position]
        except TypeError:
            return [float(position)]

    def _notify(self, event: str, **extra: Any) -> None:
        self.step_callbacks += 1
        # Checkpoint every chunk: a crash mid-step then resumes from the
        # measured pose/target instead of the last completed step.
        self._persist()
        if self.on_progress is None or self.state is None:
            return
        payload = {
            "event": event,
            "mission_id": self.state.mission_id,
            "mission_status": self.state.status,
            "current_step": self.state.current_step,
            "clock_s": self.clock.now_s(),
            "pose": self._pose(),
            **extra,
        }
        try:
            self.on_progress(payload)
        except Exception:  # noqa: BLE001 - observers must not break execution
            pass

    def _maybe_checkpoint(self) -> None:
        self._persist()

    def _persist(self) -> None:
        if self.state is None or self.state_path is None:
            return
        try:
            save_state(self.state, self.state_path)
            self.persist_error = ""
        except OSError as exc:
            # A read-only artifacts directory must not abort a flight; the
            # error is kept on the executor and reported by the caller.
            self.persist_error = f"{type(exc).__name__}: {exc}"
