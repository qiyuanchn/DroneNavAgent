"""The base motion plugin: task-level move / turn / hover / photo (goal.md M1).

The plugin is the first complete consumer of the lifecycle host. It declares the
``motion.service@1`` interface, registers a small tool surface, and owns no
mission history: photos and ``*.state.json`` files live in the caller's
artifacts directory, so uninstalling the plugin leaves an audit trail readable.

The service contract (all methods may be called by another plugin instead of
through the tool surface):

``execute(ctx, mission, *, mission_id, instruction, artifacts_dir)``
    Run a :class:`~vlm_nav.motion.MissionSpec` to completion.
``resume(ctx, *, state_path, artifacts_dir)``
    Continue a stored mission from its measured progress.
``cancel(ctx, reason)``
    Request cancellation of the current/last mission.
``state(ctx, *, state_path)``
    Return the current :class:`~vlm_nav.motion.MissionState`.
"""

from __future__ import annotations

from pathlib import Path
import re
import threading
from typing import Any, Mapping, Protocol, runtime_checkable

from ..motion import (
    MissionError,
    MissionSpec,
    MotionExecutor,
    MissionState,
    MoveStep,
    HoverStep,
    PhotoStep,
    TurnStep,
    clock_for_world,
    load_state,
    save_state,
)
from ..tools import (
    STATUS_BLOCKED,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_OK,
    EvidenceRef,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from . import Plugin, PluginContext, PluginManifest

MOTION_SERVICE = "motion.service@1"


@runtime_checkable
class MotionService(Protocol):
    """The interface another plugin is allowed to depend on.

    Callers resolve this by contract name from the host; they never import the
    plugin implementation (goal.md section 2).
    """

    def execute(
        self,
        ctx: ToolContext,
        mission: MissionSpec | Mapping,
        *,
        mission_id: str = "",
        instruction: str = "",
        artifacts_dir: str | Path | None = None,
        state_path: str | Path | None = None,
    ) -> MissionState: ...

    def resume(
        self,
        ctx: ToolContext,
        *,
        state_path: str | Path | None = None,
        artifacts_dir: str | Path | None = None,
    ) -> MissionState: ...

    def cancel(self, ctx: ToolContext, reason: str = "cancelled") -> MissionState | None: ...

    def state(
        self, ctx: ToolContext, *, state_path: str | Path | None = None
    ) -> MissionState | None: ...

#: ``ctx.extra`` keys shared by this plugin's tools.
MOTION_EXECUTOR_KEY = "motion.executor"
MOTION_STATE_KEY = "motion.state"
MOTION_STATE_PATH_KEY = "motion.state_path"
MOTION_ARTIFACTS_KEY = "motion.artifacts_dir"
MOTION_CLOCK_KEY = "motion.clock"

DEFAULT_ARTIFACTS = Path("outputs") / "motion_missions"

MOTION_TOOLS = (
    "motion_mission",
    "motion_move",
    "motion_turn",
    "motion_hover",
    "motion_photo",
    "motion_cancel",
    "motion_resume",
    "motion_state",
)

MOTION_EVENTS = (
    "motion.mission.started",
    "motion.mission.finished",
    "motion.mission.cancelled",
)


def _mission_data(args: Mapping) -> dict:
    if isinstance(args.get("mission"), Mapping):
        data = dict(args["mission"])
    elif "steps" in args:
        data = {"steps": args["steps"]}
    else:
        raise MissionError("provide a 'mission' object or a 'steps' array")
    for key in ("mission_id", "instruction"):
        if key in args and key not in data:
            data[key] = args[key]
    return data


def _step_rows(state: MissionState) -> list[dict]:
    return [
        {
            "index": index,
            "kind": record.kind,
            "status": record.status,
            "attempts": record.attempts,
            "progress": dict(record.progress),
            "artifact": record.artifact,
            "error": record.error,
        }
        for index, record in enumerate(state.steps)
    ]


def _evidence(state: MissionState) -> list[EvidenceRef]:
    refs: list[EvidenceRef] = []
    for index, record in enumerate(state.steps):
        if record.artifact:
            refs.append(EvidenceRef(
                kind="image",
                ref=record.artifact,
                note=f"mission {state.mission_id} step {index} ({record.kind})",
            ))
    return refs


def _result_for_state(state: MissionState, state_path: Path | None) -> ToolResult:
    if state.status == "completed":
        status = STATUS_OK
    elif state.status in ("cancelled", "blocked"):
        status = STATUS_BLOCKED
    else:
        status = STATUS_FAILED
    moved = any(
        record.kind == "move" and record.status == "done" for record in state.steps
    )
    data = {
        **state.summary(),
        "steps": _step_rows(state),
        "state_path": str(state_path) if state_path else "",
        "progress_kind": (
            "task" if state.status == "completed"
            else "action" if moved
            else "none"
        ),
    }
    artifacts = [
        record.artifact for record in state.steps if record.artifact
    ]
    if artifacts:
        data["artifacts"] = artifacts
    lines = [
        f"[{status}] mission {state.mission_id}: "
        f"{state.completed_steps()}/{len(state.steps)} steps, "
        f"{state.status}, clock={state.clock_s:.2f}s"
    ]
    for index, record in enumerate(state.steps):
        detail = ""
        if record.kind == "move":
            detail = (
                f"{record.progress.get('direction', '')} "
                f"{record.progress.get('moved_m', 0):.2f}/"
                f"{record.progress.get('requested_m', 0):.2f}m"
            )
        elif record.kind == "turn":
            detail = (
                f"{record.progress.get('turned_deg', 0):.1f}/"
                f"{record.progress.get('requested_deg', 0):.1f}deg"
            )
        elif record.kind == "hover":
            detail = (
                f"{record.progress.get('elapsed_s', 0):.2f}/"
                f"{record.progress.get('requested_s', 0):.2f}s"
            )
        elif record.kind == "photo":
            detail = record.artifact or "no artifact"
        if record.error:
            detail = f"{detail}; error={record.error}".strip("; ")
        lines.append(f"  {index + 1}. {record.kind} [{record.status}] {detail}".rstrip())
    return ToolResult(
        status, data, _evidence(state), prompt_text="\n".join(lines)
    )


class MotionPlugin(Plugin):
    """Installable/disable-able/uninstallable task-level motion capability."""

    manifest = PluginManifest(
        id="motion.base",
        version="0.1.0",
        provides=(MOTION_SERVICE,),
        tools=MOTION_TOOLS,
        events=MOTION_EVENTS,
        data_dirs=("state",),
        cache_dirs=("cache",),
    )

    def __init__(
        self,
        *,
        chunk_m: float = 2.0,
        chunk_turn_deg: float = 45.0,
        hover_dt_s: float = 0.1,
        artifacts_dir: str | Path | None = None,
    ) -> None:
        self.chunk_m = chunk_m
        self.chunk_turn_deg = chunk_turn_deg
        self.hover_dt_s = hover_dt_s
        self.default_artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self._ctx: PluginContext | None = None
        self._executors: list[MotionExecutor] = []
        self._busy = 0
        self._lock = threading.RLock()

    # -- lifecycle -------------------------------------------------------- #

    def activate(self, ctx: PluginContext) -> None:
        self._ctx = ctx
        ctx.services.provide(self.manifest.id, MOTION_SERVICE, self)
        for spec in self._tool_specs():
            ctx.tools.register(spec)
        ctx.activity_probe(self.active_tasks)
        ctx.abort(self.abort)

    def deactivate(self, ctx: PluginContext) -> None:
        self.abort("plugin deactivated")
        with self._lock:
            self._executors.clear()
        self._ctx = None

    def active_tasks(self) -> int:
        with self._lock:
            return int(self._busy)

    def abort(self, reason: str = "aborted") -> None:
        with self._lock:
            executors = list(self._executors)
        for executor in executors:
            if executor.running:
                executor.cancel(reason)

    # -- MotionService contract ------------------------------------------- #

    def execute(
        self,
        ctx: ToolContext,
        mission: MissionSpec | Mapping,
        *,
        mission_id: str = "",
        instruction: str = "",
        artifacts_dir: str | Path | None = None,
        state_path: str | Path | None = None,
    ) -> MissionState:
        executor = self._executor(ctx, artifacts_dir=artifacts_dir, state_path=state_path)
        self._emit(ctx, "motion.mission.started", mission_id=mission_id)
        with self._enter():
            state = executor.run(mission, mission_id=mission_id, instruction=instruction)
        self._remember(ctx, executor, state, state_path=state_path)
        self._emit(
            ctx, "motion.mission.finished",
            mission_id=state.mission_id, status=state.status,
        )
        return state

    def resume(
        self,
        ctx: ToolContext,
        *,
        state_path: str | Path | None = None,
        artifacts_dir: str | Path | None = None,
    ) -> MissionState:
        state = self._state(ctx, state_path=state_path)
        if state is None:
            raise MissionError("no mission state to resume")
        executor = self._executor(
            ctx, artifacts_dir=artifacts_dir, state_path=state_path,
        )
        executor.state = state
        with self._enter():
            state = executor.resume(state)
        self._remember(ctx, executor, state, state_path=state_path)
        self._emit(
            ctx, "motion.mission.finished",
            mission_id=state.mission_id, status=state.status, resumed=True,
        )
        return state

    def cancel(self, ctx: ToolContext, reason: str = "cancelled") -> MissionState | None:
        state = self._state(ctx)
        executor = ctx.extra.get(MOTION_EXECUTOR_KEY)
        if executor is None:
            if state is None:
                return None
            executor = self._executor(ctx)
            executor.state = state
        state = executor.cancel(reason)
        if state is not None:
            self._remember(ctx, executor, state)
            self._emit(
                ctx, "motion.mission.cancelled",
                mission_id=state.mission_id, status=state.status,
                reason=reason,
            )
        return state

    def state(
        self, ctx: ToolContext, *, state_path: str | Path | None = None
    ) -> MissionState | None:
        return self._state(ctx, state_path=state_path)

    # -- tools ------------------------------------------------------------ #

    def _tool_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="motion_mission",
                description=(
                    "Execute an ordered task-level motion mission. Steps: "
                    "{kind:'move',direction:'forward|backward|left|right|up|down',meters}, "
                    "{kind:'turn',degrees|direction}, {kind:'hover',seconds}, "
                    "{kind:'photo',label}. A move target is fixed at the step's "
                    "starting heading; long moves are split into chunks but "
                    "completion is measured against the full requested distance. "
                    "Hover advances simulation time, never wall-clock latency."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "mission": {"type": "object"},
                        "steps": {"type": "array"},
                        "mission_id": {"type": "string"},
                        "instruction": {"type": "string"},
                        "resume": {"type": "boolean"},
                        "artifacts_dir": {"type": "string"},
                        "state_path": {"type": "string"},
                    },
                },
                handler=self._tool_mission,
                side_effects=True,
            ),
            ToolSpec(
                name="motion_move",
                description=(
                    "Move a full requested distance in body-frame direction. The "
                    "direction is the heading at this step's start; execution is "
                    "segmented, the remaining distance is tracked, and a truncated "
                    "or clamped move is reported blocked instead of done."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": [
                                "forward", "backward", "left", "right", "up", "down",
                            ],
                        },
                        "meters": {"type": "number"},
                        "artifacts_dir": {"type": "string"},
                    },
                    "required": ["meters"],
                },
                handler=self._tool_move,
                side_effects=True,
            ),
            ToolSpec(
                name="motion_turn",
                description=(
                    "Turn in place by degrees (positive = left/CCW) or with "
                    "direction='left'|'right'. Completion is checked against the "
                    "target heading, not the number of chunks issued."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "degrees": {"type": "number"},
                        "direction": {"type": "string", "enum": ["left", "right"]},
                        "artifacts_dir": {"type": "string"},
                    },
                },
                handler=self._tool_turn,
                side_effects=True,
            ),
            ToolSpec(
                name="motion_hover",
                description=(
                    "Hold position for the requested simulation seconds. The "
                    "executor advances the world/simulation clock; it never treats "
                    "one model call's wall-clock latency as hover time."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "seconds": {"type": "number"},
                        "artifacts_dir": {"type": "string"},
                    },
                    "required": ["seconds"],
                },
                handler=self._tool_hover,
                side_effects=True,
            ),
            ToolSpec(
                name="motion_photo",
                description=(
                    "Render the current view and save a PNG artifact. The artifact "
                    "path is returned as evidence; resume reuses an existing "
                    "artifact instead of retaking the photo."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "artifacts_dir": {"type": "string"},
                    },
                },
                handler=self._tool_photo,
                side_effects=True,
            ),
            ToolSpec(
                name="motion_cancel",
                description=(
                    "Request cancellation. The current step keeps its measured "
                    "progress and is cancelled; completed steps stay done. Resume "
                    "continues from the remaining distance/angle/time."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                },
                handler=self._tool_cancel,
                side_effects=True,
            ),
            ToolSpec(
                name="motion_resume",
                description=(
                    "Resume the stored mission from measured progress. Completed "
                    "steps are not replayed and an existing photo artifact is not "
                    "retaken."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "state_path": {"type": "string"},
                        "artifacts_dir": {"type": "string"},
                    },
                },
                handler=self._tool_resume,
                side_effects=True,
            ),
            ToolSpec(
                name="motion_state",
                description=(
                    "Read the current (or a stored) mission state: per-step status, "
                    "completed/remaining distance, angle or simulation time, and "
                    "artifact paths."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"state_path": {"type": "string"}},
                },
                handler=self._tool_state,
                side_effects=False,
                cacheable=False,
            ),
        ]

    def _tool_mission(self, ctx: ToolContext, args: dict) -> ToolResult:
        try:
            if bool(args.get("resume")):
                state = self.resume(
                    ctx,
                    state_path=args.get("state_path"),
                    artifacts_dir=args.get("artifacts_dir"),
                )
            else:
                spec = MissionSpec.from_dict(_mission_data(args))
                state = self.execute(
                    ctx,
                    spec,
                    mission_id=str(args.get("mission_id", "") or ""),
                    instruction=str(args.get("instruction", "") or ""),
                    artifacts_dir=args.get("artifacts_dir"),
                    state_path=args.get("state_path"),
                )
            return _result_for_state(state, self._state_path(ctx))
        except Exception as exc:  # noqa: BLE001 - tool boundary reports, never crashes
            return ToolResult(STATUS_FAILED, {"error": f"{type(exc).__name__}: {exc}"})

    def _tool_move(self, ctx: ToolContext, args: dict) -> ToolResult:
        try:
            step = MoveStep(
                direction=str(args.get("direction", "forward")),
                meters=args.get("meters", 0.0),
            )
            spec = MissionSpec(
                steps=(step,),
                instruction=f"move {step.direction} {step.meters:g}m",
            )
            state = self.execute(
                ctx, spec, artifacts_dir=args.get("artifacts_dir"),
            )
            return _result_for_state(state, self._state_path(ctx))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(STATUS_FAILED, {"error": f"{type(exc).__name__}: {exc}"})

    def _tool_turn(self, ctx: ToolContext, args: dict) -> ToolResult:
        try:
            step = TurnStep.from_dict(args)
            spec = MissionSpec(
                steps=(step,), instruction=f"turn {step.degrees:g}deg"
            )
            state = self.execute(
                ctx, spec, artifacts_dir=args.get("artifacts_dir"),
            )
            return _result_for_state(state, self._state_path(ctx))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(STATUS_FAILED, {"error": f"{type(exc).__name__}: {exc}"})

    def _tool_hover(self, ctx: ToolContext, args: dict) -> ToolResult:
        try:
            step = HoverStep(seconds=args.get("seconds", 0.0))
            spec = MissionSpec(
                steps=(step,), instruction=f"hover {step.seconds:g}s"
            )
            state = self.execute(
                ctx, spec, artifacts_dir=args.get("artifacts_dir"),
            )
            return _result_for_state(state, self._state_path(ctx))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(STATUS_FAILED, {"error": f"{type(exc).__name__}: {exc}"})

    def _tool_photo(self, ctx: ToolContext, args: dict) -> ToolResult:
        try:
            step = PhotoStep(label=str(args.get("label", "photo")))
            spec = MissionSpec(steps=(step,), instruction=f"photo {step.label}")
            state = self.execute(
                ctx, spec, artifacts_dir=args.get("artifacts_dir"),
            )
            return _result_for_state(state, self._state_path(ctx))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(STATUS_FAILED, {"error": f"{type(exc).__name__}: {exc}"})

    def _tool_cancel(self, ctx: ToolContext, args: dict) -> ToolResult:
        try:
            state = self.cancel(ctx, str(args.get("reason", "cancelled")))
            if state is None:
                return ToolResult(STATUS_EMPTY, {"reason": "no active mission"})
            return _result_for_state(state, self._state_path(ctx))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(STATUS_FAILED, {"error": f"{type(exc).__name__}: {exc}"})

    def _tool_resume(self, ctx: ToolContext, args: dict) -> ToolResult:
        try:
            state = self.resume(
                ctx,
                state_path=args.get("state_path"),
                artifacts_dir=args.get("artifacts_dir"),
            )
            return _result_for_state(state, self._state_path(ctx))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(STATUS_FAILED, {"error": f"{type(exc).__name__}: {exc}"})

    def _tool_state(self, ctx: ToolContext, args: dict) -> ToolResult:
        try:
            state = self.state(ctx, state_path=args.get("state_path"))
            if state is None:
                return ToolResult(STATUS_EMPTY, {"reason": "no mission state"})
            return _result_for_state(state, self._state_path(ctx))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(STATUS_FAILED, {"error": f"{type(exc).__name__}: {exc}"})

    # -- internals -------------------------------------------------------- #

    def _executor(
        self,
        ctx: ToolContext,
        *,
        artifacts_dir: str | Path | None = None,
        state_path: str | Path | None = None,
    ) -> MotionExecutor:
        world = ctx.world
        if world is None:
            raise MissionError("motion requires a world backend")
        directory = Path(
            artifacts_dir
            or ctx.extra.get(MOTION_ARTIFACTS_KEY)
            or self.default_artifacts_dir
            or DEFAULT_ARTIFACTS
        )
        ctx.extra[MOTION_ARTIFACTS_KEY] = str(directory)
        clock = ctx.extra.get(MOTION_CLOCK_KEY)
        if clock is None:
            clock = clock_for_world(world)
            ctx.extra[MOTION_CLOCK_KEY] = clock
        executor = ctx.extra.get(MOTION_EXECUTOR_KEY)
        if executor is not None and executor.world is world:
            executor.artifacts_dir = directory
            executor.clock = clock
            executor.chunk_m = self.chunk_m
            executor.chunk_turn_deg = self.chunk_turn_deg
            executor.hover_dt_s = self.hover_dt_s
        else:
            executor = MotionExecutor(
                world,
                clock=clock,
                artifacts_dir=directory,
                chunk_m=self.chunk_m,
                chunk_turn_deg=self.chunk_turn_deg,
                hover_dt_s=self.hover_dt_s,
            )
            ctx.extra[MOTION_EXECUTOR_KEY] = executor
            with self._lock:
                self._executors.append(executor)
        if state_path:
            executor.state_path = Path(state_path)
            ctx.extra[MOTION_STATE_PATH_KEY] = str(state_path)
        return executor

    def _remember(
        self,
        ctx: ToolContext,
        executor: MotionExecutor,
        state: MissionState,
        *,
        state_path: str | Path | None = None,
    ) -> None:
        ctx.extra[MOTION_STATE_KEY] = state
        path = Path(state_path) if state_path else executor.state_path
        if path is None:
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", state.mission_id).strip("._")
            path = executor.artifacts_dir / f"{safe_id or 'mission'}.state.json"
            executor.state_path = path
        try:
            save_state(state, path)
        except OSError as exc:
            executor.persist_error = f"{type(exc).__name__}: {exc}"
        ctx.extra[MOTION_STATE_PATH_KEY] = str(path)

    def _state_path(self, ctx: ToolContext) -> Path | None:
        value = ctx.extra.get(MOTION_STATE_PATH_KEY)
        if value:
            return Path(value)
        executor = ctx.extra.get(MOTION_EXECUTOR_KEY)
        return executor.state_path if executor is not None else None

    def _state(
        self, ctx: ToolContext, *, state_path: str | Path | None = None
    ) -> MissionState | None:
        if state_path:
            state = load_state(state_path)
            ctx.extra[MOTION_STATE_KEY] = state
            ctx.extra[MOTION_STATE_PATH_KEY] = str(state_path)
            executor = ctx.extra.get(MOTION_EXECUTOR_KEY)
            if executor is not None and executor.world is ctx.world:
                executor.state = state
                executor.state_path = Path(state_path)
            return state
        state = ctx.extra.get(MOTION_STATE_KEY)
        if isinstance(state, MissionState):
            return state
        executor = ctx.extra.get(MOTION_EXECUTOR_KEY)
        if executor is not None and isinstance(executor.state, MissionState):
            return executor.state
        return None

    def _enter(self):
        plugin = self

        class _Busy:
            def __enter__(self_inner):
                with plugin._lock:
                    plugin._busy += 1
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                with plugin._lock:
                    plugin._busy = max(0, plugin._busy - 1)
                return False

        return _Busy()

    def _emit(self, ctx: ToolContext, event: str, **payload: Any) -> None:
        plugin_ctx = self._ctx
        if plugin_ctx is None:
            return
        try:
            plugin_ctx.emit(event, payload)
        except Exception:  # noqa: BLE001 - event observers must not break a mission
            pass
