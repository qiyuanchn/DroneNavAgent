"""Mixed-mission plugin: ordered motion + semantic + control steps.

The plugin exposes ``mission.service@1`` and a six-tool surface
(``mission_plan``/``mission_run``/``mission_status``/``mission_pause``/
``mission_resume``/``mission_cancel``).  It resolves the motion, semantic and
planner services through the registry and keeps the mixed mission state in
``ctx.extra`` so a pause/status/cancel call can find the executor it belongs
to.

The plugin never invents evidence: a semantic step without a semantic service
is a structured failure, and mission completion follows the executor state,
which only marks semantic steps done on candidate/arrived/accepted/verified.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping

from ..mixed_mission import (
    MIXED_CANCELLED,
    MIXED_COMPLETED,
    MIXED_FAILED,
    MIXED_BLOCKED,
    MISSION_ARTIFACTS_KEY,
    MISSION_EXECUTOR_KEY,
    MISSION_SEMANTIC_KEY,
    MISSION_STATE_KEY,
    MISSION_STATE_PATH_KEY,
    MixedMissionError,
    MixedMissionExecutor,
    MixedMissionSpec,
    MixedMissionState,
    load_state,
    save_state,
)
from ..task_planner import TaskPlannerError, plan_mission as deterministic_plan
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

MISSION_SERVICE = "mission.service@1"
MOTION_SERVICE = "motion.service@1"
SEMANTIC_SERVICE = "semantic.service@1"
PLANNER_SERVICE = "task.planner@1"

MISSION_TOOLS = (
    "mission_plan",
    "mission_run",
    "mission_status",
    "mission_pause",
    "mission_resume",
    "mission_cancel",
)

DEFAULT_ARTIFACTS = Path("outputs") / "mixed_missions"
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


def _optional_service(ctx: PluginContext, contract: str) -> Any:
    getter = getattr(ctx, "optional_service", None)
    if callable(getter):
        try:
            return getter(contract, None)
        except Exception:  # noqa: BLE001 - optional means optional
            return None
    try:
        return ctx.services.get(contract)
    except Exception:  # noqa: BLE001 - missing optional dependency
        return None


def _safe_id(value: str, fallback: str = "mission") -> str:
    cleaned = _SAFE_ID.sub("_", str(value or "")).strip("._")
    return cleaned or fallback


def _evidence(state: MixedMissionState) -> list[EvidenceRef]:
    refs: list[EvidenceRef] = []
    for index, record in enumerate(state.steps):
        artifacts = record.evidence.get("artifacts")
        if not isinstance(artifacts, (list, tuple)):
            artifact = record.evidence.get("artifact") or record.artifact
            artifacts = [artifact] if artifact else []
        for artifact in artifacts:
            if artifact:
                refs.append(EvidenceRef(
                    kind="image",
                    ref=str(artifact),
                    note=f"mission {state.mission_id} step {index} ({record.kind})",
                ))
    return refs


def _result(
    state: MixedMissionState, state_path: str | Path | None = None
) -> ToolResult:
    if state.status == MIXED_COMPLETED:
        status = STATUS_OK
    elif state.status in (MIXED_CANCELLED, MIXED_BLOCKED):
        status = STATUS_BLOCKED
    elif state.status == MIXED_FAILED:
        status = STATUS_FAILED
    else:  # pending / running / paused
        status = STATUS_EMPTY
    data = state.to_dict()
    data["state_path"] = str(state_path) if state_path else ""
    data["artifacts"] = [
        ref.ref for ref in _evidence(state)
    ]
    lines = [
        f"[{status}] mission {state.mission_id}: "
        f"{state.completed_steps()}/{len(state.steps)} steps, {state.status}, "
        f"clock={state.clock_s:.2f}s"
    ]
    if state.subgoal:
        lines.append(f"  subgoal: {state.subgoal}")
    for index, record in enumerate(state.steps):
        detail = ""
        if record.motion_state:
            motion_steps = record.motion_state.get("steps") or []
            if motion_steps:
                progress = motion_steps[0].get("progress") or {}
                if record.kind == "move":
                    detail = (
                        f"{progress.get('moved_m', 0):.2f}/"
                        f"{progress.get('requested_m', 0):.2f}m"
                    )
                elif record.kind == "turn":
                    detail = (
                        f"{progress.get('turned_deg', 0):.1f}/"
                        f"{progress.get('requested_deg', 0):.1f}deg"
                    )
                elif record.kind == "hover":
                    detail = (
                        f"{progress.get('elapsed_s', 0):.2f}/"
                        f"{progress.get('requested_s', 0):.2f}s"
                    )
        if record.evidence.get("candidate_id"):
            detail = f"candidate={record.evidence['candidate_id']}"
        if record.error:
            detail = f"{detail}; error={record.error}".strip("; ")
        lines.append(f"  {index + 1}. {record.kind} [{record.status}] {detail}".rstrip())
    return ToolResult(status, data, _evidence(state), prompt_text="\n".join(lines))


class MixedMissionPlugin(Plugin):
    """Installable mixed-mission executor/planner front-end."""

    manifest = PluginManifest(
        id="mixed.mission",
        version="0.1.0",
        provides=(MISSION_SERVICE, "task.executor@1"),
        requires=(MOTION_SERVICE,),
        optional_requires=(SEMANTIC_SERVICE, PLANNER_SERVICE),
        tools=MISSION_TOOLS,
        data_dirs=("state",),
        cache_dirs=("cache",),
    )

    def __init__(
        self,
        *,
        artifacts_dir: str | Path | None = None,
        max_semantic_retries: int = 2,
    ) -> None:
        self.default_artifacts_dir = (
            Path(artifacts_dir) if artifacts_dir else None
        )
        self.max_semantic_retries = int(max_semantic_retries)
        self._ctx: PluginContext | None = None
        self._executors: list[MixedMissionExecutor] = []

    # -- lifecycle -------------------------------------------------------- #

    def activate(self, ctx: PluginContext) -> None:
        self._ctx = ctx
        ctx.services.provide(self.manifest.id, MISSION_SERVICE, self)
        ctx.services.provide(self.manifest.id, "task.executor@1", self)
        for spec in self._tool_specs():
            ctx.tools.register(spec)
        ctx.activity_probe(self.active_tasks)
        ctx.abort(self.abort)

    def deactivate(self, ctx: PluginContext) -> None:
        self.abort("plugin deactivated")
        self._executors.clear()
        self._ctx = None

    def active_tasks(self) -> int:
        return sum(1 for executor in self._executors if executor.running)

    def abort(self, reason: str = "aborted") -> None:
        for executor in list(self._executors):
            if executor.running:
                try:
                    executor.cancel(reason)
                except Exception:  # noqa: BLE001 - abort is best effort
                    pass

    # -- mission.service@1 contract --------------------------------------- #

    def plan(self, instruction: str, *, default_target: str = "") -> MixedMissionSpec:
        """Plan an instruction through the optional task.planner service."""
        service = (
            _optional_service(self._ctx, PLANNER_SERVICE)
            if self._ctx is not None
            else None
        )
        if service is not None:
            method = getattr(service, "plan_mission", None) or getattr(
                service, "plan", None
            )
            if callable(method):
                if default_target:
                    return method(instruction, default_target=default_target)
                return method(instruction)
        return deterministic_plan(instruction, default_target=default_target)

    def run(
        self,
        ctx: ToolContext,
        mission: MixedMissionSpec | Mapping | list | None = None,
        *,
        steps: list | None = None,
        instruction: str = "",
        mission_id: str = "",
        resume: bool = False,
        state_path: str | Path | None = None,
        artifacts_dir: str | Path | None = None,
        default_target: str = "",
    ) -> MixedMissionState:
        if resume:
            return self.resume(
                ctx, state_path=state_path, artifacts_dir=artifacts_dir
            )
        if mission is not None:
            spec = MixedMissionSpec.from_dict(mission)
        elif steps is not None:
            spec = MixedMissionSpec.from_dict({"steps": steps})
        elif instruction:
            spec = self.plan(instruction, default_target=default_target)
        else:
            raise MixedMissionError(
                "mission_run requires a 'mission' object, a 'steps' array or "
                "an 'instruction'"
            )
        if mission_id or instruction:
            spec = MixedMissionSpec(
                steps=spec.steps,
                mission_id=mission_id or spec.mission_id,
                instruction=instruction or spec.instruction,
            )
        artifacts = self._artifacts_dir(ctx, artifacts_dir)
        path = self._new_state_path(ctx, spec, state_path)
        executor = self._make_executor(
            ctx, artifacts_dir=artifacts, state_path=path
        )
        ctx.extra[MISSION_EXECUTOR_KEY] = executor
        state = executor.run(
            spec,
            mission_id=spec.mission_id,
            instruction=spec.instruction,
        )
        self._remember(ctx, executor, state, path)
        return state

    def resume(
        self,
        ctx: ToolContext,
        *,
        state_path: str | Path | None = None,
        artifacts_dir: str | Path | None = None,
    ) -> MixedMissionState:
        executor = ctx.extra.get(MISSION_EXECUTOR_KEY)
        state: MixedMissionState | None = None
        if state_path:
            state = load_state(state_path)
        elif isinstance(executor, MixedMissionExecutor) and executor.state is not None:
            state = executor.state
        else:
            stored = ctx.extra.get(MISSION_STATE_KEY)
            if isinstance(stored, MixedMissionState):
                state = stored
        if state is None:
            raise MixedMissionError("no mixed mission state to resume")
        path = Path(state_path) if state_path else (
            executor.state_path if isinstance(executor, MixedMissionExecutor) else None
        )
        if path is None:
            path = ctx.extra.get(MISSION_STATE_PATH_KEY)
        if not isinstance(executor, MixedMissionExecutor) or executor.world is not ctx.world:
            executor = self._make_executor(
                ctx,
                artifacts_dir=self._artifacts_dir(ctx, artifacts_dir),
                state_path=path,
            )
            ctx.extra[MISSION_EXECUTOR_KEY] = executor
        if artifacts_dir:
            executor.artifacts_dir = self._artifacts_dir(ctx, artifacts_dir)
        if path:
            executor.state_path = Path(path)
        executor.state = state
        state = executor.resume(state)
        self._remember(ctx, executor, state, executor.state_path)
        return state

    def pause(self, ctx: ToolContext, reason: str = "") -> MixedMissionState:
        executor = ctx.extra.get(MISSION_EXECUTOR_KEY)
        if not isinstance(executor, MixedMissionExecutor):
            executor = self._executor_from_stored_state(ctx)
        if executor is None:
            raise MixedMissionError("no mixed mission state to pause")
        state = executor.pause(reason)
        self._remember(ctx, executor, state, executor.state_path)
        return state

    def cancel(self, ctx: ToolContext, reason: str = "cancelled") -> MixedMissionState:
        executor = ctx.extra.get(MISSION_EXECUTOR_KEY)
        if not isinstance(executor, MixedMissionExecutor):
            executor = self._executor_from_stored_state(ctx)
        if executor is None:
            raise MixedMissionError("no mixed mission state to cancel")
        state = executor.cancel(reason)
        self._remember(ctx, executor, state, executor.state_path)
        return state

    def state(
        self, ctx: ToolContext, *, state_path: str | Path | None = None
    ) -> MixedMissionState | None:
        if state_path:
            state = load_state(state_path)
            ctx.extra[MISSION_STATE_KEY] = state
            ctx.extra[MISSION_STATE_PATH_KEY] = str(state_path)
            return state
        executor = ctx.extra.get(MISSION_EXECUTOR_KEY)
        if isinstance(executor, MixedMissionExecutor) and executor.state is not None:
            return executor.state
        stored = ctx.extra.get(MISSION_STATE_KEY)
        return stored if isinstance(stored, MixedMissionState) else None

    # -- tools ------------------------------------------------------------ #

    def _tool_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="mission_plan",
                description=(
                    "Plan a Chinese/English instruction into a mixed mission "
                    "({schema, mission_id, instruction, steps}). Deterministic "
                    "grammar first; an unrecognized clause is reported."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "instruction": {"type": "string"},
                        "default_target": {"type": "string"},
                    },
                    "required": ["instruction"],
                },
                handler=self._tool_plan,
                side_effects=False,
                cacheable=False,
            ),
            ToolSpec(
                name="mission_run",
                description=(
                    "Execute an ordered mixed mission end to end. Provide a "
                    "'mission' object, a 'steps' array or an 'instruction'. "
                    "Motion steps (move/turn/hover/photo) use the M1 executor; "
                    "semantic steps complete only on candidate_id / arrived / "
                    "accepted / verified evidence. Pause cooperates between "
                    "steps; cancel stops at the next chunk and keeps progress."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "mission": {"type": "object"},
                        "steps": {"type": "array"},
                        "instruction": {"type": "string"},
                        "default_target": {"type": "string"},
                        "mission_id": {"type": "string"},
                        "resume": {"type": "boolean"},
                        "state_path": {"type": "string"},
                        "artifacts_dir": {"type": "string"},
                    },
                },
                handler=self._tool_run,
                side_effects=True,
                cacheable=False,
            ),
            ToolSpec(
                name="mission_status",
                description=(
                    "Read the current (or a stored) mixed mission state: step "
                    "statuses, motion progress, semantic evidence and results."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"state_path": {"type": "string"}},
                },
                handler=self._tool_status,
                side_effects=False,
                cacheable=False,
            ),
            ToolSpec(
                name="mission_pause",
                description=(
                    "Pause the mixed mission between steps. The current step "
                    "finishes; completed work and motion progress are kept."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                },
                handler=self._tool_pause,
                side_effects=True,
                cacheable=False,
            ),
            ToolSpec(
                name="mission_resume",
                description=(
                    "Resume a paused/blocked/cancelled mixed mission from its "
                    "stored progress. Completed steps are never replayed and a "
                    "partially executed motion step resumes its remaining "
                    "distance/angle/time."
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
                cacheable=False,
            ),
            ToolSpec(
                name="mission_cancel",
                description=(
                    "Cancel the mixed mission. The current step is cancelled "
                    "with its measured progress; completed steps stay done and "
                    "resume can continue from the interrupted step."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                },
                handler=self._tool_cancel,
                side_effects=True,
                cacheable=False,
            ),
        ]

    def _tool_plan(self, ctx: ToolContext, args: dict) -> ToolResult:
        instruction = str(args.get("instruction", "") or "")
        default_target = str(args.get("default_target", "") or "")
        try:
            spec = self.plan(instruction, default_target=default_target)
        except (TaskPlannerError, MixedMissionError) as exc:
            return ToolResult(
                STATUS_FAILED, {"error": str(exc), "instruction": instruction}
            )
        data = spec.to_dict()
        lines = [
            f"[ok] mission {spec.mission_id}: {len(spec.steps)} steps",
            f"instruction: {spec.instruction}",
        ]
        for index, step in enumerate(spec.steps):
            detail = ", ".join(
                f"{key}={value!r}" for key, value in step.items() if key != "kind"
            )
            lines.append(f"  {index + 1}. {step['kind']} {detail}".rstrip())
        return ToolResult(STATUS_OK, data, prompt_text="\n".join(lines))

    def _tool_run(self, ctx: ToolContext, args: dict) -> ToolResult:
        try:
            state = self.run(
                ctx,
                args.get("mission"),
                steps=args.get("steps"),
                instruction=str(args.get("instruction", "") or ""),
                mission_id=str(args.get("mission_id", "") or ""),
                resume=bool(args.get("resume")),
                state_path=args.get("state_path"),
                artifacts_dir=args.get("artifacts_dir"),
                default_target=str(args.get("default_target", "") or ""),
            )
            return _result(state, ctx.extra.get(MISSION_STATE_PATH_KEY))
        except Exception as exc:  # noqa: BLE001 - tool boundary reports
            return ToolResult(STATUS_FAILED, {"error": f"{type(exc).__name__}: {exc}"})

    def _tool_status(self, ctx: ToolContext, args: dict) -> ToolResult:
        try:
            state = self.state(ctx, state_path=args.get("state_path"))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(STATUS_FAILED, {"error": f"{type(exc).__name__}: {exc}"})
        if state is None:
            return ToolResult(STATUS_EMPTY, {"reason": "no mixed mission state"})
        return _result(state, ctx.extra.get(MISSION_STATE_PATH_KEY))

    def _tool_pause(self, ctx: ToolContext, args: dict) -> ToolResult:
        try:
            state = self.pause(ctx, str(args.get("reason", "") or ""))
            return _result(state, ctx.extra.get(MISSION_STATE_PATH_KEY))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(STATUS_FAILED, {"error": f"{type(exc).__name__}: {exc}"})

    def _tool_resume(self, ctx: ToolContext, args: dict) -> ToolResult:
        try:
            state = self.resume(
                ctx,
                state_path=args.get("state_path"),
                artifacts_dir=args.get("artifacts_dir"),
            )
            return _result(state, ctx.extra.get(MISSION_STATE_PATH_KEY))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(STATUS_FAILED, {"error": f"{type(exc).__name__}: {exc}"})

    def _tool_cancel(self, ctx: ToolContext, args: dict) -> ToolResult:
        try:
            state = self.cancel(
                ctx, str(args.get("reason", "cancelled") or "cancelled")
            )
            return _result(state, ctx.extra.get(MISSION_STATE_PATH_KEY))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(STATUS_FAILED, {"error": f"{type(exc).__name__}: {exc}"})

    # -- internals -------------------------------------------------------- #

    def _artifacts_dir(
        self, ctx: ToolContext, artifacts_dir: str | Path | None = None
    ) -> Path:
        directory = Path(
            artifacts_dir
            or ctx.extra.get(MISSION_ARTIFACTS_KEY)
            or ctx.extra.get("motion.artifacts_dir")
            or self.default_artifacts_dir
            or DEFAULT_ARTIFACTS
        )
        ctx.extra[MISSION_ARTIFACTS_KEY] = str(directory)
        return directory

    def _new_state_path(
        self,
        ctx: ToolContext,
        spec: MixedMissionSpec,
        state_path: str | Path | None,
    ) -> Path:
        if state_path:
            return Path(state_path)
        if self._ctx is not None:
            try:
                return self._ctx.data_path("state") / (
                    f"{_safe_id(spec.mission_id)}.json"
                )
            except Exception:  # noqa: BLE001 - fall back to the artifacts dir
                pass
        return self._artifacts_dir(ctx) / f"{_safe_id(spec.mission_id)}.json"

    def _make_executor(
        self,
        ctx: ToolContext,
        *,
        artifacts_dir: Path,
        state_path: str | Path | None,
    ) -> MixedMissionExecutor:
        semantic = self._semantic_service(ctx)
        registry = None
        if self._ctx is not None:
            registry = getattr(self._ctx.host, "tools", None)
        chunk_kwargs: dict[str, Any] = {}
        motion_service = (
            _optional_service(self._ctx, MOTION_SERVICE)
            if self._ctx is not None
            else None
        )
        for name in ("chunk_m", "chunk_turn_deg", "hover_dt_s"):
            value = getattr(motion_service, name, None)
            if isinstance(value, (int, float)):
                chunk_kwargs[name] = float(value)
        executor = MixedMissionExecutor(
            ctx.world,
            registry=registry,
            ctx=ctx,
            semantic_service=semantic,
            artifacts_dir=artifacts_dir,
            max_semantic_retries=self.max_semantic_retries,
            state_path=state_path,
            **chunk_kwargs,
        )
        # Keep only executors that are still running plus the new one; the
        # probe/abort hooks must not accumulate every finished mission.
        self._executors = [item for item in self._executors if item.running]
        self._executors.append(executor)
        return executor

    def _semantic_service(self, ctx: ToolContext) -> Any:
        explicit = ctx.extra.get(MISSION_SEMANTIC_KEY)
        if explicit is not None:
            return explicit
        if self._ctx is None:
            return None
        return _optional_service(self._ctx, SEMANTIC_SERVICE)

    def _executor_from_stored_state(
        self, ctx: ToolContext
    ) -> MixedMissionExecutor | None:
        state = ctx.extra.get(MISSION_STATE_KEY)
        path = ctx.extra.get(MISSION_STATE_PATH_KEY)
        if state is None and path:
            try:
                state = load_state(path)
            except MixedMissionError:
                state = None
        if not isinstance(state, MixedMissionState):
            return None
        executor = self._make_executor(
            ctx,
            artifacts_dir=self._artifacts_dir(ctx),
            state_path=path,
        )
        executor.state = state
        ctx.extra[MISSION_EXECUTOR_KEY] = executor
        return executor

    def _remember(
        self,
        ctx: ToolContext,
        executor: MixedMissionExecutor,
        state: MixedMissionState,
        state_path: str | Path | None,
    ) -> None:
        ctx.extra[MISSION_STATE_KEY] = state
        path = Path(state_path) if state_path else executor.state_path
        if path is not None:
            executor.state_path = path
            ctx.extra[MISSION_STATE_PATH_KEY] = str(path)
            try:
                save_state(state, path)
            except OSError as exc:
                executor.persist_error = f"{type(exc).__name__}: {exc}"
        ctx.extra[MISSION_EXECUTOR_KEY] = executor
