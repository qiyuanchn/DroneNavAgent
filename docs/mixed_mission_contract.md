# Mixed mission / M2-M4 implementation contract

Baseline commit: `13cab5f` (M0 plugin host + M1 motion plugin).

This document is the interface contract for parallel work on `goal.md`. Each
workstream owns the files listed for it. Do not edit files owned by another
workstream. Do not commit. Run only the tests for your workstream (the full
suite takes many minutes because of subprocess comparison drivers).

Existing pieces to build on:

- `vlm_nav/motion.py`: `MoveStep`, `TurnStep`, `HoverStep`, `PhotoStep`,
  `MissionSpec`, `MissionState`, `MotionExecutor`, `SimClock`, `VirtualClock`.
- `vlm_nav/plugins/__init__.py`: `PluginHost`, `PluginManifest`, `Plugin`,
  `PluginContext`, `ServiceRegistry`, `ResourceManager`, `LifecycleError`.
- `vlm_nav/plugins/motion.py`: `MotionPlugin`, `MotionService`,
  `motion.service@1`, 8 `motion_*` tools.
- `vlm_nav/tools.py`: `ToolRegistry.call(name, args, ctx)`,
  `ToolSpec`, `ToolContext`, `ToolResult`, scope stamping, evidence pack.
- Semantic tools (registered by the existing modules): `search_target`,
  `approach_candidate`, `confirm_candidate`, `verify_goal`,
  `go_to_viewpoint`, `return_to_observation`, `propose_viewpoints`,
  `observe`, `ground_target`, `finish`.

## File ownership

| Workstream | Owned files |
|---|---|
| A: mixed mission + planner + plugin/script | `vlm_nav/mixed_mission.py`, `vlm_nav/task_planner.py`, `vlm_nav/plugins/mixed.py`, `vlm_nav/plugins/task_planner.py`, `scripts/run_mixed_mission.py`, `tests/test_mixed_mission.py`, `tests/test_task_planner.py`, `tests/test_mixed_mission_plugin.py` |
| C: semantic service/plugin | `vlm_nav/semantic.py`, `vlm_nav/plugins/semantic.py`, `tests/test_semantic_service.py`, `tests/test_semantic_plugin.py` |
| D: host extensions, profiles, legacy plugins | `vlm_nav/plugins/__init__.py`, `vlm_nav/plugins/profiles.py`, `vlm_nav/plugins/legacy.py`, `tests/test_plugin_profiles.py`, `tests/test_legacy_plugins.py` |
| E: managed worker | `vlm_nav/plugins/worker.py`, `vlm_nav/worker_echo.py`, `tests/test_worker_isolation.py` |
| F1: hardware profile | `vlm_nav/hardware.py`, `scripts/check_hardware_profile.py`, `tests/test_hardware.py` |
| F2 (second wave): M4 driver | `scripts/run_mixed_missions.py`, `data/mixed_missions/`, `tests/test_run_mixed_missions.py` |
| Parent | README/docs, integration fixes, final tests, commit |

The parent may make small integration edits anywhere after the wave. Never
commit; the parent owns `git`.

## M2: mixed mission schema

`vlm_nav/mixed_mission.py` defines schema `drone.mixed_mission/1`.

Motion step dictionaries reuse the M1 schema exactly:

```json
{"kind": "move", "direction": "forward", "meters": 5.0}
{"kind": "turn", "direction": "right", "degrees": 90}
{"kind": "hover", "seconds": 3.0}
{"kind": "photo", "label": "window_chair"}
```

Semantic/control step dictionaries (all optional keys default sensibly):

```json
{"kind": "search", "target": "窗边的木椅", "relation": "窗边", "max_steps": 8}
{"kind": "approach", "target": "last", "standoff_m": 1.2, "max_legs": 8}
{"kind": "confirm", "target": "last", "max_views": 2}
{"kind": "verify", "target": "last", "success_dist_m": 1.2}
{"kind": "revisit", "place": "kitchen", "position": [1.0, 2.0, 1.0]}
{"kind": "goto", "position": [1.0, 2.0, 1.0]}
{"kind": "takeoff"}, {"kind": "land"}, {"kind": "orbit"}, {"kind": "patrol"}
```

- `target` for approach/confirm/verify: `"last"` means the candidate selected
  by the most recent search/ground step; otherwise a `cand_N` id.
- `takeoff/land/orbit/patrol` are capability-gated: call the world method of
  the same name if it exists; otherwise fail the step with
  `unsupported_capability:<kind>`. Never silently turn them into something else.
- `revisit` may specify `position`, `place` (resolved from
  `ctx.extra["places"]` / memory if available), `observation_id`, or
  `viewpoint_id`.

`MixedMissionSpec.from_dict` accepts `{steps, mission_id, instruction,
schema}` (or a bare list of steps). `to_dict` round-trips. Validation raises
`MixedMissionError` for unknown kinds or missing required fields.

`MixedMissionState` schema `drone.mixed_mission.state/1`:

```json
{
  "schema": "drone.mixed_mission.state/1",
  "mission_id": "mission_x",
  "status": "pending|running|paused|completed|cancelled|blocked|failed",
  "current_step": 0,
  "instruction": "...",
  "clock_s": 0.0,
  "subgoal": "step 2/5: ...",
  "error": "",
  "steps": [
    {"kind": "move", "params": {...}, "status": "done", "attempts": 1,
     "motion_state": {...}, "evidence": {...}, "error": ""}
  ]
}
```

`MixedMissionExecutor`:

```python
class MixedMissionExecutor:
    def __init__(self, world, *, registry=None, ctx=None,
                 motion_executor=None, semantic_service=None,
                 clock=None, artifacts_dir=None, max_semantic_retries=2,
                 on_progress=None, state_path=None): ...
    def run(self, mission: MixedMissionSpec | Mapping, *,
            mission_id="", instruction="") -> MixedMissionState: ...
    def resume(self, state: MixedMissionState | None = None) -> MixedMissionState: ...
    def pause(self, reason="") -> MixedMissionState: ...
    def cancel(self, reason="cancelled") -> MixedMissionState: ...
    @property
    def state(self) -> MixedMissionState | None: ...
```

Behavior:

- Execute steps strictly in order. A failed/blocked step stops the mission;
  the status is `blocked` or `failed`; remaining steps stay `pending`.
- A motion step is executed by a per-step `MissionSpec` on
  `MotionExecutor`, and the resulting `MissionState.to_dict()` is stored in
  `step["motion_state"]`. On resume, a partially done motion step calls
  `MotionExecutor.resume(motion_state)`; completed motion steps are never
  replayed.
- A semantic step calls `SemanticNavigationService` (see below). `search`
  is done when it returns a candidate id (even if unconfirmed);
  `approach` is done only when `arrived is True`; `confirm` only when
  `accepted is True`; `verify` only when `verified is True`; `revisit/goto`
  only when the movement reports `arrived`. Anything else must be marked
  blocked/failed with the tool's reason. Never set `done` from
  `progress_kind` or a free-text claim.
- After every step update `ctx.extra["subgoal"]` to
  `"mission <id> step <i>/<n>: <short label> (<remaining> left)"` (<=160
  chars) and `ctx.extra["mission.progress"]` to the state summary dict.
  The runtime reads `subgoal` already; do not touch `STATE_LABELS`.
- `pause` is cooperative between steps (motion is paused only at step
  boundaries; use `cancel` for a mid-motion stop). `resume` reopens the
  current paused/blocked step but keeps measured progress in
  `motion_state`/step progress.
- Serialization: `save_state`/`load_state` helpers; `state_path` persists
  after every step.
- Completion: `status == "completed"` only when every step is `done`.
  A mixed mission that includes a semantic target must not be completed
  from motion alone.

## M2: semantic service API (`vlm_nav/semantic.py`, workstream C)

```python
class SemanticNavigationService:
    def __init__(self, registry, *, max_retries=2): ...
    def search(self, ctx, target, *, relation="", instruction=None,
               max_steps=8, max_turns=1) -> dict: ...
    def approach(self, ctx, candidate_id=None, *, standoff_m=1.2,
                 max_legs=8) -> dict: ...
    def confirm(self, ctx, candidate_id=None, *, instruction=None,
                relation=None, max_views=2) -> dict: ...
    def verify(self, ctx, candidate_id=None, *, success_dist_m=None) -> dict: ...
    def revisit(self, ctx, *, position=None, observation_id=None,
                viewpoint_id=None, standoff_m=0.6) -> dict: ...
    def goto(self, ctx, position, *, standoff_m=0.6) -> dict: ...
```

Every method returns a dict with at least:

```
{"ok": bool, "tool": str, "status": str, "candidate_id": str,
 "task_id": str, "arrived": bool, "accepted": bool, "verified": bool,
 "calls": int, "error": str, "data": {...}}
```

Rules:

- Call only through `registry.call(name, args, ctx)`, never handlers.
- `search`: `search_target` with `instruction` (compose relation/target).
  `ok = bool(candidate_id)`. Follow `next_action`/`strategy_change` when it
  names a registered tool, bounded by `max_retries`; then retry search with
  the original instruction. Never accept a candidate when the result says
  `confirmed=false` as identity-confirmed (still return the id for approach).
- `approach`: call `approach_candidate` with `candidate_id`; continue with
  the returned `task_id` while `task_status == "running"`. On blocked,
  follow `next_action` (usually `propose_viewpoints`/`go_to_viewpoint`)
  once, then retry approach. `ok = arrived`.
- `confirm`: call `confirm_candidate`; `ok = verdict == "accepted"`. Preserve
  `uncertain`/`rejected` in the result. Do not retry a rejected candidate.
- `verify`: call `verify_goal`; `ok = data["verified"] is True`. Preserve
  `"uncertain"` as not ok.
- `revisit/goto`: use `go_to_viewpoint` (position via
  `propose_viewpoints`) or `return_to_observation`; loop the returned
  `task_id` until arrived; `ok = arrived`.
- `calls` counts every `registry.call`.
- Missing registry/candidate/world is a structured `ok=False` with error,
  not an exception.

`vlm_nav/plugins/semantic.py` defines `SemanticNavigationPlugin`:

- manifest id `"semantic.navigation"`, version `"0.1.0"`,
  provides `("semantic.service@1",)`,
  requires `("navigation.tools@1", "perception.tools@1")`,
  tools `("semantic_status",)` (read-only; returns the service contract).
- activate: provide `SemanticNavigationService(ctx.host.tools)`, register
  `semantic_status`.
- Services are stateless; methods take `ctx`.

## M2: planner API (`vlm_nav/task_planner.py`, workstream A)

```python
def plan_mission(instruction: str, *, vlm=None,
                 default_target: str = "") -> MixedMissionSpec: ...
```

- Deterministic Chinese/English grammar first. It must parse the goal.md
  example:
  `上升 1 米，向前飞 5 米，悬停 3 秒，右转 90 度，找到窗边的木椅，飞到它旁边，再向左移动 1 米并拍照。`
  and English equivalents.
- Supported vocabulary: 上升/下降/向前/向前飞/后退/向左/向右/向前移动/
  forward/back/left/right/up/down + meters; 悬停/hover + seconds;
  左转/右转/turn left|right + degrees; 找到/寻找/find/search/look for
  `<target>` (optionally `near/next to/旁边的 <anchor>`); 飞到它旁边/
  approach/fly to it; 确认/confirm; 验证/verify; 返回/回到/revisit;
  拍照/take a photo/snapshot.
- Multiple clauses split on `，,。;；然后/再/and then` (keep `and` inside
  "and take a photo" working).
- Unrecognized clause: if `vlm` is provided, ask it once for a strict JSON
  mission (`{"steps":[...]}`) and validate with `MixedMissionSpec.from_dict`;
  otherwise raise `TaskPlannerError` naming the clause. Never guess silently.
- `plugins/task_planner.py` defines `TaskPlannerPlugin`:
  manifest id `"task.planner"`, version `"0.1.0"`,
  provides `("task.planner@1",)`, tools `("plan_mission",)`.
  activate: provide `plan_mission` wrapper service; register
  `plan_mission` tool (`{"instruction": str}`, returns mission dict).
- `plugins/mixed.py` defines `MixedMissionPlugin`:
  manifest id `"mixed.mission"`, version `"0.1.0"`,
  provides `("mission.service@1",)`,
  requires `("motion.service@1",)`,
  optional_requires `("semantic.service@1", "task.planner@1")`,
  tools `("mission_plan", "mission_run", "mission_status",
          "mission_pause", "mission_resume", "mission_cancel")`,
  data_dirs `("state",)`, cache_dirs `("cache",)`.
  `mission_run` args: `mission` object or `steps` array or `instruction`,
  optional `mission_id`, `resume`, `state_path`, `artifacts_dir`.
  Side effects true, cacheable false. The plugin resolves services through
  `ctx.services.get(..., optional)`; a semantic step with no service must
  return a structured failure, not crash.
- `scripts/run_mixed_mission.py`: CLI analogous to
  `scripts/run_motion_mission.py`. Default instruction is the goal.md
  example. Support `--motion-only` (no semantic tools), `--semantic mock`
  (deterministic fake service for smoke), and `--semantic real` (install
  the legacy/semantic plugins if importable). Always exercise
  install/activate/run/deactivate/uninstall and print a summary.

## D: host additions

`vlm_nav/plugins/__init__.py`:

- `PluginContext.registry_view()` returns a `RegistryView` with `.register`,
  `.names`, `.get`, `.call`, `.batch`; `.register` forwards to
  `ctx.tools.register`; the rest forwards to `ctx.host.tools`. Existing
  `register_*` functions can be passed this view unchanged.
- `PluginManifest.validate` accepts `isolation="worker"` (do not reject it).
- Add `PluginContext.optional_service(contract, default=None)` convenience.
- Add `PluginHost.install_bundle(plugins)` that installs all manifests first
  (dependency order not required), then returns reports; activation stays
  explicit.
- Do not change existing lifecycle semantics or tests.

`vlm_nav/plugins/profiles.py`:

```python
@dataclass(frozen=True)
class Profile:
    id: str
    plugins: tuple[str, ...]
    context_defaults: dict = field(default_factory=dict)
    description: str = ""

BUILTIN_PROFILES = {
  "motion-only": Profile("motion-only", ("motion.base",)),
  "semantic-nav": Profile("semantic-nav",
      ("model.provider", "perception.core", "navigation.core",
       "task.tools", "semantic.navigation")),
  "mixed-mission": Profile("mixed-mission",
      ("motion.base", "task.planner", "perception.core",
       "navigation.core", "semantic.navigation", "mixed.mission")),
  "4090-local": Profile("4090-local", (...same as mixed-mission...,
      context_defaults={"perception.detector": "sam3_dino"})),
}
def install_bundle(host, plugins: Mapping[str, Plugin]) -> list[LifecycleReport]: ...
def activate_profile(host, profile: Profile,
                     factories: Mapping[str, Callable[[], Plugin]]) -> dict: ...
```

`activate_profile` installs missing plugins from `factories`, then activates
in profile order, returns `{"profile", "active", "reports", "catalog"}`.

`vlm_nav/plugins/legacy.py` adapter plugins (all use
`ctx.registry_view()` and declare exactly the tools the register function
adds):

- `PerceptionPlugin` id `"perception.core"`, provides
  `"perception.tools@1"`, tools `observe, inspect_region, ground_target,
  compare_candidates, query_memory, search_target`; calls
  `register_perception_tools(view, vlm=optional model provider)`.
- `NavigationPlugin` id `"navigation.core"`, provides
  `"navigation.tools@1"`, tools `propose_viewpoints, check_path,
  approach_candidate, go_to_viewpoint, return_to_observation, navigate,
  task_status, cancel_task, verify_goal, finish, batch_read,
  confirm_candidate`; calls `register_navigation_tools(view, vlm=...)`.
- `TaskToolsPlugin` id `"task.tools"`, tools `update_plan, get_plan`;
  calls `register_task_tools(view)`.
- `MemoryStorePlugin` id `"memory.store"`, provides `"memory.store@1"`;
  accepts a `factory` returning a memory object; closes it on deactivate if
  it has `close`.
- `ModelProviderPlugin` id `"model.provider"`, provides
  `"model.provider@1"`; accepts a `factory`; optional `close`.
- Do not register `semantic.navigation` here; workstream C owns it.

## E: managed worker

`vlm_nav/plugins/worker.py`:

```python
@dataclass(frozen=True)
class WorkerSpec:
    command: tuple[str, ...]
    cwd: str | None = None
    env: Mapping[str, str] | None = None
    startup_timeout_s: float = 10.0
    request_timeout_s: float = 30.0
    stop_timeout_s: float = 5.0
    restart: bool = False

class ManagedWorker:
    def start(self); def call(self, method, params=None, timeout=None) -> Any
    @property pid/state/alive; def stop(self, timeout=None) -> dict
    def logs(self) -> str
```

- Spawn with `subprocess.Popen(..., stdin=PIPE, stdout=PIPE,
  stderr=logfile, start_new_session=True, text=True, bufsize=1)`.
- JSON-lines protocol: request `{"id": n, "method": "...", "params": {}}`,
  reply `{"id": n, "ok": true, "result": ...}` or `{"id": n, "ok": false,
  "error": "..."}`. Handshake `{"method":"ping"}`.
- `stop`: send `shutdown`, close stdin, wait, `terminate` process group,
  wait stop_timeout, `kill` process group, reap; report
  `{"stopped": bool, "pid": int, "forced": bool, "returncode": ...}`.
- Capture stderr in `cache/worker.log` when a path is given.
- `WorkerPlugin(Plugin)` base: manifest `background_tasks=("worker",)`,
  `isolation="inproc"` (the host is in-process; the heavy work is in the
  managed child). `activate` starts the worker, registers
  `ctx.tasks.register("worker", handle.stop)`, and calls
  `configure(worker, ctx)`; `deactivate` stops it. If `configure` fails,
  stop the worker before re-raising.
- `vlm_nav/worker_echo.py` is a JSON-line worker with `ping`, `echo`,
  `sleep`, `shutdown`; used by tests with `sys.executable -m
  vlm_nav.worker_echo`.
- Tests: start/call/stop, pid gone after stop, worker plugin
  activate/deactivate/uninstall leaves no process, forced kill after a
  hung request, restart policy optional.

## F1: hardware profile

`vlm_nav/hardware.py`:

```python
@dataclass(frozen=True)
class HardwareProfile:
    id: str
    total_mib: int
    reserve_mib: int
    expected_peak_mib: int
    description: str = ""

PROFILES = {"a100-40gb": ..., "4090-24gb": ..., "4090-local": ...}

def snapshot(device="cuda") -> dict:
    # always returns available/gpu_name/total_mib/free_mib/
    # allocated_mib/reserved_mib/peak_allocated_mib/peak_reserved_mib/
    # driver_used_mib; available=False when torch/CUDA is missing
def reset_peak(device="cuda") -> None
def fit_report(profile, snap, components=None) -> dict:
    # fits, required_mib, headroom_mib, total_mib, reserve_mib,
    # reasons[], components
def markdown_table(report) -> str
```

`scripts/check_hardware_profile.py --profile 4090-local --device cuda
--out ...` prints a JSON/Markdown fit report and exits 0/1; with no CUDA it
must say `available: false` and exit 2 (not pretend to pass).

## F2: M4 driver (second wave)

`scripts/run_mixed_missions.py` (owns `data/mixed_missions/`):

- Groups: `action`, `semantic`, `mixed`.
- Stage counts: `2,4,8` from JSON task files; ≥2 seeds.
- Action group: motion-only `MixedMissionSpec` chains (move/turn/hover/
  photo) with 2/4/8 steps.
- Semantic group: search/approach/confirm/verify chains with 2/4/8 stages.
  `--semantic mock` uses a deterministic `MockSemanticService` that moves
  the world and records evidence per stage; `--semantic real` builds the
  plugin stack (import errors are reported, not swallowed).
- Mixed group: alternating motion and semantic stages.
- Interrupt/resume: `--interrupt-step N` stops after N completed steps
  (state saved), then `--resume` continues; verify completed steps are not
  replayed and motion remaining distance is preserved. Report
  `resume_recovered`, `replayed_steps`.
- Metrics per group/stage count: `success_rate`, `false_completion`,
  `completed`, `episodes`, `tool_calls`, `vlm_calls`, `wall_time_s`,
  `peak_allocated_mib`/`peak_reserved_mib` (via `vlm_nav.hardware`),
  `resources_after_uninstall` (plugin catalog residual / worker pids).
- Output `report.json` + `report.md`; schema
  `vlm_nav.mixed_missions.v1`.
- Tests run 2-stage mock only (fast); 4/8 are available via CLI.
- False completion: a mission that reports success while a semantic step
  lacks accepted/verified evidence counts in `false_completion`; the
  executor must prevent this, and the driver re-checks the state.

## Testing rules

- Workstream tests must be runnable individually:
  `python -m pytest tests/test_<yours>.py -q`.
- Use `tempfile.TemporaryDirectory`; never write into the repo except
  `outputs/` (gitignored).
- No network/model/GPU dependency: abstract or mock; hardware checks must
  degrade honestly when CUDA is unavailable.
- Keep existing tests green. Do not change pinned lists/order in
  `tests/test_vlm_nav_tools.py`, `tests/test_tools_runtime.py`,
  `tests/test_tools_mcp.py`, `tests/test_motion_mission.py` unless your
  workstream explicitly owns the file.
- Report files changed, tests run/results, and integration caveats.

## Implementation status (2026-10-20)

- M2: `vlm_nav/mixed_mission.py`, `vlm_nav/task_planner.py`,
  `vlm_nav/plugins/{task_planner,mixed}.py`, `scripts/run_mixed_mission.py`;
  semantic service in `vlm_nav/semantic.py` + `plugins/semantic.py`.
- M3: host `registry_view()`/`install_bundle()`, `plugins/profiles.py`,
  `plugins/legacy.py`, `plugins/worker.py`, `plugins/worker_model.py`
  (managed worker + worker-backed `model.provider@1`), and
  `plugins/infra.py` for resource-manager/world-service/mcp-adapter.
- M4: `vlm_nav/hardware.py`, `scripts/check_hardware_profile.py`,
  `scripts/run_mixed_missions.py`, `data/mixed_missions/tasks_{2,4,8}.json`.
- The default M4 semantic backend is a deterministic mock service; `--semantic
  real` installs the actual plugin stack and reports missing model/backend
  honestly. A real run needs a VLM and, for GPU numbers, `--device cuda`.
- Mid-motion cancellation/resume is covered by
  `tests/test_mixed_mission.py`; the M4 driver's `--interrupt-step` covers
  task-boundary interruption.
- `4090-local` uses `model.provider.worker` with 4-bit quantization and the
  local detector; model paths/commands come from the profile factories.
