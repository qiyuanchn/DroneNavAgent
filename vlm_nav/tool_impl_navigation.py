"""Navigation tools for the agent runtime (goal.md M3).

``register_navigation_tools(registry)`` adds the navigation contract. Movement
is split into one tool per *intent*, so "go back to where that frame was shot"
can never be confused with "go to the object in that frame":

- ``approach_candidate`` ``{candidate_id, standoff_m?, max_legs?}``: fly to a
  target *candidate* (``cand_N`` from ``ground_target``). The geometry lives in
  ``vlm_nav/skills_approach.py``: locate the candidate, end the leg on the
  standoff shell (``world._apply_standoff``), re-observe and re-localize after
  every segment. The model decides who to approach and when to give up.
- ``go_to_viewpoint`` ``{viewpoint_id, ...}``: fly to an observation *position*
  (``vp_N`` from ``propose_viewpoints``).
- ``return_to_observation`` ``{observation_id, restore_yaw=true}``: fly back to
  the *camera pose* a frame was captured at and restore its heading - never the
  object in it.
- ``navigate``: deprecated mixed-semantics tool; every call answers with an
  explicit error naming the three intent tools (and the exact replacement call
  for the arguments it was given), never with a movement.
- ``propose_viewpoints``: map-aware observation points (``vp_N``) with an
  expected coverage and a straight-line cost, stored in ``ctx.extra``.
- ``check_path``: light reachability estimate for one straight leg - obstacle
  clearance, path cost and the ratio of the leg that carries no observation
  evidence yet.
- ``task_status``: progress, budgets and the execution events of one task (or of
  every task when no ``task_id`` is given). A second query with no state change
  in between returns ``cached`` plus a hint instead of the same payload again.
- ``cancel_task``: interrupt a running task (``status=cancelled``); a task that
  is already terminal reports ``cached``.
- ``verify_goal``: independent completion check against ``world.target`` using
  ``world.depth_at_pixel`` / ``estimate_point_world``. The distance check is
  taken against the *caller's* gate when the context carries one
  (``ctx.extra["done_dist"]``: what the runner judges the episode with, i.e.
  ``--success-dist`` measured from the target surface, plus the target radius);
  its own ``max(1.5, radius + 0.5)`` default only fills the gap. The effective
  limit is reported in ``checks.distance_limit_m``. This is the one tool that
  may read ``world.target`` - and its answer reports only its own checks (FOV,
  the gate, the measured point, the recalled pose). The goal's true pose is
  never part of the return or of its evidence refs: the completion check must
  not become the shortcut past the visual evidence the episode is testing.
- ``finish``: end the episode with an explicit outcome. The claim is judged on
  the episode's *evidence pack* (``ctx.extra["evidence_pack"]``, written by this
  module's ``verify_goal``/arrivals and, for candidate identity, by the arrival
  of an approach): ``"success": true`` without a passing verification or a
  grounded candidate plus an arrival comes back ``success: false`` with
  ``claim=unverified`` and the list of what is missing, so the runtime's verdict
  follows the trail instead of the model's word (goal.md P4). ``"success":
  false`` is the honest give-up and a bare finish claims nothing.
- ``batch_read``: one call that runs several *read-only* checks (observe +
  query_memory, a compare and an inspect) and returns their merged results. A
  pose-changing tool is refused inside a batch - movement stays serial, one leg
  and one ``task_id`` per call.

Progress contract of every movement tool (goal.md section 5)
------------------------------------------------------------
One task, at most one leg per call - the call returns immediately and never
blocks until the whole journey is over. The payload carries ``task_id``, the
task lifecycle in ``task_status`` (``running|arrived|blocked|failed|cancelled``)
and the *progress* verdict in ``status``:

- ``ok``: the leg progressed or arrived (``outcome`` ``arrived`` / ``partial``).
- ``blocked``: known geometry or the world refused the leg *after* it moved
  (``collision`` / ``stalled`` / ``planner_failed`` / ``obstacle_on_path``).
- ``no_progress``: the call changed nothing - zero displacement
  (``already_at_pose`` / ``stalled`` / ``leg_timeout`` / ``outside_bound``).

A zero-displacement call is **never** reported as ``ok`` + ``arrived``: asking
to fly somewhere the drone already is answers ``status=no_progress``, an
explicit ``outcome=already_at_pose`` and a next step that is *different* from
the call that just did nothing (the "navigate to the current frame's own obs id
-> already_there -> repeat" loop this contract exists to break). Fields:
``progress_m``, ``turn_deg``, ``remaining_m``, ``blocked_reason``, ``task_id``,
``candidate_id`` / ``viewpoint_id`` / ``observation_id``, ``last_observation_id``.

A *partial* leg is not automatically progress: when the same (intent, target)
has just produced two legs that closed less than ``REPEAT_MOVE_MIN_PROGRESS_M``,
the next call is answered ``blocked`` / ``repeating_move`` before it flies -
the third consecutive fruitless attempt is refused with a ``strategy_change``
(change the target/viewpoint, ``verify_goal``, re-ground, or take a fresh view)
and a different ``next_action``. ``task_status`` reports the same task as
``stuck`` with the streak. This is the seed-7 resume loop: 13
``approach_candidate`` calls that kept answering ``ok`` + ``partial`` without a
range until the episode budget was gone.

Execution-layer events (goal.md section 5)
------------------------------------------
Each segment appends structured events to ``ctx.extra["events"]`` - the channel
``AgentRuntime`` drains after every tool call and re-emits through its
``on_event`` hooks, so an event wakes the model loop: ``arrived``,
``path_blocked`` (blocked leg or blocked pre-check), ``map_changed`` (the
observed map grew significantly), ``target_rediscovered`` (a False -> True
target-in-FOV transition, reported without target geometry - ``verify_goal``
stays the only tool that reads the goal's pose), ``task_failed`` (timeout /
segment budget / world error) and ``task_cancelled``. An event record is flat
(``{seq, event, task_id, status, progress_m, remaining_m, reason, ...}``) and is
mirrored as ``{"name": ..., "data": {...}}`` for the runtime drain; every event
is kept on its task record and in the bounded ``ctx.extra["nav_events"]`` log.

Task lifecycle and budgets
--------------------------
Task bookkeeping lives in ``ctx.extra``:

- ``nav_tasks``: ``nav_N`` -> full task record (status, reason, intent, target,
  target key, progress/remaining/moved, segments, ``max_segments``,
  ``timeout_s`` / ``deadline``, ``map_points`` / ``map_version``,
  ``target_in_fov``, events, candidate viewpoints, query bookkeeping, last
  result for cached replays)
- ``nav_events``: bounded, sequence-ordered log of every navigation event
- ``events``: the runtime-drained event queue (see above)
- ``map_version``: incremented whenever a task reports a significant map change
- ``nav_timeout_s`` / ``nav_max_segments`` / ``max_leg_m``: budget knobs, also
  settable per call through the ``timeout_s`` / ``max_segments`` args

A task that exceeds ``timeout_s`` or ``max_segments`` fails with an explicit
``timeout`` / ``segment_budget`` reason, a leg that carries the drone nowhere
reports ``no_progress`` (``leg_timeout`` / ``outside_bound`` / ``stalled``), and
a leg whose pre-check hits known geometry is ``blocked`` with alternatives.
Repeating an identical request (a flight to the same target at the same pose, a
``task_status`` query with no state change, a ``cancel_task`` on a terminal
task) returns ``cached`` plus a hint, so the model cannot burn the budget by
looping.

Everything the world does not guarantee is read with ``getattr`` so the tools
run against every backend (``SplatWorldCPU`` / ``EnvWorld`` /
``SuperPlannerWorld``) and against the fake worlds used in tests. Geometry is a
deliberately cheap estimate over the observed map (``world._sense_points`` when
the world keeps one) plus the flown trajectory; results carry structured fields
and evidence references, never prose.

Episode-scoped state kept in ``ctx.extra`` (besides the navigation task records):

- ``viewpoints``: the latest proposal (list of dicts, model-facing)
- ``viewpoints_by_id``: cumulative ``vp_N`` -> viewpoint index (id resolution)
- ``observations``: ``obs_N`` -> pose recorded for a leg/verification
- ``images``: ``obs_N`` -> rendered frame (the channel the runtime turns into
  the next turn's context; disable with ``ctx.extra["publish_frames"] = False``)
- ``last_observation_id``: most recent observation id

Typed ids (``vlm_nav/observations.py``) are enforced at the tool boundary: each
movement tool takes exactly the id type of its intent (``approach_candidate``
only ``candidate_id``, ``go_to_viewpoint`` only ``viewpoint_id``,
``return_to_observation`` only ``observation_id``), and a wrong-typed id is
answered with ``IdTypeError``'s payload - the type that was expected, the type
that was passed and the ids of the expected type that do exist. An id that just
does not exist is answered with the ids that *do* resolve (``available_targets``
/ ``available_viewpoints`` / ``available_observations``) plus an executable next
step that is a *different* call from the failing one.

Evidence refs are symbolic and stable (``map:sense_points``,
``trajectory:trajectory``, ``memory:vp_N``, ``observation:obs_N``,
``map:goal_check``, ``trajectory:nav_N`` for a task); the runtime resolves them
to real frames/resources. No ref names the goal's true pose: a citation the
model can read is a place it could fly to, and the goal is exactly what its own
sensors have to establish.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from . import skills_approach
from .actions import MAX_WAYPOINT_TRANSLATE_M, MAX_WAYPOINT_YAW_DEG, NavAction
from .observations import (
    IdTypeError,
    observation_store,
    register_observation,
    require_candidate_id,
    require_observation_id,
    require_viewpoint_id,
)
from .tool_impl_perception import (
    CANDIDATE_STORE_KEY,
    observation_poses,
    resolve_observation_pose,
)
from .tools import (
    FINISH_SCHEMA,
    MISSING_PLAN_COMPLETE,
    MISSING_SCOPE_MISMATCH,
    MISSING_STALE_CONFIRMATION,
    PROGRESS_INFORMATION,
    SCOPE_FIELDS,
    STATUS_BLOCKED,
    STATUS_CACHED,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_OK,
    EvidenceRef,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    completion_evidence,
    evidence_records_from_pack,
    evidence_pack,
    record_arrival,
    record_candidate,
    record_verification,
    register_batch_tool,
    scope_active,
    task_scope,
)
from .world import estimate_point_world

# --- geometry constants (module-level so a runtime can tune them) -------------
SENSING_RANGE_M = 6.0  # matches the harness sensing range
CLEARANCE_M = 0.3  # min distance to a known surface for a leg/point to be clear
NEAR_M = 1.0  # distance below which an obstacle penalizes the path cost
TRAJ_RADIUS_M = 1.0  # flown trajectory counts as observed space
MAX_LEG_M = 8.0  # one movement leg never exceeds this (matches the waypoint clamp)
ARRIVE_RADIUS_M = 0.35  # leg is "arrived" within this distance of the subgoal
MIN_PROGRESS_M = 0.1  # below this displacement a leg is a failure
PATH_SAMPLES = 12
CONE_BEARINGS = 9
CONE_RANGES = 5
CONE_HALF_ANGLE_DEG = 35.0
VIEWPOINT_SPREAD_DEG = 90.0
VIEWPOINT_NEAR_M = 2.0
VIEWPOINT_FAR_M = 3.0
PATH_POINT_CAP = 12000
CONE_POINT_CAP = 4000
TRAJ_POINT_CAP = 2000
# utility = coverage + 0.5*unknown - 0.1*cost: cheap, wide, unexplored first
UTILITY_COVERAGE_W = 1.0
UTILITY_UNKNOWN_W = 0.5
UTILITY_COST_W = 0.1

# --- navigation task lifecycle (goal.md section 5) ---------------------------
TASK_RUNNING = "running"
TASK_ARRIVED = "arrived"
TASK_BLOCKED = "blocked"
TASK_FAILED = "failed"
TASK_CANCELLED = "cancelled"
TERMINAL_STATUSES = (TASK_ARRIVED, TASK_BLOCKED, TASK_FAILED, TASK_CANCELLED)
_SEGMENT_RESULT_STATUS = {
    TASK_RUNNING: STATUS_OK,  # the leg progressed; the task itself is still running
    TASK_ARRIVED: STATUS_OK,
    TASK_BLOCKED: STATUS_BLOCKED,
    TASK_FAILED: STATUS_FAILED,
}
NAV_TIMEOUT_S = 120.0  # a task left unfinished this long fails on its next segment
NAV_MAX_SEGMENTS = 12  # legs one task may fly before it reports segment_budget
TASK_EVENT_CAP = 32  # events kept on one task record
NAV_EVENT_LOG_CAP = 64  # events kept in the ctx.extra["nav_events"] log
EVENT_QUEUE_CAP = 64  # runtime-drained queue cap when no runtime is attached
MAP_CHANGE_MIN_POINTS = 50  # absolute map growth that counts as significant
MAP_CHANGE_REL = 0.25  # ... or this fraction of the map known at task start

# --- navigation intents (one tool per intent, goal.md section 4) --------------
#: fly to a target candidate (``cand_N``): the *object*, not the camera that saw it
INTENT_CANDIDATE = "approach_candidate"
#: fly to an observation position (``vp_N``) proposed by ``propose_viewpoints``
INTENT_VIEWPOINT = "go_to_viewpoint"
#: fly back to the pose a frame was captured at (``obs_N``) and restore its heading
INTENT_OBSERVATION = "return_to_observation"
MOVE_INTENTS = (INTENT_CANDIDATE, INTENT_VIEWPOINT, INTENT_OBSERVATION)
_INTENT_ID_ARG = {
    INTENT_CANDIDATE: "candidate_id",
    INTENT_VIEWPOINT: "viewpoint_id",
    INTENT_OBSERVATION: "observation_id",
}

# --- unified progress verdict of every movement tool (goal.md section 5) ------
PROGRESS_OK = "ok"  # the leg progressed, or the target is reached
PROGRESS_BLOCKED = "blocked"  # known geometry / the world refused the leg
PROGRESS_NO_PROGRESS = "no_progress"  # the call changed nothing (0 m, 0 deg)
PROGRESS_STATUSES = (PROGRESS_OK, PROGRESS_BLOCKED, PROGRESS_NO_PROGRESS)
#: outcomes that mean "this call was a no-op", reported as ``no_progress``
OUTCOME_ALREADY_AT_POSE = "already_at_pose"
_NO_PROGRESS_OUTCOMES = (
    OUTCOME_ALREADY_AT_POSE,
    "stalled",
    "leg_timeout",
    "outside_bound",
    "no_progress",
)
#: A move that keeps answering ``partial`` without closing distance is the
#: seed-7 resume loop: three consecutive fruitless legs on one (intent, target)
#: are refused before the fourth flies, with a strategy change instead.
REPEAT_MOVE_LIMIT = 3
#: Per-call progress (metres) below which a ``partial`` leg counts as fruitless.
REPEAT_MOVE_MIN_PROGRESS_M = 0.5
#: yaw error below which a heading restore is not worth a turn command
YAW_RESTORE_MIN_DEG = 2.0

REGION_PIXELS = {
    "left": (250.0, 500.0),
    "center": (500.0, 500.0),
    "right": (750.0, 500.0),
    "top": (500.0, 250.0),
    "bottom": (500.0, 750.0),
}

_TARGET_ERROR_HINTS = {
    "missing_target": (
        "pass viewpoint_id (go_to_viewpoint), candidate_id (approach_candidate), "
        "observation_id (return_to_observation) or position=[x, y, z] (check_path)"
    ),
    "unknown_viewpoint": (
        "pass a viewpoint_id from available_viewpoints (propose_viewpoints) to "
        "go_to_viewpoint, or re-run propose_viewpoints"
    ),
    "unknown_target": (
        "pass an id from available_targets (cand_N from ground_target) to "
        "approach_candidate, or an obs id from available_observations to "
        "return_to_observation"
    ),
    "unknown_observation": (
        "no store knows that observation id: call observe (or query_memory) and use "
        "an id it returns, or approach the object with approach_candidate(cand_N) "
        "after ground_target - do not repeat this same id"
    ),
    "bad_position": "position must be [x, y, z] in meters",
}
_UNRESOLVED_TARGET_HINT = (
    "that candidate has no world position yet: observe (forward) and ground_target "
    "again from the current view, then approach_candidate with the new cand_N"
)
#: argument keys a target id may arrive under (canonical first, aliases after)
_TARGET_ID_KEYS = ("viewpoint_id", "target_id", "candidate_id", "observation_id", "obs_id")
#: id spelling of the observation ids ``query_memory`` / ``observe`` return
_OBSERVATION_ID_PREFIXES = ("obs_", "obs#", "observation_")


def _looks_like_observation_id(value: Any) -> bool:
    """Does ``value`` read as an observation id (``obs_3`` / ``obs#3``)?

    Used to pick the error that explains how to fly back to a remembered place
    instead of the generic unknown-target one.
    """
    if not isinstance(value, str):
        return False
    return value.strip().lower().startswith(_OBSERVATION_ID_PREFIXES)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _as_position(value: Any) -> np.ndarray | None:
    """3-element finite world position (list/tuple/array) or None."""
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size != 3 or not np.isfinite(arr).all():
        return None
    return arr


def _as_pixel(value: Any) -> tuple[float, float] | None:
    """Normalized [0,1000] image point (Qwen convention) or None."""
    if value is None or isinstance(value, str):
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size != 2 or not np.isfinite(arr).all():
        return None
    return (
        float(min(max(arr[0], 0.0), 1000.0)),
        float(min(max(arr[1], 0.0), 1000.0)),
    )


def _position(world: Any) -> np.ndarray:
    pos = _as_position(getattr(world, "position_flu", None))
    return pos if pos is not None else np.zeros(3, dtype=np.float64)


def _yaw(world: Any) -> float:
    try:
        return float(getattr(world, "yaw_rad", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _int_arg(args: dict, key: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(args.get(key, default))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, value))


def _float_arg(args: dict, key: str, default: float) -> float:
    try:
        value = float(args.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _round(value: float, digits: int = 2) -> float:
    value = float(value)
    return round(value, digits) if math.isfinite(value) else 0.0


def _round_or_none(value: float, digits: int = 2) -> float | None:
    """Rounded float, or None when the value is unknown (no observed geometry)."""
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def _pos_list(pos: np.ndarray) -> list[float]:
    return [round(float(v), 3) for v in np.asarray(pos, dtype=np.float64).reshape(3)]


def _bias_deg(bearing_rad: float) -> float:
    return (math.degrees(bearing_rad) + 180.0) % 360.0 - 180.0


def _known_points(world: Any) -> np.ndarray:
    """Observed surface points of the world, sentinels and garbage dropped.

    Worlds without an observation map (or in ``blind`` map mode, whose cloud is
    a single far-below sentinel) yield an empty array and the tools degrade to
    trajectory-only evidence instead of failing.
    """
    raw = None
    for attr in ("_sense_points", "sense_points", "obstacle_points"):
        candidate = getattr(world, attr, None)
        if candidate is None:
            continue
        try:
            if len(candidate) == 0:
                continue
        except TypeError:
            continue
        raw = candidate
        break
    if raw is None:
        return np.zeros((0, 3), dtype=np.float64)
    try:
        pts = np.asarray(raw, dtype=np.float64).reshape(-1, 3)
    except (TypeError, ValueError):
        return np.zeros((0, 3), dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.size:
        pos_z = _position(world)[2]
        pts = pts[pts[:, 2] > pos_z - 50.0]  # drop the blind-mode sentinel / debris
    return pts


def _trajectory(world: Any) -> np.ndarray:
    traj = getattr(world, "trajectory", None)
    if traj is None:
        return np.zeros((0, 3), dtype=np.float64)
    try:
        arr = np.asarray(traj, dtype=np.float64).reshape(-1, 3)
    except (TypeError, ValueError):
        return np.zeros((0, 3), dtype=np.float64)
    arr = arr[np.isfinite(arr).all(axis=1)]
    if len(arr) > TRAJ_POINT_CAP:
        arr = arr[:: max(1, len(arr) // TRAJ_POINT_CAP)]
    return arr


def _nearest_subset(pts: np.ndarray, center: np.ndarray, cap: int) -> np.ndarray:
    if len(pts) <= cap:
        return pts
    dist = np.linalg.norm(pts - np.asarray(center, dtype=np.float64), axis=1)
    return pts[np.argpartition(dist, cap - 1)[:cap]]


def _min_distances(queries: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.size == 0 or len(queries) == 0:
        return np.full(len(queries), np.inf)
    return np.linalg.norm(
        np.asarray(queries)[:, None, :] - pts[None, :, :], axis=2
    ).min(axis=1)


def _segment_distances(p0: np.ndarray, p1: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Distance from every point to the segment p0-p1."""
    if pts.size == 0:
        return np.zeros(0, dtype=np.float64)
    seg = np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
    l2 = float(seg @ seg)
    if l2 < 1e-9:
        return np.linalg.norm(pts - p0, axis=1)
    t = np.clip(((pts - p0) @ seg) / l2, 0.0, 1.0)[:, None]
    proj = p0[None, :] + t * seg[None, :]
    return np.linalg.norm(pts - proj, axis=1)


def _path_unknown_ratio(
    p0: np.ndarray, p1: np.ndarray, pts: np.ndarray, traj: np.ndarray
) -> float:
    """Fraction of the leg with no observation evidence.

    A sample counts as observed when a known surface point lies within the
    sensor range (the volume has been seen) or the flown trajectory passes
    within ``TRAJ_RADIUS_M`` (the drone has been there).
    """
    ts = np.linspace(0.1, 0.9, PATH_SAMPLES)[:, None]
    samples = np.asarray(p0)[None, :] + ts * (np.asarray(p1) - np.asarray(p0))[None, :]
    mid = 0.5 * (np.asarray(p0) + np.asarray(p1))
    subset = _nearest_subset(pts, mid, PATH_POINT_CAP)
    d_obs = _min_distances(samples, subset)
    d_traj = _min_distances(samples, traj)
    observed = (d_obs <= SENSING_RANGE_M) | (d_traj <= TRAJ_RADIUS_M)
    return float(1.0 - observed.mean())


def _cone_metrics(
    vp: np.ndarray, bearing: float, pts: np.ndarray, traj: np.ndarray
) -> tuple[float, float]:
    """(expected_coverage, unknown_ratio) of the forward sensor cone at ``vp``.

    ``expected_coverage`` = fraction of the cone free of known geometry (wide,
    unoccluded view), ``unknown_ratio`` = fraction with no observation evidence
    (unexplored in front of the viewpoint).
    """
    bearings = bearing + np.radians(
        np.linspace(-CONE_HALF_ANGLE_DEG, CONE_HALF_ANGLE_DEG, CONE_BEARINGS)
    )
    ranges = np.linspace(SENSING_RANGE_M / CONE_RANGES, SENSING_RANGE_M, CONE_RANGES)
    offsets = np.stack(
        [
            (ranges[None, :] * np.cos(bearings)[:, None]).reshape(-1),
            (ranges[None, :] * np.sin(bearings)[:, None]).reshape(-1),
            np.zeros(CONE_BEARINGS * CONE_RANGES),
        ],
        axis=1,
    )
    samples = np.asarray(vp)[None, :] + offsets
    subset = _nearest_subset(pts, vp, CONE_POINT_CAP)
    d_obs = _min_distances(samples, subset)
    d_traj = _min_distances(samples, traj)
    coverage = float((d_obs > CLEARANCE_M).mean())
    unknown = float(((d_obs > SENSING_RANGE_M) & (d_traj > TRAJ_RADIUS_M)).mean())
    return coverage, unknown


# ---------------------------------------------------------------------------
# context-state helpers
# ---------------------------------------------------------------------------


def _entry_position(entry: Any) -> np.ndarray | None:
    if entry is None:
        return None
    if isinstance(entry, dict):
        for key in ("position_flu", "position", "pos"):
            pos = _as_position(entry.get(key))
            if pos is not None:
                return pos
        data = entry.get("data")  # memory.HypothesisStore entries nest it
        if isinstance(data, dict):
            for key in ("position_flu", "position", "pos", "world_flu", "target_flu"):
                pos = _as_position(data.get(key))
                if pos is not None:
                    return pos
        return None
    for attr in ("position_flu", "position", "pos"):
        pos = _as_position(getattr(entry, attr, None))
        if pos is not None:
            return pos
    return None


def _memory_lookup(ctx: ToolContext, key: str) -> np.ndarray | None:
    """Position of a candidate/hypothesis id through the memory module.

    ``ctx.memory`` may be a ``CandidateTracker``, an ``ObservationLog`` or a
    ``HypothesisStore``; every getter is optional and probed defensively.
    """
    memory = ctx.memory
    if memory is None:
        return None
    for name in ("get_candidate", "get", "resolve_position", "get_position"):
        getter = getattr(memory, name, None)
        if not callable(getter):
            continue
        try:
            value = getter(key)
        except Exception:
            continue
        pos = _entry_position(value)
        if pos is not None:
            return pos
    return None


def _entry_id(entry: Any) -> str | None:
    if isinstance(entry, dict):
        for name in ("id", "candidate_id", "obs_id", "observation_id"):
            value = entry.get(name)
            if value not in (None, ""):
                return str(value)
        return None
    for name in ("id", "candidate_id", "obs_id", "observation_id"):
        value = getattr(entry, name, None)
        if value not in (None, ""):
            return str(value)
    return None


def _candidate_entries(ctx: ToolContext) -> list:
    """Every candidate record the context knows.

    That is ``ctx.memory.candidates`` / ``ctx.extra['candidates']`` (the
    harness' ``CandidateTracker``-style lists) plus the perception tools'
    canonical store ``ctx.extra[CANDIDATE_STORE_KEY]`` that ``ground_target``
    fills in - the store a ``cand_N`` id the model was just handed lives in.
    """
    entries: list = []
    for holder in (ctx.memory, ctx.extra):
        if holder is None:
            continue
        source = (
            holder.get("candidates") if isinstance(holder, dict)
            else getattr(holder, "candidates", None)
        )
        if isinstance(source, dict):
            entries.extend(source.values())
        elif isinstance(source, (list, tuple)):
            entries.extend(source)
    store = ctx.extra.get(CANDIDATE_STORE_KEY) if isinstance(ctx.extra, dict) else None
    if isinstance(store, dict):
        entries.extend(store.values())
    elif isinstance(store, (list, tuple)):
        entries.extend(store)
    return entries


def _candidate_ids(ctx: ToolContext, limit: int = 8) -> tuple[list[str], list[str]]:
    """(ids that resolve to a world position, ids that do not).

    Grounding an instruction yields an id the model can hand straight back to
    ``approach_candidate``; a candidate grounded without a range estimate has no
    position yet and must be reported as such instead of silently resolving
    nowhere.
    """
    resolved: list[str] = []
    unresolved: list[str] = []
    table = ctx.extra.get("candidate_positions") or {}
    if isinstance(table, dict):
        resolved.extend(str(key) for key in table)
    for entry in _candidate_entries(ctx):
        key = _entry_id(entry)
        if key is None or key in resolved or key in unresolved:
            continue
        (resolved if _entry_position(entry) is not None else unresolved).append(key)
    return resolved[-limit:], unresolved[-limit:]


def _candidate_position(ctx: ToolContext, candidate_id: Any) -> np.ndarray | None:
    """Best-effort lookup of a candidate/hypothesis world position.

    Checks, in order: ``ctx.extra['candidate_positions']``, the candidate
    records of ``ctx.memory`` and the perception store
    ``ctx.extra['perception.candidates']`` (``ground_target`` ids), then the
    memory module getters (``get_candidate`` / ``get`` / ...).
    """
    key = str(candidate_id)
    table = ctx.extra.get("candidate_positions") or {}
    if isinstance(table, dict):
        for lookup in (key, candidate_id):
            try:
                pos = _as_position(table.get(lookup))
            except TypeError:
                pos = None
            if pos is not None:
                return pos
    for entry in _candidate_entries(ctx):
        if _entry_id(entry) != key:
            continue
        pos = _entry_position(entry)
        if pos is not None:
            return pos
    return _memory_lookup(ctx, key)


def _resolve_hypothesis(ctx: ToolContext, name: Any) -> np.ndarray | None:
    """Resolve a free-form hypothesis reference to a world position.

    A hypothesis resolves through the evidence the episode holds - a declared
    hypothesis, a candidate position, a remembered place. It deliberately never
    resolves through ``world.target``: a name that happens to match the goal's
    own name used to come back as the goal's *true* position, so an arc aimed at
    it was ground-truth guidance (the alias ``target_id`` made that path easy to
    stumble into). A name nothing has located stays unresolved.
    """
    key = str(name).strip()
    table = ctx.extra.get("hypotheses") or {}
    if isinstance(table, dict):
        entry = table.get(key)
        pos = _as_position(entry)
        if pos is None:
            pos = _entry_position(entry)
        if pos is not None:
            return pos
    return _candidate_position(ctx, key)


def _viewpoint_position(ctx: ToolContext, viewpoint_id: Any) -> np.ndarray | None:
    key = str(viewpoint_id)
    by_id = ctx.extra.get("viewpoints_by_id") or {}
    if isinstance(by_id, dict) and key in by_id:
        return _entry_position(by_id[key])
    for entry in ctx.extra.get("viewpoints") or []:
        if str(entry.get("viewpoint_id")) == key:
            return _entry_position(entry)
    return None


def _resolve_target(
    ctx: ToolContext, args: dict
) -> tuple[np.ndarray | None, str | None, EvidenceRef | None]:
    """(position, error, evidence) for a {viewpoint_id|target_id|position} call.

    Reference order mirrors goal.md: an explicit position wins (the id-free
    path - no candidate or viewpoint needed), then a proposed viewpoint id
    (``vp_N``), then the ids of a place that was actually seen - a recalled
    observation id (``obs_N``, under ``target_id`` or its own
    ``observation_id`` / ``obs_id`` argument), which is read first so its
    evidence is cited as the observation it is - then a candidate id (the
    ``cand_N`` ids ``ground_target`` returns).
    """
    if args.get("position") is not None:
        pos = _as_position(args.get("position"))
        if pos is None:
            return None, "bad_position", None
        return pos, None, None
    for key in _TARGET_ID_KEYS:
        value = args.get(key)
        if value in (None, ""):
            continue
        pos = _viewpoint_position(ctx, value)
        if pos is not None:
            return pos, None, EvidenceRef(
                "memory", f"vp:{value}", note="proposed viewpoint"
            )
        if key in ("observation_id", "obs_id") or _looks_like_observation_id(value):
            pose = resolve_observation_pose(ctx, value)
            if pose is not None:
                return np.asarray(pose[0], dtype=np.float64), None, EvidenceRef(
                    "observation", str(value), note="recalled observation pose"
                )
        pos = _candidate_position(ctx, value)
        if pos is not None:
            return pos, None, EvidenceRef(
                "memory", str(value), note="candidate position"
            )
        return None, _unknown_id_error(key, value), None
    return None, "missing_target", None


def _unknown_id_error(key: str, value: Any) -> str:
    """Error reason of an id that resolves nowhere.

    An id requested through ``observation_id`` / ``obs_id``, or one that reads
    as an observation id (``obs_3``), gets its own reason: the answer then
    explains how to fly back to a remembered place instead of listing candidate
    ids only.
    """
    if key in ("observation_id", "obs_id") or _looks_like_observation_id(value):
        return "unknown_observation"
    return "unknown_viewpoint" if key == "viewpoint_id" else "unknown_target"


def _best_recall_suggestion(
    ctx: ToolContext, observations: dict[str, list[float]]
) -> tuple[str, list[float]] | None:
    """The recalled place worth naming as the next target (or None).

    ``observations`` is newest-first, but the newest one is usually the frame the
    runtime just rendered at the *current* pose: sending the drone there is not a
    move (``already_at_pose``) and naming it would recommend exactly the call that
    just did nothing. Only a place at least one arrival radius away is offered;
    when every known place is underfoot this returns None (the answer then names
    a call that *can* change something instead).
    """
    if not observations:
        return None
    here = _position(ctx.world)
    for obs_id, position in observations.items():
        if float(np.linalg.norm(np.asarray(position, dtype=np.float64) - here)) > ARRIVE_RADIUS_M:
            return obs_id, position
    return None


def _instruction(ctx: ToolContext) -> str:
    """The episode's instruction, when the runtime published one."""
    value = ctx.extra.get("instruction") if isinstance(ctx.extra, dict) else None
    return "" if value in (None, "") else str(value)


def _ground_again_instruction(ctx: ToolContext) -> str:
    """What an automatic re-grounding asks for: the current sub-goal, not the task.

    The plan's executing sub-task (with its relation anchor and preconditions)
    wins; without a plan the harness-published ``subgoal`` is the current scope;
    only with neither does the global instruction apply (goal.md v1, seed23).
    """
    try:
        from .tool_impl_task import subtask_ground_instruction

        text = subtask_ground_instruction(ctx)
    except Exception:  # noqa: BLE001 - a context without the task module
        text = ""
    if not text:
        value = ctx.extra.get("subgoal") if isinstance(ctx.extra, dict) else None
        text = str(value) if value not in (None, "") else _instruction(ctx)
    return text or "<what you are looking for>"


def _ground_again_action(ctx: ToolContext, observation_id: str = "") -> dict:
    """The executable call that turns "I see it" into a candidate to approach.

    ``ground_target`` needs an instruction and a frame; the instruction is the
    executing sub-task (its relation anchor included) and ``obs_id`` is the
    frame the model is looking at, so the model can issue the returned call
    unchanged instead of re-inventing an id. The call carries the same task
    scope as its result: task/sub-task/candidate/evidence revision.
    """
    args: dict[str, Any] = {"instruction": _ground_again_instruction(ctx)}
    if observation_id:
        args["observation_id"] = observation_id
    action: dict[str, Any] = {"tool": "ground_target", "args": args}
    scope = task_scope(ctx)
    if scope_active(ctx, scope):
        for field in SCOPE_FIELDS:
            value = scope.get(field)
            if value not in (None, ""):
                args[field] = value
        action["scope"] = {field: scope[field] for field in SCOPE_FIELDS}
    return action


def _target_error_result(ctx: ToolContext, args: dict, error: str) -> ToolResult:
    """Failure payload of a target that cannot be resolved.

    An id the tool never saw is the model's most common mistake, so the answer
    lists the ids that *do* resolve (and, separately, the candidate ids that
    exist but carry no world position yet) instead of only naming the error -
    the model can pick a real id instead of inventing the next one. When a
    recalled observation pose is known *and is a real move*, the answer goes one
    step further and carries the executable ``return_to_observation`` call;
    when every remembered place is underfoot (the zero-displacement loop this
    contract exists to break) the next step is re-grounding the object instead,
    never "fly to where you already are".
    """
    data: dict[str, Any] = {"reason": error, "hint": _TARGET_ERROR_HINTS.get(error, "")}
    if error in ("unknown_target", "unknown_viewpoint", "missing_target", "unknown_observation"):
        resolved, unresolved = _candidate_ids(ctx)
        data["available_targets"] = resolved
        data["unresolved_targets"] = unresolved
    if error == "unknown_viewpoint":
        by_id = ctx.extra.get("viewpoints_by_id") or {}
        data["available_viewpoints"] = list(by_id)[-8:] if isinstance(by_id, dict) else []
        data["next_action"] = {"tool": "propose_viewpoints", "args": {"count": 3}}
    if error in ("unknown_target", "unknown_observation", "missing_target"):
        observations = observation_poses(ctx)
        data["available_observations"] = list(observations)
        suggestion = _best_recall_suggestion(ctx, observations)
        if suggestion is not None:
            best_id, best_position = suggestion
            data["suggested_position"] = best_position
            data["next_action"] = {
                "tool": "return_to_observation",
                "args": {"observation_id": best_id},
            }
            data["hint"] = (
                f"no movement happened: fly back to the recalled observation "
                f"{best_id} at {best_position} with "
                f'return_to_observation({{"observation_id": "{best_id}"}}), then '
                "observe and ground_target what you are looking for"
            )
        else:
            # nothing to fly to: the executable next step is to look again, not
            # to repeat the id that just failed
            data["next_action"] = _ground_again_action(
                ctx, _observation_id(ctx)
            )
            data["hint"] = (
                "no recalled place is a real move (every known pose is where the "
                "drone already is): observe again and call ground_target, then "
                "approach_candidate with the cand_N it returns"
            )
    for key in _TARGET_ID_KEYS:
        value = args.get(key)
        if value not in (None, ""):
            data[key] = str(value)
            break
    requested = data.get("target_id") or data.get("candidate_id")
    if requested and requested in data.get("unresolved_targets", []):
        data["hint"] = _UNRESOLVED_TARGET_HINT
        if data.get("next_action", {}).get("tool") == "return_to_observation":
            # re-grounding is one option, but a posed observation is a place the
            # drone can fly to right now - name it as well
            best_id = data["next_action"]["args"]["observation_id"]
            data["hint"] = (
                f"{_UNRESOLVED_TARGET_HINT}; or return to the recalled observation "
                f"{best_id} at {data['suggested_position']} (next_action)"
            )
    return ToolResult(STATUS_FAILED, data)


def _max_leg_m(ctx: ToolContext) -> float:
    try:
        return float(ctx.extra.get("max_leg_m", MAX_LEG_M))
    except (TypeError, ValueError):
        return MAX_LEG_M


def _map_evidence(world: Any, pts: np.ndarray, traj: np.ndarray) -> list[EvidenceRef]:
    evidence: list[EvidenceRef] = []
    if pts.size:
        evidence.append(
            EvidenceRef("map", "sense_points", note=f"{len(pts)} observed points")
        )
    if traj.size:
        evidence.append(
            EvidenceRef("trajectory", "trajectory", note=f"{len(traj)} poses")
        )
    if callable(getattr(world, "bev_image", None)):
        evidence.append(EvidenceRef("image", "bev_latest", note="top-down map view"))
    return evidence


def _next_task_id(ctx: ToolContext) -> str:
    seq = int(ctx.extra.get("_nav_task_seq", 0)) + 1
    ctx.extra["_nav_task_seq"] = seq
    return f"nav_{seq}"


def _observation_id(ctx: ToolContext) -> str:
    """Observation id backing the current view ('' when the context has none)."""
    for key in ("current_observation", "last_observation_id"):
        value = ctx.extra.get(key)
        if value:
            return str(value)
    return ""


def _current_observation(ctx: ToolContext, kind: str) -> str:
    """Observation id of the current view, allocating and capturing one when the
    context has none yet (so the returned reference always resolves)."""
    obs_id = _observation_id(ctx)
    if obs_id:
        return obs_id
    obs_id = ctx.next_obs_id()
    _capture_observation(ctx, obs_id, kind)
    return obs_id


def _capture_observation(
    ctx: ToolContext, obs_id: str, kind: str, task_id: str = ""
) -> None:
    """Record the current pose as ``obs_id`` and publish its rendered frame.

    The frame goes to ``ctx.extra['images'][obs_id]`` - the channel the agent
    runtime turns into the next turn's context - so the ``observation`` evidence
    reference resolves to real content, and it is registered in the episode's
    unified :class:`~vlm_nav.observations.ObservationStore` (the repository every
    typed ``obs_N`` argument resolves through, see ``require_observation_id``).
    Worlds without ``render`` (fake worlds) still get the pose record.
    """
    if not obs_id:
        return
    world = ctx.world
    pose = _position(world)
    yaw = _yaw(world)
    record = {
        "observation_id": obs_id,
        "position_flu": _pos_list(pose),
        "yaw_rad": _round(yaw, 4),
        "heading_deg": _round(_bias_deg(yaw)),
        "kind": kind,
    }
    if task_id:
        record["task_id"] = task_id
    ctx.extra.setdefault("observations", {})[obs_id] = record
    ctx.extra["last_observation_id"] = obs_id
    frame = None
    if ctx.extra.get("publish_frames", True):
        render = getattr(world, "render", None)
        if callable(render):
            try:
                frame = render()
                ctx.extra.setdefault("images", {})[obs_id] = frame
            except Exception:  # rendering is best effort; the pose record stands
                frame = None
    store = observation_store(ctx, create=True)  # the unified repository's one home
    if store is not None and store.get(obs_id) is None:
        try:
            register_observation(
                ctx,
                frame,
                kind=kind,
                obs_id=obs_id,
                pose_flu=pose,
                yaw_deg=math.degrees(yaw),
                note=f"{kind} frame captured by the navigation tools",
            )
        except Exception:  # noqa: BLE001 - the pose record above is the contract
            pass


# ---------------------------------------------------------------------------
# low-altitude flight profile (altitude_audit.md section 5.1)
# ---------------------------------------------------------------------------

#: cruise window above the navigable floor: a proposed viewpoint sits inside it
CRUISE_WINDOW_MIN_M = 0.6
CRUISE_WINDOW_MAX_M = 1.2
#: no-fly height above the navigable floor: nothing is proposed above it
CRUISE_CEILING_M = 1.6
#: a ring viewpoint looks at the object from this far above its own top
VIEWPOINT_LOOK_DOWN_M = 0.3
#: terminal approach goal: bbox bottom + min(0.6 * object height, 0.35) ...
APPROACH_GOAL_HEIGHT_FRACTION = 0.6
APPROACH_GOAL_HEIGHT_MAX_M = 0.35
#: ... and never below the floor (raised to the cruise floor when that is higher)
APPROACH_GOAL_FLOOR_CLEARANCE_M = 0.4


def _declared_z(holder: Any, names: tuple[str, ...]) -> float | None:
    """First finite z of a height declaration on ``holder``, else None.

    Accepts the two spellings the repository uses - a per-env scalar sequence
    (``[value]``, the shape the env's tensors have) and a 3-vector (z =
    component 2) - and reads them through ``detach().cpu()`` so the env's CUDA
    tensors resolve like its numpy arrays.
    """
    for name in names:
        value = getattr(holder, name, None)
        if value is None:
            continue
        if hasattr(value, "detach"):  # a torch tensor declaration (env tensors)
            value = value.detach().cpu()
        try:
            values = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            continue
        if values.size == 0 or not np.isfinite(values).all():
            continue
        return float(values[2] if values.size >= 3 else values[0])
    return None


def _floor_z(world: Any) -> float | None:
    """Navigable floor of the world in FLU z, or None when it is unknown.

    The env's own ``navigation_min_flu`` bound is the floor its cruise envelope
    is measured against, so the tool layer reads the same one. None means
    "unknown": the altitude policies then leave altitudes untouched instead of
    inventing a floor out of whatever geometry happens to be mapped yet.
    """
    for holder in (getattr(world, "env", None), world):
        if holder is None:
            continue
        floor = _declared_z(holder, ("navigation_min_flu",))
        if floor is not None:
            return floor
    return None


def _target_bbox_z(world: Any) -> tuple[float, float] | None:
    """(bottom, top) z of the semantic target's box, or None when undeclared.

    The env carries it as the per-env tensors ``target_object_bbox_min_z_flu`` /
    ``target_object_bbox_size_z_flu`` (the pair ``env`` itself feeds to
    ``semantic_goal_height``); a world or target object declaring the same pair
    resolves too. An empty box is None - never a guessed extent.
    """
    for holder in (getattr(world, "target", None), getattr(world, "env", None)):
        if holder is None:
            continue
        bottom = _declared_z(
            holder, ("bbox_min_z_flu", "target_object_bbox_min_z_flu", "bbox_min_flu")
        )
        size = _declared_z(
            holder, ("bbox_size_z_flu", "target_object_bbox_size_z_flu", "bbox_size_flu")
        )
        if bottom is not None and size is not None and size > 0.0:
            return bottom, bottom + size
    return None


def _positive_or(holder: Any, name: str, default: float) -> float:
    value = _declared_z(holder, (name,))
    return float(value) if value is not None and value > 0.0 else float(default)


def _cruise_window(world: Any, floor: float | None) -> tuple[float, float] | None:
    """(lower, upper) FLU altitude a cruise viewpoint may use, or None.

    Measured above the floor the way the env's own envelope is (``cfg.env.cruise``
    -> ``env.cruise_window_*``, falling back to the audit's 0.6/1.2/1.6 m), and
    capped by the no-fly ceiling, so "look at it from another angle" can never
    mean "climb over it". Without a floor there is nothing to measure against.
    """
    if floor is None:
        return None
    min_m, max_m = CRUISE_WINDOW_MIN_M, CRUISE_WINDOW_MAX_M
    ceiling_m = CRUISE_CEILING_M
    env = getattr(world, "env", None)
    if env is not None:
        min_m = _positive_or(env, "cruise_window_min_m", min_m)
        max_m = _positive_or(env, "cruise_window_max_m", max_m)
        ceiling_m = _positive_or(env, "cruise_ceiling_m", ceiling_m)
    lower = float(floor + min_m)
    upper = float(floor + min(max(max_m, min_m), ceiling_m))
    return lower, max(upper, lower)


def _envelope_z(world: Any, z: float) -> float:
    """``z`` inside the world's declared flight envelope (``z_min`` / ``z_max``)."""
    low = _declared_z(world, ("z_min",))
    high = _declared_z(world, ("z_max",))
    if low is not None and math.isfinite(low):
        z = max(z, low)
    if high is not None and math.isfinite(high):
        z = min(z, high)
    return float(z)


def _viewpoint_altitude(world: Any, pos: Any) -> float:
    """Altitude of a proposed viewpoint (world FLU).

    The near/far ring sits within the target's own height band
    (``obj_top + 0.3``) clipped into the low cruise window, or at the drone's
    current altitude clipped into that window when no target box is known - so a
    viewpoint is never proposed above the no-fly ceiling and the ring no longer
    inherits a start altitude sampled over the object. With neither a target box
    nor a known floor the altitude is left alone.
    """
    z = float(np.asarray(pos, dtype=np.float64).reshape(3)[2])
    box = _target_bbox_z(world)
    if box is not None:
        z = float(box[1]) + VIEWPOINT_LOOK_DOWN_M
    window = _cruise_window(world, _floor_z(world))
    if window is not None:
        z = min(max(z, window[0]), window[1])
    return _envelope_z(world, z)


def _approach_goal_z(world: Any, target_point: Any) -> float | None:
    """Altitude the approach's standoff point aims at, or None without a target.

    The drone parks *beside* the object, not above it: with the target box
    declared the goal is ``bottom + min(0.6 * height, 0.35)``, without one the
    measured point's own height is the anchor. Either way the goal stays above
    the floor, raised to the cruise floor when that is higher.
    """
    point = _as_position(target_point)
    if point is None:
        return None
    z = float(point[2])
    box = _target_bbox_z(world)
    if box is not None:
        z = float(box[0]) + min(
            APPROACH_GOAL_HEIGHT_FRACTION * (box[1] - box[0]), APPROACH_GOAL_HEIGHT_MAX_M
        )
    floor = _floor_z(world)
    if floor is not None:
        window = _cruise_window(world, floor)
        clearance = APPROACH_GOAL_FLOOR_CLEARANCE_M
        if window is not None:
            clearance = max(clearance, window[0] - floor)
        z = max(z, floor + clearance)
    return _envelope_z(world, z)


# ---------------------------------------------------------------------------
# propose_viewpoints
# ---------------------------------------------------------------------------


def _build_viewpoints(
    ctx: ToolContext,
    base_bearing: float,
    count: int,
    radius_m: float | None = None,
    spread_deg: float | None = None,
) -> list[dict]:
    """Candidate observation points on an arc around the current pose.

    ``base_bearing`` is a world-frame heading (rad); candidates alternate
    between the near and far ring at the low cruise altitude
    (:func:`_viewpoint_altitude`) and are scored on map evidence only (no pose
    change, no rendering). ``radius_m`` overrides the far ring radius (the near
    ring follows at 0.65x); ``spread_deg`` widens the arc (360 = full circle,
    used for retry suggestions behind the drone).
    """
    world = ctx.world
    pos = _position(world)
    altitude = _viewpoint_altitude(world, pos)
    pts = _known_points(world)
    traj = _trajectory(world)
    bound = getattr(world, "bound_radius", None)
    if radius_m is None:
        near_m, far_m = VIEWPOINT_NEAR_M, VIEWPOINT_FAR_M
    else:
        far_m = max(0.5, min(MAX_LEG_M, float(radius_m)))
        near_m = max(0.5, min(VIEWPOINT_NEAR_M, 0.65 * far_m))
    fracs = [0.0] if count <= 1 else [i / (count - 1) - 0.5 for i in range(count)]
    spread = VIEWPOINT_SPREAD_DEG if spread_deg is None else float(spread_deg)
    candidates: list[dict] = []
    for i, frac in enumerate(fracs):
        bearing = base_bearing + frac * math.radians(spread)
        radius = near_m if i % 2 == 0 else far_m
        vp = np.asarray(
            [
                pos[0] + radius * math.cos(bearing),
                pos[1] + radius * math.sin(bearing),
                altitude,
            ],
            dtype=np.float64,
        )
        if bound is not None:
            try:
                if math.hypot(vp[0], vp[1]) > float(bound) * 0.98:
                    continue
            except (TypeError, ValueError):
                pass
        dist = float(np.linalg.norm(vp - pos))
        subset = _nearest_subset(pts, vp, PATH_POINT_CAP)
        d_seg = _segment_distances(pos, vp, subset)
        min_clear = float(d_seg.min()) if d_seg.size else float("inf")
        n_near = int((d_seg <= NEAR_M).sum()) if d_seg.size else 0
        cost = dist + 0.35 * min(n_near, 8)
        coverage, unknown = _cone_metrics(vp, bearing, pts, traj)
        candidates.append(
            {
                "position": _pos_list(vp),
                "distance_m": _round(dist),
                "bearing_deg": _round(_bias_deg(bearing)),
                "cost_m": _round(cost),
                "expected_coverage": _round(coverage, 3),
                "unknown_ratio": _round(unknown, 3),
                "min_clearance_m": _round_or_none(min_clear),
                "reachable": bool(min_clear > CLEARANCE_M),
                "utility": _round(
                    UTILITY_COVERAGE_W * coverage
                    + UTILITY_UNKNOWN_W * unknown
                    - UTILITY_COST_W * cost,
                    3,
                ),
            }
        )
    return candidates


def _register_viewpoints(ctx: ToolContext, candidates: list[dict]) -> list[dict]:
    stored = []
    seq = int(ctx.extra.get("_viewpoint_seq", 0))
    by_id = ctx.extra.setdefault("viewpoints_by_id", {})
    for cand in candidates:
        seq += 1
        entry = dict(cand)
        entry["viewpoint_id"] = f"vp_{seq}"
        by_id[entry["viewpoint_id"]] = dict(entry)
        stored.append(entry)
    ctx.extra["_viewpoint_seq"] = seq
    ctx.extra["viewpoints"] = [dict(entry) for entry in stored]
    return stored


def _tool_propose_viewpoints(ctx: ToolContext, args: dict) -> ToolResult:
    world = ctx.world
    pos = _position(world)
    yaw = _yaw(world)
    count = _int_arg(
        args, "count", _int_arg(args, "n", _int_arg(args, "max_viewpoints", 5, 1, 6), 1, 6), 1, 6
    )
    radius_m = args.get("radius_m")
    radius_m = None if radius_m is None else _float_arg(args, "radius_m", VIEWPOINT_FAR_M)
    base_bearing = yaw
    hypothesis_note = ""
    hypothesis = args.get("target_hypothesis")
    if hypothesis in (None, ""):
        hypothesis = args.get("target_id")  # MCP contract alias
    if hypothesis not in (None, ""):
        hyp_pos = _as_position(hypothesis) if not isinstance(hypothesis, str) else None
        resolved = hyp_pos if hyp_pos is not None else (
            _resolve_hypothesis(ctx, hypothesis) if isinstance(hypothesis, str) else None
        )
        if resolved is not None:
            base_bearing = math.atan2(resolved[1] - pos[1], resolved[0] - pos[0])
        else:
            hypothesis_note = f"unresolved:{hypothesis}"
    elif args.get("direction_deg") is not None:
        base_bearing = yaw + math.radians(_float_arg(args, "direction_deg", 0.0))

    candidates = _build_viewpoints(ctx, base_bearing, count, radius_m)
    stored = _register_viewpoints(ctx, candidates)
    stored.sort(key=lambda item: (-item["utility"], item["cost_m"]))
    reachable = [item for item in stored if item["reachable"]]

    pts = _known_points(world)
    traj = _trajectory(world)
    evidence = _map_evidence(world, pts, traj)
    if hypothesis_note:
        evidence.append(
            EvidenceRef("memory", str(hypothesis), note="unresolved hypothesis reference")
        )
    data: dict[str, Any] = {
        "count": len(stored),
        "reachable_count": len(reachable),
        "base_bearing_deg": _round(_bias_deg(base_bearing)),
        "viewpoints": stored,
    }
    if stored:
        # recommended pick: best reachable candidate, or the best blocked one so
        # the model still sees where the geometry is
        data["viewpoint_id"] = (reachable or stored)[0]["viewpoint_id"]
    observation_id = _observation_id(ctx)
    if observation_id:
        data["last_observation_id"] = observation_id
    if args.get("target_hypothesis") not in (None, ""):
        data["hypothesis_resolved"] = not hypothesis_note
    if hypothesis_note:
        data["hypothesis_note"] = hypothesis_note
    if not stored:
        return ToolResult(
            STATUS_BLOCKED,
            {**data, "reason": "no_candidate_viewpoint"},
            evidence,
        )
    if not reachable:
        return ToolResult(
            STATUS_BLOCKED,
            {**data, "reason": "no_reachable_viewpoint"},
            evidence,
        )
    return ToolResult(STATUS_OK, data, evidence)


# ---------------------------------------------------------------------------
# check_path
# ---------------------------------------------------------------------------


def _path_estimate(
    ctx: ToolContext, target: np.ndarray
) -> tuple[dict[str, Any], list[EvidenceRef]]:
    """Shared straight-leg estimate used by check_path and the movement tools."""
    world = ctx.world
    pos = _position(world)
    pts = _known_points(world)
    traj = _trajectory(world)
    delta = np.asarray(target, dtype=np.float64) - pos
    dist = float(np.linalg.norm(delta))
    mid = 0.5 * (pos + np.asarray(target, dtype=np.float64))
    subset = _nearest_subset(pts, mid, PATH_POINT_CAP)
    d_seg = _segment_distances(pos, np.asarray(target, dtype=np.float64), subset)
    min_clear = float(d_seg.min()) if d_seg.size else float("inf")
    n_near = int((d_seg <= NEAR_M).sum()) if d_seg.size else 0
    cost = dist + 0.35 * min(n_near, 8)
    unknown_ratio = _path_unknown_ratio(pos, np.asarray(target), pts, traj)
    max_leg = _max_leg_m(ctx)
    too_far = dist > max_leg
    blocked = min_clear <= CLEARANCE_M
    if too_far:
        reason = "too_far"
    elif blocked:
        reason = "obstacle_on_path"
    elif unknown_ratio > 0.6:
        reason = "mostly_unknown"
    else:
        reason = "clear"
    info = {
        "reachable": bool(not too_far and not blocked),
        # component flags: a movement tool may fly a graded (clamped) leg toward a
        # distant target, but never through known geometry
        "too_far": bool(too_far),
        "blocked": bool(blocked),
        "path_cost_m": _round(cost),
        "unknown_ratio": _round(unknown_ratio, 3),
        "reason": reason,
        "distance_m": _round(dist),
        "min_clearance_m": _round_or_none(min_clear),
        "max_leg_m": _round(max_leg),
        "obstacles_near_path": n_near,
    }
    return info, _map_evidence(world, pts, traj)


def _tool_check_path(ctx: ToolContext, args: dict) -> ToolResult:
    target, error, ref = _resolve_target(ctx, args)
    if error is not None:
        return _target_error_result(ctx, args, error)
    info, evidence = _path_estimate(ctx, target)
    if ref is not None:
        evidence.insert(0, ref)
    data = {**info, "target_position": _pos_list(target)}
    status = STATUS_OK if info["reachable"] else STATUS_BLOCKED
    return ToolResult(status, data, evidence)


# ---------------------------------------------------------------------------
# one leg + the blocked-leg alternatives
# ---------------------------------------------------------------------------


def _leg_action(
    pos: np.ndarray, yaw: float, target: np.ndarray, reason: str = "waypoint"
) -> NavAction:
    """One straight leg, facing the direction of travel.

    Every world applies ``yaw_deg`` before the translation, so the body-frame
    offset is expressed in the post-turn heading; ``pixel`` stays None (no
    point-to-navigate standoff - the planned waypoint *is* the destination).
    """
    delta = np.asarray(target, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
    bearing = math.atan2(float(delta[1]), float(delta[0]))
    dyaw = math.atan2(math.sin(bearing - yaw), math.cos(bearing - yaw))
    dyaw = max(
        -math.radians(MAX_WAYPOINT_YAW_DEG), min(math.radians(MAX_WAYPOINT_YAW_DEG), dyaw)
    )
    yaw_new = yaw + dyaw
    dxb = math.cos(yaw_new) * float(delta[0]) + math.sin(yaw_new) * float(delta[1])
    dyb = -math.sin(yaw_new) * float(delta[0]) + math.cos(yaw_new) * float(delta[1])
    return NavAction(
        kind="waypoint",
        forward_m=max(-MAX_WAYPOINT_TRANSLATE_M, min(MAX_WAYPOINT_TRANSLATE_M, dxb)),
        right_m=max(-MAX_WAYPOINT_TRANSLATE_M, min(MAX_WAYPOINT_TRANSLATE_M, -dyb)),
        up_m=max(-MAX_WAYPOINT_TRANSLATE_M, min(MAX_WAYPOINT_TRANSLATE_M, float(delta[2]))),
        yaw_deg=math.degrees(dyaw),
        reason=str(reason),
    )


def _outcome_keyword(feedback: Any) -> str:
    text = str(feedback or "").lower()
    for keyword in ("collision", "planner failed", "stalled", "terminated", "leg timeout"):
        if keyword in text:
            return keyword
    if "clamped" in text:
        return "clamped"
    return "moved"


def _alternative_ids(ctx: ToolContext, exclude: Any = None, limit: int = 3) -> list[str]:
    """Useful next targets for a blocked leg.

    Stored reachable viewpoints first; when the whole forward arc is blocked, a
    fresh full-circle set is proposed (and registered, so the ids resolve in a
    follow-up check_path / go_to_viewpoint). Returns [] when nothing around is
    reachable.
    """
    exclude_key = str(exclude) if exclude not in (None, "") else None
    ids = [
        str(item["viewpoint_id"])
        for item in ctx.extra.get("viewpoints") or []
        if item.get("reachable", True) and str(item.get("viewpoint_id")) != exclude_key
    ]
    if ids:
        return ids[:limit]
    fresh = _register_viewpoints(
        ctx,
        _build_viewpoints(
            ctx, _yaw(ctx.world), 8, VIEWPOINT_NEAR_M, spread_deg=360.0
        ),
    )
    return [
        item["viewpoint_id"]
        for item in fresh
        if item["reachable"] and item["viewpoint_id"] != exclude_key
    ][:limit]


def _now() -> float:
    """Wall-clock seconds; module-level so a runtime/test can freeze time."""
    return time.time()


def _target_in_fov(world: Any) -> bool | None:
    """Whether the goal target is currently visible (None without the probe)."""
    fn = getattr(world, "target_in_fov", None)
    if not callable(fn):
        return None
    try:
        return bool(fn())
    except Exception:
        return None


def _task_store(ctx: ToolContext) -> dict[str, dict]:
    """``ctx.extra["nav_tasks"]`` repaired into a dict (a runtime may leave
    ``extra`` sparse, a foreign caller may park a non-dict there)."""
    store = ctx.extra.get("nav_tasks")
    if not isinstance(store, dict):
        store = {}
        ctx.extra["nav_tasks"] = store
    return store


def _task_timeout_s(ctx: ToolContext, args: dict) -> float:
    default = NAV_TIMEOUT_S
    try:
        default = float(ctx.extra.get("nav_timeout_s", NAV_TIMEOUT_S))
    except (TypeError, ValueError):
        pass
    return max(0.0, _float_arg(args, "timeout_s", default))


def _task_max_segments(ctx: ToolContext, args: dict) -> int:
    default = NAV_MAX_SEGMENTS
    try:
        default = int(ctx.extra.get("nav_max_segments", NAV_MAX_SEGMENTS))
    except (TypeError, ValueError):
        pass
    # ``max_legs`` is the approach_candidate spelling of the same budget
    return _int_arg(args, "max_segments", _int_arg(args, "max_legs", default, 1, 64), 1, 64)


def _task_expired(task: dict) -> bool:
    try:
        return _now() > float(task.get("deadline", float("inf")))
    except (TypeError, ValueError):
        return False


def _task_summary(task: dict) -> dict:
    streak = int((task.get("repeat_move") or {}).get("streak") or 0)
    return {
        "task_id": task.get("task_id", ""),
        "intent": task.get("intent", ""),
        "status": task.get("status", ""),
        "reason": task.get("reason", ""),
        "progress_m": _round(float(task.get("progress_m", 0.0))),
        "remaining_m": _remaining_value(task),
        "segments": int(task.get("segments", 0)),
        "event_count": len(task.get("events") or []),
        "last_observation_id": task.get("last_observation_id", ""),
        "target_id": _task_target_id(task),
        "repeat_streak": streak,
        "stuck": streak >= REPEAT_MOVE_LIMIT
        or str(task.get("reason") or "") == "repeating_move",
    }


def _task_target_id(task: dict) -> str:
    """The typed id this task's intent flies to (``cand_N`` / ``vp_N`` / ``obs_N``)."""
    return str(
        task.get("candidate_id") or task.get("viewpoint_id") or task.get("observation_id") or ""
    )


def _remaining_value(task: dict):
    """Rounded remaining distance, or None when the range is unknown."""
    value = task.get("remaining_m")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return _round(number) if math.isfinite(number) else None


def _task_data(ctx: ToolContext, task: dict, extra: dict | None = None) -> dict:
    """Structured, model-facing status of one navigation task."""
    now = _now()
    arrived = str(task.get("status", TASK_RUNNING)) == TASK_ARRIVED
    data: dict[str, Any] = {
        "task_id": task.get("task_id", ""),
        "intent": task.get("intent", ""),
        "status": task.get("status", TASK_RUNNING),
        # the same lifecycle status under the movement schema's name, so one
        # payload shape serves task_status and the movement tools' replays
        "task_status": task.get("status", TASK_RUNNING),
        # the completion state, explicit: ``completed`` is the task's own verdict
        # (it reached its place), never the progress of one call
        "completed": bool(arrived),
        "arrived": bool(arrived),
        "reason": task.get("reason", ""),
        "progress_m": _round(float(task.get("progress_m", 0.0))),
        "remaining_m": _remaining_value(task),
        "last_observation_id": task.get("last_observation_id", ""),
        "target_position": list(task.get("target_position") or []),
        "candidate_viewpoints": list(task.get("candidate_viewpoints") or []),
        "segments": int(task.get("segments", 0)),
        "segments_left": max(
            0, int(task.get("max_segments", NAV_MAX_SEGMENTS)) - int(task.get("segments", 0))
        ),
        "timeout_s": _round(float(task.get("timeout_s", NAV_TIMEOUT_S))),
        "moved_m": _round(float(task.get("moved_m", 0.0))),
        "event_count": len(task.get("events") or []),
        "elapsed_s": _round(max(0.0, now - float(task.get("started_at", now)))),
        "pose": _pos_list(_position(ctx.world)),
        "heading_deg": _round(_bias_deg(_yaw(ctx.world))),
    }
    if arrived:
        # arrival: the reachable place plus the trail that says it happened
        data["arrival_evidence"] = [ref.to_dict() for ref in _task_evidence(task)]
    for key in ("candidate_id", "viewpoint_id", "observation_id"):
        value = task.get(key)
        if value:
            data[key] = str(value)
    if task.get("standoff_m") is not None:
        data["standoff_m"] = _round(float(task.get("standoff_m")))
    if extra:
        data.update(extra)
    return data


def _progress_status(outcome: str) -> str:
    """The progress verdict a segment outcome stands for (goal.md section 5)."""
    if outcome in _NO_PROGRESS_OUTCOMES:
        return PROGRESS_NO_PROGRESS
    if outcome in ("collision", "stalled", "planner_failed", "terminated", "obstacle_on_path"):
        return PROGRESS_BLOCKED
    return PROGRESS_OK


def _movement_data(
    ctx: ToolContext,
    task: dict,
    *,
    outcome: str,
    progress_status: str | None = None,
    turn_deg: float = 0.0,
    segment_progress_m: float = 0.0,
    blocked_reason: str = "",
    extra: dict | None = None,
) -> dict:
    """Unified progress payload every movement tool answers with.

    ``status`` is the progress verdict (``ok|blocked|no_progress``), the task
    lifecycle lives in ``task_status`` (also reported by ``task_status``) and the
    leg's own numbers travel together: ``progress_m`` / ``turn_deg`` /
    ``remaining_m`` (None when the range is unknown) / ``blocked_reason`` plus
    the task and id references the model should reuse.
    """
    data = _task_data(ctx, task)
    data["task_status"] = data.pop("status")  # the lifecycle status, not the progress one
    data["status"] = progress_status or _progress_status(outcome)
    data["outcome"] = outcome
    data["turn_deg"] = _round(turn_deg)
    data["segment_progress_m"] = _round(segment_progress_m)
    data["last_observation_id"] = task.get("last_observation_id", "")
    data["blocked_reason"] = str(blocked_reason or "")
    if extra:
        data.update(extra)
    return data


def _task_signature(ctx: ToolContext, task: dict) -> str:
    """State fingerprint of a task for the task_status cache semantics."""
    pose = _pos_list(_position(ctx.world))
    return "|".join(
        str(item)
        for item in (
            task.get("status", ""),
            task.get("reason", ""),
            int(task.get("segments", 0)),
            len(task.get("events") or []),
            _remaining_value(task),
            task.get("last_observation_id", ""),
            ",".join(str(v) for v in pose),
        )
    )


def _repeat_move_state(task: dict) -> dict:
    """The repeat-guard bookkeeping of one task (created on demand)."""
    state = task.get("repeat_move")
    if not isinstance(state, dict):
        state = {"streak": 0, "last_progress_m": None}
        task["repeat_move"] = state
    return state


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _repeat_progress_delta(task: dict, data: dict) -> float:
    """How much closer this leg brought the task (metres).

    ``progress_m`` is cumulative, so the per-call delta is what separates a
    continued approach (8 m, 8 m, ...) from a call that flew nowhere. When the
    range is unknown the tool reports the leg's own displacement
    (``segment_progress_m``) instead; a zero-range step that does not move is
    fruitless even though its status is ``ok``.
    """
    state = _repeat_move_state(task)
    current = _float_or_none(data.get("progress_m"))
    previous = _float_or_none(state.get("last_progress_m"))
    delta = 0.0
    if current is not None and previous is not None:
        delta = current - previous
    segment = _float_or_none(data.get("segment_progress_m"))
    if segment is not None:
        delta = max(delta, segment)
    if current is not None:
        state["last_progress_m"] = current
    return max(0.0, delta)


def _note_move_progress(task: dict, data: dict) -> int:
    """Advance the fruitless-move streak from one finished segment.

    Only a running ``partial`` leg counts: arrival, failure and a blocked
    precheck end the streak. Returns the new streak.
    """
    state = _repeat_move_state(task)
    if (
        str(data.get("task_status") or "") != TASK_RUNNING
        or str(data.get("outcome") or "") != "partial"
    ):
        state["streak"] = 0
        return 0
    if _repeat_progress_delta(task, data) < REPEAT_MOVE_MIN_PROGRESS_M:
        state["streak"] = int(state.get("streak") or 0) + 1
    else:
        state["streak"] = 0
    return int(state["streak"])


def _repeat_move_strategy(ctx: ToolContext, task: dict) -> dict:
    """The strategy change a repeated fruitless move asks for.

    Every option is an executable call of the full tool catalogue. The first
    option is what the result's ``next_action`` names, so a caller can follow it
    unchanged. When the target is already located, the first option goes back to
    that candidate (``approach_candidate``); changing the viewpoint comes only
    after it, because a fresh viewpoint does not move the drone toward a target
    it has already found (the r4 cabinet bottleneck).
    """
    target_id = _task_target_id(task)
    intent = str(task.get("intent") or "")
    observation_id = str(task.get("last_observation_id") or _observation_id(ctx))
    options: list[dict[str, Any]] = []
    located = (
        intent == INTENT_CANDIDATE
        and bool(target_id)
        and _candidate_position(ctx, target_id) is not None
    )
    if located:
        # the target is already located: the first move is to go back to that
        # candidate, not to search somewhere else (r4: the guard sent the agent
        # to new viewpoints and it never returned to the located cabinet)
        options.append(
            {
                "kind": "reapproach",
                "tool": "approach_candidate",
                "args": {"candidate_id": target_id},
                "why": (
                    f"{target_id} is already located: return to it and approach "
                    "it again (a new viewpoint may not even see it) - re-ground "
                    "it from a fresh frame first if its place is stale"
                ),
            }
        )
    if not located:
        options.append(
            {
                "kind": "reground",
                "tool": "ground_target",
                "args": dict(_ground_again_action(ctx, observation_id)["args"]),
                "why": (
                    "re-ground the target from the current frame - a candidate "
                    "without a usable range must be measured again, not flown at"
                ),
            }
        )
    options.append(
        {
            "kind": "change_viewpoint",
            "tool": "propose_viewpoints",
            "args": {"count": 3},
            "why": (
                "only after the located target has been re-approached: propose "
                "another observation point and fly there instead of repeating "
                "the dead leg"
            ),
        }
    )
    if located:
        options.append(
            {
                "kind": "reground",
                "tool": "ground_target",
                "args": dict(_ground_again_action(ctx, observation_id)["args"]),
                "why": (
                    "if the located candidate's place is stale, re-ground it from "
                    "the current frame before approaching it again"
                ),
            }
        )
    if intent == INTENT_CANDIDATE and target_id:
        verify = {"candidate_id": target_id}
    elif observation_id:
        verify = {"observation_id": observation_id}
    else:
        verify = None
    if verify is not None:
        options.append(
            {
                "kind": "verify",
                "tool": "verify_goal",
                "args": verify,
                "why": (
                    "verify what is already known (identity, range or the "
                    "recorded place) before spending another leg on it"
                ),
            }
        )
    options.append(
        {
            "kind": "observe",
            "tool": "observe",
            "args": {"mode": "forward"},
            "why": (
                "take a fresh view before choosing the next target: re-flying "
                "the same dead leg cannot add evidence"
            ),
        }
    )
    return {
        "reason": "repeating_move",
        "kind": "change_strategy",
        "target_key": str(task.get("target_key") or ""),
        "options": options,
        "why": (
            "the same move made no progress three times: return to the located "
            "target and approach it again, or change the viewpoint/re-ground it "
            "before moving again"
        ),
    }


def _repeat_move_guard_result(ctx: ToolContext, task: dict, count: int) -> ToolResult:
    """Refuse the third consecutive fruitless leg and name the strategy change."""
    change = _repeat_move_strategy(ctx, task)
    _repeat_move_state(task)["streak"] = int(count)
    task["strategy_change"] = change
    _set_task_status(
        ctx,
        task,
        TASK_BLOCKED,
        "repeating_move",
        event="repeating_move",
        repeat_count=int(count),
        target_key=str(task.get("target_key") or ""),
    )
    next_action = change["options"][0]
    data = _movement_data(
        ctx,
        task,
        outcome="repeating_move",
        progress_status=PROGRESS_BLOCKED,
        blocked_reason="repeating_move",
        extra={
            "reason": "repeating_move",
            "repeating_move": True,
            "repeat_count": int(count),
            "strategy_change": change,
            "next_action": {
                "tool": next_action["tool"],
                "args": dict(next_action["args"]),
            },
            "hint": change["why"],
        },
    )
    return _task_result(task, STATUS_BLOCKED, data, _task_evidence(task))


def _task_evidence(
    task: dict, evidence: list[EvidenceRef] | None = None, obs_id: str | None = None
) -> list[EvidenceRef]:
    """Observation pose + target reference + map evidence + the task itself."""
    refs: list[EvidenceRef] = []
    obs = obs_id or str(task.get("last_observation_id") or "")
    if obs:
        refs.append(EvidenceRef("observation", obs, note="navigation pose"))
    target_ref = task.get("target_ref")
    if isinstance(target_ref, dict) and target_ref.get("ref"):
        refs.append(
            EvidenceRef(
                str(target_ref.get("kind", "memory")),
                str(target_ref["ref"]),
                note="navigation target",
            )
        )
    refs.extend(evidence or [])
    refs.append(
        EvidenceRef(
            "trajectory",
            str(task.get("task_id", "")),
            note=f"navigation task {task.get('status', '')}",
        )
    )
    return refs


def _event_value(value: Any) -> Any:
    """Event fields stay JSON-plain: floats rounded, numpy scalars unwrapped."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (float, np.floating)):
        return _round(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (list, tuple)):
        return [_event_value(item) for item in value]
    return value


def _emit_task_event(ctx: ToolContext, task: dict, event: str, **fields: Any) -> dict:
    """Queue one execution-layer event of the navigation task (goal.md 5).

    The record is flat - ``{event, task_id, status, progress_m, remaining_m,
    reason, ...}`` - and lives on the task record and in the bounded
    ``ctx.extra["nav_events"]`` log. A ``{"name": ..., "data": {...}}`` mirror is
    appended to ``ctx.extra["events"]``, the queue ``AgentRuntime`` drains after
    every tool call and re-emits through its ``on_event`` hooks, so an event
    produced by the execution layer wakes the model loop instead of being lost.
    """
    now = _now()
    entry: dict[str, Any] = {
        "seq": int(ctx.extra.get("_nav_event_seq", 0)) + 1,
        "event": str(event),
        "task_id": str(task.get("task_id", "")),
        "status": str(task.get("status", "")),
        "progress_m": _round(float(task.get("progress_m", 0.0))),
        # an unknown range stays unknown in an event (the seed-7 guard fires
        # before a range was ever measured)
        "remaining_m": _remaining_value(task),
        "reason": str(task.get("reason", "")),
        "elapsed_s": _round(max(0.0, now - float(task.get("started_at", now)))),
    }
    for key, value in fields.items():
        if value is None:
            continue
        entry[str(key)] = _event_value(value)
    ctx.extra["_nav_event_seq"] = entry["seq"]

    events = task.setdefault("events", [])
    events.append(entry)
    del events[:-TASK_EVENT_CAP]

    log = ctx.extra.get("nav_events")
    if not isinstance(log, list):
        log = []
        ctx.extra["nav_events"] = log
    log.append(dict(entry))
    del log[:-NAV_EVENT_LOG_CAP]

    queue = ctx.extra.get("events")
    if not isinstance(queue, list):
        queue = []
        ctx.extra["events"] = queue
    item = {"name": entry["event"], "data": dict(entry)}
    item.update(entry)
    queue.append(item)
    del queue[:-EVENT_QUEUE_CAP]
    return entry


def _set_task_status(
    ctx: ToolContext, task: dict, status: str, reason: str, event: str | None = None,
    **fields: Any,
) -> dict | None:
    """Move a task to ``status`` (recording the terminal pose) and optionally
    emit the event that explains the move."""
    task["status"] = status
    task["reason"] = reason
    if status in TERMINAL_STATUSES:
        task["terminated_pose"] = _pos_list(_position(ctx.world))
    if event is None:
        return None
    return _emit_task_event(ctx, task, event, **fields)


def _task_result(
    task: dict, status: str, data: dict, evidence: list[EvidenceRef]
) -> ToolResult:
    """Return a task result and remember it for later cached replays."""
    task["last_result"] = dict(data)
    return ToolResult(status, data, evidence)


def _target_key_of(intent: str, intent_id: str) -> str:
    """Identity of one movement request: the intent plus its typed id."""
    return f"{_INTENT_ID_ARG.get(intent, 'target')}:{intent_id}"


def _create_task(
    ctx: ToolContext,
    args: dict,
    *,
    intent: str,
    target_key: str,
    target: np.ndarray | None,
    intent_id: str = "",
    standoff_m: float | None = None,
    restore_yaw: bool = False,
    target_yaw_deg: float | None = None,
) -> dict:
    """Start a movement task of one intent (task bookkeeping is intent-agnostic)."""
    world = ctx.world
    pos = _position(world)
    dist = (
        None
        if target is None
        else float(np.linalg.norm(np.asarray(target, dtype=np.float64) - pos))
    )
    if dist is not None and intent == INTENT_CANDIDATE:
        # an approach aims at the standoff shell: the distance that counts as
        # "the journey" is what is left beyond it, not the surface itself
        dist = max(0.0, dist - skills_approach.world_standoff_m(world, standoff_m))
    timeout = _task_timeout_s(ctx, args)
    now = _now()
    task: dict[str, Any] = {
        "task_id": _next_task_id(ctx),
        "intent": str(intent),
        "status": TASK_RUNNING,
        "reason": "started",
        "target_position": None if target is None else _pos_list(target),
        "target_key": str(target_key),
        "target_ref": None,
        "candidate_id": intent_id if intent == INTENT_CANDIDATE else "",
        "viewpoint_id": intent_id if intent == INTENT_VIEWPOINT else "",
        "observation_id": intent_id if intent == INTENT_OBSERVATION else "",
        "standoff_m": None if standoff_m is None else float(standoff_m),
        "restore_yaw": bool(restore_yaw),
        "target_yaw_deg": target_yaw_deg,
        "started_at": now,
        "deadline": now + timeout,
        "timeout_s": _round(timeout),
        "max_segments": _task_max_segments(ctx, args),
        "segments": 0,
        "start_distance_m": None if dist is None else _round(dist),
        "progress_m": 0.0,
        "remaining_m": None if dist is None else _round(dist),
        "moved_m": 0.0,
        "turn_deg": 0.0,
        "last_observation_id": "",
        "candidate_viewpoints": [],
        "events": [],
        "query_count": 0,
        "last_query_signature": "",
        "map_points": int(len(_known_points(world))),
        "map_version": int(ctx.extra.get("map_version", 0)),
        "target_in_fov": _target_in_fov(world),
        "terminated_pose": None,
        "last_result": None,
    }
    _task_store(ctx)[task["task_id"]] = task
    return task


def _task_with_key(ctx: ToolContext, key: str, statuses: tuple[str, ...]) -> dict | None:
    if not key:
        return None
    for task in reversed(list(_task_store(ctx).values())):
        if str(task.get("target_key", "")) != key:
            continue
        return task if str(task.get("status", "")) in statuses else None
    return None


def _repeat_state(ctx: ToolContext, task: dict) -> dict:
    """What a repeated movement request means: the task's state, not a flight.

    Answering ``already_done`` is what keeps a re-issued request from flying a
    leg the model already flew: the place is reached (the task is ``arrived``),
    nothing moves again, and the answer carries the completion state with its
    arrival evidence. An identical repeat of a *failed* task is likewise not a
    new attempt - the hint names the different call that can work.
    """
    arrived = str(task.get("status", "")) == TASK_ARRIVED
    state = {
        "task_id": task.get("task_id", ""),
        "task_status": task.get("status", ""),
        "completed": bool(arrived),
        "arrived": bool(arrived),
        "already_done": bool(arrived),
        "idempotent": True,
        "reason": task.get("reason", ""),
    }
    if arrived:
        state["arrival_evidence"] = [ref.to_dict() for ref in _task_evidence(task)]
    return state


_REPEAT_HINTS = {
    TASK_ARRIVED: (
        "identical call at the same pose; that place is already reached - do not repeat "
        "it, pick another target or another intent"
    ),
    TASK_BLOCKED: (
        "identical call at the same pose; the leg is blocked - pick one of "
        "candidate_viewpoints (go_to_viewpoint) or re-observe"
    ),
    TASK_FAILED: "identical call at the same pose previously failed; re-observe first",
    TASK_CANCELLED: "this target's task was cancelled at this pose; start a new call if needed",
}

_TERMINAL_HINTS = {
    TASK_ARRIVED: (
        "task already finished: that place is reached - do not repeat this call; fly "
        "somewhere else (go_to_viewpoint / approach_candidate) or verify_goal"
    ),
    TASK_BLOCKED: (
        "task is blocked; pick one of candidate_viewpoints with go_to_viewpoint, or "
        "re-observe and ground_target the target again"
    ),
    TASK_FAILED: "task already failed; re-observe and start a new movement call",
    TASK_CANCELLED: "task was cancelled; start a new movement call if needed",
}

#: hint/next step of a call that cannot move anything - never "call this again"
_ALREADY_AT_POSE_HINT = {
    INTENT_OBSERVATION: (
        "no movement: the drone is already at the pose that observation was captured "
        "from. Do not repeat this call - if you meant the object seen in that frame, "
        "ground it (ground_target) and approach_candidate the cand_N it returns; if "
        "you meant a different place, use its obs_N or a go_to_viewpoint target"
    ),
    INTENT_CANDIDATE: (
        "no movement: the drone is already on the approach standoff of that candidate "
        "- inspect it with verify_goal(candidate_id=...) or pick another candidate; "
        "do not repeat this call"
    ),
    INTENT_VIEWPOINT: (
        "no movement: the drone is already at that viewpoint - propose_viewpoints for "
        "a different observation point or observe forward; do not repeat this call"
    ),
}


def _already_at_pose_action(ctx: ToolContext, task: dict) -> dict:
    """The next call that *can* change something - never the call that just did not."""
    return _underfoot_action(ctx, str(task.get("intent") or ""), _task_target_id(task))


def _underfoot_action(ctx: ToolContext, intent: str, intent_id: str) -> dict:
    """Executable next step when ``intent_id`` is where the drone already is."""
    if intent == INTENT_OBSERVATION:
        return _ground_again_action(ctx, intent_id)
    if intent == INTENT_CANDIDATE:
        return {"tool": "verify_goal", "args": {"candidate_id": intent_id}}
    return {"tool": "propose_viewpoints", "args": {"count": 3}}


def _yaw_delta_rad(current: float, desired: float) -> float:
    """Shortest signed rotation from ``current`` to ``desired`` (rad)."""
    diff = math.atan2(math.sin(desired - current), math.cos(desired - current))
    return float(diff)


def _capture_pose_result(
    ctx: ToolContext, task: dict, kind: str, task_id: str = ""
) -> str:
    obs_id = ctx.next_obs_id()
    _capture_observation(ctx, obs_id, kind, task_id or str(task.get("task_id", "")))
    task["last_observation_id"] = obs_id
    return obs_id


def _already_at_pose_result(
    ctx: ToolContext, task: dict, hint: str = "", plan: Any = None
) -> ToolResult:
    """A call that cannot move anything: ``no_progress`` / ``already_at_pose``.

    The place is reached but the *call* did nothing, so the answer is never
    ``ok`` + ``arrived``: the task records ``arrived`` with the reason
    ``already_at_pose``, the progress verdict is ``no_progress``, the tool result
    is a failure (the runtime counts it as stagnation) and the hint plus
    ``next_action`` name a *different* call.
    """
    obs_id = _current_observation(ctx, OUTCOME_ALREADY_AT_POSE)
    task["last_observation_id"] = obs_id
    _set_task_status(ctx, task, TASK_ARRIVED, OUTCOME_ALREADY_AT_POSE)
    extra: dict[str, Any] = {
        "hint": hint or _ALREADY_AT_POSE_HINT.get(str(task.get("intent")), ""),
        "segment_progress_m": 0.0,
        "turn_deg": 0.0,
    }
    next_action = _already_at_pose_action(ctx, task)
    if next_action:
        extra["next_action"] = next_action
    if plan is not None:
        extra.update(plan.to_data())
    data = _movement_data(
        ctx, task, outcome=OUTCOME_ALREADY_AT_POSE,
        progress_status=PROGRESS_NO_PROGRESS, extra=extra,
    )
    return _task_result(task, STATUS_FAILED, data, _task_evidence(task, obs_id=obs_id))


def _restore_yaw_result(ctx: ToolContext, task: dict, dyaw_rad: float) -> ToolResult:
    """Turn to the heading an observation was captured at (arrival does it too)."""
    world = ctx.world
    yaw0 = _yaw(world)
    action = NavAction(kind="turn", yaw_deg=math.degrees(dyaw_rad), reason="restore_yaw")
    try:
        world.apply_action(action)
    except Exception as exc:  # noqa: BLE001 - a broken world must not kill the runtime
        _set_task_status(
            ctx, task, TASK_FAILED, f"world_error:{type(exc).__name__}", event="task_failed",
        )
        return _task_result(
            task, STATUS_FAILED,
            _movement_data(
                ctx, task, outcome="world_error",
                extra={"hint": "the world raised while restoring the heading"},
            ),
            _task_evidence(task),
        )
    turned = math.degrees(_yaw_delta_rad(yaw0, _yaw(world)))
    task["turn_deg"] = _round(float(task.get("turn_deg", 0.0)) + turned)
    obs_id = _capture_pose_result(ctx, task, "arrived_pose")
    _set_task_status(ctx, task, TASK_ARRIVED, "arrived")
    _map_change_event(ctx, task)
    _target_rediscovery_event(ctx, task, obs_id)
    _emit_task_event(ctx, task, "arrived", observation_id=obs_id, turn_deg=_round(turned))
    _record_arrival(ctx, task, "arrived", _task_evidence(task, obs_id=obs_id))
    data = _movement_data(
        ctx, task, outcome="arrived", progress_status=PROGRESS_OK, turn_deg=turned,
        extra={
            "hint": (
                "reached the observation pose and restored its heading; the frames of "
                "this pose are the ones that observation was taken from"
            ),
        },
    )
    return _task_result(task, STATUS_OK, data, _task_evidence(task, obs_id=obs_id))


def _preexisting_arrival(ctx: ToolContext, task: dict) -> ToolResult | None:
    """Result of a movement call whose target is already reached (or None).

    Only the two fixed-target intents land here (a candidate is judged by the
    approach skill's standoff plan): the drone is on the spot, so the call is a
    ``no_progress`` already_at_pose - unless a ``return_to_observation`` still
    has a heading to restore, which is real work and is executed now.
    """
    intent = str(task.get("intent") or "")
    if intent == INTENT_CANDIDATE:
        return None
    target = _as_position(task.get("target_position"))
    if target is None:
        return None
    if float(np.linalg.norm(target - _position(ctx.world))) > ARRIVE_RADIUS_M:
        return None
    if (
        intent == INTENT_OBSERVATION
        and task.get("restore_yaw")
        and task.get("target_yaw_deg") is not None
    ):
        dyaw = _yaw_delta_rad(_yaw(ctx.world), math.radians(float(task["target_yaw_deg"])))
        if abs(math.degrees(dyaw)) >= YAW_RESTORE_MIN_DEG:
            return _restore_yaw_result(ctx, task, dyaw)
    return _already_at_pose_result(ctx, task)


def _blocked_precheck_result(
    ctx: ToolContext,
    task: dict,
    args: dict,
    reason: str,
    info: dict | None = None,
    hint: str = "",
) -> ToolResult:
    """A leg that never starts: known geometry (or no locatable target)."""
    obs_id = _current_observation(ctx, "blocked_leg")
    task["last_observation_id"] = obs_id
    alternatives = _alternative_ids(ctx, exclude=args.get("viewpoint_id"))
    task["candidate_viewpoints"] = list(alternatives)
    _set_task_status(
        ctx, task, TASK_BLOCKED, reason, event="path_blocked",
        observation_id=obs_id, candidate_viewpoints=alternatives,
    )
    extra: dict[str, Any] = {"segment_progress_m": 0.0}
    if info:
        extra.update(
            {
                "path_cost_m": info.get("path_cost_m"),
                "unknown_ratio": info.get("unknown_ratio"),
                "min_clearance_m": info.get("min_clearance_m"),
            }
        )
    if alternatives:
        extra["next_action"] = {
            "tool": "go_to_viewpoint",
            "args": {"viewpoint_id": alternatives[0]},
        }
    extra["hint"] = hint or (
        "the leg is blocked by known geometry: fly to one of candidate_viewpoints "
        "(go_to_viewpoint) or observe/turn and propose_viewpoints again"
    )
    data = _movement_data(
        ctx, task, outcome="obstacle_on_path", progress_status=PROGRESS_BLOCKED,
        blocked_reason=reason, extra=extra,
    )
    return _task_result(task, STATUS_BLOCKED, data, _task_evidence(task, obs_id=obs_id))


def _segment_plan(
    ctx: ToolContext, task: dict, args: dict
) -> tuple[dict[str, Any] | None, ToolResult | None]:
    """Where this segment flies and how arrival is judged.

    Fixed-target intents (viewpoint / observation) fly to their stored position.
    A candidate goes through ``skills_approach.plan_segment``: the target is
    re-read from the candidate store every segment, the leg ends on the standoff
    shell, and a candidate that cannot be located is a blocked answer instead of
    a flight to a guessed position.
    """
    intent = str(task.get("intent") or INTENT_VIEWPOINT)
    if intent != INTENT_CANDIDATE:
        target = _as_position(task.get("target_position"))
        if target is None:
            _set_task_status(ctx, task, TASK_FAILED, "bad_target_position", event="task_failed")
            data = _movement_data(
                ctx, task, outcome="no_target_position",
                extra={"hint": "the task lost its target position; start a new call"},
            )
            return None, _task_result(task, STATUS_FAILED, data, _task_evidence(task))
        return {"waypoint": target, "arrival": target, "radius": ARRIVE_RADIUS_M}, None

    record = skills_approach.candidate_record(ctx, task.get("candidate_id"))
    target = skills_approach.candidate_target(record, str(task.get("candidate_id") or ""))
    plan = skills_approach.plan_segment(
        ctx, target, task.get("standoff_m"), _max_leg_m(ctx)
    )
    if plan.target_position is not None:
        task["target_position"] = list(plan.target_position)
    if plan.remaining_m is not None:
        task["remaining_m"] = _round(float(plan.remaining_m))
    if plan.blocked_reason:
        return None, _blocked_precheck_result(
            ctx, task, args, plan.blocked_reason, None, hint=plan.hint
        )
    if plan.arrived:
        return None, _already_at_pose_result(ctx, task, plan.hint, plan)
    waypoint = _as_position(plan.waypoint)
    if waypoint is None:
        return None, _blocked_precheck_result(
            ctx, task, args, "no_waypoint", None, hint=plan.hint
        )
    goal_z = _approach_goal_z(ctx.world, target.position)
    if goal_z is not None:
        # the standoff shell is flanked, not overflown: the leg ends at the
        # object's own height band instead of at whatever altitude the drone
        # happened to start the approach from
        waypoint = np.asarray([waypoint[0], waypoint[1], goal_z], dtype=np.float64)
        plan.waypoint = _pos_list(waypoint)
    # the standoff is the approach's goal distance whatever shape the target has
    # (a point or a ray): the arrival judgement and the reported remaining both
    # measure what is left *beyond* it
    standoff = skills_approach.world_standoff_m(ctx.world, task.get("standoff_m"))
    arrival = target.position
    return {
        "waypoint": waypoint,
        "arrival": arrival,
        "radius": skills_approach.ARRIVE_RADIUS_M,
        "standoff": standoff,
        "standoff_m": standoff,
        "plan": plan,
    }, None


def _relocalize_after_leg(ctx: ToolContext, task: dict) -> dict:
    """Re-measure a candidate from the frame the leg just ended in."""
    if str(task.get("intent") or "") != INTENT_CANDIDATE:
        return {}
    record = skills_approach.candidate_record(ctx, task.get("candidate_id"))
    target = skills_approach.candidate_target(record, str(task.get("candidate_id") or ""))
    measurement = skills_approach.relocalize(ctx, target, str(task.get("last_observation_id") or ""))
    if measurement.get("relocalized"):
        task["target_position"] = list(measurement["position_flu"])
    return measurement


def _run_segment(ctx: ToolContext, task: dict, args: dict) -> ToolResult:
    """Execute (at most) one leg of a running task and report the new status."""
    world = ctx.world
    plan_info, blocked = _segment_plan(ctx, task, args)
    if blocked is not None:
        return blocked
    waypoint = _as_position(plan_info["waypoint"])
    pos0 = _position(world)
    yaw0 = _yaw(world)
    dist = float(np.linalg.norm(waypoint - pos0))
    info, evidence = _path_estimate(ctx, waypoint)

    if info["blocked"]:  # known geometry on the leg: plan again, do not move
        return _blocked_precheck_result(
            ctx, task, args, str(info.get("reason") or "obstacle_on_path"), info
        )

    action = _leg_action(pos0, yaw0, waypoint, reason=str(task.get("intent") or "waypoint"))
    try:
        feedback = world.apply_action(action)
    except Exception as exc:  # a broken world must not kill the runtime
        obs_id = _current_observation(ctx, "failed_leg")
        task["last_observation_id"] = obs_id
        _set_task_status(
            ctx, task, TASK_FAILED, f"world_error:{type(exc).__name__}",
            event="task_failed", observation_id=obs_id, error=type(exc).__name__,
        )
        data = _movement_data(
            ctx, task, outcome="world_error",
            blocked_reason=f"world_error:{type(exc).__name__}",
            extra={"hint": "the world raised while executing the leg; re-observe before retrying"},
        )
        return _task_result(task, STATUS_FAILED, data, _task_evidence(task, evidence, obs_id))

    task["segments"] = int(task.get("segments", 0)) + 1
    pos1 = _position(world)
    moved = float(np.linalg.norm(pos1 - pos0))
    turned = math.degrees(_yaw_delta_rad(yaw0, _yaw(world)))
    task["turn_deg"] = _round(float(task.get("turn_deg", 0.0)) + turned)
    obs_id = _capture_pose_result(ctx, task, "post_leg")
    measurement = _relocalize_after_leg(ctx, task)
    task["moved_m"] = _round(float(task.get("moved_m", 0.0)) + moved)

    arrival = _as_position(plan_info.get("arrival"))
    refreshed = _as_position(task.get("target_position"))
    if measurement.get("relocalized") and refreshed is not None:
        # the leg re-measured the candidate: the fresh point is where the drone
        # is heading now, not the point the segment was planned from
        arrival = refreshed
    elif arrival is None:
        arrival = refreshed
    radius = float(plan_info.get("radius", ARRIVE_RADIUS_M))
    standoff = float(plan_info.get("standoff", 0.0) or 0.0)
    distance_to_target = None if arrival is None else float(np.linalg.norm(arrival - pos1))
    # the standoff shell is the goal of an approach: the distance *left to fly*
    # is what remains beyond it, the same quantity the plan reported before the leg
    remaining = (
        None if distance_to_target is None else max(0.0, distance_to_target - standoff)
    )
    if remaining is not None:
        task["remaining_m"] = _round(remaining)
    if task.get("start_distance_m") is None and remaining is not None:
        # the range was unknown when the task started (a ray-only candidate):
        # the journey the drone can see now is what it flew plus what is left
        task["start_distance_m"] = _round(float(task.get("moved_m", 0.0)) + remaining)
    if task.get("start_distance_m") is not None and remaining is not None:
        task["progress_m"] = _round(float(task["start_distance_m"]) - remaining)

    keyword = _outcome_keyword(feedback)
    alternatives: list[str] = []
    if moved < MIN_PROGRESS_M and dist > 1.0:
        # a leg the world refuses to fly is a hard no-op, never progress: the
        # bounds clamp and the control timeout carry their own reasons
        if keyword == "clamped":
            status, reason, outcome = TASK_FAILED, "outside_bound", "outside_bound"
        elif keyword == "leg timeout":
            status, reason, outcome = TASK_FAILED, "leg_timeout", "leg_timeout"
        else:
            status, reason, outcome = TASK_FAILED, "no_progress", "stalled"
    elif keyword in ("collision", "planner failed", "stalled", "terminated"):
        # the world reported a failed leg: it is blocked even if the leg happened
        # to end on the target, the outcome keyword is the flight's own verdict
        status, reason = TASK_BLOCKED, keyword.replace(" ", "_")
        outcome = keyword
    elif remaining is not None and remaining <= radius:
        status, reason, outcome = TASK_ARRIVED, "arrived", "arrived"
    else:
        status, reason, outcome = TASK_RUNNING, "partial", "partial"

    if status == TASK_BLOCKED:
        alternatives = _alternative_ids(ctx, exclude=args.get("viewpoint_id"))
        task["candidate_viewpoints"] = list(alternatives)

    _set_task_status(ctx, task, status, reason)
    _map_change_event(ctx, task)
    _target_rediscovery_event(ctx, task, obs_id)
    if status == TASK_ARRIVED:
        _emit_task_event(ctx, task, "arrived", observation_id=obs_id)
        _record_arrival(ctx, task, outcome, _task_evidence(task, evidence, obs_id))
    elif status == TASK_BLOCKED:
        _emit_task_event(
            ctx, task, "path_blocked", observation_id=obs_id,
            candidate_viewpoints=alternatives,
        )
    elif status == TASK_FAILED:
        _emit_task_event(ctx, task, "task_failed", observation_id=obs_id)

    # arriving at an observation also restores its heading (same call)
    if (
        status == TASK_ARRIVED
        and str(task.get("intent") or "") == INTENT_OBSERVATION
        and task.get("restore_yaw")
        and task.get("target_yaw_deg") is not None
    ):
        dyaw = _yaw_delta_rad(_yaw(world), math.radians(float(task["target_yaw_deg"])))
        if abs(math.degrees(dyaw)) >= YAW_RESTORE_MIN_DEG:
            return _restore_yaw_result(ctx, task, dyaw)

    extra = {
        "path_cost_m": info["path_cost_m"],
        "unknown_ratio": info["unknown_ratio"],
        "min_clearance_m": info["min_clearance_m"],
        "max_leg_m": info["max_leg_m"],
        "segment_progress_m": _round(dist - remaining) if remaining is not None else _round(moved),
    }
    if standoff:
        extra["standoff_m"] = _round(standoff)
    if plan_info.get("waypoint") is not None:
        extra["waypoint"] = _pos_list(waypoint)
    if measurement:
        extra["relocalization"] = measurement
    plan_hint = str(getattr(plan_info.get("plan"), "hint", "") or "")
    if status == TASK_RUNNING:
        running_hint = (
            f"task running; call {task.get('intent')} with task_id="
            f"{task.get('task_id')} for the next segment, or cancel_task to abort"
        )
        extra["hint"] = f"{plan_hint} - {running_hint}" if plan_hint else running_hint
    elif status == TASK_BLOCKED:
        if alternatives:
            extra["next_action"] = {
                "tool": "go_to_viewpoint",
                "args": {"viewpoint_id": alternatives[0]},
            }
        else:
            extra["hint"] = (
                "no reachable viewpoint around the pose; observe/turn and "
                "propose_viewpoints again"
            )
    elif status == TASK_FAILED:
        extra["hint"] = (
            "the leg carried the drone nowhere: re-observe, and either ground the "
            "target again (ground_target -> approach_candidate) or pick a different "
            "viewpoint (go_to_viewpoint) - do not repeat this call"
        )
    data = _movement_data(
        ctx, task, outcome=outcome, turn_deg=turned,
        segment_progress_m=extra["segment_progress_m"],
        blocked_reason=reason if status in (TASK_BLOCKED, TASK_FAILED) else "",
        extra=extra,
    )
    result = _task_result(
        task,
        STATUS_OK if status in (TASK_RUNNING, TASK_ARRIVED) else _SEGMENT_RESULT_STATUS[status],
        data,
        _task_evidence(task, evidence, obs_id),
    )
    _note_move_progress(task, data)
    return result


def _continue_task(
    ctx: ToolContext, task: dict, args: dict, note: str | None = None
) -> ToolResult:
    """Fly the next segment of an existing task, or replay its final state."""
    status = str(task.get("status", TASK_RUNNING))
    if status in TERMINAL_STATUSES:
        data = dict(task.get("last_result") or _task_data(ctx, task))
        data["hint"] = _TERMINAL_HINTS.get(status, f"task already {status}")
        data.update(_repeat_state(ctx, task))
        if note:
            data.setdefault("note", note)
        return ToolResult(STATUS_CACHED, data, _task_evidence(task))

    if _task_expired(task):
        _set_task_status(
            ctx, task, TASK_FAILED, "timeout", event="task_failed",
            timeout_s=task.get("timeout_s"),
        )
        data = _movement_data(
            ctx, task, outcome="timeout", blocked_reason="timeout",
            extra={"hint": "the task exceeded timeout_s; start a new movement call to retry"},
        )
        return _task_result(task, STATUS_FAILED, data, _task_evidence(task))

    max_segments = int(task.get("max_segments", NAV_MAX_SEGMENTS))
    if int(task.get("segments", 0)) >= max_segments:
        _set_task_status(
            ctx, task, TASK_FAILED, "segment_budget", event="task_failed",
            max_segments=max_segments,
        )
        data = _movement_data(
            ctx, task, outcome="segment_budget", blocked_reason="segment_budget",
            extra={"hint": "the task used its segment budget; start a new movement call"},
        )
        return _task_result(task, STATUS_FAILED, data, _task_evidence(task))

    streak = int(_repeat_move_state(task).get("streak") or 0)
    if streak >= REPEAT_MOVE_LIMIT - 1:
        # this call would be the third consecutive fruitless leg: do not fly
        # it, answer with the strategy change instead
        return _repeat_move_guard_result(ctx, task, streak + 1)

    result = _run_segment(ctx, task, args)
    if note:
        result.data["note"] = note
        if isinstance(task.get("last_result"), dict):
            task["last_result"]["note"] = note
    return result


def _start_or_continue(
    ctx: ToolContext,
    args: dict,
    *,
    intent: str,
    intent_id: str,
    target: np.ndarray | None,
    standoff_m: float | None = None,
    restore_yaw: bool = False,
    target_yaw_deg: float | None = None,
    target_ref: EvidenceRef | None = None,
) -> ToolResult:
    """Shared body of every movement tool: continue, replay or start a task.

    ``intent`` decides the tool name in the answers (and the intent a ``task_id``
    must belong to), the typed ``intent_id`` keys the task so an identical
    request continues the running task and a repeat at the same pose is answered
    ``cached`` instead of flying the leg again.
    """
    store = _task_store(ctx)
    task_id_arg = args.get("task_id") or args.get("nav_task_id")
    if task_id_arg not in (None, ""):
        task = store.get(str(task_id_arg))
        if task is None:
            return ToolResult(
                STATUS_FAILED,
                {
                    "reason": "unknown_task",
                    "task_id": str(task_id_arg),
                    "known_tasks": sorted(store),
                    "hint": "call task_status to list the navigation tasks",
                },
            )
        task_intent = str(task.get("intent") or "")
        if task_intent and task_intent != intent:
            return ToolResult(
                STATUS_FAILED,
                {
                    "reason": "task_intent_mismatch",
                    "task_id": str(task_id_arg),
                    "task_intent": task_intent,
                    "intent": intent,
                    "hint": (
                        f"{task_id_arg} is a {task_intent} task: continue it with "
                        f"{task_intent}({{\"task_id\": \"{task_id_arg}\"}})"
                    ),
                },
            )
        note = None
        passed = args.get(_INTENT_ID_ARG.get(intent, ""))
        if passed not in (None, "") and str(passed) != _task_target_id(task):
            note = "task_id given; the id of this call is ignored"
        return _continue_task(ctx, task, args, note=note)

    target_key = _target_key_of(intent, intent_id)
    running = _task_with_key(ctx, target_key, (TASK_RUNNING,))
    if running is not None:  # the model re-issued the target: keep the same task
        return _continue_task(
            ctx, running, args, note=f"continued running task {running['task_id']}"
        )

    repeat = _task_with_key(ctx, target_key, TERMINAL_STATUSES)
    if repeat is not None and repeat.get("terminated_pose") == _pos_list(_position(ctx.world)):
        data = dict(repeat.get("last_result") or _task_data(ctx, repeat))
        data["hint"] = _REPEAT_HINTS.get(
            str(repeat.get("status")),
            f"identical {intent} call repeated; task is already {repeat.get('status')}",
        )
        data.update(_repeat_state(ctx, repeat))
        return ToolResult(STATUS_CACHED, data, _task_evidence(repeat))

    task = _create_task(
        ctx, args, intent=intent, target_key=target_key, target=target, intent_id=intent_id,
        standoff_m=standoff_m, restore_yaw=restore_yaw, target_yaw_deg=target_yaw_deg,
    )
    if target_ref is not None:
        task["target_ref"] = target_ref.to_dict()

    arrival = _preexisting_arrival(ctx, task)
    if arrival is not None:
        return arrival
    return _run_segment(ctx, task, args)


# ---------------------------------------------------------------------------
# movement tools: one per intent (goal.md section 4)
# ---------------------------------------------------------------------------


def _id_error_result(ctx: ToolContext, exc: IdTypeError, args: dict) -> ToolResult:
    """Tool result of a typed id that is missing, wrong-typed or unknown.

    Beyond the type/availability payload ``IdTypeError`` carries, the answer
    names the call that *can* work next: an id of the wrong type is handed back
    under the tool of *its own* intent (an ``obs_N`` given to
    ``approach_candidate`` becomes a ``return_to_observation`` call), and an id
    that simply does not exist points at the call that produces one.
    """
    data = exc.to_data()
    data["reason"] = exc.error_name()
    if not data.get("hint"):
        data["hint"] = exc.hint or "use an id a tool of this episode returned"
    for key in (exc.expected, exc.expected.replace("_id", "")):
        if args.get(key) not in (None, ""):
            data.setdefault(key, str(args.get(key)))
    next_action = _id_next_action(ctx, exc, str(data.get("got") or ""))
    if next_action:
        data["next_action"] = next_action
    return ToolResult(STATUS_FAILED, data)


def _id_next_action(ctx: ToolContext, exc: IdTypeError, raw: str) -> dict | None:
    """The executable call a typed-id failure implies (never the same call)."""
    if exc.reason == "id_type_mismatch":
        if exc.actual == "observation_id" and raw:
            return {"tool": INTENT_OBSERVATION, "args": {"observation_id": raw}}
        if exc.actual == "viewpoint_id" and raw:
            return {"tool": INTENT_VIEWPOINT, "args": {"viewpoint_id": raw}}
        if exc.actual == "candidate_id" and raw:
            return {"tool": INTENT_CANDIDATE, "args": {"candidate_id": raw}}
        return None
    if exc.expected == "observation_id":
        if not exc.available:
            return {"tool": "observe", "args": {"mode": "forward"}}
        return {"tool": INTENT_OBSERVATION, "args": {"observation_id": exc.available[-1]}}
    if exc.expected == "candidate_id":
        return _ground_again_action(ctx, _observation_id(ctx))
    return {"tool": "propose_viewpoints", "args": {"count": 3}}


def _tool_go_to_viewpoint(ctx: ToolContext, args: dict) -> ToolResult:
    """Fly to an observation *position* (``vp_N``) proposed earlier."""
    try:
        viewpoint_id = require_viewpoint_id(
            args.get("viewpoint_id", args.get("target_id")), ctx
        )
    except IdTypeError as exc:
        return _id_error_result(ctx, exc, args)
    entry = (ctx.extra.get("viewpoints_by_id") or {}).get(viewpoint_id) or {}
    target = _viewpoint_position(ctx, viewpoint_id)
    if target is None:
        return ToolResult(
            STATUS_FAILED,
            {
                "reason": "viewpoint_has_no_position",
                "viewpoint_id": viewpoint_id,
                "available_viewpoints": list(ctx.extra.get("viewpoints_by_id") or {}),
                "hint": "re-run propose_viewpoints: this viewpoint carries no position",
            },
        )
    return _start_or_continue(
        ctx,
        args,
        intent=INTENT_VIEWPOINT,
        intent_id=viewpoint_id,
        target=np.asarray(target, dtype=np.float64),
        target_ref=EvidenceRef("memory", f"vp:{viewpoint_id}", note="proposed viewpoint"),
    )


def _tool_return_to_observation(ctx: ToolContext, args: dict) -> ToolResult:
    """Fly back to the *pose* a frame was captured at (and restore its heading)."""
    try:
        observation_id = require_observation_id(
            args.get("observation_id", args.get("obs_id")), ctx
        )
    except IdTypeError as exc:
        return _id_error_result(ctx, exc, args)
    pose = resolve_observation_pose(ctx, observation_id)
    if pose is None or _as_position(pose[0]) is None:
        return ToolResult(
            STATUS_FAILED,
            {
                "reason": "observation_pose_unknown",
                "observation_id": observation_id,
                "available_observations": list(observation_poses(ctx)),
                "hint": (
                    "no store knows where that frame was taken: call observe (or "
                    "query_memory) and use an id of a frame whose pose is recorded"
                ),
            },
        )
    position = _as_position(pose[0])
    yaw_rad = pose[1]
    restore_yaw = args.get("restore_yaw", True)
    restore_yaw = True if restore_yaw is None else bool(restore_yaw)
    return _start_or_continue(
        ctx,
        args,
        intent=INTENT_OBSERVATION,
        intent_id=observation_id,
        target=position,
        restore_yaw=restore_yaw,
        target_yaw_deg=None if yaw_rad is None else math.degrees(float(yaw_rad)),
        target_ref=EvidenceRef(
            "observation", observation_id, note="recorded capture pose"
        ),
    )


def _tool_approach_candidate(ctx: ToolContext, args: dict) -> ToolResult:
    """Approach a target *candidate* (``cand_N``) through the approach skill."""
    try:
        candidate_id = require_candidate_id(
            args.get("candidate_id", args.get("cand_id")), ctx
        )
    except IdTypeError as exc:
        return _id_error_result(ctx, exc, args)
    record = skills_approach.candidate_record(ctx, candidate_id)
    target_info = skills_approach.candidate_target(record, candidate_id)
    if not target_info.direction_known:
        resolved, unresolved = _candidate_ids(ctx)
        data: dict[str, Any] = {
            "reason": "candidate_has_no_position",
            "candidate_id": candidate_id,
            "status": PROGRESS_BLOCKED,
            "outcome": "unlocatable_candidate",
            "blocked_reason": str(target_info.source or "no_valid_depth"),
            "progress_m": 0.0,
            "turn_deg": 0.0,
            "remaining_m": None,
            "task_id": "",
            "last_observation_id": _observation_id(ctx),
            "hint": (
                "that candidate carries neither a position nor a ray: observe forward "
                "from a fresh pose and call ground_target again, then approach the new "
                "cand_N"
            ),
            "next_action": _ground_again_action(ctx, target_info.observation_id),
            "available_targets": resolved,
            "unresolved_targets": unresolved,
            "box_format": "xyxy",
        }
        if target_info.box is not None:
            data["box"] = [round(float(v), 1) for v in target_info.box]
        return ToolResult(STATUS_BLOCKED, data)
    standoff = args.get("standoff_m")
    standoff = (
        None
        if standoff is None
        else _float_arg(args, "standoff_m", skills_approach.APPROACH_STANDOFF_M)
    )
    result = _start_or_continue(
        ctx,
        args,
        intent=INTENT_CANDIDATE,
        intent_id=candidate_id,
        target=target_info.position,
        standoff_m=standoff,
        target_ref=EvidenceRef("memory", candidate_id, note="target candidate"),
    )
    # the candidate's own geometry (source, range, box) is part of the answer:
    # the model should not have to re-read ground_target to know what it flies at
    for key, value in target_info.to_data().items():
        result.data.setdefault(key, value)
    return result


#: one replacement call per intent, for the deprecation answer
_INTENT_TOOL_HINT = (
    "approach_candidate({\"candidate_id\": \"cand_N\"}) flies to the OBJECT in a "
    "frame, go_to_viewpoint({\"viewpoint_id\": \"vp_N\"}) flies to an observation "
    "POSITION, return_to_observation({\"observation_id\": \"obs_N\"}) flies back to "
    "the CAMERA POSE a frame was captured at and restores its heading"
)


def _deprecated_navigate_result(ctx: ToolContext, args: dict) -> ToolResult:
    """Answer a legacy ``navigate`` call: an explicit error + the replacement call.

    ``navigate`` mixed three different intents and produced the zero-displacement
    loop the intent split removes, so it never executes anything. The answer
    still resolves what the call *meant* - the replacement call, its arguments
    and (when it resolves) the target position - so a caller written against the
    old name can migrate in one step instead of guessing a new id.
    """
    data: dict[str, Any] = {
        "reason": "deprecated_tool",
        "error": "deprecated_navigate",
        "status": PROGRESS_NO_PROGRESS,
        "outcome": "deprecated_call",
        "progress_m": 0.0,
        "turn_deg": 0.0,
        "remaining_m": None,
        "hint": (
            "navigate was split by intent and executed nothing: " + _INTENT_TOOL_HINT
        ),
        "intent_tools": list(MOVE_INTENTS),
    }
    replacement: dict[str, Any] | None = None
    task_id_arg = args.get("task_id") or args.get("nav_task_id")
    task = _task_store(ctx).get(str(task_id_arg)) if task_id_arg not in (None, "") else None
    if task is not None:
        data["task_id"] = str(task_id_arg)
        data["task_intent"] = str(task.get("intent") or "")
        if data["task_intent"] in MOVE_INTENTS:
            replacement = {
                "tool": data["task_intent"],
                "args": {
                    _INTENT_ID_ARG[data["task_intent"]]: _task_target_id(task),
                    "task_id": str(task_id_arg),
                },
            }
    if replacement is None:
        for key, intent in (
            ("candidate_id", INTENT_CANDIDATE),
            ("viewpoint_id", INTENT_VIEWPOINT),
            ("observation_id", INTENT_OBSERVATION),
            ("obs_id", INTENT_OBSERVATION),
        ):
            value = args.get(key)
            if value in (None, ""):
                continue
            replacement = {"tool": intent, "args": {_INTENT_ID_ARG[intent]: str(value)}}
            break
    if replacement is None:
        target_value = args.get("target_id")
        if target_value not in (None, ""):
            text = str(target_value)
            intent = (
                INTENT_OBSERVATION if _looks_like_observation_id(text)
                else INTENT_CANDIDATE if text.lower().startswith(("cand_", "cand#", "cand:"))
                else INTENT_VIEWPOINT
            )
            replacement = {"tool": intent, "args": {_INTENT_ID_ARG[intent]: text}}
    if replacement is None and args.get("position") is not None:
        requested_position = _as_position(args.get("position"))
        data["target_position"] = _pos_list(
            requested_position if requested_position is not None else np.zeros(3)
        )
        replacement = {"tool": "propose_viewpoints", "args": {"count": 3}}
        data["hint"] = (
            "navigate was split by intent and executes nothing; a raw position is no "
            "longer a movement target - propose_viewpoints around it and fly with "
            "go_to_viewpoint. " + _INTENT_TOOL_HINT
        )
    if replacement is None:
        replacement = {"tool": "observe", "args": {"mode": "forward"}}
        data["hint"] = (
            "navigate was split by intent and executes nothing, and this call named no "
            "usable id: pass candidate_id / viewpoint_id / observation_id. "
            + _INTENT_TOOL_HINT
        )
    data["next_action"] = replacement
    data["replacement"] = dict(replacement)

    # resolve the legacy target so the caller can see *where* the replacement goes
    intent = str(replacement.get("tool"))
    id_key = _INTENT_ID_ARG.get(intent, "")
    id_value = replacement.get("args", {}).get(id_key)
    if id_key and id_value:
        data[id_key] = str(id_value)
    for key in _TARGET_ID_KEYS:
        value = args.get(key)
        if value not in (None, ""):
            data.setdefault(key, str(value))
            break
    requested = data.get("target_id") or data.get("candidate_id")
    if intent == INTENT_OBSERVATION and id_value:
        pose = resolve_observation_pose(ctx, id_value)
        if pose is not None and _as_position(pose[0]) is not None:
            data["target_position"] = _pos_list(_as_position(pose[0]))
            data["suggested_position"] = list(data["target_position"])
    elif intent == INTENT_VIEWPOINT and id_value:
        position = _viewpoint_position(ctx, id_value)
        if position is not None:
            data["target_position"] = _pos_list(position)
    elif intent == INTENT_CANDIDATE and id_value:
        position = _candidate_position(ctx, id_value)
        if position is not None:
            data["target_position"] = _pos_list(position)
    if not data.get("available_targets"):
        resolved, unresolved = _candidate_ids(ctx)
        data["available_targets"] = resolved
        data["unresolved_targets"] = unresolved
    data["available_observations"] = list(observation_poses(ctx))
    data["available_viewpoints"] = list(ctx.extra.get("viewpoints_by_id") or {})
    # a replacement that cannot move anything is not a next step: when the
    # resolved target is where the drone already is, name a call that can
    resolved_position = _as_position(data.get("target_position"))
    if (
        resolved_position is not None
        and intent in (INTENT_OBSERVATION, INTENT_VIEWPOINT)
        and float(np.linalg.norm(resolved_position - _position(ctx.world))) <= ARRIVE_RADIUS_M
    ):
        replacement = _underfoot_action(ctx, intent, str(id_value or ""))
        data["next_action"] = replacement
        data["replacement"] = dict(replacement)
        data["hint"] = (
            f"navigate was split by intent, and that {id_key or 'target'} is where the "
            "drone already is (flying there would move nothing): " + str(data["hint"])
        )
    if requested and requested in data.get("unresolved_targets", []):
        data["hint"] = f"{_UNRESOLVED_TARGET_HINT}. " + str(data["hint"])
    return ToolResult(STATUS_FAILED, data)


def _map_change_event(ctx: ToolContext, task: dict) -> dict | None:
    """``map_changed`` when the observed map grew significantly (goal.md 5)."""
    points_now = int(len(_known_points(ctx.world)))
    points_before = int(task.get("map_points", 0))
    threshold = max(MAP_CHANGE_MIN_POINTS, MAP_CHANGE_REL * max(0, points_before))
    if points_now - points_before < threshold:
        return None
    version = int(ctx.extra.get("map_version", 0)) + 1
    ctx.extra["map_version"] = version
    task["map_points"] = points_now
    task["map_version"] = version
    return _emit_task_event(
        ctx,
        task,
        "map_changed",
        points_before=points_before,
        points_after=points_now,
        map_version=version,
    )


def _target_rediscovery_event(ctx: ToolContext, task: dict, obs_id: str) -> dict | None:
    """``target_rediscovered`` on a False -> True target-in-FOV transition.

    Only the transition is reported - never the target position or distance,
    which stay behind ``verify_goal``, the explicit completion check: the event
    tells the model to look again, not where the goal is.
    """
    fov_now = _target_in_fov(ctx.world)
    if fov_now is None:
        return None
    was = task.get("target_in_fov")
    task["target_in_fov"] = fov_now
    if fov_now and was is False:
        return _emit_task_event(
            ctx,
            task,
            "target_rediscovered",
            observation_id=obs_id,
            hint="target suspected visible again; run verify_goal to confirm",
        )
    return None




# ---------------------------------------------------------------------------
# task_status / cancel_task: query and interrupt a navigation task (goal.md 5)
# ---------------------------------------------------------------------------


def _task_status_data(ctx: ToolContext, task: dict, limit: int) -> dict:
    events = [dict(item) for item in (task.get("events") or [])]
    data = _task_data(ctx, task)
    state = _repeat_move_state(task)
    streak = int(state.get("streak") or 0)
    data.update(
        {
            "events": events[-limit:] if limit else [],
            "event_count": len(events),
            "latest_event": events[-1] if events else None,
            "query_count": int(task.get("query_count", 0)),
            "active": task.get("status") == TASK_RUNNING,
            "timeout_s": task.get("timeout_s"),
            # the repeat guard's own state: a task that keeps answering
            # ``partial`` without progress is stuck, not merely running
            "repeat_streak": streak,
            "repeat_count": streak,
            "stuck": streak >= REPEAT_MOVE_LIMIT
            or str(task.get("reason") or "") == "repeating_move",
        }
    )
    change = task.get("strategy_change")
    if isinstance(change, dict):
        data["strategy_change"] = dict(change)
    return data


def _tool_task_status(ctx: ToolContext, args: dict) -> ToolResult:
    store = _task_store(ctx)
    limit = _int_arg(args, "limit", 8, 1, 32)
    task_id = args.get("task_id") or args.get("nav_task_id")

    if task_id in (None, "", "all", "*"):  # fleet view: every task at a glance
        if not store:
            return ToolResult(
                STATUS_EMPTY,
                {
                    "tasks": [],
                    "count": 0,
                    "active": [],
                    "hint": "no navigation task yet; start one with go_to_viewpoint or approach_candidate",
                },
            )
        signature = "||".join(
            _task_signature(ctx, task) for task in store.values()
        )
        signature = f"{signature}|limit={limit}"
        cached = signature == ctx.extra.get("_nav_status_sig")
        ctx.extra["_nav_status_sig"] = signature
        latest = list(store.values())[-1]
        data = {
            "count": len(store),
            "tasks": [_task_summary(task) for task in store.values()],
            "active": [
                task["task_id"]
                for task in store.values()
                if task.get("status") == TASK_RUNNING
            ],
            "events": [dict(item) for item in (ctx.extra.get("nav_events") or [])][-limit:],
        }
        if cached:
            data["hint"] = "no navigation state change since the last task_status query"
            return ToolResult(STATUS_CACHED, data, _task_evidence(latest))
        return ToolResult(STATUS_OK, data, _task_evidence(latest))

    task = store.get(str(task_id))
    if task is None:
        return ToolResult(
            STATUS_FAILED,
            {
                "reason": "unknown_task",
                "task_id": str(task_id),
                "known_tasks": sorted(store),
                "hint": "call task_status without task_id to list every task",
            },
        )

    signature = f"{_task_signature(ctx, task)}|limit={limit}"
    if signature == task.get("last_query_signature"):
        data = _task_status_data(ctx, task, limit)
        intent = str(task.get("intent") or "go_to_viewpoint")
        data["hint"] = (
            f"no change since the last task_status query (status={task.get('status')}); "
            f"call {intent} with task_id to continue"
        )
        return ToolResult(STATUS_CACHED, data, _task_evidence(task))

    task["last_query_signature"] = signature
    task["query_count"] = int(task.get("query_count", 0)) + 1
    data = _task_status_data(ctx, task, limit)
    intent = str(task.get("intent") or "go_to_viewpoint")
    if data.get("stuck") or str(task.get("reason") or "") == "repeating_move":
        data.setdefault(
            "hint",
            "task is stuck (repeating_move): the same leg made no progress three "
            "times. Change the strategy - propose_viewpoints/go_to_viewpoint for "
            "another place, verify_goal, observe a fresh view, or re-ground the "
            "target - instead of repeating the call",
        )
    elif task.get("status") == TASK_RUNNING:
        data.setdefault(
            "hint",
            f"task running; call {intent} with task_id for the next segment, or "
            "cancel_task to abort",
        )
    elif task.get("status") == TASK_BLOCKED:
        data.setdefault(
            "hint",
            f"blocked; pick one of candidate_viewpoints with go_to_viewpoint, or "
            "re-observe and ground_target the target again",
        )
    elif str(task.get("reason") or "") == OUTCOME_ALREADY_AT_POSE:
        data.setdefault(
            "hint",
            "the call that produced this task could not move anything "
            "(already_at_pose): " + _ALREADY_AT_POSE_HINT.get(intent, ""),
        )
    return ToolResult(STATUS_OK, data, _task_evidence(task))


def _tool_cancel_task(ctx: ToolContext, args: dict) -> ToolResult:
    store = _task_store(ctx)
    task_id = args.get("task_id") or args.get("nav_task_id")
    if task_id in (None, ""):
        return ToolResult(
            STATUS_FAILED,
            {
                "reason": "missing_task_id",
                "known_tasks": sorted(store),
                "hint": "pass task_id (task_status lists every task)",
            },
        )
    task = store.get(str(task_id))
    if task is None:
        return ToolResult(
            STATUS_FAILED,
            {
                "reason": "unknown_task",
                "task_id": str(task_id),
                "known_tasks": sorted(store),
                "hint": "call task_status to list the navigation tasks",
            },
        )
    status = str(task.get("status", TASK_RUNNING))
    if status in TERMINAL_STATUSES:
        # the *task* schema, not the last movement payload: a cancel answer is
        # about the task, and the last leg's numbers travel along as context
        data = _task_data(ctx, task)
        last = task.get("last_result") if isinstance(task.get("last_result"), dict) else {}
        for key in ("outcome", "turn_deg", "blocked_reason", "next_action", "relocalization"):
            if key in last:
                data[key] = last[key]
        data["hint"] = f"task already {status}; nothing to cancel"
        return ToolResult(STATUS_CACHED, data, _task_evidence(task))

    _set_task_status(
        ctx, task, TASK_CANCELLED, str(args.get("reason") or "cancelled"),
        event="task_cancelled",
    )
    data = _task_data(
        ctx, task,
        {"hint": "task cancelled; propose_viewpoints or approach_candidate a target"},
    )
    return _task_result(task, STATUS_OK, data, _task_evidence(task))


# ---------------------------------------------------------------------------
# verify_goal
# ---------------------------------------------------------------------------


def _resolve_pixel(args: dict) -> tuple[Any, str | None]:
    """(pixel or None, error) from {pixel|region}."""
    pixel = args.get("pixel")
    region = args.get("region")
    if pixel is None and region is not None:
        if isinstance(region, str):
            key = region.strip().lower()
            if key not in REGION_PIXELS:
                return None, f"unknown_region:{region}"
            pixel = REGION_PIXELS[key]
        else:
            pixel = region
    if pixel is None:
        return None, None
    resolved = _as_pixel(pixel)
    if resolved is None:
        return None, "bad_pixel"
    return resolved, None


def _tool_verify_goal(ctx: ToolContext, args: dict) -> ToolResult:
    world = ctx.world
    pixel, error = _resolve_pixel(args)
    if error is not None:
        return ToolResult(STATUS_FAILED, {"reason": error})

    evidence: list[EvidenceRef] = []
    unmet: list[dict] = []
    checks: dict[str, Any] = {}

    target = getattr(world, "target", None)
    target_pos = _as_position(getattr(target, "position_flu", None))
    radius = getattr(target, "radius_m", 0.5)
    try:
        radius = float(radius)
    except (TypeError, ValueError):
        radius = 0.5

    dist_fn = getattr(world, "distance_to_target", None)
    dist = None
    if callable(dist_fn):
        try:
            dist = float(dist_fn())
        except Exception:
            dist = None
    if dist is None and target_pos is not None:
        dist = float(np.linalg.norm(target_pos - _position(world)))
    # ``done_dist`` is the gate the episode is judged with (the runner's
    # ``--success-dist + radius``); without a caller gate this tool's own
    # default applies, which is only a fallback - not the evaluation gate.
    limit = ctx.extra.get("done_dist", max(1.5, radius + 0.5))
    if args.get("success_dist_m") is not None:
        limit = _float_arg(args, "success_dist_m", limit)
    try:
        limit = float(limit)
    except (TypeError, ValueError):
        limit = max(1.5, radius + 0.5)

    # -- pixel -> world point (only trusted when the world has real depth) ----
    depth = None
    point = None
    if pixel is not None:
        depth_fn = getattr(world, "depth_at_pixel", None)
        if callable(depth_fn):
            try:
                depth = depth_fn(pixel)
            except Exception:
                depth = None
        if depth is not None:
            # no depth fallback of its own: with a real depth read the pointed
            # point is a measurement (``estimate_point_world`` returns None rather
            # than inventing a range when there is none, vlm_nav/world.py).
            # The pixel is un-projected with the *frame's own* intrinsics - the
            # observation it was read in, or what the world declares - and when
            # neither exists the answer is "cannot localize", reported as such
            # instead of projected with an assumed field of view (audit I1)
            intrinsics = skills_approach.current_intrinsics(ctx, _observation_id(ctx))
            point = estimate_point_world(world, pixel, 0.0, intrinsics=intrinsics)
            checks["depth_m"] = _round(float(depth))
            if point is not None:
                checks["point_world"] = _pos_list(point)
            else:
                checks["point_localization"] = (
                    "intrinsics_unknown"
                    if intrinsics.get("hfov_deg") is None
                    else "no_range"
                )
            obs_id = _current_observation(ctx, "verify_goal")
            evidence.append(
                EvidenceRef(
                    "observation",
                    obs_id,
                    note=(
                        f"depth {float(depth):.2f} m at pixel "
                        f"({pixel[0]:.0f},{pixel[1]:.0f})"
                    ),
                )
            )

    # -- independent checks ---------------------------------------------------
    hard_fail = False
    fov_fn = getattr(world, "target_in_fov", None)
    if callable(fov_fn):
        try:
            in_fov = bool(fov_fn())
        except Exception:
            in_fov = None
        if in_fov is not None:
            checks["target_in_fov"] = in_fov
            if not in_fov:
                unmet.append({"check": "target_in_fov", "reason": "target_not_visible"})
                hard_fail = True

    if dist is not None:
        checks["distance_to_target_m"] = _round(dist)
        # the limit the verdict was taken against: the caller's gate
        # (``ctx.extra["done_dist"]``, i.e. the runner's ``--success-dist +
        # radius``) when it set one, else this tool's own default. Reported, so
        # an experiment can check that the completion check and the evaluation
        # gate really are the same number.
        checks["distance_limit_m"] = _round(limit)
        if dist > limit:
            unmet.append(
                {
                    "check": "distance_to_target_m",
                    "value_m": _round(dist),
                    "max_m": _round(limit),
                }
            )
            hard_fail = True

    if pixel is not None and depth is None:
        unmet.append({"check": "depth", "reason": "no_depth_estimate"})

    candidate_id = args.get("candidate_id")
    if candidate_id not in (None, ""):
        cand_pos = _candidate_position(ctx, candidate_id)
        if cand_pos is None:
            unmet.append(
                {
                    "check": "candidate_identity",
                    "reason": "unknown_candidate",
                    "candidate_id": str(candidate_id),
                }
            )
        else:
            key = str(candidate_id)
            evidence.append(
                EvidenceRef(
                    "memory",
                    key if not key.isdigit() else f"cand_{key}",
                    note="candidate position",
                )
            )
            if point is not None:
                offset = float(np.linalg.norm(point - cand_pos))
                checks["candidate_offset_m"] = _round(offset)
                if offset > 1.0:
                    unmet.append(
                        {
                            "check": "candidate_identity",
                            "offset_m": _round(offset),
                            "max_m": 1.0,
                        }
                    )
                    hard_fail = True

    if point is not None and target_pos is not None:
        offset = float(np.linalg.norm(point - target_pos))
        checks["point_to_target_m"] = _round(offset)
        tolerance = max(radius + 1.0, 1.5)
        if offset > tolerance:
            unmet.append(
                {
                    "check": "point_on_target",
                    "offset_m": _round(offset),
                    "max_m": _round(tolerance),
                }
            )
            hard_fail = True

    # -- a recalled observation id is verification evidence too ----------------
    # "I saw the target at obs_5" is a checkable claim: when that place is the
    # goal, the verification passes; when it is not, the answer says so and
    # names the call that gets the drone there (never a silent success).
    observation_id = args.get("observation_id", args.get("obs_id"))
    if observation_id in (None, "") and _looks_like_observation_id(args.get("target_id")):
        observation_id = args.get("target_id")
    unresolved_observation = False
    if observation_id not in (None, ""):
        recalled = resolve_observation_pose(ctx, observation_id)
        if recalled is None:
            # the cited evidence does not exist: the verdict may not come back
            # as a pass (a finish would then claim success on a phantom id)
            unresolved_observation = True
            unmet.append(
                {
                    "check": "observation_pose",
                    "reason": "unknown_observation",
                    "observation_id": str(observation_id),
                }
            )
        else:
            recalled_pos = np.asarray(recalled[0], dtype=np.float64)
            checks["recalled_observation_position"] = _pos_list(recalled_pos)
            evidence.append(
                EvidenceRef(
                    "observation",
                    str(observation_id),
                    note="recalled observation pose",
                )
            )
            if target_pos is not None:
                offset = float(np.linalg.norm(recalled_pos - target_pos))
                checks["recalled_observation_to_target_m"] = _round(offset)
                tolerance = max(radius + 1.0, 1.5)
                if offset > tolerance:
                    unmet.append(
                        {
                            "check": "recalled_observation_on_target",
                            "offset_m": _round(offset),
                            "max_m": _round(tolerance),
                        }
                    )
                    hard_fail = True

    if target_pos is not None:
        # the check cites the layer it was taken against, never the goal
        # geometry: quoting "red object at [x, y, z] r=0.20 m" handed the model
        # the answer the visual evidence exists to support - and made the
        # episode's own visual reasoning optional (G1 §6). The verdict's own
        # numbers (FOV, distance, the measured point) are what travels back.
        evidence.append(
            EvidenceRef(
                "map",
                "goal_check",
                note="completion check against the episode's own state",
            )
        )

    if hard_fail:
        verified: Any = False
        reason = "unmet_conditions"
    elif pixel is not None and depth is None:
        verified = "uncertain"
        reason = "no_depth_estimate"
    elif unresolved_observation:
        verified = "uncertain"
        reason = "unknown_observation"
    elif not checks:
        verified = "uncertain"
        reason = "no_evidence"
    else:
        verified = True
        reason = "verified"

    scope = task_scope(ctx)
    checked_candidate = str(candidate_id) if candidate_id not in (None, "") else ""
    data = {
        "verified": verified,
        "evidence": [e.to_dict() for e in evidence],
        "unmet": unmet,
        "checks": checks,
        "reason": reason,
        "hint": _verify_hint(verified, unmet),
        "task_id": scope["task_id"],
        "subtask_id": scope["subtask_id"],
        "candidate_id": checked_candidate or scope["candidate_id"],
        "evidence_revision": scope["evidence_revision"],
    }
    if pixel is not None:
        data["pixel"] = [float(pixel[0]), float(pixel[1])]
    # the verdict goes into the episode's evidence pack: it is what a later
    # finish call's claim is judged on (the same pack the MCP adapter and the
    # runner read - one trail, one verdict), bound to the candidate instance
    # and revision it was taken against
    record_verification(
        ctx,
        verified=verified,
        observation_id=next((e.ref for e in evidence if e.kind == "observation"), ""),
        checks=checks,
        evidence=evidence,
        reason=reason,
        candidate_id=checked_candidate,
    )
    return ToolResult(STATUS_OK, data, evidence)


def _verify_hint(verified: Any, unmet: list[dict]) -> str:
    """The next step a verification result implies (goal.md section 5).

    The finish rule is the one a hallucinating model breaks: a ``finish`` may
    only claim ``success`` after a verification that passed, so every
    non-verified answer names the call that can still change the verdict.
    """
    if verified is True:
        return "verified: the checks passed - finish now, citing these checks"
    if verified is False:
        failed = ", ".join(
            sorted({str(item.get("check", "")) for item in unmet if item.get("check")})
        )
        return (
            f"not verified ({failed or 'unmet_conditions'}): approach the object "
            "(ground_target then approach_candidate), return to a recalled obs_N "
            "with return_to_observation, or fly to a viewpoint, and call verify_goal "
            'again; do not finish with "success": true while a check is unmet'
        )
    return (
        "inconclusive: point at the target (pixel/region) or pass the "
        "observation_id of a remembered sighting, then call verify_goal again before "
        "finishing"
    )


def _record_arrival(
    ctx: ToolContext, task: dict, outcome: str, evidence: list[EvidenceRef] | None = None,
) -> None:
    """Record a reached place in the episode's evidence pack (goal.md P4).

    Only a leg that really arrived is recorded - ``already_at_pose`` changed
    nothing and is never evidence of having flown somewhere (the movement
    contract says so) - and the record carries the same references the task
    result does, so the completion chain and the answer cite one trail.
    """
    if outcome != "arrived":
        return
    refs = evidence if evidence is not None else _task_evidence(task)
    target_id = _task_target_id(task)
    if str(task.get("intent", "")) == INTENT_CANDIDATE and target_id:
        # arriving at a candidate is also the identity evidence behind a claim:
        # the object was grounded as cand_N and the drone is at its standoff
        record_candidate(
            ctx,
            candidate_id=target_id,
            observation_id=str(task.get("last_observation_id") or ""),
            position=_as_position(task.get("target_position")),
            evidence=refs,
        )
    record_arrival(
        ctx,
        task_id=str(task.get("task_id", "")),
        target_id=target_id,
        intent=str(task.get("intent", "")),
        outcome=str(outcome),
        observation_id=str(task.get("last_observation_id") or ""),
        position=_as_position(task.get("target_position")),
        remaining_m=_remaining_value(task),
        evidence=refs,
    )


# ---------------------------------------------------------------------------
# finish: the completion claim and its evidence chain (goal.md P4)
# ---------------------------------------------------------------------------

#: ``data["claim"]`` of a finish call - what the call itself is.
CLAIM_NONE = "none"  # a bare finish: nothing declared
CLAIM_VERIFIED = "verified"  # success declared, the evidence trail backs it
CLAIM_UNVERIFIED = "unverified"  # success declared, no trail: never counted
CLAIM_GIVE_UP = "give_up"  # an explicit success: false

FINISH_HINT_VERIFIED = (
    "verified: finish accepted on the evidence trail below (verify_goal verified, "
    "or a grounded candidate plus an arrival)"
)
FINISH_HINT_UNVERIFIED = (
    "not counted as a success: the trail this episode left does not back the claim. "
    "Gather it (ground_target then approach_candidate until outcome=arrived, or "
    "verify_goal until verified=true) and finish again - or declare the give-up with "
    '"success": false'
)
FINISH_HINT_GIVE_UP = (
    "honest give-up recorded: the episode ends unfinished, which is not a success "
    "and not a false one either"
)
FINISH_HINT_NONE = (
    'no outcome declared: state it - {"success": true} only with the evidence '
    'above, otherwise {"success": false}'
)


def _record_in_scope(record: dict, scope: dict, task_scoped: bool) -> bool:
    """Whether one pack record belongs to the *current* task/candidate instance.

    A confirmation or arrival recorded for another candidate (or another
    evidence revision) cannot stand in for the current one. A record written
    before the scoped contract carries no revision; it is only accepted when no
    candidate is current, and never for a planned task.
    """
    data = record.get("data") if isinstance(record.get("data"), dict) else {}
    current_candidate = str(scope.get("candidate_id") or "")
    record_candidate = str(data.get("candidate_id") or "")
    if current_candidate and record_candidate != current_candidate:
        return False
    revision = data.get("evidence_revision")
    if revision is None:
        if current_candidate or task_scoped:
            return False
    else:
        try:
            if int(revision) != int(scope.get("evidence_revision", 0)):
                return False
        except (TypeError, ValueError):
            return False
    if task_scoped:
        record_task = str(data.get("scope_task_id") or data.get("task_id") or "")
        if record_task and record_task != str(scope.get("task_id") or ""):
            return False
        if str(data.get("subtask_id") or "") != str(scope.get("subtask_id") or ""):
            return False
    return True


def _declared_scope_problem(args: dict, scope: dict) -> str:
    """A readable mismatch between the finish arguments and the live scope."""
    wanted = {
        "task_id": str(args.get("task_id") or ""),
        "subtask_id": str(args.get("subtask_id") or ""),
        "candidate_id": str(args.get("candidate_id") or ""),
    }
    for field, value in wanted.items():
        if value and value != str(scope.get(field) or ""):
            return (
                f"{MISSING_SCOPE_MISMATCH}: {field}={value} but the current "
                f"{field} is {scope.get(field) or '<none>'}"
            )
    if args.get("evidence_revision") not in (None, ""):
        try:
            declared_revision = int(args["evidence_revision"])
        except (TypeError, ValueError):
            return f"{MISSING_SCOPE_MISMATCH}: evidence_revision is not an integer"
        if declared_revision != int(scope.get("evidence_revision", 0)):
            return (
                f"{MISSING_SCOPE_MISMATCH}: evidence_revision={declared_revision} "
                f"but the current revision is {scope.get('evidence_revision', 0)}"
            )
    return ""


def completion_claim(ctx: ToolContext, args: dict) -> tuple[dict, list[EvidenceRef]]:
    """The verdict of one ``finish`` call: what it declares and what backs it.

    The chain is judged on the episode's own evidence pack
    (``ctx.extra["evidence_pack"]``, ``vlm_nav.tools.completion_evidence``), so
    the local adapter, an MCP client and the runner read one trail. The records
    are first filtered to the *current* task/sub-task/candidate revision: a
    confirmation of an earlier candidate instance can not be inherited.

    When a plan exists, success additionally requires every sub-task to be
    ``done`` and the terminal confirmation (or arrival) to belong to the
    executing sub-task. A single verified candidate is therefore a stage
    completion, never the whole task; the answer lists every open stage in
    ``missing_subtasks``/``missing``.
    """
    declared = args.get("success")
    declared = None if declared is None else bool(declared)
    try:
        from .tool_impl_task import plan_completion
    except Exception:  # pragma: no cover - the module is part of this package
        plan_completion = None  # type: ignore[assignment]
    plan = plan_completion(ctx) if plan_completion is not None else None
    task_scoped = plan is not None

    scope = task_scope(ctx)
    records = evidence_records_from_pack(evidence_pack(ctx))
    scoped_records = [
        record for record in records if _record_in_scope(record, scope, task_scoped)
    ]
    raw_chain = completion_evidence(records)
    chain = completion_evidence(scoped_records)

    missing: list[str] = []
    if task_scoped and not plan["complete"]:
        open_count = len(plan["open_subtasks"])
        total = len(plan["done_subtasks"]) + open_count
        missing.append(
            f"{MISSING_PLAN_COMPLETE}: {open_count}/{total} sub-task(s) still open"
        )
        missing.extend(plan["missing"])
    missing.extend(chain["missing"])
    if not chain["verified"] and raw_chain["verified"]:
        current_candidate = str(scope.get("candidate_id") or "<none>")
        missing.append(
            f"{MISSING_STALE_CONFIRMATION} (current candidate {current_candidate}, "
            f"revision {int(scope.get('evidence_revision', 0))})"
        )
    declared_problem = _declared_scope_problem(args, scope)
    if declared_problem:
        missing.append(declared_problem)

    verified = bool(chain["verified"]) and not missing
    trail = [
        EvidenceRef(
            str(ref.get("kind") or "memory"),
            str(ref.get("ref") or ""),
            str(ref.get("note") or "completion evidence"),
        )
        for ref in chain["evidence_trail"]
        if ref.get("ref")
    ]
    data: dict[str, Any] = {
        "reason": str(args.get("reason") or "")[:200],
        "claimed": declared is True,
        "verified": verified,
        "evidence_trail": [ref.to_dict() for ref in trail],
        "missing": missing,
        "evidence_kind": str(chain["kind"]),
        "task_id": scope["task_id"],
        "subtask_id": scope["subtask_id"],
        "candidate_id": scope["candidate_id"],
        "evidence_revision": int(scope["evidence_revision"]),
        "plan_complete": bool(plan["complete"]) if task_scoped else None,
        "missing_subtasks": list(plan["open_subtasks"]) if task_scoped else [],
    }
    if task_scoped:
        data["plan_id"] = plan["task_id"]
        data["plan_revision"] = plan["revision"]
        data["done_subtasks"] = list(plan["done_subtasks"])
    if declared is True and verified:
        data.update(success=True, claim=CLAIM_VERIFIED, hint=FINISH_HINT_VERIFIED)
    elif declared is True:
        hint = FINISH_HINT_UNVERIFIED
        if missing:
            hint += " missing: " + "; ".join(missing)
        data.update(success=False, claim=CLAIM_UNVERIFIED, hint=hint)
    elif declared is False:
        data.update(success=False, claim=CLAIM_GIVE_UP, hint=FINISH_HINT_GIVE_UP)
    else:
        data.update(success=None, claim=CLAIM_NONE, hint=FINISH_HINT_NONE)
    return data, trail


def _tool_finish(ctx: ToolContext, args: dict) -> ToolResult:
    """End the episode with an explicit outcome and the trail that backs it.

    The runtime calls this tool for a ``finish`` (``AgentRuntime._finish_result``
    prefers a registered finish over its own stub) and reads the ``success``
    field of the answer, so a claim without evidence is downgraded here, at the
    one place both adapters go through.

    A ``success: true`` claim inside a task scope that does not hold (an
    incomplete plan, a stale candidate confirmation, a declared scope that is
    not current) returns ``blocked`` instead of ``ok``: the runtime does not
    terminate, and the next turn reads exactly which stages/evidence are still
    missing. Outside a task scope the historical behaviour is kept (the episode
    may end, but an unbacked claim is still *unverified*, never a success).
    """
    data, trail = completion_claim(ctx, args)
    blocked = (
        args.get("success") is True
        and not data.get("verified")
        and scope_active(ctx)
    )
    return ToolResult(STATUS_BLOCKED if blocked else STATUS_OK, data, trail)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

_POINT_SCHEMA = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 3,
    "maxItems": 3,
    "description": "world position [x, y, z] in meters",
}
_TARGET_PROPERTIES = {
    "viewpoint_id": {"type": "string", "description": "id returned by propose_viewpoints"},
    "target_id": {
        "type": "string",
        "description": (
            "candidate id (cand_N from ground_target), a recalled observation id "
            "(obs_N from query_memory/observe), memory candidate id or a viewpoint "
            "id (vp_N); an unknown id comes back with the ids that do resolve, the "
            "recalled position to use instead and a ready next_action"
        ),
    },
    "observation_id": {
        "type": "string",
        "description": (
            "recalled observation id (obs_N) whose recorded pose is the target - "
            "the way back to a place seen earlier"
        ),
    },
    "position": _POINT_SCHEMA,
}


def register_navigation_tools(registry: ToolRegistry) -> None:
    """Register the three intent tools, propose_viewpoints, check_path,
    task_status, cancel_task, verify_goal, finish, batch_read and the deprecated
    navigate stub.

    The read-only tools are registered ``cacheable=False``: their answer
    depends on the live pose, observed map or task state, which an args-keyed
    result cache cannot see, so a repeat must be recomputed instead of replayed.
    ``task_status`` implements its own state-aware repeat semantics ("no change
    since the last query" hint).
    """
    registry.register(
        ToolSpec(
            name="propose_viewpoints",
            description=(
                "Propose reachable observation points around the current pose from the "
                "observed map + flown trajectory. Returns viewpoint_id (vp_N) - the "
                "recommended pick - plus position, expected_coverage (0-1, "
                "unobstructed view), unknown_ratio (0-1, unexplored in front) and "
                "cost_m for each candidate; pass one to check_path or go_to_viewpoint."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "target_hypothesis": {
                        "type": ["string", "array"],
                        "description": (
                            "candidate/hypothesis id, target name or [x, y, z] world "
                            "position to bias the viewpoints toward"
                        ),
                    },
                    "direction_deg": {
                        "type": "number",
                        "description": (
                            "bearing relative to the current heading in degrees, CCW "
                            "positive; ignored when target_hypothesis resolves"
                        ),
                    },
                    "count": {"type": "integer", "minimum": 1, "maximum": 6},
                    "n": {"type": "integer", "minimum": 1, "maximum": 6,
                          "description": "alias for count"},
                    "max_viewpoints": {"type": "integer", "minimum": 1, "maximum": 6,
                                       "description": "alias for count"},
                    "radius_m": {
                        "type": "number",
                        "description": "far-ring distance of the proposed points",
                    },
                    "target_id": {
                        "type": "string",
                        "description": "alias for target_hypothesis",
                    },
                },
                "required": [],
            },
            handler=_tool_propose_viewpoints,
            side_effects=False,
            cacheable=False,
        )
    )
    registry.register(
        ToolSpec(
            name="check_path",
            description=(
                "Check one straight leg before moving: reachable (bool), path_cost_m, "
                "unknown_ratio (0-1 of the leg with no observation evidence) and reason "
                "(clear|mostly_unknown|obstacle_on_path|too_far). The target is a "
                "viewpoint_id, a target_id (cand_N or a recalled obs_N), an "
                "observation_id or a position=[x, y, z]."
            ),
            input_schema={
                "type": "object",
                "properties": dict(_TARGET_PROPERTIES),
                "required": [],
            },
            handler=_tool_check_path,
            side_effects=False,
            cacheable=False,
        )
    )
    registry.register(
        ToolSpec(
            name=INTENT_CANDIDATE,
            description=(
                "Approach the target candidate cand_N that ground_target returned: fly "
                "AT MOST ONE leg (<= 8 m) per call and return at once with task_id, "
                "status (ok|blocked|no_progress), outcome (arrived|partial|collision|"
                "stalled|planner_failed|already_at_pose|...), progress_m, turn_deg, "
                "remaining_m, blocked_reason and last_observation_id. The leg ends "
                "standoff_m short of the candidate's world position, the tools observe "
                "and re-localize the candidate after every segment, and calling again "
                "with task_id flies the next segment. This is the way to the OBJECT in "
                "an observation - not return_to_observation (that one flies to the "
                "camera pose). An unknown candidate id comes back with the ids that do "
                "resolve plus a ready next_action."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "candidate_id": {
                        "type": "string",
                        "description": (
                            "cand_N from ground_target (a candidate is not an "
                            "observation: never pass obs_N here)"
                        ),
                    },
                    "standoff_m": {
                        "type": "number",
                        "description": (
                            "how close to get to the candidate (default: the world's "
                            "standoff, 1.2 m)"
                        ),
                    },
                    "max_legs": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 64,
                        "description": "segments this approach may fly (default 12)",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "continue an existing approach task (nav_N)",
                    },
                    "timeout_s": {
                        "type": "number",
                        "description": "seconds the whole task may stay unfinished (default 120)",
                    },
                },
                "required": ["candidate_id"],
            },
            handler=_tool_approach_candidate,
            side_effects=True,
        )
    )
    registry.register(
        ToolSpec(
            name=INTENT_VIEWPOINT,
            description=(
                "Fly to an observation POSITION vp_N proposed by propose_viewpoints: at "
                "most one leg (<= 8 m) per call, returning task_id, status (ok|blocked|"
                "no_progress), outcome, progress_m, turn_deg, remaining_m, blocked_reason "
                "and the candidate_viewpoints to try when blocked. A position, not an "
                "object: to reach the object in a frame use approach_candidate; to fly "
                "back to where a frame was shot use return_to_observation. Re-submitting "
                "the same request for a place already reached answers cached with "
                "already_done=true and its arrival_evidence - nothing flies again."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "viewpoint_id": {
                        "type": "string",
                        "description": "vp_N from propose_viewpoints",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "continue an existing task (nav_N)",
                    },
                    "timeout_s": {"type": "number"},
                    "max_segments": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 64,
                        "description": "legs the task may fly (default 12)",
                    },
                },
                "required": ["viewpoint_id"],
            },
            handler=_tool_go_to_viewpoint,
            side_effects=True,
        )
    )
    registry.register(
        ToolSpec(
            name=INTENT_OBSERVATION,
            description=(
                "Fly back to the pose an observation obs_N was captured at and (with "
                "restore_yaw=true, the default) turn to the heading it recorded - the "
                "CAMERA's place, not the object it photographed. One leg (<= 8 m) per "
                "call, same progress contract as the other movement tools; arriving is "
                "reported as outcome=arrived, and a call that cannot move anything is "
                "status=no_progress / outcome=already_at_pose with a different next step. "
                "The same request again for an already reached place answers "
                "already_done=true with the arrival_evidence instead of flying again."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "observation_id": {
                        "type": "string",
                        "description": "obs_N of a recorded view (from observe/query_memory)",
                    },
                    "restore_yaw": {
                        "type": "boolean",
                        "description": (
                            "turn to the heading that observation was captured at "
                            "(default true)"
                        ),
                    },
                    "task_id": {
                        "type": "string",
                        "description": "continue an existing task (nav_N)",
                    },
                    "timeout_s": {"type": "number"},
                    "max_segments": {"type": "integer", "minimum": 1, "maximum": 64},
                },
                "required": ["observation_id"],
            },
            handler=_tool_return_to_observation,
            side_effects=True,
        )
    )
    registry.register(
        ToolSpec(
            name="navigate",
            description=(
                "DEPRECATED - split into approach_candidate (candidate_id=cand_N), "
                "go_to_viewpoint (viewpoint_id=vp_N) and return_to_observation "
                "(observation_id=obs_N) because it mixed three intents in one "
                "signature. Every call answers with an explicit error and the exact "
                "replacement call for the arguments given; it never moves the drone."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    **_TARGET_PROPERTIES,
                    "task_id": {
                        "type": "string",
                        "description": (
                            "a task started by another movement tool whose next segment "
                            "this call is meant to fly"
                        ),
                    },
                    "timeout_s": {"type": "number"},
                    "max_segments": {"type": "integer", "minimum": 1, "maximum": 64},
                },
                "required": [],
            },
            handler=_deprecated_navigate_result,
            side_effects=False,  # it executes nothing at all
        )
    )
    registry.register(
        ToolSpec(
            name="task_status",
            description=(
                "Report a navigation task: status, progress_m, remaining_m, segments_left, "
                "candidate_viewpoints and its execution events (arrived / path_blocked / "
                "map_changed / target_rediscovered / task_failed / task_cancelled). Without "
                "task_id every task is listed. A query with no state change in between "
                "returns 'cached' plus a hint."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "nav_N id; omit (or 'all') to list every task",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 32,
                        "description": "most recent events to include (default 8)",
                    },
                },
                "required": [],
            },
            handler=_tool_task_status,
            side_effects=False,
            cacheable=False,
        )
    )
    registry.register(
        ToolSpec(
            name="cancel_task",
            description=(
                "Interrupt a running navigation task: its status becomes 'cancelled' and a "
                "task_cancelled event is queued, so later movement calls with that task_id "
                "are refused. Cancelling a task that already ended returns 'cached'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "nav_N id to interrupt"},
                    "reason": {"type": "string", "description": "why the task is cancelled"},
                },
                "required": ["task_id"],
            },
            handler=_tool_cancel_task,
            side_effects=True,
        )
    )
    registry.register(
        ToolSpec(
            name="verify_goal",
            description=(
                "Independently verify task completion against the goal target using the "
                "pointed pixel/region depth (or the current proximity and FOV when no "
                "pixel is given, or the recorded pose of a recalled observation_id): "
                "verified (true|false|uncertain), evidence, the unmet conditions and the "
                "next step. Call it before finish: a finish may only claim success after "
                "a verify_goal call that returned verified=true."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "candidate_id": {
                        "type": ["string", "integer"],
                        "description": "candidate whose identity should match the pointed region",
                    },
                    "observation_id": {
                        "type": "string",
                        "description": (
                            "recalled observation id (obs_N): checks the place where the "
                            "target was seen against the goal target"
                        ),
                    },
                    "region": {
                        "type": ["string", "array"],
                        "description": "left|center|right|top|bottom or [u, v] normalized pixel",
                    },
                    "pixel": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "normalized [0,1000] image point (Qwen convention)",
                    },
                    "success_dist_m": {
                        "type": "number",
                        "description": "override the proximity threshold of the goal check",
                    },
                },
                "required": [],
            },
            handler=_tool_verify_goal,
            side_effects=False,
            cacheable=False,
        )
    )
    registry.register(
        ToolSpec(
            name="finish",
            description=(
                "End the episode with an explicit outcome and the evidence behind it. "
                'Answer {"success": true} only when this episode left the trail: a '
                "verify_goal that returned verified=true for the current candidate and "
                "evidence_revision, or a grounded candidate plus an arrival "
                "(outcome=arrived) - and, when a plan exists, every sub-task is done. "
                "A single verified stage never finishes the whole task: the call then "
                "comes back blocked with missing_subtasks (the stages still open) and "
                "missing (the evidence they owe). The answer also reports evidence_trail, "
                "verified and claim; a success without the trail is downgraded to an "
                'unverified claim and never counted as success. Use {"success": false} '
                "for an honest give-up and always state the reason."
            ),
            input_schema=FINISH_SCHEMA,
            handler=_tool_finish,
            side_effects=False,
            cacheable=False,
        )
    )
    # several read-only checks in one turn (goal.md P4): the wrapper is registered
    # here so both adapters advertise it, and ToolRegistry.call dispatches it
    register_batch_tool(registry)


__all__ = ["register_navigation_tools", "register_tools", "register_all"]

# ``scripts/run_vlm_nav_tools.py`` and ``vlm_nav/mcp_server.py`` probe tool
# modules for these names before installing their own fallback/contract stubs
# (each probe gets a fresh registry and merges by name, so the aliases cannot
# double-register). Keep ``register_navigation_tools`` as the documented entry
# point and expose the probes so this module's tools are actually picked up.
register_tools = register_navigation_tools
register_all = register_navigation_tools
