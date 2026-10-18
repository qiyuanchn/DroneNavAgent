#!/usr/bin/env python
"""Phase-1 acceptance driver: the tool loop vs. the action loop, paired.

Phase 1 (goal.md) asks one question that a single run cannot answer: with the
*same* model, scene, start pose/heading and VLM budget, does the tool loop
produce stable, correctly directed movement and complete the simple tasks the
action loop already completes? This driver runs both agents as subprocesses
(``scripts/run_vlm_nav_3dgs.py`` = the action loop, ``scripts/run_vlm_nav_tools.py``
= the tool loop), one directory per episode, and reports the paired metrics.

Mock run (CPU world, no weights):

    python scripts/verify_phase1.py --world cpu --ply scene.ply \
        --target-pos 4 0 1 --instruction "Find the target and fly to it." \
        --backend mock --seeds 7 --max-vlm-calls 10 \
        --out outputs/vlm_nav/verify_phase1_mock

Real paired run (same model on both sides; the coordinator runs this):

    python scripts/verify_phase1.py --world higs --ply scene.ply \
        --target-pos 4 0 1 --instruction "Find the book." --backend hf \
        --model-path /models_ro/Qwen3-VL-8B --device cuda --seeds 7,11 \
        --max-vlm-calls 20 --out outputs/vlm_nav/verify_phase1_qwen

Metrics (per episode and aggregated per agent) - deliberately *not* one number:

  tool_ok_rate        tool calls whose result status is ``ok`` / tool calls.
                      The action loop reports the analogous ``call_ok_rate``
                      (parseable reply + no collision/stall outcome).
  effective_move_rate nav attempts whose measured displacement is above
                      ``--min-progress-m`` AND whose direction is within 90 deg
                      of the bearing to the target at the attempt's start pose
                      / all nav attempts. Nav attempts are the tool names in
                      ``NAV_TOOL_NAMES`` (navigate / approach_candidate /
                      go_to_viewpoint / return_to_observation) and, for the
                      action loop, the move steps (``move``/``waypoint``/
                      ``explore``/``goto``); turn/scan/hover steps are not move
                      attempts, so turning in place is not counted as a stall.
  zero_displacement_* nav attempts that moved nothing (below the progress
                      floor). ``zero_displacement_runs`` counts maximal runs of
                      >= ``--zero-run-min-length`` consecutive such attempts
                      (the observe -> navigate -> already_there death loop);
                      overlapping pairs are listed in
                      ``zero_displacement_partners`` and the marker histogram
                      (``already_there``, ``no_progress``, ``unknown_observation``,
                      ...) in ``nav_failure_reasons``.
  reacquisition_rate  visibility series over the episode: lost = a
                      visible -> not-visible transition, reacquired = the next
                      not-visible -> visible one. ``reacquisition_rate`` =
                      reacquired / lost (null when the target was never lost).
                      Visibility is the tool-reported ``target_in_fov`` when the
                      trace carries it (``visibility_source=reported``), else a
                      level-camera FOV estimate from pose + heading + the
                      observation intrinsics (``--fov-hfov-deg``/``--render-size``).
                      Where both exist, ``fov_agreement`` reports how often they
                      agree - the FOV-deviation diagnostic, per episode.
  task_success        ``reached``: the last traced pose is within
                      ``success-dist + target radius`` of the target (ground
                      truth, computed here, not read from the runner);
                      ``runner_success``: the runner's own verdict.
                      ``solved`` = reached AND runner_success: a completion
                      claim without the geometric reach is *not* a finished task
                      (``claim_without_reach`` counts those episodes).
  full VLM cost       ``vlm_calls_total`` = max(reported, decision calls +
                      declared internal calls). ``vlm_calls_decision`` is one
                      call per step/turn; internal calls are the runner-counted
                      ones beyond that (action loop: look-around scan steps and
                      done-crop reviews, counted by the harness) plus the
                      tool-internal calls each tool declares in its result data
                      (contract field ``vlm_calls``, e.g. a crop re-check). When
                      no tool declares a count, the tool-side total is a lower
                      bound (``vlm_cost_complete=false``). ``tokens_estimate``
                      and ``wall_time_s`` follow
                      ``scripts/compare_agent_configs.py``'s model.

The tool-side vocabulary and result schema are the shared contract
(``vlm_nav.tools``): ``PHASE1_TOOLS`` = observe, ground_target,
approach_candidate, go_to_viewpoint, return_to_observation, verify_goal. Which of
them the tool runner actually registered is reported per episode
(``required_tools_missing``): an episode that lost the toolset is a blocker, not
a silent failure. The driver never edits the runners; a tool that does not exist
yet shows up as missing in the report.

The tool loop runs with ``--toolset full`` (unrestricted, so the six tools are
offered as soon as a module registers them; ``run_vlm_nav_tools``' basic/memory
catalogues now carry the same intent vocabulary - ``navigate`` is deprecated and
in none of them); the report names the toolset it used and the registered tools.
A scripted mock episode over the Phase-1 chain is one flag:

    --tool-arg=--mock-script=/tmp/phase1_play.json

with the play modelled on ``run_vlm_nav_tools.ACCEPTANCE_TOOL_SCRIPT`` (call
``propose_viewpoints`` before ``go_to_viewpoint``, ``ground_target`` - which
takes the instruction as an argument - before ``approach_candidate``); the
placeholders ``$viewpoint_id``/``$candidate_id`` resolve from the previous
result.

Artifacts:
  verify_phase1_report.md    the paired report (aggregates, cells, per episode,
                             tool contract, verdict, metric definitions)
  verify_phase1_report.json  same data machine-readable (meta, episodes,
                             aggregates, cells, verdict)
  run_config.json            resolved CLI of this driver
  <agent>/<task>/<yaw>_seed<S>/  one directory per episode with the runner's own
                             summary.json / traces / frames and run.log

Exit code: 0 when the verdict passes, 1 when a check fails or an episode errored
(``--dry-run`` always 0). The verdict and every individual check are in the
report; nothing is averaged away.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python scripts/verify_phase1.py`
    sys.path.insert(0, str(ROOT))

from scripts import compare_agent_configs as compare  # noqa: E402

from vlm_nav.tools import (  # noqa: E402
    STATUS_BLOCKED,
    STATUS_CACHED,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_OK,
)

# --- agents -----------------------------------------------------------------
CONFIG_ACTION = "action"
CONFIG_TOOL = "tool"
CONFIG_ORDER = (CONFIG_ACTION, CONFIG_TOOL)
CONFIG_LABELS = {CONFIG_ACTION: "action loop", CONFIG_TOOL: "tool loop"}

# --- the Phase-1 tool contract (names + result fields) ----------------------
#: Toolset of the tool loop under test (the reserved navigation tools of
#: ``vlm_nav.tools.TOOL_CALL_DOC`` plus the perception/verify pair).
PHASE1_TOOLS = (
    "observe",
    "ground_target",
    "approach_candidate",
    "go_to_viewpoint",
    "return_to_observation",
    "verify_goal",
)
#: Tools that command a movement (one leg each): every call is a nav attempt.
NAV_TOOL_NAMES = (
    "navigate",
    "approach_candidate",
    "go_to_viewpoint",
    "return_to_observation",
)
#: Action-loop summary prefixes that command a movement.
MOVE_ACTION_TOKENS = ("move", "waypoint", "explore", "goto")
#: Result markers of "the leg produced no displacement".
ZERO_DISPLACEMENT_MARKERS = (
    "already_there",
    "already_at_pose",
    "at_pose",
    "no_progress",
    "no_movement",
    "zero_progress",
    "stalled",
)
#: Result fields carrying the displacement of one leg (per-leg first).
PROGRESS_FIELDS = ("segment_progress_m", "progress_m")
#: Result fields a tool uses to declare the VLM calls it spent internally.
VLM_COST_FIELDS = ("vlm_calls", "vlm_calls_used", "vlm_cost", "cost_vlm_calls")
#: Result fields carrying the visibility of the target at the result's pose.
VISIBILITY_FIELDS = ("target_in_fov", "target_visible", "visible", "in_fov")
#: Words that mark an action-loop outcome as a failed execution.
FAILURE_OUTCOME_TOKENS = (
    "collision",
    "stalled",
    "planner failed",
    "leg timeout",
    "timeout",
    "terminated",
    "blocked",
)

TOOL_TRACE = "tool_trace.jsonl"
ACTION_TRACE = "episode_trace.jsonl"
SUMMARY_FILE = "summary.json"
REPORT_JSON = "verify_phase1_report.json"
REPORT_MD = "verify_phase1_report.md"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _as_float(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _rate(numerator: float, denominator: float) -> float | None:
    if not denominator:
        return None
    return round(float(numerator) / float(denominator), 3)


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [_as_float(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    return round(sum(numbers) / len(numbers), 3) if numbers else None


def _sum(values: Iterable[Any]) -> float | None:
    numbers = [_as_float(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    return round(sum(numbers), 3) if numbers else None


def _max(values: Iterable[Any]) -> float | None:
    numbers = [_as_float(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    return max(numbers) if numbers else None


def _fmt(value: Any, digits: int = 2, missing: str = "-") -> str:
    if value is None:
        return missing
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _distance(a: Any, b: Any) -> float | None:
    pa, pb = _as_point(a), _as_point(b)
    if pa is None or pb is None:
        return None
    return math.dist(pa, pb)


def _horizontal(a: Any, b: Any) -> float | None:
    pa, pb = _as_point(a), _as_point(b)
    if pa is None or pb is None:
        return None
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1])


def _as_point(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        for key in ("position_flu", "pose", "position", "pos"):
            if key in value:
                return _as_point(value[key])
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return None
    return None


def _bearing_deg(origin: Any, target: Any) -> float | None:
    start, end = _as_point(origin), _as_point(target)
    if start is None or end is None:
        return None
    dx, dy = end[0] - start[0], end[1] - start[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    return math.degrees(math.atan2(dy, dx))


def _entry_data(entry: dict) -> dict:
    """Tool result payload of one trace row (module and fallback schemas)."""
    for key in ("data", "result_data", "result"):
        value = entry.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _fail_marker(event: dict, config: str) -> str:
    """Short label of why a call/leg failed, for the diagnosis histograms.

    Precedence: a cached tool call is a repeat (the runtime's own verdict that
    nothing new came back), then the explicit ``reason``/``error`` field, then
    the tool/action outcome text - action-loop outcomes are reduced to the
    failure kind (collision/stalled/...), tool results to their reason token,
    falling back to the result status when the text says nothing about failure.
    """
    data = event.get("data") or {}
    status = str(event.get("status") or "")
    if config == CONFIG_TOOL and status == STATUS_CACHED:
        return STATUS_CACHED
    explicit = ""
    for key in ("reason", "error"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            explicit = value.strip()
            break
    if config == CONFIG_ACTION:
        text = explicit or str(event.get("outcome") or "") or str(data.get("outcome") or "")
        lowered = text.lower()
        for token in FAILURE_OUTCOME_TOKENS:
            if token in lowered:
                return token
        return text[:40] if text else status or "unknown"
    if explicit:
        return explicit.split(":")[0].strip()[:40] or status or "unknown"
    text = str(event.get("outcome") or "") or str(data.get("outcome") or "")
    token = text.split(":")[0].strip()[:40]
    if token in ZERO_DISPLACEMENT_MARKERS:
        return token
    if status and status != STATUS_OK:
        return status
    return token or status or "unknown"


def _visibility_field(data: dict, entry: dict) -> bool | None:
    for source in (data, entry):
        for key in VISIBILITY_FIELDS:
            value = _as_bool(source.get(key))
            if value is not None:
                return value
    return None


# ---------------------------------------------------------------------------
# trace normalization
# ---------------------------------------------------------------------------


def timeline(trace: list[dict], config: str, *, start_pose: Any = None) -> list[dict]:
    """One normalized event per trace row (the ``finish`` turn is not one).

    Both runners' schemas are folded into the same fields: ``pose``/``yaw_deg``
    (after the call), ``prev_pose`` (the state the call started from - the
    previous row's pose, or ``start_pose``), ``move_intent``, ``status``,
    ``outcome``, ``reported_progress_m``, ``reported_visible``,
    ``reported_vlm_calls``, ``parse_failure`` and ``is_call``.
    """
    events: list[dict] = []
    previous_pose = _as_point(start_pose)
    previous_yaw = None
    for raw in trace:
        if not isinstance(raw, dict):
            continue
        tool = raw.get("tool")
        tool = str(tool) if tool not in (None, "", "(no tool call)") else None
        if str(raw.get("tool")) == "finish":
            continue
        data = _entry_data(raw)
        parse_failure = bool(raw.get("parse_failure")) or raw.get("parse_ok") is False
        entry_pose = raw.get("pose")
        pose = None
        for value in (data.get("pose"), data.get("position_flu"),
                      raw.get("position_flu"), entry_pose):
            pose = _as_point(value)
            if pose is not None:
                break
        yaw = None
        for value in (data.get("heading_deg"), data.get("yaw_deg"),
                      raw.get("yaw_deg"), raw.get("heading_deg")):
            yaw = _as_float(value)
            if yaw is not None:
                break
        if yaw is None and isinstance(entry_pose, dict):
            yaw = _as_float(entry_pose.get("yaw_deg") or entry_pose.get("heading_deg"))
        status = str(raw.get("status") or (STATUS_FAILED if parse_failure else STATUS_OK))
        if config == CONFIG_ACTION:
            summary = str(raw.get("action_summary") or "")
            token = summary.split(" ")[0].strip().lower() if summary else ""
            move_intent = token in MOVE_ACTION_TOKENS
            action = token or "unknown"
        else:
            move_intent = tool in NAV_TOOL_NAMES
            action = tool or "unknown"
        reported_progress = None
        for key in PROGRESS_FIELDS:
            reported_progress = _as_float(data.get(key))
            if reported_progress is not None:
                break
        declared_cost = 0
        for source in (data, raw):
            for key in VLM_COST_FIELDS:
                value = _as_int(source.get(key))
                if value:
                    declared_cost += value
        events.append({
            "index": len(events) + 1,
            "turn": _as_int(raw.get("turn")) or _as_int(raw.get("step")) or len(events) + 1,
            "config": config,
            "tool": tool,
            "action": action,
            "is_call": bool(tool) if config == CONFIG_TOOL else True,
            "move_intent": bool(move_intent),
            "status": status,
            "outcome": str(raw.get("outcome") or data.get("outcome")
                           or data.get("reason") or ""),
            "reason": str(data.get("reason") or raw.get("reason") or ""),
            "data": data,
            "pose": pose,
            "yaw_deg": yaw,
            "prev_pose": previous_pose,
            "prev_yaw": previous_yaw,
            "reported_progress_m": reported_progress,
            "reported_visible": _visibility_field(data, raw),
            "reported_vlm_calls": declared_cost or None,
            "parse_failure": parse_failure,
            "images_sent": _as_int(raw.get("images_sent")),
            "result_prompt": str(raw.get("result_prompt") or ""),
            "raw_reply": str(raw.get("model_reply") or raw.get("raw_reply") or ""),
        })
        if pose is not None:
            previous_pose = pose
        if yaw is not None:
            previous_yaw = yaw
    return events


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def required_tools_missing(summary: dict) -> list[str]:
    """Phase-1 tools the tool runner did not register (blocker, not a failure)."""
    registered = None
    detail = summary.get("toolset_detail")
    if isinstance(detail, dict):
        value = detail.get("registered")
        if isinstance(value, (list, tuple, set)):
            registered = {str(name) for name in value}
    if registered is None:
        tools = summary.get("tools")
        if isinstance(tools, dict):
            registered = {str(name) for name in tools}
    if registered is None:
        return list(PHASE1_TOOLS)
    return [name for name in PHASE1_TOOLS if name not in registered]


def target_in_fov_estimate(pose: Any, yaw_deg: Any, target_pos: Any, *,
                           hfov_deg: float = 86.0, width: int = 320,
                           height: int = 180) -> bool | None:
    """Is the target inside the level camera cone at ``pose``/``yaw_deg``?

    Level-camera pinhole check (the camera is not pitched in these worlds):
    forward = +x rotated by yaw, right = yaw - 90, up = +z. Returns None when
    the pose, the heading or the target is unknown.
    """
    point = _as_point(pose)
    target = _as_point(target_pos)
    yaw = _as_float(yaw_deg)
    if point is None or target is None or yaw is None:
        return None
    width = max(1, int(width))
    height = max(1, int(height))
    half_fov = math.radians(max(1e-3, float(hfov_deg))) / 2.0
    fx = width / (2.0 * math.tan(half_fov))
    radians = math.radians(yaw)
    forward = (math.cos(radians), math.sin(radians))
    right = (math.sin(radians), -math.cos(radians))
    dx, dy, dz = target[0] - point[0], target[1] - point[1], target[2] - point[2]
    depth = dx * forward[0] + dy * forward[1]
    if depth <= 1e-9:
        return False
    lateral = dx * right[0] + dy * right[1]
    u = fx * lateral / depth
    v = fx * (-dz) / depth
    return abs(u) <= width / 2.0 and abs(v) <= height / 2.0


def movement_stats(events: list[dict], target_pos: Any, *,
                   start_pose: Any = None, min_progress_m: float = 0.1,
                   zero_run_min: int = 2) -> dict:
    """Displacement metrics over the episode's nav attempts.

    ``nav_attempts``          movement calls (name/step based, see the module doc)
    ``effective_move_rate``   attempts with progress > ``min_progress_m`` and a
                              direction within 90 deg of the target bearing
    ``zero_displacement_*``   attempts with no progress above the floor; runs of
                              ``zero_run_min`` or more consecutive ones are the
                              zero-displacement loops (``_partners`` lists the
                              overlapping tool pairs, ``nav_failure_reasons`` the
                              markers behind them)
    ``distance_before/after`` distance to the target at the first attempt's start
                              pose and at the last attempt's end pose
    """
    attempts = [event for event in events if event.get("move_intent")]
    stats: dict[str, Any] = {
        "nav_attempts": len(attempts),
        "moved_attempts": 0,
        "effective_moves": 0,
        "wrong_direction_moves": 0,
        "zero_displacement_events": 0,
        "zero_displacement_runs": 0,
        "max_zero_displacement_run": 0,
        "zero_displacement_partners": [],
        "no_progress_markers": 0,
        "mean_move_m": None,
        "max_move_m": None,
        "distance_before_m": None,
        "distance_after_m": None,
        "distance_gain_m": None,
        "effective_move_rate": None,
        "nav_failure_reasons": {},
    }
    if not attempts:
        return stats

    floor = float(min_progress_m)
    moves: list[dict] = []
    failure_reasons: dict[str, int] = {}
    for event in attempts:
        origin = event.get("prev_pose")
        if origin is None:
            origin = start_pose
        progress = _horizontal(origin, event.get("pose"))
        if progress is None:
            progress = 0.0
        bearing = _bearing_deg(origin, target_pos)
        move_bearing = _bearing_deg(origin, event.get("pose"))
        angle = None
        if bearing is not None and move_bearing is not None:
            angle = abs(compare._wrap_deg(move_bearing - bearing))
        effective = progress > floor and (angle is None or angle < 90.0)
        stalled = progress <= floor
        marker = _fail_marker(event, event.get("config") or CONFIG_TOOL)
        if event.get("outcome") in ZERO_DISPLACEMENT_MARKERS or (
            marker in ZERO_DISPLACEMENT_MARKERS
        ):
            stats["no_progress_markers"] += 1
        if not effective and (
            stalled or event.get("status") in (STATUS_EMPTY, STATUS_BLOCKED, STATUS_FAILED)
        ):
            failure_reasons[marker] = failure_reasons.get(marker, 0) + 1
        moves.append({
            "tool": event.get("tool") or event.get("action"),
            "progress": progress,
            "angle": angle,
            "stalled": stalled,
            "effective": effective,
            "marker": marker,
        })

    stats["moved_attempts"] = sum(1 for move in moves if move["progress"] > floor)
    stats["effective_moves"] = sum(1 for move in moves if move["effective"])
    stats["wrong_direction_moves"] = sum(
        1 for move in moves if move["progress"] > floor and not move["effective"]
    )
    stats["zero_displacement_events"] = sum(1 for move in moves if move["stalled"])
    stats["mean_move_m"] = round(
        sum(move["progress"] for move in moves) / len(moves), 3
    ) if moves else None
    stats["max_move_m"] = round(max((move["progress"] for move in moves), default=0.0), 3)
    stats["effective_move_rate"] = _rate(stats["effective_moves"], len(moves))
    stats["nav_failure_reasons"] = failure_reasons

    # zero-displacement runs over the attempt subsequence (read-only turns in
    # between do not break a run: the loop is about repeated nav calls)
    runs: list[list[dict]] = []
    current: list[dict] = []
    for move in moves:
        if move["stalled"]:
            current.append(move)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)
    loops = [run for run in runs if len(run) >= max(1, int(zero_run_min))]
    stats["zero_displacement_runs"] = len(loops)
    stats["max_zero_displacement_run"] = max((len(run) for run in loops), default=0)
    partners: list[str] = []
    for run in loops:
        for previous, following in zip(run, run[1:]):
            partners.append(f"{previous['tool']}->{following['tool']}")
    stats["zero_displacement_partners"] = partners

    first_origin = attempts[0].get("prev_pose") or start_pose
    last_pose = None
    for event in reversed(attempts):
        if event.get("pose") is not None:
            last_pose = event["pose"]
            break
    stats["distance_before_m"] = _rounded(_distance(first_origin, target_pos))
    stats["distance_after_m"] = _rounded(_distance(last_pose, target_pos))
    if stats["distance_before_m"] is not None and stats["distance_after_m"] is not None:
        stats["distance_gain_m"] = round(
            stats["distance_before_m"] - stats["distance_after_m"], 3
        )
    return stats


def _rounded(value: Any) -> float | None:
    number = _as_float(value)
    return None if number is None else round(number, 3)


def vision_stats(events: list[dict], target_pos: Any, *, hfov_deg: float = 86.0,
                 width: int = 320, height: int = 180, start_pose: Any = None,
                 start_yaw_deg: Any = None) -> dict:
    """Visibility series -> target loss / re-acquisition and the FOV diagnostic.

    The series starts at the episode's start pose and continues with every trace
    row's own pose. A row may report the visibility itself (tool result field
    ``target_in_fov``); otherwise it is estimated from the pose, the heading and
    the camera cone. ``fov_agreement`` compares the two wherever both exist.

    A row that carries no pose/heading (a query-only call) keeps the last known
    state for the estimate: the drone did not move, so the last observed
    viewpoint is still the current one.
    """
    start_estimate = target_in_fov_estimate(start_pose, start_yaw_deg, target_pos,
                                            hfov_deg=hfov_deg, width=width,
                                            height=height)
    states: list[dict] = [{
        "index": 0,
        "visible": start_estimate,
        "estimated": start_estimate,
        "reported": None,
    }]
    last_pose = _as_point(start_pose)
    last_yaw = _as_float(start_yaw_deg)
    for event in events:
        pose = event.get("pose")
        yaw = event.get("yaw_deg")
        if pose is None:
            pose = last_pose
        else:
            last_pose = pose
        if yaw is None:
            yaw = last_yaw
        else:
            last_yaw = yaw
        estimated = target_in_fov_estimate(pose, yaw, target_pos, hfov_deg=hfov_deg,
                                           width=width, height=height)
        reported = event.get("reported_visible")
        states.append({
            "index": event.get("index"),
            "visible": reported if reported is not None else estimated,
            "estimated": estimated,
            "reported": reported,
        })

    lost_turns: list[int] = []
    reacquired_turns: list[int] = []
    previous = None
    for state in states:
        visible = state["visible"]
        if visible is None:
            continue
        if previous is True and visible is False:
            lost_turns.append(state["index"])
        elif previous is False and visible is True:
            reacquired_turns.append(state["index"])
        previous = visible

    measured = [state for state in states if state["reported"] is not None]
    compared = [state for state in measured if state["estimated"] is not None]
    mismatches = [state for state in compared if state["estimated"] != state["reported"]]
    reported_only = [state for state in states if state["reported"] is not None]
    if reported_only and len(reported_only) == len(states) - 1:
        source = "reported"
    elif reported_only:
        source = "mixed"
    else:
        source = "estimated"

    return {
        "visibility_source": source,
        "target_lost_events": len(lost_turns),
        "target_reacquired_events": len(reacquired_turns),
        "reacquisition_rate": _rate(len(reacquired_turns), len(lost_turns)),
        "loss_turns": lost_turns,
        "reacquired_turns": reacquired_turns,
        "visible_states": sum(1 for state in states if state["visible"] is True),
        "states": len(states),
        "fov_measured_states": len(measured),
        "fov_agreement": _rate(len(compared) - len(mismatches), len(compared)),
        "fov_mismatch_states": len(mismatches),
    }


def call_stats(events: list[dict], config: str) -> dict:
    """Tool/action call success accounting (``ok`` vs every other status).

    Tool loop: ``ok`` = the tool result's status; the histogram also splits
    empty/blocked/failed/cached and counts unparseable model replies separately.
    Action loop: ``ok`` = a parseable reply whose outcome is not a failure
    (collision/stall/planner failure/timeout).
    """
    calls = [event for event in events if event.get("is_call")]
    by_status: dict[str, int] = {}
    calls_by_tool: dict[str, int] = {}
    failure_reasons: dict[str, int] = {}
    ok = 0
    parse_failures = 0
    for event in events:
        if event.get("parse_failure"):
            parse_failures += 1
    for event in calls:
        status = str(event.get("status") or "")
        by_status[status] = by_status.get(status, 0) + 1
        calls_by_tool[event["action"]] = calls_by_tool.get(event["action"], 0) + 1
        if config == CONFIG_ACTION:
            failed = any(token in str(event.get("outcome") or "").lower()
                         for token in FAILURE_OUTCOME_TOKENS)
            if not failed and not event.get("parse_failure"):
                ok += 1
            if failed and not event.get("parse_failure"):
                marker = _fail_marker(event, config)
                failure_reasons[marker] = failure_reasons.get(marker, 0) + 1
            continue
        if status == STATUS_OK:
            ok += 1
        elif status not in (STATUS_CACHED,) and not event.get("parse_failure"):
            marker = _fail_marker(event, config)
            failure_reasons[marker] = failure_reasons.get(marker, 0) + 1
    return {
        "call_kind": "tool_status" if config == CONFIG_TOOL else "action_parse_outcome",
        "calls": len(calls),
        "ok": ok,
        "ok_rate": _rate(ok, len(calls)),
        "by_status": by_status,
        "calls_by_tool": calls_by_tool if config == CONFIG_TOOL else {},
        "parse_failures": parse_failures,
        "failure_reasons": failure_reasons,
    }


def image_tokens(args: argparse.Namespace) -> int:
    """Image token estimate for one VLM input image (28x28 patch by default)."""
    width, height = (args.image_size or (640, 360))
    divisor = float(args.image_token_divisor)
    if divisor <= 0:
        return 0
    return int(round(float(width) * float(height) / divisor))


def cost_stats(args: argparse.Namespace, summary: dict, events: list[dict],
               config: str) -> dict:
    """Full VLM cost: decision calls, internal calls, image/text tokens, wall time.

    ``vlm_calls_total`` = max(reported, decision + declared internal calls).
    The action harness counts its own scan/crop calls, so its total is complete;
    the tool loop's total is a lower bound unless the tools declare the calls
    they spend internally in their result data (contract field ``vlm_calls``).
    """
    decision = _as_int(summary.get("turns" if config == CONFIG_TOOL else "steps"))
    if decision is None:
        decision = len(events)
    reported = _as_int(summary.get("vlm_calls"))
    internal_declared = 0
    for event in events:
        internal_declared += int(event.get("reported_vlm_calls") or 0)
    total = max(reported or 0, decision + internal_declared)
    internal = max(0, total - decision)
    images_total = 0
    text_chars = 0
    for event in events:
        images_total += int(event.get("images_sent") or 0)
        text_chars += len(str(event.get("result_prompt") or ""))
        text_chars += len(str(event.get("raw_reply") or ""))
    if config == CONFIG_TOOL:
        if not images_total:
            images_total = decision * (int(args.context_frames or 0) + 1)
    else:
        images_per_decision = int(args.context_frames or 0) + (
            0 if getattr(args, "action_no_bev", False) else 1
        )
        images_per_scan = int(args.scan_views or 0) + (
            2 if float(args.scan_pitch_deg or 0.0) > 0.0 else 0
        )
        images_total = decision * images_per_decision + internal * images_per_scan
    text_chars = int(text_chars + float(args.prompt_chars_per_call or 0.0) * total)
    per_token = float(args.chars_per_token) or 4.0
    wall_time = _as_float(summary.get("wall_time_s"))
    return {
        "vlm_calls_decision": decision,
        "vlm_calls_reported": reported,
        "vlm_calls_internal_declared": internal_declared,
        "vlm_calls_total": total,
        "vlm_calls_internal": internal,
        "vlm_cost_complete": True if config == CONFIG_ACTION else bool(internal_declared),
        "images_total": images_total,
        "text_chars": text_chars,
        "image_tokens": image_tokens(args),
        "tokens_estimate": int(round(images_total * image_tokens(args) + text_chars / per_token)),
        "wall_time_s": wall_time,
    }


def success_stats(args: argparse.Namespace, summary: dict, events: list[dict],
                  target_pos: Any, start_pose: Any, *, radius_m: float | None = None,
                  success_dist_m: float | None = None) -> dict:
    """Ground-truth reach vs the runner's completion claim.

    ``reached`` uses the last traced pose and the same gate the runners use
    (``success_dist + target radius``). ``solved`` requires both the claim and
    the geometry, so ``already_there``-style claims never count as completion.
    """
    radius = _as_float(radius_m)
    if radius is None:
        radius = _as_float(getattr(args, "target_radius", 0.0)) or 0.0
    limited = _as_float(success_dist_m)
    if limited is None:
        limited = _as_float(getattr(args, "success_dist", None))
    gate = (limited or 0.0) + radius
    final_pose = None
    for event in reversed(events):
        if event.get("pose") is not None:
            final_pose = event["pose"]
            break
    if final_pose is None:
        final_pose = _as_point(start_pose)
    final_distance = _distance(final_pose, target_pos)
    reached = None if final_distance is None else bool(final_distance <= gate)
    runner_success = bool(summary.get("success"))
    return {
        "final_pose": final_pose,
        "final_distance_m": _rounded(final_distance),
        "gate_m": round(gate, 3),
        "reached": reached,
        "runner_success": runner_success,
        "success": runner_success,
        "solved": bool(reached) and runner_success,
        "claim_without_reach": bool(runner_success and reached is False),
    }


# ---------------------------------------------------------------------------
# per-episode extraction
# ---------------------------------------------------------------------------


def row_defaults(config: str) -> dict:
    """Metric keys every row carries, so an error row has the same shape."""
    return {
        "reached": None,
        "runner_success": False,
        "success": False,
        "solved": False,
        "claim_without_reach": False,
        "nav_attempts": 0,
        "effective_moves": 0,
        "effective_move_rate": None,
        "zero_displacement_events": 0,
        "zero_displacement_runs": 0,
        "target_lost_events": 0,
        "target_reacquired_events": 0,
        "reacquisition_rate": None,
        "fov_agreement": None,
        "calls": 0,
        "call_ok": 0,
        "call_ok_rate": None,
        "parse_failures": 0,
        "vlm_calls_total": None,
        "tokens_estimate": None,
        "wall_time_s": None,
        "tool_calls": None if config == CONFIG_ACTION else 0,
        "tool_ok": None if config == CONFIG_ACTION else 0,
        "tool_ok_rate": None,
        "finish_reason": None,
    }


def extract_metrics(args: argparse.Namespace, spec, out_dir: Path, run: dict) -> dict:
    """All Phase-1 metrics of one episode, from the runner's own artifacts."""
    out_dir = Path(out_dir)
    command = run.get("command") or build_command(args, spec, out_dir)
    row: dict[str, Any] = {
        "config": spec.config,
        "agent": CONFIG_LABELS.get(spec.config, spec.config),
        "task": spec.task.name,
        "instruction": spec.task.instruction,
        "seed": int(spec.seed),
        "yaw_mode": spec.yaw_mode,
        "start_yaw_deg": spec.start_yaw_deg,
        "out_dir": str(out_dir),
        "command": [str(part) for part in command],
        "returncode": run.get("returncode"),
        "error": run.get("error"),
        "elapsed_s": run.get("elapsed_s"),
        "skipped": False,
        "success_dist_m": round(
            float(spec.task.success_dist if spec.task.success_dist is not None
                  else args.success_dist), 3),
        "toolset": None,
        "required_tools_missing": [] if spec.config == CONFIG_ACTION else list(PHASE1_TOOLS),
        **row_defaults(spec.config),
    }
    if not (out_dir / SUMMARY_FILE).exists():
        row["error"] = row["error"] or f"missing {SUMMARY_FILE} in {out_dir}"
        return row
    summary = compare.read_json(out_dir / SUMMARY_FILE) or {}
    trace = compare.read_jsonl(out_dir / (
        TOOL_TRACE if spec.config == CONFIG_TOOL else ACTION_TRACE))
    if not trace:
        trace = compare._find_trace(out_dir, TOOL_TRACE, ACTION_TRACE, "trace.jsonl")

    start_pose = spec.task.start_pos
    events = timeline(trace, spec.config, start_pose=start_pose)
    movement = movement_stats(events, spec.task.target_pos, start_pose=start_pose,
                              min_progress_m=float(args.min_progress_m),
                              zero_run_min=int(args.zero_run_min_length))
    vision = vision_stats(events, spec.task.target_pos,
                          hfov_deg=float(args.fov_hfov_deg),
                          width=int(args.render_size[0]) if args.render_size else 320,
                          height=int(args.render_size[1]) if args.render_size else 180,
                          start_pose=start_pose, start_yaw_deg=spec.start_yaw_deg)
    calls = call_stats(events, spec.config)
    cost = cost_stats(args, summary, events, spec.config)
    success = success_stats(args, summary, events, spec.task.target_pos, start_pose,
                            radius_m=spec.task.target_radius,
                            success_dist_m=spec.task.success_dist
                            if spec.task.success_dist is not None else args.success_dist)

    row.update(movement)
    row.update(vision)
    row.update({key: value for key, value in calls.items() if key != "ok"})
    row.update(cost)
    row.update(success)
    row.update({
        "call_ok": calls["ok"],
        "call_ok_rate": calls["ok_rate"],
        "finish_reason": summary.get("finish_reason"),
        "skipped": bool(summary.get("skipped")),
        "min_distance_m": _as_float(summary.get("min_distance_m")),
        "path_length_m": _as_float(summary.get("path_length_m")),
        "initial_target_visible": summary.get("initial_target_visible"),
        "vlm_calls_reported": cost["vlm_calls_reported"],
        "vlm_budget": int(args.max_vlm_calls),
        "vlm_calls_over_budget": max(
            0, int(cost["vlm_calls_total"]) - int(args.max_vlm_calls)),
        "toolset": summary.get("toolset"),
    })
    if spec.config == CONFIG_TOOL:
        row.update({
            "tool_calls": calls["calls"],
            "tool_ok": calls["ok"],
            "tool_ok_rate": calls["ok_rate"],
            "required_tools_missing": required_tools_missing(summary),
        })
    else:
        row.update({
            "tool_calls": None,
            "tool_ok": None,
            "tool_ok_rate": None,
            "steps": _as_int(summary.get("steps")) or len(events),
        })
    if row["error"] is None and row["returncode"] not in (None, 0):
        row["error"] = f"runner exited with code {row['returncode']}"
    return row


# ---------------------------------------------------------------------------
# commands (the two runners' CLIs are the contract)
# ---------------------------------------------------------------------------


def budget_mapping(args: argparse.Namespace) -> dict:
    """The same VLM budget expressed in each runner's own knobs."""
    return {
        "max_vlm_calls": int(args.max_vlm_calls),
        "action": {"max_steps": int(args.max_vlm_calls)},
        "tool": {
            "max_turns": int(args.max_vlm_calls),
            "tool_budget": int(tool_budget(args)),
        },
        "note": ("the action loop may spend extra calls on scan steps and "
                 "done-crop reviews, the tool loop one call per turn plus the "
                 "tool-internal calls the tools declare; the actual per-episode "
                 "vlm_calls_total is recorded"),
    }


def tool_budget(args: argparse.Namespace) -> int:
    return int(args.tool_budget if args.tool_budget is not None else args.max_vlm_calls)


def action_backend(args: argparse.Namespace) -> str:
    if args.action_backend:
        return str(args.action_backend)
    backend = str(args.backend)
    if backend == "mock":  # the tool loop speaks {"tool": ...}: use the oracle
        return "mock:oracle"
    return backend


def build_action_command(args: argparse.Namespace, spec, out_dir: Path) -> list[str]:
    cmd = [sys.executable, "-m", "scripts.run_vlm_nav_3dgs"]
    cmd += compare.scene_args(args, spec)
    cmd += ["--instruction", spec.task.instruction]
    cmd += ["--backend", action_backend(args)]
    cmd += compare.model_args(args)
    cmd += ["--max-steps", str(int(args.max_vlm_calls))]
    if args.context_frames is not None:
        cmd += ["--context-frames", str(int(args.context_frames))]
    if args.scan_views is not None:
        cmd += ["--scan-views", str(int(args.scan_views))]
    if args.scan_pitch_deg is not None:
        cmd += ["--scan-pitch-deg", str(float(args.scan_pitch_deg))]
    if args.action_no_look_around:
        cmd += ["--no-look-around"]
    if args.action_no_bev:
        cmd += ["--no-bev"]
    if not args.action_video:
        cmd += ["--no-video"]
    cmd += [str(extra) for extra in (args.action_arg or [])]
    cmd += ["--out", str(out_dir)]
    return cmd


def build_tool_command(args: argparse.Namespace, spec, out_dir: Path) -> list[str]:
    cmd = [sys.executable, "-m", "scripts.run_vlm_nav_tools"]
    cmd += compare.scene_args(args, spec)
    if args.render_size:
        cmd += ["--render-size", str(int(args.render_size[0])),
                str(int(args.render_size[1]))]
    cmd += ["--instruction", spec.task.instruction]
    cmd += ["--backend", str(args.tool_backend or args.backend)]
    cmd += compare.model_args(args)
    cmd += ["--toolset", str(args.toolset)]
    cmd += ["--max-turns", str(int(args.max_vlm_calls))]
    cmd += ["--tool-budget", str(tool_budget(args))]
    if args.context_frames is not None:
        cmd += ["--context-frames", str(int(args.context_frames))]
    if args.scan_views is not None:
        cmd += ["--scan-views", str(int(args.scan_views))]
    if args.scan_pitch_deg is not None:
        cmd += ["--scan-pitch-deg", str(float(args.scan_pitch_deg))]
    cmd += [str(extra) for extra in (args.tool_arg or [])]
    cmd += ["--out", str(out_dir)]
    return cmd


def build_command(args: argparse.Namespace, spec, out_dir: Path) -> list[str]:
    if spec.config == CONFIG_ACTION:
        return build_action_command(args, spec, out_dir)
    return build_tool_command(args, spec, out_dir)


def build_episodes(args: argparse.Namespace, tasks: list) -> list:
    """One paired episode per (task, seed, yaw group) per agent."""
    episodes = []
    for task in tasks:
        for seed in args.seed_list:
            for yaw_mode in args.yaw_mode_list:
                yaw = compare.start_yaw_deg(
                    yaw_mode, task, seed, args.initial_yaw_half_fov_deg)
                for config in args.config_list:
                    episodes.append(compare.EpisodeSpec(config, task, seed,
                                                        yaw_mode, yaw))
    return episodes


def pairing_check(args: argparse.Namespace, episodes: list, commands: dict) -> dict:
    return compare.pairing_check(args, episodes, commands)


def pairing_report(args: argparse.Namespace, rows: list[dict]) -> dict:
    """Re-check the pairing from the recorded commands (works on a saved run)."""
    episodes = []
    commands: dict = {}
    for row in rows:
        command = row.get("command")
        if not command:
            return {"available": False, "cells": 0, "consistent": None, "issues": [],
                    "config_only_options": []}
        task = compare.TaskSpec(
            name=str(row.get("task") or "task"),
            instruction=str(row.get("instruction") or ""),
            target_pos=tuple(float(v) for v in (args.target_pos or (0.0, 0.0, 0.0))),
            target_radius=float(row.get("target_radius") or args.target_radius),
            start_pos=tuple(float(v) for v in args.start_pos),
        )
        spec = compare.EpisodeSpec(
            str(row.get("config")), task, int(row.get("seed") or 0),
            str(row.get("yaw_mode") or "visible"),
            float(row.get("start_yaw_deg") or 0.0),
        )
        episodes.append(spec)
        commands[(spec.config, spec.task.name, spec.seed, spec.yaw_mode)] = [
            str(part) for part in command
        ]
    report = compare.pairing_check(args, episodes, commands)
    report["available"] = True
    return report


# ---------------------------------------------------------------------------
# aggregation, verdict, report
# ---------------------------------------------------------------------------


def aggregate(rows: list[dict]) -> dict:
    """Per-agent aggregate: pooled rates (numerator/denominator) plus sums/means."""
    out: dict[str, dict] = {}
    for config in CONFIG_ORDER:
        subset = [row for row in rows if row.get("config") == config]
        if not subset:
            continue
        ok = [row for row in subset if not row.get("error")]
        finish_reasons: dict[str, int] = {}
        nav_failure_reasons: dict[str, int] = {}
        calls_by_tool: dict[str, int] = {}
        for row in ok:
            reason = str(row.get("finish_reason") or "unknown")
            finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
            for key, value in (row.get("nav_failure_reasons") or {}).items():
                nav_failure_reasons[key] = nav_failure_reasons.get(key, 0) + int(value)
            for key, value in (row.get("calls_by_tool") or {}).items():
                calls_by_tool[key] = calls_by_tool.get(key, 0) + int(value)
        attempts = sum(int(row.get("nav_attempts") or 0) for row in ok)
        effective = sum(int(row.get("effective_moves") or 0) for row in ok)
        lost = sum(int(row.get("target_lost_events") or 0) for row in ok)
        reacquired = sum(int(row.get("target_reacquired_events") or 0) for row in ok)
        measured = sum(int(row.get("fov_measured_states") or 0) for row in ok)
        agreement = [
            (int(row.get("fov_measured_states") or 0) * float(row["fov_agreement"]))
            for row in ok if row.get("fov_agreement") is not None
        ]
        agreement_states = sum(int(row.get("fov_measured_states") or 0)
                               for row in ok if row.get("fov_agreement") is not None)
        calls = sum(int(row.get("calls") or 0) for row in ok)
        call_ok = sum(int(row.get("call_ok") or 0) for row in ok)
        tool_calls = [row for row in ok if row.get("tool_calls") is not None]
        stats = {
            "config": config,
            "agent": CONFIG_LABELS.get(config, config),
            "n": len(subset),
            "n_ok": len(ok),
            "n_errors": len(subset) - len(ok),
            "n_skipped": sum(1 for row in subset if row.get("skipped")),
            "n_solved": sum(1 for row in ok if row.get("solved")),
            "n_reached": sum(1 for row in ok if row.get("reached")),
            "success_rate": _rate(sum(1 for row in ok if row.get("solved")), len(ok)),
            "reached_rate": _rate(sum(1 for row in ok if row.get("reached")), len(ok)),
            "claim_without_reach_sum": sum(
                1 for row in ok if row.get("claim_without_reach")),
            "finish_reasons": finish_reasons,
            "nav_failure_reasons": nav_failure_reasons,
            "calls_by_tool": calls_by_tool,
            "nav_attempts_sum": attempts,
            "effective_moves_sum": effective,
            "effective_move_rate_pooled": _rate(effective, attempts),
            "effective_move_rate_mean": _mean(
                [row.get("effective_move_rate") for row in ok]),
            "wrong_direction_moves_sum": sum(
                int(row.get("wrong_direction_moves") or 0) for row in ok),
            "zero_displacement_events_sum": sum(
                int(row.get("zero_displacement_events") or 0) for row in ok),
            "zero_displacement_runs_sum": sum(
                int(row.get("zero_displacement_runs") or 0) for row in ok),
            "zero_displacement_episodes": sum(
                1 for row in ok if int(row.get("zero_displacement_runs") or 0) > 0),
            "max_zero_displacement_run_max": _max(
                [row.get("max_zero_displacement_run") for row in ok]),
            "target_lost_events_sum": lost,
            "target_reacquired_events_sum": reacquired,
            "reacquisition_rate_pooled": _rate(reacquired, lost),
            "fov_measured_states_sum": measured,
            "fov_agreement_pooled": (
                _rate(sum(agreement), agreement_states) if agreement_states else None
            ),
            "call_ok_rate_pooled": _rate(call_ok, calls),
            "tool_ok_rate_pooled": (
                _rate(sum(int(row.get("tool_ok") or 0) for row in tool_calls),
                      sum(int(row.get("tool_calls") or 0) for row in tool_calls))
                if tool_calls and any(int(row.get("tool_calls") or 0) for row in tool_calls)
                else None
            ),
        }
        for key in ("vlm_calls_decision", "vlm_calls_total", "vlm_calls_internal",
                    "tokens_estimate", "wall_time_s", "images_total"):
            stats[f"{key}_mean"] = _mean([row.get(key) for row in ok])
            stats[f"{key}_sum"] = _sum([row.get(key) for row in ok])
        stats["vlm_cost_complete_episodes"] = sum(
            1 for row in ok if row.get("vlm_cost_complete"))
        out[config] = stats
    return out


def paired_cells(rows: list[dict]) -> list[dict]:
    """The two agents of one (task, yaw group, seed) cell side by side."""
    groups: dict[str, dict] = {}
    for row in rows:
        cell = f"{row.get('task')}__{row.get('yaw_mode')}__seed{int(row.get('seed') or 0)}"
        entry = groups.setdefault(cell, {
            "cell": cell,
            "task": row.get("task"),
            "seed": row.get("seed"),
            "yaw_mode": row.get("yaw_mode"),
            "start_yaw_deg": row.get("start_yaw_deg"),
        })
        entry[str(row.get("config"))] = {
            "config": row.get("config"),
            "solved": bool(row.get("solved")),
            "reached": row.get("reached"),
            "runner_success": bool(row.get("runner_success")),
            "claim_without_reach": bool(row.get("claim_without_reach")),
            "finish_reason": row.get("finish_reason"),
            "nav_attempts": row.get("nav_attempts"),
            "effective_moves": row.get("effective_moves"),
            "effective_move_rate": row.get("effective_move_rate"),
            "zero_displacement_runs": row.get("zero_displacement_runs"),
            "zero_displacement_events": row.get("zero_displacement_events"),
            "target_lost_events": row.get("target_lost_events"),
            "target_reacquired_events": row.get("target_reacquired_events"),
            "reacquisition_rate": row.get("reacquisition_rate"),
            "fov_agreement": row.get("fov_agreement"),
            "tool_ok_rate": row.get("tool_ok_rate"),
            "vlm_calls_total": row.get("vlm_calls_total"),
            "tokens_estimate": row.get("tokens_estimate"),
            "wall_time_s": row.get("wall_time_s"),
            "out_dir": row.get("out_dir"),
            "error": row.get("error"),
        }
    return [groups[cell] for cell in sorted(groups)]


def verdict(rows: list[dict], aggregates: dict, cells: list[dict],
            args: argparse.Namespace) -> dict:
    """The Phase-1 acceptance verdict: one explicit check per requirement.

    ``pass`` is true only when no check is false: the Phase-1 tools were
    registered, the tool agent solved every cell the action loop solved, its
    effective move rate clears ``--min-effective-move-rate``, its
    zero-displacement loops stay within ``--max-zero-displacement-loops`` and
    the reported/estimated FOV states agree (``--min-fov-agreement``, only when
    the trace reports visibility at all).
    """
    tool_rows = [row for row in rows if row.get("config") == CONFIG_TOOL]
    tool_rows_ok = [row for row in tool_rows if not row.get("error")]
    action_rows = [row for row in rows if row.get("config") == CONFIG_ACTION]
    missing: list[str] = []
    for row in tool_rows_ok:
        for name in row.get("required_tools_missing") or []:
            if name not in missing:
                missing.append(name)

    action_solved = [cell for cell in cells
                     if (cell.get(CONFIG_ACTION) or {}).get("solved")]
    tool_solved_action = [
        cell for cell in action_solved if (cell.get(CONFIG_TOOL) or {}).get("solved")
    ]
    checks: dict[str, bool | None] = {}
    checks["required_tools"] = (not missing) if tool_rows_ok else None
    checks["success_parity"] = (
        len(tool_solved_action) == len(action_solved) if action_solved else None
    )
    tool_attempts = sum(int(row.get("nav_attempts") or 0) for row in tool_rows_ok)
    tool_effective = sum(int(row.get("effective_moves") or 0) for row in tool_rows_ok)
    tool_rate = _rate(tool_effective, tool_attempts)
    checks["effective_move_rate"] = (
        None if tool_rate is None
        else tool_rate >= float(args.min_effective_move_rate)
    )
    loops = sum(int(row.get("zero_displacement_runs") or 0) for row in tool_rows_ok)
    checks["zero_displacement"] = loops <= int(args.max_zero_displacement_loops)
    measured = sum(int(row.get("fov_measured_states") or 0) for row in tool_rows_ok)
    agreement_states = [
        (int(row.get("fov_measured_states") or 0), float(row["fov_agreement"]))
        for row in tool_rows_ok if row.get("fov_agreement") is not None
    ]
    if not measured or not agreement_states:
        checks["fov"] = None
    else:
        agreed = sum(states * rate for states, rate in agreement_states)
        total = sum(states for states, _ in agreement_states)
        checks["fov"] = (agreed / total) >= float(args.min_fov_agreement)
    if not tool_rows_ok:
        checks["ran"] = False
    result = {
        "pass": all(check is not False for check in checks.values()),
        "checks": checks,
        "paired_cells": len(cells),
        "action_solved_cells": len(action_solved),
        "tool_solved_action_cells": len(tool_solved_action),
        "tool_cells": len(cells),
        "required_tools_missing": missing,
        "tool_effective_move_rate": tool_rate,
        "action_effective_move_rate": (aggregates.get(CONFIG_ACTION) or {}).get(
            "effective_move_rate_pooled"),
        "zero_displacement_loops_total": {
            config: sum(int(row.get("zero_displacement_runs") or 0)
                        for row in rows if row.get("config") == config)
            for config in CONFIG_ORDER
        },
        "tool_zero_displacement_events": sum(
            int(row.get("zero_displacement_events") or 0) for row in tool_rows_ok),
        "tool_wrong_direction_moves": sum(
            int(row.get("wrong_direction_moves") or 0) for row in tool_rows_ok),
        "tool_reacquisition_rate": (aggregates.get(CONFIG_TOOL) or {}).get(
            "reacquisition_rate_pooled"),
        "tool_claims_without_reach": sum(
            1 for row in tool_rows_ok if row.get("claim_without_reach")),
        "thresholds": {
            "min_effective_move_rate": float(args.min_effective_move_rate),
            "max_zero_displacement_loops": int(args.max_zero_displacement_loops),
            "min_fov_agreement": float(args.min_fov_agreement),
            "progress_floor_m": float(args.min_progress_m),
            "zero_run_min_length": int(args.zero_run_min_length),
        },
        "agent_totals": {
            config: {
                "episodes": stats["n"],
                "errors": stats["n_errors"],
                "solved": stats["n_solved"],
                "effective_move_rate": stats["effective_move_rate_pooled"],
                "zero_displacement_runs": stats["zero_displacement_runs_sum"],
                "reacquisition_rate": stats["reacquisition_rate_pooled"],
                "fov_agreement": stats["fov_agreement_pooled"],
                "vlm_calls_total_mean": stats["vlm_calls_total_mean"],
                "tokens_estimate_mean": stats["tokens_estimate_mean"],
                "wall_time_s_mean": stats["wall_time_s_mean"],
            }
            for config, stats in aggregates.items()
        },
        "notes": [
            "tool_ok_rate counts status=ok tool results; cached/empty/blocked/"
            "failed are not ok",
            "nav attempts are the movement tools (navigate/approach_candidate/"
            "go_to_viewpoint/return_to_observation) and, for the action loop, "
            "move/waypoint/explore steps; turn/scan/hover are not attempts",
            "displacement is horizontal (xy) between consecutive traced poses; "
            "the direction is compared with the bearing to the target from the "
            "attempt's start pose",
            "effective move = displacement above the floor AND within 90 deg of "
            "the target bearing at the attempt's start pose",
            "zero-displacement loop = a run of consecutive nav attempts that "
            "moved nothing (single stalls are not loops)",
            "solved = the runner claimed success AND the last traced pose is "
            "inside the success gate (claims without reach are counted "
            "separately)",
            "tool-side vlm_calls_total is a lower bound unless the tools declare "
            "the calls they spend internally (result field vlm_calls)",
            "the FOV check only applies when the trace reports target_in_fov "
            "per turn (the tool loop does; the action loop does not)",
        ],
    }
    return result


def meta_block(args: argparse.Namespace, tasks: list, rows: list[dict],
               verdict_block: dict, pairing: dict | None = None) -> dict:
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_head": compare.git_head(),
        "world": args.world,
        "ply": args.ply,
        "backend": args.backend,
        "action_backend": action_backend(args),
        "tool_backend": str(args.tool_backend or args.backend),
        "model_path": args.model_path,
        "model": args.model,
        "device": args.device,
        "seeds": list(args.seed_list),
        "yaw_modes": list(args.yaw_mode_list),
        "tasks": [task.to_dict() for task in tasks],
        "n_episodes": len(rows),
        "toolset": str(args.toolset),
        "phase1_tools": list(PHASE1_TOOLS),
        "budget": budget_mapping(args),
        "token_model": {
            "image_tokens": image_tokens(args),
            "chars_per_token": float(args.chars_per_token),
            "prompt_chars_per_call": float(args.prompt_chars_per_call),
        },
        "thresholds": verdict_block.get("thresholds"),
        "out": str(Path(args.out)),
        "pairing_check": pairing if pairing is not None else pairing_report(args, rows),
        "verdict": verdict_block,
        "files": {"report_json": REPORT_JSON, "report_md": REPORT_MD},
    }


# ---------------------------------------------------------------------------
# markdown report
# ---------------------------------------------------------------------------


def render_markdown(meta: dict, rows: list[dict], aggregates: dict,
                    cells: list[dict], verdict_block: dict) -> str:
    lines: list[str] = []
    lines.append("# Phase 1 acceptance: tool loop vs action loop (paired)")
    lines.append("")
    lines.append(f"- created: {meta['created_at']}")
    lines.append(f"- git HEAD: {meta.get('git_head') or 'unknown'}")
    lines.append(f"- world: {meta['world']}"
                 + (f" | scene: {meta['ply']}" if meta.get("ply") else ""))
    lines.append(f"- backend: {meta['backend']} (action: {meta['action_backend']}, "
                 f"tool: {meta['tool_backend']}) | model: "
                 f"{meta.get('model_path') or meta.get('model') or '-'}")
    lines.append(f"- tasks: {', '.join(str(task.get('name')) for task in meta['tasks'])}"
                 f" | seeds: {meta['seeds']} | initial-yaw groups: {meta['yaw_modes']}")
    lines.append(f"- toolset: {meta['toolset']} | Phase-1 tools: "
                 f"{', '.join(meta['phase1_tools'])}")
    lines.append(f"- VLM budget: {meta['budget']['max_vlm_calls']} calls "
                 f"(action --max-steps {meta['budget']['action']['max_steps']}; "
                 f"tools --max-turns {meta['budget']['tool']['max_turns']}, "
                 f"--tool-budget {meta['budget']['tool']['tool_budget']})")
    lines.append(f"- thresholds: effective_move_rate >= "
                 f"{meta['thresholds']['min_effective_move_rate']}, "
                 f"zero_displacement_loops <= "
                 f"{meta['thresholds']['max_zero_displacement_loops']}, "
                 f"fov_agreement >= {meta['thresholds']['min_fov_agreement']}, "
                 f"progress floor {meta['thresholds']['progress_floor_m']} m, "
                 f"zero-run min length {meta['thresholds']['zero_run_min_length']}")
    pairing = meta.get("pairing_check") or {}
    lines.append(f"- pairing check: {pairing.get('cells', 0)} cells, "
                 f"consistent={pairing.get('consistent')} "
                 f"(config-only options: {', '.join(pairing.get('config_only_options') or []) or '-'})")
    lines.append(f"- tokens_estimate = images x {meta['token_model']['image_tokens']} "
                 f"+ text_chars / {meta['token_model']['chars_per_token']:g}")
    lines.append("")

    lines.append("## Aggregates per agent")
    lines.append("")
    lines.append("| agent | group | ep | solved | reached | call_ok_rate | "
                 "tool_ok_rate | effective_move_rate | zero_displacement_loops | "
                 "reacquisition_rate | fov_agreement | vlm_calls_total | "
                 "tokens_est | wall_s |")
    lines.append("|" + "---|" * 14)
    groups: list[tuple[str, list[dict]]] = [("all", rows)]
    for mode in meta["yaw_modes"]:
        subset = [row for row in rows if row.get("yaw_mode") == mode]
        if subset and len(subset) != len(rows):
            groups.append((str(mode), subset))
    for config in CONFIG_ORDER:
        for group, subset in groups:
            stats = aggregate(subset).get(config)
            if not stats:
                continue
            lines.append(
                f"| {config} | {group} | {stats['n']} | "
                f"{stats['n_solved']}/{stats['n_ok']} ({_fmt(stats['success_rate'])}) | "
                f"{_fmt(stats['reached_rate'])} | {_fmt(stats['call_ok_rate_pooled'])} | "
                f"{_fmt(stats['tool_ok_rate_pooled'])} | "
                f"{_fmt(stats['effective_move_rate_pooled'])} | "
                f"{stats['zero_displacement_runs_sum']} | "
                f"{_fmt(stats['reacquisition_rate_pooled'])} | "
                f"{_fmt(stats['fov_agreement_pooled'])} | "
                f"{_fmt(stats['vlm_calls_total_mean'], 1)} | "
                f"{_fmt(stats['tokens_estimate_mean'], 0)} | "
                f"{_fmt(stats['wall_time_s_mean'], 1)} |"
            )
    lines.append("")

    lines.append("## Paired cells (same scene, start, seed, budget)")
    lines.append("")
    lines.append("| cell | action solved | tool solved | action move rate | "
                 "tool move rate | action zero loops | tool zero loops | "
                 "action vlm_calls | tool vlm_calls |")
    lines.append("|" + "---|" * 9)
    for cell in cells:
        action = cell.get(CONFIG_ACTION) or {}
        tool = cell.get(CONFIG_TOOL) or {}
        lines.append(
            f"| {cell['cell']} | {int(bool(action.get('solved')))} | "
            f"{int(bool(tool.get('solved')))} | "
            f"{_fmt(action.get('effective_move_rate'))} | "
            f"{_fmt(tool.get('effective_move_rate'))} | "
            f"{_fmt(action.get('zero_displacement_runs'), 0)} | "
            f"{_fmt(tool.get('zero_displacement_runs'), 0)} | "
            f"{_fmt(action.get('vlm_calls_total'), 0)} | "
            f"{_fmt(tool.get('vlm_calls_total'), 0)} |"
        )
    lines.append("")

    lines.append("## Per-episode detail")
    lines.append("")
    lines.append("| agent | task | yaw | seed | solved | reached | finish | "
                 "nav_attempts | effective_moves | effective_move_rate | "
                 "zero_runs | zero_events | tool_calls | tool_ok_rate | "
                 "reacq_rate | fov_agreement | vlm_calls_total | tokens_est | "
                 "wall_s | notes |")
    lines.append("|" + "---|" * 20)
    for row in rows:
        notes = row.get("error") or ""
        if not notes and row.get("required_tools_missing"):
            notes = "missing tools: " + ", ".join(row["required_tools_missing"])
        if not notes and row.get("claim_without_reach"):
            notes = "claimed completion without reaching the gate"
        if not notes and row.get("zero_displacement_runs"):
            reasons = ", ".join(f"{k}:{v}" for k, v in
                                sorted((row.get("nav_failure_reasons") or {}).items()))
            notes = f"zero loop ({reasons})" if reasons else "zero loop"
        if not notes and row.get("zero_displacement_events"):
            notes = f"stalled attempts: {row['zero_displacement_events']}"
        lines.append(
            f"| {row['config']} | {row['task']} | {row['yaw_mode']} | {row['seed']} | "
            f"{int(bool(row.get('solved')))} | "
            f"{_fmt(row.get('reached'), 0)} | {row.get('finish_reason') or '-'} | "
            f"{_fmt(row.get('nav_attempts'), 0)} | "
            f"{_fmt(row.get('effective_moves'), 0)} | "
            f"{_fmt(row.get('effective_move_rate'))} | "
            f"{_fmt(row.get('zero_displacement_runs'), 0)} | "
            f"{_fmt(row.get('zero_displacement_events'), 0)} | "
            f"{_fmt(row.get('tool_calls'), 0)} | {_fmt(row.get('tool_ok_rate'))} | "
            f"{_fmt(row.get('reacquisition_rate'))} | "
            f"{_fmt(row.get('fov_agreement'))} | "
            f"{_fmt(row.get('vlm_calls_total'), 0)} | "
            f"{_fmt(row.get('tokens_estimate'), 0)} | "
            f"{_fmt(row.get('wall_time_s'), 1)} | {notes} |"
        )
    lines.append("")

    lines.append("## Tool contract and failure diagnosis")
    lines.append("")
    tool_rows = [row for row in rows if row.get("config") == CONFIG_TOOL]
    missing = verdict_block.get("required_tools_missing") or []
    lines.append(f"- Phase-1 tools missing from the tool runner: "
                 f"{', '.join(missing) if missing else 'none'}")
    tool_stats = aggregates.get(CONFIG_TOOL) or {}
    if tool_stats.get("calls_by_tool"):
        lines.append("- tool calls: " + ", ".join(
            f"{name}={count}" for name, count in sorted(tool_stats["calls_by_tool"].items())))
    if tool_stats.get("nav_failure_reasons"):
        lines.append("- nav failure markers: " + ", ".join(
            f"{name}={count}" for name, count in
            sorted(tool_stats["nav_failure_reasons"].items())))
    for row in tool_rows:
        if row.get("zero_displacement_partners"):
            lines.append(f"- {row['task']}/{row['yaw_mode']}/seed{row['seed']}: "
                         f"zero-displacement pairs "
                         f"{row['zero_displacement_partners']}")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- pass: {verdict_block['pass']}")
    for key, value in verdict_block["checks"].items():
        lines.append(f"- check `{key}`: {value}")
    lines.append(f"- tool_solved_action_cells: {verdict_block['tool_solved_action_cells']}"
                 f"/{verdict_block['action_solved_cells']} action-solved cells "
                 f"({verdict_block['paired_cells']} paired cells)")
    lines.append(f"- tool_effective_move_rate: "
                 f"{_fmt(verdict_block['tool_effective_move_rate'])} "
                 f"(action: {_fmt(verdict_block['action_effective_move_rate'])})")
    lines.append(f"- zero_displacement_loops_total: "
                 f"{verdict_block['zero_displacement_loops_total']}")
    lines.append(f"- tool_reacquisition_rate: "
                 f"{_fmt(verdict_block['tool_reacquisition_rate'])} | "
                 f"tool_wrong_direction_moves: {verdict_block['tool_wrong_direction_moves']}"
                 f" | tool_claims_without_reach: {verdict_block['tool_claims_without_reach']}")
    lines.append("")
    lines.append("## Metric definitions")
    lines.append("")
    for note in verdict_block["notes"]:
        lines.append(f"- {note}")
    lines.append("- `tool call success rate` = tool_ok_rate; `effective move rate` = "
                 "effective_move_rate; `zero-displacement loop count` = "
                 "zero_displacement_runs (zero_displacement_events = every stalled "
                 "attempt); `target re-acquisition rate` = reacquisition_rate; "
                 "`task success` = solved (reached AND runner_success); "
                 "`full VLM cost` = vlm_calls_total (+tokens_estimate, wall_time_s).")
    lines.append("- the action loop does not trace per-step visibility, so its "
                 "reacquisition_rate comes from the FOV estimate; the tool loop "
                 "reports `target_in_fov` per turn and its fov_agreement is the "
                 "FOV-deviation diagnostic.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # --- matrix ---
    parser.add_argument("--seeds", type=str, default="7")
    parser.add_argument("--initial-yaw-modes", "--initial-yaw-mode", type=str,
                        default="visible", dest="initial_yaw_modes",
                        help="comma list of visible,random,target; visible (the "
                             "default) is the simple Phase-1 task: the target is "
                             "inside the camera FOV at the start")
    parser.add_argument("--max-vlm-calls", type=int, default=20,
                        help="model-call budget per episode (action: --max-steps, "
                             "tools: --max-turns)")
    parser.add_argument("--tool-budget", type=int, default=None,
                        help="tool-call cap for the tool loop (default: --max-vlm-calls)")
    parser.add_argument("--toolset", type=str, default="full",
                        help="tool catalogue the tool loop sees (full = everything "
                             "the modules register, including the Phase-1 tools "
                             "once they land)")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--timeout", type=float, default=1800.0,
                        help="per-episode subprocess timeout in seconds")
    parser.add_argument("--jobs", type=int, default=1,
                        help="episodes to run in parallel (keep 1 for GPU/model runs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the paired commands without running them")
    parser.add_argument("--initial-yaw-half-fov-deg", type=float, default=40.0,
                        help="half horizontal FOV used to build the visibility groups")
    # --- task(s) ---
    parser.add_argument("--task-file", type=str, default=None)
    parser.add_argument("--task-json", type=str, action="append", default=None)
    parser.add_argument("--task-name", type=str, default="task0")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--target-name", type=str, default="target")
    parser.add_argument("--target-pos", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"), help="FLU, scene-centered")
    parser.add_argument("--target-radius", type=float, default=0.5)
    parser.add_argument("--start-pos", type=float, nargs=3, default=(0.0, 0.0, 1.0))
    parser.add_argument("--success-dist", type=float, default=2.0,
                        help="meters from the target surface")
    # --- scene (mirrors both runners) ---
    parser.add_argument("--world", choices=["cpu", "higs", "env", "super"], default="cpu")
    parser.add_argument("--ply", type=str, default=None)
    parser.add_argument("--scene-scale", type=float, default=None)
    parser.add_argument("--min-opacity", type=float, default=None)
    parser.add_argument("--combined-ply", type=str, default=None)
    parser.add_argument("--background-ply", type=str, default=None)
    parser.add_argument("--semantic-manifest", type=str, default=None)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--render-size", type=int, nargs=2, default=None,
                        metavar=("W", "H"),
                        help="camera resolution passed to the tool runner (the "
                             "action runner's world is fixed at 320x180, so any "
                             "other value is a pairing deviation)")
    parser.add_argument("--target-index", type=int, default=None)
    parser.add_argument("--max-init-attempts", type=int, default=8)
    parser.add_argument("--worker-python", type=str, default=None)
    parser.add_argument("--map-mode", type=str, default="global",
                        choices=["global", "accumulate", "blind"])
    parser.add_argument("--sensing-range-m", type=float, default=6.0)
    parser.add_argument("--sensing", type=str, default="ideal", choices=["ideal", "moge"])
    parser.add_argument("--moge-model-path", type=str, default=None)
    parser.add_argument("--env-overrides", type=str, nargs="*", default=None)
    # --- model ---
    parser.add_argument("--backend", type=str, default="mock",
                        help="mock | mock:oracle | hf | openai")
    parser.add_argument("--action-backend", type=str, default=None,
                        help="override the action loop's backend (default: mock -> "
                             "mock:oracle, otherwise --backend)")
    parser.add_argument("--tool-backend", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--image-size", type=int, nargs=2, default=(640, 360),
                        metavar=("W", "H"), help="VLM input resize (both runners)")
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--no-think", action="store_true")
    # --- agent loop / observability parity ---
    parser.add_argument("--context-frames", type=int, default=2,
                        help="frames per decision (both runners)")
    parser.add_argument("--scan-views", type=int, default=8)
    parser.add_argument("--scan-pitch-deg", type=float, default=50.0)
    parser.add_argument("--action-no-look-around", action="store_true")
    parser.add_argument("--action-no-bev", action="store_true")
    parser.add_argument("--action-video", action="store_true",
                        help="action loop only: also write episode.mp4 (default off)")
    parser.add_argument("--action-arg", type=str, action="append", default=None,
                        help="extra flag appended verbatim to the action runner")
    parser.add_argument("--tool-arg", type=str, action="append", default=None,
                        help="extra flag appended verbatim to the tool runner")
    # --- metric definitions / thresholds ---
    parser.add_argument("--min-progress-m", type=float, default=0.1,
                        help="displacement above which a nav attempt counts as "
                             "movement (below it: no progress)")
    parser.add_argument("--zero-run-min-length", type=int, default=2,
                        help="consecutive no-progress nav attempts that make a "
                             "zero-displacement loop")
    parser.add_argument("--fov-hfov-deg", type=float, default=86.0,
                        help="horizontal FOV of the camera used by the visibility "
                             "estimate (world.py default: 86)")
    parser.add_argument("--min-effective-move-rate", type=float, default=0.6)
    parser.add_argument("--max-zero-displacement-loops", type=int, default=0)
    parser.add_argument("--min-fov-agreement", type=float, default=0.8)
    parser.add_argument("--image-token-divisor", type=float, default=784.0)
    parser.add_argument("--chars-per-token", type=float, default=4.0)
    parser.add_argument("--prompt-chars-per-call", type=float, default=1200.0)
    return parser.parse_args(argv)


def resolve_matrix(args: argparse.Namespace) -> None:
    seeds = [int(part) for part in str(args.seeds).replace(" ", "").split(",") if part]
    args.seed_list = seeds or [0]
    modes = [part.strip() for part in str(args.initial_yaw_modes).split(",")
             if part.strip()]
    unknown = [mode for mode in modes if mode not in compare.YAW_MODES]
    if unknown:
        raise SystemExit(f"unknown --initial-yaw-modes entries {unknown}; "
                         f"pick from {compare.YAW_MODES}")
    args.yaw_mode_list = modes or ["visible"]
    args.config_list = list(CONFIG_ORDER)
    if args.max_vlm_calls <= 0:
        raise SystemExit("--max-vlm-calls must be positive")
    if args.world in ("cpu", "higs") and args.render_size and tuple(args.render_size) != (320, 180):
        print(f"[verify_phase1] warning: --render-size {tuple(args.render_size)} is "
              "passed to the tool runner only; the action runner's world is fixed "
              "at 320x180, so the two agents see different image sizes", flush=True)


def load_tasks(args: argparse.Namespace) -> list:
    return compare.load_tasks(args)


# ---------------------------------------------------------------------------
# episode execution
# ---------------------------------------------------------------------------


def run_verification(args: argparse.Namespace) -> dict:
    tasks = load_tasks(args)
    episodes = build_episodes(args, tasks)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "run_config.json", "w") as handle:
        json.dump(vars(args), handle, indent=2, default=str)

    commands: dict = {}
    for spec in episodes:
        dir_path = compare.episode_dir(out_root, spec)
        commands[(spec.config, spec.task.name, spec.seed, spec.yaw_mode)] = (
            build_command(args, spec, dir_path)
        )
    if args.dry_run:
        for spec in episodes:
            line = " ".join(str(part) for part in
                            commands[(spec.config, spec.task.name, spec.seed,
                                      spec.yaw_mode)])
            print(line)
        return {
            "dry_run": True,
            "n_episodes": len(episodes),
            "commands": [" ".join(str(part) for part in
                                  commands[(spec.config, spec.task.name, spec.seed,
                                            spec.yaw_mode)])
                         for spec in episodes],
        }

    rows_by_index: dict[int, dict] = {}
    started = time.time()
    jobs = max(1, int(args.jobs))

    def execute(index: int, spec) -> tuple[int, dict]:
        dir_path = compare.episode_dir(out_root, spec)
        dir_path.mkdir(parents=True, exist_ok=True)
        cmd = commands[(spec.config, spec.task.name, spec.seed, spec.yaw_mode)]
        print(f"[verify] {index + 1}/{len(episodes)} {spec.config} {spec.task.name} "
              f"{spec.slug}", flush=True)
        run = compare.run_command(cmd, float(args.timeout), dir_path / "run.log")
        run["command"] = cmd
        row = extract_metrics(args, spec, dir_path, run)
        print(f"[verify]   -> rc={row['returncode']} solved={row.get('solved')} "
              f"move_rate={row.get('effective_move_rate')} "
              f"zero_loops={row.get('zero_displacement_runs')} "
              f"reacq={row.get('reacquisition_rate')} "
              f"vlm_calls={row.get('vlm_calls_total')}"
              + (f" error={row['error']}" if row.get("error") else ""), flush=True)
        return index, row

    if jobs == 1 or len(episodes) <= 1:
        for index, spec in enumerate(episodes):
            key, row = execute(index, spec)
            rows_by_index[key] = row
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(execute, index, spec)
                       for index, spec in enumerate(episodes)]
            for future in futures:
                key, row = future.result()
                rows_by_index[key] = row
    rows = [rows_by_index[index] for index in range(len(episodes))]

    aggregates = aggregate(rows)
    cells = paired_cells(rows)
    verdict_block = verdict(rows, aggregates, cells, args)
    pairing = pairing_report(args, rows)
    meta = meta_block(args, tasks, rows, verdict_block, pairing=pairing)
    n_errors = sum(1 for row in rows if row.get("error"))
    summary = {
        "meta": meta,
        "verdict": verdict_block,
        "aggregates": aggregates,
        "cells": cells,
        "episodes": rows,
        "n_errors": n_errors,
        "wall_time_total_s": round(time.time() - started, 3),
    }
    with open(out_root / REPORT_JSON, "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    with open(out_root / REPORT_MD, "w") as handle:
        handle.write(render_markdown(meta, rows, aggregates, cells, verdict_block))
    print(f"[verify] wrote {out_root / REPORT_JSON} and {out_root / REPORT_MD}",
          flush=True)
    for config in CONFIG_ORDER:
        stats = aggregates.get(config)
        if not stats:
            continue
        print(f"[verify] {config:6s} n={stats['n']:<3d} "
              f"solved={stats['n_solved']}/{stats['n_ok']:<3d} "
              f"move_rate={_fmt(stats['effective_move_rate_pooled'])} "
              f"zero_loops={stats['zero_displacement_runs_sum']} "
              f"reacq={_fmt(stats['reacquisition_rate_pooled'])} "
              f"vlm_calls={_fmt(stats['vlm_calls_total_mean'], 1)} "
              f"tokens={_fmt(stats['tokens_estimate_mean'], 0)}", flush=True)
    print(f"[verify] verdict pass={verdict_block['pass']} checks="
          f"{json.dumps(verdict_block['checks'])}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    resolve_matrix(args)
    summary = run_verification(args)
    if summary.get("dry_run"):
        return 0
    if int(summary.get("n_errors") or 0):
        return 1
    return 0 if summary["verdict"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
