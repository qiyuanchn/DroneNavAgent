"""Conditional task-plan tools for the agent runtime (goal.md section 5).

goal.md asks the model to maintain a *short conditional plan* while a
long-horizon task runs - "go to doorway P3, check the left side; when a
candidate appears, compare its attributes; if there is no target, move on to P4;
if the path is blocked, ask for alternative viewpoints" - and to revise that plan
as new evidence arrives. The plan is agent state, not evidence: it lives in
``ctx.extra[PLAN_KEY]`` (episode-scoped, exactly like the viewpoint/nav-task
state of ``tool_impl_navigation``) and the two tools below are its only writers
and readers.

- ``update_plan`` ``{steps: [...]}``: replace the plan with a short ordered list
  of conditional steps. ``revision`` counts only accepted changes, so
  re-submitting an identical plan answers ``changed=False`` at the same
  revision instead of pretending the plan moved on. The same call carries the
  *execution progress* of that plan - ``done``/``done_when``/``current_step`` -
  so a turn can be scored against the step it was supposed to serve; a
  progress-only call (no ``steps``) updates the plan that already exists.
- ``get_plan`` ``{}``: the current ``{plan_id, revision, steps}`` plus its
  progress, for when the earlier turns have scrolled out of the prompt context.

Both return the structured ``{plan_id, steps, revision, ...}`` contract with a
``memory:plan_N`` evidence reference. The stored record is readable outside the
tool loop through :func:`plan_state` (the raw record), :func:`plan_text` (a
compact one-line-per-step rendering) and :func:`plan_progress` (the one-line
"step 2/4 (1 done) ... done_when: ..." summary the runtime's STATE block
carries) - goal.md's "the model maintains a conditional plan" contract, without
opening a second, free-text memory notebook.

P4: every step is a *resumable sub-task record* (:data:`SUBTASK_FIELDS`), not a
sentence - ``{id, goal, preconditions[], depends_on[], expected_observation,
done_when, failure_branch[], budget{tool_calls,turns}, status, evidence[]}``,
plus the executor's own bookkeeping (``spent``, ``reason``, ``updated_turn``). A
plain text step is wrapped into one automatically, so the old plan contract
still holds: the record keeps ``steps`` (the goal texts, in order), ``done``
(1-based indices) and ``current_step`` derived from the sub-task statuses.

The *executor* (:func:`executor_step`) advances those statuses after every
tool result - the runtime calls it once per executed call, no model turn
involved:

- a result that reports the sub-task's completion (``subgoal_complete``,
  ``verified``, a ``same`` verdict, ``progress_kind="task"``, ...) marks it
  ``done`` and *activates the next eligible sub-task* whose ``depends_on`` are
  satisfied - a whole chain of already-finished steps is walked in one call.
  **Done is evidence-bound**: the signal is a claim, and the step is only
  ``done`` when the evidence recorded for it satisfies its ``done_when`` -
  a place the result reports (inside the declared ``goal_position``/``locate_m``
  when the step declares one), a flown leg, a passed verification or a grounded
  identity, always with an EvidenceRef. Without a declared place the located
  check keeps its old meaning but says ``evidence_weak``. A claim whose evidence
  or whose earlier/``depends_on`` stages are not enough is ``claimed_done``
  (with the missing requirements/predecessors), which keeps the step open and
  stops the plan from handing over (P4/A: ep_007/ep_011 closed a stage on a
  search that had only a ray direction); the claim is re-judged on every call
  and promoted once evidence and order allow it;
- a result that reports a failed leg (``planner_failed`` / ``blocked`` /
  ``stalled`` / ...) marks it ``blocked`` and records the evidence tokens, so
  its ``failure_branch`` is the concrete next step and can lead the runtime's
  recovery advice;
- a sub-task whose ``budget`` is exceeded while it is still open is judged by
  the same evidence rule: evidence that satisfies ``done_when`` marks it
  ``done`` with ``reason=budget_with_evidence`` (the budget ran out, not the
  work), otherwise it is ``failed`` with ``budget_exceeded:<dimensions>``;
- a ``failed``/``blocked`` step does not stay the charge target forever
  (ep_011): the next result is charged to the first unfinished step its
  evidence can serve, or the next eligible pending step is activated. A failed
  step whose missing evidence arrives later becomes ``claimed_done`` while an
  earlier stage is open, and is promoted to ``done`` when that stage closes.

Every status change is a *transition* (``{"id", "from", "to", "reason",
"turn"}``) kept on the record, and every transition bumps the plan revision, so
the model reads the executor's progress in the same ``get_plan`` answer it reads
its own. ``update_plan`` answers the progress of the sub-tasks (current one,
done, blocked with their reason, failed, budget consumption).

An interrupted episode (budget/turns exhausted) leaves the records in the
context; :func:`resume_plan` adopts them for the next episode of the same task,
statuses, evidence and consumed budget included, so a re-run continues where the
last one stopped instead of starting over. A plan whose every sub-task is done
is *not* adopted - it belongs to the episode that finished it. The first
evidence turn of the resumed leg re-judges every restored ``done``/``claimed``
stage (:func:`evidence_verdict`): a stage whose confirmation the new result (or
the agent's own ledger) now rejects, or whose accepted confirmation never had an
independent position anchor, is demoted to ``claimed_done`` and counted as a
resume correction - the checkpoint's wrong belief is overturnable by the agent
itself, without ground truth.

Neither tool is cacheable: ``update_plan`` rewrites episode state (registered
``side_effects=True``, so it is serialized and never cached), and ``get_plan``
must always answer with the current revision instead of a replayed one.
"""

from __future__ import annotations

import copy
import json
import math
import re
import time
from typing import Any, Iterable, Mapping

from .tools import (
    CONFIRMATION_ACCEPTED,
    PROGRESS_TASK,
    STATUS_BLOCKED,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_OK,
    EvidenceRef,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    declared_progress_kind,
)

#: ``ctx.extra`` key holding the episode's plan record.
PLAN_KEY = "plan"
#: the agent's own confirmation evidence, re-judged once after a resume
CANDIDATE_STORE_KEY = "perception.candidates"
CONFIRMATION_LEDGER_KEY = "confirmation_ledger"
RESUME_RECHECK_KEY = "resume_recheck"
MAX_PLAN_STEPS = 12
MAX_STEP_CHARS = 240
MAX_NOTE_CHARS = 200
#: the one-line progress rendering stays short enough for a prompt block
MAX_PROGRESS_CHARS = 240
#: One string step may carry several conditions ("do A; if B do C") - the same
#: separators the goal.md example plan uses.
_STEP_SPLIT_RE = re.compile(r"[\n;；]+")
_STEP_TEXT_KEYS = ("text", "step", "action", "do")
#: Progress arguments of ``update_plan``: which step is being executed, which
#: ones are done, and what ends the whole plan. They are what the runtime's
#: STATE block reports as ``plan_progress``.
_PROGRESS_KEYS = ("current_step", "done", "done_when")

# ---------------------------------------------------------------------------
# P4: the sub-task record
# ---------------------------------------------------------------------------

SUBTASK_PENDING = "pending"
SUBTASK_ACTIVE = "active"
#: A result claimed the sub-task is finished, but the evidence recorded for it
#: does not satisfy its ``done_when`` yet. The step stays open - a later result
#: with the missing evidence promotes it to ``done`` - and the plan never hands
#: over to the next sub-task on a claim alone.
SUBTASK_CLAIMED_DONE = "claimed_done"
SUBTASK_DONE = "done"
SUBTASK_FAILED = "failed"
SUBTASK_BLOCKED = "blocked"
#: ``pending`` -> ``active`` -> ``done``; ``claimed_done`` (a completion signal
#: without the evidence its ``done_when`` asks for), ``blocked`` (a leg failed /
#: a precondition cannot be met) and ``failed`` (its budget ran out while open,
#: without the evidence that would close it).
SUBTASK_STATUSES = (
    SUBTASK_PENDING,
    SUBTASK_ACTIVE,
    SUBTASK_CLAIMED_DONE,
    SUBTASK_DONE,
    SUBTASK_FAILED,
    SUBTASK_BLOCKED,
)
#: The statuses of a sub-task that still has work in it.
SUBTASK_OPEN = (SUBTASK_PENDING, SUBTASK_ACTIVE, SUBTASK_CLAIMED_DONE)

# ---------------------------------------------------------------------------
# P4/A: evidence binding - what the executor accepts as "done_when satisfied"
# ---------------------------------------------------------------------------

#: Evidence marks derived from one tool result. ``ref`` is an EvidenceRef the
#: result carries; the others are what the result itself establishes:
#: a place it reports (``located``), a flown leg (``moved``), an arrival
#: (``reached``), a passed verification (``verified``) and a grounded/confirmed
#: identity (``identity``).
MARK_REF = "ref"
MARK_LOCATED = "located"
MARK_MOVED = "moved"
#: The result reports that the drone really *arrived* (``outcome=arrived`` /
#: ``task_status=arrived``). A flown leg alone (``moved``) is progress, not an
#: arrival; a movement stage closes on this mark or on a remaining distance
#: inside its ``reach_m`` (see :func:`evidence_verdict`).
MARK_REACHED = "reached"
MARK_VERIFIED = "verified"
MARK_IDENTITY = "identity"
#: The result is an *accepted* ``confirm_candidate`` taken on a fresh frame
#: with a location-consistent measurement. Version 2: a movement stage closes on
#: this mark, not on ``identity`` (a grounded candidate id alone).
MARK_CONFIRMED = "confirmed"
EVIDENCE_MARKS = (
    MARK_REF, MARK_LOCATED, MARK_MOVED, MARK_REACHED, MARK_VERIFIED,
    MARK_IDENTITY, MARK_CONFIRMED,
)
#: What one sub-task's ``done_when`` asks the evidence to show (in report order).
EVIDENCE_REF = "evidence_ref"
EVIDENCE_LOCATED = "located"
EVIDENCE_REACHED = "reached"
EVIDENCE_CONFIRMATION = "confirmation"

#: Scene-place keys of a result payload (the same vocabulary the long-horizon
#: verifier's ``positions_in`` reads). The drone's pose and a navigation
#: ``waypoint`` are filtered out below: a waypoint is where the drone went, not
#: where the object is.
_PLACE_KEY_TOKENS = (
    "position_flu",
    "target_position",
    "candidate_position",
    "object_position",
    "position",
    "place",
    "point",
    "center",
    "centre",
    "xyz",
)
#: Payload keys carrying the drone's own pose, never a scene place.
_POSE_KEYS = ("pose", "start_pose", "end_pose", "drone_pose", "current_pose", "last_pose")
#: A navigation waypoint is the drone's destination, not the object's position -
#: it would otherwise match the generic ``point`` token above.
_NON_PLACE_KEYS = ("waypoint",)
#: Movement tools whose ``target_position`` is where the drone *went* (a
#: viewpoint or a camera pose), never where an object is. Their arrival is reach
#: evidence, but it does not locate the stage's object.
_DESTINATION_TOOLS = ("go_to_viewpoint", "return_to_observation", "navigate")
#: Payload keys whose positive value means a leg was really flown.
_MOVEMENT_KEYS = ("moved_m", "progress_m", "segment_progress_m")
#: ``done_when`` vocabulary: what condition the step's evidence has to carry.
#: Both languages the plans are written in are read (the task files are Chinese
#: in the P4 runs, the schema hints are English).
_DONE_WHEN_LOCATE_WORDS = (
    "locat",
    "position",
    "within",
    "distance",
    "place",
    "in view",
    "visible",
    "sighted",
    "find",
    "found",
    "定位",
    "位置",
    "坐标",
    "距离",
    "以内",
    "看到",
    "看见",
    "视野",
    "找到",
)
_DONE_WHEN_REACH_WORDS = (
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
    "旁边",
)
_DONE_WHEN_CONFIRM_WORDS = (
    "verif",
    "confirm",
    "identity",
    "same instance",
    "确认",
    "核实",
    "验证",
    "证明",
    "通过",
)
#: Every field of one sub-task record, in the order it is rendered. The first
#: thirteen are the model's (the schema ``update_plan`` documents); ``spent``,
#: ``reason`` and ``updated_turn`` are written by the executor.
SUBTASK_FIELDS = (
    "id",
    "goal",
    "preconditions",
    "depends_on",
    "expected_observation",
    "done_when",
    # the place the evidence has to point at (``located`` within ``locate_m``)
    "goal_position",
    "locate_m",
    # how close a movement stage has to get before its reach evidence counts
    "reach_m",
    "failure_branch",
    # the concrete call that continues the sub-task without a model turn
    "next_step",
    "budget",
    "status",
    "evidence",
    "spent",
    "reason",
    "updated_turn",
)

#: Bounds of the per-sub-task lists / budgets (a plan is "简短" - short).
MAX_LIST_ITEMS = 6
MAX_LIST_CHARS = 160
MAX_SUBTASK_BUDGET = 999
MAX_SUBTASK_EVIDENCE = 8
#: Reported places are kept longer than the token list: an in-range place must
#: not be evicted by later out-of-range results before the step is promoted.
MAX_SUBTASK_PLACES = 32
MAX_TRANSITIONS = 12
#: A single ``executor_step`` walks at most this many auto-activations.
MAX_AUTO_ADVANCE = MAX_PLAN_STEPS
MAX_INSTRUCTION_CHARS = 160
DEFAULT_SUBTASK_BUDGET = {"tool_calls": 0, "turns": 0}
#: ``located`` evidence is checked against the sub-task's ``goal_position``
#: within ``locate_m``. When the place is declared but no tolerance is, the
#: number in ``done_when`` ("within 2 m", "2 m 以内") is used; failing that this
#: default is the tolerance (the corridor task's ``DEFAULT_STAGE_LOCATE_M``).
DEFAULT_LOCATE_M = 2.0
#: A movement stage's default reach tolerance when neither ``reach_m`` nor a
#: distance in ``done_when`` names one (the long-horizon driver's
#: ``DEFAULT_STAGE_REACH_M``).
DEFAULT_REACH_M = 1.5
#: Being on the candidate's standoff shell is a reach even when the stage centre
#: is farther than ``reach_m``: ``remaining_m`` is the distance beyond that
#: shell (the approach skill's arrive radius).
CANDIDATE_REACH_TOLERANCE_M = 0.35
MAX_LOCATE_M = 1000.0
_DISTANCE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:m(?![a-z])|米|meters?|metres?)",
    re.IGNORECASE,
)
#: A plan step / failure branch written as one concrete call:
#: ``observe``, ``observe {"mode": "forward"}`` or
#: ``go_to_viewpoint({"viewpoint_id": "vp_9"})``.
_CALL_TEXT_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_\-]*)\s*(.*)$", re.S)

#: Tools that are not an execution step of a sub-task: plan bookkeeping and the
#: episode-closing ``finish`` never charge a sub-task's budget or move its status.
EXECUTOR_SKIP_TOOLS = ("update_plan", "get_plan", "finish")

#: The movement outcomes that mean "this leg failed" - the same vocabulary the
#: runtime's ``BAD_OUTCOMES`` recovery rules read, kept in sync by a contract
#: test so the executor and the recovery advice cannot drift apart.
FAILURE_OUTCOMES = (
    "planner_failed",
    "blocked",
    "collision",
    "stalled",
    "no_progress",
    "outside_bound",
    "leg_timeout",
    "clamped",
    "already_at_pose",
    "repeating_move",
)
#: Task-level signals of a tool result (the runtime's ``TASK_SIGNAL_KEYS``): a
#: result carrying one of them satisfies the ``done_when`` of the open sub-task.
TASK_SIGNAL_KEYS = (
    "subgoal_complete",
    "subgoal_completed",
    "goal_reached",
    "task_complete",
    "task_completed",
    "identity_confirmed",
    "object_confirmed",
    "confirmed",
    "verified",
    "satisfied",
    "success",
    # the executor's own vocabulary: a tool that knows which sub-task it served
    # may say so directly
    "subtask_complete",
    "step_complete",
    "done_when_met",
)
#: ``verdict`` values of ``compare_candidates`` that mean "same instance".
TASK_VERDICTS = ("same", "match", "identical", "confirmed")
#: ``status`` values of a result that mean the step it served is reached.
TASK_STATES = ("complete", "completed", "arrived")


def _extra(ctx: ToolContext) -> dict:
    """``ctx.extra`` repaired into a dict (the runtime may leave it None)."""
    if not isinstance(ctx.extra, dict):
        ctx.extra = {}
    return ctx.extra


def _step_text(item: Any) -> str:
    """One normalized step string from a model-supplied step entry."""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, Mapping):
        for key in _STEP_TEXT_KEYS:
            value = item.get(key)
            if isinstance(value, (str, int, float)) and str(value).strip():
                return str(value).strip()
        pairs = [
            f"{key}={value}"
            for key, value in item.items()
            if isinstance(value, (str, int, float, bool)) and str(value).strip()
        ]
        return "; ".join(pairs)
    if item is None or isinstance(item, bool):
        return ""
    return str(item).strip()


def normalize_steps(args: dict) -> tuple[list[str] | None, str | None]:
    """``(steps, error)`` from an ``update_plan`` argument mapping.

    Accepts a list of short strings (or step objects with a ``text``/``step``
    key) and a single string, which is split on newlines and ``;``/``；`` - the
    goal.md example plan is written as one semicolon-joined sentence. The result
    is capped in step count and step length so the plan stays "简短" (short).
    """
    raw = args.get("steps")
    if raw is None:
        raw = args.get("plan")  # tolerant alias
    if raw is None:
        return None, "missing_steps"
    if isinstance(raw, (str, bytes)):
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        items: list[Any] = _STEP_SPLIT_RE.split(text)
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        return None, "invalid_steps"
    steps: list[str] = []
    for item in items:
        text = _step_text(item)
        if text:
            steps.append(text[:MAX_STEP_CHARS])
        if len(steps) >= MAX_PLAN_STEPS:
            break
    if not steps:
        return None, "empty_plan"
    return steps, None


def _progress_step_index(value: Any, steps: list[str]) -> int | None:
    """1-based index of a step reference, or None.

    Accepts the index itself (``2`` / ``"2"``) and the step text (or an
    unambiguous prefix of it, so ``"go to vp_2"`` names the step ``"go to vp_2
    and scan"``). Anything else - including an index outside the plan - is None,
    which the caller reports instead of guessing.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        index = value
    elif isinstance(value, float) and float(value).is_integer():
        index = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            index = int(text)
        else:
            lowered = text.lower()
            for position, step in enumerate(steps, 1):
                if step.lower() == lowered:
                    return position
            for position, step in enumerate(steps, 1):
                if step.lower().startswith(lowered):
                    return position
            return None
    else:
        return None
    return index if 1 <= index <= len(steps) else None


def _done_indices(value: Any, steps: list[str], notes: list[str]) -> list[int]:
    """Sorted unique 1-based step indices of ``done`` (unusable entries noted)."""
    items = value if isinstance(value, (list, tuple, set)) else [value]
    indices: set[int] = set()
    for item in items:
        index = _progress_step_index(item, steps)
        if index is None:
            notes.append(f"ignored done entry {item!r} (not a step of this plan)")
            continue
        indices.add(index)
    return sorted(indices)


def _has_progress_args(args: dict) -> bool:
    """True when the call carries any progress argument with a value."""
    for key in _PROGRESS_KEYS:
        if args.get(key) not in (None, "", [], {}):
            return True
    for alias in ("step", "done_steps", "completed", "completion", "complete_when"):
        if args.get(alias) not in (None, "", [], {}):
            return True
    return False


#: ``done_when`` aliases of ``update_plan`` and of one step entry.
_DONE_WHEN_ALIASES = ("done_when", "completion", "complete_when", "success_when")


def explicit_progress(args: dict) -> dict:
    """Which progress arguments the call carries *with a value*.

    ``normalize_progress`` always answers a complete progress record (its
    defaults are what makes a plan executable); this tells the caller which parts
    of it were actually said by the model, so a re-stated plan does not reset the
    statuses the executor already advanced.
    """
    explicit: dict[str, str] = {}
    for name, aliases in (
        ("done", ("done", "done_steps", "completed")),
        ("current_step", ("current_step", "step")),
        ("done_when", _DONE_WHEN_ALIASES),
    ):
        for alias in aliases:
            if args.get(alias) not in (None, "", [], {}):
                explicit[name] = alias
                break
    return explicit


def normalize_progress(args: dict, steps: list[str]) -> tuple[dict, list[str]]:
    """``(progress, notes)`` of an ``update_plan`` call.

    ``progress`` is ``{"current_step": int, "done": [int], "done_when": str}`` -
    the execution state of the plan, not a second plan: ``current_step`` falls
    back to the first step that is not done, ``done`` is a sorted list of 1-based
    indices, and ``done_when`` is the condition that ends the whole plan. Entries
    that cannot be resolved to a step are listed in ``notes`` (reported to the
    model) and never fail the call - progress is advisory, the plan stays usable.
    """
    notes: list[str] = []
    raw_done = args.get("done", args.get("done_steps", args.get("completed")))
    done = [] if raw_done in (None, "") else _done_indices(raw_done, steps, notes)
    raw_current = args.get("current_step", args.get("step"))
    current = _progress_step_index(raw_current, steps)
    if current is None and raw_current not in (None, ""):
        notes.append(f"ignored current_step {raw_current!r} (not a step of this plan)")
    if current is None:
        current = next((i for i in range(1, len(steps) + 1) if i not in done), len(steps) or 1)
    done_when = args.get("done_when", args.get("completion", args.get("complete_when")))
    done_when = "" if done_when in (None, "") else str(done_when).strip()[:MAX_NOTE_CHARS]
    return {"current_step": int(current), "done": done, "done_when": done_when}, notes


# ---------------------------------------------------------------------------
# P4: sub-task normalization (the step schema and its patch form)
# ---------------------------------------------------------------------------

_SUBTASK_GOAL_KEYS = ("goal", "text", "step", "action", "do", "task")
_SUBTASK_LIST_ALIASES = {
    "preconditions": ("preconditions", "precondition", "requires", "requirements"),
    "depends_on": ("depends_on", "depends", "after", "needs"),
    "failure_branch": (
        "failure_branch",
        "failure",
        "on_failure",
        "fallback",
        "else",
        "if_blocked",
    ),
}
_SUBTASK_TEXT_ALIASES = {
    "expected_observation": (
        "expected_observation",
        "expect",
        "expected",
        "expected_view",
    ),
    "done_when": _DONE_WHEN_ALIASES,
}
_SUBTASK_POSITION_ALIASES = (
    "goal_position",
    "target_position",
    "goal_place",
    "position_flu",
)
_SUBTASK_NEXT_ALIASES = ("next_step", "next_action", "next_call", "next_tool")
_SUBTASK_SCALAR_KEYS = frozenset(
    _SUBTASK_GOAL_KEYS
    + (
        "id", "status", "done", "budget", "locate_m", "tolerance_m",
        "reach_m", "arrival_m", "reach_distance_m",
    )
    + _SUBTASK_POSITION_ALIASES
    + _SUBTASK_NEXT_ALIASES
    + tuple(name for aliases in _SUBTASK_LIST_ALIASES.values() for name in aliases)
    + tuple(name for aliases in _SUBTASK_TEXT_ALIASES.values() for name in aliases)
)
_ID_RE = re.compile(r"[^A-Za-z0-9_\-]+")


def _pick(data: Mapping, aliases: Iterable[str]) -> Any:
    """The first alias present with a value."""
    for key in aliases:
        value = data.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _bounded_int(value: Any, limit: int) -> int:
    """An int in ``[0, limit]``; anything unreadable is 0."""
    if isinstance(value, bool):
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(int(limit), number))


def _list_items(value: Any) -> list[str]:
    """A bounded, normalized list field (a string splits like a plan step)."""
    if value in (None, ""):
        return []
    if isinstance(value, (str, bytes)):
        text = value.decode("utf-8", "replace") if isinstance(value, bytes) else value
        items: list[Any] = _STEP_SPLIT_RE.split(text)
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    out: list[str] = []
    for item in items:
        text = _step_text(item)
        if text:
            out.append(text[:MAX_LIST_CHARS])
        if len(out) >= MAX_LIST_ITEMS:
            break
    return out


def _budget_of(value: Any) -> dict:
    """``{"tool_calls": int, "turns": int}`` from a number, a mapping or a phrase."""
    tool_calls, turns = 0, 0
    if isinstance(value, Mapping):
        tool_calls = value.get("tool_calls", value.get("calls", 0))
        turns = value.get("turns", 0)
    elif isinstance(value, str):
        numbers = re.findall(r"\d+", value)
        tool_calls = numbers[0] if numbers else 0
    else:
        tool_calls = value
    return {
        "tool_calls": _bounded_int(tool_calls, MAX_SUBTASK_BUDGET),
        "turns": _bounded_int(turns, MAX_SUBTASK_BUDGET),
    }


def _bounded_float(value: Any, limit: float = MAX_LOCATE_M) -> float:
    """A finite non-negative float in ``[0, limit]``; anything else is 0."""
    number = _finite_number(value)
    if number is None:
        return 0.0
    return max(0.0, min(float(limit), number))


def _finite_number(value: Any) -> float | None:
    """A finite float, or None (bools and unreadable values are not measurements)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _position_of(value: Any) -> list[float]:
    """A declared scene place as ``[x, y, z]`` (empty when it is not one)."""
    if isinstance(value, Mapping):
        value = value.get("position_flu", value.get("position", value.get("xyz")))
    point = _as_point(value)
    return [float(item) for item in point] if point is not None else []


def _next_step_of(value: Any) -> Any:
    """The declared continuation, kept structured when it is a mapping."""
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    text = str(value or "").strip()
    return text[:MAX_LIST_CHARS] if text else None


def _clean_id(value: Any) -> str:
    """A sub-task id the model can quote back (it is a tool-resolvable name)."""
    text = _ID_RE.sub("_", str(value or "").strip())[:24]
    return "" if text.strip("_") == "" else text


def _status_of(value: Any) -> str | None:
    token = str(value or "").strip().lower()
    return token if token in SUBTASK_STATUSES else None


def _blank_subtask(
    subtask_id: str = "",
    goal: str = "",
    *,
    status: str = SUBTASK_PENDING,
) -> dict:
    """A fresh sub-task record with every :data:`SUBTASK_FIELDS` field."""
    return {
        "id": subtask_id,
        "goal": goal,
        "preconditions": [],
        "depends_on": [],
        "expected_observation": "",
        "done_when": "",
        "goal_position": [],
        "locate_m": 0.0,
        "reach_m": 0.0,
        "failure_branch": [],
        "next_step": None,
        "budget": dict(DEFAULT_SUBTASK_BUDGET),
        "status": status,
        "evidence": [],
        "spent": {"tool_calls": 0, "turns": 0},
        "reason": "",
        "updated_turn": 0,
        # executor bookkeeping (not part of the model's schema): what the
        # evidence recorded for this sub-task has established so far, the
        # places it reported, and the ``done_when`` requirements it still misses
        "_evidence_marks": [],
        "_evidence_places": [],
        "_evidence_remaining": [],
        # where a movement result actually went (scopes its reach evidence)
        "_evidence_move_places": [],
        # the standoff shell the last movement aimed at (None when unknown)
        "_evidence_standoff": None,
        "_evidence_missing": [],
        "_evidence_weak": False,
        # the executor consumed this sub-task's static ``next_step`` once
        "_next_step_spent": False,
    }


def _copy_subtask(subtask: Mapping) -> dict:
    """A deep copy of one record (the lists must not be shared with the old plan)."""
    base = _blank_subtask()
    for field in SUBTASK_FIELDS:
        if field in subtask:
            base[field] = copy.deepcopy(subtask[field])
    base["_evidence_marks"] = [
        str(mark) for mark in subtask.get("_evidence_marks") or [] if str(mark) in EVIDENCE_MARKS
    ]
    base["_evidence_places"] = [
        [float(value) for value in point[:3]]
        for point in subtask.get("_evidence_places") or []
        if _as_point(point) is not None
    ][-MAX_SUBTASK_PLACES:]
    base["_evidence_missing"] = [str(item) for item in subtask.get("_evidence_missing") or []]
    base["_evidence_weak"] = bool(subtask.get("_evidence_weak"))
    base["_evidence_remaining"] = [
        float(value)
        for value in subtask.get("_evidence_remaining") or []
        if _finite_number(value) is not None
    ][-MAX_SUBTASK_PLACES:]
    base["_evidence_move_places"] = [
        [float(value) for value in point[:3]]
        for point in subtask.get("_evidence_move_places") or []
        if _as_point(point) is not None
    ][-MAX_SUBTASK_PLACES:]
    standoff = _finite_number(subtask.get("_evidence_standoff"))
    base["_evidence_standoff"] = None if standoff is None else float(standoff)
    base["_next_step_spent"] = bool(subtask.get("_next_step_spent"))
    return base


def _subtask_spec(entry: Any, *, known_ids: set[str]) -> tuple[dict | None, list[str]]:
    """``(spec, notes)`` of one ``update_plan`` step entry.

    A plain string entry carries a goal only (the legacy step); a mapping may
    declare any sub-task field, and its unknown scalar keys are rendered into the
    goal text exactly as before (``{"heading": "north"}`` -> ``"heading=north"``).
    ``spec["_patch"]`` is True when the entry only updates the sub-task its ``id``
    already names - the model's way to revise one sub-plan in place.
    """
    if not isinstance(entry, Mapping):
        goal = _step_text(entry)
        if not goal:
            return None, [f"ignored empty step entry {entry!r}"]
        return {"id": "", "goal": goal[:MAX_STEP_CHARS], "_patch": False}, []

    data = dict(entry)
    subtask_id = _clean_id(data.get("id"))
    goal = ""
    for key in _SUBTASK_GOAL_KEYS:
        value = data.get(key)
        if isinstance(value, (str, int, float)) and str(value).strip():
            goal = str(value).strip()
            break
    declared: dict[str, Any] = {}
    for field, aliases in _SUBTASK_LIST_ALIASES.items():
        value = _pick(data, aliases)
        if value is not None:
            declared[field] = _list_items(value)
    for field, aliases in _SUBTASK_TEXT_ALIASES.items():
        value = _pick(data, aliases)
        if value is not None:
            declared[field] = str(value).strip()[:MAX_NOTE_CHARS]
    position = _pick(data, _SUBTASK_POSITION_ALIASES)
    if position is not None:
        point = _position_of(position)
        if point:
            declared["goal_position"] = point
    locate = _pick(data, ("locate_m", "tolerance_m"))
    if locate is not None:
        declared["locate_m"] = _bounded_float(locate)
    reach = _pick(data, ("reach_m", "arrival_m", "reach_distance_m"))
    if reach is not None:
        declared["reach_m"] = _bounded_float(reach)
    next_step = _pick(data, _SUBTASK_NEXT_ALIASES)
    if next_step is not None:
        declared["next_step"] = _next_step_of(next_step)
    if data.get("budget") not in (None, ""):
        declared["budget"] = _budget_of(data["budget"])
    status = _status_of(data.get("status"))
    if status is None and isinstance(data.get("done"), bool):
        status = SUBTASK_DONE if data["done"] else SUBTASK_PENDING
    if status is not None:
        declared["status"] = status
    has_goal = bool(goal)
    if not goal:
        # no goal text: the legacy "key=value" rendering, then the declared fields
        pairs = [
            f"{key}={value}"
            for key, value in data.items()
            if key not in _SUBTASK_SCALAR_KEYS
            and isinstance(value, (str, int, float, bool))
            and str(value).strip()
        ]
        goal = "; ".join(pairs)
        if not goal:
            goal = "; ".join(f"{key}={value}" for key, value in declared.items())
    patch = bool(subtask_id) and subtask_id in known_ids and not has_goal
    if not goal and not patch:
        return None, [f"ignored step entry {json.dumps(entry, default=str)[:80]}: no goal"]
    spec = {"id": subtask_id, "goal": goal[:MAX_STEP_CHARS], "_patch": patch, "_goal": has_goal}
    spec.update(declared)
    return spec, []


def _normalize_step_specs(
    args: dict, record: dict | None
) -> tuple[list[dict] | None, list[str], str | None]:
    """``(specs, notes, error)`` of the step entries of an ``update_plan`` call."""
    raw = args.get("steps")
    if raw is None:
        raw = args.get("subtasks")
    if raw is None:
        raw = args.get("plan")  # tolerant alias
    if raw is None:
        return None, [], "missing_steps"
    if isinstance(raw, (str, bytes)):
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        items: list[Any] = _STEP_SPLIT_RE.split(text)
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        return None, [], "invalid_steps"

    known_ids = {
        str(subtask.get("id"))
        for subtask in (record or {}).get("subtasks") or []
        if isinstance(subtask, Mapping)
    }
    specs: list[dict] = []
    notes: list[str] = []
    limit = max(MAX_PLAN_STEPS, len(known_ids))
    for item in items:
        spec, item_notes = _subtask_spec(item, known_ids=known_ids)
        notes.extend(item_notes)
        if spec is None:
            continue
        specs.append(spec)
        if len(specs) >= limit:
            break
    if not specs:
        return None, notes, "empty_plan"
    return specs, notes, None


def _normalize_patch_specs(
    args: dict, record: dict | None
) -> tuple[list[dict] | None, list[str], str | None]:
    """``(specs, notes, error)`` of the ``patch`` argument of an update."""
    raw = args.get("patch")
    if raw is None:
        raw = args.get("update_subtasks")
    if raw is None:
        return None, [], "missing_patch"
    items = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    known_ids = {
        str(subtask.get("id"))
        for subtask in (record or {}).get("subtasks") or []
        if isinstance(subtask, Mapping)
    }
    specs: list[dict] = []
    notes: list[str] = []
    for item in items:
        spec, item_notes = _subtask_spec(item, known_ids=known_ids)
        notes.extend(item_notes)
        if spec is None:
            continue
        if not spec.get("id"):
            notes.append("a patch entry needs the id of the sub-task it updates")
            continue
        specs.append(spec)
    return specs, notes, None


def normalize_subtasks(
    args: dict, record: dict | None = None
) -> tuple[list[dict] | None, list[str], str | None]:
    """``(specs, notes, error)`` of the step entries of an ``update_plan`` call.

    Public because the runtime and its tests read the same normalization the tool
    applies: a spec carries the fields the entry declared (plus ``_patch``).
    """
    return _normalize_step_specs(args, record)


# ---------------------------------------------------------------------------
# P4: the executor
# ---------------------------------------------------------------------------


def plan_state(ctx: ToolContext) -> dict | None:
    """The episode's plan record, or None when no plan has been set yet.

    Readable by any runtime component: ``{"plan_id", "revision", "steps",
    "subtasks", "current_step", "done", "done_when", "note", "episode_id",
    "updated_t"}``. The record belongs to the episode it was written in
    (``episode_id``), so a context reused for another episode reports no plan
    rather than leaking the previous one - :func:`resume_plan` is the one explicit
    way to adopt the plan of an interrupted episode.
    """
    record = _extra(ctx).get(PLAN_KEY)
    if not isinstance(record, dict) or record.get("episode_id") != ctx.episode_id:
        return None
    if isinstance(record.get("subtasks"), list) and record["subtasks"]:
        # the sub-task statuses are the authority: keep the legacy derived view
        # (steps/done/current_step) in step with them, including changes another
        # component made directly (the long-horizon driver marking a carried
        # stage done from its own evidence chain)
        _sync(record)
    return record


def plan_subtasks(ctx: ToolContext) -> list[dict] | None:
    """The sub-task records of the episode's plan (None when there is no plan)."""
    record = plan_state(ctx)
    if record is None:
        return None
    subtasks = record.get("subtasks")
    if not isinstance(subtasks, list):
        # a record written before the sub-task schema existed: wrap its steps
        subtasks = [_blank_subtask(goal=str(step)) for step in record.get("steps") or []]
        record["subtasks"] = subtasks
    return subtasks


def current_subtask(ctx: ToolContext) -> dict | None:
    """The sub-task the plan is executing (None when there is no plan)."""
    subtasks = plan_subtasks(ctx)
    if not subtasks:
        return None
    index = _current_index(subtasks)
    return subtasks[index - 1] if index else None


def subtask_ground_instruction(ctx: ToolContext) -> str:
    """The current sub-task as a re-grounding instruction ("" without a plan).

    A recovery call must re-locate *this* stage's object, not restate the whole
    task: the sub-task's goal leads, its relation anchor
    (``expected_observation``) and its declared preconditions follow. The
    global instruction is never mixed in - an automatic ``ground_target`` that
    repeats the three-stage sentence dilutes the identity the stage is about
    (goal.md v1, seed23).
    """
    subtask = current_subtask(ctx)
    if not isinstance(subtask, Mapping):
        return ""
    parts: list[str] = []
    goal = str(subtask.get("goal") or "").strip()
    if goal:
        parts.append(goal)
    anchor = str(subtask.get("expected_observation") or "").strip()
    if anchor and anchor.casefold() not in goal.casefold():
        parts.append(anchor)
    for value in subtask.get("preconditions") or ():
        text = str(value or "").strip()
        if text and text.casefold() not in " ".join(parts).casefold():
            parts.append(text)
        if len(parts) >= 3:
            break
    return " - ".join(parts)[:MAX_STEP_CHARS]


def mark_next_step_spent(
    ctx: ToolContext, origin: str = "subtask_next_step", subtask: dict | None = None
) -> bool:
    """Mark a static ``next_step`` consumed by the executor (P4 auto steps).

    A plan/sub-task ``next_step`` is a *step*, not a loop: once the executor has
    run it, the same call is not run again on the following routine turns
    (r5 seed 11 re-ran a stale ``ground_target`` until the stagnant-grounding
    guard blocked the stage). The marker is cleared when the model patches the
    sub-task's ``next_step`` or re-states the plan.

    ``subtask`` is the record the call was issued for, captured *before* it ran:
    a step that completes its stage auto-activates the next one, and the marker
    must not land on that new (unspent) stage.
    """
    if str(origin) == "plan_next_step":
        record = plan_state(ctx)
        if isinstance(record, dict):
            record["_next_step_spent"] = True
            return True
        return False
    target = subtask if isinstance(subtask, dict) else current_subtask(ctx)
    if not isinstance(target, dict):
        return False
    target["_next_step_spent"] = True
    return True


def _current_index(subtasks: list[dict]) -> int:
    """1-based index of the sub-task the plan is on.

    The active one when there is one, else the blocked/failed one that needs the
    model's attention, else the claimed one that still owes evidence, else the
    first pending one, else the last step (the legacy "all done -> the last step"
    rendering).
    """
    if not subtasks:
        return 0
    for wanted in (
        (SUBTASK_ACTIVE,),
        (SUBTASK_BLOCKED, SUBTASK_FAILED),
        (SUBTASK_CLAIMED_DONE,),
        (SUBTASK_PENDING,),
    ):
        for position, subtask in enumerate(subtasks, 1):
            if subtask.get("status") in wanted:
                return position
    return len(subtasks)


def _next_eligible(subtasks: list[dict]) -> int | None:
    """1-based index of the first pending sub-task whose dependencies are done."""
    by_id = {str(subtask.get("id")): subtask for subtask in subtasks}
    for position, subtask in enumerate(subtasks, 1):
        if subtask.get("status") != SUBTASK_PENDING:
            continue
        unmet = [
            dep
            for dep in subtask.get("depends_on") or []
            if not _dependency_done(dep, subtasks, by_id)
        ]
        if not unmet:
            return position
    return None


def _dependency_done(dep: Any, subtasks: list[dict], by_id: Mapping[str, dict]) -> bool:
    """True when the referenced sub-task is done (an unknown reference never blocks)."""
    text = str(dep or "").strip()
    if not text:
        return True
    subtask = by_id.get(text)
    if subtask is None:
        index = _progress_step_index(
            int(text) if text.isdigit() else text,
            [str(item.get("goal") or "") for item in subtasks],
        )
        subtask = subtasks[index - 1] if index else None
    if subtask is None:
        return True  # an unresolvable dependency must not deadlock the plan
    return subtask.get("status") == SUBTASK_DONE


def _subtask_failure_branch(subtask: Mapping | None) -> list[str]:
    if not isinstance(subtask, Mapping):
        return []
    return [str(item) for item in subtask.get("failure_branch") or []]


def subtask_failure_branch(ctx: ToolContext) -> dict | None:
    """The failure branch in force, or None.

    The branch of the sub-task the plan is on: what the model said to do when
    this step fails. The runtime prepends it to its recovery advice and the STATE
    block reports its availability, so a blocked leg falls back to the plan's own
    alternative instead of an invented one.
    """
    subtasks = plan_subtasks(ctx)
    if not subtasks:
        return None
    index = _current_index(subtasks)
    subtask = subtasks[index - 1] if index else None
    branch = _subtask_failure_branch(subtask)
    if not branch:
        return None
    return {
        "index": index,
        "id": str(subtask.get("id") or ""),
        "goal": str(subtask.get("goal") or ""),
        "status": str(subtask.get("status") or ""),
        "reason": str(subtask.get("reason") or ""),
        "branch": branch,
    }


def _event_probe_verdict(subtask: Mapping, event: Mapping) -> dict:
    """``evidence_verdict`` of the sub-task *after* this result is recorded.

    The real record is not touched: a deep copy receives the event's evidence
    marks/places and is judged exactly as the executor would judge it.
    """
    probe = copy.deepcopy(subtask)
    _remember_event(probe, event)
    return evidence_verdict(probe)


def _first_feasible_subtask(
    subtasks: list[dict], event: Mapping, start: int
) -> tuple[int, dict] | None:
    """First unfinished step whose ``done_when`` this result would satisfy.

    A step that already has all the evidence its ``done_when`` asks for is
    skipped: it is waiting for order (a claim) or for a completion signal, and
    new evidence belongs to the next step. The scan starts at the failed/current
    position so a result that really closes the current step is charged there.
    """
    for position in range(start, len(subtasks) + 1):
        subtask = subtasks[position - 1]
        if subtask.get("status") == SUBTASK_DONE:
            continue
        if evidence_verdict(subtask)["ok"]:
            continue  # evidence-complete; only order/claim is missing
        if _event_probe_verdict(subtask, event)["ok"]:
            return position, subtask
    return None


def _charged_subtask(subtasks: list[dict], event: Mapping) -> tuple[int, dict] | None:
    """``(index, subtask)`` the call belongs to.

    A tool that knows which sub-task it served may name it (``subtask_id`` /
    ``step_id`` in its result); otherwise the call belongs to the sub-task the
    plan is on. The exception is the failed-current-step deadlock (ep_011): a
    ``failed``/``blocked`` current step, or one whose evidence is already
    complete while it waits for order, must not absorb the later stages'
    evidence. That result is charged to the first unfinished step whose
    ``done_when`` it satisfies; when no later step is evidence-feasible, the
    next eligible pending step is activated so the plan moves on instead of
    charging the dead step forever.
    """
    data = event.get("data") if isinstance(event.get("data"), Mapping) else {}
    for key in ("subtask_id", "step_id", "plan_step_id"):
        wanted = _clean_id(data.get(key))
        if not wanted:
            continue
        for position, subtask in enumerate(subtasks, 1):
            if str(subtask.get("id")) == wanted:
                return position, subtask
    index = _current_index(subtasks)
    if not index:
        return None
    current = subtasks[index - 1]
    status = str(current.get("status") or "")
    spent = current.get("spent") or {}
    projected = {
        "tool_calls": int(spent.get("tool_calls") or 0) + 1,
        "turns": int(spent.get("turns") or 0) + 1,
    }
    evidence_complete = (
        status != SUBTASK_DONE
        and evidence_verdict(current)["ok"]
        and not _event_closes_subtask(event)
        # a step that is about to exceed its budget closes as
        # ``budget_with_evidence``; do not starve it of this call
        and not _budget_exceeded(current.get("budget"), projected)
    )
    if status not in (SUBTASK_FAILED, SUBTASK_BLOCKED) and not evidence_complete:
        return index, current
    feasible = _first_feasible_subtask(subtasks, event, index)
    if feasible is not None:
        return feasible
    if status in (SUBTASK_FAILED, SUBTASK_BLOCKED):
        position = _next_eligible(subtasks)
        if position is not None:
            return position, subtasks[position - 1]
    return index, current


def _event_closes_subtask(event: Mapping) -> bool:
    """True when the result reports the sub-task it served as satisfied."""
    data = event.get("data") if isinstance(event.get("data"), Mapping) else {}
    if declared_progress_kind(data) == PROGRESS_TASK:
        return True
    for key in TASK_SIGNAL_KEYS:
        if data.get(key) is True:
            return True
    if str(data.get("verdict") or "").strip().lower() in TASK_VERDICTS:
        return True
    return str(data.get("status") or "").strip().lower() in TASK_STATES


def _event_blocked_reason(event: Mapping) -> str:
    """Why the call's sub-task is blocked ("" when it is not)."""
    data = event.get("data") if isinstance(event.get("data"), Mapping) else {}
    reason = str(
        data.get("blocked_reason")
        or data.get("reason")
        or data.get("outcome")
        or data.get("error")
        or ""
    ).strip().lower().replace(" ", "_")
    status = str(event.get("status") or "").strip().lower()
    if status in (STATUS_BLOCKED, STATUS_FAILED):
        return reason or status
    if reason in FAILURE_OUTCOMES:
        return reason
    return ""


def _evidence_tokens(event: Mapping) -> list[str]:
    """``kind:ref`` tokens of the result (bounded and de-duplicated by the caller)."""
    tokens: list[str] = []
    for ref in event.get("evidence") or ():
        if isinstance(ref, Mapping):
            kind, name = str(ref.get("kind") or ""), str(ref.get("ref") or "")
        else:
            kind, name = str(getattr(ref, "kind", "")), str(getattr(ref, "ref", ""))
        if kind and name:
            tokens.append(f"{kind}:{name}")
    return tokens


def _as_point(value: Any) -> tuple[float, float, float] | None:
    """A finite 3-vector from a payload value, or None (strings/bools are not places)."""
    if value is None or isinstance(value, (str, bytes, bool)):
        return None
    try:
        items = list(value)
    except TypeError:
        return None
    if len(items) < 3:
        return None
    try:
        point = tuple(float(item) for item in items[:3])
    except (TypeError, ValueError):
        return None
    return point if all(math.isfinite(item) for item in point) else None


def _places_in(payload: Any, _depth: int = 0) -> list[tuple[float, float, float]]:
    """Every *scene place* a tool result reports (candidates, targets, objects).

    The key vocabulary mirrors the long-horizon verifier's ``positions_in`` so the
    executor and the evidence verdict speak about the same places. The drone's
    own pose and a navigation ``waypoint`` are excluded: neither locates an
    object, and a completion claim has to point at the object it claims.
    """
    if _depth > 6:
        return []
    found: list[tuple[float, float, float]] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            lowered = str(key).lower()
            if lowered in _POSE_KEYS or any(
                token in lowered for token in _NON_PLACE_KEYS
            ):
                continue
            if any(token in lowered for token in _PLACE_KEY_TOKENS):
                point = _as_point(value)
                if point is not None:
                    found.append(point)
                    continue
                if isinstance(value, Mapping):
                    # e.g. relocalization: {position: [...]}
                    nested = _as_point(value.get("position_flu", value.get("position")))
                    if nested is not None:
                        found.append(nested)
                        continue
            found.extend(_places_in(value, _depth + 1))
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            found.extend(_places_in(item, _depth + 1))
    return found


def _arrived(data: Mapping) -> bool:
    """True when the result is an arrival, not merely a flown leg."""
    if str(data.get("outcome") or "").strip().lower() in ("arrived", "arrival"):
        return True
    if str(data.get("task_status") or "").strip().lower() == "arrived":
        return True
    return data.get("arrived") is True


def _moved(data: Mapping) -> bool:
    """True when the result reports a flown leg or an arrival."""
    if str(data.get("outcome") or "").strip().lower() in ("arrived", "arrival"):
        return True
    if str(data.get("task_status") or "").strip().lower() == "arrived":
        return True
    if data.get("arrived") is True:
        return True
    for key in _MOVEMENT_KEYS:
        value = data.get(key)
        if isinstance(value, bool):
            continue
        try:
            if float(value) > 0.0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _identity_signal(data: Mapping) -> bool:
    """True when the result grounds/confirms an identity (a candidate, a target)."""
    for key in ("candidate_id", "target_id", "object_id"):
        if str(data.get(key) or "").strip():
            return True
    candidates = data.get("candidates")
    if isinstance(candidates, (list, tuple)):
        for item in candidates:
            if isinstance(item, Mapping) and (
                item.get("id") is not None or item.get("candidate_id") is not None
            ):
                return True
    for key in ("confirmed", "identity_confirmed", "object_confirmed", "goal_reached"):
        if data.get(key) is True:
            return True
    return str(data.get("verdict") or "").strip().lower() in TASK_VERDICTS


def _accepted_confirmation(event: Mapping) -> bool:
    """True when the result is a *fresh, location-consistent* accepted identity.

    A v2 acceptance needs the new-frame verdict **and** the fresh depth
    measurement: a legacy/v7 acceptance that never ran the position check is not
    identity-complete evidence for a movement stage.
    """
    if str(event.get("tool") or "") != "confirm_candidate":
        return False
    data = event.get("data") if isinstance(event.get("data"), Mapping) else {}
    if str(data.get("verdict") or "").strip().lower() != CONFIRMATION_ACCEPTED:
        return False
    if data.get("new_confirmation") is False:
        return False  # a replay of an earlier verdict is not new evidence
    checks = data.get("position_check")
    return isinstance(checks, Mapping) and checks.get("available") is True


def _evidence_marks(event: Mapping) -> set[str]:
    """What one tool result establishes (the evidence the executor can bind to).

    A failed leg contributes no completion evidence beyond the fact that it has
    references: a blocked/failed movement is not an arrival and its waypoint is
    not a located object. A result with no EvidenceRef carries no ``ref`` mark,
    so a claim built on it can never satisfy a sub-task.
    """
    data = event.get("data") if isinstance(event.get("data"), Mapping) else {}
    marks: set[str] = set()
    if _evidence_tokens(event):
        marks.add(MARK_REF)
    if _event_blocked_reason(event):
        return marks
    if data.get("verified") is True:
        marks.add(MARK_VERIFIED)
    if _identity_signal(data):
        marks.add(MARK_IDENTITY)
    if _accepted_confirmation(event):
        marks.add(MARK_CONFIRMED)
    if _places_in(data) and str(event.get("tool") or "") not in _DESTINATION_TOOLS:
        marks.add(MARK_LOCATED)
    if _moved(data):
        marks.add(MARK_MOVED)
    if _arrived(data):
        marks.add(MARK_REACHED)
    return marks


def _declared_call(value: Any) -> tuple[str, dict] | None:
    """A movement continuation in either structured or text form."""
    if isinstance(value, Mapping):
        tool = value.get("tool") or value.get("name")
        if not isinstance(tool, str) or not tool.strip():
            return None
        args = value.get("args", value.get("arguments"))
        return tool.strip(), dict(args) if isinstance(args, Mapping) else {}
    return parse_call_text(value)


#: Tools that fly the drone somewhere: a stage whose continuation is one of
#: them is a movement stage even when its ``done_when`` only names a place.
_MOVEMENT_TOOLS = (
    "approach_candidate",
    "go_to_viewpoint",
    "return_to_observation",
    "navigate",
)


def is_movement_subtask(subtask: Mapping) -> bool:
    """Whether a sub-task's completion is a *movement* condition.

    Declared ``reach_m``, a movement-tool ``next_step`` or a ``done_when``/goal
    in the reach vocabulary each make it one. A bare legacy step (no
    ``done_when``) stays a reference-only step: it declares no condition to
    check, so it gets no reach requirement.
    """
    if _bounded_float(subtask.get("reach_m")) > 0:
        return True
    call = _declared_call(subtask.get("next_step"))
    if call is not None and call[0] in _MOVEMENT_TOOLS:
        return True
    done_when = str(subtask.get("done_when") or "").strip().lower()
    if not done_when:
        return False
    goal = str(subtask.get("goal") or "").strip().lower()
    return any(word in goal or word in done_when for word in _DONE_WHEN_REACH_WORDS)


def evidence_requirements(subtask: Mapping) -> list[str]:
    """What ``done_when`` asks the sub-task's evidence to show.

    Every sub-task needs an EvidenceRef (a claim must point at something); a
    ``done_when`` that names a placement, a movement or a confirmation adds the
    corresponding requirement. A movement stage (:func:`is_movement_subtask`)
    always asks for reach evidence, whether or not its ``done_when`` repeats a
    reach word: ``located`` alone is not an arrival. Version 2 also requires an
    *accepted* ``confirm_candidate`` under the current candidate/revision for
    every movement stage - the arrival proves the place, the confirmation
    proves the instance. A sub-task without ``done_when`` asks for nothing
    beyond the reference - there is no condition to check it against. The
    vocabulary is deliberately small and shared by both languages of the plans
    (English schema hints, Chinese task files).
    """
    text = str(subtask.get("done_when") or "").strip().lower()
    required = [EVIDENCE_REF]
    if _position_of(subtask.get("goal_position")):
        # a declared goal place is itself a condition: the evidence has to
        # point at it, whether or not done_when repeats the word "located"
        required.append(EVIDENCE_LOCATED)
    movement = is_movement_subtask(subtask)
    if not text:
        # a bare legacy step declares no condition; an explicit reach_m or a
        # movement-tool continuation is itself the movement condition
        if movement:
            required.append(EVIDENCE_REACHED)
            # version 2: an arrival is not identity, even when the bare step
            # does not spell the word "confirm"
            required.append(EVIDENCE_CONFIRMATION)
        return required
    if any(word in text for word in _DONE_WHEN_LOCATE_WORDS):
        if EVIDENCE_LOCATED not in required:
            required.append(EVIDENCE_LOCATED)
    if movement and EVIDENCE_REACHED not in required:
        required.append(EVIDENCE_REACHED)
    if any(word in text for word in _DONE_WHEN_REACH_WORDS):
        if EVIDENCE_REACHED not in required:
            required.append(EVIDENCE_REACHED)
    if movement and EVIDENCE_CONFIRMATION not in required:
        # every movement stage carries an identity question: the candidate the
        # navigation approached has to be accepted by confirm_candidate under
        # the current candidate/revision (goal.md version 2)
        required.append(EVIDENCE_CONFIRMATION)
    if any(word in text for word in _DONE_WHEN_CONFIRM_WORDS):
        if EVIDENCE_CONFIRMATION not in required:
            required.append(EVIDENCE_CONFIRMATION)
    return required


def _distance(a: Iterable[float], b: Iterable[float]) -> float:
    """Euclidean distance between two 3-vectors."""
    return math.dist([float(item) for item in a], [float(item) for item in b])


def _done_when_distance(subtask: Mapping) -> float | None:
    """The distance in metres a ``done_when`` names ("within 2 m", "2 米以内")."""
    match = _DISTANCE_RE.search(str(subtask.get("done_when") or ""))
    if match is None:
        return None
    return _bounded_float(match.group(1))


def _reach_limit(subtask: Mapping) -> float:
    """The tolerance a movement stage's remaining distance is checked against."""
    declared = _bounded_float(subtask.get("reach_m"))
    if declared > 0:
        return declared
    named = _done_when_distance(subtask)
    return named if named and named > 0 else DEFAULT_REACH_M


def _locate_limit(subtask: Mapping) -> float:
    """The tolerance a declared ``goal_position`` is checked against."""
    declared = _bounded_float(subtask.get("locate_m"))
    if declared > 0:
        return declared
    named = _done_when_distance(subtask)
    return named if named and named > 0 else DEFAULT_LOCATE_M


def _confirmation_current(subtask: Mapping, mark_set: set[str]) -> bool:
    """Whether the sub-task's accepted confirmation is for its current instance.

    The confirmation event itself never redefines the stage's candidate: the
    accepted record is compared with the candidate/revision of the stage's
    grounding/movement evidence. A later candidate switch or re-grounding bumps
    the revision and makes the earlier acceptance stale - unless the switch was
    to a candidate within the inheritance radius, in which case
    ``select_scope_candidate`` re-bound the acceptance to the new id/revision.
    """
    if MARK_CONFIRMED not in mark_set:
        return False
    record = subtask.get("_evidence_confirmed")
    if not isinstance(record, Mapping):
        return False
    if record.get("invalidated"):
        # the resumed-leg recheck overturned this confirmation
        return False
    current_candidate = str(subtask.get("_evidence_candidate") or "")
    confirmed_candidate = str(record.get("candidate_id") or "")
    if (
        current_candidate
        and confirmed_candidate
        and current_candidate != confirmed_candidate
    ):
        return False
    current_revision = subtask.get("_evidence_revision")
    confirmed_revision = record.get("evidence_revision")
    if current_revision is not None and confirmed_revision is not None:
        try:
            if int(current_revision) != int(confirmed_revision):
                return False
        except (TypeError, ValueError):
            return False
    return True


def evidence_verdict(subtask: Mapping) -> dict:
    """``{ok, requirements, marks, missing, evidence_weak, ...}`` of one sub-task.

    ``ok`` is the executor's ``done`` criterion: every requirement of the
    sub-task's ``done_when`` is backed by a mark taken from a tool result's own
    payload/EvidenceRef - never by the signal text alone. ``missing`` names the
    requirements still unbacked, so a ``claimed_done`` says exactly what to do.

    A ``located`` mark is type-level (the result reported *some* place). When the
    sub-task declares ``goal_position`` (+ ``locate_m`` or a distance in
    ``done_when``), the placed evidence only counts when one of its places is
    inside that tolerance (P4/A: ep_011 closed a stage on a place 2 m+ away from
    the target). Without a declared place the old behavior is kept, but the
    verdict is flagged ``evidence_weak`` because the place cannot be checked.

    A movement stage's ``reached`` requirement is satisfied by an arrival
    (``outcome=arrived`` / ``task_status=arrived``) or by a reported
    ``remaining_m`` inside ``reach_m`` (falling back to the ``done_when``
    distance, then :data:`DEFAULT_REACH_M`) - and it is scoped by the wider of
    two geometries: the movement target inside the tolerance of the located
    candidate (or its standoff shell) **or** inside ``reach_m`` of the declared
    stage place. ``reach_basis`` records which one justified it. A flown leg
    alone no longer satisfies it, and a viewpoint elsewhere is not an arrival at
    the object: ``located`` + ``moved`` (or a go_to arrival elsewhere) stays
    ``claimed_done``. Version 2 adds the ``confirmation`` requirement: it is
    satisfied by an accepted ``confirm_candidate`` marked ``confirmed`` and
    taken under the same candidate/revision as the stage's movement evidence; a
    later candidate switch or re-grounding makes the earlier acceptance stale,
    unless the new candidate is within
    :data:`vlm_nav.tools.CONFIRMATION_INHERIT_RADIUS_M` of the confirmed one in
    the same sub-task (the acceptance is inherited).
    """
    marks = [
        str(mark) for mark in subtask.get("_evidence_marks") or [] if str(mark) in EVIDENCE_MARKS
    ]
    mark_set = set(marks)
    requirements = evidence_requirements(subtask)
    movement = is_movement_subtask(subtask)
    confirmed = _confirmation_current(subtask, mark_set)
    goal = _position_of(subtask.get("goal_position")) or None
    limit = _locate_limit(subtask) if goal is not None else 0.0
    places = [
        point
        for point in subtask.get("_evidence_places") or []
        if _as_point(point) is not None
    ]
    nearest = (
        min((_distance(point, goal) for point in places), default=None)
        if goal is not None
        else None
    )
    located_in_range = nearest is not None and nearest <= limit
    reach_limit = _reach_limit(subtask)
    remaining = [
        number
        for number in (
            _finite_number(value) for value in subtask.get("_evidence_remaining") or []
        )
        if number is not None
    ]
    nearest_remaining = min(remaining, default=None)
    move_places = [
        point
        for point in subtask.get("_evidence_move_places") or []
        if _as_point(point) is not None
    ]
    # an arrival has to be an arrival *at the located object*: with a located
    # place on record, the movement target must be inside the reach tolerance
    # (a go_to_viewpoint elsewhere is progress, not this stage's reach). The
    # stage centre is the second, wider basis: it rescues an arrival that is
    # outside the centre tolerance but on the candidate's standoff shell (and
    # vice versa), and ``reach_basis`` says which geometry was used.
    standoff = _finite_number(subtask.get("_evidence_standoff"))
    candidate_limit = max(
        reach_limit, (standoff or 0.0) + CANDIDATE_REACH_TOLERANCE_M
    )
    candidate_reach = bool(places) and bool(move_places) and any(
        _distance(move, place) <= candidate_limit
        for move in move_places
        for place in places
    )
    center_reach = goal is not None and bool(move_places) and any(
        _distance(move, goal) <= reach_limit for move in move_places
    )
    scoped = True
    if EVIDENCE_REACHED in requirements and (places or goal is not None):
        scoped = candidate_reach or center_reach
    reach_basis = (
        "candidate" if candidate_reach else ("center" if center_reach else "")
    )
    reached = (
        MARK_REACHED in mark_set
        or (nearest_remaining is not None and nearest_remaining <= reach_limit)
    ) and scoped
    missing: list[str] = []
    for requirement in requirements:
        if requirement == EVIDENCE_REF and MARK_REF not in mark_set:
            missing.append(EVIDENCE_REF)
        elif requirement == EVIDENCE_LOCATED:
            if MARK_LOCATED not in mark_set:
                missing.append(EVIDENCE_LOCATED)
            elif goal is not None and not located_in_range:
                missing.append(EVIDENCE_LOCATED)
        elif requirement == EVIDENCE_REACHED and not reached:
            missing.append(EVIDENCE_REACHED)
        elif requirement == EVIDENCE_CONFIRMATION:
            if movement:
                # version 2: a movement stage needs the accepted
                # confirm_candidate of the current candidate/revision
                ok = confirmed
            else:
                ok = bool({MARK_VERIFIED, MARK_CONFIRMED} & mark_set) and (
                    MARK_CONFIRMED not in mark_set or confirmed
                )
            if not ok:
                missing.append(EVIDENCE_CONFIRMATION)
    if subtask.get("_resume_recheck_failed") and EVIDENCE_CONFIRMATION not in missing:
        # the resumed-leg recheck overturned this stage's confirmation: it
        # stays open until a fresh accepted confirmation arrives, even when the
        # stage's own done_when never spelled "confirm"
        missing.append(EVIDENCE_CONFIRMATION)
    return {
        "ok": not missing,
        "requirements": requirements,
        "marks": [mark for mark in EVIDENCE_MARKS if mark in mark_set],
        "missing": missing,
        "evidence_weak": EVIDENCE_LOCATED in requirements and goal is None,
        "goal_position": list(goal) if goal is not None else [],
        "locate_m": limit,
        "locate_distance": None if nearest is None else round(nearest, 3),
        "reach_m": round(reach_limit, 3) if EVIDENCE_REACHED in requirements else 0.0,
        "nearest_remaining_m": (
            None if nearest_remaining is None else round(nearest_remaining, 3)
        ),
        "reached": bool(reached) if EVIDENCE_REACHED in requirements else False,
        "reach_basis": reach_basis if EVIDENCE_REACHED in requirements else "",
        "reach_scoped": bool(scoped) if EVIDENCE_REACHED in requirements else False,
        "confirmed": bool(confirmed) if EVIDENCE_CONFIRMATION in requirements else False,
        "confirmation_candidate_id": (
            str((subtask.get("_evidence_confirmed") or {}).get("candidate_id") or "")
            if isinstance(subtask.get("_evidence_confirmed"), Mapping) else ""
        ),
        "confirmation_revision": (
            (subtask.get("_evidence_confirmed") or {}).get("evidence_revision")
            if isinstance(subtask.get("_evidence_confirmed"), Mapping) else None
        ),
    }


def parse_call_text(value: Any) -> tuple[str, dict] | None:
    """``(tool, args)`` when a plan step is exactly one concrete call.

    ``observe``, ``observe {"mode": "forward"}`` and
    ``go_to_viewpoint({"viewpoint_id": "vp_9"})`` qualify; prose that merely
    mentions a tool ("observe the corridor carefully", "先找到走廊里的木门") does
    not. This is the strict text half of the executor's deterministic step
    source: :func:`executor_step` and the runtime only ever execute a parsed
    call, never a sentence.
    """
    if not isinstance(value, str):
        return None
    text = value.strip().rstrip(".;。；")
    match = _CALL_TEXT_RE.match(text)
    if match is None:
        return None
    name, rest = match.group(1), match.group(2).strip()
    if not rest:
        return name, {}
    if rest.startswith("(") and rest.endswith(")"):
        rest = rest[1:-1].strip()
    elif not (rest.startswith("{") and rest.endswith("}")):
        return None
    if not rest:
        return name, {}
    try:
        args = json.loads(rest)
    except json.JSONDecodeError:
        return None
    return (name, dict(args)) if isinstance(args, dict) else None


def _blank_outcome() -> dict:
    """The executor's answer: what one call changed about the plan (if anything)."""
    return {
        "changed": False,
        "transitions": [],
        "decision": "",
        "charged": "",
        "completed": "",
        "claimed": "",
        "evidence_missing": [],
        "evidence_weak": False,
        "out_of_order": "",
        "predecessors": [],
        "promoted": [],
        "blocked": "",
        "blocked_reason": "",
        "failed": "",
        "budget_exceeded": [],
        "budget_with_evidence": "",
        "resume_recheck": {},
        "active": "",
        "current": 0,
        "failure_subtask": "",
        "failure_branch": [],
    }


def _transition(
    record: dict,
    subtask: dict,
    to_status: str,
    reason: str,
    turn: int,
    transitions: list[dict],
    event: Mapping | None = None,
) -> dict:
    """Move one sub-task to ``to_status`` and record the transition."""
    entry = {
        "id": str(subtask.get("id") or ""),
        "from": str(subtask.get("status") or ""),
        "to": to_status,
        "reason": reason,
        "turn": int(turn),
    }
    subtask["status"] = to_status
    subtask["reason"] = reason
    subtask["updated_turn"] = int(turn)
    if to_status == SUBTASK_DONE:
        subtask["_evidence_missing"] = []
    if event is not None:
        _remember_event(subtask, event)
    if to_status in (SUBTASK_DONE, SUBTASK_CLAIMED_DONE):
        subtask["_evidence_weak"] = bool(evidence_verdict(subtask)["evidence_weak"])
    transitions.append(entry)
    executor = record.setdefault("executor", {})
    history = executor.setdefault("transitions", [])
    history.append(dict(entry))
    del history[:-MAX_TRANSITIONS]
    record["revision"] = int(record.get("revision", 0)) + 1
    return entry


def _remember_evidence(
    subtask: dict,
    tokens: Iterable[str],
    marks: Iterable[str] = (),
    places: Iterable[Any] = (),
    remaining: Iterable[Any] = (),
    move_places: Iterable[Any] = (),
) -> None:
    """Keep the newest bounded evidence tokens, marks, places and ranges.

    The tokens are what the model reads (``observation:obs_3``); the marks are
    the executor's own, bounded classification of what those results really
    established (``ref``/``located``/``moved``/``reached``/``verified``/
    ``identity``); the places are the coordinates a ``located`` mark was derived
    from, so a declared ``goal_position`` can be checked against the evidence
    itself; the remaining distances are what a ``reached`` requirement is
    checked against when no arrival was reported; the move places are where a
    movement result actually went, so a reach cannot be claimed against an
    unrelated object. Everything lives on the sub-task record, so a resumed plan
    keeps what the interrupted leg established.
    """
    evidence = [str(item) for item in subtask.get("evidence") or []]
    for token in tokens:
        if token not in evidence:
            evidence.append(token)
    del evidence[:-MAX_SUBTASK_EVIDENCE]
    subtask["evidence"] = evidence
    known = [
        str(mark)
        for mark in subtask.get("_evidence_marks") or []
        if str(mark) in EVIDENCE_MARKS
    ]
    for mark in marks:
        if mark not in known:
            known.append(mark)
    subtask["_evidence_marks"] = [mark for mark in EVIDENCE_MARKS if mark in known]
    known_places = [
        [round(float(item), 3) for item in point[:3]]
        for point in subtask.get("_evidence_places") or []
        if _as_point(point) is not None
    ]
    for point in places:
        rounded = [round(float(item), 3) for item in point]
        if rounded not in known_places:
            known_places.append(rounded)
    del known_places[:-MAX_SUBTASK_PLACES]
    subtask["_evidence_places"] = known_places
    known_remaining = [
        float(value)
        for value in subtask.get("_evidence_remaining") or []
        if _finite_number(value) is not None
    ]
    for value in remaining:
        number = _finite_number(value)
        if number is None:
            continue
        rounded = round(number, 3)
        if rounded not in known_remaining:
            known_remaining.append(rounded)
    del known_remaining[:-MAX_SUBTASK_PLACES]
    subtask["_evidence_remaining"] = known_remaining
    known_moves = [
        [round(float(item), 3) for item in point[:3]]
        for point in subtask.get("_evidence_move_places") or []
        if _as_point(point) is not None
    ]
    for point in move_places:
        rounded = [round(float(item), 3) for item in point[:3]]
        if rounded not in known_moves:
            known_moves.append(rounded)
    del known_moves[:-MAX_SUBTASK_PLACES]
    subtask["_evidence_move_places"] = known_moves


def _remember_event(subtask: dict, event: Mapping) -> None:
    """Record everything one tool result established on its sub-task."""
    data = event.get("data") if isinstance(event.get("data"), Mapping) else {}
    marks = _evidence_marks(event)
    candidate = ""
    for key in ("candidate_id", "target_id", "object_id"):
        candidate = str(data.get(key) or "").strip()
        if candidate:
            break
    try:
        revision = int(data.get("evidence_revision"))
    except (TypeError, ValueError):
        revision = None
    if _accepted_confirmation(event):
        # a fresh, independently anchored confirmation is the evidence the
        # resumed-leg recheck was waiting for
        subtask.pop("_resume_recheck_failed", None)
        subtask["_evidence_confirmed"] = {
            "candidate_id": candidate,
            "evidence_revision": revision,
            "observation_ids": [
                str(item) for item in data.get("observation_ids") or () if item
            ],
            "position_delta_m": data.get("position_delta_m"),
        }
        # a confirmation does not redefine which candidate the stage is working
        # on: it is judged against the identity evidence already on record
        if candidate and not subtask.get("_evidence_candidate"):
            subtask["_evidence_candidate"] = candidate
    else:
        if candidate:
            subtask["_evidence_candidate"] = candidate
        if revision is not None:
            subtask["_evidence_revision"] = revision
    standoff = _finite_number(data.get("standoff_m"))
    if standoff is not None:
        subtask["_evidence_standoff"] = round(standoff, 3)
    movement = bool({MARK_MOVED, MARK_REACHED} & marks) or any(
        data.get(key) not in (None, "")
        for key in ("outcome", "task_status", "remaining_m")
    )
    _remember_evidence(
        subtask,
        _evidence_tokens(event),
        marks,
        _places_in(data) if MARK_LOCATED in marks else (),
        (data.get("remaining_m"),),
        _places_in(data) if movement else (),
    )


def _predecessors_open(index: int, subtasks: list[dict]) -> list[str]:
    """Ids of earlier (and declared-dependency) sub-tasks that are not done.

    A plan is an ordered chain of conditional stages: a later stage cannot be
    ``done`` while a stage it follows - by position or by ``depends_on`` - is
    still open. A completion signal for it is only a claim until they finish.
    """
    open_ids: list[str] = []
    for position, other in enumerate(subtasks, 1):
        if position >= index:
            break
        if other.get("status") != SUBTASK_DONE:
            open_ids.append(str(other.get("id") or f"st_{position}"))
    by_id = {str(item.get("id")): item for item in subtasks}
    for dep in subtasks[index - 1].get("depends_on") or []:
        if _dependency_done(dep, subtasks, by_id):
            continue
        text = str(dep).strip()
        if text and text not in open_ids:
            open_ids.append(text)
    return open_ids


def _claim_reason(missing: Iterable[str], open_ids: Iterable[str]) -> str:
    """The ``claimed_done`` reason: what evidence/order is still missing."""
    parts = []
    missing = list(missing)
    open_ids = list(open_ids)
    if missing:
        parts.append("evidence_insufficient:" + ",".join(missing))
    if open_ids:
        parts.append("predecessors_open:" + ",".join(open_ids))
    return ";".join(parts) or "evidence_insufficient"


def _promote_ready(record: dict, turn: int, transitions: list[dict]) -> list[str]:
    """Turn ``claimed_done``/``failed`` steps into ``done`` when evidence allows.

    This is the recovery half of the claim rule: a step that was claimed while
    its evidence was incomplete (P4/A: ep_007) or while an earlier stage was
    open (P4: seed 7) is re-judged on every call charged to the plan, and the
    moment ``evidence_verdict`` is satisfied *and* the predecessors are done it
    becomes ``done`` - no new completion signal required.

    A ``failed`` step (ep_011: its budget ran out while open) is re-judged the
    same way: once the missing evidence arrives it becomes ``claimed_done``
    while an earlier stage is still open, and the claim is promoted to ``done``
    when that predecessor closes. Only claimed/failed steps are promoted: an
    ``active`` step still needs the claim (or the budget outcome) that says the
    work is over, because its evidence marks are type-level - ``moved`` is any
    positive movement, not an arrival at the target.
    """
    promoted: list[str] = []
    for _ in range(MAX_AUTO_ADVANCE):
        subtasks = record.get("subtasks") or []
        ready: dict | None = None
        for position, subtask in enumerate(subtasks, 1):
            status = str(subtask.get("status") or "")
            if status not in (SUBTASK_CLAIMED_DONE, SUBTASK_FAILED):
                continue
            verdict = evidence_verdict(subtask)
            if not verdict["ok"]:
                continue
            open_ids = _predecessors_open(position, subtasks)
            if open_ids:
                if status != SUBTASK_FAILED:
                    continue
                reason = _claim_reason((), open_ids)
                if (
                    str(subtask.get("reason") or "") != reason
                    or list(subtask.get("_evidence_missing") or [])
                ):
                    subtask["_evidence_missing"] = []
                    _transition(
                        record,
                        subtask,
                        SUBTASK_CLAIMED_DONE,
                        reason,
                        turn,
                        transitions,
                    )
                continue
            ready = subtask
            break
        if ready is None:
            break
        _transition(record, ready, SUBTASK_DONE, "done_when_met", turn, transitions)
        promoted.append(str(ready.get("id") or ""))
    return promoted


def _resumed_confirmation_problem(
    ctx: ToolContext, subtask: Mapping, event: Mapping
) -> str:
    """Why a checkpoint's accepted confirmation no longer backs its stage.

    The resume path restores the agent's own belief verbatim, including a
    confirmation that was never independently anchored (the v2 position check
    re-projected the candidate and read delta 0.0, so every candidate passed).
    On the first evidence turn of the resumed leg the belief is re-judged:
    a fresh rejection of the same candidate, a candidate the agent's own ledger
    now rejects, or a confirmation with no independent position delta
    invalidates it. No ground truth is read here.
    """
    confirmed = subtask.get("_evidence_confirmed")
    if not isinstance(confirmed, Mapping):
        return ""
    candidate_id = str(confirmed.get("candidate_id") or "")
    if not candidate_id:
        return ""
    if confirmed.get("invalidated"):
        return str(confirmed.get("invalidated"))
    data = event.get("data") if isinstance(event.get("data"), Mapping) else {}
    if str(event.get("tool") or "") == "confirm_candidate":
        event_candidate = str(data.get("candidate_id") or "")
        verdict = str(data.get("verdict") or "").strip().lower()
        if event_candidate == candidate_id and verdict == CONFIRMATION_ACCEPTED:
            # the new evidence is a fresh accepted confirmation of this
            # candidate: it supersedes the checkpoint record
            return ""
        if event_candidate == candidate_id and verdict != CONFIRMATION_ACCEPTED:
            return "confirmation_rejected_by_new_evidence"
    extra = _extra(ctx)
    store = extra.get(CANDIDATE_STORE_KEY)
    candidate = store.get(candidate_id) if isinstance(store, Mapping) else None
    if isinstance(candidate, Mapping):
        status = str(
            candidate.get("status") or candidate.get("confirmation_status") or ""
        ).strip().lower()
        if status == "rejected":
            return "candidate_rejected"
    ledger = extra.get(CONFIRMATION_LEDGER_KEY)
    entry = ledger.get(candidate_id) if isinstance(ledger, Mapping) else None
    if isinstance(entry, Mapping) and str(entry.get("last_verdict") or "").lower() == "rejected":
        return "candidate_rejected"
    if confirmed.get("position_delta_m") is None:
        # accepted without an independent positional anchor: the current
        # contract cannot treat it as identity evidence
        return "confirmation_not_independently_anchored"
    return ""


def _invalidate_resumed_candidate(ctx: ToolContext, candidate_id: str, reason: str) -> None:
    """Record the agent-side rejection of a checkpoint's accepted candidate."""
    if not candidate_id:
        return
    extra = _extra(ctx)
    note = f"resume_recheck:{reason}"[:200]
    store = extra.get(CANDIDATE_STORE_KEY)
    if not isinstance(store, dict):
        store = {}
        extra[CANDIDATE_STORE_KEY] = store
    candidate = store.get(candidate_id)
    if not isinstance(candidate, dict):
        candidate = {"id": candidate_id}
        store[candidate_id] = candidate
    candidate["status"] = "rejected"
    candidate["confirmation_status"] = "rejected"
    candidate["confirmation_reason"] = note
    candidate["confirmation_missing"] = ["localization"]
    ledger = extra.get(CONFIRMATION_LEDGER_KEY)
    if not isinstance(ledger, dict):
        ledger = {}
        extra[CONFIRMATION_LEDGER_KEY] = ledger
    entry = ledger.get(candidate_id)
    if not isinstance(entry, dict):
        entry = {}
        ledger[candidate_id] = entry
    entry["last_verdict"] = "rejected"
    entry["missing"] = ["localization"]
    entry["reason"] = note


def _recheck_resumed_beliefs(
    ctx: ToolContext, record: dict, event: Mapping, turn: int,
    transitions: list[dict],
) -> dict:
    """Re-judge the checkpoint's done/claimed stages on the first resumed turn.

    ``resume_plan`` restores the statuses verbatim. The first evidence turn
    re-runs :func:`evidence_verdict` over every done/claimed stage; a stage whose
    evidence no longer holds - or whose confirmation is invalidated by the new
    result / the agent's own ledger / the missing independent position anchor -
    is demoted to ``claimed_done`` and counted as a resume correction. The
    correction lives on the plan record and in the confirmation ledger, so the
    offline checkpoint score sees the same agent-side overturn.
    """
    state = record.get(RESUME_RECHECK_KEY)
    if not isinstance(state, dict) or not state.get("pending"):
        return {}
    checked: list[str] = []
    corrected: list[str] = []
    details: list[dict] = []
    for position, subtask in enumerate(record.get("subtasks") or [], 1):
        if not isinstance(subtask, dict):
            continue
        status = str(subtask.get("status") or "")
        if status not in (SUBTASK_DONE, SUBTASK_CLAIMED_DONE):
            continue
        subtask_id = str(subtask.get("id") or f"st_{position}")
        checked.append(subtask_id)
        verdict = evidence_verdict(subtask)
        problem = _resumed_confirmation_problem(ctx, subtask, event)
        if verdict.get("ok") and not problem:
            details.append(
                {"id": subtask_id, "status": status, "corrected": False, "reason": ""}
            )
            continue
        missing = list(verdict.get("missing") or [])
        confirmed = subtask.get("_evidence_confirmed")
        if not problem and status != SUBTASK_DONE:
            # the stage was already open (``claimed_done``): re-judging it
            # confirms it still owes evidence, but nothing was overturned
            subtask["_evidence_missing"] = missing
            details.append(
                {
                    "id": subtask_id,
                    "status": status,
                    "corrected": False,
                    "reason": "already_open:" + (",".join(missing) or "evidence_insufficient"),
                    "missing": missing,
                }
            )
            continue
        if problem:
            if isinstance(confirmed, dict):
                confirmed["invalidated"] = problem
            candidate_id = (
                str(confirmed.get("candidate_id") or "")
                if isinstance(confirmed, Mapping) else ""
            )
            _invalidate_resumed_candidate(ctx, candidate_id, problem)
            if "confirmation" not in missing:
                missing.append("confirmation")
        reason = "resume_recheck:" + (
            problem or ",".join(missing) or "evidence_insufficient"
        )
        _transition(
            record, subtask, SUBTASK_CLAIMED_DONE, reason, turn, transitions
        )
        subtask["_resume_recheck_failed"] = True
        subtask["_evidence_missing"] = missing
        corrected.append(subtask_id)
        details.append(
            {
                "id": subtask_id,
                "status": status,
                "corrected": True,
                "reason": problem or "evidence_insufficient",
                "missing": missing,
            }
        )
    state.update(
        pending=False,
        turn=int(turn),
        checked=checked,
        corrected=corrected,
        details=details,
    )
    return dict(state)


def executor_step(ctx: ToolContext, event: Mapping | None) -> dict:
    """Advance the sub-task records from one finished tool call (P4).

    ``event`` is the executed call: ``{"tool", "args", "status", "data",
    "evidence", "progress_kind", "turn"}``. Returns what the step changed:

    ``{"changed", "transitions", "decision", "charged", "completed", "claimed",
    "evidence_missing", "evidence_weak", "out_of_order", "predecessors",
    "promoted", "blocked", "blocked_reason", "failed", "budget_exceeded",
    "budget_with_evidence", "active", "current", "failure_subtask",
    "failure_branch"}``

    - the call is charged to the sub-task it belongs to (its ``spent`` counter),
      and its result's evidence is recorded on that sub-task;
    - a completion signal marks that sub-task ``done`` **only when the recorded
      evidence satisfies its ``done_when``** (:func:`evidence_verdict`) *and*
      every earlier/``depends_on`` sub-task is done; otherwise the signal leaves
      it ``claimed_done`` (with the missing evidence and/or open predecessors),
      and the plan does not hand over. A claimed step is re-judged on every call
      charged to the plan, so the missing evidence arriving later (ep_007) or
      the predecessors finishing (a later stage) promotes it to ``done`` with no
      new signal. A ``done`` sub-task activates the next eligible one -
      repeatedly, so a chain of already-finished steps is walked inside one
      model turn (``decision == "subtask_switch"``, the routine decision node);
    - a ``located`` mark only satisfies a ``done_when`` that asks for a place
      when a reported place is inside ``locate_m`` of the declared
      ``goal_position`` (the distance in ``done_when`` is the fallback); with no
      declared place the old behavior is kept and the verdict is
      ``evidence_weak``;
    - a failed leg marks it ``blocked`` (its ``failure_branch`` takes over);
    - a sub-task whose budget is exceeded while it is still open is judged the
      same way: evidence that satisfies its ``done_when`` makes it ``done`` with
      ``reason=budget_with_evidence``, otherwise it is ``failed`` with
      ``budget_exceeded:<dimensions>``.

    Tools that are not an execution step (:data:`EXECUTOR_SKIP_TOOLS`) change
    nothing, and the global tool budget of the episode is never touched here -
    the registry owns it.
    """
    event = event if isinstance(event, Mapping) else {}
    outcome = _blank_outcome()
    record = plan_state(ctx)
    if not isinstance(record, dict):
        return outcome
    subtasks = plan_subtasks(ctx) or []
    if not subtasks:
        return outcome
    if str(event.get("tool") or "") in EXECUTOR_SKIP_TOOLS:
        return outcome

    turn = _bounded_int(event.get("turn"), 10**9)
    current_before = _current_index(subtasks)
    charged = _charged_subtask(subtasks, event)
    if charged is None:
        return outcome
    index, subtask = charged
    outcome["charged"] = str(subtask.get("id") or "")
    before = (index, str(subtask.get("id") or ""))
    transitions: list[dict] = []
    # a resumed plan is re-judged once, on its first evidence turn: the
    # checkpoint's done/claimed statuses are the agent's belief, not a verdict
    recheck = _recheck_resumed_beliefs(ctx, record, event, turn, transitions)
    if recheck:
        outcome["resume_recheck"] = recheck
    data = event.get("data") if isinstance(event.get("data"), Mapping) else {}
    named = any(_clean_id(data.get(key)) for key in ("subtask_id", "step_id", "plan_step_id"))
    if (
        not named
        and current_before
        and index != current_before
        and subtask.get("status") == SUBTASK_PENDING
    ):
        # the failed current step released this call: make the step it really
        # serves the active one, so the next call is charged here as well
        _transition(
            record,
            subtask,
            SUBTASK_ACTIVE,
            "redirected_after_failure",
            turn,
            transitions,
        )

    spent = subtask.setdefault("spent", {"tool_calls": 0, "turns": 0})
    spent["tool_calls"] = int(spent.get("tool_calls") or 0) + 1
    executor = record.setdefault("executor", {})
    if int(executor.get("last_turn") or 0) != turn:
        spent["turns"] = int(spent.get("turns") or 0) + 1
        executor["last_turn"] = turn
    _remember_event(subtask, event)

    subtask_id = str(subtask.get("id") or "")
    if _event_closes_subtask(event) and subtask.get("status") != SUBTASK_DONE:
        verdict = evidence_verdict(subtask)
        open_ids = _predecessors_open(index, subtasks) if verdict["ok"] else []
        outcome["evidence_weak"] = bool(verdict["evidence_weak"])
        if verdict["ok"] and not open_ids:
            _transition(
                record, subtask, SUBTASK_DONE, "done_when_met", turn, transitions, event
            )
            outcome["completed"] = subtask_id
        else:
            missing = list(verdict["missing"])
            reason = _claim_reason(missing, open_ids)
            outcome["claimed"] = subtask_id
            outcome["evidence_missing"] = missing
            if open_ids:
                outcome["out_of_order"] = subtask_id
                outcome["predecessors"] = open_ids
            if (
                subtask.get("status") != SUBTASK_CLAIMED_DONE
                or list(subtask.get("_evidence_missing") or []) != missing
                or str(subtask.get("reason") or "") != reason
            ):
                subtask["_evidence_missing"] = missing
                _transition(
                    record,
                    subtask,
                    SUBTASK_CLAIMED_DONE,
                    reason,
                    turn,
                    transitions,
                    event,
                )
    else:
        blocked_reason = _event_blocked_reason(event)
        if blocked_reason and subtask.get("status") in SUBTASK_OPEN:
            _transition(
                record, subtask, SUBTASK_BLOCKED, blocked_reason, turn, transitions
            )
            outcome["blocked"] = subtask_id
            outcome["blocked_reason"] = blocked_reason
        budget_exceeded = _budget_exceeded(subtask.get("budget"), spent)
        if budget_exceeded and subtask.get("status") not in (SUBTASK_DONE, SUBTASK_FAILED):
            verdict = evidence_verdict(subtask)
            open_ids = _predecessors_open(index, subtasks) if verdict["ok"] else []
            outcome["evidence_weak"] = bool(verdict["evidence_weak"])
            if verdict["ok"] and not open_ids:
                # the step's evidence is complete: the budget ran out, not the
                # work - record the completion instead of a failure
                _transition(
                    record,
                    subtask,
                    SUBTASK_DONE,
                    "budget_with_evidence",
                    turn,
                    transitions,
                )
                outcome["completed"] = subtask_id
                outcome["budget_with_evidence"] = subtask_id
            elif verdict["ok"]:
                # the evidence is there but an earlier stage is still open: a
                # claim, not a failure - the completion is promoted in order
                reason = _claim_reason((), open_ids)
                outcome["claimed"] = subtask_id
                outcome["out_of_order"] = subtask_id
                outcome["predecessors"] = open_ids
                if (
                    subtask.get("status") != SUBTASK_CLAIMED_DONE
                    or str(subtask.get("reason") or "") != reason
                ):
                    subtask["_evidence_missing"] = []
                    _transition(
                        record,
                        subtask,
                        SUBTASK_CLAIMED_DONE,
                        reason,
                        turn,
                        transitions,
                        event,
                    )
            else:
                subtask["_evidence_missing"] = list(verdict["missing"])
                _transition(
                    record,
                    subtask,
                    SUBTASK_FAILED,
                    f"budget_exceeded:{','.join(budget_exceeded)}",
                    turn,
                    transitions,
                )
                outcome["failed"] = subtask_id
                outcome["budget_exceeded"].append(subtask_id)

    # P4/A + P4: a claim is re-judged on every call charged to the plan, so
    # evidence that arrived without a completion signal (ep_007) and a later
    # stage whose predecessors have finished (seed 7) both become done.
    promoted = _promote_ready(record, turn, transitions)
    if promoted:
        outcome["promoted"] = promoted
        if subtask_id in promoted:
            outcome["completed"] = subtask_id
            outcome["claimed"] = ""
        outcome["evidence_weak"] = outcome["evidence_weak"] or any(
            bool(item.get("_evidence_weak"))
            for item in subtasks
            if str(item.get("id") or "") in promoted
        )

    if promoted or subtask.get("status") == SUBTASK_DONE:
        # only a *completed* sub-task hands the plan over: a blocked one stays
        # current, so its failure branch is what the model reads next
        _auto_advance(record, turn, transitions)
    after = _current_index(subtasks)
    current = subtasks[after - 1] if after else None
    if after and (after, str(current.get("id") or "")) != before:
        outcome["decision"] = "subtask_switch"

    _sync(record)
    outcome["transitions"] = transitions
    outcome["changed"] = bool(transitions)
    outcome["current"] = after
    outcome["active"] = str((current or {}).get("id") or "")
    branch = _subtask_failure_branch(current)
    if branch:
        outcome["failure_subtask"] = str((current or {}).get("id") or "")
        outcome["failure_branch"] = branch
    record["executor"]["decision"] = outcome["decision"]
    record["updated_t"] = round(time.time(), 3)
    return outcome


def _auto_advance(record: dict, turn: int, transitions: list[dict]) -> int:
    """Activate the next eligible sub-task when the plan has none active.

    This is the deterministic bookkeeping the model no longer has to spend a
    turn on: a completed sub-task hands the plan to the next one whose
    ``depends_on`` are satisfied, inside the very turn that completed it. A plan
    with an active sub-task, with an unresolved dependency, or with nothing left
    to activate is left alone; ``MAX_AUTO_ADVANCE`` bounds the walk.
    """
    advanced = 0
    for _ in range(MAX_AUTO_ADVANCE):
        subtasks = record.get("subtasks") or []
        if any(subtask.get("status") == SUBTASK_ACTIVE for subtask in subtasks):
            # a step that is executing keeps the plan
            break
        if any(
            subtask.get("status") == SUBTASK_CLAIMED_DONE
            and not _predecessors_open(position, subtasks)
            for position, subtask in enumerate(subtasks, 1)
        ):
            # a step that still owes evidence keeps the plan; a claim that is
            # only waiting for an earlier stage does not block that stage
            break
        position = _next_eligible(subtasks)
        if position is None:
            break
        _transition(
            record,
            subtasks[position - 1],
            SUBTASK_ACTIVE,
            "auto_advanced",
            turn,
            transitions,
        )
        advanced += 1
    return advanced


def _budget_exceeded(budget: Any, spent: Mapping) -> list[str]:
    """The budget dimensions this sub-task has spent past (empty when inside)."""
    if not isinstance(budget, Mapping):
        return []
    exceeded: list[str] = []
    for key in ("tool_calls", "turns"):
        limit = _bounded_int(budget.get(key), MAX_SUBTASK_BUDGET)
        if limit and int(spent.get(key) or 0) > limit:
            exceeded.append(key)
    return exceeded


# ---------------------------------------------------------------------------
# progress bookkeeping shared by the tools and the executor
# ---------------------------------------------------------------------------


def _unique_id(candidate: str, used: set[str]) -> str:
    if candidate not in used:
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in used:
        suffix += 1
    return f"{candidate}_{suffix}"


def _build_subtasks(specs: list[dict], record: dict | None) -> list[dict]:
    """The sub-task list a replacement call asks for.

    A step entry inherits the execution state of the sub-task it re-states - by
    ``id`` or by goal text - so re-submitting a plan (or extending it) never
    resets what the executor already advanced; its other fields survive the
    re-statement unless the entry declares them.
    """
    previous = list((record or {}).get("subtasks") or [])
    by_id = {str(item.get("id")): item for item in previous if isinstance(item, Mapping)}
    by_goal = {_goal_key(item.get("goal")): item for item in previous if isinstance(item, Mapping)}
    subtasks: list[dict] = []
    used: set[str] = set()
    for position, spec in enumerate(specs, 1):
        if spec.get("_patch"):
            continue  # a patch entry is applied in place, not as a new step
        inherited = by_id.get(spec.get("id") or "") or by_goal.get(_goal_key(spec.get("goal")))
        subtask = _copy_subtask(inherited) if inherited is not None else _blank_subtask()
        subtask["goal"] = spec["goal"]
        for field in SUBTASK_FIELDS:
            if field in spec and field not in ("id", "goal"):
                subtask[field] = copy.deepcopy(spec[field])
        if "next_step" in spec:
            # a re-stated continuation is a fresh step for the executor
            subtask["_next_step_spent"] = False
        subtask["id"] = _unique_id(spec.get("id") or f"st_{position}", used)
        used.add(subtask["id"])
        subtasks.append(subtask)
    _resolve_dependencies(subtasks)
    return subtasks


def _goal_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _resolve_dependencies(subtasks: list[dict]) -> None:
    """Rewrite ``depends_on`` references into the ids of the plan they name."""
    goals = [str(subtask.get("goal") or "") for subtask in subtasks]
    ids = [str(subtask.get("id") or "") for subtask in subtasks]
    for subtask in subtasks:
        resolved: list[str] = []
        for dep in subtask.get("depends_on") or []:
            text = str(dep).strip()
            if not text:
                continue
            if text in ids:
                resolved.append(text)
                continue
            index = _progress_step_index(
                int(text) if text.isdigit() else text, goals
            )
            resolved.append(ids[index - 1] if index else text)
        subtask["depends_on"] = resolved


def _apply_patches(
    record: dict, specs: list[dict], turn: int = 0, notes: list[str] | None = None
) -> list[str]:
    """Apply ``id``-named patches to the stored sub-tasks.

    A patch that declares a sub-task ``done`` is held to the same evidence rule
    as the executor: when its ``done_when`` is not satisfied by the recorded
    evidence the status becomes ``claimed_done`` (a note says why), so a patch
    cannot route around the evidence binding. Returns the ids of the sub-tasks
    the call really changed, so a patch that restates what the record already
    says does not move the plan revision.
    """
    subtasks = record.get("subtasks") or []
    by_id = {str(item.get("id")): item for item in subtasks if isinstance(item, Mapping)}
    touched: list[str] = []
    for spec in specs:
        subtask = by_id.get(spec.get("id") or "")
        if subtask is None:
            continue
        changed = False
        claimed_missing: list[str] | None = None
        for field in SUBTASK_FIELDS:
            if field not in spec or field == "id":
                continue
            if field == "goal" and not spec.get("_goal"):
                continue  # a patch that did not restate the goal keeps it
            value = copy.deepcopy(spec[field])
            if (
                field == "status"
                and value == SUBTASK_DONE
                and str(subtask.get("done_when") or "").strip()
            ):
                verdict = evidence_verdict(subtask)
                subtask["_evidence_weak"] = bool(verdict["evidence_weak"])
                if not verdict["ok"]:
                    claimed_missing = list(verdict["missing"])
                    value = SUBTASK_CLAIMED_DONE
            if subtask.get(field) != value:
                subtask[field] = value
                if field == "next_step":
                    # a patched continuation is a fresh step for the executor
                    subtask["_next_step_spent"] = False
                changed = True
        if claimed_missing is not None:
            reason = "marked_done:evidence_insufficient:" + ",".join(claimed_missing)
            subtask["_evidence_missing"] = claimed_missing
            if subtask.get("reason") != reason:
                subtask["reason"] = reason
                changed = True
            if notes is not None:
                notes.append(
                    f"{subtask.get('id')} was patched to done, but its done_when is "
                    f"not satisfied ({', '.join(claimed_missing)}): recorded as "
                    f"{SUBTASK_CLAIMED_DONE}"
                )
        if changed:
            subtask["updated_turn"] = int(turn)
            touched.append(str(subtask.get("id")))
    if touched:
        _resolve_dependencies(subtasks)
    return touched


def _unknown_patches(record: dict, specs: list[dict]) -> list[str]:
    known = {
        str(item.get("id")) for item in (record.get("subtasks") or []) if isinstance(item, Mapping)
    }
    return sorted({spec["id"] for spec in specs if spec.get("id") not in known})


def _sync_steps(record: dict) -> None:
    """``steps`` is the goal text of each sub-task, in order (the legacy view)."""
    record["steps"] = [str(subtask.get("goal") or "") for subtask in record.get("subtasks") or []]


def _sync_progress(record: dict) -> None:
    """``done`` / ``current_step`` follow the sub-task statuses."""
    subtasks = record.get("subtasks") or []
    record["done"] = [
        index
        for index, subtask in enumerate(subtasks, 1)
        if subtask.get("status") == SUBTASK_DONE
    ]
    index = _current_index(subtasks)
    record["current_step"] = index if index else (len(subtasks) or 1)


def _sync(record: dict) -> None:
    _sync_steps(record)
    _sync_progress(record)


def _activate_default(record: dict) -> bool:
    """Give a plan with no active sub-task one - the first eligible pending one.

    A fresh plan starts executing its first step; a plan whose active sub-task was
    marked done through ``done=[..]`` hands over to the next pending one. Returns
    whether a status changed.
    """
    subtasks = record.get("subtasks") or []
    if not subtasks or any(
        subtask.get("status") == SUBTASK_ACTIVE for subtask in subtasks
    ):
        return False
    index = _current_index(subtasks)
    if not index:
        return False
    current = subtasks[index - 1]
    if current.get("status") != SUBTASK_PENDING:
        return False
    if all(
        _dependency_done(dep, subtasks, {str(s.get("id")): s for s in subtasks})
        for dep in current.get("depends_on") or []
    ):
        current["status"] = SUBTASK_ACTIVE
        current["reason"] = "plan_started"
        return True
    return False


def plan_complete(record: dict | None) -> bool:
    """True when every step of the plan is marked done."""
    steps = list((record or {}).get("steps") or [])
    done = {int(index) for index in (record or {}).get("done") or []}
    return bool(steps) and all(index in done for index in range(1, len(steps) + 1))


def plan_completion(ctx: ToolContext) -> dict | None:
    """The plan's completion verdict for the ``finish`` contract (None: no plan).

    Every sub-task has to be ``done``; a ``claimed_done``/``blocked``/``failed``
    or still pending stage is reported by id, status and goal, so a refused
    ``finish`` can answer *which stages are still missing* instead of only
    "verified=false". The evidence a claimed stage still owes travels along in
    ``evidence_missing``.
    """
    record = plan_state(ctx)
    if record is None:
        return None
    subtasks = plan_subtasks(ctx) or []
    open_subtasks: list[dict] = []
    missing: list[str] = []
    for position, subtask in enumerate(subtasks, 1):
        status = str(subtask.get("status") or SUBTASK_PENDING)
        if status == SUBTASK_DONE:
            continue
        subtask_id = str(subtask.get("id") or f"st_{position}")
        goal = str(subtask.get("goal") or "")
        item = {
            "index": position,
            "subtask_id": subtask_id,
            "status": status,
            "goal": goal,
        }
        verdict = evidence_verdict(subtask)
        if verdict["missing"]:
            item["evidence_missing"] = list(verdict["missing"])
        open_subtasks.append(item)
        missing.append(f"subtask {subtask_id} ({status}): {goal}")
    return {
        "task_id": str(record.get("plan_id") or ""),
        "revision": int(record.get("revision", 0)),
        "complete": not open_subtasks,
        "missing": missing,
        "open_subtasks": open_subtasks,
        "done_subtasks": [
            str(subtask.get("id"))
            for subtask in subtasks
            if subtask.get("status") == SUBTASK_DONE
        ],
    }


def plan_progress(ctx: ToolContext) -> str:
    """One-line progress of the episode's plan (the runtime's STATE block).

    ``plan_1 rev=2 step 2/4 (1 done): if a candidate appears, compare ... |
    done_when: the candidate is verified`` - the executing step and the
    condition that ends the plan, or ``no plan`` before ``update_plan`` ran.
    """
    record = plan_state(ctx)
    if record is None:
        return "no plan"
    return _progress_text(record)


def _progress_text(record: dict) -> str:
    steps = [str(step) for step in record.get("steps") or []]
    total = len(steps)
    done = [int(index) for index in record.get("done") or []]
    current = int(record.get("current_step") or 0)
    current = current if 1 <= current <= total else (1 if total else 0)
    head = (
        f"{record.get('plan_id')} rev={int(record.get('revision', 0))} "
        f"step {current}/{total} ({len(done)} done)"
    )
    if plan_complete(record):
        head += " COMPLETE"
    text = f"{head}: {steps[current - 1]}" if 1 <= current <= total else head
    done_when = str(record.get("done_when") or "")
    if done_when:
        text += f" | done_when: {done_when}"
    return text[:MAX_PROGRESS_CHARS]


def _subtask_progress_text(record: dict, subtasks: list[dict], note: str = "") -> str:
    """The STATE rendering of the sub-tasks, from the record alone."""
    total = len(subtasks)
    if not total:
        return "no plan"
    index = _current_index(subtasks)
    current = subtasks[index - 1]
    kinds = []
    for status in (SUBTASK_DONE, SUBTASK_CLAIMED_DONE, SUBTASK_BLOCKED, SUBTASK_FAILED):
        count = sum(1 for subtask in subtasks if subtask.get("status") == status)
        if count:
            kinds.append(f"{count} {status}")
    spent = current.get("spent") or {}
    budget = current.get("budget") or {}
    limit = _bounded_int(budget.get("tool_calls"), MAX_SUBTASK_BUDGET)
    head = (
        f"{record.get('plan_id')} rev={int(record.get('revision', 0))} "
        f"st {index}/{total}"
    )
    if note:
        head += f" {str(note).strip()}"
    if plan_complete(record):
        head += " COMPLETE"
    spend_text = f"spent={int(spent.get('tool_calls') or 0)}"
    if limit:
        spend_text += f"/{limit}"
    head += (
        f" current={current.get('id')}[{current.get('status')}] "
        f"'{current.get('goal')}' {spend_text} calls"
    )
    if kinds:
        head += f" ({', '.join(kinds)})"
    if current.get("status") in (
        SUBTASK_BLOCKED,
        SUBTASK_FAILED,
        SUBTASK_CLAIMED_DONE,
    ) and current.get("reason"):
        head += f" reason={current.get('reason')}"
    branch = _subtask_failure_branch(current)
    if branch:
        head += " failure_branch=" + " | ".join(branch)
    return head[:MAX_PROGRESS_CHARS]


def subtask_progress(ctx: ToolContext, note: str = "") -> str:
    """One-line sub-task state for the runtime's STATE block (P4).

    ``plan_1 rev=3 st 2/4 current=st_2[active] 'check the cabinet' spent=1/4
    calls (1 done) failure_branch=go_to_viewpoint vp_9`` - which sub-task is
    being executed, how far its budget is consumed, and whether the plan's own
    failure branch is available, or ``no plan``. ``note`` (the runtime's
    ``revision_required=<node>``) is rendered right after the position, where the
    line's own truncation cannot hide it.
    """
    record = plan_state(ctx)
    if record is None:
        return "no plan"
    return _subtask_progress_text(record, plan_subtasks(ctx) or [], note)


def plan_text(ctx: ToolContext) -> str:
    """Compact plan rendering for a prompt/structured-state block."""
    record = plan_state(ctx)
    if record is None:
        return "no plan"
    subtasks = plan_subtasks(ctx) or []
    lines = [
        f"{record.get('plan_id')} rev={int(record.get('revision', 0))} "
        f"({len(subtasks)} steps)"
    ]
    for position, subtask in enumerate(subtasks, 1):
        line = f"  {position}. [{subtask.get('status')}] {subtask.get('goal')}"
        if subtask.get("done_when"):
            line += f" (done when: {subtask['done_when']})"
        if subtask.get("budget", {}).get("tool_calls"):
            spent = (subtask.get("spent") or {}).get("tool_calls", 0)
            line += f" [budget {spent}/{subtask['budget']['tool_calls']}]"
        lines.append(line)
    lines.append(f"  progress: {_progress_text(record)}")
    return "\n".join(lines)


def _subtask_data(subtask: Mapping) -> dict:
    """One sub-task as the model reads it, plus the evidence verdict.

    ``evidence_ok`` is the executor's judgment (``evidence_verdict``): a ``done``
    sub-task satisfies it by contract, a ``claimed_done`` one does not and carries
    the ``evidence_missing`` requirements a later result has to supply.
    """
    data = {field: copy.deepcopy(subtask.get(field)) for field in SUBTASK_FIELDS}
    status = subtask.get("status")
    if status in (SUBTASK_DONE, SUBTASK_CLAIMED_DONE):
        data["evidence_ok"] = status == SUBTASK_DONE
        data["evidence_weak"] = bool(subtask.get("_evidence_weak"))
        if status == SUBTASK_CLAIMED_DONE:
            data["claimed_done"] = True
            verdict = evidence_verdict(subtask)
            data["evidence_missing"] = list(
                verdict["missing"] or subtask.get("_evidence_missing") or []
            )
    return data


def _plan_data(record: dict) -> dict:
    subtasks = record.get("subtasks") or []
    steps = [str(step) for step in record.get("steps") or []]
    current = int(record.get("current_step") or 0)
    done = [int(index) for index in record.get("done") or []]
    index = _current_index(subtasks)
    active = subtasks[index - 1] if index and subtasks else None
    data: dict[str, Any] = {
        "plan_id": record.get("plan_id"),
        "revision": int(record.get("revision", 0)),
        "steps": steps,
        "subtasks": [_subtask_data(subtask) for subtask in subtasks],
        "n_steps": len(steps),
        "current_step": current,
        "current_step_text": steps[current - 1] if 1 <= current <= len(steps) else "",
        "current_subtask": _subtask_data(active) if active else None,
        "done_subtasks": [
            str(subtask.get("id"))
            for subtask in subtasks
            if subtask.get("status") == SUBTASK_DONE
        ],
        "claimed_subtasks": [
            {
                "id": str(subtask.get("id")),
                "goal": str(subtask.get("goal") or ""),
                "reason": str(subtask.get("reason") or ""),
                "evidence_missing": list(
                    evidence_verdict(subtask)["missing"]
                    or subtask.get("_evidence_missing")
                    or []
                ),
            }
            for subtask in subtasks
            if subtask.get("status") == SUBTASK_CLAIMED_DONE
        ],
        "blocked_subtasks": [
            {
                "id": str(subtask.get("id")),
                "goal": str(subtask.get("goal") or ""),
                "reason": str(subtask.get("reason") or ""),
                "failure_branch": list(subtask.get("failure_branch") or []),
            }
            for subtask in subtasks
            if subtask.get("status") == SUBTASK_BLOCKED
        ],
        "failed_subtasks": [
            {
                "id": str(subtask.get("id")),
                "goal": str(subtask.get("goal") or ""),
                "reason": str(subtask.get("reason") or ""),
                "failure_branch": list(subtask.get("failure_branch") or []),
            }
            for subtask in subtasks
            if subtask.get("status") == SUBTASK_FAILED
        ],
        "budget_used": {
            "tool_calls": sum(
                int((subtask.get("spent") or {}).get("tool_calls") or 0)
                for subtask in subtasks
            ),
            "turns": sum(
                int((subtask.get("spent") or {}).get("turns") or 0)
                for subtask in subtasks
            ),
        },
        "done": done,
        "done_when": str(record.get("done_when") or ""),
        "complete": plan_complete(record),
        "progress": _progress_text(record),
        "subtask_progress": _subtask_progress_text(record, subtasks),
        "updated_t": record.get("updated_t", 0.0),
    }
    if record.get("note"):
        data["note"] = record["note"]
    if record.get("resumed_from"):
        data["resumed_from"] = record["resumed_from"]
        data["resume_count"] = int(record.get("resume_count", 0))
    return data


def _plan_evidence(record: dict) -> list[EvidenceRef]:
    return [
        EvidenceRef(
            "memory",
            str(record.get("plan_id")),
            note=f"plan revision {int(record.get('revision', 0))}",
        )
    ]


def _new_plan(ctx: ToolContext, extra: dict) -> dict:
    """Start a fresh plan record for this episode (``plan_N`` ids never repeat)."""
    seq = int(extra.get("_plan_seq", 0)) + 1
    extra["_plan_seq"] = seq
    record = {
        "plan_id": f"plan_{seq}",
        "revision": 0,
        "steps": [],
        "subtasks": [],
        "current_step": 1,
        "done": [],
        "done_when": "",
        "note": "",
        "instruction": _instruction_of(ctx),
        "episode_id": ctx.episode_id,
        "updated_t": 0.0,
        "executor": {"last_turn": 0, "transitions": [], "decision": ""},
        "resumed_from": "",
        "resume_count": 0,
    }
    extra[PLAN_KEY] = record
    return record


def _instruction_of(ctx: ToolContext) -> str:
    """The instruction the plan was written for (the resume guard)."""
    value = _extra(ctx).get("instruction")
    return str(value)[:MAX_INSTRUCTION_CHARS] if value else ""


def resume_plan(ctx: ToolContext, episode_id: str, instruction: str | None = None) -> dict | None:
    """Adopt the plan an interrupted episode left behind (P4).

    The plan of the previous run is only adopted when it was written for the same
    instruction (or the record carries none), and only while it still has
    something open - a plan whose every sub-task is done belongs to the episode
    that finished it, and adopting it would tell the new run the task is over.
    Everything the executor recorded survives: the sub-task statuses, their
    evidence and their consumed budget - that is what makes the second run of an
    interrupted episode continue instead of starting over. Returns the adopted
    record, or None when there is nothing to adopt.
    """
    record = _extra(ctx).get(PLAN_KEY)
    if not isinstance(record, dict):
        return None
    if record.get("episode_id") == episode_id:
        return record
    subtasks = [
        subtask for subtask in record.get("subtasks") or [] if isinstance(subtask, Mapping)
    ]
    if not subtasks:
        return None
    if all(subtask.get("status") == SUBTASK_DONE for subtask in subtasks):
        return None
    stored = str(record.get("instruction") or "")
    if instruction and stored and _goal_key(stored) != _goal_key(
        str(instruction)[:MAX_INSTRUCTION_CHARS]
    ):
        return None
    record["resumed_from"] = str(record.get("episode_id") or "")
    record["resume_count"] = int(record.get("resume_count", 0)) + 1
    record["episode_id"] = episode_id
    # the checkpoint's statuses are adopted verbatim; the first evidence turn
    # of the resumed leg re-judges them (a wrong done must be overturnable)
    record[RESUME_RECHECK_KEY] = {
        "pending": True,
        "from": str(record.get("resumed_from") or ""),
        "turn": None,
        "checked": [],
        "corrected": [],
        "details": [],
    }
    record["executor"] = {
        "last_turn": 0,
        "transitions": list((record.get("executor") or {}).get("transitions") or []),
        "decision": "",
    }
    record["updated_t"] = round(time.time(), 3)
    return record


_STEPS_HINT = (
    'steps: ["go_to_viewpoint vp_3 and scan the left side", '
    '"if a candidate appears, ground_target then approach_candidate it", '
    '"if no target, try vp_4", '
    '"if the path is blocked, propose_viewpoints and re-observe"]'
)

_PROGRESS_HINT = (
    'progress: {"done": [1], "current_step": 2, '
    '"done_when": "the candidate is verified"}'
)

_SUBTASK_HINT = (
    'steps: [{"goal": "find the corridor on the left", "done_when": "the corridor '
    'is visible within 2 m", "goal_position": [1.2, 3.4, 0.0], "locate_m": 2.0, '
    '"next_step": {"tool": "observe", "args": {"mode": "forward"}}, '
    '"failure_branch": ["propose_viewpoints then go_to_viewpoint vp_9"], '
    '"budget": {"tool_calls": 4}}, {"goal": "check the cabinet beside the door", '
    '"depends_on": ["st_1"], "expected_observation": "a cabinet next to a door"}]'
)

_PATCH_HINT = 'patch: [{"id": "st_2", "goal": "check the cabinet left of the door"}]'


def _apply_progress(record: dict, progress: dict, explicit: dict, notes: list[str]) -> bool:
    """Move the statuses the explicit progress arguments name; returns changed.

    A declared ``done`` is held to the same evidence rule as the executor when
    the step has a ``done_when``: without the evidence the status becomes
    ``claimed_done`` (a note says what is missing), while the declaration still
    moves the plan on to the next step. A bare step (no ``done_when``) has no
    condition to check and keeps the legacy ``marked_done`` semantics.
    """
    subtasks = record.get("subtasks") or []
    before = [
        (
            subtask.get("status"),
            json.dumps(subtask.get("spent"), sort_keys=True),
            list(subtask.get("evidence") or []),
        )
        for subtask in subtasks
    ]
    if "done" in explicit:
        unproven: list[tuple[dict, list[str]]] = []
        for index in progress.get("done") or []:
            if not 1 <= index <= len(subtasks):
                continue
            subtask = subtasks[index - 1]
            if not str(subtask.get("done_when") or "").strip():
                # a bare step has no condition to check; the model's declaration
                # is the legacy plan's whole completion contract
                subtask["status"] = SUBTASK_DONE
                subtask["reason"] = "marked_done"
                subtask["_evidence_missing"] = []
                subtask["_evidence_weak"] = False
                continue
            verdict = evidence_verdict(subtask)
            subtask["_evidence_weak"] = bool(verdict["evidence_weak"])
            if verdict["ok"]:
                subtask["status"] = SUBTASK_DONE
                subtask["reason"] = "marked_done"
                subtask["_evidence_missing"] = []
            else:
                # the declaration still moves the plan forward (the model said
                # this step is finished); the status is demoted after the
                # hand-over, so the missing evidence stays visible
                subtask["status"] = SUBTASK_DONE
                subtask["reason"] = "marked_done"
                unproven.append((subtask, list(verdict["missing"])))
        if unproven:
            _activate_default(record)
            for subtask, missing in unproven:
                subtask["_evidence_missing"] = missing
                subtask["status"] = SUBTASK_CLAIMED_DONE
                subtask["reason"] = "marked_done:evidence_insufficient:" + ",".join(missing)
                notes.append(
                    f"{subtask.get('id')} was declared done, but its done_when is "
                    f"not satisfied ({', '.join(missing)}): recorded as "
                    f"{SUBTASK_CLAIMED_DONE}"
                )
    if "current_step" in explicit:
        index = int(progress.get("current_step") or 0)
        if 1 <= index <= len(subtasks):
            if subtasks[index - 1].get("status") == SUBTASK_DONE:
                notes.append(
                    f"step {index} is done but named as current_step: it is active again"
                )
            subtasks[index - 1]["status"] = SUBTASK_ACTIVE
            subtasks[index - 1]["reason"] = "marked_current"
    _activate_default(record)
    after = [
        (
            subtask.get("status"),
            json.dumps(subtask.get("spent"), sort_keys=True),
            list(subtask.get("evidence") or []),
        )
        for subtask in subtasks
    ]
    return before != after


def update_plan(ctx: ToolContext, args: dict) -> ToolResult:
    """Replace the episode's short conditional plan and its progress.

    ``steps`` is optional when the plan already exists and the call only
    advances its progress (``done``/``current_step``/``done_when``): the stored
    steps are kept, the revision moves only when something really changed. With
    ``patch`` the call updates the named sub-tasks in place instead of replacing
    the plan - the model's way to revise one sub-plan after a decision node.
    """
    args = args or {}
    extra = _extra(ctx)
    record = plan_state(ctx)
    patch_specs, patch_notes, _patch_error = _normalize_patch_specs(args, record)
    specs, notes, error = _normalize_step_specs(args, record)
    patch_only = patch_specs is not None and specs is None

    if patch_specs is not None:
        if record is None or not record.get("subtasks"):
            return ToolResult(
                STATUS_FAILED,
                {
                    "error": "no_plan",
                    "patch_notes": patch_notes,
                    "hint": f"write the plan first: update_plan with {_SUBTASK_HINT}",
                },
            )
        unknown = _unknown_patches(record, patch_specs)
        if unknown:
            known = [
                str(subtask.get("id")) for subtask in record.get("subtasks") or []
            ]
            return ToolResult(
                STATUS_FAILED,
                {
                    "error": "unknown_subtask",
                    "unknown_subtasks": unknown,
                    "hint": f"known sub-task ids: {', '.join(known)}; patch one of them",
                    "patch_notes": patch_notes,
                },
            )
    if patch_only:
        error = None  # the patch is the call; there is no step list to require
    elif error == "missing_steps" and record is not None and _has_progress_args(args):
        specs, error = None, None  # a progress-only call keeps the stored steps

    if error is not None:
        return ToolResult(
            STATUS_FAILED,
            {
                "error": error,
                "hint": f"pass one short step per list entry, e.g. {_STEPS_HINT}",
            },
        )
    if record is None:
        record = _new_plan(ctx, extra)

    # a step entry that only names an existing sub-task updates it in place; a
    # list of them is a patch, a list with real steps replaces the plan
    patch_steps = [spec for spec in (specs or []) if spec.get("_patch")]
    new_steps = [spec for spec in (specs or []) if not spec.get("_patch")]
    changed = False
    if new_steps:
        subtasks = _build_subtasks(new_steps, record)
        changed = subtasks != record.get("subtasks")
        record["subtasks"] = subtasks
        _sync(record)
    touched = _apply_patches(record, patch_steps + list(patch_specs or ()), notes=notes)
    if touched:
        changed = True

    progress, progress_notes = normalize_progress(
        args, [str(step) for step in record.get("steps") or []]
    )
    notes = list(notes) + list(patch_notes) + list(progress_notes)
    explicit = explicit_progress(args)
    if explicit:
        changed = _apply_progress(record, progress, explicit, notes) or changed
    if _activate_default(record):
        changed = True
    if progress["done_when"] and progress["done_when"] != record.get("done_when"):
        record["done_when"] = progress["done_when"]
        changed = True
    _sync(record)
    if changed:
        record["revision"] = int(record.get("revision", 0)) + 1
    if not record.get("instruction"):
        record["instruction"] = _instruction_of(ctx)
    note = args.get("note", args.get("reason"))
    if note not in (None, ""):
        record["note"] = str(note)[:MAX_NOTE_CHARS]
    record["updated_t"] = round(time.time(), 3)
    data = _plan_data(record)
    data["changed"] = changed
    if notes:
        data["progress_notes"] = notes
        data["progress_hint"] = _PROGRESS_HINT
    if not record.get("subtasks"):
        data["subtask_hint"] = _SUBTASK_HINT
    return ToolResult(STATUS_OK, data, _plan_evidence(record))


def get_plan(ctx: ToolContext, args: dict) -> ToolResult:
    """Read the current conditional plan and its execution progress (goal.md 5)."""
    record = plan_state(ctx)
    if record is None:
        return ToolResult(
            STATUS_EMPTY,
            {
                "reason": "no_plan",
                "hint": (
                    "call update_plan with a short conditional plan first, e.g. "
                    f"update_plan with {_SUBTASK_HINT}"
                ),
            },
        )
    return ToolResult(STATUS_OK, _plan_data(record), _plan_evidence(record))


def register_task_tools(registry: ToolRegistry) -> None:
    """Register ``update_plan`` / ``get_plan`` on ``registry``."""
    registry.register(
        ToolSpec(
            name="update_plan",
            description=(
                "Replace or revise your short conditional plan for this task: one "
                "step per list entry. A step is a sub-task - {id, goal, "
                "preconditions, depends_on, expected_observation, done_when, "
                "goal_position, locate_m, failure_branch, next_step, "
                "budget{tool_calls,turns}} - so the runtime's "
                "executor can track it: it marks the step done when a tool result "
                "satisfies its done_when, blocks it when an execution fails (and "
                "then offers its failure_branch as the next step) and fails it when "
                "its budget runs out. A later step stays claimed until every earlier "
                "step and every depends_on is done. A located result only counts "
                "inside locate_m of goal_position (the distance in done_when is the "
                "fallback tolerance; without a declared place the verdict is marked "
                "evidence_weak). Write next_step and every failure_branch entry as "
                "ONE concrete call (e.g. {\"tool\": \"observe\", \"args\": {}}), so "
                "the executor can run it without a model turn; prose is not executed. "
                "A plain text step is wrapped automatically. "
                "Revise one sub-plan in place with patch=[{id, goal/status/...}] "
                "instead of re-stating the whole plan; re-stating a step keeps its "
                "status and evidence. Progress is part of the same call: done=[1] "
                "marks step 1 finished, current_step names the step you are "
                "executing, done_when states the condition that ends the whole plan "
                "(the runtime carries them into its state block every turn). "
                f"Returns plan_id, revision, the steps, their sub-task records and "
                f"the rendered subtask progress, e.g. {_SUBTASK_HINT} or "
                f"{_PATCH_HINT}; re-read the plan with get_plan."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "steps": {
                        "type": ["array", "string"],
                        "items": {
                            "type": ["string", "object"],
                        },
                        "description": (
                            "ordered conditional steps, each one short line or a "
                            "sub-task object ({id, goal, preconditions, depends_on, "
                            "expected_observation, done_when, goal_position, "
                            "locate_m, failure_branch, next_step, budget, status}); "
                            "next_step/failure_branch entries should be one concrete "
                            "call with JSON args; a single string is split on ';'. "
                            "Required unless the call only advances the progress "
                            "(done/current_step/done_when) or patches a sub-task"
                        ),
                    },
                    "patch": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": (
                            "sub-tasks to update in place, each naming an existing "
                            "id plus the fields to change (goal, status, "
                            "failure_branch, budget, ...); the other steps are kept"
                        ),
                    },
                    "current_step": {
                        "type": ["integer", "string"],
                        "description": (
                            "the step being executed: its 1-based index or its text "
                            "(default: the first step that is not done)"
                        ),
                    },
                    "done": {
                        "type": "array",
                        "items": {"type": ["integer", "string"]},
                        "description": "steps already finished (1-based indices or step texts)",
                    },
                    "done_when": {
                        "type": "string",
                        "description": (
                            "the condition that ends the whole plan, e.g. 'the target "
                            "is verified and approached'"
                        ),
                    },
                    "note": {
                        "type": "string",
                        "description": "why the plan changed (optional, kept with the plan)",
                    },
                },
            },
            handler=update_plan,
            # rewriting the plan is an episode-state mutation: serialized, and
            # excluded from the repeated-call cache by the side_effects rule
            side_effects=True,
        )
    )
    registry.register(
        ToolSpec(
            name="get_plan",
            description=(
                "Read the current conditional plan (plan_id, revision, steps, "
                "subtasks) and its execution progress: current_subtask, "
                "done_subtasks, blocked_subtasks with their reason, failed_subtasks "
                "and budget_used (current_step, done, done_when, complete, progress "
                "line as well). Use it to re-check the plan after several turns. "
                "Returns empty with reason=no_plan until update_plan has been called."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=get_plan,
            side_effects=False,
            # a repeated get_plan must answer with the current revision, never
            # with a replayed one
            cacheable=False,
        )
    )


__all__ = [
    "PLAN_KEY",
    "MAX_PLAN_STEPS",
    "MAX_STEP_CHARS",
    "MAX_PROGRESS_CHARS",
    "MAX_LIST_ITEMS",
    "MAX_LIST_CHARS",
    "MAX_SUBTASK_BUDGET",
    "MAX_SUBTASK_EVIDENCE",
    "MAX_SUBTASK_PLACES",
    "MAX_TRANSITIONS",
    "SUBTASK_PENDING",
    "SUBTASK_ACTIVE",
    "SUBTASK_CLAIMED_DONE",
    "SUBTASK_DONE",
    "SUBTASK_FAILED",
    "SUBTASK_BLOCKED",
    "SUBTASK_STATUSES",
    "SUBTASK_OPEN",
    "SUBTASK_FIELDS",
    "EVIDENCE_MARKS",
    "evidence_requirements",
    "evidence_verdict",
    "parse_call_text",
    "DEFAULT_SUBTASK_BUDGET",
    "DEFAULT_LOCATE_M",
    "EXECUTOR_SKIP_TOOLS",
    "FAILURE_OUTCOMES",
    "TASK_SIGNAL_KEYS",
    "TASK_VERDICTS",
    "normalize_steps",
    "normalize_progress",
    "normalize_subtasks",
    "explicit_progress",
    "plan_complete",
    "plan_completion",
    "subtask_ground_instruction",
    "plan_state",
    "plan_subtasks",
    "plan_text",
    "plan_progress",
    "subtask_progress",
    "subtask_failure_branch",
    "current_subtask",
    "executor_step",
    "resume_plan",
    "update_plan",
    "get_plan",
    "register_task_tools",
    "register_tools",
    "register_all",
]

# Generic-discovery aliases: ``scripts/run_vlm_nav_tools.py`` and
# ``vlm_nav/mcp_server.py`` probe tool modules for these names before installing
# their own fallback/contract stubs, so a module picked up by name still
# contributes its tools. Keep ``register_task_tools`` as the documented entry
# point.
register_tools = register_task_tools
register_all = register_task_tools
