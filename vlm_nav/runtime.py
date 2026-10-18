"""Agent runtime: the multi-turn tool-calling decision loop (goal.md 2/5).

One episode = one instruction. Each turn the runtime

1. renders the current observation (``obs_N``, published for tools),
2. assembles a *budgeted* context: current frame, a few evidence/history
   frames, structured odometry state, the fixed ``STATE`` block (subgoal,
   candidate+evidence, open questions, searched coverage, recent failures,
   next information needed, budget left, plan progress), task/hypothesis
   summaries, pending events, the last tool results, and the tool catalogue
   (``registry.specs_text()``, rendered beside ``TOOL_CALL_DOC``),
3. asks the model for exactly ONE tool call, executes it through the shared
   ``ToolRegistry`` and feeds the structured ``ToolResult.to_prompt()`` back,
4. scores the turn: ``information`` (a new view / query answer), ``action`` (the
   drone moved, a route segment was flown), ``task`` (identity confirmed,
   subgoal completed, termination condition met) or ``none`` (a replay, a
   failure, or the same view reviewed again). A turn that made no progress
   counts towards ``max_stagnant_turns``; the model cannot bank progress by
   observing the same place again under a new ``obs_N``: the repeat test is
   (pose + heading + view content hash + query intent), not the id.

The loop ends on the model's ``finish`` call, an exhausted tool budget,
``max_turns``, an unparseable reply after one re-prompt, or an external abort.
A stagnant episode *recovers first*: while the registry still offers a tool
that could produce evidence, the runtime injects a concrete recovery
instruction (``recovery_advice``: change the viewpoint, magnify a region,
re-propose viewpoints with ``expected_info``, re-compare candidates, change the
approach direction) and, if a ``recovery_hook`` is configured, lets it execute
one automatic recovery call. Only when ``max_recovery_attempts`` are used up
does the runtime ask the model for an honest unfinished ``finish`` and then stop
with ``no_new_evidence`` - never by pretending the target does not exist.
``recovery_enabled=False`` (the ablation switch) keeps the pre-recovery
semantics: the stagnation limit ends the episode directly.

P4 - long-horizon execution and plan scheduling. After every executed call the
runtime hands the result to the *sub-task executor* of ``tool_impl_task``
(``executor_step``): the sub-task the call served is charged one tool call, a
result that satisfies its ``done_when`` - and every earlier/``depends_on``
stage - marks it ``done`` and activates the next eligible one, a claim that is
still missing evidence or order stays ``claimed_done`` until a later call
re-judges it, a failed leg marks it ``blocked`` (its ``failure_branch`` becomes
the recovery advice), and a sub-task whose budget runs out is ``failed``. Plan
bookkeeping therefore costs the model *no* decision turn: the whole chain of
deterministic advances happens inside one turn and shows up in the STATE block's
``subtasks`` label. Every turn is then classified - routine (absorbed by the
executor) or a *decision node* (a new candidate, a lost goal, conflicting
evidence, a failed path, a sub-task switch, :data:`DECISION_NODES`) - and a node
that invalidates the plan turns into a "PLAN REVISION NEEDED" section in the next
prompt, asking the model to patch the sub-plan (``update_plan`` with ``patch``).
At the start of an episode a caller-provided context may hand over the plan an
interrupted episode left behind (``resume_plan``): its statuses, evidence and
consumed budget are adopted, so a re-run continues instead of starting over.
The revision request - and the STATE section's ``PLAN CALLS`` hint once a plan
exists - asks for ``next_step`` and every ``failure_branch`` entry as one
concrete call with JSON args, because a prose plan gives the executor no
deterministic step to run.
A *routine* turn whose plan names a deterministic next step does not spend a
model turn either: the runtime executes that step through the registry
(``next_action`` of the last result, an explicit plan/sub-task next step, a
concrete failure branch after a fruitless result, or a sub-task goal written as
a tool call), capped at ``MAX_EXECUTOR_AUTO_STEPS`` consecutive steps and
recorded in ``EpisodeLog.executor_auto_steps``. A result that came back empty,
failed or blocked may not name itself as that next step: the same-call
continuation is only followed after a productive turn (the reported
self-repeating ``search_target`` loop).
``subtask_executor=False``, ``plan_resume=False`` and
``executor_auto_steps=False`` are the P4 ablation switches.

Model side (both paths share the same tools, context and ``EpisodeLog``):
- ``vlm.decide_with_tools(frames, system_prompt, user_prompt, tools_text,
  context)`` when the backend exposes it (vlm.py's contract). The runtime passes
  the rendered tool catalogue positionally and the structured specs under
  ``context["tools"]``, so both the constrained-JSON path and native tool
  calling work; the reply may be a ``{"tool": ..., "args": {...}}`` dict, an
  OpenAI-style ``{"tool_calls": [...]}`` dict, or a text reply.
- otherwise ``vlm.decide(frames, system, user, context)`` plus
  ``tools.parse_tool_call``; the runtime then appends ``TOOL_CALL_DOC`` and the
  tool catalogue to the system prompt itself.
  Signature variants are bound via ``inspect`` so mock backends written to
  either convention (4-arg ``..., context``) keep working.

Context handed to tool-aware backends (never ground truth): ``episode_id``,
``instruction``, ``step``/``turn``, ``max_turns``, ``tool_calls_left``,
``tool_names``, ``position_flu``, ``yaw_rad``, ``state`` (the STATE block as a
dict), ``tools`` (structured specs), ``history`` (one ``{"tool_call": ...,
"result": <ToolResult text>}`` entry per executed call) and ``on_log`` (backend
log events are collected in ``EpisodeLog.vlm_log``).

Tool side: tools get the episode ``ToolContext`` (a caller-provided
``tool_context`` is adopted, keeping its ``extra`` knobs/budget/log). Two
optional channels back to the runtime are read after every call:
- ``ctx.extra["images"][observation_id] = frame`` publishes a rendered frame,
  which the runtime caches and offers as a context image on the next turn;
- ``ctx.extra["events"]`` (list) queues events that the runtime re-emits.
A tool may also declare what its own answer means: ``data["progress_kind"]``
(one of ``tools.PROGRESS_KINDS``) overrides the runtime's classification, and a
``data["coverage"]`` mapping (``checked_deg`` / ``unchecked_deg`` /
``worth_looking`` - the search skill's contract) is carried into the STATE
block's ``searched_coverage`` and into the recovery advice.

Observations are one repository (``vlm_nav.observations``): the runtime creates
the episode's ``ObservationStore`` in ``ctx.extra["observation_store"]`` and
registers **every frame the model is shown** - the frame it renders each turn and
each frame a tool publishes under an ``obs_N`` id - with the pose, heading and
render intrinsics that were in force when that frame was rendered. The id the
model reads in the prompt (``image 1 = current view (obs_7)``,
``current_observation=obs_7``) is therefore the id every tool resolves, which is
exactly what a real trajectory broke: the model cited the frame it had just been
shown and ``inspect_region``/``ground_target`` answered "unknown observation"
because the id had never entered a store they read.

Memory is duck-typed (``context_summary`` / ``hypotheses`` / ``active_tasks`` /
``summarize`` / ... if present) and is never written by the runtime - tools own
memory writes. Ground-truth target position/distance is never placed into the
prompt: the state block is odometry only.

Event hooks: ``on_event(name, data)`` hooks observe every event; events drained
from ``ctx.extra["events"]``, from ``emit_event`` calls (any thread) and from the
pending queue wake the loop at the next turn boundary. A hook may call
``runtime.request_abort()`` to end the episode cleanly. Every recovery attempt
is emitted as a ``recovery`` event and kept in ``EpisodeLog.recoveries``.

Context budget knobs: ``max_images``, ``max_system_chars``, ``max_tools_chars``,
``max_text_chars`` and ``max_tool_chars`` (per tool result fed back).
The system prompt is budgeted catalogue-first: the rendered state-aware
catalogue is reserved before the policy template and the protocol/examples, so
the default budget (``DEFAULT_SYSTEM_CHARS``) delivers the listing whole; a doc
that does not fit is truncated and says so.
``feedback_in_prompt=False`` suppresses the runtime's own result block for
backends that already render ``context["history"]`` into the prompt.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import threading
import time
import uuid
from typing import Any, Callable, Iterable

import numpy as np

from .observations import (
    CANDIDATE_STORE_KEY,
    KIND_EVIDENCE,
    OBSERVATION_STATE_KEY,
    OBSERVATION_STORE_KEY,
    ObservationStore,
    kind_for_label,
    lookup_pose,
    observation_store,
    published_frames,
    register_observation,
)
from .tools import (
    DEFAULT_CATALOG_CHARS,
    PROGRESS_ACTION,
    PROGRESS_INFORMATION,
    PROGRESS_NONE,
    PROGRESS_TASK,
    PROMPT_EXAMPLES,
    STATUS_BLOCKED,
    STATUS_CACHED,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_OK,
    TOOL_CALL_DOC,
    ToolContext,
    ToolResult,
    declared_progress_kind,
    parse_tool_call,
)

FINISH_TOOL = "finish"
NO_TOOL = "(no tool call)"

#: Default system-prompt budget. The catalogue is reserved first
#: (``DEFAULT_CATALOG_CHARS``, the registry's own render limit); the protocol
#: doc / finish contract (~4.3k) and the policy template (~1k) fit in the rest.
#: A prompt that does not fit truncates the doc - never the catalogue.
DEFAULT_SYSTEM_CHARS = DEFAULT_CATALOG_CHARS + 7000
#: How many deterministic steps the executor may run in a row after one model
#: turn. The cap is what keeps an auto-step chain from running away when a plan
#: or a tool keeps naming the same continuation.
MAX_EXECUTOR_AUTO_STEPS = 3
#: ``TurnRecord.raw_reply`` of a call the executor issued on its own (there is
#: no model reply to quote).
AUTO_STEP_REPLY = "[executor]"
#: Tools the executor never issues on its own: closing the episode and plan
#: bookkeeping are the model's decisions, not a plan step.
AUTO_STEP_EXCLUDED_TOOLS = ("finish", "update_plan", "get_plan", "task_status", "cancel_task")
#: Prompt text carried with the STATE block once a plan exists: the executor's
#: deterministic stepping has no source unless the plan itself spells one out.
PLAN_CALL_HINT = (
    "PLAN CALLS: write each sub-task's next_step and every failure_branch entry "
    'as ONE concrete call, e.g. {"tool": "go_to_viewpoint", "args": '
    '{"viewpoint_id": "vp_9"}}; the executor runs those without a model turn '
    "(prose is not executed)."
)

REASON_FINISH = "finish"
REASON_BUDGET_EXHAUSTED = "budget_exhausted"
REASON_MAX_TURNS = "max_turns"
REASON_NO_NEW_EVIDENCE = "no_new_evidence"
REASON_UNPARSEABLE = "unparseable_output"
REASON_ABORTED = "aborted"

#: Labels of the per-turn STATE block, in the order they are always rendered.
#: A field the episode does not know yet says ``n/a`` - the block never drops a
#: label, so the model can rely on where each fact is.
STATE_LABELS = (
    "subgoal",
    "candidate",
    "evidence",
    "open_questions",
    "searched_coverage",
    "recent_failures",
    "next_info_needed",
    "budget_left",
    "plan_progress",
    "subtasks",
)

#: The decision nodes of a long-horizon plan (P4). A *routine* turn - a new
#: view, a query answer, an executor bookkeeping step - is absorbed by the
#: sub-task executor and never spends a decision turn; these are the turns at
#: which the model has to look at the plan again. Every turn is classified and
#: the node is recorded on its ``TurnRecord`` and in ``EpisodeLog.decision_nodes``.
DECISION_NEW_CANDIDATE = "new_candidate"
DECISION_GOAL_LOST = "goal_lost"
DECISION_EVIDENCE_CONFLICT = "evidence_conflict"
DECISION_PATH_FAILURE = "path_failure"
DECISION_SUBTASK_SWITCH = "subtask_switch"
DECISION_NODES = (
    DECISION_NEW_CANDIDATE,
    DECISION_GOAL_LOST,
    DECISION_EVIDENCE_CONFLICT,
    DECISION_PATH_FAILURE,
    DECISION_SUBTASK_SWITCH,
)
#: The nodes whose answer is a plan revision: the goal was lost, the evidence
#: contradicts itself, or the route is blocked - the plan's own alternative (its
#: failure branch) or a new sub-plan is what the model has to decide.
PLAN_REVISION_NODES = (
    DECISION_GOAL_LOST,
    DECISION_EVIDENCE_CONFLICT,
    DECISION_PATH_FAILURE,
)
#: ``reason``/``outcome``/``status`` tokens that mean "the target is gone".
GOAL_LOST_TOKENS = (
    "target_lost",
    "lost_target",
    "candidate_lost",
    "lost_candidate",
    "target_missing",
    "track_lost",
    "lost_sight",
)
#: ``reason``/``verdict`` tokens that mean "the candidates disagree".
CONFLICT_TOKENS = ("different", "uncertain", "conflict", "mismatch")

#: ``log.recoveries[*]["kind"]`` - what the runtime did about a stall.
RECOVERY_ADVICE = "advice"  # a recovery instruction was injected into the prompt
RECOVERY_HOOK = "hook"  # the recovery_hook executed a call automatically
RECOVERY_FINAL = "final_finish"  # one turn granted to finish honestly
RECOVERY_STOP = "stop"  # no recovery left: the loop ends

#: How far the drone must have moved for a turn to count as an action.
MOVE_EPS_M = 0.05
#: Rounded precision of the pose/heading part of a view key.
VIEW_POSE_ROUND = 2
VIEW_YAW_ROUND = 1
#: Bounded memory of the (view, query) pairs the episode has already asked for.
MAX_SEEN_VIEWS = 512
#: Bounded memory of the recent failures / evidence citations of the STATE block.
MAX_RECENT_FAILURES = 3
MAX_RECENT_EVIDENCE = 4
#: A state value longer than this is truncated in the STATE block.
MAX_STATE_CHARS = 160

_TRUNCATION_NOTE = " ...[truncated]"
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z_\-]*\d+")
_OBS_ID_RE = re.compile(r"obs_\d+")
#: An observation id inside a query is replaced by this token before the query
#: intent is compared: the *view* is carried by the view key, so re-grounding a
#: fresh obs_N of the same view with the same instruction is the same query.
_VIEW_TOKEN = "<view>"
_UNPARSEABLE_NOTE = (
    "Your previous reply contained no decodable tool call. Reply with EXACTLY "
    'ONE JSON object: {"tool": "<name>", "args": {...}} or '
    '{"tool": "finish", "args": {"reason": "<short>"}}.'
)

FINAL_FINISH_NOTE = (
    "RECOVERY EXHAUSTED: the last turns repeated a call that cannot produce "
    "anything new and the automatic recovery steps did not help. Call "
    '{"tool": "finish", "args": {"reason": "<why the task could not be '
    'finished>", "success": false}} now - an honest unfinished answer is '
    "better than repeating a call whose result you already have."
)

#: Data keys whose value changes on every call without carrying information
#: (timestamps, run times): excluded from the "has this answer changed?" test, so
#: a tool cannot look like progress by stamping the same answer with a new time.
VOLATILE_DATA_KEYS = (
    "updated_t",
    "updated_at",
    "elapsed_s",
    "wall_time_s",
    "runtime_ms",
    "ms",
)

#: Task-level signals in a tool result (``data["progress_kind"] == "task"``).
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
)
#: ``verdict`` values of ``compare_candidates`` that mean "same instance".
TASK_VERDICTS = ("same", "match", "identical", "confirmed")
#: Movement outcomes that mean "the leg went nowhere / the plan failed".
BAD_OUTCOMES = (
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
#: Tools through which the runtime may propose a recovery. Only registered ones
#: are ever named, and only a *different* tool counts: a registry with nothing
#: else to call cannot recover (the episode stops as it always did).
RECOVERY_VIEW_TOOLS = ("observe", "inspect_region", "query_memory")
RECOVERY_SEARCH_TOOLS = ("propose_viewpoints", "go_to_viewpoint")
RECOVERY_GROUND_TOOLS = ("ground_target", "compare_candidates")
#: The multi-view sweep the search skill registers: when a place is searched out,
#: one call of it sweeps and grounds somewhere new instead of re-observing.
RECOVERY_SWEEP_TOOL = "search_target"
#: The plan tools and task bookkeeping cannot produce evidence: revising a plan
#: is not a recovery step, so they never count as "something else to call".
NON_EVIDENCE_TOOLS = (
    "update_plan",
    "get_plan",
    "task_status",
    "cancel_task",
    "finish",
    "navigate",
)

DEFAULT_SYSTEM_PROMPT = """You are the decision loop of an autonomous camera drone \
flying inside a photo-realistic 3D scene. At every turn you may call ONE tool to \
obtain evidence or act, or finish the episode.

Rules:
- Reply with EXACTLY ONE JSON object and nothing else; no prose, no markdown, no \
several objects.
- One tool per turn; the next turn starts after you see its structured result.
- Results carry structured fields plus evidence references (image / observation \
/ memory / map / trajectory ids). Cite the ids you rely on.
- Never repeat a call whose result you already have; a repeated call returns \
"cached" and makes no progress.
- Prefer gathering evidence (observe / query_memory / check_path) before long \
moves, and verify before finishing.
- Call "finish" only when the instruction is satisfied (or is impossible); give \
a short reason."""


def _truncate(text: str, limit: int | None, note: str = _TRUNCATION_NOTE) -> str:
    if limit is None or limit <= 0 or len(text) <= limit:
        return text
    cut = max(0, limit - len(note))
    return text[:cut] + note


def _as_lines(value: Any) -> list[str]:
    """Coerce a memory/world summary return value into prompt lines."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [f"{key}={value[key]}" for key in value]
    if isinstance(value, (list, tuple)):
        lines = []
        for item in value:
            if isinstance(item, str):
                lines.append(item)
            elif isinstance(item, dict):
                lines.append(json.dumps(item, default=str))
            elif hasattr(item, "to_dict"):
                lines.append(json.dumps(item.to_dict(), default=str))
            else:
                lines.append(str(item))
        return lines
    return [str(value)]


def _round_floats(value: Any, ndigits: int = 3) -> Any:
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, dict):
        return {k: _round_floats(v, ndigits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(v, ndigits) for v in value]
    return value


def _frame_digest(frame: Any) -> str | None:
    """Short content hash of a frame, or None when it cannot be hashed.

    The repeat test needs the *visual content* of a view, not its id: two
    ``obs_N`` records of the same place, heading and pixels are the same view
    however often the renderer mints a new id for them.
    """
    if frame is None:
        return None
    try:
        if hasattr(frame, "detach"):  # a torch tensor
            frame = frame.detach().cpu().numpy()
        array = np.asarray(frame)
        payload = array.tobytes() + str(array.shape).encode()
    except Exception:  # noqa: BLE001 - an unhashable frame simply has no digest
        return None
    return hashlib.sha1(payload).hexdigest()[:12]


def _norm_token(value: Any) -> str:
    """Lower-case token of a tool outcome/reason for the advice rules."""
    return str(value or "").strip().lower().replace(" ", "_")


def _json_arg(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _call_from_mapping(value: Any) -> tuple[str, dict] | None:
    """``(tool, args)`` from a structured call mapping, or None.

    Accepts ``{"tool": name, "args": {...}}`` (the tool-result ``next_action``
    contract), its ``name``/``arguments`` aliases, and the single-key form
    ``{"<tool>": {...args...}}``.
    """
    if not isinstance(value, dict):
        return None
    name = value.get("tool") or value.get("name")
    if isinstance(name, str) and name.strip():
        args = value.get("args", value.get("arguments", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None
        return (name.strip(), dict(args)) if isinstance(args, dict) else None
    if len(value) == 1:
        name, args = next(iter(value.items()))
        if isinstance(name, str) and isinstance(args, dict):
            return name.strip(), dict(args)
    return None


def _call_from_value(value: Any) -> tuple[str, dict] | None:
    """A structured or text-spelled call (mapping first, then strict text).

    The strict text parser lives in ``tool_impl_task.parse_call_text`` with the
    sub-task schema it serves; a prose step ("observe the corridor carefully")
    parses to None and is never auto-executed.
    """
    call = _call_from_mapping(value)
    if call is not None:
        return call
    from .tool_impl_task import parse_call_text

    return parse_call_text(value)


def _call_args_key(args: Any) -> str:
    """Comparable identity of one call's arguments.

    ``task_id``/``nav_task_id`` are continuation plumbing, not part of *what*
    the call asks for: ``approach_candidate(cand_1)`` and
    ``approach_candidate(cand_1, task_id=nav_3)`` are the same move.
    """
    if not isinstance(args, dict):
        return "{}"
    cleaned = {
        str(key): value
        for key, value in args.items()
        if key not in ("task_id", "nav_task_id") and value is not None
    }
    try:
        return json.dumps(cleaned, sort_keys=True, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(sorted(cleaned.items()))


def _same_call(tool: str, args: Any, call: Any) -> bool:
    """True when ``call`` is exactly the ``tool`` invocation (same tool/args)."""
    parsed = _call_from_mapping(call)
    if parsed is None:
        return False
    name, call_args = parsed
    return name == str(tool) and _call_args_key(call_args) == _call_args_key(args)


def _fruitless_str(status: str) -> bool:
    """A result that cannot continue itself: empty, failed, blocked or replayed."""
    return str(status) in (STATUS_EMPTY, STATUS_FAILED, STATUS_BLOCKED, STATUS_CACHED)


def recovery_advice(
    tool: str,
    data: Any,
    status: str = "",
    *,
    available: Iterable[str] = (),
    coverage: Any = None,
    args: Any = None,
) -> list[str]:
    """Concrete recovery instructions for a turn that made no progress.

    The rules follow the failure the turn reported, and every instruction names
    only tools that are actually registered (``available``) - asking the model
    for a call it cannot make is the same bug as inventing an id:

    - a grounding that found nothing (``no_candidate``): a view that showed no
      candidate is **not** evidence the target does not exist - take a new
      viewpoint, magnify a region, or re-ground from a different heading;
    - a searched-out area (``coverage`` with nothing worth looking at):
      ``propose_viewpoints`` and fly to a new one with ``expected_info``;
    - a planned leg that went nowhere (``planner_failed`` / ``blocked`` /
      ``stalled`` / ``already_at_pose``): take the tool's own ``next_action`` or
      change the approach direction / subgoal;
    - a candidate identity conflict (``conflicts`` / an uncertain verdict):
      ``compare_candidates`` again on fresh evidence before approaching;
    - a repeated read (a replay of an answer already in context): ask a
      different query - a new view or a magnified region.

    Returns ``[]`` when the registry offers no other tool that could produce
    evidence: the caller then handles the stall exactly as before (no invented
    recovery).
    """
    names = {str(name) for name in available}
    data = data if isinstance(data, dict) else {}
    reason = _norm_token(
        data.get("blocked_reason") or data.get("reason") or data.get("outcome")
        or data.get("error") or data.get("hint")
    )
    coverage = coverage if isinstance(coverage, dict) else {}

    def present(*candidates: str) -> list[str]:
        return [name for name in candidates if name in names]

    def has(other: str) -> bool:
        return other in names and other != tool

    advice: list[str] = []
    view_tools = present(*RECOVERY_VIEW_TOOLS)
    search_tools = present(*RECOVERY_SEARCH_TOOLS)
    ground_tools = present(*RECOVERY_GROUND_TOOLS)

    # (a) an answered-but-empty grounding: absence is not inferable from one view
    no_candidate = (
        reason in ("no_candidate", "no_vlm_backend", "unparsable_reply")
        or _norm_token(data.get("status")) == "no_candidate"
        or data.get("n_candidates") == 0
        or (isinstance(data.get("candidates"), list) and not data["candidates"]
            and "candidates" in data)
    )
    # (a0) the search routine stopped at its own step cap: it returns what it has,
    # so the search is not over - the answer is to continue it, not to re-ground
    # the same view
    if tool == RECOVERY_SWEEP_TOOL and reason == "inner_steps_exhausted":
        advice = [
            "the search routine reached its step cap without a confirmed candidate: "
            "the search is unfinished, not answered"
        ]
        advice.append(
            f"do not repeat that same {RECOVERY_SWEEP_TOOL} call at this pose: "
            "take a fresh view (observe / propose_viewpoints + go_to_viewpoint "
            "with the expected_info that would settle the task) and continue the "
            f"search with {RECOVERY_SWEEP_TOOL} from there"
        )
    elif no_candidate:
        where = str(data.get("observation_id") or "the last view")
        advice = [
            f"the grounding in {where} matched no candidate - that is not evidence "
            "of absence, only that this view did not show the target"
        ]
        if view_tools:
            advice.append(
                f"take a different view before concluding anything: {', '.join(view_tools)}"
            )
        if "propose_viewpoints" in names and "go_to_viewpoint" in names:
            advice.append(
                "or fly to a viewpoint you have not used (propose_viewpoints, then "
                "go_to_viewpoint with the expected_info)"
            )
        if ground_tools:
            advice.append(f"then re-ground from that view with {ground_tools[0]}")

    # (b) the searched area is exhausted: the honest next step is elsewhere
    elif coverage and _coverage_exhausted(coverage):
        checked = _coverage_scalar(coverage.get("checked_deg"))
        advice = [
            f"the area is searched out (checked={checked if checked is not None else '?'}"
            f" deg, unchecked={_coverage_scalar(coverage.get('unchecked_deg'))} deg, "
            f"worth_looking={coverage.get('worth_looking')}): do not re-observe it"
        ]
        if search_tools:
            advice.append(
                "propose_viewpoints for a new region and fly to the best one with "
                "go_to_viewpoint, stating the expected_info that would settle the task"
            )
        if RECOVERY_SWEEP_TOOL in names and RECOVERY_SWEEP_TOOL != tool:
            advice.append(
                f"or run {RECOVERY_SWEEP_TOOL} from a new place: it sweeps several "
                "views and grounds them in one call"
            )

    # (c) an identity conflict between candidates
    elif data.get("conflicts") or _norm_token(data.get("verdict")) in (
        "different", "uncertain", "conflict", "mismatch"
    ):
        advice = [
            "the candidates disagree about the target: do not approach either yet"
        ]
        if "compare_candidates" in names:
            advice.append(
                "collect one more piece of evidence (a new view or a magnified "
                "region) and call compare_candidates again on the same two ids"
            )
        if view_tools:
            advice.append(f"the extra evidence comes from: {', '.join(view_tools)}")

    # (d) a leg that went nowhere: change the approach, not the target
    elif _norm_token(data.get("outcome")) in BAD_OUTCOMES or reason in BAD_OUTCOMES or (
        tool in ("approach_candidate", "go_to_viewpoint", "return_to_observation", "navigate")
        and status in (STATUS_BLOCKED, STATUS_FAILED)
    ):
        next_action = data.get("next_action") if isinstance(data.get("next_action"), dict) else {}
        advice = [f"the last leg did not get anywhere ({reason or data.get('outcome')})"]
        if next_action.get("tool") and str(next_action["tool"]) in names:
            if args is not None and _same_call(tool, args, next_action):
                # the result names the call that just returned nothing: echoing
                # it is the recovery half of the self-repeating loop
                advice.append(
                    "that tool's own next step is the call that just produced "
                    "nothing - do not repeat it"
                )
            else:
                advice.append(
                    f"the tool's own next step: {next_action['tool']}"
                    f"({_json_arg(next_action.get('args') or {})})"
                )
        missing = [
            name for name in ("propose_viewpoints", "go_to_viewpoint")
            if has(name)
        ]
        if missing:
            advice.append(
                "or change the approach direction: propose_viewpoints, then "
                "go_to_viewpoint to a reachable vp_N (a subgoal you can actually "
                "fly to beats repeating this one)"
                if len(missing) == 2
                else f"or change the approach direction with {missing[0]}"
            )
        for name in ("observation_id", "last_observation_id"):
            observation = data.get(name)
            if observation and has("return_to_observation"):
                advice.append(
                    f"going back to {observation} re-opens the view the leg started from"
                )
                break

    # (e) a repeated read: the answer is already in context
    elif status == STATUS_CACHED or _norm_token(data.get("hint")).startswith("repeat") or (
        _norm_token(data.get("reason")) in ("repeat", "duplicate")
    ):
        advice = [
            "that query is already answered in your context: the same call cannot "
            "produce anything new"
        ]
        if view_tools:
            advice.append(
                f"ask something you have not asked yet: {', '.join(view_tools)} "
                "(a new heading/mode, or a region you have not magnified)"
            )
        if search_tools:
            advice.append(
                f"or move to a new observation position: {', '.join(search_tools)}"
            )

    if advice and _named_tools(names, " ".join(advice)):
        return advice
    # no rule produced an actionable step: keep the model working, but only with
    # a tool that exists and is not the one it keeps repeating
    others = [
        name for name in sorted(names)
        if name not in NON_EVIDENCE_TOOLS and name != tool
    ]
    if others:
        return [
            f"'{tool}' produced nothing new: call a different tool that can still "
            f"change what you know, e.g. {', '.join(others[:3])}"
        ]
    return []


def _named_tools(names: Iterable[str], text: str) -> list[str]:
    """Registered, evidence-capable tools named in ``text`` (never bookkeeping)."""
    return [
        name for name in sorted(set(names))
        if name not in NON_EVIDENCE_TOOLS
        and re.search(rf"\b{re.escape(str(name))}\b", text)
    ]


def _as_float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fingerprint(tool: str, data: Any, evidence: Any = ()) -> str:
    """Canonical payload of a result, for tools whose answer carries no view."""
    payload = {
        "tool": tool,
        "data": {
            key: value
            for key, value in _round_floats(data if isinstance(data, dict) else {}).items()
            if key not in VOLATILE_DATA_KEYS
        },
        "evidence": sorted(
            f"{ref.get('kind')}:{ref.get('ref')}"
            for ref in (evidence or ())
            if isinstance(ref, dict)
        ),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _task_signal(data: Any) -> bool:
    """True when a result reports the task itself advancing (not just a view).

    ``{"verified": true}``, ``{"subgoal_complete": true}``, an identity confirmed
    by ``compare_candidates`` (``verdict`` same) and an explicit
    ``progress_kind="task"`` all mean: this is the milestone, not the scenery.
    """
    if not isinstance(data, dict):
        return False
    for key in TASK_SIGNAL_KEYS:
        if data.get(key) is True:
            return True
    verdict = _norm_token(data.get("verdict"))
    if verdict in TASK_VERDICTS:
        return True
    return _norm_token(data.get("status")) in ("complete", "completed", "arrived")


def _coverage_of(data: Any) -> dict | None:
    """The search coverage of a result: ``{checked_deg, unchecked_deg,
    worth_looking}`` (the contract the lookaround/search results carry)."""
    if not isinstance(data, dict):
        return None
    for key in ("coverage", "search_coverage", "observed_coverage"):
        value = data.get(key)
        if isinstance(value, dict):
            return dict(value)
    return None


def _coverage_of_extra(ctx: Any) -> dict | None:
    """Coverage a tool published on the context (``ctx.extra["coverage"]``)."""
    extra = getattr(ctx, "extra", None)
    value = extra.get("coverage") if isinstance(extra, dict) else None
    return dict(value) if isinstance(value, dict) else None


def _coverage_scalar(value: Any) -> float | None:
    """Degrees of a coverage field written as a number or as sector ranges.

    The search/observe contract carries ``checked_deg`` / ``unchecked_deg``
    either as one number or as the ``[[low, high], ...]`` sector ranges the
    lookaround report uses; both mean "how many degrees are (un)covered".
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (list, tuple)):
        total = 0.0
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                low, high = _as_float_or_none(item[0]), _as_float_or_none(item[1])
                if low is None or high is None:
                    return None
                total += abs(high - low)
            else:
                number = _as_float_or_none(item)
                if number is None:
                    return None
                total += number
        return total
    return None


def _coverage_text(coverage: Any) -> str:
    """``checked=240deg unchecked=120deg worth_looking=yes`` or ``n/a``.

    Renders both spellings of the shared coverage contract: the scalar one and
    the sector-range one of the lookaround/search report (``worth_looking`` as a
    flag or as the list of views worth a second look).
    """
    if not isinstance(coverage, dict) or not coverage:
        return "n/a"
    checked = _coverage_scalar(coverage.get("checked_deg"))
    unchecked = _coverage_scalar(coverage.get("unchecked_deg"))
    worth = coverage.get("worth_looking")
    if isinstance(worth, bool):
        worth_text = "yes" if worth else "no"
    elif isinstance(worth, (list, tuple, dict, set)):
        worth_text = f"yes({len(worth)} views)" if worth else "no"
    else:
        worth_text = "?"
    if coverage.get("directions_known") is False:
        unchecked = None  # unknown direction is not "nothing left to check"
    return (
        f"checked={_fmt_deg(checked)}deg "
        f"unchecked={_fmt_deg(unchecked)}deg "
        f"worth_looking={worth_text}"
    )


def _fmt_deg(value: float | None) -> str:
    """Degrees in a prompt-friendly form (``180`` rather than ``180.0``)."""
    if value is None:
        return "?"
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def _coverage_exhausted(coverage: Any) -> bool:
    """True when the searched area has nothing left worth looking at."""
    if not isinstance(coverage, dict) or not coverage:
        return False
    if coverage.get("directions_known") is False:
        return False  # the headings were unknown: not the same as "covered"
    worth = coverage.get("worth_looking")
    if worth is False:
        return True
    if isinstance(worth, (list, tuple, dict, set)) and not worth:
        return True
    return _coverage_scalar(coverage.get("unchecked_deg")) == 0.0


def _outcome_text(record: TurnRecord) -> str:
    """The failure token of a turn ("" when it is not a failure).

    Failed, blocked and replayed calls always report one; a successful call only
    when the tool named a reason for a partial result - which is what the STATE
    block's ``recent_failures`` and the recovery rules read.
    """
    data = record.data if isinstance(record.data, dict) else {}
    token = _norm_token(
        data.get("blocked_reason") or data.get("reason") or data.get("outcome")
        or data.get("error")
    )
    if record.status in (STATUS_FAILED, STATUS_BLOCKED, STATUS_CACHED):
        return token or record.status
    if token in BAD_OUTCOMES or token == "no_candidate":
        # an empty grounding is progress (an exclusion) *and* a failure signal:
        # the STATE block names it so the model can see where it already looked
        return token
    return ""


def _observation_ids(record: TurnRecord) -> list[str]:
    """The ``obs_N`` ids a result is about (evidence refs first, then its data)."""
    ids: list[str] = []
    for ref in record.evidence:
        if str(ref.get("kind")) == "observation":
            ids.append(str(ref.get("ref")))
    def scan(value: Any) -> None:
        if isinstance(value, str):
            ids.extend(_OBS_ID_RE.findall(value))
        elif isinstance(value, dict):
            for item in value.values():
                scan(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                scan(item)

    scan(record.data)
    ordered: list[str] = []
    for obs_id in ids:
        if obs_id and obs_id not in ordered:
            ordered.append(obs_id)
    return ordered


@dataclass
class TurnRecord:
    """One executed tool call (or one failed attempt at a call)."""

    turn: int
    tool: str
    args: dict
    status: str
    ms: float
    evidence: list[dict] = field(default_factory=list)
    used: bool = False
    new_evidence: bool = False
    images_sent: int = 0
    data: dict = field(default_factory=dict)
    raw_reply: str = field(default="", repr=False)
    result_prompt: str = field(default="", repr=False)
    detail: str = ""
    #: ``information`` | ``action`` | ``task`` | ``none`` - what the turn really
    #: changed (``new_evidence`` is the legacy name of "not none").
    progress_kind: str = PROGRESS_NONE
    #: ``""``, ``advice`` (a recovery instruction followed this turn), ``hook``
    #: (the recovery hook executed this call) or ``final_finish``.
    recovery: str = ""
    #: The decision node of the turn (one of :data:`DECISION_NODES`), or ``""``
    #: for a routine turn the executor absorbed.
    decision: str = ""
    #: True when the executor issued this call itself (no model turn was spent):
    #: ``auto_origin`` says which deterministic source named it.
    auto: bool = False
    auto_origin: str = ""

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "tool": self.tool,
            "args": self.args,
            "status": self.status,
            "ms": self.ms,
            "evidence": self.evidence,
            "data": self.data,
            "used": self.used,
            "new_evidence": self.new_evidence,
            "images_sent": self.images_sent,
            "detail": self.detail,
            "progress_kind": self.progress_kind,
            "recovery": self.recovery,
            "decision": self.decision,
            "auto": self.auto,
            "auto_origin": self.auto_origin,
            "result_prompt": self.result_prompt,
            "raw_reply": self.raw_reply,
        }


@dataclass
class EpisodeLog:
    """Per-episode call chain: what the model asked, what the tools answered."""

    episode_id: str
    instruction: str
    turns: list[TurnRecord] = field(default_factory=list)
    finish_reason: str = ""
    success: bool | None = None
    finish_args: dict = field(default_factory=dict)
    vlm_calls: int = 0
    wall_time_s: float = 0.0
    budget: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    vlm_log: list[dict] = field(default_factory=list)
    #: The fixed STATE block of the last turn (the agent's own view of itself).
    state: dict = field(default_factory=dict)
    #: One entry per stall the runtime reacted to: ``{"turn", "kind", "attempt",
    #: "tool", "advice"}`` with ``kind`` from ``RECOVERY_*``.
    recoveries: list[dict] = field(default_factory=list)
    #: One entry per sub-task status the executor advanced (P4):
    #: ``{"turn", "id", "from", "to", "reason"}``.
    plan_transitions: list[dict] = field(default_factory=list)
    #: One entry per decision node the loop hit: ``{"turn", "node", "subtask"}``.
    decision_nodes: list[dict] = field(default_factory=list)
    #: What :func:`tool_impl_task.resume_plan` adopted at the start of the
    #: episode (``{}`` when the plan was written by this episode).
    plan_resumed: dict = field(default_factory=dict)
    #: One entry per call the executor issued on its own (no model turn):
    #: ``{"turn", "tool", "args", "status", "origin", "decision"}``. Each entry
    #: is one VLM turn the deterministic plan step saved.
    executor_auto_steps: list[dict] = field(default_factory=list)
    #: What the last system prompt really held: the cap, the rendered lengths
    #: and whether the catalogue arrived whole (P4 prompt budget).
    prompt_budget: dict = field(default_factory=dict)
    #: The number of turns that asked the model (auto steps share their turn).
    model_turns: int = 0

    def tool_calls(self) -> int:
        return sum(1 for t in self.turns if t.tool not in ("", NO_TOOL))

    def tool_names(self) -> list[str]:
        return [t.tool for t in self.turns if t.tool not in ("", NO_TOOL)]

    def used_tools(self) -> list[str]:
        return [t.tool for t in self.turns if t.used]

    def progress_kinds(self) -> list[str]:
        return [t.progress_kind for t in self.turns]

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "instruction": self.instruction,
            "finish_reason": self.finish_reason,
            "success": self.success,
            "finish_args": self.finish_args,
            "vlm_calls": self.vlm_calls,
            "tool_calls": self.tool_calls(),
            "wall_time_s": self.wall_time_s,
            "budget": dict(self.budget),
            "events": list(self.events),
            "vlm_log": list(self.vlm_log),
            "state": dict(self.state),
            "recoveries": [dict(entry) for entry in self.recoveries],
            "plan_transitions": [dict(entry) for entry in self.plan_transitions],
            "decision_nodes": [dict(entry) for entry in self.decision_nodes],
            "plan_resumed": dict(self.plan_resumed),
            "executor_auto_steps": [dict(entry) for entry in self.executor_auto_steps],
            "executor_auto_step_count": len(self.executor_auto_steps),
            "vlm_turns_saved": len(self.executor_auto_steps),
            "prompt_budget": dict(self.prompt_budget),
            "model_turns": int(self.model_turns),
            "turns": [t.to_dict() for t in self.turns],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2, default=str)
        return path

    def summary_text(self) -> str:
        lines = [
            f"episode {self.episode_id} | {self.finish_reason} | "
            f"success={self.success} | turns={len(self.turns)} | "
            f"tool_calls={self.tool_calls()} | vlm_calls={self.vlm_calls} | "
            f"{self.wall_time_s:.1f}s"
        ]
        for record in self.turns:
            lines.append(
                f"  {record.turn:02d} {record.tool} "
                f"{json.dumps(record.args, default=str)[:80]} -> "
                f"[{record.status}] {record.ms}ms used={record.used} "
                f"progress={record.progress_kind} new_evidence={record.new_evidence} "
                f"images={record.images_sent}"
                + (f" recovery={record.recovery}" if record.recovery else "")
            )
        if self.recoveries:
            lines.append(
                "  recoveries: "
                + ", ".join(
                    f"t{entry.get('turn')}:{entry.get('kind')}" for entry in self.recoveries
                )
            )
        if self.plan_transitions:
            lines.append(
                "  plan: "
                + ", ".join(
                    f"t{entry.get('turn')}:{entry.get('id')}->{entry.get('to')}"
                    for entry in self.plan_transitions
                )
            )
        if self.decision_nodes:
            lines.append(
                "  decision_nodes: "
                + ", ".join(
                    f"t{entry.get('turn')}:{entry.get('node')}" for entry in self.decision_nodes
                )
            )
        if self.executor_auto_steps:
            lines.append(
                "  executor_auto_steps: "
                + ", ".join(
                    f"t{entry.get('turn')}:{entry.get('tool')}({entry.get('origin')})"
                    for entry in self.executor_auto_steps
                )
            )
        return "\n".join(lines)


class AgentRuntime:
    """Tool-calling episode loop over a world, a tool registry and a VLM.

    Per turn it renders the world, asks the model for one tool call, executes it
    and scores the answer (``information`` / ``action`` / ``task`` / ``none``);
    a streak of ``max_stagnant_turns`` no-progress turns triggers the recovery
    rules (``recovery_advice``, optionally executed by ``recovery_hook``) before
    the loop is allowed to stop. The episode's own state rides in every prompt as
    the fixed ``STATE`` block (:meth:`state`, :data:`STATE_LABELS`).

    Two switches turn those layers off for an ablation (both default to the
    current behavior): ``recovery_enabled=False`` stops at the stagnation limit
    without advice, recovery hook or honest-finish grant, and
    ``state_block=False`` keeps the fixed block out of the prompt and out of
    ``context["state"]`` while :meth:`state` still reports it.

    P4 adds three switches, all on by default: ``subtask_executor=False`` stops
    the runtime from advancing the plan's sub-task records (statuses only change
    through ``update_plan``, no decision node is classified, no plan revision is
    requested), ``plan_resume=False`` refuses to adopt the plan an interrupted
    episode left in a caller-provided context, and ``executor_auto_steps=False``
    keeps every call in the model's hands: with the switch on, a routine turn
    whose plan names a deterministic next step (the plan's/sub-task's own next
    step, the last result's ``next_action``, a concrete failure branch after a
    fruitless result, or a sub-task goal written as a tool call) executes that
    step through the registry instead of spending a model turn, capped at
    :data:`MAX_EXECUTOR_AUTO_STEPS` consecutive steps.
    """

    def __init__(
        self,
        registry,
        world,
        memory=None,
        vlm=None,
        budget: int | dict | None = 32,
        max_turns: int = 12,
        *,
        episode_id: str | None = None,
        max_images: int = 4,
        max_history_images: int = 2,
        max_cache_images: int = 24,
        max_system_chars: int = DEFAULT_SYSTEM_CHARS,
        max_text_chars: int = 6000,
        max_tool_chars: int = 400,
        max_result_history: int = 3,
        max_stagnant_turns: int = 3,
        max_recovery_attempts: int = 4,
        recovery_hook: Callable[[dict], dict | None] | None = None,
        recovery_enabled: bool = True,
        state_block: bool = True,
        subtask_executor: bool = True,
        plan_resume: bool = True,
        executor_auto_steps: bool = True,
        max_executor_auto_steps: int = MAX_EXECUTOR_AUTO_STEPS,
        max_memory_items: int = 4,
        max_history_turns: int = 8,
        feedback_in_prompt: bool = True,
        tool_context: ToolContext | None = None,
        system_prompt_template: str = DEFAULT_SYSTEM_PROMPT,
        tool_call_doc: str = TOOL_CALL_DOC,
        vlm_context: dict | None = None,
        on_event: Callable[[str, dict], None] | None = None,
    ):
        self.registry = registry
        self.world = world
        self.memory = memory
        self.vlm = vlm
        self.budget = (
            {"tool_calls_left": int(budget)}
            if isinstance(budget, int) or budget is None
            else dict(budget)
        )
        self.max_turns = int(max_turns)
        self.episode_id = episode_id
        self.max_images = max(1, int(max_images))
        self.max_history_images = max(0, int(max_history_images))
        self.max_cache_images = max(1, int(max_cache_images))
        self.max_system_chars = int(max_system_chars)
        self.max_text_chars = int(max_text_chars)
        self.max_tool_chars = int(max_tool_chars)
        self.max_result_history = max(1, int(max_result_history))
        self.max_stagnant_turns = max(1, int(max_stagnant_turns))
        self.max_memory_items = max(1, int(max_memory_items))
        self.max_history_turns = max(0, int(max_history_turns))
        self.feedback_in_prompt = bool(feedback_in_prompt)
        self.max_tools_chars = int(self.max_system_chars)
        self.tool_context = tool_context
        self.system_prompt_template = system_prompt_template
        self.tool_call_doc = tool_call_doc
        self.extra_context = dict(vlm_context or {})
        #: recovery knobs: how many instructions the model gets before the loop
        #: stops, and the optional hook that executes one recovery call itself.
        self.max_recovery_attempts = max(0, int(max_recovery_attempts))
        self.recovery_hook = recovery_hook
        #: ablation switches: off = stop at the stagnation limit without any
        #: recovery step / keep the fixed STATE block out of the prompts.
        self.recovery_enabled = bool(recovery_enabled)
        self.state_block = bool(state_block)
        #: P4 switches: the sub-task executor advances the plan record after
        #: every call; a re-run may adopt the plan an interrupted episode left;
        #: a routine turn's deterministic step runs without a model call.
        self.subtask_executor = bool(subtask_executor)
        self.plan_resume = bool(plan_resume)
        self.executor_auto_steps = bool(executor_auto_steps)
        self.max_executor_auto_steps = max(0, int(max_executor_auto_steps))
        #: What the last system prompt held (read by the runner's summary).
        self.prompt_budget: dict = {}
        self._last_catalogue: dict = {}

        self.ctx: ToolContext | None = None
        self.log: EpisodeLog | None = None
        self.instruction = ""
        self.abort_requested = False
        self.abort_reason = ""
        self.vlm_calls = 0
        self.backend_log: list[dict] = []

        self._hooks: list[Callable[[str, dict], None]] = []
        if on_event is not None:
            self.on_event(on_event)
        self._event_lock = threading.Condition()
        self._pending_events: list[dict] = []
        self._event_seq = 0

        self._image_cache: "OrderedDict[str, Any]" = OrderedDict()
        self._known_image_ids: set[str] = set()
        self._pending_image_ids: list[str] = []
        self._feedback: list[str] = []
        self._history: list[dict] = []
        self._seen_evidence: set[tuple[str, str]] = set()
        self._seen_fingerprints: set[str] = set()
        #: (query intent, view keys) pairs the episode already asked for: the
        #: repeat test of progress, keyed by pose + heading + content + query.
        self._seen_view_queries: "OrderedDict[tuple, None]" = OrderedDict()
        self._stagnant_turns = 0
        self._recovery_attempts = 0
        self._finish_granted = False
        #: the last decision of :meth:`_stall_action` ("" / RECOVERY_*), readable
        #: by a driver that reports what the loop did about a stall.
        self.last_stall_action = ""
        self._recovery_notes: list[str] = []
        self._final_note = ""
        self._coverage: dict = {}
        self._recent_failures: list[str] = []
        self._recent_evidence: list[str] = []
        self._next_info_needed = ""
        self._last_state: dict = {}
        self._last_record: TurnRecord | None = None
        #: P4: the plan revision the model still owes, the executor's transitions
        #: of the last call, and the decision nodes of the episode.
        self._plan_revision_note = ""
        self._revision_node = ""
        self._revision_subtask = ""
        self._executor_notes: list[str] = []
        self._candidate_ids: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------

    def on_event(self, hook: Callable[[str, dict], None]) -> Callable[[str, dict], None]:
        """Register an ``fn(name, data)`` hook; returns the hook unchanged."""
        self._hooks.append(hook)
        return hook

    def emit_event(self, name: str, data: dict | None = None) -> dict:
        """Queue an event (thread-safe) and notify hooks; the loop drains the
        queue at the next turn boundary, so an event from a tool or an external
        producer wakes the model loop instead of being lost."""
        with self._event_lock:
            self._event_seq += 1
            record = {
                "seq": self._event_seq,
                "name": str(name),
                "data": dict(data or {}),
            }
            self._pending_events.append(record)
            self._event_lock.notify_all()
        for hook in list(self._hooks):
            try:
                hook(record["name"], record["data"])
            except Exception:
                pass  # hooks are observational; they never break the loop
        return record

    def pending_events(self) -> list[dict]:
        with self._event_lock:
            return list(self._pending_events)

    def drain_events(self) -> list[dict]:
        with self._event_lock:
            events, self._pending_events = self._pending_events, []
            return events

    def wait_for_event(self, timeout: float | None = None) -> dict | None:
        """Pop the next event, blocking up to ``timeout`` seconds (used by a
        driver that runs the loop in another thread)."""
        with self._event_lock:
            if not self._pending_events:
                self._event_lock.wait(timeout)
            return self._pending_events.pop(0) if self._pending_events else None

    def request_abort(self, reason: str = "") -> None:
        self.abort_requested = True
        self.abort_reason = str(reason)

    # ------------------------------------------------------------------
    # episode loop
    # ------------------------------------------------------------------

    def run(
        self,
        instruction: str,
        *,
        memory=None,
        episode_id: str | None = None,
        max_turns: int | None = None,
    ) -> EpisodeLog:
        if self.vlm is None:
            raise ValueError("AgentRuntime.run() needs a VLM backend")
        if memory is not None:
            self.memory = memory
        turns_limit = int(max_turns or self.max_turns)
        episode_id = episode_id or self.episode_id or f"ep_{uuid.uuid4().hex[:8]}"
        self.episode_id = episode_id
        self.instruction = str(instruction)
        self._reset_episode_state()

        # a caller-provided context (runner/MCP) is adopted: its extra knobs,
        # budget and log survive, so tools keep whatever the harness prepared
        ctx = self.tool_context
        if ctx is None:
            ctx = ToolContext(
                world=self.world, memory=self.memory, episode_id=episode_id,
                budget=dict(self.budget), log=[],
            )
        ctx.world = self.world
        ctx.memory = self.memory
        ctx.episode_id = episode_id
        if not isinstance(ctx.budget, dict):
            ctx.budget = dict(self.budget)
        ctx.budget.setdefault("tool_calls_left", int(self.budget.get("tool_calls_left", 0)))
        if not isinstance(ctx.log, list):
            ctx.log = []
        if not isinstance(ctx.extra, dict):
            ctx.extra = {}
        ctx.extra["instruction"] = self.instruction
        ctx.extra["runtime"] = self
        ctx.extra.setdefault("images", {})
        ctx.extra.setdefault("events", [])
        self._begin_observation_store(ctx, episode_id)
        self.ctx = ctx
        log = EpisodeLog(
            episode_id=episode_id, instruction=self.instruction, budget=dict(ctx.budget)
        )
        self._resume_plan(log)
        start = time.time()
        reason, success, finish_args = REASON_MAX_TURNS, False, {}
        model_turns = 0

        for turn in range(1, turns_limit + 1):
            # events first: an event that caused an abort is still logged
            events = self.drain_events()
            log.events.extend(events)
            if self.abort_requested:
                reason = REASON_ABORTED
                break
            if int(ctx.budget.get("tool_calls_left", 0)) <= 0:
                reason = REASON_BUDGET_EXHAUSTED
                break

            model_turns = turn
            call, reply, images_sent = self._ask(turn, events)
            log.vlm_calls += 1
            if call is None:  # one re-prompt, per the parse_tool_call contract
                call, reply, images_sent = self._ask(turn, events, retry_note=_UNPARSEABLE_NOTE)
                log.vlm_calls += 1
            if call is None:
                log.turns.append(
                    TurnRecord(
                        turn=turn, tool=NO_TOOL, args={}, status=STATUS_FAILED,
                        ms=0.0, images_sent=images_sent, raw_reply=reply,
                        detail="unparseable_model_output",
                    )
                )
                reason = REASON_UNPARSEABLE
                break

            tool, args = call
            if tool == FINISH_TOOL:
                pose_before = self._pose_snapshot()
                candidates_before = self._candidate_ids_snapshot()
                t0 = time.time()
                result = self._finish_result(args)
                ms = round((time.time() - t0) * 1000.0, 1)
                terminal = result.status == STATUS_OK
                record = self._turn_record(turn, tool, args, result, ms, images_sent, reply)
                log.turns.append(record)
                self._note_result(tool, args, result, record)
                self._score_turn(record, log, pose_before)
                self._executor_step(turn, log, record, candidates_before)
                if terminal:
                    reason, finish_args = REASON_FINISH, dict(args)
                    # success must be claimed explicitly (e.g. via verify_goal or
                    # "success": true); a bare finish leaves the verdict open
                    claimed = result.data.get("success")
                    success = bool(claimed) if claimed is not None else None
                    break
                # a vetoed finish can move nothing either: it recovers or stalls
                if self._after_stall(turn, log, record) == RECOVERY_STOP:
                    reason = REASON_NO_NEW_EVIDENCE
                    break
                continue

            pose_before = self._pose_snapshot()
            candidates_before = self._candidate_ids_snapshot()
            t0 = time.time()
            result = self._call_tool(tool, args)
            ms = round((time.time() - t0) * 1000.0, 1)
            record = self._turn_record(turn, tool, args, result, ms, images_sent, reply)
            log.turns.append(record)
            self._note_result(tool, args, result, record)
            self._score_turn(record, log, pose_before)
            self._executor_step(turn, log, record, candidates_before)
            # a routine turn whose plan names a deterministic next step is
            # executed here, so the chain costs no model turn (P4); a decision
            # node stops the chain and the model sees it next turn
            record = self._run_executor_auto_steps(turn, log, record)
            if self._after_stall(turn, log, record) == RECOVERY_STOP:
                reason = REASON_NO_NEW_EVIDENCE
                break

        log.finish_reason = reason
        log.success = success if success is None else bool(success)
        log.finish_args = finish_args
        log.wall_time_s = round(time.time() - start, 3)
        log.budget = dict(ctx.budget)
        log.vlm_log = list(self.backend_log)
        self._mark_used(log)
        log.state = dict(self._last_state)
        log.prompt_budget = dict(self.prompt_budget)
        log.model_turns = int(model_turns)
        self.log = log
        return log

    def _reset_episode_state(self) -> None:
        self.abort_requested = False
        self.abort_reason = ""
        self.vlm_calls = 0
        self.backend_log = []
        self._pending_events = []
        self._image_cache = OrderedDict()
        self._known_image_ids = set()
        self._pending_image_ids = []
        self._feedback = []
        self._history = []
        self._seen_evidence = set()
        self._seen_fingerprints = set()
        self._seen_view_queries = OrderedDict()
        self._stagnant_turns = 0
        self._recovery_attempts = 0
        self._finish_granted = False
        self.last_stall_action = ""
        self._recovery_notes = []
        self._final_note = ""
        self._coverage = {}
        self._recent_failures = []
        self._recent_evidence = []
        self._next_info_needed = ""
        self._last_state = {}
        self._last_record = None
        self._plan_revision_note = ""
        self._revision_node = ""
        self._revision_subtask = ""
        self._executor_notes = []
        self._candidate_ids = ()
        self.prompt_budget = {}
        self._last_catalogue = {}

    # ------------------------------------------------------------------
    # model interface
    # ------------------------------------------------------------------

    def _ask(self, turn: int, events: list[dict], retry_note: str | None = None):
        """One model call: returns ``(normalized_call, reply_text, n_images)``."""
        frames, user = self._build_context(turn, events, retry_note)
        context = self._vlm_context(turn)
        tools_text = self.tools_text()
        decide_with_tools = getattr(self.vlm, "decide_with_tools", None)
        if callable(decide_with_tools):
            # tool-aware backends append the protocol/catalogue themselves
            reply = self._invoke_tool_path(
                decide_with_tools, frames, self._system_prompt(False), user,
                tools_text, context,
            )
            if reply is not None:
                self.vlm_calls += 1
                return self._normalize_reply(reply), self._reply_text(reply), len(frames)
        # plain text backends need the protocol and the catalogue in the prompt
        system = self._system_prompt(True)
        reply = self.vlm.decide(frames, system, user, context)
        self.vlm_calls += 1
        return self._normalize_reply(reply), self._reply_text(reply), len(frames)

    @staticmethod
    def _invoke_tool_path(
        fn, frames, system: str, user: str, tools_text: str, context: dict
    ) -> Any:
        """Call ``decide_with_tools`` across the known signature variants.

        The runtime contract is ``(frames, system_prompt, user_prompt,
        tools_text, context)``; keyword binding is tried first, then the
        positional forms down to the older 4-argument
        ``(frames, system, user, context)``. Returns None when the callable
        accepts none of them, so the caller can fall back to ``decide``.
        """
        candidates = (
            ((frames, system, user), {"tools_text": tools_text, "context": context}),
            ((frames, system, user, tools_text, context), {}),
            ((frames, system, user, context), {}),
            ((frames, system, user), {}),
        )
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            for args, kwargs in candidates:
                try:
                    signature.bind(*args, **kwargs)
                except TypeError:
                    continue
                return fn(*args, **kwargs)
            return None
        for args, kwargs in candidates:
            try:
                return fn(*args, **kwargs)
            except TypeError:
                continue
        return None

    def _catalogue_text(self) -> tuple[str, int, bool]:
        """``(bounded text, raw length, truncated)`` of the rendered catalogue."""
        if self.registry is None:
            return "", 0, False
        try:
            text = self.registry.specs_text()
        except Exception:  # noqa: BLE001 - a broken registry renders nothing
            return "", 0, False
        raw = len(text)
        if self.max_tools_chars and raw > self.max_tools_chars:
            return _truncate(text, self.max_tools_chars), raw, True
        return text, raw, False

    def tools_text(self) -> str:
        """Rendered tool catalogue passed to tool-aware backends (bounded).

        The bound is ``max_tools_chars`` (the system budget by default); with
        the default budget the state-aware catalogue of a full registry arrives
        whole. The rendered state is kept for :meth:`_system_prompt`'s prompt
        budget, which is what the runner reports.
        """
        text, raw, truncated = self._catalogue_text()
        self._last_catalogue = {
            "chars": len(text),
            "raw_chars": raw,
            "truncated": truncated,
        }
        return text

    def tool_specs(self) -> list[dict]:
        """Structured specs for native tool calling (``context["tools"]``)."""
        specs = []
        for name in self.registry.names():
            try:
                spec = self.registry.get(name)
            except Exception:
                continue
            specs.append(
                {
                    "name": str(getattr(spec, "name", name)),
                    "description": str(getattr(spec, "description", "")),
                    "parameters": getattr(spec, "input_schema", None)
                    or {"type": "object", "properties": {}},
                }
            )
        return specs

    def _record_backend_log(self, event: dict) -> None:
        """``context["on_log"]`` hook of the VLM backends (audit trail)."""
        try:
            self.backend_log.append(dict(event))
        except Exception:
            return
        if len(self.backend_log) > 64:
            del self.backend_log[:-64]

    @staticmethod
    def _reply_text(reply: Any) -> str:
        if isinstance(reply, str):
            return reply
        if isinstance(reply, (dict, list, tuple)):
            return json.dumps(reply, default=str)
        return str(reply or "")

    def _normalize_reply(self, reply: Any) -> tuple[str, dict] | None:
        if isinstance(reply, str):
            return parse_tool_call(reply)
        if isinstance(reply, (list, tuple)):
            for item in reply:
                call = self._normalize_reply(item)
                if call is not None:
                    return call
            return None
        if isinstance(reply, dict):
            return self._normalize_dict(reply)
        return None

    @staticmethod
    def _normalize_dict(data: dict) -> tuple[str, dict] | None:
        if "tool" in data:
            tool = str(data.get("tool", "")).strip()
            args = data.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if tool and isinstance(args, dict):
                return tool, dict(args)
            return None
        # OpenAI-style native tool-call payload
        calls = data.get("tool_calls")
        if isinstance(calls, list) and calls:
            first = calls[0]
            if isinstance(first, dict):
                fn = first.get("function") if isinstance(first.get("function"), dict) else first
                name = str(fn.get("name", "")).strip()
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if name and isinstance(args, dict):
                    return name, dict(args)
        return None

    def _vlm_context(self, turn: int) -> dict:
        context = dict(self.extra_context)
        context.setdefault("episode_id", self.ctx.episode_id)
        context.setdefault("instruction", self.instruction)
        context.setdefault("turn", turn)
        context.setdefault("step", turn)
        context.setdefault("max_turns", self.max_turns)
        context.setdefault("tool_calls_left", int(self.ctx.budget.get("tool_calls_left", 0)))
        context.setdefault("tool_names", list(self.registry.names()))
        context.setdefault("tools", self.tool_specs())
        context.setdefault("history", list(self._history))
        context.setdefault("state", dict(self._last_state) if self.state_block else {})
        context.setdefault("on_log", self._record_backend_log)
        try:  # odometry only - ground-truth target geometry never enters prompts
            context.setdefault(
                "position_flu",
                tuple(float(v) for v in list(self.world.position_flu)[:3]),
            )
            context.setdefault("yaw_rad", float(self.world.yaw_rad))
        except Exception:
            pass
        return context

    # ------------------------------------------------------------------
    # context assembly
    # ------------------------------------------------------------------

    def _build_context(
        self, turn: int, events: list[dict], retry_note: str | None = None
    ) -> tuple[list, str]:
        frame = None
        try:
            frame = self.world.render()
        except Exception:
            frame = None
        current_id = self.ctx.next_obs_id()
        self.ctx.extra["current_observation"] = current_id
        if frame is not None:
            self._cache_image(current_id, frame)
            self.ctx.extra["images"][current_id] = frame
            # the frame the model is about to be shown IS obs_N of the store, so
            # the id in the prompt is the id every tool can resolve
            self._register_frame(
                current_id, frame, turn=turn, note=f"current view of turn {turn}"
            )
        # the STATE block the model reads this turn (also exposed as context["state"])
        self._last_state = self.state(turn)
        frames, sent_ids = self._choose_images(current_id, frame is not None)
        user = _truncate(
            self._user_prompt(turn, sent_ids, frame is not None, events, retry_note),
            self.max_text_chars,
        )
        return frames, user

    def _system_prompt(self, include_tools: bool) -> str:
        """Role + rules, budgeted **catalogue-first** (P4).

        The catalogue is the one section a model cannot reconstruct from the
        instruction: it is rendered before anything else and truncating it is
        the *last* resort (only an explicit ``max_system_chars`` narrower than
        the catalogue ever gets there). The protocol doc and the examples are
        cut first and say so, so a default-budget prompt carries the whole
        state-aware listing instead of a catalogue cut mid-entry.
        """
        template = self.system_prompt_template
        budget = max(1, int(self.max_system_chars))
        if include_tools:
            catalogue = self.tools_text()
            catalogue_state = self._last_catalogue
        else:
            catalogue = ""
            catalogue_state = getattr(self, "_last_catalogue", {}) or {}
        doc = self.tool_call_doc if include_tools else ""
        doc_full = len(doc)

        def render(doc_text: str, tools_text: str) -> str:
            text = template.replace("{tool_doc}", doc_text).replace("{tools}", tools_text)
            if include_tools:
                if "{tool_doc}" not in template:
                    text += "\n\n" + doc_text
                if tools_text and "{tools}" not in template:
                    text += "\n\nAvailable tools:\n" + tools_text
            return text

        text = render(doc, catalogue)
        doc_truncated = False
        catalogue_truncated = bool(catalogue_state.get("truncated"))
        if include_tools and len(text) > budget:
            # the doc/examples are the flexible section: shrink them by the
            # overage (plus the note) and keep the catalogue untouched
            over = len(text) - budget
            note = (
                "\n\n[... protocol/example doc truncated: the tool catalogue has "
                "priority in this prompt ...]"
            )
            doc = _truncate(doc, max(0, len(doc) - over - len(note)), note=note)
            doc_truncated = True
            text = render(doc, catalogue)
        if len(text) > budget:
            # the explicit cap is narrower than the catalogue itself: the
            # catalogue is truncated last (the default budget never gets here)
            catalogue_truncated = True
            text = _truncate(text, budget)
        self.prompt_budget = {
            "system_chars": budget,
            "rendered_chars": len(text),
            "template_chars": len(template),
            "doc_chars": len(doc),
            "doc_full_chars": doc_full,
            "doc_truncated": bool(doc_truncated),
            "catalogue_chars": int(catalogue_state.get("chars") or 0),
            "catalogue_raw_chars": int(catalogue_state.get("raw_chars") or 0),
            "catalogue_complete": not catalogue_truncated,
            "catalogue_truncated": bool(catalogue_truncated),
            "catalogue_in_system": bool(include_tools),
        }
        return text

    def _user_prompt(
        self,
        turn: int,
        sent_ids: list[str],
        has_frame: bool,
        events: list[dict],
        retry_note: str | None,
    ) -> str:
        # sections are ordered by importance and individually capped so that one
        # huge section cannot push the images line or the protocol note out of
        # the total text budget
        limit = max(1, self.max_text_chars)
        instruction = _truncate(self.instruction, max(120, int(0.35 * limit)))
        state_block = _truncate(
            "\n".join(f"  {line}" for line in self._state_lines()),
            max(80, int(0.35 * limit)),
        )
        agent_state = _truncate(
            "\n".join(
                f"  {label}={value}"
                for label, value in (self._last_state or self.state(turn)).items()
            ),
            max(120, int(0.22 * limit)),
            note="",
        )
        sections = [
            f"Episode {self.ctx.episode_id} | turn {turn}/{self.max_turns} | "
            f"tool_calls_left={int(self.ctx.budget.get('tool_calls_left', 0))}",
            f"Instruction: {instruction}",
            self._images_line(sent_ids, has_frame),
        ]
        if self.state_block:
            sections.append(f"STATE (fixed labels):\n{agent_state}")
        sections.append(f"Structured state:\n{state_block}")
        if (
            self.subtask_executor
            and not self._plan_revision_note
            and self._plan_record() is not None
        ):
            sections.append(PLAN_CALL_HINT)
        if self.subtask_executor and self._plan_revision_note:
            sections.append(self._plan_revision_note)
        if events:
            rendered = []
            for event in events[:4]:
                data = json.dumps(event.get("data", {}), default=str)
                rendered.append(f"{event.get('name')}{_truncate(data, 120)}")
            sections.append(
                "Events since last turn: "
                + _truncate("; ".join(rendered), max(60, int(0.25 * limit)))
            )
        if self._feedback and self.feedback_in_prompt:
            feedback = _truncate(
                "\n".join(f"  {line}" for line in self._feedback),
                max(80, int(0.35 * limit)),
                note="",
            )
            sections.append(f"Recent tool results (oldest first):\n{feedback}")
        if self.subtask_executor and self._executor_notes:
            executor = _truncate(
                "\n".join(f"  - {line}" for line in self._executor_notes),
                max(80, int(0.3 * limit)),
                note="",
            )
            sections.append(
                "EXECUTOR (plan bookkeeping done for you - do not spend a call on "
                f"it):\n{executor}"
            )
        if self._recovery_notes and self.recovery_enabled:
            recovery = _truncate(
                "\n".join(f"  - {line}" for line in self._recovery_notes),
                max(120, int(0.3 * limit)),
                note="",
            )
            sections.append(
                f"RECOVERY (no new evidence in the last "
                f"{self._stagnant_turns} turns - do this instead of repeating):\n{recovery}"
            )
        if self._final_note:
            sections.append(self._final_note)
        if retry_note:
            sections.append(retry_note)
        sections.append(
            "Reply with exactly one JSON object, e.g. "
            f"{json.dumps(PROMPT_EXAMPLES[0])}."
        )
        return _truncate("\n".join(sections), limit)

    def _images_line(self, sent_ids: list[str], has_frame: bool) -> str:
        if not sent_ids:
            return "Images: none available (renderer returned no frame)."
        if len(sent_ids) == 1:
            prefix = "image 1 = current view" if has_frame else "image 1 = last cached observation"
            return f"Images: {prefix} ({sent_ids[0]})."
        head = "image 1 = current view" if has_frame else "image 1 = last cached observation"
        rest = ", ".join(sent_ids[1:])
        return (
            f"Images: {head} ({sent_ids[0]}); images 2-{len(sent_ids)} are earlier "
            f"observations ({rest}, oldest first)."
        )

    def _state_lines(self) -> list[str]:
        lines: list[str] = []
        pos = getattr(self.world, "position_flu", None)
        if pos is not None:
            try:
                p = [float(v) for v in list(pos)[:3]]
                lines.append(f"pose_flu=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) m")
            except Exception:
                pass
        yaw = getattr(self.world, "yaw_rad", None)
        if yaw is not None:
            try:
                lines.append(f"heading_deg={math.degrees(float(yaw)) % 360.0:.0f}")
            except Exception:
                pass
        trajectory = getattr(self.world, "trajectory", None)
        if trajectory is not None:
            try:
                points = [tuple(float(v) for v in list(pt)[:3]) for pt in trajectory]
                if len(points) > 1:
                    length = sum(
                        math.dist(a, b) for a, b in zip(points, points[1:])
                    )
                    lines.append(f"path_length_m={length:.2f}")
            except Exception:
                pass
        lines.append(f"current_observation={self.ctx.extra.get('current_observation')}")
        store = observation_store(self.ctx, create=False)
        if store is not None and len(store):
            # the observation ids that exist right now: the model references
            # them, so it should see them (invented ids are the common mistake)
            newest = store.ids()[:4]
            lines.append(
                f"observations_available={len(store)} newest={','.join(newest)}"
            )
        tasks = self._collect_summary(
            (self.world, self.memory), ("active_tasks", "task_summary")
        )
        for line in tasks:
            lines.append(f"active_task {line}")
        hypotheses = self._collect_summary(
            (self.memory, self.world),
            ("context_summary", "hypotheses", "active_hypotheses", "summarize"),
        )
        for line in hypotheses:
            lines.append(f"memory {line}")
        return lines

    def _collect_summary(self, sources, names) -> list[str]:
        for source in sources:
            if source is None:
                continue
            for name in names:
                fn = getattr(source, name, None)
                if not callable(fn):
                    continue
                value = None
                try:
                    value = fn()
                except TypeError:
                    try:
                        value = fn(max_items=self.max_memory_items)
                    except Exception:
                        value = None
                except Exception:
                    value = None
                if value is None:
                    continue
                lines = [
                    _truncate(line, 200) for line in _as_lines(value)
                ][: self.max_memory_items]
                if lines:
                    return lines
        return []

    # ------------------------------------------------------------------
    # the fixed STATE block (every turn, every label)
    # ------------------------------------------------------------------

    def state(self, turn: int | None = None) -> dict:
        """The agent's own state, one string per label of :data:`STATE_LABELS`.

        Rendered into every turn prompt and exposed to tool-aware backends as
        ``context["state"]``; readable outside the loop (MCP/harness/tests) as
        ``runtime.state()``. A field the episode does not know yet says ``n/a``
        rather than being dropped, so the block always has the same shape.
        """
        plan = self._plan_record()
        subgoal = self._subgoal_text(plan)
        coverage = self._coverage or _coverage_of_extra(self.ctx)
        failures = "; ".join(self._recent_failures) if self._recent_failures else "none"
        if turn is not None:
            turns_left = max(0, self.max_turns - turn + 1)
            budget = (
                f"{self._budget_left()} tool calls, "
                f"{turns_left} turns left"
            )
        else:
            budget = f"{self._budget_left()} tool calls"
        return {
            "subgoal": subgoal,
            "candidate": self._candidate_text(),
            "evidence": (
                ", ".join(self._recent_evidence[-3:][::-1]) if self._recent_evidence else "none yet"
            ),
            "open_questions": self._open_questions_text(plan),
            "searched_coverage": _coverage_text(coverage),
            "recent_failures": failures,
            "next_info_needed": self._next_info_text(),
            "budget_left": budget,
            "plan_progress": self._plan_progress_text(),
            "subtasks": self._subtask_text(),
        }

    def _next_info_text(self) -> str:
        """What the agent needs next: the recovery step when one is active, else
        the last tool's own next step, else what an honest close would say."""
        if self._recovery_notes:
            return _truncate(" ".join(self._recovery_notes), MAX_STATE_CHARS)
        if self._next_info_needed:
            return _truncate(self._next_info_needed, MAX_STATE_CHARS)
        if self._final_note:
            return _truncate(self._final_note, MAX_STATE_CHARS)
        return "n/a"

    def _plan_record(self) -> dict | None:
        """The episode's plan record (the plan tools own it), or None."""
        try:
            from .tool_impl_task import plan_state
        except Exception:  # pragma: no cover - the module is part of this package
            return None
        try:
            record = plan_state(self.ctx)
        except Exception:  # noqa: BLE001 - a foreign context may not fit
            return None
        return record if isinstance(record, dict) else None

    def _plan_progress_text(self) -> str:
        """``plan_1 rev=2 step 2/4 (1 done): ... | done_when: ...`` or ``no plan``."""
        try:
            from .tool_impl_task import plan_progress
        except Exception:  # pragma: no cover
            return "no plan"
        try:
            return str(plan_progress(self.ctx))
        except Exception:  # noqa: BLE001
            return "no plan"

    def _subtask_text(self) -> str:
        """The STATE block's ``subtasks`` line (P4).

        Which sub-task the plan is on, how much of its budget is consumed, and
        whether its failure branch is available - plus, when the model still owes
        a plan revision, the decision node that asked for it.
        """
        try:
            from .tool_impl_task import subtask_progress
        except Exception:  # pragma: no cover
            return "no plan"
        note = (
            f"revision_required={self._revision_node or 'yes'}"
            if self._plan_revision_note
            else ""
        )
        try:
            text = str(subtask_progress(self.ctx, note=note))
        except Exception:  # noqa: BLE001
            return "no plan"
        return _truncate(text, MAX_STATE_CHARS)

    # ------------------------------------------------------------------
    # the sub-task executor and the decision nodes (P4)
    # ------------------------------------------------------------------

    def _resume_plan(self, log: EpisodeLog) -> None:
        """Adopt the plan an interrupted episode left in this context (P4).

        Only a caller-provided context can carry one: a context this runtime
        created is empty. The adopted statuses, evidence and consumed budget are
        what let a second run continue the first one instead of starting over.
        The adoption is reported on the log (``plan_resumed``) and as an event.
        """
        if not self.plan_resume or self.ctx is None:
            return
        try:
            from .tool_impl_task import resume_plan
        except Exception:  # pragma: no cover
            return
        try:
            record = resume_plan(self.ctx, self.ctx.episode_id, self.instruction)
        except Exception:  # noqa: BLE001 - a broken plan record must not stop the loop
            return
        if not isinstance(record, dict) or not record.get("resumed_from"):
            return
        subtasks = [subtask for subtask in record.get("subtasks") or [] if isinstance(subtask, dict)]
        data = {
            "plan_id": record.get("plan_id"),
            "resumed_from": record.get("resumed_from"),
            "resume_count": int(record.get("resume_count", 0)),
            "current_step": int(record.get("current_step") or 0),
            "n_steps": len(subtasks),
            "statuses": {
                str(subtask.get("status")): sum(
                    1
                    for other in subtasks
                    if other.get("status") == subtask.get("status")
                )
                for subtask in subtasks
            },
        }
        log.plan_resumed = data
        self.emit_event("plan_resumed", dict(data))

    def _candidate_ids_snapshot(self) -> tuple[str, ...]:
        """The grounded candidate ids right now (the new-candidate test)."""
        store = self._ctx_extra().get(CANDIDATE_STORE_KEY)
        if not isinstance(store, dict):
            return ()
        return tuple(str(key) for key in store)

    def _executor_step(
        self,
        turn: int,
        log: EpisodeLog,
        record: TurnRecord,
        candidates_before: tuple[str, ...] = (),
    ) -> dict:
        """Let the executor advance the plan from the call that just finished (P4).

        The executor is what makes plan bookkeeping free: it marks the sub-task
        the call served done/blocked/failed and activates the next one, so a pure
        status update never costs the model a turn. On top of it the runtime
        classifies the turn: a *routine* turn is absorbed here, a *decision node*
        (a new candidate, a lost goal, an identity conflict, a failed path, a
        sub-task switch) is recorded and - when the plan itself is what has to
        change - turned into a plan-revision request for the next prompt.
        """
        outcome: dict = {"changed": False, "transitions": [], "decision": ""}
        self._executor_notes = []
        if not self.subtask_executor or self.ctx is None:
            return outcome
        try:
            from .tool_impl_task import SUBTASK_BLOCKED, SUBTASK_FAILED, executor_step
        except Exception:  # pragma: no cover
            return outcome
        event = {
            "tool": record.tool,
            "args": dict(record.args or {}),
            "status": record.status,
            "data": dict(record.data) if isinstance(record.data, dict) else {},
            "evidence": list(record.evidence or ()),
            "progress_kind": record.progress_kind,
            "turn": turn,
        }
        try:
            outcome = executor_step(self.ctx, event)
        except Exception:  # noqa: BLE001 - the executor never breaks the loop
            outcome = {"changed": False, "transitions": [], "decision": ""}
        if not isinstance(outcome, dict):
            outcome = {"changed": False, "transitions": [], "decision": ""}
        transitions = [
            dict(entry) for entry in outcome.get("transitions") or [] if isinstance(entry, dict)
        ]
        for entry in transitions:
            log.plan_transitions.append(entry)
        if transitions:
            self._executor_notes = [
                f"{entry.get('id')} {entry.get('from')}->{entry.get('to')}"
                f" ({entry.get('reason')})"
                for entry in transitions
            ]
        # the model revised the plan itself: the revision request is answered
        if record.tool == "update_plan" and record.status == STATUS_OK:
            self._plan_revision_note = ""
            self._revision_node = ""
            self._revision_subtask = ""
        elif self._plan_revision_note and self._revision_subtask:
            # the sub-task the request was about left its blocked/failed state on
            # its own (a later result satisfied it): the request is stale
            for entry in transitions:
                if str(entry.get("id")) != self._revision_subtask:
                    continue
                if str(entry.get("to")) not in (SUBTASK_BLOCKED, SUBTASK_FAILED):
                    self._plan_revision_note = ""
                    self._revision_node = ""
                    self._revision_subtask = ""
                    break
        node = self._decision_node(record, outcome, candidates_before)
        if node:
            record.decision = node
            log.decision_nodes.append(
                {
                    "turn": turn,
                    "node": node,
                    "subtask": str(outcome.get("active") or outcome.get("charged") or ""),
                }
            )
            if node in PLAN_REVISION_NODES:
                self._plan_revision_note = self._plan_revision_text(node, record, outcome)
                self._revision_node = node
        return outcome

    def _decision_node(
        self,
        record: TurnRecord,
        outcome: dict,
        candidates_before: tuple[str, ...] = (),
    ) -> str:
        """The decision node of one turn, or ``""`` for a routine turn.

        Routine turns (a new view, a query answer, a bookkeeping step the
        executor absorbed) are not returned: they cost the plan nothing. The
        nodes are the moments the plan may be wrong - a new candidate, the goal
        lost, conflicting evidence, a path that failed, and the sub-task switch
        the executor performed on its own.
        """
        if str(outcome.get("decision") or "") == DECISION_SUBTASK_SWITCH:
            return DECISION_SUBTASK_SWITCH
        data = record.data if isinstance(record.data, dict) else {}
        token = _norm_token(
            data.get("blocked_reason") or data.get("reason") or data.get("outcome")
            or data.get("error")
        )
        if token in GOAL_LOST_TOKENS:
            return DECISION_GOAL_LOST
        if self._candidates_lost(candidates_before):
            return DECISION_GOAL_LOST
        if data.get("conflicts") or _norm_token(data.get("verdict")) in CONFLICT_TOKENS:
            return DECISION_EVIDENCE_CONFLICT
        if str(outcome.get("blocked") or ""):
            return DECISION_PATH_FAILURE
        if token in BAD_OUTCOMES or record.status in (STATUS_BLOCKED, STATUS_FAILED):
            return DECISION_PATH_FAILURE
        if self._new_candidate(candidates_before):
            return DECISION_NEW_CANDIDATE
        return ""

    # ------------------------------------------------------------------
    # deterministic continuation: routine steps without a model turn
    # ------------------------------------------------------------------

    def _run_executor_auto_steps(
        self, turn: int, log: EpisodeLog, record: TurnRecord
    ) -> TurnRecord:
        """Run the plan's deterministic steps after a routine model turn.

        Only a *routine* turn (``record.decision == ""``) of a model call
        enters the chain: a decision node is exactly the moment the model has
        to look at the plan again. Each step goes through the registry like a
        model call (budget, cache, audit log) and is logged with ``auto=True``;
        the chain stops at ``max_executor_auto_steps`` (3 by default), at a
        decision node, at a cached answer, or when the plan names no further
        deterministic step. Returns the last executed record.
        """
        if (
            not self.executor_auto_steps
            or self.max_executor_auto_steps <= 0
            or self.ctx is None
            or record.decision
            or record.auto
        ):
            return record
        executed = 0
        while executed < self.max_executor_auto_steps:
            if self.abort_requested:
                break
            if int(self.ctx.budget.get("tool_calls_left", 0)) <= 0:
                break
            call = self._deterministic_next_call(record)
            if call is None:
                break
            name, args, origin = call
            spent_subtask = (
                self._current_subtask_info()
                if origin == "subtask_next_step"
                else None
            )
            pose_before = self._pose_snapshot()
            candidates_before = self._candidate_ids_snapshot()
            t0 = time.time()
            result = self._call_tool(name, args)
            ms = round((time.time() - t0) * 1000.0, 1)
            step = self._turn_record(turn, name, args, result, ms, 0, AUTO_STEP_REPLY)
            step.auto = True
            step.auto_origin = origin
            log.turns.append(step)
            self._note_result(name, args, result, step)
            self._score_turn(step, log, pose_before)
            self._executor_step(turn, log, step, candidates_before)
            log.executor_auto_steps.append(
                {
                    "turn": turn,
                    "tool": name,
                    "args": dict(args),
                    "status": result.status,
                    "origin": origin,
                    "decision": step.decision,
                }
            )
            executed += 1
            if origin in ("plan_next_step", "subtask_next_step"):
                # a static next_step is a step, not a loop: consume it so the
                # next routine turn does not run the same call again
                from .tool_impl_task import mark_next_step_spent

                mark_next_step_spent(self.ctx, origin, spent_subtask)
            record = step
            if step.decision or result.status == STATUS_CACHED:
                break
        return record

    def _deterministic_next_call(
        self, record: TurnRecord
    ) -> tuple[str, dict, str] | None:
        """The plan's next step when it is a concrete call, else None.

        Sources, in priority order (all require an *active* sub-task, so a
        blocked/failed plan is always left to the model):

        1. an explicit next step on the plan record or the active sub-task
           (``next_step`` / ``next_action`` / ``next_tool`` / ``next_call``);
        2. the last result's own ``next_action`` / structured ``next_step``
           (the tool naming the call that continues it);
        3. a concrete failure-branch call - only after a fruitless result, so
           the branch is really the plan's answer to a step that did nothing;
        4. a sub-task goal written as exactly one tool call.

        A call is only executable when its tool is registered. A continuation
        named by a *productive* turn may repeat the previous call (that is how a
        search routine or a partial leg continues); after an empty/failed/blocked
        result the same call is never executed again - that self-reference is
        the reported ``search_target`` loop. A goal/branch-derived step may not
        repeat the previous call at all.
        """
        if self.registry is None or self.ctx is None:
            return None
        plan = self._plan_record()
        if plan is None:
            return None
        current = self._current_subtask_info()
        if not isinstance(current, dict):
            return None
        try:
            from .tool_impl_task import SUBTASK_ACTIVE
        except Exception:  # pragma: no cover - the module is part of this package
            SUBTASK_ACTIVE = "active"
        if str(current.get("status") or "") != SUBTASK_ACTIVE:
            return None
        names = set(self._tool_names())
        previous = (record.tool, dict(record.args or {}))
        # a fruitless turn cannot name itself as its own continuation: empty and
        # failed results repeatedly pointed at the call that had just run
        # (search_target), and the executor flew it again inside the same turn.
        # A successful continuation (a partial leg / a search routine that made
        # progress) may still name the same call.
        fruitless = _fruitless_str(record.status) or record.progress_kind == PROGRESS_NONE

        def accept(
            value: Any, origin: str, *, allow_same: bool
        ) -> tuple[str, dict, str] | None:
            call = _call_from_value(value)
            if call is None:
                return None
            name, args = call
            if name not in names or name in AUTO_STEP_EXCLUDED_TOOLS:
                return None
            if not allow_same and (
                (name, args) == previous
                or (
                    name == previous[0]
                    and _call_args_key(args) == _call_args_key(previous[1])
                )
            ):
                return None
            return name, args, origin

        for holder, origin in (
            (plan, "plan_next_step"),
            (current, "subtask_next_step"),
        ):
            if origin == "plan_next_step" and plan.get("_next_step_spent"):
                continue
            if origin == "subtask_next_step" and current.get("_next_step_spent"):
                continue
            for key in ("next_step", "next_action", "next_tool", "next_call"):
                call = accept(holder.get(key), origin, allow_same=not fruitless)
                if call is not None:
                    return call

        data = record.data if isinstance(record.data, dict) else {}
        for key, origin in (
            ("next_action", "next_action"),
            ("next_step", "result_next_step"),
        ):
            call = accept(data.get(key), origin, allow_same=not fruitless)
            if call is not None:
                return call

        if fruitless:
            for item in current.get("failure_branch") or ():
                call = accept(item, "failure_branch", allow_same=False)
                if call is not None:
                    return call

        return accept(current.get("goal"), "subtask_goal", allow_same=False)

    def _new_candidate(self, candidates_before: tuple[str, ...]) -> bool:
        """True when the last call grounded a candidate that was not known before."""
        after = self._candidate_ids_snapshot()
        return bool(set(after) - set(candidates_before))

    def _candidates_lost(self, candidates_before: tuple[str, ...]) -> bool:
        """True when the known candidates disappeared (the tracked goal was lost)."""
        return bool(candidates_before) and not self._candidate_ids_snapshot()

    def _plan_revision_text(
        self, node: str, record: TurnRecord, outcome: dict
    ) -> str:
        """The plan-revision request injected after a decision node."""
        current = self._current_subtask_info()
        if current is None:
            return ""  # no plan: there is nothing to revise
        blocked_reason = str(outcome.get("blocked_reason") or "")
        reason = blocked_reason or _norm_token(
            (record.data or {}).get("reason") if isinstance(record.data, dict) else ""
        )
        label = f"sub-task {current.get('id')} '{current.get('goal')}'"
        if blocked_reason:
            head = (
                f"PLAN REVISION NEEDED (decision node: {node}): the plan's current "
                f"{label} is blocked ({reason})."
            )
        else:
            head = (
                f"PLAN REVISION NEEDED (decision node: {node}): the plan's current "
                f"{label} ({current.get('status')}) may no longer serve the goal"
                + (f" ({reason})" if reason else "")
                + "."
            )
        lines = [head]
        branch = [str(item) for item in current.get("failure_branch") or []]
        if branch:
            lines.append(
                "Its failure branch is the way out: " + "; ".join(branch) + "."
            )
        lines.append(
            "Write its next_step and every failure_branch entry as ONE concrete "
            'call, e.g. {"tool": "go_to_viewpoint", "args": {"viewpoint_id": '
            '"vp_9"}}, so the executor can run it without a model turn; prose is '
            "not executed."
        )
        lines.append(
            "Revise that sub-plan before continuing: update_plan(patch=[{\"id\": \""
            f"{current.get('id')}\", ...}}]) to change it in place, or replace the "
            "steps with a plan that can still reach the goal."
        )
        self._revision_subtask = str(current.get("id") or "")
        return _truncate(" ".join(lines), 640)

    def _current_subtask_info(self) -> dict | None:
        """The sub-task the plan is on (the executor's own view of it)."""
        try:
            from .tool_impl_task import current_subtask
        except Exception:  # pragma: no cover
            return None
        try:
            subtask = current_subtask(self.ctx)
        except Exception:  # noqa: BLE001
            return None
        return subtask if isinstance(subtask, dict) else None

    def _failure_branch_advice(self) -> list[str]:
        """The plan's own next step for a failed sub-task (P4).

        The failure branch is what the model wrote down *before* it got stuck, so
        it leads the recovery advice - quoted as the plan's branch, never as an
        invented tool call.
        """
        if not self.subtask_executor:
            return []
        try:
            from .tool_impl_task import subtask_failure_branch
        except Exception:  # pragma: no cover
            return []
        try:
            info = subtask_failure_branch(self.ctx)
        except Exception:  # noqa: BLE001
            return []
        if not isinstance(info, dict):
            return []
        branch = [str(item) for item in info.get("branch") or [] if str(item).strip()]
        if not branch:
            return []
        head = (
            f"the plan's current sub-task {info.get('id')} '{info.get('goal')}' is "
            f"{info.get('status')}"
        )
        if info.get("reason"):
            head += f" ({info['reason']})"
        head += ": its failure branch is the way out"
        return [head] + [f"failure branch: {item}" for item in branch]

    def _ctx_extra(self) -> dict:
        """``ctx.extra`` repaired into a dict (state rendering never crashes)."""
        extra = getattr(self.ctx, "extra", None)
        return extra if isinstance(extra, dict) else {}

    def _budget_left(self) -> int:
        budget = getattr(self.ctx, "budget", None)
        if not isinstance(budget, dict):
            return 0
        try:
            return int(budget.get("tool_calls_left", 0))
        except (TypeError, ValueError):
            return 0

    def _subgoal_text(self, plan: dict | None) -> str:
        """What the agent is doing right now: the plan's current step, or the
        harness-provided ``ctx.extra["subgoal"]``."""
        declared = self._ctx_extra().get("subgoal")
        if declared:
            return _truncate(str(declared), MAX_STATE_CHARS)
        if not plan:
            return "n/a"
        steps = [str(step) for step in plan.get("steps") or []]
        current = int(plan.get("current_step") or 0)
        if 1 <= current <= len(steps):
            return _truncate(steps[current - 1], MAX_STATE_CHARS)
        return "n/a"

    def _candidate_text(self) -> str:
        """The newest grounded candidate and how much of it is known.

        The line leads with the id *exactly as the movement tools take it*
        (``cand_N``): rendering it glued to the frame it was grounded in
        (``cand_1@obs_2``) handed the model a spelling no tool resolves, and a
        copy of it is an ``unknown_candidate_id`` turn (G1 §5.7). The frame is
        reported as its own clause, so the id stays copyable.
        """
        store = self._ctx_extra().get(CANDIDATE_STORE_KEY)
        if not isinstance(store, dict) or not store:
            return "none found yet"
        latest = next(reversed(store.values()))
        if not isinstance(latest, dict):
            return "none found yet"
        candidate_id = str(latest.get("id") or "?")
        observation = latest.get("observation_id")
        confidence = latest.get("confidence")
        known = "position known" if latest.get("position_flu") else "position unknown"
        text = f"{candidate_id}"
        if observation:
            text += f" (seen in {observation})"
        if confidence is not None:
            text += f" conf={confidence}"
        text += f" ({known}, {len(store)} candidate(s))"
        return _truncate(text, MAX_STATE_CHARS)

    def _open_questions_text(self, plan: dict | None) -> str:
        """What is still unanswered - never a claim the target cannot exist."""
        declared = self._ctx_extra().get("open_questions")
        if declared:
            return _truncate(str(declared), MAX_STATE_CHARS)
        last = self._last_turn_record()
        data = last.data if last is not None and isinstance(last.data, dict) else {}
        fruitless = (
            _norm_token(data.get("reason")) == "no_candidate"
            or data.get("n_candidates") == 0
            or (isinstance(data.get("candidates"), list) and "candidates" in data
                and not data["candidates"])
        )
        if fruitless:
            where = str(data.get("observation_id") or "the last view")
            return _truncate(
                f"no candidate matched in {where} yet - that is not evidence of "
                "absence: which viewpoint or heading still needs a look?",
                MAX_STATE_CHARS,
            )
        if not self._has_candidate():
            return (
                "no grounded candidate yet (the target may be outside every view "
                "taken so far)"
            )
        return "is the confirmed candidate the instructed one? (verify before finish)"

    def _has_candidate(self) -> bool:
        store = self._ctx_extra().get(CANDIDATE_STORE_KEY)
        return isinstance(store, dict) and bool(store)

    def _last_turn_record(self) -> TurnRecord | None:
        return self._last_record

    # ------------------------------------------------------------------
    # images
    # ------------------------------------------------------------------

    def _cache_image(self, observation_id: str, frame: Any) -> None:
        self._image_cache[observation_id] = frame
        self._known_image_ids.add(observation_id)
        while len(self._image_cache) > self.max_cache_images:
            self._image_cache.popitem(last=False)

    def _choose_images(self, current_id: str, has_frame: bool) -> tuple[list, list[str]]:
        ids: list[str] = [current_id] if has_frame else []
        for observation_id in self._pending_image_ids:
            if len(ids) >= self.max_images:
                break
            if observation_id in self._image_cache and observation_id not in ids:
                ids.append(observation_id)
        remaining = min(self.max_images - len(ids), self.max_history_images)
        if remaining > 0:
            history = [obs for obs in self._image_cache if obs not in ids]
            ids.extend(history[-remaining:])
        return [self._image_cache[obs] for obs in ids], ids

    # ------------------------------------------------------------------
    # observation repository (vlm_nav.observations)
    # ------------------------------------------------------------------

    def _begin_observation_store(self, ctx: ToolContext, episode_id: str) -> ObservationStore:
        """Bind the episode's observation store to ``ctx`` (one per episode).

        A caller-provided store is reused but reset: the ids it holds belong to
        the previous episode, while the id counter keeps counting from
        ``ctx.observation_seq`` so ids already published are never reissued. The
        store is created here, before the first turn, so the very first frame the
        model sees is registered in it.
        """
        store = ctx.extra.get(OBSERVATION_STORE_KEY) if isinstance(ctx.extra, dict) else None
        start_seq = int(getattr(ctx, "observation_seq", 0) or 0)
        if isinstance(store, ObservationStore):
            store.reset(episode_id=episode_id, start_seq=start_seq)
        else:
            store = ObservationStore(episode_id=episode_id, start_seq=start_seq)
            ctx.extra[OBSERVATION_STORE_KEY] = store
        # frames published during an earlier episode are not observations of this
        # one: mark them known so they are neither re-sent nor adopted
        self._known_image_ids.update(published_frames(ctx))
        return store

    def observation_store(self) -> ObservationStore | None:
        """The episode's observation store (None before the first turn)."""
        return observation_store(self.ctx, create=False) if self.ctx is not None else None

    def _register_frame(
        self,
        observation_id: str,
        frame: Any,
        *,
        turn: int,
        kind: str = KIND_EVIDENCE,
        note: str = "",
        pose_flu: Any = None,
        yaw_deg: Any = None,
        parent_id: str | None = None,
    ) -> None:
        """Register one model-visible frame with its capture provenance (best
        effort: an unregisterable frame must never break the turn)."""
        try:
            register_observation(
                self.ctx,
                frame,
                kind=kind,
                step=turn,
                timestamp=time.time(),
                obs_id=observation_id,
                pose_flu=pose_flu,
                yaw_deg=yaw_deg,
                parent_id=parent_id,
                note=note,
            )
        except Exception:  # noqa: BLE001 - the loop outranks the bookkeeping
            pass

    def _adopt_published_frame(self, observation_id: str, turn: int) -> None:
        """Register a frame a tool published under its own ``obs_N`` id.

        The tool that published it (the navigation tools publish the frame of
        every observation they reference) also wrote the pose it was captured
        at, which is read back here - a published frame without a known pose is
        registered as pose-unknown rather than guessed at. Without this the id
        would exist only as pixels in ``ctx.extra["images"]``, which is the
        state the "unknown observation" failure came from.
        """
        store = observation_store(self.ctx, create=False)
        if store is None or store.get(observation_id) is not None:
            return
        frame = (self.ctx.extra.get("images") or {}).get(observation_id)
        position, yaw_rad = lookup_pose(self.ctx, observation_id)
        state = self.ctx.extra.get(OBSERVATION_STATE_KEY)
        label = state.get(observation_id) if isinstance(state, dict) else None
        label = label.get("kind") if isinstance(label, dict) else None
        self._register_frame(
            observation_id,
            frame,
            turn=turn,
            kind=kind_for_label(label),
            note=f"frame published by a tool ({label or 'tool'})",
            pose_flu=position,
            yaw_deg=None if yaw_rad is None else math.degrees(yaw_rad),
        )

    # ------------------------------------------------------------------
    # tool execution
    # ------------------------------------------------------------------

    def _call_tool(self, name: str, args: dict) -> ToolResult:
        try:
            return self.registry.call(name, args, self.ctx)
        except KeyError:
            available = list(self.registry.names())[:12]
            return ToolResult(
                STATUS_FAILED,
                {"reason": "unknown_tool", "tool": name, "available_tools": available},
            )
        except Exception as exc:
            return ToolResult(
                STATUS_FAILED, {"reason": "runtime_error", "error": f"{type(exc).__name__}: {exc}"}
            )

    def _finish_result(self, args: dict) -> ToolResult:
        """A registered ``finish`` tool may veto the termination (e.g. a
        ``verify_goal`` check); without one, finish is accepted as-is. Only an
        explicit ``success`` flag (args or a verifying finish tool) counts the
        episode as successful."""
        names = list(self.registry.names()) if self.registry is not None else []
        if FINISH_TOOL in names:
            return self._call_tool(FINISH_TOOL, args)
        data: dict[str, Any] = {"reason": str(args.get("reason", ""))[:200]}
        if "success" in args:
            data["success"] = bool(args["success"])
        return ToolResult(STATUS_OK, data)

    def _turn_record(
        self,
        turn: int,
        tool: str,
        args: dict,
        result: ToolResult,
        ms: float,
        images_sent: int,
        raw_reply: str,
    ) -> TurnRecord:
        """The raw record of one executed call; ``_score_turn`` fills progress."""
        data = dict(result.data)
        if _fruitless_str(result.status) and _same_call(
            tool, args, data.get("next_action")
        ):
            # a fruitless result naming its own call is not a continuation: keep
            # it out of the prompt/STATE the model copies (the reported loop)
            data.pop("next_action", None)
            prompt = ToolResult(
                result.status, data, result.evidence, result.prompt_text
            ).to_prompt()
        else:
            prompt = result.to_prompt()
        return TurnRecord(
            turn=turn,
            tool=tool,
            args=dict(args),
            status=result.status,
            ms=ms,
            evidence=[e.to_dict() for e in result.evidence],
            images_sent=images_sent,
            data=data,
            raw_reply=raw_reply,
            result_prompt=prompt,
        )

    # ------------------------------------------------------------------
    # turn progress (information / action / task / none)
    # ------------------------------------------------------------------

    def _pose_snapshot(self) -> tuple[tuple[float, float, float] | None, float | None]:
        """(position_flu, yaw_rad) of the world, or (None, None) when unknown."""
        try:
            position = tuple(
                float(v) for v in list(getattr(self.world, "position_flu", None))[:3]
            )
        except (TypeError, ValueError):
            position = None
        try:
            yaw = float(getattr(self.world, "yaw_rad", 0.0))
        except (TypeError, ValueError):
            yaw = None
        return position, yaw

    def _score_turn(
        self,
        record: TurnRecord,
        log: EpisodeLog,
        pose_before: tuple[tuple[float, float, float] | None, float | None],
    ) -> None:
        """Classify what one turn changed and keep the episode's bookkeeping.

        Runs *after* ``_note_result`` so the frames a tool published are already
        in the observation store: the repeat test needs their pose, heading and
        content, not their id.
        """
        record.progress_kind = self._progress_kind(record, pose_before)
        record.new_evidence = record.progress_kind != PROGRESS_NONE
        self._last_record = record
        if record.new_evidence:
            self._stagnant_turns = 0
            self._recovery_notes = []
            self._final_note = ""
            self._next_info_needed = ""
        else:
            self._stagnant_turns += 1

        for ref in record.evidence:
            if str(ref.get("kind")) != "observation":
                continue  # the evidence line is about views the model can cite
            token = f"{ref.get('kind')}:{ref.get('ref')}"
            if token not in self._recent_evidence:
                self._recent_evidence.append(token)
        self._recent_evidence = self._recent_evidence[-MAX_RECENT_EVIDENCE:]

        coverage = _coverage_of(record.data)
        if coverage is not None:
            self._coverage = coverage

        outcome = _outcome_text(record)
        if outcome:
            self._recent_failures.append(
                _truncate(f"t{record.turn} {record.tool} {record.status}({outcome})", 80)
            )
            self._recent_failures = self._recent_failures[-MAX_RECENT_FAILURES:]
        self._remember_next_info(record)

    def _progress_kind(
        self,
        record: TurnRecord,
        pose_before: tuple[tuple[float, float, float] | None, float | None],
    ) -> str:
        """``information`` | ``action`` | ``task`` | ``none`` for one turn.

        A tool may declare what its own answer means (``data["progress_kind"]``);
        otherwise the rules are, in order: a replay/failure is ``none``, a task
        signal (a verified goal, a confirmed identity, a completed subgoal) is
        ``task``, a moved drone (or a reported leg/arrival) is ``action``, and a
        genuinely new view or query answer is ``information``. New means the
        (pose, heading, content) of the view and the query asked about it were
        not seen before - a fresh ``obs_N`` of the same view is not progress.
        """
        declared = declared_progress_kind(record.data)
        if declared is not None:
            return declared
        if record.tool == FINISH_TOOL and record.status == STATUS_OK:
            return PROGRESS_TASK  # the model is closing the episode
        if record.status in (STATUS_CACHED, STATUS_FAILED, STATUS_BLOCKED):
            return PROGRESS_NONE
        if _task_signal(record.data):
            return PROGRESS_TASK
        if self._moved_since(pose_before, record.data):
            return PROGRESS_ACTION
        if self._new_information(record.tool, record.args, record):
            return PROGRESS_INFORMATION
        return PROGRESS_NONE

    def _moved_since(
        self,
        pose_before: tuple[tuple[float, float, float] | None, float | None],
        data: dict,
    ) -> bool:
        """True when the call moved the drone or reported a flown leg."""
        position, _yaw = self._pose_snapshot()
        start = pose_before[0] if pose_before else None
        if position is not None and start is not None:
            distance = math.dist(position, start)
            if distance >= MOVE_EPS_M:
                return True
        if data.get("moved") is True or data.get("reached") is True:
            return True
        progress_m = _as_float_or_none(data.get("progress_m"))
        if progress_m is not None and progress_m >= MOVE_EPS_M:
            return True
        for key in ("moved_m", "segment_progress_m"):
            value = _as_float_or_none(data.get(key))
            if value is not None and value >= MOVE_EPS_M:
                return True
        return _norm_token(data.get("outcome")) in ("arrived", "partial")

    def _new_information(self, tool: str, args: dict, record: TurnRecord) -> bool:
        """New = an unseen (view, query) pair, or - for answers without a view -
        an unseen evidence reference or an unseen payload."""
        views = self._view_keys(record)
        if views:
            # JSON-render the view keys: they may carry None (unknown pose) and
            # must stay comparable/serializable as the key of a seen query
            key = (
                self._query_intent(tool, args),
                tuple(sorted(json.dumps(view, default=str) for view in views)),
            )
            if key in self._seen_view_queries:
                return False
            self._seen_view_queries[key] = None
            while len(self._seen_view_queries) > MAX_SEEN_VIEWS:
                self._seen_view_queries.popitem(last=False)
            return True
        return self._unseen_answer(tool, record)

    def _view_keys(self, record: TurnRecord) -> list[tuple]:
        """The (pose, heading, content) keys of the observations a turn returned."""
        store = self.observation_store()
        if store is None:
            return []
        keys: list[tuple] = []
        for observation_id in _observation_ids(record):
            observation = store.get(observation_id)
            if observation is None:
                continue
            yaw = observation.yaw_deg if observation.has_yaw else None
            key = (
                tuple(round(float(v), VIEW_POSE_ROUND) for v in observation.pose_flu)
                if observation.has_pose
                else None,
                None if yaw is None else round(float(yaw), VIEW_YAW_ROUND),
                _frame_digest(observation.frame),
            )
            if key not in keys:
                keys.append(key)
        return keys

    def _query_intent(self, tool: str, args: dict) -> str:
        """The query behind a turn, with observation ids made view-agnostic.

        ``ground_target(obs_5, "the book")`` and ``ground_target(obs_9, "the
        book")`` are the same query when obs_5 and obs_9 show the same view; the
        *view* is what ``_view_keys`` compares, so the ids are elided here.
        """
        text = json.dumps(dict(args or {}), sort_keys=True, default=str)
        return f"{tool}:{_OBS_ID_RE.sub(_VIEW_TOKEN, text)}"

    def _unseen_answer(self, tool: str, record: TurnRecord) -> bool:
        """Fallback for results that carry no view (a plan, a path check, ...):
        an unseen evidence reference or an unseen payload is new information."""
        fresh = False
        for ref in record.evidence:
            key = (str(ref.get("kind")), str(ref.get("ref")))
            if key not in self._seen_evidence:
                self._seen_evidence.add(key)
                fresh = True
        fingerprint = _fingerprint(tool, record.data, record.evidence)
        if fingerprint not in self._seen_fingerprints:
            self._seen_fingerprints.add(fingerprint)
            fresh = True
        return fresh

    def _remember_next_info(self, record: TurnRecord) -> None:
        """What the STATE block reports as ``next_info_needed`` after a turn."""
        if record.recovery == RECOVERY_FINAL:
            self._next_info_needed = FINAL_FINISH_NOTE
            return
        if self._recovery_notes:
            self._next_info_needed = " ".join(self._recovery_notes)
            return
        data = record.data
        next_action = data.get("next_action")
        repeats_itself = (
            _fruitless_str(record.status) or record.progress_kind == PROGRESS_NONE
        ) and _same_call(record.tool, record.args, next_action)
        if (
            isinstance(next_action, dict)
            and next_action.get("tool")
            and not repeats_itself
        ):
            self._next_info_needed = (
                f"{next_action['tool']}({_json_arg(next_action.get('args') or {})})"
            )
            return
        hint = str(data.get("hint") or data.get("next_step") or "")
        self._next_info_needed = _truncate(hint, MAX_STATE_CHARS)

    # ------------------------------------------------------------------
    # stagnation: recover first, stop only when recovery is used up
    # ------------------------------------------------------------------

    def _after_stall(self, turn: int, log: EpisodeLog, record: TurnRecord) -> str:
        """The one place a stall becomes recovery action (both call paths).

        The advice is decided first; only then is the recovery hook given the
        chance to execute a call. A hook that answers nothing leaves the stall
        labelled as advice - the prompt instruction is still the recovery.
        """
        action = self._stall_action(record, log)
        if (
            action == RECOVERY_ADVICE
            and self.recovery_enabled
            and self.recovery_hook is not None
        ):
            if self._run_recovery_hook(turn, log, record):
                self.last_stall_action = RECOVERY_HOOK
                action = RECOVERY_HOOK
        return action

    def _stall_action(self, record: TurnRecord, log: EpisodeLog) -> str:
        """What a no-progress turn means: keep going, recover, or stop.

        Returns ``""`` (continue), ``RECOVERY_ADVICE`` (an instruction follows),
        ``RECOVERY_FINAL`` (one last turn to finish honestly) or
        ``RECOVERY_STOP`` (end with ``no_new_evidence``). With
        ``recovery_enabled=False`` the stagnation limit is the end of the
        episode: no advice, no honest-finish grant, no recovery entry.
        """
        if record.new_evidence:
            self.last_stall_action = ""
            return ""
        if self._stagnant_turns < self.max_stagnant_turns:
            self.last_stall_action = ""
            return ""
        if not self.recovery_enabled:
            self.last_stall_action = RECOVERY_STOP
            return RECOVERY_STOP
        advice = self.recovery_advice(record)
        if advice and self._recovery_attempts < self.max_recovery_attempts:
            self._recovery_attempts += 1
            self._recovery_notes = list(advice)
            self.last_stall_action = RECOVERY_ADVICE
            self._record_recovery(log, record, RECOVERY_ADVICE, advice)
            return RECOVERY_ADVICE
        if advice and not self._finish_granted:
            # recovery cannot help any further: let the model close honestly
            self._finish_granted = True
            self._final_note = FINAL_FINISH_NOTE
            self._recovery_notes = []
            self.last_stall_action = RECOVERY_FINAL
            self._record_recovery(log, record, RECOVERY_FINAL, advice)
            return RECOVERY_FINAL
        self.last_stall_action = RECOVERY_STOP
        return RECOVERY_STOP

    def recovery_advice(self, record: TurnRecord) -> list[str]:
        """The recovery instructions for one stagnating turn (may be empty).

        The plan's own failure branch - what the model wrote down for "if this
        sub-task fails, do that instead" - leads them (P4); the tool-based rules
        follow. A branch is only used when it can be acted on: either the branch
        itself names a registered, evidence-capable tool, or the tool-based advice
        exists anyway (which is what keeps "never ask for a tool the registry does
        not have" true).
        """
        advice = recovery_advice(
            record.tool,
            record.data,
            record.status,
            available=self._tool_names(),
            coverage=self._coverage,
            args=record.args,
        )
        branch = self._failure_branch_advice()
        if not branch:
            return advice
        actionable = bool(_named_tools(self._tool_names(), " ".join(branch)))
        if advice or actionable:
            return branch + advice
        return advice

    def _tool_names(self) -> list[str]:
        if self.registry is None:
            return []
        try:
            return list(self.registry.names())
        except Exception:  # noqa: BLE001 - a duck-typed registry may not answer
            return []

    def _record_recovery(
        self, log: EpisodeLog, record: TurnRecord, kind: str, advice: list[str]
    ) -> None:
        entry = {
            "turn": record.turn,
            "kind": kind,
            "attempt": self._recovery_attempts,
            "stalled_tool": record.tool,
            "stalled_args": dict(record.args),
            "stagnant_turns": self._stagnant_turns,
            "progress_kind": record.progress_kind,
            "advice": list(advice),
        }
        # the stalled turn is marked with what it triggered (the executed
        # recovery *call*, when there is one, is marked separately as "hook")
        record.recovery = RECOVERY_FINAL if kind == RECOVERY_FINAL else RECOVERY_ADVICE
        log.recoveries.append(entry)
        self.emit_event("recovery", entry)

    def _can_run_recovery(self) -> bool:
        """A hook may execute a call only while the budget allows one."""
        return int(self.ctx.budget.get("tool_calls_left", 0)) > 0

    def _run_recovery_hook(
        self, turn: int, log: EpisodeLog, stalled: TurnRecord
    ) -> None:
        """Let the configured hook execute one recovery call (no model turn).

        The hook answers the stall request with ``{"tool": ..., "args": {...}}``
        or None; the call it asks for is executed through the same registry and
        logged as a turn of this episode (``recovery="hook"``), so the model sees
        its result in the next prompt. A hook that fails or asks for nothing
        changes nothing - the advice stays in the prompt.
        """
        request = {
            "turn": turn,
            "stalled_tool": stalled.tool,
            "stalled_args": dict(stalled.args),
            "stalled_status": stalled.status,
            "stagnant_turns": self._stagnant_turns,
            "attempt": self._recovery_attempts,
            "advice": list(self._recovery_notes),
            "coverage": dict(self._coverage),
            "budget_left": int(self.ctx.budget.get("tool_calls_left", 0)),
            "state": dict(self._last_state),
        }
        try:
            call = self.recovery_hook(request)
        except Exception:  # noqa: BLE001 - a broken hook must not break the loop
            call = None
        if not isinstance(call, dict) or not call.get("tool") or not self._can_run_recovery():
            return False
        name, args = str(call["tool"]), dict(call.get("args") or {})
        pose_before = self._pose_snapshot()
        t0 = time.time()
        result = self._call_tool(name, args)
        ms = round((time.time() - t0) * 1000.0, 1)
        record = self._turn_record(turn, name, args, result, ms, 0, "")
        record.recovery = RECOVERY_HOOK
        log.turns.append(record)
        self._note_result(name, args, result, record)
        self._score_turn(record, log, pose_before)
        self._executor_step(turn, log, record)
        if record.new_evidence:
            # the automatic step worked: the model gets the fresh result and a
            # clean stagnation counter instead of another recovery instruction
            self._recovery_notes = []
            self._stagnant_turns = 0
        entry = log.recoveries[-1] if log.recoveries else {}
        if isinstance(entry, dict):
            entry["kind"] = RECOVERY_HOOK
            entry["executed"] = {"tool": name, "args": args, "status": result.status}
        return True

    def _note_result(
        self, tool: str, args: dict, result: ToolResult, record: TurnRecord
    ) -> None:
        prompt = record.result_prompt or result.to_prompt()
        line = (
            f"[turn {record.turn}] {tool}({json.dumps(args, default=str)[:100]}) -> "
            f"{_truncate(prompt, self.max_tool_chars)}"
        )
        self._feedback.append(line)
        self._feedback = self._feedback[-self.max_result_history :]

        # canonical tool-turn history for backends that render it themselves
        if self.max_history_turns:
            self._history.append(
                {"tool_call": {"tool": tool, "args": dict(args)},
                 "result": prompt}
            )
            self._history = self._history[-self.max_history_turns :]

        # images published by this call lead the next turn's context
        published = self.ctx.extra.get("images") or {}
        pending = [obs for obs in published if obs not in self._known_image_ids]
        for observation_id in pending:
            self._cache_image(observation_id, published[observation_id])
            # a frame a tool published under an obs_N id becomes an observation
            # of this episode: the model can cite it and the tools can resolve it
            self._adopt_published_frame(observation_id, record.turn)
        self._pending_image_ids = pending

        # tool-queued events go through the same hook as external events
        queued = self.ctx.extra.get("events")
        if isinstance(queued, list) and queued:
            self.ctx.extra["events"] = []
            for item in queued:
                if isinstance(item, dict):
                    self.emit_event(str(item.get("name", "tool_event")), item.get("data") or {})

    # ------------------------------------------------------------------
    # usage attribution
    # ------------------------------------------------------------------

    def _mark_used(self, log: EpisodeLog) -> None:
        """Mark a result as used when a later reply cites one of its reference
        tokens (evidence refs or id-like values). A citation proxy: the runtime
        cannot observe attention, only what the model echoes back."""
        for index, record in enumerate(log.turns):
            if record.tool in ("", NO_TOOL):
                continue
            tokens = self._reference_tokens(record)
            if not tokens:
                continue
            for later in log.turns[index + 1 :]:
                if later.raw_reply and any(token in later.raw_reply for token in tokens):
                    record.used = True
                    break

    @staticmethod
    def _reference_tokens(record: TurnRecord) -> set[str]:
        tokens: set[str] = set()
        for ref in record.evidence:
            kind, ref_id = str(ref.get("kind", "")), str(ref.get("ref", ""))
            if ref_id:
                tokens.add(ref_id)
                tokens.add(f"{kind}:{ref_id}")

        def scan(value: Any) -> None:
            if isinstance(value, str):
                tokens.update(_TOKEN_RE.findall(value))
            elif isinstance(value, dict):
                for item in value.values():
                    scan(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    scan(item)

        scan(record.data)
        scan(record.args)
        return tokens


def run_episode(
    registry,
    world,
    memory,
    vlm,
    instruction: str,
    **kwargs,
) -> EpisodeLog:
    """Convenience wrapper: build an ``AgentRuntime`` and run one episode."""
    return AgentRuntime(registry, world, memory, vlm, **kwargs).run(instruction)
