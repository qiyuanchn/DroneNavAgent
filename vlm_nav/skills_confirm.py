"""The confirmation skill: reach a candidate, then confirm the instance.

Version 2 of the completion contract (goal.md): grounding proposes a place,
navigation flies to that same place - so arrival alone cannot prove that the
object is the instructed one. ``confirm_candidate`` is the bounded routine that
turns a reached candidate into an *accepted identity*:

1. **approach** - if the candidate is farther than the confirmation standoff,
   fly approach segments (``approach_candidate``) until the object can be
   inspected. A blocked path keeps the candidate identity and returns
   ``uncertain`` with ``missing=["path"]``; it never drops into a generic search.
2. **new view** - capture a fresh observation from a *different* heading
   (``observe(mode="forward", yaw_deg=...)``) and record every returned
   ``obs_N``. Independence is decided on the image/view, not on a new id alone:
   the same pixels seen again do not count as new evidence.
3. **judge** - one model call per independent view compares the candidate's
   attributes and the *whole* relation anchor named by the current sub-task
   (``木门旁边`` / ``next to the wooden door``, not just the preposition) against
   what the view shows. Each call is counted in ``vlm_calls``. At least one
   independent view has to support the target/instance/attributes (and the
   anchor, when one is named) while no *decided* view vetoes it: an explicit
   0.0/low-confidence or opposing view blocks acceptance, an undecided
   ``confidence: null`` view neither supports nor vetoes.
4. **location consistency** - the model names the target object's box on each
   fresh frame, and *that box* is measured at the frame's own heading and the
   current position; the observed point is compared with the candidate
   estimate. The delta is therefore the difference between two independent
   measurements (grounding vs. the fresh view), not a depth read at the
   candidate's own projected pixel - that returned the candidate point by
   construction (holdout v2 accepted all five wrong instances with delta 0.0).
   No box or no depth means no acceptance: appearance alone keeps the verdict
   at ``uncertain``. A candidate grounded under an unresolved depth/model
   distance conflict is refused *before* any view or model call: the range is
   the open question, not the identity, and the same frame would repeat the
   same conflict. The refusal names a viewpoint change and a re-grounding; a
   second confirmation of the same candidate is ``confirm_stalled`` and marks
   it unusable until re-grounded.
5. **verdict** - ``accepted`` only when at least one view supports, no decided
   view vetoes, *and* the fresh measurement agrees; otherwise ``uncertain``
   (with a directed ``missing``:
   ``distance`` / ``identity`` / ``relation`` / ``path`` / ``localization``) or
   ``rejected`` (explicit different-instance evidence, or a position beyond the
   reject tolerance). A candidate with repeated confirmations but no new
   accepted one is stopped with ``blocked(confirm_stalled)`` and a different
   ``next_action`` instead of confirming the same candidate forever.

Only an ``accepted`` verdict is recorded as a confirmation that ``finish`` may
count; ``uncertain`` keeps the candidate but leaves the question open. A
verdict taken from an image that was already used by an earlier confirmation
does not increase the confirmation count and does not create a new evidence
record. The internal steps and model calls are published like ``search_target``
(``steps``, ``vlm_calls``, ``inner_tool_calls``, ``cost``).
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Mapping

import numpy as np

from . import skills_approach
from .observations import CANDIDATE_STORE_KEY
from .tool_impl_perception import (
    _as_frame,
    _crop_frame,
    _depth_point,
    _draw_hint_box,
    _map_box_from_region,
    _remember_rejected_triangulation,
    _vlm_from_ctx,
    parse_box,
    resolve_observation,
)
from .tools import (
    CONFIRMATION_ACCEPTED,
    CONFIRMATION_MISSING,
    CONFIRMATION_REJECTED,
    CONFIRMATION_UNCERTAIN,
    PROGRESS_INFORMATION,
    PROGRESS_NONE,
    PROGRESS_TASK,
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
    record_confirmation,
    select_scope_candidate,
    task_scope,
)

#: Views one confirmation call may judge (bounded: a confirmation is a step,
#: not a search).
DEFAULT_CONFIRM_VIEWS = 2
MAX_CONFIRM_VIEWS = 3
#: Heading offset between two confirmation views (degrees). Two views taken at
#: the same heading show the same pixels and cannot confirm anything new.
DEFAULT_TURN_DEG = 35.0
MAX_TURN_DEG = 80.0
#: Distance deviation tolerated between a fresh depth measurement and the
#: candidate estimate; beyond this the location and the identity disagree.
CONFIRM_POSITION_TOL_M = 1.0
#: A deviation this large is a different surface, not measurement noise.
CONFIRM_POSITION_REJECT_M = 2.0
#: Slack on the relation between a candidate and the anchor stage's own
#: recorded place when the sub-task does not declare ``within_m``.
CONFIRM_ANCHOR_RELATION_M = 3.5
#: How close the drone must be to inspect the candidate.
CONFIRM_STANDOFF_M = skills_approach.APPROACH_STANDOFF_M
CONFIRM_ARRIVE_RADIUS_M = skills_approach.ARRIVE_RADIUS_M + 0.15
#: Confidence below which a model "target=true" is not enough to accept.
MIN_CONFIRM_CONFIDENCE = 0.5
#: Consecutive confirmation attempts without a new accepted confirmation: after
#: this many the routine stops re-confirming the same candidate and hands the
#: caller a different strategy (the seed-13 relation loop).
MAX_CONFIRM_ATTEMPTS = 3
#: inner steps / model calls one confirmation may spend
DEFAULT_CONFIRM_STEPS = 8
MAX_CONFIRM_STEPS = 16
#: How many approach segments the approach phase may fly before it reports
#: ``uncertain: distance`` instead of spending the episode budget.
MAX_CONFIRM_LEGS = 2
#: Two headings closer than this share the depth map of the current pose.
SAME_HEADING_DEG = 8.0
#: bounded per-candidate confirmation ledger in ``ctx.extra``
CONFIRMATION_LEDGER_KEY = "confirmation_ledger"
CONFIRMATION_LEDGER_LIMIT = 32
MAX_EVIDENCE_KEYS = 16

#: One cheap retry when a confirmation view names the target but gives no box:
#: the independent position half of confirmation is otherwise unavailable (the
#: v5 bedroom run had 15 such views). The retry asks for the box only on the
#: same fresh frame, so it costs one model call and no new capture.
CONFIRM_BOX_SYSTEM_PROMPT = (
    "You are the target-grounding step of a drone navigation agent. Return ONE "
    'JSON object and nothing else: {"box": [x1, y1, x2, y2], "target": true}. '
    "box is the visible extent of the instructed target in normalized 0-1000 "
    "image coordinates (x1, y1 = top-left, x2, y2 = bottom-right), not the wall "
    'or the whole scene. Use {"target": false, "box": null} when the target is '
    "not visible."
)

CONFIRM_BOX_USER_PROMPT = (
    "Instruction: {instruction}\n"
    "Your previous answer named the target but gave no usable box. Look at the "
    "attached fresh view again and return only the target's box (normalized "
    "0-1000, corner form)."
)

CONFIRM_SYSTEM_PROMPT = (
    "You are the identity-confirmation step of a drone navigation agent. You are "
    "shown ONE fresh view of a candidate object, taken from a new heading after "
    "the drone approached the candidate's estimated position. Decide whether this "
    "view shows the instructed target AND the same instance the candidate was "
    "grounded from. The relation anchor is the WHOLE object+relation phrase the "
    "instruction names (for example '木门旁边' / 'next to the wooden door'), not "
    "just the preposition: relation_supported is false unless this frame shows "
    "the anchor object itself in that relation to the candidate. Do not infer the "
    "relation from an earlier frame. Judge only what the frame shows. Answer with "
    "ONE short JSON object "
    "and nothing else: "
    '{"target": true, "same_instance": true, "attributes_match": true, '
    '"relation_supported": true, "box": [x1, y1, x2, y2], '
    '"confidence": 0.7, "reason": "a few words", '
    '"missing": [], "next_view": "which other view would distinguish the '
    'hypotheses"}. The values above are placeholders: do not copy them. '
    '"box" is the instructed target object itself in this fresh view, as a '
    "corner box [x1, y1, x2, y2] in normalized 0-1000 image coordinates "
    "(x1, y1 = top-left, x2, y2 = bottom-right); box the object's visible "
    "extent, not the wall or the whole scene; a box covering half the frame or "
    "more is ignored, so use null when the target is not visible or its extent "
    "cannot be told. "
    'confidence is a number from 0 to 1 rating THIS view; use 0.0 or 1.0 only '
    'for a certain judgement, and use null (not 0.0) when this view cannot '
    'decide. Use null for a property this view cannot decide, and '
    '{"target": false, "same_instance": false} when the object is absent or is '
    "clearly a different instance. Keep every value short."
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _as_int(value: Any, default: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def _norm_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _as_flag(value: Any) -> bool | None:
    """A model boolean (``true`` / ``yes`` / ``null``) or None when undecidable."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("true", "yes", "1", "y", "same", "match"):
        return True
    if text in ("false", "no", "0", "n", "different", "mismatch"):
        return False
    return None


def _position(world: Any) -> np.ndarray:
    try:
        arr = np.asarray(getattr(world, "position_flu", None), dtype=np.float64).reshape(-1)[:3]
    except (TypeError, ValueError):
        return np.zeros(3, dtype=np.float64)
    return arr if arr.size == 3 and np.isfinite(arr).all() else np.zeros(3, dtype=np.float64)


def _yaw(world: Any) -> float:
    try:
        return float(getattr(world, "yaw_rad", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _frame_digest(frame: Any) -> str:
    """Stable digest of one view's pixels (the *image*, not its id).

    The independence rule is about evidence: two ids that show the same pixels
    are the same image, so asking the model again cannot add a confirmation.
    """
    try:
        if hasattr(frame, "detach"):
            array = frame.detach().cpu().numpy()
        else:
            array = np.asarray(frame)
        array = np.ascontiguousarray(array)
        return hashlib.sha1(array.tobytes()).hexdigest()
    except Exception:  # noqa: BLE001 - without pixels the id is the key
        return ""


def _evidence_key(obs_id: str, digest: str) -> str:
    return f"frame:{digest}" if digest else f"obs:{obs_id}"


def _candidate_store(ctx: ToolContext) -> dict:
    extra = ctx.extra if isinstance(ctx.extra, dict) else {}
    store = extra.get(CANDIDATE_STORE_KEY)
    return store if isinstance(store, dict) else {}


def _set_candidate_status(ctx: ToolContext, candidate_id: str, status: str, note: str = "") -> None:
    store = _candidate_store(ctx)
    for key in (candidate_id, str(candidate_id)):
        record = store.get(key)
        if isinstance(record, dict):
            record["status"] = status
            if note:
                record["confirmation_reason"] = note[:200]
            return


def _set_candidate_confirmation(
    ctx: ToolContext, candidate_id: str, verdict: str, revision: int,
    missing: list[str], observation_ids: list[str],
) -> None:
    store = _candidate_store(ctx)
    for key in (candidate_id, str(candidate_id)):
        record = store.get(key)
        if isinstance(record, dict):
            if str(record.get("status") or "") == CONFIRMATION_REJECTED:
                return
            record["confirmation_status"] = verdict
            record["confirmation_revision"] = int(revision)
            record["confirmation_missing"] = list(missing)
            record["confirmation_observation_ids"] = list(observation_ids)
            return


def _ledger(ctx: ToolContext) -> dict:
    extra = ctx.extra if isinstance(ctx.extra, dict) else {}
    ledger = extra.get(CONFIRMATION_LEDGER_KEY)
    if not isinstance(ledger, dict):
        ledger = {}
        extra[CONFIRMATION_LEDGER_KEY] = ledger
    return ledger


def _ledger_entry(ctx: ToolContext, candidate_id: str) -> dict:
    ledger = _ledger(ctx)
    entry = ledger.get(candidate_id)
    if not isinstance(entry, dict):
        entry = {
            "evidence_keys": [],
            "observation_ids": [],
            "accepted": 0,
            "attempts": 0,
            "attempts_since_accept": 0,
            "no_progress_streak": 0,
            "last_verdict": "",
            "accepted_revision": None,
            "revision": None,
            "stalled": False,
        }
        ledger[candidate_id] = entry
        while len(ledger) > CONFIRMATION_LEDGER_LIMIT:
            ledger.pop(next(iter(ledger)))
    if not isinstance(entry.get("evidence_keys"), list):
        entry["evidence_keys"] = []
    if not isinstance(entry.get("observation_ids"), list):
        entry["observation_ids"] = []
    return entry


def _instruction(ctx: ToolContext, args: Mapping) -> str:
    text = str(args.get("instruction") or "").strip()
    if text:
        return text
    try:
        from .tool_impl_task import subtask_ground_instruction

        text = subtask_ground_instruction(ctx)
    except Exception:  # noqa: BLE001 - a context without the task module
        text = ""
    if not text:
        extra = ctx.extra if isinstance(ctx.extra, dict) else {}
        text = str(extra.get("subgoal") or extra.get("instruction") or "")
    return text.strip() or "<the instructed target>"


#: Chinese relation anchors the shared (English) phrase list does not carry.
#: The task files of the long-horizon runs are written in Chinese ("门旁边的
#: 柜子"), and the anchor is exactly what the confirmation view has to show.
CHINESE_RELATION_WORDS = (
    "旁边", "左侧", "右侧", "左边", "右边", "附近", "前面", "后面",
    "上方", "下方", "之间", "对面",
)


#: Clause separators and lead-in words of a Chinese sub-task text. The anchor
#: phrase is the object immediately before the relation word ("木门旁边"), not
#: just the preposition the shared parser returns.
_ANCHOR_SEPARATORS = "，。；、：,.;:!?！？\n"
_RELATION_LEAD_WORDS = (
    "再找到", "找到了", "寻找", "找到", "搜索", "确认", "核实", "定位",
    "观察", "看到", "看见", "然后", "接着", "最后", "首先", "现在",
    "再", "先", "从", "向", "往", "到", "在", "和", "与", "及",
)
_EN_RELATION_RE = re.compile(
    r"(next to|next by|close to|near to|nearby|beside|alongside|left of|"
    r"right of|in front of|behind|between|underneath|under|below|above|"
    r"on top of|attached to|adjacent to)\s+([^,.;!?，。；！？\n]{1,48})",
    re.IGNORECASE,
)


def _clean_anchor_head(head: str) -> str:
    """The object phrase in front of a Chinese relation word ("找到木门" -> "木门")."""
    text = str(head or "").strip()
    for separator in _ANCHOR_SEPARATORS:
        text = text.rsplit(separator, 1)[-1].strip()
    for _ in range(4):
        before = text
        for word in sorted(_RELATION_LEAD_WORDS, key=len, reverse=True):
            position = text.rfind(word)
            if position >= 0:
                tail = text[position + len(word):].strip()
                if tail:
                    text = tail
                    break
        if text == before:
            break
    if "的" in text:
        tail = text.rsplit("的", 1)[-1].strip()
        if tail:
            text = tail
    text = text.rstrip("的").strip()
    return text[-10:].strip()


def _anchor_from_text(text: str) -> str:
    """The whole object+relation phrase one text names ("" when it names none)."""
    value = str(text or "").strip()
    if not value:
        return ""
    for word in sorted(CHINESE_RELATION_WORDS, key=len, reverse=True):
        index = value.find(word)
        if index <= 0:
            continue
        head = _clean_anchor_head(value[:index])
        if head:
            return f"{head}{word}"
    match = _EN_RELATION_RE.search(value)
    if match:
        return " ".join(match.group(0).split())[:64]
    return ""


def _is_bare_relation(text: str) -> bool:
    """True when ``text`` is only a preposition, not an object+relation anchor."""
    lowered = str(text or "").strip().lower()
    if lowered in CHINESE_RELATION_WORDS:
        return True
    try:
        from .skills_search import relation_phrase

        phrase = relation_phrase(lowered)
    except Exception:  # noqa: BLE001 - the relation is optional
        phrase = ""
    return bool(phrase) and phrase == lowered


def _relation_anchor(
    ctx: ToolContext, args: Mapping, instruction: str
) -> tuple[str, str]:
    """``(anchor, source)``: the full relation phrase the current sub-task names.

    Version 2 passes the *whole* anchor to the view check ("木门旁边"), not just
    the preposition ("旁边"): the model has to see the candidate **and** the
    anchor object in that relation. The sub-task's own goal /
    ``expected_observation`` is the most specific source; the instruction and an
    explicit argument follow.
    """
    explicit = str(args.get("relation") or "").strip()
    if explicit and not _is_bare_relation(explicit):
        return explicit[:64], "argument"
    try:
        from .tool_impl_task import current_subtask

        subtask = current_subtask(ctx)
    except Exception:  # noqa: BLE001 - a context without the task module
        subtask = None
    candidates: list[tuple[str, str]] = []
    if isinstance(subtask, Mapping):
        for key in ("goal", "expected_observation"):
            value = str(subtask.get(key) or "").strip()
            if value:
                candidates.append((value, f"subtask.{key}"))
    if instruction:
        candidates.append((instruction, "instruction"))
    for text, source in candidates:
        anchor = _anchor_from_text(text)
        if anchor and (not explicit or anchor != explicit):
            return anchor, source
    if explicit:
        return explicit, "argument"
    try:
        from .skills_search import relation_phrase

        phrase = relation_phrase(instruction)
    except Exception:  # noqa: BLE001 - the relation is optional
        phrase = ""
    if phrase:
        return phrase, "relation_phrase"
    for word in CHINESE_RELATION_WORDS:
        if word in instruction:
            return word, "relation_word"
    return "", ""


def _subtask_anchor_check(ctx: ToolContext, record: Mapping) -> dict:
    """Is the candidate's claimed place consistent with this sub-task's anchor?

    Two v6 acceptances happened 2.21 m / 2.95 m from their stage while the
    stage declared ``locate_m=1.5``. Confirmation has no ground truth, so the
    check uses what the sub-task itself records: a declared ``goal_position``
    (tolerance ``locate_m``) or the anchor stage's own located evidence (a
    relation sanity bound). With neither, the check is unavailable and the old
    verdict stands - it never invents a place.
    """
    try:
        point = np.asarray(record.get("position_flu"), dtype=np.float64).reshape(-1)[:3]
    except (TypeError, ValueError):
        return {"available": False, "reason": "no_candidate_position"}
    if point.size != 3 or not np.isfinite(point).all():
        return {"available": False, "reason": "no_candidate_position"}
    try:
        from .tool_impl_task import (
            _locate_limit,
            _position_of,
            current_subtask,
            plan_subtasks,
        )

        subtask = current_subtask(ctx)
    except Exception:  # noqa: BLE001 - a context without the task module
        subtask = None
    if not isinstance(subtask, Mapping):
        return {"available": False, "reason": "no_subtask"}
    anchors: list[dict] = []
    goal = _position_of(subtask.get("goal_position"))
    if goal:
        anchors.append(
            {
                "source": "subtask.goal_position",
                "point": goal,
                "tolerance_m": float(_locate_limit(subtask)),
            }
        )
    subtasks = {
        str(item.get("id")): item
        for item in plan_subtasks(ctx) or []
        if isinstance(item, Mapping)
    }
    for dependency in subtask.get("depends_on") or ():
        anchor = subtasks.get(str(dependency))
        if not isinstance(anchor, Mapping):
            continue
        tolerance = max(float(_locate_limit(subtask)), CONFIRM_ANCHOR_RELATION_M)
        for place in anchor.get("_evidence_places") or []:
            try:
                anchor_point = np.asarray(place, dtype=np.float64).reshape(-1)[:3]
            except (TypeError, ValueError):
                continue
            if anchor_point.size != 3 or not np.isfinite(anchor_point).all():
                continue
            anchors.append(
                {
                    "source": f"anchor.{dependency}",
                    "point": [float(value) for value in anchor_point],
                    "tolerance_m": tolerance,
                }
            )
    if not anchors:
        return {"available": False, "reason": "no_subtask_anchor"}
    for anchor in anchors:
        anchor["distance_m"] = float(
            np.linalg.norm(point - np.asarray(anchor["point"], dtype=np.float64))
        )
    nearest = min(anchors, key=lambda anchor: anchor["distance_m"])
    return {
        "available": True,
        "consistent": any(
            anchor["distance_m"] <= anchor["tolerance_m"] for anchor in anchors
        ),
        "source": nearest["source"],
        "distance_m": round(nearest["distance_m"], 3),
        "tolerance_m": round(float(nearest["tolerance_m"]), 3),
        "anchors_checked": len(anchors),
    }


def _confidence_state(confidence: Any) -> str:
    """How one view's confidence reads: ``ok`` / ``low`` / ``undecided``.

    ``null`` (or a missing field) is *undecided* - it neither supports nor
    vetoes - and must not be conflated with an explicit ``0.0`` ("this view is
    certainly not support"). The v8 prompt itself showed ``0.0`` as its example
    and 15/16 views copied it, which blocked every correct candidate.
    """
    value = _as_float(confidence, None)
    if value is None:
        return "undecided"
    return "low" if value < MIN_CONFIRM_CONFIDENCE else "ok"


def _normalize_missing(values: Any) -> list[str]:
    out: list[str] = []
    items = values if isinstance(values, (list, tuple)) else [values]
    for item in items:
        text = str(item or "").strip().lower()
        if text in CONFIRMATION_MISSING and text not in out:
            out.append(text)
    return out


def _parse_reply(reply: Any) -> dict | None:
    """One confirmation reply -> the fields the verdict is built from."""
    try:
        from .skills_search import _json_object
    except Exception:  # noqa: BLE001
        return None
    payload = _json_object(reply)
    if not isinstance(payload, Mapping):
        return None
    fields = (
        ("target", ("target", "looks_like_target", "is_target", "contains_target")),
        ("same_instance", ("same_instance", "same", "identical", "same_object")),
        ("attributes_match", ("attributes_match", "attrs_match", "attributes")),
        ("relation_supported", ("relation_supported", "relation", "anchor_supported")),
    )
    parsed: dict[str, Any] = {}
    for name, keys in fields:
        value = None
        for key in keys:
            if key in payload:
                value = payload[key]
                break
        parsed[name] = _as_flag(value)
    confidence = payload.get("confidence", payload.get("score"))
    parsed["confidence"] = _as_float(confidence, None)
    parsed["confidence_state"] = _confidence_state(parsed["confidence"])
    parsed["reason"] = str(payload.get("reason", payload.get("why", "")) or "")[:160]
    parsed["missing"] = _normalize_missing(payload.get("missing"))
    parsed["next_view"] = str(payload.get("next_view", payload.get("other_view", "")) or "")[:120]
    raw_box = payload.get("box", payload.get("target_box", payload.get("bbox")))
    box, box_problem = parse_box(raw_box) if raw_box is not None else (None, "missing")
    parsed["box"] = box.to_list() if box is not None else None
    parsed["box_problem"] = "" if box is not None else str(box_problem or "invalid_box")
    if all(parsed[name] is None for name, _ in fields):
        return None
    return parsed


def _ask_view(
    ctx: ToolContext, backend: Any, instruction: str, relation: str,
    candidate_id: str, record: Mapping, view: Mapping,
    approach_distance_m: float | None, relation_source: str = "",
) -> dict:
    """One model call judging one new view; never raises.

    The candidate's *source frame* is attached as the first image when it is
    still available, so "is this the same instance?" is a visual comparison
    (source view vs. fresh view) rather than a question about a fresh frame in
    isolation. With no source frame the model can only judge the fresh view, and
    ``same_instance`` has to stay undecided.
    """
    if backend is None:
        return {"error": "no_vlm_backend"}
    attrs = record.get("attrs") if isinstance(record.get("attrs"), dict) else {}
    attr_text = ", ".join(f"{key}={value}" for key, value in list(attrs.items())[:6]) or "none"
    relation_text = relation or "none"
    source = str(record.get("observation_id") or "")
    source_record = resolve_observation(ctx, source) if source else None
    source_frame = getattr(source_record, "frame", None) if source_record is not None else None
    frames = [source_frame, view["frame"]] if source_frame is not None else [view["frame"]]
    if source_frame is not None:
        image_legend = (
            f"Image 1: the candidate's source view {source}. "
            f"Image 2: the fresh view {view.get('obs_id')}."
        )
    else:
        image_legend = (
            f"Only one image is attached: the fresh view {view.get('obs_id')} "
            "(the source frame is no longer available, so same_instance can only "
            "be judged from the candidate attributes and relation)."
        )
    relation_rule = (
        "relation_supported is false unless the frame shows both the candidate "
        f"and '{relation_text}' in that relation. "
        if relation
        else "No relation anchor is named: relation_supported is not part of this check. "
    )
    user_prompt = (
        f"Instruction: {instruction}\n"
        f"Candidate: {candidate_id} (grounded in {source or 'an earlier view'}; "
        f"attributes: {attr_text})\n"
        f"Relation anchor (whole phrase from {relation_source or 'the instruction'}): "
        f"{relation_text}\n"
        f"Fresh view: {view.get('obs_id')}, heading {view.get('heading_deg')} deg, "
        f"candidate expected about {approach_distance_m if approach_distance_m is not None else '?'} m away.\n"
        f"{image_legend}\n"
        "Does the fresh view show the instructed target, is it the SAME instance as "
        "the candidate (compare the two images when both are attached), do its "
        "attributes match, and is the relation anchor supported? "
        f"{relation_rule}Use null for a property this evidence "
        "cannot decide. Name the missing property and which other view would "
        "distinguish the hypotheses."
    )
    # the candidate's own projected pixel on this view: metadata only (a real
    # vision backend judges the image), but it lets a scripted backend draw its
    # box around the object it is asked about instead of the frame centre
    candidate_pixel = None
    candidate_position = record.get("position_flu")
    if (
        isinstance(candidate_position, (list, tuple))
        and len(candidate_position) >= 3
        and view.get("pose_flu") is not None
        and view.get("heading_deg") is not None
    ):
        candidate_pixel = skills_approach.world_to_pixel(
            candidate_position,
            view["pose_flu"],
            math.radians(float(view["heading_deg"])),
            view.get("intrinsics") or {},
        )
    try:
        reply = backend.decide(
            frames,
            CONFIRM_SYSTEM_PROMPT,
            user_prompt,
            context={
                "step": "confirm_candidate",
                "tool": "confirm_candidate",
                "candidate_id": candidate_id,
                "observation_id": view.get("obs_id"),
                "instruction": instruction,
                "relation": relation or None,
                "source_observation_id": source,
                "source_frame_attached": source_frame is not None,
                "pose": view.get("pose_flu"),
                "yaw_deg": view.get("heading_deg"),
                "candidate_pixel": (
                    None if candidate_pixel is None
                    else [round(float(candidate_pixel[0]), 1), round(float(candidate_pixel[1]), 1)]
                ),
            },
        )
    except Exception as exc:  # noqa: BLE001 - a backend failure never crashes the skill
        return {"error": f"{type(exc).__name__}: {exc}"}
    parsed = _parse_reply(reply)
    if parsed is None:
        return {"error": "unparsable_reply", "raw": str(reply)[:200], "source_frame": source_frame is not None}
    parsed["backend"] = getattr(backend, "name", None) or type(backend).__name__
    parsed["source_frame"] = source_frame is not None
    parsed["vlm_calls"] = 1
    if (
        parsed.get("target") is not False
        and not parsed.get("box")
        and getattr(backend, "allow_box_retry", True)
    ):
        # the view judged the target but supplied no outline: one box-only
        # retry on the same fresh frame (no new capture, one model call). The
        # retry carries a visual anchor - a red rectangle on the candidate's
        # projected place, else an enlarged centre crop - because the v6 plain
        # second look recovered no box in 22 ground attempts.
        retry_context = {
            "step": "confirm_candidate_box",
            "tool": "confirm_candidate",
            "candidate_id": candidate_id,
            "observation_id": view.get("obs_id"),
            "instruction": instruction,
            "pose": view.get("pose_flu"),
            "yaw_deg": view.get("heading_deg"),
        }
        retry_prompt = CONFIRM_BOX_USER_PROMPT.format(instruction=instruction)
        retry_source = _as_frame(view["frame"])
        if retry_source is None:
            retry_source = view["frame"]
        retry_frame = retry_source
        retry_region: list[float] | None = None
        if candidate_pixel is not None:
            retry_mode = "annotated_candidate_pixel"
            hint = [
                float(candidate_pixel[0]) - 80.0,
                float(candidate_pixel[1]) - 80.0,
                float(candidate_pixel[0]) + 80.0,
                float(candidate_pixel[1]) + 80.0,
            ]
            retry_frame = _draw_hint_box(retry_source, hint)
            retry_prompt += (
                "\nThe attached frame is the same fresh view with a red rectangle "
                "around where the candidate is expected. Return the target's exact "
                "box, correcting the rectangle when it is misplaced."
            )
        else:
            retry_mode = "centre_crop"
            retry_region = [200.0, 200.0, 600.0, 600.0]
            retry_frame, _size, _upscale, _origin = _crop_frame(
                retry_source, retry_region
            )
            retry_prompt += (
                "\nThe attached image is an enlarged central crop of the fresh "
                "view. Return the target's box in this crop's normalized 0-1000 "
                "coordinates."
            )
        try:
            retry_reply = backend.decide(
                [retry_frame],
                CONFIRM_BOX_SYSTEM_PROMPT,
                retry_prompt,
                context=retry_context,
            )
            parsed["vlm_calls"] = 2
            retry_parsed = _parse_reply(retry_reply)
            if retry_parsed is not None and retry_parsed.get("box"):
                box = retry_parsed["box"]
                if retry_region is not None:
                    mapped = _map_box_from_region(box, retry_region)
                    if mapped is not None:
                        box = mapped
                parsed["box"] = box
                parsed["box_problem"] = ""
                parsed["box_retry"] = True
                parsed["box_retry_mode"] = retry_mode
                if retry_region is not None:
                    parsed["box_retry_region"] = list(retry_region)
        except Exception:  # noqa: BLE001 - a failed retry is just no box
            pass
    return parsed


def _heading_delta(record: Any, yaw_rad: float) -> float:
    heading = getattr(record, "yaw_deg", None)
    if heading is None:
        heading_rad = getattr(record, "yaw_rad", None)
        if heading_rad is None:
            return float("inf")
        heading = math.degrees(float(heading_rad))
    return abs(_norm_deg(float(heading) - math.degrees(float(yaw_rad))))


def _offsets_for(target: Any, world: Any, turn_deg: float) -> list[float]:
    """Headings to capture, candidate-relative first then alternating offsets.

    The first view faces the candidate wherever it is relative to the current
    heading (a candidate behind the drone must not be judged from a frame that
    cannot hold it); the other views are ``turn_deg`` to either side of that
    bearing, so the model sees the same object from a genuinely different angle.
    """
    if target.position is not None:
        here = _position(world)
        delta = np.asarray(target.position, dtype=np.float64) - here
        bearing = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
        first = _norm_deg(bearing - math.degrees(_yaw(world)))
    else:
        first = 0.0
    offsets = [first]
    step = max(10.0, float(turn_deg))
    for k in range(1, MAX_CONFIRM_VIEWS + 2):
        for candidate in (first + step * k, first - step * k):
            candidate = _norm_deg(candidate)
            if all(abs(_norm_deg(candidate - other)) > 5.0 for other in offsets):
                offsets.append(candidate)
    return offsets


def _next_action(tool: str, args: dict, why: str) -> dict:
    return {"tool": tool, "args": dict(args), "why": why}


def _distance_action(candidate_id: str, instruction: str, target: Any, turn_deg: float) -> dict:
    if target.position is not None:
        return _next_action(
            "approach_candidate",
            {"candidate_id": candidate_id, "standoff_m": CONFIRM_STANDOFF_M},
            "close the remaining distance first: identity cannot be confirmed from "
            "too far away",
        )
    return _next_action(
        "observe",
        {"mode": "forward"},
        "the candidate has no range yet: capture the current view and ground it "
        "again (ground_target) so a depth measurement can replace the ray",
    )


def _identity_action(
    candidate_id: str, instruction: str, turn_deg: float, relation: str,
) -> dict:
    args: dict[str, Any] = {
        "candidate_id": candidate_id,
        "instruction": instruction,
        "turn_deg": _clamp(turn_deg, 10.0, MAX_TURN_DEG),
        "max_views": DEFAULT_CONFIRM_VIEWS,
    }
    if relation:
        args["relation"] = relation
    return _next_action(
        "confirm_candidate",
        args,
        "the same image cannot settle the identity: take a different heading "
        "(a new view), not another look at the same frame",
    )


def _relation_action(
    candidate_id: str, instruction: str, relation: str, turn_deg: float,
) -> dict:
    args: dict[str, Any] = {
        "candidate_id": candidate_id,
        "instruction": instruction,
        "relation": relation or "<the relation anchor>",
        "turn_deg": _clamp(max(turn_deg * 1.8, 60.0), 10.0, MAX_TURN_DEG),
        "max_views": DEFAULT_CONFIRM_VIEWS,
    }
    return _next_action(
        "confirm_candidate",
        args,
        "the relation anchor was not visible: look from an angle that holds both "
        "the candidate and the anchor object",
    )


def _path_action(result: Any, candidate_id: str) -> dict:
    data = result.data if isinstance(getattr(result, "data", None), dict) else {}
    action = data.get("next_action")
    if isinstance(action, dict) and action.get("tool") in (
        "go_to_viewpoint", "propose_viewpoints", "approach_candidate", "observe",
    ):
        return dict(action)
    return _next_action(
        "propose_viewpoints",
        {"count": 3},
        "the route is blocked, not the identity: keep the candidate and look for "
        "another way to the same place",
    )


def _change_view_action(missing: list[str], turn_deg: float) -> dict:
    """A different call after a confirmation that could not add evidence."""
    if "relation" in missing:
        return _next_action(
            "observe",
            {"mode": "lookaround", "n_views": 4, "spread_deg": 180.0},
            "the anchor was already probed from this place: look from another "
            "viewpoint instead of confirming the same candidate again",
        )
    return _next_action(
        "propose_viewpoints",
        {"count": 3},
        "this candidate has not produced a new accepted confirmation from here: "
        "change the viewpoint (or compare candidates / re-ground another "
        "instance) instead of repeating confirm_candidate",
    )


def _strategy_change(reason: str, candidate_id: str, missing: list[str]) -> dict:
    """What the caller should try instead of another confirm of this candidate."""
    return {
        "reason": reason,
        "candidate_id": candidate_id,
        "stop": "confirm_candidate",
        "missing": list(missing),
        "alternatives": [
            {"tool": "observe", "why": "look at the anchor/candidate from a new heading"},
            {"tool": "propose_viewpoints", "why": "move to an observation point that holds both objects"},
            {"tool": "compare_candidates", "why": "compare the grounded instances directly"},
            {"tool": "ground_target", "why": "ground a different candidate instance"},
        ],
    }


def _distance_unresolved(record: Mapping) -> bool:
    """Whether the candidate's place came from an unresolved depth/model conflict."""
    if record.get("distance_uncertain") is True:
        return True
    if str(record.get("distance_source") or "") == "conflict":
        return True
    conflict = record.get("distance_conflict")
    return isinstance(conflict, Mapping) and conflict.get("resolved") is False


def _reground_view_action(
    ctx: ToolContext, record: Mapping, instruction: str, turn_deg: float,
) -> dict:
    """The executable way out of a distance-uncertain candidate.

    Confirming the same view again cannot resolve which of the two ranges is
    the object's; the frame is the input to the conflict. The next step has to
    change the viewpoint first and then ground the target from the new frame.
    The returned action names ``ground_target`` (the follow-up after the view
    change), and carries the headings/unchecked sectors that would make the new
    view genuinely different.
    """
    step = max(45.0, min(MAX_TURN_DEG, float(turn_deg or DEFAULT_TURN_DEG)))
    current = math.degrees(_yaw(getattr(ctx, "world", None)))
    source_id = str(record.get("observation_id") or "")
    source = resolve_observation(ctx, source_id) if source_id else None
    if source is not None and getattr(source, "yaw_deg", None) is not None:
        current = float(source.yaw_deg)
    second = record.get("border_second_view")
    second = second if isinstance(second, Mapping) else None
    second_yaw = _as_float(second.get("yaw_deg"), None) if second else None
    offsets = [step, -step]
    if second_yaw is not None:
        # a border-large box needs the view toward the edge it runs into, not a
        # generic 45-deg turn away from it
        offsets = [float(second_yaw), -float(second_yaw), step, -step]
    unique_offsets: list[float] = []
    for offset in offsets:
        if all(abs(_norm_deg(offset - kept)) > 1.0 for kept in unique_offsets):
            unique_offsets.append(offset)
    offsets = unique_offsets
    viewpoints = [
        {
            "yaw_deg": round(offset, 1),
            "why": (
                "turn toward the frame edge the candidate occupies before "
                "grounding again: a view that returns the same pixels returns "
                "the same depth/model conflict"
                if second_yaw is not None and abs(offset - second_yaw) <= 1.0
                else (
                    "turn from the current heading before grounding: a view that "
                    "returns the same pixels returns the same depth/model conflict"
                )
            ),
        }
        for offset in offsets
    ]
    unchecked: list[list[float]] = []
    spatial = getattr(getattr(ctx, "memory", None), "spatial", None)
    coverage = getattr(spatial, "coverage", None)
    if callable(coverage):
        try:
            report = coverage() or {}
        except Exception:  # noqa: BLE001 - coverage is advisory
            report = {}
        for item in report.get("unchecked_deg") or []:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                unchecked.append([float(item[0]), float(item[1])])
        for low, high in unchecked[:2]:
            centre = 0.5 * (float(low) + float(high))
            viewpoints.append({
                "yaw_deg": round(_norm_deg(centre - current), 1),
                "sector_deg": [low, high],
                "why": f"the {low:g}..{high:g} deg sector has not been faced yet",
            })
    action = _next_action(
        "ground_target",
        {"instruction": instruction},
        (
            f"the candidate's depth and the model range disagree; observe from a "
            f"different heading first (yaw_deg around {offsets[0]:+.0f} deg), then "
            "ground_target on the new observation - do not confirm this candidate "
            "again before it is re-grounded"
        ),
    )
    action["viewpoints"] = viewpoints
    action["unchecked_deg"] = unchecked
    action["observe_first"] = {
        "tool": "observe",
        "args": {"mode": "forward", "yaw_deg": round(offsets[0], 1)},
    }
    if second is not None:
        action["dedicated_second_view"] = dict(second)
        action["step_hint"] = (
            f"step about {second.get('step_side_m') or second.get('step_m')} m to "
            f"the {second.get('side') or 'edge'} side before grounding again: "
            "rotation alone gives no baseline, the triangulation needs parallax"
        )
    return action


def _distance_uncertain_confirmation(
    ctx: ToolContext,
    record: dict,
    candidate_id: str,
    instruction: str,
    turn_deg: float,
) -> ToolResult:
    """Refuse to confirm a distance-uncertain candidate and direct a re-ground."""
    scope = task_scope(ctx)
    revision = int(scope.get("evidence_revision", 0))
    entry = _ledger_entry(ctx, candidate_id)
    store = _candidate_store(ctx)
    stored = store.get(candidate_id)
    target = stored if isinstance(stored, dict) else record
    attempts = int(target.get("distance_uncertain_confirms") or 0) + 1
    target["distance_uncertain_confirms"] = attempts
    entry["attempts"] = int(entry.get("attempts", 0)) + 1
    entry["attempts_since_accept"] = int(entry.get("attempts_since_accept", 0)) + 1
    entry["last_verdict"] = CONFIRMATION_UNCERTAIN
    conflict = record.get("distance_conflict")
    checks = {
        "available": False,
        "reason": "candidate_distance_uncertain",
        "viewpoint_check_skipped": True,
        "candidate_distance_conflict": conflict,
        "candidate_distance_source": record.get("distance_source"),
    }
    source_id = str(record.get("observation_id") or "")
    action = _reground_view_action(ctx, record, instruction, turn_deg)
    data: dict[str, Any] = {
        "candidate_id": candidate_id,
        "instruction": instruction,
        "verdict": CONFIRMATION_UNCERTAIN,
        "missing": ["localization"],
        "observation_ids": [],
        "source_observation_ids": [source_id] if source_id else [],
        "views": [],
        "position_check": checks,
        "position_delta_m": None,
        "confirmation_count": int(entry.get("accepted", 0)),
        "new_confirmation": False,
        "repeated_evidence": False,
        "stalled": False,
        "attempts_since_accept": int(entry["attempts_since_accept"]),
        "evidence_revision": revision,
        "distance_source": record.get("distance_source"),
        "distance_conflict": conflict,
        "distance_uncertain_confirms": attempts,
        "suggested_viewpoints": action.get("viewpoints") or [],
        "unchecked_deg": action.get("unchecked_deg") or [],
        "steps": [],
        "vlm_calls": 0,
        "outer_tool_calls": 1,
        "inner_tool_calls": 0,
        "model_only_steps": 0,
        "cost": {
            "outer_tool_calls": 1,
            "inner_tool_calls": 0,
            "tool_calls": 0,
            "model_only_steps": 0,
            "vlm_calls": 0,
            "inner_steps": 0,
        },
        "progress_kind": PROGRESS_NONE,
        "next_action": action,
    }
    if attempts >= 2:
        # The candidate has now been offered for confirmation twice without a
        # new grounding. Stop the loop and make re-grounding mandatory: a
        # rejected record is not approached or confirmed again.
        entry["stalled"] = True
        target["needs_reground"] = True
        _set_candidate_status(
            ctx, candidate_id, CONFIRMATION_REJECTED,
            "confirm_stalled_distance_uncertain",
        )
        record_confirmation(
            ctx, candidate_id=candidate_id, verdict=CONFIRMATION_UNCERTAIN,
            source_observation_ids=[source_id] if source_id else [],
            missing=["localization"], reason="confirm_stalled", checks=checks,
            evidence=(
                [EvidenceRef("observation", source_id, "candidate source view")]
                if source_id else []
            ),
        )
        data.update(
            reason="confirm_stalled",
            stalled=True,
            candidate_needs_reground=True,
            strategy_change=_strategy_change(
                "confirm_stalled", candidate_id, ["localization"]
            ),
            hint=(
                "confirmation stalled: this candidate was already refused for an "
                "unresolved depth/model distance, so it is no longer usable - "
                "change the viewpoint and re-ground the target; do not confirm it again"
            ),
        )
        return ToolResult(STATUS_BLOCKED, data)
    _set_candidate_confirmation(
        ctx, candidate_id, CONFIRMATION_UNCERTAIN, revision, ["localization"], [],
    )
    record_confirmation(
        ctx, candidate_id=candidate_id, verdict=CONFIRMATION_UNCERTAIN,
        source_observation_ids=[source_id] if source_id else [],
        missing=["localization"], reason="candidate_distance_uncertain", checks=checks,
        evidence=(
            [EvidenceRef("observation", source_id, "candidate source view")]
            if source_id else []
        ),
    )
    data.update(
        reason="candidate_distance_uncertain",
        hint=(
            "the candidate's position is not trusted (depth and model range "
            "disagree), so no viewpoint check can confirm it: observe from a "
            "different heading and ground_target again; the same view would "
            "mint the same conflict"
        ),
        candidate_needs_reground=False,
    )
    return ToolResult(STATUS_BLOCKED, data)


def _override_yaw(world: Any, yaw_rad: Any) -> float | None:
    """Keep ``world`` at one view's heading for a depth read; returns the old yaw."""
    state = getattr(world, "state", None)
    if state is None or yaw_rad is None or not hasattr(state, "yaw_rad"):
        return None
    try:
        old = float(state.yaw_rad)
        state.yaw_rad = float(yaw_rad)
    except (TypeError, ValueError):
        return None
    write = getattr(world, "_write_pose", None)
    if callable(write):
        try:
            write()
        except Exception:  # noqa: BLE001 - the pose write is best effort
            pass
    return old


def _restore_yaw(world: Any, old: float | None) -> None:
    if old is None:
        return
    state = getattr(world, "state", None)
    if state is None:
        return
    try:
        state.yaw_rad = float(old)
    except (TypeError, ValueError):
        return
    write = getattr(world, "_write_pose", None)
    if callable(write):
        try:
            write()
        except Exception:  # noqa: BLE001 - restoring the pose is best effort
            pass


def _measure_box_from_view(
    ctx: ToolContext, target: Any, view_record: Any, observation_id: str,
    box: Any,
) -> dict:
    """Measure the object the model points at on one fresh frame.

    ``delta_m`` is the distance between two *independent* estimates: the
    candidate's own place (grounding) and the world point of the box the model
    drew on this view. Reading depth at the candidate's projected pixel is not
    an independent check - it returns the candidate point by construction (the
    v2 ``delta_m=0.0`` that accepted every wrong candidate). The model's box is
    measured at its centre; the bottom-edge samples are only a fallback when the
    centre depth is unusable. A box covering half the frame or more is refused:
    it does not localise the object, and its samples are dominated by the floor
    or wall in front of the drone.

    ``observe(mode="forward", yaw_deg=...)`` renders the view at a heading and
    then restores the drone's yaw; ``world.depth_at_pixel`` reads the current
    pose. The world is held at the view's heading for the read, so the frame and
    the depth map are the same capture.
    """
    world = getattr(ctx, "world", None)
    result: dict[str, Any] = {
        "available": False,
        "reason": "no_depth_measurement",
        "observation_id": observation_id,
        "method": "observed_box",
    }
    if box is None:
        result["reason"] = "no_box_from_view"
        return result
    frame = getattr(view_record, "frame", None)
    frame_width = int(frame.shape[-1]) if hasattr(frame, "shape") and len(frame.shape) >= 2 else None
    frame_height = int(frame.shape[-2]) if hasattr(frame, "shape") and len(frame.shape) >= 2 else None
    parsed, problem = parse_box(box, width=frame_width, height=frame_height)
    if parsed is None:
        result["reason"] = "invalid_box"
        result["box_problem"] = problem
        return result
    result["box"] = parsed.to_list()
    area_fraction = max(0.0, parsed.width) * max(0.0, parsed.height) / 1_000_000.0
    result["box_area_fraction"] = round(area_fraction, 3)
    if area_fraction >= 0.5:
        # a box covering most of the frame is not a measurement of the object:
        # its samples are the floor/wall in front of the drone, and comparing
        # that with the candidate is tautological again (holdout v3 seed37
        # accepted a shelf 14.7 m from the true one with a 70% box, seed41 a
        # shelf 18 m away with box [0,0,1000,1000])
        result["reason"] = "box_too_large"
        return result
    depth_fn = getattr(world, "depth_at_pixel", None)
    if not callable(depth_fn):
        result["reason"] = "no_depth_sensor"
        return result
    if getattr(view_record, "pitched", False):
        result["reason"] = "pitched_view"
        return result
    heading_rad = getattr(view_record, "yaw_rad", None)
    if heading_rad is None:
        result["reason"] = "view_heading_unknown"
        return result
    here = _position(world)
    view_position = getattr(view_record, "position_flu", None)
    if view_position is not None:
        try:
            moved = float(np.linalg.norm(np.asarray(view_position, dtype=np.float64) - here))
        except (TypeError, ValueError):
            moved = float("inf")
        if moved > 0.5:
            # the frame was not captured at the current position: measuring it
            # from here would compare two different poses
            result["reason"] = "view_pose_mismatch"
            return result
    if getattr(target, "position", None) is None:
        result["reason"] = "candidate_has_no_position"
        return result
    estimate = [float(v) for v in target.position]
    result["candidate_estimate_flu"] = estimate

    centre = parsed.center
    ground_y = min(
        1000.0, max(0.0, parsed.y2 - max(2.0, 0.03 * parsed.height))
    )
    points = [
        ("centre", centre),
        ("bottom_25", (parsed.x1 + 0.25 * parsed.width, ground_y)),
        ("bottom_centre", (parsed.x1 + 0.50 * parsed.width, ground_y)),
        ("bottom_75", (parsed.x1 + 0.75 * parsed.width, ground_y)),
    ]
    old_yaw = _override_yaw(world, heading_rad)
    try:
        refresh = getattr(world, "refresh_sensing", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:  # noqa: BLE001 - depth is best effort
                pass
        samples: list[dict] = []
        for name, pixel in points:
            sample = _depth_point(ctx, view_record, pixel, depth_fn)
            if sample is not None:
                samples.append({"sample": name, **sample})
    finally:
        _restore_yaw(world, old_yaw)
    if not samples:
        result["reason"] = "no_depth_measurement"
        return result
    # the box centre is the object's own point; the support samples are only a
    # fallback when the centre depth is unusable. Taking the nearest sample
    # instead let a large box match the floor in front of the drone to any
    # candidate (the v3 seed37/41 false accepts).
    chosen = next(
        (item for item in samples if item["sample"] == "centre"),
        min(samples, key=lambda item: item["depth_m"]),
    )
    measured = np.asarray(chosen["position_flu"], dtype=np.float64)
    delta = float(np.linalg.norm(measured - np.asarray(estimate, dtype=np.float64)))
    depths = [float(item["depth_m"]) for item in samples]
    result.update(
        available=True,
        delta_m=round(delta, 2),
        tolerance_m=CONFIRM_POSITION_TOL_M,
        reject_m=CONFIRM_POSITION_REJECT_M,
        reason="observed_box_depth",
        measured_position_flu=chosen["position_flu"],
        measured_pixel=chosen["pixel"],
        depth_m=round(float(chosen["depth_m"]), 3),
        chosen_sample=chosen["sample"],
        depth_spread_m=round(max(depths) - min(depths), 3),
        samples=samples,
    )
    return result


def _view_supports(view: Mapping, relation: str) -> bool:
    """One independent view's positive identity evidence.

    A view supports only when it says target **and** same instance, does not
    deny the attributes, - when an anchor is named - sees the anchor in that
    relation, and its confidence is decided and at least
    ``MIN_CONFIRM_CONFIDENCE``. An explicit low/0.0 view is not support and
    vetoes acceptance (seed 17/19 accepted with one 0.9 and one 0.0 view); an
    undecided (``null``) view is not support either, though it does not veto.
    """
    if view.get("error"):
        return False
    if view.get("target") is not True:
        return False
    if view.get("same_instance") is not True:
        return False
    if view.get("attributes_match") is False:
        return False
    if relation and view.get("relation_supported") is not True:
        return False
    state = str(view.get("confidence_state") or _confidence_state(view.get("confidence")))
    return state == "ok"


def _view_blocks(view: Mapping, relation: str) -> bool:
    """Whether one view vetoes an acceptance the other views support.

    Only *decided* evidence vetoes: an explicit ``0.0``/low confidence, a
    decided view that does not support, or an explicit negative property
    (different instance / attributes / absent target). An undecided view
    (``confidence: null``) neither supports nor vetoes - it cannot carry an
    acceptance on its own, but it must not close the door on the views that do.
    """
    if view.get("error"):
        return True
    if view.get("same_instance") is False:
        return True
    if view.get("attributes_match") is False:
        return True
    if view.get("target") is False:
        return True
    if relation and view.get("relation_supported") is False:
        return True
    state = str(view.get("confidence_state") or _confidence_state(view.get("confidence")))
    if state == "low":
        return True
    return state == "ok" and not _view_supports(view, relation)


# ---------------------------------------------------------------------------
# the routine
# ---------------------------------------------------------------------------


def confirm_candidate(
    ctx: ToolContext,
    args: dict,
    *,
    registry: Any = None,
    vlm: Any = None,
    call: Any = None,
) -> ToolResult:
    """Run the bounded confirmation routine for one candidate; see module doc."""
    args = dict(args or {})
    try:
        from .skills_search import _RoutineBudget, _ToolCaller
    except Exception:  # noqa: BLE001 - the skill still runs without the helpers
        _RoutineBudget = None  # type: ignore[assignment]
        _ToolCaller = None  # type: ignore[assignment]

    scope = task_scope(ctx)
    candidate_id = str(args.get("candidate_id") or scope.get("candidate_id") or "")
    store = _candidate_store(ctx)
    if not candidate_id:
        live = [
            record for record in store.values()
            if isinstance(record, Mapping)
            and str(record.get("status") or "") != CONFIRMATION_REJECTED
            and not record.get("instance_suspect")
            and record.get("id")
        ]
        if live:
            candidate_id = str(live[-1].get("id"))
    if not candidate_id:
        return ToolResult(
            STATUS_EMPTY,
            {
                "reason": "no_candidate",
                "verdict": CONFIRMATION_UNCERTAIN,
                "missing": ["identity"],
                "hint": "ground the target first (ground_target), then confirm the cand_N it returns",
                "next_action": _next_action(
                    "ground_target",
                    {"instruction": _instruction(ctx, args)},
                    "there is no candidate instance to confirm yet",
                ),
            },
        )
    record = skills_approach.candidate_record(ctx, candidate_id)
    if record is None:
        return ToolResult(
            STATUS_FAILED,
            {
                "reason": "unknown_candidate",
                "candidate_id": candidate_id,
                "available": sorted(str(key) for key in store)[-8:],
                "verdict": CONFIRMATION_UNCERTAIN,
                "missing": ["identity"],
                "hint": "that id does not name a grounded candidate; ground_target returns the ids that do",
            },
        )
    if str(record.get("status") or "") == CONFIRMATION_REJECTED:
        scope = task_scope(ctx)
        return ToolResult(
            STATUS_BLOCKED,
            {
                "reason": "candidate_rejected",
                "candidate_id": candidate_id,
                "verdict": CONFIRMATION_REJECTED,
                "missing": ["identity"],
                "evidence_revision": int(scope.get("evidence_revision", 0)),
                "hint": "that candidate was rejected by an earlier confirmation; re-ground the target instead of using it again",
                "next_action": _next_action(
                    "ground_target",
                    {"instruction": _instruction(ctx, args)},
                    "a rejected candidate is not re-used: ground the target again",
                ),
            },
        )

    if bool(record.get("instance_suspect")):
        # the box failed the grounding-time instance check: a confirmation from
        # a new view cannot turn the wall/background into the instructed object
        return ToolResult(
            STATUS_BLOCKED,
            {
                "reason": "candidate_instance_suspect",
                "candidate_id": candidate_id,
                "verdict": CONFIRMATION_REJECTED,
                "missing": ["identity"],
                "instance_suspect_reason": str(
                    record.get("instance_suspect_reason") or ""
                ),
                "hint": (
                    "that candidate is not the instructed instance (wall, "
                    "background or a different object); ground the target again "
                    "from another view instead of confirming it"
                ),
                "next_action": _next_action(
                    "ground_target",
                    {"instruction": _instruction(ctx, args)},
                    "a suspect instance is not confirmed: re-ground the target",
                ),
            },
        )

    instruction = _instruction(ctx, args)
    turn_deg = _clamp(
        _as_float(args.get("turn_deg"), DEFAULT_TURN_DEG) or DEFAULT_TURN_DEG,
        10.0, MAX_TURN_DEG,
    )
    if _distance_unresolved(record):
        # The candidate's range came from a depth/model conflict. A viewpoint
        # check cannot settle which range is the object's, and the v4 holdout
        # showed the loop: confirm -> propose_viewpoints -> go_to -> confirm on
        # the same stale candidate. Refuse before flying/taking views and hand
        # back a forced re-ground from a different viewpoint.
        return _distance_uncertain_confirmation(
            ctx, record, candidate_id, instruction, turn_deg
        )
    select_scope_candidate(ctx, candidate_id)
    relation, relation_source = _relation_anchor(ctx, args, instruction)
    max_views = int(min(max(_as_int(args.get("max_views"), DEFAULT_CONFIRM_VIEWS), 1), MAX_CONFIRM_VIEWS))
    max_steps = int(min(max(_as_int(args.get("max_steps"), DEFAULT_CONFIRM_STEPS), 1), MAX_CONFIRM_STEPS))
    standoff = _as_float(args.get("standoff_m"), CONFIRM_STANDOFF_M)
    standoff = CONFIRM_STANDOFF_M if standoff is None else max(0.0, standoff)
    backend = vlm if callable(getattr(vlm, "decide", None)) else _vlm_from_ctx(ctx)

    caller = (
        _ToolCaller(ctx, registry=registry, vlm=vlm, call=call)
        if _ToolCaller is not None else None
    )
    if caller is None:  # pragma: no cover - skills_search is part of the package
        return ToolResult(STATUS_FAILED, {"error": "confirmation_helpers_unavailable"})

    def run() -> ToolResult:
        return _run_confirmation(
            ctx, args, candidate_id=candidate_id, record=dict(record),
            instruction=instruction, relation=relation,
            relation_source=relation_source, turn_deg=turn_deg,
            max_views=max_views, max_steps=max_steps, standoff=standoff,
            backend=backend, caller=caller,
        )

    if _RoutineBudget is None:  # pragma: no cover - defensive
        return run()
    with _RoutineBudget(ctx, inner_steps=max_steps, outer_cost=1):
        return run()


def _run_confirmation(
    ctx: ToolContext,
    args: dict,
    *,
    candidate_id: str,
    record: dict,
    instruction: str,
    relation: str,
    turn_deg: float,
    max_views: int,
    max_steps: int,
    standoff: float,
    backend: Any,
    caller: Any,
    relation_source: str = "",
) -> ToolResult:
    from .skills_search import _record

    steps: list[dict] = []
    vlm_calls = 0
    entry = _ledger_entry(ctx, candidate_id)
    scope = task_scope(ctx)
    revision = int(scope.get("evidence_revision", 0))
    if entry.get("revision") != revision:
        # a new candidate/revision is a new identity question: the attempt
        # counters of the previous instance do not carry over
        entry["revision"] = revision
        entry["attempts"] = 0
        entry["attempts_since_accept"] = 0
        entry["no_progress_streak"] = 0
        entry["stalled"] = False
    target = skills_approach.candidate_target(record, candidate_id)
    source_observation = str(record.get("observation_id") or "")
    evidence: list[EvidenceRef] = []
    if source_observation:
        evidence.append(
            EvidenceRef("observation", source_observation, "candidate source view")
        )
        # the grounding image cannot confirm itself: if a fresh capture shows
        # those same pixels again, it is the same evidence, not a new view
        source_record = resolve_observation(ctx, source_observation)
        source_digest = _frame_digest(getattr(source_record, "frame", None))
        if source_digest:
            source_key = _evidence_key(source_observation, source_digest)
            if source_key not in entry.get("evidence_keys", []):
                entry.setdefault("evidence_keys", []).append(source_key)
                del entry["evidence_keys"][:-MAX_EVIDENCE_KEYS]

    def steps_left() -> bool:
        return caller.tool_calls + vlm_calls < max_steps

    def result(
        verdict: str, missing: list[str], reason: str, next_action: dict | None,
        *, status: str = STATUS_OK, views: list[dict] | None = None,
        position_check: dict | None = None, repeated: bool = False,
        new_confirmation: bool = False, stalled: bool = False,
        strategy_change: dict | None = None,
    ) -> ToolResult:
        views = list(views or [])
        observation_ids = [str(view.get("obs_id")) for view in views if view.get("obs_id")]
        data: dict[str, Any] = {
            "candidate_id": candidate_id,
            "instruction": instruction,
            "relation": relation or None,
            "relation_anchor": relation or None,
            "relation_source": relation_source or None,
            "verdict": verdict,
            "missing": list(missing),
            "reason": reason,
            "observation_ids": observation_ids,
            "source_observation_ids": [source_observation] if source_observation else [],
            "views": views,
            "position_check": position_check or {"available": False},
            "position_delta_m": (position_check or {}).get("delta_m"),
            "confirmation_count": int(entry.get("accepted", 0)),
            "new_confirmation": bool(new_confirmation),
            "repeated_evidence": bool(repeated),
            "stalled": bool(stalled),
            "attempts_since_accept": int(entry.get("attempts_since_accept", 0)),
            "evidence_revision": revision,
            "steps": steps,
            "vlm_calls": vlm_calls,
            "outer_tool_calls": 1,
            "inner_tool_calls": caller.tool_calls,
            "model_only_steps": vlm_calls,
            "cost": {
                "outer_tool_calls": 1,
                "inner_tool_calls": caller.tool_calls,
                "tool_calls": caller.tool_calls,
                "model_only_steps": vlm_calls,
                "vlm_calls": vlm_calls,
                "inner_steps": max_steps,
            },
        }
        if verdict == CONFIRMATION_ACCEPTED:
            data.update(
                progress_kind=PROGRESS_TASK,
                confidence=min(
                    (
                        view["confidence"] for view in views
                        if view.get("target") is True
                        and view.get("confidence") is not None
                    ),
                    default=None,
                ),
                hint="confirmed: the candidate is the instructed instance under this revision - finish may now count it",
            )
        elif verdict == CONFIRMATION_REJECTED:
            data.update(
                progress_kind=PROGRESS_INFORMATION,
                hint="rejected: this candidate is marked rejected and will not be approached or counted by finish; re-ground the target",
            )
        elif repeated:
            data.update(
                progress_kind=PROGRESS_NONE,
                hint="the same image(s) were already used for confirmation: the count is unchanged - take a different heading or pose",
            )
        else:
            data.update(
                progress_kind=PROGRESS_INFORMATION,
                hint=_uncertain_hint(missing),
            )
        if stalled:
            data.update(
                stalled=True,
                strategy_change=strategy_change or _strategy_change(
                    reason, candidate_id, missing
                ),
                progress_kind=PROGRESS_INFORMATION,
                hint=(
                    "confirmation stalled: the same candidate has not produced a "
                    "new accepted confirmation - stop repeating confirm_candidate "
                    "and change the strategy (viewpoint / anchor view / candidate)"
                ),
            )
        if next_action is not None:
            data["next_action"] = next_action
        return ToolResult(status, data, evidence)

    def stalled_result(
        missing: list[str], reason: str, next_action: dict | None,
        *, views: list[dict] | None = None,
        position_check: dict | None = None,
    ) -> ToolResult:
        """Stop re-confirming the same candidate and hand back a different action."""
        entry["last_verdict"] = CONFIRMATION_UNCERTAIN
        entry["stalled"] = True
        _set_candidate_confirmation(
            ctx, candidate_id, CONFIRMATION_UNCERTAIN, revision, missing,
            [str(view.get("obs_id")) for view in (views or []) if view.get("obs_id")],
        )
        record_confirmation(
            ctx, candidate_id=candidate_id, verdict=CONFIRMATION_UNCERTAIN,
            observation_ids=[
                str(view.get("obs_id")) for view in (views or []) if view.get("obs_id")
            ],
            source_observation_ids=[source_observation] if source_observation else [],
            position_delta_m=(position_check or {}).get("delta_m"),
            position_tolerance_m=CONFIRM_POSITION_TOL_M,
            missing=missing, reason=reason,
            checks=position_check, evidence=evidence,
        )
        action = next_action or _change_view_action(missing, turn_deg)
        if isinstance(action, Mapping) and action.get("tool") == "confirm_candidate":
            action = _change_view_action(missing, turn_deg)
        return result(
            CONFIRMATION_UNCERTAIN, missing, reason, action,
            status=STATUS_BLOCKED, views=views, position_check=position_check,
            stalled=True,
            strategy_change=_strategy_change(reason, candidate_id, missing),
        )

    def uncertain(missing: list[str], reason: str, next_action: dict | None,
                  *, views: list[dict] | None = None,
                  position_check: dict | None = None) -> ToolResult:
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        entry["attempts_since_accept"] = int(entry.get("attempts_since_accept", 0)) + 1
        if views:
            entry["no_progress_streak"] = 0
        if views and int(entry["attempts_since_accept"]) >= MAX_CONFIRM_ATTEMPTS:
            # N judged confirmations in a row without an acceptance: the
            # relation branch used to point back at the same candidate forever
            return stalled_result(
                missing, "confirm_stalled",
                _change_view_action(missing, turn_deg),
                views=views, position_check=position_check,
            )
        entry["last_verdict"] = CONFIRMATION_UNCERTAIN
        _set_candidate_confirmation(
            ctx, candidate_id, CONFIRMATION_UNCERTAIN, revision, missing,
            [str(view.get("obs_id")) for view in (views or []) if view.get("obs_id")],
        )
        record_confirmation(
            ctx, candidate_id=candidate_id, verdict=CONFIRMATION_UNCERTAIN,
            observation_ids=[str(view.get("obs_id")) for view in (views or []) if view.get("obs_id")],
            source_observation_ids=[source_observation] if source_observation else [],
            position_delta_m=(position_check or {}).get("delta_m"),
            position_tolerance_m=CONFIRM_POSITION_TOL_M,
            missing=missing, reason=reason,
            checks=position_check, evidence=evidence,
        )
        return result(
            CONFIRMATION_UNCERTAIN, missing, reason, next_action, views=views,
            position_check=position_check,
        )

    # -- 0. the same candidate has already used up its confirmation attempts --
    if int(entry.get("attempts_since_accept", 0)) >= MAX_CONFIRM_ATTEMPTS:
        return stalled_result(
            ["identity", "localization"], "confirm_stalled",
            _change_view_action(["identity", "localization"], turn_deg),
        )

    # -- 0b. no backend: identity cannot be judged ---------------------------
    if backend is None:
        return uncertain(
            ["identity"], "no_vlm_backend",
            _identity_action(candidate_id, instruction, -turn_deg, relation),
        )
    if not target.direction_known:
        # the candidate has neither a point nor a ray: there is no distance to
        # close and no direction to look along, so re-ground it first
        return uncertain(
            ["distance", "identity"], "candidate_has_no_position",
            _next_action(
                "ground_target",
                {"instruction": instruction},
                "the candidate carries neither a position nor a ray: ground the "
                "target again from the current view instead of guessing a place",
            ),
        )

    # -- 1. approach to confirmation distance --------------------------------
    legs = 0
    while steps_left() and legs < MAX_CONFIRM_LEGS:
        here = _position(ctx.world)
        distance = (
            None if target.position is None
            else float(np.linalg.norm(target.position - here))
        )
        if distance is not None and distance <= standoff + CONFIRM_ARRIVE_RADIUS_M:
            break
        moved = caller(
            "approach_candidate",
            {"candidate_id": candidate_id, "standoff_m": round(standoff, 2)},
        )
        outcome = str(moved.data.get("outcome") or moved.data.get("task_status") or "")
        _record(
            steps,
            "approach_candidate",
            (
                "close to the confirmation standoff so the candidate can be inspected"
                if target.position is not None
                else "the candidate is a ray: fly one bounded step to gain a range"
            ),
            [str(moved.data.get("last_observation_id") or "")],
            {
                "status": moved.status,
                "outcome": outcome,
                "progress_m": moved.data.get("progress_m"),
                "remaining_m": moved.data.get("remaining_m"),
            },
        )
        if outcome == "already_at_pose":
            break
        if moved.status in (STATUS_BLOCKED, STATUS_FAILED) or outcome in (
            "collision", "stalled", "planner_failed", "no_progress", "outside_bound",
            "leg_timeout",
        ):
            return uncertain(
                ["path"], outcome or "path_blocked",
                _path_action(moved, candidate_id),
            )
        legs += 1
        record = skills_approach.candidate_record(ctx, candidate_id) or record
        target = skills_approach.candidate_target(record, candidate_id)
        if outcome == "arrived":
            break

    here = _position(ctx.world)
    distance = (
        None if target.position is None
        else float(np.linalg.norm(target.position - here))
    )
    if target.position is None:
        return uncertain(
            ["distance"], "candidate_has_no_range",
            _distance_action(candidate_id, instruction, target, turn_deg),
        )
    if distance is not None and distance > standoff + CONFIRM_ARRIVE_RADIUS_M:
        return uncertain(
            ["distance"], "not_at_confirmation_distance",
            _distance_action(candidate_id, instruction, target, turn_deg),
        )

    # -- 2/3. new views + one model judgement per independent view -----------
    offsets = _offsets_for(target, ctx.world, turn_deg)
    views: list[dict] = []
    position_check: dict = {"available": False}
    position_attempts: list[dict] = []
    measured = False
    attempted_keys: set[str] = set()
    for index, offset in enumerate(offsets):
        if len(views) >= max_views or not steps_left():
            break
        captured = caller(
            "observe", {"mode": "forward", "yaw_deg": round(float(offset), 1)}
        )
        obs_id = str(captured.data.get("observation_id") or "")
        view_record = resolve_observation(ctx, obs_id) if obs_id else None
        heading = _as_float(captured.data.get("yaw_deg"), None)
        _record(
            steps,
            "observe",
            f"take a new view at {round(float(offset), 1)} deg from the current heading",
            [obs_id],
            {
                "status": captured.status,
                "yaw_deg": heading,
                "mode": "forward",
            },
        )
        if captured.status != STATUS_OK or view_record is None:
            continue
        digest = _frame_digest(getattr(view_record, "frame", None))
        key = _evidence_key(obs_id, digest)
        if key in attempted_keys or key in entry.get("evidence_keys", []):
            _record(
                steps,
                "repeat_view",
                "the same image was already used for confirmation: a new id is not new evidence",
                [obs_id],
                {"repeated": True, "evidence_key": key[:24]},
            )
            attempted_keys.add(key)
            continue
        attempted_keys.add(key)
        entry.setdefault("evidence_keys", []).append(key)
        del entry["evidence_keys"][:-MAX_EVIDENCE_KEYS]
        entry.setdefault("observation_ids", []).append(obs_id)
        del entry["observation_ids"][:-MAX_EVIDENCE_KEYS]

        verdict = _ask_view(
            ctx, backend, instruction, relation, candidate_id, record,
            {
                "obs_id": obs_id,
                "frame": view_record.frame,
                "heading_deg": heading,
                "pose_flu": None if view_record.position_flu is None else list(view_record.position_flu),
                "intrinsics": dict(view_record.intrinsics),
            },
            distance,
            relation_source=relation_source,
        )
        vlm_calls += int(verdict.get("vlm_calls") or 1)

        # location consistency: the model names the object's box on *this*
        # fresh frame, and that box - not the candidate's own projected pixel -
        # is measured. Reading depth at the candidate's pixel returned the
        # candidate point by construction (delta 0.0 for any candidate), which
        # accepted all five wrong instances in holdout v2. A failed reading is
        # retried from the next view; the first genuine measurement wins.
        if not measured and not getattr(view_record, "pitched", False):
            measurement = _measure_box_from_view(
                ctx, target, view_record, obs_id,
                verdict.get("box") if verdict.get("target") is not False else None,
            )
            position_attempts.append(dict(measurement))
            if measurement.get("available"):
                measured = True
                position_check = dict(measurement)
            elif (
                not position_check.get("available")
                and not position_check.get("inconsistent")
            ):
                position_check = dict(measurement)
            _record(
                steps,
                "relocalize",
                "measure the box the model points at on this fresh frame and "
                "compare that observed point with the candidate estimate",
                [obs_id],
                dict(measurement),
            )
            record = skills_approach.candidate_record(ctx, candidate_id) or record
            target = skills_approach.candidate_target(record, candidate_id)
        view_entry: dict[str, Any] = {
            "view": index + 1,
            "obs_id": obs_id,
            "yaw_deg": heading,
            "target": verdict.get("target"),
            "same_instance": verdict.get("same_instance"),
            "attributes_match": verdict.get("attributes_match"),
            "relation_supported": verdict.get("relation_supported"),
            "confidence": verdict.get("confidence"),
            "confidence_state": verdict.get("confidence_state")
            or _confidence_state(verdict.get("confidence")),
            "reason": verdict.get("reason") or verdict.get("error") or "",
            "missing": list(verdict.get("missing") or []),
            "next_view": verdict.get("next_view") or "",
            "source_frame_attached": bool(verdict.get("source_frame")),
            "box": verdict.get("box"),
            "box_problem": verdict.get("box_problem") or "",
            "box_retry": bool(verdict.get("box_retry")),
            "box_retry_mode": verdict.get("box_retry_mode") or "",
            "box_retry_region": verdict.get("box_retry_region"),
            "independent": True,
        }
        if verdict.get("error"):
            view_entry["error"] = str(verdict["error"])
        if position_check.get("observation_id") == obs_id:
            view_entry["position_delta_m"] = position_check.get("delta_m")
        views.append(view_entry)
        evidence.append(EvidenceRef("observation", obs_id, "confirmation view"))

    # -- 4. fuse the per-view judgements ------------------------------------
    if not views:
        if entry.get("accepted") and int(entry.get("accepted_revision") or -1) == revision:
            return result(
                CONFIRMATION_ACCEPTED, [], "repeated_evidence", None,
                status=STATUS_CACHED, repeated=True,
            )
        # no new pixels: another confirm of the same candidate cannot add
        # evidence. Stop the repeat loop and give the caller a different action
        # (a different heading/pose or another candidate).
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        entry["attempts_since_accept"] = int(entry.get("attempts_since_accept", 0)) + 1
        entry["no_progress_streak"] = int(entry.get("no_progress_streak", 0)) + 1
        missing = ["identity", "localization"]
        return stalled_result(
            missing, "confirm_stalled",
            _change_view_action(missing, turn_deg),
        )

    if position_attempts:
        position_check = dict(position_check)
        position_check["attempts"] = [
            {
                "observation_id": attempt.get("observation_id"),
                "reason": attempt.get("reason"),
                "available": bool(attempt.get("available")),
                "delta_m": attempt.get("delta_m"),
            }
            for attempt in position_attempts
        ]
    anchor_check = _subtask_anchor_check(ctx, record)
    position_check = dict(position_check)
    position_check["subtask_anchor"] = anchor_check
    anchor_ok = not anchor_check.get("available") or bool(
        anchor_check.get("consistent")
    )
    strong_bad = [
        view for view in views
        if view.get("same_instance") is False or view.get("attributes_match") is False
    ]
    relation_bad = [
        view for view in views
        if relation and view.get("relation_supported") is False
    ]
    support = [view for view in views if _view_supports(view, relation)]
    blockers = [view for view in views if _view_blocks(view, relation)]
    # at least one decided-supporting view and no decided veto: undecided
    # (null-confidence) views are neutral, explicit low/0.0 views are not
    appearance_ok = bool(support) and not blockers
    delta = position_check.get("delta_m") if position_check.get("available") else None
    position_available = bool(position_check.get("available"))
    position_ok = position_available and delta is not None and delta <= CONFIRM_POSITION_TOL_M

    def reject(reason: str) -> ToolResult:
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        entry["last_verdict"] = CONFIRMATION_REJECTED
        _set_candidate_status(ctx, candidate_id, CONFIRMATION_REJECTED, reason)
        # a triangulated point the independent measurement refused is not
        # re-minted from the same ray pair (v6 bedroom seed37 repeated it 9x)
        _remember_rejected_triangulation(ctx, record, reason)
        record_confirmation(
            ctx, candidate_id=candidate_id, verdict=CONFIRMATION_REJECTED,
            observation_ids=[str(view.get("obs_id")) for view in views],
            source_observation_ids=[source_observation] if source_observation else [],
            position_delta_m=delta,
            position_tolerance_m=CONFIRM_POSITION_TOL_M,
            missing=["identity"], reason=reason, checks=position_check,
            evidence=evidence,
        )
        return result(
            CONFIRMATION_REJECTED, ["identity"], reason,
            _next_action(
                "ground_target",
                {"instruction": instruction},
                "the rejected candidate is not the instructed instance: ground the "
                "target again instead of using this place",
            ),
            views=views, position_check=position_check,
        )

    def accept() -> ToolResult:
        entry["accepted"] = int(entry.get("accepted", 0)) + 1
        entry["last_verdict"] = CONFIRMATION_ACCEPTED
        entry["accepted_revision"] = revision
        entry["attempts_since_accept"] = 0
        entry["no_progress_streak"] = 0
        entry["stalled"] = False
        _set_candidate_confirmation(
            ctx, candidate_id, CONFIRMATION_ACCEPTED, revision, [],
            [str(view.get("obs_id")) for view in views],
        )
        record_confirmation(
            ctx, candidate_id=candidate_id, verdict=CONFIRMATION_ACCEPTED,
            observation_ids=[str(view.get("obs_id")) for view in views],
            source_observation_ids=[source_observation] if source_observation else [],
            position_delta_m=delta,
            position_tolerance_m=CONFIRM_POSITION_TOL_M,
            missing=[], reason="accepted", checks=position_check,
            evidence=evidence,
        )
        return result(
            CONFIRMATION_ACCEPTED, [], "accepted", None,
            views=views, position_check=position_check,
            new_confirmation=True,
        )

    # -- 4. verdict ---------------------------------------------------------
    if strong_bad:
        return reject("different_instance")
    if position_check.get("inconsistent") and not position_available:
        return reject("position_inconsistent")
    if delta is not None and delta > CONFIRM_POSITION_REJECT_M:
        return reject("position_inconsistent")
    conflict_info = record.get("distance_conflict") if isinstance(record, Mapping) else None
    distance_conflicted = bool(
        str(record.get("distance_source") or "") == "conflict"
        or record.get("distance_uncertain") is True
        or (
            isinstance(conflict_info, Mapping)
            and conflict_info.get("resolved") is False
        )
    )
    if distance_conflicted:
        # the candidate's place came from a depth/model disagreement that the
        # grounding step could not resolve: a fresh view cannot turn that into
        # an accepted identity - re-ground the object where its support point
        # is measurable
        checks = dict(position_check)
        checks["candidate_distance_conflict"] = conflict_info
        checks["candidate_distance_source"] = record.get("distance_source")
        missing = ["localization"]
        return uncertain(
            missing, "candidate_distance_conflict",
            _next_action(
                "propose_viewpoints",
                {"count": 3},
                "the candidate's grounding range was contradictory (depth vs "
                "model): move to a different viewpoint and re-ground the target "
                "where its support point is measurable - repeating ground_target "
                "on the same view returns the same conflict",
            ),
            views=views, position_check=checks,
        )
    if appearance_ok and position_ok and anchor_ok:
        return accept()
    if appearance_ok and position_ok and not anchor_ok:
        # the instance and the fresh measurement agree, but the candidate sits
        # outside the place the sub-task's own anchor records: at most open,
        # never accepted (v6 accepted two such placements)
        return uncertain(
            ["location"], "outside_subtask_anchor",
            _next_action(
                "ground_target",
                {"instruction": instruction},
                "the candidate is outside the current sub-task's anchor/goal "
                "tolerance: re-ground the stage's object from a view that also "
                "shows the anchor, instead of accepting this place",
            ),
            views=views, position_check=position_check,
        )
    if appearance_ok:
        # appearance is agreed but the location could not be measured: without
        # the second, independent measurement the verdict stays open
        missing = ["distance", "localization"]
        return uncertain(
            missing, "localization_not_available",
            _change_view_action(missing, turn_deg),
            views=views, position_check=position_check,
        )
    if delta is not None and delta > CONFIRM_POSITION_TOL_M:
        return uncertain(
            ["identity", "distance", "localization"], "position_too_far_from_candidate",
            _identity_action(candidate_id, instruction, -turn_deg, relation),
            views=views, position_check=position_check,
        )
    if relation_bad:
        missing = ["relation"] + ([] if support else ["identity"])
        return uncertain(
            missing, "relation_not_supported",
            _relation_action(candidate_id, instruction, relation, turn_deg),
            views=views, position_check=position_check,
        )
    if support:
        # some view(s) support but not all: the 0.0/低-confidence/undecided view
        # keeps the identity open instead of being ignored (seed 17/19)
        return uncertain(
            ["identity"], "not_all_views_support",
            _identity_action(candidate_id, instruction, -turn_deg, relation),
            views=views, position_check=position_check,
        )
    primary = views[0]
    primary_state = str(
        primary.get("confidence_state") or _confidence_state(primary.get("confidence"))
    )
    if primary.get("target") is False and primary_state == "ok":
        # a *decided* absence is a rejection; ``null`` is undecided and stays
        # an open question instead of condemning the candidate
        return reject("target_not_visible_from_confirmation_view")
    return uncertain(
        ["identity"], "identity_not_resolved",
        _identity_action(candidate_id, instruction, -turn_deg, relation),
        views=views, position_check=position_check,
    )


def _uncertain_hint(missing: list[str]) -> str:
    if "localization" in missing:
        return (
            "not confirmed yet: the candidate could not be re-measured on the fresh "
            "frame - take a view that holds it and confirm again (appearance alone "
            "cannot accept)"
        )
    if "distance" in missing:
        return "not confirmed yet: close the remaining distance (approach_candidate) and confirm again"
    if "relation" in missing:
        return "not confirmed yet: the relation anchor was not supported - take a view that holds both objects"
    if "path" in missing:
        return "not confirmed yet: the route is blocked; keep the candidate identity and change the route"
    return "not confirmed yet: the identity is unresolved - take a different view, not the same frame"


def register_confirm_tools(registry: ToolRegistry, vlm: Any = None) -> None:
    """Register ``confirm_candidate`` (the version-2 identity confirmation)."""
    if "confirm_candidate" in registry.names():
        return
    registry.register(
        ToolSpec(
            name="confirm_candidate",
            description=(
                "Confirm that a reached candidate cand_N is the instructed instance: "
                "approach to inspection distance, capture NEW views (different "
                "heading/pose), have the model compare the target's attributes and the "
                "instruction's relation anchor in each view, check the fresh depth "
                "measurement against the candidate position, and answer verdict "
                "accepted|uncertain|rejected. Acceptance requires at least one "
                "independent view to support the target/instance/attributes and the "
                "fresh depth measurement to agree with the candidate while no "
                "decided view vetoes it (an explicit 0.0/low-confidence or opposing "
                "view blocks it; a null-confidence view is undecided, and "
                "unavailable depth keeps it uncertain). "
                "uncertain carries missing="
                "distance|identity|relation|path|localization and a directed "
                "next_action (never a "
                "generic re-search); rejected marks the candidate unusable. Re-using "
                "the same image does not increase the confirmation count. Returns "
                "steps, vlm_calls and cost like search_target. A movement stage counts "
                "as complete only with an accepted confirmation under the current "
                "candidate revision."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "candidate_id": {
                        "type": "string",
                        "description": "cand_N to confirm (default: the current scope's candidate)",
                    },
                    "instruction": {
                        "type": "string",
                        "description": "what the target is, in the task's words (default: the current sub-goal/instruction)",
                    },
                    "relation": {
                        "type": "string",
                        "description": "relation anchor to check (default: parsed from the instruction, e.g. 'next to')",
                    },
                    "max_views": {
                        "type": "integer", "minimum": 1, "maximum": MAX_CONFIRM_VIEWS,
                        "description": "new views one call may judge (default 2)",
                    },
                    "turn_deg": {
                        "type": "number",
                        "description": "heading offset between confirmation views (default 35)",
                    },
                    "standoff_m": {
                        "type": "number",
                        "description": "inspection distance from the candidate (default 1.2 m)",
                    },
                    "max_steps": {
                        "type": "integer", "minimum": 1, "maximum": MAX_CONFIRM_STEPS,
                        "description": "inner tool calls + model calls one confirmation may spend (default 8)",
                    },
                },
                "required": [],
            },
            handler=lambda ctx, args: confirm_candidate(
                ctx, args, registry=registry, vlm=vlm
            ),
            side_effects=True,
            cacheable=False,
        )
    )


__all__ = [
    "CONFIRM_ARRIVE_RADIUS_M",
    "CONFIRM_POSITION_REJECT_M",
    "CONFIRM_POSITION_TOL_M",
    "CONFIRM_STANDOFF_M",
    "CONFIRMATION_LEDGER_KEY",
    "DEFAULT_CONFIRM_VIEWS",
    "MAX_CONFIRM_VIEWS",
    "MIN_CONFIRM_CONFIDENCE",
    "confirm_candidate",
    "register_confirm_tools",
]
