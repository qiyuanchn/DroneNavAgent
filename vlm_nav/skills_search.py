"""The search skill: sweep, look, ground, confirm - and move when nothing is found.

``search_target(instruction, budget_turns?)`` is the *routine* half of "find the
object the instruction names". The model decides **what** to look for, when to
give up and whether a viewpoint is worth the flight; this module executes the
routine that turns those decisions into evidence:

1. **sweep** - ``observe(mode="lookaround")`` so every direction is on record,
   and read the coverage the capture reports (which 45 deg sectors a level view
   checked, which are unchecked, which views are worth a second look).
2. **compare the views** - when the instruction names the target by *where it
   is* ("the red object next to the lamp"), no single view is a decision:
   the relation is only readable in a frame that holds both objects, and the
   frame a one-shot grounding call would use may hold one of them. The routine
   therefore hands **every swept view to one model call** (``compare_views``,
   index + bearing per frame) and gets back a verdict per view - does it look
   like the target, why, how sure, and where in the frame. A recalled frame
   (``observation_ids``, e.g. the ids a ``query_memory`` hit reported) is
   compared the same way, so "compare before deciding" applies to the evidence
   the archive hands back and not only to a fresh sweep.
3. **choose a view** - the views the comparison selected (its confidence order)
   are grounded first, then the views the coverage flagged and the level views
   closest to "ahead"; a view is grounded, not guessed. A picked view the drone
   does not already face is *turned to* first (``return_to_observation`` restores
   the heading that frame was shot at) and the capture taken there is what is
   grounded: a depth map belongs to the frame rendered at the drone's own pose,
   so an off-axis view can only ever be given a bearing, never a range - and a
   bearing-only candidate is one the drone cannot arrive on. A small or distant
   target gets one ``inspect_region`` magnification - of the region the
   comparison named when it named one - and the crop is grounded.
4. **confirm** - the best candidate of the first view that produced one is
   verified with ``compare_candidates`` when a second candidate exists, so the
   answer carries a verdict rather than a box.
5. **change place, not words** - with no candidate in the round, the skill
   proposes observation points (``propose_viewpoints``), states *which
   information the new place is expected to add* and flies one leg
   (``go_to_viewpoint``), then sweeps again - until the budget, a candidate, or
   an unproductive move ends the loop.

The comparison is one step of the routine and one model call: it is logged
(``step=compare_views``), counted in ``vlm_calls`` / ``cost.model_only_steps``,
and it never *replaces* the search - a comparison that fails, or that flags no
view at all, falls back to the swept order, so the routine can only be better
or equal to grounding each view blindly. Instructions that name the target
without a relation keep the fast path (no extra model call).

What the skill returns is the whole routine, not one opaque call: ``steps`` is
``[{step, why, obs_ids, result}]`` (what was done, why, which frames were
involved), and ``vlm_calls`` / ``cost`` publish the real model calls the routine
made (each grounding call is a model call; the caller must see them). It never
reports a candidate it did not ground, and it never repeats a grounding the
anti-loop guard of ``tool_impl_perception`` refused.

Budget: the routine is *one decision* of the model, so it is charged like one.
The episode's ``tool_calls_left`` pays the call itself (``ROUTINE_OUTER_COST``,
and the registry charges the turn), while the steps *inside* the call run on the
routine's own allowance - the step cap (``max_steps``, default
:data:`DEFAULT_INNER_STEPS`) clipped by what the episode has left. Every inner
call is counted and published in ``cost`` (``outer_tool_calls`` /
``inner_tool_calls`` / ``vlm_calls``, plus ``model_only_steps`` for the steps
that are a model call without a tool call - the view comparison), so the full
price stays visible while a
single call can never empty the episode budget: at the cap, or when the episode
is nearly out of budget, the routine stops and returns what it has (steps,
candidates, coverage, ``next_action``).

The skill drives the *registered* tools (the registry it was registered on, the
one published in ``ctx.extra``, or the handlers themselves as a last resort), so
the budget, the tool log and the repeated-call cache stay in charge of what the
routine may do.
"""

from __future__ import annotations

import functools
import json
import math
import re
import time
from typing import Any, Callable, Mapping

from .tools import (
    STATUS_BLOCKED,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_OK,
    EvidenceRef,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
# the 45 deg sector binning of the coverage contract is defined once, in the
# module that produces it - the expected-information report must not re-derive it
from .tool_impl_perception import (
    _vlm_from_ctx,
    resolve_observation,
    sector_index,
    sector_range,
)

#: rounds the routine runs inside one call: a round is sweep -> ground -> move
DEFAULT_SEARCH_TURNS = 1
MAX_SEARCH_TURNS = 4
#: views grounded per round (each one is a model call)
DEFAULT_GROUNDINGS = 3
MAX_GROUNDINGS = 8
#: crops magnified per round before a view is given up on
DEFAULT_INSPECTS = 1
MAX_INSPECTS = 2
#: views one ``compare_views`` call may carry (a level sweep of the default 8
#: views fits: the comparison is one call, not one call per view)
DEFAULT_COMPARE_VIEWS = 8
MAX_COMPARE_VIEWS = 8
#: views the comparison may pick for grounding. The picked view comes first, the
#: rest of the sweep follows when it finds nothing, so a wrong pick cannot end
#: the search - it can only cost the groundings it ordered first.
DEFAULT_COMPARE_PICKS = 2
MAX_COMPARE_PICKS = 3
#: steps one call may run inside itself (each one is a tool call on the routine's
#: own allowance): one sweep, a few groundings, a magnified look, a proposed
#: viewpoint and one leg. The cap is what keeps a single model decision from
#: becoming a whole sub-episode - when it is reached the routine returns the
#: partial result (steps + candidates so far) and says how to continue.
DEFAULT_INNER_STEPS = 8
MAX_INNER_STEPS = 24
#: budget units the *episode* pays for one call: the caller's own turn (the
#: registry charges it) - the steps above run on the routine's allowance, so the
#: episode budget is not emptied by one search.
ROUTINE_OUTER_COST = 1
#: bounded step log / candidate report
MAX_STEPS = 32
MAX_REPORTED_CANDIDATES = 3
#: the region a view is magnified through when a target may be small or distant
ZOOM_REGION = [250, 250, 750, 750]
#: how many observation ids of a round travel with the result (the evidence pack)
MAX_IMAGE_REFS = 8

#: order in which a view is preferred for grounding (ahead first, behind last)
DIRECTION_RANK = {
    "ahead": 0,
    "ahead-left": 1,
    "ahead-right": 1,
    "left": 2,
    "right": 2,
    "behind-left": 3,
    "behind-right": 3,
    "behind": 4,
}
DIRECTION_UNKNOWN = "unknown"
#: movement outcomes that moved nothing: the routine stops instead of re-flying
DEAD_OUTCOMES = ("already_at_pose", "no_progress", "collision", "stalled", "planner_failed")

#: instructions that name the target by *where it is*: the relation ("the one
#: next to the lamp") is only readable in a frame that holds both objects, which
#: a one-shot grounding of a single view may well not be. Matching one of these
#: phrases is what turns the comparison on by default.
RELATION_PHRASES = (
    "next to",
    "next by",
    "nearest",
    "closest",
    "close to",
    "near to",
    "nearby",
    "near",
    "beside",
    "alongside",
    "left of",
    "right of",
    "in front of",
    "behind",
    "between",
    "under",
    "below",
    "underneath",
    "above",
    "on top of",
    "atop",
    "attached to",
    "adjacent to",
    "by the",
    "with the",
)
RELATION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(phrase) for phrase in RELATION_PHRASES) + r")\b",
    re.IGNORECASE,
)
#: ``compare_views`` argument values (``auto``: only when a relation is named)
COMPARE_MODES = ("auto", "on", "off")

#: the comparison prompt: one call, every view at once, a verdict per view. The
#: answer has to fit the *same* generation budget as every other call (a few
#: hundred tokens), so the per-view form is compact: the verdict for every frame,
#: the details only for the frames that are candidates.
COMPARE_VIEWS_SYSTEM_PROMPT = (
    "You are the view-comparison step of a drone navigation agent. You are given "
    "several frames of the same scene, each shot facing a different direction; the "
    "direction of every frame is listed in the user message in the same order as the "
    "attached images. Decide, for every frame, whether it shows the object the "
    "instruction names - including the relation the instruction states (for example "
    '"next to the lamp": the frame must show the object *and* the lamp) - and answer '
    "with ONE short JSON object and nothing else: "
    '{"views": [{"view": 1, "target": false}, {"view": 2, "target": true, '
    '"confidence": 0.8, "reason": "red object with the lamp beside it", '
    '"box": [x1, y1, x2, y2]}], "best_view": 2, "relation_visible": true}. '
    "List every frame in order, in the short form {\"view\": n, \"target\": false} - "
    "add \"confidence\", a few words of \"reason\" and "
    "\"box\": [x1, y1, x2, y2] (normalized 0-1000 image coordinates of the instance) "
    "only to the frames whose target is true. Keep the whole answer short: "
    "\"best_view\" is the single frame to trust, \"relation_visible\" is false when no "
    "frame shows the relation the instruction names. Judge only what the frames show: "
    "an object whose named relation is not visible in that frame is not resolved by "
    "that frame."
)


def relation_phrase(instruction: str) -> str:
    """The relation phrase the instruction names the target by (``""``: none).

    "The red object next to the lamp" names the target by where it is, not by
    what it is: the relation exists only in the views that hold both objects, so
    where the target is has to be read before any view is grounded. This is the
    trigger for comparing the swept views first.
    """
    match = RELATION_PATTERN.search(str(instruction or ""))
    return match.group(1).lower() if match else ""


def _compare_first(instruction: str, mode: Any) -> tuple[bool, str]:
    """``(compare the views first?, why)`` for this instruction and mode."""
    mode = str(mode or "auto").strip().lower()
    if mode not in COMPARE_MODES:
        mode = "auto"
    if mode == "off":
        return False, "compare_views=off: the swept views are grounded in the order of the sweep"
    if mode == "on":
        return True, (
            "compare_views=on: the views are compared in one call before any of them is "
            "grounded, so the instance the instruction names is picked from evidence"
        )
    phrase = relation_phrase(instruction)
    if phrase:
        return True, (
            f"'{phrase}' names the target by where it is: that relation is only readable "
            "in the views that hold both objects, so every view is compared in one call "
            "before any of them is grounded"
        )
    return False, ""


# --------------------------------------------------------------------------
# materials of the comparison
# --------------------------------------------------------------------------


def _as_ids(value: Any) -> list[str]:
    """Observation ids from an argument: a list, or one comma/space separated string."""
    if isinstance(value, str):
        value = [item for item in re.split(r"[,\s]+", value) if item]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _json_object(text: Any) -> Any:
    """First decodable JSON object/array in a model reply (prose tolerated)."""
    if not isinstance(text, str) or not text.strip():
        return None
    for opener, closer in (("{", "}"), ("[", "]")):
        tried = 0
        for index, char in enumerate(text):
            if char != opener:
                continue
            tried += 1
            if tried > 8:
                break
            depth = 0
            for cursor in range(index, len(text)):
                if text[cursor] == opener:
                    depth += 1
                elif text[cursor] == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[index : cursor + 1])
                        except json.JSONDecodeError:
                            break
    return None


def _heading_deg(record: Any) -> float | None:
    """Heading of a stored observation in degrees (world frame), or ``None``."""
    yaw = getattr(record, "yaw_deg", None)
    if yaw is None:
        yaw = getattr(record, "yaw_rad", None)
        if yaw is not None:
            yaw = math.degrees(float(yaw))
    value = _as_float(yaw, None)
    if value is None:
        return None
    return (value + 180.0) % 360.0 - 180.0


def _view_where(entry: Mapping) -> str:
    """Where one compared frame looked, in the words of the capture."""
    heading = _as_float(entry.get("yaw_deg"), None)
    direction = str(entry.get("direction") or DIRECTION_UNKNOWN)
    if direction in ("", DIRECTION_UNKNOWN):
        where = "a frame of this episode"
    else:
        where = f"looking {direction} of the drone's heading"
    if heading is None:
        return where
    return f"{where} (heading {round(float(heading), 1)} deg)"


def _comparison_frames(ctx: ToolContext, view_index: list, *, limit: int) -> list[dict]:
    """The frames to compare: ``[{obs_id, direction, yaw_deg, where, frame}]``.

    Every id of the list is resolved to its pixels (the capture's frames, or the
    frames a retrieval reported - the observation repository is the one place
    both come from) and only the ones with pixels travel; the direction comes
    from the capture's own view index when there is one, and from the frame's
    stored heading when the frame was recalled.
    """
    entries: list[dict] = []
    seen: set[str] = set()
    for item in view_index or []:
        if not isinstance(item, Mapping):
            continue
        obs_id = str(item.get("obs_id") or "")
        if not obs_id or obs_id in seen:
            continue
        seen.add(obs_id)
        record = resolve_observation(ctx, obs_id)
        frame = getattr(record, "frame", None) if record is not None else None
        if frame is None:
            continue
        heading = item.get("yaw_deg")
        if heading in (None, ""):
            heading = _heading_deg(record)
        entry = {
            "obs_id": obs_id,
            "direction": str(item.get("direction") or DIRECTION_UNKNOWN),
            "yaw_deg": _as_float(heading, None),
            "frame": frame,
        }
        entry["where"] = _view_where(entry)
        entries.append(entry)
        if len(entries) >= max(1, int(limit)):
            break
    return entries


def _compare_views_user_prompt(instruction: str, entries: list[dict]) -> str:
    """The comparison call's user message: the frames, their bearings, the task."""
    lines = [
        f"view {index}: {entry.get('where')} ({entry.get('obs_id')})"
        for index, entry in enumerate(entries, start=1)
    ]
    return (
        f"Instruction: {instruction}\n"
        "Frames, in the order they are attached:\n"
        + "\n".join(lines)
        + "\n\nWhich frames show the object the instruction names, and which is the one "
        "to trust? Add the box of the instance that makes a frame a candidate."
    )


def _view_entry(item: Mapping, position: int, count: int) -> dict | None:
    """One per-view verdict of a comparison reply (``None``: not a usable entry)."""
    raw_index = item.get("view", item.get("index", item.get("frame", position)))
    index = _as_int(raw_index, position)
    if index < 1 or index > count:
        return None
    flag = item.get("looks_like_target")
    if flag is None:
        for key in ("target", "match", "is_target", "contains_target", "holds_target"):
            if key in item:
                flag = item[key]
                break
    if isinstance(flag, str):
        flag = flag.strip().lower() in ("true", "yes", "1", "y")
    confidence = _as_float(item.get("confidence"), None)
    if confidence is None:
        confidence = _as_float(item.get("score"), None)
    if confidence is None and flag is not None:
        confidence = 0.6 if flag else 0.0
    raw_box = item.get("box") if item.get("box") is not None else item.get("bbox")
    box = (
        [_as_float(value, None) for value in list(raw_box)[:4]]
        if isinstance(raw_box, (list, tuple))
        else []
    )
    return {
        "view": int(index),
        "looks_like_target": bool(flag) if flag is not None else False,
        "confidence": None if confidence is None else round(max(0.0, min(1.0, confidence)), 3),
        "reason": str(item.get("reason", item.get("why", "")) or "")[:120],
        "box": [float(value) for value in box if value is not None],
    }


def _comparison_views(payload: Any, count: int) -> list[dict] | None:
    """Model reply -> per-view verdicts (``None``: nothing usable came back).

    Accepts ``{"views": [...]}`` (or ``comparisons``/``frames``/a bare list) and
    keeps the fields the routine acts on: which view, whether it looks like the
    target, the confidence, the reason and the box of the instance. A view index
    outside the compared list is dropped instead of trusted.
    """
    if isinstance(payload, Mapping):
        items = None
        for key in ("views", "comparisons", "frames", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                items = value
                break
            if isinstance(value, Mapping):
                items = [value]
                break
        if items is None and any(key in payload for key in ("view", "looks_like_target", "box")):
            items = [payload]
        best = payload.get("best_view", payload.get("best"))
        relation_visible = payload.get("relation_visible")
    elif isinstance(payload, list):
        items, best, relation_visible = payload, None, None
    else:
        return None
    if not items:
        return None
    views = [
        entry
        for entry in (_view_entry(item, position, count)
                      for position, item in enumerate(items, start=1))
        if entry is not None
    ]
    if not views:
        return None
    best_index = _as_int(best, 0)
    for view in views:
        view["best"] = bool(best_index) and int(view["view"]) == int(best_index)
    return views


def _salvage_views(text: Any, count: int) -> list[dict] | None:
    """Per-view verdicts salvaged from a reply that was cut off mid-JSON.

    The generation budget is small (the comparison shares the call budget of
    every other tool), so a listing of eight views can be truncated: the complete
    entries of the list are still evidence, and reading them beats treating the
    whole answer as lost. Only syntactically complete objects after the
    ``"views"`` key are taken.
    """
    if not isinstance(text, str) or '"views"' not in text:
        return None
    tail = text.split('"views"', 1)[1]
    start = tail.find("[")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    position = start + 1
    views: list[dict] = []
    while position < len(tail):
        while position < len(tail) and tail[position] in " \t\r\n,":
            position += 1
        if position >= len(tail) or tail[position] == "]":
            break
        try:
            item, end = decoder.raw_decode(tail, position)
        except json.JSONDecodeError:
            break
        position = end
        if isinstance(item, Mapping):
            entry = _view_entry(item, len(views) + 1, count)
            if entry is not None:
                views.append(entry)
    if not views:
        return None
    for view in views:
        view["best"] = False
    return views


def _comparison_picks(entries: list[dict], views: list[dict], limit: int) -> tuple[list[str], dict[str, list]]:
    """``(obs ids to ground, region per obs id)`` from the compared verdicts.

    A view the model flagged as showing the target comes first (confidence
    descending), then the frame it named as best; a view flagged false is not
    picked. Nothing flagged - or a comparison that answered nothing - picks
    nothing, and the routine grounds the swept order instead: the comparison can
    order the search, never end it.
    """
    by_index = {index: entry for index, entry in enumerate(entries, start=1)}
    ranked = [view for view in views if view.get("looks_like_target")]
    ranked.sort(
        key=lambda view: (
            0 if view.get("best") else 1,
            -(_as_float(view.get("confidence"), 0.0) or 0.0),
            int(view.get("view") or 0),
        )
    )
    picks: list[str] = []
    regions: dict[str, list] = {}
    for view in ranked:
        entry = by_index.get(int(view.get("view") or 0))
        if entry is None:
            continue
        obs_id = str(entry.get("obs_id"))
        if obs_id not in picks:
            picks.append(obs_id)
        box = list(view.get("box") or [])
        if len(box) == 4 and obs_id not in regions:
            regions[obs_id] = [float(value) for value in box]
        if len(picks) >= max(1, int(limit)):
            break
    return picks, regions


def _comparison_summary(result: Mapping, picks: list[str]) -> dict:
    """Bounded, JSON-plain summary of one comparison step (for the step log)."""
    summary: dict[str, Any] = {"status": "ok" if "error" not in result else "failed"}
    summary["model_call"] = True
    if result.get("backend"):
        summary["backend"] = result.get("backend")
    if "error" in result:
        summary["error"] = result["error"]
        return summary
    views = list(result.get("views") or [])
    summary["n_views"] = len(views)
    summary["n_compared"] = result.get("n_compared")
    if result.get("truncated"):
        # the reply was cut off by the generation budget: what came back is kept
        # (complete entries only), and the fact is visible instead of silent
        summary["truncated"] = True
    summary["picked"] = list(picks)
    summary["relation_visible"] = result.get("relation_visible")
    summary["views"] = [
        {
            "view": view.get("view"),
            "looks_like_target": bool(view.get("looks_like_target")),
            "confidence": view.get("confidence"),
            "reason": view.get("reason"),
            **({"box": view.get("box")} if view.get("box") else {}),
        }
        for view in views[:MAX_COMPARE_VIEWS]
    ]
    return summary


def _compare_views_call(
    ctx: ToolContext, backend: Any, instruction: str, entries: list[dict]
) -> dict:
    """One model call comparing every view at once; never raises.

    The images, their order and their bearings are the harness's job (the model
    is not asked to load or bind anything); the answer - a verdict per view - is
    parsed here and reported as it came back, so a malformed reply is visible in
    the step log instead of being papered over.
    """
    if backend is None:
        return {"error": "no_vlm_backend"}
    frames = [entry["frame"] for entry in entries]
    user_prompt = _compare_views_user_prompt(instruction, entries)
    try:
        reply = backend.decide(
            frames,
            COMPARE_VIEWS_SYSTEM_PROMPT,
            user_prompt,
            context={
                "step": "compare_views",
                "tool": "search_target",
                "instruction": instruction,
                "observation_ids": [str(entry.get("obs_id")) for entry in entries],
                "view_index": [
                    {
                        "obs_id": str(entry.get("obs_id")),
                        "yaw_deg": entry.get("yaw_deg"),
                        "direction": entry.get("direction"),
                    }
                    for entry in entries
                ],
            },
        )
    except Exception as exc:  # noqa: BLE001 - a backend failure never crashes the routine
        return {"error": f"{type(exc).__name__}: {exc}"}
    backend_name = getattr(backend, "name", None) or type(backend).__name__
    payload = _json_object(reply)
    views = _comparison_views(payload, len(entries))
    truncated = False
    salvaged = _salvage_views(reply, len(entries))
    if salvaged is not None and (views is None or len(salvaged) > len(views)):
        # the whole reply did not parse (the listing ran past the generation
        # budget), but its complete entries did: they are the answer so far. A
        # fragment that parsed *inside* an unterminated list is exactly what the
        # salvaged list supersedes, so the longer reading wins.
        views, truncated = salvaged, True
    if views is None:
        return {
            "error": "unparsable_reply",
            "backend": backend_name,
            "raw": str(reply)[:200],
        }
    relation_visible = None
    if isinstance(payload, Mapping) and payload.get("relation_visible") is not None:
        relation_visible = bool(payload.get("relation_visible"))
    return {
        "views": views,
        "relation_visible": relation_visible,
        "method": "multi_view_compare",
        "backend": backend_name,
        "truncated": truncated,
        "n_compared": len(entries),
    }


def _picked_first(order: list[dict], picks: list[str]) -> list[dict]:
    """The swept order with the comparison's picks first (same views, new order)."""
    if not picks:
        return order
    picked = {obs_id: index for index, obs_id in enumerate(picks)}
    return sorted(
        order,
        key=lambda entry: picked.get(str(entry.get("obs_id")), len(picked) + 1),
    )


def _needs_facing(entry: Mapping) -> bool:
    """Does the drone have to turn to this view before it can be grounded?

    The capture marks the one view its own heading already faces as its primary
    view: that frame is the depth map's frame and is grounded as it is. A view
    the sweep captured off-axis (every view a relation is readable in, by
    construction - the landmark is authored outside the forward view) is not, so
    grounding it yields a bearing and no range. A frame of unknown direction (a
    recalled one) is never turned to: that would fly the drone somewhere the
    caller did not ask for.
    """
    if str(entry.get("direction") or DIRECTION_UNKNOWN) in ("", DIRECTION_UNKNOWN):
        return False
    return "primary view" not in str(entry.get("note") or "")


def _face_view(caller: "_ToolCaller", steps: list[dict], entry: Mapping) -> str | None:
    """Turn to a picked view and return the frame to ground instead (``None``: failed).

    ``return_to_observation`` restores the pose a frame was shot from *including
    its heading*; for a view of the sweep just taken that is a turn in place, and
    the capture that follows is a frame at the drone's own pose - the frame whose
    depth map can be read. Both calls are steps of the routine and are logged;
    a turn that does not work leaves the picked frame to be grounded as it was,
    so the routine can only gain from the attempt.
    """
    obs_id = str(entry.get("obs_id") or "")
    direction = str(entry.get("direction") or DIRECTION_UNKNOWN)
    summary: dict[str, Any] = {}
    turned = caller("return_to_observation", {"observation_id": obs_id})
    summary["status"] = turned.status
    outcome = turned.data.get("outcome") or turned.data.get("task_status") or turned.data.get("reason")
    if outcome:
        summary["outcome"] = str(outcome)
    if turned.status in (STATUS_FAILED, STATUS_BLOCKED):
        summary["error"] = str(turned.data.get("error") or turned.data.get("reason") or "turn_failed")
        _record(
            steps,
            "face_view",
            f"turn to the {direction} view {obs_id} so its depth can be read",
            [obs_id],
            summary,
        )
        return None
    captured = caller("observe", {"mode": "forward"})
    fresh_id = str(captured.data.get("observation_id") or "")
    if captured.status != STATUS_OK or not fresh_id:
        summary["error"] = str(captured.data.get("error") or "capture_failed")
        _record(
            steps,
            "face_view",
            f"turn to the {direction} view {obs_id} so its depth can be read",
            [obs_id],
            summary,
        )
        return None
    summary["grounded_frame"] = fresh_id
    if captured.data.get("yaw_deg") is not None:
        summary["heading_deg"] = captured.data.get("yaw_deg")
    _record(
        steps,
        "face_view",
        (
            f"turn to the {direction} view {obs_id} and capture there: the frame {fresh_id} "
            "stands at the drone's own pose, so the depth map belongs to it (the off-axis "
            "frame could only be given a bearing)"
        ),
        [obs_id, fresh_id],
        summary,
    )
    return fresh_id


# --------------------------------------------------------------------------
# calling the tools the routine is made of
# --------------------------------------------------------------------------


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


def _clamp(value: int, low: int, high: int) -> int:
    return int(min(max(value, low), high))


def _budget_left(ctx: ToolContext) -> bool:
    budget = getattr(ctx, "budget", None)
    if not isinstance(budget, Mapping):
        return True
    return _as_int(budget.get("tool_calls_left", 1), 1) > 0


_NAVIGATION_HANDLERS: dict[str, Callable] | None = None


def _navigation_handlers() -> dict[str, Callable]:
    """Handlers of the navigation module, registered on a lazily built registry.

    The movement half of the routine (``propose_viewpoints`` /
    ``go_to_viewpoint`` / ``check_path``) lives in another module; asking *it*
    for its own specs keeps this skill from re-implementing movement, and keeps
    the import one-way (the module loads only when the skill really needs to
    move without a registry to call through).
    """
    global _NAVIGATION_HANDLERS
    if _NAVIGATION_HANDLERS is None:
        handlers: dict[str, Callable] = {}
        try:
            from .tool_impl_navigation import register_navigation_tools

            scratch = ToolRegistry()
            register_navigation_tools(scratch)
            handlers = {name: scratch.get(name).handler for name in scratch.names()}
        except Exception:  # noqa: BLE001 - a missing navigation module is reported per call
            handlers = {}
        _NAVIGATION_HANDLERS = handlers
    return _NAVIGATION_HANDLERS


class _RoutineBudget:
    """The routine's own allowance: inner calls never drain the episode budget.

    ``search_target`` is *one decision* for the model, so it must cost one
    decision: the episode pays ``ROUTINE_OUTER_COST`` units for the call (the
    registry charges the turn) and the steps *inside* the routine run on a
    separate, bounded allowance. The allowance is the routine's step cap clipped
    by what the episode has left, so a nearly-empty episode ends the routine at
    once instead of overdrawing it, and every inner call is counted and published
    in the result's ``cost``.

    On exit the episode budget is restored to the value captured on entry: the
    inner spending never reaches it, and the caller's own bookkeeping (the
    registry's unit for the tool call) stays untouched.
    """

    def __init__(self, ctx: ToolContext, *, inner_steps: int, outer_cost: int) -> None:
        self.ctx = ctx
        self.inner_steps = max(0, int(inner_steps))
        self.outer_cost = max(0, int(outer_cost))
        self.allowance = self.inner_steps
        self._budget: dict | None = None
        self._episode_left: int | None = None

    def __enter__(self) -> "_RoutineBudget":
        budget = getattr(self.ctx, "budget", None)
        if isinstance(budget, dict):
            self._budget = budget
            self._episode_left = _as_int(budget.get("tool_calls_left", 0), 0)
            self.allowance = max(
                0, min(self.inner_steps, self._episode_left - self.outer_cost)
            )
            budget["tool_calls_left"] = self.allowance
        return self

    def __exit__(self, *exc_info) -> bool:
        if self._budget is not None and self._episode_left is not None:
            self._budget["tool_calls_left"] = self._episode_left
        return False


class _ToolCaller:
    """Calls the tools of the routine, through the registry when there is one.

    Resolution order: the caller's override (tests), a registry that actually
    has the tool (``ctx.extra["tool_registry"]``/``["registry"]``, the runtime's
    registry, then the registry this skill was registered on), and finally the
    handler itself - in which case the call still costs one budget unit and is
    appended to ``ctx.log``, so a skill call can never hide what it spent.
    """

    def __init__(
        self,
        ctx: ToolContext,
        *,
        registry: Any = None,
        vlm: Any = None,
        call: Callable[[str, dict], ToolResult] | None = None,
    ) -> None:
        self.ctx = ctx
        self.registry = registry
        self.vlm = vlm
        self.call = call
        self.tool_calls = 0

    def _registries(self) -> list[Any]:
        extra = self.ctx.extra if isinstance(self.ctx.extra, dict) else {}
        found: list[Any] = []
        for key in ("tool_registry", "registry"):
            value = extra.get(key)
            if value is not None:
                found.append(value)
        runtime = extra.get("runtime")
        for attribute in ("registry", "tools", "tool_registry"):
            value = getattr(runtime, attribute, None)
            if value is not None:
                found.append(value)
        if self.registry is not None:
            found.append(self.registry)
        return found

    def _registry_for(self, name: str):
        for registry in self._registries():
            names = getattr(registry, "names", None)
            if not callable(names):
                continue
            try:
                if name in names():
                    return registry
            except Exception:  # noqa: BLE001 - a duck-typed registry may refuse
                continue
        return None

    def __call__(self, name: str, args: dict) -> ToolResult:
        if self.call is not None:
            self.tool_calls += 1
            return self.call(name, dict(args))
        registry = self._registry_for(name)
        if registry is not None:
            self.tool_calls += 1
            return registry.call(name, dict(args), self.ctx)
        return self._direct(name, dict(args))

    def _direct(self, name: str, args: dict) -> ToolResult:
        handler = _direct_handler(name, self.vlm)
        if handler is None:
            self.tool_calls += 1
            return ToolResult(
                STATUS_FAILED,
                {
                    "error": "tool_unavailable",
                    "tool": name,
                    "hint": "this runtime has no registry entry and no handler for that tool",
                },
            )
        if not _budget_left(self.ctx):
            self.tool_calls += 1
            return ToolResult(STATUS_FAILED, {"reason": "budget_exhausted", "tool": name})
        started = time.time()
        try:
            result = handler(self.ctx, args)
        except Exception as exc:  # noqa: BLE001 - a tool never crashes the routine
            result = ToolResult(STATUS_FAILED, {"error": f"{type(exc).__name__}: {exc}", "tool": name})
        self.tool_calls += 1
        budget = getattr(self.ctx, "budget", None)
        if isinstance(budget, dict):
            budget["tool_calls_left"] = budget.get("tool_calls_left", 1) - 1
        log = getattr(self.ctx, "log", None)
        if isinstance(log, list):
            log.append(
                {
                    "tool": name,
                    "args": dict(args),
                    "status": result.status,
                    "ms": round((time.time() - started) * 1000, 1),
                    "evidence": [ref.to_dict() for ref in result.evidence],
                    "inner": "search_target",
                }
            )
        return result


def _one_routine_budget(fn: Callable) -> Callable:
    """Run a routine under its own allowance (:class:`_RoutineBudget`).

    The wrapper parses the routine's step cap and opens the allowance for the
    whole call, including the exception path - a routine that raises must not
    leave the episode budget holding an allowance it never spent.
    """

    @functools.wraps(fn)
    def wrapper(ctx: ToolContext, args: dict, **kwargs: Any) -> ToolResult:
        parsed = dict(args or {})
        max_steps = _clamp(
            _as_int(parsed.get("max_steps"), DEFAULT_INNER_STEPS), 1, MAX_INNER_STEPS
        )
        with _RoutineBudget(ctx, inner_steps=max_steps, outer_cost=ROUTINE_OUTER_COST):
            return fn(ctx, parsed, **kwargs)

    return wrapper


def _direct_handler(name: str, vlm: Any) -> Callable | None:
    """Handler of a routine tool without a registry (perception, then navigation)."""
    from . import tool_impl_perception as perception

    handlers: dict[str, Callable] = {
        "observe": perception.observe,
        "inspect_region": perception.inspect_region,
        "ground_target": lambda ctx, args: perception.ground_target(ctx, args, vlm=vlm),
        "compare_candidates": lambda ctx, args: perception.compare_candidates(ctx, args, vlm=vlm),
        "query_memory": lambda ctx, args: perception.query_memory(ctx, args),
    }
    if name in handlers:
        return handlers[name]
    return _navigation_handlers().get(name)


# --------------------------------------------------------------------------
# reading what a step returned
# --------------------------------------------------------------------------


def _record(steps: list[dict], step: str, why: str, obs_ids: list, result: dict) -> None:
    """Append one step of the routine: what was done, why, which frames, what came back."""
    if len(steps) >= MAX_STEPS:
        return
    steps.append(
        {
            "step": step,
            "why": why,
            "obs_ids": [str(item) for item in obs_ids if item],
            "result": result,
        }
    )


def _kept(data: Mapping, keys: tuple[str, ...]) -> dict:
    out = {}
    for key in keys:
        value = data.get(key)
        if value in (None, "", [], {}):
            continue
        out[key] = value
    return out


def _scan_summary(result: ToolResult) -> dict:
    data = result.data or {}
    coverage = data.get("coverage") if isinstance(data.get("coverage"), Mapping) else {}
    summary = _kept(data, ("mode", "n_views", "note"))
    summary["status"] = result.status
    if coverage:
        summary.update(
            _kept(
                coverage,
                (
                    "checked_deg",
                    "unchecked_deg",
                    "worth_looking",
                    "directions_known",
                    "down_checked",
                ),
            )
        )
    return summary


def _ground_summary(result: ToolResult) -> dict:
    data = result.data or {}
    summary = _kept(
        data,
        ("reason", "n_candidates", "error", "observation_id", "note", "hint", "vlm_calls"),
    )
    summary["status"] = result.status
    if data.get("repeated_grounding"):
        summary["repeated_grounding"] = True
    if isinstance(data.get("grounding_retry"), Mapping):
        summary["grounding_retry"] = dict(data["grounding_retry"])
    if isinstance(data.get("strategy_change"), Mapping):
        summary["strategy_change"] = dict(data["strategy_change"])
    candidates = data.get("candidates") or []
    if isinstance(candidates, list) and candidates:
        summary["candidate_ids"] = [str(item.get("id")) for item in candidates if isinstance(item, Mapping)]
    return summary


def _crop_summary(result: ToolResult) -> dict:
    data = result.data or {}
    summary = _kept(data, ("crop_image_ref", "region", "upscale", "depth_status", "note", "error"))
    summary["status"] = result.status
    return summary


def _viewpoint_summary(result: ToolResult, pick: Mapping | None) -> dict:
    data = result.data or {}
    summary = _kept(data, ("count", "reachable_count", "viewpoint_id", "reason", "error"))
    summary["status"] = result.status
    viewpoints = data.get("viewpoints") or []
    if isinstance(viewpoints, list):
        summary["viewpoints"] = [
            _kept(item, ("viewpoint_id", "position", "distance_m", "bearing_deg", "reachable", "utility"))
            for item in viewpoints[:4]
            if isinstance(item, Mapping)
        ]
    if pick is not None:
        summary["picked"] = _kept(pick, ("viewpoint_id", "position", "distance_m", "bearing_deg", "reachable"))
    return summary


def _move_summary(result: ToolResult) -> dict:
    data = result.data or {}
    summary = _kept(
        data,
        ("task_id", "outcome", "progress_m", "remaining_m", "turn_deg", "blocked_reason", "error", "hint"),
    )
    summary["status"] = result.status
    return summary


def _compare_summary(result: ToolResult) -> dict:
    data = result.data or {}
    summary = _kept(data, ("verdict", "iou", "same_frame", "conflicts", "to_confirm", "reason", "error"))
    summary["status"] = result.status
    return summary


def _direction_rank(direction: Any) -> int:
    text = str(direction or "").removeprefix("down-")
    return DIRECTION_RANK.get(text, 5)


def _view_order(view_index: list, coverage: Mapping) -> list[dict]:
    """Which view to ground first: the flagged ones, then closest to ahead."""
    entries = [dict(entry) for entry in view_index if isinstance(entry, Mapping) and entry.get("obs_id")]
    worth = [str(entry.get("obs_id")) for entry in (coverage.get("worth_looking") or []) if isinstance(entry, Mapping)]
    level = [entry for entry in entries if not str(entry.get("direction") or "").startswith("down-")]
    pitched = [entry for entry in entries if str(entry.get("direction") or "").startswith("down-")]

    def key(entry: dict):
        obs_id = str(entry.get("obs_id"))
        rank = worth.index(obs_id) if obs_id in worth else len(worth)
        return (rank, _direction_rank(entry.get("direction")))

    return sorted(level, key=key) + sorted(pitched, key=lambda entry: _direction_rank(entry.get("direction")))


def _pick_viewpoint(viewpoints: list) -> dict | None:
    """Best observation point: reachable first, then utility, then the nearest."""
    entries = [dict(item) for item in viewpoints if isinstance(item, Mapping)]
    if not entries:
        return None
    reachable = [entry for entry in entries if entry.get("reachable")]
    pool = reachable or entries
    return max(pool, key=lambda entry: (_as_float(entry.get("utility"), 0.0) or 0.0,
                                        -(_as_float(entry.get("distance_m"), 0.0) or 0.0)))


def _expected_information(entry: Mapping, coverage: Mapping) -> dict:
    """What a viewpoint is expected to add - the reason to fly there at all.

    A new position is not information by itself: the report names the heading
    the drone will look back along, the 45 deg sector that covers, and whether
    that sector is one no view has faced yet.
    """
    viewpoint_id = str(entry.get("viewpoint_id") or "")
    distance = _as_float(entry.get("distance_m"), None)
    bearing = _as_float(entry.get("bearing_deg"), None)
    info: dict[str, Any] = {"viewpoint_id": viewpoint_id}
    if distance is not None:
        info["distance_m"] = round(distance, 2)
    if bearing is None:
        info["why"] = (
            f"moving to {viewpoint_id} changes the angle the target is seen from: "
            "occlusion and distance can be settled only from a second place"
        )
        return info
    heading = (bearing + 180.0 + 180.0) % 360.0 - 180.0
    sector = sector_range(sector_index(heading))
    unchecked = coverage.get("unchecked_deg") or []
    unchecked_sectors = [
        (float(item[0]), float(item[1])) for item in unchecked if isinstance(item, (list, tuple)) and len(item) == 2
    ]
    adds = any(abs(low - sector[0]) < 1e-6 and abs(high - sector[1]) < 1e-6 for low, high in unchecked_sectors)
    info["looks_back_deg"] = round(heading, 1)
    info["sector_deg"] = sector
    info["adds_unchecked_sector"] = bool(adds)
    where = f"{round(distance, 1)} m away at bearing {round(bearing, 1)} deg" if distance is not None else "elsewhere"
    if adds:
        info["why"] = (
            f"flying {where} puts the drone where the {sector[0]:g}..{sector[1]:g} deg sector "
            "- which no level view has faced yet - can be swept"
        )
    else:
        info["why"] = (
            f"flying {where} looks back along {round(heading, 1)} deg: the same direction from "
            "another distance and angle, which is what separates a small target from an "
            "occluded one"
        )
    return info


def _grounding_confirmed(candidate: Mapping, confidence: float | None,
                         min_confidence: float | None) -> bool:
    """Whether the routine may report its best candidate as confirmed.

    A grounded candidate is only confirmed when its range is also sane:
    distance-conflicted candidates must be re-grounded from another viewpoint
    before anything treats them as found.
    """
    distance_uncertain = bool(candidate.get("distance_uncertain")) or (
        str(candidate.get("distance_source") or "") == "conflict"
    )
    if distance_uncertain:
        return False
    return confidence is None or min_confidence is None or confidence >= min_confidence


def _candidate_summary(candidate: Mapping) -> dict:
    return _kept(
        candidate,
        (
            "id",
            "observation_id",
            "box",
            "attrs",
            "confidence",
            "distance_m",
            "uncertainty",
            "position_flu",
            "position_source",
            "range_status",
            "range_m",
            "ray",
            "distance_source",
            "distance_uncertain",
            "distance_conflict",
            "triangulation",
            "view_change_advice",
            "border_large_box",
            "border_second_view",
            "border_scan",
            "status",
            "confirmation_status",
            "rejection_inherited",
            "rejection_reason",
        ),
    )


def _candidate_trustworthy(candidate: Mapping) -> bool:
    """A candidate confirmation may accept: a real range, or two-view geometry.

    A single border/large or dispersed-depth candidate stays uncertain; the
    routine keeps looking instead of stopping on it. A triangulated candidate is
    the point two bearings agree on, which is exactly the second-view support
    the single frame could not give. A candidate whose instance a confirmation
    already rejected (or a re-grounding that inherited that rejection) is not a
    candidate at all.
    """
    if str(candidate.get("status") or "") == "rejected":
        return False
    if str(candidate.get("position_source") or "") == "triangulated":
        return True
    return not bool(candidate.get("distance_uncertain")) and str(
        candidate.get("distance_source") or ""
    ) != "conflict"


def _live_candidates(candidates: list) -> list:
    """Candidates not already rejected by an earlier confirmation."""
    return [
        dict(item) for item in candidates
        if isinstance(item, Mapping)
        and str(item.get("status") or "") != "rejected"
    ]


def _best_candidate(candidates: list) -> dict:
    return max(
        (dict(item) for item in candidates if isinstance(item, Mapping)),
        key=lambda item: (
            1 if str(item.get("position_source") or "") == "triangulated" else 0,
            _as_float(item.get("confidence"), -1.0) or -1.0,
        ),
    )


# --------------------------------------------------------------------------
# the routine
# --------------------------------------------------------------------------


@_one_routine_budget
def search_target(
    ctx: ToolContext,
    args: dict,
    *,
    registry: Any = None,
    vlm: Any = None,
    call: Callable[[str, dict], ToolResult] | None = None,
) -> ToolResult:
    """Run the bounded search routine for one instruction; see the module docstring.

    The call runs under its own allowance (``_RoutineBudget``): the episode pays
    ``ROUTINE_OUTER_COST`` unit for the decision, the steps inside run on a
    clipped, capped allowance, and every inner call is published in ``cost``.

    The routine sweeps, compares the swept views in one model call when the
    instruction names the target by a relation (``compare_views``), turns to the
    views that comparison picked and grounds the capture taken there (a frame at
    the drone's own pose, the one its depth map belongs to), magnifies the region
    the comparison named, confirms the best candidate and - with nothing found -
    flies one leg to a better viewpoint. Pass ``observation_ids`` to compare
    frames already on record instead of sweeping.
    """
    args = dict(args or {})
    instruction = str(args.get("instruction") or "").strip()
    if not instruction:
        return ToolResult(
            STATUS_FAILED,
            {
                "error": "missing_instruction",
                "hint": 'call search_target with {"instruction": "<what to look for>"}',
            },
        )
    budget_turns = _clamp(_as_int(args.get("budget_turns"), DEFAULT_SEARCH_TURNS), 1, MAX_SEARCH_TURNS)
    max_groundings = _clamp(_as_int(args.get("max_groundings"), DEFAULT_GROUNDINGS), 1, MAX_GROUNDINGS)
    max_inspects = _clamp(_as_int(args.get("max_inspects"), DEFAULT_INSPECTS), 0, MAX_INSPECTS)
    max_steps = _clamp(_as_int(args.get("max_steps"), DEFAULT_INNER_STEPS), 1, MAX_INNER_STEPS)
    min_confidence = _as_float(args.get("min_confidence"), None)
    allow_move = args.get("allow_move", True) is not False
    mode = str(args.get("mode") or "lookaround").strip().lower()
    n_views = _as_int(args.get("n_views"), 0)
    compare_limit = _clamp(
        _as_int(args.get("compare_views_max"), DEFAULT_COMPARE_VIEWS), 2, MAX_COMPARE_VIEWS
    )
    pick_limit = _clamp(
        _as_int(args.get("compare_picks"), DEFAULT_COMPARE_PICKS), 1, MAX_COMPARE_PICKS
    )
    compare_mode = str(args.get("compare_views") or "auto").strip().lower()
    face_picked = args.get("face_picked", True) is not False
    recall_ids = _as_ids(args.get("observation_ids"))

    caller = _ToolCaller(ctx, registry=registry, vlm=vlm, call=call)
    steps: list[dict] = []
    vlm_calls = 0
    model_steps = 0
    candidate_id: str | None = None
    found: list[dict] = []
    uncertain_found: list[dict] = []
    confirmed = False
    confirmation: dict | None = None
    last_coverage: dict = {}
    last_view_index: list = []
    last_image_refs: list[str] = []
    moved_to: list[str] = []
    next_action: dict | None = None
    comparison: dict | None = None
    compared_ids: list[str] = []
    picked_ids: list[str] = []
    reason = ""
    turns = 0

    def stop_reason() -> str:
        """Why the routine must stop now (``""`` = keep going).

        The inner-step cap is the routine's own bound; the allowance is what the
        episode could still pay for. Both end the call *with what it has*, they
        never drain the episode.
        """
        if caller.tool_calls + model_steps >= max_steps:
            return "inner_steps_exhausted"
        if not _budget_left(ctx):
            return "budget_exhausted"
        return ""

    for turn_index in range(budget_turns):
        reason = stop_reason()
        if reason:
            break
        turns += 1
        if recall_ids:
            # the caller handed the frames already on record (the ids a
            # ``query_memory`` hit reported): they are the views of this round,
            # "compare before deciding" applies to recalled evidence too - and a
            # recall needs no sweep, no leg and no new pixels
            coverage: dict = {}
            view_index = [{"obs_id": obs_id} for obs_id in recall_ids]
            last_view_index = view_index
        else:
            scan_args: dict[str, Any] = {"mode": mode}
            if n_views:
                scan_args["n_views"] = _clamp(n_views, 1, 16)
            scan = caller("observe", scan_args)
            if scan.status == STATUS_FAILED and not scan.data.get("view_index"):
                reason = str(scan.data.get("error") or "scan_failed")
                _record(steps, "observe", "sweep for the target", [], _scan_summary(scan))
                break
            coverage = scan.data.get("coverage") if isinstance(scan.data.get("coverage"), Mapping) else {}
            view_index = list(scan.data.get("view_index") or [])
            last_coverage = dict(coverage)
            last_view_index = view_index
            last_image_refs = [str(item) for item in (scan.data.get("image_refs") or [])][:MAX_IMAGE_REFS]
            _record(
                steps,
                "observe",
                (
                    "sweep 360 deg at the current pose: the target's direction is unknown"
                    if turn_index == 0
                    else "re-sweep from the new viewpoint: the previous place did not show the target"
                ),
                [str(entry.get("obs_id")) for entry in view_index],
                _scan_summary(scan),
            )

        order = _view_order(view_index, coverage)
        zoom_regions: dict[str, list] = {}
        compare_first, compare_why = _compare_first(instruction, compare_mode)
        if compare_first and len(order) >= 2 and not stop_reason():
            # one call compares *every* view at once: the relation the instruction
            # names is only readable where both objects are in frame, and which
            # view that is cannot be known before looking at them together
            entries = _comparison_frames(ctx, order, limit=compare_limit)
            if len(entries) >= 2:
                backend = vlm if callable(getattr(vlm, "decide", None)) else _vlm_from_ctx(ctx)
                step_result = _compare_views_call(ctx, backend, instruction, entries)
                model_steps += 1
                compared_ids = [str(entry.get("obs_id")) for entry in entries]
                if "error" not in step_result:
                    picked_ids, zoom_regions = _comparison_picks(
                        entries, list(step_result.get("views") or []), pick_limit
                    )
                _record(
                    steps,
                    "compare_views",
                    compare_why,
                    compared_ids,
                    _comparison_summary(step_result, picked_ids),
                )
                vlm_calls += 1 if backend is not None else 0
                order = _picked_first(order, picked_ids)
                if "error" not in step_result:
                    comparison = {
                        "views": list(step_result.get("views") or []),
                        "picked": list(picked_ids),
                        "relation_visible": step_result.get("relation_visible"),
                        "method": step_result.get("method"),
                        "observation_ids": compared_ids,
                    }

        grounded_views = 0
        inspects = 0
        for entry in order:
            if grounded_views >= max_groundings:
                break
            reason = stop_reason()
            if reason:
                break
            obs_id = str(entry.get("obs_id") or "")
            if not obs_id:
                continue
            grounded_views += 1
            direction = str(entry.get("direction") or DIRECTION_UNKNOWN)
            ground_id = obs_id
            if (
                face_picked
                and allow_move
                and obs_id in picked_ids
                and _needs_facing(entry)
                and not stop_reason()
            ):
                # the depth map belongs to the frame rendered at the drone's own
                # pose, so a view the sweep captured off-axis can only be given a
                # bearing - the candidate would be a ray the drone cannot arrive
                # on. Face the view it picked and ground the frame that capture
                # returns, which is the frame the depth belongs to.
                faced = _face_view(caller, steps, entry)
                if faced:
                    ground_id = faced
            grounded = caller("ground_target", {"instruction": instruction, "observation_id": ground_id})
            ground_calls = int(grounded.data.get("vlm_calls") or 0)
            if not ground_calls and grounded.data.get("backend") and not grounded.data.get("model_call_skipped"):
                ground_calls = 1
            vlm_calls += ground_calls
            _record(
                steps,
                "ground_target",
                f"look for '{instruction}' in the {direction} view {ground_id}",
                [ground_id],
                _ground_summary(grounded),
            )
            if grounded.status == STATUS_OK and grounded.data.get("candidates"):
                found = _live_candidates(grounded.data["candidates"])
                if not found:
                    # every box repeats a terminal rejection: do not stop on a
                    # candidate the confirmation already condemned
                    continue
                if any(_candidate_trustworthy(item) for item in found):
                    candidate_id = str(_best_candidate(found).get("id") or "")
                    break
                # every candidate from this view is distance-uncertain: another
                # heading at the same pose has no baseline to triangulate it, so
                # do not stop here - fly to an offset viewpoint and ground again
                uncertain_found = found
                found = []
                break
            if grounded.status == STATUS_EMPTY and grounded.data.get("reason") == "no_vlm_backend":
                # no grounding model: sweeping further cannot change that
                reason = "no_vlm_backend"
                break
            if grounded.status == STATUS_BLOCKED or grounded.data.get("repeated_grounding"):
                # the anti-loop guard refused this evidence: another view, not
                # another try on the same pixels
                continue
            worth_zoom = obs_id in {
                str(item.get("obs_id"))
                for item in (coverage.get("worth_looking") or [])
                if isinstance(item, Mapping)
            }
            # the first view of a round is magnified once even when the coverage
            # did not flag it: a distant target is exactly what a sweep misses.
            # When the comparison named the instance's box in this view, that
            # region is what gets magnified - not the middle of the frame.
            named_region = zoom_regions.get(obs_id)
            if inspects < max_inspects and (worth_zoom or grounded_views == 1) and not stop_reason():
                crop = caller(
                    "inspect_region",
                    {"observation_id": obs_id, "region": list(named_region or ZOOM_REGION)},
                )
                crop_id = str(crop.data.get("crop_image_ref") or "")
                _record(
                    steps,
                    "inspect_region",
                    (
                        f"magnify the region the comparison named in {obs_id} before giving up on it"
                        if named_region
                        else f"magnify the middle of {obs_id} before giving up on it "
                             "(small or distant target)"
                    ),
                    [obs_id, crop_id],
                    _crop_summary(crop),
                )
                if crop.status == STATUS_OK and crop_id:
                    inspects += 1
                    if stop_reason():
                        break
                    zoomed = caller(
                        "ground_target", {"instruction": instruction, "observation_id": crop_id}
                    )
                    zoom_calls = int(zoomed.data.get("vlm_calls") or 0)
                    if not zoom_calls and zoomed.data.get("backend") and not zoomed.data.get("model_call_skipped"):
                        zoom_calls = 1
                    vlm_calls += zoom_calls
                    _record(
                        steps,
                        "ground_target",
                        f"look for '{instruction}' in the magnified crop {crop_id}",
                        [crop_id],
                        _ground_summary(zoomed),
                    )
                    if zoomed.status == STATUS_OK and zoomed.data.get("candidates"):
                        zoomed_found = _live_candidates(zoomed.data["candidates"])
                        if not zoomed_found:
                            continue
                        if any(_candidate_trustworthy(item) for item in zoomed_found):
                            found = zoomed_found
                            candidate_id = str(_best_candidate(found).get("id") or "")
                            break
                        uncertain_found = zoomed_found
                        break
        if reason == "no_vlm_backend":
            break
        if reason in ("budget_exhausted", "inner_steps_exhausted"):
            break

        if found:
            best = _best_candidate(found)
            candidate_id = str(best.get("id") or candidate_id or "")
            if len(found) > 1 and not stop_reason():
                best_id = str(best.get("id") or "")
                others = [item for item in found if isinstance(item, Mapping)
                          and str(item.get("id") or "") != best_id]
                second = _best_candidate(others)
                compare = caller(
                    "compare_candidates",
                    {"candidates_a": [best.get("id")], "candidates_b": [second.get("id")],
                     "question": f"is this the same instance of '{instruction}'?"},
                )
                _record(
                    steps,
                    "compare_candidates",
                    "compare the two best candidates before committing to one",
                    [str(best.get("observation_id") or "")],
                    _compare_summary(compare),
                )
                confirmation = {
                    "verdict": compare.data.get("verdict"),
                    "to_confirm": compare.data.get("to_confirm") or [],
                    "method": compare.data.get("method"),
                }
            confidence = _as_float(best.get("confidence"), None)
            confirmed = _grounding_confirmed(best, confidence, min_confidence)
            break

        # nothing in this round: change place, not words
        reason = stop_reason()
        if reason:
            break
        proposals = caller("propose_viewpoints", {"count": _clamp(_as_int(args.get("viewpoints"), 3), 1, 6)})
        viewpoints = list(proposals.data.get("viewpoints") or [])
        pick = _pick_viewpoint(viewpoints)
        expected = _expected_information(pick, coverage) if pick is not None else None
        summary = _viewpoint_summary(proposals, pick)
        if expected is not None:
            summary["expected_information"] = expected
        _record(
            steps,
            "propose_viewpoints",
            "the target is not visible from this pose: propose observation points to look from",
            [],
            summary,
        )
        if pick is None:
            reason = str(proposals.data.get("reason") or "no_viewpoint")
            break
        viewpoint_id = str(pick.get("viewpoint_id") or "")
        if not allow_move:
            next_action = {
                "tool": "go_to_viewpoint",
                "args": {"viewpoint_id": viewpoint_id},
                "reason": "moving was not allowed in this call",
            }
            reason = reason or "viewpoint_proposed"
            break
        if not _budget_left(ctx) or caller.tool_calls >= max_steps:
            next_action = {"tool": "go_to_viewpoint", "args": {"viewpoint_id": viewpoint_id}}
            reason = stop_reason() or "budget_exhausted"
            break
        moved = caller("go_to_viewpoint", {"viewpoint_id": viewpoint_id})
        move_summary = _move_summary(moved)
        move_summary["viewpoint_id"] = viewpoint_id
        if expected is not None:
            move_summary["expected_information"] = expected
        _record(
            steps,
            "go_to_viewpoint",
            f"fly to {viewpoint_id} to look at a place this pose cannot see: "
            + str((expected or {}).get("why") or "a second angle on the target"),
            [],
            move_summary,
        )
        outcome = str(moved.data.get("outcome") or "")
        if moved.status in (STATUS_FAILED, STATUS_BLOCKED, "no_progress") or outcome in DEAD_OUTCOMES:
            reason = outcome or str(moved.data.get("error") or "move_failed")
            break
        moved_to.append(viewpoint_id)

    if candidate_id is None and uncertain_found:
        # the routine never got a trustworthy candidate, but it did see the
        # target: report the single-view candidate instead of pretending the
        # search found nothing. It stays unconfirmed and the next action asks
        # for the second viewpoint triangulation needs.
        found = list(uncertain_found)
        best = _best_candidate(found)
        candidate_id = str(best.get("id") or "") or None
        confirmed = False
        if not reason:
            reason = "candidate_distance_uncertain"
        next_action = {
            "tool": "search_target",
            "args": {"instruction": instruction},
            "reason": (
                "the target was seen but its range is uncertain (large/border box "
                "or dispersed depth): continue from an offset viewpoint so a second "
                "bearing triangulates it"
            ),
        }
    if candidate_id is None and not reason:
        reason = "no_candidate_in_budget"
    if candidate_id is None and next_action is None and (moved_to or reason == "inner_steps_exhausted"):
        # the routine changed the place already (or stopped at its own step cap):
        # the next call continues from where it stopped
        next_action = {"tool": "search_target", "args": {"instruction": instruction}}
        if reason == "inner_steps_exhausted":
            next_action["reason"] = (
                "the routine used its inner steps without finding the target: "
                "call it again to continue from the new viewpoint"
            )

    status = STATUS_OK if candidate_id else STATUS_EMPTY
    if candidate_id is None and reason in ("scan_failed", "no_viewpoint", "render_failed"):
        status = STATUS_BLOCKED
    data: dict[str, Any] = {
        "status": status,
        "instruction": instruction,
        "candidate_id": candidate_id,
        "candidates": [_candidate_summary(item) for item in found[:MAX_REPORTED_CANDIDATES]],
        "distance_uncertain_candidates": [
            str(item.get("id"))
            for item in found
            if isinstance(item, Mapping) and not _candidate_trustworthy(item)
        ],
        "confirmed": bool(confirmed),
        "steps": steps,
        "vlm_calls": vlm_calls,
        # the routine is one decision for the model: the call is one outer tool
        # call, the steps inside it are inner calls on the routine's allowance
        "outer_tool_calls": 1,
        "inner_tool_calls": caller.tool_calls,
        "model_only_steps": model_steps,
        "cost": {
            "outer_tool_calls": 1,
            "inner_tool_calls": caller.tool_calls,
            "tool_calls": caller.tool_calls,  # the inner calls (kept for the trace report)
            # the step that is a model call without a tool call (compare_views)
            "model_only_steps": model_steps,
            "vlm_calls": vlm_calls,
            "turns": turns,
            "budget_turns": budget_turns,
            "inner_steps": max_steps,
        },
        "coverage": last_coverage,
        "view_index": last_view_index,
        "image_refs": last_image_refs,
        "moved_to": moved_to,
        "next_action": next_action,
        "reason": reason,
        "steps_truncated": caller.tool_calls + model_steps > len(steps),
    }
    if confirmation is not None:
        data["confirmation"] = confirmation
    if compared_ids or picked_ids:
        data["views_compared"] = list(compared_ids)
        data["views_picked"] = list(picked_ids)
    if comparison is not None:
        data["comparison"] = comparison
    evidence: list[EvidenceRef] = []
    if candidate_id:
        evidence.append(EvidenceRef("memory", candidate_id, "confirmed candidate"))
        candidate_observation = next(
            (str(item.get("observation_id")) for item in found if item.get("observation_id")), ""
        )
        if candidate_observation:
            evidence.append(EvidenceRef("observation", candidate_observation, "candidate frame"))
    evidence.extend(
        EvidenceRef("observation", obs_id, "swept view") for obs_id in last_image_refs[:6]
    )
    if not evidence:
        evidence.extend(
            EvidenceRef("observation", str(entry.get("obs_id")), "swept view")
            for entry in last_view_index[:6]
            if entry.get("obs_id")
        )
    return ToolResult(status, data, evidence[:12])


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------

SEARCH_TARGET_SCHEMA = {
    "type": "object",
    "properties": {
        "instruction": {
            "type": "string",
            "description": "what to find, in the words of the task ('the red backpack')",
        },
        "budget_turns": {
            "type": "integer",
            "description": (
                f"rounds inside this call (default {DEFAULT_SEARCH_TURNS}): each round is "
                "sweep -> compare the views -> ground the picked one -> one leg to a "
                "better viewpoint"
            ),
        },
        "max_groundings": {
            "type": "integer",
            "description": f"views grounded per round (default {DEFAULT_GROUNDINGS}, each is a model call)",
        },
        "compare_views": {
            "type": "string",
            "enum": list(COMPARE_MODES),
            "description": (
                "auto (default): compare every swept view in one model call before "
                "grounding any of them whenever the instruction names the target by a "
                "relation ('the red object next to the lamp'); on: always; off: ground in "
                "the swept order"
            ),
        },
        "compare_views_max": {
            "type": "integer",
            "description": f"views one comparison call may carry (default {DEFAULT_COMPARE_VIEWS})",
        },
        "compare_picks": {
            "type": "integer",
            "description": (
                f"compared views the comparison may pick for grounding (default "
                f"{DEFAULT_COMPARE_PICKS}, max {MAX_COMPARE_PICKS}); the rest of the sweep "
                "follows when the picked views hold nothing"
            ),
        },
        "observation_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "frames already on record to search instead of sweeping (e.g. the "
                "observation ids a query_memory hit reported): they are compared in one "
                "call first - pass them when the instruction names a relation and the "
                "recalled frame may hold more than one instance"
            ),
        },
        "face_picked": {
            "type": "boolean",
            "description": (
                "true (default): turn to a picked view that the drone's own heading does "
                "not face (return_to_observation restores the heading it was shot at) and "
                "ground the capture taken there - a depth map belongs to the frame at the "
                "drone's own pose, so an off-axis view can only be given a bearing, never "
                "a range; false: ground the picked frame as it is"
            ),
        },
        "max_inspects": {
            "type": "integer",
            "description": f"crops magnified per round for small targets (default {DEFAULT_INSPECTS})",
        },
        "max_steps": {
            "type": "integer",
            "description": (
                f"steps the whole call may run inside itself (default {DEFAULT_INNER_STEPS}, "
                f"max {MAX_INNER_STEPS}): at the cap the call returns what it has - the "
                "episode keeps its own budget and the next call continues the search"
            ),
        },
        "min_confidence": {
            "type": "number",
            "description": "a candidate below this confidence is reported as not confirmed",
        },
        "allow_move": {
            "type": "boolean",
            "description": "fly to the proposed viewpoint when a round finds nothing (default true)",
        },
        "mode": {
            "type": "string",
            "enum": ["lookaround", "forward", "pitch_down"],
            "description": "how each round observes (default lookaround)",
        },
        "n_views": {"type": "integer", "description": "lookaround frames per sweep (default 8)"},
        "viewpoints": {"type": "integer", "description": "how many observation points to propose"},
    },
    "required": ["instruction"],
}

SEARCH_TARGET_DESCRIPTION = (
    "Search for the target the instruction names, end to end: sweep 360 deg "
    "(observe lookaround), **compare every swept view in one model call** before "
    "grounding any of them (the comparison answers, per view, whether it shows the "
    "target - relation included, why, and how sure; it runs by default when the "
    "instruction names the target by a relation such as 'the red object next to the "
    "lamp', whose answer is only readable in a view that holds both objects), ground "
    "the views it picked first, magnify the region it named for small or distant "
    "targets, confirm the best candidate with compare_candidates - and when a round "
    "finds nothing, propose observation points, state what the new place is expected "
    "to add and fly one leg there (propose_viewpoints + go_to_viewpoint), then sweep "
    "again. A picked view the drone does not already face is turned to before it is "
    "grounded (the depth map belongs to the frame at the drone's own pose, so an "
    "off-axis frame can be given a bearing but no range). Pass observation_ids to run "
    "the same comparison on frames already on "
    "record (e.g. the frames a query_memory hit reported) instead of sweeping. Use it "
    "instead of re-grounding views one by one: it refuses to ground the same evidence "
    "twice (the anti-loop guard), and it returns steps=[{step, why, obs_ids, result}] "
    "(the comparison is step=compare_views) plus vlm_calls/cost so every model call "
    "it made is visible. One call is one "
    f"decision for you: it costs the episode one tool call, runs at most "
    f"{DEFAULT_INNER_STEPS} steps inside itself (max_steps) and returns what it has "
    "when it reaches that cap. candidate_id is set when a candidate was found and "
    "confirmed; otherwise status is empty and next_action says how to continue."
)


def register_search_tools(registry: ToolRegistry, vlm: Any = None) -> None:
    """Register ``search_target`` on ``registry`` (the perception registration chain)."""
    registry.register(
        ToolSpec(
            name="search_target",
            description=SEARCH_TARGET_DESCRIPTION,
            input_schema=SEARCH_TARGET_SCHEMA,
            handler=lambda ctx, args: search_target(ctx, args, registry=registry, vlm=vlm),
            # the routine may fly the drone to the next viewpoint itself
            side_effects=True,
            cacheable=False,
        )
    )


# Generic-discovery alias: loaders probe a tuple of register-function names, so a
# module picked up by name still contributes its tools.
register_tools = register_search_tools


__all__ = [
    "DEFAULT_COMPARE_PICKS",
    "DEFAULT_COMPARE_VIEWS",
    "DEFAULT_GROUNDINGS",
    "DEFAULT_INNER_STEPS",
    "DEFAULT_INSPECTS",
    "DEFAULT_SEARCH_TURNS",
    "MAX_COMPARE_PICKS",
    "MAX_COMPARE_VIEWS",
    "MAX_GROUNDINGS",
    "MAX_INNER_STEPS",
    "MAX_INSPECTS",
    "MAX_SEARCH_TURNS",
    "MAX_STEPS",
    "ROUTINE_OUTER_COST",
    "SEARCH_TARGET_DESCRIPTION",
    "SEARCH_TARGET_SCHEMA",
    "ZOOM_REGION",
    "register_search_tools",
    "relation_phrase",
    "search_target",
]
