"""Agent memory for the tool-calling navigation runtime (goal.md section 4).

Memory is *task state with evidence*, in three layers:

- **Working memory** (:class:`WorkingMemory`): what the task is doing right now -
  the current subgoal, the candidate places in play, the open questions and the
  failed attempts. Small and fixed-capacity: every write replaces, merges or
  evicts, so the layer always fits in a prompt.
- **Spatial and semantic memory** (:class:`SpatialMemory`): landmarks, candidate
  positions, the connections between regions and the search coverage. Duplicate
  records merge into one place that keeps its provenance (which observation /
  candidate / call produced it) and its uncertainty (the best range estimate
  plus the spread of the samples). The sectors a capture checked
  (``checked_deg``) live here, not on a frame.
- **Episodic evidence** (:class:`ObservationLog`): the key frames - first
  sightings, counter-evidence, failure scenes, decision turning points - saved
  by event and retrieved on demand. Pixel-identical frames captured at the same
  pose and heading merge into one entry (the discovery stays canonical, the
  repeats are counted and their ids stay resolvable), and the pixels of one
  event are bounded, so a long episode keeps its evidence addressable.

``ObservationLog`` is also the archive of what was actually seen (frame, pose,
timestamp, event tags). The model cannot rewrite the entries; it queries them
and asks for the original frame. Two things are kept apart on purpose:

- ``goal_target`` - what *the task* is looking for (the instructed target name).
  A capture inherits it; it is a task statement, never evidence.
- ``sighting`` - what a frame *really showed*: an image/an observation id, the
  candidate (``cand_N``, a box or a pixel) and the basis it was accepted on
  (``ground_target`` hit, a model's pointed pixel plus attributes). Only
  :class:`Observation` entries carrying one count as sightings, and a query by
  target matches the sighting fields first (goal-target frames come back
  explicitly labelled ``goal_frame`` when no sighting matches).

A null result is bound to the view that produced it: ``add_negative`` records
"not found **under this viewpoint, heading, field of view and region**", never
"this area holds nothing" - a lesson from the loops where a failed look at one
heading was remembered as cleared ground.

Query vocabulary (``query(q)``, ``q`` is a string or a dict):

    log.query("red box")                      # free-text keyword relevance
    log.query({"target": "red box", "limit": 3})     # sightings first
    log.query({"goal_target": "red box"})     # frames captured while looking
    log.query({"sighting": True})             # only frames that saw something
    log.query({"negative": True})             # only scoped null results
    log.query({"event": "sighting"})          # substring match on event tags
    log.query({"point": [3.2, 1.0, 0.5], "fov_deg": 90, "max_distance_m": 8})
    log.query({"origin": [0, 0, 0], "bearing_deg": 40, "tolerance_deg": 45})
    log.query({"since": t0, "before": t1, "since_step": 4, "step": 7})

Bearing convention: world FLU, ``atan2(dy, dx)`` in degrees, 0 = +x, positive =
counter-clockwise (toward +y). ``point`` reports the point's bearing INSIDE the
observation's camera frame (``bearing_deg``, 0 = image center); ``origin`` +
``bearing_deg`` select observations whose own position lies in that sector as
seen from ``origin``. Points may be given as (x, y) - z defaults to 0.

Layers are switched per store (``spatial_memory`` / ``episodic_memory`` on
``ObservationLog`` / :func:`build_memory`) for the ablation arms. With
``spatial_memory=False`` nothing is merged into places, links or coverage - the
store has no second layer at all.

With ``episodic_memory=False`` the store keeps **no episodic evidence**: no
frame pixels, no sighting (the frame + candidate + basis pairing), no scoped
null result, and none of that layer's maintenance either (no duplicate merge,
no per-event pixel budget). The switch is enforced on the *write* side, so
nothing has to be hidden on the way out - a store that was built without the
layer never stores an image, a sighting or a null result, and therefore no
retrieval can return one. What the very same writes still keep is what the
other two layers own:

- the *capture record* (id, pose, heading, step, time, the caller's events, the
  goal target, labels, note, intrinsics, the coverage report) - which is what
  makes "where was I" answerable; and
- the writes fanned out from it: the spatial layer's candidate places and
  checked sectors, the working layer's candidates, failures and questions.

``switches()`` and ``layer_status()`` report what is on, and the two evidence
writes report ``disabled:episodic_memory_off`` instead of a status that would
suggest the evidence was kept.

A retrieval returns a handful of structured entries, never the raw archive: the
entries' ``"evidence"`` field holds ``vlm_nav.tools.EvidenceRef``
objects - the same citation currency the tool registry uses - so a tool handler
forwards them into ``ToolResult`` unchanged. ``jsonable()`` renders a whole
entry as plain JSON for the MCP resource/tool layer.

Id namespace
------------
``ObservationLog`` shares the ``obs_N`` namespace of
``vlm_nav.observations.ObservationStore``: a capture recorded here with an
explicit ``obs_id`` **is** that observation, and the log's own counter is
advanced past an adopted id so a later automatic id can never collide with one
the store (or the runtime) handed out. ``record(..., intrinsics=...)`` keeps the
render intrinsics of the capture with the entry, and ``entry["intrinsics"]``
hands them back - so a recalled hit resolves not only to a frame and a pose but
to the field of view its pixels are to be projected with.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import math
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from .tools import EvidenceRef

DEFAULT_QUERY_LIMIT = 5
DEFAULT_MAX_LIMIT = 20
DEFAULT_MAX_CONTENT_CHARS = 400

#: ``obs_N`` ids may be adopted from the observation store / the runtime, so an
#: explicit id has to advance this log's own counter past it.
_OBS_ID_RE = re.compile(r"^obs[_#:]?(\d+)$", re.IGNORECASE)

STATUS_ACTIVE = "active"
STATUS_CONFIRMED = "confirmed"
STATUS_REJECTED = "rejected"

#: working-memory question states: asked, then answered (or answered by a sighting)
STATUS_OPEN = "open"
STATUS_RESOLVED = "resolved"

#: what an evidence write answers when the store was built without its episodic
#: layer (``episodic_memory=False``): the claim was validated and the other two
#: layers were written, but nothing was kept as evidence
STATUS_EPISODIC_OFF = "disabled:episodic_memory_off"

# live beliefs outrank rejected ones when relevance ties
_STATUS_WEIGHT = {STATUS_CONFIRMED: 0.3, STATUS_ACTIVE: 0.2, STATUS_REJECTED: 0.0}

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# ---------------------------------------------------------------------------
# the three layers
# ---------------------------------------------------------------------------

LAYER_WORKING = "working"
LAYER_SPATIAL = "spatial"
LAYER_EPISODIC = "episodic"

#: what a record is: a plain frame, a sighting (the frame showed the thing), a
#: scoped null result, or a frame captured while the task looked for something
#: (the goal target - never evidence that the thing was seen)
EVIDENCE_KIND_OBSERVATION = "observation"
EVIDENCE_KIND_SIGHTING = "sighting"
EVIDENCE_KIND_NEGATIVE = "negative"
EVIDENCE_KIND_GOAL_FRAME = "goal_frame"

EVENT_SIGHTING = "sighting"
EVENT_NOT_FOUND = "not_found"

#: An observation carrying one of these is a decision turning point: a merge
#: never drops the event, and the pixels of these events are kept longest.
SALIENT_EVENTS = frozenset({
    EVENT_SIGHTING, "target_sighting", "first_sighting", EVENT_NOT_FOUND, "negative",
    "conflict", "decision", "turning_point", "strategy_change", "blocked",
    "failure", "task_failed", "target_verified", "success",
})

#: "the same picture": the tolerances two records must agree within to merge
MERGE_POS_TOL_M = 0.25
MERGE_YAW_TOL_DEG = 10.0
MERGE_MEAN_ABS_TOL = 1.0  # mean |difference| in 0-255 levels

#: episodic capacity is by event: at most this many frames of one event keep
#: their pixels (the rest keep their metadata and their materialized image path)
DEFAULT_MAX_FRAMES_PER_EVENT = 6

#: the sector binning of the coverage contract (45 deg, as in
#: ``tool_impl_perception.COVERAGE_BIN_DEG``): kept local so the memory layer
#: does not import the tool layer
SCOPE_BIN_DEG = 45.0


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _wrap_angle(rad: float) -> float:
    return (float(rad) + math.pi) % (2.0 * math.pi) - math.pi


def _tokenize(text) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9_]+", str(text or "").lower()) if t]


def _as_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)):
        value = [value]
    out = []
    for item in value:
        if isinstance(item, bytes):
            item = item.decode("utf-8", "replace")
        text = item if isinstance(item, str) else str(item)
        text = text.strip()
        if text:
            out.append(text)
    return out


def _as_position(pose) -> tuple[float, float, float]:
    """position_flu tuple; also accepts a PoseState-like object."""
    if pose is None:
        return (0.0, 0.0, 0.0)
    if hasattr(pose, "position_flu"):
        pose = pose.position_flu
    arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        raise ValueError(f"pose needs 3 components, got {arr.size}")
    return tuple(float(v) for v in arr[:3])


def _as_point3(value) -> np.ndarray:
    """Query point -> float64 (3,). 2-component points are padded with z=0."""
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 2:
        arr = np.concatenate([arr, np.zeros(1)])
    if arr.size < 3:
        raise ValueError(f"point needs 2 or 3 components, got {arr.size}")
    return arr[:3]


def _to_uint8_hwc(image) -> np.ndarray:
    """Any frame form -> uint8 HWC array (world.render() gives CHW float [0,1])."""
    if isinstance(image, Image.Image):
        arr = np.asarray(image)
    else:
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"image must be HxW or HxWxC / CxHxW, got shape {arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float64)
        if float(np.nanmax(arr)) <= 1.5:  # float frames are [0,1]
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0)
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def _guess_evidence_kind(text: str) -> str:
    low = str(text or "").strip().lower()
    if low.endswith(_IMAGE_SUFFIXES):
        return "image"
    if low.startswith(("obs_", "obs#")):
        return "observation"
    if low.startswith("hyp_"):
        return "memory"
    if low.startswith(("traj_", "trajectory")):
        return "trajectory"
    if low.startswith(("map", "bev", "vp_")):
        return "map"
    return "memory"


def normalize_evidence(value) -> list[EvidenceRef]:
    """Coerce model/tool supplied evidence into ``EvidenceRef`` citations.

    Accepts None, one citation (``EvidenceRef`` | str | mapping) or an iterable
    of those; duplicates (same kind + ref) collapse to the first occurrence so a
    repeated ``support`` call cannot inflate a hypothesis' evidence count.
    """
    if value is None:
        return []
    if isinstance(value, (EvidenceRef, str, bytes, Mapping)):
        value = [value]
    out: list[EvidenceRef] = []
    seen: set[tuple[str, str]] = set()
    for item in value:
        if isinstance(item, EvidenceRef):
            ref = item
        elif isinstance(item, Mapping):
            text = str(item.get("ref") or item.get("obs_id") or item.get("id") or "").strip()
            kind = str(item.get("kind") or "").strip() or _guess_evidence_kind(text)
            ref = EvidenceRef(kind=kind, ref=text, note=str(item.get("note", "") or ""))
        elif isinstance(item, bytes):
            ref = _text_evidence(item.decode("utf-8", "replace"))
        elif isinstance(item, str):
            ref = _text_evidence(item)
        else:
            continue
        if not ref.ref:
            continue
        key = (ref.kind, ref.ref)
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def _text_evidence(text: str) -> EvidenceRef:
    text = text.strip()
    return EvidenceRef(kind=_guess_evidence_kind(text), ref=text)


def jsonable(obj) -> Any:
    """Recursively convert memory entries (EvidenceRef, numpy, tensors) to JSON."""
    if isinstance(obj, EvidenceRef):
        return obj.to_dict()
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        return jsonable(asdict(obj))
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _merge_evidence(target: list[EvidenceRef], new: list[EvidenceRef]) -> list[EvidenceRef]:
    """Append only citations not already present; return the appended ones."""
    known = {(e.kind, e.ref) for e in target}
    added = []
    for ref in new:
        key = (ref.kind, ref.ref)
        if key in known:
            continue
        known.add(key)
        target.append(ref)
        added.append(ref)
    return added


def _clamp_limit(limit, max_limit: int) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = DEFAULT_QUERY_LIMIT
    return max(1, min(value, max(1, int(max_limit))))


def _resolve_limit(limit, spec_limit, max_limit: int) -> int:
    """Effective retrieval budget: the tighter of the call argument and the
    query's own ``limit``, clamped to ``max_limit``."""
    candidates = []
    for value in (limit, spec_limit):
        if value is None:
            continue
        try:
            candidates.append(int(value))
        except (TypeError, ValueError):
            continue
    if not candidates:
        candidates = [DEFAULT_QUERY_LIMIT]
    return max(1, min(min(candidates), max(1, int(max_limit))))


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _as_int(value, default=None):
    """An int, or ``default`` when the value is absent/not a whole number."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value, default=None):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _as_optional_position(value) -> list[float] | None:
    """[x, y, z] of a position-like value, or None when it is not one."""
    if value is None:
        return None
    if hasattr(value, "position_flu"):
        value = value.position_flu
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return [round(float(v), 3) for v in arr[:3]]


def _as_box(value) -> list[float] | None:
    """Four finite numbers (a box/region as a sequence or a mapping), or None."""
    if isinstance(value, Mapping):
        for keys in (("x1", "y1", "x2", "y2"), ("u", "v", "w", "h"),
                     ("x_min", "y_min", "x_max", "y_max")):
            if all(key in value for key in keys):
                value = [value[key] for key in keys]
                break
        else:
            return None
    try:
        numbers = [float(item) for item in list(value)[:4]]
    except (TypeError, ValueError):
        return None
    if len(numbers) != 4 or not all(math.isfinite(item) for item in numbers):
        return None
    return [round(item, 1) for item in numbers]


def _normalize_text(text) -> str:
    return " ".join(str(text or "").lower().split())


def _key_text(text) -> str:
    """Comparison key of a free-text item (word tokens, punctuation dropped)."""
    return " ".join(_tokenize(text))


def _sector_of(heading_deg: float) -> list[float]:
    """The 45 deg sector a heading points into (same binning as the coverage
    report of ``tool_impl_perception``)."""
    value = (float(heading_deg) + 180.0) % 360.0 - 180.0
    low = -180.0 + SCOPE_BIN_DEG * math.floor((value + 180.0) / SCOPE_BIN_DEG)
    return [round(low, 1), round(low + SCOPE_BIN_DEG, 1)]


def _as_pair(value) -> list[float] | None:
    """Two finite numbers (a pixel, a bearing pair, a sector), or None."""
    try:
        numbers = [float(item) for item in list(value)[:2]]
    except (TypeError, ValueError):
        return None
    if len(numbers) != 2 or not all(math.isfinite(item) for item in numbers):
        return None
    return [round(item, 1) for item in numbers]


def _sector_key(low: float, high: float) -> str:
    return f"{round(low, 1)},{round(high, 1)}"


# ---------------------------------------------------------------------------
# sighting / view scope normalisation
# ---------------------------------------------------------------------------


def normalize_sighting(value, *, image=None, strict: bool = False) -> tuple[dict | None, str]:
    """Coerce one sighting into the stored form; ``(sighting, problem)``.

    A sighting is the record of a frame that *really showed* something, so it
    must carry all three of: an image (an observation id or an image path - by
    default the frame the sighting is attached to), a candidate (a ``cand_N``
    id, a box or a pixel) and a basis (``ground_target`` hit, a model pointed at
    a pixel and read attributes, ...). Anything less is refused with the reason
    that names the missing field: ``missing_image`` / ``missing_candidate`` /
    ``missing_basis`` / ``missing_candidate_and_basis``. ``strict`` turns the
    refusal into a ``ValueError`` (the write is a programming mistake); the
    default reports it so a tool call can survive a model that invented a hit.
    """
    if not isinstance(value, Mapping):
        problem = "not_a_sighting"
        if strict:
            raise ValueError(problem)
        return None, problem
    data = {str(key): jsonable(item) for key, item in value.items()}
    basis = str(data.get("basis") or data.get("reason") or data.get("why") or "").strip()
    candidate = data.get("candidate", data.get("candidate_id"))
    candidate = None if candidate in (None, "") else str(candidate)
    box = _as_box(data.get("box", data.get("bbox")))
    pixel = _as_pair(data.get("pixel", data.get("point")))
    image_ref = data.get("image", data.get("image_ref", data.get("observation_id")))
    image_ref = image if image_ref in (None, "") else image_ref
    missing = []
    if image_ref in (None, "") or not str(image_ref).strip():
        missing.append("missing_image")
    if candidate is None and box is None and pixel is None:
        missing.append("missing_candidate")
    if not basis:
        missing.append("missing_basis")
    if missing:
        problem = "missing_candidate_and_basis" if missing == ["missing_candidate", "missing_basis"] else missing[0]
        if strict:
            raise ValueError(f"sighting refuses to be recorded: {problem}")
        return None, problem
    confidence = _as_float(data.get("confidence"))
    if confidence is not None:
        confidence = round(min(max(confidence, 0.0), 1.0), 2)
    attributes = {}
    raw_attributes = data.get("attributes", data.get("attrs"))
    if isinstance(raw_attributes, Mapping):
        attributes = {
            str(key)[:24]: str(item)[:60]
            for key, item in list(raw_attributes.items())[:8]
            if item not in (None, "")
        }
    sighting = {
        "target": str(data.get("target") or data.get("goal_target") or "").strip(),
        "image": str(image_ref).strip(),
        "basis": basis[:DEFAULT_MAX_CONTENT_CHARS],
        "candidate": candidate,
        "box": box,
        "pixel": pixel,
        "anchor": str(data.get("anchor") or data.get("relation") or "").strip()[:DEFAULT_MAX_CONTENT_CHARS],
        "confidence": confidence,
        "attributes": attributes,
        "position_flu": _as_optional_position(data.get("position_flu")),
        "position_source": str(data.get("position_source") or "").strip() or None,
        "range_m": _as_float(data.get("range_m")),
        "observation_id": str(data.get("observation_id") or "").strip() or None,
        "note": str(data.get("note") or "").strip()[:DEFAULT_MAX_CONTENT_CHARS],
        "t": _as_float(data.get("t")),
        "step": data.get("step"),
    }
    return sighting, ""


def sighting_text(sighting: Mapping | None) -> str:
    """Searchable text of one sighting (its target, basis, candidate, attrs)."""
    if not isinstance(sighting, Mapping):
        return ""
    parts = [
        str(sighting.get("target") or ""),
        str(sighting.get("basis") or ""),
        str(sighting.get("candidate") or ""),
        str(sighting.get("image") or ""),
        str(sighting.get("anchor") or ""),
        str(sighting.get("note") or ""),
        " ".join(str(key) + " " + str(value) for key, value in (sighting.get("attributes") or {}).items()),
    ]
    return " ".join(part for part in parts if part).lower()


def view_scope(
    position_flu=None,
    *,
    yaw_rad: float | None = None,
    yaw_deg: float | None = None,
    hfov_deg: float | None = None,
    pitch_down_deg: float = 0.0,
    max_distance_m: float | None = None,
    region=None,
    observation_id: str | None = None,
    note: str = "",
) -> dict:
    """The condition a null result was gathered under.

    Exactly what "I did not find X here" is allowed to mean: from this position,
    facing this heading (and this sector), with this field of view and this
    downward tilt, over this region of the frame (and this distance bound when
    one is known). A negative without a scope is a claim about the whole world;
    the scope is what keeps it a claim about one look.
    """
    if yaw_deg is None and yaw_rad is not None:
        yaw_deg = math.degrees(float(yaw_rad))
    resolved_yaw = _as_float(yaw_deg)
    if resolved_yaw is not None:
        resolved_yaw = round(_wrap_angle(math.radians(resolved_yaw)) * 180.0 / math.pi, 1)
    return {
        "position_flu": _as_optional_position(position_flu),
        "yaw_deg": resolved_yaw,
        "coverage_deg": None if resolved_yaw is None else _sector_of(resolved_yaw),
        "hfov_deg": _as_float(hfov_deg),
        "pitch_down_deg": round(float(_as_float(pitch_down_deg, 0.0) or 0.0), 1),
        "max_distance_m": _as_float(max_distance_m),
        "region": _as_box(region),
        "observation_id": str(observation_id).strip() if observation_id else None,
        "note": str(note or "").strip()[:DEFAULT_MAX_CONTENT_CHARS],
    }


def scope_note(scope: Mapping | None) -> str:
    """One line naming the condition boundary a null result is bound to."""
    if not isinstance(scope, Mapping):
        return "the view this check was made from was not recorded: treat it as one look, not as cleared ground"
    parts = []
    position = scope.get("position_flu")
    if position:
        parts.append(f"from position {', '.join(f'{v:.1f}' for v in position)}")
    yaw = _as_float(scope.get("yaw_deg"))
    if yaw is None:
        parts.append("facing an unknown heading")
    else:
        sector = scope.get("coverage_deg") or _sector_of(yaw)
        parts.append(f"facing yaw_deg={yaw:g} (the {sector[0]:g}..{sector[1]:g} deg sector)")
    hfov = _as_float(scope.get("hfov_deg"))
    parts.append(f"hfov={hfov:g} deg" if hfov is not None else "field of view unknown")
    pitch = _as_float(scope.get("pitch_down_deg"), 0.0) or 0.0
    parts.append(f"tilted {pitch:g} deg down" if pitch > 0 else "level view")
    region = scope.get("region")
    if region:
        parts.append(f"region {region}")
    distance = _as_float(scope.get("max_distance_m"))
    if distance is not None:
        parts.append(f"out to {distance:g} m")
    text = (
        "not seen only under this condition: " + ", ".join(parts)
        + " - a different viewpoint, heading or field of view is untested; "
        "this does not clear the area"
    )
    note = str(scope.get("note") or "").strip()
    return f"{text} ({note})" if note else text


# ---------------------------------------------------------------------------
# working memory (layer 1): what the task is doing right now
# ---------------------------------------------------------------------------


class WorkingMemory:
    """The task's current state at a fixed capacity.

    One subgoal, a bounded list of candidate places, a bounded list of open
    questions and a bounded list of failed attempts. Every write either
    replaces, merges into an existing item or evicts the least recently updated
    one, so the layer is always small enough to render into a prompt - and
    nothing in it is a fact about the world: it is what the task believes is
    worth doing next.
    """

    def __init__(
        self,
        *,
        max_questions: int = 6,
        max_failures: int = 6,
        max_candidates: int = 8,
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
        clock=time.time,
    ):
        self.max_questions = max(1, int(max_questions))
        self.max_failures = max(1, int(max_failures))
        self.max_candidates = max(1, int(max_candidates))
        self.max_content_chars = max(32, int(max_content_chars))
        self._clock = clock
        self._subgoal: dict | None = None
        self._candidates: list[dict] = []
        self._questions: list[dict] = []
        self._failures: list[dict] = []
        self._anchors: list[dict] = []
        self._seq = 0
        self._updated_t = 0.0

    # ---------------- writing ----------------

    def set_subgoal(self, text, *, step: int | None = None, evidence=None, note: str = "",
                    anchor: str = "") -> dict | None:
        """Replace the current subgoal (``None``/empty clears it).

        ``anchor`` is the sub-task's relation phrase ("next to the wooden
        door"). It is a *persistent* field, not prompt decoration: a recovery
        or resume re-grounding reads it back (see :meth:`anchors`).
        """
        resolved = str(text or "").strip()[: self.max_content_chars]
        if not resolved:
            self._subgoal = None
            return None
        self._updated_t = self._clock()
        self._subgoal = {
            "text": resolved,
            "step": step,
            "note": str(note or "")[: self.max_content_chars],
            "anchor": str(anchor or "").strip()[: self.max_content_chars],
            "updated_t": round(self._updated_t, 3),
            "evidence": normalize_evidence(evidence),
        }
        return dict(self._subgoal)

    def note_candidate(
        self,
        candidate,
        position=None,
        *,
        target: str = "",
        confidence=None,
        source: str = "",
        step: int | None = None,
        evidence=None,
        note: str = "",
        anchor: str = "",
        identity=None,
    ) -> dict:
        """Remember a candidate place; the same id merges into one entry.

        ``anchor`` and ``identity`` are the identity-layer fields: the relation
        phrase the instance was found under and the fingerprint/verdict record
        that says *which* instance it is. They persist with the candidate so a
        retrieval or a resume brief can name both.
        """
        key = str(candidate or "").strip()
        if not key:
            key = "candidate"
        confidence = _as_float(confidence)
        self._updated_t = self._clock()
        entry = next((item for item in self._candidates if item["id"] == key), None)
        if entry is None:
            entry = {
                "id": key,
                "position_flu": _as_optional_position(position),
                "target": str(target or "").strip()[: self.max_content_chars],
                "confidence": confidence,
                "sources": [],
                "updates": 0,
                "first_t": round(self._updated_t, 3),
                "evidence": [],
                "note": "",
                "anchor": "",
                "identity": None,
            }
            self._candidates.append(entry)
        if position is not None:
            position = _as_optional_position(position)
            if position is not None:
                entry["position_flu"] = position
        if target:
            entry["target"] = str(target).strip()[: self.max_content_chars]
        if confidence is not None:
            entry["confidence"] = confidence
        if source and source not in entry["sources"]:
            entry["sources"].append(source)
        if note:
            entry["note"] = str(note)[: self.max_content_chars]
        if anchor:
            entry["anchor"] = str(anchor).strip()[: self.max_content_chars]
            self._remember_anchor(anchor, target=target, candidate_id=key, step=step)
        if isinstance(identity, Mapping):
            entry["identity"] = dict(identity)
        entry["step"] = step
        entry["updates"] = int(entry.get("updates", 0)) + 1
        entry["updated_t"] = round(self._updated_t, 3)
        _merge_evidence(entry["evidence"], normalize_evidence(evidence))
        self._evict(self._candidates, self.max_candidates)
        return dict(entry)

    def remember_anchor(
        self, anchor, *, target: str = "", candidate_id: str = "", source: str = "",
        step: int | None = None, evidence=None,
    ) -> dict:
        """Persist a relation anchor independently of a candidate entry."""
        text = str(anchor or "").strip()[: self.max_content_chars]
        self._remember_anchor(
            text, target=target, candidate_id=candidate_id, source=source,
            step=step, evidence=evidence,
        )
        return {
            "anchor": text,
            "target": str(target or "")[: self.max_content_chars],
            "candidate_id": str(candidate_id or ""),
            "source": str(source or ""),
        }

    def _remember_anchor(self, anchor, *, target: str = "", candidate_id: str = "",
                         source: str = "", step: int | None = None,
                         evidence=None) -> None:
        text = str(anchor or "").strip()[: self.max_content_chars]
        if not text:
            return
        self._updated_t = self._clock()
        entry = next((item for item in self._anchors if item["anchor"] == text), None)
        if entry is None:
            entry = {
                "anchor": text,
                "target": "",
                "candidate_ids": [],
                "source": "",
                "updates": 0,
                "evidence": [],
                "updated_t": round(self._updated_t, 3),
            }
            self._anchors.append(entry)
        if target:
            entry["target"] = str(target).strip()[: self.max_content_chars]
        candidate_id = str(candidate_id or "").strip()
        if candidate_id and candidate_id not in entry["candidate_ids"]:
            entry["candidate_ids"].append(candidate_id)
        if source:
            entry["source"] = str(source)[:64]
        entry["step"] = step
        entry["updates"] = int(entry.get("updates", 0)) + 1
        entry["updated_t"] = round(self._updated_t, 3)
        _merge_evidence(entry["evidence"], normalize_evidence(evidence))
        del self._anchors[:-self.max_questions]

    def anchors(self, limit: int | None = None) -> list[dict]:
        """The relation anchors this task has recorded, newest first."""
        ordered = sorted(self._anchors, key=lambda item: item["updated_t"], reverse=True)
        if limit is not None:
            ordered = ordered[: max(0, int(limit))]
        return [dict(item) for item in ordered]

    def open_question(self, question, *, about: str = "", step: int | None = None, evidence=None) -> dict:
        """Register an open question (same text: counted, not duplicated)."""
        text = str(question or "").strip()[: self.max_content_chars]
        key = _key_text(text)
        existing = next(
            (item for item in self._questions if _key_text(item["text"]) == key and key), None
        )
        self._updated_t = self._clock()
        if existing is not None:
            existing["seen"] = int(existing.get("seen", 1)) + 1
            existing["updated_t"] = round(self._updated_t, 3)
            existing["step"] = step if step is not None else existing.get("step")
            _merge_evidence(existing["evidence"], normalize_evidence(evidence))
            return dict(existing)
        self._seq += 1
        entry = {
            "id": f"q_{self._seq}",
            "text": text,
            "about": str(about or "").strip()[: self.max_content_chars],
            "status": STATUS_OPEN,
            "seen": 1,
            "answer": "",
            "step": step,
            "created_t": round(self._updated_t, 3),
            "updated_t": round(self._updated_t, 3),
            "evidence": normalize_evidence(evidence),
        }
        self._questions.append(entry)
        self._evict_questions()
        return dict(entry)

    def resolve_question(self, question, *, answer: str = "", evidence=None) -> dict | None:
        """Resolve a question by id or by text; None when nothing matches."""
        key = str(question or "").strip()
        entry = next((item for item in self._questions if item["id"] == key), None)
        if entry is None:
            normalized = _key_text(key)
            entry = next(
                (item for item in self._questions
                 if normalized and _key_text(item["text"]) == normalized),
                None,
            )
        if entry is None:
            return None
        entry["status"] = STATUS_RESOLVED
        entry["answer"] = str(answer or "")[: self.max_content_chars]
        entry["updated_t"] = round(self._clock(), 3)
        _merge_evidence(entry["evidence"], normalize_evidence(evidence))
        return dict(entry)

    def resolve_open(self, about, *, answer: str = "", evidence=None) -> list[dict]:
        """Resolve every open question that is about ``about`` (target tokens)."""
        tokens = _tokenize(about)
        if not tokens:
            return []
        resolved = []
        for entry in self._questions:
            if entry["status"] != STATUS_OPEN:
                continue
            haystack = _normalize_text(f"{entry['text']} {entry.get('about') or ''}")
            if not all(tok in haystack for tok in tokens):
                continue
            entry["status"] = STATUS_RESOLVED
            entry["answer"] = str(answer or "")[: self.max_content_chars]
            entry["updated_t"] = round(self._clock(), 3)
            _merge_evidence(entry["evidence"], normalize_evidence(evidence))
            resolved.append(dict(entry))
        return resolved

    def note_failure(
        self,
        what,
        *,
        reason: str = "",
        strategy: str = "",
        step: int | None = None,
        evidence=None,
        note: str = "",
    ) -> dict:
        """Remember a failed attempt; the same (what, reason) counts up."""
        text = str(what or "").strip()[: self.max_content_chars]
        reason = str(reason or "").strip()[: self.max_content_chars]
        self._updated_t = self._clock()
        entry = next(
            (item for item in self._failures if item["what"] == text and item["reason"] == reason), None
        )
        if entry is None:
            entry = {
                "what": text,
                "reason": reason,
                "strategy": "",
                "count": 0,
                "first_t": round(self._updated_t, 3),
                "evidence": [],
                "note": "",
            }
            self._failures.append(entry)
        if strategy:
            entry["strategy"] = str(strategy).strip()[: self.max_content_chars]
        if note:
            entry["note"] = str(note)[: self.max_content_chars]
        entry["step"] = step
        entry["count"] = int(entry.get("count", 0)) + 1
        entry["updated_t"] = round(self._updated_t, 3)
        _merge_evidence(entry["evidence"], normalize_evidence(evidence))
        self._evict(self._failures, self.max_failures)
        return dict(entry)

    def clear(self) -> None:
        self._subgoal = None
        self._candidates.clear()
        self._questions.clear()
        self._failures.clear()
        self._anchors.clear()

    # ---------------- reading ----------------

    def __len__(self) -> int:
        return len(self._candidates) + len(self._questions) + len(self._failures) + (1 if self._subgoal else 0)

    def snapshot(self) -> dict:
        """The whole layer as JSON-plain data (newest first inside each list)."""
        open_questions = [item for item in self._questions if item["status"] == STATUS_OPEN]
        return {
            "subgoal": None if self._subgoal is None else dict(self._subgoal),
            "candidates": [dict(item) for item in reversed(self._candidates)],
            "open_questions": [dict(item) for item in reversed(open_questions)],
            "questions": [dict(item) for item in reversed(self._questions)],
            "failures": [dict(item) for item in reversed(self._failures)],
            "capacity": {
                "subgoal": 1,
                "candidates": self.max_candidates,
                "open_questions": self.max_questions,
                "failures": self.max_failures,
            },
            "counts": {
                "candidates": len(self._candidates),
                "open_questions": len(open_questions),
                "questions": len(self._questions),
                "failures": len(self._failures),
            },
            "updated_t": round(self._updated_t, 3),
        }

    def describe(self, limit: int = 5) -> str:
        """Compact text for a prompt: subgoal, questions, candidates, failures."""
        snapshot = self.snapshot()
        lines = []
        if snapshot["subgoal"]:
            subgoal = snapshot["subgoal"]
            step = "" if subgoal.get("step") is None else f" step={subgoal['step']}"
            lines.append(f"subgoal{step}: {subgoal['text']}")
        for question in snapshot["open_questions"][: max(1, int(limit))]:
            lines.append(f"open {question['id']}: {question['text']}")
        for candidate in snapshot["candidates"][: max(1, int(limit))]:
            line = (
                f"candidate {candidate['id']} at {candidate.get('position_flu')}"
                f"{'' if not candidate.get('target') else ' for ' + candidate['target']}"
            )
            if candidate.get("anchor"):
                line += f" anchor={candidate['anchor']}"
            lines.append(line)
        for anchor in self.anchors(limit=max(1, int(limit))):
            if anchor["anchor"]:
                lines.append(
                    f"anchor {anchor['anchor']}"
                    + (f" for {anchor['target']}" if anchor.get("target") else "")
                )
        for failure in snapshot["failures"][: max(1, int(limit))]:
            lines.append(
                f"failed {failure['count']}x: {failure['what']} ({failure['reason']})"
            )
        return "\n".join(lines) if lines else "no working memory"

    def stats(self) -> dict:
        snapshot = self.snapshot()
        return {
            "layer": LAYER_WORKING,
            "subgoal": snapshot["subgoal"] is not None,
            "candidates": snapshot["counts"]["candidates"],
            "open_questions": snapshot["counts"]["open_questions"],
            "resolved_questions": sum(
                1 for item in self._questions if item["status"] != STATUS_OPEN
            ),
            "failures": snapshot["counts"]["failures"],
            "capacity": snapshot["capacity"],
        }

    # ---------------- maintenance ----------------

    def _evict(self, items: list[dict], capacity: int) -> None:
        while len(items) > max(1, int(capacity)):
            items.pop(
                min(range(len(items)), key=lambda index: items[index].get("updated_t", 0.0))
            )

    def _evict_questions(self) -> None:
        open_items = [item for item in self._questions if item["status"] == STATUS_OPEN]
        while len(open_items) > self.max_questions:
            oldest = min(open_items, key=lambda item: item.get("updated_t", 0.0))
            self._questions.remove(oldest)
            open_items.remove(oldest)


# ---------------------------------------------------------------------------
# spatial and semantic memory (layer 2): places, links, coverage
# ---------------------------------------------------------------------------


class SpatialMemory:
    """Where things are, how the region hangs together, and what was searched.

    A place is one record however often it is seen: a name, a position, the
    provenance that produced it (``observe:obs_3``, ``ground_target:cand_2``,
    ``sighting:obs_7``), the observations and candidate ids behind it, the best
    uncertainty estimate anyone reported and the spread of the samples. Two
    records merge when they name the same candidate id, or when they sit within
    ``merge_radius_m`` of each other - a candidate that jumped 30 m is a
    different place, not a refinement.

    Search coverage (``checked_deg``) lives here too: the 45 deg sectors a level
    view has faced, which view faced each one, from where, and which of them were
    searched for a target without a result.
    """

    def __init__(
        self,
        *,
        merge_radius_m: float = 1.0,
        max_places: int = 64,
        max_links: int = 32,
        max_samples: int = 4,
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
        clock=time.time,
    ):
        self.merge_radius_m = max(0.01, float(merge_radius_m))
        self.max_places = max(1, int(max_places))
        self.max_links = max(1, int(max_links))
        self.max_samples = max(1, int(max_samples))
        self.max_content_chars = max(32, int(max_content_chars))
        self._clock = clock
        self._places: list[dict] = []
        self._by_id: dict[str, dict] = {}
        self._links: list[dict] = []
        self._sectors: dict[str, dict] = {}
        self._seq = 0
        self._evicted_places = 0

    # ---------------- writing ----------------

    def remember_place(
        self,
        position=None,
        *,
        name: str = "",
        kind: str = "landmark",
        uncertainty_m=None,
        source: str = "",
        observation_id: str = "",
        candidate_id: str = "",
        target: str = "",
        confidence=None,
        step: int | None = None,
        note: str = "",
    ) -> dict:
        """Record (or merge into) the place at ``position``; returns its entry."""
        position = _as_optional_position(position)
        if position is None:
            raise ValueError("a place needs a position")
        place = self._find_place(position, candidate_id=candidate_id)
        self._updated_t = self._clock()
        if place is None:
            self._seq += 1
            place = {
                "place_id": f"place_{self._seq}",
                "seq": self._seq,
                "kind": str(kind or "landmark"),
                "name": str(name or "").strip()[: self.max_content_chars],
                "names": [],
                "target": str(target or "").strip()[: self.max_content_chars],
                "position_flu": position,
                "position_samples": [],
                "spread_m": 0.0,
                "uncertainty_m": None,
                "confidence": None,
                "candidates": [],
                "observations": [],
                "sources": [],
                "updates": 0,
                "conflicts": [],
                "note": "",
                "first_t": round(self._updated_t, 3),
                "updated_t": round(self._updated_t, 3),
                "step": step,
                "evidence": [],
            }
            self._places.append(place)
            self._by_id[place["place_id"]] = place
        name_text = str(name or "").strip()[: self.max_content_chars]
        if name_text and not place["name"]:
            place["name"] = name_text
        elif name_text and _normalize_text(name_text) not in {
            _normalize_text(item) for item in [place["name"], *place["names"]]
        }:
            # two names on one place are a conflict to keep, not a silent rename
            place["names"].append(name_text)
            place["conflicts"].append(
                f"{observation_id or source or 'a later observation'} calls this place "
                f"'{name_text}' (first named '{place['name']}')"
            )
        place["names"] = [place["name"], *[item for item in place["names"] if item != place["name"]]] \
            if place["name"] else list(place["names"])
        uncertainty_m = _as_float(uncertainty_m)
        if uncertainty_m is not None:
            known = place["uncertainty_m"]
            if known is None or uncertainty_m <= known:
                # the most certain estimate describes the place; the old samples
                # stay, so the disagreement is visible as the spread
                place["position_flu"] = position
                place["uncertainty_m"] = round(uncertainty_m, 3)
        confidence = _as_float(confidence)
        if confidence is not None:
            place["confidence"] = round(
                max(confidence, place["confidence"] if place["confidence"] is not None else confidence), 2
            )
        if target:
            place["target"] = str(target).strip()[: self.max_content_chars]
        if source and source not in place["sources"]:
            place["sources"].append(source)
        if observation_id and observation_id not in place["observations"]:
            place["observations"].append(observation_id)
        if candidate_id and candidate_id not in place["candidates"]:
            place["candidates"].append(candidate_id)
        if note:
            place["note"] = str(note)[: self.max_content_chars]
        place["step"] = step if step is not None else place.get("step")
        place["updates"] = int(place.get("updates", 0)) + 1
        place["updated_t"] = round(self._updated_t, 3)
        samples = place["position_samples"]
        if position not in samples:
            samples.append(position)
            del samples[: -self.max_samples]
        place["spread_m"] = self._spread(samples)
        place["evidence"] = self._place_evidence(place)
        self._evict_places()
        return self.get(place["place_id"])

    def remember_candidate(
        self, candidate_id: str, position=None, *, target: str = "", confidence=None,
        source: str = "", observation_id: str = "", step: int | None = None, note: str = "",
    ) -> dict:
        """A candidate's world position: the spatial half of a sighting."""
        origin = f"ground_target:{candidate_id}" if source else ""
        return self.remember_place(
            position, name="", kind="candidate", source=source or origin,
            observation_id=observation_id, candidate_id=str(candidate_id), target=target,
            confidence=confidence, step=step, note=note,
        )

    def link(
        self, place_a: str, place_b: str, *, note: str = "", traversable: bool = True,
        blocked: bool = False, length_m=None, step: int | None = None,
    ) -> dict:
        """Record that two places are connected (one region to the next)."""
        first, second = sorted((str(place_a), str(place_b)))
        if first not in self._by_id or second not in self._by_id:
            raise KeyError(f"unknown place id: {first if first not in self._by_id else second}")
        entry = next(
            (item for item in self._links if item["a"] == first and item["b"] == second), None
        )
        if entry is None:
            entry = {
                "a": first,
                "b": second,
                "traversable": bool(traversable) and not blocked,
                "length_m": _as_float(length_m),
                "note": "",
                "updates": 0,
            }
            self._links.append(entry)
        if note:
            entry["note"] = str(note)[: self.max_content_chars]
        if length_m is not None:
            entry["length_m"] = _as_float(length_m)
        entry["traversable"] = bool(traversable) and not blocked
        entry["step"] = step
        entry["updates"] = int(entry.get("updates", 0)) + 1
        del self._links[: -self.max_links]
        return dict(entry)

    def note_coverage(
        self,
        checked_deg=None,
        *,
        position=None,
        observation_id: str = "",
        step: int | None = None,
        no_result_for=None,
        note: str = "",
    ) -> list[dict]:
        """Accumulate the sectors a capture checked (with its viewpoint).

        ``checked_deg`` is the coverage contract's list of ``[lo, hi]`` sectors -
        or, with ``position`` and no sectors, the sector the position/heading
        points into is derived by the caller. Each sector keeps which views
        faced it, from where, and which targets were searched for without a
        result, so "already checked" is answerable without re-reading a frame.
        """
        sectors = []
        for item in checked_deg or []:
            numbers = _as_pair(item)
            if numbers is None or numbers[1] <= numbers[0]:
                continue
            sectors.append(numbers)
        self._updated_t = self._clock()
        entries = []
        for sector in sectors:
            key = _sector_key(sector[0], sector[1])
            entry = self._sectors.get(key)
            if entry is None:
                entry = {
                    "sector": list(sector),
                    "views": 0,
                    "observations": [],
                    "positions": [],
                    "no_result_for": [],
                    "note": "",
                    "first_t": round(self._updated_t, 3),
                    "updated_t": round(self._updated_t, 3),
                }
                self._sectors[key] = entry
            position_list = _as_optional_position(position)
            entry["views"] = int(entry["views"]) + 1
            if observation_id and observation_id not in entry["observations"]:
                entry["observations"].append(observation_id)
            if position_list is not None and position_list not in entry["positions"]:
                entry["positions"].append(position_list)
            for target in _as_str_list(no_result_for):
                if target not in entry["no_result_for"]:
                    entry["no_result_for"].append(target)
            if note:
                entry["note"] = str(note)[: self.max_content_chars]
            entry["step"] = step
            entry["updated_t"] = round(self._updated_t, 3)
            entries.append(dict(entry))
        return entries

    # ---------------- reading ----------------

    def __len__(self) -> int:
        return len(self._places)

    def ids(self) -> list[str]:
        return [place["place_id"] for place in self._places]

    def get(self, place_id: str) -> dict | None:
        place = self._by_id.get(str(place_id))
        return self._entry(place) if place is not None else None

    def places(self, limit: int | None = None) -> list[dict]:
        """Places newest-first (bounded by ``limit`` when given)."""
        # (updated_t, seq): insertion order breaks a same-tick tie, newest first
        ordered = sorted(self._places, key=lambda place: (place["updated_t"], place["seq"]), reverse=True)
        if limit is not None:
            ordered = ordered[: _clamp_limit(limit, DEFAULT_MAX_LIMIT * 4)]
        return [self._entry(place) for place in ordered]

    def links(self, limit: int | None = None) -> list[dict]:
        ordered = list(reversed(self._links))
        if limit is not None:
            ordered = ordered[: _clamp_limit(limit, DEFAULT_MAX_LIMIT * 4)]
        return [dict(entry) for entry in ordered]

    def coverage(self, *, near=None, radius_m=None) -> dict:
        """Which sectors are checked (optionally only near a position)."""
        origin = _as_optional_position(near)
        radius = _as_float(radius_m)
        sectors = []
        for entry in self._sectors.values():
            item = dict(entry)
            if origin is not None and radius is not None:
                positions = [p for p in item["positions"] if _distance(p, origin) <= radius]
                item["positions"] = positions
                item["scoped_to_position"] = True
            sectors.append(item)
        if origin is not None and radius is not None:
            sectors = [item for item in sectors if item["positions"]]
        checked = sorted([tuple(item["sector"]) for item in sectors])
        checked_deg = [[low, high] for low, high in checked]
        all_sectors = [
            [round(-180.0 + SCOPE_BIN_DEG * index, 1), round(-180.0 + SCOPE_BIN_DEG * (index + 1), 1)]
            for index in range(int(round(360.0 / SCOPE_BIN_DEG)))
        ]
        unchecked = [sector for sector in all_sectors if tuple(sector) not in checked]
        return {
            "bin_deg": SCOPE_BIN_DEG,
            "checked_deg": checked_deg,
            "unchecked_deg": unchecked,
            "sectors": sectors,
            "scoped": origin is not None and radius is not None,
            "note": (
                "checked sectors are what a level view has faced; a sector is not "
                "checked for a target unless `no_result_for` names it (and even then "
                "only from the recorded viewpoint, heading and field of view)"
            ),
        }

    def query(self, q: str | dict, limit: int | None = DEFAULT_QUERY_LIMIT) -> list[dict]:
        """Retrieve places by text, candidate id, or a point/radius filter."""
        spec = {"text": q} if isinstance(q, str) else dict(q or {})
        limit = _resolve_limit(limit, spec.get("limit"), DEFAULT_MAX_LIMIT * 4)
        tokens = _tokenize(spec.get("text") or spec.get("keywords") or "")
        candidate = str(spec.get("candidate", spec.get("candidate_id") or "")).strip()
        point = spec.get("point", spec.get("near"))
        max_distance_m = _as_float(spec.get("max_distance_m"), 12.0)
        origin = _as_optional_position(point)
        rows = []
        for place in self._places:
            if candidate and candidate not in place["candidates"]:
                continue
            haystack = " ".join(
                [place["place_id"], place["name"], place["kind"], place["target"], place["note"],
                 *place["names"], *place["candidates"], *place["observations"], *place["sources"]]
            ).lower()
            score = 0.0
            if tokens:
                hits = sum(1 for tok in tokens if tok in haystack)
                if hits == 0:
                    continue
                score = hits / len(tokens)
            computed = {}
            if origin is not None:
                distance = _distance(place["position_flu"], origin)
                if max_distance_m is not None and distance > max_distance_m:
                    continue
                computed["distance_m"] = round(distance, 3)
                score += 1.0 / (1.0 + distance)
            rows.append((score, place, computed))
        rows.sort(key=lambda row: (row[0], row[1]["updated_t"], row[1]["seq"]), reverse=True)
        return [
            self._entry(place, score=score, extra=extra)
            for score, place, extra in rows[:limit]
        ]

    def describe(self, limit: int = 5) -> str:
        """Compact text for a prompt: places and the coverage in one block."""
        limit = _clamp_limit(limit, DEFAULT_MAX_LIMIT)
        lines = []
        for place in self.places(limit=limit):
            lines.append(
                f"{place['place_id']} [{place['kind']}] {place['name'] or 'unnamed'} "
                f"at {place['position_flu']} updates={place['updates']}"
            )
        coverage = self.coverage()
        lines.append(
            f"coverage: {len(coverage['checked_deg'])}/{len(coverage['checked_deg']) + len(coverage['unchecked_deg'])} "
            "sectors checked"
        )
        return "\n".join(lines)

    def stats(self) -> dict:
        coverage = self.coverage()
        return {
            "layer": LAYER_SPATIAL,
            "places": len(self._places),
            "candidate_places": sum(1 for place in self._places if place["kind"] == "candidate"),
            "landmarks": sum(1 for place in self._places if place["kind"] != "candidate"),
            "links": len(self._links),
            "covered_sectors": len(coverage["checked_deg"]),
            "evicted_places": self._evicted_places,
        }

    # ---------------- internals ----------------

    def _find_place(self, position: list[float], *, candidate_id: str = "") -> dict | None:
        if candidate_id:
            for place in self._places:
                if candidate_id in place["candidates"]:
                    return place
        for place in self._places:
            if _distance(place["position_flu"], position) <= self.merge_radius_m:
                return place
        return None

    @staticmethod
    def _spread(samples: list[list[float]]) -> float:
        if len(samples) < 2:
            return 0.0
        return round(
            max(_distance(a, b) for a in samples for b in samples), 3
        )

    @staticmethod
    def _place_evidence(place: dict) -> list[EvidenceRef]:
        refs = [EvidenceRef("memory", place["place_id"], note=f"{place['kind']} {place['name'] or ''}".strip())]
        for observation_id in place["observations"][:2]:
            refs.append(EvidenceRef("observation", observation_id, "place sighting"))
        for candidate_id in place["candidates"][:2]:
            refs.append(EvidenceRef("memory", candidate_id, "candidate position"))
        return refs

    def _entry(self, place: dict, score: float | None = None, extra: dict | None = None) -> dict:
        entry = {
            "place_id": place["place_id"],
            "kind": place["kind"],
            "name": place["name"],
            "names": list(place["names"]),
            "target": place["target"],
            "position_flu": list(place["position_flu"]),
            "uncertainty_m": place["uncertainty_m"],
            "spread_m": place["spread_m"],
            "confidence": place["confidence"],
            "candidates": list(place["candidates"]),
            "observations": list(place["observations"]),
            "sources": list(place["sources"]),
            "updates": place["updates"],
            "conflicts": list(place["conflicts"]),
            "step": place.get("step"),
            "updated_t": round(place["updated_t"], 3),
            "evidence": list(place["evidence"]),
        }
        if place["note"]:
            entry["note"] = place["note"]
        if score is not None:
            entry["score"] = round(float(score), 3)
        if extra:
            entry.update(extra)
        return entry

    def _evict_places(self) -> None:
        while len(self._places) > self.max_places:
            oldest = min(self._places, key=lambda place: (place["updated_t"], place["seq"]))
            self._places.remove(oldest)
            self._by_id.pop(oldest["place_id"], None)
            self._evicted_places += 1


# ---------------------------------------------------------------------------
# identity and anchor memory (layer 2b): the object, not just its place
# ---------------------------------------------------------------------------

#: The four values a confirmation can record. ``unconfirmed`` is the state a
#: sighting leaves until ``confirm_candidate``/``verify_goal`` judges it.
IDENTITY_ACCEPTED = "accepted"
IDENTITY_REJECTED = "rejected"
IDENTITY_UNCERTAIN = "uncertain"
IDENTITY_UNCONFIRMED = "unconfirmed"
IDENTITY_VERDICTS = (
    IDENTITY_ACCEPTED,
    IDENTITY_REJECTED,
    IDENTITY_UNCERTAIN,
    IDENTITY_UNCONFIRMED,
)


class IdentityMemory:
    """Object identities and relation anchors - the memory a revisit needs.

    A place record answers "where was it"; this layer answers "which one was
    it". One entry per object instance merges a sighting fingerprint (target
    description, box, position, relation phrase) with the confirmation verdict
    and evidence revision taken under it, so a delayed revisit can re-identify
    the *same instance* among decoys instead of trusting a bare position.

    The fingerprint merges by candidate id. Without one, two records merge only
    when the target and box agree and the positions are within
    ``merge_radius_m``; a same-looking object elsewhere is a second instance,
    not an update. Verdicts are the latest by ``revision`` (an earlier
    acceptance never overwrites a newer rejection), and an ``anchor`` is a
    persistent field: "the cabinet next to the wooden door" is what makes the
    instance findable when its own appearance is ambiguous.
    """

    def __init__(
        self,
        *,
        merge_radius_m: float = 0.75,
        max_identities: int = 24,
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
        clock=time.time,
    ):
        self.merge_radius_m = max(0.01, float(merge_radius_m))
        self.max_identities = max(1, int(max_identities))
        self.max_content_chars = max(32, int(max_content_chars))
        self._clock = clock
        self._items: list[dict] = []
        self._seq = 0

    # ---------------- writing ----------------

    def remember(
        self,
        candidate_id: str = "",
        *,
        target: str = "",
        box=None,
        position=None,
        anchor: str = "",
        anchor_source: str = "",
        verdict: str = "",
        revision=None,
        observation_ids=(),
        subtask_id: str = "",
        task_id: str = "",
        confidence=None,
        note: str = "",
        step=None,
        evidence=None,
    ) -> dict:
        """Record (or merge into) one object identity; returns its entry."""
        key = str(candidate_id or "").strip()
        box_value = _as_box(box)
        position_value = _as_optional_position(position)
        entry = self._find(key, target=target, box=box_value, position=position_value)
        now = float(self._clock())
        if entry is None:
            self._seq += 1
            entry = {
                "identity_id": f"id_{self._seq}",
                "seq": self._seq,
                "candidate_id": key,
                "target": "",
                "box": None,
                "position_flu": None,
                "anchor": "",
                "anchor_source": "",
                "verdict": IDENTITY_UNCONFIRMED,
                "revision": None,
                "observation_ids": [],
                "subtask_id": "",
                "task_id": "",
                "confidence": None,
                "note": "",
                "updates": 0,
                "first_t": round(now, 3),
                "updated_t": round(now, 3),
                "step": None,
                "evidence": [],
            }
            self._items.append(entry)
        if key:
            entry["candidate_id"] = key
        if target:
            entry["target"] = str(target).strip()[: self.max_content_chars]
        if box_value is not None:
            entry["box"] = box_value
        if position_value is not None:
            entry["position_flu"] = position_value
        anchor_text = str(anchor or "").strip()
        if anchor_text:
            entry["anchor"] = anchor_text[: self.max_content_chars]
            entry["anchor_source"] = str(anchor_source or "").strip()[:64]
        if subtask_id:
            entry["subtask_id"] = str(subtask_id)[:64]
        if task_id:
            entry["task_id"] = str(task_id)[:64]
        confidence = _as_float(confidence)
        if confidence is not None:
            entry["confidence"] = round(min(max(confidence, 0.0), 1.0), 2)
        if note:
            entry["note"] = str(note)[: self.max_content_chars]
        solved = self._verdict(verdict)
        incoming_revision = _as_int(revision, None)
        if solved is not None:
            known_revision = _as_int(entry.get("revision"), None)
            if (
                known_revision is None
                or incoming_revision is None
                or incoming_revision >= known_revision
            ):
                entry["verdict"] = solved
                if incoming_revision is not None:
                    entry["revision"] = incoming_revision
        for observation_id in _as_str_list(observation_ids):
            if observation_id not in entry["observation_ids"]:
                entry["observation_ids"].append(observation_id)
        del entry["observation_ids"][:-8]
        if step is not None:
            entry["step"] = step
        entry["updates"] = int(entry.get("updates", 0)) + 1
        entry["updated_t"] = round(now, 3)
        _merge_evidence(entry["evidence"], normalize_evidence(evidence))
        self._evict()
        return self.get(entry["identity_id"]) or dict(entry)

    def confirm(
        self,
        candidate_id: str,
        verdict: str,
        *,
        revision=None,
        observation_ids=(),
        anchor: str = "",
        note: str = "",
        evidence=None,
    ) -> dict | None:
        """Merge one confirm verdict/revision into the candidate's identity."""
        entry = self._find(str(candidate_id or "").strip())
        if entry is None:
            return None
        return self.remember(
            candidate_id, verdict=verdict, revision=revision,
            observation_ids=observation_ids, anchor=anchor, note=note,
            evidence=evidence,
        )

    def remember_anchor(
        self, anchor: str, *, candidate_id: str = "", target: str = "",
        source: str = "", subtask_id: str = "", step=None,
    ) -> dict | None:
        """Persist a relation anchor even when the candidate id is not known yet."""
        if not str(anchor or "").strip():
            return None
        return self.remember(
            candidate_id, target=target, anchor=anchor, anchor_source=source,
            subtask_id=subtask_id, step=step,
        )

    # ---------------- reading ----------------

    def __len__(self) -> int:
        return len(self._items)

    def get(self, identity_id: str) -> dict | None:
        key = str(identity_id or "")
        for entry in self._items:
            if entry["identity_id"] == key or (entry["candidate_id"] and entry["candidate_id"] == key):
                return self._entry(entry)
        return None

    def all(self, limit: int | None = None) -> list[dict]:
        ordered = sorted(self._items, key=lambda item: (item["updated_t"], item["seq"]), reverse=True)
        if limit is not None:
            ordered = ordered[: _clamp_limit(limit, DEFAULT_MAX_LIMIT * 4)]
        return [self._entry(entry) for entry in ordered]

    def query(self, q: str | dict, limit: int | None = DEFAULT_QUERY_LIMIT) -> list[dict]:
        """Identities matching a question, ranked identity-and-anchor first.

        A query that names a target matches its description and its relation
        anchor. An accepted confirmation outranks a rejected/uncertain one, and
        an entry with an anchor outranks one without: the four facts the
        long-horizon memory is supposed to keep are what the order is made of.
        """
        spec = {"text": q} if isinstance(q, str) else dict(q or {})
        limit = _resolve_limit(limit, spec.get("limit"), DEFAULT_MAX_LIMIT * 4)
        tokens = _tokenize(spec.get("text") or spec.get("keywords") or "")
        candidate = str(spec.get("candidate", spec.get("candidate_id") or "")).strip()
        rows = []
        for entry in self._items:
            if candidate and candidate not in (entry["candidate_id"], entry["identity_id"]):
                continue
            haystack = " ".join(
                [entry["identity_id"], entry["candidate_id"], entry["target"],
                 entry["anchor"], entry["note"], *entry["observation_ids"]]
            ).lower()
            score = 0.0
            if tokens:
                hits = sum(1 for token in tokens if token in haystack)
                if hits == 0:
                    continue
                score += hits / len(tokens)
            if entry["verdict"] == IDENTITY_ACCEPTED:
                score += 1.0
            elif entry["verdict"] == IDENTITY_UNCERTAIN:
                score += 0.25
            if entry["anchor"]:
                score += 0.5
            rows.append((score, entry))
        rows.sort(key=lambda row: (row[0], row[1]["updated_t"], row[1]["seq"]), reverse=True)
        return [self._entry(entry, score=score) for score, entry in rows[:limit]]

    def anchors(self, limit: int | None = None) -> list[str]:
        """The distinct persistent relation anchors (newest first)."""
        seen: list[str] = []
        for entry in self.all():
            anchor = str(entry.get("anchor") or "")
            if anchor and anchor not in seen:
                seen.append(anchor)
        return seen if limit is None else seen[: max(0, int(limit))]

    def describe(self, limit: int = 5) -> str:
        lines = []
        for entry in self.all(limit=limit):
            line = f"{entry['identity_id']} {entry.get('candidate_id') or '?'}"
            if entry.get("target"):
                line += f" target={entry['target']}"
            if entry.get("verdict"):
                line += f" [{entry['verdict']}]"
            if entry.get("revision") is not None:
                line += f" rev={entry['revision']}"
            if entry.get("anchor"):
                line += f" anchor={entry['anchor']}"
            lines.append(line)
        return "\n".join(lines) if lines else "no identities"

    def snapshot(self) -> dict:
        return {
            "identities": self.all(),
            "anchors": self.anchors(),
            "counts": {
                "identities": len(self._items),
                "anchors": len(self.anchors()),
                "confirmed": sum(
                    1 for item in self._items if item["verdict"] == IDENTITY_ACCEPTED
                ),
            },
        }

    def stats(self) -> dict:
        return {
            "layer": "identity",
            "identities": len(self._items),
            "anchors": len(self.anchors()),
            "confirmed": sum(1 for item in self._items if item["verdict"] == IDENTITY_ACCEPTED),
            "rejected": sum(1 for item in self._items if item["verdict"] == IDENTITY_REJECTED),
        }

    # ---------------- internals ----------------

    def _find(self, key: str = "", *, target: str = "", box=None, position=None) -> dict | None:
        if key:
            for entry in self._items:
                if entry["candidate_id"] == key:
                    return entry
            return None
        normalized_target = _normalize_text(target)
        for entry in self._items:
            if normalized_target and _normalize_text(entry["target"]) != normalized_target:
                continue
            if box is not None and entry["box"] is not None and list(box) != list(entry["box"]):
                continue
            if position is not None and entry["position_flu"] is not None:
                if _distance(position, entry["position_flu"]) > self.merge_radius_m:
                    continue
            if normalized_target or box is not None or position is not None:
                return entry
        return None

    @staticmethod
    def _verdict(value) -> str | None:
        text = str(value or "").strip().lower()
        return text if text in IDENTITY_VERDICTS else None

    def _entry(self, entry: dict, score: float | None = None) -> dict:
        out = {
            "identity_id": entry["identity_id"],
            "candidate_id": entry["candidate_id"],
            "target": entry["target"],
            "box": list(entry["box"]) if entry["box"] is not None else None,
            "position_flu": (
                list(entry["position_flu"]) if entry["position_flu"] is not None else None
            ),
            "anchor": entry["anchor"],
            "anchor_source": entry["anchor_source"],
            "verdict": entry["verdict"],
            "revision": entry["revision"],
            "observation_ids": list(entry["observation_ids"]),
            "subtask_id": entry["subtask_id"],
            "task_id": entry["task_id"],
            "confidence": entry["confidence"],
            "note": entry["note"],
            "updates": entry["updates"],
            "step": entry["step"],
            "updated_t": round(entry["updated_t"], 3),
            "evidence": list(entry["evidence"]),
        }
        if score is not None:
            out["score"] = round(float(score), 3)
        return out

    def _evict(self) -> None:
        while len(self._items) > self.max_identities:
            oldest = min(self._items, key=lambda item: (item["updated_t"], item["seq"]))
            self._items.remove(oldest)


def _distance(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def _salience_of(events) -> float:
    """How much a frame matters: discoveries, conflicts and turning points count.

    Used by the duplicate merge (the event survives it) and by the per-event
    pixel budget (a frame that carries a discovery outlives a routine capture of
    the same event).
    """
    salient = [tag for tag in _as_str_list(events) if tag.lower() in SALIENT_EVENTS]
    return float(min(3, len(salient)))


def _frames_match(first: np.ndarray, second: np.ndarray) -> bool:
    """True when two stored frames are the very same picture.

    Render noise is tolerated (a mean absolute difference below
    ``MERGE_MEAN_ABS_TOL`` over the 0-255 range); a different shape never is, so
    a crop never merges into the frame it was cut from.
    """
    if first is None or second is None or first.shape != second.shape:
        return False
    diff = np.abs(first.astype(np.int16) - second.astype(np.int16))
    return float(diff.mean()) <= MERGE_MEAN_ABS_TOL


# ---------------------------------------------------------------------------
# observation log
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    obs_id: str
    seq: int
    t: float
    position_flu: tuple[float, float, float]
    yaw_rad: float
    events: list[str] = field(default_factory=list)
    goal_target: str | None = None
    labels: list[str] = field(default_factory=list)
    note: str = ""
    step: int | None = None
    episode_id: str | None = None
    image: np.ndarray | None = field(default=None, repr=False)
    image_path: str | None = None
    extra: dict = field(default_factory=dict)
    sightings: list[dict] = field(default_factory=list)
    negative: dict | None = None
    visibility: dict | None = None
    salience: float = 0.0
    repeat_count: int = 0
    merged_ids: list[str] = field(default_factory=list)
    last_t: float | None = None

    @property
    def target_name(self) -> str | None:
        """Legacy spelling of :attr:`goal_target` (still what the prompt reads)."""
        return self.goal_target

    @property
    def sighting(self) -> dict | None:
        """The newest sighting of this frame (None when it saw nothing)."""
        return self.sightings[-1] if self.sightings else None

    def evidence_kind(self) -> str:
        """What this record is: a sighting, a scoped null result, or a frame."""
        if self.sightings:
            return EVIDENCE_KIND_SIGHTING
        if self.negative:
            return EVIDENCE_KIND_NEGATIVE
        return EVIDENCE_KIND_OBSERVATION

    def goal_text(self) -> str:
        """Searchable text of the *task* target (never evidence of seeing it)."""
        return " ".join([self.goal_target or "", *self.labels]).lower()

    def sighting_text(self) -> str:
        """Searchable text of what the frame really showed."""
        parts = [sighting_text(item) for item in self.sightings]
        if self.negative:
            parts.append(str(self.negative.get("target") or ""))
        return " ".join(part for part in parts if part)

    def search_text(self) -> str:
        parts = [self.obs_id, self.goal_target or "", *self.labels, *self.events, self.note]
        parts.append(self.sighting_text())
        parts += [
            str(v) for v in self.extra.values() if isinstance(v, (str, int, float, bool))
        ]
        return " ".join(part for part in parts if part).lower()


class ObservationLog:
    """System-written observation archive with bounded, evidence-citing retrieval.

    This is the episodic layer: key frames saved by event, duplicate frames
    merged, pixels bounded per event. The other two layers hang off it
    (``log.working``, ``log.spatial``) and are filled by the same writes, so a
    caller that only knows ``record``/``add_sighting``/``add_negative`` still
    maintains the whole memory. ``spatial_memory=False`` /
    ``episodic_memory=False`` turn the layers off for the ablation arms; with
    ``episodic_memory=False`` this layer keeps *no evidence at all* (no pixels,
    no sighting, no null result) while the other two keep being fed by the same
    writes - see the module docstring for the exact contract.

    Frames are kept as uint8 HWC arrays; ``image_path`` writes the PNG on demand
    (and returns an absolute, resolvable path for MCP resources). Every entry
    returned by ``get``/``recent``/``query`` carries a resolvable image
    reference unless ``materialize_on_query=False`` or the episodic layer is
    off - goal.md requires resource URIs to resolve to real images, not to a
    bare file address.
    """

    def __init__(
        self,
        image_dir: str | Path | None = None,
        *,
        out_dir: str | Path | None = None,
        world=None,
        episode_id: str | None = None,
        max_entries: int | None = None,
        max_limit: int = DEFAULT_MAX_LIMIT,
        materialize_on_query: bool = True,
        persist: bool = False,
        clock=time.time,
        spatial_memory: bool = True,
        episodic_memory: bool = True,
        identity_memory: bool = True,
        merge_duplicates: bool = True,
        max_frames_per_event: int = DEFAULT_MAX_FRAMES_PER_EVENT,
        spatial: "SpatialMemory | None" = None,
        working: "WorkingMemory | None" = None,
    ):
        if image_dir is None and out_dir is not None:
            image_dir = Path(out_dir) / "memory_frames"
        self.image_dir = Path(image_dir).resolve() if image_dir is not None else None
        self.world = world  # pose source for records made without an explicit pose
        self.episode_id = str(episode_id) if episode_id else None
        self.max_entries = int(max_entries) if max_entries else None
        self.max_limit = max(1, int(max_limit))
        self.materialize_on_query = bool(materialize_on_query)
        self.persist = bool(persist)
        self.clock = clock
        self._clock = clock
        # layer 2 and 3 switches (the ablation arms) + the layers themselves
        self.spatial_memory = bool(spatial_memory)
        self.episodic_memory = bool(episodic_memory)
        self.identity_memory = bool(identity_memory)
        self.merge_duplicates = bool(merge_duplicates)
        self.max_frames_per_event = max(1, int(max_frames_per_event))
        self.working = working if working is not None else WorkingMemory(clock=clock)
        self.spatial = spatial if spatial is not None else (
            SpatialMemory(clock=clock) if self.spatial_memory else None
        )
        #: object identities + relation anchors: the layer that answers "which
        #: instance was it" after a delayed revisit. It is not the episodic
        #: evidence layer - turning it off removes exactly those fields while
        #: frames, sightings, places and questions stay (the identity/anchor
        #: ablation arm).
        self.identities = IdentityMemory(clock=clock) if self.identity_memory else None
        self._observations: list[Observation] = []
        self._by_id: dict[str, Observation] = {}
        #: ids folded into a canonical entry by the duplicate merge: the model
        #: was handed them, so they must keep resolving
        self._aliases: dict[str, str] = {}
        self._seq = 0
        self._merged_duplicates = 0
        self._evicted_for_events = 0
        self._stats = {"queries": 0, "returned": 0}
        self._last_write: dict = {}

    # ---------------- switches ----------------

    def switches(self) -> dict:
        """The layer switches this store runs with (recorded in reports).

        ``spatial_memory`` / ``episodic_memory`` say which layers the store
        keeps and returns; ``merge_duplicates`` is never on without the
        episodic layer (the duplicate merge is part of it), and
        ``max_frames_per_event`` is that layer's per-event pixel budget - it is
        inert while the layer is off.
        """
        return {
            "spatial_memory": bool(self.spatial_memory and self.spatial is not None),
            "episodic_memory": bool(self.episodic_memory),
            "identity_memory": bool(self.identity_memory and self.identities is not None),
            "merge_duplicates": bool(self.merge_duplicates and self.episodic_memory),
            "working_memory": self.working is not None,
            "max_frames_per_event": self.max_frames_per_event,
        }

    def layer_status(self) -> dict:
        """Compact state of the three layers (what a report or tool result says).

        ``episodic`` is ``off`` when the store keeps no evidence: the counts
        below it (``sightings``, ``negatives``, ``duplicates_merged``,
        ``events_over_capacity``) then stay zero, while ``entries`` keeps
        counting the capture records the other two layers are read from.
        """
        spatial = self.spatial.stats() if self.spatial is not None else {}
        working = self.working.stats() if self.working is not None else {}
        return {
            "layer": LAYER_EPISODIC,
            "episodic": "on" if self.episodic_memory else "off",
            "spatial": "on" if self.spatial is not None else "off",
            "identity": "on" if self.identities is not None else "off",
            "working": "on" if self.working is not None else "off",
            "entries": len(self._observations),
            "sightings": sum(1 for obs in self._observations if obs.sightings),
            "negatives": sum(1 for obs in self._observations if obs.negative),
            "duplicates_merged": self._merged_duplicates,
            "events_over_capacity": self._evicted_for_events,
            "covered_sectors": spatial.get("covered_sectors", 0),
            "places": spatial.get("places", 0),
            "identities": 0 if self.identities is None else len(self.identities),
            "open_questions": working.get("open_questions", 0),
        }

    # ---------------- identity / anchor writes ----------------

    def remember_identity(
        self,
        candidate_id: str = "",
        *,
        target: str = "",
        box=None,
        position=None,
        anchor: str = "",
        anchor_source: str = "",
        verdict: str = "",
        revision=None,
        observation_ids=(),
        subtask_id: str = "",
        task_id: str = "",
        confidence=None,
        note: str = "",
        step=None,
        evidence=None,
    ) -> dict | None:
        """Save one object fingerprint (and its anchor/verdict) in the layer.

        This is the public write path of the identity layer: sightings feed it
        automatically (:meth:`_fan_out_sightings`), and a confirmation merges
        its verdict/revision into the same entry through :meth:`confirm_identity`.
        A store built without the layer answers ``None`` - that is the
        identity/anchor ablation, not an error.
        """
        if self.identities is None:
            return None
        entry = self.identities.remember(
            candidate_id, target=target, box=box, position=position,
            anchor=anchor, anchor_source=anchor_source, verdict=verdict,
            revision=revision, observation_ids=observation_ids,
            subtask_id=subtask_id, task_id=task_id, confidence=confidence,
            note=note, step=step, evidence=evidence,
        )
        if anchor:
            # the relation anchor is also a persistent task fact: keep it on the
            # working layer so a resume brief can carry it even if the candidate
            # entry was evicted
            self.working.remember_anchor(
                anchor, target=target or entry.get("target", ""),
                source=anchor_source, step=step,
                evidence=EvidenceRef("identity", entry["identity_id"], "relation anchor"),
            )
        return entry

    def confirm_identity(
        self,
        candidate_id: str,
        verdict: str,
        *,
        revision=None,
        observation_ids=(),
        anchor: str = "",
        target: str = "",
        box=None,
        position=None,
        subtask_id: str = "",
        task_id: str = "",
        note: str = "",
        evidence=None,
    ) -> dict | None:
        """Merge a confirm verdict/revision into memory (not only the ledger)."""
        if self.identities is None:
            return None
        existing = self.identities.get(str(candidate_id or "").strip())
        if existing is None:
            return self.remember_identity(
                candidate_id, target=target, box=box, position=position,
                anchor=anchor, verdict=verdict, revision=revision,
                observation_ids=observation_ids, subtask_id=subtask_id,
                task_id=task_id, note=note, evidence=evidence,
            )
        entry = self.identities.remember(
            candidate_id, verdict=verdict, revision=revision,
            observation_ids=observation_ids, anchor=anchor, note=note,
            evidence=evidence,
        )
        if anchor:
            self.working.remember_anchor(
                anchor, target=target or existing.get("target", ""),
                candidate_id=str(candidate_id or ""),
                source="confirm_candidate",
                evidence=EvidenceRef("identity", entry["identity_id"], "relation anchor"),
            )
        return entry

    # ---------------- writing ----------------

    def record(
        self,
        image,
        pose=None,
        yaw=None,
        event=None,
        *,
        obs_id: str | None = None,
        goal_target: str | None = None,
        target_name: str | None = None,
        sighting=None,
        negative=None,
        visibility=None,
        coverage=None,
        labels=None,
        note: str = "",
        step: int | None = None,
        t: float | None = None,
        extra: dict | None = None,
        yaw_rad: float | None = None,
        timestamp: float | None = None,
        intrinsics: dict | None = None,
        image_path: str | None = None,
    ) -> str:
        """Append one observation, return its ``obs_id`` (``obs_1``, ``obs_2``...).

        ``image`` is a CHW float [0,1] tensor / HWC array / PIL image, or None
        for an event-only record. ``pose`` is the ``position_flu``-carrying pose
        (or a world/PoseState object); it defaults to the constructor ``world``.
        ``yaw`` is radians and defaults to the pose's own ``yaw_rad``. ``event``
        is a tag or list of tags (``"target_sighting"``); ``labels`` are extra
        searchable keywords. ``yaw_rad``/``timestamp`` are accepted as aliases
        for ``yaw``/``t`` (adapter keyword spellings).

        ``obs_id`` adopts an id minted elsewhere - typically the
        ``vlm_nav.observations.ObservationStore`` id of the same capture, so the
        log and the store are one namespace; the log's counter advances past the
        adopted id. ``intrinsics`` is the render intrinsics of the capture
        (``{width, height, hfov_deg}``) and ``image_path`` the on-disk frame, so
        a recalled entry still knows how to project its pixels and where its
        image lives.

        ``goal_target`` (or its legacy alias ``target_name``) is what the *task*
        is looking for - a capture inherits it, and it stays a task statement.
        ``sighting`` is what the frame really showed (validated: an image, a
        candidate/box/pixel and a basis - see :func:`normalize_sighting`);
        ``negative`` is a scoped null result; ``coverage`` is the checked-sector
        report of the capture. All three fan out into the spatial and working
        layers, and all three are *episodic evidence* - with
        ``episodic_memory=False`` the store keeps neither the pixels nor the
        sighting nor the null result (and the events that would have marked
        them); the capture record, the fan-out and the retrieval stay. A
        pixel-identical frame at the same pose and heading merges into the entry
        that already holds it (the discovery stays canonical and its id is
        returned) while the layer is on.
        """
        goal_target = str(goal_target) if goal_target else (
            str(target_name) if target_name else None
        )
        # the id comes first: a sighting written with the frame defaults its
        # image reference to that very observation
        self._seq += 1
        obs_id = str(obs_id) if obs_id else f"obs_{self._seq}"
        adopted = _OBS_ID_RE.match(obs_id)
        if adopted is not None:
            # the id came from the shared namespace: never mint it (or a lower
            # one) again automatically
            self._seq = max(self._seq, int(adopted.group(1)))
        if obs_id in self._by_id:
            raise ValueError(f"duplicate obs_id: {obs_id}")
        resolved_sighting = None
        if sighting is not None:
            # a sighting is a claim about a frame: an event-only record has no
            # image to show, so there is nothing to sight. It is validated
            # whether or not the episodic layer keeps it, so a malformed claim
            # is still refused instead of silently dropped
            sighting_image = obs_id if (image is not None or image_path) else None
            resolved_sighting, _problem = normalize_sighting(
                sighting, image=sighting_image, strict=True
            )
        negative_entry = None
        if negative is not None and negative is not False:
            negative_entry = self._normalize_negative(negative, target=goal_target)
        resolved_visibility = dict(visibility) if isinstance(visibility, Mapping) else None
        # what the store keeps *as evidence*: with the episodic layer off the
        # frame, the sightings and the null result stay out of the entry (while
        # the fan-out below still feeds the other two layers)
        keeps_evidence = bool(self.episodic_memory)

        if pose is None:
            pose = self.world
        if yaw is None:
            yaw = yaw_rad if yaw_rad is not None else getattr(pose, "yaw_rad", 0.0)
        if t is None:
            t = timestamp
        extra = dict(extra or {})
        if intrinsics:
            extra.setdefault("intrinsics", dict(intrinsics))
        if coverage is not None:
            extra.setdefault("coverage", jsonable(coverage))
        pixels = _to_uint8_hwc(image) if (image is not None and keeps_evidence) else None
        events = _as_str_list(event)
        if keeps_evidence:
            if resolved_sighting is not None and EVENT_SIGHTING not in events:
                events.append(EVENT_SIGHTING)
            if negative_entry is not None and EVENT_NOT_FOUND not in events:
                events.append(EVENT_NOT_FOUND)
        now = float(t if t is not None else self._clock())
        if resolved_sighting is not None:
            # a sighting without an explicit image is a sighting *of this frame*
            resolved_sighting["image"] = resolved_sighting.get("image") or obs_id
            if resolved_sighting.get("t") is None:
                resolved_sighting["t"] = now
        obs = Observation(
            obs_id=obs_id,
            seq=self._seq,
            t=now,
            position_flu=_as_position(pose),
            yaw_rad=_wrap_angle(float(yaw)),
            events=events,
            goal_target=goal_target,
            labels=_as_str_list(labels),
            note=str(note or ""),
            step=int(step) if step is not None else None,
            episode_id=self.episode_id,
            image=pixels,
            image_path=str(image_path) if (image_path and keeps_evidence) else None,
            extra=extra,
            negative=negative_entry if keeps_evidence else None,
            visibility=resolved_visibility if keeps_evidence else None,
            salience=_salience_of(events),
        )
        if keeps_evidence and resolved_sighting is not None:
            obs.sightings.append(resolved_sighting)
        scope = resolved_visibility
        if negative_entry is not None and scope is None:
            # a negative without a declared scope still knows the pose it was
            # made from - the field of view stays unknown rather than assumed
            scope = self._scope_of(obs)
        if obs.negative is not None and obs.visibility is None:
            obs.visibility = scope

        merged_into = self._merge_duplicate(obs) if self._merge_duplicates_enabled() else None
        if merged_into is not None:
            self._aliases[obs_id] = merged_into.obs_id
            self._merged_duplicates += 1
            self._last_write = {
                "obs_id": merged_into.obs_id, "recorded_id": obs_id, "merged": True,
                "sighting": len(merged_into.sightings), "negative": bool(merged_into.negative),
            }
            return merged_into.obs_id
        self._observations.append(obs)
        self._by_id[obs_id] = obs
        if self.persist and obs.image is not None:
            self._materialize(obs)
        self._evict()
        layers = self._fan_out(
            obs,
            coverage=coverage,
            sightings=[resolved_sighting] if resolved_sighting is not None else [],
            negative=negative_entry,
            scope=scope,
        )
        self._last_write = {
            "obs_id": obs_id, "merged": False, "layers": layers,
            "sighting": len(obs.sightings), "negative": bool(obs.negative),
            "episodic_memory": keeps_evidence,
        }
        return obs_id

    def add_sighting(self, obs_id: str, sighting, *, image=None, step: int | None = None,
                     note: str = "") -> dict:
        """Attach one or more sightings to a frame that already exists.

        This is the write path of the perception tools: ``ground_target`` found
        candidates *in* a recorded frame, so the sighting is not a new capture
        but a piece of evidence about that frame. Invalid sightings are refused
        item by item (``rejected`` names the reason) and an id no entry knows is
        reported as ``unknown_observation`` - a tool call never fails here.

        A store whose episodic layer is off answers
        ``disabled:episodic_memory_off`` with ``added=0``: the sightings are
        still validated and still fan out to the spatial and working layers
        (the candidate place and the task fact are theirs), but the evidence -
        the frame + candidate + basis pairing - is not kept.
        """
        entry = self._resolve(obs_id)
        if entry is None:
            return {"status": "unknown_observation", "obs_id": str(obs_id), "added": 0,
                    "rejected": [], "layers": {}}
        items = sighting if isinstance(sighting, (list, tuple)) else [sighting]
        image_ref = image
        if image_ref in (None, ""):
            image_ref = entry.image_path or entry.obs_id
        accepted, rejected = [], []
        for item in items:
            resolved, problem = normalize_sighting(item, image=image_ref)
            if resolved is None:
                rejected.append({"sighting": jsonable(item), "reason": problem})
                continue
            resolved["step"] = step if step is not None else resolved.get("step")
            if self.episodic_memory:
                entry.sightings.append(resolved)
                if EVENT_SIGHTING not in entry.events:
                    entry.events.append(EVENT_SIGHTING)
                entry.salience = max(entry.salience, _salience_of(entry.events))
            accepted.append(resolved)
        layers = {}
        if accepted:
            if note and self.episodic_memory:
                entry.note = str(note)[:DEFAULT_MAX_CONTENT_CHARS]
            layers = self._fan_out_sightings(entry, accepted)
        added = len(accepted) if self.episodic_memory else 0
        if not accepted:
            status = "rejected"
        else:
            status = "ok:add_sighting" if self.episodic_memory else STATUS_EPISODIC_OFF
        self._last_write = {"obs_id": entry.obs_id, "sighting": len(entry.sightings),
                            "added": added, "rejected": len(rejected), "layers": layers}
        return {
            "status": status,
            "obs_id": entry.obs_id,
            "added": added,
            "rejected": rejected,
            "layers": layers,
        }

    def add_negative(self, obs_id: str, *, target: str = "", reason: str = "not_found",
                     visibility=None, note: str = "", step: int | None = None) -> dict:
        """Mark a frame as a *scoped* null result: not found under this view.

        The scope is the whole point: a failed look is recorded with the
        viewpoint, heading, field of view and region it was made under (derived
        from the record when the caller does not pass one), never as "this area
        holds nothing".

        A store whose episodic layer is off answers
        ``disabled:episodic_memory_off`` with ``added=0``: the sector the look
        checked (spatial) and the failed attempt it records (working) are still
        written, the null result itself is not - a null result is evidence
        about a view, and the layer that keeps evidence is off.
        """
        entry = self._resolve(obs_id)
        if entry is None:
            return {"status": "unknown_observation", "obs_id": str(obs_id), "added": 0,
                    "rejected": [], "layers": {}}
        negative = self._normalize_negative(
            {"target": target, "reason": reason, "note": note}, target=entry.goal_target
        )
        scope = dict(visibility) if isinstance(visibility, Mapping) else self._scope_of(entry)
        if self.episodic_memory:
            entry.negative = negative
            entry.visibility = scope
            if EVENT_NOT_FOUND not in entry.events:
                entry.events.append(EVENT_NOT_FOUND)
            entry.salience = max(entry.salience, _salience_of(entry.events))
        if step is not None:
            entry.step = step
        layers = self._fan_out_negative(entry, negative, scope)
        kept = bool(self.episodic_memory)
        self._last_write = {"obs_id": entry.obs_id, "negative": kept, "layers": layers}
        return {"status": "ok:add_negative" if kept else STATUS_EPISODIC_OFF,
                "obs_id": entry.obs_id, "added": 1 if kept else 0,
                "rejected": [], "layers": layers}

    def _normalize_negative(self, value, *, target: str | None = None) -> dict:
        if isinstance(value, Mapping):
            data = {str(key): jsonable(item) for key, item in value.items()}
        else:
            data = {"reason": str(value)}
        resolved_target = str(data.get("target") or data.get("goal_target") or target or "").strip()
        reason = str(data.get("reason") or data.get("outcome") or "not_found").strip()
        return {
            "target": resolved_target[:DEFAULT_MAX_CONTENT_CHARS],
            "reason": reason[:DEFAULT_MAX_CONTENT_CHARS],
            "note": str(data.get("note") or "")[:DEFAULT_MAX_CONTENT_CHARS],
            "t": _as_float(data.get("t")),
        }

    def _scope_of(self, obs: Observation) -> dict:
        intrinsics = obs.extra.get("intrinsics") if isinstance(obs.extra, dict) else None
        intrinsics = intrinsics if isinstance(intrinsics, Mapping) else {}
        region = None
        if isinstance(obs.extra, dict):
            crop = obs.extra.get("crop")
            if isinstance(crop, Mapping):
                region = crop.get("region_uvwh")
        return view_scope(
            obs.position_flu,
            yaw_rad=obs.yaw_rad,
            hfov_deg=intrinsics.get("hfov_deg"),
            pitch_down_deg=float(obs.extra.get("pitch_down_deg") or 0.0) if isinstance(obs.extra, dict) else 0.0,
            region=region,
            observation_id=obs.obs_id,
        )

    # ---------------- duplicate merging (episodic maintenance) ----------------

    def _merge_duplicates_enabled(self) -> bool:
        return bool(self.episodic_memory and self.merge_duplicates)

    def _merge_duplicate(self, obs: Observation) -> Observation | None:
        """The earlier entry holding the very same picture, if there is one."""
        if obs.image is None or not self._observations:
            return None
        pitched = float(obs.extra.get("pitch_down_deg") or 0.0)
        for candidate in reversed(self._observations[-8:]):
            if candidate.image is None:
                continue
            if abs(float(candidate.extra.get("pitch_down_deg") or 0.0) - pitched) > 1e-6:
                continue
            if _distance(candidate.position_flu, obs.position_flu) > MERGE_POS_TOL_M:
                continue
            dyaw = abs(math.degrees(_wrap_angle(candidate.yaw_rad - obs.yaw_rad)))
            if dyaw > MERGE_YAW_TOL_DEG:
                continue
            if not _frames_match(candidate.image, obs.image):
                continue
            self._absorb(candidate, obs)
            return candidate
        return None

    def _absorb(self, keep: Observation, extra: Observation) -> None:
        """Fold ``extra`` into ``keep``: the discovery stays, nothing is lost."""
        keep.repeat_count += 1
        keep.merged_ids.append(extra.obs_id)
        keep.last_t = extra.t
        keep.salience = max(keep.salience, extra.salience)
        for tag in extra.events:
            if tag not in keep.events:
                keep.events.append(tag)
        for label in extra.labels:
            if label not in keep.labels:
                keep.labels.append(label)
        for item in extra.sightings:
            if item.get("image") == extra.obs_id:
                item["image"] = keep.obs_id
            keep.sightings.append(item)
        if extra.negative:
            keep.negative = extra.negative
        if extra.visibility and not keep.visibility:
            keep.visibility = extra.visibility
        if extra.note and not keep.note:
            keep.note = extra.note
        if extra.image_path and not keep.image_path:
            keep.image_path = extra.image_path
        keep.extra.setdefault("duplicates", []).append(
            {
                "obs_id": extra.obs_id,
                "t": round(extra.t, 3),
                "step": extra.step,
                "events": list(extra.events),
            }
        )
        del keep.extra["duplicates"][:-8]
        if extra.step is not None:
            keep.extra.setdefault("last_step", extra.step)
            keep.extra["last_step"] = extra.step
        if extra.sightings:
            self._fan_out_sightings(keep, extra.sightings)
        if extra.negative:
            self._fan_out_negative(keep, extra.negative, extra.visibility or {})

    # ---------------- fan-out into the spatial / working layers ----------------

    def _fan_out(self, obs: Observation, *, coverage=None, sightings=None,
                 negative=None, scope=None) -> dict:
        """Offer one write to the spatial and working layers.

        ``sightings`` / ``negative`` / ``scope`` default to what the record
        itself carries; a caller passes them explicitly when the episodic layer
        did not keep the evidence. Either way the layers are fed: the candidate
        place of a grounding and the checked sector of a failed look are *their*
        facts, not the evidence layer's.
        """
        sightings = list(obs.sightings) if sightings is None else list(sightings)
        negative = obs.negative if negative is None else negative
        scope = (obs.visibility or {}) if scope is None else scope
        layers: dict[str, str] = {}
        if sightings:
            layers.update(self._fan_out_sightings(obs, sightings))
        if negative:
            layers.update(self._fan_out_negative(obs, negative, scope))
        if coverage and self.spatial is not None:
            report = coverage if isinstance(coverage, Mapping) else {}
            checked = [pair for pair in (_as_pair(item) for item in report.get("checked_deg") or [])
                       if pair is not None]
            if checked:
                self.spatial.note_coverage(
                    checked, position=obs.position_flu, observation_id=obs.obs_id, step=obs.step,
                )
                layers["spatial"] = "ok:note_coverage"
        return layers

    def _fan_out_sightings(self, obs: Observation, sightings: list[dict]) -> dict:
        layers: dict[str, str] = {}
        for sighting in sightings:
            target = sighting.get("target") or obs.goal_target or ""
            candidate = sighting.get("candidate")
            anchor = str(sighting.get("anchor") or "")
            source = f"sighting:{obs.obs_id}"
            identity = None
            if self.identities is not None:
                identity = self.identities.remember(
                    candidate or f"obs:{obs.obs_id}", target=target,
                    box=sighting.get("box"), position=sighting.get("position_flu"),
                    anchor=anchor, anchor_source="sighting" if anchor else "",
                    confidence=sighting.get("confidence"),
                    observation_ids=[obs.obs_id], step=obs.step,
                    evidence=EvidenceRef("observation", obs.obs_id, "identity sighting"),
                )
                layers["identity"] = "ok:remember_identity"
            if self.spatial is not None:
                position = sighting.get("position_flu")
                if position is not None:
                    self.spatial.remember_candidate(
                        candidate or f"obs:{obs.obs_id}", position, target=target,
                        confidence=sighting.get("confidence"), source=source,
                        observation_id=obs.obs_id, step=obs.step,
                    )
                    layers["spatial"] = "ok:remember_candidate"
                else:
                    # a sighting whose range is unknown is not a place yet: the
                    # direction it was seen in is all anyone has
                    layers.setdefault(
                        "spatial", "skipped:no_position" if candidate else "skipped:no_candidate"
                    )
            if self.working is not None:
                if candidate:
                    self.working.note_candidate(
                        candidate, sighting.get("position_flu"), target=target,
                        confidence=sighting.get("confidence"), source=source, step=obs.step,
                        anchor=anchor, identity=identity,
                        evidence=EvidenceRef("observation", obs.obs_id, "sighting"),
                    )
                    layers["working"] = "ok:note_candidate"
                else:
                    layers.setdefault("working", "skipped:no_candidate")
                if target:
                    resolved = self.working.resolve_open(
                        target, answer=f"seen in {obs.obs_id}",
                        evidence=EvidenceRef("observation", obs.obs_id, "sighting"),
                    )
                    if resolved:
                        layers["questions_resolved"] = f"{len(resolved)}"
        return layers

    def _fan_out_negative(self, obs: Observation, negative: dict, scope: Mapping | None) -> dict:
        layers: dict[str, str] = {}
        target = negative.get("target") or obs.goal_target or ""
        scope = scope if isinstance(scope, Mapping) else {}
        if self.spatial is not None:
            sector = _as_pair(scope.get("coverage_deg"))
            if sector is not None:
                self.spatial.note_coverage(
                    [sector], position=scope.get("position_flu") or obs.position_flu,
                    observation_id=obs.obs_id, step=obs.step,
                    no_result_for=[target] if target else [],
                )
                layers["spatial"] = "ok:note_coverage"
            else:
                layers["spatial"] = "skipped:no_scope"
        if self.working is not None:
            self.working.note_failure(
                f"look for '{target}'" if target else "look for the target",
                reason=str(negative.get("reason") or "not_found"),
                strategy=str(negative.get("note") or ""), step=obs.step,
                evidence=EvidenceRef("observation", obs.obs_id, "scoped null result"),
            )
            if target:
                self.working.open_question(
                    f"where is {target}?", about=target, step=obs.step,
                    evidence=EvidenceRef("observation", obs.obs_id, "searched without a result"),
                )
            layers["working"] = "ok:note_failure"
        return layers

    def _evict(self) -> None:
        """Bound RAM: metadata is kept for every observation, the pixels of the
        oldest ones are dropped once ``max_entries`` frames are stored.

        The episodic layer adds a per-event rule: at most
        ``max_frames_per_event`` frames of one event keep their pixels. Those
        frames are materialized to disk first, so the image of an evicted
        sighting is still a resolvable reference.
        """
        if self.max_entries:
            stored = [o for o in self._observations if o.image is not None]
            for obs in stored[: -self.max_entries]:
                obs.image = None
        if not self.episodic_memory:
            return
        counts: dict[str, int] = {}
        for obs in self._observations:
            if obs.image is None:
                continue
            for tag in obs.events:
                counts[tag] = counts.get(tag, 0) + 1
        overflow = {tag for tag, count in counts.items() if count > self.max_frames_per_event}
        if not overflow:
            return
        # the frames that carry the most keep their pixels: a discovery, a
        # conflict or a decision turning point outlives a routine capture of the
        # same event, and the newest wins a tie - what leaves RAM is written to
        # disk first, so its image reference still resolves
        kept: dict[str, int] = {}
        ordered = sorted(
            (obs for obs in self._observations if obs.image is not None),
            key=lambda obs: (obs.salience, obs.seq),
            reverse=True,
        )
        for obs in ordered:
            tags = [tag for tag in obs.events if tag in overflow]
            if not tags:
                continue
            if all(kept.get(tag, 0) < self.max_frames_per_event for tag in tags):
                for tag in tags:
                    kept[tag] = kept.get(tag, 0) + 1
                continue
            self._materialize(obs)  # keep the frame addressable before it leaves RAM
            obs.image = None
            self._evicted_for_events += 1

    # ---------------- reading ----------------

    def __len__(self) -> int:
        return len(self._observations)

    def ids(self) -> list[str]:
        return [obs.obs_id for obs in self._observations]

    def _resolve(self, obs_id: str) -> Observation | None:
        """An observation by id - following the merge aliases of folded ids."""
        key = str(obs_id)
        obs = self._by_id.get(key)
        if obs is None and key in self._aliases:
            obs = self._by_id.get(self._aliases[key])
        return obs

    def get(self, obs_id: str) -> dict | None:
        obs = self._resolve(obs_id)
        return self._entry(obs) if obs is not None else None

    def recent(self, k: int = 5) -> list[dict]:
        """The ``k`` most recent observations, newest first."""
        limit = _clamp_limit(k, self.max_limit)
        return [self._entry(obs) for obs in reversed(self._observations[-limit:])]

    def query(self, q: str | dict, limit: int | None = DEFAULT_QUERY_LIMIT) -> list[dict]:
        """Retrieve at most ``limit`` relevant observations (newest first on ties).

        See the module docstring for the accepted query keys. Unmatched
        observations are dropped rather than scored zero, so an empty result
        means "no such evidence"; a query with no criteria at all returns the
        most recent observations. The effective budget is the tighter of
        ``limit`` and the query's own ``limit`` key, never above ``max_limit``.

        ``target`` matches the *sighting* fields (what the frames really showed:
        the sighting's target, its basis, its candidate, its attributes) and a
        matched hit reports ``match="sighting"``. Only when no sighting matches
        do records whose *goal target* matches come back, each labelled
        ``match="goal_frame"`` - a frame captured while the task looked for
        something is a place to look again, never evidence that it was seen.
        ``goal_target`` asks for the task-target frames explicitly, ``sighting``
        for frames that saw something, ``negative`` for scoped null results.
        """
        spec = {"text": q} if isinstance(q, str) else dict(q or {})
        limit = _resolve_limit(limit, spec.get("limit"), self.max_limit)
        text = spec.get("text") or spec.get("keywords") or ""
        tokens = _tokenize(text)
        target_tokens = _tokenize(spec.get("target") or spec.get("target_name") or "")
        goal_tokens = _tokenize(spec.get("goal_target") or spec.get("goal") or "")
        sighting_spec = spec.get("sighting")
        sighting_tokens = _tokenize(sighting_spec) if isinstance(sighting_spec, str) else []
        sighting_only = _as_bool(sighting_spec) and not sighting_tokens
        negative_only = _as_bool(spec.get("negative"))
        events = [e.lower() for e in _as_str_list(spec.get("event") or spec.get("events"))]
        since = spec.get("since", spec.get("after"))
        before = spec.get("before", spec.get("before_t"))
        since_step = spec.get("since_step")
        before_step = spec.get("before_step")
        step = spec.get("step")
        point = spec.get("point", spec.get("near"))
        max_distance_m = float(spec.get("max_distance_m", 12.0))
        fov_deg = float(spec.get("fov_deg", 90.0))
        origin = spec.get("origin", spec.get("from_pos"))
        bearing_deg = spec.get("bearing_deg")
        tolerance_deg = float(spec.get("tolerance_deg", 45.0))

        rows: list[tuple[float, Observation, dict]] = []
        for obs in self._observations:
            if since is not None and obs.t < float(since):
                continue
            if before is not None and obs.t > float(before):
                continue
            if step is not None and obs.step != int(step):
                continue
            if since_step is not None and (obs.step is None or obs.step < int(since_step)):
                continue
            if before_step is not None and (obs.step is None or obs.step > int(before_step)):
                continue
            haystack = obs.search_text()
            if tokens:
                hits = sum(1 for tok in tokens if tok in haystack)
                if hits == 0:
                    continue
                score = hits / len(tokens)
            else:
                score = 0.0
            computed: dict[str, Any] = {}
            # what the frame really showed, not what the task was looking for
            match = None
            sighting_text_value = obs.sighting_text()
            if target_tokens:
                if all(tok in sighting_text_value for tok in target_tokens):
                    match = "sighting"
                elif obs.negative and all(tok in str(obs.negative.get("target") or "").lower()
                                          for tok in target_tokens):
                    match = "negative"
                elif all(tok in obs.goal_text() for tok in target_tokens):
                    match = "goal_frame"
                else:
                    continue
            if goal_tokens and not all(tok in obs.goal_text() for tok in goal_tokens):
                continue
            if sighting_only and not obs.sightings:
                continue
            if sighting_tokens and not all(tok in sighting_text_value for tok in sighting_tokens):
                continue
            if negative_only and not obs.negative:
                continue
            if events:
                events_text = " ".join(obs.events).lower()
                if not any(ev in events_text for ev in events):
                    continue
            if point is not None:
                rel = _as_point3(point) - np.asarray(obs.position_flu)
                dist = float(np.linalg.norm(rel))
                if dist > max_distance_m:
                    continue
                rel_bearing = _wrap_angle(math.atan2(rel[1], rel[0]) - obs.yaw_rad)
                rel_deg = math.degrees(rel_bearing)
                if abs(rel_deg) > fov_deg / 2.0:
                    continue
                computed["bearing_deg"] = round(rel_deg, 1)
                computed["distance_m"] = round(dist, 2)
                score += 1.0
                if max_distance_m > 0:
                    score += 0.5 * (1.0 - dist / max_distance_m)
                score += 0.5 * (1.0 - abs(rel_deg) / max(fov_deg / 2.0, 1e-6))
            if origin is not None and bearing_deg is not None:
                rel = np.asarray(obs.position_flu) - _as_point3(origin)
                obs_bearing = math.degrees(math.atan2(rel[1], rel[0]))
                delta = abs(_wrap_angle(math.radians(obs_bearing - float(bearing_deg))))
                if math.degrees(delta) > tolerance_deg:
                    continue
                computed["obs_bearing_deg"] = round(obs_bearing, 1)
                score += 1.0 * (1.0 - math.degrees(delta) / max(tolerance_deg, 1e-6))
            if match is not None:
                computed["match"] = match
                if match == "sighting":
                    score += 0.5  # a sighting outranks a frame that merely looked
            rows.append((score, obs, computed))

        if target_tokens:
            sighting_rows = [row for row in rows if row[2].get("match") == "sighting"]
            if sighting_rows:
                # sighting evidence first: the goal-target frames are the fallback
                rows = sighting_rows
        rows.sort(key=lambda row: (row[0], row[1].seq), reverse=True)
        out = [
            self._entry(obs, score=score, extra_fields=computed)
            for score, obs, computed in rows[:limit]
        ]
        self._stats["queries"] += 1
        self._stats["returned"] += len(out)
        return out

    def describe(self, limit: int = 3) -> str:
        """Compact observation status text for the prompt (newest first)."""
        if not self._observations:
            return "no observations"
        lines = []
        for obs in reversed(self._observations[-_clamp_limit(limit, self.max_limit) :]):
            events = ",".join(obs.events) or "-"
            line = (
                f"{obs.obs_id} step={obs.step} yaw={math.degrees(obs.yaw_rad):+.0f} deg "
                f"events=[{events}]"
            )
            if obs.goal_target:
                line += f" target={obs.goal_target}"
            if obs.sightings:
                line += f" sighting={obs.sightings[-1].get('candidate') or 'box'}"
            lines.append(line)
        return "\n".join(lines)

    def stats(self) -> dict:
        return {
            "entries": len(self._observations),
            "stored_images": sum(1 for o in self._observations if o.image is not None),
            "persisted_images": sum(1 for o in self._observations if o.image_path),
            "queries": self._stats["queries"],
            "returned": self._stats["returned"],
            "limit_budget": self.max_limit,
            "sightings": sum(1 for o in self._observations if o.sightings),
            "negatives": sum(1 for o in self._observations if o.negative),
            "merged_duplicates": self._merged_duplicates,
            "evicted_for_events": self._evicted_for_events,
            "switches": self.switches(),
        }

    # ---------------- frames for MCP resources ----------------

    def image_path(self, obs_id: str) -> str | None:
        """Write the frame to PNG (once) and return its absolute path.

        Raises KeyError for an unknown ``obs_id``; returns None when the record
        exists but its pixels are gone (never stored, or evicted).
        """
        obs = self._resolve(obs_id)
        if obs is None:
            raise KeyError(f"unknown obs_id: {obs_id}")
        return self._materialize(obs)

    def frame(self, obs_id: str) -> torch.Tensor | None:
        """Decoded frame as a CHW float [0,1] tensor, ready to inspect/re-render.

        Same error contract as ``image_path``: KeyError for an unknown id, None
        when the pixels are unavailable.
        """
        obs = self._resolve(obs_id)
        if obs is None:
            raise KeyError(f"unknown obs_id: {obs_id}")
        if obs.image is None:
            path = self._materialize(obs)
            if path is None:
                return None
            try:
                arr = np.asarray(Image.open(path).convert("RGB"))
            except Exception:  # unreadable frame -> treat as unavailable
                return None
        else:
            arr = obs.image
        # float conversion copies, so the tensor owns writable memory
        arr = np.asarray(arr, dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1) / 255.0

    def _image_dir(self) -> Path:
        if self.image_dir is None:
            self.image_dir = Path(tempfile.mkdtemp(prefix="vlm_nav_memory_"))
        return self.image_dir

    def _materialize(self, obs: Observation) -> str | None:
        if obs.image_path and Path(obs.image_path).exists():
            return obs.image_path
        if obs.image is None:
            return obs.image_path
        path = self._image_dir() / f"{obs.obs_id}.png"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(obs.image).save(path)
        except Exception:  # a broken frame must not break the runtime
            return None
        obs.image_path = str(path)
        obs.image = None  # pixels now live on disk
        return obs.image_path

    def _entry(
        self, obs: Observation, score: float | None = None, extra_fields: dict | None = None
    ) -> dict:
        if self.materialize_on_query or self.persist:
            self._materialize(obs)
        entry = {
            "obs_id": obs.obs_id,
            "seq": obs.seq,
            "step": obs.step,
            "t": round(obs.t, 3),
            "position_flu": [round(v, 3) for v in obs.position_flu],
            "yaw_rad": round(obs.yaw_rad, 4),
            "yaw_deg": round(math.degrees(obs.yaw_rad), 1),
            "events": list(obs.events),
            "goal_target": obs.goal_target,
            "target_name": obs.goal_target,  # legacy spelling of the same field
            "labels": list(obs.labels),
            "note": obs.note,
            "image": obs.image_path,
            "evidence_kind": obs.evidence_kind(),
            "repeat_count": obs.repeat_count,
            "salience": obs.salience,
        }
        if obs.sightings:
            entry["sightings"] = [jsonable(item) for item in obs.sightings]
        # the field is always present: "no sighting here" is information too
        entry["sighting"] = jsonable(obs.sightings[-1]) if obs.sightings else None
        if self.identities is not None and obs.sightings:
            # identity/anchor of the instance this frame showed: retrieval ranks
            # a hit by them, so they travel with the hit (a store built without
            # the identity layer simply has neither field)
            for sighting in reversed(obs.sightings):
                candidate = str(sighting.get("candidate") or "")
                identity = self.identities.get(candidate) if candidate else None
                if identity is None and candidate:
                    continue
                if identity is not None:
                    entry["identity"] = {
                        "identity_id": identity["identity_id"],
                        "candidate_id": identity["candidate_id"],
                        "target": identity["target"],
                        "verdict": identity["verdict"],
                        "revision": identity["revision"],
                        "observation_ids": list(identity["observation_ids"]),
                    }
                    if identity["anchor"]:
                        entry["anchor"] = identity["anchor"]
                        entry["anchor_source"] = identity["anchor_source"]
                    break
        if obs.negative:
            entry["negative"] = jsonable(obs.negative)
        if obs.visibility:
            entry["visibility"] = jsonable(obs.visibility)
            entry["scope_note"] = scope_note(obs.visibility)
        if obs.merged_ids:
            entry["merged_ids"] = list(obs.merged_ids)
        if obs.last_t is not None:
            entry["last_t"] = round(obs.last_t, 3)
        if obs.episode_id:
            entry["episode_id"] = obs.episode_id
        if obs.extra:
            entry["extra"] = jsonable(obs.extra)
        if obs.extra.get("intrinsics"):
            # the render intrinsics of the capture: a recalled hit must be
            # projectable, not only addressable
            entry["intrinsics"] = jsonable(obs.extra["intrinsics"])
        if obs.image_path:
            entry["image_path"] = obs.image_path
        if extra_fields:
            entry.update(extra_fields)
        if score is not None:
            entry["score"] = round(float(score), 3)
        refs = [EvidenceRef("observation", obs.obs_id, note=self._ref_note(obs))]
        if obs.image_path:
            refs.append(EvidenceRef("image", obs.image_path, note="frame"))
        entry["evidence"] = refs
        return entry

    @staticmethod
    def _ref_note(obs: Observation) -> str:
        note = f"yaw_deg={math.degrees(obs.yaw_rad):.0f} step={obs.step}"
        if obs.events:
            note += f" events={','.join(obs.events)}"
        return note


def build_memory(
    world=None,
    out_dir: str | Path | None = None,
    image_dir: str | Path | None = None,
    episode_id: str | None = None,
    *,
    spatial_memory: bool = True,
    episodic_memory: bool = True,
    identity_memory: bool = True,
    merge_duplicates: bool = True,
    max_frames_per_event: int = DEFAULT_MAX_FRAMES_PER_EVENT,
    **kwargs,
) -> ObservationLog:
    """The archive with its layers wired - the factory a runner builds memory with.

    ``spatial_memory`` / ``episodic_memory`` are the switches of the ablation
    arms; ``identity_memory=False`` removes exactly the identity/anchor fields
    (object fingerprints, confirm verdicts, relation anchors) while the frames,
    sightings, places and questions stay: an identity ablation, not a wiped
    archive. ``switches()`` reports what came out of the call, so a report can
    say which layers an episode really ran with. ``episodic_memory=False``
    builds the store without its evidence layer: no frame, sighting or null
    result is written or returned while the spatial layer (places, candidate
    positions, coverage) and the working layer stay on - see the module
    docstring.
    """
    return ObservationLog(
        image_dir=image_dir,
        out_dir=out_dir,
        world=world,
        episode_id=episode_id,
        spatial_memory=spatial_memory,
        episodic_memory=episodic_memory,
        identity_memory=identity_memory,
        merge_duplicates=merge_duplicates,
        max_frames_per_event=max_frames_per_event,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# hypothesis store
# ---------------------------------------------------------------------------


@dataclass
class Hypothesis:
    id: str
    kind: str
    content: str
    status: str
    seq: int
    created_t: float
    updated_t: float
    data: dict | None = None
    note: str = ""
    support: list[EvidenceRef] = field(default_factory=list)
    refute: list[EvidenceRef] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def search_text(self) -> str:
        parts = [self.id, self.kind, self.content, self.note, *self.reasons]
        parts += [f"{e.kind}:{e.ref}" for e in (*self.support, *self.refute)]
        return " ".join(parts).lower()


def _content_text(content) -> tuple[str, dict | None]:
    if isinstance(content, Mapping):
        data = {str(k): jsonable(v) for k, v in content.items()}
        text = data.get("text") or data.get("claim") or data.get("content")
        if not text:
            text = "; ".join(
                f"{k}={v}" for k, v in data.items() if k not in ("kind", "text", "content")
            )
        return str(text or ""), data
    return str(content or ""), None


class HypothesisStore:
    """Model-proposed, revisable beliefs with explicit support/refute evidence.

    Status transitions (deterministic, all audited by the retained evidence
    lists):

    - ``add`` -> confirmed when the initial support already reaches
      ``confirm_support`` distinct citations, else active.
    - ``support`` -> re-opens a rejected hypothesis as active (a belief is
      revisable, not a tombstone) and confirms it once ``confirm_support``
      distinct citations are reached.
    - ``refute`` -> rejected once ``refute_refutes`` distinct citations are
      reached.

    Repeating a call with the same citation is a no-op for the counts: the
    returned entry reports ``added=0`` so the caller can detect a duplicate.
    """

    def __init__(
        self,
        *,
        confirm_support: int = 2,
        reject_refutes: int = 1,
        max_limit: int = DEFAULT_MAX_LIMIT,
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
        prefix: str = "hyp",
        episode_id: str | None = None,
        clock=time.time,
    ):
        self.confirm_support = max(1, int(confirm_support))
        self.reject_refutes = max(1, int(reject_refutes))
        self.max_limit = max(1, int(max_limit))
        self.max_content_chars = max(32, int(max_content_chars))
        self.prefix = str(prefix or "hyp")
        self.episode_id = str(episode_id) if episode_id else None
        self._clock = clock
        self._hypotheses: list[Hypothesis] = []
        self._by_id: dict[str, Hypothesis] = {}
        self._seq = 0
        self._stats = {"queries": 0, "returned": 0}

    # ---------------- writing ----------------

    def add(
        self,
        content,
        evidence=None,
        *,
        kind: str | None = None,
        hyp_id: str | None = None,
        note: str = "",
    ) -> dict:
        """Register a hypothesis; returns its entry dict (``id`` = ``hyp_1``...).

        ``content`` is a claim string or a structured mapping such as
        ``{"kind": "identity", "subject": "red box", "claim": "is the target"}``.
        """
        self._seq += 1
        hyp_id = str(hyp_id) if hyp_id else f"{self.prefix}_{self._seq}"
        if hyp_id in self._by_id:
            raise ValueError(f"duplicate hypothesis id: {hyp_id}")
        text, data = _content_text(content)
        text = text[: self.max_content_chars]
        resolved_kind = kind or (data.get("kind") if data else None) or "note"
        now = self._clock()
        hyp = Hypothesis(
            id=hyp_id,
            kind=str(resolved_kind),
            content=text,
            status=STATUS_ACTIVE,
            seq=self._seq,
            created_t=now,
            updated_t=now,
            data=data,
            note=str(note or ""),
            support=normalize_evidence(evidence),
        )
        if len(hyp.support) >= self.confirm_support:
            hyp.status = STATUS_CONFIRMED
        self._hypotheses.append(hyp)
        self._by_id[hyp_id] = hyp
        return self._entry(hyp, added=len(hyp.support))

    def support(self, hyp_id: str, evidence) -> dict:
        hyp = self._require(hyp_id)
        added = _merge_evidence(hyp.support, normalize_evidence(evidence))
        hyp.updated_t = self._clock()
        if added and hyp.status == STATUS_REJECTED:
            hyp.status = STATUS_ACTIVE  # only genuinely new evidence re-opens a belief
        if len(hyp.support) >= self.confirm_support:
            hyp.status = STATUS_CONFIRMED
        return self._entry(hyp, added=len(added))

    def refute(self, hyp_id: str, evidence, reason: str = "") -> dict:
        hyp = self._require(hyp_id)
        added = _merge_evidence(hyp.refute, normalize_evidence(evidence))
        if reason:
            hyp.reasons.append(str(reason)[: self.max_content_chars])
        hyp.updated_t = self._clock()
        if len(hyp.refute) >= self.reject_refutes:
            hyp.status = STATUS_REJECTED
        return self._entry(hyp, added=len(added))

    # ---------------- reading ----------------

    def __len__(self) -> int:
        return len(self._hypotheses)

    def ids(self) -> list[str]:
        return [hyp.id for hyp in self._hypotheses]

    def get(self, hyp_id: str) -> dict | None:
        hyp = self._by_id.get(str(hyp_id))
        return self._entry(hyp) if hyp is not None else None

    def query(self, q: str | dict, limit: int | None = DEFAULT_QUERY_LIMIT) -> list[dict]:
        """Retrieve at most ``limit`` hypotheses; live beliefs outrank rejected
        ones on equal relevance. Keys: ``status``, ``kind``, ``text`` (or
        ``target``/``keywords``), ``limit`` (tighter of the two wins)."""
        spec = {"text": q} if isinstance(q, str) else dict(q or {})
        limit = _resolve_limit(limit, spec.get("limit"), self.max_limit)
        tokens = _tokenize(
            spec.get("text")
            or spec.get("keywords")
            or spec.get("content")
            or spec.get("target")
            or ""
        )
        status = spec.get("status")
        kinds = [k.lower() for k in _as_str_list(spec.get("kind"))]
        rows: list[tuple[float, Hypothesis]] = []
        for hyp in self._hypotheses:
            if status and hyp.status != str(status):
                continue
            if kinds and hyp.kind.lower() not in kinds:
                continue
            score = _STATUS_WEIGHT.get(hyp.status, 0.0)
            if tokens:
                haystack = hyp.search_text()
                hits = sum(1 for tok in tokens if tok in haystack)
                if hits == 0:
                    continue
                score += hits / len(tokens)
            rows.append((score, hyp))
        rows.sort(key=lambda row: (row[0], row[1].updated_t, row[1].seq), reverse=True)
        out = [self._entry(hyp, score=score) for score, hyp in rows[:limit]]
        self._stats["queries"] += 1
        self._stats["returned"] += len(out)
        return out

    def describe(self, limit: int = 5, status: str | None = None) -> str:
        """Compact one-line-per-hypothesis status text for the prompt."""
        alive = [h for h in self._hypotheses if not status or h.status == str(status)]
        if not alive:
            return "no hypotheses"
        limit = _clamp_limit(limit, self.max_limit)
        alive.sort(
            key=lambda h: (_STATUS_WEIGHT.get(h.status, 0.0), h.updated_t, h.seq), reverse=True
        )
        lines = []
        for hyp in alive[:limit]:
            line = (
                f"{hyp.id} [{hyp.status}] {hyp.kind}: {hyp.content} "
                f"support={len(hyp.support)} refute={len(hyp.refute)}"
            )
            if hyp.support:
                line += f" ev=[{','.join(e.ref for e in hyp.support[:3])}]"
            if hyp.reasons:
                line += f" refuted({hyp.reasons[-1]})"
            lines.append(line)
        return "\n".join(lines)

    def stats(self) -> dict:
        return {
            "hypotheses": len(self._hypotheses),
            "active": sum(1 for h in self._hypotheses if h.status == STATUS_ACTIVE),
            "confirmed": sum(1 for h in self._hypotheses if h.status == STATUS_CONFIRMED),
            "rejected": sum(1 for h in self._hypotheses if h.status == STATUS_REJECTED),
            "queries": self._stats["queries"],
            "returned": self._stats["returned"],
            "limit_budget": self.max_limit,
        }

    def _require(self, hyp_id: str) -> Hypothesis:
        hyp = self._by_id.get(str(hyp_id))
        if hyp is None:
            raise KeyError(f"unknown hypothesis id: {hyp_id}")
        return hyp

    def _entry(
        self, hyp: Hypothesis, score: float | None = None, added: int | None = None
    ) -> dict:
        entry = {
            "id": hyp.id,
            "kind": hyp.kind,
            "content": hyp.content,
            "status": hyp.status,
            "support": [e.to_dict() for e in hyp.support],
            "refute": [e.to_dict() for e in hyp.refute],
            "support_count": len(hyp.support),
            "refute_count": len(hyp.refute),
            "created_t": round(hyp.created_t, 3),
            "updated_t": round(hyp.updated_t, 3),
        }
        if self.episode_id:
            entry["episode_id"] = self.episode_id
        if hyp.data:
            entry["data"] = jsonable(hyp.data)
        if hyp.note:
            entry["note"] = hyp.note
        if hyp.reasons:
            entry["reasons"] = list(hyp.reasons)
        if score is not None:
            entry["score"] = round(float(score), 3)
        if added is not None:
            entry["added"] = int(added)
        refs = [EvidenceRef("memory", hyp.id, note=f"{hyp.kind} {hyp.status}")]
        refs.extend(hyp.support[:2])
        entry["evidence"] = refs
        return entry
