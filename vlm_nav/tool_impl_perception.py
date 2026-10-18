"""Perception tools for the agent runtime: ``observe`` / ``inspect_region`` /
``ground_target`` / ``compare_candidates`` / ``query_memory``.

These are the "取证" tools of the M1 tool loop (goal.md section 3): the model
decides where to look, magnifies a region of an earlier observation, grounds an
instruction into candidate boxes, and compares candidate evidence before it
commits to an approach. None of them move the drone (all four register with
``side_effects=False``): ``observe`` may rotate the view for the duration of one
render (exactly like ``world.scan_views``) but always restores the pose.

Evidence handling
-----------------
Every frame is an observation of the episode's repository
(``vlm_nav.observations``, ``ctx.extra["observation_store"]``) - the same
``obs_N`` namespace the runtime registers the frames it shows the model, so an id
the model was handed always resolves. The perception tools read observations
from the store first and only then from the older channels (this module's frame
cache under ``ctx.extra["perception.observations"]``, the frames another tool
published into ``ctx.extra["images"]``, and the memory archive), which is what
removes the "the model cites the frame it just saw and the tool says unknown"
failure. A capture is written to all of them: the frame cache and the archive
(so ``inspect_region`` can magnify it and ``query_memory`` can recall it) and the
store (so it carries pose, heading and intrinsics).

Coordinate contract
-------------------
Boxes and regions are ``[x1, y1, x2, y2]`` **corner** boxes in normalized 0-1000
image coordinates, top-left first: that is what the schemas, the grounding prompt
and the tool results all say, and what a box flowing out of ``ground_target``
into ``inspect_region`` means again. The legacy size form ``[u, v, w, h]`` (and
the same box as a dict with ``u/v/w/h`` or ``x/y/w/h`` keys) is still accepted
and normalized, because a model that learned the old spelling would otherwise
misplace every box; the format actually used is reported as ``box_format``
(``xyxy_norm`` / ``uvwh_norm`` / ``xyxy_px``). A 4-element box whose second
corner precedes the first (``x2 <= x1``) cannot be a corner box, so it is read as
the legacy size form - the one case the two shapes could not be told apart.

Pixel-to-ray math always uses the *observation's own* intrinsics
(``observation.intrinsics``, resolved from the world / camera object / caller
override when the frame was registered; the 86 deg default of
``world.pixel_to_body_offset`` is never assumed). With no depth and no model
distance the answer is a **ray**, not an invented point: the candidate keeps
``ray = {origin_flu, bearing_deg, elevation_deg, yaw_deg}`` (``position_source``
``"ray"``, ``range_status`` ``"unknown"``), and an observation whose field of
view nobody declared yields ``position_source`` ``"intrinsics_unknown"`` with the
range (when known) instead of a biased position.

Grounding into navigable ids
----------------------------
``ground_target`` writes every candidate into the canonical store
``ctx.extra["perception.candidates"]`` (``CANDIDATE_STORE_KEY``, ``cand_N`` ->
record) - the key the navigation tools resolve their ``target_id`` /
``candidate_id`` arguments from - with the canonical corner box, the spelling the
grounding model used (``box_format`` / ``box_format_used``) and, when its range
is known, the world position of the box centre (``position_flu`` with
``position_source`` ``depth`` / ``model_distance``). The memory API is owned by
another module and is reached through a duck-typed, best-effort bridge
(``_MemoryBridge``): the method is picked from a list of plausible names, its
signature is inspected and only accepted keywords are passed, and every failure
is reported as a short status string instead of an exception - a missing or
incompatible sink degrades to the local cache, never to a failed tool call.

The same bridge writes the three memory layers:

- ``observe`` records the capture as *episodic evidence* (frame, pose,
  intrinsics) together with the goal target of the task and the checked sectors
  of the capture (``coverage``), which the spatial layer keeps as
  ``checked_deg``;
- ``ground_target`` turns every candidate it finds into a *sighting*
  (``add_sighting``: the frame, the candidate id, the box the grounding model
  wrote and the attributes it read) - the piece of evidence a later
  ``query_memory`` cites - and, when it finds nothing, into a *scoped null
  result* (``add_negative``: "not found from this viewpoint, facing this
  heading, with this field of view", never "this area holds nothing");
- the working memory is fed by the same two writes (candidates, failed
  attempts, the question a sighting answers).

``memory_sink`` / ``memory_sighting`` / ``memory_negative`` / ``memory_layers``
report what each write did, and a context without a memory module simply
reports ``absent``. A store built without its episodic layer
(``episodic_memory=False``, the ``memory_spatial_only`` arm of
``scripts/run_vlm_nav_m2_memory.py``) answers ``disabled:episodic_memory_off``
to the two evidence writes: the claim is still validated and the spatial and
working layers are still written - candidate places, checked sectors, failed
looks - but no frame, sighting or null result is kept, and a later
``query_memory`` can only answer with poses and the task state.

Other tools reuse observations through :func:`resolve_observation` and
:func:`latest_observation` (a ``frame`` tensor plus the pose it was captured
from). The pose alone - without decoding any pixels - is available through
:func:`observation_pose` / :func:`resolve_observation_pose`, which is what lets a
recalled observation id be handed to ``return_to_observation`` /
``check_path`` / ``verify_goal`` as a target, and through
:func:`observation_poses` (id ->
position), the map an error answer reads to offer an executable next step.
``register_tools`` is an alias of :func:`register_perception_tools` for loaders
that probe generic register-function names.

``query_memory`` and the use of evidence
---------------------------------------
A retrieval hands back an *evidence pack* (image references, poses, the quality
of each hit, its source) and the uses of the evidence, best first: comparing the
candidates that were seen, verifying the goal against a sighting, a real move to
a located candidate, or localizing the object in the recalled frame. The flight
back to the recalled place is the *secondary* option, named with its reasons
(current position, evidence quality, expected benefit) - and withheld outright
when the recalled pose is the one the drone already hovers at, because that loop
moved nothing in the real runs. A hit says what it is: ``sighting`` (the frame
really showed the thing), ``negative`` (a null result with its visibility
boundary) or ``goal_frame`` (captured while the task looked for the target -
a place to look again, never proof that the target was seen). Against a store
whose episodic layer is off there is nothing to say that about: the hits are
capture records (pose, heading, step, no image, ``quality.frame="missing"``)
and the pack says so in ``evidence_note``, while the spatial places and the
task state it carries come from the two layers that are still on.

Depth reads are pose-bound
--------------------------
``world.depth_at_pixel`` reads the depth cache of the *current* frame, so a
depth reading is only returned when the observation was captured at the current
pose (position and yaw within tolerance, and not a pitched view). Otherwise the
result carries ``depth_status`` ``stale_observation`` / ``pose_unknown`` and no
``depth_m`` - silently mixing an old pixel with a fresh depth map is exactly the
class of error goal.md section 6 warns about.

Visual context of a capture
---------------------------
``observe`` returns more than a list of ids: ``view_index``
(``[{obs_id, yaw_deg, direction, note}]``, which way each returned view looked,
``direction`` being the 8-way label relative to the drone's own heading),
``coverage`` (which 45 deg sectors a level view has checked, which are
``unchecked_deg``, which views are ``worth_looking``, plus the note that names
the coarse resolution of a full sweep) and ``expansion`` (the ``inspect_region``
calls that pull one view back as pixels, ``[0,0,1000,1000]`` for the whole
frame). The coverage is republished under ``ctx.extra[VIEW_COVERAGE_KEY]`` so a
later answer can point at a direction the capture itself did not check.

The grounding ledger (anti-loop)
--------------------------------
``ground_target`` records every attempt per *evidence* - query intention,
capture pose/heading (``EVIDENCE_POS_TOL_M`` / ``EVIDENCE_YAW_TOL_DEG``) and the
frame's own region (a crop's ``[u, v, w, h]``), never per ``obs_N`` - in
``ctx.extra[GROUNDING_LEDGER_KEY]``. The second consecutive ``no_candidate`` on
the same evidence returns ``repeated_grounding: true`` plus a ``strategy_change``
(a concrete "inspect_region" / "change_view" / "change_viewpoint" call, with the
other two as alternatives); the third is refused as ``blocked`` /
``stagnant_grounding`` without running the model again. That is what removes the
failure the ids made invisible: grounding ``obs_10``, ``obs_11``, ``obs_12`` ...
at one pose and reading every fresh id as fresh evidence.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import inspect
import json
import math
import time
from typing import Any, Iterator, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .actions import body_delta_to_world
from .observations import (
    CANDIDATE_ID_TYPE,
    CANDIDATE_STORE_KEY,
    KIND_EVIDENCE,
    KIND_INSPECT,
    KIND_SCAN,
    OBSERVATION_STATE_KEY,
    IdTypeError,
    available_candidate_ids,
    available_observation_ids,
    classify_id,
    kind_for_label,
    lookup_pose as _store_pose,
    observation_store,
    parse_id_index,
    published_frames,
    register_observation,
    require_candidate_id,
    require_observation_id,
    resolve_intrinsics,
)
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
from .world import pixel_to_body_offset

try:  # the evidence vocabulary of the memory layer: shared, not duplicated
    from .memory import (
        EVIDENCE_KIND_GOAL_FRAME,
        EVIDENCE_KIND_NEGATIVE,
        EVIDENCE_KIND_OBSERVATION,
        EVIDENCE_KIND_SIGHTING,
        jsonable as _memory_jsonable,
        scope_note,
        view_scope,
    )
except Exception:  # noqa: BLE001 - the perception tools load without the memory module
    EVIDENCE_KIND_OBSERVATION = "observation"
    EVIDENCE_KIND_SIGHTING = "sighting"
    EVIDENCE_KIND_NEGATIVE = "negative"
    EVIDENCE_KIND_GOAL_FRAME = "goal_frame"
    scope_note = None
    view_scope = None

    def _memory_jsonable(value):  # pragma: no cover - fallback of the import guard
        return value

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

IMAGE_STORE_KEY = "perception.observations"
# ``OBSERVATION_STATE_KEY`` (``ctx.extra["observations"]``) is imported from
# ``vlm_nav.observations`` and re-exported here: the navigation tools keep the
# pose of every ``obs_N`` they capture there, and an observation id resolves
# there as well as in the observation store, the perception cache and the memory
# archive (``observation_pose``).
STORE_LIMIT = 64

# box/region formats: the canonical corner form, the legacy size form, and the
# unit each was written in (normalized 0-1000 vs absolute pixels)
BOX_FORMAT_XYXY = "xyxy"
BOX_FORMAT_UVWH = "uvwh"
BOX_UNIT_NORM = "norm"
BOX_UNIT_PX = "px"
#: what the schemas and the grounding prompt ask for
BOX_FORMAT_DOC = (
    "[x1, y1, x2, y2] corner box, top-left first, normalized 0-1000"
)
#: format of the ``box`` field of a candidate record (always canonical)
BOX_FORMAT_OUT = f"{BOX_FORMAT_XYXY}_{BOX_UNIT_NORM}"
#: values above this are read as absolute pixels and divided by the frame size
PIXEL_THRESHOLD = 1000.0

OBSERVE_MODES = ("lookaround", "forward", "pitch_down")
#: what ``observe`` does when the caller names no mode (the prompt's
#: ``observe({})`` example): one frame at the current heading
DEFAULT_OBSERVE_MODE = "forward"
DEFAULT_LOOKAROUND_VIEWS = 8
MAX_LOOKAROUND_VIEWS = 16
DEFAULT_PITCH_DOWN_DEG = 50.0

# --------------------------------------------------------------------------
# visual context: which way each view looked, what is still unchecked
# --------------------------------------------------------------------------

#: one heading sector of the coverage report (degrees): a level view "covers"
#: the sector its heading points into, so a report can say which directions a
#: capture really looked at (a bare list of observation ids cannot)
COVERAGE_BIN_DEG = 45.0
#: how many views a coverage report singles out as worth a second look
MAX_WORTH_LOOKING = 3
#: 8-way labels of a view's heading relative to the drone's own heading
#: (positive = counter-clockwise = to the left, the FLU convention)
DIRECTION_NAMES = (
    "ahead",
    "ahead-left",
    "left",
    "behind-left",
    "behind",
    "behind-right",
    "right",
    "ahead-right",
)
DIRECTION_UNKNOWN = "unknown"
#: ``ctx.extra`` key the coverage of the last capture is published under: the
#: grounding ledger reads it to suggest a concrete direction when a query keeps
#: coming back empty
VIEW_COVERAGE_KEY = "perception.view_coverage"

# --------------------------------------------------------------------------
# grounding attempt ledger (the anti-loop of goal.md section 5)
# --------------------------------------------------------------------------

#: ``ctx.extra`` key of the per-evidence grounding attempts.
GROUNDING_LEDGER_KEY = "perception.grounding_attempts"
#: bounded: how many evidence neighbourhoods / attempts per neighbourhood are kept
GROUNDING_LEDGER_LIMIT = 64
GROUNDING_NEIGHBOUR_LIMIT = 16
#: the 2nd consecutive ``no_candidate`` on the same evidence is flagged, the 3rd
#: is refused outright (without another model call)
STALL_FLAG_AT = 2
STALL_BLOCK_AT = 3
#: two attempts count as the *same evidence* within these tolerances
EVIDENCE_POS_TOL_M = 0.5
EVIDENCE_YAW_TOL_DEG = 20.0
#: granularity of the region dimension of the evidence key (normalized 0-1000)
EVIDENCE_REGION_TOL = 25.0

CROP_SHORT_SIDE_PX = 224
CROP_MAX_UPSCALE = 8.0
DEPTH_GRID = 3

POSE_POS_TOL_M = 1e-3
POSE_YAW_TOL_RAD = 1e-3

IOU_SAME = 0.5
IOU_DIFFERENT = 0.3
LOW_CONFIDENCE = 0.4

# range gates for a candidate's world position: below MIN the projected point is
# noise, above MAX the model's guess is not worth flying to
MIN_RANGE_M = 0.1
MAX_RANGE_M = 60.0

#: ``position_source`` values of a candidate location
POSITION_DEPTH = "depth"
POSITION_MODEL_DISTANCE = "model_distance"
POSITION_RAY = "ray"
POSITION_INTRINSICS_UNKNOWN = "intrinsics_unknown"
POSITION_CROP_SOURCE_UNKNOWN = "crop_source_unknown"
POSITION_NO_RANGE = "no_valid_depth"
POSITION_PITCHED = "pitched_view"
POSITION_POSE_UNKNOWN = "pose_unknown"
POSITION_NO_WORLD = "no_world"

GROUND_SYSTEM_PROMPT = (
    "You are the target-grounding module of a drone navigation agent. "
    "Reply with ONE JSON object and nothing else: "
    '{"candidates": [{"box": [x1, y1, x2, y2], "attrs": {"color": "..."}, '
    '"confidence": 0.0, "distance_m": 0.0, '
    '"uncertainty": "<what is still unclear>"}]}. '
    "box is a corner box [x1, y1, x2, y2] in normalized 0-1000 image "
    "coordinates: x1, y1 = top-left corner, x2, y2 = bottom-right corner "
    "(the older size form [u, v, w, h] is also understood, but only the corner "
    "form is exact). Keep at most 3 candidates, most likely first; "
    '"distance_m" is your rough estimate of how far the box is from the camera '
    "in meters (0 when you cannot tell); use "
    '{"candidates": []} when the instruction target is not visible.'
)


# --------------------------------------------------------------------------
# stored observation + local cache
# --------------------------------------------------------------------------


@dataclass
class PerceptionObservation:
    """One stored frame plus the pose it was captured from.

    ``metadata["intrinsics"]`` holds the intrinsics its pixels are projected
    with, and ``metadata["crop"]`` (crops only) the mapping back into the source
    frame; a record loaded from the observation store or the memory archive
    carries both when the producer recorded them.
    """

    observation_id: str
    frame: torch.Tensor
    position_flu: tuple[float, float, float] | None = None
    yaw_rad: float | None = None
    time_s: float = 0.0
    kind: str = "view"  # view | crop | memory
    parent_id: str | None = None
    region: list[float] | None = None
    note: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def yaw_deg(self) -> float | None:
        return None if self.yaw_rad is None else math.degrees(self.yaw_rad)

    @property
    def pitched(self) -> bool:
        """Pitched views do not share the depth frame of the level view."""
        return float(self.metadata.get("pitch_down_deg") or 0.0) > 0.0

    @property
    def intrinsics(self) -> dict:
        """Intrinsics recorded with this observation ({} when none were)."""
        declared = self.metadata.get("intrinsics")
        return dict(declared) if isinstance(declared, dict) else {}

    @property
    def crop(self) -> dict:
        """Crop mapping back into the source frame ({} for a full frame)."""
        declared = self.metadata.get("crop")
        return dict(declared) if isinstance(declared, dict) else {}


def _extra(ctx: ToolContext) -> dict:
    """``ctx.extra`` repaired into a dict (the runtime may leave it None)."""
    if not isinstance(ctx.extra, dict):
        ctx.extra = {}
    return ctx.extra


def _store(ctx: ToolContext) -> dict[str, PerceptionObservation]:
    extra = _extra(ctx)
    store = extra.get(IMAGE_STORE_KEY)
    if not isinstance(store, dict):
        store = {}
        extra[IMAGE_STORE_KEY] = store
    return store


def _candidate_store(ctx: ToolContext) -> dict[str, dict]:
    extra = _extra(ctx)
    store = extra.get(CANDIDATE_STORE_KEY)
    if not isinstance(store, dict):
        store = {}
        extra[CANDIDATE_STORE_KEY] = store
    return store


def _frame_from_path(path: Any) -> torch.Tensor | None:
    """Decode an on-disk frame (a store record may keep only its path)."""
    text = str(path or "")
    if not text:
        return None
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001 - PIL is optional for pure-cache runs
        return None
    try:
        with Image.open(text) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    except Exception:  # noqa: BLE001 - an unreadable frame is simply unavailable
        return None
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)


def _perception_record_of_store(ctx: ToolContext, observation: Any) -> PerceptionObservation | None:
    """Build a magnifiable record from a stored observation, or None.

    The pixels come from the record itself, then from the frames the tools
    published (``ctx.extra["images"]``), then from a materialized path and the
    memory archive: metadata-only records (a pose someone recalled) resolve for
    navigation but cannot be cropped.
    """
    obs_id = str(getattr(observation, "obs_id", "") or "")
    if not obs_id:
        return None
    frame = getattr(observation, "frame", None)
    if frame is None:
        frame = published_frames(ctx).get(obs_id)
    if frame is None:
        frame = _frame_from_path(getattr(observation, "image_path", ""))
    if frame is None and ctx.memory is not None:
        frame, _pose = _MemoryBridge(ctx.memory).fetch_frame(obs_id)
    if frame is None:
        return None
    position = getattr(observation, "pose_flu", None)
    if not isinstance(position, (list, tuple)) or not all(
        math.isfinite(float(v)) for v in list(position)[:3]
    ):
        position = None
    yaw_deg = _as_float(getattr(observation, "yaw_deg", None), None)
    detail = getattr(observation, "detail", None)
    detail = dict(detail) if isinstance(detail, dict) else {}
    metadata = {
        "intrinsics": dict(getattr(observation, "intrinsics", {}) or {}),
        "store_kind": str(getattr(observation, "kind", "") or ""),
        "step": getattr(observation, "step", None),
    }
    metadata.update(detail)
    if getattr(observation, "parent_id", None):
        metadata.setdefault("source", str(observation.parent_id))
    return PerceptionObservation(
        observation_id=obs_id,
        frame=_as_frame(frame),
        position_flu=None if position is None else tuple(float(v) for v in list(position)[:3]),
        yaw_rad=None if yaw_deg is None else math.radians(yaw_deg),
        time_s=float(getattr(observation, "timestamp", 0.0) or 0.0),
        kind="crop" if str(getattr(observation, "kind", "")) == "inspect" else "view",
        parent_id=getattr(observation, "parent_id", None),
        region=detail.get("region"),
        note=str(getattr(observation, "note", "") or ""),
        metadata=metadata,
    )


def resolve_observation(ctx: ToolContext, observation_id: str) -> PerceptionObservation | None:
    """Frame + pose + intrinsics for an observation id, or None.

    Lookup order: this module's frame cache (the frames ``observe`` /
    ``inspect_region`` just captured), the episode's observation store (every
    frame the runtime showed the model and every frame a tool published), the
    frames published under ``ctx.extra["images"]``, then the memory archive. An
    id the model was handed therefore resolves wherever it came from, which is
    the point of the unified repository.
    """
    if not observation_id:
        return None
    obs_id = str(observation_id)
    record = _store(ctx).get(obs_id)
    if record is not None:
        return record
    store = observation_store(ctx, create=False)
    if store is not None:
        observation = store.get(obs_id)
        if observation is not None:
            from_store = _perception_record_of_store(ctx, observation)
            if from_store is not None:
                return from_store
    published = published_frames(ctx).get(obs_id)
    if published is not None:
        position, yaw_rad = _store_pose(ctx, obs_id)
        return PerceptionObservation(
            observation_id=obs_id,
            frame=_as_frame(published),
            position_flu=position,
            yaw_rad=yaw_rad,
            time_s=-1.0,
            kind="view",
            note="published by a tool",
            metadata={"intrinsics": resolve_intrinsics(ctx, frame=published)},
        )
    frame, pose = _MemoryBridge(ctx.memory).fetch_frame(obs_id)
    if frame is None:
        return None
    position, yaw_rad = pose
    return PerceptionObservation(
        observation_id=obs_id,
        frame=frame,
        position_flu=position,
        yaw_rad=yaw_rad,
        time_s=-1.0,
        kind="memory",
        note="loaded from memory",
        metadata={"intrinsics": resolve_intrinsics(ctx, frame=frame)},
    )


def candidate_store(ctx: ToolContext) -> dict[str, dict]:
    """Canonical ``cand_N`` -> candidate record store (``CANDIDATE_STORE_KEY``).

    The ids a model receives from ``ground_target`` resolve here; the
    navigation tools read the same store for their ``target_id`` /
    ``candidate_id`` arguments, so this is the one place a candidate id is
    looked up.
    """
    return _candidate_store(ctx)


def latest_observation(ctx: ToolContext) -> PerceptionObservation | None:
    """Most recently captured observation of this episode, if any."""
    store = _store(ctx)
    if not store:
        return None
    return next(reversed(store.values()))


def _register_in_store(
    ctx: ToolContext,
    record: PerceptionObservation,
    *,
    kind: str,
    step: int | None = None,
    detail: dict | None = None,
) -> None:
    """Put a captured frame into the episode's observation store.

    The store is the repository every tool (and the next wave of navigation
    tools) reads, so a capture that only lived in the local cache or in memory
    would be exactly the id the model can cite but no tool can resolve. Best
    effort: a context without a store simply keeps the legacy behaviour.
    """
    store = observation_store(ctx, create=True)
    if store is None:
        return
    try:
        store.register(
            "",
            record.position_flu,
            record.yaw_deg,
            record.intrinsics or resolve_intrinsics(ctx, frame=record.frame),
            kind,
            int(record.metadata.get("step") or step or 0),
            record.time_s,
            obs_id=record.observation_id,
            frame=record.frame,
            parent_id=record.parent_id,
            note=record.note,
            detail=detail or {},
        )
    except Exception:  # noqa: BLE001 - bookkeeping never fails a capture
        return


def _remember(
    ctx: ToolContext, record: PerceptionObservation, *, coverage: dict | None = None
) -> str:
    """Cache the record locally (bounded), register it in the episode's
    observation store and push it to memory; returns a short sink status
    string for the tool result. ``coverage`` (a capture's checked sectors)
    travels with the write, so a sink that refuses the write refuses the
    coverage as well."""
    store = _store(ctx)
    store[record.observation_id] = record
    while len(store) > STORE_LIMIT:
        store.pop(next(iter(store)))
    return _MemoryBridge(ctx.memory).record(ctx, record, coverage=coverage)


# --------------------------------------------------------------------------
# memory bridge (memory.ObservationLog is owned by another module)
# --------------------------------------------------------------------------


def _memory_metadata(record: PerceptionObservation) -> dict:
    """Searchable metadata for the archive: the capture metadata plus the
    provenance a later memory query needs (kind / parent / region)."""
    metadata = dict(record.metadata)
    metadata.setdefault("kind", record.kind)
    if record.parent_id:
        metadata.setdefault("parent_id", record.parent_id)
    if record.region:
        metadata.setdefault("region", list(record.region))
    return metadata


def _memory_payload(
    ctx: ToolContext,
    record: PerceptionObservation,
    *,
    coverage: dict | None = None,
) -> dict:
    position = None if record.position_flu is None else [float(v) for v in record.position_flu]
    # Searchable provenance for later query_memory lookups. The instructed
    # target name is the *goal target* of the task - what the drone is looking
    # for - and never a sighting: a frame that merely inherited the task's
    # target has shown nothing, and the memory layer keeps the two fields apart
    # (``sighting`` is written by ``ground_target``, which has a candidate, a
    # box and a basis to show for it).
    goal_target = record.metadata.get("target_name")
    if not goal_target:
        tgt = getattr(getattr(ctx, "world", None), "target", None)
        goal_target = getattr(tgt, "name", None) if tgt is not None else None
    labels = record.metadata.get("labels")
    return {
        "observation_id": record.observation_id,
        "episode_id": ctx.episode_id,
        "image": record.frame,
        "pose": position,
        "position_flu": position,
        "yaw_rad": record.yaw_rad,
        "yaw_deg": record.yaw_deg,
        "time_s": record.time_s,
        "kind": record.kind,
        "parent_id": record.parent_id,
        "region": list(record.region) if record.region else None,
        "note": record.note,
        "goal_target": goal_target,
        "sighting": None,
        "labels": labels,
        "intrinsics": record.intrinsics or None,
        "step": record.metadata.get("step"),
        # the checked sectors of a capture belong to the spatial layer, and they
        # ride on the same write, so a store that drops the write drops the
        # coverage with it (the evidence ablation of scripts/run_vlm_nav_m2_memory)
        "coverage": coverage,
        "meta": _memory_metadata(record),
    }


def _accepted_kwargs(fn, payload: dict, aliases: dict | None = None) -> dict:
    """Filter ``payload`` down to the keywords ``fn`` actually accepts.

    ``aliases`` are alternative names for the same fields (``obs_id`` for
    ``observation_id`` and so on); they are only offered when the sink declares
    the names explicitly, so a permissive ``**kwargs`` sink sees canonical keys
    once instead of every spelling of the same field.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return dict(payload)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(payload)
    merged = {**payload, **(aliases or {})}
    keywords = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    accepted = {name: merged[name] for name, p in params.items() if p.kind in keywords and name in merged}
    missing = [
        name
        for name, p in params.items()
        if p.kind in keywords and p.default is inspect.Parameter.empty and name not in accepted
    ]
    if missing:
        raise TypeError(f"requires unsupported argument(s) {missing}")
    return accepted


class _MemoryBridge:
    """Best-effort adapter to ``memory.ObservationLog``.

    Never raises: an absent, unsupported or failing sink is reported through
    ``status`` so tool results can carry it as ``memory_sink``.
    """

    _RECORD_METHODS = (
        "add_observation",
        "record_observation",
        "log_observation",
        "store_observation",
        "record",
        "add",
        "log",
    )
    _GET_METHODS = ("get_observation", "observation", "find", "get", "lookup")
    _SIGHTING_METHODS = ("add_sighting", "record_sighting", "note_sighting")
    _NEGATIVE_METHODS = ("add_negative", "record_negative", "note_negative")

    def __init__(self, memory: Any):
        self.memory = memory
        self.status = "absent" if memory is None else "unsupported"

    def _method(self, names):
        for name in names:
            fn = getattr(self.memory, name, None)
            if callable(fn):
                return fn
        return None

    def record(
        self, ctx: ToolContext, record: PerceptionObservation, *, coverage: dict | None = None
    ) -> str:
        if self.memory is None:
            return self.status
        fn = self._method(self._RECORD_METHODS)
        if fn is None:
            return self.status
        payload = _memory_payload(ctx, record, coverage=coverage)
        aliases = {
            "obs_id": record.observation_id,
            "id": record.observation_id,
            "frame": record.frame,
            "image_chw": record.frame,
            "rgb": record.frame,
            # "target_name" is the legacy spelling of the goal target (never a
            # sighting); it is only offered to a sink that declares it
            "target_name": payload.get("goal_target"),
            "extra": _memory_metadata(record),
            # "timestamp" is the archive-side spelling of when the frame was
            # captured (wall clock); "time_s" stays episode-relative
            "timestamp": time.time(),
        }
        attempts = []
        try:
            attempts.append(_accepted_kwargs(fn, payload, aliases))
        except TypeError:
            attempts.append({"record": payload})
            attempts.append(None)  # positional payload
        for attempt in attempts:
            try:
                if attempt is None:
                    fn(payload)
                else:
                    fn(**attempt)
            except Exception as exc:  # noqa: BLE001 - memory is optional
                self.status = f"error:{fn.__name__}({type(exc).__name__})"
                continue
            self.status = f"ok:{fn.__name__}"
            return self.status
        return self.status

    @staticmethod
    def _empty_write(status: str) -> dict:
        return {"status": status, "added": 0, "rejected": [], "layers": {}}

    def add_sighting(self, observation_id: str, sightings, *, step: int | None = None) -> dict:
        """Attach the sightings of a grounding result to the frame they were taken in.

        ``sightings`` is one mapping or a list of them (target, candidate, box,
        basis, confidence, position); the sink validates each of them, so a
        sighting without a candidate or a basis comes back in ``rejected``
        instead of being written. Never raises. A sink whose episodic layer is
        off answers ``disabled:episodic_memory_off`` with ``added=0`` and the
        layers it did write (the candidate place, the task fact).
        """
        if self.memory is None:
            return self._empty_write(self.status)
        fn = self._method(self._SIGHTING_METHODS)
        if fn is None:
            return self._empty_write(self.status)
        payload = {
            "obs_id": observation_id,
            "observation_id": observation_id,
            "sighting": sightings,
            "sightings": sightings,
            "step": step,
        }
        try:
            keywords = _accepted_kwargs(fn, payload)
        except TypeError:
            keywords = {}
        try:
            result = fn(**keywords) if keywords else fn(observation_id, sightings)
        except Exception as exc:  # noqa: BLE001 - memory is optional
            return self._empty_write(f"error:{fn.__name__}({type(exc).__name__})")
        return self._write_result(result, fn.__name__)

    def add_negative(
        self,
        observation_id: str,
        *,
        target: str = "",
        reason: str = "not_found",
        visibility: dict | None = None,
        note: str = "",
        step: int | None = None,
    ) -> dict:
        """Record a null result *as scoped by this view* (never as cleared ground).

        A sink whose episodic layer is off answers ``disabled:episodic_memory_off``
        with ``added=0``: the sector it checked and the failed look are still
        written to the layers that own them, the null result itself is not.
        """
        if self.memory is None:
            return self._empty_write(self.status)
        fn = self._method(self._NEGATIVE_METHODS)
        if fn is None:
            return self._empty_write(self.status)
        payload = {
            "obs_id": observation_id,
            "observation_id": observation_id,
            "target": target,
            "reason": reason,
            "visibility": visibility,
            "note": note,
            "step": step,
        }
        try:
            keywords = _accepted_kwargs(fn, payload)
        except TypeError:
            keywords = {}
        try:
            result = fn(**keywords) if keywords else fn(observation_id)
        except Exception as exc:  # noqa: BLE001 - memory is optional
            return self._empty_write(f"error:{fn.__name__}({type(exc).__name__})")
        return self._write_result(result, fn.__name__)

    @staticmethod
    def _write_result(result, name: str) -> dict:
        """Normalise a layer write's return value into the tool-facing shape."""
        if not isinstance(result, Mapping):
            return {"status": f"ok:{name}" if result is not None else "rejected",
                    "added": 0, "rejected": [], "layers": {}}
        out = {
            "status": str(result.get("status") or f"ok:{name}"),
            "added": int(result.get("added") or 0),
            "rejected": list(result.get("rejected") or []),
            "layers": dict(result.get("layers") or {}),
        }
        for key in ("obs_id", "visibility", "scope_note"):
            if result.get(key) not in (None, ""):
                out[key] = result[key]
        return out

    def layer_status(self) -> dict:
        """What the memory store says about its three layers ({} when it has none)."""
        fn = getattr(self.memory, "layer_status", None)
        if not callable(fn):
            return {}
        try:
            return dict(fn())
        except Exception:  # noqa: BLE001 - memory is optional
            return {}

    def fetch_frame(self, observation_id: str) -> tuple[torch.Tensor | None, tuple]:
        if self.memory is None:
            return None, (None, None)
        obj = self._lookup(observation_id)
        if obj is None:
            return None, (None, None)
        frame = _frame_of(obj)
        if frame is None:
            # memory entries may keep pixels as a materialized PNG path (or
            # have dropped the pixels entirely); fall back to the archive's
            # frame() accessor when it exposes one
            frame_fn = getattr(self.memory, "frame", None)
            if callable(frame_fn):
                try:
                    frame = frame_fn(observation_id)
                except Exception:  # noqa: BLE001 - memory is optional
                    frame = None
        return frame, _pose_of(obj)

    def fetch_pose(self, observation_id: str) -> tuple:
        """Pose of a memory entry without touching its pixels.

        Used to turn a recalled id into a navigable target, so a record whose
        frame has been evicted still resolves as a place to fly back to.
        """
        if self.memory is None:
            return None, None
        obj = self._lookup(observation_id)
        if obj is None:
            return None, None
        return _pose_of(obj)

    def _lookup(self, observation_id: str):
        obj = None
        fn = self._method(self._GET_METHODS)
        if fn is not None:
            try:
                obj = fn(observation_id)
            except Exception:  # noqa: BLE001 - memory is optional
                obj = None
        if obj is None:
            obj = self._from_container(observation_id)
        return obj

    def _from_container(self, observation_id: str):
        container = getattr(self.memory, "observations", None)
        if container is None:
            container = getattr(self.memory, "records", None)
        if isinstance(container, dict):
            return container.get(observation_id)
        if isinstance(container, (list, tuple)):
            for item in container:
                if _obj_field(item, ("observation_id", "obs_id", "id")) == observation_id:
                    return item
        return None


def _obj_field(obj: Any, names: tuple[str, ...]):
    for name in names:
        if isinstance(obj, dict):
            if name in obj:
                return obj[name]
            continue
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _frame_of(obj: Any) -> torch.Tensor | None:
    if torch.is_tensor(obj):
        return obj.detach().float()
    value = _obj_field(obj, ("image", "frame", "image_chw", "rgb", "tensor"))
    if value is None:
        return None
    return _as_frame(value)


def _pose_of(obj: Any) -> tuple[tuple[float, float, float] | None, float | None]:
    position = None
    raw_pos = _obj_field(obj, ("position_flu", "pose", "position"))
    if raw_pos is not None:
        try:
            values = np.asarray(raw_pos, dtype=np.float64).reshape(-1)[:3]
            if values.size == 3 and np.all(np.isfinite(values)):
                position = tuple(float(v) for v in values)
        except (TypeError, ValueError):
            position = None
    yaw = _as_float(_obj_field(obj, ("yaw_rad", "yaw")), None)
    if yaw is None:
        yaw_deg = _as_float(_obj_field(obj, ("yaw_deg",)), None)
        yaw = None if yaw_deg is None else math.radians(yaw_deg)
    return position, yaw


# --------------------------------------------------------------------------
# recalled observation -> pose (the way back to a remembered place)
# --------------------------------------------------------------------------


def _state_poses(ctx: ToolContext) -> dict:
    """``ctx.extra["observations"]``: the ``obs_N`` pose records the navigation
    tools write for every captured view (``_capture_observation``)."""
    state = _extra(ctx).get(OBSERVATION_STATE_KEY)
    return state if isinstance(state, dict) else {}


def observation_pose(
    ctx: ToolContext, observation_id: Any
) -> tuple[tuple[float, float, float] | None, float | None]:
    """(position, yaw_rad) of an observation id, from every store that keeps one.

    Lookup order: the episode's observation store (every frame the model was
    shown, with the pose it was captured at), the perception cache (frames
    captured by ``observe``), the navigation pose records
    (``ctx.extra["observations"]``), then the memory archive - read through its
    getters, so no pixels are decoded.
    """
    if observation_id in (None, ""):
        return None, None
    obs_id = str(observation_id)
    store = observation_store(ctx, create=False)
    if store is not None:
        stored = store.get(obs_id)
        if stored is not None and stored.has_pose:
            position = tuple(float(v) for v in stored.pose_flu)
            yaw_rad = math.radians(stored.yaw_deg) if stored.has_yaw else None
            return position, yaw_rad
    record = _store(ctx).get(obs_id)
    if record is not None:
        return record.position_flu, record.yaw_rad
    entry = _state_poses(ctx).get(obs_id)
    if entry is not None:
        return _pose_of(entry)
    return _MemoryBridge(ctx.memory).fetch_pose(obs_id)


def resolve_observation_pose(
    ctx: ToolContext, observation_id: Any
) -> tuple[tuple[float, float, float], float | None] | None:
    """Pose of a recalled observation, or None when no store knows it.

    This is what makes a ``query_memory`` hit an executable target: an id that
    resolves here can be handed straight to ``return_to_observation`` /
    ``check_path`` / ``verify_goal`` as
    ``observation_id``.
    """
    position, yaw = observation_pose(ctx, observation_id)
    return None if position is None else (position, yaw)


def observation_poses(ctx: ToolContext, limit: int = 8) -> dict[str, list[float]]:
    """``obs_N`` -> ``[x, y, z]`` of every observation the drone can fly back to.

    Read newest-first - the observation store first (every frame the model was
    shown, i.e. exactly the ids it was handed), then the perception cache, the
    navigation pose records, then the memory archive - and capped at ``limit``
    ids, so an unresolved-target answer can hand the model an id *and* the
    position it stands for instead of only a list.
    """
    found: dict[str, list[float]] = {}

    def add(obs_id: Any, position: Any) -> bool:
        """Record one id; True once ``limit`` ids are known (scan may stop)."""
        key = "" if obs_id is None else str(obs_id)
        raw = None if position is None else np.asarray(position, dtype=np.float64).reshape(-1)[:3]
        if not key or key in found or raw is None or raw.size != 3:
            return len(found) >= limit
        if not np.all(np.isfinite(raw)):
            return len(found) >= limit
        found[key] = [round(float(v), 3) for v in raw]
        return len(found) >= limit

    # newest first inside each source; the frames the model was actually shown
    # come before the incidental leg poses
    store = observation_store(ctx, create=False)
    if store is not None:
        for obs_id, pose in store.poses().items():
            if add(obs_id, pose.get("position_flu")):
                return found
    sources = (list(_store(ctx).items()), list(_state_poses(ctx).items()))
    for holder in sources:
        for key, entry in reversed(holder):
            obs_id = _obj_field(entry, ("observation_id", "obs_id", "id")) or key
            if add(obs_id, _pose_of(entry)[0]):
                return found
    ids_fn = getattr(ctx.memory, "ids", None)
    if callable(ids_fn):
        bridge = _MemoryBridge(ctx.memory)
        try:
            ids = [str(item) for item in ids_fn()][-limit:]
        except Exception:  # noqa: BLE001 - memory is optional
            ids = []
        for obs_id in reversed(ids):
            add(obs_id, bridge.fetch_pose(obs_id)[0])
            if len(found) >= limit:
                break
    return found


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _as_int(value: Any, default: int) -> int:
    out = _as_float(value, None)
    return default if out is None else int(round(out))


def _as_frame(obj: Any) -> torch.Tensor | None:
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.detach().float()
    try:
        arr = np.asarray(obj)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 3 or arr.dtype == object:
        return None
    arr = arr.astype(np.float32)
    if arr.shape[0] not in (1, 3) and arr.shape[-1] in (1, 3):
        arr = np.moveaxis(arr, -1, 0)
    return torch.from_numpy(np.ascontiguousarray(arr))


def _norm_heading(deg: float | None) -> float | None:
    return None if deg is None else (deg + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class Box:
    """One image box in normalized 0-1000 corner coordinates.

    ``format_used`` records how the caller wrote it (``xyxy_norm`` /
    ``uvwh_norm`` / ``xyxy_px`` / ``uvwh_px``), so a result can say which
    convention it read instead of leaving the ambiguity silent.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    format_used: str = f"{BOX_FORMAT_XYXY}_{BOX_UNIT_NORM}"

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def to_list(self) -> list[float]:
        """Canonical ``[x1, y1, x2, y2]``."""
        return [
            round(self.x1, 1),
            round(self.y1, 1),
            round(self.x2, 1),
            round(self.y2, 1),
        ]

    def to_uvwh(self) -> list[float]:
        """Legacy internal size form ``[u, v, w, h]`` (cropping, depth grids)."""
        return [
            round(self.x1, 1),
            round(self.y1, 1),
            round(self.width, 1),
            round(self.height, 1),
        ]


def _box_components(value: Any) -> tuple[list[float], str] | None:
    """(four numbers, declared format) of a raw box/region value."""
    if isinstance(value, dict):
        declared = str(value.get("box_format", value.get("format", "")) or "").strip().lower()
        if all(key in value for key in ("x1", "y1", "x2", "y2")):
            numbers = [value[key] for key in ("x1", "y1", "x2", "y2")]
            return _four(numbers), BOX_FORMAT_XYXY
        if all(key in value for key in ("x_min", "y_min", "x_max", "y_max")):
            numbers = [value[key] for key in ("x_min", "y_min", "x_max", "y_max")]
            return _four(numbers), BOX_FORMAT_XYXY
        if all(key in value for key in ("u", "v", "w", "h")):
            numbers = [value[key] for key in ("u", "v", "w", "h")]
            return _four(numbers), BOX_FORMAT_UVWH
        if all(key in value for key in ("x", "y", "w", "h")):
            numbers = [value["x"], value["y"], value["w"], value["h"]]
            return _four(numbers), BOX_FORMAT_UVWH
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    numbers = _four(value)
    if numbers is None:
        return None
    x1, y1, x2, y2 = numbers
    if x2 <= x1 or y2 <= y1:
        # a corner box with a reversed (or empty) second corner is not a corner
        # box; the legacy [u, v, w, h] reading is the only one that fits
        return numbers, BOX_FORMAT_UVWH
    return numbers, BOX_FORMAT_XYXY


def _four(values) -> list[float] | None:
    numbers = [_as_float(v, None) for v in values]
    if any(number is None for number in numbers):
        return None
    return [float(number) for number in numbers]


def parse_box(
    value: Any,
    *,
    width: int | None = None,
    height: int | None = None,
    box_format: str | None = None,
) -> tuple[Box | None, str]:
    """Parse a box/region into normalized corner coordinates; (None, reason).

    Canonical form is ``[x1, y1, x2, y2]`` normalized 0-1000 (``box_format``
    ``"xyxy"``); the legacy size form ``[u, v, w, h]`` and its dict spellings are
    accepted and converted. A component above 1000 is read as absolute pixels
    and divided by the observation's width/height (``_px`` in ``format_used``);
    with no frame size to divide by the box is refused (``unknown_frame_size``)
    rather than silently rescaled. The result is clamped to the frame, so a box
    partly outside is cropped while one entirely outside is
    ``region_outside_frame``.
    """
    declared = str(box_format or "").strip().lower()
    components = _box_components(value)
    if components is None:
        return None, "invalid_region"
    numbers, detected = components
    if declared in (BOX_FORMAT_XYXY, BOX_FORMAT_UVWH):
        detected = declared
    if not all(math.isfinite(number) for number in numbers):
        return None, "invalid_region"
    unit = BOX_UNIT_NORM
    if max(abs(number) for number in numbers) > PIXEL_THRESHOLD:
        if not width or not height or width <= 0 or height <= 0:
            return None, "unknown_frame_size"
        unit = BOX_UNIT_PX
        numbers = [
            numbers[0] / float(width) * 1000.0,
            numbers[1] / float(height) * 1000.0,
            numbers[2] / float(width) * 1000.0,
            numbers[3] / float(height) * 1000.0,
        ]
    if detected == BOX_FORMAT_UVWH:
        u, v, w, h = numbers
        x1, y1, x2, y2 = u, v, u + w, v + h
    else:
        x1, y1, x2, y2 = numbers
    if x2 <= x1 or y2 <= y1:
        return None, "invalid_region"
    x1, y1 = max(x1, 0.0), max(y1, 0.0)
    x2, y2 = min(x2, 1000.0), min(y2, 1000.0)
    if x2 <= x1 or y2 <= y1:
        return None, "region_outside_frame"
    return Box(x1, y1, x2, y2, f"{detected}_{unit}"), ""


def _crop_frame(frame: torch.Tensor, region: list[float], target_short: int = CROP_SHORT_SIDE_PX):
    """Crop a normalized [u, v, w, h] region out of a CHW frame and upscale it
    for the VLM; returns (crop, crop size in source pixels, upscale, origin).

    ``origin`` is the top-left pixel of the crop inside the source frame, which
    is what maps a crop pixel back onto the camera's own pixel grid (a crop's
    pixel grid carries no intrinsics of its own).
    """
    height, width = int(frame.shape[-2]), int(frame.shape[-1])
    x0 = int(round(region[0] / 1000.0 * width))
    y0 = int(round(region[1] / 1000.0 * height))
    x1 = int(round((region[0] + region[2]) / 1000.0 * width))
    y1 = int(round((region[1] + region[3]) / 1000.0 * height))
    x0, y0 = max(0, min(x0, width - 1)), max(0, min(y0, height - 1))
    x1, y1 = max(x0 + 1, min(x1, width)), max(y0 + 1, min(y1, height))
    crop = frame[:, y0:y1, x0:x1]
    short = max(1, min(int(crop.shape[-2]), int(crop.shape[-1])))
    upscale = max(1.0, min(CROP_MAX_UPSCALE, target_short / short))
    if upscale > 1.001:
        crop = F.interpolate(
            crop.unsqueeze(0), scale_factor=upscale, mode="bilinear", align_corners=False
        ).squeeze(0)
    if crop.is_floating_point():
        crop = crop.clamp(0.0, 1.0)
    return crop, [x1 - x0, y1 - y0], upscale, [x0, y0]


def _region_depth(depth_fn, region: list[float]) -> tuple[float | None, int]:
    """Median of a small pixel grid inside the region (axial depth, meters)."""
    u0, v0 = region[0] / 1000.0, region[1] / 1000.0
    u1, v1 = (region[0] + region[2]) / 1000.0, (region[1] + region[3]) / 1000.0
    us = np.linspace(u0, u1, DEPTH_GRID + 2)[1:-1] * 1000.0
    vs = np.linspace(v0, v1, DEPTH_GRID + 2)[1:-1] * 1000.0
    values = []
    for u in us:
        for v in vs:
            try:
                depth = depth_fn((float(u), float(v)))
            except Exception:  # noqa: BLE001 - sensors may not cover every pixel
                depth = None
            depth = _as_float(depth, None)
            if depth is not None and depth > 0.05:
                values.append(depth)
    if not values:
        return None, 0
    return float(np.median(values)), len(values)


# --------------------------------------------------------------------------
# world pose helpers
# --------------------------------------------------------------------------


def _current_position(world) -> tuple[float, float, float] | None:
    raw = getattr(world, "position_flu", None)
    if raw is None:
        return None
    try:
        values = np.asarray(raw, dtype=np.float64).reshape(-1)[:3]
    except (TypeError, ValueError):
        return None
    if values.size != 3 or not np.all(np.isfinite(values)):
        return None
    return tuple(float(v) for v in values)


def _current_yaw(world) -> float | None:
    return _as_float(getattr(world, "yaw_rad", None), None)


@contextmanager
def _pose_peek(world, dyaw_rad: float) -> Iterator[bool]:
    """Rotate ``world`` by ``dyaw_rad`` for one render, always restoring yaw.

    Uses the same mechanism as ``world.scan_views`` (write ``state.yaw_rad``,
    push the pose into the sensors via ``_write_pose`` when the world has one).
    Yields whether the override was actually applied.
    """
    state = getattr(world, "state", None)
    yaw0 = _current_yaw(world)
    if state is None or yaw0 is None or not hasattr(state, "yaw_rad") or abs(dyaw_rad) < 1e-12:
        yield False
        return
    write_pose = getattr(world, "_write_pose", None)
    try:
        state.yaw_rad = yaw0 + dyaw_rad
        if callable(write_pose):
            write_pose()
        yield True
    finally:
        state.yaw_rad = yaw0
        if callable(write_pose):
            try:
                write_pose()
            except Exception:  # noqa: BLE001 - restore is best effort
                pass


def _scan_headings(base_deg: float | None, n_views: int, spread_deg: float | None) -> list:
    if base_deg is None:
        return [None] * n_views
    if spread_deg is None:
        return [_norm_heading(base_deg + 360.0 * k / n_views) for k in range(n_views)]
    span = float(spread_deg)
    return [
        _norm_heading(base_deg - span / 2.0 + span * k / max(1, n_views - 1))
        for k in range(n_views)
    ]


def _view(frame, yaw_deg: float | None, pitch_down_deg: float = 0.0, note: str = "") -> dict:
    return {
        "frame": frame,
        "yaw_deg": yaw_deg,
        "pitch_down_deg": float(pitch_down_deg),
        "note": note,
    }


def _call_scan(scan, n_views: int, pitch_down_deg: float, spread_deg: float | None):
    """Call ``world.scan_views`` tolerating worlds with a narrower signature."""
    attempts = [
        {"n_views": n_views, "pitch_down_deg": pitch_down_deg, "spread_deg": spread_deg},
        {"n_views": n_views, "pitch_down_deg": pitch_down_deg},
        {"n_views": n_views},
    ]
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return list(scan(**attempt))
        except TypeError as exc:
            last_error = exc
    raise last_error if last_error is not None else TypeError("scan_views failed")


def _capture_views(
    world,
    mode: str,
    yaw_offset_deg: float,
    pitch_deg: float,
    n_views: int,
    spread_deg: float | None,
) -> tuple[list[dict], str]:
    """Render the frames an ``observe`` call asks for.

    Returns the views (frame + the heading it looks at + pitch) and a short
    note describing any degradation (missing scan support, for instance).
    """
    scan = getattr(world, "scan_views", None)
    pitch = max(0.0, float(pitch_deg or 0.0))
    note = ""
    with _pose_peek(world, math.radians(yaw_offset_deg or 0.0)):
        yaw0 = _current_yaw(world)
        base_deg = None if yaw0 is None else math.degrees(yaw0)
        if mode == "lookaround":
            if not callable(scan):
                return [_view(world.render(), base_deg)], "scan_views unavailable: current heading only"
            frames = [_as_frame(f) for f in _call_scan(scan, n_views, pitch, spread_deg)]
            headings = _scan_headings(base_deg, n_views, spread_deg)
            pitched = [False] * len(headings)
            while len(frames) > len(headings):
                extra = len(headings) - n_views
                headings.append(_norm_heading(None if base_deg is None else base_deg + 180.0 * extra))
                pitched.append(True)
            views = [
                _view(frame, heading, pitch if flag else 0.0)
                for frame, heading, flag in zip(frames, headings, pitched)
            ]
            return views, note
        if mode == "pitch_down" or pitch > 0.0:
            if not callable(scan):
                return [_view(world.render(), base_deg)], "pitch unsupported by this world: level view only"
            frames = [_as_frame(f) for f in _call_scan(scan, 1, pitch or DEFAULT_PITCH_DOWN_DEG, None)]
            headings = [base_deg, base_deg, _norm_heading(None if base_deg is None else base_deg + 180.0)]
            pitched = [False, True, True]
            views = [
                _view(frame, headings[i] if i < len(headings) else base_deg, pitch if pitched[i] else 0.0)
                for i, frame in enumerate(frames)
            ]
            if len(views) == 1:
                views[0]["note"] = "pitch unsupported by this world: level view only"
            return views, note
        # forward
        return [_view(world.render(), base_deg)], note


def _pose_matches(record: PerceptionObservation, world) -> bool | None:
    """True/False when the record's pose can be compared with the world's
    current pose, None when either pose is unknown."""
    if record.pitched:
        return False  # a pitched frame does not share the level depth map
    if record.position_flu is None or record.yaw_rad is None:
        return None
    position = _current_position(world)
    yaw = _current_yaw(world)
    if position is None or yaw is None:
        return None
    delta = np.abs(np.asarray(position) - np.asarray(record.position_flu))
    if float(np.max(delta)) > POSE_POS_TOL_M:
        return False
    dyaw = math.atan2(math.sin(yaw - record.yaw_rad), math.cos(yaw - record.yaw_rad))
    return bool(abs(dyaw) <= POSE_YAW_TOL_RAD)


def _observation_intrinsics(ctx: ToolContext, record: PerceptionObservation) -> dict:
    """Intrinsics this record's pixels are projected with.

    The record's own when it carries them (every capture records them), else the
    world/context resolution - and never a hard-coded FOV: an unknown
    ``hfov_deg`` makes the projection refuse rather than bias the lateral offset.
    """
    declared = record.intrinsics
    if declared.get("width") and declared.get("height"):
        return declared
    return resolve_intrinsics(ctx, frame=record.frame)


def _observation_scope(ctx: ToolContext, record: PerceptionObservation) -> dict:
    """The condition an observation was made under - the boundary of a null result.

    "Not found here" is only ever "not found from this viewpoint, facing this
    heading, with this field of view, over this region": the scope travels with
    the negative record and comes back with every retrieval, so a single failed
    look can never be read as cleared ground.
    """
    intrinsics = _observation_intrinsics(ctx, record)
    pitch = float(record.metadata.get("pitch_down_deg") or 0.0)
    if view_scope is not None:
        return view_scope(
            record.position_flu,
            yaw_rad=record.yaw_rad,
            hfov_deg=intrinsics.get("hfov_deg"),
            pitch_down_deg=pitch,
            region=record.region,
            observation_id=record.observation_id,
        )
    return {  # pragma: no cover - fallback of the memory import guard
        "position_flu": None if record.position_flu is None
        else [round(float(v), 3) for v in record.position_flu],
        "yaw_deg": record.yaw_deg,
        "coverage_deg": None,
        "hfov_deg": intrinsics.get("hfov_deg"),
        "pitch_down_deg": pitch,
        "max_distance_m": None,
        "region": list(record.region) if record.region else None,
        "observation_id": record.observation_id,
        "note": "",
    }


def _scope_text(scope) -> str:
    """The human boundary line of a scope ("" when the memory API is absent)."""
    if scope_note is None or not isinstance(scope, Mapping):
        return ""
    try:
        return str(scope_note(scope))
    except Exception:  # noqa: BLE001 - a malformed scope is not worth an error
        return ""


def _source_pixel(
    ctx: ToolContext, record: PerceptionObservation, pixel: tuple[float, float]
) -> tuple[tuple[float, float], PerceptionObservation, str]:
    """Map a pixel of ``record`` onto the frame its intrinsics describe.

    A crop is a magnification of a window of another observation, so its own
    pixel grid is *not* the camera's: the pixel is mapped back through the crop
    mapping (region + upscale) into the source observation, which is returned
    with its own intrinsics. A full frame maps onto itself.
    """
    crop = record.crop
    if not crop or not record.parent_id:
        return pixel, record, ""
    source = resolve_observation(ctx, str(record.parent_id))
    if source is None or source is record:
        return pixel, record, POSITION_CROP_SOURCE_UNKNOWN
    source_size = crop.get("source_size_px") or []
    crop_size = crop.get("crop_size_px") or []
    if len(source_size) != 2 or len(crop_size) != 2:
        return pixel, record, POSITION_CROP_SOURCE_UNKNOWN
    upscale = _as_float(crop.get("upscale"), 1.0) or 1.0
    source_w, source_h = float(source_size[0]), float(source_size[1])
    if source_w <= 0 or source_h <= 0 or upscale <= 0:
        return pixel, record, POSITION_CROP_SOURCE_UNKNOWN
    # crop pixel (post-upscale) -> source pixel -> normalized source pixel
    px = pixel[0] / 1000.0 * float(crop_size[0]) / upscale
    py = pixel[1] / 1000.0 * float(crop_size[1]) / upscale
    origin = crop.get("origin_px") or [0.0, 0.0]
    mapped = (
        (float(origin[0]) + px) / source_w * 1000.0,
        (float(origin[1]) + py) / source_h * 1000.0,
    )
    return mapped, source, ""


def _project_pixel(
    ctx: ToolContext,
    record: PerceptionObservation,
    pixel: tuple[float, float],
    distance_m: float,
) -> tuple[list[float] | None, str]:
    """Image point + range -> world position; (None, reason) when it cannot.

    The projection uses the pose the frame was *captured* at, never the drone's
    current pose: the box belongs to that view, and the drone may have flown on
    since. The pinhole model is the observation's own intrinsics, and a pixel on
    the optical axis stays exact even when the field of view was never declared
    (the lateral offset is zero there for any FOV).
    """
    if record.position_flu is None or record.yaw_rad is None:
        return None, POSITION_POSE_UNKNOWN
    pixel, source, problem = _source_pixel(ctx, record, pixel)
    if problem:
        return None, problem
    intrinsics = _observation_intrinsics(ctx, source)
    width, height = int(intrinsics.get("width") or 0), int(intrinsics.get("height") or 0)
    hfov = _as_float(intrinsics.get("hfov_deg"), None)
    if width <= 0 or height <= 0:
        return None, POSITION_INTRINSICS_UNKNOWN
    if hfov is None:
        # pixels are normalized 0-1000, so the optical axis is the centre of the
        # normalized frame; half a pixel of slack, expressed in that scale
        on_axis = abs(pixel[0] - 500.0) <= 500.0 / width and abs(pixel[1] - 500.0) <= 500.0 / height
        if not on_axis:
            return None, POSITION_INTRINSICS_UNKNOWN
        hfov = 60.0  # any FOV maps the optical axis to the forward ray
    dx, dy, dz = pixel_to_body_offset(
        pixel, float(distance_m), width=width, height=height, hfov_deg=hfov
    )
    wx, wy, wz = body_delta_to_world(dx, dy, dz, record.yaw_rad)
    return [
        round(float(record.position_flu[0] + wx), 3),
        round(float(record.position_flu[1] + wy), 3),
        round(float(record.position_flu[2] + wz), 3),
    ], ""


def _pixel_ray(
    ctx: ToolContext, record: PerceptionObservation, pixel: tuple[float, float]
) -> dict | None:
    """World-frame ray of a pixel from the capture pose (None when unknowable).

    What a box with no range still *is*: a direction from a known place. The
    ray is expressed the way the rest of the system speaks - origin in world
    FLU, ``bearing_deg`` = atan2(dy, dx) in degrees (0 = +x), positive
    counter-clockwise, and ``elevation_deg`` above the horizontal plane.
    """
    if record.position_flu is None or record.yaw_rad is None:
        return None
    pixel, source, problem = _source_pixel(ctx, record, pixel)
    if problem:
        return None
    intrinsics = _observation_intrinsics(ctx, source)
    width, height = int(intrinsics.get("width") or 0), int(intrinsics.get("height") or 0)
    hfov = _as_float(intrinsics.get("hfov_deg"), None)
    if hfov is None or width <= 0 or height <= 0:
        return None  # a direction needs the full pinhole, axis included
    dx, dy, dz = pixel_to_body_offset(pixel, 1.0, width=width, height=height, hfov_deg=hfov)
    body_bearing = math.atan2(dy, dx)
    horizontal = math.hypot(dx, dy)
    return {
        "origin_flu": [round(float(v), 3) for v in record.position_flu],
        "bearing_deg": round(_norm_heading(math.degrees(record.yaw_rad + body_bearing)), 1),
        "elevation_deg": round(math.degrees(math.atan2(dz, horizontal)), 1),
        "yaw_deg": round(math.degrees(record.yaw_rad), 1),
    }


def _frame_depth_reader(ctx: ToolContext, record: PerceptionObservation):
    """(depth reader, reason) of the frame a candidate was grounded in.

    A depth map belongs to the frame rendered at the *current* pose, so the
    world's reader is only offered when the grounding observation was captured
    there - the same rule ``inspect_region`` follows. ``reason`` is empty when
    the reader is usable, otherwise it names why no depth may be read from this
    frame.
    """
    world = ctx.world
    if world is None:
        return None, "no_world"
    if record.pitched:
        return None, "pitched_view"  # a pitched frame has no level depth frame
    match = _pose_matches(record, world)
    if match is None:
        return None, "pose_unknown"
    if not match:
        return None, "stale_observation"
    depth_fn = getattr(world, "depth_at_pixel", None)
    if not callable(depth_fn):
        return None, "no_depth_sensor"
    refresh = getattr(world, "refresh_sensing", None)
    if callable(refresh):
        try:
            refresh()  # bind the depth cache to the current frame
        except Exception:  # noqa: BLE001 - depth is best effort
            pass
    return depth_fn, ""


def _candidate_location(
    ctx: ToolContext,
    record: PerceptionObservation,
    box: Box,
    distance_m: float | None,
    depth_fn,
    depth_reason: str,
) -> dict:
    """Where a candidate box is: a position when the range is known, else a ray.

    The range comes from the frame's depth map when ``depth_fn`` is usable, else
    from the model's own ``distance_m`` estimate. A box with neither is *not*
    placed at a guessed range - an invented point would send the drone somewhere
    the model never saw (goal.md section 6): it keeps the direction
    (``ray``, ``position_source`` ``"ray"``) and reports ``range_status``
    ``"unknown"``, so the caller can decide to observe again instead of flying
    to a fiction. ``position_source`` names either how the position was obtained
    (``depth`` / ``model_distance``) or why none exists: ``ray`` (direction
    known, range not - the ray is attached), ``intrinsics_unknown`` (a range is
    known but the field of view is not, so the direction is undetermined; the
    range is attached), ``pitched_view`` / ``pose_unknown`` / ``no_world`` /
    ``crop_source_unknown`` (this frame cannot be projected at all) or
    ``no_valid_depth`` (no range and no usable direction).
    """
    pixel = box.center
    # the depth map belongs to the frame rendered at the *current* pose, so the
    # box must be read at the pixel of that camera: a box grounded inside a crop
    # is in the crop's own pixel grid, and the pixel it maps back to is the one
    # the source frame has (feeding the crop pixel straight into the source depth
    # map measures whatever surface happens to sit under that unrelated pixel -
    # a box at the frame centre then reports the range of whatever is in front
    # of the drone instead of the range of the box)
    depth_pixel, _, pixel_problem = _source_pixel(ctx, record, pixel)
    location: dict = {
        "position_source": POSITION_NO_RANGE,
        "range_status": "unknown",
    }
    if depth_reason == POSITION_PITCHED:
        # a pitched frame does not share the level camera model, so neither a
        # position nor a ray may be derived from its pixels
        location["position_source"] = depth_reason
        return location
    if depth_reason in (POSITION_NO_WORLD, POSITION_POSE_UNKNOWN):
        # no depth sensor (or no capture pose) is no reason to withhold the
        # direction: the ray is the honest answer for a box whose range is
        # unknown, and it is exactly what the caller needs to keep working
        location["position_source"] = depth_reason
        ray = _pixel_ray(ctx, record, pixel)  # None without a known capture pose
        if ray is not None:
            location["ray"] = ray
        return location
    if depth_fn is not None and not pixel_problem:
        depth = _as_float(depth_fn(depth_pixel), None)
        if depth is not None and depth > MIN_RANGE_M:
            point, problem = _project_pixel(ctx, record, pixel, depth)
            if point is not None:
                return {
                    "position_source": POSITION_DEPTH,
                    "position_flu": point,
                    "range_m": round(float(depth), 3),
                    "range_source": POSITION_DEPTH,
                    "range_status": "known",
                }
            location["position_source"] = problem or POSITION_NO_RANGE
            if problem == POSITION_INTRINSICS_UNKNOWN:
                # the range is real, the direction is not: report the range
                location["range_m"] = round(float(depth), 3)
                location["range_source"] = POSITION_DEPTH
                location["range_status"] = "known"
    if distance_m is not None and MIN_RANGE_M < distance_m <= MAX_RANGE_M:
        point, problem = _project_pixel(ctx, record, pixel, distance_m)
        if point is not None:
            return {
                "position_source": POSITION_MODEL_DISTANCE,
                "position_flu": point,
                "range_m": round(float(distance_m), 3),
                "range_source": POSITION_MODEL_DISTANCE,
                "range_status": "known",
            }
        if location["position_source"] == POSITION_NO_RANGE:
            location["position_source"] = problem or POSITION_NO_RANGE
        if problem == POSITION_INTRINSICS_UNKNOWN and location["range_status"] == "unknown":
            location["range_m"] = round(float(distance_m), 3)
            location["range_source"] = POSITION_MODEL_DISTANCE
            location["range_status"] = "known"
    ray = _pixel_ray(ctx, record, pixel)
    if ray is not None:
        location["ray"] = ray
        if location["position_source"] == POSITION_NO_RANGE:
            # a direction from a known place: the honest answer for an unknown
            # distance is the ray, not a point at an invented range
            location["position_source"] = POSITION_RAY
    return location


def _episode_time_s(ctx: ToolContext) -> float:
    """Episode-relative seconds: the runtime may publish ``ctx.extra["time_s"]``,
    otherwise the first perception call of the episode defines t=0."""
    extra = _extra(ctx)
    published = _as_float(extra.get("time_s"), None)
    if published is not None:
        return published
    now = time.time()
    t0 = _as_float(extra.get("perception.t0"), None)
    if t0 is None:
        extra["perception.t0"] = now
        t0 = now
    return round(now - t0, 3)


def _vlm_from_ctx(ctx: ToolContext):
    """Grounding backend for a registry that was built without one: an explicit
    ``ctx.extra`` entry first, then the runtime's own backend
    (``runtime.AgentRuntime`` stores it as ``.vlm``)."""
    extra = _extra(ctx)
    for key in ("perception.vlm", "vlm"):
        backend = extra.get(key)
        if backend is not None and callable(getattr(backend, "decide", None)):
            return backend
    runtime = extra.get("runtime")
    for attribute in ("vlm", "backend", "vlm_backend"):
        backend = getattr(runtime, attribute, None)
        if backend is not None and callable(getattr(backend, "decide", None)):
            return backend
    return None


# --------------------------------------------------------------------------
# visual context: the view index and the direction coverage of one capture
# --------------------------------------------------------------------------


def sector_index(heading_deg: float) -> int:
    """Sector a heading points into (45 deg bins aligned to -180 deg).

    Public because the coverage contract is shared: ``skills_search`` reports the
    sector a viewpoint is expected to cover with the very same binning.
    """
    value = _norm_heading(float(heading_deg)) or 0.0
    return int(math.floor((value + 180.0) / COVERAGE_BIN_DEG)) % 8


def sector_range(index: int) -> list[float]:
    """``[lo, hi]`` of one sector (degrees, ascending, no wrap within a sector)."""
    low = -180.0 + COVERAGE_BIN_DEG * (int(index) % 8)
    return [round(low, 1), round(low + COVERAGE_BIN_DEG, 1)]


def _direction_name(dyaw_deg: float | None, *, pitched: bool = False) -> str:
    """8-way label of a heading offset from the drone's own heading."""
    if dyaw_deg is None:
        return DIRECTION_UNKNOWN
    value = _norm_heading(float(dyaw_deg)) or 0.0
    name = DIRECTION_NAMES[int(math.floor((value + 0.5 * COVERAGE_BIN_DEG) / COVERAGE_BIN_DEG)) % 8]
    return f"down-{name}" if pitched else name


def _view_index(records: list[PerceptionObservation], base_yaw_deg: float | None, primary_id: str) -> list[dict]:
    """``[{obs_id, yaw_deg, direction, note}]``: which way every view looked.

    A capture returns several frames; without the direction each of them faces,
    the model holds a list of ids it cannot choose from (the failure this
    contract removes). ``direction`` is relative to the drone's own heading
    (positive = left, FLU), ``yaw_deg`` is the world heading the frame looked at.
    """
    out = []
    for record in records:
        dyaw = None
        if record.yaw_deg is not None and base_yaw_deg is not None:
            dyaw = _norm_heading(float(record.yaw_deg) - float(base_yaw_deg))
        direction = _direction_name(dyaw, pitched=record.pitched)
        bits = []
        if record.pitched:
            bits.append(f"tilted {float(record.metadata.get('pitch_down_deg') or 0.0):g} deg down")
        else:
            bits.append("level view")
        if record.yaw_deg is None:
            bits.append("heading unknown")
        else:
            bits.append(f"looks {direction[len('down-'):] if record.pitched else direction}")
            if not record.pitched:
                low, high = sector_range(sector_index(record.yaw_deg))
                bits.append(f"covers the {low:g}..{high:g} deg sector")
        if record.observation_id == primary_id:
            bits.append("primary view of this capture")
        out.append(
            {
                "obs_id": record.observation_id,
                "yaw_deg": None if record.yaw_deg is None else round(_norm_heading(record.yaw_deg), 1),
                "direction": direction,
                "note": "; ".join(bits),
            }
        )
    return out


def _worth_looking(records: list[PerceptionObservation], unchecked: list[int]) -> list[dict]:
    """Views worth another look: the ones closest to a direction nobody faced."""
    level = [record for record in records if not record.pitched and record.yaw_deg is not None]
    if not level or not unchecked:
        return []
    found = []
    for index in unchecked:
        low, high = sector_range(index)
        centre = 0.5 * (low + high)
        best = min(level, key=lambda record: abs(_norm_heading(centre - float(record.yaw_deg))))
        dyaw = _norm_heading(centre - float(best.yaw_deg))
        found.append(
            {
                "obs_id": best.observation_id,
                "reason": (
                    f"unchecked {low:g}..{high:g} deg: {best.observation_id} is the closest view "
                    f"(looks {dyaw:+.0f} deg off that direction) - observe facing it, or "
                    "inspect_region this view if the target may sit at its edge"
                ),
            }
        )
    seen, unique = set(), []
    for entry in found:
        if entry["obs_id"] in seen:
            continue
        seen.add(entry["obs_id"])
        unique.append(entry)
    return unique[:MAX_WORTH_LOOKING]


def _coverage_note(checked: list[int], unchecked: list[int], *, down_checked: bool) -> str:
    parts = [f"coverage counts the {COVERAGE_BIN_DEG:g} deg sector each level view was pointed into"]
    if not unchecked:
        parts.append(
            f"full sweep: all {len(checked)} sectors are covered by a level view (coarse)"
        )
        parts.append("pick the direction you care about from view_index and magnify it with inspect_region")
    else:
        ranges = ", ".join(
            f"{sector_range(index)[0]:g}..{sector_range(index)[1]:g}" for index in unchecked
        )
        parts.append(f"{len(checked)}/8 sectors checked; unchecked: {ranges}")
    if not down_checked:
        parts.append("no downward frame: call observe(mode=pitch_down) if the target may be below")
    return "; ".join(parts)


def _coverage_report(records: list[PerceptionObservation], base_yaw_deg: float | None) -> dict:
    """What a capture checked, what it did not, and which view is worth a re-look."""
    checked: set[int] = set()
    for record in records:
        if record.pitched or record.yaw_deg is None:
            continue
        checked.add(sector_index(record.yaw_deg))
    unchecked = [index for index in range(8) if index not in checked]
    down_checked = any(record.pitched for record in records)
    directions_known = bool(checked) or any(
        record.yaw_deg is not None for record in records
    )
    report = {
        "bin_deg": COVERAGE_BIN_DEG,
        "directions_known": bool(directions_known),
        "reference_yaw_deg": None if base_yaw_deg is None else round(float(base_yaw_deg), 1),
        "checked_deg": [sector_range(index) for index in sorted(checked)],
        "unchecked_deg": [] if not directions_known else [sector_range(index) for index in unchecked],
        "all_directions_covered": bool(directions_known) and not unchecked,
        "down_checked": down_checked,
        "worth_looking": _worth_looking(records, unchecked) if directions_known else [],
        "note": "",
    }
    if not directions_known:
        report["note"] = (
            "the world exposed no heading for this capture: which direction each view "
            "faces is unknown - re-observe once a heading is available"
        )
    else:
        report["note"] = _coverage_note(sorted(checked), unchecked, down_checked=down_checked)
    return report


def _expansion_guidance(report: dict, primary_id: str) -> dict:
    """How to pull any view of this capture back as pixels (``inspect_region``)."""
    worth = report.get("worth_looking") or []
    pick = str(worth[0]["obs_id"]) if worth else primary_id
    guidance: dict[str, Any] = {
        "how": (
            "any view in view_index can be pulled back as pixels: "
            'inspect_region({"observation_id": <obs_id>, "region": [x1, y1, x2, y2]}) '
            "returns a crop observation id of that view, magnified up to 8x; region is a "
            "normalized 0-1000 corner box and [0, 0, 1000, 1000] is the whole view"
        ),
        "suggested_obs_id": pick,
        "full_view": {
            "tool": "inspect_region",
            "args": {"observation_id": pick, "region": [0, 0, 1000, 1000]},
        },
        "zoom_example": {
            "tool": "inspect_region",
            "args": {"observation_id": pick, "region": [250, 250, 750, 750]},
        },
    }
    if not report.get("down_checked"):
        guidance["pitch_down"] = {"tool": "observe", "args": {"mode": "pitch_down"}}
    unchecked = report.get("unchecked_deg") or []
    if unchecked:
        low, high = unchecked[0]
        guidance["unchecked_hint"] = (
            f"no level view faced {low:g}..{high:g} deg: "
            f"observe({{'mode': 'forward', 'yaw_deg': <offset>}}) turning toward that sector"
        )
    return guidance


# --------------------------------------------------------------------------
# observe
# --------------------------------------------------------------------------


def observe(ctx: ToolContext, args: dict) -> ToolResult:
    """Capture evidence: 360 lookaround, a forward frame, or a pitched-down view.

    ``mode`` is optional and defaults to ``forward`` (the prompt's
    ``observe({})`` example is a valid call, not an error). The result carries
    the *visual context* of the capture - ``view_index`` (which way each of the
    returned views looked), ``coverage`` (which 45 deg sectors a level view has
    checked, which are unchecked, and which view is worth a second look) and
    ``expansion`` (the ``inspect_region`` calls that pull one view's pixels back)
    - because a bare list of observation ids is not something a model can choose
    from.
    """
    world = ctx.world
    if world is None:
        return ToolResult(STATUS_FAILED, {"error": "no_world"})
    args = args or {}
    mode = str(args.get("mode") or "").strip().lower() or DEFAULT_OBSERVE_MODE
    if mode not in OBSERVE_MODES:
        return ToolResult(
            STATUS_FAILED,
            {"error": "invalid_mode", "mode": mode or None, "allowed": list(OBSERVE_MODES)},
        )
    yaw_deg = _as_float(args.get("yaw_deg"), 0.0) or 0.0
    default_pitch = DEFAULT_PITCH_DOWN_DEG if mode == "pitch_down" else 0.0
    pitch_deg = max(0.0, _as_float(args.get("pitch_deg"), default_pitch) or 0.0)
    n_views = int(min(max(_as_int(args.get("n_views"), DEFAULT_LOOKAROUND_VIEWS), 1), MAX_LOOKAROUND_VIEWS))
    spread_deg = _as_float(args.get("spread_deg"), None)

    try:
        views, capture_note = _capture_views(world, mode, yaw_deg, pitch_deg, n_views, spread_deg)
    except Exception as exc:  # noqa: BLE001 - report instead of crashing the loop
        return ToolResult(STATUS_FAILED, {"error": f"{type(exc).__name__}: {exc}", "mode": mode})

    primary_index = 0
    if mode != "forward":
        pitched = [i for i, view in enumerate(views) if view["pitch_down_deg"] > 0.0]
        if pitched:
            primary_index = pitched[0]
    kept = [(i, view) for i, view in enumerate(views) if view["frame"] is not None]
    if not kept:
        return ToolResult(STATUS_FAILED, {"error": "render_failed", "mode": mode})

    position = _current_position(world)
    time_s = _episode_time_s(ctx)
    store_kind = KIND_SCAN if mode == "lookaround" else KIND_EVIDENCE
    intrinsics = resolve_intrinsics(ctx)
    records: list[PerceptionObservation] = []
    for i, view in kept:
        yaw_rad = None if view["yaw_deg"] is None else math.radians(view["yaw_deg"])
        record = PerceptionObservation(
            observation_id=ctx.next_obs_id(),
            frame=view["frame"],
            position_flu=position,
            yaw_rad=yaw_rad,
            time_s=time_s,
            kind="view",
            note=view["note"] or f"{mode} view",
            metadata={
                "mode": mode,
                "episode_id": ctx.episode_id,
                "pitch_down_deg": view["pitch_down_deg"],
                "primary": i == primary_index,
                # the intrinsics of the render this view came from: the frame
                # shape is the resolution actually rendered
                "intrinsics": {**intrinsics, **resolve_intrinsics(ctx, frame=view["frame"])},
            },
        )
        records.append(record)
    primary_local = next((k for k, (i, _) in enumerate(kept) if i == primary_index), 0)
    primary = records[primary_local]

    # the drone's own heading during the capture (``_pose_peek`` always restores
    # it): the frame of reference of every direction label
    base_yaw_deg = _current_yaw(world)
    base_yaw_deg = None if base_yaw_deg is None else math.degrees(base_yaw_deg)
    coverage = _coverage_report(records, base_yaw_deg)
    # the checked sectors of the capture ride on the primary write: they belong
    # to the spatial layer, which is fed by the same record call
    sink = _remember(ctx, records[0], coverage=coverage)
    for record in records[1:]:
        _remember(ctx, record)
    # one observation per view in the episode's repository as well: the ids the
    # model was handed ("image_refs") must resolve for every later tool
    for record in records:
        _register_in_store(
            ctx,
            record,
            kind=store_kind,
            detail={"pitch_down_deg": float(record.metadata.get("pitch_down_deg") or 0.0)},
        )
    # publish the coverage for the tools that must suggest a *different* look
    # when a query keeps grounding to nothing (the anti-loop of the grounder).
    # The key is namespaced on purpose: the runtime reads the coverage from the
    # *result* of this call, and a context key that outlived the episode would
    # let an earlier scan look current.
    _extra(ctx)[VIEW_COVERAGE_KEY] = {
        "reference_yaw_deg": coverage["reference_yaw_deg"],
        "checked_deg": [list(item) for item in coverage["checked_deg"]],
        "unchecked_deg": [list(item) for item in coverage["unchecked_deg"]],
        "down_checked": coverage["down_checked"],
        "directions_known": coverage["directions_known"],
        "worth_looking": coverage["worth_looking"],
    }

    data = {
        "observation_id": primary.observation_id,
        "pose": None if position is None else [round(v, 3) for v in position],
        "yaw_deg": _norm_heading(primary.yaw_deg),
        "time_s": time_s,
        "image_refs": [record.observation_id for record in records],
        "n_views": len(records),
        "mode": mode,
        # the visual context of the capture comes before the id lists: the
        # runtime truncates tool results into the prompt, and "which directions
        # are checked / worth a re-look" is what the next decision needs
        "coverage": coverage,
        "view_index": _view_index(records, base_yaw_deg, primary.observation_id),
        "expansion": _expansion_guidance(coverage, primary.observation_id),
        "headings_deg": [_norm_heading(record.yaw_deg) for record in records],
        "kind": store_kind,
        "intrinsics": primary.intrinsics,
        "memory_sink": sink,
        # what the three layers hold after this write (the ablation report reads
        # coverage/places/sightings straight off the tool results)
        "memory_layers": {"sink": sink, **_MemoryBridge(ctx.memory).layer_status()},
    }
    if capture_note:
        data["note"] = capture_note
    evidence = [
        EvidenceRef(
            "observation",
            record.observation_id,
            f"{mode} yaw={_norm_heading(record.yaw_deg)} pitch={record.metadata['pitch_down_deg']:.0f}",
        )
        for record in records
    ]
    return ToolResult(STATUS_OK, data, evidence)


# --------------------------------------------------------------------------
# inspect_region
# --------------------------------------------------------------------------


def inspect_region(ctx: ToolContext, args: dict) -> ToolResult:
    """Magnify a region of an earlier observation (+ depth when still current)."""
    args = args or {}
    try:
        observation_id = require_observation_id(args.get("observation_id"), ctx)
    except IdTypeError as exc:
        return ToolResult(STATUS_FAILED, exc.to_data())
    record = resolve_observation(ctx, observation_id)
    hint = args.get("box_format")
    box = None
    if record is not None:
        box, reason = parse_box(
            args.get("region"),
            width=int(record.frame.shape[-1]),
            height=int(record.frame.shape[-2]),
            box_format=hint,
        )
    else:
        # the id is known but its pixels are gone: parse the region anyway so the
        # answer can say the frame, not the box, is the problem
        box, reason = parse_box(args.get("region"), box_format=hint)
    if box is None:
        if reason == "region_outside_frame":
            return ToolResult(
                STATUS_BLOCKED,
                {
                    "reason": reason,
                    "observation_id": observation_id,
                    "region": args.get("region"),
                    "box_format": BOX_FORMAT_DOC,
                },
            )
        return ToolResult(
            STATUS_FAILED,
            {
                "error": reason,
                "observation_id": observation_id,
                "region": args.get("region"),
                "box_format": BOX_FORMAT_DOC,
            },
        )
    if record is None:
        return ToolResult(STATUS_FAILED, _observation_pixels_missing(ctx, observation_id))
    region = box.to_uvwh()

    crop, crop_size_px, upscale, crop_origin = _crop_frame(record.frame, region)
    world = ctx.world
    depth_fn = getattr(world, "depth_at_pixel", None) if world is not None else None
    depth_m, depth_samples, depth_status = None, 0, "no_depth_sensor"
    if callable(depth_fn):
        current = _pose_matches(record, world) if world is not None else None
        if current is False:
            depth_status = "stale_observation"
        elif current is None:
            depth_status = "pose_unknown"
        else:
            refresh = getattr(world, "refresh_sensing", None)
            if callable(refresh):
                try:
                    refresh()
                except Exception:  # noqa: BLE001 - depth is best effort
                    pass
            depth_m, depth_samples = _region_depth(depth_fn, region)
            depth_status = "ok" if depth_m is not None else "no_valid_depth"

    crop_id = ctx.next_obs_id()
    source_intrinsics = _observation_intrinsics(ctx, record)
    crop_record = PerceptionObservation(
        observation_id=crop_id,
        frame=crop,
        position_flu=record.position_flu,
        yaw_rad=record.yaw_rad,
        time_s=_episode_time_s(ctx),
        kind="crop",
        parent_id=record.observation_id,
        region=region,
        note=f"crop region={region} of {record.observation_id}",
        metadata={
            "source": record.observation_id,
            "upscale": round(upscale, 2),
            "crop_size_px": crop_size_px,
            "episode_id": ctx.episode_id,
            # a crop has no camera of its own: its pixels are projected with the
            # SOURCE observation's intrinsics through this mapping
            "intrinsics": source_intrinsics,
            "crop": {
                "source_observation_id": record.observation_id,
                "region_uvwh": list(region),
                "origin_px": list(crop_origin),
                "source_size_px": [int(record.frame.shape[-1]), int(record.frame.shape[-2])],
                "crop_size_px": list(crop_size_px),
                "upscale": round(float(upscale), 3),
            },
        },
    )
    sink = _remember(ctx, crop_record)
    _register_in_store(
        ctx,
        crop_record,
        kind=KIND_INSPECT,
        detail={"crop": dict(crop_record.metadata["crop"]), "region": list(region)},
    )

    notes = []
    if upscale > 1.001:
        notes.append(f"upscaled {upscale:.1f}x from {crop_size_px[0]}x{crop_size_px[1]} px")
    else:
        notes.append(f"no upscale ({crop_size_px[0]}x{crop_size_px[1]} px)")
    if depth_m is not None:
        notes.append(f"depth from the current frame ({depth_samples} samples)")
    elif depth_status == "stale_observation":
        notes.append("depth skipped: observation is from an earlier pose")
    elif depth_status == "pose_unknown":
        notes.append("depth skipped: capture pose unknown for this observation")
    elif depth_status == "no_valid_depth":
        notes.append("no valid depth inside the region")
    else:
        notes.append("world exposes no depth sensor")

    data = {
        "crop_image_ref": crop_id,
        "source_observation_id": record.observation_id,
        "region": box.to_list(),
        "region_format": box.format_used,
        "region_uvwh": region,
        "box_format": BOX_FORMAT_DOC,
        "crop_size_px": crop_size_px,
        "upscale": round(upscale, 2),
        "intrinsics": source_intrinsics,
        "depth_status": depth_status,
        "note": "; ".join(notes),
        "memory_sink": sink,
    }
    if depth_m is not None:
        data["depth_m"] = round(depth_m, 3)
        data["depth_samples"] = depth_samples
    evidence = [
        EvidenceRef("observation", crop_id, f"crop {region} of {record.observation_id}"),
        EvidenceRef("observation", record.observation_id, "source observation"),
    ]
    return ToolResult(STATUS_OK, data, evidence)


def _observation_pixels_missing(ctx: ToolContext, observation_id: str) -> dict:
    """Answer for an id that is known but has no pixels to magnify."""
    return {
        "error": "observation_pixels_unavailable",
        "observation_id": observation_id,
        "available": available_observation_ids(ctx),
        "hint": (
            "this observation is known but its frame is no longer in memory; "
            "call observe to capture a fresh frame"
        ),
    }


# --------------------------------------------------------------------------
# ground_target: prompt + reply parsing
# --------------------------------------------------------------------------


def _ground_user_prompt(instruction: str) -> str:
    return (
        f"Instruction: {instruction}\n"
        "Return the normalized box (0-1000), the attributes and the rough "
        'distance in meters ("distance_m"; 0 if you cannot tell) of every '
        "visible instance of that target; report what you are unsure about in "
        '"uncertainty".'
    )


def _extract_json(text: str, attempts: int = 8):
    """First decodable JSON object/array in a model reply (prose tolerated)."""
    if not text:
        return None
    for opener, closer in (("{", "}"), ("[", "]")):
        tried = 0
        for i, char in enumerate(text):
            if char != opener:
                continue
            tried += 1
            if tried > attempts:
                break
            depth = 0
            for j in range(i, len(text)):
                if text[j] == opener:
                    depth += 1
                elif text[j] == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[i : j + 1])
                        except json.JSONDecodeError:
                            break
    return None


def _candidate_items(payload) -> list | None:
    """Accept {"candidates": [...]}, a bare list, or a single candidate dict."""
    if isinstance(payload, dict):
        for key in ("candidates", "targets", "objects"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                return [value]
        if any(key in payload for key in ("box", "bbox", "region")):
            return [payload]
        return None
    if isinstance(payload, list):
        return payload
    return None


def _norm_attrs(value) -> dict:
    if isinstance(value, dict):
        out = {}
        for key, item in list(value.items())[:8]:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out[str(key)[:24]] = text[:60]
        return out
    if isinstance(value, (list, tuple)):
        text = ", ".join(str(item) for item in list(value)[:5])
        return {"text": text[:120]} if text else {}
    if value is None:
        return {}
    text = str(value).strip()
    return {"text": text[:120]} if text else {}


def _normalize_candidates(items: list, store: dict, *, width=None, height=None) -> list[dict]:
    """Model reply -> candidate records with canonical ``[x1, y1, x2, y2]`` boxes.

    Each record keeps the format it was written in (``box_format``: the legacy
    ``[u, v, w, h]`` reading is normalized here, not silently re-guessed later),
    plus the legacy ``[u, v, w, h]`` view of the same box in ``box_uvwh`` so a
    consumer that still speaks the old shape does not have to convert.
    """
    candidates = []
    next_index = len(store) + 1
    for item in items[:5]:
        if not isinstance(item, dict):
            continue
        box, _reason = parse_box(
            item.get("box", item.get("bbox", item.get("region"))),
            width=width,
            height=height,
            box_format=item.get("box_format", item.get("format")),
        )
        if box is None:
            continue
        while f"cand_{next_index}" in store:
            next_index += 1
        candidate_id = f"cand_{next_index}"
        next_index += 1
        confidence = _as_float(item.get("confidence", item.get("score")), None)
        if confidence is not None:
            confidence = round(min(max(confidence, 0.0), 1.0), 2)
        distance = _as_float(item.get("distance_m", item.get("distance")), None)
        if distance is not None and not (MIN_RANGE_M < distance <= MAX_RANGE_M):
            distance = None
        uncertainty = item.get("uncertainty", item.get("uncertain", item.get("notes", "")))
        candidates.append(
            {
                "id": candidate_id,
                "box": box.to_list(),
                "box_format": BOX_FORMAT_OUT,
                "box_format_used": box.format_used,
                "box_uvwh": box.to_uvwh(),
                "attrs": _norm_attrs(item.get("attrs", item.get("attributes"))),
                "confidence": confidence,
                "distance_m": None if distance is None else round(distance, 2),
                "uncertainty": "" if uncertainty is None else str(uncertainty)[:120],
            }
        )
    return candidates


# --------------------------------------------------------------------------
# the grounding ledger: no query may grind the same evidence forever
# --------------------------------------------------------------------------


def _instruction_signature(instruction: str) -> str:
    """Normalized query intention (the "what am I looking for" dimension)."""
    return " ".join(str(instruction or "").lower().split())[:120]


def _region_signature(region: Any) -> str:
    """Coarse signature of the evidence region (a crop's ``[u, v, w, h]``).

    The region dimension of the evidence key comes from the frame *itself*
    (``PerceptionObservation.region`` of a magnified crop), not from an argument
    ``ground_target`` would ignore: two calls are the same evidence only when
    they really show the same pixels.
    """
    if not region:
        return "full"
    try:
        values = [float(value) for value in list(region)[:4]]
    except (TypeError, ValueError):
        return "full"
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return "full"
    cells = [int(round(value / EVIDENCE_REGION_TOL)) for value in values]
    return "r:" + ",".join(str(cell) for cell in cells)


def grounding_ledger(ctx: ToolContext) -> dict:
    """The episode's grounding-attempt ledger (``GROUNDING_LEDGER_KEY``).

    Keyed by ``<query signature>|<region signature>``; each value is a bounded
    list of attempts, one per evidence neighbourhood (pose + heading within the
    "same evidence" tolerances). An ``obs_N`` that changes every turn while the
    drone hovers is *one* entry here - which is what stops a model from reading
    a fresh id as fresh evidence.
    """
    extra = _extra(ctx)
    ledger = extra.get(GROUNDING_LEDGER_KEY)
    if not isinstance(ledger, dict):
        ledger = {}
        extra[GROUNDING_LEDGER_KEY] = ledger
    return ledger


def _same_evidence(entry: dict, position, yaw_deg) -> bool:
    """Is this ledger entry the evidence just grounded (pose + heading)?"""
    stored_position = entry.get("position")
    if (stored_position is None) != (position is None):
        return False
    if position is not None and stored_position is not None:
        if max(abs(float(a) - float(b)) for a, b in zip(position, stored_position)) > EVIDENCE_POS_TOL_M:
            return False
    stored_yaw = entry.get("yaw_deg")
    if (stored_yaw is None) != (yaw_deg is None):
        return False
    if yaw_deg is not None and stored_yaw is not None:
        if abs(_norm_heading(float(yaw_deg) - float(stored_yaw)) or 0.0) > EVIDENCE_YAW_TOL_DEG:
            return False
    return True


def _evidence_attempt(
    ledger: dict,
    query_signature: str,
    region_signature: str,
    position,
    yaw_deg,
    *,
    create: bool = False,
) -> dict | None:
    """The ledger entry of this evidence, optionally created."""
    key = f"{query_signature}|{region_signature}"
    entries = ledger.get(key)
    if not isinstance(entries, list):
        if not create:
            return None
        entries = []
        ledger[key] = entries
    for entry in entries:
        if _same_evidence(entry, position, yaw_deg):
            return entry
    if not create:
        return None
    entry = {
        "position": None if position is None else [round(float(value), 3) for value in position],
        "yaw_deg": None if yaw_deg is None else round(float(yaw_deg), 2),
        "attempts": 0,
        "failures": 0,
        "successes": 0,
        "last_observation_id": "",
        "last_reason": "",
    }
    entries.append(entry)
    del entries[:-GROUNDING_NEIGHBOUR_LIMIT]
    while len(ledger) > GROUNDING_LEDGER_LIMIT:
        ledger.pop(next(iter(ledger)))
    return entry


def _least_checked_sector(unchecked: list, checked: list) -> float | None:
    """Centre of the sector least like anything a level view has faced."""
    centres = [0.5 * (float(lo) + float(hi)) for lo, hi in checked]
    best, best_key = None, None
    for low, high in unchecked:
        centre = 0.5 * (float(low) + float(high))
        distance = min((abs(_norm_heading(centre - other) or 0.0) for other in centres), default=180.0)
        key = (round(distance, 3), centre)
        if best_key is None or key > best_key:
            best, best_key = centre, key
    return best


def _strategy_change(ctx: ToolContext, record: PerceptionObservation, kind: str) -> dict:
    """One concrete next move when the same evidence answered nothing again.

    ``kind`` is the recommended one of the three ("inspect_region" - magnify the
    evidence; "change_view" - turn to a direction no view has faced yet;
    "change_viewpoint" - look at the place from somewhere else); the other two
    travel along as ``alternatives``, each with the exact call to make.
    """
    coverage = _extra(ctx).get(VIEW_COVERAGE_KEY)
    coverage = dict(coverage) if isinstance(coverage, dict) else {}
    unchecked = coverage.get("unchecked_deg") or []
    reference = _as_float(coverage.get("reference_yaw_deg"), None)
    target = _least_checked_sector(unchecked, coverage.get("checked_deg") or []) if unchecked else None
    if target is not None and reference is not None:
        turn = _norm_heading(target - reference)
        change_view = {
            "kind": "change_view",
            "tool": "observe",
            "args": {"mode": "forward", "yaw_deg": round(float(turn), 1)},
            "why": (
                f"turn toward {target:g} deg - the direction no view has faced yet - and "
                "observe there: the same pixels cannot start holding the target"
            ),
        }
    else:
        change_view = {
            "kind": "change_view",
            "tool": "observe",
            "args": {"mode": "forward", "yaw_deg": 45.0},
            "why": "turn to a neighbouring direction and observe: re-reading the same frame cannot add evidence",
        }
    options = {
        "inspect_region": {
            "kind": "inspect_region",
            "tool": "inspect_region",
            "args": {"observation_id": record.observation_id, "region": [250, 250, 750, 750]},
            "why": (
                "the target may be too small at this resolution: magnify the middle of the "
                "frame and ground the returned crop id (crop_image_ref) with the same "
                "instruction instead of grounding this frame again"
            ),
        },
        "change_view": change_view,
        "change_viewpoint": {
            "kind": "change_viewpoint",
            "tool": "propose_viewpoints",
            "args": {"count": 3},
            "why": (
                "the target is not visible from this pose: propose observation points and fly "
                "to one with go_to_viewpoint (search_target runs that routine end to end), "
                "then ground from the new view"
            ),
        },
    }
    chosen = dict(options.get(kind, options["change_view"]))
    chosen["alternatives"] = [
        dict(option) for name, option in options.items() if name != chosen["kind"]
    ]
    return chosen


def _sighting_of_candidate(candidate: dict, instruction: str, record: PerceptionObservation) -> dict:
    """The sighting a grounded candidate *is*: image, candidate and basis.

    What ``ground_target`` knows that a plain capture does not: which frame,
    which candidate id, the box the grounding model wrote and the attributes it
    read. That is the judgement basis a later ``query_memory`` cites, so the
    sighting is stored with it instead of a bare target name.
    """
    attributes = candidate.get("attrs") if isinstance(candidate.get("attrs"), dict) else {}
    basis = (
        f"ground_target hit for '{instruction}': box={candidate.get('box')} "
        f"({candidate.get('box_format_used') or candidate.get('box_format')})"
    )
    if candidate.get("confidence") is not None:
        basis += f" confidence={candidate['confidence']}"
    if attributes:
        basis += " attrs=" + ",".join(
            f"{key}={value}" for key, value in list(attributes.items())[:4]
        )
    if candidate.get("position_source"):
        basis += f" position_source={candidate['position_source']}"
    return {
        "target": instruction,
        "image": record.observation_id,
        "candidate": candidate.get("id"),
        "box": candidate.get("box"),
        "basis": basis,
        "confidence": candidate.get("confidence"),
        "attributes": attributes,
        "position_flu": candidate.get("position_flu"),
        "position_source": candidate.get("position_source"),
        "range_m": candidate.get("range_m"),
        "note": candidate.get("uncertainty") or "",
        "step": record.metadata.get("step"),
    }


def _stagnant_result(
    ctx: ToolContext, instruction: str, record: PerceptionObservation, attempt: dict
) -> ToolResult:
    """``blocked`` answer for evidence that already answered ``no_candidate`` too often."""
    failures = int(attempt.get("failures", 0))
    change = _strategy_change(ctx, record, "change_viewpoint")
    data = {
        "reason": "stagnant_grounding",
        "instruction": instruction,
        "observation_id": record.observation_id,
        "attempts": failures + 1,
        "failed_attempts": failures,
        "repeated_grounding": True,
        "model_call_skipped": True,
        "hint": change["why"],
        "note": (
            f"'{instruction}' was already grounded {failures}x on this evidence (same pose, "
            "heading and region) with no candidate: a new observation id is not new "
            "evidence, so this call did not run the model again"
        ),
        "strategy_change": change,
        "next_action": {"tool": change["tool"], "args": dict(change["args"])},
        "available_candidate_ids": available_candidate_ids(ctx),
    }
    evidence = [EvidenceRef("observation", record.observation_id, "frame already grounded")]
    return ToolResult(STATUS_BLOCKED, data, evidence)


# --------------------------------------------------------------------------
# ground_target
# --------------------------------------------------------------------------


def ground_target(ctx: ToolContext, args: dict, vlm: Any = None) -> ToolResult:
    """Ground an instruction into candidate boxes inside one observation."""
    args = args or {}
    instruction = str(args.get("instruction") or "").strip()
    if not instruction:
        return ToolResult(STATUS_FAILED, {"error": "missing_instruction"})
    observation_id = None
    if args.get("observation_id") not in (None, ""):
        try:
            observation_id = require_observation_id(args.get("observation_id"), ctx)
        except IdTypeError as exc:
            return ToolResult(STATUS_FAILED, exc.to_data())
    captured = False
    record = resolve_observation(ctx, observation_id) if observation_id else None
    if observation_id and record is None:
        return ToolResult(STATUS_FAILED, _observation_pixels_missing(ctx, observation_id))
    evidence: list[EvidenceRef] = []
    if record is None:
        observed = observe(ctx, {"mode": "forward"})
        if observed.status != STATUS_OK:
            return ToolResult(
                STATUS_FAILED,
                {"error": "no_observation", "detail": observed.data.get("error")},
            )
        observation_id = observed.data["observation_id"]
        record = resolve_observation(ctx, observation_id)
        captured = True
        evidence.append(EvidenceRef("observation", observation_id, "captured for grounding"))
    if record is None:  # pragma: no cover - defensive
        return ToolResult(STATUS_FAILED, {"error": "no_observation", "observation_id": observation_id})

    backend = vlm if vlm is not None else _vlm_from_ctx(ctx)
    if backend is None:
        backend_name = None
    else:
        backend_name = getattr(backend, "name", None) or type(backend).__name__
    if not evidence:
        evidence.append(EvidenceRef("observation", record.observation_id, "grounding frame"))
    if backend is None:
        return ToolResult(
            STATUS_EMPTY,
            {
                "reason": "no_vlm_backend",
                "instruction": instruction,
                "observation_id": record.observation_id,
                "captured_observation": captured,
                "note": "rule fallback: no grounding model attached to this runtime",
            },
            evidence,
        )
    # the anti-loop (goal.md section 5): a new observation id is not new
    # evidence, so attempts are counted per evidence, and evidence that already
    # answered ``no_candidate`` twice is refused instead of ground again
    ledger = grounding_ledger(ctx)
    attempt = _evidence_attempt(
        ledger,
        _instruction_signature(instruction),
        _region_signature(record.region),
        record.position_flu,
        record.yaw_deg,
    )
    previous_failures = int(attempt.get("failures", 0)) if attempt is not None else 0
    if previous_failures >= STALL_BLOCK_AT - 1:
        return _stagnant_result(ctx, instruction, record, attempt)
    try:
        reply = backend.decide(
            [record.frame],
            GROUND_SYSTEM_PROMPT,
            _ground_user_prompt(instruction),
            context={
                "observation_id": record.observation_id,
                "instruction": instruction,
                "pose": None if record.position_flu is None else list(record.position_flu),
                "yaw_deg": record.yaw_deg,
                "intrinsics": _observation_intrinsics(ctx, record),
                "tool": "ground_target",
            },
        )
    except Exception as exc:  # noqa: BLE001 - backend failures are reported
        return ToolResult(
            STATUS_FAILED,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "observation_id": record.observation_id,
                "backend": backend_name,
            },
            evidence,
        )

    items = _candidate_items(_extract_json(reply if isinstance(reply, str) else str(reply)))
    if items is None:
        return ToolResult(
            STATUS_EMPTY,
            {
                "reason": "unparsable_reply",
                "instruction": instruction,
                "observation_id": record.observation_id,
                "backend": backend_name,
            },
            evidence,
        )
    store = _candidate_store(ctx)
    candidates = _normalize_candidates(
        items,
        store,
        width=int(record.frame.shape[-1]),
        height=int(record.frame.shape[-2]),
    )
    if not candidates:
        # one more empty answer on this evidence: the 2nd consecutive one carries
        # the "look somewhere else" alternatives, and a 3rd call is refused above
        attempt = _evidence_attempt(
            ledger,
            _instruction_signature(instruction),
            _region_signature(record.region),
            record.position_flu,
            record.yaw_deg,
            create=True,
        )
        attempt["attempts"] = int(attempt.get("attempts", 0)) + 1
        attempt["failures"] = int(attempt.get("failures", 0)) + 1
        attempt["last_observation_id"] = record.observation_id
        attempt["last_reason"] = "no_candidate"
        failures = int(attempt["failures"])
        # a null result is remembered with the exact condition it was gathered
        # under - never as "this area holds nothing" (the negative-memory rule)
        visibility = _observation_scope(ctx, record)
        negative_write = _MemoryBridge(ctx.memory).add_negative(
            record.observation_id,
            target=instruction,
            reason="no_candidate",
            visibility=visibility,
            note=f"grounded '{instruction}' from this view: no candidate",
            step=record.metadata.get("step"),
        )
        data = {
            "reason": "no_candidate",
            "instruction": instruction,
            "observation_id": record.observation_id,
            "backend": backend_name,
            "attempts": failures,
            "visibility": visibility,
            "scope_note": _scope_text(visibility),
            "memory_negative": negative_write,
        }
        if failures >= STALL_FLAG_AT:
            change = _strategy_change(
                ctx, record, "inspect_region" if not record.region else "change_view"
            )
            data["repeated_grounding"] = True
            data["hint"] = (
                f"'{instruction}' answered no_candidate {failures}x on this evidence (same "
                "pose, heading and region): do not ground it again - " + change["why"]
            )
            data["strategy_change"] = change
            data["next_action"] = {"tool": change["tool"], "args": dict(change["args"])}
        return ToolResult(STATUS_EMPTY, data, evidence)
    # a candidate was found: the evidence is not stagnant any more
    if attempt is not None:
        attempt["attempts"] = int(attempt.get("attempts", 0)) + 1
        attempt["failures"] = 0
        attempt["successes"] = int(attempt.get("successes", 0)) + 1
        attempt["last_observation_id"] = record.observation_id
        attempt["last_reason"] = "candidate"
    depth_fn, depth_reason = _frame_depth_reader(ctx, record)
    for candidate in candidates:
        candidate["observation_id"] = record.observation_id
        candidate["source"] = "ground_target"
        candidate["instruction"] = instruction[:120]
        box = parse_box(candidate["box"], box_format=BOX_FORMAT_XYXY)[0]
        if box is not None:
            candidate.update(
                _candidate_location(ctx, record, box, candidate["distance_m"], depth_fn, depth_reason)
            )
        else:  # pragma: no cover - the record's box came from this parser
            candidate["position_source"] = POSITION_NO_RANGE
            candidate["range_status"] = "unknown"
        store[candidate["id"]] = candidate
    evidence.extend(EvidenceRef("memory", candidate["id"], "candidate") for candidate in candidates)
    # every candidate is a sighting of *this* frame: image + candidate + basis,
    # which is what makes a later "where did I see it" answer evidence instead
    # of a frame that merely inherited the task's target name
    sighting_write = _MemoryBridge(ctx.memory).add_sighting(
        record.observation_id,
        [_sighting_of_candidate(candidate, instruction, record) for candidate in candidates],
        step=record.metadata.get("step"),
    )
    data = {
        "instruction": instruction,
        "observation_id": record.observation_id,
        "captured_observation": captured,
        "backend": backend_name,
        "n_candidates": len(candidates),
        "candidates": candidates,
        "box_format": BOX_FORMAT_DOC,
        "intrinsics": _observation_intrinsics(ctx, record),
        "memory_sighting": sighting_write,
    }
    return ToolResult(STATUS_OK, data, evidence)


# --------------------------------------------------------------------------
# compare_candidates
# --------------------------------------------------------------------------


def _resolve_candidates(
    ctx: ToolContext, value
) -> tuple[list[dict], list[str], list[IdTypeError]]:
    """Accept candidate ids, inline candidate dicts, or a list of either.

    Three outcomes, kept apart on purpose: resolved records, ids of the right
    type that do not exist (``unknown``, an "empty" answer), and ids of the
    *wrong* type (``IdTypeError``, a hard error naming the type and the ids of
    the type the argument wanted) - handing an ``obs_N`` to a candidate
    parameter must not look like "no such candidate".
    """
    if value is None:
        return [], [], []
    items = value if isinstance(value, (list, tuple)) else [value]
    store = _candidate_store(ctx)
    resolved, unknown, type_errors = [], [], []
    for item in items:
        if isinstance(item, dict):
            candidate = dict(item)
            candidate.setdefault("id", f"inline_{len(resolved) + 1}")
            resolved.append(candidate)
            continue
        text = str(item)
        if classify_id(text) not in (None, CANDIDATE_ID_TYPE):
            try:
                require_candidate_id(text, ctx)
            except IdTypeError as exc:
                type_errors.append(exc)
            continue
        canonical = parse_id_index(text, CANDIDATE_ID_TYPE) or text
        candidate = store.get(text) or store.get(canonical)
        if candidate is None:
            unknown.append(text)
        else:
            resolved.append(candidate)
    return resolved, unknown, type_errors


def _candidate_box(candidate: dict) -> "Box | None":
    """Canonical box of a candidate record (its own format recorded on it)."""
    box, _reason = parse_box(
        candidate.get("box"),
        box_format=candidate.get("box_format") or candidate.get("box_format_used"),
    )
    return box


def _pick_candidate(candidates: list[dict]) -> dict | None:
    if not candidates:
        return None
    return max(candidates, key=lambda c: (c.get("confidence") is not None, c.get("confidence") or 0.0))


def _iou(box_a: list[float], box_b: list[float]) -> float:
    """IoU of two canonical ``[x1, y1, x2, y2]`` corner boxes."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def _compare_attrs(attrs_a: dict, attrs_b: dict):
    basis, conflicts, to_confirm = [], [], []
    for key in sorted(set(attrs_a) & set(attrs_b)):
        value_a = attrs_a[key].strip().lower()
        value_b = attrs_b[key].strip().lower()
        if value_a == value_b:
            basis.append(f"attr {key}={attrs_a[key]}")
        else:
            conflicts.append(f"attr {key}: a={attrs_a[key]} b={attrs_b[key]}")
    for key in sorted(set(attrs_a) - set(attrs_b)):
        to_confirm.append(f"attr {key} unknown in b")
    for key in sorted(set(attrs_b) - set(attrs_a)):
        to_confirm.append(f"attr {key} unknown in a")
    return basis, conflicts, to_confirm


def compare_candidates(ctx: ToolContext, args: dict, vlm: Any = None) -> ToolResult:
    """Compare two candidate groups: same-frame geometry + attribute rules.

    Deterministic and vision-free (it consumes no VLM tokens); a free-form
    ``question`` is reported as needing a model judgement rather than answered
    here. ``vlm`` is accepted for handler-signature symmetry and is unused.
    """
    args = args or {}
    resolved_a, unknown_a, type_errors_a = _resolve_candidates(ctx, args.get("candidates_a"))
    resolved_b, unknown_b, type_errors_b = _resolve_candidates(ctx, args.get("candidates_b"))
    evidence: list[EvidenceRef] = []
    for candidate in resolved_a + resolved_b:
        evidence.append(EvidenceRef("memory", str(candidate.get("id")), "candidate"))
        observation_id = candidate.get("observation_id")
        if observation_id and not any(e.ref == str(observation_id) for e in evidence):
            evidence.append(EvidenceRef("observation", str(observation_id), "candidate frame"))
    type_errors = type_errors_a + type_errors_b
    if type_errors:
        # one id of the wrong type is a typed error, not "no such candidate":
        # the answer names the expected type and lists ids of that type
        first = type_errors[0]
        data = first.to_data()
        data["mismatched"] = [error.raw for error in type_errors]
        data["available_candidate_ids"] = available_candidate_ids(ctx)
        return ToolResult(STATUS_FAILED, data, evidence)
    if unknown_a or unknown_b:
        return ToolResult(
            STATUS_EMPTY,
            {
                "reason": "unknown_candidate",
                "unknown": unknown_a + unknown_b,
                "available": list(_candidate_store(ctx))[-8:],
                "hint": "call ground_target first, or pass inline candidate dicts",
            },
            evidence,
        )
    a = _pick_candidate(resolved_a)
    b = _pick_candidate(resolved_b)
    if a is None or b is None:
        return ToolResult(
            STATUS_EMPTY,
            {"reason": "empty_candidate_group", "n_a": len(resolved_a), "n_b": len(resolved_b)},
            evidence,
        )

    question = str(args.get("question") or "").strip()
    obs_a, obs_b = a.get("observation_id"), b.get("observation_id")
    same_frame = bool(obs_a) and obs_a == obs_b
    box_a, box_b = _candidate_box(a), _candidate_box(b)
    basis, conflicts, to_confirm = [], [], []
    iou = None
    if box_a is not None and box_b is not None:
        if same_frame:
            iou = round(_iou(box_a.to_list(), box_b.to_list()), 3)
            basis.append(f"iou={iou:.3f}")
            if iou < IOU_DIFFERENT:
                conflicts.append(f"iou={iou:.3f}<{IOU_DIFFERENT}")
        else:
            to_confirm.append("cross_frame_geometry_not_comparable: reobserve both from one pose")
    else:
        to_confirm.append("missing_box: no geometry available")

    attr_basis, attr_conflicts, attr_confirm = _compare_attrs(
        a.get("attrs") or {}, b.get("attrs") or {}
    )
    basis.extend(attr_basis)
    conflicts.extend(attr_conflicts)
    to_confirm.extend(attr_confirm)

    for candidate in (a, b):
        confidence = candidate.get("confidence")
        if confidence is not None and confidence < LOW_CONFIDENCE:
            to_confirm.append(f"low_confidence:{candidate.get('id')}={confidence}")
    if question:
        to_confirm.append(f"question_requires_vlm_judgement: {question}")

    if attr_conflicts or (iou is not None and iou < IOU_DIFFERENT):
        verdict = "different"
    elif same_frame and iou is not None and iou >= IOU_SAME and not attr_conflicts:
        verdict = "same"
    else:
        verdict = "uncertain"
    if verdict == "uncertain":
        to_confirm.append("reobserve this candidate from a second viewpoint")

    data = {
        "candidates_a": [c.get("id") for c in resolved_a],
        "candidates_b": [c.get("id") for c in resolved_b],
        "picked_a": a.get("id"),
        "picked_b": b.get("id"),
        "same_frame": same_frame,
        "iou": iou,
        "verdict": verdict,
        "match_basis": basis,
        "conflicts": conflicts,
        "to_confirm": to_confirm,
        "attrs_a": a.get("attrs") or {},
        "attrs_b": b.get("attrs") or {},
        "confidence_a": a.get("confidence"),
        "confidence_b": b.get("confidence"),
        "method": "geometry+attributes",
        "question": question or None,
        "box_format": BOX_FORMAT_DOC,
    }
    return ToolResult(STATUS_OK, data, evidence)


# --------------------------------------------------------------------------
# query_memory
# --------------------------------------------------------------------------

# model-facing fields of one memory hit; a hit is a pointer to evidence, so the
# summary carries the id, the pose it was captured at, the heading and the image
# reference - a recalled frame the model cannot address or navigate back to is
# not evidence
_POSITION_KEYS = ("position_flu", "pose", "position", "world_flu", "target_flu")
_SUMMARY_TEXTS = ("hyp_id", "id", "kind", "status", "content", "note", "target_name", "event")
_SUMMARY_LISTS = ("events", "labels", "reasons")
_SUMMARY_NUMBERS = ("step", "t", "time_s", "score")
_SUMMARY_LIST_CAP = 8
_SUMMARY_TEXT_CAP = 240


def _summary_scalar(value: Any) -> Any:
    """One JSON-plain scalar: numpy unwrapped, floats rounded, text bounded."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (int, float, np.floating)):
        out = float(value)
        return round(out, 3) if math.isfinite(out) else None
    if isinstance(value, str):
        text = value.strip()
        return text[:_SUMMARY_TEXT_CAP] if text else None
    return None


def _summary_numbers(value: Any) -> list | None:
    """Bounded list of scalars (None when the value is not a plain sequence).

    Nested structures are dropped rather than serialized: the hit list is a
    pointer to evidence, not a dump of the archive entry.
    """
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    elif torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1).tolist()
    if not isinstance(value, (list, tuple)):
        return None
    out = []
    for item in list(value)[:_SUMMARY_LIST_CAP]:
        scalar = _summary_scalar(item)
        if scalar is None:
            if item is not None:
                return None
            continue
        out.append(scalar)
    return out


def _summary_position(value: Any) -> list | None:
    """[x, y, z] of a hit, wherever the record keeps it."""
    numbers = _summary_numbers(value)
    return numbers[:3] if numbers and len(numbers) >= 3 else None


def _hit_summary(hit: dict) -> dict:
    """The model-facing subset of one memory hit.

    The id is kept under both spellings (``memory.ObservationLog`` entries are
    keyed ``obs_id`` while every tool result and the model's own vocabulary say
    ``observation_id``), and the capture pose is serialized to plain numbers:
    an id plus a pose is what turns a recalled frame back into a leg.
    A position nested in a hypothesis' ``data`` payload is lifted to
    ``position_flu`` for the same reason.

    A hit says *what it is* (``evidence_kind`` + ``source`` + ``quality``): a
    sighting (the frame showed the thing, with the candidate and the basis that
    were accepted), a scoped null result (with its ``visibility`` boundary), or
    a frame captured while the task looked for something (``goal_frame`` -
    labelled as such, never presented as proof of having seen it).
    """
    entry: dict[str, Any] = {}
    obs_id = _summary_scalar(hit.get("observation_id")) or _summary_scalar(hit.get("obs_id"))
    if obs_id is not None:
        entry["observation_id"] = obs_id
        entry["obs_id"] = obs_id
    for key in _SUMMARY_TEXTS:
        value = _summary_scalar(hit.get(key))
        if value is not None:
            entry[key] = value
    for key in _SUMMARY_LISTS:
        value = _summary_numbers(hit.get(key))
        if value:
            entry[key] = value
    data = hit.get("data") if isinstance(hit.get("data"), dict) else {}
    for key in _POSITION_KEYS:
        position = _summary_position(hit.get(key)) or _summary_position(data.get(key))
        if position is not None:
            entry["position_flu"] = position
            break
    yaw_deg = _summary_scalar(hit.get("yaw_deg"))
    if yaw_deg is None:
        yaw_rad = _summary_scalar(hit.get("yaw_rad"))
        yaw_deg = None if yaw_rad is None else round(math.degrees(yaw_rad), 1)
    if yaw_deg is not None:
        entry["yaw_deg"] = yaw_deg
    for key in _SUMMARY_NUMBERS:
        value = _summary_scalar(hit.get(key))
        if value is not None:
            entry[key] = value
    image = _summary_scalar(hit.get("image")) or _summary_scalar(hit.get("image_path"))
    if image is not None:
        entry["image"] = image
    # the evidence contract of the hit: what it is, where it came from, how
    # usable it is - and the sighting/null-result detail when it carries one
    kind = str(hit.get("evidence_kind") or EVIDENCE_KIND_OBSERVATION)
    if kind == EVIDENCE_KIND_OBSERVATION and hit.get("match") == "goal_frame":
        kind = EVIDENCE_KIND_GOAL_FRAME
    entry["evidence_kind"] = kind
    entry["source"] = f"episodic:{kind}"
    sighting = hit.get("sighting")
    if isinstance(sighting, Mapping):
        entry["sighting"] = _memory_jsonable(dict(sighting))
    if isinstance(hit.get("sightings"), list) and len(hit["sightings"]) > 1:
        entry["n_sightings"] = len(hit["sightings"])
        # every sighting of the frame: two grounded candidates on one frame are
        # two things to compare, and dropping the extras would hide the second
        entry["sightings"] = [
            _memory_jsonable(dict(item)) for item in hit["sightings"][:4]
            if isinstance(item, Mapping)
        ]
    negative = hit.get("negative")
    if isinstance(negative, Mapping):
        entry["negative"] = _memory_jsonable(dict(negative))
    visibility = hit.get("visibility")
    if isinstance(visibility, Mapping):
        entry["visibility"] = _memory_jsonable(dict(visibility))
    if isinstance(hit.get("scope_note"), str) and hit["scope_note"]:
        entry["scope_note"] = _summary_scalar(hit["scope_note"])
    if hit.get("match"):
        entry["match"] = str(hit["match"])
    if _summary_scalar(hit.get("repeat_count")):
        entry["repeat_count"] = int(hit["repeat_count"])
    entry["quality"] = _hit_quality(entry)
    return entry


#: a hit whose frame is at the drone's own position is not a flight
SAME_POSITION_TOL_M = 0.5
#: how many hits the evidence pack lists with their image and pose
MAX_PACK_IMAGES = 4
#: how many uses of the evidence are suggested (best first)
MAX_SUGGESTIONS = 3
#: confidence thresholds of the coarse quality level of a hit
STRONG_CONFIDENCE = 0.7
WEAK_CONFIDENCE = 0.4
#: strength of a hit: frame stored / intrinsics known / how confident the
#: sighting was - a frame whose pixels are gone cannot settle a comparison
_QUALITY_LEVELS = ("strong", "moderate", "weak")

#: what a retrieval says when the store has no episodic evidence layer (the
#: ``memory_spatial_only`` arm of scripts/run_vlm_nav_m2_memory): its hits are
#: capture records - pose, heading, step - and must not be read as evidence
EPISODIC_OFF_NOTE = (
    "the episodic evidence layer of this memory store is off: no frame, no "
    "sighting record and no scoped null result was written, so none can come "
    "back - a hit is where the task was (pose, heading, step), and the places "
    "and the task state come from the spatial and working layers"
)


def _episodic_layer_off(memory: Any) -> bool:
    """True when the store itself reports no episodic evidence layer.

    Read off the store (``episodic_memory``), so a duck-typed or fallback
    memory that does not declare the layer is not accused of missing it.
    """
    value = getattr(memory, "episodic_memory", None)
    return value is not None and not bool(value)


def _hit_quality(hit: dict) -> dict:
    """How much a hit can carry: frame stored, intrinsics known, confidence."""
    obs_id = hit.get("observation_id") or hit.get("obs_id")
    image = hit.get("image") or hit.get("image_path")
    frame = "stored" if image else ("missing" if obs_id else "none")
    intrinsics = hit.get("intrinsics") if isinstance(hit.get("intrinsics"), Mapping) else {}
    sighting = hit.get("sighting") if isinstance(hit.get("sighting"), Mapping) else {}
    confidence = _summary_scalar(sighting.get("confidence"))
    if confidence is None:
        confidence = _summary_scalar(hit.get("confidence"))
    if frame != "stored" or (confidence is not None and confidence < WEAK_CONFIDENCE):
        level = "weak"
    elif confidence is not None and confidence >= STRONG_CONFIDENCE:
        level = "strong"
    else:
        level = "moderate"
    return {
        "frame": frame,
        "intrinsics": bool(intrinsics.get("hfov_deg")),
        "confidence": confidence,
        "level": level,
    }


def _goal_target_of(ctx: ToolContext) -> str:
    """The instructed target of this episode (what the task is looking for)."""
    extra = _extra(ctx)
    for key in ("target_name", "goal_target", "instruction_target"):
        value = extra.get(key)
        if value:
            return str(value)
    target = getattr(getattr(ctx, "world", None), "target", None)
    return str(getattr(target, "name", "") or "") if target is not None else ""


def _hit_candidates(hit: dict) -> list[str]:
    """Every candidate id a hit's sightings name (primary and the extras)."""
    candidates: list[str] = []
    items = []
    if isinstance(hit.get("sighting"), Mapping):
        items.append(hit["sighting"])
    if isinstance(hit.get("sightings"), list):
        items.extend(item for item in hit["sightings"] if isinstance(item, Mapping))
    for item in items:
        candidate = item.get("candidate")
        if candidate and str(candidate) not in candidates:
            candidates.append(str(candidate))
    return candidates


def _uses_for_hit(hit: dict, uses: list[dict]) -> dict | None:
    """The first suggested use that names this hit's own evidence."""
    obs_id = hit.get("observation_id")
    candidates = _hit_candidates(hit)
    for use in uses:
        values = list((use.get("args") or {}).values())
        if obs_id is not None and obs_id in values:
            return dict(use)
        if any(candidate in values for candidate in candidates):
            return dict(use)
    return None


def _suggested_use(ctx: ToolContext, hits: list[dict], goal_target: str) -> list[dict]:
    """The uses of the retrieved evidence, best first.

    Comparison first, then verification, then a real move to what was seen -
    and only then the plain "localize it in this frame" fallback: a retrieval
    exists to settle a question with evidence the drone already has, and the
    flight back to the place (when it is a real move) is the *secondary*
    option, never the first suggestion.
    """
    uses: list[dict] = []
    current = _current_position(ctx.world)
    frame_hits = [hit for hit in hits if hit.get("observation_id")]
    sighting_hits = [
        hit for hit in frame_hits if hit.get("evidence_kind") == EVIDENCE_KIND_SIGHTING
    ]
    candidate_ids: list[str] = []
    for hit in sighting_hits:
        for candidate in _hit_candidates(hit):
            if candidate not in candidate_ids:
                candidate_ids.append(candidate)
    if len(candidate_ids) >= 2:
        uses.append({
            "tool": "compare_candidates",
            "args": {"candidates_a": candidate_ids[0], "candidates_b": candidate_ids[1]},
            "why": (
                f"the recalled frames hold {len(candidate_ids)} candidates "
                f"({', '.join(candidate_ids[:3])}): compare them before choosing one - "
                "the same instance seen twice or two different objects?"
            ),
            "layer": "episodic",
        })
    if sighting_hits:
        best = sighting_hits[0]
        basis = str(((best.get("sighting") or {}).get("basis")) or "")[:70]
        uses.append({
            "tool": "verify_goal",
            "args": {"observation_id": best["observation_id"]},
            "why": (
                f"{best['observation_id']} is a sighting ({basis}): check the goal "
                "conditions against the evidence you already have before moving again"
            ),
            "layer": "episodic",
        })
    for hit in sighting_hits:
        candidates = _hit_candidates(hit)
        candidate = candidates[0] if candidates else None
        position = _summary_position(hit.get("position_flu"))
        if not candidate or position is None or current is None:
            continue
        distance = float(np.linalg.norm(np.asarray(position) - np.asarray(current)))
        if distance <= SAME_POSITION_TOL_M:
            continue  # nothing to add: the drone is standing on it
        uses.append({
            "tool": "approach_candidate",
            "args": {"candidate_id": str(candidate)},
            "why": (
                f"{candidate} was located {distance:.1f} m from here (seen in "
                f"{hit['observation_id']}): a real move to the thing itself, not a re-look"
            ),
            "layer": "episodic",
            "distance_m": round(distance, 3),
        })
        break
    if frame_hits:
        first = frame_hits[0]
        uses.append({
            "tool": "ground_target",
            "args": {
                "instruction": goal_target or "<what you are looking for>",
                "observation_id": first["observation_id"],
            },
            "why": (
                "localize in the recalled frame: ground the instruction here (or magnify "
                "the view with inspect_region) and compare what comes back with the "
                "candidates you already have"
            ),
            "layer": "episodic",
        })
    return uses[:MAX_SUGGESTIONS]


def _secondary_recall(ctx: ToolContext, hits: list[dict]) -> dict:
    """The way back to a recalled place: secondary, reasoned - and never underfoot.

    Flying back is only worth it when the drone is somewhere else and the frame
    is worth standing in front of again, so the suggestion carries the three
    reasons that decide it (current position, evidence quality, expected
    benefit). When the recalled pose *is* the current position the suggestion
    is withheld outright: that loop (``return_to_observation`` to where the
    drone already hovers) was measured in the real runs and moved nothing.
    """
    current = _current_position(ctx.world)
    best = next(
        (hit for hit in hits if hit.get("position_flu") and hit.get("observation_id")), None
    )
    if best is None:
        return {
            "withheld": True,
            "reason": (
                "no hit carries both a pose and an observation id: there is no place "
                "to fly back to"
            ),
        }
    position = _summary_position(best["position_flu"])
    distance = None
    if current is not None:
        distance = float(np.linalg.norm(np.asarray(position) - np.asarray(current)))
        if distance <= SAME_POSITION_TOL_M:
            return {
                "withheld": True,
                "reason": (
                    f"you are already at this viewpoint (current position "
                    f"{[round(float(v), 2) for v in current]}, {distance:.2f} m from the "
                    f"recalled pose of {best['observation_id']}): flying back would not "
                    "move the drone - work with the evidence you have"
                ),
                "distance_m": round(distance, 3),
                "observation_id": best["observation_id"],
            }
    quality = best.get("quality") if isinstance(best.get("quality"), Mapping) else {}
    level = quality.get("level")
    benefit = (
        "standing there again puts the recalled frame's content back in view"
        if best.get("evidence_kind") == EVIDENCE_KIND_SIGHTING
        else "the frame is the only record of that place: re-observing from there is a fresh look at it"
    )
    why = []
    if distance is not None:
        why.append(
            f"the drone is {distance:.2f} m away from the pose this evidence was "
            f"captured at (current position {[round(float(v), 2) for v in current]})"
        )
    else:
        why.append("the drone's current position is unknown, so the distance to the recalled pose cannot be checked")
    confidence = quality.get("confidence")
    why.append(
        f"evidence quality: {level or 'unknown'}"
        + (f" (sighting confidence {confidence})" if confidence is not None else "")
    )
    why.append(f"expected benefit: {benefit}")
    return {
        "withheld": False,
        "fly_back": {
            "tool": "return_to_observation",
            "args": {"observation_id": best["observation_id"], "restore_yaw": True},
            "distance_m": None if distance is None else round(distance, 3),
            "evidence_quality": level,
            "expected_benefit": benefit,
            "why": why,
        },
    }


def _evidence_pack(hits: list[dict]) -> dict:
    """Images, poses, quality and source of the retrieval, in one block."""
    images = []
    quality_counts = dict.fromkeys(_QUALITY_LEVELS, 0)
    for hit in hits[:MAX_PACK_IMAGES]:
        quality = hit.get("quality") if isinstance(hit.get("quality"), Mapping) else {}
        level = str(quality.get("level") or "moderate")
        quality_counts[level] = quality_counts.get(level, 0) + 1
        images.append(
            {
                "observation_id": hit.get("observation_id"),
                "image": hit.get("image"),
                "position_flu": hit.get("position_flu"),
                "yaw_deg": hit.get("yaw_deg"),
                "evidence_kind": hit.get("evidence_kind"),
                "source": hit.get("source"),
                "quality": dict(quality),
            }
        )
    kinds = [hit.get("evidence_kind") for hit in hits if hit.get("evidence_kind")]
    note = (
        f"{len(hits)} hit(s): "
        + ", ".join(f"{kinds.count(kind)} {kind}" for kind in dict.fromkeys(kinds))
        + "; a sighting is a frame that showed the target, a goal_frame only means the "
        "task was looking for it, a negative is a null result scoped to one view"
    )
    return {"images": images, "quality": quality_counts, "n_hits": len(hits), "note": note}


def _places_of(ctx: ToolContext, query, limit: int = 3) -> list[dict] | None:
    """Spatial-memory places matching the query (None when the layer is absent).

    A memory store that has the layer but was built with ``spatial_memory=False``
    answers with an empty list: the layer exists and is switched off, which is
    not the same as a store that has no such layer at all.
    """
    memory = getattr(ctx, "memory", None)
    if memory is None or not hasattr(memory, "spatial"):
        return None
    spatial = getattr(memory, "spatial", None)
    fn = getattr(spatial, "query", None) if spatial is not None else None
    if not callable(fn):
        return []
    spec = query if isinstance(query, Mapping) else {"text": query}
    text = spec.get("text") or spec.get("target") or ""
    limit = max(1, min(int(limit), 8))
    try:
        places = fn(text, limit=limit) if text else fn({}, limit=limit)
    except Exception:  # noqa: BLE001 - the spatial layer is optional
        return []
    return [_memory_jsonable(place) for place in (places or []) if isinstance(place, Mapping)]


def _working_state(ctx: ToolContext, limit: int = 3) -> dict | None:
    """The task's current state (subgoal, open questions, failures, candidates)."""
    memory = getattr(ctx, "memory", None)
    working = getattr(memory, "working", None) if memory is not None else None
    snapshot = getattr(working, "snapshot", None)
    if not callable(snapshot):
        return None
    try:
        state = snapshot()
    except Exception:  # noqa: BLE001 - the working layer is optional
        return None
    if not isinstance(state, Mapping):
        return None
    return _memory_jsonable(
        {
            "subgoal": state.get("subgoal"),
            "open_questions": list(state.get("open_questions") or [])[:limit],
            "failures": list(state.get("failures") or [])[:limit],
            "candidates": list(state.get("candidates") or [])[:limit],
            "counts": state.get("counts"),
        }
    )


def query_memory(ctx: ToolContext, args: dict, vlm: Any = None) -> ToolResult:
    """Retrieve past evidence for a question (goal.md: "where did I see X?").

    Delegates to ``memory.query`` (ObservationLog / HypothesisStore / the
    spatial places) and returns an *evidence pack* - image references, poses,
    the quality of each hit and where it came from - plus the uses of that
    evidence, best first (``compare_candidates`` / ``verify_goal`` / a real move
    to a located candidate / localizing it in the recalled frame). The flight
    back to the recalled place is the *secondary* option, named with its reasons
    (current position, evidence quality, expected benefit) - and withheld
    outright when the recalled pose is the one the drone already hovers at,
    because that loop moved nothing in the real runs.

    A hit says what it is: ``sighting`` (the frame really showed the thing, with
    the candidate and the basis it was accepted on), ``negative`` (a null result
    *scoped* to the viewpoint, heading and field of view it was gathered under),
    or ``goal_frame`` (captured while the task looked for the target - a place
    to look again, never proof the target was seen). A store without its
    episodic evidence layer returns none of those: its hits are capture records
    (pose, heading, step) and the pack carries the spatial places and the task
    state, with ``evidence_note`` saying which layer is off.
    """
    memory = getattr(ctx, "memory", None)
    if memory is None:
        return ToolResult(STATUS_EMPTY, {"reason": "no_memory"}, [])
    fn = getattr(memory, "query", None)
    if not callable(fn):
        return ToolResult(STATUS_EMPTY, {"reason": "memory_query_unsupported"}, [])
    query = args.get("query", args.get("q", args.get("question")))
    filters = {
        k: args[k]
        for k in ("target", "goal_target", "sighting", "negative", "event", "point",
                  "bearing_deg", "since_step", "before_step")
        if k in args
    }
    if isinstance(query, str) and filters:
        query = {"text": query, **filters}
    elif query is None and filters:
        query = filters
    limit = int(args.get("limit") or 5)
    try:
        hits = fn(query, limit=limit) if query is not None else fn({}, limit=limit)
    except Exception as exc:  # noqa: BLE001 - memory is optional
        return ToolResult(STATUS_FAILED, {"error": f"{type(exc).__name__}: {exc}"}, [])
    hits = [h for h in (hits or []) if isinstance(h, dict)]
    evidence: list[EvidenceRef] = []
    summary_hits = []
    for hit in hits:
        for ref in hit.get("evidence") or ():
            if isinstance(ref, EvidenceRef):
                evidence.append(ref)
            elif isinstance(ref, dict) and ref.get("ref"):
                evidence.append(EvidenceRef(str(ref.get("kind", "observation")),
                                            str(ref["ref"]), str(ref.get("note", ""))))
        summary_hits.append(_hit_summary(hit))
    if not hits:
        return ToolResult(STATUS_EMPTY, _no_match_data(ctx, memory, query), [])
    goal_target = _goal_target_of(ctx)
    # what the evidence is for, best use first; then what each hit can carry
    suggestions = _suggested_use(ctx, summary_hits, goal_target)
    for summary in summary_hits:
        use = _uses_for_hit(summary, suggestions)
        if use is not None:
            summary["use"] = use
    if isinstance(hits[0].get("intrinsics"), Mapping):
        # the intrinsics of the recalled frames: what their pixels project with
        for summary, hit in zip(summary_hits, hits):
            if summary.get("observation_id") and isinstance(hit.get("intrinsics"), Mapping):
                summary["intrinsics"] = {
                    str(key): value for key, value in hit["intrinsics"].items()
                    if isinstance(value, (int, float, str))
                }
    secondary = _secondary_recall(ctx, summary_hits)
    data: dict[str, Any] = {
        "count": len(hits),
        "n_hits": len(hits),
        "query": query,
        "hits": summary_hits,
        "evidence_pack": _evidence_pack(summary_hits),
        "suggested_use": suggestions,
        "secondary": secondary,
        "n_sighting_hits": sum(
            1 for hit in summary_hits if hit.get("evidence_kind") == EVIDENCE_KIND_SIGHTING
        ),
        "n_goal_frame_hits": sum(
            1 for hit in summary_hits if hit.get("evidence_kind") == EVIDENCE_KIND_GOAL_FRAME
        ),
    }
    if data["n_sighting_hits"] == 0:
        # a store without its episodic layer has no sighting to return, and the
        # hits it does answer with must not be read as one
        data["evidence_note"] = EPISODIC_OFF_NOTE if _episodic_layer_off(memory) else (
            "no verified sighting of the target in these hits: they are frames captured "
            "while the task looked for it (evidence_kind=goal_frame) - treat them as places "
            "to look again, not as proof the target was seen; a sighting comes from "
            "ground_target (or a model pointing at the object) and carries the candidate "
            "and the basis it was accepted on"
        )
    if suggestions:
        data["next_action"] = {
            "tool": suggestions[0]["tool"],
            "args": dict(suggestions[0]["args"]),
        }
        hint = f"use this evidence: {suggestions[0]['why']}"
        if secondary.get("withheld"):
            hint += f"; do not fly to the recalled place: {secondary['reason']}"
        elif secondary.get("fly_back"):
            fly = secondary["fly_back"]
            hint += (
                f"; if you would rather be there in person, the secondary option is "
                f"return_to_observation with observation_id="
                f"\"{fly['args']['observation_id']}\" ({fly['why'][0]})"
            )
        data["hint"] = hint
    # the other two layers: the places the spatial memory holds and the task
    # state the working memory holds (absent when the layers are switched off)
    places = _places_of(ctx, query)
    if places is not None:
        data["places"] = places
    working = _working_state(ctx)
    if working is not None:
        data["working"] = working
    layers = _MemoryBridge(ctx.memory).layer_status()
    if layers:
        data["layers"] = layers
    # a recalled hit is an observation id of this episode: adopt it into the
    # repository so `store.get(id)` answers with the image, the pose and the
    # intrinsics the hit was recorded with (the hit list and the store must not
    # be two id spaces)
    for hit, summary in zip(hits, summary_hits):
        _adopt_recalled_hit(ctx, hit, summary)
    return ToolResult(STATUS_OK, data, evidence[:12])


def _no_match_data(ctx: ToolContext, memory: Any, query) -> dict:
    """The answer to a miss: what the archive does hold, and what was checked.

    A miss must not send the model looking for an id it never got, and it must
    not re-check a viewpoint that already answered *this* question with nothing -
    so the recent ids and the scoped null results come along.
    """
    spec = query if isinstance(query, Mapping) else {"text": query}
    text = spec.get("text") or spec.get("target") or ""
    recent_fn = getattr(memory, "recent", None)
    recent_ids = []
    if callable(recent_fn):
        try:
            recent_ids = [
                summary["observation_id"]
                for summary in (_hit_summary(h) for h in recent_fn(3) if isinstance(h, dict))
                if summary.get("observation_id")
            ]
        except Exception:  # noqa: BLE001 - memory is optional
            recent_ids = []
    checked = []
    fn = getattr(memory, "query", None)
    if callable(fn):
        # first the null results about *this* question, then the most recent ones
        negatives = []
        for request in ({"negative": True, "text": text, "limit": 3},
                        {"negative": True, "limit": 2}):
            try:
                negatives = fn(request, limit=3)
            except Exception:  # noqa: BLE001 - a store without negatives answers nothing
                negatives = []
            if negatives:
                break
        for hit in negatives or []:
            if not isinstance(hit, dict):
                continue
            summary = _hit_summary(hit)
            if summary.get("scope_note"):
                checked.append(
                    {
                        "observation_id": summary.get("observation_id"),
                        "target": (summary.get("negative") or {}).get("target"),
                        "reason": (summary.get("negative") or {}).get("reason"),
                        "scope_note": summary["scope_note"],
                    }
                )
    data = {
        "reason": "no_match",
        "query": query,
        "count": 0,
        "n_hits": 0,
        "recent_observation_ids": recent_ids,
        "hint": (
            "no stored evidence matches; look around (observe) instead of "
            "navigating to an id you do not have a result for"
        ),
    }
    if _episodic_layer_off(memory):
        # a store without the layer cannot have "evidence that does not match":
        # it never stored one, and the miss must not be read as "looked, found
        # nothing" either (the coverage of the spatial layer is what says that)
        data["evidence_note"] = EPISODIC_OFF_NOTE
    if checked:
        data["checked_without_result"] = checked
        data["hint"] += (
            "; these viewpoints were already searched for this target without a result, "
            "so a different viewpoint, heading or field of view is what is missing"
        )
    return data


def _adopt_recalled_hit(ctx: ToolContext, hit: dict, summary: dict) -> None:
    """Register one recalled memory hit in the observation repository.

    Everything the archive knows is passed on - id, pose, heading, image path and
    intrinsics - so the id the model just read in a hit resolves everywhere else
    afterwards (``store.get`` -> image + pose + intrinsics). A record of the same
    id is updated in place rather than duplicated, and nothing is invented: a hit
    without a pose keeps the pose already stored (or stays pose-unknown), and the
    intrinsics fall back to the world/context resolution only when neither the
    entry nor the stored record declares them.
    """
    obs_id = summary.get("observation_id")
    if not obs_id:
        return
    if observation_store(ctx, create=False) is None:
        return
    raw = hit.get("intrinsics")
    intrinsics = dict(raw) if isinstance(raw, dict) and raw.get("width") else None
    extra = hit.get("extra") if isinstance(hit.get("extra"), dict) else {}
    if intrinsics is None and isinstance(extra.get("intrinsics"), dict):
        intrinsics = dict(extra["intrinsics"])
    try:
        register_observation(
            ctx,
            None,
            kind=kind_for_label(hit.get("kind")),
            step=_as_int(hit.get("step"), 0),
            timestamp=_as_float(hit.get("t"), None),
            image_path=str(hit.get("image") or hit.get("image_path") or ""),
            obs_id=str(obs_id),
            pose_flu=summary.get("position_flu"),
            yaw_deg=summary.get("yaw_deg"),
            intrinsics=intrinsics,
            note=str(hit.get("note") or "recalled from memory")[:120],
        )
    except Exception:  # noqa: BLE001 - the hit list is still returned
        return


def register_perception_tools(registry: ToolRegistry, vlm: Any = None) -> None:
    """Register ``observe`` / ``inspect_region`` / ``ground_target`` /
    ``compare_candidates`` / ``query_memory`` on ``registry``.

    ``vlm`` is the optional grounding backend used by ``ground_target`` (any
    object with ``decide(frames, system_prompt, user_prompt, context)``, e.g.
    ``vlm.VLMBackend``). When it is None the tool reports an ``empty`` result
    with ``reason=no_vlm_backend`` instead of guessing.

    The search skill (``vlm_nav.skills_search``, tool ``search_target``) is
    registered on the same registry: it is a routine *over* these tools
    (sweep -> pick a view -> ground -> magnify -> propose a viewpoint -> move),
    not a second implementation of them.
    """
    registry.register(
        ToolSpec(
            name="observe",
            description=(
                "Capture first-person evidence: mode=lookaround does an in-place "
                "360 deg scan (n_views frames), mode=forward (the default) renders "
                "the current heading (optionally offset by yaw_deg), mode=pitch_down "
                "looks down for low targets. Returns observation ids, pose and time, "
                "plus the visual context of the capture: view_index (which way each "
                "returned view looked), coverage (which 45 deg sectors a level view "
                "checked, which are unchecked, which view is worth a second look) and "
                "expansion (the inspect_region calls that pull one view's pixels back). "
                "The drone's pose is restored afterwards."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": list(OBSERVE_MODES),
                        "default": DEFAULT_OBSERVE_MODE,
                        "description": (
                            "lookaround: 360 deg sweep; forward (default): one frame at "
                            "the current heading; pitch_down: look down"
                        ),
                    },
                    "yaw_deg": {"type": "number", "description": "view offset from the current heading"},
                    "pitch_deg": {"type": "number", "description": "downward tilt for pitched views"},
                    "n_views": {"type": "integer", "description": "lookaround frames (default 8)"},
                    "spread_deg": {"type": "number", "description": "narrow the scan to this arc"},
                },
                "required": [],
            },
            handler=observe,
            side_effects=False,
        )
    )
    registry.register(
        ToolSpec(
            name="inspect_region",
            description=(
                "Magnify a region of an earlier observation. region is a corner "
                "box [x1, y1, x2, y2] in normalized 0-1000 image coordinates "
                "(x1, y1 = top-left, x2, y2 = bottom-right); the legacy size form "
                "[u, v, w, h] is also accepted and reported back as "
                "region_format. Returns a crop image reference and, when the "
                "observation was captured at the current pose, a depth reading "
                "in meters for that region."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "observation_id": {
                        "type": "string",
                        "description": "an obs_N id a tool returned (a recorded view)",
                    },
                    "region": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": BOX_FORMAT_DOC,
                    },
                    "box_format": {
                        "type": "string",
                        "enum": [BOX_FORMAT_XYXY, BOX_FORMAT_UVWH],
                        "description": "optional: how the region is written (default xyxy)",
                    },
                },
                "required": ["observation_id", "region"],
            },
            handler=inspect_region,
            side_effects=False,
        )
    )
    registry.register(
        ToolSpec(
            name="ground_target",
            description=(
                "Ground an instruction to target candidates in one observation "
                "(the current view when observation_id is omitted). Returns "
                "candidates whose box is a corner box [x1, y1, x2, y2] in "
                "normalized 0-1000 image coordinates (box_format says so, "
                "box_format_used which spelling the grounding model wrote), plus "
                "attrs/confidence/uncertainty and a stable candidate id (cand_N) "
                "usable by compare_candidates and as the candidate_id of "
                "approach_candidate / the target of check_path/verify_goal. "
                "Each candidate carries position_flu when "
                "its range is known (position_source depth/model_distance) and "
                "otherwise a ray (position_source ray) or the reason no position "
                "could be derived - never a guessed point."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "instruction": {"type": "string"},
                    "observation_id": {
                        "type": "string",
                        "description": "an obs_N id a tool returned (default: capture one now)",
                    },
                },
                "required": ["instruction"],
            },
            handler=lambda ctx, args: ground_target(ctx, args, vlm=vlm),
            side_effects=False,
        )
    )
    registry.register(
        ToolSpec(
            name="compare_candidates",
            description=(
                "Compare two candidate groups (cand_N ids from ground_target, or "
                "inline candidate dicts) using same-frame box geometry (IoU of "
                "their [x1, y1, x2, y2] normalized boxes) and attributes. Returns "
                "match_basis, conflicts, to_confirm and a same/different/"
                "uncertain verdict. An obs_N or vp_N id here is a type error, not "
                "an unknown candidate."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "candidates_a": {
                        "type": ["array", "string", "object"],
                        "description": "cand_N id(s) or inline candidate dict(s)",
                    },
                    "candidates_b": {"type": ["array", "string", "object"]},
                    "question": {"type": "string", "description": "free-form question to flag for the model"},
                },
                "required": ["candidates_a", "candidates_b"],
            },
            handler=lambda ctx, args: compare_candidates(ctx, args, vlm=vlm),
            side_effects=False,
        )
    )
    registry.register(
        ToolSpec(
            name="query_memory",
            description=(
                "Search past evidence by question or filters (target, event, time "
                "window, world point/bearing, and the evidence kind: sighting / "
                "goal_target / negative). Returns an evidence pack - hits with "
                "observation_id, position_flu, yaw_deg, evidence_kind, quality and "
                "image reference - plus suggested_use (compare_candidates / "
                "verify_goal / approach_candidate / localize in the recalled frame). "
                "A target query matches what the frames really showed (sightings); "
                "frames captured while looking for the target come back labelled "
                "evidence_kind=goal_frame, and a null result comes back with the "
                "visibility it was gathered under. The flight back to a recalled "
                "place is the secondary option (secondary.fly_back) with its reasons, "
                "and it is withheld when that place is where the drone already is. "
                "A memory built without its episodic evidence layer answers with "
                "capture records only - pose, heading, step, no image, no sighting - "
                "and says so in evidence_note."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "free text or target name"},
                    "target": {
                        "type": "string",
                        "description": "what the frame showed (sightings first)",
                    },
                    "goal_target": {
                        "type": "string",
                        "description": "frames captured while looking for this target",
                    },
                    "sighting": {
                        "type": ["boolean", "string"],
                        "description": "true: only frames that saw something; a string: match it",
                    },
                    "negative": {
                        "type": "boolean",
                        "description": "true: only scoped null results (checked without a result)",
                    },
                    "event": {"type": "string"},
                    "point": {"type": "array", "items": {"type": "number"}},
                    "bearing_deg": {"type": "number"},
                    "since_step": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
            },
            handler=lambda ctx, args: query_memory(ctx, args, vlm=vlm),
            side_effects=False,
        )
    )
    # the search skill rides on the tools above; importing it lazily keeps the
    # module import graph one-way (skills_search imports this module)
    from .skills_search import register_search_tools

    register_search_tools(registry, vlm=vlm)


# Generic-discovery alias: the M1 loader (``scripts/run_vlm_nav_tools.py``) and
# ``mcp_server`` probe a tuple of register-function names, so a module picked up
# by name still contributes its tools.
register_tools = register_perception_tools
