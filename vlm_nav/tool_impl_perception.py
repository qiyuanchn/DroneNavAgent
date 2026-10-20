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
import hashlib
import inspect
import json
import math
import re
import time
from typing import Any, Iterator, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .actions import body_delta_to_world
from .detectors import DetectorBackend, target_labels
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

#: ``ctx.extra`` key of the optional detector backend (``vlm_nav.detectors``).
DETECTOR_CONTEXT_KEY = "perception.detector"
#: Geometry half of the detector gate: same instance when the boxes overlap
#: this much, one mostly contains the other, or the proposal fills most of a
#: slightly larger VLM box (a box that includes the object's background).
DETECTOR_MATCH_IOU = 0.25
DETECTOR_MATCH_CONTAINMENT = 0.6
DETECTOR_PROPOSAL_IN_BOX = 0.8
DETECTOR_PROPOSAL_AREA_SHARE = 0.5
#: A detector proposal at/above this score is authoritative: a VLM box that
#: matches none of them is not the instructed instance and is not offered.
#: SAM3 is bimodal on these renders (0.03-0.1 on a textureless wall, 0.4-0.9 on
#: an object), so 0.45 separates the two populations without needing a second
#: calibration run.
DETECTOR_AUTHORITATIVE_SCORE = 0.45
#: At most this many detector-only candidates are added when the VLM reply has
#: no accepted box (a rescue, not a second candidate list).
DETECTOR_EXTRA_CANDIDATES = 2

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

#: A depth reading and the model's own distance estimate are contradictory when
#: they differ by more than this ratio (and by more than the metric floor below).
#: A huge box centre often lands on the wall behind the object (the classroom v2
#: candidates: 16-27 m of depth against a 3-4 m model estimate).
DISTANCE_CONFLICT_RATIO = 2.0
DISTANCE_CONFLICT_DIFF_M = 3.0
#: Converged support-contact samples override a model range that is more than
#: :data:`DISTANCE_CONFLICT_RATIO` away: a tight cluster of depth reads is a
#: measurement, while the model's range is the estimate that can be short. The
#: consensus needs at least three samples within this relative and absolute
#: spread; a dispersed set is not a surface and stays uncertain.
DISTANCE_CONSENSUS_MIN_SAMPLES = 3
DISTANCE_CONSENSUS_SPREAD_RATIO = 0.25
DISTANCE_CONSENSUS_SPREAD_M = 1.0
#: A box this large (fraction of the frame) or touching the frame border is not
#: trusted at its centre: the centre is very likely background, not the object.
LARGE_BOX_FRACTION = 0.5
BOX_BORDER_MARGIN = 20.0
#: A border-touching box bigger than this is not measurable at *any* pixel by
#: the ordinary centre/ground samples: the shelf/wall ambiguity (classroom v5:
#: [480,0,999,400], area 0.208) makes both the centre and the support contact
#: background. The bottom strip is scanned separately (see
#: :func:`_border_scan_sample`); when that finds no near surface either, the
#: candidate needs a dedicated second view.
BORDER_LARGE_FRACTION = 0.15
#: Bottom-edge scan of a border-large box: columns across the box width, several
#: rows near the bottom; only a contiguous run of model-consistent readings is
#: a surface (a single stray pixel is not).
BORDER_SCAN_COLUMNS = 17
BORDER_SCAN_ROW_INSETS = (0.015, 0.04, 0.09)
BORDER_SCAN_MIN_RUN = 3
BORDER_SCAN_SPREAD_M = 0.75
BORDER_SCAN_MODEL_DIFF_M = 1.5
BORDER_SCAN_MODEL_RATIO = 1.5
#: A dedicated second view turns toward the frame edge the box occupies. The
#: sign follows the observe convention (positive yaw = left / counter-clockwise)
#: and the magnitude grows with the box centre's distance from the image centre.
BORDER_SECOND_VIEW_MIN_DEG = 30.0
BORDER_SECOND_VIEW_MAX_DEG = 45.0
BORDER_SECOND_VIEW_STEP_M = 1.5

#: Tight-box retry for a large/border box. The first box can mix the target with
#: the wall behind it (classroom v5: [480, 0, 999, 400], the shelf plus a wall),
#: and asking again for the same-sized box on the same frame cannot separate
#: them. The box region is cropped, the model is asked for a box around *only*
#: the described object inside that crop, and the reply is mapped back into the
#: source frame before the candidate is measured.
TIGHTEN_TRIGGER_FRACTION = BORDER_LARGE_FRACTION
#: at most this many tight-box calls per grounding (a reply may carry several
#: large candidates; the cost of the retry is one model call each)
TIGHTEN_MAX_CANDIDATES = 2
#: padding of the crop around the detected box, normalized 0-1000 units, so the
#: crop does not cut the object's own silhouette
TIGHTEN_CROP_PAD = 15.0
#: a tight box is accepted only when it is no longer large by the trigger
#: measure, no longer touches the frame border, and has really shrunk
TIGHTEN_MAX_FRACTION = BORDER_LARGE_FRACTION
TIGHTEN_MAX_RELATIVE_AREA = 0.65
#: a tightened bearing whose model-range point lands this far from an already
#: trusted measurement of the same target (and this many degrees off the
#: trusted bearing) is not the same instance: the candidate becomes a conflict
TIGHTEN_SANITY_ANGLE_DEG = 10.0
TIGHTEN_SANITY_DISTANCE_M = 2.5


#: Two grounded views triangulate one candidate only when the rays are a real
#: stereo pair: a baseline and an angle large enough that the intersection is
#: determined, a small closest-approach residual, and a range that agrees with
#: the trusted measurement of each view.
TRIANGULATION_MIN_BASELINE_M = 1.0
#: A 5.15-deg pair in the v6 bedroom run was self-consistent (0.012 m residual)
#: and biased by 1.6 m; below this angle the depth noise dominates the baseline
#: and the intersection is not evidence.
TRIANGULATION_MIN_ANGLE_DEG = 10.0
TRIANGULATION_MAX_RESIDUAL_M = 0.75
#: The intersection must agree with a trusted single-view range within 2x: a
#: ratio is scale-free and catches a triangulation that lands on a different
#: surface even when the absolute gap looks small at long range.
TRIANGULATION_RANGE_RATIO = 2.0
TRIANGULATION_RANGE_DIFF_M = 2.5
#: Any terminal confirmation rejection is remembered: the same triangulation
#: point (or the same bearing/box fingerprint) is not re-minted by a later
#: re-grounding. v6/v7 bedroom seed37 re-created one 21.3-deg point five times
#: because only ``position_inconsistent`` used to be remembered.
TRIANGULATION_REJECTIONS_KEY = "perception.triangulation_rejections"
TRIANGULATION_REJECTION_LIMIT = 64
TRIANGULATION_REJECTION_RADIUS_M = 0.75
TRIANGULATION_REJECTION_RAY_DEG = 3.0
TRIANGULATION_REJECTION_ORIGIN_M = 0.75
TRIANGULATION_REJECTION_BOX_IOU = 0.6
#: Reasons a confirmation may end on (the memory records all of them; the tuple
#: documents the vocabulary the confirmation routine uses).
TERMINAL_REJECTION_REASONS = frozenset({
    "different_instance",
    "position_inconsistent",
    "relation_not_supported",
    "target_not_visible_from_confirmation_view",
})

#: ``distance_source`` values: how the range a candidate was placed at was chosen
DISTANCE_DEPTH_CENTER = "depth_center"
DISTANCE_DEPTH_GROUND = "depth_ground"
DISTANCE_DEPTH_CONSISTENT = "depth_consistent"
DISTANCE_DEPTH_BORDER = "depth_border_scan"
DISTANCE_MODEL = "model"
DISTANCE_CONFLICT = "conflict"
DISTANCE_TRIANGULATED = "triangulated"
#: A depth reading that may stand on its own: it is new measurement evidence
#: and is not overridden by an earlier rejection of a triangulated point.
TRUSTED_DEPTH_SOURCES = frozenset({
    DISTANCE_DEPTH_CENTER,
    DISTANCE_DEPTH_GROUND,
    DISTANCE_DEPTH_CONSISTENT,
    DISTANCE_DEPTH_BORDER,
})

#: ``position_source`` values of a candidate location
POSITION_DEPTH = "depth"
POSITION_MODEL_DISTANCE = "model_distance"
POSITION_TRIANGULATED = "triangulated"
POSITION_RAY = "ray"
POSITION_INTRINSICS_UNKNOWN = "intrinsics_unknown"
POSITION_CROP_SOURCE_UNKNOWN = "crop_source_unknown"
POSITION_NO_RANGE = "no_valid_depth"
POSITION_PITCHED = "pitched_view"
POSITION_POSE_UNKNOWN = "pose_unknown"
POSITION_NO_WORLD = "no_world"

#: One cheap retry when the first grounding reply contains no usable box: the
#: first prompt asks for attributes and distance too, and a model can answer the
#: judgement while omitting the box. This prompt asks for the box only (no new
#: frame, no crop render) so the retry costs one model call, not a capture.
BOX_ONLY_SYSTEM_PROMPT = (
    "You are the target-grounding module of a drone navigation agent. "
    "Return ONE JSON object and nothing else: "
    '{"candidates": [{"box": [x1, y1, x2, y2], "attrs": {"color": "...", '
    '"material": "..."}, "distance_m": 0.0, "size_m": 0.0, '
    '"matches_description": true, "anchor_visible": true, "confidence": 0.0}]}. '
    "box is a corner box [x1, y1, x2, y2] in normalized 0-1000 image "
    "coordinates (x1, y1 = top-left, x2, y2 = bottom-right). "
    '"attrs" are the visible attributes of the boxed object (colour, material, '
    'content); "distance_m" is the rough camera-to-object range in metres; '
    '"size_m" is the object\'s estimated largest dimension in metres; '
    '"matches_description" must be true only when the box is the instructed '
    "instance itself - not the wall, floor, ceiling, a support surface or a "
    'neighbouring object; "anchor_visible" says whether the instruction\'s '
    "relation anchor is visible in this same frame. Return the box of the "
    "target even when attributes or "
    "distance are uncertain; the downstream instance check treats a missing "
    'attribute as unverified. Return {"candidates": []} only when the target is '
    "truly not visible in the frame."
)

BOX_ONLY_USER_PROMPT = (
    "Instruction: {instruction}\n"
    "Your previous answer contained no usable bounding box. Look at the same "
    "frame again and return only the box of the most likely instance of the "
    "instruction (normalized 0-1000, corner form)."
)

#: The tight-box retry for a large/border box: the attached frame is an enlarged
#: crop of the detected region, and the only question is which part of it is the
#: object itself. Asking in the crop (instead of drawing the old box on the full
#: frame and asking again) lets the model return a *smaller* box; the reply is
#: mapped back into the source frame (see ``_map_box_from_region``).
TIGHTEN_SYSTEM_PROMPT = (
    "You are the target-grounding module of a drone navigation agent. The "
    "attached image is an enlarged crop of one detected region. Return ONE "
    "JSON object and nothing else: "
    '{"candidates": [{"box": [x1, y1, x2, y2], "confidence": 0.0}]}. '
    "box is a corner box [x1, y1, x2, y2] in this crop's normalized 0-1000 "
    "coordinates (x1, y1 = top-left, x2, y2 = bottom-right). Box ONLY the "
    "described object itself: exclude the wall, floor, ceiling and any other "
    "background or furniture that is also visible in the crop. The box must be "
    "tight around the object's visible silhouette. Return "
    '{"candidates": []} only when the object is not visible in this crop.'
)

TIGHTEN_USER_PROMPT = (
    "Instruction: {instruction}\n"
    "The attached image is an enlarged crop of a region detected as a possible "
    "instance of the instruction, but the earlier box around it also included "
    "the wall/background. Return the tight box around only the instructed "
    "object inside this crop (normalized 0-1000, corner form)."
)

GROUND_SYSTEM_PROMPT = (
    "You are the target-grounding module of a drone navigation agent. "
    "Reply with ONE JSON object and nothing else: "
    '{"candidates": [{"box": [x1, y1, x2, y2], "attrs": {"color": "..."}, '
    '"confidence": 0.0, "distance_m": 0.0, "size_m": 0.0, '
    '"matches_description": true, "match_reason": "...", '
    '"anchor_visible": true, '
    '"uncertainty": "<what is still unclear>"}]}. '
    "box is a corner box [x1, y1, x2, y2] in normalized 0-1000 image "
    "coordinates: x1, y1 = top-left corner, x2, y2 = bottom-right corner "
    "(the older size form [u, v, w, h] is also understood, but only the corner "
    "form is exact). Keep at most 3 candidates, most likely first; "
    '"distance_m" is your rough estimate of how far the box is from the camera '
    "in meters (0 when you cannot tell); "
    '"size_m" is the object\'s estimated real size (largest dimension, meters); '
    '"matches_description" must be true only when this box is the instructed '
    "instance itself - not the wall, floor, ceiling, a support surface or a "
    "neighbouring object; \"match_reason\" names the attributes and the "
    'relation that support or contradict that judgement; "anchor_visible" '
    'answers whether the instruction\'s relation anchor (the bookshelf in '
    '"the desk next to the bookshelf") is visible in this same frame; use '
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


def _draw_hint_box(
    frame: torch.Tensor, box: list[float], color: tuple[float, float, float] = (1.0, 0.0, 0.0)
) -> torch.Tensor:
    """A copy of ``frame`` with ``box`` (normalized 0-1000) drawn as an outline.

    The v6 ground retry asked the same question twice on the same pixels and
    recovered nothing in 22 tries; a visible anchor gives the model something
    to correct instead of a blind second look.
    """
    out = frame.clone() if torch.is_tensor(frame) else torch.as_tensor(frame).clone()
    if out.ndim == 2:
        out = out.unsqueeze(0)
    height, width = int(out.shape[-2]), int(out.shape[-1])
    if height <= 0 or width <= 0:
        return out
    try:
        x1, y1, x2, y2 = [float(value) for value in list(box)[:4]]
    except (TypeError, ValueError):
        return out
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    px1 = max(0, min(width - 1, int(round(x1 / 1000.0 * width))))
    px2 = max(px1 + 1, min(width, int(round(x2 / 1000.0 * width))))
    py1 = max(0, min(height - 1, int(round(y1 / 1000.0 * height))))
    py2 = max(py1 + 1, min(height, int(round(y2 / 1000.0 * height))))
    top = max(0.0, 1.0 if out.is_floating_point() else 255.0)
    sticker = torch.tensor(
        [channel * top for channel in color], dtype=out.dtype, device=out.device
    ).view(3, 1, 1)
    if out.shape[0] == 1:
        sticker = sticker[:1]
    thickness = max(1, min(3, min(width, height) // 64))
    out[:, py1:py1 + thickness, px1:px2] = sticker
    out[:, max(py1, py2 - thickness):py2, px1:px2] = sticker
    out[:, py1:py2, px1:px1 + thickness] = sticker
    out[:, py1:py2, max(px1, px2 - thickness):px2] = sticker
    return out


def _map_box_from_region(box: list[float], region: list[float]) -> list[float] | None:
    """Map a box in an enlarged crop's 0-1000 frame back into the source frame."""
    try:
        x1, y1, x2, y2 = [float(value) for value in list(box)[:4]]
        u, v, w, h = [float(value) for value in list(region)[:4]]
    except (TypeError, ValueError):
        return None
    return [
        round(u + x1 / 1000.0 * w, 3),
        round(v + y1 / 1000.0 * h, 3),
        round(u + x2 / 1000.0 * w, 3),
        round(v + y2 / 1000.0 * h, 3),
    ]


def _tighten_region(box: Box) -> list[float]:
    """The crop region of a box, with a small pad, as ``[u, v, w, h]``."""
    x1 = max(0.0, box.x1 - TIGHTEN_CROP_PAD)
    y1 = max(0.0, box.y1 - TIGHTEN_CROP_PAD)
    x2 = min(1000.0, box.x2 + TIGHTEN_CROP_PAD)
    y2 = min(1000.0, box.y2 + TIGHTEN_CROP_PAD)
    return [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)]


def _tighten_trigger(box: Box) -> bool:
    """Whether a box is large or touches the border (the tight-box retry gate)."""
    geometry = _box_geometry(box)
    return bool(
        geometry["area_fraction"] > TIGHTEN_TRIGGER_FRACTION
        or geometry["border_touching"]
    )


def _tight_box_acceptable(original: Box, tight: Box) -> tuple[bool, str]:
    """Whether a mapped tight box is a real tightening of ``original``.

    A reply that covers the whole crop, still runs into the frame border, or
    hardly shrank is not new evidence: the original box is kept (and its
    existing uncertain/second-view handling runs), rather than replacing a
    large box with a differently wrong one.
    """
    if tight.width <= 0.0 or tight.height <= 0.0:
        return False, "degenerate"
    geometry = _box_geometry(tight)
    if geometry["border_touching"]:
        return False, "still_border"
    if geometry["area_fraction"] > TIGHTEN_MAX_FRACTION:
        return False, "still_large"
    original_area = _box_geometry(original)["area_fraction"]
    if geometry["area_fraction"] > original_area * TIGHTEN_MAX_RELATIVE_AREA:
        return False, "not_tightened"
    return True, ""


def _tighten_bearing_conflict(
    ctx: ToolContext,
    record: PerceptionObservation,
    candidate: dict,
    box: Box,
    distance_m: float | None,
) -> dict | None:
    """A tightened bearing placed at the model range must not contradict a
    trusted measurement of the same target.

    The fix this implements is a *bearing* fix: the tight box's centre replaces
    a centre that sat on the wall. If the new bearing, combined with the model's
    range, lands far from a position already measured (trusted depth) or
    triangulated for the same stage/target and points more than
    :data:`TIGHTEN_SANITY_ANGLE_DEG` away from it, the tight box is probably a
    different instance or a background patch - the candidate is marked a
    conflict instead of being placed there with confidence.
    """
    if distance_m is None or not (MIN_RANGE_M < float(distance_m) <= MAX_RANGE_M):
        return None
    origin = record.position_flu
    if origin is None:
        return None
    point, _problem = _project_pixel(ctx, record, box.center, float(distance_m))
    ray = _pixel_ray(ctx, record, box.center)
    if point is None or ray is None:
        return None
    bearing = _as_float(ray.get("bearing_deg"), None)
    if bearing is None:
        return None
    store = _candidate_store(ctx)
    subtask_key = str(candidate.get("subtask_key") or "")
    instruction = str(candidate.get("instruction") or "")
    for other in store.values():
        if not isinstance(other, Mapping) or other is candidate:
            continue
        if str(other.get("id") or "") == str(candidate.get("id") or ""):
            continue
        source = str(other.get("distance_source") or "")
        if source not in TRUSTED_DEPTH_SOURCES and source != DISTANCE_TRIANGULATED:
            continue
        if other.get("distance_uncertain") is True:
            continue
        position = other.get("position_flu")
        if not isinstance(position, (list, tuple)) or len(position) < 3:
            continue
        same_stage = bool(subtask_key) and str(other.get("subtask_key") or "") == subtask_key
        if not same_stage and not _same_target_instruction(
            str(other.get("instruction") or ""), instruction
        ):
            continue
        try:
            target = np.asarray([float(value) for value in position[:3]], dtype=np.float64)
            here = np.asarray([float(value) for value in origin[:3]], dtype=np.float64)
        except (TypeError, ValueError):
            continue
        separation = float(np.linalg.norm(target - here))
        if separation <= TIGHTEN_SANITY_DISTANCE_M:
            continue
        direction = target - here
        trusted_bearing = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
        delta = abs((trusted_bearing - float(bearing) + 180.0) % 360.0 - 180.0)
        if delta <= TIGHTEN_SANITY_ANGLE_DEG:
            continue
        projected = np.asarray([float(value) for value in point[:3]], dtype=np.float64)
        gap = float(np.linalg.norm(projected - target))
        if gap <= TIGHTEN_SANITY_DISTANCE_M:
            continue
        return {
            "model_m": round(float(distance_m), 3),
            "depth_center_m": None,
            "depth_ground_m": None,
            "ratio": None,
            "difference_m": round(gap, 3),
            "samples": [],
            "resolved": False,
            "resolution": "tight_box_sanity",
            "trusted_candidate": str(other.get("id") or ""),
            "trusted_distance_source": source,
            "trusted_position_flu": list(target),
            "angle_deg": round(delta, 2),
            "separation_m": round(gap, 3),
            "why": (
                "the tightened box's bearing, placed at the model's range, points "
                "away from a trusted measurement of the same target: the tight box "
                "is probably another instance or background, not the target"
            ),
        }
    return None


def _tighten_large_candidates(
    record: PerceptionObservation,
    instruction: str,
    candidates: list[dict],
    ask,
) -> tuple[dict | None, int]:
    """Ask the model for a tight box inside each large/border candidate's crop.

    The existing box retry draws the old box on the full frame and asks for the
    same box again; this one crops the detected region and asks for a *smaller*
    box around only the described object (the shelf, not the wall). An accepted
    reply replaces ``candidate["box"]`` before the candidate is measured, so its
    centre/bottom edge determine the bearing and the range. Rejected replies
    keep the original box (the caller's uncertain/second-view handling runs) and
    are recorded with the reason.

    Returns ``(retry_record, vlm_calls)``.
    """
    attempts: list[dict] = []
    calls = 0
    for candidate in candidates:
        if len(attempts) >= TIGHTEN_MAX_CANDIDATES:
            break
        box, _reason = parse_box(candidate.get("box"), box_format=BOX_FORMAT_XYXY)
        if box is None or not _tighten_trigger(box):
            continue
        geometry = _box_geometry(box)
        region = _tighten_region(box)
        crop, _size, _upscale, _origin = _crop_frame(record.frame, region)
        info: dict = {
            "candidate": candidate.get("id"),
            "original_box": box.to_list(),
            "original_area_fraction": geometry["area_fraction"],
            "original_border_touching": geometry["border_touching"],
            "region": list(region),
            "vlm_calls": 0,
            "accepted": False,
            "reason": "",
        }
        items = None
        try:
            reply = ask(
                TIGHTEN_SYSTEM_PROMPT,
                TIGHTEN_USER_PROMPT.format(instruction=instruction),
                [crop],
            )
            info["vlm_calls"] = 1
            calls += 1
            items = _candidate_items(
                _extract_json(reply if isinstance(reply, str) else str(reply))
            )
        except Exception:  # noqa: BLE001 - a failed retry keeps the original box
            items = None
        tight = None
        if items:
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                local, _local_reason = parse_box(
                    item.get("box", item.get("bbox", item.get("region")))
                )
                if local is None:
                    continue
                mapped = _map_box_from_region(local.to_list(), region)
                if mapped is None:
                    continue
                tight, _tight_reason = parse_box(mapped, box_format=BOX_FORMAT_XYXY)
                if tight is not None:
                    break
        if tight is None:
            info["reason"] = "no_box"
        else:
            acceptable, reason = _tight_box_acceptable(box, tight)
            info["tight_box"] = tight.to_list()
            info["tight_area_fraction"] = _box_geometry(tight)["area_fraction"]
            if acceptable:
                info["accepted"] = True
                candidate["box"] = tight.to_list()
                candidate["box_uvwh"] = tight.to_uvwh()
            else:
                info["reason"] = reason
                info["rejected_box"] = tight.to_list()
        attempts.append(info)
        candidate["tighten_retry"] = info
    if not attempts:
        return None, 0
    return {
        "attempted": len(attempts),
        "accepted": sum(1 for item in attempts if item.get("accepted")),
        "vlm_calls": calls,
        "candidates": attempts,
    }, calls


def _retry_hint_box(
    ctx: ToolContext, record: PerceptionObservation, instruction: str, subtask_key: str
) -> tuple[list[float] | None, str]:
    """A previous candidate's box (same frame or projected) for a visual retry."""
    candidates = [
        candidate
        for candidate in _candidate_store(ctx).values()
        if isinstance(candidate, Mapping) and candidate.get("box")
    ]
    for candidate in reversed(candidates):
        if str(candidate.get("observation_id") or "") != str(record.observation_id):
            continue
        same_stage = bool(subtask_key) and str(candidate.get("subtask_key") or "") == subtask_key
        if same_stage or _same_target_instruction(
            str(candidate.get("instruction") or ""), instruction
        ):
            try:
                return [float(value) for value in list(candidate["box"])[:4]], "previous_box"
            except (TypeError, ValueError):
                continue
    if record.position_flu is None or record.yaw_rad is None:
        return None, ""
    try:
        from .skills_approach import world_to_pixel
    except Exception:  # noqa: BLE001 - projection is best effort
        return None, ""
    intrinsics = record.intrinsics or _observation_intrinsics(ctx, record)
    for candidate in reversed(candidates):
        same_stage = bool(subtask_key) and str(candidate.get("subtask_key") or "") == subtask_key
        if not same_stage and not _same_target_instruction(
            str(candidate.get("instruction") or ""), instruction
        ):
            continue
        position = candidate.get("position_flu")
        if not isinstance(position, (list, tuple)) or len(position) < 3:
            continue
        pixel = world_to_pixel(
            position, record.position_flu, float(record.yaw_rad), intrinsics
        )
        if pixel is None:
            continue
        u, v = float(pixel[0]), float(pixel[1])
        return [u - 80.0, v - 80.0, u + 80.0, v + 80.0], "projected_previous_candidate"
    return None, ""


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


def _ray_vector(ray: Mapping | Any) -> np.ndarray | None:
    """Unit direction of a stored ``ray`` (bearing/elevation in degrees)."""
    if not isinstance(ray, Mapping):
        return None
    bearing = _as_float(ray.get("bearing_deg"), None)
    if bearing is None:
        return None
    elevation = _as_float(ray.get("elevation_deg"), 0.0) or 0.0
    bearing_rad = math.radians(bearing)
    elevation_rad = math.radians(elevation)
    horizontal = math.cos(elevation_rad)
    return np.asarray(
        [
            horizontal * math.cos(bearing_rad),
            horizontal * math.sin(bearing_rad),
            math.sin(elevation_rad),
        ],
        dtype=np.float64,
    )


def _ray_origin(ray: Mapping | Any) -> np.ndarray | None:
    if not isinstance(ray, Mapping):
        return None
    origin = ray.get("origin_flu")
    if not isinstance(origin, (list, tuple)) or len(origin) < 3:
        return None
    try:
        return np.asarray([float(value) for value in origin[:3]], dtype=np.float64)
    except (TypeError, ValueError):
        return None


def _triangulate_rays(ray_a: Mapping, ray_b: Mapping) -> dict | None:
    """Closest approach of two world rays (``None`` when they are degenerate).

    Two views of the same object give two bearings through its centre; the
    point where those rays meet is a position that does not depend on either
    view's depth reading. The residual is the distance between the rays at
    their closest approach - a large residual means two different objects
    (or a bearing error too large to localise).
    """
    origin_a, origin_b = _ray_origin(ray_a), _ray_origin(ray_b)
    direction_a, direction_b = _ray_vector(ray_a), _ray_vector(ray_b)
    if origin_a is None or origin_b is None or direction_a is None or direction_b is None:
        return None
    baseline = float(np.linalg.norm(origin_b - origin_a))
    if baseline < TRIANGULATION_MIN_BASELINE_M:
        return None
    cosine = float(np.clip(np.dot(direction_a, direction_b), -1.0, 1.0))
    angle = math.degrees(math.acos(min(1.0, abs(cosine))))
    if angle < TRIANGULATION_MIN_ANGLE_DEG:
        return None
    w0 = origin_a - origin_b
    b = float(np.dot(direction_a, direction_b))
    denom = 1.0 - b * b
    if denom <= 1e-6:
        return None
    d = float(np.dot(direction_a, w0))
    e = float(np.dot(direction_b, w0))
    t_a = (b * e - d) / denom
    t_b = (e - b * d) / denom
    if t_a <= MIN_RANGE_M or t_b <= MIN_RANGE_M:
        return None
    point_a = origin_a + t_a * direction_a
    point_b = origin_b + t_b * direction_b
    point = 0.5 * (point_a + point_b)
    residual = float(np.linalg.norm(point_a - point_b))
    if residual > TRIANGULATION_MAX_RESIDUAL_M:
        return None
    return {
        "point": [round(float(value), 3) for value in point],
        "range_a_m": round(t_a, 3),
        "range_b_m": round(t_b, 3),
        "baseline_m": round(baseline, 3),
        "angle_deg": round(angle, 2),
        "residual_m": round(residual, 3),
    }


def _expected_range(candidate: Mapping) -> float | None:
    """The range a view trusts: a normal depth measurement, else the model."""
    if bool(candidate.get("distance_uncertain")) or str(
        candidate.get("distance_source") or ""
    ) == DISTANCE_CONFLICT:
        value = _as_float(candidate.get("distance_m"), None)
        return None if value is None or value <= MIN_RANGE_M else value
    value = _as_float(candidate.get("range_m"), None)
    if value is not None and value > MIN_RANGE_M:
        return value
    value = _as_float(candidate.get("distance_m"), None)
    return None if value is None or value <= MIN_RANGE_M else value


def _range_check(candidate: Mapping, triangulated_m: float) -> dict:
    expected = _expected_range(candidate)
    if expected is None:
        return {"available": False}
    ratio = max(triangulated_m / expected, expected / triangulated_m)
    # a scale-free 2x gate: an absolute escape would let a 1 m range agree
    # with a 3.5 m intersection at close range
    consistent = ratio <= TRIANGULATION_RANGE_RATIO
    return {
        "available": True,
        "expected_m": round(float(expected), 3),
        "source": (
            "model"
            if bool(candidate.get("distance_uncertain"))
            or str(candidate.get("distance_source") or "") == DISTANCE_CONFLICT
            else "depth"
        ),
        "ratio": round(ratio, 3),
        "difference_m": round(abs(triangulated_m - expected), 3),
        "consistent": bool(consistent),
    }


_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_PAREN_RE = re.compile(r"[（(][^）)]*[）)]")
#: Instruction words that carry no object identity. Removing them keeps the
#: noun overlap between "带书的书架" and "一个摆满书的书架" decisive.
_ZH_FILLER = (
    "找到", "寻找", "搜索", "定位", "确认", "观察", "看到", "看见",
    "一个", "这个", "那个", "地标", "摆满", "旁边", "左侧", "右侧",
    "左边", "右边", "附近", "前面", "后面", "上方", "下方", "之间",
    "对面", "教室里", "卧室里", "教室", "卧室", "的", "在", "里",
    "和", "与", "及", "带",
)
_EN_FILLER = frozenset(
    {
        "the", "a", "an", "this", "that", "find", "locate", "search",
        "look", "for", "of", "in", "on", "at", "to", "next", "near",
        "beside", "by", "with", "and", "target", "object", "visible",
        "please", "show", "me",
    }
)


def _target_tokens(text: str) -> set[str]:
    """Identity-bearing tokens: ASCII words plus CJK character bigrams."""
    lowered = _PAREN_RE.sub(" ", str(text or "").lower())
    tokens: set[str] = set()
    for word in _WORD_RE.findall(lowered):
        if len(word) >= 3 and word not in _EN_FILLER:
            tokens.add(word)
    for run in _CJK_RE.findall(lowered):
        cleaned = run
        for filler in _ZH_FILLER:
            cleaned = cleaned.replace(filler, "")
        if len(cleaned) == 2:
            tokens.add(cleaned)
        for index in range(len(cleaned) - 1):
            tokens.add(cleaned[index:index + 2])
    return tokens


def _target_head(text: str) -> str:
    """The head noun's final character of a CJK phrase ("" when not CJK).

    Chinese compounds put the head noun last ("木椅旁边的灯" is a lamp, not a
    chair), so a shared anchor noun alone must not merge two targets.
    """
    lowered = _PAREN_RE.sub(" ", str(text or "").lower())
    runs = _CJK_RE.findall(lowered)
    if not runs:
        return ""
    head = runs[-1].rstrip("的了")
    return head[-1] if head else ""


def _same_target_instruction(first: str, second: str) -> bool:
    """Do two grounding instructions name the same object?

    Exact string matching broke the v6 classroom merge: the same shelf was
    grounded as ``带书的书架`` and as ``在教室里找到带书的书架（地标） - 一个
    摆满书的书架``. Token overlap plus the CJK head noun keeps rephrasings of
    one object together without merging an anchor noun with its relatum.
    """
    tokens_a, tokens_b = _target_tokens(first), _target_tokens(second)
    if not tokens_a or not tokens_b or not (tokens_a & tokens_b):
        return False
    head_a, head_b = _target_head(first), _target_head(second)
    if head_a and head_b and head_a != head_b:
        return False
    return True


def _subtask_key(ctx: ToolContext) -> str:
    """Stable identity of the current plan stage ("" without a plan)."""
    try:
        from .tool_impl_task import current_subtask

        subtask = current_subtask(ctx)
    except Exception:  # noqa: BLE001 - a context without the task module
        subtask = None
    if not isinstance(subtask, Mapping):
        return ""
    goal = str(subtask.get("goal") or "").strip()
    return f"stage:{_instruction_signature(goal)}" if goal else ""


def _triangulation_rejections(ctx: ToolContext) -> list[dict]:
    extra = _extra(ctx)
    items = extra.get(TRIANGULATION_REJECTIONS_KEY)
    if not isinstance(items, list):
        items = []
        extra[TRIANGULATION_REJECTIONS_KEY] = items
    return items


def _remember_rejected_triangulation(
    ctx: ToolContext, candidate: Mapping, reason: str
) -> dict | None:
    """Remember a candidate a confirmation rejected terminally.

    v6 bedroom seed37 re-minted the same 5.15-deg intersection nine times (and
    v7 did it five times with 21.3 deg) because only ``position_inconsistent``
    was remembered. Every terminal rejection now records the point, the ray pair
    and a candidate fingerprint (target identity + bearing/box), so a later
    re-grounding can neither re-mint the intersection nor hand the same
    appearance back as a fresh candidate.
    """
    if not isinstance(candidate, Mapping):
        return None
    triangulation = (
        candidate.get("triangulation")
        if isinstance(candidate.get("triangulation"), Mapping) else {}
    )
    raw_point = candidate.get("position_flu") or triangulation.get("point")
    point: list[float] = []
    if isinstance(raw_point, (list, tuple)) and len(raw_point) >= 3:
        try:
            point = [round(float(value), 3) for value in list(raw_point)[:3]]
        except (TypeError, ValueError):
            point = []
    if len(point) == 3 and not all(math.isfinite(value) for value in point):
        point = []
    store = _candidate_store(ctx)
    merged_id = str(triangulation.get("merged_candidate_id") or "")
    merged = store.get(merged_id) if merged_id else None
    own_ray = _ray_signature(candidate.get("ray"))
    merged_ray = _ray_signature(merged.get("ray")) if isinstance(merged, Mapping) else None
    box = candidate.get("box")
    target_key = _candidate_target_key(candidate)
    if not point and own_ray is None and merged_ray is None and not box and not target_key:
        return None
    entry = {
        "candidate_id": str(candidate.get("id") or ""),
        "merged_candidate_id": merged_id,
        "observation_id": str(
            triangulation.get("observation_id")
            or candidate.get("observation_id") or ""
        ),
        "merged_observation_id": str(
            triangulation.get("merged_observation_id")
            or (merged.get("observation_id") if isinstance(merged, Mapping) else "")
            or ""
        ),
        "point": point,
        "angle_deg": triangulation.get("angle_deg"),
        "baseline_m": triangulation.get("baseline_m"),
        "reason": str(reason),
        "instruction": str(candidate.get("instruction") or "")[:200],
        "subtask_key": str(candidate.get("subtask_key") or ""),
        "target_key": target_key,
        "ray": own_ray,
        "merged_ray": merged_ray,
        "box": list(box) if isinstance(box, (list, tuple)) and len(box) >= 4 else None,
        "fingerprint": _candidate_fingerprint(candidate, store),
    }
    items = _triangulation_rejections(ctx)
    items.append(entry)
    del items[:-TRIANGULATION_REJECTION_LIMIT]
    return entry


def _candidate_target_key(candidate: Mapping) -> str:
    """Identity of the target a rejection belongs to (stage, else instruction)."""
    return str(candidate.get("subtask_key") or "").strip() or _instruction_signature(
        str(candidate.get("instruction") or "")
    )


def _ray_signature(ray: Any) -> dict | None:
    """Quantized bearing/origin of a stored ray (the camera pose it came from)."""
    if not isinstance(ray, Mapping):
        return None
    bearing = _as_float(ray.get("bearing_deg"), None)
    elevation = _as_float(ray.get("elevation_deg"), 0.0)
    origin = _ray_origin(ray)
    if bearing is None or origin is None:
        return None
    return {
        "origin_flu": [round(float(value), 2) for value in origin],
        "bearing_deg": round(float(bearing), 1),
        "elevation_deg": round(float(elevation or 0.0), 1),
    }


def _rays_compatible(first: Any, second: Any) -> bool:
    """Two ray signatures from the same viewpoint and almost the same bearing."""
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        return False
    origin_a = first.get("origin_flu")
    origin_b = second.get("origin_flu")
    try:
        origin_gap = float(
            np.linalg.norm(
                np.asarray(origin_a[:3], dtype=np.float64)
                - np.asarray(origin_b[:3], dtype=np.float64)
            )
        )
        bearing_gap = abs(
            _norm_heading(
                float(first.get("bearing_deg") or 0.0)
                - float(second.get("bearing_deg") or 0.0)
            ) or 0.0
        )
        elevation_gap = abs(
            float(first.get("elevation_deg") or 0.0)
            - float(second.get("elevation_deg") or 0.0)
        )
    except (TypeError, ValueError):
        return False
    return (
        origin_gap <= TRIANGULATION_REJECTION_ORIGIN_M
        and bearing_gap <= TRIANGULATION_REJECTION_RAY_DEG
        and elevation_gap <= TRIANGULATION_REJECTION_RAY_DEG
    )


def _box_iou(first: Any, second: Any) -> float:
    try:
        ax1, ay1, ax2, ay2 = (float(value) for value in list(first)[:4])
        bx1, by1, bx2, by2 = (float(value) for value in list(second)[:4])
    except (TypeError, ValueError):
        return 0.0
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(
        0.0, bx2 - bx1
    ) * max(0.0, by2 - by1) - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def _same_rejection_target(entry: Mapping, candidate: Mapping) -> bool:
    stored_stage = str(entry.get("subtask_key") or "")
    current_stage = str(candidate.get("subtask_key") or "")
    if stored_stage and current_stage and stored_stage != current_stage:
        return False
    stored_key = str(entry.get("target_key") or "")
    current_key = _candidate_target_key(candidate)
    if stored_key and current_key:
        if stored_key == current_key:
            return True
        return _same_target_instruction(
            str(entry.get("instruction") or stored_key),
            str(candidate.get("instruction") or current_key),
        )
    return False


def _candidate_fingerprint(candidate: Mapping, store: Mapping | None = None) -> str:
    """Stable fingerprint of a candidate's identity + bearing + box.

    Re-grounding the same instance usually produces a new observation id and a
    slightly moved box; the target identity together with the ray (same pose)
    and a quantized box is what stays equal. The triangulated point is included
    when one exists, so the same two-view intersection is recognizable even
    when its observation ids changed.
    """
    triangulation = (
        candidate.get("triangulation")
        if isinstance(candidate.get("triangulation"), Mapping) else {}
    )
    merged_id = str(triangulation.get("merged_candidate_id") or "")
    merged = store.get(merged_id) if isinstance(store, Mapping) and merged_id else None
    box = candidate.get("box")
    point = candidate.get("position_flu") or triangulation.get("point")
    payload = {
        "target": _candidate_target_key(candidate),
        "point": (
            [round(float(value), 2) for value in list(point)[:3]]
            if isinstance(point, (list, tuple)) and len(point) >= 3
            else None
        ),
        "ray": _ray_signature(candidate.get("ray")),
        "merged_ray": (
            _ray_signature(merged.get("ray")) if isinstance(merged, Mapping) else None
        ),
        "box": (
            [round(float(value) / 10.0) * 10.0 for value in list(box)[:4]]
            if isinstance(box, (list, tuple)) and len(box) >= 4
            else None
        ),
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]


def _candidate_rejection_match(
    ctx: ToolContext, candidate: Mapping
) -> tuple[dict, str] | None:
    """The recorded terminal rejection this candidate repeats, if any.

    Returns ``(entry, kind)`` where kind is ``point`` (the rejected place),
    ``ray`` (the same viewpoint) or ``box`` (the same appearance in the same
    region). The caller may treat ``box`` as superseded when the candidate now
    carries a fresh, trustworthy depth measurement - that is new evidence.
    """
    point = candidate.get("position_flu")
    own_ray = _ray_signature(candidate.get("ray"))
    box = candidate.get("box")
    for entry in reversed(_triangulation_rejections(ctx)):
        if not _same_rejection_target(entry, candidate):
            continue
        stored_point = entry.get("point")
        if (
            isinstance(point, (list, tuple)) and len(point) >= 3
            and isinstance(stored_point, (list, tuple)) and len(stored_point) >= 3
        ):
            try:
                gap = float(
                    np.linalg.norm(
                        np.asarray(point[:3], dtype=np.float64)
                        - np.asarray(stored_point[:3], dtype=np.float64)
                    )
                )
            except (TypeError, ValueError):
                gap = float("inf")
            if gap <= TRIANGULATION_REJECTION_RADIUS_M:
                return entry, "point"
        if _rays_compatible(own_ray, entry.get("ray")):
            return entry, "ray"
        if _rays_compatible(own_ray, entry.get("merged_ray")):
            return entry, "ray"
        if (
            isinstance(box, (list, tuple)) and len(box) >= 4
            and isinstance(entry.get("box"), (list, tuple))
            and _box_iou(box, entry.get("box")) >= TRIANGULATION_REJECTION_BOX_IOU
        ):
            return entry, "box"
    return None


def _has_independent_depth_measurement(candidate: Mapping) -> bool:
    """A fresh single-view depth range, i.e. new measurement evidence."""
    return (
        str(candidate.get("position_source") or "") == POSITION_DEPTH
        and str(candidate.get("distance_source") or "") in TRUSTED_DEPTH_SOURCES
        and candidate.get("distance_uncertain") is not True
    )


def _mark_inherited_rejection(candidate: dict, entry: Mapping) -> dict:
    """Make a re-grounded repeat of a rejected candidate rejected in place."""
    candidate["status"] = "rejected"
    candidate["confirmation_status"] = "rejected"
    candidate["rejection_inherited"] = True
    candidate["rejection_reason"] = str(entry.get("reason") or "rejected")
    candidate["rejection_fingerprint"] = entry.get("fingerprint")
    candidate["rejected_point"] = entry.get("point")
    candidate["rejection_source_candidate"] = entry.get("candidate_id")
    return candidate


def _triangulation_rejection(
    ctx: ToolContext,
    other: Mapping,
    record: PerceptionObservation,
    point: list[float],
    candidate: Mapping | None = None,
) -> dict | None:
    """The rejection entry that forbids this ray pair, or None.

    Point radius, the exact observation pair and the ray pair are checked; the
    ray pair is order-independent because the two views may be re-grounded in
    the opposite order.
    """
    try:
        target = np.asarray(point, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        target = None
    current_ray = _ray_signature(candidate.get("ray")) if candidate else None
    other_ray = _ray_signature(other.get("ray"))
    for entry in reversed(_triangulation_rejections(ctx)):
        stored = entry.get("point")
        if target is not None and isinstance(stored, (list, tuple)) and len(stored) >= 3:
            try:
                gap = float(
                    np.linalg.norm(target - np.asarray(stored[:3], dtype=np.float64))
                )
            except (TypeError, ValueError):
                gap = float("inf")
            if gap <= TRIANGULATION_REJECTION_RADIUS_M:
                return entry
        if (
            str(entry.get("merged_candidate_id") or "") == str(other.get("id") or "")
            and str(entry.get("observation_id") or "") == str(record.observation_id)
        ):
            return entry
        if (
            str(entry.get("merged_observation_id") or "")
            == str(other.get("observation_id") or "")
            and str(entry.get("observation_id") or "") == str(record.observation_id)
        ):
            return entry
        if current_ray is not None and other_ray is not None:
            if (
                _rays_compatible(current_ray, entry.get("ray"))
                and _rays_compatible(other_ray, entry.get("merged_ray"))
            ) or (
                _rays_compatible(current_ray, entry.get("merged_ray"))
                and _rays_compatible(other_ray, entry.get("ray"))
            ):
                return entry
    return None


def _triangulation_rejected(
    ctx: ToolContext,
    other: Mapping,
    record: PerceptionObservation,
    point: list[float],
) -> bool:
    return _triangulation_rejection(ctx, other, record, point) is not None


def _triangulate_candidate(
    ctx: ToolContext,
    candidate: dict,
    record: PerceptionObservation,
    diagnostics: dict | None = None,
) -> dict | None:
    """Second-view support for a fresh candidate: ray intersection with an
    earlier grounding of the same target from a different capture pose.

    The earlier candidate's stored ray is reused (it was computed from that
    frame's own pose/intrinsics); no pixels are re-read. Same-pose views (a
    lookaround sweep) have no baseline and are deliberately skipped - their
    heading change supplies no parallax, only the later viewpoint does.
    Target identity is matched by the plan stage (same sub-task) or by noun
    overlap, never by the exact instruction string: the model rephrases the
    same object between views.

    ``diagnostics["rejection"]`` is filled when a geometrically valid pair was
    refused by the rejection memory; the caller turns that into a rejected
    candidate instead of silently dropping the bearing.
    """
    ray = candidate.get("ray")
    if not isinstance(ray, Mapping):
        return None
    subtask_key = str(candidate.get("subtask_key") or "")
    store = _candidate_store(ctx)
    best: tuple[tuple[float, float], dict, dict, dict] | None = None
    for other in store.values():
        if not isinstance(other, dict) or other is candidate:
            continue
        if str(other.get("id")) == str(candidate.get("id")):
            continue
        if str(other.get("observation_id") or "") == str(record.observation_id):
            continue
        same_stage = bool(subtask_key) and str(other.get("subtask_key") or "") == subtask_key
        if not same_stage and not _same_target_instruction(
            str(other.get("instruction") or ""), str(candidate.get("instruction") or "")
        ):
            continue
        other_ray = other.get("ray")
        if not isinstance(other_ray, Mapping):
            continue
        geometry = _triangulate_rays(ray, other_ray)
        if geometry is None:
            continue
        rejection = _triangulation_rejection(
            ctx, other, record, geometry["point"], candidate
        )
        if rejection is not None:
            if isinstance(diagnostics, dict):
                diagnostics.setdefault("rejection", rejection)
            continue
        current_check = _range_check(candidate, geometry["range_a_m"])
        other_check = _range_check(other, geometry["range_b_m"])
        available = [check for check in (current_check, other_check) if check.get("available")]
        # every available single-view range has to agree with the intersection:
        # one consistent view does not make the other view's 2.7x disagreement
        # disappear (final3 classroom seed37 accepted exactly such a pair)
        if available and not all(check.get("consistent") for check in available):
            continue
        key = (geometry["residual_m"], -geometry["angle_deg"])
        if best is None or key < best[0]:
            best = (key, other, geometry, {"current": current_check, "other": other_check})
    if best is None:
        return None
    _, other, geometry, checks = best
    return {
        "position_source": POSITION_TRIANGULATED,
        "position_flu": list(geometry["point"]),
        "range_m": geometry["range_a_m"],
        "range_source": POSITION_TRIANGULATED,
        "range_status": "known",
        "distance_source": DISTANCE_TRIANGULATED,
        "distance_uncertain": False,
        "triangulation": {
            "method": "ray_intersection",
            "merged_candidate_id": str(other.get("id") or ""),
            "merged_observation_id": str(other.get("observation_id") or ""),
            "observation_id": str(record.observation_id),
            "point": list(geometry["point"]),
            "baseline_m": geometry["baseline_m"],
            "angle_deg": geometry["angle_deg"],
            "residual_m": geometry["residual_m"],
            "range_a_m": geometry["range_a_m"],
            "range_b_m": geometry["range_b_m"],
            "merged_box": other.get("box"),
            "range_checks": checks,
            "range_check_consistent": all(
                check.get("consistent", True) for check in checks.values()
            ),
        },
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


def _distance_conflict(model_m: float | None, depth_m: float | None) -> dict | None:
    """``{model_m, depth_m, ratio, difference_m}`` when the two ranges disagree.

    The model's own estimate and a depth reading are independent measurements of
    the same range. A ratio above :data:`DISTANCE_CONFLICT_RATIO` (or a metric
    difference above :data:`DISTANCE_CONFLICT_DIFF_M`) means at least one of them
    is not measuring the box: the classroom v2 box centres reported 16-27 m of
    wall while the model estimated 3-4 m, and the wall won.
    """
    if model_m is None or depth_m is None:
        return None
    model = float(model_m)
    depth = float(depth_m)
    if model <= MIN_RANGE_M or depth <= MIN_RANGE_M:
        return None
    ratio = max(depth / model, model / depth)
    difference = abs(depth - model)
    if ratio <= DISTANCE_CONFLICT_RATIO and difference <= DISTANCE_CONFLICT_DIFF_M:
        return None
    return {
        "model_m": round(model, 3),
        "depth_m": round(depth, 3),
        "ratio": round(ratio, 3),
        "difference_m": round(difference, 3),
    }


def _box_geometry(box: Box) -> dict:
    """Frame coverage of a box: area fraction, border contact, large flag."""
    area_fraction = max(0.0, box.width) * max(0.0, box.height) / 1_000_000.0
    border = (
        box.x1 <= BOX_BORDER_MARGIN
        or box.y1 <= BOX_BORDER_MARGIN
        or box.x2 >= 1000.0 - BOX_BORDER_MARGIN
        or box.y2 >= 1000.0 - BOX_BORDER_MARGIN
    )
    return {
        "area_fraction": round(area_fraction, 3),
        "border_touching": bool(border),
        "large": bool(area_fraction > LARGE_BOX_FRACTION or border),
    }


def _depth_point(
    ctx: ToolContext,
    record: PerceptionObservation,
    pixel: tuple[float, float],
    depth_fn,
) -> dict | None:
    """One pixel's depth + world point, or None when that sample is unusable."""
    depth_pixel, _, problem = _source_pixel(ctx, record, pixel)
    if problem:
        return None
    try:
        depth = _as_float(depth_fn(depth_pixel), None)
    except Exception:  # noqa: BLE001 - a depth backend is best effort
        return None
    if depth is None or depth <= MIN_RANGE_M:
        return None
    point, problem = _project_pixel(ctx, record, pixel, depth)
    if point is None:
        return None
    return {
        "pixel": [round(float(pixel[0]), 1), round(float(pixel[1]), 1)],
        "depth_m": round(float(depth), 3),
        "position_flu": point,
    }


def _ground_points(box: Box) -> list[tuple[str, tuple[float, float]]]:
    """Multi-point samples across the box's lower half (a box centre on a large
    box usually measures the wall behind the object).

    The bottom edge is the object-support contact; the two interior rows catch
    an object whose visible silhouette is tall enough that the bottom edge still
    looks past it (a shelf, a desk) while an inside pixel lands on its near
    surface. Samples are ordered from the support contact upwards.
    """
    width, height = box.width, box.height
    # stay just inside the silhouette: the exact border pixel is a mix of the
    # object and whatever is behind it
    bottom = min(1000.0, max(0.0, box.y2 - max(2.0, 0.03 * height)))
    xs = (
        ("25", box.x1 + 0.25 * width),
        ("centre", box.x1 + 0.50 * width),
        ("75", box.x1 + 0.75 * width),
    )
    rows = (
        ("bottom", bottom),
        ("lower", min(1000.0, max(0.0, box.y2 - 0.30 * height))),
        ("middle", min(1000.0, max(0.0, box.y1 + 0.50 * height))),
    )
    return [
        (f"{row}_{label}", (x, y))
        for row, y in rows
        for label, x in xs
    ]


def _converged_sample(
    samples: list[tuple[str, dict]],
) -> tuple[str, dict, dict] | None:
    """The nearest sample of a tight cluster, or None when the set is dispersed.

    A support-contact row that reads the same surface three times is a
    measurement; the same points spread over metres are not one surface (a
    bottom edge crossing the object and the wall behind it). The nearest sample
    of a converged row is the near surface the model's range may have
    under-estimated.
    """
    if len(samples) < DISTANCE_CONSENSUS_MIN_SAMPLES:
        return None
    depths = sorted(float(sample["depth_m"]) for _, sample in samples)
    median = depths[len(depths) // 2]
    spread = depths[-1] - depths[0]
    if median <= MIN_RANGE_M:
        return None
    if spread > DISTANCE_CONSENSUS_SPREAD_M:
        return None
    if spread / median > DISTANCE_CONSENSUS_SPREAD_RATIO:
        return None
    name, sample = min(samples, key=lambda item: float(item[1]["depth_m"]))
    stats = {
        "samples": len(samples),
        "median_m": round(median, 3),
        "spread_m": round(spread, 3),
        "spread_ratio": round(spread / median, 3),
    }
    return name, sample, stats


def _border_scan_sample(
    ctx: ToolContext,
    record: PerceptionObservation,
    box: Box,
    depth_fn,
    model: float | None,
) -> dict | None:
    """A measurable support point on the bottom strip of a border-large box.

    The ordinary centre/ground samples of a shelf-sized border box read the wall
    behind it (the classroom 25/27 loop). The object's support contact can still
    be visible along the box bottom: scan columns across the box width, read
    several rows near the bottom, keep the nearest reading of each column and
    accept only a contiguous run whose depths agree with the model's range.
    Returns ``{sample, detail}`` (``None`` when no run is a surface).
    """
    if model is None or depth_fn is None:
        return None
    width = max(1.0, float(box.width))
    height = max(1.0, float(box.height))
    rows = sorted({
        round(
            min(1000.0, max(box.y1, box.y2 - fraction * height)),
            1,
        )
        for fraction in BORDER_SCAN_ROW_INSETS
    })
    columns: list[dict] = []
    for index in range(BORDER_SCAN_COLUMNS):
        x = box.x1 + (index + 0.5) / BORDER_SCAN_COLUMNS * width
        nearest = None
        for y in rows:
            sample = _depth_point(ctx, record, (x, y), depth_fn)
            if sample is None:
                continue
            if nearest is None or sample["depth_m"] < nearest["depth_m"]:
                nearest = sample
        tolerance = max(BORDER_SCAN_MODEL_DIFF_M, 0.5 * float(model))
        consistent = False
        if nearest is not None:
            gap = abs(float(nearest["depth_m"]) - float(model))
            consistent = gap <= tolerance and (
                max(float(nearest["depth_m"]), float(model))
                / max(MIN_RANGE_M, min(float(nearest["depth_m"]), float(model)))
                <= BORDER_SCAN_MODEL_RATIO
            )
        columns.append({"sample": nearest, "consistent": bool(consistent)})
    runs: list[tuple[int, int]] = []
    start = None
    for index, column in enumerate(columns + [{"consistent": False}]):
        if column["consistent"] and start is None:
            start = index
        elif not column["consistent"] and start is not None:
            runs.append((start, index - 1))
            start = None
    best: tuple[tuple[float, int], int, int, dict, dict] | None = None
    for first, last in runs:
        length = last - first + 1
        if length < BORDER_SCAN_MIN_RUN:
            continue
        samples = [columns[index]["sample"] for index in range(first, last + 1)]
        depths = sorted(float(sample["depth_m"]) for sample in samples)
        median = depths[len(depths) // 2]
        spread = depths[-1] - depths[0]
        if spread > BORDER_SCAN_SPREAD_M:
            continue
        if median > MIN_RANGE_M and spread / median > DISTANCE_CONSENSUS_SPREAD_RATIO:
            continue
        middle = (first + last) // 2
        sample = columns[middle]["sample"]
        score = (abs(float(median) - float(model)), -length)
        if best is None or score < best[0]:
            best = (score, first, last, sample, {
                "columns": BORDER_SCAN_COLUMNS,
                "rows": rows,
                "run_columns": length,
                "run_start": first,
                "run_end": last,
                "median_m": round(median, 3),
                "spread_m": round(spread, 3),
            })
    if best is None:
        return None
    _, _first, _last, sample, detail = best
    detail.update({
        "sample_pixel": sample["pixel"],
        "sample_depth_m": sample["depth_m"],
    })
    return {"sample": sample, "detail": detail}


def _border_second_view(box: Box) -> dict:
    """The dedicated second view for a border-large box.

    The next bearing must look *toward the frame edge the box occupies*, not at
    the same shelf/wall; the small sideways step supplies the baseline the
    in-place sweep cannot. Yaw follows ``observe``'s convention (positive =
    left), so a box on the right edge is a negative offset.
    """
    left = box.x1 <= BOX_BORDER_MARGIN
    right = box.x2 >= 1000.0 - BOX_BORDER_MARGIN
    top = box.y1 <= BOX_BORDER_MARGIN
    bottom = box.y2 >= 1000.0 - BOX_BORDER_MARGIN
    centre_x = box.x1 + 0.5 * box.width
    offset = centre_x - 500.0
    if right and not left:
        side = "right"
    elif left and not right:
        side = "left"
    elif top and not bottom:
        side = "top"
    elif bottom and not top:
        side = "bottom"
    else:
        side = "right" if offset >= 0.0 else "left"
    magnitude = BORDER_SECOND_VIEW_MIN_DEG + (
        BORDER_SECOND_VIEW_MAX_DEG - BORDER_SECOND_VIEW_MIN_DEG
    ) * min(1.0, abs(offset) / 500.0)
    yaw = -magnitude if offset >= 0.0 else magnitude
    return {
        "side": side,
        "yaw_deg": round(yaw, 1),
        "step_m": BORDER_SECOND_VIEW_STEP_M,
        "step_side_m": round(BORDER_SECOND_VIEW_STEP_M, 1),
        "why": (
            f"the box runs into the {side} edge: turn {abs(yaw):.0f} deg toward "
            f"that edge and take a {BORDER_SECOND_VIEW_STEP_M:g} m sideways step "
            "before grounding again; two bearings with real parallax replace the "
            "unmeasurable single-view range"
        ),
    }


def _candidate_location(
    ctx: ToolContext,
    record: PerceptionObservation,
    box: Box,
    distance_m: float | None,
    depth_fn,
    depth_reason: str,
) -> dict:
    """Where a candidate box is: a position when the range is known, else a ray.

    The range is chosen from independent measurements of the box:

    - the box centre's depth (``distance_source=depth_center``);
    - multi-point samples across the box's lower half, bottom edge and interior
      rows (``depth_ground``), selected by agreement with the model's range;
    - a converged support-contact cluster on a normal (not border/large) box
      (``depth_consistent``) when the model range is a short estimate: the
      cluster is a measurement, the model range is recorded as an underestimate;
    - a contiguous bottom-strip cluster on a border-touching large box
      (``depth_border_scan``): the centre and the support contact read the wall,
      but a run of columns whose depth agrees with the model is the near
      support surface;
    - the model's own ``distance_m`` estimate (``model``).

    When the centre depth and the model estimate contradict each other (ratio
    > 2 or difference > 3 m) the centre is not trusted: the surface samples are
    tried first, then the model. A large box gets the surface samples because
    its centre is usually background; a box that both touches the frame border
    and covers more than 15% of it is first scanned along its bottom strip, then
    (when no near surface exists) kept as a model range with a dedicated
    second-view request toward the box's frame edge. If no independent
    measurement agrees, the model range is used but the candidate is marked
    ``distance_source=conflict`` / ``distance_uncertain=True`` - confirmation
    may not accept it and navigation may not approach it until it is re-grounded
    (goal.md section 6). A box with no usable range at all is *not* placed at a
    guessed range: it keeps the direction (``ray``) and ``range_status=unknown``.

    ``position_source`` names how the position was obtained (``depth`` /
    ``model_distance``) or why none exists (``ray`` / ``intrinsics_unknown`` /
    ``pitched_view`` / ``pose_unknown`` / ``no_world`` / ``crop_source_unknown``
    / ``no_valid_depth``).
    """
    pixel = box.center
    # the depth map belongs to the frame rendered at the *current* pose, so the
    # box must be read at the pixel of that camera: a box grounded inside a crop
    # is in the crop's own pixel grid, and the pixel it maps back to is the one
    # the source frame has (feeding the crop pixel straight into the source depth
    # map measures whatever surface happens to sit under that unrelated pixel -
    # a box at the frame centre then reports the range of whatever is in front
    # of the drone instead of the range of the box)
    _, _, pixel_problem = _source_pixel(ctx, record, pixel)
    location: dict = {
        "position_source": POSITION_NO_RANGE,
        "range_status": "unknown",
        "distance_source": None,
        "distance_conflict": None,
        "distance_uncertain": False,
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

    geometry = _box_geometry(box)
    # a border-touching, large box is the shelf/wall ambiguity: its centre and
    # its support contact both read the surface behind the object, so no pixel
    # of this frame can measure it (classroom v5: [480,0,999,400], 0.208 of the
    # frame). It stays a direction with the model's estimate and needs a second
    # view with a real baseline before it can become a position.
    border_large = bool(
        geometry["border_touching"] and geometry["area_fraction"] > BORDER_LARGE_FRACTION
    )
    centre = None
    ground: list[tuple[str, dict]] = []
    usable_depth = depth_fn is not None and not pixel_problem
    model = (
        float(distance_m)
        if distance_m is not None and MIN_RANGE_M < distance_m <= MAX_RANGE_M
        else None
    )
    border_scan: dict | None = None
    if usable_depth and not border_large:
        centre = _depth_point(ctx, record, pixel, depth_fn)
        for name, point in _ground_points(box):
            sample = _depth_point(ctx, record, point, depth_fn)
            if sample is not None:
                ground.append((name, sample))
    elif usable_depth and border_large:
        # the ordinary samples are known to read the wall for this box shape;
        # the bottom strip is the one place a near support surface can still be
        # visible (v7: 25/27 classroom candidates were border-large conflicts)
        border_scan = _border_scan_sample(ctx, record, box, depth_fn, model)

    selected: tuple[str, str, dict] | None = None  # (source, sample_name, sample)
    conflict: dict | None = None
    consensus_detail: dict | None = None
    if border_scan is not None:
        selected = (
            DISTANCE_DEPTH_BORDER, "border_scan", border_scan["sample"]
        )
        consensus_detail = {
            "model_m": None if model is None else round(model, 3),
            "depth_center_m": None,
            "depth_ground_m": border_scan["sample"]["depth_m"],
            "ratio": (
                None if model is None else round(
                    max(
                        float(border_scan["sample"]["depth_m"]) / float(model),
                        float(model) / float(border_scan["sample"]["depth_m"]),
                    ),
                    3,
                )
            ),
            "difference_m": (
                None if model is None else round(
                    abs(float(border_scan["sample"]["depth_m"]) - float(model)), 3
                )
            ),
            "samples": [
                {
                    "source": DISTANCE_DEPTH_BORDER,
                    "depth_m": border_scan["sample"]["depth_m"],
                    "pixel": border_scan["sample"]["pixel"],
                }
            ],
            "border_scan": border_scan["detail"],
            "resolved": True,
            "resolution": "border_bottom_scan",
            "note": (
                "the box touches the frame border, but its bottom strip holds a "
                "contiguous near surface consistent with the model's range; that "
                "cluster is the support point (not the wall behind the object)"
            ),
        }
    if centre is not None or ground:
        candidates: list[tuple[str, str, dict]] = []
        if centre is not None:
            candidates.append((DISTANCE_DEPTH_CENTER, "centre", centre))
        candidates.extend((DISTANCE_DEPTH_GROUND, name, sample) for name, sample in ground)
        conflicts = {
            (source, name): _distance_conflict(model, sample["depth_m"])
            for source, name, sample in candidates
        }
        agreeing = [
            item for item in candidates if conflicts[(item[0], item[1])] is None
        ]
        if agreeing and model is not None:
            # with a model estimate to cross-check, prefer the support/surface
            # samples on a large box and the centre on a normal one; within that
            # group take the sample closest to the model's range (the near
            # surface, not an agreeing wall)
            preferred = (
                [item for item in agreeing if item[0] == DISTANCE_DEPTH_GROUND]
                if geometry["large"]
                else [item for item in agreeing if item[0] == DISTANCE_DEPTH_CENTER]
            )
            pool = preferred or agreeing
            selected = min(pool, key=lambda item: abs(float(item[2]["depth_m"]) - model))
        elif agreeing:
            # no model estimate: the box geometry alone decides
            preferred = (
                [item for item in agreeing if item[0] == DISTANCE_DEPTH_GROUND]
                if geometry["large"]
                else [item for item in agreeing if item[0] == DISTANCE_DEPTH_CENTER]
            )
            selected = preferred[0] if preferred else agreeing[0]
        else:
            # every depth sample contradicts the model. Two very different
            # cases look alike at this point: the samples may be the wall
            # behind the object (keep the model range, stay uncertain), or they
            # may be a tight cluster on the object's support surface while the
            # model's range is a short estimate. A converged support-contact
            # row on a normal (not border/large) box is a measurement; use its
            # near surface and record the model range as the underestimate.
            support = [
                (name, sample) for name, sample in ground
                if name.startswith("bottom")
            ] or list(ground)
            converged = (
                _converged_sample(support)
                if model is not None and not geometry["large"]
                else None
            )
            converged_conflict = (
                _distance_conflict(model, converged[1]["depth_m"])
                if converged is not None else None
            )
            if converged is not None and converged_conflict is not None:
                name, sample, consensus = converged
                selected = (DISTANCE_DEPTH_CONSISTENT, name, sample)
                consensus_detail = {
                    "model_m": round(model, 3),
                    "depth_center_m": None if centre is None else centre["depth_m"],
                    "depth_ground_m": sample["depth_m"],
                    "ratio": converged_conflict["ratio"],
                    "difference_m": converged_conflict["difference_m"],
                    "samples": [
                        {"sample": item_name, "depth_m": item["depth_m"]}
                        for item_name, item in support
                    ],
                    "consensus": consensus,
                    "resolved": True,
                    "resolution": "depth_consensus",
                    "model_underestimate": bool(float(model) < float(sample["depth_m"])),
                    "warning": (
                        "the model's range disagrees with a converged support-surface "
                        "depth reading; the sample is used as the flight range"
                    ),
                }
            else:
                # the samples do not agree with each other or the support point
                # is not trustworthy: keep the model range, but mark the
                # disagreement so confirmation cannot just accept it
                depth_details = [
                    {
                        "source": source,
                        "sample": name,
                        "depth_m": sample["depth_m"],
                        "ratio": conflicts[(source, name)]["ratio"],
                        "difference_m": conflicts[(source, name)]["difference_m"],
                    }
                    for source, name, sample in candidates
                ]
                conflict = {
                    "model_m": round(model, 3) if model is not None else None,
                    "depth_center_m": None if centre is None else centre["depth_m"],
                    "depth_ground_m": ground[0][1]["depth_m"] if ground else None,
                    "ratio": max(item["ratio"] for item in depth_details),
                    "difference_m": max(item["difference_m"] for item in depth_details),
                    "samples": depth_details,
                    "resolved": False,
                    "resolution": "model" if model is not None else "none",
                }
    elif model is None and not usable_depth:
        conflict = None

    border_second_view = (
        _border_second_view(box) if border_large and selected is None else None
    )
    if border_large and selected is None:
        # no bottom-strip support: the model's own estimate is the only range;
        # only a dedicated, offset second view can settle it
        conflict = {
            "model_m": None if model is None else round(model, 3),
            "depth_center_m": None,
            "depth_ground_m": None,
            "ratio": None,
            "difference_m": None,
            "samples": [],
            "resolved": False,
            "resolution": "border_large",
            "border_touching": True,
            "area_fraction": geometry["area_fraction"],
            "border_scan": (
                None if border_scan is None else border_scan["detail"]
            ),
            "why": (
                "the box touches the frame border and covers more than 15% of it: "
                "its centre and its support contact read the surface behind the "
                "object, and the bottom-strip scan found no near surface "
                "consistent with the model's range"
            ),
        }
    advice = None
    if conflict is not None:
        observe_yaw = (
            border_second_view["yaw_deg"] if border_second_view is not None else 45.0
        )
        advice = {
            "reason": (
                "border_large_box"
                if border_large
                else "dispersed_depth_samples"
            ),
            "recommended": "change_viewpoint",
            "tool": "ground_target",
            "args": {"instruction": None},
            "observe_first": {"tool": "observe", "args": {"mode": "forward", "yaw_deg": observe_yaw}},
            "dedicated_second_view": border_second_view,
            "alternative_tool": "propose_viewpoints",
            "why": (
                "a single view cannot resolve this box's range: fly to an offset "
                "viewpoint and ground the same instruction again; the two bearings "
                "triangulate the object, and the triangulated position replaces "
                "the single-view estimate"
            ),
        }
        if border_second_view is not None:
            advice["sequence"] = [
                {
                    "tool": "observe",
                    "args": {"mode": "forward", "yaw_deg": observe_yaw},
                },
                {"tool": "ground_target", "args": {"instruction": None}},
            ]
            advice["step_hint"] = (
                f"before grounding again, move about {border_second_view['step_side_m']:g} m "
                f"to the {border_second_view['side']} side of the current heading "
                "(a go_to_viewpoint / approach leg is fine): the second bearing needs "
                "real parallax, not only a rotation"
            )

    # the bearing of the box centre is needed by triangulation even when the
    # range came from depth (an early return must still carry it)
    ray = _pixel_ray(ctx, record, pixel)
    if selected is not None:
        source, sample_name, sample = selected
        detail = consensus_detail if consensus_detail is not None else conflict
        if detail is None and model is not None:
            # record a resolved disagreement for the trace: the centre was
            # rejected but a support sample agreed with the model
            centre_conflict = (
                _distance_conflict(model, centre["depth_m"]) if centre is not None else None
            )
            ground_conflict = (
                _distance_conflict(model, ground[0][1]["depth_m"]) if ground else None
            )
            if centre_conflict is not None or ground_conflict is not None:
                detail = {
                    "model_m": round(model, 3),
                    "depth_center_m": None if centre is None else centre["depth_m"],
                    "depth_ground_m": ground[0][1]["depth_m"] if ground else None,
                    "ratio": max(
                        item["ratio"] for item in (centre_conflict, ground_conflict)
                        if item is not None
                    ),
                    "difference_m": max(
                        item["difference_m"] for item in (centre_conflict, ground_conflict)
                        if item is not None
                    ),
                    "resolved": True,
                    "resolution": sample_name,
                }
        return {
            "position_source": POSITION_DEPTH,
            "position_flu": sample["position_flu"],
            "range_m": sample["depth_m"],
            "range_source": POSITION_DEPTH,
            "range_status": "known",
            "distance_source": source,
            "distance_conflict": detail,
            "distance_uncertain": False,
            "box_area_fraction": geometry["area_fraction"],
            "view_change_advice": advice,
            "border_large_box": border_large,
            "border_second_view": border_second_view,
            "border_scan": None if border_scan is None else border_scan["detail"],
            "ray": ray,
        }

    if model is not None:
        point, problem = _project_pixel(ctx, record, pixel, model)
        if point is not None:
            return {
                "position_source": POSITION_MODEL_DISTANCE,
                "position_flu": point,
                "range_m": round(model, 3),
                "range_source": POSITION_MODEL_DISTANCE,
                "range_status": "known",
                "distance_source": (
                    DISTANCE_CONFLICT if conflict is not None else DISTANCE_MODEL
                ),
                "distance_conflict": conflict,
                "distance_uncertain": bool(conflict is not None),
                "box_area_fraction": geometry["area_fraction"],
                "view_change_advice": advice,
                "border_large_box": border_large,
                "border_second_view": border_second_view,
                "border_scan": None if border_scan is None else border_scan["detail"],
                "ray": ray,
            }
        if location["position_source"] == POSITION_NO_RANGE:
            location["position_source"] = problem or POSITION_NO_RANGE
        if problem == POSITION_INTRINSICS_UNKNOWN:
            location["range_m"] = round(model, 3)
            location["range_source"] = POSITION_MODEL_DISTANCE
            location["range_status"] = "known"
            location["distance_source"] = (
                DISTANCE_CONFLICT if conflict is not None else DISTANCE_MODEL
            )
            location["distance_conflict"] = conflict
            location["distance_uncertain"] = bool(conflict is not None)
    elif centre is not None and conflict is None:
        # a depth exists but no model estimate needed for a decision yet the
        # projection failed: keep the measured range, not a fabricated point
        if location["position_source"] == POSITION_NO_RANGE:
            location["position_source"] = POSITION_INTRINSICS_UNKNOWN
        location["range_m"] = centre["depth_m"]
        location["range_source"] = POSITION_DEPTH
        location["range_status"] = "known"
        location["distance_source"] = DISTANCE_DEPTH_CENTER
        location["box_area_fraction"] = geometry["area_fraction"]

    if ray is not None:
        location["ray"] = ray
        if location["position_source"] == POSITION_NO_RANGE:
            # a direction from a known place: the honest answer for an unknown
            # distance is the ray, not a point at an invented range
            location["position_source"] = POSITION_RAY
    location["box_area_fraction"] = geometry["area_fraction"]
    location["view_change_advice"] = advice
    location["border_large_box"] = border_large
    location["border_second_view"] = border_second_view
    location["border_scan"] = None if border_scan is None else border_scan["detail"]
    if conflict is not None:
        location["distance_source"] = DISTANCE_CONFLICT
        location["distance_conflict"] = conflict
        location["distance_uncertain"] = True
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


def _ground_user_prompt(instruction: str, relation: Mapping | None = None) -> str:
    anchor_line = ""
    if relation is not None:
        anchor_line = (
            f' The instruction locates the target by the relation '
            f'"{(relation.get("anchor") or "")}" ({relation.get("relation")}). '
            'For every box, also report whether that anchor object is actually '
            'visible in this image ("anchor_visible": true/false); a candidate '
            'whose anchor is not visible cannot be verified.'
        )
    return (
        f"Instruction: {instruction}\n"
        "Return the normalized box (0-1000), the attributes, the rough "
        'distance in meters ("distance_m"; 0 if you cannot tell), the estimated '
        'real size ("size_m"), and for every box say whether it is the '
        'instructed instance itself ("matches_description": true/false, with '
        '"match_reason"); report what you are unsure about in "uncertainty". '
        "Do not present a wall, floor, support surface or neighbouring object "
        "as a match." + anchor_line
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


def _normalize_candidates(
    items: list,
    store: dict,
    *,
    width=None,
    height=None,
    default_anchor_visible=None,
) -> list[dict]:
    """Model reply -> candidate records with canonical ``[x1, y1, x2, y2]`` boxes.

    Each record keeps the format it was written in (``box_format``: the legacy
    ``[u, v, w, h]`` reading is normalized here, not silently re-guessed later),
    plus the legacy ``[u, v, w, h]`` view of the same box in ``box_uvwh`` so a
    consumer that still speaks the old shape does not have to convert.
    ``default_anchor_visible`` is a top-level answer the model gave instead of
    repeating it inside every candidate.
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
        match = item.get(
            "matches_description",
            item.get("match", item.get("is_target", item.get("matches"))),
        )
        size = _as_float(
            item.get("size_m", item.get("size", item.get("object_size_m"))), None
        )
        if size is not None and not (MIN_RANGE_M < size <= MAX_RANGE_M):
            size = None
        anchor_visible = _as_match_flag(
            item.get("anchor_visible", item.get("anchor_in_frame", item.get("anchor_seen")))
        )
        if anchor_visible is None:
            anchor_visible = default_anchor_visible
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
                "size_m": None if size is None else round(size, 3),
                "matches_description": _as_match_flag(match),
                "anchor_visible": anchor_visible,
                "match_reason": str(
                    item.get("match_reason", item.get("reason", "")) or ""
                )[:160],
                "uncertainty": "" if uncertainty is None else str(uncertainty)[:120],
                "instance_rank": None,
                "instance_suspect": False,
                "instance_suspect_reason": "",
            }
        )
    return candidates


# --------------------------------------------------------------------------
# instance consistency: is the box the instructed instance, or the wall?
# --------------------------------------------------------------------------

#: A box whose geometry at the claimed range implies an object this many times
#: larger than the model's own size (or the target noun's prior) is not that
#: instance: it is the wall/support surface around it.
INSTANCE_SIZE_RATIO = 4.0
#: Two views triangulating one point must see the same physical object. The
#: physical size implied by each view's box at that view's triangulated range is
#: a scale-free identity check: a box that also covers the wall behind the
#: object implies a much larger object than the other view does. A clipped box
#: can only under-state its object, so the check only fires when the *larger*
#: view is unclipped, or when the clipped box is the larger one (its implied
#: size is then a lower bound that already exceeds the other view).
INSTANCE_TWO_VIEW_SIZE_RATIO = 2.5
#: A relation ("next to the chair") should put the target within a few metres
#: of the anchor instance. Farther than this from every known anchor is a weak
#: contradiction, not proof; it is recorded as suspect.
INSTANCE_ANCHOR_MAX_M = 8.0
#: An explicit adjacency ("书架旁的课桌", "next to") is tighter than the weak
#: any-anchor bound: the relatum is a few metres away, not across the room.
INSTANCE_ADJACENT_MAX_M = 3.0
#: "附近的"/"near" is looser than strict adjacency.
INSTANCE_NEAR_MAX_M = 4.0
#: A projected anchor only has to fall inside the field of view, not inside the
#: candidate's box, so the FOV test keeps this much slack.
ANCHOR_VISIBILITY_MARGIN_DEG = 3.0
#: A known anchor farther than this cannot be a relatum visible in the frame.
ANCHOR_VISIBILITY_MAX_M = 40.0

#: (token, (width_m, height_m)) of common instructed objects. Rough on purpose:
#: the check only rejects order-of-magnitude contradictions (a 5 m "chair").
INSTANCE_SIZE_PRIORS = (
    ("bookshelf", (1.2, 1.9)), ("书架", (1.2, 1.9)),
    ("office chair", (0.65, 1.0)), ("办公椅", (0.65, 1.0)),
    ("armchair", (0.8, 0.9)), ("扶手椅", (0.8, 0.9)),
    ("chair", (0.55, 0.95)), ("椅子", (0.55, 0.95)), ("椅", (0.55, 0.95)),
    ("bottle", (0.08, 0.25)), ("瓶", (0.08, 0.25)),
    ("desk", (1.3, 0.75)), ("课桌", (1.3, 0.75)), ("桌", (1.2, 0.75)),
    ("lamp", (0.35, 0.5)), ("灯", (0.35, 0.5)),
    ("door", (0.9, 2.0)), ("门", (0.9, 2.0)),
    ("cabinet", (1.0, 1.8)), ("柜", (1.0, 1.8)),
)

_COLOR_ALIASES = {
    "black": ("black", "黑"),
    "white": ("white", "白"),
    "brown": ("brown", "棕", "褐", "木", "wood"),
    "gray": ("gray", "grey", "灰", "silver", "银"),
    "blue": ("blue", "蓝"),
    "green": ("green", "绿"),
    "red": ("red", "红"),
    "yellow": ("yellow", "黄"),
}
_MATERIAL_ALIASES = {
    "wood": ("wood", "wooden", "木"),
    "metal": ("metal", "steel", "iron", "金属", "钢", "铁"),
    "plastic": ("plastic", "塑料", "塑"),
    "glass": ("glass", "玻璃"),
    "fabric": ("fabric", "cloth", "布"),
}
_RELATION_TOKENS = (
    "旁边", "旁的", "边的", "附近的", "next to", "beside", "near ", "near the",
    " by the",
)
_TARGET_SPLIT = re.compile(
    r"(?:旁边的|旁的|边的|附近的|旁边的|的|next to|beside|near the|near)",
    re.IGNORECASE,
)

#: Relation half of an instruction: ``anchor + rel`` in Chinese ("书架旁的
#: 课桌"). The bare ``边``/``旁`` forms are the ones stage goals use ("在卧室
#: 窗边找到木椅"), so they must parse too - a re-grounding with the goal must
#: not lose the anchor the shorter label carried.
_CJK_RELATION_RE = re.compile(
    r"(?P<anchor>[^，。；;、\s]{1,14}?)"
    r"(?P<rel>旁边的|附近的|旁的|边的|旁边|附近|旁|边)"
)
#: English puts the target first ("the desk next to the bookshelf"), so the
#: relatum is the captured group after the relation.
_EN_RELATION_RE = re.compile(
    r"(?P<target>[a-z0-9 ,'\-]{2,60}?)\s+(?P<rel>next\s+to|beside|near)\s+"
    r"(?P<anchor>[a-z0-9 ,'\-]{2,60})",
    re.IGNORECASE,
)
_ANCHOR_PREFIX_RE = re.compile(r"^(?:在|到|找到|寻找)\s*")
#: Object nouns a relation can name. Categories let "带书的书架" match "书架"
#: and "办公椅" match "椅子" without merging a chair with its desk anchor.
_ANCHOR_NOUNS = (
    ("bookshelf", ("书架", "书柜", "shelf", "bookcase")),
    ("desk", ("课桌", "书桌", "桌子", "桌", "desk", "table")),
    ("chair", ("办公椅", "木椅", "椅子", "椅", "chair")),
    ("bottle", ("瓶子", "瓶", "bottle")),
    ("lamp", ("台灯", "灯", "lamp", "light")),
    ("window", ("窗", "window")),
    ("door", ("门", "door")),
    ("cabinet", ("柜", "cabinet")),
)


def _as_match_flag(value) -> bool | None:
    """The model's own instance judgement (None when it did not say)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "1", "match", "matches", "matched", "same", "是", "匹配", "一致"}:
        return True
    if text in {"false", "no", "0", "mismatch", "different", "否", "不是", "不匹配", "不一致"}:
        return False
    return None


def _target_phrase(instruction: str) -> str:
    """The part of a relation instruction that names the target instance.

    ``木椅旁边的灯`` is a lamp next to a chair: colour/material/size checks must
    read the head phrase, not the anchor noun, or every wooden anchor would
    "conflict" with the target's own colour.
    """
    text = str(instruction or "").strip()
    parts = [part for part in _TARGET_SPLIT.split(text) if part and part.strip()]
    return parts[-1].strip("（）()：: ，,。.") if parts else text


def _canonical_terms(text: str, aliases: Mapping[str, tuple]) -> set[str]:
    lowered = str(text or "").lower()
    return {
        canonical for canonical, words in aliases.items()
        if any(word.lower() in lowered for word in words)
    }


def _attribute_conflict(instruction: str, attrs: Mapping) -> str:
    """Colour/material attrs that contradict the target phrase ("" when none)."""
    if not isinstance(attrs, Mapping) or not attrs:
        return ""
    phrase = _target_phrase(instruction)
    attr_text = " ".join(f"{key} {value}" for key, value in attrs.items())
    wanted = _canonical_terms(phrase, _COLOR_ALIASES)
    seen = _canonical_terms(attr_text, _COLOR_ALIASES)
    if wanted and seen and not (wanted & seen):
        return f"color {sorted(wanted)} vs {sorted(seen)}"
    wanted = _canonical_terms(phrase, _MATERIAL_ALIASES)
    seen = _canonical_terms(attr_text, _MATERIAL_ALIASES)
    if wanted and seen and not (wanted & seen):
        return f"material {sorted(wanted)} vs {sorted(seen)}"
    return ""


def _attribute_unknown(instruction: str, attrs: Mapping) -> str:
    """Named colour/material the candidate's attributes do not cover ("" ok).

    The model's ``matches_description`` flag is not evidence (final3: 28/28
    candidates said ``true``, including a 5.6 m "bookshelf"). When the
    instruction names a colour or material, a hard deterministic answer is
    required: either the box's attributes contain that category or the candidate
    is unverified. An explicitly conflicting value is reported by
    :func:`_attribute_conflict` first; this function only reports silence.
    """
    phrase = _target_phrase(instruction)
    attr_text = " ".join(
        f"{key} {value}" for key, value in (attrs or {}).items()
    ) if isinstance(attrs, Mapping) else ""
    missing = []
    if _canonical_terms(phrase, _COLOR_ALIASES) and not _canonical_terms(
        attr_text, _COLOR_ALIASES
    ):
        missing.append("color")
    if _canonical_terms(phrase, _MATERIAL_ALIASES) and not _canonical_terms(
        attr_text, _MATERIAL_ALIASES
    ):
        missing.append("material")
    return ",".join(missing)


def _target_size_prior(instruction: str) -> tuple[float, float] | None:
    """Rough (width, height) of the target noun, latest/most specific match."""
    phrase = _target_phrase(instruction).lower()
    best = None
    best_key = (-1, -1)
    for token, size in INSTANCE_SIZE_PRIORS:
        position = phrase.rfind(token.lower())
        if position < 0:
            continue
        key = (position, len(token))
        if key > best_key:
            best, best_key = size, key
    return best


def _implied_box_size(box, distance_m: float | None, intrinsics: Mapping) -> dict | None:
    """Physical width/height a box subtends at the claimed range (metres)."""
    if distance_m is None or not (MIN_RANGE_M < float(distance_m) <= MAX_RANGE_M):
        return None
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in box[:4])
        hfov = float(intrinsics.get("hfov_deg"))
        width = float(intrinsics.get("width"))
        height = float(intrinsics.get("height"))
    except (TypeError, ValueError):
        return None
    if hfov <= 0.0 or width <= 0.0 or height <= 0.0:
        return None
    distance = float(distance_m)
    tan_h = math.tan(math.radians(hfov) / 2.0)
    tan_v = tan_h * (height / width)
    implied_w = 2.0 * distance * tan_h * max(0.0, x2 - x1) / 1000.0
    implied_h = 2.0 * distance * tan_v * max(0.0, y2 - y1) / 1000.0
    return {
        "width_m": round(implied_w, 3),
        "height_m": round(implied_h, 3),
        "largest_m": round(max(implied_w, implied_h), 3),
    }


def _size_conflict(candidate: Mapping, intrinsics: Mapping) -> dict | None:
    """Range vs the size the same box says the object has.

    A *converged depth* reading is a measurement of the surface the box sits on:
    the box may include a support surface, so this model-vs-geometry check does
    not override it. A triangulated range is not exempt: two bearings can agree
    on a point while at least one box also covers the wall behind the object
    (final3 classroom seed37 accepted a 23 m "bookshelf" this way), so the box
    still has to be plausible at the range it was measured at.
    """
    if str(candidate.get("distance_source") or "") == DISTANCE_DEPTH_CONSISTENT:
        return None
    distance = candidate.get("range_m")
    if str(candidate.get("distance_source") or "") == DISTANCE_TRIANGULATED:
        triangulation = candidate.get("triangulation")
        if isinstance(triangulation, Mapping):
            distance = triangulation.get("range_a_m", distance)
    if distance is None:
        distance = candidate.get("distance_m")
    implied = _implied_box_size(candidate.get("box"), distance, intrinsics)
    if implied is None:
        return None
    claimed = _as_float(candidate.get("size_m"), None)
    if claimed is not None and claimed > 0.0:
        ratio = implied["largest_m"] / claimed
        if ratio > INSTANCE_SIZE_RATIO or ratio < 1.0 / INSTANCE_SIZE_RATIO:
            return {
                "kind": "distance_size",
                "implied_m": implied["largest_m"],
                "claimed_m": round(claimed, 3),
                "ratio": round(max(ratio, 1.0 / ratio), 2),
            }
    prior = _target_size_prior(str(candidate.get("instruction") or ""))
    if prior is not None:
        prior_w, _prior_h = prior
        factor = 6.0 if prior_w < 0.2 else INSTANCE_SIZE_RATIO
        if implied["width_m"] > prior_w * factor:
            return {
                "kind": "distance_size_prior",
                "implied_width_m": implied["width_m"],
                "prior_width_m": prior_w,
                "ratio": round(implied["width_m"] / prior_w, 2),
            }
    return None


def _two_view_size_conflict(candidate: Mapping, intrinsics: Mapping) -> dict | None:
    """Two triangulating boxes must imply the same physical object.

    A triangulated point is only as good as the two boxes it came from. Each
    view's box is converted to physical width/height at that view's own
    triangulated range; the same object gives the same numbers, while a box
    that also covers the wall/background gives a much larger one. The check is
    one-sided where a box touches the frame border: clipping can only make a
    box under-state its object, so a ratio is decisive when the larger implied
    size comes from a clipped box (a lower bound already too large) or when
    neither box is clipped; it is skipped when the *smaller* side is clipped
    (its true size, and therefore the ratio, is unknown).
    """
    triangulation = candidate.get("triangulation")
    if not isinstance(triangulation, Mapping):
        return None
    range_a = _as_float(
        triangulation.get("range_a_m"), _as_float(candidate.get("range_m"), None)
    )
    range_b = _as_float(triangulation.get("range_b_m"), None)
    merged_box = triangulation.get("merged_box")
    if range_a is None or range_b is None or merged_box is None:
        return None
    current = _implied_box_size(candidate.get("box"), range_a, intrinsics)
    other = _implied_box_size(merged_box, range_b, intrinsics)
    if current is None or other is None:
        return None
    ratios = {
        "width": max(
            current["width_m"] / max(other["width_m"], 1e-6),
            other["width_m"] / max(current["width_m"], 1e-6),
        ),
        "height": max(
            current["height_m"] / max(other["height_m"], 1e-6),
            other["height_m"] / max(current["height_m"], 1e-6),
        ),
    }
    dimension, ratio = max(ratios.items(), key=lambda item: item[1])
    if ratio <= INSTANCE_TWO_VIEW_SIZE_RATIO:
        return None
    current_box = parse_box(candidate.get("box"), box_format=BOX_FORMAT_XYXY)[0]
    other_parsed = parse_box(merged_box, box_format=BOX_FORMAT_XYXY)[0]
    clipped = {
        "current": bool(current_box is not None and _box_geometry(current_box)["border_touching"]),
        "merged": bool(other_parsed is not None and _box_geometry(other_parsed)["border_touching"]),
    }
    larger = "current" if current[dimension + "_m"] >= other[dimension + "_m"] else "merged"
    smaller = "merged" if larger == "current" else "current"
    if clipped[smaller] and not clipped[larger]:
        # the smaller box is a lower bound: the true ratio may fall below the gate
        return None
    return {
        "kind": "two_view_size",
        "dimension": dimension,
        "ratio": round(ratio, 2),
        "limit": INSTANCE_TWO_VIEW_SIZE_RATIO,
        "current_m": round(current[dimension + "_m"], 3),
        "merged_m": round(other[dimension + "_m"], 3),
        "range_a_m": round(float(range_a), 3),
        "range_b_m": round(float(range_b), 3),
        "clipped": clipped,
    }


def _relation_named(instruction: str) -> bool:
    text = str(instruction or "").lower()
    return any(token in text for token in _RELATION_TOKENS)


def _instance_point(value) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        point = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None
    return point if all(math.isfinite(item) for item in point) else None


def _relation_anchor(instruction: str) -> dict | None:
    """The relatum an instruction names ("书架" in "书架旁的课桌"), if any."""
    text = str(instruction or "").strip()
    if not text:
        return None
    match = _CJK_RELATION_RE.search(text)
    if match is not None:
        anchor = _ANCHOR_PREFIX_RE.sub("", match.group("anchor")).strip(" 的")
        if anchor:
            relation = match.group("rel")
            limit = (
                INSTANCE_NEAR_MAX_M
                if relation in ("附近的", "附近")
                else INSTANCE_ADJACENT_MAX_M
            )
            return {"anchor": anchor[:24], "relation": relation, "limit_m": limit}
    match = _EN_RELATION_RE.search(text)
    if match is not None:
        anchor = match.group("anchor").strip(" ,.;")
        anchor = re.sub(r"^(?:the|a|an)\s+", "", anchor, flags=re.IGNORECASE)
        if anchor:
            relation = " ".join(match.group("rel").lower().split())
            limit = (
                INSTANCE_NEAR_MAX_M
                if relation == "near"
                else INSTANCE_ADJACENT_MAX_M
            )
            return {"anchor": anchor[:40], "relation": relation, "limit_m": limit}
    return None


def _anchor_category(text: str) -> str:
    """Coarse object category of a relation phrase or a candidate record."""
    lowered = str(text or "").lower()
    for category, tokens in _ANCHOR_NOUNS:
        if any(token.lower() in lowered for token in tokens):
            return category
    return ""


def _anchor_record_category(record: Mapping) -> str:
    """Category of the instance a record grounded (its target, not its anchor).

    A record's instruction can name both its own target and its relation anchor
    ("书架旁的课桌"), so the target phrase leads and attributes are the fallback.
    """
    instruction = str(record.get("instruction") or record.get("target") or "")
    phrase = _target_phrase(instruction) if instruction else ""
    attrs = record.get("attrs")
    attr_text = (
        " ".join(str(value) for value in attrs.values())
        if isinstance(attrs, Mapping)
        else ""
    )
    # the instruction's head phrase leads: an attribute such as the colour
    # "light beige" must not turn a curtained window into a lamp ("light")
    category = _anchor_category(phrase)
    if category:
        return category
    category = _anchor_category(attr_text)
    if category:
        return category
    return _anchor_category(instruction) if not phrase else ""


def _matching_anchors(anchors: list[dict], relation: Mapping | None) -> list[dict]:
    """Known candidates that are the relation's relatum, nearest-agnostic."""
    if relation is None or not anchors:
        return []
    category = _anchor_category(str(relation.get("anchor") or ""))
    if category:
        return [anchor for anchor in anchors if anchor.get("category") == category]
    phrase = str(relation.get("anchor") or "").casefold()
    if not phrase:
        return []
    return [
        anchor
        for anchor in anchors
        if phrase in str(anchor.get("text") or "").casefold()
    ]


def _anchor_geometry(
    ctx: ToolContext, record: PerceptionObservation, anchor: Mapping
) -> dict | None:
    """Project a known anchor position into this frame's field of view.

    ``None`` means the geometry could not be evaluated (unknown pose/intrinsics);
    otherwise ``visible`` says whether the anchor lies in front of the camera
    and inside the frame (or the crop region) used for this grounding.
    """
    position = record.position_flu
    yaw = record.yaw_rad
    point = _instance_point(anchor.get("position")) if isinstance(anchor, Mapping) else None
    if position is None or yaw is None or point is None:
        return None
    intrinsics = _observation_intrinsics(ctx, record)
    hfov = _as_float(intrinsics.get("hfov_deg"), None)
    width = int(intrinsics.get("width") or 0)
    height = int(intrinsics.get("height") or 0)
    if hfov is None or width <= 0 or height <= 0:
        return None
    dx = point[0] - float(position[0])
    dy = point[1] - float(position[1])
    dz = point[2] - float(position[2])
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    bx = cos_yaw * dx + sin_yaw * dy
    by = -sin_yaw * dx + cos_yaw * dy
    bz = dz
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    half_h = math.radians(hfov) / 2.0
    half_v = math.atan(math.tan(half_h) * height / width)
    margin = math.radians(ANCHOR_VISIBILITY_MARGIN_DEG)
    bearing = math.atan2(by, bx)
    elevation = math.atan2(bz, math.hypot(bx, by))
    detail = {
        "anchor_m": round(distance, 2),
        "bearing_deg": round(math.degrees(bearing), 1),
        "elevation_deg": round(math.degrees(elevation), 1),
        "hfov_deg": round(hfov, 1),
    }
    if bx <= 0.05:
        return {"visible": False, "reason": "behind_camera", **detail}
    if distance > ANCHOR_VISIBILITY_MAX_M:
        return {"visible": False, "reason": "beyond_view_range", **detail}
    pitch = math.radians(
        float((record.metadata or {}).get("pitch_down_deg") or 0.0)
    )
    if abs(bearing) > half_h + margin or abs(elevation + pitch) > half_v + margin:
        return {"visible": False, "reason": "outside_fov", **detail}
    region = record.region
    if region:
        try:
            ru, rv, rw, rh = (float(value) for value in list(region)[:4])
            u = 500.0 - 500.0 * (by / bx) / math.tan(half_h)
            v = 500.0 - 500.0 * (bz / bx) / math.tan(half_v)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
        else:
            detail.update({"u": round(u, 1), "v": round(v, 1)})
            if not (
                ru - 10.0 <= u <= ru + rw + 10.0
                and rv - 10.0 <= v <= rv + rh + 10.0
            ):
                return {"visible": False, "reason": "outside_crop", **detail}
    return {"visible": True, "reason": "in_view", **detail}


def _anchor_visibility(
    ctx: ToolContext,
    record: PerceptionObservation,
    candidate: Mapping,
    anchor: Mapping | None,
) -> dict:
    """Whether the relation anchor is visible in the candidate's evidence.

    The model's ``anchor_visible`` answer and the geometric projection are two
    independent observations; the anchor is visible when either says so, and
    not visible only when the model says it is not visible and the geometry
    cannot place it in the frame. An unverifiable relation is not approved.
    """
    model_flag = candidate.get("anchor_visible")
    model_flag = model_flag if isinstance(model_flag, bool) else None
    geometry = _anchor_geometry(ctx, record, anchor) if isinstance(anchor, Mapping) else None
    geometric = None if geometry is None else bool(geometry.get("visible"))
    if model_flag is True:
        visible, source = True, "model"
    elif geometric is True:
        visible, source = True, "geometry"
    elif model_flag is False:
        visible, source = False, "model"
    elif geometric is False:
        visible, source = False, "geometry"
    else:
        visible, source = None, "unknown"
    return {
        "visible": visible,
        "source": source,
        "model_anchor_visible": model_flag,
        "anchor_candidate": (
            None if not isinstance(anchor, Mapping) else str(anchor.get("id") or "")
        ),
        "geometric": geometry,
    }


def _anchor_position_trusted(record: Mapping) -> bool:
    """Whether an anchor's position is a measurement, not an estimate.

    A border-large window grounded by the model's own distance is not a
    geometric fact: it cannot prove a candidate 4-5 m away wrong. Such an
    anchor may still support a relation when a confirmation view shows both
    objects, but its distance is ``unknown`` for the reject test.
    """
    if record.get("distance_uncertain") is True:
        return False
    if str(record.get("distance_source") or "") == "conflict":
        return False
    if record.get("border_large_box"):
        return False
    source = str(record.get("position_source") or "")
    # a legacy record that does not say how it was placed is treated as a
    # measurement; an explicit model estimate or ray is not
    return source in ("", POSITION_DEPTH, POSITION_TRIANGULATED)


def _anchor_ray_triangulation(anchors: list[dict]) -> dict | None:
    """The best ray pair of one anchor group that intersects with real parallax.

    A ray-only anchor is a bearing from a known pose. Two bearings from
    different poses intersect at the anchor when the baseline and the angle
    between them are large enough (the same gates the candidate triangulation
    uses); the closest-approach point is then a measured anchor position.
    ``None`` when no pair has parallax (rotation alone does not).
    """
    rays = [
        anchor for anchor in anchors
        if anchor.get("ray_only") and isinstance(anchor.get("ray"), Mapping)
    ]
    best: tuple[dict, dict, dict] | None = None
    for first in range(len(rays)):
        for second in range(first + 1, len(rays)):
            combined = _triangulate_rays(rays[first]["ray"], rays[second]["ray"])
            if combined is None:
                continue
            if best is None or combined["residual_m"] < best[0]["residual_m"]:
                best = (combined, rays[first], rays[second])
    if best is None:
        return None
    combined, left, right = best
    return {
        **combined,
        "anchor_ids": [str(left.get("id") or ""), str(right.get("id") or "")],
        "sightings": len(rays),
        "min_baseline_m": TRIANGULATION_MIN_BASELINE_M,
        "min_angle_deg": TRIANGULATION_MIN_ANGLE_DEG,
    }


def _known_anchors(ctx: ToolContext, subtask_key: str) -> list[dict]:
    """Earlier-stage candidates usable as relation anchors.

    A measured (depth/triangulated) position is the normal anchor. A candidate
    grounded with a bearing only (``position_source=ray``) is kept as a
    *ray-only* anchor: it cannot place the relatum by itself, but two such
    bearings from different poses triangulate it. The triangulated point is
    appended as a located anchor so every later relation check re-runs against
    a real measurement.
    """
    anchors: list[dict] = []
    for key, record in _candidate_store(ctx).items():
        if not isinstance(record, Mapping):
            continue
        if str(record.get("subtask_key") or "") == str(subtask_key or ""):
            continue
        if record.get("instance_suspect") or record.get("distance_uncertain"):
            continue
        if str(record.get("status") or "") in ("rejected", "merged", "superseded"):
            continue
        point = _instance_point(record.get("position_flu"))
        ray = record.get("ray") if isinstance(record.get("ray"), Mapping) else None
        if point is None and ray is None:
            continue
        attrs = record.get("attrs")
        text = " ".join(
            [
                str(record.get("instruction") or ""),
                " ".join(str(value) for value in attrs.values())
                if isinstance(attrs, Mapping)
                else "",
            ]
        ).strip()
        anchors.append(
            {
                "id": str(record.get("id") or key),
                "position": point,
                "ray": None if ray is None else dict(ray),
                "ray_only": point is None and ray is not None,
                "category": _anchor_record_category(record),
                "text": text[:200],
                "position_source": str(record.get("position_source") or ""),
                "distance_uncertain": bool(record.get("distance_uncertain")),
                "border_large_box": bool(record.get("border_large_box")),
                "trusted": (
                    _anchor_position_trusted(record) if point is not None else False
                ),
            }
        )
    groups: dict[str, list[dict]] = {}
    for anchor in anchors:
        if anchor.get("ray_only"):
            groups.setdefault(
                str(anchor.get("category") or anchor.get("text") or ""), []
            ).append(anchor)
    for group in groups.values():
        if len(group) < 2:
            continue
        triangulated = _anchor_ray_triangulation(group)
        if triangulated is None:
            continue
        source = group[0]
        anchors.append(
            {
                "id": f"anchor_triangulated_{len(anchors) + 1}",
                "position": tuple(float(value) for value in triangulated["point"]),
                "ray": None,
                "ray_only": False,
                "category": source.get("category") or "",
                "text": source.get("text") or "",
                "position_source": POSITION_TRIANGULATED,
                "distance_uncertain": False,
                "border_large_box": False,
                "trusted": True,
                "triangulated": {
                    key: triangulated[key]
                    for key in (
                        "baseline_m", "angle_deg", "residual_m",
                        "anchor_ids", "sightings",
                    )
                },
            }
        )
    return anchors


def _anchor_parallax_requirement(
    relation: Mapping | None, anchors: list[dict],
) -> dict:
    """What would locate a ray-only anchor: a second bearing with a baseline.

    One bearing is a line, not a point. The requirement is explicit so the
    relation is not judged on an invented range: move sideways, look at the
    anchor again from the new pose, and the two rays intersect at the anchor.
    """
    phrase = str((relation or {}).get("anchor") or "")
    instruction = f"找到并定位{phrase}" if phrase else ""
    sightings = [
        str(anchor.get("id") or "")
        for anchor in anchors
        if anchor.get("ray_only")
    ]
    return {
        "required": True,
        "reason": "anchor_ray_only",
        "anchor": phrase,
        "anchor_candidates": sightings[-8:],
        "sightings": len(sightings),
        "min_baseline_m": TRIANGULATION_MIN_BASELINE_M,
        "min_angle_deg": TRIANGULATION_MIN_ANGLE_DEG,
        "why": (
            "the anchor was grounded with a bearing and no range: one ray "
            f"cannot place it. Take a second bearing from at least "
            f"{TRIANGULATION_MIN_BASELINE_M:g} m to the side (a new pose, not "
            "only a new heading); the two rays intersect at the anchor and "
            "the relation is re-checked against that measured point"
        ),
        "action": {
            "tool": "ground_target",
            "args": {
                "instruction": instruction,
                "require_parallax": True,
                "min_baseline_m": TRIANGULATION_MIN_BASELINE_M,
            },
            "observe_first": {
                "tool": "propose_viewpoints",
                "args": {"count": 2},
            },
        },
    }


def _anchor_conflict(
    instruction: str,
    candidate: Mapping,
    anchors: list[dict],
    relation: Mapping | None = None,
) -> dict | None:
    """Distance from the candidate to its relatum, against the relation's bound.

    When the relation names a known anchor ("书架"), only that anchor's distance
    is judged, and adjacency (3 m) or nearness (4 m) sets the bound. Without a
    matching known anchor the older, weak any-anchor bound still catches a
    candidate that landed across the room from every recorded instance.
    """
    if not anchors:
        return None
    relation = relation if relation is not None else _relation_anchor(instruction)
    matched = _matching_anchors(anchors, relation)
    pool = matched or anchors
    # only measured anchor positions may contradict a candidate: a border-large
    # window placed at the model's estimated range is not a geometric fact
    pool = [anchor for anchor in pool if anchor.get("trusted", True)]
    if not pool:
        return None
    limit = (
        float(relation["limit_m"])
        if relation is not None and matched
        else INSTANCE_ANCHOR_MAX_M
    )
    point = _instance_point(candidate.get("position_flu"))
    if point is None:
        return None
    nearest = min(
        pool,
        key=lambda anchor: math.dist(point, anchor["position"]),
    )
    distance = math.dist(point, nearest["position"])
    if distance <= limit:
        return None
    conflict = {
        "kind": "anchor_distance",
        "anchor_candidate": nearest["id"],
        "distance_m": round(distance, 2),
        "limit_m": limit,
    }
    if relation is not None:
        conflict["relation"] = relation["relation"]
        conflict["anchor_phrase"] = relation["anchor"]
        conflict["anchor_matched"] = bool(matched)
    return conflict


# ---------------------------------------------------------------------------
# relation chain: instance hypotheses and typed relation evidence
# ---------------------------------------------------------------------------
#
# The relation gate used to be stateless: a candidate of "窗边的木椅" was only
# approved when the window happened to be inside the *same* frame, and a fresh
# confirmation view that did not hold the window vetoed the candidate. That is
# wrong in both directions - "the window is not in this frame" is not "the
# chair is not by the window" - and it is exactly what stopped the bedroom
# relation chains. This block keeps the two records apart:
#
# - *instance hypotheses*: one entity per instructed target, with the candidate
#   ids that were associated to it (same target phrase, compatible attributes,
#   a position inside the accumulated uncertainty) and its observation set;
# - *relation evidence*: (anchor entity, relation type, source, confidence,
#   timestamp). A relation can be established while both objects are visible
#   (``co_visible``) or from the located anchor's world position
#   (``cross_frame_pose``); losing the anchor from the frame only makes the
#   *current* view ``unknown`` - the historical relation stays valid until an
#   explicit contradiction (the candidate measured outside the relation bound,
#   or the anchor seen in frame without the relation) or a freshness failure.
#
# Every consumer keeps three verdicts apart: ``supports`` / ``contradicts`` /
# ``unknown``. Only ``contradicts`` can reject a candidate; ``unknown`` keeps it
# and names the observation that would settle the question.

RELATION_CHAIN_KEY = "relation_chain.state"
RELATION_CHAIN_SWITCH = "relation_chain"
RELATION_SCHEMA = "relation-chain-v1"

RELATION_SUPPORTS = "supports"
RELATION_CONTRADICTS = "contradicts"
RELATION_UNKNOWN = "unknown"

RELATION_SOURCE_CO_VISIBLE = "co_visible"
RELATION_SOURCE_CROSS_FRAME = "cross_frame_pose"
RELATION_SOURCE_MODEL = "model_self_report"
RELATION_SOURCE_NONE = "none"

UNCERTAINTY_CATEGORY = "category"
UNCERTAINTY_INSTANCE = "instance"
UNCERTAINTY_DISTANCE = "distance"
UNCERTAINTY_ANCHOR_RELATION = "anchor_relation"
UNCERTAINTY_REACHABILITY = "reachability"
UNCERTAINTY_TYPES = (
    UNCERTAINTY_CATEGORY,
    UNCERTAINTY_INSTANCE,
    UNCERTAINTY_DISTANCE,
    UNCERTAINTY_ANCHOR_RELATION,
    UNCERTAINTY_REACHABILITY,
)

#: Two sightings are the same instance when their positions agree inside the
#: accumulated uncertainty plus this slack. Different frames of one chair from
#: a few metres apart stay one hypothesis; two chairs across the room do not.
RELATION_ASSOCIATION_RADIUS_M = 1.5
#: How many competing hypotheses of one target stay active (goal.md section 4).
RELATION_MAX_COMPETING = 3
#: A re-grounded anchor farther than this from the recorded anchor position is
#: an anchor re-localization conflict (unknown, never by itself a rejection).
RELATION_ANCHOR_RELOC_TOL_M = 1.5
#: A *perceptual* relation (model self-report, no located anchor) goes stale
#: after this much episode time; a geometric relation to a located anchor does
#: not (the world points do not move with the drone).
RELATION_STALE_AFTER_S = 180.0
#: A perceptual relation established from a pose farther than this from the
#: current one is no longer trusted without re-observation.
RELATION_POSE_DRIFT_MAX_M = 8.0
#: A relation is only used for a decision when its evidence reached this.
RELATION_MIN_CONFIDENCE = 0.35
#: Bounded observation-decision journal (one entry per supplementary look).
MAX_RELATION_OBSERVATIONS = 48
#: Bounded per-instance observation set.
MAX_INSTANCE_OBSERVATIONS = 12


def relation_chain_enabled(ctx: ToolContext) -> bool:
    """The relation-chain switch (``ctx.extra["relation_chain"]``).

    Absent means off: a bare ``ToolContext`` (unit tests, older callers) keeps
    the previous co-visibility gate. The runtime publishes the switch from its
    own constructor (``VLM_NAV_RELATION_CHAIN`` by default), so an experiment
    arm can enable it without changing any tool signature.
    """
    extra = _extra(ctx)
    value = extra.get(RELATION_CHAIN_SWITCH)
    if isinstance(value, Mapping):
        value = value.get("enabled", True)
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in ("on", "true", "1", "yes")
    return bool(value)


def relation_state(ctx: ToolContext) -> dict:
    """The episode's relation-chain record (created on first use)."""
    extra = _extra(ctx)
    state = extra.get(RELATION_CHAIN_KEY)
    if not isinstance(state, dict) or state.get("schema") != RELATION_SCHEMA:
        state = {
            "schema": RELATION_SCHEMA,
            "instances": {},
            "relations": {},
            "competitors": {},
            "observations": [],
            "seq": 0,
            "counts": {name: 0 for name in UNCERTAINTY_TYPES},
        }
        extra[RELATION_CHAIN_KEY] = state
    state.setdefault("instances", {})
    state.setdefault("relations", {})
    state.setdefault("competitors", {})
    state.setdefault("observations", [])
    counts = state.setdefault("counts", {})
    for name in UNCERTAINTY_TYPES:
        counts.setdefault(name, 0)
    return state


def _relation_now(ctx: ToolContext) -> float:
    try:
        return float(_episode_time_s(ctx))
    except Exception:  # noqa: BLE001 - the journal must never break a tool
        return 0.0


def _candidate_position_sigma(candidate: Mapping) -> float:
    """Position uncertainty of one candidate, in metres (rough on purpose)."""
    if candidate.get("distance_uncertain") is True:
        return 2.0
    if str(candidate.get("distance_source") or "") == "conflict":
        return 2.0
    if _instance_point(candidate.get("position_flu")) is None:
        return 2.5
    source = str(candidate.get("position_source") or "")
    return 0.5 if source in (POSITION_DEPTH, POSITION_TRIANGULATED, "depth") else 1.2


def _attrs_compatible(first: Mapping | None, second: Mapping | None) -> bool:
    """No shared attribute key carries two different values."""
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        return True
    for key in set(first) & set(second):
        value_a = str(first.get(key) or "").strip().lower()
        value_b = str(second.get(key) or "").strip().lower()
        if value_a and value_b and value_a != value_b:
            return False
    return True


def _merge_attrs(target: dict, extra: Mapping | None) -> None:
    if not isinstance(extra, Mapping):
        return
    for key, value in extra.items():
        text = str(value or "").strip()
        if text and not str(target.get(key) or "").strip():
            target[str(key)] = text[:120]


def _record_value(record: Any, name: str, default: Any = None) -> Any:
    """One field of a :class:`PerceptionObservation` or of a candidate dict."""
    if record is None:
        return default
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _instance_observation(candidate: Mapping, record: Any) -> dict:
    position = _instance_point(candidate.get("position_flu"))
    pose = _record_value(record, "position_flu")
    yaw = _record_value(record, "yaw_deg")
    if yaw is None:
        yaw_rad = _record_value(record, "yaw_rad")
        yaw = None if yaw_rad is None else math.degrees(float(yaw_rad))
    return {
        "candidate_id": str(candidate.get("id") or ""),
        "observation_id": str(candidate.get("observation_id") or ""),
        "position_flu": None if position is None else list(position),
        "sigma_m": round(_candidate_position_sigma(candidate), 3),
        "confidence": _as_float(candidate.get("confidence"), None),
        "pose": (
            None
            if pose is None
            else [round(float(value), 3) for value in list(pose)[:3]]
        ),
        "yaw_deg": None if yaw is None else round(float(yaw), 1),
        "at": _as_float(_record_value(record, "time_s"), None),
    }


def _new_instance(state: dict, candidate: Mapping, instruction: str) -> dict:
    state["seq"] = int(state.get("seq") or 0) + 1
    instance_id = f"inst_{state['seq']}"
    point = _instance_point(candidate.get("position_flu"))
    instance = {
        "id": instance_id,
        "target_key": _candidate_target_key(candidate),
        "instruction": str(instruction or candidate.get("instruction") or "")[:160],
        "candidate_ids": [],
        "observation_ids": [],
        "observations": [],
        "attrs": {},
        "position_flu": None if point is None else list(point),
        "sigma_m": round(_candidate_position_sigma(candidate), 3),
        "n_positions": 0 if point is None else 1,
        "support": 0,
        "confidence": _as_float(candidate.get("confidence"), 0.0) or 0.0,
        "status": "hypothesis",
        "rank": None,
        "first_seen_at": _as_float(candidate.get("_relation_seen_at"), 0.0) or 0.0,
        "last_seen_at": _as_float(candidate.get("_relation_seen_at"), 0.0) or 0.0,
    }
    state["instances"][instance_id] = instance
    return instance


def _update_instance(instance: dict, candidate: Mapping, record: PerceptionObservation | None) -> None:
    candidate_id = str(candidate.get("id") or "")
    if candidate_id and candidate_id not in instance["candidate_ids"]:
        instance["candidate_ids"].append(candidate_id)
    observation_id = str(candidate.get("observation_id") or "")
    if observation_id and observation_id not in instance["observation_ids"]:
        instance["observation_ids"].append(observation_id)
    observation = _instance_observation(candidate, record)
    instance["observations"].append(observation)
    del instance["observations"][:-MAX_INSTANCE_OBSERVATIONS]
    point = _instance_point(candidate.get("position_flu"))
    if point is not None:
        count = int(instance.get("n_positions") or 0)
        previous = instance.get("position_flu")
        if isinstance(previous, (list, tuple)) and len(previous) >= 3 and count > 0:
            weight = min(count, 8)
            instance["position_flu"] = [
                round((float(previous[index]) * weight + point[index]) / (weight + 1), 4)
                for index in range(3)
            ]
        else:
            instance["position_flu"] = list(point)
        instance["n_positions"] = count + 1
        instance["sigma_m"] = round(
            min(float(instance.get("sigma_m") or 2.5), _candidate_position_sigma(candidate)),
            3,
        )
    _merge_attrs(instance.setdefault("attrs", {}), candidate.get("attrs"))
    instance["support"] = int(instance.get("support") or 0) + 1
    instance["confidence"] = max(
        float(instance.get("confidence") or 0.0),
        _as_float(candidate.get("confidence"), 0.0) or 0.0,
    )
    seen = _as_float(candidate.get("_relation_seen_at"), None)
    if seen is not None:
        instance["last_seen_at"] = seen


def _associate_instance(
    state: dict,
    ctx: ToolContext,
    candidate: Mapping,
    instruction: str,
    record: PerceptionObservation | None,
) -> dict:
    """The instance hypothesis a candidate belongs to (new one when none fits).

    Association is deliberately conservative: the target phrase must agree
    (token overlap plus the CJK head noun, :func:`_same_target_instruction`),
    the attributes must not contradict, and the positions must agree inside the
    accumulated uncertainty. Two distinct boxes on the same frame are *never*
    merged (they are competing hypotheses); a position outside the radius is a
    second hypothesis, not a correction of the first.
    """
    target_key = _candidate_target_key(candidate)
    point = _instance_point(candidate.get("position_flu"))
    attrs = candidate.get("attrs") if isinstance(candidate.get("attrs"), Mapping) else {}
    observation_id = str(candidate.get("observation_id") or "")
    candidate_id = str(candidate.get("id") or "")
    best: tuple[float, dict] | None = None
    for instance in state["instances"].values():
        if instance.get("status") == "rejected":
            continue
        same_target = instance.get("target_key") == target_key
        if not same_target:
            same_target = _same_target_instruction(
                str(instance.get("instruction") or ""), instruction
            )
        if not same_target:
            continue
        if not _attrs_compatible(instance.get("attrs"), attrs):
            continue
        if candidate_id and candidate_id in instance.get("candidate_ids", []):
            return instance
        if observation_id and observation_id in instance.get("observation_ids", []):
            # the same frame grounds the same object only when the boxes overlap;
            # a second, disjoint box on that frame is another hypothesis
            continue
        instance_point = _instance_point(instance.get("position_flu"))
        if point is not None and instance_point is not None:
            gap = math.dist(point, instance_point)
            radius = (
                RELATION_ASSOCIATION_RADIUS_M
                + float(instance.get("sigma_m") or 0.0)
                + _candidate_position_sigma(candidate)
            )
            if gap <= radius and (best is None or gap < best[0]):
                best = (gap, instance)
        elif point is None and instance_point is None and same_target:
            best = (0.0, instance)
    if best is not None:
        _update_instance(best[1], candidate, record)
        return best[1]
    instance = _new_instance(state, candidate, instruction)
    _update_instance(instance, candidate, record)
    return instance


def _refresh_competitors(state: dict, target_key: str) -> list[str]:
    """Rank the hypotheses of one target; only the best few stay active."""
    active = [
        instance
        for instance in state["instances"].values()
        if instance.get("target_key") == target_key
        and instance.get("status") != "rejected"
    ]
    active.sort(
        key=lambda item: (
            float(item.get("confidence") or 0.0),
            int(item.get("support") or 0),
        ),
        reverse=True,
    )
    for rank, instance in enumerate(active, 1):
        instance["rank"] = rank if rank <= RELATION_MAX_COMPETING else None
        instance["competing"] = len(active) > 1
    selected = [str(instance["id"]) for instance in active[:RELATION_MAX_COMPETING]]
    state["competitors"][target_key] = selected
    return selected


def update_instance_hypotheses(
    ctx: ToolContext,
    record: PerceptionObservation | None,
    instruction: str,
    candidates: list[dict],
) -> dict:
    """Associate every grounded candidate with an instance hypothesis.

    Returns a compact summary the caller publishes (the candidates themselves
    carry ``instance_hypothesis``), and records the coordinate for the relation
    evidence layer. Safe to call on every grounding: a candidate already in an
    instance keeps it.
    """
    if not relation_chain_enabled(ctx) or not candidates:
        return {"enabled": False, "n_instances": 0}
    state = relation_state(ctx)
    now = _relation_now(ctx)
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        candidate.setdefault("_relation_seen_at", now)
        instance = _associate_instance(state, ctx, candidate, instruction, record)
        candidate["instance_hypothesis"] = str(instance["id"])
        candidate["instance_rank"] = instance.get("rank")
        # an anchor candidate is also an instance: the relation layer matches it
        # by category later, and a re-localized anchor is judged against the
        # position the relation was established with
        category = _anchor_record_category(candidate)
        if category:
            candidate.setdefault("instance_category", category)
            _note_anchor_reobservation(ctx, state, candidate, category, now)
    target_keys = sorted(
        {str(candidate.get("subtask_key") or "") or _candidate_target_key(candidate)
         for candidate in candidates if isinstance(candidate, dict)}
    )
    for key in target_keys:
        _refresh_competitors(state, key)
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        instance = state["instances"].get(str(candidate.get("instance_hypothesis") or ""))
        if instance is not None:
            candidate["instance_rank"] = instance.get("rank")
            candidate["competing_hypothesis"] = bool(instance.get("competing"))
    return {
        "enabled": True,
        "n_instances": len(state["instances"]),
        "instances": [
            str(instance.get("id")) for instance in state["instances"].values()
        ][-8:],
        "competitors": dict(state.get("competitors") or {}),
    }


def _note_anchor_reobservation(
    ctx: ToolContext,
    state: dict,
    candidate: Mapping,
    category: str,
    now: float,
) -> None:
    """Check a newly grounded anchor against the located anchor of each relation.

    A re-localized anchor that lands far from the recorded one makes the
    historical relation *unknown* (it needs re-checking); it never, by itself,
    proves the candidate wrong. A re-observation inside the tolerance refreshes
    the anchor position and the confidence.
    """
    point = _instance_point(candidate.get("position_flu"))
    if point is None:
        return
    candidate_id = str(candidate.get("id") or "")
    for relation in state["relations"].values():
        if str(relation.get("anchor_category") or "") != category:
            continue
        if candidate_id and candidate_id in relation.get("anchor_candidates", []):
            continue
        anchor_position = _instance_point(relation.get("anchor_position"))
        if anchor_position is None:
            continue
        gap = math.dist(point, anchor_position)
        relation.setdefault("anchor_candidates", []).append(candidate_id)
        del relation["anchor_candidates"][:-8]
        relation["anchor_last_seen_at"] = now
        if gap > RELATION_ANCHOR_RELOC_TOL_M:
            relation["anchor_reobservation_conflict"] = {
                "candidate_id": candidate_id,
                "gap_m": round(gap, 2),
                "tolerance_m": RELATION_ANCHOR_RELOC_TOL_M,
                "at": now,
            }
            relation["last_unknown_reason"] = "anchor_relocalized"
        else:
            # the anchor is confirmed: refresh the position (running mean) and
            # clear an older re-localization conflict when this view resolves it
            count = max(1, int(relation.get("anchor_observations") or 1))
            relation["anchor_position"] = [
                round((float(anchor_position[index]) * count + point[index]) / (count + 1), 4)
                for index in range(3)
            ]
            relation["anchor_observations"] = count + 1
            relation.pop("anchor_reobservation_conflict", None)


def _relation_key(instance_id: str, relation: Mapping) -> str:
    category = _anchor_category(str(relation.get("anchor") or ""))
    kind = str(relation.get("relation") or "near").strip().lower()
    return f"{instance_id}|{category or str(relation.get('anchor') or '')}|{kind}"


def _relation_record(
    state: dict,
    ctx: ToolContext,
    instance: Mapping,
    relation: Mapping,
    *,
    anchor_position: tuple[float, float, float] | None = None,
    anchor_candidate: str | None = None,
    anchor_category: str | None = None,
) -> dict:
    key = _relation_key(str(instance.get("id") or ""), relation)
    record = state["relations"].get(key)
    if isinstance(record, dict):
        if anchor_position is not None and record.get("anchor_position") is None:
            record["anchor_position"] = list(anchor_position)
        if anchor_candidate and anchor_candidate not in record.setdefault("anchor_candidates", []):
            record.setdefault("anchor_candidates", []).append(anchor_candidate)
        return record
    record = {
        "id": f"rel_{len(state['relations']) + 1}",
        "target_instance": str(instance.get("id") or ""),
        "target_candidates": list(instance.get("candidate_ids") or []),
        "target_key": str(instance.get("target_key") or ""),
        "anchor_phrase": str(relation.get("anchor") or "")[:40],
        "anchor_category": anchor_category or _anchor_category(str(relation.get("anchor") or "")),
        "relation": str(relation.get("relation") or "")[:24],
        "limit_m": float(relation.get("limit_m") or INSTANCE_ADJACENT_MAX_M),
        "anchor_position": None if anchor_position is None else list(anchor_position),
        "anchor_candidates": [] if not anchor_candidate else [anchor_candidate],
        "anchor_observations": 1 if anchor_position is not None else 0,
        "state": RELATION_UNKNOWN,
        "source": RELATION_SOURCE_NONE,
        "confidence": 0.0,
        "created_at": _relation_now(ctx),
        "last_supported_at": None,
        "last_checked_at": _relation_now(ctx),
        "last_unknown_reason": "not_established",
        "evidence_ids": [],
        "contradictions": [],
    }
    state["relations"][key] = record
    return record


def _write_relation(
    ctx: ToolContext,
    state: dict,
    instance: Mapping,
    relation: Mapping,
    *,
    verdict: str,
    source: str,
    reason: str,
    confidence: float,
    anchor_position: tuple[float, float, float] | None = None,
    anchor_candidate: str | None = None,
    anchor_category: str | None = None,
    evidence_ids: list[str] | None = None,
    distance_m: float | None = None,
) -> dict:
    record = _relation_record(
        state, ctx, instance, relation,
        anchor_position=anchor_position,
        anchor_candidate=anchor_candidate,
        anchor_category=anchor_category,
    )
    now = _relation_now(ctx)
    record["last_checked_at"] = now
    record["confidence"] = round(max(float(record.get("confidence") or 0.0), confidence), 3)
    for evidence_id in evidence_ids or []:
        text = str(evidence_id or "")
        if text and text not in record.setdefault("evidence_ids", []):
            record["evidence_ids"].append(text)
    del record["evidence_ids"][:-16]
    if distance_m is not None:
        record["distance_m"] = round(float(distance_m), 3)
    if verdict == RELATION_SUPPORTS:
        record["state"] = RELATION_SUPPORTS
        record["source"] = source
        record["reason"] = reason
        record["last_supported_at"] = now
        record["last_unknown_reason"] = ""
        record["contradictions"] = []
        if anchor_position is not None:
            record["anchor_position"] = list(anchor_position)
        pose = _current_position(ctx.world)
        if pose is not None:
            record["established_pose"] = [round(float(value), 3) for value in pose]
    elif verdict == RELATION_CONTRADICTS:
        record["state"] = RELATION_CONTRADICTS
        record["source"] = source
        record["reason"] = reason
        record["contradicted_at"] = now
        record.setdefault("contradictions", []).append(
            {"reason": reason, "source": source, "at": now,
             "distance_m": None if distance_m is None else round(float(distance_m), 3)}
        )
        del record["contradictions"][:-8]
    else:
        record["last_unknown_at"] = now
        record["last_unknown_reason"] = reason
        if record.get("state") not in (RELATION_SUPPORTS, RELATION_CONTRADICTS):
            record["state"] = RELATION_UNKNOWN
            record["source"] = source
            record["reason"] = reason
    return record


def _relation_freshness(ctx: ToolContext, record: Mapping) -> dict:
    """Whether a recorded relation is still usable, and why not (``unknown``)."""
    now = _relation_now(ctx)
    last = _as_float(record.get("last_supported_at"), None)
    age = None if last is None else max(0.0, now - float(last))
    established = record.get("established_pose")
    current = _current_position(ctx.world)
    drift = None
    if (
        isinstance(established, (list, tuple))
        and len(established) >= 3
        and current is not None
    ):
        drift = round(
            math.dist(
                [float(value) for value in established[:3]],
                [float(value) for value in current],
            ),
            2,
        )
    geometric = _instance_point(record.get("anchor_position")) is not None
    conflict = record.get("anchor_reobservation_conflict")
    if isinstance(conflict, Mapping):
        return {
            "fresh": False,
            "reason": "anchor_relocalized",
            "age_s": age,
            "pose_drift_m": drift,
            "anchor_conflict": dict(conflict),
        }
    if not geometric:
        if age is not None and age > RELATION_STALE_AFTER_S:
            return {"fresh": False, "reason": "stale", "age_s": age, "pose_drift_m": drift}
        if drift is not None and drift > RELATION_POSE_DRIFT_MAX_M:
            return {
                "fresh": False,
                "reason": "pose_drift",
                "age_s": age,
                "pose_drift_m": drift,
            }
    return {"fresh": True, "reason": "", "age_s": age, "pose_drift_m": drift}


def _relation_public(record: Mapping, freshness: Mapping | None = None) -> dict:
    """The model-facing subset of one relation record (typed verdicts only)."""
    state = str(record.get("state") or RELATION_UNKNOWN)
    reason = (
        str(record.get("last_unknown_reason") or "")
        if state == RELATION_UNKNOWN
        else str(record.get("reason") or record.get("last_unknown_reason") or "")
    )
    historical = state == RELATION_SUPPORTS
    fresh = True if freshness is None else bool(freshness.get("fresh", True))
    public_state = state
    if state == RELATION_SUPPORTS and not fresh:
        public_state = RELATION_UNKNOWN
        reason = str(freshness.get("reason") or "stale")
    return {
        "state": public_state,
        "historical_state": state,
        "historical_supports": historical,
        "current_frame_state": RELATION_UNKNOWN,
        "source": str(record.get("source") or RELATION_SOURCE_NONE),
        "reason": reason,
        "fresh": fresh,
        "age_s": None if freshness is None else freshness.get("age_s"),
        "pose_drift_m": None if freshness is None else freshness.get("pose_drift_m"),
        "anchor": str(record.get("anchor_phrase") or ""),
        "anchor_category": str(record.get("anchor_category") or ""),
        "relation": str(record.get("relation") or ""),
        "limit_m": record.get("limit_m"),
        "distance_m": record.get("distance_m"),
        "confidence": record.get("confidence"),
        "established_at": record.get("last_supported_at"),
        "anchor_candidate": (
            (record.get("anchor_candidates") or [None])[0]
            if record.get("anchor_candidates") else None
        ),
        "evidence_ids": list(record.get("evidence_ids") or [])[-4:],
        "target_instance": record.get("target_instance"),
        "anchor_conflict": record.get("anchor_reobservation_conflict"),
        "anchor_ray_only": bool(record.get("anchor_ray_only")),
        "parallax": record.get("parallax"),
    }


def _relation_instance(
    ctx: ToolContext,
    candidate: Mapping,
    instruction: str,
    record: PerceptionObservation | None,
) -> dict:
    state = relation_state(ctx)
    instance_id = str(candidate.get("instance_hypothesis") or "")
    instance = state["instances"].get(instance_id)
    if instance is None:
        instance = _associate_instance(state, ctx, candidate, instruction, record)
        if isinstance(candidate, dict):
            candidate["instance_hypothesis"] = str(instance["id"])
    return instance


def _view_relation_flags(view: Mapping | None) -> tuple[bool | None, bool | None]:
    """``(anchor_visible, relation_supported)`` as the model answered them."""
    if not isinstance(view, Mapping):
        return None, None
    visible = view.get("anchor_visible")
    supported = view.get("relation_supported")
    visible = visible if isinstance(visible, bool) else None
    supported = supported if isinstance(supported, bool) else None
    return visible, supported


def relation_evidence_for_candidate(
    ctx: ToolContext,
    candidate: Mapping,
    instruction: str,
    relation: Mapping | None = None,
    *,
    record: PerceptionObservation | None = None,
    view: Mapping | None = None,
    visibility: Mapping | None = None,
    anchors: list[dict] | None = None,
    matched_anchors: list[dict] | None = None,
) -> dict | None:
    """The typed relation evidence of one candidate (supports/contradicts/unknown).

    Precedence:

    1. what *this* view says, but only when it is decided: ``anchor_visible``
       true with ``relation_supported`` false is an explicit contradiction;
       true+true is co-visible support; an anchor that is not visible is
       ``unknown`` for this frame and falls through to the historical record;
    2. the geometry against a located anchor (a candidate measured outside the
       relation's bound is an explicit contradiction; inside it is cross-frame
       support even when the anchor is out of frame);
    3. the recorded relation, checked for freshness (expiry / pose drift /
       anchor re-localization). A stale record is ``unknown``, never negative.
    """
    if relation is None:
        relation = _relation_anchor(instruction)
    if relation is None:
        return None
    chain_enabled = relation_chain_enabled(ctx)
    instance = _relation_instance(ctx, candidate, instruction, record)
    state = relation_state(ctx)
    point = _instance_point(candidate.get("position_flu"))
    subtask_key = str(candidate.get("subtask_key") or "")
    if anchors is None:
        anchors = _known_anchors(ctx, subtask_key)
    if matched_anchors is None:
        matched_anchors = _matching_anchors(anchors, relation)
    located_anchors = [
        anchor for anchor in matched_anchors
        if _instance_point(anchor.get("position")) is not None
    ]
    anchor = None
    if located_anchors:
        # a measured anchor always outranks a ray-only sighting
        if point is not None:
            anchor = min(
                located_anchors, key=lambda item: math.dist(point, item["position"])
            )
        else:
            anchor = located_anchors[0]
    elif matched_anchors:
        anchor = matched_anchors[0]
    anchor_position = None if anchor is None else _instance_point(anchor.get("position"))
    anchor_ray_only = bool(
        anchor is not None and anchor.get("ray_only") and anchor_position is None
    )
    distance = (
        None
        if point is None or anchor_position is None
        else math.dist(point, anchor_position)
    )
    limit = float(relation.get("limit_m") or INSTANCE_ADJACENT_MAX_M)
    anchor_category = str(anchor.get("category") or "") if anchor else _anchor_category(
        str(relation.get("anchor") or "")
    )
    anchor_candidate = None if anchor is None else str(anchor.get("id") or "")
    evidence_ids = [
        value
        for value in (
            str(candidate.get("observation_id") or ""),
            None if anchor is None else str(anchor.get("observation_id") or ""),
        )
        if value
    ]
    visible, supported = _view_relation_flags(view)
    if visible is True and supported is True:
        record_state = _write_relation(
            ctx, state, instance, relation,
            verdict=RELATION_SUPPORTS,
            source=RELATION_SOURCE_CO_VISIBLE,
            reason="anchor_visible_and_relation_supported",
            confidence=max(RELATION_MIN_CONFIDENCE, _as_float((view or {}).get("confidence"), 0.7) or 0.7),
            anchor_position=anchor_position,
            anchor_candidate=anchor_candidate,
            anchor_category=anchor_category,
            evidence_ids=evidence_ids,
            distance_m=distance,
        )
        public = _relation_public(record_state, _relation_freshness(ctx, record_state))
        public["current_frame_state"] = RELATION_SUPPORTS
        return public
    if visible is True and supported is False:
        record_state = _write_relation(
            ctx, state, instance, relation,
            verdict=RELATION_CONTRADICTS,
            source=RELATION_SOURCE_CO_VISIBLE,
            reason="anchor_visible_but_relation_unsupported",
            confidence=max(RELATION_MIN_CONFIDENCE, _as_float((view or {}).get("confidence"), 0.7) or 0.7),
            anchor_position=anchor_position,
            anchor_candidate=anchor_candidate,
            anchor_category=anchor_category,
            evidence_ids=evidence_ids,
            distance_m=distance,
        )
        public = _relation_public(record_state, _relation_freshness(ctx, record_state))
        public["current_frame_state"] = RELATION_CONTRADICTS
        return public
    if anchor_ray_only:
        # a bearing cannot place the relatum: unknown, with the parallax that
        # would turn the two sightings into a measured anchor position
        requirement = _anchor_parallax_requirement(relation, matched_anchors)
        record_state = _write_relation(
            ctx, state, instance, relation,
            verdict=RELATION_UNKNOWN,
            source=RELATION_SOURCE_CROSS_FRAME,
            reason="anchor_ray_only",
            confidence=0.2,
            anchor_candidate=anchor_candidate,
            anchor_category=anchor_category,
            evidence_ids=evidence_ids,
            distance_m=None,
        )
        record_state["anchor_ray_only"] = True
        record_state["parallax"] = requirement
        public = _relation_public(record_state, _relation_freshness(ctx, record_state))
        public["current_frame_state"] = RELATION_UNKNOWN
        public["anchor_ray_only"] = True
        public["parallax"] = requirement
        return public
    if anchor_position is not None and distance is not None:
        anchor_trusted = bool(anchor.get("trusted", True)) if anchor else False
        if distance > limit:
            if not anchor_trusted:
                # an estimated anchor position cannot prove the candidate wrong:
                # this is unknown, not negative (the confirmation view can still
                # establish the relation by seeing both objects)
                record_state = _write_relation(
                    ctx, state, instance, relation,
                    verdict=RELATION_UNKNOWN,
                    source=RELATION_SOURCE_CROSS_FRAME,
                    reason="anchor_distance_uncertain",
                    confidence=0.3,
                    anchor_position=anchor_position,
                    anchor_candidate=anchor_candidate,
                    anchor_category=anchor_category,
                    evidence_ids=evidence_ids,
                    distance_m=distance,
                )
                public = _relation_public(record_state, _relation_freshness(ctx, record_state))
                public["current_frame_state"] = RELATION_UNKNOWN
                return public
            record_state = _write_relation(
                ctx, state, instance, relation,
                verdict=RELATION_CONTRADICTS,
                source=RELATION_SOURCE_CROSS_FRAME,
                reason="anchor_distance",
                confidence=0.8,
                anchor_position=anchor_position,
                anchor_candidate=anchor_candidate,
                anchor_category=anchor_category,
                evidence_ids=evidence_ids,
                distance_m=distance,
            )
            public = _relation_public(record_state, _relation_freshness(ctx, record_state))
            public["current_frame_state"] = (
                RELATION_UNKNOWN if visible is not True else RELATION_CONTRADICTS
            )
            return public
        record_state = _write_relation(
            ctx, state, instance, relation,
            verdict=RELATION_SUPPORTS,
            source=RELATION_SOURCE_CROSS_FRAME,
            reason="candidate_inside_anchor_bound",
            confidence=0.7 if anchor_trusted else 0.5,
            anchor_position=anchor_position,
            anchor_candidate=anchor_candidate,
            anchor_category=anchor_category,
            evidence_ids=evidence_ids,
            distance_m=distance,
        )
        public = _relation_public(record_state, _relation_freshness(ctx, record_state))
        public["current_frame_state"] = RELATION_UNKNOWN
        return public
    if isinstance(visibility, Mapping):
        if visibility.get("visible") is True:
            visibility_source = str(visibility.get("source") or "geometry")
            record_state = _write_relation(
                ctx, state, instance, relation,
                verdict=RELATION_SUPPORTS,
                source=(
                    RELATION_SOURCE_MODEL
                    if visibility_source == "model"
                    else RELATION_SOURCE_CO_VISIBLE
                ),
                reason=str(visibility_source),
                confidence=0.6,
                anchor_position=anchor_position,
                anchor_candidate=anchor_candidate,
                anchor_category=anchor_category,
                evidence_ids=evidence_ids,
                distance_m=distance,
            )
            public = _relation_public(record_state, _relation_freshness(ctx, record_state))
            public["current_frame_state"] = RELATION_SUPPORTS
            return public
    stored_relation = state["relations"].get(
        _relation_key(str(instance.get("id") or ""), relation)
    )
    if isinstance(stored_relation, dict):
        freshness = _relation_freshness(ctx, stored_relation)
        public = _relation_public(stored_relation, freshness)
        if public["state"] == RELATION_SUPPORTS and not public.get("reason"):
            public["reason"] = "cross_frame_relation"
        return public
    record_state = _write_relation(
        ctx, state, instance, relation,
        verdict=RELATION_UNKNOWN,
        source=RELATION_SOURCE_NONE,
        reason=(
            "anchor_not_in_frame"
            if visible is False or (isinstance(visibility, Mapping) and visibility.get("visible") is False)
            else "anchor_not_established"
        ),
        confidence=0.0,
        anchor_position=anchor_position,
        anchor_candidate=anchor_candidate,
        anchor_category=anchor_category,
        evidence_ids=evidence_ids,
        distance_m=distance,
    )
    return _relation_public(record_state, _relation_freshness(ctx, record_state))


def relation_state_summary(ctx: ToolContext) -> dict:
    """Compact, model-facing snapshot of the relation chain (JSON-plain)."""
    state = _extra(ctx).get(RELATION_CHAIN_KEY)
    if not isinstance(state, dict):
        return {"enabled": relation_chain_enabled(ctx), "n_instances": 0, "relations": []}
    relations = []
    for record in state.get("relations", {}).values():
        if isinstance(record, dict):
            relations.append(_relation_public(record, _relation_freshness(ctx, record)))
    instances = []
    for instance in state.get("instances", {}).values():
        if not isinstance(instance, dict):
            continue
        instances.append({
            "id": instance.get("id"),
            "status": instance.get("status"),
            "rank": instance.get("rank"),
            "candidates": list(instance.get("candidate_ids") or []),
            "position_flu": instance.get("position_flu"),
            "sigma_m": instance.get("sigma_m"),
            "support": instance.get("support"),
        })
    observations = list(state.get("observations") or [])
    return {
        "enabled": relation_chain_enabled(ctx),
        "n_instances": len(instances),
        "instances": instances[-8:],
        "relations": relations[-8:],
        "competitors": {
            key: list(value)
            for key, value in (state.get("competitors") or {}).items()
            if isinstance(value, list)
        },
        "unknown_counts": {
            str(name): int(count)
            for name, count in (state.get("counts") or {}).items()
            if count
        },
        "observation_decisions": {
            "n": len(observations),
            "changed": sum(1 for entry in observations if entry.get("changed")),
        },
    }


def typed_uncertainties(
    ctx: ToolContext,
    *,
    candidate_ids: list[str] | None = None,
    instruction: str = "",
    relation: Mapping | None = None,
    view: Mapping | None = None,
    path_blocked: bool = False,
) -> list[dict]:
    """The five uncertainty types that apply right now, each with its detail.

    ``category`` / ``instance`` / ``distance`` / ``anchor_relation`` /
    ``reachability`` - the same vocabulary the observation planner maps to
    evidence actions. Nothing here is a search fallback: a missing anchor maps
    to an anchor observation, a lost path to a viewpoint change.
    """
    state = relation_state(ctx) if relation_chain_enabled(ctx) else {}
    store = _candidate_store(ctx)
    ids = [
        str(value)
        for value in (candidate_ids or [])
        if str(value) and str(value) in store
    ]
    candidates = [store[value] for value in ids if isinstance(store.get(value), Mapping)]
    uncertainties: list[dict] = []
    if relation is None and instruction:
        relation = _relation_anchor(instruction)
    target_phrase = _target_phrase(instruction) if instruction else ""
    if instruction and not _anchor_category(target_phrase):
        uncertainties.append({
            "type": UNCERTAINTY_CATEGORY,
            "detail": "the instruction's object category is not one the parser knows",
            "target": target_phrase,
        })
    competitors = []
    for candidate in candidates:
        instance = state.get("instances", {}).get(
            str(candidate.get("instance_hypothesis") or "")
        ) if state else None
        if isinstance(instance, Mapping) and instance.get("competing"):
            key = str(instance.get("target_key") or "")
            competitors = list((state.get("competitors") or {}).get(key) or [])
    if len(competitors) >= 2:
        uncertainties.append({
            "type": UNCERTAINTY_INSTANCE,
            "detail": f"{len(competitors)} competing hypotheses",
            "hypotheses": competitors[:RELATION_MAX_COMPETING],
        })
    for candidate in candidates:
        if _instance_point(candidate.get("position_flu")) is None or candidate.get(
            "distance_uncertain"
        ):
            uncertainties.append({
                "type": UNCERTAINTY_DISTANCE,
                "detail": "the candidate has no trusted range",
                "candidate_id": candidate.get("id"),
                "distance_source": candidate.get("distance_source"),
            })
            break
    if relation is not None:
        evidence = None
        for candidate in candidates:
            evidence = relation_evidence_for_candidate(
                ctx, candidate, instruction, relation, view=view,
            )
            if isinstance(evidence, Mapping):
                break
        if not isinstance(evidence, Mapping) or evidence.get("state") != RELATION_SUPPORTS:
            uncertainties.append({
                "type": UNCERTAINTY_ANCHOR_RELATION,
                "detail": "the relation anchor is not established or its evidence is stale",
                "anchor": str(relation.get("anchor") or ""),
                "relation": str(relation.get("relation") or ""),
                "evidence_state": (
                    None if not isinstance(evidence, Mapping) else evidence.get("state")
                ),
                "reason": None if not isinstance(evidence, Mapping) else evidence.get("reason"),
            })
    if path_blocked:
        uncertainties.append({
            "type": UNCERTAINTY_REACHABILITY,
            "detail": "the route to the candidate is blocked",
        })
    return uncertainties


def _anchor_ground_instruction(relation: Mapping) -> str:
    return f"找到并定位{str(relation.get('anchor') or '').strip()}"


def _observation_action(
    tool: str,
    args: dict,
    why: str,
    *,
    uncertainty: str,
    expected_excluded: float,
    cost_s: float,
    **detail: Any,
) -> dict:
    cost = max(0.5, float(cost_s))
    return {
        "tool": tool,
        "args": dict(args),
        "why": why,
        "uncertainty": uncertainty,
        "expected_excluded": round(float(expected_excluded), 3),
        "cost_s": round(cost, 3),
        "score": round(float(expected_excluded) / cost, 4),
        **detail,
    }


def rank_observation_actions(
    uncertainties: list[dict] | list[str],
    *,
    candidate_ids: list[str] | None = None,
    competitor_ids: list[str] | None = None,
    instruction: str = "",
    relation: Mapping | None = None,
    candidate_position: tuple[float, float, float] | None = None,
    current_position: tuple[float, float, float] | None = None,
) -> list[dict]:
    """Rank supplementary observations by excluded hypotheses per second.

    The planner is deliberately simple and untrained: each uncertainty type has
    one evidence action, the action's expected exclusion count is divided by its
    (movement time + inference cost), and the best action is the one to try.
    ``instance`` compares the competing candidates (no flight); ``distance``
    re-grounds from a new view (a second bearing); ``anchor_relation`` finds or
    re-locates the anchor; ``reachability`` proposes another viewpoint - none of
    them fall back to a generic re-search.
    """
    actions: list[dict] = []
    ids = [str(value) for value in (candidate_ids or []) if str(value)]
    competitors = [str(value) for value in (competitor_ids or []) if str(value)]
    relation = relation if isinstance(relation, Mapping) else _relation_anchor(instruction)
    move_s = 1.0
    if candidate_position is not None and current_position is not None:
        try:
            move_s = math.dist(
                [float(value) for value in candidate_position[:3]],
                [float(value) for value in current_position[:3]],
            ) / 1.0
        except (TypeError, ValueError):
            move_s = 1.0
    search = {
        UNCERTAINTY_CATEGORY: (
            _observation_action(
                "observe",
                {"mode": "lookaround", "n_views": 4, "spread_deg": 180.0},
                "the object category itself is unresolved: a wider look plus a "
                "new grounding is what distinguishes 'not this kind' from 'not here'",
                uncertainty=UNCERTAINTY_CATEGORY,
                expected_excluded=1.0,
                cost_s=2.0,
            )
        ),
        UNCERTAINTY_INSTANCE: (
            _observation_action(
                "compare_candidates",
                {"candidates_a": [competitors[0]], "candidates_b": [competitors[1]]},
                "two candidate instances match the description: compare them "
                "directly before flying to either",
                uncertainty=UNCERTAINTY_INSTANCE,
                expected_excluded=max(1, len(competitors) - 1),
                cost_s=0.5,
                hypotheses=competitors[:RELATION_MAX_COMPETING],
            )
            if len(competitors) >= 2
            else None
        ),
        UNCERTAINTY_DISTANCE: (
            _observation_action(
                "ground_target",
                {"instruction": instruction or "<the instructed target>"},
                "the range is the open question: a second bearing from a new "
                "view triangulates it (the same frame cannot)",
                uncertainty=UNCERTAINTY_DISTANCE,
                expected_excluded=1.0,
                cost_s=2.0 + move_s,
            )
        ),
        UNCERTAINTY_ANCHOR_RELATION: (
            _observation_action(
                "ground_target",
                {"instruction": _anchor_ground_instruction(relation) if relation else "<the anchor>"},
                "the anchor is not established: find and locate the anchor "
                "object itself, then the original target resumes with its "
                "candidates and budget intact",
                uncertainty=UNCERTAINTY_ANCHOR_RELATION,
                expected_excluded=3.0,
                cost_s=3.0 + move_s,
                temporary_subtask=True,
            )
        ),
        UNCERTAINTY_REACHABILITY: (
            _observation_action(
                "propose_viewpoints",
                {"count": 3, **({"target_hypothesis": candidate_position}
                                if candidate_position is not None else {})},
                "the path is blocked, not the identity: propose another "
                "viewpoint that still holds the candidate",
                uncertainty=UNCERTAINTY_REACHABILITY,
                expected_excluded=1.0,
                cost_s=2.5,
            )
        ),
    }
    for item in uncertainties:
        name = str(item.get("type") if isinstance(item, Mapping) else item)
        action = search.get(name)
        if not isinstance(action, Mapping):
            continue
        action = dict(action)
        detail = item if isinstance(item, Mapping) else {}
        if detail:
            action["detail"] = {
                key: value for key, value in detail.items() if key != "type"
            }
        actions.append(action)
    actions.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return actions


def record_observation_decision(
    ctx: ToolContext,
    *,
    uncertainty: str,
    action: Mapping | None,
    tried_to_distinguish: str,
    before: Any,
    after: Any,
    changed: bool,
    note: str = "",
) -> dict:
    """Journal one supplementary observation: what it tried to separate, result.

    The claim is the point of the milestone: every extra look must name the
    hypothesis pair it was meant to distinguish and whether the judgement moved.
    """
    state = relation_state(ctx)
    entry = {
        "seq": len(state.get("observations") or []) + 1,
        "at": _relation_now(ctx),
        "uncertainty": str(uncertainty or ""),
        "action": (
            None if not isinstance(action, Mapping)
            else {"tool": action.get("tool"), "args": dict(action.get("args") or {})}
        ),
        "tried_to_distinguish": str(tried_to_distinguish or "")[:200],
        "before": _json_plain(before),
        "after": _json_plain(after),
        "changed": bool(changed),
        "note": str(note or "")[:200],
    }
    state.setdefault("observations", []).append(entry)
    del state["observations"][:-MAX_RELATION_OBSERVATIONS]
    name = str(uncertainty or "")
    counts = state.setdefault("counts", {})
    if name in counts:
        counts[name] = int(counts.get(name) or 0) + 1
    return entry


def _json_plain(value: Any) -> Any:
    """Small JSON-safe copy of a verdict for the decision journal."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_plain(item) for key, item in list(value.items())[:12]}
    if isinstance(value, (list, tuple)):
        return [_json_plain(item) for item in list(value)[:12]]
    return str(value)[:200]


def competing_candidates(ctx: ToolContext, candidate: Mapping) -> list[str]:
    """Candidate ids of the other active hypotheses of the same target."""
    state = _extra(ctx).get(RELATION_CHAIN_KEY)
    if not isinstance(state, dict):
        return []
    instance = state.get("instances", {}).get(
        str(candidate.get("instance_hypothesis") or "")
    )
    if not isinstance(instance, Mapping):
        return []
    target_key = str(instance.get("target_key") or "")
    out: list[str] = []
    for instance_id in (state.get("competitors") or {}).get(target_key) or []:
        other = state.get("instances", {}).get(str(instance_id))
        if not isinstance(other, Mapping) or other.get("id") == instance.get("id"):
            continue
        for candidate_id in other.get("candidate_ids") or []:
            text = str(candidate_id or "")
            if text and text not in out:
                out.append(text)
    return out[:RELATION_MAX_COMPETING]


def relation_observation_plan(
    ctx: ToolContext,
    instruction: str,
    *,
    candidate_ids: list[str] | None = None,
    relation: Mapping | None = None,
    path_blocked: bool = False,
) -> dict:
    """Typed uncertainties + ranked supplementary observations for a tool result."""
    if not relation_chain_enabled(ctx):
        return {"enabled": False, "uncertainties": [], "actions": []}
    if relation is None:
        relation = _relation_anchor(instruction)
    candidates = [
        _candidate_store(ctx).get(str(value)) for value in (candidate_ids or [])
    ]
    candidates = [item for item in candidates if isinstance(item, Mapping)]
    competitor_ids: list[str] = []
    for candidate in candidates:
        for candidate_id in competing_candidates(ctx, candidate):
            if candidate_id not in competitor_ids:
                competitor_ids.append(candidate_id)
    uncertainties = typed_uncertainties(
        ctx,
        candidate_ids=[
            str(candidate.get("id") or "") for candidate in candidates if candidate.get("id")
        ],
        instruction=instruction,
        relation=relation,
        path_blocked=path_blocked,
    )
    position = None
    for candidate in candidates:
        position = _instance_point(candidate.get("position_flu"))
        if position is not None:
            break
    actions = rank_observation_actions(
        uncertainties,
        candidate_ids=[
            str(candidate.get("id") or "") for candidate in candidates if candidate.get("id")
        ],
        competitor_ids=competitor_ids,
        instruction=instruction,
        relation=relation,
        candidate_position=position,
        current_position=_current_position(ctx.world),
    )
    return {
        "enabled": True,
        "uncertainties": uncertainties,
        "actions": actions,
        "next_action": actions[0] if actions else None,
    }


def mark_candidate_rejected(
    ctx: ToolContext,
    candidate_id: str,
    reason: str = "",
) -> dict:
    """Mark the instance a rejected candidate belonged to and switch hypothesis.

    The explored place, the other candidates of the rejected instance and the
    completed plan steps stay: only the hypothesis is closed, and the best
    remaining competitor becomes the one to observe next.
    """
    state = _extra(ctx).get(RELATION_CHAIN_KEY)
    if not isinstance(state, dict):
        return {}
    candidate_id = str(candidate_id or "")
    rejected = None
    for instance in state.get("instances", {}).values():
        if candidate_id in (instance.get("candidate_ids") or []):
            rejected = instance
            break
    if rejected is None:
        return {"rejected": candidate_id, "switched_to": None}
    rejected["status"] = "rejected"
    rejected["rank"] = None
    rejected["rejection_reason"] = str(reason or "")[:160]
    target_key = str(rejected.get("target_key") or "")
    remaining = [
        instance
        for instance in state.get("instances", {}).values()
        if instance.get("target_key") == target_key
        and instance.get("status") != "rejected"
        and instance.get("id") != rejected.get("id")
    ]
    remaining.sort(
        key=lambda item: (
            float(item.get("confidence") or 0.0),
            int(item.get("support") or 0),
        ),
        reverse=True,
    )
    switched = None if not remaining else remaining[0]
    if switched is not None:
        switched["rank"] = 1
        switched["status"] = "hypothesis"
        state.setdefault("switched", []).append(
            {
                "from": rejected.get("id"),
                "to": switched.get("id"),
                "at": _relation_now(ctx),
                "reason": str(reason or "")[:160],
            }
        )
        del state["switched"][:-12]
    if switched is not None:
        ordered = [switched] + [
            instance for instance in remaining if instance is not switched
        ][: RELATION_MAX_COMPETING - 1]
    else:
        ordered = []
    state["competitors"][target_key] = [
        str(instance.get("id")) for instance in ordered if instance is not None
    ]
    return {
        "rejected": str(rejected.get("id") or ""),
        "rejected_candidates": list(rejected.get("candidate_ids") or []),
        "switched_to": None if switched is None else str(switched.get("id") or ""),
        "switched_candidates": [] if switched is None else list(switched.get("candidate_ids") or []),
        "remaining": len(remaining),
        "reason": str(reason or "")[:160],
    }


def _same_instance(first: Mapping, second: Mapping) -> bool:
    """Two boxes describe the same instance (overlap, no attr contradiction)."""
    box_a = parse_box(first.get("box"), box_format=BOX_FORMAT_XYXY)[0]
    box_b = parse_box(second.get("box"), box_format=BOX_FORMAT_XYXY)[0]
    if box_a is not None and box_b is not None:
        iou = _iou(box_a.to_list(), box_b.to_list())
        if iou >= IOU_SAME:
            return True
    return False


def _confidence_of(candidate: Mapping) -> float:
    value = _as_float(candidate.get("confidence"), None)
    return -1.0 if value is None else value


def _instance_consistency(
    ctx: ToolContext,
    record: PerceptionObservation,
    instruction: str,
    candidates: list[dict],
) -> dict:
    """Filter/rank grounded boxes by whether they are the instructed instance.

    Deterministic judgements carry the gate; the model's self-report does not:

    - the instruction's colour/material attributes against the box attributes.
      A conflict is suspect, and *silence* is suspect too when the instruction
      names an attribute: an unverified candidate is not an approved one;
    - the range against the size its own box subtends and the target noun's
      rough size (a "chair" whose box would be 5 m wide at 3 m is a
      wall/support surface); triangulated candidates are not exempt;
    - for a triangulated candidate, the physical size implied by each view's
      box at that view's range: two bearings that see different-sized objects
      are not two views of one instance;
    - the relation to a known earlier-stage candidate ("the lamp next to the
      chair" landing 15 m from that chair);
    - the relation anchor's co-visibility in the candidate's own frame/crop:
      an instruction like "书架旁的课桌" is only verified where the bookshelf
      is actually visible (the model's ``anchor_visible`` answer or a projection
      of the known anchor position), and the candidate must sit inside the
      relation's distance bound (3 m for adjacency, 4 m for "near");
    - a candidate with no signal at all (no match flag, no attributes, no size
      and no range) is unverified and therefore suspect.

    ``matches_description`` is recorded (``model_match``) but never lets a
    candidate through: final3's model reported ``true`` for 28/28 candidates,
    including a 5.6 m "bookshelf". Its only effect is that an explicit
    ``false`` adds ``description_mismatch``.

    A candidate that fails any check is stored with ``instance_suspect=true``
    and cannot be approached or accepted by a confirmation. When several
    distinct instances survive, only the best match is approachable: a frame
    with two chairs and no relation cannot say which one the instruction means,
    so the others are marked ``not_best_match`` instead of being flown to.
    """
    intrinsics = _observation_intrinsics(ctx, record)
    subtask_key = str((candidates[0] if candidates else {}).get("subtask_key") or "")
    anchors = _known_anchors(ctx, subtask_key)
    relation = _relation_anchor(instruction)
    matched_anchors = _matching_anchors(anchors, relation)
    chain_enabled = relation_chain_enabled(ctx)
    instance_summary = (
        update_instance_hypotheses(ctx, record, instruction, candidates)
        if chain_enabled and candidates else {"enabled": False, "n_instances": 0}
    )
    relation_unknowns: dict[str, int] = {}
    reasons: dict[str, int] = {}
    model_match = {"true": 0, "false": 0, "missing": 0}
    for candidate in candidates:
        found: list[str] = []
        # a detector-verified candidate was proposed by SAM3 *conditioned on*
        # the instruction's object phrase (colour/material included), so the
        # VLM's own attribute report is not the identity evidence here
        detector_verified = bool(candidate.get("detector_verified"))
        match_flag = candidate.get("matches_description")
        if match_flag is True:
            model_match["true"] += 1
        elif match_flag is False:
            model_match["false"] += 1
            if not detector_verified:
                found.append("description_mismatch")
        else:
            model_match["missing"] += 1
        if not detector_verified:
            attr = _attribute_conflict(instruction, candidate.get("attrs") or {})
            if attr:
                found.append(f"attribute_conflict:{attr}")
            else:
                missing_attrs = _attribute_unknown(
                    instruction, candidate.get("attrs") or {}
                )
                if missing_attrs:
                    candidate["instance_attribute_unknown"] = missing_attrs
                    found.append(f"attribute_unknown:{missing_attrs}")
        size = _size_conflict(candidate, intrinsics)
        if size is not None:
            candidate["instance_size_conflict"] = size
            found.append(f"distance_size_conflict:{size['kind']}")
        two_view = _two_view_size_conflict(candidate, intrinsics)
        if two_view is not None:
            candidate["instance_two_view_conflict"] = two_view
            found.append("two_view_size_conflict")
        anchor = _anchor_conflict(instruction, candidate, anchors, relation)
        if anchor is not None:
            candidate["instance_anchor_conflict"] = anchor
            found.append("anchor_distance_conflict")
        if relation is not None:
            visibility_anchor = None
            if matched_anchors:
                point = _instance_point(candidate.get("position_flu"))
                located = [
                    anchor for anchor in matched_anchors
                    if _instance_point(anchor.get("position")) is not None
                ]
                pool = located or matched_anchors
                visibility_anchor = (
                    min(pool, key=lambda item: math.dist(point, item["position"]))
                    if point is not None and located
                    else pool[0]
                )
            visibility = _anchor_visibility(
                ctx, record, candidate, visibility_anchor
            )
            candidate["instance_anchor_visibility"] = visibility
            if chain_enabled:
                evidence = relation_evidence_for_candidate(
                    ctx, candidate, instruction, relation,
                    record=record, visibility=visibility,
                    anchors=anchors, matched_anchors=matched_anchors,
                )
                if isinstance(evidence, Mapping):
                    candidate["relation_evidence"] = evidence
                    if evidence.get("state") == RELATION_CONTRADICTS:
                        # explicit opposite evidence (seen anchor without the
                        # relation, or a measured distance outside the bound)
                        found.append(
                            "relation_conflict:"
                            + str(evidence.get("reason") or "contradicted")
                        )
                    elif evidence.get("state") == RELATION_UNKNOWN:
                        # not seeing the anchor is *unknown*, not negative: keep
                        # the candidate and name the observation that settles it
                        reason = str(evidence.get("reason") or "anchor_not_in_frame")
                        candidate["relation_unknown"] = {
                            "reason": reason,
                            "anchor": evidence.get("anchor"),
                            "evidence": evidence,
                        }
                        relation_unknowns[reason] = relation_unknowns.get(reason, 0) + 1
            elif visibility["visible"] is not True:
                # no model answer and no projection that places the anchor in
                # this frame: the relation is unverified, not approved
                found.append("anchor_not_visible")
        distance = _as_float(
            candidate.get("range_m"), _as_float(candidate.get("distance_m"), None)
        )
        # a retry/crop candidate that comes back with neither a match flag nor
        # any attribute has no identity evidence at all, only box geometry. That
        # is "unverified", not "approved" (12/40 final3 retry/crop candidates
        # took exactly this path). Direct grounding replies keep the older
        # behaviour: the prompt asks for the fields, but a missing field there is
        # not by itself proof against the instance.
        is_retry = bool(candidate.get("grounding_retry"))
        if (
            not found
            and not detector_verified
            and is_retry
            and match_flag is None
            and not (candidate.get("attrs") or {})
        ):
            found.append("instance_fields_unknown")
        # no size and no range means even the geometry check could not run
        elif (
            not found
            and not detector_verified
            and is_retry
            and candidate.get("size_m") is None
            and (distance is None or distance <= MIN_RANGE_M)
        ):
            found.append("size_unknown")
        if found:
            candidate["instance_suspect"] = True
            candidate["instance_suspect_reason"] = "; ".join(found)[:200]
            for reason in found:
                key = reason.split(":", 1)[0]
                reasons[key] = reasons.get(key, 0) + 1
    survivors = [candidate for candidate in candidates if not candidate.get("instance_suspect")]
    # most likely first: the prompt asks for the most likely candidate first, so
    # order breaks ties and confidence refines it
    survivors.sort(key=_confidence_of, reverse=True)
    for rank, candidate in enumerate(survivors, 1):
        candidate["instance_rank"] = rank
    if len(survivors) >= 2 and not _same_instance(survivors[0], survivors[1]):
        if chain_enabled:
            # competing hypotheses stay inspectable: a small set of alternatives
            # is what makes a confirmation's rejection switch to the next
            # candidate instead of re-searching from scratch. Only the rank
            # marks the preference; the instance gate does not condemn them.
            for candidate in survivors:
                candidate["competing_hypothesis"] = True
        else:
            for candidate in survivors[1:]:
                candidate["instance_suspect"] = True
                candidate["instance_suspect_reason"] = "not_best_match"
                candidate["instance_rank"] = None
                reasons["not_best_match"] = reasons.get("not_best_match", 0) + 1
    temporary_anchor = None
    if chain_enabled and relation is not None and relation_unknowns and not matched_anchors:
        temporary_anchor = _ensure_anchor_subtask(ctx, relation)
    return {
        "filter": "instance_consistency",
        "n_candidates": len(candidates),
        "n_suspect": sum(1 for item in candidates if item.get("instance_suspect")),
        "suspect": [
            {"id": item.get("id"), "reason": item.get("instance_suspect_reason")}
            for item in candidates
            if item.get("instance_suspect")
        ],
        "approvable": [
            item.get("id") for item in candidates if not item.get("instance_suspect")
        ],
        "reasons": reasons,
        "model_match": model_match,
        "anchors": [anchor["id"] for anchor in anchors],
        "anchor_relation": relation,
        "anchor_candidates": [anchor["id"] for anchor in matched_anchors],
        "relation_chain": (
            {
                "enabled": True,
                "instances": instance_summary,
                "unknown": relation_unknowns,
                "temporary_anchor_subtask": temporary_anchor,
            }
            if chain_enabled else {"enabled": False}
        ),
    }


def _ensure_anchor_subtask(ctx: ToolContext, relation: Mapping) -> dict:
    """Create the dependent "find and locate the anchor" sub-task (best effort).

    Imported lazily so the perception module stays independent of the plan
    module; a context without a plan simply reports ``no_plan``.
    """
    try:
        from .tool_impl_task import ensure_anchor_subtask

        return ensure_anchor_subtask(ctx, relation=relation)
    except Exception as exc:  # noqa: BLE001 - the anchor task is advisory
        return {"created": False, "reason": f"{type(exc).__name__}: {exc}"}


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
        "conflicts": 0,
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


def _sighting_of_candidate(candidate: dict, instruction: str, record: PerceptionObservation,
                           anchor: str = "") -> dict:
    """The sighting a grounded candidate *is*: image, candidate and basis.

    What ``ground_target`` knows that a plain capture does not: which frame,
    which candidate id, the box the grounding model wrote and the attributes it
    read. That is the judgement basis a later ``query_memory`` cites, so the
    sighting is stored with it instead of a bare target name. ``anchor`` is the
    relation phrase the caller knows from its task plan (``ctx.extra``); it is
    only stored when non-empty, so a task without a relation adds no field.
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
    sighting = {
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
    if str(anchor or "").strip():
        sighting["anchor"] = str(anchor).strip()
    return sighting


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


def _conflict_stagnant_result(
    ctx: ToolContext, instruction: str, record: PerceptionObservation, attempt: dict
) -> ToolResult:
    """``blocked`` answer for evidence whose ranges keep contradicting the model.

    The same view can return the same huge box again and again; each answer is
    a new candidate id but not new evidence. Once two groundings on this
    evidence produced unresolved ``distance_source=conflict`` candidates, the
    next call is refused with a concrete change of viewpoint instead of more
    candidates that confirmation cannot accept.
    """
    conflicts = int(attempt.get("conflicts", 0))
    change = _strategy_change(ctx, record, "change_viewpoint")
    data = {
        "reason": "stagnant_distance_conflict",
        "instruction": instruction,
        "observation_id": record.observation_id,
        "attempts": conflicts + 1,
        "conflicting_candidates": conflicts,
        "repeated_grounding": True,
        "model_call_skipped": True,
        "hint": (
            "the same evidence returned candidates whose depth contradicts the "
            "model's distance estimate: the box is likely background, so "
            "observe from a different heading/altitude and ground the object's "
            "support point instead of grounding this view again"
        ),
        "strategy_change": change,
        "next_action": {"tool": change["tool"], "args": dict(change["args"])},
        "available_candidate_ids": available_candidate_ids(ctx),
    }
    evidence = [
        EvidenceRef("observation", record.observation_id, "frame returned conflicting ranges")
    ]
    return ToolResult(STATUS_BLOCKED, data, evidence)


# --------------------------------------------------------------------------
# detector gate: an independent instance proposal for the same frame
# --------------------------------------------------------------------------


def _detector_from_ctx(ctx: ToolContext) -> DetectorBackend | None:
    extra = _extra(ctx)
    detector = extra.get(DETECTOR_CONTEXT_KEY)
    if detector is None:
        detector = extra.get("detector")
    return detector if hasattr(detector, "detect") else None


def _next_candidate_id(store: Mapping) -> str:
    index = 1
    while f"cand_{index}" in store:
        index += 1
    return f"cand_{index}"


def _detector_embedding_similarity(left, right) -> float | None:
    if left is None or right is None:
        return None
    try:
        a = np.asarray(left, dtype=np.float64).reshape(-1)
        b = np.asarray(right, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if a.size == 0 or a.size != b.size:
        return None
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0.0:
        return None
    return float(np.dot(a, b) / denominator)


def _box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _containment(inner: Sequence[float], outer: Sequence[float]) -> float:
    ix1 = max(float(inner[0]), float(outer[0]))
    iy1 = max(float(inner[1]), float(outer[1]))
    ix2 = min(float(inner[2]), float(outer[2]))
    iy2 = min(float(inner[3]), float(outer[3]))
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = _box_area(inner)
    return intersection / area if area > 0 else 0.0


def _proposal_match(
    candidate_box: Sequence[float], proposal_box: Sequence[float], dino_similarity: float | None
) -> dict:
    """One VLM box against one detector proposal (geometry + feature)."""
    iou = _box_iou(candidate_box, proposal_box)
    candidate_inside = _containment(candidate_box, proposal_box)
    proposal_inside = _containment(proposal_box, candidate_box)
    candidate_area = _box_area(candidate_box)
    proposal_area = _box_area(proposal_box)
    area_share = proposal_area / candidate_area if candidate_area > 0 else 0.0
    geometric = bool(
        iou >= DETECTOR_MATCH_IOU
        or candidate_inside >= DETECTOR_MATCH_CONTAINMENT
        or (
            proposal_inside >= DETECTOR_PROPOSAL_IN_BOX
            and area_share >= DETECTOR_PROPOSAL_AREA_SHARE
        )
    )
    feature_ok = dino_similarity is None or dino_similarity >= 0.25
    score = iou + 0.5 * candidate_inside + 0.25 * proposal_inside
    if dino_similarity is not None:
        score += max(0.0, dino_similarity)
    return {
        "matched": bool(geometric and feature_ok),
        "geometric": geometric,
        "feature_ok": feature_ok,
        "iou": round(iou, 3),
        "candidate_inside": round(candidate_inside, 3),
        "proposal_inside": round(proposal_inside, 3),
        "area_share": round(area_share, 3),
        "dino_similarity": (
            None if dino_similarity is None else round(float(dino_similarity), 3)
        ),
        "_score": score,
    }


def _proposal_boxes(detections: list) -> list[list[float] | None]:
    """Boxes aligned with ``detections`` (``None`` for an unusable proposal)."""
    boxes: list[list[float] | None] = []
    for detection in detections:
        box = getattr(detection, "box", None)
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            boxes.append(None)
            continue
        try:
            boxes.append([float(value) for value in list(box)[:4]])
        except (TypeError, ValueError):
            boxes.append(None)
    return boxes


def _apply_detector_gate(
    ctx: ToolContext,
    record: PerceptionObservation,
    instruction: str,
    candidates: list[dict],
    depth_fn: Any,
    depth_reason: str,
    subtask_key: str = "",
) -> tuple[list[dict], dict]:
    """Ground ``candidates`` against an independent detector proposal set.

    When the backend is off/unavailable/non-productive the candidates keep their
    current behaviour (``proposal_source="vlm"``). When it produced proposals:

    - a VLM box that matches a proposal (IoU/containment + DINOv2 crop
      similarity) is ``proposal_source="vlm+detector"``;
    - a VLM box that matches none of the *authoritative* proposals is not the
      instructed instance and is dropped (recorded in the gate report);
    - when no VLM candidate survives, the best proposals become candidates of
      their own (``proposal_source="detector"``) so a VLM miss can still be
      localized;
    - proposals below the authoritative score are recorded but never drop a VLM
      candidate (a weak detector must not erase the VLM path).
    """
    from .detectors import detector_report

    store = _candidate_store(ctx)
    detector = _detector_from_ctx(ctx)
    if detector is None:
        for candidate in candidates:
            candidate.setdefault("proposal_source", "vlm")
        return candidates, detector_report(None, "off", fallback="vlm", labels=[])
    name = str(getattr(detector, "name", "detector") or "detector")
    try:
        available = bool(detector.available())
    except Exception as exc:  # noqa: BLE001 - availability is a finding, not a crash
        for candidate in candidates:
            candidate.setdefault("proposal_source", "vlm")
        return candidates, detector_report(
            detector, "error", fallback="vlm", error=f"{type(exc).__name__}: {exc}"
        )
    if not available:
        for candidate in candidates:
            candidate.setdefault("proposal_source", "vlm")
        return candidates, detector_report(
            detector,
            "off" if name in ("null", "none") else "unavailable",
            fallback="vlm",
            labels=[],
        )
    labels = target_labels(instruction, candidates)
    if not labels:
        for candidate in candidates:
            candidate.setdefault("proposal_source", "vlm")
        return candidates, detector_report(
            detector, "no_labels", fallback="vlm", labels=[], note="no object phrase"
        )
    started = time.time()
    try:
        detections = list(
            detector.detect(record.frame, instruction, labels=labels)
        )
    except Exception as exc:  # noqa: BLE001 - the VLM path stays usable
        for candidate in candidates:
            candidate.setdefault("proposal_source", "vlm")
        return candidates, detector_report(
            detector,
            "error",
            fallback="vlm",
            labels=labels,
            error=f"{type(exc).__name__}: {exc}",
            elapsed_s=round(time.time() - started, 3),
        )
    elapsed = round(time.time() - started, 3)
    aligned = [
        (detection, box)
        for detection, box in zip(detections, _proposal_boxes(detections))
        if box is not None
    ]
    detections = [item[0] for item in aligned]
    proposal_boxes = [item[1] for item in aligned]
    if not proposal_boxes:
        for candidate in candidates:
            candidate.setdefault("proposal_source", "vlm")
        return candidates, detector_report(
            detector,
            "no_proposal",
            fallback="vlm",
            labels=labels,
            n_proposals=0,
            elapsed_s=elapsed,
        )
    # positions are logged for offline proposal-vs-GT attribution
    proposal_rows: list[dict] = []
    for detection, box in zip(detections, proposal_boxes):
        row = detection.to_dict() if hasattr(detection, "to_dict") else {"box": box}
        row["box"] = [round(float(value), 1) for value in box]
        parsed = parse_box(box, box_format=BOX_FORMAT_XYXY)[0]
        if parsed is not None:
            row.update(_candidate_location(ctx, record, parsed, None, depth_fn, depth_reason))
        proposal_rows.append(row)
    # DINOv2 crop features: the VLM box once, one vector per proposal
    embeddings = None
    embed_fn = getattr(detector, "embed_boxes", None)
    if callable(embed_fn):
        try:
            embeddings = embed_fn(
                record.frame, [candidate["box"] for candidate in candidates] + proposal_boxes
            )
        except Exception:  # noqa: BLE001 - features are optional
            embeddings = None
    candidate_vectors: list[Any] = []
    proposal_vectors: list[Any] = []
    if isinstance(embeddings, (list, tuple)) and len(embeddings) == len(candidates) + len(proposal_boxes):
        candidate_vectors = list(embeddings[: len(candidates)])
        proposal_vectors = list(embeddings[len(candidates):])
    # a proposal that runs into the frame border with a large area has no
    # trustworthy range/identity (its centre reads the surface behind it): it
    # stays in the report but cannot verify or reject a VLM box
    usable_indexes = [
        index
        for index, row in enumerate(proposal_rows)
        if not row.get("border_large_box")
    ]
    authoritative = [
        detections[index]
        for index in usable_indexes
        if float(getattr(detections[index], "confidence", 0.0) or 0.0)
        >= DETECTOR_AUTHORITATIVE_SCORE
    ]
    kept: list[dict] = []
    rejected: list[dict] = []
    matched_proposal_indexes: set[int] = set()
    for index, candidate in enumerate(candidates):
        candidate_box = parse_box(candidate.get("box"), box_format=BOX_FORMAT_XYXY)[0]
        best = None
        if candidate_box is not None:
            for proposal_index in usable_indexes:
                detection = detections[proposal_index]
                proposal_box = proposal_boxes[proposal_index]
                similarity = None
                if candidate_vectors and proposal_vectors:
                    similarity = _detector_embedding_similarity(
                        candidate_vectors[index], proposal_vectors[proposal_index]
                    )
                match = _proposal_match(
                    candidate_box.to_list(), proposal_box, similarity
                )
                if match["matched"] and (
                    best is None or match["_score"] > best[0]["_score"]
                ):
                    best = (match, proposal_index, detection)
        if best is not None:
            match, proposal_index, detection = best
            matched_proposal_indexes.add(proposal_index)
            match.pop("_score", None)
            match["proposal_index"] = proposal_index
            match["proposal_label"] = str(getattr(detection, "label", ""))
            match["proposal_confidence"] = (
                None
                if getattr(detection, "confidence", None) is None
                else round(float(detection.confidence), 3)
            )
            match["proposal_box"] = proposal_rows[proposal_index]["box"]
            candidate["proposal_source"] = "vlm+detector"
            candidate["detector_verified"] = True
            candidate["detector_match"] = match
            kept.append(candidate)
            continue
        if authoritative:
            rejected_row = {
                "id": candidate.get("id"),
                "box": [round(float(value), 1) for value in candidate.get("box") or []],
                "confidence": candidate.get("confidence"),
                "reason": "detector_unmatched",
                "n_authoritative": len(authoritative),
            }
            # a rejected VLM box is still measured, so the offline audit can
            # say whether the gate removed a wrong object or the right one
            rejected_box = parse_box(candidate.get("box"), box_format=BOX_FORMAT_XYXY)[0]
            if rejected_box is not None:
                rejected_location = _candidate_location(
                    ctx, record, rejected_box, candidate.get("distance_m"),
                    depth_fn, depth_reason,
                )
                for key in ("position_flu", "position_source", "range_m", "range_status"):
                    if key in rejected_location:
                        rejected_row[key] = rejected_location[key]
            rejected.append(rejected_row)
            continue
        candidate["proposal_source"] = "vlm"
        candidate["detector_gate"] = "unmatched_low_confidence"
        kept.append(candidate)
    detector_candidates: list[dict] = []
    if not kept:
        for proposal_index, detection in enumerate(detections):
            if proposal_index not in usable_indexes:
                continue
            confidence = float(getattr(detection, "confidence", 0.0) or 0.0)
            if confidence < DETECTOR_AUTHORITATIVE_SCORE:
                continue
            if proposal_index in matched_proposal_indexes:
                continue
            candidate_id = _next_candidate_id(store)
            row = proposal_rows[proposal_index]
            detector_candidates.append(
                {
                    "id": candidate_id,
                    "box": [round(float(value), 1) for value in proposal_boxes[proposal_index]],
                    "box_format": BOX_FORMAT_OUT,
                    "box_format_used": BOX_FORMAT_OUT,
                    "box_uvwh": [
                        round(float(proposal_boxes[proposal_index][0]), 1),
                        round(float(proposal_boxes[proposal_index][1]), 1),
                        round(
                            float(proposal_boxes[proposal_index][2])
                            - float(proposal_boxes[proposal_index][0]),
                            1,
                        ),
                        round(
                            float(proposal_boxes[proposal_index][3])
                            - float(proposal_boxes[proposal_index][1]),
                            1,
                        ),
                    ],
                    "attrs": {
                        "label": str(getattr(detection, "label", "")),
                        "source": name,
                    },
                    "confidence": round(confidence, 2),
                    "distance_m": None,
                    "size_m": None,
                    "matches_description": True,
                    "anchor_visible": None,
                    "match_reason": f"detector proposal ({getattr(detection, 'label', '')})",
                    "uncertainty": "",
                    "instance_rank": len(detector_candidates) + 1,
                    "instance_suspect": False,
                    "instance_suspect_reason": "",
                    "proposal_source": "detector",
                    "detector_verified": True,
                    "detector_proposal": {
                        "label": str(getattr(detection, "label", "")),
                        "confidence": round(confidence, 3),
                        "mask_area_frac": row.get("mask_area_frac"),
                    },
                    "subtask_key": str(subtask_key or ""),
                    "instruction": instruction[:120],
                }
            )
            store[candidate_id] = detector_candidates[-1]
            if len(detector_candidates) >= DETECTOR_EXTRA_CANDIDATES:
                break
    for candidate in kept:
        candidate.setdefault("proposal_source", "vlm")
    report = detector_report(
        detector,
        "ok",
        labels=labels,
        n_proposals=len(detections),
        n_usable_proposals=len(usable_indexes),
        proposals=proposal_rows[:5],
        n_matched=sum(1 for candidate in kept if candidate.get("detector_match")),
        n_rejected=len(rejected),
        rejected=rejected[:8],
        n_detector_candidates=len(detector_candidates),
        authoritative=len(authoritative),
        fallback=None,
        elapsed_s=elapsed,
    )
    return kept + detector_candidates, report


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
    previous_conflicts = int(attempt.get("conflicts", 0)) if attempt is not None else 0
    if previous_conflicts >= STALL_FLAG_AT:
        # the same pixels already produced two unresolved depth-vs-model
        # conflicts: a third call would only mint another unusable candidate
        return _conflict_stagnant_result(ctx, instruction, record, attempt)
    def ask(system_prompt: str, user_prompt: str, frames=None):
        return backend.decide(
            [record.frame] if frames is None else frames,
            system_prompt,
            user_prompt,
            context={
                "observation_id": record.observation_id,
                "instruction": instruction,
                "pose": None if record.position_flu is None else list(record.position_flu),
                "yaw_deg": record.yaw_deg,
                "intrinsics": _observation_intrinsics(ctx, record),
                "tool": "ground_target",
            },
        )

    relation = _relation_anchor(instruction)
    vlm_calls = 0
    try:
        reply = ask(GROUND_SYSTEM_PROMPT, _ground_user_prompt(instruction, relation))
        vlm_calls += 1
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

    store = _candidate_store(ctx)
    subtask_key = _subtask_key(ctx)
    frame_width = int(record.frame.shape[-1])
    frame_height = int(record.frame.shape[-2])
    payload = _extract_json(reply if isinstance(reply, str) else str(reply))
    raw_items = _candidate_items(payload)
    default_anchor_visible = (
        _as_match_flag(payload.get("anchor_visible"))
        if isinstance(payload, Mapping)
        else None
    )
    candidates = (
        _normalize_candidates(
            raw_items,
            store,
            width=frame_width,
            height=frame_height,
            default_anchor_visible=default_anchor_visible,
        )
        if raw_items is not None
        else []
    )
    retry: dict | None = None
    # a scripted backend that locks one answer per case/step opts out: its
    # call log is the experiment's record, and a second call would be a new
    # answer rather than a retry of the same question
    if not candidates and getattr(backend, "allow_box_retry", True):
        # the first answer named no usable box (empty list, prose, or boxes the
        # parser dropped): one cheap box-only retry on the same frame. The first
        # prompt asks for attributes/distance too, and the v5 bedroom run had 15
        # confirms with no box from the view; a second, narrower question is
        # cheaper than sending the drone to a new viewpoint for a box.
        retry_trigger = "unparsable_reply" if raw_items is None else "no_candidate"
        retry_calls = 0
        retry_items = None
        retry_region: list[float] | None = None
        retry_mode = "box_only"
        retry_frame = record.frame
        retry_anchor_box: list[float] | None = None
        hint_box, hint_mode = _retry_hint_box(ctx, record, instruction, subtask_key)
        retry_prompt = BOX_ONLY_USER_PROMPT.format(instruction=instruction)
        if relation is not None:
            retry_prompt += (
                f'\nThe instruction locates the target by the relation '
                f'"{relation.get("anchor")}". Also set "anchor_visible" to '
                "whether that anchor object is visible in the attached image."
            )
        if hint_box is not None:
            # draw the previous candidate's box on a copy: the model gets a
            # visual anchor to correct instead of the v6 blind second look
            retry_frame = _draw_hint_box(record.frame, hint_box)
            retry_anchor_box = list(hint_box)
            retry_mode = hint_mode
            retry_prompt += (
                "\nThe attached frame is the same view with a red rectangle around "
                "the previous candidate. Return the exact box of the instructed "
                "target, correcting the rectangle when it is misplaced."
            )
        else:
            # no previous candidate: enlarge the centre of the frame, where a
            # forward-looking grounding miss usually is, and map the box back
            retry_region = [200.0, 200.0, 600.0, 600.0]
            retry_frame, _size, _upscale, _origin = _crop_frame(record.frame, retry_region)
            retry_mode = "centre_crop"
            retry_prompt += (
                "\nThe attached image is an enlarged central crop of the same view. "
                "Return the target's box in this crop's normalized 0-1000 coordinates."
            )
        try:
            retry_reply = ask(BOX_ONLY_SYSTEM_PROMPT, retry_prompt, [retry_frame])
            retry_calls = 1
            retry_payload = _extract_json(
                retry_reply if isinstance(retry_reply, str) else str(retry_reply)
            )
            retry_items = _candidate_items(retry_payload)
        except Exception:  # noqa: BLE001 - a failed retry is just no candidate
            retry_payload = None
            retry_items = None
        retry_anchor_visible = (
            _as_match_flag(retry_payload.get("anchor_visible"))
            if isinstance(retry_payload, Mapping)
            else None
        )
        retry_candidates = (
            _normalize_candidates(
                retry_items,
                store,
                width=frame_width,
                height=frame_height,
                default_anchor_visible=retry_anchor_visible,
            )
            if retry_items is not None
            else []
        )
        if retry_region is not None:
            for retry_candidate in retry_candidates:
                mapped = _map_box_from_region(retry_candidate.get("box"), retry_region)
                if mapped is None:
                    continue
                retry_candidate["box"] = mapped
                retry_candidate["box_uvwh"] = [
                    mapped[0], mapped[1], mapped[2] - mapped[0], mapped[3] - mapped[1]
                ]
                retry_candidate["retry_region"] = list(retry_region)
        vlm_calls += retry_calls
        retry = {
            "attempted": True,
            "trigger": retry_trigger,
            "vlm_calls": retry_calls,
            "recovered": bool(retry_candidates),
            "system_prompt": "box_only",
            "mode": retry_mode,
            "capture": "same_frame",
        }
        if retry_region is not None:
            retry["region"] = list(retry_region)
        if retry_anchor_box is not None:
            retry["anchor_box"] = list(retry_anchor_box)
        if retry_candidates:
            # mark the retry origin: these candidates are the ones that used to
            # arrive without match/size/attrs and slip past the instance gate
            for retry_candidate in retry_candidates:
                retry_candidate["grounding_retry"] = True
                retry_candidate["retry_mode"] = retry_mode
            candidates = retry_candidates
    # an independent proposal set for the same frame: no-op when the optional
    # backend is off/unavailable, authoritative when it produced usable boxes.
    # It runs before the empty branch so a VLM miss can still become a candidate.
    depth_fn, depth_reason = _frame_depth_reader(ctx, record)
    candidates, detector_result = _apply_detector_gate(
        ctx, record, instruction, candidates, depth_fn, depth_reason, subtask_key
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
            "reason": "unparsable_reply" if raw_items is None else "no_candidate",
            "instruction": instruction,
            "observation_id": record.observation_id,
            "backend": backend_name,
            "attempts": failures,
            "visibility": visibility,
            "scope_note": _scope_text(visibility),
            "memory_negative": negative_write,
            "vlm_calls": vlm_calls,
            "detector": detector_result,
        }
        if retry is not None:
            data["grounding_retry"] = retry
            data["hint"] = (
                "no box even after one box-only retry on the same frame: a new "
                "viewpoint, not another re-grounding of this evidence"
            )
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
    if attempt is None:
        attempt = _evidence_attempt(
            ledger,
            _instruction_signature(instruction),
            _region_signature(record.region),
            record.position_flu,
            record.yaw_deg,
            create=True,
        )
    attempt["attempts"] = int(attempt.get("attempts", 0)) + 1
    attempt["failures"] = 0
    attempt["successes"] = int(attempt.get("successes", 0)) + 1
    attempt["last_observation_id"] = record.observation_id
    attempt["last_reason"] = "candidate"
    for candidate in candidates:
        candidate["observation_id"] = record.observation_id
        candidate["source"] = "ground_target"
        candidate["instruction"] = instruction[:120]
        candidate["subtask_key"] = subtask_key
    # a large/border box is very likely the target plus the wall behind it; the
    # full-frame retry only draws that same box again, so the region is cropped
    # and the model is asked for a tight box inside the crop. The box is replaced
    # before measurement, which is what moves the bearing off the wall.
    tighten_retry: dict | None = None
    if getattr(backend, "allow_box_retry", True):
        tighten_retry, tighten_calls = _tighten_large_candidates(
            record, instruction, candidates, ask
        )
        vlm_calls += tighten_calls
    for candidate in candidates:
        box = parse_box(candidate["box"], box_format=BOX_FORMAT_XYXY)[0]
        if box is not None:
            candidate.update(
                _candidate_location(ctx, record, box, candidate["distance_m"], depth_fn, depth_reason)
            )
            tightened = candidate.get("tighten_retry")
            if (
                isinstance(tightened, Mapping)
                and tightened.get("accepted")
                and candidate.get("position_source") == POSITION_MODEL_DISTANCE
            ):
                # the tightened bearing, placed at the model's range, must not
                # point away from a trusted depth/triangulated position of the
                # same target: if it does, this tight box is not that instance
                sanity = _tighten_bearing_conflict(
                    ctx, record, candidate, box, candidate.get("distance_m")
                )
                if sanity is not None:
                    tightened["bearing_sanity"] = "conflict"
                    candidate["distance_source"] = DISTANCE_CONFLICT
                    candidate["distance_conflict"] = sanity
                    candidate["distance_uncertain"] = True
            advice = candidate.get("view_change_advice")
            if isinstance(advice, dict) and isinstance(advice.get("args"), dict):
                if advice["args"].get("instruction") is None:
                    advice["args"]["instruction"] = instruction
            # a re-grounding of a candidate whose instance/place a confirmation
            # already rejected inherits that verdict (v7 bedroom seed37 minted
            # the same 21.3-deg point five times after different_instance)
            rejection = _candidate_rejection_match(ctx, candidate)
            if rejection is not None:
                entry, match_kind = rejection
                # a fresh depth measurement is new evidence against a same-box
                # match, but not against the rejected place or bearing
                superseded = (
                    match_kind == "box"
                    and _has_independent_depth_measurement(candidate)
                )
                if not superseded:
                    _mark_inherited_rejection(candidate, entry)
                    rejection = entry
                else:
                    rejection = None
            if rejection is None:
                # a second view of the same target is a stereo pair: the ray
                # intersection replaces the single-view range and makes the
                # candidate measurable even when neither frame's depth was
                diagnostics: dict = {}
                triangulated = _triangulate_candidate(
                    ctx, candidate, record, diagnostics
                )
                if triangulated is not None:
                    single_view_conflict = candidate.get("distance_conflict")
                    candidate.update(triangulated)
                    if single_view_conflict is not None:
                        candidate["single_view_distance_conflict"] = single_view_conflict
                    candidate["distance_conflict"] = {
                        "resolved": True,
                        "resolution": "triangulation",
                        "single_view_conflict": single_view_conflict,
                        **candidate["triangulation"],
                    }
                    candidate["view_change_advice"] = None
                    rejection = _candidate_rejection_match(ctx, candidate)
                    if rejection is not None:
                        _mark_inherited_rejection(candidate, rejection[0])
                    else:
                        merged = store.get(
                            str(candidate["triangulation"]["merged_candidate_id"])
                        )
                        if isinstance(merged, dict):
                            merged["superseded_by"] = candidate["id"]
                            merged["status"] = "merged"
                            merged["merged_into"] = candidate["id"]
                elif isinstance(diagnostics.get("rejection"), Mapping):
                    # the geometric pair is valid but its intersection is on the
                    # rejection list: report the candidate as rejected instead
                    # of silently dropping the second bearing
                    _mark_inherited_rejection(candidate, diagnostics["rejection"])
        else:  # pragma: no cover - the record's box came from this parser
            candidate["position_source"] = POSITION_NO_RANGE
            candidate["range_status"] = "unknown"
    # every located candidate is judged before anything can consume it: a box
    # that is the wall/neighbouring object (or a second, ambiguous instance)
    # stays in the store as evidence but carries ``instance_suspect``.
    instance_filter = _instance_consistency(ctx, record, instruction, candidates)
    for candidate in candidates:
        store[candidate["id"]] = candidate
    conflict_ids = [
        str(candidate["id"])
        for candidate in candidates
        if str(candidate.get("distance_source") or "") == DISTANCE_CONFLICT
    ]
    if conflict_ids:
        attempt["conflicts"] = int(attempt.get("conflicts", 0)) + len(conflict_ids)
    else:
        attempt["conflicts"] = 0
    evidence.extend(EvidenceRef("memory", candidate["id"], "candidate") for candidate in candidates)
    # every candidate is a sighting of *this* frame: image + candidate + basis,
    # which is what makes a later "where did I see it" answer evidence instead
    # of a frame that merely inherited the task's target name
    extra = ctx.extra if isinstance(ctx.extra, dict) else {}
    identity_anchor = str(extra.get("identity_anchor") or "").strip()
    sighting_write = _MemoryBridge(ctx.memory).add_sighting(
        record.observation_id,
        [_sighting_of_candidate(candidate, instruction, record, anchor=identity_anchor)
         for candidate in candidates],
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
        "vlm_calls": vlm_calls,
        "instance_filter": instance_filter,
        "detector": detector_result,
    }
    if relation_chain_enabled(ctx):
        chain_plan = relation_observation_plan(
            ctx, instruction,
            candidate_ids=[str(item.get("id") or "") for item in candidates],
            relation=relation,
        )
        data["relation_chain"] = relation_state_summary(ctx)
        data["observation_plan"] = chain_plan
        anchor_task = (
            instance_filter.get("relation_chain", {}).get("temporary_anchor_subtask")
            if isinstance(instance_filter.get("relation_chain"), Mapping)
            else None
        )
        if isinstance(anchor_task, Mapping) and anchor_task.get("created"):
            data["temporary_anchor_subtask"] = dict(anchor_task)
            data["observation_decision"] = record_observation_decision(
                ctx,
                uncertainty=UNCERTAINTY_ANCHOR_RELATION,
                action=chain_plan.get("next_action"),
                tried_to_distinguish=(
                    "whether the relation anchor exists/located vs only its "
                    "phrase in the instruction"
                ),
                before={"anchor_established": False},
                after={
                    "anchor_established": False,
                    "anchor_subtask": anchor_task.get("anchor_subtask"),
                },
                changed=True,
                note="dependent anchor sub-task created; target candidates and "
                     "budget preserved",
            )
        if isinstance(chain_plan.get("next_action"), Mapping):
            data.setdefault("next_action", dict(chain_plan["next_action"]))
    suspect_ids = [
        str(item.get("id")) for item in candidates if item.get("instance_suspect")
    ]
    if suspect_ids:
        data["instance_suspect_candidates"] = suspect_ids
    if instance_filter.get("approvable"):
        data["approvable_candidates"] = list(instance_filter["approvable"])
    if suspect_ids and not instance_filter.get("approvable"):
        data["hint"] = (
            "every box in this reply failed the instance check (wall/background, "
            "attribute conflict or size/range contradiction): change viewpoint "
            "and ground the object itself instead of approaching one of these"
        )
    if retry is not None:
        data["grounding_retry"] = retry
    if tighten_retry is not None:
        data["tighten_retry"] = tighten_retry
        if tighten_retry.get("accepted"):
            data["tighten_accepted_candidates"] = [
                str(item.get("candidate"))
                for item in tighten_retry.get("candidates") or []
                if item.get("accepted")
            ]
    if conflict_ids:
        data["distance_conflict_candidates"] = conflict_ids
        hint = (
            "these candidates' depth contradicts the model's distance estimate "
            "(the box is likely background): confirmation will keep them "
            "uncertain - observe from a different heading/altitude and ground "
            "the object's support point instead of repeating this view"
        )
        data["hint"] = f"{data['hint']} {hint}" if data.get("hint") else hint
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
_SUMMARY_TEXTS = ("hyp_id", "id", "kind", "status", "content", "note", "target_name", "event",
                  "anchor", "anchor_source")
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
    identity = hit.get("identity")
    if isinstance(identity, Mapping):
        entry["identity"] = _memory_jsonable(dict(identity))
        if not entry.get("anchor") and identity.get("anchor"):
            entry["anchor"] = str(identity.get("anchor"))
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
    index = _identity_index(ctx)
    out = []
    for place in places or []:
        if not isinstance(place, Mapping):
            continue
        item = _memory_jsonable(dict(place))
        for candidate in item.get("candidates") or []:
            identity = index.get(str(candidate))
            if identity is None:
                continue
            item["identity"] = {
                "identity_id": identity.get("identity_id"),
                "candidate_id": identity.get("candidate_id"),
                "target": identity.get("target"),
                "verdict": identity.get("verdict"),
                "revision": identity.get("revision"),
            }
            if identity.get("anchor"):
                item["anchor"] = identity.get("anchor")
            item["identity_priority"] = identity.get("verdict") == "accepted"
            break
        out.append(item)
    out.sort(key=lambda item: 0 if item.get("identity") else (1 if item.get("anchor") else 2))
    return out


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
    identity_rows = []
    identities = getattr(memory, "identities", None) if memory is not None else None
    if identities is not None:
        all_fn = getattr(identities, "all", None)
        if callable(all_fn):
            try:
                identity_rows = list(all_fn(limit))[:limit]
            except Exception:  # noqa: BLE001 - the layer is optional
                identity_rows = []
    extra = {
        "open_questions": list(state.get("open_questions") or [])[:limit],
        "anchors": [
            dict(item) for item in (getattr(working, "anchors", lambda limit=None: [])(limit=limit) or [])
            if isinstance(item, Mapping)
        ],
        "identities": identity_rows,
        "subgoal": state.get("subgoal"),
        "candidates": list(state.get("candidates") or [])[:limit],
        "failures": list(state.get("failures") or [])[:limit],
        "counts": state.get("counts"),
    }
    return _memory_jsonable(extra)


def _identity_hits(ctx: ToolContext, query, limit: int = 5) -> list[dict]:
    """The identity layer's answer to a retrieval (``[]`` when it is off).

    An identity hit says which object instance the query is about: its
    fingerprint (target, box, position), the relation anchor it was found
    under, and the confirmation verdict/revision taken on it. It is returned
    next to the episodic hits because "where was it" and "which one was it" are
    two questions a revisit has to answer together.
    """
    memory = getattr(ctx, "memory", None)
    identities = getattr(memory, "identities", None) if memory is not None else None
    fn = getattr(identities, "query", None)
    if not callable(fn):
        return []
    spec = query if isinstance(query, Mapping) else {"text": query}
    text = spec.get("text") or spec.get("target") or spec.get("goal_target") or ""
    try:
        hits = fn(text, limit=max(1, min(int(limit), 8))) if text else fn({}, limit=max(1, min(int(limit), 8)))
    except Exception:  # noqa: BLE001 - the layer is optional
        return []
    out = []
    for hit in hits or []:
        if not isinstance(hit, Mapping):
            continue
        item = _memory_jsonable(dict(hit))
        item["identity"] = {
            "identity_id": item.get("identity_id"),
            "candidate_id": item.get("candidate_id"),
            "target": item.get("target"),
            "verdict": item.get("verdict"),
            "revision": item.get("revision"),
        }
        item["anchors"] = [item["anchor"]] if item.get("anchor") else []
        out.append(item)
    return out


def _identity_index(ctx: ToolContext) -> dict:
    """``candidate_id -> identity entry`` for attaching fields to episodic hits."""
    memory = getattr(ctx, "memory", None)
    identities = getattr(memory, "identities", None) if memory is not None else None
    fn = getattr(identities, "all", None)
    if not callable(fn):
        return {}
    try:
        entries = fn()
    except Exception:  # noqa: BLE001 - the layer is optional
        return {}
    index = {}
    for entry in entries or []:
        if not isinstance(entry, Mapping):
            continue
        candidate = str(entry.get("candidate_id") or "")
        if candidate:
            index[candidate] = entry
    return index


def _summary_candidates(summary: Mapping) -> list[str]:
    """Candidate ids a model-facing hit cites (sighting / identity / use args)."""
    candidates: list[str] = []
    sighting_items = []
    if isinstance(summary.get("sighting"), Mapping):
        sighting_items.append(summary["sighting"])
    if isinstance(summary.get("sightings"), list):
        sighting_items.extend(
            item for item in summary["sightings"] if isinstance(item, Mapping)
        )
    for source in [*sighting_items, summary.get("identity") or {}, summary.get("use") or {}]:
        if not isinstance(source, Mapping):
            continue
        candidate = str(source.get("candidate") or source.get("candidate_id") or "")
        if candidate and candidate not in candidates:
            candidates.append(candidate)
        args = source.get("args")
        if isinstance(args, Mapping):
            candidate = str(args.get("candidate_id") or args.get("candidate") or "")
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _rank_by_identity(ctx: ToolContext, summaries: list[dict]) -> int:
    """Put identity- and anchor-bearing hits first; returns how many moved.

    The order is the goal's priority: a hit that names the confirmed instance
    (or its relation anchor) outranks a generic frame; among the rest the
    retrieval's own score/recency order is kept exactly as it was, so this is a
    stable partition, not a re-scoring of the episodic layer.
    """
    index = _identity_index(ctx)
    if not index:
        return 0
    ranked = []
    for position, summary in enumerate(summaries):
        identity = None
        for candidate in _summary_candidates(summary):
            identity = index.get(candidate)
            if identity is not None:
                summary["identity"] = {
                    "identity_id": identity.get("identity_id"),
                    "candidate_id": identity.get("candidate_id"),
                    "target": identity.get("target"),
                    "verdict": identity.get("verdict"),
                    "revision": identity.get("revision"),
                }
                if identity.get("anchor"):
                    summary["anchor"] = identity.get("anchor")
                    summary["anchor_source"] = identity.get("anchor_source")
                break
        if identity is not None and identity.get("verdict") == "accepted":
            priority = 0
        elif identity is not None:
            priority = 1
        elif summary.get("anchor"):
            priority = 2
        else:
            priority = 3
        ranked.append((priority, position, summary))
    ranked.sort(key=lambda row: (row[0], row[1]))
    summaries[:] = [row[2] for row in ranked]
    return sum(1 for row in ranked if row[0] < 3)


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
    identity_hits = _identity_hits(ctx, query)
    if not hits:
        data = _no_match_data(ctx, memory, query)
        if identity_hits:
            data["identities"] = identity_hits
            data["anchors"] = [item["anchor"] for item in identity_hits if item.get("anchor")]
            data["hint"] = (
                "no episodic hit matches, but the identity layer remembers these "
                "instances/anchors: use the candidate_id with compare_candidates "
                "or approach_candidate"
            )
        return ToolResult(STATUS_EMPTY, data, [])
    goal_target = _goal_target_of(ctx)
    ranked = _rank_by_identity(ctx, summary_hits)
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
        "identity_priority_hits": ranked,
    }
    if identity_hits:
        data["identities"] = identity_hits
        data["anchors"] = [item["anchor"] for item in identity_hits if item.get("anchor")]
        data["identity_note"] = (
            "identities/anchors first: these are the object fingerprints and "
            "relation anchors you recorded (with the confirm verdict/revision), "
            "not a fresh observation"
        )
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
