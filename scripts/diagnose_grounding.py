#!/usr/bin/env python
"""Offline perception-failure bench for the grounding chain (Phase 2).

The real M2 trajectory loops ``observe -> navigate(obs_10) ->
unknown_observation -> navigate(obs_9) -> already_there`` with zero
displacement: the chain never turns "the thing I see" into "where it is". The
tool answers ``no_candidate`` and the loop repeats, so every retry spends tokens
without telling anyone *why* - the model did not see it, wrote a box the parser
rejected, was handed the wrong frame, or the projection threw the box away.
Repeated calls cannot separate those cases; a fixed observation set can.

This script therefore does the opposite of the loop: it freezes a batch of real
observations (four situations - clearly visible / not visible / partly occluded /
small target), replays several grounding *strategies* on each of them offline,
and writes down for every single model call the raw reply, the exact image that
was sent, the parsed result and a failure category. ``no_candidate`` is then one
row of a distribution instead of the only conclusion.

Strategies compared per observation:

``structured_box``   today's tool: ``ground_target`` on the frame (box + attrs)
``point_pixel``      the model points at the target centre pixel (no box)
``region_then_local````inspect_region`` on a model-chosen region, then ground
                     inside the magnified crop (measures the zoom chain)
``detector_slot``    a detection/segmentation backend, if one is wired
                     (:class:`DetectorBackend`; the default slot is empty on
                     purpose and is reported as such, never faked)

Failure taxonomy (``failure.category``): ``not_seen`` (a visible target the model
did not report / reported elsewhere), ``output_format`` (unparsable reply, no
box, box outside the frame), ``wrong_image`` (the pixels the model received are
not the observation's), ``projection_error`` (a usable answer whose world
position, intrinsics or ray contradicts the coordinate contract) and ``other``
(backend exceptions, false positives, an empty detector slot). Correct
abstention on an observation where the target really is invisible is ``none`` -
a miss and an honest "I do not see it" are not the same row.

Coordinate-contract audits run on every call, because the three known bug
classes are invisible in a success rate: the box format each side used
(``box_format_used``), the intrinsics the projection ran with versus the
observation's own (a recomputed FOV deviation, including the 86 deg default of
``world.pixel_to_body_offset``), and the "unknown range stays a ray" rule (no
fabricated ``position_flu`` without a range). A fourth check compares the range a
tool projected with against the depth at the answer's own pixel, which is how a
box written in a magnified crop's coordinates (and therefore a depth read from
some other pixel) shows up as itself instead of as a mysterious few meters of
position error.

The answer key of each frozen observation is the analytic projection of the
target gated by the *rendered* depth map where the world has one: an occluder's
splats cover more than the box they stand for, so "partly occluded" is chosen
(and scored) in the picture the model was shown rather than at the geometric
boundary where the picture is already fully hidden.

``--inject-point`` / ``--inject-waypoint`` / ``--inject-wrong-image`` are
*diagnosis-only* replacements: they bypass the model to test what happens after
grounding (projection chain, executor) or to prove that a wrong-frame bug is
caught. Every record they produce is flagged ``diagnosis_only``, is reported in
its own section and is excluded from the strategy score.

Usage (mock / scripted, no weights):

    python scripts/diagnose_grounding.py --backend mock --out /tmp/diag \\
        --mock-replies replies.json

Usage (the local Qwen3-VL-8B, container `vlm-nav`, GPU 0):

    CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_grounding.py --backend hf \\
        --model-path /models_ro/Qwen3-VL-8B --device cuda \\
        --out outputs/vlm_nav/diagnose_grounding/hf_8b

Artifacts under ``--out``: ``observations/{manifest.json,case_*.png}`` (the fixed
set - reuse it with ``--source manifest`` so two runs are comparable),
``calls.jsonl`` (one line per model call, the full audit record),
``inputs/*.png`` (the exact image sent, one file per call), ``summary.json``,
``diagnose_grounding.md`` (the report) and ``run_config.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

from vlm_nav.actions import body_delta_to_world
from vlm_nav.tool_impl_perception import (
    CROP_MAX_UPSCALE,
    CROP_SHORT_SIDE_PX,
    MAX_RANGE_M,
    MIN_RANGE_M,
    POSITION_INTRINSICS_UNKNOWN,
    POSITION_NO_RANGE,
    POSITION_RAY,
    ground_target,
    inspect_region,
    parse_box,
)
from vlm_nav.observations import KIND_EVIDENCE, register_observation
from vlm_nav.tools import (
    STATUS_OK,
    ToolContext,
    ToolResult,
)
from vlm_nav.vlm import VLMBackend
from vlm_nav.world import pixel_to_body_offset

# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------

MANIFEST_FORMAT = "vlm_nav.diagnose_grounding.observations/1"
RUN_FORMAT = "vlm_nav.diagnose_grounding.run/1"

#: the four situations the fixed observation set must cover
CASE_VISIBLE = "visible"
CASE_ABSENT = "absent"
CASE_PARTIAL = "partial"
CASE_SMALL = "small"
CASE_KINDS = (CASE_VISIBLE, CASE_ABSENT, CASE_PARTIAL, CASE_SMALL)

STRATEGY_STRUCTURED = "structured_box"
STRATEGY_POINT = "point_pixel"
STRATEGY_REGION = "region_then_local"
STRATEGY_DETECTOR = "detector_slot"
STRATEGIES = (STRATEGY_STRUCTURED, STRATEGY_POINT, STRATEGY_REGION, STRATEGY_DETECTOR)

FAILURE_NONE = "none"
FAILURE_NOT_SEEN = "not_seen"
FAILURE_OUTPUT_FORMAT = "output_format"
FAILURE_WRONG_IMAGE = "wrong_image"
FAILURE_PROJECTION = "projection_error"
FAILURE_OTHER = "other"
FAILURE_CATEGORIES = (
    FAILURE_NONE,
    FAILURE_NOT_SEEN,
    FAILURE_OUTPUT_FORMAT,
    FAILURE_WRONG_IMAGE,
    FAILURE_PROJECTION,
    FAILURE_OTHER,
)
#: what a "not run" row reports (a slot nobody filled is not a failure of the model)
FAILURE_SKIPPED = "skipped"

OUTCOME_HIT = "hit"
OUTCOME_MISS = "miss"
OUTCOME_ABSTAIN = "abstain_correct"
OUTCOME_FALSE_POSITIVE = "false_positive"
OUTCOME_ERROR = "error"
OUTCOME_NOT_RUN = "not_run"
#: an audited step of a multi-step strategy that is not scored on its own
OUTCOME_INTERMEDIATE = "step_intermediate"

#: the visibility floor that makes a target "in this observation at all"
VISIBLE_FRACTION_MIN = 0.05
#: polar sample grid of the target disc (ring radius as a fraction of the
#: target radius, number of samples on that ring) - the visible fraction and the
#: visible box are computed on it
TARGET_RINGS = ((0.35, 4), (0.7, 8), (1.0, 12))

#: tolerance defaults of a "hit" (overridable on the CLI)
DEFAULT_CENTER_TOLERANCE_PX = 48.0
DEFAULT_IOU_MIN = 0.10
DEFAULT_POSITION_TOLERANCE_M = 1.5
#: the field of view ``world.pixel_to_body_offset`` assumes when a caller does
#: not pass one - the value the audit recomputes the deviation against
LEGACY_DEFAULT_HFOV_DEG = 86.0


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------


def wrap_deg(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


def _floats(value: Any, count: int = 3) -> tuple[float, ...] | None:
    """``value`` as ``count`` finite floats, or None (never a fabricated 0)."""
    if value is None:
        return None
    try:
        numbers = [float(v) for v in list(value)[:count]]
    except (TypeError, ValueError):
        return None
    if len(numbers) != count or not all(math.isfinite(v) for v in numbers):
        return None
    return tuple(numbers)


def _round_box(box: Sequence[float] | None, digits: int = 1) -> list[float] | None:
    if box is None:
        return None
    return [round(float(v), digits) for v in box]


def _box_or_none(value: Any) -> list[float] | None:
    """A stored box as four floats, or None (a wrong-length box is no box)."""
    numbers = _floats(value, 4)
    return None if numbers is None else _round_box(numbers)


def _point_or_none(value: Any, digits: int = 2) -> list[float] | None:
    """A stored normalized point as two floats, or None."""
    numbers = _floats(value, 2)
    return None if numbers is None else [round(float(v), digits) for v in numbers]


def _point_segment_distance(point, segment) -> float:
    p = np.asarray(point, dtype=np.float64).reshape(3)
    a, b = np.asarray(segment[0], dtype=np.float64), np.asarray(segment[1], dtype=np.float64)
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom <= 1e-12 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


# ---------------------------------------------------------------------------
# scene geometry: occlusion + projection ground truth
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Occluder:
    """Axis-aligned box (centre + full size, metres, FLU)."""

    center: tuple[float, float, float]
    size: tuple[float, float, float]

    @property
    def lo(self) -> tuple[float, float, float]:
        return tuple(c - s / 2.0 for c, s in zip(self.center, self.size))

    @property
    def hi(self) -> tuple[float, float, float]:
        return tuple(c + s / 2.0 for c, s in zip(self.center, self.size))

    def to_dict(self) -> dict:
        return {
            "center": [round(v, 3) for v in self.center],
            "size": [round(v, 3) for v in self.size],
        }


def occluders_from(value: Any) -> list[Occluder]:
    """Occluder boxes from manifest/JSON spellings (dicts or objects)."""
    out: list[Occluder] = []
    for item in value or ():
        center = _floats(getattr(item, "center", None) if not isinstance(item, Mapping)
                         else item.get("center"), 3)
        size = _floats(getattr(item, "size", None) if not isinstance(item, Mapping)
                       else item.get("size"), 3)
        if center is not None and size is not None:
            out.append(Occluder(center, size))
    return out


def box_blocks(p0: Iterable[float], p1: Iterable[float], box: Occluder, eps: float = 1e-9) -> bool:
    """Does the segment ``p0`` -> ``p1`` pass through ``box``? (slab method)

    The same test the M2 scenario driver uses, applied to single sample points of
    the target disc so "partly occluded" is a measured fraction, not a guess.
    """
    lo, hi = box.lo, box.hi
    t0, t1 = 0.0, 1.0
    for axis in range(3):
        a, b = float(p0[axis]), float(p1[axis])
        delta = b - a
        if abs(delta) < eps:
            if a < lo[axis] or a > hi[axis]:
                return False
            continue
        ta = (lo[axis] - a) / delta
        tb = (hi[axis] - a) / delta
        if ta > tb:
            ta, tb = tb, ta
        t0 = max(t0, ta)
        t1 = min(t1, tb)
        if t0 > t1:
            return False
    return True


@dataclass(frozen=True)
class Intrinsics:
    """The render intrinsics of one observation (``hfov_deg`` may be unknown)."""

    width: int
    height: int
    hfov_deg: float | None

    @property
    def fx(self) -> float | None:
        if not self.hfov_deg or self.width <= 0:
            return None
        return self.width / (2.0 * math.tan(math.radians(self.hfov_deg) / 2.0))

    @property
    def vfov_deg(self) -> float | None:
        if not self.hfov_deg or self.width <= 0:
            return None
        return math.degrees(
            2.0 * math.atan(math.tan(math.radians(self.hfov_deg) / 2.0) * self.height / self.width)
        )

    @property
    def fy(self) -> float | None:
        vfov = self.vfov_deg
        if vfov is None or self.height <= 0:
            return None
        return self.height / (2.0 * math.tan(math.radians(vfov) / 2.0))

    def to_dict(self) -> dict:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "hfov_deg": None if self.hfov_deg is None else round(float(self.hfov_deg), 4),
        }


def intrinsics_of(value: Mapping | None, *, width: int = 0, height: int = 0) -> Intrinsics:
    """Intrinsics of a manifest/observation mapping (unknown FOV stays None)."""
    data = value if isinstance(value, Mapping) else {}
    hfov = data.get("hfov_deg", data.get("hfov"))
    try:
        hfov = None if hfov is None else float(hfov)
    except (TypeError, ValueError):
        hfov = None
    if hfov is not None and not (0.0 < hfov < 180.0):
        hfov = None
    try:
        w = int(data.get("width") or width or 0)
        h = int(data.get("height") or height or 0)
    except (TypeError, ValueError):
        w, h = int(width or 0), int(height or 0)
    return Intrinsics(w, h, hfov)


def camera_frame_offset(
    point_flu: Iterable[float],
    pose_flu: Iterable[float],
    yaw_deg: float,
) -> tuple[float, float, float] | None:
    """(forward, left, up) offsets of a world point in the capture's body frame."""
    point = _floats(point_flu, 3)
    pose = _floats(pose_flu, 3)
    if point is None or pose is None:
        return None
    dx, dy, dz = (point[i] - pose[i] for i in range(3))
    yaw = math.radians(float(yaw_deg))
    return (
        math.cos(yaw) * dx + math.sin(yaw) * dy,
        -math.sin(yaw) * dx + math.cos(yaw) * dy,
        dz,
    )


def project_point(
    point_flu: Iterable[float],
    pose_flu: Iterable[float],
    yaw_deg: float,
    intrinsics: Intrinsics,
) -> tuple[float, float] | None:
    """World point -> normalized [0,1000] pixel of an observation, or None.

    The pinhole model is the observation's own: forward/left/up offsets in the
    capture's yaw-only body frame, then the intrinsics' focal length. A point
    behind the camera (or a frame with no declared FOV) has no pixel.
    """
    if intrinsics.fx is None or intrinsics.fy is None:
        return None
    offset = camera_frame_offset(point_flu, pose_flu, yaw_deg)
    if offset is None:
        return None
    forward, left, up = offset
    if forward <= 1e-6:
        return None
    u = intrinsics.width / 2.0 - intrinsics.fx * left / forward
    v = intrinsics.height / 2.0 - intrinsics.fy * up / forward
    return (u / intrinsics.width * 1000.0, v / intrinsics.height * 1000.0)


def bearing_deg(point_flu: Iterable[float], pose_flu: Iterable[float]) -> float | None:
    point, pose = _floats(point_flu, 3), _floats(pose_flu, 3)
    if point is None or pose is None:
        return None
    return math.degrees(math.atan2(point[1] - pose[1], point[0] - pose[0]))


def point_visible(
    point_flu: Iterable[float],
    pose_flu: Iterable[float],
    yaw_deg: float,
    intrinsics: Intrinsics,
    occluders: Sequence[Occluder] = (),
) -> bool:
    """Is one world point inside the frame and not behind an occluder?"""
    pixel = project_point(point_flu, pose_flu, yaw_deg, intrinsics)
    if pixel is None:
        return False
    if not (0.0 <= pixel[0] <= 1000.0 and 0.0 <= pixel[1] <= 1000.0):
        return False
    return not any(box_blocks(pose_flu, point_flu, box) for box in occluders)


def target_samples(
    target_pos: Iterable[float],
    radius_m: float,
    pose_flu: Iterable[float],
) -> list[tuple[float, float, float]]:
    """Sample points of the target disc as it faces the camera.

    The target is treated as a sphere of ``radius_m``; the samples live on a
    polar grid of the disc perpendicular to the view axis, so occluding a slice
    of the target shows up as a fraction instead of an all-or-nothing answer.
    """
    target = np.asarray(_floats(target_pos, 3) or (0.0, 0.0, 0.0), dtype=np.float64)
    pose = np.asarray(_floats(pose_flu, 3) or (0.0, 0.0, 0.0), dtype=np.float64)
    axis = target - pose
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return [tuple(float(v) for v in target)]
    normal = axis / norm
    helper = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(normal @ helper)) > 0.9:
        helper = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    e1 = np.cross(normal, helper)
    e1 = e1 / max(float(np.linalg.norm(e1)), 1e-12)
    e2 = np.cross(normal, e1)
    radius = float(radius_m)
    points: list[tuple[float, float, float]] = []
    for ring, count in TARGET_RINGS:
        for index in range(count):
            angle = 2.0 * math.pi * index / count
            offset = radius * ring * (math.cos(angle) * e1 + math.sin(angle) * e2)
            points.append(tuple(float(v) for v in (target + offset)))
    return points


@dataclass
class CaseTruth:
    """What the observation *really* contains - the answer key of one case."""

    visible: bool
    visible_fraction: float
    occluded_fraction: float
    in_fov: bool
    offset_deg: float | None
    distance_m: float
    box: list[float] | None
    visible_box: list[float] | None
    position_flu: list[float]
    samples: int
    #: centroid of the pixels the target is actually visible in - a point that is
    #: on the target by construction (the bounding box centre of a partly hidden
    #: target can sit on the occluder, which is what makes it a bad reference for
    #: an "inject the correct image point" check)
    visible_center: list[float] | None = None
    source: str = "analytic_projection"

    def to_dict(self) -> dict:
        return {
            "visible": bool(self.visible),
            "visible_fraction": round(float(self.visible_fraction), 3),
            "occluded_fraction": round(float(self.occluded_fraction), 3),
            "in_fov": bool(self.in_fov),
            "offset_deg": None if self.offset_deg is None else round(float(self.offset_deg), 1),
            "distance_m": round(float(self.distance_m), 3),
            "box": _round_box(self.box),
            "visible_box": _round_box(self.visible_box),
            "visible_center": None if self.visible_center is None else _point_or_none(self.visible_center),
            "position_flu": [round(float(v), 3) for v in self.position_flu],
            "samples": int(self.samples),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "CaseTruth":
        return cls(
            visible=bool(data.get("visible")),
            visible_fraction=float(data.get("visible_fraction") or 0.0),
            occluded_fraction=float(data.get("occluded_fraction") or 0.0),
            in_fov=bool(data.get("in_fov")),
            offset_deg=data.get("offset_deg"),
            distance_m=float(data.get("distance_m") or 0.0),
            box=_box_or_none(data.get("box")),
            visible_box=_box_or_none(data.get("visible_box")),
            visible_center=_point_or_none(data.get("visible_center")),
            position_flu=[float(v) for v in (data.get("position_flu") or [])],
            samples=int(data.get("samples") or 0),
            source=str(data.get("source") or "analytic_projection"),
        )


def target_ground_truth(
    target_pos: Iterable[float],
    radius_m: float,
    pose_flu: Iterable[float],
    yaw_deg: float,
    intrinsics: Intrinsics,
    occluders: Sequence[Occluder] = (),
    *,
    visible_min: float = VISIBLE_FRACTION_MIN,
    depth_fn: Callable[[tuple[float, float]], Any] | None = None,
    depth_margin_frac: float = 0.12,
    depth_margin_min: float = 0.3,
) -> CaseTruth:
    """The answer key: projection box, visible box, visible fraction, distance.

    ``box`` is the projection of the whole target, ``visible_box`` the bounding
    box of the part the camera can actually see (unoccluded *and* inside the
    frame) - for the partly occluded case that is the only fair reference, since
    the hidden half cannot be pointed at. ``visible`` is the visible fraction
    crossing ``visible_min``; a target whose samples are all outside the frame or
    all behind the wall is ``visible=False`` and has no box at all.

    With ``depth_fn`` (the rendered depth map of the observation) the occlusion
    test is the *rendered* one: a sample is hidden when a nearer surface is
    rendered in its pixel. That is the picture the model was shown, and it is the
    honest oracle where an occluder's splats cover more than the geometric box
    they stand for (the boundary of a partly hidden target); without it the
    analytic box test is used and ``depth_fn=None`` says so in the source name.
    """
    target = _floats(target_pos, 3) or (0.0, 0.0, 0.0)
    pose = _floats(pose_flu, 3) or (0.0, 0.0, 0.0)
    samples = target_samples(target, radius_m, pose)
    all_pixels: list[tuple[float, float]] = []
    seen_pixels: list[tuple[float, float]] = []
    unoccluded = 0
    depth_reads = 0
    for point in samples:
        offset = camera_frame_offset(point, pose, yaw_deg)
        axial = None if offset is None else offset[0]
        blocked = None
        if depth_fn is not None and axial is not None and axial > 0.0:
            pixel = project_point(point, pose, yaw_deg, intrinsics)
            if pixel is not None:
                depth_reads += 1
                try:
                    depth = depth_fn((float(pixel[0]), float(pixel[1])))
                except Exception:  # noqa: BLE001 - an unreadable pixel decides nothing
                    depth = None
                try:
                    depth = None if depth is None else float(depth)
                except (TypeError, ValueError):
                    depth = None
                if depth is not None and math.isfinite(depth) and depth > 0.0:
                    margin = max(float(depth_margin_min), float(depth_margin_frac) * axial)
                    blocked = bool(depth < axial - margin)
        if blocked is None:
            blocked = any(box_blocks(pose, point, box) for box in occluders)
        unoccluded += 0 if blocked else 1
        pixel = project_point(point, pose, yaw_deg, intrinsics)
        if pixel is None:
            continue
        all_pixels.append(pixel)
        if not blocked and 0.0 <= pixel[0] <= 1000.0 and 0.0 <= pixel[1] <= 1000.0:
            seen_pixels.append(pixel)

    def bbox(pixels: Sequence[tuple[float, float]]) -> list[float] | None:
        if not pixels:
            return None
        xs = [p[0] for p in pixels]
        ys = [p[1] for p in pixels]
        return [min(xs), min(ys), max(xs), max(ys)]

    total = max(1, len(samples))
    visible_fraction = len(seen_pixels) / total
    occluded_fraction = 1.0 - unoccluded / total
    offset = None
    centre = project_point(target, pose, yaw_deg, intrinsics)
    bearing = bearing_deg(target, pose)
    if centre is not None and bearing is not None:
        offset = wrap_deg(bearing - float(yaw_deg))
    visible_center = None
    if seen_pixels:
        visible_center = [
            float(np.mean([pixel[0] for pixel in seen_pixels])),
            float(np.mean([pixel[1] for pixel in seen_pixels])),
        ]
    distance = float(
        np.linalg.norm(
            np.asarray(target, dtype=np.float64) - np.asarray(pose, dtype=np.float64)
        )
    )
    return CaseTruth(
        visible=bool(visible_fraction >= visible_min),
        visible_fraction=visible_fraction,
        occluded_fraction=max(0.0, occluded_fraction),
        in_fov=bool(centre is not None and abs(offset or 0.0) <= (intrinsics.hfov_deg or 0.0) / 2.0),
        offset_deg=offset,
        distance_m=distance,
        box=bbox(all_pixels),
        visible_box=bbox(seen_pixels),
        visible_center=visible_center,
        position_flu=[float(v) for v in target],
        samples=len(samples),
        source=(
            "analytic_projection"
            if depth_fn is None or depth_reads == 0
            else "analytic_projection+rendered_depth"
        ),
    )


# ---------------------------------------------------------------------------
# the fixed observation set
# ---------------------------------------------------------------------------


def frame_to_png(frame: torch.Tensor, path: Path) -> Path:
    """Write a CHW float frame as the PNG the whole run will use (8-bit RGB).

    The PNG is the observation's pixels from here on: the tools, the audit and
    the next run read the same bytes, so "the model was sent a different image"
    is a hash mismatch rather than a suspicion.
    """
    array = (frame.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0 + 0.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(array.astype(np.uint8)).save(path)
    return path


def load_frame(path: Path) -> torch.Tensor:
    """CHW float frame of an observation PNG (the fixed set's pixels)."""
    from PIL import Image

    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).contiguous()


def frame_sha256(frame: Any) -> str:
    """Hash of the normalized pixels (the identity of one observation frame)."""
    if isinstance(frame, torch.Tensor):
        array = frame.detach().cpu().to(torch.float32).clamp(0.0, 1.0).numpy()
    else:
        array = np.asarray(frame, dtype=np.float32)
    quantized = np.rint(np.asarray(array, dtype=np.float32) * 255.0).astype(np.uint8)
    return hashlib.sha256(np.ascontiguousarray(quantized).tobytes()).hexdigest()[:16]


def frames_equal(frame_a: Any, frame_b: Any, tol: float = 0.05) -> bool:
    """Do two frames hold the same picture? (shape equal, pixels within ``tol``)"""
    if isinstance(frame_a, torch.Tensor) and isinstance(frame_b, torch.Tensor):
        if tuple(frame_a.shape) != tuple(frame_b.shape):
            return False
        delta = (frame_a.detach().cpu().float() - frame_b.detach().cpu().float()).abs().max()
        return bool(float(delta) <= float(tol))
    array_a = np.asarray(frame_a, dtype=np.float32)
    array_b = np.asarray(frame_b, dtype=np.float32)
    return bool(array_a.shape == array_b.shape and np.abs(array_a - array_b).max() <= tol)


def render_stats(frame: torch.Tensor, box: Sequence[float] | None) -> dict:
    """Render sanity numbers of one observation (is the frame a picture at all?).

    A black or flat frame is the silent failure the probe scripts caught after a
    PLY was read in the wrong frame; it belongs in the manifest so the report can
    refuse to draw conclusions from a blank image.
    """
    array = frame.detach().cpu().numpy()
    stats = {
        "mean": round(float(array.mean()), 4),
        "std": round(float(array.std()), 4),
        "nonblack_px": int((array.max(axis=0) > 1.0 / 255.0).sum()),
        "inside_box_mean": None,
    }
    if box is not None:
        height, width = int(array.shape[-2]), int(array.shape[-1])
        x0 = max(0, min(width - 1, int(round(box[0] / 1000.0 * width))))
        x1 = max(x0 + 1, min(width, int(round(box[2] / 1000.0 * width))))
        y0 = max(0, min(height - 1, int(round(box[1] / 1000.0 * height))))
        y1 = max(y0 + 1, min(height, int(round(box[3] / 1000.0 * height))))
        patch = array[:, y0:y1, x0:x1]
        if patch.size:
            stats["inside_box_mean"] = [round(float(v), 4) for v in patch.mean(axis=(1, 2))]
    return stats


@dataclass
class ObservationCase:
    """One frozen observation: the image, where it was taken from, the answer key."""

    case_id: str
    kind: str
    instruction: str
    image_path: str
    pose_flu: tuple[float, float, float]
    yaw_deg: float
    intrinsics: dict
    truth: CaseTruth
    note: str = ""
    render: dict = field(default_factory=dict)
    tag: str = ""

    @property
    def intr(self) -> Intrinsics:
        return intrinsics_of(self.intrinsics, width=self.intrinsics.get("width", 0))

    @property
    def target_box(self) -> list[float] | None:
        """The box a model can fairly be asked to find.

        Only what is visible: on an observation where the target is occluded or
        out of frame there is nothing to localize, and a fallback to the full
        projection would grade the answer against a box nobody can see.
        """
        return self.truth.visible_box if self.truth.visible else None

    @property
    def target_point(self) -> list[float] | None:
        """A pixel that is on the visible target by construction (centroid)."""
        if not self.truth.visible:
            return None
        if self.truth.visible_center is not None:
            return [float(v) for v in self.truth.visible_center]
        return None if self.target_box is None else list(_box_center(self.target_box))

    def load_frame(self) -> torch.Tensor:
        return load_frame(Path(self.image_path))

    def to_manifest(self, root: Path) -> dict:
        image = Path(self.image_path)
        try:
            image_ref = str(image.resolve().relative_to(Path(root).resolve()))
        except ValueError:
            image_ref = str(image)
        return {
            "case_id": self.case_id,
            "kind": self.kind,
            "tag": self.tag,
            "instruction": self.instruction,
            "image": image_ref,
            "pose_flu": [round(float(v), 4) for v in self.pose_flu],
            "yaw_deg": round(float(self.yaw_deg), 4),
            "intrinsics": dict(self.intrinsics),
            "truth": self.truth.to_dict(),
            "note": self.note,
            "render": dict(self.render),
        }

    @classmethod
    def from_manifest(cls, entry: Mapping, root: Path) -> "ObservationCase":
        image = Path(str(entry.get("image") or ""))
        if not image.is_absolute():
            image = Path(root) / image
        return cls(
            case_id=str(entry["case_id"]),
            kind=str(entry.get("kind") or ""),
            instruction=str(entry.get("instruction") or ""),
            image_path=str(image),
            pose_flu=_floats(entry.get("pose_flu"), 3) or (math.nan,) * 3,
            yaw_deg=float(entry.get("yaw_deg") or 0.0),
            intrinsics=dict(entry.get("intrinsics") or {}),
            truth=CaseTruth.from_dict(entry.get("truth") or {}),
            note=str(entry.get("note") or ""),
            render=dict(entry.get("render") or {}),
            tag=str(entry.get("tag") or ""),
        )


def save_observations(
    cases: Sequence[ObservationCase],
    manifest_path: Path,
    *,
    scene: Mapping | None = None,
    source: str = "",
    notes: Sequence[str] = (),
) -> Path:
    manifest_path = Path(manifest_path)
    manifest = {
        "format": MANIFEST_FORMAT,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": source,
        "scene": dict(scene or {}),
        "notes": list(notes),
        "kinds": sorted({case.kind for case in cases}),
        "missing_kinds": [kind for kind in CASE_KINDS if kind not in {c.kind for c in cases}],
        "cases": [case.to_manifest(manifest_path.parent) for case in cases],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def load_observations(manifest_path: Path) -> tuple[list[ObservationCase], dict]:
    """Read a fixed observation set back; (cases, manifest).

    The set is the unit of comparison between runs: the same frames, poses and
    answer keys, so a second run (another backend, another strategy) differs only
    by the model.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(
            f"{manifest_path} is not a diagnose_grounding observation set "
            f"(format={manifest.get('format')!r})"
        )
    cases = [ObservationCase.from_manifest(entry, manifest_path.parent)
             for entry in manifest.get("cases", [])]
    return cases, manifest


# ---------------------------------------------------------------------------
# scene geometry + case selection
# ---------------------------------------------------------------------------


@dataclass
class SceneSpec:
    """The geometry the fixed set is authored in (one frame, one target)."""

    ply_path: str
    target_pos: tuple[float, float, float]
    target_radius: float
    start_pos: tuple[float, float, float]
    start_yaw_deg: float
    hfov_deg: float
    occluders: list[Occluder] = field(default_factory=list)
    loss_pose: tuple[float, float, float] | None = None
    origin_flu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_name: str = "target"
    source: str = "synthetic"

    def to_dict(self) -> dict:
        return {
            "ply": self.ply_path,
            "source": self.source,
            "target_name": self.target_name,
            "target_pos": [round(float(v), 4) for v in self.target_pos],
            "target_radius": float(self.target_radius),
            "start_pos": [round(float(v), 4) for v in self.start_pos],
            "start_yaw_deg": float(self.start_yaw_deg),
            "hfov_deg": float(self.hfov_deg),
            "origin_flu": [round(float(v), 4) for v in self.origin_flu],
            "loss_pose": None if self.loss_pose is None else list(self.loss_pose),
            "occluders": [box.to_dict() for box in self.occluders],
        }


def scene_from_scenario_json(path: Path) -> SceneSpec:
    """SceneSpec of a runner's ``scenario.json`` (the M2 scene description)."""
    data = json.loads(Path(path).read_text())
    scenario = data.get("scenario", data)
    occluders = occluders_from(scenario.get("occluders"))
    start = _floats(scenario.get("start_pos"), 3) or (0.0, 0.0, 0.0)
    loss: tuple[float, float, float] | None = None
    if scenario.get("loss_bearing_deg") is not None and scenario.get("loss_distance_m"):
        bearing = math.radians(float(scenario["loss_bearing_deg"]))
        distance = float(scenario["loss_distance_m"])
        loss = (
            start[0] + distance * math.cos(bearing),
            start[1] + distance * math.sin(bearing),
            start[2],
        )
    return SceneSpec(
        ply_path=str(data.get("scene_ply") or ""),
        target_pos=_floats(scenario.get("target_pos"), 3) or (0.0, 0.0, 0.0),
        target_radius=float(scenario.get("target_radius") or 0.2),
        start_pos=start,
        start_yaw_deg=float(scenario.get("start_yaw_deg") or 0.0),
        hfov_deg=float(scenario.get("hfov_deg") or 60.0),
        occluders=occluders,
        loss_pose=loss,
        origin_flu=_floats(data.get("scene_origin_flu"), 3) or (0.0, 0.0, 0.0),
        target_name=str(scenario.get("target_name") or "target"),
        source=str(path),
    )


@dataclass(frozen=True)
class CasePose:
    """One selected viewpoint of the fixed set."""

    kind: str
    pose_flu: tuple[float, float, float]
    yaw_deg: float
    tag: str
    note: str = ""
    #: the analytic numbers the choice was made on (visible fraction, occluded
    #: fraction, target size in pixels) - kept so a caller can rank candidates
    #: without recomputing the geometry
    detail: dict = field(default_factory=dict)


def _truth_at(scene: SceneSpec, pose, yaw_deg: float, intrinsics: Intrinsics) -> CaseTruth:
    return target_ground_truth(
        scene.target_pos,
        scene.target_radius,
        pose,
        yaw_deg,
        intrinsics,
        scene.occluders,
    )


def _bearing_to_target(scene: SceneSpec, pose) -> float:
    """Heading that looks straight at the target from ``pose`` (degrees)."""
    values = _floats(pose, 3)
    if values is None:
        return 0.0
    delta = (scene.target_pos[0] - values[0], scene.target_pos[1] - values[1])
    return math.degrees(math.atan2(delta[1], delta[0]))


def case_candidates(scene: SceneSpec, intrinsics: Intrinsics) -> dict[str, list[CasePose]]:
    """Ranked viewpoint candidates for each situation (analytic geometry only).

    Every kind is *searched for* rather than assumed, because a case that does
    not actually show the situation would silently invalidate the row it
    produces:

    ``visible``  the start pose facing the target (falling back to the yaw that
                 sees the most of it)
    ``absent``   a pose/yaw where no sample of the target is visible (the M2 loss
                 pose first: behind the occluder and out of the field of view)
    ``partial``  viewpoints where the occluder cuts the target disc, ordered from
                 "almost all of it visible" towards "almost hidden" - the order a
                 rendered-depth refinement walks when it looks for the pose the
                 *picture* is half hidden in
    ``small``    a clear view from farther away, so the target is a few pixels
    """
    target = np.asarray(scene.target_pos, dtype=np.float64)
    start = np.asarray(scene.start_pos, dtype=np.float64)
    bearing = math.degrees(math.atan2(*(target - start)[:2][::-1]))
    out: dict[str, list[CasePose]] = {}

    visible = []
    for pose, yaw, tag in ((tuple(start), scene.start_yaw_deg, "start"),
                           (tuple(start), bearing, "bearing")):
        truth = _truth_at(scene, pose, yaw, intrinsics)
        if truth.visible:
            visible.append((truth.visible_fraction, CasePose(
                CASE_VISIBLE, pose, yaw, f"visible_{tag}",
                f"visible fraction {truth.visible_fraction:.2f} at {truth.distance_m:.2f} m",
                detail={"visible_fraction": truth.visible_fraction,
                        "occluded_fraction": truth.occluded_fraction},
            )))
    out[CASE_VISIBLE] = [pose for _score, pose in sorted(visible, key=lambda item: -item[0])]

    absent = []
    away = bearing + 180.0
    if scene.loss_pose is not None:
        absent.append((scene.loss_pose, scene.start_yaw_deg, "loss"))
        absent.append((scene.loss_pose, bearing, "loss_facing"))
    absent.append((tuple(start), away, "turn_away"))
    out[CASE_ABSENT] = [
        CasePose(CASE_ABSENT, tuple(pose), yaw, f"absent_{tag}",
                 f"visible fraction {truth.visible_fraction:.2f} "
                 f"(occluded {truth.occluded_fraction:.2f}, in_fov {truth.in_fov})",
                 detail={"visible_fraction": truth.visible_fraction,
                         "occluded_fraction": truth.occluded_fraction})
        for pose, yaw, tag in absent
        for truth in [_truth_at(scene, pose, yaw, intrinsics)]
        if not truth.visible
    ]

    base = np.asarray(scene.loss_pose if scene.loss_pose is not None else scene.start_pos,
                      dtype=np.float64)
    span = max(4.0, float(np.linalg.norm(target - base)) or 6.0)
    z_start = float(scene.start_pos[2])
    partial = []
    for dx in np.linspace(-span, span, 13):
        for dy in np.linspace(-span, span, 13):
            pose = (float(target[0] + dx), float(target[1] + dy), z_start)
            aim = _bearing_to_target(scene, pose)
            for dyaw in (-15.0, 0.0, 15.0):
                truth = _truth_at(scene, pose, aim + dyaw, intrinsics)
                # any occlusion at all: the "safe" end of the walk is a pose the
                # geometry calls almost fully visible, because a rendered
                # occluder hides more than its box and the half-hidden viewpoint
                # can sit just inside it
                if not truth.visible or truth.occluded_fraction <= 0.0:
                    continue
                partial.append((truth.visible_fraction, CasePose(
                    CASE_PARTIAL, pose, aim + dyaw, "partial_occluder",
                    f"visible fraction {truth.visible_fraction:.2f} "
                    f"(occluded {truth.occluded_fraction:.2f})",
                    detail={"visible_fraction": truth.visible_fraction,
                            "occluded_fraction": truth.occluded_fraction},
                )))
    # the walk order starts at "almost all of it visible" and moves towards
    # "almost hidden": an occluder that renders wider than its box shifts the
    # boundary that way, so a rendered refinement starts from the safe end
    partial.sort(key=lambda item: -item[0])
    out[CASE_PARTIAL] = [pose for _score, pose in partial]

    direction = start - target
    norm = float(np.linalg.norm(direction))
    small = []
    if norm > 1e-6:
        direction = direction / norm
        want_px = 0.08 * float(intrinsics.width)
        for distance in np.linspace(3.0, 14.0, 12):
            pose = target + direction * float(distance)
            yaw = _bearing_to_target(scene, pose)
            truth = _truth_at(scene, pose, yaw, intrinsics)
            if not truth.visible or truth.visible_fraction < 0.8 or truth.visible_box is None:
                continue
            box = truth.visible_box
            size_px = max(
                (box[2] - box[0]) / 1000.0 * intrinsics.width,
                (box[3] - box[1]) / 1000.0 * intrinsics.height,
            )
            small.append((abs(size_px - want_px), CasePose(
                CASE_SMALL, tuple(float(v) for v in pose), yaw, "small_far",
                f"target {size_px:.1f} px across at {truth.distance_m:.2f} m",
                detail={"visible_fraction": truth.visible_fraction,
                        "size_px": size_px},
            )))
    out[CASE_SMALL] = [pose for _score, pose in sorted(small, key=lambda item: item[0])]
    return out


def situation_score(kind: str, truth: CaseTruth, intrinsics: Intrinsics) -> float | None:
    """How well a rendered answer key *is* the situation (None = it is not).

    The score is what the refinement minimises, and the ``None`` cases are the
    ones that must never be accepted: a "partially hidden" observation whose
    target is fully hidden (or fully visible) is not that situation.
    """
    if kind == CASE_VISIBLE:
        if not truth.visible or truth.visible_fraction < 0.9:
            return None
        return 1.0 - truth.visible_fraction
    if kind == CASE_ABSENT:
        return None if truth.visible else 0.0
    if kind == CASE_PARTIAL:
        if not truth.visible or not (0.15 <= truth.visible_fraction <= 0.65):
            return None
        if truth.occluded_fraction < 0.15:
            return None
        return abs(truth.visible_fraction - 0.5)
    if kind == CASE_SMALL:
        if not truth.visible or truth.visible_fraction < 0.8 or truth.visible_box is None:
            return None
        box = truth.visible_box
        size_px = max(
            (box[2] - box[0]) / 1000.0 * intrinsics.width,
            (box[3] - box[1]) / 1000.0 * intrinsics.height,
        )
        return abs(size_px - 0.08 * float(intrinsics.width))
    return None


def _best_analytic_candidate(kind: str, options: Sequence[CasePose]) -> CasePose:
    """The candidate an analytic run (no rendered oracle) would pick.

    Every kind's list is already ranked best-first except the partly hidden one,
    whose walk order starts at "almost all of it visible" (the order a rendered
    refinement needs): the analytic pick wants the viewpoint closest to half
    visible instead.
    """
    if kind == CASE_PARTIAL:
        return min(options, key=lambda pose: abs(
            float(pose.detail.get("visible_fraction") or 0.0) - 0.5
        ))
    return options[0]


def choose_case_poses(
    scene: SceneSpec,
    intrinsics: Intrinsics,
    *,
    truth_of: Callable[[tuple[float, float, float], float], CaseTruth] | None = None,
    max_probes_per_kind: int = 10,
) -> tuple[list[CasePose], list[str]]:
    """One viewpoint per situation (``(poses, notes)``).

    ``truth_of`` is the answer-key oracle of a *rendered* observation (the depth
    map of the pose). It matters because an occluder's splats cover more than the
    geometric box they stand for: at the analytic boundary of a partly hidden
    target the rendered picture can already be fully hidden, and a fixed set
    graded against the analytic box would then ask the model for something it
    cannot see. With an oracle the candidate list is walked (bounded) until the
    candidate really shows the situation.
    """
    candidates = case_candidates(scene, intrinsics)
    poses: list[CasePose] = []
    notes: list[str] = []
    visible_anchor: tuple[float, float, float] | None = None
    for kind in CASE_KINDS:
        options = candidates.get(kind) or []
        if not options:
            notes.append(f"no viewpoint found for the '{kind}' situation in this scene")
            continue
        if truth_of is None:
            poses.append(_best_analytic_candidate(kind, options))
            continue
        if kind == CASE_PARTIAL:
            refined = _refine_partial_pose(
                scene, intrinsics, truth_of, options,
                visible_anchor=visible_anchor,
                max_probes=int(max_probes_per_kind),
            )
            if refined is not None:
                poses.append(refined)
                continue
        best = None
        probes = 0
        for option in options[: max(1, int(max_probes_per_kind))]:
            truth = truth_of(option.pose_flu, option.yaw_deg)
            probes += 1
            score = situation_score(kind, truth, intrinsics)
            if score is None:
                continue
            if best is None or score < best[0]:
                best = (score, option, truth)
        if best is None:
            notes.append(
                f"the rendered observation never shows the '{kind}' situation "
                f"({probes} candidate viewpoints probed); the analytic choice is kept"
            )
            poses.append(_best_analytic_candidate(kind, options))
            continue
        _score, option, truth = best
        note = (
            f"{option.note} | rendered: visible fraction {truth.visible_fraction:.2f}, "
            f"occluded {truth.occluded_fraction:.2f} ({probes} probed)"
        )
        poses.append(CasePose(option.kind, option.pose_flu, option.yaw_deg, option.tag, note))
        if kind == CASE_VISIBLE and truth.visible_fraction > 0.65:
            visible_anchor = option.pose_flu
    return poses, notes


def _lerp_pose(a, b, s: float) -> tuple[float, float, float]:
    return tuple(float(a[i] + (b[i] - a[i]) * float(s)) for i in range(3))


def _refine_partial_pose(
    scene: SceneSpec,
    intrinsics: Intrinsics,
    truth_of: Callable[[tuple[float, float, float], float], CaseTruth],
    options: Sequence[CasePose],
    *,
    visible_anchor: tuple[float, float, float] | None = None,
    max_probes: int = 10,
) -> CasePose | None:
    """Find a viewpoint the *picture* shows half hidden.

    An occluder's splats cover more than the box they stand for, so the picture
    is already fully hidden at the geometric boundary of a partly hidden target
    and the half-hidden zone can sit between two grid candidates. The walk starts
    at the candidates the geometry calls "almost entirely visible" and probes
    until it holds a viewpoint that sees the target and one that does not
    (the pose of the clearly visible case, when it is known, is that first end by
    construction); between the two the rendered fraction is bisected. Every probe
    uses the same oracle as the answer key, so what is selected is the situation
    in the image the model is shown.
    """
    best: tuple[float, CasePose, CaseTruth] | None = None
    visible_end: tuple[tuple[float, float, float], float] | None = (
        None if visible_anchor is None
        else (visible_anchor, _bearing_to_target(scene, visible_anchor))
    )
    hidden_end: tuple[tuple[float, float, float], float] | None = None
    probes = 0
    for option in options[: max(2, int(max_probes))]:
        truth = truth_of(option.pose_flu, option.yaw_deg)
        probes += 1
        score = situation_score(CASE_PARTIAL, truth, intrinsics)
        if score is not None and (best is None or score < best[0]):
            best = (score, _partial_pose(option, truth, probes), truth)
            if 0.4 <= truth.visible_fraction <= 0.6:
                return best[1]
        if truth.visible_fraction > 0.65 and visible_end is None:
            visible_end = (option.pose_flu, option.yaw_deg)
        if truth.visible_fraction < 0.15 and hidden_end is None:
            hidden_end = (option.pose_flu, option.yaw_deg)
        if visible_end is not None and hidden_end is not None:
            break
    if best is not None:
        return best[1]
    if visible_end is None or hidden_end is None:
        return None
    # s = 0 sees the target, s = 1 does not: the fraction falls as s grows
    lo, hi = 0.0, 1.0
    for _ in range(max(2, int(max_probes))):
        s = 0.5 * (lo + hi)
        pose = _lerp_pose(visible_end[0], hidden_end[0], s)
        yaw = _bearing_to_target(scene, pose)
        truth = truth_of(pose, yaw)
        probes += 1
        score = situation_score(CASE_PARTIAL, truth, intrinsics)
        if score is not None:
            candidate = CasePose(
                CASE_PARTIAL, pose, yaw, "partial_bisect",
                f"visible fraction {truth.visible_fraction:.2f} "
                f"(occluded {truth.occluded_fraction:.2f}) | rendered: "
                f"visible fraction {truth.visible_fraction:.2f}, "
                f"occluded {truth.occluded_fraction:.2f} ({probes} probed)",
            )
            if best is None or score < best[0]:
                best = (score, candidate, truth)
            if 0.4 <= truth.visible_fraction <= 0.6:
                return candidate
        if truth.visible_fraction > 0.65:
            lo = s
        elif truth.visible_fraction < 0.15:
            hi = s
        else:
            hi = s  # visible, but not the half-hidden situation yet
    return None if best is None else best[1]


def _partial_pose(option: CasePose, truth: CaseTruth, probes: int) -> CasePose:
    return CasePose(
        CASE_PARTIAL, option.pose_flu, option.yaw_deg, option.tag,
        f"{option.note} | rendered: visible fraction {truth.visible_fraction:.2f}, "
        f"occluded {truth.occluded_fraction:.2f} ({probes} probed)",
    )


# ---------------------------------------------------------------------------
# building the fixed set from a scene (the only path that renders)
# ---------------------------------------------------------------------------


def load_m2_helpers(path: Path | None = None):
    """``scripts/run_vlm_nav_m2_memory.py`` as a module (the scenario scene builder).

    Read-only use: the synthetic scene (floor + occluder slab + target cluster),
    the scenario geometry and the CPU splat world are exactly the ones the real
    M2 episodes render, so the fixed set is a batch of *real* observations of
    that scenario instead of a fixture nobody ever flew in.
    """
    path = Path(path or Path(__file__).resolve().with_name("run_vlm_nav_m2_memory.py"))
    name = "diffaero_script_run_vlm_nav_m2_memory"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def default_world_factory(scene: SceneSpec, render_size: tuple[int, int]):
    """CPU splat world at the scene's PLY (the renderer the diagnostics assume)."""
    from vlm_nav.world import SplatWorldCPU, TargetSpec

    return SplatWorldCPU(
        ply_path=Path(scene.ply_path),
        target=TargetSpec(
            name=scene.target_name,
            position_flu=tuple(scene.target_pos),
            radius_m=float(scene.target_radius),
        ),
        start_flu=tuple(scene.start_pos),
        start_yaw_deg=float(scene.start_yaw_deg),
        width=int(render_size[0]),
        height=int(render_size[1]),
        hfov_deg=float(scene.hfov_deg),
    )


def set_world_pose(world: Any, pose_flu: Iterable[float], yaw_deg: float) -> None:
    """Move a world to a viewpoint (the pose its frames and depth maps describe).

    Both the state object and the sensor write-back are used, mirroring
    ``world.scan_views``: an observation is only projectable with depth while the
    world still stands where the frame was rendered.
    """
    pose = np.asarray(_floats(pose_flu, 3) or (0.0, 0.0, 0.0), dtype=np.float64)
    state = getattr(world, "state", None)
    if state is not None and hasattr(state, "position_flu"):
        state.position_flu = pose
    elif hasattr(world, "position_flu"):
        try:
            world.position_flu = pose
        except Exception:  # noqa: BLE001 - a read-only world keeps its pose
            pass
    yaw_rad = math.radians(float(yaw_deg))
    if state is not None and hasattr(state, "yaw_rad"):
        state.yaw_rad = yaw_rad
    write_pose = getattr(world, "_write_pose", None)
    if callable(write_pose):
        try:
            write_pose()
        except Exception:  # noqa: BLE001 - sensor write-back is best effort
            pass


def build_cases_from_scene(
    scene: SceneSpec,
    out_dir: Path,
    *,
    render_size: tuple[int, int] = (320, 180),
    instruction: str = "",
    world_factory: Callable[[SceneSpec, tuple[int, int]], Any] | None = None,
    cases_dir: Path | None = None,
    max_probes_per_kind: int = 10,
) -> tuple[list[ObservationCase], list[str]]:
    """Render the four situations into ``out_dir``; (cases, notes).

    Rendering happens once, here: the resulting PNGs are the fixed set every
    strategy reads, so the comparison between strategies is not polluted by
    different frames (the failure mode this bench exists to remove). The
    situations are picked against the *rendered* observation when the world can
    answer a depth probe - an occluder's splats hide more than its geometric box,
    and a "partly hidden" case chosen on the box alone would be a fully hidden
    target in the picture the model is shown.
    """
    factory = world_factory or default_world_factory
    width, height = int(render_size[0]), int(render_size[1])
    intrinsics = Intrinsics(width, height, float(scene.hfov_deg))
    notes: list[str] = []
    world = factory(scene, (width, height))
    depth_fn = getattr(world, "depth_at_pixel", None)
    depth_fn = depth_fn if callable(depth_fn) else None

    def truth_of(pose, yaw_deg: float) -> CaseTruth:
        set_world_pose(world, pose, yaw_deg)
        return target_ground_truth(
            scene.target_pos, scene.target_radius, pose, yaw_deg, intrinsics,
            scene.occluders, depth_fn=depth_fn,
        )

    poses, selection_notes = choose_case_poses(
        scene, intrinsics,
        truth_of=truth_of if depth_fn is not None else None,
        max_probes_per_kind=max_probes_per_kind,
    )
    notes.extend(selection_notes)
    found = {pose.kind for pose in poses}
    for kind in CASE_KINDS:
        if kind not in found:
            notes.append(f"no viewpoint found for the '{kind}' situation in this scene")
    cases_dir = Path(cases_dir or (Path(out_dir) / "observations"))
    cases: list[ObservationCase] = []
    for index, pose in enumerate(poses):
        set_world_pose(world, pose.pose_flu, pose.yaw_deg)
        frame = world.render()
        frame = frame.detach().to(torch.float32).cpu()
        truth = target_ground_truth(
            scene.target_pos,
            scene.target_radius,
            pose.pose_flu,
            pose.yaw_deg,
            intrinsics,
            scene.occluders,
            depth_fn=depth_fn,
        )
        if not truth.visible and pose.kind != CASE_ABSENT:
            notes.append(
                f"case {pose.kind}: the target is not visible in the rendered observation "
                "(the model is asked about an image that does not contain it)"
            )
        case_id = f"{pose.kind}_{index}"
        image_path = frame_to_png(frame, cases_dir / f"{case_id}.png")
        stats = render_stats(frame, truth.visible_box or truth.box)
        if depth_fn is not None and truth.visible:
            point = _box_center_of(truth.visible_center or truth.visible_box)
            try:
                probe = float(depth_fn((float(point[0]), float(point[1]))))
            except Exception:  # noqa: BLE001 - a depth probe is evidence, not a gate
                probe = None
            stats["depth_at_target_m"] = None if probe is None else round(probe, 3)
            stats["depth_matches_target"] = (
                None if probe is None
                else bool(abs(probe - truth.distance_m) <= max(0.35, 0.2 * truth.distance_m))
            )
        cases.append(
            ObservationCase(
                case_id=case_id,
                kind=pose.kind,
                instruction=instruction or f"the {scene.target_name}",
                image_path=str(image_path),
                pose_flu=tuple(float(v) for v in pose.pose_flu),
                yaw_deg=float(pose.yaw_deg),
                intrinsics=intrinsics.to_dict(),
                truth=truth,
                note=pose.note,
                render=stats,
                tag=pose.tag,
            )
        )
    return cases, notes


def build_cases_from_probe(
    probe_json: Path,
    scene_json: Path,
    *,
    frames_dir: Path | None = None,
    instruction: str = "",
) -> tuple[list[ObservationCase], SceneSpec, list[str]]:
    """Fixed set from an existing probe bundle (frames + poses, no rendering).

    ``scripts/run_vlm_nav_m2_memory.py``'s probe artifacts (``probe/post_fix.json``
    plus ``probe/frames_post_fix/``) are already a frozen batch: a frame per
    pose, the pose it was rendered from and the analytic visibility. Reading them
    back gives the bench a real fixed set without paying for a render, and the
    answer key is rebuilt from the scenario geometry of the same scene.
    """
    probe = json.loads(Path(probe_json).read_text())
    scene = scene_from_scenario_json(scene_json)
    render = probe.get("render") or {}
    width = int(render.get("width") or 0)
    height = int(render.get("height") or 0)
    hfov = float(render.get("hfov_deg") or scene.hfov_deg)
    frames_dir = Path(frames_dir or (Path(probe_json).parent / f"frames_{probe.get('label', '')}"))
    notes: list[str] = []
    cases: list[ObservationCase] = []
    for index, entry in enumerate(probe.get("poses") or []):
        tag = str(entry.get("tag") or f"pose{index}")
        pose = _floats(entry.get("pose"), 3)
        if pose is None:
            notes.append(f"probe pose '{tag}' has no position; skipped")
            continue
        yaw_deg = float(entry.get("yaw_deg") or 0.0)
        image = frames_dir / f"{tag}.png"
        if not image.is_file():
            notes.append(f"probe frame '{image}' is missing; skipped")
            continue
        frame = load_frame(image)
        frame_height, frame_width = int(frame.shape[-2]), int(frame.shape[-1])
        intrinsics = Intrinsics(frame_width, frame_height, hfov)
        if (frame_width, frame_height) != (width, height) and width and height:
            notes.append(
                f"probe frame '{tag}' is {frame_width}x{frame_height} while the probe "
                f"rendered {width}x{height}; the frame's own size is used"
            )
        truth = target_ground_truth(
            scene.target_pos, scene.target_radius, pose, yaw_deg, intrinsics, scene.occluders
        )
        kind = CASE_VISIBLE if "start" in tag else CASE_ABSENT
        cases.append(
            ObservationCase(
                case_id=f"{kind}_{index}",
                kind=kind,
                instruction=instruction or f"the {scene.target_name}",
                image_path=str(image),
                pose_flu=pose,
                yaw_deg=yaw_deg,
                intrinsics=intrinsics.to_dict(),
                truth=truth,
                note=f"probe {tag}",
                render=render_stats(frame, truth.visible_box or truth.box),
                tag=tag,
            )
        )
    present = {case.kind for case in cases}
    for kind in CASE_KINDS:
        if kind not in present:
            notes.append(f"the probe bundle has no '{kind}' frame")
    return cases, scene, notes


# ---------------------------------------------------------------------------
# model call plumbing: prompts, the recording wrapper, mock/oracle backends
# ---------------------------------------------------------------------------

POINT_SYSTEM_PROMPT = (
    "You are the target-localization module of a drone navigation agent. "
    "Reply with ONE JSON object and nothing else: "
    '{"point": [u, v], "visible": true, "confidence": 0.0, "distance_m": 0.0}. '
    "[u, v] is the centre of the instructed target as a normalized [0, 1000] "
    "image point (u grows to the right, v grows downwards). "
    'When the target is not in the image reply {"visible": false, "point": null} '
    "and nothing else. distance_m is your rough distance estimate in meters "
    "(0 when you cannot tell)."
)

REGION_SYSTEM_PROMPT = (
    "You are the target-localization module of a drone navigation agent. "
    "Reply with ONE JSON object and nothing else: "
    '{"region": [x1, y1, x2, y2], "visible": true, "confidence": 0.0}. '
    "region is the corner box [x1, y1, x2, y2] normalized 0-1000 around the "
    "instructed target: x1, y1 = top-left, x2, y2 = bottom-right. Make it a "
    "little wider than the target. "
    'When the target is not in the image reply {"visible": false, "region": null}.'
)


def point_user_prompt(instruction: str) -> str:
    return (
        f"Instruction: {instruction}\n"
        "Give the normalized [0, 1000] image point of the centre of that target."
    )


def region_user_prompt(instruction: str, *, inside_crop: bool = False) -> str:
    where = "this magnified crop" if inside_crop else "this image"
    return (
        f"Instruction: {instruction}\n"
        f"Give a normalized 0-1000 corner box around that target in {where}."
    )


def first_json_object(text: Any) -> Any:
    """First decodable JSON object/array in a model reply (prose tolerated).

    Local on purpose: the bench must be able to report "the reply had no JSON"
    about a backend it does not otherwise trust, so the parser it judges with
    cannot be a private helper of the module under test.
    """
    if not isinstance(text, str) or not text:
        return None
    for opener, closer in (("{", "}"), ("[", "]")):
        for index, char in enumerate(text):
            if char != opener:
                continue
            depth = 0
            for position in range(index, len(text)):
                if text[position] == opener:
                    depth += 1
                elif text[position] == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[index:position + 1])
                        except json.JSONDecodeError:
                            break
    return None


def parse_point_reply(raw: Any) -> tuple[tuple[float, float] | None, bool, float | None, str]:
    """``(point, visible, distance_m, reason)`` of a point-prompt reply.

    ``reason`` is empty when the reply was understood; otherwise it names what
    was wrong with it (no JSON / no point / out of range), which is the
    difference between "the model did not see it" and "the model wrote prose".
    """
    payload = first_json_object(raw if isinstance(raw, str) else str(raw or ""))
    if not isinstance(payload, dict):
        return None, False, None, "no_json_object"
    visible = bool(payload.get("visible", True))
    point = payload.get("point", payload.get("pixel"))
    distance = payload.get("distance_m", payload.get("distance"))
    try:
        distance = None if distance in (None, "") else float(distance)
    except (TypeError, ValueError):
        distance = None
    if not visible and point in (None, []):
        return None, False, distance, ""
    numbers = _floats(point, 2)
    if numbers is None:
        return None, visible, distance, "no_point"
    if not all(0.0 <= value <= 1000.0 for value in numbers):
        return None, visible, distance, "point_out_of_frame"
    return numbers, visible, distance, ""


def parse_region_reply(raw: Any) -> tuple[list[float] | None, bool, str]:
    """``(box, visible, reason)`` of a region-prompt reply (canonical xyxy)."""
    payload = first_json_object(raw if isinstance(raw, str) else str(raw or ""))
    if not isinstance(payload, dict):
        return None, False, "no_json_object"
    visible = bool(payload.get("visible", True))
    value = payload.get("region", payload.get("box", payload.get("bbox")))
    if not visible and value in (None, []):
        return None, False, ""
    numbers = _floats(value, 4)
    if numbers is None:
        return None, visible, "no_region"
    x1, y1, x2, y2 = numbers
    if x2 <= x1 or y2 <= y1:
        return None, visible, "region_not_corners"
    if not all(0.0 <= v <= 1000.0 for v in numbers):
        return None, visible, "region_out_of_frame"
    return [x1, y1, x2, y2], visible, ""


class CallRecorder:
    """The audit trail: one record per model call, with the image that was sent.

    Nothing about a call is reconstructed later - the frames handed to the
    backend are cloned and hashed as they go out, so the record can prove which
    pixels the model saw. ``inject_wrong_image`` deliberately replaces them (a
    fault injection to check that the audit catches exactly this bug); every
    record it touches is marked ``diagnosis_only``.
    """

    def __init__(
        self,
        out_dir: Path,
        *,
        inject_wrong_image: bool = False,
        inputs_dir: str = "inputs",
        call_log: str = "calls.jsonl",
    ):
        self.out_dir = Path(out_dir)
        self.inputs_dir = self.out_dir / inputs_dir
        self.call_log = self.out_dir / call_log
        self.records: list[dict] = []
        self.inject_wrong_image = bool(inject_wrong_image)
        self._wrong_frame: Any = None
        self._active: dict | None = None
        self._case: ObservationCase | None = None

    # -- step scope -------------------------------------------------------

    def set_case(self, case: ObservationCase) -> None:
        self._case = case

    def set_wrong_frame(self, frame: Any) -> None:
        """The frame the fault injection sends instead of the observation's."""
        self._wrong_frame = self._clone(frame)

    def begin_step(
        self,
        *,
        strategy: str,
        step: str,
        expected_frames: Sequence[Any] = (),
        expected_source: str = "case_frame",
        diagnosis_only: bool = False,
        obs_id: str | None = None,
        crop: Mapping | None = None,
        extra: Mapping | None = None,
    ) -> None:
        self._active = {
            "strategy": strategy,
            "step": step,
            "expected_frames": [self._clone(frame) for frame in expected_frames],
            "expected_source": expected_source,
            "diagnosis_only": bool(diagnosis_only),
            "obs_id": obs_id,
            "crop": dict(crop or {}),
            "extra": dict(extra or {}),
        }

    def end_step(self) -> None:
        self._active = None

    def diagnosis_context(self) -> dict:
        """What a mock/oracle backend may read (never sent to a real model)."""
        active = self._active or {}
        case = self._case
        truth = case.truth.to_dict() if case is not None else {}
        return {
            "case_id": case.case_id if case is not None else "",
            "kind": case.kind if case is not None else "",
            "strategy": active.get("strategy", ""),
            "step": active.get("step", ""),
            "reply_key": f"{case.case_id if case else ''}|{active.get('strategy', '')}|"
                         f"{active.get('step', '')}",
            "frame_size": [case.intrinsics.get("width"), case.intrinsics.get("height")]
                          if case is not None else None,
            "truth": truth,
            "crop": dict(active.get("crop") or {}),
        }

    # -- call scope -------------------------------------------------------

    def prepare(self, frames: Any) -> tuple[list[Any], dict]:
        """The frames a call really sends + the record being filled in.

        Also the "wrong image" fault injection: with ``inject_wrong_image`` the
        frames are replaced by another observation's, which the audit then has to
        catch (the bug class no success rate can see).
        """
        active = self._active or {}
        sent = [self._clone(frame) for frame in _as_frame_list(frames)]
        if self.inject_wrong_image and self._wrong_frame is not None:
            sent = [self._clone(self._wrong_frame) for _ in sent] or [self._clone(self._wrong_frame)]
        expected = list(active.get("expected_frames") or [])
        matches = bool(expected) and len(sent) == len(expected) and all(
            frames_equal(want, got) for want, got in zip(expected, sent)
        )
        record: dict = {
            "call": len(self.records) + 1,
            "case": self._case.case_id if self._case is not None else "",
            "kind": self._case.kind if self._case is not None else "",
            "strategy": active.get("strategy", ""),
            "step": active.get("step", ""),
            "diagnosis_only": bool(active.get("diagnosis_only")) or self.inject_wrong_image,
            "obs_id": active.get("obs_id"),
            "input_image": {
                "path": None,
                "sha256": None,
                "shape": [int(v) for v in getattr(sent[0], "shape", ())] if sent else [],
                "expected_sha256": [frame_sha256(frame) for frame in expected],
                "expected_source": active.get("expected_source", ""),
                "matches_expected": matches,
                "injected": bool(self.inject_wrong_image and self._wrong_frame is not None),
                "injected_reason": "wrong-frame fault injection" if self.inject_wrong_image else None,
            },
            "raw_reply": None,
            "backend": None,
            "ms": None,
            "error": None,
            "tool": None,
            "parsed": None,
            "measure": None,
            "failure": {"category": FAILURE_NONE, "detail": ""},
            "outcome": None,
            #: intermediate step of a multi-step strategy (the crop of
            #: ``region_then_local``): it is audited but never scored on its own
            "final": True,
        }
        record["_sent"] = sent
        if active.get("extra"):
            record.update({key: value for key, value in active["extra"].items() if key not in record})
        return sent, record

    def finish(self, record: dict, raw: Any, *, backend: str = "", ms: float | None = None) -> None:
        """Close a call: dump the image that was sent, then log the record."""
        sent = record.pop("_sent", [])
        record["raw_reply"] = raw if raw is None or isinstance(raw, str) else str(raw)
        record["backend"] = backend
        record["ms"] = None if ms is None else round(float(ms), 1)
        if sent:
            image_path = self.inputs_dir / f"{record['call']:03d}_{record['case']}_{record['step']}.png"
            frame_to_png(sent[0] if len(sent) == 1 else torch.cat(list(sent), dim=-1), image_path)
            record["input_image"]["path"] = str(image_path)
            record["input_image"]["sha256"] = frame_sha256(sent[0])
        self.records.append(record)
        self.flush()

    def flush(self) -> None:
        self.call_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.call_log, "w") as handle:
            for record in self.records:
                handle.write(json.dumps(record, default=str) + "\n")

    def last(self) -> dict | None:
        return self.records[-1] if self.records else None

    @staticmethod
    def _clone(frame: Any) -> torch.Tensor:
        if isinstance(frame, torch.Tensor):
            return frame.detach().to(torch.float32).cpu().clone()
        return torch.as_tensor(np.asarray(frame, dtype=np.float32)).clone()


def _as_frame_list(frames: Any) -> list[Any]:
    if frames is None:
        return []
    if isinstance(frames, torch.Tensor):
        return [frames]
    if isinstance(frames, np.ndarray):
        return [frames]
    return list(frames)


class RecordingBackend(VLMBackend):
    """The backend the runner calls: records what went out, then delegates.

    Also the one place the diagnosis context is attached, so a scripted or
    oracle mock can answer for the *current* case without the tool code (which
    builds its own context dict) knowing anything about the bench.
    """

    def __init__(self, backend: Any, recorder: CallRecorder):
        self.backend = backend
        self.recorder = recorder
        self.name = f"recording:{getattr(backend, 'name', type(backend).__name__)}"
        self.on_log = getattr(backend, "on_log", None)

    def decide(self, frames, system_prompt, user_prompt, context=None):
        sent, record = self.recorder.prepare(frames)
        record["system_prompt"] = system_prompt
        record["user_prompt"] = user_prompt
        started = time.time()
        injected = dict(context) if isinstance(context, Mapping) else {}
        injected["diagnosis"] = self.recorder.diagnosis_context()
        try:
            raw = self.backend.decide(sent, system_prompt, user_prompt, injected)
        except Exception as exc:  # noqa: BLE001 - a failing backend is a data point
            record["error"] = f"{type(exc).__name__}: {exc}"
            self.recorder.finish(record, None, backend=self.name, ms=(time.time() - started) * 1000.0)
            raise
        self.recorder.finish(record, raw, backend=self.name, ms=(time.time() - started) * 1000.0)
        return raw

    def decide_with_tools(self, frames, system_prompt, user_prompt, tools_text="", context=None):
        return self.backend.decide_with_tools(
            frames, system_prompt, user_prompt, tools_text, context
        )

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()


class KeyedMockVLM(VLMBackend):
    """Replies looked up by ``"<case>|<strategy>|<step>"`` (deterministic runs).

    An ordered script cannot express "the third call of the second strategy"; a
    keyed one can, and a missing key is reported as an empty reply instead of
    silently reusing another case's answer.
    """

    name = "mock"

    def __init__(self, replies: Mapping[str, Any], default: str = ""):
        self.replies = {str(k): v for k, v in dict(replies or {}).items()}
        self.default = default
        self.missing: list[str] = []

    def decide(self, frames, system_prompt, user_prompt, context=None):
        diagnosis = (context or {}).get("diagnosis") or {}
        key = str(diagnosis.get("reply_key") or "")
        if key not in self.replies:
            self.missing.append(key)
            return self.default
        reply = self.replies[key]
        return reply if isinstance(reply, str) else json.dumps(reply)


class OracleReplyVLM(VLMBackend):
    """Answers from the fixed set's own answer key (harness self-test only).

    This is not a model measurement: it proves the bench, the tools and the
    projection chain agree when the localization is perfect. Every run it drives
    is flagged as an oracle self-test in the report and in ``run_config.json``.
    """

    name = "mock_oracle"

    def decide(self, frames, system_prompt, user_prompt, context=None):
        diagnosis = (context or {}).get("diagnosis") or {}
        truth = diagnosis.get("truth") or {}
        step = str(diagnosis.get("step") or "")
        case_id = str(diagnosis.get("case_id") or "")
        box = _box_or_none(truth.get("visible_box"))
        crop = diagnosis.get("crop") or {}
        if step == "region":
            if box is None:
                return json.dumps({"visible": False, "region": None})
            pad = 40.0
            return json.dumps({
                "region": [
                    max(0.0, box[0] - pad), max(0.0, box[1] - pad),
                    min(1000.0, box[2] + pad), min(1000.0, box[3] + pad),
                ],
                "visible": True,
                "confidence": 0.9,
                "case_id": case_id,
            })
        if step.endswith("local") or crop:
            local = _box_or_none(crop.get("truth_box"))
            if local is None:
                return json.dumps({"candidates": [], "visible": False})
            return json.dumps({
                "candidates": [{
                    "box": local, "box_format": "xyxy", "confidence": 0.9,
                    "distance_m": truth.get("distance_m") or 0.0,
                    "attrs": {"source": "oracle"},
                }],
                "visible": True,
            })
        if step == "point":
            if box is None:
                return json.dumps({"visible": False, "point": None})
            centre = _floats(truth.get("visible_center"), 2)
            point = list(centre) if centre is not None else [
                (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            ]
            return json.dumps({
                "point": point, "visible": True, "confidence": 0.9,
                "distance_m": truth.get("distance_m") or 0.0,
            })
        if box is None:
            return json.dumps({"candidates": [], "visible": False})
        return json.dumps({
            "candidates": [{
                "box": box, "box_format": "xyxy", "confidence": 0.9,
                "distance_m": truth.get("distance_m") or 0.0,
                "attrs": {"source": "oracle"},
            }],
            "visible": True,
        })


# ---------------------------------------------------------------------------
# the detector/segmentation slot
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    """One detector output: a canonical box, optional mask and label."""

    box: list[float]
    confidence: float | None = None
    label: str = ""
    mask: Any = None

    def to_dict(self) -> dict:
        return {
            "box": _round_box(self.box),
            "confidence": None if self.confidence is None else round(float(self.confidence), 3),
            "label": str(self.label),
            "has_mask": self.mask is not None,
        }


class DetectorBackend:
    """Slot for a detection/segmentation backend (SAM3, DINOv2, ...).

    The bench calls ``available()`` first and, when it is false, reports the
    strategy as *not run* instead of inventing a number for it. ``detect`` gets
    the observation's pixels and the instruction and returns canonical
    ``[x1, y1, x2, y2]`` boxes (0-1000) in the frame's own coordinates; a mask of
    any shape may ride along untouched, so a segmentation backend of the second
    phase plugs in here without touching the bench.
    """

    name = "base"
    note = ""

    def available(self) -> bool:
        return False

    def detect(self, frame: Any, instruction: str, *, case: Any = None) -> list[Detection]:
        raise NotImplementedError


class NullDetector(DetectorBackend):
    """The empty slot (the default): no backend is wired, and it says so."""

    name = "null"
    note = "no detection/segmentation backend configured (slot reserved)"

    def available(self) -> bool:
        return False


class OracleDetector(DetectorBackend):
    """Returns the answer key's box (diagnosis-only self-test of the slot)."""

    name = "oracle"
    note = "oracle detector: returns the case's own visible box (diagnosis only)"

    def __init__(self, case: Any = None):
        self.case = case

    def available(self) -> bool:
        return True

    def detect(self, frame: Any, instruction: str, *, case: Any = None) -> list[Detection]:
        case = case or self.case
        box = case.target_box if case is not None else None
        return [] if box is None else [Detection(list(box), 0.9, "oracle", {"source": "ground_truth"})]


def resolve_detector(spec: str, *, case: Any = None) -> DetectorBackend:
    """``"--detector null|oracle|module:factory"`` -> a detector backend.

    The dotted form is the extension point: any importable module that exposes a
    factory returning a :class:`DetectorBackend` can be wired from the command
    line, which is what makes this a slot rather than a stub.
    """
    spec = str(spec or "null").strip()
    if spec in ("", "null", "none"):
        return NullDetector()
    if spec == "oracle":
        return OracleDetector(case)
    module_name, _, attribute = spec.partition(":")
    if not attribute:
        raise ValueError(f"--detector {spec!r} must be 'null', 'oracle' or 'module:factory'")
    module = __import__(module_name, fromlist=[attribute])
    factory = getattr(module, attribute)
    detector = factory() if callable(factory) else factory
    if not hasattr(detector, "detect"):
        raise TypeError(f"{spec!r} does not return a DetectorBackend")
    return detector


# ---------------------------------------------------------------------------
# the executor (what happens after a waypoint exists)
# ---------------------------------------------------------------------------


@dataclass
class ExecutorResult:
    """One waypoint actually flown by the world (the post-grounding half)."""

    target_flu: list[float]
    start_flu: list[float]
    final_flu: list[float]
    error_m: float
    legs: int
    status: str
    feedback: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "target_flu": [round(float(v), 3) for v in self.target_flu],
            "start_flu": [round(float(v), 3) for v in self.start_flu],
            "final_flu": [round(float(v), 3) for v in self.final_flu],
            "error_m": round(float(self.error_m), 3),
            "legs": int(self.legs),
            "status": self.status,
            "feedback": list(self.feedback),
        }


def world_executor(
    world: Any,
    target_flu: Iterable[float],
    *,
    max_leg_m: float = 8.0,
    max_legs: int = 8,
    arrive_radius_m: float = 0.6,
) -> ExecutorResult:
    """Fly ``world`` to a waypoint with the documented waypoint action semantics.

    Each leg turns toward the target and then translates along the *new* heading
    (``NavAction(kind="waypoint")``), which is exactly what the navigation tool
    hands the world - so a wrong waypoint and a broken executor are separable:
    the injected waypoint is right by construction, and what is left is the
    execution.
    """
    from vlm_nav.actions import NavAction

    target = np.asarray(_floats(target_flu, 3) or (0.0, 0.0, 0.0), dtype=np.float64)
    start = np.asarray(getattr(world, "position_flu", (0.0, 0.0, 0.0)), dtype=np.float64).reshape(3)
    feedback: list[str] = []
    legs = 0
    status = "no_move"
    for _ in range(max(1, int(max_legs))):
        position = np.asarray(getattr(world, "position_flu", start), dtype=np.float64).reshape(3)
        delta = target - position
        distance = float(np.linalg.norm(delta))
        if distance <= arrive_radius_m:
            status = "arrived"
            break
        yaw = float(getattr(world, "yaw_rad", 0.0) or 0.0)
        desired = math.atan2(float(delta[1]), float(delta[0]))
        dyaw_deg = wrap_deg(math.degrees(desired - yaw))
        forward = min(distance, float(max_leg_m))
        up = float(np.clip(delta[2], -float(max_leg_m), float(max_leg_m)))
        try:
            feedback.append(str(world.apply_action(
                NavAction(kind="waypoint", yaw_deg=dyaw_deg, forward_m=forward, up_m=up,
                          reason="diagnosis: injected waypoint")
            )))
        except Exception as exc:  # noqa: BLE001 - an executor that raises is the finding
            status = f"world_error:{type(exc).__name__}"
            feedback.append(f"{type(exc).__name__}: {exc}")
            break
        legs += 1
        moved = float(
            np.linalg.norm(
                np.asarray(getattr(world, "position_flu", position), dtype=np.float64).reshape(3)
                - position
            )
        )
        if moved < 1e-3:
            status = "stalled"
            break
    else:
        status = "leg_budget"
    final = np.asarray(getattr(world, "position_flu", start), dtype=np.float64).reshape(3)
    return ExecutorResult(
        target_flu=[float(v) for v in target],
        start_flu=[float(v) for v in start],
        final_flu=[float(v) for v in final],
        error_m=float(np.linalg.norm(final - target)),
        legs=legs,
        status=status,
        feedback=feedback,
    )


# ---------------------------------------------------------------------------
# the coordinate contract, mirrored for the audit
# ---------------------------------------------------------------------------


def pixel_to_world(
    pixel: Iterable[float],
    distance_m: float,
    pose_flu: Iterable[float],
    yaw_deg: float,
    intrinsics: Intrinsics,
    *,
    hfov_deg: float | None = None,
) -> list[float] | None:
    """Normalized pixel + axial range -> world position of the capture pose.

    The formula is the repository's own (``world.pixel_to_body_offset`` +
    ``actions.body_delta_to_world``) with the observation's intrinsics, so a
    disagreement with a tool is a plumbing finding; ``hfov_deg`` overrides the
    field of view, which is how the audit recomputes what the legacy 86 deg
    default would have produced.
    """
    hfov = float(hfov_deg if hfov_deg is not None else (intrinsics.hfov_deg or 0.0))
    if hfov <= 0.0 or intrinsics.width <= 0 or intrinsics.height <= 0:
        return None
    pixel = _floats(pixel, 2)
    pose = _floats(pose_flu, 3)
    if pixel is None or pose is None:
        return None
    dx, dy, dz = pixel_to_body_offset(
        pixel, float(distance_m),
        width=int(intrinsics.width), height=int(intrinsics.height), hfov_deg=hfov,
    )
    wx, wy, wz = body_delta_to_world(dx, dy, dz, math.radians(float(yaw_deg)))
    return [round(float(pose[0] + wx), 3), round(float(pose[1] + wy), 3), round(float(pose[2] + wz), 3)]


def pixel_ray(
    pixel: Iterable[float],
    pose_flu: Iterable[float],
    yaw_deg: float,
    intrinsics: Intrinsics,
) -> dict | None:
    """World-frame ray of a pixel from the capture pose (the honest no-range answer)."""
    pixel = _floats(pixel, 2)
    pose = _floats(pose_flu, 3)
    if pixel is None or pose is None or intrinsics.hfov_deg is None:
        return None
    dx, dy, dz = pixel_to_body_offset(
        pixel, 1.0, width=int(intrinsics.width), height=int(intrinsics.height),
        hfov_deg=float(intrinsics.hfov_deg),
    )
    body_bearing = math.atan2(dy, dx)
    horizontal = math.hypot(dx, dy)
    return {
        "origin_flu": [round(float(v), 3) for v in pose],
        "bearing_deg": round(wrap_deg(math.degrees(math.radians(float(yaw_deg)) + body_bearing)), 1),
        "elevation_deg": round(math.degrees(math.atan2(dz, horizontal)), 1),
        "yaw_deg": round(float(yaw_deg), 1),
    }


def fov_deviation_m(
    pixel: Iterable[float],
    distance_m: float,
    pose_flu: Iterable[float],
    yaw_deg: float,
    intrinsics: Intrinsics,
    *,
    legacy_hfov_deg: float = LEGACY_DEFAULT_HFOV_DEG,
) -> float | None:
    """How far the legacy 86 deg assumption moves a pointed world point (m).

    The value the coordinate contract exists to remove: the same pixel and range
    projected with the observation's own field of view and with the default one
    land in different places, and the distance between them is what a wrong FOV
    silently adds to every lateral offset.
    """
    own = pixel_to_world(pixel, distance_m, pose_flu, yaw_deg, intrinsics)
    legacy = pixel_to_world(pixel, distance_m, pose_flu, yaw_deg, intrinsics, hfov_deg=legacy_hfov_deg)
    if own is None or legacy is None:
        return None
    return float(np.linalg.norm(np.asarray(own) - np.asarray(legacy)))


def crop_frame_for_region(
    frame: torch.Tensor,
    region: Sequence[float],
    *,
    target_short: int = CROP_SHORT_SIDE_PX,
    max_upscale: float = CROP_MAX_UPSCALE,
) -> tuple[torch.Tensor, list[int], float]:
    """The magnified crop ``inspect_region`` should produce for a region.

    ``(crop, origin_px, upscale)`` with the same rounding and bilinear upscale as
    the tool's crop, so the audit can tell "the model studied the magnified region
    I asked for" from "the model was sent some other image".
    """
    import torch.nn.functional as F

    canonical = _floats(region, 4) or (0.0, 0.0, 1.0, 1.0)
    width, height = int(frame.shape[-1]), int(frame.shape[-2])
    u, v = canonical[0], canonical[1]
    w, h = canonical[2] - canonical[0], canonical[3] - canonical[1]
    x0 = int(round(u / 1000.0 * width))
    y0 = int(round(v / 1000.0 * height))
    x1 = int(round((u + w) / 1000.0 * width))
    y1 = int(round((v + h) / 1000.0 * height))
    x0, y0 = max(0, min(x0, width - 1)), max(0, min(y0, height - 1))
    x1, y1 = max(x0 + 1, min(x1, width)), max(y0 + 1, min(y1, height))
    crop = frame[:, y0:y1, x0:x1]
    short = max(1, min(int(crop.shape[-2]), int(crop.shape[-1])))
    upscale = max(1.0, min(float(max_upscale), float(target_short) / short))
    if upscale > 1.001:
        crop = F.interpolate(
            crop.unsqueeze(0), scale_factor=upscale, mode="bilinear", align_corners=False
        ).squeeze(0)
    if crop.is_floating_point():
        crop = crop.clamp(0.0, 1.0)
    return crop, [x0, y0, x1, y1], upscale


def box_to_crop_box(box: Sequence[float] | None, region: Sequence[float] | None,
                    origin_px: Sequence[float] | None = None,
                    source_size: Sequence[int] | None = None) -> list[float] | None:
    """A pixel in the source frame -> the same pixel in the crop's 0-1000 frame."""
    box = _floats(box, 4)
    region = _floats(region, 4)
    if box is None or region is None or source_size is None:
        return None
    x0 = float(origin_px[0]) if origin_px else region[0] / 1000.0 * float(source_size[0])
    y0 = float(origin_px[1]) if origin_px else region[1] / 1000.0 * float(source_size[1])
    crop_w = max(1e-6, (region[2] - region[0]) / 1000.0 * float(source_size[0]))
    crop_h = max(1e-6, (region[3] - region[1]) / 1000.0 * float(source_size[1]))
    out = []
    for index, value in enumerate(box):
        size = crop_w if index % 2 == 0 else crop_h
        offset = x0 if index % 2 == 0 else y0
        source_px = value / 1000.0 * float(source_size[index % 2])
        out.append(max(0.0, min(1000.0, (source_px - offset) / size * 1000.0)))
    return [round(float(v), 2) for v in out]


def crop_box_to_source_box(box: Sequence[float] | None, region: Sequence[float] | None,
                           source_size: Sequence[int] | None,
                           origin_px: Sequence[float] | None = None) -> list[float] | None:
    """A box inside the crop -> the same box in the source observation's frame."""
    box = _floats(box, 4)
    region = _floats(region, 4)
    if box is None or region is None or source_size is None:
        return None
    x0 = float(origin_px[0]) if origin_px else region[0] / 1000.0 * float(source_size[0])
    y0 = float(origin_px[1]) if origin_px else region[1] / 1000.0 * float(source_size[1])
    crop_w = max(1e-6, (region[2] - region[0]) / 1000.0 * float(source_size[0]))
    crop_h = max(1e-6, (region[3] - region[1]) / 1000.0 * float(source_size[1]))
    out = []
    for index, value in enumerate(box):
        size = crop_w if index % 2 == 0 else crop_h
        offset = x0 if index % 2 == 0 else y0
        cw, ch = float(source_size[0]), float(source_size[1])
        pixel = offset + value / 1000.0 * size
        out.append(max(0.0, min(1000.0, pixel / (cw if index % 2 == 0 else ch) * 1000.0)))
    return [round(float(v), 2) for v in out]


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


@dataclass
class Tolerances:
    """What counts as a hit (one place, so two runs are comparable)."""

    center_px: float = DEFAULT_CENTER_TOLERANCE_PX
    iou_min: float = DEFAULT_IOU_MIN
    position_m: float = DEFAULT_POSITION_TOLERANCE_M

    def to_dict(self) -> dict:
        return {"center_px": self.center_px, "iou_min": self.iou_min, "position_m": self.position_m}


def _box_center(box: Sequence[float]) -> tuple[float, float]:
    return (float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0


def _box_center_of(box: Sequence[float] | None) -> tuple[float, float]:
    """Image centre of a box, tolerating a 2-element point (the visible centre)."""
    if box is None:
        return (500.0, 500.0)
    values = [value for value in box]
    if len(values) == 2:
        point = _floats(values, 2)
        return (500.0, 500.0) if point is None else (float(point[0]), float(point[1]))
    numbers = _floats(values, 4)
    return (500.0, 500.0) if numbers is None else _box_center(numbers)


def box_center_error_px(
    answer_box: Sequence[float] | None,
    truth_box: Sequence[float] | None,
    intrinsics: Intrinsics,
) -> float | None:
    """Distance between the answer's centre and the target's, in frame pixels."""
    if answer_box is None or truth_box is None:
        return None
    ax, ay = _box_center(answer_box)
    tx, ty = _box_center(truth_box)
    dx = (ax - tx) / 1000.0 * float(intrinsics.width)
    dy = (ay - ty) / 1000.0 * float(intrinsics.height)
    return float(math.hypot(dx, dy))


def point_error_px(
    point: Sequence[float] | None,
    truth_box: Sequence[float] | None,
    intrinsics: Intrinsics,
) -> float | None:
    """Distance between a pointed pixel and the centre of the visible target."""
    if point is None or truth_box is None:
        return None
    tx, ty = _box_center(truth_box)
    dx = (float(point[0]) - tx) / 1000.0 * float(intrinsics.width)
    dy = (float(point[1]) - ty) / 1000.0 * float(intrinsics.height)
    return float(math.hypot(dx, dy))


def box_iou(box_a: Sequence[float] | None, box_b: Sequence[float] | None) -> float | None:
    if box_a is None or box_b is None:
        return None
    ax0, ay0, ax1, ay1 = (float(v) for v in box_a)
    bx0, by0, bx1, by1 = (float(v) for v in box_b)
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else float(inter / union)


def point_in_box(point: Sequence[float] | None, box: Sequence[float] | None) -> bool | None:
    if point is None or box is None:
        return None
    return bool(box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3])


def evaluate_answer(
    case: ObservationCase,
    *,
    answer_box: Sequence[float] | None = None,
    answer_point: Sequence[float] | None = None,
    parsed_ok: bool = False,
    model_abstained: bool = False,
    tool_reason: str = "",
    position_flu: Sequence[float] | None = None,
    position_source: str = "",
    range_status: str = "",
    range_m: float | None = None,
    ray: Mapping | None = None,
    intrinsics_used: Mapping | None = None,
    expect_position: bool = False,
    backend_error: str | None = None,
    tol: Tolerances | None = None,
) -> dict:
    """Measures + failure category + outcome of one answer (pure function).

    The failure taxonomy the diagnosis needs is decided here and nowhere else:
    "the model did not report a visible target" (``not_seen``) is a different row
    from "it wrote prose" (``output_format``), "it was handed another frame"
    (``wrong_image``), "the geometry it produced contradicts the contract"
    (``projection_error``) and "the backend blew up / it saw a target that is not
    there" (``other``). An honest "not visible" on an observation where the
    target really is invisible is *not* a failure: it is ``none`` with an
    abstention outcome, which is exactly the distinction ``no_candidate`` hides.
    """
    tol = tol or Tolerances()
    truth = case.truth
    intrinsics = case.intr
    truth_box = case.target_box
    measure: dict[str, Any] = {
        "target_visible": bool(truth.visible),
        "target_visible_fraction": round(float(truth.visible_fraction), 3),
        "target_occluded_fraction": round(float(truth.occluded_fraction), 3),
        "center_error_px": None,
        "iou": None,
        "point_in_gt_box": None,
        "position_error_m": None,
        "position_source": position_source or None,
        "range_status": range_status or None,
        "range_m": None if range_m is None else round(float(range_m), 3),
        "position_flu": None if position_flu is None else [round(float(v), 3) for v in position_flu],
        "fov_declared_deg": intrinsics.hfov_deg,
        "fov_used_deg": None,
        "intrinsics_match": None,
        "has_ray": bool(ray),
    }
    if isinstance(intrinsics_used, Mapping) and intrinsics_used:
        used = intrinsics_of(intrinsics_used)
        measure["fov_used_deg"] = used.hfov_deg
        measure["intrinsics_used"] = used.to_dict()
        measure["intrinsics_match"] = bool(
            used.width == intrinsics.width
            and used.height == intrinsics.height
            and _hfov_equal(used.hfov_deg, intrinsics.hfov_deg)
        )
    if answer_box is not None:
        measure["center_error_px"] = _round_opt(box_center_error_px(answer_box, truth_box, intrinsics), 2)
        measure["iou"] = _round_opt(box_iou(answer_box, truth_box), 3)
        measure["answer_box"] = _round_box(answer_box)
    if answer_point is not None:
        measure["center_error_px"] = _round_opt(point_error_px(answer_point, truth_box, intrinsics), 2)
        measure["point_in_gt_box"] = point_in_box(answer_point, truth_box)
        measure["answer_point"] = [round(float(v), 2) for v in answer_point]
    if position_flu is not None and truth.position_flu:
        error = float(
            np.linalg.norm(
                np.asarray([float(v) for v in position_flu]) - np.asarray(truth.position_flu)
            )
        )
        measure["position_error_m"] = round(error, 3)

    failure = {"category": FAILURE_NONE, "detail": ""}
    outcome = OUTCOME_MISS
    if backend_error:
        failure = {"category": FAILURE_OTHER, "detail": f"backend_error:{backend_error}"}
        outcome = OUTCOME_ERROR
    elif not parsed_ok:
        if truth.visible:
            detail = "model_reported_absent" if model_abstained else f"no_usable_answer:{tool_reason or 'unknown'}"
            failure = {"category": FAILURE_NOT_SEEN, "detail": detail}
            outcome = OUTCOME_MISS
        else:
            failure = {
                "category": FAILURE_NONE,
                "detail": "abstain_correct" if model_abstained else "no_answer_on_absent_target",
            }
            outcome = OUTCOME_ABSTAIN
    elif not truth.visible:
        failure = {"category": FAILURE_OTHER, "detail": "false_positive_where_target_absent"}
        outcome = OUTCOME_FALSE_POSITIVE
    elif truth_box is None:
        failure = {"category": FAILURE_OTHER, "detail": "no_ground_truth_box_for_a_visible_target"}
        outcome = OUTCOME_ERROR
    else:
        error = measure["center_error_px"]
        iou_ok = measure["iou"] is None or measure["iou"] >= float(tol.iou_min)
        if error is None or error > float(tol.center_px) or not iou_ok:
            failure = {
                "category": FAILURE_NOT_SEEN,
                "detail": f"answer_far_from_target:center_error_px={error},iou={measure['iou']}",
            }
            outcome = OUTCOME_MISS
        else:
            outcome = OUTCOME_HIT
            detail = "hit"
            if measure["intrinsics_match"] is False:
                failure = {
                    "category": FAILURE_PROJECTION,
                    "detail": f"intrinsics_mismatch:used={measure['fov_used_deg']},"
                              f"declared={measure['fov_declared_deg']}",
                }
                outcome = OUTCOME_MISS
            elif measure["position_flu"] is not None and range_status != "known":
                failure = {
                    "category": FAILURE_PROJECTION,
                    "detail": f"position_without_range:range_status={range_status or 'missing'}",
                }
                outcome = OUTCOME_MISS
            elif measure["position_flu"] is not None and (measure["position_error_m"] or 0.0) > float(tol.position_m):
                hint = (
                    "; the depth at the box centre may be the occluder, not the target"
                    if measure["target_occluded_fraction"] >= 0.15
                    else ""
                )
                failure = {
                    "category": FAILURE_PROJECTION,
                    "detail": f"position_error_m={measure['position_error_m']}"
                              f":position_source={position_source or 'missing'}"
                              f",target_visible_fraction={measure['target_visible_fraction']}"
                              f",occluded_fraction={measure['target_occluded_fraction']}"
                              + hint,
                }
                outcome = OUTCOME_MISS
            elif measure["position_flu"] is None and expect_position:
                failure = {
                    "category": FAILURE_PROJECTION,
                    "detail": f"no_position_despite_range:position_source={position_source or 'missing'}",
                }
                outcome = OUTCOME_MISS
            elif measure["position_flu"] is None and not ray:
                failure = {
                    "category": FAILURE_PROJECTION,
                    "detail": f"no_position_and_no_ray:position_source={position_source or 'missing'}",
                }
                outcome = OUTCOME_MISS
            else:
                failure = {"category": FAILURE_NONE, "detail": detail}
    return {"measure": measure, "failure": failure, "outcome": outcome}


def _round_opt(value: float | None, digits: int) -> float | None:
    return None if value is None else round(float(value), digits)


def _hfov_equal(a: float | None, b: float | None) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= 0.05


# ---------------------------------------------------------------------------
# the runner: one case, several strategies, every call on disk
# ---------------------------------------------------------------------------


@dataclass
class CaseRun:
    """Everything one observation needs: pixels, ids, pose, world."""

    case: ObservationCase
    frame: torch.Tensor
    ctx: ToolContext
    obs_id: str
    world: Any = None

    @property
    def intr(self) -> Intrinsics:
        return self.case.intr


@dataclass
class BenchConfig:
    """How a bench run behaves (all defaults are the diagnostic defaults)."""

    out_dir: Path
    strategies: Sequence[str] = STRATEGIES
    tolerances: Tolerances = field(default_factory=Tolerances)
    detector: DetectorBackend | None = None
    inject_point: str = ""
    inject_waypoint: str = ""
    inject_wrong_image: bool = False
    oracle_replies: bool = False
    render_size: tuple[int, int] = (320, 180)
    hfov_deg: float = 60.0
    max_cases: int | None = None
    world_factory: Callable[[SceneSpec, tuple[int, int]], Any] | None = None
    log: Callable[[str], None] | None = None

    def log_line(self, text: str) -> None:
        if callable(self.log):
            self.log(text)


def prepare_case(case: ObservationCase, *, world: Any = None) -> CaseRun:
    """Register the frozen observation in a fresh context (the tools' entry point).

    The observation store is the same repository the runtime uses, so the case
    goes through the real resolution path: an ``obs_N`` the bench registered is
    an id ``ground_target`` / ``inspect_region`` resolve, with the capture pose,
    the heading and the intrinsics the frame was rendered with.
    """
    frame = case.load_frame()
    ctx = ToolContext(world=world, episode_id=f"diag_{case.case_id}", extra={})
    obs_id = ctx.next_obs_id()
    register_observation(
        ctx,
        frame,
        kind=KIND_EVIDENCE,
        obs_id=obs_id,
        pose_flu=case.pose_flu,
        yaw_deg=case.yaw_deg,
        intrinsics=case.intrinsics,
        image_path=case.image_path,
        note=f"fixed observation {case.case_id} ({case.kind})",
    )
    if world is not None:
        set_world_pose(world, case.pose_flu, case.yaw_deg)
    return CaseRun(case=case, frame=frame, ctx=ctx, obs_id=obs_id, world=world)


def read_depth(run: CaseRun, pixel: Sequence[float]) -> tuple[float | None, str]:
    """Axial depth at a normalized pixel of the case's frame; (None, reason)."""
    world = run.world
    if world is None:
        return None, "no_world"
    depth_fn = getattr(world, "depth_at_pixel", None)
    if not callable(depth_fn):
        return None, "no_depth_sensor"
    if not all(math.isfinite(float(v)) for v in run.case.pose_flu):
        return None, "pose_unknown"
    try:
        depth = depth_fn((float(pixel[0]), float(pixel[1])))
        refresh = getattr(world, "refresh_sensing", None)
        if callable(refresh):
            refresh()
            depth = depth_fn((float(pixel[0]), float(pixel[1])))
    except Exception as exc:  # noqa: BLE001 - a depth failure is a finding, not a crash
        return None, f"depth_error:{type(exc).__name__}"
    if depth is None:
        return None, "no_valid_depth"
    depth = float(depth)
    return (None, "no_valid_depth") if depth <= MIN_RANGE_M else (depth, "")


def locate_pixel(run: CaseRun, pixel: Sequence[float], distance_m: float | None = None) -> dict:
    """The documented post-grounding chain: pixel (+ range) -> position or ray.

    Mirrors what the tools do with a box centre - depth map first, the model's own
    distance second, a ray when there is no range and nothing invented in
    between - and additionally records what the legacy 86 deg field of view would
    have produced, so the FOV deviation is a number in the report instead of a
    suspicion.
    """
    case = run.case
    intrinsics = run.intr
    located: dict[str, Any] = {
        "pixel": [round(float(pixel[0]), 2), round(float(pixel[1]), 2)],
        "position_flu": None,
        "position_source": POSITION_NO_RANGE,
        "range_m": None,
        "range_source": None,
        "range_status": "unknown",
        "ray": None,
        "depth_m": None,
        "depth_reason": "",
        "legacy_position_flu": None,
        "fov_deviation_m": fov_deviation_m(pixel, 1.0, case.pose_flu, case.yaw_deg, intrinsics),
        "fov_deviation_at_1m_m": fov_deviation_m(pixel, 1.0, case.pose_flu, case.yaw_deg, intrinsics),
    }
    depth, reason = read_depth(run, pixel)
    located["depth_m"] = None if depth is None else round(depth, 3)
    located["depth_reason"] = reason
    range_m = depth
    range_source = "depth" if depth is not None else None
    if range_m is None and distance_m is not None and MIN_RANGE_M < float(distance_m) <= MAX_RANGE_M:
        range_m = float(distance_m)
        range_source = "model_distance"
    if range_m is not None:
        located["range_m"] = round(float(range_m), 3)
        located["range_source"] = range_source
        position = pixel_to_world(pixel, range_m, case.pose_flu, case.yaw_deg, intrinsics)
        if position is not None:
            located["position_flu"] = position
            located["position_source"] = "depth" if range_source == "depth" else "model_distance"
            located["range_status"] = "known"
        else:
            # a real range with no declared field of view: report the range and
            # refuse the direction, exactly like the tools do
            located["position_source"] = POSITION_INTRINSICS_UNKNOWN
            located["range_status"] = "known"
        located["legacy_position_flu"] = pixel_to_world(
            pixel, range_m, case.pose_flu, case.yaw_deg, intrinsics,
            hfov_deg=LEGACY_DEFAULT_HFOV_DEG,
        )
        located["fov_deviation_m"] = fov_deviation_m(
            pixel, range_m, case.pose_flu, case.yaw_deg, intrinsics
        )
    ray = pixel_ray(pixel, case.pose_flu, case.yaw_deg, intrinsics)
    if ray is not None:
        located["ray"] = ray
        if located["position_flu"] is None and located["range_status"] == "unknown":
            located["position_source"] = POSITION_RAY
    return located


def _record_tool(record: dict, name: str, result: ToolResult) -> None:
    data = result.data if isinstance(result.data, dict) else {}
    record["tool"] = {
        "name": name,
        "status": result.status,
        "reason": data.get("reason"),
        "error": data.get("error"),
        "n_candidates": data.get("n_candidates"),
        "box_format": data.get("box_format"),
        "intrinsics": data.get("intrinsics"),
        "depth_status": data.get("depth_status"),
    }


def _first_candidate(result: ToolResult) -> dict | None:
    candidates = (result.data or {}).get("candidates") or []
    return candidates[0] if candidates else None


def _call_tool_step(
    recorder: CallRecorder,
    case: ObservationCase,
    strategy: str,
    step: str,
    frames: Sequence[Any],
    call: Callable[[], ToolResult],
    **step_kwargs: Any,
) -> tuple[dict, ToolResult | None, str | None]:
    """Run one tool call as a recorded step; ``(record, result, error)``.

    The record is the one the *model call inside the tool* produced; a tool that
    refuses before reaching the model (an id that does not resolve, a region the
    parser rejects) still gets a record of its own - otherwise the audit would
    silently attribute the refusal to the previous step.
    """
    recorder.begin_step(strategy=strategy, step=step, expected_frames=list(frames), **step_kwargs)
    before = len(recorder.records)
    result: ToolResult | None = None
    error: str | None = None
    try:
        result = call()
    except Exception as exc:  # noqa: BLE001 - a tool that raises is a finding
        error = f"{type(exc).__name__}: {exc}"
    record = recorder.records[-1] if len(recorder.records) > before else None
    recorder.end_step()
    if record is None:
        record = _synthetic_record(
            recorder, case, strategy, step, frames=frames, diagnosis_only=False
        )
        record["backend"] = "no_model_call"
    return record, result, error


def _range_consistency(
    run: CaseRun,
    answer_box: Sequence[float] | None,
    range_m: Any,
    position_source: str,
) -> tuple[dict | None, str]:
    """Was the range a tool projected with read at the answer's own pixel?

    The failure this catches is invisible in a success rate: ``ground_target``
    reads the depth at the *box centre* it was given, and a box in a magnified
    crop's coordinates is not a pixel of the camera frame - so the depth of an
    unrelated pixel (the frame centre) can end up in the position of a perfectly
    localized target. Recomputation is the only way to see it, hence the check.
    """
    if answer_box is None or position_source != "depth":
        return None, ""
    try:
        tool_range = None if range_m is None else float(range_m)
    except (TypeError, ValueError):
        tool_range = None
    depth, reason = read_depth(run, _box_center_of(answer_box))
    if depth is None:
        return None, ""
    check = {
        "depth_at_answer_px_m": round(float(depth), 3),
        "tool_range_m": None if tool_range is None else round(tool_range, 3),
        "matches": None if tool_range is None else bool(
            abs(tool_range - float(depth)) <= max(0.2, 0.15 * float(depth))
        ),
    }
    if not check["matches"]:
        check["note"] = f"depth read elsewhere ({reason or 'pixel mismatch'})"
        return check, (
            "; the range does not belong to the answer's pixel "
            "(a box in crop coordinates reads the depth of another pixel)"
        )
    return check, ""


def _finish_record(record: dict, evaluation: Mapping, *, detail_suffix: str = "") -> None:
    failure = dict(evaluation["failure"])
    if detail_suffix:
        failure["detail"] = f"{failure.get('detail') or ''}{detail_suffix}"
    record["measure"] = evaluation["measure"]
    record["failure"] = failure
    record["outcome"] = evaluation["outcome"]


def strategy_structured_box(run: CaseRun, backend: Any, config: BenchConfig,
                            recorder: CallRecorder) -> None:
    """(a) today's tool: ``ground_target`` on the frozen frame."""
    case = run.case
    record, result, error = _call_tool_step(
        recorder, case, STRATEGY_STRUCTURED, "ground_target", [run.frame],
        lambda: ground_target(
            run.ctx, {"observation_id": run.obs_id, "instruction": case.instruction}, vlm=backend
        ),
        obs_id=run.obs_id,
    )
    if result is None:
        evaluation = evaluate_answer(
            case, parsed_ok=False, tool_reason=str(error), backend_error=error,
            tol=config.tolerances,
        )
        _finish_record(record, evaluation)
        return
    _record_tool(record, "ground_target", result)
    candidate = _first_candidate(result)
    if candidate is None:
        record["parsed"] = {"n_candidates": (result.data or {}).get("n_candidates", 0)}
        evaluation = evaluate_answer(
            case,
            parsed_ok=False,
            model_abstained=(result.data or {}).get("reason") == "no_candidate",
            tool_reason=str((result.data or {}).get("reason") or (result.data or {}).get("error") or ""),
            backend_error=record.get("error"),
            intrinsics_used=(result.data or {}).get("intrinsics"),
            tol=config.tolerances,
        )
        _finish_record(record, evaluation)
        return
    box, box_reason = parse_box(
        candidate.get("box"),
        width=int(case.intrinsics.get("width") or 0),
        height=int(case.intrinsics.get("height") or 0),
        box_format=candidate.get("box_format"),
    )
    record["parsed"] = {
        "candidate_id": candidate.get("id"),
        "box": candidate.get("box"),
        "box_uvwh": candidate.get("box_uvwh"),
        "box_format": candidate.get("box_format"),
        "box_format_used": candidate.get("box_format_used"),
        "attrs": candidate.get("attrs"),
        "confidence": candidate.get("confidence"),
        "model_distance_m": candidate.get("distance_m"),
        "uncertainty": candidate.get("uncertainty"),
        "position_flu": candidate.get("position_flu"),
        "position_source": candidate.get("position_source"),
        "range_m": candidate.get("range_m"),
        "range_status": candidate.get("range_status"),
        "ray": candidate.get("ray"),
        "parse_reason": box_reason,
    }
    if box is None:
        evaluation = evaluate_answer(
            case, parsed_ok=False, tool_reason=f"unusable_box:{box_reason}",
            intrinsics_used=(result.data or {}).get("intrinsics"), tol=config.tolerances,
        )
        _finish_record(record, evaluation)
        return
    answer_box = box.to_list()
    depth, _reason = read_depth(run, box.center)
    expect_position = bool(
        case.truth.visible
        and depth is not None
        and (case.intr.hfov_deg is not None or abs(box.center[0] - 500.0) <= 500.0 / max(1, case.intr.width))
    )
    range_check, range_detail = _range_consistency(
        run, answer_box, candidate.get("range_m"), str(candidate.get("position_source") or "")
    )
    if range_check is not None:
        record["parsed"]["range_check"] = range_check
    evaluation = evaluate_answer(
        case,
        answer_box=answer_box,
        parsed_ok=True,
        position_flu=candidate.get("position_flu"),
        position_source=str(candidate.get("position_source") or ""),
        range_status=str(candidate.get("range_status") or ""),
        range_m=candidate.get("range_m"),
        ray=candidate.get("ray"),
        intrinsics_used=(result.data or {}).get("intrinsics"),
        expect_position=expect_position,
        tol=config.tolerances,
    )
    _finish_record(record, evaluation, detail_suffix=range_detail)


def strategy_point_pixel(run: CaseRun, backend: Any, config: BenchConfig,
                         recorder: CallRecorder) -> None:
    """(b) the model points at the target centre; the chain projects that pixel."""
    case = run.case
    recorder.begin_step(
        strategy=STRATEGY_POINT, step="point",
        expected_frames=[run.frame], obs_id=run.obs_id,
    )
    try:
        raw = backend.decide(
            [run.frame],
            POINT_SYSTEM_PROMPT,
            point_user_prompt(case.instruction),
            context={"observation_id": run.obs_id, "tool": "point_pixel"},
        )
        error = None
    except Exception as exc:  # noqa: BLE001 - a backend that raises is a data point
        raw, error = None, f"{type(exc).__name__}: {exc}"
    record = recorder.last() or {}
    recorder.end_step()
    if not record:
        return
    point, visible, distance, reason = parse_point_reply(raw if raw is not None else "")
    located = locate_pixel(run, point, distance) if point is not None else {}
    record["parsed"] = {
        "point": None if point is None else [round(float(v), 2) for v in point],
        "visible": bool(visible),
        "model_distance_m": distance,
        "parse_reason": reason,
        "projection": located,
    }
    evaluation = evaluate_answer(
        case,
        answer_point=point,
        parsed_ok=point is not None and not error,
        model_abstained=not visible,
        tool_reason=reason,
        position_flu=located.get("position_flu"),
        position_source=str(located.get("position_source") or ""),
        range_status=str(located.get("range_status") or ""),
        range_m=located.get("range_m"),
        ray=located.get("ray"),
        intrinsics_used=case.intrinsics,
        expect_position=bool(located.get("range_status") == "known" and case.truth.visible),
        backend_error=error,
        tol=config.tolerances,
    )
    _finish_record(record, evaluation)


def strategy_region_then_local(run: CaseRun, backend: Any, config: BenchConfig,
                               recorder: CallRecorder) -> None:
    """(c) model-chosen region -> ``inspect_region`` -> ground inside the crop.

    Three recorded steps: the region the model proposed, the crop the tool made
    of it (which is where a wrong-region or wrong-image bug shows up) and the
    local grounding inside the magnified frame. Only the last one is scored; the
    first two stay in the audit with their own measures.
    """
    case = run.case
    recorder.begin_step(
        strategy=STRATEGY_REGION, step="region",
        expected_frames=[run.frame], obs_id=run.obs_id,
    )
    before = len(recorder.records)
    raw: Any = None
    error: str | None = None
    try:
        raw = backend.decide(
            [run.frame],
            REGION_SYSTEM_PROMPT,
            region_user_prompt(case.instruction),
            context={"observation_id": run.obs_id, "tool": STRATEGY_REGION},
        )
    except Exception as exc:  # noqa: BLE001 - a backend that raises is a data point
        error = f"{type(exc).__name__}: {exc}"
    region_record = recorder.records[-1] if len(recorder.records) > before else None
    recorder.end_step()
    if region_record is None:
        region_record = _synthetic_record(
            recorder, case, STRATEGY_REGION, "region", frames=[run.frame], diagnosis_only=False
        )
        region_record["backend"] = "no_model_call"
    region, visible, reason = parse_region_reply(raw if raw is not None else "")
    region_record["parsed"] = {
        "region": region,
        "visible": bool(visible),
        "parse_reason": reason,
    }
    region_record["measure"] = {"target_visible": bool(case.truth.visible)}
    if region is None:
        if error:
            region_record["failure"] = {"category": FAILURE_OTHER, "detail": f"backend_error:{error}"}
            region_record["outcome"] = OUTCOME_ERROR
        elif not visible and not case.truth.visible:
            region_record["failure"] = {"category": FAILURE_NONE, "detail": "abstain_correct"}
            region_record["outcome"] = OUTCOME_ABSTAIN
        elif not visible:
            region_record["failure"] = {"category": FAILURE_NOT_SEEN, "detail": "model_reported_absent"}
            region_record["outcome"] = OUTCOME_MISS
        else:
            region_record["failure"] = {"category": FAILURE_OUTPUT_FORMAT, "detail": f"region:{reason}"}
            region_record["outcome"] = OUTCOME_MISS
        return

    truth_box = case.target_box
    region_record["measure"].update({
        "iou": None if truth_box is None else _round_opt(box_iou(region, truth_box), 3),
        "point_in_gt_box": point_in_box(_box_center(region), truth_box),
        "region": _round_box(region),
    })

    # the region is magnified by the real tool, so the crop chain is measured too
    crop_record, crop_result, crop_error = _call_tool_step(
        recorder, case, STRATEGY_REGION, "crop", [run.frame],
        lambda: inspect_region(
            run.ctx, {"observation_id": run.obs_id, "region": region, "box_format": "xyxy"}
        ),
        obs_id=run.obs_id,
    )
    region_record["tool"] = {
        "name": "inspect_region",
        "status": None if crop_result is None else crop_result.status,
        "error": crop_error or ((crop_result.data or {}).get("error") if crop_result else None),
    }
    if crop_result is None or crop_result.status != STATUS_OK:
        reason_text = crop_error or str(
            (crop_result.data or {}).get("reason") or (crop_result.data or {}).get("error") or "unknown"
        )
        region_record["failure"] = {"category": FAILURE_OUTPUT_FORMAT, "detail": f"crop_refused:{reason_text}"}
        region_record["outcome"] = OUTCOME_MISS
        crop_record["final"] = False
        crop_record["failure"] = {"category": FAILURE_OUTPUT_FORMAT, "detail": f"crop_refused:{reason_text}"}
        crop_record["outcome"] = OUTCOME_INTERMEDIATE
        crop_record["measure"] = {"target_visible": bool(case.truth.visible)}
        return

    crop_id = str((crop_result.data or {}).get("crop_image_ref"))
    region_out = (crop_result.data or {}).get("region") or region
    crop_size = (crop_result.data or {}).get("crop_size_px") or []
    source_size = [int(case.intrinsics.get("width") or 0), int(case.intrinsics.get("height") or 0)]
    truth_in_crop = box_to_crop_box(truth_box, region_out, None, source_size)
    expected_crop, origin_px, upscale = crop_frame_for_region(run.frame, region_out)
    region_record["final"] = False
    region_record["failure"] = {"category": FAILURE_NONE, "detail": "region_selected"}
    region_record["outcome"] = OUTCOME_INTERMEDIATE
    crop_record["parsed"] = {
        "crop_image_ref": crop_id,
        "region": _round_box(region_out),
        "crop_size_px": crop_size,
        "upscale": round(float(upscale), 2),
        "truth_box_in_crop": truth_in_crop,
    }
    crop_record["measure"] = {
        "target_visible": bool(case.truth.visible),
        "iou": None if truth_box is None else _round_opt(box_iou(region_out, truth_box), 3),
        "point_in_gt_box": point_in_box(_box_center(region_out), truth_box),
    }
    crop_record["failure"] = {"category": FAILURE_NONE, "detail": "crop_produced"}
    crop_record["outcome"] = OUTCOME_INTERMEDIATE
    crop_record["final"] = False

    local_record, local_result, local_error = _call_tool_step(
        recorder, case, STRATEGY_REGION, "ground_local", [expected_crop],
        lambda: ground_target(
            run.ctx, {"observation_id": crop_id, "instruction": case.instruction}, vlm=backend
        ),
        obs_id=crop_id,
        expected_source="crop_of_fixed_observation",
        crop={"crop_image_ref": crop_id, "region": region_out, "origin_px": origin_px,
              "crop_size_px": crop_size, "truth_box": truth_in_crop},
    )
    local_record["obs_id"] = crop_id
    if local_result is None:
        _finish_record(local_record, evaluate_answer(
            case, parsed_ok=False, tool_reason=str(local_error), backend_error=local_error,
            tol=config.tolerances,
        ))
        return
    _record_tool(local_record, "ground_target(crop)", local_result)
    candidate = _first_candidate(local_result)
    if candidate is None:
        local_record["parsed"] = {"n_candidates": (local_result.data or {}).get("n_candidates", 0)}
        _finish_record(local_record, evaluate_answer(
            case,
            parsed_ok=False,
            model_abstained=(local_result.data or {}).get("reason") == "no_candidate",
            tool_reason=str((local_result.data or {}).get("reason")
                            or (local_result.data or {}).get("error") or ""),
            intrinsics_used=(local_result.data or {}).get("intrinsics"),
            tol=config.tolerances,
        ))
        return
    source_box = crop_box_to_source_box(candidate.get("box"), region_out, source_size, origin_px)
    local_record["parsed"] = {
        "candidate_id": candidate.get("id"),
        "crop_box": candidate.get("box"),
        "box_format_used": candidate.get("box_format_used"),
        "source_box": source_box,
        "position_flu": candidate.get("position_flu"),
        "position_source": candidate.get("position_source"),
        "range_m": candidate.get("range_m"),
        "range_status": candidate.get("range_status"),
        "ray": candidate.get("ray"),
        "attrs": candidate.get("attrs"),
    }
    if source_box is None:
        _finish_record(local_record, evaluate_answer(
            case, parsed_ok=False, tool_reason="no_crop_mapping",
            intrinsics_used=(local_result.data or {}).get("intrinsics"), tol=config.tolerances,
        ))
        return
    range_check, range_detail = _range_consistency(
        run, source_box, candidate.get("range_m"), str(candidate.get("position_source") or "")
    )
    if range_check is not None:
        local_record["parsed"]["range_check"] = range_check
    _finish_record(local_record, evaluate_answer(
        case,
        answer_box=source_box,
        parsed_ok=True,
        position_flu=candidate.get("position_flu"),
        position_source=str(candidate.get("position_source") or ""),
        range_status=str(candidate.get("range_status") or ""),
        range_m=candidate.get("range_m"),
        ray=candidate.get("ray"),
        intrinsics_used=(local_result.data or {}).get("intrinsics"),
        expect_position=False,
        tol=config.tolerances,
    ), detail_suffix=range_detail)


def strategy_detector_slot(run: CaseRun, backend: Any, config: BenchConfig,
                           recorder: CallRecorder) -> None:
    """(d) a detection/segmentation backend, when one is wired.

    The default slot is empty and every point of that layer (image in, canonical
    box out, projection, score) is still exercised - as ``not_run`` with
    ``detector_slot_empty``, never as a fabricated hit.
    """
    case = run.case
    detector = config.detector
    record = _synthetic_record(
        recorder, case, STRATEGY_DETECTOR, "detector", frames=[run.frame],
        diagnosis_only=isinstance(detector, OracleDetector),
    )
    record["backend"] = f"detector:{getattr(detector, 'name', 'none')}"
    record["system_prompt"] = None
    record["user_prompt"] = None
    if detector is None or not detector.available():
        record["parsed"] = {
            "detector": getattr(detector, "name", "null"),
            "note": getattr(detector, "note", "") or "no detector backend",
        }
        record["failure"] = {"category": FAILURE_SKIPPED, "detail": "detector_slot_empty"}
        record["outcome"] = OUTCOME_NOT_RUN
        record["measure"] = {"target_visible": bool(case.truth.visible), "slot": "empty"}
        recorder.flush()
        return
    try:
        detections = list(detector.detect(run.frame, case.instruction, case=case))
        detect_error = None
    except Exception as exc:  # noqa: BLE001 - a detector that raises is a finding
        detections, detect_error = [], f"{type(exc).__name__}: {exc}"
    if detect_error is not None:
        record["parsed"] = {"detector": getattr(detector, "name", ""), "error": detect_error}
        record["failure"] = {"category": FAILURE_OTHER, "detail": f"detector_error:{detect_error}"}
        record["outcome"] = OUTCOME_ERROR
        record["measure"] = {"target_visible": bool(case.truth.visible)}
        recorder.flush()
        return
    best = detections[0] if detections else None
    record["parsed"] = {
        "detector": getattr(detector, "name", ""),
        "n_detections": len(detections),
        "detections": [detection.to_dict() for detection in detections[:5]],
    }
    answer_box = None if best is None else _floats(best.box, 4)
    if answer_box is not None:
        located = locate_pixel(run, _box_center(answer_box), None)
        position = located.get("position_flu")
        record["parsed"]["projection"] = located
    else:
        located, position = {}, None
    evaluation = evaluate_answer(
        case,
        answer_box=answer_box,
        parsed_ok=answer_box is not None,
        position_flu=position,
        position_source=str(located.get("position_source") or ""),
        range_status=str(located.get("range_status") or ""),
        range_m=located.get("range_m"),
        ray=located.get("ray"),
        intrinsics_used=case.intrinsics,
        expect_position=bool(located.get("range_status") == "known" and case.truth.visible),
        tol=config.tolerances,
    )
    _finish_record(record, evaluation)
    recorder.flush()


STRATEGY_FUNCTIONS = {
    STRATEGY_STRUCTURED: strategy_structured_box,
    STRATEGY_POINT: strategy_point_pixel,
    STRATEGY_REGION: strategy_region_then_local,
    STRATEGY_DETECTOR: strategy_detector_slot,
}


# ---------------------------------------------------------------------------
# diagnosis-only replacements (never part of the score)
# ---------------------------------------------------------------------------


def injection_point_of(spec: str, case: ObservationCase) -> list[float] | None:
    """``--inject-point`` value -> the pixel injected for this case.

    ``gt`` injects the centre of the target as the answer key knows it (the
    correct image point); ``u,v`` injects a literal normalized pixel (for
    probing the projection with a point the model would never produce).
    """
    spec = str(spec or "").strip().lower()
    if not spec:
        return None
    if spec == "gt":
        # the centroid of the visible pixels, not the bounding box centre: on a
        # partly hidden target the box centre can fall on the occluder
        return case.target_point
    numbers = _floats([part for part in spec.replace(" ", ",").split(",") if part], 2)
    return None if numbers is None else [float(numbers[0]), float(numbers[1])]


def injection_waypoint_of(spec: str, case: ObservationCase) -> list[float] | None:
    """``--inject-waypoint`` value -> the world waypoint injected for this case."""
    spec = str(spec or "").strip().lower()
    if not spec:
        return None
    if spec == "gt":
        position = _floats(case.truth.position_flu, 3)
        return None if position is None else [float(v) for v in position]
    numbers = _floats([part for part in spec.replace(" ", ",").split(",") if part], 3)
    return None if numbers is None else [float(v) for v in numbers]


def run_injections(
    run: CaseRun,
    config: BenchConfig,
    recorder: CallRecorder,
    *,
    executor: Callable[[Any, Sequence[float]], ExecutorResult] | None = None,
) -> None:
    """The diagnosis-only replacements: correct point, correct waypoint.

    Both exist to separate the two halves of the failure "the drone did not
    move": a projection chain that cannot turn a known-good pixel into the right
    world point, and an executor that cannot fly to a known-good waypoint. A
    wrong answer here is a bug in the wrapper, not in the model, and the records
    are flagged ``diagnosis_only`` so nobody reads them as a score.
    """
    case = run.case
    pixel = injection_point_of(config.inject_point, case)
    if pixel is not None:
        located = locate_pixel(run, pixel)
        record = _synthetic_record(
            recorder, case, "inject_point", "project",
            frames=[run.frame],
            record_extra={"parsed": {"injected_point": [round(v, 2) for v in pixel],
                                     "projection": located}},
        )
        expect_position = case.truth.visible
        evaluation = evaluate_answer(
            case,
            answer_point=pixel,
            parsed_ok=True,
            position_flu=located.get("position_flu"),
            position_source=str(located.get("position_source") or ""),
            range_status=str(located.get("range_status") or ""),
            range_m=located.get("range_m"),
            ray=located.get("ray"),
            intrinsics_used=case.intrinsics,
            expect_position=expect_position,
            tol=config.tolerances,
        )
        _finish_record(record, evaluation)
        record["failure"]["detail"] = f"injected_point:{record['failure']['detail']}"
        recorder.flush()

    waypoint = injection_waypoint_of(config.inject_waypoint, case)
    if waypoint is not None and run.world is not None:
        set_world_pose(run.world, case.pose_flu, case.yaw_deg)
        fly = executor or world_executor
        result = fly(run.world, waypoint)
        record = _synthetic_record(
            recorder, case, "inject_waypoint", "execute",
            frames=[run.frame],
            record_extra={"parsed": {"injected_waypoint": [round(v, 3) for v in waypoint],
                                     "execution": result.to_dict()}},
        )
        error = float(result.error_m)
        record["measure"] = {
            "target_visible": bool(case.truth.visible),
            "position_error_m": round(error, 3),
            "position_flu": [round(v, 3) for v in result.final_flu],
            "waypoint": [round(v, 3) for v in waypoint],
            "legs": result.legs,
            "executor_status": result.status,
        }
        if error <= float(config.tolerances.position_m):
            record["failure"] = {"category": FAILURE_NONE, "detail": "executor_reached_waypoint"}
            record["outcome"] = OUTCOME_HIT
        else:
            record["failure"] = {
                "category": FAILURE_OTHER,
                "detail": f"executor_error_m={round(error, 3)}:status={result.status}",
            }
            record["outcome"] = OUTCOME_MISS
        # leave the world where the strategies of the *next* case expect it
        set_world_pose(run.world, case.pose_flu, case.yaw_deg)
        recorder.flush()


def _synthetic_record(
    recorder: CallRecorder,
    case: ObservationCase,
    strategy: str,
    step: str,
    *,
    frames: Sequence[Any] = (),
    record_extra: Mapping | None = None,
    diagnosis_only: bool = True,
) -> dict:
    """A record for a step that has no model call (an injected answer)."""
    recorder.begin_step(
        strategy=strategy, step=step, expected_frames=list(frames),
        expected_source="case_frame", diagnosis_only=diagnosis_only,
    )
    _, record = recorder.prepare(frames)
    if record_extra:
        record.update({key: value for key, value in record_extra.items() if key not in record})
    record["backend"] = "diagnosis"
    record["system_prompt"] = None
    record["user_prompt"] = None
    recorder.finish(record, None, backend="diagnosis")
    recorder.end_step()
    return record


# ---------------------------------------------------------------------------
# the bench itself
# ---------------------------------------------------------------------------


def run_bench(
    cases: Sequence[ObservationCase],
    backend: Any,
    config: BenchConfig,
    *,
    scene: SceneSpec | None = None,
    recorder: CallRecorder | None = None,
    executor: Callable[[Any, Sequence[float]], ExecutorResult] | None = None,
) -> dict:
    """Replay every strategy on every frozen observation; returns the summary.

    One world for the whole run (re-posed per case, so depth reads belong to the
    observation being studied), one recording wrapper in front of the backend,
    and one JSONL line per call - the run is reproducible from its own artifacts.
    """
    config.out_dir = Path(config.out_dir)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    recorder = recorder or CallRecorder(config.out_dir, inject_wrong_image=config.inject_wrong_image)
    wrapper = RecordingBackend(backend, recorder)
    world = None
    if scene is not None and (scene.ply_path or config.world_factory is not None):
        factory = config.world_factory or default_world_factory
        try:
            world = factory(scene, config.render_size)
        except Exception as exc:  # noqa: BLE001 - no world costs depth, not the run
            world = None
            config.log_line(f"no world for this scene ({type(exc).__name__}: {exc}); "
                            "positions will come from the model's own distance or stay rays")
    notes: list[str] = []
    started = time.time()
    for index, case in enumerate(cases):
        config.log_line(f"[{index + 1}/{len(cases)}] case {case.case_id} ({case.kind})")
        recorder.set_case(case)
        run = prepare_case(case, world=world)
        if config.inject_wrong_image and len(cases) > 1:
            other = cases[(index + 1) % len(cases)]
            recorder.set_wrong_frame(other.load_frame())
        for strategy in config.strategies:
            function = STRATEGY_FUNCTIONS.get(strategy)
            if function is None:
                notes.append(f"unknown strategy '{strategy}' skipped")
                continue
            try:
                function(run, wrapper, config, recorder)
            except Exception as exc:  # noqa: BLE001 - one broken strategy must not end the run
                notes.append(f"{case.case_id}/{strategy} raised {type(exc).__name__}: {exc}")
            recorder.flush()
        if config.inject_point or config.inject_waypoint:
            try:
                run_injections(run, config, recorder, executor=executor)
            except Exception as exc:  # noqa: BLE001 - injections are findings too
                notes.append(f"{case.case_id}/injections raised {type(exc).__name__}: {exc}")
            recorder.flush()
    wall_time = time.time() - started
    apply_image_audit(recorder.records)
    recorder.flush()
    summary = summarize(
        cases,
        recorder.records,
        config=config,
        backend=backend,
        world=world,
        wall_time_s=wall_time,
        notes=notes,
    )
    summary["calls_path"] = str(recorder.call_log)
    (config.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def _answer_projection(record: Mapping) -> dict:
    """The projection a record's answer produced (tools and pointed pixels alike)."""
    parsed = record.get("parsed") or {}
    projection = parsed.get("projection")
    if isinstance(projection, Mapping):
        return dict(projection)
    if any(key in parsed for key in ("position_flu", "position_source", "range_status")):
        return {
            "position_flu": parsed.get("position_flu"),
            "position_source": parsed.get("position_source"),
            "range_status": parsed.get("range_status"),
            "range_m": parsed.get("range_m"),
            "ray": parsed.get("ray"),
        }
    return {}


def apply_image_audit(records: Sequence[dict]) -> None:
    """Mark the calls that did not receive the step's own observation.

    "The model was handed the wrong frame" is invisible in a success rate (a
    scripted or lucky answer can still be right) and invisible in the tool result
    (the tool never sees the pixels the backend got) - so it is decided here,
    from the hashes the recorder took, and it overrides the verdict of the call:
    a right answer about another image is not evidence about this one.
    """
    for record in records:
        image = record.get("input_image") or {}
        if image.get("matches_expected") is not False:
            continue
        previous = record.get("failure") or {}
        record["diagnosis_only"] = True
        record["failure"] = {
            "category": FAILURE_WRONG_IMAGE,
            "detail": "frames_sent_are_not_the_observation"
                      f"(injected={bool(image.get('injected'))},"
                      f"expected={image.get('expected_source') or 'none'},"
                      f"was={previous.get('category')})",
        }
        record["outcome"] = OUTCOME_MISS


def _stat_row(records: Sequence[Mapping]) -> dict:
    """Aggregate one group of scored records (one strategy / one step)."""
    outcomes = [str(record.get("outcome") or "") for record in records]
    categories: dict[str, int] = {}
    details: dict[str, int] = {}
    for record in records:
        failure = record.get("failure") or {}
        category = str(failure.get("category") or FAILURE_NONE)
        categories[category] = categories.get(category, 0) + 1
        detail = str(failure.get("detail") or "")
        if detail:
            details[detail.split(":")[0]] = details.get(detail.split(":")[0], 0) + 1
    center_errors = [
        float((record.get("measure") or {}).get("center_error_px"))
        for record in records
        if (record.get("measure") or {}).get("center_error_px") is not None
    ]
    ious = [
        float((record.get("measure") or {}).get("iou"))
        for record in records
        if (record.get("measure") or {}).get("iou") is not None
    ]
    positions = [
        float((record.get("measure") or {}).get("position_error_m"))
        for record in records
        if (record.get("measure") or {}).get("position_error_m") is not None
    ]
    hits = outcomes.count(OUTCOME_HIT)
    misses = outcomes.count(OUTCOME_MISS)
    false_positives = outcomes.count(OUTCOME_FALSE_POSITIVE)
    errors = outcomes.count(OUTCOME_ERROR)
    scored = hits + misses + false_positives + errors
    return {
        "calls": len(records),
        "scored": scored,
        "hits": hits,
        "misses": misses,
        "abstain_correct": outcomes.count(OUTCOME_ABSTAIN),
        "false_positives": false_positives,
        "errors": errors,
        "not_run": outcomes.count(OUTCOME_NOT_RUN),
        "intermediate_steps": outcomes.count(OUTCOME_INTERMEDIATE),
        "success_rate": None if scored == 0 else round(hits / scored, 3),
        "mean_center_error_px": None if not center_errors else round(float(np.mean(center_errors)), 2),
        "median_center_error_px": None if not center_errors else round(float(np.median(center_errors)), 2),
        "mean_iou": None if not ious else round(float(np.mean(ious)), 3),
        "mean_position_error_m": None if not positions else round(float(np.mean(positions)), 3),
        "failure_categories": categories,
        "failure_details": details,
    }


def summarize(
    cases: Sequence[ObservationCase],
    records: Sequence[Mapping],
    *,
    config: BenchConfig,
    backend: Any,
    world: Any,
    wall_time_s: float,
    notes: Sequence[str] = (),
) -> dict:
    """Everything a reader needs: per-strategy scores, the failure split, audits."""
    scored_records = [record for record in records if not record.get("diagnosis_only")]
    final_records = [record for record in scored_records if record.get("final", True)]
    strategies: dict[str, dict] = {}
    for strategy in config.strategies:
        group = [record for record in final_records if record.get("strategy") == strategy]
        steps = {}
        for record in scored_records:
            if record.get("strategy") != strategy:
                continue
            step = str(record.get("step") or "")
            steps.setdefault(step, []).append(record)
        row = _stat_row(group)
        row["steps"] = {
            step: {key: value for key, value in _stat_row(step_records).items()
                   if key in ("calls", "hits", "misses", "abstain_correct", "not_run",
                              "intermediate_steps", "failure_categories")}
            for step, step_records in steps.items()
        }
        strategies[strategy] = row

    categories: dict[str, int] = {}
    details: dict[str, int] = {}
    tool_reasons: dict[str, int] = {}
    for record in final_records:
        failure = record.get("failure") or {}
        category = str(failure.get("category") or FAILURE_NONE)
        categories[category] = categories.get(category, 0) + 1
        detail = str(failure.get("detail") or "").split(":")[0]
        if detail:
            details[detail] = details.get(detail, 0) + 1
        tool = record.get("tool") or {}
        tool_reason = str(tool.get("reason") or tool.get("error") or "")
        if tool_reason:
            tool_reasons[tool_reason] = tool_reasons.get(tool_reason, 0) + 1

    audit = {
        "box_formats_used": {},
        "intrinsics_checked": 0,
        "intrinsics_mismatches": 0,
        "fov_deviations_m": [],
        "positions_with_known_range": 0,
        "positions_without_range": 0,
        "ray_only_answers": 0,
        "image_mismatches": 0,
        "injected_images": 0,
    }
    for record in records:
        parsed = record.get("parsed") or {}
        used = parsed.get("box_format_used") or (record.get("tool") or {}).get("box_format")
        if used:
            audit["box_formats_used"][str(used)] = audit["box_formats_used"].get(str(used), 0) + 1
        measure = record.get("measure") or {}
        if measure.get("intrinsics_match") is not None:
            audit["intrinsics_checked"] += 1
            if measure["intrinsics_match"] is False:
                audit["intrinsics_mismatches"] += 1
        projection = _answer_projection(record)
        deviation = projection.get("fov_deviation_m") or projection.get("fov_deviation_at_1m_m")
        if deviation is not None:
            audit["fov_deviations_m"].append(round(float(deviation), 3))
        if projection:
            has_position = projection.get("position_flu") is not None
            if has_position and projection.get("range_status") == "known":
                audit["positions_with_known_range"] += 1
            elif has_position:
                audit["positions_without_range"] += 1
            if projection.get("ray") is not None and not has_position:
                audit["ray_only_answers"] += 1
        if record.get("input_image", {}).get("matches_expected") is False:
            audit["image_mismatches"] += 1
        if record.get("input_image", {}).get("injected"):
            audit["injected_images"] += 1

    diagnosis_records = [record for record in records if record.get("diagnosis_only")]
    return {
        "format": RUN_FORMAT,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_time_s": round(float(wall_time_s), 1),
        "backend": {
            "name": getattr(backend, "name", type(backend).__name__),
            "oracle_replies": bool(config.oracle_replies) or isinstance(backend, OracleReplyVLM),
            "model_path": getattr(backend, "model_path", None),
            "device": getattr(backend, "device", None),
        },
        "detector": {
            "name": getattr(config.detector, "name", "null"),
            "available": bool(config.detector.available()) if config.detector else False,
            "note": getattr(config.detector, "note", ""),
        },
        "world": {
            "available": world is not None,
            "type": None if world is None else type(world).__name__,
            "depth": bool(world is not None and callable(getattr(world, "depth_at_pixel", None))),
            "render_size": list(config.render_size),
        },
        "out_dir": str(config.out_dir),
        "tolerances": config.tolerances.to_dict(),
        "strategies": strategies,
        "failure_categories": categories,
        "failure_details": details,
        "tool_reasons": tool_reasons,
        "audit": audit,
        "cases": [
            {
                "case_id": case.case_id,
                "kind": case.kind,
                "tag": case.tag,
                "image": case.image_path,
                "pose_flu": [round(float(v), 3) for v in case.pose_flu],
                "yaw_deg": round(float(case.yaw_deg), 2),
                "intrinsics": dict(case.intrinsics),
                "truth": case.truth.to_dict(),
                "render": dict(case.render),
                "note": case.note,
            }
            for case in cases
        ],
        "diagnosis_only": {
            "calls": len(diagnosis_records),
            "note": "diagnosis-only replacements; excluded from every score above",
            "records": [
                {
                    "call": record.get("call"),
                    "case": record.get("case"),
                    "strategy": record.get("strategy"),
                    "step": record.get("step"),
                    "outcome": record.get("outcome"),
                    "failure": record.get("failure"),
                    "measure": record.get("measure"),
                    "parsed": record.get("parsed"),
                }
                for record in diagnosis_records
            ],
        },
        "notes": list(notes),
    }


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_cell(item) for item in value) + "]"
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "_(no rows)_\n"
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def render_report(summary: Mapping, records: Sequence[Mapping]) -> str:
    """The markdown diagnosis: scores, the failure split, and the audits."""
    backend = summary.get("backend") or {}
    oracle = bool(backend.get("oracle_replies"))
    lines: list[str] = []
    lines.append("# Perception-failure bench: grounding strategies on a fixed observation set")
    lines.append("")
    lines.append(
        f"- backend: **{_cell(backend.get('name'))}**"
        + (f" (model `{_cell(backend.get('model_path'))}`)" if backend.get("model_path") else "")
        + f" | world: {_cell((summary.get('world') or {}).get('type'))}"
        + f" | depth: {_cell((summary.get('world') or {}).get('depth'))}"
    )
    lines.append(
        f"- detector slot: **{_cell((summary.get('detector') or {}).get('name'))}** "
        f"(available={_cell((summary.get('detector') or {}).get('available'))})"
        + f" | tolerances: {_cell(summary.get('tolerances'))}"
        + f" | wall time: {_cell(summary.get('wall_time_s'))} s"
    )
    if oracle:
        lines.append("")
        lines.append(
            "> **HARNESS SELF-TEST, NOT A MODEL MEASUREMENT.** The replies came from the "
            "answer key (oracle mock): they show that the bench, the tools and the "
            "coordinate contract agree when the localization is perfect."
        )
    lines.append("")
    lines.append(
        "> Every model call of this run is on disk: `calls.jsonl` (raw reply, parsed "
        "result, measures, failure category) and `inputs/` (the exact image that was "
        "sent). `diagnosis_only` rows are listed separately and are excluded from "
        "every score."
    )
    lines.append("")

    lines.append("## 1. Fixed observation set")
    lines.append("")
    lines.append(
        "The answer key is the analytic projection of the target, gated by the "
        "*rendered* depth map where the world has one (`analytic_projection+rendered_depth`): "
        "an occluder's splats cover more than the box they stand for, so a partly "
        "hidden target is measured against the half of it the picture really shows."
    )
    lines.append("")
    rows = []
    for case in summary.get("cases") or []:
        truth = case.get("truth") or {}
        render = case.get("render") or {}
        rows.append([
            case.get("case_id"), case.get("kind"),
            f"{_cell(truth.get('visible'))} ({_cell(truth.get('visible_fraction'))})",
            _cell(truth.get("visible_box")),
            _cell(truth.get("in_fov")),
            _cell(truth.get("distance_m")),
            _cell(truth.get("occluded_fraction")),
            _cell(render.get("depth_matches_target")),
            _cell((case.get("intrinsics") or {}).get("hfov_deg")),
            _cell(case.get("note")),
        ])
    lines.append(_table(
        ["case", "kind", "target visible (fraction)", "gt box (visible)", "in fov",
         "distance m", "occluded", "depth@target ok", "hfov deg", "note"],
        rows,
    ))
    lines.append("")

    lines.append("## 2. Per-strategy result")
    lines.append("")
    rows = []
    for strategy, row in (summary.get("strategies") or {}).items():
        rows.append([
            strategy, row.get("calls"), row.get("scored"), row.get("hits"), row.get("misses"),
            row.get("abstain_correct"), row.get("false_positives"), row.get("errors"),
            row.get("not_run"), row.get("success_rate"),
            row.get("mean_center_error_px"), row.get("mean_iou"), row.get("mean_position_error_m"),
        ])
    lines.append(_table(
        ["strategy", "calls", "scored", "hits", "misses", "abstain ok", "false pos", "errors",
         "not run", "success rate", "mean center err px", "mean IoU", "mean pos err m"],
        rows,
    ))
    lines.append("")
    for strategy, row in (summary.get("strategies") or {}).items():
        steps = row.get("steps") or {}
        if len(steps) <= 1:
            continue
        lines.append(f"Steps of `{strategy}`:")
        lines.append("")
        lines.append(_table(
            ["step", "calls", "hits", "misses", "abstain ok", "not run", "failures"],
            [[step, data.get("calls"), data.get("hits"), data.get("misses"),
              data.get("abstain_correct"), data.get("not_run"),
              _cell(data.get("failure_categories"))] for step, data in steps.items()],
        ))
        lines.append("")

    lines.append("## 3. Failure categories (all scored calls)")
    lines.append("")
    lines.append(
        "`no_candidate` is one row here, not the conclusion: a target the model did not "
        "report is `not_seen`, a reply nobody could parse is `output_format`, another "
        "frame is `wrong_image`, geometry against the contract is `projection_error`. "
        "`none` with `abstain_correct` is the honest \"I do not see it\" on an "
        "observation where the target really is invisible."
    )
    lines.append("")
    lines.append(_table(
        ["category", "calls"],
        sorted((summary.get("failure_categories") or {}).items(), key=lambda item: -item[1]),
    ))
    lines.append("")
    lines.append(_table(
        ["detail", "calls"],
        sorted((summary.get("failure_details") or {}).items(), key=lambda item: -item[1]),
    ))
    lines.append("")
    lines.append(
        "What the tools themselves answered in those calls (`reason` / `error` of the "
        "tool result): the row that used to be the whole diagnosis."
    )
    lines.append("")
    lines.append(_table(
        ["tool reason", "calls"],
        sorted((summary.get("tool_reasons") or {}).items(), key=lambda item: -item[1])
        or [["-", 0]],
    ))
    lines.append("")

    lines.append("## 4. Per-call detail")
    lines.append("")
    rows = []
    for record in records:
        if record.get("diagnosis_only"):
            continue
        measure = record.get("measure") or {}
        failure = record.get("failure") or {}
        tool = record.get("tool") or {}
        rows.append([
            record.get("call"), record.get("case"), record.get("kind"), record.get("strategy"),
            record.get("step"),
            "yes" if measure.get("target_visible") else "no",
            tool.get("reason") or tool.get("error") or "-",
            record.get("outcome"),
            _cell(measure.get("center_error_px")), _cell(measure.get("iou")),
            _cell(measure.get("position_error_m")), _cell(measure.get("position_source")),
            failure.get("category"), failure.get("detail"),
        ])
    lines.append(_table(
        ["call", "case", "kind", "strategy", "step", "target visible", "tool reason",
         "outcome", "center err px", "IoU", "pos err m", "position source", "failure", "detail"],
        rows,
    ))
    lines.append("")

    lines.append("## 5. Coordinate-contract audit")
    lines.append("")
    audit = summary.get("audit") or {}
    lines.append(
        f"- box formats actually read: {_cell(audit.get('box_formats_used'))} "
        "(canonical is `xyxy_norm`)"
    )
    lines.append(
        f"- intrinsics checks: {_cell(audit.get('intrinsics_checked'))} "
        f"with {_cell(audit.get('intrinsics_mismatches'))} mismatch(es) between the "
        "projection's FOV and the observation's own"
    )
    deviations = list(audit.get("fov_deviations_m") or [])
    lines.append(
        "- FOV deviation (own intrinsics vs the legacy "
        f"{LEGACY_DEFAULT_HFOV_DEG:.0f} deg default): "
        + (_cell(deviations) if deviations
           else "not measured on this run (no pointed pixel with a range)")
    )
    lines.append(
        f"- answers with a position: {_cell(audit.get('positions_with_known_range'))} with a known "
        f"range, {_cell(audit.get('positions_without_range'))} without one (a position with no "
        "range is a fabricated point); ray-only answers: "
        f"{_cell(audit.get('ray_only_answers'))} (the honest answer for an unknown distance)"
    )
    lines.append(
        f"- frames that did not match the step's expected image: "
        f"{_cell(audit.get('image_mismatches'))} "
        f"| injected (fault-injected) images: {_cell(audit.get('injected_images'))}"
    )
    lines.append(
        "- per-call detail: `calls.jsonl` carries the raw reply, the parsed result, the "
        "`range_check` (was the depth read at the answer's own pixel?) and the failure "
        "category of every call, including the audited intermediate steps."
    )
    lines.append("")

    diagnosis = summary.get("diagnosis_only") or {}
    if diagnosis.get("calls"):
        lines.append("## 6. Diagnosis-only replacements (NOT part of the score)")
        lines.append("")
        lines.append(diagnosis.get("note") or "")
        lines.append("")
        rows = []
        for record in diagnosis.get("records") or []:
            measure = record.get("measure") or {}
            failure = record.get("failure") or {}
            parsed = record.get("parsed") or {}
            detail = failure.get("detail")
            if record.get("strategy") == "inject_point":
                detail = f"position err {_cell(measure.get('position_error_m'))} m | {detail}"
            rows.append([
                record.get("call"), record.get("case"), record.get("strategy"),
                record.get("step"), record.get("outcome"),
                _cell(measure.get("center_error_px")), _cell(measure.get("position_error_m")),
                _cell(parsed.get("injected_point")), _cell(parsed.get("injected_waypoint")),
                failure.get("category"), detail,
            ])
        lines.append(_table(
            ["call", "case", "replacement", "step", "outcome", "center err px", "pos err m",
             "injected point", "injected waypoint", "failure", "detail"],
            rows,
        ))
        lines.append("")

    notes = summary.get("notes") or []
    if notes:
        lines.append("## Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("# mock (no weights): replies keyed by <case>|<strategy>|<step>")
    lines.append("python scripts/diagnose_grounding.py --backend mock --mock-replies replies.json \\")
    lines.append("    --out outputs/vlm_nav/diagnose_grounding/mock")
    lines.append("")
    lines.append("# real model (container vlm-nav, one A100)")
    lines.append("CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_grounding.py --backend hf \\")
    lines.append("    --model-path /models_ro/Qwen3-VL-8B --device cuda \\")
    lines.append("    --out outputs/vlm_nav/diagnose_grounding/hf_8b")
    lines.append("")
    lines.append("# same fixed set, next backend: the observations are reused, not re-rendered")
    lines.append("python scripts/diagnose_grounding.py --backend hf --model-path ... \\")
    lines.append("    --source manifest --observations outputs/vlm_nav/diagnose_grounding/hf_8b/observations \\")
    lines.append("    --out outputs/vlm_nav/diagnose_grounding/hf_8b_run2")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_size(text: str) -> tuple[int, int]:
    parts = str(text or "320x180").lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected WxH, got {text!r}")
    return int(parts[0]), int(parts[1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", type=str, default=None,
                        help="output directory (default outputs/vlm_nav/diagnose_grounding/run_<utc>)")
    parser.add_argument("--backend", type=str, default="mock",
                        help="mock | hf | openai (mock: --mock-replies, else the oracle self-test)")
    parser.add_argument("--model-path", type=str, default=None, help="hf backend model directory")
    parser.add_argument("--device", type=str, default="cpu", help="hf backend device (cuda)")
    parser.add_argument("--model", type=str, default=None, help="openai backend model name")
    parser.add_argument("--base-url", type=str, default=None, help="openai backend base url")
    parser.add_argument("--image-size", type=_parse_size, default=(640, 360),
                        help="resolution the model receives (WxH)")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--mock-replies", type=str, default=None,
                        help="JSON file mapping <case>|<strategy>|<step> to a reply")
    parser.add_argument("--source", type=str, default="scene",
                        choices=("scene", "manifest", "probe"),
                        help="where the fixed observation set comes from")
    parser.add_argument("--observations", type=str, default=None,
                        help="manifest source: the fixed set directory (or where to write it)")
    parser.add_argument("--ply", type=str, default=None,
                        help="scene source: render a real scene PLY instead of the synthetic one")
    parser.add_argument("--scene-json", type=str, default=None,
                        help="scenario.json of the scene (target/start/occluders)")
    parser.add_argument("--probe-json", type=str, default=None,
                        help="probe source: post_fix.json (frames + poses, no rendering)")
    parser.add_argument("--probe-frames", type=str, default=None, help="probe source: frames dir")
    parser.add_argument("--target-pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--target-radius", type=float, default=None)
    parser.add_argument("--start-pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--start-yaw-deg", type=float, default=None)
    parser.add_argument("--render-size", type=_parse_size, default=(320, 180))
    parser.add_argument("--hfov-deg", type=float, default=60.0)
    parser.add_argument("--instruction", type=str, default="",
                        help="the target phrase asked for (default: the scene target name)")
    parser.add_argument("--strategies", type=str, default=",".join(STRATEGIES))
    parser.add_argument("--detector", type=str, default="null",
                        help="null | oracle | module:factory (the detection slot)")
    parser.add_argument("--inject-point", type=str, default="",
                        help="DIAGNOSIS ONLY: 'gt' or 'u,v' - inject a correct image point")
    parser.add_argument("--inject-waypoint", type=str, default="",
                        help="DIAGNOSIS ONLY: 'gt' or 'x,y,z' - inject a correct waypoint")
    parser.add_argument("--inject-wrong-image", action="store_true",
                        help="DIAGNOSIS ONLY: send another observation's frame (fault injection)")
    parser.add_argument("--center-tolerance-px", type=float, default=DEFAULT_CENTER_TOLERANCE_PX)
    parser.add_argument("--iou-min", type=float, default=DEFAULT_IOU_MIN)
    parser.add_argument("--position-tolerance-m", type=float, default=DEFAULT_POSITION_TOLERANCE_M)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=str, default=None, help="report path (default <out>/diagnose_grounding.md)")
    return parser


def make_backend_from_args(args: argparse.Namespace):
    """(backend, oracle_replies) of the CLI arguments."""
    if args.backend == "mock":
        if args.mock_replies:
            replies = json.loads(Path(args.mock_replies).read_text())
            return KeyedMockVLM(replies), False
        return OracleReplyVLM(), True
    if args.backend == "hf":
        if not args.model_path:
            raise SystemExit("--backend hf needs --model-path")
        from vlm_nav.vlm import make_backend

        return make_backend(
            "hf",
            model_path=args.model_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            image_size=tuple(args.image_size),
        ), False
    if args.backend == "openai":
        from vlm_nav.vlm import make_backend

        return make_backend(
            "openai",
            model=args.model,
            base_url=args.base_url,
            max_new_tokens=args.max_new_tokens,
            image_size=tuple(args.image_size),
        ), False
    raise SystemExit(f"unknown --backend {args.backend!r} (mock | hf | openai)")


def load_scene_from_args(args: argparse.Namespace, out_dir: Path) -> SceneSpec:
    """The scene of a ``--source scene`` run: the synthetic M2 scene or a PLY.

    Without ``--target-pos`` / ``--start-pos`` the synthetic path uses the M2
    scenario's own geometry (the target the real M2 episodes fly to, the occluder
    slab between the loss pose and the target) - authoring a scene at the
    *default* SceneSpec position instead would put the start pose on top of the
    target and produce an observation set in which nothing is ever visible.
    """
    if args.scene_json:
        scene = scene_from_scenario_json(Path(args.scene_json))
        if args.target_pos is not None:
            scene.target_pos = tuple(float(v) for v in args.target_pos)
        if args.target_radius is not None:
            scene.target_radius = float(args.target_radius)
        if args.start_pos is not None:
            scene.start_pos = tuple(float(v) for v in args.start_pos)
        if args.start_yaw_deg is not None:
            scene.start_yaw_deg = float(args.start_yaw_deg)
        scene.hfov_deg = float(args.hfov_deg)
        if args.ply:
            scene.ply_path = str(Path(args.ply))
            scene.source = str(args.ply)
        return scene

    m2 = load_m2_helpers()
    overrides: dict[str, Any] = {"hfov_deg": float(args.hfov_deg)}
    if args.target_pos is not None:
        overrides["target_pos"] = tuple(float(v) for v in args.target_pos)
    if args.target_radius is not None:
        overrides["target_radius"] = float(args.target_radius)
    if args.start_pos is not None:
        overrides["start_pos"] = tuple(float(v) for v in args.start_pos)
    if args.start_yaw_deg is not None:
        overrides["start_yaw_deg"] = float(args.start_yaw_deg)
    scenario = m2.Scenario(**overrides)

    if args.ply:
        # a real scene: its geometry is the caller's, and its occluders are
        # unknown - so none are assumed (a phantom wall would invent occlusion
        # that the scene does not have)
        return SceneSpec(
            ply_path=str(Path(args.ply)),
            target_pos=tuple(float(v) for v in scenario.target_pos),
            target_radius=float(scenario.target_radius),
            start_pos=tuple(float(v) for v in scenario.start_pos),
            start_yaw_deg=float(scenario.start_yaw_deg),
            hfov_deg=float(scenario.hfov_deg),
            occluders=[],
            loss_pose=None,
            target_name=str(scenario.target_name),
            source=str(args.ply),
        )

    scene_dir = Path(out_dir) / "observations"
    scene_dir.mkdir(parents=True, exist_ok=True)
    ply, origin = m2.write_scenario_scene(scene_dir / "scenario_scene.ply", scenario, seed=args.seed)
    aligned = m2.align_scenario(scenario, origin)
    return SceneSpec(
        ply_path=str(ply),
        target_pos=tuple(float(v) for v in aligned.target_pos),
        target_radius=float(aligned.target_radius),
        start_pos=tuple(float(v) for v in aligned.start_pos),
        start_yaw_deg=float(aligned.start_yaw_deg),
        hfov_deg=float(aligned.hfov_deg),
        occluders=occluders_from(aligned.occluders),
        loss_pose=tuple(float(v) for v in aligned.loss_pose(aligned.start_pos)),
        origin_flu=tuple(float(v) for v in origin),
        target_name=str(aligned.target_name),
        source="synthetic_m2",
    )


def build_fixed_set(
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[list[ObservationCase], SceneSpec | None, list[str]]:
    """The fixed observation set of this run (rendered once, or loaded as-is)."""
    if args.source == "manifest":
        directory = Path(args.observations) if args.observations else (out_dir / "observations")
        manifest_path = directory / "manifest.json" if directory.is_dir() else directory
        cases, manifest = load_observations(manifest_path)
        scene = scene_from_dict(manifest.get("scene"), manifest_path.parent)
        notes = list(manifest.get("notes") or [])
        if args.instruction:
            for case in cases:
                case.instruction = args.instruction
        return cases, scene, notes
    if args.source == "probe":
        probe_json = Path(args.probe_json or
                          "outputs/vlm_nav/eval/m2_hf_8b_v2/probe/post_fix.json")
        scene_json = Path(args.scene_json or
                          "outputs/vlm_nav/eval/m2_hf_8b_v2/yaw0/scenario.json")
        cases, scene, notes = build_cases_from_probe(
            probe_json, scene_json, frames_dir=args.probe_frames and Path(args.probe_frames),
            instruction=args.instruction,
        )
        return cases, scene, notes
    scene = load_scene_from_args(args, out_dir)
    instruction = args.instruction or f"the {scene.target_name}"
    if args.ply and not Path(scene.ply_path).is_file():
        raise SystemExit(f"--ply {scene.ply_path} does not exist")
    cases, notes = build_cases_from_scene(
        scene, out_dir, render_size=tuple(args.render_size), instruction=instruction,
        cases_dir=out_dir / "observations",
    )
    if args.ply and not scene.occluders:
        notes.append(
            "occlusion is not modelled for this scene (no --scene-json occluders): "
            "the 'partial' situation needs a scene description that names them"
        )
    save_observations(
        cases, out_dir / "observations" / "manifest.json", scene=scene.to_dict(),
        source=f"scene:{scene.source}", notes=notes,
    )
    return cases, scene, notes


def scene_from_dict(data: Mapping | None, root: Path | None = None) -> SceneSpec | None:
    """SceneSpec of a manifest's ``scene`` block (the world of the fixed set)."""
    if not isinstance(data, Mapping) or not data:
        return None
    ply = str(data.get("ply") or "")
    if ply and root is not None and not Path(ply).is_absolute() and not Path(ply).is_file():
        candidate = Path(root) / ply
        if candidate.is_file():
            ply = str(candidate)
    loss = _floats(data.get("loss_pose"), 3)
    return SceneSpec(
        ply_path=ply,
        target_pos=tuple(_floats(data.get("target_pos"), 3) or (0.0, 0.0, 0.0)),
        target_radius=float(data.get("target_radius") or 0.2),
        start_pos=tuple(_floats(data.get("start_pos"), 3) or (0.0, 0.0, 0.0)),
        start_yaw_deg=float(data.get("start_yaw_deg") or 0.0),
        hfov_deg=float(data.get("hfov_deg") or 60.0),
        occluders=occluders_from(data.get("occluders")),
        loss_pose=None if loss is None else tuple(float(v) for v in loss),
        origin_flu=tuple(_floats(data.get("origin_flu"), 3) or (0.0, 0.0, 0.0)),
        target_name=str(data.get("target_name") or "target"),
        source=str(data.get("source") or ""),
    )


def print_summary(summary: Mapping, out_dir: Path) -> None:
    """Compact console table (the markdown report has the full detail)."""
    print("")
    print(f"backend={((summary.get('backend') or {}).get('name'))} out={out_dir}")
    header = (f"{'strategy':<20} {'calls':>5} {'scored':>6} {'hits':>5} {'miss':>5} "
              f"{'abstain':>7} {'false':>5} {'err':>4} {'n/a':>4} {'rate':>6} "
              f"{'ctr px':>7} {'pos m':>6}")
    print(header)
    for strategy, row in (summary.get("strategies") or {}).items():
        rate = row.get("success_rate")
        center = row.get("mean_center_error_px")
        position = row.get("mean_position_error_m")
        print(f"{strategy:<20} {row['calls']:>5} {row['scored']:>6} {row['hits']:>5} "
              f"{row['misses']:>5} {row['abstain_correct']:>7} {row['false_positives']:>5} "
              f"{row['errors']:>4} {row['not_run']:>4} "
              f"{'-' if rate is None else format(rate, '.2f'):>6} "
              f"{'-' if center is None else format(center, '.1f'):>7} "
              f"{'-' if position is None else format(position, '.2f'):>6}")
    print("")
    print(f"failure categories: {summary.get('failure_categories')}")
    diagnosis = summary.get("diagnosis_only") or {}
    if diagnosis.get("calls"):
        print(f"diagnosis-only calls (not scored): {diagnosis['calls']}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out or
                   time.strftime("outputs/vlm_nav/diagnose_grounding/run_%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cases, scene, notes = build_fixed_set(args, out_dir)
    if not cases:
        raise SystemExit("no observations in the fixed set; nothing to diagnose")
    if args.max_cases:
        cases = cases[: max(1, int(args.max_cases))]
    kinds = {case.kind for case in cases}
    missing = [kind for kind in CASE_KINDS if kind not in kinds]
    if missing:
        print(f"warning: the fixed set has no '{', '.join(missing)}' observation(s); "
              "those situations are not covered by this run")
    backend, oracle = make_backend_from_args(args)
    detector = resolve_detector(args.detector)

    def log(text: str) -> None:
        print(text, flush=True)

    config = BenchConfig(
        out_dir=out_dir,
        strategies=tuple(part.strip() for part in args.strategies.split(",") if part.strip()),
        tolerances=Tolerances(
            center_px=float(args.center_tolerance_px),
            iou_min=float(args.iou_min),
            position_m=float(args.position_tolerance_m),
        ),
        detector=detector,
        inject_point=args.inject_point,
        inject_waypoint=args.inject_waypoint,
        inject_wrong_image=bool(args.inject_wrong_image),
        oracle_replies=oracle,
        render_size=tuple(args.render_size),
        hfov_deg=float(args.hfov_deg),
        max_cases=args.max_cases,
        log=log,
    )
    summary = run_bench(cases, backend, config, scene=scene)
    summary["notes"] = list(summary.get("notes") or []) + list(notes)
    closed = getattr(backend, "close", None)
    if callable(closed):
        closed()
    report = render_report(summary, _read_calls(Path(summary["calls_path"])))
    report_path = Path(args.report or (out_dir / "diagnose_grounding.md"))
    report_path.write_text(report)
    summary["report_path"] = str(report_path)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (out_dir / "run_config.json").write_text(
        json.dumps({key: value for key, value in vars(args).items()}, indent=2, default=str) + "\n"
    )
    print_summary(summary, out_dir)
    print(f"report: {report_path}")
    print(f"calls:  {summary['calls_path']}")
    return 0


def _read_calls(path: Path) -> list[dict]:
    records = []
    try:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    except FileNotFoundError:
        return []
    return records


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
