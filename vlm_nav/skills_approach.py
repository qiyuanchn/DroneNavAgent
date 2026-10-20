"""The approach skill: fly the drone to a candidate target (goal.md 3/4).

``approach_candidate`` is the *execution* half of navigation: the model decides
**which** candidate to approach (and when to give up), this module decides how
the drone gets there. One call plans and flies at most one segment:

1. **localize** - read the candidate record ``ground_target`` wrote (canonical
   ``[x1, y1, x2, y2]`` box, its world position when the range is known, else the
   ray of the box centre) and turn it into a *target point*: a position, or -
   when only a direction is known - a bounded step along that ray. Nothing is
   invented: a candidate with neither a position nor a ray is reported as
   unlocatable instead of being flown to at a guessed range.
2. **standoff waypoint** - the target point is not a parking spot, so the leg
   ends ``standoff_m`` short of it (``world._apply_standoff`` when the world
   declares its own standoff, the same rule otherwise).
3. **segment execution / re-localization** - the leg flies through the world's
   waypoint action and, after it, the caller observes again and this module
   re-measures the candidate from the new pose (``relocalize``): the target is
   projected into the current frame and the current depth at that pixel is read
   back, so the next segment steers on a fresh measurement instead of on the
   range of a frame taken several metres ago.

Everything is duck-typed (``getattr``) so the skill runs against ``SplatWorldCPU``
/ ``EnvWorld`` / ``SuperPlannerWorld`` and against the fake worlds of the tests,
and every world hook (``depth_at_pixel``, ``_apply_standoff``, intrinsics) is
optional: the skill degrades to the information that really exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .observations import (
    CANDIDATE_STORE_KEY,
    observation_store,
    resolve_intrinsics,
)
from .world import body_delta_to_world, pixel_to_body_offset

#: default parking distance from the candidate: close enough to inspect it,
#: clear of the surface it sits on
APPROACH_STANDOFF_M = 1.2
#: one approach segment never exceeds the waypoint clamp
APPROACH_MAX_LEG_M = 8.0
#: a candidate whose range is unknown is approached along its ray, one bounded
#: step per segment, until a depth measurement replaces the ray with a point
RAY_STEP_M = 3.0
#: the target point is "reached" within this distance of the standoff shell
ARRIVE_RADIUS_M = 0.35
#: a depth read below this is not a distance (same gate as the grounder)
MIN_RANGE_M = 0.1
#: a re-localized point farther than this from the previous estimate is a
#: different surface (a wall behind the target), not a refinement of the same one
MAX_RELOCALIZE_JUMP_M = 3.0
#: shortest ray-following step (a shorter leg is not worth a turn)
MIN_RAY_STEP_M = 0.5
#: A candidate above/below the level frame is measured through a pitched camera
#: centred on its elevation (the v8 ``target_out_of_view`` geometry). Beyond
#: this elevation the target is nearly overhead and the read is not meaningful.
MAX_MEASUREMENT_ELEVATION_DEG = 70.0

POSITION_KEYS = ("position_flu", "position", "world_flu", "pos")
RAY_KEYS = ("ray",)
BOX_KEYS = ("box", "bbox", "region")


# ---------------------------------------------------------------------------
# the candidate record and what it says about where the target is
# ---------------------------------------------------------------------------


def candidate_record(ctx: Any, candidate_id: Any) -> dict | None:
    """The stored record of ``cand_N`` (perception's canonical store first)."""
    extra = getattr(ctx, "extra", None)
    store = extra.get(CANDIDATE_STORE_KEY) if isinstance(extra, Mapping) else None
    if not isinstance(store, Mapping):
        return None
    for key in (str(candidate_id), candidate_id):
        record = store.get(key)
        if isinstance(record, Mapping):
            return dict(record)
    return None


def _as_point(value: Any) -> np.ndarray | None:
    """Finite 3-vector or None (never a fabricated point)."""
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)[:3]
    except (TypeError, ValueError):
        return None
    if arr.size != 3 or not np.isfinite(arr).all():
        return None
    return arr


def _as_ray(value: Any) -> dict | None:
    """A usable direction (+ its origin): ``{origin_flu, bearing_deg, elevation_deg}``."""
    if not isinstance(value, Mapping):
        return None
    origin = _as_point(value.get("origin_flu"))
    try:
        bearing = float(value.get("bearing_deg"))
    except (TypeError, ValueError):
        return None
    if origin is None or not math.isfinite(bearing):
        return None
    try:
        elevation = float(value.get("elevation_deg", 0.0) or 0.0)
    except (TypeError, ValueError):
        elevation = 0.0
    return {
        "origin_flu": origin,
        "bearing_deg": bearing,
        "elevation_deg": elevation if math.isfinite(elevation) else 0.0,
    }


def box_center(box: Any) -> tuple[float, float] | None:
    """Centre pixel of a canonical ``[x1, y1, x2, y2]`` box (normalized 0-1000)."""
    if box is None or isinstance(box, str):
        return None
    try:
        arr = np.asarray(box, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size != 4 or not np.isfinite(arr).all():
        return None
    x1, y1, x2, y2 = (float(v) for v in arr)
    if x2 < x1 or y2 < y1:  # a [u, v, w, h] box the reader did not normalize
        return None
    return (
        float(min(max(0.5 * (x1 + x2), 0.0), 1000.0)),
        float(min(max(0.5 * (y1 + y2), 0.0), 1000.0)),
    )


@dataclass
class CandidateTarget:
    """Where an approach skill call believes the candidate is.

    ``position`` is set when the range is known (depth or the model's own
    estimate), ``ray`` when only the direction is; ``source`` names which of the
    two (or ``unknown``). ``box`` is the candidate's canonical corner box, kept
    so a later frame can re-measure it.
    """

    candidate_id: str
    observation_id: str = ""
    position: np.ndarray | None = None
    ray: dict | None = None
    box: tuple[float, float, float, float] | None = None
    source: str = "unknown"
    range_m: float | None = None

    @property
    def located(self) -> bool:
        return self.position is not None

    @property
    def direction_known(self) -> bool:
        return self.position is not None or self.ray is not None

    @property
    def pixel(self) -> tuple[float, float] | None:
        return box_center(self.box)

    def to_data(self) -> dict:
        data: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "position_source": self.source,
        }
        if self.observation_id:
            data["source_observation_id"] = self.observation_id
        if self.position is not None:
            data["target_position"] = [round(float(v), 3) for v in self.position]
        if self.range_m is not None:
            data["range_m"] = round(float(self.range_m), 2)
        if self.box is not None:
            data["box"] = [round(float(v), 1) for v in self.box]
            data["box_format"] = "xyxy"
        if self.ray is not None:
            data["ray"] = {
                "bearing_deg": round(float(self.ray["bearing_deg"]), 1),
                "elevation_deg": round(float(self.ray["elevation_deg"]), 1),
                "origin_flu": [round(float(v), 3) for v in self.ray["origin_flu"]],
            }
        return data


def candidate_target(record: Mapping | None, candidate_id: str = "") -> CandidateTarget:
    """Turn a candidate record into the target the approach skill flies to."""
    target = CandidateTarget(candidate_id=candidate_id or str((record or {}).get("id", "")))
    if not isinstance(record, Mapping):
        return target
    target.observation_id = str(record.get("observation_id") or "")
    for key in POSITION_KEYS:
        point = _as_point(record.get(key))
        if point is not None:
            target.position = point
            break
    for key in RAY_KEYS:
        ray = _as_ray(record.get(key))
        if ray is not None:
            target.ray = ray
            break
    for key in BOX_KEYS:
        box = record.get(key)
        if box_center(box) is not None:
            target.box = tuple(float(v) for v in np.asarray(box).reshape(-1)[:4])
            break
    target.source = str(record.get("position_source") or ("position" if target.located else "unknown"))
    try:
        range_m = float(record.get("range_m"))
    except (TypeError, ValueError):
        range_m = None
    target.range_m = range_m if range_m is not None and math.isfinite(range_m) else None
    if target.range_m is None and target.located:
        origin = target.ray["origin_flu"] if target.ray else None
        if origin is not None:
            target.range_m = float(np.linalg.norm(target.position - origin))
    return target


# ---------------------------------------------------------------------------
# standoff geometry
# ---------------------------------------------------------------------------


def body_offset(delta_world: Any, yaw_rad: float) -> np.ndarray:
    """World-frame delta -> yaw-only body FLU offset (+x forward, +y left)."""
    dx, dy, dz = (float(v) for v in np.asarray(delta_world, dtype=np.float64).reshape(3))
    cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.asarray(
        [
            cos_yaw * dx + sin_yaw * dy,
            -sin_yaw * dx + cos_yaw * dy,
            dz,
        ],
        dtype=np.float64,
    )


def _shrink(offset: np.ndarray, standoff_m: float) -> np.ndarray:
    """The world's standoff rule: stop ``standoff_m`` short along the offset."""
    total = float(np.linalg.norm(offset))
    if total <= standoff_m or total <= 0.0:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(offset, dtype=np.float64) * ((total - standoff_m) / total)


def standoff_offset(world: Any, offset: Any, standoff_m: float | None = None) -> np.ndarray:
    """Shrink a body-frame offset so the leg ends at the standoff shell.

    ``world._apply_standoff`` is the shared implementation every world backend
    already carries and is used when the caller does not override the standoff
    (so a world that declares ``standoff_m`` keeps it); an explicit
    ``standoff_m`` argument re-applies the same rule with that value.
    """
    offset = np.asarray(offset, dtype=np.float64).reshape(3)
    if standoff_m is None:
        method = getattr(world, "_apply_standoff", None)
        if callable(method):
            try:
                shrunk = np.asarray(method(*[float(v) for v in offset]), dtype=np.float64).reshape(3)
                if np.isfinite(shrunk).all():
                    return shrunk
            except Exception:  # noqa: BLE001 - a duck-typed world may not fit
                pass
        standoff_m = APPROACH_STANDOFF_M
    return _shrink(offset, max(0.0, float(standoff_m)))


def world_standoff_m(world: Any, standoff_m: float | None = None) -> float:
    """The standoff the next leg will use (the world's own, or the caller's)."""
    if standoff_m is not None:
        return max(0.0, float(standoff_m))
    value = getattr(world, "standoff_m", None)
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return APPROACH_STANDOFF_M


def ray_point(origin: Any, ray: Mapping, distance_m: float) -> np.ndarray:
    """Point ``distance_m`` along a world-frame ray (bearing/elevation, degrees)."""
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    bearing = math.radians(float(ray.get("bearing_deg", 0.0)))
    elevation = math.radians(float(ray.get("elevation_deg", 0.0)))
    horizontal = math.cos(elevation) * float(distance_m)
    return origin + np.asarray(
        [
            horizontal * math.cos(bearing),
            horizontal * math.sin(bearing),
            math.sin(elevation) * float(distance_m),
        ],
        dtype=np.float64,
    )


def standoff_point(
    world: Any, position_flu: Any, target_point: Any, standoff_m: float | None = None
) -> np.ndarray:
    """The leg's end point: ``standoff_m`` short of ``target_point``."""
    here = np.asarray(position_flu, dtype=np.float64).reshape(3)
    target = np.asarray(target_point, dtype=np.float64).reshape(3)
    yaw = _yaw(world)
    offset = body_offset(target - here, yaw)
    shrunk = standoff_offset(world, offset, standoff_m)
    return here + np.asarray(body_delta_to_world(*[float(v) for v in shrunk], yaw), dtype=np.float64)


# ---------------------------------------------------------------------------
# planning one segment
# ---------------------------------------------------------------------------


@dataclass
class SegmentPlan:
    """What the next approach segment does, in the terms the tool layer needs."""

    waypoint: list[float] | None = None
    target_position: list[float] | None = None
    remaining_m: float | None = None
    standoff_m: float = APPROACH_STANDOFF_M
    source: str = "unknown"
    arrived: bool = False
    blocked_reason: str | None = None
    hint: str = ""
    target: CandidateTarget | None = field(default=None, repr=False)

    def to_data(self) -> dict:
        data: dict[str, Any] = {
            "standoff_m": round(float(self.standoff_m), 2),
            "position_source": self.source,
        }
        if self.waypoint is not None:
            data["waypoint"] = list(self.waypoint)
        if self.target_position is not None:
            data["target_position"] = list(self.target_position)
        if self.remaining_m is not None:
            data["remaining_m"] = round(float(self.remaining_m), 2)
        if self.blocked_reason:
            data["blocked_reason"] = self.blocked_reason
        return data


def plan_segment(
    ctx: Any,
    target: CandidateTarget,
    standoff_m: float | None = None,
    max_leg_m: float = APPROACH_MAX_LEG_M,
) -> SegmentPlan:
    """Plan one approach segment (never flies anything).

    A located candidate gets a standoff waypoint (``arrived`` once the drone is
    on that shell). A candidate known only as a ray gets a bounded step along
    it. A candidate with neither is ``blocked`` with the reason - the caller
    reports it and asks the model to re-ground, instead of flying to a guess.
    """
    world = getattr(ctx, "world", None)
    here = _position(world)
    standoff = world_standoff_m(world, standoff_m)
    plan = SegmentPlan(standoff_m=standoff, source=target.source, target=target)

    if target.position is not None:
        distance = float(np.linalg.norm(target.position - here))
        plan.target_position = _pos_list(target.position)
        plan.remaining_m = max(0.0, distance - standoff)
        if distance <= standoff + ARRIVE_RADIUS_M:
            plan.arrived = True
            plan.waypoint = _pos_list(here)
            plan.hint = (
                f"already at the approach standoff ({distance:.2f} m from the target, "
                f"standoff {standoff:.2f} m): nothing to fly - verify the candidate "
                "instead of repeating this call"
            )
            return plan
        waypoint = standoff_point(world, here, target.position, standoff)
        # a long approach is flown as several clamped legs, never one hop
        delta = waypoint - here
        length = float(np.linalg.norm(delta))
        if length > max_leg_m > 0:
            waypoint = here + delta * (max_leg_m / length)
        plan.waypoint = _pos_list(waypoint)
        return plan

    if target.ray is not None:
        step = max(MIN_RAY_STEP_M, min(float(max_leg_m), RAY_STEP_M))
        origin = target.ray["origin_flu"]
        point = ray_point(origin, target.ray, step)
        plan.waypoint = _pos_list(point)
        plan.source = "ray"
        plan.remaining_m = None  # the range is unknown; a step is still a step
        plan.hint = (
            "the candidate has no range yet: flying along its ray; the next "
            "observation re-measures it"
        )
        return plan

    plan.blocked_reason = "candidate_has_no_position"
    plan.hint = (
        "that candidate carries neither a position nor a ray: observe forward and "
        "call ground_target again from the current view, then approach the new cand_N"
    )
    return plan


# ---------------------------------------------------------------------------
# re-localization: measure the candidate again from the current frame
# ---------------------------------------------------------------------------


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


def _pos_list(pos: Any) -> list[float]:
    return [round(float(v), 3) for v in np.asarray(pos, dtype=np.float64).reshape(3)]


def world_to_pixel(
    point: Any,
    position_flu: Any,
    yaw_rad: float,
    intrinsics: Mapping | None,
    pitch_rad: float = 0.0,
) -> tuple[float, float] | None:
    """Project a world point into a camera (normalized [0,1000] pixel) or None.

    None means "that point is behind the camera or outside the frame" (or the
    intrinsics are unknown) - the honest answer, never a clamped edge pixel.
    ``pitch_rad`` is a body pitch, **nose-down positive**: a level camera cannot
    hold a candidate whose elevation exceeds the vertical half-FOV, while a
    camera pitched by ``-elevation`` centres it (the v8 position-check bug).
    """
    hfov = _hfov(intrinsics)
    width, height = _size(intrinsics)
    if hfov is None or width <= 0 or height <= 0:
        return None
    delta = np.asarray(point, dtype=np.float64).reshape(3) - np.asarray(
        position_flu, dtype=np.float64
    ).reshape(3)
    offset = body_offset(delta, yaw_rad)
    forward, left, up = (float(v) for v in offset)
    if forward <= MIN_RANGE_M:
        return None
    cos_p, sin_p = math.cos(float(pitch_rad)), math.sin(float(pitch_rad))
    cam_forward = forward * cos_p - up * sin_p
    cam_up = forward * sin_p + up * cos_p
    if cam_forward <= MIN_RANGE_M:
        return None
    fx = width / (2.0 * math.tan(math.radians(hfov) / 2.0))
    vfov = 2.0 * math.atan(math.tan(math.radians(hfov) / 2.0) * height / width)
    fy = height / (2.0 * math.tan(vfov / 2.0))
    u = width / 2.0 - left / cam_forward * fx
    v = height / 2.0 - cam_up / cam_forward * fy
    if not (0.0 <= u <= width and 0.0 <= v <= height):
        return None
    return (u / width * 1000.0, v / height * 1000.0)


def elevation_deg(point: Any, position_flu: Any) -> float:
    """Elevation of ``point`` above the level camera axis, in degrees."""
    delta = np.asarray(point, dtype=np.float64).reshape(3) - np.asarray(
        position_flu, dtype=np.float64
    ).reshape(3)
    horizontal = float(math.hypot(float(delta[0]), float(delta[1])))
    return math.degrees(math.atan2(float(delta[2]), max(horizontal, 1e-9)))


def _pitched_camera(
    point: Any, position_flu: Any, yaw_rad: float, intrinsics: Mapping | None,
    world: Any,
) -> tuple[float, tuple[float, float], float] | None:
    """``(pitch_rad, pixel, elevation_deg)`` for a pitched read, or None.

    Used only after the level projection refused: the camera is pitched
    ``-elevation`` (nose-up for a target above the drone) so the candidate's
    camera axis holds it. A world without pitched depth support (or a target
    behind/beside the frame, or nearly overhead) gets None - the caller reports
    the honest ``target_out_of_view`` instead of inventing a measurement.
    """
    if not getattr(world, "supports_pitched_depth", False):
        return None
    elevation = elevation_deg(point, position_flu)
    if abs(elevation) > MAX_MEASUREMENT_ELEVATION_DEG:
        return None
    pitch_rad = -math.radians(elevation)
    pixel = world_to_pixel(point, position_flu, yaw_rad, intrinsics, pitch_rad=pitch_rad)
    if pixel is None:
        return None
    return pitch_rad, pixel, elevation


def direction_to_pixel(
    bearing_deg: float, elevation_deg: float, yaw_rad: float, intrinsics: Mapping | None
) -> tuple[float, float] | None:
    """Project a world-frame direction (degrees) into the current camera."""
    hfov = _hfov(intrinsics)
    width, height = _size(intrinsics)
    if hfov is None or width <= 0 or height <= 0:
        return None
    relative = math.radians(bearing_deg) - yaw_rad
    elevation = math.radians(elevation_deg)
    forward = math.cos(relative) * math.cos(elevation)
    left = math.sin(relative) * math.cos(elevation)
    up = math.sin(elevation)
    if forward <= MIN_RANGE_M:
        return None
    fx = width / (2.0 * math.tan(math.radians(hfov) / 2.0))
    vfov = 2.0 * math.atan(math.tan(math.radians(hfov) / 2.0) * height / width)
    fy = height / (2.0 * math.tan(vfov / 2.0))
    u = width / 2.0 - left / forward * fx
    v = height / 2.0 - up / forward * fy
    if not (0.0 <= u <= width and 0.0 <= v <= height):
        return None
    return (u / width * 1000.0, v / height * 1000.0)


def _hfov(intrinsics: Mapping | None) -> float | None:
    if not isinstance(intrinsics, Mapping):
        return None
    try:
        value = float(intrinsics.get("hfov_deg"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 0.0 < value < 180.0 else None


def _size(intrinsics: Mapping | None) -> tuple[int, int]:
    if not isinstance(intrinsics, Mapping):
        return 0, 0
    try:
        return int(intrinsics.get("width") or 0), int(intrinsics.get("height") or 0)
    except (TypeError, ValueError):
        return 0, 0


def current_intrinsics(ctx: Any, observation_id: str = "") -> dict:
    """Intrinsics of the frame the drone is looking through right now."""
    store = observation_store(ctx, create=False)
    if store is not None and observation_id:
        record = store.get(observation_id)
        if record is not None and record.intrinsics:
            declared = dict(record.intrinsics)
            if _hfov(declared) is not None and _size(declared) != (0, 0):
                return declared
    return resolve_intrinsics(ctx)


def _update_store(ctx: Any, target: CandidateTarget, measurement: dict) -> None:
    """Write a fresh measurement back into the candidate store (best effort)."""
    extra = getattr(ctx, "extra", None)
    if not isinstance(extra, dict):
        return
    store = extra.get(CANDIDATE_STORE_KEY)
    if not isinstance(store, dict):
        return
    for key in (target.candidate_id, str(target.candidate_id)):
        record = store.get(key)
        if not isinstance(record, dict):
            continue
        if measurement.get("position_flu") is not None:
            record["position_flu"] = measurement["position_flu"]
            record["position_source"] = measurement.get("position_source", "depth")
            record["range_m"] = measurement.get("range_m")
        record["relocalized_from"] = measurement.get("observation_id")
        record["relocalized_count"] = int(record.get("relocalized_count", 0)) + 1
        return


def relocalize(
    ctx: Any,
    target: CandidateTarget,
    observation_id: str = "",
    *,
    allow_pitched: bool = False,
) -> dict:
    """Re-measure the target from the current pose; returns the measurement.

    The candidate is projected into the current frame (from its last known point,
    or along its ray when only the direction is known) and the world's depth at
    that pixel is read back - a fresh range for the next segment instead of the
    range of a frame taken several metres ago. A measurement that would jump the
    target farther than ``MAX_RELOCALIZE_JUMP_M`` (a wall behind it) or that is
    not available at all is reported, not applied: ``relocalized`` stays False
    and ``reason`` says why.

    With ``allow_pitched`` a level projection that lands outside the frame is
    retried through a camera pitched at the candidate's elevation (nose-up for a
    target above the drone) when the world serves pitched depth. A candidate
    whose centre is 0.9 m above a 1.2 m-away camera (36.6 deg, the v8 seed7
    geometry) therefore gets measured instead of ``target_out_of_view``; the
    returned ``pitch_down_deg``/``elevation_deg``/``anchor`` name the read.
    """
    world = getattr(ctx, "world", None)
    here = _position(world)
    yaw = _yaw(world)
    intrinsics = current_intrinsics(ctx, observation_id)
    result: dict[str, Any] = {
        "relocalized": False,
        "candidate_id": target.candidate_id,
        "observation_id": observation_id or "",
        "reason": "",
    }
    if not target.direction_known:
        result["reason"] = "no_target_direction"
        return result
    if _hfov(intrinsics) is None or _size(intrinsics) == (0, 0):
        # without the frame's field of view the pixel of a direction is unknown:
        # the honest answer is "cannot re-measure", never a biased projection
        result["reason"] = "intrinsics_unknown"
        return result

    pixel = None
    pitch_rad = 0.0
    anchor = "camera_axis"
    if target.position is not None:
        result["elevation_deg"] = round(elevation_deg(target.position, here), 1)
        pixel = world_to_pixel(target.position, here, yaw, intrinsics)
        if pixel is None and allow_pitched:
            pitched = _pitched_camera(target.position, here, yaw, intrinsics, world)
            if pitched is not None:
                pitch_rad, pixel, elevation = pitched
                anchor = "pitched_camera"
                result.update(
                    anchor=anchor,
                    pitch_down_deg=round(math.degrees(pitch_rad), 1),
                    elevation_deg=round(float(elevation), 1),
                )
        result["pixel"] = None if pixel is None else [round(pixel[0], 1), round(pixel[1], 1)]
        if pixel is None:
            result["reason"] = "target_out_of_view"
            return result
    elif target.ray is not None:
        pixel = direction_to_pixel(
            target.ray["bearing_deg"], target.ray["elevation_deg"], yaw, intrinsics
        )
        result["pixel"] = None if pixel is None else [round(pixel[0], 1), round(pixel[1], 1)]
        if pixel is None:
            result["reason"] = "ray_out_of_view"
            return result

    depth_fn = getattr(world, "depth_at_pixel", None)
    depth = None
    if callable(depth_fn):
        try:
            value = (
                depth_fn(pixel, pitch_rad=pitch_rad) if pitch_rad else depth_fn(pixel)
            )
            depth = None if value is None else float(value)
        except TypeError:
            if pitch_rad:
                result["reason"] = "pitched_depth_unsupported"
                return result
            depth = None
        except Exception:  # noqa: BLE001 - depth is best effort
            depth = None
    if depth is None or not math.isfinite(depth) or depth <= MIN_RANGE_M:
        result["reason"] = "no_depth_measurement"
        return result

    width, height = _size(intrinsics)
    hfov = _hfov(intrinsics)
    dx, dy, dz = pixel_to_body_offset(
        pixel, depth, width=width, height=height, hfov_deg=hfov
    )
    if pitch_rad:
        # the pixel was projected through the pitched camera: rotate its
        # camera-frame offset back into the body frame before the yaw rotation
        cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
        dx, dy, dz = (
            dx * cos_p + dz * sin_p,
            dy,
            -dx * sin_p + dz * cos_p,
        )
    point = here + np.asarray(body_delta_to_world(dx, dy, dz, yaw), dtype=np.float64)
    if target.position is not None:
        jump = float(np.linalg.norm(point - target.position))
        result["jump_m"] = round(jump, 2)
        if jump > MAX_RELOCALIZE_JUMP_M:
            result["reason"] = "jump_too_large"
            return result

    result.update(
        {
            "relocalized": True,
            "reason": "depth",
            "position_flu": _pos_list(point),
            "position_source": "depth",
            "range_m": round(depth, 2),
            "target_distance_m": round(float(np.linalg.norm(point - here)), 2),
            "anchor": anchor,
        }
    )
    target.position = point
    target.range_m = depth
    target.source = "depth"
    _update_store(ctx, target, result)
    return result


__all__ = [
    "APPROACH_MAX_LEG_M",
    "APPROACH_STANDOFF_M",
    "ARRIVE_RADIUS_M",
    "RAY_STEP_M",
    "CandidateTarget",
    "SegmentPlan",
    "box_center",
    "candidate_record",
    "candidate_target",
    "current_intrinsics",
    "direction_to_pixel",
    "elevation_deg",
    "plan_segment",
    "ray_point",
    "relocalize",
    "standoff_offset",
    "standoff_point",
    "world_standoff_m",
    "world_to_pixel",
]
