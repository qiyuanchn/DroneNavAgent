"""World backends for the semantic navigation harness.

Both backends expose the same minimal interface used by the harness:

- ``render() -> Tensor[3, H, W]`` float in [0, 1]: current first-person RGB.
- ``apply_action(action) -> str``: execute one NavAction, return feedback text.
- ``position_flu`` / ``yaw_rad``: current pose (FLU, scene-centered).
- ``target``: TargetSpec with the semantic goal.
- ``distance_to_target() -> float``.

Depth reads (``depth_at_pixel`` on normalized [0, 1000] pixels, and
``depth_map``) follow the ``SuperPlannerWorld`` contract: axial metres from the
current pose, ``None``/0 where nothing is visible. The splat backends derive
them from their own renderer, ``EnvWorld`` from a CPU mirror of the env's
activated Gaussians and ``SuperPlannerWorld`` from the MoGe estimate of the
last frame.

Pixel-to-ray conversions go through the intrinsics the world itself declares
(``world_intrinsics`` -> ``observations.resolve_intrinsics``); a world that
declares no field of view makes them refuse (``None`` / the action's own body
deltas) instead of projecting with a default frame.

A ``waypoint`` action that points at a pixel flies the range it has - the depth
read at that pixel, or the action's own ``distance_m`` estimate. With neither,
there is no range to fly: the world answers with :func:`no_range_feedback` and
stays put instead of inventing a distance.

``SplatWorldCPU`` runs anywhere (pure torch CPU splat renderer + kinematic
point model). ``EnvWorld`` wraps the production
``ObstacleAvoidanceGaussianSplatting`` env exactly like
``script/eval_gwp_searching_3dgs.py`` and requires CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from .actions import NavAction, action_to_body_delta, body_delta_to_world
from .cpu_renderer import (
    CameraSpec,
    CpuGaussianScene,
    CpuSplatRenderer,
    yaw_to_quat_xyzw,
)
from .observations import resolve_intrinsics


def diffaero_root() -> Path:
    """Root of the optional diffaero validation checkout.

    The agent package runs without diffaero.  When the validation submodule is
    present (``third_party/diffaero``) its ``cfg`` tree and native SUPER
    extension live there; ``DIFFAERO_ROOT`` overrides the lookup for checkouts
    kept elsewhere.
    """
    override = os.environ.get("DIFFAERO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    repo_root = Path(__file__).resolve().parent.parent
    bundled = repo_root / "third_party" / "diffaero"
    return bundled if bundled.is_dir() else repo_root

#: depth reads take the median over a (2*WINDOW+1)^2 pixel neighbourhood, the
#: robustness window ``SuperPlannerWorld._depth_at_pixel`` uses as well
DEPTH_WINDOW_PX = 2

#: a depth step this large between the pointed pixel and its window median means
#: the window straddles a silhouette (absolute metres, or this fraction of the
#: reading for distant surfaces)
DEPTH_STEP_TOL_M = 0.05
DEPTH_STEP_TOL_FRAC = 0.02

#: depth samples below this are "no return", never a distance
DEPTH_MIN_M = 0.05

#: a normalized [0,1000] pixel this close to 500/500 is the optical axis itself:
#: it maps to the forward ray under any pinhole, so it needs no declared FOV
ON_AXIS_TOL = 0.5

#: an estimated range at or below this is "no estimate" (the rule
#: ``estimate_point_world`` applies); when the pointed pixel has no depth read
#: either, an action has no range and is answered honestly instead of being
#: filled in with a fabricated distance
RANGE_MIN_M = 0.1


@dataclass(frozen=True)
class TargetSpec:
    name: str
    position_flu: tuple[float, float, float]
    radius_m: float = 0.5


@dataclass
class PoseState:
    position_flu: np.ndarray
    yaw_rad: float


def pixel_to_body_offset(
    pixel: tuple[float, float],
    distance_m: float,
    width: int = 320,
    height: int = 180,
    hfov_deg: float = 86.0,
) -> tuple[float, float, float]:
    """Normalized [0,1000] image point + axial depth -> body-frame (FLU) offset.

    ``distance_m`` is AXIAL depth (distance along the camera's forward axis),
    matching depth-map convention (e.g. MoGe-2 output) - NOT Euclidean range.
    Offset = (d, -(u-cx)/fx * d, -(v-cy)/fy * d) in body FLU, the pinhole of
    :func:`pinhole_pixels`.

    The defaults are the legacy fixed 320x180@86 deg frame and are only for
    probes that compare against it; every production caller must pass the
    intrinsics its frames were rendered with (``world_intrinsics``).
    """
    fx, fy, cx, cy = pinhole_pixels(width, height, hfov_deg)
    u = pixel[0] / 1000.0 * width
    v = pixel[1] / 1000.0 * height
    d = float(distance_m)
    return d, -(u - cx) / fx * d, -(v - cy) / fy * d


def pinhole_pixels(width: int, height: int, hfov_deg: float) -> tuple[float, float, float, float]:
    """Pixel pinhole ``(fx, fy, cx, cy)`` of the camera contract used here.

    Square pixels (``fy == fx``), principal point at the frame centre, vertical
    FOV derived from the horizontal one through the aspect ratio - the same
    model the CPU/HiGS renderers and the production sensors use.
    """
    hfov = math.radians(float(hfov_deg))
    fx = float(width) / (2.0 * math.tan(hfov / 2.0))
    return fx, fx, float(width) / 2.0, float(height) / 2.0


def depth_from_window(
    window, centre: float | None = None, floor: float = 0.0
) -> float | None:
    """One axial depth read out of a window of samples around a pointed pixel.

    The (2*``DEPTH_WINDOW_PX``+1)^2 median is the robustness window, but a
    median taken across a silhouette reports whatever is *behind* a target that
    is smaller than the window (audit I7: a 9x10 px target read 0.137 m / 6.5 %
    too far, and the error grows with the window). When the pointed pixel is
    itself a valid return and sits clearly nearer than the window median, that
    pixel is the surface the caller aimed at: return it. The median stays the
    answer for a flat window and for a pixel with no return of its own.
    """
    values = np.asarray(window, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values) & (values > float(floor))]
    if values.size == 0:
        return None
    median = float(np.median(values))
    if centre is None:
        return median
    centre = float(centre)
    if not math.isfinite(centre) or centre <= float(floor):
        return median
    gap = max(DEPTH_STEP_TOL_M, DEPTH_STEP_TOL_FRAC * median)
    return centre if centre < median - gap else median


def world_intrinsics(world, override=None) -> dict:
    """The render intrinsics ``world`` declares for its frames.

    :func:`~vlm_nav.observations.resolve_intrinsics` is the single resolver: an
    explicit ``override`` mapping (``{width, height, hfov_deg}``) first, then
    the world and its camera/sensor objects. ``hfov_deg`` stays ``None`` when
    nothing declares it, and the pixel-to-ray maths in this module then refuses
    to run instead of projecting with the legacy 86 deg default (audit I1).
    """
    return resolve_intrinsics(world=world, override=override)


def render_bev(
    position_flu,
    yaw_rad: float,
    trajectory,
    points=None,
    target_estimate=None,
    size: int = 192,
) -> torch.Tensor:
    """Top-down memory map as a CHW float [0,1] image, HEADING-UP: the drone is
    at the image center and always faces the TOP of the image (egocentric -
    matches how the first-person frames read). Gray = observed obstacle points,
    green = flown trajectory, blue tick = drone, red = estimated target.
    """
    img = np.full((size, size, 3), 0.06, dtype=np.float32)
    pos = np.asarray(position_flu, dtype=np.float64)
    collections = [np.asarray(trajectory, dtype=np.float64).reshape(-1, 3)]
    if points is not None and len(points):
        collections.append(np.asarray(points, dtype=np.float64).reshape(-1, 3))
    if target_estimate is not None:
        collections.append(np.asarray(target_estimate, dtype=np.float64).reshape(1, 3))
    all_xy = np.concatenate([c[:, :2] for c in collections], axis=0)

    # rotate world into the drone's frame: +x' = forward (image up), +y' = left
    a = math.pi / 2.0 - yaw_rad
    ca, sa = math.cos(a), math.sin(a)
    rel = all_xy - pos[:2]
    rot = np.stack(
        [ca * rel[:, 0] - sa * rel[:, 1], sa * rel[:, 0] + ca * rel[:, 1]], axis=1
    )
    span = float(max(np.abs(rot).max() * 2.0, 2.0)) + 1.0

    def to_px(xy):
        # rotated frame: x' right, y' up; drone at the image center
        px = np.asarray(xy) / span * (size - 1) + (size - 1) / 2.0
        return np.stack([px[..., 0], (size - 1) - px[..., 1]], axis=-1)

    def splat_rel(rel_xy, color, r=0):
        xy = np.asarray(rel_xy).reshape(-1, 2)
        # convert world-frame xy to the rotated drone frame
        rel = xy - pos[:2]
        rot_xy = np.stack(
            [ca * rel[:, 0] - sa * rel[:, 1], sa * rel[:, 0] + ca * rel[:, 1]], axis=1
        )
        p = np.round(to_px(rot_xy)).astype(int)
        ok = (p[:, 0] >= r) & (p[:, 0] < size - r) & (p[:, 1] >= r) & (p[:, 1] < size - r)
        p = p[ok]
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                img[p[:, 1] + dy, p[:, 0] + dx] = color

    if points is not None and len(points):
        splat_rel(np.asarray(points)[:, :2], (0.55, 0.55, 0.55))
    traj = np.asarray(trajectory, dtype=np.float64).reshape(-1, 3)
    if len(traj):
        splat_rel(traj[:, :2], (0.2, 0.9, 0.2))
    if target_estimate is not None:
        splat_rel(np.asarray(target_estimate)[:2].reshape(1, 2), (0.95, 0.2, 0.2), r=1)
    # drone at the center, heading tick straight up (egocentric convention)
    splat_rel(pos[:2].reshape(1, 2), (0.2, 0.4, 1.0), r=1)
    nose_world = np.asarray([[0.0, span * 0.08]])  # straight up in drone frame
    p = np.round(to_px(nose_world)).astype(int)
    if 0 <= p[0, 0] < size and 0 <= p[0, 1] < size:
        img[p[0, 1], p[0, 0]] = (0.4, 0.7, 1.0)
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()


def yaw_pitch_quat_xyzw(yaw_rad: float, pitch_rad: float = 0.0) -> np.ndarray:
    """ZYX euler (roll=0) -> quaternion xyzw."""
    sp, cp = math.sin(pitch_rad / 2.0), math.cos(pitch_rad / 2.0)
    sy, cy = math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)
    return np.asarray([-sp * sy, sp * cy, cp * sy, cp * cy])


def estimate_point_world(world, pixel: tuple[float, float], distance_m: float,
                         intrinsics=None):
    """Pointed pixel -> estimated world position (world FLU), or None.

    Distance priority: the world's observed depth at that pixel (MoGe cache),
    then the model's own ``distance_m`` estimate. With neither there is **no
    range**, and the answer is None: a fabricated fallback distance (the old
    2 m) put the point on whatever surface happened to sit at that range, which
    then travelled into the candidate memory and the navigation targets. The
    caller reports the unknown range instead (a ray, or "observe again").

    Intrinsics priority: an explicit ``intrinsics`` mapping (e.g. an
    observation's own ``{width, height, hfov_deg}``), then what the world and
    its camera/sensor objects declare. Off-axis pixels of a world that declares
    no field of view get None: the old silent 320x180@86 deg default stretched
    every lateral offset (1.615x on a 60 deg frame, 0.36 m at 2 m; audit I1).
    A pixel on the optical axis is the one case that needs no FOV at all (any
    pinhole maps the frame centre to the forward ray), so it is still answered.
    """
    depth_fn = getattr(world, "depth_at_pixel", None)
    depth = depth_fn(pixel) if callable(depth_fn) else None
    dist = depth if depth is not None else (
        distance_m if distance_m > RANGE_MIN_M else None
    )
    if dist is None:
        return None
    dist = float(dist)
    if not math.isfinite(dist) or dist <= RANGE_MIN_M:
        return None
    declared = world_intrinsics(world, override=intrinsics)
    hfov, width, height = declared["hfov_deg"], declared["width"], declared["height"]
    if hfov is None or width <= 0 or height <= 0:
        if abs(pixel[0] - 500.0) > ON_AXIS_TOL or abs(pixel[1] - 500.0) > ON_AXIS_TOL:
            return None
        dx, dy, dz = dist, 0.0, 0.0
    else:
        dx, dy, dz = pixel_to_body_offset(
            pixel, dist, width=width, height=height, hfov_deg=float(hfov)
        )
    wx, wy, wz = body_delta_to_world(dx, dy, dz, float(world.yaw_rad))
    return np.asarray(world.position_flu, dtype=np.float64) + np.asarray([wx, wy, wz])


def no_range_feedback(pixel: tuple[float, float], missing: str) -> str:
    """Honest result of a pointed-pixel action whose range is unknown.

    A pointed pixel is a direction, not a distance: with no depth return at the
    pixel and no ``distance_m`` estimate there is nothing to turn into metres,
    so the answer names what is missing and nothing is flown - the same
    contract :func:`estimate_point_world` follows. The old 2.0 m fallback put
    the point on whatever surface happened to sit at that range and flew the
    action as if it had been measured.
    """
    return (
        f"no range to the pointed pixel ({float(pixel[0]):.0f}, "
        f"{float(pixel[1]):.0f}): {missing}; the pixel is a direction, not a "
        "distance - nothing was flown (re-point with a depth reading or a "
        "distance_m estimate, or send explicit body deltas)"
    )


def moge_pixel_intrinsics(out, width: int, height: int, hfov_deg=None):
    """Pixel pinhole ``(fx, fy, cx, cy)`` of a MoGe-2 output, or None.

    A world that declares its hfov (and that value is then also handed to
    ``MoGeModel.infer(fov_x=...)``) defines the geometry - MoGe returns its
    depth in exactly that pinhole. With no declared hfov the model recovers a
    focal from its own point map and reports it in ``out["intrinsics"]``
    (normalized, principal point 0.5/0.5); using *that* keeps depth and
    unprojection consistent. The old hard-coded 86 deg did not (audit I3: 54.5
    deg recovered on an 86 deg frame scaled the same pixel by 2.35x).
    """
    if hfov_deg is not None and width > 0 and height > 0:
        return pinhole_pixels(int(width), int(height), float(hfov_deg))
    matrix = out.get("intrinsics") if hasattr(out, "get") else None
    if matrix is None:
        return None
    matrix = matrix.detach().cpu().numpy() if hasattr(matrix, "detach") else np.asarray(matrix)
    k = np.asarray(matrix, dtype=np.float64).reshape(-1, 3, 3)[0]
    return (float(k[0, 0]) * width, float(k[1, 1]) * height,
            float(k[0, 2]) * width, float(k[1, 2]) * height)


def unproject_depth_points(depth, mask, *, pose_flu, yaw_rad, intrinsics,
                           max_range_m: float, stride: int = 2,
                           min_range_m: float = DEPTH_MIN_M) -> np.ndarray | None:
    """Depth map + camera pose -> world FLU point cloud (the accumulated map).

    ``intrinsics`` is the pixel pinhole ``(fx, fy, cx, cy)`` the depth map is
    expressed in; the caller must pass the same value it handed the depth
    estimator, otherwise the map is scaled by ``f_est/f_true``.
    """
    depth = np.asarray(depth, dtype=np.float64)
    height, width = depth.shape
    fx, fy, cx, cy = (float(v) for v in intrinsics)
    if fx <= 0.0 or fy <= 0.0:
        return None
    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    valid = (np.asarray(mask, dtype=bool) & np.isfinite(depth)
             & (depth > float(min_range_m)) & (depth <= float(max_range_m)))
    step = max(1, int(stride))
    # strided subsample to keep the map compact
    valid &= (us % step == 0) & (vs % step == 0)
    d = depth[valid]
    if d.size == 0:
        return None
    # body FLU: +x forward, +y left, +z up
    x_b = d
    y_b = -(us[valid] - cx) / fx * d
    z_b = -(vs[valid] - cy) / fy * d
    pos = np.asarray(pose_flu, dtype=np.float64)
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    x_w = pos[0] + cos_y * x_b - sin_y * y_b
    y_w = pos[1] + sin_y * x_b + cos_y * y_b
    z_w = pos[2] + z_b
    return np.stack([x_w, y_w, z_w], axis=1).astype(np.float64)


class _StandoffMixin:
    """Shared pixel-aimed offset handling: every world backend that executes a
    ``waypoint`` + ``pixel`` action shrinks the pointed offset the same way."""

    def _apply_standoff(self, dx: float, dy: float, dz: float) -> tuple[float, float, float]:
        """Shrink a pointed offset so the drone stops `standoff_m` short of the
        estimated target point (the target point is not a parking spot)."""
        total = math.sqrt(dx * dx + dy * dy + dz * dz)
        standoff = getattr(self, "standoff_m", 1.2)
        if total <= standoff:
            return 0.0, 0.0, 0.0  # already at the standoff shell
        scale = (total - standoff) / total
        return dx * scale, dy * scale, dz * scale


class SplatWorldCPU(_StandoffMixin):
    """Kinematic drone inside a raw 3DGS PLY, rendered on CPU."""

    def __init__(
        self,
        ply_path: str | Path,
        target: TargetSpec,
        start_flu: tuple[float, float, float] = (0.0, 0.0, 0.0),
        start_yaw_deg: float = 0.0,
        width: int = 320,
        height: int = 180,
        hfov_deg: float = 86.0,
        scene_scale: float = 1.0,
        cruise_speed_mps: float = 1.5,
        yaw_rate_dps: float = 90.0,
        move_dt_s: float = 0.1,
        z_min: float = 0.2,
        z_max: float | None = None,
        bound_radius: float | None = None,
        min_opacity: float = 0.0,
    ):
        self.width, self.height, self.hfov_deg = int(width), int(height), float(hfov_deg)
        self._init_backend(ply_path, scene_scale=scene_scale, min_opacity=min_opacity)
        self.target = target
        self.state = PoseState(
            position_flu=np.asarray(start_flu, dtype=np.float64),
            yaw_rad=math.radians(start_yaw_deg),
        )
        self.cruise_speed_mps = float(cruise_speed_mps)
        self.yaw_rate_dps = float(yaw_rate_dps)
        self.move_dt_s = float(move_dt_s)
        self.z_min = float(z_min)
        self.z_max = float(z_max) if z_max is not None else float("inf")
        self.bound_radius = (
            float(bound_radius) if bound_radius is not None
            else max(1.0, self.scene.radius - 0.5)
        )
        self.trajectory: list[tuple[float, float, float]] = [
            tuple(self.state.position_flu)
        ]
        # depth cache: the frame only changes when the pose does, and
        # ground_target reads one depth per candidate box
        self._depth_key: tuple | None = None
        self._depth_frame: torch.Tensor | None = None

    def _init_backend(self, ply_path: str | Path, scene_scale: float, min_opacity: float) -> None:
        self.scene = CpuGaussianScene(ply_path, scale=scene_scale, min_opacity=min_opacity)
        self.renderer = CpuSplatRenderer(self.scene, CameraSpec(self.width, self.height, self.hfov_deg))

    @property
    def position_flu(self) -> np.ndarray:
        return self.state.position_flu

    @property
    def yaw_rad(self) -> float:
        return self.state.yaw_rad

    #: depth reads can be taken through a pitched camera (a candidate above
    #: or below the level frame still has to be measurable)
    supports_pitched_depth = True

    def quat_xyzw(self, pitch_rad: float = 0.0) -> np.ndarray:
        if pitch_rad:
            return yaw_pitch_quat_xyzw(self.state.yaw_rad, float(pitch_rad))
        return yaw_to_quat_xyzw(self.state.yaw_rad)

    def render(self) -> torch.Tensor:
        return self.renderer.render(self.state.position_flu, self.quat_xyzw())

    def _depth_renderer(self) -> CpuSplatRenderer:
        """Renderer that answers depth reads (CUDA backends override this: their
        own renderer returns RGB only)."""
        return self.renderer

    def depth_map(self, pitch_rad: float = 0.0) -> torch.Tensor:
        """Axial-depth map [H, W] of a camera at the current pose, in scene
        units (metres for a metric scene), 0 where no Gaussian reached the pixel.

        The world is static and the renderer deterministic, so the frame
        rendered from the current pose *is* the frame an observation taken here
        was captured from; the map is cached per pose because a single
        ``ground_target`` call reads one depth per candidate. ``pitch_rad`` is a
        body pitch (nose-down positive): a candidate above the level frame is
        measurable through a pitched camera, and the cache key carries the pitch
        so the two renders never share a map.
        """
        key = (
            np.asarray(self.state.position_flu, dtype=np.float64).tobytes(),
            float(self.state.yaw_rad).hex(),
            float(pitch_rad).hex(),
        )
        if self._depth_key != key:
            _, depth = self._depth_renderer().render_with_depth(
                self.state.position_flu, self.quat_xyzw(pitch_rad)
            )
            self._depth_key, self._depth_frame = key, depth
        return self._depth_frame

    def depth_at_pixel(self, pixel: tuple[float, float], pitch_rad: float = 0.0) -> float | None:
        """Axial depth (metres, along the camera axis) at a normalized [0, 1000]
        pixel of the current frame.

        Same contract as ``SuperPlannerWorld.depth_at_pixel``: normalized pixel
        in, axial metres out, ``None`` when no Gaussian reaches the pixel - so
        ``ground_target`` can turn a box centre into a world position. See
        :func:`depth_from_window` for how the robustness window and the pointed
        pixel itself are combined.
        """
        depth = self.depth_map(pitch_rad)
        height, width = depth.shape
        u = min(max(int(pixel[0] / 1000.0 * width), 0), width - 1)
        v = min(max(int(pixel[1] / 1000.0 * height), 0), height - 1)
        window = depth[
            max(0, v - DEPTH_WINDOW_PX):v + DEPTH_WINDOW_PX + 1,
            max(0, u - DEPTH_WINDOW_PX):u + DEPTH_WINDOW_PX + 1,
        ]
        return depth_from_window(window.numpy(), centre=float(depth[v, u]))

    def scan_views(self, n_views: int = 8, pitch_down_deg: float = 0.0,
                   spread_deg: float | None = None) -> list[torch.Tensor]:
        """Render ``n_views`` frames while rotating in place; restores yaw after.
        With pitch_down_deg > 0, appends 2 extra downward-looking views (at the
        initial heading and its opposite) so targets below the flight level are
        visible during scans. With spread_deg set, the level views span only
        yaw0 +- spread/2 (narrow re-acquisition scan instead of a full circle)."""
        yaw0 = self.state.yaw_rad
        views = []
        try:
            for k in range(n_views):
                if spread_deg is None:
                    yaw = yaw0 + 2.0 * math.pi * k / n_views
                else:
                    s = math.radians(spread_deg)
                    yaw = yaw0 - s / 2.0 + s * k / max(1, n_views - 1)
                self.state.yaw_rad = yaw
                views.append(self.render())
            for dyaw in (0.0, math.pi):
                if pitch_down_deg > 0.0:
                    # +pitch about body +y is nose-down in FLU (audit 4.1: the
                    # negated angle made every "downward" view look at the ceiling)
                    quat = yaw_pitch_quat_xyzw(yaw0 + dyaw, math.radians(pitch_down_deg))
                    views.append(self._render_pose(self.state.position_flu, quat))
        finally:
            self.state.yaw_rad = yaw0
        return views

    def _render_pose(self, pos_flu, quat_xyzw) -> torch.Tensor:
        return self.renderer.render(pos_flu, quat_xyzw)

    def distance_to_target(self) -> float:
        return float(
            np.linalg.norm(
                self.state.position_flu - np.asarray(self.target.position_flu)
            )
        )

    def bev_image(self, target_estimate=None, size: int = 192) -> torch.Tensor:
        """Top-down spatial-memory map (trajectory + drone; subclasses add
        observed obstacle points)."""
        return render_bev(
            self.position_flu, self.yaw_rad, self.trajectory,
            points=None, target_estimate=target_estimate, size=size,
        )

    def apply_action(self, action: NavAction) -> str:
        if action.is_terminal or action.kind == "hover":
            return action.kind
        dx, dy, dz, dyaw = action_to_body_delta(action)
        if action.kind == "waypoint" and action.pixel is not None:
            # point-to-navigate: the pointed pixel + a real range beat
            # hand-guessed offsets. The action's own distance_m comes first, a
            # depth read at the pixel is a measurement, and with neither the
            # pixel has no range: say so (no_range_feedback) instead of
            # inventing one
            dist = float(action.distance_m) if action.distance_m > RANGE_MIN_M else None
            if dist is None:
                depth = self.depth_at_pixel(action.pixel)
                dist = float(depth) if depth is not None else None
            if dist is None:
                return no_range_feedback(
                    action.pixel,
                    "the action carries no distance_m and the renderer returned "
                    "no depth there",
                )
            dx, dy, dz = pixel_to_body_offset(
                action.pixel, dist, width=self.width, height=self.height,
                hfov_deg=self.hfov_deg,
            )
            dx, dy, dz = self._apply_standoff(dx, dy, dz)
            dyaw = 0.0
        turn_time = abs(math.degrees(dyaw)) / self.yaw_rate_dps if dyaw else 0.0
        self.state.yaw_rad = math.atan2(
            math.sin(self.state.yaw_rad + dyaw), math.cos(self.state.yaw_rad + dyaw)
        )
        wx, wy, wz = body_delta_to_world(dx, dy, dz, self.state.yaw_rad)
        dist = math.sqrt(wx * wx + wy * wy + wz * wz)
        move_time = dist / self.cruise_speed_mps if dist > 0 else 0.0
        steps = max(1, int(math.ceil(move_time / self.move_dt_s))) if dist > 0 else 0
        for i in range(1, steps + 1):
            interp = i / steps
            self.trajectory.append(
                tuple(self.state.position_flu + interp * np.asarray([wx, wy, wz]))
            )
        new_pos = self.state.position_flu + np.asarray([wx, wy, wz])
        self.last_subgoal = new_pos.copy()
        clamped = False
        if new_pos[2] < self.z_min:
            new_pos[2] = self.z_min
            clamped = True
        if new_pos[2] > self.z_max:
            new_pos[2] = self.z_max
            clamped = True
        radius_xy = math.hypot(new_pos[0], new_pos[1])
        if radius_xy > self.bound_radius:
            scale = self.bound_radius / radius_xy
            new_pos[0] *= scale
            new_pos[1] *= scale
            clamped = True
        self.state.position_flu = new_pos
        self.trajectory.append(tuple(new_pos))
        feedback = f"moved to ({new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f})"
        if clamped:
            feedback += " (clamped by scene bounds)"
        return feedback


class SplatWorldHiGS(SplatWorldCPU):
    """Kinematic drone rendered by the production HiGS backend (CUDA).

    Same kinematics/bounds as ``SplatWorldCPU`` but renders through
    ``HiGSUTRenderer`` + ``GaussianSplattingCamera``, i.e. the exact sensor
    path of the production ``ObstacleAvoidanceGaussianSplatting`` env.
    """

    def __init__(self, *args, device: str = "cuda:0", **kwargs):
        self._higs_device = torch.device(device)
        super().__init__(*args, **kwargs)

    def _init_backend(self, ply_path: str | Path, scene_scale: float, min_opacity: float) -> None:
        from diffaero.utils.gaussian_splatting.higs.renderer import HiGSUTRenderer
        from diffaero.utils.gaussian_splatting.runtime import (
            GaussianModelLite,
            GaussianSplattingCamera,
        )

        self.scene = GaussianModelLite(Path(ply_path), self._higs_device, scale=scene_scale)
        self.renderer = HiGSUTRenderer(width=self.width, height=self.height, device=self._higs_device)
        self.renderer.prepare_gaussians(self.scene)
        hfov = math.radians(self.hfov_deg)
        vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * self.height / self.width)
        self._camera = GaussianSplattingCamera(hfov, vfov, self.width, self.height, self._higs_device)

    def render(self) -> torch.Tensor:
        return self._render_pose(self.state.position_flu, self.quat_xyzw())

    def _depth_renderer(self) -> CpuSplatRenderer:
        """HiGS renders on the GPU and returns RGB only, so depth reads run on a
        CPU mirror of the same activated Gaussians (built once, lazily): the
        mirror shares the camera convention, so it projects the same geometry."""
        mirror = getattr(self, "_depth_cpu_mirror", None)
        if mirror is None:
            mirror = CpuSplatRenderer(
                CpuGaussianScene.from_gaussian_model(self.scene),
                CameraSpec(self.width, self.height, self.hfov_deg),
            )
            self._depth_cpu_mirror = mirror
        return mirror

    def _render_pose(self, pos_flu, quat_xyzw) -> torch.Tensor:
        device = self._higs_device
        pos = torch.as_tensor(pos_flu, dtype=torch.float32, device=device).reshape(1, 3)
        quat = torch.as_tensor(quat_xyzw, dtype=torch.float32, device=device).reshape(1, 4)
        self._camera.update_transformation_batch(pos, quat, self.scene.center)
        frame = self.renderer.render_batch(self.scene, self._camera)["render"][0]
        return frame.float().clamp(0.0, 1.0).cpu()


class _NoOpPlanner:
    def configure(self, *args, **kwargs):
        return None

    def mark_reset(self, *args, **kwargs):
        return None

    def close(self):
        return None


def pin_camera_mount(env) -> None:
    """Zero every sensor's onboard mount rotation (roll/pitch/yaw).

    ``RandomizerBase.__init__`` draws a random mount the moment a sensor is
    built (pitch +-5 deg, yaw/roll +-3 deg, `cfg/sensor/3dgs_rgb.yaml`) and only
    ``env.reset()`` puts it back through ``RandomizerManager.refresh``. Every
    pixel-to-ray conversion in this module assumes the render camera *is* the
    body frame, while the env's own visibility check does apply the mount - so
    until the first reset the two disagree by up to 0.26 m at 3 m (audit I6).
    Sampling the default here makes construction and reset agree.
    """
    sensors = getattr(env, "sensors", None)
    items = sensors.values() if callable(getattr(sensors, "values", None)) else sensors
    for sensor in items or ():
        for name in ("roll_angle", "pitch_angle", "yaw_angle"):
            reset = getattr(getattr(sensor, name, None), "default", None)
            if callable(reset):
                reset()


class EnvWorld(_StandoffMixin):
    """Production 3DGS env world (requires CUDA), mirroring the eval script.

    Pose overrides follow `script/eval_gwp_searching_3dgs.py::apply_pose`: the
    dynamics state is written directly and sensors re-rendered; `env.step()`
    is not used.
    """

    def __init__(
        self,
        hydra_overrides: list[str] | None = None,
        combined_ply: str | Path | None = None,
        semantic_manifest: str | Path | None = None,
        background_ply: str | Path | None = None,
        voxel_size: float = 0.15,
        target: TargetSpec | None = None,
        device: str = "cuda:0",
        use_super_planner: bool = False,
        worker_python: str | Path | None = None,
    ):
        import gymnasium as gym  # noqa: F401
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        import diffaero.env  # noqa: F401  (gym.register side effect)

        overrides = ["env=oa_3dgs", "sensors=3dgs_rgb", "n_envs=1"]
        overrides += list(hydra_overrides or [])
        config_dir = str((diffaero_root() / "cfg").resolve())
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(config_name="config_test", overrides=overrides)
        # Match script/eval_gwp_searching_3dgs.py main(): semantic start/goal mode.
        OmegaConf.update(cfg, "env.gaussian_splatting.start_goal.mode", "semantic", force_add=True)
        self.env = gym.make(
            id=cfg.env.name, cfg=cfg.env, device=torch.device(device)
        ).unwrapped
        # the mount is drawn randomly by the sensor constructors: pin it before
        # anything renders, so construction and the first reset agree (audit I6)
        pin_camera_mount(self.env)

        if use_super_planner:
            self._prepare_super_worker(worker_python)
        else:
            self._install_noop_planner()
        self._before_gaussian_reload()
        if combined_ply is not None:
            self.env.reload_gaussian_model(
                background_path=Path(background_ply or combined_ply),
                combined_path=Path(combined_ply),
                voxel_size=voxel_size,
                semantic_manifest_path=(
                    Path(semantic_manifest) if semantic_manifest else None
                ),
            )
            if not use_super_planner:
                self.env._super_planners = _NoOpPlanner()
        self._target_override = target
        self.state = PoseState(np.zeros(3), 0.0)
        self.trajectory: list[tuple[float, float, float]] = []
        # depth reads run on a CPU mirror of the activated Gaussians: the HiGS
        # renderer returns RGB only and the env declares no depth sensor
        self._depth_cpu_mirror: CpuSplatRenderer | None = None
        self._depth_key: tuple | None = None
        self._depth_frame: torch.Tensor | None = None

    def _install_noop_planner(self) -> None:
        self.env._configure_super_planner = lambda: None
        self.env._super_planners = _NoOpPlanner()

    def _before_gaussian_reload(self) -> None:
        """Hook for subclasses: runs after env creation, before scene reload."""

    @staticmethod
    def _prepare_super_worker(worker_python: str | Path | None) -> None:
        """Point the SUPER worker subprocess at a Python that can import the
        cp312 ``super_planner_py`` module (the harness venv may be py311).

        Sets ``sys.executable`` (used by SuperPlannerWorker.spawn as the worker
        interpreter) and prepends the repo root to PYTHONPATH so the worker can
        import ``super_planner_py``. Process-wide by necessity; call once before
        ``reload_gaussian_model`` (which spawns the workers).
        """
        if worker_python is not None:
            import sys

            sys.executable = str(worker_python)

        repo_root = str(diffaero_root())
        existing = os.environ.get("PYTHONPATH", "")
        if repo_root not in existing.split(os.pathsep):
            os.environ["PYTHONPATH"] = (
                repo_root + os.pathsep + existing if existing else repo_root
            )

    @property
    def target(self) -> TargetSpec:
        if self._target_override is not None:
            return self._target_override
        pos = self.env.target_pos[0].detach().cpu().numpy()
        radius = float(self.env.target_object_radii_flu[0].detach().cpu())
        name = "target"
        try:
            name = self.env.target_object_entry(0).get("name", "target")
        except Exception:
            pass
        return TargetSpec(name=name, position_flu=tuple(float(v) for v in pos),
                          radius_m=radius)

    @property
    def position_flu(self) -> np.ndarray:
        return self.state.position_flu

    @property
    def yaw_rad(self) -> float:
        return self.state.yaw_rad

    def reset(self, seed: int = 0) -> None:
        self.env.reset(seed=seed)
        pos = self.env.dynamics._state["position"][0].detach().cpu().numpy().copy()
        quat = self.env.dynamics._state["quat_xyzw"][0].detach().cpu().numpy()
        x, y, z, w = quat
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        self.state = PoseState(pos, float(yaw))
        self.trajectory = [tuple(pos)]

    def render(self) -> torch.Tensor:
        obs = self.env.get_observations()
        return obs["exteroception_rgb_camera"][0].detach().cpu()

    #: the CPU depth mirror renders any attitude, so a candidate above or below
    #: the level frame is measurable through a pitched read
    supports_pitched_depth = True

    def quat_xyzw(self, pitch_rad: float = 0.0) -> np.ndarray:
        """Camera quaternion of the current wrapper pose (``pitch_rad`` is a
        body pitch, nose-down positive)."""
        return yaw_pitch_quat_xyzw(self.state.yaw_rad, float(pitch_rad))

    def _depth_renderer(self) -> CpuSplatRenderer:
        """CPU mirror of the env's activated Gaussians, for depth reads.

        The HiGS renderer returns RGB only and the env declares no depth sensor,
        so ``EnvWorld`` had no ``depth_at_pixel`` at all: every confirmation's
        location check returned ``no_depth_measurement``. The mirror shares the
        activated model and the camera convention of the GPU renderer (the same
        path ``SplatWorldHiGS._depth_renderer`` uses). It renders at a reduced
        but aspect-correct size; normalized pixels are resolution independent.
        """
        mirror = getattr(self, "_depth_cpu_mirror", None)
        if mirror is not None:
            return mirror
        model = getattr(self.env, "gaussians", None)
        if model is None or not hasattr(model, "sh"):
            raise RuntimeError("no_gaussian_model")
        intrinsics = world_intrinsics(self)
        hfov = intrinsics.get("hfov_deg")
        if hfov is None:
            raise RuntimeError("hfov_unknown")
        width = int(intrinsics.get("width") or 0)
        height = int(intrinsics.get("height") or 0)
        if width <= 0 or height <= 0:
            width, height = 320, 180
        scale = min(1.0, 320.0 / float(width))
        width = max(64, int(round(width * scale)))
        height = max(36, int(round(height * scale)))
        mirror = CpuSplatRenderer(
            CpuGaussianScene.from_gaussian_model(model),
            CameraSpec(width, height, float(hfov)),
        )
        self._depth_cpu_mirror = mirror
        return mirror

    def depth_map(self, pitch_rad: float = 0.0) -> torch.Tensor:
        """Axial-depth map ``[H, W]`` of a camera at the current pose, in scene
        units.

        Cached per pose *and pitch*, exactly like the splat backends: every
        depth read in one confirmation shares one render. ``pitch_rad`` is a
        body pitch (nose-down positive) - the mirror renders the candidate's
        elevation instead of reporting ``target_out_of_view`` for everything
        above the level frame.
        """
        key = (
            np.asarray(self.state.position_flu, dtype=np.float64).tobytes(),
            float(self.state.yaw_rad).hex(),
            float(pitch_rad).hex(),
        )
        if (
            getattr(self, "_depth_key", None) != key
            or getattr(self, "_depth_frame", None) is None
        ):
            _, depth = self._depth_renderer().render_with_depth(
                self.state.position_flu, self.quat_xyzw(pitch_rad)
            )
            self._depth_key, self._depth_frame = key, depth
        return self._depth_frame

    def depth_at_pixel(self, pixel: tuple[float, float], pitch_rad: float = 0.0) -> float | None:
        """Axial depth at a normalized [0, 1000] pixel of the current frame.

        Same contract as :meth:`SplatWorldCPU.depth_at_pixel` and
        ``SuperPlannerWorld.depth_at_pixel``: ``None`` where no Gaussian reaches
        the pixel (or no activated model is available).
        """
        try:
            depth = self.depth_map(pitch_rad)
        except (RuntimeError, AttributeError, TypeError):
            return None
        height, width = depth.shape
        u = min(max(int(pixel[0] / 1000.0 * width), 0), width - 1)
        v = min(max(int(pixel[1] / 1000.0 * height), 0), height - 1)
        window = depth[
            max(0, v - DEPTH_WINDOW_PX):v + DEPTH_WINDOW_PX + 1,
            max(0, u - DEPTH_WINDOW_PX):u + DEPTH_WINDOW_PX + 1,
        ]
        return depth_from_window(window.numpy(), centre=float(depth[v, u]))

    def distance_to_target(self) -> float:
        return float(
            np.linalg.norm(
                self.state.position_flu - np.asarray(self.target.position_flu)
            )
        )

    def apply_action(self, action: NavAction) -> str:
        if action.is_terminal or action.kind == "hover":
            return action.kind
        dx, dy, dz, dyaw = action_to_body_delta(action)
        self.state.yaw_rad = math.atan2(
            math.sin(self.state.yaw_rad + dyaw), math.cos(self.state.yaw_rad + dyaw)
        )
        wx, wy, wz = body_delta_to_world(dx, dy, dz, self.state.yaw_rad)
        self.state.position_flu = self.state.position_flu + np.asarray([wx, wy, wz])
        self.last_subgoal = self.state.position_flu.copy()
        self.trajectory.append(tuple(self.state.position_flu))
        self._write_pose()
        p = self.state.position_flu
        return f"moved to ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"

    def _write_pose(self, pitch_rad: float = 0.0) -> None:
        """Push the wrapper pose into the dynamics state and re-render sensors.

        ``pitch_rad`` is a body pitch in FLU: **positive is nose-down**, which
        is what a ``pitch_down_deg`` argument means (audit 4.1).
        """
        state = self.env.dynamics._state
        device = state["position"].device
        state["position"] = torch.as_tensor(
            self.state.position_flu, dtype=torch.float32, device=device
        ).reshape(1, 3)
        state["velocity"] = torch.zeros(1, 3, device=device)
        state["acceleration"] = torch.zeros(1, 3, device=device)
        state["acc_thrust"] = torch.tensor([[0.0, 0.0, 9.81]], device=device)
        state["quat_xyzw"] = torch.as_tensor(
            yaw_pitch_quat_xyzw(self.state.yaw_rad, pitch_rad),
            dtype=torch.float32, device=device,
        ).reshape(1, 4)
        state.q2R()
        self.env.progress += 1
        self.env.update_sensor_data()

    def scan_views(self, n_views: int = 8, pitch_down_deg: float = 0.0,
                   spread_deg: float | None = None) -> list[torch.Tensor]:
        yaw0 = self.state.yaw_rad
        views = []
        try:
            for k in range(n_views):
                if spread_deg is None:
                    yaw = yaw0 + 2.0 * math.pi * k / n_views
                else:
                    s = math.radians(spread_deg)
                    yaw = yaw0 - s / 2.0 + s * k / max(1, n_views - 1)
                self.state.yaw_rad = yaw
                self._write_pose()
                views.append(self.render())
            for dyaw in (0.0, math.pi):
                if pitch_down_deg > 0.0:
                    self.state.yaw_rad = yaw0 + dyaw
                    # positive body pitch is nose-down (audit 4.1)
                    self._write_pose(pitch_rad=math.radians(pitch_down_deg))
                    views.append(self.render())
            self.state.yaw_rad = yaw0
        finally:
            self.state.yaw_rad = yaw0
            self._write_pose()
        return views

    def bev_image(self, target_estimate=None, size: int = 192) -> torch.Tensor:
        return render_bev(
            self.position_flu, self.yaw_rad, self.trajectory,
            points=None, target_estimate=target_estimate, size=size,
        )

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


class SuperPlannerWorld(EnvWorld):
    """Two-layer world: the VLM picks subgoals, the production SUPER planner flies them.

    Unlike EnvWorld, the SUPER planner is NOT stubbed: translation actions are
    pushed to ``env.target_pos`` as intermediate subgoals and executed through
    the env's native ``_step_with_super_planner`` path (real dynamics, replans
    every ``replan_dt``, collision/OOB/stall flags surfaced). Yaw-only actions
    write the pose directly because SUPER plans translation only.

    Initialization mirrors the production eval loop
    (``script/eval_gwp_searching_3dgs.py``): ``reset_with_fixed_target`` +
    ``apply_initial_yaw`` semantics, planner preflight priming, and the env's
    own collision-aware start sampling.
    """

    #: ``depth_at_pixel`` serves the last level MoGe estimate: a pitched read
    #: would return that level map at a pitched pixel, so it is not offered
    supports_pitched_depth = False

    def __init__(
        self,
        *args,
        worker_python: str | Path | None = None,
        leg_arrive_radius: float = 0.25,
        leg_max_time_s: float = 5.0,
        map_mode: str = "global",
        sensing_range_m: float = 6.0,
        sensing_backend: str = "ideal",
        moge_model_path: str | None = None,
        moge_device: str = "cuda:0",
        **kwargs,
    ):
        kwargs["use_super_planner"] = True
        kwargs["worker_python"] = worker_python
        if map_mode not in ("global", "accumulate", "blind"):
            raise ValueError(f"unknown map_mode {map_mode!r}")
        self._map_mode = str(map_mode)
        self._sensing_range = float(sensing_range_m)
        self._sensing_backend = str(sensing_backend)
        self._moge_model_path = moge_model_path or "Ruicheng/moge-2-vitl-normal"
        self._moge_device = moge_device
        self._moge = None
        self._last_depth: np.ndarray | None = None
        self._sense_points: np.ndarray | None = None
        self._sense_revision = 0
        self._sense_full: np.ndarray | None = None
        super().__init__(*args, **kwargs)
        self.env.defer_reset = True
        self._leg_arrive_radius = float(leg_arrive_radius)
        self._leg_max_steps = max(1, int(round(leg_max_time_s / float(self.env.dt))))
        self._semantic_target: TargetSpec | None = None
        self.collision_hits = 0

    # ------------------------------------------------------------------
    # Real-time (accumulated) mapping
    # ------------------------------------------------------------------

    def _before_gaussian_reload(self) -> None:
        if self._map_mode == "accumulate":
            self._install_realtime_map_hooks()
        elif self._map_mode == "blind":
            self._install_blind_map_hooks()

    def _install_blind_map_hooks(self) -> None:
        """SUPER plans with a permanently empty map: no privileged geometry and
        no depth sensing at all - pure VLM navigation with straight-line legs
        and no obstacle avoidance (ablation floor)."""
        env = self.env
        self._sense_points = np.asarray([[0.0, 0.0, -100.0]], dtype=np.float64)
        env._super_point_cloud_provider = lambda env_idx: self._sense_points
        env._super_point_cloud_id_provider = lambda env_idx: "blind"
        env._super_map_revisions = lambda: torch.zeros(
            env.n_envs, device=env.device, dtype=torch.long
        )

    def _install_realtime_map_hooks(self) -> None:
        """Replace the spawn-time global SUPER map with an observation-built map.

        The env's map hooks are instance-shadowed so that every planner request
        re-reads the current accumulated observation cloud; bumping
        ``_sense_revision`` makes ``_event_indices`` request a reload (the
        worker then calls load_static_points(clear=True) with our cloud).
        """
        env = self.env
        # bootstrap cloud: non-empty so the spawn handshake sees a map
        self._sense_points = np.asarray([[0.0, 0.0, -100.0]], dtype=np.float64)
        env._super_point_cloud_provider = lambda env_idx: self._sense_points
        env._super_point_cloud_id_provider = lambda env_idx: f"rt-{self._sense_revision}"
        env._super_map_revisions = lambda: torch.full(
            (env.n_envs,), self._sense_revision, device=env.device, dtype=torch.long
        )

    def _sense_universe(self) -> np.ndarray:
        """Full-scene SUPER surface cloud (sensing truth for the ideal sensor)."""
        if self._sense_full is None:
            self._sense_full = self.env._super_point_cloud_for(None, None)
        return self._sense_full

    def _observe(self) -> None:
        """Add one observation to the accumulated map; bump the revision (which
        makes the planner reload the map) only when the observed set grew."""
        if self._sensing_backend == "moge":
            new_pts = self._moge_points()
        else:
            new_pts = self._ideal_points()
        if new_pts is None or new_pts.size == 0:
            return
        merged = np.concatenate([self._sense_points, new_pts], axis=0)
        keys = np.floor(merged / 0.1).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        deduped = merged[np.sort(first)]
        if len(deduped) > len(self._sense_points):
            self._sense_points = deduped
            self._sense_revision += 1

    def _pose_flu(self) -> tuple[np.ndarray, float]:
        state = self.env.dynamics._state
        pos = state["position"][0].detach().cpu().numpy()
        qx, qy, qz, qw = state["quat_xyzw"][0].detach().cpu().numpy()
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return pos, yaw

    def _ideal_points(self) -> np.ndarray | None:
        """Ideal depth-camera model: true scene surface points within sensing
        range and the forward frustum (the declared hfov half-angle + slack, or
        the whole forward hemisphere when the world declares no hfov)."""
        full = torch.as_tensor(self._sense_universe())
        pos, yaw = self._pose_flu()
        rel = full - torch.as_tensor(pos)
        dist = rel.norm(dim=-1)
        keep = dist <= self._sensing_range
        fwd = np.asarray([math.cos(yaw), math.sin(yaw), 0.0])
        rel_np = rel.numpy()
        horiz = rel_np[:, :2]
        horiz_n = np.linalg.norm(horiz, axis=1).clip(min=1e-6)
        cos_a = (horiz * fwd[:2]).sum(axis=1) / horiz_n
        hfov = world_intrinsics(self)["hfov_deg"]
        half = math.radians(0.5 * float(hfov) + 15.0) if hfov else math.pi
        keep &= torch.as_tensor(cos_a >= math.cos(half), dtype=torch.bool)
        return full[keep].numpy()

    def depth_at_pixel(self, pixel: tuple[float, float]) -> float | None:
        return self._depth_at_pixel(pixel)

    def _depth_at_pixel(self, pixel: tuple[float, float]) -> float | None:
        """Axial depth around a normalized [0,1000] pixel from the last MoGe
        estimate (None when no monocular depth has been computed yet).

        ``depth_from_window`` combines the robustness window with the pointed
        pixel itself, so a target smaller than the window is not replaced by
        the background it stands in front of."""
        depth = getattr(self, "_last_depth", None)
        if depth is None:
            return None
        H, W = depth.shape
        u = int(pixel[0] / 1000.0 * W)
        v = int(pixel[1] / 1000.0 * H)
        u = min(max(u, 0), W - 1)
        v = min(max(v, 0), H - 1)
        window = depth[
            max(0, v - DEPTH_WINDOW_PX):v + DEPTH_WINDOW_PX + 1,
            max(0, u - DEPTH_WINDOW_PX):u + DEPTH_WINDOW_PX + 1,
        ]
        return depth_from_window(window, centre=float(depth[v, u]), floor=DEPTH_MIN_M)

    # ---------------- monocular RGB sensing (MoGe-2 depth) ----------------
    def _moge_model(self):
        if self._moge is None:
            from moge.model.v2 import MoGeModel  # pip install git+https://github.com/microsoft/MoGe

            self._moge = MoGeModel.from_pretrained(self._moge_model_path)
            self._moge = self._moge.to(self._moge_device)
            self._moge.eval()
        return self._moge

    def _moge_points(self) -> np.ndarray | None:
        """Monocular path: render the RGB frame, estimate metric depth with
        MoGe-2, unproject through the known pinhole intrinsics and yaw pose.

        The frame's own hfov is handed to MoGe as ``fov_x`` and used for the
        unprojection, so a focal MoGe would have guessed differently cannot
        rescale the accumulated map (audit I3). With no declared hfov the model
        keeps its own recovered intrinsics and those are used instead.
        """
        frame = self.render()  # (3,H,W) float [0,1]
        hfov_deg = world_intrinsics(self)["hfov_deg"]
        model = self._moge_model()
        image = frame.to(self._moge_device).unsqueeze(0)
        with torch.no_grad():
            if hfov_deg is None:
                out = model.infer(image)
            else:
                out = model.infer(image, fov_x=float(hfov_deg))
        depth = out["depth"].squeeze(0).float().cpu().numpy()  # (H,W) meters
        mask = out["mask"].squeeze(0).cpu().numpy().astype(bool)
        self._last_depth = depth  # cached for point-to-navigate distance reads
        H, W = depth.shape
        intrinsics = moge_pixel_intrinsics(out, W, H, hfov_deg)
        if intrinsics is None:
            return None
        pos, yaw = self._pose_flu()
        return unproject_depth_points(
            depth, mask, pose_flu=pos, yaw_rad=yaw, intrinsics=intrinsics,
            max_range_m=self._sensing_range, stride=2,
        )

    @property
    def target(self) -> TargetSpec:
        if self._semantic_target is not None:
            return self._semantic_target
        return super().target

    def reset(self, seed: int = 0, target_index: int | None = None,
              initial_yaw_mode: str = "target", max_init_attempts: int = 8) -> None:
        """Reset with eval-aligned re-sampling: in `visible` mode retry until
        the target is actually in the camera FOV, and in all modes retry while
        the SUPER preflight cannot plan from the sampled start (the evaluator's
        load-time skip lives here; raises RuntimeError when exhausted)."""
        env = self.env
        if initial_yaw_mode == "visible":
            env.require_line_of_sight = True  # mirrors eval main()
        planner_failed = False
        for attempt in range(max(1, int(max_init_attempts))):
            env.reset(seed=seed if attempt == 0 else None)
            env_idx = torch.arange(env.n_envs, device=env.device, dtype=torch.long)
            if target_index is not None:
                # Mirrors eval_gwp_searching_3dgs.py::reset_with_fixed_target.
                target_idx = torch.full_like(env_idx, int(target_index))
                env.randomizer.refresh(env_idx)
                env._assign_semantic_targets(env_idx, target_idx)
                p_reset = env._sample_start_for_target_batch_simple(
                    env.target_pos[env_idx],
                    ensure_line_of_sight=env.require_line_of_sight,
                    distance_min=env.start_goal_min_dist + env.get_semantic_offset(env_idx),
                    distance_max=env.start_goal_max_dist,
                    target_radii=env.target_object_radii_flu[env_idx],
                )
                p_new = env.dynamics._state.p.detach().clone()
                p_new[env_idx] = p_reset
                env.dynamics.reset_idx(env_idx, p_new=p_new)
                env.init_pos[env_idx] = p_reset
                env.progress[env_idx] = 0
                env._best_target_distance[env_idx] = torch.linalg.vector_norm(
                    p_reset - env.target_pos[env_idx], dim=-1
                )
                env._last_distance_improvement_step[env_idx] = 0
                env.arrive_time[env_idx] = 0.0
                env.last_action[env_idx] = 0.0
                env._super_planners.mark_reset(env_idx.detach().cpu().tolist())
                env._clear_super_trajectory_points(env_idx)
            self._apply_initial_yaw(initial_yaw_mode)
            # Snapshot the semantic target BEFORE subgoals overwrite env.target_pos.
            entry = {}
            try:
                entry = env.target_object_entry(0)
            except Exception:
                pass
            self._semantic_target = TargetSpec(
                name=str(entry.get("name") or "target"),
                position_flu=tuple(float(v) for v in env.target_pos[0].detach().cpu().numpy()),
                radius_m=float(env.target_object_radii_flu[0].detach().cpu()),
            )
            if initial_yaw_mode == "visible" and not self.target_in_fov():
                continue  # re-sample until the target is genuinely visible
            # Planner preflight: prime the first trajectory (eval/collect equivalent).
            if self._map_mode == "accumulate":
                self._sync_from_env()  # the wrapper pose must match the sampled
                # env start, otherwise _write_pose below clobbers it with zeros
                self._sense_points = None
                self._sense_revision = 0
                self._install_realtime_map_hooks()
                yaw0 = self.state.yaw_rad
                for k in range(4):  # seed the map all around the start pose
                    self.state.yaw_rad = yaw0 + math.pi / 2 * k
                    self._write_pose()  # push the heading into the env before observing
                    self._observe()
                self.state.yaw_rad = yaw0
                self._write_pose()
            for _ in range(self.env.super_planner_failure_truncate_steps * 8):
                env._super_planners.prime_trajectory_batch(env)
                if not any(env._super_planners.failed):
                    break
            if any(env._super_planners.failed):
                planner_failed = True
                continue  # SUPER cannot plan from this start: re-sample
            self._sync_from_env()
            self.trajectory = [tuple(self.state.position_flu)]
            return
        raise RuntimeError(
            f"no feasible start after {max(1, int(max_init_attempts))} attempts "
            f"(planner_failed={planner_failed}, yaw_mode={initial_yaw_mode})"
        )

    def _apply_initial_yaw(self, mode: str, visible_yaw_frac: float = 0.8) -> None:
        """Mirrors eval_gwp_searching_3dgs.py::apply_initial_yaw."""
        from diffaero.utils.math import euler_to_quaternion

        env = self.env
        if mode in {"target", "visible"}:
            target_yaw = math.atan2(
                float(env.target_pos[0, 1] - env.p[0, 1]),
                float(env.target_pos[0, 0] - env.p[0, 0]),
            )
            if mode == "visible":
                sensor = env.sensors.get("rgb_camera")
                half_hfov = math.radians(float(sensor.hfov)) * 0.5 if sensor is not None else 0.0
                yaw = target_yaw + (random.random() * 2.0 - 1.0) * half_hfov * visible_yaw_frac
            else:
                yaw = target_yaw + (random.random() - 0.5) * (math.pi / 2.0)
        else:
            yaw = random.random() * 2.0 * math.pi - math.pi
        quat = euler_to_quaternion(
            torch.zeros(1, device=env.device),
            torch.zeros(1, device=env.device),
            torch.as_tensor([yaw], device=env.device),
        )
        env.dynamics._state["quat_xyzw"][0] = quat[0]
        env.dynamics._state.q2R()
        env.update_sensor_data(torch.zeros(1, device=env.device, dtype=torch.long))

    def _sync_from_env(self) -> None:
        pos = self.env.dynamics._state["position"][0].detach().cpu().numpy().copy()
        quat = self.env.dynamics._state["quat_xyzw"][0].detach().cpu().numpy()
        x, y, z, w = quat
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        self.state = PoseState(pos, float(yaw))

    def target_in_fov(self) -> bool:
        """Eval-aligned FOV check (env.target_in_fov_mask) against the SEMANTIC
        target - env.target_pos holds subgoals during legs, so evaluate with the
        manifest target temporarily swapped in."""
        if self._semantic_target is None:
            return False
        env = self.env
        saved_pos = env.target_pos[0].clone()
        saved_r = env.target_object_radii_flu[0].clone()
        env.target_pos[0] = torch.as_tensor(
            self._semantic_target.position_flu, dtype=torch.float32, device=env.device
        )
        env.target_object_radii_flu[0] = float(self._semantic_target.radius_m)
        try:
            mask = env.target_in_fov_mask()
        finally:
            env.target_pos[0] = saved_pos
            env.target_object_radii_flu[0] = saved_r
        return bool(mask[0])

    def apply_action(self, action: NavAction) -> str:
        if action.is_terminal or action.kind == "hover":
            return action.kind
        dx, dy, dz, dyaw = action_to_body_delta(action)
        if action.kind == "waypoint" and action.pixel is not None:
            # point-to-navigate: convert pixel + depth/estimate to a world subgoal
            declared = world_intrinsics(self)
            width, height = declared["width"], declared["height"]
            hfov = declared["hfov_deg"]
            if hfov is not None and width > 0 and height > 0:
                if self._map_mode == "accumulate" and self._sensing_backend == "moge":
                    # refresh the depth so pixel reads bind to the CURRENT frame,
                    # never to a depth map cached at a previous leg's pose
                    self._observe()
                # the measured depth wins over the action's own estimate; with
                # neither the pixel has no range to fly and none is invented
                # (no_range_feedback)
                dist = self._depth_at_pixel(action.pixel)
                if dist is None and action.distance_m > RANGE_MIN_M:
                    dist = float(action.distance_m)
                if dist is None:
                    return no_range_feedback(
                        action.pixel,
                        "the action carries no distance_m and the depth estimate "
                        "has no return there",
                    )
                dist = float(dist)
                dx, dy, dz = pixel_to_body_offset(
                    action.pixel, dist, width=width, height=height, hfov_deg=float(hfov)
                )
                dx, dy, dz = self._apply_standoff(dx, dy, dz)
                dyaw = 0.0
            # a world that declares no render intrinsics cannot turn the pointed
            # pixel into an offset: the action's own body deltas are used as-is
        if dyaw:
            self.state.yaw_rad = math.atan2(
                math.sin(self.state.yaw_rad + dyaw), math.cos(self.state.yaw_rad + dyaw)
            )
            self._write_pose()
        wx, wy, wz = body_delta_to_world(dx, dy, dz, self.state.yaw_rad)
        if not (wx or wy or wz):
            return f"turned to yaw {math.degrees(self.state.yaw_rad):.0f} deg"
        subgoal = self.state.position_flu + np.asarray([wx, wy, wz])
        return self._fly_leg(subgoal)

    def _fly_leg(self, subgoal: np.ndarray) -> str:
        env = self.env
        if self._map_mode == "accumulate":
            self._observe()
            # leg length follows evidence (goal.md section 4): never flew a long
            # leg into space the map has never observed
            subgoal = self._cap_leg_by_coverage(self.state.position_flu, subgoal)
        self.last_subgoal = np.asarray(subgoal, dtype=np.float64).copy()
        leg_len = float(np.linalg.norm(subgoal - self.state.position_flu))
        # The arrive gate must be well inside the leg, otherwise the leg ends
        # after the first sim steps without flying anywhere.
        arrive_gate = min(self._leg_arrive_radius, 0.4 * leg_len)
        env.target_pos[0] = torch.as_tensor(subgoal, dtype=torch.float32, device=env.device)
        env._super_planners.mark_target_changed([0])
        leg_steps = max(self._leg_max_steps, int(leg_len / 0.8 / float(env.dt)))
        steps = 0
        outcome = "leg timeout"
        observe_every = max(1, int(round(0.5 / float(env.dt))))  # ~0.5 s
        for _ in range(leg_steps):
            _, _, terminated, extra = env.step(None, return_observations=False)
            steps += 1
            # online observation during the leg, not just at its start (goal.md
            # section 5): the accumulated map grows as new area comes into view
            if self._map_mode == "accumulate" and steps % observe_every == 0:
                self._observe()
            pos = env.p[0].detach().cpu().numpy()
            self.trajectory.append(tuple(float(v) for v in pos))
            # collision audit: the env's own voxel contract (production step
            # hard-codes collision_now=0, so the harness verifies instead)
            hit = env._check_positions_collision(env.p[:1])
            if bool(hit[0]):
                self.collision_hits += 1
                outcome = "collision"
                break
            dist = float(np.linalg.norm(pos - subgoal))
            if dist < arrive_gate:
                outcome = "arrived subgoal"
                break
            if bool(extra["super_planner_failed"][0]):
                outcome = "planner failed"
                break
            if bool(extra["stalled"][0]):
                outcome = "stalled"
                break
            if bool(terminated[0]):
                outcome = "terminated"
                break
        self._sync_from_env()
        p = self.state.position_flu
        return (
            f"moved to ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) "
            f"[{outcome}, {steps} sim steps]"
        )

    def _cap_leg_by_coverage(self, start: np.ndarray, subgoal: np.ndarray) -> np.ndarray:
        """Shorten legs whose midpoint lies in never-observed map space."""
        leg_len = float(np.linalg.norm(subgoal - start))
        if leg_len <= 2.0 or self._sense_points is None:
            return subgoal
        mid = (np.asarray(start) + np.asarray(subgoal)) / 2.0
        d = np.linalg.norm(self._sense_points - mid, axis=1).min()
        if d > 1.0:
            return np.asarray(start) + (np.asarray(subgoal) - np.asarray(start)) * (2.0 / leg_len)
        return subgoal

    def refresh_sensing(self) -> None:
        """Re-observe at the current pose (binds depth to the current frame)."""
        if self._map_mode == "accumulate":
            self._observe()

    def bev_image(self, target_estimate=None, size: int = 192) -> torch.Tensor:
        """BEV map with the observation-built obstacle cloud when accumulating."""
        pts = None
        if self._map_mode == "accumulate" and self._sense_points is not None:
            pts = self._sense_points
            if len(pts) > 50000:
                pts = pts[:: len(pts) // 50000]
        return render_bev(
            self.position_flu, self.yaw_rad, self.trajectory,
            points=pts, target_estimate=target_estimate, size=size,
        )
