"""Frame-contract regressions for ``vlm_nav.world`` (A task).

The coordinate-frame audit (``outputs/vlm_nav/audit/frames_audit.md``) and the
altitude audit (``altitude_audit.md``) found five defects that live in
``vlm_nav/world.py``; each one is pinned here, CPU only:

- the ``pitch_down_deg`` sign: a "downward" scan view rendered the ceiling,
- the pointed pixel projected with a fixed 320x180 @ 86 deg default (I1/I2),
- the MoGe depth map unprojected with a hard-coded 86 deg (I3),
- the 5x5 depth median reading a small target's background (I7),
- the camera mount drawn randomly at sensor-construction time (I6),
- a pointed-pixel action inventing a 2.0 m range when neither the action nor
  the depth had one (``FINAL_REPORT.md`` item 7).

The numbers these tests assert are the ones the audit probes print for the same
scenes (``probe_frames.py --sections synthetic --hfov-deg 60``,
``probe_pitch_sign.py``, ``probe_room_brightness.py``).
"""

from __future__ import annotations

import math
import struct
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vlm_nav.actions import NavAction
from vlm_nav.world import (
    EnvWorld,
    PoseState,
    SplatWorldCPU,
    SuperPlannerWorld,
    TargetSpec,
    depth_from_window,
    estimate_point_world,
    pin_camera_mount,
    yaw_pitch_quat_xyzw,
)

#: FLU -> 3DGS, the rotation the CPU loader/renderer stamp on the geometry
_FLU_TO_3DGS = np.asarray(
    [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _write_test_ply(path, gaussians):
    """gaussians: iterable of (xyz, sh_dc_rgb, opacity_logit, log_scale, rot_wxyz)."""
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(gaussians)}",
            "property float x",
            "property float y",
            "property float z",
            "property float f_dc_0",
            "property float f_dc_1",
            "property float f_dc_2",
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header",
        ]
    )
    rows = [
        list(xyz) + list(sh) + [op] + list(sc) + list(rot)
        for xyz, sh, op, sc, rot in gaussians
    ]
    with open(path, "wb") as handle:
        handle.write((header + "\n").encode("ascii"))
        for row in rows:
            handle.write(struct.pack("<14f", *row))


def _write_flu_ply(path, gaussians_flu):
    """``_write_test_ply`` for geometry authored in FLU metres.

    The loader reads PLY vertices as absolute 3DGS coordinates and puts the FLU
    origin at their centroid, so FLU geometry is rotated into the 3DGS frame
    around its own centroid; the returned centroid is the shift between the
    authored frame and the world's FLU frame (``align_scenario``).
    """
    positions = np.asarray([xyz for xyz, *_ in gaussians_flu], dtype=np.float64)
    origin = positions.mean(axis=0)
    gaussians = [
        (tuple((np.asarray(xyz, dtype=np.float64) - origin) @ _FLU_TO_3DGS.T),
         sh, opacity, log_scale, rot)
        for xyz, sh, opacity, log_scale, rot in gaussians_flu
    ]
    _write_test_ply(path, gaussians)
    return origin


def _room_rows(n: int = 26, half: float = 5.0, ceiling_z: float = 3.0):
    """Dark floor, bright ceiling, grey walls (the pitch-sign probe's room)."""
    white = (1.0 - 0.5) / 0.28209479177387814
    black = (0.0 - 0.5) / 0.28209479177387814
    opacity = _logit(0.99)
    scale = (math.log(0.25),) * 3
    rot = (1.0, 0.0, 0.0, 0.0)
    rows = []
    for i in np.linspace(-half, half, n):
        for j in np.linspace(-half, half, n):
            rows.append(((float(i), float(j), 0.0), (black, black, black), opacity, scale, rot))
            rows.append(((float(i), float(j), ceiling_z), (white, white, white),
                         opacity, scale, rot))
    for x, y in ((-half, 0.0), (half, 0.0), (0.0, -half), (0.0, half)):
        for t in np.linspace(-half, half, n):
            for z in np.linspace(0.0, ceiling_z, 8):
                p = (x, t, z) if x != 0.0 else (t, y, z)
                rows.append((tuple(float(v) for v in p), (0.0, 0.0, 0.0),
                             opacity, scale, rot))
    return rows


class PitchDownSignTest(unittest.TestCase):
    """``pitch_down_deg > 0`` must render the floor, not the ceiling (4.1).

    Measured on this room (96x54): level views mean 0.488 (bright ceiling on
    top, dark floor on the bottom), the pitched scan views 0.000 (all floor) -
    before the sign fix they read 0.995 (all ceiling), which is the frame the
    runs showed while the model was told it was a downward view.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ply = f"{self._tmp.name}/room.ply"
        origin = _write_flu_ply(self.ply, _room_rows())
        self.world = SplatWorldCPU(
            ply_path=self.ply,
            target=TargetSpec(name="floor", position_flu=(0.0, 0.0, 0.0), radius_m=0.3),
            start_flu=(0.0, 0.0, 1.5 - float(origin[2])),  # room centre, 1.5 m up
            start_yaw_deg=0.0,
            width=96,
            height=54,
            hfov_deg=86.0,
        )

    def test_scan_views_pitch_down_looks_at_the_dark_floor(self):
        views = self.world.scan_views(n_views=2, pitch_down_deg=50.0)
        self.assertEqual(len(views), 4)  # 2 level + 2 pitched

        level = [float(view.mean()) for view in views[:2]]
        pitched = [float(view.mean()) for view in views[2:]]
        for mean in level:
            self.assertGreater(mean, 0.3, "the room's ceiling is bright: a level view sees it")
            self.assertGreater(
                float(views[0][:, : views[0].shape[1] // 3].mean()), 0.5
            )  # bright ceiling on top, as the audit's probe reports
        for mean in pitched:
            self.assertLess(mean, 0.1, "a pitch_down_deg view renders the dark floor")
        # the contrast is the whole point: 0.0 (floor) vs ~0.49 (ceiling + floor)
        self.assertLess(max(pitched), min(level) - 0.2)

    def test_positive_body_pitch_is_nose_down(self):
        pos = self.world.position_flu
        down = float(self.world._render_pose(
            pos, yaw_pitch_quat_xyzw(0.0, math.radians(50.0))).mean())
        up = float(self.world._render_pose(
            pos, yaw_pitch_quat_xyzw(0.0, -math.radians(50.0))).mean())
        level = float(self.world.render().mean())

        self.assertLess(down, 0.1)  # +pitch: the dark floor
        self.assertGreater(up, 0.9)  # -pitch: the bright ceiling
        self.assertLess(down, level)
        self.assertLess(level, up)


class EnvWorldPitchSignTest(unittest.TestCase):
    """The production backend's scan writes the same downward pitch (4.1)."""

    def test_env_world_pitched_scan_writes_a_downward_pitch(self):
        # no CUDA env needed: only the pose writes of scan_views are under test
        world = EnvWorld.__new__(EnvWorld)
        world.state = PoseState(np.zeros(3), 0.0)
        written: list = []
        world._write_pose = lambda pitch_rad=0.0: written.append(float(pitch_rad))
        world.render = lambda: torch.zeros(3, 4, 4)

        views = EnvWorld.scan_views(world, n_views=1, pitch_down_deg=50.0)

        self.assertEqual(len(views), 3)  # 1 level + 2 pitched
        self.assertEqual([p for p in written if p != 0.0], [math.radians(50.0)] * 2)
        self.assertGreater(written[1], 0.0)  # + = nose down in body FLU


class EstimatePointWorldIntrinsicsTest(unittest.TestCase):
    """A pointed pixel is projected with the world's own camera (I1/I2).

    The audit measured a 1.615x lateral stretch when a world is projected with
    the legacy 320x180 @ 86 deg default (0.355 m at 2 m on a 60 deg frame).
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ply = f"{self._tmp.name}/dot.ply"
        _write_flu_ply(self.ply, [
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), _logit(0.9),
             (math.log(0.15),) * 3, (1.0, 0.0, 0.0, 0.0)),
        ])
        self.world = SplatWorldCPU(
            ply_path=self.ply,
            target=TargetSpec(name="dot", position_flu=(5.0, 0.0, 0.0), radius_m=0.2),
            start_flu=(0.0, 0.0, 0.0),
            start_yaw_deg=0.0,
            width=64,
            height=64,
            hfov_deg=60.0,
            bound_radius=20.0,
        )
        self.world.depth_at_pixel = lambda pixel: 2.0  # 2 m axial at the pixel

    def test_off_axis_pixel_uses_the_world_hfov(self):
        point = estimate_point_world(self.world, (750.0, 500.0), 0.0)

        self.assertIsNotNone(point)
        # d = 2 m, u = 0.75 * 64 = 48 px: 16 px right of centre on a 60 deg frame
        # -> lateral = -(16 / fx) * 2 = -0.5774 m (the 86 deg default: -0.9325 m)
        self.assertAlmostEqual(float(point[0]), 2.0, places=9)
        self.assertAlmostEqual(float(point[1]), -0.57735, places=4)
        self.assertAlmostEqual(float(point[2]), 0.0, places=9)

    def test_an_explicit_intrinsics_mapping_wins(self):
        point = estimate_point_world(
            self.world, (750.0, 500.0), 2.0,
            intrinsics={"width": 64, "height": 64, "hfov_deg": 86.0},
        )
        lateral = -0.25 * 64.0 / (64.0 / (2.0 * math.tan(math.radians(43.0)))) * 2.0
        self.assertAlmostEqual(float(point[1]), lateral, places=6)

    def test_a_world_without_intrinsics_refuses_an_off_axis_pixel(self):
        bare = SimpleNamespace(
            position_flu=np.zeros(3), yaw_rad=0.0,
            depth_at_pixel=lambda pixel: 2.0,
        )
        self.assertIsNone(estimate_point_world(bare, (750.0, 500.0), 2.0))

    def test_the_optical_axis_needs_no_declared_intrinsics(self):
        # the frame centre is the axis under any pinhole: no FOV is involved,
        # so the honest answer survives a world that declares no camera
        bare = SimpleNamespace(
            position_flu=np.zeros(3), yaw_rad=0.0,
            depth_at_pixel=lambda pixel: 2.0,
        )
        point = estimate_point_world(bare, (500.0, 500.0), 2.0)
        np.testing.assert_allclose(np.asarray(point), [2.0, 0.0, 0.0], atol=1e-9)


class _FakeMogeModel:
    """Records the ``fov_x`` it is handed and returns a fronto-parallel plane."""

    def __init__(self, depth=2.0):
        self.depth = float(depth)
        self.calls: list = []

    def infer(self, image, fov_x=None):
        self.calls.append(fov_x)
        height, width = image.shape[-2:]
        depth = np.full((1, height, width), self.depth, dtype=np.float32)
        return {
            "depth": torch.as_tensor(depth),
            "mask": torch.ones(1, height, width, dtype=torch.bool),
            "intrinsics": torch.as_tensor(
                [[[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]]
            ),
        }


class _MogeWorldStub:
    """Just enough world for ``SuperPlannerWorld._moge_points`` (no CUDA env)."""

    def __init__(self, model, *, hfov_deg=60.0, width=64, height=64):
        self._model = model
        self._moge_device = "cpu"
        self._sensing_range = 10.0
        self._last_depth = None
        self.width, self.height = int(width), int(height)
        if hfov_deg is not None:
            self.hfov_deg = float(hfov_deg)

    def render(self):
        return torch.zeros(3, self.height, self.width)

    def _moge_model(self):
        return self._model

    def _pose_flu(self):
        return np.zeros(3), 0.0


class MogeDepthIntrinsicsTest(unittest.TestCase):
    """MoGe's depth is unprojected through the FOV it was given (I3).

    The audit: with no ``fov_x`` MoGe recovered 54.5 deg on an 86 deg frame and
    the same pixel read 2.35x; the hard-coded 86 deg unprojection scaled the
    accumulated map by ``f_est/f_true``.
    """

    def test_the_frame_hfov_is_handed_to_moge_and_used_to_unproject(self):
        model = _FakeMogeModel(depth=2.0)
        world = _MogeWorldStub(model, hfov_deg=60.0, width=64, height=64)

        points = SuperPlannerWorld._moge_points(world)

        self.assertEqual(model.calls, [60.0])
        self.assertIsNotNone(world._last_depth)
        self.assertIsNotNone(points)
        # a 2 m plane seen through 60 deg: |y| <= 2 * tan(30) = 1.155 m
        self.assertAlmostEqual(float(points[:, 0].max()), 2.0, places=6)
        self.assertAlmostEqual(float(np.abs(points[:, 1]).max()), 2.0 * math.tan(
            math.radians(30.0)), places=3)
        # the 86 deg default would have stretched it to 2 * tan(43) = 1.865 m
        self.assertLess(float(np.abs(points[:, 1]).max()), 1.5)

    def test_a_world_without_hfov_uses_moges_own_intrinsics(self):
        model = _FakeMogeModel(depth=2.0)  # reports fx = 0.5 * W -> 90 deg
        world = _MogeWorldStub(model, hfov_deg=None, width=64, height=64)

        points = SuperPlannerWorld._moge_points(world)

        self.assertEqual(model.calls, [None])
        self.assertIsNotNone(points)
        # 90 deg: |y| <= 2 * tan(45) = 2.0 m, not the 86 deg 1.865 m
        self.assertAlmostEqual(float(np.abs(points[:, 1]).max()), 2.0, places=3)

    def test_a_depth_map_with_no_intrinsics_produces_no_map(self):
        class _BareModel(_FakeMogeModel):
            def infer(self, image, fov_x=None):
                out = super().infer(image, fov_x=fov_x)
                out.pop("intrinsics")
                return out

        world = _MogeWorldStub(_BareModel(), hfov_deg=None)

        self.assertIsNone(SuperPlannerWorld._moge_points(world))


class DepthWindowTest(unittest.TestCase):
    """The pointed pixel wins over the window median across a silhouette (I7).

    The audit's synthetic scene (a 9x10 px target with a surface 0.5 m behind
    it, 320x180 @ 60 deg): the old 5x5 median read 2.2245 m for a true axial
    depth of 2.0881 m (+0.1365 m); the pointed pixel reads 2.1113 m (+0.0233).
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ply = f"{self._tmp.name}/synth.ply"
        self.width, self.height, self.hfov = 320, 180, 60.0

        target = (2.0, 0.6, 0.35)
        start = (0.0, 0.0, 0.2)
        # the audit probe's authoring rule: opacity logit(0.95), isotropic
        # log-scaled splats (probe_frames.py::author_scene)
        opacity = _logit(0.95)
        rows = [(target, (2.0, -1.0, -1.0), opacity, (math.log(0.02),) * 3,
                 (1.0, 0.0, 0.0, 0.0))]
        base = np.asarray(target) + np.asarray([0.5, 0.0, -0.15])
        for dy in (-0.6, 0.0, 0.6):
            for dz in (-0.3, 0.0, 0.3):
                rows.append((tuple(base + np.asarray([0.0, dy, dz])),
                             (0.15, 0.15, 0.15), opacity,
                             (math.log(0.35),) * 3, (1.0, 0.0, 0.0, 0.0)))
        for dx in (-1.0, 1.0):
            for dy in (-1.0, 1.0):
                rows.append(((target[0] + dx, target[1] + dy, -1.5),
                             (0.05, 0.05, 0.05), opacity,
                             (math.log(0.2),) * 3, (1.0, 0.0, 0.0, 0.0)))
        origin = _write_flu_ply(self.ply, rows)
        self.target_flu = np.asarray(target) - origin
        start_flu = np.asarray(start) - origin
        self.world = SplatWorldCPU(
            ply_path=self.ply,
            target=TargetSpec(name="target", position_flu=tuple(self.target_flu),
                              radius_m=0.2),
            start_flu=tuple(start_flu),
            start_yaw_deg=0.0,
            width=self.width,
            height=self.height,
            hfov_deg=self.hfov,
            bound_radius=20.0,
        )
        bearing = math.atan2(self.target_flu[1] - start_flu[1],
                             self.target_flu[0] - start_flu[0])
        self.world.state.yaw_rad = bearing
        self.pixel, self.axial = self._project(bearing)

    def _project(self, yaw):
        """Analytic pixel + axial depth of the target from the current pose."""
        rel = self.target_flu - np.asarray(self.world.position_flu)
        forward = float(rel[0] * math.cos(yaw) + rel[1] * math.sin(yaw))
        left = float(-rel[0] * math.sin(yaw) + rel[1] * math.cos(yaw))
        fx = self.width / (2.0 * math.tan(math.radians(self.hfov) / 2.0))
        fy = fx  # square pixels
        u = 500.0 - left / forward * fx / self.width * 1000.0
        v = 500.0 - float(rel[2]) / forward * fy / self.height * 1000.0
        return (u, v), forward

    def _raw_window_median(self):
        depth = self.world.depth_map().numpy()
        u = min(max(int(self.pixel[0] / 1000.0 * self.width), 0), self.width - 1)
        v = min(max(int(self.pixel[1] / 1000.0 * self.height), 0), self.height - 1)
        window = depth[max(0, v - 2):v + 3, max(0, u - 2):u + 3].reshape(-1)
        window = window[window > 0.0]
        return float(np.median(window))

    def test_a_small_target_reads_its_own_surface_not_the_background(self):
        depth = self.world.depth_at_pixel(self.pixel)

        self.assertIsNotNone(depth)
        # the audit's scene: axial 2.0881 m; the old 5x5 median read 2.2245 m
        self.assertAlmostEqual(self.axial, 2.0881, places=4)
        old = self._raw_window_median()
        self.assertGreater(old - self.axial, 0.1)  # the defect this pins
        self.assertLess(abs(depth - self.axial), 0.05)
        self.assertLess(abs(depth - self.axial), abs(old - self.axial) / 4.0)

    def test_the_5x5_median_still_answers_a_flat_surface(self):
        flat = np.full((5, 5), 2.0)
        self.assertEqual(depth_from_window(flat, centre=2.0), 2.0)
        # a hole at the pointed pixel falls back to the window's median
        holed = np.full((5, 5), 2.0)
        holed[2, 2] = 0.0
        self.assertEqual(depth_from_window(holed, centre=0.0), 2.0)
        self.assertIsNone(depth_from_window(np.zeros((5, 5)), centre=0.0))

    def test_a_nearer_surface_elsewhere_does_not_override_the_median(self):
        # the pointed pixel is the measurement: a nearer object at the window's
        # edge (pointer just off its silhouette) must not grab the read
        window = np.full((5, 5), 4.0)
        window[0, 0] = 2.0
        self.assertEqual(depth_from_window(window, centre=4.0), 4.0)


class PointedPixelWithoutRangeTest(unittest.TestCase):
    """A pointed pixel with no range is reported, never filled in with 2 m.

    ``FINAL_REPORT.md`` item 7: the pixel paths of ``SplatWorldCPU`` and
    ``SuperPlannerWorld`` substituted a fabricated 2.0 m when the action carried
    no ``distance_m`` and the pointed pixel had no depth return - the same
    invented range the audit removed from ``estimate_point_world``. On this
    world (64x64 @ 60 deg, pixel (750, 500)) that fallback moved the drone by
    (0.8471, -0.2445, 0) m, bit-identical to an explicit ``distance_m=2.0``:
    the pixel was flown as if a surface sat 2 m ahead.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ply = f"{self._tmp.name}/dot.ply"
        self.origin = _write_flu_ply(self.ply, [
            ((3.0, 0.0, 0.0), (0.0, 0.0, 0.0), _logit(0.9),
             (math.log(0.15),) * 3, (1.0, 0.0, 0.0, 0.0)),
        ])
        self.target_flu = np.asarray([3.0, 0.0, 0.0]) - self.origin
        self.start_flu = np.asarray([0.0, 0.0, 0.2]) - self.origin
        self.pixel = (750.0, 500.0)  # off-axis, no surface in the frame there
        # the audit's displacement for this world / pixel / 2 m fallback
        self.old_2m_move = np.asarray([0.84707729, -0.24453015, 0.0])
        self.three_m_move = np.asarray([1.84707729, -0.53320529, 0.0])

    def _world(self) -> SplatWorldCPU:
        return SplatWorldCPU(
            ply_path=self.ply,
            target=TargetSpec(name="dot", position_flu=tuple(self.target_flu),
                              radius_m=0.2),
            start_flu=tuple(self.start_flu),
            start_yaw_deg=0.0,
            width=64,
            height=64,
            hfov_deg=60.0,
            bound_radius=20.0,
        )

    def _bare_super_world(self):
        """Bare ``SuperPlannerWorld`` (no env/CUDA) with a recorded ``_fly_leg``."""
        world = SuperPlannerWorld.__new__(SuperPlannerWorld)
        world.state = PoseState(np.asarray([0.0, 0.0, 0.2], dtype=np.float64), 0.0)
        world._map_mode = "global"  # skip the accumulated-map branch
        world.trajectory = [tuple(world.state.position_flu)]
        world.width, world.height, world.hfov_deg = 320, 180, 86.0
        world._last_depth = None  # no monocular depth measured yet
        legs: list = []
        world._fly_leg = (
            lambda subgoal: legs.append(np.asarray(subgoal, dtype=np.float64)) or "flew"
        )
        return world, legs

    def test_the_pointed_pixel_has_no_depth_return(self):
        # the scene's own renderer answers None there, so the refusals below are
        # not an artifact of stubbing the depth out
        world = self._world()
        self.assertIsNone(world.depth_at_pixel(self.pixel))
        self.assertAlmostEqual(
            float(world.depth_at_pixel((500.0, 500.0))), 3.0, delta=0.05
        )

    def test_no_distance_and_no_depth_does_not_invent_one(self):
        world = self._world()
        before = world.position_flu.copy()
        trajectory_len = len(world.trajectory)

        feedback = world.apply_action(NavAction(kind="waypoint", pixel=self.pixel))

        np.testing.assert_allclose(world.position_flu, before)
        self.assertEqual(len(world.trajectory), trajectory_len)  # no phantom path
        # the fabrication this pins would have flown 0.85 m, so the assertion
        # above is not vacuous
        self.assertGreater(float(np.linalg.norm(self.old_2m_move)), 0.5)
        for fragment in ("no range", "distance_m", "depth"):
            self.assertIn(fragment, feedback, feedback)

    def test_a_zero_distance_is_unknown_like_a_missing_one(self):
        world = self._world()
        before = world.position_flu.copy()

        feedback = world.apply_action(
            NavAction(kind="waypoint", pixel=self.pixel, distance_m=0.0)
        )

        np.testing.assert_allclose(world.position_flu, before)
        self.assertIn("no range", feedback, feedback)

    def test_an_explicit_distance_still_flies_the_same_offset(self):
        world = self._world()
        before = world.position_flu.copy()

        world.apply_action(
            NavAction(kind="waypoint", pixel=self.pixel, distance_m=2.0)
        )

        # the removed fallback used to equal this: an estimate still wins
        np.testing.assert_allclose(
            world.position_flu - before, self.old_2m_move, atol=1e-6
        )
        # and the same pixel at 3 m keeps its longer, standoff-shrunk offset
        longer = self._world()
        before_longer = longer.position_flu.copy()
        longer.apply_action(
            NavAction(kind="waypoint", pixel=self.pixel, distance_m=3.0)
        )
        np.testing.assert_allclose(
            longer.position_flu - before_longer, self.three_m_move, atol=1e-6
        )

    def test_a_measured_depth_is_a_range(self):
        world = self._world()
        world.depth_at_pixel = lambda pixel: 3.0  # a surface 3 m along the ray
        before = world.position_flu.copy()

        world.apply_action(
            NavAction(kind="waypoint", pixel=self.pixel, distance_m=0.0)
        )

        # a depth read is a measurement, not an invention: the pixel flies the
        # offset its own range implies (and never the 2 m one)
        np.testing.assert_allclose(
            world.position_flu - before, self.three_m_move, atol=1e-6
        )

    def test_the_super_planner_refuses_a_pixel_with_no_range(self):
        world, legs = self._bare_super_world()

        feedback = world.apply_action(
            NavAction(kind="waypoint", pixel=(500.0, 500.0), reason="the book is there")
        )

        self.assertEqual([], legs)
        np.testing.assert_allclose(world.state.position_flu, [0.0, 0.0, 0.2])
        for fragment in ("no range", "distance_m", "depth"):
            self.assertIn(fragment, feedback, feedback)

    def test_the_super_planner_still_flies_a_measured_depth(self):
        world, legs = self._bare_super_world()
        world._last_depth = np.full((64, 64), 3.0)  # 3 m axial at every pixel

        feedback = world.apply_action(
            NavAction(kind="waypoint", pixel=(500.0, 500.0))
        )

        self.assertEqual("flew", feedback)
        self.assertEqual(1, len(legs))
        # 3.0 m pointed ahead, 1.2 m standoff -> a 1.8 m leg, as before
        np.testing.assert_allclose(legs[0], [1.8, 0.0, 0.2], atol=1e-6)

    def test_the_super_planner_still_flies_an_explicit_distance(self):
        world, legs = self._bare_super_world()

        world.apply_action(
            NavAction(kind="waypoint", pixel=(500.0, 500.0), distance_m=2.0)
        )

        # the removed fallback used to equal this
        self.assertEqual(1, len(legs))
        np.testing.assert_allclose(legs[0], [0.8, 0.0, 0.2], atol=1e-6)


class CameraMountTest(unittest.TestCase):
    """The sensor mount is the identity from construction on (I6).

    ``RandomizerBase.__init__`` draws a random roll/pitch/yaw the moment a
    sensor is built, and only the first ``env.reset()`` zeroes it; every
    pixel-to-ray conversion assumes the render camera is the body frame.
    """

    @staticmethod
    def _sensor():
        pytest.importorskip("diffaero.utils.randomizer")
        from diffaero.utils.randomizer import UniformRandomizer

        def axis(default, low, high):
            return UniformRandomizer(
                shape=(1,), default_value=default, device=torch.device("cpu"),
                enabled=True, randomize_on_reset=True, low=low, high=high,
            )

        return SimpleNamespace(
            roll_angle=axis(0.0, -3.0, 3.0),
            pitch_angle=axis(0.0, -5.0, 5.0),
            yaw_angle=axis(0.0, -3.0, 3.0),
        )

    def test_every_sensor_mount_is_zeroed(self):
        sensors = {"rgb_camera": self._sensor(), "depth_camera": self._sensor()}
        env = SimpleNamespace(sensors=sensors)

        pin_camera_mount(env)

        for sensor in sensors.values():
            for name in ("roll_angle", "pitch_angle", "yaw_angle"):
                self.assertEqual(float(getattr(sensor, name).value), 0.0, name)

    def test_a_sensor_without_randomizers_is_left_alone(self):
        env = SimpleNamespace(sensors={"static": SimpleNamespace()})
        pin_camera_mount(env)  # must not raise
        pin_camera_mount(SimpleNamespace())  # and a world without sensors either


if __name__ == "__main__":
    unittest.main()
