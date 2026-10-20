import json
import math
import unittest

import numpy as np
import torch

from vlm_nav.actions import (
    NavAction,
    action_to_body_delta,
    body_delta_to_world,
    parse_action,
    parse_scan_choice,
)
from vlm_nav.cpu_renderer import (
    CameraSpec,
    CpuGaussianScene,
    CpuSplatRenderer,
    yaw_to_quat_xyzw,
)
from vlm_nav.harness import SemanticNavHarness, _flight_altitude_band
from vlm_nav.prompts import build_user_prompt
from vlm_nav.tool_impl_navigation import register_navigation_tools
from vlm_nav.tool_impl_perception import (
    CANDIDATE_STORE_KEY,
    ground_target,
    observe,
)
from vlm_nav.tools import ToolContext, ToolRegistry
from vlm_nav.vlm import MockVLM
from vlm_nav.world import (
    EnvWorld,
    PoseState,
    SplatWorldCPU,
    SplatWorldHiGS,
    SuperPlannerWorld,
    TargetSpec,
)


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
    rows = []
    for xyz, sh, opacity, log_scale, rot in gaussians:
        rows.append(list(xyz) + list(sh) + [opacity] + list(log_scale) + list(rot))
    import struct

    with open(path, "wb") as handle:
        handle.write((header + "\n").encode("ascii"))
        for row in rows:
            handle.write(struct.pack("<14f", *row))


def _logit(p):
    return math.log(p / (1 - p))


#: FLU -> 3DGS, the convention the CPU loader/renderer stamp on the geometry
_FLU_TO_3DGS = np.asarray(
    [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
)


def _write_flu_ply(path, gaussians_flu):
    """``_write_test_ply`` for scenes authored in FLU metres.

    ``CpuGaussianScene`` reads PLY vertices as absolute 3DGS coordinates and
    puts the FLU origin at their centroid, so geometry written in plain FLU
    renders rotated: rotate it into the 3DGS frame around its own FLU centroid
    instead, and return that centroid - the world's FLU frame is the authored
    one shifted by it (``scripts/run_vlm_nav_m2_memory.py::align_scenario``).
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


class ParseActionTest(unittest.TestCase):
    def test_valid_move(self):
        action = parse_action(
            'Here is my plan. {"action": "move", "forward_m": 1.5, '
            '"right_m": -0.5, "yaw_deg": 10, "reason": "target ahead"}'
        )
        self.assertTrue(action.parse_ok)
        self.assertEqual(action.kind, "move")
        self.assertAlmostEqual(action.forward_m, 1.5)
        self.assertAlmostEqual(action.right_m, -0.5)
        self.assertAlmostEqual(action.yaw_deg, 10.0)

    def test_clamping(self):
        action = parse_action(
            '{"action": "move", "forward_m": 99.0, "yaw_deg": -180.0}'
        )
        self.assertAlmostEqual(action.forward_m, 2.0)
        self.assertAlmostEqual(action.yaw_deg, -45.0)

    def test_garbage_degrades_to_hover(self):
        action = parse_action("I think the book is over there!")
        self.assertFalse(action.parse_ok)
        self.assertEqual(action.kind, "hover")
        self.assertFalse(action.is_terminal)

    def test_done(self):
        action = parse_action('{"action": "done", "reason": "arrived"}')
        self.assertTrue(action.is_terminal)

    def test_primitive_forward(self):
        action = parse_action('{"action": "forward", "meters": 1.0, "reason": "ahead"}')
        self.assertTrue(action.parse_ok)
        self.assertEqual(action.kind, "move")
        self.assertAlmostEqual(action.forward_m, 1.0)
        self.assertAlmostEqual(action.right_m, 0.0)

    def test_primitive_turn_left_default_degrees(self):
        action = parse_action('{"action": "turn_left", "reason": "scan"}')
        self.assertEqual(action.kind, "turn")
        self.assertAlmostEqual(action.yaw_deg, 30.0)

    def test_primitive_backward_clamped(self):
        action = parse_action('{"action": "backward", "meters": 5.0}')
        self.assertEqual(action.kind, "move")
        self.assertAlmostEqual(action.forward_m, -2.0)

    def test_primitive_left_maps_to_flu(self):
        # "left" strafe = +y in FLU body frame = negative right_m
        action = parse_action('{"action": "left", "meters": 0.5}')
        dx, dy, dz, _ = action_to_body_delta(action)
        self.assertAlmostEqual(dx, 0.0)
        self.assertAlmostEqual(dy, 0.5)

    def test_scan_choice(self):
        view, visible, dist = parse_scan_choice(
            '{"view": 5, "target_visible": true, "distance_m": 3.5}', 8)
        self.assertEqual(view, 5)
        self.assertTrue(visible)
        self.assertAlmostEqual(dist, 3.5)
        view, visible, dist = parse_scan_choice('{"view": 99}', 8)
        self.assertIsNone(view)
        view, visible, dist = parse_scan_choice("no json here", 8)
        self.assertIsNone(view)
        self.assertFalse(visible)
        self.assertAlmostEqual(dist, 0.0)

    def test_scan_action(self):
        action = parse_action('{"action": "scan", "reason": "lost the target"}')
        self.assertTrue(action.parse_ok)
        self.assertEqual(action.kind, "scan")
        self.assertFalse(action.is_terminal)


class CandidateTrackerTest(unittest.TestCase):
    def test_merge_same_instance(self):
        from vlm_nav.candidates import CandidateTracker

        tracker = CandidateTracker(merge_radius_m=1.0, confirm_sightings=2)
        c1 = tracker.observe((1.0, 0.0, 0.0), step=1)
        c2 = tracker.observe((1.4, 0.1, 0.0), step=2)  # 0.41 m away -> merges
        self.assertEqual(c1.id, c2.id)
        self.assertEqual(c1.sightings, 2)
        self.assertEqual(c1.status, "confirmed")
        self.assertEqual(len(tracker.candidates), 1)

    def test_new_instance_when_far(self):
        from vlm_nav.candidates import CandidateTracker

        tracker = CandidateTracker(merge_radius_m=1.0)
        tracker.observe((1.0, 0.0, 0.0), step=1)
        tracker.observe((5.0, 0.0, 0.0), step=2)
        self.assertEqual(len(tracker.candidates), 2)
        tracker.reject(2, "did not pan out")
        self.assertEqual(tracker.best().id, 1)

    def test_describe_relative(self):
        import math

        from vlm_nav.candidates import CandidateTracker

        tracker = CandidateTracker()
        tracker.observe((1.0, 0.0, 0.0), step=1)
        text = tracker.describe((0.0, 0.0, 0.0), math.pi / 2)
        self.assertIn("#1", text)
        self.assertIn("-90 deg", text)  # +x point is to the right when facing +y

    def test_waypoint_action_big_offsets(self):
        action = parse_action(
            '{"action": "waypoint", "forward_m": 6.0, "right_m": -3.0, '
            '"yaw_deg": 120, "reason": "target is 6m ahead-left"}'
        )
        self.assertTrue(action.parse_ok)
        self.assertEqual(action.kind, "waypoint")
        self.assertAlmostEqual(action.forward_m, 6.0)
        self.assertAlmostEqual(action.right_m, -3.0)
        self.assertAlmostEqual(action.yaw_deg, 120.0)
        # beyond the waypoint clamp
        action = parse_action('{"action": "waypoint", "forward_m": 99.0}')
        self.assertAlmostEqual(action.forward_m, 8.0)

    def test_waypoint_pixel_fields(self):
        action = parse_action(
            '{"action": "waypoint", "pixel": [500, 320], "distance_m": 3.0}'
        )
        self.assertEqual(action.kind, "waypoint")
        self.assertEqual(action.pixel, (500.0, 320.0))
        self.assertAlmostEqual(action.distance_m, 3.0)
        # done also carries pixel for verification
        action = parse_action('{"action": "done", "pixel": [480, 300]}')
        self.assertTrue(action.is_terminal)
        self.assertEqual(action.pixel, (480.0, 300.0))

    def test_body_frame_mapping(self):
        action = NavAction(kind="move", forward_m=1.0, right_m=1.0, up_m=0.5)
        dx, dy, dz, _ = action_to_body_delta(action)
        self.assertAlmostEqual(dx, 1.0)
        self.assertAlmostEqual(dy, -1.0)  # FLU: right is -y
        self.assertAlmostEqual(dz, 0.5)
        wx, wy, wz = body_delta_to_world(1.0, 0.0, 0.0, math.pi / 2)
        self.assertAlmostEqual(wx, 0.0, places=6)
        self.assertAlmostEqual(wy, 1.0, places=6)


class CpuRendererTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.ply_path = f"{self._tmp.name}/tiny.ply"

    def tearDown(self):
        self._tmp.cleanup()

    def _render_single_gaussian(self, world_xyz_flu, cam_pos_flu=(0, 0, 0), yaw=0.0,
                                color_sh=(2.0, -2.0, -2.0)):
        # one red gaussian (SH0 red>0), placed at a FLU position
        gaussians = [
            (
                tuple(world_xyz_flu),
                tuple(color_sh),
                _logit(0.9),
                (math.log(0.15), math.log(0.15), math.log(0.15)),
                (1.0, 0.0, 0.0, 0.0),
            )
        ]
        _write_test_ply(self.ply_path, gaussians)
        scene = CpuGaussianScene(self.ply_path)
        renderer = CpuSplatRenderer(scene, CameraSpec(width=64, height=64, hfov_deg=60.0))
        quat = yaw_to_quat_xyzw(yaw)
        return renderer.render(np.asarray(cam_pos_flu, dtype=np.float32), quat), scene

    def test_gaussian_ahead_renders_center(self):
        # place at FLU (+3, 0, 0) relative to scene center; camera at center
        # looking forward (+x FLU). Scene center = the gaussian itself, so put
        # the gaussian at 3dgs coords that map to FLU (+3,0,0)-ish: with a
        # single vertex the center equals the vertex, so instead offset the
        # camera backwards.
        image, scene = self._render_single_gaussian(
            (0, 0, 0), cam_pos_flu=(-3.0, 0.0, 0.0)
        )
        self.assertEqual(image.shape, (3, 64, 64))
        center = image[:, 28:36, 28:36]
        corner = image[:, :6, :6]
        self.assertGreater(float(center[0].mean()), 0.2)  # red channel on
        self.assertLess(float(center[1].mean()), 0.2)     # green/blue off
        self.assertLess(float(corner.mean()), 0.05)       # corners empty

    def test_gaussian_behind_camera_is_dark(self):
        image, _ = self._render_single_gaussian((0, 0, 0), cam_pos_flu=(3.0, 0.0, 0.0))
        self.assertLess(float(image.mean()), 1e-3)


class StandoffWorldTest(unittest.TestCase):
    """A pixel-aimed waypoint stops short of the pointed point on EVERY world.

    run15: ``SuperPlannerWorld.apply_action`` called ``self._apply_standoff(...)``
    for a model reply that carries a pixel while moving, but only
    ``SplatWorldCPU`` defined it - the episode died with AttributeError.
    """

    def _bare_super_world(self, start_flu=(0.0, 0.0, 0.2)):
        from vlm_nav.world import PoseState

        # no env / CUDA needed: only the shared standoff path is under test
        world = SuperPlannerWorld.__new__(SuperPlannerWorld)
        world.state = PoseState(np.asarray(start_flu, dtype=np.float64), 0.0)
        world._map_mode = "global"  # skip the accumulated-map branch
        world.trajectory = [tuple(world.state.position_flu)]
        # the render camera a real world declares (cfg/sensor/3dgs_rgb:
        # 320x180 @ 86 deg); a pixel of a world that declares none carries no
        # offset and only the action's own body deltas are flown
        world.width, world.height, world.hfov_deg = 320, 180, 86.0
        legs: list = []
        world._fly_leg = (
            lambda subgoal: legs.append(np.asarray(subgoal, dtype=np.float64)) or "flew"
        )
        return world, legs

    def _cpu_world(self):
        import tempfile

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ply = f"{tmp.name}/tiny.ply"
        _write_test_ply(ply, [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), _logit(0.9),
                               (math.log(0.15),) * 3, (1.0, 0.0, 0.0, 0.0))])
        return SplatWorldCPU(
            ply_path=ply,
            target=TargetSpec(name="target", position_flu=(4.0, 0.0, 0.0), radius_m=0.2),
            start_flu=(0.0, 0.0, 0.2),
            start_yaw_deg=0.0,
            width=64,
            height=64,
            hfov_deg=60.0,
            bound_radius=20.0,
        )

    def test_every_world_backend_shares_the_standoff_helper(self):
        for world_cls in (SplatWorldCPU, SplatWorldHiGS, EnvWorld, SuperPlannerWorld):
            self.assertIs(world_cls._apply_standoff, SplatWorldCPU._apply_standoff,
                          world_cls.__name__)

    def test_super_planner_world_executes_a_pixel_waypoint(self):
        world, legs = self._bare_super_world()
        feedback = world.apply_action(
            NavAction(kind="waypoint", pixel=(500.0, 500.0), distance_m=4.0,
                      reason="the book is right there")
        )
        self.assertEqual("flew", feedback)
        self.assertEqual(1, len(legs))
        # 4.0 m pointed straight ahead, 1.2 m standoff -> a 2.8 m leg
        np.testing.assert_allclose(legs[0], [2.8, 0.0, 0.2], atol=1e-6)

    def test_standoff_semantics_match_the_cpu_world(self):
        action = NavAction(kind="waypoint", pixel=(500.0, 500.0), distance_m=3.0)
        cpu = self._cpu_world()
        cpu.apply_action(action)
        world, legs = self._bare_super_world()
        world.apply_action(action)
        self.assertEqual(1, len(legs))
        np.testing.assert_allclose(legs[0], cpu.position_flu, atol=1e-6)

    def test_pointed_point_inside_the_standoff_shell_does_not_move(self):
        world, legs = self._bare_super_world()
        feedback = world.apply_action(
            NavAction(kind="waypoint", pixel=(500.0, 500.0), distance_m=0.8)
        )
        self.assertEqual([], legs)
        self.assertIn("yaw", feedback)
        np.testing.assert_allclose(world.state.position_flu, [0.0, 0.0, 0.2])


class MockHarnessLoopTest(unittest.TestCase):
    def test_oracle_reaches_target(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            # sparse floor-ish gaussians so renders are non-trivial
            gaussians = []
            rng = np.random.default_rng(0)
            for _ in range(200):
                p = rng.uniform(-6, 6, size=3)
                p[2] = -1.5
                gaussians.append(
                    (
                        tuple(p),
                        (0.0, 0.0, 0.0),
                        _logit(0.8),
                        (math.log(0.2), math.log(0.2), math.log(0.05)),
                        (1.0, 0.0, 0.0, 0.0),
                    )
                )
            ply = f"{tmp}/scene.ply"
            _write_test_ply(ply, gaussians)
            target = TargetSpec(name="blob", position_flu=(3.0, 1.0, 0.0), radius_m=0.2)
            world = SplatWorldCPU(
                ply_path=ply,
                target=target,
                start_flu=(0.0, 0.0, 0.0),
                start_yaw_deg=0.0,
                width=64,
                height=64,
                hfov_deg=60.0,
            )
            harness = SemanticNavHarness(
                world=world,
                vlm=MockVLM("oracle"),
                instruction="go to the blob",
                out_dir=f"{tmp}/out",
                max_steps=25,
                success_dist_m=1.0,
                make_video=False,
                save_frames=False,
                look_around=False,
            )
            summary = harness.run()
            self.assertTrue(summary.success, msg=summary.finish_reason)
            self.assertLessEqual(summary.final_distance_m, 1.0 + target.radius_m)
            self.assertEqual(summary.parse_failures, 0)


class _HarnessStubWorld:
    """Minimal world surface for harness unit tests; records every action."""

    def __init__(self, yaw_deg: float = 0.0, declare_intrinsics: bool = True,
                 z_min: float | None = None, z_max: float | None = None,
                 env=None):
        self.position_flu = np.zeros(3, dtype=np.float64)
        self.yaw_rad = math.radians(yaw_deg)
        self.trajectory = [tuple(self.position_flu)]
        self.collision_hits = 0
        self.last_subgoal = np.asarray([9.0, -9.0, 0.0])
        self.target = TargetSpec(name="book", position_flu=(5.0, 0.0, 0.0), radius_m=0.0)
        self.actions: list[NavAction] = []
        self.bev_calls: list = []
        # the altitude envelope a real SplatWorld declares (``z_min`` / ``z_max``)
        # and the env behind it (``navigation_min_flu`` + the cruise window)
        if z_min is not None:
            self.z_min = float(z_min)
        if z_max is not None:
            self.z_max = float(z_max)
        if env is not None:
            self.env = env
        if declare_intrinsics:
            # the camera every real world declares (width/height/hfov); a world
            # that declares none cannot turn a pointed pixel into a position
            self.width, self.height, self.hfov_deg = 32, 24, 86.0

    def render(self) -> torch.Tensor:
        return torch.zeros(3, 24, 32)

    def scan_views(self, n_views: int = 8, pitch_down_deg: float = 0.0,
                   spread_deg: float | None = None) -> list[torch.Tensor]:
        views = [torch.full((3, 24, 32), 0.05 * k) for k in range(int(n_views))]
        if pitch_down_deg > 0.0:
            # the two pitched views every real backend appends (initial heading
            # and its opposite), so the harness's pitched branch is reachable
            views += [torch.full((3, 24, 32), 0.9)] * 2
        return views

    def apply_action(self, action: NavAction) -> str:
        self.actions.append(action)
        return "ok"

    def distance_to_target(self) -> float:
        return 5.0

    def target_in_fov(self) -> bool:
        return False

    def bev_image(self, target_estimate=None, size: int = 192) -> torch.Tensor:
        self.bev_calls.append(target_estimate)
        return torch.zeros(3, 8, 8)


class BevMarkerTest(unittest.TestCase):
    def _harness(self, tmp, vlm, **kwargs):
        world = _HarnessStubWorld()
        harness = SemanticNavHarness(
            world=world,
            vlm=vlm,
            instruction="fly to the book",
            out_dir=f"{tmp}/out",
            make_video=False,
            save_frames=False,
            look_around=False,
            rescan_after=0,
            **kwargs,
        )
        return world, harness

    def test_red_dot_is_absent_without_a_sighting(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            world, harness = self._harness(tmp, MockVLM(mode="forward"), max_steps=1)
            harness.run()
            # the minimap is drawn, but never with last_subgoal passed off as a
            # target estimate (run13: the model read its own wake as the goal)
            self.assertEqual(len(world.bev_calls), 1)
            self.assertIsNone(world.bev_calls[0])
            self.assertIsNotNone(world.last_subgoal)

    def test_red_dot_marks_the_pointed_sighting(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            vlm = MockVLM(mode="script", script=[
                {"action": "waypoint", "pixel": [500, 500], "distance_m": 3.0,
                 "reason": "the book is there"},
                {"action": "hover", "reason": "stop"},
            ])
            world, harness = self._harness(tmp, vlm, max_steps=2)
            harness.run()
            self.assertIsNone(world.bev_calls[0])
            self.assertIsNotNone(world.bev_calls[1])
            self.assertEqual(
                tuple(world.bev_calls[1]),
                tuple(harness._active_candidate.position_flu),
            )


class PitchedScanDescentTest(unittest.TestCase):
    """The pitched-view descent is a body-FLU ``-z`` displacement, not a climb.

    A positive ``pitch_down_deg`` is nose-down in body FLU, so a pitched scan
    view looks *down* and the harness follows it by descending: ``up_m < 0``
    means body FLU ``-z`` for every heading, which is what the world's
    ``body_delta_to_world`` applies.
    """

    def test_a_pitched_view_descends_along_body_down(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            world = _HarnessStubWorld()
            harness = SemanticNavHarness(
                world=world,
                vlm=MockVLM(mode="script", script=[
                    {"view": 8, "target_visible": True, "distance_m": 4.0},
                ]),
                instruction="fly to the book",
                out_dir=f"{tmp}/out",
                make_video=False,
                save_frames=False,
                look_around=False,
            )
            history = []
            harness._look_around(history)  # view 8 of 10 is the first pitched view
            turns = [a for a in world.actions if a.kind == "turn"]
            legs = [a for a in world.actions if a.kind == "waypoint"]
            self.assertEqual(turns, [])  # a downward view sits at the current heading
            self.assertEqual(len(legs), 1)
            self.assertLess(legs[0].up_m, 0.0)  # min(1.5, 0.4 * 4.0): descend
            self.assertAlmostEqual(legs[0].up_m, -1.5, places=6)
            self.assertAlmostEqual(legs[0].forward_m, 2.8, places=6)
            for yaw_rad in (0.0, math.pi / 2.0, 2.5):
                _, _, dz = body_delta_to_world(0.0, 0.0, legs[0].up_m, yaw_rad)
                self.assertAlmostEqual(dz, legs[0].up_m, places=9)
                self.assertLess(dz, 0.0)  # body FLU +z is up: down is negative


class _StubCruiseEnv:
    """The env attributes the harness's flight band reads."""

    def __init__(self, floor_z: float, window_min: float = 0.6,
                 window_max: float = 1.2):
        self.navigation_min_flu = np.asarray([0.0, 0.0, floor_z], dtype=np.float64)
        self.cruise_window_min_m = float(window_min)
        self.cruise_window_max_m = float(window_max)
        self.cruise_ceiling_m = 1.6


class PitchedScanDescentClampTest(unittest.TestCase):
    """A pitched-view descent is clamped into the flyable altitude band.

    ``_look_around`` follows a downward view by descending
    ``min(1.5, 0.4 * distance)`` m and nothing checked where that step ends: the
    world clamps only at its own ``z_min``, and the env's cruise window is a
    *goal* rule. The step is clamped above the navigable floor (the env's
    ``navigation_min_flu``, else the world's ``z_min`` - what the splat-world
    runners pass as the flyable floor), inside the env's cruise window and under
    the world's ceiling, so "look down" can never dive below the floor and an
    episode that starts above the window is not left there.
    """

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def _leg(self, world, distance_m: float = 4.0):
        harness = SemanticNavHarness(
            world=world,
            vlm=MockVLM(mode="script", script=[
                {"view": 8, "target_visible": True, "distance_m": distance_m},
            ]),
            instruction="fly to the book",
            out_dir=f"{self.tmp}/out",
            make_video=False,
            save_frames=False,
            look_around=False,
        )
        harness._look_around([])  # view 8 of 10 is the first pitched view
        legs = [a for a in world.actions if a.kind == "waypoint"]
        self.assertEqual(len(legs), 1)
        return legs[0]

    def test_the_descent_stops_above_the_declared_scene_floor(self):
        world = _HarnessStubWorld(z_min=0.2)
        world.position_flu = np.asarray([0.0, 0.0, 0.5])
        # desired min(1.5, 0.4 * 4) = -1.5 would end at -1.0, under the floor
        self.assertAlmostEqual(-0.3, self._leg(world).up_m, places=6)

    def test_the_env_floor_and_window_bound_the_step_from_below(self):
        world = _HarnessStubWorld(z_min=-5.0, env=_StubCruiseEnv(floor_z=1.0))
        world.position_flu = np.asarray([0.0, 0.0, 2.6])
        # desired 1.1; the cruise window's own floor (1.0 + 0.6) is the end
        self.assertAlmostEqual(-1.0, self._leg(world).up_m, places=6)

    def test_a_shallow_descent_from_above_the_window_reaches_it(self):
        world = _HarnessStubWorld(z_min=-5.0, env=_StubCruiseEnv(floor_z=1.0))
        world.position_flu = np.asarray([0.0, 0.0, 2.6])
        # desired -0.24 (a 0.6 m estimate); the window's ceiling (2.2) is where
        # the step has to end instead of staying at 2.36
        self.assertAlmostEqual(-0.4, self._leg(world, distance_m=0.6).up_m, places=6)

    def test_the_world_ceiling_bounds_the_step(self):
        world = _HarnessStubWorld(z_min=0.0, z_max=1.0)
        world.position_flu = np.asarray([0.0, 0.0, 1.5])
        # desired -0.24 ends above the declared no-fly height; 1.0 is the limit
        self.assertAlmostEqual(-0.5, self._leg(world, distance_m=0.6).up_m, places=6)

    def test_no_pitched_step_leaves_the_band(self):
        env = _StubCruiseEnv(floor_z=0.0)
        for z in (0.1, 0.6, 1.2, 2.4):
            for distance_m in (0.6, 1.2, 4.0, 20.0):
                world = _HarnessStubWorld(z_min=-5.0, env=env)
                world.position_flu = np.asarray([0.0, 0.0, z])
                end_z = z + self._leg(world, distance_m=distance_m).up_m
                self.assertGreaterEqual(end_z, 0.6 - 1.0e-6)  # window floor
                self.assertLessEqual(end_z, 1.2 + 1.0e-6)     # window ceiling

    def test_the_band_reads_a_real_splat_worlds_declared_envelope(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            ply = f"{tmp}/target.ply"
            _write_flu_ply(ply, [
                ((3.0, 0.0, 0.0), (2.0, -1.0, -1.0), _logit(0.9),
                 (math.log(0.15),) * 3, (1.0, 0.0, 0.0, 0.0)),
            ])
            world = SplatWorldCPU(
                ply_path=ply,
                target=TargetSpec(name="book", position_flu=(3.0, 0.0, 0.0),
                                  radius_m=0.2),
                start_flu=(0.0, 0.0, 0.5),
                z_min=0.35, z_max=1.4,
            )
            self.assertEqual((0.35, 1.4), _flight_altitude_band(world))


class PointedPixelIntrinsicsTest(unittest.TestCase):
    """A pointed pixel is only a position when a camera is declared.

    ``frames_audit.md`` I1: the harness-side projection used to fall through to
    a 320x180 @ 86 deg default. Without declared intrinsics the honest answer is
    "not localized" - no candidate is remembered out of an assumed field of
    view, and the episode summary counts it.
    """

    def _harness(self, tmp, world, script, **kwargs):
        harness = SemanticNavHarness(
            world=world,
            vlm=MockVLM(mode="script", script=script),
            instruction="fly to the book",
            out_dir=f"{tmp}/out",
            make_video=False,
            save_frames=False,
            look_around=False,
            rescan_after=0,
            max_steps=2,
            **kwargs,
        )
        return world, harness

    def test_a_pixel_aimed_sighting_is_localized_with_the_declared_camera(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            script = [
                {"action": "waypoint", "pixel": [500, 500], "distance_m": 3.0,
                 "reason": "the book is there"},
                {"action": "hover", "reason": "stop"},
            ]
            world, harness = self._harness(tmp, _HarnessStubWorld(), script)
            summary = harness.run()
            self.assertIsNotNone(harness._active_candidate)
            self.assertEqual(summary.unlocalized_sightings, 0)

    def test_a_pixel_aimed_sighting_without_intrinsics_is_not_invented(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            script = [
                {"action": "waypoint", "pixel": [750, 500], "distance_m": 3.0,
                 "reason": "the book is there"},
                {"action": "hover", "reason": "stop"},
            ]
            world = _HarnessStubWorld(declare_intrinsics=False)
            world, harness = self._harness(tmp, world, script)
            summary = harness.run()
            # the model's range estimate is real, the direction is not: nothing
            # is remembered about a place that was never localized
            self.assertIsNone(harness._active_candidate)
            self.assertIsNone(world.bev_calls[0])
            self.assertEqual(summary.unlocalized_sightings, 1)


class ScanClaimGuardTest(unittest.TestCase):
    def _harness(self, tmp, script, **kwargs):
        world = _HarnessStubWorld()
        harness = SemanticNavHarness(
            world=world,
            vlm=MockVLM(mode="script", script=script),
            instruction="fly to the book",
            out_dir=f"{tmp}/out",
            make_video=False,
            save_frames=False,
            look_around=False,
            **kwargs,
        )
        return world, harness

    def test_scan_without_visible_target_keeps_heading(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            world, harness = self._harness(tmp, script=[
                {"view": 6, "target_visible": False, "distance_m": 3.0,
                 "reason": "the book is not visible in any view"},
            ])
            history = []
            harness._look_around(history)
            # neither the view heading nor the claimed range may be executed
            self.assertEqual(world.actions, [])
            self.assertIn("kept heading", history[-1]["outcome"])
            self.assertNotIn("turned", history[-1]["outcome"])

    def test_scan_with_visible_target_turns_and_closes_in(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            world, harness = self._harness(tmp, script=[
                {"view": 2, "target_visible": True, "distance_m": 3.0},
            ])
            history = []
            harness._look_around(history)
            turns = [a for a in world.actions if a.kind == "turn"]
            self.assertAlmostEqual(sum(a.yaw_deg for a in turns), 90.0, places=6)
            legs = [a for a in world.actions if a.kind == "waypoint"]
            self.assertEqual(len(legs), 1)
            self.assertAlmostEqual(legs[0].forward_m, 2.1)  # 0.7 * 3.0
            self.assertIn("turned to heading", history[-1]["outcome"])

    def test_scan_claim_with_2x_distance_jump_is_crop_reviewed(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            world, harness = self._harness(tmp, script=[
                {"view": 4, "target_visible": True, "distance_m": 2.5},
                {"view": 4, "target_visible": True, "distance_m": 7.5},
                {"verified": False, "reason": "no book in this crop"},
            ])
            history = []
            harness._look_around(history)
            after_first = len(world.actions)
            self.assertIn("turned to heading", history[-1]["outcome"])
            harness._look_around(history)
            self.assertEqual(harness.vlm.script, [])  # the crop review did run
            self.assertEqual(len(world.actions), after_first)  # nothing executed
            self.assertIn("crop review rejected", history[-1]["outcome"])

    def test_crop_review_passing_or_inconclusive_keeps_the_claim(self):
        import tempfile

        for verdict in ('{"verified": true, "reason": "it is the book"}',
                        "no json at all"):
            with self.subTest(verdict=verdict):
                with tempfile.TemporaryDirectory() as tmp:
                    world, harness = self._harness(tmp, script=[
                        {"view": 0, "target_visible": True, "distance_m": 2.5},
                        {"view": 0, "target_visible": True, "distance_m": 7.5},
                        verdict,
                    ])
                    history = []
                    harness._look_around(history)
                    harness._look_around(history)
                    legs = [a for a in world.actions if a.kind == "waypoint"]
                    self.assertAlmostEqual(legs[-1].forward_m, 5.25)  # 0.7 * 7.5
                    self.assertEqual(harness.vlm.script, [])


class BevPromptTextTest(unittest.TestCase):
    def test_red_dot_is_described_as_an_own_sighting_only(self):
        text = build_user_prompt("fly to the book", 1, 10, [], bev=True)
        self.assertIn("does NOT show where the target is", text)
        self.assertIn("no red dot means you have not seen the target yet", text)
        self.assertNotIn("estimated target position", text)


class _StaticGroundingVLM:
    """Grounding backend stub: one fixed reply, no weights."""

    name = "static"

    def __init__(self, reply: str):
        self.reply = reply

    def decide(self, frames, system_prompt, user_prompt, context=None):
        return self.reply


class SplatWorldDepthTest(unittest.TestCase):
    """Depth reads of the splat backends, and the id -> position chain they feed.

    The M2/section 7 gap: ``ground_target`` could store a candidate id but never
    a position, because ``SplatWorldCPU``/``SplatWorldHiGS`` had no
    ``depth_at_pixel`` (only ``SuperPlannerWorld`` did) - so
    ``approach_candidate{candidate_id: cand_1}`` resolved 0/36. These tests pin the depth
    contract (normalized pixel in, axial metres out, as in ``SuperPlannerWorld``)
    and the end-to-end chain from a grounded box to a world position.
    """

    TARGET_FLU = (3.0, 0.0, 0.0)  # authored: 3 m straight ahead of the start

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ply = f"{self._tmp.name}/target.ply"
        origin = _write_flu_ply(self.ply, self._target_gaussians())
        # the scene's FLU frame is the authored one shifted by the vertex
        # centroid, so the scenario is re-expressed in it (align_scenario)
        self.world_target = np.asarray(self.TARGET_FLU) - origin
        self.start_flu = -np.asarray(origin)

    def _target_gaussians(self):
        """Small red cluster at the scenario's target position."""
        return [
            (tuple(np.asarray(self.TARGET_FLU) + np.asarray(offset)),
             (2.0, -1.0, -1.0), _logit(0.9), (math.log(0.15),) * 3,
             (1.0, 0.0, 0.0, 0.0))
            for offset in ((0.0, 0.0, 0.0), (0.05, 0.03, 0.02), (-0.03, -0.04, 0.03))
        ]

    def _world(self, **kwargs):
        return SplatWorldCPU(
            ply_path=self.ply,
            target=TargetSpec(
                name="target", position_flu=tuple(self.world_target), radius_m=0.2
            ),
            start_flu=tuple(self.start_flu),
            start_yaw_deg=0.0,
            bound_radius=20.0,
            **kwargs,
        )

    def test_every_splat_backend_serves_depth_at_pixel(self):
        for world_cls in (SplatWorldCPU, SplatWorldHiGS):
            self.assertTrue(
                callable(getattr(world_cls, "depth_at_pixel", None)), world_cls.__name__
            )
        # the CUDA backend cannot reuse its own renderer (RGB only)
        self.assertIsNot(SplatWorldHiGS._depth_renderer, SplatWorldCPU._depth_renderer)

    def test_depth_at_pixel_is_axial_metres_at_the_normalized_pixel(self):
        world = self._world()
        depth = world.depth_at_pixel((500.0, 500.0))  # the cluster sits on the axis

        self.assertIsNotNone(depth)
        self.assertAlmostEqual(depth, 3.0, delta=0.1)
        self.assertEqual(tuple(world.depth_map().shape), (world.height, world.width))

    def test_depth_at_pixel_is_none_where_nothing_is_visible(self):
        world = self._world()
        self.assertIsNone(world.depth_at_pixel((40.0, 40.0)))  # empty sky corner

    def test_env_world_serves_depth_from_its_activated_gaussians(self):
        """``EnvWorld`` had no depth hook at all (the v7 0/16 position checks)."""
        from types import SimpleNamespace

        from diffaero.utils.gaussian_splatting.runtime import GaussianModelLite

        model = GaussianModelLite(self.ply, device=torch.device("cpu"))
        world = EnvWorld.__new__(EnvWorld)
        world.state = PoseState(
            np.asarray(self.start_flu, dtype=np.float64), 0.0
        )
        world.env = SimpleNamespace(gaussians=model)
        world.hfov_deg, world.width, world.height = 86.0, 320, 180

        depth = world.depth_at_pixel((500.0, 500.0))
        self.assertIsNotNone(depth)
        self.assertAlmostEqual(depth, 3.0, delta=0.2)
        self.assertEqual(tuple(world.depth_map().shape), (180, 320))

    def test_a_pitched_camera_reads_a_target_above_the_level_frame(self):
        """The v8 seed7/seed17 geometry: an elevated centre is out of a level frame.

        The position check used to project such a candidate into the level
        camera and give up (``target_out_of_view``). A camera pitched nose-up at
        the candidate's elevation has to read its depth, and the pitch has to be
        part of the depth cache key.
        """
        import tempfile

        center = np.asarray((3.0, 0.0, 2.0))
        with tempfile.TemporaryDirectory() as tmp:
            ply = f"{tmp}/above.ply"
            origin = _write_flu_ply(ply, [
                (tuple(center + np.asarray(offset)),
                 (2.0, -1.0, -1.0), _logit(0.9), (math.log(0.15),) * 3,
                 (1.0, 0.0, 0.0, 0.0))
                for offset in ((0.0, 0.0, 0.0), (0.05, 0.03, 0.02), (-0.03, -0.04, 0.03))
            ])
            world = SplatWorldCPU(
                ply_path=ply,
                target=TargetSpec(
                    name="target", position_flu=tuple(center - origin), radius_m=0.2
                ),
                start_flu=tuple(-np.asarray(origin)),
                start_yaw_deg=0.0,
                bound_radius=20.0,
            )

            elevation = math.degrees(math.atan2(2.0, 3.0))
            self.assertIsNone(world.depth_at_pixel((500.0, 500.0)))  # level: empty sky
            depth = world.depth_at_pixel(
                (500.0, 500.0), pitch_rad=math.radians(-elevation)
            )
            self.assertIsNotNone(depth, "a nose-up camera must hold the candidate")
            self.assertAlmostEqual(depth, math.hypot(3.0, 2.0), delta=0.15)
            # the read really is pitched: the level camera still cannot see it
            self.assertIsNone(world.depth_at_pixel((500.0, 500.0)))

    def test_the_depth_frame_follows_the_pose(self):
        world = self._world()
        self.assertAlmostEqual(world.depth_at_pixel((500.0, 500.0)), 3.0, delta=0.1)

        # 1 m short of the target: the same pixel now reads 1 m
        world.state.position_flu = self.world_target - np.asarray([1.0, 0.0, 0.0])
        self.assertAlmostEqual(world.depth_at_pixel((500.0, 500.0)), 1.0, delta=0.1)

    def test_ground_target_publishes_a_world_position_and_approach_uses_it(self):
        world = self._world()
        ctx = ToolContext(world=world, episode_id="ep_depth", extra={})
        observed = observe(ctx, {"mode": "forward"})
        self.assertEqual(observed.status, "ok")

        # the model's box centre: the target's own pixels, read from the frame
        depth = world.depth_map()
        rows, cols = (depth > 0.0).nonzero(as_tuple=True)
        u = (float(cols.float().mean()) + 0.5) / world.width * 1000.0
        v = (float(rows.float().mean()) + 0.5) / world.height * 1000.0
        reply = json.dumps(
            {"candidates": [{"box": [u - 40.0, v - 40.0, 80.0, 80.0], "confidence": 0.8}]}
        )
        grounded = ground_target(
            ctx,
            {
                "instruction": "the red target",
                "observation_id": observed.data["observation_id"],
            },
            vlm=_StaticGroundingVLM(reply),
        )

        self.assertEqual(grounded.status, "ok")
        candidate = grounded.data["candidates"][0]
        self.assertEqual(candidate["position_source"], "depth")
        position = np.asarray(candidate["position_flu"], dtype=np.float64)
        error_m = float(np.linalg.norm(position - self.world_target))
        # measured 0.03 m: the box centre and the depth are pixel-quantized, so
        # the bound only has to catch a wrong range (a model-distance fallback
        # or the euclidean range instead of the axial one move it by > 0.1 m)
        self.assertLess(error_m, 0.2, f"grounded {position} vs truth {self.world_target}")
        self.assertEqual(
            ctx.extra[CANDIDATE_STORE_KEY][candidate["id"]]["position_flu"],
            candidate["position_flu"],
        )

        # the chain the runs lost: the stored candidate id resolves to that
        # position and the intent tool for "the object in the frame" uses it
        registry = ToolRegistry()
        register_navigation_tools(registry)
        moved = registry.call("approach_candidate", {"candidate_id": candidate["id"]}, ctx)

        self.assertEqual(moved.status, "ok")
        self.assertEqual(moved.data["candidate_id"], candidate["id"])
        self.assertAlmostEqual(
            float(np.linalg.norm(np.asarray(moved.data["target_position"]) - position)),
            0.0,
            places=4,
        )
        self.assertGreater(moved.data["progress_m"], 0.5)
        self.assertLess(
            float(np.linalg.norm(world.position_flu - self.world_target)), 3.0
        )


if __name__ == "__main__":
    unittest.main()
