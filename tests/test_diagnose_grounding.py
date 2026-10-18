"""Tests for the perception-failure diagnostic bench (``scripts/diagnose_grounding.py``).

The bench exists because a real trajectory answered ``no_candidate`` over and over
and nobody could tell whether the model had not seen the target, had written
something the parser rejected, was looking at another frame or had its box thrown
away by the projection. These tests pin the contract of the answer: a fixed set of
observations covering the four situations, one disk record per model call (raw
reply + the image that was sent + a failure category) and diagnosis-only
replacements that never enter a score.

No model, no CUDA, no 3DGS data: the scene is a fake geometry (a target and a slab
occluder) rendered by a fake world, and the "model" is the repository's own mock
plus the bench's keyed mock. The one thing the fake world must get right is the
camera semantics the real worlds have - yaw-only frames, a depth map of the
current pose and ``apply_action`` waypoints - because the bench audits exactly
those.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSER_PATH = ROOT / "scripts" / "diagnose_grounding.py"


def _load_diagnoser():
    name = "diffaero_script_diagnose_grounding"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, DIAGNOSER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


D = _load_diagnoser()

WIDTH, HEIGHT = 96, 54
HFOV_DEG = 60.0
TARGET_POS = (3.0, 0.0, 0.0)
TARGET_RADIUS = 0.25
START_POS = (0.0, 0.0, 0.0)
#: a slab perpendicular to y at y = -1: it leaves the start pose's view of the
#: target clear, hides the target from the loss pose and cuts it in half from a
#: viewpoint beside its edge
WALL = D.Occluder(center=(1.0, -1.0, 0.0), size=(6.0, 0.2, 3.0))
LOSS_POSE = (0.0, -3.0, 0.0)


def fake_scene(**overrides):
    data = dict(
        ply_path="",
        target_pos=TARGET_POS,
        target_radius=TARGET_RADIUS,
        start_pos=START_POS,
        start_yaw_deg=0.0,
        hfov_deg=HFOV_DEG,
        occluders=[WALL],
        loss_pose=LOSS_POSE,
        target_name="red object",
    )
    data.update(overrides)
    return D.SceneSpec(**data)


def _ray_box_2d(origin, direction, lo, hi, eps=1e-9):
    """Near distance at which a 2D ray enters an axis-aligned box, or None."""
    t0, t1 = 0.0, float("inf")
    for axis in range(2):
        a, delta = float(origin[axis]), float(direction[axis])
        if abs(delta) < eps:
            if a < lo[axis] or a > hi[axis]:
                return None
            continue
        ta = (lo[axis] - a) / delta
        tb = (hi[axis] - a) / delta
        if ta > tb:
            ta, tb = tb, ta
        t0, t1 = max(t0, ta), min(t1, tb)
        if t0 > t1:
            return None
    return t0


class FakeWorld:
    """Kinematic CPU world with the three things the bench reads.

    ``render`` draws the visible part of the target in red at the projection the
    answer key computes (so the frozen frame and the key agree by construction),
    ``depth_at_pixel`` returns the distance of the target plane near the target
    and NaN elsewhere, and ``apply_action`` implements the documented waypoint
    semantics (turn, then translate along the new heading).
    """

    def __init__(self, scene, size):
        self.scene = scene
        self.width, self.height = int(size[0]), int(size[1])
        self.state = type("State", (), {})()
        self.state.position_flu = np.asarray(scene.start_pos, dtype=np.float64)
        self.state.yaw_rad = math.radians(float(scene.start_yaw_deg))
        self.actions = []

    @property
    def position_flu(self):
        return self.state.position_flu

    @property
    def yaw_rad(self):
        return self.state.yaw_rad

    def _intrinsics(self):
        return D.Intrinsics(self.width, self.height, self.scene.hfov_deg)

    def render(self):
        intrinsics = self._intrinsics()
        frame = torch.zeros(3, self.height, self.width)
        frame[2] = 0.3
        truth = D.target_ground_truth(
            self.scene.target_pos, self.scene.target_radius,
            tuple(self.state.position_flu), math.degrees(self.state.yaw_rad),
            intrinsics, self.scene.occluders,
        )
        box = truth.visible_box
        if box is not None:
            x0 = max(0, int(box[0] / 1000.0 * self.width))
            x1 = min(self.width, max(x0 + 1, int(box[2] / 1000.0 * self.width)))
            y0 = max(0, int(box[1] / 1000.0 * self.height))
            y1 = min(self.height, max(y0 + 1, int(box[3] / 1000.0 * self.height)))
            frame[0, y0:y1, x0:x1] = 0.95
            frame[1, y0:y1, x0:x1] = 0.1
            frame[2, y0:y1, x0:x1] = 0.1
        return frame

    def _target_pixel(self):
        intrinsics = self._intrinsics()
        return D.project_point(
            self.scene.target_pos, tuple(self.state.position_flu),
            math.degrees(self.state.yaw_rad), intrinsics,
        )

    def _depth_occluders(self):
        """The boxes the depth map hides things with (splats render wider)."""
        return self.scene.occluders

    def depth_at_pixel(self, pixel):
        """Axial distance of the nearest surface along that pixel's ray.

        A little raycaster over the same geometry the answer key uses (the
        occluder slabs and the target sphere), so the rendered depth *is* the
        picture the model was shown: without it a fake depth that ignores the
        wall would make the answer key disagree with the observation and the
        occlusion cases would grade the model against a target it cannot see.
        """
        intrinsics = self._intrinsics()
        if intrinsics.fx is None or intrinsics.fy is None:
            return float("nan")
        pixel_x = float(pixel[0]) / 1000.0 * intrinsics.width
        pixel_y = float(pixel[1]) / 1000.0 * intrinsics.height
        left = -(pixel_x - intrinsics.width / 2.0) / intrinsics.fx
        up = -(pixel_y - intrinsics.height / 2.0) / intrinsics.fy
        yaw = self.state.yaw_rad
        direction = np.asarray([
            math.cos(yaw) - math.sin(yaw) * left,
            math.sin(yaw) + math.cos(yaw) * left,
        ])
        direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
        origin = np.asarray(self.state.position_flu, dtype=np.float64)[:2]
        distances = []
        for box in self._depth_occluders():
            hit = _ray_box_2d(origin, direction, np.asarray(box.lo, dtype=np.float64)[:2],
                              np.asarray(box.hi, dtype=np.float64)[:2])
            if hit is not None:
                distances.append(hit)
        target = np.asarray(self.scene.target_pos, dtype=np.float64)
        centre = target[:2] - origin
        along = float(centre @ direction)
        if along > 0.0:
            closest = origin + along * direction
            if float(np.linalg.norm(target[:2] - closest)) <= self.scene.target_radius:
                distances.append(math.hypot(along, float(target[2] - self.state.position_flu[2])))
        if not distances:
            return float("nan")
        return float(min(distances))

    def apply_action(self, action):
        self.actions.append(action)
        self.state.yaw_rad += math.radians(float(action.yaw_deg))
        distance = float(action.forward_m)
        self.state.position_flu = self.state.position_flu + np.asarray(
            [math.cos(self.state.yaw_rad) * distance,
             math.sin(self.state.yaw_rad) * distance,
             float(action.up_m)]
        )
        self.last_subgoal = self.state.position_flu.copy()
        return f"moved to {np.round(self.state.position_flu, 2).tolist()}"


def fake_world_factory(scene, size):
    return FakeWorld(scene, size)


class BlurredOccluderWorld(FakeWorld):
    """The occluder renders wider than its box - what a splat scene does.

    Every slab is inflated by ``blur_m`` before the depth raycast, so the hidden
    region of the *picture* extends past the geometric boundary: exactly the
    situation in which an answer key built on the box alone would ask the model
    for a target that is not in the image.
    """

    blur_m = 0.1

    def _depth_occluders(self):
        return [
            D.Occluder(box.center, tuple(size + 2.0 * self.blur_m for size in box.size))
            for box in self.scene.occluders
        ]


def build_cases(directory: Path, scene=None, **overrides):
    """The bench's own fixed set built on the fake scene (no 3DGS, no CUDA)."""
    scene = scene or fake_scene(**overrides)
    cases, notes = D.build_cases_from_scene(
        scene, directory, render_size=(WIDTH, HEIGHT),
        instruction="the red object", world_factory=fake_world_factory,
        cases_dir=directory / "observations",
    )
    D.save_observations(
        cases, directory / "observations" / "manifest.json", scene=scene.to_dict(),
        source="fake", notes=notes,
    )
    return cases, scene


def case_by_kind(cases, kind):
    for case in cases:
        if case.kind == kind:
            return case
    raise AssertionError(f"no '{kind}' case in the fixed set: {[c.kind for c in cases]}")


def run(cases, backend, directory, *, scene=None, executor=None, **config_kwargs):
    """Run the bench and return ``(summary, records)``."""
    config = D.BenchConfig(
        out_dir=directory,
        world_factory=fake_world_factory,
        log=None,
        **config_kwargs,
    )
    summary = D.run_bench(
        cases, backend, config, scene=scene if scene is not None else fake_scene(),
        executor=executor,
    )
    records = D._read_calls(Path(summary["calls_path"]))
    return summary, records


def records_of(records, **fields):
    return [
        record for record in records
        if all(record.get(key) == value for key, value in fields.items())
    ]


class FakeDetector(D.DetectorBackend):
    """A wired detector: returns a box shifted off the answer key by ``offset``."""

    name = "fake"

    def __init__(self, offset=0.0, error=None):
        self.offset = float(offset)
        self.error = error
        self.calls = []

    def available(self):
        return True

    def detect(self, frame, instruction, *, case=None):
        self.calls.append((tuple(frame.shape), instruction))
        if self.error:
            raise self.error
        box = case.target_box
        if box is None:
            return []
        return [D.Detection([box[0] + self.offset, box[1] + self.offset,
                             box[2] + self.offset, box[3] + self.offset], 0.8, "fake")]


class FixedSetTests(unittest.TestCase):
    """The fixed observation set: four situations, on disk, with an answer key."""

    def test_scene_cases_cover_the_four_situations(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases, _scene = build_cases(Path(tmp))
            self.assertEqual({case.kind for case in cases}, set(D.CASE_KINDS))

            revealed = case_by_kind(cases, D.CASE_VISIBLE)
            absent = case_by_kind(cases, D.CASE_ABSENT)
            partial = case_by_kind(cases, D.CASE_PARTIAL)
            small = case_by_kind(cases, D.CASE_SMALL)

            self.assertTrue(revealed.truth.visible)
            self.assertGreaterEqual(revealed.truth.visible_fraction, 0.9)
            self.assertTrue(revealed.target_box and len(revealed.target_box) == 4)

            self.assertFalse(absent.truth.visible)
            self.assertIsNone(absent.target_box, "an invisible target has nothing to localize")

            self.assertTrue(partial.truth.visible)
            self.assertGreaterEqual(partial.truth.occluded_fraction, 0.15,
                                    "the partial case must be cut by the occluder, not the frame")
            self.assertLess(partial.truth.visible_fraction, 0.9)

            box = small.target_box
            self.assertIsNotNone(box)
            width_px = (box[2] - box[0]) / 1000.0 * WIDTH
            self.assertLess(width_px, 0.25 * WIDTH, "the small case is a small target")
            self.assertGreater(small.truth.distance_m, revealed.truth.distance_m)

            for case in cases:
                image = Path(case.image_path)
                self.assertTrue(image.is_file(), f"{case.case_id} has no frozen frame")
                self.assertEqual(case.intrinsics["hfov_deg"], HFOV_DEG)
                self.assertTrue(all(math.isfinite(v) for v in case.pose_flu))
                self.assertGreater(case.render.get("nonblack_px", 0), 0,
                                   "a frozen frame must be a picture")

    def test_manifest_round_trip_keeps_frames_and_answer_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases, _scene = build_cases(Path(tmp))
            loaded, manifest = D.load_observations(Path(tmp) / "observations" / "manifest.json")
            self.assertEqual(manifest["format"], D.MANIFEST_FORMAT)
            self.assertEqual(manifest["missing_kinds"], [])
            self.assertEqual([case.case_id for case in loaded], [case.case_id for case in cases])
            for original, restored in zip(cases, loaded):
                self.assertEqual(restored.truth.to_dict(), original.truth.to_dict())
                self.assertEqual(restored.target_box, D._round_box(original.target_box))
                self.assertEqual(restored.intrinsics, original.intrinsics)
                self.assertTrue(
                    D.frames_equal(restored.load_frame(), original.load_frame()),
                    "the fixed set must re-read the same pixels",
                )

    def test_answer_key_uses_the_rendered_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases, _scene = build_cases(Path(tmp))
            for case in cases:
                self.assertEqual(case.truth.source, "analytic_projection+rendered_depth")
            revealed = case_by_kind(cases, D.CASE_VISIBLE)
            self.assertTrue(revealed.render.get("depth_matches_target"),
                            "the depth at the visible target must be the target's own")
            self.assertAlmostEqual(revealed.render["depth_at_target_m"], revealed.truth.distance_m,
                                   delta=0.3)

    def test_partial_case_is_partial_in_the_rendered_picture(self):
        """A rendered occluder hides more than its box; the answer key follows the picture."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = fake_scene()
            intrinsics = D.Intrinsics(WIDTH, HEIGHT, HFOV_DEG)
            cases, notes = D.build_cases_from_scene(
                scene, root, render_size=(WIDTH, HEIGHT), instruction="the red object",
                world_factory=lambda s, size: BlurredOccluderWorld(s, size),
                cases_dir=root / "observations",
            )
            self.assertEqual({case.kind for case in cases}, set(D.CASE_KINDS))
            partial = case_by_kind(cases, D.CASE_PARTIAL)
            self.assertTrue(partial.truth.visible)
            self.assertGreaterEqual(partial.truth.visible_fraction, 0.15)
            self.assertLessEqual(partial.truth.visible_fraction, 0.65)
            self.assertGreaterEqual(partial.truth.occluded_fraction, 0.15)
            self.assertIn("rendered:", partial.note)
            self.assertEqual(partial.truth.source, "analytic_projection+rendered_depth")

            # the same pose under the geometric test is more visible than the picture:
            # the answer key has to be the rendered one, not the box
            analytic = D.target_ground_truth(
                scene.target_pos, scene.target_radius, partial.pose_flu, partial.yaw_deg,
                intrinsics, scene.occluders,
            )
            world = BlurredOccluderWorld(scene, (WIDTH, HEIGHT))
            D.set_world_pose(world, partial.pose_flu, partial.yaw_deg)
            rendered = D.target_ground_truth(
                scene.target_pos, scene.target_radius, partial.pose_flu, partial.yaw_deg,
                intrinsics, scene.occluders, depth_fn=world.depth_at_pixel,
            )
            self.assertGreater(analytic.visible_fraction, rendered.visible_fraction + 0.05,
                               "the rendered occluder hides more than its geometric box")

    def test_unknown_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"format": "something/else", "cases": []}))
            with self.assertRaises(ValueError):
                D.load_observations(path)

    def test_probe_bundle_becomes_a_fixed_set_without_rendering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "frames_post_fix"
            frames.mkdir()
            scene = fake_scene()
            # a frame rendered at the start pose: the probe bundle's own picture
            intr = D.Intrinsics(WIDTH, HEIGHT, HFOV_DEG)
            truth = D.target_ground_truth(
                scene.target_pos, scene.target_radius, START_POS, 0.0, intr, scene.occluders
            )
            frame = torch.zeros(3, HEIGHT, WIDTH)
            box = truth.visible_box
            x0 = int(box[0] / 1000.0 * WIDTH)
            x1 = max(x0 + 1, int(box[2] / 1000.0 * WIDTH))
            y0 = int(box[1] / 1000.0 * HEIGHT)
            y1 = max(y0 + 1, int(box[3] / 1000.0 * HEIGHT))
            frame[0, y0:y1, x0:x1] = 1.0
            D.frame_to_png(frame, frames / "start_yaw0.png")
            # the loss view: the target is behind the wall, so its own frame keeps
            # only the background (and the slab, drawn dark)
            hidden = torch.zeros(3, HEIGHT, WIDTH)
            hidden[2] = 0.3
            D.frame_to_png(hidden, frames / "loss_yaw180.png")
            probe = {
                "label": "post_fix",
                "render": {"width": WIDTH, "height": HEIGHT, "hfov_deg": HFOV_DEG},
                "poses": [
                    {"tag": "start_yaw0", "yaw_deg": 0.0, "pose": list(START_POS)},
                    {"tag": "loss_yaw180", "yaw_deg": 180.0, "pose": list(LOSS_POSE)},
                ],
            }
            (root / "probe.json").write_text(json.dumps(probe))
            scenario_path = root / "scenario.json"
            scenario_path.write_text(json.dumps({
                "scenario": {
                    "target_name": "red object",
                    "target_pos": list(scene.target_pos),
                    "target_radius": scene.target_radius,
                    "start_pos": list(scene.start_pos),
                    "start_yaw_deg": 0.0,
                    "hfov_deg": HFOV_DEG,
                    "occluders": [box.to_dict() for box in scene.occluders],
                    "loss_bearing_deg": -90.0,
                    "loss_distance_m": 3.0,
                },
                "scene_ply": "",
                "scene_origin_flu": [0.0, 0.0, 0.0],
            }))
            cases, scene_out, notes = D.build_cases_from_probe(
                root / "probe.json", scenario_path, frames_dir=frames, instruction="the red object"
            )
            self.assertEqual(len(cases), 2)
            self.assertEqual(cases[0].kind, D.CASE_VISIBLE)
            self.assertEqual(cases[1].kind, D.CASE_ABSENT)
            self.assertTrue(cases[0].truth.visible)
            self.assertFalse(cases[1].truth.visible)
            self.assertIsNotNone(cases[0].target_box)
            self.assertIsNone(cases[1].target_box)
            self.assertTrue(any("partial" in note for note in notes),
                            "the probe bundle cannot cover every situation and says so")
            self.assertEqual(scene_out.target_pos, scene.target_pos)


class GeometryTests(unittest.TestCase):
    """The answer key itself: projection, occlusion, FOV deviation."""

    def test_projection_and_occlusion(self):
        intrinsics = D.Intrinsics(WIDTH, HEIGHT, HFOV_DEG)
        centre = D.project_point(TARGET_POS, START_POS, 0.0, intrinsics)
        self.assertAlmostEqual(centre[0], 500.0, places=3)
        self.assertAlmostEqual(centre[1], 500.0, places=3)

        visible = D.target_ground_truth(
            TARGET_POS, TARGET_RADIUS, START_POS, 0.0, intrinsics, [WALL]
        )
        self.assertTrue(visible.visible)
        self.assertAlmostEqual(visible.visible_fraction, 1.0)
        self.assertEqual(visible.occluded_fraction, 0.0)
        self.assertAlmostEqual(visible.distance_m, 3.0, places=3)

        blocked = D.target_ground_truth(
            TARGET_POS, TARGET_RADIUS, LOSS_POSE, 45.0, intrinsics, [WALL]
        )
        self.assertFalse(blocked.visible)
        self.assertAlmostEqual(blocked.occluded_fraction, 1.0)
        self.assertIsNone(blocked.visible_box)

        behind = D.target_ground_truth(
            TARGET_POS, TARGET_RADIUS, (4.0, 0.0, 0.0), 0.0, intrinsics, []
        )
        self.assertFalse(behind.in_fov)
        self.assertFalse(behind.visible)

    def test_unknown_field_of_view_refuses_the_projection(self):
        blind = D.Intrinsics(WIDTH, HEIGHT, None)
        self.assertIsNone(D.project_point(TARGET_POS, START_POS, 0.0, blind))
        self.assertIsNone(D.pixel_ray((500.0, 500.0), START_POS, 0.0, blind))
        self.assertIsNone(D.pixel_to_world((500.0, 500.0), 3.0, START_POS, 0.0, blind))
        self.assertTrue(D.frame_sha256(torch.zeros(3, 4, 4)))

    def test_fov_deviation_is_reported_for_off_axis_pixels(self):
        intrinsics = D.Intrinsics(WIDTH, HEIGHT, HFOV_DEG)
        on_axis = D.fov_deviation_m((500.0, 500.0), 3.0, START_POS, 0.0, intrinsics)
        off_axis = D.fov_deviation_m((800.0, 500.0), 3.0, START_POS, 0.0, intrinsics)
        self.assertAlmostEqual(on_axis, 0.0, places=6)
        self.assertGreater(off_axis, 0.2,
                           "the 86 deg default must visibly move a pointed world point")

    def test_box_metrics(self):
        intrinsics = D.Intrinsics(WIDTH, HEIGHT, HFOV_DEG)
        truth = D.target_ground_truth(
            TARGET_POS, TARGET_RADIUS, START_POS, 0.0, intrinsics, [WALL]
        )
        self.assertAlmostEqual(D.box_center_error_px(truth.visible_box, truth.visible_box, intrinsics), 0.0)
        self.assertAlmostEqual(D.box_iou(truth.visible_box, truth.visible_box), 1.0)
        shifted = [truth.visible_box[0] + 100, truth.visible_box[1],
                   truth.visible_box[2] + 100, truth.visible_box[3]]
        self.assertGreater(D.box_center_error_px(shifted, truth.visible_box, intrinsics), 5.0)
        self.assertIsNone(D.box_center_error_px(None, truth.visible_box, intrinsics))


class FailureTaxonomyTests(unittest.TestCase):
    """``evaluate_answer`` is where a failure gets its name (pure function)."""

    def setUp(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases, _scene = build_cases(Path(tmp))
        self.cases = cases
        self.visible = case_by_kind(cases, D.CASE_VISIBLE)
        self.absent = case_by_kind(cases, D.CASE_ABSENT)

    def test_backend_error_is_other(self):
        result = D.evaluate_answer(self.visible, parsed_ok=False, backend_error="RuntimeError: boom")
        self.assertEqual(result["failure"]["category"], D.FAILURE_OTHER)
        self.assertEqual(result["outcome"], D.OUTCOME_ERROR)
        self.assertIn("boom", result["failure"]["detail"])

    def test_unparsable_reply_is_output_format_not_a_miss_reason(self):
        result = D.evaluate_answer(self.visible, parsed_ok=False, tool_reason="unparsable_reply")
        self.assertEqual(result["failure"]["category"], D.FAILURE_NOT_SEEN)
        self.assertIn("unparsable_reply", result["failure"]["detail"])

    def test_abstention_on_an_invisible_target_is_not_a_failure(self):
        result = D.evaluate_answer(self.absent, parsed_ok=False, model_abstained=True)
        self.assertEqual(result["failure"]["category"], D.FAILURE_NONE)
        self.assertEqual(result["outcome"], D.OUTCOME_ABSTAIN)
        self.assertEqual(result["failure"]["detail"], "abstain_correct")

    def test_box_on_an_invisible_target_is_a_false_positive(self):
        result = D.evaluate_answer(
            self.absent, answer_box=[400.0, 400.0, 600.0, 600.0], parsed_ok=True,
            intrinsics_used=self.absent.intrinsics,
        )
        self.assertEqual(result["failure"]["category"], D.FAILURE_OTHER)
        self.assertEqual(result["outcome"], D.OUTCOME_FALSE_POSITIVE)

    def test_box_far_from_the_target_is_not_seen(self):
        result = D.evaluate_answer(
            self.visible, answer_box=[0.0, 0.0, 50.0, 50.0], parsed_ok=True,
            intrinsics_used=self.visible.intrinsics,
        )
        self.assertEqual(result["failure"]["category"], D.FAILURE_NOT_SEEN)
        self.assertEqual(result["outcome"], D.OUTCOME_MISS)

    def test_missing_position_with_depth_available_is_a_projection_error(self):
        box = self.visible.target_box
        result = D.evaluate_answer(
            self.visible, answer_box=box, parsed_ok=True, intrinsics_used=self.visible.intrinsics,
            position_flu=None, position_source="ray", range_status="unknown",
            ray={"bearing_deg": 0.0}, expect_position=True,
        )
        self.assertEqual(result["failure"]["category"], D.FAILURE_PROJECTION)
        self.assertIn("no_position_despite_range", result["failure"]["detail"])
        self.assertEqual(result["outcome"], D.OUTCOME_MISS)

    def test_position_without_a_range_is_a_projection_error(self):
        box = self.visible.target_box
        result = D.evaluate_answer(
            self.visible, answer_box=box, parsed_ok=True, intrinsics_used=self.visible.intrinsics,
            position_flu=self.visible.truth.position_flu, position_source="depth",
            range_status="unknown", expect_position=False,
        )
        self.assertEqual(result["failure"]["category"], D.FAILURE_PROJECTION)
        self.assertIn("position_without_range", result["failure"]["detail"])

    def test_wrong_field_of_view_in_the_answer_is_a_projection_error(self):
        box = self.visible.target_box
        result = D.evaluate_answer(
            self.visible, answer_box=box, parsed_ok=True,
            intrinsics_used={"width": WIDTH, "height": HEIGHT, "hfov_deg": 86.0},
            position_flu=self.visible.truth.position_flu, position_source="depth",
            range_status="known", expect_position=True,
        )
        self.assertEqual(result["failure"]["category"], D.FAILURE_PROJECTION)
        self.assertIn("intrinsics_mismatch", result["failure"]["detail"])
        self.assertFalse(result["measure"]["intrinsics_match"])

    def test_position_with_known_range_and_the_observation_intrinsics_is_a_hit(self):
        box = self.visible.target_box
        result = D.evaluate_answer(
            self.visible, answer_box=box, parsed_ok=True, intrinsics_used=self.visible.intrinsics,
            position_flu=self.visible.truth.position_flu, position_source="depth",
            range_status="known", range_m=3.0, expect_position=True,
        )
        self.assertEqual(result["failure"]["category"], D.FAILURE_NONE)
        self.assertEqual(result["outcome"], D.OUTCOME_HIT)
        self.assertEqual(result["measure"]["position_error_m"], 0.0)

    def test_ray_is_accepted_when_no_depth_is_expected(self):
        box = self.visible.target_box
        result = D.evaluate_answer(
            self.visible, answer_box=box, parsed_ok=True, intrinsics_used=self.visible.intrinsics,
            position_flu=None, position_source="ray", range_status="unknown",
            ray={"bearing_deg": 0.0, "origin_flu": [0.0, 0.0, 0.0]}, expect_position=False,
        )
        self.assertEqual(result["failure"]["category"], D.FAILURE_NONE)
        self.assertEqual(result["outcome"], D.OUTCOME_HIT)


class HarnessTests(unittest.TestCase):
    """End-to-end runs with a scripted or oracle backend, and what they write."""

    def test_oracle_self_test_is_green_and_fully_audited(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases, scene = build_cases(root)
            summary, records = run(cases, D.OracleReplyVLM(), root / "run", scene=scene)

            self.assertEqual(summary["format"], D.RUN_FORMAT)
            self.assertTrue(summary["backend"]["oracle_replies"])
            for strategy in (D.STRATEGY_STRUCTURED, D.STRATEGY_POINT, D.STRATEGY_REGION):
                row = summary["strategies"][strategy]
                self.assertEqual(row["hits"], 3, f"{strategy} should localize the visible targets")
                self.assertEqual(row["abstain_correct"], 1)
                self.assertEqual(row["misses"], 0)
                self.assertEqual(row["failure_categories"], {D.FAILURE_NONE: 4})
            self.assertEqual(summary["strategies"][D.STRATEGY_DETECTOR]["not_run"], 4)
            self.assertEqual(summary["strategies"][D.STRATEGY_DETECTOR]["hits"], 0)

            self.assertGreater(len(records), 10)
            for record in records:
                image = record["input_image"]
                self.assertTrue(image["path"], "every call must dump the image it sent")
                self.assertTrue(Path(image["path"]).is_file())
                self.assertTrue(image["sha256"])
                self.assertTrue(image["matches_expected"], record["call"])
                if record["backend"] == "no_model_call" or str(record["backend"]).startswith("detector:"):
                    self.assertIsNone(record["raw_reply"])
                else:
                    self.assertIsNotNone(record["raw_reply"])
                self.assertIsNotNone(record["outcome"])
                self.assertIsNotNone(record["failure"])
            self.assertEqual(summary["audit"]["image_mismatches"], 0)
            self.assertEqual(set(summary["audit"]["box_formats_used"]), {"xyxy_norm"})
            self.assertGreaterEqual(summary["audit"]["box_formats_used"]["xyxy_norm"], 6,
                                    "one recorded box format per answered call")
            self.assertEqual(summary["audit"]["intrinsics_mismatches"], 0)
            self.assertEqual(summary["audit"]["positions_without_range"], 0)
            self.assertGreater(summary["audit"]["positions_with_known_range"], 0)
            self.assertGreaterEqual(len(summary["audit"]["fov_deviations_m"]), 1)

            report = D.render_report(summary, records)
            self.assertIn("Perception-failure bench", report)
            self.assertIn("Fixed observation set", report)
            self.assertIn("Failure categories", report)
            self.assertIn("HARNESS SELF-TEST", report)
            for case in cases:
                self.assertIn(case.case_id, report)

    def test_scripted_mock_answers_are_locked_per_case_and_step(self):
        """The repository's own mock drives the bench; the taxonomy names the reply."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases, scene = build_cases(root)
            visible = case_by_kind(cases, D.CASE_VISIBLE)
            box = visible.target_box
            good = {"candidates": [{"box": box, "box_format": "xyxy", "confidence": 0.9}]}
            prose = {"candidates": []}
            replies = {
                f"{visible.case_id}|{D.STRATEGY_STRUCTURED}|ground_target": good,
                "anything|else": prose,
            }
            summary, records = run(cases, D.KeyedMockVLM(replies), root / "run", scene=scene,
                                   strategies=(D.STRATEGY_STRUCTURED,))

            answers = records_of(records, case=visible.case_id)
            self.assertEqual(len(answers), 1)
            self.assertEqual(answers[0]["outcome"], D.OUTCOME_HIT)
            self.assertEqual(answers[0]["failure"]["category"], D.FAILURE_NONE)
            self.assertEqual(answers[0]["measure"]["center_error_px"], 0.0)

            # the cases the keyed mock has no reply for answer with the empty default
            other = [record for record in records if record["case"] != visible.case_id]
            self.assertEqual(len(other), len(cases) - 1)
            for record in other:
                self.assertIn(record["outcome"], (D.OUTCOME_ABSTAIN, D.OUTCOME_MISS))
                if record["kind"] == D.CASE_ABSENT:
                    self.assertEqual(record["failure"]["category"], D.FAILURE_NONE)
                    self.assertEqual(record["outcome"], D.OUTCOME_ABSTAIN)
                else:
                    self.assertEqual(record["failure"]["category"], D.FAILURE_NOT_SEEN)
            self.assertIn(D.FAILURE_NONE, summary["failure_categories"])

    def test_repo_mock_vlm_drives_a_single_case(self):
        from vlm_nav.vlm import MockVLM

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases, scene = build_cases(root)
            visible = case_by_kind(cases, D.CASE_VISIBLE)
            backend = MockVLM(mode="script", script=[{
                "candidates": [{"box": visible.target_box, "box_format": "xyxy", "confidence": 0.8}]
            }])
            summary, records = run([visible], backend, root / "run", scene=scene,
                                   strategies=(D.STRATEGY_STRUCTURED,))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["outcome"], D.OUTCOME_HIT)
            self.assertEqual(summary["strategies"][D.STRATEGY_STRUCTURED]["hits"], 1)

    def test_scripted_prose_reply_is_recorded_as_output_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases, scene = build_cases(root)
            visible = case_by_kind(cases, D.CASE_VISIBLE)
            replies = {
                f"{visible.case_id}|{D.STRATEGY_STRUCTURED}|ground_target":
                    "I cannot see anything like that in the image.",
            }
            _summary, records = run(cases, D.KeyedMockVLM(replies), root / "run", scene=scene,
                                    strategies=(D.STRATEGY_STRUCTURED,))
            record = records_of(records, case=visible.case_id)[0]
            self.assertIn("I cannot see anything", record["raw_reply"])
            self.assertEqual(record["tool"]["reason"], "unparsable_reply")
            self.assertEqual(record["failure"]["category"], D.FAILURE_NOT_SEEN)
            self.assertTrue(record["failure"]["detail"].startswith("no_usable_answer"))
            self.assertEqual(record["outcome"], D.OUTCOME_MISS)

    def test_no_candidate_is_split_by_what_the_observation_really_holds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases, scene = build_cases(root)
            empty = {"candidates": []}
            replies = {f"{case.case_id}|{D.STRATEGY_STRUCTURED}|ground_target": empty
                       for case in cases}
            summary, records = run(cases, D.KeyedMockVLM(replies), root / "run", scene=scene,
                                   strategies=(D.STRATEGY_STRUCTURED,))
            by_kind = {record["kind"]: record for record in records}
            self.assertEqual(by_kind[D.CASE_ABSENT]["failure"]["category"], D.FAILURE_NONE)
            self.assertEqual(by_kind[D.CASE_ABSENT]["outcome"], D.OUTCOME_ABSTAIN)
            for kind in (D.CASE_VISIBLE, D.CASE_PARTIAL, D.CASE_SMALL):
                self.assertEqual(by_kind[kind]["failure"]["category"], D.FAILURE_NOT_SEEN)
                self.assertEqual(by_kind[kind]["failure"]["detail"], "model_reported_absent")
            self.assertEqual(summary["failure_categories"][D.FAILURE_NOT_SEEN], 3)
            self.assertEqual(summary["failure_categories"][D.FAILURE_NONE], 1)

    def test_point_strategy_projects_through_the_observation_intrinsics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases, scene = build_cases(root)
            visible = case_by_kind(cases, D.CASE_VISIBLE)
            box = visible.target_box
            centre = [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]
            replies = {
                f"{visible.case_id}|{D.STRATEGY_POINT}|point":
                    {"point": centre, "visible": True, "distance_m": 3.0},
            }
            _summary, records = run(cases, D.KeyedMockVLM(replies), root / "run", scene=scene,
                                    strategies=(D.STRATEGY_POINT,))
            record = records_of(records, case=visible.case_id)[0]
            self.assertEqual(record["outcome"], D.OUTCOME_HIT)
            projection = record["parsed"]["projection"]
            self.assertEqual(projection["range_status"], "known")
            self.assertEqual(projection["position_source"], "depth")
            error = float(np.linalg.norm(
                np.asarray(projection["position_flu"]) - np.asarray(visible.truth.position_flu)
            ))
            self.assertLess(error, 0.2, "a correct pixel must project onto the target")
            # the audit recomputes what the legacy 86 deg default would have added
            self.assertIsNotNone(projection["fov_deviation_at_1m_m"])

    def test_point_strategy_off_target_pixel_is_a_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases, scene = build_cases(root)
            visible = case_by_kind(cases, D.CASE_VISIBLE)
            box = visible.target_box
            centre = [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]
            far = [min(1000.0, centre[0] + 300.0), centre[1]]
            replies = {
                f"{visible.case_id}|{D.STRATEGY_POINT}|point": {"point": far, "visible": True},
            }
            _summary, records = run(cases, D.KeyedMockVLM(replies), root / "run", scene=scene,
                                    strategies=(D.STRATEGY_POINT,),
                                    tolerances=D.Tolerances(center_px=20.0))
            record = records_of(records, case=visible.case_id)[0]
            self.assertEqual(record["outcome"], D.OUTCOME_MISS)
            self.assertEqual(record["failure"]["category"], D.FAILURE_NOT_SEEN)
            self.assertGreater(record["measure"]["center_error_px"], 20.0)

    def test_region_then_local_chain_maps_the_crop_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases, scene = build_cases(root)
            visible = case_by_kind(cases, D.CASE_VISIBLE)
            box = visible.target_box
            # a region the model proposes, and the same target written in the
            # crop's own 0-1000 frame (computed here, not by the module)
            region = [max(0.0, box[0] - 100), max(0.0, box[1] - 100),
                      min(1000.0, box[2] + 100), min(1000.0, box[3] + 100)]
            x0 = region[0] / 1000.0 * WIDTH
            crop_w = (region[2] - region[0]) / 1000.0 * WIDTH
            y0 = region[1] / 1000.0 * HEIGHT
            crop_h = (region[3] - region[1]) / 1000.0 * HEIGHT
            crop_box = [
                (box[0] / 1000.0 * WIDTH - x0) / crop_w * 1000.0,
                (box[1] / 1000.0 * HEIGHT - y0) / crop_h * 1000.0,
                (box[2] / 1000.0 * WIDTH - x0) / crop_w * 1000.0,
                (box[3] / 1000.0 * HEIGHT - y0) / crop_h * 1000.0,
            ]
            replies = {
                f"{visible.case_id}|{D.STRATEGY_REGION}|region":
                    {"region": region, "visible": True},
                f"{visible.case_id}|{D.STRATEGY_REGION}|ground_local":
                    {"candidates": [{"box": crop_box, "box_format": "xyxy", "confidence": 0.9}]},
            }
            _summary, records = run(cases, D.KeyedMockVLM(replies), root / "run", scene=scene,
                                    strategies=(D.STRATEGY_REGION,))
            chosen = records_of(records, case=visible.case_id)
            self.assertEqual([record["step"] for record in chosen], ["region", "crop", "ground_local"])
            region_record, crop_record, local_record = chosen
            self.assertFalse(region_record["final"])
            self.assertEqual(region_record["outcome"], D.OUTCOME_INTERMEDIATE)
            self.assertFalse(crop_record["final"])
            self.assertTrue(local_record["final"])
            self.assertEqual(local_record["outcome"], D.OUTCOME_HIT)
            source_box = local_record["parsed"]["source_box"]
            expected = [round(v, 0) for v in box]
            for got, want in zip(source_box, expected):
                self.assertAlmostEqual(got, want, delta=25.0)
            self.assertTrue(local_record["input_image"]["matches_expected"])
            self.assertEqual(local_record["input_image"]["expected_source"],
                             "crop_of_fixed_observation")

    def test_wrong_frame_fault_injection_is_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases, scene = build_cases(root)
            good = {"candidates": [{"box": cases[0].target_box or [0, 0, 10, 10],
                                    "box_format": "xyxy", "confidence": 0.9}]}
            replies = {f"{case.case_id}|{D.STRATEGY_STRUCTURED}|ground_target": good
                       for case in cases}
            summary, records = run(cases, D.KeyedMockVLM(replies), root / "run", scene=scene,
                                   strategies=(D.STRATEGY_STRUCTURED,),
                                   inject_wrong_image=True)
            self.assertTrue(records)
            for record in records:
                self.assertTrue(record["diagnosis_only"],
                                "a fault injection must not look like a score")
                self.assertFalse(record["input_image"]["matches_expected"])
                self.assertTrue(record["input_image"]["injected"])
                self.assertEqual(record["failure"]["category"], D.FAILURE_WRONG_IMAGE)
                self.assertEqual(record["outcome"], D.OUTCOME_MISS)
            self.assertEqual(summary["audit"]["image_mismatches"], len(records))
            self.assertEqual(summary["strategies"][D.STRATEGY_STRUCTURED]["calls"], 0)
            report = D.render_report(summary, records)
            self.assertIn("diagnosis-only", report.lower())

    def test_injected_point_checks_the_projection_and_injected_waypoint_the_executor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases, scene = build_cases(root)
            visible = case_by_kind(cases, D.CASE_VISIBLE)
            empty = {"candidates": []}
            replies = {f"{case.case_id}|{D.STRATEGY_STRUCTURED}|ground_target": empty
                       for case in cases}
            summary, records = run([visible], D.KeyedMockVLM(replies), root / "run", scene=scene,
                                   strategies=(D.STRATEGY_STRUCTURED,),
                                   inject_point="gt", inject_waypoint="gt")
            projected = records_of(records, strategy="inject_point")[0]
            self.assertTrue(projected["diagnosis_only"])
            self.assertEqual(projected["outcome"], D.OUTCOME_HIT)
            self.assertLess(projected["measure"]["position_error_m"], 0.25)
            self.assertTrue(projected["failure"]["detail"].startswith("injected_point:"))

            flown = records_of(records, strategy="inject_waypoint")[0]
            self.assertTrue(flown["diagnosis_only"])
            self.assertEqual(flown["outcome"], D.OUTCOME_HIT)
            self.assertEqual(flown["measure"]["executor_status"], "arrived")
            self.assertLessEqual(flown["measure"]["position_error_m"],
                                 summary["tolerances"]["position_m"])
            self.assertGreaterEqual(flown["measure"]["legs"], 1)

            # the injections stay out of the score
            self.assertEqual(summary["strategies"][D.STRATEGY_STRUCTURED]["calls"], 1)
            self.assertEqual(summary["diagnosis_only"]["calls"], 2)
            report = D.render_report(summary, records)
            self.assertIn("Diagnosis-only replacements", report)

    def test_injected_waypoint_reports_a_broken_executor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases, scene = build_cases(root)
            visible = case_by_kind(cases, D.CASE_VISIBLE)
            empty = {"candidates": []}

            def stalled_executor(world, target):
                start = [float(v) for v in world.position_flu]
                return D.ExecutorResult(
                    target_flu=[float(v) for v in target], start_flu=start, final_flu=start,
                    error_m=float(np.linalg.norm(np.asarray(start) - np.asarray(list(target)))),
                    legs=0, status="stalled", feedback=["no movement"],
                )

            replies = {f"{visible.case_id}|{D.STRATEGY_STRUCTURED}|ground_target": empty}
            _summary, records = run([visible], D.KeyedMockVLM(replies), root / "run", scene=scene,
                                    strategies=(D.STRATEGY_STRUCTURED,),
                                    inject_waypoint="gt", executor=stalled_executor)
            flown = records_of(records, strategy="inject_waypoint")[0]
            self.assertEqual(flown["outcome"], D.OUTCOME_MISS)
            self.assertIn("executor_error_m", flown["failure"]["detail"])
            self.assertIn("stalled", flown["failure"]["detail"])

    def test_detector_slot_is_empty_by_default_and_wired_when_a_backend_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases, scene = build_cases(root)
            visible = case_by_kind(cases, D.CASE_VISIBLE)
            empty = {"candidates": []}
            replies = {f"{visible.case_id}|{D.STRATEGY_STRUCTURED}|ground_target": empty}

            _summary, records = run([visible], D.KeyedMockVLM(replies), root / "a", scene=scene,
                                    strategies=(D.STRATEGY_DETECTOR,))
            slot = records_of(records, strategy=D.STRATEGY_DETECTOR)[0]
            self.assertEqual(slot["outcome"], D.OUTCOME_NOT_RUN)
            self.assertEqual(slot["failure"]["category"], D.FAILURE_SKIPPED)
            self.assertEqual(slot["failure"]["detail"], "detector_slot_empty")

            detector = FakeDetector()
            summary, records = run([visible], D.KeyedMockVLM(replies), root / "b", scene=scene,
                                   strategies=(D.STRATEGY_DETECTOR,), detector=detector)
            wired = records_of(records, strategy=D.STRATEGY_DETECTOR)[0]
            self.assertEqual(wired["outcome"], D.OUTCOME_HIT)
            self.assertEqual(wired["parsed"]["n_detections"], 1)
            self.assertEqual(detector.calls[0][1], visible.instruction)
            self.assertEqual(summary["strategies"][D.STRATEGY_DETECTOR]["hits"], 1)

            broken = FakeDetector(error=RuntimeError("no weights"))
            _summary, records = run([visible], D.KeyedMockVLM(replies), root / "c", scene=scene,
                                    strategies=(D.STRATEGY_DETECTOR,), detector=broken)
            failed = records_of(records, strategy=D.STRATEGY_DETECTOR)[0]
            self.assertEqual(failed["failure"]["category"], D.FAILURE_OTHER)
            self.assertIn("detector_error", failed["failure"]["detail"])

            offset = FakeDetector(offset=250.0)
            _summary, records = run([visible], D.KeyedMockVLM(replies), root / "d", scene=scene,
                                    strategies=(D.STRATEGY_DETECTOR,), detector=offset)
            missed = records_of(records, strategy=D.STRATEGY_DETECTOR)[0]
            self.assertEqual(missed["outcome"], D.OUTCOME_MISS)

    def test_detector_spec_resolution(self):
        self.assertFalse(D.resolve_detector("null").available())
        self.assertTrue(D.resolve_detector("oracle").available())
        with self.assertRaises(ValueError):
            D.resolve_detector("numpy")

    def test_range_check_catches_a_depth_read_at_the_wrong_pixel(self):
        """The crop-coordinate depth bug is named, not left as a distance."""
        with tempfile.TemporaryDirectory() as tmp:
            cases, scene = build_cases(Path(tmp))
            visible = case_by_kind(cases, D.CASE_VISIBLE)
            run = D.prepare_case(visible, world=FakeWorld(scene, (WIDTH, HEIGHT)))
            good, detail = D._range_consistency(run, visible.target_box, 3.0, "depth")
            self.assertTrue(good["matches"])
            self.assertEqual(detail, "")
            bad, detail = D._range_consistency(run, visible.target_box, 1.2, "depth")
            self.assertFalse(bad["matches"])
            self.assertIn("crop coordinates", detail)
            # a box that is not in this observation's frame at all is not a meter
            self.assertEqual(D._range_consistency(run, visible.target_box, 3.0, "ray"), (None, ""))


class CliTests(unittest.TestCase):
    """The CLI path: fixed set from a manifest, plan on disk, report written."""

    def test_manifest_cli_run_writes_every_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases, _scene = build_cases(root)
            replies = {
                f"{case.case_id}|{D.STRATEGY_STRUCTURED}|ground_target":
                    {"candidates": [{"box": case.target_box, "box_format": "xyxy",
                                     "confidence": 0.9}]} if case.target_box else {"candidates": []}
                for case in cases
            }
            replies_path = root / "replies.json"
            replies_path.write_text(json.dumps(replies))
            out_dir = root / "run"
            code = D.main([
                "--source", "manifest",
                "--observations", str(root / "observations"),
                "--backend", "mock",
                "--mock-replies", str(replies_path),
                "--strategies", D.STRATEGY_STRUCTURED,
                "--inject-point", "gt",
                "--out", str(out_dir),
            ])
            self.assertEqual(code, 0)
            for name in ("summary.json", "run_config.json", "calls.jsonl",
                         "diagnose_grounding.md"):
                self.assertTrue((out_dir / name).is_file(), f"{name} was not written")
            summary = json.loads((out_dir / "summary.json").read_text())
            self.assertEqual(summary["format"], D.RUN_FORMAT)
            self.assertEqual(summary["strategies"][D.STRATEGY_STRUCTURED]["calls"], len(cases))
            self.assertEqual(summary["diagnosis_only"]["calls"], 3)
            report = (out_dir / "diagnose_grounding.md").read_text()
            self.assertIn("Fixed observation set", report)
            self.assertIn("DIAGNOSIS", report.upper())
            self.assertIn("python scripts/diagnose_grounding.py", report)
            config = json.loads((out_dir / "run_config.json").read_text())
            self.assertEqual(config["source"], "manifest")
            # every dumped input image is on disk and non-empty
            inputs = sorted((out_dir / "inputs").glob("*.png"))
            self.assertGreaterEqual(len(inputs), len(cases))

    def test_cli_without_a_world_keeps_the_direction_it_knows(self):
        """No PLY means no depth - not "nothing to say".

        The recorded frame still carries the pose it was shot from and its
        intrinsics, so the box keeps a direction from a known place: the honest
        answer is a ray with no position, which the bench counts as a
        ray-only answer instead of filing it as a projection error.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases, _scene = build_cases(root)
            visible = case_by_kind(cases, D.CASE_VISIBLE)
            replies = {f"{visible.case_id}|{D.STRATEGY_STRUCTURED}|ground_target":
                       {"candidates": [{"box": visible.target_box, "box_format": "xyxy"}]}}
            replies_path = root / "replies.json"
            replies_path.write_text(json.dumps(replies))
            out_dir = root / "run"
            code = D.main([
                "--source", "manifest",
                "--observations", str(root / "observations"),
                "--backend", "mock",
                "--mock-replies", str(replies_path),
                "--strategies", D.STRATEGY_STRUCTURED,
                "--max-cases", "1",
                "--out", str(out_dir),
            ])
            self.assertEqual(code, 0)
            summary = json.loads((out_dir / "summary.json").read_text())
            self.assertFalse(summary["world"]["available"])
            # the box was right: no position without a world, but the direction
            # the frame does know is reported (no invented point) and counted
            self.assertEqual(summary["failure_categories"], {D.FAILURE_NONE: 1})
            self.assertEqual(summary["audit"]["ray_only_answers"], 1)
            self.assertNotIn("no_position_and_no_ray", summary["failure_details"])


if __name__ == "__main__":  # pragma: no cover - direct execution
    unittest.main()
