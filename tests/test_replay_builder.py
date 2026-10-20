"""The window-chair replay builder (``scripts/build_replay_window_chair.py``).

The builder turns one frozen scene + the holdout manifest into the replay set
the resource profiles share. These tests pin the view plan (at least 12 views,
level scan + pitched views + approach views, every yaw pointing at the chair)
and the manifest it writes, without needing the GPU renderer.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_replay_window_chair.py"


def _load_module():
    name = "diffaero_script_build_replay_window_chair"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = _load_module()
TARGET = (2.457343768699508, 2.8264845233543716, -2.372089195470126)


class ViewPlanTest(unittest.TestCase):
    def test_plan_has_at_least_twelve_unique_views(self):
        views = BUILDER.plan_window_chair_views(TARGET)
        self.assertGreaterEqual(len(views), 12)
        self.assertEqual(len({view["id"] for view in views}), len(views))
        self.assertEqual(len({tuple(view["pose_flu"]) for view in views}), len(views))

    def test_plan_mixes_approach_scan_and_pitched_views(self):
        views = BUILDER.plan_window_chair_views(TARGET)
        phases = {view["phase"] for view in views}
        self.assertEqual(phases, {"approach", "scan", "pitched"})
        self.assertGreaterEqual(sum(1 for view in views if view["phase"] == "scan"), 8)
        self.assertGreaterEqual(sum(1 for view in views if view["phase"] == "pitched"), 2)

    def test_level_views_face_the_chair(self):
        views = BUILDER.plan_window_chair_views(TARGET)
        for view in views:
            if view["phase"] == "pitched":
                continue
            x, y, _ = view["pose_flu"]
            expected = math.degrees(math.atan2(TARGET[1] - y, TARGET[0] - x))
            delta = (view["yaw_deg"] - expected + 180.0) % 360.0 - 180.0
            self.assertLess(abs(delta), 1e-6, view)

    def test_pitched_views_look_down(self):
        views = BUILDER.plan_window_chair_views(TARGET)
        pitched = [view for view in views if view["phase"] == "pitched"]
        self.assertTrue(pitched)
        for view in pitched:
            self.assertGreater(view["pitch_deg"], 0.0)
            self.assertLessEqual(view["pitch_deg"], 60.0)

    def test_views_stay_inside_a_reasonable_radius(self):
        views = BUILDER.plan_window_chair_views(TARGET, radius_m=2.4)
        for view in views:
            distance = math.dist(view["pose_flu"], TARGET)
            self.assertGreaterEqual(distance, 1.0)
            self.assertLessEqual(distance, 6.0)


class ManifestTest(unittest.TestCase):
    def _write_frames(self, root: Path, views):
        for view in views:
            path = root / view["file"]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (320, 180), (12, 34, 56)).save(path)
            view["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def test_manifest_round_trips_through_the_loader(self):
        from scripts.replay_frames import load_replay

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "window_chair"
            views = BUILDER.plan_window_chair_views(TARGET)
            for index, view in enumerate(views):
                view["file"] = f"frames/view_{index:02d}.png"
            self._write_frames(root, views)
            manifest = BUILDER.build_manifest(
                name="window_chair",
                instruction="在卧室窗边找到木椅",
                views=views,
                camera=BUILDER.DEFAULT_CAMERA,
                scene={
                    "target_name": "wooden spindle back chair",
                    "target_position_flu": list(TARGET),
                    "source": "unit-test",
                },
                generator={"backend": "unit-test", "command": "unit-test"},
            )
            BUILDER.write_manifest(root, manifest)
            replay = load_replay(root)
        self.assertEqual(len(replay.views), len(views))
        self.assertEqual(replay.name, "window_chair")
        self.assertEqual(replay.scene["target_position_flu"], list(TARGET))
        self.assertEqual(replay.camera.width, 320)

    def test_manifest_lists_every_view_with_pose_and_intrinsics(self):
        views = BUILDER.plan_window_chair_views(TARGET)
        for index, view in enumerate(views):
            view["file"] = f"frames/view_{index:02d}.png"
        manifest = BUILDER.build_manifest(
            name="window_chair",
            instruction="find the chair",
            views=views,
            camera=BUILDER.DEFAULT_CAMERA,
            scene={"target_position_flu": list(TARGET)},
            generator={"backend": "unit-test"},
        )
        self.assertEqual(manifest["schema"], "vlm_nav.replay.v1")
        self.assertEqual(len(manifest["views"]), len(views))
        for view in manifest["views"]:
            self.assertEqual(set(view["intrinsics"]), {"width", "height", "hfov_deg", "fx", "fy", "cx", "cy"})
            self.assertEqual(len(view["pose_flu"]), 3)
            self.assertIn("yaw_deg", view)
            self.assertIn("pitch_deg", view)


class DefaultAssetTest(unittest.TestCase):
    def test_default_scene_and_manifest_exist_in_this_checkout(self):
        if not BUILDER.DEFAULT_MANIFEST.is_file():
            self.skipTest("holdout bedroom manifest is not in this checkout")
        data = json.loads(BUILDER.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        self.assertIn("object_positions_flu", data)
        self.assertEqual(data["source_scene"]["key"], "bedroom")
        self.assertTrue(BUILDER.DEFAULT_COMBINED_SPZ.is_file() or BUILDER.DEFAULT_COMBINED_PLY.is_file())


if __name__ == "__main__":
    unittest.main()
