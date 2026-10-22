"""DiffAero scene+object+obstacle manifests as harness episodes.

The frame contract is the one that matters: the manifest stores object
positions in scene-centred FLU, but its placement diagnostics keep the raw
3DGS bounding box and ground plane. Mixing the two shifts the floor by the
merge centre, which makes an auto-start pose fly below the room and rejects
valid floor placements.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vlm_nav.scene_combo import (
    choose_start_pose,
    combo_summary,
    load_scene_combo,
    scene_flu_points,
)


def _manifest(path: Path, *, ground_y: float = -1.5,
              center_y: float = 0.5, with_global_ground: bool = True) -> Path:
    objects = [{
        "key": "wooden_crate",
        "name": "wooden crate",
        "path": "/assets/objects/wooden_crate.ply",
        "scale": 1.0,
        "role": "target",
    }]
    instance = {
        "key": "wooden_crate",
        "name": "wooden crate",
        "role": "target",
        "position_flu": [-1.0, 4.0, -3.0],
        "xy_radius_flu": 0.4,
        "bbox_min_flu": [-1.4, 3.6, -3.5],
        "bbox_max_flu": [-0.6, 4.4, -2.5],
    }
    if with_global_ground:
        instance["placement_global_ground_z_flu"] = ground_y - center_y
    payload = {
        "merged_ply": str(path.with_suffix(".spz")),
        "background_ply": str(path.with_name(path.stem + "_background.spz")),
        "scene_scale": 1.0,
        "merged_scene_center_3dgs": [1.0, center_y, 2.0],
        "source_scene": {"key": "warehouse", "name": "warehouse"},
        "source_objects": objects,
        "source_obstacles": [{
            "key": "barrier_gate",
            "name": "barrier gate",
            "path": "/assets/obstacles/barrier_gate.ply",
            "scale": 1.0,
            "role": "obstacle",
        }],
        "object_positions_flu": [[-1.0, 4.0, -3.0]],
        "object_xy_radii_flu": [0.4],
        "object_bbox_mins_flu": [[-1.4, 3.6, -3.5]],
        "object_bbox_maxs_flu": [[-0.6, 4.4, -2.5]],
        "object_instances": [instance],
        "obstacle_positions_flu": [[-6.0, -5.0, -3.0]],
        "obstacle_xy_radii_flu": [0.5],
        "obstacle_bbox_mins_flu": [[-6.5, -5.5, -4.0]],
        "obstacle_bbox_maxs_flu": [[-5.5, -4.5, -2.0]],
        # absolute 3DGS bounds: 8 m x 12 m, centred later by the merge centre
        "placement": {
            "ground_y": ground_y,
            "bbox_min_xz": [-3.0, -4.0],
            "bbox_max_xz": [5.0, 8.0],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class LoadSceneComboTest(unittest.TestCase):
    def test_parses_entities_and_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _manifest(Path(directory) / "combo.json")
            combo = load_scene_combo(path)
        self.assertEqual(combo.scene_key, "warehouse")
        self.assertEqual(len(combo.objects), 1)
        self.assertEqual(len(combo.obstacles), 1)
        self.assertEqual(combo.objects[0].name, "wooden crate")
        self.assertEqual(combo.obstacles[0].kind, "obstacle")
        self.assertTrue(combo.merged_ply.is_absolute())
        self.assertTrue(combo.merged_ply.name.endswith(".spz"))

    def test_bounds_and_ground_are_scene_centred(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            combo = load_scene_combo(_manifest(Path(directory) / "combo.json"))
        # 3DGS x [-3, 5] - centre_x 1 -> rel [-4, 4] -> FLU y [-4, 4]
        # 3DGS z [-4, 8] - centre_z 2 -> rel [-6, 6] -> FLU x [-6, 6]
        (min_x, min_y), (max_x, max_y) = combo.xy_bounds_flu()
        self.assertAlmostEqual(min_x, -6.0)
        self.assertAlmostEqual(max_x, 6.0)
        self.assertAlmostEqual(min_y, -4.0)
        self.assertAlmostEqual(max_y, 4.0)
        # ground_y -1.5 - centre_y 0.5 == -2.0
        self.assertAlmostEqual(combo.ground_z_flu(), -2.0)

    def test_ground_falls_back_to_centre_subtraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            combo = load_scene_combo(
                _manifest(Path(directory) / "combo.json",
                          with_global_ground=False)
            )
        self.assertAlmostEqual(combo.ground_z_flu(), -2.0)

    def test_edge_distance_uses_the_flipped_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            combo = load_scene_combo(_manifest(Path(directory) / "combo.json"))
        # target FLU (-1, 4); box x [-6, 6], y [-4, 4]
        self.assertAlmostEqual(combo.edge_distance_m(0), 0.0)

    def test_summary_is_json_friendly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            combo = load_scene_combo(_manifest(Path(directory) / "combo.json"))
        summary = combo_summary(combo)
        self.assertEqual(summary["scene_key"], "warehouse")
        self.assertEqual(summary["num_objects"], 1)
        self.assertEqual(summary["num_obstacles"], 1)
        json.dumps(summary)


class StartPoseTest(unittest.TestCase):
    def test_start_is_in_bounds_above_floor_and_faces_away(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            combo = load_scene_combo(_manifest(Path(directory) / "combo.json"))
        (x, y, z), yaw_deg = choose_start_pose(combo, inward_m=3.0)
        (min_x, min_y), (max_x, max_y) = combo.xy_bounds_flu()
        self.assertTrue(min_x < x < max_x)
        self.assertTrue(min_y < y < max_y)
        self.assertAlmostEqual(z, combo.ground_z_flu() + 1.2)
        target_x, target_y = combo.objects[0].position_flu[:2]
        # the camera forward vector must point away from the target
        forward = np.asarray([np.cos(np.radians(yaw_deg)),
                              np.sin(np.radians(yaw_deg))])
        to_target = np.asarray([target_x - x, target_y - y])
        self.assertLess(float(forward @ to_target), 0.0)


class FluPointsTest(unittest.TestCase):
    def test_matches_the_merge_convention(self) -> None:
        xyz = np.asarray([[12.0, 25.0, 27.0], [11.0, 20.0, 30.0]])
        center = np.asarray([10.0, 20.0, 30.0])
        points = scene_flu_points(xyz, center)
        # rel (2, 5, -3) -> (-rel_z, -rel_x, rel_y) = (3, -2, 5)
        np.testing.assert_allclose(points[0], [3.0, -2.0, 5.0])
        np.testing.assert_allclose(points[1], [0.0, -1.0, 0.0])

    def test_accepts_torch_like_tensors(self) -> None:
        class FakeTensor:
            def __init__(self, values):
                self.values = np.asarray(values, dtype=np.float32)

            def detach(self):
                return self

            def float(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.values

        points = scene_flu_points(
            FakeTensor([[10.0, 20.0, 30.0]]),
            FakeTensor([10.0, 20.0, 30.0]),
        )
        np.testing.assert_allclose(points, [[0.0, 0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
