"""DiffAero combination builder acceptance rules (no DiffAero import).

``_evaluate_combo`` decides whether a generated combination is a usable edge
episode. It must use the navigable floor extent, not the raw vertex bbox: large
scenes contain distant outliers that would otherwise label every seed central.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_harness_scenes.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "diffaero_script_build_harness_scenes", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


class FakeEntity:
    def __init__(self, position, bbox_min):
        self.position_flu = position
        self.bbox_min_flu = bbox_min


class FakeCombo:
    def __init__(self, *, edge_distance: float, span: float,
                 floor_offset: float = 0.05):
        self._edge = edge_distance
        self._span = span
        self._floor_offset = floor_offset

    def object_at(self, index: int = 0) -> FakeEntity:
        return FakeEntity((0.0, 0.0, 0.0), (0.0, 0.0, self._floor_offset))

    def xy_bounds_flu(self):
        return ((0.0, 0.0), (self._span, self._span))

    def ground_z_flu(self) -> float:
        return 0.0

    def edge_distance_m(self, index: int = 0) -> float:
        return self._edge


def _start(combo, target_index: int = 0):
    return (0.0, 0.0, 1.2), 180.0


class AcceptanceTest(unittest.TestCase):
    def test_relative_edge_fraction_accepts_a_large_room(self) -> None:
        # 7.25 m from the edge of a 32 m navigable span is 0.227: the absolute
        # 2 m rule rejects it, the 0.25 fraction accepts it.
        combo = FakeCombo(edge_distance=7.25, span=32.0)
        _, _, start, yaw, reasons = builder._evaluate_combo(
            combo, edge_margin_m=2.0, edge_fraction=0.25,
            require_floor=True, choose_start_pose=_start,
        )
        self.assertEqual(reasons, [])
        self.assertEqual(start, (0.0, 0.0, 1.2))
        self.assertEqual(yaw, 180.0)

    def test_absolute_margin_still_rejects_a_central_target(self) -> None:
        combo = FakeCombo(edge_distance=12.0, span=32.0)
        _, _, _, _, reasons = builder._evaluate_combo(
            combo, edge_margin_m=2.0, edge_fraction=0.25,
            require_floor=True, choose_start_pose=_start,
        )
        self.assertTrue(reasons)
        self.assertIn("effective_edge_margin_m", reasons[0])

    def test_floor_offset_is_a_separate_rejection(self) -> None:
        combo = FakeCombo(edge_distance=1.0, span=32.0, floor_offset=1.0)
        _, _, _, _, reasons = builder._evaluate_combo(
            combo, edge_margin_m=2.0, edge_fraction=0.25,
            require_floor=True, choose_start_pose=_start,
        )
        self.assertTrue(any("floor_offset_m" in reason for reason in reasons))


class PresetTest(unittest.TestCase):
    def test_presets_name_a_scene_object_and_obstacles(self) -> None:
        self.assertIn("warehouse_crate", builder.PRESETS)
        preset = builder.PRESETS["warehouse_crate"]
        self.assertEqual(preset.scene, "warehouse")
        self.assertEqual(preset.obj, "wooden_crate")
        self.assertTrue(preset.obstacles)
        self.assertIn("wooden crate", preset.instruction)


if __name__ == "__main__":
    unittest.main()
