"""Batch runner over DiffAero scene+object+obstacle combination episodes.

These tests pin the parts that do not need a GPU or a vLLM endpoint: the
subprocess command shape, the matrix reader, and the aggregation of per-episode
summaries (including failures and timeouts).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_combo_harness.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "diffaero_script_run_combo_harness", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


harness = _load_module()

SPEC = {
    "name": "warehouse_crate_seed_1",
    "preset": "warehouse_crate",
    "scene": "warehouse",
    "combined_ply": "/assets/combo.spz",
    "semantic_manifest": "/assets/combo.json",
    "target_index": 0,
    "target_name": "wooden crate",
    "instruction": "Find the wooden crate and fly next to it.",
    "success_dist": 1.8,
    "target_edge_distance_m": 7.3,
}


def _args(**overrides):
    values = {
        "python": "/usr/bin/python3",
        "model": "qwen3-vl-8b",
        "base_url": "http://127.0.0.1:8000/v1",
        "image_size": (320, 180),
        "max_turns": 30,
        "tool_budget": 60,
        "timeout_s": 30.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class CommandTest(unittest.TestCase):
    def test_command_carries_manifest_and_auto_start(self) -> None:
        command = harness.episode_command(
            SPEC, "/tmp/run/ep", _args(runner="tools")
        )
        self.assertEqual(command[1], "scripts/run_vlm_nav_tools.py")
        self.assertIn("--semantic-manifest", command)
        self.assertIn("--auto-start", command)
        self.assertIn("--target-index", command)
        self.assertEqual(command[command.index("--target-index") + 1], "0")
        self.assertEqual(command[command.index("--render-size") + 1:][:2],
                         ["320", "180"])
        self.assertEqual(command[command.index("--success-dist") + 1], "1.8")

    def test_missing_required_field_is_reported(self) -> None:
        broken = dict(SPEC)
        broken.pop("semantic_manifest")
        with self.assertRaises(KeyError):
            harness.episode_command(broken, "/tmp/run/ep", _args())


class MatrixTest(unittest.TestCase):
    def test_reads_episodes_object_and_bare_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.json"
            path.write_text(json.dumps({"episodes": [SPEC]}), encoding="utf-8")
            self.assertEqual(harness.load_matrix(path), [SPEC])
            path.write_text(json.dumps([SPEC]), encoding="utf-8")
            self.assertEqual(harness.load_matrix(path), [SPEC])

    def test_rejects_bad_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.json"
            path.write_text(json.dumps({"nope": []}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                harness.load_matrix(path)

    def test_episode_name_must_be_one_component(self) -> None:
        self.assertEqual(harness.episode_name(SPEC), "warehouse_crate_seed_1")
        with self.assertRaises(SystemExit):
            harness.episode_name({"name": "../escape"})


class AggregationTest(unittest.TestCase):
    def test_success_and_evidence_are_counted_separately(self) -> None:
        rows = [
            {"success": True, "success_evidence": True, "wall_time_s": 10.0},
            {"success": True, "success_evidence": False, "wall_time_s": 20.0},
            {"success": False, "success_evidence": False, "wall_time_s": None},
        ]
        totals = harness.aggregate_results(rows)
        self.assertEqual(totals["episodes"], 3)
        self.assertEqual(totals["successes"], 2)
        self.assertEqual(totals["evidence_backed"], 1)
        self.assertAlmostEqual(totals["mean_wall_time_s"], 15.0)

    def test_empty_rows(self) -> None:
        self.assertEqual(
            harness.aggregate_results([]),
            {
                "episodes": 0,
                "successes": 0,
                "evidence_backed": 0,
                "mean_wall_time_s": 0.0,
                "mean_position_error_m": None,
            },
        )

    def test_mean_position_error_ignores_missing_values(self) -> None:
        totals = harness.aggregate_results([
            {"success": True, "position_error_m": 0.5},
            {"success": False, "position_error_m": None},
            {"success": True, "position_error_m": 1.5},
        ])
        self.assertAlmostEqual(totals["mean_position_error_m"], 1.0)

    def test_nonzero_exit_forces_failure_and_records_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            (out_dir / "summary.json").write_text(
                json.dumps({
                    "success": True,
                    "success_evidence": True,
                    "finish_reason": "finish",
                    "wall_time_s": 4.0,
                }),
                encoding="utf-8",
            )
            row = harness.episode_row(SPEC, out_dir, 124, "timeout after 30s", 30.0)
        self.assertFalse(row["success"])
        self.assertFalse(row["success_evidence"])
        self.assertEqual(row["exit_code"], 124)
        self.assertIn("timeout", row["error"])

    def test_successful_summary_is_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            (out_dir / "summary.json").write_text(
                json.dumps({
                    "success": True,
                    "success_evidence": True,
                    "finish_reason": "finish",
                    "final_distance_m": 1.2,
                    "tool_calls": 9,
                    "turns": 5,
                    "vlm_calls": 7,
                    "wall_time_s": 12.5,
                }),
                encoding="utf-8",
            )
            row = harness.episode_row(SPEC, out_dir, 0)
        self.assertTrue(row["success"])
        self.assertTrue(row["success_evidence"])
        self.assertEqual(row["tool_calls"], 9)
        self.assertEqual(row["target_name"], "wooden crate")
        self.assertEqual(row["target_edge_distance_m"], 7.3)
        self.assertIsNone(row["error"])

    def test_demo_summary_is_normalised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            (out_dir / "summary.json").write_text(
                json.dumps({
                    "candidate_position": [1.0, 2.0, 3.0],
                    "target_ground_truth": {"position_flu": [1.0, 2.0, 3.5]},
                    "results": {
                        "finish": {
                            "status": "ok",
                            "data": {
                                "success": True,
                                "verified": True,
                                "claim": "verified",
                                "reason": "done",
                            },
                        },
                        "verify": {"data": {"measured_distance_m": 1.5}},
                    },
                    "wall_time_s": 8.0,
                }),
                encoding="utf-8",
            )
            row = harness.episode_row(SPEC, out_dir, 0)
        self.assertTrue(row["success"])
        self.assertTrue(row["success_evidence"])
        self.assertEqual(row["finish_reason"], "verified")
        self.assertAlmostEqual(row["final_distance_m"], 1.5)
        self.assertAlmostEqual(row["position_error_m"], 0.5)

    def test_demo_command_uses_manifest_and_demo_runner(self) -> None:
        command = harness.episode_command(
            SPEC, "/tmp/run/ep", _args(runner="demo", device="cuda:1")
        )
        self.assertEqual(command[1], "scripts/run_semantic_corner_demo.py")
        self.assertIn("--manifest", command)
        self.assertNotIn("--ply", command)
        self.assertEqual(command[command.index("--device") + 1], "cuda:1")
        self.assertEqual(command[command.index("--prompt") + 1], SPEC["instruction"])


if __name__ == "__main__":
    unittest.main()
