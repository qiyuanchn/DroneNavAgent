"""Replay scan accuracy check: truth, scoring and the contact-sheet path."""

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
SCRIPT = ROOT / "scripts" / "replay_accuracy.py"
TARGET = (2.0, 0.0, 1.0)


def _load_script():
    name = "drone_script_replay_accuracy"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ACCURACY = _load_script()


def _pose_at(bearing_deg: float, distance: float) -> tuple[float, float, float]:
    angle = math.radians(bearing_deg)
    return (
        TARGET[0] + distance * math.cos(angle),
        TARGET[1] + distance * math.sin(angle),
        TARGET[2] + 0.15,
    )


def _yaw_to(pose) -> float:
    return math.degrees(math.atan2(TARGET[1] - pose[1], TARGET[0] - pose[0]))


def _write_replay(root: Path, *, n_scan: int = 8, n_pitched: int = 4) -> Path:
    frames = root / "frames"
    frames.mkdir(parents=True)
    views = []
    for index in range(n_scan + n_pitched):
        pose = _pose_at(360.0 * index / n_scan, 1.5)
        pitch = 0.0 if index < n_scan else 20.0
        frame = Image.new("RGB", (32, 18), (20 * index, 40, 60))
        path = frames / f"view_{index:02d}.png"
        frame.save(path, format="PNG")
        views.append({
            "id": f"view_{index:02d}",
            "file": f"frames/view_{index:02d}.png",
            "pose_flu": [round(v, 6) for v in pose],
            "yaw_deg": round(_yaw_to(pose), 6),
            "pitch_deg": pitch,
            "phase": "scan" if index < n_scan else "pitched",
            "index": index,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    camera = {"width": 32, "height": 18, "hfov_deg": 86.0}
    manifest = {
        "schema": "vlm_nav.replay.v1",
        "name": "synthetic_scan",
        "instruction": "find the target",
        "camera": camera,
        "scene": {"target_name": "target", "target_position_flu": list(TARGET)},
        "generator": {},
        "views": views,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


class _ReplyBackend:
    name = "reply"

    def __init__(self, reply='{"view": 0, "target_visible": true, "distance_m": 1.5}'):
        self.reply = reply
        self.on_log = None
        self.calls: list[dict] = []

    def decide(self, frames, system_prompt, user_prompt, context=None):
        self.calls.append({
            "frames": len(frames) if isinstance(frames, (list, tuple)) else 1,
            "system": system_prompt,
            "user": user_prompt,
            "context": dict(context or {}),
        })
        if callable(self.on_log):
            self.on_log({
                "usage_metrics": {
                    "prompt_tokens": 100,
                    "completion_tokens": 8,
                    "cached_tokens": 40,
                }
            })
        return self.reply


class ReplayAccuracyTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = _write_replay(Path(self._tmp.name))
        self.replay = ACCURACY.load_replay(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_truth_projects_the_target_and_marks_visible_views(self):
        views = ACCURACY.scan_views(self.replay)
        self.assertEqual(len(views), 8)
        truth = ACCURACY.ground_truth(views, TARGET)
        self.assertEqual(truth["best_index"], 0)
        self.assertEqual(truth["visible_indices"], list(range(8)))
        self.assertFalse(truth["exact_available"])  # all views point at the target
        self.assertTrue(all(item["visible"] for item in truth["views"]))

    def test_score_choice(self):
        truth = {"best_index": 2, "visible_indices": [1, 2, 3]}
        self.assertTrue(ACCURACY.score_choice(2, truth, 8)["exact"])
        self.assertTrue(ACCURACY.score_choice(3, truth, 8)["within_one"])
        self.assertTrue(ACCURACY.score_choice(3, truth, 8)["visible"])
        self.assertFalse(ACCURACY.score_choice(4, truth, 8)["visible"])
        self.assertFalse(ACCURACY.score_choice(None, truth, 8)["parse_ok"])

    def test_evaluation_uses_one_sheet_for_the_optimized_call(self):
        backend = _ReplyBackend()
        report = ACCURACY.evaluate(self.replay, backend, repeats=2)
        self.assertEqual(len(backend.calls), 4)  # baseline + optimized, twice
        baseline_calls = [c for i, c in enumerate(backend.calls) if i % 2 == 0]
        optimized_calls = [c for i, c in enumerate(backend.calls) if i % 2 == 1]
        self.assertTrue(all(call["frames"] == 8 for call in baseline_calls))
        self.assertTrue(all(call["frames"] == 1 for call in optimized_calls))
        self.assertTrue(all(call["context"].get("contact_sheet") for call in optimized_calls))
        self.assertEqual(report["baseline"]["summary"]["exact_rate"], 1.0)
        self.assertEqual(report["optimized"]["summary"]["exact_rate"], 1.0)
        self.assertEqual(report["optimized"]["summary"]["images_per_call"], 1)
        self.assertLess(
            report["comparison"]["pixel_ratio"],
            report["comparison"]["image_count_ratio"],
        )
        self.assertEqual(report["optimized"]["summary"]["usage"]["cached_tokens"], 80)
        self.assertTrue(report["comparison"]["not_worse"])

    def test_dry_run_writes_truth_and_sheet(self):
        with tempfile.TemporaryDirectory() as out:
            code = ACCURACY.main([
                "--replay-dir", str(self.root), "--dry-run", "--out", out,
            ])
            self.assertEqual(code, 0)
            report = json.loads(
                (Path(out) / "replay_accuracy.json").read_text()
            )
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["replay"]["truth"]["best_index"], 0)
        self.assertEqual(report["sheet"]["columns"], 4)
        self.assertEqual((report["sheet"]["width"], report["sheet"]["height"]), (74, 56))


if __name__ == "__main__":
    unittest.main()
