"""Resource-profile driver contract (``scripts/profile_resources.py``).

The driver is the comparable measurement layer for the A100/4090 question: it
records peak allocated/reserved memory, first-call vs steady-state latency, the
per-turn token rate, and it audits every episode summary for the fields that
make a silent OOM fallback visible. These tests pin the pure/reporting half so
the GPU sections cannot quietly change shape:

- latency summaries (first/steady/p95) and quantization arithmetic;
- the mandatory-summary-field audit (complete summary passes, a legacy one
  reports exactly what it lacks);
- replay dry-run through the CLI (no model, no GPU);
- the Markdown/JSON report writer.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "profile_resources.py"


def _load_module():
    name = "diffaero_script_profile_resources"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve string annotations via sys.modules
    spec.loader.exec_module(module)
    return module


PROFILE = _load_module()


def complete_summary() -> dict:
    return {
        "instruction": "find the chair",
        "success": False,
        "finish_reason": "max_turns",
        "runtime": {
            "backend": {
                "kind": "hf",
                "model": "/models_ro/Qwen3-VL-8B",
                "device": "cuda:0",
                "dtype": "bfloat16",
                "quantization": "none",
                "image_size": [640, 360],
                "max_new_tokens": 192,
            },
            "detector": {
                "requested": "sam3_dino",
                "active": True,
                "reason": "",
                "device": "cuda:0",
                "precision": "fp32",
            },
            "renderer": {
                "backend": "higs_ut",
                "device": "cuda:0",
                "width": 320,
                "height": 180,
                "extension_cache_key": "diffaero_higs_a0_e44503d14e1c",
            },
        },
        "resources": {
            "peak_allocated_mib": 18300.0,
            "peak_reserved_mib": 20100.0,
            "peak_driver_used_mib": 21500.0,
        },
        "fallbacks": [],
    }


class TimingSummaryTest(unittest.TestCase):
    def test_first_call_and_steady_state_are_separate(self):
        summary = PROFILE.summarize_timings([2.0, 1.0, 1.2, 0.8, 1.1])
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["first_s"], 2.0)
        self.assertAlmostEqual(summary["steady_mean_s"], (1.0 + 1.2 + 0.8 + 1.1) / 4)
        self.assertAlmostEqual(summary["steady_median_s"], 1.05)
        self.assertAlmostEqual(summary["min_s"], 0.8)
        self.assertAlmostEqual(summary["max_s"], 2.0)
        self.assertGreaterEqual(summary["p95_s"], summary["steady_median_s"])

    def test_single_call_is_not_reported_as_steady(self):
        summary = PROFILE.summarize_timings([3.5])
        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["first_s"], 3.5)
        self.assertIsNone(summary["steady_mean_s"])

    def test_empty_timings(self):
        summary = PROFILE.summarize_timings([])
        self.assertEqual(summary["count"], 0)
        self.assertIsNone(summary["first_s"])

    def test_tokens_per_second(self):
        self.assertAlmostEqual(PROFILE.tokens_per_second(32, 2.0), 16.0)
        self.assertIsNone(PROFILE.tokens_per_second(0, 2.0))
        self.assertIsNone(PROFILE.tokens_per_second(10, 0.0))


class QuantizationEstimateTest(unittest.TestCase):
    def test_eight_billion_parameter_conversions(self):
        estimate = PROFILE.quantization_estimate(8_000_000_000, overhead_gib=1.0)
        self.assertAlmostEqual(estimate["bf16_weights_gib"], 14.901, places=2)
        self.assertAlmostEqual(estimate["int8_weights_gib"], 7.451, places=2)
        self.assertAlmostEqual(estimate["int4_weights_gib"], 3.725, places=2)
        self.assertAlmostEqual(estimate["bf16_with_overhead_gib"], 15.901, places=2)
        self.assertAlmostEqual(estimate["int8_with_overhead_gib"], 8.451, places=2)
        self.assertAlmostEqual(estimate["int4_with_overhead_gib"], 4.725, places=2)
        # a 24 GB 4090 leaves room for activations/vision towers either way
        self.assertLess(estimate["int8_with_overhead_gib"], 24.0)
        self.assertLess(estimate["int4_with_overhead_gib"], 24.0)

    def test_quantization_is_labelled_as_estimated(self):
        estimate = PROFILE.quantization_estimate(8_000_000_000)
        self.assertEqual(estimate["kind"], "estimate")
        self.assertIn("bitsandbytes", estimate["note"].lower())


class SummaryAuditTest(unittest.TestCase):
    def test_complete_summary_passes(self):
        audit = PROFILE.audit_episode_summary(complete_summary())
        self.assertTrue(audit["ok"], audit)
        self.assertEqual(audit["missing"], [])

    def test_legacy_summary_reports_the_missing_fallbacks(self):
        legacy = {
            "instruction": "find the chair",
            "success": False,
            "finish_reason": "max_turns",
            "stack_backend": "modules",
        }
        audit = PROFILE.audit_episode_summary(legacy)
        self.assertFalse(audit["ok"])
        fields = {item["field"] for item in audit["missing"]}
        self.assertIn("runtime.backend.kind", fields)
        self.assertIn("runtime.backend.quantization", fields)
        self.assertIn("runtime.detector.active", fields)
        self.assertIn("runtime.renderer.backend", fields)
        self.assertIn("resources.peak_allocated_mib", fields)
        self.assertIn("fallbacks", fields)

    def test_a_legacy_stack_backend_is_reported_as_a_silent_fallback(self):
        summary = complete_summary()
        summary["stack_backend"] = "fallback"
        audit = PROFILE.audit_episode_summary(summary)
        self.assertFalse(audit["ok"])
        self.assertIn("stack_backend=fallback", json.dumps(audit))

    def test_null_values_count_as_missing(self):
        summary = complete_summary()
        summary["runtime"]["backend"]["device"] = None
        audit = PROFILE.audit_episode_summary(summary)
        self.assertIn("runtime.backend.device", {item["field"] for item in audit["missing"]})

    def test_audit_walks_a_summary_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a").mkdir()
            (root / "a" / "summary.json").write_text(json.dumps(complete_summary()))
            (root / "b").mkdir()
            (root / "b" / "summary.json").write_text(
                json.dumps({"instruction": "x", "stack_backend": "fallback"})
            )
            audit = PROFILE.audit_summary_paths([root])
        self.assertEqual(audit["summaries"], 2)
        self.assertEqual(audit["complete"], 1)
        self.assertEqual(len(audit["incomplete"]), 1)

    def test_required_fields_are_documented(self):
        fields = PROFILE.REQUIRED_SUMMARY_FIELDS
        self.assertIn("runtime.backend.kind", fields)
        self.assertIn("resources.peak_allocated_mib", fields)
        self.assertIn("fallbacks", fields)


class ReportWriterTest(unittest.TestCase):
    def test_writes_json_and_markdown(self):
        results = {
            "meta": {
                "device": "NVIDIA A100-SXM4-80GB",
                "cuda_visible_devices": "0",
                "torch": "2.7.1+cu128",
            },
            "vlm": {
                "load": {"seconds": 12.5, "peak_allocated_mib": 16000.0,
                         "peak_reserved_mib": 17000.0},
                "single_frame": {"count": 5, "first_s": 1.4, "steady_median_s": 0.9,
                                 "p95_s": 1.3, "tokens_per_s": 21.0},
                "eight_views": {"count": 3, "first_s": 2.9, "steady_median_s": 2.4,
                                "p95_s": 2.8, "tokens_per_s": 8.0},
            },
            "audit": {"summaries": 25, "complete": 0, "incomplete": [{"path": "x"}]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            report = PROFILE.write_report(tmp, results, title="Resource Profile")
            markdown = (Path(tmp) / "report.md").read_text(encoding="utf-8")
            payload = json.loads((Path(tmp) / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["markdown"], str(Path(tmp) / "report.md"))
        self.assertIn("# Resource Profile", markdown)
        self.assertIn("Peak allocated", markdown)
        self.assertIn("steady", markdown.lower())
        self.assertIn("OOM", markdown)
        self.assertEqual(payload["schema"], PROFILE.REPORT_SCHEMA)
        self.assertEqual(payload["results"]["vlm"]["load"]["seconds"], 12.5)

    def test_merge_keeps_one_section_from_each_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "vlm.json"
            second = Path(tmp) / "render.json"
            first.write_text(json.dumps({
                "schema": PROFILE.REPORT_SCHEMA,
                "results": {
                    "meta": {"gpu_name": "A100"},
                    "vlm": {"load": {"seconds": 6.5}},
                    "requested_sections": ["vlm"],
                },
            }))
            second.write_text(json.dumps({
                "schema": PROFILE.REPORT_SCHEMA,
                "results": {
                    "meta": {"gpu_name": "A100"},
                    "render": {"status": "ok"},
                    "requested_sections": ["render"],
                },
            }))
            merged = PROFILE.merge_reports([first, second])
        self.assertEqual(set(merged) & {"vlm", "render"}, {"vlm", "render"})
        self.assertEqual(merged["meta"]["gpu_name"], "A100")
        self.assertEqual(merged["sections_isolated"]["vlm"], str(first))
        self.assertNotIn("requested_sections", merged)


class CliDryRunTest(unittest.TestCase):
    def test_replay_dry_run_needs_no_gpu(self):
        from tests.test_replay_frames import write_replay

        with tempfile.TemporaryDirectory() as tmp:
            replay_dir = Path(tmp) / "replay"
            out = Path(tmp) / "out"
            write_replay(replay_dir)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.profile_resources",
                    "--sections",
                    "replay",
                    "--replay-dir",
                    str(replay_dir),
                    "--dry-run",
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["results"]["replay"]["frame_count"], 12)
            self.assertIn("window_chair", (out / "report.md").read_text(encoding="utf-8"))

    def test_parser_has_the_measurement_switches(self):
        parser = PROFILE.build_parser()
        args = parser.parse_args(
            ["--sections", "vlm,replay", "--replay", "--device", "cuda:0",
             "--repeat", "3", "--out", "x"]
        )
        self.assertTrue(args.replay)
        self.assertEqual(args.device, "cuda:0")
        self.assertEqual(args.repeat, 3)
        self.assertEqual(args.sections, "vlm,replay")


if __name__ == "__main__":
    unittest.main()
