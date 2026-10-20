"""Tests for the four-configuration comparison driver (goal.md section 7).

Unit tests cover the pieces that define the experiment: the initial-yaw
visibility groups, the paired command construction (same world/scene/start/seed/
budget, only the capability set differs), the metric extraction from both
runners' artifacts (vlm_calls, token estimate, repeat-scan/revisit accounting)
and the aggregation/Markdown report.

The integration test drives ``scripts/compare_agent_configs.py`` end to end as a
subprocess with ``--backend mock`` on a tiny CPU 3DGS scene: the four
configurations (action loop, ``--toolset basic``, ``--toolset memory``,
``--toolset full``) really run and the comparison table is produced. Each
episode is its own interpreter (the runners import the world stack), so the test
runs them with ``--jobs 4`` and a two-turn budget to stay fast.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = ROOT / "scripts" / "compare_agent_configs.py"


def _load_driver():
    name = "diffaero_script_compare_agent_configs"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve string annotations via sys.modules
    spec.loader.exec_module(module)
    return module


DRIVER = _load_driver()

TARGET_FLU = (2.0, 0.5, 0.0)
INSTRUCTION = "Find the target and fly to it."


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _write_test_ply(path: Path, gaussians) -> None:
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
    rows = [list(xyz) + list(sh) + [opacity] + list(log_scale) + list(rot)
            for xyz, sh, opacity, log_scale, rot in gaussians]
    with open(path, "wb") as handle:
        handle.write((header + "\n").encode("ascii"))
        for row in rows:
            handle.write(struct.pack("<14f", *row))


def _make_scene(path: Path) -> None:
    """Tiny CPU 3DGS scene: sparse floor + a red cluster at the target."""
    rng = np.random.default_rng(0)
    gaussians = []
    for _ in range(120):
        point = rng.uniform(-6, 6, size=3)
        point[2] = -1.5
        gaussians.append(
            (
                tuple(float(v) for v in point),
                (0.0, 0.0, 0.0),
                _logit(0.8),
                (math.log(0.2), math.log(0.2), math.log(0.05)),
                (1.0, 0.0, 0.0, 0.0),
            )
        )
    for offset in ((0.0, 0.0, 0.0), (0.15, 0.1, 0.0), (0.0, 0.2, 0.1)):
        gaussians.append(
            (
                tuple(np.asarray(TARGET_FLU) + np.asarray(offset)),
                (2.0, 0.0, 0.0),
                _logit(0.95),
                (math.log(0.25), math.log(0.25), math.log(0.25)),
                (1.0, 0.0, 0.0, 0.0),
            )
        )
    _write_test_ply(path, gaussians)


def _args(*extra: str, **overrides):
    argv = [
        "--world", "cpu",
        "--ply", "/scene.ply",
        "--target-pos", "2", "0.5", "0",
        "--instruction", INSTRUCTION,
        "--out", "/tmp/compare",
        *extra,
    ]
    args = DRIVER.parse_args(argv)
    DRIVER.resolve_matrix(args)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _task(name: str = "book", **overrides) -> "DRIVER.TaskSpec":
    data = dict(
        name=name,
        instruction=INSTRUCTION,
        target_pos=TARGET_FLU,
        target_name="target",
        target_radius=0.5,
        start_pos=(0.0, 0.0, 1.0),
    )
    data.update(overrides)
    return DRIVER.TaskSpec(**data)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# initial-yaw groups (goal.md: keep the visible and invisible task classes)
# ---------------------------------------------------------------------------


class StartYawTest(unittest.TestCase):
    def setUp(self):
        self.task = _task()
        self.bearing = DRIVER.target_bearing_deg(self.task.target_pos, self.task.start_pos)

    def test_visible_group_keeps_the_target_inside_the_fov(self):
        for seed in (0, 7, 11, 13, 101):
            yaw = DRIVER.start_yaw_deg("visible", self.task, seed, half_fov_deg=40.0)
            self.assertLessEqual(abs(DRIVER._wrap_deg(yaw - self.bearing)), 40.0)

    def test_random_group_keeps_the_target_outside_the_fov(self):
        for seed in (0, 7, 11, 13, 101):
            yaw = DRIVER.start_yaw_deg("random", self.task, seed, half_fov_deg=40.0)
            self.assertGreater(abs(DRIVER._wrap_deg(yaw - self.bearing)), 40.0)

    def test_groups_are_deterministic_and_distinct(self):
        for mode in ("visible", "random"):
            self.assertEqual(
                DRIVER.start_yaw_deg(mode, self.task, 7, 40.0),
                DRIVER.start_yaw_deg(mode, self.task, 7, 40.0),
            )
        self.assertNotEqual(
            DRIVER.start_yaw_deg("visible", self.task, 7, 40.0),
            DRIVER.start_yaw_deg("random", self.task, 7, 40.0),
        )

    def test_target_mode_centres_on_the_bearing(self):
        self.assertAlmostEqual(
            DRIVER.start_yaw_deg("target", self.task, 3), round(self.bearing, 3), places=3
        )

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            DRIVER.start_yaw_deg("sideways", self.task, 3)


# ---------------------------------------------------------------------------
# paired command construction + budget mapping
# ---------------------------------------------------------------------------


class CommandBuildTest(unittest.TestCase):
    def _episodes(self, args, tasks=None):
        return DRIVER.build_episodes(args, tasks or [_task()])

    def _commands(self, args, episodes):
        return {
            (spec.config, spec.task.name, spec.seed, spec.yaw_mode):
                DRIVER.build_command(args, spec, DRIVER.episode_dir(Path(args.out), spec))
            for spec in episodes
        }

    def test_matrix_is_configs_x_tasks_x_seeds_x_yaw_modes(self):
        args = _args("--seeds", "7,11", "--initial-yaw-modes", "visible,random")
        tasks = [_task("book"), _task("barrel", target_pos=(1.0, -2.0, 0.4))]
        episodes = self._episodes(args, tasks)
        self.assertEqual(len(episodes), 4 * 2 * 2 * 2)
        cells = {spec.cell for spec in episodes}
        self.assertEqual(len(cells), 2 * 2 * 2)
        # every cell carries the four configurations, all with the same start yaw
        for cell in cells:
            group = [spec for spec in episodes if spec.cell == cell]
            self.assertEqual([spec.config for spec in group], list(DRIVER.CONFIG_ORDER))
            self.assertEqual(len({spec.start_yaw_deg for spec in group}), 1)
        # the two yaw groups of a task/seed differ
        pair = {spec.yaw_mode: spec.start_yaw_deg for spec in episodes
                if spec.task.name == "book" and spec.seed == 7}
        self.assertNotEqual(pair["visible"], pair["random"])

    def test_all_four_configs_share_the_paired_scene_and_start_options(self):
        args = _args("--seeds", "7", "--initial-yaw-modes", "visible")
        episodes = self._episodes(args)
        commands = self._commands(args, episodes)
        check = DRIVER.pairing_check(args, episodes, commands)
        self.assertEqual(check["cells"], 1)
        self.assertTrue(check["consistent"], check["issues"])
        self.assertEqual(check["issues"], [])

        maps = {config: DRIVER._args_map(cmd) for (config, *_), cmd in commands.items()}
        action, tool_full = maps["action"], maps["tool_full"]
        for key in ("--world", "--ply", "--target-pos", "--start-pos", "--start-yaw-deg",
                    "--seed", "--initial-yaw-mode", "--success-dist", "--image-size",
                    "--context-frames", "--scan-views"):
            self.assertEqual(action[key], tool_full[key], key)
        self.assertNotIn("--render-size", action)  # action runner has no such flag

    def test_budget_mapping_uses_max_steps_for_the_action_loop(self):
        args = _args("--max-vlm-calls", "12")
        mapping = DRIVER.budget_mapping(args)
        self.assertEqual(mapping["max_vlm_calls"], 12)
        self.assertEqual(mapping["action"], {"max_steps": 12})
        self.assertEqual(mapping["tool"], {"max_turns": 12, "tool_budget": 12})
        episodes = self._episodes(args)
        commands = self._commands(args, episodes)
        action = DRIVER._args_map(commands[("action", "book", 7, "visible")])
        self.assertEqual(action["--max-steps"], "12")
        self.assertNotIn("--toolset", action)
        for config, toolset in (("tool_basic", "basic"), ("tool_memory", "memory"),
                                ("tool_full", "full")):
            options = DRIVER._args_map(commands[(config, "book", 7, "visible")])
            self.assertEqual(options["--toolset"], toolset)
            self.assertEqual(options["--max-turns"], "12")
            self.assertEqual(options["--tool-budget"], "12")
            self.assertNotIn("--max-steps", options)

    def test_tool_budget_can_be_set_independently(self):
        args = _args("--max-vlm-calls", "12", "--tool-budget", "30")
        self.assertEqual(DRIVER.tool_budget(args), 30)
        options = DRIVER._args_map(
            DRIVER.build_command(args, self._episodes(args)[1],
                                 DRIVER.episode_dir(Path(args.out), self._episodes(args)[1]))
        )
        self.assertEqual(options["--tool-budget"], "30")
        self.assertEqual(options["--max-turns"], "12")

    def test_mock_backend_uses_the_action_oracle_and_the_tool_script(self):
        args = _args("--backend", "mock")
        episodes = self._episodes(args)
        commands = self._commands(args, episodes)
        action = DRIVER._args_map(commands[("action", "book", 7, "visible")])
        tool = DRIVER._args_map(commands[("tool_basic", "book", 7, "visible")])
        self.assertEqual(action["--backend"], "mock:oracle")
        self.assertEqual(tool["--backend"], "mock")
        # explicit overrides win
        args = _args("--backend", "hf", "--model-path", "/weights/qwen",
                     "--action-backend", "mock:forward")
        self.assertEqual(DRIVER.action_backend(args), "mock:forward")
        model_args = DRIVER.model_args(args)
        self.assertEqual(model_args[:2], ["--model-path", "/weights/qwen"])
        self.assertEqual(model_args[2:4], ["--image-size", "320"])

    def test_hf_backend_requires_a_model_path(self):
        args = _args("--backend", "hf")
        with self.assertRaises(SystemExit):
            DRIVER.load_tasks(args)

    def test_config_selection_keeps_the_ladder_order(self):
        args = _args("--configs", "tool_full,action")
        self.assertEqual(args.config_list, ["action", "tool_full"])
        with self.assertRaises(SystemExit):
            DRIVER.resolve_matrix(_args("--configs", "nope"))


# ---------------------------------------------------------------------------
# metric extraction from the two runners' artifacts
# ---------------------------------------------------------------------------


class ActionMetricsTest(unittest.TestCase):
    def _row(self, out_dir: Path, summary: dict, trace: list[dict]):
        args = _args("--max-vlm-calls", "6")
        spec = DRIVER.EpisodeSpec("action", _task(), 7, "visible", 12.5)
        _write_json(out_dir / "summary.json", summary)
        _write_jsonl(out_dir / "episode_trace.jsonl", trace)
        return DRIVER.extract_metrics(args, spec, out_dir, {"returncode": 0, "error": None})

    def test_counts_repeat_scans_and_region_revisits(self):
        trace = [
            {"step": 1, "action_summary": "scan (target lost)", "outcome": "rescan",
             "position_flu": [0.5, 0.5, 1.0], "yaw_deg": 0.0, "distance_to_target": 5.0,
             "parse_ok": True, "raw_response": "{\"action\": \"scan\"}"},
            {"step": 2, "action_summary": "move fwd +2.00m", "outcome": "moved",
             "position_flu": [2.5, 0.5, 1.0], "yaw_deg": 0.0, "distance_to_target": 3.0,
             "parse_ok": True, "raw_response": "{\"action\": \"move\"}"},
            {"step": 3, "action_summary": "move fwd -2.00m", "outcome": "moved",
             "position_flu": [0.5, 0.5, 1.0], "yaw_deg": 0.0, "distance_to_target": 4.0,
             "parse_ok": True, "raw_response": "{\"action\": \"move\"}"},
            {"step": 4, "action_summary": "scan (again)", "outcome": "rescan",
             "position_flu": [0.5, 0.5, 1.0], "yaw_deg": 0.0, "distance_to_target": 4.0,
             "parse_ok": True, "raw_response": "{\"action\": \"scan\"}"},
        ]
        summary = {
            "success": True, "finish_reason": "within_success_distance", "steps": 4,
            "vlm_calls": 6, "wall_time_s": 2.5, "final_distance_m": 1.8,
            "path_length_m": 7.5, "collision_hits": 1, "parse_failures": 0,
            "initial_target_visible": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(Path(tmp), summary, trace)
        self.assertTrue(row["success"])
        self.assertIsNone(row["error"])
        self.assertEqual(row["vlm_calls"], 6)
        self.assertEqual(row["steps"], 4)
        self.assertEqual(row["collisions"], 1)
        # initial look-around + the two recorded scan steps
        self.assertEqual(row["scan_calls"], 3)
        # the step-1 and step-4 scans re-scan the cell scanned by the look-around
        self.assertEqual(row["repeat_scans"], 2)
        self.assertEqual(row["region_revisits"], 1)
        self.assertEqual(row["repeat_search_calls"], 3)
        self.assertTrue(row["initial_target_visible"])
        # 3 decisions x (2 context frames + bev) + 3 scans x (8 views + 2 pitched)
        self.assertEqual(row["images_total"], 3 * 3 + 3 * 10)
        text_chars = sum(len(entry["raw_response"]) for entry in trace) + 6 * 1200
        self.assertEqual(row["text_chars"], text_chars)
        self.assertEqual(
            row["tokens_estimate"],
            39 * row["image_tokens"] + text_chars // int(row["chars_per_token"]),
        )

    def test_action_metrics_survive_a_trace_without_scan_flags(self):
        summary = {"success": False, "finish_reason": "max_steps", "steps": 2,
                   "vlm_calls": 3, "wall_time_s": 1.0}
        trace = [
            {"step": 1, "action_summary": "done (reached)", "outcome": "done",
             "position_flu": [0.0, 0.0, 1.0], "raw_response": "{}"},
            {"step": 2, "action_summary": "hover (waiting)", "outcome": "hover",
             "position_flu": [0.0, 0.0, 1.0], "raw_response": "{}"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(Path(tmp), summary, trace)
        self.assertEqual(row["repeat_scans"], 0)
        self.assertEqual(row["region_revisits"], 0)
        self.assertFalse(row["success"])
        self.assertEqual(row["finish_reason"], "max_steps")


class ToolMetricsTest(unittest.TestCase):
    def _row(self, out_dir: Path, summary: dict, trace: list[dict]):
        args = _args("--max-vlm-calls", "8")
        spec = DRIVER.EpisodeSpec("tool_memory", _task(), 11, "random", -120.0)
        _write_json(out_dir / "summary.json", summary)
        _write_jsonl(out_dir / "tool_trace.jsonl", trace)
        return DRIVER.extract_metrics(args, spec, out_dir, {"returncode": 0, "error": None})

    def test_reads_the_module_runtime_trace(self):
        trace = [
            {"turn": 1, "tool": "observe", "args": {"mode": "lookaround"},
             "status": "ok", "images_sent": 1, "data": {"pose": [0.0, 0.0, 1.0]},
             "result_prompt": "[ok] observation_id=obs_2", "raw_reply": "{\"tool\": \"observe\"}"},
            {"turn": 2, "tool": "observe", "args": {"mode": "forward"},
             "status": "ok", "images_sent": 2, "data": {"pose": [0.0, 0.0, 1.0]},
             "result_prompt": "[ok] observation_id=obs_3", "raw_reply": "{\"tool\": \"observe\"}"},
            {"turn": 3, "tool": "observe", "args": {"mode": "lookaround"},
             "status": "ok", "images_sent": 1, "data": {"pose": [0.0, 0.0, 1.0]},
             "result_prompt": "[ok] observation_id=obs_4", "raw_reply": "{\"tool\": \"observe\"}"},
            {"turn": 4, "tool": "navigate", "args": {"bearing_deg": 0.0},
             "status": "cached", "images_sent": 1, "data": {"pose": [4.0, 0.0, 1.0]},
             "result_prompt": "[cached] task_id=nav_1", "raw_reply": "{\"tool\": \"navigate\"}"},
            {"turn": 5, "tool": "observe", "args": {"mode": "forward"},
             "status": "ok", "images_sent": 2, "data": {"pose": [0.2, 0.1, 1.0]},
             "result_prompt": "[ok] observation_id=obs_5", "raw_reply": "{\"tool\": \"observe\"}"},
        ]
        summary = {
            "success": True, "finish_reason": "finish", "turns": 5, "tool_calls": 5,
            "vlm_calls": 5, "wall_time_s": 3.5, "final_distance_m": 1.9,
            "new_evidence_total": 7, "parse_failures": 0,
            "tools": {"observe": {"calls": 4}, "navigate": {"calls": 1}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(Path(tmp), summary, trace)
        self.assertTrue(row["success"])
        self.assertEqual(row["vlm_calls"], 5)
        self.assertEqual(row["tool_calls"], 5)
        self.assertEqual(row["steps"], 5)
        self.assertEqual(row["images_total"], 7)
        self.assertEqual(row["cached_calls"], 1)
        self.assertEqual(row["new_evidence"], 7)
        self.assertEqual(row["tools_used"], {"observe": 4, "navigate": 1})
        # two lookaround calls, the second repeats the scan of cell (0, 0)
        self.assertEqual(row["scan_calls"], 2)
        self.assertEqual(row["repeat_scans"], 1)
        # returning from (2, 0) to (0, 0) counts as one revisit
        self.assertEqual(row["region_revisits"], 1)
        self.assertEqual(row["repeat_search_calls"], 2)
        # 4 m out to (4, 0) and 3.801 m back to (0.2, 0.1)
        self.assertAlmostEqual(row["path_length_m"], 7.801, places=3)

    def test_reads_the_fallback_runtime_trace_schema(self):
        trace = [
            {"turn": 1, "tool": "observe", "args": {"mode": "lookaround"},
             "status": "ok", "pose": {"position_flu": [0.0, 0.0, 1.0]},
             "result_prompt": "[ok]", "model_reply": "{}"},
            {"turn": 2, "tool": "observe", "args": {"mode": "lookaround"},
             "status": "ok", "pose": {"position_flu": [0.0, 0.0, 1.0]},
             "result_prompt": "[ok]", "model_reply": "{}"},
        ]
        summary = {"success": False, "finish_reason": "max_turns", "turns": 2,
                   "vlm_calls": 2, "wall_time_s": 0.5}
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(Path(tmp), summary, trace)
        self.assertEqual(row["scan_calls"], 2)
        self.assertEqual(row["repeat_scans"], 1)
        # no images_sent in this schema: estimated from the context budget
        self.assertEqual(row["images_total"], 2 * (2 + 1))
        self.assertIsNone(row["cached_calls"])

    def test_missing_summary_becomes_an_error_row(self):
        args = _args()
        spec = DRIVER.EpisodeSpec("tool_basic", _task(), 7, "visible", 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            row = DRIVER.extract_metrics(
                args, spec, out_dir, {"returncode": 0, "error": None}
            )
            self.assertIsNone(row["success"] or None, "success must not claim a run")
            self.assertIn("missing summary.json", row["error"])
            row = DRIVER.extract_metrics(
                args, spec, out_dir, {"returncode": 2, "error": "runner exited with code 2"}
            )
            self.assertEqual(row["error"], "runner exited with code 2")


# ---------------------------------------------------------------------------
# aggregation + Markdown report
# ---------------------------------------------------------------------------


class TokenModelTest(unittest.TestCase):
    def test_tokens_are_images_times_patch_plus_text(self):
        args = _args("--image-size", "140", "140", "--chars-per-token", "2",
                     "--prompt-chars-per-call", "0")
        row = {"images_total": 3, "text_chars": 40}
        DRIVER._token_model(args, row)
        self.assertEqual(row["image_tokens"], 25)  # 140*140/784
        self.assertEqual(row["tokens_estimate"], 3 * 25 + 20)

    def test_divisor_and_missing_values(self):
        args = _args("--image-size", "28", "28", "--image-token-divisor", "28")
        self.assertEqual(DRIVER.image_tokens(args), 28)
        row = {"images_total": None, "text_chars": None}
        DRIVER._token_model(args, row)
        self.assertEqual(row["tokens_estimate"], 0)


class AggregateTest(unittest.TestCase):
    def _rows(self):
        rows = []
        for config, success, calls, tokens, wall, repeats in (
            ("action", True, 6, 3000, 2.0, 0),
            ("tool_basic", False, 8, 2000, 3.0, 2),
            ("tool_memory", True, 5, 1500, 4.0, 1),
            ("tool_full", True, 4, 1200, 5.0, 0),
        ):
            for yaw_mode in ("visible", "random"):
                rows.append(
                    {
                        "config": config, "config_label": config, "task": "book",
                        "seed": 7, "yaw_mode": yaw_mode, "success": success,
                        "finish_reason": "finish" if success else "max_turns",
                        "vlm_calls": calls, "tool_calls": calls - 1,
                        "steps": calls, "wall_time_s": wall, "tokens_estimate": tokens,
                        "images_total": 10, "text_chars": 400,
                        "repeat_search_calls": repeats, "repeat_scans": repeats,
                        "region_revisits": 0, "cached_calls": repeats,
                        "error": None, "skipped": False,
                    }
                )
        return rows

    def test_aggregate_reports_success_rate_and_means(self):
        stats = DRIVER.aggregate(self._rows())
        self.assertEqual(stats["n"], 8)
        self.assertEqual(stats["n_success"], 6)
        self.assertEqual(stats["success_rate"], 0.75)
        self.assertEqual(stats["n_errors"], 0)
        self.assertAlmostEqual(stats["vlm_calls_mean"], 5.75)
        self.assertAlmostEqual(stats["repeat_search_calls_mean"], 0.75)
        self.assertEqual(stats["finish_reasons"], {"finish": 6, "max_turns": 2})

    def test_groups_cover_every_config_and_yaw_mode(self):
        rows = self._rows()
        groups = DRIVER.grouping(rows, ["visible", "random"])
        keys = {(config, group) for config, group, _ in groups}
        for config in DRIVER.CONFIG_ORDER:
            self.assertIn((config, "all"), keys)
            self.assertIn((config, "visible"), keys)
            self.assertIn((config, "random"), keys)

    def test_markdown_table_holds_the_required_columns(self):
        rows = self._rows()
        args = _args("--seeds", "7", "--initial-yaw-modes", "visible,random")
        tasks = [_task()]
        episodes = DRIVER.build_episodes(args, tasks)
        commands = {
            (spec.config, spec.task.name, spec.seed, spec.yaw_mode):
                DRIVER.build_command(args, spec, DRIVER.episode_dir(Path(args.out), spec))
            for spec in episodes
        }
        args.config_list = list(DRIVER.CONFIG_ORDER)
        args.seed_list = [7]
        args.yaw_mode_list = ["visible", "random"]
        groups = DRIVER.grouping(rows, ["visible", "random"])
        aggregates = {
            f"{config}|{group}": DRIVER.aggregate(subset)
            for config, group, subset in groups
        }
        meta = DRIVER.meta_block(args, tasks, episodes, commands, aggregates)
        markdown = DRIVER.render_markdown(meta, rows, groups, aggregates)
        for column in ("success", "vlm_calls", "tokens_est", "wall_s", "repeat_search"):
            self.assertIn(column, markdown)
        for config in DRIVER.CONFIG_ORDER:
            self.assertIn(f"| {config} | all |", markdown)
        self.assertIn("| action | visible |", markdown)
        self.assertIn("| action | all | 2 | 2/2 (1.00) |", markdown)
        self.assertIn("| tool_basic | all | 2 | 0/2 (0.00) |", markdown)
        self.assertEqual(DRIVER.aggregate(rows)["success_rate"], 0.75)
        self.assertIn("- tasks: book", markdown)

    def test_aggregate_marks_error_episodes(self):
        rows = self._rows()
        rows.append({"config": "action", "yaw_mode": "visible", "success": False,
                     "error": "runner exited with code 1"})
        stats = DRIVER.aggregate(rows)
        self.assertEqual(stats["n"], 9)
        self.assertEqual(stats["n_errors"], 1)
        self.assertEqual(stats["n_ok"], 8)
        self.assertEqual(stats["success_rate"], 0.75)

    def test_pairing_check_flags_a_divergent_scene_option(self):
        args = _args()
        episodes = DRIVER.build_episodes(args, [_task()])
        commands = {
            (spec.config, spec.task.name, spec.seed, spec.yaw_mode):
                DRIVER.build_command(args, spec, DRIVER.episode_dir(Path(args.out), spec))
            for spec in episodes
        }
        check = DRIVER.pairing_check(args, episodes, commands)
        self.assertTrue(check["consistent"])
        # tamper with one configuration's seed -> the cell must be flagged
        key = ("tool_full", "book", 7, "visible")
        commands[key] = [part if part != "--seed" else "--seed" for part in commands[key]]
        index = commands[key].index("--seed")
        commands[key][index + 1] = "999"
        check = DRIVER.pairing_check(args, episodes, commands)
        self.assertFalse(check["consistent"])
        self.assertEqual(check["issues"][0]["option"], "--seed")
        self.assertEqual(check["issues"][0]["cell"], "book__visible__seed7")


# ---------------------------------------------------------------------------
# end-to-end: the four configurations really run and produce the report
# ---------------------------------------------------------------------------


class CompareIntegrationTest(unittest.TestCase):
    """Runs the driver as a subprocess (mock backends, tiny CPU world)."""

    def _run_driver(self, out_dir: Path, scene: Path, *extra: str):
        cmd = [
            sys.executable, "-m", "scripts.compare_agent_configs",
            "--world", "cpu",
            "--ply", str(scene),
            "--target-name", "target",
            "--target-pos", "2", "0.5", "0",
            "--target-radius", "0.5",
            "--start-pos", "0", "0", "1",
            "--instruction", INSTRUCTION,
            "--backend", "mock",
            "--out", str(out_dir),
            *extra,
        ]
        env = dict(os.environ)
        env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
        return subprocess.run(
            cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=900
        )

    def test_dry_run_lists_the_paired_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene.ply"
            _make_scene(scene)
            proc = self._run_driver(Path(tmp) / "out", scene, "--dry-run",
                                    "--seeds", "7,11",
                                    "--initial-yaw-modes", "visible,random")
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            commands = [
                DRIVER._args_map(line.split())
                for line in proc.stdout.splitlines()
                if line.startswith(sys.executable)
            ]
            self.assertEqual(len(commands), 4 * 2 * 2)  # configs x seeds x yaw modes
            self.assertEqual(sum("--max-steps" in cmd for cmd in commands), 4)
            self.assertEqual(sum(cmd.get("--toolset") == "basic" for cmd in commands), 4)
            self.assertEqual(sum(cmd.get("--toolset") == "memory" for cmd in commands), 4)
            self.assertEqual(sum(cmd.get("--toolset") == "full" for cmd in commands), 4)
            for seed in ("7", "11"):
                headings = {
                    cmd["--initial-yaw-mode"]: cmd["--start-yaw-deg"]
                    for cmd in commands
                    if cmd["--seed"] == seed and cmd.get("--toolset") == "basic"
                }
                self.assertEqual(set(headings), {"visible", "random"})
                self.assertNotEqual(headings["visible"], headings["random"])

    def test_four_configs_run_and_produce_the_comparison_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene.ply"
            _make_scene(scene)
            out_dir = Path(tmp) / "compare"
            proc = self._run_driver(
                out_dir, scene,
                "--seeds", "7",
                "--initial-yaw-modes", "visible",
                "--max-vlm-calls", "2",
                "--scan-views", "4",
                "--jobs", "4",
            )
            # diagnostics for every later assertion: the driver's own log (which
            # prints each episode's returncode/error) plus its stderr
            diagnostics = (
                "\n--- driver stdout ---\n" + proc.stdout[-4000:]
                + "\n--- driver stderr ---\n" + proc.stderr[-2000:]
            )
            self.assertEqual(proc.returncode, 0, "driver failed" + diagnostics)

            summary_path = out_dir / "compare_summary.json"
            table_path = out_dir / "compare_table.md"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            diagnostics += "\n--- episodes ---\n" + "\n".join(
                f"{row.get('config')} {row.get('yaw_mode')} seed={row.get('seed')} "
                f"rc={row.get('returncode')} error={row.get('error')} "
                f"success={row.get('success')} vlm_calls={row.get('vlm_calls')}"
                for row in summary.get("episodes", [])
            )
            self.assertTrue(summary_path.exists(), "no compare_summary.json" + diagnostics)
            self.assertTrue(table_path.exists(), "no compare_table.md" + diagnostics)
            self.assertEqual(summary.get("n_errors"), 0, diagnostics)
            self.assertEqual(len(summary.get("episodes", [])), 4, diagnostics)
            self.assertEqual(
                {row["config"] for row in summary["episodes"]},
                set(DRIVER.CONFIG_ORDER), diagnostics,
            )
            self.assertTrue(summary["meta"]["pairing_check"]["consistent"], diagnostics)
            self.assertEqual(summary["meta"]["budget"]["action"]["max_steps"], 2, diagnostics)
            self.assertEqual(summary["meta"]["budget"]["tool"]["max_turns"], 2, diagnostics)

            cells = {(row["task"], row["seed"], row["yaw_mode"]) for row in summary["episodes"]}
            self.assertEqual(len(cells), 1,
                             "all four configs share one paired cell" + diagnostics)
            for row in summary["episodes"]:
                self.assertEqual(row["returncode"], 0, row)
                self.assertIsNone(row["error"], row)
                self.assertIsNotNone(row["vlm_calls"], row)
                self.assertIsNotNone(row["tokens_estimate"], row)
                self.assertIsNotNone(row["wall_time_s"], row)
                self.assertIsNotNone(row["repeat_search_calls"], row)
                self.assertTrue(Path(row["out_dir"], "summary.json").exists(), row["out_dir"])
                self.assertTrue(Path(row["out_dir"], "run.log").exists(), row["out_dir"])
            toolsets = {row["config"]: row["toolset"] for row in summary["episodes"]}
            self.assertEqual(
                toolsets,
                {"action": None, "tool_basic": "basic", "tool_memory": "memory",
                 "tool_full": "full"},
                diagnostics,
            )
            # the ladder really restricted the catalogue the model saw
            for row in summary["episodes"]:
                if row["config"] == "action":
                    self.assertIsNone(row["toolset_detail"])
                    continue
                self.assertEqual(row["toolset_detail"]["name"], row["toolset"], row)
                self.assertIn("observe", row["toolset_detail"]["registered"])
            self.assertEqual(
                {row["start_yaw_deg"] for row in summary["episodes"]},
                {summary["episodes"][0]["start_yaw_deg"]},
                "the paired cell must share one start heading" + diagnostics,
            )

            aggregates = summary["meta"]["aggregates"]
            for config in DRIVER.CONFIG_ORDER:
                self.assertIn(f"{config}|all", aggregates, diagnostics)
                self.assertIn(f"{config}|visible", aggregates, diagnostics)
                self.assertEqual(f"{config}|random" in aggregates, False, diagnostics)
                self.assertEqual(aggregates[f"{config}|all"]["n"], 1, diagnostics)
                self.assertEqual(aggregates[f"{config}|all"]["n_errors"], 0, diagnostics)

            table = table_path.read_text()
            for config in DRIVER.CONFIG_ORDER:
                self.assertIn(f"| {config} | all |", table, diagnostics)
            self.assertIn("| tool_full | visible |", table, diagnostics)
            self.assertIn("tokens_est", table, diagnostics)
            self.assertIn("repeat_search", table, diagnostics)


if __name__ == "__main__":
    unittest.main()
