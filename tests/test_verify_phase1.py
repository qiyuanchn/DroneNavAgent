"""Tests for the Phase-1 acceptance driver (``scripts/verify_phase1.py``).

The Phase-1 acceptance question is a *paired* one: with the same model, scene,
start pose/heading and VLM budget, does the tool loop produce stable, correctly
directed movement and solve the simple tasks the action loop already solves?
``scripts/verify_phase1.py`` drives both runners as subprocesses and reports the
metrics goal.md splits apart:

  tool call success rate   ok tool results / tool calls
  effective move rate      nav attempts with progress > threshold AND a heading
                           within 90 deg of the target bearing
  zero-displacement loops  consecutive nav attempts that moved nothing
                           (the observe -> navigate -> already_there death loop)
  target re-acquisition    visibility series: lost -> seen again
  task success             ground-truth distance within the success gate, and
                           a completion claim only counts when geometry agrees
  full VLM cost            model calls including scan/crop/tool-internal calls,
                           plus wall time

These tests pin the metric definitions on synthetic traces of both runners'
schemas, lock the tool-name/result-schema contract with a stub registry driven
through the real tool runner, and finally run the driver end to end with
``--backend mock`` on a tiny CPU 3DGS scene.
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
VERIFY_PATH = ROOT / "scripts" / "verify_phase1.py"
TOOL_RUNNER_PATH = ROOT / "scripts" / "run_vlm_nav_tools.py"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VERIFY = _load("diffaero_script_verify_phase1", VERIFY_PATH)
Runner = _load("diffaero_script_run_vlm_nav_tools_for_phase1", TOOL_RUNNER_PATH)

from vlm_nav.tools import (  # noqa: E402  (after the runner load)
    STATUS_BLOCKED,
    STATUS_CACHED,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_OK,
    TOOL_CALL_DOC,
    EvidenceRef,
    ToolSpec,
)

TARGET = (4.0, 0.0, 1.0)
START_POSE = (0.0, 0.0, 1.0)
INSTRUCTION = "Find the target and fly to it."


def _args(*extra: str, **overrides):
    argv = [
        "--world", "cpu",
        "--ply", "/scene.ply",
        "--target-pos", "4", "0", "1",
        "--instruction", INSTRUCTION,
        "--out", "/tmp/verify_phase1",
        *extra,
    ]
    args = VERIFY.parse_args(argv)
    VERIFY.resolve_matrix(args)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _task(**overrides):
    data = dict(name="book", instruction=INSTRUCTION, target_pos=TARGET,
                target_name="target", target_radius=0.5, start_pos=START_POSE,
                success_dist=2.0)
    data.update(overrides)
    return VERIFY.compare.TaskSpec(**data)


def _spec(config: str = VERIFY.CONFIG_TOOL):
    return VERIFY.compare.EpisodeSpec(config, _task(), 7, "visible", 0.0)


def _tool_trace(entries):
    """Tool-loop trace rows (module runtime schema: data carries pose/metrics)."""
    out = []
    for index, entry in enumerate(entries, start=1):
        entry = dict(entry)
        data = dict(entry.pop("data", {}))
        row = {
            "turn": index,
            "tool": entry.pop("tool"),
            "args": entry.pop("args", {}),
            "status": entry.pop("status", STATUS_OK),
            "data": data,
            "evidence": entry.pop("evidence", []),
            "images_sent": entry.pop("images_sent", 0),
            "result_prompt": entry.pop("result_prompt", ""),
            "raw_reply": entry.pop("raw_reply", ""),
            "parse_failure": entry.pop("parse_failure", False),
        }
        row.update(entry)
        out.append(row)
    return out


def _nav(tool: str, pose, **data):
    payload = {"pose": [float(v) for v in pose], "heading_deg": 0.0}
    payload.update(data)
    return {"tool": tool, "data": payload}


def _action_trace(entries):
    out = []
    for index, entry in enumerate(entries, start=1):
        out.append({
            "step": index,
            "action_summary": entry[0],
            "outcome": entry[1],
            "position_flu": [float(v) for v in entry[2]],
            "yaw_deg": float(entry[3]) if len(entry) > 3 else 0.0,
            "distance_to_target": entry[4] if len(entry) > 4 else 1.0,
            "parse_ok": entry[5] if len(entry) > 5 else True,
            "raw_response": "{}",
        })
    return out


def _tool_summary(**overrides):
    summary = {
        "success": True,
        "finish_reason": "finish",
        "turns": 8,
        "tool_calls": 7,
        "vlm_calls": 8,
        "wall_time_s": 3.0,
        "final_distance_m": 0.4,
        "parse_failures": 0,
        "toolset": "full",
        "toolset_detail": {"name": "full", "registered": list(VERIFY.PHASE1_TOOLS)},
        "tools": {},
    }
    summary.update(overrides)
    return summary


# ---------------------------------------------------------------------------
# contract vocabulary (tool names / result fields of the Phase-1 toolset)
# ---------------------------------------------------------------------------


class ContractTest(unittest.TestCase):
    def test_phase1_toolset_names_match_the_shared_contract(self):
        self.assertEqual(
            VERIFY.PHASE1_TOOLS,
            ("observe", "ground_target", "approach_candidate", "go_to_viewpoint",
             "return_to_observation", "verify_goal"),
        )
        for name in VERIFY.PHASE1_TOOLS:
            self.assertIn(name, TOOL_CALL_DOC, name)
        # "go somewhere" tools (by id type) plus the continuous navigation tool
        self.assertEqual(
            set(VERIFY.NAV_TOOL_NAMES),
            {"navigate", "approach_candidate", "go_to_viewpoint", "return_to_observation"},
        )
        # the driver reuses the shared status vocabulary instead of redefining it
        self.assertEqual((VERIFY.STATUS_OK, VERIFY.STATUS_FAILED), (STATUS_OK, STATUS_FAILED))
        self.assertEqual(VERIFY.CONFIG_ORDER, (VERIFY.CONFIG_ACTION, VERIFY.CONFIG_TOOL))

    def test_required_tool_check_reads_the_toolset_detail(self):
        summary = _tool_summary()
        self.assertEqual(VERIFY.required_tools_missing(summary), [])
        summary["toolset_detail"]["registered"] = ["observe", "navigate", "verify_goal"]
        self.assertEqual(
            VERIFY.required_tools_missing(summary),
            ["ground_target", "approach_candidate", "go_to_viewpoint",
             "return_to_observation"],
        )
        # falls back to the per-tool summary when there is no toolset detail
        self.assertEqual(
            VERIFY.required_tools_missing({"tools": {"observe": {"calls": 1}}}),
            ["ground_target", "approach_candidate", "go_to_viewpoint",
             "return_to_observation", "verify_goal"],
        )
        # an episode that never reached the tool runner reports everything missing
        self.assertEqual(VERIFY.required_tools_missing({}), list(VERIFY.PHASE1_TOOLS))


# ---------------------------------------------------------------------------
# trace normalization
# ---------------------------------------------------------------------------


class TimelineTest(unittest.TestCase):
    def test_tool_trace_rows_carry_pose_yaw_and_move_intent(self):
        trace = _tool_trace([
            {"tool": "observe", "data": {"pose": [0.0, 0.0, 1.0], "heading_deg": 12.5}},
            {"tool": "navigate", "data": {"pose": [1.0, 0.0, 1.0], "heading_deg": 0.0,
                                          "segment_progress_m": 1.0, "outcome": "moved"}},
            {"tool": "finish", "data": {"reason": "done"}},
        ])
        events = VERIFY.timeline(trace, VERIFY.CONFIG_TOOL, start_pose=START_POSE)
        self.assertEqual(len(events), 2)  # finish is a termination, not a tool call
        self.assertEqual(events[0]["tool"], "observe")
        self.assertAlmostEqual(events[0]["yaw_deg"], 12.5)
        self.assertFalse(events[0]["move_intent"])
        self.assertEqual(events[0]["prev_pose"], list(START_POSE))
        self.assertTrue(events[1]["move_intent"])
        self.assertEqual(events[1]["prev_pose"], [0.0, 0.0, 1.0])
        self.assertEqual(events[1]["reported_progress_m"], 1.0)
        self.assertEqual(events[1]["outcome"], "moved")

    def test_fallback_tool_trace_schema_and_action_trace_schema(self):
        fallback = [
            {"turn": 1, "tool": "navigate", "status": STATUS_OK,
             "pose": {"position_flu": [2.0, 0.0, 1.0], "yaw_deg": 30.0},
             "data": {"progress_m": 2.0, "outcome": "moved"}, "model_reply": "{}"},
        ]
        events = VERIFY.timeline(fallback, VERIFY.CONFIG_TOOL, start_pose=START_POSE)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["pose"], [2.0, 0.0, 1.0])
        self.assertAlmostEqual(events[0]["yaw_deg"], 30.0)
        self.assertTrue(events[0]["move_intent"])
        self.assertEqual(events[0]["reported_progress_m"], 2.0)

        action = _action_trace([
            ("scan (target lost)", "rescan", (0.0, 0.0, 1.0), 0.0, 4.0),
            ("move fwd +1.50m", "moved 1.5m", (1.5, 0.0, 1.0), 0.0, 2.5),
            ("turn yaw +90deg", "turned", (1.5, 0.0, 1.0), 90.0, 2.5),
            ("waypoint fwd +1.00m", "moved 1.0m", (2.5, 0.0, 1.0), 0.0, 1.5),
        ])
        events = VERIFY.timeline(action, VERIFY.CONFIG_ACTION, start_pose=START_POSE)
        self.assertEqual([event["move_intent"] for event in events],
                         [False, True, False, True])
        self.assertEqual(events[1]["pose"], [1.5, 0.0, 1.0])

    def test_turn_and_scan_steps_are_not_move_attempts(self):
        trace = _action_trace([
            ("turn yaw +45deg", "turned", (0.0, 0.0, 1.0), 45.0, 4.0),
            ("scan (target lost)", "rescan", (0.0, 0.0, 1.0), 45.0, 4.0),
            ("hover (waiting)", "hover", (0.0, 0.0, 1.0), 45.0, 4.0),
        ])
        events = VERIFY.timeline(trace, VERIFY.CONFIG_ACTION, start_pose=START_POSE)
        stats = VERIFY.movement_stats(events, TARGET, start_pose=START_POSE)
        self.assertEqual(stats["nav_attempts"], 0)
        self.assertIsNone(stats["effective_move_rate"])
        self.assertEqual(stats["zero_displacement_events"], 0)
        self.assertEqual(stats["zero_displacement_runs"], 0)


# ---------------------------------------------------------------------------
# movement metrics
# ---------------------------------------------------------------------------


class MovementTest(unittest.TestCase):
    def _events(self, rows):
        return VERIFY.timeline(_tool_trace(rows), VERIFY.CONFIG_TOOL, start_pose=START_POSE)

    def test_effective_move_rate_separates_progress_from_direction(self):
        events = self._events([
            {"tool": "observe", "data": {"pose": START_POSE}},
            _nav("approach_candidate", (1.5, 0.0, 1.0), outcome="moved"),
            _nav("approach_candidate", (3.0, 0.0, 1.0), outcome="moved"),
            # moved, but away from the target: progress > 0, direction 180 deg off
            _nav("go_to_viewpoint", (2.0, 0.0, 1.0), outcome="moved"),
            _nav("approach_candidate", (3.5, 0.0, 1.0), outcome="moved"),
        ])
        stats = VERIFY.movement_stats(events, TARGET, start_pose=START_POSE,
                                      min_progress_m=0.1)
        self.assertEqual(stats["nav_attempts"], 4)
        self.assertEqual(stats["moved_attempts"], 4)
        self.assertEqual(stats["effective_moves"], 3)
        self.assertEqual(stats["wrong_direction_moves"], 1)
        self.assertEqual(stats["zero_displacement_events"], 0)
        self.assertEqual(stats["zero_displacement_runs"], 0)
        self.assertAlmostEqual(stats["effective_move_rate"], 0.75, places=3)
        self.assertAlmostEqual(stats["mean_move_m"], 1.375, places=3)
        self.assertAlmostEqual(stats["max_move_m"], 1.5, places=3)
        # net progress along the target bearing: 3 x 1.5 fwd - 1.0 back
        self.assertAlmostEqual(stats["distance_gain_m"], 3.5, places=3)
        self.assertAlmostEqual(stats["distance_before_m"], 4.0, places=3)
        self.assertAlmostEqual(stats["distance_after_m"], 0.5, places=3)
        self.assertEqual(stats["nav_failure_reasons"], {})

    def test_zero_displacement_loop_counts_consecutive_already_there_attempts(self):
        events = self._events([
            {"tool": "observe", "data": {"pose": START_POSE}},
            _nav("approach_candidate", (2.0, 0.0, 1.0), outcome="moved"),
            # the observed death loop: repeated nav calls that move nothing
            _nav("return_to_observation", (2.0, 0.0, 1.0), outcome="already_there"),
            _nav("return_to_observation", (2.0, 0.0, 1.0), outcome="already_there"),
            _nav("approach_candidate", (2.0, 0.0, 1.0), outcome="no_progress",
                 status=STATUS_BLOCKED),
            {"tool": "observe", "data": {"pose": [2.0, 0.0, 1.0]}},
            _nav("approach_candidate", (3.0, 0.0, 1.0), outcome="moved"),
            # a single zero attempt is not a loop
            _nav("go_to_viewpoint", (3.0, 0.0, 1.0), outcome="already_at_pose"),
        ])
        stats = VERIFY.movement_stats(events, TARGET, start_pose=START_POSE)
        self.assertEqual(stats["nav_attempts"], 6)
        self.assertEqual(stats["zero_displacement_events"], 4)
        self.assertEqual(stats["zero_displacement_runs"], 1)
        self.assertEqual(stats["max_zero_displacement_run"], 3)
        self.assertEqual(
            stats["zero_displacement_partners"],
            ["return_to_observation->return_to_observation",
             "return_to_observation->approach_candidate"],
        )
        self.assertEqual(stats["no_progress_markers"], 4)
        self.assertEqual(stats["effective_moves"], 2)
        self.assertAlmostEqual(stats["effective_move_rate"], 0.333, places=3)
        self.assertEqual(stats["nav_failure_reasons"],
                         {"already_there": 2, "no_progress": 1,
                          "already_at_pose": 1})

    def test_failed_navigation_attempts_count_as_zero_displacement(self):
        events = self._events([
            _nav("approach_candidate", START_POSE, status=STATUS_EMPTY,
                 reason="unknown_observation"),
            _nav("approach_candidate", START_POSE, status=STATUS_EMPTY,
                 reason="unknown_observation"),
        ])
        stats = VERIFY.movement_stats(events, TARGET, start_pose=START_POSE)
        self.assertEqual(stats["nav_attempts"], 2)
        self.assertEqual(stats["zero_displacement_events"], 2)
        self.assertEqual(stats["zero_displacement_runs"], 1)
        self.assertEqual(stats["nav_failure_reasons"], {"unknown_observation": 2})

    def test_zero_run_min_length_is_configurable(self):
        events = self._events([
            _nav("approach_candidate", START_POSE, outcome="already_there"),
        ])
        stats = VERIFY.movement_stats(events, TARGET, start_pose=START_POSE,
                                      zero_run_min=1)
        self.assertEqual(stats["zero_displacement_events"], 1)
        self.assertEqual(stats["zero_displacement_runs"], 1)
        self.assertEqual(stats["max_zero_displacement_run"], 1)

    def test_action_loop_moves_use_the_same_rules(self):
        trace = _action_trace([
            ("move fwd +1.00m", "moved 1.0m", (1.0, 0.0, 1.0), 0.0, 3.0),
            ("turn yaw +90deg", "turned", (1.0, 0.0, 1.0), 90.0, 3.0),
            ("move fwd +0.00m", "moved 0.0m", (1.0, 0.0, 1.0), 90.0, 3.0),
            ("move fwd +0.00m", "moved 0.0m", (1.0, 0.0, 1.0), 90.0, 3.0),
            ("move fwd +2.00m", "moved 2.0m", (3.0, 0.0, 1.0), 90.0, 1.0),
        ])
        events = VERIFY.timeline(trace, VERIFY.CONFIG_ACTION, start_pose=START_POSE)
        stats = VERIFY.movement_stats(events, TARGET, start_pose=START_POSE)
        self.assertEqual(stats["nav_attempts"], 4)
        self.assertEqual(stats["effective_moves"], 2)
        self.assertEqual(stats["zero_displacement_events"], 2)
        self.assertEqual(stats["zero_displacement_runs"], 1)
        self.assertAlmostEqual(stats["effective_move_rate"], 0.5, places=3)

    def test_cached_repeat_calls_are_labelled_as_repeats(self):
        leg = {"outcome": "arrived", "reason": "arrived", "segment_progress_m": 1.3}
        events = self._events([
            _nav("go_to_viewpoint", (0.9, -0.9, 1.0), **leg),
            # the runtime answered the identical repeat from its call cache
            {"tool": "go_to_viewpoint", "status": STATUS_CACHED,
             "data": _nav("go_to_viewpoint", (0.9, -0.9, 1.0), **leg)["data"]},
        ])
        stats = VERIFY.movement_stats(events, TARGET, start_pose=START_POSE)
        self.assertEqual(stats["nav_attempts"], 2)
        self.assertEqual(stats["effective_moves"], 1)
        self.assertEqual(stats["zero_displacement_events"], 1)
        self.assertEqual(stats["zero_displacement_runs"], 0)
        self.assertEqual(stats["nav_failure_reasons"], {"cached": 1})

    def test_movement_stats_are_none_safe_without_poses(self):
        events = VERIFY.timeline(
            _tool_trace([{"tool": "navigate", "status": STATUS_OK,
                          "data": {"progress_m": 1.0}}]),
            VERIFY.CONFIG_TOOL, start_pose=None,
        )
        stats = VERIFY.movement_stats(events, TARGET, start_pose=None)
        self.assertEqual(stats["nav_attempts"], 1)
        self.assertEqual(stats["moved_attempts"], 0)
        self.assertEqual(stats["effective_moves"], 0)
        self.assertEqual(stats["effective_move_rate"], 0.0)


# ---------------------------------------------------------------------------
# visibility / re-acquisition
# ---------------------------------------------------------------------------


class FovTest(unittest.TestCase):
    def test_target_in_fov_estimate_uses_the_camera_cone(self):
        # straight ahead: inside the horizontal FOV
        self.assertTrue(VERIFY.target_in_fov_estimate(
            (0.0, 0.0, 1.0), 0.0, TARGET, hfov_deg=86.0, width=320, height=180))
        # target behind the drone
        self.assertFalse(VERIFY.target_in_fov_estimate(
            (0.0, 0.0, 1.0), 180.0, TARGET, hfov_deg=86.0, width=320, height=180))
        # outside the horizontal half-FOV (bearing 60 deg vs half-FOV 43 deg)
        self.assertFalse(VERIFY.target_in_fov_estimate(
            (0.0, 0.0, 1.0), -60.0, TARGET, hfov_deg=86.0, width=320, height=180))
        # well below the vertical FOV: the camera looks level
        self.assertFalse(VERIFY.target_in_fov_estimate(
            (0.0, 0.0, 5.0), 0.0, (4.0, 0.0, 1.0), hfov_deg=86.0, width=320, height=180))
        self.assertIsNone(VERIFY.target_in_fov_estimate(
            None, 0.0, TARGET, hfov_deg=86.0, width=320, height=180))
        self.assertIsNone(VERIFY.target_in_fov_estimate(
            (0.0, 0.0, 1.0), None, TARGET, hfov_deg=86.0, width=320, height=180))

    def test_reported_visibility_drives_loss_and_reacquisition(self):
        events = VERIFY.timeline(_tool_trace([
            {"tool": "observe", "data": {"pose": START_POSE, "target_in_fov": True}},
            {"tool": "approach_candidate", "data": {"pose": [2.0, 0.0, 1.0],
                                                    "heading_deg": 180.0,
                                                    "target_in_fov": False}},
            {"tool": "approach_candidate", "data": {"pose": [2.5, 0.5, 1.0],
                                                    "heading_deg": 180.0,
                                                    "target_in_fov": False}},
            {"tool": "observe", "data": {"pose": [3.0, 0.0, 1.0], "heading_deg": 0.0,
                                         "target_in_fov": True}},
        ]), VERIFY.CONFIG_TOOL, start_pose=START_POSE)
        stats = VERIFY.vision_stats(events, TARGET, hfov_deg=86.0,
                                    start_pose=START_POSE, start_yaw_deg=0.0)
        self.assertEqual(stats["visibility_source"], "reported")
        self.assertEqual(stats["target_lost_events"], 1)
        self.assertEqual(stats["target_reacquired_events"], 1)
        self.assertAlmostEqual(stats["reacquisition_rate"], 1.0)
        self.assertEqual(stats["fov_measured_states"], 4)
        self.assertEqual(stats["fov_agreement"], 1.0)
        self.assertEqual(stats["fov_mismatch_states"], 0)

    def test_never_seen_again_is_a_zero_reacquisition_rate(self):
        events = VERIFY.timeline(_tool_trace([
            {"tool": "observe", "data": {"pose": START_POSE, "target_in_fov": True}},
            {"tool": "approach_candidate", "data": {"pose": [-2.0, 0.0, 1.0],
                                                    "target_in_fov": False}},
        ]), VERIFY.CONFIG_TOOL, start_pose=START_POSE)
        stats = VERIFY.vision_stats(events, TARGET, hfov_deg=86.0,
                                    start_pose=START_POSE, start_yaw_deg=0.0)
        self.assertEqual(stats["target_lost_events"], 1)
        self.assertEqual(stats["target_reacquired_events"], 0)
        self.assertAlmostEqual(stats["reacquisition_rate"], 0.0)

    def test_estimated_source_falls_back_to_geometry(self):
        events = VERIFY.timeline(_tool_trace([
            {"tool": "approach_candidate", "data": {"pose": [2.0, 0.0, 1.0]}},
            {"tool": "approach_candidate", "data": {"pose": [0.0, 0.0, 1.0],
                                                    "heading_deg": 180.0}},
        ]), VERIFY.CONFIG_TOOL, start_pose=START_POSE)
        stats = VERIFY.vision_stats(events, TARGET, hfov_deg=86.0,
                                    start_pose=START_POSE, start_yaw_deg=0.0)
        self.assertEqual(stats["visibility_source"], "estimated")
        # seen at 2 m ahead, then lost: back at the start pose facing away
        self.assertEqual(stats["target_lost_events"], 1)
        self.assertEqual(stats["target_reacquired_events"], 0)
        self.assertIsNone(stats["fov_agreement"])
        self.assertEqual(stats["fov_measured_states"], 0)

    def test_fov_disagreement_is_reported_as_a_blocker_signal(self):
        events = VERIFY.timeline(_tool_trace([
            {"tool": "observe", "data": {"pose": START_POSE, "target_in_fov": False}},
            {"tool": "observe", "data": {"pose": START_POSE, "target_in_fov": False}},
        ]), VERIFY.CONFIG_TOOL, start_pose=START_POSE)
        stats = VERIFY.vision_stats(events, TARGET, hfov_deg=86.0,
                                    start_pose=START_POSE, start_yaw_deg=0.0)
        # the tool reports "not visible" while the target is 4 m straight ahead
        self.assertEqual(stats["fov_measured_states"], 2)
        self.assertEqual(stats["fov_agreement"], 0.0)
        self.assertEqual(stats["fov_mismatch_states"], 2)


# ---------------------------------------------------------------------------
# tool-call success, cost and completion
# ---------------------------------------------------------------------------


class CallStatsTest(unittest.TestCase):
    def test_tool_ok_rate_counts_every_status(self):
        trace = _tool_trace([
            {"tool": "observe", "status": STATUS_OK, "data": {"pose": START_POSE}},
            {"tool": "approach_candidate", "status": STATUS_OK,
             "data": {"pose": [1.0, 0.0, 1.0], "segment_progress_m": 1.0}},
            {"tool": "approach_candidate", "status": STATUS_EMPTY,
             "data": {"pose": [1.0, 0.0, 1.0], "reason": "unknown_observation"}},
            {"tool": "approach_candidate", "status": STATUS_BLOCKED,
             "data": {"pose": [1.0, 0.0, 1.0], "reason": "path_blocked"}},
            {"tool": "verify_goal", "status": STATUS_FAILED,
             "data": {"error": "no_target"}},
            {"tool": "observe", "status": STATUS_CACHED,
             "data": {"pose": [1.0, 0.0, 1.0]}},
            {"tool": "observe", "status": STATUS_OK, "parse_failure": True,
             "data": {"pose": [1.0, 0.0, 1.0]}},
            {"tool": "finish", "status": STATUS_OK, "data": {"reason": "done"}},
        ])
        events = VERIFY.timeline(trace, VERIFY.CONFIG_TOOL, start_pose=START_POSE)
        stats = VERIFY.call_stats(events, VERIFY.CONFIG_TOOL)
        self.assertEqual(stats["call_kind"], "tool_status")
        self.assertEqual(stats["calls"], 7)
        self.assertEqual(stats["ok"], 3)
        self.assertAlmostEqual(stats["ok_rate"], 0.429, places=3)
        self.assertEqual(stats["by_status"],
                         {"ok": 3, "empty": 1, "blocked": 1, "failed": 1, "cached": 1})
        self.assertEqual(stats["parse_failures"], 1)
        self.assertEqual(stats["calls_by_tool"], {"observe": 3, "approach_candidate": 3,
                                                  "verify_goal": 1})
        self.assertEqual(stats["failure_reasons"],
                         {"unknown_observation": 1, "path_blocked": 1,
                          "no_target": 1})

    def test_action_ok_rate_uses_parse_ok_and_failure_outcomes(self):
        trace = _action_trace([
            ("move fwd +1.00m", "moved 1.0m", (1.0, 0.0, 1.0), 0.0, 3.0),
            ("move fwd +1.00m", "collision: path blocked", (1.0, 0.0, 1.0), 0.0, 3.0),
            ("move fwd +1.00m", "moved 1.0m", (2.0, 0.0, 1.0), 0.0, 2.0, False),
        ])
        events = VERIFY.timeline(trace, VERIFY.CONFIG_ACTION, start_pose=START_POSE)
        stats = VERIFY.call_stats(events, VERIFY.CONFIG_ACTION)
        self.assertEqual(stats["call_kind"], "action_parse_outcome")
        self.assertEqual(stats["calls"], 3)
        self.assertEqual(stats["ok"], 1)
        self.assertAlmostEqual(stats["ok_rate"], 0.333, places=3)
        self.assertEqual(stats["parse_failures"], 1)
        self.assertEqual(stats["failure_reasons"], {"collision": 1})


class CostTest(unittest.TestCase):
    def test_vlm_cost_includes_calls_beyond_one_per_decision(self):
        args = _args()
        trace = _tool_trace([
            {"tool": "observe", "data": {"pose": START_POSE, "vlm_calls": 1},
             "images_sent": 2},
            {"tool": "ground_target", "data": {"pose": START_POSE, "vlm_calls": 1},
             "images_sent": 1},
            {"tool": "finish", "data": {"reason": "done"}},
        ])
        events = VERIFY.timeline(trace, VERIFY.CONFIG_TOOL, start_pose=START_POSE)
        stats = VERIFY.cost_stats(args, _tool_summary(turns=3, vlm_calls=3), events,
                                  VERIFY.CONFIG_TOOL)
        self.assertEqual(stats["vlm_calls_decision"], 3)
        self.assertEqual(stats["vlm_calls_reported"], 3)
        self.assertEqual(stats["vlm_calls_internal_declared"], 2)
        self.assertEqual(stats["vlm_calls_total"], 5)
        self.assertEqual(stats["vlm_calls_internal"], 2)
        self.assertTrue(stats["vlm_cost_complete"])
        self.assertEqual(stats["images_total"], 3)
        self.assertGreater(stats["tokens_estimate"], 0)

    def test_tool_cost_is_a_lower_bound_without_declared_internal_calls(self):
        args = _args()
        trace = _tool_trace([
            {"tool": "observe", "data": {"pose": START_POSE}},
            {"tool": "finish", "data": {"reason": "done"}},
        ])
        events = VERIFY.timeline(trace, VERIFY.CONFIG_TOOL, start_pose=START_POSE)
        stats = VERIFY.cost_stats(args, _tool_summary(turns=2, vlm_calls=2), events,
                                  VERIFY.CONFIG_TOOL)
        self.assertEqual(stats["vlm_calls_internal_declared"], 0)
        self.assertFalse(stats["vlm_cost_complete"])
        self.assertEqual(stats["vlm_calls_total"], 2)

    def test_action_cost_counts_scan_and_crop_calls_as_internal(self):
        args = _args()
        trace = _action_trace([
            ("move fwd +1.00m", "moved 1.0m", (1.0, 0.0, 1.0), 0.0, 3.0),
            ("scan (target lost)", "rescan", (1.0, 0.0, 1.0), 0.0, 3.0),
        ])
        events = VERIFY.timeline(trace, VERIFY.CONFIG_ACTION, start_pose=START_POSE)
        summary = {"success": False, "finish_reason": "max_steps", "steps": 2,
                   "vlm_calls": 5, "wall_time_s": 1.5}
        stats = VERIFY.cost_stats(args, summary, events, VERIFY.CONFIG_ACTION)
        self.assertEqual(stats["vlm_calls_decision"], 2)
        self.assertEqual(stats["vlm_calls_reported"], 5)
        self.assertEqual(stats["vlm_calls_internal"], 3)
        self.assertEqual(stats["vlm_calls_total"], 5)
        self.assertTrue(stats["vlm_cost_complete"])

    def test_tokens_and_wall_time_follow_the_budget_model(self):
        args = _args("--image-size", "140", "140", "--chars-per-token", "4",
                     "--prompt-chars-per-call", "0")
        trace = _tool_trace([
            {"tool": "observe", "data": {"pose": START_POSE}, "images_sent": 3,
             "result_prompt": "x" * 40, "raw_reply": "y" * 4},
            {"tool": "finish", "data": {"reason": "done"}},
        ])
        events = VERIFY.timeline(trace, VERIFY.CONFIG_TOOL, start_pose=START_POSE)
        stats = VERIFY.cost_stats(args, _tool_summary(turns=2, vlm_calls=2,
                                                      wall_time_s=7.5),
                                  events, VERIFY.CONFIG_TOOL)
        self.assertEqual(stats["image_tokens"], 25)
        self.assertEqual(stats["images_total"], 3)
        self.assertEqual(stats["text_chars"], 44)
        self.assertEqual(stats["tokens_estimate"], 3 * 25 + 11)
        self.assertAlmostEqual(stats["wall_time_s"], 7.5)

    def test_images_are_estimated_when_the_trace_has_no_image_counts(self):
        args = _args("--context-frames", "2")
        trace = [{"turn": 1, "tool": "observe", "data": {"pose": START_POSE},
                  "result_prompt": "", "model_reply": "{}"},
                 {"turn": 2, "tool": "finish", "data": {}, "result_prompt": "",
                  "model_reply": "{}"}]
        events = VERIFY.timeline(trace, VERIFY.CONFIG_TOOL, start_pose=START_POSE)
        stats = VERIFY.cost_stats(args, _tool_summary(turns=2, vlm_calls=2), events,
                                  VERIFY.CONFIG_TOOL)
        self.assertEqual(stats["images_total"], 2 * (2 + 1))


class SuccessTest(unittest.TestCase):
    def _stats(self, args, summary, poses):
        events = VERIFY.timeline(
            _tool_trace([_nav("approach_candidate", pose) for pose in poses]),
            VERIFY.CONFIG_TOOL, start_pose=START_POSE,
        )
        return VERIFY.success_stats(args, summary, events, TARGET, START_POSE)

    def test_reach_uses_the_ground_truth_distance_gate(self):
        args = _args()
        stats = self._stats(args, _tool_summary(success=True, final_distance_m=0.4),
                            [(2.0, 0.0, 1.0)])
        self.assertTrue(stats["reached"])
        self.assertTrue(stats["runner_success"])
        self.assertTrue(stats["solved"])
        self.assertFalse(stats["claim_without_reach"])
        self.assertAlmostEqual(stats["final_distance_m"], 2.0, places=3)
        self.assertAlmostEqual(stats["gate_m"], 2.5, places=3)

    def test_a_claim_without_reach_is_not_a_completed_task(self):
        args = _args()
        stats = self._stats(args, _tool_summary(success=True, finish_reason="finish"),
                            [(0.0, 0.0, 1.0)])
        self.assertFalse(stats["reached"])
        self.assertTrue(stats["runner_success"])
        self.assertFalse(stats["solved"])
        self.assertTrue(stats["claim_without_reach"])

    def test_reach_without_a_claim_is_reported_separately(self):
        args = _args()
        stats = self._stats(args, _tool_summary(success=False, finish_reason="max_turns"),
                            [(2.0, 0.0, 1.0)])
        self.assertTrue(stats["reached"])
        self.assertFalse(stats["runner_success"])
        self.assertFalse(stats["solved"])
        self.assertFalse(stats["claim_without_reach"])

    def test_missing_pose_keeps_distance_unknown(self):
        args = _args()
        events = VERIFY.timeline(
            _tool_trace([{"tool": "approach_candidate", "data": {"progress_m": 1.0}}]),
            VERIFY.CONFIG_TOOL, start_pose=None,
        )
        stats = VERIFY.success_stats(args, _tool_summary(success=False,
                                                         final_distance_m=None),
                                     events, TARGET, None)
        self.assertIsNone(stats["final_distance_m"])
        self.assertIsNone(stats["reached"])
        self.assertFalse(stats["solved"])


# ---------------------------------------------------------------------------
# paired command construction and metric extraction
# ---------------------------------------------------------------------------


class CommandTest(unittest.TestCase):
    def _episodes(self, args, tasks=None):
        return VERIFY.build_episodes(args, tasks or [_task()])

    def _commands(self, args, episodes):
        return {
            (spec.config, spec.task.name, spec.seed, spec.yaw_mode):
                VERIFY.build_command(args, spec,
                                     VERIFY.compare.episode_dir(Path(args.out), spec))
            for spec in episodes
        }

    def test_two_agents_share_every_paired_option(self):
        args = _args("--seeds", "7,11", "--initial-yaw-modes", "visible,random")
        episodes = self._episodes(args)
        self.assertEqual({spec.config for spec in episodes},
                         {VERIFY.CONFIG_ACTION, VERIFY.CONFIG_TOOL})
        commands = self._commands(args, episodes)
        check = VERIFY.pairing_check(args, episodes, commands)
        self.assertEqual(check["cells"], 4)
        self.assertTrue(check["consistent"], check["issues"])
        action = VERIFY.compare._args_map(commands[("action", "book", 7, "visible")])
        tool = VERIFY.compare._args_map(commands[("tool", "book", 7, "visible")])
        for key in ("--world", "--ply", "--target-pos", "--start-pos",
                    "--start-yaw-deg", "--seed", "--success-dist", "--instruction",
                    "--context-frames", "--image-size", "--target-radius"):
            self.assertEqual(action[key], tool[key], key)
        self.assertEqual(action["--backend"], "mock:oracle")
        self.assertEqual(tool["--backend"], "mock")
        self.assertEqual(action["--max-steps"], "20")
        self.assertEqual(tool["--max-turns"], "20")
        self.assertEqual(tool["--tool-budget"], "20")
        self.assertEqual(tool["--toolset"], "full")
        self.assertNotIn("--render-size", action)

    def test_budget_mapping_and_hf_model_flags(self):
        args = _args("--backend", "hf", "--model-path", "/models_ro/Qwen3-VL-8B",
                     "--max-vlm-calls", "12", "--tool-budget", "30",
                     "--device", "cuda")
        mapping = VERIFY.budget_mapping(args)
        self.assertEqual(mapping["max_vlm_calls"], 12)
        self.assertEqual(mapping["action"]["max_steps"], 12)
        self.assertEqual(mapping["tool"]["max_turns"], 12)
        self.assertEqual(mapping["tool"]["tool_budget"], 30)
        spec = self._episodes(args)[0]
        for command in (VERIFY.build_action_command(args, spec, Path("/tmp/x")),
                        VERIFY.build_tool_command(args, spec, Path("/tmp/x"))):
            self.assertIn("--model-path", command)
            self.assertIn("/models_ro/Qwen3-VL-8B", command)
            self.assertEqual(
                VERIFY.compare._args_map(command)["--max-vlm-calls"] if False else
                VERIFY.compare._args_map(command)["--instruction"], INSTRUCTION)

    def test_dry_run_lists_both_agents(self):
        args = _args("--dry-run", "--seeds", "7")
        summary = VERIFY.run_verification(args)
        self.assertTrue(summary["dry_run"])
        self.assertEqual(summary["n_episodes"], 2)
        commands = [VERIFY.compare._args_map(line.split()) for line in summary["commands"]]
        self.assertEqual(sum(("--max-steps" in cmd) for cmd in commands), 1)
        self.assertEqual(sum(cmd.get("--toolset") == "full" for cmd in commands), 1)


class MetricExtractionTest(unittest.TestCase):
    def _row(self, out_dir: Path, config: str, summary: dict, trace: list[dict]):
        args = _args()
        (out_dir).mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary))
        name = "tool_trace.jsonl" if config == VERIFY.CONFIG_TOOL else "episode_trace.jsonl"
        (out_dir / name).write_text("\n".join(json.dumps(row) for row in trace))
        return VERIFY.extract_metrics(
            args, _spec(config), out_dir,
            {"returncode": 0, "error": None, "elapsed_s": 1.0})

    def test_tool_episode_row_carries_every_metric_family(self):
        trace = _tool_trace([
            {"tool": "observe", "data": {"pose": START_POSE, "target_in_fov": True}},
            _nav("approach_candidate", (1.5, 0.0, 1.0), segment_progress_m=1.5,
                 outcome="moved", target_in_fov=True),
            _nav("return_to_observation", (1.5, 0.0, 1.0), outcome="already_there",
                 target_in_fov=True),
            _nav("return_to_observation", (1.5, 0.0, 1.0), outcome="already_there",
                 target_in_fov=False),
            {"tool": "observe", "data": {"pose": [1.5, 0.0, 1.0], "target_in_fov": True}},
            _nav("approach_candidate", (2.2, 0.0, 1.0), segment_progress_m=0.7,
                 outcome="moved", target_in_fov=True),
            {"tool": "verify_goal", "status": STATUS_OK,
             "data": {"verified": True, "pose": [2.2, 0.0, 1.0], "target_in_fov": True}},
            {"tool": "finish", "data": {"reason": "target verified"}},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(Path(tmp), VERIFY.CONFIG_TOOL, _tool_summary(), trace)
        self.assertEqual(row["config"], "tool")
        self.assertEqual(row["call_kind"], "tool_status")
        self.assertEqual(row["tool_calls"], 7)
        self.assertEqual(row["tool_ok"], 7)
        self.assertAlmostEqual(row["tool_ok_rate"], 1.0)
        self.assertEqual(row["required_tools_missing"], [])
        self.assertEqual(row["nav_attempts"], 4)
        self.assertEqual(row["effective_moves"], 2)
        self.assertEqual(row["zero_displacement_events"], 2)
        self.assertEqual(row["zero_displacement_runs"], 1)
        self.assertAlmostEqual(row["effective_move_rate"], 0.5, places=3)
        self.assertEqual(row["target_lost_events"], 1)
        self.assertEqual(row["target_reacquired_events"], 1)
        self.assertAlmostEqual(row["reacquisition_rate"], 1.0)
        self.assertTrue(row["reached"])
        self.assertTrue(row["solved"])
        self.assertEqual(row["vlm_calls_decision"], 8)
        self.assertIsNotNone(row["tokens_estimate"])
        self.assertIsNone(row["error"])

    def test_action_episode_row_uses_the_action_trace_schema(self):
        trace = _action_trace([
            ("move fwd +1.00m", "moved 1.0m", (1.0, 0.0, 1.0), 0.0, 3.0),
            ("move fwd +1.20m", "moved 1.2m", (2.2, 0.0, 1.0), 0.0, 1.8),
            ("move fwd +1.00m", "moved 1.0m", (3.2, 0.0, 1.0), 0.0, 0.8),
        ])
        summary = {"success": True, "finish_reason": "within_success_distance",
                   "steps": 3, "vlm_calls": 5, "wall_time_s": 4.0,
                   "final_distance_m": 0.8, "collision_hits": 0, "parse_failures": 0,
                   "initial_target_visible": True, "min_distance_m": 0.8,
                   "path_length_m": 3.2}
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(Path(tmp), VERIFY.CONFIG_ACTION, summary, trace)
        self.assertEqual(row["config"], "action")
        self.assertEqual(row["call_kind"], "action_parse_outcome")
        self.assertEqual(row["nav_attempts"], 3)
        self.assertEqual(row["effective_moves"], 3)
        self.assertAlmostEqual(row["effective_move_rate"], 1.0)
        self.assertEqual(row["zero_displacement_runs"], 0)
        self.assertTrue(row["reached"])
        self.assertTrue(row["solved"])
        self.assertEqual(row["vlm_calls_decision"], 3)
        self.assertEqual(row["vlm_calls_total"], 5)
        self.assertIsNone(row["tool_ok_rate"])
        self.assertIsNone(row["tool_calls"])
        self.assertEqual(row["required_tools_missing"], [])

    def test_missing_artifacts_become_an_error_row(self):
        args = _args()
        with tempfile.TemporaryDirectory() as tmp:
            row = VERIFY.extract_metrics(args, _spec(), Path(tmp),
                                         {"returncode": 3, "error": None})
        self.assertIn("missing summary.json", row["error"])
        self.assertEqual(row["returncode"], 3)
        self.assertFalse(row["solved"])


# ---------------------------------------------------------------------------
# aggregation, verdict and report
# ---------------------------------------------------------------------------


def _row(config: str, **overrides) -> dict:
    row = {
        "config": config, "task": "book", "seed": 7, "yaw_mode": "visible",
        "runner_success": True, "success": True, "solved": True, "reached": True,
        "claim_without_reach": False, "finish_reason": "finish",
        "nav_attempts": 4, "effective_moves": 4, "effective_move_rate": 1.0,
        "zero_displacement_events": 0, "zero_displacement_runs": 0,
        "max_zero_displacement_run": 0, "wrong_direction_moves": 0,
        "target_lost_events": 1, "target_reacquired_events": 1,
        "reacquisition_rate": 1.0, "fov_measured_states": 4, "fov_agreement": 1.0,
        "calls": 7 if config == VERIFY.CONFIG_TOOL else 4, "call_ok": 7,
        "call_ok_rate": 1.0, "parse_failures": 0,
        "vlm_calls_decision": 8, "vlm_calls_total": 9, "vlm_calls_internal": 1,
        "vlm_cost_complete": True, "tokens_estimate": 1000, "wall_time_s": 2.0,
        "images_total": 10, "error": None, "skipped": False,
        "required_tools_missing": [],
    }
    if config == VERIFY.CONFIG_TOOL:
        row.update(tool_calls=7, tool_ok=7, tool_ok_rate=1.0, toolset="full")
    else:
        row.update(tool_calls=None, tool_ok=None, tool_ok_rate=None, steps=4)
    row.update(overrides)
    return row


class AggregateTest(unittest.TestCase):
    def _rows(self):
        return [
            _row(VERIFY.CONFIG_ACTION, effective_moves=1, effective_move_rate=0.25,
                 zero_displacement_events=3, zero_displacement_runs=2,
                 max_zero_displacement_run=3),
            _row(VERIFY.CONFIG_TOOL, tool_calls=7, tool_ok=6, tool_ok_rate=0.857),
        ]

    def test_aggregate_per_agent_pools_the_rates(self):
        stats = VERIFY.aggregate(self._rows())
        self.assertEqual(stats["tool"]["n"], 1)
        self.assertEqual(stats["action"]["n"], 1)
        self.assertAlmostEqual(stats["tool"]["effective_move_rate_pooled"], 1.0)
        self.assertAlmostEqual(stats["action"]["effective_move_rate_pooled"], 0.25)
        self.assertEqual(stats["action"]["zero_displacement_runs_sum"], 2)
        self.assertEqual(stats["action"]["zero_displacement_episodes"], 1)
        self.assertEqual(stats["tool"]["n_solved"], 1)
        self.assertEqual(stats["action"]["n_solved"], 1)
        self.assertEqual(stats["tool"]["success_rate"], 1.0)
        self.assertEqual(stats["tool"]["vlm_calls_total_mean"], 9.0)
        self.assertAlmostEqual(stats["tool"]["tool_ok_rate_pooled"], 0.857, places=3)
        self.assertIsNone(stats["action"]["tool_ok_rate_pooled"])

    def test_paired_cells_and_verdict_compare_the_two_agents(self):
        rows = self._rows()
        rows[1].update(solved=False, success=False, runner_success=False,
                       reached=False, required_tools_missing=["approach_candidate"])
        cells = VERIFY.paired_cells(rows)
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0]["cell"], "book__visible__seed7")
        self.assertTrue(cells[0]["action"]["solved"])
        self.assertFalse(cells[0]["tool"]["solved"])
        verdict = VERIFY.verdict(rows, VERIFY.aggregate(rows), cells, _args())
        self.assertEqual(verdict["action_solved_cells"], 1)
        self.assertEqual(verdict["tool_solved_action_cells"], 0)
        self.assertFalse(verdict["checks"]["success_parity"])
        self.assertFalse(verdict["checks"]["required_tools"])
        self.assertFalse(verdict["pass"])
        self.assertEqual(verdict["zero_displacement_loops_total"],
                         {"action": 2, "tool": 0})
        self.assertEqual(verdict["required_tools_missing"], ["approach_candidate"])

    def test_verdict_passes_when_the_tool_agent_matches_and_moves(self):
        rows = [_row(VERIFY.CONFIG_ACTION), _row(VERIFY.CONFIG_TOOL)]
        cells = VERIFY.paired_cells(rows)
        verdict = VERIFY.verdict(rows, VERIFY.aggregate(rows), cells, _args())
        self.assertTrue(verdict["pass"], verdict)
        self.assertTrue(all(check is not False for check in verdict["checks"].values()))

    def test_verdict_fails_on_a_zero_displacement_loop(self):
        args = _args()
        rows = [_row(VERIFY.CONFIG_ACTION), _row(
            VERIFY.CONFIG_TOOL, zero_displacement_events=4, zero_displacement_runs=2,
            max_zero_displacement_run=3)]
        cells = VERIFY.paired_cells(rows)
        verdict = VERIFY.verdict(rows, VERIFY.aggregate(rows), cells, args)
        self.assertFalse(verdict["checks"]["zero_displacement"])
        self.assertFalse(verdict["pass"])
        # a wider allowance makes the same run pass that check
        verdict = VERIFY.verdict(rows, VERIFY.aggregate(rows), cells,
                                 _args("--max-zero-displacement-loops", "2"))
        self.assertTrue(verdict["checks"]["zero_displacement"])

    def test_verdict_requires_the_documented_effective_move_rate(self):
        rows = [_row(VERIFY.CONFIG_ACTION),
                _row(VERIFY.CONFIG_TOOL, effective_moves=1, effective_move_rate=0.25)]
        cells = VERIFY.paired_cells(rows)
        verdict = VERIFY.verdict(rows, VERIFY.aggregate(rows), cells, _args())
        self.assertFalse(verdict["checks"]["effective_move_rate"])
        self.assertFalse(verdict["pass"])
        verdict = VERIFY.verdict(rows, VERIFY.aggregate(rows), cells,
                                 _args("--min-effective-move-rate", "0.2"))
        self.assertTrue(verdict["checks"]["effective_move_rate"])

    def test_report_holds_the_required_metric_columns(self):
        rows = self._rows()
        args = _args()
        aggregates = VERIFY.aggregate(rows)
        cells = VERIFY.paired_cells(rows)
        verdict = VERIFY.verdict(rows, aggregates, cells, args)
        meta = VERIFY.meta_block(args, [_task()], rows, verdict)
        markdown = VERIFY.render_markdown(meta, rows, aggregates, cells, verdict)
        for column in ("effective_move_rate", "zero_displacement_loops",
                       "reacquisition_rate", "tool_ok_rate", "vlm_calls_total",
                       "tokens_est", "wall_s", "fov_agreement"):
            self.assertIn(column, markdown, column)
        self.assertIn("| tool | all |", markdown)
        self.assertIn("| action | all |", markdown)
        self.assertIn("tool_solved_action_cells", markdown)
        self.assertIn("Verdict", markdown)


# ---------------------------------------------------------------------------
# contract fake: the Phase-1 toolset through the real tool runner
# ---------------------------------------------------------------------------


class _StubTarget:
    name = "target"
    radius_m = 0.5
    position_flu = np.asarray(TARGET, dtype=np.float64)


class _StubWorld:
    """Minimal world: pose, target geometry, one-action leg of 1.6 m."""

    LEG_M = 1.6

    def __init__(self, start=START_POSE):
        self.position_flu = np.asarray(start, dtype=np.float64)
        self.yaw_rad = 0.0
        self.target = _StubTarget()
        self.trajectory = [tuple(float(v) for v in self.position_flu)]

    def render(self):
        import torch

        return torch.zeros(3, 32, 32)

    def apply_action(self, action):
        before = np.asarray(self.position_flu, dtype=np.float64)
        delta = self.target.position_flu - before
        step = np.zeros(3)
        horizontal = float(np.linalg.norm(delta[:2]))
        if horizontal > 0.0:
            step[:2] = delta[:2] / horizontal * min(self.LEG_M, horizontal)
        self.position_flu = before + step
        self.trajectory.append(tuple(float(v) for v in self.position_flu))
        return "moved" if step.any() else "stalled"

    def distance_to_target(self):
        return float(np.linalg.norm(self.target.position_flu - self.position_flu))

    def target_in_fov(self):
        bearing = math.degrees(math.atan2(
            self.target.position_flu[1] - self.position_flu[1],
            self.target.position_flu[0] - self.position_flu[0]))
        return abs(VERIFY.compare._wrap_deg(bearing - math.degrees(self.yaw_rad))) <= 43.0


def _stub_registry(modules=None, vlm=None):
    """The Phase-1 toolset as a contract fake: names and result schema only."""
    registry = Runner.RecordingToolRegistry()

    def _pose(ctx):
        return [float(v) for v in np.asarray(ctx.world.position_flu).reshape(-1)[:3]]

    def observe(ctx, args):
        return Runner.ToolResult(
            STATUS_OK,
            {"observation_id": ctx.next_obs_id(), "pose": _pose(ctx),
             "heading_deg": 0.0, "target_in_fov": bool(ctx.world.target_in_fov()),
             "vlm_calls": 1},
            [EvidenceRef("observation", "obs_1", "view")],
        )

    def ground_target(ctx, args):
        return Runner.ToolResult(
            STATUS_OK,
            {"candidate_id": "cand_1", "observation_id": "obs_1", "pose": _pose(ctx),
             "box": [400, 400, 500, 500], "box_format_used": "[x1,y1,x2,y2]",
             "target_in_fov": True, "vlm_calls": 1},
            [EvidenceRef("observation", "obs_1", "grounded view")],
        )

    def approach_candidate(ctx, args):
        before = np.asarray(ctx.world.position_flu, dtype=np.float64)
        ctx.world.apply_action(None)
        after = np.asarray(ctx.world.position_flu, dtype=np.float64)
        moved = float(np.linalg.norm(after - before))
        return Runner.ToolResult(
            STATUS_OK,
            {"task_id": "nav_1", "status": "running", "outcome": "moved",
             "candidate_id": "cand_1", "pose": [float(v) for v in after],
             "heading_deg": 0.0, "segment_progress_m": moved, "progress_m": moved,
             "remaining_m": float(np.linalg.norm(self_target(ctx) - after)),
             "target_in_fov": bool(ctx.world.target_in_fov())},
            [EvidenceRef("trajectory", "nav_1", "leg")],
        )

    def self_target(ctx):
        return np.asarray(ctx.world.target.position_flu, dtype=np.float64)

    def already_there(ctx, args):
        return Runner.ToolResult(
            STATUS_OK,
            {"task_id": "nav_2", "status": "arrived", "reason": "already_there",
             "outcome": "already_there", "pose": _pose(ctx), "heading_deg": 0.0,
             "segment_progress_m": 0.0,
             "target_in_fov": bool(ctx.world.target_in_fov())},
            [EvidenceRef("observation", "obs_1", "already there")],
        )

    def verify_goal(ctx, args):
        distance = float(ctx.world.distance_to_target())
        return Runner.ToolResult(
            STATUS_OK,
            {"verified": distance <= 2.5, "satisfied": distance <= 2.5,
             "distance_m": distance, "pose": _pose(ctx), "target_in_fov": True},
            [EvidenceRef("observation", "obs_1", "verification")],
        )

    for name, handler, side_effects in (
        ("observe", observe, False),
        ("ground_target", ground_target, False),
        ("approach_candidate", approach_candidate, True),
        ("go_to_viewpoint", already_there, True),
        ("return_to_observation", already_there, True),
        ("verify_goal", verify_goal, False),
    ):
        registry.register(ToolSpec(
            name=name, description=f"stub {name}",
            input_schema={"type": "object", "properties": {}},
            handler=handler, side_effects=side_effects,
        ))
    return registry, "stub", {name: "stub" for name in registry.names()}


class ContractFakeEpisodeTest(unittest.TestCase):
    """Run the Phase-1 tool contract through the real runner, then measure it."""

    SCRIPT = [
        {"tool": "observe", "args": {"reason": "look"}},
        {"tool": "ground_target", "args": {"observation_id": "obs_1"}},
        {"tool": "approach_candidate", "args": {"candidate_id": "cand_1"}},
        {"tool": "return_to_observation", "args": {"observation_id": "obs_1"}},
        {"tool": "return_to_observation", "args": {"observation_id": "obs_1"}},
        {"tool": "verify_goal", "args": {"reason": "check"}},
        {"tool": "finish", "args": {"reason": "verified", "success": True}},
    ]

    def _run(self, out_dir: Path) -> dict:
        args = Runner.parse_args([
            "--world", "cpu",
            "--target-pos", "4", "0", "1",
            "--instruction", INSTRUCTION,
            "--backend", "mock",
            "--max-turns", "10",
            "--tool-budget", "20",
            "--out", str(out_dir),
        ])
        vlm = Runner.ScriptedToolVLM([dict(entry) for entry in self.SCRIPT])
        return Runner.run_episode(_StubWorld(), vlm, args, out_dir,
                                  episode_id="ep_contract",
                                  registry_factory=_stub_registry)

    def test_metrics_lock_the_tool_names_and_result_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            summary = self._run(out_dir)
            row = VERIFY.extract_metrics(_args(), _spec(), out_dir,
                                         {"returncode": 0, "error": None})
            self.assertIsNone(row["error"])
            self.assertEqual(row["required_tools_missing"], [])
            self.assertEqual(row["toolset"], "full")
            self.assertEqual(row["tool_calls"], 6)
            self.assertEqual(row["tool_ok"], 6)
            self.assertAlmostEqual(row["tool_ok_rate"], 1.0)
            # every Phase-1 tool the stub registers is available; the scripted
            # episode calls five of them (go_to_viewpoint stays uncalled)
            self.assertEqual(set(row["calls_by_tool"]),
                             set(VERIFY.PHASE1_TOOLS) - {"go_to_viewpoint"})
            self.assertEqual(summary["toolset_detail"]["registered"],
                             sorted(VERIFY.PHASE1_TOOLS))
            # one real leg towards the target, then the already_there replay loop
            self.assertEqual(row["nav_attempts"], 3)
            self.assertEqual(row["effective_moves"], 1)
            self.assertEqual(row["zero_displacement_events"], 2)
            self.assertEqual(row["zero_displacement_runs"], 1)
            self.assertEqual(row["max_zero_displacement_run"], 2)
            self.assertEqual(row["nav_failure_reasons"], {"already_there": 2})
            # tool-internal VLM calls declared through the contract are counted
            self.assertGreaterEqual(row["vlm_calls_internal_declared"], 2)
            self.assertTrue(row["vlm_cost_complete"])
            self.assertGreater(row["vlm_calls_total"], row["vlm_calls_decision"])
            self.assertTrue(row["reached"], row)
            self.assertTrue(summary["success"])


# ---------------------------------------------------------------------------
# end to end (mock backends, tiny CPU world)
# ---------------------------------------------------------------------------


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _make_scene(path: Path) -> None:
    header = "\n".join([
        "ply", "format binary_little_endian 1.0", "element vertex 123",
        "property float x", "property float y", "property float z",
        "property float f_dc_0", "property float f_dc_1", "property float f_dc_2",
        "property float opacity", "property float scale_0", "property float scale_1",
        "property float scale_2", "property float rot_0", "property float rot_1",
        "property float rot_2", "property float rot_3", "end_header",
    ])
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(120):
        point = rng.uniform(-6, 6, size=3)
        point[2] = -1.5
        rows.append(list(point) + [0.0, 0.0, 0.0, _logit(0.8)]
                    + [math.log(0.2), math.log(0.2), math.log(0.05)]
                    + [1.0, 0.0, 0.0, 0.0])
    for offset in ((0.0, 0.0, 0.0), (0.15, 0.1, 0.0), (0.0, 0.2, 0.1)):
        point = np.asarray(TARGET) + np.asarray(offset)
        rows.append(list(point) + [2.0, -1.0, -1.0, _logit(0.9)]
                    + [math.log(0.25), math.log(0.25), math.log(0.25)]
                    + [1.0, 0.0, 0.0, 0.0])
    with open(path, "wb") as handle:
        handle.write((header + "\n").encode("ascii"))
        for row in rows:
            handle.write(struct.pack("<14f", *row))


class EndToEndTest(unittest.TestCase):
    def test_mock_run_writes_the_paired_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene.ply"
            _make_scene(scene)
            out_dir = Path(tmp) / "verify"
            cmd = [
                sys.executable, "-m", "scripts.verify_phase1",
                "--world", "cpu", "--ply", str(scene),
                "--target-name", "target",
                "--target-pos", "4", "0", "1", "--target-radius", "0.5",
                "--start-pos", "0", "0", "1",
                "--instruction", INSTRUCTION,
                "--backend", "mock",
                "--seeds", "7",
                "--initial-yaw-modes", "visible",
                "--max-vlm-calls", "3",
                "--tool-budget", "4",
                "--scan-views", "4",
                "--success-dist", "3.0",
                "--jobs", "2",
                "--out", str(out_dir),
            ]
            env = dict(os.environ)
            env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True,
                                  text=True, timeout=900)
            diagnostics = (proc.stdout[-4000:] + "\n--- stderr ---\n" + proc.stderr[-2000:])
            # 0 = verdict pass, 1 = verdict failed (the report says which check)
            self.assertIn(proc.returncode, (0, 1), diagnostics)

            report = out_dir / "verify_phase1_report.json"
            markdown = out_dir / "verify_phase1_report.md"
            self.assertTrue(report.exists(), diagnostics)
            self.assertTrue(markdown.exists(), diagnostics)
            data = json.loads(report.read_text())
            self.assertEqual(data["n_errors"], 0, diagnostics)
            self.assertEqual(len(data["episodes"]), 2, diagnostics)
            self.assertEqual({row["config"] for row in data["episodes"]},
                             {"action", "tool"}, diagnostics)
            self.assertEqual(data["verdict"]["paired_cells"], 1)
            self.assertEqual(proc.returncode, 0 if data["verdict"]["pass"] else 1,
                             diagnostics)
            for row in data["episodes"]:
                self.assertEqual(row["returncode"], 0, row)
                for key in ("effective_move_rate", "zero_displacement_runs",
                            "tool_ok_rate", "reacquisition_rate", "vlm_calls_total",
                            "tokens_estimate", "wall_time_s", "reached", "solved",
                            "required_tools_missing"):
                    self.assertIn(key, row, key)
                self.assertTrue(Path(row["out_dir"], "summary.json").exists(), row)
                self.assertTrue(Path(row["out_dir"], "run.log").exists(), row)
            self.assertTrue(data["meta"]["pairing_check"]["consistent"], diagnostics)
            self.assertEqual(data["meta"]["budget"]["action"]["max_steps"], 3)
            self.assertEqual(data["meta"]["budget"]["tool"]["max_turns"], 3)
            text = markdown.read_text()
            self.assertIn("effective_move_rate", text)
            self.assertIn("Verdict", text)
            self.assertIn("| tool | all |", text)


if __name__ == "__main__":
    unittest.main()
