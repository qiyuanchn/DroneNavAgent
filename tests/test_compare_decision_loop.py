"""Tests for the decision-loop comparison driver (``scripts/compare_decision_loop.py``).

The question this driver answers is *which layer of the harness upgrade pays*:
baseline vs. multi-view visual context vs. failure recovery, with the model,
scene, start pose/heading and budgets held fixed. The tests pin the pieces that
define the experiment:

  arm switches     the ``baseline``/``vision_context``/``recovery``/``memory``/
                   ``memory_spatial_only`` table, the probe that filters declared
                   switches to the ones the tool runner really has, ``--arm-arg``
                   overrides, the reserved-arm machinery and ``--strict-arms``
  scenarios        the ``book``/``synth`` presets, the paired episode matrix
                   (same world/start/heading/budget per cell across arms) and
                   the pairing check
  metrics          solved vs. claim, the false-confirmation rate (a claim is an
                   explicit declaration: a bare finish is an honest give-up, not
                   a claim), the repeat search ratio (same region / same
                   evidence), the reported and estimated evidence coverage, the
                   total VLM cost including tool-internal calls, and the
                   toward-target diagnostic
  annotation       the causal chain extracted from the tool trace, the
                   ``progress_kind`` field and the search skill's ``steps`` log
                   ("why the viewpoint changed, what was learned, what changed")
  contract stub    a stub registry with the observation/search result contract
                   run through the *real* tool runner, then measured by the
                   driver's extractor (fields, not prose)
  end to end       the driver as a subprocess on a tiny CPU world with
                   ``--backend mock``: three arms run, the report and the
                   per-episode annotations are written

No weights and no GPU: the end-to-end test only needs the CPU world.
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
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = ROOT / "scripts" / "compare_decision_loop.py"
TOOL_RUNNER_PATH = ROOT / "scripts" / "run_vlm_nav_tools.py"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve string annotations via sys.modules
    spec.loader.exec_module(module)
    return module


DRIVER = _load("diffaero_script_compare_decision_loop", DRIVER_PATH)
Runner = _load("diffaero_script_run_vlm_nav_tools_for_decision", TOOL_RUNNER_PATH)

from vlm_nav.tools import (  # noqa: E402  (after the runner load)
    STATUS_BLOCKED,
    STATUS_CACHED,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_OK,
    EvidenceRef,
    ToolResult,
    ToolSpec,
)

TARGET = (2.0, 0.5, 0.0)
TASK_JSON = {
    "name": "target0",
    "instruction": "Find the target and fly to it.",
    "target_pos": list(TARGET),
    "target_name": "target",
    "target_radius": 0.5,
    "start_pos": [0.0, 0.0, 1.0],
    "success_dist": 1.0,
}


def _args(*extra: str, **overrides):
    argv = [
        "--scenario", "cpu",
        "--world", "cpu",
        "--ply", "/scene.ply",
        "--task-json", json.dumps(TASK_JSON),
        "--out", "/tmp/decision_loop",
        *extra,
    ]
    args = DRIVER.parse_args(argv)
    DRIVER.resolve_matrix(args)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _switch_map(tokens) -> dict:
    """``--flag value`` pairs of an arm's switch list, as a dict."""
    switches = ("--context-frames", "--scan-views", "--recovery", "--state-block",
                "--search-skill", "--toolset", "--memory-spatial",
                "--memory-episodic")
    items = [str(token) for token in tokens]
    out: dict = {}
    for index, token in enumerate(items):
        if token in switches and index + 1 < len(items):
            out[token] = items[index + 1]
    return out


def _task(**overrides) -> "DRIVER.compare.TaskSpec":
    data = dict(TASK_JSON)
    data.update(overrides)
    return DRIVER.compare.TaskSpec.from_dict(data, data.get("success_dist"))


def _scenario(tasks=None) -> "DRIVER.ScenarioSpec":
    return DRIVER.ScenarioSpec(name="cpu", world="cpu", ply="/scene.ply",
                               tasks=list(tasks or [_task()]))


def _spec(arm: str = "vision_context", task=None, seed: int = 7,
          yaw_mode: str = "visible", yaw: float = 0.0) -> "DRIVER.EpisodeSpec":
    return DRIVER.EpisodeSpec(arm, "cpu", task or _task(), seed, yaw_mode, yaw)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _entry(turn: int, tool: str, data: dict, *, args: dict | None = None,
           status: str = STATUS_OK, new_evidence: bool | None = True,
           image_refs=None, **extra) -> dict:
    """One tool-trace row in the module runtime's schema."""
    row = {
        "turn": turn,
        "tool": tool,
        "args": args or {},
        "status": status,
        "new_evidence": new_evidence,
        "images_sent": extra.pop("images_sent", 1),
        "data": dict(data),
        "evidence": [{"kind": "observation", "ref": "obs_1", "note": ""}],
        "result_prompt": f"[{status}] tool={tool}",
        "model_reply": json.dumps({"tool": tool, "args": args or {}}),
    }
    row.update(extra)
    return row


def _summary(**overrides) -> dict:
    data = {
        "success": True,
        "finish_reason": "finish",
        "turns": 4,
        "tool_calls": 4,
        "vlm_calls": 4,
        "wall_time_s": 12.0,
        "final_distance_m": 1.0,
        "success_dist_m": 1.5,
        "new_evidence_total": 6,
        "parse_failures": 0,
        "toolset": "full",
    }
    data.update(overrides)
    return data


def _row_from_trace(tmp: str, trace: list[dict], summary: dict | None = None,
                    task=None, arm: str = "vision_context"):
    """Run the driver's per-episode extraction over a synthetic trace."""
    args = _args("--max-vlm-calls", "10")
    out_dir = Path(tmp) / "ep"
    _write_json(out_dir / "summary.json", summary or _summary())
    _write_jsonl(out_dir / DRIVER.TOOL_TRACE, trace)
    spec = _spec(arm, task)
    row = DRIVER.extract_metrics(args, _scenario(), spec, out_dir,
                                 {"returncode": 0, "error": None})
    return row


RICH_TRACE = [
    _entry(1, "observe", {"observation_id": "obs_2", "pose": [0.0, 0.0, 1.0],
                          "view_index": 3, "heading_deg": 0.0,
                          "coverage": {"checked_deg": 86.0, "unchecked_deg": 274.0,
                                       "worth_looking": True}},
           args={"mode": "lookaround", "n_views": 8, "reason": "target not visible"},
           progress_kind="information", images_sent=1),
    _entry(2, "ground_target", {"reason": "no_candidate", "pose": [0.0, 0.0, 1.0]},
           args={"observation_id": "obs_2", "reason": "locate the book"},
           status=STATUS_EMPTY, new_evidence=False, progress_kind="information"),
    _entry(3, "search_target", {"candidate_id": "cand_1", "observation_id": "obs_3",
                                "pose": [0.0, 0.0, 1.0], "vlm_calls": 4,
                                "outer_tool_calls": 1, "inner_tool_calls": 6,
                                "cost": {"outer_tool_calls": 1, "inner_tool_calls": 6,
                                         "tool_calls": 6, "vlm_calls": 4,
                                         "turns": 1, "budget_turns": 1, "inner_steps": 8},
                                "coverage": {"checked_deg": 172.0, "unchecked_deg": 188.0,
                                             "worth_looking": False},
                                "steps": [
                                    {"step": 1, "why": "grounding failed",
                                     "obs_ids": ["obs_2"], "result": "no_candidate"},
                                    {"step": 2, "why": "rotate to the unchecked sector",
                                     "obs_ids": ["obs_3"], "result": "candidate cand_1"},
                                ]},
           args={"reason": "scan the unchecked sector"}, progress_kind="information"),
    _entry(4, "go_to_viewpoint", {"viewpoint_id": "vp_1", "outcome": "moved",
                                  "segment_progress_m": 1.0, "pose": [1.0, 0.0, 1.0]},
           args={"viewpoint_id": "vp_1", "reason": "closer observation point"},
           progress_kind="action"),
    _entry(5, "approach_candidate", {"candidate_id": "cand_1", "outcome": "moved",
                                     "segment_progress_m": 1.4, "pose": [1.8, 0.4, 1.0]},
           args={"candidate_id": "cand_1", "reason": "close in"},
           progress_kind="action"),
    _entry(6, "verify_goal", {"verified": True, "distance_m": 0.4, "pose": [1.8, 0.4, 1.0]},
           args={"reason": "confirm"}, progress_kind="none"),
    _entry(7, "finish", {"reason": "done"}, args={"reason": "done"}, progress_kind="none"),
]


# ---------------------------------------------------------------------------
# arm switches: declared -> probed -> effective
# ---------------------------------------------------------------------------


class ArmSwitchTest(unittest.TestCase):
    def test_default_arms_are_the_three_layers_in_order(self):
        args = _args()
        self.assertEqual(args.arm_list, ["baseline", "vision_context", "recovery"])

    def test_the_memory_arms_are_wired_not_reserved(self):
        for name in ("memory", "memory_spatial_only"):
            with self.subTest(arm=name):
                self.assertIn(name, DRIVER.ARM_ORDER)
                self.assertFalse(DRIVER.ARM_TABLE[name].reserved)
        args = _args("--arms", "baseline,memory,memory_spatial_only")
        self.assertEqual(args.arm_list,
                         ["baseline", "memory", "memory_spatial_only"])
        self.assertEqual([record["arm"] for record in args.arm_support],
                         ["baseline", "memory", "memory_spatial_only"])
        for record in args.arm_support:
            self.assertEqual([], record["dropped"], record)
            self.assertEqual("full", record["support"], record)

    def test_the_memory_arms_add_the_layer_switches_on_top_of_recovery(self):
        args = _args("--arms", "recovery,memory,memory_spatial_only")
        switches = {record["arm"]: _switch_map(record["effective"])
                    for record in args.arm_support}
        # the full memory arm: both stores on
        self.assertEqual("on", switches["memory"]["--memory-spatial"])
        self.assertEqual("on", switches["memory"]["--memory-episodic"])
        # ... and it holds every layer of the arm before it fixed, including the
        # catalogue: the M2 ``memory`` catalogue has no search_target, so running
        # it here would mix the loss of the search skill into the memory delta
        for switch in ("--recovery", "--state-block", "--search-skill",
                       "--context-frames"):
            self.assertEqual(switches["recovery"][switch], switches["memory"][switch],
                             switch)
        self.assertNotIn("--toolset", switches["memory"])
        # the spatial-only arm is the same arm with the episodic store off
        self.assertEqual("on", switches["memory_spatial_only"]["--memory-spatial"])
        self.assertEqual("off", switches["memory_spatial_only"]["--memory-episodic"])
        for switch in ("--recovery", "--state-block", "--search-skill",
                       "--context-frames"):
            self.assertEqual(switches["memory"][switch],
                             switches["memory_spatial_only"][switch], switch)

    def test_the_m2_memory_catalogue_is_still_runnable_as_an_arm_override(self):
        """The goal.md M2 configuration of the memory arm: a per-arm catalogue is
        an arm switch (`--arm-arg`), and it is reported as one."""
        args = _args("--arms", "recovery,memory",
                     "--arm-arg",
                     f"memory=--toolset {DRIVER.M2_MEMORY_TOOLSET} --context-frames 4 "
                     "--search-skill on --state-block on --recovery on "
                     "--memory-spatial on --memory-episodic on",
                     "--seeds", "7", "--initial-yaw-modes", "visible", "--dry-run")
        by_arm = {record["arm"]: record for record in args.arm_support}
        self.assertEqual([], by_arm["memory"]["dropped"])
        switches = _switch_map(by_arm["memory"]["effective"])
        self.assertEqual(DRIVER.M2_MEMORY_TOOLSET, switches["--toolset"])
        self.assertEqual("on", switches["--search-skill"])

    def test_the_pre_memory_arms_switch_the_memory_layer_off(self):
        """The memory layer is a layer: the arms before it must turn it off, or
        the paired ``recovery -> memory`` delta would compare two identical
        configurations (the runner's default is on)."""
        args = _args()
        switches = {record["arm"]: _switch_map(record["effective"])
                    for record in args.arm_support}
        for arm in ("baseline", "vision_context", "recovery"):
            with self.subTest(arm=arm):
                self.assertEqual("off", switches[arm]["--memory-spatial"])
                self.assertEqual("off", switches[arm]["--memory-episodic"])
                # ... and they leave the shared catalogue alone
                self.assertNotIn("--toolset", switches[arm])

    def test_unknown_arm_is_rejected(self):
        with self.assertRaises(SystemExit):
            _args("--arms", "nope")

    def test_vision_context_raises_the_frame_budget_and_recovery_adds_its_switch(self):
        args = _args()
        by_arm = {record["arm"]: record for record in args.arm_support}
        memory_off = ["--memory-spatial", "off", "--memory-episodic", "off"]
        self.assertEqual(by_arm["baseline"]["effective"],
                         ["--recovery", "off", "--state-block", "off",
                          "--search-skill", "off", *memory_off])
        self.assertEqual(by_arm["vision_context"]["effective"],
                         ["--context-frames", "4", "--search-skill", "on",
                          "--recovery", "off", "--state-block", "off", *memory_off])
        self.assertIn("--context-frames", by_arm["recovery"]["effective"])

    def test_every_arm_switch_is_a_switch_the_runner_really_has(self):
        """No degraded arm: each declared switch is in the runner's argparse."""
        args = _args()
        flags = DRIVER.runner_flags_from_source()
        by_arm = {record["arm"]: record for record in args.arm_support}
        for name, record in by_arm.items():
            if DRIVER.ARM_TABLE[name].reserved:
                continue
            with self.subTest(arm=name):
                self.assertEqual([], record["dropped"], record)
                self.assertEqual("full", record["support"])
                for token in record["effective"]:
                    if token.startswith("-"):
                        self.assertIn(token, flags)
        # ... and the runner parses them (a flag may exist but demand a value)
        for name, spec in DRIVER.ARM_TABLE.items():
            if spec.reserved:
                continue
            with self.subTest(arm=name):
                parsed = Runner.parse_args(
                    ["--instruction", "x", "--out", "y", *spec.runner_args]
                )
                declared = _switch_map(spec.runner_args)
                for switch, attribute, default in (
                        ("--recovery", "recovery", "on"),
                        ("--state-block", "state_block", "on"),
                        ("--search-skill", "search_skill", "on"),
                        ("--memory-spatial", "memory_spatial", "on"),
                        ("--memory-episodic", "memory_episodic", "on"),
                        ("--toolset", "toolset", "full")):
                    self.assertEqual(declared.get(switch, default),
                                     getattr(parsed, attribute), switch)

    def test_the_recovery_arm_switches_the_layer_on_after_the_vision_arm_switched_it_off(self):
        args = _args()
        by_arm = {record["arm"]: record for record in args.arm_support}
        switches = {name: _switch_map(record["effective"])
                    for name, record in by_arm.items()}
        memory_off = {"--memory-spatial": "off", "--memory-episodic": "off"}
        # baseline runs the old loop: every new mechanism off
        self.assertEqual(switches["baseline"],
                         {"--recovery": "off", "--state-block": "off",
                          "--search-skill": "off", **memory_off})
        # the vision arm adds the multi-view context and the search skill only
        self.assertEqual(switches["vision_context"],
                         {"--context-frames": "4", "--recovery": "off",
                          "--state-block": "off", "--search-skill": "on", **memory_off})
        # the recovery arm is the vision arm plus the STATE block and recovery
        self.assertEqual(switches["recovery"],
                         {"--context-frames": "4", "--recovery": "on",
                          "--state-block": "on", "--search-skill": "on", **memory_off})

    def test_the_applied_switches_of_the_runner_summaries_reach_the_report(self):
        rows = [
            {"arm": "baseline", "switches": {"applied": {"recovery": False, "state_block": False,
                                                         "search_skill": False, "runtime": "modules"}}},
            {"arm": "baseline", "switches": {"applied": {"recovery": False, "state_block": False,
                                                         "search_skill": False, "runtime": "modules"}}},
            {"arm": "recovery", "switches": {"applied": {"recovery": True, "state_block": True,
                                                         "search_skill": True, "runtime": "modules"}}},
        ]
        report = DRIVER.mechanism_report(rows)
        by_arm = {record["arm"]: record for record in report}
        self.assertEqual(2, by_arm["baseline"]["n_episodes"])
        self.assertEqual({"recovery": False, "state_block": False, "search_skill": False,
                          "spatial_memory": None, "episodic_memory": None},
                         by_arm["baseline"]["applied"])
        self.assertTrue(by_arm["recovery"]["consistent"])
        # a runner that ignored the switch is visible as an inconsistency
        mixed = DRIVER.mechanism_report(
            rows + [{"arm": "recovery", "switches": {"applied": {
                "recovery": False, "state_block": True, "search_skill": True, "runtime": "modules"}}}]
        )
        by_arm = {record["arm"]: record for record in mixed}
        self.assertFalse(by_arm["recovery"]["consistent"])
        self.assertIn("recovery", by_arm["recovery"]["disagreements"])

    def test_the_memory_switches_and_toolset_of_the_arms_reach_the_report(self):
        """What the memory arm really ran: the layer switches the runner
        summaries report plus the catalogue they ran in."""
        def row(asked_spatial, asked_episodic, applied_spatial, applied_episodic):
            return {
                "arm": "memory",
                "toolset": "memory",
                "switches": {
                    "requested": {"recovery": "on", "state_block": "on",
                                  "search_skill": "on",
                                  "spatial_memory": asked_spatial,
                                  "episodic_memory": asked_episodic},
                    "applied": {"recovery": True, "state_block": True,
                                "search_skill": True, "runtime": "modules",
                                "spatial_memory": applied_spatial,
                                "episodic_memory": applied_episodic},
                },
            }

        record = DRIVER.mechanism_report([row("on", "on", True, True)])[0]
        self.assertEqual("memory", record["toolset"])
        self.assertEqual(True, record["applied"]["spatial_memory"])
        self.assertEqual(True, record["applied"]["episodic_memory"])
        self.assertEqual([], record["unattested"])
        self.assertTrue(record["consistent"])

        # a memory that says nothing about its layers: unattested, never "off"
        record = DRIVER.mechanism_report([row("on", "on", None, None)])[0]
        self.assertIsNone(record["applied"]["spatial_memory"])
        self.assertIsNone(record["applied"]["episodic_memory"])
        self.assertEqual(["spatial_memory", "episodic_memory"], record["unattested"])
        # an arm that asked for the layers off has nothing to attest
        record = DRIVER.mechanism_report([row("off", "off", None, None)])[0]
        self.assertEqual([], record["unattested"])
        self.assertEqual({"spatial_memory": "off", "episodic_memory": "off"},
                         {name: record["requested"][name]
                          for name in ("spatial_memory", "episodic_memory")})

    def test_a_per_arm_toolset_is_an_arm_switch_not_a_pairing_violation(self):
        """An arm that runs its own catalogue (the M2 variant of the memory arm)
        is an arm switch the pairing check must report, not a broken cell."""
        args = _args("--arms", "recovery,memory", "--seeds", "7",
                     "--initial-yaw-modes", "visible",
                     "--arm-arg",
                     f"memory=--toolset {DRIVER.M2_MEMORY_TOOLSET} "
                     "--memory-spatial on --memory-episodic on")
        scenarios = [_scenario()]
        episodes = DRIVER.build_episodes(args, scenarios)
        by_name = {scenario.name: scenario for scenario in scenarios}
        commands = {
            (spec.arm, spec.scenario, spec.task.name, spec.seed, spec.yaw_mode):
                DRIVER.build_command(args, by_name[spec.scenario], spec,
                                     DRIVER.episode_dir(Path(args.out), spec))
            for spec in episodes
        }
        check = DRIVER.pairing_check(args, episodes, commands)
        self.assertTrue(check["consistent"], check["issues"])
        self.assertIn("--toolset", check["arm_only_options"])
        self.assertIn("--toolset", check["checked_keys"])

    def test_arm_arg_override_replaces_and_none_clears(self):
        args = _args("--arm-arg", "baseline=--context-frames 3",
                     "--arm-arg", "vision_context=none")
        by_arm = {record["arm"]: record for record in args.arm_support}
        self.assertEqual(by_arm["baseline"]["declared"], ["--context-frames", "3"])
        self.assertEqual(by_arm["baseline"]["effective"], ["--context-frames", "3"])
        self.assertEqual(by_arm["vision_context"]["effective"], [])
        self.assertEqual(by_arm["vision_context"]["declared"], [])
        with self.assertRaises(SystemExit):
            _args("--arm-arg", "nope=--x")

    def test_probe_reads_the_runner_flags_and_drops_unknown_switches(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = Path(tmp) / "runner.py"
            stub.write_text(
                "import argparse\n"
                "parser = argparse.ArgumentParser()\n"
                "parser.add_argument('--context-frames', type=int, default=2)\n"
                "parser.add_argument(\"--scan-views\", type=int, default=8)\n"
            )
            flags = DRIVER.runner_flags_from_source(stub)
            self.assertEqual(flags, {"--context-frames", "--scan-views"})
            kept, dropped = DRIVER.filter_supported_args(
                ["--context-frames", "4", "--recovery", "--scan-views", "8"], flags
            )
            self.assertEqual(kept, ["--context-frames", "4", "--scan-views", "8"])
            self.assertEqual(dropped, ["--recovery"])
            # no probe: everything is passed through
            kept, dropped = DRIVER.filter_supported_args(["--recovery"], None)
            self.assertEqual((kept, dropped), (["--recovery"], []))

    def test_the_real_runner_source_probe_finds_the_documented_switches(self):
        flags = DRIVER.runner_flags_from_source()
        for switch in ("--world", "--ply", "--context-frames", "--scan-views",
                       "--max-turns", "--tool-budget", "--mock-script", "--out"):
            self.assertIn(switch, flags)

    def test_degraded_arm_is_reported_and_strict_mode_refuses(self):
        args = _args("--arm-arg", "recovery=--context-frames 4 --no-such-switch here")
        by_arm = {record["arm"]: record for record in args.arm_support}
        self.assertEqual(by_arm["recovery"]["support"], "degraded")
        self.assertEqual(by_arm["recovery"]["dropped"], ["--no-such-switch"])
        self.assertEqual(by_arm["recovery"]["effective"], ["--context-frames", "4"])
        self.assertIn("override with --arm-arg recovery=", by_arm["recovery"]["reason"])
        with self.assertRaises(SystemExit):
            _args("--strict-arms", "--arm-arg", "recovery=--no-such-switch")

    def test_no_probe_keeps_every_declared_switch(self):
        args = _args("--no-probe")
        by_arm = {record["arm"]: record for record in args.arm_support}
        self.assertIsNone(args.runner_flags_probed)
        self.assertIn("--recovery", by_arm["recovery"]["effective"])


# ---------------------------------------------------------------------------
# scenarios, episode matrix and the pairing check
# ---------------------------------------------------------------------------


class ScenarioTest(unittest.TestCase):
    def test_book_and_synth_presets_carry_their_phase1_parameters(self):
        book = DRIVER.SCENARIOS["book"]
        synth = DRIVER.SCENARIOS["synth"]
        self.assertEqual(book.world, "higs")
        self.assertEqual(book.ply, "/scene_book.ply")
        self.assertTrue(Path(book.task_file).exists(), book.task_file)
        self.assertTrue(Path(synth.ply).exists(), synth.ply)
        self.assertTrue(Path(synth.task_file).exists(), synth.task_file)
        self.assertIn("synth_fair", synth.task_file)
        self.assertEqual(DRIVER.DEFAULT_SCENARIOS, ("book", "synth"))

        args = _args("--scenario", "book", "--scenario", "synth")
        self.assertEqual(args.scenario_list[-2:], ["book", "synth"])
        # the presets carry their own tasks when the CLI does not override them
        args = _args()
        args.scenario_list = ["synth"]
        args.task_json = None
        args.ply = None    # let the preset bring its own scene
        args.world = None  # and its own world
        scenarios = DRIVER.load_scenarios(args)
        self.assertEqual([task.name for task in scenarios[0].tasks],
                         ["synth_near_fair", "synth_far_fair"])
        self.assertEqual(scenarios[0].world, "higs")

    def test_cli_scene_and_task_options_override_a_preset(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene.ply"
            scene.write_bytes(b"ply\n")
            args = _args("--scenario", "book")
            args.ply = str(scene)
            scenario = DRIVER.load_scenarios(args)[0]
        self.assertEqual(scenario.ply, str(scene))
        self.assertEqual([task.name for task in scenario.tasks], ["target0"])
        self.assertIsNone(scenario.task_file)

    def test_hf_backend_and_the_preset_scenarios_build_a_real_command(self):
        scene = Path(DRIVER.SCENARIOS["synth"].ply)
        args = _args("--scenario", "book", "--scenario", "synth",
                     "--backend", "hf", "--model-path", "/models_ro/Qwen3-VL-8B",
                     "--device", "cuda", "--max-vlm-calls", "20", "--dry-run")
        args.scenario_list = ["book", "synth"]
        args.world = None
        args.ply = None
        args.task_json = None
        scenarios = DRIVER.load_scenarios(args)
        self.assertEqual([scenario.name for scenario in scenarios], ["book", "synth"])
        self.assertTrue(scene.exists())
        episodes = DRIVER.build_episodes(args, scenarios)
        commands = self._commands(args, scenarios, episodes)
        book = next(cmd for key, cmd in commands.items()
                    if key[1] == "book" and key[0] == "baseline")
        options = DRIVER.compare._args_map(book)
        self.assertEqual(options["--backend"], "hf")
        self.assertEqual(options["--model-path"], "/models_ro/Qwen3-VL-8B")
        self.assertEqual(options["--device"], "cuda")
        self.assertEqual(options["--ply"], "/scene_book.ply")
        self.assertEqual(options["--max-turns"], "20")
        self.assertEqual(options["--tool-budget"], "20")
        self.assertEqual(options["--target-name"], "book")
        self.assertIn("instruction", " ".join(book))
        synth = next(cmd for key, cmd in commands.items()
                     if key[1] == "synth" and key[0] == "baseline")
        self.assertEqual(DRIVER.compare._args_map(synth)["--ply"], str(scene))

    def test_unknown_scenario_and_missing_scene_are_rejected(self):
        with self.assertRaises(SystemExit):
            _args("--scenario", "nope")
        args = _args("--scenario", "book")
        args.task_json = [json.dumps(TASK_JSON)]
        with self.assertRaises(SystemExit):  # /scene_book.ply is container-only
            DRIVER.load_scenarios(args)
        args.dry_run = True  # a dry run only prints the commands
        self.assertEqual(DRIVER.load_scenarios(args)[-1].name, "book")

    def test_cpu_scenario_needs_a_scene_and_tasks(self):
        args = _args()
        args.task_json = None
        with self.assertRaises(SystemExit):
            DRIVER.load_scenarios(args)

    def test_matrix_is_arms_x_scenarios_x_tasks_x_seeds_x_yaw_modes(self):
        args = _args("--seeds", "7,11", "--initial-yaw-modes", "visible,random")
        scenarios = [_scenario([_task()]), _scenario([_task(name="other")])]
        scenarios[1] = DRIVER.ScenarioSpec(name="synth", world="cpu", ply="/scene.ply",
                                           tasks=[_task(name="synth_far_fair")])
        episodes = DRIVER.build_episodes(args, scenarios)
        self.assertEqual(len(episodes), 3 * 2 * 1 * 2 * 2)
        cells: dict[str, list] = {}
        for spec in episodes:
            cells.setdefault(spec.cell, []).append(spec)
        self.assertEqual(len(cells), 2 * 2 * 2)
        for group in cells.values():
            self.assertEqual([spec.arm for spec in group], args.arm_list)
            self.assertEqual(len({spec.start_yaw_deg for spec in group}), 1)
        visible = next(spec for spec in episodes if spec.yaw_mode == "visible" and spec.seed == 7)
        random_ = next(spec for spec in episodes if spec.yaw_mode == "random" and spec.seed == 7)
        bearing = DRIVER.compare.target_bearing_deg(TARGET, TASK_JSON["start_pos"])
        self.assertLessEqual(abs(DRIVER._wrap_deg(visible.start_yaw_deg - bearing)), 40.0)
        self.assertGreater(abs(DRIVER._wrap_deg(random_.start_yaw_deg - bearing)), 40.0)

    def _commands(self, args, scenarios, episodes):
        by_name = {scenario.name: scenario for scenario in scenarios}
        return {
            (spec.arm, spec.scenario, spec.task.name, spec.seed, spec.yaw_mode):
                DRIVER.build_command(args, by_name[spec.scenario], spec,
                                     DRIVER.episode_dir(Path(args.out), spec))
            for spec in episodes
        }

    def test_commands_share_scene_start_budget_and_only_differ_by_arm(self):
        args = _args("--max-vlm-calls", "12", "--seeds", "7",
                     "--initial-yaw-modes", "visible")
        scenarios = [_scenario()]
        episodes = DRIVER.build_episodes(args, scenarios)
        commands = self._commands(args, scenarios, episodes)
        check = DRIVER.pairing_check(args, episodes, commands)
        self.assertEqual(check["cells"], 1)
        self.assertTrue(check["consistent"], check["issues"])
        self.assertIn("--context-frames", check["arm_only_options"])
        self.assertIn("--max-turns", check["checked_keys"])

        maps = {spec.arm: DRIVER.compare._args_map(
            commands[(spec.arm, spec.scenario, spec.task.name, spec.seed, spec.yaw_mode)])
            for spec in episodes}
        for key in ("--world", "--ply", "--target-pos", "--start-pos", "--start-yaw-deg",
                    "--seed", "--initial-yaw-mode", "--success-dist", "--max-turns",
                    "--tool-budget", "--toolset", "--instruction"):
            self.assertEqual(maps["baseline"][key], maps["recovery"][key], key)
        self.assertEqual(maps["baseline"]["--max-turns"], "12")
        self.assertEqual(maps["baseline"]["--tool-budget"], "12")
        self.assertEqual(maps["baseline"]["--toolset"], "full")
        self.assertEqual(maps["vision_context"]["--context-frames"], "4")
        self.assertEqual(maps["recovery"]["--context-frames"], "4")
        self.assertNotIn("--max-steps", maps["baseline"])

    def test_pairing_check_flags_a_divergent_budget(self):
        args = _args("--seeds", "7", "--initial-yaw-modes", "visible")
        scenarios = [_scenario()]
        episodes = DRIVER.build_episodes(args, scenarios)
        commands = self._commands(args, scenarios, episodes)
        key = ("recovery", "cpu", "target0", 7, "visible")
        index = commands[key].index("--max-turns")
        commands[key][index + 1] = "99"
        check = DRIVER.pairing_check(args, episodes, commands)
        self.assertFalse(check["consistent"])
        self.assertEqual(check["issues"][0]["option"], "--max-turns")

    def test_arm_switches_win_over_the_shared_context_flag(self):
        args = _args("--context-frames", "2")
        scenario = _scenario()
        spec = _spec("vision_context")
        cmd = DRIVER.build_command(args, scenario, spec, Path("/tmp/x"))
        options = DRIVER.compare._args_map(cmd)
        # the arm's switch is appended last: argparse keeps the last value
        positions = [index for index, token in enumerate(cmd)
                     if token == "--context-frames"]
        self.assertEqual(len(positions), 2)
        self.assertEqual(cmd[positions[0] + 1], "2")   # the shared value
        self.assertEqual(cmd[positions[1] + 1], "4")   # the arm's value wins
        self.assertEqual(options["--context-frames"], "4")
        self.assertEqual(options["--out"], "/tmp/x")


# ---------------------------------------------------------------------------
# metrics: success / false confirmation / repeats / coverage / cost
# ---------------------------------------------------------------------------


class FakeArtifactsMixin:
    def _row(self, tmp: str, trace: list[dict], summary: dict | None = None,
             task=None, arm: str = "vision_context") -> dict:
        return _row_from_trace(tmp, trace, summary, task, arm)


class SuccessMetricTest(FakeArtifactsMixin, unittest.TestCase):
    def test_solved_needs_the_claim_and_the_geometric_reach(self):
        trace = [
            _entry(1, "approach_candidate",
                   {"candidate_id": "cand_1", "outcome": "moved",
                    "pose": [1.0, 0.4, 1.0]},
                   args={"candidate_id": "cand_1"}),
            _entry(2, "finish", {"reason": "done", "success": True},
                   args={"reason": "done", "success": True}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, trace)
        self.assertTrue(row["claim"])
        self.assertTrue(row["reached"])          # 1.0 m < 1.0 gate + 0.5 radius
        self.assertTrue(row["solved"])
        self.assertFalse(row["false_confirmation"])

    def test_a_claim_without_the_reach_is_a_false_confirmation(self):
        trace = [
            _entry(1, "observe", {"observation_id": "obs_1", "pose": [0.0, 0.0, 1.0]}),
            _entry(2, "finish", {"reason": "target verified", "success": True},
                   args={"reason": "done", "success": True}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, trace, _summary(success=False, finish_reason="finish"))
        self.assertTrue(row["claim"])
        self.assertFalse(row["reached"])
        self.assertTrue(row["false_confirmation"])
        self.assertEqual(row["false_confirmation_reason"], "unreached")

    def test_verify_goal_verified_counts_as_a_claim_and_explicit_failure_does_not(self):
        trace = [
            _entry(1, "verify_goal", {"verified": True, "distance_m": 0.4,
                                      "pose": [0.0, 0.0, 1.0]}),
            _entry(2, "finish", {"reason": "gave up"}, args={"reason": "gave up",
                                                             "success": False}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, trace, _summary(success=False))
        self.assertTrue(row["claim"])
        self.assertEqual(row["claim_kinds"], ["verify_goal"])  # finish said success=false
        self.assertEqual(row["verify_goal_verified"], 1)
        self.assertTrue(row["false_confirmation"])

    def test_an_unmet_verification_is_not_a_claim(self):
        trace = [
            _entry(1, "verify_goal", {"verified": False, "reason": "unmet_conditions",
                                      "pose": [0.0, 0.0, 1.0]}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, trace, _summary(success=False, finish_reason="max_turns"))
        self.assertFalse(row["claim"])
        self.assertEqual(row["verify_goal_unmet"], 1)
        self.assertFalse(row["false_confirmation"])

    def test_a_bare_finish_is_an_honest_give_up_not_a_claim(self):
        """A finish that only carries a reason declares nothing.

        The last real run counted 7 honest "not found" finishes as claims; a
        claim is an explicit declaration (``success: true``) or a verified goal.
        """
        trace = [
            _entry(1, "observe", {"observation_id": "obs_1", "pose": [0.0, 0.0, 1.0]}),
            _entry(2, "finish", {"reason": "could not find the target"},
                   args={"reason": "could not find the target"}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, trace, _summary(success=False, finish_reason="finish"))
        self.assertFalse(row["claim"])
        self.assertEqual([], row["claim_kinds"])
        self.assertEqual(0, row["n_claims"])
        self.assertIsNone(row["finish_declared"])
        self.assertTrue(row["honest_finish"])
        self.assertFalse(row["false_confirmation"])
        self.assertEqual("", row["false_confirmation_reason"])
        self.assertFalse(row["solved"])
        # the annotation says why the finish is not a claim
        events = [event for event in row["annotation"]["events"]
                  if event["kind"] == "finish_no_claim"]
        self.assertEqual(1, len(events))
        self.assertIsNone(events[0]["declared"])
        narrative = " ".join(row["annotation"]["narrative"])
        self.assertIn("honest finish", narrative)
        self.assertNotIn("FALSE CONFIRMATION", narrative)

    def test_a_finish_that_declares_success_true_is_a_claim(self):
        trace = [
            _entry(1, "approach_candidate",
                   {"candidate_id": "cand_1", "outcome": "moved", "pose": [1.0, 0.4, 1.0]},
                   args={"candidate_id": "cand_1"}),
            _entry(2, "finish", {"reason": "arrived", "success": True},
                   args={"reason": "arrived", "success": True}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, trace)
        self.assertTrue(row["claim"])
        self.assertEqual(["finish"], row["claim_kinds"])
        self.assertTrue(row["finish_declared"])
        self.assertFalse(row["honest_finish"])
        self.assertTrue(row["solved"])
        self.assertFalse(row["false_confirmation"])

    def test_a_finish_that_declares_success_false_is_an_honest_give_up(self):
        trace = [
            _entry(1, "observe", {"observation_id": "obs_1", "pose": [0.0, 0.0, 1.0]}),
            _entry(2, "finish", {"reason": "gave up", "success": False},
                   args={"reason": "gave up", "success": False}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, trace, _summary(success=False, finish_reason="finish"))
        self.assertFalse(row["claim"])
        self.assertIs(False, row["finish_declared"])
        self.assertTrue(row["honest_finish"])
        self.assertFalse(row["false_confirmation"])
        events = [event for event in row["annotation"]["events"]
                  if event["kind"] == "finish_no_claim"]
        self.assertIs(False, events[0]["declared"])
        self.assertIn("declared failure",
                      " ".join(row["annotation"]["narrative"]))

    def test_the_summary_finish_args_carry_the_declaration_without_a_trace_entry(self):
        """A missing/older trace: only an explicit declaration is a claim."""
        trace = [_entry(1, "observe", {"observation_id": "obs_1", "pose": [0.0, 0.0, 1.0]})]
        with tempfile.TemporaryDirectory() as tmp:
            claimed = self._row(tmp, trace, _summary(
                success=True, finish_reason="finish",
                finish_args={"reason": "arrived", "success": True}))
            honest = self._row(tmp, trace, _summary(
                success=False, finish_reason="finish",
                finish_args={"reason": "could not find it"}))
        self.assertTrue(claimed["claim"])
        self.assertEqual(["finish"], claimed["claim_kinds"])
        self.assertTrue(claimed["finish_declared"])
        self.assertFalse(honest["claim"])
        self.assertTrue(honest["honest_finish"])
        self.assertFalse(honest["false_confirmation"])

    def test_a_grounded_candidate_outside_the_gate_is_a_false_confirmation(self):
        trace = [
            _entry(1, "ground_target",
                   {"candidate_id": "cand_1", "position_flu": [6.0, 0.0, 1.0],
                    "pose": [6.0, 0.0, 1.0]}),
            _entry(2, "approach_candidate", {"candidate_id": "cand_1", "outcome": "moved",
                                             "pose": [5.0, 0.0, 1.0]}),
            _entry(3, "finish", {"reason": "done", "success": True},
                   args={"reason": "done", "success": True}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, trace, _summary(success=False))
        self.assertTrue(row["candidate_wrong"])
        self.assertTrue(row["false_confirmation"])


class VerifyGateAuditTest(FakeArtifactsMixin, unittest.TestCase):
    """The report records the gate ``verify_goal`` used next to the eval gate.

    The v3 decision-loop run lost four of its claims to a *threshold* gap, not
    to a model error: ``verify_goal`` checked its own ``max(1.5, radius + 0.5)``
    default while the driver evaluated ``--success-dist + radius`` (1.2 m on
    book_near). ``verify_limit_m`` / ``verify_gate_consistent`` make that visible
    per episode instead of leaving it to a trace post-mortem.
    """

    def test_a_verify_limit_equal_to_the_gate_is_consistent(self):
        trace = [
            _entry(1, "verify_goal", {"verified": True, "distance_m": 1.4,
                                      "checks": {"distance_to_target_m": 1.4,
                                                 "distance_limit_m": 1.5},
                                      "pose": [0.6, 0.5, 1.0]}),
            _entry(2, "finish", {"reason": "verified", "success": True},
                   args={"reason": "verified", "success": True}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, trace, _summary(success=True, done_dist_m=1.5))
        self.assertEqual(1.5, row["gate_m"])          # 1.0 success-dist + 0.5 radius
        self.assertEqual(1.5, row["verify_limit_m"])
        self.assertTrue(row["verify_gate_consistent"])

    def test_the_v3_book_near_claim_is_reported_as_a_gate_mismatch(self):
        """1.32 m at the old 1.5 m default with a 1.2 m gate: a false claim."""
        task = _task(success_dist=0.9, target_radius=0.3)
        trace = [
            _entry(1, "verify_goal", {"verified": True,
                                      "checks": {"distance_to_target_m": 1.32,
                                                 "distance_limit_m": 1.5},
                                      "pose": [0.68, 0.6, 1.0]}),
            _entry(2, "finish", {"reason": "verified", "success": True},
                   args={"reason": "verified", "success": True}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, trace, _summary(success=False, finish_reason="finish",
                                                 final_distance_m=1.315, done_dist_m=1.5),
                            task=task)
        self.assertEqual(1.2, row["gate_m"])
        self.assertEqual(1.5, row["verify_limit_m"])
        self.assertFalse(row["verify_gate_consistent"])
        self.assertTrue(row["false_confirmation"])

    def test_the_unified_episode_is_consistent_and_claims_nothing_past_the_gate(self):
        """After the fix the same standoff comes back *unverified* and honest."""
        task = _task(success_dist=0.9, target_radius=0.3)
        trace = [
            _entry(1, "verify_goal", {"verified": False, "reason": "unmet_conditions",
                                      "checks": {"distance_to_target_m": 1.32,
                                                 "distance_limit_m": 1.2},
                                      "unmet": [{"check": "distance_to_target_m",
                                                 "value_m": 1.32, "max_m": 1.2}],
                                      "pose": [0.68, 0.6, 1.0]}),
            _entry(2, "finish", {"reason": "gave up"},
                   args={"reason": "gave up", "success": False}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, trace, _summary(success=False, finish_reason="finish",
                                                 final_distance_m=1.32, done_dist_m=1.2),
                            task=task)
        self.assertEqual(1.2, row["verify_limit_m"])
        self.assertTrue(row["verify_gate_consistent"])
        self.assertFalse(row["claim"])
        self.assertFalse(row["false_confirmation"])
        self.assertTrue(row["honest_finish"])

    def test_a_legacy_summary_without_the_gate_stays_unknown(self):
        trace = [_entry(1, "observe", {"observation_id": "obs_1", "pose": [0.0, 0.0, 1.0]})]
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, trace, _summary(success=False, finish_reason="max_turns"))
        self.assertIsNone(row["verify_limit_m"])
        self.assertIsNone(row["verify_gate_consistent"])

    def test_a_legacy_claim_past_the_gate_is_flagged_without_the_recorded_limit(self):
        """The v3 artifacts predate ``distance_limit_m``: the claim itself proves it."""
        task = _task(success_dist=0.9, target_radius=0.3)
        trace = [
            _entry(1, "verify_goal", {"verified": True,
                                      "checks": {"distance_to_target_m": 1.32},
                                      "pose": [0.68, 0.6, 1.0]}),
            _entry(2, "finish", {"reason": "verified", "success": True},
                   args={"reason": "verified", "success": True}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, trace, _summary(success=False, finish_reason="finish"),
                            task=task)
        self.assertIsNone(row["verify_limit_m"])
        self.assertIs(row["verify_gate_consistent"], False)
        self.assertEqual(1, row["verify_claims_past_gate"])
        self.assertTrue(row["false_confirmation"])

    def test_the_report_marks_the_gate_identity(self):
        gate = DRIVER.verify_gate_summary([
            {"arm": "memory", "scenario": "book", "task": "book_near", "seed": 11,
             "yaw_mode": "visible", "error": None, "skipped": False,
             "verify_gate_consistent": False, "verify_claims_past_gate": 1},
            {"arm": "baseline", "scenario": "book", "task": "book_near", "seed": 11,
             "yaw_mode": "visible", "error": None, "skipped": False,
             "verify_gate_consistent": True, "verify_claims_past_gate": 0},
        ])
        self.assertEqual({"checked": 2, "consistent": 1, "mismatch": 1,
                          "unreported": 0, "claims_past_gate": 1},
                         {key: gate[key] for key in ("checked", "consistent", "mismatch",
                                                     "unreported", "claims_past_gate")})
        self.assertIn("memory|book|book_near|visible|11", gate["mismatch_cells"][0])
        line = DRIVER.verify_gate_line(gate)
        self.assertIn("1/2", line)
        self.assertIn("threshold artifact", line)
        clean = DRIVER.verify_gate_line({"checked": 2, "consistent": 2, "mismatch": 0,
                                        "unreported": 0, "claims_past_gate": 0,
                                        "mismatch_cells": []})
        self.assertIn("2/2", clean)
        self.assertNotIn("threshold artifact", clean)

    def test_the_aggregate_counts_the_gate_mismatches(self):
        rows = [
            {"arm": "memory", "error": None, "skipped": False, "claim": True,
             "false_confirmation": True, "verify_limit_m": 1.5,
             "verify_gate_consistent": False},
            {"arm": "memory", "error": None, "skipped": False, "claim": False,
             "false_confirmation": False, "verify_limit_m": 1.2,
             "verify_gate_consistent": True},
            {"arm": "baseline", "error": None, "skipped": False, "claim": False,
             "false_confirmation": False, "verify_limit_m": None,
             "verify_gate_consistent": None},
        ]
        stats = DRIVER.aggregate(rows)
        self.assertEqual(stats["n_verify_gate_checked"], 2)
        self.assertEqual(stats["n_verify_gate_mismatch"], 1)
        self.assertEqual(stats["verify_gate_consistent_rate"], 0.5)


class PromptIdentityCheckTest(unittest.TestCase):
    """The arms of one cell must show the model the same episode identity.

    ``episode_id`` is part of the prompt's first line ("Episode ep_... | turn
    ..."), and the E2 arms used to spell their mode into it - a constant,
    prompt-visible difference between arms that has nothing to do with the
    mechanism under test. The check reads it out of the runner summaries.
    """

    def _rows(self, *ids, seed: int = 7) -> list[dict]:
        rows = []
        for arm, prompt_id in zip(("baseline", "memory"), ids):
            rows.append({"arm": arm, "scenario": "cpu", "task": "target0", "seed": seed,
                         "yaw_mode": "visible", "error": None, "skipped": False,
                         "prompt_episode_id": prompt_id})
        return rows

    def test_neutral_ids_across_the_arms_of_a_cell_are_consistent(self):
        check = DRIVER.prompt_identity_check(self._rows("ep_007", "ep_007"),
                                             ["baseline", "memory"])
        self.assertTrue(check["consistent"])
        self.assertEqual([], check["issues"])
        self.assertEqual(1, check["cells"])
        self.assertEqual(2, check["episodes_checked"])

    def test_an_arm_name_in_the_prompt_id_is_flagged(self):
        check = DRIVER.prompt_identity_check(self._rows("ep_007", "m2_memory_000_00"),
                                             ["baseline", "memory"])
        self.assertFalse(check["consistent"])
        issue = check["issues"][0]
        self.assertEqual("memory", issue["arm"])
        self.assertIn("m2_memory_000_00", str(issue["prompt_episode_id"]))
        self.assertEqual("arm_name_in_prompt_id", issue["reason"])

    def test_divergent_prompt_ids_of_one_cell_are_flagged(self):
        check = DRIVER.prompt_identity_check(self._rows("ep_007", "ep_011"),
                                             ["baseline", "memory"])
        self.assertFalse(check["consistent"])
        self.assertEqual("prompt_id_differs_across_arms", check["issues"][0]["reason"])

    def test_missing_prompt_ids_are_reported_as_unknown_not_as_neutral(self):
        rows = self._rows("ep_007", None)
        check = DRIVER.prompt_identity_check(rows, ["baseline", "memory"])
        self.assertFalse(check["consistent"])
        self.assertEqual("prompt_id_missing", check["issues"][0]["reason"])
        self.assertEqual("memory", check["issues"][0]["arm"])


class RepeatSearchTest(FakeArtifactsMixin, unittest.TestCase):
    def test_repeat_checks_count_same_region_and_same_evidence(self):
        trace = [
            _entry(1, "observe", {"observation_id": "obs_1", "pose": [0.0, 0.0, 1.0]},
                   args={"mode": "lookaround"}, new_evidence=True),
            _entry(2, "observe", {"observation_id": "obs_2", "pose": [0.0, 0.0, 1.0]},
                   args={"mode": "lookaround"}, new_evidence=False),
            _entry(3, "ground_target", {"reason": "no_candidate", "pose": [0.0, 0.0, 1.0]},
                   args={"observation_id": "obs_2"}, status=STATUS_EMPTY,
                   new_evidence=False),
            _entry(4, "query_memory", {"hits": [], "pose": [0.0, 0.0, 1.0]},
                   args={"query": "target"}, status=STATUS_CACHED, new_evidence=False),
            _entry(5, "search_target", {"candidate_id": "cand_1", "pose": [3.0, 0.0, 1.0]},
                   args={"reason": "fresh sector"}, new_evidence=True),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, trace)
        # the second lookaround repeats the scan of cell (0,0)
        self.assertEqual(row["repeat_scans"], 1)
        self.assertEqual(row["repeat_region_calls"], 1)
        # the empty grounding and the cached memory query carry no new evidence
        self.assertEqual(row["repeat_evidence_calls"], 3)
        self.assertEqual(row["info_calls"], 5)
        # turn 2 is both a region repeat and an evidence repeat, so the union is 3
        self.assertEqual(row["repeat_search_checks"], 3)
        self.assertAlmostEqual(row["repeat_search_ratio"], 0.6)

    def test_region_and_evidence_repeats_are_unioned_over_the_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, RICH_TRACE)
        # the empty grounding repeats evidence already seen
        self.assertEqual(row["repeat_evidence_calls"], 1)
        # the search call scans from the cell the lookaround already scanned
        self.assertEqual(row["repeat_region_calls"], 1)
        # turns 2 and 3, over the three information calls
        self.assertEqual(row["repeat_search_checks"], 2)
        self.assertEqual(row["repeat_search_ratio"], 0.667)


class CoverageTest(FakeArtifactsMixin, unittest.TestCase):
    def test_reported_contract_fields_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, RICH_TRACE)
        self.assertEqual(row["coverage_source"], "reported")
        self.assertEqual(row["coverage_calls_reported"], 2)
        self.assertEqual(row["coverage_checked_deg"], 172.0)      # max over the calls
        self.assertEqual(row["coverage_unchecked_deg"], 188.0)    # last reported
        self.assertEqual(row["coverage_worth_looking"], 1)
        self.assertEqual(row["view_index_max"], 3)
        self.assertEqual(row["views_expanded"], 1)

    def test_sector_list_coverage_matches_the_observation_contract(self):
        """The lookaround reports sectors, not degrees (the real contract)."""
        trace = [
            _entry(1, "observe",
                   {"observation_id": "obs_1", "pose": [0.0, 0.0, 1.0],
                    "view_index": [{"obs_id": "obs_1", "yaw_deg": 17.0, "direction": "ahead"},
                                   {"obs_id": "obs_2", "yaw_deg": 107.0, "direction": "left"}],
                    "coverage": {
                        "bin_deg": 45.0,
                        "directions_known": True,
                        "checked_deg": [[0.0, 45.0], [90.0, 135.0]],
                        "unchecked_deg": [[45.0, 90.0], [135.0, 180.0]],
                        "worth_looking": [{"obs_id": "obs_2", "yaw_deg": 107.0},
                                          {"obs_id": "obs_1", "yaw_deg": 17.0}],
                        "all_directions_covered": False,
                        "down_checked": True,
                    }},
                   args={"mode": "lookaround"}),
            _entry(2, "search_target",
                   {"candidate_id": "cand_1", "observation_id": "obs_3",
                    "coverage": {"bin_deg": 45.0, "directions_known": True,
                                 "checked_deg": [[-180.0 + 45.0 * i, -135.0 + 45.0 * i]
                                                 for i in range(8)],
                                 "unchecked_deg": [], "worth_looking": [],
                                 "all_directions_covered": True, "down_checked": False},
                    "vlm_calls": 3, "steps": [{"step": 1, "why": "sweep",
                                               "obs_ids": ["obs_3"],
                                               "result": "candidate cand_1"}]},
                   args={"instruction": "book"}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, trace)
        self.assertEqual(row["coverage_source"], "reported")
        # the widest capture covers all 8 sectors; the first one two
        self.assertEqual(row["coverage_checked_deg"], 360.0)
        self.assertEqual(row["coverage_checked_sectors"], 8)
        self.assertEqual(row["coverage_total_sectors"], 8)
        self.assertEqual(row["coverage_unchecked_sectors"], 0)
        self.assertEqual(row["direction_coverage"], 1.0)
        self.assertEqual(row["coverage_worth_looking"], 2)
        self.assertEqual(row["coverage_full_sweeps"], 1)
        self.assertEqual(row["coverage_down_checked"], 1)
        self.assertEqual(row["views_expanded"], 2)
        self.assertEqual(row["direction_bins"],
                         ["-180..-135", "-135..-90", "-90..-45", "-45..0", "0..45",
                          "45..90", "90..135", "135..180"])

    def test_heading_coverage_is_estimated_without_the_contract(self):
        trace = [
            _entry(1, "observe", {"observation_id": "obs_1", "pose": [0.0, 0.0, 1.0],
                                  "headings_deg": [0.0, 90.0, 180.0, -90.0]},
                   args={"mode": "lookaround"}),
            _entry(2, "observe", {"observation_id": "obs_2", "pose": [0.0, 0.0, 1.0],
                                  "heading_deg": 45.0},
                   args={"mode": "forward"}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, trace)
        self.assertEqual(row["coverage_source"], "estimated")
        self.assertIsNone(row["coverage_checked_deg"])
        self.assertEqual(len(row["direction_bins"]), 5)
        self.assertAlmostEqual(row["direction_coverage"], 0.625)
        self.assertGreater(row["coverage_checked_deg_estimate"], 90.0)

    def test_crop_checks_are_counted(self):
        trace = [
            _entry(1, "inspect_region", {"crop_image_ref": "obs_11", "zoom": 2.0,
                                         "pose": [0.0, 0.0, 1.0]},
                   args={"observation_id": "obs_2", "zoom": 2.0}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, trace)
        self.assertEqual(row["crop_checks"], 1)


class CostTest(FakeArtifactsMixin, unittest.TestCase):
    def test_total_cost_includes_the_search_skill_internal_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, RICH_TRACE, _summary(turns=7, tool_calls=6, vlm_calls=7))
        self.assertEqual(row["vlm_calls_decision"], 7)
        self.assertEqual(row["vlm_calls_declared"], 4)  # search_target's vlm_calls
        self.assertEqual(row["vlm_calls_total"], 11)
        self.assertEqual(row["vlm_calls_internal"], 4)
        self.assertGreater(row["tokens_estimate"], 0)
        self.assertEqual(row["wall_time_s"], 12.0)

    def test_the_routines_inner_and_outer_calls_are_reported_per_episode(self):
        """A search is one outer call and several inner ones - both are reported.

        The episode paid one budget unit for the search (its inner calls run on
        the routine's own allowance), and the driver keeps the split so a run can
        say how much sub-routine work a decision really triggered.
        """
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, RICH_TRACE, _summary(turns=7, tool_calls=6, vlm_calls=7))
        self.assertEqual(row["declared_inner_tool_calls"], 6)
        self.assertEqual(row["declared_outer_tool_calls"], 1)
        self.assertEqual(row["declared_routine_vlm_calls"], 4)
        self.assertEqual(row["search_target_calls"], 1)

    def test_a_legacy_cost_block_falls_back_to_its_tool_calls(self):
        """Traces from before the split published ``cost["tool_calls"]`` = inner."""
        trace = [
            _entry(1, "search_target",
                   {"candidate_id": "cand_1", "vlm_calls": 3,
                    "cost": {"tool_calls": 5, "vlm_calls": 3, "turns": 1, "budget_turns": 2}},
                   args={"instruction": "the red book"}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, trace, _summary(turns=1, tool_calls=1, vlm_calls=1))
        self.assertEqual(row["declared_inner_tool_calls"], 5)
        self.assertEqual(row["declared_outer_tool_calls"], 0)  # not declared back then
        self.assertEqual(row["declared_routine_vlm_calls"], 3)
        self.assertEqual(row["search_target_calls"], 1)

    def test_an_episode_without_a_routine_reports_zero(self):
        trace = [_entry(1, "observe", {"observation_id": "obs_1", "pose": [0.0, 0.0, 1.0]})]
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, trace, _summary(turns=1, tool_calls=1, vlm_calls=1))
        self.assertEqual(row["declared_inner_tool_calls"], 0)
        self.assertEqual(row["declared_outer_tool_calls"], 0)
        self.assertEqual(row["search_target_calls"], 0)

    def test_tokens_follow_the_shared_model(self):
        args = _args("--chars-per-token", "4", "--prompt-chars-per-call", "0",
                     "--image-size", "28", "28", "--image-token-divisor", "28")
        row = {"images_total": 3, "text_chars": 40}
        DRIVER._token_model(args, row, decision=2)
        self.assertEqual(row["image_tokens"], 28)
        self.assertEqual(row["tokens_estimate"], 3 * 28 + 10)


class MovementDiagnosticTest(FakeArtifactsMixin, unittest.TestCase):
    def test_toward_target_is_a_diagnostic_not_a_gate(self):
        trace = [
            _entry(1, "observe", {"observation_id": "obs_1", "pose": [0.0, 0.0, 1.0]}),
            # a search leg that leaves the target bearing (1 m away from it)
            _entry(2, "go_to_viewpoint", {"viewpoint_id": "vp_1", "outcome": "moved",
                                          "segment_progress_m": 1.0,
                                          "pose": [-1.0, 0.0, 1.0]},
                   args={"viewpoint_id": "vp_1"}),
            _entry(3, "approach_candidate", {"candidate_id": "cand_1", "outcome": "moved",
                                             "segment_progress_m": 3.0,
                                             "pose": [2.0, 0.5, 1.0]},
                   args={"candidate_id": "cand_1"}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(tmp, trace)
        self.assertEqual(row["nav_attempts"], 2)
        self.assertEqual(row["moved_legs"], 2)
        self.assertEqual(row["toward_target_legs"], 1)
        self.assertAlmostEqual(row["toward_target_rate"], 0.5)
        self.assertAlmostEqual(row["progress_m"], 4.041)


class ProgressKindTest(unittest.TestCase):
    def test_reported_kind_is_kept_and_missing_kind_is_derived(self):
        reported = _entry(1, "observe", {"observation_id": "obs_1"}, progress_kind="action")
        step = DRIVER.step_annotation(reported, 0)
        self.assertEqual(step["progress_kind"], "action")
        self.assertEqual(step["progress_kind_source"], "reported")

        derived = _entry(1, "ground_target", {"candidate_id": "cand_1"}, new_evidence=True)
        step = DRIVER.step_annotation(derived, 0)
        self.assertEqual(step["progress_kind"], "information")
        self.assertEqual(step["progress_kind_source"], "derived")

        nav = DRIVER.step_annotation(_entry(2, "go_to_viewpoint", {}, new_evidence=False), 0)
        self.assertEqual(nav["progress_kind"], "action")

        none = DRIVER.step_annotation(
            _entry(3, "observe", {}, status=STATUS_CACHED, new_evidence=False), 0)
        self.assertEqual(none["progress_kind"], "none")

        # a progress kind outside the contract is not trusted
        bogus = DRIVER.step_annotation(
            _entry(4, "observe", {}, progress_kind="maybe", new_evidence=True), 0)
        self.assertEqual(bogus["progress_kind"], "information")
        self.assertEqual(bogus["progress_kind_source"], "derived")


# ---------------------------------------------------------------------------
# decision annotation: the causal chain
# ---------------------------------------------------------------------------


class AnnotationTest(unittest.TestCase):
    def _annotation(self, trace: list[dict], summary: dict | None = None, arm="recovery"):
        with tempfile.TemporaryDirectory() as tmp:
            row = _row_from_trace(tmp, trace, summary, arm=arm)
        return row, row["annotation"]

    def test_chain_proves_reason_new_information_and_the_changed_action(self):
        trace = [
            _entry(1, "observe", {"observation_id": "obs_1", "pose": [0.0, 0.0, 1.0],
                                  "coverage": {"checked_deg": 86.0, "unchecked_deg": 274.0,
                                               "worth_looking": True}},
                   args={"mode": "lookaround", "reason": "target not visible"},
                   new_evidence=True),
            _entry(2, "ground_target", {"reason": "no_candidate"},
                   args={"observation_id": "obs_1", "reason": "locate"}, status=STATUS_EMPTY,
                   new_evidence=False),
            _entry(3, "ground_target", {"reason": "no_candidate"},
                   args={"observation_id": "obs_1", "reason": "locate again"},
                   status=STATUS_EMPTY, new_evidence=False),
            _entry(4, "go_to_viewpoint",
                   {"viewpoint_id": "vp_3", "outcome": "moved", "segment_progress_m": 1.2,
                    "pose": [1.2, 0.0, 1.0]},
                   args={"viewpoint_id": "vp_3", "reason": "a different observation point"},
                   new_evidence=True),
            _entry(5, "observe", {"observation_id": "obs_4", "pose": [1.2, 0.0, 1.0],
                                  "view_index": 2,
                                  "coverage": {"checked_deg": 258.0, "unchecked_deg": 102.0,
                                               "worth_looking": False}},
                   args={"mode": "forward", "reason": "re-observe from the new viewpoint"},
                   new_evidence=True),
            _entry(6, "ground_target", {"candidate_id": "cand_2", "observation_id": "obs_4"},
                   args={"observation_id": "obs_4", "reason": "locate the book"},
                   new_evidence=True),
            _entry(7, "approach_candidate", {"candidate_id": "cand_2", "outcome": "moved",
                                             "segment_progress_m": 1.4,
                                             "pose": [1.9, 0.4, 1.0]},
                   args={"candidate_id": "cand_2", "reason": "close in"}, new_evidence=True),
            _entry(8, "verify_goal", {"verified": True, "distance_m": 0.4,
                                      "pose": [1.9, 0.4, 1.0]}, new_evidence=True),
            _entry(9, "finish", {"reason": "done"}, args={"reason": "done"}),
        ]
        row, annotation = self._annotation(trace, _summary(turns=9, tool_calls=8, vlm_calls=9))
        kinds = [event["kind"] for event in annotation["events"]]
        self.assertIn("no_candidate", kinds)
        self.assertIn("strategy_change", kinds)
        self.assertIn("viewpoint_change", kinds)
        self.assertIn("candidate_found", kinds)
        self.assertIn("claim", kinds)

        change = next(event for event in annotation["events"]
                      if event["kind"] == "strategy_change")
        self.assertEqual(change["turn"], 4)
        self.assertEqual(change["after"], "ground_target")
        self.assertEqual(change["streak"], [2, 3])
        self.assertEqual(change["action"], "go_to_viewpoint")
        self.assertEqual(change["marker"], "no_candidate")
        self.assertTrue(change["consumed_by"] is None or "vp_3" in json.dumps(change))

        viewpoint = next(event for event in annotation["events"]
                         if event["kind"] == "viewpoint_change")
        self.assertEqual(viewpoint["referenced"], ["vp_3"])
        self.assertTrue(viewpoint["new_info_before"])

        # the new observation of the new viewpoint was consumed by the grounding
        used = {(item["id"], item["used_by"]) for item in annotation["used_by"]}
        self.assertIn(("obs_4", "ground_target"), used)
        self.assertIn(("cand_2", "approach_candidate"), used)

        narrative = "\n".join(annotation["narrative"])
        self.assertIn("STRATEGY CHANGE after 2 unproductive ground_target", narrative)
        self.assertIn("new information: obs_4", narrative)
        self.assertIn("coverage checked=258deg", narrative)
        self.assertIn("candidate cand_2 found", narrative)
        self.assertIn("chain:", narrative)
        self.assertEqual(annotation["counters"]["strategy_changes"], 1)
        self.assertEqual(annotation["counters"]["viewpoint_changes"], 1)
        self.assertEqual(annotation["counters"]["candidates_found"], 1)
        self.assertTrue(row["solved"])

    def test_search_skill_steps_and_internal_calls_are_expanded(self):
        trace = [
            _entry(1, "search_target",
                   {"candidate_id": "cand_9", "observation_id": "obs_7", "vlm_calls": 3,
                    "steps": [
                        {"step": 1, "why": "grounding failed", "obs_ids": ["obs_1"],
                         "result": "no_candidate"},
                        {"step": 2, "why": "rotate to the unchecked sector",
                         "obs_ids": ["obs_7"], "result": "candidate cand_9"},
                    ]},
                   args={"reason": "search the sector"}, new_evidence=True),
        ]
        _, annotation = self._annotation(trace)
        with_steps = [event for event in annotation["events"] if event.get("search_steps")]
        self.assertEqual(len(with_steps), 1)
        self.assertEqual(with_steps[0]["kind"], "candidate_found")
        self.assertEqual(len(with_steps[0]["search_steps"]), 2)
        self.assertIn("rotate to the unchecked sector", with_steps[0]["search_steps"][1])
        self.assertEqual(with_steps[0]["search_vlm_calls"], 3)
        self.assertEqual(annotation["counters"]["search_skill_calls"], 1)
        self.assertIn("steps: 1) grounding failed", "\n".join(annotation["narrative"]))

    def test_a_failed_leg_is_unproductive_and_counts_as_a_strategy_change(self):
        trace = [
            _entry(1, "ground_target", {"candidate_id": "cand_1",
                                        "candidate_position": [2.0, 0.5, 0.0]},
                   args={"observation_id": "obs_1"}, new_evidence=True),
            _entry(2, "approach_candidate",
                   {"reason": "unknown_candidate_id", "error": "unknown_candidate_id",
                    "pose": [0.0, 0.0, 1.0]},
                   args={"candidate_id": "cand_9"}, status=STATUS_FAILED,
                   new_evidence=False),
            _entry(3, "observe", {"observation_id": "obs_2", "pose": [0.0, 0.0, 1.0]},
                   args={"mode": "forward", "reason": "recover after the failed leg"}),
        ]
        _, annotation = self._annotation(trace)
        failed = next(event for event in annotation["events"]
                      if event["kind"] == "move_failed")
        self.assertEqual(failed["marker"], "unknown_candidate_id")
        change = next(event for event in annotation["events"]
                      if event["kind"] == "strategy_change")
        self.assertEqual(change["after"], "approach_candidate")
        self.assertEqual(change["action"], "observe")

    def test_repeating_the_same_call_is_not_a_strategy_change(self):
        trace = [
            _entry(1, "ground_target", {"reason": "no_candidate"},
                   args={"observation_id": "obs_1"}, status=STATUS_EMPTY,
                   new_evidence=False),
            _entry(2, "ground_target", {"reason": "no_candidate", "candidate_id": None},
                   args={"observation_id": "obs_2"}, status=STATUS_EMPTY,
                   new_evidence=False),
        ]
        _, annotation = self._annotation(trace)
        kinds = [event["kind"] for event in annotation["events"]]
        self.assertEqual(kinds.count("no_candidate"), 2)
        self.assertNotIn("strategy_change", kinds)

    def test_repeated_call_is_marked_as_a_repeat(self):
        trace = [
            _entry(1, "observe", {"observation_id": "obs_1"}, args={"mode": "lookaround"}),
            _entry(2, "observe", {"observation_id": "obs_1"}, args={"mode": "lookaround"},
                   status=STATUS_CACHED, new_evidence=False),
        ]
        _, annotation = self._annotation(trace)
        repeats = [event for event in annotation["events"] if event["kind"] == "repeat"]
        self.assertEqual(len(repeats), 1)
        self.assertEqual(repeats[0]["turn"], 2)
        self.assertEqual(annotation["counters"]["repeats"], 1)

    def test_annotation_survives_a_trace_with_missing_fields(self):
        trace = [
            {"turn": 1, "tool": "observe", "status": "ok", "data": {}},
            {"turn": 2, "tool": None, "status": "failed", "parse_failure": True},
            {"turn": 3, "tool": "finish", "status": "ok", "args": {}},
        ]
        row, annotation = self._annotation(trace)
        self.assertEqual(len(annotation["steps"]), 3)
        self.assertTrue(annotation["narrative"])
        self.assertFalse(row["claim"] is None)

    def test_search_tool_uses_the_steps_field_only(self):
        # ``steps`` is the search skill contract; an arbitrary list does not count
        trace = [_entry(1, "observe", {"observation_id": "obs_1", "candidates": [1, 2]},
                        args={"mode": "forward"})]
        _, annotation = self._annotation(trace)
        self.assertEqual(annotation["counters"]["search_skill_calls"], 0)


# ---------------------------------------------------------------------------
# aggregation, pairing and layer deltas
# ---------------------------------------------------------------------------


def _rows_for_deltas():
    rows = []
    values = {
        "baseline": dict(solved=False, false_confirmation=False, claim=False,
                         repeat_search_ratio=0.5, coverage_checked_deg=86.0,
                         direction_coverage=0.25, vlm_calls_total=6, tokens_estimate=3000,
                         wall_time_s=20.0),
        "vision_context": dict(solved=False, false_confirmation=True, claim=True,
                               repeat_search_ratio=0.4, coverage_checked_deg=172.0,
                               direction_coverage=0.5, vlm_calls_total=9, tokens_estimate=4200,
                               wall_time_s=25.0),
        "recovery": dict(solved=True, false_confirmation=False, claim=True,
                         repeat_search_ratio=0.2, coverage_checked_deg=258.0,
                         direction_coverage=0.75, vlm_calls_total=8, tokens_estimate=4000,
                         wall_time_s=23.0),
    }
    for arm, metrics in values.items():
        for cell in (("book", "book_near", 7), ("synth", "synth_near_fair", 11)):
            rows.append({
                "arm": arm, "scenario": cell[0], "task": cell[1], "seed": cell[2],
                "yaw_mode": "visible", "error": None, "skipped": False,
                "finish_reason": "finish", **metrics,
            })
    return rows


class AggregateTest(unittest.TestCase):
    def test_aggregate_reports_rates_and_means(self):
        stats = DRIVER.aggregate(_rows_for_deltas())
        self.assertEqual(stats["n"], 6)
        self.assertEqual(stats["n_ok"], 6)
        self.assertEqual(stats["n_solved"], 2)
        self.assertAlmostEqual(stats["solved_rate"], 0.333)
        self.assertEqual(stats["n_claims"], 4)
        self.assertEqual(stats["n_false_confirmation"], 2)
        self.assertAlmostEqual(stats["false_confirmation_rate"], 0.5)
        self.assertAlmostEqual(stats["repeat_search_ratio_mean"], 0.367)
        self.assertAlmostEqual(stats["vlm_calls_total_mean"], 7.667)
        self.assertEqual(stats["vlm_calls_total_sum"], 46)

    def test_honest_give_ups_are_counted_apart_from_the_claims(self):
        rows = [
            {"arm": "recovery", "scenario": "book", "task": "book_near", "seed": 7,
             "yaw_mode": "visible", "error": None, "skipped": False,
             "claim": False, "honest_finish": True, "false_confirmation": False},
            {"arm": "recovery", "scenario": "book", "task": "book_far", "seed": 7,
             "yaw_mode": "visible", "error": None, "skipped": False,
             "claim": True, "honest_finish": False, "false_confirmation": True},
        ]
        stats = DRIVER.aggregate(rows)
        self.assertEqual(1, stats["n_claims"])
        self.assertEqual(1, stats["n_false_confirmation"])
        self.assertAlmostEqual(1.0, stats["false_confirmation_rate"])
        self.assertEqual(1, stats["n_honest_finish"])
        self.assertAlmostEqual(0.5, stats["honest_finish_rate"])

    def test_error_and_skipped_rows_do_not_pollute_the_rates(self):
        rows = _rows_for_deltas()
        rows.append({"arm": "recovery", "error": "runner exited with code 1",
                     "skipped": False, "solved": False, "claim": True,
                     "false_confirmation": True})
        rows.append({"arm": "memory", "skipped": True, "error": None, "solved": False})
        stats = DRIVER.aggregate(rows)
        self.assertEqual(stats["n"], 8)
        self.assertEqual(stats["n_ok"], 6)
        self.assertEqual(stats["n_errors"], 1)
        self.assertEqual(stats["n_skipped"], 1)
        self.assertAlmostEqual(stats["false_confirmation_rate"], 0.5)

    def test_layer_deltas_are_paired_per_cell(self):
        rows = _rows_for_deltas()
        deltas = DRIVER.layer_deltas(rows, ["baseline", "vision_context", "recovery"])
        pairs = {(record["layer"], record["metric"]): record for record in deltas}
        first = pairs[("baseline->vision_context", "solved")]
        self.assertEqual(first["n_cells"], 2)
        self.assertEqual(first["mean_delta"], 0.0)   # the vision layer never solved here
        second = pairs[("vision_context->recovery", "solved")]
        self.assertEqual(second["mean_delta"], 1.0)
        self.assertEqual(second["wins"], 2)
        self.assertEqual(second["losses"], 0)
        false_confirm = pairs[("vision_context->recovery", "false_confirmation")]
        self.assertEqual(false_confirm["mean_delta"], -1.0)
        coverage = pairs[("baseline->vision_context", "coverage_checked_deg")]
        self.assertEqual(coverage["mean_delta"], 86.0)

    def test_layer_deltas_skip_cells_missing_an_arm(self):
        rows = [row for row in _rows_for_deltas()
                if not (row["arm"] == "recovery" and row["scenario"] == "book")]
        deltas = DRIVER.layer_deltas(rows, ["baseline", "vision_context", "recovery"])
        pairs = {(record["layer"], record["metric"]): record for record in deltas}
        self.assertEqual(pairs[("baseline->vision_context", "solved")]["n_cells"], 2)
        self.assertEqual(pairs[("vision_context->recovery", "solved")]["n_cells"], 1)

    def test_groups_cover_every_arm_scenario_and_yaw_mode(self):
        rows = _rows_for_deltas()
        groups = DRIVER.grouping(rows, ["book", "synth"], ["visible"], list(DRIVER.ARM_ORDER[:3]))
        keys = {(arm, group) for arm, group, _ in groups}
        for arm in DRIVER.ARM_ORDER[:3]:
            self.assertIn((arm, "all"), keys)
            self.assertIn((arm, "book"), keys)
            self.assertIn((arm, "synth"), keys)
            self.assertIn((arm, "yaw:visible"), keys)

    def test_markdown_report_holds_the_required_columns_and_annotations(self):
        rows = _rows_for_deltas()
        args = _args("--seeds", "7", "--initial-yaw-modes", "visible")
        scenario = _scenario()
        episodes = DRIVER.build_episodes(args, [scenario])
        commands = {
            (spec.arm, spec.scenario, spec.task.name, spec.seed, spec.yaw_mode):
                DRIVER.build_command(args, scenario, spec, DRIVER.episode_dir(Path(args.out), spec))
            for spec in episodes
        }
        groups = DRIVER.grouping(rows, ["book", "synth"], ["visible"], list(DRIVER.ARM_ORDER[:3]))
        aggregates = {f"{arm}|{group}": DRIVER.aggregate(subset)
                      for arm, group, subset in groups}
        rows[0]["annotation"] = {"episode": "baseline|cpu|t0|visible_seed7",
                                 "narrative": ["t1 observe [ok] -> new information: obs_1"],
                                 "counters": {"steps": 1}}
        meta = DRIVER.meta_block(args, [scenario], episodes, commands, aggregates)
        markdown = DRIVER.render_markdown(meta, rows, groups, aggregates,
                                          DRIVER.layer_deltas(rows, list(DRIVER.ARM_ORDER[:3])))
        for column in ("solved", "false_confirm", "repeat_search", "coverage", "vlm_calls",
                       "tokens", "wall_s"):
            self.assertIn(column, markdown)
        self.assertIn("| baseline | all |", markdown)
        self.assertIn("## Layer contributions (paired cells)", markdown)
        self.assertIn("## Decision annotation", markdown)
        self.assertIn("| vision_context->recovery | solved |", markdown)
        self.assertIn("new information: obs_1", markdown)


# ---------------------------------------------------------------------------
# contract fake: the observation/search result contract through the real runner
# ---------------------------------------------------------------------------


class _StubTarget:
    name = "target"
    radius_m = 0.5
    position_flu = np.asarray(TARGET, dtype=np.float64)


class _StubWorld:
    """Minimal world: pose, target geometry, per-leg travel."""

    LEG_M = 1.6

    def __init__(self, start=(0.0, 0.0, 1.0)):
        self.position_flu = np.asarray(start, dtype=np.float64)
        self.target = _StubTarget()
        self.trajectory = [tuple(float(v) for v in self.position_flu)]

    def distance_to_target(self):
        return float(np.linalg.norm(self.target.position_flu - self.position_flu))

    def apply_action(self, action):
        before = np.asarray(self.position_flu, dtype=np.float64)
        delta = self.target.position_flu - before
        horizontal = float(np.linalg.norm(delta[:2]))
        step = np.zeros(3)
        if horizontal > 0.0:
            step[:2] = delta[:2] / horizontal * min(self.LEG_M, horizontal)
        self.position_flu = before + step
        self.trajectory.append(tuple(float(v) for v in self.position_flu))
        return "moved"


def _stub_registry(modules=None, vlm=None):
    """The observation/search contract as a fake: field names only.

    ``observe`` carries the observation contract (``view_index`` +
    ``coverage{checked_deg, unchecked_deg, worth_looking}``), ``search_target``
    the search-skill contract (``steps`` + ``vlm_calls`` + ``candidate_id``)
    and ``ground_target`` answers ``no_candidate`` first so the recovery chain
    has an unproductive call to change away from.
    """
    registry = Runner.RecordingToolRegistry()
    state = {"ground": 0}

    def _pose(ctx):
        return [float(v) for v in np.asarray(ctx.world.position_flu).reshape(-1)[:3]]

    def observe(ctx, args):
        ctx.extra.setdefault("obs_seq", 0)
        ctx.extra["obs_seq"] += 1
        obs_id = f"obs_{ctx.extra['obs_seq']}"
        return ToolResult(
            STATUS_OK,
            {"observation_id": obs_id, "pose": _pose(ctx),
             "view_index": [{"obs_id": obs_id, "yaw_deg": 90.0, "direction": "left",
                             "note": "stub level view"}],
             "coverage": {"bin_deg": 45.0, "directions_known": True,
                          "checked_deg": [[90.0, 135.0]],
                          "unchecked_deg": [[0.0, 45.0]],
                          "worth_looking": [{"obs_id": obs_id, "yaw_deg": 90.0}],
                          "all_directions_covered": False, "down_checked": False},
             "vlm_calls": 1},
            [EvidenceRef("observation", obs_id, "stub view")],
        )

    def ground_target(ctx, args):
        state["ground"] += 1
        if state["ground"] == 1:
            return ToolResult(STATUS_EMPTY, {"reason": "no_candidate", "pose": _pose(ctx)},
                              [EvidenceRef("observation", "obs_1", "empty grounding")])
        return ToolResult(
            STATUS_OK,
            {"candidate_id": "cand_1", "observation_id": "obs_2", "pose": _pose(ctx),
             "candidate_position": [2.0, 0.5, 0.0]},
            [EvidenceRef("observation", "obs_2", "grounded view")],
        )

    def search_target(ctx, args):
        return ToolResult(
            STATUS_OK,
            {"candidate_id": "cand_1", "observation_id": "obs_2", "pose": _pose(ctx),
             "vlm_calls": 4, "cost": {"vlm_calls": 4, "turns": 2, "tool_calls": 3},
             "coverage": {"bin_deg": 45.0, "directions_known": True,
                          "checked_deg": [[-180.0 + 45.0 * i, -135.0 + 45.0 * i]
                                          for i in range(8)],
                          "unchecked_deg": [], "worth_looking": [],
                          "all_directions_covered": True, "down_checked": False},
             "view_index": [{"obs_id": "obs_2", "yaw_deg": 17.0, "direction": "ahead"}],
             "steps": [
                 {"step": 1, "why": "grounding failed", "obs_ids": ["obs_1"],
                  "result": "no_candidate"},
                 {"step": 2, "why": "rotate to the unchecked sector",
                  "obs_ids": ["obs_2"], "result": "candidate cand_1"},
             ]},
            [EvidenceRef("observation", "obs_2", "search result")],
        )

    def approach_candidate(ctx, args):
        ctx.world.apply_action(None)
        pose = _pose(ctx)
        return ToolResult(
            STATUS_OK,
            {"candidate_id": "cand_1", "outcome": "moved", "segment_progress_m": 1.6,
             "progress_m": 1.6, "pose": pose},
            [EvidenceRef("trajectory", "nav_1", "leg")],
        )

    def verify_goal(ctx, args):
        distance = float(ctx.world.distance_to_target())
        return ToolResult(
            STATUS_OK,
            {"verified": distance <= 2.0, "satisfied": distance <= 2.0,
             "distance_m": distance, "pose": _pose(ctx)},
            [EvidenceRef("observation", "obs_2", "verification")],
        )

    for name, handler, side_effects in (
        ("observe", observe, False),
        ("ground_target", ground_target, False),
        ("search_target", search_target, True),
        ("approach_candidate", approach_candidate, True),
        ("verify_goal", verify_goal, False),
    ):
        registry.register(ToolSpec(name=name, description=f"stub {name}",
                                   input_schema={"type": "object", "properties": {}},
                                   handler=handler, side_effects=side_effects))
    return registry, "stub", {name: "stub" for name in registry.names()}


class ContractStubEpisodeTest(unittest.TestCase):
    """Run the stub contract through the real runner, then measure the trace."""

    SCRIPT = [
        {"tool": "observe", "args": {"mode": "forward", "reason": "look"}},
        {"tool": "ground_target", "args": {"observation_id": "obs_1",
                                           "reason": "locate the target"}},
        {"tool": "search_target", "args": {"reason": "search the unchecked sector"}},
        {"tool": "approach_candidate", "args": {"candidate_id": "cand_1",
                                                "reason": "close in"}},
        {"tool": "verify_goal", "args": {"reason": "confirm"}},
        {"tool": "finish", "args": {"reason": "verified"}},
    ]

    def _run(self, out_dir: Path) -> dict:
        args = Runner.parse_args([
            "--world", "cpu",
            "--target-pos", "2", "0.5", "0",
            "--instruction", TASK_JSON["instruction"],
            "--backend", "mock",
            "--max-turns", "10",
            "--tool-budget", "20",
            "--out", str(out_dir),
        ])
        vlm = Runner.ScriptedToolVLM([dict(entry) for entry in self.SCRIPT])
        return Runner.run_episode(_StubWorld(), vlm, args, out_dir,
                                  episode_id="ep_contract",
                                  registry_factory=_stub_registry)

    def test_the_driver_reads_the_contract_fields_of_a_real_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            summary = self._run(out_dir)
            self.assertIsNone(summary.get("skipped"))
            driver_args = _args("--max-vlm-calls", "10")
            row = DRIVER.extract_metrics(driver_args, _scenario(),
                                         _spec("recovery"), out_dir,
                                         {"returncode": 0, "error": None})
            self.assertIsNone(row["error"])
            # observe/search view_index + sector coverage made it into the trace
            self.assertEqual(row["coverage_source"], "reported")
            self.assertEqual(row["coverage_checked_deg"], 360.0)
            self.assertEqual(row["coverage_checked_sectors"], 8)
            self.assertEqual(row["coverage_total_sectors"], 8)
            self.assertEqual(row["coverage_unchecked_sectors"], 0)
            self.assertEqual(row["direction_coverage"], 1.0)
            self.assertEqual(row["coverage_worth_looking"], 1)
            self.assertEqual(row["coverage_full_sweeps"], 1)
            self.assertGreaterEqual(row["views_expanded"], 1)
            # search_target's declared internal calls join the cost
            self.assertGreaterEqual(row["vlm_calls_declared"], 4)
            self.assertGreater(row["vlm_calls_total"], row["vlm_calls_decision"])
            # the annotation expands the search skill's step log
            annotation = row["annotation"]
            self.assertEqual(annotation["counters"]["search_skill_calls"], 1)
            self.assertEqual(annotation["counters"]["candidates_found"], 1)
            self.assertGreaterEqual(annotation["counters"]["strategy_changes"], 1)
            with_steps = [event for event in annotation["events"]
                          if event.get("search_steps")]
            self.assertEqual(len(with_steps), 1)
            self.assertEqual(len(with_steps[0]["search_steps"]), 2)
            self.assertIn("candidate cand_1", with_steps[0]["search_steps"][1])
            # progress_kind: reported when the runtime provides it, else derived
            reported = [step for step in annotation["steps"]
                        if step["progress_kind_source"] == "reported"]
            derived = [step for step in annotation["steps"]
                       if step["progress_kind_source"] == "derived"]
            self.assertTrue(reported or derived)
            for step in annotation["steps"]:
                self.assertIn(step["progress_kind"], DRIVER.PROGRESS_KINDS)
            for kind in [entry.get("progress_kind") for entry in
                         DRIVER.trace_entries(out_dir) if entry.get("progress_kind")]:
                self.assertIn(kind, DRIVER.PROGRESS_KINDS)


# ---------------------------------------------------------------------------
# end to end: the driver runs three arms on the CPU world with mock backends
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


MOCK_SCRIPT = [
    {"tool": "observe", "args": {"mode": "forward", "reason": "look ahead"}},
    {"tool": "ground_target", "args": {"observation_id": "$observation_id",
                                       "reason": "locate the target"}},
    {"tool": "verify_goal", "args": {"reason": "confirm before finishing"}},
    {"tool": "finish", "args": {"reason": "done"}},
]


class EndToEndTest(unittest.TestCase):
    def _run_driver(self, out_dir: Path, scene: Path, *extra: str):
        cmd = [
            sys.executable, "-m", "scripts.compare_decision_loop",
            "--scenario", "cpu",
            "--world", "cpu",
            "--ply", str(scene),
            "--target-name", "target",
            "--target-pos", "2", "0.5", "0",
            "--target-radius", "0.5",
            "--start-pos", "0", "0", "1",
            "--success-dist", "1.0",
            "--instruction", TASK_JSON["instruction"],
            "--backend", "mock",
            "--out", str(out_dir),
            *extra,
        ]
        env = dict(os.environ)
        env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
        return subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True,
                              text=True, timeout=1800)

    def test_dry_run_lists_three_arms_per_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene.ply"
            _make_scene(scene)
            proc = self._run_driver(Path(tmp) / "out", scene, "--dry-run",
                                    "--seeds", "7",
                                    "--initial-yaw-modes", "visible,random")
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            commands = [line for line in proc.stdout.splitlines()
                        if line.startswith(sys.executable)]
            self.assertEqual(len(commands), 3 * 2)  # arms x yaw modes
            maps = [DRIVER.compare._args_map(line.split()) for line in commands]
            self.assertEqual(sum("--context-frames" in cmd for cmd in maps), 2 * 2)
            for key in ("--ply", "--start-pos", "--seed", "--instruction"):
                self.assertEqual(len({cmd[key] for cmd in maps}), 1, key)

    def test_mock_run_writes_the_decision_report_with_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene.ply"
            _make_scene(scene)
            script = Path(tmp) / "mock_script.json"
            script.write_text(json.dumps(MOCK_SCRIPT))
            out_dir = Path(tmp) / "decision"
            proc = self._run_driver(
                out_dir, scene,
                "--seeds", "7",
                "--initial-yaw-modes", "visible,random",
                "--max-vlm-calls", "4",
                "--tool-budget", "6",
                "--scan-views", "4",
                "--jobs", "3",
                f"--tool-arg=--mock-script={script}",
            )
            diagnostics = (proc.stdout[-4000:] + "\n--- stderr ---\n" + proc.stderr[-2000:])
            self.assertEqual(proc.returncode, 0, "driver failed" + diagnostics)

            report_path = out_dir / DRIVER.REPORT_JSON
            markdown_path = out_dir / DRIVER.REPORT_MD
            self.assertTrue(report_path.exists(), diagnostics)
            self.assertTrue(markdown_path.exists(), diagnostics)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["n_errors"], 0, diagnostics)
            self.assertEqual(len(report["episodes"]), 3 * 2, diagnostics)
            self.assertEqual({row["arm"] for row in report["episodes"]},
                             {"baseline", "vision_context", "recovery"}, diagnostics)
            self.assertTrue(report["meta"]["pairing_check"]["consistent"], diagnostics)
            self.assertEqual(len(report["meta"]["arm_support"]), 3, diagnostics)
            for record in report["meta"]["arm_support"]:
                self.assertIn(record["support"], ("full", "degraded"), record)
            self.assertEqual(report["meta"]["budget"]["tool"]["max_turns"], 4)
            self.assertEqual(len(report["layer_deltas"]), 2 * len(DRIVER.LAYER_METRICS))

            for row in report["episodes"]:
                self.assertEqual(row["returncode"], 0, row)
                self.assertIsNone(row["error"], row)
                for key in ("solved", "success", "claim", "finish_declared",
                            "honest_finish", "false_confirmation",
                            "repeat_search_ratio", "coverage_source", "vlm_calls_total",
                            "tokens_estimate", "wall_time_s", "toward_target_rate",
                            "annotation"):
                    self.assertIn(key, row, key)
                self.assertIsNotNone(row["vlm_calls_total"], row)
                self.assertTrue(Path(row["out_dir"], "summary.json").exists(), row)
                self.assertTrue(Path(row["out_dir"], "run.log").exists(), row)
                # the prompt identity of the episode is neutral (never the arm
                # name) and the completion check used the evaluation gate
                self.assertTrue(row["prompt_episode_id"], row)
                self.assertNotIn(row["arm"], row["prompt_episode_id"])
                self.assertEqual(1.5, row["gate_m"], row)      # 1.0 + 0.5 radius
                self.assertEqual(1.5, row["verify_limit_m"], row)
                self.assertTrue(row["verify_gate_consistent"], row)
                annotation = row["annotation"]
                self.assertTrue(annotation["narrative"], row)
                self.assertEqual(annotation["episode"].split("|")[0], row["arm"])
                self.assertTrue(any(line.startswith("chain:") for line in annotation["narrative"]))

            gate = report["meta"]["verify_gate"]
            self.assertEqual(6, gate["checked"], diagnostics)
            self.assertEqual(0, gate["mismatch"], diagnostics)
            identity = report["meta"]["prompt_identity_check"]
            self.assertTrue(identity["consistent"], diagnostics)
            self.assertEqual(2, identity["cells"], diagnostics)      # two yaw modes
            self.assertEqual(6, identity["episodes_checked"], diagnostics)

            aggregates = report["meta"]["aggregates"]
            for arm in ("baseline", "vision_context", "recovery"):
                self.assertIn(f"{arm}|all", aggregates, diagnostics)
                self.assertIn(f"{arm}|cpu", aggregates, diagnostics)
                self.assertIn(f"{arm}|yaw:visible", aggregates, diagnostics)
                self.assertEqual(aggregates[f"{arm}|all"]["n"], 2, diagnostics)

            text = markdown_path.read_text()
            for column in ("solved", "false_confirm", "honest_finish", "repeat_search",
                           "coverage", "tokens", "wall_s"):
                self.assertIn(column, text)
            self.assertIn("## Decision annotation", text)
            self.assertIn("| baseline | all |", text)
            self.assertIn("## Layer contributions (paired cells)", text)
            for arm in ("baseline", "vision_context", "recovery"):
                self.assertIn(f"### {arm}|cpu|", text)

    def test_the_memory_arms_run_the_runner_with_their_memory_layer(self):
        """The memory arms are wired, not reserved: their episodes really run the
        tool runner with the memory layer switches, on top of the recovery arm."""
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene.ply"
            _make_scene(scene)
            script = Path(tmp) / "mock_script.json"
            script.write_text(json.dumps(MOCK_SCRIPT))
            out_dir = Path(tmp) / "memory_arms"
            proc = self._run_driver(
                out_dir, scene,
                "--arms", "recovery,memory,memory_spatial_only",
                "--seeds", "7",
                "--initial-yaw-modes", "visible",
                "--max-vlm-calls", "4",
                "--tool-budget", "6",
                "--scan-views", "4",
                "--jobs", "3",
                f"--tool-arg=--mock-script={script}",
            )
            diagnostics = (proc.stdout[-4000:] + "\n--- stderr ---\n" + proc.stderr[-2000:])
            self.assertEqual(proc.returncode, 0, "driver failed" + diagnostics)
            report = json.loads((out_dir / DRIVER.REPORT_JSON).read_text())
            self.assertEqual(report["n_errors"], 0, diagnostics)
            self.assertEqual(report["n_skipped"], 0, diagnostics)
            rows = report["episodes"]
            self.assertEqual({row["arm"] for row in rows},
                             {"recovery", "memory", "memory_spatial_only"}, diagnostics)
            for row in rows:
                self.assertFalse(row["skipped"], row)
                self.assertNotEqual("reserved_arm", row["finish_reason"], row)
                self.assertEqual(0, row["returncode"], row)

            expected = {
                "recovery": {"--toolset": "full", "spatial_memory": "off",
                             "episodic_memory": "off"},
                "memory": {"--toolset": "full", "spatial_memory": "on",
                           "episodic_memory": "on"},
                "memory_spatial_only": {"--toolset": "full", "spatial_memory": "on",
                                        "episodic_memory": "off"},
            }
            for arm, want in expected.items():
                summaries = [json.loads(Path(row["out_dir"], "summary.json").read_text())
                             for row in rows if row["arm"] == arm]
                self.assertTrue(summaries, arm)
                for summary in summaries:
                    requested = summary["switches"]["requested"]
                    self.assertEqual(want["--toolset"], summary["toolset"], arm)
                    self.assertEqual(want["spatial_memory"],
                                     requested["spatial_memory"], arm)
                    self.assertEqual(want["episodic_memory"],
                                     requested["episodic_memory"], arm)
                    # the memory arms keep the shared catalogue, so the search
                    # skill of the layer before them really stays registered
                    self.assertTrue(summary["switches"]["applied"]["search_skill"], arm)
                    # the built memory either attests exactly what the arm asked
                    # for, or does not report the layer at all - the runner never
                    # invents a memory layer
                    for name in ("spatial_memory", "episodic_memory"):
                        asked_on = want[name] == "on"
                        self.assertIn(summary["switches"]["applied"][name],
                                      (asked_on, None), summary["switches"]["applied"])
                    # ... and the memory's own stats block says the same: the
                    # layer really reached the store, not just the CLI
                    stated = (summary.get("memory") or {}).get("switches")
                    if isinstance(stated, dict):
                        for name in ("spatial_memory", "episodic_memory"):
                            self.assertIs(stated.get(name), want[name] == "on",
                                          summary["memory"])

            mechanisms = {record["arm"]: record for record in report["meta"]["mechanism_check"]}
            self.assertEqual(set(expected), set(mechanisms), diagnostics)
            for arm, want in expected.items():
                record = mechanisms[arm]
                self.assertEqual(want["--toolset"], record["toolset"], record)
                self.assertTrue(record["consistent"], record)
                requested = record["requested"]
                self.assertEqual(want["spatial_memory"], requested["spatial_memory"], arm)
                self.assertEqual(want["episodic_memory"], requested["episodic_memory"], arm)
            # an arm that asked for a layer the memory did not confirm says so
            for arm in ("memory", "memory_spatial_only"):
                unattested = mechanisms[arm]["unattested"]
                for name in unattested:
                    self.assertIn(name, ("spatial_memory", "episodic_memory"))

            text = (out_dir / DRIVER.REPORT_MD).read_text()
            for column in ("spatial_memory", "episodic_memory", "toolset"):
                self.assertIn(column, text)
            for arm in ("memory", "memory_spatial_only"):
                self.assertIn(f"### {arm}|cpu|", text)

    def test_the_m2_catalogue_variant_is_reported_with_what_it_really_ran(self):
        """The goal.md M2 memory arm (``--toolset memory``) as an override: the
        report says the arm ran that catalogue - and that the catalogue cannot
        carry the search skill the arm declared, instead of pretending it did."""
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene.ply"
            _make_scene(scene)
            script = Path(tmp) / "mock_script.json"
            script.write_text(json.dumps(MOCK_SCRIPT))
            out_dir = Path(tmp) / "m2_catalogue"
            proc = self._run_driver(
                out_dir, scene,
                "--arms", "recovery,memory",
                "--arm-arg",
                f"memory=--toolset {DRIVER.M2_MEMORY_TOOLSET} --context-frames 4 "
                "--search-skill on --state-block on --recovery on "
                "--memory-spatial on --memory-episodic on",
                "--seeds", "7",
                "--initial-yaw-modes", "visible",
                "--max-vlm-calls", "4",
                "--tool-budget", "6",
                "--scan-views", "4",
                "--jobs", "2",
                f"--tool-arg=--mock-script={script}",
            )
            diagnostics = (proc.stdout[-4000:] + "\n--- stderr ---\n" + proc.stderr[-2000:])
            self.assertEqual(proc.returncode, 0, "driver failed" + diagnostics)
            report = json.loads((out_dir / DRIVER.REPORT_JSON).read_text())
            self.assertEqual(report["n_errors"], 0, diagnostics)
            mechanisms = {record["arm"]: record for record in report["meta"]["mechanism_check"]}
            memory = mechanisms["memory"]
            self.assertEqual(DRIVER.M2_MEMORY_TOOLSET, memory["toolset"])
            self.assertEqual("on", memory["requested"]["search_skill"])
            # the memory catalogue carries no search_target: the applied value
            # says off, so the report never claims a skill the arm could not use
            self.assertIs(False, memory["applied"]["search_skill"])
            self.assertTrue(memory["consistent"], memory)
            # ... and the catalogue is visible in the episode's own summary
            row = next(row for row in report["episodes"] if row["arm"] == "memory")
            summary = json.loads(Path(row["out_dir"], "summary.json").read_text())
            registered = summary["toolset_detail"]["registered"]
            self.assertNotIn(Runner.SEARCH_SKILL_TOOL, registered, summary["toolset_detail"])
            self.assertIn("query_memory", registered)
            text = (out_dir / DRIVER.REPORT_MD).read_text()
            self.assertIn("--toolset memory", text)

    def test_a_reserved_arm_is_still_skipped_when_one_is_configured(self):
        """The reserved-arm machinery stays (the next round's identity memory
        will use it): a reserved arm records ``skipped``, never a fake run."""
        reserved = DRIVER.ArmSpec(
            name="memory_identity", label="+ visual identity memory (next round)",
            runner_args=(), note="reserved: interface only", reserved=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene.ply"
            _make_scene(scene)
            out_dir = Path(tmp) / "reserved"
            with mock.patch.dict(DRIVER.ARM_TABLE, {"memory_identity": reserved}), \
                    mock.patch.object(DRIVER, "ARM_ORDER",
                                      DRIVER.ARM_ORDER + ("memory_identity",)):
                args = DRIVER.parse_args([
                    "--scenario", "cpu", "--world", "cpu", "--ply", str(scene),
                    "--target-pos", "2", "0.5", "0", "--instruction", TASK_JSON["instruction"],
                    "--backend", "mock", "--arms", "memory_identity", "--toolset", "full",
                    "--seeds", "7", "--initial-yaw-modes", "visible",
                    "--max-vlm-calls", "2", "--out", str(out_dir),
                ])
                DRIVER.resolve_matrix(args)
                summary = DRIVER.run_comparison(args)
            self.assertEqual(summary["n_errors"], 0)
            self.assertEqual(summary["n_skipped"], 1)
            row = summary["episodes"][0]
            self.assertTrue(row["skipped"])
            self.assertEqual(row["finish_reason"], "reserved_arm")
            self.assertIsNone(row["error"])
            # no runner was started: the episode directory holds no artifacts
            self.assertEqual(list(Path(row["out_dir"]).glob("*.json*")), [])
            report = json.loads((out_dir / DRIVER.REPORT_JSON).read_text())
            self.assertEqual(report["meta"]["arms"][0]["reserved"], True)


if __name__ == "__main__":
    unittest.main()
