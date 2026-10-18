"""Integration tests for the M1 tool-calling runner (scripts/run_vlm_nav_tools.py).

Covers the goal.md acceptance play end-to-end on a CPU world with a scripted
MockVLM: tools are really executed, results are fed back (the next call consumes
the previous result id), every call is logged with status/time/evidence, evidence
refs resolve to real files, and summary.json carries the per-tool usage stats.

The parallel M1 modules (``vlm_nav.runtime`` / ``vlm_nav.memory`` /
``vlm_nav.tool_impl_*``) are not required: the runner's fallback stack is used
(or stubs injected through ``sys.modules`` to check the adapter path).
"""

from __future__ import annotations

import importlib.util
import json
import math
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_vlm_nav_tools.py"


def _load_runner():
    name = "diffaero_script_run_vlm_nav_tools"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()

from vlm_nav.tools import (  # noqa: E402  (import after the runner load)
    STATUS_BLOCKED,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_OK,
    EvidenceRef,
    ToolContext,
    ToolRegistry,
    ToolSpec,
    parse_tool_call,
)
from vlm_nav.vlm import MockVLM  # noqa: E402
from vlm_nav.world import SplatWorldCPU, TargetSpec  # noqa: E402

_M1_MODULE_KEYS = ("runtime", "memory", "perception", "navigation")


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


TARGET_FLU = (2.0, 0.5, 0.0)


def _make_world(directory: Path, size: int = 64):
    """Tiny CPU 3DGS scene: sparse floor + a red cluster at the target."""
    rng = np.random.default_rng(0)
    gaussians = []
    for _ in range(200):
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
                (2.0, -1.0, -1.0),
                _logit(0.9),
                (math.log(0.15), math.log(0.15), math.log(0.15)),
                (1.0, 0.0, 0.0, 0.0),
            )
        )
    ply = directory / "scene.ply"
    _write_test_ply(ply, gaussians)
    world = SplatWorldCPU(
        ply_path=ply,
        target=TargetSpec(name="target", position_flu=TARGET_FLU, radius_m=0.2),
        start_flu=(0.0, 0.0, 0.2),
        start_yaw_deg=0.0,
        width=size,
        height=size,
        hfov_deg=60.0,
    )
    return world


def _runner_args(out_dir: Path, **overrides) -> "object":
    argv = [
        "--world", "cpu",
        "--target-name", "target",
        "--target-pos", "2.0", "0.5", "0.0",
        "--target-radius", "0.2",
        "--start-pos", "0", "0", "0.2",
        "--instruction", "Find the target and fly to it.",
        "--render-size", "64", "64",
        "--success-dist", "2.5",
        "--max-turns", "20",
        "--tool-budget", "40",
        "--out", str(out_dir),
    ]
    for key, value in overrides.items():
        argv.extend([f"--{key.replace('_', '-')}", str(value)])
    return RUNNER.parse_args(argv)


def _scripted_vlm(script=None):
    # ScriptedToolVLM resolves "$field" placeholders from tool results and the
    # world (the acceptance script queries "$target_name"); a plain MockVLM
    # would leak the placeholder through to the tools.
    return RUNNER.ScriptedToolVLM([dict(entry) for entry in
                                   (script or RUNNER.ACCEPTANCE_TOOL_SCRIPT)])


def _read_trace(out_dir: Path) -> list[dict]:
    path = out_dir / "tool_trace.jsonl"
    assert path.exists(), f"missing {path}"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class FallbackStackTest(unittest.TestCase):
    def test_registry_exposes_m1_tools(self):
        registry = ToolRegistry()
        added = RUNNER.register_fallback_tools(registry)
        self.assertEqual(set(RUNNER.M1_TOOL_NAMES), set(added))
        self.assertEqual(set(RUNNER.M1_TOOL_NAMES), set(registry.names()))

        specs_text = registry.specs_text()
        for name in RUNNER.M1_TOOL_NAMES:
            spec = registry.get(name)
            self.assertTrue(spec.description, name)
            self.assertEqual(spec.input_schema.get("type"), "object", name)
            self.assertIn(name, specs_text)
        # pose-changing tools must be serialized by the runtime
        self.assertTrue(registry.get("navigate").side_effects)
        self.assertFalse(registry.get("observe").side_effects)

        # skip_existing makes the fallback composable with module tools
        added_again = RUNNER.register_fallback_tools(registry)
        self.assertEqual(set(), added_again)

    def test_mock_script_window_matches_acceptance_play(self):
        # goal.md: unclear target -> inspect_region; pick a viewpoint ->
        # go_to_viewpoint; lost -> query_memory; the object -> ground_target +
        # approach_candidate; re-observe -> verify_goal -> finish. Every movement
        # is an intent tool: the deprecated ``navigate`` is not in the play.
        names = [entry["tool"] for entry in RUNNER.ACCEPTANCE_TOOL_SCRIPT]
        self.assertEqual(
            ["observe", "inspect_region", "ground_target", "propose_viewpoints",
             "go_to_viewpoint", "query_memory", "observe", "approach_candidate",
             "verify_goal", "finish"],
            names,
        )
        self.assertNotIn("navigate", names)
        region = RUNNER.ACCEPTANCE_TOOL_SCRIPT[1]["args"]["region"]
        self.assertEqual([400, 400, 600, 600], region)  # canonical corner box

    def test_load_mock_script_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scripts.json"
            script = [{"tool": "observe", "args": {"mode": "forward"}}]
            path.write_text(json.dumps(script))
            loaded = RUNNER._load_mock_script(str(path))
            self.assertEqual(script, loaded)
            wrapped = Path(tmp) / "wrapped.json"
            wrapped.write_text(json.dumps({"script": script}))
            self.assertEqual(script, RUNNER._load_mock_script(str(wrapped)))


class FallbackInspectRegionTest(unittest.TestCase):
    """The fallback stack reads the same box contract as the module tools.

    The duplicated parser used to assume the legacy ``[u, v, w, h]`` spelling
    (and said so in a comment claiming it matched ``tool_impl_perception``);
    the canonical spelling is the corner box, and the answer names the one it
    used.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.out_dir = self.tmp / "out"
        self.world = _make_world(self.tmp)
        self.args = _runner_args(self.out_dir)
        self.registry = ToolRegistry()
        RUNNER.register_fallback_tools(self.registry)
        self.ctx = RUNNER.make_tool_context(
            self.world, RUNNER._FallbackMemory(), self.args, self.out_dir, "ep_crop"
        )
        observed = self.registry.call("observe", {"mode": "forward"}, self.ctx)
        self.assertEqual(STATUS_OK, observed.status)
        self.obs_id = observed.data["observation_id"]

    def _inspect(self, region):
        return self.registry.call(
            "inspect_region", {"observation_id": self.obs_id, "region": region}, self.ctx
        )

    def test_a_corner_region_is_read_as_a_corner_box(self):
        result = self._inspect([400, 400, 600, 600])

        self.assertEqual(STATUS_OK, result.status)
        self.assertEqual("xyxy_norm", result.data["region_format"])
        self.assertEqual([400.0, 400.0, 600.0, 600.0], result.data["region"])
        self.assertTrue((self.out_dir / "frames" /
                         f"{result.data['observation_id']}.png").exists())

    def test_the_legacy_size_region_is_still_read(self):
        result = self._inspect([400, 400, 200, 200])  # u, v, w, h

        self.assertEqual(STATUS_OK, result.status)
        self.assertEqual("uvwh_norm", result.data["region_format"])
        self.assertEqual([400.0, 400.0, 600.0, 600.0], result.data["region"])

    def test_a_declared_format_overrides_the_guess(self):
        result = self._inspect([400, 400, 600, 600])
        declared = self.registry.call(
            "inspect_region",
            {"observation_id": self.obs_id, "region": [400, 400, 200, 100],
             "box_format": "uvwh"},
            self.ctx,
        )

        self.assertEqual("uvwh_norm", declared.data["region_format"])
        self.assertEqual([400.0, 400.0, 600.0, 500.0], declared.data["region"])
        self.assertEqual("xyxy_norm", result.data["region_format"])

    def test_a_region_outside_the_frame_is_a_named_error(self):
        result = self._inspect([1500, 1500, 1600, 1600])

        self.assertEqual(STATUS_FAILED, result.status)
        self.assertEqual("region_outside_frame", result.data["error"])
        self.assertIn("x1, y1, x2, y2", result.data["box_format"])


class ScriptedVlmTest(unittest.TestCase):
    """The scripted mock is the model of the loop: it consumes result fields."""

    def setUp(self):
        self.vlm = RUNNER.ScriptedToolVLM(
            [
                {"tool": "inspect_region",
                 "args": {"observation_id": "$observation_id", "region": [0, 0, 10, 10]}},
                {"prose": "still thinking"},
                "not even json",
            ]
        )
        self.frames = [torch.zeros(3, 8, 8)]

    def _ask(self):
        # the module AgentRuntime asks through decide_with_tools
        return self.vlm.decide_with_tools(self.frames, "system", "user", "tools")

    def test_placeholders_resolved_from_the_result_index(self):
        self.vlm.bind_result_index({"observation_id": "obs_7"})
        parsed = parse_tool_call(self._ask())
        self.assertEqual(("inspect_region",
                          {"observation_id": "obs_7", "region": [0, 0, 10, 10]}), parsed)

    def test_unparseable_entries_and_exhausted_script(self):
        first = parse_tool_call(self._ask())                  # scripted tool call
        self.assertEqual("inspect_region", first[0])
        self.assertIsNone(parse_tool_call(self._ask()))       # scripted prose
        self.assertIsNone(parse_tool_call(self._ask()))       # scripted raw text
        parsed = parse_tool_call(self._ask())                 # exhausted -> finish
        self.assertEqual("finish", parsed[0])
        self.assertIn("reason", parsed[1])

    def test_legacy_decide_path_shares_the_script(self):
        self.vlm.bind_result_index({"observation_id": "obs_1"})
        parsed = parse_tool_call(self.vlm.decide(self.frames[0], "system", "user"))
        self.assertEqual("obs_1", parsed[1]["observation_id"])


class TraceAggregationTest(unittest.TestCase):
    def test_resolve_refs(self):
        last = {"observation_id": "obs_3", "viewpoint_id": "vp_1"}
        resolved = RUNNER._resolve_refs(
            {"observation_id": "$observation_id", "region": [0, 0, 10, 10],
             "viewpoint_id": "$viewpoint_id", "unknown": "$nope", "n": 2},
            last,
        )
        self.assertEqual("obs_3", resolved["observation_id"])
        self.assertEqual("vp_1", resolved["viewpoint_id"])
        self.assertEqual("$nope", resolved["unknown"])  # left untouched
        self.assertEqual([0, 0, 10, 10], resolved["region"])

    def test_summarize_trace_counts_status_usage_and_evidence(self):
        entries = [
            {"turn": 1, "tool": "observe", "status": "ok", "ms": 10.0,
             "args": {"mode": "forward"},
             "data": {"observation_id": "obs_1"},
             "evidence": [{"kind": "image", "ref": "frames/obs_1.png"},
                          {"kind": "observation", "ref": "obs_1"}]},
            {"turn": 2, "tool": None, "status": "parse_failed", "ms": 0.0,
             "parse_failure": True, "data": {}, "evidence": []},
            {"turn": 3, "tool": "inspect_region", "status": "ok", "ms": 6.0,
             "args": {"observation_id": "obs_1", "region": [400, 400, 200, 200]},
             "data": {"observation_id": "obs_1_crop"},
             "evidence": [{"kind": "image", "ref": "frames/obs_1_crop.png"},
                          {"kind": "image", "ref": "frames/obs_1.png"}]},
            {"turn": 4, "tool": "navigate", "status": "blocked", "ms": 20.0,
             "args": {"viewpoint_id": "vp_1"},
             "data": {"task_id": "nav_1"}, "evidence": [{"kind": "trajectory", "ref": "nav_1"}]},
            {"turn": 5, "tool": "finish", "status": "ok", "ms": 0.0,
             "data": {"reason": "done"}, "evidence": []},
        ]
        stats = RUNNER.summarize_trace(entries)
        self.assertEqual(3, stats["total_calls"])
        self.assertEqual(1, stats["parse_failures"])
        self.assertEqual(1, stats["finish_calls"])
        self.assertEqual(5, stats["turns"])
        self.assertEqual(["inspect_region", "navigate", "observe"],
                         sorted(stats["tools"]))
        observe = stats["tools"]["observe"]
        self.assertEqual(1, observe["calls"])
        self.assertEqual(1, observe["ok"])
        self.assertEqual(2, observe["new_evidence"])
        self.assertTrue(observe["consumed"])  # obs_1 used by turn 3
        self.assertAlmostEqual(10.0, observe["time_ms_mean"])
        self.assertEqual(1, stats["tools"]["navigate"]["blocked"])
        self.assertEqual(0.0, stats["tools"]["navigate"]["ok_rate"])
        self.assertFalse(stats["tools"]["navigate"]["consumed"])
        # "frames/obs_1.png" seen twice -> only the first is new evidence
        self.assertEqual(4, stats["new_evidence_total"])
        self.assertEqual({"image": 3, "observation": 1, "trajectory": 1},
                         stats["evidence_kinds"])

    def test_summarize_trace_does_not_confuse_id_prefixes(self):
        entries = [
            {"turn": 1, "tool": "observe", "status": "ok", "ms": 1.0,
             "data": {"observation_id": "obs_1"}, "evidence": []},
            {"turn": 2, "tool": "observe", "status": "ok", "ms": 1.0,
             "data": {"observation_id": "obs_10"}, "evidence": []},
        ]
        stats = RUNNER.summarize_trace(entries)
        # obs_1 must not be "consumed" by the unrelated obs_10 id
        self.assertFalse(stats["tools"]["observe"]["consumed"])


class RuntimeContractTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.out_dir = self.tmp / "out"
        self.world = _make_world(self.tmp)

    def test_tool_calls_are_logged_with_status_time_and_evidence(self):
        args = _runner_args(self.out_dir, max_turns=4, tool_budget=2)
        registry = ToolRegistry()
        RUNNER.register_fallback_tools(registry)
        memory = RUNNER._FallbackMemory()
        ctx = RUNNER.make_tool_context(self.world, memory, args, self.out_dir, "ep_test")
        script = [
            {"tool": "observe", "args": {"mode": "forward"}},
            {"tool": "query_memory", "args": {"query": "target"}},
            {"tool": "observe", "args": {"mode": "forward"}},
            {"tool": "observe", "args": {"mode": "forward"}},
            {"tool": "finish", "args": {"reason": "unreachable"}},
        ]
        runtime = RUNNER._ToolRuntimeFallback(
            world=self.world, vlm=_scripted_vlm(script), registry=registry, ctx=ctx,
            instruction=args.instruction, max_turns=args.max_turns,
            context_frames=args.context_frames, success_dist_m=args.success_dist,
        )
        result = runtime.run()

        self.assertEqual("budget_exhausted", result["finish_reason"])
        self.assertEqual(0, ctx.budget["tool_calls_left"])
        # registry.call is the single log source: 2 executed calls, no more
        self.assertEqual(2, len(ctx.log))
        for entry in ctx.log:
            self.assertIn(entry["tool"], registry.names())
            self.assertIn(entry["status"], (STATUS_OK, STATUS_EMPTY, STATUS_BLOCKED,
                                            STATUS_FAILED, "cached"))
            self.assertGreaterEqual(entry["ms"], 0.0)
            self.assertIsInstance(entry["evidence"], list)
        # the trace mirrors the registry log (timing + evidence included)
        tool_entries = [e for e in runtime.trace if e["tool"] is not None]
        self.assertEqual([e["ms"] for e in ctx.log], [e["ms"] for e in tool_entries])
        self.assertEqual(["observe", "query_memory"], [e["tool"] for e in tool_entries])
        self.assertEqual(3, result["turns"])  # turn 3 stops on the exhausted budget
        # results are fed back into the model context, never bare prose
        for entry in tool_entries:
            self.assertTrue(entry["result_prompt"].startswith("["))
            self.assertIn(f"[{entry['status']}]", entry["result_prompt"])


class EpisodeAcceptanceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.out_dir = self.tmp / "out"
        self.world = _make_world(self.tmp)

    def test_acceptance_play_runs_the_whole_chain(self):
        # the acceptance play is written in the Phase-1 vocabulary (intent tools
        # + ground_target/approach_candidate), so it runs on the module stack;
        # the fallback stack implements none of the movement tools
        args = _runner_args(self.out_dir)
        summary = RUNNER.run_episode(self.world, _scripted_vlm(), args, self.out_dir,
                                     episode_id="ep_accept")
        self.assertEqual("finish", summary["finish_reason"])  # module runtime
        self.assertTrue(summary["success"], summary)
        self.assertEqual("modules", summary["stack_backend"])

        entries = _read_trace(self.out_dir)
        expected = [entry["tool"] for entry in RUNNER.ACCEPTANCE_TOOL_SCRIPT]
        self.assertEqual(expected, [entry["tool"] for entry in entries])
        by_tool = {entry["tool"]: entry for entry in entries}
        first_observe = entries[0]

        # result feedback: later calls consume ids produced by earlier ones
        self.assertEqual(first_observe["data"]["observation_id"],
                         by_tool["inspect_region"]["args"]["observation_id"])
        self.assertEqual(by_tool["propose_viewpoints"]["data"]["viewpoint_id"],
                         by_tool["go_to_viewpoint"]["args"]["viewpoint_id"])
        self.assertEqual(by_tool["ground_target"]["data"]["candidates"][0]["id"],
                         by_tool["approach_candidate"]["args"]["candidate_id"])
        self.assertTrue(by_tool["finish"]["args"]["reason"])

        for entry in entries:
            self.assertGreaterEqual(entry["ms"], 0.0)
            self.assertTrue(entry["result_prompt"].startswith(f"[{entry['status']}]"))
            for evidence in entry["evidence"]:
                self.assertIn(evidence["kind"], RUNNER.EVIDENCE_KINDS)
                self.assertTrue(str(evidence["ref"]))
                if evidence["kind"] in ("image", "map") and "/" in str(evidence["ref"]):
                    # a path-shaped ref is a frame of this episode on disk; a
                    # logical ref (``bev_latest``) is resolved by the runtime
                    self.assertTrue((self.out_dir / evidence["ref"]).exists(),
                                    evidence["ref"])
        for entry in entries:  # every observation carries its capture pose
            if entry["tool"] == "observe":
                self.assertIn("pose", entry["data"])
        for tool in ("observe", "inspect_region", "verify_goal", "propose_viewpoints"):
            self.assertTrue(by_tool[tool]["evidence"], tool)
        # the movement legs are intent tools and report the progress schema
        self.assertEqual(STATUS_OK, by_tool["go_to_viewpoint"]["status"])
        self.assertIn(by_tool["go_to_viewpoint"]["data"]["outcome"],
                      ("arrived", "partial", "timeout"))
        self.assertGreaterEqual(by_tool["go_to_viewpoint"]["data"]["progress_m"], 0.0)
        self.assertTrue(str(by_tool["go_to_viewpoint"]["data"]["task_id"]).startswith("nav_"))
        self.assertEqual(STATUS_OK, by_tool["approach_candidate"]["status"])
        self.assertEqual("approach_candidate",
                         by_tool["approach_candidate"]["data"]["intent"])
        self.assertGreater(by_tool["approach_candidate"]["data"]["progress_m"], 0.0)
        # the grounded candidate carries a world position: that is what the
        # approach flew to (the mock's grounding is the oracle form; the range is
        # the frame's depth where the render has one, the model's estimate else)
        candidate = by_tool["ground_target"]["data"]["candidates"][0]
        self.assertIn(candidate["position_source"], ("depth", "model_distance"))
        self.assertLess(
            float(np.linalg.norm(np.asarray(candidate["position_flu"]) - TARGET_FLU)), 0.5
        )
        # query_memory after losing the target: hits the observation log and
        # cites the recalled evidence (the memory module's own refs)
        self.assertEqual(STATUS_OK, by_tool["query_memory"]["status"])
        self.assertGreaterEqual(by_tool["query_memory"]["data"]["n_hits"], 1)
        recalled_kinds = {ev["kind"] for ev in by_tool["query_memory"]["evidence"]}
        self.assertTrue(recalled_kinds & {"observation", "image", "memory"},
                        recalled_kinds)

        # summary.json: per-tool success / usage / new evidence / timing
        tools = summary["tools"]
        for tool in ("observe", "inspect_region", "propose_viewpoints",
                     "go_to_viewpoint", "query_memory", "ground_target",
                     "approach_candidate", "verify_goal"):
            self.assertIn(tool, tools)
            self.assertGreaterEqual(tools[tool]["calls"], 1, tool)
            self.assertTrue(tools[tool]["used"], tool)
            self.assertGreaterEqual(tools[tool]["time_ms_mean"], 0.0, tool)
        self.assertNotIn("navigate", tools)  # deprecated: the play never calls it
        self.assertEqual(2, tools["observe"]["calls"])
        self.assertEqual(2, tools["observe"]["ok"])
        self.assertTrue(tools["observe"]["consumed"])
        self.assertTrue(tools["propose_viewpoints"]["consumed"])
        self.assertTrue(tools["verify_goal"]["new_evidence"] >= 1)
        self.assertGreaterEqual(summary["evidence_kinds"].get("image", 0), 1)
        self.assertEqual(0, summary["parse_failures"])
        self.assertEqual(9, summary["tool_calls"])
        # the finish call is a registered tool since P4 (it judges the claim on
        # the evidence pack), so it is executed, logged and charged like any
        # other call: nine tool turns + the finish = ten budget units
        self.assertEqual(40 - 9 - 1, summary["tool_budget"]["left"])
        self.assertTrue((self.out_dir / "summary.json").exists())
        # the memory module materializes the pixels a record still carries (the
        # crop of the inspection among them), not one file per call
        frames = sorted(path.name for path in (self.out_dir / "frames").glob("*.png"))
        self.assertTrue(frames, "no frame evidence written")
        self.assertIn(f"{by_tool['inspect_region']['data']['crop_image_ref']}.png", frames)

    def test_budget_exhaustion_and_unknown_tool_are_structured(self):
        # the fallback loop is the one that reports an unparseable reply as a
        # parse failure (the module runtime re-prompts inside the turn), so this
        # script runs against the runner's own stack on purpose
        real_loader = RUNNER.load_m1_modules
        RUNNER.load_m1_modules = lambda: ({key: None for key in _M1_MODULE_KEYS}, {})
        self.addCleanup(setattr, RUNNER, "load_m1_modules", real_loader)
        script = [
            {"prose": "let me think about this"},
            {"tool": "observe", "args": {"mode": "forward"}},
            {"tool": "no_such_tool", "args": {}},
            {"tool": "observe", "args": {"mode": "forward"}},
            {"tool": "observe", "args": {"mode": "forward"}},
        ]
        args = _runner_args(self.out_dir, max_turns=10, tool_budget=2)
        summary = RUNNER.run_episode(self.world, _scripted_vlm(script), args,
                                     self.out_dir, episode_id="ep_budget")
        self.assertEqual("budget_exhausted", summary["finish_reason"])
        self.assertEqual(1, summary["parse_failures"])
        self.assertEqual(0, summary["tool_budget"]["left"])
        self.assertEqual(2, summary["tools"]["observe"]["calls"])
        self.assertEqual(1, summary["tools"]["no_such_tool"]["failed"])

        entries = _read_trace(self.out_dir)
        failed = next(e for e in entries if e["tool"] == "no_such_tool")
        self.assertEqual(STATUS_FAILED, failed["status"])
        self.assertIn("unknown_tool", failed["data"]["error"])
        parse_failed = [e for e in entries if e["parse_failure"]]
        self.assertEqual(1, len(parse_failed))
        self.assertIsNone(parse_failed[0]["tool"])


class ModuleAdapterTest(unittest.TestCase):
    """The runner must prefer the parallel M1 modules over the fallback stack."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.out_dir = self.tmp / "out"
        self.world = _make_world(self.tmp)
        self._injected = []
        self._install_fakes()
        self.addCleanup(self._remove_fakes)

    def tearDown(self):
        self._remove_fakes()

    def _remove_fakes(self):
        for name in self._injected:
            sys.modules.pop(name, None)
        self._injected = []

    def _install_fakes(self):
        prefix = "vlm_nav."

        perception = types.ModuleType(prefix + "tool_impl_perception")

        def register_tools(registry):
            registry.register(
                ToolSpec(
                    name="fake_observe",
                    description="module perception stub",
                    input_schema={"type": "object", "properties": {}, "required": []},
                    handler=lambda ctx, args: RUNNER.ToolResult(
                        STATUS_OK, {"observation_id": "obs_fake"},
                        [EvidenceRef("observation", "obs_fake", "stub")],
                    ),
                )
            )

        perception.register_tools = register_tools

        navigation = types.ModuleType(prefix + "tool_impl_navigation")

        def register_navigation_tools(registry):
            registry.register(
                ToolSpec(
                    name="fake_navigate",
                    description="module navigation stub",
                    input_schema={"type": "object", "properties": {}, "required": []},
                    handler=lambda ctx, args: RUNNER.ToolResult(
                        STATUS_OK, {"task_id": "nav_fake"},
                        [EvidenceRef("trajectory", "nav_fake", "stub")],
                    ),
                    side_effects=True,
                )
            )

        navigation.register_navigation_tools = register_navigation_tools

        memory_module = types.ModuleType(prefix + "memory")

        class ObservationLog:
            """Stub with different field names, to exercise the adapter aliases."""

            def __init__(self, world=None, out_dir=None, episode_id=None):
                self.records = []

            def add_observation(self, obs_id, kind, image="", note="", pose=None,
                                extra=None):
                record = {"obs_id": obs_id, "kind": kind, "image": image,
                          "note": note, "pose": pose or {}, "data": extra or {}}
                self.records.append(record)
                return record

            def search(self, query, limit=3):
                needle = (query or "").lower()
                hits = [r for r in self.records if needle in json.dumps(r).lower()]
                return list(reversed(hits[-limit:]))

            def stats(self):
                return {"records": len(self.records)}

        memory_module.ObservationLog = ObservationLog

        runtime_module = types.ModuleType(prefix + "runtime")

        class AgentRuntime:
            """Stub runtime that exercises registry + memory through ToolContext."""

            def __init__(self, world=None, vlm=None, registry=None, memory=None,
                         instruction="", out_dir=None, trace_path=None, max_turns=20,
                         tool_budget=40, context_frames=2, success_dist_m=2.0,
                         tool_context=None, tools_source=None):
                self.registry = registry
                self.ctx = tool_context
                self.trace_path = Path(trace_path) if trace_path else None
                self.world = world

            def run(self):
                entries = []
                for turn, (name, args) in enumerate(
                    [
                        ("fake_observe", {}),
                        ("observe", {"mode": "forward"}),
                        ("query_memory", {"query": "target"}),
                    ],
                    start=1,
                ):
                    result = self.registry.call(name, args, self.ctx)
                    log = self.ctx.log[-1]
                    entries.append(
                        {
                            "turn": turn,
                            "tool": name,
                            "args": args,
                            "status": result.status,
                            "data": result.data,
                            "evidence": [e.to_dict() for e in result.evidence],
                            "result_prompt": result.to_prompt(),
                            "model_reply": json.dumps({"tool": name, "args": args}),
                            "ms": log["ms"],
                            "parse_failure": False,
                        }
                    )
                entries.append({"turn": 4, "tool": "finish",
                                "args": {"reason": "module runtime done"},
                                "status": STATUS_OK, "data": {"reason": "module runtime done"},
                                "evidence": [], "result_prompt": "", "ms": 0.0,
                                "parse_failure": False})
                return {"finish_reason": "vlm_finish", "turns": 4, "trace": entries,
                        "success": True}

        runtime_module.AgentRuntime = AgentRuntime

        for module in (perception, navigation, memory_module, runtime_module):
            sys.modules[module.__name__] = module
            self._injected.append(module.__name__)

    def test_adapters_pick_up_injected_modules(self):
        modules, errors = RUNNER.load_m1_modules()
        self.assertIsNotNone(modules["perception"])
        self.assertIsNotNone(modules["memory"])
        self.assertIsNotNone(modules["runtime"])
        self.assertIsNotNone(modules["navigation"])
        # only import failures of *other* modules may be reported here
        self.assertNotIn("perception", errors)
        self.assertNotIn("runtime", errors)

        registry, backend, tools_source = RUNNER.build_registry(modules)
        self.assertEqual("modules", backend)
        self.assertIn("fake_observe", registry.names())
        self.assertIn("fake_navigate", registry.names())
        self.assertTrue(set(RUNNER.M1_TOOL_NAMES).issubset(set(registry.names())))
        self.assertEqual("modules", tools_source["fake_observe"])
        self.assertEqual("fallback", tools_source["observe"])

        memory, memory_backend, _ = RUNNER.build_memory(modules, self.world,
                                                        self.out_dir, "ep_fake")
        self.assertEqual("modules", memory_backend)
        self.assertIsInstance(memory, sys.modules["vlm_nav.memory"].ObservationLog)

    def test_run_episode_prefers_module_runtime(self):
        args = _runner_args(self.out_dir, tool_budget=10)
        summary = RUNNER.run_episode(self.world, _scripted_vlm([]), args, self.out_dir,
                                     episode_id="ep_fake")
        self.assertEqual("modules", summary["stack_backend"])
        self.assertEqual("modules", summary["stack_detail"]["runtime"])
        self.assertEqual("modules", summary["stack_detail"]["memory"])
        self.assertIsNone(summary["stack_detail"]["runtime_error"])
        self.assertEqual("vlm_finish", summary["finish_reason"])
        # since P4 a runtime verdict is not enough on its own: this stub episode
        # leaves no evidence chain (three read-only calls and a hand-written
        # finish entry), so its reported success is not counted - the summary
        # says how close it got and what the trace was missing
        self.assertTrue(summary["success_computed"])
        self.assertIs(summary["success_runtime"], True)
        self.assertFalse(summary["success"])
        self.assertFalse(summary["success_evidence"])
        self.assertTrue(summary["completion"]["missing"])
        self.assertEqual(1, summary["tools"]["fake_observe"]["calls"])
        self.assertEqual(1, summary["tools"]["observe"]["calls"])
        self.assertEqual(1, summary["tools"]["query_memory"]["ok"])
        self.assertEqual(1, summary["memory"]["records"])  # adapter wrote + read memory
        # stub runtime shares the runner's ToolContext: 3 calls were consumed
        self.assertEqual(7, summary["tool_budget"]["left"])
        entries = _read_trace(self.out_dir)
        self.assertEqual(["fake_observe", "observe", "query_memory", "finish"],
                         [entry["tool"] for entry in entries])


class CliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.out_dir = self.tmp / "out"
        self.ply = self.tmp / "scene.ply"
        _make_world(self.tmp)  # writes self.tmp/scene.ply

    def test_main_cli_mock_end_to_end(self):
        summary = RUNNER.main(
            [
                "--world", "cpu",
                "--ply", str(self.ply),
                "--target-name", "target",
                "--target-pos", "2.0", "0.5", "0.0",
                "--target-radius", "0.2",
                "--start-pos", "0", "0", "0.2",
                "--instruction", "Find the target and fly to it.",
                "--backend", "mock",
                "--render-size", "64", "64",
                "--max-turns", "20",
                "--tool-budget", "16",
                "--success-dist", "2.5",
                "--seed", "3",
                "--out", str(self.out_dir),
            ]
        )
        self.assertEqual("finish", summary["finish_reason"])
        self.assertTrue(summary["success"], summary)
        self.assertEqual(9, summary["tool_calls"])
        for name in ("summary.json", "tool_trace.jsonl", "run_config.json"):
            self.assertTrue((self.out_dir / name).exists(), name)
        # the play is written in the Phase-1 vocabulary: ten turns, the last
        # one the finish call
        self.assertEqual(10, len(_read_trace(self.out_dir)))
        on_disk = json.loads((self.out_dir / "summary.json").read_text())
        self.assertEqual(summary["finish_reason"], on_disk["finish_reason"])
        self.assertEqual("modules", on_disk["stack_backend"])
        self.assertIn("observe", on_disk["tools"])
        self.assertNotIn("navigate", on_disk["tools"])


class ToolsetFilterTest(unittest.TestCase):
    """``--toolset basic|memory|full`` restricts the catalogue the model sees."""

    def test_toolset_vocabulary(self):
        self.assertEqual(("basic", "memory", "full"), RUNNER.TOOLSET_CHOICES)
        self.assertEqual("full", RUNNER.DEFAULT_TOOLSET)
        # the movement vocabulary is one tool per intent; the deprecated
        # ``navigate`` is in no set (the answer only names its replacement), and
        # ``batch_read`` (several read-only checks in one turn) rides every set
        # because it is a dispatch affordance, not extra evidence
        self.assertEqual(
            {"observe", "batch_read", "propose_viewpoints", "approach_candidate",
             "go_to_viewpoint", "return_to_observation", "verify_goal", "finish"},
            set(RUNNER.BASIC_TOOL_NAMES),
        )
        self.assertNotIn("navigate", RUNNER.FULL_TOOL_NAMES)
        self.assertEqual(
            set(RUNNER.BASIC_TOOL_NAMES)
            | {"query_memory", "inspect_region", "ground_target", "compare_candidates"},
            set(RUNNER.MEMORY_TOOL_NAMES),
        )
        for name in ("propose_viewpoints", "check_path", *RUNNER.TASK_TOOL_NAMES):
            self.assertIn(name, RUNNER.FULL_TOOL_NAMES)
        for name in RUNNER.FULL_TOOL_NAMES:
            self.assertTrue(RUNNER.toolset_allows("full", name), name)
        for name in ("query_memory", "inspect_region", "ground_target",
                     "compare_candidates", "check_path"):
            self.assertFalse(RUNNER.toolset_allows("basic", name), name)
        for name in ("check_path", *RUNNER.TASK_TOOL_NAMES):
            self.assertFalse(RUNNER.toolset_allows("memory", name), name)
        for name in ("observe", "propose_viewpoints", "approach_candidate",
                     "go_to_viewpoint", "return_to_observation", "verify_goal",
                     "finish"):
            self.assertTrue(RUNNER.toolset_allows("basic", name), name)
            self.assertTrue(RUNNER.toolset_allows("memory", name), name)
        # both sets move the same way: the difference is retrieval, not motion
        self.assertEqual(
            set(RUNNER.MEMORY_TOOL_NAMES) - set(RUNNER.BASIC_TOOL_NAMES),
            {"query_memory", "inspect_region", "ground_target", "compare_candidates"},
        )
        # an unknown tool set name is not a filter (full semantics)
        self.assertTrue(RUNNER.toolset_allows("nonsense", "propose_viewpoints"))

    def test_report_lists_what_the_toolset_could_not_provide(self):
        report = RUNNER.toolset_report("full", ["observe"])
        self.assertEqual("full", report["name"])
        self.assertFalse(report["restricted"])
        self.assertEqual("all", report["allowed"])
        expected = [name for name in RUNNER.FULL_TOOL_NAMES
                    if name not in ("finish", "observe")]
        self.assertEqual(expected, report["missing"])
        self.assertTrue(any("not registered" in note for note in report["notes"]))
        report = RUNNER.toolset_report("memory", RUNNER.MEMORY_TOOL_NAMES)
        self.assertFalse(report["missing"])
        self.assertEqual(sorted(RUNNER.MEMORY_TOOL_NAMES), report["registered"])
        self.assertTrue(report["restricted"])

    def test_fallback_registration_respects_the_toolset(self):
        # the fallback stack implements none of the intent tools: for basic /
        # memory it fills what it does implement, and the missing movement
        # vocabulary is reported in toolset_detail instead of being faked
        registry = ToolRegistry()
        added = RUNNER.register_fallback_tools(registry, toolset="basic")
        self.assertEqual({"observe", "propose_viewpoints", "verify_goal"}, added)
        self.assertRaises(KeyError, registry.get, "query_memory")
        self.assertRaises(KeyError, registry.get, "navigate")

        registry = ToolRegistry()
        added = RUNNER.register_fallback_tools(registry, toolset="memory")
        self.assertEqual({"observe", "propose_viewpoints", "verify_goal",
                          "query_memory", "inspect_region"}, added)
        self.assertRaises(KeyError, registry.get, "go_to_viewpoint")

        # full keeps every fallback tool, and stays composable
        registry = ToolRegistry()
        added = RUNNER.register_fallback_tools(registry, toolset="full")
        self.assertEqual(set(RUNNER.M1_TOOL_NAMES), added)

    def test_build_registry_drops_tools_outside_the_toolset(self):
        modules, _ = RUNNER.load_m1_modules()
        for toolset in ("basic", "memory"):
            registry, backend, tools_source = RUNNER.build_registry(modules, toolset=toolset)
            names = set(registry.names())
            for name in names:
                self.assertTrue(RUNNER.toolset_allows(toolset, name), (toolset, name))
            self.assertTrue(
                {"observe", "propose_viewpoints", "go_to_viewpoint",
                 "return_to_observation", "verify_goal"}.issubset(names), toolset
            )
            self.assertNotIn("navigate", names, toolset)  # deprecated stub
            self.assertNotIn("check_path", names, toolset)
            self.assertNotIn("update_plan", names, toolset)
            self.assertIn("tools_source", {"tools_source": True})  # returned for the summary
            self.assertIn(backend, ("modules", "fallback"))
            self.assertEqual(set(tools_source), names)
        registry, _, _ = RUNNER.build_registry(modules, toolset="memory")
        self.assertIn("query_memory", registry.names())
        self.assertIn("inspect_region", registry.names())
        registry, _, _ = RUNNER.build_registry(modules, toolset="full")
        for name in RUNNER.M1_TOOL_NAMES:
            self.assertIn(name, registry.names(), name)

    def test_cli_parses_the_toolset(self):
        args = _runner_args(Path("out"))
        self.assertEqual("full", args.toolset)
        args = _runner_args(Path("out"), toolset="basic")
        self.assertEqual("basic", args.toolset)
        with self.assertRaises(SystemExit):
            RUNNER.parse_args(["--instruction", "x", "--out", "y", "--toolset", "bogus"])


class SearchSkillBudgetTest(unittest.TestCase):
    """``search_target`` is one decision: it cannot swallow the episode budget.

    The skill's inner calls run on the routine's own allowance, so an episode
    that searches still has budget to act on what the search returned - while the
    full cost (inner tool calls and the model calls they made) stays reported.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.out_dir = self.tmp / "out"
        self.world = _make_world(self.tmp)

    def test_a_search_leaves_the_episode_budget_alone(self):
        script = [
            {"tool": "search_target", "args": {"instruction": "$target_name"}},
            {"tool": "observe", "args": {"mode": "forward"}},
            {"tool": "verify_goal", "args": {}},
            {"tool": "finish", "args": {"reason": "searched and verified"}},
        ]
        args = _runner_args(self.out_dir, tool_budget=20, max_turns=8)
        summary = RUNNER.run_episode(self.world, _scripted_vlm(script), args,
                                     self.out_dir, episode_id="ep_search")

        entries = _read_trace(self.out_dir)
        search = next(e for e in entries if e["tool"] == "search_target")
        cost = search["data"]["cost"]
        self.assertEqual(cost["outer_tool_calls"], 1)
        self.assertGreaterEqual(cost["inner_tool_calls"], 2)
        self.assertEqual(search["data"]["inner_tool_calls"], cost["inner_tool_calls"])
        # the routine really ran tools and, if it grounded, made model calls
        self.assertGreaterEqual(len(search["data"]["steps"]), 2)
        self.assertEqual(search["data"]["vlm_calls"], cost["vlm_calls"])

        # the episode went on: the search cost it one decision, not its budget
        self.assertEqual(3, summary["tool_calls"])
        self.assertGreaterEqual(summary["tool_budget"]["left"], 16)
        self.assertNotEqual("budget_exhausted", summary["finish_reason"])

        # the summary aggregates the routine cost for the report
        routines = summary["routine_cost"]
        self.assertEqual(routines["outer_tool_calls"], 1)
        self.assertEqual(routines["inner_tool_calls"], cost["inner_tool_calls"])
        self.assertEqual(routines["vlm_calls"], cost["vlm_calls"])

    def test_the_budget_metrics_of_an_episode_without_a_search_stay_zero(self):
        script = [
            {"tool": "observe", "args": {"mode": "forward"}},
            {"tool": "finish", "args": {"reason": "done"}},
        ]
        args = _runner_args(self.out_dir)
        summary = RUNNER.run_episode(self.world, _scripted_vlm(script), args,
                                     self.out_dir, episode_id="ep_plain")
        self.assertEqual(summary["routine_cost"],
                         {"outer_tool_calls": 0, "inner_tool_calls": 0, "vlm_calls": 0})


class MechanismSwitchTest(unittest.TestCase):
    """``--recovery``/``--state-block``/``--search-skill`` (+ the memory layer
    switches of :class:`MemoryLayerSwitchTest`): the ablation switches.

    The decision-loop comparison drives these flags to build its arms, so each
    one must reach the place it owns (the runtime's stall handling, the prompt,
    the registry) and be reported truthfully in ``run_config.json`` and
    ``summary.json`` - a report that claims a mechanism ran while the runner
    ignored the flag would be worthless.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.world = _make_world(self.tmp)

    def test_cli_defaults_keep_every_mechanism_on(self):
        args = _runner_args(self.tmp / "out")
        self.assertEqual("on", args.recovery)
        self.assertEqual("on", args.state_block)
        self.assertEqual("on", args.search_skill)
        self.assertTrue(RUNNER.switch_enabled(args.recovery))
        self.assertFalse(RUNNER.switch_enabled("off"))
        self.assertEqual(RUNNER.SEARCH_SKILL_TOOL, "search_target")

    def test_cli_rejects_an_unknown_switch_value(self):
        for switch in ("--recovery", "--state-block", "--search-skill",
                       "--memory-spatial", "--memory-episodic"):
            with self.subTest(switch=switch), self.assertRaises(SystemExit):
                RUNNER.parse_args(["--instruction", "x", "--out", "y", switch, "maybe"])

    def test_search_skill_off_keeps_search_target_out_of_the_registry(self):
        modules, _ = RUNNER.load_m1_modules()
        registry, _, _ = RUNNER.build_registry(modules, toolset="full")
        self.assertIn(RUNNER.SEARCH_SKILL_TOOL, registry.names())

        registry, _, _ = RUNNER.build_registry(
            modules, toolset="full", exclude={RUNNER.SEARCH_SKILL_TOOL}
        )
        self.assertNotIn(RUNNER.SEARCH_SKILL_TOOL, registry.names())
        for name in ("observe", "ground_target", "propose_viewpoints", "verify_goal"):
            self.assertIn(name, registry.names())  # only the skill is dropped

    def test_the_switches_are_recorded_and_really_applied(self):
        script = [
            {"tool": "search_target", "args": {"instruction": "target"}},
            {"tool": "observe", "args": {"mode": "forward"}},
            {"tool": "verify_goal", "args": {}},
            {"tool": "finish", "args": {"reason": "done"}},
        ]
        off_dir, on_dir = self.tmp / "off", self.tmp / "on"
        args = _runner_args(off_dir, recovery="off", state_block="off", search_skill="off")
        summary = RUNNER.run_episode(self.world, _scripted_vlm(script), args, off_dir,
                                     episode_id="ep_off")

        self.assertEqual(
            {"recovery": "off", "state_block": "off", "search_skill": "off",
             "spatial_memory": "on", "episodic_memory": "on"},
            summary["switches"]["requested"],
        )
        applied = summary["switches"]["applied"]
        self.assertFalse(applied["recovery"])
        self.assertFalse(applied["state_block"])
        self.assertFalse(applied["search_skill"])
        self.assertEqual("modules", applied["runtime"])
        self.assertNotIn(RUNNER.SEARCH_SKILL_TOOL, summary["toolset_detail"]["registered"])
        # the model cannot call what the registry does not carry
        entry = next(e for e in _read_trace(off_dir) if e["tool"] == RUNNER.SEARCH_SKILL_TOOL)
        self.assertEqual(STATUS_FAILED, entry["status"])
        self.assertIn("unknown_tool", json.dumps(entry["data"]))

        args = _runner_args(on_dir)
        summary = RUNNER.run_episode(self.world, _scripted_vlm(script), args, on_dir,
                                     episode_id="ep_on")
        self.assertEqual(
            {"recovery": "on", "state_block": "on", "search_skill": "on",
             "spatial_memory": "on", "episodic_memory": "on"},
            summary["switches"]["requested"],
        )
        applied = summary["switches"]["applied"]
        self.assertTrue(applied["recovery"])
        self.assertTrue(applied["state_block"])
        self.assertTrue(applied["search_skill"])
        self.assertIn(RUNNER.SEARCH_SKILL_TOOL, summary["toolset_detail"]["registered"])

    def test_run_config_records_the_switches(self):
        out_dir = self.tmp / "main"
        summary = RUNNER.main(
            [
                "--world", "cpu",
                "--ply", str(self.tmp / "scene.ply"),
                "--target-name", "target",
                "--target-pos", "2.0", "0.5", "0.0",
                "--target-radius", "0.2",
                "--start-pos", "0", "0", "0.2",
                "--instruction", "Find the target and fly to it.",
                "--backend", "mock",
                "--render-size", "64", "64",
                "--tool-budget", "16",
                "--recovery", "off",
                "--state-block", "off",
                "--search-skill", "off",
                "--out", str(out_dir),
            ]
        )
        config = json.loads((out_dir / "run_config.json").read_text())
        self.assertEqual("off", config["recovery"])
        self.assertEqual("off", config["state_block"])
        self.assertEqual("off", config["search_skill"])
        on_disk = json.loads((out_dir / "summary.json").read_text())
        self.assertEqual(summary["switches"], on_disk["switches"])
        self.assertFalse(on_disk["switches"]["applied"]["recovery"])
        self.assertFalse(on_disk["switches"]["applied"]["state_block"])
        self.assertNotIn(RUNNER.SEARCH_SKILL_TOOL, on_disk["toolset_detail"]["registered"])

    def test_recovery_off_ends_a_stalling_episode_the_loop_recovers_from_when_on(self):
        script = [{"tool": "observe", "args": {"mode": "forward"}}] * 6
        off_dir, on_dir = self.tmp / "stall_off", self.tmp / "stall_on"
        off = RUNNER.run_episode(
            self.world, _scripted_vlm(script),
            _runner_args(off_dir, recovery="off", max_turns=8), off_dir, episode_id="ep_stall_off",
        )
        on = RUNNER.run_episode(
            self.world, _scripted_vlm(script),
            _runner_args(on_dir, recovery="on", max_turns=8), on_dir, episode_id="ep_stall_on",
        )
        # the same repeating model: without recovery the loop stops at the
        # stagnation limit, with it the episode goes on
        self.assertEqual("no_new_evidence", off["finish_reason"])
        self.assertGreater(on["turns"], off["turns"])


class MemoryLayerSwitchTest(unittest.TestCase):
    """``--memory-spatial``/``--memory-episodic``: the two memory stores.

    The decision-loop ablation builds its memory arms from these flags, so each
    one must reach ``build_memory`` (the memory module's own constructor
    parameters) and be reported truthfully: a summary that claims the spatial
    layer ran while the built memory ignored the flag would make the arm
    comparison worthless.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.world = _make_world(self.tmp)

    def test_cli_defaults_keep_both_memory_stores_on(self):
        args = _runner_args(self.tmp / "out")
        self.assertEqual("on", args.memory_spatial)
        self.assertEqual("on", args.memory_episodic)
        off = _runner_args(self.tmp / "out", memory_spatial="off", memory_episodic="off")
        report = RUNNER.switch_report(off)
        self.assertEqual({"spatial_memory": "off", "episodic_memory": "off"},
                         {name: report[name] for name in RUNNER.MEMORY_SWITCHES})

    def test_build_memory_hands_the_layer_switches_to_the_memory_module(self):
        seen: dict = {}

        class Memory:
            # exactly the M1 constructor contract: the two layer parameters
            def __init__(self, spatial_memory=True, episodic_memory=True, **rest):
                self.spatial_memory = bool(spatial_memory)
                self.episodic_memory = bool(episodic_memory)
                seen.update(rest)

        module = types.ModuleType("stub_memory_module")
        module.ObservationLog = Memory
        memory, backend, _ = RUNNER.build_memory(
            {"memory": module}, self.world, self.tmp, "ep_1",
            spatial_memory=False, episodic_memory=True,
        )
        self.assertEqual("modules", backend)
        self.assertIsInstance(memory, Memory)
        self.assertFalse(memory.spatial_memory)
        self.assertTrue(memory.episodic_memory)
        applied = RUNNER.applied_switches(None, None, "modules", memory=memory)
        self.assertIs(False, applied["spatial_memory"])
        self.assertIs(True, applied["episodic_memory"])

    def test_a_memory_without_the_layer_parameters_stays_unattested(self):
        """A memory that does not implement the layer must not be reported as
        if it were switched off: the summary says "unknown", not "off"."""
        class LegacyMemory:
            def __init__(self, image_dir=None, **rest):
                self.image_dir = image_dir

        module = types.ModuleType("legacy_memory_module")
        module.ObservationLog = LegacyMemory
        memory, backend, _ = RUNNER.build_memory(
            {"memory": module}, self.world, self.tmp, "ep_2",
            spatial_memory=False, episodic_memory=False,
        )
        self.assertEqual("modules", backend)
        applied = RUNNER.applied_switches(None, None, "modules", memory=memory)
        self.assertIsNone(applied["spatial_memory"])
        self.assertIsNone(applied["episodic_memory"])
        # the fallback memory has no layer split either
        fallback = RUNNER._FallbackMemory()
        applied = RUNNER.applied_switches(None, None, "fallback", memory=fallback)
        self.assertIsNone(applied["spatial_memory"])
        self.assertIsNone(applied["episodic_memory"])

    def test_a_memory_that_reports_its_switches_through_a_method_is_read(self):
        class Memory:
            def __init__(self, spatial_memory=True, episodic_memory=True):
                self._spatial = bool(spatial_memory)
                self._episodic = bool(episodic_memory)

            def memory_switches(self):
                return {"spatial_memory": self._spatial, "episodic_memory": self._episodic}

        applied = RUNNER.applied_switches(None, None, "modules",
                                          memory=Memory(spatial_memory=True,
                                                        episodic_memory=False))
        self.assertIs(True, applied["spatial_memory"])
        self.assertIs(False, applied["episodic_memory"])

    def test_the_episode_summary_records_what_the_memory_really_got(self):
        out_dir = self.tmp / "memory_off"
        args = _runner_args(out_dir, memory_spatial="off", memory_episodic="off",
                            toolset="memory")
        script = [{"tool": "observe", "args": {"mode": "forward"}},
                  {"tool": "query_memory", "args": {"query": "target"}},
                  {"tool": "finish", "args": {"reason": "done", "success": False}}]
        summary = RUNNER.run_episode(self.world, _scripted_vlm(script), args, out_dir,
                                     episode_id="ep_mem_off")
        self.assertEqual("memory", summary["toolset"])
        requested = summary["switches"]["requested"]
        self.assertEqual("off", requested["spatial_memory"])
        self.assertEqual("off", requested["episodic_memory"])
        applied = summary["switches"]["applied"]
        for name in ("spatial_memory", "episodic_memory"):
            self.assertIn(applied[name], (False, None), applied)
        on_disk = json.loads((out_dir / "summary.json").read_text())
        self.assertEqual(summary["switches"], on_disk["switches"])
        # ... and the CLI flags land in run_config.json
        args = _runner_args(out_dir, memory_spatial="on", memory_episodic="off")
        self.assertEqual("on", args.memory_spatial)
        self.assertEqual("off", args.memory_episodic)

    def test_the_finish_contract_asks_for_an_explicit_declaration(self):
        """The runtime counts a claim only when the finish declares it, so the
        tool doc must ask for the flag instead of tolerating a bare finish."""
        doc = RUNNER.finish_contract_doc()
        self.assertIn('"success": true', doc)
        self.assertIn('"success": false', doc)
        self.assertIn(RUNNER.FINISH_CONTRACT, doc)
        # the permissive sentence of the sibling module's doc is gone: nothing in
        # the prompt still says a finish may leave the outcome undeclared
        self.assertNotIn("finish with a reason but no success flag", doc)

        modules, _ = RUNNER.load_m1_modules()
        registry, _, _ = RUNNER.build_registry(modules, toolset="full")
        self.assertIn(RUNNER.FINISH_CONTRACT, RUNNER._system_prompt("find it", registry))

        # ... and run_episode hands the same doc to the module runtime
        seen: dict = {}
        original = RUNNER.build_runtime

        def spy(modules, **kwargs):
            seen.update(kwargs)
            return original(modules, **kwargs)

        out_dir = self.tmp / "contract"
        RUNNER.build_runtime = spy
        try:
            RUNNER.run_episode(
                self.world, _scripted_vlm([{"tool": "finish",
                                            "args": {"reason": "x", "success": False}}]),
                _runner_args(out_dir), out_dir, episode_id="ep_contract",
            )
        finally:
            RUNNER.build_runtime = original
        self.assertIn(RUNNER.FINISH_CONTRACT, str(seen.get("tool_call_doc")))


class ToolsetEpisodeTest(unittest.TestCase):
    """The tool set reaches the loop: the model cannot call what it cannot see."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.out_dir = self.tmp / "out"
        self.world = _make_world(self.tmp)

    def test_basic_toolset_blocks_query_memory_end_to_end(self):
        script = [
            {"tool": "observe", "args": {"mode": "forward"}},
            {"tool": "query_memory", "args": {"query": "target"}},
            {"tool": "verify_goal", "args": {}},
            {"tool": "finish", "args": {"reason": "done"}},
        ]
        args = _runner_args(self.out_dir, toolset="basic")
        summary = RUNNER.run_episode(self.world, _scripted_vlm(script), args,
                                     self.out_dir, episode_id="ep_basic")
        self.assertEqual("basic", summary["toolset"])
        # the module stack provides the whole basic vocabulary: the three intent
        # tools (the deprecated navigate is not in the set), observe, verify_goal,
        # and since P4 the dispatch verb ``batch_read`` plus the evidence-backed
        # ``finish``
        self.assertEqual(["approach_candidate", "batch_read", "finish", "go_to_viewpoint",
                          "observe", "propose_viewpoints", "return_to_observation",
                          "verify_goal"],
                         summary["toolset_detail"]["registered"])
        self.assertTrue(summary["toolset_detail"]["restricted"])
        entries = _read_trace(self.out_dir)
        blocked = next(e for e in entries if e["tool"] == "query_memory")
        self.assertEqual(STATUS_FAILED, blocked["status"])
        # both runtimes report the missing tool: {"error": ...} / {"reason": ...}
        self.assertIn("unknown_tool", json.dumps(blocked["data"]))
        # the missing tools are reported, never silently registered
        self.assertNotIn("query_memory", RUNNER.toolset_report(
            "basic", summary["toolset_detail"]["registered"])["registered"])

    def test_memory_toolset_keeps_retrieval_but_not_exploration(self):
        script = [
            {"tool": "observe", "args": {"mode": "forward"}},
            {"tool": "query_memory", "args": {"query": "$target_name", "limit": 2}},
            {"tool": "propose_viewpoints", "args": {"n": 2}},
            {"tool": "check_path", "args": {"viewpoint_id": "$viewpoint_id"}},
            {"tool": "verify_goal", "args": {}},
            {"tool": "finish", "args": {"reason": "memory chain done"}},
        ]
        args = _runner_args(self.out_dir, toolset="memory")
        summary = RUNNER.run_episode(self.world, _scripted_vlm(script), args,
                                     self.out_dir, episode_id="ep_memory")
        self.assertEqual("memory", summary["toolset"])
        self.assertIn("query_memory", summary["toolset_detail"]["registered"])
        self.assertIn("propose_viewpoints", summary["toolset_detail"]["registered"])
        self.assertNotIn("check_path", summary["toolset_detail"]["registered"])
        self.assertGreaterEqual(summary["tools"]["query_memory"]["ok"], 1)
        self.assertEqual(1, summary["tools"]["propose_viewpoints"]["ok"])
        entries = _read_trace(self.out_dir)
        self.assertEqual(STATUS_OK, next(e for e in entries
                                         if e["tool"] == "query_memory")["status"])
        blocked = next(e for e in entries if e["tool"] == "check_path")
        self.assertEqual(STATUS_FAILED, blocked["status"])  # outside the set


M2_PATH = ROOT / "scripts" / "run_vlm_nav_m2_memory.py"


def _load_m2_script():
    name = "diffaero_script_run_vlm_nav_m2_memory"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, M2_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M2 = _load_m2_script()


def _m2_args(out_dir: Path, **overrides) -> list[str]:
    argv = [
        "--backend", "mock",
        "--render-size", "32", "32",
        "--scan-views", "4",
        "--tool-budget", "12",
        "--max-turns", "10",
        "--out", str(out_dir),
    ]
    for key, value in overrides.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, (list, tuple)):
            argv.extend([flag, *[str(item) for item in value]])
        elif isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    return argv


def _read_ply_rows(path: Path) -> np.ndarray:
    """PLY vertices as a (N, 14) array in file order (x,y,z, sh, opacity, scale, rot)."""
    with open(path, "rb") as handle:
        names = []
        count = 0
        while True:
            line = handle.readline().decode("ascii")
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property"):
                names.append(line.split()[-1])
            elif line.startswith("end_header"):
                break
        return np.frombuffer(handle.read(), dtype="<f4").reshape(count, len(names))


def _red_pixels(frame) -> np.ndarray:
    """Boolean mask of target-red pixels (the target cluster is SH0 red)."""
    arr = (frame.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(int)
    return (arr[:, :, 0] - arr[:, :, 1]) > 40


class _PromptRecorder:
    """Backend wrapper that keeps the first prompt it is asked to answer."""

    def __init__(self, inner):
        self.inner = inner
        self.first: list[str] | None = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def decide_with_tools(self, *args, **kwargs):
        self._record(args, kwargs)
        return self.inner.decide_with_tools(*args, **kwargs)

    def decide(self, *args, **kwargs):
        self._record(args, kwargs)
        return self.inner.decide(*args, **kwargs)

    def _record(self, args, kwargs) -> None:
        if self.first is not None:
            return
        payload = list(args) + list(kwargs.values())
        self.first = [value if isinstance(value, str) else repr(value)
                      for value in payload]

    def user_prompt(self) -> str:
        assert self.first is not None, "the backend was never asked"
        return next(value for value in self.first if value.startswith("Episode "))


class VerifyGateTest(unittest.TestCase):
    """The episode hands ``verify_goal`` the gate it is evaluated with.

    ``--success-dist`` is the standoff from the target *surface*, so the runner's
    completion gate is ``--success-dist + target radius`` (the ``success_dist_m``
    of ``summary.json``). That value is what a completion check must use: the
    decision-loop v3 run claimed four goals at a 1.32 m standoff while the
    book_near episodes were evaluated against 1.2 m, because ``verify_goal``
    fell back to its own ``max(1.5, radius + 0.5)`` default.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.world = _make_world(self.tmp)  # target (2.0, 0.5, 0.0), r = 0.2

    def _standoff(self, distance_m: float) -> None:
        """Put the drone ``distance_m`` from the target (target along +x)."""
        self.world.state.position_flu = np.asarray([2.0 - distance_m, 0.5, 0.0])

    def _run(self, out_name: str, success_dist: float) -> tuple[dict, dict]:
        out_dir = self.tmp / out_name
        script = [
            {"tool": "verify_goal", "args": {"reason": "confirm"}},
            {"tool": "finish", "args": {"reason": "verified", "success": True}},
        ]
        args = _runner_args(out_dir, success_dist=success_dist,
                            max_turns=4, tool_budget=8)
        summary = RUNNER.run_episode(self.world, _scripted_vlm(script), args, out_dir)
        verify = next(entry for entry in _read_trace(out_dir)
                      if entry["tool"] == "verify_goal")
        return summary, verify

    def test_a_standoff_outside_the_gate_is_not_verified(self):
        # 1.0 m from the surface + 0.2 m radius = a 1.2 m gate; 1.32 m is outside
        # it and inside the 1.5 m default the tool used to fall back to
        self._standoff(1.32)
        summary, verify = self._run("outside", success_dist=1.0)

        self.assertIs(verify["data"]["verified"], False)
        unmet = next(item for item in verify["data"]["unmet"]
                     if item["check"] == "distance_to_target_m")
        self.assertAlmostEqual(1.2, unmet["max_m"], places=3)
        self.assertAlmostEqual(1.2, verify["data"]["checks"]["distance_limit_m"], places=3)
        # the model asked to finish as success; the geometry vetoes it
        self.assertEqual(1.2, summary["success_dist_m"])
        self.assertEqual(1.2, summary["done_dist_m"])
        self.assertFalse(summary["success"])

    def test_a_standoff_inside_the_gate_still_verifies(self):
        self._standoff(1.15)
        summary, verify = self._run("inside", success_dist=1.0)

        self.assertIs(verify["data"]["verified"], True)
        self.assertEqual([], verify["data"]["unmet"])
        self.assertAlmostEqual(1.15, verify["data"]["checks"]["distance_to_target_m"],
                               places=3)
        self.assertEqual(1.2, summary["success_dist_m"])
        self.assertTrue(summary["success"])

    def test_the_gate_is_the_same_formula_the_runner_judges_with(self):
        args = _runner_args(self.tmp / "out", success_dist=0.9)
        self.assertAlmostEqual(1.1, RUNNER.success_gate_m(self.world, args), places=6)
        ctx = RUNNER.make_tool_context(self.world, RUNNER._FallbackMemory(), args,
                                       self.tmp / "out", "ep_gate")
        self.assertAlmostEqual(1.1, ctx.extra["done_dist"], places=6)
        self.assertAlmostEqual(0.9, ctx.extra["success_dist_m"], places=6)


class PromptIdentityTest(unittest.TestCase):
    """The model-visible episode id carries no arm/mode name.

    E2's ``dump_first_prompt`` showed the three memory arms' first prompts
    differing *only* in the episode id (``m2_memory_000_00`` vs
    ``m2_basic_000_00``): under greedy decoding that constant string is enough
    to move the first decision, so an arm difference would not be attributable
    to the mechanism. The label stays in the artifact directory, ``summary.json``
    and ``run_config.json``; the prompt gets the neutral ``ep_<seed>`` identity.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _first_prompt(self, label: str) -> tuple[str, dict]:
        out_dir = self.tmp / label
        world = _make_world(self.tmp)
        script = [
            {"tool": "observe", "args": {"mode": "forward"}},
            {"tool": "finish", "args": {"reason": "looked around"}},
        ]
        recorder = _PromptRecorder(_scripted_vlm(script))
        args = _runner_args(out_dir, max_turns=4, tool_budget=8)
        summary = RUNNER.run_episode(world, recorder, args, out_dir, episode_id=label)
        return recorder.user_prompt(), summary

    def test_two_arms_differing_only_in_the_label_see_the_same_prompt(self):
        memory_prompt, memory_summary = self._first_prompt("m2_memory_000_00")
        basic_prompt, basic_summary = self._first_prompt("m2_basic_000_00")

        self.assertEqual(memory_prompt, basic_prompt)
        for leaked in ("m2_memory", "m2_basic", "Episode m2"):
            self.assertNotIn(leaked, memory_prompt)
        self.assertIn("Episode ep_000 |", memory_prompt)  # the neutral identity
        # the label is still what the artifacts say the run was
        self.assertEqual("m2_memory_000_00", memory_summary["episode_id"])
        self.assertEqual("m2_basic_000_00", basic_summary["episode_id"])
        self.assertEqual("ep_000", memory_summary["prompt_episode_id"])
        self.assertEqual("ep_000", basic_summary["prompt_episode_id"])
        self.assertEqual("m2_memory_000_00",
                         json.loads((self.tmp / "m2_memory_000_00" /
                                     "summary.json").read_text())["episode_id"])

    def test_a_caller_supplied_neutral_prompt_id_is_kept(self):
        out_dir = self.tmp / "custom"
        world = _make_world(self.tmp)
        recorder = _PromptRecorder(
            _scripted_vlm([{"tool": "finish", "args": {"reason": "done"}}])
        )
        args = _runner_args(out_dir, max_turns=2, tool_budget=4)
        summary = RUNNER.run_episode(world, recorder, args, out_dir,
                                     episode_id="m2_memory_000_00",
                                     prompt_episode_id="ep_cell_7")
        self.assertEqual("ep_cell_7", summary["prompt_episode_id"])
        self.assertIn("Episode ep_cell_7 |", recorder.user_prompt())


class M2SceneFrameTest(unittest.TestCase):
    """The synthetic M2 scene must describe the geometry the analytics describe.

    ``vlm_nav.cpu_renderer`` reads PLY vertices as absolute 3DGS coordinates and
    takes the vertex centroid as the FLU origin of the world. Writing the scene in
    plain FLU coordinates made every frame exactly black (mean = max = 0, no
    target pixel), so a real model was blind (outputs/vlm_nav/eval/m2_hf_8b/REPORT.md).
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _scene(self, **overrides):
        args = M2.parse_args(_m2_args(self.tmp / "out", render_size=[64, 64], **overrides))
        scenario = M2.build_scenario(args)
        ply, scene, origin = M2.scene_for_scenario(args, scenario, self.tmp)
        return args, scenario, ply, scene, origin

    def test_occluder_slab_fills_the_box_instead_of_collapsing(self):
        # the default 2.5 x 0.12 x 3.2 m wall: the old step max(spacing, hi-lo)
        # gave two corner splats that hid nothing in the render while the slab
        # test still reported the target occluded
        wall = M2.Box(center=(2.0, -1.5, 0.0), size=(2.5, 0.12, 3.2))
        splats = M2._box_splats(wall, spacing=0.2)
        self.assertEqual(14 * 2 * 17, len(splats))
        coords = np.asarray([xyz for xyz, *_ in splats])
        for axis, layers in enumerate((14, 2, 17)):
            values = np.unique(np.round(coords[:, axis], 6))
            self.assertEqual(layers, values.size, f"axis {axis}")
            self.assertAlmostEqual(float(values[0]), wall.lo[axis], places=6)
            self.assertAlmostEqual(float(values[-1]), wall.hi[axis], places=6)
            self.assertLessEqual(float(np.diff(values).max()), 0.2 + 1e-9)

    def test_scene_is_authored_in_the_renderers_frame(self):
        args, scenario, ply, scene, origin = self._scene()
        rows = _read_ply_rows(ply)
        # 1) the renderer puts the FLU origin at the vertex centroid: the authored
        #    scene must be centred there, otherwise it renders translated
        np.testing.assert_allclose(rows[:, :3].mean(axis=0), 0.0, atol=1e-5)
        # 2) the scenario follows the scene with the same rigid translation
        self.assertGreater(float(np.linalg.norm(origin)), 0.1)
        for cli, aligned in ((scenario.start_pos, scene.start_pos),
                             (scenario.target_pos, scene.target_pos),
                             (scenario.occluders[0].center, scene.occluders[0].center)):
            np.testing.assert_allclose(np.asarray(aligned),
                                       np.asarray(cli) - np.asarray(origin), atol=1e-9)
        self.assertEqual(scenario.occluders[0].size, scene.occluders[0].size)
        # 3) so the effective FLU geometry of the PLY *is* the scenario geometry
        centroid = rows[:, :3].mean(axis=0)
        flu = (rows[:, :3] - centroid) @ M2.M_FLU_TO_3DGS  # 3DGS -> FLU
        target_rows = flu[np.isclose(rows[:, 3], 2.0)]  # target cluster: SH red
        wall_rows = flu[np.isclose(rows[:, 3], 0.15)]
        self.assertEqual(3, len(target_rows))
        np.testing.assert_allclose(target_rows.mean(axis=0), scene.target_pos, atol=0.25)
        np.testing.assert_allclose(wall_rows.mean(axis=0), scene.occluders[0].center, atol=0.01)
        # the frame change is a translation only: distances are untouched
        self.assertAlmostEqual(
            float(np.linalg.norm(np.asarray(scene.target_pos) - np.asarray(scene.start_pos))),
            float(np.linalg.norm(np.asarray(scenario.target_pos)
                                 - np.asarray(scenario.start_pos))),
            places=9,
        )

    def test_start_frame_shows_the_target_the_analytics_describe(self):
        args, scenario, ply, scene, origin = self._scene()
        world = M2.build_world(args, scene, ply)
        frame = world.render()
        self.assertGreater(float(frame.mean()), 0.05)  # not a black frame
        self.assertTrue(M2.target_visibility(
            scene.start_pos, math.radians(scene.start_yaw_deg), scene)["visible"])
        mask = _red_pixels(frame)
        columns = np.nonzero(mask)[1]
        self.assertGreater(int(mask.sum()), 0, "no target pixel in the start frame")
        # the target must land where the analytic bearing puts it
        delta = np.asarray(scene.target_pos) - np.asarray(scene.start_pos)
        offset_deg = M2._wrap_deg(
            math.degrees(math.atan2(delta[1], delta[0])) - scene.start_yaw_deg
        )
        fx = args.render_size[0] / (2 * math.tan(math.radians(scene.hfov_deg) / 2))
        predicted = args.render_size[0] / 2 - fx * math.tan(math.radians(offset_deg))
        self.assertGreaterEqual(int(columns.min()), 0)
        self.assertLessEqual(int(columns.min()) - 2, predicted)
        self.assertGreaterEqual(int(columns.max()) + 2, predicted)

    def test_a_supplied_ply_keeps_its_own_frame(self):
        custom = self.tmp / "custom.ply"
        args = M2.parse_args(_m2_args(
            self.tmp / "out", ply=str(custom),
            occluder=[2.0, -1.5, 0.0, 2.5, 0.12, 3.2],
        ))
        scenario = M2.build_scenario(args)
        ply, scene, origin = M2.scene_for_scenario(args, scenario, self.tmp)
        self.assertEqual(custom, ply)
        self.assertEqual(scenario, scene)  # the caller's scene defines the frame
        np.testing.assert_allclose(origin, np.zeros(3))


class M2EvidenceMemoryTest(unittest.TestCase):
    """The M2 re-acquisition experiment (goal.md section 4 + milestone M2)."""

    @classmethod
    def setUpClass(cls):
        # the default comparison is the acceptance run: one episode per tool set,
        # shared by the tests below (each episode renders many CPU frames)
        cls._shared = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._shared.cleanup)
        cls.shared_dir = Path(cls._shared.name) / "m2"
        cls.comparison = M2.main(_m2_args(cls.shared_dir))

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.out_dir = self.tmp / "m2"

    def _run(self, **overrides):
        return M2.main(_m2_args(self.out_dir, **overrides))

    def _metrics(self, mode: str, out_dir: Path | None = None) -> dict:
        path = (out_dir or self.out_dir) / "modes" / mode / "metrics.json"
        assert path.exists(), f"missing {path}"
        return json.loads(path.read_text())

    def test_mock_comparison_memory_reacquires_basic_does_not(self):
        comparison = self.comparison
        self.assertEqual("memory_toolset_superior", comparison["verdict"])
        # same scene, instruction and *initial* budget for both tool sets (the
        # remaining budget differs because the memory mode spends more calls)
        self.assertEqual({"tool_budget": 12, "max_turns": 10, "trials": 1},
                         comparison["budget"])
        self.assertEqual(comparison["modes"]["memory"]["tool_budget"]["initial"],
                         comparison["modes"]["basic"]["tool_budget"]["initial"])
        rates = comparison["comparison"]["success_rate"]
        self.assertEqual(1.0, rates["memory"])
        self.assertEqual(0.0, rates["basic"])
        self.assertEqual({"memory": True, "basic": False},
                         comparison["comparison"]["memory_evidence_hit"])

        memory = self._metrics("memory", self.shared_dir)
        self.assertEqual("memory", memory["toolset"])
        self.assertTrue(memory["success"])
        self.assertEqual("finish", memory["finish_reason"])
        self.assertLessEqual(memory["final_distance_m"], memory["success_dist_m"])
        self.assertTrue(memory["target_visible_at_end"])
        self.assertEqual("memory", json.loads(
            (self.shared_dir / "modes" / "memory" / "summary.json").read_text()
        )["toolset"])

        # the loss of view really happened, and a scan from there cannot see it
        pre = memory["scenario_preconditions"]
        self.assertTrue(pre["target_visible_at_start"])
        self.assertTrue(pre["loss_occurred"])
        self.assertFalse(pre["target_visible_after_loss"])
        self.assertFalse(pre["scan_from_loss_pose_visible"])

        # query_memory returned the old evidence: observation id + image + pose
        recalled = memory["memory"]
        self.assertTrue(recalled["tool_available"])
        self.assertEqual(1, recalled["query_calls"])
        self.assertGreaterEqual(recalled["hits"], 1)
        self.assertTrue(recalled["recall_observation_ids"])
        self.assertTrue(recalled["recall_images"])
        for image in recalled["recall_images"]:
            self.assertTrue(Path(image).exists(), image)
        # ... and the model used it: cite, re-localize on the recalled pose, re-look
        self.assertTrue(recalled["evidence_cited_later"])
        self.assertIsNotNone(recalled["reposition_error_m"])
        self.assertLessEqual(recalled["reposition_error_m"], 0.5)
        self.assertTrue(recalled["relocalized"])
        self.assertTrue(recalled["target_visible_on_relook"])
        self.assertGreaterEqual(recalled["frames_published"], 1)
        self.assertGreaterEqual(recalled["images_sent_after_query"], 1)
        self.assertIs(True, memory["verify_goal"]["verified"])

        # the baseline has no retrieval tool at all
        basic = self._metrics("basic", self.shared_dir)
        self.assertEqual("basic", basic["toolset"])
        self.assertFalse(basic["success"])
        self.assertFalse(basic["memory"]["tool_available"])
        self.assertEqual(0, basic["memory"]["query_calls"])
        self.assertEqual(0, basic["memory"]["hits"])
        self.assertIsNone(basic["memory"]["reposition_error_m"])
        self.assertFalse(basic["memory"]["relocalized"])
        self.assertGreater(basic["final_distance_m"], basic["success_dist_m"])
        # the retrieval tools are part of the memory set only: both sets move
        # with the same intent vocabulary (that is what makes the comparison
        # fair), and the memory set adds exactly the four evidence tools
        self.assertEqual(
            set(memory["toolset_registered"]) - set(basic["toolset_registered"]),
            {"query_memory", "inspect_region", "ground_target", "compare_candidates"},
        )
        self.assertNotIn("query_memory", basic["toolset_registered"])
        self.assertIn("query_memory", memory["toolset_registered"])
        self.assertIn("return_to_observation", memory["toolset_registered"])
        self.assertIn("go_to_viewpoint", basic["toolset_registered"])

    def test_artifacts_and_trace_support_the_claim(self):
        for name in ("run_config.json", "scenario.json", "comparison.json"):
            self.assertTrue((self.shared_dir / name).exists(), name)
        report = json.loads((self.shared_dir / "scenario.json").read_text())
        # the synthetic scene is authored in the renderer's frame, so the reported
        # geometry is the frame the episode actually ran in, and the shift from the
        # command-line coordinates is recorded
        self.assertGreater(float(np.linalg.norm(report["scene_origin_flu"])), 0.1)
        self.assertTrue(Path(report["scene_ply"]).exists())
        trace = _read_trace(self.shared_dir / "modes" / "memory")
        names = [entry["tool"] for entry in trace]
        # the Phase-1 chain: every movement is an intent tool; a leg to a *new*
        # place is propose_viewpoints + go_to_viewpoint (a raw position is not a
        # movement target any more), the way back is return_to_observation
        self.assertEqual(
            ["observe", "propose_viewpoints", "go_to_viewpoint", "query_memory",
             "return_to_observation", "observe", "propose_viewpoints",
             "go_to_viewpoint", "verify_goal", "finish"],
            names,
        )
        self.assertNotIn("navigate", names)
        # the loss leg went to the scenario's loss pose: propose_viewpoints was
        # asked for a point there and go_to_viewpoint flew to exactly that one
        scenario = report["scenario"]
        bearing = math.radians(float(scenario["loss_bearing_deg"]))
        loss_pose = np.asarray(scenario["start_pos"], dtype=np.float64) + np.asarray(
            [math.cos(bearing), math.sin(bearing), 0.0]
        ) * float(scenario["loss_distance_m"])
        loss = trace[names.index("go_to_viewpoint")]
        self.assertEqual("go_to_viewpoint", loss["tool"])
        self.assertAlmostEqual(
            float(np.linalg.norm(
                np.asarray(loss["data"]["target_position"], dtype=np.float64) - loss_pose
            )),
            0.0,
            places=2,
        )
        self.assertIn("viewpoint_id", loss["args"])
        query = trace[names.index("query_memory")]
        self.assertEqual(STATUS_OK, query["status"])
        self.assertGreaterEqual(query["data"]["n_hits"], 1)
        images = [ev["ref"] for ev in query["evidence"] if ev["kind"] == "image"]
        self.assertTrue(images)
        for ref in images:
            self.assertTrue(Path(ref).exists(), ref)
        recalled_id = query["evidence"][0]["ref"]
        # the next call flies back to the recalled pose (of the frame the query
        # returned) and cites that observation
        returned = trace[names.index("query_memory") + 1]
        self.assertEqual("return_to_observation", returned["tool"])
        self.assertEqual(recalled_id, returned["args"]["observation_id"])
        self.assertIn(recalled_id, returned["model_reply"])
        self.assertEqual(report["scenario"]["start_pos"],
                         returned["data"]["target_position"])

    def test_saved_observation_frames_show_the_target(self):
        """The evidence frames the model reads must not be blind: with the scene
        authored in raw FLU they were exactly black (M2 REPORT, scene defect 1)."""
        from PIL import Image

        frames = sorted((self.shared_dir / "modes" / "memory" / "frames").glob("*.png"))
        self.assertTrue(frames)
        red = {}
        for path in frames:
            arr = np.asarray(Image.open(path).convert("RGB"), dtype=int)
            red[path.name] = int(((arr[:, :, 0] - arr[:, :, 1]) > 40).sum())
        self.assertGreater(max(red.values()), 0, f"no target pixel in any frame: {red}")

    def test_environment_triggered_loss_works_for_real_backends(self):
        # a real model is not assumed to fly away on cue: the environment moves it
        self._run(loss_trigger="evidence", modes="memory")
        metrics = self._metrics("memory")
        loss = metrics["loss"]
        self.assertEqual("environment", loss["mechanism"])
        self.assertEqual("observe", loss["after_tool"])
        self.assertEqual(0.0, loss["pose_error_m"])
        self.assertTrue(loss["visible_before"])
        self.assertFalse(loss["visible_after"])
        self.assertTrue(metrics["success"])
        self.assertTrue(metrics["memory"]["evidence_cited_later"])

    def test_without_occluder_the_scenario_is_reported_as_not_exercising_m2(self):
        # the preconditions are *measured*, not assumed: with no occluder a scan
        # from the loss pose would see the target again, so the episode no longer
        # demonstrates retrieval-driven re-acquisition - and the report says so
        self._run(modes="memory", occluder_free=True)
        report = json.loads((self.out_dir / "scenario.json").read_text())
        self.assertTrue(report["preconditions"]["target_visible_at_start"])
        self.assertTrue(report["preconditions"]["scan_from_loss_pose_visible"])
        metrics = self._metrics("memory")
        pre = metrics["scenario_preconditions"]
        self.assertTrue(pre["loss_occurred"])
        self.assertFalse(pre["target_visible_after_loss"])  # out of the camera cone
        self.assertTrue(pre["scan_from_loss_pose_visible"])  # but a scan could find it

    def test_recall_extraction_and_frame_resolution(self):
        # fallback stack on purpose: the M2 hooks must not require the modules
        args = RUNNER.parse_args([
            "--instruction", "find the target", "--out", str(self.tmp / "ep"),
            "--render-size", "32", "32", "--tool-budget", "8", "--max-turns", "4",
        ])
        world = _make_world(self.tmp)
        out_dir = self.tmp / "ep"
        memory, _, _ = RUNNER.build_memory({}, world, out_dir, "ep_fallback")
        registry = M2.M2Registry()
        RUNNER.register_fallback_tools(registry, toolset="memory")
        ctx = RUNNER.make_tool_context(world, memory, args, out_dir, "ep_fallback")
        observe = registry.call("observe", {"mode": "forward"}, ctx)
        self.assertEqual(STATUS_OK, observe.status)
        obs_id = observe.data["observation_id"]
        result = registry.call("query_memory", {"query": "target"}, ctx)
        self.assertEqual(STATUS_OK, result.status)
        recall = registry.latest_recall()
        self.assertIsNotNone(recall, result.data)
        self.assertEqual(obs_id, recall["observation_id"])
        self.assertIsNotNone(recall["pose"])
        # the derived recall fields let a script reference the retrieved evidence
        self.assertEqual(recall["pose"], registry.result_index["recall_pos"])
        self.assertEqual(obs_id, registry.result_index["recall_observation_id"])
        # the recalled frame is resolved and published for the next turn
        published = ctx.extra.get("images") or {}
        self.assertIn(obs_id, published, sorted(published))
        self.assertEqual(3, published[obs_id].shape[0])

    def test_real_model_backend_is_wired_like_the_m1_runner(self):
        """``--backend hf``: same flags as run_vlm_nav_tools, built once."""
        captured: dict = {}
        real_build = M2.RUNNER.build_vlm
        M2.RUNNER.build_vlm = lambda ns: (
            captured.update(vars(ns)) or {"fake": "backend"}
        )
        self.addCleanup(setattr, M2.RUNNER, "build_vlm", real_build)
        argv = [
            "--backend", "hf", "--model-path", "/weights/Qwen3-VL-8B",
            "--device", "cuda:1", "--max-new-tokens", "64",
            "--image-size", "320", "180", "--num-threads", "4", "--no-think",
            "--out", str(self.tmp / "hf"),
        ]
        args = M2.parse_args(argv)
        factory = M2.build_vlm_factory(args, M2.build_scenario(args))
        vlm = factory("memory", None)
        self.assertEqual("hf", captured["backend"])
        self.assertEqual("/weights/Qwen3-VL-8B", captured["model_path"])
        self.assertEqual("cuda:1", captured["device"])
        self.assertEqual(64, captured["max_new_tokens"])
        self.assertEqual((320, 180), captured["image_size"])
        self.assertEqual(4, captured["num_threads"])
        self.assertTrue(captured["no_think"])
        # one backend instance shared by every mode/trial, and the environment
        # performs the loss of view (a real model is not scripted)
        self.assertIs(vlm, factory("basic", None))
        self.assertEqual("evidence", M2.resolve_loss_trigger(args))
        self.assertEqual("model", M2.resolve_loss_trigger(
            M2.parse_args(["--backend", "mock", "--out", str(self.tmp / "m")])
        ))
        self.assertEqual("model", M2.resolve_loss_trigger(
            M2.parse_args(["--backend", "hf", "--loss-trigger", "model",
                           "--out", str(self.tmp / "m")])
        ))
        # the factory defers to the runner's own validation of the backend flags
        M2.RUNNER.build_vlm = real_build
        missing_model = M2.parse_args(["--backend", "hf", "--out", str(self.tmp / "us")])
        factory = M2.build_vlm_factory(missing_model, M2.build_scenario(missing_model))
        with self.assertRaises(SystemExit):
            factory("memory", None)


class M2EvidenceAblationTest(unittest.TestCase):
    """The third arm: the memory tool set with the evidence removed.

    ``memory_no_evidence`` runs the *same* catalogue and the same instruction as
    ``memory`` - the only difference is the archive: an observation written
    before the loss of view is accepted and dropped, so ``query_memory`` has
    nothing to return. The three-way mock run is the deterministic proof that
    the ablation removes exactly the evidence the M2 claim rests on.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.out_dir = self.tmp / "m2"

    def _run(self, **overrides):
        return M2.main(_m2_args(self.out_dir, **overrides))

    def _metrics(self, mode: str, out_dir: Path | None = None) -> dict:
        path = (out_dir or self.out_dir) / "modes" / mode / "metrics.json"
        assert path.exists(), f"missing {path}"
        return json.loads(path.read_text())

    def test_three_arm_comparison_with_the_evidence_removed(self):
        comparison = self._run(modes="memory,memory_no_evidence,basic")
        self.assertEqual(["memory", "memory_no_evidence", "basic"],
                         list(comparison["modes"]))
        self.assertEqual(
            {"memory": 1.0, "memory_no_evidence": 0.0, "basic": 0.0},
            comparison["comparison"]["success_rate"],
        )
        self.assertEqual("memory_toolset_superior", comparison["verdict"])
        self.assertEqual("evidence_required", comparison["ablation_verdict"])

        memory = self._metrics("memory")
        ablated = self._metrics("memory_no_evidence")
        # same catalogue, same prompt, same scenario: only the archive differs
        self.assertEqual(memory["toolset_registered"], ablated["toolset_registered"])
        self.assertEqual("memory", ablated["toolset"])
        self.assertEqual(memory["scenario_preconditions"], ablated["scenario_preconditions"])
        self.assertTrue(ablated["scenario_preconditions"]["loss_occurred"])

        # the retrieval tool is still there and still called - it answers nothing
        self.assertTrue(ablated["memory"]["tool_available"])
        self.assertGreaterEqual(ablated["memory"]["query_calls"], 1)
        self.assertEqual(0, ablated["memory"]["hits"])
        self.assertEqual([], ablated["memory"]["recall_observation_ids"])
        self.assertEqual([], ablated["memory"]["recall_images"])
        self.assertFalse(ablated["memory"]["evidence_cited_later"])
        self.assertFalse(ablated["memory"]["relocalized"])
        self.assertFalse(ablated["success"])
        self.assertGreater(ablated["final_distance_m"], ablated["success_dist_m"])
        # ... while the memory arm is answered by the archive it wrote
        self.assertGreaterEqual(memory["memory"]["hits"], 1)
        self.assertTrue(memory["evidence_ablation"]["first_sighting_archived"])

        # the ablation removed the evidence, it did not hide it: the writes are
        # accepted (the tool result is unchanged) and dropped
        ablation = ablated["evidence_ablation"]
        self.assertTrue(ablation["enabled"])
        self.assertGreaterEqual(ablation["dropped_records"], 1)
        self.assertEqual(0, ablation["kept_records"])
        self.assertEqual([], ablation["archived_observation_ids"])
        self.assertFalse(ablation["first_sighting_archived"])
        self.assertFalse(memory["evidence_ablation"]["enabled"])

        # no arm invented a movement loop: every movement is an intent tool
        for mode in ("memory", "memory_no_evidence", "basic"):
            sequence = self._metrics(mode)["calls"]["sequence"]
            self.assertNotIn("navigate", sequence)

    def test_ablate_evidence_flag_adds_the_arm(self):
        comparison = self._run(modes="memory", ablate_evidence=True)
        self.assertEqual(["memory", "memory_no_evidence"], list(comparison["modes"]))
        self.assertEqual("evidence_required", comparison["ablation_verdict"])
        self.assertEqual(True, json.loads(
            (self.out_dir / "run_config.json").read_text()
        )["ablate_evidence"])
        # an arm that is already requested is not duplicated
        comparison = self._run(modes="memory_no_evidence", ablate_evidence=True)
        self.assertEqual(["memory_no_evidence"], list(comparison["modes"]))
        self.assertIsNone(comparison["ablation_verdict"])

    def test_only_the_memory_arms_are_ablated(self):
        self._run(modes="basic", ablate_evidence=True)
        self.assertEqual(["basic", "memory_no_evidence"], list(json.loads(
            (self.out_dir / "scenario.json").read_text()
        )["modes"]))
        basic = self._metrics("basic")
        self.assertFalse(basic["evidence_ablation"]["enabled"])
        # the baseline has no retrieval tool at all, so the ablation is a no-op
        self.assertEqual(0, basic["evidence_ablation"]["dropped_records"])

    def test_model_calls_are_counted_inside_and_outside_the_tools(self):
        """The cost record must include the calls a tool makes internally."""
        comparison = self._run(modes="memory")
        self.assertEqual(1.0, comparison["comparison"]["success_rate"]["memory"])
        metrics = self._metrics("memory")
        cost = metrics["vlm_cost"]
        self.assertTrue(cost["recorded"])
        self.assertEqual(cost["total_calls"], cost["loop_calls"] + cost["tool_calls"])
        self.assertGreaterEqual(cost["loop_calls"], 1)
        self.assertEqual(metrics["vlm_calls"], cost["loop_calls"])
        self.assertGreater(cost["prompt_chars"], 0)
        self.assertEqual(
            sum(cost["tool_calls_by_tool"].values()), cost["tool_calls"]
        )

    def test_the_call_counter_files_in_tool_calls_under_the_tool(self):
        class StubBackend:
            def __init__(self):
                self.seen = []

            def decide(self, frames, system_prompt, user_prompt, context=None):
                self.seen.append((len(frames), system_prompt))
                return "{\"tool\": \"finish\", \"args\": {}}"

            def decide_with_tools(self, frames, system_prompt, user_prompt,
                                  tools_text="", context=None):
                self.seen.append((len(frames), tools_text))
                return "{\"tool\": \"finish\", \"args\": {}}"

        inner = StubBackend()
        counter = M2.CountingVLM(inner)
        self.assertEqual("{\"tool\": \"finish\", \"args\": {}}",
                         counter.decide_with_tools([1, 2], "sys", "user", "tools"))
        self.assertEqual("{\"tool\": \"finish\", \"args\": {}}",
                         counter.decide([1], "sys", "user", None))
        # a decide issued while a tool handler runs is the tool's call
        counter.tool_name = "ground_target"
        counter.decide([1], "sys", "user", None)
        counter.tool_name = None
        report = M2.vlm_cost_report(counter)
        self.assertEqual(2, report["loop_calls"])
        self.assertEqual(1, report["tool_calls"])
        self.assertEqual({"ground_target": 1}, report["tool_calls_by_tool"])
        self.assertEqual(3, report["total_calls"])
        self.assertEqual(4, report["images"])
        self.assertEqual(1, report["tool_images"])
        self.assertGreater(report["tool_prompt_chars"], 0)
        # the proxy is transparent: the backend's own attributes are still there
        self.assertEqual(3, len(inner.seen))

    def test_a_grounding_capture_counts_as_the_first_sighting(self):
        """Batch A grounds without calling ``observe``: the capture it grounds in
        is the sighting, so the archive must be judged on that id too."""
        class StubMemory:
            def __init__(self, ids):
                self._ids = list(ids)

            def ids(self):
                return list(self._ids)

        class StubRegistry:
            ablate_evidence = False

            def __init__(self, ids):
                self.ctx = type("Ctx", (), {"memory": StubMemory(ids)})()

        driver = M2.ScenarioDriver(None, M2.Scenario(), trigger="model")
        driver.states = [{"call": 1, "tool": "ground_target", "visible": True}]
        entries = [{"tool": "ground_target", "status": "ok",
                    "data": {"observation_id": "obs_2",
                             "candidates": [{"id": "cand_1", "position_source": "depth"}]}}]
        report = M2.ablation_report(StubRegistry(["obs_2"]), driver, entries,
                                    M2.Scenario())
        self.assertEqual(["obs_2"], report["pre_loss_observation_ids"])
        self.assertEqual(["obs_2"], report["pre_loss_sighting_ids"])
        self.assertTrue(report["first_sighting_archived"])
        # the same episode with the evidence ablated: the capture was dropped
        report = M2.ablation_report(StubRegistry([]), driver, entries, M2.Scenario())
        self.assertEqual(["obs_2"], report["pre_loss_sighting_ids"])
        self.assertFalse(report["first_sighting_archived"])
        # a grounding that returned no candidate saw nothing: not a sighting
        empty = [{"tool": "ground_target", "status": "empty",
                  "data": {"observation_id": "obs_2", "n_candidates": 0}}]
        report = M2.ablation_report(StubRegistry(["obs_2"]), driver, empty, M2.Scenario())
        self.assertEqual([], report["pre_loss_sighting_ids"])

    def test_capture_trigger_arms_on_any_capture_that_saw_the_target(self):
        """``--loss-trigger capture``: the M2 precondition, whatever the tool.

        ``evidence`` arms on an ``observe`` call only, so a model that grounds
        first (``ground_target`` captures the frame it grounds in) never faces
        the loss the memory arm is supposed to recover from.
        """
        comparison = self._run(loss_trigger="capture",
                               modes="memory,memory_no_evidence,basic")
        self.assertEqual("evidence_required", comparison["ablation_verdict"])
        memory = self._metrics("memory")
        loss = memory["loss"]
        self.assertEqual("environment", loss["mechanism"])
        self.assertEqual("observe", loss["after_tool"])
        self.assertTrue(loss["visible_before"])
        self.assertFalse(loss["visible_after"])
        self.assertFalse(memory["scenario_preconditions"]["scan_from_loss_pose_visible"])
        # the sighting is in the archive, the retrieval answers, the return leg
        # flies back to it and the target is seen again from there
        self.assertTrue(memory["evidence_ablation"]["first_sighting_archived"])
        self.assertGreaterEqual(memory["memory"]["hits"], 1)
        self.assertTrue(memory["memory"]["relocalized"])
        self.assertTrue(memory["success"])
        # with the evidence removed the same episode has nothing to recall
        ablated = self._metrics("memory_no_evidence")
        self.assertEqual("environment", ablated["loss"]["mechanism"])
        self.assertEqual(0, ablated["memory"]["hits"])
        self.assertFalse(ablated["evidence_ablation"]["first_sighting_archived"])
        self.assertFalse(ablated["success"])

    def test_the_archive_proxy_drops_only_pre_loss_writes(self):
        class StubLog:
            def __init__(self):
                self.entries: list[dict] = []

            def record(self, **kwargs):
                self.entries.append(kwargs)
                return str(kwargs.get("observation_id") or f"obs_{len(self.entries)}")

            def query(self, q, limit: int = 5):
                return list(self.entries)[:limit]

            def get(self, obs_id):
                for entry in self.entries:
                    if entry["observation_id"] == obs_id:
                        return entry
                return None

            def ids(self):
                return [str(entry["observation_id"]) for entry in self.entries]

        log = StubLog()
        driver = M2.ScenarioDriver(None, M2.Scenario(), trigger="model")
        proxy = M2.AblatedEvidenceMemory(log, driver)
        # the drone still sees the target: the write is accepted and dropped
        self.assertIsNone(proxy.record(observation_id="obs_1"))
        self.assertEqual([], log.entries)
        self.assertEqual(1, proxy.dropped)
        self.assertTrue(proxy.blocked())
        # the target left the view: from here on the archive is the real one
        driver.loss = {"mechanism": "test"}
        proxy.record(observation_id="obs_2")
        self.assertEqual(["obs_2"], log.ids())
        self.assertEqual(1, proxy.kept)
        self.assertFalse(proxy.blocked())
        # reads are the archive's own: only writes are filtered
        self.assertEqual(log.entries, proxy.query("target"))
        self.assertEqual(["obs_2"], proxy.ids())
        self.assertEqual("obs_2", proxy.get("obs_2")["observation_id"])
        # a proxy without a driver keeps the archive as it is
        plain = M2.AblatedEvidenceMemory(StubLog(), None)
        plain.record(observation_id="obs_1")
        self.assertEqual(0, plain.dropped)
        self.assertEqual(1, plain.kept)


class M2LayerArmTest(unittest.TestCase):
    """The layer arms really separate the layers (``--memory-episodic`` /
    ``--recall-images``, the E2 round).

    ``memory_spatial_only`` runs the memory catalogue with the episodic
    evidence layer off: a retrieval still answers with the pose of a remembered
    capture (that is the spatial/task record), but the hit carries no image and
    no sighting, and the runner attaches no recalled frame to the next turn -
    the arm must not look like the full arm in ``recall_images`` /
    ``frames_published``. ``memory_text_only`` is the *consumption* side of the
    same picture: the retrieval result still carries the frame, the runner does
    not attach it.
    """

    @classmethod
    def setUpClass(cls):
        cls._shared = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._shared.cleanup)
        cls.shared_dir = Path(cls._shared.name) / "m2_layers"
        cls.comparison = M2.main(_m2_args(
            cls.shared_dir, modes="memory,memory_spatial_only,memory_text_only"
        ))

    def _metrics(self, mode: str) -> dict:
        path = self.shared_dir / "modes" / mode / "metrics.json"
        assert path.exists(), f"missing {path}"
        return json.loads(path.read_text())

    def _query_entry(self, mode: str) -> dict:
        entries = _read_trace(self.shared_dir / "modes" / mode)
        return next(entry for entry in entries if entry["tool"] == "query_memory")

    def test_the_arms_pin_their_own_switch(self):
        switches = self.comparison["comparison"]["switches"]
        self.assertEqual(
            {"spatial_memory": "on", "episodic_memory": "on"},
            switches["requested"]["memory"],
        )
        self.assertEqual(
            {"spatial_memory": "on", "episodic_memory": "off"},
            switches["requested"]["memory_spatial_only"],
        )
        self.assertEqual(
            {"spatial_memory": "on", "episodic_memory": "on"},
            switches["requested"]["memory_text_only"],
        )
        self.assertEqual({"memory": "on", "memory_spatial_only": "on",
                          "memory_text_only": "off"},
                         switches["recall_images"])
        # ... and the memory the arm really built reports it
        self.assertTrue(self._metrics("memory")["memory_switches"]["applied"]["episodic_memory"])
        self.assertFalse(
            self._metrics("memory_spatial_only")["memory_switches"]["applied"]["episodic_memory"]
        )

    def test_the_episodic_arm_retrieves_poses_without_evidence(self):
        full = self._metrics("memory")["memory"]
        ablated = self._metrics("memory_spatial_only")["memory"]

        self.assertGreaterEqual(ablated["hits"], 1)  # the pose is still retrieved
        self.assertEqual(ablated["recall_observation_ids"],
                         full["recall_observation_ids"])
        # ... but that is all it is: no image, and no recalled frame attached
        self.assertEqual([], ablated["recall_images"])
        self.assertTrue(full["recall_images"])
        self.assertEqual(0, ablated["frames_published"])
        self.assertGreaterEqual(full["frames_published"], 1)
        self.assertLess(ablated["images_sent_after_query"],
                        full["images_sent_after_query"])

        entry = self._query_entry("memory_spatial_only")
        hit = entry["data"]["hits"][0]
        self.assertNotIn("image", hit)
        self.assertNotIn("sighting", hit)
        self.assertEqual("missing", hit["quality"]["frame"])
        self.assertEqual("off", entry["data"]["layers"]["episodic"])
        self.assertIn("episodic", entry["data"]["evidence_note"])
        # the id is still an id of this episode: the way back stays executable
        self.assertEqual(["observation"],
                         [ref["kind"] for ref in entry["evidence"]])

    def test_the_recall_images_switch_is_independent_of_the_layers(self):
        text_only = self._metrics("memory_text_only")["memory"]

        # the retrieval result still carries the frame (episodic layer on) ...
        self.assertEqual(
            [Path(image).name for image in self._metrics("memory")["memory"]["recall_images"]],
            [Path(image).name for image in text_only["recall_images"]],
        )
        # ... the runner just does not attach it to the next turn
        self.assertEqual(0, text_only["frames_published"])
        entry = self._query_entry("memory_text_only")
        self.assertEqual("on", entry["data"]["layers"]["episodic"])
        self.assertTrue(entry["data"]["hits"][0]["image"])

    def test_the_ablated_arm_is_compared_as_a_layer_control(self):
        verdict = self.comparison["discrimination_verdict"]
        report = self.comparison["comparison"]
        self.assertIn("instance_final_rate", report)
        # no decoy scene in this run: the layer verdict fields are still there
        # (each one None) and the arm stays the same catalogue
        self.assertEqual({"memory", "memory_spatial_only", "memory_text_only"},
                         set(self.comparison["modes"]))
        self.assertEqual(
            [], [mode for mode in self.comparison["modes"]
                 if self._metrics(mode)["toolset"] != "memory"]
        )
        self.assertIsNone(verdict)


class M2LowAltitudeVariantTest(unittest.TestCase):
    """The low-altitude acceptance variant: ground target + ground obstacle.

    basis: outputs/vlm_nav/audit/altitude_audit.md §5. The default M2 scene has
    its target floating 1.5 m up and no ground-level obstacle, so it cannot show
    whether the drone flies low and goes *around*; this variant puts the target
    on the floor behind a 0.8 m box and gives the world the matching altitude
    envelope (floor + 0.25 m, no-fly ceiling below the obstacle top + drone
    radius, so over-flying it is out).
    """

    @classmethod
    def setUpClass(cls):
        cls._shared = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._shared.cleanup)
        cls.shared_dir = Path(cls._shared.name) / "m2_low"
        cls.comparison = M2.main(
            _m2_args(cls.shared_dir, scene_variant="low_altitude", modes="memory")
        )

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _scenario(self, **overrides):
        args = M2.parse_args(
            _m2_args(self.tmp / "out", scene_variant="low_altitude", **overrides)
        )
        scenario = M2.build_scenario(args)
        ply, scene, origin = M2.scene_for_scenario(args, scenario, self.tmp)
        return args, scenario, ply, scene

    def test_the_start_opens_level_with_the_target_top(self):
        args, _raw, ply, scene = self._scenario()
        envelope = M2.flight_envelope(args, scene)
        lower, upper = envelope.cruise_window_z
        self.assertAlmostEqual(0.35, float(scene.start_pos[2]) - scene.target_top_z, places=6)
        self.assertGreaterEqual(float(scene.start_pos[2]), lower - 1.0e-9)
        self.assertLessEqual(float(scene.start_pos[2]), upper + 1.0e-9)
        # the target sits on the floor and the obstacle is 0.8 m tall
        self.assertAlmostEqual(0.2, float(scene.target_pos[2]) - scene.ground_z, places=6)
        obstacle = scene.occluders[0]
        top = float(obstacle.center[2]) + 0.5 * float(obstacle.size[2])
        self.assertAlmostEqual(0.8, top - scene.ground_z, places=6)
        # ... and the flyable floor is the nearest surface + 0.25 m
        self.assertAlmostEqual(scene.ground_z + 0.25, envelope.z_min, places=6)

    def test_the_obstacle_cannot_be_overflown(self):
        args, _raw, ply, scene = self._scenario()
        envelope = M2.flight_envelope(args, scene)
        obstacle_top = float(scene.occluders[0].center[2]) + 0.5 * float(scene.occluders[0].size[2])
        self.assertTrue(envelope.overflight_blocked)
        self.assertLessEqual(envelope.ceiling_z, obstacle_top + envelope.drone_radius_m)
        world = M2.build_world(args, scene, ply)
        self.assertAlmostEqual(envelope.z_min, float(world.z_min), places=6)
        self.assertAlmostEqual(envelope.ceiling_z, float(world.z_max), places=6)
        # the envelope is enforced, not just declared: a full-throttle climb is cut
        world.apply_action(M2.NavAction(kind="waypoint", up_m=6.0, reason="test climb"))
        self.assertLessEqual(float(world.position_flu[2]), envelope.ceiling_z + 1.0e-9)
        self.assertAlmostEqual(envelope.ceiling_z, float(world.position_flu[2]), places=6)

    def test_the_obstacle_blocks_the_way_back_but_not_the_opening_view(self):
        _args, _raw, ply, scene = self._scenario()
        start_view = M2.target_visibility(
            scene.start_pos, math.radians(scene.start_yaw_deg), scene
        )
        self.assertTrue(start_view["visible"], "the episode must open with the target in view")
        loss_view = M2.target_visibility(
            scene.loss_pose(scene.start_pos), math.radians(scene.loss_bearing_deg), scene
        )
        self.assertFalse(loss_view["visible"])
        self.assertTrue(loss_view["occluded"], "the ground obstacle must hide it")
        self.assertFalse(M2.scan_visibility(scene.loss_pose(scene.start_pos), scene)["visible"])
        # the direct route back crosses the obstacle, so it is a detour
        self.assertTrue(scene.occluders[0].blocks(scene.loss_pose(scene.start_pos),
                                                  scene.target_pos))

    def test_the_variant_rejects_a_conflicting_geometry(self):
        # the variant owns the geometry, so a second definition of it is an error
        # rather than a silently ignored flag
        for conflict in ({"target_pos": [1.0, 1.0, 1.0]},
                         {"occluder": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]},
                         {"ply": str(self.tmp / "scene.ply")}):
            args = M2.parse_args(
                _m2_args(self.tmp / "out", scene_variant="low_altitude", **conflict)
            )
            with self.assertRaises(SystemExit):
                M2.build_scenario(args)

    def test_the_low_altitude_run_stays_in_the_envelope_and_goes_around(self):
        metrics = json.loads(
            (self.shared_dir / "modes" / "memory" / "metrics.json").read_text()
        )
        altitude = metrics["altitude"]
        self.assertTrue(metrics["success"])
        self.assertTrue(altitude["straight_route_blocked"])
        self.assertTrue(altitude["overflight_blocked"])
        self.assertEqual(0, altitude["ceiling_violations"])
        self.assertEqual(0, altitude["obstacle_violations"])
        self.assertAlmostEqual(0.35, altitude["start_above_target_top_m"], places=3)
        lower, upper = altitude["cruise_window_z"]
        self.assertGreaterEqual(altitude["start_z"], lower - 1.0e-6)
        self.assertLessEqual(altitude["start_z"], upper + 1.0e-6)
        self.assertLessEqual(altitude["max_z"], altitude["z_max"] + 1.0e-6)
        report = json.loads((self.shared_dir / "scenario.json").read_text())
        self.assertEqual("low_altitude", report["scene_variant"])
        self.assertTrue(report["flight_envelope"]["overflight_blocked"])
        self.assertLess(report["flight_envelope"]["z_max"],
                        report["flight_envelope"]["z_min"] + 1.0)


if __name__ == "__main__":
    unittest.main()
