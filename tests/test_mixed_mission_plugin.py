"""Mixed-mission plugin tools and the end-to-end CLI.

The plugin tests use the real lifecycle host and the real motion plugin, with a
fake semantic service.  The CLI tests run the repository CPU fixture in
``--motion-only`` and ``--semantic mock`` modes, so no VLM/GPU/network is
needed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from vlm_nav.actions import action_to_body_delta, body_delta_to_world
from vlm_nav.mixed_mission import (
    MIXED_COMPLETED,
    MIXED_FAILED,
    MIXED_STEP_DONE,
    MixedMissionState,
)
from vlm_nav.plugins import PluginHost
from vlm_nav.plugins.mixed import (
    MISSION_SERVICE,
    MISSION_TOOLS,
    MixedMissionPlugin,
)
from vlm_nav.plugins.motion import MOTION_SERVICE, MotionPlugin
from vlm_nav.plugins.task_planner import TaskPlannerPlugin
from vlm_nav.tools import ToolContext, ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "phase1" / "scenes" / "scenario_scene.ply"
SCRIPT = ROOT / "scripts" / "run_mixed_mission.py"


def _load_script():
    name = "drone_script_run_mixed_mission"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeWorld:
    def __init__(self) -> None:
        self.position = np.array([0.0, 0.0, 1.0])
        self.yaw = 0.0
        self.calls = 0

    @property
    def position_flu(self):
        return self.position

    @property
    def yaw_rad(self):
        return self.yaw

    def apply_action(self, action):
        self.calls += 1
        dx, dy, dz, dyaw = action_to_body_delta(action)
        self.yaw += dyaw
        wx, wy, wz = body_delta_to_world(dx, dy, dz, self.yaw)
        self.position = self.position + np.array([wx, wy, wz])
        return "ok"

    def render(self):
        return np.zeros((3, 8, 8), dtype=np.float32)


class FakeSemanticService:
    def __init__(self, *, arrived: bool = True) -> None:
        self.arrived = arrived
        self.calls: list[tuple[str, str]] = []

    def _reply(self, tool: str, **fields) -> dict:
        data = {
            "ok": True,
            "tool": tool,
            "status": "ok",
            "candidate_id": "cand_1",
            "task_id": "task_1",
            "arrived": True,
            "accepted": True,
            "verified": True,
            "calls": 1,
            "error": "",
            "data": {},
        }
        data.update(fields)
        return data

    def search(self, ctx, target, *, relation="", instruction=None,
               max_steps=8, max_turns=1):
        self.calls.append(("search", str(target)))
        return self._reply("search_target")

    def approach(self, ctx, candidate_id=None, *, standoff_m=1.2, max_legs=8):
        self.calls.append(("approach", str(candidate_id or "")))
        return self._reply("approach_candidate", arrived=self.arrived)

    def confirm(self, ctx, candidate_id=None, *, instruction=None,
                relation=None, max_views=2):
        self.calls.append(("confirm", str(candidate_id or "")))
        return self._reply("confirm_candidate", accepted=True)

    def verify(self, ctx, candidate_id=None, *, success_dist_m=None):
        self.calls.append(("verify", str(candidate_id or "")))
        return self._reply("verify_goal", verified=True)

    def revisit(self, ctx, *, position=None, observation_id=None,
                viewpoint_id=None, standoff_m=0.6):
        self.calls.append(("revisit", str(position or observation_id or viewpoint_id)))
        return self._reply("return_to_observation", arrived=True)

    def goto(self, ctx, position, *, standoff_m=0.6):
        self.calls.append(("goto", str(position)))
        return self._reply("go_to_viewpoint", arrived=True)


def _build_stack(tmp: Path, world, semantic=None):
    registry = ToolRegistry()
    host = PluginHost(root_dir=tmp / "plugins", tool_registry=registry)
    artifacts = tmp / "artifacts"
    plugins = (
        MotionPlugin(artifacts_dir=artifacts),
        TaskPlannerPlugin(),
        MixedMissionPlugin(artifacts_dir=artifacts),
    )
    for plugin in plugins:
        host.install(plugin)
    for plugin in plugins:
        host.activate(plugin.manifest.id)
    ctx = ToolContext(
        world=world,
        episode_id="mixed_test",
        budget={"tool_calls_left": 500},
        extra={},
    )
    if semantic is not None:
        ctx.extra["mission.semantic_service"] = semantic
    return host, registry, ctx, plugins


class PluginToolTest(unittest.TestCase):
    def test_manifest_and_tool_surface(self):
        manifest = MixedMissionPlugin.manifest
        self.assertEqual(manifest.id, "mixed.mission")
        self.assertEqual(manifest.version, "0.1.0")
        self.assertIn(MISSION_SERVICE, manifest.provides)
        self.assertIn(MOTION_SERVICE, manifest.requires)
        self.assertIn("semantic.service@1", manifest.optional_requires)
        self.assertIn("task.planner@1", manifest.optional_requires)
        self.assertEqual(tuple(manifest.tools), MISSION_TOOLS)
        self.assertEqual(tuple(manifest.data_dirs), ("state",))
        self.assertEqual(tuple(manifest.cache_dirs), ("cache",))

    def test_plan_then_run_end_to_end(self):
        world = FakeWorld()
        semantic = FakeSemanticService()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            host, registry, ctx, plugins = _build_stack(
                tmp_path, world, semantic
            )
            self.assertIn("mission.service", host.services.catalog())
            self.assertIn("task.planner", host.services.catalog())
            planned = registry.call("mission_plan", {
                "instruction": (
                    "向前飞 3 米，找到木椅，飞到它旁边，确认它，验证它，拍照"
                ),
            }, ctx)
            self.assertEqual(planned.status, "ok", planned.data)
            self.assertEqual(len(planned.data["steps"]), 6)

            state_path = tmp_path / "state.json"
            result = registry.call("mission_run", {
                "mission": planned.data,
                "state_path": str(state_path),
                "artifacts_dir": str(tmp_path / "artifacts"),
            }, ctx)
            self.assertEqual(result.status, "ok", result.data)
            state = ctx.extra["mission.state"]
            self.assertIsInstance(state, MixedMissionState)
            self.assertEqual(state.status, MIXED_COMPLETED)
            self.assertEqual(state.completed_steps(), 6)
            self.assertEqual(semantic.calls, [
                ("search", "木椅"),
                ("approach", "cand_1"),
                ("confirm", "cand_1"),
                ("verify", "cand_1"),
            ])
            self.assertAlmostEqual(world.position[0], 3.0)
            self.assertTrue(state_path.exists())
            self.assertTrue(ctx.extra["subgoal"].startswith("mission "))
            self.assertEqual(ctx.extra["mission.progress"]["status"], MIXED_COMPLETED)

            status = registry.call("mission_status", {}, ctx)
            self.assertEqual(status.status, "ok")
            self.assertEqual(status.data["status"], MIXED_COMPLETED)

            for plugin in reversed(plugins):
                host.deactivate(plugin.manifest.id)
            self.assertEqual(registry.names(), [])
            for plugin in reversed(plugins):
                report = host.uninstall(plugin.manifest.id)
                self.assertEqual(report.residual, [])
            self.assertEqual(host.catalog()["plugins"], [])
            self.assertEqual(host.services.catalog(), {})

    def test_run_instruction_directly_uses_planner_service(self):
        world = FakeWorld()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            host, registry, ctx, _ = _build_stack(tmp_path, world)
            result = registry.call("mission_run", {
                "instruction": "向前飞 2 米",
                "artifacts_dir": str(tmp_path / "artifacts"),
            }, ctx)
            self.assertEqual(result.status, "ok", result.data)
            self.assertEqual(ctx.extra["mission.state"].status, MIXED_COMPLETED)
            self.assertAlmostEqual(world.position[0], 2.0)

    def test_missing_semantic_service_is_structured_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            host, registry, ctx, _ = _build_stack(tmp_path, FakeWorld())
            result = registry.call("mission_run", {
                "steps": [
                    {"kind": "search", "target": "chair"},
                    {"kind": "move", "direction": "forward", "meters": 1},
                ],
                "artifacts_dir": str(tmp_path / "artifacts"),
            }, ctx)
            state = ctx.extra["mission.state"]
        self.assertEqual(result.status, "failed")
        self.assertEqual(state.status, MIXED_FAILED)
        # The lazy semantic service is built but the registry has no semantic
        # tools: a structured tool failure, never an executor crash.
        self.assertEqual(state.steps[0].status, "failed")
        self.assertIn("search not completed", state.steps[0].error)
        self.assertEqual(state.steps[1].status, "pending")

    def test_resume_tool_reopens_failed_semantic_step(self):
        world = FakeWorld()
        semantic = FakeSemanticService(arrived=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            host, registry, ctx, plugins = _build_stack(
                tmp_path, world, semantic
            )
            state_path = tmp_path / "state.json"
            first = registry.call("mission_run", {
                "steps": [
                    {"kind": "search", "target": "chair"},
                    {"kind": "approach", "target": "last"},
                ],
                "state_path": str(state_path),
                "artifacts_dir": str(tmp_path / "artifacts"),
            }, ctx)
            self.assertEqual(first.status, "failed")
            state = ctx.extra["mission.state"]
            self.assertEqual(state.status, MIXED_FAILED)
            attempts = state.steps[1].attempts

            semantic.arrived = True
            result = registry.call("mission_resume", {
                "state_path": str(state_path),
            }, ctx)
        self.assertEqual(result.status, "ok", result.data)
        state = ctx.extra["mission.state"]
        self.assertEqual(state.status, MIXED_COMPLETED)
        self.assertEqual(state.steps[1].attempts, attempts + 1)
        self.assertEqual(state.steps[0].attempts, 1)

    def test_pause_status_tools_on_paused_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            host, registry, ctx, plugins = _build_stack(tmp_path, FakeWorld())
            result = registry.call("mission_run", {
                "steps": [{"kind": "move", "direction": "forward", "meters": 1}],
                "artifacts_dir": str(tmp_path / "artifacts"),
            }, ctx)
            self.assertEqual(result.status, "ok")
            # Pausing a completed mission is a no-op, not a status regression.
            paused = registry.call("mission_pause", {}, ctx)
            self.assertEqual(paused.data["status"], MIXED_COMPLETED)
            resumed = registry.call("mission_resume", {}, ctx)
            self.assertEqual(resumed.data["status"], MIXED_COMPLETED)
            self.assertEqual(resumed.data["steps"][0]["attempts"], 1)


class ScriptTest(unittest.TestCase):
    def _run(self, *args: str) -> tuple[int, dict]:
        script = _load_script()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            code = script.main([
                "--width", "32", "--height", "18",
                "--out", str(out),
                "--artifacts-dir", str(Path(tmp) / "artifacts"),
                "--plugin-root", str(Path(tmp) / "plugins"),
                *args,
            ])
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        return code, summary

    def test_motion_only_cli(self):
        code, summary = self._run("--motion-only")
        self.assertEqual(code, 0, summary)
        self.assertEqual(summary["mode"], "none")
        self.assertEqual(summary["mission_status"], "completed")
        self.assertEqual(summary["reports"]["uninstall"][-1]["status"], "uninstalled")
        preserved = summary["reports"]["artifacts_preserved"]
        self.assertTrue(any(path.endswith(".png") for path in preserved))
        self.assertTrue(any(path.endswith(".json") for path in preserved))

    def test_mock_semantic_cli(self):
        code, summary = self._run("--semantic", "mock")
        self.assertEqual(code, 0, summary)
        self.assertEqual(summary["mode"], "mock")
        self.assertEqual(summary["mission_status"], "completed")
        steps = summary["reports"]["mission"]["state"]["steps"]
        self.assertEqual([step["kind"] for step in steps], [
            "move", "move", "hover", "turn", "search", "approach", "move",
            "photo",
        ])
        self.assertEqual(steps[4]["status"], MIXED_STEP_DONE)
        self.assertEqual(steps[5]["evidence"]["candidate_id"], "cand_mock_1")

    def test_motion_only_resume_state_cli(self):
        script = _load_script()
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.json"
            common = [
                "--motion-only", "--width", "32", "--height", "18",
                "--artifacts-dir", str(Path(tmp) / "artifacts"),
                "--plugin-root", str(Path(tmp) / "plugins"),
            ]
            first_out = Path(tmp) / "first"
            code = script.main([
                *common, "--out", str(first_out), "--state-path", str(state),
            ])
            self.assertEqual(code, 0)
            first = json.loads(
                (first_out / "summary.json").read_text(encoding="utf-8")
            )
            attempts = [
                step["attempts"] for step in first["reports"]["mission"]["state"]["steps"]
            ]
            self.assertTrue(state.exists())

            second_out = Path(tmp) / "second"
            code = script.main([
                *common, "--out", str(second_out), "--resume-state", str(state),
            ])
            second = json.loads(
                (second_out / "summary.json").read_text(encoding="utf-8")
            )
        self.assertEqual(code, 0)
        self.assertEqual(second["mission_status"], "completed")
        self.assertEqual(
            [step["attempts"] for step in second["reports"]["mission"]["state"]["steps"]],
            attempts,
        )

    def test_real_mode_reports_missing_stack(self):
        script = _load_script()
        # The real stack may exist (workstreams C/D) or not; either outcome is
        # a reported status, never a crash.
        try:
            from vlm_nav.plugins.legacy import PerceptionPlugin  # noqa: F401
            from vlm_nav.plugins.semantic import (  # noqa: F401
                SemanticNavigationPlugin,
            )
        except ImportError:
            with tempfile.TemporaryDirectory() as tmp:
                code = script.main([
                    "--semantic", "real",
                    "--width", "32", "--height", "18",
                    "--out", str(Path(tmp) / "out"),
                    "--artifacts-dir", str(Path(tmp) / "artifacts"),
                    "--plugin-root", str(Path(tmp) / "plugins"),
                    "--budget", "400",
                ])
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
