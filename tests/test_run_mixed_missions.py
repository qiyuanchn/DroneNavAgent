"""M4 mixed-mission matrix driver (``scripts/run_mixed_missions.py``).

The tests stay CPU-only and fast: every matrix invocation uses the repository
PLY fixture, 2-stage task files, 2 seeds, small frames and a temporary output
directory.  They pin:

* the 2/4/8-stage task files are valid mixed missions with the group contract
  (action = motion only, semantic = search/approach/confirm/verify, mixed =
  alternating motion/semantic);
* the mock semantic service answers the service contract, issues real registry
  calls and moves the world toward ``ctx.extra["mission.target_pose"]``;
* report.json/report.md use schema ``vlm_nav.mixed_missions.v1`` and contain the
  per-(group, stages) metrics, including false_completion and peak memory;
* interrupt/resume records the steps completed before the interruption, never
  replays a completed step and preserves measured motion progress;
* the full plugin lifecycle is exercised and leaves no residual catalogue,
  tools, services, worker pids or files;
* ``--semantic real`` reports an ImportError instead of silently falling back.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_nav.mixed_mission import (  # noqa: E402
    MOTION_STEP_KINDS,
    MixedMissionSpec,
)
from vlm_nav.tools import ToolContext, ToolRegistry  # noqa: E402

SCRIPT_PATH = ROOT / "scripts" / "run_mixed_missions.py"
FIXTURE = ROOT / "data" / "phase1" / "scenes" / "scenario_scene.ply"
TASKS_DIR = ROOT / "data" / "mixed_missions"


def _load_script():
    name = "diffaero_script_run_mixed_missions"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script()


class FakeWorld:
    """Writable kinematic pose; enough for the mock movement check."""

    def __init__(self) -> None:
        self.position = np.array([0.0, 0.0, 1.0])
        self.state = SimpleNamespace(position_flu=self.position)

    @property
    def position_flu(self):
        return self.position


def _context(world, target):
    return ToolContext(
        world=world,
        budget={"tool_calls_left": 32},
        extra={"mission.target_pose": list(target)},
    )


class TaskFileTest(unittest.TestCase):
    def test_stage_files_match_the_group_contract(self):
        for stages in (2, 4, 8):
            tasks = SCRIPT.load_tasks(TASKS_DIR, stages)
            self.assertEqual(set(tasks), set(SCRIPT.GROUPS))
            for group, missions in tasks.items():
                self.assertGreaterEqual(
                    len(missions), 2, f"{group}/{stages} needs >= 2 seeds"
                )
                for mission in missions:
                    with self.subTest(group=group, stages=stages, name=mission["name"]):
                        self.assertTrue(mission["name"])
                        self.assertTrue(mission["instruction"])
                        self.assertEqual(len(mission["steps"]), stages)
                        spec = MixedMissionSpec.from_dict(
                            {
                                "steps": mission["steps"],
                                "instruction": mission["instruction"],
                            }
                        )
                        self.assertEqual(len(spec.steps), stages)
                        kinds = [step["kind"] for step in spec.steps]
                        if group == "action":
                            self.assertTrue(
                                all(kind in MOTION_STEP_KINDS for kind in kinds)
                            )
                        elif group == "semantic":
                            self.assertTrue(
                                all(
                                    kind in SCRIPT.SEMANTIC_GROUP_KINDS
                                    for kind in kinds
                                )
                            )
                        else:
                            categories = [
                                "motion" if kind in MOTION_STEP_KINDS else "semantic"
                                for kind in kinds
                            ]
                            self.assertIn("motion", categories)
                            self.assertIn("semantic", categories)
                            for index in range(1, len(categories)):
                                self.assertNotEqual(
                                    categories[index], categories[index - 1]
                                )

    def test_task_file_rejects_wrong_group_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks_2.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": SCRIPT.TASKS_SCHEMA,
                        "stages": 2,
                        "groups": {
                            "action": [
                                {
                                    "name": "ok",
                                    "instruction": "up",
                                    "steps": [
                                        {"kind": "move", "direction": "up", "meters": 1},
                                        {"kind": "hover", "seconds": 0.1},
                                    ],
                                }
                            ],
                            "semantic": [
                                {
                                    "name": "wrong",
                                    "instruction": "up",
                                    "steps": [
                                        {"kind": "move", "direction": "up", "meters": 1},
                                        {"kind": "search", "target": "chair"},
                                    ],
                                }
                            ],
                            "mixed": [
                                {
                                    "name": "mixed",
                                    "instruction": "up, search",
                                    "steps": [
                                        {"kind": "move", "direction": "up", "meters": 1},
                                        {"kind": "search", "target": "chair"},
                                    ],
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(SCRIPT.TaskFileError):
                SCRIPT.load_tasks(tmp, 2)


class MockServiceTest(unittest.TestCase):
    def test_contract_calls_and_target_movement(self):
        registry = ToolRegistry()
        world = FakeWorld()
        service = SCRIPT.MockSemanticService(
            registry, target_pose=(100.0, 0.0, 1.0), seed=7
        )
        service.register()
        try:
            ctx = _context(world, (4.0, 0.0, 1.0))
            search = service.search(ctx, "窗边的木椅", relation="窗边")
            self.assertTrue(search["ok"])
            self.assertTrue(search["candidate_id"].startswith("cand_mock_7_"))
            for key in (
                "ok", "tool", "status", "candidate_id", "task_id",
                "arrived", "accepted", "verified", "calls", "error", "data",
            ):
                self.assertIn(key, search)

            approach = service.approach(ctx, search["candidate_id"])
            self.assertTrue(approach["ok"])
            self.assertTrue(approach["arrived"])
            self.assertGreater(world.position[0], 0.0)
            self.assertLess(world.position[0], 4.0)

            confirm = service.confirm(ctx, search["candidate_id"])
            self.assertTrue(confirm["accepted"])
            verify = service.verify(ctx, search["candidate_id"])
            self.assertTrue(verify["verified"])
            self.assertEqual(service.calls, 4)
            self.assertEqual(len(registry.names()), 7)
        finally:
            service.close()
        self.assertEqual(registry.names(), [])

    def test_extra_target_wins_over_constructor_target(self):
        registry = ToolRegistry()
        service = SCRIPT.MockSemanticService(
            registry, target_pose=(100.0, 0.0, 1.0)
        )
        service.register()
        try:
            world = FakeWorld()
            ctx = _context(world, (4.0, 0.0, 1.0))
            service.search(ctx, "chair")
            service.approach(ctx, "cand_mock_0_01", standoff_m=0.5)
            self.assertAlmostEqual(world.position[0], 1.0, places=6)
        finally:
            service.close()


class FalseCompletionTest(unittest.TestCase):
    def _state(self, evidence, status="done"):
        return SCRIPT.MixedMissionState.from_dict(
            {
                "schema": "drone.mixed_mission.state/1",
                "mission_id": "forged",
                "status": "completed",
                "current_step": 2,
                "steps": [
                    {
                        "kind": "move",
                        "params": {"direction": "forward", "meters": 1.0},
                        "status": "done",
                        "attempts": 1,
                    },
                    {
                        "kind": "confirm",
                        "params": {"target": "last"},
                        "status": status,
                        "attempts": 1,
                        "evidence": evidence,
                    },
                ],
            }
        )

    def test_completed_state_without_semantic_evidence_is_false_completion(self):
        self.assertTrue(SCRIPT.false_completion(self._state({})))
        self.assertTrue(
            SCRIPT.false_completion(self._state({"accepted": False}))
        )
        self.assertTrue(
            SCRIPT.false_completion(self._state({"accepted": True}, status="pending"))
        )
        self.assertFalse(
            SCRIPT.false_completion(self._state({"accepted": True}))
        )
        self.assertFalse(
            SCRIPT.false_completion(
                SCRIPT.MixedMissionState.from_dict(
                    {
                        "schema": "drone.mixed_mission.state/1",
                        "mission_id": "running",
                        "status": "failed",
                        "current_step": 0,
                        "steps": [
                            {
                                "kind": "search",
                                "params": {"target": "chair"},
                                "status": "failed",
                                "attempts": 1,
                            }
                        ],
                    }
                )
            )
        )


class MatrixTest(unittest.TestCase):
    def _run(self, tmp: str, *args: str, semantic: str = "mock"):
        out = Path(tmp) / "out"
        code = SCRIPT.main(
            [
                "--groups", "action,semantic,mixed",
                "--stages", "2",
                "--seeds", "2",
                "--world", "cpu",
                "--ply", str(FIXTURE),
                "--target-pos", "4", "0", "1",
                "--out", str(out),
                "--plugin-root", str(Path(tmp) / "plugins"),
                "--artifacts-dir", str(Path(tmp) / "artifacts"),
                "--width", "24",
                "--height", "14",
                "--semantic", semantic,
                *args,
            ]
        )
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        markdown = (out / "report.md").read_text(encoding="utf-8")
        return code, report, markdown

    def test_two_stage_matrix_report_and_clean_uninstall(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, report, markdown = self._run(tmp, "--interrupt-step", "1")

        self.assertEqual(code, 0, report.get("summary"))
        self.assertEqual(report["schema"], "vlm_nav.mixed_missions.v1")
        self.assertIn("Mixed mission matrix", markdown)
        self.assertEqual(report["semantic_backend"]["mode"], "mock")
        self.assertTrue(report["semantic_backend"]["available"])
        # 2/2/3 task fixtures (action/semantic/mixed) x 2 seeds.
        expected_by_group = {"action": 4, "semantic": 6, "mixed": 4}
        self.assertEqual(
            report["summary"]["episodes"], sum(expected_by_group.values())
        )
        self.assertEqual(
            report["summary"]["completed"], sum(expected_by_group.values())
        )
        self.assertEqual(report["summary"]["success_rate"], 1.0)
        self.assertEqual(report["summary"]["false_completion"], 0)
        self.assertEqual(report["summary"]["skipped_cells"], 0)

        expected_metrics = {
            "episodes", "completed", "success_rate", "false_completion",
            "tool_calls", "vlm_calls", "wall_time_s", "peak_allocated_mib",
            "peak_reserved_mib", "resume_recovered", "replayed_steps",
            "interrupted", "motion_progress_preserved", "clean_uninstall",
        }
        for group in SCRIPT.GROUPS:
            self.assertIn(group, report["metrics"])
            metrics = report["metrics"][group]["2"]
            self.assertTrue(expected_metrics.issubset(metrics), metrics)
            self.assertEqual(metrics["episodes"], expected_by_group[group], group)
            self.assertEqual(metrics["completed"], expected_by_group[group], group)
            self.assertEqual(metrics["success_rate"], 1.0, group)
            self.assertEqual(metrics["false_completion"], 0, group)
            self.assertEqual(
                metrics["resume_recovered"], expected_by_group[group], group
            )
            self.assertEqual(metrics["replayed_steps"], 0, group)
            self.assertEqual(
                metrics["interrupted"], expected_by_group[group], group
            )
            self.assertTrue(metrics["motion_progress_preserved"])
            self.assertTrue(metrics["clean_uninstall"])
            self.assertIsInstance(metrics["tool_calls"], int)
            self.assertEqual(metrics["vlm_calls"], 0)
            for key in ("peak_allocated_mib", "peak_reserved_mib"):
                self.assertTrue(
                    metrics[key] is None
                    or isinstance(metrics[key], (int, float)),
                    f"{key}={metrics[key]!r}",
                )
            if group == "action":
                self.assertEqual(metrics["tool_calls"], 0)
            else:
                self.assertGreater(metrics["tool_calls"], 0)

        cell = {
            item["group"]: item for item in report["cells"]
        }
        for group, item in cell.items():
            self.assertEqual(item["episodes"][0]["group"], group)
            self.assertEqual(
                sorted(item["lifecycle"]["active"]),
                sorted(
                    plugin["plugin_id"]
                    for plugin in item["lifecycle"]["install"]
                ),
            )
            self.assertTrue(item["lifecycle"]["uninstall"])
            for lifecycle_report in item["lifecycle"]["uninstall"]:
                self.assertEqual(lifecycle_report["status"], "uninstalled")
                self.assertEqual(lifecycle_report["residual"], [])
            resources = item["resources_after_uninstall"]
            self.assertTrue(resources["clean"], resources)
            self.assertEqual(resources["plugin_catalog_count"], 0)
            self.assertEqual(resources["service_count"], 0)
            self.assertEqual(resources["tool_count"], 0)
            self.assertEqual(resources["worker_pids"], [])
            self.assertEqual(resources["leftover_paths"], [])
            probe = item["lifecycle"]["probe"]
            self.assertEqual(probe["motion_mission"]["status"], "ok")
            if group != "action":
                self.assertEqual(probe["plan_mission"]["status"], "ok")

        resources = report["resources_after_uninstall"]
        self.assertTrue(resources["clean"], resources)
        self.assertEqual(resources["plugin_catalog_count"], 0)
        self.assertEqual(resources["residual"], [])
        self.assertEqual(resources["worker_pids"], [])
        self.assertEqual(resources["leftover_paths"], [])

    def test_interrupt_resume_never_replays_completed_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, report, _ = self._run(tmp, "--interrupt-step", "1")
        self.assertEqual(code, 0)
        episodes = report["episodes"]
        self.assertEqual(len(episodes), 14)
        for episode in episodes:
            with self.subTest(
                group=episode["group"], mission=episode["mission"]
            ):
                self.assertTrue(episode["interrupted"])
                self.assertEqual(episode["status"], "completed")
                self.assertTrue(episode["success"])
                resume = episode["resume"]
                self.assertTrue(resume["attempted"])
                self.assertTrue(resume["recovered"])
                self.assertEqual(resume["replayed_steps"], 0)
                self.assertTrue(resume["motion_progress_preserved"])
                self.assertGreaterEqual(resume["completed_before"], 1)
                self.assertEqual(resume["completed_after"], 2)
                self.assertTrue(episode["state_persisted"])
                final_steps = {
                    index: step
                    for index, step in enumerate(episode["state"]["steps"])
                }
                for before in resume["before"]:
                    if before["status"] != "done":
                        continue
                    after = final_steps[before["index"]]
                    self.assertEqual(after["status"], "done")
                    self.assertEqual(after["attempts"], before["attempts"])
                    if before["motion"] is not None:
                        motion_steps = (after.get("motion_state") or {}).get(
                            "steps"
                        ) or []
                        self.assertTrue(motion_steps)
                        progress = motion_steps[0].get("progress") or {}
                        if "target" in before["motion"]:
                            self.assertEqual(
                                [float(value) for value in progress["target"]],
                                [float(value) for value in before["motion"]["target"]],
                            )
                        for key in (
                            "requested_m", "moved_m", "remaining_m", "chunks",
                            "requested_deg", "turned_deg", "remaining_deg",
                            "requested_s", "elapsed_s", "remaining_s",
                        ):
                            if key in before["motion"]:
                                self.assertAlmostEqual(
                                    float(progress[key]),
                                    float(before["motion"][key]),
                                    places=6,
                                )
                for step in episode["state"]["steps"]:
                    self.assertEqual(step["attempts"], 1)

    def test_mock_semantic_steps_move_toward_the_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, report, _ = self._run(tmp)
        self.assertEqual(code, 0)
        semantic_episodes = [
            episode for episode in report["episodes"]
            if episode["group"] == "semantic"
        ]
        self.assertTrue(semantic_episodes)
        for episode in semantic_episodes:
            start = np.asarray(episode["initial_position"], dtype=float)
            final = np.asarray(episode["final_position"], dtype=float)
            target = np.asarray(episode["target_position"], dtype=float)
            self.assertLess(
                float(np.linalg.norm(target - final)),
                float(np.linalg.norm(target - start)),
            )

    def test_real_import_error_is_reported_not_faked(self):
        original = SCRIPT._real_plugin_classes

        def unavailable():
            raise ImportError("legacy navigation stack is not importable")

        SCRIPT._real_plugin_classes = unavailable
        try:
            with tempfile.TemporaryDirectory() as tmp:
                code, report, _ = self._run(
                    tmp, "--interrupt-step", "1", semantic="real"
                )
        finally:
            SCRIPT._real_plugin_classes = original

        self.assertEqual(code, 2)
        backend = report["semantic_backend"]
        self.assertEqual(backend["mode"], "real")
        self.assertFalse(backend["available"])
        self.assertIn("ImportError", backend["error"])
        cells = {cell["group"]: cell for cell in report["cells"]}
        self.assertFalse(cells["action"]["skipped"])
        self.assertEqual(cells["action"]["metrics"]["episodes"], 4)
        self.assertEqual(cells["action"]["metrics"]["success_rate"], 1.0)
        for group in ("semantic", "mixed"):
            self.assertTrue(cells[group]["skipped"])
            self.assertEqual(cells[group]["metrics"]["episodes"], 0)
            self.assertFalse(cells[group]["resources_after_uninstall"]["clean"])
        self.assertTrue(report["resources_after_uninstall"]["clean"])

    def test_without_interrupt_resume_metrics_stay_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, report, _ = self._run(tmp, "--groups", "action")
        self.assertEqual(code, 0)
        metrics = report["metrics"]["action"]["2"]
        self.assertEqual(metrics["interrupted"], 0)
        self.assertEqual(metrics["resume_recovered"], 0)
        self.assertEqual(metrics["replayed_steps"], 0)
        self.assertTrue(metrics["motion_progress_preserved"])
        for episode in report["episodes"]:
            self.assertFalse(episode["interrupted"])
            self.assertFalse(episode["resume"]["attempted"])


if __name__ == "__main__":
    unittest.main()
