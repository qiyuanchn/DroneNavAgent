"""Runner bootstrap: install the mission stack into an existing registry."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from vlm_nav.plugins.bootstrap import install_mission_stack
from vlm_nav.tools import STATUS_OK, ToolContext, ToolRegistry, ToolResult, ToolSpec


def _stub(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=name,
        input_schema={"type": "object", "properties": {}},
        handler=lambda ctx, args: ToolResult(STATUS_OK, {"tool": name}),
        side_effects=name in ("approach_candidate", "go_to_viewpoint"),
    )


class BootstrapTest(unittest.TestCase):
    def test_auto_semantic_stack_shares_the_existing_registry(self):
        registry = ToolRegistry()
        for name in (
            "search_target", "approach_candidate", "confirm_candidate",
            "verify_goal",
        ):
            registry.register(_stub(name))
        with tempfile.TemporaryDirectory() as tmp:
            host = install_mission_stack(registry, Path(tmp) / "plugins")
            names = set(registry.names())
            self.assertTrue(
                {
                    "motion_mission", "mission_plan", "mission_run",
                    "plan_mission", "semantic_status",
                }.issubset(names)
            )
            self.assertEqual(
                host.status("semantic.navigation"), "active"
            )
            self.assertIn("semantic.service", host.services.catalog())
            self.assertIn("mission.service", host.services.catalog())
            # the caller's four tools are still there, not replaced
            self.assertTrue(
                {"search_target", "approach_candidate", "confirm_candidate",
                 "verify_goal"}.issubset(names)
            )
            for plugin_id in (
                "mixed.mission", "semantic.navigation", "task.planner",
                "motion.base",
            ):
                host.deactivate(plugin_id)
            self.assertEqual(
                set(registry.names()),
                {"search_target", "approach_candidate", "confirm_candidate",
                 "verify_goal"},
            )

    def test_motion_only_when_semantic_tools_are_absent(self):
        registry = ToolRegistry()
        with tempfile.TemporaryDirectory() as tmp:
            host = install_mission_stack(registry, Path(tmp) / "plugins")
            self.assertTrue(
                {"motion_mission", "mission_plan", "mission_run",
                 "plan_mission"}.issubset(registry.names())
            )
            self.assertNotIn("semantic.service", host.services.catalog())
            self.assertNotIn("semantic_status", registry.names())

    def test_explicit_semantic_requires_the_tools(self):
        from vlm_nav.plugins import LifecycleError

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(LifecycleError):
                install_mission_stack(
                    ToolRegistry(), Path(tmp) / "plugins", semantic=True
                )


if __name__ == "__main__":
    unittest.main()
