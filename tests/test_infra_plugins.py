"""Infrastructure plugins from goal.md section 2.

resource.manager exposes the host's refcounted resource table, world.service
exposes an episode-scoped world factory and closes what it created, and
mcp.adapter reports the existing MCP service vocabulary without making the
optional ``mcp`` package a hard import.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from vlm_nav.plugins import PluginHost, ServiceRegistry
from vlm_nav.plugins.infra import (
    MCP_SERVICE,
    RESOURCE_SERVICE,
    WORLD_SERVICE,
    McpAdapterPlugin,
    ResourceManagerPlugin,
    WorldServicePlugin,
)
from vlm_nav.tools import STATUS_OK, ToolContext, ToolRegistry


class _World:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class InfraPluginTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.registry = ToolRegistry()
        self.host = PluginHost(
            root_dir=Path(self._tmp.name) / "plugins",
            tool_registry=self.registry,
        )
        self.ctx = ToolContext(world=None, budget={"tool_calls_left": 5})

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_resource_manager_service_is_the_host_manager(self):
        self.host.install(ResourceManagerPlugin())
        self.host.activate("resource.manager")
        self.assertIs(
            self.host.services.get(RESOURCE_SERVICE), self.host.resources
        )
        self.host.deactivate("resource.manager")
        self.assertFalse(self.host.services.has(RESOURCE_SERVICE))

    def test_world_service_closes_created_worlds(self):
        worlds: list[_World] = []

        def factory() -> _World:
            world = _World()
            worlds.append(world)
            return world

        self.host.install(WorldServicePlugin(factory))
        self.host.activate("world.service")
        create = self.host.services.get(WORLD_SERVICE)
        first = create()
        second = create()
        self.assertIs(first, worlds[0])
        self.assertIs(second, worlds[1])
        self.host.deactivate("world.service")
        self.assertTrue(first.closed)
        self.assertTrue(second.closed)
        self.assertFalse(self.host.services.has(WORLD_SERVICE))

    def test_mcp_adapter_reports_services_and_tool(self):
        self.host.install(McpAdapterPlugin())
        self.host.activate("mcp.adapter")
        adapter = self.host.services.get(MCP_SERVICE)
        self.assertIn("env", adapter.services)
        self.assertIn("evidence", adapter.services)
        self.assertIn("mcp_services", self.registry.names())
        result = self.registry.call("mcp_services", {}, self.ctx)
        self.assertEqual(result.status, STATUS_OK)
        self.assertIn("env", result.data["services"])
        self.assertTrue(result.data["available"])
        self.host.deactivate("mcp.adapter")
        self.assertFalse(self.host.services.has(MCP_SERVICE))
        self.assertEqual(self.registry.names(), [])


class RuntimeMissionStateTest(unittest.TestCase):
    def test_structured_state_reports_mission_progress(self):
        from vlm_nav.runtime import AgentRuntime

        runtime = object.__new__(AgentRuntime)
        runtime.world = _World()
        runtime.memory = None
        runtime.ctx = ToolContext(world=runtime.world)
        runtime.ctx.extra["mission.progress"] = {
            "mission_id": "m1",
            "status": "running",
            "current_step": 1,
            "steps_total": 4,
            "steps_done": 1,
        }
        lines = runtime._state_lines()
        mission = [line for line in lines if line.startswith("mission=")]
        self.assertEqual(len(mission), 1)
        self.assertIn("m1", mission[0])
        self.assertIn("status=running", mission[0])
        self.assertIn("step=2/4", mission[0])
        self.assertIn("done=1", mission[0])


class ServiceRegistryOptionalTest(unittest.TestCase):
    def test_optional_get_returns_default(self):
        registry = ServiceRegistry()
        self.assertIsNone(registry.get("missing.service@1", optional=True))
        self.assertEqual(
            registry.get("missing.service@1", optional=True, default=7), 7
        )


if __name__ == "__main__":
    unittest.main()
