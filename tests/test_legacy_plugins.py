"""Legacy tool modules as lifecycle plugins (workstream D adapters).

The adapters exist so ``tool_impl_perception``/``tool_impl_navigation``/
``tool_impl_task`` keep working under the plugin host. The central assertion is
that each manifest declares exactly the tools its register function adds: an
undeclared or forgotten tool would otherwise make the catalogue lie.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from vlm_nav.plugins import (
    PLUGIN_ACTIVE,
    LifecycleError,
    Plugin,
    PluginContext,
    PluginHost,
    PluginManifest,
    RegistryView,
)
from vlm_nav.plugins import legacy
from vlm_nav.plugins.legacy import (
    MEMORY_SERVICE,
    MODEL_SERVICE,
    NAVIGATION_SERVICE,
    NAVIGATION_TOOLS,
    PERCEPTION_SERVICE,
    PERCEPTION_TOOLS,
    TASK_TOOLS,
    MemoryStorePlugin,
    ModelProviderPlugin,
    NavigationPlugin,
    PerceptionPlugin,
    TaskToolsPlugin,
)
from vlm_nav.tools import (
    STATUS_OK,
    ToolContext,
    ToolResult,
    ToolSpec,
)


class FakeBackend:
    def decide(self, frames, system_prompt, user_prompt, context):  # pragma: no cover
        return {"choice": "fake"}


class FakeStore:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class NoCloseProvider:
    pass


class DemoPlugin(Plugin):
    manifest = PluginManifest(
        id="demo.unrelated", version="0.1.0", tools=("demo_ping",),
    )

    def __init__(self) -> None:
        self.activated = False

    def activate(self, ctx: PluginContext) -> None:
        self.activated = True
        ctx.tools.register(ToolSpec(
            name="demo_ping",
            description="demo ping",
            input_schema={"type": "object", "properties": {}},
            handler=lambda c, a: ToolResult(STATUS_OK, {"pong": True}),
        ))


class LegacyPluginTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.host = PluginHost(root_dir=Path(self._tmp.name) / "plugins")
        self.ctx = ToolContext(world=None, budget={"tool_calls_left": 50})

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _activate(self, plugin: Plugin) -> None:
        self.host.install(plugin)
        report = self.host.activate(plugin.manifest.id)
        self.assertEqual(report.status, PLUGIN_ACTIVE)

    def test_perception_tools_are_exact(self):
        plugin = PerceptionPlugin()
        self._activate(plugin)

        self.assertEqual(
            self.host.tools.names(), sorted(PERCEPTION_TOOLS)
        )
        self.assertEqual(
            sorted(self.host.record("perception.core").tools),
            sorted(PERCEPTION_TOOLS),
        )
        self.assertTrue(self.host.services.has(PERCEPTION_SERVICE))
        self.assertIs(self.host.services.get(PERCEPTION_SERVICE), plugin)

        self.host.deactivate("perception.core")
        self.assertEqual(self.host.tools.names(), [])
        self.assertEqual(self.host.services.catalog(), {})

    def test_navigation_tools_are_exact(self):
        self._activate(NavigationPlugin())

        self.assertEqual(
            self.host.tools.names(), sorted(NAVIGATION_TOOLS)
        )
        self.assertEqual(
            sorted(self.host.record("navigation.core").tools),
            sorted(NAVIGATION_TOOLS),
        )
        self.assertTrue(self.host.services.has(NAVIGATION_SERVICE))

        self.host.deactivate("navigation.core")
        self.assertEqual(self.host.tools.names(), [])
        self.assertEqual(self.host.services.catalog(), {})

    def test_task_tools_are_exact(self):
        self._activate(TaskToolsPlugin())
        self.assertEqual(self.host.tools.names(), sorted(TASK_TOOLS))
        self.host.deactivate("task.tools")
        self.assertEqual(self.host.tools.names(), [])

    def test_manifests_match_register_function_output(self):
        # Introspect the real registries once: the constants must be complete.
        from vlm_nav.tools import ToolRegistry
        from vlm_nav.tool_impl_navigation import register_navigation_tools
        from vlm_nav.tool_impl_perception import register_perception_tools
        from vlm_nav.tool_impl_task import register_task_tools

        for register, names in (
            (register_perception_tools, PERCEPTION_TOOLS),
            (register_navigation_tools, NAVIGATION_TOOLS),
            (register_task_tools, TASK_TOOLS),
        ):
            registry = ToolRegistry()
            register(registry)
            self.assertEqual(sorted(registry.names()), sorted(names))

    def test_perception_uses_active_model_provider(self):
        provider = FakeBackend()
        provider_plugin = ModelProviderPlugin(lambda: provider)
        self._activate(provider_plugin)

        seen = {}
        real = legacy.register_perception_tools

        def spy(registry, vlm=None):
            seen["registry"] = registry
            seen["vlm"] = vlm
            return real(registry, vlm=vlm)

        with mock.patch.object(legacy, "register_perception_tools", spy):
            self._activate(PerceptionPlugin())

        self.assertIsInstance(seen["registry"], RegistryView)
        self.assertIs(seen["vlm"], provider)

    def test_navigation_uses_constructor_vlm_override(self):
        backend = FakeBackend()
        seen = {}
        real = legacy.register_navigation_tools

        def spy(registry, vlm=None):
            seen["vlm"] = vlm
            return real(registry, vlm=vlm)

        with mock.patch.object(legacy, "register_navigation_tools", spy):
            self._activate(NavigationPlugin(vlm=backend))

        self.assertIs(seen["vlm"], backend)

    def test_memory_store_is_provided_and_closed(self):
        store = FakeStore()
        plugin = MemoryStorePlugin(lambda: store)
        self._activate(plugin)
        self.assertIs(self.host.services.get(MEMORY_SERVICE), store)

        self.host.deactivate("memory.store")
        self.assertEqual(store.closed, 1)
        self.assertFalse(self.host.services.has(MEMORY_SERVICE))

        self.host.uninstall("memory.store", purge_data=True)
        self.assertEqual(store.closed, 1, "repeated teardown must not close twice")

    def test_model_provider_close_is_optional(self):
        with_close = FakeStore()
        plugin = ModelProviderPlugin(lambda: with_close)
        self._activate(plugin)
        self.assertIs(self.host.services.get(MODEL_SERVICE), with_close)
        self.host.deactivate("model.provider")
        self.assertEqual(with_close.closed, 1)

        plain = NoCloseProvider()
        self.host.uninstall("model.provider")
        plugin = ModelProviderPlugin(lambda: plain)
        self._activate(plugin)
        self.assertIs(self.host.services.get(MODEL_SERVICE), plain)
        self.host.deactivate("model.provider")

    def test_factory_is_required(self):
        with self.assertRaises(TypeError):
            MemoryStorePlugin(None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ModelProviderPlugin(None)  # type: ignore[arg-type]

    def test_deactivation_leaves_no_tools_or_services(self):
        plugins = [
            ModelProviderPlugin(lambda: FakeBackend()),
            PerceptionPlugin(),
            NavigationPlugin(),
            TaskToolsPlugin(),
            MemoryStorePlugin(lambda: FakeStore()),
        ]
        for plugin in plugins:
            self._activate(plugin)
        self.assertEqual(self.host.tools.names(), sorted(
            set(PERCEPTION_TOOLS) | set(NAVIGATION_TOOLS) | set(TASK_TOOLS)
        ))
        self.assertEqual(
            len(self.host.services.catalog()), 4,
            "perception, navigation, memory and model services",
        )

        for plugin in plugins:
            self.host.deactivate(plugin.manifest.id)
        self.assertEqual(self.host.tools.names(), [])
        self.assertEqual(self.host.services.catalog(), {})

    def test_unrelated_plugin_keeps_working_across_legacy_lifecycle(self):
        demo = DemoPlugin()
        self._activate(demo)
        ping = self.host.tools.call("demo_ping", {}, self.ctx)
        self.assertEqual(ping.status, STATUS_OK)

        perception = PerceptionPlugin()
        navigation = NavigationPlugin()
        self._activate(perception)
        self._activate(navigation)
        self.host.deactivate("navigation.core")
        self.host.deactivate("perception.core")

        ping = self.host.tools.call("demo_ping", {"probe": 2}, self.ctx)
        self.assertEqual(ping.status, STATUS_OK)
        self.assertEqual(self.host.status("demo.unrelated"), PLUGIN_ACTIVE)

    def test_undeclared_tool_fails_activation(self):
        class SneakyPlugin(PerceptionPlugin):
            manifest = PluginManifest(
                id="perception.sneaky",
                version="0.1.0",
                provides=(PERCEPTION_SERVICE,),
                tools=PERCEPTION_TOOLS[:-1],
            )

        with self.assertRaises(LifecycleError) as caught:
            self.host.install(SneakyPlugin())
            self.host.activate("perception.sneaky")
        self.assertIn("search_target", str(caught.exception))
        self.assertEqual(self.host.tools.names(), [])


if __name__ == "__main__":
    unittest.main()
