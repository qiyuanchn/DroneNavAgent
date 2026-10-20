"""Host extensions used by the mixed-mission wave: registry view, profiles.

Covers the additions the parallel workstreams rely on without touching the
existing lifecycle semantics (``tests/test_plugin_lifecycle.py`` remains the
regression for those).
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from vlm_nav.plugins import (
    PLUGIN_ACTIVE,
    PLUGIN_INSTALLED,
    LifecycleError,
    Plugin,
    PluginContext,
    PluginHost,
    PluginManifest,
    RegistryView,
    ServiceRegistry,
)
from vlm_nav.plugins.profiles import (
    BUILTIN_PROFILES,
    Profile,
    activate_profile,
    install_bundle,
)
from vlm_nav.tools import (
    STATUS_OK,
    ToolContext,
    ToolResult,
    ToolSpec,
)


def _tool(name: str, handler=None) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=name,
        input_schema={"type": "object", "properties": {}},
        handler=handler or (
            lambda ctx, args: ToolResult(STATUS_OK, {"tool": name, "args": dict(args)})
        ),
        side_effects=False,
    )


class ViewPlugin(Plugin):
    manifest = PluginManifest(
        id="demo.view", version="0.1.0", tools=("view_echo", "view_read"),
    )

    def __init__(self) -> None:
        self.view: RegistryView | None = None

    def activate(self, ctx: PluginContext) -> None:
        self.view = ctx.registry_view()
        self.view.register(_tool("view_echo"))
        self.view.register(_tool("view_read"))


class UndeclaredViewPlugin(Plugin):
    manifest = PluginManifest(id="demo.viewbad", version="0.1.0")

    def activate(self, ctx: PluginContext) -> None:
        ctx.registry_view().register(_tool("sneaky"))


class WorkerIsolatedPlugin(Plugin):
    manifest = PluginManifest(
        id="demo.worker", version="0.1.0", isolation="worker",
        tools=("worker_ping",),
    )

    def activate(self, ctx: PluginContext) -> None:
        ctx.registry_view().register(_tool("worker_ping"))


class OptionalConsumerPlugin(Plugin):
    manifest = PluginManifest(
        id="demo.optional", version="0.1.0", tools=("optional_probe",),
    )

    def __init__(self) -> None:
        self.seen = None

    def activate(self, ctx: PluginContext) -> None:
        self.seen = ctx.optional_service("demo.value@1", "fallback")
        ctx.tools.register(_tool("optional_probe"))


class ValuePlugin(Plugin):
    manifest = PluginManifest(
        id="demo.value", version="0.1.0", provides=("demo.value@1",),
    )

    def __init__(self, value=42) -> None:
        self.value = value

    def activate(self, ctx: PluginContext) -> None:
        ctx.services.provide(self.manifest.id, "demo.value@1", self.value)


class RecordingPlugin(Plugin):
    """A dependency-ordered fake used by the profile tests."""

    def __init__(
        self,
        plugin_id: str,
        *,
        provides=(),
        requires=(),
        optional=(),
        tools=(),
        order: list[str] | None = None,
    ) -> None:
        self.manifest = PluginManifest(
            id=plugin_id,
            version="0.1.0",
            provides=tuple(provides),
            requires=tuple(requires),
            optional_requires=tuple(optional),
            tools=tuple(tools),
        )
        self.order = order

    def activate(self, ctx: PluginContext) -> None:
        for contract in self.manifest.provides:
            ctx.services.provide(self.manifest.id, contract, self)
        for name in self.manifest.tools:
            ctx.tools.register(_tool(name))
        if self.order is not None:
            self.order.append(self.manifest.id)


class RegistryViewTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.host = PluginHost(root_dir=Path(self._tmp.name) / "plugins")
        self.tool_ctx = ToolContext(world=None, budget={"tool_calls_left": 20})

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_register_names_get_call_and_batch(self):
        plugin = ViewPlugin()
        self.host.install(plugin)
        self.host.activate("demo.view")
        view = plugin.view
        self.assertIsInstance(view, RegistryView)

        self.assertEqual(view.names(), ["view_echo", "view_read"])
        self.assertEqual(view.get("view_echo").name, "view_echo")
        self.assertIn("view_read", view)

        result = view.call("view_echo", {"x": 1}, self.tool_ctx)
        self.assertEqual(result.status, STATUS_OK)
        self.assertEqual(result.data["args"], {"x": 1})

        batch = view.batch(
            {"calls": [{"tool": "view_read", "args": {"y": 2}}]}, self.tool_ctx,
        )
        self.assertEqual(batch.status, STATUS_OK)
        self.assertEqual(batch.data["ok"], 1)
        self.assertEqual(batch.data["results"][0]["tool"], "view_read")
        self.assertEqual(batch.data["results"][0]["data"]["args"], {"y": 2})

    def test_register_is_still_manifest_scoped(self):
        with self.assertRaises(LifecycleError):
            self.host.install(UndeclaredViewPlugin())
            self.host.activate("demo.viewbad")
        self.assertEqual(self.host.tools.names(), [])
        self.assertEqual(self.host.status("demo.viewbad"), "failed")

    def test_view_registrations_are_revoked_on_deactivate(self):
        self.host.install(ViewPlugin())
        self.host.activate("demo.view")
        self.assertEqual(len(self.host.tools.names()), 2)
        report = self.host.deactivate("demo.view")
        self.assertEqual(report.status, "inactive")
        self.assertEqual(self.host.tools.names(), [])


class IsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.host = PluginHost(root_dir=Path(self._tmp.name) / "plugins")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_worker_isolation_is_accepted(self):
        manifest = PluginManifest(id="demo.worker", isolation="worker")
        manifest.validate()
        self.assertEqual(manifest.to_dict()["isolation"], "worker")

        self.host.install(WorkerIsolatedPlugin())
        report = self.host.activate("demo.worker")
        self.assertEqual(report.status, PLUGIN_ACTIVE)
        self.assertIn("worker_ping", self.host.tools.names())

    def test_unknown_isolation_is_still_rejected(self):
        with self.assertRaises(LifecycleError):
            PluginManifest(id="demo.other", isolation="thread").validate()


class OptionalServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.host = PluginHost(root_dir=Path(self._tmp.name) / "plugins")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_optional_service_returns_default_then_value(self):
        consumer = OptionalConsumerPlugin()
        self.host.install(consumer)
        self.host.activate("demo.optional")
        self.assertEqual(consumer.seen, "fallback")

        self.host.install(ValuePlugin(7))
        self.host.activate("demo.value")
        self.host.deactivate("demo.optional")
        self.host.activate("demo.optional")
        self.assertEqual(consumer.seen, 7)

    def test_service_registry_get_optional_keyword(self):
        services = ServiceRegistry()
        self.assertIsNone(services.get("demo.value@1", optional=True))
        self.assertEqual(
            services.get("demo.value@1", optional=True, default="d"), "d"
        )
        services.provide("p", "demo.value@1", 5)
        self.assertEqual(services.get("demo.value@1", optional=True), 5)


class InstallBundleTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.host = PluginHost(root_dir=Path(self._tmp.name) / "plugins")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_bundle_orders_dependencies_but_activates_nothing(self):
        consumer = RecordingPlugin("demo.consumer", requires=("demo.value@1",))
        provider = ValuePlugin()
        # Consumer first on purpose: install_bundle must not depend on order.
        reports = self.host.install_bundle(
            {"demo.consumer": consumer, "demo.value": provider}
        )
        self.assertEqual(
            [report.plugin_id for report in reports], ["demo.value", "demo.consumer"]
        )
        self.assertEqual(self.host.status("demo.value"), PLUGIN_INSTALLED)
        self.assertEqual(self.host.status("demo.consumer"), PLUGIN_INSTALLED)
        self.assertEqual(self.host.tools.names(), [])
        self.assertEqual(self.host.services.catalog(), {})

        self.host.activate("demo.consumer")
        self.assertEqual(self.host.status("demo.value"), PLUGIN_ACTIVE)

    def test_bundle_skips_already_installed_plugins(self):
        self.host.install(ValuePlugin())
        report = self.host.install_bundle([ValuePlugin()])
        self.assertEqual(len(report), 1)
        self.assertIn("already installed", report[0].detail)

    def test_bundle_rejects_duplicate_ids(self):
        with self.assertRaises(LifecycleError):
            self.host.install_bundle([ValuePlugin(), ValuePlugin()])

    def test_bundle_still_rejects_unavailable_requires(self):
        with self.assertRaises(LifecycleError) as caught:
            self.host.install_bundle(
                [RecordingPlugin("demo.consumer", requires=("demo.value@1",))]
            )
        self.assertIn("demo.value@1", str(caught.exception))


class ProfileTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.host = PluginHost(root_dir=Path(self._tmp.name) / "plugins")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_builtin_profiles_are_complete_and_ordered(self):
        self.assertEqual(
            set(BUILTIN_PROFILES),
            {"motion-only", "semantic-nav", "mixed-mission", "4090-local"},
        )
        self.assertEqual(
            BUILTIN_PROFILES["motion-only"].plugins,
            ("resource.manager", "motion.base"),
        )
        self.assertEqual(
            BUILTIN_PROFILES["semantic-nav"].plugins,
            (
                "resource.manager", "model.provider", "perception.core",
                "navigation.core", "task.tools", "semantic.navigation",
                "mcp.adapter",
            ),
        )
        mixed = BUILTIN_PROFILES["mixed-mission"].plugins
        self.assertEqual(
            mixed,
            (
                "resource.manager", "model.provider", "perception.core",
                "navigation.core", "task.tools", "task.planner",
                "semantic.navigation", "motion.base", "mixed.mission",
                "mcp.adapter",
            ),
        )
        self.assertIn("model.provider.worker", BUILTIN_PROFILES["4090-local"].plugins)
        self.assertNotIn("model.provider", BUILTIN_PROFILES["4090-local"].plugins)
        self.assertEqual(
            BUILTIN_PROFILES["4090-local"].context_defaults["perception.detector"],
            "sam3_dino",
        )
        self.assertEqual(
            BUILTIN_PROFILES["4090-local"].context_defaults["model.provider.worker"],
            {"device": "cuda:0", "dtype": "bfloat16", "quantization": "4bit"},
        )
        self.assertEqual(
            Profile("p", ("a",)).to_dict(),
            {"id": "p", "plugins": ["a"], "context_defaults": {},
             "description": ""},
        )

    def _mixed_factories(self, order: list[str]) -> dict:
        return {
            "resource.manager": lambda: RecordingPlugin(
                "resource.manager", provides=("resource.manager@1",), order=order,
            ),
            "model.provider": lambda: RecordingPlugin(
                "model.provider", provides=("model.provider@1",), order=order,
            ),
            "motion.base": lambda: RecordingPlugin(
                "motion.base", provides=("motion.service@1",), order=order,
            ),
            "task.tools": lambda: RecordingPlugin(
                "task.tools", provides=(), order=order,
            ),
            "task.planner": lambda: RecordingPlugin(
                "task.planner", provides=("task.planner@1",), order=order,
            ),
            "perception.core": lambda: RecordingPlugin(
                "perception.core", provides=("perception.tools@1",), order=order,
            ),
            "navigation.core": lambda: RecordingPlugin(
                "navigation.core", provides=("navigation.tools@1",), order=order,
            ),
            "semantic.navigation": lambda: RecordingPlugin(
                "semantic.navigation",
                provides=("semantic.service@1",),
                requires=("navigation.tools@1", "perception.tools@1"),
                order=order,
            ),
            "mixed.mission": lambda: RecordingPlugin(
                "mixed.mission",
                provides=("mission.service@1",),
                requires=("motion.service@1",),
                optional=("semantic.service@1", "task.planner@1"),
                order=order,
            ),
            "mcp.adapter": lambda: RecordingPlugin(
                "mcp.adapter", provides=("mcp.adapter@1",), order=order,
            ),
        }

    def test_activate_profile_installs_then_activates_in_order(self):
        order: list[str] = []
        profile = BUILTIN_PROFILES["mixed-mission"]
        factories = self._mixed_factories(order)

        result = activate_profile(self.host, profile, factories)

        self.assertEqual(result["profile"], "mixed-mission")
        self.assertEqual(result["active"], list(profile.plugins))
        self.assertEqual(order, list(profile.plugins))
        self.assertEqual(
            [status["action"] for status in result["reports"]],
            ["install"] * len(profile.plugins) + ["activate"] * len(profile.plugins),
        )
        self.assertEqual(
            set(result["catalog"]["tools"]),
            set(),  # the fakes declare no tools
        )
        self.assertEqual(
            sorted(plugin["id"] for plugin in result["catalog"]["plugins"]),
            sorted(profile.plugins),
        )

    def test_already_installed_plugin_skips_factory(self):
        self.host.install(ValuePlugin())
        result = activate_profile(
            self.host,
            Profile("demo", ("demo.value",)),
            {"demo.value": lambda: self.fail("factory must not run")},
        )
        self.assertEqual(result["active"], ["demo.value"])
        self.assertEqual(result["reports"][0]["action"], "activate")

    def test_missing_factory_is_reported(self):
        with self.assertRaises(LifecycleError) as caught:
            activate_profile(self.host, Profile("demo", ("demo.nope",)), {})
        self.assertIn("demo.nope", str(caught.exception))

    def test_factory_returning_the_wrong_plugin_is_rejected(self):
        with self.assertRaises(LifecycleError):
            activate_profile(
                self.host, Profile("demo", ("demo.wanted",)),
                {"demo.wanted": lambda: ValuePlugin()},
            )

    def test_motion_only_profile_with_real_plugin(self):
        from vlm_nav.plugins.infra import ResourceManagerPlugin
        from vlm_nav.plugins.motion import MOTION_TOOLS, MotionPlugin

        result = activate_profile(
            self.host, BUILTIN_PROFILES["motion-only"],
            {
                "motion.base": MotionPlugin,
                "resource.manager": ResourceManagerPlugin,
            },
        )
        self.assertEqual(result["active"], ["resource.manager", "motion.base"])
        self.assertEqual(
            self.host.tools.names(), sorted(MOTION_TOOLS)
        )
        self.host.deactivate("motion.base")
        self.host.deactivate("resource.manager")
        self.assertEqual(self.host.tools.names(), [])
        self.assertEqual(self.host.services.catalog(), {})

    def test_module_install_bundle_delegates(self):
        reports = install_bundle(self.host, {"demo.value": ValuePlugin()})
        self.assertEqual([report.plugin_id for report in reports], ["demo.value"])
        self.assertEqual(self.host.status("demo.value"), PLUGIN_INSTALLED)


if __name__ == "__main__":
    unittest.main()
