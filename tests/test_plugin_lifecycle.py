"""Plugin lifecycle host (goal.md M0/M3 groundwork).

The contract under test is the one the motion plugin relies on: declared
services are resolved without importing another plugin, every registration can
be revoked, deactivation refuses to yank a capability a task or another plugin
is using, uninstall removes private caches but not mission history, and cleanup
failures stay visible and retryable.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from vlm_nav.plugins import (
    LifecycleError,
    Plugin,
    PluginContext,
    PluginHost,
    PluginManifest,
    ResourceManager,
    ServiceRegistry,
)
from vlm_nav.tools import (
    STATUS_OK,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)


def _tool(name: str, handler):
    return ToolSpec(
        name=name,
        description=name,
        input_schema={"type": "object", "properties": {}},
        handler=handler,
        side_effects=False,
    )


class EchoPlugin(Plugin):
    manifest = PluginManifest(
        id="demo.echo",
        version="0.1.0",
        tools=("echo",),
        prompts=("demo.prompt",),
        events=("demo.ping",),
        data_dirs=("logs",),
        cache_dirs=("tmp",),
    )

    def __init__(self) -> None:
        self.pings: list[dict] = []

    def activate(self, ctx: PluginContext) -> None:
        ctx.tools.register(_tool("echo", lambda c, a: ToolResult(STATUS_OK, {"echo": a})))
        ctx.prompts.register("demo.prompt", "echo instructions")
        ctx.subscribe("demo.ping", self.pings.append)
        ctx.emit("demo.ping", {"hello": "world"})
        # data/cache paths are owned and already created by the host
        (ctx.data_path("logs") / "note.txt").write_text("kept", encoding="utf-8")

    def deactivate(self, ctx: PluginContext) -> None:
        self.deactivated = True


class ProviderPlugin(Plugin):
    manifest = PluginManifest(
        id="demo.provider",
        version="0.1.0",
        provides=("demo.value@1",),
        tools=("provider_value",),
    )

    def activate(self, ctx: PluginContext) -> None:
        ctx.services.provide(self.manifest.id, "demo.value@1", {"answer": 42})
        ctx.tools.register(_tool(
            "provider_value",
            lambda c, a: ToolResult(STATUS_OK, {"value": 42}),
        ))


class ConsumerPlugin(Plugin):
    manifest = PluginManifest(
        id="demo.consumer",
        version="0.1.0",
        requires=("demo.value@1",),
        tools=("consumer_value",),
    )

    def activate(self, ctx: PluginContext) -> None:
        value = ctx.services.get("demo.value@1", self.manifest.id)
        ctx.tools.register(_tool(
            "consumer_value",
            lambda c, a: ToolResult(STATUS_OK, dict(value)),
        ))


class ActiveTaskPlugin(Plugin):
    manifest = PluginManifest(id="demo.active", version="0.1.0")

    def __init__(self) -> None:
        self.busy = 1
        self.aborted = False

    def activate(self, ctx: PluginContext) -> None:
        ctx.activity_probe(lambda: self.busy)
        ctx.abort(self._abort)

    def _abort(self, reason: str) -> None:
        self.aborted = True
        self.busy = 0


class FailingCleanupPlugin(Plugin):
    manifest = PluginManifest(id="demo.badcleanup", version="0.1.0")

    def __init__(self) -> None:
        self.attempts = 0

    def activate(self, ctx: PluginContext) -> None:
        ctx.add_cleanup(self._cleanup, "flaky")

    def _cleanup(self) -> None:
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("cuda context still busy")


class FailingActivationPlugin(Plugin):
    manifest = PluginManifest(
        id="demo.fail", version="0.1.0", tools=("half",),
    )

    def activate(self, ctx: PluginContext) -> None:
        ctx.tools.register(_tool(
            "half", lambda c, a: ToolResult(STATUS_OK, {"half": True}),
        ))
        raise RuntimeError("model failed to load")


class UndeclaredToolPlugin(Plugin):
    manifest = PluginManifest(id="demo.undeclared", version="0.1.0")

    def activate(self, ctx: PluginContext) -> None:
        ctx.tools.register(_tool("sneaky", lambda c, a: ToolResult(STATUS_OK, {})))


class ProtectedPlugin(Plugin):
    manifest = PluginManifest(
        id="demo.protected", version="0.1.0", protected=True
    )


class LifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "plugins"
        self.registry = ToolRegistry()
        self.host = PluginHost(root_dir=self.root, tool_registry=self.registry)
        self.ctx = ToolContext(world=None, budget={"tool_calls_left": 10})

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_install_activate_call_deactivate_uninstall(self):
        plugin = EchoPlugin()
        self.host.install(plugin)
        self.assertEqual(self.host.status("demo.echo"), "installed")
        report = self.host.activate("demo.echo")
        self.assertEqual(report.status, "active")
        self.assertEqual(plugin.pings, [{"hello": "world"}])
        self.assertEqual(self.registry.names(), ["echo"])
        self.assertEqual(self.host.prompts(), {"demo.prompt": "echo instructions"})
        self.assertEqual(self.host.catalog()["plugins"][0]["tools"], ["echo"])
        self.assertEqual(self.host.events.subscription_count("demo.ping"), 1)
        data_note = self.root / "demo.echo" / "data" / "logs" / "note.txt"
        self.assertTrue(data_note.exists())
        self.assertTrue((self.root / "demo.echo" / "cache" / "tmp").is_dir())

        result = self.registry.call("echo", {"x": 1}, self.ctx)
        self.assertEqual(result.status, STATUS_OK)
        self.assertEqual(result.data["echo"], {"x": 1})

        report = self.host.deactivate("demo.echo")
        self.assertEqual(report.status, "inactive")
        self.assertEqual(self.registry.names(), [])
        self.assertEqual(self.host.prompts(), {})
        self.assertEqual(self.host.events.subscription_count("demo.ping"), 0)
        self.assertTrue(plugin.deactivated)
        self.assertTrue(data_note.exists(), "deactivate must preserve user data")
        self.assertEqual(self.host.services.catalog(), {})

        report = self.host.uninstall("demo.echo")
        self.assertEqual(report.status, "uninstalled")
        self.assertFalse((self.root / "demo.echo" / "cache").exists())
        self.assertTrue(data_note.exists(), "uninstall must preserve data by default")
        self.assertEqual(self.host.plugins(), [])
        self.assertEqual(self.registry.names(), [])

    def test_purge_data_is_explicit(self):
        self.host.install(EchoPlugin())
        self.host.activate("demo.echo")
        note = self.root / "demo.echo" / "data" / "logs" / "note.txt"
        self.assertTrue(note.exists())
        report = self.host.uninstall("demo.echo", purge_data=True)
        self.assertFalse(note.exists())
        self.assertIn("purge", report.detail)

    def test_data_can_be_purged_after_uninstall(self):
        self.host.install(EchoPlugin())
        self.host.activate("demo.echo")
        note = self.root / "demo.echo" / "data" / "logs" / "note.txt"
        self.host.uninstall("demo.echo")
        self.assertTrue(note.exists())
        report = self.host.purge_data("demo.echo")
        self.assertEqual(report.action, "purge_data")
        self.assertFalse(note.exists())

    def test_dependency_auto_activation_and_refusal(self):
        self.host.install(ProviderPlugin())
        self.host.install(ConsumerPlugin())
        # Activating the consumer activates the provider it needs.
        report = self.host.activate("demo.consumer")
        self.assertEqual(report.status, "active")
        self.assertEqual(self.host.status("demo.provider"), "active")
        value = self.registry.call("consumer_value", {}, self.ctx)
        self.assertEqual(value.data["answer"], 42)

        with self.assertRaises(LifecycleError) as caught:
            self.host.deactivate("demo.provider")
        self.assertEqual(caught.exception.dependents, ("demo.consumer",))
        with self.assertRaises(LifecycleError) as caught:
            self.host.uninstall("demo.provider")
        self.assertEqual(caught.exception.dependents, ("demo.consumer",))

        forced = self.host.deactivate("demo.provider", force=True)
        self.assertIn("demo.consumer", forced.dependents)
        self.assertEqual(self.host.status("demo.consumer"), "inactive")
        self.assertEqual(self.host.status("demo.provider"), "inactive")
        self.host.uninstall("demo.consumer")
        report = self.host.uninstall("demo.provider")
        self.assertEqual(report.status, "uninstalled")

    def test_missing_dependency_is_rejected_at_install(self):
        with self.assertRaises(LifecycleError):
            self.host.install(ConsumerPlugin())

    def test_active_task_blocks_deactivation_until_forced(self):
        plugin = ActiveTaskPlugin()
        self.host.install(plugin)
        self.host.activate("demo.active")
        with self.assertRaises(LifecycleError) as caught:
            self.host.deactivate("demo.active")
        self.assertEqual(caught.exception.active_tasks, 1)
        self.assertFalse(plugin.aborted)

        report = self.host.deactivate("demo.active", force=True)
        self.assertTrue(plugin.aborted)
        self.assertEqual(report.status, "inactive")
        record = self.host.record("demo.active")
        self.assertEqual(record.interrupted_tasks, 1)

    def test_activation_failure_rolls_back_registrations(self):
        with self.assertRaises(LifecycleError) as caught:
            self.host.install(FailingActivationPlugin())
            self.host.activate("demo.fail")
        self.assertIn("model failed to load", str(caught.exception))
        self.assertEqual(self.registry.names(), [])
        self.assertEqual(self.host.status("demo.fail"), "failed")
        self.assertEqual(self.host.services.catalog(), {})
        report = self.host.uninstall("demo.fail")
        self.assertEqual(report.status, "uninstalled")

    def test_undeclared_registration_is_a_manifest_violation(self):
        with self.assertRaises(LifecycleError):
            self.host.install(UndeclaredToolPlugin())
            self.host.activate("demo.undeclared")
        self.assertEqual(self.registry.names(), [])
        self.assertEqual(self.host.status("demo.undeclared"), "failed")

    def test_cleanup_failure_is_reported_and_retryable(self):
        plugin = FailingCleanupPlugin()
        self.host.install(plugin)
        self.host.activate("demo.badcleanup")
        report = self.host.deactivate("demo.badcleanup")
        self.assertEqual(report.status, "degraded")
        self.assertTrue(report.residual[0]["label"].endswith("flaky"))
        retried = self.host.retry_cleanup("demo.badcleanup")
        self.assertEqual(retried.status, "inactive")
        self.assertEqual(retried.residual, [])
        self.assertEqual(plugin.attempts, 2)

    def test_protected_plugin_cannot_be_removed(self):
        self.host.install(ProtectedPlugin())
        self.host.activate("demo.protected")
        with self.assertRaises(LifecycleError):
            self.host.deactivate("demo.protected")
        with self.assertRaises(LifecycleError):
            self.host.uninstall("demo.protected")

    def test_reinstall_after_uninstall(self):
        self.host.install(EchoPlugin())
        self.host.activate("demo.echo")
        self.host.uninstall("demo.echo")
        fresh = EchoPlugin()
        self.host.install(fresh)
        self.host.activate("demo.echo")
        self.assertEqual(self.host.status("demo.echo"), "active")
        self.assertEqual(self.registry.names(), ["echo"])


class ResourceManagerTest(unittest.TestCase):
    def test_shared_resource_lives_until_last_release(self):
        manager = ResourceManager()
        created = []
        cleaned = []

        def factory():
            created.append(True)
            return {"model": "qwen"}

        first = manager.acquire("a", "model:qwen", factory, cleanup=cleaned.append)
        second = manager.acquire("b", "model:qwen", factory, cleanup=cleaned.append)
        self.assertIs(first, second)
        self.assertEqual(len(created), 1)
        self.assertEqual(manager.refs("model:qwen"), 2)
        self.assertFalse(manager.release("a", "model:qwen"))
        self.assertEqual(cleaned, [])
        self.assertTrue(manager.release("b", "model:qwen"))
        self.assertEqual(cleaned, [{"model": "qwen"}])
        self.assertEqual(manager.refs("model:qwen"), 0)

    def test_cleanup_failure_is_retryable(self):
        manager = ResourceManager()
        attempts = []

        def cleanup(_value):
            attempts.append(True)
            if len(attempts) == 1:
                raise RuntimeError("worker still alive")

        manager.acquire("a", "worker", lambda: object(), cleanup=cleanup)
        manager.release("a", "worker")
        self.assertEqual(len(manager.failures("a")), 1)
        self.assertEqual(manager.retry_failures("a"), [])
        self.assertEqual(len(attempts), 2)


class ServiceRegistryTest(unittest.TestCase):
    def test_version_mismatch_and_duplicate_are_rejected(self):
        registry = ServiceRegistry()
        registry.provide("a", "demo.value@1", 1)
        with self.assertRaises(LifecycleError):
            registry.provide("b", "demo.value@1", 2)
        self.assertTrue(registry.has("demo.value"))
        self.assertFalse(registry.has("demo.value@2"))
        with self.assertRaises(LifecycleError):
            registry.get("demo.value@2")


if __name__ == "__main__":
    unittest.main()
