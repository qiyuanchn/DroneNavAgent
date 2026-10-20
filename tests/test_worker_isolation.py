"""Managed worker isolation (goal.md M3).

These tests run a real child process (``python -m vlm_nav.worker_echo``) but no
VLM, GPU or network. They pin the lifecycle contract the plugin host relies on:
a worker starts with a ping handshake, a timed-out request is an error rather
than a hang, and every exit path (stop, deactivate, uninstall, failed
configure) leaves no process behind.
"""

from __future__ import annotations

import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest

from vlm_nav.plugins import (
    LifecycleError,
    PluginHost,
    PluginManifest,
)
from vlm_nav.plugins.worker import (
    ManagedWorker,
    WorkerPlugin,
    WorkerSpec,
    WorkerTimeout,
)
from vlm_nav.tools import ToolRegistry
from vlm_nav.worker_echo import BANNER_ENV

ROOT = Path(__file__).resolve().parents[1]
COMMAND = (sys.executable, "-m", "vlm_nav.worker_echo")


def _spec(**overrides) -> WorkerSpec:
    values = {
        "command": COMMAND,
        "cwd": str(ROOT),
        "startup_timeout_s": 10.0,
        "request_timeout_s": 5.0,
        "stop_timeout_s": 2.0,
    }
    values.update(overrides)
    return WorkerSpec(**values)


def _assert_pid_gone(case: unittest.TestCase, pid: int) -> None:
    with case.assertRaises(ProcessLookupError):
        os.kill(pid, 0)


class EchoWorkerPlugin(WorkerPlugin):
    manifest = PluginManifest(
        id="demo.worker.echo",
        version="0.1.0",
        background_tasks=("worker",),
        isolation="worker",
    )

    def worker_spec(self, ctx) -> WorkerSpec:
        return _spec()

    def configure(self, worker: ManagedWorker, ctx) -> None:
        self.ping_result = worker.call("ping")
        self.echo_result = worker.call("echo", {"value": 7})


class FailingConfigurePlugin(EchoWorkerPlugin):
    manifest = PluginManifest(
        id="demo.worker.fail",
        version="0.1.0",
        background_tasks=("worker",),
        isolation="worker",
    )

    def configure(self, worker: ManagedWorker, ctx) -> None:
        raise RuntimeError("configure failed while loading weights")


class ManagedWorkerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_start_call_stop(self):
        log_path = self.tmp / "worker.log"
        worker = ManagedWorker(
            _spec(env={BANNER_ENV: "echo worker banner"}), log_path=log_path
        )
        worker.start()
        pid = worker.pid
        try:
            self.assertTrue(worker.alive)
            self.assertEqual(worker.state, "running")
            self.assertEqual(worker.call("echo", {"value": 7}), {"value": 7})
            self.assertEqual(worker.call("ping")["pid"], pid)
            self.assertEqual(worker.call("sleep", {"seconds": 0.05}), {"slept": 0.05})
            self.assertTrue(log_path.exists())
            self.assertIn("echo worker banner", worker.logs())
        finally:
            report = worker.stop()

        self.assertTrue(report["stopped"])
        self.assertFalse(report["forced"])
        self.assertEqual(report["pid"], pid)
        self.assertEqual(report["returncode"], 0)
        self.assertFalse(worker.alive)
        self.assertEqual(worker.state, "stopped")
        _assert_pid_gone(self, pid)

    def test_hung_method_times_out_and_force_stop_reaps(self):
        worker = ManagedWorker(
            _spec(request_timeout_s=0.2, stop_timeout_s=0.5)
        )
        worker.start()
        pid = worker.pid
        started = time.monotonic()
        with self.assertRaises(WorkerTimeout):
            worker.call("sleep", {"seconds": 30})
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertTrue(worker.alive, "a timed-out request must not fake a stop")

        report = worker.stop(force=True)
        self.assertTrue(report["stopped"])
        self.assertTrue(report["forced"])
        self.assertFalse(worker.alive)
        _assert_pid_gone(self, pid)

    def test_stop_escalates_sigterm_after_hung_graceful_shutdown(self):
        worker = ManagedWorker(
            _spec(request_timeout_s=0.2, stop_timeout_s=0.3)
        )
        worker.start()
        pid = worker.pid
        with self.assertRaises(WorkerTimeout):
            worker.call("sleep", {"seconds": 30})
        started = time.monotonic()
        report = worker.stop()
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertTrue(report["stopped"])
        self.assertTrue(report["forced"], "SIGTERM escalation must be reported")
        _assert_pid_gone(self, pid)

    def test_restart_policy_starts_a_new_process(self):
        worker = ManagedWorker(_spec(restart=True))
        worker.start()
        first = worker.pid
        os.kill(first, signal.SIGKILL)
        deadline = time.monotonic() + 5.0
        while worker.alive and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertFalse(worker.alive)

        self.assertEqual(worker.call("echo", {"value": "again"}), {"value": "again"})
        second = worker.pid
        self.assertNotEqual(second, first)
        self.assertTrue(worker.alive)
        worker.stop()
        _assert_pid_gone(self, second)


class WorkerPluginTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "plugins"
        self.host = PluginHost(root_dir=self.root, tool_registry=ToolRegistry())

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_activate_deactivate_uninstall_leaves_no_process(self):
        plugin = EchoWorkerPlugin()
        self.host.install(plugin)
        self.assertEqual(self.host.status("demo.worker.echo"), "installed")
        report = self.host.activate("demo.worker.echo")
        self.assertEqual(report.status, "active")

        first = plugin.worker.pid
        self.assertTrue(plugin.worker.alive)
        self.assertEqual(plugin.ping_result["pid"], first)
        self.assertEqual(plugin.echo_result, {"value": 7})
        log = self.root / "demo.worker.echo" / "cache" / "worker.log"
        self.assertTrue(log.exists())

        report = self.host.deactivate("demo.worker.echo")
        self.assertEqual(report.status, "inactive")
        self.assertFalse(plugin.worker.alive)
        _assert_pid_gone(self, first)

        # Reactivation creates a fresh child; uninstall must reap it too.
        self.host.activate("demo.worker.echo")
        second = plugin.worker.pid
        self.assertTrue(plugin.worker.alive)
        report = self.host.uninstall("demo.worker.echo")
        self.assertEqual(report.status, "uninstalled")
        self.assertFalse(plugin.worker.alive)
        _assert_pid_gone(self, second)
        self.assertEqual(self.host.plugins(), [])
        self.assertFalse((self.root / "demo.worker.echo" / "cache").exists())

    def test_configure_failure_stops_worker_and_does_not_leak(self):
        plugin = FailingConfigurePlugin()
        self.host.install(plugin)
        with self.assertRaises(LifecycleError) as caught:
            self.host.activate("demo.worker.fail")
        self.assertIn("configure failed", str(caught.exception))

        pid = plugin.worker.pid
        self.assertIsNotNone(pid)
        self.assertFalse(plugin.worker.alive)
        _assert_pid_gone(self, pid)
        self.assertEqual(self.host.status("demo.worker.fail"), "failed")
        self.assertIsNone(self.host.record("demo.worker.fail").ctx)

        # The failed record keeps no process or registration work to retry.
        report = self.host.uninstall("demo.worker.fail")
        self.assertEqual(report.status, "uninstalled")


if __name__ == "__main__":
    unittest.main()
