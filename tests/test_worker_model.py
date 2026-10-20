"""Worker-backed model provider (goal.md M3/M4, workstream G).

The tests run a real child process (``python -m vlm_nav.worker_model``) but no
VLM weights, GPU or network: the worker's ``--fake echo``/``--fake crash``/
``--fake fatal`` backends exercise the process boundary, and an injected fake
backend exercises ``serve`` in-process. They pin the contract the runtime
relies on:

* proxy round-trip and byte-exact reply passthrough for both backend methods;
* frame encoding: torch CHW float [0, 1] and HWC uint8 arrays become PNG
  base64 and arrive as torch CHW float tensors;
* plugin lifecycle: ``model.provider@1`` is provided on activate, revoked on
  deactivate, and the child pid is reaped;
* a crashed/erroring worker produces a structured ``WorkerModelError`` instead
  of a hang or an invented reply;
* a failing ``configure`` leaves no process behind.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest

import numpy as np
from PIL import Image
import torch

from vlm_nav.plugins import LifecycleError, PluginHost
from vlm_nav.plugins.worker import ManagedWorker, WorkerSpec
from vlm_nav.plugins.worker_model import (
    MODEL_SERVICE,
    WorkerModelError,
    WorkerModelProviderPlugin,
    jsonable_context,
)
from vlm_nav.worker_model import (
    EchoBackend,
    FrameEncodingError,
    WorkerBackendError,
    build_backend,
    decode_frame,
    encode_frame,
    encode_frames,
    serve,
)

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ID = "model.provider.worker"
_HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


def _worker_spec(fake: str = "echo", **overrides) -> WorkerSpec:
    values = {
        "command": (
            sys.executable, "-u", "-m", "vlm_nav.worker_model",
            "--fake", fake,
        ),
        "cwd": str(ROOT),
        "startup_timeout_s": 15.0,
        "request_timeout_s": 10.0,
        "stop_timeout_s": 3.0,
    }
    values.update(overrides)
    return WorkerSpec(**values)


def _assert_pid_gone(case: unittest.TestCase, pid: int) -> None:
    with case.assertRaises(ProcessLookupError):
        os.kill(pid, 0)


def _json_reply(reply: str) -> dict:
    data = json.loads(reply)
    if not isinstance(data, dict):
        raise AssertionError(f"worker reply is not a JSON object: {reply!r}")
    return data


class FrameEncodingTest(unittest.TestCase):
    def test_chw_float_round_trip(self):
        frame = (
            torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5) / 60.0
        )
        payload = encode_frame(frame)
        self.assertEqual(
            base64.b64decode(payload)[:8], b"\x89PNG\r\n\x1a\n",
            "frames must cross the wire as PNG",
        )
        decoded = decode_frame(payload)
        self.assertEqual(tuple(decoded.shape), (3, 4, 5))
        self.assertEqual(decoded.dtype, torch.float32)
        # 8-bit quantization is the only lossy step.
        self.assertTrue(torch.allclose(decoded, frame, atol=1.0 / 255.0 + 1e-6))
        self.assertGreaterEqual(float(decoded.min()), 0.0)
        self.assertLessEqual(float(decoded.max()), 1.0)

    def test_hwc_uint8_numpy_and_torch(self):
        hwc = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        expected = (
            torch.from_numpy(hwc).permute(2, 0, 1).to(torch.float32) / 255.0
        )
        for frame in (hwc, torch.from_numpy(hwc)):
            decoded = decode_frame(encode_frame(frame))
            self.assertEqual(tuple(decoded.shape), (3, 4, 5))
            self.assertTrue(torch.allclose(decoded, expected, atol=1e-6))

    def test_pil_frame_and_single_frame_list(self):
        image = Image.new("RGB", (5, 4), (255, 0, 0))
        decoded = decode_frame(encode_frame(image))
        self.assertEqual(tuple(decoded.shape), (3, 4, 5))
        self.assertTrue(torch.allclose(decoded[:, 0, 0], torch.tensor([1.0, 0.0, 0.0])))

        payloads = encode_frames(torch.zeros(3, 2, 2))
        self.assertEqual(len(payloads), 1, "a bare tensor is one frame")
        self.assertEqual(len(encode_frames([torch.zeros(3, 2, 2)] * 3)), 3)

    def test_unsupported_and_corrupt_frames_are_reported(self):
        with self.assertRaises(FrameEncodingError):
            encode_frame(torch.zeros(1, 2, 3, 4))
        with self.assertRaises(FrameEncodingError):
            decode_frame("not base64")
        with self.assertRaises(FrameEncodingError):
            encode_frames(None)


class BackendFactoryTest(unittest.TestCase):
    def test_injected_fake_wins_over_the_spec(self):
        echo = EchoBackend()
        self.assertIs(build_backend("hf", backend=echo), echo)

    def test_mock_spec_needs_no_model(self):
        backend = build_backend("mock:forward")
        self.assertEqual(backend.name, "mock")
        reply = backend.decide([torch.zeros(3, 2, 2)], "s", "u")
        self.assertEqual(json.loads(reply)["action"], "move")

    def test_hf_without_model_path_is_a_startup_error(self):
        with self.assertRaises(WorkerBackendError):
            build_backend("hf")

    def test_hf_precision_options_are_validated_before_loading(self):
        # These fail before touching the model path, so no weights are needed.
        with self.assertRaises(WorkerBackendError) as dtype_error:
            build_backend(
                "hf", model_path="/models/missing", dtype="float8"
            )
        self.assertIn("unknown dtype", str(dtype_error.exception))
        with self.assertRaises(WorkerBackendError) as quant_error:
            build_backend(
                "hf", model_path="/models/missing", quantization="2bit"
            )
        self.assertIn("unknown quantization", str(quant_error.exception))

    @unittest.skipUnless(
        _HAS_TRANSFORMERS, "transformers is not installed"
    )
    def test_hf_wrapper_injects_dtype_and_quantization(self):
        from unittest import mock
        from transformers import AutoModelForImageTextToText
        from vlm_nav import worker_model as worker_module

        captured = {}

        def fake_factory(spec, **options):
            AutoModelForImageTextToText.from_pretrained("fake-model")
            return EchoBackend()

        def fake_from_pretrained(*args, **kwargs):
            captured.update(kwargs)
            return object()

        with mock.patch.object(worker_module, "make_backend", fake_factory):
            with mock.patch.object(
                AutoModelForImageTextToText,
                "from_pretrained",
                staticmethod(fake_from_pretrained),
            ):
                backend = worker_module.build_backend(
                    "hf",
                    model_path="/models/missing",
                    device="cpu",
                    dtype="float16",
                    quantization="4bit",
                    max_new_tokens=8,
                )
        self.assertIsInstance(backend, EchoBackend)
        self.assertEqual(captured["dtype"], torch.float16)
        config = captured["quantization_config"]
        self.assertTrue(config.load_in_4bit)
        self.assertFalse(config.load_in_8bit)
        self.assertEqual(config.bnb_4bit_compute_dtype, torch.float16)


class ServeTest(unittest.TestCase):
    """In-process server with an injectable fake backend (no subprocess)."""

    class FakeBackend:
        name = "fake-serve"

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def decide(self, frames, system_prompt, user_prompt, context=None):
            self.calls.append({
                "method": "decide", "frames": frames,
                "system_prompt": system_prompt, "user_prompt": user_prompt,
                "context": context,
            })
            return "DECIDE:" + str(tuple(frames[0].shape))

        def decide_with_tools(
            self, frames, system_prompt, user_prompt, tools_text="", context=None
        ):
            self.calls.append({
                "method": "tools", "frames": frames,
                "system_prompt": system_prompt, "user_prompt": user_prompt,
                "tools_text": tools_text, "context": context,
            })
            return "TOOLS:" + tools_text

    def test_serve_serves_both_methods_and_shuts_down(self):
        backend = self.FakeBackend()
        frame = encode_frame(torch.zeros(3, 6, 8))
        requests = [
            {"id": 1, "method": "ping", "params": {}},
            {
                "id": 2, "method": "vlm.decide",
                "params": {
                    "frames": [frame], "system_prompt": "S",
                    "user_prompt": "U", "context": {"step": 2},
                },
            },
            {
                "id": 3, "method": "vlm.decide_with_tools",
                "params": {
                    "frames": [frame], "system_prompt": "S",
                    "user_prompt": "U", "tools_text": "CALLS",
                    "context": None,
                },
            },
            {"id": 4, "method": "shutdown", "params": {}},
        ]
        stdout = io.StringIO()
        code = serve(
            backend,
            stdin=io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n"),
            stdout=stdout,
        )
        self.assertEqual(code, 0)
        replies = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual([reply["id"] for reply in replies], [1, 2, 3, 4])
        self.assertTrue(replies[0]["result"]["pong"])
        self.assertEqual(replies[0]["result"]["backend"], "fake-serve")
        self.assertEqual(replies[1]["result"], "DECIDE:(3, 6, 8)")
        self.assertEqual(replies[2]["result"], "TOOLS:CALLS")
        self.assertTrue(replies[3]["result"]["shutdown"])

        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(tuple(backend.calls[0]["frames"][0].shape), (3, 6, 8))
        self.assertEqual(backend.calls[0]["context"], {"step": 2})
        self.assertEqual(backend.calls[1]["tools_text"], "CALLS")

    def test_serve_reports_backend_and_protocol_errors(self):
        class Broken:
            name = "broken"

            def decide(self, *args, **kwargs):
                raise RuntimeError("backend exploded")

            def decide_with_tools(self, *args, **kwargs):
                raise RuntimeError("backend exploded")

        requests = [
            {"id": 1, "method": "vlm.decide", "params": {}},
            {"id": 2, "method": "no.such.method", "params": {}},
            {"id": 3, "method": "shutdown", "params": {}},
        ]
        stdout = io.StringIO()
        serve(
            Broken(),
            stdin=io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n"),
            stdout=stdout,
        )
        replies = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertFalse(replies[0]["ok"])
        self.assertIn("backend exploded", replies[0]["error"])
        self.assertFalse(replies[1]["ok"])
        self.assertIn("unknown method", replies[1]["error"])
        self.assertTrue(replies[2]["ok"])


class RemoteProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _start(self, fake: str = "echo"):
        worker = ManagedWorker(_worker_spec(fake))
        worker.start()
        return worker

    def test_proxy_round_trip_is_byte_exact(self):
        from vlm_nav.plugins.worker_model import RemoteModelProvider

        worker = self._start("echo")
        provider = RemoteModelProvider(worker)
        pid = worker.pid
        frame = torch.rand(3, 12, 16)
        context = {
            "episode_id": "ep_1",
            "history": [],
            "position_flu": (1.0, 2.0, 3.0),
            "np_scalar": np.float32(0.5),
            "tensor": torch.tensor([1, 2]),
            "nested": {"pair": (4, 5)},
            "nan": float("nan"),
            "on_log": lambda event: None,
        }
        try:
            reply = provider.decide([frame], "SYSTEM", "USER", context)
            # The same call through the worker must produce the same text: the
            # proxy neither parses nor rewrites the child's reply.
            expected = worker.call(
                "vlm.decide",
                {
                    "frames": encode_frames([frame]),
                    "system_prompt": "SYSTEM",
                    "user_prompt": "USER",
                    "context": jsonable_context(context),
                },
            )
            self.assertEqual(reply, expected)

            data = _json_reply(reply)
            self.assertEqual(data["backend"], "echo")
            self.assertEqual(data["method"], "decide")
            self.assertEqual(data["system_prompt"], "SYSTEM")
            self.assertEqual(data["user_prompt"], "USER")
            self.assertEqual(data["frames"][0]["shape"], [3, 12, 16])
            self.assertEqual(data["frames"][0]["dtype"], "torch.float32")
            self.assertGreaterEqual(data["frames"][0]["min"], 0.0)
            self.assertLessEqual(data["frames"][0]["max"], 1.0)
            sent = data["context"]
            self.assertNotIn("on_log", sent, "callables must not cross the boundary")
            self.assertEqual(sent["position_flu"], [1.0, 2.0, 3.0])
            self.assertEqual(sent["np_scalar"], 0.5)
            self.assertEqual(sent["tensor"], [1, 2])
            self.assertEqual(sent["nested"], {"pair": [4, 5]})
            self.assertIsNone(sent["nan"])

            tools_reply = provider.decide_with_tools(
                [frame], "SYS", "USR", "TOOL CATALOGUE", {"step": 7}
            )
            tools_data = _json_reply(tools_reply)
            self.assertEqual(tools_data["method"], "decide_with_tools")
            self.assertEqual(tools_data["tools_text"], "TOOL CATALOGUE")
            self.assertEqual(tools_data["context"], {"step": 7})
            self.assertEqual(tools_data["frames"][0]["shape"], [3, 12, 16])
            self.assertEqual(provider.pid, pid)
        finally:
            report = provider.close()

        self.assertTrue(report["stopped"])
        self.assertFalse(report["already_closed"])
        self.assertFalse(worker.alive)
        _assert_pid_gone(self, pid)

        # close() is idempotent, and a closed provider refuses calls instead of
        # silently producing a fake reply.
        second = provider.close()
        self.assertTrue(second["already_closed"])
        with self.assertRaises(WorkerModelError) as caught:
            provider.decide([torch.zeros(3, 2, 2)], "s", "u")
        self.assertFalse(caught.exception.to_dict()["ok"])

    def test_worker_protocol_error_is_structured_and_worker_survives(self):
        from vlm_nav.plugins.worker_model import RemoteModelProvider

        worker = self._start("crash")
        provider = RemoteModelProvider(worker)
        pid = worker.pid
        try:
            with self.assertRaises(WorkerModelError) as caught:
                provider.decide(torch.zeros(3, 4, 4), "s", "u")
            data = caught.exception.to_dict()
            self.assertFalse(data["ok"])
            self.assertEqual(data["method"], "vlm.decide")
            self.assertEqual(data["pid"], pid)
            self.assertIn("fake backend crash", data["worker_error"])
            self.assertTrue(worker.alive, "a backend error must not kill the process")
        finally:
            provider.close()
        self.assertFalse(worker.alive)
        _assert_pid_gone(self, pid)

    def test_worker_crash_returns_structured_error(self):
        from vlm_nav.plugins.worker_model import RemoteModelProvider

        worker = self._start("fatal")
        provider = RemoteModelProvider(worker)
        pid = worker.pid
        try:
            with self.assertRaises(WorkerModelError) as caught:
                provider.decide(torch.zeros(3, 4, 4), "s", "u")
            data = caught.exception.to_dict()
            self.assertFalse(data["ok"])
            self.assertEqual(data["method"], "vlm.decide")
            self.assertEqual(data["pid"], pid)
            self.assertEqual(data["returncode"], 17)
            self.assertTrue(data["worker_error"])
            self.assertIn(data["state"], ("exited", "failed", "running"))
            self.assertFalse(worker.alive)
        finally:
            provider.close()
        _assert_pid_gone(self, pid)

    def test_externally_killed_worker_returns_structured_error(self):
        from vlm_nav.plugins.worker_model import RemoteModelProvider

        worker = self._start("echo")
        provider = RemoteModelProvider(worker)
        pid = worker.pid
        try:
            os.kill(pid, signal.SIGKILL)
            deadline = time.monotonic() + 5.0
            while worker.alive and time.monotonic() < deadline:
                time.sleep(0.01)
            with self.assertRaises(WorkerModelError) as caught:
                provider.decide(torch.zeros(3, 4, 4), "s", "u")
            self.assertEqual(caught.exception.to_dict()["pid"], pid)
        finally:
            provider.close()
        _assert_pid_gone(self, pid)


class WorkerModelPluginTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "plugins"
        self.host = PluginHost(root_dir=self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_manifest_and_command(self):
        plugin = WorkerModelProviderPlugin(
            backend_spec="hf",
            model_path="/models/qwen3-vl",
            device="cuda:0",
            dtype="float16",
            quantization="8bit",
            python="/custom/python",
            max_new_tokens=256,
        )
        self.assertEqual(plugin.manifest.id, PLUGIN_ID)
        self.assertEqual(plugin.manifest.provides, (MODEL_SERVICE,))
        self.assertEqual(plugin.manifest.background_tasks, ("worker",))
        self.assertEqual(plugin.manifest.isolation, "inproc")

        command = plugin.worker_command()

        def value(flag: str) -> str:
            return command[command.index(flag) + 1]

        self.assertEqual(command[:4], (
            "/custom/python", "-u", "-m", "vlm_nav.worker_model",
        ))
        self.assertEqual(value("--backend-spec"), "hf")
        self.assertEqual(value("--model-path"), "/models/qwen3-vl")
        self.assertEqual(value("--device"), "cuda:0")
        self.assertEqual(value("--dtype"), "float16")
        self.assertEqual(value("--quantization"), "8bit")
        self.assertEqual(value("--max-new-tokens"), "256")

        spec = plugin.worker_spec(None)
        self.assertIsInstance(spec, WorkerSpec)
        self.assertEqual(spec.command, command)
        self.assertEqual(spec.startup_timeout_s, 300.0)
        self.assertEqual(spec.request_timeout_s, 120.0)

        fake = WorkerModelProviderPlugin(fake="echo")
        fake_command = fake.worker_command()
        self.assertEqual(fake_command[fake_command.index("--fake") + 1], "echo")
        self.assertNotIn("--backend-spec", fake_command)

        override = WorkerSpec(command=(sys.executable, "-m", "vlm_nav.worker_echo"))
        pinned = WorkerModelProviderPlugin(spec=override)
        self.assertIs(pinned.worker_spec(None), override)

    def test_activate_provides_service_and_deactivate_reaps_worker(self):
        plugin = WorkerModelProviderPlugin(fake="echo")
        self.host.install(plugin)
        report = self.host.activate(PLUGIN_ID)
        self.assertEqual(report.status, "active")
        self.assertTrue(self.host.services.has(MODEL_SERVICE))

        provider = self.host.services.get(MODEL_SERVICE)
        self.assertIsInstance(provider, type(plugin.provider))
        self.assertIs(provider.worker, plugin.worker)
        pid = provider.pid
        self.assertTrue(plugin.worker.alive)

        reply = provider.decide(torch.zeros(3, 8, 9), "SYSTEM", "USER")
        self.assertEqual(_json_reply(reply)["frames"][0]["shape"], [3, 8, 9])

        log = self.root / PLUGIN_ID / "cache" / "model_worker.log"
        self.assertTrue(log.exists(), "worker stderr is captured in the cache dir")

        report = self.host.deactivate(PLUGIN_ID)
        self.assertEqual(report.status, "inactive")
        self.assertFalse(self.host.services.has(MODEL_SERVICE))
        self.assertFalse(plugin.worker.alive)
        self.assertIsNone(plugin.provider)
        _assert_pid_gone(self, pid)

        # Calling the stale provider is a structured failure, not a new process.
        with self.assertRaises(WorkerModelError):
            provider.decide(torch.zeros(3, 8, 9), "s", "u")

    def test_plugin_runs_the_configured_backend_spec(self):
        # A non-fake spec exercises the full worker_command path without a model.
        plugin = WorkerModelProviderPlugin(backend_spec="mock:forward")
        self.host.install(plugin)
        self.host.activate(PLUGIN_ID)
        provider = self.host.services.get(MODEL_SERVICE)
        try:
            reply = provider.decide(torch.zeros(3, 4, 4), "system", "user")
            self.assertEqual(json.loads(reply)["action"], "move")
        finally:
            self.host.deactivate(PLUGIN_ID)
        self.assertFalse(plugin.worker.alive)

    def test_configure_failure_leaks_no_process(self):
        class FailingConfigure(WorkerModelProviderPlugin):
            def configure(self, worker, ctx):
                super().configure(worker, ctx)
                raise RuntimeError("configure failed while loading weights")

        plugin = FailingConfigure(fake="echo")
        self.host.install(plugin)
        with self.assertRaises(LifecycleError) as caught:
            self.host.activate(PLUGIN_ID)
        self.assertIn("configure failed", str(caught.exception))

        pid = plugin.worker.pid
        self.assertIsNotNone(pid)
        self.assertFalse(plugin.worker.alive)
        _assert_pid_gone(self, pid)
        self.assertEqual(self.host.status(PLUGIN_ID), "failed")
        self.assertFalse(self.host.services.has(MODEL_SERVICE))
        self.assertIsNone(self.host.record(PLUGIN_ID).ctx)

        report = self.host.uninstall(PLUGIN_ID)
        self.assertEqual(report.status, "uninstalled")
        self.assertFalse(plugin.worker.alive)


if __name__ == "__main__":
    unittest.main()
