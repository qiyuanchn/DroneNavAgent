"""Worker-backed model provider plugin: ``model.provider@1`` in a child process.

The plugin keeps the VLM *weights and CUDA context* in a managed child process
(:mod:`vlm_nav.worker_model`) and exposes only a thin proxy in the agent
process. Consumers resolve the ordinary ``model.provider@1`` contract and see
the same ``decide``/``decide_with_tools`` interface as
:class:`vlm_nav.vlm.MockVLM`/:class:`vlm_nav.vlm.OpenAIVLM`:

    reply = provider.decide(frames, system_prompt, user_prompt, context)
    reply = provider.decide_with_tools(
        frames, system_prompt, user_prompt, tools_text="", context=None
    )

Frames are encoded as base64 PNGs on the manager side
(:func:`vlm_nav.worker_model.encode_frames`); the child decodes them back to
torch CHW ``[0, 1]`` tensors before calling its backend. The proxy returns the
child's reply text unchanged and never re-parses or rewrites it.

Failure is explicit: a dead/timed-out/erroring worker raises
:class:`WorkerModelError` (a :class:`~vlm_nav.plugins.worker.WorkerError`) with
a :meth:`WorkerModelError.to_dict` payload naming the method, pid and worker
state. It never fabricates a model reply. ``close()`` stops the worker, so a
caller that holds the provider directly can release the process even without
the plugin host.
"""

from __future__ import annotations

import json
import math
import sys
import time
from typing import Any, Mapping, Sequence

from ..tools import parse_tool_call
from ..vlm import VLMBackend
from ..worker_model import encode_frames
from . import PluginContext, PluginManifest
from .worker import (
    WORKER_TASK_NAME,
    ManagedWorker,
    WorkerError,
    WorkerPlugin,
    WorkerSpec,
)

MODEL_SERVICE = "model.provider@1"
WORKER_MODEL_PLUGIN_ID = "model.provider.worker"
DEFAULT_WORKER_MODULE = "vlm_nav.worker_model"
DEFAULT_LOG_NAME = "model_worker.log"


class WorkerModelError(WorkerError):
    """A remote model call failed; carries a structured report.

    ``WorkerError`` keeps the management code's existing catch semantics, and
    ``to_dict()`` is the machine-readable form the runtime/tool boundaries can
    log (``ok`` is always false: no reply text is invented).
    """

    def __init__(
        self,
        message: str,
        *,
        method: str = "",
        error: str = "",
        pid: int | None = None,
        state: str = "",
        returncode: int | None = None,
    ) -> None:
        super().__init__(message)
        self.method = str(method or "")
        self.error = str(error or "")
        self.pid = pid
        self.state = str(state or "")
        self.returncode = returncode

    def to_dict(self) -> dict:
        return {
            "ok": False,
            "error": str(self),
            "worker_error": self.error,
            "method": self.method,
            "pid": self.pid,
            "state": self.state,
            "returncode": self.returncode,
        }


# ---------------------------------------------------------------------------
# context serialization
# ---------------------------------------------------------------------------

_DROP = object()


def _jsonable(value: Any, depth: int = 0) -> Any:
    """Convert a runtime context value into JSON-safe data; drop callables."""
    if depth > 24:
        return str(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if callable(value):
        return _DROP
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            item = _jsonable(item, depth + 1)
            if item is not _DROP:
                converted[str(key)] = item
        return converted
    if isinstance(value, (list, tuple, set, frozenset)):
        items: list[Any] = []
        for item in value:
            item = _jsonable(item, depth + 1)
            if item is not _DROP:
                items.append(item)
        return items
    # numpy scalars/arrays and torch tensors expose tolist()/item(); prefer
    # them over the repr fallback so geometry survives the JSON boundary.
    for name in ("tolist", "item"):
        method = getattr(value, name, None)
        if callable(method):
            try:
                converted = method()
            except Exception:  # noqa: BLE001 - fall through to str()
                continue
            if converted is value:
                continue
            return _jsonable(converted, depth + 1)
    return str(value)


def jsonable_context(context: Any) -> Any:
    """The JSON-safe form of a VLM context (``on_log`` callbacks are dropped)."""
    if context is None:
        return None
    return _jsonable(context)


# ---------------------------------------------------------------------------
# manager-side proxy
# ---------------------------------------------------------------------------


def _frame_count(frames: Any) -> int:
    if isinstance(frames, (list, tuple)):
        return len(frames)
    return 1 if frames is not None else 0


def _reply_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if result is None:
        return ""
    if isinstance(result, (dict, list, tuple)):
        return json.dumps(result, ensure_ascii=False, default=str)
    return str(result)


class RemoteModelProvider(VLMBackend):
    """A ``vlm_nav.vlm`` backend whose model lives in a managed child process."""

    name = "worker"

    def __init__(
        self,
        worker: ManagedWorker,
        *,
        request_timeout_s: float | None = None,
        on_log: Any = None,
        close_worker: bool = True,
    ) -> None:
        if worker is None:
            raise ValueError("RemoteModelProvider requires a ManagedWorker")
        self.worker = worker
        self.request_timeout_s = request_timeout_s
        self.on_log = on_log
        self.close_worker = bool(close_worker)
        self._closed = False

    # -- introspection ---------------------------------------------------- #

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pid(self) -> int | None:
        return self.worker.pid

    # -- VLMBackend interface --------------------------------------------- #

    def decide(
        self,
        frames: Any,
        system_prompt: str,
        user_prompt: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        params = self._params(frames, system_prompt, user_prompt, context)
        return self._call("vlm.decide", params, context, _frame_count(frames))

    def decide_with_tools(
        self,
        frames: Any,
        system_prompt: str,
        user_prompt: str,
        tools_text: str = "",
        context: dict[str, Any] | None = None,
    ) -> str:
        params = self._params(frames, system_prompt, user_prompt, context)
        params["tools_text"] = str(tools_text or "")
        return self._call(
            "vlm.decide_with_tools", params, context, _frame_count(frames)
        )

    def close(self) -> dict:
        """Stop the worker (idempotent); returns the managed-worker report."""
        if self._closed:
            return {
                "stopped": not self.worker.alive,
                "pid": self.worker.pid or 0,
                "forced": False,
                "returncode": self.worker.returncode,
                "already_closed": True,
            }
        self._closed = True
        if not self.close_worker:
            return {
                "stopped": not self.worker.alive,
                "pid": self.worker.pid or 0,
                "forced": False,
                "returncode": self.worker.returncode,
                "already_closed": False,
                "close_worker": False,
            }
        report = dict(self.worker.stop())
        report["already_closed"] = False
        return report

    # -- internals -------------------------------------------------------- #

    def _params(
        self,
        frames: Any,
        system_prompt: str,
        user_prompt: str,
        context: Any,
    ) -> dict:
        return {
            "frames": encode_frames(frames),
            "system_prompt": str(system_prompt or ""),
            "user_prompt": str(user_prompt or ""),
            "context": jsonable_context(context),
        }

    def _call(
        self,
        method: str,
        params: Mapping[str, Any],
        context: Any,
        n_frames: int,
    ) -> str:
        if self._closed:
            raise WorkerModelError(
                "remote model provider is closed",
                method=method,
                error="provider closed",
                pid=self.worker.pid,
                state=self.worker.state,
                returncode=self.worker.returncode,
            )
        started = time.time()
        try:
            result = self.worker.call(
                method, dict(params), timeout=self.request_timeout_s
            )
        except WorkerError as exc:
            raise WorkerModelError(
                f"worker model call {method!r} failed: {exc}",
                method=method,
                error=str(exc),
                pid=self.worker.pid,
                state=self.worker.state,
                returncode=self.worker.returncode,
            ) from exc
        raw = _reply_text(result)
        parsed = parse_tool_call(raw)
        calls = [{"tool": parsed[0], "args": parsed[1]}] if parsed else []
        self._log_tool_call(
            "remote",
            raw,
            calls,
            started,
            context if isinstance(context, dict) else None,
            n_frames,
            extra={
                "worker_method": method,
                "worker_pid": self.worker.pid,
            },
        )
        return raw


# ---------------------------------------------------------------------------
# plugin
# ---------------------------------------------------------------------------


class WorkerModelProviderPlugin(WorkerPlugin):
    """Start a VLM worker during ``activate`` and publish ``model.provider@1``.

    ``backend_spec``/``model_path``/``device``/``dtype``/``quantization`` are
    turned into the child command; ``fake`` selects a model-free test backend
    (``echo``/``crash``/``fatal``). The worker is stopped by the inherited
    ``WorkerPlugin`` teardown on deactivate/uninstall, so the host's "no
    process left behind" contract holds even when ``configure`` fails.
    """

    manifest = PluginManifest(
        id=WORKER_MODEL_PLUGIN_ID,
        version="0.1.0",
        provides=(MODEL_SERVICE,),
        background_tasks=(WORKER_TASK_NAME,),
        isolation="inproc",
    )

    def __init__(
        self,
        *,
        backend_spec: str = "",
        model_path: str = "",
        device: str = "cpu",
        dtype: str = "bfloat16",
        quantization: str = "none",
        python: str = "",
        worker_module: str = DEFAULT_WORKER_MODULE,
        fake: str = "",
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        startup_timeout_s: float = 300.0,
        request_timeout_s: float = 120.0,
        stop_timeout_s: float = 10.0,
        max_new_tokens: int | None = None,
        log_name: str = DEFAULT_LOG_NAME,
        command: Sequence[str] | None = None,
        spec: WorkerSpec | None = None,
        on_log: Any = None,
    ) -> None:
        super().__init__(spec=spec, log_name=log_name)
        self.backend_spec = str(backend_spec or "")
        self.model_path = str(model_path or "")
        self.device = str(device or "cpu")
        self.dtype = str(dtype or "bfloat16")
        self.quantization = str(quantization or "none")
        self.python = str(python or sys.executable)
        self.worker_module = str(worker_module or DEFAULT_WORKER_MODULE)
        self.fake = str(fake or "")
        self.env = dict(env or {})
        self.cwd = str(cwd) if cwd is not None else None
        self.startup_timeout_s = float(startup_timeout_s)
        self.request_timeout_s = float(request_timeout_s)
        self.stop_timeout_s = float(stop_timeout_s)
        self.max_new_tokens = int(max_new_tokens) if max_new_tokens else None
        self.command = tuple(str(item) for item in command) if command else None
        self.on_log = on_log
        self.provider: RemoteModelProvider | None = None

    # -- WorkerPlugin hooks ----------------------------------------------- #

    def worker_command(self) -> tuple[str, ...]:
        """The child command (also used by tests/profile reporting)."""
        if self.command is not None:
            return self.command
        command = [self.python, "-u", "-m", self.worker_module]
        if self.fake:
            command.extend(["--fake", self.fake])
        else:
            spec = self.backend_spec or ("hf" if self.model_path else "mock:oracle")
            command.extend(["--backend-spec", spec])
        if self.model_path:
            command.extend(["--model-path", self.model_path])
        command.extend([
            "--device", self.device,
            "--dtype", self.dtype,
            "--quantization", self.quantization,
        ])
        if self.max_new_tokens:
            command.extend(["--max-new-tokens", str(self.max_new_tokens)])
        return tuple(command)

    def worker_spec(self, ctx: PluginContext) -> WorkerSpec:
        if self.spec is not None:
            return self.spec
        return WorkerSpec(
            command=self.worker_command(),
            cwd=self.cwd,
            env=self.env or None,
            startup_timeout_s=self.startup_timeout_s,
            request_timeout_s=self.request_timeout_s,
            stop_timeout_s=self.stop_timeout_s,
            restart=False,
        )

    def configure(self, worker: ManagedWorker, ctx: PluginContext) -> None:
        provider = RemoteModelProvider(
            worker,
            request_timeout_s=self.request_timeout_s,
            on_log=self.on_log,
        )
        ctx.services.provide(self.manifest.id, MODEL_SERVICE, provider)
        self.provider = provider

    def deactivate(self, ctx: PluginContext) -> None:
        provider, self.provider = self.provider, None
        try:
            if provider is not None:
                provider.close()
        finally:
            super().deactivate(ctx)


__all__ = [
    "DEFAULT_LOG_NAME",
    "DEFAULT_WORKER_MODULE",
    "MODEL_SERVICE",
    "RemoteModelProvider",
    "WORKER_MODEL_PLUGIN_ID",
    "WorkerModelError",
    "WorkerModelProviderPlugin",
    "jsonable_context",
]
