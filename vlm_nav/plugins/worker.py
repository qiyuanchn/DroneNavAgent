"""Managed worker process and the plugin base that owns it (goal.md M3).

A heavy plugin (model weights, CUDA context, fragile dependency tree) may not be
safe to load into the agent process. :class:`ManagedWorker` starts such a plugin
as a child process in its own session/process group and speaks a JSON-lines RPC
protocol over stdin/stdout:

request  ``{"id": 1, "method": "ping", "params": {}}``
reply    ``{"id": 1, "ok": true, "result": ...}``
         ``{"id": 1, "ok": false, "error": "..."}``

The lifecycle is deliberately explicit about failure:

* ``start`` launches the process with ``start_new_session=True`` (so the whole
  process group can be signalled later) and waits for a ``ping`` handshake.
* ``call`` waits at most ``request_timeout_s``; a hung worker raises
  :class:`WorkerTimeout` instead of blocking the caller forever.
* ``stop`` sends ``shutdown``, closes stdin, waits, then escalates to SIGTERM
  and SIGKILL on the *process group* and reaps the child. The returned report
  says whether escalation was needed and what the child's return code was.
* stderr is captured to ``cache/worker.log`` (or a temporary file) so a crashed
  child leaves an explanation behind.

:class:`WorkerPlugin` is the lifecycle glue: it starts the worker during
``activate``, registers ``worker.stop`` as a declared background task, hands the
handle to ``configure(worker, ctx)``, and stops the process during
``deactivate``. If ``configure`` raises, the worker is stopped before the error
is re-raised so a failed activation cannot leak a process.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any, Mapping

from . import LifecycleError, Plugin, PluginContext, PluginManifest

WORKER_TASK_NAME = "worker"

WORKER_NEW = "new"
WORKER_STARTING = "starting"
WORKER_RUNNING = "running"
WORKER_STOPPING = "stopping"
WORKER_STOPPED = "stopped"
WORKER_FAILED = "failed"
WORKER_EXITED = "exited"

#: Reply messages are matched by id; a small out-of-order backlog is plenty.
_MAX_PENDING = 64


class WorkerError(RuntimeError):
    """A managed worker could not be started, called or stopped."""


class WorkerNotRunning(WorkerError):
    """``call`` was made without a live child process."""


class WorkerTimeout(WorkerError):
    """The worker did not answer within the request/startup timeout."""


class WorkerRemoteError(WorkerError):
    """The worker answered ``{"ok": false, "error": ...}``."""

    def __init__(self, message: str, *, method: str = "", error: str = "") -> None:
        super().__init__(message)
        self.method = method
        self.error = error


@dataclass(frozen=True)
class WorkerSpec:
    """How to launch one managed worker process.

    ``env`` entries are *added* to the inherited environment; the worker still
    sees ``PATH``/``PYTHONPATH`` and whatever the parent process needs to import
    its module.
    """

    command: tuple[str, ...]
    cwd: str | None = None
    env: Mapping[str, str] | None = None
    startup_timeout_s: float = 10.0
    request_timeout_s: float = 30.0
    stop_timeout_s: float = 5.0
    restart: bool = False

    def __post_init__(self) -> None:
        command = self.command
        if isinstance(command, (str, bytes)):
            raise WorkerError("worker command must be a sequence of arguments, not a string")
        try:
            tokens = tuple(str(item) for item in command)
        except TypeError as exc:
            raise WorkerError("worker command must be a sequence of arguments") from exc
        if not tokens:
            raise WorkerError("worker command must not be empty")
        object.__setattr__(self, "command", tokens)
        if self.cwd is not None:
            object.__setattr__(self, "cwd", str(self.cwd))
        if self.env is not None:
            object.__setattr__(self, "env", {
                str(key): str(value) for key, value in dict(self.env).items()
            })
        for label in ("startup_timeout_s", "request_timeout_s", "stop_timeout_s"):
            value = float(getattr(self, label))
            if not math.isfinite(value) or value <= 0:
                raise WorkerError(f"worker {label} must be finite and > 0")
            object.__setattr__(self, label, value)
        object.__setattr__(self, "restart", bool(self.restart))


class ManagedWorker:
    """A child process speaking JSON-lines RPC over stdin/stdout."""

    def __init__(
        self,
        spec: WorkerSpec,
        *,
        log_path: str | Path | None = None,
        name: str = "worker",
    ) -> None:
        if not isinstance(spec, WorkerSpec):
            raise WorkerError("ManagedWorker requires a WorkerSpec")
        self.spec = spec
        self.name = str(name or "worker")
        self._log_path = Path(log_path) if log_path is not None else None
        self._stderr_handle: Any = None
        self._proc: subprocess.Popen | None = None
        self._pid: int | None = None
        self._state = WORKER_NEW
        self._reader: threading.Thread | None = None
        self._queue: queue.Queue = queue.Queue()
        self._pending: dict[int, dict] = {}
        self._pending_lock = threading.Lock()
        self._stdin_lock = threading.Lock()
        self._call_lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._ids = itertools.count(1)
        self._intentional_stop = False
        self._protocol_errors: list[str] = []

    # -- introspection ---------------------------------------------------- #

    @property
    def pid(self) -> int | None:
        """The pid of the most recently started child (kept after it exits)."""
        return self._pid

    @property
    def state(self) -> str:
        if (
            self._state == WORKER_RUNNING
            and self._proc is not None
            and self._proc.poll() is not None
        ):
            return WORKER_EXITED
        return self._state

    @property
    def alive(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    @property
    def returncode(self) -> int | None:
        proc = self._proc
        return proc.returncode if proc is not None else None

    @property
    def log_path(self) -> Path | None:
        return self._log_path

    @property
    def protocol_errors(self) -> list[str]:
        return list(self._protocol_errors)

    def logs(self) -> str:
        """Captured stderr, or ``""`` when no log file exists yet."""
        path = self._log_path
        if path is None:
            return ""
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    # -- lifecycle -------------------------------------------------------- #

    def start(self) -> "ManagedWorker":
        with self._lifecycle_lock:
            if self.alive:
                return self
            self._cleanup_previous()
            self._intentional_stop = False
            self._state = WORKER_STARTING
            self._stderr_handle = self._open_log()
            environment = dict(os.environ)
            if self.spec.env:
                environment.update(self.spec.env)
            try:
                proc = subprocess.Popen(
                    list(self.spec.command),
                    cwd=self.spec.cwd,
                    env=environment,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=self._stderr_handle,
                    start_new_session=True,
                    text=True,
                    bufsize=1,
                    encoding="utf-8",
                    errors="replace",
                )
            except Exception as exc:  # noqa: BLE001 - reported as WorkerError
                self._close_log()
                self._state = WORKER_FAILED
                raise WorkerError(
                    f"cannot start worker {self.name}: {type(exc).__name__}: {exc}"
                ) from exc
            self._proc = proc
            self._pid = proc.pid
            self._queue = queue.Queue()
            with self._pending_lock:
                self._pending.clear()
            self._reader = threading.Thread(
                target=self._reader_loop,
                args=(proc.stdout, self._queue),
                name=f"{self.name}-reader",
                daemon=True,
            )
            self._reader.start()
            try:
                self._request("ping", {}, self.spec.startup_timeout_s)
            except Exception as exc:  # noqa: BLE001 - a bad handshake is fatal
                self._stop_process_locked(force=True)
                self._state = WORKER_FAILED
                raise WorkerError(
                    f"worker {self.name} failed its ping handshake: {exc}"
                ) from exc
            self._state = WORKER_RUNNING
            return self

    def call(
        self, method: str, params: Mapping | None = None, timeout: float | None = None
    ) -> Any:
        """Send one request and return its ``result`` (or raise)."""
        method = str(method or "").strip()
        if not method:
            raise WorkerError("worker method is required")
        self._ensure_alive()
        with self._call_lock:
            return self._request(method, params or {}, timeout)

    def stop(self, timeout: float | None = None, *, force: bool = False) -> dict:
        """Stop the child: shutdown -> stdin EOF -> SIGTERM -> SIGKILL -> reap.

        ``force=True`` skips the graceful ``shutdown`` request. ``timeout``
        overrides ``spec.stop_timeout_s`` for each escalation step.
        """
        with self._lifecycle_lock:
            return self._stop_process_locked(timeout=timeout, force=force)

    # -- internals -------------------------------------------------------- #

    def _ensure_alive(self) -> None:
        if self.alive:
            return
        if (
            self.spec.restart
            and not self._intentional_stop
            and self._state == WORKER_RUNNING
        ):
            self.start()
            return
        raise WorkerNotRunning(f"worker {self.name} is not running")

    def _stop_process_locked(
        self, timeout: float | None = None, *, force: bool = False
    ) -> dict:
        self._intentional_stop = True
        self._state = WORKER_STOPPING
        proc = self._proc
        pid = self._pid or 0
        if proc is None:
            self._state = WORKER_STOPPED
            return {"stopped": True, "pid": 0, "forced": False, "returncode": None}
        budget = float(self.spec.stop_timeout_s if timeout is None else timeout)
        budget = max(0.05, budget)
        forced = bool(force)
        returncode = proc.poll()
        if returncode is None and not force:
            try:
                self._request("shutdown", {}, budget)
            except WorkerError:
                pass
            finally:
                self._close_stdin()
            returncode = self._wait_for_exit(budget)
        if returncode is None:
            forced = True
            self._close_stdin()
            self._signal_group(signal.SIGTERM)
            returncode = self._wait_for_exit(budget)
            if returncode is None:
                self._signal_group(signal.SIGKILL)
                returncode = self._wait_for_exit(budget)
        if returncode is None:
            # Last resort; the group signal above should have been enough.
            try:
                proc.kill()
            except OSError:
                pass
            returncode = self._wait_for_exit(budget)
        self._join_reader()
        self._close_stdout()
        self._close_log()
        stopped = returncode is not None
        self._state = WORKER_STOPPED if stopped else WORKER_FAILED
        return {
            "stopped": stopped,
            "pid": pid,
            "forced": forced,
            "returncode": returncode,
        }

    def _request(self, method: str, params: Mapping | None, timeout: float | None) -> Any:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise WorkerNotRunning(f"worker {self.name} is not running")
        if proc.poll() is not None:
            raise WorkerError(
                f"worker {self.name} exited with code {proc.returncode}"
            )
        request_id = next(self._ids)
        payload = json.dumps(
            {"id": request_id, "method": method, "params": dict(params or {})},
            ensure_ascii=False,
        )
        try:
            with self._stdin_lock:
                proc.stdin.write(payload + "\n")
                proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as exc:
            raise WorkerError(
                f"worker {self.name} is not accepting requests: {exc}"
            ) from exc
        budget = float(
            self.spec.request_timeout_s if timeout is None else timeout
        )
        deadline = time.monotonic() + max(0.0, budget)
        while True:
            with self._pending_lock:
                reply = self._pending.pop(request_id, None)
            if reply is not None:
                return self._unpack(reply, method)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerTimeout(
                    f"worker {self.name} request {method!r} timed out after {budget:g}s"
                )
            try:
                reply = self._queue.get(timeout=remaining)
            except queue.Empty:
                raise WorkerTimeout(
                    f"worker {self.name} request {method!r} timed out after {budget:g}s"
                )
            if reply is None:
                # EOF: keep the sentinel for any other waiter and report death.
                self._queue.put(None)
                returncode = proc.poll()
                if returncode is None:
                    try:
                        returncode = proc.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        returncode = None
                raise WorkerError(
                    f"worker {self.name} closed its stdout "
                    f"(returncode={returncode})"
                )
            if not isinstance(reply, dict):
                continue
            reply_id = reply.get("id")
            if reply_id == request_id:
                return self._unpack(reply, method)
            if reply_id is None:
                # Notification; the protocol has no subscription semantics yet.
                continue
            with self._pending_lock:
                if len(self._pending) >= _MAX_PENDING:
                    self._pending.pop(next(iter(self._pending)))
                self._pending[reply_id] = reply

    def _unpack(self, reply: Mapping, method: str) -> Any:
        if bool(reply.get("ok")):
            return reply.get("result")
        error = str(reply.get("error") or "unknown worker error")
        raise WorkerRemoteError(
            f"worker method {method!r} failed: {error}",
            method=method,
            error=error,
        )

    def _reader_loop(self, stream: Any, inbox: queue.Queue) -> None:
        try:
            if stream is None:
                return
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as exc:
                    self._protocol_errors.append(
                        f"invalid worker output: {line[:200]!r} ({exc})"
                    )
                    continue
                if not isinstance(message, dict):
                    self._protocol_errors.append(
                        f"worker output is not an object: {line[:200]!r}"
                    )
                    continue
                inbox.put(message)
        except (OSError, ValueError):
            # The parent closed the pipe while tearing the child down; that is
            # an ordinary end of stream, not an error to print from a thread.
            pass
        finally:
            inbox.put(None)

    def _wait_for_exit(self, budget: float) -> int | None:
        proc = self._proc
        if proc is None:
            return None
        try:
            return proc.wait(timeout=max(0.0, budget))
        except subprocess.TimeoutExpired:
            return None

    def _signal_group(self, sig: int) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            pgid = None
        try:
            if pgid is not None and pgid != os.getpgrp():
                os.killpg(pgid, sig)
                return
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            if sig == signal.SIGKILL:
                proc.kill()
            else:
                proc.terminate()
        except (ProcessLookupError, OSError):
            pass

    def _cleanup_previous(self) -> None:
        old = self._proc
        if old is not None:
            returncode = old.poll()
            if returncode is None:
                # A live worker must be stopped explicitly, never silently
                # replaced by a second process.
                raise WorkerError(f"worker {self.name} is still running")
            try:
                old.wait(timeout=0)
            except subprocess.TimeoutExpired:
                pass
        self._join_reader()
        self._close_stdout()
        self._close_log()
        self._proc = None

    def _open_log(self) -> Any:
        if self._log_path is None:
            descriptor, name = tempfile.mkstemp(
                prefix="vlm_nav_worker_", suffix=".log"
            )
            os.close(descriptor)
            self._log_path = Path(name)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        return open(
            self._log_path, "a", encoding="utf-8", errors="replace"
        )

    def _close_stdin(self) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            return
        try:
            proc.stdin.close()
        except (OSError, ValueError):
            pass

    def _close_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            proc.stdout.close()
        except (OSError, ValueError):
            pass

    def _close_log(self) -> None:
        handle, self._stderr_handle = self._stderr_handle, None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    def _join_reader(self) -> None:
        reader, self._reader = self._reader, None
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=2.0)

    def __enter__(self) -> "ManagedWorker":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop(force=exc_type is not None)
        return False

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ManagedWorker(name={self.name!r}, pid={self._pid!r}, "
            f"state={self.state!r})"
        )


class WorkerPlugin(Plugin):
    """Plugin base whose heavy work runs in a managed child process.

    Subclasses set ``manifest`` (declaring ``background_tasks=("worker",)``) and
    override :meth:`worker_spec` and/or :meth:`configure`. The host stays
    in-process (``isolation="inproc"``); the child process is owned here, and
    ``activate``/``deactivate``/``uninstall`` all funnel through
    :meth:`ManagedWorker.stop`, so a deactivated plugin never leaves a process
    behind.
    """

    manifest = PluginManifest(
        id="worker.base",
        version="0.1.0",
        background_tasks=(WORKER_TASK_NAME,),
        isolation="inproc",
    )

    def __init__(
        self,
        spec: WorkerSpec | None = None,
        *,
        log_name: str = "worker.log",
    ) -> None:
        self.spec = spec
        self.log_name = str(log_name or "worker.log")
        self.worker: ManagedWorker | None = None
        self._task_handle: Any = None

    # -- subclass hooks --------------------------------------------------- #

    def worker_spec(self, ctx: PluginContext) -> WorkerSpec:
        """Return the launch spec; the constructor ``spec`` wins when given."""
        if self.spec is not None:
            return self.spec
        raise LifecycleError(
            f"plugin {self.manifest.id}: no WorkerSpec configured"
        )

    def configure(self, worker: ManagedWorker, ctx: PluginContext) -> None:
        """Attach tools/services after the ping handshake. Default: no-op."""
        return None

    # -- lifecycle -------------------------------------------------------- #

    def activate(self, ctx: PluginContext) -> None:
        previous = self.worker
        if previous is not None and previous.alive:
            previous.stop()
        spec = self.worker_spec(ctx)
        if not isinstance(spec, WorkerSpec):
            raise LifecycleError(
                f"plugin {self.manifest.id}: worker_spec() must return a WorkerSpec"
            )
        log_path = ctx.cache_dir / self.log_name
        worker = ManagedWorker(spec, log_path=log_path, name=ctx.plugin_id)
        self.worker = worker
        registered = None
        try:
            worker.start()
            registered = ctx.tasks.register(WORKER_TASK_NAME, worker.stop)
            self._task_handle = registered
            self.configure(worker, ctx)
        except BaseException:
            self._task_handle = None
            if registered is not None:
                try:
                    registered.dispose()
                except Exception:  # noqa: BLE001 - stop again below
                    pass
            self._stop_worker(worker)
            raise

    def deactivate(self, ctx: PluginContext) -> None:
        handle, self._task_handle = self._task_handle, None
        worker = self.worker
        if worker is not None:
            self._stop_worker(worker)
        if handle is not None:
            try:
                handle.dispose()
            except Exception:  # noqa: BLE001 - cleanup is reported by the host
                pass

    # -- helpers ---------------------------------------------------------- #

    def _stop_worker(self, worker: ManagedWorker) -> None:
        try:
            worker.stop()
        except Exception:  # noqa: BLE001 - a failed graceful stop is retried hard
            try:
                worker.stop(force=True)
            except Exception:  # noqa: BLE001 - the host reports leftover work
                pass


__all__ = [
    "ManagedWorker",
    "WorkerError",
    "WorkerNotRunning",
    "WorkerPlugin",
    "WorkerRemoteError",
    "WorkerSpec",
    "WorkerTimeout",
    "WORKER_TASK_NAME",
]
