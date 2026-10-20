"""Plugin lifecycle host for the drone task system (goal.md M0/M3 groundwork).

The host is deliberately small and server-agnostic: a plugin declares *what it
provides, needs and owns* in a :class:`PluginManifest`, receives a
:class:`PluginContext`, and registers its runtime contributions through that
context. Every registration returns a disposable handle, so ``deactivate`` can
revoke tools, prompts, event listeners and background tasks without unloading
the process.

The lifecycle separates three operations (goal.md section 3):

``deactivate``
    Revoke runtime contributions. The plugin package, its data and its mission
    history stay in place.
``uninstall``
    Deactivate, then remove the plugin's private cache. Mission artifacts are
    *not* plugin-private and are preserved; ``purge_data=True`` additionally
    removes the plugin's declared data directories (explicit data clearing).
``purge_data``
    Delete the plugin's persistent data directories; shared resources stay with
    the :class:`ResourceManager` until their last reference is released.

Failure is reported, not hidden: cleanup errors stay in the record as
``residual`` entries and :meth:`PluginHost.retry_cleanup` retries them. A
plugin that is being used by an active task cannot be deactivated unless the
caller explicitly forces it; the forced path calls the plugin's abort hooks
and records the interruption. Heavy ``worker`` isolation is declared in the
manifest but rejected for now - M1 only needs in-process plugins, and pretending
to manage a worker process we did not start would make the uninstall contract
untrue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..tools import ToolRegistry, ToolSpec

API_VERSION = "1"
MANIFEST_SCHEMA = "vlm_nav.plugin_manifest/1"

PLUGIN_INSTALLED = "installed"
PLUGIN_ACTIVATING = "activating"
PLUGIN_ACTIVE = "active"
PLUGIN_INACTIVE = "inactive"
PLUGIN_DEGRADED = "degraded"
PLUGIN_FAILED = "failed"
PLUGIN_UNINSTALLED = "uninstalled"

_TERMINAL_STATUSES = (PLUGIN_UNINSTALLED,)

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_CONTRACT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class LifecycleError(RuntimeError):
    """A refused or failed lifecycle operation.

    ``report`` carries the structured picture a caller needs to act on:
    dependents that block a deactivation, active tasks, residual cleanup work.
    """

    def __init__(
        self,
        message: str,
        *,
        report: "LifecycleReport | None" = None,
        dependents: Sequence[str] = (),
        active_tasks: int = 0,
    ) -> None:
        super().__init__(message)
        self.report = report
        self.dependents = tuple(dependents)
        self.active_tasks = int(active_tasks)

    def to_dict(self) -> dict:
        data: dict[str, Any] = {"error": str(self)}
        if self.dependents:
            data["dependents"] = list(self.dependents)
        if self.active_tasks:
            data["active_tasks"] = self.active_tasks
        if self.report is not None:
            data["report"] = self.report.to_dict()
        return data


def split_contract(text: str) -> tuple[str, str]:
    """Split ``name`` or ``name@major`` into a normalized (name, major) pair."""
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty service contract")
    name, _, version = raw.partition("@")
    name = name.strip()
    version = (version.strip() or API_VERSION).split(".", 1)[0]
    if not _CONTRACT_RE.match(name):
        raise ValueError(f"invalid service name: {raw!r}")
    if not version.isdigit():
        raise ValueError(f"invalid service version: {raw!r}")
    return name, version


def contracts_compatible(required: str, provided: str) -> bool:
    try:
        r_name, r_major = split_contract(required)
        p_name, p_major = split_contract(provided)
    except ValueError:
        return False
    return r_name == p_name and r_major == p_major


def _safe_relative(name: str) -> str:
    path = Path(str(name))
    if not str(name).strip() or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"plugin path must be a relative name: {name!r}")
    return str(path)


@dataclass(frozen=True)
class PluginManifest:
    """What a plugin declares before it is allowed to run.

    ``provides``/``requires`` use ``service@major`` (``service`` means major
    ``1``). ``tools``/``prompts``/``events``/``background_tasks`` are the names
    the plugin may register; the host rejects an undeclared registration, so the
    manifest stays an honest description of the running contribution.
    """

    id: str
    version: str = "0.1.0"
    api_version: str = API_VERSION
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    optional_requires: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    prompts: tuple[str, ...] = ()
    events: tuple[str, ...] = ()
    background_tasks: tuple[str, ...] = ()
    data_dirs: tuple[str, ...] = ()
    cache_dirs: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    isolation: str = "inproc"
    protected: bool = False
    schema: str = MANIFEST_SCHEMA

    def validate(self) -> None:
        if not _ID_RE.match(str(self.id or "")):
            raise LifecycleError(f"invalid plugin id: {self.id!r}")
        if not str(self.version or "").strip():
            raise LifecycleError(f"plugin {self.id}: version is required")
        if self.schema != MANIFEST_SCHEMA:
            raise LifecycleError(
                f"plugin {self.id}: unknown manifest schema {self.schema!r}"
            )
        api_major = str(self.api_version).split(".", 1)[0]
        if api_major != API_VERSION:
            raise LifecycleError(
                f"plugin {self.id}: api_version {self.api_version!r} is not "
                f"compatible with host API {API_VERSION}"
            )
        if self.isolation not in ("inproc", "worker"):
            raise LifecycleError(
                f"plugin {self.id}: unknown isolation {self.isolation!r}"
            )
        if self.isolation == "worker":
            raise LifecycleError(
                f"plugin {self.id}: worker isolation is not implemented yet; "
                "an uninstall that cannot stop the worker would be a false promise"
            )
        for contract in tuple(self.provides) + tuple(self.requires) + tuple(
            self.optional_requires
        ):
            split_contract(contract)
        for group in (
            self.provides, self.requires, self.optional_requires, self.tools,
            self.prompts, self.events, self.background_tasks, self.data_dirs,
            self.cache_dirs, self.resources,
        ):
            if len(set(group)) != len(group):
                raise LifecycleError(f"plugin {self.id}: duplicate declaration in {group!r}")
        for name in tuple(self.data_dirs) + tuple(self.cache_dirs):
            _safe_relative(name)

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "id": self.id,
            "version": self.version,
            "api_version": self.api_version,
            "provides": list(self.provides),
            "requires": list(self.requires),
            "optional_requires": list(self.optional_requires),
            "tools": list(self.tools),
            "prompts": list(self.prompts),
            "events": list(self.events),
            "background_tasks": list(self.background_tasks),
            "data_dirs": list(self.data_dirs),
            "cache_dirs": list(self.cache_dirs),
            "resources": list(self.resources),
            "isolation": self.isolation,
            "protected": self.protected,
        }


class Disposable:
    """Idempotent undo handle returned by every registration."""

    def __init__(self, fn: Callable[[], Any], label: str = "") -> None:
        self._fn = fn
        self.label = label
        self._done = False
        self._lock = threading.Lock()

    @property
    def disposed(self) -> bool:
        return self._done

    def dispose(self) -> bool:
        with self._lock:
            if self._done:
                return False
            self._done = True
        self._fn()
        return True


class CleanupScope:
    """LIFO cleanup registry that keeps failed entries for a retry."""

    def __init__(self, owner: str = "") -> None:
        self.owner = owner
        self._items: list[tuple[str, Callable[[], Any]]] = []
        self._lock = threading.Lock()

    def add(self, fn: Callable[[], Any], label: str = "") -> Disposable:
        with self._lock:
            self._items.append((label or "cleanup", fn))
        return Disposable(self._dispose_one(fn), label or "cleanup")

    def _dispose_one(self, fn: Callable[[], Any]) -> Callable[[], Any]:
        def dispose() -> None:
            self._drop(fn)
            fn()

        return dispose

    def _drop(self, fn: Callable[[], Any]) -> None:
        with self._lock:
            for index in range(len(self._items) - 1, -1, -1):
                if self._items[index][1] is fn:
                    self._items.pop(index)
                    break

    def __len__(self) -> int:
        return len(self._items)

    def dispose_all(self) -> list[dict]:
        """Run every remaining cleanup in reverse order; return failures."""
        failures: list[dict] = []
        failed_items: list[tuple[str, Callable[[], Any]]] = []
        with self._lock:
            items = list(reversed(self._items))
            self._items.clear()
        for label, fn in items:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - cleanup reports, never hides
                failures.append({"label": label, "error": f"{type(exc).__name__}: {exc}"})
                failed_items.append((label, fn))
        if failed_items:
            with self._lock:
                # restore registration order so the next attempt is LIFO again
                self._items.extend(reversed(failed_items))
        return failures

    def labels(self) -> list[str]:
        with self._lock:
            return [label for label, _ in reversed(self._items)]


@dataclass
class ServiceEntry:
    contract: str
    name: str
    version: str
    provider: str
    value: Any

    def to_dict(self) -> dict:
        return {
            "contract": self.contract,
            "provider": self.provider,
            "version": self.version,
        }


class ServiceRegistry:
    """Interface-name based service table (plugins never import each other)."""

    def __init__(self) -> None:
        self._entries: dict[str, ServiceEntry] = {}
        self._lock = threading.Lock()

    def provide(self, provider: str, contract: str, value: Any) -> Disposable:
        name, version = split_contract(contract)
        with self._lock:
            existing = self._entries.get(name)
            if existing is not None:
                raise LifecycleError(
                    f"service {name!r} is already provided by {existing.provider}"
                )
            self._entries[name] = ServiceEntry(
                contract=contract, name=name, version=version,
                provider=provider, value=value,
            )

        def remove() -> None:
            with self._lock:
                current = self._entries.get(name)
                if current is not None and current.provider == provider:
                    del self._entries[name]

        return Disposable(remove, f"service:{name}")

    def has(self, contract: str) -> bool:
        name, version = split_contract(contract)
        with self._lock:
            entry = self._entries.get(name)
        return entry is not None and entry.version == version

    def get(self, contract: str, requester: str = "") -> Any:
        name, version = split_contract(contract)
        with self._lock:
            entry = self._entries.get(name)
        if entry is None or entry.version != version:
            who = f" for {requester}" if requester else ""
            raise LifecycleError(f"required service {contract!r} is not available{who}")
        return entry.value

    def resolve(
        self, requester: str, requires: Sequence[str], optional: Sequence[str] = ()
    ) -> dict:
        resolved: dict[str, Any] = {}
        missing: list[str] = []
        for contract in requires:
            try:
                resolved[split_contract(contract)[0]] = self.get(contract, requester)
            except LifecycleError:
                missing.append(contract)
        if missing:
            raise LifecycleError(
                f"plugin {requester} is missing required services: {', '.join(missing)}"
            )
        for contract in optional:
            try:
                resolved[split_contract(contract)[0]] = self.get(contract, requester)
            except LifecycleError:
                continue
        return resolved

    def remove_provider(self, provider: str) -> list[str]:
        removed: list[str] = []
        with self._lock:
            for name, entry in list(self._entries.items()):
                if entry.provider == provider:
                    del self._entries[name]
                    removed.append(name)
        return removed

    def catalog(self) -> dict:
        with self._lock:
            return {name: entry.to_dict() for name, entry in sorted(self._entries.items())}


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[tuple[str, Callable[[dict], Any]]]] = {}
        self._lock = threading.Lock()

    def subscribe(self, event: str, callback: Callable[[dict], Any], owner: str = "") -> Disposable:
        with self._lock:
            self._subs.setdefault(event, []).append((owner, callback))

        def remove() -> None:
            with self._lock:
                entries = self._subs.get(event, [])
                for index, (sub_owner, fn) in enumerate(entries):
                    if fn is callback and sub_owner == owner:
                        entries.pop(index)
                        break
                if not entries:
                    self._subs.pop(event, None)

        return Disposable(remove, f"event:{event}")

    def emit(self, event: str, payload: Mapping | None = None) -> int:
        with self._lock:
            callbacks = [fn for _, fn in self._subs.get(event, [])]
        data = dict(payload or {})
        for callback in callbacks:
            callback(data)
        return len(callbacks)

    def subscription_count(self, event: str | None = None) -> int:
        with self._lock:
            if event is None:
                return sum(len(items) for items in self._subs.values())
            return len(self._subs.get(event, []))


@dataclass
class _SharedResource:
    value: Any
    refs: set[str] = field(default_factory=set)
    cleanup: Callable[[Any], Any] | None = None


class ResourceManager:
    """Reference-counted shared resources (models, renderers, caches).

    A shared resource is destroyed only when its *last* holder releases it, so
    uninstalling plugin A cannot delete weights plugin B is still using. Cleanup
    failures are kept and retryable instead of silently leaking.
    """

    def __init__(self) -> None:
        self._resources: dict[str, _SharedResource] = {}
        self._failures: list[dict] = []
        self._lock = threading.Lock()

    def acquire(
        self,
        owner: str,
        key: str,
        factory: Callable[[], Any] | None = None,
        cleanup: Callable[[Any], Any] | None = None,
    ) -> Any:
        if not str(key or "").strip():
            raise LifecycleError("resource key is required")
        # Held through the factory: two concurrent owners must get the same
        # object, and the second must not create a value nobody can release.
        with self._lock:
            entry = self._resources.get(key)
            if entry is None:
                value = factory() if factory is not None else None
                entry = _SharedResource(value=value, cleanup=cleanup)
                self._resources[key] = entry
            entry.refs.add(owner)
            return entry.value

    def release(self, owner: str, key: str) -> bool:
        """Drop one reference; return True when the resource was destroyed."""
        with self._lock:
            entry = self._resources.get(key)
            if entry is None or owner not in entry.refs:
                return False
            entry.refs.discard(owner)
            if entry.refs:
                return False
            del self._resources[key]
        if entry.cleanup is not None:
            try:
                entry.cleanup(entry.value)
            except Exception as exc:  # noqa: BLE001 - surfaced through residual
                with self._lock:
                    self._failures.append({
                        "resource": key,
                        "owner": owner,
                        "error": f"{type(exc).__name__}: {exc}",
                        "cleanup": entry.cleanup,
                        "value": entry.value,
                    })
        return True

    def release_all(self, owner: str) -> list[dict]:
        with self._lock:
            keys = [
                key for key, entry in self._resources.items() if owner in entry.refs
            ]
        for key in keys:
            self.release(owner, key)
        return self.failures(owner)

    def failures(self, owner: str | None = None) -> list[dict]:
        with self._lock:
            entries = list(self._failures)
        if owner is None:
            return [
                {k: v for k, v in item.items() if k not in ("cleanup", "value")}
                for item in entries
            ]
        return [
            {k: v for k, v in item.items() if k not in ("cleanup", "value")}
            for item in entries if item["owner"] == owner
        ]

    def retry_failures(self, owner: str) -> list[dict]:
        with self._lock:
            entries = [item for item in self._failures if item["owner"] == owner]
        remaining: list[dict] = []
        for item in entries:
            try:
                item["cleanup"](item["value"])
            except Exception as exc:  # noqa: BLE001 - keep for another retry
                item["error"] = f"{type(exc).__name__}: {exc}"
                remaining.append(item)
        with self._lock:
            self._failures = [item for item in self._failures if item not in entries]
            self._failures.extend(remaining)
        return [
            {k: v for k, v in item.items() if k not in ("cleanup", "value")}
            for item in remaining
        ]

    def refs(self, key: str) -> int:
        with self._lock:
            entry = self._resources.get(key)
            return len(entry.refs) if entry else 0

    def keys(self, owner: str | None = None) -> list[str]:
        with self._lock:
            if owner is None:
                return sorted(self._resources)
            return sorted(key for key, entry in self._resources.items() if owner in entry.refs)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                key: {"refs": sorted(entry.refs)}
                for key, entry in sorted(self._resources.items())
            }


class ToolRegistrar:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        owner: str,
        declared: Sequence[str],
        scope: CleanupScope,
    ) -> None:
        self._registry = registry
        self._owner = owner
        self._declared = frozenset(declared)
        self._scope = scope
        self._names: list[str] = []

    @property
    def names(self) -> list[str]:
        return list(self._names)

    def register(self, spec: ToolSpec) -> Disposable:
        if not isinstance(spec, ToolSpec):
            raise LifecycleError(f"plugin {self._owner}: register() expects a ToolSpec")
        if spec.name not in self._declared:
            raise LifecycleError(
                f"plugin {self._owner}: tool {spec.name!r} is not declared in the manifest"
            )
        self._registry.register(spec)
        self._names.append(spec.name)

        def remove() -> None:
            self._registry.unregister(spec.name)
            if spec.name in self._names:
                self._names.remove(spec.name)

        return self._scope.add(remove, f"tool:{spec.name}")

    def missing(self) -> list[str]:
        return sorted(self._declared - set(self._names))


class PromptRegistrar:
    def __init__(
        self,
        prompts: dict[str, tuple[str, str]],
        *,
        owner: str,
        declared: Sequence[str],
        scope: CleanupScope,
    ) -> None:
        self._prompts = prompts
        self._owner = owner
        self._declared = frozenset(declared)
        self._scope = scope
        self._names: list[str] = []

    @property
    def names(self) -> list[str]:
        return list(self._names)

    def register(self, name: str, text: str) -> Disposable:
        if name not in self._declared:
            raise LifecycleError(
                f"plugin {self._owner}: prompt {name!r} is not declared in the manifest"
            )
        if name in self._prompts:
            raise LifecycleError(f"prompt already registered: {name}")
        self._prompts[name] = (self._owner, text)
        self._names.append(name)

        def remove() -> None:
            current = self._prompts.get(name)
            if current is not None and current[0] == self._owner:
                del self._prompts[name]
            if name in self._names:
                self._names.remove(name)

        return self._scope.add(remove, f"prompt:{name}")

    def missing(self) -> list[str]:
        return sorted(self._declared - set(self._names))


class TaskRegistrar:
    def __init__(
        self,
        *,
        owner: str,
        declared: Sequence[str],
        scope: CleanupScope,
    ) -> None:
        self._owner = owner
        self._declared = frozenset(declared)
        self._scope = scope
        self._names: list[str] = []

    @property
    def names(self) -> list[str]:
        return list(self._names)

    def register(self, name: str, stop: Callable[[], Any]) -> Disposable:
        if name not in self._declared:
            raise LifecycleError(
                f"plugin {self._owner}: background task {name!r} is not declared"
            )
        self._names.append(name)

        def remove() -> None:
            stop()
            if name in self._names:
                self._names.remove(name)

        return self._scope.add(remove, f"task:{name}")


class PluginContext:
    """The only door a plugin gets into the host.

    ``services`` resolves declared dependencies, ``tools``/``prompts``/``events``/
    ``tasks`` register revocable contributions, and ``resources`` shares
    reference-counted objects with other plugins.
    """

    def __init__(
        self,
        *,
        host: "PluginHost",
        manifest: PluginManifest,
        services: ServiceRegistry,
        tools: ToolRegistrar,
        prompts: PromptRegistrar,
        events: EventBus,
        tasks: TaskRegistrar,
        resources: ResourceManager,
        signal_scope: CleanupScope,
        cleanup_scope: CleanupScope,
        data_dir: Path,
        cache_dir: Path,
    ) -> None:
        self.host = host
        self.manifest = manifest
        self.plugin_id = manifest.id
        self.services = services
        self.tools = tools
        self.prompts = prompts
        self.events = events
        self.tasks = tasks
        self.resources = resources
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self._signal_scope = signal_scope
        self._cleanup_scope = cleanup_scope
        self._activity_probes: list[Callable[[], int]] = []
        self._abort_hooks: list[Callable[[str], Any]] = []

    def subscribe(self, event: str, callback: Callable[[dict], Any]) -> Disposable:
        if event not in set(self.manifest.events):
            raise LifecycleError(
                f"plugin {self.plugin_id}: event {event!r} is not declared in the manifest"
            )
        disposable = self.events.subscribe(event, callback, self.plugin_id)
        self._signal_scope.add(disposable.dispose, f"event:{event}")
        return disposable

    def emit(self, event: str, payload: Mapping | None = None) -> int:
        if event not in set(self.manifest.events):
            raise LifecycleError(
                f"plugin {self.plugin_id}: event {event!r} is not declared in the manifest"
            )
        return self.events.emit(event, payload)

    def add_cleanup(self, fn: Callable[[], Any], label: str = "") -> Disposable:
        return self._cleanup_scope.add(fn, f"plugin:{label or 'cleanup'}")

    def activity_probe(self, fn: Callable[[], int]) -> None:
        self._activity_probes.append(fn)

    def abort(self, fn: Callable[[str], Any]) -> None:
        self._abort_hooks.append(fn)

    def active_tasks(self) -> int:
        total = 0
        for probe in self._activity_probes:
            try:
                total += max(0, int(probe()))
            except Exception:  # noqa: BLE001 - a broken probe is not a busy signal
                continue
        return total

    def abort_all(self, reason: str) -> None:
        for hook in self._abort_hooks:
            try:
                hook(reason)
            except Exception:  # noqa: BLE001 - abort must not block teardown
                continue

    def data_path(self, name: str | None = None) -> Path:
        if name is None:
            return self.data_dir
        _safe_relative(name)
        if name not in set(self.manifest.data_dirs):
            raise LifecycleError(
                f"plugin {self.plugin_id}: data dir {name!r} is not declared"
            )
        return self.data_dir / name

    def cache_path(self, name: str | None = None) -> Path:
        if name is None:
            return self.cache_dir
        _safe_relative(name)
        if name not in set(self.manifest.cache_dirs):
            raise LifecycleError(
                f"plugin {self.plugin_id}: cache dir {name!r} is not declared"
            )
        return self.cache_dir / name


class Plugin:
    """Base class; ``activate``/``deactivate`` may be omitted."""

    manifest: PluginManifest

    def activate(self, ctx: PluginContext) -> None:  # pragma: no cover - default
        return None

    def deactivate(self, ctx: PluginContext) -> None:  # pragma: no cover - default
        return None


@dataclass
class LifecycleReport:
    plugin_id: str
    action: str
    status: str
    removed: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    residual: list[dict] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)
    detail: str = ""

    @property
    def ok(self) -> bool:
        return not self.residual

    def to_dict(self) -> dict:
        return {
            "plugin_id": self.plugin_id,
            "action": self.action,
            "status": self.status,
            "ok": self.ok,
            "removed": list(self.removed),
            "kept": list(self.kept),
            "residual": list(self.residual),
            "dependents": list(self.dependents),
            "detail": self.detail,
        }


@dataclass
class PluginRecord:
    plugin: Any
    manifest: PluginManifest
    status: str = PLUGIN_INSTALLED
    ctx: PluginContext | None = None
    tool_scope: CleanupScope | None = None
    signal_scope: CleanupScope | None = None
    cleanup_scope: CleanupScope | None = None
    tools: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    services: list[str] = field(default_factory=list)
    installed_at: float = field(default_factory=time.time)
    activated_at: float | None = None
    deactivated_at: float | None = None
    interrupted_tasks: int = 0
    error: str = ""
    last_report: LifecycleReport | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.manifest.id,
            "version": self.manifest.version,
            "status": self.status,
            "provides": list(self.manifest.provides),
            "requires": list(self.manifest.requires),
            "optional_requires": list(self.manifest.optional_requires),
            "tools": list(self.tools),
            "prompts": list(self.prompts),
            "services": list(self.services),
            "installed_at": self.installed_at,
            "activated_at": self.activated_at,
            "deactivated_at": self.deactivated_at,
            "interrupted_tasks": self.interrupted_tasks,
            "error": self.error,
        }


class PluginHost:
    """Install/activate/deactivate/uninstall transaction host."""

    def __init__(
        self,
        *,
        root_dir: str | Path,
        tool_registry: ToolRegistry | None = None,
        event_bus: EventBus | None = None,
        services: ServiceRegistry | None = None,
        resources: ResourceManager | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.tools = tool_registry if tool_registry is not None else ToolRegistry()
        self.events = event_bus if event_bus is not None else EventBus()
        self.services = services if services is not None else ServiceRegistry()
        self.resources = resources if resources is not None else ResourceManager()
        self._records: dict[str, PluginRecord] = {}
        self._prompts: dict[str, tuple[str, str]] = {}
        self._revision = 0
        self._shutdown = False
        self._lock = threading.RLock()

    # -- inspection ------------------------------------------------------- #

    def record(self, plugin_id: str) -> PluginRecord:
        record = self._records.get(plugin_id)
        if record is None or record.status == PLUGIN_UNINSTALLED:
            raise LifecycleError(f"unknown plugin: {plugin_id}")
        return record

    def status(self, plugin_id: str) -> str:
        return self.record(plugin_id).status

    def plugins(self, *, include_uninstalled: bool = False) -> list[dict]:
        return [
            record.to_dict()
            for record in self._records.values()
            if include_uninstalled or record.status != PLUGIN_UNINSTALLED
        ]

    def prompts(self) -> dict[str, str]:
        return {name: text for name, (_, text) in sorted(self._prompts.items())}

    def catalog(self) -> dict:
        """The published capability catalogue (stable revision on change)."""
        plugins = [
            {
                **record.to_dict(),
                "data_dir": str(self.root_dir / record.manifest.id),
            }
            for record in self._records.values()
            if record.status != PLUGIN_UNINSTALLED
        ]
        return {
            "revision": self._revision,
            "plugins": plugins,
            "services": self.services.catalog(),
            "tools": self.tools.names(),
            "prompts": sorted(self._prompts),
            "resources": self.resources.snapshot(),
        }

    # -- install / activate ----------------------------------------------- #

    def install(self, plugin: Any) -> LifecycleReport:
        manifest = getattr(plugin, "manifest", None)
        if not isinstance(manifest, PluginManifest):
            raise LifecycleError("plugin must expose a PluginManifest as .manifest")
        manifest.validate()
        with self._lock:
            if self._shutdown:
                raise LifecycleError("plugin host is shut down")
            existing = self._records.get(manifest.id)
            if existing is not None and existing.status != PLUGIN_UNINSTALLED:
                raise LifecycleError(
                    f"plugin {manifest.id} is already {existing.status}"
                )
            missing = self._missing_contracts(manifest.requires)
            if missing:
                raise LifecycleError(
                    f"plugin {manifest.id} requires unavailable services: "
                    f"{', '.join(missing)}"
                )
            record = PluginRecord(plugin=plugin, manifest=manifest)
            self._records[manifest.id] = record
            self._bump()
        return LifecycleReport(
            plugin_id=manifest.id, action="install", status=record.status,
            detail="installed (not active)",
        )

    def activate(self, plugin_id: str) -> LifecycleReport:
        with self._lock:
            record = self.record(plugin_id)
            if record.status == PLUGIN_ACTIVE:
                return LifecycleReport(
                    plugin_id, "activate", record.status, detail="already active"
                )
            if record.status == PLUGIN_ACTIVATING:
                raise LifecycleError(
                    f"service dependency cycle while activating {plugin_id}"
                )
            if record.status not in (PLUGIN_INSTALLED, PLUGIN_INACTIVE, PLUGIN_DEGRADED, PLUGIN_FAILED):
                raise LifecycleError(
                    f"plugin {plugin_id} cannot be activated from {record.status}"
                )
            record.status = PLUGIN_ACTIVATING
            try:
                self._ensure_providers(record.manifest.requires)
            except Exception:
                record.status = PLUGIN_FAILED if record.status == PLUGIN_ACTIVATING else record.status
                raise
            missing = self._missing_contracts(record.manifest.requires)
            if missing:
                record.status = PLUGIN_FAILED
                raise LifecycleError(
                    f"plugin {plugin_id} requires unavailable services: "
                    f"{', '.join(missing)}"
                )

            base = self.root_dir / plugin_id
            data_dir = base / "data"
            cache_dir = base / "cache"
            data_dir.mkdir(parents=True, exist_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            for name in record.manifest.data_dirs:
                (data_dir / name).mkdir(parents=True, exist_ok=True)
            for name in record.manifest.cache_dirs:
                (cache_dir / name).mkdir(parents=True, exist_ok=True)

            tool_scope = CleanupScope(f"{plugin_id}:tools")
            signal_scope = CleanupScope(f"{plugin_id}:signals")
            cleanup_scope = CleanupScope(f"{plugin_id}:cleanup")
            tool_registrar = ToolRegistrar(
                self.tools, owner=plugin_id, declared=record.manifest.tools,
                scope=tool_scope,
            )
            prompt_registrar = PromptRegistrar(
                self._prompts, owner=plugin_id, declared=record.manifest.prompts,
                scope=signal_scope,
            )
            task_registrar = TaskRegistrar(
                owner=plugin_id, declared=record.manifest.background_tasks,
                scope=signal_scope,
            )
            ctx = PluginContext(
                host=self,
                manifest=record.manifest,
                services=self.services,
                tools=tool_registrar,
                prompts=prompt_registrar,
                events=self.events,
                tasks=task_registrar,
                resources=self.resources,
                signal_scope=signal_scope,
                cleanup_scope=cleanup_scope,
                data_dir=data_dir,
                cache_dir=cache_dir,
            )
            record.ctx = ctx
            record.tool_scope = tool_scope
            record.signal_scope = signal_scope
            record.cleanup_scope = cleanup_scope
            record.tools = []
            record.prompts = []
            record.services = []
            record.error = ""

            try:
                record.plugin.activate(ctx)
                # The manifest is a contract: a declared service/prompt/tool
                # that never appeared means the plugin's description is a lie,
                # and the catalogue would be wrong.
                provided = set()
                for contract in record.manifest.provides:
                    name = split_contract(contract)[0]
                    provided.add(name)
                    if not self.services.has(contract):
                        raise LifecycleError(
                            f"plugin {plugin_id} declared service {contract!r} "
                            "but did not provide it"
                        )
                record.services = sorted(provided)
                missing_tools = tool_registrar.missing()
                if missing_tools:
                    raise LifecycleError(
                        f"plugin {plugin_id} declared tools that were never "
                        f"registered: {', '.join(missing_tools)}"
                    )
                missing_prompts = prompt_registrar.missing()
                if missing_prompts:
                    raise LifecycleError(
                        f"plugin {plugin_id} declared prompts that were never "
                        f"registered: {', '.join(missing_prompts)}"
                    )
            except Exception as exc:  # noqa: BLE001 - roll back and report
                residual = self._dispose_record(record, abort_reason="activation failed")
                record.status = PLUGIN_FAILED
                record.error = f"{type(exc).__name__}: {exc}"
                self._bump()
                report = LifecycleReport(
                    plugin_id, "activate", record.status,
                    residual=residual, detail=record.error,
                )
                record.last_report = report
                if isinstance(exc, LifecycleError):
                    raise
                raise LifecycleError(
                    f"plugin {plugin_id} activation failed: {record.error}",
                    report=report,
                ) from exc

            record.tools = tool_registrar.names
            record.prompts = prompt_registrar.names
            record.status = PLUGIN_ACTIVE
            record.activated_at = time.time()
            record.deactivated_at = None
            self._bump()
            report = LifecycleReport(
                plugin_id, "activate", record.status,
                detail=f"active with {len(record.tools)} tools",
            )
            record.last_report = report
            return report

    def _ensure_providers(self, contracts: Sequence[str]) -> None:
        """Activate installed providers of required contracts, transitively."""
        seen: set[str] = set()

        def ensure(contract: str) -> None:
            if self.services.has(contract):
                return
            provider = self._provider_of(contract)
            if provider is None:
                return
            if provider in seen:
                raise LifecycleError(
                    f"service dependency cycle while activating {provider}"
                )
            seen.add(provider)
            if self._records[provider].status != PLUGIN_ACTIVE:
                self.activate(provider)

        for contract in contracts:
            ensure(contract)

    def _provider_of(self, contract: str) -> str | None:
        name, _ = split_contract(contract)
        for record in self._records.values():
            if record.status in _TERMINAL_STATUSES:
                continue
            for provided in record.manifest.provides:
                if split_contract(provided)[0] == name:
                    return record.manifest.id
        return None

    def _missing_contracts(self, contracts: Sequence[str]) -> list[str]:
        missing: list[str] = []
        for contract in contracts:
            if self.services.has(contract):
                continue
            provider = self._provider_of(contract)
            if provider is None:
                missing.append(contract)
        return missing

    # -- deactivate / uninstall ------------------------------------------- #

    def deactivate(self, plugin_id: str, *, force: bool = False) -> LifecycleReport:
        with self._lock:
            record = self.record(plugin_id)
            if record.manifest.protected:
                raise LifecycleError(
                    f"plugin {plugin_id} is protected and cannot be deactivated"
                )
            if record.status != PLUGIN_ACTIVE:
                return LifecycleReport(
                    plugin_id, "deactivate", record.status,
                    detail="not active; nothing to revoke",
                )
            dependents = self._active_dependents(plugin_id)
            if dependents:
                if not force:
                    raise LifecycleError(
                        f"plugin {plugin_id} is required by active plugins: "
                        f"{', '.join(dependents)}",
                        dependents=dependents,
                    )
                for dependent in dependents:
                    self.deactivate(dependent, force=True)

            busy = record.ctx.active_tasks() if record.ctx is not None else 0
            if busy:
                if not force:
                    raise LifecycleError(
                        f"plugin {plugin_id} has {busy} active task(s); finish or "
                        "force deactivation so the interruption is recorded",
                        active_tasks=busy,
                    )
                record.ctx.abort_all("plugin deactivated")
                record.interrupted_tasks += busy

            revoked_tools = len(record.tools)
            residual = self._dispose_record(record, abort_reason="plugin deactivated")
            record.status = PLUGIN_DEGRADED if residual else PLUGIN_INACTIVE
            record.deactivated_at = time.time()
            self._bump()
            kept = [
                str(self.root_dir / plugin_id),
            ]
            report = LifecycleReport(
                plugin_id, "deactivate", record.status,
                kept=kept,
                residual=residual,
                dependents=dependents,
                detail=(
                    f"revoked {revoked_tools} tools; data preserved"
                    if not residual else "revoked, but cleanup left residuals"
                ),
            )
            record.last_report = report
            return report

    def _active_dependents(self, plugin_id: str) -> list[str]:
        provided = {
            split_contract(contract)[0]
            for contract in self._records[plugin_id].manifest.provides
        }
        dependents: list[str] = []
        for other in self._records.values():
            if other.manifest.id == plugin_id or other.status != PLUGIN_ACTIVE:
                continue
            needed = {
                split_contract(contract)[0]
                for contract in other.manifest.requires
            }
            if needed & provided:
                dependents.append(other.manifest.id)
        return sorted(dependents)

    def _dispose_record(self, record: PluginRecord, *, abort_reason: str) -> list[dict]:
        residual: list[dict] = []
        if record.ctx is not None:
            try:
                record.ctx.abort_all(abort_reason)
            except Exception:  # noqa: BLE001 - abort hooks are best effort
                pass
        # 1. runtime contributions first: stop new tool calls and signals.
        for scope in (record.tool_scope, record.signal_scope):
            if scope is not None:
                residual.extend(scope.dispose_all())
        # 2. let the plugin release its own connections/state.
        if record.ctx is not None:
            try:
                record.plugin.deactivate(record.ctx)
            except Exception as exc:  # noqa: BLE001 - reported as residual
                residual.append({
                    "label": "plugin.deactivate",
                    "error": f"{type(exc).__name__}: {exc}",
                })
        # 3. remaining plugin-owned cleanups, services, shared resources.
        if record.cleanup_scope is not None:
            residual.extend(record.cleanup_scope.dispose_all())
        self.services.remove_provider(record.manifest.id)
        residual.extend(self.resources.release_all(record.manifest.id))
        record.ctx = None
        record.tools = []
        record.prompts = []
        record.services = []
        return residual

    def uninstall(self, plugin_id: str, *, purge_data: bool = False) -> LifecycleReport:
        with self._lock:
            record = self.record(plugin_id)
            if record.manifest.protected:
                raise LifecycleError(
                    f"plugin {plugin_id} is protected and cannot be uninstalled"
                )
            if record.status == PLUGIN_ACTIVE:
                self.deactivate(plugin_id)
            dependents = self._installed_dependents(plugin_id)
            if dependents:
                raise LifecycleError(
                    f"plugin {plugin_id} is required by installed plugins: "
                    f"{', '.join(dependents)}; uninstall them first",
                    dependents=dependents,
                )

            removed: list[str] = []
            kept: list[str] = []
            residual: list[dict] = []
            # A failed/degraded record may still own cleanup work; retry it
            # before declaring the uninstall finished.
            if record.tool_scope is not None and len(record.tool_scope):
                residual.extend(record.tool_scope.dispose_all())
            if record.signal_scope is not None and len(record.signal_scope):
                residual.extend(record.signal_scope.dispose_all())
            if record.cleanup_scope is not None and len(record.cleanup_scope):
                residual.extend(record.cleanup_scope.dispose_all())
            residual.extend(self.resources.release_all(plugin_id))
            self.services.remove_provider(plugin_id)

            base = self.root_dir / plugin_id
            cache_dir = base / "cache"
            data_dir = base / "data"
            if cache_dir.exists():
                try:
                    shutil.rmtree(cache_dir)
                    removed.append(str(cache_dir))
                except Exception as exc:  # noqa: BLE001 - report the leftover
                    residual.append({
                        "label": "cache", "path": str(cache_dir),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
            if purge_data and data_dir.exists():
                try:
                    shutil.rmtree(data_dir)
                    removed.append(str(data_dir))
                except Exception as exc:  # noqa: BLE001 - report the leftover
                    residual.append({
                        "label": "data", "path": str(data_dir),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
            elif data_dir.exists():
                kept.append(str(data_dir))
            if base.exists() and not any(base.iterdir()):
                try:
                    base.rmdir()
                    removed.append(str(base))
                except OSError:
                    pass
            elif base.exists():
                kept.append(str(base))

            record.status = PLUGIN_UNINSTALLED
            record.deactivated_at = record.deactivated_at or time.time()
            record.error = "" if not residual else "residual cleanup"
            self._bump()
            report = LifecycleReport(
                plugin_id, "uninstall", record.status,
                removed=removed, kept=kept, residual=residual,
                detail=(
                    "uninstalled; mission history outside data_dir preserved"
                    if not purge_data else "uninstalled and private data purged"
                ),
            )
            record.last_report = report
            return report

    def _installed_dependents(self, plugin_id: str) -> list[str]:
        provided = {
            split_contract(contract)[0]
            for contract in self._records[plugin_id].manifest.provides
        }
        dependents: list[str] = []
        for other in self._records.values():
            if other.manifest.id == plugin_id:
                continue
            if other.status in _TERMINAL_STATUSES:
                continue
            needed = {
                split_contract(contract)[0]
                for contract in other.manifest.requires
            }
            if needed & provided:
                dependents.append(other.manifest.id)
        return sorted(dependents)

    def purge_data(self, plugin_id: str) -> LifecycleReport:
        """Explicitly clear a plugin's private data, even after uninstall.

        Mission history (photos, state files) lives outside ``data_dir`` and is
        deliberately not touched by this operation.
        """
        with self._lock:
            record = self._records.get(plugin_id)
            if record is None:
                raise LifecycleError(f"unknown plugin: {plugin_id}")
            data_dir = self.root_dir / plugin_id / "data"
            removed: list[str] = []
            residual: list[dict] = []
            if data_dir.exists():
                try:
                    shutil.rmtree(data_dir)
                    removed.append(str(data_dir))
                except Exception as exc:  # noqa: BLE001
                    residual.append({
                        "label": "data", "path": str(data_dir),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
            self._bump()
            report = LifecycleReport(
                plugin_id, "purge_data", record.status,
                removed=removed, residual=residual,
            )
            record.last_report = report
            return report

    def retry_cleanup(self, plugin_id: str) -> LifecycleReport:
        with self._lock:
            record = self.record(plugin_id)
            residual: list[dict] = []
            for scope in (
                record.tool_scope, record.signal_scope, record.cleanup_scope
            ):
                if scope is not None and len(scope):
                    residual.extend(scope.dispose_all())
            # A retry may finish the teardown of a degraded record.
            self.services.remove_provider(plugin_id)
            residual.extend(self.resources.retry_failures(plugin_id))
            if record.status == PLUGIN_DEGRADED and not residual:
                record.status = PLUGIN_INACTIVE
            self._bump()
            report = LifecycleReport(
                plugin_id, "retry_cleanup", record.status, residual=residual,
            )
            record.last_report = report
            return report

    def shutdown(self) -> list[dict]:
        """Deactivate every active plugin in reverse activation order."""
        with self._lock:
            self._shutdown = True
            active = [
                record for record in self._records.values()
                if record.status == PLUGIN_ACTIVE
            ]
            active.sort(key=lambda r: r.activated_at or 0.0, reverse=True)
            reports: list[dict] = []
            for record in active:
                try:
                    reports.append(
                        self.deactivate(record.manifest.id, force=True).to_dict()
                    )
                except LifecycleError as exc:
                    reports.append({"plugin_id": record.manifest.id, **exc.to_dict()})
            return reports

    def _bump(self) -> None:
        self._revision += 1
