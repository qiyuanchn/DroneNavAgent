"""Adapter plugins around the pre-plugin tool modules.

These plugins keep the existing ``tool_impl_*`` registration functions working
under the lifecycle host. They declare exactly the tools those functions add,
so activation can verify the manifest and deactivation revokes every tool; a
module that grows a tool without updating the adapter here fails activation
instead of quietly widening what the plugin owns.

They register through :meth:`PluginContext.registry_view`, which forwards
``register`` to the plugin's declared-tool scope and reads/calls through the
host's shared registry. The VLM backend is optional: it is the
``model.provider@1`` service when another plugin provides one, otherwise the
tool modules run in their no-VLM mode (structured ``empty``/``uncertain``
answers, never a guess).
"""

from __future__ import annotations

from typing import Any, Callable

from ..tool_impl_navigation import register_navigation_tools
from ..tool_impl_perception import register_perception_tools
from ..tool_impl_task import register_task_tools
from . import Plugin, PluginContext, PluginManifest

PLUGIN_VERSION = "0.1.0"

PERCEPTION_SERVICE = "perception.tools@1"
NAVIGATION_SERVICE = "navigation.tools@1"
MEMORY_SERVICE = "memory.store@1"
MODEL_SERVICE = "model.provider@1"

PERCEPTION_TOOLS = (
    "observe",
    "inspect_region",
    "ground_target",
    "compare_candidates",
    "query_memory",
    "search_target",
)

NAVIGATION_TOOLS = (
    "propose_viewpoints",
    "check_path",
    "approach_candidate",
    "go_to_viewpoint",
    "return_to_observation",
    "navigate",
    "task_status",
    "cancel_task",
    "verify_goal",
    "finish",
    "batch_read",
    "confirm_candidate",
)

TASK_TOOLS = (
    "update_plan",
    "get_plan",
)


class PerceptionPlugin(Plugin):
    """Wrap ``register_perception_tools`` as an installable capability."""

    manifest = PluginManifest(
        id="perception.core",
        version=PLUGIN_VERSION,
        provides=(PERCEPTION_SERVICE,),
        optional_requires=(MODEL_SERVICE,),
        tools=PERCEPTION_TOOLS,
    )

    def __init__(self, *, vlm: Any = None) -> None:
        self.vlm = vlm
        self._ctx: PluginContext | None = None

    def activate(self, ctx: PluginContext) -> None:
        self._ctx = ctx
        backend = self.vlm
        if backend is None:
            backend = ctx.optional_service(MODEL_SERVICE)
        ctx.services.provide(self.manifest.id, PERCEPTION_SERVICE, self)
        register_perception_tools(ctx.registry_view(), vlm=backend)

    def deactivate(self, ctx: PluginContext) -> None:
        self._ctx = None


class NavigationPlugin(Plugin):
    """Wrap ``register_navigation_tools`` as an installable capability."""

    manifest = PluginManifest(
        id="navigation.core",
        version=PLUGIN_VERSION,
        provides=(NAVIGATION_SERVICE,),
        optional_requires=(MODEL_SERVICE,),
        tools=NAVIGATION_TOOLS,
    )

    def __init__(self, *, vlm: Any = None) -> None:
        self.vlm = vlm
        self._ctx: PluginContext | None = None

    def activate(self, ctx: PluginContext) -> None:
        self._ctx = ctx
        backend = self.vlm
        if backend is None:
            backend = ctx.optional_service(MODEL_SERVICE)
        ctx.services.provide(self.manifest.id, NAVIGATION_SERVICE, self)
        register_navigation_tools(ctx.registry_view(), vlm=backend)

    def deactivate(self, ctx: PluginContext) -> None:
        self._ctx = None


class TaskToolsPlugin(Plugin):
    """Wrap ``register_task_tools`` (``update_plan``/``get_plan``)."""

    manifest = PluginManifest(
        id="task.tools",
        version=PLUGIN_VERSION,
        tools=TASK_TOOLS,
    )

    def activate(self, ctx: PluginContext) -> None:
        register_task_tools(ctx.registry_view())


class MemoryStorePlugin(Plugin):
    """Provide a memory store built by ``factory``; close it on deactivate."""

    manifest = PluginManifest(
        id="memory.store",
        version=PLUGIN_VERSION,
        provides=(MEMORY_SERVICE,),
    )

    def __init__(self, factory: Callable[[], Any]) -> None:
        if not callable(factory):
            raise TypeError("MemoryStorePlugin requires a callable factory")
        self._factory = factory
        self.store: Any = None

    def activate(self, ctx: PluginContext) -> None:
        self.store = self._factory()
        ctx.services.provide(self.manifest.id, MEMORY_SERVICE, self.store)

    def deactivate(self, ctx: PluginContext) -> None:
        store, self.store = self.store, None
        close = getattr(store, "close", None)
        if callable(close):
            close()


class ModelProviderPlugin(Plugin):
    """Provide a model backend built by ``factory``; close it on deactivate."""

    manifest = PluginManifest(
        id="model.provider",
        version=PLUGIN_VERSION,
        provides=(MODEL_SERVICE,),
    )

    def __init__(self, factory: Callable[[], Any]) -> None:
        if not callable(factory):
            raise TypeError("ModelProviderPlugin requires a callable factory")
        self._factory = factory
        self.provider: Any = None

    def activate(self, ctx: PluginContext) -> None:
        self.provider = self._factory()
        ctx.services.provide(self.manifest.id, MODEL_SERVICE, self.provider)

    def deactivate(self, ctx: PluginContext) -> None:
        provider, self.provider = self.provider, None
        close = getattr(provider, "close", None)
        if callable(close):
            close()
