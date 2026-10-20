"""Infrastructure plugins named by goal.md section 2.

These are deliberately thin adapters, not a second implementation of the
runtime:

- ``resource.manager`` exposes the host's reference-counted
  :class:`~vlm_nav.plugins.ResourceManager` as a service so another plugin can
  acquire/release shared models or GPU memory without importing the host.
- ``world.service`` exposes a world *factory* (a world is episode-scoped, not a
  process singleton); the plugin's cleanup closes the last world it created.
- ``mcp.adapter`` exposes the existing ``vlm_nav.mcp_server`` service map and
  session builder. Its tool reports whether the optional ``mcp`` package is
  installed instead of failing activation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..tools import STATUS_OK, ToolContext, ToolResult, ToolSpec
from . import Plugin, PluginContext, PluginManifest

RESOURCE_SERVICE = "resource.manager@1"
WORLD_SERVICE = "world.service@1"
MCP_SERVICE = "mcp.adapter@1"

PLUGIN_VERSION = "0.1.0"


class ResourceManagerPlugin(Plugin):
    manifest = PluginManifest(
        id="resource.manager",
        version=PLUGIN_VERSION,
        provides=(RESOURCE_SERVICE,),
    )

    def activate(self, ctx: PluginContext) -> None:
        ctx.services.provide(self.manifest.id, RESOURCE_SERVICE, ctx.host.resources)


class WorldServicePlugin(Plugin):
    """Provide a callable world factory under ``world.service@1``."""

    manifest = PluginManifest(
        id="world.service",
        version=PLUGIN_VERSION,
        provides=(WORLD_SERVICE,),
    )

    def __init__(self, factory: Callable[[], Any]) -> None:
        if not callable(factory):
            raise TypeError("WorldServicePlugin requires a callable factory")
        self._factory = factory
        self._created: list[Any] = []

    def activate(self, ctx: PluginContext) -> None:
        def create() -> Any:
            world = self._factory()
            self._created.append(world)
            return world

        create.factory = self._factory  # type: ignore[attr-defined]
        ctx.services.provide(self.manifest.id, WORLD_SERVICE, create)

    def deactivate(self, ctx: PluginContext) -> None:
        created, self._created = self._created, []
        for world in created:
            close = getattr(world, "close", None)
            if callable(close):
                close()


@dataclass
class McpAdapter:
    available: bool
    services: Mapping[str, Any]
    build_session: Callable[..., Any] | None = None
    error: str = ""
    all_tools: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "services": {
                name: {"server": value[0], "tools": list(value[1])}
                for name, value in self.services.items()
            },
            "all_tools": list(self.all_tools),
            "error": self.error,
        }


class McpAdapterPlugin(Plugin):
    manifest = PluginManifest(
        id="mcp.adapter",
        version=PLUGIN_VERSION,
        provides=(MCP_SERVICE,),
        tools=("mcp_services",),
    )

    def activate(self, ctx: PluginContext) -> None:
        try:
            from .. import mcp_server

            adapter = McpAdapter(
                available=True,
                services=dict(mcp_server.SERVICES),
                build_session=mcp_server.build_session,
                all_tools=tuple(sorted(mcp_server.ALL_TOOL_NAMES)),
            )
        except Exception as exc:  # noqa: BLE001 - optional mcp dependency
            adapter = McpAdapter(
                available=False, services={}, error=f"{type(exc).__name__}: {exc}"
            )
        ctx.services.provide(self.manifest.id, MCP_SERVICE, adapter)
        ctx.tools.register(ToolSpec(
            name="mcp_services",
            description=(
                "List the MCP services, their server names and tool vocabulary, "
                "and whether the optional mcp package is available."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=lambda call_ctx, args: ToolResult(
                STATUS_OK, adapter.to_dict()
            ),
            side_effects=False,
            cacheable=False,
        ))
