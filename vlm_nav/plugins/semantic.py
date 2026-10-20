"""Semantic navigation plugin: publishes ``semantic.service@1``.

The plugin is a thin lifecycle wrapper around
:class:`~vlm_nav.semantic.SemanticNavigationService`. It requires the
navigation and perception tool bundles to be published by the host and
registers one read-only tool, ``semantic_status``, that reports the service
contract and which of the semantic tools the active registry exposes.

The service itself is stateless; every method takes the caller's ``ctx``. The
plugin holds no mission state, so deactivate/uninstall only revoke the service
and the tool registration - nothing else is owned.
"""

from __future__ import annotations

from typing import Any

from ..semantic import (
    SEMANTIC_METHODS,
    SEMANTIC_SERVICE,
    SEMANTIC_TOOL_NAMES,
    SemanticNavigationService,
)
from ..tools import STATUS_FAILED, STATUS_OK, ToolContext, ToolResult, ToolSpec
from . import Plugin, PluginContext, PluginManifest

NAVIGATION_CONTRACT = "navigation.tools@1"
PERCEPTION_CONTRACT = "perception.tools@1"

SEMANTIC_STATUS_TOOL = "semantic_status"
SEMANTIC_TOOLS = (SEMANTIC_STATUS_TOOL,)

SEMANTIC_STATUS_DESCRIPTION = (
    "Report the semantic navigation service contract: the methods it offers "
    "(search/approach/confirm/verify/revisit/goto), the tools it drives, which "
    "of them are currently registered, and the retry bound. Read-only: it "
    "changes no pose and no task."
)

SEMANTIC_STATUS_SCHEMA = {
    "type": "object",
    "properties": {},
    "required": [],
}


class SemanticNavigationPlugin(Plugin):
    """Installable semantic-navigation capability."""

    manifest = PluginManifest(
        id="semantic.navigation",
        version="0.1.0",
        provides=(SEMANTIC_SERVICE,),
        requires=(NAVIGATION_CONTRACT, PERCEPTION_CONTRACT),
        tools=SEMANTIC_TOOLS,
    )

    def __init__(self, *, max_retries: int = 2) -> None:
        self.max_retries = max(0, int(max_retries))

    # -- lifecycle -------------------------------------------------------- #

    def activate(self, ctx: PluginContext) -> None:
        service = SemanticNavigationService(
            ctx.host.tools, max_retries=self.max_retries
        )
        ctx.services.provide(self.manifest.id, SEMANTIC_SERVICE, service)
        ctx.tools.register(self._status_spec(service))

    def deactivate(self, ctx: PluginContext) -> None:
        # Every contribution is revoked by the host's registration handles;
        # the service has no resource to close.
        return None

    # -- tools ------------------------------------------------------------ #

    def _status_spec(self, service: SemanticNavigationService) -> ToolSpec:
        plugin = self

        def handler(ctx: ToolContext, args: dict) -> ToolResult:
            return plugin._tool_status(service)

        return ToolSpec(
            name=SEMANTIC_STATUS_TOOL,
            description=SEMANTIC_STATUS_DESCRIPTION,
            input_schema=dict(SEMANTIC_STATUS_SCHEMA),
            handler=handler,
            side_effects=False,
            # the available tool set changes with the plugin lifecycle, so an
            # args-keyed replay would be stale
            cacheable=False,
        )

    def _tool_status(self, service: SemanticNavigationService) -> ToolResult:
        try:
            contract = service.contract()
        except Exception as exc:  # noqa: BLE001 - the tool boundary reports
            return ToolResult(
                STATUS_FAILED, {"error": f"{type(exc).__name__}: {exc}"}
            )
        data: dict[str, Any] = {
            "plugin": self.manifest.id,
            "requires": list(self.manifest.requires),
            "methods": list(SEMANTIC_METHODS),
            **contract,
            "ready": not contract.get("missing"),
        }
        data["tool_count"] = len(SEMANTIC_TOOL_NAMES)
        return ToolResult(STATUS_OK, data)


__all__ = [
    "NAVIGATION_CONTRACT",
    "PERCEPTION_CONTRACT",
    "SEMANTIC_STATUS_TOOL",
    "SEMANTIC_TOOLS",
    "SemanticNavigationPlugin",
]
