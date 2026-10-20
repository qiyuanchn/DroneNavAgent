"""Compose the mixed-mission capability stack on an existing tool registry.

The runner and MCP sessions already own a ``ToolRegistry``. This helper installs
the motion/planner/mixed plugins into that same registry, and adds the
semantic-navigation service when the registry already exposes the search /
approach / confirm tools. It is the bridge between the agent loop and the
plugin lifecycle host without duplicating tool implementations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from . import LifecycleError, PluginHost
from .mixed import MixedMissionPlugin
from .motion import MotionPlugin
from .semantic import SemanticNavigationPlugin
from .task_planner import TaskPlannerPlugin

SEMANTIC_TOOL_NAMES = (
    "search_target",
    "approach_candidate",
    "confirm_candidate",
    "verify_goal",
)


def install_mission_stack(
    registry: Any,
    root_dir: str | Path,
    *,
    artifacts_dir: str | Path | None = None,
    semantic: bool | None = None,
    activation_order: Iterable[str] | None = None,
) -> PluginHost:
    """Install and activate the mission stack; return the lifecycle host.

    ``semantic=None`` auto-detects: the semantic service is added only when the
    registry already has the underlying tools **and** those tools are not
    already provided by an installed semantic plugin. Passing ``semantic=False``
    gives a motion-only stack.
    """
    root = Path(root_dir)
    artifacts = Path(artifacts_dir) if artifacts_dir else root / "artifacts"
    host = PluginHost(root_dir=root, tool_registry=registry)
    plugins: list[Any] = [
        MotionPlugin(artifacts_dir=artifacts),
        TaskPlannerPlugin(),
    ]
    names = set(registry.names())
    if semantic is None:
        semantic = all(name in names for name in SEMANTIC_TOOL_NAMES)
    if semantic:
        if not all(name in names for name in SEMANTIC_TOOL_NAMES):
            missing = [name for name in SEMANTIC_TOOL_NAMES if name not in names]
            raise LifecycleError(
                "semantic mission stack requires tools: " + ", ".join(missing)
            )
        # The tools already live in this registry; advertise the interfaces the
        # semantic plugin depends on without registering a second copy.
        host.services.provide("runner", "navigation.tools@1", True)
        host.services.provide("runner", "perception.tools@1", True)
        plugins.append(SemanticNavigationPlugin())
    plugins.append(MixedMissionPlugin(artifacts_dir=artifacts))

    host.install_bundle(plugins)
    order = list(activation_order or [plugin.manifest.id for plugin in plugins])
    by_id = {plugin.manifest.id: plugin for plugin in plugins}
    for plugin_id in order:
        if plugin_id not in by_id:
            raise LifecycleError(f"unknown plugin in activation order: {plugin_id}")
        host.activate(plugin_id)
    return host
