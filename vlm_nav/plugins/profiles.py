"""Named plugin profiles: which plugins make up a deployment, in what order.

A profile is deliberately just data plus two helpers. It names plugin ids; the
caller supplies the factories because the host must not import every plugin
implementation (goal.md section 2: plugins depend on contracts, not on each
other). Activation is always explicit and follows the profile order, so a
dependency declared later in the list is still activated by the host's normal
provider resolution, never silently skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from . import (
    PLUGIN_ACTIVE,
    LifecycleError,
    LifecycleReport,
    Plugin,
    PluginHost,
)


@dataclass(frozen=True)
class Profile:
    """A named, ordered plugin set with context defaults for its runtime."""

    id: str
    plugins: tuple[str, ...]
    context_defaults: dict = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "plugins": list(self.plugins),
            "context_defaults": dict(self.context_defaults),
            "description": self.description,
        }


BUILTIN_PROFILES: dict[str, Profile] = {
    "motion-only": Profile(
        id="motion-only",
        plugins=("resource.manager", "motion.base"),
        description="Task-level motion only; no perception, navigation or VLM.",
    ),
    "semantic-nav": Profile(
        id="semantic-nav",
        plugins=(
            "resource.manager",
            "model.provider",
            "perception.core",
            "navigation.core",
            "task.tools",
            "semantic.navigation",
            "mcp.adapter",
        ),
        description="Legacy perception/navigation tool stack plus the semantic service.",
    ),
    "mixed-mission": Profile(
        id="mixed-mission",
        plugins=(
            "resource.manager",
            "model.provider",
            "perception.core",
            "navigation.core",
            "task.tools",
            "task.planner",
            "semantic.navigation",
            "motion.base",
            "mixed.mission",
            "mcp.adapter",
        ),
        description="Motion, semantic navigation, planner and mixed-mission executor.",
    ),
    "4090-local": Profile(
        id="4090-local",
        plugins=(
            "resource.manager",
            "model.provider.worker",
            "perception.core",
            "navigation.core",
            "task.tools",
            "task.planner",
            "semantic.navigation",
            "motion.base",
            "mixed.mission",
            "mcp.adapter",
        ),
        context_defaults={
            "perception.detector": "sam3_dino",
            "model.provider.worker": {
                "device": "cuda:0",
                "dtype": "bfloat16",
                "quantization": "4bit",
            },
        },
        description=(
            "mixed-mission on a 24 GiB RTX 4090: model in a managed worker "
            "process with 4-bit quantization and the local detector."
        ),
    ),
}


def install_bundle(
    host: PluginHost, plugins: Mapping[str, Plugin]
) -> list[LifecycleReport]:
    """Install a profile's plugin objects (activation stays explicit)."""
    return host.install_bundle(plugins)


def activate_profile(
    host: PluginHost,
    profile: Profile,
    factories: Mapping[str, Callable[[], Plugin]],
) -> dict:
    """Install missing profile plugins from ``factories``, then activate them.

    Returns ``{"profile", "active", "reports", "catalog"}`` (plus the profile's
    ``context_defaults``). ``active`` preserves the profile order; ``reports``
    are :meth:`LifecycleReport.to_dict` entries in install-then-activate order.
    """
    reports: list[LifecycleReport] = []
    bundle: dict[str, Plugin] = {}
    for plugin_id in profile.plugins:
        if _is_installed(host, plugin_id):
            continue
        factory = factories.get(plugin_id)
        if factory is None:
            raise LifecycleError(
                f"profile {profile.id}: no factory for plugin {plugin_id!r}"
            )
        plugin = factory()
        manifest = getattr(plugin, "manifest", None)
        if manifest is None or manifest.id != plugin_id:
            found = getattr(manifest, "id", None)
            raise LifecycleError(
                f"profile {profile.id}: factory for {plugin_id!r} returned "
                f"plugin {found!r}"
            )
        bundle[plugin_id] = plugin
    if bundle:
        reports.extend(install_bundle(host, bundle))
    for plugin_id in profile.plugins:
        reports.append(host.activate(plugin_id))

    active = [
        plugin_id for plugin_id in profile.plugins
        if host.status(plugin_id) == PLUGIN_ACTIVE
    ]
    return {
        "profile": profile.id,
        "active": active,
        "reports": [report.to_dict() for report in reports],
        "catalog": host.catalog(),
        "context_defaults": dict(profile.context_defaults),
    }


def _is_installed(host: PluginHost, plugin_id: str) -> bool:
    try:
        host.record(plugin_id)
    except LifecycleError:
        return False
    return True
