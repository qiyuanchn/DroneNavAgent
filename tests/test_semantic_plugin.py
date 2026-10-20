"""Tests for the semantic navigation plugin (workstream C).

The plugin is exercised against a real ``PluginHost`` whose required
``navigation.tools@1`` / ``perception.tools@1`` services are host-provided
stubs. Stub tool specs are registered directly on the host's registry where a
test needs them; no real perception/navigation module is imported.
"""

from __future__ import annotations

import pytest

from vlm_nav.plugins import LifecycleError, PluginHost
from vlm_nav.plugins.semantic import (
    NAVIGATION_CONTRACT,
    PERCEPTION_CONTRACT,
    SEMANTIC_STATUS_TOOL,
    SemanticNavigationPlugin,
)
from vlm_nav.semantic import SEMANTIC_METHODS, SEMANTIC_SERVICE, SEMANTIC_TOOL_NAMES
from vlm_nav.semantic import SemanticNavigationService
from vlm_nav.tools import STATUS_OK, ToolContext, ToolResult, ToolSpec


def make_host(tmp_path, *, tool_services: bool = True) -> PluginHost:
    host = PluginHost(root_dir=tmp_path / "plugins")
    if tool_services:
        # what workstream D's legacy plugins provide; here they are stubs
        host.services.provide("test.navigation", NAVIGATION_CONTRACT, object())
        host.services.provide("test.perception", PERCEPTION_CONTRACT, object())
    return host


def install_stub(host: PluginHost, name: str, data: dict | None = None) -> None:
    host.tools.register(
        ToolSpec(
            name=name,
            description=f"stub {name}",
            input_schema={"type": "object", "properties": {}},
            handler=lambda ctx, args: ToolResult(STATUS_OK, dict(data or {})),
        )
    )


def test_manifest_declares_the_contract():
    manifest = SemanticNavigationPlugin.manifest
    assert manifest.id == "semantic.navigation"
    assert manifest.version == "0.1.0"
    assert manifest.provides == (SEMANTIC_SERVICE,)
    assert manifest.requires == (NAVIGATION_CONTRACT, PERCEPTION_CONTRACT)
    assert manifest.tools == (SEMANTIC_STATUS_TOOL,)
    assert manifest.isolation == "inproc"


def test_activation_provides_service_and_status_tool(tmp_path):
    host = make_host(tmp_path)
    report = host.install(SemanticNavigationPlugin(max_retries=3))
    assert report.status == "installed"

    report = host.activate("semantic.navigation")
    assert report.status == "active"

    service = host.services.get(SEMANTIC_SERVICE)
    assert isinstance(service, SemanticNavigationService)
    assert service.registry is host.tools
    assert service.max_retries == 3
    assert SEMANTIC_STATUS_TOOL in host.tools.names()

    ctx = ToolContext(world=None)
    result = host.tools.call(SEMANTIC_STATUS_TOOL, {}, ctx)
    assert result.status == STATUS_OK
    data = result.data
    assert data["plugin"] == "semantic.navigation"
    assert data["service"] == SEMANTIC_SERVICE
    assert data["methods"] == list(SEMANTIC_METHODS)
    assert set(data["available"]) == set(SEMANTIC_TOOL_NAMES)
    assert data["requires"] == [NAVIGATION_CONTRACT, PERCEPTION_CONTRACT]
    assert data["max_retries"] == 3
    # none of the semantic tools are registered in this bare host
    assert data["ready"] is False
    assert set(data["missing"]) == set(SEMANTIC_TOOL_NAMES)


def test_install_requires_both_tool_contracts(tmp_path):
    host = make_host(tmp_path, tool_services=False)
    with pytest.raises(LifecycleError) as excinfo:
        host.install(SemanticNavigationPlugin())
    message = str(excinfo.value)
    assert NAVIGATION_CONTRACT in message
    assert PERCEPTION_CONTRACT in message


def test_activation_fails_when_a_required_service_disappears(tmp_path):
    host = make_host(tmp_path)
    host.install(SemanticNavigationPlugin())
    host.services.remove_provider("test.navigation")
    with pytest.raises(LifecycleError):
        host.activate("semantic.navigation")
    assert host.status("semantic.navigation") == "failed"


def test_status_reports_registered_semantic_tools(tmp_path):
    host = make_host(tmp_path)
    for name in SEMANTIC_TOOL_NAMES:
        install_stub(host, name)
    host.install(SemanticNavigationPlugin())
    host.activate("semantic.navigation")

    result = host.tools.call(
        SEMANTIC_STATUS_TOOL, {}, ToolContext(world=None)
    )
    assert result.status == STATUS_OK
    assert result.data["ready"] is True
    assert result.data["missing"] == []
    assert all(result.data["available"].values())


def test_deactivate_revokes_service_and_tool(tmp_path):
    host = make_host(tmp_path)
    host.install(SemanticNavigationPlugin())
    host.activate("semantic.navigation")

    report = host.deactivate("semantic.navigation")
    assert report.status == "inactive"
    assert report.ok
    assert not host.services.has(SEMANTIC_SERVICE)
    assert SEMANTIC_STATUS_TOOL not in host.tools.names()

    report = host.uninstall("semantic.navigation")
    assert report.status == "uninstalled"
    assert report.ok
    assert host.plugins() == []


def test_service_uses_the_host_registry(tmp_path):
    host = make_host(tmp_path)
    host.tools.register(
        ToolSpec(
            name="search_target",
            description="stub search",
            input_schema={"type": "object", "properties": {}},
            handler=lambda ctx, args: ToolResult(
                STATUS_OK,
                {"candidate_id": "cand_9", "confirmed": False, "reason": "found"},
            ),
            side_effects=True,
            cacheable=False,
        )
    )
    host.install(SemanticNavigationPlugin())
    host.activate("semantic.navigation")
    service = host.services.get(SEMANTIC_SERVICE)

    out = service.search(ToolContext(world=object()), "chair")
    assert out["ok"] is True
    assert out["candidate_id"] == "cand_9"
    assert out["accepted"] is False
    assert out["calls"] == 1
    # the call went through the host registry (budget charged, log appended)
    assert out["data"]["confirmed"] is False
