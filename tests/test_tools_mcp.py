"""MCP adapters for the tool runtime (goal.md section 6).

Covers, on the CPU with a mocked world/memory and a stand-in toolset module:

- tool discovery over MCP returns exactly the registry's schemas,
- a tool call over MCP goes through the shared registry/log/budget (and the
  direct-call adapter sees the same call in the same log),
- resource URIs are resolved by the server into real image bytes / JSON,
- the ``python -m vlm_nav.mcp_server`` entry point serves a real stdio
  client (subprocess) with tool discovery, a tool call and an image resource.
"""

import base64
import importlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import anyio
import numpy as np
import torch
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.memory import create_connected_server_and_client_session
from PIL import Image

from vlm_nav import mcp_server
from vlm_nav.tools import EvidenceRef, ToolContext, ToolRegistry, ToolResult, ToolSpec

REPO_ROOT = Path(__file__).resolve().parents[1]


class _MockWorld:
    """Minimal stand-in for SplatWorldCPU / EnvWorld."""

    def __init__(self, width: int = 32, height: int = 18) -> None:
        self.width = width
        self.height = height
        self.position_flu = np.array([0.0, 0.0, 1.0])
        self.yaw_rad = 0.0
        self.trajectory = [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (2.0, 0.0, 1.0)]
        self.target = SimpleNamespace(name="red chair", position_flu=(6.0, 2.0, 1.0), radius_m=0.4)
        self.last_subgoal = None
        self.applied = []

    def render(self) -> torch.Tensor:
        frame = torch.zeros(3, self.height, self.width)
        frame[0] = 0.5
        return frame

    def scan_views(self, n_views: int = 8, pitch_down_deg: float = 0.0, spread_deg=None):
        return [self.render() for _ in range(n_views)]

    def bev_image(self, target_estimate=None, size: int = 192) -> torch.Tensor:
        return torch.full((3, size, size), 0.2)

    def distance_to_target(self) -> float:
        return 4.25

    def target_in_fov(self) -> bool:
        return False

    def apply_action(self, action) -> str:
        self.applied.append(action)
        return "ok"


class _MockMemory:
    """ObservationLog stand-in exposing ``.observations`` / ``.get`` / ``.candidates``."""

    def __init__(self) -> None:
        self.observations: dict = {}
        self.candidates: list = []

    def add(self, obs_id: str, image, **meta) -> None:
        self.observations[obs_id] = {"image": image, "tool": "observe", **meta}

    def get(self, obs_id: str):
        return self.observations.get(obs_id)


class _FakeToolset:
    """Stand-in for the toolset module under parallel development."""

    OBSERVE_SCHEMA = {
        "type": "object",
        "properties": {"mode": {"type": "string", "enum": ["forward", "lookaround"]}},
        "required": [],
    }
    QUERY_SCHEMA = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    INSPECT_SCHEMA = {
        "type": "object",
        "properties": {
            "observation_id": {"type": "string"},
            "region": {"type": "array", "items": {"type": "number"}},
        },
        "required": ["observation_id", "region"],
    }

    def __init__(self) -> None:
        self.calls: list = []

    def build_registry(self, ctx, world=None, memory=None, **kwargs) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="observe",
                description="fake observe",
                input_schema=self.OBSERVE_SCHEMA,
                handler=self._observe,
                side_effects=True,
            )
        )
        registry.register(
            ToolSpec(
                name="query_memory",
                description="fake query_memory",
                input_schema=self.QUERY_SCHEMA,
                handler=self._query_memory,
            )
        )
        registry.register(
            ToolSpec(
                name="inspect_region",
                description="fake inspect_region",
                input_schema=self.INSPECT_SCHEMA,
                handler=self._inspect_region,
            )
        )
        return registry

    def _observe(self, ctx, args) -> ToolResult:
        self.calls.append(("observe", dict(args)))
        frame = ctx.world.render()
        obs_id = mcp_server.record_observation(
            ctx,
            frame,
            pose={"position_flu": [float(v) for v in ctx.world.position_flu], "yaw_rad": float(ctx.world.yaw_rad)},
            tool="observe",
            mode=args.get("mode", "forward"),
        )
        return ToolResult(
            "ok",
            {"observation_id": obs_id, "distance_to_target_m": ctx.world.distance_to_target()},
            [EvidenceRef("observation", obs_id, "current pose frame")],
        )

    def _query_memory(self, ctx, args) -> ToolResult:
        records = dict(getattr(ctx.memory, "observations", {}) or {})
        records.update(ctx.extra.get("obs_store") or {})
        hits = [oid for oid, record in records.items() if args.get("query") in str(record.get("tool", ""))]
        return ToolResult(
            "ok" if hits else "empty",
            {"query": args.get("query"), "count": len(hits), "hits": hits},
            [EvidenceRef("memory", oid, "recorded frame") for oid in hits],
        )

    def _inspect_region(self, ctx, args) -> ToolResult:
        self.calls.append(("inspect_region", dict(args)))
        crop = ctx.world.render()
        obs_id = mcp_server.record_observation(
            ctx,
            crop,
            tool="inspect_region",
            source_observation_id=args.get("observation_id"),
        )
        return ToolResult(
            "ok",
            {"observation_id": obs_id, "source_observation_id": args.get("observation_id"), "image": crop},
            [EvidenceRef("image", obs_id, "crop")],
        )


def _session(**kwargs):
    world = kwargs.pop("world", None) or _MockWorld()
    memory = kwargs.pop("memory", None) or _MockMemory()
    provider = kwargs.pop("provider", "default")
    ctx = ToolContext(world=world, memory=memory, episode_id=kwargs.pop("episode_id", "ep_test"))
    if provider == "default":
        provider = _FakeToolset()
    session = mcp_server.build_session(ctx, provider=provider, **kwargs)
    return session, world, memory


def _run_in_memory(service: str, session, scenario):
    async def main():
        server = mcp_server.build_server(service, session)
        async with create_connected_server_and_client_session(server) as client:
            return await scenario(client)

    return anyio.run(main)


def _blob(contents):
    assert len(contents) == 1, f"expected exactly one resource content, got {len(contents)}"
    return contents[0]


def _json_content(contents) -> dict:
    return json.loads(_blob(contents).text)


def _error_text(exc: BaseException) -> str:
    """Flatten exception groups (anyio task groups wrap client/server errors)."""
    parts = [f"{type(exc).__name__}: {exc}"]
    for sub in getattr(exc, "exceptions", None) or []:
        parts.append(_error_text(sub))
    return " | ".join(parts)


class TestSessionRegistry(unittest.TestCase):
    def test_services_split_the_same_registry(self) -> None:
        session, _, _ = _session(services=("env", "evidence"))
        self.assertEqual(session.names("env"), list(mcp_server.ENV_TOOL_NAMES))
        self.assertEqual(session.names("evidence"), list(mcp_server.EVIDENCE_TOOL_NAMES))
        # subset registries reuse the ToolSpec objects (same handler, same schema)
        self.assertIs(session.registry_for("env").get("observe"), session.registry.get("observe"))
        self.assertIs(
            session.registry_for("evidence").get("inspect_region"),
            session.registry.get("inspect_region"),
        )

    def test_missing_implementations_are_advertised_as_stubs(self) -> None:
        session, _, _ = _session()
        self.assertEqual(
            session.stub_tools,
            set(mcp_server.ALL_TOOL_NAMES) - {"observe", "query_memory", "inspect_region"},
        )
        stub = session.registry.get("go_to_viewpoint")
        self.assertIn(mcp_server.STUB_MARKER, stub.description)
        self.assertEqual(
            stub.input_schema, mcp_server.TOOL_CONTRACTS["go_to_viewpoint"]["input_schema"]
        )
        self.assertTrue(stub.side_effects)  # a movement tool is serialized

        result = session.call("go_to_viewpoint", {"viewpoint_id": "vp_3"})
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["reason"], "tool_not_implemented")
        self.assertTrue(result.data["stub"])
        self.assertEqual(session.tool_log()[-1]["tool"], "go_to_viewpoint")

    def test_provider_specs_are_used_verbatim(self) -> None:
        session, _, _ = _session()
        spec = session.registry.get("observe")
        self.assertEqual(spec.input_schema, _FakeToolset.OBSERVE_SCHEMA)
        self.assertTrue(spec.side_effects)

    def test_search_target_is_served_by_the_environment_service(self) -> None:
        """The search routine is part of the env tool vocabulary, with a contract.

        Without the name the MCP-only client cannot see the skill at all; without
        the contract a provider that does not implement it would raise instead of
        advertising the stub.
        """
        self.assertIn("search_target", mcp_server.ENV_TOOL_NAMES)
        self.assertIn("search_target", mcp_server.ALL_TOOL_NAMES)
        contract = mcp_server.TOOL_CONTRACTS["search_target"]
        self.assertEqual(contract["input_schema"]["required"], ["instruction"])
        self.assertIn("instruction", contract["input_schema"]["properties"])
        self.assertEqual(mcp_server.TOOL_SERVICE["search_target"], "env")

        session, _, _ = _session()
        self.assertEqual(session.names("env"), list(mcp_server.ENV_TOOL_NAMES))
        self.assertIn("search_target", session.names("env"))
        self.assertNotIn("search_target", session.names("evidence"))
        # the fake provider does not implement it: the contract is served as a stub
        self.assertIn("search_target", session.stub_tools)
        stub = session.registry.get("search_target")
        self.assertIn(mcp_server.STUB_MARKER, stub.description)
        self.assertEqual(stub.input_schema, contract["input_schema"])

    def test_the_search_skill_is_registered_when_the_module_provides_it(self) -> None:
        toolsets = mcp_server._tool_module_refs(None)
        if "vlm_nav.tool_impl_perception" not in toolsets:
            self.skipTest("the perception module is not on the discovery path")
        try:
            importlib.import_module("vlm_nav.tool_impl_perception")
        except ImportError as exc:  # pragma: no cover - depends on parallel workstreams
            self.skipTest(f"perception module unavailable: {exc}")

        session, _, _ = _session(provider=None)
        self.assertNotIn("search_target", session.stub_tools)
        spec = session.registry.get("search_target")
        from vlm_nav.skills_search import SEARCH_TARGET_SCHEMA

        self.assertEqual(spec.input_schema, SEARCH_TARGET_SCHEMA)
        self.assertTrue(spec.side_effects)  # the routine may fly the drone itself

    def test_the_served_contract_matches_the_skill_schema(self) -> None:
        """The MCP fallback contract is a copy of the skill's own schema.

        A stub is only honest when it advertises what the tool really takes: the
        two sources must agree on the arguments (the descriptions may differ).
        """
        try:
            from vlm_nav.skills_search import SEARCH_TARGET_SCHEMA
        except ImportError as exc:  # pragma: no cover - depends on parallel workstreams
            self.skipTest(f"search skill unavailable: {exc}")
        contract = mcp_server.TOOL_CONTRACTS["search_target"]["input_schema"]
        self.assertEqual(set(contract["properties"]), set(SEARCH_TARGET_SCHEMA["properties"]))
        self.assertEqual(contract["required"], SEARCH_TARGET_SCHEMA["required"])
        # the budget knobs the routine is bounded by are part of the contract
        for name in ("max_steps", "budget_turns", "max_groundings", "max_inspects"):
            self.assertIn(name, contract["properties"])

    def test_unknown_tool_is_rejected(self) -> None:
        session, _, _ = _session()
        with self.assertRaises(KeyError):
            session.call("teleport", {})

    def test_log_sink_records_direct_and_mcp_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calls.jsonl"
            session, _, _ = _session(log_sink=mcp_server._jsonl_sink(str(path)))
            session.call("observe", {"mode": "forward"})
            _run_in_memory("evidence", session, lambda client: client.call_tool("inspect_region", {"observation_id": "obs_1", "region": [0, 0, 1, 1]}))
            session.call("go_to_viewpoint", {"viewpoint_id": "vp_1"})
            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(
                [line["tool"] for line in lines],
                ["observe", "inspect_region", "go_to_viewpoint"],
            )
            self.assertTrue(all(line["episode_id"] == "ep_test" for line in lines))
            self.assertEqual(lines[0]["status"], "ok")
            self.assertEqual(lines[2]["status"], "failed")  # stub
            self.assertEqual(len(lines), len(session.tool_log()))


class TestToolDiscovery(unittest.TestCase):
    """The toolset module lands in parallel; discovery must find its specs."""

    PROVIDER_SOURCE = '''
from vlm_nav.tools import ToolResult, ToolSpec


def _ping(ctx, args):
    return ToolResult("ok", {"pong": args.get("query", "")})


def observe_spec(ctx=None, world=None, **kwargs):
    return ToolSpec(
        name="observe",
        description="module observe",
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=_ping,
        side_effects=True,
    )


def register_all(registry, ctx=None, world=None, memory=None, **kwargs):
    registry.register(
        ToolSpec(
            name="check_path",
            description="module check_path",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=_ping,
        )
    )


def build_evidence_registry(ctx=None, **kwargs):
    return [
        ToolSpec(
            name="ground_target",
            description="module ground_target",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=_ping,
        )
    ]


TOOL_SPECS = [
    ToolSpec(
        name="query_memory",
        description="module query_memory",
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=_ping,
    )
]
'''

    def _provider_path(self, directory: Path) -> Path:
        path = directory / "fake_toolset.py"
        path.write_text(self.PROVIDER_SOURCE, encoding="utf-8")
        return path

    def test_module_file_specs_are_imported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._provider_path(Path(tmp))
            session, _, _ = _session(provider=None, tools_module=str(path))
            self.assertTrue(session.provider.startswith(str(path)))
            # every supported export convention is picked up
            for name in ("observe", "check_path", "ground_target", "query_memory"):
                self.assertNotIn(name, session.stub_tools)
                self.assertEqual(session.registry.get(name).description, f"module {name}")
            # the MCP schema is the module's schema, not the stub contract
            listed = _run_in_memory(
                "env",
                session,
                lambda client: client.list_tools(),
            )
            observe = next(tool for tool in listed.tools if tool.name == "observe")
            self.assertEqual(observe.inputSchema, {"type": "object", "properties": {}, "required": []})
            self.assertTrue(observe.annotations.readOnlyHint is False)

    def test_explicit_tools_are_used_exclusively(self) -> None:
        spec = ToolSpec(
            name="verify_goal",
            description="injected verify",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=lambda ctx, args: ToolResult("ok", {"injected": True}),
        )
        session, _, _ = _session(tools=[spec])
        self.assertEqual(session.provider, "tools")
        self.assertEqual(session.registry.get("verify_goal").description, "injected verify")
        self.assertEqual(session.stub_tools, set(mcp_server.ALL_TOOL_NAMES) - {"verify_goal"})
        # every result now carries the task scope (goal.md v1): the injected
        # payload survives, plus the four scope fields under ``scope``
        data = session.call("verify_goal", {}).data
        self.assertIs(data["injected"], True)
        self.assertEqual(
            set(data["scope"]),
            {"task_id", "subtask_id", "candidate_id", "evidence_revision"},
        )

    def test_explicit_missing_module_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.py"
            with self.assertRaises(Exception):
                mcp_server.build_session(ToolContext(world=_MockWorld()), tools_module=str(missing))

    def test_auto_discovery_merges_the_m1_modules(self) -> None:
        try:
            perception = importlib.import_module("vlm_nav.tool_impl_perception")
            navigation = importlib.import_module("vlm_nav.tool_impl_navigation")
        except ImportError as exc:  # pragma: no cover - depends on parallel workstreams
            self.skipTest(f"M1 tool modules unavailable: {exc}")
        self.assertTrue(hasattr(perception, "register_perception_tools"))
        self.assertTrue(hasattr(navigation, "register_navigation_tools"))

        session, _, _ = _session(provider=None)
        self.assertEqual(session.provider, "vlm_nav.tool_impl_perception+vlm_nav.tool_impl_navigation")
        # perception + navigation provide every tool of both services except the
        # memory query, which still falls back to the contract stub
        self.assertTrue(
            {
                "observe",
                "inspect_region",
                "ground_target",
                "compare_candidates",
                "propose_viewpoints",
                "check_path",
                "approach_candidate",
                "go_to_viewpoint",
                "return_to_observation",
                "verify_goal",
            }
            <= set(mcp_server.ALL_TOOL_NAMES) - session.stub_tools
        )
        self.assertTrue(session.stub_tools <= {"query_memory"})
        for name in mcp_server.ALL_TOOL_NAMES:
            spec = session.registry.get(name)
            self.assertEqual(spec.input_schema.get("type"), "object", msg=name)
            self.assertTrue(spec.description)
        # one registry, both service views, real handlers registered (not stubs)
        self.assertIs(session.registry_for("env").get("observe"), session.registry.get("observe"))
        for service in ("env", "evidence"):
            listed = _run_in_memory(service, session, lambda client: client.list_tools())
            self.assertEqual(sorted(tool.name for tool in listed.tools), sorted(session.names(service)))


class TestMcpTools(unittest.TestCase):
    def test_tool_discovery_returns_registry_schemas(self) -> None:
        session, world, _ = _session()

        async def scenario(client):
            return await client.list_tools()

        for service in ("env", "evidence"):
            with self.subTest(service=service):
                listed = _run_in_memory(service, session, scenario)
                # the served set is exactly the service's tools, in the episode
                # state's order (P4: discovery follows the state, like the local
                # catalogue the same registry renders)
                self.assertEqual(sorted(tool.name for tool in listed.tools),
                                 sorted(session.names(service)))
                self.assertEqual([tool.name for tool in listed.tools],
                                 list(session.catalog(service).order))
                for tool in listed.tools:
                    spec = session.registry.get(tool.name)
                    self.assertEqual(tool.inputSchema, spec.input_schema)
                    self.assertEqual(tool.description, spec.description)
                    self.assertEqual(tool.annotations.readOnlyHint, not spec.side_effects)
                    # a movement tool is idempotent (a reached place repeats as
                    # already_done); a read-only one is not annotated as such
                    self.assertEqual(bool(tool.annotations.idempotentHint),
                                     bool(spec.side_effects))

    def test_tool_call_passes_through_shared_registry_and_log(self) -> None:
        session, world, memory = _session(services=("env", "evidence"))

        async def call_observe(client):
            return await client.call_tool("observe", {"mode": "forward"})

        result = _run_in_memory("env", session, call_observe)
        self.assertFalse(result.isError)
        structured = result.structuredContent
        self.assertEqual(structured["tool"], "observe")
        self.assertEqual(structured["episode_id"], "ep_test")
        self.assertEqual(structured["status"], "ok")
        self.assertEqual(structured["data"]["observation_id"], "obs_1")
        self.assertEqual(structured["evidence"][0]["kind"], "observation")
        self.assertIn("observation_id=obs_1", result.content[0].text)
        self.assertTrue(result.content[0].text.startswith("[ok]"))

        # the direct-call adapter shares registry, budget and log with MCP
        direct = session.call("observe", {"mode": "lookaround"})
        self.assertEqual(direct.status, "ok")
        self.assertEqual([entry["tool"] for entry in session.tool_log()], ["observe", "observe"])
        self.assertEqual(session.ctx.budget["tool_calls_left"], 40 - 2)

        # the evidence service sees the same log (one episode, one registry)
        async def call_query(client):
            return await client.call_tool("query_memory", {"query": "observe"})

        query = _run_in_memory("evidence", session, call_query)
        self.assertEqual(query.structuredContent["status"], "ok")
        self.assertEqual(query.structuredContent["data"]["hits"], ["obs_1", "obs_2"])
        self.assertEqual(len(session.tool_log()), 3)
        self.assertTrue(query.content[0].text.startswith("[ok]"))
        self.assertIn("query=observe", query.content[0].text)

    def test_tool_call_rejecting_unknown_tool_name(self) -> None:
        session, _, _ = _session()

        async def scenario(client):
            return await client.call_tool("go_to_viewpoint", {"viewpoint_id": "vp_1"})

        result = _run_in_memory("evidence", session, scenario)
        self.assertTrue(result.isError)
        self.assertIn("unknown tool for nav-evidence", result.content[0].text)

    def test_tool_result_image_is_returned_as_image_content(self) -> None:
        session, world, _ = _session()

        async def scenario(client):
            return await client.call_tool("inspect_region", {"observation_id": "obs_1", "region": [0, 0, 500, 500]})

        result = _run_in_memory("evidence", session, scenario)
        types = [block.type for block in result.content]
        self.assertEqual(types, ["text", "image"])
        image = result.content[1]
        self.assertEqual(image.mimeType, "image/png")
        with Image.open(io.BytesIO(base64.b64decode(image.data))) as decoded:
            self.assertEqual(decoded.size, (world.width, world.height))
        # the raw frame stays out of the structured payload
        self.assertIsInstance(result.structuredContent["data"]["image"], str)

    def test_referenced_observation_is_resolved_into_image_content(self) -> None:
        session, world, _ = _session()

        async def scenario(client):
            return await client.call_tool("observe", {"mode": "forward"})

        result = _run_in_memory("env", session, scenario)
        observed = result.structuredContent["data"]["observation_id"]
        self.assertEqual(observed, "obs_1")
        # the result refers to the frame by id; the server resolves it into bytes
        self.assertEqual([block.type for block in result.content], ["text", "image"])
        image = result.content[1]
        self.assertEqual(image.mimeType, "image/png")
        with Image.open(io.BytesIO(base64.b64decode(image.data))) as decoded:
            self.assertEqual(decoded.size, (world.width, world.height))


class TestMcpResources(unittest.TestCase):
    def _seeded_session(self, frames_dir: Path):
        session, world, memory = _session(services=("env", "evidence"))
        session.ctx.extra["frames_dir"] = str(frames_dir)
        (frames_dir / "step_009.png").write_bytes(
            mcp_server._encode_png(np.full((6, 8, 3), 200, dtype=np.uint8))
        )
        memory.add("obs_7", torch.full((3, 4, 6), 0.3), yaw_rad=0.0)
        session.call("observe", {"mode": "forward"})  # records obs_1 through the runtime helper
        session.ctx.extra["candidates"] = [
            {"id": 1, "position_flu": [1.0, 2.0, 1.0], "sightings": 2, "status": "confirmed"}
        ]
        return session, world, memory

    def test_image_meta_and_map_resources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session, world, _ = self._seeded_session(Path(tmp))

            async def scenario(client):
                return {
                    "obs_image": await client.read_resource("nav://obs/obs_1/image"),
                    "memory_image": await client.read_resource("nav://obs/obs_7/image"),
                    "frame_image": await client.read_resource("nav://obs/obs_009/image"),
                    "meta": await client.read_resource("nav://obs/obs_1/meta"),
                    "index": await client.read_resource("nav://obs"),
                    "bev": await client.read_resource("nav://map/bev?size=64"),
                    "state": await client.read_resource("nav://map/state"),
                    "traj": await client.read_resource("nav://traj/ep_test"),
                    "segment": await client.read_resource("nav://traj/ep_test/segment/0"),
                    "candidates": await client.read_resource("nav://candidates"),
                    "log": await client.read_resource("nav://episode/ep_test/log"),
                    "resources": await client.list_resources(),
                    "templates": await client.list_resource_templates(),
                }

            out = _run_in_memory("env", session, scenario)

            # recorded observation -> PNG bytes (never a file path)
            blob = _blob(out["obs_image"].contents)
            self.assertEqual(blob.mimeType, "image/png")
            raw = base64.b64decode(blob.blob)
            self.assertTrue(raw.startswith(b"\x89PNG\r\n\x1a\n"))
            with Image.open(io.BytesIO(raw)) as decoded:
                self.assertEqual(decoded.size, (world.width, world.height))

            # memory-backed observation
            memory_blob = _blob(out["memory_image"].contents)
            with Image.open(io.BytesIO(base64.b64decode(memory_blob.blob))) as decoded:
                self.assertEqual(decoded.size, (6, 4))

            # harness-style frame on disk, returned verbatim
            frame_blob = _blob(out["frame_image"].contents)
            self.assertEqual(base64.b64decode(frame_blob.blob), (Path(tmp) / "step_009.png").read_bytes())

            meta = _json_content(out["meta"].contents)
            self.assertEqual(meta["observation_id"], "obs_1")
            self.assertEqual(meta["pose"]["yaw_rad"], 0.0)
            self.assertAlmostEqual(meta["pose"]["position_flu"][2], 1.0)

            index = _json_content(out["index"].contents)
            self.assertEqual({entry["observation_id"] for entry in index["observations"]}, {"obs_1", "obs_7", "obs_009"})
            self.assertEqual(index["observations"][0]["image_uri"], "nav://obs/obs_1/image")

            bev = _blob(out["bev"].contents)
            with Image.open(io.BytesIO(base64.b64decode(bev.blob))) as decoded:
                self.assertEqual(decoded.size, (64, 64))
            state = _json_content(out["state"].contents)
            # agent-facing state only: no target identity/position/distance/FOV
            self.assertEqual(state["position_flu"], [0.0, 0.0, 1.0])
            self.assertNotIn("distance_to_target_m", state)
            self.assertNotIn("target_in_fov", state)
            self.assertNotIn("target", state)

            traj = _json_content(out["traj"].contents)
            self.assertEqual(traj["step_count"], 3)
            self.assertEqual(traj["poses"][0]["position_flu"], [0.0, 0.0, 1.0])
            self.assertEqual(_json_content(out["segment"].contents)["segment_index"], 0)

            self.assertEqual(_json_content(out["candidates"].contents)["count"], 1)
            log = _json_content(out["log"].contents)
            self.assertEqual([entry["tool"] for entry in log["tool_calls"]], ["observe"])
            self.assertIn("budget", log)

            uris = [str(resource.uri) for resource in out["resources"].resources]
            self.assertIn("nav://obs/obs_1/image", uris)
            self.assertIn("nav://map/bev", uris)
            templates = [tpl.uriTemplate for tpl in out["templates"].resourceTemplates]
            self.assertIn("nav://obs/{observation_id}/image", templates)
            self.assertIn("nav://traj/{episode_id}/segment/{segment_index}", templates)

    def test_resource_ownership_per_service(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session, _, _ = self._seeded_session(Path(tmp))

            async def scenario(client):
                return await client.list_resources()

            env_uris = [str(r.uri) for r in _run_in_memory("env", session, scenario).resources]
            evidence_uris = [str(r.uri) for r in _run_in_memory("evidence", session, scenario).resources]
            self.assertIn("nav://map/bev", env_uris)
            self.assertNotIn("nav://candidates", env_uris)
            self.assertIn("nav://candidates", evidence_uris)
            self.assertNotIn("nav://map/bev", evidence_uris)
            # every service can follow a reference handed over by the other one
            self.assertIn("nav://obs/obs_1/image", env_uris + evidence_uris)

    def test_unknown_resource_raises(self) -> None:
        session, _, _ = _session()

        async def bad_obs(client):
            return await client.read_resource("nav://obs/obs_404/image")

        with self.assertRaises(Exception) as caught:
            _run_in_memory("env", session, bad_obs)
        self.assertIn("unknown observation", _error_text(caught.exception))

        async def bad_scheme(client):
            return await client.read_resource("file:///etc/passwd")

        with self.assertRaises(Exception) as caught:
            _run_in_memory("env", session, bad_scheme)
        self.assertIn("not a nav:// resource", _error_text(caught.exception))


class TestCliEntryPoint(unittest.TestCase):
    def test_list_prints_inventory(self) -> None:
        buffer = io.StringIO()
        stdout, sys.stdout = sys.stdout, buffer
        try:
            code = mcp_server.main(["--service", "evidence", "--episode-id", "ep_cli", "--list"])
        finally:
            sys.stdout = stdout
        self.assertEqual(code, 0)
        inventory = json.loads(buffer.getvalue())
        self.assertEqual(inventory["service"], "evidence")
        self.assertEqual(inventory["server"], "nav-evidence")
        # the service serves exactly its own tools, in the episode state's order
        # (P4: discovery is state-aware; the state here is a fresh episode)
        listed = [tool["name"] for tool in inventory["tools"]]
        self.assertEqual(sorted(listed), sorted(mcp_server.EVIDENCE_TOOL_NAMES))
        self.assertEqual(listed, list(inventory["catalog"]["order"]))
        self.assertEqual(inventory["state"]["name"], "start")
        self.assertEqual(listed[0], "query_memory")
        for tool in inventory["tools"]:
            self.assertEqual(tool["input_schema"]["type"], "object")
            self.assertIn("side_effects", tool)
            self.assertIn("stub", tool)
        template_uris = [template["uri_template"] for template in inventory["resource_templates"]]
        self.assertIn("nav://obs/{observation_id}/image", template_uris)

    def test_main_requires_service(self) -> None:
        with self.assertRaises(SystemExit):
            mcp_server.main([])

    def test_stdio_streams_leave_process_streams_intact(self) -> None:
        # an embedded server must not close the host process' stdout on teardown
        stdin, stdout = mcp_server._stdio_streams()
        try:
            stdout.write("\n")
            stdout.flush()
        finally:
            stdin.close()
            stdout.close()
        self.assertFalse(sys.stdout.closed)
        self.assertFalse(sys.stdin.closed)


class TestStdioService(unittest.TestCase):
    """End-to-end: a real client spawns the documented module entry point."""

    def test_subprocess_discovery_call_and_image_resource(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frames = Path(tmp) / "frames"
            frames.mkdir()
            expected = mcp_server._encode_png(np.full((6, 8, 3), 200, dtype=np.uint8))
            (frames / "obs_000.png").write_bytes(expected)

            env = dict(os.environ)
            env["MPLCONFIGDIR"] = tmp
            env["PYTHONPATH"] = os.pathsep.join(
                [str(REPO_ROOT.parent), env.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)
            params = StdioServerParameters(
                command=sys.executable,
                args=[
                    "-m",
                    "vlm_nav.mcp_server",
                    "--service",
                    "evidence",
                    "--episode-id",
                    "ep_stdio",
                    "--frames-dir",
                    str(frames),
                ],
                env=env,
                cwd=str(REPO_ROOT),
            )

            async def scenario():
                async with stdio_client(params) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as client:
                        await client.initialize()
                        tools = await client.list_tools()
                        call = await client.call_tool("query_memory", {"query": "chair"})
                        image = await client.read_resource("nav://obs/obs_000/image")
                        resources = await client.list_resources()
                        return tools, call, image, resources

            tools, call, image, resources = anyio.run(scenario)

            # the served list is exactly the service's tools, state-ordered
            listed = [tool.name for tool in tools.tools]
            self.assertEqual(sorted(listed), sorted(mcp_server.EVIDENCE_TOOL_NAMES))
            self.assertEqual(listed[0], "query_memory")
            self.assertEqual(call.structuredContent["tool"], "query_memory")
            self.assertEqual(call.structuredContent["episode_id"], "ep_stdio")
            self.assertIn(call.structuredContent["status"], {"ok", "empty", "blocked", "failed", "cached"})

            blob = _blob(image.contents)
            self.assertEqual(blob.mimeType, "image/png")
            self.assertEqual(base64.b64decode(blob.blob), expected)
            self.assertIn("nav://obs/obs_000/image", [str(r.uri) for r in resources.resources])


class _FrameListToolset:
    """Toolset whose ``observe`` publishes several frames and refers to them.

    The frames differ in size so a test can tell which observation a resolved
    image content actually came from - an attachment bound by list position
    instead of by ``obs_id`` would show up as the wrong size.
    """

    SIZES = ((8, 6), (12, 4), (16, 9), (20, 5), (24, 7), (28, 6), (32, 10), (36, 8))

    def __init__(self, frames: int = 8, refs: bool = True) -> None:
        self.frames = int(frames)
        self.refs = bool(refs)
        self.calls = []

    def build_registry(self, ctx, world=None, memory=None, **kwargs) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="observe",
                description="scan",
                input_schema={"type": "object", "properties": {}},
                handler=self._observe,
            )
        )
        return registry

    def _observe(self, ctx, args) -> ToolResult:
        self.calls.append(dict(args))
        ids = []
        for width, height in self.SIZES[: self.frames]:
            obs_id = mcp_server.record_observation(
                ctx,
                torch.full((3, height, width), 0.4),
                pose={"position_flu": [0.0, 0.0, 1.0], "yaw_rad": 0.0},
                tool="observe",
            )
            ids.append(obs_id)
        data = {"observation_id": ids[0], "n_views": len(ids)}
        if self.refs:
            data["image_refs"] = list(ids)
        return ToolResult(
            "ok", data, [EvidenceRef("observation", obs_id, "scan view") for obs_id in ids]
        )


class TestResultImageBudget(unittest.TestCase):
    """A result's evidence pack carries the frames it refers to (bound by id)."""

    def _session(self, toolset):
        ctx = ToolContext(world=_MockWorld(), memory=_MockMemory(), episode_id="ep_images")
        return mcp_server.build_session(ctx, provider=toolset, services=("env",))

    def test_a_scan_attaches_every_referenced_view(self) -> None:
        toolset = _FrameListToolset(frames=8)
        session = self._session(toolset)

        async def scenario(client):
            return await client.call_tool("observe", {})

        result = _run_in_memory("env", session, scenario)

        self.assertGreaterEqual(mcp_server.MAX_RESULT_IMAGES, 8)
        images = [block for block in result.content if block.type == "image"]
        self.assertEqual(len(images), 8)
        self.assertEqual([block.type for block in result.content], ["text"] + ["image"] * 8)

    def test_each_attached_frame_is_bound_to_its_observation_id(self) -> None:
        toolset = _FrameListToolset(frames=6)
        session = self._session(toolset)

        async def scenario(client):
            return await client.call_tool("observe", {})

        result = _run_in_memory("env", session, scenario)

        ids = result.structuredContent["data"]["image_refs"]
        images = [block for block in result.content if block.type == "image"]
        self.assertEqual(len(images), len(ids))
        for obs_id, block in zip(ids, images):
            width, height = _FrameListToolset.SIZES[int(obs_id.split("_")[1]) - 1]
            with Image.open(io.BytesIO(base64.b64decode(block.data))) as decoded:
                self.assertEqual(decoded.size, (width, height), obs_id)

    def test_evidence_references_resolve_frames_without_image_refs(self) -> None:
        toolset = _FrameListToolset(frames=3, refs=False)
        session = self._session(toolset)

        async def scenario(client):
            return await client.call_tool("observe", {})

        result = _run_in_memory("env", session, scenario)

        self.assertNotIn("image_refs", result.structuredContent["data"])
        images = [block for block in result.content if block.type == "image"]
        self.assertEqual(len(images), 3)
        for index, block in enumerate(images):
            width, height = _FrameListToolset.SIZES[index]
            with Image.open(io.BytesIO(base64.b64decode(block.data))) as decoded:
                self.assertEqual(decoded.size, (width, height))

    def test_the_budget_still_bounds_a_huge_evidence_pack(self) -> None:
        toolset = _FrameListToolset(frames=8)
        session = self._session(toolset)

        async def scenario(client):
            return await client.call_tool("observe", {})

        result = _run_in_memory("env", session, scenario)

        images = [block for block in result.content if block.type == "image"]
        self.assertLessEqual(len(images), mcp_server.MAX_RESULT_IMAGES)


class TestTaskScopeContract(unittest.TestCase):
    """The MCP service carries the same task scope and finish contract."""

    def test_finish_over_mcp_reports_missing_subtasks_and_scope(self) -> None:
        from vlm_nav.tool_impl_navigation import register_navigation_tools
        from vlm_nav.tool_impl_task import register_task_tools

        registry = ToolRegistry()
        register_navigation_tools(registry)
        register_task_tools(registry)
        ctx = ToolContext(world=_MockWorld(), memory=_MockMemory(), episode_id="ep_scope")
        session = mcp_server.build_session(ctx, tools=registry, services=("env",))
        session.call(
            "update_plan",
            {
                "steps": [
                    {"id": "st_1", "goal": "confirm the cabinet"},
                    {"id": "st_2", "goal": "approach the cabinet"},
                    {"id": "st_3", "goal": "report what is inside"},
                ]
            },
        )
        session.call("update_plan", {"patch": [{"id": "st_1", "status": "done"}]})

        async def scenario(client):
            return await client.call_tool(
                "finish", {"reason": "the cabinet is verified", "success": True}
            )

        result = _run_in_memory("env", session, scenario)
        data = result.structuredContent["data"]

        self.assertEqual(result.structuredContent["status"], "blocked")
        self.assertIs(data["success"], False)
        self.assertEqual(
            [item["subtask_id"] for item in data["missing_subtasks"]],
            ["st_2", "st_3"],
        )
        for key in ("task_id", "subtask_id", "candidate_id", "evidence_revision"):
            self.assertIn(key, data)
        # the shared registry stamps the direct call the same way
        self.assertIn("task_id", session.call("get_plan", {}).data)


if __name__ == "__main__":
    unittest.main()
