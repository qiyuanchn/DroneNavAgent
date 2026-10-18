"""P4-B: tool dispatch, batching and the traceable success chain.

The decision loop asks four things of the tool layer that the earlier rounds did
not:

- the *catalogue* the model reads is trimmed to the state the episode is in
  (observe/search before anything is grounded, compare/approach/verify once a
  candidate exists, task_status/cancel while a leg is running) and arrives
  **whole** - the full arguments of the tools a state calls - instead of being
  cut mid-entry by a character budget;
- read-only checks may be packed into one call (``batch_read``), so one model
  turn can observe and query the memory at once, while pose-changing tools stay
  serial with their own ``task_id`` contract;
- a repeated movement request for a place the drone already reached answers
  ``already_done`` instead of flying the leg again;
- ``finish`` with ``success: true`` needs a traceable evidence chain
  (``verify_goal`` verified, or a grounded candidate plus an arrival), and the
  runtime/runner record the trail - an unbacked claim is downgraded and never
  counted as success.

CPU-only: the fake world, the fake VLM and the stub tools of this file run
without a GPU; the runner tests use the mock backend.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

from vlm_nav import mcp_server
from vlm_nav import tool_impl_navigation as navigation_module
from vlm_nav.actions import NavAction, action_to_body_delta, body_delta_to_world
from vlm_nav.tool_impl_navigation import register_navigation_tools
from vlm_nav.tool_impl_perception import register_perception_tools
from vlm_nav.tool_impl_task import register_task_tools
from vlm_nav.tools import (
    BATCH_TOOL_NAME,
    EVIDENCE_PACK_KEY,
    STATUS_CACHED,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_OK,
    TOOL_STATE_CANDIDATE,
    TOOL_STATE_MOVING,
    TOOL_STATE_SEARCH,
    TOOL_STATE_START,
    TOOL_STATE_VERIFIED,
    EvidenceRef,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    completion_evidence,
    evidence_pack,
    register_batch_tool,
    tool_catalog,
    tool_state,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# fakes (CPU only)
# ---------------------------------------------------------------------------
class FakeTarget:
    def __init__(self, name="blob", position=(3.0, 0.0, 0.0), radius_m=0.5):
        self.name = name
        self.position_flu = tuple(position)
        self.radius_m = radius_m


class FakeWorld:
    """Kinematic fake world with the real worlds' action conventions."""

    def __init__(self, position=(0.0, 0.0, 0.0), yaw_deg=0.0, target=(3.0, 0.0, 0.0),
                 radius_m=0.5, move=True, outcome=""):
        self.position = np.asarray(position, dtype=np.float64)
        self.yaw_rad = math.radians(yaw_deg)
        self.target = FakeTarget(position=target, radius_m=radius_m)
        self.outcome = outcome
        self.move = bool(move)
        self.applied: list = []
        self.trajectory = [tuple(self.position)]
        self.width, self.height, self.hfov_deg = 320, 180, 90.0

    @property
    def position_flu(self):
        return self.position

    def render(self):
        return np.zeros((3, 4, 4), dtype=np.float32)

    def distance_to_target(self):
        return float(np.linalg.norm(self.position - np.asarray(self.target.position_flu)))

    def target_in_fov(self):
        return True

    def depth_at_pixel(self, pixel):
        return float(np.linalg.norm(np.asarray(self.target.position_flu) - self.position))

    def apply_action(self, action: NavAction) -> str:
        self.applied.append(action)
        if action.is_terminal or action.kind == "hover":
            return action.kind
        if not self.move:
            return "did not move"
        dx, dy, dz, dyaw = action_to_body_delta(action)
        self.yaw_rad = math.atan2(math.sin(self.yaw_rad + dyaw), math.cos(self.yaw_rad + dyaw))
        wx, wy, wz = body_delta_to_world(dx, dy, dz, self.yaw_rad)
        self.position = self.position + np.asarray([wx, wy, wz])
        self.trajectory.append(tuple(self.position))
        p = self.position
        text = f"moved to ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"
        return f"{text} [{self.outcome}]" if self.outcome else text


def make_context(world=None, extra=None, calls=40):
    return ToolContext(
        world=world or FakeWorld(),
        budget={"tool_calls_left": calls},
        extra=dict(extra or {}),
    )


def full_registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_perception_tools(registry)
    register_navigation_tools(registry)
    register_task_tools(registry)
    return registry


def observe_stub(ctx, args):
    return ToolResult(STATUS_OK, {"observation_id": "obs_9", "mode": args.get("mode", "forward")},
                      [EvidenceRef("observation", "obs_9", "stub frame")])


def memory_stub(ctx, args):
    return ToolResult(STATUS_EMPTY, {"query": args.get("query", ""), "hits": []})


def move_stub(ctx, args):
    return ToolResult(STATUS_OK, {"task_id": "nav_1", "outcome": "arrived"})


def read_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="observe",
            description="look around",
            input_schema={"type": "object", "properties": {"mode": {"type": "string"}}},
            handler=observe_stub,
        )
    )
    registry.register(
        ToolSpec(
            name="query_memory",
            description="ask the memory",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}},
                          "required": ["query"]},
            handler=memory_stub,
        )
    )
    registry.register(
        ToolSpec(
            name="approach_candidate",
            description="fly to a candidate",
            input_schema={"type": "object", "properties": {"candidate_id": {"type": "string"}}},
            handler=move_stub,
            side_effects=True,
        )
    )
    register_batch_tool(registry)
    return registry


def scripted_vlm(script):
    """A tool-aware backend that replays one JSON call per turn."""

    class Scripted:
        def __init__(self):
            self.script = list(script)
            self.seen: list = []

        def _next(self):
            item = self.script.pop(0) if self.script else {
                "tool": "finish", "args": {"reason": "script exhausted"}}
            return json.dumps(item)

        def decide_with_tools(self, frames, system_prompt, user_prompt, tools_text="",
                              context=None):
            self.seen.append({"tools_text": tools_text, "system": system_prompt,
                              "tool_calls_left": (context or {}).get("tool_calls_left")})
            return self._next()

        def decide(self, frame_chw, system_prompt, user_prompt, context=None):
            self.seen.append({"tools_text": "", "system": system_prompt})
            return self._next()

    return Scripted()


# ---------------------------------------------------------------------------
# 1. state-aware catalogue
# ---------------------------------------------------------------------------
class ToolStateTest(unittest.TestCase):
    """The state the episode is in chooses which tools lead the catalogue."""

    def test_a_fresh_episode_is_in_the_start_state(self):
        state = tool_state(make_context().extra)
        self.assertEqual(state.name, TOOL_STATE_START)
        order = state.priority()
        self.assertEqual(order[0], "observe")
        # nothing is grounded yet: look and search come before approach/verify
        self.assertLess(order.index("query_memory"), order.index("approach_candidate"))
        self.assertLess(order.index("search_target"), order.index("approach_candidate"))

    def test_recorded_observations_without_a_candidate_are_the_search_state(self):
        ctx = make_context()
        ctx.extra["observations"] = {"obs_1": {"position_flu": [0.0, 0.0, 0.0]}}
        state = tool_state(ctx.extra)
        self.assertEqual(state.name, TOOL_STATE_SEARCH)
        order = state.priority()
        self.assertEqual(order[0], "search_target")
        for name in ("observe", "ground_target"):
            self.assertIn(name, order[:4])

    def test_a_grounded_candidate_leads_with_compare_approach_verify(self):
        ctx = make_context()
        ctx.extra["perception.candidates"] = {
            "cand_1": {"id": "cand_1", "observation_id": "obs_2",
                       "position_flu": [3.0, 0.0, 0.0]}
        }
        state = tool_state(ctx.extra)
        self.assertEqual(state.name, TOOL_STATE_CANDIDATE)
        self.assertEqual(state.candidates, ("cand_1",))
        order = state.priority()
        for name in ("compare_candidates", "approach_candidate", "verify_goal"):
            self.assertIn(name, order[:5], order)
        self.assertIn("cand_1", state.hint)

    def test_a_running_task_leads_with_task_status_and_cancel(self):
        ctx = make_context()
        ctx.extra["nav_tasks"] = {
            "nav_4": {"task_id": "nav_4", "intent": "approach_candidate", "status": "running",
                      "candidate_id": "cand_2", "remaining_m": 2.5}
        }
        state = tool_state(ctx.extra)
        self.assertEqual(state.name, TOOL_STATE_MOVING)
        order = state.priority()
        self.assertEqual(order[0], "task_status")
        self.assertEqual(order[1], "cancel_task")
        # the intent tool of the running task is what continues it
        self.assertIn("approach_candidate", order[:4])
        self.assertIn("nav_4", state.hint)

    def test_a_verified_goal_leads_with_the_completion_check_and_finish(self):
        ctx = make_context(FakeWorld(position=(2.2, 0.0, 0.0)))  # at the target
        navigation_module._tool_verify_goal(ctx, {"reason": "check"})
        state = tool_state(ctx.extra)
        self.assertEqual(state.name, TOOL_STATE_VERIFIED)
        order = state.priority()
        self.assertIn("finish", order[:3], order)
        self.assertIn("verify_goal", order[:3], order)


class ToolCatalogTest(unittest.TestCase):
    """The rendered catalogue is complete, state-ordered and never cut mid-entry."""

    def setUp(self):
        self.registry = full_registry()

    def test_the_catalog_names_every_registered_tool(self):
        catalog = tool_catalog(self.registry, limit=9000)
        self.assertEqual(sorted(catalog.order), self.registry.names())
        tiers = [name for tier in (catalog.full, catalog.compact, catalog.named)
                 for name in tier]
        self.assertEqual(sorted(tiers), self.registry.names())
        # the dispatch rules (batch, already_done, the completion claim) lead the
        # listing the model reads
        self.assertIn("batch_read", catalog.text)
        self.assertIn("already_done", catalog.text)
        self.assertIn("success=true is only counted", catalog.text)

    def test_the_states_own_tools_arrive_with_their_full_arguments(self):
        contexts = (
            {},
            {"observations": {"obs_1": {}}},
            {"perception.candidates": {"cand_1": {"id": "cand_1"}}},
            {"nav_tasks": {"nav_1": {"task_id": "nav_1", "intent": "go_to_viewpoint",
                                     "status": "running"}}},
        )
        for extra in contexts:
            with self.subTest(extra=sorted(extra)):
                state = tool_state(extra)
                catalog = tool_catalog(self.registry, state=state, limit=9000)
                self.assertGreaterEqual(len(catalog.full), 3, catalog.text)
                for name in catalog.full:
                    spec = self.registry.get(name)
                    self.assertIn(
                        f"- {name}: {spec.description} "
                        f"args={json.dumps(spec.input_schema, separators=(',', ':'))}",
                        catalog.text,
                    )
                # the tier lists run in the state's priority order
                ranked = state.priority()
                ranks = [ranked.index(name) for name in catalog.full]
                self.assertEqual(ranks, sorted(ranks))

    def test_the_listing_is_bounded_and_never_cut_inside_an_entry(self):
        for limit in (700, 1200, 3000, 9000):
            with self.subTest(limit=limit):
                catalog = tool_catalog(self.registry, limit=limit)
                self.assertLessEqual(len(catalog.text), limit)
                self.assertIn(catalog.state.name, catalog.text)
                # every "- " line is a whole tool entry: a name, a description
                # and (when one is shown) a parseable schema
                for line in catalog.text.splitlines():
                    if not line.startswith("- "):
                        continue
                    name, _, rest = line[2:].partition(": ")
                    self.assertIn(name, self.registry.names(), line)
                    self.assertTrue(rest, line)
                    if " args=" in rest:
                        schema = rest.rpartition(" args=")[2]
                        self.assertIsInstance(json.loads(schema), dict)
                # ... and the names not listed with arguments are still named
                for name in catalog.named:
                    self.assertIn(name, catalog.text)

    def test_a_small_budget_degrades_to_names_but_keeps_the_top_tool(self):
        state = tool_state({"perception.candidates": {"cand_1": {"id": "cand_1"}}})
        # a budget that fits one entry: the state's leading tool is complete
        catalog = tool_catalog(self.registry, state=state, limit=2000)
        self.assertLessEqual(len(catalog.text), 2000)
        self.assertEqual(catalog.full[0], catalog.order[0])
        self.assertIn(catalog.order[0], catalog.text)
        # ... and a budget too small for a full entry still names every tool
        tiny = tool_catalog(self.registry, state=state, limit=700)
        self.assertLessEqual(len(tiny.text), 700)
        self.assertEqual(tiny.full, ())
        self.assertEqual(sorted(tiny.full + tiny.compact + tiny.named),
                         self.registry.names())
        for name in self.registry.names():
            self.assertIn(name, tiny.text)

    def test_specs_text_is_state_aware_once_a_source_is_attached(self):
        ctx = make_context()
        registry = full_registry()
        legacy = registry.specs_text()  # no state source: the full catalogue
        self.assertIn("- approach_candidate:", legacy)

        registry.attach_state(lambda: ctx.extra, limit=9000)
        text = registry.specs_text()
        self.assertLess(len(text), len(legacy))
        self.assertIn("STATE=start", text)
        self.assertIn("- observe:", text)

        # the state moves with the context: a grounded candidate re-leads it
        ctx.extra["perception.candidates"] = {"cand_1": {"id": "cand_1"}}
        moved = registry.specs_text()
        self.assertIn("STATE=candidate", moved)
        self.assertNotEqual(text, moved)

    def test_the_catalogue_is_bounded_by_the_registry_limit(self):
        registry = full_registry()
        registry.attach_state(lambda: {}, limit=1200)
        self.assertLessEqual(len(registry.specs_text()), 1200)


# ---------------------------------------------------------------------------
# 2. batch read-only calls
# ---------------------------------------------------------------------------
class BatchReadTest(unittest.TestCase):
    """One call may pack several read-only checks; movement stays serial."""

    def setUp(self):
        self.registry = read_registry()
        self.ctx = make_context()

    def test_a_batch_runs_every_read_only_call_and_merges_the_results(self):
        result = self.registry.call(
            BATCH_TOOL_NAME,
            {"calls": [{"tool": "observe", "args": {"mode": "forward"}},
                       {"tool": "query_memory", "args": {"query": "target"}}]},
            self.ctx,
        )
        self.assertEqual(result.status, STATUS_OK)
        self.assertEqual(result.data["count"], 2)
        self.assertEqual([item["tool"] for item in result.data["results"]],
                         ["observe", "query_memory"])
        self.assertEqual(result.data["ok"], 1)
        self.assertEqual(result.data["empty"], 1)
        # the merged evidence pack is what the model cites
        self.assertIn(("observation", "obs_9"), [(e.kind, e.ref) for e in result.evidence])
        # both inner calls are in the audit log
        self.assertEqual([entry["tool"] for entry in self.ctx.log],
                         [BATCH_TOOL_NAME, "observe", "query_memory"])

    def test_a_batch_costs_one_budget_unit_per_inner_call(self):
        before = self.ctx.budget["tool_calls_left"]
        self.registry.call(
            BATCH_TOOL_NAME,
            {"calls": [{"tool": "observe", "args": {}}, {"tool": "query_memory", "args": {}}]},
            self.ctx,
        )
        self.assertEqual(before - 2, self.ctx.budget["tool_calls_left"])

    def test_a_pose_changing_call_is_refused_with_its_serial_next_step(self):
        result = self.registry.call(
            BATCH_TOOL_NAME,
            {"calls": [{"tool": "observe", "args": {}},
                       {"tool": "approach_candidate", "args": {"candidate_id": "cand_1"}}]},
            self.ctx,
        )
        self.assertEqual(result.status, STATUS_FAILED)
        refused = next(item for item in result.data["results"]
                       if item["tool"] == "approach_candidate")
        self.assertEqual(refused["status"], STATUS_FAILED)
        self.assertEqual(refused["data"]["reason"], "not_read_only")
        self.assertEqual(refused["data"]["next_action"],
                         {"tool": "approach_candidate",
                          "args": {"candidate_id": "cand_1"}})
        # the read-only sibling still ran: one refused call does not void the pack
        observe = next(item for item in result.data["results"] if item["tool"] == "observe")
        self.assertEqual(observe["status"], STATUS_OK)

    def test_unknown_and_nested_calls_are_refused_by_name(self):
        result = self.registry.call(
            BATCH_TOOL_NAME,
            {"calls": [{"tool": "teleport", "args": {}},
                       {"tool": BATCH_TOOL_NAME, "args": {"calls": []}}]},
            self.ctx,
        )
        self.assertEqual(result.status, STATUS_FAILED)
        by_tool = {item["tool"]: item for item in result.data["results"]}
        self.assertEqual(by_tool["teleport"]["data"]["reason"], "unknown_tool")
        self.assertIn("known_tools", by_tool["teleport"]["data"])
        self.assertEqual(by_tool[BATCH_TOOL_NAME]["data"]["reason"], "nested_batch")

    def test_an_empty_or_malformed_batch_is_a_failed_call_not_a_crash(self):
        for args in ({}, {"calls": []}, {"calls": "observe"},
                     {"calls": [{"args": {}}]}, {"calls": [{"tool": "observe", "args": {}}] * 99}):
            with self.subTest(args=str(args)[:40]):
                result = self.registry.call(BATCH_TOOL_NAME, args, self.ctx)
                self.assertEqual(result.status, STATUS_FAILED)
                self.assertTrue(result.data.get("hint"))

    def test_a_repeated_inner_call_is_served_from_the_cache_inside_the_batch(self):
        first = self.registry.call(
            BATCH_TOOL_NAME, {"calls": [{"tool": "observe", "args": {"mode": "forward"}}]},
            self.ctx,
        )
        self.assertEqual(first.data["results"][0]["status"], STATUS_OK)
        second = self.registry.call(
            BATCH_TOOL_NAME, {"calls": [{"tool": "observe", "args": {"mode": "forward"}}]},
            self.ctx,
        )
        self.assertEqual(second.data["results"][0]["status"], STATUS_CACHED)
        self.assertEqual(second.data["cached"], 1)
        self.assertIn("cached_from_turn", second.data["results"][0]["data"])

    def test_the_model_facing_rendering_stays_compact(self):
        result = self.registry.call(
            BATCH_TOOL_NAME,
            {"calls": [{"tool": "observe", "args": {}},
                       {"tool": "query_memory", "args": {"query": "target"}}]},
            self.ctx,
        )
        text = result.to_prompt()
        self.assertLess(len(text), 400)
        self.assertIn("observe", text)
        self.assertIn("query_memory", text)
        self.assertIn("obs_9", text)

    def test_the_registered_batch_tool_advertises_its_schema(self):
        spec = self.registry.get(BATCH_TOOL_NAME)
        self.assertFalse(spec.side_effects)
        self.assertEqual(spec.input_schema["required"], ["calls"])
        item = spec.input_schema["properties"]["calls"]["items"]
        self.assertEqual(item["required"], ["tool"])
        self.assertIn(BATCH_TOOL_NAME, self.registry.specs_text())


# ---------------------------------------------------------------------------
# 3. movement idempotency
# ---------------------------------------------------------------------------
class MovementIdempotencyTest(unittest.TestCase):
    """A completed movement is never flown twice for the same place."""

    def setUp(self):
        self.registry = full_registry()
        self.world = FakeWorld(position=(0.0, 0.0, 0.0))
        self.ctx = make_context(self.world)
        self.ctx.extra["viewpoints_by_id"] = {
            "vp_1": {"viewpoint_id": "vp_1", "position": [1.0, 0.0, 0.0],
                     "reachable": True, "cost_m": 1.0}
        }
        self.ctx.extra["viewpoints"] = [
            {"viewpoint_id": "vp_1", "position": [1.0, 0.0, 0.0], "reachable": True,
             "cost_m": 1.0}
        ]

    def test_a_repeated_request_for_a_reached_place_answers_already_done(self):
        first = self.registry.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, self.ctx)
        self.assertEqual(first.data["task_status"], "arrived")
        legs = len(self.world.applied)

        again = self.registry.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, self.ctx)
        self.assertEqual(again.status, STATUS_CACHED)
        self.assertIs(again.data["already_done"], True)
        self.assertEqual(again.data["task_status"], "arrived")
        self.assertEqual(len(self.world.applied), legs)  # nothing was flown again
        self.assertTrue(again.data["arrival_evidence"])

    def test_resubmitting_with_the_task_id_is_the_same_idempotent_answer(self):
        first = self.registry.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, self.ctx)
        task_id = first.data["task_id"]
        legs = len(self.world.applied)
        again = self.registry.call(
            "go_to_viewpoint", {"viewpoint_id": "vp_1", "task_id": task_id}, self.ctx
        )
        self.assertEqual(again.status, STATUS_CACHED)
        self.assertIs(again.data["already_done"], True)
        self.assertEqual(len(self.world.applied), legs)

    def test_task_status_reports_the_completion_and_its_arrival_evidence(self):
        first = self.registry.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, self.ctx)
        task_id = first.data["task_id"]
        status = self.registry.call("task_status", {"task_id": task_id}, self.ctx)
        self.assertIs(status.data["completed"], True)
        self.assertIs(status.data["arrived"], True)
        self.assertEqual(status.data["task_status"], "arrived")
        self.assertTrue(status.data["arrival_evidence"])
        refs = {(ref["kind"], ref["ref"]) for ref in status.data["arrival_evidence"]}
        self.assertTrue(refs)

    def test_a_place_that_is_not_reached_yet_is_not_already_done(self):
        self.ctx.extra["viewpoints_by_id"]["vp_2"] = {
            "viewpoint_id": "vp_2", "position": [40.0, 0.0, 0.0], "reachable": True,
            "cost_m": 40.0,
        }
        first = self.registry.call("go_to_viewpoint", {"viewpoint_id": "vp_2"}, self.ctx)
        self.assertNotEqual(first.data["task_status"], "arrived")
        self.assertIsNot(first.data.get("already_done"), True)


# ---------------------------------------------------------------------------
# 4. the traceable success chain
# ---------------------------------------------------------------------------
def verify_record(verified=True, evidence=True):
    return {
        "tool": "verify_goal",
        "status": STATUS_OK,
        "data": {"verified": verified, "checks": {"distance": True}},
        "evidence": ([EvidenceRef("observation", "obs_7", "verification").to_dict()]
                     if evidence else []),
    }


def candidate_record():
    return {
        "tool": "ground_target",
        "status": STATUS_OK,
        "data": {"candidate_id": "cand_1", "candidates": [{"id": "cand_1"}]},
        "evidence": [EvidenceRef("observation", "obs_3", "grounding").to_dict()],
    }


def arrival_record(outcome="arrived"):
    return {
        "tool": "approach_candidate",
        "status": STATUS_OK,
        "data": {"outcome": outcome, "task_id": "nav_2", "task_status": "arrived",
                 "candidate_id": "cand_1"},
        "evidence": [EvidenceRef("observation", "obs_4", "navigation pose").to_dict()],
    }


class CompletionEvidenceTest(unittest.TestCase):
    """The chain behind ``success: true``: verification, or identity + arrival."""

    def test_a_verified_goal_is_the_strongest_evidence(self):
        chain = completion_evidence([verify_record()])
        self.assertTrue(chain["verified"])
        self.assertEqual(chain["kind"], "verification")
        self.assertTrue(chain["evidence_trail"])
        self.assertEqual(chain["missing"], [])
        self.assertEqual(chain["evidence_trail"],
                         [{"kind": "observation", "ref": "obs_7", "note": "verification"}])

    def test_a_candidate_plus_an_arrival_is_evidence_too(self):
        chain = completion_evidence([candidate_record(), arrival_record()])
        self.assertTrue(chain["verified"])
        self.assertEqual(chain["kind"], "identity+arrival")
        self.assertEqual(chain["candidate_ids"], ["cand_1"])

    def test_an_unmet_verification_does_not_count(self):
        chain = completion_evidence([verify_record(verified=False), candidate_record()])
        self.assertFalse(chain["verified"])
        self.assertTrue(chain["missing"])

    def test_a_candidate_without_an_arrival_is_not_completion_evidence(self):
        chain = completion_evidence([candidate_record(), arrival_record("partial")])
        self.assertFalse(chain["verified"])
        self.assertTrue(any("arrival" in item for item in chain["missing"]))

    def test_a_chain_without_evidence_references_is_not_a_chain(self):
        chain = completion_evidence([verify_record(evidence=False)])
        self.assertFalse(chain["verified"])
        self.assertTrue(any("reference" in item for item in chain["missing"]))

    def test_nothing_at_all_is_reported_as_missing_both_ways(self):
        chain = completion_evidence([])
        self.assertFalse(chain["verified"])
        self.assertGreaterEqual(len(chain["missing"]), 2)


class FinishClaimTest(unittest.TestCase):
    """``finish`` is a registered tool and only claims what the trail backs."""

    def setUp(self):
        self.registry = full_registry()
        self.world = FakeWorld(position=(2.2, 0.0, 0.0))
        self.ctx = make_context(self.world)

    def test_a_bare_finish_claims_nothing(self):
        result = self.registry.call("finish", {"reason": "gave up"}, self.ctx)
        self.assertEqual(result.status, STATUS_OK)
        self.assertIsNone(result.data["success"])
        self.assertEqual(result.data["claim"], "none")
        self.assertIs(result.data["claimed"], False)

    def test_success_without_evidence_is_downgraded_to_an_unverified_claim(self):
        result = self.registry.call("finish", {"reason": "done", "success": True}, self.ctx)
        self.assertIs(result.data["success"], False)
        self.assertEqual(result.data["claim"], "unverified")
        self.assertIs(result.data["claimed"], True)
        self.assertIs(result.data["verified"], False)
        self.assertTrue(result.data["missing"])
        self.assertTrue(result.data["hint"])

    def test_a_verified_trail_backs_the_claim(self):
        self.registry.call("verify_goal", {"reason": "check"}, self.ctx)
        result = self.registry.call("finish", {"reason": "done", "success": True}, self.ctx)
        self.assertIs(result.data["success"], True)
        self.assertEqual(result.data["claim"], "verified")
        self.assertTrue(result.data["evidence_trail"])
        self.assertTrue(all({"kind", "ref"} <= set(item)
                            for item in result.data["evidence_trail"]))

    def test_a_declared_failure_stays_an_honest_give_up(self):
        result = self.registry.call("finish", {"reason": "cannot find it", "success": False},
                                    self.ctx)
        self.assertIs(result.data["success"], False)
        self.assertEqual(result.data["claim"], "give_up")

    def test_the_evidence_pack_records_verifications_and_arrivals(self):
        self.registry.call("verify_goal", {"reason": "check"}, self.ctx)
        pack = evidence_pack(self.ctx)
        self.assertTrue(pack["verifications"])
        self.assertIs(self.ctx.extra[EVIDENCE_PACK_KEY]["verifications"][0]["verified"], True)

    def test_the_registry_advertises_finish_with_its_arguments(self):
        spec = self.registry.get("finish")
        self.assertIn("success", spec.input_schema["properties"])
        self.assertIn("reason", spec.input_schema["properties"])
        self.assertFalse(spec.side_effects)


class RuntimeClaimTest(unittest.TestCase):
    """The runtime's verdict follows the evidence, not the model's word."""

    def _run(self, script):
        from vlm_nav.runtime import AgentRuntime

        registry = full_registry()
        world = FakeWorld(position=(2.2, 0.0, 0.0))
        log = AgentRuntime(registry, world, None, scripted_vlm(script), budget=8,
                           max_turns=6).run("reach it")
        return log

    def test_an_unbacked_success_claim_is_not_counted_as_success(self):
        log = self._run([
            {"tool": "observe", "args": {"mode": "forward"}},
            {"tool": "finish", "args": {"reason": "I am there", "success": True}},
        ])
        self.assertEqual(log.finish_reason, "finish")
        self.assertIsNot(log.success, True)
        finish = log.turns[-1]
        self.assertEqual(finish.data["claim"], "unverified")
        self.assertIs(finish.data["verified"], False)

    def test_a_verified_episode_carries_the_evidence_trail(self):
        log = self._run([
            {"tool": "verify_goal", "args": {"reason": "check"}},
            {"tool": "finish", "args": {"reason": "verified", "success": True}},
        ])
        self.assertIs(log.success, True)
        finish = log.turns[-1]
        self.assertEqual(finish.data["claim"], "verified")
        self.assertTrue(finish.data["evidence_trail"])


# ---------------------------------------------------------------------------
# 5. MCP parity
# ---------------------------------------------------------------------------
class McpDispatchTest(unittest.TestCase):
    """The MCP side serves the same state, the same pack and the same tools."""

    def setUp(self):
        self.ctx = make_context()
        self.session = mcp_server.build_session(self.ctx, tools=full_registry())

    def test_the_services_offer_the_dispatch_and_completion_tools(self):
        for name in (BATCH_TOOL_NAME, "finish"):
            with self.subTest(name=name):
                self.assertIn(name, mcp_server.ALL_TOOL_NAMES)
                self.assertIn(name, self.session.names())
                spec = self.session.registry.get(name)
                self.assertEqual(spec.input_schema, mcp_server.TOOL_CONTRACTS[name]["input_schema"])

    def test_the_mcp_catalogue_matches_the_local_state_order(self):
        local = tool_catalog(self.session.registry, state=tool_state(self.ctx.extra), limit=9000)
        catalog = self.session.catalog("env")
        self.assertEqual(catalog.state.name, local.state.name)
        served = set(self.session.names("env"))
        self.assertEqual([name for name in local.order if name in served],
                         [name for name in catalog.order])
        inventory = mcp_server.service_inventory("env", self.session)
        self.assertEqual(inventory["state"]["name"], local.state.name)
        self.assertEqual(inventory["catalog"]["order"], list(catalog.order))
        self.assertEqual([tool["name"] for tool in inventory["tools"]], list(catalog.order))

    def test_the_state_moves_the_mcp_listing_too(self):
        before = self.session.catalog().order
        self.session.call("observe", {"mode": "forward"})
        self.ctx.extra["perception.candidates"] = {"cand_1": {"id": "cand_1"}}
        after = self.session.catalog().order
        self.assertEqual(after[0], "compare_candidates")
        self.assertNotEqual(before, after)

    def test_a_batch_call_over_mcp_shares_the_local_evidence_pack(self):
        result = self.session.call(
            BATCH_TOOL_NAME, {"calls": [{"tool": "observe", "args": {"mode": "forward"}}]}
        )
        self.assertEqual(result.status, STATUS_OK)
        self.assertEqual(result.data["results"][0]["tool"], "observe")
        # the same episode log: the inner call is what the local adapter sees
        self.assertEqual([entry["tool"] for entry in self.session.tool_log()],
                         [BATCH_TOOL_NAME, "observe"])

    def test_the_mcp_finish_and_the_local_finish_agree(self):
        over_mcp = self.session.call("finish", {"reason": "done", "success": True})
        direct = full_registry().call("finish", {"reason": "done", "success": True},
                                      make_context())
        for key in ("success", "claim", "verified", "missing"):
            self.assertEqual(over_mcp.data[key], direct.data[key], key)


# ---------------------------------------------------------------------------
# 6. the runner: budget, catalogue and the claim in summary.json
# ---------------------------------------------------------------------------
def _load_runner():
    name = "diffaero_script_run_vlm_nav_tools_dispatch"
    if name in sys.modules:
        return sys.modules[name]
    path = REPO_ROOT / "scripts" / "run_vlm_nav_tools.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _make_scene(path: Path, target=(2.0, 0.5, 0.0)) -> None:
    header = "\n".join([
        "ply", "format binary_little_endian 1.0", "element vertex 4",
        "property float x", "property float y", "property float z",
        "property float f_dc_0", "property float f_dc_1", "property float f_dc_2",
        "property float opacity", "property float scale_0", "property float scale_1",
        "property float scale_2", "property float rot_0", "property float rot_1",
        "property float rot_2", "property float rot_3", "end_header",
    ])

    def logit(p):
        return math.log(p / (1 - p))

    rows = []
    for offset in ((0.0, 0.0, 0.0), (0.1, 0.1, 0.0), (0.0, 0.1, 0.1), (0.1, 0.0, 0.1)):
        rows.append((target[0] + offset[0], target[1] + offset[1], target[2] + offset[2],
                     2.0, -1.0, -1.0, logit(0.9), math.log(0.25), math.log(0.25),
                     math.log(0.25), 1.0, 0.0, 0.0, 0.0))
    with open(path, "wb") as handle:
        handle.write((header + "\n").encode("ascii"))
        for row in rows:
            handle.write(struct.pack("<14f", *row))


class RunnerDispatchTest(unittest.TestCase):
    """The runner wires the state catalogue, its budget and the claim trail."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.runner = _load_runner()
        self.ply = self.tmp / "scene.ply"
        _make_scene(self.ply)

    def _args(self, out_dir, max_turns=6, tool_budget=12, success_dist=2.5, **extra):
        argv = [
            "--world", "cpu",
            "--ply", str(self.ply),
            "--target-name", "target",
            "--target-pos", "2.0", "0.5", "0.0",
            "--target-radius", "0.2",
            "--start-pos", "0", "0", "0.2",
            "--instruction", "Find the target and fly to it.",
            "--backend", "mock",
            "--render-size", "64", "64",
            "--max-turns", str(max_turns),
            "--tool-budget", str(tool_budget),
            "--success-dist", str(success_dist),
            "--out", str(out_dir),
        ]
        for key, value in extra.items():
            argv.extend([f"--{key.replace('_', '-')}", str(value)])
        return self.runner.parse_args(argv)

    def _world(self):
        from vlm_nav.world import SplatWorldCPU, TargetSpec

        return SplatWorldCPU(
            ply_path=self.ply,
            target=TargetSpec(name="target", position_flu=(2.0, 0.5, 0.0), radius_m=0.2),
            start_flu=(0.0, 0.0, 0.2),
            start_yaw_deg=0.0,
            width=64,
            height=64,
            hfov_deg=60.0,
        )

    def _vlm(self, script):
        """Scripted mock backend that records what the runtime sent it."""

        class Recording(self.runner.ScriptedToolVLM):
            def __init__(self):
                super().__init__([dict(entry) for entry in script])
                self.prompts: list = []

            def decide_with_tools(self, frames, system_prompt, user_prompt, tools_text="",
                                  context=None):
                self.prompts.append({"system": system_prompt, "tools_text": tools_text,
                                     "user": user_prompt})
                return super().decide_with_tools(frames, system_prompt, user_prompt,
                                                 tools_text, context)

            def decide(self, frame_chw, system_prompt, user_prompt, context=None):
                self.prompts.append({"system": system_prompt, "tools_text": "",
                                     "user": user_prompt})
                return super().decide(frame_chw, system_prompt, user_prompt, context)

        return Recording()

    def _run(self, out_dir, script, **kwargs):
        vlm = self._vlm(script)
        summary = self.runner.run_episode(self._world(), vlm, self._args(out_dir, **kwargs),
                                          out_dir, episode_id="ep_dispatch")
        return summary, vlm

    def test_the_prompt_carries_the_state_catalogue_whole(self):
        out_dir = self.tmp / "catalogue"
        summary, vlm = self._run(out_dir, [
            {"tool": "observe", "args": {"mode": "forward"}},
            {"tool": "finish", "args": {"reason": "done", "success": False}},
        ])
        tools_text = vlm.prompts[0]["tools_text"]
        self.assertTrue(tools_text)
        self.assertNotIn("[truncated", tools_text)
        self.assertIn("STATE=", tools_text)
        catalog = summary["prompt_budget"]["catalog"]
        self.assertTrue(catalog["full"])
        for name in catalog["full"]:
            self.assertIn(f"- {name}:", tools_text)
        self.assertLessEqual(len(tools_text), summary["prompt_budget"]["catalog_chars"])
        # every registered tool is still named somewhere in the catalogue
        for name in summary["toolset_detail"]["registered"]:
            self.assertIn(name, tools_text)
        self.assertNotIn("[truncated", vlm.prompts[0]["system"])
        # and the state's own tools really arrive with their full arguments: the
        # exact entry line of the registry (description + complete schema)
        modules, _ = self.runner.load_m1_modules()
        registry, _, _ = self.runner.build_registry(modules, toolset="full")
        for name in catalog["full"]:
            spec = registry.get(name)
            self.assertIn(
                f"- {name}: {spec.description} "
                f"args={json.dumps(spec.input_schema, separators=(',', ':'))}",
                tools_text,
            )
        on_disk = json.loads((out_dir / "summary.json").read_text())["prompt_budget"]
        self.assertEqual(on_disk["catalog"], catalog)

    def test_the_prompt_budget_reports_what_the_runtime_rendered(self):
        out_dir = self.tmp / "budget"
        summary, vlm = self._run(out_dir, [
            {"tool": "observe", "args": {"mode": "forward"}},
            {"tool": "finish", "args": {"reason": "done", "success": False}},
        ])
        budget = summary["prompt_budget"]

        self.assertEqual(budget["system_chars"], self.runner.SYSTEM_CHARS)
        self.assertTrue(budget["catalogue_complete"], budget)
        self.assertFalse(budget["catalogue_truncated"], budget)
        self.assertFalse(budget["doc_truncated"], budget)
        # the runtime's own numbers, not the runner's constants
        self.assertEqual(budget["catalogue_chars"], len(vlm.prompts[-1]["tools_text"]))
        self.assertGreater(budget["rendered_chars"], 0)
        self.assertLessEqual(budget["catalogue_chars"], budget["catalog_chars"])

    def test_the_executor_auto_step_is_recorded_without_a_model_turn(self):
        out_dir = self.tmp / "auto_step"
        summary, vlm = self._run(out_dir, [
            {"tool": "update_plan", "args": {"steps": ["observe"]}},
            {"tool": "finish", "args": {"reason": "done", "success": False}},
        ])

        self.assertEqual(summary["executor"]["auto_steps"], 1)
        self.assertEqual(summary["executor"]["saved_vlm_turns"], 1)
        self.assertEqual(summary["executor"]["switch"], "on")
        self.assertEqual(summary["executor"]["steps"][0]["origin"], "subtask_goal")
        self.assertEqual(summary["executor_auto_steps"], 1)
        self.assertEqual(summary["vlm_turns_saved"], 1)
        # two model decisions, three tool calls: the observe was free
        self.assertEqual(summary["vlm_calls"], 2)
        self.assertEqual(summary["turns"], 2)
        self.assertEqual(summary["tool_turns"], 3)
        entries = [
            json.loads(line)
            for line in (out_dir / "tool_trace.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual([entry["tool"] for entry in entries], ["update_plan", "observe", "finish"])
        self.assertTrue(entries[1]["auto"])
        self.assertEqual(entries[1]["auto_origin"], "subtask_goal")
        # the model saw exactly two turns: the plan turn and the finish
        self.assertEqual(len(vlm.prompts), 2)

    def test_the_executor_auto_step_switch_reaches_the_runtime(self):
        out_dir = self.tmp / "auto_step_off"
        summary, vlm = self._run(out_dir, [
            {"tool": "update_plan", "args": {"steps": ["observe"]}},
            {"tool": "observe", "args": {"mode": "forward"}},
            {"tool": "finish", "args": {"reason": "done", "success": False}},
        ], executor_auto_steps="off")

        self.assertEqual(summary["executor"]["requested"], "off")
        self.assertEqual(summary["executor"]["switch"], "off")
        self.assertEqual(summary["executor"]["auto_steps"], 0)
        self.assertEqual(summary["executor_auto_steps"], 0)
        # the same executed trajectory now costs a model turn per call
        self.assertEqual(summary["vlm_calls"], 3)
        self.assertEqual(summary["turns"], 3)
        self.assertEqual(summary["tool_turns"], 3)
        self.assertFalse(summary["switches"]["applied"]["executor_auto_steps"])
        self.assertEqual(len(vlm.prompts), 3)

    def test_an_unbacked_success_claim_is_not_a_success_in_the_summary(self):
        out_dir = self.tmp / "claim"
        summary, _ = self._run(out_dir, [
            {"tool": "observe", "args": {"mode": "forward"}},
            {"tool": "finish", "args": {"reason": "I am there", "success": True}},
        ])
        self.assertTrue(summary["success_computed"], summary)
        self.assertFalse(summary["success"], summary)
        self.assertTrue(summary["completion"]["claimed"])
        self.assertFalse(summary["completion"]["verified"])
        self.assertFalse(summary["success_evidence"], summary)
        self.assertTrue(summary["completion"]["missing"])
        on_disk = json.loads((out_dir / "summary.json").read_text())
        self.assertIs(on_disk["completion"]["verified"], False)

    def test_a_verified_chain_is_recorded_in_the_summary(self):
        out_dir = self.tmp / "verified"
        summary, _ = self._run(out_dir, [
            {"tool": "observe", "args": {"mode": "forward"}},
            {"tool": "verify_goal", "args": {"reason": "check"}},
            {"tool": "finish", "args": {"reason": "verified", "success": True}},
        ])
        self.assertTrue(summary["success"], summary)
        self.assertTrue(summary["completion"]["verified"])
        self.assertTrue(summary["completion"]["evidence_trail"])
        self.assertEqual(summary["completion"]["kind"], "verification")
        self.assertTrue(summary["success_evidence"])

    def test_the_fallback_loop_routes_finish_through_the_registered_tool(self):
        """Both runtimes share one finish contract (the fallback is not exempt)."""
        runner = self.runner
        registry = ToolRegistry()
        # the fallback stack carries its own legacy propose_viewpoints; the
        # module tool set is merged on top (as ``build_registry`` does by name)
        runner.register_fallback_tools(
            registry, exclude=("propose_viewpoints", "navigate", "verify_goal"))
        register_navigation_tools(registry)
        out_dir = self.tmp / "fallback"
        ctx = runner.make_tool_context(self._world(), runner._FallbackMemory(),
                                       self._args(out_dir), out_dir, "ep_fb")
        runtime = runner._ToolRuntimeFallback(
            world=ctx.world,
            vlm=scripted_vlm([{"tool": "finish", "args": {"reason": "done", "success": True}}]),
            registry=registry, ctx=ctx, instruction="reach it", max_turns=4,
            context_frames=0, success_dist_m=2.5,
        )
        result = runtime.run()
        finish = next(entry for entry in result["trace"] if entry["tool"] == "finish")
        self.assertEqual(finish["data"]["claim"], "unverified")
        self.assertIs(finish["data"]["success"], False)


if __name__ == "__main__":
    unittest.main()
