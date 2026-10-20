"""Unit tests for the search skill (``vlm_nav.skills_search``).

The skill is exercised against the *real* perception tools (a fake world with
``scan_views`` / depth, a scripted grounding backend) plus two double-stand-in
tools for the movement half (``propose_viewpoints`` / ``go_to_viewpoint``), so
the tests check the routine the skill runs - sweep, pick a view, ground, zoom,
propose a viewpoint, fly one leg - and the log it returns for it, without a
renderer, a model or a GPU.
"""

import json
import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from vlm_nav import skills_search
from vlm_nav.tool_impl_perception import register_perception_tools
from vlm_nav.tools import ToolContext, ToolRegistry, ToolResult, ToolSpec


class FakeWorld:
    """Kinematic CPU world with a scan, a depth stub and no obstacles."""

    def __init__(self, width=64, height=32, position=(0.0, 0.0, 1.0), yaw_deg=0.0, depth=6.0):
        self.width, self.height = int(width), int(height)
        self.state = SimpleNamespace(
            position_flu=np.asarray(position, dtype=np.float64),
            yaw_rad=math.radians(yaw_deg),
        )
        self.depth = depth
        self.scan_calls = []
        self.scan_positions = []

    @property
    def position_flu(self):
        return self.state.position_flu

    @property
    def yaw_rad(self):
        return self.state.yaw_rad

    def render(self):
        frame = torch.zeros(3, self.height, self.width)
        frame[0, 0, 0] = (math.degrees(self.state.yaw_rad) % 360.0) / 360.0
        return frame

    def scan_views(self, n_views=8, pitch_down_deg=0.0, spread_deg=None):
        self.scan_calls.append({"n_views": n_views, "pitch_down_deg": pitch_down_deg})
        self.scan_positions.append(tuple(float(v) for v in self.state.position_flu))
        yaw0 = self.state.yaw_rad
        frames = []
        try:
            for k in range(n_views):
                self.state.yaw_rad = yaw0 + 2.0 * math.pi * k / n_views
                frames.append(self.render())
        finally:
            self.state.yaw_rad = yaw0
        return frames

    def depth_at_pixel(self, pixel):
        return self.depth


class ScriptedGroundingVLM:
    """Grounding backend answering a per-observation-id script (reply or None)."""

    name = "scripted"

    def __init__(self, replies=None, default='{"candidates": []}'):
        self.replies = dict(replies or {})
        self.default = default
        self.calls = []

    def decide(self, frames, system_prompt, user_prompt, context=None):
        context = context or {}
        obs_id = str(context.get("observation_id") or "")
        self.calls.append({"observation_id": obs_id, "context": context})
        reply = self.replies.get(obs_id, self.default)
        if isinstance(reply, Exception):
            raise reply
        return reply


def comparison_reply(picks, *, views=None, note="red object beside the lamp",
                     confidence=0.8, relation_visible=True, box=(300, 300, 520, 520)):
    """One compare-views reply: the picked views look like the target, the rest do not."""
    picks = list(picks)
    if views is None:
        views = max(picks or [1]) if picks else 1
        views = max(int(views), 8)
    entries = []
    for index in range(1, int(views) + 1):
        hit = index in picks
        entry = {
            "view": index,
            "looks_like_target": bool(hit),
            "confidence": float(confidence) if hit else 0.1,
            "reason": note if hit else "no such object in this frame",
        }
        if hit and box:
            entry["box"] = [int(v) for v in box]
        entries.append(entry)
    return json.dumps(
        {
            "views": entries,
            "best_view": picks[0] if picks else None,
            "relation_visible": bool(relation_visible),
        }
    )


class CompareAwareVLM(ScriptedGroundingVLM):
    """Scripted backend answering both the view comparison and the grounding.

    The comparison call is recognized by its context (``step=compare_views``):
    it is scripted with the 1-based *view* indexes of the compared list to flag,
    so a test can name the view the comparison should pick without knowing which
    observation id the sweep gave it.
    """

    def __init__(self, *, picks=(2,), compare_reply=None, compare_error=None, **kwargs):
        super().__init__(**kwargs)
        self.picks = list(picks)
        self.compare_reply = compare_reply
        self.compare_error = compare_error
        self.compare_calls = []

    def decide(self, frames, system_prompt, user_prompt, context=None):
        context = context or {}
        if str(context.get("step") or "") == "compare_views":
            observation_ids = [str(item) for item in (context.get("observation_ids") or [])]
            self.compare_calls.append(
                {
                    "n_frames": len(frames),
                    "observation_ids": observation_ids,
                    "user_prompt": user_prompt,
                    "system_prompt": system_prompt,
                    "context": context,
                }
            )
            if self.compare_error is not None:
                raise self.compare_error
            if self.compare_reply is not None:
                return self.compare_reply
            return comparison_reply(self.picks, views=len(observation_ids))
        return super().decide(frames, system_prompt, user_prompt, context)


def make_search_registry(world, vlm, *, moved=None, viewpoints=3):
    """Perception tools + viewpoint doubles on one registry (the skill's home)."""
    registry = ToolRegistry()
    register_perception_tools(registry, vlm=vlm)
    if "propose_viewpoints" not in registry.names():
        registry.register(
            ToolSpec(
                name="propose_viewpoints",
                description="propose observation points",
                input_schema={"type": "object", "properties": {"count": {"type": "integer"}}},
                handler=lambda ctx, args: _propose(ctx, viewpoints),
            )
        )
    if "go_to_viewpoint" not in registry.names():
        registry.register(
            ToolSpec(
                name="go_to_viewpoint",
                description="fly to an observation point",
                input_schema={"type": "object", "properties": {"viewpoint_id": {"type": "string"}}},
                handler=lambda ctx, args: _go_to(ctx, args, world, moved),
                side_effects=True,
            )
        )
    return registry


def _propose(ctx, count):
    store = ctx.extra.setdefault("viewpoints_by_id", {})
    seq = int(ctx.extra.get("_viewpoint_seq", 0))
    viewpoints = []
    for index in range(int(count)):
        seq += 1
        entry = {
            "viewpoint_id": f"vp_{seq}",
            "position": [float(index + 1), 0.0, 1.5],
            "distance_m": float(index + 1),
            "bearing_deg": 0.0,
            "reachable": True,
            "expected_coverage": 0.2,
            "utility": 0.5,
        }
        store[entry["viewpoint_id"]] = dict(entry)
        viewpoints.append(entry)
    ctx.extra["_viewpoint_seq"] = seq
    return ToolResult("ok", {"count": len(viewpoints), "viewpoints": viewpoints,
                             "viewpoint_id": viewpoints[0]["viewpoint_id"] if viewpoints else None})


def _go_to(ctx, args, world, moved):
    viewpoint_id = str(args.get("viewpoint_id") or "")
    store = ctx.extra.get("viewpoints_by_id") or {}
    if viewpoint_id not in store:
        return ToolResult("failed", {"error": "unknown_viewpoint_id", "viewpoint_id": viewpoint_id})
    entry = store[viewpoint_id]
    if moved is not None:
        world.state.position_flu = np.asarray(entry["position"], dtype=np.float64)
        moved.append(viewpoint_id)
    return ToolResult(
        "ok",
        {
            "task_id": "nav_1",
            "viewpoint_id": viewpoint_id,
            "status": "ok",
            "outcome": "arrived",
            "progress_m": entry["distance_m"],
            "remaining_m": 0.0,
        },
    )


class MovableWorld(FakeWorld):
    """FakeWorld plus the pose-changing action the real navigation tools call.

    ``tool_impl_navigation`` flies a leg through ``world.apply_action(NavAction)``
    (a body-frame waypoint: forward/right/up in metres and a yaw change), so this
    world makes the *real* movement tools run against the *real* search skill.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.actions = []

    def apply_action(self, action):
        from vlm_nav.actions import body_delta_to_world

        self.actions.append(action)
        yaw = float(self.state.yaw_rad) + math.radians(float(getattr(action, "yaw_deg", 0.0) or 0.0))
        forward = float(getattr(action, "forward_m", 0.0) or 0.0)
        right = float(getattr(action, "right_m", 0.0) or 0.0)
        up = float(getattr(action, "up_m", 0.0) or 0.0)
        delta = body_delta_to_world(forward, -right, up, yaw)
        self.state.position_flu = (
            np.asarray(self.state.position_flu, dtype=np.float64) + np.asarray(delta, dtype=np.float64)
        )
        self.state.yaw_rad = yaw
        return "moved"


def make_ctx(world, **extra):
    return ToolContext(world=world, episode_id="ep_search", extra=dict(extra))


CANDIDATES = (
    '{"candidates": [{"box": [400, 400, 600, 600], "confidence": 0.85, '
    '"attrs": {"color": "red"}}]}'
)


class SearchTargetTest(unittest.TestCase):
    def test_a_found_target_comes_back_with_its_step_log(self):
        world = FakeWorld()
        vlm = ScriptedGroundingVLM(default=CANDIDATES)
        registry = make_search_registry(world, vlm)
        ctx = make_ctx(world)

        result = registry.call("search_target", {"instruction": "red chair"}, ctx)

        self.assertEqual(result.status, "ok")
        self.assertTrue(result.data["candidate_id"].startswith("cand_"))
        self.assertTrue(result.data["confirmed"])
        self.assertEqual(result.data["instruction"], "red chair")
        # the step log says what was done and why, in order
        self.assertEqual([step["step"] for step in result.data["steps"]][0], "observe")
        self.assertIn("ground_target", [step["step"] for step in result.data["steps"]])
        for step in result.data["steps"]:
            self.assertEqual(set(step), {"step", "why", "obs_ids", "result"})
            self.assertTrue(step["why"])
            self.assertIsInstance(step["obs_ids"], list)
            self.assertIsInstance(step["result"], dict)
        # the real cost of the routine is published, not hidden in one call
        self.assertEqual(result.data["vlm_calls"], len(vlm.calls))
        self.assertEqual(result.data["cost"]["vlm_calls"], result.data["vlm_calls"])
        self.assertEqual(result.data["cost"]["tool_calls"], len(result.data["steps"]))
        # the scan's coverage travels with the answer
        coverage = result.data["coverage"]
        self.assertEqual(len(coverage["checked_deg"]), 8)
        json.dumps(result.to_dict())

    def test_a_single_view_conflict_continues_to_a_second_viewpoint_without_low_parallax_triangulation(self):
        """The sweep must not stop on a distance-uncertain candidate.

        Round 1 sees the target but neither its depth nor its model range can
        localise it (dispersed samples, 3 m model). The routine changes place
        and grounds again. This pair's parallax is only ~8 deg, below the
        10-deg stereo floor: the second depth consensus is reported, but the
        bearings are not triangulated (v6 bedroom's biased 5.15-deg pair).
        """
        class DispersedDepthWorld(FakeWorld):
            def depth_at_pixel(self, pixel):
                return 7.0 if pixel[0] < 775.0 else (12.0 if pixel[0] < 825.0 else 22.0)

        world = DispersedDepthWorld(position=(0.0, 0.0, 1.0), yaw_deg=0.0)
        world.hfov_deg = 130.0
        # the same world object at (3.7, -4.7, 1) is at u=800 from the first
        # pose and u=896 after the viewpoint move: two different bearings
        view_one = (
            '{"candidates": [{"box": [750, 450, 100, 100], '
            '"distance_m": 6.0, "confidence": 0.8}]}'
        )
        view_two = (
            '{"candidates": [{"box": [846, 450, 100, 100], '
            '"distance_m": 6.0, "confidence": 0.8}]}'
        )
        vlm = ScriptedGroundingVLM(replies={"obs_1": view_one, "obs_9": view_two})
        moved: list[str] = []
        registry = make_search_registry(world, vlm, moved=moved)
        ctx = make_ctx(world)

        result = registry.call(
            "search_target",
            {"instruction": "the bookshelf", "budget_turns": 2, "max_steps": 20},
            ctx,
        )

        self.assertEqual(result.status, "ok", result.data)
        self.assertTrue(result.data["confirmed"])
        candidate = result.data["candidates"][0]
        self.assertNotEqual(candidate["position_source"], "triangulated")
        self.assertNotIn("triangulation", candidate)
        self.assertIs(candidate["distance_uncertain"], False)
        tools = [step["step"] for step in result.data["steps"]]
        self.assertIn("go_to_viewpoint", tools)
        self.assertGreaterEqual(tools.count("ground_target"), 2)
        self.assertEqual(len(world.scan_positions), 2)
        self.assertNotEqual(world.scan_positions[0], world.scan_positions[1])

    def test_the_grounding_calls_stop_at_the_first_view_with_candidates(self):
        world = FakeWorld()
        vlm = ScriptedGroundingVLM(
            replies={"obs_1": CANDIDATES, "obs_2": CANDIDATES, "obs_3": CANDIDATES}
        )
        registry = make_search_registry(world, vlm)
        ctx = make_ctx(world)

        result = registry.call("search_target", {"instruction": "the red chair"}, ctx)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["vlm_calls"], 1)
        self.assertEqual(result.data["candidate_id"], result.data["candidates"][0]["id"])
        grounded = [step for step in result.data["steps"] if step["step"] == "ground_target"]
        self.assertEqual(len(grounded), 1)

    def test_no_candidate_moves_to_a_viewpoint_and_says_what_it_expects(self):
        world = FakeWorld()
        vlm = ScriptedGroundingVLM()
        moved = []
        registry = make_search_registry(world, vlm, moved=moved)
        ctx = make_ctx(world)

        result = registry.call(
            "search_target", {"instruction": "red chair", "budget_turns": 1}, ctx
        )

        self.assertEqual(result.status, "empty")
        self.assertIsNone(result.data["candidate_id"])
        steps = {step["step"]: step for step in result.data["steps"]}
        self.assertIn("propose_viewpoints", steps)
        self.assertIn("go_to_viewpoint", steps)
        self.assertEqual(moved, [steps["go_to_viewpoint"]["result"]["viewpoint_id"]])
        # the viewpoint step states which information the new place is expected
        # to add, instead of moving on faith
        expected = steps["go_to_viewpoint"]["result"]["expected_information"]
        self.assertTrue(expected["why"])
        self.assertIn("viewpoint_id", expected)
        json.dumps(result.to_dict())

    def test_a_second_round_rescans_after_moving(self):
        world = FakeWorld()
        vlm = ScriptedGroundingVLM()
        moved = []
        registry = make_search_registry(world, vlm, moved=moved)
        ctx = make_ctx(world)

        result = registry.call(
            # a second sweep-and-fly round costs more inner steps than the cap
            # one call gets by default, so it has to be asked for
            "search_target", {"instruction": "red chair", "budget_turns": 2, "max_steps": 16}, ctx
        )

        self.assertEqual(result.status, "empty")
        # every round ends by flying to the viewpoint it proposed, so the second
        # sweep happens somewhere else
        self.assertEqual(moved[0], "vp_1")
        self.assertEqual(len(moved), 2)
        self.assertEqual(len(world.scan_positions), 2)
        self.assertNotEqual(world.scan_positions[0], world.scan_positions[1])
        tools = [step["step"] for step in result.data["steps"]]
        self.assertEqual(tools.count("observe"), 2)
        first_move = tools.index("go_to_viewpoint")
        second_scan = tools.index("observe", tools.index("observe") + 1)
        self.assertLess(first_move, second_scan)
        self.assertEqual(result.data["cost"]["turns"], 2)
        self.assertGreaterEqual(len(vlm.calls), 2)
        self.assertTrue(result.data["next_action"])

    def test_the_budget_bounds_the_grounding_calls(self):
        world = FakeWorld()
        vlm = ScriptedGroundingVLM()
        registry = make_search_registry(world, vlm)
        ctx = make_ctx(world)

        result = registry.call(
            "search_target",
            {"instruction": "red chair", "budget_turns": 1, "max_groundings": 2,
             "allow_move": False},
            ctx,
        )

        self.assertEqual(result.status, "empty")
        # 2 views x (view + one zoom) x (first ask + box-only retry)
        self.assertLessEqual(len(vlm.calls), 8)
        self.assertEqual(result.data["vlm_calls"], len(vlm.calls))
        self.assertEqual(result.data["cost"]["budget_turns"], 1)
        self.assertIsNone(result.data["candidate_id"])

    def test_moving_can_be_switched_off(self):
        world = FakeWorld()
        vlm = ScriptedGroundingVLM()
        moved = []
        registry = make_search_registry(world, vlm, moved=moved)
        ctx = make_ctx(world)

        result = registry.call(
            "search_target",
            {"instruction": "red chair", "budget_turns": 1, "allow_move": False},
            ctx,
        )

        self.assertEqual(moved, [])
        tools = [step["step"] for step in result.data["steps"]]
        self.assertNotIn("go_to_viewpoint", tools)
        self.assertIn("propose_viewpoints", tools)
        self.assertEqual(result.data["next_action"]["tool"], "go_to_viewpoint")

    def test_a_small_target_is_looked_for_in_a_crop_of_the_view(self):
        # the swept frames answer empty; the magnified crop of a view holds it
        world = FakeWorld()
        scan_ids = set()

        class CropAwareVLM(ScriptedGroundingVLM):
            def decide(self, frames, system_prompt, user_prompt, context=None):
                obs_id = str((context or {}).get("observation_id") or "")
                if obs_id and obs_id not in scan_ids:
                    self.calls.append({"observation_id": obs_id, "context": context or {}})
                    return CANDIDATES
                return super().decide(frames, system_prompt, user_prompt, context)

        vlm = CropAwareVLM()
        registry = make_search_registry(world, vlm)
        ctx = make_ctx(world)
        scan_ids.update(f"obs_{index}" for index in range(1, 9))

        result = registry.call(
            "search_target", {"instruction": "red chair", "budget_turns": 1}, ctx
        )

        self.assertEqual(result.status, "ok")
        steps = [step["step"] for step in result.data["steps"]]
        self.assertIn("inspect_region", steps)
        self.assertGreaterEqual(result.data["vlm_calls"], 2)
        self.assertTrue(result.data["confirmed"])
        crop_step = next(step for step in result.data["steps"] if step["step"] == "inspect_region")
        self.assertTrue(crop_step["obs_ids"])

    def test_the_skill_is_registered_with_its_schema(self):
        registry = ToolRegistry()
        register_perception_tools(registry)

        spec = registry.get("search_target")
        self.assertEqual(spec.input_schema["required"], ["instruction"])
        self.assertTrue(spec.side_effects)
        self.assertIn("search_target", registry.specs_text())
        self.assertIn("instruction", spec.input_schema["properties"])

    def test_a_missing_instruction_fails_without_touching_the_world(self):
        world = FakeWorld()
        registry = make_search_registry(world, ScriptedGroundingVLM())
        ctx = make_ctx(world)

        result = registry.call("search_target", {}, ctx)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["error"], "missing_instruction")
        self.assertEqual(world.scan_calls, [])

    def test_without_a_grounding_backend_it_stops_after_the_scan(self):
        world = FakeWorld()
        registry = make_search_registry(world, None)
        ctx = make_ctx(world)

        result = registry.call("search_target", {"instruction": "red chair"}, ctx)

        self.assertEqual(result.status, "empty")
        self.assertEqual(result.data["reason"], "no_vlm_backend")
        # the sweep still happened and its coverage is reported
        self.assertTrue(result.data["coverage"]["checked_deg"])
        self.assertEqual(result.data["vlm_calls"], 0)

    def test_the_scan_honors_the_requested_view_count(self):
        world = FakeWorld()
        registry = make_search_registry(world, ScriptedGroundingVLM())
        ctx = make_ctx(world)

        registry.call(
            "search_target",
            {"instruction": "red chair", "n_views": 4, "budget_turns": 1, "allow_move": False},
            ctx,
        )

        self.assertEqual(world.scan_calls[0]["n_views"], 4)

    def test_the_step_log_stays_bounded_and_json_plain(self):
        world = FakeWorld()
        vlm = ScriptedGroundingVLM()
        registry = make_search_registry(world, vlm)
        ctx = make_ctx(world)

        result = registry.call(
            "search_target",
            {"instruction": "red chair", "budget_turns": 3, "max_groundings": 8},
            ctx,
        )

        self.assertLessEqual(len(result.data["steps"]), skills_search.MAX_STEPS)
        json.dumps(result.to_dict())

    def test_the_routine_drives_the_real_navigation_tools(self):
        """The whole path, with ``tool_impl_navigation`` doing the flying."""
        from vlm_nav.tool_impl_navigation import register_navigation_tools

        world = MovableWorld(depth=6.0)
        vlm = ScriptedGroundingVLM()
        registry = ToolRegistry()
        register_perception_tools(registry, vlm=vlm)
        register_navigation_tools(registry)
        ctx = make_ctx(world)
        start = np.asarray(world.position_flu, dtype=np.float64).copy()

        result = registry.call(
            "search_target", {"instruction": "red chair", "budget_turns": 1}, ctx
        )

        self.assertEqual(result.status, "empty")
        steps = {step["step"]: step for step in result.data["steps"]}
        self.assertIn("propose_viewpoints", steps)
        self.assertIn("go_to_viewpoint", steps)
        move = steps["go_to_viewpoint"]["result"]
        self.assertIn(move["status"], {"ok", "no_progress", "blocked"})
        self.assertTrue(steps["go_to_viewpoint"]["result"]["viewpoint_id"].startswith("vp_"))
        # the drone really flew the leg (the navigation tools moved it)
        self.assertGreater(
            float(np.linalg.norm(np.asarray(world.position_flu, dtype=np.float64) - start)), 0.2
        )
        self.assertTrue(world.actions)
        # the scan after the proposal still reports the visual context
        self.assertTrue(result.data["coverage"]["checked_deg"])
        json.dumps(result.to_dict())


class SearchBudgetTest(unittest.TestCase):
    """One ``search_target`` call must cost the *episode* one decision.

    The routine is a decision: it sweeps, grounds, magnifies, proposes and flies
    on its own allowance, and the episode pays a fixed small number of budget
    units for the whole call. Its inner calls are counted and published in
    ``cost`` (``outer_tool_calls`` / ``inner_tool_calls`` / ``vlm_calls``), so the
    full price stays visible without the budget being emptied by one call.
    """

    def _ctx(self, world, budget):
        return ToolContext(world=world, episode_id="ep_budget",
                           budget={"tool_calls_left": int(budget)})

    def test_one_call_costs_the_episode_a_fixed_small_number_of_units(self):
        world = FakeWorld()
        vlm = ScriptedGroundingVLM()
        registry = make_search_registry(world, vlm)
        ctx = self._ctx(world, 20)

        result = registry.call("search_target", {"instruction": "red chair"}, ctx)

        spent = 20 - ctx.budget["tool_calls_left"]
        self.assertGreaterEqual(spent, 1)  # a decision is not free
        self.assertLessEqual(spent, 2)
        # ... while the routine really did run several steps
        self.assertGreaterEqual(len(result.data["steps"]), 3)

    def test_the_inner_and_outer_calls_are_reported_separately(self):
        world = FakeWorld()
        vlm = ScriptedGroundingVLM()
        registry = make_search_registry(world, vlm)
        ctx = self._ctx(world, 20)

        result = registry.call("search_target", {"instruction": "red chair"}, ctx)

        cost = result.data["cost"]
        self.assertEqual(cost["outer_tool_calls"], 1)
        self.assertEqual(cost["inner_tool_calls"], len(result.data["steps"]))
        self.assertEqual(result.data["inner_tool_calls"], cost["inner_tool_calls"])
        self.assertEqual(cost["vlm_calls"], result.data["vlm_calls"])
        self.assertGreater(cost["inner_tool_calls"], cost["outer_tool_calls"])
        # the whole price stays visible: the model calls the routine made
        self.assertEqual(result.data["vlm_calls"], len(vlm.calls))

    def test_the_inner_steps_stop_at_the_cap_and_return_what_they_have(self):
        world = FakeWorld()
        vlm = ScriptedGroundingVLM()
        registry = make_search_registry(world, vlm)
        ctx = self._ctx(world, 40)

        result = registry.call(
            "search_target",
            {"instruction": "red chair", "max_steps": 3, "allow_move": False},
            ctx,
        )

        self.assertEqual(result.status, "empty")
        self.assertEqual(len(result.data["steps"]), 3)  # partial result, not swallowed
        self.assertEqual(result.data["cost"]["inner_tool_calls"], 3)
        self.assertEqual(result.data["reason"], "inner_steps_exhausted")
        self.assertIn("next_action", result.data)
        self.assertGreaterEqual(ctx.budget["tool_calls_left"], 38)

    def test_a_nearly_empty_episode_budget_ends_the_call_at_once(self):
        world = FakeWorld()
        vlm = ScriptedGroundingVLM()
        registry = make_search_registry(world, vlm)
        ctx = self._ctx(world, 1)

        result = registry.call("search_target", {"instruction": "red chair"}, ctx)

        self.assertEqual(result.status, "empty")
        self.assertEqual(result.data["reason"], "budget_exhausted")
        self.assertEqual(result.data["steps"], [])
        self.assertEqual(result.data["cost"]["inner_tool_calls"], 0)
        self.assertEqual(ctx.budget["tool_calls_left"], 0)  # the call itself was paid

    def test_a_low_budget_clips_the_inner_allowance(self):
        world = FakeWorld()
        vlm = ScriptedGroundingVLM()
        registry = make_search_registry(world, vlm)
        ctx = self._ctx(world, 3)

        result = registry.call("search_target", {"instruction": "red chair"}, ctx)

        # only the two units the episode can still pay for inside the call
        self.assertEqual(len(result.data["steps"]), 2)
        self.assertEqual(result.data["cost"]["inner_tool_calls"], 2)
        self.assertEqual(ctx.budget["tool_calls_left"], 2)

    def test_the_defaults_bound_one_call(self):
        # the defaults are part of the contract: one call is a sweep, a few
        # groundings and one leg - not an unbounded sub-episode
        self.assertEqual(skills_search.DEFAULT_INNER_STEPS, 8)
        self.assertEqual(skills_search.DEFAULT_SEARCH_TURNS, 1)
        self.assertEqual(skills_search.DEFAULT_GROUNDINGS, 3)
        self.assertEqual(skills_search.DEFAULT_INSPECTS, 1)
        self.assertLessEqual(skills_search.MAX_INNER_STEPS, skills_search.MAX_STEPS)
        self.assertLessEqual(skills_search.MAX_GROUNDINGS, 8)
        self.assertLessEqual(skills_search.MAX_SEARCH_TURNS, 4)
        # the schema documents the cap and carries the tightened defaults
        schema = skills_search.SEARCH_TARGET_SCHEMA["properties"]
        self.assertIn("max_steps", schema)
        self.assertIn(str(skills_search.DEFAULT_INNER_STEPS),
                      schema["max_steps"]["description"])
        self.assertIn(str(skills_search.DEFAULT_GROUNDINGS),
                      schema["max_groundings"]["description"])
        self.assertIn(str(skills_search.DEFAULT_SEARCH_TURNS),
                      schema["budget_turns"]["description"])

    def test_a_second_round_costs_more_inner_steps_than_the_default_cap(self):
        """A second sweep-and-fly round is opt-in: it needs a raised ``max_steps``."""
        world = FakeWorld()
        vlm = ScriptedGroundingVLM()
        moved = []
        registry = make_search_registry(world, vlm, moved=moved)
        ctx = self._ctx(world, 40)

        capped = registry.call("search_target", {"instruction": "red chair"}, ctx)
        # the default call flies one leg and stops at its own step cap
        self.assertLessEqual(capped.data["cost"]["inner_tool_calls"],
                             skills_search.DEFAULT_INNER_STEPS)
        self.assertEqual(len(moved), 1)

        world = FakeWorld()
        moved = []
        registry = make_search_registry(world, vlm, moved=moved)
        ctx = self._ctx(world, 40)
        full = registry.call(
            "search_target",
            {"instruction": "red chair", "budget_turns": 2, "max_steps": 16},
            ctx,
        )
        self.assertEqual(full.data["cost"]["turns"], 2)
        self.assertGreater(len(moved), 1)
        self.assertGreater(full.data["cost"]["inner_tool_calls"],
                           capped.data["cost"]["inner_tool_calls"])
        self.assertEqual(40 - ctx.budget["tool_calls_left"], 1)


RELATION_INSTRUCTION = "the red chair next to the lamp"


class SearchCompareViewsTest(unittest.TestCase):
    """Sweep-and-pick: a relation instruction is settled by comparing the views.

    The sweep returns several views of one pose. The relation the instruction
    names ("next to the lamp") is only readable in a view that holds *both*
    objects, and the frame a plain grounding call uses may hold only one of
    them - so the routine compares every swept view in **one** model call before
    it grounds any of them, grounds the views that comparison selected first,
    and falls back to the swept order whenever the comparison is unusable.
    """

    def test_a_relation_instruction_compares_every_view_before_grounding(self):
        world = FakeWorld()
        vlm = CompareAwareVLM(picks=(2,), default=CANDIDATES)
        registry = make_search_registry(world, vlm)
        ctx = make_ctx(world)

        result = registry.call("search_target", {"instruction": RELATION_INSTRUCTION}, ctx)

        self.assertEqual(result.status, "ok")
        steps = [step["step"] for step in result.data["steps"]]
        self.assertEqual(steps[:2], ["observe", "compare_views"])
        self.assertEqual(len(vlm.compare_calls), 1)  # one call, every view at once
        call = vlm.compare_calls[0]
        self.assertEqual(call["n_frames"], len(call["observation_ids"]))
        self.assertGreaterEqual(call["n_frames"], 2)
        self.assertIn(RELATION_INSTRUCTION, call["user_prompt"])
        # the grounding came after the comparison and used the view it picked
        picked_id = call["observation_ids"][1]
        grounded = [step for step in result.data["steps"] if step["step"] == "ground_target"]
        self.assertTrue(grounded)
        self.assertEqual(grounded[0]["obs_ids"], [picked_id])
        self.assertEqual(result.data["candidate_id"], result.data["candidates"][0]["id"])
        # per-view verdicts travel with the step (what the comparison saw, and why)
        compare_step = next(step for step in result.data["steps"] if step["step"] == "compare_views")
        self.assertTrue(compare_step["why"])
        self.assertEqual(compare_step["obs_ids"], call["observation_ids"])
        self.assertEqual(compare_step["result"]["picked"], [picked_id])
        self.assertTrue(compare_step["result"]["views"])
        self.assertTrue(compare_step["result"]["views"][0]["reason"])
        json.dumps(result.to_dict())

    def test_the_first_grounded_view_is_the_one_the_comparison_picked(self):
        world = FakeWorld()
        vlm = CompareAwareVLM(picks=(4,), default=CANDIDATES)
        registry = make_search_registry(world, vlm)
        ctx = make_ctx(world)

        result = registry.call("search_target", {"instruction": RELATION_INSTRUCTION}, ctx)

        picked_id = vlm.compare_calls[0]["observation_ids"][3]
        self.assertEqual(vlm.calls[0]["observation_id"], picked_id)
        grounded = [step for step in result.data["steps"] if step["step"] == "ground_target"]
        self.assertEqual(grounded[0]["obs_ids"], [picked_id])
        self.assertEqual(result.data["vlm_calls"], 1 + 1)  # the comparison + the grounding

    def test_a_magnified_look_of_the_picked_view_uses_the_region_it_named(self):
        world = FakeWorld()

        class CropAware(CompareAwareVLM):
            """The swept views answer empty; any other frame (the crop) holds it."""

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.swept = set()

            def decide(self, frames, system_prompt, user_prompt, context=None):
                context = context or {}
                if str(context.get("step") or "") == "compare_views":
                    self.swept.update(str(item) for item in (context.get("observation_ids") or []))
                    return super().decide(frames, system_prompt, user_prompt, context)
                obs_id = str(context.get("observation_id") or "")
                if obs_id in self.swept:
                    self.calls.append({"observation_id": obs_id, "context": context})
                    return '{"candidates": []}'
                return super().decide(frames, system_prompt, user_prompt, context)

        vlm = CropAware(picks=(2,), default=CANDIDATES)
        registry = make_search_registry(world, vlm)
        ctx = make_ctx(world)

        result = registry.call("search_target", {"instruction": RELATION_INSTRUCTION}, ctx)

        self.assertEqual(result.status, "ok")  # the magnified crop of the region held it
        picked_id = vlm.compare_calls[0]["observation_ids"][1]
        crop_step = next(step for step in result.data["steps"] if step["step"] == "inspect_region")
        self.assertEqual(crop_step["obs_ids"][0], picked_id)
        self.assertEqual(crop_step["result"]["region"], [300, 300, 520, 520])
        grounded = [step for step in result.data["steps"] if step["step"] == "ground_target"]
        self.assertEqual(grounded[0]["obs_ids"], [picked_id])  # the picked view came first

    def test_a_plain_instruction_keeps_the_fast_path(self):
        world = FakeWorld()
        vlm = CompareAwareVLM(picks=(2,), default=CANDIDATES)
        registry = make_search_registry(world, vlm)
        ctx = make_ctx(world)

        result = registry.call("search_target", {"instruction": "red chair"}, ctx)

        self.assertEqual(result.status, "ok")
        self.assertEqual(vlm.compare_calls, [])  # no relation named: no extra model call
        self.assertEqual([step["step"] for step in result.data["steps"]][:2],
                         ["observe", "ground_target"])
        self.assertEqual(result.data["vlm_calls"], 1)
        self.assertEqual(result.data["cost"]["model_only_steps"], 0)

    def test_the_comparison_can_be_forced_on_and_off(self):
        world = FakeWorld()
        vlm = CompareAwareVLM(picks=(2,), default=CANDIDATES)
        registry = make_search_registry(world, vlm)
        ctx = make_ctx(world)

        forced = registry.call(
            "search_target", {"instruction": "red chair", "compare_views": "on"}, ctx
        )
        self.assertTrue(vlm.compare_calls)
        self.assertEqual(forced.data["cost"]["model_only_steps"], 1)

        world = FakeWorld()
        vlm = CompareAwareVLM(picks=(2,), default=CANDIDATES)
        registry = make_search_registry(world, vlm)
        ctx = make_ctx(world)
        off = registry.call(
            "search_target",
            {"instruction": RELATION_INSTRUCTION, "compare_views": "off"},
            ctx,
        )
        self.assertEqual(vlm.compare_calls, [])
        self.assertEqual(off.data["cost"]["model_only_steps"], 0)
        self.assertEqual(off.status, "ok")

    def test_a_failed_comparison_falls_back_to_the_swept_order(self):
        world = FakeWorld()
        vlm = CompareAwareVLM(
            picks=(4,), default=CANDIDATES, compare_error=RuntimeError("model down")
        )
        registry = make_search_registry(world, vlm)
        ctx = make_ctx(world)

        result = registry.call("search_target", {"instruction": RELATION_INSTRUCTION}, ctx)

        # the comparison failed, the search did not: the swept order still runs
        self.assertEqual(result.status, "ok")
        compare_step = [step for step in result.data["steps"] if step["step"] == "compare_views"]
        self.assertTrue(compare_step)
        self.assertIn("error", compare_step[0]["result"])
        self.assertNotEqual(vlm.calls[0]["observation_id"], None)
        self.assertEqual(result.data["vlm_calls"], len(vlm.calls) + len(vlm.compare_calls))

    def test_the_comparison_is_published_in_the_steps_and_the_cost(self):
        world = FakeWorld()
        vlm = CompareAwareVLM(picks=(2,), default=CANDIDATES)
        registry = make_search_registry(world, vlm)
        ctx = make_ctx(world, **{"perception.vlm": vlm})

        result = registry.call(
            "search_target", {"instruction": RELATION_INSTRUCTION, "max_groundings": 1}, ctx
        )

        cost = result.data["cost"]
        self.assertEqual(cost["outer_tool_calls"], 1)  # the episode paid one decision
        self.assertEqual(cost["model_only_steps"], 1)  # the comparison is one step of it
        self.assertEqual(cost["vlm_calls"], result.data["vlm_calls"])
        self.assertEqual(result.data["vlm_calls"], 1 + len(vlm.calls))
        tool_steps = [step for step in result.data["steps"] if step["step"] != "compare_views"]
        self.assertEqual(cost["inner_tool_calls"], len(tool_steps))
        self.assertEqual(len(result.data["steps"]), cost["inner_tool_calls"] + 1)
        self.assertEqual(result.data["comparison"]["picked"],
                         [vlm.compare_calls[0]["observation_ids"][1]])
        json.dumps(result.to_dict())

    def test_the_comparison_is_a_model_call_on_the_backend_the_tools_use(self):
        """The M2 runner wraps the backend in a counting proxy: it must see it.

        ``ground_target`` grounds with the backend handed to the registry
        (``ctx.extra["perception.vlm"]``, the runner's counting proxy); the
        comparison is the same kind of call and has to be visible to the same
        accountant, otherwise ``vlm_calls``/``vlm_cost`` would miss it.
        """

        class CountingProxy:
            def __init__(self, vlm):
                self._vlm = vlm
                self.model_calls = []

            def __getattr__(self, name):
                return getattr(self._vlm, name)

            def decide(self, *args, **kwargs):
                reply = self._vlm.decide(*args, **kwargs)
                frames = args[0] if args else kwargs.get("frames", kwargs.get("frame_chw"))
                self.model_calls.append(
                    {
                        "frames": len(frames) if isinstance(frames, (list, tuple)) else 1,
                        "context": dict(kwargs.get("context") or {}),
                    }
                )
                return reply

        world = FakeWorld()
        inner = CompareAwareVLM(picks=(2,), default=CANDIDATES)
        proxy = CountingProxy(inner)
        registry = make_search_registry(world, proxy)
        ctx = make_ctx(world)

        result = registry.call("search_target", {"instruction": RELATION_INSTRUCTION}, ctx)

        self.assertEqual(len(proxy.model_calls), 1 + len(inner.calls))
        self.assertEqual(proxy.model_calls[0]["context"].get("step"), "compare_views")
        self.assertEqual(proxy.model_calls[0]["frames"], len(inner.compare_calls[0]["observation_ids"]))
        self.assertEqual(result.data["vlm_calls"], len(proxy.model_calls))

    def test_the_comparison_is_bounded_by_the_requested_view_count(self):
        world = FakeWorld()
        vlm = CompareAwareVLM(picks=(2,), default=CANDIDATES)
        registry = make_search_registry(world, vlm)
        ctx = make_ctx(world)

        registry.call(
            "search_target",
            {"instruction": RELATION_INSTRUCTION, "compare_views_max": 3},
            ctx,
        )

        self.assertEqual(vlm.compare_calls[0]["n_frames"], 3)

    def test_recalled_frames_are_compared_without_a_new_sweep(self):
        world = FakeWorld()
        vlm = CompareAwareVLM(picks=(1,), default=CANDIDATES)
        registry = make_search_registry(world, vlm)
        ctx = make_ctx(world)
        scanned = registry.call("observe", {"mode": "lookaround", "n_views": 4}, ctx)
        recalled = [str(entry["obs_id"]) for entry in scanned.data["view_index"]][2:]
        sweeps_before = len(world.scan_calls)

        result = registry.call(
            "search_target",
            {"instruction": RELATION_INSTRUCTION, "observation_ids": recalled},
            ctx,
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(len(world.scan_calls), sweeps_before)  # the frames were already on record
        self.assertEqual(vlm.compare_calls[0]["observation_ids"], recalled)
        self.assertEqual(vlm.calls[0]["observation_id"], recalled[0])
        self.assertEqual([step["step"] for step in result.data["steps"]][0], "compare_views")
        # the frames the instruction is settled in are the ones that were compared
        self.assertEqual(result.data["comparison"]["picked"], [recalled[0]])
        json.dumps(result.to_dict())

    def test_a_reply_cut_off_by_the_token_budget_still_names_its_views(self):
        """A truncated comparison is not a lost comparison.

        The generation budget of a call is small; a listing of eight views can be
        cut mid-object. The complete entries that came back are evidence and are
        used (the routine orders them first), and the truncation is recorded.
        """
        truncated = (
            '{"views": [{"view": 1, "target": false}, '
            '{"view": 2, "target": true, "confidence": 0.9, '
            '"reason": "red object beside the lamp", "box": [300, 300, 520, 520]}, '
            '{"view": 3, "target": fa'
        )
        world = FakeWorld()
        vlm = CompareAwareVLM(picks=(2,), compare_reply=truncated, default=CANDIDATES)
        registry = make_search_registry(world, vlm)
        ctx = make_ctx(world)

        result = registry.call("search_target", {"instruction": RELATION_INSTRUCTION}, ctx)

        self.assertEqual(result.status, "ok")
        compare_step = next(step for step in result.data["steps"] if step["step"] == "compare_views")
        self.assertTrue(compare_step["result"]["truncated"])
        self.assertEqual(compare_step["result"]["n_views"], 2)
        picked_id = vlm.compare_calls[0]["observation_ids"][1]
        self.assertEqual(result.data["views_picked"], [picked_id])
        grounded = [step for step in result.data["steps"] if step["step"] == "ground_target"]
        self.assertEqual(grounded[0]["obs_ids"], [picked_id])
        json.dumps(result.to_dict())

    def test_the_picked_view_is_faced_before_it_is_grounded(self):
        """The depth map belongs to the frame at the drone's own heading.

        The relation the instruction names is off-axis by construction, and an
        off-axis frame cannot be ranged (``range_status: unknown``: a bearing
        only, so the drone can never arrive). The routine therefore turns to the
        picked view first (``return_to_observation`` restores the heading that
        frame was shot at) and grounds the frame that the *capture at that pose*
        returns - which is the frame the depth belongs to.
        """
        from vlm_nav.tool_impl_navigation import register_navigation_tools

        world = MovableWorld(depth=6.0)
        vlm = CompareAwareVLM(picks=(2,), default=CANDIDATES)
        registry = ToolRegistry()
        register_perception_tools(registry, vlm=vlm)
        register_navigation_tools(registry)
        ctx = make_ctx(world)

        result = registry.call("search_target", {"instruction": RELATION_INSTRUCTION}, ctx)

        self.assertEqual(result.status, "ok")
        steps = [step["step"] for step in result.data["steps"]]
        self.assertIn("face_view", steps)
        picked_id = vlm.compare_calls[0]["observation_ids"][1]
        face_step = next(step for step in result.data["steps"] if step["step"] == "face_view")
        self.assertEqual(face_step["obs_ids"][0], picked_id)
        grounded_id = face_step["result"]["grounded_frame"]
        self.assertNotEqual(grounded_id, picked_id)  # a fresh frame at the drone's heading
        self.assertEqual(vlm.calls[0]["observation_id"], grounded_id)
        # the drone really turned to the view it picked (yaw 0 -> 45 deg)
        self.assertAlmostEqual(math.degrees(float(world.state.yaw_rad)), 45.0, places=3)
        json.dumps(result.to_dict())

    def test_a_picked_view_that_already_faces_the_drone_is_grounded_as_it_is(self):
        from vlm_nav.tool_impl_navigation import register_navigation_tools

        world = MovableWorld(depth=6.0)
        vlm = CompareAwareVLM(picks=(1,), default=CANDIDATES)  # the "ahead" view
        registry = ToolRegistry()
        register_perception_tools(registry, vlm=vlm)
        register_navigation_tools(registry)
        ctx = make_ctx(world)
        start_yaw = float(world.state.yaw_rad)

        result = registry.call("search_target", {"instruction": RELATION_INSTRUCTION}, ctx)

        self.assertEqual(result.status, "ok")
        self.assertNotIn("face_view", [step["step"] for step in result.data["steps"]])
        picked_id = vlm.compare_calls[0]["observation_ids"][0]
        self.assertEqual(vlm.calls[0]["observation_id"], picked_id)
        self.assertEqual(float(world.state.yaw_rad), start_yaw)  # nothing turned

    def test_facing_can_be_switched_off(self):
        world = FakeWorld()
        vlm = CompareAwareVLM(picks=(2,), default=CANDIDATES)
        registry = make_search_registry(world, vlm)
        ctx = make_ctx(world)

        result = registry.call(
            "search_target",
            {"instruction": RELATION_INSTRUCTION, "face_picked": False},
            ctx,
        )

        self.assertEqual(result.status, "ok")
        self.assertNotIn("face_view", [step["step"] for step in result.data["steps"]])
        picked_id = vlm.compare_calls[0]["observation_ids"][1]
        self.assertEqual(vlm.calls[0]["observation_id"], picked_id)

    def test_a_turn_that_does_not_work_falls_back_to_the_picked_frame(self):
        # no movement tool in this registry: the turn cannot happen, and the
        # picked frame is grounded as it is (the routine never loses the search)
        world = FakeWorld()
        vlm = CompareAwareVLM(picks=(2,), default=CANDIDATES)
        registry = make_search_registry(world, vlm)
        ctx = make_ctx(world)

        result = registry.call("search_target", {"instruction": RELATION_INSTRUCTION}, ctx)

        self.assertEqual(result.status, "ok")
        face_step = next(step for step in result.data["steps"] if step["step"] == "face_view")
        self.assertTrue(face_step["result"].get("error") or face_step["result"]["status"] != "ok")
        picked_id = vlm.compare_calls[0]["observation_ids"][1]
        self.assertEqual(vlm.calls[0]["observation_id"], picked_id)

    def test_a_recalled_frame_of_unknown_direction_is_still_comparable(self):
        world = FakeWorld()
        vlm = CompareAwareVLM(picks=(1,), default=CANDIDATES)
        registry = make_search_registry(world, vlm)
        ctx = make_ctx(world)
        scanned = registry.call("observe", {"mode": "lookaround", "n_views": 4}, ctx)
        recalled = [str(entry["obs_id"]) for entry in scanned.data["view_index"]]

        result = registry.call(
            "search_target",
            {"instruction": RELATION_INSTRUCTION, "observation_ids": recalled[:1]},
            ctx,
        )

        # one frame is nothing to compare: the routine does not pretend it compared
        self.assertEqual(vlm.compare_calls, [])
        self.assertEqual(result.data["cost"]["model_only_steps"], 0)
        self.assertTrue(any(step["step"] == "ground_target" for step in result.data["steps"]))

    def test_relation_phrases_are_recognized(self):
        for text in (
            "the red object next to the lamp",
            "fly to the box near the door",
            "the backpack closest to the bench",
            "pick the chair on the left of the table",
            "the crate between the two walls",
            "the marker beside the cone",
        ):
            self.assertTrue(skills_search.relation_phrase(text), text)
        for text in ("the red chair", "find the blue backpack", "go to the target", ""):
            self.assertEqual(skills_search.relation_phrase(text), "", text)

    def test_the_routine_documents_the_comparison(self):
        schema = skills_search.SEARCH_TARGET_SCHEMA["properties"]
        self.assertIn("compare_views", schema)
        self.assertIn("observation_ids", schema)
        self.assertIn("face_picked", schema)
        self.assertIn("compare", skills_search.SEARCH_TARGET_DESCRIPTION.lower())
        self.assertIn("compare", (skills_search.search_target.__doc__ or "").lower())
        self.assertLessEqual(skills_search.MAX_COMPARE_VIEWS, 8)
        self.assertGreaterEqual(skills_search.DEFAULT_COMPARE_VIEWS, 2)


class GroundingConfirmationTest(unittest.TestCase):
    """search_target may only call a candidate confirmed when its range is sane."""

    def test_a_distance_uncertain_candidate_is_not_reported_confirmed(self):
        candidate = {"confidence": 0.9, "distance_uncertain": True}
        self.assertFalse(
            skills_search._grounding_confirmed(candidate, 0.9, 0.5)
        )

    def test_a_conflicted_distance_source_is_not_reported_confirmed(self):
        candidate = {"confidence": 0.9, "distance_source": "conflict"}
        self.assertFalse(
            skills_search._grounding_confirmed(candidate, 0.9, 0.5)
        )

    def test_a_sane_high_confidence_candidate_is_confirmed(self):
        candidate = {"confidence": 0.9, "distance_source": "depth_consistent"}
        self.assertTrue(
            skills_search._grounding_confirmed(candidate, 0.9, 0.5)
        )

    def test_the_candidate_summary_carries_the_distance_fields(self):
        summary = skills_search._candidate_summary(
            {"id": "cand_1", "distance_source": "conflict",
             "distance_uncertain": True, "distance_conflict": {"resolved": False}}
        )
        self.assertEqual(summary["distance_source"], "conflict")
        self.assertTrue(summary["distance_uncertain"])
        self.assertEqual(summary["distance_conflict"], {"resolved": False})


if __name__ == "__main__":
    unittest.main()
