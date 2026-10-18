"""Tests for the navigation tool set (vlm_nav.tool_impl_navigation).

CPU-only: a fake world mirrors the pose/action conventions of
``SplatWorldCPU`` / ``EnvWorld`` (yaw applied before translation, camera
conventions of ``actions.py``) and exposes the ``getattr``-optional sensing
hooks of the real backends.

The movement contract is one tool per *intent* - ``approach_candidate`` (the
object in a frame), ``go_to_viewpoint`` (an observation position),
``return_to_observation`` (the camera pose a frame was shot from) - and the
deprecated ``navigate`` answers with an explicit error plus the replacement
call. Every movement tool reports the same progress schema
(``status`` ok|blocked|no_progress, ``outcome``, ``progress_m``, ``turn_deg``,
``remaining_m``, ``blocked_reason``, ``task_id``, the intent id,
``last_observation_id``): a call that moves nothing is ``no_progress`` /
``already_at_pose`` and never ``ok`` + ``arrived``, and every hint names a
*different* executable next step (the zero-displacement loop this contract
removes). Besides the intents this covers ``propose_viewpoints`` /
``check_path`` / ``verify_goal`` and the M3 continuous task
(``task_status`` / ``cancel_task`` / events / budgets).
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from vlm_nav.actions import (
    NavAction,
    action_to_body_delta,
    body_delta_to_world,
)
from vlm_nav import skills_approach
from vlm_nav import tool_impl_navigation as navigation_module
from vlm_nav.observations import (
    CANDIDATE_STORE_KEY,
    KIND_EVIDENCE,
    observation_store,
    register_observation,
)
from vlm_nav.tool_impl_navigation import register_navigation_tools
from vlm_nav.tools import ToolContext, ToolRegistry

TOOL_NAMES = [
    "approach_candidate",
    # the read-only batch verb (P4: several checks in one turn) and the
    # evidence-backed completion claim are part of this module's contract
    "batch_read",
    "cancel_task",
    "check_path",
    "finish",
    "go_to_viewpoint",
    "navigate",
    "propose_viewpoints",
    "return_to_observation",
    "task_status",
    "verify_goal",
]
MOVE_TOOLS = ("approach_candidate", "go_to_viewpoint", "return_to_observation")


class FakeTarget:
    def __init__(self, name="blob", position=(3.0, 0.0, 0.0), radius_m=0.5):
        self.name = name
        self.position_flu = tuple(position)
        self.radius_m = radius_m


class FakeWorld:
    """Kinematic fake world with the real worlds' action conventions."""

    def __init__(
        self,
        position=(0.0, 0.0, 0.0),
        yaw_deg=0.0,
        points=None,
        target=(3.0, 0.0, 0.0),
        radius_m=0.5,
        depth=None,
        in_fov=True,
        have_fov=True,
        have_depth_fn=True,
        have_bev=True,
        have_trajectory=True,
        bound_radius=None,
        outcome="",
        move=True,
        intrinsics=False,
        width=320,
        height=180,
        hfov_deg=90.0,
    ):
        self.position = np.asarray(position, dtype=np.float64)
        self.yaw_rad = math.radians(yaw_deg)
        self.target = FakeTarget(position=target, radius_m=radius_m)
        self._sense_points = (
            np.asarray(points, dtype=np.float64) if points is not None else None
        )
        self.trajectory = [tuple(self.position)] if have_trajectory else None
        self.depth = depth
        self.in_fov = in_fov
        self.have_fov = have_fov
        self.bound_radius = bound_radius
        self.outcome = outcome
        self.move = move
        self.applied: list[NavAction] = []
        if intrinsics:
            # a camera that declares its frame: the intrinsics a real world
            # exposes (the projection of a target needs a field of view)
            self.width, self.height, self.hfov_deg = int(width), int(height), float(hfov_deg)
        if not have_fov:
            self.target_in_fov = None  # instance attribute shadows the method
        if not have_depth_fn:
            self.depth_at_pixel = None
        if not have_bev:
            self.bev_image = None

    @property
    def position_flu(self):
        return self.position

    def distance_to_target(self):
        return float(np.linalg.norm(self.position - np.asarray(self.target.position_flu)))

    def target_in_fov(self):
        return bool(self.in_fov)

    def depth_at_pixel(self, pixel):
        return self.depth

    def bev_image(self, target_estimate=None, size=192):
        return np.zeros((3, size, size), dtype=np.float32)

    def render(self):
        return np.zeros((3, 4, 4), dtype=np.float32)

    def apply_action(self, action: NavAction) -> str:
        self.applied.append(action)
        if action.is_terminal or action.kind == "hover":
            return action.kind
        dx, dy, dz, dyaw = action_to_body_delta(action)
        if not self.move:
            return self._feedback()
        self.yaw_rad = math.atan2(
            math.sin(self.yaw_rad + dyaw), math.cos(self.yaw_rad + dyaw)
        )
        wx, wy, wz = body_delta_to_world(dx, dy, dz, self.yaw_rad)
        self.position = self.position + np.asarray([wx, wy, wz])
        if self.trajectory is not None:
            self.trajectory.append(tuple(self.position))
        return self._feedback()

    def _feedback(self) -> str:
        p = self.position
        text = f"moved to ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"
        if self.outcome:
            text += f" [{self.outcome}]"
        return text


class DepthWorld(FakeWorld):
    """Fake world whose depth sensor measures the axial range to a wall.

    ``surface_m`` is where the wall (the surface the target sits on) really is,
    so a candidate grounded at a wrong range is refined by the *fresh* depth
    read the approach skill takes after a leg - which is what re-localization
    has to prove.
    """

    def __init__(self, surface_m=4.0, **kwargs):
        kwargs.setdefault("intrinsics", True)
        kwargs.setdefault("depth", 4.0)
        super().__init__(**kwargs)
        self.surface_m = float(surface_m)

    def depth_at_pixel(self, pixel):
        # plane x = surface_m seen from a pose whose heading is +x
        yaw = self.yaw_rad
        forward_range = (self.surface_m - float(self.position[0])) / max(
            math.cos(yaw), 1e-6
        )
        return None if forward_range <= 0.05 else float(forward_range)


def make_context(world, memory=None, calls=40, extra=None):
    return ToolContext(
        world=world, memory=memory, budget={"tool_calls_left": calls}, extra=dict(extra or {})
    )


def registry():
    reg = ToolRegistry()
    register_navigation_tools(reg)
    return reg


def bearing_of(world, position):
    pos = np.asarray(world.position_flu, dtype=np.float64)
    delta = np.asarray(position, dtype=np.float64) - pos
    return math.degrees(math.atan2(delta[1], delta[0]))


def add_viewpoint(ctx, position, viewpoint_id="vp_1", reachable=True):
    """Register a viewpoint the way ``propose_viewpoints`` stores one."""
    entry = {
        "viewpoint_id": viewpoint_id,
        "position": [round(float(v), 3) for v in position],
        "distance_m": round(float(np.linalg.norm(np.asarray(position, dtype=np.float64))), 2),
        "reachable": bool(reachable),
        "cost_m": round(float(np.linalg.norm(np.asarray(position, dtype=np.float64))), 2),
    }
    by_id = ctx.extra.setdefault("viewpoints_by_id", {})
    by_id[viewpoint_id] = entry
    ctx.extra["viewpoints"] = [entry]
    return viewpoint_id


def store_candidate(
    ctx,
    candidate_id="cand_1",
    *,
    position=None,
    box=(450.0, 450.0, 550.0, 550.0),
    observation_id="",
    ray=None,
    source="depth",
    range_m=None,
):
    """Put a candidate record in the store ``ground_target`` writes to."""
    record = {
        "id": candidate_id,
        "source": "ground_target",
        "box": [float(v) for v in box],
        "box_format": "xyxy",
        "observation_id": observation_id,
    }
    if position is not None:
        record["position_flu"] = [float(v) for v in position]
        record["position_source"] = source
        record["range_status"] = "known"
    if range_m is not None:
        record["range_m"] = float(range_m)
    if ray is not None:
        record["ray"] = dict(ray)
        record.setdefault("position_source", "ray")
    ctx.extra.setdefault(CANDIDATE_STORE_KEY, {})[candidate_id] = record
    return candidate_id


def seen_at(ctx, position, yaw_deg=0.0, kind=KIND_EVIDENCE):
    """Register one observation at ``position`` (the drone "flew on" after)."""
    world = ctx.world
    pos0, yaw0 = world.position, world.yaw_rad
    world.position = np.asarray(position, dtype=np.float64)
    world.yaw_rad = math.radians(yaw_deg)
    record = register_observation(ctx, None, kind=kind)
    world.position, world.yaw_rad = pos0, yaw0
    return record.obs_id


class RegistrationTest(unittest.TestCase):
    def test_registers_the_navigation_tools(self):
        reg = registry()
        self.assertEqual(reg.names(), TOOL_NAMES)
        text = reg.specs_text()
        for name in TOOL_NAMES:
            self.assertIn(name, text)

    def test_probe_aliases_point_at_the_same_entry_point(self):
        # the driver script / MCP server look for these names
        import vlm_nav.tool_impl_navigation as module

        self.assertIs(module.register_tools, register_navigation_tools)
        self.assertIs(module.register_all, register_navigation_tools)

    def test_side_effects_flags(self):
        reg = registry()
        # the three movement tools change the pose and are serialized
        for name in MOVE_TOOLS:
            self.assertTrue(reg.get(name).side_effects, name)
        self.assertTrue(reg.get("cancel_task").side_effects)
        # the deprecated navigate executes nothing at all
        self.assertFalse(reg.get("navigate").side_effects)
        for name in ("propose_viewpoints", "check_path", "task_status", "verify_goal"):
            self.assertFalse(reg.get(name).side_effects, name)

    def test_movement_tools_declare_their_own_id_argument(self):
        reg = registry()
        schema = reg.get("approach_candidate").input_schema
        self.assertEqual(schema["required"], ["candidate_id"])
        self.assertEqual(
            registry().get("go_to_viewpoint").input_schema["required"], ["viewpoint_id"]
        )
        self.assertEqual(
            registry().get("return_to_observation").input_schema["required"],
            ["observation_id"],
        )

    def test_read_only_tools_are_not_served_from_the_args_keyed_cache(self):
        # their answer depends on the live pose/map/task state, which an
        # args-keyed result cache cannot see: a repeat must recompute
        reg = registry()
        for name in ("propose_viewpoints", "check_path", "task_status", "verify_goal"):
            self.assertFalse(reg.get(name).cacheable, name)

    def test_registry_call_enforces_budget_and_logs(self):
        reg = registry()
        ctx = make_context(FakeWorld())
        result = reg.call("check_path", {"position": [3.0, 0.0, 0.0]}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertEqual(ctx.budget["tool_calls_left"], 39)
        self.assertEqual(len(ctx.log), 1)
        self.assertEqual(ctx.log[0]["tool"], "check_path")
        ctx.budget["tool_calls_left"] = 0
        drained = reg.call("check_path", {"position": [3.0, 0.0, 0.0]}, ctx)
        self.assertEqual(drained.status, "failed")
        self.assertEqual(drained.data["reason"], "budget_exhausted")


class ProposeViewpointsTest(unittest.TestCase):
    def test_structured_viewpoints_are_registered(self):
        ctx = make_context(FakeWorld())
        result = registry().call("propose_viewpoints", {}, ctx)
        self.assertEqual(result.status, "ok")
        data = result.data
        self.assertEqual(data["count"], 5)
        self.assertEqual(data["reachable_count"], 5)
        self.assertEqual(len(data["viewpoints"]), 5)
        ids = [item["viewpoint_id"] for item in data["viewpoints"]]
        self.assertEqual(sorted(ids), [f"vp_{i}" for i in range(1, 6)])
        for item in data["viewpoints"]:
            self.assertEqual(len(item["position"]), 3)
            self.assertIsInstance(item["position"][0], float)
            self.assertGreaterEqual(item["expected_coverage"], 0.0)
            self.assertLessEqual(item["expected_coverage"], 1.0)
            self.assertGreaterEqual(item["unknown_ratio"], 0.0)
            self.assertLessEqual(item["unknown_ratio"], 1.0)
            self.assertGreater(item["cost_m"], 0.0)
            self.assertTrue(item["reachable"])
            self.assertIsNone(item["min_clearance_m"])  # no observed geometry yet
        # stored for later resolution
        self.assertEqual(len(ctx.extra["viewpoints"]), 5)
        self.assertEqual(len(ctx.extra["viewpoints_by_id"]), 5)
        # map/trajectory/image evidence references
        kinds = {e.kind for e in result.evidence}
        self.assertIn("trajectory", kinds)
        self.assertIn("image", kinds)
        json.dumps(result.to_dict())

    def test_direction_deg_biases_the_arc(self):
        ctx = make_context(FakeWorld())
        result = registry().call("propose_viewpoints", {"direction_deg": 90.0}, ctx)
        self.assertEqual(result.data["base_bearing_deg"], 90.0)
        for item in result.data["viewpoints"]:
            self.assertGreaterEqual(item["bearing_deg"], 45.0)
            self.assertLessEqual(item["bearing_deg"], 135.0)

    def test_target_hypothesis_position_and_count(self):
        ctx = make_context(FakeWorld())
        result = registry().call(
            "propose_viewpoints", {"target_hypothesis": [0.0, 5.0, 0.0], "count": 1}, ctx
        )
        self.assertEqual(result.data["count"], 1)
        self.assertTrue(result.data["hypothesis_resolved"])
        self.assertEqual(result.data["base_bearing_deg"], 90.0)
        self.assertAlmostEqual(result.data["viewpoints"][0]["distance_m"], 2.0, places=3)

    def test_target_hypothesis_resolves_memory_candidate(self):
        ctx = make_context(
            FakeWorld(), extra={"candidate_positions": {"cand_1": [0.0, 5.0, 0.0]}}
        )
        result = registry().call(
            "propose_viewpoints", {"target_hypothesis": "cand_1"}, ctx
        )
        self.assertTrue(result.data["hypothesis_resolved"])
        self.assertEqual(result.data["base_bearing_deg"], 90.0)

    def test_unresolved_hypothesis_degrades_to_current_heading(self):
        ctx = make_context(FakeWorld(yaw_deg=10.0))
        result = registry().call(
            "propose_viewpoints", {"target_hypothesis": "ghost"}, ctx
        )
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.data["hypothesis_resolved"])
        self.assertIn("ghost", result.data["hypothesis_note"])
        for item in result.data["viewpoints"]:
            self.assertLessEqual(abs(item["bearing_deg"] - 10.0), 45.0)

    def test_the_goal_name_is_not_a_position_lookup(self):
        """``target_hypothesis`` resolves from evidence, never from the goal.

        Naming the world target used to return its *true* position - a
        hypothesis that reads like the instruction came back ``resolved`` with
        the arc aimed at the goal the episode is supposed to find (and the
        module's own rule is that ``verify_goal`` is the only place ground truth
        is read). A name nothing has located is unresolved.
        """
        # the goal sits due north: a truth lookup would answer 90 deg
        ctx = make_context(FakeWorld(yaw_deg=0.0, target=(0.0, 5.0, 0.0)))
        result = registry().call(
            "propose_viewpoints", {"target_hypothesis": "blob"}, ctx
        )
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.data["hypothesis_resolved"])
        self.assertIn("blob", result.data["hypothesis_note"])
        self.assertEqual(result.data["base_bearing_deg"], 0.0)
        for item in result.data["viewpoints"]:
            self.assertLessEqual(abs(item["bearing_deg"]), 45.0)
        # the grounding path still resolves the same name once a candidate exists
        ctx = make_context(
            FakeWorld(yaw_deg=0.0),
            extra={"candidate_positions": {"cand_1": [0.0, 5.0, 0.0]}},
        )
        grounded = registry().call("propose_viewpoints", {"target_hypothesis": "cand_1"}, ctx)
        self.assertTrue(grounded.data["hypothesis_resolved"])
        self.assertEqual(grounded.data["base_bearing_deg"], 90.0)

    def test_wall_ahead_blocks_forward_viewpoints(self):
        wall = [[1.5, y, 0.0] for y in (-1.0, -0.5, 0.0, 0.5, 1.0)]
        ctx = make_context(FakeWorld(points=wall))
        result = registry().call("propose_viewpoints", {}, ctx)
        self.assertEqual(result.status, "ok")
        viewpoints = result.data["viewpoints"]
        forward = min(viewpoints, key=lambda item: abs(item["bearing_deg"]))
        self.assertFalse(forward["reachable"])
        self.assertLessEqual(forward["min_clearance_m"], 0.3)
        self.assertLess(result.data["reachable_count"], result.data["count"])
        self.assertGreaterEqual(result.data["reachable_count"], 1)

    def test_no_candidates_when_everything_is_out_of_bounds(self):
        ctx = make_context(FakeWorld(bound_radius=1.0))
        result = registry().call("propose_viewpoints", {}, ctx)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.data["reason"], "no_candidate_viewpoint")

    def test_missing_optional_world_hooks_do_not_crash(self):
        world = FakeWorld(have_fov=False, have_depth_fn=False, have_bev=False,
                          have_trajectory=False)
        world._sense_points = None
        ctx = make_context(world)
        result = registry().call("propose_viewpoints", {}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["reachable_count"], 5)
        self.assertEqual(result.evidence, [])

    def test_driver_style_aliases_n_radius_and_target_id(self):
        # scripts/run_vlm_nav_tools.py and the MCP contract use these arg names
        ctx = make_context(
            FakeWorld(), extra={"candidate_positions": {"cand_1": [0.0, 5.0, 0.0]}}
        )
        result = registry().call(
            "propose_viewpoints",
            {"n": 3, "radius_m": 2.0, "target_id": "cand_1"},
            ctx,
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["count"], 3)
        self.assertEqual(result.data["base_bearing_deg"], 90.0)
        distances = sorted({item["distance_m"] for item in result.data["viewpoints"]})
        self.assertEqual(distances, [1.3, 2.0])
        # the recommended pick is exposed at the top level for $viewpoint_id
        self.assertIn(result.data["viewpoint_id"], ctx.extra["viewpoints_by_id"])

    def test_check_path_still_accepts_every_target_spelling(self):
        score = {"candidate_positions": {"cand_1": [3.0, 0.0, 0.0]}}
        ctx = make_context(FakeWorld(), extra=dict(score))
        checked = registry().call("check_path", {"target_id": "cand_1"}, ctx)
        self.assertTrue(checked.data["reachable"])
        self.assertAlmostEqual(checked.data["distance_m"], 3.0, places=6)
        unknown = registry().call("check_path", {"target_id": "cand_9"}, ctx)
        self.assertEqual((unknown.status, unknown.data["reason"]),
                         ("failed", "unknown_target"))


class CheckPathTest(unittest.TestCase):
    def test_clear_path(self):
        ctx = make_context(FakeWorld())
        result = registry().call("check_path", {"position": [3.0, 0.0, 0.0]}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.data["reachable"])
        self.assertEqual(result.data["reason"], "mostly_unknown")
        self.assertAlmostEqual(result.data["path_cost_m"], 3.0, places=6)
        self.assertEqual(result.data["target_position"], [3.0, 0.0, 0.0])
        self.assertGreater(result.data["unknown_ratio"], 0.0)
        self.assertLess(result.data["unknown_ratio"], 1.0)
        # no observed geometry -> clearance is unknown, not "touching"
        self.assertIsNone(result.data["min_clearance_m"])

    def test_clear_path_reason_when_fully_observed(self):
        # a dense flown trajectory along the leg marks all samples as observed
        world = FakeWorld()
        world.trajectory = [(x * 0.25, 0.0, 0.0) for x in range(13)]
        ctx = make_context(world)
        result = registry().call("check_path", {"position": [3.0, 0.0, 0.0]}, ctx)
        self.assertEqual(result.data["reason"], "clear")
        self.assertAlmostEqual(result.data["unknown_ratio"], 0.0, places=3)

    def test_obstacle_on_path(self):
        wall = [[x * 0.1, 0.0, 0.0] for x in range(10, 20)]
        ctx = make_context(FakeWorld(points=wall))
        result = registry().call("check_path", {"position": [3.0, 0.0, 0.0]}, ctx)
        self.assertEqual(result.status, "blocked")
        self.assertFalse(result.data["reachable"])
        self.assertEqual(result.data["reason"], "obstacle_on_path")
        self.assertGreater(result.data["obstacles_near_path"], 0)
        self.assertIsNotNone(result.data["min_clearance_m"])
        self.assertLessEqual(result.data["min_clearance_m"], 0.3)

    def test_too_far(self):
        ctx = make_context(FakeWorld())
        result = registry().call("check_path", {"position": [20.0, 0.0, 0.0]}, ctx)
        self.assertEqual(result.status, "blocked")
        self.assertFalse(result.data["reachable"])
        self.assertEqual(result.data["reason"], "too_far")
        self.assertEqual(result.data["max_leg_m"], 8.0)

    def test_resolves_viewpoint_id_from_previous_proposal(self):
        ctx = make_context(FakeWorld())
        proposed = registry().call("propose_viewpoints", {}, ctx)
        entry = proposed.data["viewpoints"][0]
        result = registry().call(
            "check_path", {"viewpoint_id": entry["viewpoint_id"]}, ctx
        )
        self.assertEqual(result.data["target_position"], entry["position"])
        self.assertEqual(result.evidence[0].kind, "memory")

    def test_argument_errors(self):
        reg = registry()
        ctx = make_context(FakeWorld())
        missing = reg.call("check_path", {}, ctx)
        self.assertEqual((missing.status, missing.data["reason"]), ("failed", "missing_target"))
        unknown = reg.call("check_path", {"viewpoint_id": "vp_99"}, ctx)
        self.assertEqual((unknown.status, unknown.data["reason"]), ("failed", "unknown_viewpoint"))
        bad = reg.call("check_path", {"position": [1.0, 2.0]}, ctx)
        self.assertEqual((bad.status, bad.data["reason"]), ("failed", "bad_position"))


class MovementIntentTest(unittest.TestCase):
    """One tool per intent, and only the id type of that intent is accepted."""

    def test_the_three_intents_are_separate_tools(self):
        reg = registry()
        self.assertEqual(len([name for name in MOVE_TOOLS if name in reg.names()]), 3)
        # the deprecated name is still registered, but only to answer with the split
        spec = reg.get("navigate")
        self.assertIn("DEPRECATED", spec.description)
        self.assertFalse(spec.side_effects)

    def test_each_tool_takes_only_its_own_id_type(self):
        ctx = make_context(FakeWorld())
        obs_id = seen_at(ctx, (1.0, 0.0, 0.0))
        add_viewpoint(ctx, [1.0, 0.0, 0.0])
        store_candidate(ctx, "cand_1", position=[4.0, 0.0, 0.0])
        reg = registry()

        crossed = reg.call("approach_candidate", {"candidate_id": obs_id}, ctx)
        self.assertEqual(crossed.status, "failed")
        self.assertEqual(crossed.data["error"], "id_type_mismatch")
        self.assertEqual(crossed.data["expected"], "candidate_id")
        self.assertEqual(crossed.data["actual_type"], "observation_id")
        self.assertIn("not a candidate id", crossed.data["hint"])

        wrong = reg.call("go_to_viewpoint", {"viewpoint_id": "cand_1"}, ctx)
        self.assertEqual(wrong.data["error"], "id_type_mismatch")
        self.assertEqual(wrong.data["expected"], "viewpoint_id")

        wrong = reg.call("return_to_observation", {"observation_id": "vp_1"}, ctx)
        self.assertEqual(wrong.data["error"], "id_type_mismatch")
        self.assertEqual(wrong.data["expected"], "observation_id")

    def test_missing_and_unknown_ids_are_answered_with_available_ones(self):
        ctx = make_context(FakeWorld())
        store_candidate(ctx, "cand_1", position=[4.0, 0.0, 0.0])
        reg = registry()

        missing = reg.call("approach_candidate", {}, ctx)
        self.assertEqual(missing.status, "failed")
        self.assertEqual(missing.data["error"], "missing_candidate_id")

        unknown = reg.call("approach_candidate", {"candidate_id": "cand_9"}, ctx)
        self.assertEqual(unknown.data["error"], "unknown_candidate_id")
        self.assertEqual(unknown.data["available_candidate_ids"], ["cand_1"])

        unknown = reg.call("go_to_viewpoint", {"viewpoint_id": "vp_9"}, ctx)
        self.assertEqual(unknown.data["error"], "unknown_viewpoint_id")

        unknown = reg.call("return_to_observation", {"observation_id": "obs_404"}, ctx)
        self.assertEqual(unknown.data["error"], "unknown_observation_id")
        json.dumps(unknown.to_dict())

    def test_a_pose_less_observation_is_refused_with_its_own_error(self):
        # an id whose pose no store knows cannot be flown to
        ctx = make_context(FakeWorld())
        ctx.extra["observations"] = {"obs_1": {"observation_id": "obs_1"}}
        result = registry().call("return_to_observation", {"observation_id": "obs_1"}, ctx)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["reason"], "observation_pose_unknown")
        self.assertIn("observe", result.data["hint"])


class DeprecatedNavigateTest(unittest.TestCase):
    """``navigate`` answers with an explicit error and the replacement call."""

    def test_candidate_target_is_routed_to_approach_candidate(self):
        ctx = make_context(FakeWorld())
        store_candidate(ctx, "cand_1", position=[4.0, 0.0, 0.0])
        before = ctx.world.position.copy()

        result = registry().call("navigate", {"target_id": "cand_1"}, ctx)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["reason"], "deprecated_tool")
        self.assertEqual(result.data["error"], "deprecated_navigate")
        self.assertEqual(result.data["status"], "no_progress")
        self.assertEqual(result.data["outcome"], "deprecated_call")
        self.assertEqual(
            result.data["next_action"],
            {"tool": "approach_candidate", "args": {"candidate_id": "cand_1"}},
        )
        self.assertEqual(result.data["target_position"], [4.0, 0.0, 0.0])
        self.assertEqual(result.data["candidate_id"], "cand_1")
        self.assertEqual(list(before), list(ctx.world.position))  # nothing moved
        for name in MOVE_TOOLS:
            self.assertIn(name, result.data["hint"])

    def test_viewpoint_and_observation_targets_are_routed_to_their_intent(self):
        ctx = make_context(FakeWorld())
        add_viewpoint(ctx, [2.0, 0.0, 0.0])
        obs_id = seen_at(ctx, (3.0, 0.0, 0.0))
        reg = registry()

        moved = reg.call("navigate", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(
            moved.data["next_action"],
            {"tool": "go_to_viewpoint", "args": {"viewpoint_id": "vp_1"}},
        )
        recalled = reg.call("navigate", {"observation_id": obs_id}, ctx)
        self.assertEqual(
            recalled.data["next_action"],
            {"tool": "return_to_observation", "args": {"observation_id": obs_id}},
        )
        self.assertEqual(recalled.data["target_position"], [3.0, 0.0, 0.0])

    def test_an_idempotent_replacement_is_not_suggested(self):
        """The obs id of the pose the drone is at: the answer must not send it back."""
        ctx = make_context(FakeWorld())
        obs_id = seen_at(ctx, (0.0, 0.0, 0.0))  # captured where the drone is

        result = registry().call("navigate", {"target_id": obs_id}, ctx)

        self.assertEqual(result.data["next_action"]["tool"], "ground_target")
        self.assertIn("already", result.data["hint"])
        self.assertIn("approach_candidate", result.data["hint"])

    def test_the_call_executes_nothing_and_can_be_continued_by_its_intent(self):
        ctx = make_context(FakeWorld())
        add_viewpoint(ctx, [20.0, 0.0, 0.0])  # far enough to stay a running task
        reg = registry()
        started = reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(started.data["task_status"], "running")

        legacy = reg.call("navigate", {"task_id": started.data["task_id"]}, ctx)

        self.assertEqual(legacy.status, "failed")
        self.assertEqual(legacy.data["reason"], "deprecated_tool")
        self.assertEqual(
            legacy.data["next_action"],
            {
                "tool": "go_to_viewpoint",
                "args": {"viewpoint_id": "vp_1", "task_id": started.data["task_id"]},
            },
        )
        self.assertEqual(len(ctx.world.applied), 1)  # the legacy call flew nothing

    def test_a_finished_task_is_not_sent_back_to_its_own_place(self):
        """A replacement that cannot move is replaced again (the loop guard)."""
        ctx = make_context(FakeWorld())
        add_viewpoint(ctx, [3.0, 0.0, 0.0])
        reg = registry()
        started = reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(started.data["task_status"], "arrived")

        legacy = reg.call("navigate", {"task_id": started.data["task_id"]}, ctx)

        self.assertEqual(legacy.data["next_action"]["tool"], "propose_viewpoints")
        self.assertIn("already", legacy.data["hint"])

    def test_an_empty_legacy_call_names_the_three_tools(self):
        ctx = make_context(FakeWorld())
        result = registry().call("navigate", {}, ctx)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["next_action"]["tool"], "observe")
        self.assertEqual(sorted(result.data["intent_tools"]), sorted(MOVE_TOOLS))

    def test_a_task_of_another_intent_is_refused(self):
        ctx = make_context(FakeWorld())
        add_viewpoint(ctx, [3.0, 0.0, 0.0])
        store_candidate(ctx, "cand_1", position=[3.0, 0.0, 0.0])
        reg = registry()
        started = reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)

        crossed = reg.call(
            "approach_candidate", {"candidate_id": "cand_1", "task_id": started.data["task_id"]}, ctx
        )

        self.assertEqual(crossed.status, "failed")
        self.assertEqual(crossed.data["reason"], "task_intent_mismatch")
        self.assertIn("go_to_viewpoint", crossed.data["hint"])


class ProgressSchemaTest(unittest.TestCase):
    """One progress schema for every movement tool (goal.md section 5)."""

    KEYS = (
        "status",
        "task_status",
        "outcome",
        "progress_m",
        "turn_deg",
        "remaining_m",
        "blocked_reason",
        "task_id",
        "last_observation_id",
    )

    def setUp(self):
        self.ctx = make_context(FakeWorld())
        add_viewpoint(self.ctx, [3.0, 0.0, 0.0])
        self.reg = registry()

    def assert_schema(self, result):
        for key in self.KEYS:
            self.assertIn(key, result.data, key)
        self.assertIn(result.data["status"], navigation_module.PROGRESS_STATUSES)
        self.assertTrue(result.data["task_id"])
        json.dumps(result.to_dict())

    def test_every_movement_tool_answers_with_the_same_fields(self):
        carried = self.reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, self.ctx)
        self.assert_schema(carried)
        self.assertEqual((carried.data["status"], carried.data["outcome"]), ("ok", "arrived"))
        self.assertAlmostEqual(carried.data["progress_m"], 3.0, places=6)
        self.assertAlmostEqual(carried.data["turn_deg"], 0.0, places=6)
        self.assertAlmostEqual(carried.data["remaining_m"], 0.0, places=6)

        ctx = make_context(FakeWorld())
        store_candidate(ctx, "cand_1", position=[3.0, 0.0, 0.0])
        approached = registry().call(
            "approach_candidate", {"candidate_id": "cand_1", "standoff_m": 1.0}, ctx
        )
        self.assert_schema(approached)
        self.assertEqual(approached.data["candidate_id"], "cand_1")
        self.assertAlmostEqual(approached.data["progress_m"], 2.0, places=6)
        # remaining_m is the distance still to fly, i.e. beyond the standoff
        self.assertAlmostEqual(approached.data["remaining_m"], 0.0, places=6)
        self.assertAlmostEqual(approached.data["standoff_m"], 1.0, places=6)

        ctx = make_context(FakeWorld())
        obs_id = seen_at(ctx, (2.0, 0.0, 0.0))
        returned = registry().call("return_to_observation", {"observation_id": obs_id}, ctx)
        self.assert_schema(returned)
        self.assertEqual(returned.data["observation_id"], obs_id)

    def test_the_lifecycle_status_stays_visible_next_to_the_progress_one(self):
        result = self.reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, self.ctx)
        self.assertEqual(result.data["task_status"], "arrived")
        self.assertEqual(result.data["status"], "ok")
        self.assertEqual(result.data["reason"], "arrived")
        # and the task record keeps its own lifecycle
        task = self.ctx.extra["nav_tasks"][result.data["task_id"]]
        self.assertEqual(task["status"], "arrived")
        self.assertEqual(task["intent"], "go_to_viewpoint")
        self.assertEqual(task["viewpoint_id"], "vp_1")

    def test_partial_progress_is_ok(self):
        ctx = make_context(FakeWorld())
        add_viewpoint(ctx, [20.0, 0.0, 0.0])
        result = registry().call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(result.data["status"], "ok")
        self.assertEqual(result.data["task_status"], "running")
        self.assertEqual(result.data["outcome"], "partial")
        self.assertAlmostEqual(result.data["progress_m"], 8.0, places=6)
        self.assertAlmostEqual(result.data["remaining_m"], 12.0, places=6)

    def test_zero_displacement_is_no_progress(self):
        ctx = make_context(FakeWorld(move=False))
        add_viewpoint(ctx, [3.0, 0.0, 0.0])
        result = registry().call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["status"], "no_progress")
        self.assertEqual(result.data["task_status"], "failed")
        self.assertEqual(result.data["outcome"], "stalled")
        self.assertEqual(result.data["blocked_reason"], "no_progress")
        self.assertAlmostEqual(result.data["progress_m"], 0.0, places=6)
        self.assertIn("do not repeat this call", result.data["hint"])

    def test_bounds_clamp_is_no_progress_with_its_own_outcome(self):
        ctx = make_context(FakeWorld(move=False, outcome="clamped by scene bounds"))
        add_viewpoint(ctx, [3.0, 0.0, 0.0])
        result = registry().call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(result.data["status"], "no_progress")
        self.assertEqual(result.data["outcome"], "outside_bound")

    def test_a_blocked_leg_is_blocked_not_failed(self):
        wall = [[x * 0.1, 0.0, 0.0] for x in range(10, 20)]
        ctx = make_context(FakeWorld(points=wall))
        add_viewpoint(ctx, [3.0, 0.0, 0.0])
        result = registry().call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.data["status"], "blocked")
        self.assertEqual(result.data["outcome"], "obstacle_on_path")
        self.assertEqual(result.data["blocked_reason"], "obstacle_on_path")
        self.assertAlmostEqual(result.data["progress_m"], 0.0, places=6)

    def test_collision_outcome_after_a_moved_leg_is_blocked(self):
        ctx = make_context(FakeWorld(outcome="collision"))
        add_viewpoint(ctx, [3.0, 0.0, 0.0])
        result = registry().call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.data["status"], "blocked")
        self.assertEqual(result.data["outcome"], "collision")
        self.assertGreater(result.data["progress_m"], 0.0)


class GoToViewpointTest(unittest.TestCase):
    def test_single_leg_reaches_the_viewpoint(self):
        world = FakeWorld()
        ctx = make_context(world)
        add_viewpoint(ctx, [3.0, 0.0, 0.0])
        result = registry().call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(result.status, "ok")
        data = result.data
        self.assertEqual(data["task_id"], "nav_1")
        self.assertEqual(data["task_status"], "arrived")
        self.assertEqual(data["reason"], "arrived")
        self.assertAlmostEqual(data["progress_m"], 3.0, places=6)
        self.assertAlmostEqual(data["moved_m"], 3.0, places=6)
        self.assertAlmostEqual(data["remaining_m"], 0.0, places=6)
        self.assertAlmostEqual(data["segment_progress_m"], 3.0, places=6)
        self.assertEqual(data["segments"], 1)
        self.assertEqual(data["candidate_viewpoints"], [])
        obs_id = data["last_observation_id"]
        self.assertEqual(obs_id, "obs_1")  # the first frame of this episode
        self.assertEqual(ctx.extra["last_observation_id"], obs_id)
        self.assertIn(obs_id, ctx.extra["observations"])
        # the post-leg frame is published for the runtime's next-turn context ...
        self.assertEqual(ctx.extra["observations"][obs_id]["kind"], "post_leg")
        self.assertIn(obs_id, ctx.extra["images"])
        # ... and registered in the unified observation store
        store = observation_store(ctx)
        self.assertIsNotNone(store.get(obs_id))
        self.assertIsNotNone(store.get(obs_id).pose_flu[0])
        task = ctx.extra["nav_tasks"]["nav_1"]
        self.assertEqual(task["status"], "arrived")
        self.assertEqual(task["target_position"], [3.0, 0.0, 0.0])
        self.assertEqual(task["segments"], 1)
        self.assertEqual(task["intent"], "go_to_viewpoint")
        self.assertEqual([e["event"] for e in task["events"]], ["arrived"])
        self.assertEqual(len(world.applied), 1)
        action = world.applied[0]
        self.assertEqual(action.kind, "waypoint")
        self.assertAlmostEqual(action.forward_m, 3.0, places=6)
        self.assertAlmostEqual(action.right_m, 0.0, places=6)
        self.assertAlmostEqual(action.yaw_deg, 0.0, places=6)
        self.assertEqual(action.reason, "go_to_viewpoint")
        kinds = {e.kind for e in result.evidence}
        self.assertIn("observation", kinds)
        self.assertIn("trajectory", kinds)
        self.assertIn("task_id=nav_1", result.to_prompt())
        json.dumps(result.to_dict())

    def test_leg_faces_the_direction_of_travel(self):
        world = FakeWorld(yaw_deg=-90.0)
        ctx = make_context(world)
        add_viewpoint(ctx, [0.0, 3.0, 0.0])
        result = registry().call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["status"], "ok")
        action = world.applied[0]
        self.assertAlmostEqual(action.forward_m, 3.0, places=6)
        self.assertAlmostEqual(action.right_m, 0.0, places=6)
        self.assertAlmostEqual(action.yaw_deg, 180.0, places=6)
        self.assertAlmostEqual(math.degrees(world.yaw_rad), 90.0, places=6)
        self.assertAlmostEqual(result.data["progress_m"], 3.0, places=6)
        self.assertAlmostEqual(result.data["turn_deg"], 180.0, places=6)

    def test_blocked_by_collision_reports_alternatives(self):
        world = FakeWorld(outcome="collision")
        ctx = make_context(world)
        add_viewpoint(ctx, [3.0, 0.0, 0.0])
        result = registry().call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.data["task_status"], "blocked")
        self.assertEqual(result.data["outcome"], "collision")
        alternatives = result.data["candidate_viewpoints"]
        self.assertGreaterEqual(len(alternatives), 1)
        for vid in alternatives:
            self.assertIn(vid, ctx.extra["viewpoints_by_id"])
        self.assertEqual(
            result.data["next_action"],
            {"tool": "go_to_viewpoint", "args": {"viewpoint_id": alternatives[0]}},
        )
        self.assertEqual(
            [e["event"] for e in ctx.extra["nav_tasks"]["nav_1"]["events"]],
            ["path_blocked"],
        )

    def test_blocked_by_precheck_does_not_move(self):
        wall = [[x * 0.1, 0.0, 0.0] for x in range(10, 20)]
        world = FakeWorld(points=wall)
        ctx = make_context(world)
        add_viewpoint(ctx, [3.0, 0.0, 0.0])
        result = registry().call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.data["reason"], "obstacle_on_path")
        self.assertEqual(world.applied, [])
        self.assertAlmostEqual(result.data["progress_m"], 0.0, places=6)
        task = ctx.extra["nav_tasks"]["nav_1"]
        self.assertEqual(task["status"], "blocked")
        self.assertEqual(task["segments"], 0)

    def test_blocked_leg_suggests_reachable_viewpoint_behind(self):
        wall = [[x * 0.1, 0.0, 0.0] for x in range(10, 20)]  # wall ahead at y = 0
        world = FakeWorld(points=wall)
        ctx = make_context(world)
        add_viewpoint(ctx, [3.0, 0.0, 0.0])
        result = registry().call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(result.status, "blocked")
        alternatives = result.data["candidate_viewpoints"]
        self.assertGreaterEqual(len(alternatives), 1)
        for vid in alternatives:
            self.assertTrue(ctx.extra["viewpoints_by_id"][vid]["reachable"])

    def test_leg_farther_than_the_leg_limit_is_flown_graded(self):
        # a long journey is executed as several clamped legs, never as one
        # impossible hop and never as a hard failure
        world = FakeWorld()
        ctx = make_context(world)
        add_viewpoint(ctx, [20.0, 0.0, 0.0])
        result = registry().call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["task_status"], "running")
        self.assertEqual(result.data["reason"], "partial")
        self.assertAlmostEqual(result.data["remaining_m"], 12.0, places=6)
        self.assertAlmostEqual(result.data["progress_m"], 8.0, places=6)
        self.assertEqual(result.data["max_leg_m"], 8.0)
        self.assertEqual(len(world.applied), 1)
        self.assertAlmostEqual(world.applied[0].forward_m, 8.0, places=6)

    def test_no_progress_is_failed(self):
        world = FakeWorld(move=False)
        ctx = make_context(world)
        add_viewpoint(ctx, [3.0, 0.0, 0.0])
        result = registry().call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["task_status"], "failed")
        self.assertEqual(result.data["reason"], "no_progress")
        self.assertAlmostEqual(result.data["progress_m"], 0.0, places=6)
        self.assertEqual(
            [e["event"] for e in ctx.extra["nav_tasks"]["nav_1"]["events"]],
            ["task_failed"],
        )

    def test_leg_timeout_without_progress_is_failed(self):
        world = FakeWorld(move=False, outcome="leg timeout")
        ctx = make_context(world)
        add_viewpoint(ctx, [3.0, 0.0, 0.0])
        result = registry().call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["reason"], "leg_timeout")

    def test_bounds_clamped_leg_fails_with_its_own_reason(self):
        # the world refuses to fly (scene bounds): a hard failure, not progress
        world = FakeWorld(move=False, outcome="clamped by scene bounds")
        ctx = make_context(world)
        add_viewpoint(ctx, [3.0, 0.0, 0.0])
        result = registry().call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["task_status"], "failed")
        self.assertEqual(result.data["reason"], "outside_bound")
        self.assertEqual(ctx.extra["nav_tasks"]["nav_1"]["status"], "failed")

    def test_already_there_is_no_progress_not_arrival(self):
        """The zero-displacement case the old ``navigate`` reported as ok+arrived."""
        world = FakeWorld()
        ctx = make_context(world)
        add_viewpoint(ctx, [0.2, 0.0, 0.0])
        result = registry().call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["status"], "no_progress")
        self.assertEqual(result.data["outcome"], "already_at_pose")
        self.assertEqual(result.data["task_status"], "arrived")
        self.assertEqual(result.data["reason"], "already_at_pose")
        self.assertEqual(world.applied, [])
        self.assertEqual(result.data["task_id"], "nav_1")
        self.assertEqual(result.data["turn_deg"], 0.0)
        self.assertNotEqual(result.data["next_action"]["tool"], "go_to_viewpoint")
        task = ctx.extra["nav_tasks"]["nav_1"]
        self.assertEqual(task["events"], [])

    def test_task_ids_increment_and_resolve_viewpoints(self):
        world = FakeWorld()
        ctx = make_context(world)
        proposed = registry().call("propose_viewpoints", {}, ctx)
        entry = proposed.data["viewpoints"][0]
        first = registry().call("go_to_viewpoint", {"viewpoint_id": entry["viewpoint_id"]}, ctx)
        add_viewpoint(ctx, [3.0, 0.0, 0.0], viewpoint_id="vp_9")
        second = registry().call("go_to_viewpoint", {"viewpoint_id": "vp_9"}, ctx)
        self.assertEqual(first.data["task_id"], "nav_1")
        self.assertEqual(second.data["task_id"], "nav_2")
        self.assertEqual(sorted(ctx.extra["nav_tasks"]), ["nav_1", "nav_2"])

    def test_unknown_viewpoint_fails_cleanly(self):
        ctx = make_context(FakeWorld())
        result = registry().call("go_to_viewpoint", {"viewpoint_id": "vp_42"}, ctx)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["error"], "unknown_viewpoint_id")
        # a refused call starts no task at all
        self.assertNotIn("nav_tasks", ctx.extra)

    def test_unknown_task_id_fails_cleanly(self):
        ctx = make_context(FakeWorld())
        add_viewpoint(ctx, [3.0, 0.0, 0.0])
        result = registry().call(
            "go_to_viewpoint", {"viewpoint_id": "vp_1", "task_id": "nav_9"}, ctx
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["reason"], "unknown_task")
        self.assertEqual(result.data["known_tasks"], [])


class ApproachCandidateTest(unittest.TestCase):
    """``approach_candidate``: locate -> standoff waypoint -> segment -> re-look."""

    def test_it_stops_on_the_standoff_shell_of_the_candidate(self):
        ctx = make_context(FakeWorld())
        store_candidate(ctx, "cand_1", position=[4.0, 0.0, 0.0])
        result = registry().call(
            "approach_candidate", {"candidate_id": "cand_1", "standoff_m": 1.0}, ctx
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["status"], "ok")
        self.assertEqual(result.data["outcome"], "arrived")
        self.assertEqual(result.data["task_status"], "arrived")
        self.assertEqual(result.data["candidate_id"], "cand_1")
        self.assertEqual(result.data["position_source"], "depth")
        self.assertAlmostEqual(result.data["progress_m"], 3.0, places=6)
        self.assertAlmostEqual(result.data["moved_m"], 3.0, places=6)
        self.assertAlmostEqual(result.data["remaining_m"], 0.0, places=6)
        self.assertAlmostEqual(result.data["standoff_m"], 1.0, places=6)
        self.assertEqual(result.data["segments"], 1)
        self.assertAlmostEqual(float(np.linalg.norm(ctx.world.position)), 3.0, places=6)
        # only the leg's own leg flew: the offset stops 1 m short of the target
        self.assertEqual(len(ctx.world.applied), 1)
        self.assertAlmostEqual(ctx.world.applied[0].forward_m, 3.0, places=6)
        json.dumps(result.to_dict())

    def test_a_long_approach_is_flown_in_segments(self):
        ctx = make_context(FakeWorld())
        store_candidate(ctx, "cand_1", position=[20.0, 0.0, 0.0])
        reg = registry()
        first = reg.call(
            "approach_candidate", {"candidate_id": "cand_1", "standoff_m": 1.2}, ctx
        )
        self.assertEqual(first.data["status"], "ok")
        self.assertEqual(first.data["outcome"], "partial")
        self.assertAlmostEqual(first.data["progress_m"], 8.0, places=6)
        self.assertAlmostEqual(first.data["remaining_m"], 10.8, places=6)
        self.assertAlmostEqual(first.data["standoff_m"], 1.2, places=6)
        task_id = first.data["task_id"]

        second = reg.call(
            "approach_candidate", {"candidate_id": "cand_1", "task_id": task_id}, ctx
        )
        self.assertEqual(second.data["task_id"], task_id)
        self.assertAlmostEqual(second.data["progress_m"], 16.0, places=6)

        third = reg.call(
            "approach_candidate", {"candidate_id": "cand_1", "task_id": task_id}, ctx
        )
        self.assertEqual(third.data["outcome"], "arrived")
        self.assertEqual(third.data["task_status"], "arrived")
        self.assertAlmostEqual(third.data["progress_m"], 18.8, places=6)
        self.assertAlmostEqual(float(ctx.world.position[0]), 18.8, places=6)
        # every segment re-observed: one observation id per leg
        self.assertEqual(third.data["segments"], 3)
        self.assertEqual(len(ctx.extra["observations"]), 3)

    def test_the_leg_ends_short_of_the_surface_and_re_measures_it(self):
        """The re-localization contract: a fresh depth read replaces a stale range."""
        world = DepthWorld(surface_m=5.0)
        ctx = make_context(world)
        # the candidate was grounded at 4 m, the surface is really at 5 m
        store_candidate(ctx, "cand_1", position=[4.0, 0.0, 0.0], range_m=4.0)
        reg = registry()
        first = reg.call(
            "approach_candidate", {"candidate_id": "cand_1", "standoff_m": 1.2}, ctx
        )
        measurement = first.data["relocalization"]
        self.assertTrue(measurement["relocalized"])
        self.assertEqual(measurement["reason"], "depth")
        self.assertEqual(first.data["target_position"], [5.0, 0.0, 0.0])
        self.assertEqual(
            ctx.extra[CANDIDATE_STORE_KEY]["cand_1"]["position_flu"], [5.0, 0.0, 0.0]
        )
        self.assertEqual(ctx.extra[CANDIDATE_STORE_KEY]["cand_1"]["relocalized_count"], 1)

        second = reg.call(
            "approach_candidate",
            {"candidate_id": "cand_1", "task_id": first.data["task_id"]},
            ctx,
        )
        self.assertEqual(second.data["outcome"], "arrived")
        self.assertAlmostEqual(float(ctx.world.position[0]), 3.8, places=6)

    def test_a_candidate_known_only_as_a_ray_is_stepped_and_then_measured(self):
        world = DepthWorld(surface_m=5.0)
        ctx = make_context(world)
        store_candidate(
            ctx,
            "cand_1",
            ray={"origin_flu": [0.0, 0.0, 0.0], "bearing_deg": 0.0, "elevation_deg": 0.0},
        )
        reg = registry()
        first = reg.call("approach_candidate", {"candidate_id": "cand_1"}, ctx)
        self.assertEqual(first.data["status"], "ok")
        self.assertEqual(first.data["outcome"], "partial")
        self.assertEqual(first.data["position_source"], "ray")
        self.assertAlmostEqual(float(ctx.world.position[0]), 3.0, places=6)  # one ray step
        self.assertEqual(first.data["waypoint"], [3.0, 0.0, 0.0])
        # the post-leg frame measured the range: 5 m wall seen from 3 m
        self.assertTrue(first.data["relocalization"]["relocalized"])
        self.assertEqual(first.data["target_position"], [5.0, 0.0, 0.0])
        self.assertAlmostEqual(first.data["remaining_m"], 0.8, places=6)

        second = reg.call(
            "approach_candidate",
            {"candidate_id": "cand_1", "task_id": first.data["task_id"]},
            ctx,
        )
        self.assertEqual(second.data["target_position"], [5.0, 0.0, 0.0])
        self.assertEqual(second.data["outcome"], "arrived")

    def test_a_ray_without_any_depth_keeps_stepping_and_says_so(self):
        ctx = make_context(FakeWorld(depth=None))  # no depth sensor at all
        store_candidate(
            ctx,
            "cand_1",
            ray={"origin_flu": [0.0, 0.0, 0.0], "bearing_deg": 0.0, "elevation_deg": 0.0},
        )
        result = registry().call("approach_candidate", {"candidate_id": "cand_1"}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["status"], "ok")
        self.assertIsNone(result.data["remaining_m"])
        self.assertIn("no range", result.data["hint"])
        self.assertEqual(result.data["status"], "ok")
        self.assertFalse(result.data["relocalization"].get("relocalized"))

    def test_a_candidate_with_no_location_is_blocked_not_guessed(self):
        ctx = make_context(FakeWorld())
        store_candidate(ctx, "cand_1", position=None, observation_id="obs_1")
        result = registry().call("approach_candidate", {"candidate_id": "cand_1"}, ctx)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.data["reason"], "candidate_has_no_position")
        self.assertEqual(result.data["outcome"], "unlocatable_candidate")
        self.assertEqual(result.data["available_targets"], [])
        self.assertEqual(result.data["unresolved_targets"], ["cand_1"])
        self.assertEqual(result.data["next_action"]["tool"], "ground_target")
        self.assertEqual(ctx.world.applied, [])
        json.dumps(result.to_dict())

    def test_the_standoff_default_is_the_worlds_own(self):
        class StandoffWorld(FakeWorld):
            standoff_m = 2.0

            def _apply_standoff(self, dx, dy, dz):
                total = math.sqrt(dx * dx + dy * dy + dz * dz)
                if total <= self.standoff_m:
                    return 0.0, 0.0, 0.0
                scale = (total - self.standoff_m) / total
                return dx * scale, dy * scale, dz * scale

        ctx = make_context(StandoffWorld())
        store_candidate(ctx, "cand_1", position=[5.0, 0.0, 0.0])
        result = registry().call("approach_candidate", {"candidate_id": "cand_1"}, ctx)
        self.assertAlmostEqual(result.data["standoff_m"], 2.0, places=6)
        self.assertAlmostEqual(float(ctx.world.position[0]), 3.0, places=6)
        self.assertEqual(result.data["outcome"], "arrived")

    def test_the_skill_geometry_helpers_are_reusable(self):
        # the standoff rule and the ray step are the skill's public contract
        offset = skills_approach.standoff_offset(None, [4.0, 0.0, 0.0], 1.0)
        self.assertAlmostEqual(float(offset[0]), 3.0, places=6)
        self.assertTrue(np.allclose(skills_approach.standoff_offset(None, [0.5, 0.0, 0.0], 1.0), 0.0))
        point = skills_approach.ray_point(
            [0.0, 0.0, 0.0], {"bearing_deg": 90.0, "elevation_deg": 0.0}, 2.0
        )
        self.assertAlmostEqual(float(point[1]), 2.0, places=6)
        # a world without a declared FOV cannot be re-projected: the skill says
        # so instead of assuming one
        ctx = make_context(FakeWorld())
        target = skills_approach.CandidateTarget(
            "cand_1", position=np.asarray([3.0, 0.0, 0.0])
        )
        self.assertEqual(
            skills_approach.relocalize(ctx, target)["reason"], "intrinsics_unknown"
        )

    def test_approach_repeats_are_cached_and_still_report_the_progress(self):
        ctx = make_context(FakeWorld())
        store_candidate(ctx, "cand_1", position=[4.0, 0.0, 0.0])
        reg = registry()
        first = reg.call("approach_candidate", {"candidate_id": "cand_1"}, ctx)
        self.assertEqual(first.data["outcome"], "arrived")
        repeat = reg.call("approach_candidate", {"candidate_id": "cand_1"}, ctx)
        self.assertEqual(repeat.status, "cached")
        self.assertEqual(repeat.data["outcome"], "arrived")
        self.assertIn("do not repeat", repeat.data["hint"])
        self.assertEqual(len(ctx.world.applied), 1)

    def test_a_stuck_world_reports_no_progress(self):
        ctx = make_context(FakeWorld(move=False))
        store_candidate(ctx, "cand_1", position=[4.0, 0.0, 0.0])
        result = registry().call("approach_candidate", {"candidate_id": "cand_1"}, ctx)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["status"], "no_progress")
        self.assertEqual(result.data["outcome"], "stalled")
        self.assertIn("do not repeat this call", result.data["hint"])

    def test_once_on_the_shell_the_call_is_already_at_pose(self):
        ctx = make_context(FakeWorld(position=(3.0, 0.0, 0.0)))
        store_candidate(ctx, "cand_1", position=[4.0, 0.0, 0.0])
        result = registry().call(
            "approach_candidate", {"candidate_id": "cand_1", "standoff_m": 1.0}, ctx
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["status"], "no_progress")
        self.assertEqual(result.data["outcome"], "already_at_pose")
        self.assertEqual(result.data["next_action"]["tool"], "verify_goal")
        self.assertEqual(ctx.world.applied, [])


class RepeatedMoveGuardTest(unittest.TestCase):
    """A movement that keeps answering ``partial`` without progress is guarded.

    The round-3 seed-7 resume spent 13 consecutive ``approach_candidate`` calls
    on the same candidate, every one ``ok`` + ``partial`` with no range, until
    the budget was gone. Three consecutive no-progress moves on one
    (intent, target) now end in ``blocked`` + a ``strategy_change``, and
    ``task_status`` reports the stuck task.
    """

    RAY = {"origin_flu": [0.0, 0.0, 0.0], "bearing_deg": 0.0, "elevation_deg": 0.0}

    def stuck_context(self):
        ctx = make_context(FakeWorld(depth=None))  # no depth: the range stays unknown
        store_candidate(ctx, "cand_1", ray=self.RAY)
        return ctx

    def test_the_third_fruitless_move_is_blocked_with_a_strategy_change(self):
        ctx = self.stuck_context()
        reg = registry()
        first = reg.call("approach_candidate", {"candidate_id": "cand_1"}, ctx)
        second = reg.call("approach_candidate", {"candidate_id": "cand_1"}, ctx)
        third = reg.call("approach_candidate", {"candidate_id": "cand_1"}, ctx)
        blocked = reg.call("approach_candidate", {"candidate_id": "cand_1"}, ctx)

        # the first leg really moved; the two after it changed nothing
        self.assertEqual(first.data["outcome"], "partial")
        self.assertEqual(second.data["outcome"], "partial")
        self.assertEqual(third.data["outcome"], "partial")
        self.assertEqual(blocked.status, "blocked")
        self.assertEqual(blocked.data["reason"], "repeating_move")
        self.assertEqual(blocked.data["outcome"], "repeating_move")
        self.assertEqual(blocked.data["task_status"], "blocked")
        self.assertTrue(blocked.data["repeating_move"])
        self.assertEqual(blocked.data["repeat_count"], 3)
        # the guard fires before the leg: the fourth call flew nothing
        self.assertEqual(len(ctx.world.applied), 3)
        self.assertNotEqual(
            blocked.data["next_action"],
            {"tool": "approach_candidate", "args": {"candidate_id": "cand_1"}},
        )
        change = blocked.data["strategy_change"]
        self.assertEqual(change["reason"], "repeating_move")
        kinds = [option["kind"] for option in change["options"]]
        # the stuck candidate has only a ray (no world position): re-ground it
        # first, then change the viewpoint; searching elsewhere is not the lead
        self.assertEqual(kinds[0], "reground")
        self.assertEqual(kinds[1], "change_viewpoint")
        self.assertEqual(
            set(kinds), {"reground", "change_viewpoint", "verify", "observe"}
        )
        for option in change["options"]:
            with self.subTest(option=option["kind"]):
                self.assertTrue(option["tool"])
                self.assertTrue(option["why"])

    def test_the_strategy_prefers_reapproaching_already_located_target(self):
        """A repeating move against a *located* candidate leads with the call
        that goes back to that candidate: change_viewpoint is second."""
        ctx = self.stuck_context()
        store_candidate(ctx, "cand_2", position=[5.0, 0.0, 0.0])
        task = {
            "task_id": "nav_1",
            "intent": "approach_candidate",
            "status": "blocked",
            "reason": "repeating_move",
            "candidate_id": "cand_2",
            "repeat_move": {"streak": 3},
        }
        change = navigation_module._repeat_move_strategy(ctx, task)
        first, second = change["options"][:2]
        self.assertEqual(first["kind"], "reapproach")
        self.assertEqual(first["tool"], "approach_candidate")
        self.assertEqual(first["args"]["candidate_id"], "cand_2")
        self.assertEqual(second["kind"], "change_viewpoint")
        self.assertEqual(second["tool"], "propose_viewpoints")

    def test_a_progressing_approach_is_not_blocked(self):
        """Real progress resets the streak (the long approach reaches the shell)."""
        ctx = make_context(FakeWorld())
        store_candidate(ctx, "cand_1", position=[20.0, 0.0, 0.0])
        reg = registry()
        first = reg.call(
            "approach_candidate", {"candidate_id": "cand_1", "standoff_m": 1.2}, ctx
        )
        second = reg.call(
            "approach_candidate",
            {"candidate_id": "cand_1", "task_id": first.data["task_id"]},
            ctx,
        )
        third = reg.call(
            "approach_candidate",
            {"candidate_id": "cand_1", "task_id": first.data["task_id"]},
            ctx,
        )
        self.assertEqual(second.data["outcome"], "partial")
        self.assertEqual(third.data["outcome"], "arrived")
        self.assertNotEqual(third.status, "blocked")

    def test_the_guard_is_per_target(self):
        """A stuck target does not block a different candidate."""
        ctx = make_context(FakeWorld(depth=None))
        store_candidate(ctx, "cand_1", ray=self.RAY)
        store_candidate(
            ctx,
            "cand_2",
            ray={"origin_flu": [0.0, 0.0, 0.0], "bearing_deg": 90.0, "elevation_deg": 0.0},
        )
        reg = registry()
        for _ in range(4):
            reg.call("approach_candidate", {"candidate_id": "cand_1"}, ctx)
        other = reg.call("approach_candidate", {"candidate_id": "cand_2"}, ctx)
        self.assertEqual(other.data["outcome"], "partial")
        self.assertNotEqual(other.data["reason"], "repeating_move")

    def test_task_status_reports_the_stuck_task(self):
        ctx = self.stuck_context()
        reg = registry()
        blocked = None
        for _ in range(4):
            blocked = reg.call("approach_candidate", {"candidate_id": "cand_1"}, ctx)
        self.assertEqual(blocked.status, "blocked")

        status = reg.call(
            "task_status", {"task_id": blocked.data["task_id"]}, ctx
        )
        self.assertTrue(status.data["stuck"])
        self.assertEqual(status.data["reason"], "repeating_move")
        self.assertEqual(status.data["repeat_count"], 3)
        self.assertIn("stuck", status.data["hint"])
        self.assertIn("strategy", status.data["hint"])
        self.assertEqual(
            status.data["strategy_change"]["reason"], "repeating_move"
        )


class ReturnToObservationTest(unittest.TestCase):
    def test_it_flies_back_to_the_capture_pose(self):
        ctx = make_context(FakeWorld())
        obs_id = seen_at(ctx, (2.0, 0.0, 0.0), yaw_deg=180.0)
        result = registry().call("return_to_observation", {"observation_id": obs_id}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["outcome"], "arrived")
        self.assertEqual(result.data["observation_id"], obs_id)
        self.assertEqual(result.data["target_position"], [2.0, 0.0, 0.0])
        self.assertAlmostEqual(float(ctx.world.position[0]), 2.0, places=6)
        # the recorded heading is restored in the same call
        self.assertAlmostEqual(math.degrees(ctx.world.yaw_rad), 180.0, places=3)
        self.assertAlmostEqual(result.data["turn_deg"], 180.0, places=3)
        self.assertIn(f"observation:{obs_id}", result.to_prompt())

    def test_restore_yaw_false_keeps_the_flight_heading(self):
        ctx = make_context(FakeWorld())
        obs_id = seen_at(ctx, (2.0, 0.0, 0.0), yaw_deg=180.0)
        result = registry().call(
            "return_to_observation", {"observation_id": obs_id, "restore_yaw": False}, ctx
        )
        self.assertEqual(result.data["outcome"], "arrived")
        self.assertAlmostEqual(math.degrees(ctx.world.yaw_rad), 0.0, places=3)
        self.assertEqual(registry().get("return_to_observation").input_schema
                         ["properties"]["restore_yaw"]["type"], "boolean")

    def test_the_capture_pose_of_the_current_frame_is_already_at_pose(self):
        ctx = make_context(FakeWorld())
        obs_id = seen_at(ctx, (0.0, 0.0, 0.0))
        result = registry().call("return_to_observation", {"observation_id": obs_id}, ctx)
        self.assertEqual(result.data["status"], "no_progress")
        self.assertEqual(result.data["outcome"], "already_at_pose")
        self.assertNotEqual(result.status, "ok")
        self.assertEqual(ctx.world.applied, [])

    def test_a_heading_to_restore_is_real_work_not_a_no_op(self):
        ctx = make_context(FakeWorld(yaw_deg=90.0))
        obs_id = seen_at(ctx, (0.0, 0.0, 0.0), yaw_deg=0.0)
        result = registry().call("return_to_observation", {"observation_id": obs_id}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["status"], "ok")
        self.assertEqual(result.data["outcome"], "arrived")
        self.assertAlmostEqual(result.data["progress_m"], 0.0, places=6)
        self.assertAlmostEqual(result.data["turn_deg"], -90.0, places=3)
        self.assertAlmostEqual(math.degrees(ctx.world.yaw_rad), 0.0, places=3)

    def test_a_navigation_pose_record_is_a_usable_observation(self):
        ctx = make_context(FakeWorld())
        add_viewpoint(ctx, [2.0, 0.0, 0.0])
        first = registry().call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        obs_id = first.data["last_observation_id"]
        self.assertTrue(obs_id)

        ctx.world.position = np.asarray([5.0, 0.0, 0.0])  # the drone flew on
        back = registry().call("return_to_observation", {"observation_id": obs_id}, ctx)
        self.assertEqual(back.status, "ok", back.data)
        self.assertEqual(back.data["target_position"], [2.0, 0.0, 0.0])


class ZeroDisplacementLoopTest(unittest.TestCase):
    """The reported real-trajectory loop: the model cites the current frame's obs
    id, the flight moves nothing, and the same call is suggested again. Under the
    intent contract that call is ``no_progress`` + ``already_at_pose`` with a
    *different* executable next step, and the episode ends instead of looping."""

    CURRENT_FRAME = "obs_1"

    def loop_context(self):
        ctx = make_context(FakeWorld())
        seen_at(ctx, (0.0, 0.0, 0.0))  # the frame the runtime shows at this pose
        return ctx

    def test_the_current_frames_id_is_no_progress_never_ok_arrived(self):
        ctx = self.loop_context()
        result = registry().call(
            "return_to_observation", {"observation_id": self.CURRENT_FRAME}, ctx
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["status"], "no_progress")
        self.assertEqual(result.data["outcome"], "already_at_pose")
        self.assertNotEqual(
            (result.status, result.data["outcome"]), ("ok", "arrived")
        )
        self.assertAlmostEqual(result.data["progress_m"], 0.0, places=6)
        self.assertAlmostEqual(result.data["turn_deg"], 0.0, places=6)

    def test_the_recovery_step_is_different_and_executable(self):
        ctx = self.loop_context()
        result = registry().call(
            "return_to_observation", {"observation_id": self.CURRENT_FRAME}, ctx
        )

        next_action = result.data["next_action"]
        self.assertEqual(next_action["tool"], "ground_target")
        self.assertEqual(next_action["args"]["observation_id"], self.CURRENT_FRAME)
        self.assertIn("approach_candidate", result.data["hint"])
        # the advice is not "call this again"
        self.assertNotEqual(
            {"tool": "return_to_observation", "args": {"observation_id": self.CURRENT_FRAME}},
            next_action,
        )
        # and it is really callable: the tool the hint names exists in the
        # catalogue the model is handed (navigation + perception tools)
        from vlm_nav.tool_impl_perception import register_perception_tools

        full = registry()
        register_perception_tools(full)
        self.assertIn(next_action["tool"], full.names())
        self.assertIn("approach_candidate", full.names())

    def test_repeating_the_dead_call_answers_cached_with_a_stop_hint(self):
        ctx = self.loop_context()
        reg = registry()
        reg.call("return_to_observation", {"observation_id": self.CURRENT_FRAME}, ctx)
        repeat = reg.call(
            "return_to_observation", {"observation_id": self.CURRENT_FRAME}, ctx
        )
        self.assertEqual(repeat.status, "cached")
        self.assertEqual(repeat.data["outcome"], "already_at_pose")
        self.assertIn("do not repeat", repeat.data["hint"])
        self.assertEqual(ctx.world.applied, [])

    def test_the_model_loop_ends_instead_of_spinning(self):
        """The whole loop through AgentRuntime: no ok+arrived, no infinite spin."""
        from vlm_nav.runtime import AgentRuntime

        class LoopingVLM:
            """A model that keeps citing the frame it was just shown."""

            def __init__(self, observation_id):
                self.observation_id = observation_id
                self.calls = 0

            def decide_with_tools(self, frames, system_prompt, user_prompt,
                                  tools_text="", context=None):
                self.calls += 1
                if self.calls > 6:
                    return json.dumps({"tool": "finish", "args": {"reason": "done"}})
                return json.dumps(
                    {
                        "tool": "return_to_observation",
                        "args": {"observation_id": self.observation_id},
                    }
                )

        world = FakeWorld()
        runtime = AgentRuntime(
            registry(), world, None, LoopingVLM(self.CURRENT_FRAME), budget=20, max_turns=8
        )
        log = runtime.run("go back to where this picture was taken")

        movements = [
            record for record in log.turns
            if record.tool == "return_to_observation"
        ]
        self.assertTrue(movements)
        for record in movements:
            self.assertNotEqual(
                (record.result_status if hasattr(record, "result_status") else record.status,
                 record.data.get("outcome")),
                ("ok", "arrived"),
            )
        self.assertEqual(world.applied, [])
        self.assertLessEqual(log.turns[-1].turn, 8)


class TaskLifecycleTest(unittest.TestCase):
    """A movement tool is a continuous task: one segment per call, task_id resumes."""

    def test_repeated_calls_fly_the_task_to_arrival(self):
        world = FakeWorld()
        ctx = make_context(world)
        add_viewpoint(ctx, [20.0, 0.0, 0.0])
        reg = registry()
        first = reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual((first.data["task_status"], first.data["reason"]), ("running", "partial"))
        self.assertAlmostEqual(first.data["progress_m"], 8.0, places=6)
        self.assertEqual(first.data["segments"], 1)
        self.assertEqual(first.data["segments_left"], 11)

        second = reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1", "task_id": "nav_1"}, ctx)
        self.assertEqual(second.status, "ok")
        self.assertEqual(second.data["task_id"], "nav_1")
        self.assertEqual(second.data["task_status"], "running")
        self.assertAlmostEqual(second.data["progress_m"], 16.0, places=6)
        self.assertAlmostEqual(second.data["remaining_m"], 4.0, places=6)
        self.assertEqual(second.data["segments"], 2)

        third = reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1", "task_id": "nav_1"}, ctx)
        self.assertEqual(third.status, "ok")
        self.assertEqual(third.data["task_status"], "arrived")
        self.assertEqual(third.data["reason"], "arrived")
        self.assertAlmostEqual(third.data["progress_m"], 20.0, places=6)
        self.assertAlmostEqual(third.data["remaining_m"], 0.0, places=6)
        self.assertEqual(third.data["segments"], 3)
        self.assertEqual(len(world.applied), 3)
        self.assertEqual(
            [e["event"] for e in ctx.extra["nav_tasks"]["nav_1"]["events"]],
            ["arrived"],
        )

        # a terminal task is never flown again: the replay is cached + hinted
        replay = reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1", "task_id": "nav_1"}, ctx)
        self.assertEqual(replay.status, "cached")
        self.assertEqual(replay.data["task_status"], "arrived")
        self.assertIn("task already finished", replay.data["hint"])
        self.assertEqual(len(world.applied), 3)

    def test_running_task_is_continued_when_the_same_target_is_requested_again(self):
        world = FakeWorld()
        ctx = make_context(world)
        add_viewpoint(ctx, [20.0, 0.0, 0.0])
        reg = registry()
        reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        again = reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(again.status, "ok")
        self.assertEqual(again.data["task_id"], "nav_1")
        self.assertIn("continued running task nav_1", again.data["note"])
        self.assertEqual(sorted(ctx.extra["nav_tasks"]), ["nav_1"])
        self.assertEqual(again.data["segments"], 2)

    def test_continuing_a_task_ignores_an_id_passed_in_the_same_call(self):
        ctx = make_context(FakeWorld())
        add_viewpoint(ctx, [20.0, 0.0, 0.0])
        add_viewpoint(ctx, [0.0, 20.0, 0.0], viewpoint_id="vp_2")
        reg = registry()
        reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        continued = reg.call(
            "go_to_viewpoint", {"task_id": "nav_1", "viewpoint_id": "vp_2"}, ctx
        )
        self.assertEqual(continued.data["task_id"], "nav_1")
        self.assertEqual(continued.data["target_position"], [20.0, 0.0, 0.0])
        self.assertIn("id of this call is ignored", continued.data["note"])

    def test_repeating_a_terminal_request_at_the_same_pose_is_cached(self):
        wall = [[x * 0.1, 0.0, 0.0] for x in range(10, 20)]
        world = FakeWorld(points=wall)
        ctx = make_context(world)
        add_viewpoint(ctx, [3.0, 0.0, 0.0])
        reg = registry()
        first = reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(first.status, "blocked")
        repeat = reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(repeat.status, "cached")
        self.assertEqual(repeat.data["reason"], "obstacle_on_path")
        self.assertGreaterEqual(len(repeat.data["candidate_viewpoints"]), 1)
        self.assertIn("identical call at the same pose", repeat.data["hint"])
        self.assertEqual(world.applied, [])
        self.assertEqual(sorted(ctx.extra["nav_tasks"]), ["nav_1"])

    def test_an_arrived_target_requested_from_elsewhere_starts_a_new_task(self):
        ctx = make_context(FakeWorld())
        add_viewpoint(ctx, [3.0, 0.0, 0.0])
        reg = registry()
        first = reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(first.data["task_status"], "arrived")

        ctx.world.position = np.asarray([6.0, 0.0, 0.0])  # another module moved it
        back = reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(back.data["task_id"], "nav_2")
        self.assertEqual(back.data["task_status"], "arrived")
        self.assertAlmostEqual(float(ctx.world.position[0]), 3.0, places=6)
        self.assertEqual(sorted(ctx.extra["nav_tasks"]), ["nav_1", "nav_2"])


class TaskStatusTest(unittest.TestCase):
    def test_reports_progress_and_events_and_caches_a_repeated_query(self):
        world = FakeWorld()
        ctx = make_context(world)
        add_viewpoint(ctx, [20.0, 0.0, 0.0])
        reg = registry()
        reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)

        status = reg.call("task_status", {"task_id": "nav_1"}, ctx)
        self.assertEqual(status.status, "ok")
        data = status.data
        self.assertEqual(data["task_id"], "nav_1")
        self.assertEqual(data["status"], "running")
        self.assertEqual(data["intent"], "go_to_viewpoint")
        self.assertTrue(data["active"])
        self.assertAlmostEqual(data["progress_m"], 8.0, places=6)
        self.assertAlmostEqual(data["remaining_m"], 12.0, places=6)
        self.assertEqual(data["segments"], 1)
        self.assertEqual(data["segments_left"], 11)
        self.assertAlmostEqual(data["timeout_s"], 120.0, places=6)
        self.assertEqual(data["event_count"], 0)
        self.assertEqual(data["events"], [])
        self.assertIsNone(data["latest_event"])
        self.assertIn("task running", data["hint"])
        self.assertIn("go_to_viewpoint", data["hint"])
        kinds = {e.kind for e in status.evidence}
        self.assertIn("observation", kinds)
        self.assertIn("trajectory", kinds)
        self.assertIn("trajectory:nav_1", [f"{e.kind}:{e.ref}" for e in status.evidence])
        json.dumps(status.to_dict())

        # repeated query, no state change: cached + hint instead of the payload
        repeat = reg.call("task_status", {"task_id": "nav_1"}, ctx)
        self.assertEqual(repeat.status, "cached")
        self.assertEqual(repeat.data["status"], "running")
        self.assertIn("no change since the last task_status query", repeat.data["hint"])

        # the next segment invalidates the cached answer
        reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1", "task_id": "nav_1"}, ctx)
        fresh = reg.call("task_status", {"task_id": "nav_1"}, ctx)
        self.assertEqual(fresh.status, "ok")
        self.assertEqual(fresh.data["segments"], 2)
        self.assertAlmostEqual(fresh.data["remaining_m"], 4.0, places=6)
        self.assertEqual(fresh.data["query_count"], 2)

    def test_events_are_reported_with_the_task(self):
        wall = [[x * 0.1, 0.0, 0.0] for x in range(10, 20)]
        ctx = make_context(FakeWorld(points=wall))
        add_viewpoint(ctx, [3.0, 0.0, 0.0])
        reg = registry()
        reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)

        status = reg.call("task_status", {"task_id": "nav_1"}, ctx)
        self.assertEqual(status.data["status"], "blocked")
        self.assertEqual(status.data["event_count"], 1)
        event = status.data["events"][0]
        self.assertEqual(event["event"], "path_blocked")
        self.assertEqual(event["task_id"], "nav_1")
        self.assertEqual(event["status"], "blocked")
        self.assertEqual(event["reason"], "obstacle_on_path")
        self.assertAlmostEqual(event["progress_m"], 0.0, places=6)
        self.assertGreaterEqual(len(event["candidate_viewpoints"]), 1)
        self.assertEqual(status.data["latest_event"], event)
        self.assertIn("blocked", status.data["hint"])

        limited = reg.call("task_status", {"task_id": "nav_1", "limit": 1}, ctx)
        self.assertEqual(limited.status, "ok")
        self.assertEqual(len(limited.data["events"]), 1)

    def test_the_already_at_pose_task_explains_itself(self):
        ctx = make_context(FakeWorld())
        add_viewpoint(ctx, [0.0, 0.0, 0.0])
        reg = registry()
        started = reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertEqual(started.data["outcome"], "already_at_pose")
        status = reg.call("task_status", {"task_id": started.data["task_id"]}, ctx)
        self.assertEqual(status.data["reason"], "already_at_pose")
        self.assertIn("already_at_pose", status.data["hint"])

    def test_a_pose_change_invalidates_the_cached_query(self):
        world = FakeWorld()
        ctx = make_context(world)
        add_viewpoint(ctx, [20.0, 0.0, 0.0])
        reg = registry()
        reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        reg.call("task_status", {"task_id": "nav_1"}, ctx)

        world.position = np.asarray([5.0, 1.0, 0.0])  # e.g. another module moved the drone
        fresh = reg.call("task_status", {"task_id": "nav_1"}, ctx)
        self.assertEqual(fresh.status, "ok")
        self.assertEqual(fresh.data["pose"], [5.0, 1.0, 0.0])

    def test_without_task_id_lists_every_task(self):
        ctx = make_context(FakeWorld())
        add_viewpoint(ctx, [20.0, 0.0, 0.0])
        add_viewpoint(ctx, [0.0, 20.0, 0.0], viewpoint_id="vp_2")
        reg = registry()
        empty = reg.call("task_status", {}, ctx)
        self.assertEqual(empty.status, "empty")
        self.assertEqual(empty.data["count"], 0)

        reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        reg.call("go_to_viewpoint", {"viewpoint_id": "vp_2"}, ctx)
        listing = reg.call("task_status", {}, ctx)
        self.assertEqual(listing.status, "ok")
        self.assertEqual(listing.data["count"], 2)
        self.assertEqual(
            [task["task_id"] for task in listing.data["tasks"]], ["nav_1", "nav_2"]
        )
        self.assertEqual(listing.data["active"], ["nav_1", "nav_2"])
        self.assertEqual(listing.data["tasks"][0]["status"], "running")
        self.assertEqual(listing.data["tasks"][0]["intent"], "go_to_viewpoint")

        repeat = reg.call("task_status", {}, ctx)
        self.assertEqual(repeat.status, "cached")
        self.assertIn("no navigation state change", repeat.data["hint"])

    def test_unknown_task_fails_cleanly(self):
        ctx = make_context(FakeWorld())
        result = registry().call("task_status", {"task_id": "nav_77"}, ctx)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["reason"], "unknown_task")
        self.assertEqual(result.data["known_tasks"], [])


class TaskCancelTest(unittest.TestCase):
    def test_cancel_marks_the_task_queues_an_event_and_stops_continuation(self):
        world = FakeWorld()
        ctx = make_context(world)
        add_viewpoint(ctx, [20.0, 0.0, 0.0])
        reg = registry()
        reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)

        cancelled = reg.call(
            "cancel_task", {"task_id": "nav_1", "reason": "new evidence"}, ctx
        )
        self.assertEqual(cancelled.status, "ok")
        self.assertEqual(cancelled.data["status"], "cancelled")
        self.assertEqual(cancelled.data["reason"], "new evidence")
        self.assertAlmostEqual(cancelled.data["progress_m"], 8.0, places=6)
        self.assertIn("cancelled", cancelled.data["hint"])
        task = ctx.extra["nav_tasks"]["nav_1"]
        self.assertEqual(task["status"], "cancelled")
        self.assertEqual([e["event"] for e in task["events"]], ["task_cancelled"])
        self.assertEqual(task["events"][0]["reason"], "new evidence")

        # a cancelled task cannot be flown on: cached replay with the reason
        refused = reg.call(
            "go_to_viewpoint", {"viewpoint_id": "vp_1", "task_id": "nav_1"}, ctx
        )
        self.assertEqual(refused.status, "cached")
        self.assertEqual(refused.data["task_status"], "cancelled")
        self.assertIn("cancelled", refused.data["hint"])
        self.assertEqual(len(world.applied), 1)

        again = reg.call("cancel_task", {"task_id": "nav_1"}, ctx)
        self.assertEqual(again.status, "cached")
        self.assertIn("already cancelled", again.data["hint"])

    def test_cancel_a_finished_task_is_cached(self):
        ctx = make_context(FakeWorld())
        add_viewpoint(ctx, [3.0, 0.0, 0.0])
        reg = registry()
        reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        result = reg.call("cancel_task", {"task_id": "nav_1"}, ctx)
        self.assertEqual(result.status, "cached")
        self.assertEqual(result.data["status"], "arrived")
        self.assertIn("already arrived", result.data["hint"])

    def test_missing_and_unknown_task_ids_fail_cleanly(self):
        reg = registry()
        ctx = make_context(FakeWorld())
        missing = reg.call("cancel_task", {}, ctx)
        self.assertEqual((missing.status, missing.data["reason"]), ("failed", "missing_task_id"))
        unknown = reg.call("cancel_task", {"task_id": "nav_9"}, ctx)
        self.assertEqual((unknown.status, unknown.data["reason"]), ("failed", "unknown_task"))
        self.assertEqual(unknown.data["known_tasks"], [])


class TaskEventTest(unittest.TestCase):
    """Event shape, the runtime drain mirror, and the event triggers."""

    def test_events_carry_the_flat_shapes_the_drain_contract_needs(self):
        wall = [[x * 0.1, 0.0, 0.0] for x in range(10, 20)]
        ctx = make_context(FakeWorld(points=wall))
        add_viewpoint(ctx, [3.0, 0.0, 0.0])
        registry().call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)

        queued = ctx.extra["events"]
        self.assertEqual(len(queued), 1)
        item = queued[-1]
        # AgentRuntime drains {"name": ..., "data": {...}} and re-emits it
        self.assertEqual(item["name"], "path_blocked")
        self.assertIsInstance(item["data"], dict)
        self.assertEqual(item["data"]["event"], "path_blocked")
        # goal.md section 5 flat shape lives on the same record
        self.assertEqual(item["event"], "path_blocked")
        self.assertEqual(item["task_id"], "nav_1")
        self.assertEqual(item["status"], "blocked")
        self.assertEqual(item["reason"], "obstacle_on_path")
        self.assertIsInstance(item["progress_m"], float)
        self.assertIsInstance(item["remaining_m"], float)
        json.dumps(item, default=str)

        # the durable, sequence-ordered log holds the same record
        log = ctx.extra["nav_events"]
        self.assertEqual([event["event"] for event in log], ["path_blocked"])
        self.assertEqual(log[0]["seq"], 1)
        self.assertEqual(log[0]["seq"], item["data"]["seq"])

    def test_map_changed_when_the_observed_map_grows(self):
        world = FakeWorld()
        ctx = make_context(world)
        add_viewpoint(ctx, [10.0, 0.0, 0.0])
        reg = registry()
        reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        # 80 new surface points, far enough from the leg not to block it
        world._sense_points = np.asarray(
            [[x * 0.5, y, -1.5] for x in range(-20, 20) for y in (-10.0, -9.5)],
            dtype=np.float64,
        )
        second = reg.call(
            "go_to_viewpoint", {"viewpoint_id": "vp_1", "task_id": "nav_1"}, ctx
        )
        self.assertEqual(second.data["task_status"], "arrived")

        names = [event["event"] for event in ctx.extra["nav_events"]]
        self.assertEqual(names, ["map_changed", "arrived"])
        event = ctx.extra["nav_events"][0]
        self.assertEqual(event["points_before"], 0)
        self.assertEqual(event["points_after"], 80)
        self.assertEqual(event["map_version"], 1)
        self.assertEqual(ctx.extra["map_version"], 1)
        self.assertEqual(ctx.extra["nav_tasks"]["nav_1"]["map_points"], 80)

    def test_target_rediscovery_reports_the_transition_without_geometry(self):
        world = FakeWorld(in_fov=False)
        ctx = make_context(world)
        add_viewpoint(ctx, [10.0, 0.0, 0.0])
        reg = registry()
        reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertIs(ctx.extra["nav_tasks"]["nav_1"]["target_in_fov"], False)

        world.in_fov = True
        reg.call("go_to_viewpoint", {"viewpoint_id": "vp_1", "task_id": "nav_1"}, ctx)
        names = [event["event"] for event in ctx.extra["nav_events"]]
        self.assertEqual(names, ["target_rediscovered", "arrived"])
        event = ctx.extra["nav_events"][0]
        self.assertEqual(event["task_id"], "nav_1")
        self.assertTrue(event["observation_id"])
        self.assertIn("verify_goal", event["hint"])
        # ground truth stays behind verify_goal: no target geometry in the event
        for key in ("target_position", "distance_to_target_m", "target_distance_m"):
            self.assertNotIn(key, event)

    def test_runtime_drains_the_events_and_wakes_the_model(self):
        from vlm_nav.runtime import AgentRuntime

        class RecordingTools(ToolRegistry):
            def __init__(self):
                super().__init__()
                self.last_result = None

            def call(self, name, args, ctx):
                result = super().call(name, args, ctx)
                self.last_result = (name, result)
                return result

        class FollowViewpointVLM:
            """The model of the loop: propose, then fly to the id it was handed."""

            def __init__(self, tools, script):
                self.tools = tools
                self.script = list(script)
                self.seen = []

            def decide_with_tools(self, frames, system_prompt, user_prompt,
                                  tools_text="", context=None):
                self.seen.append(user_prompt)
                name, result = self.tools.last_result or (None, None)
                if self.script and self.script[0] == "fly_viewpoint":
                    self.script.pop(0)
                    viewpoint_id = (result.data.get("viewpoint_id") if result else None) or "vp_1"
                    return json.dumps(
                        {"tool": "go_to_viewpoint", "args": {"viewpoint_id": viewpoint_id}}
                    )
                item = self.script.pop(0) if self.script else {
                    "tool": "finish",
                    "args": {"reason": "done"},
                }
                return json.dumps(item)

        # a ring of known surface all around the drone: every proposed viewpoint
        # is behind geometry, so the flight is blocked at the pre-check
        ring = [
            [1.6 * math.cos(t), 1.6 * math.sin(t), 0.0]
            for t in np.linspace(0.0, 2.0 * math.pi, 96)
        ]
        world = FakeWorld(points=ring)
        tools = RecordingTools()
        register_navigation_tools(tools)
        vlm = FollowViewpointVLM(
            tools,
            [
                {"tool": "propose_viewpoints", "args": {}},
                "fly_viewpoint",
                {"tool": "finish", "args": {"reason": "blocked"}},
            ],
        )
        runtime = AgentRuntime(tools, world, None, vlm, budget=8, max_turns=4)
        log = runtime.run("reach the target")

        self.assertEqual(log.finish_reason, "finish")
        self.assertEqual(
            [record.tool for record in log.turns],
            ["propose_viewpoints", "go_to_viewpoint", "finish"],
        )
        self.assertEqual([event["name"] for event in log.events], ["path_blocked"])
        self.assertEqual(log.events[0]["data"]["task_id"], "nav_1")
        # the event is rendered into the next turn's prompt: the model wakes
        self.assertNotIn("path_blocked", vlm.seen[1])
        self.assertIn("path_blocked", vlm.seen[2])
        self.assertIn("nav_1", vlm.seen[2])


class TaskBudgetTest(unittest.TestCase):
    def test_timeout_fails_the_task_on_the_next_segment(self):
        world = FakeWorld()
        ctx = make_context(world)
        add_viewpoint(ctx, [20.0, 0.0, 0.0])
        reg = registry()
        clock = {"now": 1000.0}
        with mock.patch.object(navigation_module, "_now", lambda: clock["now"]):
            first = reg.call(
                "go_to_viewpoint", {"viewpoint_id": "vp_1", "timeout_s": 10.0}, ctx
            )
            self.assertEqual(first.data["task_status"], "running")
            clock["now"] = 1020.0  # past the deadline
            timed_out = reg.call(
                "go_to_viewpoint", {"viewpoint_id": "vp_1", "task_id": "nav_1"}, ctx
            )

        self.assertEqual(timed_out.status, "failed")
        self.assertEqual(timed_out.data["task_status"], "failed")
        self.assertEqual(timed_out.data["reason"], "timeout")
        self.assertEqual(timed_out.data["outcome"], "timeout")
        self.assertIn("timeout_s", timed_out.data["hint"])
        task = ctx.extra["nav_tasks"]["nav_1"]
        self.assertEqual(task["status"], "failed")
        self.assertEqual([e["event"] for e in task["events"]], ["task_failed"])
        self.assertEqual(task["events"][0]["reason"], "timeout")
        self.assertEqual(len(world.applied), 1)  # the expired task never flew

        replay = reg.call(
            "go_to_viewpoint", {"viewpoint_id": "vp_1", "task_id": "nav_1"}, ctx
        )
        self.assertEqual(replay.status, "cached")
        self.assertEqual(replay.data["task_status"], "failed")

    def test_segment_budget_fails_the_task(self):
        world = FakeWorld()
        ctx = make_context(world)
        add_viewpoint(ctx, [20.0, 0.0, 0.0])
        reg = registry()
        first = reg.call(
            "go_to_viewpoint", {"viewpoint_id": "vp_1", "max_segments": 1}, ctx
        )
        self.assertEqual(first.data["task_status"], "running")
        self.assertEqual(first.data["segments_left"], 0)

        exhausted = reg.call(
            "go_to_viewpoint", {"viewpoint_id": "vp_1", "task_id": "nav_1"}, ctx
        )
        self.assertEqual(exhausted.status, "failed")
        self.assertEqual(exhausted.data["task_status"], "failed")
        self.assertEqual(exhausted.data["reason"], "segment_budget")
        task = ctx.extra["nav_tasks"]["nav_1"]
        self.assertEqual(task["status"], "failed")
        self.assertEqual(task["events"][0]["event"], "task_failed")
        self.assertEqual(task["events"][0]["max_segments"], 1)
        self.assertEqual(len(world.applied), 1)

    def test_the_approach_spelling_of_the_segment_budget_works(self):
        ctx = make_context(FakeWorld())
        store_candidate(ctx, "cand_1", position=[30.0, 0.0, 0.0])
        result = registry().call(
            "approach_candidate", {"candidate_id": "cand_1", "max_legs": 1}, ctx
        )
        self.assertEqual(result.data["task_status"], "running")
        self.assertEqual(result.data["segments_left"], 0)

    def test_context_budget_knobs_are_the_defaults(self):
        ctx = make_context(
            FakeWorld(), extra={"nav_timeout_s": 5.0, "nav_max_segments": 2}
        )
        add_viewpoint(ctx, [20.0, 0.0, 0.0])
        result = registry().call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        self.assertAlmostEqual(result.data["timeout_s"], 5.0, places=6)
        self.assertEqual(result.data["segments_left"], 1)


class VerifyGoalTest(unittest.TestCase):
    def test_verified_when_pointed_point_sits_on_the_goal(self):
        # a declared camera: the pointed pixel is only a position with one
        world = FakeWorld(
            position=(1.0, 0.0, 0.0), target=(2.4, 0.0, 0.0), depth=1.4, intrinsics=True
        )
        ctx = make_context(world)
        result = registry().call("verify_goal", {"pixel": [500.0, 500.0]}, ctx)
        self.assertEqual(result.status, "ok")
        data = result.data
        self.assertIs(data["verified"], True)
        self.assertEqual(data["unmet"], [])
        self.assertEqual(data["pixel"], [500.0, 500.0])
        self.assertAlmostEqual(data["checks"]["depth_m"], 1.4, places=6)
        self.assertAlmostEqual(data["checks"]["point_to_target_m"], 0.0, places=3)
        self.assertAlmostEqual(data["checks"]["distance_to_target_m"], 1.4, places=6)
        self.assertTrue(data["checks"]["target_in_fov"])
        kinds = {item["kind"] for item in data["evidence"]}
        self.assertIn("observation", kinds)
        self.assertIn("map", kinds)
        json.dumps(result.to_dict())

    def test_neither_the_return_nor_its_evidence_quotes_the_goal_pose(self):
        """The completion check may not hand the model the answer.

        It used to cite ``{"kind": "map", "ref": "target", "note": "blob at
        [7.0, 8.0, 9.0] r=0.50 m"}``: the true goal position travelled back in
        the very result that is supposed to *check* the goal, so the visual
        evidence the episode exists to test became optional (G1 §6). A verdict
        reports its own checks (FOV, distance, the measured point) and names the
        layer it was taken against - never goal geometry.
        """
        world = FakeWorld(position=(0.0, 0.0, 0.0), target=(7.0, 8.0, 9.0))
        result = registry().call("verify_goal", {"reason": "confirm"}, make_context(world))

        for channel in (json.dumps(result.to_dict(), default=str), result.to_prompt()):
            for token in ("[7.0, 8.0, 9.0]", "7.00", "8.00", "9.00"):
                self.assertNotIn(token, channel)
        # the layer the check ran against is still citable ...
        self.assertIn("map", {item["kind"] for item in result.data["evidence"]})
        # ... and the check's own numbers (a sensor-readable distance) survive
        self.assertAlmostEqual(
            result.data["checks"]["distance_to_target_m"], 13.93, places=2
        )

    def test_the_goal_pose_never_reaches_the_next_prompt(self):
        """Same lock at the channel the model really reads: the turn prompt."""
        from vlm_nav.runtime import AgentRuntime

        class VerifyThenFinishVLM:
            name = "verify-then-finish"

            def __init__(self):
                self.seen: list[str] = []

            def decide_with_tools(self, frames, system_prompt, user_prompt,
                                  tools_text="", context=None):
                self.seen.append(user_prompt)
                if len(self.seen) == 1:
                    return json.dumps(
                        {"tool": "verify_goal", "args": {"reason": "confirm"}}
                    )
                return json.dumps({"tool": "finish", "args": {"reason": "done"}})

        world = FakeWorld(position=(0.0, 0.0, 0.0), target=(7.0, 8.0, 9.0))
        vlm = VerifyThenFinishVLM()
        runtime = AgentRuntime(registry(), world, None, vlm, budget=6, max_turns=3)
        log = runtime.run("reach the goal")
        self.assertEqual([turn.tool for turn in log.turns], ["verify_goal", "finish"])

        prompt = vlm.seen[1]
        for token in ("[7.0, 8.0, 9.0]", "7.00", "8.00", "9.00"):
            self.assertNotIn(token, prompt)
        # the prompt still carries the honest verdict of the call
        self.assertIn("verified=False", prompt)

    def test_no_navigation_answer_quotes_the_goal_pose(self):
        """Registry-wide lock: the goal's pose is not an answer any tool gives.

        The leak G1 found sat in *one* return (``verify_goal``'s map evidence);
        this scans every registered navigation tool on a world whose goal sits
        at a position no call has any reason to report, so a later tool that
        echoes ``world.target`` fails here instead of quietly turning the visual
        evidence into decoration.
        """
        goal = (7.0, 8.0, 9.0)
        ctx = make_context(FakeWorld(position=(0.0, 0.0, 0.0), target=goal))
        reg = registry()
        calls = {
            "verify_goal": {"reason": "confirm"},
            "propose_viewpoints": {},
            "check_path": {},
            "task_status": {},
            "cancel_task": {},
            "navigate": {},
            "approach_candidate": {},
            "go_to_viewpoint": {},
            "return_to_observation": {},
            "finish": {"reason": "done"},
            # the read-only batch: its inner call answers, the wrapper must not
            # introduce a channel either
            "batch_read": {"calls": [{"tool": "check_path", "args": {}}]},
        }
        for name in TOOL_NAMES:
            result = reg.call(name, calls[name], ctx)
            for channel in (json.dumps(result.to_dict(), default=str), result.to_prompt()):
                with self.subTest(tool=name, status=result.status):
                    self.assertNotIn("[7.0, 8.0, 9.0]", channel)
                    self.assertNotIn("7.00", channel)

    def test_depth_beats_any_fabricated_range(self):
        world = FakeWorld(
            position=(1.0, 0.0, 0.0), target=(5.0, 0.0, 0.0), depth=4.0, intrinsics=True
        )
        ctx = make_context(world, extra={"done_dist": 10.0})
        result = registry().call("verify_goal", {"pixel": [500.0, 500.0]}, ctx)
        self.assertAlmostEqual(result.data["checks"]["point_world"][0], 5.0, places=3)

    def test_uncertain_without_depth(self):
        world = FakeWorld(position=(1.0, 0.0, 0.0), target=(2.4, 0.0, 0.0), depth=None)
        ctx = make_context(world)
        result = registry().call("verify_goal", {"pixel": [500.0, 500.0]}, ctx)
        self.assertEqual(result.data["verified"], "uncertain")
        self.assertEqual(result.data["reason"], "no_depth_estimate")
        self.assertIn(
            {"check": "depth", "reason": "no_depth_estimate"}, result.data["unmet"]
        )

    def test_false_when_target_is_too_far(self):
        world = FakeWorld(position=(0.0, 0.0, 0.0), target=(5.0, 0.0, 0.0), depth=5.0)
        ctx = make_context(world)
        result = registry().call("verify_goal", {"pixel": [500.0, 500.0]}, ctx)
        self.assertIs(result.data["verified"], False)
        self.assertEqual(result.data["reason"], "unmet_conditions")
        checks = {item["check"] for item in result.data["unmet"]}
        self.assertIn("distance_to_target_m", checks)

    def test_false_when_target_is_out_of_view(self):
        world = FakeWorld(position=(1.0, 0.0, 0.0), target=(2.4, 0.0, 0.0),
                          depth=1.4, in_fov=False)
        ctx = make_context(world)
        result = registry().call("verify_goal", {}, ctx)
        self.assertIs(result.data["verified"], False)
        self.assertIn("target_in_fov", {item["check"] for item in result.data["unmet"]})

    def test_a_1_32_m_standoff_is_not_verified_against_a_1_2_m_gate(self):
        """The runner's gate is *the* gate: 1.32 m is outside 1.2 m.

        The decision-loop v3 run claimed exactly this standoff: ``verify_goal``
        fell back to its own ``max(1.5, radius + 0.5)`` default while the
        episodes were evaluated against ``--success-dist + radius`` (1.2 m), so
        four arms "verified" a goal they had not reached (book_near, seed 11).
        """
        world = FakeWorld(position=(0.68, 0.0, 0.0), target=(2.0, 0.0, 0.0),
                          radius_m=0.3)
        ctx = make_context(world, extra={"done_dist": 1.2})
        result = registry().call("verify_goal", {"reason": "confirm"}, ctx)
        self.assertIs(result.data["verified"], False)
        self.assertEqual(result.data["reason"], "unmet_conditions")
        unmet = next(item for item in result.data["unmet"]
                     if item["check"] == "distance_to_target_m")
        self.assertEqual(1.2, unmet["max_m"])
        self.assertAlmostEqual(1.32, unmet["value_m"], places=3)

    def test_the_check_reports_the_limit_it_used(self):
        world = FakeWorld(position=(1.0, 0.0, 0.0), target=(2.4, 0.0, 0.0),
                          radius_m=0.3)
        inside = make_context(world, extra={"done_dist": 1.5})
        result = registry().call("verify_goal", {"reason": "confirm"}, inside)
        self.assertIs(result.data["verified"], True)
        self.assertAlmostEqual(1.5, result.data["checks"]["distance_limit_m"], places=6)
        outside = make_context(world, extra={"done_dist": 1.2})
        result = registry().call("verify_goal", {"reason": "confirm"}, outside)
        self.assertIs(result.data["verified"], False)
        self.assertAlmostEqual(1.2, result.data["checks"]["distance_limit_m"], places=6)

    def test_region_maps_to_a_normalized_pixel(self):
        world = FakeWorld(position=(1.0, 0.0, 0.0), target=(2.4, 0.0, 0.0), depth=1.4)
        ctx = make_context(world)
        result = registry().call("verify_goal", {"region": "center"}, ctx)
        self.assertEqual(result.data["pixel"], [500.0, 500.0])
        self.assertIs(result.data["verified"], True)
        bad = registry().call("verify_goal", {"region": "nowhere"}, ctx)
        self.assertEqual((bad.status, bad.data["reason"]), ("failed", "unknown_region:nowhere"))

    def test_candidate_identity_match_and_mismatch(self):
        world = FakeWorld(
            position=(1.0, 0.0, 0.0), target=(2.4, 0.0, 0.0), depth=1.4, intrinsics=True
        )
        ctx = make_context(world, extra={"candidate_positions": {"cand_7": [2.4, 0.0, 0.0]}})
        matched = registry().call(
            "verify_goal", {"pixel": [500.0, 500.0], "candidate_id": "cand_7"}, ctx
        )
        self.assertIs(matched.data["verified"], True)
        self.assertAlmostEqual(matched.data["checks"]["candidate_offset_m"], 0.0, places=3)
        self.assertIn("cand_7", [item["ref"] for item in matched.data["evidence"]])

        ctx = make_context(world, extra={"candidate_positions": {"cand_7": [9.0, 0.0, 0.0]}})
        mismatched = registry().call(
            "verify_goal", {"pixel": [500.0, 500.0], "candidate_id": "cand_7"}, ctx
        )
        self.assertIs(mismatched.data["verified"], False)
        self.assertIn(
            "candidate_identity", {item["check"] for item in mismatched.data["unmet"]}
        )

    def test_candidate_resolves_through_memory_tracker(self):
        class Memory:
            def __init__(self):
                self.candidates = []

        class FakeCandidate:
            id = 3
            position_flu = (2.4, 0.0, 0.0)

        memory = Memory()
        memory.candidates = [FakeCandidate()]
        world = FakeWorld(position=(1.0, 0.0, 0.0), target=(2.4, 0.0, 0.0), depth=1.4)
        ctx = make_context(world, memory=memory)
        result = registry().call(
            "verify_goal", {"pixel": [500.0, 500.0], "candidate_id": 3}, ctx
        )
        self.assertIs(result.data["verified"], True)

    def test_hypothesis_store_entries_resolve_positions(self):
        # vlm_nav.memory.HypothesisStore.get(id) -> {"id": ..., "data": {...}}
        class Store:
            def get(self, key):
                if str(key) != "hyp_2":
                    return None
                return {"id": "hyp_2", "data": {"position": [0.0, 5.0, 0.0]}}

        ctx = make_context(FakeWorld(), memory=Store())
        proposed = registry().call(
            "propose_viewpoints", {"target_hypothesis": "hyp_2"}, ctx
        )
        self.assertTrue(proposed.data["hypothesis_resolved"])
        self.assertEqual(proposed.data["base_bearing_deg"], 90.0)
        checked = registry().call("check_path", {"target_id": "hyp_2"}, ctx)
        self.assertEqual(checked.data["target_position"], [0.0, 5.0, 0.0])

    def test_unknown_candidate_is_reported_without_crashing(self):
        world = FakeWorld(position=(1.0, 0.0, 0.0), target=(2.4, 0.0, 0.0), depth=1.4)
        ctx = make_context(world)
        result = registry().call(
            "verify_goal", {"pixel": [500.0, 500.0], "candidate_id": "cand_99"}, ctx
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(
            result.data["unmet"],
            [
                {
                    "check": "candidate_identity",
                    "reason": "unknown_candidate",
                    "candidate_id": "cand_99",
                }
            ],
        )

    def test_no_evidence_is_uncertain(self):
        class BareWorld:
            position_flu = (0.0, 0.0, 0.0)
            yaw_rad = 0.0

        ctx = make_context(BareWorld())
        result = registry().call("verify_goal", {}, ctx)
        self.assertEqual(result.data["verified"], "uncertain")
        self.assertEqual(result.data["reason"], "no_evidence")


def env_style_declaration(world, *, floor_z=None, bottom_z=None, size_z=None):
    """Declare the env-owned heights the way the production env carries them.

    ``ObstacleAvoidanceGaussianSplatting`` keeps the navigable floor in
    ``navigation_min_flu`` and the semantic target's box in the per-env tensors
    ``target_object_bbox_min_z_flu`` / ``target_object_bbox_size_z_flu`` (the
    pair ``env`` itself feeds to ``semantic_goal_height``); this is that
    declaration on a fake world.
    """
    env = SimpleNamespace()
    if floor_z is not None:
        env.navigation_min_flu = np.asarray([0.0, 0.0, float(floor_z)])
    if bottom_z is not None and size_z is not None:
        env.target_object_bbox_min_z_flu = np.asarray([float(bottom_z)])
        env.target_object_bbox_size_z_flu = np.asarray([float(size_z)])
    world.env = env
    return world


class PointLocalizationIntrinsicsTest(unittest.TestCase):
    """A pointed pixel is projected with the frame's *real* camera.

    ``frames_audit.md`` I1: the pixel -> world projection used to fall through
    to a hard-coded 320x180 @ 86 deg, so in a 60 deg world every lateral offset
    came out tan(43)/tan(30) = 1.615x too wide (0.355 m at 2 m). The tool layer
    must project with the intrinsics the world/frame declares, and with none
    declared it must report that instead of assuming a field of view.
    """

    def test_verify_goal_projects_with_the_declared_camera(self):
        world = FakeWorld(
            position=(0.0, 0.0, 0.0), target=(5.0, 0.0, 0.0), radius_m=1.0,
            depth=2.0, intrinsics=True, width=64, height=64, hfov_deg=60.0,
        )
        ctx = make_context(world, extra={"done_dist": 10.0})
        result = registry().call("verify_goal", {"pixel": [750.0, 500.0]}, ctx)
        point = result.data["checks"]["point_world"]  # reported to 5 mm
        fx = 64.0 / (2.0 * math.tan(math.radians(30.0)))
        self.assertAlmostEqual(point[0], 2.0, places=3)  # axial depth, 60 deg or not
        self.assertAlmostEqual(point[1], -(16.0 / fx) * 2.0, places=3)  # 0.75*64 = 48 px
        # the assumption that used to be baked in is measurably different
        default_fx = 320.0 / (2.0 * math.tan(math.radians(43.0)))
        default_y = -(80.0 / default_fx) * 2.0
        self.assertGreater(abs(default_y / point[1]), 1.6)  # the audited 1.615x
        self.assertNotAlmostEqual(point[1], default_y, places=2)

    def test_verify_goal_reports_missing_intrinsics_instead_of_guessing(self):
        world = FakeWorld(position=(1.0, 0.0, 0.0), target=(2.4, 0.0, 0.0), depth=1.4)
        ctx = make_context(world)
        result = registry().call("verify_goal", {"pixel": [750.0, 500.0]}, ctx)
        # the depth is a real measurement and is reported as such ...
        self.assertAlmostEqual(result.data["checks"]["depth_m"], 1.4, places=6)
        # ... but the pixel cannot be turned into a position without a camera
        self.assertNotIn("point_world", result.data["checks"])
        self.assertEqual(
            result.data["checks"]["point_localization"], "intrinsics_unknown"
        )


class LowAltitudeProfileTest(unittest.TestCase):
    """The low-altitude flight profile of ``altitude_audit.md`` 5.1.

    Cruise altitudes come from the ground the drone has actually seen
    (``floor + 0.6 ... floor + 1.2``) and from the target's own box, so a ring
    viewpoint is never proposed above ``floor + 1.6`` and an approach ends
    *beside* the object instead of hovering over it.
    """

    FLOOR_Z = -2.44

    def floor_world(self, **kwargs):
        points = [
            [float(x), float(y), self.FLOOR_Z]
            for x in range(-6, 7)
            for y in (-6.0, -3.0, 0.0, 3.0, 6.0)
        ]
        kwargs.setdefault("position", (0.0, 0.0, 1.0))
        return env_style_declaration(
            FakeWorld(points=points, **kwargs), floor_z=self.FLOOR_Z
        )

    def test_viewpoint_ring_stays_inside_the_cruise_window(self):
        ctx = make_context(self.floor_world())
        result = registry().call("propose_viewpoints", {}, ctx)
        self.assertEqual(result.status, "ok")
        zs = [item["position"][2] for item in result.data["viewpoints"]]
        self.assertEqual(len(zs), 5)
        for z in zs:
            self.assertGreaterEqual(z, self.FLOOR_Z + 0.6 - 1e-9)
            self.assertLessEqual(z, self.FLOOR_Z + 1.2 + 1e-9)
            self.assertLessEqual(z, self.FLOOR_Z + 1.6)  # never above the ceiling
        # the ring no longer inherits a start altitude sampled above the object
        self.assertNotAlmostEqual(zs[0], 1.0, places=3)

    def test_viewpoint_ring_uses_the_target_height_band(self):
        ctx = make_context(
            env_style_declaration(
                self.floor_world(target=(3.0, 0.0, -2.2)),
                floor_z=self.FLOOR_Z, bottom_z=-2.46, size_z=0.51,
            )
        )
        result = registry().call("propose_viewpoints", {}, ctx)
        # obj_top + 0.3 = -1.95 + 0.3, inside the cruise window
        for item in result.data["viewpoints"]:
            self.assertAlmostEqual(item["position"][2], -1.65, places=6)

    def test_the_env_cruise_window_is_honoured(self):
        world = self.floor_world()
        world.env.cruise_window_min_m = 0.2
        world.env.cruise_window_max_m = 0.5
        world.env.cruise_ceiling_m = 0.8
        result = registry().call("propose_viewpoints", {}, make_context(world))
        for item in result.data["viewpoints"]:
            self.assertGreaterEqual(item["position"][2], self.FLOOR_Z + 0.2 - 1e-9)
            self.assertLessEqual(item["position"][2], self.FLOOR_Z + 0.5 + 1e-9)

    def test_the_env_tensor_spelling_resolves_too(self):
        # the env keeps these as torch tensors (CUDA in production), not lists
        import torch

        world = FakeWorld(position=(0.0, 0.0, 1.0))
        world.env = SimpleNamespace(
            navigation_min_flu=torch.tensor([0.0, 0.0, self.FLOOR_Z]),
            target_object_bbox_min_z_flu=torch.tensor([-2.46]),
            target_object_bbox_size_z_flu=torch.tensor([0.51]),
        )
        result = registry().call("propose_viewpoints", {}, make_context(world))
        for item in result.data["viewpoints"]:
            self.assertAlmostEqual(item["position"][2], -1.65, places=6)

    def test_viewpoint_altitude_is_left_alone_without_a_floor(self):
        ctx = make_context(FakeWorld(position=(0.0, 0.0, 0.5)))
        result = registry().call("propose_viewpoints", {}, ctx)
        for item in result.data["viewpoints"]:
            self.assertAlmostEqual(item["position"][2], 0.5, places=6)

    def test_the_approach_goal_sits_beside_the_object_not_above_it(self):
        world = env_style_declaration(
            FakeWorld(position=(0.0, 0.0, 1.5), target=(5.0, 0.0, -1.5), radius_m=0.4),
            bottom_z=-2.0, size_z=1.0,
        )
        ctx = make_context(world)
        store_candidate(ctx, "cand_1", position=[5.0, 0.0, -1.5])
        result = registry().call("approach_candidate", {"candidate_id": "cand_1"}, ctx)
        goal_z = -2.0 + 0.35  # bbox bottom + min(0.6 * height, 0.35)
        self.assertAlmostEqual(result.data["waypoint"][2], goal_z, places=6)
        self.assertAlmostEqual(float(ctx.world.position[2]), goal_z, places=6)
        self.assertLess(float(ctx.world.position[2]), -1.0)  # not above the object
        self.assertEqual(result.data["outcome"], "arrived")

    def test_the_approach_goal_uses_the_measured_height_without_a_box(self):
        ctx = make_context(FakeWorld(position=(0.0, 0.0, 1.5)))
        store_candidate(ctx, "cand_1", position=[5.0, 0.0, -1.2])
        result = registry().call("approach_candidate", {"candidate_id": "cand_1"}, ctx)
        self.assertAlmostEqual(result.data["waypoint"][2], -1.2, places=6)
        self.assertAlmostEqual(float(ctx.world.position[2]), -1.2, places=6)

    def test_the_approach_goal_never_dives_below_the_cruise_floor(self):
        world = env_style_declaration(
            FakeWorld(position=(0.0, 0.0, 1.5), target=(5.0, 0.0, -2.0), radius_m=0.4),
            floor_z=-2.0, bottom_z=-3.0, size_z=0.4,
        )
        ctx = make_context(world)
        store_candidate(ctx, "cand_1", position=[5.0, 0.0, -2.0])
        result = registry().call("approach_candidate", {"candidate_id": "cand_1"}, ctx)
        # the box asks for -2.88 m; the floor's own cruise floor allows -1.4 m
        self.assertAlmostEqual(result.data["waypoint"][2], -1.4, places=6)
        self.assertAlmostEqual(float(ctx.world.position[2]), -1.4, places=6)


def _write_test_ply(path, gaussians):
    """gaussians: iterable of (xyz, sh_dc_rgb, opacity_logit, log_scale, rot_wxyz)."""
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(gaussians)}",
            "property float x",
            "property float y",
            "property float z",
            "property float f_dc_0",
            "property float f_dc_1",
            "property float f_dc_2",
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header",
        ]
    )
    rows = [list(xyz) + list(sh) + [opacity] + list(log_scale) + list(rot)
            for xyz, sh, opacity, log_scale, rot in gaussians]
    import struct

    with open(path, "wb") as handle:
        handle.write((header + "\n").encode("ascii"))
        for row in rows:
            handle.write(struct.pack("<14f", *row))


class StubGroundingVLM:
    """Minimal grounding backend for ``tool_impl_perception.ground_target``."""

    name = "stub"

    def __init__(self, reply):
        self.reply = reply

    def decide(self, frames, system_prompt, user_prompt, context=None):
        return self.reply


class GroundTargetChainTest(unittest.TestCase):
    """Regression for the chain a real model picks: ``ground_target`` ->
    ``approach_candidate`` / ``check_path`` / ``verify_goal``. The candidate ids
    used to be stored under a key the navigation tools never read, so every
    movement call with a ``cand_N`` id failed with ``unknown_target``."""

    REPLY = '{"candidates": [{"box": [450, 450, 100, 100]}]}'  # corner box, centre pixel

    def ground(self, ctx, reply=None):
        from vlm_nav.tool_impl_perception import ground_target, observe

        observed = observe(ctx, {"mode": "forward"})
        return ground_target(
            ctx,
            {
                "instruction": "the target",
                "observation_id": observed.data["observation_id"],
            },
            vlm=StubGroundingVLM(reply or self.REPLY),
        )

    def test_approach_and_check_path_resolve_a_grounded_candidate(self):
        ctx = make_context(FakeWorld(yaw_deg=0.0, depth=4.0))
        grounded = self.ground(ctx)
        candidate = grounded.data["candidates"][0]
        self.assertEqual(candidate["id"], "cand_1")
        self.assertEqual(candidate["position_flu"], [4.0, 0.0, 0.0])
        self.assertEqual(candidate["box_format"], "xyxy_norm")

        checked = registry().call("check_path", {"target_id": "cand_1"}, ctx)
        self.assertEqual(checked.status, "ok", checked.data)
        self.assertEqual(checked.data["target_position"], [4.0, 0.0, 0.0])

        moved = registry().call(
            "approach_candidate", {"candidate_id": "cand_1", "standoff_m": 1.0}, ctx
        )
        self.assertEqual((moved.status, moved.data["outcome"]), ("ok", "arrived"))
        self.assertEqual(moved.data["candidate_id"], "cand_1")
        self.assertAlmostEqual(float(ctx.world.position[0]), 3.0, places=3)

    def test_verify_goal_checks_a_grounded_candidate_identity(self):
        world = FakeWorld(position=(0.0, 0.0, 0.0), target=(4.0, 0.0, 0.0), depth=4.0)
        ctx = make_context(world, extra={"done_dist": 5.0})
        self.ground(ctx)

        result = registry().call(
            "verify_goal", {"pixel": [500.0, 500.0], "candidate_id": "cand_1"}, ctx
        )

        self.assertIs(result.data["verified"], True)
        self.assertAlmostEqual(
            result.data["checks"]["candidate_offset_m"], 0.0, places=3
        )
        self.assertNotIn(
            "candidate_identity", {item["check"] for item in result.data["unmet"]}
        )

    def test_an_unknown_candidate_is_answered_with_the_ones_that_exist(self):
        ctx = make_context(FakeWorld(depth=4.0))
        self.ground(ctx)
        result = registry().call("approach_candidate", {"candidate_id": "cand_9"}, ctx)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["error"], "unknown_candidate_id")
        self.assertEqual(result.data["available_candidate_ids"], ["cand_1"])
        self.assertEqual(result.data["next_action"]["tool"], "ground_target")

    def test_a_candidate_without_a_position_is_reported_as_unresolved(self):
        ctx = make_context(FakeWorld(depth=None))  # no depth, no distance_m given
        grounded = self.ground(ctx)
        self.assertNotIn("position_flu", grounded.data["candidates"][0])

        result = registry().call("approach_candidate", {"candidate_id": "cand_1"}, ctx)

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.data["reason"], "candidate_has_no_position")
        self.assertEqual(result.data["available_targets"], [])
        self.assertEqual(result.data["unresolved_targets"], ["cand_1"])
        self.assertTrue(result.data["hint"])
        self.assertEqual(ctx.world.applied, [])

    def test_raw_positions_stay_checkable_but_are_not_movement_targets(self):
        """A position is a place to *evaluate* now; movement takes typed ids."""
        ctx = make_context(FakeWorld())
        checked = registry().call("check_path", {"position": [3.0, 0.0, 0.0]}, ctx)
        self.assertEqual(checked.data["target_position"], [3.0, 0.0, 0.0])
        legacy = registry().call("navigate", {"position": [3.0, 0.0, 0.0]}, ctx)
        self.assertEqual(legacy.status, "failed")
        self.assertEqual(legacy.data["next_action"]["tool"], "propose_viewpoints")
        self.assertEqual(ctx.world.applied, [])

    def test_a_recalled_observation_id_is_navigable(self):
        """M2's round trip: a query_memory hit carries the pose it was taken at
        and the id that flies the drone back there."""
        from vlm_nav.memory import ObservationLog
        from vlm_nav.tool_impl_perception import observe, query_memory

        log = ObservationLog()
        world = FakeWorld(position=(2.0, 0.0, 0.0), yaw_deg=0.0)
        ctx = make_context(world, memory=log)
        observe(ctx, {"mode": "forward"})
        world.position = np.asarray([0.0, 0.0, 0.0])  # the drone flew on

        hits = query_memory(ctx, {"target": "blob"})

        hit = hits.data["hits"][0]
        self.assertEqual(hit["position_flu"], [2.0, 0.0, 0.0])
        moved = registry().call(
            "return_to_observation", {"observation_id": hit["observation_id"]}, ctx
        )
        self.assertEqual(moved.status, "ok", moved.data)
        self.assertEqual(moved.data["target_position"], [2.0, 0.0, 0.0])
        self.assertLess(
            float(np.linalg.norm(ctx.world.position - np.asarray([2.0, 0.0, 0.0]))),
            0.35,
        )


class StateCandidateIdContractTest(unittest.TestCase):
    """Every id the STATE block shows is an id the tool it names accepts.

    The block rendered the newest candidate as ``cand_1@obs_2`` - a spelling no
    tool resolves and no result ever returns. In the G1 matrix a model copied it
    verbatim into ``approach_candidate``, got ``unknown_candidate_id`` and spent
    the rest of the episode re-typing ids instead of navigating
    (``offview_left_y3/spatial_only``: 12 turns, 5.996 m). The frame the
    candidate was grounded in is a separate clause, never a suffix of its id.
    """

    REPLY = '{"candidates": [{"box": [450, 450, 100, 100]}]}'  # corner box, centre pixel

    def episode_prompt(self, ctx):
        """Render one turn prompt of a finished episode over ``ctx``."""
        from vlm_nav.runtime import AgentRuntime

        class FinishVLM:
            name = "state-id-mock"

            def __init__(self):
                self.seen: list[str] = []

            def decide_with_tools(self, frames, system_prompt, user_prompt,
                                  tools_text="", context=None):
                self.seen.append(user_prompt)
                return json.dumps({"tool": "finish", "args": {"reason": "done"}})

        vlm = FinishVLM()
        runtime = AgentRuntime(
            registry(), ctx.world, None, vlm, budget=4, max_turns=1, tool_context=ctx
        )
        runtime.run("reach the goal")
        return runtime, vlm.seen[0]

    def test_the_rendered_candidate_id_is_the_one_approach_candidate_takes(self):
        from vlm_nav.tool_impl_perception import ground_target, observe

        ctx = make_context(FakeWorld(yaw_deg=0.0, depth=4.0))
        observed = observe(ctx, {"mode": "forward"})
        frame = observed.data["observation_id"]
        ground_target(
            ctx,
            {"instruction": "the target", "observation_id": frame},
            vlm=StubGroundingVLM(self.REPLY),
        )
        record = ctx.extra[CANDIDATE_STORE_KEY]["cand_1"]
        self.assertEqual(record["observation_id"], frame)  # what used to be glued on

        runtime, prompt = self.episode_prompt(ctx)
        value = runtime.state()["candidate"]
        self.assertIn(value, prompt)  # the model reads exactly this
        token = value.split()[0]  # the id a model copies off the line

        self.assertEqual(token, "cand_1")
        self.assertIn(frame, value)  # the frame is still reported, as its own clause
        moved = registry().call("approach_candidate", {"candidate_id": token}, ctx)
        self.assertNotEqual(moved.data.get("reason"), "unknown_candidate_id")
        self.assertEqual(moved.data["candidate_id"], "cand_1")


class RecalledObservationTargetTest(unittest.TestCase):
    """M2's return leg: a recalled ``obs_N`` flies the drone back to its pose.

    ``query_memory`` returns observation ids together with the pose they were
    taken at; ``return_to_observation`` is the tool that uses them, and an id
    that does not resolve comes back with the ids that do plus a next step that
    is *not* the call that just failed.
    """

    def seen_at(self, ctx, position, yaw_deg=0.0):
        return seen_at(ctx, position, yaw_deg=yaw_deg)

    def recalled_context(self, memory=True, position=(0.0, 0.0, 0.0)):
        from vlm_nav.memory import ObservationLog

        return make_context(
            FakeWorld(position=position),
            memory=ObservationLog() if memory else None,
        )

    def test_return_to_observation_resolves_a_recalled_id(self):
        ctx = self.recalled_context()
        obs_id = self.seen_at(ctx, (2.0, 0.0, 0.0), yaw_deg=180.0)
        self.assertEqual(obs_id, "obs_1")

        result = registry().call("return_to_observation", {"observation_id": obs_id}, ctx)

        self.assertEqual(result.status, "ok", result.data)
        self.assertEqual(result.data["outcome"], "arrived")
        self.assertEqual(result.data["target_position"], [2.0, 0.0, 0.0])
        self.assertIn("observation:obs_1", result.to_prompt())

    def test_check_path_accepts_the_observation_id_argument(self):
        ctx = self.recalled_context()
        obs_id = self.seen_at(ctx, (2.0, 0.0, 0.0))

        checked = registry().call("check_path", {"observation_id": obs_id}, ctx)
        self.assertEqual(checked.status, "ok", checked.data)
        self.assertEqual(checked.data["target_position"], [2.0, 0.0, 0.0])
        self.assertIn("observation:obs_1", checked.to_prompt())

    def test_the_perception_cache_alone_resolves_an_observation_target(self):
        # no memory sink at all: the frames this episode captured are enough
        ctx = make_context(FakeWorld())
        obs_id = self.seen_at(ctx, (2.0, 0.0, 0.0))

        result = registry().call("check_path", {"target_id": obs_id}, ctx)

        self.assertEqual(result.status, "ok", result.data)
        self.assertEqual(result.data["target_position"], [2.0, 0.0, 0.0])

    def test_a_navigation_pose_record_is_a_navigable_target(self):
        world = FakeWorld()
        ctx = make_context(world)
        add_viewpoint(ctx, [2.0, 0.0, 0.0])
        first = registry().call("go_to_viewpoint", {"viewpoint_id": "vp_1"}, ctx)
        obs_id = first.data["last_observation_id"]
        self.assertTrue(obs_id)

        world.position = np.asarray([4.0, 0.0, 0.0])  # the drone flew on
        back = registry().call("return_to_observation", {"observation_id": obs_id}, ctx)

        self.assertEqual(back.status, "ok", back.data)
        self.assertEqual(back.data["target_position"], [2.0, 0.0, 0.0])

    def test_an_unknown_observation_id_answers_with_an_executable_next_step(self):
        ctx = self.recalled_context(memory=False)
        obs_id = self.seen_at(ctx, (2.0, 1.0, 0.5), yaw_deg=45.0)

        result = registry().call("return_to_observation", {"observation_id": "obs_9"}, ctx)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["error"], "unknown_observation_id")
        self.assertEqual(result.data["available_observation_ids"], [obs_id])
        # the next step is a *different* call, and it is executable
        self.assertEqual(
            result.data["next_action"],
            {"tool": "return_to_observation", "args": {"observation_id": obs_id}},
        )
        self.assertIn("observe", result.data["hint"])
        json.dumps(result.to_dict())

        follow = registry().call(
            "return_to_observation", dict(result.data["next_action"]["args"]), ctx
        )
        self.assertEqual(follow.status, "ok", follow.data)
        self.assertEqual(follow.data["target_position"], [2.0, 1.0, 0.5])

    def test_the_next_step_for_a_plain_unknown_id_is_re_grounding(self):
        ctx = self.recalled_context(memory=False)
        result = registry().call("return_to_observation", {"observation_id": "obs_9"}, ctx)
        # nothing to fly to: the answer sends the model to observe again
        self.assertEqual(result.data["next_action"]["tool"], "observe")

    def test_the_check_path_suggestion_skips_the_place_underfoot(self):
        # the newest observation is the runtime's own frame at the current pose:
        # flying there is not a move, so the answer must name the older place
        ctx = make_context(FakeWorld())
        ctx.extra["observations"] = {
            "obs_1": {"observation_id": "obs_1", "position_flu": [2.0, 0.0, 0.0]},
            "obs_2": {"observation_id": "obs_2", "position_flu": [0.0, 0.0, 0.0]},
        }

        result = registry().call("check_path", {"target_id": "obs_9"}, ctx)

        self.assertEqual(result.data["available_observations"], ["obs_2", "obs_1"])
        self.assertEqual(
            result.data["next_action"],
            {"tool": "return_to_observation", "args": {"observation_id": "obs_1"}},
        )
        self.assertEqual(result.data["suggested_position"], [2.0, 0.0, 0.0])

    def test_without_any_recalled_place_the_answer_still_names_a_next_step(self):
        ctx = make_context(FakeWorld())
        result = registry().call("check_path", {"target_id": "cand_9"}, ctx)
        self.assertEqual(result.data["reason"], "unknown_target")
        self.assertEqual(result.data["available_observations"], [])
        self.assertNotIn("suggested_position", result.data)
        self.assertEqual(result.data["next_action"]["tool"], "ground_target")

    def test_an_unresolved_candidate_also_offers_the_recalled_place(self):
        """The M2 failure loop: the model re-asks for a grounded candidate that
        carries no position. The answer says how to re-ground it - and names the
        recalled place it can fly to instead."""
        ctx = self.recalled_context(memory=False)
        store_candidate(ctx, "cand_1", position=None)
        obs_id = self.seen_at(ctx, (3.0, 0.0, 0.0))

        result = registry().call("check_path", {"target_id": "cand_1"}, ctx)

        self.assertEqual(result.data["reason"], "unknown_target")
        self.assertEqual(result.data["unresolved_targets"], ["cand_1"])
        self.assertIn("ground_target", result.data["hint"])
        self.assertIn(obs_id, result.data["hint"])
        self.assertEqual(
            result.data["next_action"]["args"], {"observation_id": obs_id}
        )

    def test_verify_goal_checks_a_recalled_observation_pose(self):
        world = FakeWorld(position=(2.5, 0.0, 0.0), target=(3.0, 0.0, 0.0))
        ctx = make_context(world)
        obs_id = self.seen_at(ctx, (2.9, 0.0, 0.0), yaw_deg=0.0)

        result = registry().call("verify_goal", {"observation_id": obs_id}, ctx)

        self.assertIs(result.data["verified"], True)
        self.assertEqual(
            result.data["checks"]["recalled_observation_position"], [2.9, 0.0, 0.0]
        )
        self.assertAlmostEqual(
            result.data["checks"]["recalled_observation_to_target_m"], 0.1, places=3
        )
        self.assertIn(obs_id, [item["ref"] for item in result.data["evidence"]])
        self.assertIn("finish", result.data["hint"])

    def test_verify_goal_rejects_a_recalled_place_that_is_not_the_goal(self):
        world = FakeWorld(position=(2.5, 0.0, 0.0), target=(3.0, 0.0, 0.0))
        ctx = make_context(world)
        obs_id = self.seen_at(ctx, (0.0, 0.0, 0.0), yaw_deg=0.0)

        result = registry().call("verify_goal", {"target_id": obs_id}, ctx)

        self.assertIs(result.data["verified"], False)
        self.assertEqual(result.data["reason"], "unmet_conditions")
        self.assertIn(
            "recalled_observation_on_target",
            {item["check"] for item in result.data["unmet"]},
        )
        # the answer must never read like "you may finish now"
        self.assertIn("approach_candidate", result.data["hint"])
        self.assertIn("return_to_observation", result.data["hint"])
        self.assertIn("verify_goal", result.data["hint"])
        self.assertIn('"success": true', result.data["hint"])

    def test_verify_goal_reports_an_unknown_recalled_id(self):
        world = FakeWorld(position=(2.5, 0.0, 0.0), target=(3.0, 0.0, 0.0))
        ctx = make_context(world)

        result = registry().call("verify_goal", {"observation_id": "obs_404"}, ctx)

        self.assertEqual(
            [item for item in result.data["unmet"] if item["check"] == "observation_pose"],
            [
                {
                    "check": "observation_pose",
                    "reason": "unknown_observation",
                    "observation_id": "obs_404",
                }
            ],
        )
        # a phantom id may never come back as a pass: the finish that followed
        # would claim success on evidence that does not exist
        self.assertEqual(result.data["verified"], "uncertain")
        self.assertEqual(result.data["reason"], "unknown_observation")
        self.assertIn("observation_id", result.data["hint"])
        self.assertIn("verify_goal", result.data["hint"])


class RecallFlightLoopTest(unittest.TestCase):
    """The whole M2 chain through the real tool loop (AgentRuntime + memory).

    The scripted model below copies what the 8B actually did: retrieve with
    ``query_memory``, then answer with the recalled *observation id* as the
    target. That spelling now goes to ``return_to_observation`` - the tool whose
    meaning is "fly back to the pose that frame was taken at" - and the return
    leg has to happen for real.
    """

    class RecordingTools(ToolRegistry):
        """Registry that remembers the last result, the way the model reads it."""

        def __init__(self):
            super().__init__()
            self.last_result: tuple = (None, None)

        def call(self, name, args, ctx):
            result = super().call(name, args, ctx)
            self.last_result = (name, result)
            return result

    class RecallThenReturnVLM:
        """Scripted stand-in for the model, driven by the tool results."""

        name = "recall-id-mock"
        STAGES = ("observe", "propose", "fly", "query", "return", "verify", "finish")

        def __init__(self, tools):
            self.tools = tools
            self.stage = 0
            self.recalled: dict | None = None
            self.flew_to: str = ""

        def decide(self, frames, system_prompt, user_prompt, context=None):
            name, result = self.tools.last_result
            if name == "query_memory" and result is not None:
                hits = result.data.get("hits") or []
                if hits:
                    self.recalled = hits[0]
            stage = self.STAGES[self.stage]
            self.stage += 1
            if stage == "observe":
                return json.dumps({"tool": "observe", "args": {"mode": "forward"}})
            if stage == "propose":
                return json.dumps(
                    {"tool": "propose_viewpoints", "args": {"direction_deg": 0.0, "count": 1}}
                )
            if stage == "fly":
                # fly to the viewpoint the proposal handed out: the drone really
                # leaves the pose the observation was taken at
                self.flew_to = str((result.data or {}).get("viewpoint_id") or "vp_1")
                return json.dumps(
                    {"tool": "go_to_viewpoint", "args": {"viewpoint_id": self.flew_to}}
                )
            if stage == "query":
                return json.dumps(
                    {"tool": "query_memory", "args": {"query": "blob", "limit": 3}}
                )
            if stage == "return":
                self.assert_recall()
                return json.dumps(
                    {
                        "tool": "return_to_observation",
                        "args": {"observation_id": self.recalled["observation_id"]},
                    }
                )
            if stage == "verify":
                return json.dumps(
                    {"tool": "verify_goal", "args": {"success_dist_m": 3.5}}
                )
            return json.dumps(
                {"tool": "finish",
                 "args": {"reason": "returned to the recalled view", "success": True}}
            )

        def assert_recall(self):
            if self.recalled is None or self.recalled.get("position_flu") is None:
                raise AssertionError(f"no actionable recall hit: {self.recalled}")

    def test_the_recalled_observation_id_flies_the_drone_back(self):
        from vlm_nav.memory import ObservationLog
        from vlm_nav.runtime import AgentRuntime
        from vlm_nav.tool_impl_perception import register_perception_tools

        tools = self.RecordingTools()
        register_perception_tools(tools)
        register_navigation_tools(tools)
        world = FakeWorld(position=(0.0, 0.0, 0.0), target=(3.0, 0.0, 0.0))
        vlm = self.RecallThenReturnVLM(tools)

        log = AgentRuntime(
            tools, world, ObservationLog(), vlm, budget=12, max_turns=12
        ).run("find the target again")

        self.assertEqual(
            [turn.tool for turn in log.turns],
            [
                "observe",
                "propose_viewpoints",
                "go_to_viewpoint",
                "query_memory",
                "return_to_observation",
                "verify_goal",
                "finish",
            ],
        )
        # the drone really left the observed pose (the go_to_viewpoint leg) and
        # the recall leg brought it back to it
        self.assertGreater(log.turns[2].data["moved_m"], 0.5)
        returned = log.turns[4]
        self.assertEqual(returned.status, "ok")
        self.assertEqual(returned.data["outcome"], "arrived")
        self.assertGreater(returned.data["moved_m"], 0.5)
        self.assertEqual(returned.data["target_position"], [0.0, 0.0, 0.0])
        recalled_id = vlm.recalled["observation_id"]
        self.assertIn(f"observation:{recalled_id}", returned.result_prompt)
        self.assertLess(float(np.linalg.norm(world.position)), 0.35)
        # and the verified finish the episode reports is backed by evidence
        self.assertIs(log.turns[5].data["verified"], True)
        self.assertEqual(log.finish_reason, "finish")
        self.assertTrue(log.success)


class RealWorldIntegrationTest(unittest.TestCase):
    """The tool set against the real CPU 3DGS world (no GPU required)."""

    def setUp(self):
        import tempfile

        from vlm_nav.world import SplatWorldCPU, TargetSpec

        self._tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(0)
        gaussians = []
        for _ in range(200):
            p = rng.uniform(-6, 6, size=3)
            p[2] = -1.5
            gaussians.append(
                (
                    tuple(p),
                    (0.0, 0.0, 0.0),
                    math.log(0.8 / 0.2),
                    (math.log(0.2), math.log(0.2), math.log(0.05)),
                    (1.0, 0.0, 0.0, 0.0),
                )
            )
        ply = f"{self._tmp.name}/scene.ply"
        _write_test_ply(ply, gaussians)
        self.world = SplatWorldCPU(
            ply_path=ply,
            target=TargetSpec(name="blob", position_flu=(3.0, 1.0, 0.0), radius_m=0.2),
            start_flu=(0.0, 0.0, 0.0),
            start_yaw_deg=0.0,
            width=64,
            height=64,
            hfov_deg=60.0,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_propose_check_go_to_viewpoint_against_real_world(self):
        reg = registry()
        ctx = make_context(self.world)
        proposed = reg.call("propose_viewpoints", {}, ctx)
        self.assertEqual(proposed.status, "ok")
        self.assertEqual(proposed.data["reachable_count"], 5)
        entry = proposed.data["viewpoints"][0]
        self.assertGreaterEqual(entry["cost_m"], 2.0)

        checked = reg.call("check_path", {"viewpoint_id": entry["viewpoint_id"]}, ctx)
        self.assertTrue(checked.data["reachable"])
        self.assertAlmostEqual(
            checked.data["target_position"], entry["position"], places=3
        )

        before = np.asarray(self.world.position_flu, dtype=np.float64).copy()
        moved = reg.call("go_to_viewpoint", {"viewpoint_id": entry["viewpoint_id"]}, ctx)
        self.assertEqual(moved.status, "ok")
        self.assertEqual(moved.data["task_id"], "nav_1")
        self.assertAlmostEqual(moved.data["progress_m"], 2.0, delta=0.25)
        self.assertAlmostEqual(
            float(np.linalg.norm(self.world.position_flu - before)),
            moved.data["moved_m"],
            places=3,
        )
        self.assertIn(moved.data["last_observation_id"], ctx.extra["observations"])
        json.dumps(moved.to_dict())

    def test_verify_goal_reports_depth_and_distance_without_depth_sensor(self):
        # SplatWorldCPU has no depth_at_pixel: the point cannot be localized
        ctx = make_context(self.world)
        far = registry().call("verify_goal", {"pixel": [500.0, 500.0]}, ctx)
        self.assertEqual(far.status, "ok")
        self.assertIs(far.data["verified"], False)  # target is 3.2 m away
        self.assertEqual(
            {item["check"] for item in far.data["unmet"]},
            {"distance_to_target_m", "depth"},
        )

        # inside the goal radius, but still no depth -> honest uncertainty
        self.world.state.position_flu = np.asarray([3.0, 1.0, 0.0])
        near = registry().call("verify_goal", {"pixel": [500.0, 500.0]}, ctx)
        self.assertEqual(near.data["verified"], "uncertain")
        self.assertEqual(near.data["reason"], "no_depth_estimate")

    def test_no_fabricated_target_point_without_a_range(self):
        """``estimate_point_world`` must not invent a range (the old 2 m fallback)."""
        from vlm_nav.world import estimate_point_world

        self.assertIsNone(estimate_point_world(self.world, (500.0, 500.0), 0.0))
        point = estimate_point_world(self.world, (500.0, 500.0), 3.0)
        self.assertIsNotNone(point)
        self.assertAlmostEqual(float(point[0]), 3.0, places=3)

    def test_the_ground_approach_verify_finish_chain_moves_the_drone(self):
        """The user's scenario end to end, through the real module stack.

        ``observe -> ground_target -> approach_candidate -> verify_goal`` driven
        by ``AgentRuntime`` with the perception + navigation modules registered,
        one scripted backend playing both the tool-calling model and the
        grounding model (the way the real 8B does). The approach must produce a
        real displacement and the verification that closes the episode must rest
        on the place the drone actually reached.
        """
        from vlm_nav.runtime import AgentRuntime
        from vlm_nav.tool_impl_perception import register_perception_tools

        class Tools(ToolRegistry):
            def __init__(self):
                super().__init__()
                self.last_result = None

            def call(self, name, args, ctx):
                result = super().call(name, args, ctx)
                self.last_result = (name, result)
                return result

        class ToolAndGroundingVLM:
            """Scripted model + grounding backend in one object."""

            GROUND_REPLY = '{"candidates": [{"box": [450, 450, 550, 550]}]}'

            def __init__(self, tools, script):
                self.tools = tools
                self.script = list(script)
                self.seen = []

            def _is_grounding(self, system_prompt):
                return "grounding" in str(system_prompt or "").lower()

            def decide(self, frames, system_prompt, user_prompt, context=None):
                if self._is_grounding(system_prompt):
                    return self.GROUND_REPLY
                return self._tool_call()

            def decide_with_tools(self, frames, system_prompt, user_prompt,
                                  tools_text="", context=None):
                if self._is_grounding(system_prompt):
                    return self.GROUND_REPLY
                return self._tool_call()

            def _tool_call(self):
                name, result = self.tools.last_result or (None, None)
                item = dict(self.script.pop(0)) if self.script else {
                    "tool": "finish", "args": {"reason": "done"},
                }
                args = dict(item.get("args") or {})
                for key, value in list(args.items()):
                    if value == "$observation_id" and result is not None:
                        args[key] = result.data.get("observation_id")
                    if value == "$candidate_id" and result is not None:
                        candidates = result.data.get("candidates") or []
                        args[key] = candidates[0]["id"] if candidates else value
                item["args"] = args
                return json.dumps(item)

        world = DepthWorld(surface_m=4.0, target=(4.0, 0.0, 0.0), radius_m=0.3)
        tools = Tools()
        vlm = ToolAndGroundingVLM(
            tools,
            [
                {"tool": "observe", "args": {"mode": "forward"}},
                {"tool": "ground_target",
                 "args": {"instruction": "the blob", "observation_id": "$observation_id"}},
                {"tool": "approach_candidate",
                 "args": {"candidate_id": "$candidate_id", "standoff_m": 1.0}},
                {"tool": "verify_goal", "args": {"success_dist_m": 3.5}},
                {"tool": "finish", "args": {"reason": "approached and verified",
                                            "success": True}},
            ],
        )
        register_perception_tools(tools, vlm=vlm)  # ground_target uses this backend
        register_navigation_tools(tools)
        runtime = AgentRuntime(tools, world, None, vlm, budget=12, max_turns=8)
        log = runtime.run("approach the blob")

        self.assertEqual(
            [turn.tool for turn in log.turns],
            ["observe", "ground_target", "approach_candidate", "verify_goal", "finish"],
        )
        grounded = log.turns[1].data
        self.assertEqual(grounded["candidates"][0]["id"], "cand_1")
        # the candidate really landed on the surface the depth sensor measured
        self.assertLess(
            float(np.linalg.norm(
                np.asarray(grounded["candidates"][0]["position_flu"]) - np.asarray([4.0, 0.0, 0.0])
            )),
            0.4,
        )
        approach = log.turns[2]
        self.assertEqual(approach.status, "ok")
        self.assertEqual(approach.data["outcome"], "arrived")
        self.assertAlmostEqual(approach.data["moved_m"], 3.0, places=3)
        self.assertTrue(approach.data["last_observation_id"])
        self.assertLess(float(np.linalg.norm(world.position - np.asarray([4.0, 0.0, 0.0]))), 1.4)
        # the episode's success rests on the verification of that place
        self.assertIs(log.turns[3].data["verified"], True)
        self.assertEqual(log.finish_reason, "finish")
        self.assertTrue(log.success)

    def test_a_navigate_only_script_cannot_move_the_drone(self):
        """The deprecated call is answered, never executed: a legacy script that
        only knows ``navigate`` moves nothing and is told what to call instead."""
        from vlm_nav.tool_impl_perception import register_perception_tools

        ctx = make_context(FakeWorld())
        store_candidate(ctx, "cand_1", position=[4.0, 0.0, 0.0])
        reg = registry()
        register_perception_tools(reg)
        result = reg.call("navigate", {"target_id": "cand_1"}, ctx)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["next_action"]["tool"], "approach_candidate")
        self.assertEqual(ctx.world.applied, [])
        # the replacement the answer names really flies
        followed = reg.call("approach_candidate", dict(result.data["next_action"]["args"]), ctx)
        self.assertEqual(followed.status, "ok")
        self.assertGreater(float(np.linalg.norm(ctx.world.position)), 0.5)


class NavigationVocabularyTest(unittest.TestCase):
    """``scripts/run_vlm_nav_tools.py`` hands the model the intent tools."""

    def load_runner(self):
        import importlib.util
        import sys
        from pathlib import Path

        name = "diffaero_script_run_vlm_nav_tools_nav_vocab"
        if name in sys.modules:
            return sys.modules[name]
        path = Path(__file__).resolve().parents[1] / "scripts" / "run_vlm_nav_tools.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def test_the_default_catalogue_offers_the_three_intents(self):
        runner = self.load_runner()
        modules, _ = runner.load_m1_modules()
        registry, backend, tools_source = runner.build_registry(modules)
        self.assertEqual("modules", backend)
        for name in MOVE_TOOLS:
            self.assertIn(name, registry.names(), name)
            self.assertEqual(tools_source[name], "modules")
        for name in MOVE_TOOLS:
            self.assertIn(name, runner.FULL_TOOL_NAMES)

    def test_the_legacy_fallback_stack_is_unchanged(self):
        # the fallback registry (module-less runs) keeps the legacy navigate
        runner = self.load_runner()
        registry, backend, tools_source = runner.build_registry(
            {key: None for key in ("runtime", "memory", "perception", "navigation", "task")}
        )
        self.assertEqual("fallback", backend)
        self.assertIn("navigate", registry.names())
        for name in MOVE_TOOLS:
            self.assertNotIn(name, registry.names(), name)


if __name__ == "__main__":
    unittest.main()
