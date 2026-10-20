"""Mechanism regressions for version 2: arrival is not confirmation.

``confirm_candidate`` is the bounded routine that turns a reached candidate
into an accepted identity: approach -> new view -> attribute/relation check ->
location consistency -> accepted / uncertain / rejected. These tests pin the
four properties the completion contract now depends on:

- a candidate whose place is wrong cannot be accepted from the new view;
- an image already used for confirmation cannot add a confirmation;
- ``uncertain`` routes to a directed next step (distance / relation / path),
  never to a generic re-search;
- ``finish`` refuses a reached-but-unconfirmed candidate.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from vlm_nav.actions import NavAction, action_to_body_delta, body_delta_to_world
from vlm_nav.mcp_server import ENV_TOOL_NAMES, TOOL_CONTRACTS
from vlm_nav.tool_impl_navigation import register_navigation_tools
from vlm_nav.tool_impl_perception import register_perception_tools
from vlm_nav.tools import (
    STATUS_BLOCKED,
    STATUS_CACHED,
    STATUS_OK,
    ToolContext,
    ToolRegistry,
    evidence_pack,
)

ACCEPTED = {
    "target": True,
    "same_instance": True,
    "attributes_match": True,
    "relation_supported": True,
    "confidence": 0.9,
    "reason": "same brown cabinet beside the wooden door",
    # the confirmation asks the model to point at the object it judged: the
    # position check measures this box on the fresh frame, never the candidate's
    # own projected pixel
    "box": [450, 450, 550, 550],
}
WRONG_INSTANCE = {
    "target": False,
    "same_instance": False,
    "attributes_match": False,
    "relation_supported": False,
    "confidence": 0.85,
    "reason": "a different green cabinet",
}
RELATION_UNRESOLVED = {
    "target": True,
    "same_instance": None,
    "attributes_match": True,
    "relation_supported": False,
    "confidence": 0.5,
    "reason": "the door is not in this view",
    "missing": ["relation"],
    "next_view": "look from the left so both objects are in frame",
}


class FakeTarget:
    def __init__(self, position=(3.0, 0.0, 0.0), name="cabinet", radius_m=0.5):
        self.position_flu = tuple(float(v) for v in position)
        self.name = name
        self.radius_m = radius_m


class ConfirmWorld:
    """Kinematic fake world whose rendered view changes with the heading.

    ``vary_view=False`` renders the same pixels from every heading - the
    "same image with a new id" case the independence rule must reject.
    """

    def __init__(
        self,
        position=(0.0, 0.0, 0.0),
        yaw_deg=0.0,
        target=(3.0, 0.0, 0.0),
        *,
        vary_view=True,
        blocked=False,
        depth=None,
    ):
        self.state = SimpleNamespace(yaw_rad=math.radians(yaw_deg))
        self.position = np.asarray(position, dtype=np.float64)
        self.target = FakeTarget(position=target)
        self.vary_view = bool(vary_view)
        self.blocked = bool(blocked)
        self.depth = depth
        self.width, self.height, self.hfov_deg = 320, 180, 90.0
        self.applied: list[NavAction] = []

    @property
    def yaw_rad(self):
        return self.state.yaw_rad

    @yaw_rad.setter
    def yaw_rad(self, value):
        self.state.yaw_rad = float(value)

    @property
    def position_flu(self):
        return self.position

    def distance_to_target(self):
        return float(np.linalg.norm(self.position - np.asarray(self.target.position_flu)))

    def target_in_fov(self):
        return True

    def depth_at_pixel(self, pixel):
        if self.depth is not None:
            return float(self.depth)
        return max(0.05, float(np.linalg.norm(
            np.asarray(self.target.position_flu) - self.position
        )))

    def render(self):
        frame = np.zeros((3, 8, 8), dtype=np.float32)
        frame[0, 0, 0] = 0.25
        if self.vary_view:
            frame[1, 0, 0] = math.degrees(self.state.yaw_rad) / 1000.0
            frame[2, 0, 0] = float(self.position[0]) / 100.0
        return frame

    def scan_views(self, n_views=8, pitch_down_deg=0.0, spread_deg=None):
        return [self.render() for _ in range(int(n_views))]

    def apply_action(self, action: NavAction) -> str:
        self.applied.append(action)
        if action.is_terminal or action.kind == "hover":
            return action.kind
        if self.blocked:
            return "stalled"
        dx, dy, dz, dyaw = action_to_body_delta(action)
        self.state.yaw_rad = math.atan2(
            math.sin(self.state.yaw_rad + dyaw), math.cos(self.state.yaw_rad + dyaw)
        )
        wx, wy, wz = body_delta_to_world(dx, dy, dz, self.state.yaw_rad)
        self.position = self.position + np.asarray([wx, wy, wz], dtype=np.float64)
        return f"moved to ({self.position[0]:.2f}, {self.position[1]:.2f})"


class ElevatedWallWorld(ConfirmWorld):
    """A vertical wall, i.e. the seed7/seed17 geometry: a candidate centre above
    the level camera.

    The wall sits at ``wall_m`` ahead of the drone; the candidate is the point
    on it at ``target_z``. ``depth_at_pixel`` is the axial distance of the ray
    through a pixel for a camera pitched by ``pitch_rad`` (nose-down positive),
    so a level camera cannot put the centre in frame while a nose-up read can.
    That is exactly the projection the v8 position check failed on.
    """

    supports_pitched_depth = True

    def __init__(
        self,
        position=(0.0, 0.0, 0.0),
        yaw_deg=0.0,
        *,
        wall_m=1.196,
        target_z=0.889,
        width=640,
        height=360,
        hfov_deg=86.0,
    ):
        super().__init__(position=position, yaw_deg=yaw_deg, target=(wall_m, 0.0, target_z))
        self.wall_m = float(wall_m)
        self.target_z = float(target_z)
        self.width, self.height, self.hfov_deg = int(width), int(height), float(hfov_deg)
        self.pitch_calls: list[float] = []

    def depth_at_pixel(self, pixel, pitch_rad=0.0):
        self.pitch_calls.append(float(pitch_rad))
        fx = self.width / (2.0 * math.tan(math.radians(self.hfov_deg) / 2.0))
        u = pixel[0] / 1000.0 * self.width
        v = pixel[1] / 1000.0 * self.height
        left = -(u - self.width / 2.0) / fx
        up = -(v - self.height / 2.0) / fx
        cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
        # camera-frame ray (1, left, up) rotated into the body frame
        forward = cos_p + up * sin_p
        if forward <= 1e-6:
            return None
        return self.wall_m / forward

    def render(self):
        frame = np.zeros((3, self.height, self.width), dtype=np.float32)
        frame[0, 0, 0] = 0.25
        if self.vary_view:
            frame[1, 0, 0] = math.degrees(self.state.yaw_rad) / 1000.0
            frame[2, 0, 0] = float(self.position[0]) / 100.0
        return frame


class TwoSurfaceWorld(ConfirmWorld):
    """A candidate point and the true object on the same fronto-parallel wall.

    The candidate is at (1.5, 0, 0); the object the model points at is at
    (1.5, 2.2, 0), inside a 120 deg horizontal field of view. Every pixel reads
    the wall's axial depth (1.5 m), so the *old* check re-projected the
    candidate onto its own pixel and measured delta 0.0. Only a measurement of
    the model's box can see the 2.5 m offset.
    """

    def __init__(self):
        super().__init__(position=(0.0, 0.0, 0.0), yaw_deg=0.0, target=(1.5, 0.0, 0.0))
        self.width, self.height, self.hfov_deg = 640, 360, 120.0
        self.object_position = (1.5, 2.2, 0.0)

    def depth_at_pixel(self, pixel):
        return 1.5


class ScriptedVLM:
    name = "scripted-confirm"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def decide(self, frames, system_prompt, user_prompt, context=None):
        self.calls.append(
            {
                "frames": len(frames),
                "frames_raw": list(frames),
                "context": dict(context or {}),
                "prompt": user_prompt,
            }
        )
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return reply if isinstance(reply, str) else json.dumps(reply)


def make_context(world, vlm=None, calls=80):
    extra = {
        "done_dist": 10.0,
        "instruction": "the wooden cabinet next to the wooden door",
    }
    if vlm is not None:
        extra["perception.vlm"] = vlm
    return ToolContext(world=world, budget={"tool_calls_left": calls}, extra=extra)


def make_registry(vlm):
    registry = ToolRegistry()
    register_perception_tools(registry, vlm=vlm)
    register_navigation_tools(registry, vlm=vlm)
    return registry


def store_candidate(
    ctx,
    candidate_id="cand_1",
    *,
    position=(3.0, 0.0, 0.0),
    observation_id="obs_src",
    status=None,
    ray=None,
):
    record = {
        "id": candidate_id,
        "source": "ground_target",
        "box": [450.0, 450.0, 550.0, 550.0],
        "box_format": "xyxy",
        "attrs": {"color": "brown", "material": "wood"},
        "confidence": 0.8,
        "uncertainty": "",
        "observation_id": observation_id,
        "instruction": "the wooden cabinet",
        "position_source": "depth",
    }
    if position is not None:
        record["position_flu"] = [float(v) for v in position]
        record["range_m"] = 3.0
    if ray is not None:
        record["ray"] = dict(ray)
        record["position_source"] = "ray"
        record["range_status"] = "unknown"
    if status is not None:
        record["status"] = status
    ctx.extra.setdefault("perception.candidates", {})[candidate_id] = record
    return record


class ConfirmCandidateTest(unittest.TestCase):
    def test_a_correct_instance_is_accepted_from_new_views(self):
        vlm = ScriptedVLM([ACCEPTED, ACCEPTED])
        ctx = make_context(ConfirmWorld(), vlm)
        registry = make_registry(vlm)
        source = registry.call("observe", {"mode": "forward"}, ctx)
        store_candidate(ctx, observation_id=source.data["observation_id"])

        result = registry.call(
            "confirm_candidate",
            {"candidate_id": "cand_1", "instruction": "the wooden cabinet next to the wooden door"},
            ctx,
        )

        self.assertEqual(result.status, STATUS_OK, result.data)
        self.assertEqual(result.data["verdict"], "accepted")
        self.assertIs(result.data["new_confirmation"], True)
        self.assertEqual(result.data["confirmation_count"], 1)
        self.assertGreaterEqual(result.data["vlm_calls"], 1)
        self.assertTrue(result.data["observation_ids"])
        self.assertEqual(result.data["missing"], [])
        self.assertIn("steps", result.data)
        self.assertIn("cost", result.data)
        # every judged view counts as a model call and is published
        self.assertEqual(result.data["cost"]["vlm_calls"], result.data["vlm_calls"])
        # the source frame and the new view both travel to the model
        self.assertEqual(vlm.calls[0]["frames"], 2)
        self.assertIs(vlm.calls[0]["context"]["source_frame_attached"], True)
        # location consistency ran on the fresh view and agrees with the
        # candidate estimate: appearance alone is not what accepted it
        self.assertIs(result.data["position_check"]["available"], True)
        self.assertLessEqual(
            result.data["position_check"]["delta_m"], 1.0, result.data["position_check"]
        )

    def test_a_candidate_outside_the_subtask_anchor_is_not_accepted(self):
        """v6 accepted two confirmations 2.2-3.0 m from a stage with locate 1.5."""
        vlm = ScriptedVLM([ACCEPTED, ACCEPTED])
        ctx = make_context(ConfirmWorld(), vlm)
        registry = make_registry(vlm)
        source = registry.call("observe", {"mode": "forward"}, ctx)
        store_candidate(ctx, observation_id=source.data["observation_id"])

        from vlm_nav.tool_impl_task import update_plan

        plan = update_plan(
            ctx,
            {
                "steps": [
                    {
                        "id": "st_1",
                        "goal": "the wooden cabinet",
                        "position_flu": [8.0, 0.0, 0.0],
                        "locate_m": 1.5,
                    }
                ]
            },
        )
        self.assertEqual(plan.status, STATUS_OK, plan.data)

        result = registry.call(
            "confirm_candidate",
            {"candidate_id": "cand_1", "instruction": "the wooden cabinet"},
            ctx,
        )

        self.assertEqual(result.data["verdict"], "uncertain", result.data)
        self.assertEqual(result.data["reason"], "outside_subtask_anchor")
        self.assertEqual(result.data["missing"], ["location"])
        anchor = result.data["position_check"]["subtask_anchor"]
        self.assertIs(anchor["available"], True, anchor)
        self.assertIs(anchor["consistent"], False, anchor)
        self.assertGreater(anchor["distance_m"], anchor["tolerance_m"])
        self.assertEqual(result.data["confirmation_count"], 0)

    def test_a_wrong_candidate_is_rejected_and_finish_cannot_count_it(self):
        vlm = ScriptedVLM([WRONG_INSTANCE])
        world = ConfirmWorld(target=(3.0, 0.0, 0.0))
        ctx = make_context(world, vlm)
        registry = make_registry(vlm)
        store_candidate(ctx, position=(5.0, 1.5, 0.0))

        arrived = registry.call("approach_candidate", {"candidate_id": "cand_1"}, ctx)
        self.assertEqual(arrived.data["outcome"], "arrived", arrived.data)

        result = registry.call(
            "confirm_candidate",
            {"candidate_id": "cand_1", "instruction": "the wooden cabinet"},
            ctx,
        )

        self.assertIn(result.data["verdict"], ("rejected", "uncertain"))
        self.assertNotEqual(result.data["verdict"], "accepted")
        self.assertIn("identity", result.data["missing"])
        rejected_record = ctx.extra["perception.candidates"]["cand_1"]
        self.assertEqual(rejected_record["status"], "rejected")

        # a rejected candidate is not usable by another approach
        again = registry.call("approach_candidate", {"candidate_id": "cand_1"}, ctx)
        self.assertEqual(again.status, STATUS_BLOCKED)
        self.assertEqual(again.data["reason"], "candidate_rejected")
        self.assertEqual(again.data["next_action"]["tool"], "ground_target")

        finish = registry.call(
            "finish", {"reason": "arrived at the candidate", "success": True}, ctx
        )
        self.assertEqual(finish.status, STATUS_BLOCKED)
        self.assertIs(finish.data["success"], False)
        self.assertIs(finish.data["confirmation_required"], True)
        self.assertIs(finish.data["confirmation_accepted"], False)
        self.assertIn("identity confirmation", " ".join(finish.data["missing"]))

    def test_the_same_image_repeated_does_not_add_a_confirmation(self):
        # every heading renders the same pixels: a fresh obs id is not new evidence
        vlm = ScriptedVLM([ACCEPTED])
        ctx = make_context(ConfirmWorld(vary_view=False), vlm)
        registry = make_registry(vlm)
        store_candidate(ctx)

        first = registry.call("confirm_candidate", {"candidate_id": "cand_1"}, ctx)
        self.assertEqual(first.data["verdict"], "accepted")
        self.assertEqual(first.data["confirmation_count"], 1)
        self.assertEqual(first.data["vlm_calls"], 1)
        revision = first.data["evidence_revision"]
        calls_after_first = len(vlm.calls)

        second = registry.call("confirm_candidate", {"candidate_id": "cand_1"}, ctx)

        self.assertEqual(second.status, STATUS_CACHED)
        self.assertIs(second.data["new_confirmation"], False)
        self.assertIs(second.data["repeated_evidence"], True)
        self.assertEqual(second.data["confirmation_count"], 1)
        self.assertEqual(second.data["evidence_revision"], revision)
        self.assertEqual(len(vlm.calls), calls_after_first)  # no model call either

    def test_the_measurement_runs_at_the_fresh_view_heading(self):
        """The depth read must use the view's heading, not the yaw it restored.

        ``observe`` renders at the candidate bearing and then restores the
        drone's yaw; reading depth afterwards measured a frame that did not hold
        the candidate (the v7 ``no_depth_measurement``/``target_out_of_view``
        failures).
        """
        vlm = ScriptedVLM([ACCEPTED])
        world = ConfirmWorld(position=(1.8, 0.0, 0.0), yaw_deg=90.0)

        def depth_at_view_heading(pixel):
            yaw_deg = (math.degrees(world.state.yaw_rad) + 180.0) % 360.0 - 180.0
            if abs(yaw_deg) > 5.0:  # only the frame that faces the candidate
                return None
            return 1.2

        world.depth_at_pixel = depth_at_view_heading
        ctx = make_context(world, vlm)
        registry = make_registry(vlm)
        store_candidate(ctx)

        result = registry.call(
            "confirm_candidate",
            {"candidate_id": "cand_1", "instruction": "the wooden cabinet"},
            ctx,
        )

        self.assertEqual(result.data["verdict"], "accepted", result.data)
        self.assertIs(result.data["position_check"]["available"], True)
        self.assertLessEqual(result.data["position_check"]["delta_m"], 1.0)
        # the world heading is restored after the measurement
        self.assertAlmostEqual(math.degrees(world.yaw_rad), 90.0, places=3)

    def test_the_position_check_measures_the_box_the_model_points_at(self):
        """delta is the difference between two independent measurements.

        The candidate is at (1.5, 0, 0); the model points at the object at
        (1.5, 2.2, 0). Reading depth at the *candidate's* projected pixel (the
        old check) returns the candidate point itself and delta 0.0, so any
        candidate was accepted. Measuring the model's box sees the 2.2 m gap
        and rejects the candidate.
        """
        from vlm_nav.skills_approach import world_to_pixel

        world = TwoSurfaceWorld()
        pixel = world_to_pixel(
            world.object_position,
            world.position_flu,
            0.0,
            {"width": world.width, "height": world.height, "hfov_deg": world.hfov_deg},
        )
        self.assertIsNotNone(pixel)
        box = [pixel[0] - 40.0, pixel[1] - 40.0, pixel[0] + 40.0, pixel[1] + 40.0]
        vlm = ScriptedVLM([dict(ACCEPTED, box=box)])
        ctx = make_context(world, vlm)
        registry = make_registry(vlm)
        store_candidate(ctx, position=(1.5, 0.0, 0.0))

        result = registry.call(
            "confirm_candidate",
            {"candidate_id": "cand_1", "instruction": "the wooden cabinet"},
            ctx,
        )

        self.assertEqual(result.data["verdict"], "rejected", result.data)
        self.assertEqual(result.data["reason"], "position_inconsistent")
        check = result.data["position_check"]
        self.assertIs(check["available"], True, check)
        self.assertEqual(check["method"], "observed_box")
        self.assertAlmostEqual(check["delta_m"], 2.2, delta=0.2)
        self.assertNotEqual(check.get("candidate_estimate_flu"), check.get("measured_position_flu"))
        self.assertEqual(result.data["confirmation_count"], 0)

    def test_a_view_without_a_box_cannot_supply_a_position(self):
        """No box = no independent measurement: acceptance stays uncertain."""
        reply = {key: value for key, value in ACCEPTED.items() if key != "box"}
        vlm = ScriptedVLM([reply])
        ctx = make_context(ConfirmWorld(position=(1.8, 0.0, 0.0)), vlm)
        registry = make_registry(vlm)
        store_candidate(ctx)

        result = registry.call(
            "confirm_candidate",
            {"candidate_id": "cand_1", "instruction": "the wooden cabinet"},
            ctx,
        )

        self.assertEqual(result.data["verdict"], "uncertain", result.data)
        self.assertIn("localization", result.data["missing"])
        check = result.data["position_check"]
        self.assertIs(check["available"], False, check)
        self.assertEqual(check["reason"], "no_box_from_view")
        self.assertEqual(result.data["confirmation_count"], 0)

    def test_a_view_without_a_box_is_retried_once_for_the_box(self):
        """The independent position half needs a box; a box-only retry supplies
        it without another view or capture (v5 bedroom: 15 such views)."""
        no_box = {key: value for key, value in ACCEPTED.items() if key != "box"}
        retry = {
            "target": True,
            "box": [450, 450, 550, 550],
            "confidence": 0.8,
            "reason": "box only",
        }
        vlm = ScriptedVLM([no_box, retry, ACCEPTED])
        world = ConfirmWorld(position=(1.8, 0.0, 0.0), target=(3.0, 0.0, 0.0))
        ctx = make_context(world, vlm)
        registry = make_registry(vlm)
        store_candidate(ctx)

        result = registry.call(
            "confirm_candidate",
            {"candidate_id": "cand_1", "instruction": "the wooden cabinet"},
            ctx,
        )

        self.assertEqual(result.data["verdict"], "accepted", result.data)
        # view 1: no box -> retry, view 2: normal answer
        self.assertEqual(result.data["vlm_calls"], 3)
        check = result.data["position_check"]
        self.assertIs(check["available"], True, check)
        self.assertEqual(check["method"], "observed_box")
        self.assertIsNotNone(check.get("box"))
        # the retry carried a visual anchor (the candidate's projected box);
        # the plain same-frame text retry recovered 0/22 in v6
        retry_frame = torch.as_tensor(vlm.calls[1]["frames_raw"][0])
        red = (retry_frame[0] > 0.9) & (retry_frame[1] < 0.1) & (retry_frame[2] < 0.1)
        self.assertTrue(bool(red.any()))

    def test_a_whole_frame_box_is_not_an_object_measurement(self):
        """The v3 seed41 failure: Qwen boxed [0,0,1000,1000] and the nearest
        sample (the floor in front of the drone) matched the wrong candidate.

        A box covering (almost) the whole frame cannot localise the object, so
        it supplies no position evidence and the verdict stays uncertain.
        """
        world = ConfirmWorld(position=(0.0, 0.0, 0.0), target=(1.5, 0.0, 0.0), depth=1.5)
        vlm = ScriptedVLM([dict(ACCEPTED, box=[0, 0, 1000, 1000])])
        ctx = make_context(world, vlm)
        registry = make_registry(vlm)
        store_candidate(ctx, position=(1.5, 0.0, 0.0))

        result = registry.call(
            "confirm_candidate",
            {"candidate_id": "cand_1", "instruction": "the wooden cabinet"},
            ctx,
        )

        self.assertEqual(result.data["verdict"], "uncertain", result.data)
        self.assertNotEqual(result.data["verdict"], "accepted")
        check = result.data["position_check"]
        self.assertIs(check["available"], False, check)
        self.assertEqual(check["reason"], "box_too_large")
        self.assertGreaterEqual(check["box_area_fraction"], 0.5)
        self.assertEqual(result.data["confirmation_count"], 0)

    def test_a_large_mostly_frame_box_is_not_an_object_measurement(self):
        """The v3 seed37 failure: a 70% box (top-to-bottom) matched the floor.

        The box covers 0.70 of the frame, below the old whole-frame threshold,
        but its samples are still the floor in front of the drone. Large boxes
        supply no position evidence; the candidate stays uncertain.
        """
        world = ConfirmWorld(position=(0.0, 0.0, 0.0), target=(1.5, 0.0, 0.0), depth=1.5)
        vlm = ScriptedVLM([dict(ACCEPTED, box=[150, 0, 850, 999])])
        ctx = make_context(world, vlm)
        registry = make_registry(vlm)
        store_candidate(ctx, position=(1.5, 0.0, 0.0))

        result = registry.call(
            "confirm_candidate",
            {"candidate_id": "cand_1", "instruction": "the wooden cabinet"},
            ctx,
        )

        self.assertEqual(result.data["verdict"], "uncertain", result.data)
        check = result.data["position_check"]
        self.assertIs(check["available"], False, check)
        self.assertEqual(check["reason"], "box_too_large")
        self.assertGreaterEqual(check["box_area_fraction"], 0.5)
        self.assertEqual(result.data["confirmation_count"], 0)

    def test_the_box_centre_is_the_measurement_not_the_nearest_sample(self):
        """A normal box measures its centre; a closer bottom sample must not win."""
        world = ConfirmWorld(position=(0.0, 0.0, 0.0), target=(1.5, 0.0, 0.0))
        world.depth_at_pixel = lambda pixel: (
            0.5 if float(pixel[1]) > 530.0 else 1.5
        )
        vlm = ScriptedVLM([ACCEPTED])
        ctx = make_context(world, vlm)
        registry = make_registry(vlm)
        store_candidate(ctx, position=(1.5, 0.0, 0.0))

        result = registry.call(
            "confirm_candidate",
            {"candidate_id": "cand_1", "instruction": "the wooden cabinet"},
            ctx,
        )

        self.assertEqual(result.data["verdict"], "accepted", result.data)
        check = result.data["position_check"]
        self.assertEqual(check["chosen_sample"], "centre")
        self.assertAlmostEqual(check["depth_m"], 1.5, places=3)

    def test_a_distance_conflicted_candidate_is_refused_before_any_view(self):
        """A grounding whose depth contradicted the model cannot be confirmed.

        The v4 holdout looped confirm -> propose_viewpoints -> confirm on the
        same stale candidate, spending 17 classroom confirm calls. A candidate
        whose range is unresolved has no identity question a new view can
        answer: the call is blocked immediately, takes no view and no model
        call, and returns a forced re-ground from a different viewpoint.
        """
        world = ConfirmWorld(position=(0.0, 0.0, 0.0), target=(1.5, 0.0, 0.0), depth=1.5)
        vlm = ScriptedVLM([ACCEPTED])
        ctx = make_context(world, vlm)
        registry = make_registry(vlm)
        record = store_candidate(ctx, position=(1.5, 0.0, 0.0))
        record.update(
            distance_source="conflict",
            distance_uncertain=True,
            distance_conflict={
                "model_m": 3.0,
                "depth_center_m": 16.0,
                "ratio": 5.33,
                "difference_m": 13.0,
                "resolved": False,
            },
        )

        result = registry.call(
            "confirm_candidate",
            {"candidate_id": "cand_1", "instruction": "the wooden cabinet"},
            ctx,
        )

        self.assertEqual(result.status, STATUS_BLOCKED)
        self.assertEqual(result.data["verdict"], "uncertain", result.data)
        self.assertEqual(result.data["reason"], "candidate_distance_uncertain")
        self.assertIn("localization", result.data["missing"])
        # no viewpoint was checked and no model was asked: the range is the
        # question, not the identity
        self.assertEqual(vlm.calls, [])
        self.assertEqual(result.data["views"], [])
        # the forced next step changes the viewpoint and then re-grounds
        self.assertEqual(result.data["next_action"]["tool"], "ground_target")
        self.assertTrue(result.data["next_action"].get("viewpoints"))
        self.assertTrue(result.data["next_action"].get("observe_first"))
        self.assertEqual(result.data["confirmation_count"], 0)
        self.assertFalse(result.data["position_check"].get("candidate_distance_conflict") is None)
        self.assertEqual(record["distance_uncertain_confirms"], 1)

    def test_a_second_distance_uncertain_confirm_stalls_and_requires_reground(self):
        """The same stale candidate confirmed twice is downgraded, not retried.

        The first refusal is a direction; the second identical call proves the
        loop did not move. The candidate is marked unusable for confirmation
        and approach, and the result names the stall instead of opening another
        viewpoint round.
        """
        world = ConfirmWorld(position=(0.0, 0.0, 0.0), target=(1.5, 0.0, 0.0), depth=1.5)
        vlm = ScriptedVLM([ACCEPTED])
        ctx = make_context(world, vlm)
        registry = make_registry(vlm)
        record = store_candidate(ctx, position=(1.5, 0.0, 0.0))
        record.update(
            distance_source="conflict",
            distance_uncertain=True,
            distance_conflict={"model_m": 3.0, "resolved": False},
        )

        first = registry.call(
            "confirm_candidate",
            {"candidate_id": "cand_1", "instruction": "the wooden cabinet"},
            ctx,
        )
        second = registry.call(
            "confirm_candidate",
            {"candidate_id": "cand_1", "instruction": "the wooden cabinet"},
            ctx,
        )

        self.assertEqual(first.data["reason"], "candidate_distance_uncertain")
        self.assertEqual(second.status, STATUS_BLOCKED)
        self.assertEqual(second.data["reason"], "confirm_stalled")
        self.assertIs(second.data["stalled"], True)
        self.assertIs(second.data["candidate_needs_reground"], True)
        self.assertEqual(second.data["next_action"]["tool"], "ground_target")
        self.assertEqual(record["status"], "rejected")
        self.assertEqual(record["needs_reground"], True)
        self.assertEqual(vlm.calls, [])

    def test_the_prompt_asks_for_the_objects_box(self):
        from vlm_nav.skills_confirm import CONFIRM_SYSTEM_PROMPT

        self.assertIn('"box"', CONFIRM_SYSTEM_PROMPT)
        self.assertIn("0-1000", CONFIRM_SYSTEM_PROMPT)
        self.assertIn("null", CONFIRM_SYSTEM_PROMPT.lower())

    def test_the_confirmation_prompt_does_not_example_a_zero_confidence(self):
        """The v8 prompt's own ``"confidence": 0.0`` example was copied 15/16x."""
        from vlm_nav.skills_confirm import CONFIRM_SYSTEM_PROMPT

        self.assertNotIn('"confidence": 0.0', CONFIRM_SYSTEM_PROMPT)
        self.assertIn('"confidence": 0.7', CONFIRM_SYSTEM_PROMPT)
        self.assertIn("null", CONFIRM_SYSTEM_PROMPT)
        self.assertIn("certain", CONFIRM_SYSTEM_PROMPT.lower())

    def test_null_and_zero_confidence_are_parsed_apart(self):
        """null/missing = undecided; an explicit <0.5 = low; 0.7 = usable."""
        from vlm_nav.skills_confirm import _parse_reply

        base = {
            "target": True,
            "same_instance": True,
            "attributes_match": True,
            "relation_supported": True,
        }
        undecided = _parse_reply(json.dumps(dict(base, confidence=None)))
        missing = _parse_reply(json.dumps(base))
        zero = _parse_reply(json.dumps(dict(base, confidence=0.0)))
        sure = _parse_reply(json.dumps(dict(base, confidence=0.7)))

        self.assertIsNone(undecided["confidence"])
        self.assertEqual(undecided["confidence_state"], "undecided")
        self.assertIsNone(missing["confidence"])
        self.assertEqual(missing["confidence_state"], "undecided")
        self.assertEqual(zero["confidence"], 0.0)
        self.assertEqual(zero["confidence_state"], "low")
        self.assertEqual(sure["confidence"], 0.7)
        self.assertEqual(sure["confidence_state"], "ok")

    def test_an_undecided_confidence_view_neither_supports_nor_vetoes(self):
        """null confidence cannot carry an acceptance, but it is not a veto."""
        undecided = dict(ACCEPTED, confidence=None, reason="cannot decide from this view")
        vlm = ScriptedVLM([ACCEPTED, undecided])
        world = ConfirmWorld(position=(1.8, 0.0, 0.0))
        ctx = make_context(world, vlm)
        registry = make_registry(vlm)
        store_candidate(ctx)

        result = registry.call(
            "confirm_candidate",
            {"candidate_id": "cand_1", "instruction": "the wooden cabinet"},
            ctx,
        )

        self.assertEqual(result.data["verdict"], "accepted", result.data)
        self.assertEqual(result.data["confirmation_count"], 1)
        states = [view.get("confidence_state") for view in result.data["views"]]
        self.assertIn("undecided", states, result.data["views"])
        # the acceptance came from the decided view, not from the undecided one
        self.assertEqual(states.count("ok"), 1, result.data["views"])

    def test_an_undecided_view_alone_cannot_accept(self):
        """With no decided-supporting view the identity stays open."""
        undecided = dict(ACCEPTED, confidence=None, reason="cannot decide from this view")
        vlm = ScriptedVLM([undecided, undecided])
        world = ConfirmWorld(position=(1.8, 0.0, 0.0))
        ctx = make_context(world, vlm)
        registry = make_registry(vlm)
        store_candidate(ctx)

        result = registry.call(
            "confirm_candidate",
            {"candidate_id": "cand_1", "instruction": "the wooden cabinet"},
            ctx,
        )

        self.assertEqual(result.data["verdict"], "uncertain", result.data)
        self.assertEqual(result.data["confirmation_count"], 0)
        self.assertIn("identity", result.data["missing"])
        self.assertNotEqual(
            ctx.extra["perception.candidates"]["cand_1"].get("status"), "rejected"
        )

    def test_unknown_distance_asks_for_distance_not_a_global_search(self):
        world = ConfirmWorld()
        # a ray only and no depth sensor: the range cannot become known, so
        # confirmation has to ask for distance instead of guessing identity
        world.depth_at_pixel = None
        ctx = make_context(world)
        registry = make_registry(ScriptedVLM([ACCEPTED]))
        store_candidate(
            ctx, position=None,
            ray={"origin_flu": [0.0, 0.0, 0.0], "bearing_deg": 90.0, "elevation_deg": 0.0},
        )

        result = registry.call("confirm_candidate", {"candidate_id": "cand_1"}, ctx)

        self.assertIn(result.data["verdict"], ("uncertain", "rejected"))
        self.assertIn("distance", result.data["missing"])
        self.assertNotEqual(result.data["next_action"]["tool"], "search_target")
        self.assertIn(result.data["next_action"]["tool"], ("approach_candidate", "observe"))
        self.assertNotEqual(
            ctx.extra["perception.candidates"]["cand_1"].get("status"), "rejected"
        )

    def test_a_missing_relation_asks_for_relation_evidence(self):
        vlm = ScriptedVLM([RELATION_UNRESOLVED])
        ctx = make_context(ConfirmWorld(), vlm)
        registry = make_registry(vlm)
        store_candidate(ctx)

        result = registry.call(
            "confirm_candidate",
            {"candidate_id": "cand_1", "instruction": "the cabinet next to the wooden door"},
            ctx,
        )

        self.assertEqual(result.data["verdict"], "uncertain")
        self.assertIn("relation", result.data["missing"])
        self.assertEqual(result.data["next_action"]["tool"], "confirm_candidate")
        self.assertGreaterEqual(result.data["next_action"]["args"]["turn_deg"], 35.0)
        self.assertNotEqual(
            ctx.extra["perception.candidates"]["cand_1"].get("status"), "rejected"
        )

    def test_a_blocked_path_keeps_the_candidate_and_changes_the_route(self):
        vlm = ScriptedVLM([ACCEPTED])
        world = ConfirmWorld(blocked=True)
        ctx = make_context(world, vlm)
        registry = make_registry(vlm)
        store_candidate(ctx, position=(4.0, 0.0, 0.0))

        result = registry.call("confirm_candidate", {"candidate_id": "cand_1"}, ctx)

        self.assertEqual(result.data["verdict"], "uncertain")
        self.assertIn("path", result.data["missing"])
        self.assertNotEqual(result.data["next_action"]["tool"], "search_target")
        self.assertNotEqual(
            ctx.extra["perception.candidates"]["cand_1"].get("status"), "rejected"
        )

    def test_arrival_plus_accepted_confirmation_backs_finish(self):
        vlm = ScriptedVLM([ACCEPTED])
        ctx = make_context(ConfirmWorld(), vlm)
        registry = make_registry(vlm)
        store_candidate(ctx)

        arrived = registry.call("approach_candidate", {"candidate_id": "cand_1"}, ctx)
        self.assertEqual(arrived.data["outcome"], "arrived", arrived.data)
        # the arrival names the confirmation as its next step
        self.assertEqual(arrived.data["next_action"]["tool"], "confirm_candidate")
        self.assertIs(arrived.data["confirmation_required"], True)

        confirmed = registry.call("confirm_candidate", {"candidate_id": "cand_1"}, ctx)
        self.assertEqual(confirmed.data["verdict"], "accepted")
        finish = registry.call(
            "finish", {"reason": "arrived and confirmed", "success": True}, ctx
        )

        self.assertEqual(finish.status, STATUS_OK, finish.data)
        self.assertIs(finish.data["success"], True)
        self.assertEqual(finish.data["missing"], [])
        self.assertIs(finish.data["confirmation_required"], True)
        self.assertIs(finish.data["confirmation_accepted"], True)
        self.assertTrue(evidence_pack(ctx)["confirmations"])

    def test_reaching_without_confirmation_is_reported_as_missing_identity(self):
        ctx = make_context(ConfirmWorld())
        registry = make_registry(ScriptedVLM([ACCEPTED]))
        store_candidate(ctx)

        registry.call("approach_candidate", {"candidate_id": "cand_1"}, ctx)
        finish = registry.call(
            "finish", {"reason": "arrived", "success": True}, ctx
        )

        self.assertEqual(finish.status, STATUS_BLOCKED)
        self.assertIs(finish.data["success"], False)
        self.assertIn("identity confirmation", " ".join(finish.data["missing"]))

    def test_a_completed_movement_plan_still_needs_the_confirmation(self):
        from vlm_nav.tool_impl_task import executor_step, register_task_tools

        vlm = ScriptedVLM([ACCEPTED])
        ctx = make_context(ConfirmWorld(), vlm)
        registry = make_registry(vlm)
        register_task_tools(registry)
        store_candidate(ctx)

        plan = registry.call(
            "update_plan",
            {
                "steps": [
                    {
                        "id": "st_1",
                        "goal": "approach the cabinet candidate",
                        "done_when": "the cabinet candidate is reached",
                        "next_step": "approach_candidate",
                    }
                ]
            },
            ctx,
        )
        self.assertEqual(plan.status, STATUS_OK, plan.data)
        arrived = registry.call("approach_candidate", {"candidate_id": "cand_1"}, ctx)
        self.assertEqual(arrived.data["outcome"], "arrived")
        approach_data = dict(arrived.data)
        approach_data["subgoal_complete"] = True  # the executor's task signal
        outcome = executor_step(
            ctx,
            {
                "tool": "approach_candidate",
                "args": {"candidate_id": "cand_1"},
                "status": arrived.status,
                "data": approach_data,
                "evidence": [ref.to_dict() for ref in arrived.evidence],
                "turn": 1,
            },
        )
        # reach alone only claims the stage: the identity is still open
        self.assertEqual(outcome["completed"], "", outcome)
        self.assertEqual(outcome["claimed"], "st_1", outcome)
        self.assertIn("confirmation", outcome["evidence_missing"], outcome)

        without = registry.call(
            "finish", {"reason": "plan complete", "success": True}, ctx
        )
        self.assertEqual(without.status, STATUS_BLOCKED)
        self.assertIn("identity confirmation", " ".join(without.data["missing"]))

        confirmed = registry.call("confirm_candidate", {"candidate_id": "cand_1"}, ctx)
        self.assertEqual(confirmed.data["verdict"], "accepted")
        closed = executor_step(
            ctx,
            {
                "tool": "confirm_candidate",
                "args": {"candidate_id": "cand_1"},
                "status": confirmed.status,
                "data": dict(confirmed.data),
                "evidence": [ref.to_dict() for ref in confirmed.evidence],
                "turn": 2,
            },
        )
        self.assertEqual(closed["completed"], "st_1", closed)
        finish = registry.call(
            "finish", {"reason": "plan complete and confirmed", "success": True}, ctx
        )
        self.assertEqual(finish.status, STATUS_OK, finish.data)
        self.assertIs(finish.data["success"], True)
        self.assertIs(finish.data["confirmation_required"], True)
        self.assertIs(finish.data["confirmation_accepted"], True)

    def test_a_confirmation_without_a_depth_measurement_is_not_accepted(self):
        """Appearance alone cannot accept: no position evidence -> uncertain."""
        vlm = ScriptedVLM([ACCEPTED])
        world = ConfirmWorld(position=(1.8, 0.0, 0.0))
        world.depth_at_pixel = None  # the world has no depth sensor at all
        ctx = make_context(world, vlm)
        registry = make_registry(vlm)
        store_candidate(ctx)

        result = registry.call(
            "confirm_candidate",
            {"candidate_id": "cand_1", "instruction": "the wooden cabinet"},
            ctx,
        )

        self.assertEqual(result.data["verdict"], "uncertain", result.data)
        self.assertNotEqual(result.data["verdict"], "accepted")
        self.assertIn("localization", result.data["missing"])
        self.assertIn("distance", result.data["missing"])
        self.assertIs(result.data["position_check"]["available"], False)
        self.assertIs(result.data["new_confirmation"], False)
        self.assertEqual(result.data["confirmation_count"], 0)

    def test_a_position_inconsistent_confirmation_is_rejected(self):
        """A fresh depth point 2+ m from the candidate is a different surface."""
        vlm = ScriptedVLM([ACCEPTED])
        world = ConfirmWorld(position=(1.8, 0.0, 0.0))
        world.depth = 3.3  # candidate is 1.2 m ahead; the ray hits a wall at 3.3 m
        ctx = make_context(world, vlm)
        registry = make_registry(vlm)
        store_candidate(ctx)

        result = registry.call(
            "confirm_candidate",
            {"candidate_id": "cand_1", "instruction": "the wooden cabinet"},
            ctx,
        )

        self.assertEqual(result.data["verdict"], "rejected", result.data)
        self.assertEqual(result.data["reason"], "position_inconsistent")
        self.assertEqual(
            ctx.extra["perception.candidates"]["cand_1"].get("status"), "rejected"
        )

    def test_a_zero_confidence_view_blocks_acceptance(self):
        """Every independent view has to support; a 0.0 view cannot be ignored."""
        zero = dict(ACCEPTED, confidence=0.0, reason="cannot tell from here")
        vlm = ScriptedVLM([ACCEPTED, zero])
        world = ConfirmWorld(position=(1.8, 0.0, 0.0))
        ctx = make_context(world, vlm)
        registry = make_registry(vlm)
        store_candidate(ctx)

        result = registry.call(
            "confirm_candidate",
            {"candidate_id": "cand_1", "instruction": "the wooden cabinet"},
            ctx,
        )

        self.assertEqual(result.data["verdict"], "uncertain", result.data)
        self.assertNotEqual(result.data["verdict"], "accepted")
        self.assertEqual(result.data["confirmation_count"], 0)
        self.assertIn("identity", result.data["missing"])
        self.assertGreaterEqual(len(result.data["views"]), 2, result.data["views"])
        states = [view.get("confidence_state") for view in result.data["views"]]
        self.assertIn("low", states, result.data["views"])

    def test_an_opposing_view_prevents_acceptance(self):
        """One view naming a different instance is at least uncertain/rejected."""
        vlm = ScriptedVLM([ACCEPTED, WRONG_INSTANCE])
        world = ConfirmWorld(position=(1.8, 0.0, 0.0))
        ctx = make_context(world, vlm)
        registry = make_registry(vlm)
        store_candidate(ctx)

        result = registry.call(
            "confirm_candidate",
            {"candidate_id": "cand_1", "instruction": "the wooden cabinet"},
            ctx,
        )

        self.assertEqual(result.data["verdict"], "rejected", result.data)
        self.assertEqual(result.data["confirmation_count"], 0)
        self.assertEqual(
            ctx.extra["perception.candidates"]["cand_1"].get("status"), "rejected"
        )
        # every terminal rejection is remembered, including different_instance
        rejections = ctx.extra.get("perception.triangulation_rejections") or []
        self.assertTrue(rejections)
        self.assertEqual(rejections[-1]["reason"], "different_instance")
        self.assertEqual(rejections[-1]["candidate_id"], "cand_1")

    def test_the_relation_prompt_carries_the_full_anchor(self):
        """The view check names the sub-task's whole anchor (object + relation)."""
        from vlm_nav.tool_impl_task import register_task_tools

        vlm = ScriptedVLM([RELATION_UNRESOLVED])
        world = ConfirmWorld(position=(1.8, 0.0, 0.0))
        ctx = make_context(world, vlm)
        registry = make_registry(vlm)
        register_task_tools(registry)
        plan = registry.call(
            "update_plan",
            {
                "steps": [
                    {
                        "id": "st_1",
                        "goal": "再找到木门旁边的柜子",
                        "expected_observation": "木门附近的柜子",
                        "done_when": "柜子被定位并到达",
                    }
                ]
            },
            ctx,
        )
        self.assertEqual(plan.status, STATUS_OK, plan.data)
        store_candidate(ctx)

        result = registry.call("confirm_candidate", {"candidate_id": "cand_1"}, ctx)

        self.assertIn("木门旁边", vlm.calls[0]["prompt"])
        self.assertIn("木门旁边", result.data["relation"])
        self.assertEqual(result.data["relation"], "木门旁边")
        # the anchor travels as its own field, not only inside the long text
        self.assertEqual(vlm.calls[0]["context"]["relation"], "木门旁边")

    def test_a_repeated_confirmation_stalls_with_a_different_action(self):
        """The same pixels cannot be confirmed again: stop with a new strategy."""
        vlm = ScriptedVLM([RELATION_UNRESOLVED])
        world = ConfirmWorld(position=(1.8, 0.0, 0.0), vary_view=False)
        ctx = make_context(world, vlm)
        registry = make_registry(vlm)
        store_candidate(ctx)

        first = registry.call(
            "confirm_candidate",
            {"candidate_id": "cand_1", "instruction": "找到木门旁边的柜子"},
            ctx,
        )
        self.assertEqual(first.data["verdict"], "uncertain", first.data)

        second = registry.call(
            "confirm_candidate",
            {"candidate_id": "cand_1", "instruction": "找到木门旁边的柜子"},
            ctx,
        )

        self.assertEqual(second.status, STATUS_BLOCKED, second.data)
        self.assertEqual(second.data["reason"], "confirm_stalled")
        self.assertEqual(second.data["verdict"], "uncertain")
        self.assertIs(second.data["stalled"], True)
        self.assertTrue(second.data["strategy_change"])
        self.assertNotEqual(second.data["next_action"]["tool"], "confirm_candidate")
        self.assertIn(second.data["next_action"]["tool"], ("observe", "propose_viewpoints"))

    def test_the_confirmation_attempt_cap_stalls_the_loop(self):
        """N confirmations without a new accepted one stop the repeat loop."""
        vlm = ScriptedVLM([RELATION_UNRESOLVED])
        world = ConfirmWorld(position=(1.8, 0.0, 0.0))
        ctx = make_context(world, vlm)
        registry = make_registry(vlm)
        store_candidate(ctx)

        results = [
            registry.call(
                "confirm_candidate",
                {
                    "candidate_id": "cand_1",
                    "instruction": "找到木门旁边的柜子",
                    "max_views": 2,
                    "turn_deg": turn,
                },
                ctx,
            )
            for turn in (25.0, 55.0, 75.0)
        ]

        self.assertEqual(results[0].data["verdict"], "uncertain")
        self.assertEqual(results[1].data["verdict"], "uncertain")
        self.assertEqual(results[2].status, STATUS_BLOCKED, results[2].data)
        self.assertEqual(results[2].data["reason"], "confirm_stalled")
        self.assertEqual(results[2].data["verdict"], "uncertain")
        self.assertTrue(results[2].data["strategy_change"])
        self.assertNotEqual(
            results[2].data["next_action"]["tool"], "confirm_candidate"
        )

    def test_the_mcp_contract_serves_confirm_candidate(self):
        self.assertIn("confirm_candidate", ENV_TOOL_NAMES)
        self.assertIn("confirm_candidate", TOOL_CONTRACTS)
        contract = TOOL_CONTRACTS["confirm_candidate"]
        self.assertTrue(contract["side_effects"])
        self.assertIn("candidate_id", contract["input_schema"]["properties"])
        self.assertEqual(contract["input_schema"]["required"], [])


if __name__ == "__main__":
    unittest.main()
