"""Mechanism tests for the relation chain (window -> wooden chair).

The milestone replaces the stateless co-visibility gate with a maintained
object/relation state:

- instance hypotheses associate cand_N across frames (and keep a small set of
  competing hypotheses instead of condemning every alternative);
- relation evidence is typed (supports / contradicts / unknown) and sourced
  (co-visible / cross-frame pose / model self-report); only an explicit
  contradiction rejects, while an anchor outside the frame is unknown;
- a missing anchor auto-creates a dependent "find and locate the anchor"
  sub-task that preserves the target's candidates, evidence and budget;
- the observation planner maps the five uncertainty types to evidence actions
  and journals what each observation tried to distinguish.

No CUDA, no model weights: fake world / fake context only.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from vlm_nav.tool_impl_perception import (
    RELATION_CONTRADICTS,
    RELATION_SUPPORTS,
    RELATION_UNKNOWN,
    relation_chain_enabled,
    relation_evidence_for_candidate,
    relation_observation_plan,
    relation_state,
    relation_state_summary,
    update_instance_hypotheses,
    mark_candidate_rejected,
    rank_observation_actions,
    record_observation_decision,
    typed_uncertainties,
    UNCERTAINTY_ANCHOR_RELATION,
    UNCERTAINTY_CATEGORY,
    UNCERTAINTY_DISTANCE,
    UNCERTAINTY_INSTANCE,
    UNCERTAINTY_REACHABILITY,
)
from vlm_nav.tool_impl_task import (
    SUBTASK_ACTIVE,
    SUBTASK_PENDING,
    ensure_anchor_subtask,
    plan_state,
    plan_subtasks,
    update_plan,
)
from vlm_nav.tools import ToolContext, ToolRegistry


class FakeWorld:
    def __init__(self, position=(0.0, 0.0, 0.0), yaw_deg=0.0):
        self.state = type("State", (), {})()
        self.state.position_flu = np.asarray(position, dtype=np.float64)
        self.state.yaw_rad = math.radians(yaw_deg)
        self.width, self.height, self.hfov_deg = 320, 180, 90.0

    @property
    def position_flu(self):
        return self.state.position_flu

    @property
    def yaw_rad(self):
        return self.state.yaw_rad

    def render(self):
        frame = torch.zeros((3, 8, 8), dtype=torch.float32)
        frame[0, 0, 0] = 0.25
        frame[2, 0, 0] = float(self.state.position_flu[0]) / 100.0
        return frame


def make_ctx(*, relation_chain=True, position=(0.0, 0.0, 0.0)):
    extra = {}
    if relation_chain:
        extra["relation_chain"] = True
    return ToolContext(
        world=FakeWorld(position=position),
        episode_id="ep_relation",
        budget={"tool_calls_left": 20},
        extra=extra,
    )


def candidate(
    candidate_id,
    *,
    position=(2.0, 0.0, 0.0),
    observation_id="obs_target",
    subtask_key="stage:target",
    instruction="窗边的木椅",
    attrs=None,
    confidence=0.8,
    box=None,
):
    record = {
        "id": candidate_id,
        "source": "ground_target",
        "box": box or [450.0, 400.0, 550.0, 700.0],
        "box_format": "xyxy",
        "attrs": {"material": "wood", **(attrs or {})},
        "confidence": confidence,
        "observation_id": observation_id,
        "instruction": instruction,
        "subtask_key": subtask_key,
        "position_source": "depth",
        "range_m": 2.0,
        "size_m": 0.9,
    }
    if position is not None:
        record["position_flu"] = [float(value) for value in position]
    return record


def store_anchor(ctx, *, position=(0.0, 2.0, 0.0), category="window", anchor_id="cand_anchor"):
    record = {
        "id": anchor_id,
        "source": "ground_target",
        "box": [100.0, 200.0, 300.0, 500.0],
        "box_format": "xyxy",
        "attrs": {"object_type": category},
        "confidence": 0.9,
        "observation_id": "obs_anchor",
        "instruction": f"找到并定位{category}",
        "subtask_key": "stage:anchor",
        "position_source": "depth",
        "range_m": 2.0,
    }
    if position is not None:
        record["position_flu"] = [float(value) for value in position]
    ctx.extra.setdefault("perception.candidates", {})[anchor_id] = record
    return record


def store_ray_anchor(
    ctx,
    *,
    origin=(0.0, 0.0, 0.0),
    bearing_deg=90.0,
    elevation_deg=0.0,
    category="window",
    anchor_id="cand_anchor",
    observation_id="obs_anchor",
):
    """An anchor sighted as a bearing only: no range, no world point."""
    record = {
        "id": anchor_id,
        "source": "ground_target",
        "box": [100.0, 200.0, 300.0, 500.0],
        "box_format": "xyxy",
        "attrs": {"object_type": category},
        "confidence": 0.9,
        "observation_id": observation_id,
        "instruction": f"找到并定位{category}",
        "subtask_key": "stage:anchor",
        "position_source": "ray",
        "range_status": "unknown",
        "ray": {
            "origin_flu": [float(value) for value in origin],
            "bearing_deg": float(bearing_deg),
            "elevation_deg": float(elevation_deg),
            "yaw_deg": 0.0,
        },
    }
    ctx.extra.setdefault("perception.candidates", {})[anchor_id] = record
    return record


class SwitchTest(unittest.TestCase):
    def test_switch_is_off_by_default_and_on_when_published(self):
        plain = ToolContext(world=FakeWorld(), extra={})
        self.assertFalse(relation_chain_enabled(plain))
        enabled = make_ctx()
        self.assertTrue(relation_chain_enabled(enabled))


class InstanceHypothesisTest(unittest.TestCase):
    def test_two_frames_of_one_chair_are_one_instance(self):
        ctx = make_ctx()
        first = candidate("cand_1", position=(2.0, 0.0, 0.0))
        second = candidate("cand_2", position=(2.6, 0.2, 0.0), observation_id="obs_2")
        summary = update_instance_hypotheses(ctx, None, "窗边的木椅", [first, second])
        self.assertTrue(summary["enabled"])
        self.assertEqual(first["instance_hypothesis"], second["instance_hypothesis"])
        state = relation_state(ctx)
        instance = state["instances"][first["instance_hypothesis"]]
        self.assertEqual(instance["candidate_ids"], ["cand_1", "cand_2"])
        self.assertEqual(len(state["instances"]), 1)

    def test_two_disjoint_boxes_on_one_frame_are_competing_hypotheses(self):
        ctx = make_ctx()
        first = candidate("cand_1", position=(2.0, 0.0, 0.0), observation_id="obs_1")
        second = candidate("cand_2", position=(5.0, 0.0, 0.0), observation_id="obs_1")
        update_instance_hypotheses(ctx, None, "窗边的木椅", [first, second])
        self.assertNotEqual(first["instance_hypothesis"], second["instance_hypothesis"])
        state = relation_state(ctx)
        competitors = state["competitors"]["stage:target"]
        self.assertEqual(len(competitors), 2)

    def test_at_most_three_competing_hypotheses_stay_active(self):
        ctx = make_ctx()
        candidates = [
            candidate(f"cand_{index}", position=(index * 4.0, 0.0, 0.0))
            for index in range(1, 5)
        ]
        update_instance_hypotheses(ctx, None, "窗边的木椅", candidates)
        state = relation_state(ctx)
        self.assertEqual(len(state["instances"]), 4)
        self.assertEqual(len(state["competitors"]["stage:target"]), 3)

    def test_rejecting_one_hypothesis_switches_to_the_next(self):
        ctx = make_ctx()
        first = candidate("cand_1", position=(2.0, 0.0, 0.0))
        second = candidate("cand_2", position=(6.0, 0.0, 0.0))
        update_instance_hypotheses(ctx, None, "窗边的木椅", [first, second])
        switch = mark_candidate_rejected(ctx, "cand_1", "different_instance")
        self.assertEqual(switch["rejected_candidates"], ["cand_1"])
        self.assertEqual(switch["switched_candidates"], ["cand_2"])
        state = relation_state(ctx)
        rejected = state["instances"][first["instance_hypothesis"]]
        self.assertEqual(rejected["status"], "rejected")
        switched = state["instances"][second["instance_hypothesis"]]
        self.assertEqual(switched["rank"], 1)
        # completed exploration is not discarded
        self.assertEqual(rejected["candidate_ids"], ["cand_1"])


class RelationEvidenceTest(unittest.TestCase):
    def test_cross_frame_relation_supports_when_anchor_left_the_frame(self):
        ctx = make_ctx()
        store_anchor(ctx, position=(0.0, 2.0, 0.0))
        chair = candidate("cand_1", position=(2.0, 0.0, 0.0))
        evidence = relation_evidence_for_candidate(
            ctx, chair, "窗边的木椅",
            view={"anchor_visible": False, "relation_supported": False, "confidence": 0.6},
        )
        self.assertEqual(evidence["state"], RELATION_SUPPORTS)
        self.assertEqual(evidence["source"], "cross_frame_pose")
        self.assertEqual(evidence["current_frame_state"], RELATION_UNKNOWN)
        self.assertFalse(evidence["reason"] == "anchor_visible_but_relation_unsupported")

    def test_anchor_out_of_frame_without_history_is_unknown_not_negative(self):
        ctx = make_ctx()
        chair = candidate("cand_1", position=(2.0, 0.0, 0.0))
        evidence = relation_evidence_for_candidate(
            ctx, chair, "窗边的木椅",
            view={"anchor_visible": False, "relation_supported": None, "confidence": 0.6},
        )
        self.assertEqual(evidence["state"], RELATION_UNKNOWN)
        self.assertIn(evidence["reason"], ("anchor_not_in_frame", "anchor_not_established"))
        self.assertFalse(evidence["state"] == RELATION_CONTRADICTS)

    def test_candidate_measured_outside_the_anchor_bound_contradicts(self):
        ctx = make_ctx()
        store_anchor(ctx, position=(0.0, 2.0, 0.0))
        chair = candidate("cand_1", position=(7.0, 0.0, 0.0))
        evidence = relation_evidence_for_candidate(ctx, chair, "窗边的木椅")
        self.assertEqual(evidence["state"], RELATION_CONTRADICTS)
        self.assertEqual(evidence["reason"], "anchor_distance")
        self.assertGreater(evidence["distance_m"], evidence["limit_m"])

    def test_an_estimated_anchor_position_cannot_contradict(self):
        """A border-large window at the model's own range is not a measurement."""
        ctx = make_ctx()
        anchor = store_anchor(ctx, position=(0.0, 2.0, 0.0))
        anchor["position_source"] = "model_distance"
        anchor["border_large_box"] = True
        chair = candidate("cand_1", position=(7.0, 0.0, 0.0))
        evidence = relation_evidence_for_candidate(ctx, chair, "窗边的木椅")
        self.assertEqual(evidence["state"], RELATION_UNKNOWN, evidence)
        self.assertEqual(evidence["reason"], "anchor_distance_uncertain")
        self.assertFalse(evidence["state"] == RELATION_CONTRADICTS)

    def test_a_ray_only_anchor_is_unknown_and_asks_for_parallax(self):
        """A bearing cannot place the anchor: the relation is unknown (never
        negative) and the evidence names the parallax that would locate it."""
        ctx = make_ctx()
        store_ray_anchor(ctx, origin=(0.0, 0.0, 0.0), bearing_deg=90.0)
        chair = candidate("cand_1", position=(2.0, 0.0, 0.0))
        evidence = relation_evidence_for_candidate(ctx, chair, "窗边的木椅")
        self.assertEqual(evidence["state"], RELATION_UNKNOWN, evidence)
        self.assertEqual(evidence["reason"], "anchor_ray_only")
        self.assertFalse(evidence["state"] == RELATION_CONTRADICTS)
        self.assertIs(evidence["anchor_ray_only"], True)
        parallax = evidence["parallax"]
        self.assertIs(parallax["required"], True)
        self.assertGreaterEqual(parallax["min_baseline_m"], 1.0)
        self.assertGreaterEqual(parallax["min_angle_deg"], 10.0)
        self.assertEqual(parallax["reason"], "anchor_ray_only")

    def test_two_anchor_bearings_triangulate_and_the_relation_is_rechecked(self):
        """Two bearings from different poses intersect at the anchor; the
        located anchor then supports a candidate inside the relation bound."""
        ctx = make_ctx()
        # anchor at (3, 3, 0): 45 deg from the origin, 18.4 deg from (0, 2, 0)
        store_ray_anchor(
            ctx, anchor_id="cand_a1", origin=(0.0, 0.0, 0.0), bearing_deg=45.0
        )
        store_ray_anchor(
            ctx, anchor_id="cand_a2", origin=(0.0, 2.0, 0.0), bearing_deg=18.435,
            observation_id="obs_anchor_2",
        )
        chair = candidate("cand_1", position=(3.5, 1.5, 0.0))
        evidence = relation_evidence_for_candidate(ctx, chair, "窗边的木椅")
        self.assertEqual(evidence["state"], RELATION_SUPPORTS, evidence)
        self.assertEqual(evidence["source"], "cross_frame_pose")
        self.assertLessEqual(evidence["distance_m"], evidence["limit_m"])
        self.assertNotEqual(evidence["reason"], "anchor_ray_only")
        state = relation_state(ctx)
        located = [
            anchor
            for anchor in state["relations"].values()
            if anchor.get("anchor_position") is not None
        ]
        self.assertTrue(located, state)
        self.assertAlmostEqual(located[0]["anchor_position"][0], 3.0, places=1)
        self.assertAlmostEqual(located[0]["anchor_position"][1], 3.0, places=1)

    def test_parallel_anchor_bearings_do_not_triangulate(self):
        """Two rays with no angle between them are not parallax: the anchor
        stays unlocated and the relation stays unknown."""
        ctx = make_ctx()
        store_ray_anchor(ctx, anchor_id="cand_a1", origin=(0.0, 0.0, 0.0), bearing_deg=90.0)
        store_ray_anchor(
            ctx, anchor_id="cand_a2", origin=(1.0, 0.0, 0.0), bearing_deg=90.0,
            observation_id="obs_anchor_2",
        )
        chair = candidate("cand_1", position=(2.0, 0.0, 0.0))
        evidence = relation_evidence_for_candidate(ctx, chair, "窗边的木椅")
        self.assertEqual(evidence["state"], RELATION_UNKNOWN, evidence)
        self.assertEqual(evidence["reason"], "anchor_ray_only")
        self.assertIs(evidence["parallax"]["required"], True)

    def test_seen_anchor_without_the_relation_contradicts(self):
        ctx = make_ctx()
        chair = candidate("cand_1", position=(2.0, 0.0, 0.0))
        evidence = relation_evidence_for_candidate(
            ctx, chair, "窗边的木椅",
            view={"anchor_visible": True, "relation_supported": False, "confidence": 0.8},
        )
        self.assertEqual(evidence["state"], RELATION_CONTRADICTS)
        self.assertEqual(evidence["reason"], "anchor_visible_but_relation_unsupported")

    def test_a_model_anchor_visibility_supports_the_relation(self):
        """A first-stage anchor with no located instance is model self-report."""
        ctx = make_ctx()
        chair = candidate("cand_1", position=(2.0, 0.0, 0.0))
        evidence = relation_evidence_for_candidate(
            ctx, chair, "窗边的木椅",
            visibility={"visible": True, "source": "model"},
        )
        self.assertEqual(evidence["state"], RELATION_SUPPORTS, evidence)
        self.assertEqual(evidence["source"], "model_self_report")

    def test_a_perceptual_relation_goes_unknown_when_stale(self):
        ctx = make_ctx()
        ctx.extra["time_s"] = 0.0
        chair = candidate("cand_1", position=(2.0, 0.0, 0.0))
        fresh = relation_evidence_for_candidate(
            ctx, chair, "窗边的木椅",
            view={"anchor_visible": True, "relation_supported": True, "confidence": 0.9},
        )
        self.assertEqual(fresh["state"], RELATION_SUPPORTS)
        self.assertEqual(fresh["source"], "co_visible")
        ctx.extra["time_s"] = 500.0
        stale = relation_evidence_for_candidate(
            ctx, chair, "窗边的木椅",
            view={"anchor_visible": False, "relation_supported": None, "confidence": 0.3},
        )
        self.assertEqual(stale["state"], RELATION_UNKNOWN)
        self.assertEqual(stale["reason"], "stale")
        self.assertTrue(stale["historical_supports"])

    def test_a_distant_pose_does_not_invalidate_a_located_anchor(self):
        ctx = make_ctx()
        store_anchor(ctx, position=(0.0, 2.0, 0.0))
        chair = candidate("cand_1", position=(2.0, 0.0, 0.0))
        relation_evidence_for_candidate(ctx, chair, "窗边的木椅")
        ctx.extra["time_s"] = 500.0
        ctx.world.state.position_flu = np.asarray([20.0, 0.0, 0.0])
        evidence = relation_evidence_for_candidate(ctx, chair, "窗边的木椅")
        self.assertEqual(evidence["state"], RELATION_SUPPORTS)
        self.assertEqual(evidence["source"], "cross_frame_pose")

    def test_an_anchor_relocalized_far_away_makes_the_relation_unknown(self):
        ctx = make_ctx()
        store_anchor(ctx, position=(0.0, 2.0, 0.0), anchor_id="cand_anchor_1")
        chair = candidate("cand_1", position=(2.0, 0.0, 0.0))
        relation_evidence_for_candidate(ctx, chair, "窗边的木椅")
        # a second window candidate 4 m from the recorded one
        store_anchor(ctx, position=(0.0, 6.0, 0.0), anchor_id="cand_anchor_2")
        update_instance_hypotheses(
            ctx, None, "找到并定位窗",
            [ctx.extra["perception.candidates"]["cand_anchor_2"]],
        )
        evidence = relation_evidence_for_candidate(ctx, chair, "窗边的木椅")
        self.assertEqual(evidence["state"], RELATION_UNKNOWN)
        self.assertEqual(evidence["reason"], "anchor_relocalized")


class AnchorSubtaskTest(unittest.TestCase):
    def test_missing_anchor_creates_a_dependent_subtask_and_preserves_target(self):
        ctx = make_ctx()
        plan = update_plan(
            ctx,
            {
                "steps": [
                    {
                        "id": "st_1",
                        "goal": "在卧室窗边找到木椅（地标）",
                        "done_when": "木椅被定位并到达",
                    }
                ]
            },
        )
        self.assertEqual(plan.status, "ok", plan.data)
        target = plan_subtasks(ctx)[0]
        target["_evidence_marks"] = ["located"]
        target["spent"] = {"tool_calls": 4, "turns": 2}
        ctx.extra.setdefault("perception.candidates", {})["cand_target"] = candidate(
            "cand_target",
            subtask_key="stage:" + " ".join(str(target.get("goal") or "").lower().split())[:120],
        )

        result = ensure_anchor_subtask(
            ctx, relation={"anchor": "窗", "relation": "边的", "limit_m": 3.0}
        )
        self.assertTrue(result["created"], result)
        subtasks = plan_subtasks(ctx)
        self.assertEqual(len(subtasks), 2)
        anchor = subtasks[0]
        target = subtasks[1]
        self.assertEqual(anchor["id"], result["anchor_subtask"])
        self.assertEqual(anchor["status"], SUBTASK_ACTIVE)
        self.assertIn("窗", anchor["goal"])
        self.assertEqual(target["status"], SUBTASK_PENDING)
        self.assertIn(anchor["id"], target["depends_on"])
        self.assertEqual(target["_evidence_marks"], ["located"])
        self.assertEqual(target["spent"]["tool_calls"], 4)
        self.assertEqual(result["candidates_preserved"], 1)
        self.assertEqual(result["spent_preserved"]["tool_calls"], 4)

        again = ensure_anchor_subtask(
            ctx, relation={"anchor": "窗", "relation": "边的", "limit_m": 3.0}
        )
        self.assertFalse(again["created"])
        self.assertTrue(again.get("existing"))
        self.assertEqual(len(plan_subtasks(ctx)), 2)

    def test_finishing_the_anchor_returns_to_the_original_target(self):
        from vlm_nav.tool_impl_task import executor_step

        ctx = make_ctx()
        update_plan(
            ctx,
            {
                "steps": [
                    {
                        "id": "st_1",
                        "goal": "在卧室窗边找到木椅（地标）",
                        "done_when": "木椅被定位并到达",
                    }
                ]
            },
        )
        target = plan_subtasks(ctx)[0]
        target["_evidence_marks"] = ["located"]
        target["spent"] = {"tool_calls": 4, "turns": 2}
        created = ensure_anchor_subtask(
            ctx, relation={"anchor": "窗", "relation": "边的", "limit_m": 3.0}
        )
        self.assertTrue(created["created"])
        anchor_id = created["anchor_subtask"]

        outcome = executor_step(
            ctx,
            {
                "tool": "ground_target",
                "status": "ok",
                "evidence": [{"kind": "observation", "ref": "obs_anchor_frame"}],
                "data": {
                    "instruction": "找到并定位窗",
                    "candidates": [
                        {"id": "cand_anchor", "position_flu": [0.0, 2.0, 0.0]}
                    ],
                },
            },
        )
        self.assertEqual(outcome["completed"], anchor_id, outcome)
        anchor = next(item for item in plan_subtasks(ctx) if item["id"] == anchor_id)
        target = next(item for item in plan_subtasks(ctx) if item["id"] == "st_1")
        self.assertEqual(anchor["status"], "done")
        self.assertEqual(target["status"], SUBTASK_ACTIVE)
        self.assertEqual(target["_evidence_marks"], ["located"])
        self.assertEqual(target["spent"]["tool_calls"], 4)

    def test_the_target_grounding_that_created_the_anchor_does_not_close_it(self):
        from vlm_nav.tool_impl_task import executor_step

        ctx = make_ctx()
        update_plan(
            ctx,
            {
                "steps": [
                    {
                        "id": "st_1",
                        "goal": "在卧室窗边找到木椅（地标）",
                        "done_when": "木椅被定位并到达",
                    }
                ]
            },
        )
        created = ensure_anchor_subtask(
            ctx, relation={"anchor": "卧室窗", "relation": "边的", "limit_m": 3.0}
        )
        anchor_id = created["anchor_subtask"]
        outcome = executor_step(
            ctx,
            {
                "tool": "ground_target",
                "status": "ok",
                "evidence": [{"kind": "observation", "ref": "obs_chair"}],
                "data": {
                    "instruction": "在卧室窗边找到木椅（地标）",
                    "candidates": [{"id": "cand_1", "position_flu": [2.0, 0.0, 0.0]}],
                },
            },
        )
        anchor = next(item for item in plan_subtasks(ctx) if item["id"] == anchor_id)
        target = next(item for item in plan_subtasks(ctx) if item["id"] == "st_1")
        self.assertEqual(anchor["status"], SUBTASK_ACTIVE, outcome)
        self.assertEqual(target["status"], SUBTASK_PENDING)
        self.assertFalse(outcome["completed"])

        # the anchor's own grounding does close it, and the target resumes
        outcome = executor_step(
            ctx,
            {
                "tool": "ground_target",
                "status": "ok",
                "evidence": [{"kind": "observation", "ref": "obs_window"}],
                "data": {
                    "instruction": "找到并定位卧室窗",
                    "candidates": [{"id": "cand_window", "position_flu": [0.0, 2.0, 0.0]}],
                },
            },
        )
        anchor = next(item for item in plan_subtasks(ctx) if item["id"] == anchor_id)
        target = next(item for item in plan_subtasks(ctx) if item["id"] == "st_1")
        self.assertEqual(anchor["status"], "done", outcome)
        self.assertEqual(target["status"], SUBTASK_ACTIVE)

    def test_a_blocked_anchor_releases_the_original_target(self):
        from vlm_nav.tool_impl_task import executor_step

        ctx = make_ctx()
        update_plan(
            ctx,
            {
                "steps": [
                    {
                        "id": "st_1",
                        "goal": "在卧室窗边找到木椅（地标）",
                        "done_when": "木椅被定位并到达",
                    }
                ]
            },
        )
        created = ensure_anchor_subtask(
            ctx, relation={"anchor": "窗", "relation": "边的", "limit_m": 3.0}
        )
        anchor_id = created["anchor_subtask"]
        outcome = executor_step(
            ctx,
            {
                "tool": "ground_target",
                "status": "blocked",
                "data": {"reason": "stagnant_grounding"},
                "evidence": [{"kind": "observation", "ref": "obs_anchor_frame"}],
            },
        )
        self.assertEqual(outcome["blocked"], anchor_id, outcome)
        self.assertTrue(outcome["anchor_released"]["released"], outcome)
        anchor = next(item for item in plan_subtasks(ctx) if item["id"] == anchor_id)
        target = next(item for item in plan_subtasks(ctx) if item["id"] == "st_1")
        self.assertEqual(anchor["status"], "blocked")
        self.assertEqual(target["status"], SUBTASK_ACTIVE)
        self.assertNotIn(anchor_id, target["depends_on"])
        self.assertEqual(target["_anchor_released"]["reason"], "stagnant_grounding")

    def test_anchor_subtask_is_only_created_with_a_plan(self):
        ctx = make_ctx()
        result = ensure_anchor_subtask(ctx, relation={"anchor": "窗"})
        self.assertFalse(result["created"])
        self.assertEqual(result["reason"], "no_plan")


class ObservationPlannerTest(unittest.TestCase):
    def test_each_uncertainty_maps_to_a_different_evidence_action(self):
        actions = rank_observation_actions(
            [
                {"type": UNCERTAINTY_CATEGORY},
                {"type": UNCERTAINTY_INSTANCE},
                {"type": UNCERTAINTY_DISTANCE},
                {"type": UNCERTAINTY_ANCHOR_RELATION},
                {"type": UNCERTAINTY_REACHABILITY},
            ],
            candidate_ids=["cand_1"],
            competitor_ids=["cand_1", "cand_2"],
            instruction="窗边的木椅",
            relation={"anchor": "窗", "relation": "边的", "limit_m": 3.0},
            candidate_position=(2.0, 0.0, 0.0),
            current_position=(0.0, 0.0, 0.0),
        )
        self.assertEqual(len(actions), 5)
        by_type = {action["uncertainty"]: action for action in actions}
        self.assertEqual(by_type[UNCERTAINTY_INSTANCE]["tool"], "compare_candidates")
        self.assertEqual(by_type[UNCERTAINTY_DISTANCE]["tool"], "ground_target")
        self.assertEqual(by_type[UNCERTAINTY_ANCHOR_RELATION]["tool"], "ground_target")
        self.assertIn("窗", by_type[UNCERTAINTY_ANCHOR_RELATION]["args"]["instruction"])
        self.assertEqual(by_type[UNCERTAINTY_REACHABILITY]["tool"], "propose_viewpoints")
        self.assertEqual(by_type[UNCERTAINTY_CATEGORY]["tool"], "observe")
        scores = [action["score"] for action in actions]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # comparison is the cheapest way to exclude a competitor
        self.assertEqual(actions[0]["uncertainty"], UNCERTAINTY_INSTANCE)

    def test_typed_uncertainties_come_from_the_state_not_a_global_search(self):
        ctx = make_ctx()
        first = candidate("cand_1", position=(2.0, 0.0, 0.0))
        second = candidate("cand_2", position=(6.0, 0.0, 0.0))
        update_instance_hypotheses(ctx, None, "窗边的木椅", [first, second])
        ctx.extra["perception.candidates"] = {"cand_1": first, "cand_2": second}
        unknown_category = typed_uncertainties(
            ctx, candidate_ids=["cand_1"], instruction="那个东西"
        )
        self.assertEqual(unknown_category[0]["type"], UNCERTAINTY_CATEGORY)
        types = {
            item["type"]
            for item in typed_uncertainties(
                ctx, candidate_ids=["cand_1"], instruction="窗边的木椅"
            )
        }
        self.assertIn(UNCERTAINTY_INSTANCE, types)
        self.assertIn(UNCERTAINTY_ANCHOR_RELATION, types)
        # a candidate without a position asks for distance evidence
        first["position_flu"] = None
        types = {
            item["type"]
            for item in typed_uncertainties(
                ctx, candidate_ids=["cand_1"], instruction="窗边的木椅"
            )
        }
        self.assertIn(UNCERTAINTY_DISTANCE, types)

    def test_observation_decision_records_what_it_tried_and_whether_it_changed(self):
        ctx = make_ctx()
        entry = record_observation_decision(
            ctx,
            uncertainty=UNCERTAINTY_ANCHOR_RELATION,
            action={"tool": "ground_target", "args": {"instruction": "找到并定位窗"}},
            tried_to_distinguish="window located vs only named in the instruction",
            before={"verdict": "uncertain"},
            after={"verdict": "accepted", "anchor": "window"},
            changed=True,
        )
        self.assertTrue(entry["changed"])
        self.assertIn("window located", entry["tried_to_distinguish"])
        state = relation_state(ctx)
        self.assertEqual(len(state["observations"]), 1)
        summary = relation_state_summary(ctx)
        self.assertEqual(summary["observation_decisions"]["n"], 1)
        self.assertEqual(summary["observation_decisions"]["changed"], 1)
        self.assertEqual(summary["unknown_counts"][UNCERTAINTY_ANCHOR_RELATION], 1)

    def test_relation_observation_plan_prefers_the_cheapest_exclusion(self):
        ctx = make_ctx()
        first = candidate("cand_1", position=(2.0, 0.0, 0.0))
        second = candidate("cand_2", position=(6.0, 0.0, 0.0))
        update_instance_hypotheses(ctx, None, "窗边的木椅", [first, second])
        ctx.extra["perception.candidates"] = {"cand_1": first, "cand_2": second}
        plan = relation_observation_plan(
            ctx,
            "窗边的木椅",
            candidate_ids=["cand_1", "cand_2"],
        )
        self.assertTrue(plan["enabled"])
        self.assertTrue(plan["actions"])
        self.assertEqual(plan["actions"][0]["uncertainty"], UNCERTAINTY_INSTANCE)
        self.assertEqual(plan["next_action"]["tool"], "compare_candidates")


class RuntimeStateTest(unittest.TestCase):
    def test_runtime_relation_line_is_rendered_only_when_state_exists(self):
        from vlm_nav.runtime import AgentRuntime

        runtime = AgentRuntime(ToolRegistry(), FakeWorld(), None, None)
        runtime.ctx = make_ctx()
        self.assertEqual(runtime._identity_anchor_text(), "n/a")
        state = relation_state(runtime.ctx)
        state["relations"]["rel_1"] = {
            "id": "rel_1",
            "state": RELATION_UNKNOWN,
            "source": "cross_frame_pose",
            "reason": "anchor_not_in_frame",
            "anchor_phrase": "窗",
            "target_instance": "inst_1",
            "anchor_candidates": ["cand_anchor"],
            "evidence_ids": ["obs_1"],
        }
        text = runtime._identity_anchor_text()
        self.assertIn("rel=窗[unknown/cross_frame_pose]", text)
        questions = runtime._open_questions_text(None)
        self.assertIn("rel=窗", questions)


if __name__ == "__main__":
    unittest.main()
