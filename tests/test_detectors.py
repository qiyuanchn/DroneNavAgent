"""Unit tests for the optional detector gate in ``ground_target``.

These are mechanism tests: a fake detector pins the gate (match -> keep,
authoritative mismatch -> reject, weak proposal -> no rejection, error/empty ->
VLM fallback, detector-only rescue), and the label/probe helpers are checked
without loading any weights. No CUDA, no SAM3, no DINOv2.
"""

import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from vlm_nav.detectors import (
    Detection,
    NullDetector,
    Sam3DinoDetector,
    frame_to_numpy,
    resolve_detector,
    target_labels,
)
from vlm_nav.tool_impl_perception import (
    CANDIDATE_STORE_KEY,
    ground_target,
)
from vlm_nav.tools import ToolContext


class FakeWorld:
    """Kinematic CPU world: a gradient frame and a constant depth."""

    def __init__(self, width=64, height=32, position=(0.0, 0.0, 0.0), yaw_deg=0.0, depth=3.0):
        self.width, self.height = int(width), int(height)
        self.state = SimpleNamespace(
            position_flu=np.asarray(position, dtype=np.float64),
            yaw_rad=math.radians(yaw_deg),
        )
        self.depth = float(depth)
        self.render_calls = 0

    @property
    def position_flu(self):
        return self.state.position_flu

    @property
    def yaw_rad(self):
        return self.state.yaw_rad

    def render(self):
        self.render_calls += 1
        gradient = torch.arange(self.width, dtype=torch.float32) / float(self.width)
        return gradient.reshape(1, 1, -1).expand(3, self.height, self.width).clone()

    def scan_views(self, n_views=8, pitch_down_deg=0.0, spread_deg=None):
        return [self.render() for _ in range(int(n_views))]

    def depth_at_pixel(self, pixel):
        return self.depth


class FakeVLM:
    name = "fake"
    allow_box_retry = True

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def decide(self, frames, system_prompt, user_prompt, context=None):
        self.calls.append(
            {
                "frames": frames,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "context": context,
            }
        )
        return self.reply


class FakeDetector:
    name = "fake"

    def __init__(self, proposals=(), *, available=True, error=None, embeddings=None):
        self.proposals = list(proposals)
        self._available = bool(available)
        self.error = error
        self.embeddings = embeddings
        self.calls = []

    def available(self):
        return self._available

    def detect(self, frame, instruction, *, labels=None, context=None):
        self.calls.append({"instruction": instruction, "labels": list(labels or [])})
        if self.error is not None:
            raise self.error
        return list(self.proposals)

    def embed_boxes(self, frame, boxes):
        return self.embeddings


def make_ctx(detector, **extra):
    return ToolContext(
        world=FakeWorld(),
        episode_id="ep_test",
        extra={"perception.detector": detector, **extra},
    )


def reply(*candidates):
    import json

    return json.dumps({"candidates": list(candidates)})


class FrameConversionTest(unittest.TestCase):
    def test_unit_float_chw_is_scaled_to_uint8_rgb(self):
        frame = torch.tensor(
            [[[0.0, 0.5], [1.0, 0.25]],
             [[0.0, 0.0], [1.0, 1.0]],
             [[1.0, 0.5], [0.0, 0.0]]],
            dtype=torch.float32,
        )
        array = frame_to_numpy(frame)
        self.assertEqual(array.shape, (2, 2, 3))
        self.assertEqual(array.max(), 255)
        self.assertEqual(array[0, 1].tolist(), [127, 0, 127])
        self.assertEqual(array[1, 0].tolist(), [255, 255, 0])


class TargetLabelTest(unittest.TestCase):
    def test_relation_instruction_keeps_the_target_not_the_anchor(self):
        labels = target_labels("书架旁边的黑色办公椅")
        self.assertIn("office chair", labels)
        self.assertNotIn("bookshelf", labels)
        # the attribute-qualified prompt is available for the instance gate
        self.assertIn("black office chair", labels)

    def test_english_instruction_uses_its_noun(self):
        self.assertIn("wooden chair", target_labels("the wooden chair next to the window"))

    def test_whole_multi_stage_instruction_falls_back_to_known_nouns(self):
        labels = target_labels("先确认木椅，再到木椅旁边的灯旁停稳。每段都要确认。")
        self.assertIn("wooden chair", labels)
        self.assertIn("lamp", labels)

    def test_unknown_phrase_is_not_invented(self):
        self.assertEqual(target_labels("看看那个东西"), [])


class ResolveDetectorTest(unittest.TestCase):
    def test_off_is_the_null_backend(self):
        detector = resolve_detector("off")
        self.assertIsInstance(detector, NullDetector)
        self.assertFalse(detector.available())

    def test_sam3_without_checkpoints_is_unavailable(self):
        detector = resolve_detector(
            "sam3", sam3_path="/nonexistent/sam3", dino_path="/nonexistent/dino"
        )
        self.assertIsInstance(detector, Sam3DinoDetector)
        self.assertFalse(detector.available())
        probe = detector.probe()
        self.assertFalse(probe["sam3_exists"])
        self.assertFalse(probe["dino_exists"])

    def test_unknown_spec_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_detector("nonsense")


class DetectorGateTest(unittest.TestCase):
    def test_matching_proposal_marks_the_vlm_candidate_verified(self):
        detector = FakeDetector([Detection([300, 300, 500, 500], 0.9, "chair")])
        ctx = make_ctx(detector)
        vlm = FakeVLM(reply({"box": [300, 300, 500, 500], "attrs": {"shape": "chair"}}))
        result = ground_target(ctx, {"instruction": "find the chair"}, vlm=vlm)

        self.assertEqual(result.status, "ok")
        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["proposal_source"], "vlm+detector")
        self.assertTrue(candidate["detector_verified"])
        self.assertTrue(candidate["detector_match"]["matched"])
        self.assertEqual(result.data["detector"]["status"], "ok")
        self.assertEqual(result.data["detector"]["n_matched"], 1)
        self.assertEqual(detector.calls[0]["labels"], ["chair"])

    def test_authoritative_mismatch_drops_the_vlm_box_and_returns_the_proposal(self):
        detector = FakeDetector([Detection([500, 400, 700, 600], 0.9, "desk")])
        ctx = make_ctx(detector)
        vlm = FakeVLM(reply({"box": [100, 100, 200, 200], "attrs": {"shape": "chair"}}))
        result = ground_target(ctx, {"instruction": "find the desk"}, vlm=vlm)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["n_candidates"], 1)
        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["proposal_source"], "detector")
        self.assertEqual(candidate["box"], [500, 400, 700, 600])
        self.assertTrue(candidate["detector_verified"])
        self.assertFalse(candidate["instance_suspect"])
        self.assertEqual(result.data["detector"]["n_rejected"], 1)
        self.assertEqual(result.data["detector"]["n_detector_candidates"], 1)
        # the rejected VLM box was never put in the candidate store
        self.assertEqual(
            [key for key, value in ctx.extra[CANDIDATE_STORE_KEY].items()
             if value.get("box") == [100, 100, 200, 200]],
            [],
        )

    def test_weak_proposal_does_not_drop_the_vlm_box(self):
        detector = FakeDetector([Detection([500, 400, 700, 600], 0.3, "desk")])
        ctx = make_ctx(detector)
        vlm = FakeVLM(reply({"box": [100, 100, 200, 200], "attrs": {"shape": "chair"}}))
        result = ground_target(ctx, {"instruction": "find the desk"}, vlm=vlm)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["n_candidates"], 1)
        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["proposal_source"], "vlm")
        self.assertEqual(candidate["detector_gate"], "unmatched_low_confidence")
        self.assertEqual(result.data["detector"]["n_rejected"], 0)

    def test_border_large_proposal_cannot_drop_the_vlm_box(self):
        # a high-score mask that runs into the frame border with a large area
        # has no trustworthy range/identity: it must not act as the authority
        detector = FakeDetector([Detection([0, 0, 400, 800], 0.9, "chair")])
        ctx = make_ctx(detector)
        vlm = FakeVLM(reply({"box": [500, 400, 700, 600], "attrs": {"shape": "desk"}}))
        result = ground_target(ctx, {"instruction": "find the desk"}, vlm=vlm)

        self.assertEqual(result.data["n_candidates"], 1)
        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["proposal_source"], "vlm")
        self.assertEqual(candidate["detector_gate"], "unmatched_low_confidence")
        self.assertEqual(result.data["detector"]["n_rejected"], 0)

    def test_error_falls_back_to_the_vlm_reply(self):
        detector = FakeDetector(error=RuntimeError("boom"))
        ctx = make_ctx(detector)
        vlm = FakeVLM(reply({"box": [300, 300, 500, 500], "attrs": {"shape": "chair"}}))
        result = ground_target(ctx, {"instruction": "find the chair"}, vlm=vlm)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["n_candidates"], 1)
        self.assertEqual(result.data["candidates"][0]["proposal_source"], "vlm")
        self.assertEqual(result.data["detector"]["status"], "error")
        self.assertIn("boom", result.data["detector"]["error"])

    def test_empty_proposal_set_falls_back_to_the_vlm_reply(self):
        detector = FakeDetector([])
        ctx = make_ctx(detector)
        vlm = FakeVLM(reply({"box": [300, 300, 500, 500], "attrs": {"shape": "chair"}}))
        result = ground_target(ctx, {"instruction": "find the chair"}, vlm=vlm)

        self.assertEqual(result.data["detector"]["status"], "no_proposal")
        self.assertEqual(result.data["candidates"][0]["proposal_source"], "vlm")

    def test_dino_similarity_can_reject_a_geometric_match(self):
        # same box, orthogonal crop features: the feature half of the gate says
        # these crops are not the same instance
        detector = FakeDetector(
            [Detection([300, 300, 500, 500], 0.9, "desk")],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )
        ctx = make_ctx(detector)
        vlm = FakeVLM(reply({"box": [300, 300, 500, 500], "attrs": {"shape": "chair"}}))
        result = ground_target(ctx, {"instruction": "find the desk"}, vlm=vlm)

        candidate = result.data["candidates"][0]
        # the VLM box is rejected; the detector's own proposal is offered
        self.assertEqual(candidate["proposal_source"], "detector")
        self.assertEqual(result.data["detector"]["n_rejected"], 1)

    def test_detector_rescues_the_empty_vlm_reply(self):
        detector = FakeDetector([Detection([500, 400, 700, 600], 0.9, "desk")])
        ctx = make_ctx(detector)
        vlm = FakeVLM('{"candidates": []}')
        vlm.allow_box_retry = False
        result = ground_target(ctx, {"instruction": "find the desk"}, vlm=vlm)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["n_candidates"], 1)
        self.assertEqual(result.data["candidates"][0]["proposal_source"], "detector")
        self.assertEqual(result.data["detector"]["status"], "ok")

    def test_detector_candidate_survives_the_attribute_gate(self):
        # the instruction names a colour, and the detector candidate carries
        # only its text label: the text-conditioned proposal is the verification
        detector = FakeDetector([Detection([500, 400, 700, 600], 0.9, "black office chair")])
        ctx = make_ctx(detector)
        vlm = FakeVLM('{"candidates": []}')
        vlm.allow_box_retry = False
        result = ground_target(ctx, {"instruction": "find the black office chair"}, vlm=vlm)

        candidate = result.data["candidates"][0]
        self.assertFalse(candidate["instance_suspect"])
        self.assertIn(candidate["id"], result.data["instance_filter"]["approvable"])


if __name__ == "__main__":
    unittest.main()
