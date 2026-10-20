"""Unit tests for the perception tools (``vlm_nav.tool_impl_perception``).

Fake world / fake VLM / fake memory: no CUDA, no model weights, no 3DGS data.
The world mirrors the real ``world.py`` camera semantics (a ``scan_views`` that
restores yaw, pitched extras appended after the level views) and encodes its
heading into each rendered frame so the observations' recorded headings can be
checked against what was actually rendered.
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from vlm_nav.tool_impl_perception import (
    CANDIDATE_STORE_KEY,
    IMAGE_STORE_KEY,
    TRIANGULATION_REJECTIONS_KEY,
    _remember_rejected_triangulation,
    latest_observation,
    compare_candidates,
    ground_target,
    inspect_region,
    observe,
    observation_pose,
    observation_poses,
    parse_box,
    query_memory,
    register_perception_tools,
    resolve_observation,
    resolve_observation_pose,
)
from vlm_nav.observations import observation_store, register_observation
from vlm_nav.tools import EvidenceRef, ToolContext, ToolRegistry


class FakeWorld:
    """Kinematic CPU world: heading-coded gradient frames + a depth stub."""

    def __init__(
        self, width=64, height=32, position=(0.0, 0.0, 0.0), yaw_deg=30.0, depth=None, hfov_deg=None
    ):
        self.width, self.height = int(width), int(height)
        if hfov_deg is not None:
            # the way the real worlds declare their render FOV (SplatWorldCPU
            # exposes hfov_deg, EnvWorld keeps it on its camera object)
            self.hfov_deg = float(hfov_deg)
        self.state = SimpleNamespace(
            position_flu=np.asarray(position, dtype=np.float64),
            yaw_rad=math.radians(yaw_deg),
        )
        self.depth = 3.0 if depth is None else depth
        self.render_calls = 0
        self.refresh_calls = 0
        self.depth_reads = []
        self.scan_calls = []

    @property
    def position_flu(self):
        return self.state.position_flu

    @property
    def yaw_rad(self):
        return self.state.yaw_rad

    def render(self):
        self.render_calls += 1
        gradient = torch.arange(self.width, dtype=torch.float32) / float(self.width)
        frame = gradient.reshape(1, 1, -1).expand(3, self.height, self.width).clone()
        heading_code = (math.degrees(self.state.yaw_rad) % 360.0) / 360.0
        frame[:, 0, 0] = heading_code  # marker: tests read the rendered heading
        return frame

    def scan_views(self, n_views=8, pitch_down_deg=0.0, spread_deg=None):
        self.scan_calls.append(
            {"n_views": n_views, "pitch_down_deg": pitch_down_deg, "spread_deg": spread_deg}
        )
        yaw0 = self.state.yaw_rad
        views = []
        try:
            for k in range(n_views):
                if spread_deg is None:
                    yaw = yaw0 + 2.0 * math.pi * k / n_views
                else:
                    span = math.radians(spread_deg)
                    yaw = yaw0 - span / 2.0 + span * k / max(1, n_views - 1)
                self.state.yaw_rad = yaw
                views.append(self.render())
            for dyaw in (0.0, math.pi):
                if pitch_down_deg > 0.0:
                    self.state.yaw_rad = yaw0 + dyaw
                    views.append(self.render())
        finally:
            self.state.yaw_rad = yaw0
        return views

    def depth_at_pixel(self, pixel):
        self.depth_reads.append((float(pixel[0]), float(pixel[1])))
        if callable(self.depth):
            return self.depth(pixel)
        return self.depth

    def refresh_sensing(self):
        self.refresh_calls += 1


class NoPoseWorld:
    """Renders and has depth, but exposes no pose (staleness unknowable)."""

    def __init__(self, frame, depth=2.0):
        self.frame = frame
        self.depth_reads = 0

    def render(self):
        return self.frame

    def depth_at_pixel(self, pixel):
        self.depth_reads += 1
        return 2.0


class FakeMemory:
    """ObservationLog-shaped sink: stores records and can serve them back."""

    def __init__(self):
        self.records = []

    def add_observation(self, observation_id, image, yaw_deg=None, episode_id=None, **kwargs):
        self.records.append(
            {
                "observation_id": observation_id,
                "image": image,
                "yaw_deg": yaw_deg,
                "episode_id": episode_id,
                **kwargs,
            }
        )

    def get_observation(self, observation_id):
        for record in self.records:
            if record["observation_id"] == observation_id:
                return SimpleNamespace(
                    image=record["image"],
                    position_flu=record.get("position_flu"),
                    yaw_rad=record.get("yaw_rad"),
                )
        return None


class PickyMemory:
    """Exposes an unusable add method (extra required argument)."""

    def add_observation(self, record, mandatory_extra):
        raise AssertionError("should not be called successfully")


class OddMemory:
    """Exposes no method the bridge knows about."""

    def push(self, payload):
        return payload


class ObservationLogLike:
    """Mirrors ``memory.ObservationLog.record``'s real signature."""

    def __init__(self):
        self.entries = []

    def record(
        self,
        image,
        pose=None,
        yaw=None,
        event=None,
        *,
        obs_id=None,
        target_name=None,
        labels=None,
        note="",
        step=None,
        t=None,
        extra=None,
        yaw_rad=None,
        timestamp=None,
    ):
        self.entries.append(
            {
                "obs_id": obs_id,
                "image": image,
                "pose": pose,
                "yaw_rad": yaw_rad if yaw_rad is not None else yaw,
                "note": note,
                "extra": extra or {},
                "timestamp": timestamp if timestamp is not None else t,
            }
        )
        return obs_id


class FakeVLM:
    name = "fake"

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
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def make_ctx(world, memory=None, **extra):
    return ToolContext(world=world, memory=memory, episode_id="ep_test", extra=dict(extra))


def heading_of(ctx, observation_id):
    """Heading the world actually rendered for a stored observation (degrees)."""
    record = resolve_observation(ctx, observation_id)
    return float(record.frame[0, 0, 0]) * 360.0


def angle_delta(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


class ObserveTest(unittest.TestCase):
    def test_forward_returns_the_documented_contract(self):
        ctx = make_ctx(FakeWorld(yaw_deg=30.0))
        result = observe(ctx, {"mode": "forward"})

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["observation_id"], "obs_1")
        self.assertEqual(result.data["image_refs"], ["obs_1"])
        self.assertEqual(result.data["pose"], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(result.data["yaw_deg"], 30.0, places=3)
        self.assertIsInstance(result.data["time_s"], float)
        self.assertEqual(result.data["mode"], "forward")
        self.assertEqual(result.data["memory_sink"], "absent")
        self.assertEqual(ctx.observation_seq, 1)
        self.assertEqual([e.kind for e in result.evidence], ["observation"])

    def test_lookaround_scans_and_restores_yaw(self):
        world = FakeWorld(yaw_deg=30.0)
        ctx = make_ctx(world)
        result = observe(ctx, {"mode": "lookaround", "n_views": 4})

        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.data["image_refs"]), 4)
        self.assertAlmostEqual(math.degrees(world.yaw_rad), 30.0, places=6)
        self.assertEqual(world.scan_calls[0]["n_views"], 4)
        for ref, heading in zip(result.data["image_refs"], result.data["headings_deg"]):
            self.assertLess(angle_delta(heading, heading_of(ctx, ref)), 1e-3)
        self.assertAlmostEqual(result.data["yaw_deg"], 30.0, places=3)

    def test_lookaround_appends_pitched_views(self):
        world = FakeWorld(yaw_deg=0.0)
        ctx = make_ctx(world)
        result = observe(ctx, {"mode": "lookaround", "n_views": 2, "pitch_deg": 45.0})

        self.assertEqual(len(result.data["image_refs"]), 4)  # 2 level + 2 pitched
        pitched = [resolve_observation(ctx, ref) for ref in result.data["image_refs"][2:]]
        self.assertTrue(all(record.pitched for record in pitched))
        self.assertFalse(resolve_observation(ctx, result.data["image_refs"][0]).pitched)
        self.assertEqual(result.data["observation_id"], result.data["image_refs"][2])
        self.assertAlmostEqual(math.degrees(world.yaw_rad), 0.0, places=6)

    def test_forward_yaw_offset_peeks_then_restores(self):
        world = FakeWorld(yaw_deg=30.0)
        ctx = make_ctx(world)
        result = observe(ctx, {"mode": "forward", "yaw_deg": 90.0})

        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.data["yaw_deg"], 120.0, places=3)
        self.assertLess(angle_delta(heading_of(ctx, result.data["observation_id"]), 120.0), 1e-3)
        self.assertAlmostEqual(math.degrees(world.yaw_rad), 30.0, places=6)

    def test_pitch_down_mode_defaults_and_primary_view(self):
        world = FakeWorld(yaw_deg=0.0)
        ctx = make_ctx(world)
        result = observe(ctx, {"mode": "pitch_down"})

        self.assertEqual(len(result.data["image_refs"]), 3)
        self.assertEqual(result.data["observation_id"], result.data["image_refs"][1])
        record = resolve_observation(ctx, result.data["observation_id"])
        self.assertAlmostEqual(record.metadata["pitch_down_deg"], 50.0)
        self.assertTrue(record.pitched)
        self.assertEqual(world.scan_calls[0]["pitch_down_deg"], 50.0)

    def test_narrow_scan_passes_spread_deg(self):
        world = FakeWorld(yaw_deg=10.0)
        ctx = make_ctx(world)
        result = observe(ctx, {"mode": "lookaround", "n_views": 3, "spread_deg": 60.0})

        self.assertEqual(result.status, "ok")
        self.assertEqual(world.scan_calls[0]["spread_deg"], 60.0)
        headings = result.data["headings_deg"]
        self.assertLess(angle_delta(headings[0], -20.0), 1e-3)
        self.assertLess(angle_delta(headings[-1], 40.0), 1e-3)

    def test_world_without_scan_falls_back_to_one_frame(self):
        world = FakeWorld(yaw_deg=15.0)
        world.scan_views = None
        ctx = make_ctx(world)
        result = observe(ctx, {"mode": "lookaround"})

        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.data["image_refs"]), 1)
        self.assertIn("scan_views", result.data["note"])
        self.assertAlmostEqual(result.data["yaw_deg"], 15.0, places=3)

    def test_world_without_pose_still_returns_frames(self):
        world = NoPoseWorld(torch.full((3, 8, 8), 0.5))
        ctx = make_ctx(world)
        result = observe(ctx, {"mode": "forward"})

        self.assertEqual(result.status, "ok")
        self.assertIsNone(result.data["pose"])
        self.assertIsNone(result.data["yaw_deg"])
        self.assertIsNone(result.data["headings_deg"][0])

    def test_invalid_mode_fails_without_consuming_ids(self):
        ctx = make_ctx(FakeWorld())
        result = observe(ctx, {"mode": "sideways"})

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["error"], "invalid_mode")
        self.assertEqual(result.data["allowed"], ["lookaround", "forward", "pitch_down"])
        self.assertEqual(ctx.observation_seq, 0)

    def test_frames_are_written_to_memory_once_per_view(self):
        memory = FakeMemory()
        ctx = make_ctx(FakeWorld(yaw_deg=30.0), memory=memory)
        result = observe(ctx, {"mode": "lookaround", "n_views": 3})

        self.assertTrue(result.data["memory_sink"].startswith("ok:"))
        self.assertEqual(len(memory.records), 3)
        first = memory.records[0]
        self.assertEqual(first["observation_id"], result.data["image_refs"][0])
        self.assertIsInstance(first["image"], torch.Tensor)
        self.assertAlmostEqual(first["yaw_deg"], 30.0, places=3)
        self.assertEqual(first["episode_id"], "ep_test")

    def test_incompatible_memory_is_reported_not_raised(self):
        ctx = make_ctx(FakeWorld(), memory=PickyMemory())
        result = observe(ctx, {"mode": "forward"})

        self.assertEqual(result.status, "ok")
        self.assertTrue(result.data["memory_sink"].startswith("error:"))
        self.assertEqual(len(result.data["image_refs"]), 1)

    def test_unknown_memory_api_is_reported(self):
        ctx = make_ctx(FakeWorld(), memory=OddMemory())
        result = observe(ctx, {"mode": "forward"})

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["memory_sink"], "unsupported")

    def test_capture_lands_in_an_observation_log_shaped_sink(self):
        memory = ObservationLogLike()
        ctx = make_ctx(FakeWorld(yaw_deg=30.0), memory=memory)
        result = observe(ctx, {"mode": "forward"})

        self.assertEqual(result.data["memory_sink"], "ok:record")
        entry = memory.entries[0]
        self.assertEqual(entry["obs_id"], result.data["observation_id"])
        self.assertIsInstance(entry["image"], torch.Tensor)
        self.assertAlmostEqual(math.degrees(entry["yaw_rad"]), 30.0, places=3)
        self.assertEqual(entry["pose"], [0.0, 0.0, 0.0])
        self.assertEqual(entry["extra"]["kind"], "view")
        self.assertEqual(entry["extra"]["mode"], "forward")
        self.assertGreater(entry["timestamp"], 1.0e9)  # wall clock, not episode time

    def test_crops_land_in_memory_with_parent_and_region(self):
        memory = ObservationLogLike()
        ctx = make_ctx(FakeWorld(), memory=memory)
        observed = observe(ctx, {"mode": "forward"})
        inspected = inspect_region(
            ctx, {"observation_id": observed.data["observation_id"], "region": [0, 0, 500, 1000]}
        )

        self.assertEqual(inspected.data["memory_sink"], "ok:record")
        entry = memory.entries[-1]
        self.assertEqual(entry["obs_id"], inspected.data["crop_image_ref"])
        self.assertEqual(entry["extra"]["kind"], "crop")
        self.assertEqual(entry["extra"]["parent_id"], observed.data["observation_id"])
        self.assertEqual(entry["extra"]["region"], [0.0, 0.0, 500.0, 1000.0])

    def test_scan_views_with_a_narrow_signature_is_tolerated(self):
        world = FakeWorld(yaw_deg=0.0)

        def scan_views(n_views=8):
            return [world.render() for _ in range(n_views)]

        world.scan_views = scan_views
        ctx = make_ctx(world)
        result = observe(ctx, {"mode": "lookaround", "n_views": 3})

        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.data["image_refs"]), 3)

    def test_extra_none_is_repaired(self):
        ctx = make_ctx(FakeWorld())
        ctx.extra = None
        result = observe(ctx, {"mode": "forward"})

        self.assertEqual(result.status, "ok")
        self.assertIsInstance(ctx.extra, dict)
        self.assertIn(result.data["observation_id"], ctx.extra[IMAGE_STORE_KEY])

    def test_registry_integration_and_budget(self):
        registry = ToolRegistry()
        register_perception_tools(registry)
        self.assertEqual(
            registry.names(),
            [
                "compare_candidates",
                "ground_target",
                "inspect_region",
                "observe",
                "query_memory",
                "search_target",
            ],
        )
        # the evidence tools only look; ``search_target`` is the skill riding on
        # them and it may fly the drone to the next viewpoint, so it is the one
        # tool of this module the runtime has to serialize
        self.assertEqual(
            [name for name in registry.names() if registry.get(name).side_effects],
            ["search_target"],
        )

        ctx = make_ctx(FakeWorld())
        result = registry.call("observe", {"mode": "forward"}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertEqual(ctx.budget["tool_calls_left"], 39)
        self.assertEqual(ctx.log[0]["tool"], "observe")
        self.assertEqual(ctx.log[0]["status"], "ok")
        self.assertIn("observe", registry.specs_text())

    def test_schemas_declare_required_arguments(self):
        registry = ToolRegistry()
        register_perception_tools(registry)
        # ``mode`` is optional with a documented default: the prompt's
        # ``observe({})`` example must not be an interface error
        observe_schema = registry.get("observe").input_schema
        self.assertEqual(observe_schema["required"], [])
        self.assertEqual(observe_schema["properties"]["mode"]["default"], "forward")
        self.assertEqual(
            registry.get("inspect_region").input_schema["required"], ["observation_id", "region"]
        )
        self.assertEqual(registry.get("ground_target").input_schema["required"], ["instruction"])
        self.assertEqual(
            registry.get("compare_candidates").input_schema["required"],
            ["candidates_a", "candidates_b"],
        )
        self.assertEqual(
            registry.get("search_target").input_schema["required"], ["instruction"]
        )

    def test_generic_register_alias(self):
        from vlm_nav import tool_impl_perception as module

        self.assertIs(module.register_tools, module.register_perception_tools)
        registry = ToolRegistry()
        module.register_tools(registry)
        self.assertIn("observe", registry.names())


class InspectRegionTest(unittest.TestCase):
    def test_crop_geometry_and_upscale(self):
        ctx = make_ctx(FakeWorld(width=64, height=32, yaw_deg=0.0))
        observed = observe(ctx, {"mode": "forward"})
        result = inspect_region(
            ctx, {"observation_id": observed.data["observation_id"], "region": [500, 0, 500, 1000]}
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["source_observation_id"], observed.data["observation_id"])
        self.assertEqual(result.data["crop_size_px"], [32, 32])
        self.assertAlmostEqual(result.data["upscale"], 7.0, places=2)
        crop = resolve_observation(ctx, result.data["crop_image_ref"]).frame
        self.assertEqual(tuple(crop.shape), (3, 224, 224))
        # right half of the gradient frame: 0.5 .. 1.0 (upscaling never dips below)
        self.assertGreater(float(crop.min()), 0.48)
        self.assertGreater(float(crop.max()), 0.9)
        self.assertEqual([e.kind for e in result.evidence], ["observation", "observation"])

    def test_region_is_clamped_to_the_frame(self):
        ctx = make_ctx(FakeWorld())
        observed = observe(ctx, {"mode": "forward"})
        result = inspect_region(
            ctx, {"observation_id": observed.data["observation_id"], "region": [-100, -100, 300, 300]}
        )

        self.assertEqual(result.status, "ok")
        # canonical corner box: the region reaches (300, 300) in normalized
        # coordinates and its negative origin is clamped to the frame
        self.assertEqual(result.data["region"], [0.0, 0.0, 300.0, 300.0])
        self.assertEqual(result.data["region_uvwh"], [0.0, 0.0, 300.0, 300.0])
        self.assertEqual(result.data["region_format"], "xyxy_norm")

    def test_a_legacy_region_is_normalized_and_reported(self):
        ctx = make_ctx(FakeWorld())
        observed = observe(ctx, {"mode": "forward"})
        result = inspect_region(
            ctx, {"observation_id": observed.data["observation_id"], "region": [500, 0, 500, 1000]}
        )

        # [u, v, w, h] under the old spelling is detected (a corner box with
        # x2 <= x1 cannot be corner-shaped), converted and named in the answer
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["region_format"], "uvwh_norm")
        self.assertEqual(result.data["region"], [500.0, 0.0, 1000.0, 1000.0])
        self.assertEqual(result.data["region_uvwh"], [500.0, 0.0, 500.0, 1000.0])

    def test_depth_read_from_the_current_observation(self):
        world = FakeWorld(depth=3.25)
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        result = inspect_region(
            ctx, {"observation_id": observed.data["observation_id"], "region": [250, 250, 500, 500]}
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["depth_status"], "ok")
        self.assertAlmostEqual(result.data["depth_m"], 3.25, places=3)
        self.assertEqual(result.data["depth_samples"], 9)
        self.assertEqual(world.refresh_calls, 1)
        self.assertTrue(all(250.0 <= u <= 750.0 for u, _ in world.depth_reads))

    def test_depth_is_skipped_for_a_stale_observation(self):
        world = FakeWorld(yaw_deg=30.0, depth=2.0)
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "lookaround", "n_views": 2})
        result = inspect_region(
            ctx, {"observation_id": observed.data["image_refs"][1], "region": [0, 0, 1000, 1000]}
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["depth_status"], "stale_observation")
        self.assertNotIn("depth_m", result.data)
        self.assertEqual(world.refresh_calls, 0)
        self.assertIn("earlier pose", result.data["note"])

    def test_depth_is_skipped_for_a_pitched_view(self):
        world = FakeWorld(yaw_deg=0.0)
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "pitch_down"})
        result = inspect_region(
            ctx, {"observation_id": observed.data["observation_id"], "region": [0, 0, 1000, 1000]}
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["depth_status"], "stale_observation")
        self.assertEqual(world.refresh_calls, 0)

    def test_depth_status_when_no_depth_sensor(self):
        world = FakeWorld()
        world.depth_at_pixel = None
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        result = inspect_region(
            ctx, {"observation_id": observed.data["observation_id"], "region": [0, 0, 1000, 1000]}
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["depth_status"], "no_depth_sensor")

    def test_depth_status_when_pose_is_unknown(self):
        world = NoPoseWorld(torch.full((3, 16, 16), 0.4))
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        result = inspect_region(
            ctx, {"observation_id": observed.data["observation_id"], "region": [0, 0, 1000, 1000]}
        )

        self.assertEqual(result.data["depth_status"], "pose_unknown")
        self.assertEqual(world.depth_reads, 0)

    def test_unknown_observation_id_fails_with_available_ids(self):
        ctx = make_ctx(FakeWorld())
        observe(ctx, {"mode": "forward"})
        result = inspect_region(ctx, {"observation_id": "obs_99", "region": [0, 0, 10, 10]})

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["error"], "unknown_observation_id")
        self.assertIn("obs_1", result.data["available"])

    def test_invalid_region_fails(self):
        ctx = make_ctx(FakeWorld())
        observed = observe(ctx, {"mode": "forward"})
        for region in ([0, 0, 0, 100], [0, 0, 100], "middle", [1, 2, "x", 4]):
            with self.subTest(region=region):
                result = inspect_region(
                    ctx, {"observation_id": observed.data["observation_id"], "region": region}
                )
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.data["error"], "invalid_region")

    def test_region_outside_the_frame_is_blocked(self):
        ctx = make_ctx(FakeWorld())
        observed = observe(ctx, {"mode": "forward"})
        result = inspect_region(
            ctx, {"observation_id": observed.data["observation_id"], "region": [1200, 0, 100, 100]}
        )

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.data["reason"], "region_outside_frame")

    def test_crop_of_a_crop(self):
        ctx = make_ctx(FakeWorld())
        observed = observe(ctx, {"mode": "forward"})
        first = inspect_region(
            ctx, {"observation_id": observed.data["observation_id"], "region": [0, 0, 500, 1000]}
        )
        second = inspect_region(
            ctx, {"observation_id": first.data["crop_image_ref"], "region": [0, 0, 500, 1000]}
        )

        self.assertEqual(second.status, "ok")
        self.assertEqual(second.data["source_observation_id"], first.data["crop_image_ref"])
        self.assertEqual(second.data["crop_size_px"], [112, 224])

    def test_crop_is_readable_from_memory_when_not_in_the_local_cache(self):
        memory = FakeMemory()
        ctx = make_ctx(FakeWorld(), memory=memory)
        observed = observe(ctx, {"mode": "forward"})
        ctx.extra[IMAGE_STORE_KEY].clear()  # simulate eviction of the local cache
        result = inspect_region(
            ctx, {"observation_id": observed.data["observation_id"], "region": [0, 0, 500, 1000]}
        )

        self.assertEqual(result.status, "ok")
        # the frame and the capture pose came back out of memory: a same-pose
        # observation still yields a real depth reading
        self.assertEqual(result.data["depth_status"], "ok")
        self.assertAlmostEqual(result.data["depth_m"], 3.0, places=3)


class GroundTargetTest(unittest.TestCase):
    REPLY = (
        "Sure, here are the candidates:\n"
        "```json\n"
        '{"candidates": ['
        '{"box": [100, 200, 300, 400], "attrs": {"color": "white", "size": "small"}, '
        '"confidence": 1.4, "uncertainty": "far away"}, '
        '{"bbox": [500, 500, 200, 200], "attributes": ["red", "round"], "score": 0.4}'
        "]}\n```"
    )

    def test_candidates_are_parsed_and_normalized(self):
        world = FakeWorld()
        vlm = FakeVLM(self.REPLY)
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx, {"instruction": "find the chair", "observation_id": observed.data["observation_id"]},
            vlm=vlm,
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["n_candidates"], 2)
        self.assertEqual(result.data["backend"], "fake")
        first, second = result.data["candidates"]
        self.assertEqual(first["id"], "cand_1")
        self.assertEqual(first["box"], [100.0, 200.0, 300.0, 400.0])
        self.assertEqual(first["attrs"], {"color": "white", "size": "small"})
        self.assertEqual(first["confidence"], 1.0)  # clamped from 1.4
        self.assertEqual(first["uncertainty"], "far away")
        self.assertEqual(second["attrs"], {"text": "red, round"})
        self.assertEqual(second["confidence"], 0.4)
        self.assertEqual(second["observation_id"], observed.data["observation_id"])
        self.assertIn("cand_1", ctx.extra[CANDIDATE_STORE_KEY])
        # one image, matching the observation, and the instruction in the prompt
        self.assertEqual(len(vlm.calls[0]["frames"]), 1)
        self.assertTrue(torch.equal(vlm.calls[0]["frames"][0], resolve_observation(ctx, observed.data["observation_id"]).frame))
        self.assertIn("find the chair", vlm.calls[0]["user_prompt"])
        self.assertIn("0-1000", vlm.calls[0]["system_prompt"])
        self.assertEqual([e.kind for e in result.evidence][:2], ["observation", "memory"])

    def test_missing_observation_captures_the_current_view(self):
        world = FakeWorld()
        vlm = FakeVLM(self.REPLY)
        ctx = make_ctx(world)
        result = ground_target(ctx, {"instruction": "find the door"}, vlm=vlm)

        self.assertEqual(result.status, "ok")
        self.assertTrue(result.data["captured_observation"])
        self.assertEqual(result.data["observation_id"], "obs_1")
        self.assertEqual(world.render_calls, 1)
        self.assertTrue(
            torch.equal(vlm.calls[0]["frames"][0], resolve_observation(ctx, "obs_1").frame)
        )
        self.assertEqual([e.kind for e in result.evidence][0], "observation")

    def test_without_a_backend_the_result_is_empty_with_a_reason(self):
        ctx = make_ctx(FakeWorld())
        observed = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx, {"instruction": "find the chair", "observation_id": observed.data["observation_id"]}
        )

        self.assertEqual(result.status, "empty")
        self.assertEqual(result.data["reason"], "no_vlm_backend")
        self.assertIn("rule fallback", result.data["note"])
        self.assertEqual(result.evidence[0].kind, "observation")

    def test_backend_from_ctx_extra_is_used(self):
        vlm = FakeVLM(self.REPLY)
        ctx = make_ctx(FakeWorld(), **{"vlm": vlm})
        registry = ToolRegistry()
        register_perception_tools(registry, vlm=None)
        result = registry.call("ground_target", {"instruction": "find the chair"}, ctx)

        self.assertEqual(result.status, "ok")
        self.assertEqual(len(vlm.calls), 1)
        self.assertEqual(result.data["backend"], "fake")

    def test_backend_from_the_runtime_in_ctx_extra_is_used(self):
        vlm = FakeVLM(self.REPLY)
        runtime = SimpleNamespace(vlm=vlm)  # AgentRuntime stores its backend as .vlm
        ctx = make_ctx(FakeWorld(), **{"runtime": runtime})
        result = ground_target(ctx, {"instruction": "find the chair"})

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["backend"], "fake")
        self.assertEqual(len(vlm.calls), 1)

    def test_backend_failure_is_reported(self):
        ctx = make_ctx(FakeWorld())
        observed = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx, {"instruction": "find the chair", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM(RuntimeError("boom")),
        )

        self.assertEqual(result.status, "failed")
        self.assertIn("RuntimeError", result.data["error"])
        self.assertEqual(result.data["observation_id"], observed.data["observation_id"])

    def test_unparsable_reply_is_empty(self):
        ctx = make_ctx(FakeWorld())
        observed = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx, {"instruction": "find the chair", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM("I cannot see anything like that."),
        )

        self.assertEqual(result.status, "empty")
        self.assertEqual(result.data["reason"], "unparsable_reply")

    def test_empty_candidate_list_is_empty(self):
        ctx = make_ctx(FakeWorld())
        observed = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx, {"instruction": "find the chair", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM('{"candidates": []}'),
        )

        self.assertEqual(result.status, "empty")
        self.assertEqual(result.data["reason"], "no_candidate")

    def test_an_empty_box_answer_is_retried_once_with_a_box_only_prompt(self):
        """A model that judged the target but omitted its box gets one retry.

        The v5 bedroom confirms had 15 views with no box, so no independent
        position measurement was possible. The retry asks for the box only on
        the same frame (one model call, no new render) and recovers the box.
        """
        class TwoReplyVLM(FakeVLM):
            def __init__(self):
                super().__init__(None)
                self.replies = [
                    '{"candidates": []}',
                    '{"candidates": [{"box": [450, 450, 100, 100], "confidence": 0.8}]}',
                ]

            def decide(self, frames, system_prompt, user_prompt, context=None):
                self.calls.append(
                    {"frames": frames, "system_prompt": system_prompt,
                     "user_prompt": user_prompt, "context": context}
                )
                return self.replies.pop(0)

        world = FakeWorld(position=(0.0, 0.0, 0.0), yaw_deg=0.0, depth=4.0)
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        vlm = TwoReplyVLM()
        result = ground_target(
            ctx,
            {"instruction": "find the chair", "observation_id": observed.data["observation_id"]},
            vlm=vlm,
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["n_candidates"], 1)
        self.assertEqual(result.data["vlm_calls"], 2)
        self.assertEqual(len(vlm.calls), 2)
        self.assertIn("only the box", vlm.calls[1]["user_prompt"])
        self.assertEqual(result.data["grounding_retry"]["trigger"], "no_candidate")
        self.assertIs(result.data["grounding_retry"]["recovered"], True)
        # no previous candidate to draw: the centre is enlarged and the box is
        # mapped back into the source frame
        self.assertEqual(result.data["grounding_retry"]["mode"], "centre_crop")
        self.assertIn("region", result.data["grounding_retry"])
        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["position_source"], "depth")
        self.assertEqual(candidate["position_flu"], [4.0, 0.0, 0.0])

    def test_an_empty_retry_stays_empty_and_reports_both_calls(self):
        ctx = make_ctx(FakeWorld())
        observed = observe(ctx, {"mode": "forward"})
        vlm = FakeVLM('{"candidates": []}')
        result = ground_target(
            ctx,
            {"instruction": "find the chair", "observation_id": observed.data["observation_id"]},
            vlm=vlm,
        )

        self.assertEqual(result.status, "empty")
        self.assertEqual(result.data["reason"], "no_candidate")
        self.assertEqual(result.data["vlm_calls"], 2)
        self.assertEqual(len(vlm.calls), 2)
        self.assertIs(result.data["grounding_retry"]["recovered"], False)

    def test_candidates_without_a_valid_box_are_dropped(self):
        ctx = make_ctx(FakeWorld())
        observed = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx, {"instruction": "find the chair", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [0, 0, 0, 0]}, {"box": [10, 10, 100, 100]}]}'),
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["n_candidates"], 1)
        self.assertEqual(result.data["candidates"][0]["confidence"], None)

    def test_candidate_ids_do_not_collide_across_calls(self):
        ctx = make_ctx(FakeWorld())
        observed = observe(ctx, {"mode": "forward"})
        args = {"instruction": "find the chair", "observation_id": observed.data["observation_id"]}
        first = ground_target(ctx, args, vlm=FakeVLM(self.REPLY))
        second = ground_target(ctx, args, vlm=FakeVLM(self.REPLY))

        ids = [c["id"] for c in first.data["candidates"] + second.data["candidates"]]
        self.assertEqual(ids, ["cand_1", "cand_2", "cand_3", "cand_4"])
        self.assertEqual(sorted(ctx.extra[CANDIDATE_STORE_KEY]), ids)

    def test_missing_instruction_fails(self):
        ctx = make_ctx(FakeWorld())
        result = ground_target(ctx, {"observation_id": "obs_1"}, vlm=FakeVLM(self.REPLY))

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["error"], "missing_instruction")

    def test_unknown_observation_id_fails(self):
        ctx = make_ctx(FakeWorld())
        result = ground_target(ctx, {"instruction": "chair", "observation_id": "obs_42"}, vlm=FakeVLM(self.REPLY))

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["error"], "unknown_observation_id")


class CandidatePositionTest(unittest.TestCase):
    """A ``cand_N`` id is only useful if it resolves to a world position: the
    navigation tools receive nothing but the id, so ``ground_target`` has to
    publish the box centre in world coordinates."""

    BOX = '{"candidates": [{"box": [450, 450, 100, 100]}]}'  # u,v centre = image centre

    def test_the_box_centre_lands_in_the_candidate_store(self):
        world = FakeWorld(position=(1.0, 0.0, 0.5), yaw_deg=0.0, depth=4.0)
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM(self.BOX),
        )

        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["position_source"], "depth")
        self.assertEqual(candidate["position_flu"], [5.0, 0.0, 0.5])
        # the navigation tools resolve ids out of this store
        self.assertEqual(
            ctx.extra[CANDIDATE_STORE_KEY]["cand_1"]["position_flu"], [5.0, 0.0, 0.5]
        )

    def test_depth_is_preferred_when_it_agrees_with_the_model_distance(self):
        world = FakeWorld(position=(0.0, 0.0, 0.0), yaw_deg=0.0, depth=4.0)
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [450, 450, 100, 100], "distance_m": 5.0}]}'),
        )

        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["position_source"], "depth")
        self.assertEqual(candidate["position_flu"], [4.0, 0.0, 0.0])
        self.assertEqual(candidate["distance_source"], "depth_center")
        self.assertIsNone(candidate["distance_conflict"])
        self.assertFalse(candidate["distance_uncertain"])

    def test_a_box_centre_depth_that_contradicts_the_model_is_flagged(self):
        """A huge, border-touching box whose centre depth is background.

        The classroom v2 candidate was placed 16-27 m away from a shelf the
        model estimated at 3-4 m. The centre depth must not be trusted when it
        contradicts the model and no ground-contact sample resolves the
        disagreement: the candidate keeps the model's range, is marked
        ``distance_source=conflict`` and cannot be accepted by confirmation.
        """
        world = FakeWorld(
            position=(0.0, 0.0, 0.0), yaw_deg=0.0,
            depth=lambda pixel: 16.0, hfov_deg=90.0,
        )
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx,
            {"instruction": "bookshelf", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM(
                '{"candidates": [{"box": [480, 0, 999, 400], "distance_m": 3.0}]}'
            ),
        )

        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["position_source"], "model_distance")
        self.assertEqual(candidate["distance_source"], "conflict")
        self.assertAlmostEqual(candidate["range_m"], 3.0)
        self.assertIs(candidate["distance_conflict"]["resolved"], False)
        self.assertEqual(candidate["distance_conflict"]["resolution"], "border_large")
        self.assertIs(candidate["distance_uncertain"], True)
        self.assertIs(candidate["border_large_box"], True)
        # no pixel of this frame is measurable; the advice is a new view with
        # a baseline, not another depth read of the same shelf/wall
        advice = candidate["view_change_advice"]
        self.assertEqual(advice["recommended"], "change_viewpoint")
        self.assertEqual(advice["alternative_tool"], "propose_viewpoints")
        # the model's range (not the 16 m wall) is what navigation can use
        position = np.asarray(candidate["position_flu"], dtype=np.float64)
        self.assertGreater(position[0], 0.0)
        self.assertLess(float(np.linalg.norm(position)), 6.0)

    def test_border_large_bottom_strip_finds_a_near_support_surface(self):
        """The classroom shelf: the centre and the wall behind it are 15 m, but
        the bottom strip holds a contiguous near surface at the model's range.

        The v7 run left 25/27 classroom candidates unmeasurable because a
        border-large box was rejected before any pixel was read. The bottom-strip
        scan is the measurable alternative: a run of columns that agree with the
        model's 3.5 m is the support point, not the 15 m wall.
        """

        def depth(pixel):
            x, y = float(pixel[0]), float(pixel[1])
            if y >= 430.0 and 600.0 <= x <= 820.0:
                return 3.4
            return 15.0

        world = FakeWorld(
            position=(0.0, 0.0, 0.0), yaw_deg=0.0, depth=depth, hfov_deg=90.0,
        )
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx,
            {"instruction": "bookshelf", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM(
                '{"candidates": [{"box": [480, 0, 999, 500], "distance_m": 3.5}]}'
            ),
        )

        candidate = result.data["candidates"][0]
        self.assertIs(candidate["border_large_box"], True)
        self.assertEqual(candidate["position_source"], "depth")
        self.assertEqual(candidate["distance_source"], "depth_border_scan")
        self.assertIs(candidate["distance_uncertain"], False)
        self.assertAlmostEqual(candidate["range_m"], 3.4, places=3)
        self.assertIsNone(candidate["border_second_view"])
        conflict = candidate["distance_conflict"]
        self.assertIs(conflict["resolved"], True)
        self.assertEqual(conflict["resolution"], "border_bottom_scan")
        self.assertGreaterEqual(conflict["border_scan"]["run_columns"], 3)
        # the selected point is the support cluster, not the 15 m background
        self.assertAlmostEqual(conflict["border_scan"]["sample_depth_m"], 3.4)

    def test_border_large_without_a_near_surface_requests_a_dedicated_view(self):
        """Only the 15 m wall is readable: the candidate must ask for a view
        toward the edge the box occupies (right edge -> negative yaw), with a
        sideways step for the baseline - not another 45-deg look at the wall."""
        world = FakeWorld(
            position=(0.0, 0.0, 0.0), yaw_deg=0.0,
            depth=lambda pixel: 15.0, hfov_deg=90.0,
        )
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx,
            {"instruction": "bookshelf", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM(
                '{"candidates": [{"box": [480, 0, 999, 500], "distance_m": 3.5}]}'
            ),
        )

        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["position_source"], "model_distance")
        self.assertEqual(candidate["distance_source"], "conflict")
        self.assertEqual(candidate["distance_conflict"]["resolution"], "border_large")
        self.assertIs(candidate["distance_uncertain"], True)
        second = candidate["border_second_view"]
        self.assertEqual(second["side"], "right")
        self.assertGreaterEqual(abs(second["yaw_deg"]), 30.0)
        self.assertLessEqual(abs(second["yaw_deg"]), 45.0)
        self.assertLess(second["yaw_deg"], 0.0)
        self.assertEqual(second["step_m"], 1.5)
        advice = candidate["view_change_advice"]
        self.assertEqual(advice["observe_first"]["args"]["yaw_deg"], second["yaw_deg"])
        self.assertEqual(advice["dedicated_second_view"], second)
        self.assertTrue(advice.get("step_hint"))

    def test_a_big_box_prefers_the_ground_contact_depth(self):
        """A large box whose centre is background but whose bottom edge lands
        on the object's support: the ground sample resolves the disagreement."""
        world = FakeWorld(
            position=(0.0, 0.0, 0.0), yaw_deg=0.0,
            depth=lambda pixel: 16.0 if pixel[1] <= 300.0 else 3.2,
            hfov_deg=90.0,
        )
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx,
            {"instruction": "bookshelf", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM(
                '{"candidates": [{"box": [480, 100, 950, 500], "distance_m": 3.0}]}'
            ),
        )

        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["position_source"], "depth")
        self.assertEqual(candidate["distance_source"], "depth_ground")
        self.assertAlmostEqual(candidate["range_m"], 3.2)
        self.assertIs(candidate["distance_conflict"]["resolved"], True)
        self.assertIs(candidate["distance_uncertain"], False)
        self.assertAlmostEqual(candidate["distance_conflict"]["depth_center_m"], 16.0)

    def test_multi_point_sampling_finds_a_near_surface_inside_the_box(self):
        """The centre and the bottom edge can both read the wall behind the
        object while an interior sample lands on the near surface. The chosen
        sample has to agree with the model's range, not merely be the first."""
        world = FakeWorld(
            position=(0.0, 0.0, 0.0), yaw_deg=0.0,
            depth=lambda pixel: 3.4 if 350.0 <= pixel[1] <= 400.0 else 15.0,
            hfov_deg=90.0,
        )
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx,
            {"instruction": "bookshelf", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM(
                '{"candidates": [{"box": [470, 100, 950, 500], "distance_m": 3.0}]}'
            ),
        )

        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["position_source"], "depth")
        self.assertEqual(candidate["distance_source"], "depth_ground")
        self.assertAlmostEqual(candidate["range_m"], 3.4)
        self.assertIs(candidate["distance_uncertain"], False)
        self.assertIs(candidate["distance_conflict"]["resolved"], True)
        # the centre is still the wall: the conflict is recorded, not hidden
        self.assertEqual(candidate["distance_conflict"]["depth_center_m"], 15.0)

    def test_converged_support_samples_override_a_model_underestimate(self):
        """A tight support-contact cluster is a measurement, not a conflict.

        The bedroom v4 seed41 lamp read 3.16-3.23 m along its bottom edge while
        the model guessed 1.2 m; the old symmetric ratio rule refused the
        probably-correct candidate. Converged bottom samples on a normal box
        select the near surface and record the model range as an underestimate.
        """
        world = FakeWorld(
            position=(0.0, 0.0, 0.0), yaw_deg=0.0,
            depth=lambda pixel: (
                2.6 if pixel[0] < 487.0 else (2.8 if pixel[0] < 512.0 else 3.0)
            ),
            hfov_deg=90.0,
        )
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx,
            {"instruction": "the lamp", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM(
                '{"candidates": [{"box": [450, 450, 100, 100], "distance_m": 1.2}]}'
            ),
        )

        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["position_source"], "depth")
        self.assertEqual(candidate["distance_source"], "depth_consistent")
        self.assertIs(candidate["distance_uncertain"], False)
        self.assertAlmostEqual(candidate["range_m"], 2.6, places=3)
        conflict = candidate["distance_conflict"]
        self.assertIs(conflict["resolved"], True)
        self.assertEqual(conflict["resolution"], "depth_consensus")
        self.assertIs(conflict["model_underestimate"], True)
        self.assertIn("warning", conflict)
        self.assertEqual(conflict["consensus"]["samples"], 3)

    def test_dispersed_support_samples_stay_uncertain(self):
        """Samples spread over metres are not one surface: keep the conflict.

        The old rule is what protects a box whose bottom edge crosses the
        object and the wall behind it. Only a converged cluster may override
        the model range.
        """
        world = FakeWorld(
            position=(0.0, 0.0, 0.0), yaw_deg=0.0,
            depth=lambda pixel: (
                2.6 if pixel[0] < 487.0 else (5.2 if pixel[0] < 512.0 else 9.0)
            ),
            hfov_deg=90.0,
        )
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx,
            {"instruction": "the lamp", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM(
                '{"candidates": [{"box": [450, 450, 100, 100], "distance_m": 1.2}]}'
            ),
        )

        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["position_source"], "model_distance")
        self.assertEqual(candidate["distance_source"], "conflict")
        self.assertIs(candidate["distance_uncertain"], True)
        self.assertIs(candidate["distance_conflict"]["resolved"], False)
        self.assertAlmostEqual(candidate["range_m"], 1.2, places=3)

    def test_two_views_triangulate_the_candidate_position(self):
        """Two bearings from different poses replace the single-view range.

        Camera A at (0,0,1) looks along +x; camera B at (5,5,1) looks along -y.
        Both see the same target at (5,0,1). Each single view is only a bearing
        plus a range estimate; their intersection is the position.
        """
        world = FakeWorld(position=(0.0, 0.0, 1.0), yaw_deg=0.0, depth=5.0, hfov_deg=90.0)
        ctx = make_ctx(world)
        first_obs = observe(ctx, {"mode": "forward"})
        first = ground_target(
            ctx,
            {"instruction": "the red chair", "observation_id": first_obs.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [450, 450, 100, 100], "distance_m": 5.0}]}'),
        )
        self.assertEqual(first.status, "ok")
        first_candidate = first.data["candidates"][0]
        self.assertEqual(first_candidate["position_source"], "depth")

        world.state.position_flu = np.asarray([5.0, 5.0, 1.0])
        world.state.yaw_rad = math.radians(-90.0)
        second_obs = observe(ctx, {"mode": "forward"})
        second = ground_target(
            ctx,
            {"instruction": "the red chair", "observation_id": second_obs.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [450, 450, 100, 100], "distance_m": 5.0}]}'),
        )

        candidate = second.data["candidates"][0]
        self.assertEqual(candidate["position_source"], "triangulated")
        self.assertEqual(candidate["distance_source"], "triangulated")
        self.assertIs(candidate["distance_uncertain"], False)
        position = np.asarray(candidate["position_flu"], dtype=np.float64)
        error = float(np.linalg.norm(position - np.asarray([5.0, 0.0, 1.0])))
        self.assertLess(error, 0.5)
        triangulation = candidate["triangulation"]
        self.assertAlmostEqual(triangulation["angle_deg"], 90.0, places=1)
        self.assertLess(triangulation["residual_m"], 0.5)
        self.assertEqual(
            triangulation["merged_candidate_id"], first_candidate["id"]
        )
        self.assertEqual(
            ctx.extra[CANDIDATE_STORE_KEY][first_candidate["id"]]["status"], "merged"
        )

    def test_same_pose_views_do_not_triangulate(self):
        """A lookaround changes heading, not position: no baseline, no stereo."""
        world = FakeWorld(position=(0.0, 0.0, 1.0), yaw_deg=0.0, depth=5.0, hfov_deg=90.0)
        ctx = make_ctx(world)
        first_obs = observe(ctx, {"mode": "forward"})
        ground_target(
            ctx,
            {"instruction": "chair", "observation_id": first_obs.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [450, 450, 100, 100], "distance_m": 5.0}]}'),
        )
        turned = observe(ctx, {"mode": "forward", "yaw_deg": 45.0})
        result = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": turned.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [450, 450, 100, 100], "distance_m": 5.0}]}'),
        )

        candidate = result.data["candidates"][0]
        self.assertNotEqual(candidate["position_source"], "triangulated")
        self.assertNotIn("triangulation", candidate)

    def test_a_rephrased_instruction_still_reuses_the_first_bearing(self):
        """The model renames the same target between views (v6 classroom).

        ``带书的书架`` and ``在教室里找到带书的书架（地标） - 一个摆满书的书架``
        are one object; an exact instruction signature dropped the second
        bearing and the shelf produced 28 unresolved conflicts but 3 merges.
        """
        world = FakeWorld(position=(0.0, 0.0, 1.0), yaw_deg=0.0, depth=5.0, hfov_deg=90.0)
        ctx = make_ctx(world)
        first_obs = observe(ctx, {"mode": "forward"})
        first = ground_target(
            ctx,
            {"instruction": "带书的书架", "observation_id": first_obs.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [450, 450, 100, 100], "distance_m": 5.0}]}'),
        )
        first_candidate = first.data["candidates"][0]

        world.state.position_flu = np.asarray([5.0, 5.0, 1.0])
        world.state.yaw_rad = math.radians(-90.0)
        second_obs = observe(ctx, {"mode": "forward"})
        second = ground_target(
            ctx,
            {
                "instruction": "在教室里找到带书的书架（地标） - 一个摆满书的书架",
                "observation_id": second_obs.data["observation_id"],
            },
            vlm=FakeVLM('{"candidates": [{"box": [450, 450, 100, 100], "distance_m": 5.0}]}'),
        )

        candidate = second.data["candidates"][0]
        self.assertEqual(candidate["position_source"], "triangulated", candidate)
        self.assertEqual(
            candidate["triangulation"]["merged_candidate_id"], first_candidate["id"]
        )
        self.assertGreaterEqual(candidate["triangulation"]["angle_deg"], 10.0)

    def test_low_parallax_views_do_not_triangulate(self):
        """5.7 deg / 1.0 m pair: the v6 bedroom pair was biased by 1.6 m."""
        world = FakeWorld(position=(0.0, 0.0, 1.0), yaw_deg=0.0, depth=10.0, hfov_deg=90.0)
        ctx = make_ctx(world)
        first_obs = observe(ctx, {"mode": "forward"})
        ground_target(
            ctx,
            {"instruction": "chair", "observation_id": first_obs.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [450, 450, 100, 100], "distance_m": 10.0}]}'),
        )

        # 1.0 m baseline at 10 m range: atan(0.1) = 5.71 deg, below the floor
        world.state.position_flu = np.asarray([0.0, 1.0, 1.0])
        world.state.yaw_rad = math.radians(math.degrees(math.atan2(-1.0, 10.0)))
        second_obs = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": second_obs.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [450, 450, 100, 100], "distance_m": 10.0}]}'),
        )

        candidate = result.data["candidates"][0]
        self.assertNotEqual(candidate["position_source"], "triangulated")
        self.assertNotIn("triangulation", candidate)

    def test_a_rejected_triangulation_is_not_minted_again(self):
        """v6/v7 bedroom seed37: one intersection was minted five to nine times
        because only ``position_inconsistent`` used to be remembered."""
        world = FakeWorld(position=(0.0, 0.0, 1.0), yaw_deg=0.0, depth=5.0, hfov_deg=90.0)
        ctx = make_ctx(world)
        first_obs = observe(ctx, {"mode": "forward"})
        ground_target(
            ctx,
            {"instruction": "chair", "observation_id": first_obs.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [450, 450, 100, 100], "distance_m": 5.0}]}'),
        )

        world.state.position_flu = np.asarray([5.0, 5.0, 1.0])
        world.state.yaw_rad = math.radians(-90.0)
        second_obs = observe(ctx, {"mode": "forward"})
        second = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": second_obs.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [450, 450, 100, 100], "distance_m": 5.0}]}'),
        )
        rejected = second.data["candidates"][0]
        self.assertEqual(rejected["position_source"], "triangulated")
        # a terminal identity rejection is remembered too (not only a position
        # mismatch); v7 seed37's 21.3-deg point was rejected as different_instance
        entry = _remember_rejected_triangulation(ctx, rejected, "different_instance")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["reason"], "different_instance")
        self.assertEqual(len(ctx.extra[TRIANGULATION_REJECTIONS_KEY]), 1)
        self.assertTrue(entry["fingerprint"])
        self.assertTrue(entry["ray"])

        # a third view of the same pair: the same intersection must not return,
        # and the re-grounded candidate inherits the rejected status
        third_obs = observe(ctx, {"mode": "forward"})
        third = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": third_obs.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [450, 450, 100, 100], "distance_m": 5.0}]}'),
        )

        candidate = third.data["candidates"][0]
        self.assertNotEqual(candidate["position_source"], "triangulated", candidate)
        self.assertNotIn("triangulation", candidate)
        self.assertEqual(candidate["status"], "rejected")
        self.assertEqual(candidate["confirmation_status"], "rejected")
        self.assertIs(candidate["rejection_inherited"], True)
        self.assertEqual(candidate["rejection_reason"], "different_instance")
        self.assertEqual(
            ctx.extra[CANDIDATE_STORE_KEY][candidate["id"]]["status"], "rejected"
        )

    def test_a_same_view_regrounding_inherits_the_rejection(self):
        """The same bearing/box after a terminal rejection is the same
        candidate; it must not be offered as a fresh one even when no second
        bearing exists yet."""
        world = FakeWorld(position=(0.0, 0.0, 1.0), yaw_deg=0.0, depth=5.0, hfov_deg=90.0)
        ctx = make_ctx(world)
        first_obs = observe(ctx, {"mode": "forward"})
        first = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": first_obs.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [450, 450, 100, 100], "distance_m": 5.0}]}'),
        )
        rejected = first.data["candidates"][0]
        _remember_rejected_triangulation(ctx, rejected, "relation_not_supported")

        # a new frame at the same pose: same ray, same box, new observation id
        again = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": again.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [452, 448, 100, 100], "distance_m": 5.0}]}'),
        )

        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["status"], "rejected")
        self.assertIs(candidate["rejection_inherited"], True)
        self.assertEqual(candidate["rejection_reason"], "relation_not_supported")

    def test_ground_retry_draws_the_previous_candidate_box(self):
        """A visual anchor is what the v6 same-frame text retry lacked."""

        class TwoReplyVLM(FakeVLM):
            def __init__(self):
                super().__init__(None)
                self.replies = [
                    '{"candidates": []}',
                    '{"candidates": [{"box": [450, 450, 100, 100], "confidence": 0.8}]}',
                ]

            def decide(self, frames, system_prompt, user_prompt, context=None):
                self.calls.append(
                    {"frames": frames, "system_prompt": system_prompt,
                     "user_prompt": user_prompt, "context": context}
                )
                return self.replies.pop(0)

        world = FakeWorld(position=(0.0, 0.0, 0.0), yaw_deg=0.0, depth=4.0)
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        original = resolve_observation(ctx, observed.data["observation_id"]).frame.clone()
        ground_target(
            ctx,
            {"instruction": "chair", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [450, 450, 100, 100], "distance_m": 4.0}]}'),
        )
        vlm = TwoReplyVLM()
        result = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": observed.data["observation_id"]},
            vlm=vlm,
        )

        self.assertEqual(result.status, "ok")
        retry = result.data["grounding_retry"]
        self.assertEqual(retry["mode"], "previous_box")
        self.assertTrue(retry["recovered"])
        self.assertIn("red rectangle", vlm.calls[1]["user_prompt"])
        self.assertFalse(torch.equal(vlm.calls[1]["frames"][0], original))

    def test_repeated_conflicting_groundings_on_one_view_are_stopped(self):
        """Two unresolved conflicts on the same pixels stop the candidate loop.

        A new observation id over the same view is not new evidence: repeating
        the grounding there returned the same background box and the agent
        burned its budget re-grounding. The third call is refused with a
        viewpoint change instead of another unusable candidate.
        """
        world = FakeWorld(
            position=(0.0, 0.0, 0.0), yaw_deg=0.0,
            depth=lambda pixel: 16.0, hfov_deg=90.0,
        )
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        vlm = FakeVLM(
            '{"candidates": [{"box": [480, 0, 999, 400], "distance_m": 3.0}]}'
        )
        args = {
            "instruction": "bookshelf",
            "observation_id": observed.data["observation_id"],
        }

        first = ground_target(ctx, args, vlm=vlm)
        second = ground_target(ctx, args, vlm=vlm)
        third = ground_target(ctx, args, vlm=vlm)

        self.assertEqual(first.status, "ok", first.data)
        self.assertEqual(second.status, "ok", second.data)
        self.assertEqual(third.status, "blocked", third.data)
        self.assertEqual(third.data["reason"], "stagnant_distance_conflict")
        self.assertIs(third.data["model_call_skipped"], True)
        self.assertEqual(len(vlm.calls), 2)
        self.assertNotEqual(third.data["next_action"]["tool"], "ground_target")

    def test_an_older_frame_uses_the_capture_pose_and_the_model_distance(self):
        world = FakeWorld(position=(0.0, 0.0, 0.0), yaw_deg=0.0, depth=6.0)
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        world.state.position_flu = np.asarray([0.0, 10.0, 0.0])  # the drone flew on
        result = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [450, 450, 100, 100], "distance_m": 3.0}]}'),
        )

        candidate = result.data["candidates"][0]
        # the depth map belongs to the old frame (never mixed with a newer pose),
        # so the model's own range is projected from the capture pose
        self.assertEqual(candidate["position_source"], "model_distance")
        self.assertEqual(candidate["position_flu"], [3.0, 0.0, 0.0])

    def test_pitched_views_are_not_projected(self):
        world = FakeWorld(depth=4.0)
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "pitch_down"})
        result = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [450, 450, 100, 100], "distance_m": 3.0}]}'),
        )

        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["position_source"], "pitched_view")
        self.assertNotIn("position_flu", candidate)

    def test_without_depth_or_distance_no_position_is_invented(self):
        world = FakeWorld()
        world.depth = None  # depth sensor present, but no reading at that pixel
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM(self.BOX),
        )

        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["position_source"], "no_valid_depth")
        self.assertNotIn("position_flu", candidate)

    def test_an_unknown_capture_pose_is_reported(self):
        ctx = make_ctx(NoPoseWorld(torch.zeros(3, 8, 8)))
        observed = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [450, 450, 100, 100], "distance_m": 3.0}]}'),
        )

        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["position_source"], "pose_unknown")
        self.assertNotIn("position_flu", candidate)


class CompareCandidatesTest(unittest.TestCase):
    def test_same_frame_overlap_reads_as_same(self):
        ctx = make_ctx(FakeWorld())
        a = {"id": "a", "box": [100, 100, 200, 200], "attrs": {"color": "white"}, "confidence": 0.8,
             "observation_id": "obs_1"}
        b = {"id": "b", "box": [120, 110, 200, 210], "attrs": {"color": "white"}, "confidence": 0.7,
             "observation_id": "obs_1"}
        result = compare_candidates(ctx, {"candidates_a": [a], "candidates_b": [b]})

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["verdict"], "same")
        self.assertTrue(result.data["same_frame"])
        self.assertGreater(result.data["iou"], 0.5)
        self.assertEqual(result.data["conflicts"], [])
        self.assertIn("attr color=white", result.data["match_basis"])

    def test_same_frame_disjoint_reads_as_different(self):
        ctx = make_ctx(FakeWorld())
        a = {"id": "a", "box": [0, 0, 100, 100], "observation_id": "obs_1"}
        b = {"id": "b", "box": [800, 800, 100, 100], "observation_id": "obs_1"}
        result = compare_candidates(ctx, {"candidates_a": a, "candidates_b": b})

        self.assertEqual(result.data["verdict"], "different")
        self.assertEqual(result.data["iou"], 0.0)
        self.assertTrue(any("iou" in conflict for conflict in result.data["conflicts"]))

    def test_attribute_conflict_reads_as_different(self):
        ctx = make_ctx(FakeWorld())
        a = {"id": "a", "box": [100, 100, 200, 200], "attrs": {"color": "white"}, "observation_id": "obs_1"}
        b = {"id": "b", "box": [105, 105, 200, 200], "attrs": {"color": "brown"}, "observation_id": "obs_1"}
        result = compare_candidates(ctx, {"candidates_a": a, "candidates_b": b})

        self.assertEqual(result.data["verdict"], "different")
        self.assertEqual(result.data["conflicts"], ["attr color: a=white b=brown"])

    def test_cross_frame_comparison_is_uncertain(self):
        ctx = make_ctx(FakeWorld())
        a = {"id": "a", "box": [100, 100, 200, 200], "observation_id": "obs_1"}
        b = {"id": "b", "box": [100, 100, 200, 200], "observation_id": "obs_2"}
        result = compare_candidates(ctx, {"candidates_a": a, "candidates_b": b})

        self.assertEqual(result.data["verdict"], "uncertain")
        self.assertFalse(result.data["same_frame"])
        self.assertIsNone(result.data["iou"])
        self.assertTrue(any("cross_frame" in item for item in result.data["to_confirm"]))

    def test_low_confidence_and_question_are_flagged(self):
        ctx = make_ctx(FakeWorld())
        a = {"id": "a", "box": [100, 100, 200, 200], "confidence": 0.1, "observation_id": "obs_1"}
        b = {"id": "b", "box": [110, 110, 200, 200], "confidence": 0.9, "observation_id": "obs_1"}
        result = compare_candidates(
            ctx, {"candidates_a": a, "candidates_b": b, "question": "is it the same chair?"}
        )

        self.assertTrue(any("low_confidence:a" in item for item in result.data["to_confirm"]))
        self.assertTrue(
            any("question_requires_vlm_judgement" in item for item in result.data["to_confirm"])
        )
        self.assertEqual(result.data["question"], "is it the same chair?")

    def test_candidates_are_resolved_from_ground_target(self):
        ctx = make_ctx(FakeWorld())
        observed = observe(ctx, {"mode": "forward"})
        grounded = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM(
                '{"candidates": [{"box": [100, 100, 200, 200], "attrs": {"color": "white"}, '
                '"confidence": 0.9}, {"box": [600, 600, 100, 100], "attrs": {"color": "gray"}, '
                '"confidence": 0.3}]}'
            ),
        )
        candidates = grounded.data["candidates"]
        result = compare_candidates(
            ctx, {"candidates_a": [candidates[0]["id"]], "candidates_b": [candidates[1]["id"]]}
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["picked_a"], "cand_1")
        self.assertEqual(result.data["picked_b"], "cand_2")
        self.assertEqual(result.data["confidence_a"], 0.9)
        kinds = [e.kind for e in result.evidence]
        self.assertIn("memory", kinds)
        self.assertIn("observation", kinds)

    def test_best_candidate_of_each_group_is_picked(self):
        ctx = make_ctx(FakeWorld())
        a = [{"id": "a1", "box": [0, 0, 100, 100], "confidence": 0.2, "observation_id": "obs_1"},
             {"id": "a2", "box": [100, 100, 200, 200], "confidence": 0.95, "observation_id": "obs_1"}]
        b = [{"id": "b1", "box": [110, 110, 200, 200], "confidence": 0.9, "observation_id": "obs_1"}]
        result = compare_candidates(ctx, {"candidates_a": a, "candidates_b": b})

        self.assertEqual(result.data["picked_a"], "a2")
        self.assertEqual(result.data["candidates_a"], ["a1", "a2"])

    def test_unknown_candidate_id_is_empty(self):
        ctx = make_ctx(FakeWorld())
        result = compare_candidates(ctx, {"candidates_a": "cand_99", "candidates_b": "cand_98"})

        self.assertEqual(result.status, "empty")
        self.assertEqual(result.data["reason"], "unknown_candidate")
        self.assertEqual(result.data["unknown"], ["cand_99", "cand_98"])

    def test_empty_group_is_reported(self):
        ctx = make_ctx(FakeWorld())
        result = compare_candidates(ctx, {"candidates_a": [], "candidates_b": []})

        self.assertEqual(result.status, "empty")
        self.assertEqual(result.data["reason"], "empty_candidate_group")


class ContractTest(unittest.TestCase):
    def test_results_are_json_serializable(self):
        world = FakeWorld()
        ctx = make_ctx(world, memory=FakeMemory())
        observed = observe(ctx, {"mode": "lookaround", "n_views": 2, "pitch_deg": 20.0})
        inspected = inspect_region(
            ctx, {"observation_id": observed.data["observation_id"], "region": [0, 0, 400, 400]}
        )
        grounded = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [0, 0, 10, 10], "attrs": {"c": "w"}, "confidence": 0.5}]}'),
        )
        compared = compare_candidates(
            ctx, {"candidates_a": ["cand_1"], "candidates_b": ["cand_1"]}
        )

        for name, result in (
            ("observe", observed),
            ("inspect_region", inspected),
            ("ground_target", grounded),
            ("compare_candidates", compared),
        ):
            with self.subTest(tool=name):
                payload = json.dumps(result.to_dict())
                self.assertIn(result.status, payload)
                self.assertTrue(result.to_prompt().startswith(f"[{result.status}]"))
                self.assertEqual(result.to_prompt().count("\n"), 0)
        self.assertIn("cand_1", ctx.extra[CANDIDATE_STORE_KEY])

    def test_observations_are_reachable_for_other_tools(self):
        ctx = make_ctx(FakeWorld(yaw_deg=45.0))
        self.assertIsNone(latest_observation(ctx))
        observed = observe(ctx, {"mode": "forward"})

        latest = latest_observation(ctx)
        self.assertEqual(latest.observation_id, observed.data["observation_id"])
        self.assertLess(angle_delta(math.degrees(latest.yaw_rad), 45.0), 1e-6)
        self.assertEqual(tuple(latest.position_flu), (0.0, 0.0, 0.0))
        self.assertEqual(resolve_observation(ctx, "obs_404"), None)


class RecalledPoseTest(unittest.TestCase):
    """The pose lookup that turns a recalled observation id into a target.

    ``navigate`` resolves ``obs_N`` through these helpers, so each store that can
    answer "where was that frame taken?" is covered here: the perception cache,
    the navigation pose records and the memory archive.
    """

    def test_pose_comes_from_the_perception_cache(self):
        world = FakeWorld(position=(1.5, -0.5, 0.75), yaw_deg=30.0)
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})

        pose = resolve_observation_pose(ctx, observed.data["observation_id"])

        self.assertIsNotNone(pose)
        self.assertEqual(pose[0], (1.5, -0.5, 0.75))
        self.assertLess(angle_delta(math.degrees(pose[1]), 30.0), 1e-6)

    def test_pose_comes_from_the_navigation_pose_records(self):
        ctx = make_ctx(FakeWorld())
        ctx.extra["observations"] = {
            "obs_7": {
                "observation_id": "obs_7",
                "position_flu": [1.0, 2.0, 0.5],
                "yaw_rad": 0.25,
                "kind": "post_leg",
            }
        }

        pose = resolve_observation_pose(ctx, "obs_7")

        self.assertEqual(pose[0], (1.0, 2.0, 0.5))
        self.assertAlmostEqual(pose[1], 0.25, places=6)

    def test_pose_comes_from_the_memory_archive_without_its_pixels(self):
        from vlm_nav.memory import ObservationLog

        log = ObservationLog()
        world = FakeWorld(position=(2.0, 0.0, 0.0), yaw_deg=0.0)
        ctx = make_ctx(world, memory=log)
        observe(ctx, {"mode": "forward"})
        ctx.extra[IMAGE_STORE_KEY].clear()  # only the archive remembers it now

        pose = resolve_observation_pose(ctx, "obs_1")

        self.assertEqual(pose[0], (2.0, 0.0, 0.0))

    def test_an_unknown_or_poseless_id_does_not_resolve(self):
        ctx = make_ctx(FakeWorld())
        ctx.extra["observations"] = {"obs_2": {"observation_id": "obs_2", "kind": "leg"}}

        self.assertIsNone(resolve_observation_pose(ctx, "obs_2"))
        self.assertIsNone(resolve_observation_pose(ctx, "obs_404"))
        self.assertEqual(observation_pose(ctx, ""), (None, None))

    def test_the_id_map_is_newest_first_bounded_and_position_only(self):
        ctx = make_ctx(FakeWorld())
        ctx.extra["observations"] = {
            "obs_1": {"observation_id": "obs_1", "position_flu": [1.0, 0.0, 0.0]},
            "obs_2": {"observation_id": "obs_2"},  # no pose: not a target
            "obs_3": {"observation_id": "obs_3", "position_flu": [3.0, 0.0, 0.0]},
        }

        poses = observation_poses(ctx)

        self.assertEqual(list(poses), ["obs_3", "obs_1"])
        self.assertEqual(poses["obs_3"], [3.0, 0.0, 0.0])
        self.assertEqual(observation_poses(ctx, limit=1), {"obs_3": [3.0, 0.0, 0.0]})

    def test_the_id_map_reads_the_memory_archive_too(self):
        from vlm_nav.memory import ObservationLog

        log = ObservationLog()
        world = FakeWorld(position=(0.0, 4.0, 0.0), yaw_deg=0.0)
        ctx = make_ctx(world, memory=log)
        observe(ctx, {"mode": "forward"})
        ctx.extra[IMAGE_STORE_KEY].clear()

        poses = observation_poses(ctx)

        self.assertEqual(poses, {"obs_1": [0.0, 4.0, 0.0]})


class QueryMemoryTest(unittest.TestCase):
    def test_query_returns_hits_and_evidence(self):
        from vlm_nav.tool_impl_perception import query_memory

        class StubMemory:
            def query(self, q, limit=5):
                return [
                    {"observation_id": "obs_1", "note": "a", "time_s": 1.0,
                     "evidence": [EvidenceRef("observation", "obs_1")]},
                    {"observation_id": "obs_2", "note": "b", "time_s": 2.0,
                     "evidence": [EvidenceRef("image", "frames/obs_2.png")]},
                ]

        ctx = make_ctx(FakeWorld(), memory=StubMemory())
        result = query_memory(ctx, {"query": "cabinet", "limit": 5})
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["count"], 2)
        self.assertEqual([e.ref for e in result.evidence], ["obs_1", "frames/obs_2.png"])
        self.assertIn("observation:obs_1", result.to_prompt())

    def test_query_empty_reports_recent(self):
        from vlm_nav.tool_impl_perception import query_memory

        class StubMemory:
            def query(self, q, limit=5):
                return []

            def recent(self, k=3):
                return [{"observation_id": "obs_9"}]

        result = query_memory(make_ctx(FakeWorld(), memory=StubMemory()), {"query": "zzz"})
        self.assertEqual(result.status, "empty")
        self.assertEqual(result.data["reason"], "no_match")
        self.assertEqual(result.data["recent_observation_ids"], ["obs_9"])
        # a miss must not send the model looking for an id it never got
        self.assertIn("observe", result.data["hint"])

    def test_hits_carry_the_use_and_the_secondary_way_back(self):
        """The recall chain of M2, re-ordered by the evidence contract: the hit
        is an observation id plus a pose, the *primary* advice is a use of the
        evidence (verify / compare / localize) and the flight back is the
        secondary option - named with the reasons that make it worth it."""
        from vlm_nav.memory import ObservationLog

        log = ObservationLog()
        world = FakeWorld(position=(-1.5, 0.9, 0.64), yaw_deg=20.0)
        world.target = SimpleNamespace(name="red object", radius_m=0.5)
        ctx = make_ctx(world, memory=log)
        observe(ctx, {"mode": "forward"})
        world.state.position_flu = np.asarray([1.0, 1.0, 0.3])  # the drone moved on

        result = query_memory(ctx, {"target": "red object"})

        hit = result.data["hits"][0]
        self.assertEqual(hit["observation_id"], "obs_1")
        self.assertEqual(hit["position_flu"], [-1.5, 0.9, 0.64])
        self.assertEqual(hit["yaw_deg"], 20.0)
        self.assertTrue(hit["use"]["why"])
        self.assertEqual(
            result.data["next_action"],
            {"tool": hit["use"]["tool"], "args": hit["use"]["args"]},
        )
        fly = result.data["secondary"]["fly_back"]
        self.assertIn('"observation_id": "obs_1"', json.dumps(fly))
        self.assertTrue(any("away" in reason for reason in fly["why"]))
        self.assertIn("return_to_observation", result.data["hint"])
        json.dumps(result.to_dict())
        self.assertEqual(result.to_prompt().count("\n"), 0)

    def test_a_hit_without_a_pose_gets_no_next_action(self):
        class StubMemory:
            def query(self, q, limit=5):
                return [
                    {
                        "id": "hyp_1",
                        "kind": "identity",
                        "content": "the target is a red box",
                        "refute": [],
                    }
                ]

        result = query_memory(make_ctx(FakeWorld(), memory=StubMemory()), {"query": "box"})

        self.assertEqual(result.status, "ok")
        self.assertNotIn("next_action", result.data["hits"][0])
        # nothing to fly to: the top-level hint must not promise one
        self.assertNotIn("next_action", result.data)
        self.assertNotIn("hint", result.data)

    def test_hits_keep_obs_id_pose_yaw_and_image(self):
        """A recalled frame is only a way back if the hit carries the id, the
        pose it was taken at and its image reference - that is what the model
        turns into a navigate call."""
        from vlm_nav.memory import ObservationLog

        log = ObservationLog()
        world = FakeWorld(position=(1.0, 2.0, 0.5), yaw_deg=30.0)
        world.target = SimpleNamespace(name="bookshelf", radius_m=0.5)
        ctx = make_ctx(world, memory=log)
        observe(ctx, {"mode": "forward"})

        result = query_memory(ctx, {"target": "bookshelf"})

        self.assertEqual(result.status, "ok")
        hit = result.data["hits"][0]
        # memory.ObservationLog entries are keyed obs_id, the tool contract says
        # observation_id: the model must find the id under both names
        self.assertEqual(hit["obs_id"], "obs_1")
        self.assertEqual(hit["observation_id"], "obs_1")
        self.assertEqual(hit["position_flu"], [1.0, 2.0, 0.5])
        self.assertEqual(hit["yaw_deg"], 30.0)
        self.assertTrue(hit["image"].endswith("obs_1.png"))
        json.dumps(result.to_dict())
        self.assertEqual(result.to_prompt().count("\n"), 0)

    def test_hits_lift_a_position_nested_in_data(self):
        class StubMemory:
            def query(self, q, limit=5):
                return [
                    {
                        "id": "hyp_1",
                        "kind": "object",
                        "content": "white chair by the window",
                        "status": "supported",
                        "data": {"position_flu": [1.5, -2.0, 0.5]},
                    }
                ]

        result = query_memory(make_ctx(FakeWorld(), memory=StubMemory()), {"query": "chair"})

        hit = result.data["hits"][0]
        self.assertEqual(hit["id"], "hyp_1")
        self.assertEqual(hit["position_flu"], [1.5, -2.0, 0.5])

    def test_hits_stay_bounded_and_json_plain(self):
        class StubMemory:
            def query(self, q, limit=5):
                return [
                    {
                        "obs_id": "obs_7",
                        "position_flu": np.asarray([0.0, 1.0, 2.0]),
                        "yaw_rad": 0.5,
                        "events": ["target_sighting", "scan"],
                        "labels": ["red", "round"],
                        "image_path": "frames/obs_7.png",
                        "note": "forward view",
                        "extra": {"unbounded": {"nested": "payload"}},
                        "evidence": [],
                    }
                ]

        result = query_memory(make_ctx(FakeWorld(), memory=StubMemory()), {"query": "red"})

        hit = result.data["hits"][0]
        self.assertEqual(hit["obs_id"], "obs_7")
        self.assertEqual(hit["observation_id"], "obs_7")
        self.assertEqual(hit["position_flu"], [0.0, 1.0, 2.0])
        self.assertEqual(hit["yaw_deg"], 28.6)  # derived from yaw_rad
        self.assertEqual(hit["image"], "frames/obs_7.png")
        self.assertEqual(hit["events"], ["target_sighting", "scan"])
        self.assertNotIn("extra", hit)
        json.dumps(result.to_dict())

    def test_observes_label_memory_with_target_name(self):
        """The capture carries the world's target name as the *goal target* of
        the task - what the drone is looking for - and never as a sighting: a
        frame that merely inherits the task target is not evidence of seeing it
        (the goal/sighting split of the evidence memory)."""
        from vlm_nav.memory import ObservationLog
        from vlm_nav.tool_impl_perception import query_memory

        log = ObservationLog()
        world = FakeWorld()
        world.target = SimpleNamespace(name="bookshelf", radius_m=0.5)
        ctx = make_ctx(world, memory=log)
        observe(ctx, {"mode": "forward"})
        obs_id = ctx.extra[IMAGE_STORE_KEY] and next(iter(ctx.extra[IMAGE_STORE_KEY]))
        entry = log.get(obs_id)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.get("target_name"), "bookshelf")
        self.assertEqual(entry.get("goal_target"), "bookshelf")
        self.assertIsNone(entry.get("sighting"))
        hits = query_memory(ctx, {"target": "bookshelf"})
        self.assertEqual(hits.status, "ok")
        self.assertGreaterEqual(hits.data["count"], 1)
        # ... and the hit says what it is: a frame captured while looking, not
        # a verified sighting of the target
        self.assertEqual(hits.data["hits"][0]["evidence_kind"], "goal_frame")
        self.assertIn("no verified sighting", hits.data["evidence_note"])


class MemoryEvidenceWriteTest(unittest.TestCase):
    """What ``observe`` / ``ground_target`` write into memory, and what they
    must never write: the task target is not a sighting, and a null result is
    bound to the view that produced it."""

    def _scene(self, **world_kwargs):
        from vlm_nav.memory import ObservationLog

        log = ObservationLog()
        world = FakeWorld(depth=3.0, hfov_deg=60.0, **world_kwargs)
        world.target = SimpleNamespace(name="red object", radius_m=0.5)
        return log, world, make_ctx(world, memory=log)

    def test_observe_records_the_goal_target_and_never_a_sighting(self):
        log, world, ctx = self._scene(yaw_deg=30.0)

        result = observe(ctx, {"mode": "forward"})

        entry = log.get(result.data["observation_id"])
        self.assertEqual(entry["goal_target"], "red object")
        self.assertIsNone(entry["sighting"])
        self.assertEqual(entry["evidence_kind"], "observation")
        self.assertIn("memory_layers", result.data)

    def test_observe_writes_the_checked_sectors_into_spatial_memory(self):
        log, world, ctx = self._scene(yaw_deg=0.0)

        result = observe(ctx, {"mode": "lookaround", "n_views": 4})

        coverage = log.spatial.coverage()
        self.assertEqual(len(coverage["checked_deg"]), 4)
        self.assertEqual(log.layer_status()["covered_sectors"], 4)
        self.assertEqual(result.data["memory_layers"]["spatial"], "on")
        self.assertEqual(result.data["memory_layers"]["episodic"], "on")

    def test_ground_target_writes_a_sighting_with_frame_candidate_and_basis(self):
        log, world, ctx = self._scene()
        observed = observe(ctx, {"mode": "forward"})

        result = ground_target(
            ctx,
            {"instruction": "red object", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM(json.dumps({"candidates": [
                {"box": [100, 100, 300, 300], "confidence": 0.8, "attrs": {"color": "red"}},
            ]})),
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["memory_sighting"]["status"], "ok:add_sighting")
        self.assertEqual(result.data["memory_sighting"]["added"], 1)
        entry = log.get(observed.data["observation_id"])
        sighting = entry["sighting"]
        self.assertEqual(sighting["target"], "red object")
        self.assertEqual(sighting["candidate"], "cand_1")
        self.assertEqual(sighting["image"], observed.data["observation_id"])
        self.assertIn("ground_target", sighting["basis"])
        self.assertEqual(sighting["confidence"], 0.8)
        self.assertEqual(sighting["attributes"], {"color": "red"})
        self.assertEqual(entry["evidence_kind"], "sighting")
        # the candidate is a place now, and the sighting answers its question
        self.assertEqual(log.spatial.places()[0]["candidates"], ["cand_1"])
        self.assertEqual(log.working.snapshot()["candidates"][0]["id"], "cand_1")

    def test_the_sighting_is_found_by_target_later(self):
        log, world, ctx = self._scene()
        observed = observe(ctx, {"mode": "forward"})
        ground_target(
            ctx,
            {"instruction": "red object", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [100, 100, 300, 300], "confidence": 0.8}]}'),
        )

        hits = query_memory(ctx, {"target": "red object"})

        hit = hits.data["hits"][0]
        self.assertEqual(hit["evidence_kind"], "sighting")
        self.assertEqual(hit["sighting"]["candidate"], "cand_1")
        self.assertEqual(hit["observation_id"], observed.data["observation_id"])
        self.assertEqual(hit["quality"]["level"], "strong")
        self.assertEqual(hits.data["secondary"]["withheld"], True)  # still at the pose
        json.dumps(hits.data)

    def test_a_goal_frame_is_never_returned_as_a_sighting(self):
        log, world, ctx = self._scene()
        observe(ctx, {"mode": "forward"})

        hits = query_memory(ctx, {"target": "red object"})

        hit = hits.data["hits"][0]
        self.assertEqual(hit["evidence_kind"], "goal_frame")
        self.assertNotIn("sighting", hit)
        self.assertEqual(hits.data["n_sighting_hits"], 0)
        self.assertEqual(hits.data["n_goal_frame_hits"], 1)
        self.assertIn("no verified sighting", hits.data["evidence_note"])

    def test_no_candidate_writes_a_scoped_negative(self):
        log, world, ctx = self._scene(yaw_deg=30.0)
        observed = observe(ctx, {"mode": "forward"})

        result = ground_target(
            ctx,
            {"instruction": "red object", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM('{"candidates": []}'),
        )

        self.assertEqual(result.status, "empty")
        self.assertEqual(result.data["memory_negative"]["status"], "ok:add_negative")
        visibility = result.data["visibility"]
        self.assertEqual(visibility["position_flu"], [0.0, 0.0, 0.0])
        self.assertEqual(visibility["yaw_deg"], 30.0)
        self.assertEqual(visibility["hfov_deg"], 60.0)
        self.assertEqual(visibility["observation_id"], observed.data["observation_id"])
        self.assertIn("does not clear the area", result.data["scope_note"])
        self.assertTrue(log.get(observed.data["observation_id"])["negative"])

        hits = query_memory(ctx, {"negative": True})

        hit = hits.data["hits"][0]
        self.assertEqual(hit["evidence_kind"], "negative")
        self.assertEqual(hit["visibility"]["yaw_deg"], 30.0)
        self.assertIn("does not clear the area", hit["scope_note"])

    def test_a_miss_reports_what_was_checked_without_a_result(self):
        log, world, ctx = self._scene(yaw_deg=30.0)
        observed = observe(ctx, {"mode": "forward"})
        ground_target(
            ctx,
            {"instruction": "red object", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM('{"candidates": []}'),
        )

        miss = query_memory(ctx, {"query": "zzz"})

        self.assertEqual(miss.status, "empty")
        self.assertEqual(miss.data["reason"], "no_match")
        checked = miss.data["checked_without_result"]
        self.assertEqual(checked[0]["target"], "red object")
        self.assertEqual(checked[0]["reason"], "no_candidate")
        self.assertIn("does not clear the area", checked[0]["scope_note"])
        self.assertIn("different viewpoint", miss.data["hint"])
        # a miss still names the frames that do exist
        self.assertTrue(miss.data["recent_observation_ids"])

    def test_the_negative_keeps_the_checked_sector_and_the_question(self):
        log, world, ctx = self._scene(yaw_deg=30.0)
        observed = observe(ctx, {"mode": "forward"})
        ground_target(
            ctx,
            {"instruction": "red object", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM('{"candidates": []}'),
        )

        coverage = log.spatial.coverage()
        sector = next(
            item for item in coverage["sectors"] if item["sector"] == [0.0, 45.0]
        )
        self.assertEqual(sector["no_result_for"], ["red object"])
        # the sector is marked as checked, the area is not marked as empty
        self.assertEqual(coverage["unchecked_deg"][0], [-180.0, -135.0])
        snapshot = log.working.snapshot()
        self.assertEqual(snapshot["failures"][0]["reason"], "no_candidate")
        self.assertEqual(snapshot["open_questions"][0]["about"], "red object")

    def test_memory_writes_are_reported_absent_without_a_sink(self):
        ctx = make_ctx(FakeWorld(hfov_deg=60.0), memory=None)

        observed = observe(ctx, {"mode": "forward"})
        grounded = ground_target(
            ctx,
            {"instruction": "red object"},
            vlm=FakeVLM('{"candidates": [{"box": [100, 100, 300, 300]}]}'),
        )

        self.assertEqual(observed.data["memory_sink"], "absent")
        self.assertEqual(observed.data["memory_layers"]["sink"], "absent")
        self.assertEqual(grounded.status, "ok")
        self.assertEqual(grounded.data["memory_sighting"]["status"], "absent")


class EpisodicLayerOffTest(unittest.TestCase):
    """The tools against a memory whose episodic evidence layer is off.

    ``observe`` / ``ground_target`` still write what the spatial and working
    layers own (the candidate and its position), and ``query_memory`` still
    answers with poses, places and the task state - but no hit carries a
    sighting, a picture or a scoped null result: the layer that keeps them is
    off.
    """

    def _scene(self, **kwargs):
        from vlm_nav.memory import ObservationLog

        log = ObservationLog(**kwargs)
        world = FakeWorld(depth=3.0, hfov_deg=60.0)
        world.target = SimpleNamespace(name="red object", radius_m=0.5)
        return log, world, make_ctx(world, memory=log)

    def _ground(self, ctx, observation_id):
        return ground_target(
            ctx,
            {"instruction": "red object", "observation_id": observation_id},
            vlm=FakeVLM('{"candidates": [{"box": [100, 100, 300, 300], "confidence": 0.8}]}'),
        )

    def test_grounding_reports_a_sighting_the_layer_did_not_take(self):
        log, world, ctx = self._scene(episodic_memory=False)
        observed = observe(ctx, {"mode": "forward"})

        result = self._ground(ctx, observed.data["observation_id"])

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["n_candidates"], 1)  # the position channel is intact
        write = result.data["memory_sighting"]
        self.assertEqual(write["status"], "disabled:episodic_memory_off")
        self.assertEqual(write["added"], 0)
        self.assertEqual(write["layers"]["spatial"], "ok:remember_candidate")
        self.assertEqual(log.layer_status()["episodic"], "off")
        self.assertIsNone(log.get(observed.data["observation_id"])["sighting"])

    def test_query_memory_returns_no_evidence_and_no_image(self):
        log, world, ctx = self._scene(episodic_memory=False)
        observed = observe(ctx, {"mode": "forward"})
        self._ground(ctx, observed.data["observation_id"])

        hits = query_memory(ctx, {"target": "red object"})

        self.assertEqual(hits.status, "ok")
        self.assertEqual(hits.data["n_sighting_hits"], 0)
        self.assertGreaterEqual(hits.data["n_goal_frame_hits"], 1)
        pack = hits.data["evidence_pack"]
        self.assertEqual([entry["image"] for entry in pack["images"]], [None])
        self.assertEqual(pack["quality"]["weak"], len(pack["images"]))
        for hit in hits.data["hits"]:
            self.assertEqual(hit["evidence_kind"], "goal_frame")
            self.assertNotIn("image", hit)
            self.assertNotIn("sighting", hit)
            self.assertEqual(hit["quality"]["frame"], "missing")
        self.assertEqual(hits.data["layers"]["episodic"], "off")
        self.assertIn("episodic", hits.data["evidence_note"])
        # the other two layers are still in the pack, and the pose is navigable
        self.assertTrue(hits.data["places"])
        self.assertTrue(hits.data["working"]["candidates"])
        self.assertEqual(hits.data["hits"][0]["position_flu"], [0.0, 0.0, 0.0])

    def test_the_negative_write_is_reported_and_only_the_coverage_survives(self):
        log, world, ctx = self._scene(episodic_memory=False)
        observed = observe(ctx, {"mode": "forward"})

        result = ground_target(
            ctx,
            {"instruction": "red object", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM('{"candidates": []}'),
        )

        self.assertEqual(result.status, "empty")
        self.assertEqual(result.data["memory_negative"]["status"],
                         "disabled:episodic_memory_off")
        self.assertEqual(result.data["memory_negative"]["added"], 0)
        # the boundary of the look is still reported to the model ...
        self.assertIn("does not clear the area", result.data["scope_note"])
        # ... and its sector stays in the spatial layer
        sector = next(
            item for item in log.spatial.coverage()["sectors"] if item["sector"] == [0.0, 45.0]
        )
        self.assertEqual(sector["no_result_for"], ["red object"])
        self.assertEqual(query_memory(ctx, {"negative": True}).status, "empty")

    def test_the_same_writes_store_the_evidence_with_the_layer_on(self):
        log, world, ctx = self._scene()
        observed = observe(ctx, {"mode": "forward"})

        result = self._ground(ctx, observed.data["observation_id"])

        self.assertEqual(result.data["memory_sighting"]["status"], "ok:add_sighting")
        self.assertEqual(result.data["memory_sighting"]["added"], 1)
        hits = query_memory(ctx, {"target": "red object"})
        hit = hits.data["hits"][0]
        self.assertEqual(hit["evidence_kind"], "sighting")
        self.assertEqual(hit["sighting"]["candidate"], "cand_1")
        self.assertTrue(hit["image"])
        self.assertEqual(hit["quality"]["frame"], "stored")


class QueryMemoryEvidencePackTest(unittest.TestCase):
    """``query_memory`` hands back an evidence pack with a use, and treats the
    flight back as a reasoned secondary option - never as a call to where the
    drone already is."""

    def _scene(self, position=(0.0, 0.0, 0.0)):
        from vlm_nav.memory import ObservationLog

        log = ObservationLog()
        world = FakeWorld(position=position, yaw_deg=0.0, depth=3.0, hfov_deg=60.0)
        world.target = SimpleNamespace(name="red object", radius_m=0.5)
        return log, world, make_ctx(world, memory=log)

    def _ground(self, ctx, observation_id, reply):
        return ground_target(
            ctx, {"instruction": "red object", "observation_id": observation_id},
            vlm=FakeVLM(reply),
        )

    def test_each_hit_carries_quality_source_and_a_use(self):
        log, world, ctx = self._scene()
        observed = observe(ctx, {"mode": "forward"})
        self._ground(ctx, observed.data["observation_id"], json.dumps({"candidates": [
            {"box": [100, 100, 300, 300], "confidence": 0.8},
        ]}))

        hits = query_memory(ctx, {"target": "red object"})

        hit = hits.data["hits"][0]
        self.assertEqual(hit["source"], "episodic:sighting")
        self.assertEqual(hit["quality"]["level"], "strong")
        self.assertEqual(hit["quality"]["frame"], "stored")
        self.assertEqual(hit["quality"]["confidence"], 0.8)
        self.assertEqual(hit["use"]["tool"], hits.data["suggested_use"][0]["tool"])
        self.assertTrue(hit["use"]["why"])
        json.dumps(hits.data)

    def test_the_evidence_pack_names_images_poses_quality_and_source(self):
        log, world, ctx = self._scene(position=(1.5, 0.0, 0.5))
        observed = observe(ctx, {"mode": "forward"})
        self._ground(ctx, observed.data["observation_id"], json.dumps({"candidates": [
            {"box": [100, 100, 300, 300], "confidence": 0.9},
        ]}))

        hits = query_memory(ctx, {"target": "red object"})

        pack = hits.data["evidence_pack"]
        image = pack["images"][0]
        self.assertEqual(image["observation_id"], observed.data["observation_id"])
        self.assertTrue(image["image"].endswith(".png"))
        self.assertTrue(Path(image["image"]).is_file())
        self.assertEqual(image["position_flu"], [1.5, 0.0, 0.5])
        self.assertEqual(image["yaw_deg"], 0.0)
        self.assertEqual(image["source"], "episodic:sighting")
        self.assertEqual(image["quality"]["level"], "strong")
        self.assertEqual(pack["quality"], {"strong": 1, "moderate": 0, "weak": 0})
        self.assertIn("sighting", pack["note"])

    def test_suggested_use_covers_compare_verify_and_localize(self):
        log, world, ctx = self._scene()
        observed = observe(ctx, {"mode": "forward"})
        self._ground(ctx, observed.data["observation_id"], json.dumps({"candidates": [
            {"box": [100, 100, 300, 300], "confidence": 0.9},
            {"box": [500, 100, 700, 300], "confidence": 0.5},
        ]}))

        hits = query_memory(ctx, {"target": "red object"})

        tools = [use["tool"] for use in hits.data["suggested_use"]]
        self.assertEqual(tools[0], "compare_candidates")
        self.assertIn("verify_goal", tools)
        self.assertTrue(any(tool in tools for tool in ("ground_target", "inspect_region")))
        for use in hits.data["suggested_use"]:
            self.assertTrue(use["why"], use)
            self.assertIsInstance(use["args"], dict)
            self.assertEqual(use["layer"], "episodic")
        self.assertEqual(
            hits.data["next_action"],
            {"tool": tools[0], "args": hits.data["suggested_use"][0]["args"]},
        )

    def test_flying_back_is_secondary_and_gives_reasons(self):
        log, world, ctx = self._scene(position=(2.0, 0.0, 0.0))
        observed = observe(ctx, {"mode": "forward"})
        self._ground(ctx, observed.data["observation_id"], json.dumps({"candidates": [
            {"box": [100, 100, 300, 300], "confidence": 0.8},
        ]}))
        world.state.position_flu = np.asarray([0.0, 0.0, 0.0])  # the drone flew on

        hits = query_memory(ctx, {"target": "red object"})

        secondary = hits.data["secondary"]
        self.assertFalse(secondary["withheld"])
        fly = secondary["fly_back"]
        self.assertEqual(fly["tool"], "return_to_observation")
        self.assertEqual(fly["args"]["observation_id"], observed.data["observation_id"])
        self.assertAlmostEqual(fly["distance_m"], 2.0, places=1)
        self.assertTrue(any("away" in reason for reason in fly["why"]), fly["why"])
        self.assertTrue(any("quality" in reason for reason in fly["why"]), fly["why"])
        self.assertTrue(fly["expected_benefit"])
        # the primary next step is a use of the evidence, not the flight back
        self.assertNotEqual(hits.data["next_action"]["tool"], "return_to_observation")

    def test_no_fly_back_advice_at_the_current_position(self):
        log, world, ctx = self._scene(position=(2.0, 0.0, 0.0))
        observe(ctx, {"mode": "forward"})

        hits = query_memory(ctx, {"target": "red object"})

        secondary = hits.data["secondary"]
        self.assertTrue(secondary["withheld"])
        self.assertIn("already at this viewpoint", secondary["reason"])
        self.assertNotIn("return_to_observation", json.dumps(secondary))
        self.assertNotIn("return_to_observation", json.dumps(hits.data["hits"]))
        self.assertNotIn("return_to_observation", str(hits.data.get("hint") or ""))
        # what is left is a use of the frame the drone already has
        self.assertIn(hits.data["next_action"]["tool"], ("ground_target", "inspect_region"))

    def test_places_and_task_state_ride_along(self):
        log, world, ctx = self._scene(position=(1.0, 2.0, 0.5))
        observed = observe(ctx, {"mode": "forward"})
        self._ground(ctx, observed.data["observation_id"], json.dumps({"candidates": [
            {"box": [100, 100, 300, 300], "confidence": 0.7},
        ]}))

        hits = query_memory(ctx, {"target": "red object"})

        place = hits.data["places"][0]
        self.assertEqual(place["candidates"], ["cand_1"])
        self.assertEqual(place["kind"], "candidate")
        self.assertIn("evidence", place)
        working = hits.data["working"]
        self.assertEqual(working["candidates"][0]["id"], "cand_1")
        self.assertIsNone(working["subgoal"])
        self.assertEqual(hits.data["layers"]["spatial"], "on")

    def test_the_switches_are_reported_and_the_layers_can_be_off(self):
        from vlm_nav.memory import ObservationLog

        log = ObservationLog(spatial_memory=False)
        world = FakeWorld(position=(0.0, 0.0, 0.0), depth=3.0, hfov_deg=60.0)
        world.target = SimpleNamespace(name="red object", radius_m=0.5)
        ctx = make_ctx(world, memory=log)
        observed = observe(ctx, {"mode": "forward"})

        hits = query_memory(ctx, {"target": "red object"})

        self.assertEqual(hits.data["layers"]["spatial"], "off")
        self.assertEqual(hits.data["places"], [])
        # the episodic evidence is untouched by the switch
        self.assertEqual(hits.data["hits"][0]["observation_id"], observed.data["observation_id"])


class UnifiedObservationTest(unittest.TestCase):
    """The failure of the real trajectory: the model is shown a frame, cites it
    on the next turn, and the tool answers "unknown observation" because the id
    never entered a store the tool reads. Every frame the model is shown is now
    an observation of the episode's repository, so these ids always resolve."""

    def test_a_frame_the_store_knows_is_inspectable_and_groundable(self):
        world = FakeWorld(width=64, height=32, hfov_deg=60.0, depth=3.0)
        ctx = make_ctx(world)
        # exactly what the runtime does for the frame it shows the model
        runtime_frame = torch.full((3, 32, 64), 0.4)
        ctx.next_obs_id()  # the runtime's own id reservation
        observation = register_observation(
            ctx, runtime_frame, kind="evidence", step=1, obs_id="obs_1"
        )
        self.assertEqual(observation.obs_id, "obs_1")

        inspected = inspect_region(ctx, {"observation_id": "obs_1", "region": [100, 100, 200, 200]})
        grounded = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": "obs_1"},
            vlm=FakeVLM('{"candidates": [{"box": [100, 100, 200, 200]}]}'),
        )

        self.assertEqual(inspected.status, "ok")
        self.assertEqual(inspected.data["source_observation_id"], "obs_1")
        self.assertEqual(grounded.status, "ok")
        self.assertEqual(grounded.data["observation_id"], "obs_1")

    def test_an_observation_id_passed_as_a_candidate_is_a_typed_error(self):
        ctx = make_ctx(FakeWorld(hfov_deg=60.0))
        observed = observe(ctx, {"mode": "forward"})

        result = compare_candidates(
            ctx, {"candidates_a": observed.data["observation_id"], "candidates_b": ["cand_1"]}
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["error"], "id_type_mismatch")
        self.assertEqual(result.data["expected"], "candidate_id")
        self.assertEqual(result.data["got"], "obs_1")
        self.assertIn("available_candidate_ids", result.data)

    def test_a_candidate_id_passed_as_an_observation_is_a_typed_error(self):
        ctx = make_ctx(FakeWorld(hfov_deg=60.0))
        observe(ctx, {"mode": "forward"})
        ground_target(
            ctx,
            {"instruction": "chair"},
            vlm=FakeVLM('{"candidates": [{"box": [100, 100, 300, 300]}]}'),
        )

        result = inspect_region(ctx, {"observation_id": "cand_1", "region": [0, 0, 100, 100]})

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.data["error"], "id_type_mismatch")
        self.assertEqual(result.data["expected"], "observation_id")
        self.assertEqual(result.data["actual_type"], "candidate_id")
        self.assertIn("obs_1", result.data["available"])

    def test_a_recalled_hit_is_registered_in_the_store(self):
        from vlm_nav.memory import ObservationLog

        log = ObservationLog()
        world = FakeWorld(position=(1.0, 2.0, 0.5), yaw_deg=30.0, hfov_deg=60.0)
        ctx = make_ctx(world, memory=log)
        observe(ctx, {"mode": "forward"})

        result = query_memory(ctx, {"query": "forward"})

        self.assertEqual(result.status, "ok")
        obs_id = result.data["hits"][0]["observation_id"]
        stored = observation_store(ctx).get(obs_id)
        self.assertIsNotNone(stored)
        self.assertEqual(tuple(round(v, 3) for v in stored.pose_flu), (1.0, 2.0, 0.5))
        self.assertEqual(stored.hfov_deg, 60.0)
        self.assertTrue(stored.image_path.endswith(".png"))
        # ... and the recalled id is a navigable target, not just a printed one
        self.assertIsNotNone(resolve_observation_pose(ctx, obs_id))


class ProjectionIntrinsicsTest(unittest.TestCase):
    """Pixel -> ray geometry runs on the observation's own intrinsics: the
    86 deg default used to bias every lateral offset by ~1.6x."""

    def ground(self, hfov_deg, box=(800, 400, 100, 100), **extra):
        world = FakeWorld(width=640, height=360, yaw_deg=0.0, depth=4.0, hfov_deg=hfov_deg)
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx,
            {
                "instruction": "chair",
                "observation_id": observed.data["observation_id"],
                **extra,
            },
            vlm=FakeVLM(json.dumps({"candidates": [{"box": list(box)}]})),
        )
        return result.data["candidates"][0]

    def test_the_lateral_offset_uses_the_declared_fov(self):
        candidate = self.ground(60.0)

        self.assertEqual(candidate["position_source"], "depth")
        # box centre u = 850/1000 * 640 = 544 px; fx = 640 / (2 tan 30 deg) = 554.26 px
        expected_y = -((544 - 320) / 554.2563) * 4.0
        self.assertAlmostEqual(candidate["position_flu"][1], expected_y, places=2)

    def test_the_86_degree_default_would_have_been_1_6x_off(self):
        wide = self.ground(60.0)["position_flu"][1]
        narrow = self.ground(86.0)["position_flu"][1]

        # tan(43 deg) / tan(30 deg) = 1.615: the old hard-coded FOV stretched
        # every lateral offset by that factor
        self.assertAlmostEqual(abs(narrow) / abs(wide), 1.615, places=2)

    def test_an_unknown_fov_refuses_an_off_axis_projection(self):
        world = FakeWorld(width=640, height=360, depth=4.0)  # no hfov declared
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [800, 400, 200, 200], "distance_m": 5.0}]}'),
        )

        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["position_source"], "intrinsics_unknown")
        self.assertNotIn("position_flu", candidate)
        # the range itself is real knowledge and is reported as such
        self.assertEqual(candidate["range_status"], "known")
        self.assertIn(candidate["range_m"], (4.0, 5.0))

    def test_the_optical_axis_stays_exact_without_a_declared_fov(self):
        world = FakeWorld(width=640, height=360, yaw_deg=0.0, depth=4.0)
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [400, 400, 200, 200]}]}'),  # centre
        )

        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["position_source"], "depth")
        self.assertEqual(candidate["position_flu"], [4.0, 0.0, 0.0])

    def test_an_unknown_range_returns_a_ray_instead_of_a_position(self):
        world = FakeWorld(width=640, height=360, hfov_deg=60.0)
        world.depth_at_pixel = None  # no depth sensor at all
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [400, 400, 200, 200]}]}'),
        )

        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["position_source"], "ray")
        self.assertEqual(candidate["range_status"], "unknown")
        self.assertNotIn("position_flu", candidate)
        self.assertEqual(candidate["ray"]["origin_flu"], [0.0, 0.0, 0.0])
        self.assertEqual(candidate["ray"]["bearing_deg"], 30.0)  # the capture yaw
        self.assertEqual(candidate["ray"]["elevation_deg"], 0.0)
        json.dumps(result.to_dict())

    def test_a_pixel_box_is_normalized_through_the_observation_size(self):
        world = FakeWorld(width=1280, height=720, hfov_deg=60.0, depth=4.0)
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [0, 0, 1280, 720]}]}'),
        )

        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["box_format_used"], "xyxy_px")
        self.assertEqual(candidate["box"], [0.0, 0.0, 1000.0, 1000.0])
        self.assertEqual(candidate["box_uvwh"], [0.0, 0.0, 1000.0, 1000.0])

    def test_a_legacy_box_is_read_as_a_size_box(self):
        world = FakeWorld(width=640, height=360, hfov_deg=60.0, depth=4.0)
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        result = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": observed.data["observation_id"]},
            vlm=FakeVLM('{"candidates": [{"box": [450, 450, 100, 100]}]}'),
        )

        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["box_format_used"], "uvwh_norm")
        self.assertEqual(candidate["box"], [450.0, 450.0, 550.0, 550.0])
        self.assertEqual(candidate["box_uvwh"], [450.0, 450.0, 100.0, 100.0])
        self.assertEqual(candidate["position_source"], "depth")

    def test_a_pixel_box_without_a_known_frame_size_is_refused(self):
        # a pixel box can only be normalized against a frame size; with none
        # known the box is refused instead of being rescaled by a guess
        box, reason = parse_box([0, 0, 4000, 2000])

        self.assertIsNone(box)
        self.assertEqual(reason, "unknown_frame_size")
        self.assertIsNotNone(parse_box([0, 0, 4000, 2000], width=1280, height=720)[0])

    def test_a_crop_projects_through_its_source_observation(self):
        """A crop's pixel grid is not the camera's: grounding a box inside a crop
        must land where the same box lands in the source frame."""
        world = FakeWorld(width=640, height=360, hfov_deg=60.0, depth=4.0)
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        source = observed.data["observation_id"]

        # the same window, asked two ways: directly on the source frame ...
        direct = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": source},
            vlm=FakeVLM('{"candidates": [{"box": [700, 400, 100, 200]}]}'),
        ).data["candidates"][0]
        # ... and on the crop of its right half (crop centre = source u = 750)
        crop = inspect_region(
            ctx, {"observation_id": source, "region": [500, 0, 1000, 1000]}
        ).data["crop_image_ref"]
        through_crop = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": crop},
            vlm=FakeVLM('{"candidates": [{"box": [400, 400, 200, 200]}]}'),
        ).data["candidates"][0]

        self.assertEqual(direct["position_source"], "depth")
        self.assertEqual(through_crop["position_source"], "depth")
        self.assertEqual(through_crop["position_flu"], direct["position_flu"])

    def test_a_crop_reads_depth_at_the_pixel_its_box_maps_back_to(self):
        """A crop's box is in the crop's pixel grid, the depth map is the camera's.

        Reading the crop centre straight off the source depth map measured
        whatever surface sits under that unrelated pixel: the diagnosed run
        reported 1.44 m for a box at 0.95 IoU whose own depth is 3.54 m. The
        depth must come from the pixel the crop's centre maps back to.
        """
        # 2 m in front of the left half of the frame, 6 m in front of the right
        world = FakeWorld(width=640, height=360, hfov_deg=60.0,
                          depth=lambda px: 2.0 if float(px[0]) < 600.0 else 6.0)
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        source = observed.data["observation_id"]
        # the same window: source u = 750 (right half), v = 500
        direct = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": source},
            vlm=FakeVLM('{"candidates": [{"box": [700, 400, 100, 200]}]}'),
        ).data["candidates"][0]
        crop = inspect_region(
            ctx, {"observation_id": source, "region": [500, 0, 1000, 1000]}
        ).data["crop_image_ref"]
        through_crop = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": crop},
            vlm=FakeVLM('{"candidates": [{"box": [400, 400, 200, 200]}]}'),
        ).data["candidates"][0]

        # the crop pixel (500, 500) is source u = 750: the depth under *that*
        # pixel (6 m), not the depth under the crop's own coordinates (2 m).
        # The support samples along the bottom edge are also read, but the
        # centre sample is the one this normal-sized box uses.
        self.assertIn((750.0, 500.0), world.depth_reads)
        self.assertEqual(direct["position_source"], "depth")
        self.assertEqual(direct["range_m"], 6.0)
        self.assertEqual(through_crop["position_source"], "depth")
        self.assertEqual(through_crop["range_m"], 6.0)
        self.assertEqual(through_crop["position_flu"], direct["position_flu"])
        # the box itself stays in the frame it was grounded in (the crop), only
        # the pixel used for the depth is the source frame's
        self.assertEqual(through_crop["observation_id"], crop)
        self.assertEqual(through_crop["box"], [400.0, 400.0, 600.0, 600.0])

    def test_a_box_without_a_world_keeps_its_direction(self):
        """``no_world`` means "no depth sensor here", not "nothing to say".

        The frame still has a capture pose and intrinsics, so the box's direction
        is known even when no position can be derived - the same ray the
        world-less depth path reports. Without it a candidate is unaddressable
        for approach_candidate (no position, no ray), which is how a memory-only
        runtime ended up unable to act on what it had seen.
        """
        world = FakeWorld(width=640, height=360, hfov_deg=60.0, depth=4.0)
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        source = observed.data["observation_id"]
        ctx.world = None  # the evidence service holds the frame, not the drone

        result = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": source},
            vlm=FakeVLM('{"candidates": [{"box": [400, 400, 200, 200]}]}'),
        )

        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["position_source"], "no_world")
        self.assertEqual(candidate["range_status"], "unknown")
        self.assertNotIn("position_flu", candidate)
        # the ray of the same box under the "no depth sensor" reason
        self.assertEqual(candidate["ray"]["origin_flu"], [0.0, 0.0, 0.0])
        self.assertEqual(candidate["ray"]["bearing_deg"], 30.0)  # the capture yaw
        self.assertEqual(candidate["ray"]["elevation_deg"], 0.0)
        json.dumps(result.to_dict())

    def test_a_crop_whose_source_is_gone_has_no_position(self):
        world = FakeWorld(width=640, height=360, hfov_deg=60.0, depth=4.0)
        ctx = make_ctx(world)
        observed = observe(ctx, {"mode": "forward"})
        source = observed.data["observation_id"]
        crop = inspect_region(
            ctx, {"observation_id": source, "region": [500, 0, 500, 1000]}
        ).data["crop_image_ref"]
        # the source frame is gone from every cache: the crop's pixel grid can no
        # longer be mapped onto a camera, so no position may be derived
        ctx.extra[IMAGE_STORE_KEY].pop(source)
        observation_store(ctx).forget(source)

        result = ground_target(
            ctx,
            {"instruction": "chair", "observation_id": crop},
            vlm=FakeVLM('{"candidates": [{"box": [400, 400, 200, 200], "distance_m": 3.0}]}'),
        )

        candidate = result.data["candidates"][0]
        self.assertEqual(candidate["position_source"], "crop_source_unknown")
        self.assertNotIn("position_flu", candidate)


DIRECTIONS = (
    "ahead",
    "ahead-left",
    "left",
    "behind-left",
    "behind",
    "behind-right",
    "right",
    "ahead-right",
    "unknown",
)


def wrap_deg(value):
    return (float(value) + 180.0) % 360.0 - 180.0


class ObserveViewContextTest(unittest.TestCase):
    """The visual-context contract of ``observe``: which way each view looked,
    which 45 deg sectors are checked, and how to pull a view's pixels back."""

    def test_view_index_names_every_view_by_its_direction(self):
        world = FakeWorld(yaw_deg=30.0)
        ctx = make_ctx(world)
        result = observe(ctx, {"mode": "lookaround", "n_views": 4})

        view_index = result.data["view_index"]
        self.assertEqual([entry["obs_id"] for entry in view_index], result.data["image_refs"])
        for entry, heading in zip(view_index, result.data["headings_deg"]):
            self.assertAlmostEqual(entry["yaw_deg"], heading, places=3)
            self.assertTrue(entry["note"])
            self.assertIn(entry["direction"], DIRECTIONS)
        # the headings of a 4-view sweep from yaw 30: 30, 120, 210, 300
        self.assertEqual(
            [entry["direction"] for entry in view_index],
            ["ahead", "left", "behind", "right"],
        )
        self.assertEqual(result.data["coverage"]["reference_yaw_deg"], 30.0)

    def test_a_partial_scan_reports_the_unchecked_directions(self):
        world = FakeWorld(yaw_deg=30.0)
        ctx = make_ctx(world)
        result = observe(ctx, {"mode": "forward"})

        coverage = result.data["coverage"]
        self.assertEqual(coverage["checked_deg"], [[0.0, 45.0]])
        self.assertEqual(len(coverage["unchecked_deg"]), 7)
        self.assertFalse(coverage["all_directions_covered"])
        worth = coverage["worth_looking"]
        self.assertTrue(worth)
        self.assertIn(worth[0]["obs_id"], result.data["image_refs"])
        self.assertIn("unchecked", worth[0]["reason"])

    def test_a_full_sweep_reports_every_sector_as_checked(self):
        ctx = make_ctx(FakeWorld(yaw_deg=0.0))
        result = observe(ctx, {"mode": "lookaround", "n_views": 8})

        coverage = result.data["coverage"]
        self.assertEqual(len(coverage["checked_deg"]), 8)
        self.assertEqual(coverage["unchecked_deg"], [])
        self.assertTrue(coverage["all_directions_covered"])
        # nothing is unchecked, so no view is singled out as worth a second look:
        # the coverage note names the (coarse) resolution and the expansion call
        self.assertEqual(coverage["worth_looking"], [])
        self.assertIn("inspect_region", coverage["note"])
        covered = sum(hi - lo for lo, hi in coverage["checked_deg"])
        self.assertAlmostEqual(covered, 360.0, places=3)

    def test_a_narrow_scan_leaves_the_rest_of_the_circle_unchecked(self):
        ctx = make_ctx(FakeWorld(yaw_deg=0.0))
        result = observe(ctx, {"mode": "lookaround", "n_views": 3, "spread_deg": 60.0})

        coverage = result.data["coverage"]
        # headings -30, 0, 30: two 45 deg sectors covered, six unchecked
        self.assertEqual(sorted(coverage["checked_deg"]), [[-45.0, 0.0], [0.0, 45.0]])
        self.assertEqual(len(coverage["unchecked_deg"]), 6)
        self.assertTrue(coverage["worth_looking"])

    def test_expansion_guidance_names_the_calls_that_pull_a_view_back(self):
        ctx = make_ctx(FakeWorld(yaw_deg=10.0))
        result = observe(ctx, {"mode": "lookaround", "n_views": 4})

        expansion = result.data["expansion"]
        self.assertIn("inspect_region", expansion["how"])
        for key in ("full_view", "zoom_example"):
            call = expansion[key]
            self.assertEqual(call["tool"], "inspect_region")
            self.assertIn(call["args"]["observation_id"], result.data["image_refs"])
            self.assertEqual(len(call["args"]["region"]), 4)
        self.assertEqual(expansion["full_view"]["args"]["region"], [0, 0, 1000, 1000])
        # no pitched frame in this capture: the note names the downward check
        self.assertEqual(expansion["pitch_down"]["tool"], "observe")
        self.assertEqual(expansion["pitch_down"]["args"]["mode"], "pitch_down")
        json.dumps(result.to_dict())

    def test_pitched_views_are_marked_down_in_the_view_index(self):
        ctx = make_ctx(FakeWorld(yaw_deg=0.0))
        result = observe(ctx, {"mode": "pitch_down"})

        pitched = [
            entry for entry in result.data["view_index"] if entry["direction"].startswith("down-")
        ]
        self.assertTrue(pitched)
        self.assertIn("down", pitched[0]["note"].lower())
        self.assertTrue(result.data["coverage"]["down_checked"])
        self.assertNotIn("pitch_down", result.data["expansion"])

    def test_without_a_heading_the_directions_are_unknown(self):
        world = NoPoseWorld(torch.full((3, 8, 8), 0.5))
        ctx = make_ctx(world)
        result = observe(ctx, {"mode": "forward"})

        self.assertEqual(result.data["view_index"][0]["direction"], "unknown")
        coverage = result.data["coverage"]
        self.assertFalse(coverage["directions_known"])
        self.assertIsNone(coverage["reference_yaw_deg"])
        self.assertEqual(coverage["checked_deg"], [])
        self.assertEqual(coverage["unchecked_deg"], [])
        self.assertIn("heading", coverage["note"])

    def test_the_scan_covers_are_written_for_the_grounding_ledger(self):
        ctx = make_ctx(FakeWorld(yaw_deg=0.0))
        observe(ctx, {"mode": "forward"})

        recorded = ctx.extra["perception.view_coverage"]
        self.assertEqual(recorded["reference_yaw_deg"], 0.0)
        self.assertEqual(recorded["checked_deg"], [[0.0, 45.0]])
        self.assertEqual(len(recorded["unchecked_deg"]), 7)

    def test_observe_without_arguments_looks_forward(self):
        ctx = make_ctx(FakeWorld(yaw_deg=30.0))

        result = observe(ctx, {})

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["mode"], "forward")
        self.assertEqual(len(result.data["image_refs"]), 1)
        self.assertEqual(len(result.data["view_index"]), 1)
        self.assertEqual(len(result.data["coverage"]["checked_deg"]), 1)

    def test_a_blank_mode_defaults_to_forward(self):
        ctx = make_ctx(FakeWorld(yaw_deg=30.0))

        result = observe(ctx, {"mode": "  "})

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["mode"], "forward")


class GroundingStallTest(unittest.TestCase):
    """The anti-loop guard: the same evidence may not be ground-repeated forever.

    ``obs_N`` changes every turn while the drone hovers at one pose; treating
    each new id as new evidence is exactly the loop that burned a real budget on
    ``no_candidate``, so attempts are keyed by the *evidence* (instruction,
    pose/heading, region), never by the observation id.
    """

    EMPTY = '{"candidates": []}'
    HIT = '{"candidates": [{"box": [400, 400, 600, 600], "confidence": 0.9}]}'

    def setUp(self):
        self.world = FakeWorld(yaw_deg=0.0)
        self.ctx = make_ctx(self.world)
        self.vlm = FakeVLM(self.EMPTY)

    def ground(self, instruction="red chair", **kwargs):
        observed = observe(self.ctx, kwargs.pop("observe_args", {"mode": "forward"}))
        return ground_target(
            self.ctx,
            {"instruction": instruction, "observation_id": observed.data["observation_id"]},
            vlm=self.vlm,
        )

    def test_the_second_failure_on_the_same_evidence_flags_the_repetition(self):
        first = self.ground()
        second = self.ground()

        self.assertFalse(first.data.get("repeated_grounding", False))
        self.assertTrue(second.data["repeated_grounding"])
        change = second.data["strategy_change"]
        self.assertEqual(change["kind"], "inspect_region")
        self.assertEqual(change["tool"], "inspect_region")
        self.assertTrue(change["why"])
        self.assertEqual(len(change["alternatives"]), 2)
        self.assertEqual(
            {item["kind"] for item in change["alternatives"]},
            {"change_view", "change_viewpoint"},
        )
        json.dumps(second.to_dict())

    def test_the_third_attempt_is_blocked_without_another_model_call(self):
        self.ground()
        self.ground()
        # each empty grounding is one box-only retry: 2 views x 2 calls
        self.assertEqual(len(self.vlm.calls), 4)

        third = self.ground()

        self.assertEqual(third.status, "blocked")
        self.assertEqual(third.data["reason"], "stagnant_grounding")
        self.assertTrue(third.data["repeated_grounding"])
        self.assertEqual(len(self.vlm.calls), 4)  # the model was not asked again
        self.assertEqual(third.data["strategy_change"]["kind"], "change_viewpoint")
        self.assertEqual(third.data["strategy_change"]["tool"], "propose_viewpoints")
        self.assertTrue(third.data["strategy_change"]["why"])
        self.assertEqual(third.data["attempts"], 3)
        self.assertTrue(third.evidence)  # the frame is still named as evidence

    def test_the_suggested_turn_points_into_an_unchecked_sector(self):
        self.ground()
        self.ground()
        blocked = self.ground()

        alternatives = {
            item["kind"]: item for item in blocked.data["strategy_change"]["alternatives"]
        }
        self.assertIn("change_view", alternatives)
        args = alternatives["change_view"]["args"]
        self.assertEqual(args["mode"], "forward")
        coverage = self.ctx.extra["perception.view_coverage"]
        heading = wrap_deg(coverage["reference_yaw_deg"] + args["yaw_deg"])
        self.assertTrue(
            any(lo <= heading < hi for lo, hi in coverage["unchecked_deg"]),
            (heading, coverage["unchecked_deg"]),
        )

    def test_a_new_candidate_resets_the_stall_counter(self):
        self.ground()
        self.vlm.reply = self.HIT
        self.assertEqual(self.ground().data["n_candidates"], 1)
        self.vlm.reply = self.EMPTY
        after_success = self.ground()

        self.assertFalse(after_success.data.get("repeated_grounding", False))
        self.assertTrue(self.ground().data["repeated_grounding"])

    def test_a_different_heading_is_different_evidence(self):
        self.ground()
        turned = self.ground(observe_args={"mode": "forward", "yaw_deg": 90.0})

        self.assertFalse(turned.data.get("repeated_grounding", False))

    def test_a_different_pose_is_different_evidence(self):
        self.ground()
        self.world.state.position_flu = np.asarray([1.5, 0.0, 0.0], dtype=np.float64)
        moved = self.ground()

        self.assertFalse(moved.data.get("repeated_grounding", False))

    def test_a_different_instruction_is_a_different_query(self):
        self.ground(instruction="red chair")
        other = self.ground(instruction="blue door")

        self.assertFalse(other.data.get("repeated_grounding", False))

    def test_a_cropped_region_is_different_evidence(self):
        first = self.ground()
        source = first.data["observation_id"]
        crop = inspect_region(
            self.ctx, {"observation_id": source, "region": [300, 300, 700, 700]}
        ).data["crop_image_ref"]

        result = ground_target(
            self.ctx,
            {"instruction": "red chair", "observation_id": crop},
            vlm=self.vlm,
        )

        self.assertFalse(result.data.get("repeated_grounding", False))

    def test_a_failed_attempt_without_a_backend_does_not_poison_the_ledger(self):
        for _ in range(3):
            observed = observe(self.ctx, {"mode": "forward"})
            ground_target(
                self.ctx,
                {"instruction": "red chair", "observation_id": observed.data["observation_id"]},
                vlm=None,
            )

        result = self.ground()

        self.assertFalse(result.data.get("repeated_grounding", False))


if __name__ == "__main__":
    unittest.main()
