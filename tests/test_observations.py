"""Unit tests for ``vlm_nav.observations``: the unified observation repository,
the typed id contract and the render intrinsics it stores.

The failure these tests pin down is the one seen in a real trajectory: the
runtime showed the model ``obs_10``, the model acted on it, and the tools said
"unknown observation" - the frame id the model was shown never entered a store
the tools read. Everything below is checked against that contract: an id the
model can see resolves to image + pose + intrinsics, and an id of the wrong type
produces a typed error that names the ids of the type the argument wants.
"""

import math
from types import SimpleNamespace
import unittest

import numpy as np

from vlm_nav.memory import ObservationLog
from vlm_nav.observations import (
    CANDIDATE_STORE_KEY,
    CANDIDATE_ID_TYPE,
    KIND_EVIDENCE,
    KIND_SCAN,
    OBSERVATION_ID_TYPE,
    OBSERVATION_STORE_KEY,
    VIEWPOINT_ID_TYPE,
    VIEWPOINT_STORE_KEY,
    IdTypeError,
    Observation,
    ObservationStore,
    available_candidate_ids,
    available_observation_ids,
    classify_id,
    known_observation_id,
    lookup_pose,
    observation_store,
    parse_id_index,
    register_observation,
    require_candidate_id,
    require_observation_id,
    require_viewpoint_id,
    resolve_intrinsics,
)
from vlm_nav.tools import ToolContext


class SensorWorld:
    """Production-shaped world: pose + a camera object that declares the FOV."""

    def __init__(self, hfov_deg=60.0, width=640, height=360, position=(1.0, 2.0, 0.5), yaw_deg=30.0):
        sensor = SimpleNamespace(hfov=hfov_deg, W=width, H=height)
        self.env = SimpleNamespace(sensors={"rgb_camera": sensor, "imu": SimpleNamespace()})
        self.state = SimpleNamespace(
            position_flu=np.asarray(position, dtype=np.float64),
            yaw_rad=math.radians(yaw_deg),
        )

    @property
    def position_flu(self):
        return self.state.position_flu

    @property
    def yaw_rad(self):
        return self.state.yaw_rad

    def render(self):
        return np.zeros((3, 48, 64), dtype=np.float32)


class BareWorld:
    """World without pose and without any declared camera."""

    def render(self):
        return np.zeros((3, 16, 16), dtype=np.float32)


def make_ctx(world=None, **extra):
    return ToolContext(world=world, episode_id="ep_test", extra=dict(extra))


class IntrinsicsTest(unittest.TestCase):
    def test_hfov_comes_from_the_world_not_a_default(self):
        world = SensorWorld(hfov_deg=60.0, width=320, height=180)
        world.hfov_deg = 60.0  # SplatWorldCPU-style declaration

        intrinsics = resolve_intrinsics(make_ctx(world))

        self.assertEqual(intrinsics["hfov_deg"], 60.0)
        self.assertEqual(intrinsics["hfov_source"], "hfov_deg")
        self.assertEqual((intrinsics["width"], intrinsics["height"]), (320, 180))
        self.assertNotEqual(intrinsics["hfov_deg"], 86.0)

    def test_hfov_comes_from_the_env_camera_object(self):
        intrinsics = resolve_intrinsics(make_ctx(SensorWorld(hfov_deg=60.0)))

        self.assertEqual(intrinsics["hfov_deg"], 60.0)
        self.assertEqual(intrinsics["hfov_source"], "sensors.rgb_camera.hfov")
        self.assertEqual((intrinsics["width"], intrinsics["height"]), (640, 360))

    def test_an_explicit_override_beats_the_world(self):
        ctx = make_ctx(SensorWorld(hfov_deg=86.0), **{"perception.hfov_deg": 60.0})
        self.assertEqual(resolve_intrinsics(ctx)["hfov_deg"], 60.0)
        self.assertEqual(resolve_intrinsics(ctx)["hfov_source"], "ctx.extra")

        pinned = resolve_intrinsics(ctx, override={"width": 100, "height": 50, "hfov_deg": 40.0})
        self.assertEqual(pinned["hfov_deg"], 40.0)
        self.assertEqual((pinned["width"], pinned["height"]), (100, 50))

    def test_an_undeclared_fov_stays_unknown(self):
        intrinsics = resolve_intrinsics(make_ctx(BareWorld()))

        self.assertIsNone(intrinsics["hfov_deg"])
        self.assertEqual(intrinsics["hfov_source"], "unknown")

    def test_the_rendered_frame_size_wins_over_the_declared_one(self):
        ctx = make_ctx(SensorWorld(hfov_deg=60.0, width=640, height=360))
        frame = np.zeros((3, 90, 160), dtype=np.float32)

        intrinsics = resolve_intrinsics(ctx, frame=frame)

        self.assertEqual((intrinsics["width"], intrinsics["height"]), (160, 90))
        self.assertEqual(intrinsics["hfov_deg"], 60.0)


class ObservationStoreTest(unittest.TestCase):
    def make_store(self, **kwargs):
        store = ObservationStore(**kwargs)
        store.register("", (1.0, 2.0, 3.0), 90.0, {"width": 320, "height": 180, "hfov_deg": 60.0}, KIND_SCAN, 3, 10.0)
        store.register("", (4.0, 5.0, 6.0), 45.0, {"width": 320, "height": 180, "hfov_deg": 60.0}, KIND_EVIDENCE, 4, 20.0)
        return store

    def test_register_get_and_ordering(self):
        store = self.make_store()

        self.assertEqual(store.ids(), ["obs_2", "obs_1"])  # newest first
        self.assertEqual(store.ids(kind=KIND_SCAN), ["obs_1"])
        self.assertEqual(store.latest().obs_id, "obs_2")
        self.assertEqual(store.latest(kind=KIND_SCAN).obs_id, "obs_1")
        record = store.get("obs_1")
        self.assertEqual(record.pose_flu, (1.0, 2.0, 3.0))
        self.assertEqual(record.yaw_deg, 90.0)
        self.assertEqual(record.step, 3)
        self.assertEqual(record.timestamp, 10.0)
        self.assertEqual(record.hfov_deg, 60.0)
        self.assertEqual(record.pose_dict(), {"position_flu": [1.0, 2.0, 3.0], "yaw_deg": 90.0})

    def test_bare_index_and_spelling_aliases_resolve(self):
        store = self.make_store()

        self.assertEqual(store.resolve_id("2"), "obs_2")
        self.assertEqual(store.resolve_id("OBS_2"), "obs_2")
        self.assertEqual(store.resolve_id("obs#1"), "obs_1")
        self.assertEqual(store.get("2").obs_id, "obs_2")
        self.assertIsNone(store.resolve_id("obs_9"))
        self.assertIsNone(store.resolve_id("cand_2"))  # another type: never here
        self.assertIsNone(store.resolve_id(None))

    def test_reserved_ids_are_never_minted_twice(self):
        ctx = make_ctx()
        store = observation_store(ctx, create=True)
        first = ctx.next_obs_id()
        second = ctx.next_obs_id()

        store.register(frame=None, kind=KIND_EVIDENCE, step=1, obs_id=first)
        store.register(frame=None, kind=KIND_EVIDENCE, step=1, obs_id=second)

        self.assertEqual((first, second), ("obs_1", "obs_2"))
        self.assertEqual(store.next_id(), "obs_3")
        self.assertEqual(store.ids(), ["obs_2", "obs_1"])

    def test_eviction_drops_the_oldest_records(self):
        store = ObservationStore(max_entries=2)
        for step in range(4):
            store.register("", (float(step), 0.0, 0.0), 0.0, {}, KIND_EVIDENCE, step, float(step))

        self.assertEqual(store.ids(), ["obs_4", "obs_3"])
        self.assertIsNone(store.get("obs_1"))
        self.assertEqual(store.latest().step, 3)

    def test_reregistering_updates_in_place_and_keeps_position(self):
        store = self.make_store()
        store.register(
            "frames/obs_1.png", (7.0, 7.0, 7.0), 10.0, {"width": 10, "height": 10, "hfov_deg": 60.0},
            KIND_SCAN, 3, 11.0, obs_id="obs_1",
        )

        self.assertEqual(store.ids(), ["obs_2", "obs_1"])  # order kept
        record = store.get("obs_1")
        self.assertEqual(record.image_path, "frames/obs_1.png")
        self.assertEqual(record.pose_flu, (7.0, 7.0, 7.0))
        self.assertEqual(record.timestamp, 11.0)

    def test_an_unknown_pose_is_stored_honestly(self):
        store = ObservationStore()
        store.register("", None, None, {"width": 320, "height": 180, "hfov_deg": None}, KIND_EVIDENCE, 1)

        record = store.latest()
        self.assertFalse(record.has_pose)
        self.assertFalse(record.has_yaw)
        self.assertTrue(all(math.isnan(v) for v in record.pose_flu))
        self.assertIsNone(record.hfov_deg)
        self.assertEqual(record.pose_dict()["yaw_deg"], None)
        # a place nobody can point at is not offered as a target ...
        self.assertEqual(store.poses(), {})
        # ... but the frame is still addressable
        self.assertEqual(store.ids(), ["obs_1"])

    def test_poses_are_newest_first_and_position_only(self):
        store = ObservationStore()
        store.register("", (1.0, 0.0, 0.0), 0.0, {}, KIND_EVIDENCE, 1)
        store.register("", None, None, {}, KIND_EVIDENCE, 2)

        self.assertEqual(list(store.poses()), ["obs_1"])

    def test_state_poses_of_the_navigation_tools_are_read(self):
        ctx = make_ctx()
        ctx.extra["observations"] = {
            "obs_7": {"observation_id": "obs_7", "position_flu": [1.0, 2.0, 0.5], "yaw_rad": 0.25}
        }

        position, yaw_rad = lookup_pose(ctx, "obs_7")

        self.assertEqual(position, (1.0, 2.0, 0.5))
        self.assertAlmostEqual(yaw_rad, 0.25, places=6)
        self.assertEqual(known_observation_id(ctx, "obs_7"), "obs_7")
        self.assertIn("obs_7", available_observation_ids(ctx))

    def test_the_store_is_the_first_pose_source(self):
        ctx = make_ctx()
        ctx.extra["observations"] = {"obs_1": {"position_flu": [9.0, 9.0, 9.0], "yaw_rad": 0.0}}
        register_observation(
            ctx, None, kind=KIND_EVIDENCE, step=1, obs_id="obs_1",
            pose_flu=(1.0, 2.0, 3.0), yaw_deg=90.0, intrinsics={"width": 10, "height": 10, "hfov_deg": 60.0},
        )

        position, yaw_rad = lookup_pose(ctx, "obs_1")

        self.assertEqual(position, (1.0, 2.0, 3.0))
        self.assertAlmostEqual(math.degrees(yaw_rad), 90.0, places=6)


class RegisterObservationTest(unittest.TestCase):
    def test_world_pose_yaw_and_intrinsics_are_recorded(self):
        ctx = make_ctx(SensorWorld(position=(2.5, -1.0, 0.75), yaw_deg=120.0))
        observation = register_observation(ctx, SensorWorld().render(), kind=KIND_SCAN, step=4, timestamp=1.5)

        self.assertEqual(observation.obs_id, "obs_1")
        self.assertEqual(ctx.observation_seq, 1)
        self.assertEqual(observation.kind, "scan")
        self.assertEqual(observation.step, 4)
        self.assertEqual(observation.timestamp, 1.5)
        self.assertEqual(tuple(round(v, 3) for v in observation.pose_flu), (2.5, -1.0, 0.75))
        self.assertAlmostEqual(observation.yaw_deg, 120.0, places=3)
        self.assertEqual(observation.hfov_deg, 60.0)
        self.assertEqual((observation.width, observation.height), (64, 48))  # rendered size
        self.assertIsNotNone(observation.frame)

    def test_an_explicit_pose_wins_over_the_world(self):
        """A lookaround view records the heading it was rendered at, not the
        heading the world was restored to afterwards."""
        ctx = make_ctx(SensorWorld(yaw_deg=0.0))
        observation = register_observation(ctx, None, kind=KIND_SCAN, yaw_deg=270.0, pose_flu=(0.0, 0.0, 1.0))

        self.assertAlmostEqual(observation.yaw_deg, 270.0, places=6)
        self.assertEqual(observation.pose_flu, (0.0, 0.0, 1.0))

    def test_adopting_an_id_keeps_the_single_namespace(self):
        ctx = make_ctx(SensorWorld())
        ctx.next_obs_id()
        ctx.next_obs_id()
        register_observation(ctx, None, obs_id="obs_2", kind=KIND_EVIDENCE)

        self.assertEqual(observation_store(ctx).ids(), ["obs_2"])
        self.assertEqual(ctx.next_obs_id(), "obs_3")

    def test_a_world_without_pose_records_nan_not_zeros(self):
        ctx = make_ctx(BareWorld())
        observation = register_observation(ctx, BareWorld().render())

        self.assertFalse(observation.has_pose)
        self.assertTrue(all(math.isnan(v) for v in observation.pose_flu))


class TypedIdTest(unittest.TestCase):
    def make_ctx(self):
        ctx = make_ctx()
        store = observation_store(ctx, create=True)
        store.register("", (0.0, 0.0, 0.0), 0.0, {"width": 10, "height": 10, "hfov_deg": 60.0}, KIND_EVIDENCE, 1)
        store.register("", (1.0, 0.0, 0.0), 0.0, {"width": 10, "height": 10, "hfov_deg": 60.0}, KIND_EVIDENCE, 2)
        ctx.extra[CANDIDATE_STORE_KEY] = {"cand_1": {"id": "cand_1", "position_flu": [3.0, 0.0, 0.0]}}
        ctx.extra[VIEWPOINT_STORE_KEY] = {"vp_5": {"viewpoint_id": "vp_5", "position": [4.0, 0.0, 0.0]}}
        return ctx

    def test_each_type_is_classified_and_aliased(self):
        self.assertEqual(classify_id("obs_10"), OBSERVATION_ID_TYPE)
        self.assertEqual(classify_id("cand_2"), CANDIDATE_ID_TYPE)
        self.assertEqual(classify_id("vp_8"), VIEWPOINT_ID_TYPE)
        self.assertIsNone(classify_id("7"))  # a bare index carries no type
        self.assertIsNone(classify_id("hyp_1"))
        self.assertEqual(parse_id_index("7", OBSERVATION_ID_TYPE), "obs_7")
        self.assertEqual(parse_id_index("cand_7", CANDIDATE_ID_TYPE), "cand_7")
        self.assertIsNone(parse_id_index("cand_7", VIEWPOINT_ID_TYPE))

    def test_a_valid_id_of_each_type_resolves(self):
        ctx = self.make_ctx()

        self.assertEqual(require_observation_id("obs_2", ctx), "obs_2")
        self.assertEqual(require_observation_id("2", ctx), "obs_2")  # bare index alias
        self.assertEqual(require_candidate_id("cand_1", ctx), "cand_1")
        self.assertEqual(require_viewpoint_id("vp_5", ctx), "vp_5")

    def test_an_observation_id_passed_as_a_candidate_is_a_type_error(self):
        ctx = self.make_ctx()

        with self.assertRaises(IdTypeError) as caught:
            require_candidate_id("obs_2", ctx)

        error = caught.exception
        self.assertEqual(error.reason, "id_type_mismatch")
        self.assertEqual(error.expected, CANDIDATE_ID_TYPE)
        self.assertEqual(error.actual, OBSERVATION_ID_TYPE)
        self.assertEqual(error.available, ["cand_1"])  # ids of the WANTED type
        self.assertIn("observation_id", error.message())
        self.assertIn("cand_1", error.message())
        data = error.to_data()
        self.assertEqual(data["error"], "id_type_mismatch")
        self.assertEqual(data["got"], "obs_2")
        self.assertEqual(data["available_candidate_ids"], ["cand_1"])

    def test_a_candidate_id_passed_as_an_observation_is_a_type_error(self):
        ctx = self.make_ctx()

        with self.assertRaises(IdTypeError) as caught:
            require_observation_id("cand_1", ctx)

        error = caught.exception
        self.assertEqual(error.reason, "id_type_mismatch")
        self.assertEqual(error.expected, OBSERVATION_ID_TYPE)
        self.assertEqual(error.actual, CANDIDATE_ID_TYPE)
        self.assertIn("obs_2", error.available)  # the usable observations
        self.assertIn("candidate_id", error.message())

    def test_a_viewpoint_id_passed_as_an_observation_is_a_type_error(self):
        ctx = self.make_ctx()

        with self.assertRaises(IdTypeError) as caught:
            require_observation_id("vp_5", ctx)

        self.assertEqual(caught.exception.reason, "id_type_mismatch")
        self.assertEqual(caught.exception.actual, VIEWPOINT_ID_TYPE)

    def test_an_unknown_id_of_the_right_type_is_reported_with_alternatives(self):
        ctx = self.make_ctx()

        with self.assertRaises(IdTypeError) as caught:
            require_observation_id("obs_42", ctx)

        error = caught.exception
        self.assertEqual(error.reason, "unknown_observation_id")
        self.assertEqual(error.available[:2], ["obs_2", "obs_1"])
        self.assertIn("obs_42", error.message())
        self.assertEqual(error.to_data()["observation_id"], "obs_42")

    def test_missing_and_untyped_ids_are_reported(self):
        ctx = self.make_ctx()

        for raw in (None, "", "   "):
            with self.subTest(raw=raw):
                with self.assertRaises(IdTypeError) as caught:
                    require_observation_id(raw, ctx)
                self.assertEqual(caught.exception.reason, "missing_id")
        with self.assertRaises(IdTypeError) as caught:
            require_candidate_id("the chair", ctx)
        self.assertEqual(caught.exception.reason, "unknown_candidate_id")
        self.assertEqual(caught.exception.available, ["cand_1"])

    def test_available_id_lists_are_bounded_and_newest_first(self):
        ctx = self.make_ctx()

        self.assertEqual(available_observation_ids(ctx, limit=1), ["obs_2"])
        self.assertEqual(available_candidate_ids(ctx), ["cand_1"])


class MemoryNamespaceTest(unittest.TestCase):
    def test_a_recalled_id_resolves_in_the_store(self):
        """query_memory hits are bridges, not a second id space: an id the model
        recalls is registered in the store with its image, pose and intrinsics."""
        log = ObservationLog()
        obs_id = log.record(
            np.zeros((4, 4, 3), dtype=np.uint8),
            pose=[1.0, 2.0, 0.5],
            yaw=math.radians(30.0),
            step=2,
            note="forward view",
            intrinsics={"width": 320, "height": 180, "hfov_deg": 60.0},
            image_path="frames/obs_1.png",
        )
        ctx = make_ctx(memory=log)
        entry = log.get(obs_id)

        self.assertEqual(obs_id, "obs_1")
        self.assertEqual(entry["intrinsics"]["hfov_deg"], 60.0)
        adopted = register_observation(
            ctx,
            None,
            kind=KIND_EVIDENCE,
            step=entry.get("step") or 0,
            image_path=entry.get("image") or "",
            obs_id=obs_id,
            pose_flu=entry.get("position_flu"),
            yaw_deg=entry.get("yaw_deg"),
            intrinsics=entry.get("intrinsics"),
        )

        self.assertEqual(adopted.obs_id, "obs_1")
        stored = observation_store(ctx).get("obs_1")
        self.assertEqual(stored.pose_flu, (1.0, 2.0, 0.5))
        self.assertTrue(stored.image_path.endswith("obs_1.png"))
        self.assertEqual(stored.hfov_deg, 60.0)
        self.assertEqual(require_observation_id("obs_1", ctx), "obs_1")

    def test_an_adopted_id_does_not_collide_with_a_later_automatic_one(self):
        log = ObservationLog()
        log.record(None, pose=[0.0, 0.0, 0.0], yaw=0.0, obs_id="obs_7")

        self.assertEqual(log.record(None, pose=[0.0, 0.0, 0.0], yaw=0.0), "obs_8")
        self.assertIsNotNone(log.get("obs_7"))
        self.assertIsNotNone(log.get("obs_8"))


if __name__ == "__main__":
    unittest.main()
