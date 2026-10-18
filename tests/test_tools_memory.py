"""Tests for the agent memory infrastructure (vlm_nav/memory.py, goal.md sec 4).

Covers the observation archive (record / query / frame materialization) and the
revisable hypothesis store (support-refute revisions, retrieval budgets,
EvidenceRef citations). CPU only, no world or VLM required.
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from vlm_nav.memory import (
    EVIDENCE_KIND_GOAL_FRAME,
    EVIDENCE_KIND_NEGATIVE,
    EVIDENCE_KIND_OBSERVATION,
    EVIDENCE_KIND_SIGHTING,
    EVENT_NOT_FOUND,
    EVENT_SIGHTING,
    HypothesisStore,
    ObservationLog,
    SpatialMemory,
    WorkingMemory,
    build_memory,
    jsonable,
    normalize_evidence,
    scope_note,
    view_scope,
)
from vlm_nav.tools import EvidenceRef


def _frame(value: float = 0.5, size: int = 8) -> torch.Tensor:
    return torch.full((3, size, size), float(value))


class _LogCase(unittest.TestCase):
    """Shared fixture: temp image dir + deterministic clock."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.image_dir = Path(self._tmp.name) / "frames"
        self._now = [1000.0]

    def _clock(self) -> float:
        self._now[0] += 1.0
        return self._now[0]

    def _log(self, **kwargs) -> ObservationLog:
        kwargs.setdefault("image_dir", self.image_dir)
        kwargs.setdefault("clock", self._clock)
        return ObservationLog(**kwargs)


class ObservationLogTest(_LogCase):
    def test_record_get_recent(self):
        log = self._log()
        log.record(_frame(0.2), (0.0, 0.0, 0.0), 0.0, event="start", step=0)
        log.record(_frame(0.4), (1.0, 2.0, 0.0), math.radians(90), event=["turn", "scan"], step=1)
        log.record(
            _frame(0.6), (2.0, 2.0, 0.5), math.radians(-45), event="sighting",
            target_name="red box", step=2,
        )
        self.assertEqual(log.ids(), ["obs_1", "obs_2", "obs_3"])
        self.assertEqual(len(log), 3)

        entry = log.get("obs_3")
        self.assertEqual(entry["obs_id"], "obs_3")
        self.assertEqual(entry["position_flu"], [2.0, 2.0, 0.5])
        self.assertAlmostEqual(entry["yaw_deg"], -45.0)
        self.assertEqual(entry["events"], ["sighting"])
        self.assertEqual(entry["target_name"], "red box")
        self.assertEqual(entry["step"], 2)
        self.assertEqual(entry["t"], 1003.0)

        self.assertEqual([r["obs_id"] for r in log.recent(2)], ["obs_3", "obs_2"])
        self.assertIsNone(log.get("obs_99"))

    def test_record_reads_yaw_from_pose_object(self):
        log = self._log()
        pose = SimpleNamespace(position_flu=(3.0, 1.0, 0.25), yaw_rad=math.radians(30))
        obs_id = log.record(_frame(), pose)
        entry = log.get(obs_id)
        self.assertEqual(entry["position_flu"], [3.0, 1.0, 0.25])
        self.assertAlmostEqual(entry["yaw_deg"], 30.0)

    def test_runtime_wiring_episode_world_out_dir_aliases(self):
        # the runner builds ObservationLog(world=..., out_dir=..., episode_id=...)
        world = SimpleNamespace(
            position_flu=np.asarray([2.0, 1.0, 0.0]), yaw_rad=math.radians(30)
        )
        log = ObservationLog(
            world=world, out_dir=Path(self._tmp.name) / "ep", episode_id="ep_7",
            clock=self._clock,
        )
        self.assertEqual(log.episode_id, "ep_7")
        obs_id = log.record(_frame())  # pose and yaw come from the world
        entry = log.get(obs_id)
        self.assertEqual(entry["position_flu"], [2.0, 1.0, 0.0])
        self.assertAlmostEqual(entry["yaw_deg"], 30.0)
        self.assertEqual(entry["episode_id"], "ep_7")
        self.assertEqual(log.image_path(obs_id), str(log.image_dir / "obs_1.png"))
        self.assertEqual(log.image_dir, Path(self._tmp.name) / "ep" / "memory_frames")

        # adapter keyword spellings (yaw_rad / timestamp) are honoured
        log.record(_frame(), (0.0, 0.0, 0.0), yaw_rad=math.radians(-90), timestamp=42.5,
                   event="scan")
        entry = log.get("obs_2")
        self.assertAlmostEqual(entry["yaw_deg"], -90.0)
        self.assertEqual(entry["t"], 42.5)

    def test_record_without_frame_is_event_only(self):
        log = self._log()
        obs_id = log.record(None, (0.0, 0.0, 0.0), 0.0, event="leg_timeout", step=9)
        entry = log.get(obs_id)
        self.assertIsNone(entry["image"])
        self.assertEqual([r.kind for r in entry["evidence"]], ["observation"])
        self.assertIsNone(log.image_path(obs_id))
        self.assertEqual([r["obs_id"] for r in log.query({"event": "leg_timeout"})], [obs_id])

    def test_record_accepts_numpy_and_pil_frames(self):
        log = self._log()
        arr = np.zeros((6, 9, 3), dtype=np.uint8)
        arr[:, :, 0] = 255
        path = log.image_path(log.record(arr, (0, 0, 0), 0.0))
        with Image.open(path) as image:
            self.assertEqual(image.size, (9, 6))
            self.assertEqual(image.convert("RGB").getpixel((0, 0)), (255, 0, 0))

        path = log.image_path(log.record(Image.new("RGB", (4, 4), (0, 128, 0)), (0, 0, 0), 0.0))
        with Image.open(path) as image:
            self.assertEqual(image.size, (4, 4))

        # world.render()-style CHW float frame
        path = log.image_path(log.record(torch.zeros(3, 5, 7), (0, 0, 0), 0.0))
        with Image.open(path) as image:
            self.assertEqual(image.size, (7, 5))

    def test_record_duplicate_obs_id_rejected(self):
        log = self._log()
        log.record(_frame(), (0, 0, 0), 0.0, obs_id="obs_5")
        with self.assertRaises(ValueError):
            log.record(_frame(), (0, 0, 0), 0.0, obs_id="obs_5")

    def test_image_path_writes_readable_png(self):
        log = self._log()
        obs_id = log.record(_frame(0.4), (0, 0, 0), 0.0)
        path = log.image_path(obs_id)
        self.assertTrue(Path(path).is_file())
        self.assertTrue(path.endswith("obs_1.png"))
        self.assertEqual(log.image_path(obs_id), path)  # cached, not rewritten
        self.assertEqual(log.get(obs_id)["image"], path)
        with self.assertRaises(KeyError):
            log.image_path("obs_404")

    def test_frame_returns_chw_tensor_by_id(self):
        log = self._log()
        arr = np.zeros((6, 9, 3), dtype=np.uint8)
        arr[:, :, 0] = 255
        obs_id = log.record(arr, (0.0, 0.0, 0.0), 0.0)
        frame = log.frame(obs_id)
        self.assertEqual(tuple(frame.shape), (3, 6, 9))
        self.assertEqual(frame.dtype, torch.float32)
        self.assertAlmostEqual(float(frame[0].max()), 1.0)
        self.assertAlmostEqual(float(frame[1].max()), 0.0)
        # once the pixels move to disk the frame still loads identically
        self.assertTrue(Path(log.image_path(obs_id)).is_file())
        self.assertIsNone(log._by_id[obs_id].image)
        self.assertTrue(torch.equal(frame, log.frame(obs_id)))
        self.assertIsNone(log.frame(log.record(None, (0.0, 0.0, 0.0), 0.0)))
        with self.assertRaises(KeyError):
            log.frame("obs_404")

    def test_default_image_dir_is_usable(self):
        log = ObservationLog(clock=self._clock)
        obs_id = log.record(_frame(), (0, 0, 0), 0.0)
        self.assertTrue(Path(log.image_path(obs_id)).is_file())

    def test_query_by_text_and_event_keywords(self):
        log = self._log()
        log.record(_frame(), (0, 0, 0), 0.0, event="target_sighting", target_name="red box")
        log.record(_frame(), (1, 0, 0), 0.0, event="scan")
        self.assertEqual([r["obs_id"] for r in log.query("red box")], ["obs_1"])
        self.assertEqual([r["obs_id"] for r in log.query({"text": "SCAN"})], ["obs_2"])
        self.assertEqual(log.query("nothing matches here"), [])
        self.assertEqual([r["obs_id"] for r in log.query({"event": "sighting"})], ["obs_1"])
        self.assertAlmostEqual(log.query("red box")[0]["score"], 1.0)

    def test_query_target_label_and_event_filters(self):
        log = self._log()
        log.record(_frame(), (0, 0, 0), 0.0, event="target_sighting", target_name="red box",
                   labels=["round"])
        log.record(_frame(), (1, 0, 0), 0.0, event="target_sighting", target_name="blue crate")
        log.record(_frame(), (2, 0, 0), 0.0, event="blocked")
        self.assertEqual([r["obs_id"] for r in log.query({"target": "red box"})], ["obs_1"])
        self.assertEqual(
            sorted(r["obs_id"] for r in log.query({"event": "target_sight"})),
            ["obs_1", "obs_2"],
        )
        self.assertEqual([r["obs_id"] for r in log.query({"event": "blocked"})], ["obs_3"])
        self.assertEqual([r["obs_id"] for r in log.query({"text": "round"})], ["obs_1"])

    def test_query_point_in_frustum_reports_bearing(self):
        log = self._log()
        log.record(_frame(), (0.0, 0.0, 0.0), 0.0, event="ahead")
        log.record(_frame(), (0.0, 0.0, 0.0), math.radians(90), event="left")
        hits = log.query({"point": [3.0, 0.0, 0.0], "fov_deg": 90.0, "max_distance_m": 8.0})
        self.assertEqual([r["obs_id"] for r in hits], ["obs_1"])
        self.assertAlmostEqual(hits[0]["bearing_deg"], 0.0)
        self.assertAlmostEqual(hits[0]["distance_m"], 3.0)
        # outside the frustum / out of range -> no matches
        self.assertEqual(log.query({"point": [3.0, 0.0, 0.0], "max_distance_m": 1.0}), [])
        # 2-component points default to z=0
        self.assertEqual(len(log.query({"point": [3.0, 0.0]})), 1)
        self.assertEqual(log.query({"point": [99.0, 0.0]}), [])

    def test_query_bearing_sector_from_origin(self):
        log = self._log()
        log.record(_frame(), (5.0, 0.0, 0.0), 0.0, event="east")
        log.record(_frame(), (-5.0, 0.0, 0.0), 0.0, event="west")
        hits = log.query({"origin": [0.0, 0.0, 0.0], "bearing_deg": 0.0, "tolerance_deg": 45.0})
        self.assertEqual([r["obs_id"] for r in hits], ["obs_1"])
        self.assertAlmostEqual(hits[0]["obs_bearing_deg"], 0.0)
        hits = log.query({"origin": [0.0, 0.0, 0.0], "bearing_deg": 180.0, "tolerance_deg": 45.0})
        self.assertEqual([r["obs_id"] for r in hits], ["obs_2"])
        self.assertEqual(
            log.query({"origin": [0.0, 0.0, 0.0], "bearing_deg": 90.0, "tolerance_deg": 45.0}),
            [],
        )

    def test_query_time_and_step_windows(self):
        log = self._log()
        # distinct frames on purpose: pixel-identical frames at one pose and
        # heading now merge into a single entry (EpisodicEvidenceTest), and this
        # test is about the time/step windows, not about duplicate frames
        for step in range(4):
            log.record(_frame(0.1 * (step + 1)), (0.0, 0.0, 0.0), 0.0, step=step)
        self.assertEqual([r["obs_id"] for r in log.query({"step": 1})], ["obs_2"])
        self.assertEqual(
            sorted(r["obs_id"] for r in log.query({"since_step": 2})), ["obs_3", "obs_4"]
        )
        self.assertEqual(
            sorted(r["obs_id"] for r in log.query({"since": 1003.0})), ["obs_3", "obs_4"]
        )
        self.assertEqual(
            sorted(r["obs_id"] for r in log.query({"before": 1002.0})), ["obs_1", "obs_2"]
        )

    def test_query_limit_budget(self):
        log = self._log()
        for index in range(6):  # distinct frames: duplicates would merge
            log.record(_frame(0.1 * (index + 1)), (0.0, 0.0, 0.0), 0.0, event="scan")
        self.assertEqual(len(log.query({"event": "scan"}, limit=2)), 2)
        self.assertEqual(len(log.query({"event": "scan", "limit": 3})), 3)
        # the tighter budget wins
        self.assertEqual(len(log.query({"event": "scan", "limit": 1}, limit=10)), 1)
        self.assertEqual(log.stats()["queries"], 3)
        self.assertEqual(log.stats()["returned"], 6)

        capped = self._log(max_limit=3)
        for index in range(5):
            capped.record(_frame(0.1 * (index + 1)), (0.0, 0.0, 0.0), 0.0, event="scan")
        self.assertEqual(len(capped.query({"event": "scan"}, limit=100)), 3)

    def test_query_without_criteria_returns_recent(self):
        log = self._log()
        for step in range(3):  # distinct frames: identical ones merge
            log.record(_frame(0.1 * (step + 1)), (0.0, 0.0, 0.0), 0.0, step=step)
        self.assertEqual([r["obs_id"] for r in log.query({})], ["obs_3", "obs_2", "obs_1"])
        self.assertEqual([r["obs_id"] for r in log.query("")], ["obs_3", "obs_2", "obs_1"])

    def test_entries_cite_evidence_refs(self):
        log = self._log()
        obs_id = log.record(_frame(), (0.0, 0.0, 0.0), 0.0, event="target_sighting", step=4)
        entry = log.get(obs_id)
        refs = entry["evidence"]
        self.assertEqual([r.kind for r in refs], ["observation", "image"])
        self.assertTrue(all(isinstance(r, EvidenceRef) for r in refs))
        self.assertEqual(refs[0].ref, obs_id)
        self.assertIn("sighting", refs[0].note)
        self.assertEqual(refs[1].ref, entry["image"])
        self.assertTrue(Path(entry["image"]).is_file())  # resource resolves to a real frame
        json.dumps(jsonable(entry))  # MCP-embeddable

    def test_max_entries_evicts_pixels_not_metadata(self):
        log = self._log(max_entries=1)
        first = log.record(_frame(), (0.0, 0.0, 0.0), 0.0, event="first")
        # a different heading (the two frames are otherwise identical): the two
        # records must stay two entries for the eviction rule to be observable
        log.record(_frame(), (0.0, 0.0, 0.0), math.radians(20.0), event="second")
        self.assertIsNone(log.image_path(first))
        self.assertEqual(log.get(first)["events"], ["first"])
        self.assertEqual(log.stats()["stored_images"], 1)

    def test_describe_is_compact(self):
        log = self._log()
        self.assertEqual(log.describe(), "no observations")
        log.record(_frame(), (0.0, 0.0, 0.0), math.radians(45), event="sighting", step=7,
                   target_name="red box")
        text = log.describe(limit=1)
        self.assertIn("obs_1", text)
        self.assertIn("step=7", text)
        self.assertIn("sighting", text)
        self.assertIn("red box", text)
        self.assertEqual(len(text.splitlines()), 1)


class HypothesisStoreTest(unittest.TestCase):
    def test_add_and_get(self):
        store = HypothesisStore()
        entry = store.add("red box is the target", evidence="obs_3")
        self.assertEqual(entry["id"], "hyp_1")
        self.assertEqual(entry["status"], "active")
        self.assertEqual(entry["support_count"], 1)
        self.assertEqual(entry["added"], 1)
        self.assertEqual(store.get("hyp_1")["content"], "red box is the target")
        self.assertEqual(len(store), 1)
        self.assertIsNone(store.get("hyp_9"))

    def test_add_structured_content(self):
        store = HypothesisStore()
        entry = store.add(
            {"kind": "identity", "subject": "red box", "claim": "is the crate by the door"}
        )
        self.assertEqual(entry["kind"], "identity")
        self.assertEqual(entry["content"], "is the crate by the door")
        self.assertEqual(entry["data"]["subject"], "red box")
        self.assertEqual(store.add("may be north of the hallway", kind="location")["kind"],
                         "location")

    def test_add_content_is_truncated(self):
        store = HypothesisStore(max_content_chars=32)
        entry = store.add("x" * 200)
        self.assertEqual(len(entry["content"]), 32)

    def test_support_confirms_and_dedupes(self):
        store = HypothesisStore(confirm_support=2)
        store.add("red box is the target")
        result = store.support("hyp_1", "obs_3")
        self.assertEqual(result["status"], "active")
        self.assertEqual(result["added"], 1)
        result = store.support("hyp_1", ["obs_7", "obs_3", "obs_7"])
        self.assertEqual(result["status"], "confirmed")
        self.assertEqual(result["support_count"], 2)
        self.assertEqual(result["added"], 1)  # only obs_7 was new
        result = store.support("hyp_1", "obs_7")
        self.assertEqual(result["added"], 0)  # duplicate evidence is a no-op
        self.assertEqual(result["support_count"], 2)

    def test_refute_rejects_with_reason(self):
        store = HypothesisStore()
        store.add("blue crate is the target", evidence=["obs_2", "obs_3"])
        self.assertEqual(store.get("hyp_1")["status"], "confirmed")
        result = store.refute("hyp_1", "obs_9", reason="wrong shape")
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["refute_count"], 1)
        self.assertEqual(result["reasons"], ["wrong shape"])

    def test_support_reopens_rejected_hypothesis(self):
        store = HypothesisStore(confirm_support=3)
        store.add("red box is the target", evidence="obs_3")
        store.refute("hyp_1", "obs_9", reason="not visible from vp_4")
        self.assertEqual(store.get("hyp_1")["status"], "rejected")
        # repeating known evidence changes nothing
        result = store.support("hyp_1", "obs_3")
        self.assertEqual(result["added"], 0)
        self.assertEqual(result["status"], "rejected")
        result = store.support("hyp_1", "obs_11")
        self.assertEqual(result["status"], "active")
        self.assertEqual(result["refute_count"], 1)  # refutation history is retained
        result = store.support("hyp_1", "obs_12")
        self.assertEqual(result["status"], "confirmed")

    def test_query_filters_and_limit(self):
        store = HypothesisStore()
        store.add("red box is the target", kind="identity")
        store.add("blue crate is the target", kind="identity")
        store.add("target may be behind the door", kind="location")
        store.refute("hyp_2", "obs_9", reason="wrong shape")
        self.assertEqual([h["id"] for h in store.query({"status": "rejected"})], ["hyp_2"])
        self.assertEqual([h["id"] for h in store.query({"kind": "location"})], ["hyp_3"])
        # live beliefs outrank the rejected one on equal relevance
        self.assertEqual([h["id"] for h in store.query("target")], ["hyp_3", "hyp_1", "hyp_2"])
        self.assertEqual(store.query("nonsense token"), [])
        self.assertEqual(len(store.query("target", limit=2)), 2)
        self.assertEqual(len(HypothesisStore(max_limit=2).query({})), 0)
        self.assertEqual(store.stats()["queries"], 5)

    def test_describe_is_compact(self):
        store = HypothesisStore()
        self.assertEqual(store.describe(), "no hypotheses")
        store.add("red box is the target", evidence=["obs_3", "obs_7"])
        text = store.describe()
        self.assertIn("hyp_1", text)
        self.assertIn("[confirmed]", text)
        self.assertIn("support=2", text)
        self.assertIn("obs_3", text)
        store.refute("hyp_1", "obs_11", reason="wrong colour")
        text = store.describe()
        self.assertIn("[rejected]", text)
        self.assertIn("refuted(wrong colour)", text)
        self.assertEqual(store.describe(status="active"), "no hypotheses")
        self.assertTrue(store.describe(status="rejected").startswith("hyp_1"))

    def test_entries_cite_evidence_refs(self):
        store = HypothesisStore(episode_id="ep_7")
        entry = store.add("red box is the target", evidence="obs_3")
        self.assertEqual(entry["episode_id"], "ep_7")
        self.assertEqual(entry["support"][0]["kind"], "observation")
        self.assertEqual(entry["support"][0]["ref"], "obs_3")
        self.assertEqual(entry["evidence"][0].kind, "memory")
        self.assertEqual(entry["evidence"][0].ref, "hyp_1")
        self.assertEqual(entry["evidence"][1].ref, "obs_3")
        self.assertTrue(all(isinstance(r, EvidenceRef) for r in entry["evidence"]))
        json.dumps(jsonable(entry))

    def test_unknown_id_raises(self):
        store = HypothesisStore()
        with self.assertRaises(KeyError):
            store.support("hyp_9", "obs_1")
        with self.assertRaises(KeyError):
            store.refute("hyp_9", "obs_1")


class EvidenceHelperTest(unittest.TestCase):
    def test_normalize_evidence_variants(self):
        refs = normalize_evidence("obs_3")
        self.assertEqual((refs[0].kind, refs[0].ref), ("observation", "obs_3"))
        refs = normalize_evidence({"ref": "frames/step_012.png", "note": "frame"})
        self.assertEqual((refs[0].kind, refs[0].note), ("image", "frame"))
        refs = normalize_evidence(EvidenceRef("map", "bev_4"))
        self.assertEqual(refs, [EvidenceRef("map", "bev_4")])
        refs = normalize_evidence(["obs_1", "obs_1", "hyp_2", "grab the crate"])
        self.assertEqual([r.kind for r in refs], ["observation", "memory", "memory"])
        self.assertEqual(len(refs), 3)
        self.assertEqual(normalize_evidence(None), [])
        self.assertEqual(normalize_evidence([]), [])

    def test_jsonable_converts_nested_entries(self):
        payload = jsonable(
            {
                "evidence": [EvidenceRef("observation", "obs_1")],
                "pose": (1.0, 2.0, 3.0),
                "arr": np.zeros(2, dtype=np.float32),
                "tensor": torch.ones(2),
            }
        )
        self.assertEqual(payload["evidence"][0]["ref"], "obs_1")
        self.assertEqual(payload["pose"], [1.0, 2.0, 3.0])
        self.assertEqual(payload["arr"], [0.0, 0.0])
        self.assertEqual(payload["tensor"], [1.0, 1.0])
        json.dumps(payload)


class WorkingMemoryTest(unittest.TestCase):
    """Layer 1: what the task is doing right now, at fixed capacity."""

    def _memory(self, **kwargs) -> WorkingMemory:
        kwargs.setdefault("max_questions", 2)
        kwargs.setdefault("max_failures", 2)
        kwargs.setdefault("max_candidates", 2)
        return WorkingMemory(**kwargs)

    def test_subgoal_is_single_and_replaced(self):
        memory = self._memory()
        self.assertIsNone(memory.snapshot()["subgoal"])
        entry = memory.set_subgoal("find the red box", step=4)
        self.assertEqual(entry["text"], "find the red box")
        self.assertEqual(entry["step"], 4)
        memory.set_subgoal("verify the red box", step=9)
        snapshot = memory.snapshot()
        self.assertEqual(snapshot["subgoal"]["text"], "verify the red box")
        self.assertEqual(snapshot["subgoal"]["step"], 9)

    def test_candidates_merge_by_id_and_evict_the_oldest(self):
        memory = self._memory()
        first = memory.note_candidate(
            "cand_1", [1.0, 2.0, 0.5], target="red box", confidence=0.4, source="ground_target",
        )
        self.assertEqual(first["id"], "cand_1")
        self.assertEqual(first["position_flu"], [1.0, 2.0, 0.5])
        self.assertEqual(first["updates"], 1)
        again = memory.note_candidate("cand_1", [1.1, 2.0, 0.5], confidence=0.8, source="observe")
        self.assertEqual(again["updates"], 2)
        self.assertEqual(again["confidence"], 0.8)
        self.assertEqual(again["sources"], ["ground_target", "observe"])
        self.assertEqual(len(memory.snapshot()["candidates"]), 1)

        memory.note_candidate("cand_2", [3.0, 0.0, 0.5])
        memory.note_candidate("cand_3", [4.0, 0.0, 0.5])
        ids = [entry["id"] for entry in memory.snapshot()["candidates"]]
        self.assertEqual(ids, ["cand_3", "cand_2"])  # capacity 2, newest first

    def test_open_questions_dedupe_and_resolve(self):
        memory = self._memory()
        first = memory.open_question("where is the red box?", step=3)
        self.assertEqual(first["id"], "q_1")
        self.assertEqual(first["status"], "open")
        again = memory.open_question("Where is  the red box ?", step=5)
        self.assertEqual(again["id"], "q_1")  # normalized text, no duplicate
        self.assertEqual(again["seen"], 2)
        resolved = memory.resolve_question("q_1", answer="at place_1", evidence="obs_7")
        self.assertEqual(resolved["status"], "resolved")
        self.assertEqual(resolved["answer"], "at place_1")
        self.assertIsNone(memory.resolve_question("q_404"))
        snapshot = memory.snapshot()
        self.assertEqual(snapshot["open_questions"], [])
        self.assertEqual(len(snapshot["questions"]), 1)
        self.assertEqual(snapshot["questions"][0]["status"], "resolved")

    def test_only_open_questions_fill_the_capacity(self):
        memory = self._memory(max_questions=2)
        memory.open_question("q one")
        memory.resolve_question("q_1")
        memory.open_question("q two")
        memory.open_question("q three")
        open_ids = [entry["id"] for entry in memory.snapshot()["open_questions"]]
        self.assertEqual(open_ids, ["q_3", "q_2"])
        self.assertEqual(memory.stats()["open_questions"], 2)
        self.assertEqual(memory.stats()["resolved_questions"], 1)

    def test_failures_merge_repeats_and_stay_bounded(self):
        memory = self._memory(max_failures=2)
        first = memory.note_failure(
            "ground 'red box'", reason="no_candidate", strategy="inspect_region", step=5
        )
        self.assertEqual(first["count"], 1)
        again = memory.note_failure("ground 'red box'", reason="no_candidate", step=7)
        self.assertEqual(again["count"], 2)
        self.assertEqual(len(memory.snapshot()["failures"]), 1)
        memory.note_failure("ground 'red box'", reason="blocked_stagnant", step=8)
        memory.note_failure("reach vp_2", reason="timeout", step=9)
        reasons = [entry["reason"] for entry in memory.snapshot()["failures"]]
        self.assertEqual(reasons, ["timeout", "blocked_stagnant"])  # oldest evicted

    def test_snapshot_describe_and_stats_are_json_plain(self):
        memory = self._memory()
        memory.set_subgoal("find the red box", step=1)
        memory.note_candidate("cand_1", [1.0, 2.0, 0.5], target="red box")
        memory.open_question("where is the red box?", step=2)
        memory.note_failure("ground 'red box'", reason="no_candidate", step=3)
        snapshot = memory.snapshot()
        self.assertEqual(
            set(snapshot), {"subgoal", "candidates", "open_questions", "questions", "failures",
                            "capacity", "counts", "updated_t"}
        )
        json.dumps(jsonable(snapshot))
        self.assertEqual(memory.stats()["candidates"], 1)
        text = memory.describe()
        self.assertIn("find the red box", text)
        self.assertIn("where is the red box?", text)
        self.assertIn("no_candidate", text)
        self.assertEqual(len(text.splitlines()), 4)  # subgoal + question + candidate + failure

    def test_clear_empties_every_slot(self):
        memory = self._memory()
        memory.set_subgoal("find it")
        memory.open_question("where?")
        memory.note_failure("look", reason="empty")
        memory.clear()
        snapshot = memory.snapshot()
        self.assertIsNone(snapshot["subgoal"])
        self.assertEqual(snapshot["open_questions"], [])
        self.assertEqual(snapshot["failures"], [])
        self.assertEqual(memory.describe(), "no working memory")


class ViewScopeTest(unittest.TestCase):
    """Negative memory is bound to a viewpoint, a heading and a field of view."""

    def test_scope_records_the_view_condition(self):
        scope = view_scope(
            position_flu=(2.0, 0.0, 0.5), yaw_deg=30.0, hfov_deg=90.0, region=[0, 0, 500, 500],
            observation_id="obs_4", note="looked for the red box",
        )
        self.assertEqual(scope["position_flu"], [2.0, 0.0, 0.5])
        self.assertEqual(scope["yaw_deg"], 30.0)
        self.assertEqual(scope["hfov_deg"], 90.0)
        self.assertEqual(scope["region"], [0.0, 0.0, 500.0, 500.0])
        self.assertEqual(scope["observation_id"], "obs_4")
        # the 45 deg bin the heading falls into, as the coverage report bins it
        self.assertEqual(scope["coverage_deg"], [0.0, 45.0])
        json.dumps(scope)

    def test_scope_note_names_the_boundary_and_does_not_clear_the_area(self):
        note = scope_note(view_scope(position_flu=(2.0, 0.0, 0.5), yaw_deg=30.0, hfov_deg=90.0))
        self.assertIn("yaw_deg=30", note)
        self.assertIn("90", note)
        self.assertIn("2.0, 0.0, 0.5", note)
        self.assertIn("does not clear the area", note)
        unknown = scope_note(view_scope(position_flu=(1.0, 1.0, 0.0)))
        self.assertIn("field of view unknown", unknown)

    def test_scope_is_json_plain_even_for_numpy_poses(self):
        scope = view_scope(position_flu=np.asarray([1, 2, 3], dtype=np.float64), yaw_rad=0.5)
        self.assertEqual(scope["position_flu"], [1.0, 2.0, 3.0])
        self.assertEqual(scope["yaw_deg"], 28.6)


class SpatialMemoryTest(unittest.TestCase):
    """Layer 2: places (landmarks, candidate positions, links) and coverage."""

    def test_places_merge_within_the_radius_keeping_the_provenance(self):
        memory = SpatialMemory(merge_radius_m=1.0)
        first = memory.remember_place(
            [1.0, 2.0, 0.5], name="red box", kind="landmark", uncertainty_m=0.8,
            source="observe:obs_3", observation_id="obs_3", step=2,
        )
        again = memory.remember_place(
            [1.2, 2.1, 0.5], name="red box", uncertainty_m=0.3,
            source="ground_target:cand_2", candidate_id="cand_2", observation_id="obs_7", step=5,
        )
        self.assertEqual(again["place_id"], first["place_id"])
        self.assertEqual(again["updates"], 2)
        self.assertEqual(again["sources"], ["observe:obs_3", "ground_target:cand_2"])
        self.assertEqual(again["observations"], ["obs_3", "obs_7"])
        self.assertEqual(again["candidates"], ["cand_2"])
        # the most certain estimate is kept, the spread names the disagreement
        self.assertEqual(again["uncertainty_m"], 0.3)
        self.assertGreater(again["spread_m"], 0.2)
        self.assertEqual(len(memory.places()), 1)
        memory.remember_place([9.0, 9.0, 0.5], name="door")
        self.assertEqual(len(memory.places()), 2)
        self.assertEqual([place["name"] for place in memory.places()], ["door", "red box"])

    def test_a_name_conflict_on_one_place_is_recorded(self):
        memory = SpatialMemory(merge_radius_m=1.0)
        memory.remember_place([1.0, 2.0, 0.5], name="red box", observation_id="obs_3")
        merged = memory.remember_place([1.1, 2.0, 0.5], name="blue crate", observation_id="obs_7")
        self.assertEqual(merged["name"], "red box")  # the discovery names the place
        self.assertEqual(merged["names"], ["red box", "blue crate"])
        self.assertTrue(any("blue crate" in item for item in merged["conflicts"]))

    def test_candidate_places_merge_by_candidate_id(self):
        memory = SpatialMemory()
        first = memory.remember_candidate("cand_5", [3.0, 0.0, 0.5], target="chair", confidence=0.5)
        self.assertEqual(first["kind"], "candidate")
        self.assertEqual(first["candidates"], ["cand_5"])
        again = memory.remember_candidate("cand_5", [3.1, 0.0, 0.5], confidence=0.9)
        self.assertEqual(again["place_id"], first["place_id"])
        self.assertEqual(again["updates"], 2)
        self.assertEqual(again["confidence"], 0.9)
        self.assertEqual(memory.stats()["candidate_places"], 1)
        # one candidate id is one place: a wild jump is a disagreement to keep,
        # not a second place for the same candidate
        distant = memory.remember_candidate("cand_5", [30.0, 0.0, 0.5])
        self.assertEqual(distant["place_id"], first["place_id"])
        self.assertEqual(len(distant["candidates"]), 1)
        self.assertAlmostEqual(distant["spread_m"], 27.0, places=1)  # the jump is the uncertainty
        self.assertEqual(memory.stats()["candidate_places"], 1)
        # another candidate at that distance is a place of its own
        other = memory.remember_candidate("cand_9", [31.0, 0.0, 0.5])
        self.assertNotEqual(other["place_id"], first["place_id"])
        self.assertEqual(memory.stats()["candidate_places"], 2)

    def test_links_record_the_region_adjacency(self):
        memory = SpatialMemory()
        place_a = memory.remember_place([0.0, 0.0, 0.5], name="start")
        place_b = memory.remember_place([5.0, 0.0, 0.5], name="door")
        link = memory.link(place_a["place_id"], place_b["place_id"], note="open doorway", length_m=5.0)
        self.assertEqual(link["a"], place_a["place_id"])
        self.assertEqual(link["b"], place_b["place_id"])
        self.assertTrue(link["traversable"])
        self.assertEqual(link["updates"], 1)
        again = memory.link(place_b["place_id"], place_a["place_id"], blocked=True)
        self.assertEqual(again["updates"], 2)  # one connection, whichever way it is named
        self.assertFalse(again["traversable"])
        self.assertEqual(len(memory.links()), 1)

    def test_coverage_accumulates_checked_sectors_with_their_viewpoints(self):
        memory = SpatialMemory()
        memory.note_coverage(
            [[0.0, 45.0], [45.0, 90.0]], position=[0.0, 0.0, 0.2], observation_id="obs_1", step=1
        )
        memory.note_coverage(
            [[45.0, 90.0], [90.0, 135.0]], position=[1.0, 0.0, 0.2], observation_id="obs_9", step=6,
            no_result_for=["red box"],
        )
        coverage = memory.coverage()
        self.assertEqual(coverage["checked_deg"], [[0.0, 45.0], [45.0, 90.0], [90.0, 135.0]])
        self.assertEqual(len(coverage["unchecked_deg"]), 5)
        middle = next(
            sector for sector in coverage["sectors"] if sector["sector"] == [45.0, 90.0]
        )
        self.assertEqual(middle["views"], 2)
        self.assertEqual(middle["observations"], ["obs_1", "obs_9"])
        self.assertEqual(middle["positions"], [[0.0, 0.0, 0.2], [1.0, 0.0, 0.2]])
        self.assertEqual(middle["no_result_for"], ["red box"])
        near = memory.coverage(near=[1.0, 0.0, 0.2], radius_m=0.5)
        self.assertEqual(near["checked_deg"], [[45.0, 90.0], [90.0, 135.0]])
        self.assertEqual(memory.stats()["covered_sectors"], 3)

    def test_query_matches_name_candidate_and_point(self):
        memory = SpatialMemory()
        place = memory.remember_place(
            [1.0, 2.0, 0.5], name="red box", observation_id="obs_3", source="observe:obs_3"
        )
        memory.remember_candidate("cand_5", [3.0, 0.0, 0.5], target="chair")
        hits = memory.query("red box")
        self.assertEqual([hit["place_id"] for hit in hits], [place["place_id"]])
        self.assertEqual(hits[0]["evidence"][0].kind, "memory")
        self.assertEqual(memory.query("nonsense"), [])
        self.assertEqual(
            [hit["place_id"] for hit in memory.query({"candidate": "cand_5"})], ["place_2"]
        )
        near = memory.query({"point": [1.1, 2.0, 0.5], "max_distance_m": 0.5})
        self.assertEqual([hit["place_id"] for hit in near], [place["place_id"]])
        self.assertAlmostEqual(near[0]["distance_m"], 0.1, places=3)
        self.assertEqual(memory.query({"point": [20.0, 2.0, 0.5]}), [])

    def test_capacity_and_stats_are_reported(self):
        memory = SpatialMemory(max_places=2)
        for index in range(3):
            memory.remember_place([float(index) * 10.0, 0.0, 0.0], name=f"place {index}")
        self.assertEqual(len(memory.places()), 2)
        stats = memory.stats()
        self.assertEqual(stats["places"], 2)
        self.assertEqual(stats["evicted_places"], 1)
        self.assertEqual(memory.describe(), memory.describe(limit=2))
        json.dumps(jsonable(memory.stats()))


class EpisodicEvidenceTest(_LogCase):
    """Layer 3: key frames saved by event, duplicates merged, pixels bounded."""

    def test_identical_frames_merge_into_one_entry(self):
        log = self._log()
        first = log.record(_frame(), (1.0, 2.0, 0.0), math.radians(30), event="scan", step=3)
        again = log.record(_frame(), (1.0, 2.0, 0.0), math.radians(30), event="scan", step=4)
        self.assertEqual(first, "obs_1")
        self.assertEqual(again, first)  # the duplicate folds into the discovery
        self.assertEqual(len(log), 1)
        entry = log.get(first)
        self.assertEqual(entry["repeat_count"], 1)
        self.assertEqual(entry["merged_ids"], ["obs_2"])
        self.assertEqual(entry["step"], 3)
        self.assertEqual(entry["last_t"], 1002.0)
        self.assertEqual(log.stats()["merged_duplicates"], 1)
        # the id the second capture was handed still resolves (the model saw it)
        self.assertEqual(log.get("obs_2")["obs_id"], first)
        self.assertEqual(log.layer_status()["duplicates_merged"], 1)

    def test_a_merge_keeps_the_discovery_and_folds_in_the_sighting(self):
        log = self._log()
        first = log.record(_frame(), (1.0, 2.0, 0.0), 0.0, event="scan", step=3)
        log.record(
            _frame(), (1.0, 2.0, 0.0), 0.0, step=4,
            sighting={"target": "red box", "candidate": "cand_1", "basis": "ground_target hit"},
        )
        entry = log.get(first)
        self.assertEqual(entry["sighting"]["candidate"], "cand_1")
        self.assertEqual(entry["sighting"]["image"], first)
        self.assertIn(EVENT_SIGHTING, entry["events"])
        self.assertEqual(entry["evidence_kind"], EVIDENCE_KIND_SIGHTING)
        self.assertEqual(entry["repeat_count"], 1)

    def test_different_content_or_heading_is_not_merged(self):
        log = self._log()
        first = log.record(_frame(0.2), (0.0, 0.0, 0.0), 0.0, step=1)
        log.record(_frame(0.8), (0.0, 0.0, 0.0), 0.0, step=2)  # different pixels
        log.record(_frame(0.2), (0.0, 0.0, 0.0), math.radians(40), step=3)  # different heading
        log.record(_frame(0.2), (4.0, 0.0, 0.0), 0.0, step=4)  # different pose
        self.assertEqual(len(log), 4)
        self.assertEqual(log.get(first)["repeat_count"], 0)

    def test_event_only_records_never_merge(self):
        log = self._log()
        first = log.record(None, (0.0, 0.0, 0.0), 0.0, event="leg_timeout", step=1)
        second = log.record(None, (0.0, 0.0, 0.0), 0.0, event="leg_timeout", step=2)
        self.assertNotEqual(first, second)
        self.assertEqual(len(log), 2)

    def test_pixels_are_capped_per_event_but_the_metadata_stays(self):
        log = self._log(max_frames_per_event=1)
        first = log.record(_frame(), (0.0, 0.0, 0.0), math.radians(10), event=EVENT_SIGHTING)
        second = log.record(_frame(0.4), (1.0, 0.0, 0.0), math.radians(20), event=EVENT_SIGHTING)
        self.assertEqual(log.get(first)["events"], [EVENT_SIGHTING])
        self.assertTrue(log.get(first)["image"])  # materialized, still resolvable
        self.assertTrue(Path(log.get(first)["image"]).is_file())
        self.assertIsNone(log._by_id[first].image)  # the pixels left RAM
        self.assertIsNotNone(log._by_id[second].image)
        self.assertEqual(log.stats()["evicted_for_events"], 1)
        with self.assertRaises(KeyError):
            Path(log.image_path("obs_404"))

    def test_a_more_salient_frame_outlives_a_routine_one(self):
        log = self._log(max_frames_per_event=1)
        routine = log.record(_frame(0.3), (0.0, 0.0, 0.0), 0.0, event="scan", step=1)
        visited = log.record(_frame(0.4), (0.0, 0.0, 0.0), 0.0, event="scan", step=2)
        turning_point = log.record(
            _frame(0.5), (0.0, 0.0, 0.0), 0.0, event=["scan", "decision"], step=3
        )
        # the discovery/turning point keeps its pixels, the routine captures lose them
        self.assertIsNotNone(log._by_id[turning_point].image)
        self.assertIsNone(log._by_id[visited].image)
        self.assertIsNone(log._by_id[routine].image)
        self.assertEqual(log.get(turning_point)["salience"], 1.0)
        self.assertEqual(log.get(routine)["salience"], 0.0)
        self.assertEqual(log.stats()["evicted_for_events"], 2)

    def test_switches_report_the_layers(self):
        log = self._log()
        self.assertEqual(
            log.switches(),
            {"spatial_memory": True, "episodic_memory": True, "merge_duplicates": True,
             "working_memory": True, "max_frames_per_event": log.max_frames_per_event},
        )
        self.assertIsInstance(log.working, WorkingMemory)
        self.assertIsInstance(log.spatial, SpatialMemory)
        self.assertEqual(log.layer_status()["spatial"], "on")

    def test_the_switches_disable_the_layers(self):
        log = self._log(spatial_memory=False, episodic_memory=False)
        self.assertIsNone(log.spatial)
        self.assertEqual(log.layer_status()["spatial"], "off")
        self.assertEqual(log.layer_status()["episodic"], "off")
        first = log.record(_frame(), (1.0, 2.0, 0.0), math.radians(30), event="scan")
        again = log.record(_frame(), (1.0, 2.0, 0.0), math.radians(30), event="scan")
        self.assertNotEqual(first, again)  # no episodic maintenance: no merge
        self.assertEqual(len(log), 2)
        self.assertEqual(log.switches()["episodic_memory"], False)

    def test_build_memory_wires_the_switches(self):
        memory = build_memory(world=None, out_dir=Path(self._tmp.name) / "ep2", episode_id="ep_9")
        self.assertIsInstance(memory, ObservationLog)
        self.assertEqual(memory.episode_id, "ep_9")
        self.assertIsInstance(memory.spatial, SpatialMemory)
        off = build_memory(spatial_memory=False, episodic_memory=False)
        self.assertIsNone(off.spatial)
        self.assertEqual(off.switches()["episodic_memory"], False)


class EpisodicLayerSwitchTest(_LogCase):
    """``episodic_memory=False``: the evidence layer is off, the other two are not.

    The episodic layer is what the store keeps *about a frame*: its pixels, the
    sightings (image + candidate + basis), the scoped null results and the
    per-event maintenance of them. With the layer off none of it is written, so
    none of it can be read back - while the very same calls still feed the
    spatial layer (candidate places, checked sectors) and the working layer.
    """

    SIGHTING = {
        "target": "red box",
        "candidate": "cand_1",
        "basis": "ground_target hit",
        "position_flu": [2.0, 0.0, 0.5],
        "confidence": 0.8,
    }

    def test_no_evidence_is_written_with_the_layer_off(self):
        off = self._log(episodic_memory=False)
        on = self._log()
        for log in (off, on):
            log.record(_frame(), (1.0, 0.0, 0.2), 0.0, goal_target="red box", step=1,
                       sighting=dict(self.SIGHTING))

        ablated = off.get("obs_1")
        self.assertIsNone(ablated["image"])
        self.assertNotIn("image_path", ablated)
        self.assertIsNone(ablated["sighting"])
        self.assertNotIn("sightings", ablated)
        self.assertEqual(ablated["evidence_kind"], EVIDENCE_KIND_OBSERVATION)
        self.assertNotIn(EVENT_SIGHTING, ablated["events"])
        self.assertIsNone(off.frame("obs_1"))
        self.assertIsNone(off.image_path("obs_1"))
        self.assertEqual(off.stats()["sightings"], 0)
        self.assertEqual(off.stats()["stored_images"], 0)
        self.assertEqual(off.layer_status()["sightings"], 0)
        # the same call with the layer on writes the evidence as before
        kept = on.get("obs_1")
        self.assertTrue(kept["image"])
        self.assertEqual(kept["sighting"]["candidate"], "cand_1")
        self.assertEqual(kept["evidence_kind"], EVIDENCE_KIND_SIGHTING)
        self.assertIsNotNone(on.frame("obs_1"))

    def test_the_capture_record_still_carries_what_the_other_layers_read(self):
        log = self._log(episodic_memory=False)
        log.record(_frame(), (1.0, 0.0, 0.2), math.radians(30), goal_target="red box", step=3,
                   sighting=dict(self.SIGHTING), coverage={"checked_deg": [[0.0, 45.0]]})

        entry = log.get("obs_1")
        self.assertEqual(entry["position_flu"], [1.0, 0.0, 0.2])
        self.assertEqual(entry["yaw_deg"], 30.0)
        self.assertEqual(entry["goal_target"], "red box")
        self.assertEqual(entry["step"], 3)
        self.assertIsNone(entry["image"])  # the capture is a record, not a picture
        self.assertIsNone(entry["sighting"])  # nor a claim about what it showed
        # the spatial layer got the candidate place and the checked sector, the
        # working layer the candidate (the switch is about evidence, not places)
        self.assertEqual(log.spatial.places()[0]["candidates"], ["cand_1"])
        self.assertEqual(log.spatial.coverage()["checked_deg"], [[0.0, 45.0]])
        self.assertEqual(log.working.snapshot()["candidates"][0]["id"], "cand_1")
        self.assertEqual(log.switches()["episodic_memory"], False)
        self.assertEqual(log.layer_status()["episodic"], "off")

    def test_a_null_result_is_not_kept_but_its_sector_is(self):
        log = self._log(episodic_memory=False)
        log.record(_frame(), (0.0, 0.0, 0.2), math.radians(10), goal_target="red box", step=1)

        result = log.add_negative("obs_1", target="red box", reason="no_candidate")

        self.assertEqual(result["status"], "disabled:episodic_memory_off")
        self.assertEqual(result["added"], 0)
        entry = log.get("obs_1")
        self.assertNotIn("negative", entry)
        self.assertNotIn("visibility", entry)
        self.assertEqual(log.query({"negative": True}), [])
        self.assertEqual(log.stats()["negatives"], 0)
        # "checked for red box without a result" is a coverage fact (spatial),
        # and the failed look is a task fact (working) - both survive the switch
        self.assertEqual(log.spatial.coverage()["sectors"][0]["no_result_for"], ["red box"])
        self.assertEqual(log.working.snapshot()["failures"][0]["what"], "look for 'red box'")
        # the same call with the layer on keeps the scoped null result
        on = self._log()
        on.record(_frame(), (0.0, 0.0, 0.2), math.radians(10), goal_target="red box", step=1)
        self.assertEqual(on.add_negative("obs_1", target="red box")["status"], "ok:add_negative")
        self.assertEqual(on.query({"negative": True})[0]["obs_id"], "obs_1")

    def test_a_late_sighting_is_reported_and_only_reaches_the_other_layers(self):
        log = self._log(episodic_memory=False)
        log.record(_frame(), (0.0, 0.0, 0.0), 0.0, event="capture", step=1)

        result = log.add_sighting("obs_1", dict(self.SIGHTING), step=2)

        self.assertEqual(result["status"], "disabled:episodic_memory_off")
        self.assertEqual(result["added"], 0)
        self.assertEqual(result["layers"]["spatial"], "ok:remember_candidate")
        self.assertIsNone(log.get("obs_1")["sighting"])
        self.assertEqual(log.spatial.places()[0]["candidates"], ["cand_1"])
        # the same call with the layer on stores the sighting
        on = self._log()
        on.record(_frame(), (0.0, 0.0, 0.0), 0.0, event="capture", step=1)
        ok = on.add_sighting("obs_1", dict(self.SIGHTING), step=2)
        self.assertEqual(ok["status"], "ok:add_sighting")
        self.assertEqual(ok["added"], 1)
        self.assertEqual(on.get("obs_1")["sighting"]["candidate"], "cand_1")

    def test_a_malformed_sighting_is_still_refused_with_the_layer_off(self):
        log = self._log(episodic_memory=False)
        log.record(_frame(), (0.0, 0.0, 0.0), 0.0, step=1)

        result = log.add_sighting("obs_1", {"target": "red box", "basis": "I think so"})

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["rejected"][0]["reason"], "missing_candidate")
        self.assertEqual(len(log.spatial), 0)


class SightingVsGoalTest(_LogCase):
    """What the task is looking for is not what a frame shows."""

    def test_record_separates_goal_target_from_sighting(self):
        log = self._log()
        obs_id = log.record(_frame(), (0.0, 0.0, 0.0), 0.0, goal_target="red box", step=1)
        entry = log.get(obs_id)
        self.assertEqual(entry["goal_target"], "red box")
        self.assertEqual(entry["target_name"], "red box")  # legacy spelling
        self.assertIsNone(entry["sighting"])
        self.assertEqual(entry["evidence_kind"], EVIDENCE_KIND_OBSERVATION)
        # the legacy alias is still the goal target, never a sighting
        legacy = log.record(_frame(0.3), (1.0, 0.0, 0.0), 0.0, target_name="blue crate", step=2)
        self.assertEqual(log.get(legacy)["goal_target"], "blue crate")
        self.assertIsNone(log.get(legacy)["sighting"])

    def test_a_sighting_needs_a_frame_a_candidate_and_a_basis(self):
        log = self._log()
        log.record(_frame(), (0.0, 0.0, 0.0), 0.0,
                   sighting={"target": "red box", "candidate": "cand_1", "basis": "ground_target"})
        with self.assertRaises(ValueError):  # no candidate/box: nothing was pointed at
            log.record(_frame(0.3), (0.0, 0.0, 0.0), 0.0,
                       sighting={"target": "red box", "basis": "I think so"})
        with self.assertRaises(ValueError):  # no basis: an assertion without a reason
            log.record(_frame(0.4), (0.0, 0.0, 0.0), 0.0,
                       sighting={"target": "red box", "candidate": "cand_1"})
        with self.assertRaises(ValueError):  # event-only record: no image to show
            log.record(None, (0.0, 0.0, 0.0), 0.0,
                       sighting={"target": "red box", "candidate": "cand_1", "basis": "b"})
        self.assertEqual(len(log), 1)
        self.assertEqual(log.get("obs_1")["sighting"]["candidate"], "cand_1")
        self.assertEqual(log.stats()["queries"], 0)

    def test_a_sighting_may_name_a_box_instead_of_a_candidate_id(self):
        log = self._log()
        obs_id = log.record(
            _frame(), (0.0, 0.0, 0.0), 0.0,
            sighting={
                "target": "red box", "box": [100, 100, 300, 300], "pixel": [200, 200],
                "basis": "the model pointed at the red box and read 'round'",
                "confidence": 0.7, "attributes": {"color": "red"},
            },
        )
        entry = log.get(obs_id)
        self.assertEqual(entry["sighting"]["box"], [100.0, 100.0, 300.0, 300.0])
        self.assertEqual(entry["sighting"]["image"], obs_id)
        self.assertEqual(entry["sighting"]["confidence"], 0.7)
        self.assertEqual(entry["sighting"]["attributes"], {"color": "red"})

    def test_query_by_target_matches_sightings_only(self):
        log = self._log()
        goal = log.record(_frame(0.2), (0.0, 0.0, 0.0), 0.0, goal_target="red box", step=1)
        seen = log.record(
            _frame(0.6), (2.0, 0.0, 0.0), 0.0, step=2,
            sighting={"target": "red box", "candidate": "cand_1", "basis": "ground_target hit"},
        )
        hits = log.query({"target": "red box"})
        self.assertEqual([hit["obs_id"] for hit in hits], [seen])
        self.assertEqual(hits[0]["evidence_kind"], EVIDENCE_KIND_SIGHTING)
        self.assertEqual(hits[0]["match"], "sighting")
        self.assertEqual(hits[0]["sighting"]["candidate"], "cand_1")

        # a goal frame alone is reported as what it is, never as a sighting
        only_goal = self._log()
        frame = only_goal.record(_frame(), (0.0, 0.0, 0.0), 0.0, goal_target="red box")
        fallback = only_goal.query({"target": "red box"})
        self.assertEqual([hit["obs_id"] for hit in fallback], [frame])
        self.assertEqual(fallback[0]["evidence_kind"], EVIDENCE_KIND_OBSERVATION)
        self.assertEqual(fallback[0]["match"], "goal_frame")

    def test_query_by_goal_target_and_by_sighting_flag(self):
        log = self._log()
        goal = log.record(_frame(0.2), (0.0, 0.0, 0.0), 0.0, goal_target="red box")
        seen = log.record(
            _frame(0.6), (2.0, 0.0, 0.0), 0.0,
            sighting={"target": "blue crate", "candidate": "cand_2", "basis": "ground_target"},
        )
        self.assertEqual([hit["obs_id"] for hit in log.query({"goal_target": "red box"})], [goal])
        self.assertEqual([hit["obs_id"] for hit in log.query({"sighting": True})], [seen])
        self.assertEqual([hit["obs_id"] for hit in log.query({"sighting": "blue crate"})], [seen])
        self.assertEqual(log.query({"sighting": "nothing like that"}), [])

    def test_a_sighting_fans_out_to_the_spatial_and_working_layers(self):
        log = self._log()
        log.working.open_question("where is the red box?")
        obs_id = log.record(
            _frame(), (1.0, 2.0, 0.5), 0.0, step=3,
            sighting={
                "target": "red box", "candidate": "cand_1", "basis": "ground_target hit",
                "confidence": 0.8, "position_flu": [1.0, 2.0, 0.5],
            },
        )
        places = log.spatial.places()
        self.assertEqual(len(places), 1)
        self.assertEqual(places[0]["candidates"], ["cand_1"])
        self.assertEqual(places[0]["position_flu"], [1.0, 2.0, 0.5])
        self.assertIn(f"sighting:{obs_id}", places[0]["sources"])
        candidates = log.working.snapshot()["candidates"]
        self.assertEqual(candidates[0]["id"], "cand_1")
        self.assertEqual(candidates[0]["position_flu"], [1.0, 2.0, 0.5])
        # finding the thing answers the question about it
        self.assertEqual(log.working.snapshot()["open_questions"], [])

    def test_attaching_a_sighting_to_an_existing_frame(self):
        log = self._log()
        obs_id = log.record(_frame(), (1.0, 2.0, 0.5), 0.0, goal_target="red box", step=3)
        result = log.add_sighting(
            obs_id,
            {"target": "red box", "candidate": "cand_2", "basis": "ground_target box=[1,2,3,4]",
             "confidence": 0.6, "position_flu": [1.2, 2.0, 0.5]},
        )
        self.assertEqual(result["status"], "ok:add_sighting")
        self.assertEqual(result["added"], 1)
        self.assertEqual(result["rejected"], [])
        self.assertEqual(result["layers"]["spatial"], "ok:remember_candidate")
        entry = log.get(obs_id)
        self.assertEqual(entry["sighting"]["candidate"], "cand_2")
        self.assertIn(EVENT_SIGHTING, entry["events"])
        self.assertEqual(entry["evidence_kind"], EVIDENCE_KIND_SIGHTING)
        self.assertEqual(log.spatial.places()[0]["candidates"], ["cand_2"])
        # a second sighting on the same frame is kept side by side
        log.add_sighting(obs_id, {"target": "red box", "candidate": "cand_3",
                                  "basis": "the model pointed at a second box"})
        self.assertEqual([item["candidate"] for item in log.get(obs_id)["sightings"]],
                         ["cand_2", "cand_3"])

    def test_an_invalid_or_unknown_sighting_is_reported_not_raised(self):
        log = self._log()
        obs_id = log.record(_frame(), (0.0, 0.0, 0.0), 0.0)
        result = log.add_sighting(obs_id, {"target": "red box"})  # no candidate, no basis
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["added"], 0)
        self.assertEqual(result["rejected"][0]["reason"], "missing_candidate_and_basis")
        self.assertIsNone(log.get(obs_id)["sighting"])
        unknown = log.add_sighting("obs_404", {"target": "red box", "candidate": "cand_1",
                                               "basis": "ground_target"})
        self.assertEqual(unknown["status"], "unknown_observation")
        self.assertEqual(unknown["added"], 0)

    def test_the_entry_marks_the_evidence_kind(self):
        log = self._log()
        sighting = log.record(_frame(0.1), (0.0, 0.0, 0.0), 0.0,
                              sighting={"target": "red box", "candidate": "cand_1", "basis": "g"})
        negative = log.record(_frame(0.2), (1.0, 0.0, 0.0), 0.0,
                              negative={"target": "red box", "reason": "no_candidate"},
                              visibility=view_scope(position_flu=(1.0, 0.0, 0.0), yaw_deg=0.0))
        plain = log.record(_frame(0.4), (2.0, 0.0, 0.0), 0.0)
        kinds = {obs_id: log.get(obs_id)["evidence_kind"]
                 for obs_id in (sighting, negative, plain)}
        self.assertEqual(
            kinds,
            {sighting: EVIDENCE_KIND_SIGHTING, negative: EVIDENCE_KIND_NEGATIVE,
             plain: EVIDENCE_KIND_OBSERVATION},
        )
        # the negative is searchable as such, with its condition boundary
        hits = log.query({"negative": True})
        self.assertEqual([hit["obs_id"] for hit in hits], [negative])
        self.assertEqual(hits[0]["visibility"]["yaw_deg"], 0.0)
        self.assertIn("does not clear the area", hits[0]["scope_note"])
        self.assertIn(EVENT_NOT_FOUND, hits[0]["events"])


class NegativeScopeTest(_LogCase):
    """A null result is bound to the view it was gathered from."""

    def test_a_negative_records_the_view_condition(self):
        log = self._log()
        obs_id = log.record(_frame(), (2.0, 0.0, 0.5), math.radians(30), step=4)
        result = log.add_negative(
            obs_id, target="red box", reason="no_candidate",
            visibility=view_scope(position_flu=(2.0, 0.0, 0.5), yaw_deg=30.0, hfov_deg=90.0,
                                  observation_id=obs_id),
        )
        self.assertEqual(result["status"], "ok:add_negative")
        entry = log.get(obs_id)
        self.assertTrue(entry["negative"])
        self.assertEqual(entry["negative"]["target"], "red box")
        self.assertEqual(entry["negative"]["reason"], "no_candidate")
        self.assertEqual(entry["visibility"]["hfov_deg"], 90.0)
        self.assertIn("does not clear the area", entry["scope_note"])
        self.assertIn(EVENT_NOT_FOUND, entry["events"])
        # the failed check is working memory, and it opens the question it failed
        snapshot = log.working.snapshot()
        self.assertEqual(snapshot["failures"][0]["reason"], "no_candidate")
        self.assertEqual(snapshot["open_questions"][0]["about"], "red box")

    def test_a_negative_without_a_scope_derives_it_from_the_record(self):
        log = self._log()
        obs_id = log.record(_frame(), (3.0, 1.0, 0.5), math.radians(-20))
        log.add_negative(obs_id, target="red box", reason="no_candidate")
        entry = log.get(obs_id)
        self.assertEqual(entry["visibility"]["position_flu"], [3.0, 1.0, 0.5])
        self.assertEqual(entry["visibility"]["yaw_deg"], -20.0)
        self.assertIn("field of view unknown", entry["scope_note"])

    def test_the_negative_marks_the_sector_checked_without_a_result(self):
        log = self._log()
        obs_id = log.record(_frame(), (2.0, 0.0, 0.5), math.radians(30))
        log.add_negative(
            obs_id, target="red box", reason="no_candidate",
            visibility=view_scope(position_flu=(2.0, 0.0, 0.5), yaw_deg=30.0, hfov_deg=90.0),
        )
        coverage = log.spatial.coverage()
        self.assertEqual(coverage["checked_deg"], [[0.0, 45.0]])
        sector = coverage["sectors"][0]
        self.assertEqual(sector["no_result_for"], ["red box"])
        self.assertEqual(sector["observations"], [obs_id])

    def test_an_unknown_frame_is_reported_and_changes_nothing(self):
        log = self._log()
        result = log.add_negative("obs_404", target="red box", reason="no_candidate")
        self.assertEqual(result["status"], "unknown_observation")
        self.assertEqual(log.spatial.stats()["covered_sectors"], 0)
        self.assertEqual(log.working.snapshot()["failures"], [])


if __name__ == "__main__":
    unittest.main()
