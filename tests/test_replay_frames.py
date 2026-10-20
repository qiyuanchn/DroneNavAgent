"""The frozen observation replay contract (``scripts/replay_frames.py``).

A replay set is the comparable input for resource measurements across the A100
and a 4090 (or between harness variants): frames, poses and intrinsics that do
not move when the renderer or the model changes. These tests pin the schema and
the refusals - too few views, a missing/unreadable frame, an unsafe relative
path, a non-finite pose, an unknown schema, a checksum mismatch - so a broken
replay is a load error and never a silently degraded benchmark.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "replay_frames.py"


def _load_module():
    name = "diffaero_script_replay_frames"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve string annotations via sys.modules
    spec.loader.exec_module(module)
    return module


REPLAY = _load_module()


def _write_frame(path: Path, color: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 180), color).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_replay(
    root: Path,
    *,
    n_views: int = 12,
    schema: str = REPLAY.REPLAY_SCHEMA,
    instruction: str = "在卧室窗边找到木椅",
    camera: dict | None = None,
    pose=None,
    with_checksums: bool = True,
    file_names: list[str] | None = None,
) -> Path:
    """A minimal replay on disk; each view is a solid colour so it is easy to
    tell frames apart without any renderer."""
    camera = dict(
        camera
        or {"width": 320, "height": 180, "hfov_deg": 86.0}
    )
    views = []
    for index in range(n_views):
        name = (
            file_names[index]
            if file_names is not None and index < len(file_names)
            else f"frames/view_{index:02d}.png"
        )
        path = root / name
        digest = _write_frame(
            path, ((index * 17) % 256, (index * 31) % 256, (index * 47) % 256)
        )
        view_pose = list(pose[index]) if isinstance(pose, list) else [1.0, 2.0, -2.4]
        view = {
            "id": f"view_{index:02d}",
            "file": name,
            "pose_flu": view_pose,
            "yaw_deg": float(index * 30 % 360),
            "pitch_deg": 0.0,
            "phase": "scan" if index < n_views - 2 else "pitched",
            "index": index,
        }
        if with_checksums:
            view["sha256"] = digest
        views.append(view)
    manifest = {
        "schema": schema,
        "name": "window_chair",
        "instruction": instruction,
        "created_utc": "2026-10-20T00:00:00Z",
        "camera": camera,
        "views": views,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


class ReplayLoadTest(unittest.TestCase):
    def test_loads_frozen_views_with_pose_and_intrinsics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_replay(root)
            replay = REPLAY.load_replay(root)

            self.assertEqual(replay.name, "window_chair")
            self.assertEqual(replay.instruction, "在卧室窗边找到木椅")
            self.assertEqual(len(replay.views), 12)
            self.assertEqual(len(replay), 12)
            self.assertEqual(replay.camera.width, 320)
            self.assertEqual(replay.camera.height, 180)
            self.assertAlmostEqual(replay.camera.hfov_deg, 86.0)
            intrinsics = replay.camera.intrinsics()
            self.assertEqual(intrinsics["width"], 320)
            self.assertTrue(math.isfinite(intrinsics["fx"]))
            self.assertGreater(intrinsics["fx"], 0.0)

            first = replay.views[0]
            self.assertEqual(first.view_id, "view_00")
            self.assertEqual(first.pose_flu, (1.0, 2.0, -2.4))
            self.assertEqual(first.phase, "scan")
            self.assertEqual(replay.phases(), {"scan": 10, "pitched": 2})

            frames = replay.frames()
            self.assertEqual(tuple(frames.shape), (12, 3, 180, 320))
            self.assertGreaterEqual(float(frames.min()), 0.0)
            self.assertLessEqual(float(frames.max()), 1.0)
            self.assertEqual(replay.pil_frame(replay.views[0]).size, (320, 180))

    def test_resized_frames_follow_the_vlm_input_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_replay(root)
            replay = REPLAY.load_replay(root)
            frame = replay.views[0].load_frame(image_size=(640, 360))
        self.assertEqual(tuple(frame.shape), (3, 360, 640))

    def test_too_few_views_is_a_load_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_replay(Path(tmp), n_views=11)
            with self.assertRaises(REPLAY.ReplayError) as caught:
                REPLAY.load_replay(Path(tmp))
        self.assertIn("12", str(caught.exception))

    def test_missing_frame_is_a_load_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_replay(root)
            (root / "frames" / "view_03.png").unlink()
            with self.assertRaises(REPLAY.ReplayError) as caught:
                REPLAY.load_replay(root)
        self.assertIn("view_03", str(caught.exception))

    def test_unsafe_relative_path_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = [f"frames/view_{index:02d}.png" for index in range(11)]
            names.append("../escape.png")
            write_replay(root, file_names=names)
            # the traversal target exists outside the replay root
            (root.parent / "escape.png").write_bytes(
                (root / "frames" / "view_00.png").read_bytes()
            )
            with self.assertRaises(REPLAY.ReplayError) as caught:
                REPLAY.load_replay(root)
        self.assertIn("outside", str(caught.exception).lower())

    def test_absolute_frame_path_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = [f"frames/view_{index:02d}.png" for index in range(11)]
            names.append("/tmp/x.png")
            write_replay(root, file_names=names)
            with self.assertRaises(REPLAY.ReplayError):
                REPLAY.load_replay(root)

    def test_unknown_schema_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_replay(Path(tmp), schema="something.else")
            with self.assertRaises(REPLAY.ReplayError) as caught:
                REPLAY.load_replay(Path(tmp))
        self.assertIn("schema", str(caught.exception).lower())

    def test_non_finite_pose_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            poses = [[1.0, 2.0, -2.4] for _ in range(12)]
            poses[4] = [1.0, float("nan"), -2.4]
            write_replay(root, pose=poses)
            with self.assertRaises(REPLAY.ReplayError) as caught:
                REPLAY.load_replay(root)
        self.assertIn("pose", str(caught.exception).lower())

    def test_non_finite_yaw_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_replay(root)
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["views"][2]["yaw_deg"] = float("inf")
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(REPLAY.ReplayError):
                REPLAY.load_replay(root)

    def test_checksum_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_replay(root)
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["views"][0]["sha256"] = "0" * 64
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(REPLAY.ReplayError) as caught:
                REPLAY.load_replay(root)
        self.assertIn("sha256", str(caught.exception).lower())

    def test_checksum_can_be_skipped_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_replay(root)
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["views"][0]["sha256"] = "0" * 64
            (root / "manifest.json").write_text(json.dumps(manifest))
            replay = REPLAY.load_replay(root, verify_checksums=False)
        self.assertEqual(len(replay.views), 12)

    def test_lookup_by_id_and_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_replay(root)
            replay = REPLAY.load_replay(root)
        self.assertEqual(replay.view("view_05").view_id, "view_05")
        self.assertEqual(len(replay.views_in_phase("pitched")), 2)
        self.assertEqual(len(replay.views_in_phase("scan")), 10)
        with self.assertRaises(KeyError):
            replay.view("view_99")

    def test_summary_has_no_live_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_replay(root)
            replay = REPLAY.load_replay(root)
            summary = replay.summary()
        self.assertEqual(summary["schema"], REPLAY.REPLAY_SCHEMA)
        self.assertEqual(summary["frame_count"], 12)
        self.assertEqual(summary["phases"], {"scan": 10, "pitched": 2})
        self.assertIn("manifest_sha256", summary)


class FrozenReplayAssetTest(unittest.TestCase):
    """The committed experiment artifact, when it exists in this checkout."""

    def test_window_chair_replay_if_present(self):
        root = ROOT / "outputs" / "vlm_nav" / "replay" / "window_chair"
        if not (root / "manifest.json").is_file():
            self.skipTest("window_chair replay has not been generated in this checkout")
        replay = REPLAY.load_replay(root)
        self.assertGreaterEqual(len(replay.views), 12)
        self.assertTrue(replay.instruction)
        phases = replay.phases()
        self.assertGreaterEqual(len(phases), 2)
        # every view is a real frame at the declared camera size
        for view in replay.views:
            frame = view.load_frame()
            self.assertEqual(frame.shape[-1], replay.camera.width)
            self.assertEqual(frame.shape[-2], replay.camera.height)


if __name__ == "__main__":
    unittest.main()
