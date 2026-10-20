"""Contact-sheet scan path and the VLM input-size defaults."""

from __future__ import annotations

import tempfile
import unittest

import numpy as np
import torch

from vlm_nav.harness import SemanticNavHarness
from vlm_nav.prompts import SCAN_SYSTEM_PROMPT, build_scan_prompt
from vlm_nav.vision import contact_sheet


def _frame(value: float) -> torch.Tensor:
    return torch.full((3, 18, 32), float(value))


class _ScanWorld:
    def __init__(self, n_views: int = 10) -> None:
        self.position_flu = np.array([0.0, 0.0, 1.0])
        self.yaw_rad = 0.0
        self.target = None
        self.n_views = n_views
        self.rendered = 0

    def scan_views(self, n_views=8, pitch_down_deg=0.0, spread_deg=None):
        frames = [_frame(index / 10.0) for index in range(n_views)]
        self.rendered += len(frames)
        return frames


class _RecordingVLM:
    name = "recording"

    def __init__(self, reply: str = '{"view": 0, "target_visible": false}') -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def decide(self, frames, system_prompt, user_prompt, context=None):
        self.calls.append({
            "frames": list(frames) if isinstance(frames, (list, tuple)) else [frames],
            "system": system_prompt,
            "user": user_prompt,
            "context": dict(context or {}),
        })
        return self.reply


class ContactSheetTest(unittest.TestCase):
    def test_layout_labels_and_tensor_shape(self):
        frames = [_frame(index / 8.0) for index in range(8)]
        sheet = contact_sheet(frames, columns=4, start_index=0)
        self.assertEqual((sheet.columns, sheet.rows), (4, 2))
        self.assertEqual(sheet.tile_width, 32)
        self.assertEqual(sheet.tile_height, 18)
        self.assertEqual(
            sheet.image.size,
            (4 * 32 + 5 * 2, 2 * (18 + 16) + 3 * 2),
        )
        self.assertEqual(tuple(sheet.tensor().shape), (3, sheet.image.size[1], sheet.image.size[0]))
        # tile 5 is in the second row, third column
        left, top, right, bottom = sheet.tile_box(5)
        self.assertEqual((right - left, bottom - top), (32, 18))
        self.assertEqual(left, 2 + 1 * (32 + 2))
        self.assertEqual(top, 2 + 1 * (18 + 16 + 2) + 16)

    def test_map_box_from_sheet_to_tile(self):
        sheet = contact_sheet([_frame(0.0), _frame(0.5)], columns=2, start_index=1)
        # the whole second tile in sheet-normalized coordinates
        left, top, right, bottom = sheet.tile_box(1)
        sheet_w, sheet_h = sheet.image.size
        box = [
            left / sheet_w * 1000.0,
            top / sheet_h * 1000.0,
            right / sheet_w * 1000.0,
            bottom / sheet_h * 1000.0,
        ]
        mapped = sheet.map_box(1, box)
        for value in mapped:
            self.assertAlmostEqual(value, 0.0 if value < 500 else 1000.0, places=3)

    def test_scan_prompt_describes_one_contact_sheet(self):
        text = build_scan_prompt("find the chair", 10, n_pitch=2)
        self.assertIn("ONE contact sheet", text)
        self.assertIn("numbered cells", text)
        self.assertIn("numbers 8 to 9", text)
        self.assertIn("one contact sheet", SCAN_SYSTEM_PROMPT.lower())


class HarnessScanSheetTest(unittest.TestCase):
    def test_look_around_sends_one_sheet_not_ten_frames(self):
        world = _ScanWorld(n_views=10)
        vlm = _RecordingVLM()
        with tempfile.TemporaryDirectory() as tmp:
            harness = SemanticNavHarness(
                world=world,
                vlm=vlm,
                instruction="find the chair",
                out_dir=tmp,
                look_around=False,
                save_frames=False,
                make_video=False,
                scan_views=8,
                scan_pitch_deg=50.0,
            )
            harness._look_around([])
        self.assertEqual(len(vlm.calls), 1)
        self.assertEqual(len(vlm.calls[0]["frames"]), 1, "scan must send one contact sheet")
        self.assertTrue(vlm.calls[0]["context"].get("contact_sheet"))
        self.assertTrue(vlm.calls[0]["context"].get("json_mode"))
        self.assertEqual(vlm.calls[0]["frames"][0].shape[0], 3)
        self.assertIn("contact sheet", vlm.calls[0]["user"].lower())


class DefaultSizeTest(unittest.TestCase):
    def test_backend_defaults_are_sensor_native(self):
        from vlm_nav.vlm import HFLocalVLM, OpenAIVLM

        # read the defaults without constructing a model/session
        import inspect

        hf = inspect.signature(HFLocalVLM.__init__).parameters["image_size"].default
        oa = inspect.signature(OpenAIVLM.__init__).parameters["image_size"].default
        self.assertEqual(hf, (320, 180))
        self.assertEqual(oa, (320, 180))
        self.assertEqual(
            inspect.signature(HFLocalVLM.__init__).parameters["max_new_tokens"].default,
            96,
        )
        from vlm_nav.runtime import AgentRuntime

        self.assertEqual(
            inspect.signature(AgentRuntime.__init__)
            .parameters["max_history_images"].default,
            1,
        )


if __name__ == "__main__":
    unittest.main()
