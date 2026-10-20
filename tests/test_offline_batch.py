"""Offline vLLM batch path: one scheduler call for the whole replay batch."""

from __future__ import annotations

import base64
import io
import sys
import unittest
from unittest import mock

from PIL import Image
import torch

from vlm_nav.offline_batch import (
    OfflineBatchError,
    VLLMBatchRunner,
    image_to_data_url,
    messages_with_images,
)


class _Output:
    def __init__(self, text: str):
        self.outputs = [type("O", (), {"text": text})()]


class _FakeLLM:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def chat(self, conversations, **kwargs):
        self.calls.append({"conversations": conversations, "kwargs": kwargs})
        return [_Output(text) for text in self.texts[: len(conversations)]]


class BatchRunnerTest(unittest.TestCase):
    def test_one_batch_call_and_text_extraction(self):
        llm = _FakeLLM(["first", "second"])
        runner = VLLMBatchRunner(llm=llm)
        texts = runner.chat([
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
        ])
        self.assertEqual(texts, ["first", "second"])
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(len(llm.calls[0]["conversations"]), 2)

    def test_dict_shaped_output_is_supported(self):
        class _DictLLM:
            def chat(self, conversations, **kwargs):
                return [{"outputs": [{"text": "dict text"}]}]

        self.assertEqual(
            VLLMBatchRunner(llm=_DictLLM()).chat([[{"role": "user", "content": "x"}]]),
            ["dict text"],
        )

    def test_missing_vllm_is_a_clear_error(self):
        with mock.patch.dict(sys.modules, {"vllm": None}):
            with self.assertRaises(OfflineBatchError):
                VLLMBatchRunner(model_path="/models/qwen")

    def test_empty_batch_does_not_call_the_model(self):
        llm = _FakeLLM([])
        self.assertEqual(VLLMBatchRunner(llm=llm).chat([]), [])
        self.assertEqual(llm.calls, [])


class MessageTest(unittest.TestCase):
    def test_image_encoding_and_message_shape(self):
        frame = torch.zeros(3, 18, 32)
        url = image_to_data_url(frame)
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
            self.assertEqual(image.size, (32, 18))
        messages = messages_with_images("SYS", "USER", [frame, frame])
        content = messages[1]["content"]
        self.assertEqual([item["type"] for item in content], ["image_url", "image_url", "text"])
        self.assertEqual(content[-1], {"type": "text", "text": "USER"})
        self.assertEqual(messages[0], {"role": "system", "content": "SYS"})


if __name__ == "__main__":
    unittest.main()
