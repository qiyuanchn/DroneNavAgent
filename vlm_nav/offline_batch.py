"""Offline batch inference through vLLM's in-process ``LLM.chat``.

Per-episode HTTP calls are the right shape for a live agent; an offline replay
benchmark should instead submit the whole batch to vLLM once so its scheduler
can apply continuous batching and prefix sharing across prompts. This module
keeps that path optional: importing it never imports vLLM, and a fake ``llm``
can be injected for tests.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Iterable, Sequence

from PIL import Image


class OfflineBatchError(RuntimeError):
    pass


def _text_of(output: Any) -> str:
    outputs = getattr(output, "outputs", None)
    if outputs is None and isinstance(output, dict):
        outputs = output.get("outputs")
    if not outputs:
        return ""
    first = outputs[0]
    if isinstance(first, dict):
        return str(first.get("text", "") or "")
    return str(getattr(first, "text", "") or "")


class VLLMBatchRunner:
    """Batch chat over a vLLM ``LLM`` instance (or an injected stand-in)."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        llm: Any = None,
        llm_kwargs: dict | None = None,
        sampling_params: Any = None,
    ) -> None:
        if llm is None:
            try:
                from vllm import LLM  # noqa: PLC0415 - optional heavy dependency
            except Exception as exc:  # noqa: BLE001
                raise OfflineBatchError(
                    "vLLM is required for offline batch inference: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not model_path:
                raise OfflineBatchError(
                    "model_path is required when no vLLM LLM instance is injected"
                )
            llm = LLM(model=model_path, **dict(llm_kwargs or {}))
        self.llm = llm
        self.sampling_params = sampling_params

    def chat(
        self,
        conversations: Sequence[Sequence[dict]],
        **chat_kwargs: Any,
    ) -> list[str]:
        """Submit all conversations in one scheduler batch; return texts."""
        if not conversations:
            return []
        kwargs = dict(chat_kwargs)
        if self.sampling_params is not None and "sampling_params" not in kwargs:
            kwargs["sampling_params"] = self.sampling_params
        outputs = self.llm.chat(list(conversations), **kwargs)
        return [_text_of(output) for output in outputs]


def image_to_data_url(
    image: Any, *, image_format: str = "jpeg", quality: int = 90
) -> str:
    """Encode a PIL image or CHW/HWC array for an OpenAI-style message."""
    if not isinstance(image, Image.Image):
        import numpy as np

        array = image
        if hasattr(array, "detach"):
            array = array.detach().clamp(0, 1).cpu().numpy()
        array = np.asarray(array)
        if array.ndim == 3 and array.shape[0] in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=2)
        if array.dtype != np.uint8:
            array = (np.clip(array.astype("float32"), 0.0, 1.0) * 255).astype("uint8")
        image = Image.fromarray(array).convert("RGB")
    buffer = io.BytesIO()
    fmt = str(image_format or "jpeg").lower()
    if fmt == "png":
        image.save(buffer, format="PNG")
        mime = "image/png"
    else:
        image.convert("RGB").save(
            buffer, format="JPEG", quality=max(1, min(100, int(quality)))
        )
        mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buffer.getvalue()).decode()


def messages_with_images(
    system_prompt: str,
    user_prompt: str,
    images: Iterable[Any] = (),
    *,
    image_format: str = "jpeg",
    quality: int = 90,
) -> list[dict]:
    content: list[dict] = [
        {
            "type": "image_url",
            "image_url": {"url": image_to_data_url(
                image, image_format=image_format, quality=quality
            )},
        }
        for image in images
    ]
    content.append({"type": "text", "text": user_prompt})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
