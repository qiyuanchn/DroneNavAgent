"""JSON-line manager-side worker that hosts a VLM backend (goal.md M3/M4).

The heavy part of a model provider - transformers, CUDA context, quantized
weights - belongs in its own process. This module is that process:

    python -u -m vlm_nav.worker_model --backend-spec hf \\
        --model-path /models/qwen3-vl --device cuda:0 --dtype bfloat16 \\
        --quantization none

It speaks the same protocol as :mod:`vlm_nav.worker_echo` (one JSON object per
line on stdin/stdout) and serves the two calls of the ``vlm_nav.vlm`` backend
interface:

``vlm.decide``
    params ``{frames, system_prompt, user_prompt, context}`` -> the backend's
    reply text.
``vlm.decide_with_tools``
    params ``{frames, system_prompt, user_prompt, tools_text, context}`` -> the
    backend's reply text.

``ping`` is the handshake and ``shutdown`` ends the process. Frames arrive as
base64 PNG strings; :func:`decode_frame` converts them back to the
``torch.float32`` CHW ``[0, 1]`` tensors that every backend in
:mod:`vlm_nav.vlm` already expects. :func:`encode_frames` is the inverse used by
the manager-side proxy, and accepts the two shapes the runtime produces:
torch CHW floats in ``[0, 1]`` and HWC uint8 arrays (numpy or torch).

A backend is built by :func:`vlm_nav.vlm.make_backend` from ``--backend-spec``
(``mock:*``, ``hf``, ``openai``). Tests and embedders may instead call
:func:`serve` with any object that has the backend methods - no model, GPU or
network is needed for the ``--fake echo``/``--fake crash``/``--fake fatal``
backends below.

Backend stdout is redirected to stderr while loading and while answering, so a
chatty library cannot corrupt the JSON-lines protocol. Protocol replies are
written to the dedicated stdout stream only.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from .vlm import make_backend

METHOD_PING = "ping"
METHOD_DECIDE = "vlm.decide"
METHOD_DECIDE_WITH_TOOLS = "vlm.decide_with_tools"
METHOD_SHUTDOWN = "shutdown"

BACKEND_METHODS = (METHOD_DECIDE, METHOD_DECIDE_WITH_TOOLS)
WORKER_METHODS = (METHOD_PING, *BACKEND_METHODS, METHOD_SHUTDOWN)

DEFAULT_BACKEND_SPEC = "mock:oracle"


class WorkerBackendError(RuntimeError):
    """The worker could not build its VLM backend."""


class FrameEncodingError(ValueError):
    """A frame is not a supported CHW-float or HWC-uint8 image."""


# ---------------------------------------------------------------------------
# frames: torch CHW [0, 1] / HWC uint8 <-> base64 PNG
# ---------------------------------------------------------------------------


def _as_numpy(frame: Any) -> np.ndarray:
    if isinstance(frame, Image.Image):
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)
    if isinstance(frame, torch.Tensor):
        tensor = frame.detach()
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        try:
            return tensor.numpy()
        except Exception as exc:  # noqa: BLE001 - reported as a frame error
            raise FrameEncodingError(
                f"cannot convert a {tensor.dtype} tensor frame: {exc}"
            ) from exc
    if isinstance(frame, np.ndarray):
        return frame
    try:
        return np.asarray(frame)
    except Exception as exc:  # noqa: BLE001 - reported as a frame error
        raise FrameEncodingError(
            f"cannot read frame of type {type(frame).__name__}: {exc}"
        ) from exc


def _to_hwc_uint8(array: np.ndarray) -> np.ndarray:
    """Normalize one frame array to HWC uint8 (2-D grayscale stays 2-D)."""
    array = np.asarray(array)
    if array.ndim == 2:
        if array.dtype.kind == "f":
            array = np.clip(array, 0.0, 1.0) * 255.0
        return np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim != 3:
        raise FrameEncodingError(
            f"frame must be 2-D (H,W) or 3-D (C,H,W)/(H,W,C), got shape "
            f"{tuple(array.shape)}"
        )
    channels_last = array.shape[-1] in (1, 3, 4)
    channels_first = array.shape[0] in (1, 3, 4)
    if array.dtype.kind == "f":
        # The backend contract is torch CHW float in [0, 1].
        if not channels_first:
            raise FrameEncodingError(
                f"float frames must be CHW, got shape {tuple(array.shape)}"
            )
        array = np.transpose(array, (1, 2, 0))
    elif not channels_last:
        if not channels_first:
            raise FrameEncodingError(
                f"frame needs 1/3/4 channels, got shape {tuple(array.shape)}"
            )
        array = np.transpose(array, (1, 2, 0))
    if array.dtype.kind == "f":
        array = np.clip(array, 0.0, 1.0) * 255.0
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255)
    array = array.astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[..., 0]
    return array


def encode_frame(frame: Any) -> str:
    """Encode one frame as a base64 PNG string.

    Accepts a torch CHW float tensor in ``[0, 1]``, an HWC uint8 numpy/torch
    array (or a PIL image). The PNG is lossless, so the only precision loss is
    the 8-bit quantization the manager applies before sending.
    """
    array = _to_hwc_uint8(_as_numpy(frame))
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_frames(frames: Any) -> list[str]:
    """Encode one frame or a sequence of frames; returns base64 PNG strings."""
    if frames is None:
        raise FrameEncodingError("frames is required")
    if isinstance(frames, (torch.Tensor, np.ndarray, Image.Image)):
        frames = [frames]
    elif isinstance(frames, (str, bytes)):
        raise FrameEncodingError(
            "frames must be image arrays/tensors, not encoded strings"
        )
    try:
        items = list(frames)
    except TypeError as exc:
        raise FrameEncodingError(
            f"frames must be a frame or a sequence of frames, got "
            f"{type(frames).__name__}"
        ) from exc
    return [encode_frame(frame) for frame in items]


def decode_frame(payload: str | bytes) -> torch.Tensor:
    """Decode one base64 PNG frame to a ``torch.float32`` CHW ``[0, 1]`` tensor."""
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise FrameEncodingError("frame payload is not base64 text") from exc
    if not isinstance(payload, str) or not payload.strip():
        raise FrameEncodingError("frame payload must be a non-empty base64 string")
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as exc:  # noqa: BLE001 - bad payload is a frame error
        raise FrameEncodingError(f"frame payload is not valid base64: {exc}") from exc
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        raise FrameEncodingError(f"frame payload is not a decodable image: {exc}") from exc
    tensor = torch.from_numpy(np.array(array, dtype=np.uint8, copy=True))
    return tensor.permute(2, 0, 1).to(dtype=torch.float32).div_(255.0).contiguous()


def decode_frames(payloads: Any) -> list[torch.Tensor]:
    """Decode the worker's ``frames`` parameter into a list of CHW tensors."""
    if payloads is None:
        return []
    if isinstance(payloads, (str, bytes)):
        payloads = [payloads]
    if not isinstance(payloads, (list, tuple)):
        raise FrameEncodingError(
            f"frames must be a list of base64 strings, got {type(payloads).__name__}"
        )
    frames: list[torch.Tensor] = []
    for index, payload in enumerate(payloads):
        try:
            frames.append(decode_frame(payload))
        except FrameEncodingError as exc:
            raise FrameEncodingError(f"frame {index}: {exc}") from exc
    return frames


def frame_metadata(frame: Any) -> dict:
    """Shape/dtype/value summary of one decoded frame (for echo tests)."""
    tensor = frame if isinstance(frame, torch.Tensor) else torch.as_tensor(frame)
    tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    values = tensor.to(dtype=torch.float32)
    return {
        "shape": [int(size) for size in tensor.shape],
        "dtype": str(tensor.dtype),
        "min": round(float(values.min()), 6) if values.numel() else 0.0,
        "max": round(float(values.max()), 6) if values.numel() else 0.0,
        "mean": round(float(values.mean()), 6) if values.numel() else 0.0,
    }


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------


class EchoBackend:
    """Deterministic backend that echoes the call and the decoded frame shape.

    It needs no model and is the worker used by the proxy/plugin tests: the
    reply is strict JSON, so a test can assert exactly what crossed the process
    boundary (prompts, tools text, sanitized context and frame tensors).
    """

    name = "echo"

    def _reply(
        self,
        method: str,
        frames: Sequence[torch.Tensor],
        system_prompt: str,
        user_prompt: str,
        tools_text: str,
        context: Any,
    ) -> str:
        return json.dumps(
            {
                "backend": self.name,
                "method": method,
                "frames": [frame_metadata(frame) for frame in frames],
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "tools_text": tools_text,
                "context": context,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    def decide(self, frames, system_prompt, user_prompt, context=None) -> str:
        return self._reply("decide", frames, system_prompt, user_prompt, "", context)

    def decide_with_tools(
        self, frames, system_prompt, user_prompt, tools_text="", context=None
    ) -> str:
        return self._reply(
            "decide_with_tools", frames, system_prompt, user_prompt, tools_text, context
        )

    def close(self) -> None:
        return None


class CrashBackend:
    """Backend whose calls raise: exercises the protocol error reply path."""

    name = "crash"

    def decide(self, frames, system_prompt, user_prompt, context=None) -> str:
        raise RuntimeError("fake backend crash")

    def decide_with_tools(
        self, frames, system_prompt, user_prompt, tools_text="", context=None
    ) -> str:
        raise RuntimeError("fake backend crash")


class FatalBackend:
    """Backend whose calls kill the worker process: exercises the crash path."""

    name = "fatal"

    def decide(self, frames, system_prompt, user_prompt, context=None) -> str:
        os._exit(17)  # noqa: SLF001 - a deliberate process crash for tests

    def decide_with_tools(
        self, frames, system_prompt, user_prompt, tools_text="", context=None
    ) -> str:
        os._exit(17)  # noqa: SLF001 - a deliberate process crash for tests


def fake_backend(name: str) -> Any:
    """The named test backend, or ``None`` when ``name`` is empty."""
    key = str(name or "").strip().lower()
    if not key:
        return None
    if key == "echo":
        return EchoBackend()
    if key == "crash":
        return CrashBackend()
    if key == "fatal":
        return FatalBackend()
    raise WorkerBackendError(f"unknown fake backend {name!r}; choose echo/crash/fatal")


def _warn_unapplied_options(backend: Any, options: Mapping[str, Any]) -> None:
    """Report requested HF precision options the built backend did not apply.

    ``vlm_nav.vlm.make_backend`` currently forwards neither ``dtype`` nor
    ``quantization`` to ``HFLocalVLM``. Loading BF16 when 8/4-bit was requested
    must not be silent, so the effective model is inspected after construction;
    the warning goes to stderr and thus to the managed worker's log file.
    """
    model = getattr(backend, "model", None)
    if model is None:
        return
    requested_quant = str(options.get("quantization") or "none").strip().lower()
    quantized = bool(
        getattr(model, "is_quantized", False)
        or getattr(model, "is_loaded_in_4bit", False)
        or getattr(model, "is_loaded_in_8bit", False)
    )
    if requested_quant not in ("", "none") and not quantized:
        sys.stderr.write(
            f"vlm_nav.worker_model: requested quantization "
            f"{requested_quant!r} was not applied by the backend factory\n"
        )
        sys.stderr.flush()
    # A quantized model's parameter dtype is a storage dtype (e.g. uint8), not
    # the requested compute dtype, so only compare on the unquantized path.
    requested_dtype = str(options.get("dtype") or "").strip().lower()
    if requested_dtype and not quantized:
        try:
            actual = str(next(model.parameters()).dtype).replace("torch.", "")
        except (StopIteration, AttributeError, TypeError):
            actual = ""
        if actual and actual != requested_dtype:
            sys.stderr.write(
                f"vlm_nav.worker_model: requested dtype {requested_dtype!r} "
                f"was not applied by the backend factory (effective {actual!r})\n"
            )
            sys.stderr.flush()


def _torch_dtype(name: str) -> Any:
    try:
        return getattr(torch, str(name or "bfloat16"))
    except AttributeError as exc:
        raise WorkerBackendError(f"unknown dtype {name!r}") from exc


def _quantization_config(name: str, dtype: str) -> Any:
    """Build a BitsAndBytesConfig, or raise before any weights are loaded.

    A requested 8/4-bit load without bitsandbytes is a startup error, never a
    silent BF16 load (docs/vlm_nav_resource_profile_20261020.md).
    """
    quantization = str(name or "none").strip().lower()
    if quantization in ("", "none"):
        return None
    if quantization not in ("8bit", "4bit"):
        raise WorkerBackendError(
            f"unknown quantization {name!r}; choose none/8bit/4bit"
        )
    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:  # noqa: BLE001 - reported as a startup failure
        raise WorkerBackendError(
            f"quantization {quantization!r} requires transformers/bitsandbytes: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=_torch_dtype(dtype),
        bnb_4bit_quant_type="nf4",
    )


def _make_hf_backend(spec: str, options: Mapping[str, Any]) -> Any:
    """Call ``make_backend('hf')`` with the requested dtype/quantization.

    ``vlm_nav.vlm.make_backend`` does not forward these two options to
    ``HFLocalVLM`` in the baseline. The default (BF16, no quantization) path is
    passed through untouched; otherwise ``from_pretrained`` is wrapped once,
    inside the worker process, so the requested precision is what actually
    loads. A missing bitsandbytes fails startup instead of loading BF16.
    """
    quantization = str(options.get("quantization") or "none").strip().lower()
    dtype = str(options.get("dtype") or "bfloat16").strip().lower()
    if quantization in ("", "none") and dtype == "bfloat16":
        return make_backend(spec, **options)
    torch_dtype = _torch_dtype(dtype)
    quantization_config = _quantization_config(quantization, dtype)
    try:
        from transformers import AutoModelForImageTextToText
    except Exception as exc:  # noqa: BLE001 - the HF path needs transformers
        raise WorkerBackendError(
            f"cannot load an HF backend: {type(exc).__name__}: {exc}"
        ) from exc
    original = AutoModelForImageTextToText.from_pretrained

    def from_pretrained(*args, **kwargs):
        kwargs["dtype"] = torch_dtype
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        return original(*args, **kwargs)

    AutoModelForImageTextToText.from_pretrained = from_pretrained
    try:
        return make_backend(spec, **options)
    finally:
        AutoModelForImageTextToText.from_pretrained = original


def build_backend(spec: str = "", *, backend: Any = None, **kwargs) -> Any:
    """Build the worker backend: an injected object wins, else ``make_backend``.

    ``spec`` uses the :func:`vlm_nav.vlm.make_backend` spelling
    (``mock:oracle``, ``mock:forward``, ``hf``, ``openai``). For ``hf`` the
    requested ``dtype``/``quantization`` are applied around the factory; an
    effective precision that still differs is reported on stderr rather than
    being mistaken for the request.
    """
    if backend is not None:
        return backend
    text = str(spec or DEFAULT_BACKEND_SPEC).strip()
    kind = text.partition(":")[0].strip().lower()
    options = {key: value for key, value in kwargs.items() if value is not None}
    if kind == "hf" and not str(options.get("model_path") or "").strip():
        raise WorkerBackendError("backend spec 'hf' requires --model-path")
    if kind == "openai" and not str(options.get("model") or "").strip():
        raise WorkerBackendError("backend spec 'openai' requires --model")
    try:
        built = (
            _make_hf_backend(text, options) if kind == "hf"
            else make_backend(text, **options)
        )
    except WorkerBackendError:
        raise
    except Exception as exc:  # noqa: BLE001 - reported as a startup failure
        raise WorkerBackendError(
            f"cannot build backend {text!r}: {type(exc).__name__}: {exc}"
        ) from exc
    if kind == "hf":
        _warn_unapplied_options(built, options)
    return built


# ---------------------------------------------------------------------------
# JSON-lines server
# ---------------------------------------------------------------------------


def _backend_text(result: Any) -> str:
    """The backend reply as text, preserving a string byte for byte."""
    if isinstance(result, str):
        return result
    if result is None:
        return ""
    if isinstance(result, (dict, list, tuple)):
        return json.dumps(result, ensure_ascii=False, default=str)
    return str(result)


def _stderr_stream() -> Any:
    return sys.stderr if sys.stderr is not None else io.StringIO()


def _call_silently(fn, *args, **kwargs) -> Any:
    """Call a library function without letting it write to protocol stdout."""
    with contextlib.redirect_stdout(_stderr_stream()):
        return fn(*args, **kwargs)


def dispatch(backend: Any, method: str, params: Mapping[str, Any]) -> Any:
    """Run one protocol method and return its JSON-able result."""
    method = str(method or "").strip()
    if method == METHOD_PING:
        return {
            "pong": True,
            "pid": os.getpid(),
            "backend": str(getattr(backend, "name", type(backend).__name__)),
            "methods": list(BACKEND_METHODS),
        }
    if method == METHOD_DECIDE:
        frames = decode_frames(params.get("frames"))
        decide = getattr(backend, "decide", None)
        if not callable(decide):
            raise TypeError("backend has no decide(frames, system_prompt, user_prompt, context)")
        result = _call_silently(
            decide,
            frames,
            str(params.get("system_prompt") or ""),
            str(params.get("user_prompt") or ""),
            params.get("context"),
        )
        return _backend_text(result)
    if method == METHOD_DECIDE_WITH_TOOLS:
        frames = decode_frames(params.get("frames"))
        decide_with_tools = getattr(backend, "decide_with_tools", None)
        if not callable(decide_with_tools):
            raise TypeError(
                "backend has no decide_with_tools("
                "frames, system_prompt, user_prompt, tools_text, context)"
            )
        result = _call_silently(
            decide_with_tools,
            frames,
            str(params.get("system_prompt") or ""),
            str(params.get("user_prompt") or ""),
            str(params.get("tools_text") or ""),
            params.get("context"),
        )
        return _backend_text(result)
    raise ValueError(f"unknown method: {method}")


def _write_reply(
    stream: Any, request_id: Any, *, result: Any = None, error: str = ""
) -> None:
    message: dict[str, Any] = {"id": request_id, "ok": not error}
    if error:
        message["error"] = error
    else:
        message["result"] = result
    stream.write(json.dumps(message, ensure_ascii=False, default=str) + "\n")
    flush = getattr(stream, "flush", None)
    if callable(flush):
        flush()


def serve(
    backend: Any,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    """Serve ``backend`` over JSON lines until ``shutdown`` or EOF."""
    if backend is None:
        raise WorkerBackendError("serve requires a backend")
    input_stream = sys.stdin if stdin is None else stdin
    output_stream = sys.stdout if stdout is None else stdout
    for line in input_stream:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _write_reply(output_stream, None, error=f"invalid json: {exc}")
            continue
        if not isinstance(request, dict):
            _write_reply(output_stream, None, error="request must be a JSON object")
            continue
        request_id = request.get("id")
        method = str(request.get("method", "") or "")
        params = request.get("params") or {}
        if not isinstance(params, dict):
            _write_reply(output_stream, request_id, error="params must be an object")
            continue
        if method == METHOD_SHUTDOWN:
            _write_reply(
                output_stream, request_id, result={"shutdown": True, "pid": os.getpid()}
            )
            return 0
        try:
            result = dispatch(backend, method, params)
        except Exception as exc:  # noqa: BLE001 - reported over the protocol
            _write_reply(
                output_stream, request_id, error=f"{type(exc).__name__}: {exc}"
            )
            continue
        _write_reply(output_stream, request_id, result=result)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _image_size(text: str) -> tuple[int, int]:
    value = str(text or "").lower().replace(" ", "")
    for separator in ("x", ",", "*"):
        if separator in value:
            width, _, height = value.partition(separator)
            try:
                return int(width), int(height)
            except ValueError:
                break
    raise argparse.ArgumentTypeError(f"invalid image size {text!r}; use WxH")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlm_nav.worker_model",
        description="Serve a vlm_nav VLM backend over stdin/stdout JSON lines.",
    )
    parser.add_argument(
        "--backend-spec", default=DEFAULT_BACKEND_SPEC,
        help="make_backend spec: mock:oracle, mock:forward, hf, openai",
    )
    parser.add_argument("--fake", default="",
                        choices=("", "echo", "crash", "fatal"),
                        help="test backend instead of --backend-spec")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization", default="none")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--image-size", type=_image_size, default=(320, 180))
    parser.add_argument("--num-threads", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    backend: Any = None
    try:
        fake = fake_backend(args.fake)
        options: dict[str, Any] = {
            "model_path": args.model_path or None,
            "model": args.model or None,
            "base_url": args.base_url or None,
            "device": args.device,
            "dtype": args.dtype,
            "quantization": args.quantization,
            "max_new_tokens": int(args.max_new_tokens),
            "image_size": tuple(args.image_size),
            "num_threads": int(args.num_threads) or None,
        }
        backend = _call_silently(
            build_backend, args.backend_spec, backend=fake, **options
        )
    except Exception as exc:  # noqa: BLE001 - startup failure goes to the log
        sys.stderr.write(
            f"vlm_nav.worker_model: backend startup failed: "
            f"{type(exc).__name__}: {exc}\n"
        )
        sys.stderr.flush()
        return 2
    try:
        return serve(backend)
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - shutdown is best effort
                pass


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BACKEND_METHODS",
    "CrashBackend",
    "DEFAULT_BACKEND_SPEC",
    "EchoBackend",
    "FatalBackend",
    "FrameEncodingError",
    "METHOD_DECIDE",
    "METHOD_DECIDE_WITH_TOOLS",
    "METHOD_PING",
    "METHOD_SHUTDOWN",
    "WORKER_METHODS",
    "WorkerBackendError",
    "build_backend",
    "build_parser",
    "decode_frame",
    "decode_frames",
    "dispatch",
    "encode_frame",
    "encode_frames",
    "fake_backend",
    "frame_metadata",
    "main",
    "serve",
]
