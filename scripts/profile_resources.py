#!/usr/bin/env python
"""Measure the resources of the VLM-navigation stack and freeze the numbers.

This driver answers the 4090 question with A100 measurements instead of guesses:
it records, for every component, the peak CUDA memory (allocated and reserved,
plus the driver's own used bytes) and the first-call / steady-state latency.

Sections
--------
``vlm``
    Qwen3-VL-8B load; one-frame and eight-view inference; first call vs
    steady state; serial calls and the memory they leave behind (the KV-cache
    question); token/s. ``--quantization 8bit|4bit`` loads through
    bitsandbytes when it is installed and records an explicit ``unavailable``
    result when it is not - a failed quantization never silently becomes BF16.
``detector``
    SAM3 + DINOv2 load, one ``detect`` call and one ``embed_boxes`` call.
``render``
    HiGS single-scene load, one render, an eight-view batch and eight
    production ``scan_views`` renders, for the bedroom and classroom scenes.
``confirm``
    One complete ``confirm_candidate`` routine on the window-chair scene
    (approach + fresh views + VLM judgement + box depth + verdict) and one
    pitched-depth relocalization, both measured end to end.
``replay``
    The frozen observation set at ``outputs/vlm_nav/replay/window_chair``
    (``scripts/build_replay_window_chair.py``): frame count, phases, camera.
``audit``
    Every episode ``summary.json`` under ``--audit-root`` is checked against
    the minimal fields that make a silent fallback (detector off, mock backend,
    no depth, CPU renderer, OOM) visible after the fact.

Outputs ``report.json`` and ``report.md`` under ``--out`` (default
``outputs/vlm_nav/resource_profile``). ``--dry-run`` skips every GPU section but
still validates/writes the replay and audit sections, so the report shape is
testable without a device.

    CUDA_VISIBLE_DEVICES=0 python scripts/profile_resources.py --sections all \
        --out outputs/vlm_nav/resource_profile

    # comparable VLM replay measurement on any host
    CUDA_VISIBLE_DEVICES=0 python scripts/profile_resources.py --sections vlm \
        --replay --replay-dir outputs/vlm_nav/replay/window_chair
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
REPORT_SCHEMA = "vlm_nav.resource_profile.v1"
DEFAULT_OUT = REPO / "outputs" / "vlm_nav" / "resource_profile"
DEFAULT_REPLAY = REPO / "outputs" / "vlm_nav" / "replay" / "window_chair"
DEFAULT_MODEL = "/models_ro/Qwen3-VL-8B"
DEFAULT_SAM3 = "/models_ro/sam3"
DEFAULT_DINO = "/models_ro/dinov2-base"
DEFAULT_BEDROOM_PLY = "/assets/scenes/bedroom.ply"
DEFAULT_CLASSROOM_PLY = "/assets/scenes/classroom.ply"
DEFAULT_HOLDOUT_JSON = (
    REPO / "outputs" / "vlm_nav" / "verify_long_horizon_holdout" / "scenes_bedroom"
    / "combo_000000_seed_31001_bedroom_1f3a2fb80c29.json"
)
DEFAULT_HOLDOUT_SPZ = DEFAULT_HOLDOUT_JSON.with_suffix(".spz")

#: The minimal per-episode summary fields an OOM/fallback audit needs. Without
#: them a run that fell back to the mock backend, disabled SAM3, lost depth or
#: silently ran on CPU looks identical to a healthy one.
REQUIRED_SUMMARY_FIELDS: tuple[str, ...] = (
    "runtime.backend.kind",
    "runtime.backend.model",
    "runtime.backend.device",
    "runtime.backend.dtype",
    "runtime.backend.quantization",
    "runtime.detector.requested",
    "runtime.detector.active",
    "runtime.detector.reason",
    "runtime.renderer.backend",
    "runtime.renderer.device",
    "resources.peak_allocated_mib",
    "resources.peak_reserved_mib",
    "resources.peak_driver_used_mib",
    "fallbacks",
)

REQUIRED_SUMMARY_WHY: dict[str, str] = {
    "runtime.backend.kind": "hf | openai | mock - mock must never look like a real model",
    "runtime.backend.model": "path or model id actually loaded",
    "runtime.backend.device": "cuda:0 vs cpu; the GPU the process really used",
    "runtime.backend.dtype": "bfloat16 / float16 / float32",
    "runtime.backend.quantization": "none | 8bit | 4bit (and the bitsandbytes fallback)",
    "runtime.detector.requested": "off | sam3 | module:factory as asked for",
    "runtime.detector.active": "whether the gate really ran or was skipped",
    "runtime.detector.reason": "missing weights / no CUDA / import / disabled (empty when active)",
    "runtime.renderer.backend": "higs_ut | cpu_splat | env",
    "runtime.renderer.device": "the render device",
    "resources.peak_allocated_mib": "torch.cuda.max_memory_allocated for the episode",
    "resources.peak_reserved_mib": "torch.cuda.max_memory_reserved for the episode",
    "resources.peak_driver_used_mib": "driver-side used bytes (context included)",
    "fallbacks": "list of {component, requested, active, reason, step}; [] when clean",
}

#: Fields where an empty string is a legitimate value (an active detector has
#: no failure reason). Every other required string must be non-empty.
ALLOW_EMPTY_STRING_FIELDS = frozenset({"runtime.detector.reason"})

FALLBACK_CHECK_FIELDS = ("stack_backend", "stack_detail.tools_source", "stack_detail.memory")

#: The points where a component can degrade without raising: each one must
#: leave an explicit event or field in the episode summary.
FALLBACK_CHECKPOINTS: tuple[tuple[str, str], ...] = (
    (
        "backend selection",
        "make_backend / --backend defaults to mock:oracle; a mock run must carry "
        "runtime.backend.kind=mock and can never count as a model success.",
    ),
    (
        "detector availability",
        "Sam3DinoDetector.available() is False on missing weights, no CUDA or an "
        "import error; resolve_detector then returns NullDetector. Record "
        "requested/active/reason.",
    ),
    (
        "depth reads",
        "world.depth_at_pixel returns None (no Gaussian, missing mirror, no "
        "model); confirm_candidate reports no_depth_measurement. Record the "
        "depth backend and the reason per episode.",
    ),
    (
        "renderer backend",
        "HiGS requires a built CUDA extension; the CPU splat renderer is a "
        "different backend. Record renderer.backend/device and the extension "
        "cache key / TORCH_CUDA_ARCH_LIST.",
    ),
    (
        "quantization",
        "A requested 8-bit/4-bit load needs bitsandbytes; _load_hf returns "
        "available=False with quantization_error instead of loading BF16 and "
        "calling it quantized.",
    ),
    (
        "VLM device",
        "run_vlm_nav_3dgs.py / run_vlm_nav_tools.py default --device cpu; an "
        "accidental CPU run is not an OOM but is equally incomparable. Record "
        "the device actually used.",
    ),
    (
        "OOM retry",
        "If a runner catches torch.cuda.OutOfMemoryError and retries with fewer "
        "views / smaller images / another device, append a fallbacks entry "
        "with the original request, the retry and the step.",
    ),
    (
        "scene load",
        "A failed SPZ/PLY load may skip a cell (finish_reason=infeasible_skip); "
        "record the scene path, scale and Gaussian count so a skip is not read "
        "as a navigation failure.",
    ),
)

#: Hard-coded/assumption inventory for the 4090 port. Each entry is
#: ``(area, location, finding, action)``; locations are repo-relative.
ADAPTATION_NOTES: tuple[tuple[str, str, str, str], ...] = (
    (
        "VLM dtype",
        "vlm_nav/vlm.py:633",
        'HFLocalVLM defaults to dtype="bfloat16"; the runner passes no override.',
        "Keep BF16 on Ada (4090 supports it). For 24 GB, 8-bit is the first step; 4-bit only if the vision tower also fits.",
    ),
    (
        "VLM device",
        "vlm_nav/vlm.py:656; scripts/run_vlm_nav_3dgs.py:84; scripts/run_vlm_nav_tools.py:946",
        'device_map=self.device; runner defaults to --device cpu (a silent slow path if forgotten).',
        "Pass --device cuda:0 explicitly and record it in summary.json. On a dual-GPU host CUDA_VISIBLE_DEVICES must pin one GPU.",
    ),
    (
        "Detector paths",
        "vlm_nav/detectors.py:51-52",
        '/models_ro/sam3 and /models_ro/dinov2-base are container mount points.',
        "On a 4090 host either mount the same paths or pass --sam3-path/--dino-path; record both in the summary.",
    ),
    (
        "Detector device/precision",
        "vlm_nav/detectors.py:405; 494-503",
        'Sam3DinoDetector defaults to "cuda:0" and loads SAM3/DINOv2 in FP32 (no dtype argument).',
        "FP32 is ~2x the BF16 footprint; there is no detector-device CLI knob, so pin the physical GPU with CUDA_VISIBLE_DEVICES and keep FP32 for parity unless measured, then try FP16/BF16 with a score-parity check.",
    ),
    (
        "HiGS extension cache",
        "utils/gaussian_splatting/higs/native/build.py:31-49",
        "The extension name hashes sources + compiler flags + TORCH_CUDA_ARCH_LIST + torch/CUDA versions; it lands in TORCH_EXTENSIONS_DIR (or ~/.cache/torch_extensions).",
        "On 4090 set TORCH_CUDA_ARCH_LIST=8.9 (or 8.0;8.9), TORCH_EXTENSIONS_DIR=<repo>/outputs/vlm_nav/torch_extensions, and expect a fresh compile hash.",
    ),
    (
        "Rasterizer arch list",
        "script/build_batch_rasterizer.sh:8,40",
        'DEFAULT_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX" - 8.9 (Ada/4090) is already included.',
        "Keep the default; a 4090-only host may narrow it to 8.9 to cut build time.",
    ),
    (
        "Renderer precision",
        "utils/gaussian_splatting/higs/native/backend.py:42-80,111-115",
        "Means stay FP32; quaternions/scales/opacity/colors and the render buffer are FP16. Output is FP16 [V,H,W,4].",
        "No 4090 change needed; the FP16 path is Ada-native. Peak memory scales with the 1.2M-Gaussian packed scene.",
    ),
    (
        "Frame size",
        "vlm_nav/vlm.py:637; cfg/sensor/3dgs_rgb.yaml:3-5",
        "VLM input is resized to 640x360 by default; the sensor renders 320x180.",
        "The 8-view call is dominated by vision-tower tokens; lowering image_size is the cheapest 4090 lever, but it changes the benchmark - freeze it in the replay manifest.",
    ),
    (
        "Generation cap",
        "vlm_nav/vlm.py:634; scripts/run_vlm_nav_3dgs.py:87",
        "192 new tokens by default; per-turn context may only raise it.",
        "Keep the cap in the summary; decode time and KV growth are not comparable otherwise.",
    ),
    (
        "Scene assets",
        "vlm_nav/world.py:619-625; scripts/build_replay_window_chair.py:65-70",
        "HiGS loads PLY scenes; the replay source is an SPZ converted once to PLY (1.2M Gaussians).",
        "Ship the PLY or pre-convert on the 4090 host; report the Gaussian count and scene scale with every measurement.",
    ),
    (
        "CUDA arch env",
        "container env / scripts/verify_long_horizon.py:158",
        "TORCH_CUDA_ARCH_LIST and TORCH_EXTENSIONS_DIR are set by the shell, not the code.",
        "Record both in summary.json; a cache hit built without 8.9 will JIT-fail or rebuild on a 4090.",
    ),
)


# ---------------------------------------------------------------------------
# small pure helpers (unit-tested on CPU)
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Linear-interpolated percentile; ``None`` for an empty sample."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(max(float(fraction), 0.0), 1.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_timings(seconds: Sequence[float]) -> dict:
    """First-call and steady-state latency of one measurement phase.

    The first call carries CUDA kernel warmup / cache allocation; folding it
    into the mean hides exactly the cost a 4090 port must budget for. So
    ``first_s`` and the ``steady_*`` statistics (calls 2..N) are separate.
    """
    values = [float(value) for value in seconds if value is not None]
    if not values:
        return {
            "count": 0, "total_s": 0.0, "first_s": None, "mean_s": None,
            "median_s": None, "min_s": None, "max_s": None, "p90_s": None,
            "p95_s": None, "steady_mean_s": None, "steady_median_s": None,
            "steady_count": 0,
        }
    steady = values[1:]
    return {
        "count": len(values),
        "total_s": round(sum(values), 6),
        "first_s": round(values[0], 6),
        "mean_s": round(statistics.fmean(values), 6),
        "median_s": round(statistics.median(values), 6),
        "min_s": round(min(values), 6),
        "max_s": round(max(values), 6),
        "p90_s": round(percentile(values, 0.90) or 0.0, 6),
        "p95_s": round(percentile(values, 0.95) or 0.0, 6),
        "steady_count": len(steady),
        "steady_mean_s": round(statistics.fmean(steady), 6) if steady else None,
        "steady_median_s": round(statistics.median(steady), 6) if steady else None,
    }


def tokens_per_second(tokens: int, seconds: float | None) -> float | None:
    if seconds is None or seconds <= 0.0 or int(tokens) <= 0:
        return None
    return round(int(tokens) / float(seconds), 3)


def quantization_estimate(
    parameter_count: float,
    *,
    overhead_gib: float = 1.0,
) -> dict:
    """Static weight/activation budget for BF16, INT8 and INT4.

    A *conversion*, explicitly not a 4090 measurement: the parameter count is
    the model's, the per-parameter bytes are nominal, and ``overhead_gib`` is
    the caller's allowance for the vision tower, activations, CUDA context and
    the KV cache. The measured A100 BF16 peak replaces the estimate when it is
    available (``measured_bf16_peak_mib`` in the report).
    """
    gib = 1024.0 ** 3
    params = max(0.0, float(parameter_count))
    bf16 = params * 2.0 / gib
    int8 = params * 1.0 / gib
    int4 = params * 0.5 / gib
    resolved_overhead = max(0.0, float(overhead_gib))
    return {
        "kind": "estimate",
        "parameter_count": params,
        "overhead_gib": resolved_overhead,
        "bf16_weights_gib": round(bf16, 4),
        "int8_weights_gib": round(int8, 4),
        "int4_weights_gib": round(int4, 4),
        "bf16_with_overhead_gib": round(bf16 + resolved_overhead, 4),
        "int8_with_overhead_gib": round(int8 + resolved_overhead, 4),
        "int4_with_overhead_gib": round(int4 + resolved_overhead, 4),
        "note": (
            "conversion only, not a 4090 measurement; bitsandbytes must be "
            "installed and the same model/processor must be used for a real run"
        ),
    }


def _lookup(summary: Mapping, dotted: str) -> tuple[bool, Any]:
    current: Any = summary
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _is_missing(field: str, value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return field not in ALLOW_EMPTY_STRING_FIELDS
    return False


def audit_episode_summary(summary: Mapping) -> dict:
    """Check one episode summary for the fields that expose silent fallbacks."""
    missing: list[dict] = []
    present: dict[str, Any] = {}
    for field in REQUIRED_SUMMARY_FIELDS:
        found, value = _lookup(summary, field)
        if not found or _is_missing(field, value):
            missing.append(
                {"field": field, "reason": REQUIRED_SUMMARY_WHY.get(field, "")}
            )
        else:
            present[field] = value
    # The runner already records these on the module path; a fallback value is
    # an explicit silent-degradation signal and must be surfaced as a finding.
    for field in FALLBACK_CHECK_FIELDS:
        found, value = _lookup(summary, field)
        if not found or value is None:
            continue
        text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
        if field == "stack_backend" and value == "fallback":
            missing.append(
                {
                    "field": field,
                    "reason": "fallback registry/runtime was active",
                    "signal": "stack_backend=fallback",
                }
            )
        elif field == "stack_detail.tools_source" and isinstance(value, Mapping):
            fallback_tools = sorted(name for name, source in value.items() if source == "fallback")
            if fallback_tools:
                missing.append(
                    {
                        "field": field,
                        "reason": f"fallback tools: {', '.join(fallback_tools[:8])}",
                        "signal": "stack_detail.tools_source=fallback",
                    }
                )
        elif field == "stack_detail.memory" and "fallback" in str(text).lower():
            missing.append(
                {
                    "field": field,
                    "reason": "fallback memory was active",
                    "signal": "stack_detail.memory=fallback",
                }
            )
    return {"ok": not missing, "missing": missing, "present": present}


def _summary_files(paths: Iterable[str | Path]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw).expanduser()
        candidates: list[Path]
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            direct = path / "summary.json"
            candidates = [direct] if direct.is_file() else []
            candidates += sorted(path.glob("**/summary.json"))
        else:
            continue
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(candidate)
    return found


def audit_summary_paths(
    paths: Iterable[str | Path],
    *,
    max_summaries: int = 2000,
) -> dict:
    """Audit every ``summary.json`` under the given files/directories."""
    files = _summary_files(paths)[: int(max_summaries)]
    incomplete: list[dict] = []
    field_counts: dict[str, int] = {}
    complete = 0
    for path in files:
        try:
            summary = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            incomplete.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        audit = audit_episode_summary(summary)
        if audit["ok"]:
            complete += 1
            continue
        for item in audit["missing"]:
            field_counts[item["field"]] = field_counts.get(item["field"], 0) + 1
        incomplete.append(
            {
                "path": str(path),
                "missing": audit["missing"],
                "stack_backend": summary.get("stack_backend"),
                "success": summary.get("success"),
                "finish_reason": summary.get("finish_reason"),
            }
        )
    return {
        "summaries": len(files),
        "complete": complete,
        "incomplete": incomplete,
        "missing_field_counts": dict(sorted(field_counts.items())),
        "required_fields": [
            {"field": field, "why": REQUIRED_SUMMARY_WHY.get(field, "")}
            for field in REQUIRED_SUMMARY_FIELDS
        ],
    }


# ---------------------------------------------------------------------------
# CUDA measurement helpers (imported lazily so --dry-run stays CPU-only)
# ---------------------------------------------------------------------------


def _torch():
    import torch

    return torch


def cuda_memory(device: Any) -> dict:
    torch = _torch()
    if not torch.cuda.is_available():
        return {"cuda": False}
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    mib = 1024.0 * 1024.0
    return {
        "cuda": True,
        "allocated_mib": round(torch.cuda.memory_allocated(device) / mib, 1),
        "reserved_mib": round(torch.cuda.memory_reserved(device) / mib, 1),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / mib, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / mib, 1),
        "driver_used_mib": round((total - free) / mib, 1),
        "free_mib": round(free / mib, 1),
        "total_mib": round(total / mib, 1),
    }


def reset_peak(device: Any) -> dict:
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return cuda_memory(device)


def _timed(fn: Callable[[], Any], device: Any) -> tuple[Any, float, dict]:
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    return result, seconds, cuda_memory(device)


def _environment_meta(device: Any) -> dict:
    import os

    torch = _torch()
    meta = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST", ""),
        "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR", ""),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import transformers

        meta["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001 - a missing library is a finding, not a crash
        meta["transformers"] = None
    if torch.cuda.is_available():
        meta["gpu_name"] = torch.cuda.get_device_name(device)
        meta["gpu_capability"] = ".".join(str(v) for v in torch.cuda.get_device_capability(device))
        meta["gpu_total_mib"] = round(torch.cuda.get_device_properties(device).total_memory / 2**20, 1)
    try:
        import bitsandbytes as bnb

        meta["bitsandbytes"] = bnb.__version__
    except Exception:  # noqa: BLE001
        meta["bitsandbytes"] = None
    return meta


# ---------------------------------------------------------------------------
# VLM section
# ---------------------------------------------------------------------------


def _load_hf(model_path: str, device: str, dtype: str, quantization: str) -> dict:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
    except Exception:  # noqa: BLE001 - the logging knob is best effort
        pass

    torch_dtype = getattr(torch, dtype)
    kwargs: dict[str, Any] = {"dtype": torch_dtype, "device_map": device}
    quantization_error = None
    if quantization and quantization != "none":
        try:
            from transformers import BitsAndBytesConfig

            if quantization == "8bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            elif quantization == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                raise ValueError(f"unknown quantization {quantization!r}")
        except Exception as exc:  # noqa: BLE001 - recorded, never hidden
            quantization_error = f"{type(exc).__name__}: {exc}"
            return {"available": False, "reason": quantization_error}
    try:
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
        model.eval()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    parameter_count = None
    try:
        parameter_count = int(model.num_parameters())
    except Exception:  # noqa: BLE001
        parameter_count = sum(p.numel() for p in model.parameters())
    first_parameter = next(model.parameters(), None)
    return {
        "available": True,
        "processor": processor,
        "model": model,
        "parameter_count": parameter_count,
        "parameter_dtype": str(first_parameter.dtype) if first_parameter is not None else None,
        "quantization": quantization,
        "quantization_error": quantization_error,
        "model_path": model_path,
        "device": device,
    }


def _chat_prompt(processor: Any, system: str, user: str, n_images: int) -> str:
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                *[{"type": "image"} for _ in range(int(n_images))],
                {"type": "text", "text": user},
            ],
        },
    ]
    try:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:  # noqa: BLE001 - small templates lack a system role
        fallback = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in range(int(n_images))],
                    {"type": "text", "text": system + "\n\n" + user},
                ],
            }
        ]
        return processor.apply_chat_template(
            fallback, tokenize=False, add_generation_prompt=True
        )


def _as_pil_image(image: Any, image_size: Sequence[int] | None = None):
    """The VLM backend's image contract: RGB PIL, optionally resized.

    The confirm flow hands CHW float tensors (the harness frame contract), while
    the replay path already hands PIL. A raw torch tensor given to the Qwen
    processor is not an image - it produced black frames in the first profile
    run - so every frame is normalized here, exactly like ``HFLocalVLM``.
    """
    from PIL import Image

    if isinstance(image, Image.Image):
        frame = image.convert("RGB")
    else:
        import numpy as np
        import torch

        from vlm_nav.vlm import frame_to_pil

        if isinstance(image, torch.Tensor):
            frame = frame_to_pil(image)
        else:
            array = np.asarray(image)
            if array.ndim == 2:
                frame = Image.fromarray(array.astype(np.uint8), mode="L").convert("RGB")
            elif array.ndim == 3 and array.shape[-1] in (3, 4):
                frame = Image.fromarray(array[..., :3].astype(np.uint8), mode="RGB")
            elif array.ndim == 3 and array.shape[0] in (1, 3):
                frame = frame_to_pil(torch.as_tensor(array, dtype=torch.float32))
            else:
                raise ValueError(f"cannot convert image with shape {array.shape} to PIL")
    if image_size is not None:
        width, height = (int(v) for v in image_size)
        if width > 0 and height > 0 and frame.size != (width, height):
            frame = frame.resize((width, height), Image.BILINEAR)
    return frame


def _generate_once(
    processor: Any,
    model: Any,
    images: Sequence[Any],
    system: str,
    user: str,
    device: Any,
    max_new_tokens: int,
    image_size: Sequence[int] | None = None,
) -> dict:
    import torch

    pil_images = [_as_pil_image(image, image_size=image_size) for image in images]
    text = _chat_prompt(processor, system, user, len(pil_images))
    started = time.perf_counter()
    inputs = processor(text=[text], images=pil_images, return_tensors="pt").to(device)
    preprocess_s = time.perf_counter() - started
    prompt_tokens = int(inputs["input_ids"].shape[1])
    torch.cuda.synchronize(device) if torch.cuda.is_available() else None
    decode_started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=int(max_new_tokens), do_sample=False
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - decode_started
    generated = int(output.shape[1] - prompt_tokens)
    reply = processor.decode(output[0, prompt_tokens:], skip_special_tokens=True)
    return {
        "seconds": seconds,
        "preprocess_s": preprocess_s,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated,
        "tokens_per_s": tokens_per_second(generated, seconds),
        "reply": reply.strip(),
        "reply_preview": reply.strip()[:200],
    }


def _replay_frames(replay_dir: Path | None, image_size: Sequence[int]) -> tuple[list, dict]:
    """Replay frames when the set exists; otherwise a neutral gradient frame."""
    from PIL import Image

    if replay_dir is not None and Path(replay_dir).is_dir():
        try:
            from scripts.replay_frames import load_replay
        except ImportError:  # pragma: no cover
            from replay_frames import load_replay  # type: ignore[no-redef]
        replay = load_replay(replay_dir)
        return replay.pil_frames(image_size=image_size), replay.summary()
    width, height = int(image_size[0]), int(image_size[1])
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (x * 255 // max(width - 1, 1), y * 255 // max(height - 1, 1), 96)
    return [image], {"source": "synthetic_gradient", "frame_count": 1}


def run_vlm_profile(args: argparse.Namespace, *, dry_run: bool) -> dict:
    if dry_run:
        return {"status": "skipped", "reason": "dry_run"}
    torch = _torch()
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "cuda_unavailable"}
    image_size = (int(args.image_size[0]), int(args.image_size[1]))
    frames, replay_info = _replay_frames(
        Path(args.replay_dir) if args.replay or args.replay_dir else None, image_size
    )
    system = "You are the perception module of a drone. Answer in one short sentence."
    user = "Name the main object in the image and its approximate position."
    result: dict[str, Any] = {
        "model_path": args.model_path,
        "device": str(device),
        "requested_dtype": args.dtype,
        "requested_quantization": args.quantization,
        "image_size": list(image_size),
        "max_new_tokens": int(args.max_new_tokens),
        "replay": replay_info,
    }

    baseline = reset_peak(device)
    loaded, load_seconds, load_memory = _timed(
        lambda: _load_hf(args.model_path, args.device, args.dtype, args.quantization),
        device,
    )
    result["load"] = {
        "seconds": round(load_seconds, 4),
        "baseline": baseline,
        "after": load_memory,
        "peak_allocated_mib": load_memory.get("peak_allocated_mib"),
        "peak_reserved_mib": load_memory.get("peak_reserved_mib"),
        "available": bool(loaded.get("available")),
        "reason": loaded.get("reason"),
        "parameter_count": loaded.get("parameter_count"),
        "parameter_dtype": loaded.get("parameter_dtype"),
        "quantization": loaded.get("quantization"),
        "quantization_error": loaded.get("quantization_error"),
    }
    if not loaded.get("available"):
        # This is the explicit no-silent-fallback contract: the report says the
        # requested quantization failed instead of presenting a BF16 run as one.
        result["status"] = "unavailable"
        result["reason"] = loaded.get("reason")
        return result
    processor, model = loaded["processor"], loaded["model"]
    result["status"] = "ok"
    result["parameter_count"] = loaded.get("parameter_count")
    result["quantization_estimate"] = quantization_estimate(
        loaded.get("parameter_count") or 0.0,
        overhead_gib=float(args.quantization_overhead_gib),
    )

    def measure(label: str, images: Sequence[Any], repeat: int) -> dict:
        timings: list[float] = []
        calls: list[dict] = []
        peaks: list[dict] = []
        for index in range(max(1, int(repeat))):
            reset_peak(device)
            call, seconds, memory = _timed(
                lambda: _generate_once(
                    processor, model, images, system, user, device,
                    int(args.max_new_tokens), image_size=image_size,
                ),
                device,
            )
            timings.append(call["seconds"])
            call["wall_seconds_including_preprocess"] = round(seconds, 4)
            call["memory"] = memory
            calls.append(call)
            peaks.append(
                {
                    "peak_allocated_mib": memory.get("peak_allocated_mib"),
                    "peak_reserved_mib": memory.get("peak_reserved_mib"),
                    "allocated_after_mib": memory.get("allocated_mib"),
                    "reserved_after_mib": memory.get("reserved_mib"),
                }
            )
        return {
            "timing": summarize_timings(timings),
            "calls": calls,
            "memory": peaks,
            "peak_allocated_mib": max(
                (item.get("peak_allocated_mib") or 0.0) for item in peaks
            ),
            "peak_reserved_mib": max(
                (item.get("peak_reserved_mib") or 0.0) for item in peaks
            ),
            "tokens_per_s_steady": (
                round(
                    statistics.fmean(
                        call["tokens_per_s"] for call in calls[1:] if call["tokens_per_s"]
                    ),
                    3,
                )
                if len(calls) > 1 and any(call["tokens_per_s"] for call in calls[1:])
                else (calls[0]["tokens_per_s"] if calls else None)
            ),
            "prompt_tokens": calls[0]["prompt_tokens"] if calls else None,
            "generated_tokens": calls[0]["generated_tokens"] if calls else None,
        }

    result["single_frame"] = measure("single_frame", [frames[0]], int(args.repeat))
    eight = frames[:8] if len(frames) >= 8 else [frames[index % len(frames)] for index in range(8)]
    result["eight_views"] = measure("eight_views", eight, max(1, int(args.repeat) // 2))

    # Serial calls: does anything accumulate (a persistent KV cache, a growing
    # allocator pool)? Record allocated/reserved after every call, not only peak.
    serial: list[dict] = []
    for index in range(int(args.serial_calls)):
        reset_peak(device)
        call, seconds, memory = _timed(
            lambda: _generate_once(
                processor, model, [frames[index % len(frames)]], system, user, device,
                int(args.max_new_tokens), image_size=image_size,
            ),
            device,
        )
        serial.append(
            {
                "call": index,
                "seconds": round(seconds, 4),
                "generated_tokens": call["generated_tokens"],
                "allocated_mib": memory.get("allocated_mib"),
                "reserved_mib": memory.get("reserved_mib"),
                "peak_allocated_mib": memory.get("peak_allocated_mib"),
                "peak_reserved_mib": memory.get("peak_reserved_mib"),
            }
        )
    if serial:
        first, last = serial[0], serial[-1]
        result["serial_calls"] = {
            "calls": serial,
            "allocated_growth_mib": round(
                (last.get("allocated_mib") or 0.0) - (first.get("allocated_mib") or 0.0), 1
            ),
            "reserved_growth_mib": round(
                (last.get("reserved_mib") or 0.0) - (first.get("reserved_mib") or 0.0), 1
            ),
            "peak_allocated_max_mib": max(item.get("peak_allocated_mib") or 0.0 for item in serial),
            "note": (
                "HF generate allocates a per-call KV cache; growth after the last "
                "call is allocator retention, not a retained cache unless it keeps rising"
            ),
        }
    return result


# ---------------------------------------------------------------------------
# detector section
# ---------------------------------------------------------------------------


def run_detector_profile(args: argparse.Namespace, *, dry_run: bool) -> dict:
    if dry_run:
        return {"status": "skipped", "reason": "dry_run"}
    torch = _torch()
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "cuda_unavailable"}
    from vlm_nav.detectors import Sam3DinoDetector

    detector = Sam3DinoDetector(
        sam3_path=args.sam3_path,
        dino_path=args.dino_path,
        device=args.device,
    )
    result: dict[str, Any] = {
        "sam3_path": args.sam3_path,
        "dino_path": args.dino_path,
        "device": args.device,
        "probe": detector.probe(),
    }
    if not detector.available():
        result["status"] = "unavailable"
        result["reason"] = detector.probe()
        return result
    baseline = reset_peak(device)
    loaded, load_seconds, load_memory = _timed(detector._ensure_loaded, device)
    result["load"] = {
        "seconds": round(load_seconds, 4),
        "available": bool(loaded),
        "baseline": baseline,
        "after": load_memory,
        "peak_allocated_mib": load_memory.get("peak_allocated_mib"),
        "peak_reserved_mib": load_memory.get("peak_reserved_mib"),
    }
    if not loaded:
        result["status"] = "unavailable"
        result["reason"] = detector._load_error
        return result
    result["status"] = "ok"
    model = detector._model
    dino = detector._dino
    parameter = next(model.parameters(), None)
    result["precision"] = {
        "sam3": str(parameter.dtype) if parameter is not None else None,
        "dinov2": str(next(dino.parameters()).dtype) if dino is not None else None,
    }
    frames, replay_info = _replay_frames(
        Path(args.replay_dir) if args.replay or args.replay_dir else None,
        tuple(args.image_size),
    )
    result["replay"] = replay_info
    calls: list[dict] = []
    for index in range(max(1, int(args.detector_calls))):
        frame = frames[index % len(frames)]
        detector._cache.clear()
        detector._feature_cache.clear()
        reset_peak(device)
        detections, seconds, memory = _timed(
            lambda: detector.detect(frame, "窗边的木椅", labels=["wooden chair"]),
            device,
        )
        calls.append(
            {
                "call": index,
                "seconds": round(seconds, 4),
                "detections": len(detections),
                "boxes": [item.to_dict() for item in detections[:5]],
                "memory": memory,
            }
        )
    result["detect"] = {
        "timing": summarize_timings([call["seconds"] for call in calls]),
        "calls": calls,
        "peak_allocated_mib": max(
            (call["memory"].get("peak_allocated_mib") or 0.0) for call in calls
        ),
        "peak_reserved_mib": max(
            (call["memory"].get("peak_reserved_mib") or 0.0) for call in calls
        ),
        "driver_used_max_mib": max(
            (call["memory"].get("driver_used_mib") or 0.0) for call in calls
        ),
        "proposals": calls[0]["detections"] if calls else 0,
    }
    detector._feature_cache.clear()
    reset_peak(device)
    embeddings, seconds, memory = _timed(
        lambda: detector.embed_boxes(frames[0], [[400.0, 350.0, 650.0, 750.0]]),
        device,
    )
    vectors = [] if embeddings is None else [len(vector) for vector in embeddings]
    result["embed_boxes"] = {
        "seconds": round(seconds, 4),
        "available": embeddings is not None,
        "vectors": vectors,
        "memory": memory,
        "peak_allocated_mib": memory.get("peak_allocated_mib"),
        "peak_reserved_mib": memory.get("peak_reserved_mib"),
    }
    return result


# ---------------------------------------------------------------------------
# renderer section
# ---------------------------------------------------------------------------


def _scene_pose_batch(world: Any, n_views: int) -> tuple[Any, Any]:
    import torch

    from vlm_nav.world import yaw_pitch_quat_xyzw

    device = world._higs_device
    phase = torch.arange(n_views, dtype=torch.float32, device=device) / max(n_views, 1)
    angle = phase * 2.0 * torch.pi
    radius = 3.0 + 1.5 * torch.remainder(torch.arange(n_views, device=device), 5) / 4.0
    pos = torch.stack(
        [radius * torch.cos(angle), radius * torch.sin(angle),
         torch.full_like(angle, 0.75) + 0.4 * torch.sin(angle * 3.0)],
        dim=-1,
    )
    yaw = torch.atan2(-pos[:, 1], -pos[:, 0])
    quat = torch.stack(
        [
            torch.as_tensor(
                yaw_pitch_quat_xyzw(float(item), 0.0),
                dtype=torch.float32,
                device=device,
            )
            for item in yaw.cpu()
        ]
    )
    return pos, quat


def _render_one_scene(
    *,
    name: str,
    ply: str,
    scene_scale: float,
    device: str,
    repeat: int,
    batch_views: int,
) -> dict:
    import torch

    from vlm_nav.world import SplatWorldHiGS, TargetSpec

    if not Path(ply).is_file():
        return {"status": "skipped", "reason": f"scene PLY does not exist: {ply}"}
    dev = torch.device(device)
    result: dict[str, Any] = {"name": name, "ply": ply, "scene_scale": scene_scale}
    baseline = reset_peak(dev)
    try:
        world, load_seconds, load_memory = _timed(
            lambda: SplatWorldHiGS(
                ply_path=ply,
                target=TargetSpec(name="target", position_flu=(0.0, 0.0, 0.0), radius_m=0.5),
                start_flu=(0.0, 0.0, 1.0),
                start_yaw_deg=0.0,
                scene_scale=scene_scale,
                z_min=-10.0,
                z_max=10.0,
                device=device,
            ),
            dev,
        )
    except Exception as exc:  # noqa: BLE001 - a broken extension is a finding
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    result["load"] = {
        "seconds": round(load_seconds, 4),
        "baseline": baseline,
        "after": load_memory,
        "peak_allocated_mib": load_memory.get("peak_allocated_mib"),
        "peak_reserved_mib": load_memory.get("peak_reserved_mib"),
    }
    result["n_gaussians"] = int(world.scene.xyz.shape[0])
    result["camera"] = {"width": world.width, "height": world.height, "hfov_deg": world.hfov_deg}
    try:
        single_seconds: list[float] = []
        single_peaks: list[dict] = []
        for _index in range(max(1, int(repeat))):
            reset_peak(dev)
            _, seconds, memory = _timed(world.render, dev)
            single_seconds.append(seconds)
            single_peaks.append(memory)
        result["single_render"] = {
            "timing": summarize_timings(single_seconds),
            "peak_allocated_mib": max(
                (item.get("peak_allocated_mib") or 0.0) for item in single_peaks
            ),
            "peak_reserved_mib": max(
                (item.get("peak_reserved_mib") or 0.0) for item in single_peaks
            ),
        }

        pos, quat = _scene_pose_batch(world, int(batch_views))

        def batch() -> Any:
            world._camera.update_transformation_batch(pos, quat, world.scene.center)
            return world.renderer.render_batch(world.scene, world._camera)["render"]

        batch_seconds: list[float] = []
        batch_peaks: list[dict] = []
        for _index in range(max(1, int(repeat))):
            reset_peak(dev)
            output, seconds, memory = _timed(batch, dev)
            batch_seconds.append(seconds)
            batch_peaks.append(memory)
        batch_timing = summarize_timings(batch_seconds)
        result["batch_render"] = {
            "views": int(batch_views),
            "shape": list(output.shape),
            "timing": batch_timing,
            "per_view_ms": round(
                1000.0 * (
                    batch_timing.get("steady_median_s")
                    or batch_timing.get("median_s")
                    or 0.0
                ) / max(1, int(batch_views)),
                3,
            ),
            "peak_allocated_mib": max(
                (item.get("peak_allocated_mib") or 0.0) for item in batch_peaks
            ),
            "peak_reserved_mib": max(
                (item.get("peak_reserved_mib") or 0.0) for item in batch_peaks
            ),
        }

        scan_seconds: list[float] = []
        scan_peaks: list[dict] = []
        for _index in range(max(1, int(repeat))):
            reset_peak(dev)
            views, seconds, memory = _timed(
                lambda: world.scan_views(int(batch_views), pitch_down_deg=0.0), dev
            )
            scan_seconds.append(seconds)
            scan_peaks.append(memory)
        scan_timing = summarize_timings(scan_seconds)
        result["scan_views"] = {
            "views": len(views),
            "timing": scan_timing,
            "per_view_ms": round(
                1000.0 * (
                    scan_timing.get("steady_median_s")
                    or scan_timing.get("median_s")
                    or 0.0
                ) / max(1, len(views)),
                3,
            ),
            "peak_allocated_mib": max(
                (item.get("peak_allocated_mib") or 0.0) for item in scan_peaks
            ),
            "peak_reserved_mib": max(
                (item.get("peak_reserved_mib") or 0.0) for item in scan_peaks
            ),
        }
        result["status"] = "ok"
    finally:
        close = getattr(world, "close", None)
        if callable(close):
            close()
    return result


def run_render_profile(args: argparse.Namespace, *, dry_run: bool) -> dict:
    if dry_run:
        return {"status": "skipped", "reason": "dry_run"}
    torch = _torch()
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "cuda_unavailable"}
    scenes = {
        "bedroom": (args.bedroom_ply, float(args.bedroom_scale)),
        "classroom": (args.classroom_ply, float(args.classroom_scale)),
    }
    return {
        "status": "ok",
        "device": args.device,
        "scenes": {
            name: _render_one_scene(
                name=name,
                ply=ply,
                scene_scale=scale,
                device=args.device,
                repeat=int(args.render_repeat),
                batch_views=int(args.render_views),
            )
            for name, (ply, scale) in scenes.items()
        },
    }


# ---------------------------------------------------------------------------
# confirm section
# ---------------------------------------------------------------------------


def _read_holdout_target(manifest_path: Path, target_index: int) -> tuple[str, list[float]]:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    positions = data.get("object_positions_flu") or data.get("entity_positions_flu")
    if not positions:
        raise ValueError(f"no object_positions_flu in {manifest_path}")
    index = int(target_index)
    name = "target"
    instances = data.get("object_instances") or []
    if index < len(instances):
        name = str(instances[index].get("name") or name)
    return name, [float(v) for v in positions[index]]


def _ensure_holdout_ply(args: argparse.Namespace) -> Path:
    ply = Path(args.confirm_ply) if args.confirm_ply else (
        REPO / "outputs" / "vlm_nav" / "replay" / "window_chair" / "scene" / "combo.ply"
    )
    if ply.is_file():
        return ply
    try:
        from scripts.build_replay_window_chair import ensure_scene_ply
    except ImportError:  # pragma: no cover
        from build_replay_window_chair import ensure_scene_ply  # type: ignore[no-redef]
    spz = Path(args.confirm_spz)
    if not spz.is_file():
        raise FileNotFoundError(f"confirm scene SPZ does not exist: {spz}")
    return ensure_scene_ply(spz, ply)


def _pixel_of(target: Sequence[float], pose: Sequence[float], yaw_deg: float,
              width: int, height: int, hfov_deg: float) -> list[float] | None:
    import numpy as np

    from vlm_nav.skills_approach import world_to_pixel

    yaw = math.radians(float(yaw_deg))
    pose = np.asarray(pose, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    delta = target - pose
    forward = delta[0] * math.cos(yaw) + delta[1] * math.sin(yaw)
    if forward <= 0.05:
        return None
    pixel = world_to_pixel(
        target, pose, yaw,
        {"width": int(width), "height": int(height), "hfov_deg": float(hfov_deg)},
    )
    return None if pixel is None else [float(pixel[0]), float(pixel[1])]


def run_confirm_profile(args: argparse.Namespace, *, dry_run: bool) -> dict:
    if dry_run:
        return {"status": "skipped", "reason": "dry_run"}
    import numpy as np

    torch = _torch()
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "cuda_unavailable"}
    try:
        ply = _ensure_holdout_ply(args)
    except Exception as exc:  # noqa: BLE001
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    target_name, target = _read_holdout_target(Path(args.confirm_manifest), args.target_index)
    from vlm_nav.tool_impl_navigation import register_navigation_tools
    from vlm_nav.tool_impl_perception import register_perception_tools
    from vlm_nav.tools import ToolContext, ToolRegistry
    from vlm_nav.world import SplatWorldHiGS, TargetSpec

    result: dict[str, Any] = {
        "scene": str(ply),
        "target_name": target_name,
        "target_position_flu": target,
        "device": args.device,
    }
    load_baseline = reset_peak(device)
    world, world_seconds, world_memory = _timed(
        lambda: SplatWorldHiGS(
            ply_path=ply,
            target=TargetSpec(name=target_name, position_flu=tuple(target), radius_m=0.37),
            start_flu=tuple(target),
            start_yaw_deg=0.0,
            scene_scale=1.0,
            z_min=target[2] - 4.0,
            z_max=target[2] + 4.0,
            device=args.device,
        ),
        device,
    )
    result["renderer_load"] = {
        "seconds": round(world_seconds, 4),
        "peak_allocated_mib": world_memory.get("peak_allocated_mib"),
        "peak_reserved_mib": world_memory.get("peak_reserved_mib"),
        "baseline": load_baseline,
        "n_gaussians": int(world.scene.xyz.shape[0]),
    }

    # The confirm flow needs a backend; the real VLM is loaded here so the
    # measured peak includes model + renderer + depth mirror in one process.
    vlm_load_baseline = reset_peak(device)
    loaded, vlm_load_seconds, vlm_load_memory = _timed(
        lambda: _load_hf(args.model_path, args.device, args.dtype, "none"), device
    )
    result["vlm_load"] = {
        "seconds": round(vlm_load_seconds, 4),
        "peak_allocated_mib": vlm_load_memory.get("peak_allocated_mib"),
        "peak_reserved_mib": vlm_load_memory.get("peak_reserved_mib"),
        "baseline": vlm_load_baseline,
        "available": bool(loaded.get("available")),
        "reason": loaded.get("reason"),
        "parameter_count": loaded.get("parameter_count"),
    }
    if not loaded.get("available"):
        result["status"] = "unavailable"
        result["reason"] = loaded.get("reason")
        close = getattr(world, "close", None)
        if callable(close):
            close()
        return result
    processor, model = loaded["processor"], loaded["model"]

    class _Backend:
        name = "hf-profile"

        def decide(self, frames, system_prompt, user_prompt, context=None):
            reply = _generate_once(
                processor, model, frames, system_prompt, user_prompt, device,
                int(args.confirm_max_new_tokens), image_size=tuple(args.image_size),
            )
            return reply["reply"]

    backend = _Backend()
    registry = ToolRegistry()
    register_perception_tools(registry, vlm=backend)
    register_navigation_tools(registry, vlm=backend)
    ctx = ToolContext(
        world=world,
        budget={"tool_calls_left": int(args.confirm_tool_budget)},
        extra={
            "instruction": "在卧室窗边找到木椅",
            "done_dist": 3.0,
            "perception.vlm": backend,
        },
    )
    # stand just outside the confirmation shell (1.2 m standoff + 0.15 m)
    distance = 1.30
    pose = (target[0] + distance, target[1], target[2] + 0.15)
    yaw_deg = 180.0
    world.state.position_flu = np.asarray(pose, dtype=np.float64)
    world.state.yaw_rad = math.radians(yaw_deg)
    observed = registry.call("observe", {"mode": "forward"}, ctx)
    obs_id = str(observed.data.get("observation_id") or "")
    pixel = _pixel_of(target, pose, yaw_deg, world.width, world.height, world.hfov_deg)
    if pixel is None:
        pixel = [500.0, 500.0]
    else:
        pixel = [max(0.0, min(1000.0, pixel[0])), max(0.0, min(1000.0, pixel[1]))]
    half_w, half_h = 55.0, 90.0
    box = [
        max(0.0, pixel[0] - half_w), max(0.0, pixel[1] - half_h),
        min(1000.0, pixel[0] + half_w), min(1000.0, pixel[1] + half_h),
    ]
    range_m = float(math.dist(pose, target))
    ctx.extra.setdefault("perception.candidates", {})["cand_1"] = {
        "id": "cand_1",
        "source": "profile_seed",
        "box": box,
        "box_format": "xyxy",
        "attrs": {"material": "wood", "shape": "chair"},
        "confidence": 0.9,
        "observation_id": obs_id,
        "instruction": "卧室窗边的木椅",
        "position_source": "depth",
        "position_flu": list(target),
        "range_m": range_m,
        "range_status": "known",
        "ray": {
            "origin_flu": list(pose),
            "bearing_deg": yaw_deg,
            "elevation_deg": 0.0,
            "yaw_deg": yaw_deg,
        },
    }
    result["seeded_candidate"] = {
        "candidate_id": "cand_1",
        "source_observation": obs_id,
        "box": box,
        "range_m": round(range_m, 3),
        "position_flu": list(target),
        "note": "profiling seed from the holdout manifest; the flow still judges it with the real model",
    }
    # The production env exposes depth through a CPU mirror of the activated
    # Gaussians (HiGS renders RGB only). Building it is a one-time cost; time it
    # separately so the confirmation number is not hiding a minute of setup.
    try:
        depth_baseline = reset_peak(device)
        _, depth_seconds, depth_memory = _timed(world._depth_renderer, device)
        result["depth_backend_load"] = {
            "seconds": round(depth_seconds, 4),
            "peak_allocated_mib": depth_memory.get("peak_allocated_mib"),
            "peak_reserved_mib": depth_memory.get("peak_reserved_mib"),
            "baseline": depth_baseline,
            "backend": "cpu_splat_mirror",
        }
    except Exception as exc:  # noqa: BLE001 - a missing mirror is a finding
        result["depth_backend_load"] = {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    reset_peak(device)
    confirmation, seconds, memory = _timed(
        lambda: registry.call(
            "confirm_candidate",
            {
                "candidate_id": "cand_1",
                "instruction": "卧室窗边的木椅",
                "max_views": int(args.confirm_views),
                "max_steps": int(args.confirm_steps),
                "turn_deg": 35.0,
            },
            ctx,
        ),
        device,
    )
    data = confirmation.data if isinstance(confirmation.data, dict) else {}
    result["confirm_candidate"] = {
        "seconds": round(seconds, 4),
        "status": confirmation.status,
        "verdict": data.get("verdict"),
        "missing": data.get("missing"),
        "reason": data.get("reason"),
        "vlm_calls": data.get("vlm_calls"),
        "observation_ids": data.get("observation_ids"),
        "position_check": data.get("position_check"),
        "views": data.get("views"),
        "steps": data.get("steps"),
        "peak_allocated_mib": memory.get("peak_allocated_mib"),
        "peak_reserved_mib": memory.get("peak_reserved_mib"),
        "allocated_after_mib": memory.get("allocated_mib"),
        "memory": memory,
    }
    # Explicit pitched-depth measurement: the candidate is placed above the
    # level frame so only a pitched camera can measure it (the v8 seed7 case).
    from vlm_nav import skills_approach

    pitched_target = tuple([target[0], target[1], target[2] + 1.15])
    record = dict(ctx.extra["perception.candidates"]["cand_1"])
    record["position_flu"] = list(pitched_target)
    record["range_m"] = float(math.dist(pose, pitched_target))
    target_obj = skills_approach.candidate_target(record, "cand_1")
    reset_peak(device)
    measurement, pitched_seconds, pitched_memory = _timed(
        lambda: skills_approach.relocalize(ctx, target_obj, obs_id, allow_pitched=True),
        device,
    )
    result["pitched_depth"] = {
        "seconds": round(pitched_seconds, 4),
        "relocalized": measurement.get("relocalized"),
        "reason": measurement.get("reason"),
        "anchor": measurement.get("anchor"),
        "pitch_down_deg": measurement.get("pitch_down_deg"),
        "elevation_deg": measurement.get("elevation_deg"),
        "position_flu": measurement.get("position_flu"),
        "memory": pitched_memory,
        "peak_allocated_mib": pitched_memory.get("peak_allocated_mib"),
        "peak_reserved_mib": pitched_memory.get("peak_reserved_mib"),
    }
    result["status"] = "ok"
    close = getattr(world, "close", None)
    if callable(close):
        close()
    return result


# ---------------------------------------------------------------------------
# replay + audit sections
# ---------------------------------------------------------------------------


def run_replay_section(args: argparse.Namespace, *, dry_run: bool) -> dict:
    replay_dir = Path(args.replay_dir)
    if not (replay_dir / "manifest.json").is_file():
        return {"status": "unavailable", "reason": f"no replay manifest at {replay_dir}"}
    try:
        from scripts.replay_frames import load_replay
    except ImportError:  # pragma: no cover
        from replay_frames import load_replay  # type: ignore[no-redef]
    replay = load_replay(replay_dir)
    summary = replay.summary()
    summary["path"] = str(replay_dir)
    summary["status"] = "ok"
    if not dry_run:
        import torch

        if torch.cuda.is_available():
            device = torch.device(args.device)
            reset_peak(device)
            _, seconds, memory = _timed(lambda: replay.frames(), device)
            summary["stack_seconds"] = round(seconds, 4)
            summary["stack_memory"] = memory
    return summary


def run_audit_section(args: argparse.Namespace, *, dry_run: bool) -> dict:
    roots = list(args.audit_root or [])
    if not roots:
        return {"status": "skipped", "reason": "no --audit-root given"}
    return audit_summary_paths(roots, max_summaries=int(args.audit_limit))


# ---------------------------------------------------------------------------
# report writer
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def _vlm_markdown(data: Mapping[str, Any]) -> list[str]:
    lines = ["## VLM (Qwen3-VL-8B)"]
    if data.get("status") not in (None, "ok"):
        lines.append(f"Unavailable: {data.get('reason')}")
        return lines
    load = data.get("load") or {}
    lines.append(
        _table(
            ["phase", "seconds", "peak alloc MiB", "peak reserved MiB", "notes"],
            [
                ["load", load.get("seconds"), load.get("peak_allocated_mib"),
                 load.get("peak_reserved_mib"),
                 f"params={data.get('parameter_count')} dtype={load.get('parameter_dtype')} "
                 f"quant={load.get('quantization')}"],
            ],
        )
    )
    rows = []
    for key, label in (("single_frame", "single frame"), ("eight_views", "eight views")):
        item = data.get(key) or {}
        timing = item.get("timing") or {}
        rows.append(
            [label, timing.get("first_s"), timing.get("steady_median_s"),
             timing.get("p95_s"), item.get("tokens_per_s_steady"),
             item.get("peak_allocated_mib"), item.get("peak_reserved_mib")]
        )
    lines.append("")
    lines.append(
        _table(
            ["phase", "first s", "steady median s", "p95 s", "tokens/s",
             "peak alloc MiB", "peak reserved MiB"],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "Peak allocated and peak reserved are `torch.cuda.max_memory_allocated` "
        "and `max_memory_reserved` for the phase; the driver-side peak in "
        "`report.json` includes the CUDA context."
    )
    serial = data.get("serial_calls") or {}
    if serial:
        lines.append("")
        lines.append(
            f"Serial calls: {len(serial.get('calls') or [])}; allocated growth "
            f"{_fmt(serial.get('allocated_growth_mib'))} MiB, reserved growth "
            f"{_fmt(serial.get('reserved_growth_mib'))} MiB, peak "
            f"{_fmt(serial.get('peak_allocated_max_mib'))} MiB. "
            f"{serial.get('note', '')}"
        )
    if data.get("quantization_estimate"):
        estimate = data["quantization_estimate"]
        lines.append("")
        lines.append(
            "Quantization conversion (not a 4090 measurement): "
            f"BF16 {_fmt(estimate['bf16_weights_gib'], 3)} GiB weights / "
            f"{_fmt(estimate['bf16_with_overhead_gib'], 3)} GiB with "
            f"{_fmt(estimate['overhead_gib'], 1)} GiB overhead; "
            f"INT8 {_fmt(estimate['int8_with_overhead_gib'], 3)} GiB; "
            f"INT4 {_fmt(estimate['int4_with_overhead_gib'], 3)} GiB. "
            f"{estimate['note']}."
        )
    return lines


def _detector_markdown(data: Mapping[str, Any]) -> list[str]:
    lines = ["## Detector (SAM3 + DINOv2)"]
    if data.get("status") != "ok":
        lines.append(f"Unavailable: {data.get('reason')}")
        return lines
    load = data.get("load") or {}
    detect = data.get("detect") or {}
    embed = data.get("embed_boxes") or {}
    timing = detect.get("timing") or {}
    lines.append(
        _table(
            ["phase", "seconds", "peak alloc MiB", "peak reserved MiB", "notes"],
            [
                ["load", load.get("seconds"), load.get("peak_allocated_mib"),
                 load.get("peak_reserved_mib"),
                 f"precision={data.get('precision')}"],
                ["detect", timing.get("steady_median_s") or timing.get("first_s"),
                 detect.get("peak_allocated_mib"), detect.get("peak_reserved_mib"),
                 f"proposals={detect.get('proposals')}"],
                ["embed_boxes", embed.get("seconds"), embed.get("peak_allocated_mib"),
                 embed.get("peak_reserved_mib"), f"vectors={embed.get('vectors')}"],
            ],
        )
    )
    return lines


def _render_markdown(data: Mapping[str, Any]) -> list[str]:
    lines = ["## HiGS renderer"]
    if data.get("status") != "ok":
        lines.append(f"Unavailable: {data.get('reason')}")
        return lines
    rows = []
    for name, scene in (data.get("scenes") or {}).items():
        if scene.get("status") != "ok":
            rows.append([name, "-", "-", "-", "-", "-", "-", scene.get("reason")])
            continue
        load = scene.get("load") or {}
        single = scene.get("single_render") or {}
        batch = scene.get("batch_render") or {}
        scan = scene.get("scan_views") or {}
        rows.append(
            [name, scene.get("n_gaussians"), load.get("seconds"),
             1000.0 * ((single.get("timing") or {}).get("steady_median_s") or 0.0),
             1000.0 * ((batch.get("timing") or {}).get("steady_median_s") or 0.0),
             1000.0 * ((scan.get("timing") or {}).get("steady_median_s") or 0.0),
             single.get("peak_allocated_mib"), batch.get("peak_allocated_mib")]
        )
    lines.append(
        _table(
            ["scene", "Gaussians", "load s", "1 render ms", "8-view batch ms",
             "8 scan_views ms", "1-render peak MiB", "batch peak MiB"],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "The batch call is the native `render_batch` GPU path; `scan_views` is "
        "eight production `world.render()` calls including the CPU copy. The "
        "first scene in a process also pays CUDA/extension warmup, so its load "
        "seconds are not pure I/O."
    )
    return lines


def _confirm_markdown(data: Mapping[str, Any]) -> list[str]:
    lines = ["## confirm_candidate flow"]
    if data.get("status") != "ok":
        lines.append(f"Unavailable: {data.get('reason')}")
        return lines
    confirm = data.get("confirm_candidate") or {}
    pitched = data.get("pitched_depth") or {}
    vlm_load = data.get("vlm_load") or {}
    depth_load = data.get("depth_backend_load") or {}
    lines.append(
        _table(
            ["phase", "seconds", "peak alloc MiB", "peak reserved MiB", "result"],
            [
                ["VLM load", vlm_load.get("seconds"),
                 vlm_load.get("peak_allocated_mib"), vlm_load.get("peak_reserved_mib"),
                 f"params={vlm_load.get('parameter_count')}"],
                ["depth mirror build", depth_load.get("seconds"),
                 depth_load.get("peak_allocated_mib"), depth_load.get("peak_reserved_mib"),
                 depth_load.get("backend") or depth_load.get("reason")],
                ["confirm_candidate", confirm.get("seconds"),
                 confirm.get("peak_allocated_mib"), confirm.get("peak_reserved_mib"),
                 f"verdict={confirm.get('verdict')} vlm_calls={confirm.get('vlm_calls')} "
                 f"missing={confirm.get('missing')}"],
                ["pitched depth", pitched.get("seconds"),
                 pitched.get("peak_allocated_mib"), pitched.get("peak_reserved_mib"),
                 f"relocalized={pitched.get('relocalized')} anchor={pitched.get('anchor')} "
                 f"pitch={pitched.get('pitch_down_deg')}"],
            ],
        )
    )
    lines.append("")
    lines.append(
        "The flow seeds `cand_1` from the holdout manifest (profiling only) and "
        "then runs the real routine: approach/standoff, fresh views, one VLM "
        "judgement per view, box-depth measurement, verdict. The pitched row is "
        "a separate nose-up measurement of a target 1.15 m above the level frame."
    )
    return lines


def _replay_markdown(data: Mapping[str, Any]) -> list[str]:
    lines = ["## Frozen observation replay"]
    if data.get("status") != "ok":
        lines.append(f"Unavailable: {data.get('reason')}")
        return lines
    lines.append(
        f"`{data.get('path')}` - {data.get('frame_count')} views, phases "
        f"{data.get('phases')}, camera {data.get('camera')}, manifest sha256 "
        f"`{str(data.get('manifest_sha256'))[:16]}...`."
    )
    lines.append("")
    if data.get("benchmark"):
        lines.append(f"Benchmark metadata: {data.get('benchmark')}.")
        lines.append("")
    lines.append(
        "Load it with `scripts.replay_frames.load_replay(path)`; "
        "`ReplaySet.frames()` returns `[N, 3, H, W]` and `pil_frames()` the RGB "
        "list a VLM backend takes. The same manifest is the comparable input for "
        "A100/4090 runs and harness variants."
    )
    return lines


def _audit_markdown(data: Mapping[str, Any]) -> list[str]:
    lines = ["## OOM / silent-fallback audit"]
    if data.get("status") == "skipped":
        lines.append(f"Skipped: {data.get('reason')}")
        return lines
    lines.append(
        f"Scanned {data.get('summaries')} summary.json files: "
        f"{data.get('complete')} complete, {len(data.get('incomplete') or [])} missing "
        "at least one required field."
    )
    counts = data.get("missing_field_counts") or {}
    if counts:
        lines.append("")
        lines.append(
            _table(
                ["required summary field", "missing in # summaries"],
                [[field, count] for field, count in counts.items()],
            )
        )
    lines.append("")
    lines.append("### Minimal per-episode fields")
    lines.append(
        _table(
            ["field", "why it is required"],
            [[item["field"], item["why"]] for item in data.get("required_fields", [])],
        )
    )
    lines.append("")
    lines.append(
        "An episode summary without these fields cannot distinguish a healthy "
        "run from a silent fallback (mock backend, detector off, no depth, CPU "
        "renderer, or an OOM retry). The audit reports the gap, it never fills "
        "it with a guessed value."
    )
    lines.append("")
    lines.append("### Silent-fallback checkpoints")
    lines.append(
        _table(
            ["checkpoint", "what can degrade silently"],
            [[name, detail] for name, detail in FALLBACK_CHECKPOINTS],
        )
    )
    return lines


def _adaptation_markdown(results: Mapping[str, Any]) -> list[str]:
    lines = ["## 4090 adaptation inventory"]
    lines.append(
        _table(
            ["area", "location", "finding", "action"],
            [list(item) for item in ADAPTATION_NOTES],
        )
    )
    vlm = results.get("vlm") or {}
    estimate = vlm.get("quantization_estimate")
    if estimate:
        lines.append("")
        lines.append(
            "Measured A100 BF16 peak (this report) is the anchor; the 8-bit/4-bit "
            "figures below are conversions, not 4090 runs:"
        )
        lines.append("")
        lines.append(
            _table(
                ["precision", "weights GiB", "with overhead GiB", "source"],
                [
                    ["BF16", estimate["bf16_weights_gib"], estimate["bf16_with_overhead_gib"],
                     "parameters x 2 bytes"],
                    ["INT8", estimate["int8_weights_gib"], estimate["int8_with_overhead_gib"],
                     "parameters x 1 byte (bitsandbytes)"],
                    ["INT4", estimate["int4_weights_gib"], estimate["int4_with_overhead_gib"],
                     "parameters x 0.5 byte (NF4)"],
                ],
            )
        )
        lines.append("")
        lines.append(
            "Recommendation: try BF16 first on a 24 GB 4090 and watch the measured "
            "peak; if the VLM plus the 1.2M-Gaussian renderer and the FP32 "
            "detector do not fit, move the VLM to 8-bit before 4-bit and re-run "
            "the replay set on both hosts. 4-bit is the fallback for a 16 GB "
            "card, not the first choice: it changes numeric behaviour and must "
            "be validated against the same replay before it is compared."
        )
    return lines


def render_report_markdown(results: Mapping[str, Any], *, title: str) -> str:
    lines = [f"# {title}", ""]
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    meta = results.get("meta") or {}
    if meta:
        lines.append("")
        lines.append("## Environment")
        lines.append(
            _table(
                ["key", "value"],
                [[key, value] for key, value in meta.items()],
            )
        )
    isolated = results.get("sections_isolated")
    if isolated:
        lines.append("")
        lines.append("## Section provenance")
        lines.append(
            _table(
                ["section", "source report"],
                [[key, value] for key, value in isolated.items()],
            )
        )
    for key, renderer in (
        ("vlm", _vlm_markdown),
        ("detector", _detector_markdown),
        ("render", _render_markdown),
        ("confirm", _confirm_markdown),
        ("replay", _replay_markdown),
        ("audit", _audit_markdown),
    ):
        if key in results:
            lines.append("")
            lines.extend(renderer(results[key]))
    lines.append("")
    lines.extend(_adaptation_markdown(results))
    lines.append("")
    lines.append("## Caveats")
    lines.append(
        "- All numbers in this report are A100 measurements unless a section says "
        "otherwise; no 4090 was available, so the quantization table is a "
        "conversion and is labelled as such."
    )
    lines.append(
        "- `torch.cuda.max_memory_allocated/reserved` exclude the CUDA context; "
        "`peak_driver_used_mib` (from `mem_get_info`) includes it. Use the driver "
        "number for a 24 GB fit decision and the allocator number for leaks."
    )
    lines.append(
        "- Latency is wall-clock on this host with `CUDA_VISIBLE_DEVICES` pinned; "
        "only replay-based runs are comparable across hosts."
    )
    lines.append(
        "- Per-component peaks are only comparable across runs when the section "
        "runs in its own process: SAM3 is a process-wide singleton and a loaded "
        "VLM keeps its weights. `report.json.sections_isolated` records which "
        "section report each number came from (use `--sections render` / "
        "`--sections confirm` separately and `--merge`)."
    )
    lines.append("")
    return "\n".join(lines)


def merge_reports(paths: Iterable[str | Path]) -> dict:
    """Combine section reports from isolated processes.

    Running every GPU section in one process contaminates the per-component
    peaks: the SAM3 detector is a process-wide singleton and a live VLM keeps
    its weights, so a later render/confirm peak includes memory that belongs to
    an earlier section. The report is therefore assembled from one process per
    GPU section (``--sections vlm``, ``--sections render``, ...) and the
    ``sections_isolated`` map records where each number came from.
    """
    merged: dict[str, Any] = {"sections_isolated": {}}
    meta: dict[str, Any] | None = None
    for raw in paths:
        path = Path(raw).expanduser()
        data = json.loads(path.read_text(encoding="utf-8"))
        results = data.get("results", data)
        if not isinstance(results, Mapping):
            continue
        if meta is None and isinstance(results.get("meta"), Mapping):
            meta = dict(results["meta"])
        for key, value in results.items():
            if key in ("meta", "requested_sections", "sections_isolated"):
                continue
            merged[key] = value
            merged["sections_isolated"][key] = str(path)
    if meta is not None:
        merged["meta"] = meta
    return merged


def write_report(
    out_dir: str | Path,
    results: Mapping[str, Any],
    *,
    title: str = "VLM-navigation resource profile",
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": REPORT_SCHEMA,
        "title": title,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": dict(results),
    }
    json_path = out / "report.json"
    markdown_path = out / "report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_report_markdown(results, title=title), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sections", type=str, default="all",
                        help="all | comma list of vlm,detector,render,confirm,replay,audit")
    parser.add_argument("--merge", type=Path, nargs="+", default=None,
                        help="merge section report.json files from isolated runs "
                             "instead of measuring (see merge_reports)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--title", type=str, default="VLM-navigation resource profile")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--quantization", type=str, default="none",
                        choices=["none", "8bit", "4bit"])
    parser.add_argument("--quantization-overhead-gib", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, nargs=2, default=(640, 360),
                        metavar=("W", "H"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--serial-calls", type=int, default=10)
    parser.add_argument("--replay", action="store_true",
                        help="use the frozen replay set for the VLM/detector frames")
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--sam3-path", type=str, default=DEFAULT_SAM3)
    parser.add_argument("--dino-path", type=str, default=DEFAULT_DINO)
    parser.add_argument("--detector-calls", type=int, default=3)
    parser.add_argument("--bedroom-ply", type=str, default=DEFAULT_BEDROOM_PLY)
    parser.add_argument("--classroom-ply", type=str, default=DEFAULT_CLASSROOM_PLY)
    parser.add_argument("--bedroom-scale", type=float, default=6.0)
    parser.add_argument("--classroom-scale", type=float, default=8.0)
    parser.add_argument("--render-views", type=int, default=8)
    parser.add_argument("--render-repeat", type=int, default=5)
    parser.add_argument("--confirm-ply", type=str, default=None)
    parser.add_argument("--confirm-spz", type=Path, default=DEFAULT_HOLDOUT_SPZ)
    parser.add_argument("--confirm-manifest", type=Path, default=DEFAULT_HOLDOUT_JSON)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--confirm-views", type=int, default=2)
    parser.add_argument("--confirm-steps", type=int, default=6)
    parser.add_argument("--confirm-tool-budget", type=int, default=12)
    parser.add_argument("--confirm-max-new-tokens", type=int, default=192,
                        help="generation cap for the confirmation JSON (the VLM "
                             "section uses --max-new-tokens)")
    parser.add_argument("--audit-root", type=Path, action="append", default=None)
    parser.add_argument("--audit-limit", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true",
                        help="skip GPU sections; still write replay/audit sections")
    return parser


def _requested_sections(value: str) -> set[str]:
    known = {"vlm", "detector", "render", "confirm", "replay", "audit"}
    if str(value).strip().lower() == "all":
        return set(known)
    requested = {item.strip().lower() for item in str(value).split(",") if item.strip()}
    unknown = requested - known
    if unknown:
        raise SystemExit(f"unknown sections: {sorted(unknown)}; choose from {sorted(known)}")
    return requested


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.merge:
        results = merge_reports(args.merge)
        written = write_report(args.out, results, title=args.title)
        print(json.dumps(written, indent=2))
        return 0
    sections = _requested_sections(args.sections)
    results: dict[str, Any] = {"requested_sections": sorted(sections)}
    if not args.dry_run:
        try:
            results["meta"] = _environment_meta(args.device)
        except Exception as exc:  # noqa: BLE001 - a CPU-only dry report is still useful
            results["meta"] = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        results["meta"] = {"dry_run": True}

    runners = {
        "vlm": run_vlm_profile,
        "detector": run_detector_profile,
        "render": run_render_profile,
        "confirm": run_confirm_profile,
        "replay": run_replay_section,
        "audit": run_audit_section,
    }
    for name in ("vlm", "detector", "render", "confirm", "replay", "audit"):
        if name not in sections:
            continue
        print(f"[profile_resources] section {name}", flush=True)
        try:
            results[name] = runners[name](args, dry_run=bool(args.dry_run))
        except Exception as exc:  # noqa: BLE001 - one section must not kill the report
            results[name] = {
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            print(f"[profile_resources] section {name} failed: {exc}", flush=True)
    written = write_report(args.out, results, title=args.title)
    print(json.dumps(written, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
