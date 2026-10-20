"""Hardware profiles and honest GPU-memory fit checks (goal.md M4, 4090 port).

The 4090 question is a budget question: the measured A100 stack peaks are about
17.3 GiB for Qwen3-VL-8B (eight views), 4.2 GiB for the FP32 SAM3+DINOv2
detector and roughly 1.5 GiB for the HiGS renderer. Those are *isolated* peaks
from separate processes. Summing them is not a measurement of the combined
live set, so a fit report must say what it knows and what it does not:

* :func:`snapshot` reads the live GPU through ``torch.cuda`` and returns an
  explicit ``available: false`` snapshot when torch, CUDA or the requested
  device is missing. It never substitutes zeros for GPU metrics.
* :class:`HardwareProfile` states the target envelope (total and the reserve
  kept for the CUDA context / other processes) plus an *expected* mixed peak.
  The expected peak is an informed estimate, not the sum of component peaks.
* :func:`fit_report` compares a component budget (or the expected peak) plus
  the profile reserve against the total, and additionally refuses the run when
  the measured free/reusable memory is below the required peak. ``reasons``
  explains every decision.
* :func:`markdown_table` renders the same report for a human.

The module is import-safe on a CPU-only host: torch is imported lazily inside
:func:`snapshot`/:func:`reset_peak`, never at module import.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

MIB = 1024.0 * 1024.0

#: Every snapshot carries these keys, whatever the device state is.
SNAPSHOT_KEYS = (
    "available",
    "gpu_name",
    "total_mib",
    "free_mib",
    "allocated_mib",
    "reserved_mib",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "driver_used_mib",
)


def _number(value: Any) -> float | None:
    """Finite float or ``None``; booleans are not measurements."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mib(value: Any) -> float | None:
    number = _number(value)
    if number is None:
        return None
    return round(number / MIB, 1)


def _load_torch() -> Any:
    """Import torch lazily so a host without it can still import this module."""
    import torch

    return torch


def _unavailable(device: str, reason: str) -> dict:
    snapshot = {
        "available": False,
        "gpu_name": "",
        "device": device,
        "total_mib": None,
        "free_mib": None,
        "allocated_mib": None,
        "reserved_mib": None,
        "peak_allocated_mib": None,
        "peak_reserved_mib": None,
        "driver_used_mib": None,
        "reason": str(reason or "CUDA is unavailable"),
        "torch_version": "",
        "cuda_version": "",
    }
    return snapshot


def snapshot(device: str = "cuda") -> dict:
    """Read the CUDA memory state without pretending a CPU is a GPU.

    Returns the keys in :data:`SNAPSHOT_KEYS` plus ``device`` and ``reason``.
    When torch, CUDA or the device is missing, ``available`` is false, every
    MiB value is ``None`` (not zero) and ``reason`` says why.
    """
    text = str(device or "").strip() or "cuda"
    lowered = text.lower()
    if lowered == "cpu" or lowered.startswith("cpu:"):
        return _unavailable(text, "device is CPU; no CUDA memory metrics")
    try:
        torch = _load_torch()
    except Exception as exc:  # noqa: BLE001 - a missing/broken torch is a finding
        return _unavailable(
            text, f"torch is unavailable: {type(exc).__name__}: {exc}"
        )
    if torch is None:
        return _unavailable(text, "torch is not installed")
    try:
        cuda = torch.cuda
        if not bool(cuda.is_available()):
            return _unavailable(text, "CUDA is not available to torch")
        try:
            cuda.synchronize(text)
        except Exception:  # noqa: BLE001 - sync is best effort
            pass
        free_bytes, total_bytes = cuda.mem_get_info(text)
        total_mib = _mib(total_bytes)
        free_mib = _mib(free_bytes)
        if total_mib is None or free_mib is None:
            return _unavailable(text, "CUDA returned no device memory information")
        allocated_mib = _mib(cuda.memory_allocated(text))
        reserved_mib = _mib(cuda.memory_reserved(text))
        peak_allocated_mib = _mib(cuda.max_memory_allocated(text))
        peak_reserved_mib = _mib(cuda.max_memory_reserved(text))
        gpu_name = str(cuda.get_device_name(text) or "")
        version = getattr(torch, "version", None)
        return {
            "available": True,
            "gpu_name": gpu_name,
            "device": text,
            "total_mib": total_mib,
            "free_mib": free_mib,
            "allocated_mib": allocated_mib,
            "reserved_mib": reserved_mib,
            "peak_allocated_mib": peak_allocated_mib,
            "peak_reserved_mib": peak_reserved_mib,
            "driver_used_mib": round(total_mib - free_mib, 1),
            "reason": "",
            "torch_version": str(getattr(torch, "__version__", "") or ""),
            "cuda_version": str(getattr(version, "cuda", "") or ""),
        }
    except Exception as exc:  # noqa: BLE001 - an unreadable GPU is a finding
        return _unavailable(
            text, f"CUDA memory query failed: {type(exc).__name__}: {exc}"
        )


def reset_peak(device: str = "cuda") -> None:
    """Reset the CUDA peak counters for ``device``; a no-op without CUDA."""
    text = str(device or "").strip() or "cuda"
    lowered = text.lower()
    if lowered == "cpu" or lowered.startswith("cpu:"):
        return
    try:
        torch = _load_torch()
    except Exception:  # noqa: BLE001 - reset is best effort
        return
    if torch is None:
        return
    try:
        if not bool(torch.cuda.is_available()):
            return
        try:
            torch.cuda.synchronize(text)
        except Exception:  # noqa: BLE001
            pass
        torch.cuda.reset_peak_memory_stats(text)
    except Exception:  # noqa: BLE001 - never crash a caller that only wanted stats
        return


@dataclass(frozen=True)
class HardwareProfile:
    """A target GPU envelope, not a measurement.

    ``expected_peak_mib`` is the combined live-set estimate for the full mixed
    stack. It is deliberately *not* the sum of the isolated component peaks:
    the VLM, detector and renderer do not all peak at the same instant, and the
    CUDA context / allocator overhead is covered by ``reserve_mib``.
    """

    id: str
    total_mib: int
    reserve_mib: int
    expected_peak_mib: int
    description: str = ""

    def __post_init__(self) -> None:
        profile_id = str(self.id or "").strip()
        if not profile_id:
            raise ValueError("hardware profile id is required")
        object.__setattr__(self, "id", profile_id)
        values: dict[str, int] = {}
        for name in ("total_mib", "reserve_mib", "expected_peak_mib"):
            try:
                value = int(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be an integer MiB value") from exc
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
            values[name] = value
        if values["total_mib"] <= 0:
            raise ValueError("total_mib must be > 0")
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def usable_mib(self) -> int:
        """Total minus the reserved headroom."""
        return max(0, self.total_mib - self.reserve_mib)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "total_mib": self.total_mib,
            "reserve_mib": self.reserve_mib,
            "expected_peak_mib": self.expected_peak_mib,
            "description": self.description,
        }


#: Profiles used by the harness and the M4 driver. The A100 numbers are the
#: measured anchor; the 4090 expected peak is the documented ~21 GiB estimate
#: for the BF16 stack (VLM + detector + renderer + context), which is lower
#: than the sum of the isolated peaks on purpose.
PROFILES: dict[str, HardwareProfile] = {
    "a100-40gb": HardwareProfile(
        id="a100-40gb",
        total_mib=40960,
        reserve_mib=4096,
        expected_peak_mib=21504,
        description=(
            "A100-SXM4 40 GiB reference. The 21 GiB expected mixed peak is an "
            "informed combined estimate anchored on the measured A100 profile, "
            "not a sum of the isolated VLM/detector/render peaks."
        ),
    ),
    "4090-24gb": HardwareProfile(
        id="4090-24gb",
        total_mib=24576,
        reserve_mib=2048,
        expected_peak_mib=21504,
        description=(
            "24 GiB Ada target (RTX 4090). Expected mixed peak is the ~21 GiB "
            "live-set estimate for the BF16 stack plus a 2 GiB reserve; measure "
            "the actual peak before treating it as a result."
        ),
    ),
    "4090-local": HardwareProfile(
        id="4090-local",
        total_mib=24576,
        reserve_mib=2048,
        expected_peak_mib=21504,
        description=(
            "Local 4090 alias of 4090-24gb: 24576 MiB total, 2048 MiB kept as "
            "reserve, ~21 GiB honest expected mixed peak (not the sum of "
            "isolated component peaks)."
        ),
    ),
}

PROFILE_IDS = tuple(PROFILES)


def _resolve_profile(profile: HardwareProfile | str | Mapping) -> HardwareProfile:
    if isinstance(profile, HardwareProfile):
        return profile
    if isinstance(profile, str):
        try:
            return PROFILES[profile]
        except KeyError as exc:
            raise ValueError(
                f"unknown hardware profile {profile!r}; choose from {sorted(PROFILES)}"
            ) from exc
    if isinstance(profile, Mapping):
        try:
            return HardwareProfile(**dict(profile))
        except TypeError as exc:
            raise TypeError(f"invalid hardware profile mapping: {exc}") from exc
    raise TypeError(
        f"profile must be a HardwareProfile, a profile id or a mapping, got "
        f"{type(profile).__name__}"
    )


def _component_mib(value: Any) -> float:
    if isinstance(value, Mapping):
        for key in ("peak_mib", "peak_allocated_mib", "mib", "MiB", "value"):
            if key in value:
                value = value[key]
                break
        else:
            raise ValueError(
                "component mapping must contain peak_mib, peak_allocated_mib, "
                "mib or value"
            )
    number = _number(value)
    if number is None:
        raise ValueError(f"component MiB must be a finite number, got {value!r}")
    if number < 0:
        raise ValueError(f"component MiB must be >= 0, got {number}")
    return number


def _normalize_components(components: Any) -> dict[str, float]:
    if components is None:
        return {}
    if isinstance(components, (str, bytes)):
        raise TypeError("components must be a mapping of name -> MiB")
    items = components.items() if isinstance(components, Mapping) else components
    normalized: dict[str, float] = {}
    try:
        iterator = list(items)
    except TypeError as exc:
        raise TypeError("components must be a mapping of name -> MiB") from exc
    for item in iterator:
        if isinstance(item, Mapping):
            name = item.get("name", item.get("id"))
            if name is None:
                raise TypeError(
                    "component records must contain a name and a peak MiB field"
                )
            value = item
        else:
            try:
                name, value = item
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "components must be a mapping or (name, MiB) pairs"
                ) from exc
        key = str(name or "").strip()
        if not key:
            raise ValueError("component name is required")
        normalized[key] = round(_component_mib(value), 1)
    return normalized


def fit_report(
    profile: HardwareProfile | str | Mapping,
    snap: Mapping | None = None,
    components: Any = None,
) -> dict:
    """Compare a component budget (or the expected peak) with a hardware profile.

    ``required_mib`` is the sum of ``components`` when any are given, otherwise
    the profile's ``expected_peak_mib``. The fit condition is::

        required_mib + reserve_mib <= total_mib
        and (when a live snapshot exists) required_mib <= free/reusable MiB

    ``total_mib`` uses the profile envelope, capped by a *smaller* measured GPU
    total so a 24 GiB profile cannot pass on a 16 GiB card. A larger measured
    total does not raise the budget: the profile describes the target hardware.
    A missing/CPU snapshot never turns into a pass: ``available`` stays false
    and the caller (the CLI, the M4 driver) must treat that as unknown, not fit.
    """
    resolved = _resolve_profile(profile)
    snapshot_data: dict = {}
    if snap is not None:
        if not isinstance(snap, Mapping):
            raise TypeError(f"snap must be a mapping, got {type(snap).__name__}")
        snapshot_data = dict(snap)
    components_out = _normalize_components(components)

    available = bool(snapshot_data.get("available", False))
    profile_total = float(resolved.total_mib)
    reserve = float(resolved.reserve_mib)
    measured_total = _number(snapshot_data.get("total_mib"))
    measured_total = (
        round(measured_total, 1)
        if measured_total is not None and measured_total > 0
        else None
    )

    total = profile_total
    total_source = "profile"
    reasons: list[str] = []
    if measured_total is not None and measured_total < profile_total:
        total = measured_total
        total_source = "measured"

    if components_out:
        required = round(sum(components_out.values()), 1)
        source = "components"
    else:
        required = round(float(resolved.expected_peak_mib), 1)
        source = "expected_peak"

    headroom = round(total - required - reserve, 1)
    fits = headroom >= 0

    if total_source == "measured":
        reasons.append(
            f"measured GPU total {measured_total:.1f} MiB is below the "
            f"{resolved.id} profile total {profile_total:.1f} MiB; the measured "
            "total is the effective budget"
        )
    if headroom < 0:
        reasons.append(
            f"required {required:.1f} MiB + reserve {reserve:.1f} MiB exceeds "
            f"total {total:.1f} MiB by {-headroom:.1f} MiB"
        )

    usable_free: float | None = None
    if available:
        free = _number(snapshot_data.get("free_mib"))
        reserved = _number(snapshot_data.get("reserved_mib"))
        allocated = _number(snapshot_data.get("allocated_mib"))
        if free is not None:
            # free is immediately allocatable; (reserved - allocated) is the
            # allocator's cached-but-unused pool, which this process can reuse.
            reusable = free + max(0.0, (reserved or 0.0) - (allocated or 0.0))
            usable_free = round(reusable, 1)
            if usable_free + 1e-9 < required:
                fits = False
                reasons.append(
                    f"GPU has {usable_free:.1f} MiB free/reusable; required "
                    f"peak is {required:.1f} MiB"
                )
    else:
        detail = str(snapshot_data.get("reason", "") or "").strip()
        if snap is None:
            detail = "no GPU snapshot provided"
        reasons.append(
            "GPU metrics unavailable"
            + (f": {detail}" if detail else "")
            + "; fit is a static budget check only"
        )

    return {
        "profile": resolved.id,
        "profile_description": resolved.description,
        "available": available,
        "device": str(snapshot_data.get("device", "") or ""),
        "gpu_name": str(snapshot_data.get("gpu_name", "") or ""),
        "fits": bool(fits),
        "required_mib": required,
        "headroom_mib": headroom,
        "total_mib": round(total, 1),
        "reserve_mib": round(reserve, 1),
        "profile_total_mib": round(profile_total, 1),
        "measured_total_mib": measured_total,
        "total_source": total_source,
        "expected_peak_mib": int(resolved.expected_peak_mib),
        "source": source,
        "components": components_out,
        "components_total_mib": (
            round(sum(components_out.values()), 1) if components_out else 0.0
        ),
        "free_mib": snapshot_data.get("free_mib"),
        "usable_free_mib": usable_free,
        "allocated_mib": snapshot_data.get("allocated_mib"),
        "reserved_mib": snapshot_data.get("reserved_mib"),
        "peak_allocated_mib": snapshot_data.get("peak_allocated_mib"),
        "peak_reserved_mib": snapshot_data.get("peak_reserved_mib"),
        "driver_used_mib": snapshot_data.get("driver_used_mib"),
        "snapshot_reason": str(snapshot_data.get("reason", "") or ""),
        "reasons": reasons,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.1f}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def markdown_table(report: Mapping) -> str:
    """Render a :func:`fit_report` result as one Markdown table."""
    if not isinstance(report, Mapping):
        raise TypeError(f"report must be a mapping, got {type(report).__name__}")
    rows: list[tuple[str, Any]] = [
        ("profile", report.get("profile")),
        ("gpu_available", bool(report.get("available"))),
        ("gpu_name", report.get("gpu_name") or "-"),
        ("device", report.get("device") or "-"),
        ("total_mib", report.get("total_mib")),
        ("reserve_mib", report.get("reserve_mib")),
        ("required_mib", report.get("required_mib")),
        ("headroom_mib", report.get("headroom_mib")),
        ("fits", bool(report.get("fits"))),
        ("source", report.get("source")),
    ]
    components = report.get("components") or {}
    if isinstance(components, Mapping):
        for name in sorted(components):
            rows.append((f"component:{name}", components[name]))
    usable = report.get("usable_free_mib")
    if usable is not None:
        rows.append(("usable_free_mib", usable))
    peak_allocated = report.get("peak_allocated_mib")
    if peak_allocated is not None:
        rows.append(("peak_allocated_mib", peak_allocated))
    peak_reserved = report.get("peak_reserved_mib")
    if peak_reserved is not None:
        rows.append(("peak_reserved_mib", peak_reserved))
    driver_used = report.get("driver_used_mib")
    if driver_used is not None:
        rows.append(("driver_used_mib", driver_used))
    reasons = report.get("reasons") or []
    if reasons:
        rows.append(("reasons", "; ".join(str(reason) for reason in reasons)))
    lines = ["| metric | value |", "|---|---|"]
    lines.extend(f"| {_fmt(name)} | {_fmt(value)} |" for name, value in rows)
    return "\n".join(lines)
