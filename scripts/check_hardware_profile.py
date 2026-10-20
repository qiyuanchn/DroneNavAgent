#!/usr/bin/env python
"""Check whether a GPU hardware profile can hold the mixed navigation stack.

Examples:

    python scripts/check_hardware_profile.py --profile 4090-local --device cuda \
        --out outputs/hardware_profile

    # explicit component budget instead of the profile's expected mixed peak
    python scripts/check_hardware_profile.py --profile 4090-24gb \
        --component vlm=17300 --component detector=4200 --component renderer=1500

Exit codes are the contract:

``0``
    The profile fits and the GPU metrics were available.
``1``
    The GPU is available but the required peak + reserve does not fit.
``2``
    CUDA/torch/device unavailable. The report is still written, with
    ``available: false`` and explicit reasons; a static budget check is not a
    GPU result.

Writes ``report.json`` and ``report.md`` under ``--out``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from vlm_nav import hardware as _hardware  # noqa: E402
from vlm_nav.hardware import PROFILES, fit_report, markdown_table  # noqa: E402


def snapshot(device: str = "cuda") -> dict:
    """Delegate through ``vlm_nav.hardware``; tests may patch either name."""
    return _hardware.snapshot(device)


def reset_peak(device: str = "cuda") -> None:
    """Delegate through ``vlm_nav.hardware``; tests may patch either name."""
    return _hardware.reset_peak(device)


SCHEMA = "vlm_nav.hardware_profile.v1"
DEFAULT_OUT = REPO / "outputs" / "hardware_profile"

EXIT_FIT = 0
EXIT_NO_FIT = 1
EXIT_UNAVAILABLE = 2

STATUS_FIT = "fit"
STATUS_NO_FIT = "no_fit"
STATUS_UNAVAILABLE = "unavailable"


def _component(text: str) -> tuple[str, float]:
    name, separator, raw = str(text).partition("=")
    name = name.strip()
    if not separator or not name:
        raise argparse.ArgumentTypeError("expected NAME=MIB, e.g. vlm=17300")
    try:
        mib = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"component {name!r} MiB is not a number") from exc
    if not math.isfinite(mib) or mib < 0:
        raise argparse.ArgumentTypeError(
            f"component {name!r} MiB must be finite and >= 0"
        )
    return name, mib


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check a hardware profile against the live GPU and write a fit report."
    )
    parser.add_argument(
        "--profile", default="4090-local", choices=sorted(PROFILES),
        help="hardware profile to check (default: 4090-local)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="torch device string, e.g. cuda, cuda:0, cuda:1 (default: cuda)",
    )
    parser.add_argument(
        "--component", action="append", type=_component, default=None,
        metavar="NAME=MIB",
        help="component peak in MiB; repeatable. Without it the profile's "
             "expected_peak_mib is the required budget.",
    )
    parser.add_argument(
        "--reset-peak", action="store_true",
        help="reset the CUDA peak counters before reading the snapshot",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=f"directory for report.json/report.md (default: {DEFAULT_OUT})",
    )
    return parser


def _status_for(report: dict, snap: dict) -> tuple[str, int]:
    if not snap.get("available"):
        return STATUS_UNAVAILABLE, EXIT_UNAVAILABLE
    if report.get("fits"):
        return STATUS_FIT, EXIT_FIT
    return STATUS_NO_FIT, EXIT_NO_FIT


def _snapshot_table(snap: dict) -> str:
    rows = [
        ("available", "yes" if snap.get("available") else "no"),
        ("gpu_name", snap.get("gpu_name") or "-"),
        ("device", snap.get("device") or "-"),
        ("total_mib", snap.get("total_mib")),
        ("free_mib", snap.get("free_mib")),
        ("allocated_mib", snap.get("allocated_mib")),
        ("reserved_mib", snap.get("reserved_mib")),
        ("peak_allocated_mib", snap.get("peak_allocated_mib")),
        ("peak_reserved_mib", snap.get("peak_reserved_mib")),
        ("driver_used_mib", snap.get("driver_used_mib")),
        ("reason", snap.get("reason") or "-"),
    ]
    lines = ["| snapshot field | value |", "|---|---|"]
    for name, value in rows:
        if isinstance(value, float):
            value = f"{value:.1f}"
        text = str(value).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {name} | {text} |")
    return "\n".join(lines)


def _markdown(payload: dict) -> str:
    report = payload["report"]
    snap = payload["snapshot"]
    status = payload["status"]
    lines = [f"# Hardware profile check: {report['profile']}", ""]
    lines.append(f"Generated: {payload['generated_utc']}")
    lines.append("")
    lines.append(f"Status: **{status}** (exit `{payload['exit_code']}`)")
    lines.append("")
    lines.append(f"- GPU available: {'true' if snap.get('available') else 'false'}")
    lines.append(f"- Reason: {snap.get('reason') or 'none'}")
    lines.append(f"- Device: `{snap.get('device')}`")
    if snap.get("gpu_name"):
        lines.append(f"- GPU: {snap['gpu_name']}")
    lines.append("")
    lines.append(markdown_table(report))
    lines.append("")
    lines.append("## Snapshot")
    lines.append("")
    lines.append(_snapshot_table(snap))
    lines.append("")
    if not snap.get("available"):
        lines.append(
            "The GPU metrics are unavailable; `fits` above is only a static "
            "budget check and must not be read as a passed hardware check."
        )
        lines.append("")
    lines.append(
        "`required_mib` is the component sum or the profile's expected mixed "
        "peak. `headroom_mib = total_mib - reserve_mib - required_mib`; a "
        "negative value does not fit. Isolated component peaks are not additive "
        "measurements of the combined live set."
    )
    lines.append("")
    return "\n".join(lines)


def write_report(out: str | Path, payload: dict) -> dict:
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "report.json"
    markdown_path = directory / "report.md"
    written = {"json": str(json_path), "markdown": str(markdown_path)}
    payload = {**payload, "written": written}
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return written


def evaluate(
    profile_id: str,
    *,
    device: str = "cuda",
    components: dict | None = None,
    do_reset_peak: bool = False,
) -> dict:
    """Build the JSON payload without writing files (used by tests and the CLI)."""
    profile = PROFILES[profile_id]
    if do_reset_peak:
        reset_peak(device)
    snap = snapshot(device)
    report = fit_report(profile, snap, components or None)
    status, code = _status_for(report, snap)
    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile": profile.to_dict(),
        "components": report["components"],
        "available": bool(snap.get("available")),
        "fits": bool(report.get("fits")),
        "snapshot": snap,
        "report": report,
        "status": status,
        "exit_code": code,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    components = dict(args.component or [])
    payload = evaluate(
        args.profile,
        device=args.device,
        components=components or None,
        do_reset_peak=bool(args.reset_peak),
    )
    written = write_report(args.out, payload)
    payload["written"] = written
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return int(payload["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
