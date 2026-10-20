#!/usr/bin/env python
"""Replay-set accuracy check for the scan image-size / contact-sheet change.

The frozen replay set has known view poses and a known target position, so the
ground truth for "which scan heading shows the target" is computable without a
model: project the target into every replay view and keep the visible views.
This script asks the same VLM the same question twice:

``baseline``
    N individual frames at ``--baseline-size`` (default 640x360), the old
    prompt/contract.
``optimized``
    one numbered contact sheet built from the native 320x180 frames (half-size
    tiles), the new prompt/contract.

It reports parse rate, exact/visible/within-one-view accuracy, latency, image
count/pixels and prompt/completion/cached tokens. Exit 0 when the optimized
configuration is not worse than the baseline on exact/visible accuracy; exit 1
when it is worse, 2 when the replay set or backend is unavailable.

    python scripts/replay_accuracy.py --replay-dir outputs/vlm_nav/replay/window_chair \
        --backend openai --model qwen3-vl-8b \
        --base-url http://127.0.0.1:8000/v1 --repeat 3 \
        --out outputs/vlm_nav/replay_accuracy

    # no model needed: validate the replay/truth/sheet path only
    python scripts/replay_accuracy.py --replay-dir ... --dry-run
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from replay_frames import ReplayError, ReplaySet, load_replay  # noqa: E402
from vlm_nav.actions import parse_scan_choice  # noqa: E402
from vlm_nav.prompts import SCAN_SYSTEM_PROMPT, build_scan_prompt  # noqa: E402
from vlm_nav.skills_approach import world_to_pixel  # noqa: E402
from vlm_nav.vision import contact_sheet  # noqa: E402
from vlm_nav.vlm import make_backend, usage_metrics  # noqa: E402

SCHEMA = "vlm_nav.replay_accuracy.v1"

BASELINE_SYSTEM_PROMPT = """You are the navigation controller of a camera drone. The drone has \
just rotated in place and you see the views it captured, in order, one per equal \
heading step (view 0 is the initial heading; headings increase by the same angle \
per view, covering a full circle).

Reply with EXACTLY ONE JSON object and nothing else:
{"view": <int>, "target_visible": <true|false>, "distance_m": <float>, "reason": "<short>"}

- "view": index of the view that shows the target best, or the view whose heading \
points closest to where the target is most likely to be.
- "target_visible": true only if you can actually see the target object in that view.
- "distance_m": if the target is visible, your estimate of how far away it is in \
meters; otherwise 0.
- Never output anything except the JSON object."""


def baseline_scan_prompt(instruction: str, n_views: int) -> str:
    return (
        f"Navigation instruction: {instruction}\n\n"
        f"There are {n_views} views (index 0 to {n_views - 1}), taken while rotating "
        "in place over a full circle. Which view should the drone face next?"
    )


def _target_of(replay: ReplaySet) -> tuple[float, float, float] | None:
    raw = replay.scene.get("target_position_flu")
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return None
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    return values if all(math.isfinite(value) for value in values) else None


def scan_views(replay: ReplaySet) -> list:
    """The level scan views (fallback: all level views)."""
    views = list(replay.views_in_phase("scan"))
    if not views:
        views = [view for view in replay.views if abs(view.pitch_deg) < 1e-9]
    if not views:
        views = list(replay.views)
    return views


def ground_truth(views: Sequence, target: Sequence[float]) -> dict:
    """Project the target into every view; visible/centred view is the truth."""
    projected: list[dict] = []
    for index, view in enumerate(views):
        pixel = world_to_pixel(
            target,
            view.pose_flu,
            view.yaw_rad,
            view.intrinsics,
            pitch_rad=view.pitch_rad,
        )
        projected.append({
            "index": index,
            "view_id": view.view_id,
            "pixel": None if pixel is None else [round(float(pixel[0]), 3),
                                                round(float(pixel[1]), 3)],
            "visible": pixel is not None,
        })
    visible = [item for item in projected if item["visible"]]
    if not visible:
        raise ReplayError(
            "the target is not visible in any scan view; the replay set cannot "
            "provide ground truth for this check"
        )
    best = min(
        visible,
        key=lambda item: math.hypot(
            item["pixel"][0] - 500.0, item["pixel"][1] - 500.0
        ),
    )
    return {
        "best_index": int(best["index"]),
        "best_view_id": best["view_id"],
        "visible_indices": [int(item["index"]) for item in visible],
        "views": projected,
    }


def _circular_distance(a: int, b: int, n: int) -> int:
    delta = abs(int(a) - int(b)) % max(1, n)
    return min(delta, max(1, n) - delta)


def score_choice(
    choice: int | None, truth: Mapping, n_views: int
) -> dict:
    best = int(truth["best_index"])
    visible = {int(index) for index in truth["visible_indices"]}
    if choice is None:
        return {
            "parse_ok": False,
            "exact": False,
            "visible": False,
            "within_one": False,
        }
    return {
        "parse_ok": True,
        "exact": int(choice) == best,
        "visible": int(choice) in visible,
        "within_one": _circular_distance(int(choice), best, n_views) <= 1,
    }


def _frame_pixels(size: tuple[int, int]) -> int:
    return int(size[0]) * int(size[1])


def _usage_add(total: dict, metrics: Mapping | None) -> None:
    if not isinstance(metrics, Mapping):
        return
    for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
        try:
            total[key] = int(total.get(key, 0)) + int(metrics.get(key) or 0)
        except (TypeError, ValueError):
            continue


def run_configuration(
    backend: Any,
    views: Sequence,
    truth: Mapping,
    *,
    instruction: str,
    baseline_size: tuple[int, int],
    repeats: int = 1,
) -> dict:
    """Run baseline and optimized scan calls and return per-call records."""
    frames_native = [view.load_frame() for view in views]
    frames_baseline = [
        view.load_frame(image_size=baseline_size) for view in views
    ]
    sheet = contact_sheet(frames_native, scale=0.5)
    baseline_records: list[dict] = []
    optimized_records: list[dict] = []
    usage = {"baseline": {}, "optimized": {}}
    for _ in range(max(1, int(repeats))):
        started = time.perf_counter()
        raw = backend.decide(
            frames_baseline,
            BASELINE_SYSTEM_PROMPT,
            baseline_scan_prompt(instruction, len(views)),
            {"step": "replay_scan_baseline", "json_mode": True},
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        choice, visible, distance = parse_scan_choice(raw, len(views))
        baseline_records.append({
            "raw": str(raw),
            "choice": choice,
            "visible": visible,
            "distance_m": distance,
            "latency_ms": round(elapsed, 3),
            **score_choice(choice, truth, len(views)),
            "images_sent": len(views),
            "image_pixels": len(views) * _frame_pixels(baseline_size),
        })
        started = time.perf_counter()
        raw = backend.decide(
            [sheet.tensor()],
            SCAN_SYSTEM_PROMPT,
            build_scan_prompt(instruction, len(views)),
            {
                "step": "replay_scan_contact_sheet",
                "json_mode": True,
                "contact_sheet": True,
                "image_size": None,
            },
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        choice, visible, distance = parse_scan_choice(raw, len(views))
        optimized_records.append({
            "raw": str(raw),
            "choice": choice,
            "visible": visible,
            "distance_m": distance,
            "latency_ms": round(elapsed, 3),
            **score_choice(choice, truth, len(views)),
            "images_sent": 1,
            "image_pixels": (
                sheet.image.size[0] * sheet.image.size[1]
            ),
        })
    return {
        "baseline": baseline_records,
        "optimized": optimized_records,
        "sheet": {
            "width": sheet.image.size[0],
            "height": sheet.image.size[1],
            "columns": sheet.columns,
            "rows": sheet.rows,
            "tile_width": sheet.tile_width,
            "tile_height": sheet.tile_height,
        },
    }


def aggregate(records: Sequence[dict]) -> dict:
    total = max(1, len(records))
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    for record in records:
        _usage_add(usage, record.get("usage_metrics"))
    return {
        "calls": len(records),
        "parse_rate": sum(1 for r in records if r["parse_ok"]) / total,
        "exact_rate": sum(1 for r in records if r["exact"]) / total,
        "visible_rate": sum(1 for r in records if r["visible"]) / total,
        "within_one_rate": sum(1 for r in records if r["within_one"]) / total,
        "mean_latency_ms": round(
            sum(float(r["latency_ms"]) for r in records) / total, 3
        ),
        "images_per_call": records[0]["images_sent"] if records else 0,
        "image_pixels_per_call": records[0]["image_pixels"] if records else 0,
        "usage": usage,
        "choices": [r["choice"] for r in records],
    }


def evaluate(
    replay: ReplaySet,
    backend: Any,
    *,
    baseline_size: tuple[int, int] = (640, 360),
    repeats: int = 1,
) -> dict:
    target = _target_of(replay)
    if target is None:
        raise ReplayError(
            "replay manifest has no scene.target_position_flu; cannot build truth"
        )
    views = scan_views(replay)
    if len(views) < 2:
        raise ReplayError("replay set has fewer than two scan views")
    truth = ground_truth(views, target)
    records: dict[str, list[dict]] = {"baseline": [], "optimized": []}
    logs: list[dict] = []
    original_log = getattr(backend, "on_log", None)
    backend.on_log = logs.append
    try:
        for _ in range(max(1, int(repeats))):
            run = run_configuration(
                backend,
                views,
                truth,
                instruction=replay.instruction,
                baseline_size=baseline_size,
                repeats=1,
            )
            records["baseline"].extend(run["baseline"])
            records["optimized"].extend(run["optimized"])
    finally:
        backend.on_log = original_log
    # attach backend usage metrics to each call in order (when reported)
    metrics = [
        usage_metrics(event.get("usage_metrics"))
        for event in logs if isinstance(event, dict)
    ]
    metrics = [item for item in metrics if item]
    for name in ("baseline", "optimized"):
        for index, record in enumerate(records[name]):
            if index < len(metrics):
                record["usage_metrics"] = metrics[index]
    result = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "replay": {
            "name": replay.name,
            "root": str(replay.root),
            "instruction": replay.instruction,
            "n_views": len(views),
            "target_position_flu": list(target),
            "truth": truth,
            "camera": {
                "width": replay.camera.width,
                "height": replay.camera.height,
                "hfov_deg": replay.camera.hfov_deg,
            },
        },
        "baseline": {
            "image_size": list(baseline_size),
            "summary": aggregate(records["baseline"]),
            "calls": records["baseline"],
        },
        "optimized": {
            "image_size": [replay.camera.width, replay.camera.height],
            "summary": aggregate(records["optimized"]),
            "calls": records["optimized"],
        },
    }
    baseline = result["baseline"]["summary"]
    optimized = result["optimized"]["summary"]
    result["comparison"] = {
        "exact_delta": optimized["exact_rate"] - baseline["exact_rate"],
        "visible_delta": optimized["visible_rate"] - baseline["visible_rate"],
        "within_one_delta": (
            optimized["within_one_rate"] - baseline["within_one_rate"]
        ),
        "image_count_ratio": (
            optimized["images_per_call"] / baseline["images_per_call"]
            if baseline["images_per_call"] else None
        ),
        "pixel_ratio": (
            optimized["image_pixels_per_call"] / baseline["image_pixels_per_call"]
            if baseline["image_pixels_per_call"] else None
        ),
        "prompt_token_ratio": (
            optimized["usage"]["prompt_tokens"] / baseline["usage"]["prompt_tokens"]
            if baseline["usage"].get("prompt_tokens") else None
        ),
        "not_worse": (
            optimized["exact_rate"] >= baseline["exact_rate"]
            and optimized["visible_rate"] >= baseline["visible_rate"]
        ),
    }
    return result


def write_report(report: Mapping, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "replay_accuracy.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    comparison = report.get("comparison", {}) if isinstance(report, dict) else {}
    baseline = report.get("baseline", {}).get("summary", {}) if report else {}
    optimized = report.get("optimized", {}).get("summary", {}) if report else {}
    lines = [
        "# Replay scan accuracy",
        "",
        f"- schema: `{report.get('schema', '')}`",
        f"- replay: `{report.get('replay', {}).get('name', '')}`",
        f"- scan views: {report.get('replay', {}).get('n_views', 0)}",
        f"- truth best view: {report.get('replay', {}).get('truth', {}).get('best_index')}",
        "",
        "| configuration | images/call | pixels/call | parse | exact | visible | within 1 | "
        "latency ms | prompt tok | cached tok |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in ("baseline", "optimized"):
        summary = report.get(name, {}).get("summary", {})
        usage = summary.get("usage", {})
        lines.append(
            f"| {name} | {summary.get('images_per_call')} | "
            f"{summary.get('image_pixels_per_call')} | "
            f"{summary.get('parse_rate', 0):.2f} | "
            f"{summary.get('exact_rate', 0):.2f} | "
            f"{summary.get('visible_rate', 0):.2f} | "
            f"{summary.get('within_one_rate', 0):.2f} | "
            f"{summary.get('mean_latency_ms', 0):.1f} | "
            f"{usage.get('prompt_tokens', 0)} | {usage.get('cached_tokens', 0)} |"
        )
    lines += [
        "",
        f"- exact delta: {comparison.get('exact_delta', 0):+.3f}",
        f"- visible delta: {comparison.get('visible_delta', 0):+.3f}",
        f"- image count ratio: {comparison.get('image_count_ratio')}",
        f"- pixel ratio: {comparison.get('pixel_ratio')}",
        f"- prompt token ratio: {comparison.get('prompt_token_ratio')}",
        f"- not worse: {comparison.get('not_worse')}",
    ]
    md_path = out_dir / "replay_accuracy.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare baseline scan frames with the 320x180 contact sheet "
                    "on a frozen replay set."
    )
    parser.add_argument("--replay-dir", required=True,
                        help="replay set directory (manifest.json + frames/)")
    parser.add_argument("--out", default=str(REPO / "outputs" / "vlm_nav" / "replay_accuracy"))
    parser.add_argument("--backend", default="openai",
                        choices=["openai", "hf", "mock"])
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--baseline-size", type=int, nargs=2, default=(640, 360),
                        metavar=("W", "H"))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--dry-run", action="store_true",
                        help="build truth/contact sheet and exit without a model")
    parser.add_argument("--strict", action="store_true",
                        help="also fail when optimized within-one accuracy drops")
    return parser


def build_backend(args: argparse.Namespace) -> Any:
    if args.backend == "openai":
        if not args.model:
            raise SystemExit("--backend openai requires --model")
        return make_backend(
            "openai", model=args.model, base_url=args.base_url,
            max_new_tokens=args.max_new_tokens,
        )
    if args.backend == "hf":
        if not args.model_path:
            raise SystemExit("--backend hf requires --model-path")
        return make_backend(
            "hf", model_path=args.model_path, device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
    return make_backend("mock:script")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        replay = load_replay(args.replay_dir)
        target = _target_of(replay)
        if target is None:
            raise ReplayError("replay manifest has no target_position_flu")
        views = scan_views(replay)
        truth = ground_truth(views, target)
        sheet = contact_sheet([view.load_frame() for view in views], scale=0.5)
    except ReplayError as exc:
        print(f"replay accuracy unavailable: {exc}", flush=True)
        return 2
    if args.dry_run:
        report = {
            "schema": SCHEMA,
            "dry_run": True,
            "replay": {
                "name": replay.name,
                "n_views": len(views),
                "target_position_flu": list(target),
                "truth": truth,
            },
            "sheet": {
                "width": sheet.image.size[0],
                "height": sheet.image.size[1],
                "columns": sheet.columns,
                "rows": sheet.rows,
            },
        }
        paths = write_report(report, Path(args.out))
        print(json.dumps({"dry_run": True, "truth": truth, "sheet": report["sheet"],
                          "written": paths}, indent=2))
        return 0
    backend = build_backend(args)
    try:
        report = evaluate(
            replay, backend, baseline_size=tuple(args.baseline_size),
            repeats=max(1, int(args.repeat)),
        )
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()
    paths = write_report(report, Path(args.out))
    comparison = report["comparison"]
    print(
        f"exact baseline→optimized: "
        f"{report['baseline']['summary']['exact_rate']:.2f}→"
        f"{report['optimized']['summary']['exact_rate']:.2f}; "
        f"visible {report['baseline']['summary']['visible_rate']:.2f}→"
        f"{report['optimized']['summary']['visible_rate']:.2f}; "
        f"images {report['baseline']['summary']['images_per_call']}→"
        f"{report['optimized']['summary']['images_per_call']}",
        flush=True,
    )
    print(f"report: {paths['json']}", flush=True)
    if not comparison["not_worse"]:
        return 1
    if args.strict and comparison["within_one_delta"] < 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
