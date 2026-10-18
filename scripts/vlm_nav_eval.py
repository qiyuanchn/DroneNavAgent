#!/usr/bin/env python
"""Batch evaluation driver for the VLM navigation harness.

Reads a matrix JSON file:

{
  "episodes": [
    {
      "name": "classroom_bookshelf_qwen8b",
      "world": "super",
      "instruction": "找到书架，飞到它旁边停下",
      "backend": "hf", "model_path": "/models_ro/Qwen3-VL-8B", "device": "cuda:0",
      "combined_ply": "...", "background_ply": "...", "semantic_manifest": "...",
      "voxel_size": 0.4, "target_index": 0, "initial_yaw_mode": "random",
      "max_steps": 30, "success_dist": 2.0, "seed": 7,
      "env_overrides": ["env.gaussian_splatting.start_goal.min_dist=2.5", ...],
      "worker_python": ".../venvs/super312/bin/python"
    },
    ...
  ]
}

Every spec key that matches a run_vlm_nav_3dgs.py argument is forwarded as
--kebab-case; booleans look_around/no_video map to --no-look-around/--no-video
when false/true respectively. Episodes are distributed round-robin over the
device lanes and run sequentially per lane. Each episode writes its artifacts
to <out_root>/<name>/; the aggregated table lands in <out_root>/matrix_summary.json.

    python scripts/vlm_nav_eval.py --matrix outputs/vlm_nav/eval/matrix_v1.json \
        --out-root outputs/vlm_nav/eval/run1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN_SCRIPT = REPO / "scripts" / "run_vlm_nav_3dgs.py"

# spec key -> (CLI flag, append when value == this)
BOOL_FLAGS = {
    "look_around": ("--no-look-around", False),
    "no_video": ("--no-video", True),
    "no_think": ("--no-think", True),
    "no_done_retry": ("--no-done-retry", True),
    "bev": ("--no-bev", False),
    "done_verify": ("--no-done-verify", False),
    "candidates_enabled": ("--no-candidates", False),
    "keyframe_enabled": ("--no-keyframe", False),
}
SKIP_KEYS = {"name"}


def spec_to_cmd(spec: dict, out_dir: Path, python: str) -> list[str]:
    cmd = [python, str(RUN_SCRIPT), "--world", spec["world"],
           "--instruction", spec["instruction"], "--backend", spec["backend"],
           "--out", str(out_dir)]
    for key, value in spec.items():
        if key in SKIP_KEYS or key in ("world", "instruction", "backend"):
            continue
        flag = "--" + key.replace("_", "-")
        if key in BOOL_FLAGS:
            flag, when = BOOL_FLAGS[key]
            if bool(value) == when:
                cmd.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (list, tuple)):
            cmd.append(flag)
            cmd.extend(str(v) for v in value)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def run_lane(lane_episodes: list[tuple[dict, Path]], python: str, results: dict,
             lock: threading.Lock) -> None:
    for spec, out_dir in lane_episodes:
        name = spec["name"]
        t0 = time.time()
        proc = subprocess.run(
            spec_to_cmd(spec, out_dir, python),
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        (out_dir / "driver.log").write_text(proc.stdout or "", encoding="utf-8")
        row = {
            "name": name,
            "returncode": proc.returncode,
            "wall_time_s": round(time.time() - t0, 1),
        }
        summary_path = out_dir / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            row.update({
                "success": summary["success"],
                "finish_reason": summary["finish_reason"],
                "steps": summary["steps"],
                "final_distance_m": summary["final_distance_m"],
                "min_distance_m": summary["min_distance_m"],
                "path_length_m": summary["path_length_m"],
                "parse_failures": summary["parse_failures"],
            })
        with lock:
            results[name] = row
        print(f"[done] {name}: rc={proc.returncode} "
              f"success={row.get('success')} reason={row.get('finish_reason')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--only", type=str, default=None,
                        help="comma-separated episode name filter")
    args = parser.parse_args()

    matrix = json.loads(Path(args.matrix).read_text())
    episodes = matrix["episodes"]
    if args.only:
        keep = set(args.only.split(","))
        episodes = [e for e in episodes if e["name"] in keep]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    lanes: dict[str, list] = {}
    for spec in episodes:
        lanes.setdefault(str(spec.get("device", "cuda:0")), []).append(
            (spec, out_root / spec["name"])
        )
    results: dict = {}
    lock = threading.Lock()
    threads = [
        threading.Thread(target=run_lane, args=(lane, args.python, results, lock))
        for lane in lanes.values()
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rows = [results[e["name"]] for e in episodes if e["name"] in results]
    (out_root / "matrix_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n{'episode':44s} {'ok':5s} {'reason':24s} {'steps':>5s} "
          f"{'min_d':>6s} {'path':>6s} {'wall_s':>7s}")
    for row in rows:
        print(f"{row['name']:44s} {str(row.get('success')):5s} "
              f"{str(row.get('finish_reason'))[:24]:24s} {row.get('steps', '-'):>5} "
              f"{row.get('min_distance_m', 0):6.2f} {row.get('path_length_m', 0):6.2f} "
              f"{row.get('wall_time_s', 0):7.1f}")


if __name__ == "__main__":
    main()
