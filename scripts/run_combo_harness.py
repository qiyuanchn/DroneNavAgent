#!/usr/bin/env python
"""Batch-evaluate DiffAero scene+object+obstacle combinations through the harness.

The matrix is the JSON written by ``scripts/build_harness_scenes.py``: a
``{"episodes": [...]}`` object whose specs carry the merged PLY, semantic
manifest, target index, instruction, success distance and the combination
metadata (``name``, ``preset``, ``scene``, ``target_edge_distance_m``).

Every episode is one subprocess of ``scripts/run_vlm_nav_tools.py`` using the
OpenAI-compatible backend against one shared vLLM endpoint (``--base-url``).
Episodes run sequentially: one endpoint serves one episode at a time, and a
GPU renderer should not be shared by concurrent runs. Each episode writes its
artifacts plus a ``runner.log`` (stdout/stderr) under ``<out-root>/<name>/``;
the aggregated table lands in ``<out-root>/matrix_summary.json``::

    {"episodes": [{name, preset, scene, target_name, success,
                   success_evidence, finish_reason, final_distance_m,
                   tool_calls, turns, vlm_calls, wall_time_s,
                   target_edge_distance_m, exit_code, error}, ...],
     "totals": {episodes, successes, evidence_backed, mean_wall_time_s}}

``--resume`` skips an episode whose ``summary.json`` already exists (the
existing summary is still aggregated). A nonzero exit or an unreadable summary
records ``success: false`` with the error and the batch continues with the next
episode. ``--dry-run`` prints the subprocess commands and writes nothing.

    python scripts/run_combo_harness.py \
        --matrix outputs/vlm_nav/combos/matrix.json \
        --out-root outputs/vlm_nav/combos/run1 \
        --base-url http://127.0.0.1:8000/v1 --model qwen3-vl-8b
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = Path("scripts") / "run_vlm_nav_tools.py"
DEMO_RUNNER = Path("scripts") / "run_semantic_corner_demo.py"
SUMMARY_NAME = "summary.json"
LOG_NAME = "runner.log"
MATRIX_SUMMARY_NAME = "matrix_summary.json"
DEFAULT_MODEL = "qwen3-vl-8b"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
TIMEOUT_EXIT_CODE = 124


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", type=str, required=True,
                        help="matrix JSON from scripts/build_harness_scenes.py")
    parser.add_argument("--out-root", type=str, required=True,
                        help="root directory for the per-episode runs")
    parser.add_argument("--runner", choices=["demo", "tools"], default="demo",
                        help="demo: the bounded search->approach->confirm->verify "
                             "pipeline with video/XY artifacts; tools: the full "
                             "AgentRuntime tool loop")
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="interpreter that runs the selected runner")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="model name sent to the OpenAI-compatible endpoint")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="HiGS CUDA device for the demo runner")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL,
                        help="shared OpenAI-compatible vLLM endpoint")
    parser.add_argument("--image-size", type=int, nargs=2, default=(320, 180),
                        metavar=("W", "H"),
                        help="camera render resolution passed as --render-size")
    parser.add_argument("--max-turns", type=int, default=30,
                        help="model decision turns per episode")
    parser.add_argument("--tool-budget", type=int, default=60,
                        help="max tool calls per episode")
    parser.add_argument("--timeout-s", type=float, default=900.0,
                        help="per-episode subprocess timeout in seconds")
    parser.add_argument("--resume", action="store_true",
                        help="skip episodes whose summary.json already exists")
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N episodes of the matrix")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the subprocess commands without running")
    return parser.parse_args(argv)


def load_matrix(path: str | Path) -> list[dict]:
    """Read ``{"episodes": [...]}`` (a bare JSON list is accepted too)."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        episodes = data.get("episodes")
    elif isinstance(data, list):
        episodes = data
    else:
        episodes = None
    if not isinstance(episodes, list):
        raise SystemExit(f"{path}: expected a JSON object with an 'episodes' list")
    for index, spec in enumerate(episodes):
        if not isinstance(spec, dict):
            raise SystemExit(f"{path}: episode {index} is not a JSON object")
    return episodes


def episode_name(spec: Mapping[str, Any], index: int = 0) -> str:
    """The artifact-directory name: one safe path component per episode."""
    name = str(spec.get("name") or f"episode_{index:03d}")
    if name in ("", ".", "..") or Path(name).name != name:
        raise SystemExit(
            f"episode {index}: name {name!r} is not a single path component"
        )
    return name


def _required(spec: Mapping[str, Any], key: str) -> Any:
    value = spec.get(key)
    if value is None:
        raise KeyError(f"matrix episode is missing {key!r}")
    return value


def _image_size(args: argparse.Namespace) -> tuple[int, int]:
    width, height = tuple(getattr(args, "image_size", (320, 180)))
    return int(width), int(height)


def episode_command(spec: Mapping[str, Any], out_dir: str | Path,
                    args: argparse.Namespace) -> list[str]:
    """The subprocess command for one episode (demo or full tool runtime)."""
    width, height = _image_size(args)
    python = str(getattr(args, "python", None) or sys.executable)
    manifest = str(_required(spec, "semantic_manifest"))
    target_index = str(int(spec.get("target_index", 0)))
    if str(getattr(args, "runner", "demo")) == "tools":
        return [
            python,
            RUNNER.as_posix(),
            "--world", str(spec.get("world") or "higs"),
            "--ply", str(_required(spec, "combined_ply")),
            "--semantic-manifest", manifest,
            "--target-index", target_index,
            "--auto-start",
            "--out", str(out_dir),
            "--success-dist", str(_required(spec, "success_dist")),
            "--instruction", str(_required(spec, "instruction")),
            "--backend", "openai",
            "--model", str(getattr(args, "model", DEFAULT_MODEL)),
            "--base-url", str(getattr(args, "base_url", DEFAULT_BASE_URL)),
            "--render-size", str(width), str(height),
            "--max-turns", str(int(getattr(args, "max_turns", 30))),
            "--tool-budget", str(int(getattr(args, "tool_budget", 60))),
        ]
    return [
        python,
        DEMO_RUNNER.as_posix(),
        "--manifest", manifest,
        "--target-index", target_index,
        "--auto-start",
        "--out", str(out_dir),
        "--success-dist", str(_required(spec, "success_dist")),
        "--prompt", str(_required(spec, "instruction")),
        "--renderer", "higs",
        "--device", str(getattr(args, "device", "cuda:0")),
        "--model", str(getattr(args, "model", DEFAULT_MODEL)),
        "--base-url", str(getattr(args, "base_url", DEFAULT_BASE_URL)),
        "--width", str(width),
        "--height", str(height),
        "--fps", "6",
        "--hold-frames", "4",
        "--max-new-tokens", "256",
        "--tool-budget", str(int(getattr(args, "tool_budget", 120))),
    ]


def runner_env() -> dict[str, str]:
    """The inherited environment with the repo root on ``PYTHONPATH``."""
    env = os.environ.copy()
    root = str(REPO_ROOT)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root + (os.pathsep + existing if existing else "")
    return env


def read_summary(path: str | Path) -> tuple[dict | None, str | None]:
    """Read ``summary.json``; returns (summary, error)."""
    path = Path(path)
    if not path.is_file():
        return None, "summary.json missing"
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        return None, f"summary.json unreadable: {type(exc).__name__}: {exc}"
    if not isinstance(data, dict):
        return None, "summary.json is not a JSON object"
    return data, None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalise_summary(summary: dict) -> dict:
    """Map the demo runner's nested verdict onto the tools runner's fields.

    ``run_semantic_corner_demo.py`` writes ``finish.{success,verified,claim}``
    and a ``results.verify`` block; the tools runner writes those at the top
    level. The batch table wants one shape.
    """
    finish = summary.get("finish")
    if not isinstance(finish, dict):
        results = summary.get("results") or {}
        finish = results.get("finish") if isinstance(results, dict) else None
    if not isinstance(finish, dict):
        return summary
    if isinstance(finish.get("data"), dict):
        finish = finish["data"]
    summary = dict(summary)
    summary.setdefault("success", finish.get("success"))
    summary.setdefault("success_evidence", finish.get("verified"))
    summary.setdefault(
        "finish_reason", finish.get("claim") or finish.get("reason")
    )
    verify = ((summary.get("results") or {}).get("verify") or {}).get("data") or {}
    if summary.get("final_distance_m") is None:
        summary["final_distance_m"] = verify.get("measured_distance_m")
    candidate = summary.get("candidate_position")
    truth = (summary.get("target_ground_truth") or {}).get("position_flu")
    if (
        isinstance(candidate, (list, tuple)) and len(candidate) >= 3
        and isinstance(truth, (list, tuple)) and len(truth) >= 3
    ):
        summary["position_error_m"] = round(
            math.dist([float(v) for v in candidate[:3]],
                      [float(v) for v in truth[:3]]),
            3,
        )
    return summary


def episode_row(spec: Mapping[str, Any], out_dir: str | Path,
                exit_code: int | None, error: str | None = None,
                elapsed_s: float | None = None) -> dict:
    """One aggregated row; nonzero exit or a bad summary forces failure."""
    summary, summary_error = read_summary(Path(out_dir) / SUMMARY_NAME)
    summary = _normalise_summary(summary or {})
    problems = [error] if error else []
    if exit_code is None and not error:
        problems.append("runner did not start")
    elif exit_code not in (0, None) and not error:
        problems.append(f"runner exited with code {exit_code}")
    if summary_error and (exit_code == 0 or summary_error != "summary.json missing"):
        problems.append(summary_error)

    success = bool(summary.get("success"))
    evidence = summary.get("success_evidence")
    if evidence is not None:
        evidence = bool(evidence)
    if exit_code != 0:
        success = False
        evidence = False
    wall_time = summary.get("wall_time_s")
    if not _is_number(wall_time):
        wall_time = elapsed_s
    target_name = summary.get("target_name")
    if target_name is None:
        target_name = spec.get("target_name")
    return {
        "name": str(spec.get("name") or Path(out_dir).name),
        "preset": spec.get("preset"),
        "scene": spec.get("scene"),
        "target_name": target_name,
        "success": success,
        "success_evidence": evidence,
        "finish_reason": summary.get("finish_reason"),
        "final_distance_m": summary.get("final_distance_m"),
        "tool_calls": summary.get("tool_calls"),
        "turns": summary.get("turns"),
        "vlm_calls": summary.get("vlm_calls"),
        "position_error_m": summary.get("position_error_m"),
        "wall_time_s": wall_time,
        "target_edge_distance_m": spec.get("target_edge_distance_m"),
        "exit_code": exit_code,
        "error": "; ".join(problems) or None,
    }


def run_episode(spec: Mapping[str, Any], out_dir: str | Path,
                args: argparse.Namespace,
                command: Sequence[str] | None = None) -> dict:
    """Run one episode with stdout/stderr streamed into ``runner.log``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if command is None:
        command = episode_command(spec, out_dir, args)
    else:
        command = [str(part) for part in command]
    timeout = float(getattr(args, "timeout_s", 900.0))
    started = time.time()
    exit_code: int | None = None
    error: str | None = None
    with open(out_dir / LOG_NAME, "w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(command) + "\n\n")
        log.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                env=runner_env(),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                text=True,
            )
            exit_code = int(completed.returncode)
        except subprocess.TimeoutExpired:
            exit_code = TIMEOUT_EXIT_CODE
            error = f"timeout after {timeout:g}s"
        except OSError as exc:
            error = f"{type(exc).__name__}: {exc}"
    return episode_row(
        spec, out_dir, exit_code, error, round(time.time() - started, 3)
    )


def aggregate_results(rows: Sequence[Mapping[str, Any]]) -> dict:
    """Totals over the episode rows.

    ``mean_wall_time_s`` is the mean of the rows that carry a numeric
    ``wall_time_s`` (0.0 when none do); a skipped resume still has the wall
    time its own summary recorded.
    """
    rows = list(rows)
    times = [
        float(row["wall_time_s"]) for row in rows if _is_number(row.get("wall_time_s"))
    ]
    errors = [
        float(row["position_error_m"])
        for row in rows
        if _is_number(row.get("position_error_m"))
    ]
    return {
        "episodes": len(rows),
        "successes": sum(1 for row in rows if row.get("success")),
        "evidence_backed": sum(1 for row in rows if row.get("success_evidence")),
        "mean_wall_time_s": round(sum(times) / len(times), 3) if times else 0.0,
        "mean_position_error_m": (
            round(sum(errors) / len(errors), 3) if errors else None
        ),
    }


def write_matrix_summary(out_root: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write ``<out-root>/matrix_summary.json`` and return its path."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / MATRIX_SUMMARY_NAME
    payload = {"episodes": list(rows), "totals": aggregate_results(rows)}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")
    return path


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    episodes = load_matrix(args.matrix)
    if args.limit is not None:
        episodes = episodes[: max(0, int(args.limit))]
    names = [episode_name(spec, index) for index, spec in enumerate(episodes)]
    if len(set(names)) != len(names):
        raise SystemExit("matrix episode names must be unique")
    out_root = Path(args.out_root).expanduser().resolve()

    if args.dry_run:
        commands = []
        for spec, name in zip(episodes, names):
            command = episode_command(spec, out_root / name, args)
            commands.append(command)
            print(shlex.join(command), flush=True)
        print(f"[dry-run] {len(commands)} episode(s); nothing run", flush=True)
        return {"dry_run": True, "commands": commands}

    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    total = len(episodes)
    for index, (spec, name) in enumerate(zip(episodes, names), start=1):
        out_dir = out_root / name
        if args.resume and (out_dir / SUMMARY_NAME).is_file():
            row = episode_row(spec, out_dir, 0, None, None)
            print(f"[{index}/{total}] {name}: resume (existing summary.json)",
                  flush=True)
        else:
            try:
                command = episode_command(spec, out_dir, args)
            except (KeyError, TypeError, ValueError) as exc:
                out_dir.mkdir(parents=True, exist_ok=True)
                row = episode_row(spec, out_dir, None,
                                  f"invalid episode spec: {exc}", 0.0)
            else:
                print(f"[{index}/{total}] {name}: running", flush=True)
                row = run_episode(spec, out_dir, args, command)
        rows.append(row)
        print(
            f"[{index}/{total}] {name}: success={row['success']} "
            f"reason={row['finish_reason']} rc={row['exit_code']}",
            flush=True,
        )
        write_matrix_summary(out_root, rows)

    path = write_matrix_summary(out_root, rows)
    totals = aggregate_results(rows)
    print(
        f"[combo] {totals['successes']}/{totals['episodes']} success, "
        f"{totals['evidence_backed']} evidence-backed, "
        f"mean {totals['mean_wall_time_s']} s -> {path}",
        flush=True,
    )
    return {"episodes": rows, "totals": totals, "matrix_summary": str(path)}


if __name__ == "__main__":
    main()
