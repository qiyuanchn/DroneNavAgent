#!/usr/bin/env python
"""Four-configuration comparison driver for the VLM navigation agent (goal.md 7).

The goal.md experiment ladder is run as one paired matrix. Every configuration
sees the same world, scene, target, start pose, initial heading, seed and VLM
call budget; only the agent capability set differs:

  1. ``action``      current action loop        scripts/run_vlm_nav_3dgs.py
  2. ``tool_basic``  tool loop                  scripts/run_vlm_nav_tools.py --toolset basic
  3. ``tool_memory`` tool loop + evidence memory        --toolset memory
  4. ``tool_full``   + active viewpoints/persistent task --toolset full

Mock run (CPU world, no weights):

    python scripts/compare_agent_configs.py --world cpu --ply scene.ply \
        --target-pos 0 0 1.0 --instruction "Find the target and fly to it." \
        --backend mock --seeds 7,11 --out outputs/vlm_nav/eval/compare_mock

Real model run:

    python scripts/compare_agent_configs.py --world higs --ply scene.ply \
        --target-pos 0 0 0.5 --instruction "Find the book." --backend hf \
        --model-path /path/Qwen3-VL-8B --device cuda --seeds 7,11,13 \
        --max-vlm-calls 20 --out outputs/vlm_nav/eval/compare_qwen

Both runners are driven as subprocesses (their CLIs are the contract), one
directory per episode:

    <out>/<config>/<task>/<yaw_mode>_seed<seed>/

Budget mapping ("same max VLM calls"): ``--max-vlm-calls B`` becomes
``--max-steps B`` for the action loop and ``--max-turns B`` (+ ``--tool-budget``)
for the tool loops; the *actual* ``vlm_calls`` reported by each runner is
recorded per episode (the harness spends extra calls on scans and on the
done-verification crop, the tool loop one per turn).

Initial-yaw groups (goal.md: keep the initially-visible and initially-invisible
task classes): ``--initial-yaw-modes visible`` starts with the target inside the
camera FOV, ``random`` picks a deterministic (seeded) heading for which the
target is outside the FOV. Either way the same ``--start-yaw-deg`` is passed to
all four configurations of a cell, so the pair is identical.

Metrics (per episode and aggregated per configuration / yaw group):
  success          runner verdict (``summary.json``)
  vlm_calls        model calls actually spent
  tokens_estimate  images_total * image_tokens + text_chars / chars_per_token
                   (``--image-token-divisor``: 28x28 patch -> 640x360 ~ 294)
  wall_time_s      runner wall clock
  repeat_search    repeat_scans + region_revisits
  repeat_scans     scan-type calls issued from a cell that was already scanned
  region_revisits  entries into a 2 m cell that was already visited
  cached_calls     tool calls the runtime answered as "cached" (no progress)

Episodes run one at a time by default; ``--jobs N`` runs N subprocesses in
parallel (mock/CPU sweeps; keep 1 when the children share a GPU). A failed
episode is recorded as an error row and the remaining ones still run; the driver
exits 1 when any episode failed.

Artifacts:
  compare_summary.json  meta (matrix, budget mapping, pairing check), per-episode
                        metrics, aggregates per configuration x yaw group
  compare_table.md      the Markdown comparison table
  episodes.jsonl        one line per episode (same rows)
  run_config.json       resolved CLI of the driver
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ACTION_RUNNER = ROOT / "scripts" / "run_vlm_nav_3dgs.py"
TOOL_RUNNER = ROOT / "scripts" / "run_vlm_nav_tools.py"

# goal.md section 7 ladder, in report order
CONFIG_ORDER = ("action", "tool_basic", "tool_memory", "tool_full")
CONFIG_LABELS = {
    "action": "action loop",
    "tool_basic": "tool loop",
    "tool_memory": "tool loop + evidence memory",
    "tool_full": "tool loop + viewpoints + persistent task",
}
CONFIG_TOOLSETS = {"tool_basic": "basic", "tool_memory": "memory", "tool_full": "full"}
ACTION_CONFIG = "action"

YAW_MODES = ("target", "visible", "random")
SCAN_MODES = ("lookaround", "scan", "pan")

# scene/budget options that must be identical across the four configurations
# of one (task, seed, yaw-mode) cell - checked and reported in the summary
PAIRING_KEYS = (
    "--world", "--ply", "--scene-scale", "--min-opacity", "--combined-ply",
    "--background-ply", "--semantic-manifest", "--voxel-size", "--target-index",
    "--target-name", "--target-pos", "--target-radius", "--start-pos",
    "--start-yaw-deg", "--success-dist", "--seed", "--initial-yaw-mode",
    "--render-size", "--image-size", "--context-frames", "--scan-views",
    "--scan-pitch-deg", "--device", "--max-new-tokens",
)

SUMMARY_FILE = "summary.json"
ACTION_TRACE = "episode_trace.jsonl"
TOOL_TRACE = "tool_trace.jsonl"
# tool loop traces written by older runners nest the tool result under "data"
_TOOL_DATA_KEYS = ("data", "result_data")
_SCAN_RE = re.compile(r"^(re)?scan\b|\bscan\b.*->\s*view|->\s*view\s*\d", re.IGNORECASE)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _int_list(text: str) -> list[int]:
    return [int(part) for part in str(text).replace(" ", "").split(",") if part]


def _str_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _wrap_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def _as_float(value) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def read_json(path: Path) -> dict | None:
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def read_jsonl(path: Path) -> list[dict]:
    entries: list[dict] = []
    try:
        text = path.read_text()
    except OSError:
        return entries
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            entries.append(data)
    return entries


def _find_trace(out_dir: Path, *names: str) -> list[dict]:
    for name in names:
        entries = read_jsonl(out_dir / name)
        if entries:
            return entries
    for path in sorted(out_dir.glob("*trace*.jsonl")):
        entries = read_jsonl(path)
        if entries:
            return entries
    return []


def _result_data(entry: dict) -> dict:
    for key in _TOOL_DATA_KEYS:
        value = entry.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _pose_of(entry: dict) -> list[float] | None:
    """Best-effort pose of one trace entry (module and fallback trace schemas)."""
    candidates = [
        _result_data(entry).get("pose"),
        _result_data(entry).get("position_flu"),
        entry.get("position_flu"),
    ]
    top = entry.get("pose")
    if isinstance(top, dict):
        candidates.append(top.get("position_flu"))
    elif isinstance(top, (list, tuple)):
        candidates.append(top)
    for value in candidates:
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            try:
                return [float(v) for v in value[:3]]
            except (TypeError, ValueError):
                continue
    return None


def _cell_of(pose, grid_m: float) -> tuple[int, int] | None:
    if not isinstance(pose, (list, tuple)) or len(pose) < 2:
        return None
    return (int(math.floor(float(pose[0]) / grid_m)), int(math.floor(float(pose[1]) / grid_m)))


def search_accounting(events: list[dict], grid_m: float = 2.0) -> dict:
    """Repeat-search counters over chronological ``{"scan": bool, "pose": [...]}``.

    ``repeat_scans``    scan events issued from a cell that was already scanned
    ``region_revisits`` entries into a cell that was already visited (a return,
                        not hovering: only cell *transitions* count)
    """
    visited: set = set()
    scanned: set = set()
    current = None
    repeat_scans = region_revisits = scan_calls = 0
    for event in events:
        cell = _cell_of(event.get("pose"), grid_m)
        if cell is not None and cell != current:
            if cell in visited:
                region_revisits += 1
            visited.add(cell)
            current = cell
        if event.get("scan"):
            scan_calls += 1
            if cell is not None:
                if cell in scanned:
                    repeat_scans += 1
                scanned.add(cell)
    return {
        "scan_calls": scan_calls,
        "repeat_scans": repeat_scans,
        "region_revisits": region_revisits,
        "repeat_search_calls": repeat_scans + region_revisits,
        "unique_cells": len(visited),
    }


def path_length_m(poses: list) -> float | None:
    points = [p for p in poses if isinstance(p, (list, tuple)) and len(p) >= 3]
    if len(points) < 2:
        return None
    total = 0.0
    for previous, current in zip(points, points[1:]):
        total += math.dist(
            [float(previous[0]), float(previous[1]), float(previous[2])],
            [float(current[0]), float(current[1]), float(current[2])],
        )
    return round(total, 3)


# ---------------------------------------------------------------------------
# tasks, episodes and the initial-yaw groups
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    name: str
    instruction: str
    target_pos: tuple[float, float, float]
    target_name: str = "target"
    target_radius: float = 0.5
    start_pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
    success_dist: float | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "instruction": self.instruction,
            "target_name": self.target_name,
            "target_pos": list(self.target_pos),
            "target_radius": float(self.target_radius),
            "start_pos": list(self.start_pos),
            "success_dist": self.success_dist,
        }

    @classmethod
    def from_dict(cls, data: dict, default_success_dist: float | None = None) -> "TaskSpec":
        if "instruction" not in data:
            raise ValueError("task needs an 'instruction'")
        return cls(
            name=str(data.get("name") or data.get("task") or "task"),
            instruction=str(data["instruction"]),
            target_pos=tuple(float(v) for v in data.get("target_pos", (0.0, 0.0, 0.0))),
            target_name=str(data.get("target_name") or "target"),
            target_radius=float(data.get("target_radius", 0.5)),
            start_pos=tuple(float(v) for v in data.get("start_pos", (0.0, 0.0, 1.0))),
            success_dist=(
                float(data["success_dist"])
                if data.get("success_dist") is not None
                else default_success_dist
            ),
        )


@dataclass(frozen=True)
class EpisodeSpec:
    config: str
    task: TaskSpec
    seed: int
    yaw_mode: str
    start_yaw_deg: float

    @property
    def slug(self) -> str:
        return f"{self.yaw_mode}_seed{int(self.seed)}"

    @property
    def cell(self) -> str:
        """Identical for the four configurations of one paired episode."""
        return f"{self.task.name}__{self.yaw_mode}__seed{int(self.seed)}"

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "task": self.task.name,
            "seed": int(self.seed),
            "yaw_mode": self.yaw_mode,
            "start_yaw_deg": self.start_yaw_deg,
        }


def target_bearing_deg(target_pos, start_pos) -> float:
    dx = float(target_pos[0]) - float(start_pos[0])
    dy = float(target_pos[1]) - float(start_pos[1])
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


def start_yaw_deg(yaw_mode: str, task: TaskSpec, seed: int,
                  half_fov_deg: float = 40.0) -> float:
    """Deterministic start heading for one group of the visibility split.

    ``visible``: target bearing +- a seeded jitter inside the FOV.
    ``random``: seeded heading with the target kept outside the FOV, so the
                "initially invisible" class is really invisible.
    """
    bearing = target_bearing_deg(task.target_pos, task.start_pos)
    if yaw_mode == "target":
        return round(bearing, 3)
    if yaw_mode not in ("visible", "random"):
        raise ValueError(f"unknown yaw mode {yaw_mode!r}")
    rng = random.Random(int(seed) * 1000003 + 7919)
    if yaw_mode == "visible":
        offset = rng.uniform(-0.5, 0.5) * float(half_fov_deg)
        return round(_wrap_deg(bearing + offset), 3)
    margin = min(120.0, float(half_fov_deg) + 25.0)
    yaw = rng.uniform(0.0, 360.0)
    for _ in range(8):
        if abs(_wrap_deg(yaw - bearing)) > margin:
            return round(_wrap_deg(yaw), 3)
        yaw += 137.507764  # golden angle: deterministic re-draws
    return round(_wrap_deg(bearing + 180.0), 3)


def build_episodes(args: argparse.Namespace, tasks: list[TaskSpec]) -> list[EpisodeSpec]:
    episodes: list[EpisodeSpec] = []
    for task in tasks:
        for seed in args.seed_list:
            for yaw_mode in args.yaw_mode_list:
                yaw = start_yaw_deg(yaw_mode, task, seed, args.initial_yaw_half_fov_deg)
                for config in args.config_list:
                    episodes.append(
                        EpisodeSpec(config, task, seed, yaw_mode, yaw)
                    )
    return episodes


def episode_dir(out_root: Path, spec: EpisodeSpec) -> Path:
    return out_root / spec.config / spec.task.name / spec.slug


# ---------------------------------------------------------------------------
# command construction (the two runners' CLIs are the contract)
# ---------------------------------------------------------------------------


def scene_args(args: argparse.Namespace, spec: EpisodeSpec) -> list[str]:
    """Scene/start/seed options shared by both runners, in a stable order."""
    task = spec.task
    out: list[str] = ["--world", str(args.world)]
    if args.ply:
        out += ["--ply", str(args.ply)]
    if args.scene_scale is not None:
        out += ["--scene-scale", str(args.scene_scale)]
    if args.min_opacity is not None:
        out += ["--min-opacity", str(args.min_opacity)]
    if args.combined_ply:
        out += ["--combined-ply", str(args.combined_ply)]
    if args.background_ply:
        out += ["--background-ply", str(args.background_ply)]
    if args.semantic_manifest:
        out += ["--semantic-manifest", str(args.semantic_manifest)]
    if args.voxel_size is not None:
        out += ["--voxel-size", str(args.voxel_size)]
    out += ["--target-name", str(task.target_name)]
    out += ["--target-pos", *(str(float(v)) for v in task.target_pos)]
    out += ["--target-radius", str(float(task.target_radius))]
    out += ["--start-pos", *(str(float(v)) for v in task.start_pos)]
    out += ["--start-yaw-deg", str(spec.start_yaw_deg)]
    out += ["--success-dist", str(float(success_dist(args, task)))]
    out += ["--seed", str(int(spec.seed))]
    out += ["--initial-yaw-mode", spec.yaw_mode if spec.yaw_mode in ("visible", "random")
            else "target"]
    out += ["--max-init-attempts", str(int(args.max_init_attempts))]
    if args.target_index is not None:
        out += ["--target-index", str(int(args.target_index))]
    if args.world == "super":
        if args.worker_python:
            out += ["--worker-python", str(args.worker_python)]
        out += ["--map-mode", str(args.map_mode)]
        out += ["--sensing-range-m", str(float(args.sensing_range_m))]
        out += ["--sensing", str(args.sensing)]
        if args.moge_model_path:
            out += ["--moge-model-path", str(args.moge_model_path)]
    if args.env_overrides:
        out += ["--env-overrides", *(str(v) for v in args.env_overrides)]
    if args.device:
        out += ["--device", str(args.device)]
    return out


def success_dist(args: argparse.Namespace, task: TaskSpec) -> float:
    return float(task.success_dist if task.success_dist is not None else args.success_dist)


def model_args(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    if args.model_path:
        out += ["--model-path", str(args.model_path)]
    if args.model:
        out += ["--model", str(args.model)]
    if args.base_url:
        out += ["--base-url", str(args.base_url)]
    if args.max_new_tokens is not None:
        out += ["--max-new-tokens", str(int(args.max_new_tokens))]
    if args.image_size:
        out += ["--image-size", str(int(args.image_size[0])), str(int(args.image_size[1]))]
    if args.num_threads is not None:
        out += ["--num-threads", str(int(args.num_threads))]
    if args.no_think:
        out += ["--no-think"]
    return out


def tool_budget(args: argparse.Namespace) -> int:
    return int(args.tool_budget if args.tool_budget is not None else args.max_vlm_calls)


def budget_mapping(args: argparse.Namespace) -> dict:
    """Same VLM-call budget expressed in each runner's own knobs."""
    return {
        "max_vlm_calls": int(args.max_vlm_calls),
        "action": {"max_steps": int(args.max_vlm_calls)},
        "tool": {
            "max_turns": int(args.max_vlm_calls),
            "tool_budget": tool_budget(args),
        },
        "note": (
            "the action harness may spend extra model calls on scans and the "
            "done-verification crop, the tool loop one call per turn; the actual "
            "per-episode vlm_calls is recorded"
        ),
    }


def action_backend(args: argparse.Namespace) -> str:
    if args.action_backend:
        return str(args.action_backend)
    backend = str(args.backend)
    if backend == "mock":  # the tool loop speaks {"tool": ...}: use the oracle policy
        return "mock:oracle"
    return backend


def build_action_command(args: argparse.Namespace, spec: EpisodeSpec, out_dir: Path) -> list[str]:
    cmd = [sys.executable, "-m", "scripts.run_vlm_nav_3dgs"]
    cmd += scene_args(args, spec)
    cmd += ["--instruction", spec.task.instruction]
    cmd += ["--backend", action_backend(args)]
    cmd += model_args(args)
    cmd += ["--max-steps", str(int(args.max_vlm_calls))]
    if args.context_frames is not None:
        cmd += ["--context-frames", str(int(args.context_frames))]
    if args.scan_views is not None:
        cmd += ["--scan-views", str(int(args.scan_views))]
    if args.scan_pitch_deg is not None:
        cmd += ["--scan-pitch-deg", str(float(args.scan_pitch_deg))]
    if args.action_no_look_around:
        cmd += ["--no-look-around"]
    if args.action_no_bev:
        cmd += ["--no-bev"]
    if not args.action_video:
        cmd += ["--no-video"]
    cmd += [str(extra) for extra in (args.action_arg or [])]
    cmd += ["--out", str(out_dir)]
    return cmd


def build_tool_command(args: argparse.Namespace, spec: EpisodeSpec, out_dir: Path) -> list[str]:
    cmd = [sys.executable, "-m", "scripts.run_vlm_nav_tools"]
    cmd += scene_args(args, spec)
    if args.render_size:
        cmd += ["--render-size", str(int(args.render_size[0])), str(int(args.render_size[1]))]
    cmd += ["--instruction", spec.task.instruction]
    cmd += ["--backend", str(args.tool_backend or args.backend)]
    cmd += model_args(args)
    cmd += ["--toolset", CONFIG_TOOLSETS[spec.config]]
    cmd += ["--max-turns", str(int(args.max_vlm_calls))]
    cmd += ["--tool-budget", str(tool_budget(args))]
    if args.context_frames is not None:
        cmd += ["--context-frames", str(int(args.context_frames))]
    if args.scan_views is not None:
        cmd += ["--scan-views", str(int(args.scan_views))]
    if args.scan_pitch_deg is not None:
        cmd += ["--scan-pitch-deg", str(float(args.scan_pitch_deg))]
    cmd += [str(extra) for extra in (args.tool_arg or [])]
    cmd += ["--out", str(out_dir)]
    return cmd


def build_command(args: argparse.Namespace, spec: EpisodeSpec, out_dir: Path) -> list[str]:
    if spec.config == ACTION_CONFIG:
        return build_action_command(args, spec, out_dir)
    return build_tool_command(args, spec, out_dir)


# ---------------------------------------------------------------------------
# episode execution
# ---------------------------------------------------------------------------


def runner_env() -> dict:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing if existing else "")
    return env


def run_command(cmd: list[str], timeout: float, log_path: Path) -> dict:
    """Run one episode; stdout/stderr are tee'd to ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with open(log_path, "w") as handle:
        handle.write("$ " + " ".join(str(part) for part in cmd) + "\n\n")
        handle.flush()
        try:
            proc = subprocess.run(
                cmd, cwd=str(ROOT), env=runner_env(), stdout=handle,
                stderr=subprocess.STDOUT, timeout=float(timeout), text=True,
            )
            code = int(proc.returncode)
            error = None if code == 0 else f"runner exited with code {code}"
        except subprocess.TimeoutExpired:
            code = -1
            error = f"timeout after {timeout:.0f}s"
        except OSError as exc:
            code = -1
            error = f"{type(exc).__name__}: {exc}"
    return {"returncode": code, "error": error, "elapsed_s": round(time.time() - started, 3)}


def _row_skeleton(args, spec: EpisodeSpec, out_dir: Path, cmd: list[str]) -> dict:
    return {
        "config": spec.config,
        "config_label": CONFIG_LABELS.get(spec.config, spec.config),
        "toolset": CONFIG_TOOLSETS.get(spec.config),
        "task": spec.task.name,
        "instruction": spec.task.instruction,
        "seed": int(spec.seed),
        "yaw_mode": spec.yaw_mode,
        "start_yaw_deg": spec.start_yaw_deg,
        "out_dir": str(out_dir),
        "command": [str(part) for part in cmd],
        "returncode": None,
        "elapsed_s": None,
        "skipped": False,
        "error": None,
        "success": False,
        "finish_reason": None,
        "vlm_calls": None,
        "vlm_budget": int(args.max_vlm_calls),
        "vlm_calls_over_budget": None,
        "tool_calls": None,
        "steps": None,
        "wall_time_s": None,
        "final_distance_m": None,
        "success_dist_m": success_dist(args, spec.task),
        "path_length_m": None,
        "collisions": None,
        "parse_failures": None,
        "initial_target_visible": None,
        "images_total": None,
        "text_chars": None,
        "tokens_estimate": None,
        "scan_calls": None,
        "repeat_scans": None,
        "region_revisits": None,
        "repeat_search_calls": None,
        "unique_cells": None,
        "cached_calls": None,
        "new_evidence": None,
        "tools_used": None,
        "toolset_detail": None,
        "image_tokens": None,
        "chars_per_token": None,
        "prompt_chars_per_call": None,
    }


def _is_scan_summary(summary: str) -> bool:
    return bool(_SCAN_RE.search(str(summary or "")))


def image_tokens(args: argparse.Namespace) -> int:
    """Image token estimate for one VLM input image (28x28 patch by default)."""
    width, height = (args.image_size or (640, 360))
    divisor = float(args.image_token_divisor)
    if divisor <= 0:
        return 0
    return int(round(float(width) * float(height) / divisor))


def _token_model(args, row: dict) -> None:
    row["image_tokens"] = image_tokens(args)
    row["chars_per_token"] = float(args.chars_per_token)
    row["prompt_chars_per_call"] = float(args.prompt_chars_per_call)
    images = _as_int(row.get("images_total")) or 0
    text_chars = _as_int(row.get("text_chars")) or 0
    per_token = float(args.chars_per_token) or 4.0
    row["tokens_estimate"] = int(round(
        images * float(row["image_tokens"]) + text_chars / per_token
    ))


def extract_action_metrics(args, spec: EpisodeSpec, out_dir: Path, row: dict) -> dict:
    """Metrics of ``scripts/run_vlm_nav_3dgs.py`` (summary.json + episode_trace)."""
    summary = read_json(out_dir / SUMMARY_FILE) or {}
    trace = _find_trace(out_dir, ACTION_TRACE)
    look_around = not args.action_no_look_around
    bev = not args.action_no_bev
    scan_views = int(args.scan_views or 8)
    context_frames = int(args.context_frames if args.context_frames is not None else 3)
    pitch = float(args.scan_pitch_deg if args.scan_pitch_deg is not None else 50.0)

    steps = _as_int(summary.get("steps"))
    if steps is None:
        steps = len(trace)
    scan_steps = sum(1 for entry in trace if _is_scan_summary(entry.get("action_summary", "")))
    scan_calls = scan_steps + (1 if (look_around and (trace or steps)) else 0)
    vlm_calls = _as_int(summary.get("vlm_calls"))
    if vlm_calls is None:
        vlm_calls = steps + scan_calls
    images_per_decision = context_frames + (1 if bev else 0)
    images_per_scan = scan_views + (2 if pitch > 0.0 else 0)
    decision_calls = max(0, vlm_calls - scan_calls)

    events: list[dict] = []
    if scan_calls:
        events.append({"scan": True, "pose": (trace[0].get("position_flu") if trace else None)})
    for entry in trace:
        events.append(
            {
                "scan": _is_scan_summary(entry.get("action_summary", "")),
                "pose": entry.get("position_flu"),
            }
        )
    accounting = search_accounting(events, grid_m=float(args.grid_m))

    row.update(accounting)
    row.update(
        {
            "success": bool(summary.get("success")),
            "finish_reason": summary.get("finish_reason"),
            "vlm_calls": vlm_calls,
            "tool_calls": None,
            "steps": steps,
            "wall_time_s": _as_float(summary.get("wall_time_s")),
            "final_distance_m": _as_float(summary.get("final_distance_m")),
            "path_length_m": _as_float(summary.get("path_length_m")),
            "collisions": _as_int(summary.get("collision_hits")),
            "parse_failures": _as_int(summary.get("parse_failures")),
            "initial_target_visible": summary.get("initial_target_visible"),
            "images_total": decision_calls * images_per_decision + scan_calls * images_per_scan,
            "text_chars": int(sum(len(str(entry.get("raw_response") or "")) for entry in trace)
                              + float(args.prompt_chars_per_call) * vlm_calls),
            "cached_calls": None,
            "new_evidence": None,
            "skipped": bool(summary.get("skipped")),
        }
    )
    tools: dict[str, int] = {}
    for entry in trace:
        key = str(entry.get("action_summary") or "").split(" ")[0] or "unknown"
        tools[key] = tools.get(key, 0) + 1
    row["tools_used"] = tools or None
    return row


def extract_tool_metrics(args, spec: EpisodeSpec, out_dir: Path, row: dict) -> dict:
    """Metrics of ``scripts/run_vlm_nav_tools.py`` (summary.json + tool_trace)."""
    summary = read_json(out_dir / SUMMARY_FILE) or {}
    trace = _find_trace(out_dir, TOOL_TRACE)
    context_frames = int(args.context_frames if args.context_frames is not None else 2)

    tool_names: dict[str, int] = {}
    scan_calls = cached_calls = images_total = text_chars = 0
    events: list[dict] = []
    poses: list[list[float]] = []
    for entry in trace:
        tool = str(entry.get("tool") or "")
        call_args = entry.get("args") if isinstance(entry.get("args"), dict) else {}
        status = str(entry.get("status") or "")
        if tool and tool not in ("finish", "(no tool call)", "None"):
            tool_names[tool] = tool_names.get(tool, 0) + 1
        if status == "cached":
            cached_calls += 1
        mode = str(call_args.get("mode") or "").lower()
        is_scan = tool == "observe" and mode in SCAN_MODES
        pose = _pose_of(entry)
        if pose is not None:
            poses.append(pose)
        events.append({"scan": is_scan, "pose": pose})
        images_total += _as_int(entry.get("images_sent")) or 0
        text_chars += len(str(entry.get("result_prompt") or ""))
        text_chars += len(str(entry.get("raw_reply") or entry.get("model_reply") or ""))

    turns = _as_int(summary.get("turns"))
    if turns is None:
        turns = len(trace)
    vlm_calls = _as_int(summary.get("vlm_calls"))
    if vlm_calls is None:
        vlm_calls = turns
    if not images_total:  # trace without images_sent: estimate from the runtime budget
        images_total = turns * (context_frames + 1)
    text_chars = int(text_chars + float(args.prompt_chars_per_call) * vlm_calls)
    tool_calls = _as_int(summary.get("tool_calls"))
    if tool_calls is None:
        tool_calls = sum(tool_names.values())

    accounting = search_accounting(events, grid_m=float(args.grid_m))
    row.update(accounting)
    row.update(
        {
            "success": bool(summary.get("success")),
            "finish_reason": summary.get("finish_reason"),
            "vlm_calls": vlm_calls,
            "tool_calls": tool_calls,
            "steps": turns,
            "wall_time_s": _as_float(summary.get("wall_time_s")),
            "final_distance_m": _as_float(summary.get("final_distance_m")),
            "path_length_m": path_length_m(poses),
            "collisions": None,
            "parse_failures": _as_int(summary.get("parse_failures")),
            "initial_target_visible": _initial_visible(summary, trace),
            "images_total": images_total,
            "text_chars": text_chars,
            "cached_calls": cached_calls or None,
            "new_evidence": _as_int(summary.get("new_evidence_total")),
            "toolset_detail": summary.get("toolset_detail"),
            "skipped": bool(summary.get("skipped")),
        }
    )
    summary_tools = summary.get("tools") if isinstance(summary.get("tools"), dict) else None
    if summary_tools:
        row["tools_used"] = {
            name: _as_int((stats or {}).get("calls")) or 0
            for name, stats in summary_tools.items()
        }
    else:
        row["tools_used"] = tool_names or None
    return row


def _initial_visible(summary: dict, trace: list[dict]):
    if "initial_target_visible" in summary:
        return bool(summary.get("initial_target_visible"))
    for entry in trace:
        value = _result_data(entry).get("target_in_fov")
        if isinstance(value, bool):
            return value
    return None


def extract_metrics(args, spec: EpisodeSpec, out_dir: Path, run: dict) -> dict:
    cmd = run.get("command") or build_command(args, spec, out_dir)
    row = _row_skeleton(args, spec, out_dir, cmd)
    row["returncode"] = run.get("returncode")
    row["error"] = run.get("error")
    row["elapsed_s"] = run.get("elapsed_s")
    if not (out_dir / SUMMARY_FILE).exists():
        if row["error"] is None:
            row["error"] = f"missing {SUMMARY_FILE} in {out_dir}"
        _token_model(args, row)
        return row
    try:
        if spec.config == ACTION_CONFIG:
            extract_action_metrics(args, spec, out_dir, row)
        else:
            extract_tool_metrics(args, spec, out_dir, row)
    except Exception as exc:  # a broken artifact must not lose the other configs
        row["error"] = f"metric extraction failed: {type(exc).__name__}: {exc}"
    _token_model(args, row)
    if row["vlm_calls"] is not None:
        row["vlm_calls_over_budget"] = max(0, int(row["vlm_calls"]) - int(args.max_vlm_calls))
    return row


# ---------------------------------------------------------------------------
# aggregation + report
# ---------------------------------------------------------------------------


def aggregate(rows: list[dict]) -> dict:
    ok = [row for row in rows if not row.get("error")]
    reasons: dict[str, int] = {}
    for row in ok:
        reason = str(row.get("finish_reason") or "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1
    summary = {
        "n": len(rows),
        "n_ok": len(ok),
        "n_errors": len(rows) - len(ok),
        "n_skipped": sum(1 for row in rows if row.get("skipped")),
        "n_success": sum(1 for row in ok if row.get("success")),
        "finish_reasons": reasons,
    }
    summary["success_rate"] = (
        round(summary["n_success"] / len(ok), 3) if ok else None
    )
    for key in (
        "vlm_calls", "tokens_estimate", "wall_time_s", "repeat_search_calls",
        "repeat_scans", "region_revisits", "cached_calls", "tool_calls", "steps",
        "images_total", "text_chars", "path_length_m", "final_distance_m",
        "vlm_calls_over_budget", "new_evidence", "unique_cells",
    ):
        values = [_as_float(row.get(key)) for row in ok]
        values = [value for value in values if value is not None]
        summary[f"{key}_mean"] = _mean(values)
        summary[f"{key}_sum"] = round(sum(values), 3) if values else None
    return summary


def grouping(rows: list[dict], yaw_modes: list[str]) -> list[tuple[str, str, list[dict]]]:
    """(config, group, rows) with one ``all`` group plus one per yaw mode."""
    out: list[tuple[str, str, list[dict]]] = []
    for config in CONFIG_ORDER:
        config_rows = [row for row in rows if row.get("config") == config]
        if not config_rows:
            continue
        out.append((config, "all", config_rows))
        for mode in yaw_modes:
            subset = [row for row in config_rows if row.get("yaw_mode") == mode]
            if subset:
                out.append((config, mode, subset))
    return out


def _fmt(value, digits: int = 2, missing: str = "-") -> str:
    if value is None:
        return missing
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _task_names(tasks) -> list[str]:
    return [str(task.get("name")) if isinstance(task, dict) else str(task) for task in tasks]


def render_markdown(meta: dict, rows: list[dict], groups: list[tuple[str, str, list[dict]]],
                    aggregates: dict) -> str:
    lines: list[str] = []
    lines.append("# VLM navigation agent - configuration comparison (goal.md 7)")
    lines.append("")
    lines.append(f"- created: {meta['created_at']}")
    lines.append(f"- git HEAD: {meta.get('git_head') or 'unknown'}")
    lines.append(f"- world: {meta['world']}" + (f" | scene: {meta['ply']}" if meta.get("ply") else ""))
    lines.append(f"- backend: {meta['backend']} | model: {meta.get('model_path') or meta.get('model') or '-'}")
    lines.append(f"- tasks: {', '.join(_task_names(meta['tasks']))} | seeds: {meta['seeds']} | "
                 f"initial-yaw groups: {meta['yaw_modes']}")
    lines.append(f"- VLM budget: {meta['budget']['max_vlm_calls']} calls "
                 f"(action --max-steps {meta['budget']['action']['max_steps']}; "
                 f"tools --max-turns {meta['budget']['tool']['max_turns']}, "
                 f"--tool-budget {meta['budget']['tool']['tool_budget']})")
    lines.append(f"- tokens_estimate = images x {meta['token_model']['image_tokens']} "
                 f"+ text_chars / {meta['token_model']['chars_per_token']:g} "
                 f"(+{meta['token_model']['prompt_chars_per_call']:.0f} prompt chars/call)")
    lines.append(f"- pairing check: {meta['pairing_check']['cells']} cells, "
                 f"consistent={meta['pairing_check']['consistent']}")
    lines.append("")
    lines.append("## Success and cost by configuration")
    lines.append("")
    header = (
        "| config | group | ep | success | vlm_calls | tokens_est | wall_s | "
        "repeat_search | repeat_scans | region_revisits | cached |"
    )
    lines.append(header)
    lines.append("|" + "---|" * 11)
    for config, group, subset in groups:
        stats = aggregates[f"{config}|{group}"]
        lines.append(
            f"| {config} | {group} | {stats['n']} | "
            f"{stats['n_success']}/{stats['n_ok']} ({_fmt(stats['success_rate'])}) | "
            f"{_fmt(stats['vlm_calls_mean'], 1)} | "
            f"{_fmt(stats['tokens_estimate_mean'], 0)} | "
            f"{_fmt(stats['wall_time_s_mean'], 1)} | "
            f"{_fmt(stats['repeat_search_calls_mean'], 2)} | "
            f"{_fmt(stats['repeat_scans_mean'], 2)} | "
            f"{_fmt(stats['region_revisits_mean'], 2)} | "
            f"{_fmt(stats['cached_calls_mean'], 2)} |"
        )
    lines.append("")
    lines.append("## Per-episode detail")
    lines.append("")
    lines.append("| config | task | yaw | seed | success | finish | vlm_calls | tool_calls | "
                 "tokens_est | wall_s | repeat_search | scan | revisit | notes |")
    lines.append("|" + "---|" * 14)
    for row in rows:
        notes = row.get("error") or ("skipped" if row.get("skipped") else "")
        lines.append(
            f"| {row['config']} | {row['task']} | {row['yaw_mode']} | {row['seed']} | "
            f"{int(bool(row.get('success')))} | {row.get('finish_reason') or '-'} | "
            f"{_fmt(row.get('vlm_calls'), 0)} | {_fmt(row.get('tool_calls'), 0)} | "
            f"{_fmt(row.get('tokens_estimate'), 0)} | {_fmt(row.get('wall_time_s'), 1)} | "
            f"{_fmt(row.get('repeat_search_calls'), 0)} | {_fmt(row.get('repeat_scans'), 0)} | "
            f"{_fmt(row.get('region_revisits'), 0)} | {notes} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Every configuration of a cell shares world, scene, target, start pose, "
                 "initial heading, seed and VLM budget; the pairing check compares the "
                 "scene-level CLI options of the four runners.")
    lines.append("- `repeat_scans`: scan-type calls (action loop: scan steps + the initial "
                 "look-around; tool loop: `observe(mode=lookaround)`) issued from a 2 m cell "
                 "that was already scanned.")
    lines.append("- `region_revisits`: entries into a 2 m cell that was already visited "
                 "(returns, not hovering).")
    lines.append("- `cached`: tool calls the runtime answered as `cached` (repeat call, no new "
                 "evidence) - tool configurations only.")
    lines.append("- `tokens_estimate` is a model-side estimate (image patch count + text "
                 "chars/4), not a tokenizer measurement.")
    lines.append("- `wall_time_s` is the runner's own episode clock; the directory also "
                 "holds `run.log` and the driver records `elapsed_s` (that plus process "
                 "start-up) per episode in `compare_summary.json`.")
    lines.append("")
    return "\n".join(lines)


def pairing_check(args, episodes: list[EpisodeSpec], commands: dict) -> dict:
    """Verify that the four configurations of a cell got identical paired options."""
    cells: dict[str, dict[str, dict]] = {}
    for spec in episodes:
        cells.setdefault(spec.cell, {})[spec.config] = _args_map(
            commands[(spec.config, spec.task.name, spec.seed, spec.yaw_mode)]
        )
    issues: list[dict] = []
    config_only: set[str] = set()
    for cell, per_config in cells.items():
        if len(per_config) < 2:
            continue
        shared_keys = None
        for options in per_config.values():
            keys = {key for key in PAIRING_KEYS if key in options}
            shared_keys = keys if shared_keys is None else (shared_keys & keys)
        for options in per_config.values():
            config_only |= {key for key in options if key in PAIRING_KEYS} - (shared_keys or set())
        reference_config, reference = next(iter(per_config.items()))
        for config, options in per_config.items():
            for key in sorted(shared_keys or ()):
                if options.get(key) != reference.get(key):
                    issues.append(
                        {
                            "cell": cell,
                            "option": key,
                            config: options.get(key),
                            reference_config: reference.get(key),
                        }
                    )
    return {
        "cells": len(cells),
        "consistent": not issues,
        "issues": issues[:20],
        "config_only_options": sorted(config_only),
    }


def _args_map(cmd: list[str]) -> dict:
    out: dict[str, Any] = {}
    index = 0
    while index < len(cmd):
        token = str(cmd[index])
        if token.startswith("--"):
            if index + 1 < len(cmd) and not str(cmd[index + 1]).startswith("--"):
                out[token] = str(cmd[index + 1])
                index += 1
            else:
                out[token] = True
        index += 1
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # --- matrix ---
    parser.add_argument("--configs", type=str, default=",".join(CONFIG_ORDER),
                        help="comma list of " + ",".join(CONFIG_ORDER))
    parser.add_argument("--seeds", type=str, default="7,11")
    parser.add_argument("--initial-yaw-modes", "--initial-yaw-mode", type=str,
                        default="visible,random", dest="initial_yaw_modes",
                        help="comma list of visible,random,target (goal.md: keep both "
                             "the initially visible and invisible task classes)")
    parser.add_argument("--max-vlm-calls", type=int, default=20,
                        help="model-call budget per episode (action: --max-steps, "
                             "tools: --max-turns)")
    parser.add_argument("--tool-budget", type=int, default=None,
                        help="tool-call cap for the tool loops (default: --max-vlm-calls)")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--timeout", type=float, default=1800.0,
                        help="per-episode subprocess timeout in seconds")
    parser.add_argument("--jobs", type=int, default=1,
                        help="episodes to run in parallel (each is its own process; "
                             "keep 1 for GPU/model runs, raise it for mock/CPU sweeps)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands without running them")
    parser.add_argument("--initial-yaw-half-fov-deg", type=float, default=40.0,
                        help="half horizontal FOV used to build the visibility groups")
    parser.add_argument("--grid-m", type=float, default=2.0,
                        help="cell size for the repeat-search accounting")
    # --- task(s) ---
    parser.add_argument("--task-file", type=str, default=None,
                        help="JSON list of tasks: {name, instruction, target_pos, ...}")
    parser.add_argument("--task-json", type=str, action="append", default=None,
                        help="one task as a JSON object (repeatable)")
    parser.add_argument("--task-name", type=str, default="task0")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--target-name", type=str, default="target")
    parser.add_argument("--target-pos", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"), help="FLU, scene-centered")
    parser.add_argument("--target-radius", type=float, default=0.5)
    parser.add_argument("--start-pos", type=float, nargs=3, default=(0.0, 0.0, 1.0))
    parser.add_argument("--success-dist", type=float, default=2.0,
                        help="meters from the target surface")
    # --- scene (mirrors both runners) ---
    parser.add_argument("--world", choices=["cpu", "higs", "env", "super"], default="cpu")
    parser.add_argument("--ply", type=str, default=None)
    parser.add_argument("--scene-scale", type=float, default=None)
    parser.add_argument("--min-opacity", type=float, default=None)
    parser.add_argument("--combined-ply", type=str, default=None)
    parser.add_argument("--background-ply", type=str, default=None)
    parser.add_argument("--semantic-manifest", type=str, default=None)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--render-size", type=int, nargs=2, default=None,
                        metavar=("W", "H"),
                        help="camera resolution for the tool runner (the action runner's "
                             "CPU world is fixed at 320x180)")
    parser.add_argument("--target-index", type=int, default=None)
    parser.add_argument("--max-init-attempts", type=int, default=8)
    parser.add_argument("--worker-python", type=str, default=None)
    parser.add_argument("--map-mode", type=str, default="global",
                        choices=["global", "accumulate", "blind"])
    parser.add_argument("--sensing-range-m", type=float, default=6.0)
    parser.add_argument("--sensing", type=str, default="ideal", choices=["ideal", "moge"])
    parser.add_argument("--moge-model-path", type=str, default=None)
    parser.add_argument("--env-overrides", type=str, nargs="*", default=None)
    # --- model ---
    parser.add_argument("--backend", type=str, default="mock",
                        help="mock | mock:oracle | hf | openai")
    parser.add_argument("--action-backend", type=str, default=None,
                        help="override the backend of the action loop (default: mock "
                             "-> mock:oracle, otherwise --backend)")
    parser.add_argument("--tool-backend", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--image-size", type=int, nargs=2, default=(640, 360),
                        metavar=("W", "H"), help="VLM input resize (both runners)")
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--no-think", action="store_true")
    # --- agent loop / observability parity ---
    parser.add_argument("--context-frames", type=int, default=2,
                        help="frames per decision (both runners)")
    parser.add_argument("--scan-views", type=int, default=8)
    parser.add_argument("--scan-pitch-deg", type=float, default=50.0)
    parser.add_argument("--action-no-look-around", action="store_true",
                        help="action loop only: skip the initial 360-degree scan")
    parser.add_argument("--action-no-bev", action="store_true",
                        help="action loop only: disable the top-down map image")
    parser.add_argument("--action-video", action="store_true",
                        help="action loop only: also write episode.mp4 (default off)")
    parser.add_argument("--action-arg", type=str, action="append", default=None,
                        help="extra flag appended verbatim to the action runner")
    parser.add_argument("--tool-arg", type=str, action="append", default=None,
                        help="extra flag appended verbatim to the tool runner")
    # --- metric model ---
    parser.add_argument("--image-token-divisor", type=float, default=784.0,
                        help="pixels per image token (28x28 patch)")
    parser.add_argument("--chars-per-token", type=float, default=4.0)
    parser.add_argument("--prompt-chars-per-call", type=float, default=1200.0,
                        help="per-call prompt overhead used in the text-token estimate")
    return parser.parse_args(argv)


def resolve_matrix(args: argparse.Namespace) -> None:
    """Validate and expand the comma lists into attributes the driver uses."""
    configs = _str_list(args.configs)
    unknown = [name for name in configs if name not in CONFIG_ORDER]
    if unknown:
        raise SystemExit(f"unknown --configs entries {unknown}; pick from {CONFIG_ORDER}")
    args.config_list = [name for name in CONFIG_ORDER if name in configs]
    args.seed_list = _int_list(args.seeds) or [0]
    modes = _str_list(args.initial_yaw_modes) or ["visible", "random"]
    unknown_modes = [mode for mode in modes if mode not in YAW_MODES]
    if unknown_modes:
        raise SystemExit(f"unknown --initial-yaw-modes entries {unknown_modes}")
    args.yaw_mode_list = modes
    if args.max_vlm_calls <= 0:
        raise SystemExit("--max-vlm-calls must be positive")


def load_tasks(args: argparse.Namespace) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    if args.task_file:
        data = read_json(Path(args.task_file))
        if data is None:
            with open(args.task_file) as handle:
                data = json.load(handle)
        if isinstance(data, dict):
            data = data.get("tasks") or [data]
        for entry in data or []:
            tasks.append(TaskSpec.from_dict(entry, args.success_dist))
    for blob in (args.task_json or []):
        tasks.append(TaskSpec.from_dict(json.loads(blob), args.success_dist))
    if not tasks:
        if not (args.instruction and args.target_pos):
            raise SystemExit(
                "give --instruction with --target-pos, or --task-file/--task-json"
            )
        tasks.append(
            TaskSpec(
                name=args.task_name,
                instruction=args.instruction,
                target_pos=tuple(float(v) for v in args.target_pos),
                target_name=args.target_name,
                target_radius=float(args.target_radius),
                start_pos=tuple(float(v) for v in args.start_pos),
                success_dist=float(args.success_dist),
            )
        )
    if args.world in ("cpu", "higs") and not args.ply:
        raise SystemExit("--world cpu/higs requires --ply")
    if args.backend.startswith("hf") and not args.model_path:
        raise SystemExit("--backend hf requires --model-path")
    if args.backend == "openai" and not args.model:
        raise SystemExit("--backend openai requires --model")
    return tasks


def git_head() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True,
            text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() or None if proc.returncode == 0 else None


def meta_block(args: argparse.Namespace, tasks: list[TaskSpec], episodes: list[EpisodeSpec],
               commands: dict, aggregates: dict) -> dict:
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_head": git_head(),
        "world": args.world,
        "ply": args.ply,
        "backend": args.backend,
        "action_backend": action_backend(args),
        "tool_backend": str(args.tool_backend or args.backend),
        "model_path": args.model_path,
        "model": args.model,
        "device": args.device,
        "seeds": args.seed_list,
        "yaw_modes": args.yaw_mode_list,
        "tasks": [task.to_dict() for task in tasks],
        "configs": args.config_list,
        "n_episodes": len(episodes),
        "budget": budget_mapping(args),
        "token_model": {
            "image_tokens": image_tokens(args),
            "chars_per_token": float(args.chars_per_token),
            "prompt_chars_per_call": float(args.prompt_chars_per_call),
        },
        "grid_m": float(args.grid_m),
        "out": str(Path(args.out)),
        "pairing_check": pairing_check(args, episodes, commands),
        "aggregates": aggregates,
    }


def run_comparison(args: argparse.Namespace) -> dict:
    tasks = load_tasks(args)
    episodes = build_episodes(args, tasks)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    commands: dict = {}
    for spec in episodes:
        dir_path = episode_dir(out_root, spec)
        commands[(spec.config, spec.task.name, spec.seed, spec.yaw_mode)] = build_command(
            args, spec, dir_path
        )

    if args.dry_run:
        for spec in episodes:
            print(" ".join(
                str(part) for part in
                commands[(spec.config, spec.task.name, spec.seed, spec.yaw_mode)]
            ))
        return {"dry_run": True, "n_episodes": len(episodes)}

    rows_by_index: dict[int, dict] = {}
    started = time.time()
    jobs = max(1, int(args.jobs))

    def execute(index: int, spec: EpisodeSpec) -> tuple[int, dict]:
        dir_path = episode_dir(out_root, spec)
        dir_path.mkdir(parents=True, exist_ok=True)
        cmd = commands[(spec.config, spec.task.name, spec.seed, spec.yaw_mode)]
        print(f"[compare] {index + 1}/{len(episodes)} {spec.config} "
              f"{spec.task.name} {spec.slug}", flush=True)
        run = run_command(cmd, float(args.timeout), dir_path / "run.log")
        run["command"] = cmd
        row = extract_metrics(args, spec, dir_path, run)
        print(f"[compare]   -> rc={row['returncode']} success={row.get('success')} "
              f"vlm_calls={row.get('vlm_calls')} tokens={row.get('tokens_estimate')} "
              f"repeat_search={row.get('repeat_search_calls')}"
              + (f" error={row['error']}" if row.get("error") else ""), flush=True)
        return index, row

    if jobs == 1 or len(episodes) <= 1:
        for index, spec in enumerate(episodes):
            key, row = execute(index, spec)
            rows_by_index[key] = row
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(execute, index, spec)
                       for index, spec in enumerate(episodes)]
            for future in futures:
                key, row = future.result()
                rows_by_index[key] = row
    rows = [rows_by_index[index] for index in range(len(episodes))]

    groups = grouping(rows, args.yaw_mode_list)
    aggregates = {f"{config}|{group}": aggregate(subset) for config, group, subset in groups}
    overall = aggregate(rows)
    meta = meta_block(args, tasks, episodes, commands, aggregates)
    summary = {
        "meta": meta,
        "overall": overall,
        "aggregates": aggregates,
        "episodes": rows,
        "n_errors": overall["n_errors"],
        "wall_time_total_s": round(time.time() - started, 3),
    }
    with open(out_root / "compare_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    with open(out_root / "episodes.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    with open(out_root / "compare_table.md", "w") as handle:
        handle.write(render_markdown(meta, rows, groups, aggregates))
    with open(out_root / "run_config.json", "w") as handle:
        json.dump(vars(args), handle, indent=2, default=str)
    print(f"[compare] wrote {out_root / 'compare_summary.json'} and "
          f"{out_root / 'compare_table.md'}", flush=True)
    for key, stats in aggregates.items():
        config, _, group = key.partition("|")
        print(f"[compare] {config:12s} {group:8s} n={stats['n']:<3d} "
              f"success={stats['n_success']}/{stats['n_ok']:<3d} "
              f"vlm_calls={_fmt(stats['vlm_calls_mean'], 1):>5s} "
              f"tokens={_fmt(stats['tokens_estimate_mean'], 0):>6s} "
              f"wall_s={_fmt(stats['wall_time_s_mean'], 1):>5s} "
              f"repeat_search={_fmt(stats['repeat_search_calls_mean'], 2)}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    resolve_matrix(args)
    try:
        summary = run_comparison(args)
    except SystemExit:
        raise
    if summary.get("dry_run"):
        return 0
    errors = int(summary.get("n_errors") or 0)
    if errors:
        print(f"[compare] {errors} episode(s) failed; see compare_summary.json", flush=True)
        return 1
    print(f"[compare] all {summary['overall']['n']} episodes completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
