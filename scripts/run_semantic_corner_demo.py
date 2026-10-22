#!/usr/bin/env python
"""One-prompt semantic corner/edge-object demo.

Pipeline (the same tools the agent uses):

    search_target -> approach_candidate loop -> confirm_candidate ->
    verify_goal -> finish

It writes:

- a first-person MP4 assembled from every observation/action frame,
- an XY scene map (coloured 3DGS points) with the flown trajectory and the
  grounded candidate,
- ``summary.json`` with the tool verdicts and evidence.

The world can be the portable CPU renderer or DiffAero HiGS. With HiGS, run
this script inside the GPU container with ``PYTHONPATH`` pointing at the
DiffAero shim/repo; with CPU no GPU is needed.

    python scripts/run_semantic_corner_demo.py --renderer cpu \
      --ply /assets/scenes/bedroom.ply ... --out outputs/vlm_nav/semantic_demo

    # inside the GPU container:
    python scripts/run_semantic_corner_demo.py --renderer higs --device cuda:0 \
      --base-url http://172.17.0.1:8000/v1 --out outputs/vlm_nav/semantic_demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from vlm_nav.tool_impl_navigation import register_navigation_tools  # noqa: E402
from vlm_nav.tool_impl_perception import register_perception_tools  # noqa: E402
from vlm_nav.tools import STATUS_FAILED, ToolContext, ToolRegistry  # noqa: E402
from vlm_nav.vlm import OpenAIVLM  # noqa: E402
from vlm_nav.world import SplatWorldCPU, SplatWorldHiGS, TargetSpec  # noqa: E402

DEFAULT_PROMPT = (
    "Find the mirror on the white dressing table at the right side of the "
    "bedroom and fly next to it."
)
DEFAULT_START = (0.2772, -0.0595, 0.1254)
DEFAULT_PLY = "/assets/scenes/bedroom.ply"


def _frame_pil(frame: Any) -> Image.Image:
    from vlm_nav.vlm import frame_to_pil

    return frame_to_pil(frame).convert("RGB")


def _overlay(image: Image.Image, label: str, clock: float, index: int) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    bar = max(22, height // 12)
    draw.rectangle((0, 0, width, bar), fill=(0, 0, 0))
    draw.text((8, 4), f"{label}   t={clock:.2f}s   frame {index:04d}", fill=(255, 255, 255))
    return canvas


def _title(width: int, height: int, prompt: str) -> Image.Image:
    image = Image.new("RGB", (width, height), (8, 12, 20))
    draw = ImageDraw.Draw(image)
    draw.text((24, 24), "DroneNavAgent - one-prompt semantic corner-object demo", fill=(255, 255, 255))
    draw.text((24, 60), "Prompt:", fill=(180, 200, 230))
    words = prompt.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > 78:
            lines.append(current)
            current = word
        else:
            current = (current + " " + word).strip()
    if current:
        lines.append(current)
    y = 84
    for line in lines[:4]:
        draw.text((24, y), line, fill=(220, 230, 245))
        y += 22
    draw.text((24, y + 16), "search -> approach -> confirm -> verify -> finish", fill=(255, 220, 80))
    return image


def _scene_colors(scene: Any) -> np.ndarray | None:
    colors = getattr(scene, "precomputed_colors", None)
    if colors is not None:
        arr = colors.detach().float().cpu().numpy() if hasattr(colors, "detach") else np.asarray(colors)
    else:
        arr = getattr(scene, "color", None)
        if arr is None:
            return None
        arr = arr.detach().float().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    return np.clip(arr[:, :3], 0.0, 1.0)


def _xy_map(
    scene: Any,
    trajectory: list,
    candidate_position: Any,
    out_path: Path,
    *,
    prompt: str,
    width: int,
    height: int,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = scene.xyz
    xyz = xyz.detach().float().cpu().numpy() if hasattr(xyz, "detach") else np.asarray(xyz)
    colors = _scene_colors(scene)
    if colors is not None and len(colors) != len(xyz):
        colors = None
    count = len(xyz)
    step = max(1, count // 160000)
    xyz = xyz[::step]
    if colors is not None:
        colors = colors[::step]
    if len(xyz) == 0:
        return {"error": "scene has no points"}
    # Remove a thin floor/ceiling layer so furniture is visible from above.
    z = xyz[:, 2]
    z_min, z_max = float(np.percentile(z, 2)), float(np.percentile(z, 98))
    keep = (z > z_min + 0.05) & (z < z_max - 0.05)
    if keep.sum() < 100:
        keep = np.ones(len(xyz), dtype=bool)
    points = xyz[keep]
    point_colors = colors[keep] if colors is not None else None
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    ax.set_facecolor("#0b1018")
    fig.patch.set_facecolor("#0b1018")
    scatter_kwargs = {"s": 1.2, "alpha": 0.5, "linewidths": 0}
    if point_colors is not None:
        scatter_kwargs["c"] = point_colors
    else:
        scatter_kwargs["c"] = "#9fb0c8"
    ax.scatter(points[:, 0], points[:, 1], **scatter_kwargs)
    traj = np.asarray(trajectory, dtype=float).reshape(-1, 3) if trajectory else np.zeros((0, 3))
    if len(traj):
        ax.plot(traj[:, 0], traj[:, 1], color="#ffd34d", lw=2.5, label="flown trajectory")
        ax.scatter(traj[0, 0], traj[0, 1], c="#48e08b", s=80, marker="o", label="start")
        ax.scatter(traj[-1, 0], traj[-1, 1], c="#ff6b8a", s=80, marker="X", label="end")
        # heading ticks
        stride = max(1, len(traj) // 16)
        for index in range(0, len(traj) - 1, stride):
            ax.annotate(
                "", xy=(traj[index + 1, 0], traj[index + 1, 1]),
                xytext=(traj[index, 0], traj[index, 1]),
                arrowprops={"arrowstyle": "->", "color": "#ffd34d", "lw": 1.0},
            )
    if candidate_position is not None:
        cand = np.asarray(candidate_position, dtype=float).reshape(-1)
        if cand.size >= 2:
            ax.scatter(cand[0], cand[1], c="#ff4df0", s=140, marker="*",
                       edgecolors="white", linewidths=0.6, label="grounded candidate")
            ax.annotate(
                "grounded candidate", xy=(cand[0], cand[1]),
                xytext=(8, 8), textcoords="offset points",
                color="#ffb3f7", fontsize=8,
            )
    ax.set_title(
        "XY scene map (3DGS points coloured by RGB) + trajectory\n" + prompt[:110],
        color="white", fontsize=10,
    )
    ax.set_xlabel("x (m)", color="white")
    ax.set_ylabel("y (m)", color="white")
    ax.tick_params(colors="white", labelsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(color="#2a3648", lw=0.4, alpha=0.5)
    legend = ax.legend(loc="upper right", fontsize=8, facecolor="#121a26", edgecolor="#4a5a70")
    for text in legend.get_texts():
        text.set_color("white")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {
        "path": str(out_path),
        "scene_points": int(len(points)),
        "trajectory_points": int(len(traj)),
        "candidate_position": None if candidate_position is None else list(candidate_position),
    }


def _result_data(result: Any, keys: tuple[str, ...]) -> dict:
    data = getattr(result, "data", {}) or {}
    return {key: data.get(key) for key in keys if key in data}


def run_demo(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for path in frames_dir.glob("frame_*.png"):
        path.unlink()
    target = TargetSpec("candidate", tuple(args.target_pos), radius_m=0.3)
    if args.renderer == "higs":
        world = SplatWorldHiGS(
            args.ply, target,
            start_flu=tuple(args.start_pos), start_yaw_deg=args.start_yaw_deg,
            width=args.width, height=args.height, hfov_deg=86.0,
            bound_radius=100.0, z_min=-20.0, z_max=20.0,
            device=args.device,
        )
    else:
        world = SplatWorldCPU(
            args.ply, target,
            start_flu=tuple(args.start_pos), start_yaw_deg=args.start_yaw_deg,
            width=args.width, height=args.height, hfov_deg=86.0,
            bound_radius=100.0, z_min=-20.0, z_max=20.0,
        )
    vlm = OpenAIVLM(
        model=args.model, base_url=args.base_url,
        image_size=(args.width, args.height),
        max_new_tokens=args.max_new_tokens,
        timeout_s=args.timeout_s,
    )
    registry = ToolRegistry()
    register_perception_tools(registry, vlm=vlm)
    register_navigation_tools(registry, vlm=vlm)
    ctx = ToolContext(
        world=world,
        episode_id="semantic_corner_demo",
        budget={"tool_calls_left": args.tool_budget},
        extra={
            "instruction": args.prompt,
            "done_dist": args.success_dist,
            "scan_views": args.scan_views,
        },
    )
    phase = {"name": "start"}
    frames: list[Image.Image] = []
    frame_meta: list[dict] = []
    clock_start = time.time()
    max_frames = args.max_frames

    def capture(frame: Any, label: str) -> None:
        if len(frames) >= max_frames:
            return
        image = _frame_pil(frame)
        index = len(frames)
        if (image.width, image.height) != (args.width, args.height):
            image = image.resize((args.width, args.height), Image.BILINEAR)
        stamped = _overlay(image, label, time.time() - clock_start, index)
        stamped.save(frames_dir / f"frame_{index:04d}.png")
        frames.append(stamped)
        frame_meta.append({"index": index, "phase": label, "clock_s": round(time.time() - clock_start, 3)})

    original_render = world.render
    original_apply = world.apply_action

    def render_and_capture():
        frame = original_render()
        capture(frame, phase["name"])
        return frame

    def apply_and_capture(action):
        result = original_apply(action)
        try:
            capture(original_render(), f"{phase['name']} / move")
        except Exception:
            pass
        return result

    world.render = render_and_capture
    world.apply_action = apply_and_capture
    title = _title(args.width, args.height, args.prompt)
    title.save(frames_dir / "frame_0000.png")
    frames.append(title)
    frame_meta.append({"index": 0, "phase": "title", "clock_s": 0.0})

    results: dict[str, Any] = {}
    phase["name"] = "search"
    search = registry.call(
        "search_target",
        {
            "instruction": args.prompt,
            "n_views": args.scan_views,
            "max_steps": args.search_steps,
            "allow_move": False,
        },
        ctx,
    )
    results["search"] = {
        "status": search.status,
        "data": _result_data(search, (
            "candidate_id", "confirmed", "reason", "next_action", "n_candidates",
            "vlm_calls", "steps", "cost",
        )),
    }
    candidate_id = str((search.data or {}).get("candidate_id") or "")
    candidate_position = None
    if candidate_id:
        store = ctx.extra.get("perception.candidates") or {}
        record = store.get(candidate_id) if isinstance(store, dict) else None
        if isinstance(record, dict):
            candidate_position = record.get("position_flu")
    if not candidate_id:
        # Last-resort direct grounding of the current view; still one prompt.
        phase["name"] = "ground"
        obs = registry.call("observe", {"mode": "forward"}, ctx)
        ground = registry.call(
            "ground_target",
            {"instruction": args.prompt, "observation_id": obs.data.get("observation_id")},
            ctx,
        )
        results["ground_fallback"] = {
            "status": ground.status,
            "data": _result_data(ground, ("candidate_id", "n_candidates", "reason", "hint")),
        }
        candidate_id = str((ground.data or {}).get("candidate_id") or "")
        store = ctx.extra.get("perception.candidates") or {}
        record = store.get(candidate_id) if isinstance(store, dict) else None
        if isinstance(record, dict):
            candidate_position = record.get("position_flu")

    approach: list[dict] = []
    if candidate_id:
        phase["name"] = "approach"
        task_id = ""
        for _ in range(args.approach_legs):
            args_call = {"candidate_id": candidate_id, "max_legs": 1}
            if task_id:
                args_call = {"task_id": task_id, "max_legs": 1}
            step = registry.call("approach_candidate", args_call, ctx)
            data = step.data or {}
            approach.append({
                "status": step.status,
                "task_status": data.get("task_status"),
                "arrived": data.get("arrived"),
                "progress_m": data.get("progress_m"),
                "remaining_m": data.get("remaining_m"),
                "reason": data.get("reason"),
            })
            task_id = str(data.get("task_id") or task_id)
            if data.get("arrived") is True or str(data.get("task_status")) in (
                "arrived", "blocked", "failed", "cancelled",
            ):
                break
    results["approach"] = approach

    if candidate_id:
        phase["name"] = "confirm"
        confirm = registry.call(
            "confirm_candidate",
            {"candidate_id": candidate_id, "max_views": 2, "max_steps": 12},
            ctx,
        )
        results["confirm"] = {
            "status": confirm.status,
            "data": _result_data(confirm, (
                "verdict", "missing", "reason", "position_delta_m",
                "confirmation_count", "new_confirmation", "next_action",
            )),
        }
        phase["name"] = "verify"
        verify = registry.call(
            "verify_goal",
            {"candidate_id": candidate_id, "success_dist_m": args.success_dist},
            ctx,
        )
        results["verify"] = {
            "status": verify.status,
            "data": _result_data(verify, (
                "verified", "reason", "measured_distance_m", "distance_source",
                "unmet", "next_action",
            )),
        }
        phase["name"] = "finish"
        finish = registry.call(
            "finish",
            {"success": True, "reason": args.prompt, "candidate_id": candidate_id},
            ctx,
        )
        results["finish"] = {
            "status": finish.status,
            "data": _result_data(finish, (
                "success", "verified", "claim", "missing", "reason",
                "confirmation_required", "confirmation_accepted",
            )),
        }
    # final frame after the last action
    capture(original_render(), "end")
    world.render = original_render
    world.apply_action = original_apply

    video_path = out_dir / "first_person_demo.mp4"
    if frames:
        try:
            import imageio.v2 as imageio

            with imageio.get_writer(
                video_path, fps=args.fps, codec="libx264", quality=8,
                macro_block_size=None,
            ) as writer:
                hold = max(1, int(args.hold_frames))
                for image in frames:
                    for _ in range(hold):
                        writer.append_data(np.asarray(image))
        except Exception as exc:  # noqa: BLE001
            video_path = out_dir / "first_person_demo.gif"
            frames[0].save(
                video_path, save_all=True, append_images=frames[1:],
                duration=int(1000 / max(1, args.fps * max(1, args.hold_frames))),
                loop=0,
            )
            results["video_error"] = f"{type(exc).__name__}: {exc}"
    map_info = _xy_map(
        world.scene, list(getattr(world, "trajectory", [])),
        candidate_position, out_dir / "xy_trajectory.png",
        prompt=args.prompt, width=args.width, height=args.height,
    )
    trajectory = np.asarray(getattr(world, "trajectory", []), dtype=float)
    if trajectory.size:
        np.save(out_dir / "trajectory.npy", trajectory)
    summary = {
        "schema": "vlm_nav.semantic_corner_demo.v1",
        "prompt": args.prompt,
        "renderer": args.renderer,
        "model": args.model,
        "start_position": list(args.start_pos),
        "start_yaw_deg": args.start_yaw_deg,
        "candidate_id": candidate_id,
        "candidate_position": candidate_position,
        "results": results,
        "trajectory_points": int(len(trajectory)),
        "frames": len(frames),
        "video": str(video_path),
        "map": map_info,
        "wall_time_s": round(time.time() - clock_start, 3),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    vlm.close()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--ply", default=DEFAULT_PLY)
    parser.add_argument("--renderer", choices=["cpu", "higs"], default="cpu")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-pos", type=float, nargs=3, default=DEFAULT_START)
    parser.add_argument("--start-yaw-deg", type=float, default=90.0)
    parser.add_argument("--target-pos", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        help="dummy TargetSpec position (not used for grounding)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--hold-frames", type=int, default=2,
                        help="repeat each captured frame this many times")
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--model", default="qwen3-vl-8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--tool-budget", type=int, default=120)
    parser.add_argument("--scan-views", type=int, default=8)
    parser.add_argument("--search-steps", type=int, default=8)
    parser.add_argument("--approach-legs", type=int, default=10)
    parser.add_argument("--success-dist", type=float, default=1.8)
    parser.add_argument("--out", default=str(REPO / "outputs" / "vlm_nav" / "semantic_demo"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_demo(args)
    print(json.dumps({
        "candidate_id": summary["candidate_id"],
        "candidate_position": summary["candidate_position"],
        "search": summary["results"].get("search", {}).get("status"),
        "approach": [
            (step.get("task_status"), step.get("arrived"), step.get("remaining_m"))
            for step in summary["results"].get("approach", [])
        ],
        "confirm": summary["results"].get("confirm", {}).get("data", {}).get("verdict"),
        "verify": summary["results"].get("verify", {}).get("data", {}).get("verified"),
        "finish": summary["results"].get("finish", {}).get("data", {}),
        "frames": summary["frames"],
        "video": summary["video"],
        "map": summary["map"].get("path"),
    }, indent=2, ensure_ascii=False))
    finish = summary["results"].get("finish", {}).get("data", {})
    return 0 if finish.get("success") is True and finish.get("verified") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
