#!/usr/bin/env python
"""Render a first-person multi-stage mission demo video.

The video is rendered from the same CPU 3DGS world and the same
``MixedMissionExecutor`` used by the agent, not from a hand-drawn animation.
Frames are captured after every motion chunk (and at every stage boundary),
labelled with the stage number and the sub-goal, then encoded to MP4.

Default mission (the goal.md chain, semantic steps served by the deterministic
mock service so no model call is needed):

    ascend 1 m -> forward 5 m -> hover 3 s -> turn right 90 deg ->
    search the red object -> approach it -> move left 1 m -> photo

    python scripts/make_mixed_mission_video.py
    python scripts/make_mixed_mission_video.py --motion-only
    python scripts/make_mixed_mission_video.py --mission-json my_mission.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from vlm_nav.mixed_mission import (  # noqa: E402
    MixedMissionExecutor,
    MixedMissionSpec,
    MixedMissionState,
)
from vlm_nav.motion import MotionExecutor  # noqa: E402
from vlm_nav.tools import ToolContext, ToolRegistry  # noqa: E402
from vlm_nav.vision import contact_sheet  # noqa: E402
from vlm_nav.vlm import frame_to_pil  # noqa: E402
from vlm_nav.world import SplatWorldCPU, SplatWorldHiGS, TargetSpec  # noqa: E402

DEFAULT_FIXTURE = REPO / "data" / "phase1" / "scenes" / "scenario_scene.ply"
DEFAULT_TARGET = (0.48, 1.404, 0.442)
DEFAULT_START = (-1.52, 0.904, 0.642)
DEFAULT_MISSION = {
    "mission_id": "first_person_multi_stage",
    "instruction": (
        "上升 1 米，向前飞 5 米，悬停 3 秒，右转 90 度，"
        "找到红色物体，飞到它旁边，再向左移动 1 米并拍照。"
    ),
    "steps": [
        {"kind": "move", "direction": "up", "meters": 1.0},
        {"kind": "move", "direction": "forward", "meters": 5.0},
        {"kind": "hover", "seconds": 3.0},
        {"kind": "turn", "direction": "right", "degrees": 90},
        {"kind": "search", "target": "红色物体", "relation": ""},
        {"kind": "approach", "target": "last", "standoff_m": 1.2},
        {"kind": "move", "direction": "left", "meters": 1.0},
        {"kind": "photo", "label": "final"},
    ],
}
MOTION_ONLY = {
    "mission_id": "first_person_motion",
    "instruction": (
        "上升 1 米，向前飞 5 米，悬停 3 秒，右转 90 度，"
        "继续前进 1 米并拍照。"
    ),
    "steps": [
        {"kind": "move", "direction": "up", "meters": 1.0},
        {"kind": "move", "direction": "forward", "meters": 5.0},
        {"kind": "hover", "seconds": 3.0},
        {"kind": "turn", "direction": "right", "degrees": 90},
        {"kind": "move", "direction": "forward", "meters": 1.0},
        {"kind": "photo", "label": "final"},
    ],
}


def _load_mock_service():
    path = REPO / "scripts" / "run_mixed_mission.py"
    spec = importlib.util.spec_from_file_location("demo_mock_service", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MockSemanticService


def _label(record: Any, index: int, total: int) -> str:
    if isinstance(record, dict):
        kind = str(record.get("kind", ""))
        params = record
    else:
        kind = str(getattr(record, "kind", ""))
        params = getattr(record, "params", {}) or {}
    if kind == "move":
        text = f"move {params.get('direction', '?')} {params.get('meters', '?')} m"
    elif kind == "turn":
        degrees = params.get("degrees")
        if degrees is None:
            degrees = params.get("direction", "?")
        text = f"turn {degrees}"
    elif kind == "hover":
        text = f"hover {params.get('seconds', '?')} s"
    elif kind == "search":
        text = "search target (mock)"
    elif kind == "approach":
        text = "approach candidate (mock)"
    elif kind == "photo":
        text = "take a photo"
    elif kind == "revisit":
        text = "revisit a known place"
    elif kind == "goto":
        text = f"goto {params.get('position')}"
    else:
        text = kind or "stage"
    return f"Stage {index + 1}/{total}: {text}"


def _overlay(image: Image.Image, text: str, clock: float, stage: str) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    bar = max(26, height // 9)
    draw.rectangle((0, 0, width, bar), fill=(0, 0, 0))
    draw.text((8, 5), text[:90], fill=(255, 255, 255))
    draw.text((8, bar + 6), f"{stage}   t={clock:.2f}s", fill=(255, 220, 80))
    return canvas


def _card(width: int, height: int, title: str, lines: list[str]) -> Image.Image:
    image = Image.new("RGB", (width, height), (8, 12, 20))
    draw = ImageDraw.Draw(image)
    draw.text((24, 24), title, fill=(255, 255, 255))
    y = 60
    for line in lines:
        draw.text((24, y), line[:80], fill=(210, 220, 235))
        y += 22
    return image


def make_video(
    out_dir: Path,
    *,
    mission: dict,
    width: int = 640,
    height: int = 360,
    fps: int = 8,
    semantic_mock: bool = True,
    fixture: Path = DEFAULT_FIXTURE,
    target: tuple[float, float, float] = DEFAULT_TARGET,
    start: tuple[float, float, float] = DEFAULT_START,
    start_yaw_deg: float = 0.0,
    renderer: str = "cpu",
    device: str = "cuda:0",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("frame_*.png"):
        old.unlink()
    if renderer == "higs":
        world = SplatWorldHiGS(
            fixture,
            TargetSpec("red object", target, radius_m=0.2),
            start_flu=start,
            start_yaw_deg=start_yaw_deg,
            width=width,
            height=height,
            hfov_deg=86.0,
            bound_radius=30.0,
            z_min=0.05,
            z_max=20.0,
            device=device,
        )
    else:
        world = SplatWorldCPU(
            fixture,
            TargetSpec("red object", target, radius_m=0.2),
            start_flu=start,
            start_yaw_deg=start_yaw_deg,
            width=width,
            height=height,
            hfov_deg=86.0,
            bound_radius=30.0,
            z_min=0.05,
            z_max=20.0,
        )
    registry = ToolRegistry()
    ctx = ToolContext(world=world, episode_id="demo_video")
    motion = MotionExecutor(world, artifacts_dir=out_dir / "artifacts")
    service = None
    if semantic_mock:
        MockSemanticService = _load_mock_service()
        service = MockSemanticService(world)
    executor = MixedMissionExecutor(
        world,
        registry=registry,
        ctx=ctx,
        motion_executor=motion,
        semantic_service=service,
        artifacts_dir=out_dir / "artifacts",
        state_path=out_dir / "mission.state.json",
    )
    spec = MixedMissionSpec.from_dict(mission)
    frames: list[Image.Image] = []
    labels: list[str] = []
    started = time.time()

    def stage_of() -> tuple[int, int, str]:
        state = executor.state
        if state is None:
            return 0, len(spec.steps), "starting"
        index = min(state.current_step, len(state.steps) - 1)
        record = state.steps[index] if state.steps else None
        return index, len(state.steps), (
            _label(record, index, len(state.steps)) if record else "starting"
        )

    def capture(event: dict | None = None) -> None:
        index, total, text = stage_of()
        image = frame_to_pil(world.render())
        if image.size != (width, height):
            image = image.resize((width, height), Image.BILINEAR)
        clock = float(event.get("clock_s", 0.0)) if isinstance(event, dict) else 0.0
        frames.append(_overlay(image, text, clock, f"frame {len(frames) + 1:04d}"))
        labels.append(text)

    # Title card (held for one second). The default PIL font has no CJK glyphs,
    # so keep the card text ASCII; the exact instruction stays in the JSON.
    instruction_text = str(mission.get("instruction", ""))
    if any(ord(ch) > 127 for ch in instruction_text):
        instruction_text = "first-person multi-stage demo (instruction in video_summary.json)"
    frames.append(_card(
        width, height,
        "DroneNavAgent - first-person multi-stage mission",
        [
            "Mission: " + instruction_text[:76],
            f"Rendered by the {'HiGS GPU' if renderer == 'higs' else 'CPU'} "
            "3DGS renderer; executor = MixedMissionExecutor.",
            "Stages: " + " -> ".join(
                _label(spec.steps[i], i, len(spec.steps))
                for i in range(min(len(spec.steps), 8))
            ),
            "Semantic steps use the deterministic mock service." if semantic_mock else "",
        ],
    ))

    executor.on_progress = lambda event: capture(event) if event.get("event") == "step.finished" else None
    motion.on_progress = lambda event: capture(event) if event.get("event") == "step.chunk" else None
    capture({"clock_s": 0.0})
    state: MixedMissionState = executor.run(spec)
    capture({"clock_s": state.clock_s})
    # a short end card
    display_out = str(out_dir)
    if "/workspace/DroneNavAgent/" in display_out:
        display_out = display_out.split("/workspace/DroneNavAgent/", 1)[1]
    frames.append(_card(
        width, height,
        "Mission finished: " + state.status,
        [
            f"steps: {state.completed_steps()}/{len(state.steps)}",
            f"sim clock: {state.clock_s:.2f}s",
            f"position: {[round(float(v), 2) for v in world.position_flu]}",
            "outputs: " + display_out,
        ],
    ))

    frames_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(frames):
        image.save(frames_dir / f"frame_{index:04d}.png")
    video = out_dir / "first_person_multi_stage.mp4"
    try:
        import imageio.v2 as imageio

        with imageio.get_writer(
            video, fps=fps, codec="libx264", quality=8, macro_block_size=None
        ) as writer:
            for image in frames:
                writer.append_data(np.asarray(image))
    except Exception as exc:  # noqa: BLE001 - fall back to GIF
        video = out_dir / "first_person_multi_stage.gif"
        frames[0].save(
            video, save_all=True, append_images=frames[1:],
            duration=int(1000 / max(1, fps)), loop=0,
        )
        print(f"mp4 export failed ({exc}); wrote GIF instead")
    summary = {
        "schema": "vlm_nav.mixed_mission_video.v1",
        "mission": spec.to_dict(),
        "status": state.status,
        "steps": [
            {"kind": record.kind, "status": record.status, "params": record.params}
            for record in state.steps
        ],
        "frames": len(frames),
        "renderer": renderer,
        "device": device if renderer == "higs" else "cpu",
        "fps": fps,
        "duration_s": round(len(frames) / fps, 3),
        "video": str(video),
        "frames_dir": str(frames_dir),
        "wall_time_s": round(time.time() - started, 3),
    }
    (out_dir / "video_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(REPO / "outputs" / "vlm_nav" / "demo_mixed_mission"))
    parser.add_argument("--mission-json", default=None)
    parser.add_argument("--motion-only", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--renderer", choices=["cpu", "higs"], default="cpu",
                        help="cpu: portable SplatWorldCPU; higs: DiffAero GPU renderer")
    parser.add_argument("--device", default="cuda:0",
                        help="device for --renderer higs")
    parser.add_argument("--ply", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--target-pos", type=float, nargs=3, default=DEFAULT_TARGET)
    parser.add_argument("--start-pos", type=float, nargs=3, default=DEFAULT_START)
    parser.add_argument("--start-yaw-deg", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.mission_json:
        mission = json.loads(Path(args.mission_json).read_text(encoding="utf-8"))
        semantic = any(
            str(step.get("kind")) in (
                "search", "approach", "confirm", "verify", "revisit", "goto"
            )
            for step in mission.get("steps", [])
        )
    else:
        mission = MOTION_ONLY if args.motion_only else DEFAULT_MISSION
        semantic = not args.motion_only
    summary = make_video(
        Path(args.out),
        mission=mission,
        width=args.width,
        height=args.height,
        fps=args.fps,
        semantic_mock=semantic,
        fixture=Path(args.ply),
        target=tuple(args.target_pos),
        start=tuple(args.start_pos),
        start_yaw_deg=args.start_yaw_deg,
        renderer=args.renderer,
        device=args.device,
    )
    print(json.dumps({
        "video": summary["video"],
        "frames": summary["frames"],
        "duration_s": summary["duration_s"],
        "status": summary["status"],
    }, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
