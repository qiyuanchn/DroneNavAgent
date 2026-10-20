#!/usr/bin/env python
"""Freeze the bedroom "chair by the window" stage as a replay set.

The replay set is the comparable input for the resource profiles: one frozen
scene, one instruction, one camera, and at least twelve views whose pixels,
poses and intrinsics never change when the renderer or the model changes. It is
built once with the production HiGS renderer from the same holdout combination
the long-horizon runs used (wooden chair + LED lamp in the bedroom), so the
frames show the actual stage the runs kept failing at.

Directory layout::

    outputs/vlm_nav/replay/window_chair/
      manifest.json            # schema vlm_nav.replay.v1
      frames/view_00.png ...   # 320x180 RGB, native sensor size
      scene/combo.ply          # cached conversion of the source SPZ (not needed
                               # by the replay loader, only for regeneration)

Usage (inside the vlm-nav container, GPU0)::

    CUDA_VISIBLE_DEVICES=0 python scripts/build_replay_window_chair.py

    # validate an existing set without rendering
    python scripts/build_replay_window_chair.py --verify

The view plan has three phases: ``approach`` (farther views on the way in),
``scan`` (eight headings around the chair at the confirmation standoff) and
``pitched`` (nose-down views, the elevation case a level-only confirmation
cannot measure). Every level view's yaw is computed from the chair position, so
the plan is reproducible from the manifest alone.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "outputs" / "vlm_nav" / "replay" / "window_chair"
HOLDOUT = REPO / "outputs" / "vlm_nav" / "verify_long_horizon_holdout" / "scenes_bedroom"
DEFAULT_COMBINED_SPZ = HOLDOUT / "combo_000000_seed_31001_bedroom_1f3a2fb80c29.spz"
DEFAULT_MANIFEST = HOLDOUT / "combo_000000_seed_31001_bedroom_1f3a2fb80c29.json"
DEFAULT_COMBINED_PLY = DEFAULT_OUT / "scene" / "combo.ply"
DEFAULT_CAMERA = {"width": 320, "height": 180, "hfov_deg": 86.0}
INSTRUCTION = (
    "两段式任务：先到卧室的窗边确认那把木椅（看到它并报出它的位置），"
    "再到木椅旁边亮着的灯旁停稳。每段都要用工具确认自己找对了物体；"
    "只有到达灯旁边并且确认无误才算完成。"
)
DEFAULT_RADIUS_M = 2.2
#: approach bearings/distances (degrees around the chair, metres). All three sit
#: on the window/bed side of the room; the +x side of this scene is a wall, so a
#: farther view from there would freeze the inside of that wall.
APPROACH = ((270.0, 3.2), (225.0, 3.2), (315.0, 3.2))
#: pitched-down views: (bearing_deg, distance_m, pitch_deg); the radii are
#: deliberately different from the level scan so no two views share a pose.
PITCHED = ((45.0, 1.8, 20.0), (135.0, 1.9, 25.0),
           (225.0, 2.0, 30.0), (315.0, 2.1, 35.0))


def _finite(value: Any, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be a number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite, got {value!r}")
    return number


def _target(value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"target position must be a 3-vector, got {value!r}")
    return tuple(_finite(item, "target") for item in value)  # type: ignore[return-value]


def _pose_at(target: tuple[float, float, float], bearing_deg: float, distance_m: float,
             z_offset_m: float = 0.15) -> tuple[float, float, float]:
    angle = math.radians(float(bearing_deg))
    return (
        target[0] + float(distance_m) * math.cos(angle),
        target[1] + float(distance_m) * math.sin(angle),
        target[2] + float(z_offset_m),
    )


def _yaw_to(target: tuple[float, float, float],
            pose: tuple[float, float, float]) -> float:
    return math.degrees(math.atan2(target[1] - pose[1], target[0] - pose[0]))


def plan_window_chair_views(
    target: Sequence[float],
    *,
    radius_m: float = DEFAULT_RADIUS_M,
    level_views: int = 8,
    pitch_views: int = 4,
    approach_views: int = 3,
    z_offset_m: float = 0.15,
) -> list[dict]:
    """Deterministic view plan around the chair (at least twelve views).

    - ``approach``: ``approach_views`` far views on the way in;
    - ``scan``: ``level_views`` headings around the chair at ``radius_m``;
    - ``pitched``: ``pitch_views`` nose-down views at close range.

    Every normal view's yaw points at the chair; a pitched view keeps that yaw
    and adds a positive (nose-down) body pitch.
    """
    target = _target(target)
    level_views = max(1, int(level_views))
    radius_m = _finite(radius_m, "radius_m")
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")
    views: list[dict] = []

    def add(pose, yaw_deg, pitch_deg, phase, note):
        rounded = [round(float(v), 6) for v in pose]
        views.append(
            {
                "id": f"view_{len(views):02d}",
                "file": f"frames/view_{len(views):02d}.png",
                "pose_flu": rounded,
                "yaw_deg": round(float(yaw_deg), 6),
                "pitch_deg": round(float(pitch_deg), 4),
                "phase": phase,
                "note": note,
                "index": len(views),
                "distance_m": round(float(math.dist(rounded, target)), 4),
                "bearing_deg": round(
                    math.degrees(math.atan2(rounded[1] - target[1], rounded[0] - target[0])), 4
                ),
            }
        )

    for bearing, distance in APPROACH[: max(0, int(approach_views))]:
        pose = _pose_at(target, bearing, distance, z_offset_m)
        add(pose, _yaw_to(target, tuple(round(v, 6) for v in pose)), 0.0, "approach",
            "approach to the window-chair stage from a farther view")

    for index in range(level_views):
        bearing = 360.0 * index / level_views
        pose = _pose_at(target, bearing, radius_m, z_offset_m)
        add(pose, _yaw_to(target, tuple(round(v, 6) for v in pose)), 0.0, "scan",
            f"level scan heading {bearing:.0f} deg around the chair")

    for bearing, distance, pitch in PITCHED:
        if sum(1 for view in views if view["phase"] == "pitched") >= int(pitch_views):
            break
        pose = _pose_at(target, bearing, distance, z_offset_m)
        add(pose, _yaw_to(target, tuple(round(v, 6) for v in pose)), pitch, "pitched",
            "nose-down view for a target below the level frame")
    return views


def _camera_dict(camera: Mapping[str, Any]) -> dict:
    width = int(camera["width"])
    height = int(camera["height"])
    hfov = float(camera["hfov_deg"])
    fx = width / (2.0 * math.tan(math.radians(hfov) / 2.0))
    return {
        "width": width,
        "height": height,
        "hfov_deg": hfov,
        "fx": fx,
        "fy": fx,
        "cx": width / 2.0,
        "cy": height / 2.0,
    }


def build_manifest(
    *,
    name: str,
    instruction: str,
    views: Iterable[Mapping[str, Any]],
    camera: Mapping[str, Any],
    scene: Mapping[str, Any],
    generator: Mapping[str, Any],
    benchmark: Mapping[str, Any] | None = None,
    created_utc: str | None = None,
) -> dict:
    """The ``vlm_nav.replay.v1`` manifest for a rendered view plan."""
    intrinsics = _camera_dict(camera)
    entries = []
    for index, view in enumerate(views):
        entry = {
            "id": str(view["id"]),
            "file": str(view["file"]),
            "pose_flu": [float(v) for v in view["pose_flu"]],
            "yaw_deg": float(view["yaw_deg"]),
            "pitch_deg": float(view.get("pitch_deg", 0.0)),
            "phase": str(view.get("phase") or "unspecified"),
            "index": int(view.get("index", index)),
            "intrinsics": dict(intrinsics),
        }
        if view.get("sha256"):
            entry["sha256"] = str(view["sha256"])
        if view.get("note"):
            entry["note"] = str(view["note"])
        if view.get("distance_m") is not None:
            entry["distance_m"] = float(view["distance_m"])
        if view.get("bearing_deg") is not None:
            entry["bearing_deg"] = float(view["bearing_deg"])
        entries.append(entry)
    return {
        "schema": "vlm_nav.replay.v1",
        "name": str(name),
        "instruction": str(instruction),
        "created_utc": created_utc or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "camera": {key: intrinsics[key] for key in ("width", "height", "hfov_deg")},
        "scene": dict(scene),
        "generator": dict(generator),
        "benchmark": dict(benchmark or {}),
        "views": entries,
    }


def _display_path(path: Path) -> str:
    """Repo-relative path when possible, so the manifest travels with the repo."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO.resolve()))
    except ValueError:
        return str(resolved)


def write_manifest(root: Path, manifest: Mapping[str, Any]) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_readme(root: Path, manifest: Mapping[str, Any]) -> Path:
    """Human entry point next to the manifest (the loader ignores this file)."""
    root = Path(root)
    views = manifest.get("views") or []
    phases: dict[str, int] = {}
    for view in views:
        phase = str(view.get("phase") or "unspecified")
        phases[phase] = phases.get(phase, 0) + 1
    camera = manifest.get("camera") or {}
    scene = manifest.get("scene") or {}
    text = f"""# {manifest.get('name', 'replay')} replay

Frozen observation set for the bedroom "chair by the window" stage, built once
with the production HiGS renderer. Use it for measurements that must be
comparable across the A100, a 4090, or different harness variants.

- views: {len(views)} ({phases})
- camera: {camera.get('width')}x{camera.get('height')} @ {camera.get('hfov_deg')} deg
- instruction: {manifest.get('instruction', '')}
- scene: {scene.get('target_name', '')} at {scene.get('target_position_flu')}
- source: {scene.get('source_combined_spz', '')}
- benchmark: {manifest.get('benchmark', {})}

## Load

```python
from scripts.replay_frames import load_replay

replay = load_replay("outputs/vlm_nav/replay/window_chair")
frames = replay.frames()            # [N, 3, H, W] float, CHW
images = replay.pil_frames()        # RGB PIL, the VLM image path
first = replay.views[0]             # pose_flu, yaw_deg, pitch_deg, intrinsics
```

Every frame is checked against its SHA-256 on load; a missing frame, a
non-finite pose or a checksum mismatch is a `ReplayError`, never a silently
degraded benchmark.

## Rebuild

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/build_replay_window_chair.py
python scripts/build_replay_window_chair.py --verify
```

The manifest records the scene source, the view plan and the generator
(torch/CUDA/GPU) so a future host can reproduce the same sequence.
"""
    path = root / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_scene_ply(spz: Path, ply: Path, *, force: bool = False) -> Path:
    """Convert the combined SPZ once; the replay build needs a PLY renderer."""
    spz, ply = Path(spz), Path(ply)
    if ply.is_file() and not force:
        if not spz.is_file() or ply.stat().st_mtime >= spz.stat().st_mtime:
            return ply
    if not spz.is_file():
        raise FileNotFoundError(f"combined scene SPZ does not exist: {spz}")
    try:
        from diffaero.utils.gaussian_splatting.spz import read_spz
    except ImportError as exc:  # pragma: no cover - validation side only
        raise RuntimeError(
            "building the replay from SPZ needs the diffaero SPZ decoder; "
            "initialize third_party/diffaero, put it on PYTHONPATH or set "
            "DIFFAERO_ROOT"
        ) from exc
    try:
        from scripts.spz_to_3dgs_ply import write_ply
    except ImportError:  # pragma: no cover - running from the script directory
        try:
            from spz_to_3dgs_ply import write_ply  # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeError(
                "building the replay from SPZ needs the SPZ -> PLY converter; "
                "this repository does not bundle scripts/spz_to_3dgs_ply.py - "
                "take it from the diffaero validation tree or pass a PLY"
            ) from exc
    cloud = read_spz(spz)
    write_ply(ply, cloud, comment=f"window-chair replay source={spz.name}")
    return ply


def render_view_plan(world, views: Sequence[Mapping[str, Any]], root: Path) -> list[dict]:
    """Render every planned view from one HiGS world and hash the saved frame."""
    from vlm_nav.vlm import frame_to_pil
    from vlm_nav.world import yaw_pitch_quat_xyzw
    import numpy as np

    root = Path(root)
    rendered: list[dict] = []
    for view in views:
        pose = np.asarray(view["pose_flu"], dtype=np.float64)
        yaw_rad = math.radians(float(view["yaw_deg"]))
        pitch_rad = math.radians(float(view.get("pitch_deg", 0.0)))
        world.state.position_flu = pose
        world.state.yaw_rad = yaw_rad
        if pitch_rad:
            frame = world._render_pose(pose, yaw_pitch_quat_xyzw(yaw_rad, pitch_rad))
        else:
            frame = world.render()
        path = root / str(view["file"])
        path.parent.mkdir(parents=True, exist_ok=True)
        frame_to_pil(frame).save(path)
        entry = dict(view)
        entry["sha256"] = _sha256(path)
        entry["frame_mean"] = round(float(frame.mean()), 6)
        rendered.append(entry)
    return rendered


def _read_target(manifest_path: Path, target_index: int) -> tuple[str, list[float]]:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    positions = data.get("object_positions_flu") or data.get("entity_positions_flu")
    if not positions:
        raise ValueError(f"no object_positions_flu in {manifest_path}")
    index = int(target_index)
    if not 0 <= index < len(positions):
        raise IndexError(f"target index {index} out of range (0..{len(positions) - 1})")
    name = "target"
    instances = data.get("object_instances") or []
    if index < len(instances):
        name = str(instances[index].get("name") or name)
    elif data.get("source_object"):
        name = str(data["source_object"].get("name") or name)
    return name, [float(v) for v in positions[index]]


def build(
    *,
    out: Path = DEFAULT_OUT,
    combined_ply: Path | None = None,
    combined_spz: Path = DEFAULT_COMBINED_SPZ,
    manifest_json: Path = DEFAULT_MANIFEST,
    target_index: int = 0,
    camera: Mapping[str, Any] = DEFAULT_CAMERA,
    device: str = "cuda:0",
    force: bool = False,
    radius_m: float = DEFAULT_RADIUS_M,
) -> dict:
    """Build the replay set on the production HiGS renderer."""
    import torch

    from vlm_nav.world import SplatWorldHiGS, TargetSpec

    out = Path(out)
    target_name, target = _read_target(Path(manifest_json), target_index)
    ply = (
        Path(combined_ply)
        if combined_ply is not None
        else out / "scene" / "combo.ply"
    )
    if not ply.is_file() or force:
        ply = ensure_scene_ply(Path(combined_spz), ply, force=force)
    views = plan_window_chair_views(target, radius_m=radius_m)
    start = views[0]
    world = SplatWorldHiGS(
        ply_path=ply,
        target=TargetSpec(name=target_name, position_flu=tuple(target), radius_m=0.37),
        start_flu=tuple(start["pose_flu"]),
        start_yaw_deg=float(start["yaw_deg"]),
        scene_scale=1.0,
        z_min=float(target[2]) - 4.0,
        z_max=float(target[2]) + 4.0,
        device=device,
    )
    try:
        rendered = render_view_plan(world, views, out)
        scene_info = {
            "key": "bedroom",
            "world": "higs",
            "target_name": target_name,
            "target_index": int(target_index),
            "target_position_flu": list(target),
            "source_combined_spz": _display_path(Path(combined_spz)),
            "source_manifest": _display_path(Path(manifest_json)),
            "render_ply": _display_path(ply),
            "scene_scale": 1.0,
            "n_gaussians": int(world.scene.xyz.shape[0]),
        }
        generator = {
            "backend": "higs_ut",
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(torch.device(device)) if torch.cuda.is_available() else "",
            "command": "python scripts/build_replay_window_chair.py",
            "view_plan": {
                "radius_m": float(radius_m),
                "approach": [list(item) for item in APPROACH],
                "pitched": [list(item) for item in PITCHED],
            },
        }
        manifest = build_manifest(
            name="window_chair",
            instruction=INSTRUCTION,
            views=rendered,
            camera=camera,
            scene=scene_info,
            generator=generator,
            benchmark={
                "vlm_image_size": [320, 180],
                "sensor_size": [int(camera["width"]), int(camera["height"])],
                "note": "the VLM input resize the harness uses; keep it identical across hosts",
            },
        )
        write_manifest(out, manifest)
        write_readme(out, manifest)
    finally:
        close = getattr(world, "close", None)
        if callable(close):
            close()
    return {
        "out": str(out),
        "views": len(views),
        "phases": {phase: sum(1 for view in views if view["phase"] == phase)
                   for phase in ("approach", "scan", "pitched")},
        "target_name": target_name,
        "target_position_flu": list(target),
        "manifest": str(out / "manifest.json"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--combined-ply", type=Path, default=None,
                        help="cached combined-scene PLY (default: convert --combined-spz)")
    parser.add_argument("--combined-spz", type=Path, default=DEFAULT_COMBINED_SPZ)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--target-index", type=int, default=0,
                        help="object_positions_flu index of the chair")
    parser.add_argument("--width", type=int, default=DEFAULT_CAMERA["width"])
    parser.add_argument("--height", type=int, default=DEFAULT_CAMERA["height"])
    parser.add_argument("--hfov", type=float, default=DEFAULT_CAMERA["hfov_deg"])
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS_M)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--force", action="store_true", help="re-render/re-convert existing files")
    parser.add_argument("--verify", action="store_true",
                        help="only validate an existing replay set")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify:
        try:
            from scripts.replay_frames import load_replay
        except ImportError:  # pragma: no cover
            from replay_frames import load_replay  # type: ignore[no-redef]
        replay = load_replay(args.out)
        print(json.dumps(replay.summary(), ensure_ascii=False, indent=2))
        return 0
    camera = {"width": int(args.width), "height": int(args.height), "hfov_deg": float(args.hfov)}
    result = build(
        out=args.out,
        combined_ply=args.combined_ply,
        combined_spz=args.combined_spz,
        manifest_json=args.manifest_json,
        target_index=args.target_index,
        camera=camera,
        device=args.device,
        force=args.force,
        radius_m=args.radius,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
