#!/usr/bin/env python
"""VLM-in-the-loop semantic navigation in a 3DGS scene.

CPU world (no GPU needed, pure-torch splat renderer):

    python scripts/run_vlm_nav_3dgs.py \
        --world cpu --ply /path/scene.ply \
        --target-name book --target-pos 0 0 1.0 --target-radius 0.4 \
        --start-pos -4 0 1.2 --start-yaw-deg 0 \
        --instruction "Find the book and fly to it." \
        --backend hf --model-path /home/dataset-assist-0/models/Qwen3-VL-8B \
        --out outputs/vlm_nav/book_demo

GPU kinematic world (production HiGS renderer, needs CUDA but no env/manifest):

    python scripts/run_vlm_nav_3dgs.py \
        --world higs --ply /path/scene.ply \
        --target-name book --target-pos 0 0 0.5 --target-radius 0.5 \
        --start-pos -2.5 0 1.0 \
        --instruction "Find the book and fly to it." \
        --backend hf --device cuda --model-path /path/Qwen3-VL-8B \
        --out outputs/vlm_nav/book_higs

GPU world (production env, requires CUDA; mirrors eval_gwp_searching_3dgs.py):

    python scripts/run_vlm_nav_3dgs.py \
        --world env --combined-ply combo.ply --semantic-manifest manifest.json \
        --instruction "Find the target object." --backend openai --model gpt-4o

Backends: mock:oracle | mock:forward | hf (local HF weights) | openai
(any OpenAI-compatible endpoint, e.g. `vllm serve`; uses OPENAI_BASE_URL /
OPENAI_API_KEY when set).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from vlm_nav.harness import SemanticNavHarness
from vlm_nav.scene_combo import choose_start_pose, load_scene_combo
from vlm_nav.vlm import make_backend
from vlm_nav.world import (
    EnvWorld,
    SplatWorldCPU,
    SplatWorldHiGS,
    SuperPlannerWorld,
    TargetSpec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", choices=["cpu", "higs", "env", "super"], default="cpu")
    parser.add_argument("--ply", type=str, default=None,
                        help="CPU/HiGS 3DGS PLY/SPZ asset; optional with "
                             "--semantic-manifest (uses its merged_ply)")
    parser.add_argument("--scene-scale", type=float, default=1.0)
    parser.add_argument("--min-opacity", type=float, default=0.0,
                        help="drop PLY Gaussians below this opacity at load")
    parser.add_argument("--combined-ply", type=str, default=None)
    parser.add_argument("--background-ply", type=str, default=None)
    parser.add_argument("--semantic-manifest", type=str, default=None,
                        help="DiffAero combo_*.json; supplies the merged asset, "
                             "target, and scene bounds")
    parser.add_argument("--voxel-size", type=float, default=0.15)
    parser.add_argument("--target-name", type=str, default="target")
    parser.add_argument("--target-pos", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"), help="FLU, scene-centered")
    parser.add_argument("--target-radius", type=float, default=0.5)
    parser.add_argument("--start-pos", type=float, nargs=3, default=(0.0, 0.0, 1.0))
    parser.add_argument("--start-yaw-deg", type=float, default=0.0)
    parser.add_argument("--auto-start", action="store_true",
                        help="with --semantic-manifest: pick the start position/yaw "
                             "from choose_start_pose instead of --start-pos")
    parser.add_argument("--z-min", type=float, default=0.2,
                        help="flyable floor of the splat world (FLU z, scene-centered)")
    parser.add_argument("--z-max", type=float, default=None,
                        help="no-fly ceiling of the splat world; without it the drone "
                             "may climb to the top of the scene")
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--backend", type=str, default="mock:oracle")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model", type=str, default=None,
                        help="model name for the openai backend")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu",
                        help="device for the local hf backend")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--image-size", type=int, nargs=2, default=(320, 180),
                        metavar=("W", "H"), help="VLM input resize")
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--no-think", action="store_true",
                        help="disable chain-of-thought in the chat template "
                             "(thinking-style models)")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--context-frames", type=int, default=3,
                        help="frames sent to the VLM per decision (current + history)")
    parser.add_argument("--no-look-around", action="store_true",
                        help="skip the initial 360-degree scan phase")
    parser.add_argument("--scan-views", type=int, default=8)
    parser.add_argument("--rescan-after", type=int, default=6,
                        help="force a look-around after N steps without distance "
                             "improvement (0 = off)")
    parser.add_argument("--no-done-retry", action="store_true",
                        help="end the episode on a premature done instead of "
                             "rejecting it with distance feedback")
    parser.add_argument("--no-bev", action="store_true",
                        help="disable the top-down spatial-memory map image")
    parser.add_argument("--scan-pitch-deg", type=float, default=50.0,
                        help="add downward-looking views to the scan strip "
                             "(0 = level-only)")
    parser.add_argument("--no-done-verify", action="store_true",
                        help="skip the zoomed-crop second check on done claims")
    parser.add_argument("--no-candidates", action="store_true",
                        help="disable candidate target memory (no fusion, no prompt list)")
    parser.add_argument("--no-keyframe", action="store_true",
                        help="disable pinning the last target-sighting frame")
    parser.add_argument("--success-dist", type=float, default=2.0,
                        help="meters from the target surface")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--env-overrides", type=str, nargs="*", default=None,
                        help="extra hydra overrides for --world env")
    parser.add_argument("--target-index", type=int, default=None,
                        help="fixed manifest target index (cpu/higs/super)")
    parser.add_argument("--initial-yaw-mode", type=str, default="target",
                        choices=["target", "visible", "random"])
    parser.add_argument("--max-init-attempts", type=int, default=8,
                        help="re-sample starts until visible (visible mode) and "
                             "plannable (SUPER preflight), then skip the cell")
    parser.add_argument("--worker-python", type=str, default=None,
                        help="python for SUPER worker subprocesses (must import "
                             "super_planner_py; default: keep harness interpreter)")
    parser.add_argument("--map-mode", type=str, default="global",
                        choices=["global", "accumulate", "blind"],
                        help="global: full scene map at spawn (production default); "
                             "accumulate: SUPER starts blind and the map is built "
                             "from simulated onboard depth observations; "
                             "blind: permanently empty map (no avoidance, ablation floor)")
    parser.add_argument("--sensing-range-m", type=float, default=6.0)
    parser.add_argument("--sensing", type=str, default="ideal",
                        choices=["ideal", "moge"],
                        help="ideal: privileged scene-surface sensing; "
                             "moge: monocular RGB -> MoGe-2 depth (no privileged geometry)")
    parser.add_argument("--moge-model-path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Seed every RNG before anything runs (mirrors eval_gwp_searching_3dgs.py):
    # without this, python `random` (initial-yaw jitter, scan plumbing) varies
    # per process and identical --seed runs are not paired.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.world in ("cpu", "higs"):
        manifest = args.semantic_manifest
        if not manifest and not args.ply:
            raise SystemExit("--world cpu/higs requires --ply or --semantic-manifest")
        if not manifest and not args.target_pos:
            raise SystemExit(
                "--world cpu/higs requires --target-pos or --semantic-manifest"
            )
        combo = load_scene_combo(manifest) if manifest else None
        target_index = int(args.target_index or 0)
        target = None
        if args.target_pos:
            target = TargetSpec(
                name=args.target_name,
                position_flu=tuple(args.target_pos),
                radius_m=args.target_radius,
            )
        start_flu = tuple(args.start_pos)
        start_yaw_deg = float(args.start_yaw_deg)
        z_min = float(args.z_min)
        z_max = args.z_max
        if combo is not None:
            if args.auto_start:
                start_flu, start_yaw_deg = choose_start_pose(combo, target_index)
                print(
                    f"[vlm_nav] auto start pose: position_flu="
                    f"({start_flu[0]:.2f}, {start_flu[1]:.2f}, {start_flu[2]:.2f}) "
                    f"yaw_deg={start_yaw_deg:.1f}",
                    flush=True,
                )
            ground_z = combo.ground_z_flu()
            if z_min == 0.2:
                z_min = ground_z + 0.2
            if z_max is None:
                z_max = ground_z + 4.0
        world_cls = SplatWorldHiGS if args.world == "higs" else SplatWorldCPU
        extra = {"device": args.device} if args.world == "higs" else {}
        world = world_cls(
            ply_path=args.ply or None,
            target=target,
            semantic_manifest=manifest,
            target_index=target_index,
            start_flu=start_flu,
            start_yaw_deg=start_yaw_deg,
            scene_scale=args.scene_scale,
            min_opacity=args.min_opacity,
            z_min=z_min,
            z_max=z_max,
            **extra,
        )
    else:
        target = None
        if args.target_pos:
            target = TargetSpec(
                name=args.target_name,
                position_flu=tuple(args.target_pos),
                radius_m=args.target_radius,
            )
        world_cls = SuperPlannerWorld if args.world == "super" else EnvWorld
        extra_env: dict = {}
        if args.world == "super":
            extra_env["worker_python"] = args.worker_python
            extra_env["map_mode"] = args.map_mode
            extra_env["sensing_range_m"] = args.sensing_range_m
            extra_env["sensing_backend"] = args.sensing
            extra_env["moge_model_path"] = args.moge_model_path
            extra_env["moge_device"] = args.device if args.device.startswith("cuda") else "cuda:0"
        world = world_cls(
            hydra_overrides=args.env_overrides,
            combined_ply=args.combined_ply,
            background_ply=args.background_ply,
            semantic_manifest=args.semantic_manifest,
            voxel_size=args.voxel_size,
            target=target,
            device=args.device if args.device.startswith("cuda") else "cuda:0",
            **extra_env,
        )
        if args.world == "super":
            try:
                world.reset(seed=args.seed, target_index=args.target_index,
                            initial_yaw_mode=args.initial_yaw_mode,
                            max_init_attempts=args.max_init_attempts)
            except RuntimeError as exc:
                # infeasible cell (no plannable/visible start): record a skip,
                # do not count it as a navigation failure (mirrors the
                # infer_hyb evaluator's load-time skip of unusable entries)
                print(f"[vlm_nav] SKIP: {exc}", flush=True)
                skip = {
                    "success": False,
                    "finish_reason": "infeasible_skip",
                    "skipped": True,
                    "steps": 0,
                    "vlm_calls": 0,
                    "collision_hits": 0,
                    "instruction": args.instruction,
                    "target_name": args.target_name,
                    "final_distance_m": float("nan"),
                    "min_distance_m": float("nan"),
                    "path_length_m": 0.0,
                    "wall_time_s": 0.0,
                    "parse_failures": 0,
                }
                with open(out_dir / "summary.json", "w") as handle:
                    json.dump(skip, handle, indent=2)
                return
        else:
            world.reset(seed=args.seed)

    backend_kwargs: dict = {}
    if args.backend.startswith("hf"):
        if not args.model_path:
            raise SystemExit("--backend hf requires --model-path")
        backend_kwargs.update(
            model_path=args.model_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            image_size=tuple(args.image_size),
            num_threads=args.num_threads,
        )
        if args.no_think:
            backend_kwargs["template_kwargs"] = {"enable_thinking": False}
    elif args.backend == "openai":
        if not args.model:
            raise SystemExit("--backend openai requires --model")
        backend_kwargs.update(
            model=args.model,
            base_url=args.base_url,
            max_new_tokens=args.max_new_tokens,
            image_size=tuple(args.image_size),
        )
    vlm = make_backend(args.backend, **backend_kwargs)

    with open(out_dir / "run_config.json", "w") as handle:
        json.dump(vars(args), handle, indent=2, default=str)

    harness = SemanticNavHarness(
        world=world,
        vlm=vlm,
        instruction=args.instruction,
        out_dir=out_dir,
        max_steps=args.max_steps,
        success_dist_m=args.success_dist,
        make_video=not args.no_video,
        context_frames=args.context_frames,
        look_around=not args.no_look_around,
        scan_views=args.scan_views,
        rescan_after=args.rescan_after,
        done_retry=not args.no_done_retry,
        bev=not args.no_bev,
        scan_pitch_deg=args.scan_pitch_deg,
        done_verify=not args.no_done_verify,
        candidates_enabled=not args.no_candidates,
        keyframe_enabled=not args.no_keyframe,
    )
    summary = harness.run()
    print(json.dumps({k: v for k, v in vars(summary).items() if k != "records"},
                     indent=2))


if __name__ == "__main__":
    main()
