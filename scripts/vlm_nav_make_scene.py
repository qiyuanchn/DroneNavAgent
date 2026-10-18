#!/usr/bin/env python
"""Build semantic-navigation test scenes with the repo's 3DGS combination pipeline.

Wraps ``create_scene_object_combination`` (the same builder collect_3dgs.py uses):
keeps the asset catalogs' metadata (names, scales, voxel sizes) but resolves the
asset files as PLYs under --assets-root, because the catalogs reference an SPZ
library that may not be unpacked locally.

Each produced combination is a triple ready for ``run_vlm_nav_3dgs.py --world env``:
  combo_*.spz (merged scene+objects) + combo_*_background.spz + combo_*.json
  (semantic manifest with object_positions_flu etc.)

Example (inside the vlm-nav container, /assets = the 3DGS asset library):

    python scripts/vlm_nav_make_scene.py --assets-root /assets \
        --scene-key classroom --object-keys bookshelf,ammo_box --n 2 \
        --out outputs/vlm_nav/combos
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent
DIFFAERO = Path(
    os.environ.get("DIFFAERO_ROOT", REPO / "third_party" / "diffaero")
).expanduser().resolve()
if not DIFFAERO.is_dir():
    DIFFAERO = REPO
CATALOGS = {
    "scene": (DIFFAERO / "cfg/env/assets/3dgs_scenes.yaml", "3dgs_scenes"),
    "object": (DIFFAERO / "cfg/env/assets/3dgs_objects.yaml", "3dgs_objects"),
    "obstacle": (DIFFAERO / "cfg/env/assets/3dgs_obstacles.yaml", "3dgs_obstacles"),
}


def load_catalog_entries(
    kind: str, assets_root: Path, selected_keys: list[str] | None
) -> list[AssetEntry]:
    """Mirror load_asset_entries() but root at --assets-root and PLY files."""
    from diffaero.utils.gaussian_splatting.merge_scene_object import AssetEntry

    yaml_path, collection_key = CATALOGS[kind]
    cfg = OmegaConf.load(yaml_path)
    subdir = Path(str(cfg.assets.root)).name  # e.g. "assets/scenes" -> "scenes"
    entries: list[AssetEntry] = []
    for key in cfg.assets[collection_key].keys():
        if selected_keys and str(key) not in selected_keys:
            continue
        item = cfg.assets[collection_key][key]
        path = (assets_root / subdir / Path(str(item.path)).with_suffix(".ply").name).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{kind} asset {key!r} not found at {path}")
        if hasattr(item, "gaussian_splatting"):
            scale = float(item.gaussian_splatting.scene_scale)
            voxel_size = float(item.gaussian_splatting.voxel_size)
        else:
            scale, voxel_size = float(item.scale), 0.15
        entries.append(
            AssetEntry(
                key=str(key),
                name=str(item.get("name", key)),
                path=path,
                scale=scale,
                voxel_size=voxel_size,
                source_up_axis=str(item.get("source_up_axis", "y")),
            )
        )
    missing = set(selected_keys or []) - {e.key for e in entries}
    if missing:
        raise KeyError(f"{kind} keys not in catalog: {sorted(missing)}")
    if not entries:
        raise ValueError(f"no {kind} entries selected from {yaml_path}")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--assets-root", type=str, required=True,
                        help="dir containing scenes/ objects/ obstacles/ PLY libraries")
    parser.add_argument("--scene-key", type=str, required=True)
    parser.add_argument("--object-keys", type=str, required=True,
                        help="comma-separated semantic target objects")
    parser.add_argument("--obstacle-keys", type=str, default=None)
    parser.add_argument("--num-objects", type=int, default=None,
                        help="objects per combination (default: all --object-keys)")
    parser.add_argument("--min-obstacles", type=int, default=0)
    parser.add_argument("--max-obstacles", type=int, default=0)
    parser.add_argument("--n", type=int, default=1, help="combinations to build")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    assets_root = Path(args.assets_root)
    scene_keys = [k.strip() for k in args.scene_key.split(",")]
    object_keys = [k.strip() for k in args.object_keys.split(",")]
    obstacle_keys = [k.strip() for k in (args.obstacle_keys or "").split(",") if k.strip()]

    scenes = load_catalog_entries("scene", assets_root, scene_keys)
    objects = load_catalog_entries("object", assets_root, object_keys)
    obstacles = (
        load_catalog_entries("obstacle", assets_root, obstacle_keys)
        if obstacle_keys or args.max_obstacles > 0
        else []
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    from diffaero.utils.gaussian_splatting.merge_scene_object import (
        PlacementError,
        create_scene_object_combination,
    )

    built = 0
    for index in range(args.n):
        seed = (args.seed + index * 7919) % 2**32
        rng = random.Random(seed)
        try:
            combo = create_scene_object_combination(
                index=index,
                seed=seed,
                rng=rng,
                scenes=scenes,
                objects=objects,
                obstacles=obstacles,
                output_dir=out_dir,
                num_objects=args.num_objects,
                min_obstacles=args.min_obstacles,
                max_obstacles=args.max_obstacles,
                agent_radius=0.2,
                boundary_clearance=1.2,
            )
        except PlacementError as exc:
            print(f"[combo {index}] placement failed: {exc}", file=sys.stderr)
            continue
        built += 1
        print(f"[combo {index}] seed={seed}")
        print(f"  merged:    {combo.ply_path}")
        print(f"  background:{combo.background_path}")
        print(f"  manifest:  {combo.manifest_path}")
        print(f"  targets:   {[o.key for o in combo.objects]}")
    if built == 0:
        raise SystemExit("no combination could be built")


if __name__ == "__main__":
    main()
