#!/usr/bin/env python
"""Build HiGS harness episodes from the DiffAero scene+object+obstacle pipeline.

For every preset and seed this calls ``create_scene_object_combination`` from
``third_party/diffaero``, rejects combinations whose target is too far from the
room edge (and, by default, not resting near the floor), then writes:

    <out>/<preset>/seed_<seed>/   combo PLY/SPZ assets + semantic manifest
    <out>/matrix.json             harness episodes (world/renderer higs)
    <out>/build_summary.json      every attempt, selected seed, rejection reason

The generator script ``third_party/diffaero/script/vlm_nav_make_scene.py`` is
loaded with importlib at runtime, so ``--help`` works on hosts without DiffAero.
Run the actual build inside the vlm-nav container (diffaero + omegaconf + numpy).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import random
import sys
from typing import Any

REPO = Path(__file__).resolve().parent.parent
GENERATOR_SCRIPT = REPO / "third_party" / "diffaero" / "script" / "vlm_nav_make_scene.py"

AGENT_RADIUS_M = 0.2
FLOOR_MAX_OFFSET_M = 0.35
SUCCESS_DIST_M = 1.8
DEFAULT_SEED_START = 0
DEFAULT_SEED_END = 40
DEFAULT_MAX_SUPPORT_OFFSET_M = 0.25
_MAX_SUPPORT_UNSET = object()


@dataclass(frozen=True)
class Preset:
    """One scene + target object + obstacle layout + instruction."""

    name: str
    scene: str
    obj: str
    obstacles: tuple[str, ...]
    instruction: str


# Add a preset by adding one entry here; --preset choices are generated from it.
PRESETS: dict[str, Preset] = {
    "warehouse_crate": Preset(
        name="warehouse_crate",
        scene="warehouse",
        obj="wooden_crate",
        obstacles=("barrier_gate", "weathered_concrete_jersey_barrier"),
        instruction=(
            "Find the wooden crate near the edge of the warehouse and fly next to it."
        ),
    ),
    "classroom_plant": Preset(
        name="classroom_plant",
        scene="classroom",
        obj="potted_plant",
        obstacles=("broken_barrier_section",),
        instruction=(
            "Find the potted plant near the wall of the classroom and fly next to it."
        ),
    ),
    "gallery_plant": Preset(
        name="gallery_plant",
        scene="indoor_009_museum_grand_gallery",
        obj="potted_plant",
        obstacles=("barrier_gate",),
        instruction=(
            "Find the potted plant at the edge of the museum gallery and fly next to it."
        ),
    ),
}


@dataclass
class Candidate:
    """A built combination that passed manifest loading and pose selection."""

    seed: int
    combo: Any
    record: dict[str, Any]
    edge_distance_m: float
    floor_offset_m: float
    start_pos: tuple[float, float, float] | None
    start_yaw_deg: float | None


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return parsed


def _optional_float(value: str) -> float | None:
    if value.strip().lower() in ("none", "null", "off"):
        return None
    return float(value)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return parsed


def _split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_seeds(value: str) -> list[int]:
    """Parse ``START:END`` (END exclusive) or a comma-separated seed list."""
    text = value.strip()
    if not text:
        return []
    if ":" in text:
        if text.count(":") != 1:
            raise ValueError(f"--seeds {value!r}: expected START:END")
        start_text, end_text = text.split(":")
        start = int(start_text)
        end = int(end_text)
        if start < 0 or end < 0:
            raise ValueError(f"--seeds {value!r}: seeds must be >= 0")
        if end <= start:
            raise ValueError(f"--seeds {value!r}: END must be greater than START")
        return list(range(start, end))
    seeds: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        seed = int(item)
        if seed < 0:
            raise ValueError(f"--seeds {value!r}: seeds must be >= 0")
        if seed not in seeds:
            seeds.append(seed)
    return seeds


def _resolve_seeds(seeds_arg: str | None, seed_arg: int | None) -> list[int]:
    if seeds_arg is not None:
        seeds = _parse_seeds(seeds_arg)
        if seeds:
            if seed_arg is not None:
                print(
                    "warning: --seed is ignored because --seeds is non-empty",
                    file=sys.stderr,
                )
            return seeds
        if seed_arg is None:
            raise ValueError("--seeds produced no seeds; pass --seed or a non-empty range")
        return [seed_arg]
    if seed_arg is not None:
        return [seed_arg]
    return list(range(DEFAULT_SEED_START, DEFAULT_SEED_END))


def _resolve_presets(
    names: list[str] | None,
    scene_override: str | None,
    objects_override: str | None,
    obstacles_override: str | None,
) -> list[Preset]:
    selected_names = names or ["all"]
    ordered: list[Preset] = []
    seen: set[str] = set()
    for name in selected_names:
        candidates = PRESETS.values() if name == "all" else (PRESETS[name],)
        for preset in candidates:
            if preset.name not in seen:
                seen.add(preset.name)
                ordered.append(preset)

    scene = scene_override.strip() if scene_override is not None else None
    if scene is not None and not scene:
        raise ValueError("--scene must not be empty")

    object_keys = _split_csv(objects_override)
    if len(object_keys) > 1:
        raise ValueError(
            "--objects accepts exactly one key: combos place a single target object"
        )
    obj = object_keys[0] if object_keys else None

    obstacle_keys = (
        None if obstacles_override is None else tuple(_split_csv(obstacles_override))
    )

    resolved: list[Preset] = []
    for preset in ordered:
        resolved.append(
            Preset(
                name=preset.name,
                scene=scene or preset.scene,
                obj=obj or preset.obj,
                obstacles=(
                    preset.obstacles if obstacle_keys is None else obstacle_keys
                ),
                instruction=preset.instruction,
            )
        )
    return resolved


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of placement diagnostics to JSON types."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except Exception:
        return repr(value)


def _load_generator(path: Path = GENERATOR_SCRIPT) -> Any:
    """Import the DiffAero generator script without making it a package module."""
    if not path.is_file():
        raise FileNotFoundError(f"generator script not found: {path}")
    name = "diffaero_script_vlm_nav_make_scene"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _load_builder_api() -> tuple[Any, type[Exception]]:
    from diffaero.utils.gaussian_splatting.merge_scene_object import (  # noqa: PLC0415
        PlacementError,
        create_scene_object_combination,
    )

    return create_scene_object_combination, PlacementError


def _load_scene_combo_api() -> tuple[Any, Any, Any]:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from vlm_nav.scene_combo import (  # noqa: PLC0415
        choose_start_pose,
        combo_summary,
        load_scene_combo,
    )

    return choose_start_pose, combo_summary, load_scene_combo


def _catalog_entries(
    generator: Any,
    assets_root: Path,
    cache: dict[tuple[str, tuple[str, ...]], list[Any]],
    kind: str,
    keys: tuple[str, ...],
) -> list[Any]:
    cache_key = (kind, keys)
    if cache_key not in cache:
        cache[cache_key] = generator.load_catalog_entries(
            kind, assets_root, list(keys)
        )
    return cache[cache_key]


def _obstacle_entries(
    generator: Any,
    assets_root: Path,
    keys: tuple[str, ...],
    cache: dict[tuple[str, tuple[str, ...]], list[Any]],
) -> list[Any]:
    """Load obstacle assets, falling back to the object catalog by key.

    The obstacle catalog does not contain every asset used as an obstacle
    (``oil_drum`` and ``concrete_column`` are object-catalog entries), while
    ``create_scene_object_combination`` only cares about the ``AssetEntry``.
    """
    if not keys:
        return []
    entries: list[Any] = []
    unresolved: list[str] = []
    for key in keys:
        try:
            entries.extend(
                _catalog_entries(generator, assets_root, cache, "obstacle", (key,))
            )
            continue
        except KeyError:
            pass
        try:
            entries.extend(
                _catalog_entries(generator, assets_root, cache, "object", (key,))
            )
        except (KeyError, FileNotFoundError, ValueError) as exc:
            unresolved.append(f"{key}: {exc}")
    if unresolved:
        raise RuntimeError(
            "obstacle keys not found in the obstacle or object catalogs: "
            + "; ".join(unresolved)
        )
    return entries


def _evaluate_combo(
    combo: Any,
    *,
    edge_margin_m: float,
    edge_fraction: float,
    require_floor: bool,
    choose_start_pose: Any,
) -> tuple[float, float, tuple[float, float, float] | None, float | None, list[str]]:
    target = combo.object_at(0)
    edge_distance = float(combo.edge_distance_m(0))
    (min_x, min_y), (max_x, max_y) = combo.xy_bounds_flu()
    navigable_span = min(max_x - min_x, max_y - min_y)
    effective_margin = max(float(edge_margin_m), float(edge_fraction) * navigable_span)
    edge_ratio = edge_distance / navigable_span if navigable_span > 0 else float("inf")
    low_z = (
        float(target.bbox_min_flu[2])
        if target.bbox_min_flu is not None
        else float(target.position_flu[2])
    )
    floor_offset = low_z - float(combo.ground_z_flu())

    reasons: list[str] = []
    if edge_distance > effective_margin:
        reasons.append(
            f"edge_distance_m={edge_distance:.3f} exceeds "
            f"effective_edge_margin_m={effective_margin:.3f} "
            f"(edge_margin_m={edge_margin_m:.3f}, "
            f"edge_fraction={edge_fraction:.3f} of {navigable_span:.3f} m)"
        )
    if require_floor and floor_offset > FLOOR_MAX_OFFSET_M:
        reasons.append(
            f"floor_offset_m={floor_offset:.3f} exceeds "
            f"floor_max_offset_m={FLOOR_MAX_OFFSET_M:.3f}"
        )

    start_pos: tuple[float, float, float] | None = None
    start_yaw_deg: float | None = None
    try:
        raw_pos, raw_yaw = choose_start_pose(combo)
        start_pos = tuple(float(value) for value in raw_pos)
        start_yaw_deg = float(raw_yaw)
    except Exception as exc:  # a combo without a usable start cannot be an episode
        reasons.append(f"start_pose_failed: {type(exc).__name__}: {exc}")

    return edge_distance, floor_offset, start_pos, start_yaw_deg, reasons


def _build_attempt(
    *,
    preset: Preset,
    seed: int,
    seed_dir: Path,
    scenes: list[Any],
    objects: list[Any],
    obstacles: list[Any],
    options: argparse.Namespace,
    create_combination: Any,
    placement_error: type[Exception],
    load_scene_combo: Any,
    choose_start_pose: Any,
) -> tuple[dict[str, Any], Candidate | None]:
    seed_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "preset": preset.name,
        "seed": seed,
        "index": seed,
        "output_dir": str(seed_dir),
        "status": "build_failed",
        "reasons": [],
    }

    rng = random.Random(seed)
    try:
        raw = create_combination(
            index=seed,
            seed=seed,
            rng=rng,
            scenes=scenes,
            objects=objects,
            obstacles=obstacles,
            output_dir=seed_dir,
            num_objects=1,
            min_obstacles=len(obstacles),
            max_obstacles=len(obstacles),
            agent_radius=AGENT_RADIUS_M,
            boundary_clearance=options.boundary_clearance,
            target_max_support_offset=options.max_support_offset,
        )
    except placement_error as exc:
        reason = str(getattr(exc, "reason", exc))
        record.update(
            {
                "error_type": "PlacementError",
                "placement_reason": reason,
                "error": str(exc),
                "diagnostics": _jsonable(getattr(exc, "diagnostics", {})),
                "reasons": [f"placement_{reason}"],
            }
        )
        return record, None
    except Exception as exc:
        record.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "reasons": [f"build_failed: {type(exc).__name__}: {exc}"],
            }
        )
        return record, None

    record.update(
        {
            "status": "manifest_failed",
            "combined_ply": str(raw.ply_path),
            "background_ply": str(raw.background_path),
            "semantic_manifest": str(raw.manifest_path),
        }
    )
    try:
        combo = load_scene_combo(raw.manifest_path)
    except Exception as exc:
        record.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "reasons": [f"manifest_load_failed: {type(exc).__name__}: {exc}"],
            }
        )
        return record, None

    edge, floor_offset, start_pos, start_yaw, reasons = _evaluate_combo(
        combo,
        edge_margin_m=options.edge_margin,
        edge_fraction=options.edge_fraction,
        require_floor=options.require_floor,
        choose_start_pose=choose_start_pose,
    )
    target = combo.object_at(0)
    (min_x, min_y), (max_x, max_y) = combo.xy_bounds_flu()
    navigable_span = min(max_x - min_x, max_y - min_y)
    effective_margin = max(
        float(options.edge_margin), float(options.edge_fraction) * navigable_span
    )
    edge_ratio = edge / navigable_span if navigable_span > 0 else None
    record.update(
        {
            "status": "rejected" if reasons else "accepted",
            "reasons": reasons,
            "combined_ply": str(combo.merged_ply),
            "background_ply": (
                None if combo.background_ply is None else str(combo.background_ply)
            ),
            "semantic_manifest": str(combo.path),
            "scene": combo.scene_key,
            "target_index": 0,
            "target_key": target.key,
            "target_name": target.name,
            "edge_distance_m": edge,
            "edge_ok": edge <= effective_margin,
            "edge_margin_m": options.edge_margin,
            "effective_edge_margin_m": effective_margin,
            "edge_fraction": options.edge_fraction,
            "edge_ratio": edge_ratio,
            "navigable_span_m": navigable_span,
            "ground_z_flu": float(combo.ground_z_flu()),
            "floor_offset_m": floor_offset,
            "floor_ok": floor_offset <= FLOOR_MAX_OFFSET_M,
            "floor_required": options.require_floor,
            "floor_max_offset_m": FLOOR_MAX_OFFSET_M,
            "start_pos": None if start_pos is None else list(start_pos),
            "start_yaw_deg": start_yaw,
        }
    )
    candidate = Candidate(
        seed=seed,
        combo=combo,
        record=record,
        edge_distance_m=edge,
        floor_offset_m=floor_offset,
        start_pos=start_pos,
        start_yaw_deg=start_yaw,
    )
    return record, candidate


def _make_episode(preset: Preset, candidate: Candidate) -> dict[str, Any]:
    if candidate.start_pos is None or candidate.start_yaw_deg is None:
        raise ValueError(f"seed {candidate.seed} has no start pose")
    combo = candidate.combo
    target = combo.object_at(0)
    return {
        "name": f"{preset.name}_seed_{candidate.seed}",
        "preset": preset.name,
        "world": "higs",
        "combined_ply": str(combo.merged_ply),
        "background_ply": (
            None if combo.background_ply is None else str(combo.background_ply)
        ),
        "semantic_manifest": str(combo.path),
        "target_index": 0,
        "target_name": target.name,
        "instruction": preset.instruction,
        "start_pos": [float(value) for value in candidate.start_pos],
        "start_yaw_deg": float(candidate.start_yaw_deg),
        "success_dist": SUCCESS_DIST_M,
        "renderer": "higs",
        "scene": combo.scene_key,
        "target_edge_distance_m": float(candidate.edge_distance_m),
    }


def _build_preset(
    *,
    preset: Preset,
    options: argparse.Namespace,
    seeds: list[int],
    generator: Any,
    assets_root: Path,
    cache: dict[tuple[str, tuple[str, ...]], list[Any]],
    create_combination: Any,
    placement_error: type[Exception],
    load_scene_combo: Any,
    choose_start_pose: Any,
    combo_summary: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    scenes = _catalog_entries(
        generator, assets_root, cache, "scene", (preset.scene,)
    )
    objects = _catalog_entries(
        generator, assets_root, cache, "object", (preset.obj,)
    )
    obstacles = _obstacle_entries(
        generator, assets_root, preset.obstacles, cache
    )

    attempts: list[dict[str, Any]] = []
    accepted: list[Candidate] = []
    built: list[Candidate] = []
    for seed in seeds:
        seed_dir = (options.out_root / preset.name / f"seed_{seed}").resolve()
        record, candidate = _build_attempt(
            preset=preset,
            seed=seed,
            seed_dir=seed_dir,
            scenes=scenes,
            objects=objects,
            obstacles=obstacles,
            options=options,
            create_combination=create_combination,
            placement_error=placement_error,
            load_scene_combo=load_scene_combo,
            choose_start_pose=choose_start_pose,
        )
        attempts.append(record)
        if candidate is None:
            continue
        built.append(candidate)
        if record["status"] == "accepted":
            accepted.append(candidate)
            if len(accepted) >= options.n:
                break

    fallback_used = False
    if not accepted and built:
        candidates = [item for item in built if item.start_pos is not None]
        if candidates:
            best = min(
                candidates,
                key=lambda item: (
                    item.edge_distance_m,
                    item.floor_offset_m,
                    item.seed,
                ),
            )
            best.record["selected_as_fallback"] = True
            accepted = [best]
            fallback_used = True

    if not accepted:
        status = "failed"
    elif fallback_used:
        status = "fallback"
    else:
        status = "accepted"

    episodes = [_make_episode(preset, candidate) for candidate in accepted]
    summary: dict[str, Any] = {
        "name": preset.name,
        "scene": preset.scene,
        "object": preset.obj,
        "objects": [preset.obj],
        "obstacles": list(preset.obstacles),
        "instruction": preset.instruction,
        "status": status,
        "requested_accepted": options.n,
        "accepted_count": len(accepted),
        "fallback_used": fallback_used,
        "selected_seeds": [candidate.seed for candidate in accepted],
        "selection": (
            "none"
            if not accepted
            else ("best_edge_fallback" if fallback_used else "first_n_accepted")
        ),
        "selection_reason": (
            "no seed met acceptance; selected the smallest edge_distance_m"
            if fallback_used
            else (
                "no usable combination was built"
                if not accepted
                else "first seeds that met acceptance"
            )
        ),
        "attempts": attempts,
        "selected": [
            {
                "seed": candidate.seed,
                "edge_distance_m": candidate.edge_distance_m,
                "floor_offset_m": candidate.floor_offset_m,
                "combo": combo_summary(candidate.combo),
            }
            for candidate in accepted
        ],
    }
    return summary, episodes, status


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--preset",
        action="append",
        choices=[*PRESETS.keys(), "all"],
        metavar="NAME",
        default=None,
        help="preset to build; repeatable (default: all)",
    )
    parser.add_argument(
        "--scene", default=None, help="override the scene key for every preset"
    )
    parser.add_argument(
        "--objects",
        default=None,
        help="override the target object key (comma-separated; exactly one)",
    )
    parser.add_argument(
        "--obstacles",
        default=None,
        help="override obstacle keys (comma-separated; empty string clears)",
    )
    parser.add_argument(
        "--assets-root",
        default="/assets",
        help="directory containing scenes/, objects/, obstacles/ PLY libraries",
    )
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--seeds",
        default=None,
        help="seed range START:END (END exclusive) or comma list (default: 0:40)",
    )
    parser.add_argument(
        "--seed",
        type=_nonnegative_int,
        default=None,
        help="single seed used when --seeds is omitted",
    )
    parser.add_argument(
        "--n",
        type=_positive_int,
        default=1,
        help="accepted combinations per preset (default: 1)",
    )
    parser.add_argument(
        "--edge-margin",
        type=_nonnegative_float,
        default=2.0,
        metavar="METRES",
        help="absolute max target distance from the navigable-floor edge "
             "(default: 2.0)",
    )
    parser.add_argument(
        "--edge-fraction",
        type=_nonnegative_float,
        default=0.25,
        metavar="FRACTION",
        help="also accept a target within this fraction of the smaller "
             "navigable-floor span (default: 0.25)",
    )
    parser.add_argument(
        "--require-floor",
        action="store_true",
        default=True,
        help="require the target to rest within 0.35 m of the floor (default: on)",
    )
    parser.add_argument(
        "--allow-support-placement",
        action="store_true",
        help="disable the floor requirement (allows tables/shelves as support)",
    )
    parser.add_argument(
        "--boundary-clearance",
        type=_nonnegative_float,
        default=0.5,
        metavar="METRES",
        help="generator placement boundary clearance (default: 0.5)",
    )
    parser.add_argument(
        "--max-support-offset",
        type=_optional_float,
        default=_MAX_SUPPORT_UNSET,
        metavar="METRES",
        help="generator target support-height filter; 'none' disables it "
        "(default: 0.25, or none with --allow-support-placement)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        presets = _resolve_presets(
            args.preset, args.scene, args.objects, args.obstacles
        )
        seeds = _resolve_seeds(args.seeds, args.seed)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))

    if args.max_support_offset is _MAX_SUPPORT_UNSET:
        args.max_support_offset = (
            None if args.allow_support_placement else DEFAULT_MAX_SUPPORT_OFFSET_M
        )
    if args.max_support_offset is not None and args.max_support_offset < 0:
        parser.error("--max-support-offset must be >= 0 or 'none'")

    args.require_floor = not args.allow_support_placement
    args.out_root = Path(args.out).expanduser().resolve()
    assets_root = Path(args.assets_root).expanduser().resolve()
    args.out_root.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat()

    try:
        generator = _load_generator()
        create_combination, placement_error = _load_builder_api()
        choose_start_pose, combo_summary, load_scene_combo = _load_scene_combo_api()
    except Exception as exc:
        print(
            "error: this builder must run inside the vlm-nav container with "
            "diffaero, omegaconf and numpy installed",
            file=sys.stderr,
        )
        print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    cache: dict[tuple[str, tuple[str, ...]], list[Any]] = {}
    preset_summaries: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    failed_presets: list[str] = []

    for preset in presets:
        print(f"[{preset.name}] building scene={preset.scene} object={preset.obj}")
        try:
            summary, preset_episodes, status = _build_preset(
                preset=preset,
                options=args,
                seeds=seeds,
                generator=generator,
                assets_root=assets_root,
                cache=cache,
                create_combination=create_combination,
                placement_error=placement_error,
                load_scene_combo=load_scene_combo,
                choose_start_pose=choose_start_pose,
                combo_summary=combo_summary,
            )
        except Exception as exc:
            status = "failed"
            preset_episodes = []
            summary = {
                "name": preset.name,
                "scene": preset.scene,
                "object": preset.obj,
                "objects": [preset.obj],
                "obstacles": list(preset.obstacles),
                "instruction": preset.instruction,
                "status": "failed",
                "requested_accepted": args.n,
                "accepted_count": 0,
                "fallback_used": False,
                "selected_seeds": [],
                "selection": "none",
                "attempts": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  failed: {type(exc).__name__}: {exc}", file=sys.stderr)

        preset_summaries.append(summary)
        episodes.extend(preset_episodes)
        if status == "failed":
            failed_presets.append(preset.name)

        if status == "fallback" and summary["selected_seeds"]:
            seed = summary["selected_seeds"][0]
            print(
                f"  fallback: no seed met the acceptance rules; selected seed {seed} "
                f"(edge {summary['selected'][0]['edge_distance_m']:.3f} m)"
            )
        else:
            print(
                f"  {status}: {len(preset_episodes)} episode(s)"
                + (
                    f", selected seeds {summary['selected_seeds']}"
                    if summary["selected_seeds"]
                    else ""
                )
            )

    matrix = {"generated_at": generated_at, "episodes": episodes}
    summary_payload = {
        "generated_at": generated_at,
        "output_root": str(args.out_root),
        "assets_root": str(assets_root),
        "config": {
            "presets": [preset.name for preset in presets],
            "seeds": seeds,
            "n": args.n,
            "edge_margin_m": args.edge_margin,
            "edge_fraction": args.edge_fraction,
            "require_floor": args.require_floor,
            "floor_max_offset_m": (
                FLOOR_MAX_OFFSET_M if args.require_floor else None
            ),
            "boundary_clearance_m": args.boundary_clearance,
            "max_support_offset_m": args.max_support_offset,
            "agent_radius_m": AGENT_RADIUS_M,
            "success_dist_m": SUCCESS_DIST_M,
        },
        "presets": preset_summaries,
        "failed_presets": failed_presets,
    }

    matrix_path = args.out_root / "matrix.json"
    summary_path = args.out_root / "build_summary.json"
    _write_json(matrix_path, matrix)
    _write_json(summary_path, summary_payload)

    print(f"wrote {matrix_path} ({len(episodes)} episodes)")
    print(f"wrote {summary_path}")
    if failed_presets:
        print(
            "error: no usable combination for preset(s): "
            + ", ".join(failed_presets),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
