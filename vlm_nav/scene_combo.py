"""Scene + object + obstacle combinations produced by the DiffAero pipeline.

``script/vlm_nav_make_scene.py`` in ``third_party/diffaero`` merges a 3DGS
scene with catalog objects and obstacles and writes the merged asset plus a
semantic manifest. This module is the harness-side reader: it validates the
manifest, converts 3DGS coordinates to the world's FLU frame, and picks a
search start pose that leaves the target near the room edge.

Frame convention (matches ``vlm_nav.cpu_renderer`` and
``diffaero.utils.gaussian_splatting.merge_scene_object``)::

    abs_3dgs = flu_to_3dgs(p_flu) + center
    flu_to_3dgs(x, y, z) = (-y, z, -x)

so for a scene-relative 3DGS point ``rel = abs_3dgs - center``::

    p_flu = (-rel_z, -rel_x, rel_y)

The manifest's ``object_positions_flu`` fields are already in that FLU frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class SceneEntity:
    """One placed object or obstacle from a combination manifest."""

    key: str
    name: str
    kind: str
    position_flu: tuple[float, float, float]
    xy_radius_m: float
    bbox_min_flu: tuple[float, float, float] | None = None
    bbox_max_flu: tuple[float, float, float] | None = None

    @property
    def is_target(self) -> bool:
        return self.kind == "object"


@dataclass(frozen=True)
class SceneCombo:
    """Validated view over one ``combo_*.json`` semantic manifest."""

    path: Path
    merged_ply: Path
    background_ply: Path | None
    scene_key: str
    scene_name: str
    objects: tuple[SceneEntity, ...]
    obstacles: tuple[SceneEntity, ...]
    scene_scale: float
    raw: dict[str, Any]

    @property
    def entities(self) -> tuple[SceneEntity, ...]:
        return self.objects + self.obstacles

    def object_at(self, index: int = 0) -> SceneEntity:
        if not self.objects:
            raise ValueError(f"{self.path} has no target objects")
        return self.objects[index % len(self.objects)]

    def _scene_center_3dgs(self) -> np.ndarray:
        center = self.raw.get("merged_scene_center_3dgs")
        if center is None:
            return np.zeros(3, dtype=float)
        return np.asarray(center, dtype=float).reshape(3)

    def xy_bounds_flu(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Scene XY extent in FLU, from the placement diagnostics when present.

        ``placement.bbox_min_xz`` is in absolute 3DGS x/z, while FLU is
        scene-centred, so the merge centre must be subtracted before the axis
        swap (``x_flu = -rel_z``, ``y_flu = -rel_x``).
        """
        placement = self.raw.get("placement") or {}
        placeable = placement.get("placeable_points_3dgs")
        if placeable:
            # The raw bbox includes distant outliers; the placeable ground
            # grid is the navigable floor the objects are sampled from, which
            # is what "near the room edge" has to mean.
            points = np.asarray(placeable, dtype=float)
            center = self._scene_center_3dgs()
            rel = points - center.reshape(1, 3)
            flu_x = -rel[:, 2]
            flu_y = -rel[:, 0]
            return (
                (float(flu_x.min()), float(flu_y.min())),
                (float(flu_x.max()), float(flu_y.max())),
            )
        xz_min = placement.get("bbox_min_xz")
        xz_max = placement.get("bbox_max_xz")
        if xz_min is not None and xz_max is not None:
            center = self._scene_center_3dgs()
            rel_x_min = float(xz_min[0]) - center[0]
            rel_x_max = float(xz_max[0]) - center[0]
            rel_z_min = float(xz_min[1]) - center[2]
            rel_z_max = float(xz_max[1]) - center[2]
            return (
                (-rel_z_max, -rel_x_max),
                (-rel_z_min, -rel_x_min),
            )
        points = [
            entity.position_flu for entity in self.entities
        ]
        if not points:
            raise ValueError(f"{self.path} carries no placement bounds")
        arr = np.asarray(points, dtype=float)
        return (
            (float(arr[:, 0].min()), float(arr[:, 1].min())),
            (float(arr[:, 0].max()), float(arr[:, 1].max())),
        )

    def ground_z_flu(self) -> float:
        """Floor height in FLU.

        The instances carry the already-converted
        ``placement_global_ground_z_flu``. The top-level
        ``placement.ground_y`` is absolute 3DGS y, so it needs the merge
        centre subtracted before it can be used as a FLU altitude.
        """
        for instance in self.raw.get("object_instances") or []:
            value = instance.get("placement_global_ground_z_flu")
            if value is not None:
                return float(value)
        placement = self.raw.get("placement") or {}
        if "ground_y" in placement:
            return float(placement["ground_y"]) - float(self._scene_center_3dgs()[1])
        lows = [
            entity.bbox_min_flu[2]
            for entity in self.entities
            if entity.bbox_min_flu is not None
        ]
        if lows:
            return float(min(lows))
        return min(entity.position_flu[2] for entity in self.entities)

    def edge_distance_m(self, index: int = 0) -> float:
        """Distance from the target's XY to the nearest scene XY edge."""
        target = self.object_at(index)
        (min_x, min_y), (max_x, max_y) = self.xy_bounds_flu()
        x, y = target.position_flu[:2]
        return float(min(x - min_x, max_x - x, y - min_y, max_y - y))


def _resolve_path(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _entity(
    raw_instances: Sequence[dict[str, Any]],
    positions: Sequence[Sequence[float]],
    radii: Sequence[float],
    bbox_mins: Sequence[Sequence[float]] | None,
    bbox_maxs: Sequence[Sequence[float]] | None,
    *,
    kind: str,
) -> tuple[SceneEntity, ...]:
    entities: list[SceneEntity] = []
    for index, position in enumerate(positions):
        source = raw_instances[index] if index < len(raw_instances) else {}
        key = str(source.get("key", f"{kind}_{index}"))
        name = str(source.get("name", key)).strip() or key
        bbox_min = None
        bbox_max = None
        if bbox_mins is not None and index < len(bbox_mins):
            bbox_min = tuple(float(v) for v in bbox_mins[index][:3])
        if bbox_maxs is not None and index < len(bbox_maxs):
            bbox_max = tuple(float(v) for v in bbox_maxs[index][:3])
        radius = float(radii[index]) if index < len(radii) else 0.5
        entities.append(
            SceneEntity(
                key=key,
                name=name,
                kind=kind,
                position_flu=tuple(float(v) for v in position[:3]),
                xy_radius_m=max(radius, 0.05),
                bbox_min_flu=bbox_min,
                bbox_max_flu=bbox_max,
            )
        )
    return tuple(entities)


def load_scene_combo(path: str | Path) -> SceneCombo:
    """Read and validate a ``create_scene_object_combination`` manifest."""
    manifest_path = Path(path).expanduser().resolve()
    with open(manifest_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{manifest_path}: manifest must be a JSON object")

    positions = raw.get("object_positions_flu") or []
    if not positions:
        raise ValueError(f"{manifest_path}: no object_positions_flu entries")
    object_radii = raw.get("object_xy_radii_flu") or [0.5] * len(positions)
    objects = _entity(
        raw.get("source_objects") or [],
        positions,
        object_radii,
        raw.get("object_bbox_mins_flu"),
        raw.get("object_bbox_maxs_flu"),
        kind="object",
    )
    obstacle_positions = raw.get("obstacle_positions_flu") or []
    obstacles = _entity(
        raw.get("source_obstacles") or [],
        obstacle_positions,
        raw.get("obstacle_xy_radii_flu") or [0.5] * len(obstacle_positions),
        raw.get("obstacle_bbox_mins_flu"),
        raw.get("obstacle_bbox_maxs_flu"),
        kind="obstacle",
    )

    merged = _resolve_path(manifest_path.parent, raw.get("merged_ply"))
    if merged is None:
        raise ValueError(f"{manifest_path}: merged_ply is missing")
    scene = raw.get("source_scene") or {}
    return SceneCombo(
        path=manifest_path,
        merged_ply=merged,
        background_ply=_resolve_path(manifest_path.parent, raw.get("background_ply")),
        scene_key=str(scene.get("key", "scene")),
        scene_name=str(scene.get("name", scene.get("key", "scene"))),
        objects=objects,
        obstacles=obstacles,
        scene_scale=float(raw.get("scene_scale", 1.0) or 1.0),
        raw=raw,
    )


def scene_flu_points(scene_xyz: Any, scene_center: Any) -> np.ndarray:
    """Convert absolute 3DGS points to FLU (metres, scene-centred)."""
    xyz = scene_xyz.detach().float().cpu().numpy() if hasattr(scene_xyz, "detach") else np.asarray(scene_xyz)
    center = scene_center.detach().float().cpu().numpy() if hasattr(scene_center, "detach") else np.asarray(scene_center)
    rel = np.asarray(xyz, dtype=np.float64) - np.asarray(center, dtype=np.float64).reshape(1, 3)
    return np.stack([-rel[:, 2], -rel[:, 0], rel[:, 1]], axis=1)


def choose_start_pose(
    combo: SceneCombo,
    target_index: int = 0,
    *,
    inward_m: float = 3.0,
    altitude_m: float = 1.2,
    edge_inset_m: float = 0.6,
) -> tuple[tuple[float, float, float], float]:
    """Pick a collision-free-looking start that faces away from the target.

    The drone starts ``inward_m`` from the target, towards the scene centre and
    at ``altitude_m`` above the local floor, with its yaw pointing away from
    the target. The search scan therefore has to turn around to find an object
    placed at a room edge, which is the scenario the combination is built for.

    Returns ``(position_flu, yaw_deg)``. This is a geometric heuristic, not a
    collision query: the caller can render a probe frame and override it.
    """
    target = combo.object_at(target_index)
    (min_x, min_y), (max_x, max_y) = combo.xy_bounds_flu()
    x, y = target.position_flu[:2]
    center = np.asarray([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0])
    to_center = center - np.asarray([x, y])
    norm = float(np.linalg.norm(to_center))
    if norm < 1e-6:
        to_center = np.asarray([1.0, 0.0])
        norm = 1.0
    step = min(float(inward_m), max(1.0, norm * 0.8))
    start_xy = np.asarray([x, y]) + to_center / norm * step
    start_xy[0] = min(max(start_xy[0], min_x + edge_inset_m), max_x - edge_inset_m)
    start_xy[1] = min(max(start_xy[1], min_y + edge_inset_m), max_y - edge_inset_m)
    z = combo.ground_z_flu() + float(altitude_m)
    yaw = math.atan2(y - start_xy[1], x - start_xy[0]) + math.pi
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    return (float(start_xy[0]), float(start_xy[1]), float(z)), math.degrees(yaw)


def combo_summary(combo: SceneCombo) -> dict[str, Any]:
    """Compact JSON-friendly description used in episode summaries."""
    return {
        "manifest": str(combo.path),
        "merged_ply": str(combo.merged_ply),
        "background_ply": None if combo.background_ply is None else str(combo.background_ply),
        "scene_key": combo.scene_key,
        "scene_name": combo.scene_name,
        "num_objects": len(combo.objects),
        "num_obstacles": len(combo.obstacles),
        "objects": [
            {"key": e.key, "name": e.name, "position_flu": list(e.position_flu)}
            for e in combo.objects
        ],
        "obstacles": [
            {"key": e.key, "name": e.name, "position_flu": list(e.position_flu)}
            for e in combo.obstacles
        ],
        "xy_bounds_flu": [list(v) for v in combo.xy_bounds_flu()],
        "ground_z_flu": combo.ground_z_flu(),
        "target_edge_distance_m": combo.edge_distance_m(0),
    }
