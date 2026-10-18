#!/usr/bin/env python
"""M2 experiment (goal.md section 4 + milestone M2): memory re-acquires a target
that left the view.

The scenario is the one goal.md asks M2 to demonstrate - *"after the target
leaves the field of view, retrieve the old evidence and re-localize"*:

1. the drone starts where the target is visible and the model captures it
   (``observe``) - the observation (frame + pose) lands in the evidence memory;
2. the target leaves the view: the model flies on (``--loss-trigger model``) or
   the environment relocates the drone behind an occluder
   (``--loss-trigger evidence``; a real model is not assumed to choose the move);
3. from there a 360 deg scan cannot see the target any more (the occluder blocks
   it - ``scenario.json`` records the analytic visibility check);
4. with the ``memory`` tool set the model calls ``query_memory``, which returns
   the old observation id, its image reference and its pose; it flies back to
   that camera pose with ``return_to_observation`` (which also restores the
   heading the frame was shot at), re-observes, goes on along the remembered
   heading with ``go_to_viewpoint`` and confirms with ``verify_goal``;
5. with the ``basic`` tool set (observe / the movement tools / verify_goal - the
   fixed-keyframe baseline) the same policy has no way to retrieve where the
   target was and stays lost.

Movement is expressed with the Phase-1 intent tools: ``return_to_observation``
for "back to the pose that frame was shot from" and ``go_to_viewpoint`` for
"on to a new place" (a raw world position stopped being a movement target, so a
leg starts with ``propose_viewpoints`` around the intended point).

Both modes run the *same* scene, instruction and budget, so the only difference
is the tool catalogue (``--toolset memory`` vs ``--toolset basic`` of
``scripts/run_vlm_nav_tools.py``). Metrics per mode (``metrics.json``) and the
side-by-side comparison (``comparison.json``) report success, calls, VLM calls,
whether memory evidence was hit, whether the model re-localized on the recalled
pose, and the final distance.

A third arm answers the question the two arms cannot: is the success of the
``memory`` mode *caused* by the retrieved evidence? ``memory_no_evidence`` runs
the *same* tool set and the same instruction, with one difference in the
archive: an observation written while the target is still in view (i.e. before
the loss of view this episode is about) is accepted and dropped, so the archive
only ever holds what the drone saw after it lost the target. ``query_memory``
is still registered, is still callable and is still called - it simply answers
``no_match``, because nothing was archived; every other result, the prompt and
the tool schemas stay identical. ``--ablate-evidence`` adds that arm to
``--modes`` (``comparison.json`` then also carries ``ablation_verdict``:
``evidence_required`` when the memory arm does better than the ablated one).

Two more arms separate the *layers* of that memory (the E2 round):

- ``memory_spatial_only`` runs the memory catalogue with the episodic *evidence*
  layer off (``--memory-episodic off``, i.e. ``episodic_memory=False``): no
  frame, no sighting and no scoped null result is written, so a retrieval can
  only answer with capture records - pose, heading, step, no image - while the
  spatial layer (candidate places, coverage) and the working layer are still
  fed by the same calls. That is the arm that tells what the *evidence* layer
  adds from what the positions alone carry.
- ``memory_text_only`` withholds the *visual* half of a retrieval: the frames a
  ``query_memory`` result carries are still there (the episodic layer is on)
  but are not decoded into the next turn's context images, so the model reads
  poses, candidate positions and the sighting record without looking at the
  recalled picture again. That is the arm that tells "the position carried it"
  from "the picture carried it" at the *consumption* side;
  ``--recall-images off`` is the same switch for a single-arm run, and it is
  independent of the two layer switches.

``--memory-spatial`` / ``--memory-episodic`` (the runner's own memory-layer
switches) set the layers of the plain ``memory`` arm, so an ablation can also be
run one arm per invocation.

The discrimination scene (E2's second task) adds a *decoy* instance and a
landmark to the synthetic scene (``--decoy-pos`` / ``--landmark-pos``): the
target is then one of two similar objects, the instruction names the relation
to the landmark, and ``metrics.json.discrimination`` reports which instance the
drone's legs and its final position went to, next to the usual success/claim
metrics.

``--scene-variant visual_required`` is the *relation* scene (G1): two identical
instances, the instruction names the correct one by a bright landmark, and the
landmark is authored **outside the start's forward view** - the frame a
``ground_target`` capture grounds in - while staying inside a scan view of the
same pose (see :func:`visual_required_geometry` and
:func:`visual_required_checks`, which refuse a layout that does not have that
property). The loss pose sees none of the three, so after the loss the relation
exists only in whatever the drone recorded. ``--relation-visible-at-start on``
is the control layout: the same scene with the landmark beside the correct
instance *inside* the forward view, so the grounding view resolves it - the same
knob moved, which is what makes the two layouts a causal contrast.
``--loss-trigger relation`` relocates the drone on the first capture whose frame
held the target *and* the landmark at once (the ``capture`` trigger fires on the
first capture that held the target, which is the grounding capture).

Mock (no weights; the scripted acceptance chain of point 1-4):

    python scripts/run_vlm_nav_m2_memory.py --backend mock \
        --instruction "Fly to the target. If it leaves your view, use your \
observation memory to find it again." --out outputs/vlm_nav/m2_mock

Real model (same scene; the environment performs the loss of view):

    python scripts/run_vlm_nav_m2_memory.py --backend hf --device cuda \
        --model-path /path/to/Qwen3-VL-8B --ply /path/scene.ply \
        --target-pos 2.0 0.5 0.0 --start-pos 0 0 0.2 \
        --instruction "Fly to the target and confirm you reached it." \
        --loss-trigger evidence --tool-budget 12 --max-turns 12 \
        --out outputs/vlm_nav/m2_hf

Without ``--ply`` the runner synthesizes the scenario scene (floor + occluder
slab + target cluster), so ``--backend mock`` is self-contained; the scene is
authored in the frame ``vlm_nav.cpu_renderer`` reads (3DGS coordinates around
its vertex centroid) and the scenario is expressed in that same frame, which
``scenario.json`` records via ``scene_origin_flu``. ``--modes`` selects which
tool sets to compare (default ``memory,basic``), ``--trials`` repeats an episode
on the same scene (sampling noise of a real model; the mock is deterministic).

Artifacts (per ``--out``): ``run_config.json``, ``scenario.json``,
``scenario_scene.ply`` (synthetic scene only), ``comparison.json``, and
``modes/<mode>/{tool_trace.jsonl,summary.json,frames/,metrics.json}``.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import struct
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from PIL import Image

RUNNER_PATH = Path(__file__).resolve().with_name("run_vlm_nav_tools.py")


def load_runner(path: Path = RUNNER_PATH):
    """Import ``scripts/run_vlm_nav_tools.py`` (sibling script, not a package)."""
    name = "diffaero_script_run_vlm_nav_tools"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner()

from vlm_nav.actions import NavAction  # noqa: E402
from vlm_nav.tools import STATUS_OK, ToolResult  # noqa: E402
from vlm_nav.vlm import MockVLM  # noqa: E402
from vlm_nav.world import SplatWorldCPU, SplatWorldHiGS, TargetSpec  # noqa: E402

MODES = ("memory", "memory_no_evidence", "basic",
         "memory_spatial_only", "memory_text_only")
#: the retrieval arms run the *same* catalogue; ``memory_no_evidence`` ablates
#: the archive, the two layer arms ablate one memory layer each (that is what
#: makes them controls)
MODE_TOOLSET = {"memory": "memory", "memory_no_evidence": "memory", "basic": "basic",
                "memory_spatial_only": "memory", "memory_text_only": "memory"}
#: the layer each arm pins, by the M2 CLI switch it pins (the other switches of
#: the arm stay the caller's, so ``--memory-spatial on`` still reaches the
#: memory arm's own constructor); the plain ``memory`` arm pins nothing.
#: ``memory_spatial_only`` pins the *production* side of the episodic layer
#: (nothing is written as evidence), ``memory_text_only`` the *consumption* side
#: of the recalled picture (the result still carries it, the runner does not
#: attach it)
MODE_SWITCH_OVERRIDES = {
    "memory_spatial_only": {"memory_episodic": "off"},
    "memory_text_only": {"recall_images": "off"},
}
#: the arms whose tool set carries the retrieval tool
EVIDENCE_MODES = ("memory", "memory_no_evidence", "memory_spatial_only", "memory_text_only")
#: the arm that removes the evidence the M2 claim rests on
ABLATED_MODE = "memory_no_evidence"
LOSS_TRIGGERS = ("auto", "model", "evidence", "capture", "relation")
#: the memory-layer switches the M2 CLI spells like the runner
SWITCH_CHOICES = ("on", "off")
DEFAULT_SWITCH = "on"
#: how far from two instances the drone has to be for the metrics to call one of
#: them "the instance it chose" (two instances closer than this are a tie)
INSTANCE_TIE_M = 0.5
#: the movement tools of the Phase-1 vocabulary; ``navigate`` is deprecated and
#: moves nothing, so the mock policy must never issue one
MOVE_TOOL_NAMES = ("approach_candidate", "go_to_viewpoint", "return_to_observation")


# ---------------------------------------------------------------------------
# scenario geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """Axis-aligned occluder (center + full size, meters, FLU)."""

    center: tuple[float, float, float]
    size: tuple[float, float, float]

    @property
    def lo(self) -> tuple[float, float, float]:
        return tuple(c - s / 2.0 for c, s in zip(self.center, self.size))

    @property
    def hi(self) -> tuple[float, float, float]:
        return tuple(c + s / 2.0 for c, s in zip(self.center, self.size))

    def blocks(self, p0: Iterable[float], p1: Iterable[float], eps: float = 1e-9) -> bool:
        """Does the segment ``p0``->``p1`` pass through the box? (slab method)"""
        lo, hi = self.lo, self.hi
        t0, t1 = 0.0, 1.0
        for axis in range(3):
            a, b = float(p0[axis]), float(p1[axis])
            delta = b - a
            if abs(delta) < eps:
                if a < lo[axis] or a > hi[axis]:
                    return False
                continue
            ta = (lo[axis] - a) / delta
            tb = (hi[axis] - a) / delta
            if ta > tb:
                ta, tb = tb, ta
            t0 = max(t0, ta)
            t1 = min(t1, tb)
            if t0 > t1:
                return False
        return True

    def to_dict(self) -> dict:
        return {"center": [round(v, 3) for v in self.center],
                "size": [round(v, 3) for v in self.size]}


# the default occluder of the synthetic scene: a slab between the loss pose and
# the target, wide/tall enough to hide it, placed clear of the flight path
DEFAULT_WALL = Box(center=(2.0, -1.5, 0.0), size=(2.5, 0.12, 3.2))

#: synthetic scene floor (FLU z of the authored floor layer, raw scenario frame)
SYNTHETIC_FLOOR_Z = -1.5
#: scene variants: ``default`` is the floating target of the M2 acceptance runs,
#: ``low_altitude`` puts the target on the ground behind a ground-level obstacle,
#: ``visual_required`` is the relation scene (two identical instances, a landmark
#: only some views hold, a loss pose that sees none of them)
SCENE_VARIANTS = ("default", "low_altitude", "visual_required")
LOW_ALTITUDE_OBSTACLE_HEIGHT_M = 0.8
LOW_ALTITUDE_TARGET_ABOVE_FLOOR_M = 0.2
#: ``visual_required`` design numbers, all bearings in degrees relative to the
#: start heading: the correct instance off the nose, the decoy (the same object,
#: pixel for pixel) almost straight ahead and nearer - the instance a policy with
#: no relation information drifts to - and the landmark beside the correct
#: instance, ``VISUAL_LAMP_OFFSET_M`` away, perpendicular to the start->instance
#: sight line on ``--relation-side``
VISUAL_INSTANCE_BEARING_DEG = 30.0
VISUAL_INSTANCE_RANGE_M = 3.9
VISUAL_DECOY_BEARING_DEG = -5.0
VISUAL_DECOY_RANGE_M = 3.4
VISUAL_LAMP_OFFSET_M = 0.85
VISUAL_LANDMARK_Z = 0.35
#: design margins (deg): the landmark must sit this far *outside* the forward
#: view (the frame a ``ground_target`` capture takes, so the relation cannot be
#: grounded there) and the instances this far inside it
VISUAL_RELATION_MARGIN_DEG = 2.0
VISUAL_INSTANCE_MARGIN_DEG = 3.0
#: ``--relation-visible-at-start on`` (the control layout): the landmark stands
#: beside the correct instance at this bearing offset *inside* the forward view,
#: at the same distance from the start - the same scene with the one knob the
#: experiment is about moved into the grounding view
VISUAL_LAMP_INVIEW_OFFSET_DEG = 9.0
#: the landmark must be at least this much closer to the correct instance than to
#: the decoy ("next to the lamp" has to name exactly one of the two)
VISUAL_RELATION_GAP_M = 1.0
#: parser defaults, needed to tell "the variant sets the geometry" from a real
#: conflict with an explicitly passed position
DEFAULT_TARGET_POS = (2.0, 0.5, 0.0)
DEFAULT_START_POS = (0.0, 0.0, 0.2)


@dataclass(frozen=True)
class FlightEnvelope:
    """Altitude envelope of a synthetic M2 scene (audit altitude_audit.md §5).

    ``z_min`` is the flyable floor (nearest surface + ``floor_margin_m``) and
    ``z_max`` the no-fly height. The cruise ceiling is ``ceiling_m`` above the
    floor, pulled down to just under an obstacle's top when that is lower: an
    obstacle whose top plus the drone radius reaches past the ceiling cannot be
    flown over, so the route has to go around it.
    """

    ground_z: float
    floor_margin_m: float = 0.25
    window_min_m: float = 0.6
    window_max_m: float = 1.2
    ceiling_m: float = 1.6
    drone_radius_m: float = 0.2
    obstacle_tops_z: tuple[float, ...] = ()

    @property
    def z_min(self) -> float:
        return self.ground_z + self.floor_margin_m

    @property
    def ceiling_z(self) -> float:
        ceiling = self.ground_z + self.ceiling_m
        for top in self.obstacle_tops_z:
            ceiling = min(ceiling, float(top) + self.drone_radius_m)
        return ceiling

    @property
    def cruise_window_z(self) -> tuple[float, float]:
        return self.ground_z + self.window_min_m, self.ground_z + self.window_max_m

    def cruise_z(self, z: float) -> float:
        lower, upper = self.cruise_window_z
        return min(max(float(z), lower), upper)

    @property
    def overflight_blocked(self) -> bool:
        """No gap between the highest obstacle top and the ceiling."""
        if not self.obstacle_tops_z:
            return False
        return self.ceiling_z <= max(self.obstacle_tops_z) + self.drone_radius_m

    def to_dict(self) -> dict:
        lower, upper = self.cruise_window_z
        return {
            "ground_z": round(self.ground_z, 3),
            "z_min": round(self.z_min, 3),
            "z_max": round(self.ceiling_z, 3),
            "cruise_window_z": [round(lower, 3), round(upper, 3)],
            "obstacle_tops_z": [round(float(v), 3) for v in self.obstacle_tops_z],
            "overflight_blocked": bool(self.overflight_blocked),
        }


@dataclass
class Scenario:
    """One M2 re-acquisition episode: where the target is, and how it gets lost."""

    target_name: str = "target"
    target_pos: tuple[float, float, float] = (2.0, 0.5, 0.0)
    target_radius: float = 0.2
    start_pos: tuple[float, float, float] = (0.0, 0.0, 0.2)
    start_yaw_deg: float = 0.0
    loss_bearing_deg: float = -90.0  # world-frame bearing of the "fly on" leg
    loss_distance_m: float = 3.0
    approach_m: float = 0.8  # closing leg along the recalled heading
    success_dist_m: float = 2.5
    hfov_deg: float = 60.0
    scan_views: int = 8
    occluders: tuple[Box, ...] = (DEFAULT_WALL,)
    #: FLU z of the scene floor, when the scene has a known ground plane; the
    #: altitude envelope and the "how high did it fly" metrics are measured
    #: against it
    ground_z: float | None = None
    #: the discrimination variant's second instance - a similar object the task
    #: is *not* about, so "did it pick the right one" becomes measurable
    decoy_pos: tuple[float, float, float] | None = None
    #: the landmark the instruction names the target by ("the red object next to
    #: the lamp"): rendered next to the target, away from the decoy
    landmark_pos: tuple[float, float, float] | None = None

    @property
    def target_top_z(self) -> float:
        return float(self.target_pos[2]) + float(self.target_radius)

    def loss_pose(self, pose: Iterable[float]) -> np.ndarray:
        """Where the drone ends up when the target leaves the view."""
        pose = np.asarray(pose, dtype=np.float64).reshape(3)
        bearing = math.radians(self.loss_bearing_deg)
        return pose + np.asarray(
            [self.loss_distance_m * math.cos(bearing),
             self.loss_distance_m * math.sin(bearing), 0.0]
        )

    def to_dict(self) -> dict:
        return {
            "target_name": self.target_name,
            "target_pos": [round(v, 3) for v in self.target_pos],
            "target_radius": self.target_radius,
            "start_pos": [round(v, 3) for v in self.start_pos],
            "start_yaw_deg": self.start_yaw_deg,
            "loss_bearing_deg": self.loss_bearing_deg,
            "loss_distance_m": self.loss_distance_m,
            "approach_m": self.approach_m,
            "success_dist_m": self.success_dist_m,
            "hfov_deg": self.hfov_deg,
            "scan_views": self.scan_views,
            "occluders": [box.to_dict() for box in self.occluders],
            "ground_z": None if self.ground_z is None else round(self.ground_z, 3),
            "decoy_pos": (
                None if self.decoy_pos is None
                else [round(float(v), 3) for v in self.decoy_pos]
            ),
            "landmark_pos": (
                None if self.landmark_pos is None
                else [round(float(v), 3) for v in self.landmark_pos]
            ),
        }


def _wrap_deg(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


def target_visibility(pose, yaw_rad: float, scenario: Scenario,
                      hfov_deg: float | None = None) -> dict:
    """Is the target inside the camera cone and not hidden by an occluder?

    Scenario geometry, evaluated analytically: the rendered frames come from the
    world, this check tells the experiment whether the target is *there* at all
    (the precondition "it left the view" would otherwise be vacuous).
    """
    pose = np.asarray(pose, dtype=np.float64).reshape(3)
    target = np.asarray(scenario.target_pos, dtype=np.float64).reshape(3)
    delta = target - pose
    distance = float(np.linalg.norm(delta))
    bearing = math.atan2(float(delta[1]), float(delta[0]))
    offset_deg = _wrap_deg(math.degrees(bearing - float(yaw_rad)))
    hfov = float(hfov_deg if hfov_deg is not None else scenario.hfov_deg)
    in_fov = abs(offset_deg) <= hfov / 2.0
    blocking = [box for box in scenario.occluders if box.blocks(pose, target)]
    return {
        "visible": bool(in_fov and not blocking),
        "in_fov": bool(in_fov),
        "occluded": bool(blocking),
        "offset_deg": round(offset_deg, 1),
        "bearing_deg": round(_wrap_deg(math.degrees(bearing)), 1),
        "distance_m": round(distance, 2),
    }


def scan_visibility(pose, scenario: Scenario, hfov_deg: float | None = None) -> dict:
    """Best case of a full-circle in-place scan from ``pose``."""
    pose = np.asarray(pose, dtype=np.float64).reshape(3)
    n = max(4, int(scenario.scan_views))
    best = None
    for k in range(n):
        yaw = 2.0 * math.pi * k / n
        state = target_visibility(pose, yaw, scenario, hfov_deg)
        if best is None or state["distance_m"] < best["distance_m"]:
            best = state
        if state["visible"]:
            return {**state, "headings_checked": k + 1}
    return {**(best or {}), "headings_checked": n}


# ---------------------------------------------------------------------------
# synthetic scenario scene
# ---------------------------------------------------------------------------


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


# FLU -> 3DGS: (-y, z, -x). ``vlm_nav.cpu_renderer`` reads PLY vertices as
# absolute 3DGS coordinates and takes the vertex centroid as the FLU origin.
M_FLU_TO_3DGS = np.asarray([[0.0, -1.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [-1.0, 0.0, 0.0]])


def _write_ply(path: Path, gaussians: Iterable[tuple]) -> None:
    gaussians = list(gaussians)
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(gaussians)}",
            "property float x",
            "property float y",
            "property float z",
            "property float f_dc_0",
            "property float f_dc_1",
            "property float f_dc_2",
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write((header + "\n").encode("ascii"))
        for xyz, sh, opacity, scale, rot in gaussians:
            handle.write(struct.pack("<14f", *list(xyz), *list(sh), opacity,
                                     *list(scale), *list(rot)))


def _box_splats(box: Box, spacing: float = 0.2,
                color: tuple[float, float, float] = (0.15, 0.15, 0.15)) -> list[tuple]:
    """Solid slab of opaque splats filling ``box`` (renders as an obstacle).

    Every axis gets its own step over the box extent (at least two layers), so a
    box larger than ``spacing`` is filled instead of collapsing into two corner
    gaussians that hide nothing.
    """
    lo, hi = np.asarray(box.lo, dtype=np.float64), np.asarray(box.hi, dtype=np.float64)
    axes = []
    for axis in range(3):
        span = float(hi[axis] - lo[axis])
        layers = max(2, int(math.ceil(span / spacing)) + 1)
        axes.append(np.linspace(lo[axis], hi[axis], layers))
    out = []
    for x in axes[0]:
        for y in axes[1]:
            for z in axes[2]:
                out.append(
                    ((float(x), float(y), float(z)), color, _logit(0.95),
                     (math.log(0.22),) * 3, (1.0, 0.0, 0.0, 0.0))
                )
    return out


def _instance_cluster(position: Iterable[float],
                      color: tuple[float, float, float] = (2.0, -1.0, -1.0)) -> list[tuple]:
    """One small opaque object (the M2 target cluster's shape and scale)."""
    center = np.asarray(position, dtype=np.float64).reshape(3)
    out = []
    for offset in ((0.0, 0.0, 0.0), (0.15, 0.1, -0.05), (0.0, 0.2, 0.1)):
        point = center + np.asarray(offset)
        out.append(
            ((float(point[0]), float(point[1]), float(point[2])), color,
             _logit(0.9), (math.log(0.15),) * 3, (1.0, 0.0, 0.0, 0.0))
        )
    return out


def _landmark_cluster(position: Iterable[float]) -> list[tuple]:
    """The instructed landmark: a bright warm block the target stands next to.

    Bigger splats than an instance on purpose - the relation ("next to the
    lamp") has to be visible in a 320x180 render, and the landmark is not what
    the drone has to find.
    """
    center = np.asarray(position, dtype=np.float64).reshape(3)
    out = []
    for offset in ((0.0, 0.0, 0.0), (0.0, 0.0, 0.35), (0.12, 0.1, 0.18)):
        point = center + np.asarray(offset)
        out.append(
            ((float(point[0]), float(point[1]), float(point[2])), (1.8, 1.6, 0.0),
             _logit(0.9), (math.log(0.18),) * 3, (1.0, 0.0, 0.0, 0.0))
        )
    return out


def scenario_gaussians(scenario: Scenario, seed: int = 0) -> list[tuple]:
    """Floor + occluders + target cluster (+ decoy and landmark) in FLU frame."""
    rng = np.random.default_rng(seed)
    gaussians: list[tuple] = []
    for _ in range(200):
        point = rng.uniform(-6.0, 6.0, size=3)
        point[2] = -1.5
        gaussians.append(
            ((float(point[0]), float(point[1]), float(point[2])), (0.0, 0.0, 0.0),
             _logit(0.8), (math.log(0.2), math.log(0.2), math.log(0.05)),
             (1.0, 0.0, 0.0, 0.0))
        )
    for box in scenario.occluders:
        gaussians.extend(_box_splats(box))
    gaussians.extend(_instance_cluster(scenario.target_pos))
    if scenario.decoy_pos is not None:
        # the same object, pixel for pixel: only the relation to the landmark
        # tells the two apart, which is what the discrimination task measures
        gaussians.extend(_instance_cluster(scenario.decoy_pos))
    if scenario.landmark_pos is not None:
        gaussians.extend(_landmark_cluster(scenario.landmark_pos))
    return gaussians


def write_scenario_scene(path: Path, scenario: Scenario,
                         seed: int = 0) -> tuple[Path, np.ndarray]:
    """Floor + occluders + target cluster written the way the renderer reads it.

    ``vlm_nav.cpu_renderer`` reads PLY vertices as absolute 3DGS coordinates and
    places the FLU origin at the vertex centroid, so gaussians written in plain
    FLU coordinates render rotated against the scenario (and the leftover
    centroid offset shifts them on top of that): the start frame comes out black
    and no frame contains the target. The geometry is therefore rotated into the
    3DGS frame around its own FLU centroid, and that centroid - the origin of the
    authored scene's FLU frame - is returned so the caller can re-express the
    scenario in the same frame (:func:`align_scenario`). All rows are authored
    with an identity rotation, so only their positions change frame.
    """
    gaussians = scenario_gaussians(scenario, seed=seed)
    positions = np.asarray([xyz for xyz, *_ in gaussians], dtype=np.float64)
    origin = positions.mean(axis=0)
    rows = [
        (tuple(float(v) for v in (np.asarray(xyz, dtype=np.float64) - origin)
               @ M_FLU_TO_3DGS.T), sh, opacity, scale, rot)
        for xyz, sh, opacity, scale, rot in gaussians
    ]
    _write_ply(path, rows)
    return path, origin


def align_scenario(scenario: Scenario, origin_flu: Iterable[float]) -> Scenario:
    """The scenario re-expressed around ``origin_flu`` (a rigid translation).

    Only positions move, so distances, bearings, the loss leg and the occlusion
    of the scenario are unchanged; this is what makes the analytics and the
    rendered scene describe one world.
    """
    origin = np.asarray(origin_flu, dtype=np.float64).reshape(3)

    def shifted(values: Iterable[float]) -> tuple[float, float, float]:
        return tuple(float(v) for v in np.asarray(values, dtype=np.float64) - origin)

    return replace(
        scenario,
        start_pos=shifted(scenario.start_pos),
        target_pos=shifted(scenario.target_pos),
        occluders=tuple(
            Box(center=shifted(box.center), size=box.size) for box in scenario.occluders
        ),
        ground_z=(
            None if scenario.ground_z is None else float(scenario.ground_z - origin[2])
        ),
        decoy_pos=None if scenario.decoy_pos is None else shifted(scenario.decoy_pos),
        landmark_pos=(
            None if scenario.landmark_pos is None else shifted(scenario.landmark_pos)
        ),
    )


def scene_for_scenario(args: argparse.Namespace, scenario: Scenario,
                       out_dir: Path) -> tuple[Path, Scenario, np.ndarray]:
    """(scene PLY, scenario in that scene's FLU frame, the frame's origin).

    With ``--ply`` the scene is the caller's, so its own centroid already defines
    the frame the scenario is expected in. Otherwise the synthetic scene is
    authored here and the scenario is moved into the authored frame.
    """
    if args.ply:
        return Path(args.ply), scenario, np.zeros(3)
    ply, origin = write_scenario_scene(
        out_dir / "scenario_scene.ply", scenario, seed=int(getattr(args, "seed", 0))
    )
    return ply, align_scenario(scenario, origin), origin


def flight_envelope(args: argparse.Namespace, scenario: Scenario) -> FlightEnvelope | None:
    """Altitude envelope of the scenario's scene (None: no known ground plane).

    The low-altitude variant adds the ground-level obstacle tops, so its ceiling
    drops to a height the obstacle cannot be flown over at.
    """
    if scenario.ground_z is None:
        return None
    obstacle_tops: tuple[float, ...] = ()
    if str(getattr(args, "scene_variant", "default")) == "low_altitude":
        obstacle_tops = tuple(
            float(box.center[2]) + 0.5 * float(box.size[2])
            for box in scenario.occluders
            if float(box.center[2]) - 0.5 * float(box.size[2]) <= scenario.ground_z + 0.1
        )
    return FlightEnvelope(ground_z=float(scenario.ground_z), obstacle_tops_z=obstacle_tops)


def build_world(args: argparse.Namespace, scenario: Scenario, ply_path: Path | str):
    """The M2 world for an already-authored scene.

    ``scenario`` must be expressed in that scene's FLU frame - build it with
    :func:`scene_for_scenario`, otherwise the rendered geometry and the analytics
    disagree. A scenario with a known ground plane also carries its altitude
    envelope: the flyable floor is the nearest surface + 0.25 m and - in the
    low-altitude variant - the cruise ceiling is the no-fly height, so the drone
    cannot climb over the ground-level obstacle.
    """
    target = TargetSpec(name=scenario.target_name,
                        position_flu=tuple(scenario.target_pos),
                        radius_m=float(scenario.target_radius))
    world_cls = SplatWorldHiGS if args.world == "higs" else SplatWorldCPU
    extra = {"device": args.device} if args.world == "higs" else {}
    envelope = flight_envelope(args, scenario)
    if envelope is not None:
        extra["z_min"] = envelope.z_min
        if str(getattr(args, "scene_variant", "default")) == "low_altitude":
            extra["z_max"] = envelope.ceiling_z
    return world_cls(
        ply_path=Path(ply_path),
        target=target,
        start_flu=tuple(scenario.start_pos),
        start_yaw_deg=float(scenario.start_yaw_deg),
        width=int(args.render_size[0]),
        height=int(args.render_size[1]),
        hfov_deg=float(scenario.hfov_deg),
        scene_scale=float(args.scene_scale),
        min_opacity=float(args.min_opacity),
        **extra,
    )


# ---------------------------------------------------------------------------
# environment script: causing the loss of view
# ---------------------------------------------------------------------------


def _vec(value: Iterable[float]) -> list[float]:
    return [round(float(v), 3) for v in np.asarray(value, dtype=np.float64).reshape(-1)]


class ScenarioDriver:
    """Watches the drone and applies the scenario's "target left the view" event.

    ``trigger="model"``: the model's own navigation leg does it (mock play).
    ``trigger="evidence"``: after the first successful ``observe`` the environment
    relocates the drone behind the occluder - a real model is not assumed to fly
    away on cue, and both modes still face the identical event.
    ``trigger="capture"``: the same relocation, armed by the first capture that
    actually had the target in view (an ``observe``, or the frame a
    ``ground_target`` grounded in - it captures too, and a model that grounds
    before it observes never arms the ``evidence`` trigger at all).

    Either way the driver records the pose/visibility transition, so the metrics
    can prove the loss actually happened and that a scan from there cannot see
    the target.
    """

    def __init__(self, world: Any, scenario: Scenario, trigger: str = "model",
                 after_observations: int = 1) -> None:
        self.world = world
        self.scenario = scenario
        self.trigger = trigger
        self.after_observations = max(1, int(after_observations))
        self.observations = 0
        self.calls = 0
        self.loss: dict | None = None
        self.states: list[dict] = []
        self.events: list[dict] = []

    # -- pose/visibility --------------------------------------------------
    def pose(self) -> np.ndarray:
        return np.asarray(self.world.position_flu, dtype=np.float64).reshape(3)

    def yaw(self) -> float:
        return float(getattr(self.world, "yaw_rad", 0.0) or 0.0)

    def visibility(self) -> dict:
        return target_visibility(self.pose(), self.yaw(), self.scenario)

    # -- hooks ------------------------------------------------------------
    def view_heading(self, name: str, args: dict | None = None):
        """The heading a capture looked at (``None``: a full-circle scan)."""
        if str(name) == "observe" and str((args or {}).get("mode") or "") == "lookaround":
            return None
        offset_deg = 0.0
        if str(name) == "observe":
            try:
                offset_deg = float((args or {}).get("yaw_deg") or 0.0)
            except (TypeError, ValueError):
                offset_deg = 0.0
        return self.yaw() + math.radians(offset_deg)

    def seen_in_view(self, name: str, args: dict | None = None, position=None) -> bool:
        """Was ``position`` (default: the target) in view of the frame captured?

        An ``observe`` can look at an offset heading (``yaw_deg``) or scan the
        whole circle (``mode=lookaround``); every other capture is the current
        pose looking forward. ``position`` is what makes the same check answer
        for the landmark - the relation scene's loss trigger is armed by a frame
        that held the target *and* the landmark at once.
        """
        where = self.scenario
        if position is not None:
            where = replace(self.scenario, target_pos=tuple(position))
        heading = self.view_heading(name, args)
        if heading is None:
            return bool(scan_visibility(self.pose(), where).get("visible"))
        return bool(target_visibility(self.pose(), heading, where)["visible"])

    @staticmethod
    def captured(result: Any) -> bool:
        """Did the call store a frame of this episode?"""
        data = getattr(result, "data", None)
        if not isinstance(data, dict):
            return False
        return bool(data.get("observation_id") or data.get("image_refs"))

    @staticmethod
    def captured_frame(result: Any) -> str | None:
        """The observation id of the frame a capture stored (``None``: none)."""
        data = getattr(result, "data", None)
        if not isinstance(data, dict):
            return None
        observation_id = data.get("observation_id")
        return None if observation_id in (None, "") else str(observation_id)

    def on_tool_call(self, name: str, args: dict, result: ToolResult) -> None:
        self.calls += 1
        status = getattr(result, "status", "")
        observing = name == "observe" and status == STATUS_OK
        if observing:
            self.observations += 1
        if self.loss is None:
            if (self.trigger == "evidence" and observing
                    and self.observations >= self.after_observations):
                self.apply_loss(mechanism="environment", after_tool=name)
            elif (self.trigger == "capture" and self.captured(result)
                    and self.seen_in_view(name, args)):
                # the M2 precondition itself: the drone just captured a frame
                # that held the target, and now the target leaves the view.
                # ``evidence`` only arms on an ``observe`` call, so a model that
                # grounds first (ground_target captures too) would never face
                # the loss it is supposed to recover from.
                self.apply_loss(mechanism="environment", after_tool=name)
            elif (self.trigger == "relation" and self.captured(result)
                    and self.seen_in_view(name, args)
                    and self.seen_in_view(name, args, self.scenario.landmark_pos)):
                # the relation scene's precondition: the drone just captured the
                # one kind of frame that names which instance the task is about -
                # both the object and the landmark in the same view - and now the
                # target leaves the view. From here on that frame is the only
                # thing that still holds the relation.
                self.apply_loss(
                    mechanism="environment", after_tool=name,
                    frame=self.captured_frame(result),
                )
        self.record_state(name, args)

    def record_state(self, name: str, args: dict | None = None) -> dict:
        """Pose state after a call; ``observe`` also gets its *view* heading.

        ``visible`` stays the drone-pose check (the loss transition and the
        metrics use it); ``view_visible``/``view_yaw_deg`` describe the frame an
        ``observe`` actually looked at (``yaw_deg`` is a view offset, the pose is
        restored afterwards).
        """
        state = self.visibility()
        entry = {
            "call": self.calls,
            "tool": str(name),
            "visible": state["visible"],
            "in_fov": state["in_fov"],
            "occluded": state["occluded"],
            "distance_m": state["distance_m"],
            "pose": _vec(self.pose()),
            "yaw_deg": round(math.degrees(self.yaw()), 1),
        }
        if str(name) == "observe":
            offset_deg = 0.0
            try:
                offset_deg = float((args or {}).get("yaw_deg") or 0.0)
            except (TypeError, ValueError):
                offset_deg = 0.0
            view_yaw = self.yaw() + math.radians(offset_deg)
            entry["view_yaw_deg"] = round(math.degrees(view_yaw), 1)
            entry["view_visible"] = target_visibility(
                self.pose(), view_yaw, self.scenario
            )["visible"]
            entry["view_mode"] = str((args or {}).get("mode") or "forward")
        if self.loss is None and self.states and self.states[-1]["visible"] and not entry["visible"]:
            self._record_loss(self.states[-1], entry, mechanism="model_move", after_tool=name)
        self.states.append(entry)
        return entry

    def apply_loss(self, mechanism: str = "environment", after_tool: str = "",
                   frame: str | None = None) -> dict:
        """Relocate the drone to the scenario's loss pose (turn + one leg)."""
        if self.loss is not None:
            return self.loss
        before = self.visibility()
        before_pose, before_yaw = self.pose().copy(), self.yaw()
        dyaw = math.radians(self.scenario.loss_bearing_deg) - before_yaw
        dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
        feedback = []
        try:
            feedback.append(str(self.world.apply_action(
                NavAction(kind="waypoint", yaw_deg=math.degrees(dyaw), reason="scenario: turn away")
            )))
            feedback.append(str(self.world.apply_action(
                NavAction(kind="waypoint", forward_m=float(self.scenario.loss_distance_m),
                          reason="scenario: fly on")
            )))
        except Exception as exc:  # a world without apply_action cannot be scripted
            feedback.append(f"{type(exc).__name__}: {exc}")
        after = self.visibility()
        expected = self.scenario.loss_pose(before_pose)
        self._record_loss(
            {"pose": _vec(before_pose), "yaw_deg": round(math.degrees(before_yaw), 1),
             "visible": before["visible"]},
            {"pose": _vec(self.pose()), "yaw_deg": round(math.degrees(self.yaw()), 1),
             "visible": after["visible"], "distance_m": after["distance_m"]},
            mechanism=mechanism,
            after_tool=after_tool,
            pose_error_m=float(np.linalg.norm(self.pose() - expected)),
            world_feedback=feedback,
            frame=frame,
        )
        return self.loss

    def _record_loss(self, before: dict, after: dict, mechanism: str,
                     after_tool: str, pose_error_m: float | None = None,
                     world_feedback: list | None = None,
                     frame: str | None = None) -> None:
        scan = scan_visibility(self.pose(), self.scenario)
        self.loss = {
            "mechanism": mechanism,
            "after_tool": str(after_tool),
            "call_index": self.calls,
            "pose_before": before.get("pose"),
            "yaw_before_deg": before.get("yaw_deg"),
            "pose": after.get("pose"),
            "yaw_deg": after.get("yaw_deg"),
            "visible_before": before.get("visible"),
            "visible_after": after.get("visible"),
            "scan_visible_after": bool(scan.get("visible")),
            "distance_m_after": after.get("distance_m"),
            "pose_error_m": None if pose_error_m is None else round(float(pose_error_m), 3),
            "world_feedback": list(world_feedback or []),
            # the frame whose capture armed the loss (the relation scene: the one
            # frame that held the object and the landmark together)
            "frame": frame,
        }
        self.events.append({"event": "loss_of_view", **self.loss})


# ---------------------------------------------------------------------------
# runtime-side evidence resolution (recalled frames + recall fields)
# ---------------------------------------------------------------------------


def _mapping(obj: Any) -> dict:
    """dict | dataclass-ish (``to_dict``/``__dict__``) -> plain dict."""
    if isinstance(obj, dict):
        return dict(obj)
    fn = getattr(obj, "to_dict", None)
    if callable(fn):
        try:
            value = fn()
        except Exception:
            value = None
        if isinstance(value, dict):
            return value
    data = getattr(obj, "__dict__", None)
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if not k.startswith("_")}
    return {}


def _frame_from_path(path: Any):
    try:
        with Image.open(path) as handle:
            arr = np.asarray(handle.convert("RGB"), dtype=np.float32)
    except Exception:
        return None
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1) / 255.0


def load_frame(ctx: Any, ref: str, kind: str, modules: dict | None = None):
    """Resolve one evidence reference to a CHW float frame (goal.md section 4:
    a resource URI must turn into something the model can actually read).

    ``observation`` ids go through the memory module (``frame`` / ``image_path``)
    or the perception module's ``resolve_observation``; ``image`` refs are paths.
    """
    if kind == "image":
        candidates = [Path(ref)] if ref else []
        extra = getattr(ctx, "extra", None)
        if isinstance(extra, dict) and extra.get("out_dir") and ref:
            candidates.append(Path(str(extra["out_dir"])) / ref)
        for candidate in candidates:
            if candidate.exists():
                return _frame_from_path(candidate), Path(ref).stem
        return None, None
    memory = getattr(ctx, "memory", None)
    for name in ("frame",):
        fn = getattr(memory, name, None)
        if callable(fn):
            try:
                frame = fn(ref)
            except Exception:
                frame = None
            if frame is not None:
                return frame, ref
    fn = getattr(memory, "image_path", None)
    if callable(fn):
        try:
            path = fn(ref)
        except Exception:
            path = None
        if path:
            return _frame_from_path(path), ref
    perception = (modules or {}).get("perception")
    resolver = getattr(perception, "resolve_observation", None)
    if callable(resolver):
        try:
            record = resolver(ctx, ref)
        except Exception:
            record = None
        frame = getattr(record, "frame", None) if record is not None else None
        if frame is not None:
            return frame, ref
    return None, None


def _pose_from_hit(hit: dict) -> tuple[list[float] | None, float | None]:
    for key in ("pose", "position_flu", "position", "world_flu"):
        value = hit.get(key) if isinstance(hit, dict) else None
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            try:
                return [float(v) for v in value[:3]], hit.get("yaw_deg")
            except (TypeError, ValueError):
                continue
    return None, None


def _pose_from_memory(memory: Any, obs_id: str) -> tuple[list[float] | None, float | None]:
    getter = getattr(memory, "get", None)
    if not callable(getter):
        return None, None
    try:
        record = getter(obs_id)
    except Exception:
        return None, None
    data = _mapping(record)
    pose, yaw = _pose_from_hit(data)
    if pose is None:
        nested = data.get("record")
        if isinstance(nested, dict):
            pose, yaw = _pose_from_hit(nested)
    return pose, yaw


def extract_recall(entry: dict | None, memory: Any) -> dict | None:
    """The evidence a model reads out of a ``query_memory`` result.

    Returns ``{observation_id, image, pose, yaw_deg, note}`` for the best hit, or
    None when the call produced no hit. The pose comes from the hit when the tool
    forwards it, otherwise from the memory record the hit cites (the M1
    ``query_memory`` summary drops list-valued fields; the record is what the
    image reference points at).
    """
    if not isinstance(entry, dict) or entry.get("tool") != "query_memory":
        return None
    data = entry.get("data") or {}
    hits = data.get("hits") if isinstance(data.get("hits"), list) else None
    if hits is None:
        matches = data.get("matches")
        hits = matches if isinstance(matches, list) else []
    evidence = entry.get("evidence") or []
    obs_id = ""
    hit: dict = {}
    for candidate in hits:
        if not isinstance(candidate, dict):
            continue
        obs_id = str(candidate.get("observation_id") or candidate.get("obs_id")
                     or candidate.get("id") or "")
        hit = candidate
        if obs_id:
            break
    if not obs_id:
        for ref in evidence:
            if str(ref.get("kind")) == "observation" and ref.get("ref"):
                obs_id = str(ref["ref"])
                break
    if not obs_id:
        return None
    pose, yaw = _pose_from_hit(hit)
    if pose is None:
        pose, yaw = _pose_from_memory(memory, obs_id)
    image = hit.get("image") or hit.get("image_path")
    if not image:
        for ref in evidence:
            if str(ref.get("kind")) == "image" and ref.get("ref"):
                image = str(ref["ref"])
                break
    return {
        "observation_id": obs_id,
        "image": str(image) if image else None,
        "pose": pose,
        "yaw_deg": yaw,
        "note": str(hit.get("note") or ""),
    }


class AblatedEvidenceMemory:
    """``memory`` proxy of the evidence ablation: pre-loss evidence is not stored.

    The archive is otherwise the real one: the object graph, the tool schemas,
    the prompt and every other result stay the model's ``memory`` arm. Only
    ``record`` is filtered - while the loss of view has not happened yet, the
    write is *accepted and dropped*, so the tool result keeps saying
    ``memory_sink=ok:record`` and the model cannot tell the difference until it
    calls ``query_memory`` and gets ``no_match``. That keeps the ablation inside
    the evidence channel: nothing is patched on the read side, and a caller that
    still has an ``obs_N`` (or the frame in its own context) is not blocked -
    what is gone is the archive of what the drone saw before the target was
    lost, which is exactly the history M2 claims the model re-localizes from.
    """

    def __init__(self, memory: Any, driver: ScenarioDriver | None = None) -> None:
        self._memory = memory
        self._driver = driver
        self.dropped = 0
        self.kept = 0

    def __getattr__(self, name: str) -> Any:
        # `_memory`/`_driver` are set by __init__; the guard only matters for an
        # instance created without it (copy/pickle), where __getattr__ would
        # otherwise recurse
        if name in ("_memory", "_driver"):
            raise AttributeError(name)
        return getattr(self._memory, name)

    def blocked(self) -> bool:
        """True while a write would be the target-visible, pre-loss evidence."""
        return self._driver is not None and self._driver.loss is None

    def record(self, *args: Any, **kwargs: Any) -> Any:
        """Store the observation, or drop it when it is the evidence under test."""
        if self.blocked():
            self.dropped += 1
            return None
        self.kept += 1
        return self._memory.record(*args, **kwargs)


class CountingVLM:
    """Counts every model call of one episode, inside *and* outside the tools.

    The tools call the same backend the loop drives (``ground_target`` grounds
    with ``ctx.extra["perception.vlm"]``, the perception harness verifies with
    it), so a trace of the loop alone undercounts what an episode spent - the
    M2 acceptance explicitly asks for the full cost. This proxy stays out of the
    way (every attribute is the backend's own) and files each ``decide`` /
    ``decide_with_tools`` call under ``loop`` or under the tool that made it,
    with the cost drivers a later analysis cannot reconstruct: prompt
    characters, context images and reply characters.
    """

    def __init__(self, vlm: Any) -> None:
        self._vlm = vlm
        self.model_calls: list[dict] = []
        self.tool_name: str | None = None

    def __getattr__(self, name: str) -> Any:
        if name in ("_vlm", "model_calls", "tool_name"):
            raise AttributeError(name)
        return getattr(self._vlm, name)

    def _record(self, call: str, frames: Any, system: Any, user: Any,
                tools_text: Any, reply: Any) -> None:
        if frames is None:
            images = 0
        elif isinstance(frames, (list, tuple)):
            images = len(frames)
        else:
            images = 1
        self.model_calls.append({
            "call": call,
            "kind": "tool" if self.tool_name else "loop",
            "tool": self.tool_name or "",
            "prompt_chars": (len(str(system or "")) + len(str(user or ""))
                             + len(str(tools_text or ""))),
            "images": int(images),
            "reply_chars": len(str(reply or "")),
        })

    def decide(self, *args: Any, **kwargs: Any) -> Any:
        reply = self._vlm.decide(*args, **kwargs)
        frames = args[0] if args else kwargs.get("frames", kwargs.get("frame_chw"))
        self._record(
            "decide", frames,
            args[1] if len(args) > 1 else kwargs.get("system_prompt"),
            args[2] if len(args) > 2 else kwargs.get("user_prompt"),
            None, reply,
        )
        return reply

    def decide_with_tools(self, *args: Any, **kwargs: Any) -> Any:
        reply = self._vlm.decide_with_tools(*args, **kwargs)
        frames = args[0] if args else kwargs.get("frames")
        self._record(
            "decide_with_tools", frames,
            args[1] if len(args) > 1 else kwargs.get("system_prompt"),
            args[2] if len(args) > 2 else kwargs.get("user_prompt"),
            args[3] if len(args) > 3 else kwargs.get("tools_text"),
            reply,
        )
        return reply


class M2Registry(RUNNER.RecordingToolRegistry):
    """Episode registry with the three runtime duties the M2 experiment needs.

    1. *Evidence resolution*: recalled ``image``/``observation`` references from a
       ``query_memory`` result are decoded into frames and published on
       ``ctx.extra["images"]`` - the channel ``AgentRuntime`` turns into the next
       turn's context images - so a recalled observation is not just a path string
       in the prompt.
    2. *Scenario hook*: every call is forwarded to the ``ScenarioDriver`` (loss of
       view) and to the model-side callbacks (the scripted mock policy reads the
       same payload a real model is told).
    3. *Evidence ablation* (``ablate_evidence=True``, the ``memory_no_evidence``
       arm): the tool context's archive is pointed at an
       :class:`AblatedEvidenceMemory` proxy before the first call, so every
       observation written before the loss of view is accepted and dropped while
       the catalogue, the prompt and all other results stay the memory arm's.
    4. *Call accounting* (``vlm_counter``): every tool handler runs with the
       :class:`CountingVLM` proxy flagged with the tool's name, so the model
       calls a tool makes internally are filed under it instead of passing as
       loop calls (or being lost).
    5. *Visual-evidence ablation* (``publish_recall_images=False``, the
       ``memory_text_only`` arm): duty 1 is switched off - a retrieval still
       returns poses, candidate positions and the sighting record, but the
       recalled frame is not attached to the next turn's context, so the model
       works from the text alone.

    Derived recall fields (``recall_observation_id`` / ``recall_pos`` /
    ``recall_yaw_deg`` / ``recall_image``) are added to ``result_index``, so a
    scripted mock can reference the retrieved evidence with ``$recall_pos``.
    """

    def __init__(self, driver: ScenarioDriver | None = None,
                 modules: dict | None = None, max_entries: int = 256,
                 ablate_evidence: bool = False, vlm_counter: Any = None,
                 publish_recall_images: bool = True) -> None:
        super().__init__(max_entries=max_entries)
        self.driver = driver
        self.modules = dict(modules or {})
        self.ablate_evidence = bool(ablate_evidence)
        self.publish_recall_images = bool(publish_recall_images)
        self.ablation: AblatedEvidenceMemory | None = None
        self.vlm_counter = vlm_counter
        self.ctx: Any = None
        self.recall_log: list[dict] = []
        self.published_frames: list[str] = []
        self.on_result: list[Callable[[dict], None]] = []

    def call(self, name: str, args: dict, ctx: Any) -> ToolResult:
        self._install_ablation(ctx)
        if self.vlm_counter is not None:
            # whatever the backend answers while this handler runs is a model
            # call the tool made, not one the loop made
            self.vlm_counter.tool_name = str(name)
        try:
            result = super().call(name, args, ctx)
        finally:
            if self.vlm_counter is not None:
                self.vlm_counter.tool_name = None
        self.ctx = ctx
        entry = self.results[-1] if self.results else None
        if entry is not None:
            self._publish_evidence(ctx, entry)
            recall = extract_recall(entry, getattr(ctx, "memory", None))
            if recall is not None:
                self.recall_log.append(recall)
                self.result_index.update(
                    {
                        "recall_observation_id": recall["observation_id"],
                        "recall_image": recall["image"] or "",
                        "recall_pos": recall["pose"],
                        "recall_pose": recall["pose"],
                        "recall_yaw_deg": recall["yaw_deg"],
                    }
                )
        if self.driver is not None:
            self.driver.on_tool_call(name, args, result)
        for callback in self.on_result:
            try:
                callback(entry)
            except Exception:
                continue
        return result

    def latest_recall(self) -> dict | None:
        return self.recall_log[-1] if self.recall_log else None

    def _install_ablation(self, ctx: Any) -> None:
        """Put the evidence ablation between the tools and the archive.

        The runtime hands the very same archive to the tool context and to
        itself (``ctx.memory`` / ``runtime.memory``); both are pointed at the
        proxy so no write path can bypass it. Done on the first call of the
        episode, before the first tool runs, so even the very first ``observe``
        is filtered.
        """
        if not self.ablate_evidence:
            return
        memory = getattr(ctx, "memory", None)
        if memory is None or isinstance(memory, AblatedEvidenceMemory):
            return
        proxy = AblatedEvidenceMemory(memory, self.driver)
        ctx.memory = proxy
        extra = getattr(ctx, "extra", None)
        runtime = extra.get("runtime") if isinstance(extra, dict) else None
        if runtime is not None and getattr(runtime, "memory", None) is memory:
            runtime.memory = proxy
        self.ablation = proxy

    def _publish_evidence(self, ctx: Any, entry: dict) -> None:
        """Resolve *recalled* evidence into context images.

        Only retrieval results are resolved (the runtime's own renders are the
        fixed keyframes the baseline works with); this is the mechanism that makes
        a recalled observation more than a path string in the prompt. With
        ``publish_recall_images=False`` (the ``memory_text_only`` arm) the
        retrieval result is left as it is: the model keeps the poses, the
        candidate positions and the sighting record, the frame stays on disk.

        What gets attached is what the retrieval *cites*: an ``image`` ref is a
        path the archive materialized, and an ``observation`` ref is a frame only
        when its hit reported one. A store whose episodic layer is off (the
        ``memory_spatial_only`` arm) answers with capture records - pose,
        heading, step - and no frame, so nothing is attached for it; resolving
        the id through the live observation repository instead would hand the
        model the very picture that layer did not carry.
        """
        if not self.publish_recall_images:
            return
        if str(entry.get("tool") or "") != "query_memory":
            return
        extra = ctx.extra if isinstance(ctx.extra, dict) else None
        if extra is None:
            return
        images = extra.setdefault("images", {})
        with_frames = _frames_reported(entry)
        for ref in entry.get("evidence") or []:
            kind, ref_id = str(ref.get("kind")), str(ref.get("ref") or "")
            if kind not in ("image", "observation") or not ref_id:
                continue
            if kind == "observation" and ref_id not in with_frames:
                continue
            frame, observation_id = load_frame(ctx, ref_id, kind, self.modules)
            if frame is None or not observation_id or observation_id in images:
                continue
            images[observation_id] = frame
            self.published_frames.append(observation_id)


# ---------------------------------------------------------------------------
# the scripted mock policy
# ---------------------------------------------------------------------------


def result_pose(entry: dict | None) -> tuple[list[float], float | None] | None:
    """(position, yaw_deg) reported by an observe/movement result."""
    data = (entry or {}).get("data") or {}
    for key in ("pose", "position_flu", "position"):
        value = data.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            try:
                pose = [float(v) for v in value[:3]]
            except (TypeError, ValueError):
                continue
            yaw = data.get("heading_deg", data.get("yaw_deg"))
            return pose, (None if yaw is None else float(yaw))
    return None


class ReacquireVLM(MockVLM):
    """Scripted stand-in for the model: the M2 acceptance chain, both tool sets.

    It consumes the tool results the runtime hands back (``on_tool_result``) the
    way a model reads them, and picks the next call from that state - the target
    position itself is never read, only poses the tools reported:

    memory: observe -> (lose the view) -> query_memory -> return_to_observation
            on the *recalled* observation (its pose and heading are restored) ->
            observe -> go_to_viewpoint on a position along the recalled heading
            -> verify_goal -> finish
    basic:  observe -> (lose the view) -> lookaround -> search step -> verify_goal
            -> finish(success=False)

    The ``memory_no_evidence`` arm plays the very same memory script: the
    catalogue and the policy are identical, and the episode ends where the
    evidence it is about is gone (``query_memory`` answers ``no_match``).

    Movement goes through the Phase-1 intent tools: ``return_to_observation``
    (the camera pose an observation was shot from) for the way back and
    ``go_to_viewpoint`` (a proposed observation position) for every leg that
    goes to a *new* place - ``navigate`` is deprecated, moves nothing and is in
    no tool set. A raw world position is no longer a movement target, so each
    "fly there" leg is two calls: propose_viewpoints around the intended point
    (``target_hypothesis``) and then go_to_viewpoint on the returned viewpoint
    closest to it.
    """

    name = "mock"

    def __init__(self, mode: str, scenario: Scenario, driver: ScenarioDriver,
                 target_name: str = "target", approach_m: float | None = None,
                 verify_dist_m: float | None = None) -> None:
        super().__init__(mode="tools", tool_script=[])
        self.mode = mode
        self.scenario = scenario
        self.driver = driver
        self.target_name = target_name
        self.approach_m = float(scenario.approach_m if approach_m is None else approach_m)
        self.verify_dist_m = float(
            scenario.success_dist_m if verify_dist_m is None else verify_dist_m
        )
        self.registry: M2Registry | None = None
        self.observed = False
        self.queried = False
        self.returned = False
        self.relooked = False
        self.approached = False
        self.verified = False
        self.scanned = False
        self.searched = False
        self.pending_leg: dict | None = None
        self.last_pose: list[float] | None = None
        self.last_yaw_deg: float | None = None
        self.calls: list[dict] = []

    # -- model side -------------------------------------------------------
    def bind_registry(self, registry: Any) -> None:
        self.registry = registry

    def on_tool_result(self, entry: dict | None) -> None:
        if not entry:
            return
        pose = result_pose(entry)
        if pose is not None:
            self.last_pose, self.last_yaw_deg = pose
        tool = entry.get("tool")
        if tool == "observe" and entry.get("status") == STATUS_OK:
            self.observed = True
        if tool == "query_memory":
            self.queried = True
        if tool == "verify_goal":
            self.verified = True

    def _recall(self) -> dict | None:
        return self.registry.latest_recall() if self.registry is not None else None

    def _tool_props(self, name: str) -> dict:
        try:
            spec = self.registry.get(name)
        except Exception:
            return {}
        schema = getattr(spec, "input_schema", None) or {}
        props = schema.get("properties")
        return props if isinstance(props, dict) else {}

    def _fly_to(self, target: Iterable[float], reason: str) -> dict:
        """The two calls that get the drone to ``target``: propose, then fly.

        A world position is not a movement target any more (that is exactly what
        the intent split removed), so the leg starts with
        ``propose_viewpoints(target_hypothesis=<target>)``; the viewpoint it
        proposed closest to the intended point is then flown to with
        ``go_to_viewpoint``. ``count=3`` puts one candidate on the exact bearing
        to the hypothesis (the ``count=2`` arc would rotate both by 22.5 deg), and
        ``radius_m`` is the distance so the far-ring candidate lands on it.
        """
        target = [float(v) for v in np.asarray(target, dtype=np.float64).reshape(3)]
        pose, _ = self._pose(None)
        distance = float(np.hypot(target[0] - pose[0], target[1] - pose[1]))
        return {
            "tool": "propose_viewpoints",
            "args": {
                "target_hypothesis": [round(v, 3) for v in target],
                "radius_m": round(max(0.5, min(8.0, distance)), 3),
                "count": 3,
                "reason": f"propose an observation point {reason}",
            },
        }

    def _viewpoint_for(self, entry: dict | None, target: Iterable[float]) -> dict:
        """The ``go_to_viewpoint`` call for the viewpoint nearest ``target``."""
        wanted = np.asarray(list(target), dtype=np.float64).reshape(3)
        viewpoints = ((entry or {}).get("data") or {}).get("viewpoints") or []
        best_id, best_error = "", None
        for candidate in viewpoints:
            if not isinstance(candidate, dict) or not candidate.get("viewpoint_id"):
                continue
            position = candidate.get("position")
            if not isinstance(position, (list, tuple)) or len(position) < 3:
                continue
            error = float(np.linalg.norm(
                np.asarray(position[:3], dtype=np.float64) - wanted
            ))
            if best_error is None or error < best_error:
                best_id, best_error = str(candidate["viewpoint_id"]), error
        if not best_id:
            best_id = str(((entry or {}).get("data") or {}).get("viewpoint_id") or "")
        return {"tool": "go_to_viewpoint",
                "args": {"viewpoint_id": best_id,
                         "reason": f"fly to the proposed observation point {best_id}"}}

    def _return_to(self, observation_id: str, reason: str) -> dict:
        """The intended way back: ``return_to_observation`` restores the heading."""
        return {"tool": "return_to_observation",
                "args": {"observation_id": str(observation_id), "restore_yaw": True,
                         "reason": reason}}

    def _pose(self, context: dict | None) -> tuple[list[float], float]:
        """Odometry the runtime publishes (state block), then the last result."""
        context = context if isinstance(context, dict) else {}
        position = context.get("position_flu")
        yaw = context.get("yaw_rad")
        if isinstance(position, (list, tuple)) and len(position) >= 3 and yaw is not None:
            return [float(v) for v in position[:3]], float(yaw)
        if self.last_pose is not None:
            yaw_deg = self.last_yaw_deg if self.last_yaw_deg is not None else 0.0
            return list(self.last_pose), math.radians(yaw_deg)
        pose = self.driver.pose() if self.driver is not None else np.zeros(3)
        yaw = self.driver.yaw() if self.driver is not None else 0.0
        return [float(v) for v in pose[:3]], float(yaw)

    def _next_call(self, context: dict | None) -> dict:
        if self.pending_leg is not None:
            # the leg itself: the viewpoints proposed on the previous turn
            leg, self.pending_leg = self.pending_leg, None
            return self._viewpoint_for(self._last_result(), leg["target"])
        pose, yaw = self._pose(context)
        if not self.observed:
            return {"tool": "observe",
                    "args": {"mode": "forward", "reason": "look for the target first"}}
        if self.driver.loss is None and self.driver.trigger == "model":
            target = self.scenario.loss_pose(pose)
            return self._start_leg(target, "on to the next area")
        if self.mode in EVIDENCE_MODES:
            return self._next_memory_call(pose, yaw)
        return self._next_basic_call(pose, yaw)

    def _start_leg(self, target: Iterable[float], reason: str) -> dict:
        """Remember the intended position, then ask where to fly (two calls)."""
        self.pending_leg = {"target": [float(v) for v in np.asarray(
            target, dtype=np.float64).reshape(3)]}
        return self._fly_to(self.pending_leg["target"], reason)

    def _last_result(self) -> dict | None:
        results = getattr(self.registry, "results", None) if self.registry is not None else None
        return results[-1] if isinstance(results, list) and results else None

    def _next_memory_call(self, pose: list[float], yaw: float) -> dict:
        if not self.queried:
            return {"tool": "query_memory",
                    "args": {"query": self.target_name, "limit": 3,
                             "reason": "the target left the view: where did I see it?"}}
        recall = self._recall()
        if recall is None or not recall.get("pose"):
            return {"tool": "finish",
                    "args": {"reason": "no remembered observation to re-localize from",
                             "success": False}}
        recalled = [float(v) for v in recall["pose"]]
        obs_id = recall["observation_id"]
        if not self.returned:
            self.returned = True
            return self._return_to(
                obs_id,
                f"return to the viewpoint where I last saw the target ({obs_id})",
            )
        if not self.relooked:
            self.relooked = True
            args: dict[str, Any] = {"mode": "forward",
                                    "reason": f"re-localize the target seen in {obs_id}"}
            # the observation was captured at a heading; look along it again when
            # the tool supports view offsets (the return leg faced the way it flew)
            delta = None
            if recall.get("yaw_deg") is not None and self.last_yaw_deg is not None:
                delta = _wrap_deg(float(recall["yaw_deg"]) - float(self.last_yaw_deg))
            if delta is not None and abs(delta) > 5.0 and "yaw_deg" in self._tool_props("observe"):
                args["yaw_deg"] = round(delta, 1)
            return {"tool": "observe", "args": args}
        if not self.approached:
            self.approached = True
            heading = math.radians(
                recall["yaw_deg"] if recall.get("yaw_deg") is not None
                else math.degrees(yaw)
            )
            approach = np.asarray(recalled, dtype=np.float64) + self.approach_m * np.asarray(
                [math.cos(heading), math.sin(heading), 0.0]
            )
            return self._start_leg(approach, f"along the heading of {obs_id}")
        if not self.verified:
            self.verified = True
            return {"tool": "verify_goal",
                    "args": {"success_dist_m": self.verify_dist_m,
                             "reason": "confirm the re-acquired target"}}
        return {"tool": "finish",
                "args": {"reason": f"re-acquired the target from the remembered "
                                   f"observation {obs_id}"}}

    def _next_basic_call(self, pose: list[float], yaw: float) -> dict:
        if not self.scanned:
            self.scanned = True
            return {"tool": "observe",
                    "args": {"mode": "lookaround", "n_views": int(self.scenario.scan_views),
                             "reason": "search around the current position"}}
        if not self.searched:
            self.searched = True
            ahead = np.asarray(pose, dtype=np.float64) + 1.5 * np.asarray(
                [math.cos(yaw), math.sin(yaw), 0.0]
            )
            return self._start_leg(ahead, "to search ahead")
        if not self.verified:
            self.verified = True
            return {"tool": "verify_goal",
                    "args": {"success_dist_m": self.verify_dist_m,
                             "reason": "check whether the target is in reach"}}
        return {"tool": "finish",
                "args": {"reason": "the target was not re-acquired without memory evidence",
                         "success": False}}

    # -- the runtime calls these ------------------------------------------
    def decide_with_tools(self, frames, system_prompt, user_prompt, tools_text="",
                          context=None):
        call = self._next_call(context)
        self.calls.append(call)
        return json.dumps(call)

    def decide(self, frame_chw, system_prompt, user_prompt, context=None):
        call = self._next_call(context)
        self.calls.append(call)
        return json.dumps(call)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _is_tool(entry: dict, name: str) -> bool:
    return str(entry.get("tool") or "") == name


def _query_hits(entry: dict) -> list[dict]:
    data = entry.get("data") or {}
    for key in ("hits", "matches"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _query_hit_count(entry: dict) -> int:
    data = entry.get("data") or {}
    for key in ("n_hits", "count"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return len(_query_hits(entry))


def _frames_reported(entry: dict) -> set[str]:
    """The observation ids a retrieval says it carries a frame for.

    The hit list is the model-facing statement of what came back: an id whose
    hit reports no image is a hit without a picture (that is how a store whose
    episodic evidence layer is off answers), and nothing may be attached for it.
    """
    reported = set()
    for hit in _query_hits(entry):
        image = hit.get("image") or hit.get("image_path")
        obs_id = hit.get("observation_id") or hit.get("obs_id") or hit.get("id")
        if image and obs_id:
            reported.add(str(obs_id))
    return reported


def _as_xyz(value: Any) -> list[float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return [float(v) for v in value[:3]]
        except (TypeError, ValueError):
            return None
    return None


def _leg_target(entry: dict) -> list[float] | None:
    """World position a movement call went to.

    The intent tools report it as ``target_position`` in the task payload (the
    id a call names is resolved to the position *inside* the tool), so the
    report is where the leg's destination comes from.
    """
    data = entry.get("data") or {}
    return _as_xyz(data.get("target_position")) or _as_xyz(data.get("target_flu"))


def _recall_ids(entry: dict) -> list[str]:
    ids = []
    for hit in _query_hits(entry):
        obs_id = hit.get("observation_id") or hit.get("obs_id") or hit.get("id")
        if obs_id:
            ids.append(str(obs_id))
    for ref in entry.get("evidence") or []:
        if str(ref.get("kind")) == "observation" and ref.get("ref"):
            ids.append(str(ref["ref"]))
    seen: list[str] = []
    for obs_id in ids:
        if obs_id not in seen:
            seen.append(obs_id)
    return seen


def _recall_images(entry: dict) -> list[str]:
    images = []
    for hit in _query_hits(entry):
        value = hit.get("image") or hit.get("image_path")
        if value:
            images.append(str(value))
    for ref in entry.get("evidence") or []:
        if str(ref.get("kind")) == "image" and ref.get("ref"):
            images.append(str(ref["ref"]))
    seen: list[str] = []
    for image in images:
        if image not in seen:
            seen.append(image)
    return seen


def _archived_ids(registry: Any) -> list[str]:
    """The observation ids the archive holds at the end of the episode.

    Read through the registered memory (the ablation proxy delegates to the real
    archive, so this reports what was *stored*, not what was written) - the
    counter that proves the ablation removed the evidence instead of only
    hiding it.
    """
    ctx = getattr(registry, "ctx", None)
    memory = getattr(ctx, "memory", None)
    ids = getattr(memory, "ids", None)
    if not callable(ids):
        return []
    try:
        found = [str(obs_id) for obs_id in ids()]
    except Exception:  # noqa: BLE001 - metrics never fail an episode
        return []
    return sorted(found, key=lambda obs_id: (len(obs_id), obs_id))


def vlm_cost_report(counter: Any) -> dict:
    """Full model-call cost of one episode, loop and in-tool calls apart.

    The loop's own calls are also in the trace (``result_prompt`` /
    ``images_sent`` / ``model_reply``); the calls a tool makes internally are
    not, so this is where the accounting for them lives.
    """
    calls = list(getattr(counter, "model_calls", []) or [])
    tools = [call for call in calls if call.get("kind") == "tool"]
    by_tool: dict[str, int] = {}
    for call in tools:
        name = str(call.get("tool") or "?")
        by_tool[name] = by_tool.get(name, 0) + 1
    return {
        "recorded": bool(calls),
        "loop_calls": sum(1 for call in calls if call.get("kind") == "loop"),
        "tool_calls": len(tools),
        "total_calls": len(calls),
        "tool_calls_by_tool": by_tool,
        "prompt_chars": sum(int(call.get("prompt_chars") or 0) for call in calls),
        "tool_prompt_chars": sum(int(call.get("prompt_chars") or 0) for call in tools),
        "images": sum(int(call.get("images") or 0) for call in calls),
        "tool_images": sum(int(call.get("images") or 0) for call in tools),
        "reply_chars": sum(int(call.get("reply_chars") or 0) for call in calls),
        "tool_reply_chars": sum(int(call.get("reply_chars") or 0) for call in tools),
    }


def executed_pairs(entries: list[dict], states: list[dict]) -> list[tuple[dict, dict]]:
    """Pair every trace entry the registry executed with its pose state.

    Two kinds of trace line are not calls of the tool registry: a tool outside
    the episode's set is answered by the runtime itself (``reason=unknown_tool``)
    and ``finish`` ends the loop, so neither has a driver state. Everything else
    has exactly one, in the same order - matched by name so a skipped line cannot
    shift the association.
    """
    pairs: list[tuple[dict, dict]] = []
    cursor = 0
    for entry in entries:
        tool = str(entry.get("tool") or "")
        if tool == "finish" or str((entry.get("data") or {}).get("reason") or "") == "unknown_tool":
            continue
        while cursor < len(states) and str(states[cursor].get("tool") or "") != tool:
            cursor += 1
        if cursor >= len(states):
            continue
        pairs.append((entry, states[cursor]))
        cursor += 1
    return pairs


def capture_is_sighting(entry: dict, state: dict | None, scenario: Scenario) -> bool:
    """Did the frame this capture stored hold the target?

    ``observe`` reports the pose it rendered from and the headings it rendered,
    so the analytic check runs on the frame itself - the drone may have been
    relocated since (the ``capture`` loss trigger fires on this very call).
    ``ground_target`` reports the candidates it grounded in the frame, and a
    candidate for the instructed object is the same statement. A capture that
    says neither falls back to the driver's view check.
    """
    data = entry.get("data") or {}
    if str(entry.get("tool") or "") == "ground_target":
        return any(isinstance(cand, dict) for cand in (data.get("candidates") or []))
    pose = _as_xyz(data.get("pose") or data.get("position_flu"))
    yaws: list[float] = []
    headings = data.get("headings_deg")
    if isinstance(headings, (list, tuple)):
        yaws = [math.radians(float(head)) for head in headings
                if isinstance(head, (int, float))]
    if not yaws and data.get("yaw_deg") is not None:
        try:
            yaws = [math.radians(float(data["yaw_deg"]))]
        except (TypeError, ValueError):
            yaws = []
    if pose is not None and yaws:
        return any(target_visibility(pose, yaw, scenario)["visible"] for yaw in yaws)
    if state is not None and state.get("view_visible") is not None:
        return bool(state["view_visible"])
    return bool(state.get("visible")) if state is not None else False


def ablation_report(registry: Any, driver: ScenarioDriver,
                    entries: list[dict], scenario: Scenario | None = None) -> dict:
    """Did the ablation run, what did it remove, and was it the evidence?

    ``first_sighting_archived`` is the target's first capture of the episode -
    an ``observe`` whose view held the target, or the ``ground_target`` frame a
    candidate was grounded in - found in the archive: True in the ``memory``
    arm, and False by construction wherever the ablation is on.
    ``pre_loss_observation_ids`` are all the ids the tools handed out before the
    loss, so "the sighting was not archived" can be told apart from "there was
    no sighting". It counts *capture records*: an arm whose episodic layer is
    off (``memory_spatial_only``) still archives the captures - what it does not
    keep is the evidence written about them (``sightings`` in
    ``layer_status()`` stays 0), which is what ``memory_switches`` and each
    ``query_memory`` result's ``layers`` report.
    """
    scenario = scenario if scenario is not None else driver.scenario
    ablation = getattr(registry, "ablation", None)
    archived = _archived_ids(registry)
    loss = driver.loss or {}
    loss_call = int(loss.get("call_index") or 0)
    pre_loss_ids: list[str] = []
    sighting_ids: list[str] = []
    first_sighting = False
    for entry, state in executed_pairs(entries, driver.states):
        tool = str(entry.get("tool") or "")
        if tool not in ("observe", "ground_target"):
            continue
        if loss and int(state.get("call") or 0) > loss_call:
            continue  # found again after the loss: new evidence, not the history
        obs_id = str((entry.get("data") or {}).get("observation_id") or "")
        if not obs_id:
            continue
        pre_loss_ids.append(obs_id)
        if not capture_is_sighting(entry, state, scenario):
            continue
        sighting_ids.append(obs_id)
        first_sighting = first_sighting or obs_id in archived
    return {
        "enabled": bool(getattr(registry, "ablate_evidence", False)),
        "dropped_records": int(getattr(ablation, "dropped", 0)),
        "kept_records": int(getattr(ablation, "kept", 0)),
        "archived_observation_ids": archived,
        "pre_loss_observation_ids": pre_loss_ids,
        "pre_loss_sighting_ids": sighting_ids,
        "first_sighting_archived": bool(first_sighting),
    }


def altitude_metrics(poses: list, scenario: Scenario, world: Any,
                     envelope: FlightEnvelope | None = None) -> dict:
    """Flown-altitude profile of one episode (audit altitude_audit.md §5.3).

    The low-altitude variant asks two questions beyond "did it arrive": did the
    drone stay under the no-fly height (``ceiling_violations``) and did it go
    *around* the ground-level obstacle instead of through it
    (``obstacle_violations``, one per flown leg that crosses an occluder)?
    """
    points = [
        np.asarray(pose, dtype=np.float64).reshape(3)
        for pose in poses
        if pose is not None
    ]
    if len(points) < 2:
        points = [np.asarray(world.position_flu, dtype=np.float64).reshape(3)]
    heights = [float(point[2]) for point in points]
    violations = 0
    for before, after in zip(points, points[1:]):
        violations += sum(1 for box in scenario.occluders if box.blocks(before, after))

    ceiling = float(getattr(world, "z_max", float("inf")))
    if not math.isfinite(ceiling):
        ceiling = None
    report = {
        "floor_z": None if scenario.ground_z is None else round(float(scenario.ground_z), 3),
        "start_z": round(heights[0], 3),
        "min_z": round(min(heights), 3),
        "max_z": round(max(heights), 3),
        "span_m": round(max(heights) - min(heights), 3),
        "z_min": round(float(getattr(world, "z_min", float("nan"))), 3),
        "z_max": None if ceiling is None else round(ceiling, 3),
        "ceiling_violations": (
            0 if ceiling is None else sum(1 for z in heights if z > ceiling + 1.0e-6)
        ),
        "obstacle_violations": int(violations),
        "straight_route_blocked": bool(
            any(
                box.blocks(scenario.loss_pose(scenario.start_pos), scenario.target_pos)
                for box in scenario.occluders
            )
        ),
        "steps": len(points),
    }
    if scenario.ground_z is not None:
        report["start_above_floor_m"] = round(heights[0] - float(scenario.ground_z), 3)
        report["start_above_target_top_m"] = round(
            heights[0] - float(scenario.target_top_z), 3
        )
    if envelope is not None:
        report["cruise_window_z"] = [round(value, 3) for value in envelope.cruise_window_z]
        report["overflight_blocked"] = bool(envelope.overflight_blocked)
    return report


def _instance_of(point, scenario: Scenario) -> str | None:
    """Which of the two instances a position is at (``None``: no decoy scene).

    Within ``INSTANCE_TIE_M`` of both counts as ``tie``: two objects closer than
    that are one place, and calling the nearer one "picked" would invent a
    decision the drone did not make.
    """
    if scenario.decoy_pos is None:
        return None
    correct = np.asarray(scenario.target_pos, dtype=np.float64).reshape(3)
    decoy = np.asarray(scenario.decoy_pos, dtype=np.float64).reshape(3)
    point = np.asarray(point, dtype=np.float64).reshape(3)
    distance_correct = float(np.linalg.norm(point - correct))
    distance_decoy = float(np.linalg.norm(point - decoy))
    if abs(distance_correct - distance_decoy) <= INSTANCE_TIE_M:
        return "tie"
    return "correct" if distance_correct < distance_decoy else "decoy"


def discrimination_metrics(scenario: Scenario, entries: list[dict], final_pose,
                           summary: dict) -> dict | None:
    """Which instance the drone went to (the discrimination scene only).

    Every movement intent is attributed to the instance it flies to, and so is
    the final pose. ``claimed_wrong_instance`` is the claim the finish contract
    counts (``summary["success"]``, i.e. the declared outcome) made while the
    drone stands at the decoy - the error a false confirmation would be.
    """
    if scenario.decoy_pos is None:
        return None
    correct = np.asarray(scenario.target_pos, dtype=np.float64).reshape(3)
    decoy = np.asarray(scenario.decoy_pos, dtype=np.float64).reshape(3)
    final = np.asarray(final_pose, dtype=np.float64).reshape(3)
    legs: list[dict] = []
    for entry in entries:
        if str(entry.get("tool") or "") not in MOVE_TOOL_NAMES:
            continue
        target_pos = _leg_target(entry)
        if target_pos is None:
            continue
        legs.append({
            "tool": str(entry.get("tool")),
            "to": [round(float(v), 3) for v in np.asarray(target_pos).reshape(3)],
            "instance": _instance_of(target_pos, scenario),
        })
    counts = {"correct": 0, "decoy": 0, "tie": 0}
    for leg in legs:
        counts[str(leg["instance"])] = counts.get(str(leg["instance"]), 0) + 1
    final_instance = _instance_of(final, scenario)
    claimed = bool(summary.get("success"))
    return {
        "final_distance_correct_m": round(float(np.linalg.norm(final - correct)), 3),
        "final_distance_decoy_m": round(float(np.linalg.norm(final - decoy)), 3),
        "final_instance": final_instance,
        "first_leg_instance": legs[0]["instance"] if legs else None,
        "legs": legs,
        "legs_to_correct": counts["correct"],
        "legs_to_decoy": counts["decoy"],
        "legs_to_neither": counts["tie"],
        "claimed": claimed,
        "claimed_wrong_instance": bool(claimed and final_instance == "decoy"),
        "instance_separation_m": round(float(np.linalg.norm(correct - decoy)), 3),
    }


def compute_metrics(mode: str, summary: dict, entries: list[dict], world: Any,
                    scenario: Scenario, driver: ScenarioDriver,
                    registry: Any = None, hfov_deg: float | None = None,
                    relocate_tol_m: float = 0.5,
                    envelope: FlightEnvelope | None = None) -> dict:
    """M2 metrics of one trial: success, calls, memory hit, re-localization."""
    target = np.asarray(scenario.target_pos, dtype=np.float64).reshape(3)
    final_pose = np.asarray(world.position_flu, dtype=np.float64).reshape(3)
    final_distance = float(np.linalg.norm(final_pose - target))
    at_end = target_visibility(final_pose, float(getattr(world, "yaw_rad", 0.0)
                                                  or 0.0), scenario, hfov_deg)

    query_entries = [entry for entry in entries if _is_tool(entry, "query_memory")]
    hits = sum(_query_hit_count(entry) for entry in query_entries)
    recall_ids: list[str] = []
    recall_images: list[str] = []
    for entry in query_entries:
        for obs_id in _recall_ids(entry):
            if obs_id not in recall_ids:
                recall_ids.append(obs_id)
        for image in _recall_images(entry):
            if image not in recall_images:
                recall_images.append(image)
    stems = [Path(image).stem for image in recall_images]

    cited = False
    for index, entry in enumerate(entries):
        if not _is_tool(entry, "query_memory"):
            continue
        later = " ".join(str(item.get("model_reply") or "") for item in entries[index + 1:])
        if any(obs_id and obs_id in later for obs_id in recall_ids) or \
                any(stem and stem in later for stem in stems):
            cited = True
            break

    reposition_error_m = None
    relocalized = False
    if query_entries:
        first_index = entries.index(query_entries[0])
        recalled = getattr(registry, "latest_recall", None) if registry is not None else None
        recalled = recalled() if callable(recalled) else None
        recalled_pose = _as_xyz((recalled or {}).get("pose"))
        if recalled_pose is None:
            recalled_pose = _as_xyz((extract_recall(query_entries[-1], None) or {}).get("pose"))
        for entry in entries[first_index + 1:]:
            # the first leg after the retrieval: every intent tool reports the
            # position it is flying to as its ``target_position``
            if str(entry.get("tool") or "") not in MOVE_TOOL_NAMES:
                continue
            target_pos = _leg_target(entry)
            if target_pos is None:
                continue
            if recalled_pose is not None:
                reposition_error_m = round(
                    float(np.linalg.norm(np.asarray(target_pos) - np.asarray(recalled_pose))), 3
                )
                relocalized = reposition_error_m <= float(relocate_tol_m)
            break

    images_sent_after_query = 0
    for index, entry in enumerate(entries):
        if _is_tool(entry, "query_memory") and index + 1 < len(entries):
            images_sent_after_query = max(
                images_sent_after_query, int(entries[index + 1].get("images_sent") or 0)
            )

    # did an observe *after* the retrieval put the target back in a view?
    relook_visible = None
    seen_query = False
    for state in driver.states:
        if state.get("tool") == "query_memory":
            seen_query = True
            continue
        if seen_query and state.get("tool") == "observe" and state.get("view_visible") is not None:
            relook_visible = bool(relook_visible or state["view_visible"])
    if relook_visible is None:
        relook_visible = False

    verify_entries = [entry for entry in entries if _is_tool(entry, "verify_goal")]
    verified = None
    unmet: list[str] = []
    if verify_entries:
        data = verify_entries[-1].get("data") or {}
        verified = data.get("verified")
        unmet = [str(item.get("check")) for item in (data.get("unmet") or [])
                 if isinstance(item, dict)]

    loss = driver.loss or {}
    preconditions = {
        "target_visible_at_start": target_visibility(
            scenario.start_pos, math.radians(scenario.start_yaw_deg), scenario, hfov_deg
        )["visible"],
        "loss_occurred": driver.loss is not None,
        "loss_mechanism": loss.get("mechanism"),
        "target_visible_after_loss": loss.get("visible_after"),
        "scan_from_loss_pose_visible": loss.get("scan_visible_after"),
        "loss_pose_error_m": loss.get("pose_error_m"),
    }

    tool_is_available = {
        "query_memory": "query_memory" in (summary.get("toolset_detail") or {}).get(
            "registered", []
        )
    }

    return {
        "mode": mode,
        "toolset": summary.get("toolset"),
        "toolset_registered": (summary.get("toolset_detail") or {}).get("registered"),
        "instruction": summary.get("instruction"),
        "success": bool(summary.get("success")),
        "finish_reason": summary.get("finish_reason"),
        "final_distance_m": round(final_distance, 3),
        "success_dist_m": summary.get("success_dist_m"),
        "turns": summary.get("turns"),
        "tool_calls": summary.get("tool_calls"),
        "vlm_calls": summary.get("vlm_calls"),
        "tool_budget": summary.get("tool_budget"),
        "target_visible_at_end": bool(at_end["visible"]),
        "altitude": altitude_metrics(
            [state.get("pose") for state in driver.states] + [final_pose.tolist()],
            scenario,
            world,
            envelope,
        ),
        "calls": {
            "tools": summary.get("tools") or {},
            "sequence": [str(entry.get("tool") or "(no tool call)") for entry in entries],
        },
        "memory": {
            "tool_available": tool_is_available["query_memory"],
            "query_calls": len(query_entries),
            "queries_with_hits": sum(1 for entry in query_entries
                                     if _query_hit_count(entry) > 0),
            "hits": hits,
            "recall_observation_ids": recall_ids,
            "recall_images": recall_images,
            "evidence_cited_later": bool(cited),
            "reposition_error_m": reposition_error_m,
            "relocalized": bool(relocalized),
            "target_visible_on_relook": bool(relook_visible),
            "frames_published": len(getattr(registry, "published_frames", []) or []),
            "images_sent_after_query": images_sent_after_query,
        },
        "evidence_ablation": ablation_report(registry, driver, entries, scenario),
        "vlm_cost": vlm_cost_report(getattr(registry, "vlm_counter", None)),
        "verify_goal": {
            "calls": len(verify_entries),
            "verified": verified,
            "unmet_checks": unmet,
        },
        "loss": loss,
        "discrimination": discrimination_metrics(scenario, entries, final_pose, summary),
        "visibility_trace": driver.states,
        "scenario_preconditions": preconditions,
    }


def aggregate(trials: list[dict]) -> dict:
    """Per-mode aggregate over trials (same budget for every mode)."""
    count = max(1, len(trials))

    def mean(values: list[float]) -> float:
        return round(float(sum(values)) / count, 3)

    def numbers(key: str) -> list[float]:
        return [float(t.get(key) or 0.0) for t in trials]

    def nested(section: str, key: str) -> list[float]:
        return [float((t.get(section) or {}).get(key) or 0.0) for t in trials]

    first = trials[0] if trials else {}
    return {
        "trials": len(trials),
        "success_rate": round(sum(1 for t in trials if t.get("success")) / count, 3),
        "successes": sum(1 for t in trials if t.get("success")),
        "tool_calls_mean": mean(numbers("tool_calls")),
        "vlm_calls_mean": mean(numbers("vlm_calls")),
        "turns_mean": mean(numbers("turns")),
        "final_distance_m_mean": mean(numbers("final_distance_m")),
        "target_visible_at_end_rate": round(
            sum(1 for t in trials if t.get("target_visible_at_end")) / count, 3
        ),
        "memory_evidence_hit": any(
            (t.get("memory") or {}).get("hits", 0) > 0 for t in trials
        ),
        "memory_hits_mean": mean(nested("memory", "hits")),
        "memory_evidence_cited_rate": round(
            sum(1 for t in trials if (t.get("memory") or {}).get("evidence_cited_later")) / count,
            3,
        ),
        "relocalized_rate": round(
            sum(1 for t in trials if (t.get("memory") or {}).get("relocalized")) / count, 3
        ),
        "queries_with_hits_mean": mean(nested("memory", "queries_with_hits")),
        "target_visible_on_relook_rate": round(
            sum(1 for t in trials
                if (t.get("memory") or {}).get("target_visible_on_relook")) / count, 3
        ),
        "vlm_loop_calls_mean": mean(nested("vlm_cost", "loop_calls")),
        "vlm_tool_calls_mean": mean(nested("vlm_cost", "tool_calls")),
        "vlm_total_calls_mean": mean(nested("vlm_cost", "total_calls")),
        "vlm_tool_prompt_chars_mean": mean(nested("vlm_cost", "tool_prompt_chars")),
        "vlm_tool_images_mean": mean(nested("vlm_cost", "tool_images")),
        "verify_goal_verified_rate": round(
            sum(1 for t in trials if (t.get("verify_goal") or {}).get("verified") is True) / count,
            3,
        ),
        "ablation": {
            "enabled": bool((first.get("evidence_ablation") or {}).get("enabled")),
            "dropped_records_mean": mean(nested("evidence_ablation", "dropped_records")),
            "kept_records_mean": mean(nested("evidence_ablation", "kept_records")),
            "first_sighting_archived_rate": round(
                sum(1 for t in trials
                    if (t.get("evidence_ablation") or {}).get("first_sighting_archived")) / count,
                3,
            ),
            "evidence_recallable_rate": round(
                sum(1 for t in trials if (t.get("memory") or {}).get("hits", 0) > 0) / count, 3
            ),
        },
        "discrimination": _aggregate_discrimination(trials),
        "switches": {
            "requested": (first.get("memory_switches") or {}).get("requested"),
            "applied": (first.get("memory_switches") or {}).get("applied"),
            "recall_images": first.get("recall_images"),
        },
        "mode": first.get("mode"),
        "toolset": first.get("toolset"),
        "toolset_registered": first.get("toolset_registered"),
        "tool_budget": first.get("tool_budget"),
    }


def _aggregate_discrimination(trials: list[dict]) -> dict | None:
    """Per-arm rates of the discrimination metrics (``None``: no decoy scene)."""
    reports = [t["discrimination"] for t in trials if t.get("discrimination")]
    if not reports:
        return None
    count = len(reports)

    def rate(key: str, value: str) -> float:
        return round(sum(1 for r in reports if r.get(key) == value) / count, 3)

    def mean(key: str) -> float:
        return round(sum(float(r.get(key) or 0) for r in reports) / count, 3)

    return {
        "trials": count,
        "instance_separation_m": reports[0].get("instance_separation_m"),
        "final_instance_rate": {
            name: rate("final_instance", name) for name in ("correct", "decoy", "tie")
        },
        "first_leg_instance_rate": {
            name: rate("first_leg_instance", name) for name in ("correct", "decoy", "tie")
        },
        "legs_to_correct_mean": mean("legs_to_correct"),
        "legs_to_decoy_mean": mean("legs_to_decoy"),
        "claimed_rate": round(sum(1 for r in reports if r.get("claimed")) / count, 3),
        "claimed_wrong_instance_rate": round(
            sum(1 for r in reports if r.get("claimed_wrong_instance")) / count, 3
        ),
    }


def discrimination_verdict(aggregates: dict[str, dict]) -> dict | None:
    """What each memory layer is worth in the discrimination scene.

    The paired arms differ in one layer each: ``memory`` vs
    ``memory_text_only`` (the recalled frame in the next turn's context) and
    ``memory`` vs ``memory_spatial_only`` (the episodic evidence layer itself).
    The verdict is read off the rate of ending at the *instructed* instance;
    ``no_difference_at_this_budget`` is reported as it is, never dressed up as
    support.
    """
    if not any((agg.get("discrimination") or {}) for agg in aggregates.values()):
        return None

    def correct_rate(mode: str) -> float | None:
        report = (aggregates.get(mode) or {}).get("discrimination") or {}
        return (report.get("final_instance_rate") or {}).get("correct")

    def pair(arm: str, control: str, layer: str) -> dict | None:
        if correct_rate(arm) is None or correct_rate(control) is None:
            return None
        with_layer, without = correct_rate(arm), correct_rate(control)
        if with_layer > without:
            verdict = f"{layer}_required"
        elif with_layer == without:
            verdict = "no_difference_at_this_budget"
        else:
            verdict = f"{layer}_harmful"
        return {
            "arm": arm, "control": control, "layer": layer,
            "correct_instance_rate": {arm: with_layer, control: without},
            "verdict": verdict,
        }

    out = {
        "visual_evidence": pair("memory", "memory_text_only", "visual_evidence"),
        "episodic_layer": pair("memory", "memory_spatial_only", "episodic_layer"),
    }
    return {key: value for key, value in out.items() if value is not None} or None


def ablation_verdict(aggregates: dict[str, dict]) -> str | None:
    """Did removing the evidence weaken the memory arm? (the M2 claim)

    ``evidence_required``: the arm that must retrieve the history does better
    with it. ``evidence_not_needed``: the two arms are equally good, so what
    made the memory arm succeed was not the retrieved evidence. ``evidence_harmful``
    is the inversion (reported, not interpreted away).
    """
    if "memory" not in aggregates or ABLATED_MODE not in aggregates:
        return None
    with_evidence = aggregates["memory"]["success_rate"]
    without = aggregates[ABLATED_MODE]["success_rate"]
    if with_evidence > without:
        return "evidence_required"
    if with_evidence == without:
        return "evidence_not_needed"
    return "evidence_harmful"


def build_comparison(aggregates: dict[str, dict], args: argparse.Namespace,
                     scenario: Scenario) -> dict:
    """Side-by-side metrics of the tool sets, all under the same budget."""
    modes = list(aggregates)

    def row(key: str) -> dict:
        return {mode: aggregates[mode].get(key) for mode in modes}

    comparison = {
        "success_rate": row("success_rate"),
        "tool_calls_mean": row("tool_calls_mean"),
        "vlm_calls_mean": row("vlm_calls_mean"),
        "vlm_total_calls_mean": row("vlm_total_calls_mean"),
        "vlm_tool_calls_mean": row("vlm_tool_calls_mean"),
        "turns_mean": row("turns_mean"),
        "final_distance_m_mean": row("final_distance_m_mean"),
        "memory_evidence_hit": row("memory_evidence_hit"),
        "memory_hits_mean": row("memory_hits_mean"),
        "memory_evidence_cited_rate": row("memory_evidence_cited_rate"),
        "relocalized_via_recalled_pose": row("relocalized_rate"),
        "target_visible_on_relook_rate": row("target_visible_on_relook_rate"),
        "verify_goal_verified_rate": row("verify_goal_verified_rate"),
        "first_sighting_archived_rate": {
            mode: aggregates[mode].get("ablation", {}).get("first_sighting_archived_rate")
            for mode in modes
        },
        "instance_final_rate": {
            mode: ((aggregates[mode].get("discrimination") or {})
                   .get("final_instance_rate"))
            for mode in modes
        },
        "instance_claimed_wrong_rate": {
            mode: ((aggregates[mode].get("discrimination") or {})
                   .get("claimed_wrong_instance_rate"))
            for mode in modes
        },
        "instance_first_leg_rate": {
            mode: ((aggregates[mode].get("discrimination") or {})
                   .get("first_leg_instance_rate"))
            for mode in modes
        },
        "switches": {
            "requested": {mode: (aggregates[mode].get("switches") or {}).get("requested")
                          for mode in modes},
            "applied": {mode: (aggregates[mode].get("switches") or {}).get("applied")
                        for mode in modes},
            "recall_images": {mode: (aggregates[mode].get("switches") or {}).get("recall_images")
                              for mode in modes},
        },
    }
    verdict = "inconclusive"
    if "memory" in aggregates and "basic" in aggregates:
        memory, basic = aggregates["memory"], aggregates["basic"]
        if memory["success_rate"] > basic["success_rate"]:
            verdict = "memory_toolset_superior"
        elif memory["success_rate"] == basic["success_rate"]:
            verdict = "no_difference_at_this_budget"
        else:
            verdict = "baseline_superior"
    ablated = ablation_verdict(aggregates)
    return {
        "experiment": "M2 target re-acquisition (goal.md section 4 / milestone M2)",
        "backend": args.backend,
        "budget": {
            "tool_budget": int(args.tool_budget),
            "max_turns": int(args.max_turns),
            "trials": int(args.trials),
        },
        "instruction": args.instruction,
        "scenario": scenario.to_dict(),
        "modes": aggregates,
        "comparison": comparison,
        "verdict": verdict,
        "ablation_verdict": ablated,
        "evidence_ablation": {
            "mode": ABLATED_MODE,
            "enabled": any(
                aggregates[mode].get("ablation", {}).get("enabled") for mode in modes
            ),
            "rule": (
                "an observation written before the loss of view (the target still "
                "in view) is accepted and dropped, so query_memory has no history "
                "to return; the tool set, the prompt and every other result are "
                "the memory arm's"
            ),
            "verdict": ablated,
        },
        "discrimination_verdict": discrimination_verdict(aggregates),
    }


def print_comparison(comparison: dict) -> None:
    aggregates = comparison["modes"]
    rows = [
        ("mode", lambda agg: agg.get("toolset")),
        ("success", lambda agg: f"{agg['successes']}/{agg['trials']} ({agg['success_rate']:.2f})"),
        ("tool_calls", lambda agg: f"{agg['tool_calls_mean']:.1f}"),
        ("vlm_calls", lambda agg: f"{agg['vlm_calls_mean']:.1f}"),
        ("vlm_total", lambda agg: f"{agg['vlm_total_calls_mean']:.1f}"),
        ("vlm_in_tool", lambda agg: f"{agg['vlm_tool_calls_mean']:.1f}"),
        ("mem_hits", lambda agg: f"{agg['memory_hits_mean']:.1f}"),
        ("cited", lambda agg: f"{agg['memory_evidence_cited_rate']:.2f}"),
        ("relocalized", lambda agg: f"{agg['relocalized_rate']:.2f}"),
        ("final_dist", lambda agg: f"{agg['final_distance_m_mean']:.2f} m"),
        ("verified", lambda agg: f"{agg['verify_goal_verified_rate']:.2f}"),
    ]
    switch_row = (comparison.get("comparison") or {}).get("switches") or {}
    if switch_row.get("requested"):
        rows.append(
            ("layers", lambda agg: "/".join(
                str((switch_row.get("requested", {}).get(agg.get("mode"), {}) or {})
                    .get(name, "?"))[:3]
                for name in ("spatial_memory", "episodic_memory")
            ) + ":img" + str(switch_row.get("recall_images", {}).get(agg.get("mode"), "?"))[:3])
        )
    discrimination = (comparison.get("comparison") or {}).get("instance_final_rate") or {}
    if any(value for value in discrimination.values()):
        rows.extend([
            ("inst_correct", lambda agg: _rate_cell(discrimination.get(agg.get("mode")), "correct")),
            ("inst_decoy", lambda agg: _rate_cell(discrimination.get(agg.get("mode")), "decoy")),
            ("wrong_claim", lambda agg: str(
                ((comparison.get("comparison") or {})
                 .get("instance_claimed_wrong_rate", {}).get(agg.get("mode")))
            )),
        ])
    header = f"{'':>12}" + "".join(f"{mode:>18}" for mode in aggregates)
    print("[m2] " + header)
    for label, getter in rows:
        line = "".join(f"{getter(aggregates[mode])!s:>18}" for mode in aggregates)
        print(f"[m2] {label:>12}{line}")
    print(f"[m2] verdict: {comparison['verdict']} "
          f"(budget: {comparison['budget']})", flush=True)
    if comparison.get("ablation_verdict"):
        print(f"[m2] evidence ablation: {comparison['ablation_verdict']} "
              f"(memory vs {ABLATED_MODE})", flush=True)
    for layer, record in (comparison.get("discrimination_verdict") or {}).items():
        print(f"[m2] {layer}: {record['verdict']} "
              f"({record['arm']} vs {record['control']}, "
              f"correct-instance rate {record['correct_instance_rate']})", flush=True)


def _rate_cell(rates: dict | None, name: str) -> str:
    """One cell of an instance-rate row ("-" when the arm has no decoy scene)."""
    if not isinstance(rates, dict):
        return "-"
    value = rates.get(name)
    return "-" if value is None else f"{float(value):.2f}"


# ---------------------------------------------------------------------------
# episode orchestration
# ---------------------------------------------------------------------------


def resolve_loss_trigger(args: argparse.Namespace) -> str:
    """``auto``: the scripted policy flies on (mock), the environment moves a
    real model's drone (a model is not assumed to fly away on cue)."""
    trigger = str(getattr(args, "loss_trigger", "auto"))
    if trigger == "auto":
        return "model" if str(getattr(args, "backend", "mock")).startswith("mock") else "evidence"
    return trigger


def run_trial(mode: str, scenario: Scenario, args: argparse.Namespace, out_dir: Path,
              make_vlm: Callable[[str, ScenarioDriver], Any],
              episode_id: str, ply_path: Path | str) -> dict:
    """One M2 episode in one tool set; writes trace/summary/metrics artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    world = build_world(args, scenario, ply_path)
    trigger = resolve_loss_trigger(args)
    driver = ScenarioDriver(world, scenario, trigger=trigger,
                            after_observations=int(args.loss_after_observations))
    holder: dict = {}
    vlm = make_vlm(mode, driver)
    # every model call of the episode goes through the counter: the loop's own
    # calls and the ones a tool makes internally (ground_target grounds with the
    # backend it is handed)
    counter = CountingVLM(vlm) if vlm is not None else None
    backend = counter if counter is not None else vlm
    mode_args = copy.copy(args)
    mode_args.toolset = MODE_TOOLSET.get(mode, "memory")
    mode_args.out = str(out_dir)
    for switch, value in MODE_SWITCH_OVERRIDES.get(mode, {}).items():
        # the arm's own switch is pinned; every other switch of the arm is the
        # caller's, so ``--memory-spatial off`` still reaches the memory arm
        setattr(mode_args, switch, value)
    ablate = mode == ABLATED_MODE
    publish_recall_images = RUNNER.switch_enabled(
        getattr(mode_args, "recall_images", DEFAULT_SWITCH)
    )

    def registry_factory(modules, vlm_backend):
        # the standard catalogue for this tool set, then the M2 hooks on top
        base, registry_backend, source = RUNNER.build_registry(
            modules, vlm=vlm_backend, toolset=mode_args.toolset
        )
        registry = M2Registry(driver=driver, modules=modules, ablate_evidence=ablate,
                              vlm_counter=counter,
                              publish_recall_images=publish_recall_images)
        for name in base.names():
            registry.register(base.get(name))
        callback = getattr(vlm_backend, "on_tool_result", None)
        if callable(callback):
            registry.on_result.append(callback)
        binder = getattr(vlm_backend, "bind_registry", None)
        if callable(binder):
            binder(registry)
        holder["registry"] = registry
        return registry, registry_backend, dict(source)

    ctx_extra = {
        # same gate the runner judges with: success_dist + target radius
        "done_dist": float(scenario.success_dist_m) + float(scenario.target_radius),
        "success_dist_m": float(scenario.success_dist_m),
        "loss_mode": trigger,
    }
    if backend is not None:
        ctx_extra["perception.vlm"] = backend  # ground_target stays usable in the loop
    print(f"[m2] {mode}: toolset={mode_args.toolset} loss-trigger={trigger} "
          f"evidence-ablation={'on' if ablate else 'off'} "
          f"spatial_memory={getattr(mode_args, 'memory_spatial', DEFAULT_SWITCH)} "
          f"episodic_memory={getattr(mode_args, 'memory_episodic', DEFAULT_SWITCH)} "
          f"recall-images={'on' if publish_recall_images else 'off'} "
          f"scene={args.ply or 'synthetic'}", flush=True)
    started = time.time()
    summary = RUNNER.run_episode(
        world, backend, mode_args, out_dir, episode_id=episode_id,
        registry_factory=registry_factory, ctx_extra=ctx_extra,
    )
    entries = RUNNER._read_trace(out_dir / "tool_trace.jsonl")
    metrics = compute_metrics(
        mode, summary, entries, world, scenario, driver,
        registry=holder.get("registry"), hfov_deg=scenario.hfov_deg,
        relocate_tol_m=float(args.relocate_tol_m),
        envelope=flight_envelope(args, scenario),
    )
    metrics["episode_id"] = episode_id
    metrics["wall_time_s"] = round(time.time() - started, 3)
    metrics["recall_images"] = "on" if publish_recall_images else "off"
    metrics["memory_switches"] = {
        "requested": {
            "spatial_memory": getattr(mode_args, "memory_spatial", DEFAULT_SWITCH),
            "episodic_memory": getattr(mode_args, "memory_episodic", DEFAULT_SWITCH),
        },
        "applied": (summary.get("switches") or {}).get("applied"),
    }
    metrics["model_calls"] = list(getattr(vlm, "calls", []) or [])
    with open(out_dir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2, default=str)
    close = getattr(world, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
    return metrics


def low_altitude_geometry(args: argparse.Namespace) -> tuple[
    tuple[float, float, float], tuple[float, float, float], Box
]:
    """(target, start, obstacle) of the low-altitude variant, raw frame.

    Ground target, ground-level obstacle (audit altitude_audit.md §5.2.8): the
    target sits on the floor and a 0.8 m box between the loss pose and the
    target blocks the direct way back, so the re-acquisition leg has to go
    around it. The start follows the new start band - level with the target's
    top - inside the cruise window. The obstacle sits clear of the start->target
    sight line (the episode still has to open with the target in view) but in
    the loss-pose->target line, which is what makes the return leg a detour.
    """
    target = (
        2.0,
        0.5,
        SYNTHETIC_FLOOR_Z + LOW_ALTITUDE_TARGET_ABOVE_FLOOR_M,
    )
    top_z = target[2] + float(args.target_radius)
    envelope = FlightEnvelope(ground_z=SYNTHETIC_FLOOR_Z)
    obstacle = Box(
        center=(1.0, -1.3, SYNTHETIC_FLOOR_Z + 0.5 * LOW_ALTITUDE_OBSTACLE_HEIGHT_M),
        size=(0.4, 1.4, LOW_ALTITUDE_OBSTACLE_HEIGHT_M),
    )
    start = (0.0, 0.0, envelope.cruise_z(top_z + 0.35))
    return target, start, obstacle


def _visual_required_mirror(args: argparse.Namespace) -> float:
    """-1 when the relation scene is mirrored (the landmark on the other side)."""
    return -1.0 if str(getattr(args, "relation_side", "left")) == "right" else 1.0


def visual_required_wall(args: argparse.Namespace) -> Box:
    """The relation scene's loss occluder, mirrored with the layout.

    The wall only hides the layout when it sits on the same side as the
    instances: the mirrored scene needs the mirrored wall, otherwise the loss
    pose sees the target (which the design checks then refuse). It is wider than
    :data:`DEFAULT_WALL` (x from 0.25 to 3.25 m) because every layout the variant
    accepts has to be hidden from the loss pose, and it still clears the loss
    leg, which runs along x = 0.
    """
    return Box(center=(1.75, -1.5 * _visual_required_mirror(args), 0.0),
               size=(3.0, 0.12, 3.2))


def visual_required_loss_bearing(args: argparse.Namespace) -> float:
    """The loss leg of the relation scene, mirrored with the layout.

    ``--loss-bearing-deg`` is expressed for the authored (left) layout; the
    mirrored scene flies its loss leg to the mirrored bearing so the two scenes
    are reflections of each other instead of one scene and one broken one.
    """
    return _visual_required_mirror(args) * float(args.loss_bearing_deg)


def visual_required_geometry(args: argparse.Namespace) -> tuple[
    tuple[float, float, float], tuple[float, float, float],
    tuple[float, float, float], tuple[float, float, float], tuple[Box, ...]
]:
    """(target, decoy, landmark, start, occluders) of the relation scene.

    Authored *around the start pose*, which is what makes the design
    orientation-independent: the correct instance sits ``--instance-bearing-deg``
    off the nose of the start pose, the decoy - the same object, pixel for pixel,
    and nearer - almost straight ahead (**the instance a policy that cannot read
    the relation drifts to**), and the bright landmark ``--lamp-offset-m`` beside
    the correct instance, perpendicular to the start->instance sight line on
    ``--relation-side``. Being that far to the side, the landmark falls outside
    the *forward* view - the frame a ``ground_target`` capture grounds in - while
    staying inside a 45 deg scan view of the same pose, so the pair *is*
    capturable by turning. The wall sits between the loss pose and all three, so
    from there no heading sees any of them.

    The layout is authored in the world frame with the start heading at 0, so
    ``--start-yaw-deg`` turns the *camera* and gives a genuinely different
    framing of the same scene - within the band
    :func:`visual_required_checks` accepts (the landmark has to stay outside the
    forward view). The checks that make those claims measurable live there, and
    :func:`build_scenario` runs them before an episode starts.
    """
    # ``right`` is the mirrored layout (the landmark on the other side of the
    # sight line), which is what makes a second, independent scene out of the
    # same design: the whole arrangement is reflected, not rotated
    mirror = _visual_required_mirror(args)

    def polar(bearing_deg: float, distance: float) -> tuple[float, float, float]:
        angle = mirror * math.radians(float(bearing_deg))
        return (distance * math.cos(angle), distance * math.sin(angle), 0.0)

    target = polar(args.instance_bearing_deg, args.instance_range_m)
    decoy = polar(args.decoy_bearing_deg, args.decoy_range_m)
    start = (0.0, 0.0, float(DEFAULT_START_POS[2]))
    occluders = tuple(
        Box((float(box[0]), float(box[1]), float(box[2])),
            (float(box[3]), float(box[4]), float(box[5])))
        for box in (args.occluder or [])
    )
    if RUNNER.switch_enabled(getattr(args, "relation_visible_at_start", "off")):
        # the control layout: same scene, same instance pair, but the landmark
        # stands beside the correct instance *inside* the forward view - the
        # relation is then resolvable where the drone grounds, which is the
        # property under test
        inside = polar(float(args.instance_bearing_deg) - VISUAL_LAMP_INVIEW_OFFSET_DEG,
                       args.instance_range_m)
        landmark = (inside[0], inside[1], VISUAL_LANDMARK_Z)
        return target, decoy, landmark, start, occluders
    offset = float(args.lamp_offset_m)
    normal = mirror * (math.radians(float(args.instance_bearing_deg)) + 0.5 * math.pi)
    landmark = (
        target[0] + offset * math.cos(normal),
        target[1] + offset * math.sin(normal),
        VISUAL_LANDMARK_Z,
    )
    return target, decoy, landmark, start, occluders


def visual_required_checks(scenario: Scenario, args: argparse.Namespace) -> list[str]:
    """Design failures of the ``visual_required`` scene (``[]``: it holds).

    Every claim the relation scene rests on is re-derived from the authored
    geometry here, so a run cannot report on a scene that does not have the
    property it is supposed to measure:

    - both instances inside the start's *forward* view (the drone can see the
      choice it has to make) and the landmark outside it by
      :data:`VISUAL_RELATION_MARGIN_DEG` (the grounding view cannot resolve
      "next to the lamp");
    - the landmark inside the scan window of the same pose (turning is enough to
      capture the pair, which is what fills the archive);
    - the landmark closer to the correct instance than to the decoy by
      :data:`VISUAL_RELATION_GAP_M` (the relation names exactly one instance);
    - from the loss pose, no heading sees the target, the decoy or the landmark
      (the decision has to come from what was recorded).
    """
    failures: list[str] = []
    hfov = float(scenario.hfov_deg)
    yaw = math.radians(float(scenario.start_yaw_deg))

    def bearing_of(point) -> float:
        delta = np.asarray(point, dtype=np.float64) - np.asarray(
            scenario.start_pos, dtype=np.float64)
        return math.degrees(math.atan2(float(delta[1]), float(delta[0])))

    def offset_from_start(point) -> float:
        return _wrap_deg(bearing_of(point) - math.degrees(yaw))

    def scan_window(offset_deg: float) -> float:
        """Smallest |offset| an in-place scan view can have (0 if none covers it)."""
        n = max(4, int(scenario.scan_views))
        return min(abs(_wrap_deg(offset_deg - 360.0 * k / n)) for k in range(n))

    for name, point in (("target", scenario.target_pos), ("decoy", scenario.decoy_pos)):
        if point is None:
            failures.append(f"the relation scene needs a decoy (the {name} slot is empty)")
            continue
        state = target_visibility(scenario.start_pos, yaw, replace(scenario, target_pos=point))
        if not state["visible"]:
            failures.append(f"{name} is not visible in the start forward view ({state})")
        elif abs(offset_from_start(point)) > hfov / 2.0 - VISUAL_INSTANCE_MARGIN_DEG:
            failures.append(
                f"{name} is only {round(hfov / 2.0 - abs(offset_from_start(point)), 2)} deg "
                f"inside the forward view (design margin "
                f"{VISUAL_INSTANCE_MARGIN_DEG} deg)"
            )
    if scenario.landmark_pos is None:
        failures.append("--landmark-pos is the relation scene's whole point")
        return failures
    landmark_state = target_visibility(
        scenario.start_pos, yaw, replace(scenario, target_pos=scenario.landmark_pos))
    offset = offset_from_start(scenario.landmark_pos)
    in_view = RUNNER.switch_enabled(getattr(args, "relation_visible_at_start", "off"))
    if in_view:
        # the control layout asks for the opposite property: the landmark is in
        # the grounding view, next to the correct instance and far from the decoy
        if not landmark_state["visible"]:
            failures.append(
                f"--relation-visible-at-start on, but the landmark is not in the start "
                f"forward view ({landmark_state})"
            )
    elif landmark_state["visible"]:
        failures.append(
            f"the landmark is visible in the start forward view (offset {round(offset, 2)} "
            f"deg, hfov {hfov}): the relation could be grounded where the drone grounds"
        )
    elif abs(offset) < hfov / 2.0 + VISUAL_RELATION_MARGIN_DEG:
        failures.append(
            f"the landmark is only {round(abs(offset) - hfov / 2.0, 2)} deg outside the "
            f"forward view (design margin {VISUAL_RELATION_MARGIN_DEG} deg)"
        )
    if not in_view and scan_window(offset) > hfov / 2.0:
        failures.append(
            f"no scan view of the start pose covers the landmark (offset {round(offset, 2)} "
            f"deg, {scenario.scan_views} views of {hfov} deg): the pair is not capturable"
        )
    to_target = float(np.linalg.norm(np.asarray(scenario.landmark_pos)
                                     - np.asarray(scenario.target_pos)))
    to_decoy = float(np.linalg.norm(np.asarray(scenario.landmark_pos)
                                    - np.asarray(scenario.decoy_pos)))
    if to_decoy - to_target < VISUAL_RELATION_GAP_M:
        failures.append(
            f"the landmark is {round(to_target, 2)} m from the target and {round(to_decoy, 2)} m "
            f"from the decoy (design gap {VISUAL_RELATION_GAP_M} m): the relation is ambiguous"
        )
    loss_pose = scenario.loss_pose(scenario.start_pos)
    for name, point in (("target", scenario.target_pos), ("decoy", scenario.decoy_pos),
                        ("landmark", scenario.landmark_pos)):
        seen = scan_visibility(loss_pose, replace(scenario, target_pos=point))
        if seen.get("visible"):
            failures.append(
                f"the loss pose still sees the {name} ({seen}): the decision would not need "
                "the archive"
            )
    return failures


def build_scenario(args: argparse.Namespace) -> Scenario:
    if str(getattr(args, "scene_variant", "default")) not in SCENE_VARIANTS:
        raise SystemExit(
            f"unknown --scene-variant {args.scene_variant}; choose from {list(SCENE_VARIANTS)}"
        )
    variant = str(getattr(args, "scene_variant", "default"))
    occluders = tuple(
        Box((float(box[0]), float(box[1]), float(box[2])),
            (float(box[3]), float(box[4]), float(box[5])))
        for box in (args.occluder or [])
    )
    target_pos = tuple(float(v) for v in args.target_pos)
    start_pos = tuple(float(v) for v in args.start_pos)
    loss_bearing_deg = float(args.loss_bearing_deg)
    ground_z = None if args.ply else SYNTHETIC_FLOOR_Z
    if variant == "low_altitude":
        if args.ply:
            raise SystemExit(
                "--scene-variant low_altitude authors its own synthetic scene; "
                "pass --ply without the variant, or drop --ply"
            )
        if args.occluder is not None or args.occluder_free:
            raise SystemExit(
                "--scene-variant low_altitude ships its own ground-level "
                "obstacle; --occluder/--occluder-free would contradict it"
            )
        if (target_pos != DEFAULT_TARGET_POS) or (start_pos != DEFAULT_START_POS):
            raise SystemExit(
                "--scene-variant low_altitude fixes the target and start "
                "geometry (target on the ground, start level with its top); "
                "drop --target-pos/--start-pos"
            )
        target_pos, start_pos, obstacle = low_altitude_geometry(args)
        occluders = (obstacle,)
    elif variant == "visual_required":
        if args.ply:
            raise SystemExit(
                "--scene-variant visual_required authors its own synthetic scene; "
                "pass --ply without the variant, or drop --ply"
            )
        if args.occluder is not None or args.occluder_free:
            raise SystemExit(
                "--scene-variant visual_required ships its own wall (the loss "
                "occluder); --occluder/--occluder-free would contradict it"
            )
        if target_pos != DEFAULT_TARGET_POS:
            raise SystemExit(
                "--scene-variant visual_required authors the target position from "
                "--instance-bearing-deg; drop --target-pos"
            )
        if start_pos != DEFAULT_START_POS:
            raise SystemExit(
                "--scene-variant visual_required starts at the origin (the layout is "
                "authored around the start heading); drop --start-pos"
            )
        target_pos, decoy, landmark, start_pos, occluders = visual_required_geometry(args)
        occluders = occluders or (visual_required_wall(args),)
        loss_bearing_deg = visual_required_loss_bearing(args)
    elif args.occluder is None and not args.occluder_free and not args.ply:
        occluders = (DEFAULT_WALL,)  # the synthetic scene ships with its wall
    scenario = Scenario(
        target_name=str(args.target_name),
        target_pos=target_pos,
        target_radius=float(args.target_radius),
        start_pos=start_pos,
        start_yaw_deg=float(args.start_yaw_deg),
        loss_bearing_deg=loss_bearing_deg,
        loss_distance_m=float(args.loss_distance_m),
        approach_m=float(args.approach_m),
        success_dist_m=float(args.success_dist),
        hfov_deg=float(args.hfov_deg),
        scan_views=int(args.scan_views),
        occluders=occluders,
        ground_z=ground_z,
        decoy_pos=(
            None if getattr(args, "decoy_pos", None) is None
            else tuple(float(v) for v in args.decoy_pos)
        ),
        landmark_pos=(
            None if getattr(args, "landmark_pos", None) is None
            else tuple(float(v) for v in args.landmark_pos)
        ),
    )
    if variant == "visual_required":
        # the decoy is what makes "did it pick the right one" measurable, so the
        # variant authors it rather than taking it from --decoy-pos
        scenario = replace(scenario, decoy_pos=decoy, landmark_pos=landmark)
        failures = visual_required_checks(scenario, args)
        if failures:
            raise SystemExit(
                "--scene-variant visual_required does not hold with these numbers:\n  - "
                + "\n  - ".join(failures)
            )
    return scenario


DEFAULT_INSTRUCTION = (
    "Fly to the target and confirm you reached it. If the target leaves your "
    "view, use the observations you already collected to find it again."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # --- scene (mirrors scripts/run_vlm_nav_tools.py) ---
    parser.add_argument("--world", choices=["cpu", "higs"], default="cpu")
    parser.add_argument("--ply", type=str, default=None,
                        help="3DGS scene PLY; omit to synthesize the scenario scene")
    parser.add_argument("--scene-variant", choices=list(SCENE_VARIANTS), default="default",
                        help="default: the floating target of the M2 acceptance "
                             "runs; low_altitude: a ground target behind a ground-level "
                             "obstacle, with the low-altitude start band and altitude "
                             "envelope (target/start/obstacle geometry is fixed); "
                             "visual_required: the relation scene - two identical "
                             "instances, a landmark only some views hold, and a loss "
                             "pose that sees none of them (see "
                             "visual_required_geometry)")
    parser.add_argument("--scene-scale", type=float, default=1.0)
    parser.add_argument("--min-opacity", type=float, default=0.0)
    parser.add_argument("--render-size", type=int, nargs=2, default=(64, 64),
                        metavar=("W", "H"))
    parser.add_argument("--hfov-deg", type=float, default=60.0)
    parser.add_argument("--start-pos", type=float, nargs=3, default=DEFAULT_START_POS)
    parser.add_argument("--start-yaw-deg", type=float, default=0.0)
    parser.add_argument("--target-name", type=str, default="target")
    parser.add_argument("--target-pos", type=float, nargs=3, default=DEFAULT_TARGET_POS,
                        metavar=("X", "Y", "Z"), help="FLU, scene-centered")
    parser.add_argument("--target-radius", type=float, default=0.2)
    parser.add_argument("--success-dist", type=float, default=2.5,
                        help="meters to the target surface that counts as success")
    # --- scenario ---
    parser.add_argument("--loss-bearing-deg", type=float, default=-90.0,
                        help="world-frame bearing of the leg that loses the view")
    parser.add_argument("--loss-distance-m", type=float, default=3.0)
    parser.add_argument("--approach-m", type=float, default=0.8,
                        help="closing leg along the recalled heading after the return")
    parser.add_argument("--loss-trigger", choices=list(LOSS_TRIGGERS), default="auto",
                        help="auto: model (mock) / evidence (real backend); model: the "
                             "model's own navigation leg; evidence: the environment "
                             "relocates the drone after the first observation; "
                             "capture: the environment relocates it after the first "
                             "capture that had the target in view (observe or a "
                             "grounding capture) - the M2 precondition itself; "
                             "relation: the same relocation, armed by the first "
                             "capture that held the target *and* the landmark at once "
                             "- the one frame that names which instance is the target")
    parser.add_argument("--loss-after-observations", type=int, default=1,
                        help="with --loss-trigger evidence: observations before the move")
    parser.add_argument("--occluder", type=float, nargs=6, action="append", default=None,
                        metavar=("X", "Y", "Z", "DX", "DY", "DZ"),
                        help="occluder box (center + size); repeatable. It is both "
                             "rendered into the synthetic scene and used for the "
                             "visibility check - with --ply the scene must already "
                             "contain the geometry. Default: the synthetic wall when "
                             "--ply is omitted, none otherwise")
    parser.add_argument("--occluder-free", action="store_true",
                        help="no occluder even for the synthetic scene")
    parser.add_argument("--scan-views", type=int, default=8,
                        help="views of observe(mode=lookaround)")
    parser.add_argument("--scan-pitch-deg", type=float, default=50.0)
    parser.add_argument("--relocate-tol-m", type=float, default=0.5,
                        help="distance to the recalled pose that counts as re-localized")
    parser.add_argument("--decoy-pos", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="synthetic scene: a second instance (same object as "
                             "the target, rendered next to it) the task is not "
                             "about; with it the metrics report which instance "
                             "each leg and the final pose went to")
    parser.add_argument("--landmark-pos", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="synthetic scene: the landmark the instruction names "
                             "the target by (a bright block next to the target)")
    # --- the relation scene (--scene-variant visual_required) ---
    parser.add_argument("--relation-side", choices=("left", "right"), default="left",
                        help="visual_required: the side of the start->instance sight "
                             "line the landmark stands on. left is the authored "
                             "layout, right is its mirror (a second, independent "
                             "scene of the same design)")
    parser.add_argument("--instance-bearing-deg", type=float,
                        default=VISUAL_INSTANCE_BEARING_DEG,
                        help="visual_required: bearing of the correct instance from "
                             "the start heading (the landmark lands ~11 deg further "
                             "out, outside the forward view)")
    parser.add_argument("--instance-range-m", type=float, default=VISUAL_INSTANCE_RANGE_M,
                        help="visual_required: distance of the correct instance")
    parser.add_argument("--decoy-bearing-deg", type=float, default=VISUAL_DECOY_BEARING_DEG,
                        help="visual_required: bearing of the decoy - the same object, "
                             "nearer and almost straight ahead")
    parser.add_argument("--decoy-range-m", type=float, default=VISUAL_DECOY_RANGE_M,
                        help="visual_required: distance of the decoy")
    parser.add_argument("--lamp-offset-m", type=float, default=VISUAL_LAMP_OFFSET_M,
                        help="visual_required: how far the landmark stands from the "
                             "correct instance (it is the relation the instruction "
                             "names, and it must stay outside the forward view)")
    parser.add_argument("--relation-visible-at-start", type=str, default="off",
                        choices=list(SWITCH_CHOICES), dest="relation_visible_at_start",
                        help="visual_required: off (the experiment) puts the landmark "
                             "outside the forward view, so the grounding view cannot "
                             "resolve the relation; on (the control) puts it beside the "
                             "correct instance inside that view - the same scene with "
                             "exactly the property under test moved")
    # --- agent loop ---
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--backend", type=str, default="mock",
                        help="mock | hf | openai")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model", type=str, default=None,
                        help="model name for the openai backend")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--image-size", type=int, nargs=2, default=(640, 360),
                        metavar=("W", "H"))
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--tool-budget", type=int, default=12)
    parser.add_argument("--context-frames", type=int, default=2)
    # --- memory layers (the runner's switches; the E2 arms pin one each) ---
    parser.add_argument("--memory-spatial", type=str, default=DEFAULT_SWITCH,
                        dest="memory_spatial", choices=list(SWITCH_CHOICES),
                        help="on (default): the memory keeps the spatial store "
                             "(candidates and places); off: built with "
                             "spatial_memory=False")
    parser.add_argument("--memory-episodic", type=str, default=DEFAULT_SWITCH,
                        dest="memory_episodic", choices=list(SWITCH_CHOICES),
                        help="on (default): the memory keeps its episodic evidence "
                             "layer (frames, sightings, scoped null results, the "
                             "duplicate merge); off: built with "
                             "episodic_memory=False - nothing is written or returned "
                             "as evidence, a query_memory hit is a capture record "
                             "(pose, heading, step, no image) and the spatial and "
                             "working layers stay on - the memory_spatial_only arm "
                             "pins this")
    parser.add_argument("--recall-images", type=str, default=DEFAULT_SWITCH,
                        dest="recall_images", choices=list(SWITCH_CHOICES),
                        help="on (default): the frames a query_memory result "
                             "carries are decoded into the next turn's context "
                             "images; off: the retrieval result is left as it is "
                             "and no recalled frame is attached (poses, candidate "
                             "positions and the sighting record stay) - the "
                             "memory_text_only arm pins this, and it is independent "
                             "of the two --memory-* layer switches")
    # --- experiment ---
    parser.add_argument("--modes", type=str, default="memory,basic",
                        help="tool sets to compare: memory (--toolset memory, the "
                             "layers the two --memory-* switches set), "
                             "memory_no_evidence (the same tool set with the "
                             "evidence ablation on), basic (--toolset basic, the "
                             "fixed-keyframe baseline), memory_spatial_only (the "
                             "memory catalogue with --memory-episodic off) and "
                             "memory_text_only (the memory catalogue with "
                             "--recall-images off)")
    parser.add_argument("--ablate-evidence", action="store_true",
                        help="add the memory_no_evidence arm to --modes (an arm "
                             "already requested is not duplicated): the retrieval "
                             "tool set stays exactly the same, but an observation "
                             "written before the loss of view is accepted and "
                             "dropped, so query_memory has no history to return")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args(argv)


def build_vlm_factory(args: argparse.Namespace, scenario: Scenario
                      ) -> Callable[[str, ScenarioDriver], Any]:
    """``make_vlm(mode, driver)``: the scripted policy for mock, one backend else."""
    backend: dict[str, Any] = {}

    def make_vlm(mode: str, driver: ScenarioDriver):
        if str(args.backend).startswith("mock"):
            return ReacquireVLM(mode=mode, scenario=scenario, driver=driver,
                                target_name=scenario.target_name)
        if "vlm" not in backend:
            runner_args = argparse.Namespace(
                backend=args.backend,
                mock_script=None,
                model_path=args.model_path,
                model=args.model,
                base_url=args.base_url,
                device=args.device,
                max_new_tokens=int(args.max_new_tokens),
                image_size=tuple(args.image_size),
                num_threads=args.num_threads,
                no_think=bool(args.no_think),
            )
            backend["vlm"] = RUNNER.build_vlm(runner_args)
        return backend["vlm"]

    return make_vlm


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    import numpy as _np

    _np.random.seed(int(args.seed))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_config.json", "w") as handle:
        json.dump(vars(args), handle, indent=2, default=str)

    scenario = build_scenario(args)
    modes = [mode.strip() for mode in str(args.modes).split(",") if mode.strip()]
    if bool(getattr(args, "ablate_evidence", False)) and ABLATED_MODE not in modes:
        # the flag is the explicit spelling of the third arm: same catalogue as
        # ``memory``, one archive rule different
        modes.append(ABLATED_MODE)
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown:
        raise SystemExit(f"unknown --modes {unknown}; choose from {list(MODES)}")
    # one scene for the whole run, and the scenario in that scene's FLU frame:
    # both the rendered world and every analytic check then describe one world
    ply_path, scenario, scene_origin = scene_for_scenario(args, scenario, out_dir)

    wall = scenario.occluders[0] if scenario.occluders else None
    envelope = flight_envelope(args, scenario)
    scenario_report = {
        "scenario": scenario.to_dict(),
        "scene_ply": str(ply_path),
        "scene_origin_flu": [round(float(v), 6) for v in np.asarray(scene_origin)],
        "preconditions": {
            "target_visible_at_start": target_visibility(
                scenario.start_pos, math.radians(scenario.start_yaw_deg), scenario
            )["visible"],
            "target_visible_at_loss_pose": target_visibility(
                scenario.loss_pose(scenario.start_pos),
                math.radians(scenario.loss_bearing_deg), scenario
            )["visible"],
            "scan_from_loss_pose_visible": scan_visibility(
                scenario.loss_pose(scenario.start_pos), scenario
            )["visible"],
            "occluder": wall.to_dict() if wall is not None else None,
            "decoy_visible_at_start": (
                None if scenario.decoy_pos is None
                else target_visibility(
                    scenario.start_pos, math.radians(scenario.start_yaw_deg),
                    replace(scenario, target_pos=scenario.decoy_pos)
                )["visible"]
            ),
            "landmark_visible_at_start": (
                None if scenario.landmark_pos is None
                else target_visibility(
                    scenario.start_pos, math.radians(scenario.start_yaw_deg),
                    replace(scenario, target_pos=scenario.landmark_pos)
                )["visible"]
            ),
        },
        "scene_variant": str(getattr(args, "scene_variant", "default")),
        "flight_envelope": (
            None if envelope is None else envelope.to_dict()
        ),
        "modes": modes,
        "budget": {"tool_budget": int(args.tool_budget), "max_turns": int(args.max_turns)},
        "note": (
            "geometry of the scenario, evaluated analytically; the target must be "
            "visible at the start and invisible (in FOV, occluded, or both) after "
            "the loss move, otherwise the episode does not exercise re-acquisition. "
            "All coordinates are in the scene's FLU frame: for the synthetic scene "
            "that origin is the authored vertex centroid (scene_origin_flu is where "
            "it sits in the coordinates given on the command line)"
        ),
    }
    with open(out_dir / "scenario.json", "w") as handle:
        json.dump(scenario_report, handle, indent=2, default=str)

    make_vlm = build_vlm_factory(args, scenario)
    trials_by_mode: dict[str, list[dict]] = {}
    for mode in modes:
        trials: list[dict] = []
        for trial in range(max(1, int(args.trials))):
            if int(args.trials) > 1:
                trial_dir = out_dir / "modes" / mode / f"trial_{trial:02d}"
            else:
                trial_dir = out_dir / "modes" / mode
            trials.append(
                run_trial(
                    mode, scenario, args, trial_dir, make_vlm,
                    episode_id=f"m2_{mode}_{int(args.seed):03d}_{trial:02d}",
                    ply_path=ply_path,
                )
            )
        trials_by_mode[mode] = trials

    aggregates = {mode: aggregate(trials) for mode, trials in trials_by_mode.items()}
    comparison = build_comparison(aggregates, args, scenario)
    comparison["scenario_preconditions"] = scenario_report["preconditions"]
    with open(out_dir / "comparison.json", "w") as handle:
        json.dump(comparison, handle, indent=2, default=str)
    print_comparison(comparison)
    return comparison


if __name__ == "__main__":
    main()
