"""Episode loop: render one frame -> VLM decision -> apply action -> repeat.

Metrics and artifacts follow the spirit of `script/eval_gwp_searching_3dgs.py`:
per-step frames, an action trace, and a summary JSON with success /
distance-reach / collision-style flags.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from .actions import NavAction, parse_action, parse_scan_choice
from .candidates import CandidateTracker
from .observations import resolve_intrinsics
from .prompts import (
    SCAN_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_scan_prompt,
    build_user_prompt,
    summarize_action,
)
from .vlm import VLMBackend, frame_to_pil
from .world import estimate_point_world


def _declared_z(value: Any) -> float | None:
    """FLU z of a declared position or bound (3-vector or scalar), or None."""
    if value is None:
        return None
    if hasattr(value, "detach"):  # a torch tensor declaration (env tensors)
        value = value.detach().cpu()
    try:
        values = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.size == 0 or not np.isfinite(values).all():
        return None
    return float(values[2] if values.size >= 3 else values[0])


def _flight_altitude_band(world: Any) -> tuple[float, float] | None:
    """(lower, upper) FLU altitude the harness may fly to, or None when unknown.

    The floor is the env's ``navigation_min_flu`` when there is one, else the
    world's declared ``z_min`` (the splat worlds take the flyable floor the
    runner measured); the env's cruise window caps it from above and the
    world's ``z_max`` is the no-fly ceiling. A world that declares no floor at
    all leaves the caller's step untouched.
    """
    env = getattr(world, "env", None)
    floor = None if env is None else _declared_z(
        getattr(env, "navigation_min_flu", None)
    )
    if floor is None:
        floor = _declared_z(getattr(world, "z_min", None))
    if floor is None:
        return None
    lower, upper = floor, math.inf
    if env is not None:
        window_max = _declared_z(getattr(env, "cruise_window_max_m", None))
        if window_max is not None and window_max > 0.0:
            window_min = _declared_z(getattr(env, "cruise_window_min_m", None)) or 0.0
            lower = floor + max(0.0, window_min)
            upper = floor + window_max
    ceiling = _declared_z(getattr(world, "z_max", None))
    if ceiling is not None:
        upper = min(upper, ceiling)
    return lower, max(upper, lower)


@dataclass
class StepRecord:
    step: int
    action_summary: str
    outcome: str
    position_flu: tuple[float, float, float]
    yaw_deg: float
    distance_to_target: float
    parse_ok: bool
    raw_response: str = field(repr=False, default="")


@dataclass
class EpisodeSummary:
    instruction: str
    target_name: str
    success: bool
    finish_reason: str
    steps: int
    final_distance_m: float
    min_distance_m: float
    path_length_m: float
    wall_time_s: float
    parse_failures: int
    vlm_calls: int
    collision_hits: int
    records: list[dict[str, Any]]
    # eval-aligned visibility metrics (target_in_fov per decision step)
    first_visible_step: int = -1
    visible_steps: int = 0
    max_visible_streak: int = 0
    initial_target_visible: bool = False
    # pointed pixels that could not be turned into a world position (no declared
    # camera for an off-axis pixel, or no range): reported, never guessed
    unlocalized_sightings: int = 0


class SemanticNavHarness:
    def __init__(
        self,
        world,
        vlm: VLMBackend,
        instruction: str,
        out_dir: str | Path,
        max_steps: int = 30,
        success_dist_m: float = 2.0,
        success_uses_surface: bool = True,
        done_requires_success_dist: bool = True,
        save_frames: bool = True,
        make_video: bool = True,
        vlm_context: dict[str, Any] | None = None,
        context_frames: int = 3,
        look_around: bool = True,
        scan_views: int = 8,
        rescan_after: int | None = 6,
        done_retry: bool = True,
        bev: bool = True,
        scan_pitch_deg: float = 50.0,
        done_verify: bool = True,
        candidates_enabled: bool = True,
        keyframe_enabled: bool = True,
    ):
        self.world = world
        self.vlm = vlm
        self.instruction = instruction
        self.out_dir = Path(out_dir)
        self.max_steps = int(max_steps)
        self.success_dist_m = float(success_dist_m)
        self.success_uses_surface = bool(success_uses_surface)
        self.done_requires_success_dist = bool(done_requires_success_dist)
        self.save_frames = bool(save_frames)
        self.make_video = bool(make_video)
        self.vlm_context = dict(vlm_context or {})
        self._vlm_calls = 0
        self.candidates = CandidateTracker()
        self._active_candidate = None
        self._keyframe: torch.Tensor | None = None
        self.unlocalized_sightings = 0
        # range claimed by the last trusted scan sighting (0 = none yet); a
        # later claim that disagrees with it by 2x gets a crop review
        self._last_scan_claim_dist = 0.0
        self.context_frames = max(1, int(context_frames))
        self.look_around = bool(look_around)
        self.scan_views = max(4, int(scan_views))
        self.rescan_after = int(rescan_after) if rescan_after else 0
        self.done_retry = bool(done_retry)
        self.bev = bool(bev)
        self.scan_pitch_deg = float(scan_pitch_deg)
        self.done_verify = bool(done_verify)
        self.candidates_enabled = bool(candidates_enabled)
        self.keyframe_enabled = bool(keyframe_enabled)
        self.frames_dir = self.out_dir / "frames"

    def _success_distance(self) -> float:
        threshold = self.success_dist_m
        if self.success_uses_surface:
            threshold += float(getattr(self.world.target, "radius_m", 0.0))
        return threshold

    @staticmethod
    def _unlocalized_reason(intrinsics: dict[str, Any]) -> str:
        """Why a pointed pixel has no world position: no camera, or no range."""
        return "intrinsics_unknown" if intrinsics.get("hfov_deg") is None else "no_range"

    def _look_around(self, history: list[dict[str, Any]], tag: int | str = 0,
                     n_views: int | None = None, spread_deg: float | None = None) -> None:
        """Scan the surroundings and let the VLM pick a heading.

        Full circle by default; with ``spread_deg`` + fewer ``n_views`` this is
        a narrow re-acquisition scan around the current heading (used after a
        rejected done).

        A scan reply is a claim, not evidence: when the model reports
        ``target_visible=false`` the drone keeps its current heading instead of
        rotating onto the view it picked (run13 community_atm s11 was turned 270
        degrees away from a target that was in front of it), and a sighting
        whose range disagrees with the previous trusted claim by 2x is
        re-checked with one zoomed crop before the drone moves on it."""
        n_level = int(n_views) if n_views else self.scan_views
        views = self.world.scan_views(n_level, pitch_down_deg=self.scan_pitch_deg,
                                      spread_deg=spread_deg)
        n_pitch = len(views) - n_level
        if self.save_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            for k, view in enumerate(views):
                frame_to_pil(view).save(self.frames_dir / f"scan_{tag}_{k:02d}.png")
        prompt = build_scan_prompt(self.instruction, len(views), n_pitch=n_pitch)
        self._vlm_calls += 1
        raw = self.vlm.decide(views, SCAN_SYSTEM_PROMPT, prompt, self._vlm_context())
        view_idx, visible, scan_dist = parse_scan_choice(raw, len(views))
        print(f"[scan] chosen view={view_idx} visible={visible} dist={scan_dist:.1f} | {raw[:200]}", flush=True)
        if view_idx is None:
            history.append({"step": tag, "action_summary": "scan (unparsed)",
                            "outcome": "kept initial heading"})
            return
        visible, reason = self._trust_scan_claim(views, view_idx, visible, scan_dist)
        current = math.degrees(float(self.world.yaw_rad)) % 360.0
        if not visible:
            # never steer onto a view the model itself says shows no target:
            # acting on such a claim can rotate the nose away from a target
            # that is actually in front of the drone
            history.append({"step": tag, "action_summary": f"scan -> view {view_idx}",
                            "outcome": f"kept heading {current:.0f} deg ({reason})"})
            return
        # view k heading: full circle = current + 360*k/n; narrow spread =
        # current - spread/2 + spread*k/(n-1); pitched-down views sit at the
        # current / opposite heading and imply a downward target
        pitched = view_idx >= n_level
        if pitched:
            target_yaw = current
            if view_idx == n_level + 1:
                target_yaw = (target_yaw + 180.0) % 360.0
        elif spread_deg is None:
            target_yaw = (current + 360.0 * view_idx / n_level) % 360.0
        else:
            s = float(spread_deg)
            target_yaw = (current - s / 2.0 + s * view_idx / max(1, n_level - 1) + 360.0) % 360.0
        delta = (target_yaw - current + 180.0) % 360.0 - 180.0
        # apply the turn in clamp-sized chunks through the normal action path
        while abs(delta) > 1e-6:
            step_deg = max(-45.0, min(45.0, delta))
            self.world.apply_action(NavAction(kind="turn", yaw_deg=step_deg))
            delta -= step_deg
        outcome = f"turned to heading {target_yaw:.0f} deg"
        # the model spotted the target: fly most of the estimated distance now
        if scan_dist > 0.5:
            leg = min(scan_dist * 0.7, 8.0)
            up_m = 0.0
            if pitched:
                up_m = self._clamped_vertical_step(-min(1.5, 0.4 * scan_dist))
            fly_outcome = self.world.apply_action(
                NavAction(kind="waypoint", forward_m=leg, up_m=up_m)
            )
            outcome += f"; flew {leg:.1f} m toward it ({fly_outcome})"
        history.append({"step": tag, "action_summary": f"scan -> view {view_idx}",
                        "outcome": outcome})

    def _clamped_vertical_step(self, up_m: float) -> float:
        """``up_m`` rewritten so the step ends inside the flyable altitude band.

        A pitched scan view asks for a descent toward a target below. The world
        clamps only at its own bounds, so the step is clamped here instead:
        above the navigable floor, inside the env's cruise window and under the
        world's no-fly ceiling (:func:`_flight_altitude_band`). Without a
        declared floor the offset is returned unchanged.
        """
        band = _flight_altitude_band(self.world)
        if band is None:
            return float(up_m)
        position = np.asarray(self.world.position_flu, dtype=np.float64).reshape(-1)
        current_z = float(position[2])
        target_z = min(max(current_z + float(up_m), band[0]), band[1])
        return target_z - current_z

    def _trust_scan_claim(self, views: list[torch.Tensor], view_idx: int,
                          visible: bool, scan_dist: float) -> tuple[bool, str]:
        """Decide whether a scan sighting claim may steer the drone.

        Returns ``(visible, reason)``; ``reason`` is non-empty only for a
        rejected claim. A range estimate that disagrees with the previous
        trusted claim by 2x marks the sighting as probably hallucinated, so the
        picked view gets one zoomed crop review before anything is executed;
        only a clear "no target in the crop" verdict discards the claim.
        """
        if not visible:
            return False, "scan reported no visible target"
        claim_dist = float(scan_dist)
        previous = self._last_scan_claim_dist
        if claim_dist > 0.0 and previous > 0.0:
            ratio = max(claim_dist, previous) / min(claim_dist, previous)
            if ratio >= 2.0:
                verdict = self._verify_view_crop(views[view_idx])
                print(f"[scan] crop review of view {view_idx} "
                      f"({claim_dist:.1f} m claimed vs {previous:.1f} m before): {verdict}",
                      flush=True)
                if verdict is False:
                    return False, (
                        "crop review rejected the claimed sighting "
                        f"({claim_dist:.1f} m claimed vs {previous:.1f} m before)"
                    )
        self._last_scan_claim_dist = claim_dist
        return True, ""

    def _save_frame(self, frame: torch.Tensor, step: int) -> None:
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        frame_to_pil(frame).save(self.frames_dir / f"step_{step:03d}.png")

    def _verify_view_crop(self, view: torch.Tensor) -> bool | None:
        """Re-ask the verifier about the center of a scan view the model claims
        shows the target. True/False/None(inconclusive)."""
        return self._verify_target_crop(view, None)

    def _verify_done_crop(self, frame: torch.Tensor, pixel: tuple[float, float]) -> bool | None:
        """Second cheap VLM call: zoom into the pointed region and ask whether
        it really is the instructed target. True/False/None(inconclusive)."""
        return self._verify_target_crop(frame, pixel)

    def _verify_target_crop(self, frame: torch.Tensor,
                            pixel: tuple[float, float] | None) -> bool | None:
        """One VLM call on an enlarged crop: the region around ``pixel`` (or the
        frame center when ``pixel`` is None) is re-examined for the instructed
        target - used to veto a done claim and a suspicious scan sighting."""
        from .prompts import VERIFY_SYSTEM_PROMPT

        H, W = int(frame.shape[1]), int(frame.shape[2])
        if pixel is None:
            u, v = W // 2, H // 2
        else:
            u = int(pixel[0] / 1000.0 * W)
            v = int(pixel[1] / 1000.0 * H)
        r = max(32, int(0.30 * W))
        crop = frame[:, max(0, v - r): min(H, v + r), max(0, u - r): min(W, u + r)]
        if crop.numel() == 0:
            return None
        prompt = (
            f"Navigation instruction: {self.instruction}\n\n"
            "This crop zooms into the region the drone marked as the target. "
            "Does it clearly show the described target object?"
        )
        self._vlm_calls += 1
        raw = self.vlm.decide([crop], VERIFY_SYSTEM_PROMPT, prompt, None)
        text = raw.replace(" ", "").lower()
        if '"verified":true' in text:
            return True
        if '"verified":false' in text:
            return False
        return None

    def _explore_step(self, history: list[dict[str, Any]], step: int,
                      pos, sectors: dict[tuple[int, int], set[int]]) -> None:
        """Current spot is fully scanned but produced no candidate: fly 2.5 m
        toward an unobserved sector and scan there (move-and-scan beats
        rescan-in-place, goal.md P2)."""
        cell = (int(pos[0] // 2), int(pos[1] // 2))
        covered = sectors.get(cell, set())
        heading = math.degrees(float(self.world.yaw_rad)) % 360.0
        pick = next((s for s in range(8) if s not in covered), None)
        if pick is None:
            pick = random.randrange(8)
        bearing = pick * 45.0 + 22.5  # sector center, world frame
        delta = (bearing - heading + 180.0) % 360.0 - 180.0
        if abs(delta) > 1.0:
            self.world.apply_action(NavAction(kind="turn", yaw_deg=max(-45.0, min(45.0, delta))))
        outcome = self.world.apply_action(NavAction(kind="waypoint", forward_m=2.5))
        history.append({"step": step, "action_summary": f"explore -> bearing {bearing:.0f}",
                        "outcome": outcome})
        self._look_around(history, tag=f"{step}-explore")

    def _vlm_context(self) -> dict[str, Any]:
        ctx = dict(self.vlm_context)
        ctx.setdefault("position_flu", tuple(float(v) for v in self.world.position_flu))
        ctx.setdefault("yaw_rad", float(self.world.yaw_rad))
        target = getattr(self.world, "target", None)
        if target is not None:
            ctx.setdefault("target_pos_flu", tuple(float(v) for v in target.position_flu))
        ctx.setdefault("done_dist", self._success_distance())
        return ctx

    def run(self) -> EpisodeSummary:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, Any]] = []
        records: list[StepRecord] = []
        start = time.time()
        min_dist = self.world.distance_to_target()
        last_feedback: str | None = None
        finish_reason = "max_steps"
        success = False
        parse_failures = 0
        step = 0
        start_yaw = math.degrees(float(self.world.yaw_rad))
        # odometry-based progress tracking (no ground truth in the agent loop)
        last_progress_step = 0
        last_progress_pos = np.asarray(self.world.position_flu, dtype=np.float64).copy()
        self._vlm_calls = 0
        visible_steps = 0
        visible_streak = 0
        max_visible_streak = 0
        first_visible_step = -1
        # sector-coverage memory: compass sectors observed per 2 m cell
        sectors: dict[tuple[int, int], set[int]] = {}
        frame_history: deque[torch.Tensor] = deque(maxlen=self.context_frames - 1)
        if self.look_around:
            self._look_around(history)
        vis_fn0 = getattr(self.world, "target_in_fov", None)
        initial_target_visible = bool(vis_fn0()) if callable(vis_fn0) else False

        for step in range(1, self.max_steps + 1):
            frame = self.world.render()
            if self.save_frames:
                self._save_frame(frame, step)
            # eval-aligned visibility bookkeeping (FOV vs the semantic target)
            vis_fn = getattr(self.world, "target_in_fov", None)
            visible_now = bool(vis_fn()) if callable(vis_fn) else False
            if visible_now:
                visible_steps += 1
                visible_streak += 1
                max_visible_streak = max(max_visible_streak, visible_streak)
                if first_visible_step < 0:
                    first_visible_step = step
            else:
                visible_streak = 0
            heading = math.degrees(float(self.world.yaw_rad))
            net_turn = (heading - start_yaw + 180.0) % 360.0 - 180.0
            p = self.world.position_flu
            odometry = (
                f"Odometry: position=({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) m, "
                f"heading={heading:.0f} deg (turned {net_turn:+.0f} deg from start)."
            )
            center_sector = int((heading % 360.0) / 45.0)
            seen = sectors.setdefault((int(p[0] // 2), int(p[1] // 2)), set())
            seen.update({(center_sector - 1) % 8, center_sector, (center_sector + 1) % 8})
            frames = [*frame_history, frame]
            if self.keyframe_enabled and self._keyframe is not None and len(frames) >= 2:
                # pin the frame where the target was last pointed at as the
                # oldest context image (goal.md P3: keyframe over FIFO eviction)
                frames[0] = self._keyframe
            bev_img = None
            if self.bev:
                bev_fn = getattr(self.world, "bev_image", None)
                if callable(bev_fn):
                    # the red dot means "a place the model itself pointed at":
                    # world.last_subgoal is where the drone just flew to, and
                    # drawing it as the target made run13 cells read their own
                    # wake as a target estimate and fly away from the goal
                    marker = self._active_candidate
                    if marker is None:
                        marker = self.candidates.best()
                    estimate = marker.position_flu if marker is not None else None
                    bev_img = bev_fn(target_estimate=estimate)
            if bev_img is not None:
                frames = [bev_img, *frames]
                if self.save_frames:
                    frame_to_pil(bev_img).save(self.frames_dir / f"bev_{step:03d}.png")
            user_prompt = build_user_prompt(
                self.instruction, step, self.max_steps, history, last_feedback,
                odometry=odometry, n_frames=len(frames) - (1 if bev_img is not None else 0),
                bev=bev_img is not None,
                candidates_text=(
                    self.candidates.describe(p, float(self.world.yaw_rad))
                    if self.candidates_enabled else None
                ),
                keyframe=self._keyframe is not None,
            )
            t0 = time.time()
            self._vlm_calls += 1
            raw = self.vlm.decide(frames, SYSTEM_PROMPT, user_prompt, self._vlm_context())
            latency = time.time() - t0
            frame_history.append(frame)
            action: NavAction = parse_action(raw)
            if not action.parse_ok:
                parse_failures += 1
                last_feedback = (
                    "your previous reply was not valid JSON; reply with exactly "
                    "one JSON object."
                )
            else:
                last_feedback = None
            if action.pixel is not None and action.kind == "move":
                # pixel-servo: the pointed direction + depth wins over the blind
                # forward/right magnitudes - the autopilot steers precisely to
                # the pointed point instead of letting heading estimates drift
                action = NavAction(
                    kind="waypoint", pixel=action.pixel,
                    distance_m=action.distance_m, reason=action.reason, parse_ok=True,
                )
            if action.pixel is not None:
                # point-to-navigate / done-with-pixel: fuse into candidate memory
                if self.candidates_enabled:
                    refresh_fn = getattr(self.world, "refresh_sensing", None)
                    if callable(refresh_fn):
                        refresh_fn()
                    # the pointed pixel is un-projected with the frame's own
                    # camera: the resolution that was actually rendered and the
                    # world's declared field of view (audit I1 - the legacy
                    # 320x180@86 deg default stretched a 60 deg frame by 1.615x).
                    # With no camera declared the pixel has no world position,
                    # so nothing is remembered about it
                    intrinsics = resolve_intrinsics(world=self.world, frame=frame)
                    estimate = estimate_point_world(
                        self.world, action.pixel, action.distance_m,
                        intrinsics=intrinsics,
                    )
                    if estimate is None:
                        self.unlocalized_sightings += 1
                        print(
                            f"[harness] pointed pixel ({action.pixel[0]:.0f},"
                            f"{action.pixel[1]:.0f}) not localized "
                            f"({self._unlocalized_reason(intrinsics)}): "
                            "no candidate remembered",
                            flush=True,
                        )
                    else:
                        self._active_candidate = self.candidates.observe(
                            estimate, step, note=f"{action.kind}: {action.reason}"
                        )
                        if self.keyframe_enabled:
                            self._keyframe = frame  # pin the sighting as long-term context
            if action.kind == "scan":
                self._look_around(history, tag=step)
                outcome = "rescan"
                last_progress_step = step
            else:
                outcome = self.world.apply_action(action)
            dist = self.world.distance_to_target()
            min_dist = min(min_dist, dist)
            if any(k in str(outcome) for k in (
                    "planner failed", "collision", "stalled", "leg timeout",
                    "terminated", "no range")):
                # typed recovery: the failure kind drives the next prompt's advice
                last_feedback = (
                    f"the last leg failed ({outcome}); the direct path is blocked or "
                    "unreachable - pick a different subgoal (another approach angle or "
                    "a different standoff point), or reply scan to re-assess the area."
                )
            # progress is measured by odometry (displacement), never by the
            # ground-truth target distance
            if float(np.linalg.norm(p - last_progress_pos)) > 0.4:
                last_progress_pos = np.asarray(p, dtype=np.float64).copy()
                last_progress_step = step
            summary = summarize_action(action)
            records.append(
                StepRecord(
                    step=step,
                    action_summary=summary,
                    outcome=outcome,
                    position_flu=tuple(float(v) for v in self.world.position_flu),
                    yaw_deg=math.degrees(float(self.world.yaw_rad)),
                    distance_to_target=dist,
                    parse_ok=action.parse_ok,
                    raw_response=raw,
                )
            )
            history.append({"step": step, "action_summary": summary, "outcome": outcome})
            print(
                f"[step {step:02d}] {summary} | dist={dist:.2f}m "
                f"| {latency:.1f}s | {action.reason}",
                flush=True,
            )
            if dist <= self._success_distance():
                success, finish_reason = True, "within_success_distance"
                break
            # independent done verification (goal.md P1): zoom into the pointed
            # pixel and re-ask the VLM whether that crop really is the target -
            # catches wrong-instance stops even inside the distance gate
            if action.is_terminal and action.pixel is not None and self.done_verify:
                verified = self._verify_done_crop(frame, action.pixel)
                if verified is False:
                    if self._active_candidate is not None:
                        self.candidates.reject(self._active_candidate.id,
                                               "crop verification failed")
                        self._active_candidate = None
                        self._keyframe = None
                    outcome = "done rejected: crop does not match the target"
                    records[-1].outcome = outcome
                    history[-1]["outcome"] = outcome
                    last_feedback = (
                        "I zoomed into the point you marked: it does NOT match the "
                        f"described target ({self.instruction}). You were tracking the "
                        "wrong object - discard it and look for a different one."
                    )
                    # typed recovery: narrow re-acquisition scan around the
                    # current heading after a rejected done
                    self._look_around(history, tag=f"{step}-reacquire",
                                      n_views=5, spread_deg=120.0)
                    continue
            if action.is_terminal:
                if self.done_requires_success_dist:
                    if self.done_retry and step < self.max_steps:
                        # Reject premature "done" with OBSERVABLE evidence only
                        # (depth read at the pointed pixel); the true target
                        # distance stays evaluator-side and never enters prompts.
                        evidence = []
                        depth_fn = getattr(self.world, "depth_at_pixel", None)
                        if action.pixel is not None and callable(depth_fn):
                            obs_d = depth_fn(action.pixel)
                            if obs_d is not None:
                                evidence.append(
                                    f"the point you marked reads about {obs_d:.1f} m "
                                    "on the depth estimate"
                                )
                        if action.pixel is None:
                            evidence.append(
                                'report "done" WITH the target\'s "pixel" location '
                                "so the claim can be verified"
                            )
                        evidence.append("keep approaching, or reply scan to look around again")
                        outcome = "done rejected: not verifiably close"
                        records[-1].outcome = outcome
                        history[-1]["outcome"] = outcome
                        last_feedback = (
                            "you replied done, but I cannot verify that the target is "
                            "close and correctly identified. " + ". ".join(evidence) + "."
                        )
                        # narrow re-acquisition scan around the current heading
                        self._look_around(history, tag=f"{step}-reacquire",
                                          n_views=5, spread_deg=120.0)
                        continue
                    finish_reason = "done_but_too_far"
                else:
                    success, finish_reason = True, "vlm_done"
                break
            if (
                self.rescan_after
                and action.kind != "scan"
                and step - last_progress_step >= self.rescan_after
                and step < self.max_steps
            ):
                cell = (int(p[0] // 2), int(p[1] // 2))
                if len(sectors.get(cell, set())) >= 7:
                    # fully scanned here: move to a new viewpoint instead of
                    # scanning in place again (goal.md P2)
                    self._explore_step(history, step, p, sectors)
                else:
                    self._look_around(history, tag=f"{step}-auto")
                last_progress_step = step

        trajectory = np.asarray(getattr(self.world, "trajectory", []), dtype=np.float64)
        path_length = float(
            np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum()
        ) if len(trajectory) > 1 else 0.0
        summary = EpisodeSummary(
            instruction=self.instruction,
            target_name=getattr(self.world.target, "name", "target"),
            success=success,
            finish_reason=finish_reason,
            steps=step,
            final_distance_m=self.world.distance_to_target(),
            min_distance_m=min_dist,
            path_length_m=path_length,
            wall_time_s=time.time() - start,
            parse_failures=parse_failures,
            vlm_calls=self._vlm_calls,
            collision_hits=int(getattr(self.world, "collision_hits", 0)),
            records=[asdict(r) for r in records],
            first_visible_step=first_visible_step,
            visible_steps=visible_steps,
            max_visible_streak=max_visible_streak,
            initial_target_visible=initial_target_visible,
            unlocalized_sightings=self.unlocalized_sightings,
        )
        self._write_artifacts(summary)
        return summary

    def _write_artifacts(self, summary: EpisodeSummary) -> None:
        with open(self.out_dir / "episode_trace.jsonl", "w") as handle:
            for record in summary.records:
                handle.write(json.dumps(record) + "\n")
        serializable = asdict(summary)
        serializable.pop("records")
        with open(self.out_dir / "summary.json", "w") as handle:
            json.dump(serializable, handle, indent=2)
        trajectory = np.asarray(getattr(self.world, "trajectory", []), dtype=np.float64)
        np.save(self.out_dir / "trajectory.npy", trajectory)
        if self.make_video and self.frames_dir.exists():
            try:
                import imageio.v2 as imageio

                frames = sorted(self.frames_dir.glob("step_*.png"))
                if frames:
                    with imageio.get_writer(
                        self.out_dir / "episode.mp4", fps=2
                    ) as writer:
                        for path in frames:
                            writer.append_data(imageio.imread(path))
            except Exception as exc:  # video is best-effort only
                print(f"[harness] video export skipped: {exc}")
