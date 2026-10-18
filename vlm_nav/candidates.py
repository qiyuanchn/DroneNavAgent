"""Candidate target memory for the VLM navigation harness (goal.md P1).

The VLM's per-frame "I see the target" claims are fused into persistent
candidates with estimated world positions, so a one-frame hallucination does
not become a multi-step approach, and the same object seen from different
viewpoints stays one instance.

A candidate is created whenever the model points at the target (a waypoint or
done action carrying a pixel). Sightings within ``merge_radius_m`` of an
existing candidate merge into it (multi-view consistency); a candidate becomes
"confirmed" after ``confirm_sightings`` merged sightings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class Candidate:
    id: int
    position_flu: tuple[float, float, float]
    sightings: int
    first_seen_step: int
    last_seen_step: int
    status: str = "active"  # active | confirmed | rejected
    notes: str = field(default="", repr=False)


class CandidateTracker:
    def __init__(self, merge_radius_m: float = 1.0, confirm_sightings: int = 2):
        self.merge_radius_m = float(merge_radius_m)
        self.confirm_sightings = int(confirm_sightings)
        self.candidates: list[Candidate] = []
        self._next_id = 1

    def observe(self, position_flu, step: int, note: str = "") -> Candidate:
        """Fuse a new position estimate into the nearest active/confirmed
        candidate within merge radius, else spawn a new one."""
        pos = np.asarray(position_flu, dtype=np.float64)
        best: Candidate | None = None
        best_d = float("inf")
        for cand in self.candidates:
            if cand.status == "rejected":
                continue
            d = float(np.linalg.norm(np.asarray(cand.position_flu) - pos))
            if d < best_d:
                best, best_d = cand, d
        if best is not None and best_d <= self.merge_radius_m:
            # exponential merge: later sightings refine but do not jump
            alpha = 0.5
            merged = (1.0 - alpha) * np.asarray(best.position_flu) + alpha * pos
            best.position_flu = tuple(float(v) for v in merged)
            best.sightings += 1
            best.last_seen_step = step
            if best.sightings >= self.confirm_sightings and best.status == "active":
                best.status = "confirmed"
            if note:
                best.notes = note[:200]
            return best
        cand = Candidate(
            id=self._next_id,
            position_flu=tuple(float(v) for v in pos),
            sightings=1,
            first_seen_step=step,
            last_seen_step=step,
            notes=note[:200],
        )
        self._next_id += 1
        self.candidates.append(cand)
        return cand

    def reject(self, candidate_id: int, note: str = "") -> None:
        for cand in self.candidates:
            if cand.id == candidate_id:
                cand.status = "rejected"
                if note:
                    cand.notes = note[:200]

    def best(self) -> Candidate | None:
        """Most recently seen non-rejected candidate."""
        live = [c for c in self.candidates if c.status != "rejected"]
        if not live:
            return None
        return max(live, key=lambda c: (c.status == "confirmed", c.last_seen_step))

    def describe(self, position_flu, yaw_rad: float, limit: int = 3) -> str:
        """Compact text for the prompt: candidates relative to the drone."""
        live = [c for c in self.candidates if c.status != "rejected"]
        if not live:
            return "none yet"
        pos = np.asarray(position_flu, dtype=np.float64)
        lines = []
        for cand in sorted(live, key=lambda c: -c.last_seen_step)[:limit]:
            rel = np.asarray(cand.position_flu) - pos
            dist = float(np.linalg.norm(rel))
            bearing = math.degrees(math.atan2(rel[1], rel[0]) - yaw_rad)
            bearing = (bearing + 180.0) % 360.0 - 180.0
            lines.append(
                f"#{cand.id} {cand.status}, {dist:.1f} m at {bearing:+.0f} deg "
                f"(seen {cand.sightings}x, last step {cand.last_seen_step})"
            )
        return "; ".join(lines)
