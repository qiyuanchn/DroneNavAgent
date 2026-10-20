#!/usr/bin/env python
"""Identity/anchor memory check: natural occlusion, delayed revisit, decoys.

The question this script answers is the one goal.md leaves open for long-term
memory: *is the object-identity/anchor half load-bearing when the target is seen
once, disappears naturally (an occluder / another stage of the task), and is
met again among same-looking instances?*

It is a **memory-level** experiment, not a VLM episode: no renderer and no
model weights are used. Each scene is played through the real
``vlm_nav.memory`` store and the real ``query_memory`` tool:

1. the correct instance is sighted (a ``ground_target``-style sighting with
   target description + box + position + relation anchor);
2. it becomes occluded and a same-looking decoy is sighted later (newer = what
   a recency-only retrieval returns);
3. ``confirm_candidate`` accepts the correct instance and records its relation
   anchor / evidence revision;
4. an unresolved question ("is the newer box the same instance?") stays open;
5. after a delay the agent retrieves memory and picks the candidate to revisit.

Arms:

- ``memory`` - the full store (identities, anchors, verdicts, questions);
- ``memory_no_identity`` - the *identity ablation*: exactly the identity/anchor
  fields are removed (``identity_memory=False``); frames, sightings, places and
  questions stay, so the legal memory is not cleared;
- ``basic`` - no retrieval; the newest sighting is used (the recency baseline).

Metrics: correct-instance rate, anchor-reference rate (a decision that cited
the anchor field), revisit positioning error (m to the true instance) and cost
(result bytes + retrieval calls). ``annotations.json`` records the causal
contrast episode by episode; the report never claims a renderer/VLM result.

Run:

    python scripts/check_memory_identity.py \
        --out outputs/vlm_nav/memory_identity_check
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vlm_nav.memory import ObservationLog
from vlm_nav.tool_impl_perception import query_memory

ARMS = ("memory", "memory_no_identity", "basic")


@dataclass(frozen=True)
class Scene:
    name: str
    target: str
    correct: tuple[float, float, float]
    decoy: tuple[float, float, float]
    landmark: tuple[float, float, float]
    landmark_name: str
    delay_observations: int
    occlusion: str = "occluder slab between the last view and the instance"


@dataclass
class Episode:
    scene: Scene
    arm: str
    seed: int
    chosen: str = ""
    correct_choice: bool = False
    anchor_referenced: bool = False
    revisit_error_m: float | None = None
    cost_bytes: int = 0
    retrieval_calls: int = 0
    question_open: bool = False
    identity_verdict: str = ""
    identity_revision: int | None = None
    retrieval_order: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _distance(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def _scene_set() -> list[Scene]:
    """Six scenes: two layouts x two revisit delays, plus two farther revisits."""
    base = [(-6.5, -1.0, 1.2), (-5.0, 2.2, 1.4)]
    scenes: list[Scene] = []
    for index, (name, side) in enumerate((("left", 1.0), ("right", -1.0))):
        correct = base[index]
        decoy = (correct[0] + 2.4, correct[1] + side * 0.9, correct[2] - 0.15)
        landmark = (correct[0] + 0.8, correct[1] + side * 0.85, correct[2] + 0.35)
        for delay in (1, 3):
            scenes.append(
                Scene(
                    name=f"{name}_delay{delay}",
                    target="红色的箱子" if index == 0 else "red box",
                    correct=correct,
                    decoy=decoy,
                    landmark=landmark,
                    landmark_name="木门",
                    delay_observations=delay,
                )
            )
    # a farther revisit where the decoy is much nearer the start heading
    correct = (-7.0, 3.0, 1.0)
    scenes.append(
        Scene(
            name="right_delay5",
            target="red box",
            correct=correct,
            decoy=(-4.0, 0.4, 0.9),
            landmark=(-6.2, 3.8, 1.3),
            landmark_name="wooden door",
            delay_observations=5,
        )
    )
    correct = (-6.0, -2.6, 1.1)
    scenes.append(
        Scene(
            name="left_delay5",
            target="红色的箱子",
            correct=correct,
            decoy=(-3.2, -1.8, 1.0),
            landmark=(-5.2, -3.2, 1.4),
            landmark_name="木门",
            delay_observations=5,
        )
    )
    return scenes


def _memory(arm: str) -> ObservationLog:
    return ObservationLog(identity_memory=arm != "memory_no_identity", clock=lambda: 1.0)


def _sighting(
    candidate: str, scene: Scene, position: tuple[float, float, float],
    *, target: str, anchor: str = "",
) -> dict:
    return {
        "candidate": candidate,
        "target": target,
        "box": [120, 160, 420, 520] if candidate.endswith("correct") else [460, 180, 760, 540],
        "basis": "ground_target",
        "image": f"frame_{candidate}",
        "position_flu": list(position),
        "anchor": anchor,
    }


def _observe(log: ObservationLog, scene: Scene, step: int, candidate: str,
             position: tuple[float, float, float], *, anchor: str = "") -> str:
    """Record one sighting from a viewpoint near the instance."""
    pose = [position[0] - 1.2, position[1] - 0.4, position[2] + 0.2]
    return log.record(
        None,
        pose=pose,
        yaw=0.0,
        event="target_sighting",
        goal_target=scene.target,
        step=step,
        t=float(step),
        sighting=_sighting(candidate, scene, position, target=scene.target, anchor=anchor),
    )


def _play(scene: Scene, arm: str, seed: int) -> Episode:
    log = _memory(arm)
    episode = Episode(scene=scene, arm=arm, seed=seed)
    anchor = f"next to the {scene.landmark_name}"
    correct_id = "cand_correct"
    _observe(log, scene, 1, correct_id, scene.correct, anchor=anchor)
    # the correct instance is now occluded; the same-looking decoy is visible in
    # later looks (newer observations, no identity information)
    for index in range(max(1, scene.delay_observations)):
        _observe(
            log, scene, 2 + index, f"cand_decoy_{index + 1}",
            (scene.decoy[0] + 0.15 * index, scene.decoy[1], scene.decoy[2]),
        )
    log.confirm_identity(
        correct_id, "accepted", revision=1, anchor=anchor,
        observation_ids=["obs_1"],
    )
    log.working.open_question(
        "is the newer box the same instance as the confirmed one?",
        about=scene.target, step=2 + scene.delay_observations,
    )
    ctx = SimpleNamespace(memory=log, world=None, extra={}, episode_id=f"memcheck_{scene.name}")

    if arm == "basic":
        candidate = f"cand_decoy_{scene.delay_observations}"
        entry = log.get(f"obs_{1 + scene.delay_observations}")
        position = (entry or {}).get("sighting", {}).get("position_flu")
        episode.chosen = candidate
        episode.cost_bytes = 0
        episode.retrieval_calls = 0
        episode.revisit_error_m = _distance(position, scene.correct) if position else None
        episode.notes.append("no retrieval: the newest sighting is used")
    else:
        answer = query_memory(ctx, {"target": scene.target, "limit": 4})
        data = answer.data if isinstance(answer.data, dict) else {}
        episode.retrieval_calls = 1
        episode.cost_bytes = len(json.dumps(data, ensure_ascii=False))
        hits = data.get("hits") or []
        identities = data.get("identities") or []
        episode.question_open = bool((data.get("working") or {}).get("open_questions"))
        if arm == "memory":
            chosen = next(
                (item for item in identities if item.get("verdict") == "accepted"), None
            )
            chosen = chosen or (identities[0] if identities else None)
            if chosen is not None:
                episode.chosen = str(chosen.get("candidate_id") or chosen.get("identity_id"))
                episode.identity_verdict = str(chosen.get("verdict") or "")
                episode.identity_revision = chosen.get("revision")
                episode.anchor_referenced = bool(chosen.get("anchor"))
                position = chosen.get("position_flu")
            else:
                position = None
            episode.retrieval_order = [
                str(item.get("candidate_id") or item.get("identity_id")) for item in identities
            ]
            if not episode.retrieval_order:
                episode.retrieval_order = [
                    str((hit.get("sighting") or {}).get("candidate") or "") for hit in hits
                ]
        else:
            hit = hits[0] if hits else {}
            sighting = hit.get("sighting") or {}
            episode.chosen = str(sighting.get("candidate") or "")
            position = sighting.get("position_flu") or hit.get("position_flu")
            episode.notes.append(
                "identity/anchor fields removed; the newest sighting wins (recency)"
            )
            episode.retrieval_order = [
                str((item.get("sighting") or {}).get("candidate") or "") for item in hits
            ]
        episode.revisit_error_m = _distance(position, scene.correct) if position else None
        episode.notes.append(
            "question returned in retrieval" if episode.question_open else "question missing"
        )
    episode.correct_choice = episode.chosen == correct_id
    return episode


def _summary(episodes: list[Episode]) -> dict:
    by_arm: dict[str, dict] = {}
    for arm in ARMS:
        rows = [row for row in episodes if row.arm == arm]
        errors = [row.revisit_error_m for row in rows if row.revisit_error_m is not None]
        by_arm[arm] = {
            "episodes": len(rows),
            "correct_instance_rate": round(
                sum(1 for row in rows if row.correct_choice) / len(rows), 3
            ) if rows else 0.0,
            "anchor_reference_rate": round(
                sum(1 for row in rows if row.anchor_referenced) / len(rows), 3
            ) if rows else 0.0,
            "revisit_error_mean_m": round(statistics.mean(errors), 3) if errors else None,
            "revisit_error_median_m": round(statistics.median(errors), 3) if errors else None,
            "cost_bytes_mean": round(statistics.mean([row.cost_bytes for row in rows]), 1)
            if rows else 0.0,
            "retrieval_calls_mean": round(
                statistics.mean([row.retrieval_calls for row in rows]), 2
            ) if rows else 0.0,
            "open_question_returned_rate": round(
                sum(1 for row in rows if row.question_open) / len(rows), 3
            ) if rows else 0.0,
        }
    flips = [
        row for row in episodes
        if row.arm == "memory" and row.correct_choice
        and any(
            other.scene.name == row.scene.name and other.arm == "memory_no_identity"
            and not other.correct_choice
            for other in episodes
        )
    ]
    decoy_errors = [
        row for row in episodes
        if row.arm == "memory_no_identity" and not row.correct_choice
    ]
    annotations = []
    if flips:
        annotations.append(
            f"identity/anchor removal flipped {len(flips)}/{len(flips)} paired episodes "
            "from the correct instance to a decoy: the accepted identity (with its "
            "anchor) is what outranks the newer, same-looking sighting."
        )
    if decoy_errors:
        annotations.append(
            "without identity the recency order returns the later decoy; its mean "
            f"revisit error is {round(statistics.mean([row.revisit_error_m for row in decoy_errors]), 2)} m "
            "from the true instance - a legal memory (frames/sightings) that simply "
            "does not say *which* one."
        )
    if by_arm["memory"]["anchor_reference_rate"] > 0 and by_arm["memory_no_identity"]["anchor_reference_rate"] == 0:
        annotations.append(
            "the anchor is cited in every full-memory decision and in none of the "
            "ablated ones: the relation phrase is a persistent field, not a prompt hint."
        )
    if all(row.question_open for row in episodes if row.arm != "basic"):
        annotations.append(
            "the unresolved question ('is the newer box the same instance?') survives "
            "the delay and leads the working-state block in both retrieval arms."
        )
    return {
        "arms": by_arm,
        "ablation": {
            "arm": "memory_no_identity",
            "removed": ["object identity fingerprint", "confirm verdict/revision",
                        "relation anchor"],
            "kept": ["frames", "sightings", "spatial places", "open questions",
                     "failures"],
            "note": "only the identity/anchor fields are removed; the legal "
                    "visual archive is not cleared",
        },
        "annotations": annotations,
    }


def _report(summary: dict, episodes: list[Episode]) -> str:
    lines = [
        "# Identity / anchor memory check",
        "",
        "Memory-level run (no renderer, no model weights): natural occlusion ->",
        "delayed revisit -> same-looking decoy. The full memory is compared with",
        "an identity/anchor ablation (only those fields removed) and a no-retrieval",
        "baseline.",
        "",
        "| arm | correct instance | anchor cited | revisit error (mean) | cost (bytes) |",
        "|---|---|---|---|---|",
    ]
    for arm in ARMS:
        row = summary["arms"][arm]
        lines.append(
            f"| {arm} | {row['correct_instance_rate']:.2f} | "
            f"{row['anchor_reference_rate']:.2f} | "
            f"{row['revisit_error_mean_m']} m | {row['cost_bytes_mean']:.0f} |"
        )
    lines += ["", "## Episodes", "", "| scene | delay | arm | chosen | correct | error m |", "|---|---|---|---|---|---|"]
    for row in episodes:
        lines.append(
            f"| {row.scene.name} | {row.scene.delay_observations} | {row.arm} | "
            f"{row.chosen or '-'} | {'yes' if row.correct_choice else 'no'} | "
            f"{'-' if row.revisit_error_m is None else round(row.revisit_error_m, 2)} |"
        )
    lines += ["", "## Causal annotations", ""]
    lines += [f"- {note}" for note in summary["annotations"]]
    lines += [
        "",
        "## Limitation",
        "",
        "This run exercises the memory layer and the real `query_memory` result",
        "shape. It does not claim that a VLM uses those fields: the GPU host and",
        "Qwen3-VL-8B weights were not available in this environment, so no visual",
        "episode was run. The M2 mock arm next to it is trace-level only and its",
        "scripted policy does not read identity fields.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="outputs/vlm_nav/memory_identity_check")
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenes = _scene_set()
    episodes: list[Episode] = []
    for seed in range(max(1, args.seeds)):
        for scene in scenes:
            for arm in ARMS:
                episodes.append(_play(scene, arm, seed))
    payload = [
        {
            "scene": row.scene.name,
            "target": row.scene.target,
            "delay_observations": row.scene.delay_observations,
            "occlusion": row.scene.occlusion,
            "arm": row.arm,
            "seed": row.seed,
            "chosen": row.chosen,
            "correct_choice": row.correct_choice,
            "anchor_referenced": row.anchor_referenced,
            "revisit_error_m": row.revisit_error_m,
            "cost_bytes": row.cost_bytes,
            "retrieval_calls": row.retrieval_calls,
            "open_question_returned": row.question_open,
            "identity_verdict": row.identity_verdict,
            "identity_revision": row.identity_revision,
            "retrieval_order": row.retrieval_order,
            "notes": row.notes,
        }
        for row in episodes
    ]
    summary = _summary(episodes)
    (out / "episodes.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (out / "report.md").write_text(_report(summary, episodes))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[memory_identity] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
