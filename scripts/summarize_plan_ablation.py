#!/usr/bin/env python
"""Compare the given-plan and self-plan arms of the goal.md section-4 ablation.

Reads the four driver runs (``<root>/{given,self}/{bedroom,classroom}/report.json``)
and writes ``<root>/summary.json`` and ``<root>/summary.md``. Every number is
taken from the driver's own report: stage verdicts, wrong-instance acceptance,
wrong completion, cost (outer + inner tool calls), the plan-quality audit and
the reach-attempt counts by typed done_when criterion. Nothing is recomputed
from the truth places here.

Usage:
    python scripts/summarize_plan_ablation.py --root outputs/vlm_nav/verify_long_horizon_plan_ablation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


ARMS = ("given", "self")
TASKS = ("bedroom", "classroom")


def _load(root: Path, arm: str, task: str) -> dict | None:
    directory = root / arm / task
    for name in ("report_reanalyzed.json", "report.json"):
        path = directory / name
        if path.exists():
            with open(path) as handle:
                return json.load(handle)
    return None


def _numbers(agg: Mapping[str, Any]) -> dict:
    quality = agg.get("plan_quality") or {}
    cost = agg.get("cost") or {}
    return {
        "n_episodes": agg.get("n_episodes"),
        "chain_complete": (agg.get("complete_ordered_task_success") or {}).get("count"),
        "ordered_success_rate": agg.get("ordered_success_rate"),
        "stage_complete_rate": agg.get("stage_complete_rate"),
        "stage_complete_by_type": agg.get("stage_complete_by_type") or {},
        "reach_attempts": agg.get("reach_attempts"),
        "wrong_instance_rate": (agg.get("wrong_instance_acceptance") or {}).get("rate"),
        "wrong_instance_wrong": (agg.get("wrong_instance_acceptance") or {}).get("wrong"),
        "wrong_instance_resolved": (agg.get("wrong_instance_acceptance") or {}).get("resolved"),
        "wrong_completion": (agg.get("wrong_completion") or {}).get("wrong"),
        "wrong_completion_claims": (agg.get("wrong_completion") or {}).get("episodes_claiming"),
        "plan_count_match": quality.get("count_match"),
        "plan_order_ok": quality.get("order_ok"),
        "plan_mean_coverage": quality.get("mean_coverage"),
        "plan_type_accuracy": quality.get("done_when_type_accuracy"),
        "plan_structure_ok": quality.get("structure_ok"),
        "plans": quality.get("plans"),
        "wall_time_s": cost.get("wall_time_s"),
        "model_calls": cost.get("model_calls"),
        "tool_calls": cost.get("tool_calls"),
        "outer_tool_calls": cost.get("outer_tool_calls"),
        "inner_tool_calls": cost.get("inner_tool_calls"),
        "executor_legs": cost.get("executor_legs"),
    }


def _type_denominator(row: Mapping[str, Any], kind: str) -> int:
    """Stage-episode pairs of one type (the driver reports distinct stages)."""
    by_type = row.get("stage_complete_by_type") or {}
    stages = int((by_type.get(kind) or {}).get("stages") or 0)
    return stages * int(row.get("n_episodes") or 0)


def _causal_annotations(rows: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[str]:
    """At least two mechanism-level statements, each anchored on a report field."""
    lines: list[str] = []
    for task in TASKS:
        given = rows["given"].get(task)
        self_arm = rows["self"].get(task)
        if not given or not self_arm:
            continue
        failed_first_turn = [
            episode.get("seed")
            for episode in self_arm.get("episodes") or []
            if str(episode.get("success_finish_reason") or "") == "unparseable_output"
            and int((episode.get("plan_quality") or {}).get("n_subtasks") or 0) == 0
        ]
        if failed_first_turn:
            lines.append(
                f"{task}: the self arm's first-turn update_plan reply was unparseable "
                f"for seeds {failed_first_turn} (raw reply begins with <tool_call> and "
                f"the 4-stage plan exceeds the 192-token generation cap), so no plan "
                f"was accepted and no tool ran; this cell measures output-format "
                f"capacity, not planning quality."
            )
        given_nums = _numbers(given["aggregate"])
        self_nums = _numbers(self_arm["aggregate"])
        # 1. how the model decomposed the mission (self arm only)
        plans = [
            (episode.get("plan_quality") or {})
            for episode in self_arm.get("episodes") or []
        ]
        available = [plan for plan in plans if plan.get("available")]
        if available:
            rows_text = "; ".join(
                f"seed {episode.get('seed')}: {plan.get('n_subtasks')} sub-tasks, "
                f"types {[row.get('done_when_type') for row in plan.get('rows') or []]}, "
                f"order {plan.get('order_indices')}"
                for episode, plan in zip(self_arm["episodes"], plans)
                if plan.get("available")
            )
            lines.append(
                f"{task}: the self-plan model wrote {len(available)}/"
                f"{len(plans)} valid plans; stage count matched "
                f"{self_nums['plan_count_match']}/{self_nums['plans']}, order matched "
                f"{self_nums['plan_order_ok']}/{self_nums['plans']}, mean coverage "
                f"{self_nums['plan_mean_coverage']}, done_when type accuracy "
                f"{self_nums['plan_type_accuracy']}. Decomposition: {rows_text}."
            )
        # 2. did the semantic criterion save motion?
        sighted_self = (self_nums["stage_complete_by_type"] or {}).get("sighted") or {}
        arrived_self = (self_nums["stage_complete_by_type"] or {}).get("arrived") or {}
        sighted_given = (given_nums["stage_complete_by_type"] or {}).get("sighted") or {}
        arrived_given = (given_nums["stage_complete_by_type"] or {}).get("arrived") or {}
        lines.append(
            f"{task}: sighted stages made {sighted_self.get('reach_attempts', 0)} reach "
            f"attempt(s) in the self arm vs {sighted_given.get('reach_attempts', 0)} in "
            f"the given arm; arrived stages made {arrived_self.get('reach_attempts', 0)} "
            f"(self) vs {arrived_given.get('reach_attempts', 0)} (given). Sighted stages "
            f"completed {sighted_self.get('complete', 0)}/"
            f"{_type_denominator(self_nums, 'sighted')} (self) and "
            f"{sighted_given.get('complete', 0)}/"
            f"{_type_denominator(given_nums, 'sighted')} (given); arrived stages "
            f"{arrived_self.get('complete', 0)}/"
            f"{_type_denominator(self_nums, 'arrived')} (self) and "
            f"{arrived_given.get('complete', 0)}/"
            f"{_type_denominator(given_nums, 'arrived')} (given)."
        )
        # 3. cost and safety together, since a cheaper arm that lies is not better
        lines.append(
            f"{task}: cost {self_nums['tool_calls']} tool calls ("
            f"{self_nums['inner_tool_calls']} inner) / {self_nums['model_calls']} model "
            f"calls (self) vs {given_nums['tool_calls']} ("
            f"{given_nums['inner_tool_calls']} inner) / {given_nums['model_calls']} "
            f"(given); wrong-instance acceptance {self_nums['wrong_instance_rate']} vs "
            f"{given_nums['wrong_instance_rate']}; wrong completions "
            f"{self_nums['wrong_completion']} vs {given_nums['wrong_completion']}."
        )
    return lines


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, default="outputs/vlm_nav/verify_long_horizon_plan_ablation")
    args = parser.parse_args(argv)
    root = Path(args.root)
    rows: dict[str, dict[str, dict]] = {arm: {} for arm in ARMS}
    missing: list[str] = []
    for arm in ARMS:
        for task in TASKS:
            report = _load(root, arm, task)
            if report is None:
                missing.append(f"{arm}/{task}")
                continue
            rows[arm][task] = report
    summary: dict[str, Any] = {
        "root": str(root),
        "missing": missing,
        "arms": {},
        "episodes": [],
    }
    for arm in ARMS:
        for task in TASKS:
            report = rows[arm].get(task)
            if report is None:
                continue
            numbers = _numbers(report.get("aggregate") or {})
            summary["arms"][f"{arm}/{task}"] = {
                "plan_mode": (report.get("meta") or {}).get("plan_mode"),
                **numbers,
            }
            for episode in report.get("episodes") or []:
                quality = episode.get("plan_quality") or {}
                legs = episode.get("legs") or [{}]
                leg = legs[0]
                outer = int(leg.get("outer_tool_calls") or 0)
                inner = int(leg.get("inner_tool_calls") or 0)
                summary["episodes"].append(
                    {
                        "arm": arm,
                        "task": task,
                        "seed": episode.get("seed"),
                        "stages": episode.get("stages", {}).get("complete_count"),
                        "n_stages": episode.get("stages", {}).get("n_stages"),
                        "chain_complete": episode.get("stages", {}).get("chain_complete"),
                        "ordered_success": episode.get("ordered_success"),
                        "success": episode.get("success"),
                        "traceable_success": episode.get("traceable_success"),
                        "wrong_instance": (episode.get("instance_acceptance") or {}).get("wrong"),
                        "wrong_completion": (episode.get("completion_claim") or {}).get("wrong"),
                        "tool_calls": outer + inner,
                        "inner_tool_calls": inner,
                        "model_calls": int(leg.get("model_calls") or 0),
                        "wall_time_s": leg.get("wall_time_s"),
                        "finish_reason": episode.get("success_finish_reason"),
                        "plan_subtasks": quality.get("n_subtasks"),
                        "plan_count_match": quality.get("count_match"),
                        "plan_order_ok": quality.get("order_ok"),
                        "plan_coverage": quality.get("coverage_rate"),
                        "plan_type_accuracy": quality.get("done_when_type_accuracy"),
                        "plan_structure_ok": quality.get("structure_ok"),
                        "reach_attempts_by_type": (
                            episode.get("stages", {}).get("reach_attempts_by_type")
                        ),
                    }
                )
    summary["causal_annotations"] = _causal_annotations(rows)
    with open(root / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)

    lines = [
        "# Given plan vs self plan (goal.md section 4)",
        "",
        "| arm/task | episodes | ordered complete | stage complete | sighted complete | arrived complete | reach attempts (sighted/arrived) | wrong inst. | wrong completion | tool calls (inner) | model calls | plan count/order/coverage/type |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, row in summary["arms"].items():
        by_type = row.get("stage_complete_by_type") or {}
        sighted = by_type.get("sighted") or {}
        arrived = by_type.get("arrived") or {}
        lines.append(
            f"| {key} | {row['n_episodes']} | {row['chain_complete']} | "
            f"{row['stage_complete_rate']} | {sighted.get('complete', 0)}/"
            f"{_type_denominator(row, 'sighted')} | {arrived.get('complete', 0)}/"
            f"{_type_denominator(row, 'arrived')} | "
            f"{sighted.get('reach_attempts', 0)}/{arrived.get('reach_attempts', 0)} | "
            f"{row['wrong_instance_rate']} | {row['wrong_completion']} | "
            f"{row['tool_calls']} ({row['inner_tool_calls']}) | {row['model_calls']} | "
            f"{row['plan_count_match']}/{row['plan_order_ok']}/"
            f"{row['plan_mean_coverage']}/{row['plan_type_accuracy']} |"
        )
    lines += [
        "",
        "## per episode",
        "",
        "| arm | task | seed | stages | ordered | success | finish | wrong inst. | wrong completion | tools (inner) | model calls | plan n / count / order / coverage / type |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary["episodes"]:
        lines.append(
            f"| {row['arm']} | {row['task']} | {row['seed']} | {row['stages']}/"
            f"{row['n_stages']} | {row['ordered_success']} | {row['success']} | "
            f"{row.get('finish_reason') or '-'} | "
            f"{row['wrong_instance']} | {row['wrong_completion']} | "
            f"{row['tool_calls']} ({row['inner_tool_calls']}) | {row['model_calls']} | "
            f"{row['plan_subtasks']} / {row['plan_count_match']} / "
            f"{row['plan_order_ok']} / {row['plan_coverage']} / "
            f"{row['plan_type_accuracy']} |"
        )
    lines += ["", "## causal annotations", ""]
    lines += [f"- {line}" for line in summary["causal_annotations"]]
    if missing:
        lines += ["", f"missing runs: {', '.join(missing)}"]
    with open(root / "summary.md", "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return summary


if __name__ == "__main__":
    main()
