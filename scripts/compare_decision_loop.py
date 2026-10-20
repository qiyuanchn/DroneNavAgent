#!/usr/bin/env python
"""Decision-loop comparison driver: baseline vs. vision context vs. recovery vs. memory.

The harness-upgrade round asks which *decision-loop* layer pays off, holding the
model, scenes, start poses/headings and budgets fixed:

  1. ``baseline``             the pre-decision-loop tool loop: a small per-turn
                              visual context, no STATE block, no stall recovery,
                              no search skill, no memory layer (each mechanism
                              explicitly switched off)
  2. ``vision_context``       + richer visual context (more frames per decision)
                              and the multi-view search skill (``search_target``)
  3. ``recovery``             + the fixed progress STATE block and the
                              progress-aware failure recovery (strategy change
                              before termination)
  4. ``memory``               + the task-state memory: the *spatial* store (what
                              the task is looking for, where it was seen) and the
                              *episodic* store (which view saw what) - the two
                              constructor parameters of the memory module
  5. ``memory_spatial_only``  the same memory arm with the episodic store off:
                              the paired ``memory -> memory_spatial_only`` delta
                              is what the episodic store is worth

The memory layer is a layer like the others: every arm declares all five
mechanism switches (``--recovery``, ``--state-block``, ``--search-skill`` and
the memory pairs ``--memory-spatial`` / ``--memory-episodic``). The runner's
default is ``on``, so the arms *before* the memory layer say ``off`` - otherwise
``recovery -> memory`` would compare two identical configurations. The memory
arms keep the shared catalogue (``full``), so their paired delta is the memory
layer and nothing else: the goal.md M2 catalogue (``--toolset memory``) has no
``search_target`` and no M3 tools, which would mix the loss of the search skill
into the memory delta; ``--arm-arg memory=--toolset memory --context-frames 4
--search-skill on --state-block on --recovery on --memory-spatial on
--memory-episodic on`` runs that variant instead (the report then says which
catalogue each arm really ran in).

The arm -> runner switch mapping is data (``ARM_TABLE``) and can be overridden
per arm on the command line (``--arm-arg recovery=--recovery on``). Before
running, the driver reads the tool runner's argparse source, keeps only the
switches that runner actually accepts and records the result as ``arm_support``
per arm (``full`` / ``degraded``, with the dropped switches); a
``--strict-arms`` run refuses to start instead of degrading. That keeps the
comparison honest when a switch is renamed: the report says which arm really got
which mechanism. Next to the declared switches the report carries what the
runner summaries say was really applied per arm (``meta.mechanism_check``),
including the toolset the arm ran in and the memory layers the built memory
attested (a layer the memory does not confirm is reported as ``?``, unknown -
never as a mechanism that ran).

Every cell is the same scene, target, start pose, initial heading, seed and VLM
budget across arms; only the arm's runner switches differ. Two scenario presets
come from the Phase-1 rounds - ``book`` (HiGS, /scene_book.ply) and ``synth``
(the fair-gate synthetic scene) - and ``cpu`` takes the scene/tasks from the
command line. Scenarios, seeds and initial-yaw groups (``visible``/``random``)
multiply into the episode matrix.

Mock run (CPU world, no weights):

    python scripts/compare_decision_loop.py --scenario cpu --world cpu \
        --ply scene.ply --target-pos 2 0.5 0 --instruction "Find the target." \
        --backend mock --seeds 7 --initial-yaw-modes visible,random \
        --max-vlm-calls 6 --jobs 3 --out outputs/vlm_nav/decision_loop_mock

Real model run (the coordinator runs this in the container):

    python scripts/compare_decision_loop.py --scenario book --scenario synth \
        --backend hf --model-path /models_ro/Qwen3-VL-8B --device cuda \
        --seeds 7,11 --initial-yaw-modes visible,random --max-vlm-calls 20 \
        --out outputs/vlm_nav/decision_loop_qwen

Metrics (per episode, aggregated per arm x group, plus paired layer deltas):

  solved                the runner's own verdict (a terminating finish) AND the
                        last traced pose is inside the geometric gate
                        (``--success-dist + radius``). What the docstring means
                        is what ``verify_phase1.success_stats`` does: *both* are
                        required. Reaching the gate without a terminating finish
                        (``reached: true``, ``solved: false`` - the drone stops
                        inside the gate and the episode ends on a limit, not on
                        a ``finish``) is **not** solved; a ``finish`` outside the
                        gate is not either. ``reached`` and ``final_distance_m``
                        report the geometry on their own, ``success`` alone is
                        the runner's verdict, ``claim`` is what the agent
                        *declared*.
  success               the runner's own verdict (``summary.json``).
  claim                 an *explicit* declaration of completion: a ``finish``
                        turn whose args say ``success: true`` (or a verifying
                        finish tool that answers so), or a ``verify_goal``
                        answered ``verified: true``. A bare finish - a reason
                        and nothing else - claims nothing: the runtime leaves
                        the verdict open for it, so the honest "I could not find
                        it" of a give-up is no longer counted as a claim
                        (``finish_declared`` / ``honest_finish`` keep it apart).
  false_confirmation    the agent claimed success/confirmation while the
                        geometric reach says otherwise; ``false_confirmation_rate``
                        is the share of *claims* that were false (honest
                        give-ups are in neither the numerator nor the
                        denominator). A grounded candidate whose reported
                        position is beyond the gate is also a false
                        confirmation (``candidate_wrong``).
  verify gate           ``verify_limit_m`` is the distance limit the episode's
                        ``verify_goal`` calls recorded, ``verify_gate_consistent``
                        whether that limit (and the gate the runner reports it
                        handed to the tools, ``summary.done_dist_m``) equals the
                        evaluation gate. A claim whose check used another limit
                        is a threshold artifact, not evidence: the v3 book_near
                        claims came from ``verify_goal``'s own 1.5 m default
                        against a 1.2 m gate, and ``meta.verify_gate`` counts
                        those episodes per run.
  prompt identity       ``prompt_episode_id`` is the identity the model saw in
                        its prompt (the runner reports it; the arm name never
                        reaches the prompt). ``meta.prompt_identity_check``
                        requires it to be present, arm-free and equal across the
                        arms of one cell - otherwise the arms differed by a
                        prompt constant that has nothing to do with the layer
                        under test.
  repeat_search_ratio   repeat checks / information-gathering calls, where a
                        repeat check is a scan-type call issued from an already
                        scanned cell, or an information call whose result carried
                        no new evidence (``new_evidence: false`` / ``cached``).
  evidence coverage     what the episode really looked at: the reported
                        coverage of the observation/search contract -
                        ``coverage.checked_deg`` / ``unchecked_deg`` are a degree
                        count or the lookaround's ``[lo, hi]`` sector list,
                        ``worth_looking`` a bool or the flagged views
                        (``coverage_source: reported``) - else the distinct
                        45-degree heading bins and the heading windows of the
                        traced observations (``estimated``); plus the view
                        expansion (``views_expanded`` from ``view_index``) and
                        the crop re-checks (``crop_checks``).
  total cost            ``vlm_calls_total`` = decision turns + the internal calls
                        the tools declare (contract field ``vlm_calls``, e.g. the
                        search skill's steps), never below the runner's reported
                        count; plus ``tokens_estimate`` and ``wall_time_s``.
  toward-target move    diagnostic only (``toward_target_rate``): a search or
                        obstacle-avoidance leg may legitimately leave the
                        bearing to the target, so it is never a pass/fail gate.

Decision annotation (report requirement: prove *why* the agent changed its
viewpoint, what it learned and how the next action changed). For every episode
the driver reconstructs the causal chain from the tool trace, the ``progress_kind``
field of the progress contract and the search skill's ``steps`` log:

    t3 ground_target(obs_2) [empty] -> no_candidate
    t4 STRATEGY CHANGE: 1x ground_target unproductive -> propose_viewpoints
    t5 go_to_viewpoint(vp_1) [ok] -> a new viewpoint; the next observe returns...
    t6 observe(mode=forward) [ok] -> obs_3, view_index=1, coverage checked=86
       -> this new view was used by t8 approach_candidate(cand_1)

Artifacts (under ``--out``):
  compare_decision_loop_report.md    aggregates, layer deltas, per-episode rows
                                     and the causal annotations
  compare_decision_loop_report.json  the same machine-readable (meta, arms,
                                     aggregates, layer_deltas, episodes+annotation)
  episodes.jsonl                     one line per episode (metrics + annotation)
  run_config.json                    the resolved CLI
  <arm>/<scenario>/<task>/<yaw>_seed<S>/  runner artifacts (summary, trace, log)

Exit code: 0 when every episode ran, 1 when any episode errored; ``--dry-run``
always 0. Reserved (``memory``) episodes are ``skipped``, not errors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python scripts/compare_decision_loop.py`
    sys.path.insert(0, str(ROOT))

from scripts import compare_agent_configs as compare  # noqa: E402
from scripts import verify_phase1 as phase1  # noqa: E402

TOOL_RUNNER = ROOT / "scripts" / "run_vlm_nav_tools.py"

#: The mechanism switches of the tool runner, as its ``summary.json.switches``
#: block reports them (requested and applied). The last two are the memory
#: layers: the runner names them after the memory implementation's own
#: constructor parameters (``--memory-spatial`` / ``--memory-episodic``), and a
#: memory that does not attest them reports ``None`` (unknown, not ``off``).
MECHANISM_SWITCHES = ("recovery", "state_block", "search_skill",
                      "spatial_memory", "episodic_memory")

# ---------------------------------------------------------------------------
# arms: the capability layers of the comparison
# ---------------------------------------------------------------------------

ARM_BASELINE = "baseline"
ARM_VISION = "vision_context"
ARM_RECOVERY = "recovery"
ARM_MEMORY = "memory"
ARM_MEMORY_SPATIAL = "memory_spatial_only"
ARM_ORDER = (ARM_BASELINE, ARM_VISION, ARM_RECOVERY, ARM_MEMORY, ARM_MEMORY_SPATIAL)

#: The memory layer switches every arm declares (the runner's default is ``on``,
#: so the pre-memory arms have to say ``off``).
MEMORY_OFF_ARGS = ("--memory-spatial", "off", "--memory-episodic", "off")
#: The catalogue of the goal.md M2 memory experiment. The memory *arms* of this
#: driver run the shared catalogue instead (see ``ARM_TABLE``): ``memory`` has no
#: ``search_target`` and no M3 tools, so a per-arm ``--toolset memory`` would mix
#: the memory layer with the loss of the search skill. It stays available as an
#: ``--arm-arg memory=--toolset memory ...`` override and is reported per arm.
M2_MEMORY_TOOLSET = "memory"


@dataclass(frozen=True)
class ArmSpec:
    """One capability layer: a label, the runner switches that enable it and
    whether it is part of this round at all."""

    name: str
    label: str
    runner_args: tuple[str, ...] = ()
    note: str = ""
    reserved: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "runner_args": list(self.runner_args),
            "note": self.note,
            "reserved": bool(self.reserved),
        }


#: The switch names below are the defaults of this driver; the runner source is
#: probed at run time and unsupported switches are dropped (``arm_support``),
#: so renaming one is a one-line edit here or an ``--arm-arg`` override. Every
#: arm names all three mechanism switches: the runner's default is ``on``, so an
#: arm that wants the old behavior has to say ``off`` (and the report then shows
#: the exact switch set each arm ran with).
ARM_TABLE: dict[str, ArmSpec] = {
    ARM_BASELINE: ArmSpec(
        name=ARM_BASELINE,
        label="baseline decision loop",
        runner_args=("--recovery", "off", "--state-block", "off", "--search-skill", "off",
                     *MEMORY_OFF_ARGS),
        note="the pre-decision-loop runner: no STATE block, no stall recovery, "
             "no search skill, no memory layer, the default 2 frames per decision",
    ),
    ARM_VISION: ArmSpec(
        name=ARM_VISION,
        label="+ multi-view visual context and the search skill",
        # ``--context-frames`` is the frames-per-decision knob: raising it feeds
        # several rendered views of the same state to the model (the multi-view
        # search skill adds its own views through ``scan_views``). The search
        # skill is the second half of the same layer, so it is switched on here.
        runner_args=("--context-frames", "4", "--search-skill", "on",
                     "--recovery", "off", "--state-block", "off", *MEMORY_OFF_ARGS),
        note="baseline + 4 frames/decision + the search_target routine; STATE "
             "block, recovery and the memory layer still off",
    ),
    ARM_RECOVERY: ArmSpec(
        name=ARM_RECOVERY,
        label="+ progress state and failure recovery",
        runner_args=("--context-frames", "4", "--search-skill", "on",
                     "--state-block", "on", "--recovery", "on", *MEMORY_OFF_ARGS),
        note="vision context plus the fixed STATE block and the progress-aware "
             "recovery (strategy change before termination); memory still off",
    ),
    ARM_MEMORY: ArmSpec(
        name=ARM_MEMORY,
        label="+ task-state memory: spatial + episodic",
        # The memory layer is the two stores the memory module keeps: the spatial
        # store (what the task is looking for, where it was seen) and the
        # episodic store (which view saw it). The catalogue stays the shared one
        # (``full``): the M2 ``memory`` catalogue carries no ``search_target``,
        # so running it here would drop the search skill of the previous layer
        # and mix it into the memory delta. ``--arm-arg memory=--toolset memory
        # ...`` restores that M2 configuration, and the report then says so.
        runner_args=("--context-frames", "4", "--search-skill", "on",
                     "--state-block", "on", "--recovery", "on",
                     "--memory-spatial", "on", "--memory-episodic", "on"),
        note="recovery + both memory stores (the runner's --memory-spatial/ "
             "--memory-episodic reach the memory module's own constructor "
             "parameters); catalogue unchanged (the M2 catalogue has no "
             "search_target - override with --arm-arg to run it)",
    ),
    ARM_MEMORY_SPATIAL: ArmSpec(
        name=ARM_MEMORY_SPATIAL,
        label="+ task-state memory: spatial only",
        # The paired delta against ``memory`` is the episodic store: same
        # switches, same catalogue, only --memory-episodic differs.
        runner_args=("--context-frames", "4", "--search-skill", "on",
                     "--state-block", "on", "--recovery", "on",
                     "--memory-spatial", "on", "--memory-episodic", "off"),
        note="the memory arm with the episodic store off: candidates and places "
             "are still remembered, the sighting image/observation pairing is not",
    ),
}

# ---------------------------------------------------------------------------
# scenarios: the two Phase-1 scenes (book, synth) plus a command-line CPU one
# ---------------------------------------------------------------------------

SCENARIO_BOOK = "book"
SCENARIO_SYNTH = "synth"
SCENARIO_SYNTH_HARD = "synth_hard"
SCENARIO_CPU = "cpu"

PHASE1_QWEN = Path(
    os.environ.get("VLM_NAV_PHASE1_DIR", ROOT / "data" / "phase1")
).expanduser().resolve()
PHASE1_LOW = PHASE1_QWEN


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    world: str | None = None
    ply: str | None = None
    scene_scale: float | None = None
    task_file: str | None = None
    note: str = ""
    tasks: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "world": self.world,
            "ply": self.ply,
            "scene_scale": self.scene_scale,
            "task_file": self.task_file,
            "tasks": [task.to_dict() for task in self.tasks],
            "note": self.note,
        }


SCENARIOS: dict[str, ScenarioSpec] = {
    SCENARIO_BOOK: ScenarioSpec(
        name=SCENARIO_BOOK,
        world="higs",
        ply="/scene_book.ply",
        task_file=str(PHASE1_QWEN / "tasks_book.json"),
        note="Phase-1 book scene (HiGS); gates 0.9 m (near) / 1.2 m (far)",
    ),
    SCENARIO_SYNTH: ScenarioSpec(
        name=SCENARIO_SYNTH,
        world="higs",
        ply=str(PHASE1_QWEN / "scenes" / "scenario_scene.ply"),
        task_file=str(PHASE1_LOW / "tasks_synth_fair.json"),
        note="Phase-1 synthetic scene with the fair approach gate (1.2 m)",
    ),
    SCENARIO_SYNTH_HARD: ScenarioSpec(
        name=SCENARIO_SYNTH_HARD,
        world="higs",
        ply=str(PHASE1_QWEN / "scenes" / "scenario_scene.ply"),
        task_file=str(PHASE1_QWEN / "tasks_synth.json"),
        note="synthetic scene with the older 0.6 m gate (structurally hard)",
    ),
    SCENARIO_CPU: ScenarioSpec(
        name=SCENARIO_CPU,
        world="cpu",
        note="CPU world; scene and tasks come from the command line",
    ),
}

DEFAULT_SCENARIOS = (SCENARIO_BOOK, SCENARIO_SYNTH)

# ---------------------------------------------------------------------------
# the trace vocabulary this driver reads (shared contracts)
# ---------------------------------------------------------------------------

#: Tools that gather information (a repeat of one is a repeated *check*).
STATUS_EMPTY = phase1.STATUS_EMPTY
STATUS_FAILED = phase1.STATUS_FAILED
STATUS_BLOCKED = phase1.STATUS_BLOCKED
STATUS_CACHED = phase1.STATUS_CACHED
STATUS_OK = phase1.STATUS_OK
INFO_TOOL_NAMES = (
    "observe",
    "ground_target",
    "search_target",
    "search",
    "inspect_region",
    "query_memory",
    "compare_candidates",
    "propose_viewpoints",
    "check_path",
    "task_status",
    "get_plan",
)
#: The search skill of the vision-context layer (contract: ``steps`` + ``vlm_calls``).
SEARCH_TOOL_NAMES = ("search_target", "search")
#: Tools that command one movement leg (mirrors ``verify_phase1.NAV_TOOL_NAMES``).
NAV_TOOL_NAMES = tuple(phase1.NAV_TOOL_NAMES)
#: Tools that maintain the persistent task cycle.
TASK_TOOL_NAMES = ("update_plan", "cancel_task")
#: Tools that report a located object.
GROUND_TOOL_NAMES = ("ground_target", "search_target", "search")
#: Tools that move to an observation position.
VIEWPOINT_TOOL_NAMES = ("go_to_viewpoint", "return_to_observation")
#: observe modes that sweep several views of the surroundings.
SCAN_MODES = tuple(compare.SCAN_MODES)
#: Tools/args that re-check a cropped region.
CROP_TOOL_NAMES = ("inspect_region", "crop", "zoom", "inspect_crop", "crop_check")
#: ``finish_reason`` values that mean "the agent declared completion".
CLAIM_REASONS = ("finish", "finished", "vlm_finish", "target_verified")
#: result statuses that carry no progress.
UNPRODUCTIVE_STATUS = (STATUS_EMPTY, STATUS_FAILED, STATUS_BLOCKED)
#: progress kinds of the progress contract (``TurnRecord.progress_kind``).
PROGRESS_KINDS = ("information", "action", "task", "none")
#: Result fields the observation contract uses for the view expansion.
VIEW_INDEX_FIELDS = ("view_index", "views_expanded", "view_count")
#: Result fields the observation contract uses for the direction coverage.
COVERAGE_FIELDS = ("coverage", "view_coverage", "scan_coverage")
COVERAGE_CHECKED_KEYS = ("checked_deg", "checked", "checked_degrees")
COVERAGE_UNCHECKED_KEYS = ("unchecked_deg", "unchecked", "unchecked_degrees")
COVERAGE_WORTH_KEYS = ("worth_looking", "worth", "worth_looking_now")
#: The search skill's step log (contract: ``[{step, why, obs_ids, result}]``).
SEARCH_STEP_FIELDS = ("steps", "search_steps")
#: Result fields carrying a located object's position (ground truth check).
CANDIDATE_POSITION_FIELDS = (
    "candidate_position",
    "candidate_pos_flu",
    "candidate_pose",
    "object_position_flu",
    "position_flu",
)
#: Result fields that declare how many VLM calls a tool spent internally.
VLM_COST_FIELDS = tuple(phase1.VLM_COST_FIELDS)
#: Result fields carrying the target visibility at the result pose.
VISIBILITY_FIELDS = tuple(phase1.VISIBILITY_FIELDS)
#: Displacement fields of one movement leg.
PROGRESS_FIELDS = tuple(phase1.PROGRESS_FIELDS)

TOOL_TRACE = "tool_trace.jsonl"
SUMMARY_FILE = "summary.json"
REPORT_JSON = "compare_decision_loop_report.json"
REPORT_MD = "compare_decision_loop_report.md"

#: Scene/budget options every arm of a cell must share (the sibling drivers'
#: pairing keys plus the budget/toolset options of the tool runner).
PAIRING_KEYS = tuple(compare.PAIRING_KEYS) + ("--max-turns", "--tool-budget", "--toolset")


# ---------------------------------------------------------------------------
# small helpers (reuse the sibling drivers where they already define one)
# ---------------------------------------------------------------------------

_as_float = compare._as_float
_as_int = compare._as_int
_mean = compare._mean
_fmt = compare._fmt
_wrap_deg = compare._wrap_deg
_int_list = compare._int_list
_str_list = compare._str_list
read_json = compare.read_json
read_jsonl = compare.read_jsonl
_result_data = compare._result_data
_pose_of = compare._pose_of
search_accounting = compare.search_accounting
runner_env = compare.runner_env
run_command = compare.run_command
git_head = compare.git_head


def _as_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _rate(numerator: float, denominator: float) -> float | None:
    if not denominator:
        return None
    return round(float(numerator) / float(denominator), 3)


def _sum(values: Iterable[Any]) -> float | None:
    numbers = [_as_float(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    return round(sum(numbers), 3) if numbers else None


def _max(values: Iterable[Any]) -> float | None:
    numbers = [_as_float(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    return max(numbers) if numbers else None


def _min(values: Iterable[Any]) -> float | None:
    numbers = [_as_float(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    return min(numbers) if numbers else None


def _round(value: Any, digits: int = 3) -> float | None:
    number = _as_float(value)
    return round(number, digits) if number is not None else None


def _num(value: Any) -> float | None:
    """``0.0``/``1.0`` for a flag, the float itself otherwise (deltas of
    ``solved``/``claim`` style columns are counted, not dropped)."""
    if isinstance(value, bool):
        return float(value)
    return _as_float(value)


def _tri_state(value: Any) -> str:
    """``on``/``off`` for a reported switch, ``?`` when nothing was reported."""
    if isinstance(value, bool):
        return "on" if value else "off"
    return "?"


def _as_point(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        value = value.get("position_flu") or value.get("position") or value.get("pos")
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return [float(v) for v in value[:3]]
    except (TypeError, ValueError):
        return None


def _distance(a: Any, b: Any) -> float | None:
    point_a, point_b = _as_point(a), _as_point(b)
    if point_a is None or point_b is None:
        return None
    return math.dist(point_a, point_b)


def _bearing_deg(origin: Any, target: Any) -> float | None:
    start, end = _as_point(origin), _as_point(target)
    if start is None or end is None:
        return None
    dx, dy = end[0] - start[0], end[1] - start[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    return math.degrees(math.atan2(dy, dx))


def _id_list(*values: Any) -> list[str]:
    """Typed ids (``obs_3``, ``cand_1``, ``vp_2``) found in the given values."""
    found: list[str] = []
    for value in values:
        if isinstance(value, str):
            found += re.findall(r"\b(?:obs|cand|vp|nav|task)_\d+\b", value)
        elif isinstance(value, dict):
            found += _id_list(*value.values())
        elif isinstance(value, (list, tuple, set)):
            found += _id_list(*value)
    seen: set[str] = set()
    out: list[str] = []
    for item in found:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# arm resolution: declared switches -> switches the runner really has
# ---------------------------------------------------------------------------

_FLAG_RE = re.compile(r"add_argument\(\s*[\"'](-{1,2}[A-Za-z0-9][A-Za-z0-9_-]*)[\"']")


def runner_flags_from_source(path: Path | str = TOOL_RUNNER) -> set[str]:
    """Option strings of the tool runner, read from its argparse source.

    A source scan (not an import) so the probe stays cheap: the runner imports
    torch and the world stack. Only the ``add_argument("--flag", ...)`` spelling
    the runner/this repo uses is recognised; an option the scan misses is only
    ever *dropped* from an arm, never invented, and ``--arm-arg`` bypasses the
    probe entirely.
    """
    try:
        text = Path(path).read_text()
    except OSError:
        return set()
    return set(_FLAG_RE.findall(text))


def split_flag_value(tokens: list[str]) -> list[tuple[str, list[str]]]:
    """Group a runner argv into ``(option, tokens)`` pairs (value included)."""
    groups: list[tuple[str, list[str]]] = []
    index = 0
    while index < len(tokens):
        token = str(tokens[index])
        group = [token]
        if index + 1 < len(tokens) and not str(tokens[index + 1]).startswith("-"):
            group.append(str(tokens[index + 1]))
            index += 1
        groups.append((token, group))
        index += 1
    return groups


def filter_supported_args(tokens: Iterable[str], supported: set[str] | None) -> tuple[list[str], list[str]]:
    """``(kept, dropped_flags)`` for one arm's declared runner switches.

    A ``--flag value`` pair is dropped whole when the runner does not know
    ``--flag``; ``--flag=value`` is a single token. ``supported=None`` keeps
    everything (the probe was skipped).
    """
    tokens = [str(token) for token in tokens]
    if supported is None:
        return tokens, []
    kept: list[str] = []
    dropped: list[str] = []
    for option, group in split_flag_value(tokens):
        name = option.split("=", 1)[0]
        if name.startswith("-") and name not in supported:
            dropped.append(name)
            continue
        kept += group
    return kept, dropped


def apply_arm_overrides(args: argparse.Namespace) -> dict[str, list[str]]:
    """``--arm-arg ARM=<tokens>`` overrides: ``ARM=none`` clears the defaults."""
    overrides: dict[str, list[str]] = {}
    for blob in (getattr(args, "arm_args", None) or []):
        name, _, rest = str(blob).partition("=")
        name = name.strip()
        if name not in ARM_TABLE:
            raise SystemExit(
                f"unknown arm {name!r} in --arm-arg; pick from {', '.join(ARM_ORDER)}"
            )
        rest = rest.strip()
        overrides[name] = [] if rest.lower() in ("", "none", "-") else rest.split()
    return overrides


def mechanism_report(rows: list[dict]) -> list[dict]:
    """Per arm: the mechanism switches the runner summaries really applied.

    ``row["switches"]`` is the runner's ``summary.json.switches`` block. A switch
    whose value differs between the episodes of one arm (or a summary that does
    not report the block at all) makes the arm inconsistent: the report then says
    the layer was not reliably enabled instead of presenting it as real.

    The memory layers are the exception to the "missing knob counts as off" rule:
    a memory that does not report its layers leaves the applied value ``None``.
    An arm that *asked* for such a layer is then listed in ``unattested`` - the
    report says "unknown", because "the memory did not confirm the layer" is not
    the same statement as "the layer was off". ``toolset`` says which catalogue
    the arm really ran in, so an arm that runs its own catalogue (the M2 variant
    of the memory arm) is visible next to the arms that share the default one.
    """
    order: list[str] = []
    for row in rows:
        arm = row.get("arm")
        if arm is not None and arm not in order:
            order.append(arm)
    out: list[dict] = []
    for arm in order:
        values: dict[str, set] = {name: set() for name in MECHANISM_SWITCHES}
        requested: dict[str, set] = {name: set() for name in MECHANISM_SWITCHES}
        runtimes: set = set()
        toolsets: set = set()
        n_reported = n_missing = 0
        for row in rows:
            if row.get("arm") != arm:
                continue
            if row.get("toolset"):
                toolsets.add(str(row["toolset"]))
            block = row.get("switches")
            applied = block.get("applied") if isinstance(block, dict) else None
            if not isinstance(applied, dict):
                n_missing += 1
                continue
            n_reported += 1
            if applied.get("runtime"):
                runtimes.add(str(applied["runtime"]))
            for name in MECHANISM_SWITCHES:
                if name in applied:
                    value = applied[name]
                    # a memory layer that is not attested stays None (unknown)
                    values[name].add(bool(value) if isinstance(value, bool) else None)
            asked = block.get("requested") if isinstance(block, dict) else None
            if isinstance(asked, dict):
                for name in MECHANISM_SWITCHES:
                    if name in asked:
                        requested[name].add(str(asked[name]))

        def single(seen: set) -> Any:
            return next(iter(seen)) if len(seen) == 1 else None

        disagreements = sorted(name for name, seen in values.items() if len(seen) > 1)
        applied_values = {name: single(seen) for name, seen in values.items()}
        asked_values = {name: single(seen) for name, seen in requested.items()}
        out.append({
            "arm": arm,
            "n_episodes": n_reported + n_missing,
            "n_reported": n_reported,
            "n_missing": n_missing,
            "runtime": ", ".join(sorted(runtimes)) or None,
            "toolset": ", ".join(sorted(toolsets)) or None,
            "applied": applied_values,
            "requested": asked_values,
            "unattested": [
                name for name in MECHANISM_SWITCHES
                if asked_values.get(name) == "on" and applied_values.get(name) is None
            ],
            "disagreements": disagreements,
            "consistent": not disagreements and n_missing == 0 and n_reported > 0,
        })
    return out


def resolve_arms(args: argparse.Namespace,
                 supported: set[str] | None) -> tuple[list[ArmSpec], list[dict]]:
    """Resolved arms (declared args filtered to the runner's real switches).

    Returns the arms in ``ARM_ORDER`` order plus one support record per arm:
    ``{arm, declared, effective, dropped, support: full|degraded, reason}``.
    Raises ``SystemExit`` when ``--strict-arms`` and an arm lost a switch.
    """
    overrides = apply_arm_overrides(args)
    arms: list[ArmSpec] = []
    support: list[dict] = []
    for name in args.arm_list:
        spec = ARM_TABLE[name]
        declared = tuple(overrides.get(name, list(spec.runner_args)))
        kept, dropped = filter_supported_args(declared, supported)
        resolved = ArmSpec(
            name=spec.name,
            label=spec.label,
            runner_args=tuple(kept),
            note=spec.note,
            reserved=spec.reserved,
        )
        arms.append(resolved)
        support.append({
            "arm": name,
            "declared": list(declared),
            "effective": list(kept),
            "dropped": sorted(set(dropped)),
            "support": "degraded" if dropped else "full",
            "reserved": bool(spec.reserved),
            "reason": (
                f"runner has no {', '.join(sorted(set(dropped)))}; "
                f"override with --arm-arg {name}=<switch>"
                if dropped else ""
            ),
        })
    broken = [row for row in support if row["dropped"]]
    if broken and getattr(args, "strict_arms", False):
        raise SystemExit(
            "arms need switches this runner does not have: "
            + "; ".join(f"{row['arm']}: {row['reason']}" for row in broken)
        )
    return arms, support


# ---------------------------------------------------------------------------
# scenarios, tasks and episodes
# ---------------------------------------------------------------------------

def load_scenarios(args: argparse.Namespace) -> list[ScenarioSpec]:
    """Resolve every ``--scenario`` into a concrete scene + task list.

    Command-line scene/task options override the preset (so ``--ply``/``--task-file``
    re-point a preset), which is how the mock/CPU runs reuse the same matrix.
    """
    custom_tasks: list[compare.TaskSpec] = []
    if args.task_file:
        custom_tasks += _tasks_from_file(args.task_file, args.success_dist)
    for blob in (args.task_json or []):
        custom_tasks.append(compare.TaskSpec.from_dict(json.loads(blob), args.success_dist))
    if not custom_tasks and args.instruction and args.target_pos:
        custom_tasks.append(
            compare.TaskSpec(
                name=args.task_name,
                instruction=args.instruction,
                target_pos=tuple(float(v) for v in args.target_pos),
                target_name=args.target_name,
                target_radius=float(args.target_radius),
                start_pos=tuple(float(v) for v in args.start_pos),
                success_dist=float(args.success_dist),
            )
        )

    scenarios: list[ScenarioSpec] = []
    for name in args.scenario_list:
        preset = SCENARIOS.get(name)
        if preset is None:
            raise SystemExit(
                f"unknown --scenario {name!r}; pick from {', '.join(SCENARIOS)}"
            )
        if custom_tasks:
            tasks = list(custom_tasks)
        else:
            tasks = _tasks_from_file(preset.task_file, args.success_dist)
        scenarios.append(
            ScenarioSpec(
                name=preset.name,
                world=args.world or preset.world,
                ply=args.ply or preset.ply,
                scene_scale=args.scene_scale if args.scene_scale is not None else preset.scene_scale,
                task_file=None if custom_tasks else preset.task_file,
                note=preset.note,
                tasks=tasks,
            )
        )
    for scenario in scenarios:
        if not scenario.tasks:
            raise SystemExit(
                f"scenario {scenario.name!r} has no tasks: give --task-file/--task-json "
                "or --instruction with --target-pos"
            )
        if scenario.world in ("cpu", "higs", "env") and not scenario.ply:
            raise SystemExit(f"scenario {scenario.name!r} needs --ply")
        if not args.dry_run and not _scene_exists(scenario):
            raise SystemExit(
                f"scenario {scenario.name!r}: scene {scenario.ply!r} does not exist; "
                "pass --ply or run inside the container"
            )
    if args.backend.startswith("hf") and not args.model_path:
        raise SystemExit("--backend hf requires --model-path")
    if args.backend == "openai" and not args.model:
        raise SystemExit("--backend openai requires --model")
    return scenarios


def _tasks_from_file(path: str | None, default_success_dist: float) -> list:
    if not path:
        return []
    data = read_json(Path(path))
    if data is None:
        with open(path) as handle:
            data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("tasks") or [data]
    return [compare.TaskSpec.from_dict(entry, default_success_dist) for entry in data or []]


def _scene_exists(scenario: ScenarioSpec) -> bool:
    if scenario.world == "super":
        return True
    if not scenario.ply:
        return False
    return Path(scenario.ply).exists()


@dataclass(frozen=True)
class EpisodeSpec:
    arm: str
    scenario: str
    task: compare.TaskSpec
    seed: int
    yaw_mode: str
    start_yaw_deg: float

    @property
    def slug(self) -> str:
        return f"{self.yaw_mode}_seed{int(self.seed)}"

    @property
    def cell(self) -> str:
        """Identical for every arm of one paired episode."""
        return f"{self.scenario}__{self.task.name}__{self.yaw_mode}__seed{int(self.seed)}"

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "scenario": self.scenario,
            "task": self.task.name,
            "seed": int(self.seed),
            "yaw_mode": self.yaw_mode,
            "start_yaw_deg": self.start_yaw_deg,
        }


def build_episodes(args: argparse.Namespace, scenarios: list[ScenarioSpec]) -> list[EpisodeSpec]:
    episodes: list[EpisodeSpec] = []
    for scenario in scenarios:
        for task in scenario.tasks:
            for seed in args.seed_list:
                for yaw_mode in args.yaw_mode_list:
                    yaw = compare.start_yaw_deg(
                        yaw_mode, task, seed, args.initial_yaw_half_fov_deg
                    )
                    for arm in args.arm_list:
                        episodes.append(
                            EpisodeSpec(arm, scenario.name, task, seed, yaw_mode, yaw)
                        )
    return episodes


def episode_dir(out_root: Path, spec: EpisodeSpec) -> Path:
    return Path(out_root) / spec.arm / spec.scenario / spec.task.name / spec.slug


# ---------------------------------------------------------------------------
# command construction
# ---------------------------------------------------------------------------

def arm_of(args: argparse.Namespace, name: str) -> ArmSpec:
    return args.arm_specs[name]


def scene_args(args: argparse.Namespace, scenario: ScenarioSpec) -> list[str]:
    out: list[str] = ["--world", str(scenario.world)]
    if scenario.ply:
        out += ["--ply", str(scenario.ply)]
    if scenario.scene_scale is not None:
        out += ["--scene-scale", str(scenario.scene_scale)]
    if args.min_opacity is not None:
        out += ["--min-opacity", str(args.min_opacity)]
    if args.combined_ply:
        out += ["--combined-ply", str(args.combined_ply)]
    if args.background_ply:
        out += ["--background-ply", str(args.background_ply)]
    if args.semantic_manifest:
        out += ["--semantic-manifest", str(args.semantic_manifest)]
    if args.voxel_size is not None:
        out += ["--voxel-size", str(args.voxel_size)]
    if args.target_index is not None:
        out += ["--target-index", str(int(args.target_index))]
    if args.env_overrides:
        out += ["--env-overrides", *(str(v) for v in args.env_overrides)]
    if args.device:
        out += ["--device", str(args.device)]
    return out


def model_args(args: argparse.Namespace) -> list[str]:
    return compare.model_args(args)


def success_dist(args: argparse.Namespace, task: compare.TaskSpec) -> float:
    return float(task.success_dist if task.success_dist is not None else args.success_dist)


def tool_budget(args: argparse.Namespace) -> int:
    return int(args.tool_budget if args.tool_budget is not None else args.max_vlm_calls)


def budget_mapping(args: argparse.Namespace) -> dict:
    return {
        "max_vlm_calls": int(args.max_vlm_calls),
        "tool": {
            "max_turns": int(args.max_vlm_calls),
            "tool_budget": tool_budget(args),
        },
        "note": (
            "every arm gets the same turn and tool-call budget; the budget is "
            "part of the pairing check, only the arm switches differ"
        ),
    }


def build_command(args: argparse.Namespace, scenario: ScenarioSpec,
                  spec: EpisodeSpec, out_dir: Path) -> list[str]:
    """The tool runner's CLI: shared scene/start/budget flags + the arm switches."""
    task = spec.task
    arm = arm_of(args, spec.arm)
    cmd = [sys.executable, "-m", "scripts.run_vlm_nav_tools"]
    cmd += scene_args(args, scenario)
    if args.render_size:
        cmd += ["--render-size", str(int(args.render_size[0])), str(int(args.render_size[1]))]
    cmd += ["--target-name", str(task.target_name)]
    cmd += ["--target-pos", *(str(float(v)) for v in task.target_pos)]
    cmd += ["--target-radius", str(float(task.target_radius))]
    cmd += ["--start-pos", *(str(float(v)) for v in task.start_pos)]
    cmd += ["--start-yaw-deg", str(spec.start_yaw_deg)]
    cmd += ["--success-dist", str(float(success_dist(args, task)))]
    cmd += ["--seed", str(int(spec.seed))]
    cmd += ["--initial-yaw-mode", spec.yaw_mode if spec.yaw_mode in ("visible", "random") else "target"]
    cmd += ["--max-init-attempts", str(int(args.max_init_attempts))]
    cmd += ["--instruction", str(task.instruction)]
    cmd += ["--backend", str(args.backend)]
    cmd += model_args(args)
    cmd += ["--toolset", str(args.toolset)]
    cmd += ["--max-turns", str(int(args.max_vlm_calls))]
    cmd += ["--tool-budget", str(tool_budget(args))]
    if args.context_frames is not None:
        cmd += ["--context-frames", str(int(args.context_frames))]
    if args.scan_views is not None:
        cmd += ["--scan-views", str(int(args.scan_views))]
    if args.scan_pitch_deg is not None:
        cmd += ["--scan-pitch-deg", str(float(args.scan_pitch_deg))]
    cmd += [str(extra) for extra in (args.tool_arg or [])]
    cmd += list(arm.runner_args)  # the arm's own switches win over the shared ones
    cmd += ["--out", str(out_dir)]
    return cmd


def pairing_check(args: argparse.Namespace, episodes: list[EpisodeSpec],
                  commands: dict) -> dict:
    """The scene/start/budget options of every arm of a cell must be identical.

    Everything else is *reported* as ``arm_only_options`` so the report shows
    exactly what the comparison changed: an option an arm declares in its own
    switch list (``--context-frames 4`` of the vision arm, a per-arm
    ``--toolset memory``) is an arm switch by construction, so the check does not
    require the other arms to agree on it.
    """
    declared: set[str] = set()
    for spec in getattr(args, "arm_specs", {}).values():
        declared |= {str(token).split("=", 1)[0] for token in spec.runner_args
                     if str(token).startswith("-")}
    cells: dict[str, dict[str, dict]] = {}
    for spec in episodes:
        key = (spec.arm, spec.scenario, spec.task.name, spec.seed, spec.yaw_mode)
        cells.setdefault(spec.cell, {})[spec.arm] = compare._args_map(commands[key])
    issues: list[dict] = []
    arm_only: set[str] = set()
    for cell, per_arm in cells.items():
        if len(per_arm) < 2:
            continue
        shared_keys = None
        for options in per_arm.values():
            keys = {key for key in PAIRING_KEYS if key in options and key not in declared}
            shared_keys = keys if shared_keys is None else (shared_keys & keys)
        for options in per_arm.values():
            arm_only |= set(options) - (shared_keys or set())
        reference_arm, reference = next(iter(per_arm.items()))
        for arm, options in per_arm.items():
            for key in sorted(shared_keys or ()):
                if options.get(key) != reference.get(key):
                    issues.append({
                        "cell": cell,
                        "option": key,
                        arm: options.get(key),
                        reference_arm: reference.get(key),
                    })
    return {
        "cells": len(cells),
        "consistent": not issues,
        "issues": issues[:20],
        "arm_only_options": sorted(arm_only),
        "checked_keys": [key for key in PAIRING_KEYS],
    }


# ---------------------------------------------------------------------------
# metrics: one pass over the trace
# ---------------------------------------------------------------------------


def trace_entries(out_dir: Path) -> list[dict]:
    entries = compare._find_trace(out_dir, TOOL_TRACE)
    return [entry for entry in entries if isinstance(entry, dict)]


def _entry_turn(entry: dict, index: int) -> int:
    value = _as_int(entry.get("turn"))
    return value if value is not None else index + 1


def _declared_vlm_calls(*sources: dict) -> int:
    total = 0
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in VLM_COST_FIELDS:
            value = _as_int(source.get(key))
            if value:
                total += value
    return total


def declared_routine_calls(entries: list[dict]) -> dict:
    """Sum of what the tools declare about their own sub-routines.

    ``search_target`` publishes ``data["cost"] = {"outer_tool_calls": 1,
    "inner_tool_calls": N, "vlm_calls": V}``: the episode paid one tool call, the
    routine ran N steps inside it and made V model calls. The per-episode row
    keeps the split so the report can say how much sub-routine work a decision
    triggered - and whether the routine stayed inside its allowance.
    """
    inner = outer = vlm = calls = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        data = entry.get("data") if isinstance(entry.get("data"), dict) else {}
        cost = data.get("cost") if isinstance(data.get("cost"), dict) else {}
        if not cost:
            continue
        if str(entry.get("tool") or "") == "search_target":
            calls += 1
        outer += max(0, _as_int(cost.get("outer_tool_calls")) or 0)
        # a trace written before the split published its inner calls as
        # ``cost["tool_calls"]``: the meaning is the same, so old runs keep
        # comparing (the routine's own steps, not the episode's call)
        if "inner_tool_calls" in cost:
            inner += max(0, _as_int(cost.get("inner_tool_calls")) or 0)
        else:
            inner += max(0, _as_int(cost.get("tool_calls")) or 0)
        vlm += max(0, _as_int(cost.get("vlm_calls")) or 0)
    return {
        "declared_outer_tool_calls": outer,
        "declared_inner_tool_calls": inner,
        "declared_routine_vlm_calls": vlm,
        "search_target_calls": calls,
    }


def scan_like(tool: str, args: dict) -> bool:
    """A call that sweeps several views of the surroundings (a *scan*)."""
    if tool in SEARCH_TOOL_NAMES:
        return True
    if tool != "observe":
        return False
    mode = str(args.get("mode") or "").lower()
    return mode in SCAN_MODES


def info_call(tool: str, args: dict) -> bool:
    """A call that gathers evidence (verify/finish/movement calls are not checks)."""
    if tool in ("finish", "verify_goal", ""):
        return False
    if tool in INFO_TOOL_NAMES or tool in SEARCH_TOOL_NAMES:
        return True
    return tool not in NAV_TOOL_NAMES and tool not in TASK_TOOL_NAMES


def derive_progress_kind(entry: dict, tool: str, status: str, data: dict,
                         new_evidence: bool | None) -> tuple[str, str]:
    """``(kind, source)``: the progress contract's field when present, else a
    derivation from the tool family and ``new_evidence`` (``source="derived"``)."""
    explicit = entry.get("progress_kind")
    if explicit is None:
        explicit = data.get("progress_kind")
    if isinstance(explicit, str) and explicit in PROGRESS_KINDS:
        return explicit, "reported"
    if tool in NAV_TOOL_NAMES:
        return "action", "derived"
    if tool in TASK_TOOL_NAMES:
        return "task", "derived"
    if tool in ("finish", ""):
        return "none", "derived"
    if new_evidence is True:
        return "information", "derived"
    if status == STATUS_CACHED:
        return "none", "derived"
    return "none", "derived"


def step_annotation(entry: dict, index: int) -> dict:
    """One trace row as an annotation step (raw fields, no verdicts yet)."""
    tool = str(entry.get("tool") or "")
    args = entry.get("args") if isinstance(entry.get("args"), dict) else {}
    data = _result_data(entry)
    status = str(entry.get("status") or "")
    new_evidence = _as_bool(entry.get("new_evidence"))
    if new_evidence is None and status == STATUS_CACHED:
        new_evidence = False
    kind, kind_source = derive_progress_kind(entry, tool, status, data, new_evidence)
    coverage = None
    for key in COVERAGE_FIELDS:
        if isinstance(data.get(key), dict):
            coverage = data[key]
            break
    view_index = None
    for key in VIEW_INDEX_FIELDS:
        if data.get(key) is not None:
            view_index = data[key]
            break
    steps = None
    for key in SEARCH_STEP_FIELDS:
        if isinstance(data.get(key), list):
            steps = data[key]
            break
    # the finish declaration: the model's own argument, or what a verifying
    # finish tool answered. ``None`` = a bare finish (nothing declared)
    declared = _as_bool(args.get("success"))
    if declared is None:
        declared = _as_bool(data.get("success"))
    return {
        "turn": _entry_turn(entry, index),
        "tool": tool or "(no tool call)",
        "args": dict(args),
        "why": str(args.get("reason") or args.get("why") or args.get("intent") or ""),
        "status": status,
        "reason": str(data.get("reason") or entry.get("reason") or ""),
        "outcome": str(data.get("outcome") or entry.get("outcome") or ""),
        "detail": str(entry.get("detail") or ""),
        "progress_kind": kind,
        "progress_kind_source": kind_source,
        "new_evidence": new_evidence,
        "candidate_id": str(data.get("candidate_id")) if data.get("candidate_id") else None,
        "viewpoint_id": str(data.get("viewpoint_id")) if data.get("viewpoint_id") else None,
        "observation_id": str(data.get("observation_id")) if data.get("observation_id") else None,
        "view_index": view_index,
        "coverage": coverage,
        "steps": steps,
        "image_refs": data.get("image_refs") if isinstance(data.get("image_refs"), list) else None,
        "headings_deg": _headings_of(data),
        "vlm_calls": _declared_vlm_calls(data, entry),
        "verify_verified": _as_bool(data.get("verified")),
        "success": declared,
        "pose": _pose_of(entry),
        "ids": _id_list(args, data),
    }


def _headings_of(data: dict) -> list[float] | None:
    for key in ("headings_deg", "headings", "view_heading_deg"):
        value = data.get(key)
        if isinstance(value, (list, tuple)):
            numbers = [_as_float(item) for item in value]
            numbers = [number for number in numbers if number is not None]
            if numbers:
                return numbers
    for key in ("heading_deg", "yaw_deg"):
        number = _as_float(data.get(key))
        if number is not None:
            return [number]
    return None


def repeat_stats(entries: list[dict], grid_m: float) -> dict:
    """Repeat *checks*: same region or same evidence looked at again.

    Region repeats follow ``compare_agent_configs.search_accounting``: a
    scan-type call issued from a 2 m cell that was already scanned. Evidence
    repeats are information calls whose result carried no new evidence
    (``new_evidence: false``, or the runtime's ``cached`` status). The union of
    their turn indices over the number of information calls is
    ``repeat_search_ratio``.
    """
    scan_events: list[dict] = []
    scan_turns: set[int] = set()
    evidence_turns: set[int] = set()
    info_calls = 0
    seen_cells: set = set()
    scanned_cells: set = set()
    current_cell = None
    for index, entry in enumerate(entries):
        tool = str(entry.get("tool") or "")
        args = entry.get("args") if isinstance(entry.get("args"), dict) else {}
        status = str(entry.get("status") or "")
        pose = _pose_of(entry)
        cell = compare._cell_of(pose, grid_m) if pose else None
        turn = _entry_turn(entry, index)
        is_scan = scan_like(tool, args)
        scan_events.append({"scan": is_scan, "pose": pose})
        if cell is not None and cell != current_cell:
            seen_cells.add(cell)
            current_cell = cell
        if is_scan:
            scan_turns.add(turn)
            if cell is not None:
                scanned_cells.add(cell)
        if info_call(tool, args):
            info_calls += 1
            fresh = _as_bool(entry.get("new_evidence"))
            if fresh is False or status == STATUS_CACHED:
                evidence_turns.add(turn)
    accounting = search_accounting(scan_events, grid_m=grid_m)
    region_turns: set[int] = set()
    scanned: set = set()
    for index, entry in enumerate(entries):
        tool = str(entry.get("tool") or "")
        args = entry.get("args") if isinstance(entry.get("args"), dict) else {}
        if not scan_like(tool, args):
            continue
        # a scan-type call issued from a cell already scanned (same rule as
        # ``compare_agent_configs.search_accounting.repeat_scans``)
        pose = _pose_of(entry)
        cell = compare._cell_of(pose, grid_m) if pose else None
        if cell is not None:
            if cell in scanned:
                region_turns.add(_entry_turn(entry, index))
            scanned.add(cell)
    repeats = region_turns | evidence_turns
    return {
        "scan_calls": accounting["scan_calls"],
        "repeat_scans": accounting["repeat_scans"],
        "region_revisits": accounting["region_revisits"],
        "unique_cells": accounting["unique_cells"],
        "info_calls": info_calls,
        "repeat_evidence_calls": len(evidence_turns),
        "repeat_region_calls": len(region_turns),
        "repeat_search_checks": len(repeats),
        "repeat_scans_total": accounting["repeat_scans"] + accounting["region_revisits"],
        "repeat_search_ratio": _rate(len(repeats), info_calls),
        "repeat_turns": sorted(repeats),
    }


def coverage_stats(entries: list[dict], *, hfov_deg: float = 86.0,
                   bins: int = 8) -> dict:
    """Evidence coverage: reported contract fields first, else estimated.

    The observation/search contract reports coverage as
    ``{checked_deg, unchecked_deg, worth_looking, bin_deg, all_directions_covered,
    down_checked}`` where ``checked_deg``/``unchecked_deg`` are a *number of
    degrees* or a list of ``[lo, hi]`` sector ranges (the lookaround reports the
    sector list). The driver sums the sector widths, counts the flagged
    ``worth_looking`` views and derives ``direction_coverage`` from the sectors
    checked over the whole circle. Without the contract it estimates the looked-at
    directions from the traced observation headings (distinct 45-degree bins and
    the union of their FOV windows) and marks ``coverage_source: estimated``, so a
    comparison never silently mixes the two.
    """
    reported = 0
    checked_deg_max = None
    checked_sectors_max = None
    unchecked_last = None
    unchecked_sectors_last = None
    worth = 0
    full_sweeps = 0
    down_checked_calls = 0
    total_sectors = bins
    views = 0
    view_index_max = None
    crops = 0
    headings: list[float] = []
    sector_labels: dict[str, None] = {}
    for entry in entries:
        tool = str(entry.get("tool") or "")
        data = _result_data(entry)
        for key in COVERAGE_FIELDS:
            value = data.get(key)
            if not isinstance(value, dict):
                continue
            reported += 1
            bin_deg = _as_float(value.get("bin_deg"))
            if bin_deg and bin_deg > 0:
                total_sectors = int(round(360.0 / bin_deg))
            checked_deg, checked_sectors, labels = _sector_span(value, COVERAGE_CHECKED_KEYS)
            unchecked_deg, unchecked_sectors, _ = _sector_span(value, COVERAGE_UNCHECKED_KEYS)
            if checked_deg is not None:
                checked_deg_max = checked_deg if checked_deg_max is None \
                    else max(checked_deg_max, checked_deg)
            if checked_sectors is not None:
                checked_sectors_max = checked_sectors if checked_sectors_max is None \
                    else max(checked_sectors_max, checked_sectors)
            if unchecked_deg is not None:
                unchecked_last = unchecked_deg
                unchecked_sectors_last = unchecked_sectors
            for label in labels:
                sector_labels.setdefault(label, None)
            worth_flag = next((value.get(k) for k in COVERAGE_WORTH_KEYS
                               if value.get(k) is not None), None)
            if isinstance(worth_flag, (list, tuple)):
                worth += len(worth_flag)
            elif worth_flag is True:
                worth += 1
            if value.get("all_directions_covered") is True:
                full_sweeps += 1
            if value.get("down_checked") is True:
                down_checked_calls += 1
            break
        view_index = next((data.get(key) for key in VIEW_INDEX_FIELDS
                           if data.get(key) is not None), None)
        if isinstance(view_index, (list, tuple)):
            views += len(view_index)
            numbers = [_as_int(item) for item in view_index]
            numbers = [number for number in numbers if number is not None]
            if numbers:
                view_index_max = max(numbers) if view_index_max is None else max(view_index_max, *numbers)
        elif view_index is not None:
            views += 1
            number = _as_int(view_index)
            if number is not None:
                view_index_max = number if view_index_max is None else max(view_index_max, number)
        headings += _headings_of(data) or []
        if tool in CROP_TOOL_NAMES or any("crop" in str(key) for key in data):
            crops += 1
    heading_bins = sorted({int(math.floor((_wrap_deg(h) + 180.0) / (360.0 / bins)))
                           % bins for h in headings})
    coverage_deg_estimate = None
    if headings:
        covered = set()
        half = max(0.0, float(hfov_deg)) / 2.0
        for heading in headings:
            start = int(math.floor((_wrap_deg(heading - half) + 180.0) / 2.0)) * 2
            for step in range(int(math.ceil(hfov_deg / 2.0)) + 1):
                covered.add((start + 2 * step) % 360)
        coverage_deg_estimate = float(len(covered))
    direction_coverage = (
        _rate(checked_sectors_max, total_sectors) if checked_sectors_max is not None
        else _rate(len(heading_bins), bins)
    )
    return {
        "coverage_calls_reported": reported,
        "coverage_source": "reported" if reported else ("estimated" if headings else "absent"),
        "coverage_checked_deg": _round(checked_deg_max, 1),
        "coverage_checked_sectors": checked_sectors_max,
        "coverage_total_sectors": total_sectors if reported else None,
        "coverage_unchecked_deg": _round(unchecked_last, 1),
        "coverage_unchecked_sectors": unchecked_sectors_last,
        "coverage_worth_looking": worth,
        "coverage_full_sweeps": full_sweeps,
        "coverage_down_checked": down_checked_calls,
        "coverage_checked_deg_estimate": _round(coverage_deg_estimate, 1),
        "direction_bins": (
            sorted(sector_labels, key=_sector_label_low) if sector_labels else heading_bins
        ),
        "direction_coverage": direction_coverage,
        "views_expanded": views or None,
        "view_index_max": view_index_max,
        "crop_checks": crops,
    }


def _sector_label_low(label: str) -> float:
    try:
        return float(str(label).split("..")[0])
    except (TypeError, ValueError):
        return 0.0


def _sector_span(coverage: dict, keys: tuple[str, ...]) -> tuple[float | None, int | None, list[str]]:
    """``(degrees, sectors, labels)`` of ``checked_deg``/``unchecked_deg``.

    Accepts the contract's two spellings: a plain number of degrees, or the
    lookaround's list of ``[lo, hi]`` sector ranges (degrees summed, one sector
    each). An empty list is a real answer ("nothing"), not a missing field.
    """
    present = None
    for key in keys:
        if key in coverage and coverage.get(key) is not None:
            present = coverage.get(key)
            break
    if present is None:
        return None, None, []
    number = _as_float(present)
    if number is not None:
        return number, None, []
    if not isinstance(present, (list, tuple)):
        return None, None, []
    if not present:
        return 0.0, 0, []
    total = 0.0
    labels: list[str] = []
    count = 0
    for item in present:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        low, high = _as_float(item[0]), _as_float(item[1])
        if low is None or high is None:
            continue
        span = (high - low) % 360.0
        if span == 0.0 and high != low:
            span = 360.0
        total += span
        count += 1
        labels.append(f"{low:g}..{high:g}")
    if count == 0:
        return None, None, []
    return total, count, labels


def movement_stats(entries: list[dict], target_pos: Any, *,
                   min_progress_m: float = 0.05) -> dict:
    """Movement diagnostic only: legs that really moved, and how many of them
    headed towards the target (search/avoidance legs may legitimately leave)."""
    nav_attempts = moved = toward = 0
    progress_total = 0.0
    previous_pose = None
    for entry in entries:
        tool = str(entry.get("tool") or "")
        data = _result_data(entry)
        pose = _pose_of(entry)
        args = entry.get("args") if isinstance(entry.get("args"), dict) else {}
        if info_call(tool, args) or tool in ("finish", ""):
            if pose is not None:
                previous_pose = pose
            continue
        if tool not in NAV_TOOL_NAMES:
            continue
        nav_attempts += 1
        start = previous_pose
        leg = _distance(start, pose) if start is not None else None
        if leg is not None and leg >= min_progress_m:
            moved += 1
            progress_total += leg
            bearing = _bearing_deg(start, target_pos)
            heading = _bearing_deg(start, pose)
            if bearing is not None and heading is not None:
                if abs(_wrap_deg(heading - bearing)) <= 90.0:
                    toward += 1
        if pose is not None:
            previous_pose = pose
    return {
        "nav_attempts": nav_attempts,
        "moved_legs": moved,
        "progress_m": _round(progress_total),
        "toward_target_legs": toward,
        "toward_target_rate": _rate(toward, moved),
    }


def confirmation_stats(entries: list[dict], summary: dict, *, reached: bool | None,
                       gate_m: float | None, target_pos: Any) -> dict:
    """Claims vs. ground truth: the false-confirmation metric.

    A **claim** is an explicit declaration, and only that:

    - a ``finish`` turn whose ``success`` is ``true`` (in its args, or in the
      result data of a verifying finish tool), or
    - a ``verify_goal`` answered ``verified: true``.

    A bare finish - a reason and nothing else - declares nothing and is *not* a
    claim: the runtime leaves the verdict open for it, so the honest "I could
    not find it" of a give-up stops counting as a (false) confirmation. The row
    keeps that distinction in ``finish_declared`` (``True``/``False``/``None``,
    the last finish turn) and ``honest_finish``. A claim is *false* when the
    final pose is outside the geometric gate, or when the last grounded
    candidate's reported position is outside it.
    """
    finish_reason = str(summary.get("finish_reason") or "").lower()
    claims: list[dict] = []
    verify_true = verify_false = 0
    candidate_wrong: bool | None = None
    finish_declared: bool | None = None
    finishes = 0
    for index, entry in enumerate(entries):
        tool = str(entry.get("tool") or "")
        args = entry.get("args") if isinstance(entry.get("args"), dict) else {}
        data = _result_data(entry)
        turn = _entry_turn(entry, index)
        if tool == "finish":
            finishes += 1
            declared = _as_bool(args.get("success"))
            if declared is None:  # a verifying finish tool answers in the data
                declared = _as_bool(data.get("success"))
            finish_declared = declared
            if declared is True:
                claims.append({"turn": turn, "kind": "finish",
                               "why": str(args.get("reason") or "")})
        elif tool == "verify_goal":
            verified = _as_bool(data.get("verified"))
            if verified is True:
                verify_true += 1
                claims.append({"turn": turn, "kind": "verify_goal",
                               "why": str(args.get("reason") or "")})
            elif verified is False:
                verify_false += 1
        if tool in GROUND_TOOL_NAMES and data.get("candidate_id"):
            position = next((data.get(key) for key in CANDIDATE_POSITION_FIELDS
                             if data.get(key) is not None), None)
            distance = _distance(position, target_pos)
            if distance is not None and gate_m is not None:
                wrong = bool(distance > gate_m)
                candidate_wrong = wrong if candidate_wrong is None else (candidate_wrong or wrong)
    if not claims and finish_reason in CLAIM_REASONS:
        # the trace may be missing (or older than the finish contract): the
        # summary still records the finish turn and its args, and only an
        # explicit ``success: true`` is a claim - a bare finish stays honest
        finish_args = summary.get("finish_args")
        finish_args = finish_args if isinstance(finish_args, dict) else {}
        declared = _as_bool(finish_args.get("success"))
        finishes = max(finishes, 1)      # the summary's finish reason is a finish
        if finish_declared is None:
            finish_declared = declared
        if declared is True:
            claims.append({"turn": None, "kind": "finish",
                           "why": str(finish_args.get("reason") or "")})
    claim = bool(claims)
    false = bool(claim and (reached is False or candidate_wrong is True))
    honest_finish = bool(finishes and finish_declared is not True)
    return {
        "claim": claim,
        "claim_kinds": sorted({item["kind"] for item in claims}),
        "claims": claims[:8],
        "n_claims": len(claims),
        # what the finish itself said: True (claimed), False (declared failure),
        # None (a bare finish - the honest give-up of the last round's metric)
        "finish_declared": finish_declared,
        "honest_finish": honest_finish,
        "n_finishes": finishes,
        "verify_goal_verified": verify_true,
        "verify_goal_unmet": verify_false,
        "candidate_wrong": candidate_wrong,
        "false_confirmation": false,
        "false_confirmation_reason": (
            "" if not false
            else ("unreached" if reached is False else "candidate_position")
        ),
    }


def extract_metrics(args: argparse.Namespace, scenario: ScenarioSpec,
                    spec: EpisodeSpec, out_dir: Path, run: dict) -> dict:
    """All decision-loop metrics of one episode, from the runner's artifacts."""
    row = row_defaults(spec, args, out_dir)
    row["returncode"] = run.get("returncode")
    row["error"] = run.get("error")
    row["elapsed_s"] = run.get("elapsed_s")
    row["command"] = run.get("command")
    if not (out_dir / SUMMARY_FILE).exists():
        if row["error"] is None:
            row["error"] = f"missing {SUMMARY_FILE} in {out_dir}"
        _token_model(args, row, decision=None)
        return row
    try:
        summary = read_json(out_dir / SUMMARY_FILE) or {}
        entries = trace_entries(out_dir)
        row.update(_episode_metrics(args, scenario, spec, summary, entries, out_dir))
    except Exception as exc:  # a broken artifact must not lose the other arms
        row["error"] = f"metric extraction failed: {type(exc).__name__}: {exc}"
    _token_model(args, row, decision=row.get("vlm_calls_decision"))
    return row


def row_defaults(spec: EpisodeSpec, args: argparse.Namespace, out_dir: Path) -> dict:
    return {
        "arm": spec.arm,
        "arm_label": ARM_TABLE[spec.arm].label,
        "scenario": spec.scenario,
        "task": spec.task.name,
        "instruction": spec.task.instruction,
        "seed": int(spec.seed),
        "yaw_mode": spec.yaw_mode,
        "start_yaw_deg": spec.start_yaw_deg,
        "out_dir": str(out_dir),
        "command": None,
        "returncode": None,
        "elapsed_s": None,
        "error": None,
        "skipped": False,
        "success": False,
        "solved": False,
        "reached": None,
        "final_distance_m": None,
        "gate_m": None,
        # the gate ``verify_goal`` used (same number, or the claim is suspect)
        "verify_limit_m": None,
        "verify_runner_gate_m": None,
        "verify_gate_consistent": None,
        "verify_claims_past_gate": 0,
        "prompt_episode_id": None,
        "finish_reason": None,
        "switches": None,
        "declared_outer_tool_calls": 0,
        "declared_inner_tool_calls": 0,
        "declared_routine_vlm_calls": 0,
        "search_target_calls": 0,
        "claim": False,
        "finish_declared": None,
        "honest_finish": False,
        "false_confirmation": False,
        "candidate_wrong": None,
        "turns": None,
        "tool_calls": None,
        "info_calls": None,
        "vlm_calls_decision": None,
        "vlm_calls_declared": 0,
        "vlm_calls_reported": None,
        "vlm_calls_total": None,
        "vlm_calls_internal": None,
        "tokens_estimate": None,
        "images_total": None,
        "wall_time_s": None,
        "repeat_search_ratio": None,
        "repeat_search_checks": None,
        "repeat_evidence_calls": None,
        "repeat_region_calls": None,
        "coverage_source": None,
        "coverage_checked_deg": None,
        "coverage_checked_sectors": None,
        "coverage_total_sectors": None,
        "coverage_unchecked_deg": None,
        "coverage_unchecked_sectors": None,
        "coverage_worth_looking": None,
        "coverage_full_sweeps": None,
        "coverage_down_checked": None,
        "coverage_checked_deg_estimate": None,
        "direction_coverage": None,
        "direction_bins": None,
        "views_expanded": None,
        "view_index_max": None,
        "crop_checks": None,
        "nav_attempts": None,
        "progress_m": None,
        "toward_target_rate": None,
        "annotations": None,
    }


def _episode_metrics(args: argparse.Namespace, scenario: ScenarioSpec, spec: EpisodeSpec,
                     summary: dict, entries: list[dict], out_dir: Path) -> dict:
    task = spec.task
    start_pose = task.start_pos
    events = phase1.timeline(entries, phase1.CONFIG_TOOL, start_pose=start_pose)
    success = phase1.success_stats(
        args, summary, events, task.target_pos, start_pose,
        radius_m=float(task.target_radius), success_dist_m=success_dist(args, task),
    )
    cost = phase1.cost_stats(args, summary, events, phase1.CONFIG_TOOL)
    gate = success["gate_m"]
    confirmation = confirmation_stats(
        entries, summary, reached=success["reached"], gate_m=gate,
        target_pos=task.target_pos,
    )
    verify_gate = verify_gate_stats(summary, entries, gate)
    repeats = repeat_stats(entries, grid_m=float(args.grid_m))
    coverage = coverage_stats(entries, hfov_deg=float(args.hfov_deg))
    movement = movement_stats(entries, task.target_pos,
                              min_progress_m=float(args.min_progress_m))
    ledger = phase1.call_stats(events, phase1.CONFIG_TOOL)
    row = {
        "skipped": bool(summary.get("skipped")),
        "success": bool(summary.get("success")),
        "runner_success": bool(summary.get("success")),
        "solved": bool(success["solved"]),
        "reached": success["reached"],
        "final_distance_m": success["final_distance_m"],
        "gate_m": gate,
        # the completion check really used the same gate as the evaluation
        "verify_limit_m": verify_gate["verify_limit_m"],
        "verify_runner_gate_m": verify_gate["verify_runner_gate_m"],
        "verify_gate_consistent": verify_gate["verify_gate_consistent"],
        "verify_claims_past_gate": verify_gate["claims_past_gate"],
        # the identity the model saw (never the arm label); the driver checks
        # that it is neutral and identical across the arms of one cell
        "prompt_episode_id": (str(summary["prompt_episode_id"])
                              if summary.get("prompt_episode_id") else None),
        "finish_reason": summary.get("finish_reason"),
        "switches": summary.get("switches") if isinstance(summary.get("switches"), dict) else None,
        "claim": confirmation["claim"],
        "claim_kinds": confirmation["claim_kinds"],
        "claims": confirmation["claims"],
        "n_claims": confirmation["n_claims"],
        "finish_declared": confirmation["finish_declared"],
        "honest_finish": confirmation["honest_finish"],
        "false_confirmation": confirmation["false_confirmation"],
        "false_confirmation_reason": confirmation["false_confirmation_reason"],
        "candidate_wrong": confirmation["candidate_wrong"],
        "verify_goal_verified": confirmation["verify_goal_verified"],
        "verify_goal_unmet": confirmation["verify_goal_unmet"],
        "turns": _as_int(summary.get("turns")) if _as_int(summary.get("turns")) is not None
                 else len(entries),
        "tool_calls": _as_int(summary.get("tool_calls")) or len([e for e in entries if e.get("tool")]),
        "tool_ok": ledger.get("ok"),
        "trace_entries": len(entries),
        "vlm_calls_decision": cost["vlm_calls_decision"],
        "vlm_calls_reported": cost["vlm_calls_reported"],
        "vlm_calls_declared": cost["vlm_calls_internal_declared"],
        "vlm_calls_total": cost["vlm_calls_total"],
        "vlm_calls_internal": cost["vlm_calls_internal"],
        "vlm_cost_complete": cost["vlm_cost_complete"],
        "wall_time_s": cost["wall_time_s"],
        "new_evidence_total": _as_int(summary.get("new_evidence_total")),
        "parse_failures": _as_int(summary.get("parse_failures")),
        "toolset": summary.get("toolset"),
        "toolset_detail": summary.get("toolset_detail"),
        "tools_used": (summary.get("tools") or None),
    }
    row.update(declared_routine_calls(entries))
    row.update(repeats)
    row.update(coverage)
    row.update(movement)
    row["annotation"] = annotate_episode(entries, row, spec, task)
    return row


def _token_model(args: argparse.Namespace, row: dict, decision: int | None) -> None:
    """tokens_estimate = images x patch tokens + text chars / chars-per-token."""
    image_tokens = phase1.image_tokens(args)
    images = _as_int(row.get("images_total"))
    if images is None:
        images = (int(decision or 0)) * (int(args.context_frames if args.context_frames is not None else 2) + 1)
        row["images_total"] = images
    text_chars = _as_int(row.get("text_chars"))
    if text_chars is None:
        total = int(row.get("vlm_calls_total") or 0)
        text_chars = int(float(args.prompt_chars_per_call) * total)
        row["text_chars"] = text_chars
    per_token = float(args.chars_per_token) or 4.0
    row["image_tokens"] = image_tokens
    row["chars_per_token"] = float(args.chars_per_token)
    row["prompt_chars_per_call"] = float(args.prompt_chars_per_call)
    row["tokens_estimate"] = int(round(images * image_tokens + text_chars / per_token))


# ---------------------------------------------------------------------------
# decision annotation: why the viewpoint changed, what was learned, what changed
# ---------------------------------------------------------------------------


def annotate_episode(entries: list[dict], row: dict, spec: EpisodeSpec,
                     task: compare.TaskSpec) -> dict:
    """The causal chain of one episode (see the module docstring's vocabulary)."""
    steps = [step_annotation(entry, index) for index, entry in enumerate(entries)]
    introduced: dict[str, int] = {}
    events: list[dict] = []
    narrative: list[str] = []
    streak: list[int] = []  # indices of consecutive unproductive information calls
    pending_trigger: dict | None = None

    for index, step in enumerate(steps):
        tool = step["tool"]
        status = step["status"]
        data_ids = step["ids"]
        event_start = len(events)
        found_candidate = bool(step["candidate_id"]) and tool in GROUND_TOOL_NAMES \
            and status not in UNPRODUCTIVE_STATUS
        nav_failed = tool in NAV_TOOL_NAMES and (
            status in UNPRODUCTIVE_STATUS or _nav_failed(step)
        )
        unproductive = nav_failed or tool in INFO_TOOL_NAMES and (
            status in UNPRODUCTIVE_STATUS
            or (tool in GROUND_TOOL_NAMES and step["new_evidence"] is False
                and not found_candidate)
        )

        # a call that follows an unproductive streak with a different tool is
        # the strategy change the recovery layer is about
        if streak and pending_trigger is None and tool != steps[streak[-1]]["tool"]:
            pending_trigger = {
                "kind": "strategy_change",
                "turn": step["turn"],
                "streak": [steps[i]["turn"] for i in streak],
                "after": steps[streak[-1]]["tool"],
                "marker": _marker_of(steps[streak[-1]]),
                "why": step["why"],
                "action": tool,
            }

        if tool in NAV_TOOL_NAMES:
            event = {
                "kind": "viewpoint_change" if tool in VIEWPOINT_TOOL_NAMES else "move",
                "turn": step["turn"],
                "action": tool,
                "referenced": _id_list(step["args"]),
                "new_info_before": [item for item in introduced
                                    if introduced[item] < index][-3:],
                "why": step["why"],
                "progress_kind": step["progress_kind"],
            }
            if nav_failed:
                event["kind"] = "move_failed"
                event["marker"] = _marker_of(step)
                streak.append(index)  # a failed leg is an unproductive call too
            events.append(event)
        elif unproductive:
            streak.append(index)
            events.append({
                "kind": "no_candidate" if tool in GROUND_TOOL_NAMES else "unproductive",
                "turn": step["turn"],
                "action": tool,
                "marker": _marker_of(step),
                "why": step["why"],
                "progress_kind": step["progress_kind"],
                "new_evidence": step["new_evidence"],
            })
        elif found_candidate:
            introduced.setdefault(step["candidate_id"], index)
            event = {
                "kind": "candidate_found",
                "turn": step["turn"],
                "action": tool,
                "candidate_id": step["candidate_id"],
                "observation_id": step["observation_id"],
                "why": step["why"],
            }
            if step["steps"]:  # the search skill found it: keep its step log
                event["search_steps"] = [_search_step_text(item) for item in step["steps"]]
                event["search_vlm_calls"] = step["vlm_calls"]
            events.append(event)
        elif tool in TASK_TOOL_NAMES:
            events.append({"kind": "task", "turn": step["turn"], "action": tool,
                           "why": step["why"]})
        elif tool == "verify_goal":
            verified = step["verify_verified"]
            events.append({
                "kind": "claim" if verified is True else "verification_unmet",
                "turn": step["turn"], "action": tool,
                "verified": verified, "why": step["why"],
            })
            if verified is True:
                introduced.setdefault(f"claim@{step['turn']}", index)
        elif tool == "finish":
            # only an explicit success:true is a claim; a bare finish declares
            # nothing (honest give-up), success:false declares the failure
            events.append({
                "kind": "claim" if step["success"] is True else "finish_no_claim",
                "turn": step["turn"], "action": tool, "why": step["why"],
                "declared": step["success"],
            })
        elif step["new_evidence"] is True or step["steps"]:
            note = {
                "kind": "information",
                "turn": step["turn"], "action": tool,
                "observation_id": step["observation_id"],
                "view_index": step["view_index"],
                "coverage": step["coverage"],
                "headings_deg": step["headings_deg"],
                "image_refs": step["image_refs"],
                "why": step["why"],
                "progress_kind": step["progress_kind"],
            }
            if step["steps"]:
                note["search_steps"] = [_search_step_text(item) for item in step["steps"]]
                note["search_vlm_calls"] = step["vlm_calls"]
            events.append(note)
        elif tool in ("", "(no tool call)"):
            events.append({"kind": "no_tool_call", "turn": step["turn"]})
        else:
            events.append({
                "kind": "repeat" if step["new_evidence"] is False else "call",
                "turn": step["turn"], "action": tool,
                "status": status, "why": step["why"],
                "progress_kind": step["progress_kind"],
            })

        for item in data_ids:
            introduced.setdefault(item, index)
        for item in step["args"].values():
            for match in _id_list(item):
                introduced.setdefault(match, index)
        if pending_trigger is not None:
            pending_trigger["new_info_after"] = _ids_after(steps, index)
            pending_trigger["consumed_by"] = _consumed_by(steps, index, introduced)
            events.insert(event_start, pending_trigger)  # the reason precedes the call
            pending_trigger = None
            streak = []
        elif not unproductive:
            # a productive call or a movement ends the unproductive streak
            streak = []

    # what used each newly introduced id: "this new view was used by t8 ..."
    used_by: list[dict] = []
    for item, index in sorted(introduced.items(), key=lambda pair: pair[1]):
        if item.startswith("claim@"):
            continue
        user = _first_user(steps, index + 1, item)
        if user is not None:
            used_by.append({"id": item, "introduced_turn": steps[index]["turn"],
                            "used_turn": user["turn"], "used_by": user["tool"]})

    narrative = _narrative(steps, events, row, task, used_by)
    counters = {
        "steps": len(steps),
        "information_calls": sum(1 for step in steps if step["tool"] in INFO_TOOL_NAMES),
        "nav_calls": sum(1 for step in steps if step["tool"] in NAV_TOOL_NAMES),
        "strategy_changes": sum(1 for event in events if event["kind"] == "strategy_change"),
        "viewpoint_changes": sum(1 for event in events if event["kind"] == "viewpoint_change"),
        "candidates_found": sum(1 for event in events if event["kind"] == "candidate_found"),
        "repeats": sum(1 for event in events if event["kind"] == "repeat"),
        "progress_kind_reported": sum(1 for step in steps
                                      if step["progress_kind_source"] == "reported"),
        "search_skill_calls": sum(1 for step in steps if step["steps"]),
    }
    return {
        "episode": f"{spec.arm}|{spec.scenario}|{spec.task.name}|{spec.slug}",
        "steps": steps,
        "events": events,
        "used_by": used_by,
        "narrative": narrative,
        "summary": _chain_summary(steps, events, row),
        "counters": counters,
    }


def _marker_of(step: dict) -> str:
    for source in (step.get("reason"), step.get("outcome"), step.get("detail"),
                   step.get("status")):
        text = str(source or "").strip()
        if text:
            return text.split(":")[0].strip()[:40]
    return step.get("status") or "unknown"


def _nav_failed(step: dict) -> bool:
    outcome = str(step.get("outcome") or "").lower()
    progress = None
    for key in PROGRESS_FIELDS:
        progress = _as_float(step["args"].get(key))
        if progress is not None:
            break
    if progress is not None:
        return progress <= 0.0
    return any(token in outcome for token in phase1.ZERO_DISPLACEMENT_MARKERS) \
        or "fail" in outcome or "blocked" in outcome


def _ids_after(steps: list[dict], index: int) -> list[str]:
    ids: list[str] = []
    for step in steps[index + 1: index + 3]:
        for item in (step["observation_id"], step["candidate_id"], step["viewpoint_id"]):
            if item and item not in ids:
                ids.append(item)
        for item in (step["image_refs"] or []):
            if item not in ids:
                ids.append(item)
    return ids


def _consumed_by(steps: list[dict], index: int, introduced: dict[str, int]) -> dict | None:
    produced = {item for item, origin in introduced.items() if origin == index}
    for later in steps[index + 1:]:
        for item in produced:
            if item in later["ids"] or item in _id_list(later["args"]):
                return {"turn": later["turn"], "tool": later["tool"], "id": item}
    return None


def _first_user(steps: list[dict], start: int, item: str) -> dict | None:
    for step in steps[start:]:
        if item in _id_list(step["args"]) or item in (step["ids"] or []):
            return step
    return None


def _search_step_text(step: Any) -> str:
    if not isinstance(step, dict):
        return str(step)
    parts = []
    number = step.get("step")
    if number is not None:
        parts.append(f"{number})")
    if step.get("why"):
        parts.append(str(step["why"]))
    if step.get("obs_ids"):
        parts.append("obs=" + ",".join(str(item) for item in step["obs_ids"]))
    if step.get("result"):
        parts.append("-> " + str(step["result"]))
    return " ".join(parts) or json.dumps(step, default=str)[:80]


def _narrative(steps: list[dict], events: list[dict], row: dict,
               task: compare.TaskSpec, used_by: list[dict]) -> list[str]:
    """Human-readable causal chain, one line per event."""
    by_turn: dict[Any, list[dict]] = {}
    for event in events:
        by_turn.setdefault(event.get("turn"), []).append(event)
    used_at = {item["used_turn"]: item for item in used_by}
    lines: list[str] = []
    for step in steps:
        turn = step["turn"]
        head = f"t{turn} {step['tool']}"
        if step["why"]:
            head += f" ({step['why'][:80]})"
        detail: list[str] = []
        for event in by_turn.get(turn, []):
            kind = event["kind"]
            if kind == "strategy_change":
                detail.append(
                    f"STRATEGY CHANGE after {len(event['streak'])} unproductive "
                    f"{event['after']} call(s) ({event['marker']}) -> {event['action']}"
                )
            elif kind == "no_candidate":
                detail.append(f"no_candidate ({event['marker']})")
            elif kind == "unproductive":
                detail.append(f"unproductive ({event['marker']})")
            elif kind == "candidate_found":
                note = f"candidate {event['candidate_id']} found"
                if event.get("search_steps"):
                    note += " | steps: " + "; ".join(event["search_steps"])
                if event.get("search_vlm_calls"):
                    note += f" | internal vlm_calls={event['search_vlm_calls']}"
                detail.append(note)
            elif kind == "information":
                pieces = []
                if event.get("observation_id"):
                    pieces.append(str(event["observation_id"]))
                index_value = event.get("view_index")
                if isinstance(index_value, (list, tuple)):
                    pieces.append(f"view_index: {len(index_value)} view(s)")
                else:
                    if index_value is not None:
                        pieces.append(f"view_index={index_value}")
                    if event.get("image_refs"):
                        pieces.append(f"image_refs={len(event['image_refs'])}")
                if event.get("headings_deg"):
                    pieces.append(f"headings={len(event['headings_deg'])}")
                if isinstance(event.get("coverage"), dict):
                    checked_deg, sectors, _ = _sector_span(event["coverage"],
                                                           COVERAGE_CHECKED_KEYS)
                    if checked_deg is not None:
                        pieces.append(f"coverage checked={checked_deg:g}deg"
                                      + (f" ({sectors} sectors)" if sectors else ""))
                note = "new information: " + ", ".join(pieces) if pieces else "new information"
                if event.get("search_steps"):
                    note += " | steps: " + "; ".join(event["search_steps"])
                if event.get("search_vlm_calls"):
                    note += f" | internal vlm_calls={event['search_vlm_calls']}"
                detail.append(note)
            elif kind in ("viewpoint_change", "move"):
                note = kind
                if event.get("referenced"):
                    note += " -> " + ",".join(str(item) for item in event["referenced"][:2])
                if event.get("new_info_before"):
                    note += f" (new info so far: {','.join(event['new_info_before'][:2])})"
                detail.append(note)
            elif kind == "move_failed":
                detail.append(f"leg failed ({event.get('marker')})")
            elif kind == "claim":
                detail.append(
                    "confirmation (verified=true)" if event.get("action") == "verify_goal"
                    else "CLAIM success/confirmation"
                )
            elif kind == "finish_no_claim":
                detail.append(
                    "honest finish (no success declared)"
                    if event.get("declared") is None
                    else "declared failure (success=false)"
                )
            elif kind == "verification_unmet":
                detail.append(f"verification unmet (verified={event.get('verified')})")
            elif kind == "repeat":
                detail.append("repeat check (no new evidence)")
            elif kind == "task":
                detail.append("task bookkeeping")
        if step["progress_kind"] and step["progress_kind_source"] == "reported":
            detail.append(f"progress_kind={step['progress_kind']}")
        if turn in used_at:
            item = used_at[turn]
            detail.append(f"uses {item['id']} (introduced t{item['introduced_turn']})")
        lines.append(f"{head} [{step['status']}] -> " + ("; ".join(detail) if detail else "ok"))
    verdict = []
    if row.get("claim"):
        verdict.append("CLAIMED success")
    if row.get("honest_finish"):
        declared = row.get("finish_declared")
        verdict.append("honest give-up (no success claimed)"
                       if declared is None else "declared FAILURE (success=false)")
    if row.get("reached") is False:
        verdict.append(f"did NOT reach the gate ({row.get('gate_m')} m)")
    elif row.get("reached") is True:
        verdict.append("reached the gate")
    if row.get("false_confirmation"):
        verdict.append(f"FALSE CONFIRMATION ({row.get('false_confirmation_reason')})")
    if verdict:
        lines.append("verdict: " + "; ".join(verdict))
    lines.append(f"chain: {_chain_summary(steps, events, row)}")
    return lines


def _chain_summary(steps: list[dict], events: list[dict], row: dict) -> str:
    """One compact causal sentence per episode."""
    pieces: list[str] = []
    for event in events:
        kind = event["kind"]
        if kind == "information":
            label = "obs" if event.get("observation_id") else "info"
            index_value = event.get("view_index")
            if isinstance(index_value, (list, tuple)):
                label += f"({len(index_value)} views)"
            elif index_value is not None:
                label += f"(view {index_value})"
            pieces.append(label)
        elif kind == "strategy_change":
            pieces.append(f"!{len(event['streak'])}x{event['after']}->{event['action']}")
        elif kind == "candidate_found":
            pieces.append(f"cand({event['candidate_id']})")
        elif kind == "viewpoint_change":
            pieces.append("viewpoint")
        elif kind == "move":
            pieces.append(event["action"])
        elif kind == "move_failed":
            pieces.append(f"!{event['action']}")
        elif kind == "claim":
            pieces.append("claim")
        elif kind == "finish_no_claim":
            pieces.append("honest-finish" if event.get("declared") is None
                          else "declared-failure")
        elif kind == "repeat":
            pieces.append("repeat")
    tail = ""
    if row.get("false_confirmation"):
        tail = " => FALSE CONFIRMATION"
    elif row.get("solved"):
        tail = " => solved"
    elif row.get("claim"):
        tail = " => claim without reach"
    return " -> ".join(pieces[:14]) + tail


# ---------------------------------------------------------------------------
# aggregation, layer deltas and the report
# ---------------------------------------------------------------------------


def verify_gate_stats(summary: dict, entries: list[dict], gate_m: float | None) -> dict:
    """The distance gate ``verify_goal`` used, next to the evaluation gate.

    A completion check is only evidence when it checks what the experiment
    scores: the v3 decision-loop run's four book_near claims came from the
    tool's own ``max(1.5, radius + 0.5)`` default while the episode was
    evaluated against ``--success-dist + radius`` (1.2 m) - a threshold gap, not
    a model error, and one the trace could not have revealed per episode.

    ``verify_limit_m`` is the limit the traces recorded
    (``checks.distance_limit_m``, or the ``max_m`` of an unmet distance check),
    ``runner_gate_m`` the gate the runner reports it handed to the tools
    (``summary.done_dist_m``). ``verify_gate_consistent`` is True when every
    reported number equals the evaluation gate, False when one does not, and
    None when the artifacts predate the report (unknown - never "consistent").
    A ``verify_goal`` answered ``verified: true`` from beyond the gate is a
    mismatch *by construction* - the check must have used a larger limit - so
    the v3-style artifacts are flagged even without the recorded limit.
    """
    limits: set[float] = set()
    claims_past_gate = 0
    for entry in entries:
        if str(entry.get("tool") or "") != "verify_goal":
            continue
        data = _result_data(entry)
        checks = data.get("checks") if isinstance(data.get("checks"), dict) else {}
        limit = _num(checks.get("distance_limit_m"))
        if limit is not None:
            limits.add(round(limit, 3))
        unmet = data.get("unmet") if isinstance(data.get("unmet"), list) else []
        for item in unmet:
            if isinstance(item, dict) and item.get("check") == "distance_to_target_m":
                value = _num(item.get("max_m"))
                if value is not None:
                    limits.add(round(value, 3))
        if _as_bool(data.get("verified")) is True and gate_m is not None:
            distance = _num(checks.get("distance_to_target_m"))
            if distance is not None and distance > float(gate_m) + 1.0e-3:
                claims_past_gate += 1
    verify_limit = next(iter(limits)) if len(limits) == 1 else None
    runner_gate = _num(summary.get("done_dist_m"))
    if runner_gate is not None:
        runner_gate = round(runner_gate, 3)
    if claims_past_gate:
        consistent: bool | None = False
    elif gate_m is None or (verify_limit is None and runner_gate is None):
        consistent = None
    else:
        reported = [value for value in (verify_limit, runner_gate) if value is not None]
        consistent = all(abs(value - float(gate_m)) <= 1.0e-3 for value in reported)
    return {
        "verify_limit_m": verify_limit,
        "verify_runner_gate_m": runner_gate,
        "verify_gate_consistent": consistent,
        "verify_calls": len(limits),
        "claims_past_gate": claims_past_gate,
    }


def prompt_identity_check(rows: list[dict], arms: list[str]) -> dict:
    """The arms of one cell must show the model the same, arm-free episode id.

    The episode id is written into every prompt's first line ("Episode ep_... |
    turn ..."). The E2 rounds spelled the mode into it
    (``m2_memory_000_00``), which is a constant, prompt-visible difference
    between the arms of one cell and nothing to do with the mechanism under
    test - under greedy decoding it can move the first decision. The check reads
    the identity the runner reports as ``summary.prompt_episode_id``: it must be
    present, must not contain an arm name, and must be the same for every arm of
    a cell. A missing report stays an issue (unknown is not neutral).
    """
    known_arms = [str(arm) for arm in arms]
    cells: dict[str, dict[str, dict]] = {}
    issues: list[dict] = []
    checked = 0
    for row in rows:
        if row.get("error") or row.get("skipped"):
            continue
        arm = str(row.get("arm"))
        cell = "{}|{}|{}|{}".format(row.get("scenario"), row.get("task"),
                                   row.get("yaw_mode"), row.get("seed"))
        prompt_id = row.get("prompt_episode_id")
        entry = {"cell": cell, "arm": arm, "prompt_episode_id": prompt_id}
        checked += 1
        if not prompt_id:
            issues.append({**entry, "reason": "prompt_id_missing"})
        else:
            leaked = next((name for name in known_arms
                           if name and name in str(prompt_id)), None)
            if leaked is not None:
                issues.append({**entry, "reason": "arm_name_in_prompt_id",
                               "arm_name": leaked})
        cells.setdefault(cell, {})[arm] = entry
    for cell, per_arm in sorted(cells.items()):
        seen = {str(entry["prompt_episode_id"]) for entry in per_arm.values()
                if entry["prompt_episode_id"]}
        if len(seen) > 1:
            for arm, entry in sorted(per_arm.items()):
                issues.append({"cell": cell, "arm": arm,
                               "prompt_episode_id": entry["prompt_episode_id"],
                               "reason": "prompt_id_differs_across_arms"})
    return {
        "cells": len(cells),
        "episodes_checked": checked,
        "arms": known_arms,
        "consistent": not issues,
        "issues": issues[:20],
    }


def verify_gate_summary(rows: list[dict]) -> dict:
    """Episode counts of the verify-gate check (see :func:`verify_gate_stats`)."""
    ok = [row for row in rows if not row.get("error") and not row.get("skipped")]
    checked = [row for row in ok if row.get("verify_gate_consistent") is not None]
    mismatch = [row for row in checked if row.get("verify_gate_consistent") is False]
    return {
        "episodes": len(ok),
        "checked": len(checked),
        "consistent": len(checked) - len(mismatch),
        "mismatch": len(mismatch),
        "claims_past_gate": sum(int(row.get("verify_claims_past_gate") or 0) for row in ok),
        "unreported": len(ok) - len(checked),
        "mismatch_cells": sorted(
            "{}|{}|{}|{}|{}".format(row.get("arm"), row.get("scenario"), row.get("task"),
                                    row.get("yaw_mode"), row.get("seed"))
            for row in mismatch
        )[:20],
    }


def verify_gate_line(gate: dict) -> str:
    """The report line for the verify/evaluation gate identity."""
    if not gate.get("mismatch") and not gate.get("unreported"):
        return (f"- verify gate: {gate['consistent']}/{gate['checked']} episodes report "
                f"`verify_goal` checking the evaluation gate (`--success-dist + radius`)"
                + (f" ({gate['unreported']} episode(s) without the report)"
                   if gate.get("unreported") else ""))
    parts = [f"- verify gate: **{gate['mismatch']}/{gate['checked']}** checked episode(s) "
             f"used a gate other than the evaluation gate"]
    if gate.get("claims_past_gate"):
        parts.append(f"  - {gate['claims_past_gate']} `verify_goal` claim(s) sit past the "
                     "gate: the check passed at a distance the episode is not credited "
                     "for (a threshold artifact, not evidence)")
    if gate.get("mismatch_cells"):
        parts.append("  - " + ", ".join(f"`{cell}`" for cell in gate["mismatch_cells"]))
    if gate.get("unreported"):
        parts.append(f"  - {gate['unreported']} episode(s) predate the gate report "
                     "(unknown, not consistent)")
    return "\n".join(parts)


def prompt_identity_line(check: dict) -> str:
    """The report line for the prompt-hygiene check."""
    if check.get("consistent"):
        return (f"- prompt identity: {check['episodes_checked']} episodes in "
                f"{check['cells']} cells, every arm sees the same arm-free episode id "
                "(`ep_<seed>`) - the arm name never reaches the prompt")
    issues = check.get("issues") or []
    first = issues[0] if issues else {}
    return (f"- prompt identity: **NOT clean** ({len(issues)} issue(s) shown, first: "
            f"{first.get('reason')} in `{first.get('cell')}` / `{first.get('arm')}` = "
            f"`{first.get('prompt_episode_id')}`) - an arm difference of this run is "
            "confounded by the prompt")


def aggregate(rows: list[dict]) -> dict:
    ok = [row for row in rows if not row.get("error") and not row.get("skipped")]
    claims = [row for row in ok if row.get("claim")]
    false_claims = [row for row in claims if row.get("false_confirmation")]
    honest = [row for row in ok if row.get("honest_finish")]
    reasons: dict[str, int] = {}
    for row in ok:
        reason = str(row.get("finish_reason") or "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1
    stats = {
        "n": len(rows),
        "n_ok": len(ok),
        "n_errors": sum(1 for row in rows if row.get("error")),
        "n_skipped": sum(1 for row in rows if row.get("skipped")),
        "n_solved": sum(1 for row in ok if row.get("solved")),
        "n_success": sum(1 for row in ok if row.get("success")),
        "n_claims": len(claims),
        "n_false_confirmation": len(false_claims),
        "n_honest_finish": len(honest),
        "finish_reasons": reasons,
    }
    stats["solved_rate"] = _rate(stats["n_solved"], len(ok))
    stats["success_rate"] = _rate(stats["n_success"], len(ok))
    stats["false_confirmation_rate"] = _rate(len(false_claims), len(claims))
    stats["honest_finish_rate"] = _rate(len(honest), len(ok))
    gate_checked = [row for row in ok if row.get("verify_gate_consistent") is not None]
    gate_mismatch = [row for row in gate_checked if row.get("verify_gate_consistent") is False]
    stats["n_verify_gate_checked"] = len(gate_checked)
    stats["n_verify_gate_mismatch"] = len(gate_mismatch)
    stats["verify_gate_consistent_rate"] = _rate(len(gate_checked) - len(gate_mismatch),
                                                 len(gate_checked))
    for key in (
        "vlm_calls_total", "vlm_calls_decision", "vlm_calls_declared", "tokens_estimate",
        "wall_time_s", "repeat_search_ratio", "repeat_search_checks", "repeat_evidence_calls",
        "repeat_region_calls", "info_calls", "coverage_checked_deg", "direction_coverage",
        "coverage_checked_sectors", "coverage_worth_looking", "views_expanded",
        "crop_checks", "nav_attempts", "toward_target_rate", "progress_m",
        "final_distance_m", "tool_calls",
        # what the tools declare about their own sub-routines (the search skill):
        # one outer call per search, the inner steps and model calls it ran
        "declared_outer_tool_calls", "declared_inner_tool_calls",
        "declared_routine_vlm_calls", "search_target_calls",
    ):
        values = [_as_float(row.get(key)) for row in ok]
        values = [value for value in values if value is not None]
        stats[f"{key}_mean"] = _mean(values)
        stats[f"{key}_sum"] = _sum(values) if key in (
            "vlm_calls_total", "tokens_estimate", "wall_time_s", "repeat_search_checks",
            "crop_checks", "views_expanded", "declared_outer_tool_calls",
            "declared_inner_tool_calls", "declared_routine_vlm_calls", "search_target_calls",
        ) else None
    return stats


def grouping(rows: list[dict], scenarios: list[str], yaw_modes: list[str],
             arms: list[str]) -> list[tuple[str, str, list[dict]]]:
    out: list[tuple[str, str, list[dict]]] = []
    for arm in arms:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        if not arm_rows:
            continue
        out.append((arm, "all", arm_rows))
        for scenario in scenarios:
            subset = [row for row in arm_rows if row.get("scenario") == scenario]
            if subset:
                out.append((arm, scenario, subset))
        for mode in yaw_modes:
            subset = [row for row in arm_rows if row.get("yaw_mode") == mode]
            if subset:
                out.append((arm, f"yaw:{mode}", subset))
    return out


#: delta metrics of the layer comparison (paired per cell)
LAYER_METRICS = (
    "solved", "success", "false_confirmation", "repeat_search_ratio",
    "coverage_checked_deg", "direction_coverage", "vlm_calls_total",
    "tokens_estimate", "wall_time_s",
)


def layer_deltas(rows: list[dict], arms: list[str]) -> list[dict]:
    """What each added layer bought, on the *paired* cells.

    For every cell (scenario, task, seed, yaw mode) present in both arms the
    driver subtracts the earlier arm from the later one, so the result is a
    per-cell delta, not a difference of two different populations.
    """
    usable = [arm for arm in arms if not ARM_TABLE[arm].reserved]
    by_cell: dict[str, dict[str, dict]] = {}
    for row in rows:
        if row.get("error") or row.get("skipped") or row.get("arm") not in usable:
            continue
        cell = f"{row['scenario']}|{row['task']}|{row['seed']}|{row['yaw_mode']}"
        by_cell.setdefault(cell, {})[row["arm"]] = row
    out: list[dict] = []
    for previous, current in zip(usable, usable[1:]):
        paired = [cells for cells in by_cell.values()
                  if previous in cells and current in cells]
        for metric in LAYER_METRICS:
            deltas: list[float] = []
            for cells in paired:
                before = _num(cells[previous].get(metric))
                after = _num(cells[current].get(metric))
                if before is None or after is None:
                    continue
                deltas.append(after - before)
            wins = sum(1 for value in deltas if value > 0)
            losses = sum(1 for value in deltas if value < 0)
            out.append({
                "layer": f"{previous}->{current}",
                "from": previous,
                "to": current,
                "metric": metric,
                "n_cells": len(deltas),
                "mean_delta": _round(sum(deltas) / len(deltas)) if deltas else None,
                "sum_delta": _round(sum(deltas)) if deltas else None,
                "wins": wins,
                "losses": losses,
                "ties": len(deltas) - wins - losses,
            })
    return out


def render_markdown(meta: dict, rows: list[dict], groups: list[tuple[str, str, list[dict]]],
                    aggregates: dict, deltas: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Decision-loop comparison: baseline vs. vision context vs. "
                 "recovery vs. memory")
    lines.append("")
    lines.append(f"- created: {meta['created_at']}")
    lines.append(f"- git HEAD: {meta.get('git_head') or 'unknown'}")
    lines.append(f"- backend: {meta['backend']}"
                 + (f" | model: {meta['model_path'] or meta['model']}" if (meta['model_path'] or meta['model']) else ""))
    lines.append(f"- scenarios: {', '.join(item['name'] for item in meta['scenarios'])}")
    for scenario in meta["scenarios"]:
        lines.append(f"  - {scenario['name']}: world={scenario['world']} "
                     f"ply={scenario['ply'] or '-'} task_file={scenario['task_file'] or '-'} "
                     f"({scenario['note']})")
    lines.append(f"- arms: {', '.join(record['name'] for record in meta['arms'])} | "
                 f"seeds: {meta['seeds']} | "
                 f"initial-yaw groups: {meta['yaw_modes']}")
    lines.append(f"- budget: --max-vlm-calls {meta['budget']['max_vlm_calls']} "
                 f"(tool turns {meta['budget']['tool']['max_turns']}, "
                 f"tool budget {meta['budget']['tool']['tool_budget']})")
    lines.append(f"- pairing check: {meta['pairing_check']['cells']} cells, "
                 f"consistent={meta['pairing_check']['consistent']} "
                 f"(arm-only options: {', '.join(meta['pairing_check']['arm_only_options']) or '-'})")
    prompt_identity = meta.get("prompt_identity_check") or {}
    if prompt_identity:
        lines.append("")
        lines.append(prompt_identity_line(prompt_identity))
    lines.append("")
    lines.append("## Arms and switches")
    lines.append("")
    lines.append("| arm | label | runner switches | support | dropped | note |")
    lines.append("|" + "---|" * 6)
    for record in meta["arm_support"]:
        lines.append(
            f"| {record['arm']} | {ARM_TABLE[record['arm']].label} | "
            f"{' '.join(record['effective']) or '-'} | {record['support']} | "
            f"{', '.join(record['dropped']) or '-'} | {ARM_TABLE[record['arm']].note} |"
        )
    degraded = [record for record in meta["arm_support"] if record["dropped"]]
    for record in degraded:
        lines.append("")
        lines.append(f"> `{record['arm']}` is **degraded**: {record['reason']}. "
                     "Its episodes ran the subset above, so the layer it stands for "
                     "was not really enabled.")
    mechanisms = meta.get("mechanism_check") or []
    if mechanisms:
        lines.append("")
        lines.append("What the runner summaries report as really applied per arm "
                     "(`summary.json.switches.applied`; `?` = the memory did not "
                     "attest the layer):")
        lines.append("")
        lines.append("| arm | episodes | toolset | recovery | state_block | search_skill | "
                     "spatial_memory | episodic_memory | runtime | consistent |")
        lines.append("|" + "---|" * 10)
        for record in mechanisms:
            applied = record["applied"]
            lines.append(
                f"| {record['arm']} | {record['n_reported']}/{record['n_episodes']} | "
                f"{record.get('toolset') or '-'} | "
                f"{_tri_state(applied.get('recovery'))} | "
                f"{_tri_state(applied.get('state_block'))} | "
                f"{_tri_state(applied.get('search_skill'))} | "
                f"{_tri_state(applied.get('spatial_memory'))} | "
                f"{_tri_state(applied.get('episodic_memory'))} | "
                f"{record.get('runtime') or '-'} | "
                f"{'yes' if record['consistent'] else 'no'} |"
            )
        for record in mechanisms:
            asked_off = [
                name for name in ("spatial_memory", "episodic_memory")
                if record["requested"].get(name) == "off"
                and record["applied"].get(name) is None
            ]
            if record["unattested"]:
                lines.append("")
                lines.append(
                    f"> `{record['arm']}` asked for "
                    + ", ".join(f"`{name}=on`" for name in record["unattested"])
                    + " but the built memory did not confirm "
                    + ("it" if len(record["unattested"]) == 1 else "them")
                    + ": the run reads `?` (unknown), not `on` - the memory in use "
                      "does not implement that layer."
                )
            elif asked_off:
                lines.append("")
                lines.append(
                    f"> `{record['arm']}` asked for "
                    + ", ".join(f"`{name}=off`" for name in asked_off)
                    + " and the built memory did not confirm the layer either: the "
                      "conservative reading (`off`) is what the arm claims, the "
                      "switch itself is unattested."
                )
        for record in mechanisms:
            if record["consistent"]:
                continue
            details = []
            if record["disagreements"]:
                details.append("episodes disagree on " + ", ".join(record["disagreements"]))
            if record["n_missing"]:
                details.append(f"{record['n_missing']} episode(s) reported no switches block")
            lines.append("")
            lines.append(f"> `{record['arm']}` did **not** run a consistent switch set: "
                         + "; ".join(details) + ". Read its layer delta with care.")
    gate = meta.get("verify_gate") or {}
    if gate:
        lines.append("")
        lines.append(verify_gate_line(gate))
    lines.append("")
    lines.append("## Task success and cost by arm")
    lines.append("")
    lines.append("| arm | group | ep | solved | success | claims | false_confirm (rate) | "
                 "honest_finish | repeat_search | info_calls | search (out/in) | coverage | "
                 "vlm_calls | tokens | wall_s |")
    lines.append("|" + "---|" * 15)
    for arm, group, _ in groups:
        stats = aggregates[f"{arm}|{group}"]
        lines.append(
            f"| {arm} | {group} | {stats['n']} | "
            f"{stats['n_solved']}/{stats['n_ok']} ({_fmt(stats['solved_rate'])}) | "
            f"{stats['n_success']}/{stats['n_ok']} | {stats['n_claims']} | "
            f"{stats['n_false_confirmation']} ({_fmt(stats['false_confirmation_rate'])}) | "
            f"{stats['n_honest_finish']} | "
            f"{_fmt(stats['repeat_search_ratio_mean'])} "
            f"[{_fmt(stats['repeat_search_checks_mean'], 1)}] | "
            f"{_fmt(stats['info_calls_mean'], 1)} | "
            f"{_fmt(stats['declared_outer_tool_calls_sum'], 0)}/"
            f"{_fmt(stats['declared_inner_tool_calls_sum'], 0)} | "
            f"{_fmt(stats['coverage_checked_deg_mean'], 0)} | "
            f"{_fmt(stats['vlm_calls_total_mean'], 1)} | "
            f"{_fmt(stats['tokens_estimate_mean'], 0)} | "
            f"{_fmt(stats['wall_time_s_mean'], 1)} |"
        )
    lines.append("")
    lines.append("`claims` counts the episodes whose finish declared `success: true` "
                 "(or whose `verify_goal` was verified); `honest_finish` counts the "
                 "episodes that ended with an undeclared finish or an explicit "
                 "`success: false` - a give-up, not a claim. "
                 "`repeat_search` = repeat checks / information calls "
                 "(the bracketed value is the mean count of repeat checks); "
                 "`coverage` = reported checked degrees when the observation contract "
                 "carries them, else the estimated heading coverage; "
                 "`search (out/in)` = the search-skill calls of the arm and the routine "
                 "steps they ran inside (one search costs the episode one tool call - "
                 "the inner steps run on the routine's own allowance).")
    lines.append("")
    lines.append("## Layer contributions (paired cells)")
    lines.append("")
    lines.append("| layer | metric | cells | mean delta | sum | wins | losses | ties |")
    lines.append("|" + "---|" * 8)
    for record in deltas:
        lines.append(
            f"| {record['layer']} | {record['metric']} | {record['n_cells']} | "
            f"{_fmt(record['mean_delta'])} | {_fmt(record['sum_delta'])} | "
            f"{record['wins']} | {record['losses']} | {record['ties']} |"
        )
    lines.append("")
    lines.append("## Per-episode metrics")
    lines.append("")
    lines.append("| arm | scenario | task | yaw | seed | solved | success | claim | "
                 "finish_decl | honest | false_confirm | "
                 "finish | repeat_search | coverage(deg) | views | crops | vlm_calls | tokens | wall_s | notes |")
    lines.append("|" + "---|" * 20)
    for row in rows:
        notes = row.get("error") or ("skipped (reserved arm)" if row.get("skipped") else "")
        coverage = row.get("coverage_checked_deg")
        if coverage is None:
            coverage = row.get("coverage_checked_deg_estimate")
        declared = row.get("finish_declared")
        lines.append(
            f"| {row['arm']} | {row['scenario']} | {row['task']} | {row['yaw_mode']} | "
            f"{row['seed']} | {int(bool(row.get('solved')))} | {int(bool(row.get('success')))} | "
            f"{int(bool(row.get('claim')))} | "
            f"{'-' if declared is None else int(bool(declared))} | "
            f"{int(bool(row.get('honest_finish')))} | "
            f"{int(bool(row.get('false_confirmation')))} | {row.get('finish_reason') or '-'} | "
            f"{_fmt(row.get('repeat_search_ratio'))} | {_fmt(coverage, 0)} | "
            f"{_fmt(row.get('views_expanded'), 0)} | {_fmt(row.get('crop_checks'), 0)} | "
            f"{_fmt(row.get('vlm_calls_total'), 0)} | {_fmt(row.get('tokens_estimate'), 0)} | "
            f"{_fmt(row.get('wall_time_s'), 1)} | {notes} |"
        )
    lines.append("")
    lines.append("## Decision annotation (why the viewpoint changed, what changed next)")
    lines.append("")
    for row in rows:
        annotation = row.get("annotation")
        if not annotation:
            continue
        lines.append(f"### {annotation['episode']}")
        lines.append("")
        lines.append(f"- outcome: solved={bool(row.get('solved'))} "
                     f"success={bool(row.get('success'))} "
                     f"claim={bool(row.get('claim'))} "
                     f"false_confirmation={bool(row.get('false_confirmation'))} "
                     f"finish={row.get('finish_reason')}")
        lines.append(f"- counters: {json.dumps(annotation['counters'], sort_keys=True)}")
        lines.append("")
        for line in annotation["narrative"]:
            lines.append(f"  - {line}")
        lines.append("")
    lines.append("## Metric definitions")
    lines.append("")
    lines.append("- `solved`: the runner's own verdict (a terminating finish) AND the "
                 "last traced pose inside `--success-dist + target radius`; *both* are "
                 "required, exactly as `verify_phase1.success_stats` computes it. Reaching "
                 "the gate without a terminating finish (`reached=true`, `solved=false` - "
                 "the episode ended on a limit) is not solved, and neither is a `finish` "
                 "outside the gate. `reached` / `final_distance_m` report the geometry on "
                 "their own, `success` alone is that verdict, `claim` is what the agent "
                 "*declared*.")
    lines.append("- `verify_gate`: the distance limit the episode's `verify_goal` calls "
                 "used (`verify_limit_m`, from `checks.distance_limit_m`) must be the "
                 "evaluation gate, else the completion check and the score disagree - a "
                 "claim taken against the tool's own default (1.5 m) while the gate is "
                 "1.2 m is a threshold artifact, not evidence. `verify_gate_consistent` is "
                 "`-` for episodes that predate the report (unknown).")
    lines.append("- `prompt_episode_id`: the episode identity the model saw in its prompt. "
                 "The runner hands it a neutral `ep_<seed>` id (never the arm label), and "
                 "`meta.prompt_identity_check` requires the arms of one cell to report the "
                 "same, arm-free id - else the arms differ by a prompt constant and every "
                 "delta of the run is confounded. `-` means the episode did not report it.")
    lines.append("- `claim`: only an explicit declaration counts - a `finish` whose args "
                 "say `success: true` (or a verifying finish tool that answers so), or a "
                 "`verify_goal` answered `verified: true`. A bare finish (a reason and "
                 "nothing else) claims nothing: it is the honest give-up of an episode "
                 "that did not find the target, and the runtime leaves the verdict open "
                 "for it. `finish_decl` is that declaration per episode (`1` = claimed, "
                 "`0` = `success: false`, `-` = undeclared), `honest_finish` counts the "
                 "episodes whose finish did not claim success.")
    lines.append("- `false_confirm`: a *claim* without the geometric reach, or a grounded "
                 "candidate reported outside the gate. `false_rate` = false claims / claims "
                 "(a share over the claims only - honest give-ups stay out of both the "
                 "numerator and the denominator).")
    lines.append("- the finish contract of this round asks the model for the explicit "
                 "flag (`scripts/run_vlm_nav_tools.py` appends it to the tool doc), which "
                 "is what makes `claim` a statement about the agent rather than about "
                 "the harness.")
    lines.append("- `spatial_memory` / `episodic_memory` in the applied-switch table are "
                 "the two memory stores of the memory arm; `?` means the built memory did "
                 "not attest the layer (it does not implement it), never that the layer ran.")
    lines.append("- `repeat_search`: scan-type calls from an already scanned (2 m) cell "
                 "plus information calls whose result carried no new evidence, over all "
                 "information calls.")
    lines.append("- `coverage`: the observation contract's reported coverage - "
                 "`coverage.checked_deg` as a degree count or as the lookaround's "
                 "`[lo, hi]` sector list (`coverage_checked_sectors` / "
                 "`coverage_total_sectors`, `direction_coverage` = sectors / all) - "
                 "else the estimated union of the observed headings' FOV windows; "
                 "`views` counts the expanded view indices, `crops` the crop/zoom re-checks.")
    lines.append("- `vlm_calls`: decision turns + the internal calls the tools declare "
                 "(`vlm_calls` in the result data - the search skill's own count); "
                 "`tokens` follows the sibling drivers' model "
                 f"(images x {meta['token_model']['image_tokens']} + text/{meta['token_model']['chars_per_token']:g}).")
    lines.append("- `search (out/in)`: the search-skill calls of the arm and the routine steps "
                 "inside them (`cost.outer_tool_calls` / `cost.inner_tool_calls`). One search is "
                 "*one* tool call for the episode; the routine runs its steps on its own "
                 "allowance (`max_steps`, clipped by the episode budget) and returns what it has "
                 "when the cap is reached. The episode-level budget metric is "
                 "`tool_budget.left` in the runner summaries; the rows also carry "
                 "`declared_routine_vlm_calls` (the model calls the routines made - counted in "
                 "`vlm_calls_total` as well).")
    lines.append("- `toward_target_rate` (diagnostic only, in the JSON) counts moving legs "
                 "with real displacement whose heading is within 90 deg of the target "
                 "bearing: search and obstacle-avoidance legs may legitimately leave it.")
    lines.append("- The arm support column says whether the runner really has the switch "
                 "an arm declared; `degraded` arms ran a subset of their layer. The "
                 "applied-switch table above comes from the runners' own summaries "
                 "(`summary.json.switches`), so it says what the episodes really ran with.")
    lines.append("")
    return "\n".join(lines)


def meta_block(args: argparse.Namespace, scenarios: list[ScenarioSpec],
               episodes: list[EpisodeSpec], commands: dict, aggregates: dict,
               rows: list[dict] | None = None) -> dict:
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_head": git_head(),
        "backend": args.backend,
        "model_path": args.model_path,
        "model": args.model,
        "device": args.device,
        "arm_list": list(args.arm_list),
        "arms": [ARM_TABLE[name].to_dict() for name in args.arm_list],
        "arm_support": args.arm_support,
        "scenarios": [scenario.to_dict() for scenario in scenarios],
        "seeds": list(args.seed_list),
        "yaw_modes": list(args.yaw_mode_list),
        "n_episodes": len(episodes),
        "budget": budget_mapping(args),
        "toolset": args.toolset,
        "token_model": {
            "image_tokens": phase1.image_tokens(args),
            "chars_per_token": float(args.chars_per_token),
            "prompt_chars_per_call": float(args.prompt_chars_per_call),
        },
        "grid_m": float(args.grid_m),
        "runner_flags_probed": args.runner_flags_probed,
        "mechanism_check": mechanism_report(rows or []),
        # the two confounds this driver has to rule out itself: the completion
        # check must use the evaluation gate, and no arm may differ from another
        # by the episode id it shows the model
        "verify_gate": verify_gate_summary(rows or []),
        "prompt_identity_check": prompt_identity_check(rows or [], list(args.arm_list)),
        "out": str(Path(args.out)),
        "pairing_check": pairing_check(args, episodes, commands),
        "aggregates": aggregates,
    }


def run_comparison(args: argparse.Namespace) -> dict:
    scenarios = load_scenarios(args)
    episodes = build_episodes(args, scenarios)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    by_name = {scenario.name: scenario for scenario in scenarios}
    commands: dict = {}
    for spec in episodes:
        out_dir = episode_dir(out_root, spec)
        commands[(spec.arm, spec.scenario, spec.task.name, spec.seed, spec.yaw_mode)] = \
            build_command(args, by_name[spec.scenario], spec, out_dir)

    if args.dry_run:
        for spec in episodes:
            print(f"# arm={spec.arm} scenario={spec.scenario} cell={spec.cell}")
            print(" ".join(str(part) for part in
                           commands[(spec.arm, spec.scenario, spec.task.name, spec.seed, spec.yaw_mode)]))
        return {"dry_run": True, "n_episodes": len(episodes)}

    rows_by_index: dict[int, dict] = {}
    started = time.time()
    jobs = max(1, int(args.jobs))

    def execute(index: int, spec: EpisodeSpec) -> tuple[int, dict]:
        out_dir = episode_dir(out_root, spec)
        out_dir.mkdir(parents=True, exist_ok=True)
        scenario = by_name[spec.scenario]
        if ARM_TABLE[spec.arm].reserved:
            print(f"[decision] {index + 1}/{len(episodes)} {spec.arm} "
                  f"{spec.cell} -> skipped (reserved arm)", flush=True)
            row = row_defaults(spec, args, out_dir)
            row["skipped"] = True
            row["finish_reason"] = "reserved_arm"
            return index, row
        cmd = commands[(spec.arm, spec.scenario, spec.task.name, spec.seed, spec.yaw_mode)]
        print(f"[decision] {index + 1}/{len(episodes)} {spec.arm} {spec.cell}", flush=True)
        run = run_command(cmd, float(args.timeout), out_dir / "run.log")
        run["command"] = cmd
        row = extract_metrics(args, scenario, spec, out_dir, run)
        print(f"[decision]   -> rc={row['returncode']} solved={row.get('solved')} "
              f"success={row.get('success')} false_confirm={row.get('false_confirmation')} "
              f"repeat={row.get('repeat_search_ratio')} vlm={row.get('vlm_calls_total')}"
              + (f" error={row['error']}" if row.get("error") else ""), flush=True)
        return index, row

    if jobs == 1 or len(episodes) <= 1:
        for index, spec in enumerate(episodes):
            key, row = execute(index, spec)
            rows_by_index[key] = row
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(execute, index, spec)
                       for index, spec in enumerate(episodes)]
            for future in futures:
                key, row = future.result()
                rows_by_index[key] = row
    rows = [rows_by_index[index] for index in range(len(episodes))]

    arms = [name for name in args.arm_list]
    groups = grouping(rows, [scenario.name for scenario in scenarios],
                      list(args.yaw_mode_list), arms)
    aggregates = {f"{arm}|{group}": aggregate(subset) for arm, group, subset in groups}
    deltas = layer_deltas(rows, arms)
    meta = meta_block(args, scenarios, episodes, commands, aggregates, rows)
    summary = {
        "meta": meta,
        "overall": aggregate(rows),
        "aggregates": aggregates,
        "layer_deltas": deltas,
        "episodes": rows,
        "n_errors": sum(1 for row in rows if row.get("error")),
        "n_skipped": sum(1 for row in rows if row.get("skipped")),
        "wall_time_total_s": round(time.time() - started, 3),
    }
    with open(out_root / REPORT_JSON, "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    with open(out_root / "episodes.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    with open(out_root / REPORT_MD, "w") as handle:
        handle.write(render_markdown(meta, rows, groups, aggregates, deltas))
    with open(out_root / "run_config.json", "w") as handle:
        json.dump(vars(args), handle, indent=2, default=str)
    print(f"[decision] wrote {out_root / REPORT_JSON} and {out_root / REPORT_MD}", flush=True)
    for key, stats in aggregates.items():
        arm, _, group = key.partition("|")
        print(f"[decision] {arm:15s} {group:12s} n={stats['n']:<3d} "
              f"solved={stats['n_solved']}/{stats['n_ok']:<3d} "
              f"false_confirm={stats['n_false_confirmation']}/{stats['n_claims']:<3d} "
              f"repeat={_fmt(stats['repeat_search_ratio_mean'])} "
              f"vlm={_fmt(stats['vlm_calls_total_mean'], 1)}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # --- matrix ---
    parser.add_argument("--arms", type=str, default=",".join(ARM_ORDER[:3]),
                        help="comma list of " + ",".join(ARM_ORDER)
                             + " (default: the three layers before memory; the "
                               "memory arms add the memory layer switches on top "
                               "of recovery)")
    parser.add_argument("--scenario", "--scenarios", type=str, action="append",
                        default=None, dest="scenario_args",
                        help="scenario preset, repeatable (default: "
                             + ",".join(DEFAULT_SCENARIOS)
                             + "; presets: " + ",".join(SCENARIOS) + ")")
    parser.add_argument("--seeds", type=str, default="7,11")
    parser.add_argument("--initial-yaw-modes", "--initial-yaw-mode", type=str,
                        default="visible,random", dest="initial_yaw_modes")
    parser.add_argument("--max-vlm-calls", type=int, default=20,
                        help="turn budget per episode (identical for every arm)")
    parser.add_argument("--tool-budget", type=int, default=None,
                        help="tool-call cap per episode (default: --max-vlm-calls)")
    parser.add_argument("--toolset", type=str, default="full")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--initial-yaw-half-fov-deg", type=float, default=40.0)
    parser.add_argument("--grid-m", type=float, default=2.0)
    parser.add_argument("--min-progress-m", type=float, default=0.05,
                        help="displacement floor of a counted moving leg")
    parser.add_argument("--hfov-deg", type=float, default=86.0,
                        help="horizontal FOV used by the estimated coverage")
    # --- arms ---
    parser.add_argument("--arm-arg", type=str, action="append", default=None, dest="arm_args",
                        help="override an arm's runner switches: ARM=<tokens> "
                             "(e.g. 'recovery=--recovery on --state-block on'; "
                             "ARM=none clears the defaults); repeatable")
    parser.add_argument("--strict-arms", action="store_true",
                        help="refuse to start when an arm declares a switch the "
                             "runner does not have (default: degrade + report)")
    parser.add_argument("--no-probe", action="store_true",
                        help="skip the runner-source switch probe (pass every declared switch)")
    # --- task(s) / scene (mirrors the sibling drivers) ---
    parser.add_argument("--task-file", type=str, default=None)
    parser.add_argument("--task-json", type=str, action="append", default=None)
    parser.add_argument("--task-name", type=str, default="task0")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--target-name", type=str, default="target")
    parser.add_argument("--target-pos", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"), help="FLU, scene-centered")
    parser.add_argument("--target-radius", type=float, default=0.5)
    parser.add_argument("--start-pos", type=float, nargs=3, default=(0.0, 0.0, 1.0))
    parser.add_argument("--success-dist", type=float, default=2.0)
    parser.add_argument("--world", choices=["cpu", "higs", "env", "super"], default=None,
                        help="override the scenario's world")
    parser.add_argument("--ply", type=str, default=None)
    parser.add_argument("--scene-scale", type=float, default=None)
    parser.add_argument("--min-opacity", type=float, default=None)
    parser.add_argument("--combined-ply", type=str, default=None)
    parser.add_argument("--background-ply", type=str, default=None)
    parser.add_argument("--semantic-manifest", type=str, default=None)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--render-size", type=int, nargs=2, default=None, metavar=("W", "H"))
    parser.add_argument("--target-index", type=int, default=None)
    parser.add_argument("--max-init-attempts", type=int, default=8)
    parser.add_argument("--env-overrides", type=str, nargs="*", default=None)
    # --- model ---
    parser.add_argument("--backend", type=str, default="mock",
                        help="mock | mock:oracle | hf | openai")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--image-size", type=int, nargs=2, default=(320, 180),
                        metavar=("W", "H"))
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--no-think", action="store_true")
    # --- context / observability ---
    parser.add_argument("--context-frames", type=int, default=None,
                        help="shared frames per decision (an arm may override it)")
    parser.add_argument("--scan-views", type=int, default=None)
    parser.add_argument("--scan-pitch-deg", type=float, default=None)
    parser.add_argument("--tool-arg", type=str, action="append", default=None,
                        help="extra flag appended verbatim to every runner invocation")
    # --- metric model ---
    parser.add_argument("--image-token-divisor", type=float, default=784.0)
    parser.add_argument("--chars-per-token", type=float, default=4.0)
    parser.add_argument("--prompt-chars-per-call", type=float, default=1200.0)
    return parser.parse_args(argv)


def resolve_matrix(args: argparse.Namespace) -> None:
    arms = _str_list(args.arms) or list(ARM_ORDER[:3])
    unknown = [name for name in arms if name not in ARM_TABLE]
    if unknown:
        raise SystemExit(f"unknown --arms entries {unknown}; pick from {ARM_ORDER}")
    args.arm_list = [name for name in ARM_ORDER if name in arms]
    args.seed_list = _int_list(args.seeds) or [0]
    modes = _str_list(args.initial_yaw_modes) or ["visible", "random"]
    unknown_modes = [mode for mode in modes if mode not in compare.YAW_MODES]
    if unknown_modes:
        raise SystemExit(f"unknown --initial-yaw-modes entries {unknown_modes}")
    args.yaw_mode_list = modes
    if args.max_vlm_calls <= 0:
        raise SystemExit("--max-vlm-calls must be positive")
    requested = _str_list(",".join(args.scenario_args or [])) or list(DEFAULT_SCENARIOS)
    unknown_scenarios = [name for name in requested if name not in SCENARIOS]
    if unknown_scenarios:
        raise SystemExit(
            f"unknown --scenario entries {unknown_scenarios}; pick from {list(SCENARIOS)}"
        )
    args.scenario_list = requested
    if getattr(args, "no_probe", False):
        args.runner_flags_probed = None
    else:
        flags = runner_flags_from_source()
        args.runner_flags_probed = sorted(flags)
    resolved, support = resolve_arms(args, args.runner_flags_probed)
    args.arm_specs = {spec.name: spec for spec in resolved}
    args.arm_support = support
    reserved = [spec.name for spec in resolved if spec.reserved]
    if reserved:
        print(f"[decision] note: reserved arm(s) {', '.join(reserved)}: "
              "their episodes are skipped", flush=True)
    memory_arms = [spec.name for spec in resolved
                   if spec.name in (ARM_MEMORY, ARM_MEMORY_SPATIAL)]
    if memory_arms:
        print(f"[decision] note: {', '.join(memory_arms)} add the memory layer "
              "(both stores / spatial only) on top of the recovery arm; the "
              "report shows the switches and the catalogue each arm really ran in "
              f"(the M2 catalogue is '{M2_MEMORY_TOOLSET}' and can be requested "
              "with --arm-arg)", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    resolve_matrix(args)
    summary = run_comparison(args)
    if summary.get("dry_run"):
        return 0
    errors = int(summary.get("n_errors") or 0)
    if errors:
        print(f"[decision] {errors} episode(s) failed; see {REPORT_JSON}", flush=True)
        return 1
    print(f"[decision] all {summary['overall']['n']} episodes completed "
          f"({summary.get('n_skipped', 0)} skipped)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
