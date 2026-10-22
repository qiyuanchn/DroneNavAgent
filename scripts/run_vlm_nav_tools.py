#!/usr/bin/env python
"""M1 acceptance runner: the VLM navigation agent as a tool-calling runtime (goal.md M1).

Where ``run_vlm_nav_3dgs.py`` sends one image and expects one motion action, this
runner gives the model a *tool registry* (``vlm_nav.tools``) and executes
the multi-turn loop: the model requests a tool with one constrained-JSON call,
the runtime executes it, feeds the structured ``ToolResult`` back, and the model
decides the next call until it emits ``{"tool": "finish", ...}``.

Acceptance play (goal.md): the target is unclear -> ``inspect_region``; the target
is lost -> ``query_memory``; pick an observation point -> ``propose_viewpoints`` +
``go_to_viewpoint``; approach the object -> ``ground_target`` +
``approach_candidate``; re-observe -> ``observe``; confirm -> ``verify_goal`` ->
``finish``.

Navigation intent vocabulary: the movement tools come one per intent from
``vlm_nav.tool_impl_navigation`` - ``approach_candidate(candidate_id)``
(the object in a frame), ``go_to_viewpoint(viewpoint_id)`` (an observation
position) and ``return_to_observation(observation_id)`` (the camera pose a frame
was shot from), fed by ``propose_viewpoints``. The mixed-semantics ``navigate``
is registered only to answer with an explicit error naming the replacement call
and is in no tool set; the legacy fallback stack (used when that module is
missing) still provides its own ``navigate``, marked deprecated, so a
module-less run keeps a movement tool.

``--toolset basic|memory|full`` (default ``full``) selects the tool *catalogue* the
model sees, so the same episode can be replayed against increasing capability
(goal.md section 7's 1->4 comparison):

  basic   observe, propose_viewpoints, approach_candidate, go_to_viewpoint,
          return_to_observation, verify_goal (no memory: the fixed-keyframe
          baseline - it can move, but nothing tells it where the target was)
  memory  basic + query_memory, inspect_region, ground_target, compare_candidates
  full    memory + check_path and the M3 task-cycle tools (task_status /
          cancel_task / update_plan / get_plan); ``full`` is unrestricted, so a
          tool a module registers is picked up automatically and names nothing
          provides are listed in ``summary.toolset_detail``.

Tool names outside the requested set are dropped before they reach the registry,
so they never appear in ``registry.specs_text()`` (the model's catalogue) nor in
``ToolResult`` errors. ``scripts/run_vlm_nav_m2_memory.py`` drives this runner for
the goal.md M2 experiment (memory vs. basic on a target that left the view).

``--recovery {on,off}`` / ``--state-block {on,off}`` / ``--search-skill {on,off}``
(default ``on``: the behavior every earlier run had) switch the three mechanism
layers off one at a time - they are what ``scripts/compare_decision_loop.py``
drives to build its ablation arms:

  --recovery off    a stagnating episode ends at the stagnation limit; no
                    recovery advice, no honest-finish grant, no recovery hook
                    (the pre-recovery semantics of the runtime)
  --state-block off the fixed STATE block is not injected into the prompt and
                    ``context["state"]`` stays empty (``runtime.state()`` still
                    reports it)
  --search-skill off ``search_target`` is not registered, so the model can
                    neither see nor call the search routine

``--memory-spatial {on,off}`` / ``--memory-episodic {on,off}`` (default ``on``)
are the two *memory layers* of the decision-loop ablation. They are handed to
the memory implementation as its own constructor parameters
(``spatial_memory`` / ``episodic_memory``: the spatial store keeps where the
task target is and where it was seen, the episodic store keeps which view saw
what), so an arm can run with one layer, both or neither:

  --memory-spatial off  no spatial store: the memory answers from what it has,
                        but the task target's world place is not kept
  --memory-episodic off no episodic evidence store at all: sightings
                        (frame + candidate + basis), stored frames and scoped
                        negatives are not kept (capture records and the
                        spatial/working fan-out remain)

The memory module owns the enforcement; the runner only passes the flags and
reports what the built memory confirms. ``summary.json.switches`` records both
the requested and the *applied* value of each switch (a runtime that does not
expose the knob counts as off; a memory that does not report the layer stays
``None`` - "unattested", never a fake ``on``), and ``run_config.json`` keeps
the raw CLI.

The system prompt carries this round's finish contract: the runtime counts a
completion *claim* only when the ``finish`` call declares it
(``success: true`` after a verified goal, ``success: false`` for an honest
give-up), so the tool doc asks for the flag explicitly. A bare finish claims
nothing - exactly the semantics the runtime implements.

Tool dispatch and the catalogue (P4). The registry is handed the live
``ctx.extra``, so ``registry.specs_text()`` renders a *state-aware* catalogue:
the tools the episode state calls come first with their complete description and
arguments (observe/search/query_memory when nothing is grounded yet,
compare/approach/verify once a candidate exists, task_status/cancel while a leg
runs), every other registered tool is still named behind them, and no entry is
ever cut in half. Since P4-B the runtime budgets its system prompt
*catalogue-first*: the state-aware listing is reserved before the policy
template and the protocol doc, and the doc is truncated (with a note) first.
``SYSTEM_CHARS`` is therefore only an upper bound handed to the runtime (whose
own default is the same width), so the whole ``CATALOG_CHARS`` catalogue
arrives; ``summary.prompt_budget`` reports the state, the tiers, the rendered
size and a digest of what the model read. The same registry object
serves the MCP adapters (``vlm_nav.mcp_server``), which order their tool listing
by the same state.

Completion is traceable (P4). ``finish`` is a registered tool
(``vlm_nav.tool_impl_navigation``) that judges a ``"success": true`` on the
episode's evidence pack: a ``verify_goal`` that returned ``verified=true``, or a
grounded candidate plus an arrival. Without that trail the claim is downgraded
(``success: false``, ``claim=unverified``) and never counted - and the runner
judges the *same* chain on the trace (``summary.completion``,
``success_evidence``), so ``success`` needs ground truth (distance + a finishing
call) **and** the evidence, and ``summary.completion.evidence_trail`` is the trail
that backed it.

One gate, one identity. ``--success-dist`` is a standoff from the target
*surface*, so the geometric gate this runner judges with is
``--success-dist + target radius`` (``summary.json.success_dist_m``); the tools
receive that same number as ``ctx.extra["done_dist"]``, so the in-loop
completion check (``verify_goal``) and the evaluation gate can never disagree.
The episode identity the *model* sees is likewise built from neutral inputs
(``ep_<seed>``, :func:`neutral_episode_id`): a caller may label an episode with
its arm or mode (that label names the artifacts and ``summary.episode_id``), but
the prompt never carries it - two arms of one cell get byte-identical first
prompts, so an arm difference cannot be an artifact of the episode id.

Mock (no weights, scripted acceptance play):

    python scripts/run_vlm_nav_tools.py --world cpu --ply /path/scene.ply \
        --target-name target --target-pos 0 0 1.0 --target-radius 0.4 \
        --start-pos -2 0 1.2 --instruction "Find the target and fly to it." \
        --backend mock --out outputs/vlm_nav/tools_mock

Real model (local HF weights / OpenAI-compatible endpoint):

    python scripts/run_vlm_nav_tools.py --world higs --ply /path/scene.ply \
        --target-pos 0 0 0.5 --instruction "Find the book." \
        --backend hf --device cuda --model-path /home/.../Qwen3-VL-8B \
        --max-turns 20 --tool-budget 40 --out outputs/vlm_nav/tools_higs

``--mock-script FILE`` replays a JSON list of tool calls instead of the built-in
acceptance play; the scripted mock VLM (``ScriptedToolVLM``) substitutes a string arg
``"$<field>"`` with the latest tool-result value of that field, so the script itself
consumes the previous result the way a real model reading the returned fields would
(a real model emits concrete ids). ``--world``, ``--ply``, ``--combined-ply``,
``--semantic-manifest`` and the other scene flags follow ``scripts/run_vlm_nav_3dgs.py``.

Artifacts:
  tool_trace.jsonl  one line per turn (model reply, executed call, result, evidence)
  summary.json      success/finish reason + per-tool calls/ok/blocked/used/new
                    evidence/time + evidence-kind histogram + stack provenance
                    + the resolved ``toolset`` + the completion claim with its
                    evidence trail and the prompt/catalogue budget
  frames/           the images referenced by the tool evidence refs
  run_config.json   resolved CLI configuration

Stack provenance: the M1 modules of the parallel workstreams (``vlm_nav.runtime``,
``vlm_nav.memory``, ``vlm_nav.tool_impl_perception``, ``vlm_nav.tool_impl_navigation``)
are used when importable, through these entry points:

  tool_impl_perception.register_perception_tools(registry, vlm=None)
  tool_impl_navigation.register_navigation_tools(registry)
  memory.ObservationLog(image_dir=out_dir/frames, ...)
  runtime.AgentRuntime(registry, world, memory, vlm, budget=..., max_turns=...).run(instruction)

``summary.json.stack_detail`` records what was actually wired (per-tool provenance in
``tools_source``). Names the modules do not register (e.g. ``query_memory`` today) are
filled from the fallback registry below, which also runs the episode on its own when
``vlm_nav.runtime`` is unavailable; the fallback block is marked TODO(M1) so it can be
deleted once every tool and the runtime module are wired.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import random
import re
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from vlm_nav.actions import NavAction
from vlm_nav.tools import (
    DEFAULT_CATALOG_CHARS,
    EVIDENCE_PACK_KEY,
    STATUS_BLOCKED,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_OK,
    TOOL_CALL_DOC,
    EvidenceRef,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    completion_evidence,
    evidence_records_from_pack,
    evidence_records_from_trace,
    evidence_pack,
    parse_tool_call,
    tool_state,
)
from vlm_nav.detectors import resolve_detector
from vlm_nav.scene_combo import choose_start_pose, load_scene_combo
from vlm_nav.vlm import MockVLM, frame_to_pil, make_backend
from vlm_nav.world import (
    EnvWorld,
    SplatWorldCPU,
    SplatWorldHiGS,
    SuperPlannerWorld,
    TargetSpec,
)

EVIDENCE_KINDS = ("image", "observation", "memory", "map", "trajectory")

# M1 six abilities (goal.md section 7); the fallback registry implements these
M1_TOOL_NAMES = (
    "observe",
    "inspect_region",
    "query_memory",
    "propose_viewpoints",
    "navigate",
    "verify_goal",
)
# The navigation-intent tools `tool_impl_navigation` registers in place of the
# mixed-semantics ``navigate`` (one tool per intent: the object in a frame, an
# observation position, the camera pose a frame was shot from). ``navigate``
# itself is deprecated and is not part of any tool set: it only answers with an
# explicit error naming the replacement call. The runner's own fallback stack
# (used when the modules are missing) keeps the legacy implementation under the
# legacy name and marks it deprecated in its spec.
NAV_INTENT_TOOL_NAMES = (
    "approach_candidate",
    "go_to_viewpoint",
    "return_to_observation",
)

# --- tool sets (goal.md section 7: action loop -> tool loop -> +memory -> +M3) --
# A tool set moves the drone with the intent vocabulary only: proposing the
# viewpoint ids go_to_viewpoint consumes belongs to moving, so every set that
# can move at all carries it. The sets differ in *evidence*, not in movement:
# basic has no retrieval tool at all, memory adds the four evidence tools.
#
# ``batch_read`` (pack several read-only checks into one turn) and ``finish``
# (the explicit, evidence-backed outcome) are *dispatch* affordances, not extra
# evidence: every set carries them, so no arm loses the ability to end an episode
# with an outcome or to spend one turn on two checks.
MOVEMENT_TOOL_NAMES = ("propose_viewpoints",) + NAV_INTENT_TOOL_NAMES
BASIC_TOOL_NAMES = (
    ("observe", "batch_read") + MOVEMENT_TOOL_NAMES
    + ("verify_goal", "confirm_candidate", "finish")
)
MEMORY_TOOL_NAMES = BASIC_TOOL_NAMES + (
    "query_memory",
    "inspect_region",
    "ground_target",
    "compare_candidates",
)
# M3 long-horizon tools: tool_impl_task / tool_impl_navigation may provide them
TASK_TOOL_NAMES = ("task_status", "cancel_task", "update_plan", "get_plan")
FULL_TOOL_NAMES = MEMORY_TOOL_NAMES + ("check_path",) + TASK_TOOL_NAMES
# None = unrestricted: every tool a module registers is kept (a module that lands
# later is picked up without touching this table)
TOOLSETS: dict[str, frozenset | None] = {
    "basic": frozenset(BASIC_TOOL_NAMES),
    "memory": frozenset(MEMORY_TOOL_NAMES),
    "full": None,
}
TOOLSET_CHOICES = ("basic", "memory", "full")
DEFAULT_TOOLSET = "full"

#: Mechanism switches of the decision-loop ablation (``compare_decision_loop``):
#: ``on`` keeps the mechanism - the behavior of every earlier run - and ``off``
#: removes it, so an arm is reproducible from its ``run_config.json``.
SWITCH_CHOICES = ("on", "off")
DEFAULT_SWITCH = "on"

# --- prompt budget of the tool catalogue (goal.md P4) -------------------------
# The module runtime passes ``registry.specs_text()`` as the tools catalogue.
# Before P4-B it truncated the assembled system prompt at ``max_system_chars``,
# so a 4000-character default cut the (then ~20k) listing mid-entry. The runtime
# now reserves the rendered catalogue first and truncates the protocol
# doc/examples instead (see ``vlm_nav.runtime._system_prompt``); its default
# budget (``DEFAULT_SYSTEM_CHARS`` = catalogue + 7k) holds this round's
# state-aware listing whole. The runner still hands the same width as an upper
# bound and reports what the runtime really rendered.
CATALOG_CHARS = DEFAULT_CATALOG_CHARS
#: system-prompt budget: the runtime's policy template (~1k) + the finish
#: contract (~4.3k) + the whole catalogue (9k), with headroom for the other
#: sections. Since P4-B the runtime budgets catalogue-first on its own (the
#: doc/examples are truncated before the catalogue), so this is an upper bound
#: handed to the runtime, not a workaround: the runtime's default is the same
#: number, and ``summary.prompt_budget`` reports what was really rendered.
SYSTEM_CHARS = 16000
#: The search *skill* (a routine over the perception tools, not a tool of its own
#: vocabulary): ``--search-skill off`` keeps it out of the registry.
SEARCH_SKILL_TOOL = "search_target"
#: The three memory layers, by the names the memory implementation uses for its
#: own constructor parameters (``spatial_memory`` / ``episodic_memory`` /
#: ``identity_memory``). The CLI spells them ``--memory-spatial`` /
#: ``--memory-episodic`` / ``--memory-identity``.
MEMORY_SWITCHES = ("spatial_memory", "episodic_memory", "identity_memory")

#: The finish contract of this round, appended to the tool doc of both runtimes.
#: The runtime counts a completion claim only when the finish declares it (a bare
#: finish leaves the verdict open), and the decision-loop metric counts exactly
#: that declaration - so the prompt has to ask for the flag. Since P4 the claim
#: also has to be *backed*: the registered finish tool judges it on the episode's
#: evidence pack, and an unbacked ``success: true`` comes back downgraded (never a
#: success), so the prompt names the two chains that count.
FINISH_CONTRACT = (
    "Finish contract: always state the outcome in the finish call - "
    '{"tool": "finish", "args": {"reason": "<short>", "success": true}} only '
    "after a verify_goal call of this episode answered verified=true, or after a "
    "grounded candidate was reached with outcome=arrived AND confirm_candidate "
    "accepted that instance from a new view under the current revision; "
    '{"tool": "finish", "args": {"reason": "<what blocked you>", "success": '
    "false}} when you give up. A finish that carries no success flag claims "
    "nothing: an honest unfinished answer is expected, an undeclared one is "
    "not - say success: false instead. A success without that evidence is "
    "downgraded to an unverified claim and never counted: the answer reports "
    "verified, claim, evidence_trail and what is missing."
)


def finish_contract_doc() -> str:
    """``TOOL_CALL_DOC`` with the explicit-success finish contract of this round.

    The sibling module's doc (``vlm_nav.tools.TOOL_CALL_DOC``) still tolerates a
    bare finish; the runtime does not, so the sentence is replaced when present
    and the contract is appended either way (the prompt then never tells the
    model that a finish without the flag is fine).
    """
    text = TOOL_CALL_DOC.replace(
        "or finish with a reason but no success flag.",
        'or finish with "success": false - never leave the outcome undeclared.',
    )
    return f"{text}\n\n{FINISH_CONTRACT}"


# modules written by the parallel M1 workstreams; used when importable
_M1_MODULES = {
    "runtime": "vlm_nav.runtime",
    "memory": "vlm_nav.memory",
    "perception": "vlm_nav.tool_impl_perception",
    "navigation": "vlm_nav.tool_impl_navigation",
    "task": "vlm_nav.tool_impl_task",
}
_REGISTER_FN_NAMES = (
    "register_perception_tools",
    "register_navigation_tools",
    "register_task_tools",
    "register_all",
    "register_default_tools",
    "register_tools",
    "register",
)

# goal.md acceptance play in the Phase-1 vocabulary: unclear target -> inspect,
# pick a viewpoint -> go_to_viewpoint, lost target -> query_memory, locate the
# object -> ground_target + approach_candidate, re-observe -> verify -> finish.
# Every movement goes through an intent tool (``navigate`` is deprecated and in no
# tool set); the region is the canonical [x1, y1, x2, y2] corner box.
ACCEPTANCE_TOOL_SCRIPT: list[dict] = [
    {"tool": "observe", "args": {"mode": "lookaround", "n_views": 8,
                                 "reason": "target is not clearly visible yet"}},
    {"tool": "inspect_region", "args": {"observation_id": "$observation_id",
                                        "region": [400, 400, 600, 600], "zoom": 2.0,
                                        "reason": "target is unclear at this distance"}},
    {"tool": "ground_target", "args": {"instruction": "$target_name",
                                       "observation_id": "$observation_id",
                                       "reason": "locate the object in the frame I saw it in"}},
    {"tool": "propose_viewpoints", "args": {"n": 3, "radius_m": 2.0,
                                            "reason": "pick a closer observation point"}},
    {"tool": "go_to_viewpoint", "args": {"viewpoint_id": "$viewpoint_id",
                                         "reason": "fly to the chosen observation point"}},
    {"tool": "query_memory", "args": {"query": "$target_name", "limit": 3,
                                      "reason": "where was the target last seen?"}},
    {"tool": "observe", "args": {"mode": "forward",
                                 "reason": "re-observe the target direction"}},
    {"tool": "approach_candidate", "args": {"candidate_id": "$candidate_id",
                                            "standoff_m": 0.8,
                                            "reason": "close in on the located object"}},
    {"tool": "confirm_candidate", "args": {"candidate_id": "$candidate_id",
                                           "reason": "verify the reached instance from a new view"}},
    {"tool": "verify_goal", "args": {"reason": "confirm the target before finishing"}},
    {"tool": "finish", "args": {"reason": "target reached, confirmed and verified"}},
]

_ID_RE = re.compile(r"^[a-z][a-z0-9]*_\d+$")

#: what the region/box schemas ask for - the canonical contract of
#: ``vlm_nav.tool_impl_perception`` (corner box, top-left first, normalized
#: 0-1000); the legacy ``[u, v, w, h]`` size form is still *read* when the
#: corner reading is reversed or empty, and a component above 1000 is taken as
#: absolute pixels
BOX_FORMAT_DOC = "[x1, y1, x2, y2] corner box, top-left first, normalized 0-1000"
_PIXEL_THRESHOLD = 1000.0


def _parse_region(region: Any, width: int, height: int,
                  box_format: Any = None) -> tuple[tuple[float, ...] | None, str, str]:
    """``region`` -> (normalized corner box, format_used, ""), else (None, "", reason).

    The canonical reading is ``[x1, y1, x2, y2]``; ``[u, v, w, h]`` (top-left +
    size) is the legacy spelling and is still understood when the corner reading
    is degenerate, exactly as ``tool_impl_perception.parse_box`` does - the box a
    tool returns can be fed back into it.
    """
    if isinstance(region, dict):
        if all(key in region for key in ("x_min", "y_min", "x_max", "y_max")):
            region = [region[key] for key in ("x_min", "y_min", "x_max", "y_max")]
            box_format = box_format or "xyxy"
        elif all(key in region for key in ("u", "v", "w", "h")):
            region = [region[key] for key in ("u", "v", "w", "h")]
            box_format = box_format or "uvwh"
    if not isinstance(region, (list, tuple)) or len(region) != 4:
        return None, "", "invalid_region"
    try:
        numbers = [float(value) for value in region]
    except (TypeError, ValueError):
        return None, "", "invalid_region"
    if not all(math.isfinite(number) for number in numbers):
        return None, "", "invalid_region"
    unit = "norm"
    if max(abs(number) for number in numbers) > _PIXEL_THRESHOLD:
        if width <= 0 or height <= 0:
            return None, "", "unknown_frame_size"
        unit = "px"
        numbers = [
            numbers[0] / float(width) * 1000.0,
            numbers[1] / float(height) * 1000.0,
            numbers[2] / float(width) * 1000.0,
            numbers[3] / float(height) * 1000.0,
        ]
    declared = str(box_format or "").strip().lower()
    used = declared if declared in ("xyxy", "uvwh") else ""
    if not used:
        used = "uvwh" if numbers[2] <= numbers[0] or numbers[3] <= numbers[1] else "xyxy"
    if used == "uvwh":
        u, v, w, h = numbers
        x1, y1, x2, y2 = u, v, u + w, v + h
    else:
        x1, y1, x2, y2 = numbers
    x1, y1 = max(x1, 0.0), max(y1, 0.0)
    x2, y2 = min(x2, 1000.0), min(y2, 1000.0)
    if x2 <= x1 or y2 <= y1:
        return None, "", "region_outside_frame"
    return (x1, y1, x2, y2), f"{used}_{unit}", ""


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------


def _as_int(value: Any, default: int = 0) -> int:
    """Best-effort int of a tool-declared number (``default`` when unusable)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_mapping(obj: Any) -> dict:
    """dict | dataclass | object with __dict__ -> plain dict ({} otherwise)."""
    if isinstance(obj, dict):
        return dict(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {}


def _construct(factory: Any, **kwargs) -> Any:
    """Call ``factory`` with the subset of ``kwargs`` its signature accepts."""
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(**kwargs)
    has_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    accepted = {
        name: value
        for name, value in kwargs.items()
        if has_var_kw or name in signature.parameters
    }
    return factory(**accepted)


def switch_enabled(value: Any) -> bool:
    """``on``/``True`` -> True, anything else -> False (CLI switch helper)."""
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("on", "true", "1", "yes")


def switch_token(value: Any) -> str:
    """The ``on``/``off`` spelling the summary records."""
    return "on" if switch_enabled(value) else "off"


def switch_report(args: argparse.Namespace) -> dict:
    """The mechanism switches as requested on the command line.

    The three decision-loop mechanisms plus the two memory layers (the memory
    layer names are the implementation's own constructor parameters).
    """
    return {
        "recovery": switch_token(getattr(args, "recovery", DEFAULT_SWITCH)),
        "state_block": switch_token(getattr(args, "state_block", DEFAULT_SWITCH)),
        "search_skill": switch_token(getattr(args, "search_skill", DEFAULT_SWITCH)),
        "spatial_memory": switch_token(getattr(args, "memory_spatial", DEFAULT_SWITCH)),
        "episodic_memory": switch_token(getattr(args, "memory_episodic", DEFAULT_SWITCH)),
        "identity_memory": switch_token(getattr(args, "memory_identity", DEFAULT_SWITCH)),
        "detector": str(getattr(args, "detector", "off") or "off"),
    }


def vlm_usage_summary(events: Any) -> dict:
    """Aggregate backend ``usage_metrics`` (including vLLM cached tokens).

    The per-call logs live in the episode's ``vlm_log``; this compact block is
    what a batch report can sum without reading every backend event.
    """
    totals = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
    }
    seen = False
    for event in events or []:
        if not isinstance(event, dict):
            continue
        metrics = event.get("usage_metrics")
        if not isinstance(metrics, dict) or not metrics:
            continue
        seen = True
        totals["calls"] += 1
        for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
            try:
                totals[key] += int(metrics.get(key) or 0)
            except (TypeError, ValueError):
                continue
    if not seen:
        return {}
    return totals


def memory_layer_report(memory: Any) -> dict:
    """The two memory layers as the *built memory* reports them.

    A memory that answers for its layers (``spatial_memory`` /
    ``episodic_memory`` attributes, or a ``memory_switches()`` / ``switches``
    mapping) is taken at its word; a memory that says nothing keeps ``None`` -
    unattested - so the summary (and the decision-loop report) can say "unknown"
    instead of pretending the layer was off or on. The fallback memory has no
    layer split at all, so it never attests.
    """
    reported: Any = None
    for name in ("memory_switches", "switches"):
        accessor = getattr(memory, name, None)
        if callable(accessor):
            try:
                reported = accessor()
            except Exception:  # noqa: BLE001 - an accessor that breaks attests nothing
                reported = None
        elif isinstance(accessor, dict):
            reported = accessor
        if reported is not None:
            break
    out: dict[str, bool | None] = {}
    for name in MEMORY_SWITCHES:
        value = reported.get(name) if isinstance(reported, dict) else None
        if not isinstance(value, bool):
            value = getattr(memory, name, None)
        out[name] = value if isinstance(value, bool) else None
    return out


def applied_switches(runtime: Any, registry: Any, runtime_backend: str,
                     memory: Any = None) -> dict:
    """What the episode really ran with (a missing knob counts as off).

    The fallback loop has neither the STATE block nor the recovery rules and
    answers no ``recovery_enabled`` attribute: reporting it as "on" would claim
    a layer that never ran, so a missing knob is reported as ``False``. The
    memory layers are the exception: a memory that does not attest them is
    reported as ``None`` (unknown), because "the memory did not confirm the
    layer" is not the same statement as "the layer was off".
    """
    try:
        names = set(registry.names())
    except Exception:  # noqa: BLE001 - a duck-typed registry may not answer
        names = set()
    applied = {
        "recovery": bool(getattr(runtime, "recovery_enabled", False)),
        "state_block": bool(getattr(runtime, "state_block", False)),
        "search_skill": SEARCH_SKILL_TOOL in names,
        # P4-B: a runtime without the knob (the fallback loop) reports off, the
        # same rule as recovery/state_block
        "executor_auto_steps": bool(getattr(runtime, "executor_auto_steps", False)),
        "runtime": runtime_backend,
    }
    applied.update(memory_layer_report(memory))
    return applied


def toolset_allows(toolset: str | None, name: str,
                   exclude: Iterable[str] | None = None) -> bool:
    """Is ``name`` part of the requested tool set? (``full`` -> everything is).

    ``exclude`` removes names on top of the tool set (the mechanism switches).
    """
    if exclude and name in set(exclude):
        return False
    allowed = TOOLSETS.get(str(toolset or DEFAULT_TOOLSET))
    return True if allowed is None else name in allowed


def toolset_report(toolset: str | None, registered: Iterable[str]) -> dict:
    """What the tool set asked for vs. what is actually in the registry.

    ``missing`` lists vocabulary names the request expects but no module (nor the
    fallback registry) provided - e.g. the M3 task-cycle tools while no module
    registers them.
    """
    name = str(toolset or DEFAULT_TOOLSET)
    allowed = TOOLSETS.get(name, None)
    registered = list(registered)
    missing = [
        tool
        for tool in FULL_TOOL_NAMES
        if tool != "finish" and tool not in registered and toolset_allows(name, tool)
    ]
    notes: list[str] = []
    if allowed is None:
        notes.append(
            "full is unrestricted: every tool a module registers is kept, "
            "including tools outside the documented vocabulary"
        )
    if missing:
        notes.append(
            "not registered (no module provides them yet): " + ", ".join(missing)
        )
    return {
        "name": name,
        "restricted": allowed is not None,
        "allowed": sorted(allowed) if allowed is not None else "all",
        "registered": sorted(registered),
        "missing": missing,
        "notes": notes,
    }


def _resolve_refs(value: Any, last_data: dict) -> Any:
    """Replace mock-script placeholders ``"$field"`` with ``last_data["field"]``.

    Mock scripting convenience only (see module docstring): real model replies
    carry concrete ids.
    """
    if isinstance(value, str) and value.startswith("$"):
        key = value[1:]
        if key in last_data:
            return last_data[key]
        return value
    if isinstance(value, dict):
        return {k: _resolve_refs(v, last_data) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_resolve_refs(v, last_data) for v in value]
    return value


def _hstack(frames: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([f.detach().clamp(0.0, 1.0) for f in frames], dim=2)


def _pose_fields(world: Any) -> dict:
    try:
        pos = [float(v) for v in np.asarray(world.position_flu).reshape(-1)[:3]]
        yaw = float(world.yaw_rad)
    except Exception:
        return {}
    return {"position_flu": pos, "yaw_deg": round(math.degrees(yaw), 1)}


def _world_distance(world: Any) -> float | None:
    fn = getattr(world, "distance_to_target", None)
    if not callable(fn):
        return None
    try:
        return float(fn())
    except Exception:
        return None


def _target_radius_m(world: Any) -> float:
    radius = getattr(getattr(world, "target", None), "radius_m", 0.0)
    try:
        return float(radius)
    except (TypeError, ValueError):
        return 0.0


def success_gate_m(world: Any, args: Any) -> float:
    """The gate this episode is judged with: ``--success-dist + target radius``.

    ``--success-dist`` is a standoff from the target *surface*, the distance
    the world reports is centre to centre, so the geometric gate - the one
    ``summary.json.success_dist_m`` carries and ``success`` compares against -
    adds the target radius. A completion check inside the loop must use the
    same number, otherwise the episode can pass its own verification and still
    count as unsolved (or the other way round).
    """
    limited = getattr(args, "success_dist", None)
    if limited is None:
        limited = getattr(args, "success_dist_m", 2.0)
    return float(limited) + _target_radius_m(world)


def neutral_episode_id(seed: Any, index: Any = None) -> str:
    """The arm-free episode identity the model may see in its prompt.

    The episode id is part of every prompt's first line ("Episode ep_7 | turn
    ..."), so an id that spells the comparison arm ("m2_memory_000_00") is a
    constant, prompt-visible difference between arms that has nothing to do
    with the mechanism under test - under greedy decoding it is enough to move
    the first decision. The neutral id is built from the seed (and an optional
    episode index) only; the caller's label stays in the artifact directory,
    ``summary.json.episode_id`` and ``run_config.json``.
    """
    try:
        value = int(seed)
    except (TypeError, ValueError):
        value = 0
    if index is None:
        return f"ep_{value:03d}"
    try:
        ordinal = int(index)
    except (TypeError, ValueError):
        return f"ep_{value:03d}"
    return f"ep_{value:03d}_{ordinal:02d}"


def _task_text(ctx: ToolContext) -> str:
    """The instruction the agent was given (never the scene's target identity)."""
    extra = ctx.extra if isinstance(ctx.extra, dict) else {}
    return str(extra.get("instruction") or "the instructed task")[:240]


def _target_name(world: Any) -> str:
    return str(getattr(getattr(world, "target", None), "name", "target"))


def _ctx_out_dir(ctx: ToolContext) -> Path:
    out_dir = ctx.extra.get("out_dir")
    return Path(out_dir) if out_dir else Path(".")


def _save_frame(ctx: ToolContext, frame: torch.Tensor, name: str) -> str:
    """Persist one frame under ``frames/`` and return its evidence ref."""
    ref = f"frames/{name}"
    path = _ctx_out_dir(ctx) / ref
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_to_pil(frame).save(path)
    return ref


def _cache_frame(ctx: ToolContext, frame: torch.Tensor, keep: int = 4) -> None:
    cache = ctx.extra.setdefault("frame_cache", [])
    cache.append(frame.detach())
    del cache[:-keep]


# ---------------------------------------------------------------------------
# trace + summary aggregation (independent of which runtime produced the trace)
# ---------------------------------------------------------------------------


def _write_trace(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, default=str) + "\n")


def _read_trace(path: Path) -> list[dict]:
    entries = []
    if not path.exists():
        return entries
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def summarize_trace(entries: list[dict]) -> dict:
    """Per-tool success/usage/new-evidence/timing stats from the tool trace.

    ``used``   - the model called the tool at least once.
    ``consumed`` - an id produced by the tool shows up in a later turn (the model
                 actually read the result instead of ignoring it).

    ``routine_cost`` sums what the *tools* declare about their own work
    (``cost.outer_tool_calls`` / ``cost.inner_tool_calls`` / ``cost.vlm_calls``,
    e.g. the search skill's steps and the model calls they made): one search is
    one outer call for the episode, its inner calls run on the routine's
    allowance, and both are reported here.
    """
    tools: dict[str, dict] = {}
    evidence_kinds: dict[str, int] = {}
    seen_evidence: set[str] = set()
    total_calls = parse_failures = finish_calls = 0
    new_evidence_total = 0
    max_turn = 0
    routine_cost = {"outer_tool_calls": 0, "inner_tool_calls": 0, "vlm_calls": 0}

    for entry in entries:
        max_turn = max(max_turn, int(entry.get("turn") or 0))
        tool = str(entry.get("tool") or entry.get("tool_name") or "")
        if entry.get("parse_failure") or not tool:
            parse_failures += 1
            continue
        declared = declared_routine_cost(entry)
        for key in routine_cost:
            routine_cost[key] += int(declared.get(key, 0))
        if tool == "finish":
            finish_calls += 1
            continue
        total_calls += 1
        row = tools.setdefault(
            tool,
            {
                "calls": 0,
                "ok": 0,
                "empty": 0,
                "blocked": 0,
                "failed": 0,
                "cached": 0,
                "ok_rate": 0.0,
                "used": True,
                "consumed": False,
                "new_evidence": 0,
                "time_ms_total": 0.0,
                "time_ms_mean": 0.0,
            },
        )
        row["calls"] += 1
        status = str(entry.get("status") or "")
        if status in row:
            row[status] += 1
        row["time_ms_total"] += float(entry.get("ms") or 0.0)
        for ev in entry.get("evidence") or []:
            if not isinstance(ev, dict):
                continue
            kind = str(ev.get("kind") or "")
            evidence_kinds[kind] = evidence_kinds.get(kind, 0) + 1
            key = f"{kind}:{ev.get('ref', '')}"
            if key not in seen_evidence:
                seen_evidence.add(key)
                row["new_evidence"] += 1
                new_evidence_total += 1

    # "consumed": an id from turn i appears (quoted) in a later turn
    for index, entry in enumerate(entries):
        tool = str(entry.get("tool") or "")
        if not tool or tool == "finish" or tool not in tools:
            continue
        ids = [
            value
            for value in (entry.get("data") or {}).values()
            if isinstance(value, str) and _ID_RE.match(value)
        ]
        if not ids:
            continue
        later = json.dumps(entries[index + 1:], default=str)
        if any(f'"{value}"' in later for value in ids):
            tools[tool]["consumed"] = True

    for row in tools.values():
        row["ok_rate"] = round(row["ok"] / row["calls"], 3) if row["calls"] else 0.0
        row["time_ms_total"] = round(row["time_ms_total"], 1)
        row["time_ms_mean"] = round(row["time_ms_total"] / row["calls"], 1) if row["calls"] else 0.0

    return {
        "total_calls": total_calls,
        "finish_calls": finish_calls,
        "parse_failures": parse_failures,
        "turns": max_turn,
        "tools": tools,
        "evidence_kinds": evidence_kinds,
        "new_evidence_total": new_evidence_total,
        "unique_evidence": len(seen_evidence),
        "routine_cost": routine_cost,
    }


def declared_routine_cost(entry: dict) -> dict:
    """What one trace entry declares about the routine behind it (0 when none).

    A tool that runs its own sub-routine (the search skill) publishes
    ``data["cost"] = {"outer_tool_calls", "inner_tool_calls", "vlm_calls"}``:
    the episode pays the outer call, the inner calls and the model calls they
    made are the *reported* price. A tool without such a cost block contributes
    nothing but the (already counted) call itself.
    """
    data = entry.get("data") if isinstance(entry.get("data"), dict) else {}
    cost = data.get("cost") if isinstance(data.get("cost"), dict) else {}
    if not cost:
        return {}
    return {
        "outer_tool_calls": max(0, _as_int(cost.get("outer_tool_calls"), 1)),
        "inner_tool_calls": max(0, _as_int(cost.get("inner_tool_calls"), 0)),
        "vlm_calls": max(0, _as_int(cost.get("vlm_calls"), 0)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # --- scene (mirrors scripts/run_vlm_nav_3dgs.py) ---
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
                        help="flyable floor of the splat world (FLU z, scene-centered); "
                             "with a manifest the default becomes ground_z + 0.2")
    parser.add_argument("--z-max", type=float, default=None,
                        help="no-fly ceiling of the splat world; with a manifest the "
                             "default becomes ground_z + 4.0")
    parser.add_argument("--render-size", type=int, nargs=2, default=(320, 180),
                        metavar=("W", "H"), help="camera resolution for cpu/higs worlds")
    parser.add_argument("--success-dist", type=float, default=2.0,
                        help="meters from the target surface")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-index", type=int, default=None,
                        help="fixed manifest target index (cpu/higs/super)")
    parser.add_argument("--initial-yaw-mode", type=str, default="target",
                        choices=["target", "visible", "random"])
    parser.add_argument("--max-init-attempts", type=int, default=8)
    parser.add_argument("--worker-python", type=str, default=None)
    parser.add_argument("--map-mode", type=str, default="global",
                        choices=["global", "accumulate", "blind"])
    parser.add_argument("--sensing-range-m", type=float, default=6.0)
    parser.add_argument("--sensing", type=str, default="ideal", choices=["ideal", "moge"])
    parser.add_argument("--moge-model-path", type=str, default=None)
    parser.add_argument("--env-overrides", type=str, nargs="*", default=None)
    # --- agent loop ---
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--backend", type=str, default="mock",
                        help="mock | mock:script | hf | openai")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model", type=str, default=None,
                        help="model name for the openai backend")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--detector", type=str, default="off",
                        help="optional grounding detector: off (VLM-only) | "
                             "sam3 (SAM3 masks + DINOv2 crop features) | "
                             "module:factory")
    parser.add_argument("--max-new-tokens", type=int, default=96,
                        help="per-call generation cap; plan turns raise it")
    parser.add_argument("--image-size", type=int, nargs=2, default=(320, 180),
                        metavar=("W", "H"),
                        help="VLM input resize (default: sensor-native 320x180)")
    parser.add_argument("--tool-choice", type=str, default="auto",
                        choices=("auto", "required", "none"),
                        help="native tool_choice sent by the OpenAI backend")
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--no-think", action="store_true",
                        help="disable chain-of-thought in the chat template")
    parser.add_argument("--mock-script", type=str, default=None,
                        help="JSON file with a list of tool calls (default: the "
                             "built-in goal.md acceptance play)")
    parser.add_argument("--toolset", type=str, default=DEFAULT_TOOLSET,
                        choices=list(TOOLSET_CHOICES),
                        help="basic: observe/propose_viewpoints + the three "
                             "navigation-intent tools/verify_goal (fixed-keyframe "
                             "baseline); memory: basic + query_memory/inspect_region/"
                             "ground_target/compare_candidates; full: memory + "
                             "check_path and the M3 task tools "
                             "(unrestricted, the default)")
    parser.add_argument("--max-turns", type=int, default=20,
                        help="model decision turns (one tool call each)")
    parser.add_argument("--tool-budget", type=int, default=40,
                        help="max tool calls per episode")
    parser.add_argument("--context-frames", type=int, default=2,
                        help="recent observation frames re-sent to the VLM")
    parser.add_argument("--recovery", type=str, default=DEFAULT_SWITCH,
                        choices=list(SWITCH_CHOICES),
                        help="on (default): a stagnating episode gets the bounded "
                             "recovery steps (advice, then the honest-finish grant) "
                             "before the loop stops; off: it stops at the stagnation "
                             "limit - the ablation switch of compare_decision_loop")
    parser.add_argument("--state-block", type=str, default=DEFAULT_SWITCH, dest="state_block",
                        choices=list(SWITCH_CHOICES),
                        help="on (default): every prompt carries the fixed STATE block; "
                             "off: it is not injected (context['state'] stays empty)")
    parser.add_argument("--search-skill", type=str, default=DEFAULT_SWITCH, dest="search_skill",
                        choices=list(SWITCH_CHOICES),
                        help="on (default): the registry carries the search_target "
                             "routine; off: it is not registered (the model cannot "
                             "see or call it)")
    parser.add_argument("--mission-plugin", type=str, default="off",
                        dest="mission_plugin", choices=["off", "on"],
                        help="off (default): the pre-plugin tool registry only; "
                             "on: install the motion/planner/mixed mission plugins "
                             "(and the semantic service when the search/approach/"
                             "confirm tools are in the registry)")
    parser.add_argument("--executor-auto-steps", type=str, default=DEFAULT_SWITCH,
                        dest="executor_auto_steps", choices=list(SWITCH_CHOICES),
                        help="on (default): a routine turn whose plan names a "
                             "deterministic next step executes it without spending a "
                             "model turn (capped at 3 consecutive steps); off: every "
                             "call costs a model decision")
    parser.add_argument("--plan-mode", type=str, default="given",
                        dest="plan_mode", choices=["given", "self"],
                        help="given (default): the caller wrote the structured plan "
                             "(an experiment driver bootstrap); self: only the "
                             "natural-language instruction is given and the model "
                             "must write its own sub-task plan with update_plan "
                             "(the runtime asks on the first turn and corrects an "
                             "invalid plan once)")
    parser.add_argument("--memory-spatial", type=str, default=DEFAULT_SWITCH,
                        dest="memory_spatial", choices=list(SWITCH_CHOICES),
                        help="on (default): the memory keeps the spatial store (where "
                             "the task target is / was seen); off: the memory is built "
                             "with spatial_memory=False - the memory layer switch of "
                             "compare_decision_loop's memory arms")
    parser.add_argument("--memory-episodic", type=str, default=DEFAULT_SWITCH,
                        dest="memory_episodic", choices=list(SWITCH_CHOICES),
                        help="on (default): the memory keeps the episodic evidence "
                             "store (sightings with frames, stored frames, scoped "
                             "negatives); off: it is built with episodic_memory=False "
                             "(no evidence is kept; capture records remain)")
    parser.add_argument("--memory-identity", type=str, default=DEFAULT_SWITCH,
                        dest="memory_identity", choices=list(SWITCH_CHOICES),
                        help="on (default): the memory keeps object identities and "
                             "relation anchors (fingerprints, confirm verdicts, "
                             "anchors); off: those fields are removed while frames, "
                             "sightings, places and questions stay (identity ablation)")
    parser.add_argument("--scan-views", type=int, default=8,
                        help="views rendered by observe(mode=lookaround)")
    parser.add_argument("--scan-pitch-deg", type=float, default=50.0)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args(argv)


def build_world(args: argparse.Namespace):
    """World construction that mirrors ``scripts/run_vlm_nav_3dgs.py``."""
    width, height = int(args.render_size[0]), int(args.render_size[1])
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
        z_min = float(getattr(args, "z_min", 0.2))
        z_max = getattr(args, "z_max", None)
        if combo is not None:
            if getattr(args, "auto_start", False):
                start_flu, start_yaw_deg = choose_start_pose(combo, target_index)
                print(
                    f"[vlm_nav_tools] auto start pose: position_flu="
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
        return world_cls(
            ply_path=args.ply or None,
            target=target,
            semantic_manifest=manifest,
            target_index=target_index,
            start_flu=start_flu,
            start_yaw_deg=start_yaw_deg,
            width=width,
            height=height,
            scene_scale=args.scene_scale,
            min_opacity=args.min_opacity,
            z_min=z_min,
            z_max=z_max,
            **extra,
        )
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
        # infeasible cell -> RuntimeError, handled by main() as a skip
        world.reset(seed=args.seed, target_index=args.target_index,
                    initial_yaw_mode=args.initial_yaw_mode,
                    max_init_attempts=args.max_init_attempts)
    else:
        world.reset(seed=args.seed)
    return world


def _load_mock_script(path: str | None) -> list:
    """Tool-call script for the mock backend: dicts, or raw strings (see MockVLM)."""
    if not path:
        return [dict(entry) for entry in ACCEPTANCE_TOOL_SCRIPT]
    with open(path) as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("tool_script") or data.get("script") or data.get("tool_calls") or []
    if not isinstance(data, list):
        raise SystemExit(f"--mock-script {path} must hold a JSON list of tool calls")
    return [entry for entry in data if isinstance(entry, (dict, str))]


class ScriptedToolVLM(MockVLM):
    """Scripted MockVLM that plays the model's side of the tool loop.

    On top of ``MockVLM(mode="tools", tool_script=...)`` it resolves
    ``"$<field>"`` placeholders from the latest tool-result values
    (``bind_result_index``): the script consumes the previous result the way a
    real model reading the returned fields would, so one acceptance script works
    with any runtime, including the module ``AgentRuntime`` that knows nothing
    about mock placeholders. Script entries without a ``tool`` key (e.g.
    ``{"prose": "..."}``) stay verbatim and exercise the runtime's
    unparseable-reply re-prompt path; an exhausted script asks to finish.

    A *grounding* call (``ground_target`` asks the backend for candidates, not
    for the next tool) is answered by ``ground_reply`` instead of by the script:
    the script holds tool calls, so popping one as the grounding answer would
    both fail to parse and shift the play. The default answer is the oracle form
    - the harness's declared target (``fallback_fields["target_pos_flu"]``)
    projected into the frame the tool asked about, exactly the target the mock
    world was built around - which is what lets the built-in acceptance play
    exercise ``ground_target`` -> ``approach_candidate`` with ``--backend mock``.
    """

    name = "mock"

    def __init__(self, script: list | None = None, fallback_fields: dict | None = None,
                 ground_reply: Any = None) -> None:
        super().__init__(mode="tools", tool_script=list(script or []))
        self.result_index: dict = {}
        self.fallback_fields: dict = dict(fallback_fields or {})
        self.ground_reply = ground_reply

    def bind_result_index(self, result_index: dict) -> None:
        self.result_index = result_index

    def _resolved_reply(self, raw: str) -> str:
        parsed = parse_tool_call(raw)
        if parsed is None:
            return raw
        name, args = parsed
        fields = {**self.fallback_fields, **dict(self.result_index)}
        resolved = _resolve_refs(args, fields)
        if resolved == args:
            return raw
        return json.dumps({"tool": name, "args": resolved})

    def _grounding_reply(self, context: Any) -> str | None:
        """The mock's answer to a ``ground_target`` call (None: not a grounding call)."""
        if not isinstance(context, dict) or str(context.get("tool") or "") != "ground_target":
            return None
        if callable(self.ground_reply):
            return self.ground_reply(context)
        target = self.fallback_fields.get("target_pos_flu")
        pose = context.get("pose")
        yaw_deg = context.get("yaw_deg")
        if target is None or pose is None or yaw_deg is None:
            return None
        try:
            from vlm_nav.skills_approach import world_to_pixel
        except Exception:  # pragma: no cover - the projection module is optional
            return None
        pixel = world_to_pixel(target, pose, math.radians(float(yaw_deg)),
                              context.get("intrinsics"))
        if pixel is None:
            # the declared target is not in that frame: an honest empty answer
            return json.dumps({"candidates": []})
        half = 60.0  # a 120x120 normalized window around the target's pixel
        delta = np.asarray(target, dtype=np.float64) - np.asarray(pose, dtype=np.float64)
        forward = float(np.dot(delta[:2], [math.cos(math.radians(float(yaw_deg))),
                                           math.sin(math.radians(float(yaw_deg)))]))
        return json.dumps(
            {
                "candidates": [
                    {
                        "box": [pixel[0] - half, pixel[1] - half,
                                pixel[0] + half, pixel[1] + half],
                        "confidence": 0.9,
                        "distance_m": round(max(forward, 0.0), 3),
                        "attrs": {"source": "scripted-mock-oracle"},
                    }
                ]
            }
        )

    def decide(self, frame_chw, system_prompt, user_prompt, context=None):
        grounding = self._grounding_reply(context)
        if grounding is not None:
            return grounding
        return self._resolved_reply(
            super().decide(frame_chw, system_prompt, user_prompt, context)
        )

    def decide_with_tools(self, frames, system_prompt, user_prompt, tools_text="",
                          context=None):
        return self._resolved_reply(
            super().decide_with_tools(frames, system_prompt, user_prompt, tools_text,
                                      context)
        )


def build_vlm(args: argparse.Namespace):
    if args.backend.startswith("mock"):
        _, _, mode = args.backend.partition(":")
        if mode in ("oracle", "forward"):
            print("[vlm_nav_tools] the tool loop speaks {\"tool\": ...}; "
                  "using the scripted mock policy instead of the action oracle")
        return ScriptedToolVLM(_load_mock_script(args.mock_script))
    if args.backend.startswith("hf"):
        if not args.model_path:
            raise SystemExit("--backend hf requires --model-path")
        kwargs: dict = dict(
            model_path=args.model_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            image_size=tuple(args.image_size),
            num_threads=args.num_threads,
        )
        if args.no_think:
            kwargs["template_kwargs"] = {"enable_thinking": False}
        return make_backend("hf", **kwargs)
    if args.backend == "openai":
        if not args.model:
            raise SystemExit("--backend openai requires --model")
        return make_backend(
            "openai", model=args.model, base_url=args.base_url,
            max_new_tokens=args.max_new_tokens,
            image_size=tuple(args.image_size),
            tool_choice=getattr(args, "tool_choice", "auto"),
        )
    raise SystemExit(f"unknown backend {args.backend!r}")


# ---------------------------------------------------------------------------
# M1 module wiring (preferred) with fallback registration
# ---------------------------------------------------------------------------


def load_m1_modules() -> tuple[dict[str, Any], dict[str, str]]:
    """Import the parallel M1 modules; returns (modules, import_errors)."""
    modules: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for key, dotted in _M1_MODULES.items():
        try:
            modules[key] = importlib.import_module(dotted)
        except ImportError as exc:
            modules[key] = None
            if dotted not in str(exc):
                # the module exists but its own imports failed: surface it
                errors[key] = f"{type(exc).__name__}: {exc}"
    return modules, errors


def _merge_tools(registry: ToolRegistry, source: ToolRegistry,
                 toolset: str | None = None,
                 exclude: Iterable[str] | None = None) -> list[str]:
    """Copy the allowed tools of ``source`` into ``registry`` (by name)."""
    added: list[str] = []
    for tool_name in source.names():
        if tool_name in registry.names() or not toolset_allows(toolset, tool_name, exclude):
            continue
        registry.register(source.get(tool_name))
        added.append(tool_name)
    return added


def _register_module_tools(registry: ToolRegistry, module: Any,
                           vlm: Any = None, toolset: str | None = None,
                           exclude: Iterable[str] | None = None) -> str | None:
    """Register a module's tools, filtered through the requested tool set.

    A module registers on a scratch registry so the tool set can drop names
    before they reach the episode registry: the model must never see (nor be
    able to call) a tool outside its tool set.
    """
    scratch = ToolRegistry()
    for name in _REGISTER_FN_NAMES:
        fn = getattr(module, name, None)
        if callable(fn):
            # registration functions look like register_perception_tools(registry,
            # vlm=None) / register_tools(registry); extra args are filtered out
            _construct(fn, registry=scratch, vlm=vlm)
            _merge_tools(registry, scratch, toolset, exclude)
            return name
    for name in ("build_registry", "make_registry"):
        fn = getattr(module, name, None)
        if callable(fn):
            _merge_tools(registry, fn(), toolset, exclude)
            return name
    return None


def seed_memory_identities(memory: Any, extra: Mapping | None) -> int:
    """Load checkpoint identities/anchors into a freshly built memory store.

    A resumed episode restores the identity layer the interrupted leg saved:
    fingerprints, verdicts/revisions and relation anchors are replayed through
    the layer's own ``remember`` (merging, not replacing), so a retrieval in the
    resumed leg ranks the same instance first. An extra without identities, or
    a memory without the layer, is a no-op.
    """
    if not isinstance(extra, Mapping):
        return 0
    identities = getattr(memory, "identities", None)
    remember = getattr(identities, "remember", None)
    if not callable(remember):
        return 0
    entries = extra.get("identities")
    if isinstance(entries, Mapping):
        entries = entries.get("identities")
    count = 0
    for entry in entries or []:
        if not isinstance(entry, Mapping):
            continue
        remember(
            str(entry.get("candidate_id") or ""),
            target=str(entry.get("target") or ""),
            box=entry.get("box"), position=entry.get("position_flu"),
            anchor=str(entry.get("anchor") or ""),
            anchor_source=str(entry.get("anchor_source") or ""),
            verdict=str(entry.get("verdict") or ""),
            revision=entry.get("revision"),
            observation_ids=entry.get("observation_ids") or (),
            subtask_id=str(entry.get("subtask_id") or ""),
            task_id=str(entry.get("task_id") or ""),
            confidence=entry.get("confidence"),
            note=str(entry.get("note") or ""),
        )
        count += 1
    return count


def build_memory(modules: dict[str, Any], world: Any, out_dir: Path,
                 episode_id: str, spatial_memory: bool = True,
                 episodic_memory: bool = True,
                 identity_memory: bool = True) -> tuple[Any, str, str | None]:
    """The episode memory, built with the three layer switches.

    ``spatial_memory`` / ``episodic_memory`` are the memory implementation's own
    constructor parameters (the decision-loop memory arms); ``identity_memory``
    is the identity/anchor layer (off in the identity ablation arm - frames,
    sightings, places and questions stay). A memory whose signature does not
    accept them is still built, it simply cannot attest the layer (``_construct``
    passes only what the factory accepts).
    """
    module = modules.get("memory")
    detail = None
    if module is not None:
        for name in ("build_memory", "make_memory", "ObservationLog", "Memory"):
            factory = getattr(module, name, None)
            if factory is None:
                continue
            try:
                # ObservationLog(image_dir=..., ...) keeps frames on disk under
                # out_dir/frames, mirroring the fallback frame layout
                return _construct(
                    factory,
                    world=world,
                    out_dir=out_dir,
                    image_dir=out_dir / "frames",
                    episode_id=episode_id,
                    spatial_memory=bool(spatial_memory),
                    episodic_memory=bool(episodic_memory),
                    identity_memory=bool(identity_memory),
                ), "modules", name
            except Exception as exc:
                detail = f"{name}: {type(exc).__name__}: {exc}"
    return _FallbackMemory(), "fallback", detail


class RecordingToolRegistry(ToolRegistry):
    """Registry that indexes every result's data fields and keeps the payloads.

    ``result_index`` lets the scripted mock VLM resolve ``$field`` placeholders
    from the latest tool results (``registry.call`` is the single funnel both
    runtimes go through), and gives the runner a runtime-independent view of what
    the model was told; ``results`` keeps the full (unflattened) payloads of the
    last ``max_entries`` calls so a caller can inspect what a tool returned.
    ``max_entries`` bounds both for long episodes.
    """

    def __init__(self, max_entries: int = 256) -> None:
        super().__init__()
        self.result_index: dict[str, Any] = {}
        self.result_log: list[dict] = []
        self.results: list[dict] = []
        self.max_entries = int(max_entries)

    def call(self, name: str, args: dict, ctx: ToolContext) -> ToolResult:
        result = super().call(name, args, ctx)
        if isinstance(result, ToolResult):
            data = dict(result.data or {})
            for key, value in data.items():
                if isinstance(value, (str, int, float, bool)):
                    self.result_index[key] = value
            # a nested id a script needs as a scalar: the top candidate of a
            # ground_target answer is what a model reads as "candidate_id" and
            # passes to approach_candidate ($candidate_id in a mock script)
            candidates = data.get("candidates")
            if isinstance(candidates, list) and candidates:
                top = candidates[0] if isinstance(candidates[0], dict) else {}
                if top.get("id"):
                    self.result_index["candidate_id"] = str(top["id"])
            self.result_log.append({"tool": name, "status": result.status})
            self.results.append(
                {
                    "tool": name,
                    "args": dict(args or {}),
                    "status": result.status,
                    "data": dict(result.data or {}),
                    "evidence": [e.to_dict() for e in result.evidence],
                }
            )
            del self.result_log[:-self.max_entries]
            del self.results[:-self.max_entries]
        return result

    def last_result(self, tool: str | None = None) -> dict | None:
        """Newest recorded payload, optionally the newest one of ``tool``."""
        for entry in reversed(self.results):
            if tool is None or entry["tool"] == tool:
                return entry
        return None


def build_registry(modules: dict[str, Any], vlm: Any = None,
                   toolset: str | None = None,
                   exclude: Iterable[str] | None = None
                   ) -> tuple[ToolRegistry, str, dict[str, str]]:
    """Tool registry: module tools when available, fallback tools otherwise.

    Fallback specs are skipped for names the modules already registered, so a
    mixed setup (e.g. only perception landed) still yields one working registry
    and a per-tool provenance map in ``summary.json.stack_detail.tools_source``.
    ``toolset`` (basic|memory|full) restricts the catalogue; the default ``full``
    keeps every tool the modules expose. ``exclude`` drops names on top of the
    tool set (``--search-skill off`` drops ``search_target``).
    """
    registry = RecordingToolRegistry()
    module_tools: list[str] = []
    for key in ("perception", "navigation", "task"):
        module = modules.get(key)
        if module is None:
            continue
        try:
            fn_name = _register_module_tools(registry, module, vlm=vlm, toolset=toolset,
                                             exclude=exclude)
        except Exception as exc:
            print(f"[vlm_nav_tools] {key} registration failed: "
                  f"{type(exc).__name__}: {exc}")
            continue
        if fn_name is None:
            print(f"[vlm_nav_tools] {_M1_MODULES[key]} exposes no register function")
            continue
        module_tools = registry.names()
    register_fallback_tools(registry, skip_existing=True, toolset=toolset, exclude=exclude)
    tools_source = {name: ("modules" if name in module_tools else "fallback")
                    for name in registry.names()}
    backend = "modules" if module_tools else "fallback"
    return registry, backend, tools_source


def build_runtime(modules: dict[str, Any], **kwargs) -> tuple[Any, str, str | None]:
    """Instantiate the module AgentRuntime when available."""
    module = modules.get("runtime")
    if module is None:
        return None, "fallback", None
    cls = None
    for name in ("AgentRuntime", "ToolRuntime", "ToolAgentRuntime"):
        candidate = getattr(module, name, None)
        if candidate is not None:
            cls = candidate
            break
    if cls is None:
        return None, "fallback", "runtime module exposes no AgentRuntime"
    try:
        return _construct(cls, **kwargs), "modules", None
    except Exception as exc:
        return None, "fallback", f"{type(cls).__name__}: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Fallback stack (TODO(M1): delete once vlm_nav/{runtime,memory,tool_impl_*}.py land)
# ---------------------------------------------------------------------------


class _FallbackMemory:
    """Minimal observation log so the tool loop can query past evidence."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def add(self, observation_id: str, kind: str, ref: str = "", note: str = "",
            pose: dict | None = None, data: dict | None = None) -> dict:
        record = {
            "observation_id": observation_id,
            "kind": kind,
            "ref": ref,
            "note": note,
            "pose": pose or {},
            "data": data or {},
        }
        self.records.append(record)
        return record

    def get(self, observation_id: str) -> dict | None:
        for record in reversed(self.records):
            if record["observation_id"] == observation_id:
                return record
        return None

    def latest(self, kind: str | None = None) -> dict | None:
        for record in reversed(self.records):
            if kind is None or record["kind"] == kind:
                return record
        return None

    def search(self, query: str, kind: str | None = None, limit: int = 3) -> list[dict]:
        needle = (query or "").strip().lower()
        hits = [
            record
            for record in self.records
            if (kind is None or record["kind"] == kind)
            and (not needle or needle in json.dumps(record, default=str).lower())
        ]
        return list(reversed(hits[-max(1, int(limit)):]))

    def stats(self) -> dict:
        kinds: dict[str, int] = {}
        for record in self.records:
            kinds[record["kind"]] = kinds.get(record["kind"], 0) + 1
        return {"records": len(self.records), "kinds": kinds}


class _MemoryAdapter:
    """Uniform view over ``_FallbackMemory`` or an M1 memory implementation."""

    _ADD = ("add_observation", "add", "record", "log_observation", "append")
    _GET = ("get", "get_observation", "lookup")
    _LATEST = ("latest", "latest_observation", "last")
    _SEARCH = ("search", "query", "retrieve", "find")

    def __init__(self, memory: Any) -> None:
        self.raw = memory

    def _records(self) -> list:
        for attr in ("records", "entries", "observations"):
            value = getattr(self.raw, attr, None)
            if isinstance(value, list):
                return value
        return []

    def add(self, observation_id: str, kind: str, ref: str = "", note: str = "",
            pose: Any = None, yaw: float | None = None,
            data: dict | None = None) -> Any:
        # alias soup on purpose: memory implementations name these differently
        # (memory.ObservationLog: record(image, pose, yaw=..., obs_id=..., extra=...))
        kwargs = dict(observation_id=observation_id, obs_id=observation_id, kind=kind,
                      ref=ref, image=ref, image_path=ref, frame=ref, note=note,
                      pose=pose, position=pose, yaw=yaw, data=data, extra=data)
        for name in self._ADD:
            fn = getattr(self.raw, name, None)
            if callable(fn):
                try:
                    return _construct(fn, **kwargs)
                except Exception:
                    continue
        return None

    def get(self, observation_id: str) -> Any:
        for name in self._GET:
            fn = getattr(self.raw, name, None)
            if callable(fn):
                try:
                    found = fn(observation_id)
                except Exception:
                    found = None
                if found is not None:
                    return found
        for record in reversed(self._records()):
            view = _record_view(record)
            if view["observation_id"] == observation_id:
                return record
        return None

    def latest(self, kind: str | None = None) -> Any:
        for name in self._LATEST:
            fn = getattr(self.raw, name, None)
            if callable(fn):
                try:
                    found = _construct(fn, kind=kind)
                except Exception:
                    found = None
                if found is not None:
                    return found
        for record in reversed(self._records()):
            if kind is None or _record_view(record)["kind"] == kind:
                return record
        return None

    def search(self, query: str, kind: str | None = None, limit: int = 3) -> list:
        for name in self._SEARCH:
            fn = getattr(self.raw, name, None)
            if callable(fn):
                try:
                    # "q" is memory.ObservationLog.query(q=..., limit=...)
                    hits = _construct(fn, query=query, q=query, text=query,
                                      kind=kind, limit=limit)
                except Exception:
                    continue
                if hits is None:
                    continue
                return list(hits)[: max(1, int(limit))]
        needle = (query or "").strip().lower()
        hits = [
            record
            for record in self._records()
            if (kind is None or _record_view(record)["kind"] == kind)
            and (not needle or needle in json.dumps(record, default=str).lower())
        ]
        return list(reversed(hits[-max(1, int(limit)):]))

    def stats(self) -> Any:
        fn = getattr(self.raw, "stats", None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                return None
        return None


def _record_view(record: Any) -> dict:
    """Normalize a memory record into the fields the tools expose."""
    data = _as_mapping(record)
    if not data:
        return {"observation_id": str(record), "kind": "observation", "image": None,
                "note": str(record), "record": record}

    def first(*keys: str) -> Any:
        for key in keys:
            value = data.get(key)
            if value not in (None, ""):
                return value
        return None

    observation_id = first("observation_id", "obs_id", "id", "name", "ref")
    image = first("image", "image_path", "frame", "frame_path", "ref", "path")
    note = first("note", "text", "summary", "description", "query")
    return {
        "observation_id": str(observation_id) if observation_id is not None else "",
        "kind": str(first("kind", "type") or "observation"),
        "image": str(image) if image is not None else None,
        "note": str(note) if note is not None else "",
        "record": data,
    }


def _memory_add(ctx: ToolContext, observation_id: str, kind: str, ref: str = "",
                note: str = "", data: dict | None = None) -> None:
    if ctx.memory is None:
        return
    try:
        position = None
        yaw = None
        try:
            position = [float(v) for v in np.asarray(ctx.world.position_flu).reshape(-1)[:3]]
            yaw = float(ctx.world.yaw_rad)
        except Exception:
            pass
        _MemoryAdapter(ctx.memory).add(
            observation_id=observation_id, kind=kind, ref=ref, note=note,
            pose=position, yaw=yaw, data=data,
        )
    except Exception:
        pass


def _memory_lookup(ctx: ToolContext, observation_id: str) -> dict | None:
    if ctx.memory is None:
        return None
    try:
        found = _MemoryAdapter(ctx.memory).get(observation_id)
    except Exception:
        return None
    return _record_view(found) if found is not None else None


def _image_ref_for(ctx: ToolContext, observation_id: str, view: dict | None) -> str | None:
    index = ctx.extra.get("observation_index") or {}
    if observation_id in index:
        return index[observation_id]
    if view and view.get("image"):
        return str(view["image"])
    return None


# --- fallback tool handlers -------------------------------------------------


def _tool_observe(ctx: ToolContext, args: dict) -> ToolResult:
    world = ctx.world
    mode = str(args.get("mode") or "forward").lower()
    obs_id = ctx.next_obs_id()
    evidence: list[EvidenceRef] = []
    data: dict = {"observation_id": obs_id}
    frame: torch.Tensor

    if mode == "lookaround":
        n_views = max(4, int(args.get("n_views") or ctx.extra.get("scan_views", 8)))
        pitch = float(args.get("pitch_down_deg", ctx.extra.get("scan_pitch_deg", 0.0)))
        views = world.scan_views(n_views, pitch_down_deg=pitch)
        strip = _hstack(views)
        ref = _save_frame(ctx, strip, f"{obs_id}_scan.png")
        frame = strip
        yaw_deg = math.degrees(float(world.yaw_rad))
        data.update(
            {
                "mode": "lookaround",
                "n_views": len(views),
                "view_headings_deg": [round((yaw_deg + 360.0 * k / n_views) % 360.0, 1)
                                      for k in range(n_views)],
            }
        )
        evidence.append(EvidenceRef("image", ref, f"{len(views)}-view scan strip"))
    elif mode == "bev":
        bev = world.bev_image(target_estimate=None, size=192)
        ref = _save_frame(ctx, bev, f"{obs_id}_bev.png")
        frame = bev
        data.update({"mode": "bev"})
        evidence.append(EvidenceRef("map", ref, "top-down map"))
    else:
        mode = "forward"
        frame = world.render()
        ref = _save_frame(ctx, frame, f"{obs_id}.png")
        data.update({"mode": "forward"})
        evidence.append(EvidenceRef("image", ref, "forward camera"))

    data.update(_pose_fields(world))
    evidence.append(EvidenceRef("observation", obs_id, mode))

    ctx.extra.setdefault("observation_index", {})[obs_id] = ref
    ctx.extra["last_observation_id"] = obs_id
    _cache_frame(ctx, frame)
    _memory_add(
        ctx, obs_id, "observation", ref,
        note=f"{mode} observation captured while working on: {_task_text(ctx)}",
        data=data,
    )
    return ToolResult(STATUS_OK, data, evidence)


def _tool_inspect_region(ctx: ToolContext, args: dict) -> ToolResult:
    obs_id = str(args.get("observation_id") or ctx.extra.get("last_observation_id") or "")
    view = _memory_lookup(ctx, obs_id)
    image_ref = _image_ref_for(ctx, obs_id, view)
    if not obs_id or not image_ref:
        return ToolResult(
            STATUS_EMPTY,
            {"reason": "unknown_observation", "observation_id": obs_id},
        )
    source = _ctx_out_dir(ctx) / image_ref
    if not source.exists():
        return ToolResult(
            STATUS_EMPTY,
            {"reason": "observation_image_missing", "observation_id": obs_id,
             "image": image_ref},
        )

    from PIL import Image

    with Image.open(source) as handle:
        image = handle.convert("RGB")
        width, height = image.size
        # the same box contract the module tools speak (canonical corner box);
        # the legacy size spelling is read, never assumed
        region, region_format, problem = _parse_region(
            args.get("region"), width, height, args.get("box_format")
        )
        if problem:
            return ToolResult(
                STATUS_FAILED,
                {"error": problem, "region": args.get("region"),
                 "observation_id": obs_id, "box_format": BOX_FORMAT_DOC},
            )
        box = (
            int(region[0] / 1000.0 * width),
            int(region[1] / 1000.0 * height),
            int(region[2] / 1000.0 * width),
            int(region[3] / 1000.0 * height),
        )
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            box = (int(0.35 * width), int(0.35 * height), int(0.65 * width), int(0.65 * height))
        crop = image.crop(box)
        zoom = max(1.0, float(args.get("zoom", 2.0)))
        crop = crop.resize((max(1, int(crop.width * zoom)), max(1, int(crop.height * zoom))),
                           Image.BILINEAR)
        ref = f"frames/{obs_id}_crop.png"
        path = _ctx_out_dir(ctx) / ref
        path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(path)

    cached = torch.from_numpy(
        np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / 255.0
    ).contiguous()
    _cache_frame(ctx, cached)
    _memory_add(ctx, f"{obs_id}_crop", "image", ref,
                note=f"zoom crop of {obs_id} while working on: {_task_text(ctx)}",
                data={"source_observation_id": obs_id})
    return ToolResult(
        STATUS_OK,
        {
            "observation_id": f"{obs_id}_crop",
            "source_observation_id": obs_id,
            "region": [round(float(v), 1) for v in region],
            "region_format": region_format,
            "box_format": BOX_FORMAT_DOC,
            "crop_px": list(box),
            "source_px": [width, height],
            "zoom": zoom,
        },
        [EvidenceRef("image", ref, f"zoomed crop of {obs_id}"),
         EvidenceRef("observation", obs_id, "source frame")],
    )


def _tool_query_memory(ctx: ToolContext, args: dict) -> ToolResult:
    query = str(args.get("query") or "")
    kind = args.get("kind")
    limit = max(1, int(args.get("limit", 3)))
    hits: list[dict] = []
    adapter = _MemoryAdapter(ctx.memory) if ctx.memory is not None else None
    if adapter is not None:
        try:
            raw_hits = adapter.search(query, kind=str(kind) if kind else None, limit=limit)
        except Exception:
            raw_hits = []
        hits = [_record_view(hit) for hit in raw_hits]
    if not hits:
        # report what the archive does hold, so the model can follow up with
        # inspect_region/observe instead of only learning "nothing matched"
        recent: list[dict] = []
        if adapter is not None:
            try:
                recent = [_record_view(item) for item in adapter.search("", limit=limit)]
            except Exception:
                recent = []
        evidence = [EvidenceRef("memory", item["observation_id"], item["note"][:80])
                    for item in recent if item["observation_id"]]
        return ToolResult(
            STATUS_EMPTY,
            {"query": query, "n_hits": 0, "reason": "no_matching_memory",
             "recent_observation_ids": [item["observation_id"] for item in recent
                                        if item["observation_id"]],
             "hint": "no observation text matched the query; "
                     "inspect_region a recent id or observe again",
             "last_observation_id": ctx.extra.get("last_observation_id")},
            evidence,
        )
    matches = [
        {
            "observation_id": hit["observation_id"],
            "kind": hit["kind"],
            "note": hit["note"],
            "image": hit["image"],
            "pose": (hit["record"] or {}).get("pose") if isinstance(hit["record"], dict) else None,
        }
        for hit in hits
    ]
    evidence: list[EvidenceRef] = []
    for match in matches:
        evidence.append(EvidenceRef("memory", match["observation_id"], match["note"][:80]))
        if match["image"] and (_ctx_out_dir(ctx) / str(match["image"])).exists():
            evidence.append(EvidenceRef("image", str(match["image"]), "recalled observation"))
    return ToolResult(
        STATUS_OK,
        {"query": query, "n_hits": len(matches), "matches": matches,
         "last_observation_id": ctx.extra.get("last_observation_id")},
        evidence,
    )


def _reachable(world: Any, point: np.ndarray) -> tuple[bool, str]:
    z_min = getattr(world, "z_min", None)
    z_max = getattr(world, "z_max", None)
    bound = getattr(world, "bound_radius", None)
    if z_min is not None and point[2] < float(z_min):
        return False, "below_floor"
    if z_max is not None and math.isfinite(float(z_max)) and point[2] > float(z_max):
        return False, "above_ceiling"
    if bound is not None and math.hypot(point[0], point[1]) > float(bound):
        return False, "outside_bounds"
    return True, ""


def _tool_propose_viewpoints(ctx: ToolContext, args: dict) -> ToolResult:
    world = ctx.world
    n = max(1, int(args.get("n", 3)))
    radius = max(0.5, float(args.get("radius_m", 2.0)))
    pos = np.asarray(world.position_flu, dtype=np.float64)
    yaw = float(world.yaw_rad)
    trajectory = np.asarray(getattr(world, "trajectory", []), dtype=np.float64).reshape(-1, 3)

    n_sample = max(4 * n, 12)
    scored = []
    for k in range(n_sample):
        bearing_deg = (math.degrees(yaw) + 360.0 * k / n_sample) % 360.0
        radians = math.radians(bearing_deg)
        point = pos + radius * np.asarray([math.cos(radians), math.sin(radians), 0.0])
        reachable, reason = _reachable(world, point)
        gap = (
            float(np.min(np.linalg.norm(trajectory[:, :2] - point[:2], axis=1)))
            if len(trajectory) else float("inf")
        )
        scored.append(
            {
                "bearing_deg": round(bearing_deg, 1),
                "distance_m": round(radius, 2),
                "world_flu": [round(float(v), 3) for v in point],
                "reachable": reachable,
                "unreachable_reason": reason,
                "trajectory_gap_m": round(gap if math.isfinite(gap) else radius, 2),
                "score": (gap if math.isfinite(gap) else radius) - (0.0 if reachable else 1e3),
            }
        )
    scored.sort(key=lambda item: -item["score"])
    seq = int(ctx.extra.get("viewpoint_seq", 0))
    viewpoints = []
    for item in scored[:n]:
        seq += 1
        item = dict(item)
        item["viewpoint_id"] = f"vp_{seq}"
        item.pop("score", None)
        item["reachable"] = bool(item["reachable"])
        viewpoints.append(item)
        ctx.extra.setdefault("viewpoints", {})[item["viewpoint_id"]] = item
    ctx.extra["viewpoint_seq"] = seq

    map_seq = int(ctx.extra.get("map_seq", 0)) + 1
    ctx.extra["map_seq"] = map_seq
    bev = world.bev_image(target_estimate=None, size=192)
    ref = _save_frame(ctx, bev, f"map_{map_seq:03d}_bev.png")
    _cache_frame(ctx, bev)
    _memory_add(ctx, f"map_{map_seq}", "map", ref,
                note=f"map snapshot before moving while working on: {_task_text(ctx)}",
                data={"viewpoints": [v["viewpoint_id"] for v in viewpoints]})

    if not viewpoints:
        return ToolResult(STATUS_EMPTY, {"reason": "no_viewpoints", "n": 0},
                          [EvidenceRef("map", ref, "top-down map")])
    data = {
        "status": STATUS_OK,
        "viewpoint_id": viewpoints[0]["viewpoint_id"],
        "viewpoints": viewpoints,
        "radius_m": radius,
        "map_observation_id": f"map_{map_seq}",
        "last_observation_id": ctx.extra.get("last_observation_id"),
    }
    return ToolResult(STATUS_OK, data,
                      [EvidenceRef("map", ref, "top-down map"),
                       EvidenceRef("trajectory", f"map_{map_seq}",
                                   "trajectory used for novelty scoring")])


def _tool_navigate(ctx: ToolContext, args: dict) -> ToolResult:
    world = ctx.world
    viewpoint_id = args.get("viewpoint_id")
    viewpoint = None
    if viewpoint_id:
        viewpoint = (ctx.extra.get("viewpoints") or {}).get(str(viewpoint_id))
        if viewpoint is None:
            return ToolResult(
                STATUS_FAILED,
                {"reason": "unknown_viewpoint", "viewpoint_id": str(viewpoint_id),
                 "known_viewpoints": sorted((ctx.extra.get("viewpoints") or {}).keys())},
            )
        target_flu = np.asarray(viewpoint["world_flu"], dtype=np.float64)
    elif args.get("bearing_deg") is not None:
        distance = float(args.get("distance_m") or 2.0)
        radians = math.radians(float(args["bearing_deg"]))
        target_flu = np.asarray(world.position_flu, dtype=np.float64) + distance * np.asarray(
            [math.cos(radians), math.sin(radians), 0.0]
        )
    else:
        return ToolResult(STATUS_FAILED, {"reason": "missing_target",
                                          "hint": "viewpoint_id or bearing_deg+distance_m"})

    start = np.asarray(world.position_flu, dtype=np.float64)
    delta = target_flu[:2] - start[:2]
    distance = float(np.linalg.norm(delta))
    seq = int(ctx.extra.get("nav_seq", 0)) + 1
    ctx.extra["nav_seq"] = seq
    task_id = f"nav_{seq}"

    yaw = float(world.yaw_rad)
    heading_error = math.atan2(float(delta[1]), float(delta[0])) - yaw
    heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
    action = NavAction(
        kind="waypoint",
        forward_m=distance * math.cos(heading_error),
        right_m=-distance * math.sin(heading_error),
        up_m=float(target_flu[2] - start[2]),
        yaw_deg=max(-180.0, min(180.0, math.degrees(heading_error))),
        reason=f"navigate to {viewpoint_id or 'bearing target'}",
    )
    feedback = ""
    try:
        feedback = str(world.apply_action(action))
    except Exception as exc:
        return ToolResult(STATUS_FAILED, {"task_id": task_id,
                                          "reason": "action_error",
                                          "error": f"{type(exc).__name__}: {exc}"})
    refresh = getattr(world, "refresh_sensing", None)
    if callable(refresh):
        try:
            refresh()
        except Exception:
            pass

    end = np.asarray(world.position_flu, dtype=np.float64)
    remaining = float(np.linalg.norm(target_flu - end))
    progress = float(np.linalg.norm(end[:2] - start[:2]))
    outcome = "moved"
    status = STATUS_OK
    reason = "reached" if remaining <= 0.5 else "partial"
    lowered = feedback.lower()
    if "collision" in lowered:
        status, outcome, reason = STATUS_BLOCKED, "collision", "new_obstacle"
    elif "stalled" in lowered:
        status, outcome, reason = STATUS_BLOCKED, "stalled", "stalled"
    elif "planner failed" in lowered:
        status, outcome, reason = STATUS_BLOCKED, "planner_failed", "planner_failed"
    elif "timeout" in lowered:
        outcome, reason = "timeout", "leg_timeout"

    data = {
        "status": status,
        "task_id": task_id,
        "viewpoint_id": str(viewpoint_id) if viewpoint_id else None,
        "target_flu": [round(float(v), 2) for v in target_flu],
        "outcome": outcome,
        "reason": reason,
        "arrived": bool(remaining <= 0.5),
        "progress_m": round(progress, 2),
        "remaining_m": round(remaining, 2),
        "last_observation_id": ctx.extra.get("last_observation_id"),
    }
    data.update(_pose_fields(world))
    evidence = [EvidenceRef("trajectory", task_id,
                            f"leg {[round(float(v), 2) for v in start]} -> "
                            f"{[round(float(v), 2) for v in end]}")]
    last_obs = ctx.extra.get("last_observation_id")
    if last_obs:
        evidence.append(EvidenceRef("observation", last_obs, "observation before the leg"))
    _memory_add(ctx, task_id, "event", "",
                note=f"navigation leg to {viewpoint_id or 'bearing target'} ({outcome})",
                data={"task_id": task_id, "outcome": outcome, "progress_m": data["progress_m"]})
    return ToolResult(status, data, evidence)


def _fallback_position(value: Any) -> np.ndarray | None:
    try:
        point = np.asarray(value, dtype=np.float64).reshape(-1)[:3]
    except (TypeError, ValueError):
        return None
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return None
    return point


def _fallback_candidate_position(ctx: ToolContext, candidate_id: Any) -> np.ndarray | None:
    """The agent's own candidate estimate (never ``world.target``)."""
    key = str(candidate_id)
    table = ctx.extra.get("candidate_positions")
    if isinstance(table, dict):
        point = _fallback_position(table.get(key, table.get(candidate_id)))
        if point is not None:
            return point
    store = ctx.extra.get("perception.candidates")
    if isinstance(store, dict):
        record = store.get(key)
        if isinstance(record, dict):
            point = _fallback_position(record.get("position_flu"))
            if point is not None:
                return point
    for item in ctx.extra.get("candidates") or []:
        if isinstance(item, dict) and str(item.get("id")) == key:
            point = _fallback_position(item.get("position_flu"))
            if point is not None:
                return point
    return None


def _fallback_observation_position(ctx: ToolContext, observation_id: str) -> np.ndarray | None:
    record = (ctx.extra.get("observations") or {}).get(observation_id)
    if isinstance(record, dict):
        point = _fallback_position(record.get("position_flu"))
        if point is not None:
            return point
    store = ctx.extra.get("observation_store")
    getter = getattr(store, "get", None)
    if callable(getter):
        observation = getter(observation_id)
        point = _fallback_position(getattr(observation, "pose_flu", None))
        if point is not None:
            return point
    return None


def _fallback_identity_status(ctx: ToolContext, candidate_id: str) -> dict:
    scope = ctx.extra.get("task_scope") or {}
    revision = int(scope.get("evidence_revision", 0) or 0)
    ledger = ctx.extra.get("confirmation_ledger")
    entry = ledger.get(candidate_id) if isinstance(ledger, dict) else {}
    entry = entry if isinstance(entry, dict) else {}
    store = ctx.extra.get("perception.candidates")
    record = store.get(candidate_id) if isinstance(store, dict) else {}
    record = record if isinstance(record, dict) else {}
    status = str(entry.get("last_verdict") or record.get("confirmation_status") or "")
    if not status:
        status = {"confirmed": "accepted"}.get(str(record.get("status") or ""), "unconfirmed")
    accepted_revision = entry.get("accepted_revision", record.get("confirmation_revision"))
    try:
        accepted_revision = None if accepted_revision is None else int(accepted_revision)
    except (TypeError, ValueError):
        accepted_revision = None
    if status == "accepted" and (
        accepted_revision is None or accepted_revision != revision
    ):
        status = "stale"
    return {
        "status": status,
        "candidate_id": candidate_id,
        "evidence_revision": revision,
        "accepted_revision": accepted_revision,
        "source": "confirm_candidate",
    }


def _tool_verify_goal(ctx: ToolContext, args: dict) -> ToolResult:
    """Fallback completion check over the agent's own state (no GT reads)."""
    world = ctx.world
    if args.get("success_dist_m") is not None:
        limit = float(args["success_dist_m"])
    elif ctx.extra.get("done_dist") is not None:
        limit = float(ctx.extra["done_dist"])
    else:
        limit = float(ctx.extra.get("success_dist_m", 2.0))
    obs_id = ctx.next_obs_id()
    frame = world.render()
    ref = _save_frame(ctx, frame, f"{obs_id}_verify.png")
    _cache_frame(ctx, frame)

    position = _fallback_position(getattr(world, "position_flu", None))
    measured_distance = None
    distance_source = ""
    checks: dict[str, Any] = {}
    unmet: list[dict] = []

    pixel = args.get("pixel")
    if isinstance(pixel, (list, tuple)) and len(pixel) >= 2:
        depth_fn = getattr(world, "depth_at_pixel", None)
        depth = None
        if callable(depth_fn):
            try:
                depth = float(depth_fn((float(pixel[0]), float(pixel[1]))))
            except Exception:  # noqa: BLE001 - no depth is an honest uncertainty
                depth = None
        if depth is not None:
            measured_distance = depth
            distance_source = "depth"
            checks["depth_m"] = round(depth, 3)
        else:
            unmet.append({"check": "depth", "reason": "no_depth_estimate"})

    candidate_id = str(args.get("candidate_id") or "")
    identity = {
        "status": "not_checked",
        "candidate_id": candidate_id,
        "evidence_revision": int((ctx.extra.get("task_scope") or {}).get("evidence_revision", 0) or 0),
        "source": "confirm_candidate",
    }
    if candidate_id:
        candidate_position = _fallback_candidate_position(ctx, candidate_id)
        if candidate_position is None:
            unmet.append(
                {
                    "check": "candidate_identity",
                    "reason": "unknown_candidate",
                    "candidate_id": candidate_id,
                }
            )
        elif measured_distance is None and position is not None:
            measured_distance = float(np.linalg.norm(candidate_position - position))
            distance_source = "dead_reckoning"
        identity = _fallback_identity_status(ctx, candidate_id)
        checks["identity_status"] = dict(identity)
        if identity["status"] == "rejected":
            unmet.append(
                {"check": "identity_status", "status": "rejected",
                 "candidate_id": candidate_id}
            )
        elif identity["status"] == "stale":
            unmet.append(
                {"check": "identity_status", "status": "stale",
                 "candidate_id": candidate_id}
            )

    observation_id = str(args.get("observation_id") or "")
    if observation_id and measured_distance is None:
        recalled = _fallback_observation_position(ctx, observation_id)
        if recalled is None:
            unmet.append(
                {"check": "observation_pose", "reason": "unknown_observation",
                 "observation_id": observation_id}
            )
        else:
            checks["recalled_observation_position"] = [float(v) for v in recalled]
            if position is not None:
                measured_distance = float(np.linalg.norm(recalled - position))
                distance_source = "dead_reckoning"

    plan = ctx.extra.get("plan")
    subtasks = plan.get("subtasks") if isinstance(plan, dict) else None
    if isinstance(subtasks, list) and subtasks:
        done = [
            str(item.get("id")) for item in subtasks
            if isinstance(item, dict) and str(item.get("status")) == "done"
        ]
        plan_completion = {
            "available": True,
            "source": "plan",
            "plan_id": str(plan.get("plan_id") or ""),
            "revision": int(plan.get("revision", 0) or 0),
            "complete": len(done) == len(subtasks),
            "done_subtasks": done,
            "open_subtasks": [
                {"subtask_id": str(item.get("id")), "status": str(item.get("status"))}
                for item in subtasks
                if isinstance(item, dict) and str(item.get("status")) != "done"
            ],
        }
    else:
        plan_completion = {"available": False, "source": "plan"}

    arrivals = [
        item for item in evidence_pack(ctx).get("arrivals") or ()
        if isinstance(item, dict) and str(item.get("outcome")) == "arrived"
    ]
    arrival = (
        {
            "available": True,
            "source": "evidence_pack",
            "task_id": str(arrivals[-1].get("task_id") or ""),
            "candidate_id": str(arrivals[-1].get("candidate_id") or ""),
            "intent": str(arrivals[-1].get("intent") or ""),
            "outcome": "arrived",
        }
        if arrivals
        else {"available": False, "source": "evidence_pack"}
    )

    checks["distance_limit_m"] = round(limit, 3)
    if measured_distance is not None:
        checks["measured_distance_m"] = round(measured_distance, 3)
        checks["distance_source"] = distance_source
        if measured_distance > limit:
            unmet.append(
                {
                    "check": "measured_distance_m",
                    "value_m": round(measured_distance, 3),
                    "max_m": round(limit, 3),
                    "source": distance_source,
                }
            )
    checks["plan_completion"] = plan_completion
    checks["arrival"] = arrival

    rejected = identity["status"] == "rejected"
    too_far = any(item.get("check") == "measured_distance_m" for item in unmet)
    has_self_evidence = bool(
        measured_distance is not None
        or identity["status"] == "accepted"
        or arrival["available"]
    )
    if rejected or too_far or (pixel is not None and measured_distance is None
                               and any(item.get("check") == "depth" for item in unmet)):
        verified: Any = False
        reason = "unmet_conditions"
    elif identity["status"] == "stale":
        verified = "uncertain"
        reason = "stale_identity"
    elif not has_self_evidence:
        verified = "uncertain"
        reason = "no_evidence"
    else:
        verified = True
        reason = "verified"

    data = {
        "verified": verified,
        "observation_id": obs_id,
        "checks": checks,
        "unmet": unmet,
        "reason": reason,
        "measured_distance_m": (
            None if measured_distance is None else round(measured_distance, 3)
        ),
        "distance_source": distance_source,
        "identity_status": identity,
        "plan_completion": plan_completion,
        "arrival": arrival,
        "last_observation_id": obs_id,
    }
    data.update(_pose_fields(world))
    evidence = [EvidenceRef("image", ref, "verification frame"),
                EvidenceRef("observation", obs_id, "verification observation")]
    _memory_add(ctx, obs_id, "observation", ref,
                note=f"verification frame while working on: {_task_text(ctx)}",
                data=data)
    return ToolResult(STATUS_OK, data, evidence)


def register_fallback_tools(registry: ToolRegistry, skip_existing: bool = True,
                            toolset: str | None = None,
                            exclude: Iterable[str] | None = None) -> set[str]:
    """Register the fallback M1 tool specs; returns the names actually added.

    ``toolset`` drops names outside the requested set (``full`` keeps them all),
    ``exclude`` drops the mechanism switches' names on top of it.
    """
    specs = [
        ToolSpec(
            name="observe",
            description="Look through the drone camera (mode=forward), sweep the "
                        "surroundings (mode=lookaround) or render the top-down map "
                        "(mode=bev). Returns an observation id.",
            input_schema={
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["forward", "lookaround", "bev"]},
                    "n_views": {"type": "integer", "minimum": 4},
                    "pitch_down_deg": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": [],
            },
            handler=_tool_observe,
            side_effects=False,
        ),
        ToolSpec(
            name="inspect_region",
            description=(
                "Zoom into a region of a previous observation to resolve small or "
                "unclear objects. region = " + BOX_FORMAT_DOC + " (the legacy "
                "[u, v, w, h] size spelling is understood; the answer names the "
                "region_format it used)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "observation_id": {"type": "string"},
                    "region": {"type": "array", "items": {"type": "number"}},
                    "box_format": {"type": "string", "enum": ["xyxy", "uvwh"]},
                    "zoom": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["observation_id", "region"],
            },
            handler=_tool_inspect_region,
            side_effects=False,
        ),
        ToolSpec(
            name="query_memory",
            description="Search past observations/events for a target, place or "
                        "event; returns matching observation ids and image refs.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "kind": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1},
                },
                "required": ["query"],
            },
            handler=_tool_query_memory,
            side_effects=False,
        ),
        ToolSpec(
            name="propose_viewpoints",
            description="Propose reachable observation points around the current "
                        "pose (bearing/distance/world position + novelty), so the "
                        "drone explores instead of turning in place.",
            input_schema={
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "minimum": 1},
                    "radius_m": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": [],
            },
            handler=_tool_propose_viewpoints,
            side_effects=False,
        ),
        ToolSpec(
            name="navigate",
            description="DEPRECATED fallback leg (the module stack splits this into "
                        "approach_candidate / go_to_viewpoint / "
                        "return_to_observation, which no fallback implements): fly "
                        "to a proposed viewpoint (viewpoint_id) or to a "
                        "bearing_deg/distance_m target. Returns a task id, arrival "
                        "flag and structured progress (blocked on collision/stall).",
            input_schema={
                "type": "object",
                "properties": {
                    "viewpoint_id": {"type": "string"},
                    "bearing_deg": {"type": "number"},
                    "distance_m": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": [],
            },
            handler=_tool_navigate,
            side_effects=True,
        ),
        ToolSpec(
            name="verify_goal",
            description="Verify completion from the agent's own records: a measured "
                        "distance (pixel depth or its own candidate/observation "
                        "estimate), confirmation identity_status, plan_completion and "
                        "arrival evidence - never scene ground truth.",
            input_schema={
                "type": "object",
                "properties": {
                    "pixel": {"type": "array", "items": {"type": "number"},
                              "minItems": 2, "maxItems": 2},
                    "candidate_id": {"type": ["string", "integer"]},
                    "observation_id": {"type": "string"},
                    "success_dist_m": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": [],
            },
            handler=_tool_verify_goal,
            side_effects=False,
        ),
    ]
    added: set[str] = set()
    for spec in specs:
        if skip_existing and spec.name in registry.names():
            continue
        if not toolset_allows(toolset, spec.name, exclude):
            continue
        registry.register(spec)
        added.add(spec.name)
    return added


def _system_prompt(instruction: str, registry: ToolRegistry) -> str:
    return (
        "You are the navigation agent of a camera drone flying inside a 3D scene.\n"
        f"Mission: {instruction}\n\n"
        "You work by calling tools and reading their structured results. "
        "Inspect the attached camera/observation images before deciding.\n"
        "Available tools:\n"
        f"{registry.specs_text()}\n\n"
        f"{finish_contract_doc()}"
    )


def _user_prompt(turn: int, max_turns: int, budget_left: int, n_frames: int,
                 last_result_text: str) -> str:
    return "\n".join(
        [
            f"Turn {turn} of {max_turns}; tool calls left: {budget_left}.",
            f"You see {n_frames} image(s): observation history first, current view last.",
            "Archived observations and frames stay available through query_memory "
            "and inspect_region.",
            "Last tool result:",
            last_result_text,
        ]
    )


class _ToolRuntimeFallback:
    """Minimal M1 tool loop.

    TODO(M1): delete once ``vlm_nav.runtime.AgentRuntime`` exists; this
    runner prefers that module and only falls back to this loop.
    """

    def __init__(
        self,
        world: Any,
        vlm: Any,
        registry: ToolRegistry,
        ctx: ToolContext,
        instruction: str,
        max_turns: int = 20,
        context_frames: int = 2,
        success_dist_m: float = 2.0,
    ) -> None:
        self.world = world
        self.vlm = vlm
        self.registry = registry
        self.ctx = ctx
        self.instruction = instruction
        self.max_turns = max(1, int(max_turns))
        self.context_frames = max(0, int(context_frames))
        self.success_dist_m = float(success_dist_m)
        self.system_prompt = _system_prompt(instruction, registry)
        self.trace: list[dict] = []
        self.finish_reason = "max_turns"
        self.finish_args: dict = {}
        self.parse_failures = 0
        self.turns = 0

    def _frames(self) -> list[torch.Tensor]:
        frames = list(self.ctx.extra.get("frame_cache", []))[-self.context_frames:]
        frames.append(self.world.render())
        return frames

    def _log_entry(self) -> dict:
        return self.ctx.log[-1] if self.ctx.log else {}

    def run(self) -> dict:
        last_result_text = "none yet"
        consecutive_failures = 0
        for turn in range(1, self.max_turns + 1):
            self.turns = turn
            frames = self._frames()
            budget_left = int(self.ctx.budget.get("tool_calls_left", 0))
            # the catalogue follows the episode: a turn after a grounding shows
            # the candidate tools first, a turn with a running leg the task tools
            self.system_prompt = _system_prompt(self.instruction, self.registry)
            reply = self.vlm.decide(
                frames,
                self.system_prompt,
                _user_prompt(turn, self.max_turns, budget_left, len(frames), last_result_text),
                dict(self.ctx.extra.get("vlm_context", {})),
            )
            parsed = parse_tool_call(reply)
            if parsed is None:
                self.parse_failures += 1
                consecutive_failures += 1
                self.trace.append(
                    {
                        "turn": turn,
                        "tool": None,
                        "args": {},
                        "status": "parse_failed",
                        "data": {"reason": "not_one_json_tool_call"},
                        "evidence": [],
                        "result_prompt": "",
                        "model_reply": reply,
                        "ms": 0.0,
                        "parse_failure": True,
                        "pose": _pose_fields(self.world),
                    }
                )
                if consecutive_failures >= 2:
                    self.finish_reason = "no_tool_call"
                    break
                last_result_text = (
                    "error: your reply was not one JSON tool call. " + finish_contract_doc()
                )
                continue
            consecutive_failures = 0
            name, raw_args = parsed
            args = _resolve_refs(raw_args, self.ctx.extra.get("last_result_data") or {})

            if name == "finish":
                # the registered finish tool (when the registry has one) owns the
                # outcome: it judges the claim on the episode's evidence pack, so
                # the fallback loop cannot report a success the runtime would not
                result = None
                if "finish" in self.registry.names():
                    result = self.registry.call("finish", args, self.ctx)
                self.finish_reason = "finished"
                self.finish_args = dict(args)
                self.trace.append(
                    {
                        "turn": turn,
                        "tool": "finish",
                        "args": args,
                        "status": result.status if result is not None else STATUS_OK,
                        "data": (dict(result.data) if result is not None
                                 else {"reason": args.get("reason", "")}),
                        "evidence": ([e.to_dict() for e in result.evidence]
                                     if result is not None else []),
                        "result_prompt": result.to_prompt() if result is not None else "",
                        "model_reply": reply,
                        "ms": 0.0,
                        "parse_failure": False,
                        "pose": _pose_fields(self.world),
                    }
                )
                print(f"[vlm_nav_tools] turn {turn}: finish ({args.get('reason', '')})",
                      flush=True)
                break
            if budget_left <= 0:
                self.finish_reason = "budget_exhausted"
                break

            try:
                spec = self.registry.get(name)
            except KeyError:
                result = ToolResult(STATUS_FAILED, {"error": f"unknown_tool: {name}"})
                spec = None
            else:
                result = self.registry.call(name, args, self.ctx)

            if spec is not None and spec.side_effects:
                refresh = getattr(self.world, "refresh_sensing", None)
                if callable(refresh):
                    try:
                        refresh()
                    except Exception:
                        pass

            log_entry = self._log_entry()
            self.trace.append(
                {
                    "turn": turn,
                    "tool": name,
                    "args": args,
                    "status": result.status,
                    "data": result.data,
                    "evidence": [e.to_dict() for e in result.evidence],
                    "result_prompt": result.to_prompt(),
                    "model_reply": reply,
                    "ms": float(log_entry.get("ms", 0.0)),
                    "parse_failure": False,
                    "pose": _pose_fields(self.world),
                }
            )
            self.ctx.extra["last_result_data"] = dict(result.data)
            last_result_text = result.to_prompt()
            print(f"[vlm_nav_tools] turn {turn}: {name} -> {result.status} "
                  f"({log_entry.get('ms', 0.0)} ms)", flush=True)

        return {
            "finish_reason": self.finish_reason,
            "finish_args": self.finish_args,
            "turns": self.turns,
            "parse_failures": self.parse_failures,
            "trace": self.trace,
        }


# ---------------------------------------------------------------------------
# episode orchestration
# ---------------------------------------------------------------------------


def make_tool_context(world: Any, memory: Any, args: argparse.Namespace,
                      out_dir: Path, episode_id: str,
                      extra: dict | None = None) -> ToolContext:
    """Episode context; ``extra`` (when given) overrides/adds context knobs.

    ``done_dist`` is the gate the episode is judged with (``success_dist_m`` +
    the target radius, the value the summary reports and ``verify_goal``
    checks), handed to the tools so the in-loop completion check and the
    evaluation gate are the same number.
    """
    knobs = {
        "out_dir": str(out_dir),
        "instruction": args.instruction,
        "scan_views": int(args.scan_views),
        "scan_pitch_deg": float(args.scan_pitch_deg),
        "success_dist_m": float(args.success_dist),
        "done_dist": success_gate_m(world, args),
        "viewpoints": {},
        "observation_index": {},
        "frame_cache": [],
        "viewpoint_seq": 0,
        "map_seq": 0,
        "nav_seq": 0,
    }
    knobs.update(dict(extra or {}))
    return ToolContext(
        world=world,
        memory=memory,
        episode_id=episode_id,
        observation_seq=0,
        budget={"tool_calls_left": max(0, int(args.tool_budget))},
        extra=knobs,
    )


def _coerce_trace(result: dict, runtime: Any) -> list[dict]:
    """Trace entries for the summary, from either runtime implementation.

    The module runtime returns ``EpisodeLog.turns`` (``TurnRecord.to_dict()``),
    the fallback returns its own ``trace`` list; both are normalized to the
    schema documented in the module docstring.
    """
    for source in (
        result.get("trace"),
        result.get("tool_trace"),
        result.get("turns"),
        result.get("records"),
        getattr(runtime, "trace", None),
        getattr(runtime, "records", None),
    ):
        if isinstance(source, (list, tuple)) and source:
            entries = [_normalize_trace_entry(_as_mapping(item)) for item in source]
            entries = [entry for entry in entries if entry]
            if entries:
                return entries
    return []


def _normalize_trace_entry(entry: dict) -> dict:
    if not entry:
        return {}
    entry = dict(entry)
    if "model_reply" not in entry and "raw_reply" in entry:
        entry["model_reply"] = entry.get("raw_reply", "")
    tool = entry.get("tool")
    if tool in ("", "(no tool call)") or (
        entry.get("parse_failure") is None and entry.get("detail") == "unparseable_model_output"
    ):
        entry["tool"] = None
        entry["parse_failure"] = True
    entry.setdefault("parse_failure", False)
    entry.setdefault("evidence", [])
    entry.setdefault("data", {})
    entry.setdefault("args", {})
    entry.setdefault("ms", 0.0)
    entry.setdefault("turn", 0)
    return entry


def _call_run(runtime: Any, instruction: str) -> Any:
    """``runtime.run(instruction)`` when supported, else ``runtime.run()``."""
    run = getattr(runtime, "run", None)
    if not callable(run):
        raise TypeError(f"{type(runtime).__name__} has no run()")
    try:
        signature = inspect.signature(run)
    except (TypeError, ValueError):
        return run(instruction)
    accepts_instruction = any(
        parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for parameter in signature.parameters.values()
    )
    if "instruction" in signature.parameters or accepts_instruction:
        return run(instruction)
    return run()


def run_episode(world: Any, vlm: Any, args: argparse.Namespace, out_dir: Path,
                episode_id: str = "ep_0",
                registry_factory: Any = None,
                ctx_extra: dict | None = None,
                prompt_episode_id: str | None = None) -> dict:
    """Build the stack, run the tool loop and write tool_trace.jsonl + summary.json.

    ``registry_factory(modules, vlm)`` may build the registry instead of
    ``build_registry`` (an experiment can wrap it, e.g. to resolve recalled
    evidence into context images); it returns a ``ToolRegistry`` or a
    ``(registry, backend, tools_source)`` tuple. ``ctx_extra`` adds/overrides
    ``ToolContext.extra`` knobs (per-experiment thresholds and switches).

    ``episode_id`` is the run's *label*: it names the artifacts (the summary,
    the trace, the memory archive) and may well carry the arm or mode name.
    ``prompt_episode_id`` is the identity the model sees in every prompt - it
    defaults to the neutral ``ep_<seed>`` of :func:`neutral_episode_id`, never
    to the label, so two arms of one cell cannot differ by the string in their
    prompts (see E2's ``dump_first_prompt`` finding).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "tool_trace.jsonl"
    toolset = str(getattr(args, "toolset", DEFAULT_TOOLSET) or DEFAULT_TOOLSET)
    # the mechanism switches: what the registry/runtime are told below, and what
    # summary.json reports back as "requested" vs. really "applied"
    requested = switch_report(args)
    exclude = () if requested["search_skill"] == "on" else (SEARCH_SKILL_TOOL,)
    label = str(episode_id)
    prompt_id = str(prompt_episode_id) if prompt_episode_id else neutral_episode_id(
        getattr(args, "seed", 0), getattr(args, "episode_index", None)
    )
    threshold = success_gate_m(world, args)

    modules, module_errors = load_m1_modules()
    memory, memory_backend, memory_detail = build_memory(
        modules, world, out_dir, prompt_id,
        spatial_memory=switch_enabled(requested["spatial_memory"]),
        episodic_memory=switch_enabled(requested["episodic_memory"]),
        identity_memory=switch_enabled(requested["identity_memory"]),
    )
    seed_memory_identities(memory, ctx_extra)
    # optional independent detection backend: off is the VLM-only path; a
    # backend that cannot load is reported and the episode keeps running
    detector_spec = str(getattr(args, "detector", "off") or "off")
    if detector_spec.strip().lower() not in ("", "off", "none", "null", "0", "false"):
        try:
            detector = resolve_detector(
                detector_spec,
                device=(
                    args.device
                    if str(getattr(args, "device", "cpu")).startswith("cuda")
                    else "cuda:0"
                ),
            )
            ctx_extra = dict(ctx_extra or {})
            ctx_extra["perception.detector"] = detector
            print(f"[vlm_nav_tools] detector: {detector_spec} "
                  f"(available={bool(detector.available())})", flush=True)
        except Exception as exc:  # noqa: BLE001 - an optional backend must not stop the run
            print(f"[vlm_nav_tools] detector {detector_spec!r} failed to build: "
                  f"{type(exc).__name__}: {exc}", flush=True)
    if registry_factory is not None:
        built = registry_factory(modules, vlm)
        if isinstance(built, tuple):
            registry, registry_backend, tools_source = built
        else:
            registry = built
            registry_backend = "custom"
            tools_source = {name: "custom" for name in registry.names()}
    else:
        registry, registry_backend, tools_source = build_registry(
            modules, vlm=vlm, toolset=toolset, exclude=exclude
        )
    ctx = make_tool_context(world, memory, args, out_dir, prompt_id, extra=ctx_extra)
    # the catalogue both runtimes render is state-aware: the registry reads the
    # live context, and the runtime is budgeted so the listing arrives whole
    registry.attach_state(lambda: ctx.extra, limit=CATALOG_CHARS)
    mission_host = None
    if str(getattr(args, "mission_plugin", "off") or "off").lower() == "on":
        from vlm_nav.plugins.bootstrap import install_mission_stack  # noqa: PLC0415

        try:
            mission_host = install_mission_stack(
                registry,
                out_dir / "mission_plugins",
                artifacts_dir=out_dir / "mission_artifacts",
            )
            print(
                "[vlm_nav_tools] mission plugins: "
                + ", ".join(
                    str(item.get("id")) for item in mission_host.plugins()
                ),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - the base tool loop must still run
            print(
                f"[vlm_nav_tools] mission plugin install failed, continuing "
                f"without it: {type(exc).__name__}: {exc}",
                flush=True,
            )
            mission_host = None
    state_before = tool_state(ctx.extra)
    catalog_before = registry.catalog()
    print(f"[vlm_nav_tools] catalogue: state={state_before.name} "
          f"full={len(catalog_before.full)} compact={len(catalog_before.compact)} "
          f"named={len(catalog_before.named)} chars={len(catalog_before.text)} "
          f"(limit={CATALOG_CHARS})", flush=True)
    bind_index = getattr(vlm, "bind_result_index", None)
    if callable(bind_index):  # scripted mock: consume $placeholders from results
        bind_index(registry.result_index)
    target = getattr(world, "target", None)
    target_name = getattr(target, "name", None) if target is not None else None
    if target_name and hasattr(vlm, "fallback_fields"):
        # $target_name placeholders (e.g. query_memory) resolve to the
        # instructed target even before any tool result carries it
        vlm.fallback_fields.setdefault("target_name", target_name)
    target_pos = getattr(target, "position_flu", None) if target is not None else None
    if target_pos is not None and hasattr(vlm, "fallback_fields"):
        # the mock's oracle answer to a grounding call (see ScriptedToolVLM):
        # where the harness put the target, not something the tools handed back
        vlm.fallback_fields.setdefault(
            "target_pos_flu", [float(v) for v in np.asarray(target_pos).reshape(-1)[:3]]
        )
    print(f"[vlm_nav_tools] tools: {', '.join(registry.names())} "
          f"(registry={registry_backend}, memory={memory_backend})", flush=True)

    runtime, runtime_backend, runtime_error = build_runtime(
        modules,
        world=world,
        vlm=vlm,
        registry=registry,
        memory=memory,
        instruction=args.instruction,
        out_dir=out_dir,
        trace_path=trace_path,
        episode_id=prompt_id,
        budget=int(args.tool_budget),
        max_turns=int(args.max_turns),
        tool_budget=int(args.tool_budget),
        context_frames=int(args.context_frames),
        success_dist_m=float(args.success_dist),
        recovery_enabled=requested["recovery"] == "on",
        state_block=requested["state_block"] == "on",
        executor_auto_steps=switch_enabled(
            getattr(args, "executor_auto_steps", DEFAULT_SWITCH)
        ),
        plan_mode=str(getattr(args, "plan_mode", "given") or "given"),
        tool_context=ctx,
        tools_source=tools_source,
        # upper bound of the runtime's system prompt; the runtime itself budgets
        # catalogue-first (doc first to be truncated), so this only keeps the
        # finish contract + policy template + the whole catalogue inside one cap
        max_system_chars=SYSTEM_CHARS,
        # the round's finish contract rides the tool doc of the runtime (the
        # runtime appends it to its system prompt)
        tool_call_doc=finish_contract_doc(),
    )
    if runtime is None:
        if runtime_error:
            print(f"[vlm_nav_tools] module runtime unavailable ({runtime_error}); "
                  "using the fallback loop", flush=True)
        runtime = _ToolRuntimeFallback(
            world=world,
            vlm=vlm,
            registry=registry,
            ctx=ctx,
            instruction=args.instruction,
            max_turns=int(args.max_turns),
            context_frames=int(args.context_frames),
            success_dist_m=float(args.success_dist),
        )
        runtime_backend = "fallback"

    print(f"[vlm_nav_tools] switches: recovery={requested['recovery']} "
          f"state_block={requested['state_block']} "
          f"search_skill={requested['search_skill']} "
          f"spatial_memory={requested['spatial_memory']} "
          f"episodic_memory={requested['episodic_memory']} "
          f"(runtime={runtime_backend})", flush=True)
    started = time.time()
    result = _as_mapping(_call_run(runtime, args.instruction))
    wall_time = time.time() - started

    entries = _coerce_trace(result, runtime)
    if not entries:
        entries = _read_trace(trace_path)
    if entries:
        _write_trace(trace_path, entries)
    elif not trace_path.exists():
        trace_path.write_text("")

    stats = summarize_trace(entries)
    final_distance = _world_distance(world)
    # both runtimes report their own reason: runtime.REASON_FINISH is "finish"
    finish_reason = str(result.get("finish_reason") or "")
    if not finish_reason:
        finish_entry = next(
            (e for e in reversed(entries) if e.get("tool") == "finish"), None
        )
        if finish_entry is not None:
            finish_reason = "finished"
        elif int(ctx.budget.get("tool_calls_left", 0)) <= 0:
            finish_reason = "budget_exhausted"
        else:
            finish_reason = "max_turns"
    finish_args = result.get("finish_args") or next(
        (e.get("args") or {} for e in reversed(entries) if e.get("tool") == "finish"),
        {},
    )
    computed_success = bool(
        finish_reason in ("finished", "finish", "vlm_finish", "target_verified")
        and final_distance is not None
        and final_distance <= threshold
    )
    runtime_success = result.get("success")
    # the completion claim and its evidence chain: what the finish declared and
    # what the episode's own trace backs (P4 - a success must be traceable, not
    # just a sentence). The chain is judged on the tool results of this episode,
    # so the module runtime and the fallback loop are held to one rule.
    records = evidence_records_from_trace(entries)
    completion = completion_evidence(records)
    declared = finish_args.get("success") if isinstance(finish_args, dict) else None
    if declared is None:  # a verifying finish tool answers in the result data
        declared = next(
            (entry.get("data", {}).get("claimed") for entry in reversed(entries)
             if entry.get("tool") == "finish"), None
        )
    completion_block = {
        "claimed": bool(declared) if declared is not None else False,
        "declared": declared if isinstance(declared, bool) else None,
        "verified": bool(completion["verified"]),
        "kind": completion["kind"],
        "evidence_trail": completion["evidence_trail"],
        "missing": completion["missing"],
        "candidate_ids": completion["candidate_ids"],
        "arrivals": completion["arrivals"],
        "verifications": completion["verifications"],
        "unverified_claim": bool(declared is True and not completion["verified"]),
        "honest_finish": bool(finish_reason in ("finished", "finish", "vlm_finish",
                                                "target_verified")
                              and declared is not True),
    }
    # ground truth (distance + terminating finish) is the eval criterion; an
    # explicit runtime verdict may veto (failed verification) but not inflate;
    # and since P4 a success also needs the evidence chain - a completion the
    # trace cannot back is not a success, however close the drone ended up
    success = bool(computed_success) and runtime_success is not False and bool(
        completion["verified"]
    )

    memory_stats = None
    if memory is not None:
        try:
            memory_stats = _MemoryAdapter(memory).stats()
        except Exception:
            memory_stats = None

    # the module runtime owns its own ToolContext/budget: prefer its numbers
    runtime_budget = result.get("budget") if isinstance(result.get("budget"), dict) else {}
    budget_left = int(runtime_budget.get("tool_calls_left",
                                         ctx.budget.get("tool_calls_left", 0)))
    turns_value = result.get("turns")
    if isinstance(turns_value, (list, tuple)):  # EpisodeLog.turns (auto steps included)
        tool_turns = len(turns_value)
    elif isinstance(turns_value, int):
        tool_turns = int(turns_value)
    else:
        tool_turns = stats["turns"]
    auto_steps = [
        dict(entry) for entry in (result.get("executor_auto_steps") or [])
        if isinstance(entry, dict)
    ]
    model_turns = result.get("model_turns")
    if not isinstance(model_turns, int):
        model_turns = max(0, tool_turns - len(auto_steps))
    turns = int(model_turns)
    # the catalogue the model read (state-aware), and the budget it was rendered
    # with: the report can prove the listing arrived whole (goal.md P4). Since
    # P4-B the runtime reports its own rendering (catalogue-first budget), so
    # the summary says what was really sent instead of repeating a constant.
    catalog_now = registry.catalog()
    runtime_prompt = (
        result.get("prompt_budget") if isinstance(result.get("prompt_budget"), dict) else {}
    )
    prompt_budget = {
        "system_chars": int(runtime_prompt.get("system_chars", SYSTEM_CHARS)),
        "catalog_chars": int(CATALOG_CHARS),
        "state": catalog_now.state.name,
        "state_hint": catalog_now.state.hint,
        "catalog": catalog_now.sections(),
        "catalog_order": list(catalog_now.order),
        "catalog_rendered_chars": len(catalog_now.text),
        # the runtime's own numbers (the registry render the model received)
        "catalogue_chars": int(
            runtime_prompt.get("catalogue_chars", len(catalog_now.text))
        ),
        "catalogue_raw_chars": int(
            runtime_prompt.get("catalogue_raw_chars", len(catalog_now.text))
        ),
        "catalogue_complete": bool(
            runtime_prompt.get("catalogue_complete", True)
        ),
        "catalogue_truncated": bool(
            runtime_prompt.get("catalogue_truncated", False)
        ),
        "catalogue_in_system": runtime_prompt.get("catalogue_in_system"),
        "doc_chars": int(runtime_prompt.get("doc_chars", 0)),
        "doc_full_chars": int(runtime_prompt.get("doc_full_chars", 0)),
        "doc_truncated": bool(runtime_prompt.get("doc_truncated", False)),
        "template_chars": int(runtime_prompt.get("template_chars", 0)),
        "rendered_chars": int(runtime_prompt.get("rendered_chars", 0)),
        "source": "runtime" if runtime_prompt else "runner",
        # the digest lets a report prove two runs showed the same catalogue
        # without copying a 8k-character block into every summary.json
        "catalog_sha1": hashlib.sha1(catalog_now.text.encode("utf-8")).hexdigest()[:16],
    }
    executor_block = {
        "requested": switch_token(
            getattr(args, "executor_auto_steps", DEFAULT_SWITCH)
        ),
        "switch": "on" if bool(getattr(runtime, "executor_auto_steps", False)) else "off",
        "limit": int(getattr(runtime, "max_executor_auto_steps", 0) or 0),
        "auto_steps": len(auto_steps),
        "saved_vlm_turns": len(auto_steps),
        "steps": auto_steps,
    }
    summary = {
        # the run's label (may name the arm); `prompt_episode_id` is what the
        # model saw - the neutral identity, never the label
        "episode_id": label,
        "prompt_episode_id": str(result.get("episode_id") or prompt_id),
        "instruction": args.instruction,
        "target_name": _target_name(world),
        "success": success,
        "success_computed": bool(computed_success),
        "success_runtime": runtime_success if isinstance(runtime_success, bool) else None,
        # the evidence chain this verdict rests on (P4): a success the trace
        # cannot back is not counted, and an unbacked claim is reported as such
        "success_evidence": bool(completion["verified"]),
        "completion": completion_block,
        "finish_reason": finish_reason,
        "finish_args": finish_args,
        # model decisions vs executed calls: an executor auto step adds a tool
        # turn but not a model turn (P4-B)
        "turns": turns,
        "model_turns": turns,
        "tool_turns": tool_turns,
        "executor": executor_block,
        "executor_auto_steps": len(auto_steps),
        "vlm_turns_saved": len(auto_steps),
        "toolset": toolset,
        "toolset_detail": toolset_report(toolset, registry.names()),
        "plan_mode": str(getattr(args, "plan_mode", "given") or "given"),
        # the state-aware catalogue the model was shown, and the budget that
        # carries it (``catalog_chars`` is what ``tools_text`` may use)
        "prompt_budget": prompt_budget,
        "switches": {
            "requested": requested,
            "applied": applied_switches(runtime, registry, runtime_backend,
                                        memory=memory),
        },
        "tool_calls": stats["total_calls"],
        "parse_failures": int(result.get("parse_failures") or stats["parse_failures"]),
        "vlm_calls": result.get("vlm_calls"),
        "vlm_usage": vlm_usage_summary(
            getattr(getattr(runtime, "log", None), "vlm_log", [])
        ),
        "tool_budget": {
            "initial": int(args.tool_budget),
            "left": budget_left,
        },
        "final_distance_m": round(final_distance, 3) if final_distance is not None else None,
        # the gate the episode is judged with AND the one the tools were handed
        # (``done_dist``): equal by construction, reported side by side so a
        # report can prove the completion check and the evaluation agree
        "success_dist_m": round(threshold, 3),
        "done_dist_m": round(threshold, 3),
        "wall_time_s": round(float(result.get("wall_time_s") or wall_time), 3),
        "stack_backend": registry_backend,
        "stack_detail": {
            "registry": registry_backend,
            "memory": memory_backend,
            "memory_detail": memory_detail,
            "runtime": runtime_backend,
            "runtime_error": runtime_error,
            "module_import_errors": module_errors,
            "tools_source": tools_source,
        },
        "tools": stats["tools"],
        "routine_cost": stats["routine_cost"],
        "evidence_kinds": stats["evidence_kinds"],
        "new_evidence_total": stats["new_evidence_total"],
        "unique_evidence": stats["unique_evidence"],
        "memory": memory_stats,
        "artifacts": {
            "tool_trace": trace_path.name if trace_path.exists() else None,
            "frames": "frames",
        },
    }
    if mission_host is not None:
        summary["mission_plugins"] = {
            "active": [
                item["id"] for item in mission_host.plugins()
                if item.get("status") == "active"
            ],
            "tools": [
                name for name in registry.names()
                if name.startswith("mission_") or name == "plan_mission"
            ],
            "services": sorted(mission_host.services.catalog()),
        }
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(f"[vlm_nav_tools] episode done: success={summary['success']} "
          f"finish={finish_reason} tool_calls={summary['tool_calls']} "
          f"turns={summary['turns']} gate={summary['success_dist_m']} m "
          f"episode={summary['prompt_episode_id']}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_config.json", "w") as handle:
        json.dump(vars(args), handle, indent=2, default=str)

    try:
        world = build_world(args)
    except RuntimeError as exc:
        # infeasible SUPER cell (mirrors run_vlm_nav_3dgs.py): record a skip
        print(f"[vlm_nav_tools] SKIP: {exc}", flush=True)
        summary = {
            "instruction": args.instruction,
            "success": False,
            "skipped": True,
            "finish_reason": "infeasible_skip",
            "turns": 0,
            "tool_calls": 0,
            "toolset": str(getattr(args, "toolset", DEFAULT_TOOLSET) or DEFAULT_TOOLSET),
            "switches": {"requested": switch_report(args), "applied": None},
            "final_distance_m": None,
            "tools": {},
            "evidence_kinds": {},
            "stack_backend": "fallback",
        }
        with open(out_dir / "summary.json", "w") as handle:
            json.dump(summary, handle, indent=2)
        return summary

    try:
        vlm = build_vlm(args)
        # the CLI has no arm label: the neutral id is both the artifact label
        # and the prompt identity (a driver that wants a labelled artifact
        # passes its own episode_id to run_episode instead)
        summary = run_episode(world, vlm, args, out_dir,
                              episode_id=neutral_episode_id(args.seed))
    finally:
        close = getattr(world, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main()
