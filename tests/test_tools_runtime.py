"""Tests for the tool-calling agent runtime (vlm_nav/runtime.py, goal.md 2/5).

The runtime is exercised on CPU with a fake world, fake tools registered in the
real ``ToolRegistry`` and scripted mock VLM backends - no model, no renderer.
"""

import json
import math
import tempfile
import time
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from vlm_nav.actions import NavAction
from vlm_nav.runtime import (
    DEFAULT_SYSTEM_PROMPT,
    FINISH_TOOL,
    NO_TOOL,
    PLAN_CALL_HINT,
    RECOVERY_ADVICE,
    RECOVERY_FINAL,
    RECOVERY_HOOK,
    RECOVERY_STOP,
    STATE_LABELS,
    REASON_ABORTED,
    REASON_BUDGET_EXHAUSTED,
    REASON_FINISH,
    REASON_MAX_TURNS,
    REASON_NO_NEW_EVIDENCE,
    REASON_UNPARSEABLE,
    AgentRuntime,
    recovery_advice,
    run_episode,
)
from vlm_nav.tools import (
    PROGRESS_KINDS,
    PROMPT_EXAMPLES,
    PROGRESS_ACTION,
    PROGRESS_INFORMATION,
    PROGRESS_NONE,
    PROGRESS_TASK,
    REPEAT_HINT,
    STATUS_BLOCKED,
    STATUS_CACHED,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_OK,
    TOOL_CALL_DOC,
    EvidenceRef,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    schema_problems,
)
from vlm_nav.tool_impl_task import plan_state, register_task_tools
from vlm_nav.vlm import MockVLM


class FakeWorld:
    """Minimal world: renders a constant frame, records executed actions."""

    def __init__(self):
        self.position_flu = np.zeros(3)
        self.yaw_rad = 0.0
        self.trajectory = [tuple(float(v) for v in self.position_flu)]
        self.target = SimpleNamespace(
            name="red book", position_flu=(7.0, 8.0, 9.0), radius_m=0.2
        )
        self.actions = []
        self.render_calls = 0
        self.render_error = False

    def render(self):
        self.render_calls += 1
        if self.render_error:
            raise RuntimeError("sensor offline")
        return torch.full((3, 4, 4), 0.5)

    def apply_action(self, action):
        self.actions.append(action)
        self.position_flu = self.position_flu + np.asarray([1.0, 0.0, 0.0])
        self.trajectory.append(tuple(float(v) for v in self.position_flu))
        return "moved to (1.00, 0.00, 0.00)"

    def distance_to_target(self):
        return 42.0

    def target_in_fov(self):
        return False

    def scan_views(self, n_views=8, pitch_down_deg=0.0, spread_deg=None):
        return [self.render() for _ in range(n_views)]


class RecordingScriptedVLM(MockVLM):
    """MockVLM tool scripting plus prompt/frame recording.

    ``tool_script`` entries are ``{"tool": ..., "args": {...}}`` dicts (or raw
    strings for malformed-reply cases); the runtime's ``decide_with_tools``
    path is exercised, as with a real backend.
    """

    def __init__(self, tool_script=None, mode="tools"):
        super().__init__(mode=mode, tool_script=tool_script)
        self.seen = []

    def _record(self, frames, system_prompt, user_prompt, context, tools_text=""):
        self.seen.append(
            {
                "n_images": len(frames),
                "system": system_prompt,
                "tools_text": tools_text,
                "user": user_prompt,
                "context": dict(context or {}),
            }
        )

    def decide_with_tools(self, frames, system_prompt, user_prompt, tools_text="", context=None):
        self._record(frames, system_prompt, user_prompt, context, tools_text)
        return super().decide_with_tools(frames, system_prompt, user_prompt, tools_text, context)

    def decide(self, frames, system_prompt, user_prompt, context=None):
        self._record(frames, system_prompt, user_prompt, context)
        return super().decide(frames, system_prompt, user_prompt, context)


class FakeMemory:
    def __init__(self):
        self.queries = 0

    def context_summary(self):
        self.queries += 1
        return "hyp_7: red book candidate at 4.2 m, support=2"

    def active_tasks(self):
        return [{"task_id": "nav_3", "state": "running", "progress_m": 0.5}]


# ----------------------------------------------------------------------
# fake tools
# ----------------------------------------------------------------------


def _observe(ctx, args):
    obs = ctx.next_obs_id()
    ctx.extra.setdefault("images", {})[obs] = torch.full((3, 4, 4), 0.25)
    return ToolResult(
        STATUS_OK,
        {"observation_id": obs, "mode": args.get("mode", "lookaround")},
        [EvidenceRef("observation", obs, "current frame")],
    )


def _many_views(ctx, args):
    images = ctx.extra.setdefault("images", {})
    ids = []
    for _ in range(6):
        obs = ctx.next_obs_id()
        images[obs] = torch.full((3, 4, 4), 0.2)
        ids.append(obs)
    return ToolResult(
        STATUS_OK,
        {"observation_id": ids[0], "view_ids": ids},
        [EvidenceRef("observation", ids[0], "scan strip")],
    )


def _propose_viewpoints(ctx, args):
    return ToolResult(
        STATUS_OK,
        {"candidate_viewpoints": ["vp_12", "vp_15"], "coverage_gain_m2": 6.5},
        [EvidenceRef("map", "map_1", "observed map")],
    )


def _check_path(ctx, args):
    return ToolResult(
        STATUS_OK, {"reachable": True, "cost_m": 3.25, "unknown_ratio": 0.1}
    )


def _navigate(ctx, args):
    viewpoint = str(args.get("viewpoint_id", "vp_0"))
    feedback = ctx.world.apply_action(NavAction(kind="waypoint", forward_m=1.0))
    return ToolResult(
        STATUS_OK,
        {
            "task_id": "nav_17",
            "viewpoint_id": viewpoint,
            "progress_m": 2.4,
            "reason": "arrived_subgoal",
        },
        [EvidenceRef("trajectory", "trace_1", feedback)],
    )


def _long_note(ctx, args):
    return ToolResult(
        STATUS_OK, {"notes": "x" * 2000, "sector": "north", "clear": True}
    )


def _step_ok(ctx, args):
    """A tool result that satisfies the current sub-task (P4 executor test)."""
    obs = ctx.next_obs_id()
    ctx.extra.setdefault("images", {})[obs] = torch.full((3, 4, 4), 0.75)
    return ToolResult(
        STATUS_OK,
        {"subgoal_complete": True, "checked": args.get("where", "the corridor")},
        [EvidenceRef("observation", obs, "step evidence")],
    )


def _step_blocked(ctx, args):
    """A leg that went nowhere (the executor marks the sub-task blocked)."""
    return ToolResult(
        STATUS_BLOCKED,
        {
            "outcome": "planner_failed",
            "reason": "planner_failed",
            "blocked_reason": "planner_failed",
        },
        [EvidenceRef("trajectory", "trace_9", "blocked leg")],
    )


def _candidate(ctx, args):
    """A newly grounded candidate (a decision node, not a stall)."""
    from vlm_nav.observations import CANDIDATE_STORE_KEY

    store = ctx.extra.setdefault(CANDIDATE_STORE_KEY, {})
    cand_id = f"cand_{len(store) + 1}"
    store[cand_id] = {
        "id": cand_id,
        "source": "ground_target",
        "observation_id": ctx.extra.get("current_observation"),
        "confidence": 0.7,
    }
    return ToolResult(STATUS_OK, {"candidate_id": cand_id})


def _conflict(ctx, args):
    """Two candidates that disagree (a decision node, not a stall)."""
    return ToolResult(STATUS_OK, {"verdict": "uncertain", "conflicts": ["colour"]})


def _target_lost(ctx, args):
    """The tracked target left the view (the plan has to be revised)."""
    return ToolResult(STATUS_OK, {"reason": "target_lost", "note": "left the view"})


def _next_probe(ctx, args):
    """A routine answer that names an executable next step (auto-step source)."""
    return ToolResult(
        STATUS_OK, {"note": "routine", "next_action": {"tool": "counter", "args": {}}}
    )


def _candidate_next(ctx, args):
    """A new candidate *and* a next step: the decision node must still win."""
    result = _candidate(ctx, args)
    result.data["next_action"] = {"tool": "counter", "args": {}}
    return result


def _chain_tick(ctx, args):
    """A continuation that keeps naming itself (the auto-step cap test)."""
    step = int(ctx.extra.get("_chain_tick", 0)) + 1
    ctx.extra["_chain_tick"] = step
    return ToolResult(
        STATUS_OK,
        {
            "tick": step,
            "next_action": {"tool": "chain_tick", "args": dict(args)},
        },
    )


def _empty_probe(ctx, args):
    """A fruitless result (status empty) - routine, but the plan has a branch."""
    return ToolResult(
        STATUS_EMPTY, {"reason": "no_candidate", "observation_id": "obs_1"}
    )


def _loop_probe(ctx, args):
    """A fruitless result whose next_action points at the call that produced it."""
    return ToolResult(
        STATUS_EMPTY,
        {
            "reason": "no_candidate",
            "observation_id": "obs_1",
            "next_action": {"tool": "loop_probe", "args": dict(args)},
        },
    )


def _emit_events(ctx, args):
    ctx.extra["runtime"].emit_event("arrive", {"task_id": "nav_17", "at": "vp_12"})
    ctx.extra.setdefault("events", []).append(
        {"name": "obstacle", "data": {"task_id": "nav_17", "reason": "new_obstacle"}}
    )
    return ToolResult(STATUS_OK, {"queued": 2})


def _abort(ctx, args):
    ctx.extra["runtime"].request_abort("operator stop")
    return ToolResult(STATUS_OK, {"aborting": True})


def _boom(ctx, args):
    raise ValueError("boom")


def _probe(ctx, args):
    cache = ctx.extra.setdefault("_probe_cache", set())
    key = json.dumps(args, sort_keys=True)
    if key in cache:
        return ToolResult(STATUS_CACHED, {"checked": "sector_3", "reason": "repeat"})
    cache.add(key)
    return ToolResult(STATUS_OK, {"checked": "sector_3", "clear": True})


def _counter(ctx, args):
    step = int(ctx.extra.setdefault("_counter", 0)) + 1
    ctx.extra["_counter"] = step
    return ToolResult(STATUS_OK, {"checked": "sector_3", "probe": step})


def _finish_check(ctx, args):
    if args.get("verified"):
        return ToolResult(
            STATUS_OK, {"verified": True, "success": True, "reason": args.get("reason", "")}
        )
    return ToolResult(
        STATUS_BLOCKED,
        {"reason": "conditions_unmet", "missing": ["target_verified"]},
        [EvidenceRef("observation", "obs_99", "last frame")],
    )


def _spec(name, handler, description=None, side_effects=False, cacheable=True):
    return ToolSpec(
        name=name,
        description=description or f"{name} test double",
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=handler,
        side_effects=side_effects,
        cacheable=cacheable,
    )


def make_registry(*names, finish_tool=False):
    registry = ToolRegistry()
    handlers = {
        "observe": (_observe, False),
        "many_views": (_many_views, False),
        "propose_viewpoints": (_propose_viewpoints, False),
        "check_path": (_check_path, False),
        "navigate": (_navigate, True),
        "long_note": (_long_note, False),
        "emit_events": (_emit_events, False),
        "abort": (_abort, False),
        "boom": (_boom, False),
        "probe": (_probe, False),
        "counter": (_counter, False),
        "step_ok": (_step_ok, False),
        "step_blocked": (_step_blocked, False),
        "candidate": (_candidate, False),
        "candidate_next": (_candidate_next, False),
        "conflict": (_conflict, False),
        "target_lost": (_target_lost, False),
        "next_probe": (_next_probe, False),
        "chain_tick": (_chain_tick, False),
        "empty_probe": (_empty_probe, False),
        "loop_probe": (_loop_probe, False),
    }
    for name in names:
        handler, side_effects = handlers[name]
        # "counter" answers a *changing* payload for an identical call (the
        # fingerprint test relies on that), so it opts out of the call cache;
        # "chain_tick" must run every time its own continuation names it.
        registry.register(
            _spec(
                name,
                handler,
                side_effects=side_effects,
                cacheable=name not in ("counter", "chain_tick"),
            )
        )
    if finish_tool:
        registry.register(_spec("finish", _finish_check))
    return registry


def tool_call(name, **args):
    return {"tool": name, "args": args}


# ----------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------


class ToolLoopTest(unittest.TestCase):
    def test_call_chain_order_and_log(self):
        world = FakeWorld()
        registry = make_registry(
            "observe", "propose_viewpoints", "check_path", "navigate"
        )
        vlm = RecordingScriptedVLM(
            [
                tool_call("observe", mode="lookaround"),
                tool_call("propose_viewpoints"),
                tool_call("navigate", viewpoint_id="vp_12"),
                tool_call("finish", reason="target verified", success=True),
            ]
        )
        log = AgentRuntime(registry, world, FakeMemory(), vlm, budget=10, max_turns=6).run(
            "find the red book"
        )

        self.assertEqual(log.finish_reason, REASON_FINISH)
        self.assertTrue(log.success)
        self.assertEqual(
            log.tool_names(), ["observe", "propose_viewpoints", "navigate", "finish"]
        )
        self.assertEqual(log.tool_calls(), 4)
        self.assertEqual(log.vlm_calls, 4)
        self.assertEqual([t.turn for t in log.turns], [1, 2, 3, 4])
        self.assertTrue(all(t.status == STATUS_OK for t in log.turns))
        self.assertTrue(all(t.images_sent >= 1 for t in log.turns))
        self.assertEqual(len(world.actions), 1)
        self.assertEqual(world.actions[0].kind, "waypoint")
        # the observation tool minted its own id and it is logged as evidence
        self.assertEqual(log.turns[0].evidence[0]["kind"], "observation")
        self.assertTrue(log.turns[0].evidence[0]["ref"].startswith("obs_"))
        # citation-based usage: vp_12 comes from the viewpoint proposal
        self.assertTrue(log.turns[1].used)
        self.assertFalse(log.turns[0].used)
        self.assertFalse(log.turns[2].used)
        self.assertTrue(all(t.new_evidence for t in log.turns))
        # the tool catalog goes to the backend as rendered schema text, while the
        # runtime's own system prompt carries the policy and the JSON protocol
        self.assertIn("- navigate:", vlm.seen[0]["tools_text"])
        self.assertIn("You are the decision loop", vlm.seen[0]["system"])
        self.assertIn("Reply with exactly one JSON object", vlm.seen[0]["user"])
        # tool results are fed back as structured text, not prose
        self.assertIn("[ok] observation_id=obs_", vlm.seen[1]["user"])
        self.assertIn("candidate_viewpoints=['vp_12', 'vp_15']", vlm.seen[2]["user"])

    def test_decide_with_tools_is_preferred(self):
        class DictVLM:
            """5-argument runtime contract (vlm.py)."""

            def __init__(self, replies):
                self.replies = list(replies)
                self.seen = []

            def decide_with_tools(
                self, frames, system_prompt, user_prompt, tools_text="", context=None
            ):
                self.seen.append({"tools_text": tools_text, "context": dict(context or {})})
                return self.replies.pop(0)

            def decide(self, *args, **kwargs):  # pragma: no cover - must not run
                raise AssertionError("decide() must not be called")

        vlm = DictVLM(
            [
                {"tool": "observe", "args": {"mode": "lookahead"}},
                {"tool": "finish", "args": {"reason": "done"}},
            ]
        )
        log = AgentRuntime(make_registry("observe"), FakeWorld(), None, vlm).run("go")
        self.assertEqual(log.tool_names(), ["observe", "finish"])
        self.assertEqual(log.turns[0].args, {"mode": "lookahead"})
        self.assertEqual(log.finish_reason, REASON_FINISH)
        # schemas and history travel through tools_text/context
        self.assertIn("- observe:", vlm.seen[0]["tools_text"])
        self.assertEqual(vlm.seen[0]["context"]["tools"][0]["name"], "observe")
        self.assertEqual(vlm.seen[1]["context"]["history"][0]["tool_call"]["tool"], "observe")
        self.assertIn("[ok]", vlm.seen[1]["context"]["history"][0]["result"])
        self.assertEqual(vlm.seen[1]["context"]["step"], 2)

    def test_four_argument_decide_with_tools_still_works(self):
        class LegacyVLM:
            """Older 4-argument convention: (frames, system, user, context)."""

            def __init__(self):
                self.replies = [
                    {"tool": "observe", "args": {"mode": "forward"}},
                    {"tool": "finish", "args": {"reason": "ok"}},
                ]
                self.contexts = []

            def decide_with_tools(self, frames, system_prompt, user_prompt, context=None):
                self.contexts.append(dict(context or {}))
                return self.replies.pop(0)

        vlm = LegacyVLM()
        log = AgentRuntime(make_registry("observe"), FakeWorld(), None, vlm).run("go")
        self.assertEqual(log.tool_names(), ["observe", "finish"])
        self.assertEqual(vlm.contexts[0]["instruction"], "go")

    def test_plain_decide_fallback_gets_catalogue_in_system_prompt(self):
        class PlainVLM:
            """No decide_with_tools: the runtime must render the tools itself."""

            def __init__(self, replies):
                self.replies = list(replies)
                self.seen = []

            def decide(self, frames, system_prompt, user_prompt, context=None):
                self.seen.append({"system": system_prompt, "user": user_prompt})
                reply = self.replies.pop(0)
                return reply if isinstance(reply, str) else json.dumps(reply)

        vlm = PlainVLM([tool_call("observe"), tool_call("finish", reason="ok")])
        log = AgentRuntime(make_registry("observe", "navigate"), FakeWorld(), None, vlm).run("go")
        self.assertEqual(log.tool_names(), ["observe", "finish"])
        system = vlm.seen[0]["system"]
        self.assertIn("Available tools:", system)
        self.assertIn("- navigate:", system)
        self.assertIn(TOOL_CALL_DOC.splitlines()[0], system)

    def test_openai_style_tool_calls(self):
        class NativeVLM:
            def __init__(self):
                self.replies = [
                    {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "observe",
                                    "arguments": '{"mode": "forward"}',
                                },
                            }
                        ]
                    },
                    {"tool": "finish", "args": {"reason": "ok"}},
                ]

            def decide_with_tools(
                self, frames, system_prompt, user_prompt, tools_text="", context=None
            ):
                return self.replies.pop(0)

        log = AgentRuntime(make_registry("observe"), FakeWorld(), None, NativeVLM()).run("go")
        self.assertEqual(log.turns[0].tool, "observe")
        self.assertEqual(log.turns[0].args, {"mode": "forward"})

    def test_reprompt_after_unparseable_then_recovers(self):
        world = FakeWorld()
        vlm = RecordingScriptedVLM(
            [
                "I will look around first.",
                tool_call("observe", mode="lookaround"),
                tool_call("finish", reason="ok"),
            ]
        )
        log = AgentRuntime(make_registry("observe"), world, None, vlm).run("go")
        self.assertEqual(log.vlm_calls, 3)
        self.assertEqual(log.turns[0].tool, "observe")
        self.assertIn("no decodable tool call", vlm.seen[1]["user"])
        self.assertEqual(log.finish_reason, REASON_FINISH)

    def test_unparseable_output_terminates(self):
        vlm = RecordingScriptedVLM(["prose one", "prose two", "prose three"])
        log = AgentRuntime(make_registry("observe"), FakeWorld(), None, vlm).run("go")
        self.assertEqual(log.finish_reason, REASON_UNPARSEABLE)
        self.assertEqual(log.vlm_calls, 2)
        self.assertEqual(len(log.turns), 1)
        self.assertEqual(log.turns[0].tool, NO_TOOL)
        self.assertEqual(log.turns[0].status, STATUS_FAILED)
        self.assertEqual(log.turns[0].detail, "unparseable_model_output")

    def test_unknown_tool_is_reported_and_recoverable(self):
        world = FakeWorld()
        vlm = RecordingScriptedVLM(
            [
                tool_call("teleport", destination="vp_1"),
                tool_call("finish", reason="ok"),
            ]
        )
        log = AgentRuntime(make_registry("observe"), world, None, vlm).run("go")
        self.assertEqual(log.finish_reason, REASON_FINISH)
        self.assertEqual(log.turns[0].status, STATUS_FAILED)
        self.assertEqual(log.turns[0].data["reason"], "unknown_tool")
        self.assertIn("observe", log.turns[0].data["available_tools"])
        # the model sees the failure and the tool catalogue before turn 2
        self.assertIn("unknown_tool", vlm.seen[1]["user"])
        self.assertIn("available_tools", vlm.seen[1]["user"])


class TerminationTest(unittest.TestCase):
    def test_budget_exhausted_stops_before_next_model_call(self):
        world = FakeWorld()
        vlm = RecordingScriptedVLM([tool_call("observe")] * 5)
        log = AgentRuntime(make_registry("observe"), world, None, vlm, budget=2, max_turns=6).run(
            "go"
        )
        self.assertEqual(log.tool_calls(), 2)
        self.assertEqual(log.vlm_calls, 2)
        self.assertEqual(log.finish_reason, REASON_BUDGET_EXHAUSTED)
        self.assertEqual(log.budget["tool_calls_left"], 0)
        self.assertEqual(world.render_calls, 2)

    def test_max_turns(self):
        vlm = RecordingScriptedVLM([tool_call("observe")] * 5)
        log = AgentRuntime(
            make_registry("observe"), FakeWorld(), None, vlm, budget=20, max_turns=3
        ).run("go")
        self.assertEqual(log.finish_reason, REASON_MAX_TURNS)
        self.assertEqual(len(log.turns), 3)

    def test_stagnation_stops_loop(self):
        vlm = RecordingScriptedVLM([tool_call("probe", sector="3")] * 5)
        log = AgentRuntime(
            make_registry("probe"),
            FakeWorld(),
            None,
            vlm,
            budget=20,
            max_turns=6,
            max_stagnant_turns=1,
        ).run("go")
        self.assertEqual(log.finish_reason, REASON_NO_NEW_EVIDENCE)
        self.assertEqual(len(log.turns), 2)
        self.assertEqual(log.vlm_calls, 2)
        self.assertTrue(log.turns[0].new_evidence)
        self.assertFalse(log.turns[1].new_evidence)
        self.assertEqual(log.turns[1].status, STATUS_CACHED)

    def test_distinct_payloads_count_as_new_evidence(self):
        # fingerprint-based: a tool with no evidence refs still feeds the model
        # as long as the payload changes
        vlm = RecordingScriptedVLM([tool_call("counter")] * 3)
        log = AgentRuntime(
            make_registry("counter"),
            FakeWorld(),
            None,
            vlm,
            budget=20,
            max_turns=3,
            max_stagnant_turns=1,
        ).run("go")
        self.assertEqual(log.finish_reason, REASON_MAX_TURNS)
        self.assertEqual(len(log.turns), 3)
        self.assertTrue(all(t.new_evidence for t in log.turns))

    def test_abort_request_from_tool_ends_episode(self):
        runtime = AgentRuntime(
            make_registry("abort", "observe"), FakeWorld(), None,
            RecordingScriptedVLM([tool_call("abort")] * 4),
        )
        log = runtime.run("go")
        self.assertEqual(log.finish_reason, REASON_ABORTED)
        self.assertEqual(len(log.turns), 1)
        self.assertEqual(runtime.abort_reason, "operator stop")

    def test_tool_exception_becomes_failed_result(self):
        vlm = RecordingScriptedVLM([tool_call("boom"), tool_call("finish", reason="ok")])
        log = AgentRuntime(
            make_registry("boom"), FakeWorld(), None, vlm, max_stagnant_turns=3
        ).run("go")
        self.assertEqual(log.turns[0].status, STATUS_FAILED)
        self.assertIn("ValueError", log.turns[0].data["error"])
        self.assertEqual(log.finish_reason, REASON_FINISH)

    def test_registered_finish_tool_can_veto(self):
        vlm = RecordingScriptedVLM(
            [
                tool_call("finish", reason="arrived"),
                tool_call("finish", reason="arrived", verified=True),
            ]
        )
        log = AgentRuntime(
            make_registry("observe", finish_tool=True), FakeWorld(), None, vlm
        ).run("go")
        self.assertEqual(len(log.turns), 2)
        self.assertEqual(log.turns[0].status, STATUS_BLOCKED)
        self.assertEqual(log.turns[1].status, STATUS_OK)
        self.assertEqual(log.finish_reason, REASON_FINISH)
        self.assertTrue(log.success)
        self.assertIn("conditions_unmet", vlm.seen[1]["user"])

    def test_finish_success_flag_is_honoured(self):
        vlm = RecordingScriptedVLM(
            [tool_call("finish", reason="give up", success=False)]
        )
        log = AgentRuntime(make_registry("observe"), FakeWorld(), None, vlm).run("go")
        self.assertEqual(log.finish_reason, REASON_FINISH)
        self.assertFalse(log.success)
        self.assertEqual(log.finish_args["reason"], "give up")

    def test_bare_finish_without_claim_is_not_success(self):
        vlm = RecordingScriptedVLM([tool_call("finish", reason="done looking")])
        log = AgentRuntime(make_registry("observe"), FakeWorld(), None, vlm).run("go")
        self.assertEqual(log.finish_reason, REASON_FINISH)
        self.assertIsNone(log.success)

    def test_the_protocol_tells_the_model_to_verify_before_claiming_success(self):
        """The hallucinated finish (goal.md section 7) is a prompt-contract bug:
        the tool-call doc every tool-aware backend appends has to name
        verify_goal as the gate of a `success` claim, and the recalled obs ids
        as navigable targets."""
        self.assertIn("verify_goal", TOOL_CALL_DOC)
        self.assertIn('"success": true', TOOL_CALL_DOC)
        self.assertIn("obs_N", TOOL_CALL_DOC)
        self.assertIn('"target_id": "obs_N"', TOOL_CALL_DOC)

    def test_the_protocol_states_the_typed_id_contract(self):
        """The three id types mean three different things; a model that mixes
        them gets a typed error, and the doc has to say so - including the names
        of the reserved navigation tools of the next wave."""
        for id_type, prefix in (
            ("observation_id", "obs_N"),
            ("candidate_id", "cand_N"),
            ("viewpoint_id", "vp_N"),
        ):
            with self.subTest(id_type=id_type):
                self.assertIn(id_type, TOOL_CALL_DOC)
                self.assertIn(prefix, TOOL_CALL_DOC)
        for tool in ("approach_candidate", "go_to_viewpoint", "return_to_observation"):
            with self.subTest(tool=tool):
                self.assertIn(tool, TOOL_CALL_DOC)
        # the two "go somewhere" tools must not be confusable: returning to the
        # camera's place is not approaching the object in the frame
        self.assertIn("return_to_observation", TOOL_CALL_DOC.split("Reserved navigation tools")[1])
        self.assertIn("[x1, y1, x2, y2]", TOOL_CALL_DOC)


class ContextBudgetTest(unittest.TestCase):
    def test_text_and_image_budget(self):
        world = FakeWorld()
        registry = make_registry("many_views")
        vlm = RecordingScriptedVLM([tool_call("many_views"), tool_call("finish", reason="ok")])
        AgentRuntime(
            registry,
            world,
            None,
            vlm,
            max_images=3,
            max_history_images=2,
            max_text_chars=300,
        ).run("find " + "the red book " * 60)

        self.assertLessEqual(len(vlm.seen[0]["user"]), 300)
        self.assertLessEqual(len(vlm.seen[1]["user"]), 300)
        self.assertLessEqual(vlm.seen[0]["n_images"], 3)
        self.assertLessEqual(vlm.seen[1]["n_images"], 3)
        self.assertIn("truncated", vlm.seen[0]["user"])
        self.assertIn("image 1 = current view", vlm.seen[0]["user"])
        # image 1 of turn 2 is the fresh frame, images 2+ come from the tool
        self.assertIn("image 1 = current view", vlm.seen[1]["user"])
        self.assertIn("earlier observations", vlm.seen[1]["user"])

    def test_tool_result_truncation(self):
        vlm = RecordingScriptedVLM([tool_call("long_note"), tool_call("finish", reason="ok")])
        AgentRuntime(
            make_registry("long_note"),
            FakeWorld(),
            None,
            vlm,
            max_tool_chars=80,
        ).run("go")
        self.assertIn("[truncated]", vlm.seen[1]["user"])
        feedback = [line for line in vlm.seen[1]["user"].splitlines() if "long_note(" in line]
        self.assertEqual(len(feedback), 1)
        self.assertLessEqual(len(feedback[0]), 80 + 40)

    def test_system_prompt_holds_protocol_and_catalogue(self):
        # tool-aware path: policy in the system prompt, schemas via tools_text
        # (the backend appends TOOL_CALL_DOC itself, so the runtime must not)
        vlm = RecordingScriptedVLM(tool_script=[tool_call("finish", reason="ok")])
        AgentRuntime(
            make_registry("observe", "navigate"), FakeWorld(), None, vlm
        ).run("go")
        self.assertIn("You are the decision loop", vlm.seen[0]["system"])
        self.assertNotIn("Available tools:", vlm.seen[0]["system"])
        self.assertIn("- observe:", vlm.seen[0]["tools_text"])
        self.assertIn("- navigate:", vlm.seen[0]["tools_text"])

        # system prompt cap still holds
        capped = RecordingScriptedVLM(tool_script=[tool_call("finish", reason="ok")])
        AgentRuntime(
            make_registry("observe", "navigate"),
            FakeWorld(),
            None,
            capped,
            max_system_chars=200,
        ).run("go")
        self.assertLessEqual(len(capped.seen[0]["system"]), 200)
        self.assertIn("You are the decision loop", capped.seen[0]["system"])
        self.assertLessEqual(len(capped.seen[0]["tools_text"]), 200)

    def test_render_failure_is_survivable(self):
        world = FakeWorld()
        world.render_error = True
        vlm = RecordingScriptedVLM([tool_call("finish", reason="ok")])
        log = AgentRuntime(make_registry("observe"), world, None, vlm).run("go")
        self.assertEqual(log.finish_reason, REASON_FINISH)
        self.assertEqual(log.turns[0].images_sent, 0)
        self.assertIn("none available", vlm.seen[0]["user"])


class MemoryAndEventTest(unittest.TestCase):
    def test_memory_summary_enters_prompt(self):
        memory = FakeMemory()
        vlm = RecordingScriptedVLM([tool_call("finish", reason="ok")])
        AgentRuntime(make_registry("observe"), FakeWorld(), memory, vlm).run("go")
        self.assertIn("hyp_7", vlm.seen[0]["user"])
        self.assertIn("active_task", vlm.seen[0]["user"])
        self.assertIn("nav_3", vlm.seen[0]["user"])
        self.assertGreaterEqual(memory.queries, 1)

    def test_memory_api_is_optional(self):
        vlm = RecordingScriptedVLM([tool_call("finish", reason="ok")])
        log = AgentRuntime(
            make_registry("observe"), FakeWorld(), object(), vlm
        ).run("go")
        self.assertEqual(log.finish_reason, REASON_FINISH)

    def test_events_reach_hooks_and_next_prompt(self):
        runtime = AgentRuntime(
            make_registry("emit_events"), FakeWorld(), None,
            RecordingScriptedVLM([tool_call("emit_events"), tool_call("finish", reason="ok")]),
        )
        seen = []
        runtime.on_event(lambda name, data: seen.append((name, data)))
        log = runtime.run("go")

        names = [e["name"] for e in log.events]
        self.assertIn("arrive", names)
        self.assertIn("obstacle", names)
        self.assertIn("nav_17", json.dumps(log.events))
        self.assertIn("arrive", [name for name, _ in seen])
        # the drained events are in the prompt of the following turn
        self.assertIn("Events since last turn", runtime.vlm.seen[1]["user"])
        self.assertIn("arrive", runtime.vlm.seen[1]["user"])

    def test_external_event_wakes_next_turn(self):
        runtime = AgentRuntime(
            make_registry("emit_events", "observe"), FakeWorld(), None,
            RecordingScriptedVLM([tool_call("emit_events"), tool_call("observe")] * 2),
        )

        def hook(name, data):
            if name == "arrive":
                runtime.request_abort("event-driven stop")

        runtime.on_event(hook)
        log = runtime.run("go")
        self.assertEqual(log.finish_reason, REASON_ABORTED)
        self.assertEqual(runtime.abort_reason, "event-driven stop")
        # the event that woke the loop is still visible in the log
        self.assertIn("arrive", [e["name"] for e in log.events])
        self.assertEqual(len(log.turns), 1)

    def test_wait_for_event_returns_queued_event(self):
        runtime = AgentRuntime(make_registry("observe"), FakeWorld(), None, MockVLM("script"))
        runtime.emit_event("arrive", {"task_id": "nav_1"})
        event = runtime.wait_for_event(timeout=0.01)
        self.assertEqual(event["name"], "arrive")
        self.assertIsNone(runtime.wait_for_event(timeout=0.01))


class LogTest(unittest.TestCase):
    def _run(self):
        vlm = RecordingScriptedVLM(
            [
                tool_call("observe", mode="lookaround"),
                tool_call("propose_viewpoints"),
                tool_call("navigate", viewpoint_id="vp_12"),
                tool_call("finish", reason="done", success=True),
            ]
        )
        return AgentRuntime(
            make_registry("observe", "propose_viewpoints", "navigate"),
            FakeWorld(),
            None,
            vlm,
            budget={"tool_calls_left": 8},
            episode_id="ep_test",
        ).run("find the red book")

    def test_turn_record_keys(self):
        log = self._run()
        required = {
            "turn", "tool", "args", "status", "ms", "evidence",
            "used", "new_evidence", "images_sent",
        }
        self.assertEqual(log.episode_id, "ep_test")
        for record in log.to_dict()["turns"]:
            self.assertTrue(required.issubset(record.keys()))
        self.assertLessEqual(log.budget["tool_calls_left"], 8)

    def test_save_writes_json(self):
        log = self._run()
        with tempfile.TemporaryDirectory() as tmp:
            path = log.save(f"{tmp}/nested/episode_log.json")
            with open(path) as handle:
                data = json.load(handle)
        self.assertEqual(data["episode_id"], "ep_test")
        self.assertEqual(len(data["turns"]), 4)
        self.assertTrue(data["success"])
        self.assertIn("observe", log.summary_text())
        self.assertIn("navigate", log.summary_text())

    def test_run_episode_wrapper(self):
        vlm = RecordingScriptedVLM([tool_call("finish", reason="done")])
        log = run_episode(
            make_registry("observe"), FakeWorld(), None, vlm, "go", budget=3, max_turns=2
        )
        self.assertEqual(log.finish_reason, REASON_FINISH)

    def test_no_ground_truth_in_prompts(self):
        vlm = RecordingScriptedVLM([tool_call("finish", reason="done")])
        AgentRuntime(make_registry("observe"), FakeWorld(), None, vlm).run("go")
        prompt = vlm.seen[0]["user"]
        self.assertNotIn("42.0", prompt)  # true target distance
        self.assertNotIn("(7.00, 8.00, 9.00)", prompt)  # true target position
        self.assertNotIn("target_pos_flu", vlm.seen[0]["context"])
        self.assertIn("pose_flu=(0.00, 0.00, 0.00)", prompt)


class ToolContextContractTest(unittest.TestCase):
    def test_context_fields_seen_by_tools(self):
        captured = {}

        def capture(ctx, args):
            captured["episode_id"] = ctx.episode_id
            captured["budget"] = dict(ctx.budget)
            captured["instruction"] = ctx.extra.get("instruction")
            captured["current_observation"] = ctx.extra.get("current_observation")
            captured["has_runtime"] = ctx.extra.get("runtime") is not None
            captured["world_is_fake"] = isinstance(ctx.world, FakeWorld)
            captured["log_len"] = len(ctx.log)
            return ToolResult(STATUS_OK, {"captured": True})

        registry = ToolRegistry()
        registry.register(_spec("capture", capture))
        vlm = RecordingScriptedVLM([tool_call("capture"), tool_call("finish", reason="ok")])
        log = AgentRuntime(
            registry, FakeWorld(), None, vlm, episode_id="ep_ctx"
        ).run("find the red book")

        self.assertEqual(captured["episode_id"], "ep_ctx")
        self.assertEqual(captured["instruction"], "find the red book")
        self.assertEqual(captured["current_observation"], "obs_1")
        self.assertTrue(captured["has_runtime"])
        self.assertTrue(captured["world_is_fake"])
        self.assertEqual(captured["log_len"], 0)
        self.assertGreaterEqual(captured["budget"]["tool_calls_left"], 1)
        # registry.call logged the call with timing
        self.assertEqual(log.turns[0].ms >= 0.0, True)

    def test_observation_ids_are_episode_scoped(self):
        world = FakeWorld()
        # three *distinct* observe calls: an identical repeat is answered from
        # the repeated-call cache (tests/test_tools_task.py) and captures nothing
        vlm = RecordingScriptedVLM(
            [
                tool_call("observe", mode="lookaround"),
                tool_call("observe", mode="forward"),
                tool_call("observe", mode="pitch_down"),
            ]
        )
        log = AgentRuntime(
            make_registry("observe"), world, None, vlm, budget=3, max_turns=3
        ).run("go")
        # turn frames and tool observations share one counter: obs_1 (turn 1),
        # obs_2 (tool), obs_3 (turn 2), ...
        ids = [t.evidence[0]["ref"] for t in log.turns if t.evidence]
        self.assertEqual(ids, ["obs_2", "obs_4", "obs_6"])

    def test_caller_tool_context_is_adopted(self):
        from vlm_nav.tools import ToolContext

        captured = {}

        def capture(ctx, args):
            captured["out_dir"] = ctx.extra.get("out_dir")
            captured["budget"] = dict(ctx.budget)
            return ToolResult(STATUS_OK, {"captured": True})

        registry = ToolRegistry()
        registry.register(_spec("capture", capture))
        ctx = ToolContext(
            world=None,
            memory=None,
            episode_id="placeholder",
            budget={"tool_calls_left": 5},
            extra={"out_dir": "/tmp/episode", "scan_views": 6},
        )
        vlm = RecordingScriptedVLM([tool_call("capture"), tool_call("finish", reason="ok")])
        log = AgentRuntime(
            registry, FakeWorld(), None, vlm, budget=2, tool_context=ctx,
            episode_id="ep_ctx",
        ).run("find the red book")

        # the caller's knobs, budget and log are used, not the runtime's copy
        self.assertEqual(captured["out_dir"], "/tmp/episode")
        self.assertEqual(captured["budget"]["tool_calls_left"], 5)
        self.assertEqual(ctx.budget["tool_calls_left"], 4)
        self.assertEqual(log.budget["tool_calls_left"], 4)
        self.assertEqual(len(ctx.log), 1)
        self.assertEqual(ctx.log[0]["tool"], "capture")
        self.assertEqual(ctx.episode_id, "ep_ctx")
        self.assertEqual(ctx.extra["instruction"], "find the red book")
        self.assertIs(ctx.extra["runtime"], log and ctx.extra["runtime"])


class ObservationRepositoryTest(unittest.TestCase):
    """Every frame the model is shown is an observation of the episode.

    The trajectory failure this pins down: the runtime rendered a frame, gave it
    an id, showed it to the model - and the next turn a tool that took that same
    id answered "unknown observation", because the id had never entered a store
    the tools read. Here the store is checked as the model's counter would: the
    id in the prompt resolves, and a frame another tool published under its own
    id is adopted with the pose it was captured at.
    """

    class CameraWorld(FakeWorld):
        """FakeWorld that declares its render intrinsics like the real ones."""

        def __init__(self):
            super().__init__()
            self.hfov_deg = 60.0

    @staticmethod
    def publisher(ctx, args):
        """A navigation-style tool: mint an id, record the pose, publish the frame."""
        obs_id = ctx.next_obs_id()
        ctx.extra.setdefault("observations", {})[obs_id] = {
            "observation_id": obs_id,
            "position_flu": [float(v) for v in ctx.world.position_flu],
            "yaw_rad": float(ctx.world.yaw_rad),
            "kind": "post_leg",
        }
        ctx.extra.setdefault("images", {})[obs_id] = torch.full((3, 4, 4), 0.75)
        return ToolResult(
            STATUS_OK,
            {"observation_id": obs_id, "published": True},
            [EvidenceRef("observation", obs_id, "post_leg")],
        )

    @staticmethod
    def publisher_away(ctx, args):
        """Same, but for a place 3 m ahead - somewhere a leg can actually fly to."""
        obs_id = ctx.next_obs_id()
        ctx.extra.setdefault("observations", {})[obs_id] = {
            "observation_id": obs_id,
            "position_flu": [3.0, 0.0, 0.0],
            "yaw_rad": 0.0,
            "kind": "post_leg",
        }
        ctx.extra.setdefault("images", {})[obs_id] = torch.full((3, 4, 4), 0.25)
        return ToolResult(
            STATUS_OK,
            {"observation_id": obs_id, "published": True},
            [EvidenceRef("observation", obs_id, "post_leg")],
        )

    def registry_with_perception(self):
        from vlm_nav.tool_impl_perception import register_perception_tools

        registry = ToolRegistry()
        registry.register(_spec("publish_frame", self.publisher))
        registry.register(_spec("publish_frame_away", self.publisher_away))
        register_perception_tools(registry)
        return registry

    def test_the_frame_of_every_turn_is_registered_with_its_provenance(self):
        world = self.CameraWorld()
        vlm = RecordingScriptedVLM([tool_call("observe", mode="forward"), tool_call("finish", reason="ok")])
        runtime = AgentRuntime(make_registry("observe"), world, None, vlm, max_turns=2)
        runtime.run("go")

        store = runtime.observation_store()
        records = store.ids()
        # obs_1 = the frame of turn 1, obs_2 = the frame of turn 2 (the observe
        # call itself adds its own captures to the same namespace)
        self.assertIn("obs_1", records)
        frame = store.get("obs_1")
        self.assertEqual(frame.kind, "evidence")
        self.assertEqual(frame.step, 1)
        self.assertEqual(tuple(frame.pose_flu), (0.0, 0.0, 0.0))
        self.assertEqual(frame.hfov_deg, 60.0)
        self.assertEqual((frame.width, frame.height), (4, 4))  # rendered size
        self.assertIsNotNone(store.frame("obs_1"))
        # the id the prompt named is the id of the store
        self.assertIn("current_observation=obs_1", vlm.seen[0]["user"])

    def test_a_frame_a_tool_published_is_adopted_with_its_capture_pose(self):
        world = self.CameraWorld()
        world.position_flu = np.asarray([3.0, 0.0, 0.5])
        world.yaw_rad = 1.0
        vlm = RecordingScriptedVLM([tool_call("publish_frame"), tool_call("finish", reason="ok")])
        runtime = AgentRuntime(
            self.registry_with_perception(), world, None, vlm, max_turns=2
        )
        runtime.run("go")

        store = runtime.observation_store()
        # turn 1 = obs_1, the published frame = obs_2
        adopted = store.get("obs_2")
        self.assertIsNotNone(adopted)
        self.assertEqual(adopted.kind, "evidence")
        self.assertEqual(tuple(round(v, 3) for v in adopted.pose_flu), (3.0, 0.0, 0.5))
        self.assertAlmostEqual(adopted.yaw_deg, math.degrees(1.0), places=3)
        self.assertEqual(adopted.hfov_deg, 60.0)

    def test_the_model_can_inspect_the_id_a_tool_published(self):
        world = self.CameraWorld()
        vlm = RecordingScriptedVLM(
            [
                tool_call("publish_frame"),
                tool_call("inspect_region", observation_id="obs_2", region=[0, 0, 500, 500]),
                tool_call("finish", reason="ok"),
            ]
        )
        runtime = AgentRuntime(
            self.registry_with_perception(), world, None, vlm, max_turns=3
        )
        log = runtime.run("go")

        inspected = log.turns[1]
        self.assertEqual(inspected.status, STATUS_OK)
        self.assertEqual(inspected.data["source_observation_id"], "obs_2")
        self.assertTrue(inspected.data["crop_image_ref"])

    def test_an_unregistered_frame_id_is_still_an_honest_error(self):
        """The store makes ids resolvable; it does not invent them: an id no
        tool ever returned still comes back as unknown, with the ids that do
        exist."""
        world = self.CameraWorld()
        vlm = RecordingScriptedVLM(
            [
                tool_call("inspect_region", observation_id="obs_99", region=[0, 0, 500, 500]),
                tool_call("finish", reason="ok"),
            ]
        )
        runtime = AgentRuntime(
            self.registry_with_perception(), world, None, vlm, max_turns=2
        )
        log = runtime.run("go")

        failure = log.turns[0]
        self.assertEqual(failure.status, STATUS_FAILED)
        self.assertEqual(failure.data["error"], "unknown_observation_id")
        self.assertIn("obs_1", failure.data["available"])

    def test_the_repository_is_episode_scoped(self):
        world = self.CameraWorld()
        ctx = ToolContext(
            world=world, episode_id="ep_1", budget={"tool_calls_left": 6}, extra={}
        )
        runtime = AgentRuntime(
            make_registry("observe"),
            world,
            None,
            RecordingScriptedVLM([tool_call("finish", reason="ok")]),
            max_turns=1,
            tool_context=ctx,
        )
        runtime.run("first")
        store = runtime.observation_store()
        first_ids = store.ids()

        runtime.vlm = RecordingScriptedVLM([tool_call("finish", reason="ok")])
        runtime.run("second")

        self.assertEqual(first_ids, ["obs_1"])
        # the previous episode's records are gone ...
        self.assertNotIn("obs_1", store.ids())
        # ... and the id counter continued instead of re-issuing obs_1
        self.assertEqual(store.ids(), ["obs_2"])

    def test_the_id_the_prompt_showed_resolves_for_the_navigation_tools(self):
        """The reported loop, step by step: the model is shown obs_1 as the
        current view, cites it on the next turn, and ``navigate`` must resolve it
        instead of answering ``unknown_observation``. (Flying to where you
        already are is a 0 m leg - that is the navigation side's semantics, and
        why the reserved ``return_to_observation`` exists for "go back to the
        pose this frame was taken at".)"""
        from vlm_nav.tool_impl_navigation import register_navigation_tools
        from vlm_nav.tool_impl_perception import resolve_observation_pose

        world = self.CameraWorld()
        registry = self.registry_with_perception()
        register_navigation_tools(registry)
        vlm = RecordingScriptedVLM(
            [tool_call("navigate", target_id="obs_1"), tool_call("finish", reason="ok")]
        )
        runtime = AgentRuntime(registry, world, None, vlm, max_turns=2)
        log = runtime.run("go")

        failure = log.turns[0]
        self.assertNotEqual(failure.data.get("reason"), "unknown_observation")
        self.assertNotEqual(failure.data.get("error"), "unknown_observation_id")
        self.assertEqual(failure.data.get("target_position"), [0.0, 0.0, 0.0])
        # the pose behind the id is the one the navigation tools resolve
        self.assertIsNotNone(
            resolve_observation_pose(runtime.ctx, "obs_1")
        )

    def test_a_recalled_place_is_a_navigable_target(self):
        from vlm_nav.tool_impl_navigation import register_navigation_tools

        world = self.CameraWorld()
        registry = self.registry_with_perception()
        register_navigation_tools(registry)
        vlm = RecordingScriptedVLM(
            [
                tool_call("publish_frame_away"),
                tool_call("return_to_observation", observation_id="obs_2"),
                tool_call("finish", reason="ok"),
            ]
        )
        log = AgentRuntime(registry, world, None, vlm, max_turns=3).run("go")

        leg = log.turns[1]
        self.assertEqual(leg.status, STATUS_OK)
        # one leg flown this call (the fake world moves 1 m per action and the
        # task keeps the pose it is heading for), not yet the arrival
        self.assertEqual(leg.data.get("outcome"), "partial")
        self.assertEqual(leg.data.get("task_status"), "running")
        self.assertEqual(leg.data.get("observation_id"), "obs_2")
        self.assertGreater(leg.data.get("moved_m") or 0.0, 0.0)
        self.assertEqual(leg.data.get("target_position"), [3.0, 0.0, 0.0])

    def test_the_store_summary_is_visible_in_the_prompt(self):
        world = self.CameraWorld()
        vlm = RecordingScriptedVLM([tool_call("observe", mode="forward"), tool_call("finish", reason="ok")])
        AgentRuntime(make_registry("observe"), world, None, vlm, max_turns=2).run("go")

        # turn 2 can see which observations exist (the ids it may cite)
        self.assertIn("observations_available=", vlm.seen[1]["user"])
        self.assertIn("obs_1", vlm.seen[1]["user"])


# ----------------------------------------------------------------------
# decision loop: fixed STATE block, semantic progress, failure recovery
# ----------------------------------------------------------------------

#: The ten labels of the per-turn STATE block, in the order it must render.
EXPECTED_STATE_LABELS = (
    "subgoal",
    "candidate",
    "evidence",
    "open_questions",
    "searched_coverage",
    "recent_failures",
    "next_info_needed",
    "budget_left",
    "plan_progress",
    "subtasks",
)
STATE_MARKER = "STATE (fixed labels):"


def state_block(prompt: str) -> str:
    """The fixed STATE section of a runtime prompt (its label lines only)."""
    index = prompt.index(STATE_MARKER)
    lines = []
    for line in prompt[index:].splitlines()[1:]:
        if not line.startswith("  ") or "=" not in line:
            break
        lines.append(line.strip())
    return "\n".join(lines)


def state_value(prompt: str, label: str) -> str:
    """The rendered value of one STATE label ("" when the label is missing)."""
    prefix = f"{label}="
    for line in state_block(prompt).splitlines():
        if line.startswith(prefix):
            return line[len(prefix):]
    return ""


def registry_of(*specs) -> ToolRegistry:
    registry = ToolRegistry()
    for spec in specs:
        registry.register(spec)
    return registry


def _register_view(ctx, obs_id: str) -> None:
    """Publish a frame + its capture pose under ``obs_id`` (tool-side channel)."""
    ctx.extra.setdefault("images", {})[obs_id] = ctx.world.render()
    ctx.extra.setdefault("observations", {})[obs_id] = {
        "observation_id": obs_id,
        "position_flu": [float(v) for v in ctx.world.position_flu],
        "yaw_rad": float(ctx.world.yaw_rad),
        "kind": "view",
    }


def _publish_current_view(ctx, args):
    """Mint a fresh obs id for the view the drone is looking at right now."""
    obs_id = ctx.next_obs_id()
    _register_view(ctx, obs_id)
    return ToolResult(
        STATUS_OK,
        {"observation_id": obs_id, "mode": args.get("mode", "forward")},
        [EvidenceRef("observation", obs_id, "current view")],
    )


def _turn_then_publish(ctx, args):
    """Turn in place (zero displacement) and publish the new view."""
    ctx.world.yaw_rad = float(ctx.world.yaw_rad) + math.radians(float(args.get("turn_deg", 45.0)))
    return _publish_current_view(ctx, args)


def _fly_one_leg(ctx, args):
    feedback = ctx.world.apply_action(NavAction(kind="waypoint", forward_m=1.0))
    return ToolResult(
        STATUS_OK,
        {
            "task_id": "nav_1",
            "progress_m": 1.0,
            "remaining_m": 4.0,
            "outcome": "partial",
            "reason": "partial",
        },
        [EvidenceRef("trajectory", "trace_1", feedback)],
    )


def _blocked_leg(ctx, args):
    return ToolResult(
        STATUS_BLOCKED,
        {
            "task_id": "nav_1",
            "outcome": "planner_failed",
            "reason": "planner_failed",
            "blocked_reason": "planner_failed",
            "progress_m": 0.0,
            # a *different* call: the recovery contract never re-offers the
            # call that just failed
            "next_action": {"tool": "go_to_viewpoint", "args": {"viewpoint_id": "vp_9"}},
        },
    )


def _repeating_move(ctx, args):
    """A movement stopped by the repeat guard (blocked, with a strategy change)."""
    return ToolResult(
        STATUS_BLOCKED,
        {
            "task_id": "nav_1",
            "outcome": "repeating_move",
            "reason": "repeating_move",
            "blocked_reason": "repeating_move",
            "progress_m": 0.0,
            "strategy_change": {"reason": "repeating_move", "options": []},
            "next_action": {"tool": "propose_viewpoints", "args": {"count": 3}},
        },
    )


def _search_area(ctx, args):
    """A multi-view search result: the contract the search skill returns."""
    obs_id = ctx.next_obs_id()
    _register_view(ctx, obs_id)
    return ToolResult(
        STATUS_OK,
        {
            "status": "exhausted",
            "steps": [
                {
                    "step": "view 0",
                    "why": "widest view of the doorway",
                    "obs_ids": [obs_id],
                    "result": "no_candidate",
                }
            ],
            "vlm_calls": 1,
            "cost": 1,
            "coverage": {
                "checked_deg": 240.0,
                "unchecked_deg": 120.0,
                "worth_looking": True,
            },
        },
        [EvidenceRef("observation", obs_id, "search step")],
    )


def _grounds_nothing(ctx, args):
    obs_id = str(args.get("observation_id") or ctx.extra.get("current_observation") or "")
    return ToolResult(
        STATUS_EMPTY,
        {
            "reason": "no_candidate",
            "instruction": str(args.get("instruction", "")),
            "observation_id": obs_id,
        },
        [EvidenceRef("observation", obs_id, "grounding frame")],
    )


def _grounds_current_view(ctx, args):
    """Ground the view of this turn: the id changes every call, the view does not."""
    obs_id = ctx.next_obs_id()
    _register_view(ctx, obs_id)
    return ToolResult(
        STATUS_EMPTY,
        {
            "reason": "no_candidate",
            "instruction": str(args.get("instruction", "")),
            "observation_id": obs_id,
        },
        [EvidenceRef("observation", obs_id, "grounding frame")],
    )


def _fails(ctx, args):
    return ToolResult(
        STATUS_FAILED,
        {"error": "unknown_observation_id", "observation_id": args.get("observation_id")},
    )


def _looks(ctx, args):
    return ToolResult(STATUS_OK, {"sector": args.get("sector", "north"), "clear": True})


class StateBlockTest(unittest.TestCase):
    """Every turn carries the same ten STATE labels, filled from real state."""

    def test_the_labels_are_the_documented_block(self):
        self.assertEqual(tuple(STATE_LABELS), EXPECTED_STATE_LABELS)

    def test_every_turn_prompt_carries_the_fixed_state_labels(self):
        vlm = RecordingScriptedVLM(
            [tool_call("observe", mode="forward"), tool_call("finish", reason="ok")]
        )
        AgentRuntime(make_registry("observe"), FakeWorld(), None, vlm, max_turns=2).run("go")

        for seen in vlm.seen:
            with self.subTest(prompt=seen["user"][:48]):
                self.assertIn(STATE_MARKER, seen["user"])
                rendered = [line for line in state_block(seen["user"]).splitlines()]
                self.assertEqual(
                    [line.split("=", 1)[0] for line in rendered],
                    list(EXPECTED_STATE_LABELS),
                )
                self.assertTrue(state_value(seen["user"], "budget_left"))
        # what the episode does not know yet says so instead of disappearing
        self.assertEqual(state_value(vlm.seen[0]["user"], "subgoal"), "n/a")
        self.assertEqual(state_value(vlm.seen[0]["user"], "plan_progress"), "no plan")

    def test_plan_progress_and_the_completion_condition_reach_the_block(self):
        registry = registry_of(_spec("observe", _publish_current_view))
        register_task_tools(registry)
        vlm = RecordingScriptedVLM(
            [
                tool_call(
                    "update_plan",
                    steps=["observe forward", "if clear, go to vp_1"],
                    current_step=2,
                    done=[1],
                    done_when="the doorway is confirmed clear",
                ),
                tool_call("finish", reason="ok"),
            ]
        )
        AgentRuntime(registry, FakeWorld(), None, vlm, budget=6, max_turns=3).run("inspect")

        prompt = vlm.seen[1]["user"]
        progress = state_value(prompt, "plan_progress")
        self.assertIn("plan_1", progress)
        self.assertIn("step 2/2", progress)
        self.assertIn("1 done", progress)
        self.assertIn("the doorway is confirmed clear", progress)
        self.assertEqual(state_value(prompt, "subgoal"), "if clear, go to vp_1")

    def test_search_coverage_reaches_the_block(self):
        registry = registry_of(_spec("search", _search_area))
        vlm = RecordingScriptedVLM(
            [tool_call("search", query="the red book"), tool_call("finish", reason="ok")]
        )
        AgentRuntime(registry, FakeWorld(), None, vlm, max_turns=2).run("find the red book")

        coverage = state_value(vlm.seen[1]["user"], "searched_coverage")
        self.assertIn("checked=240", coverage)
        self.assertIn("unchecked=120", coverage)
        self.assertIn("worth_looking=yes", coverage)

    def test_the_sector_coverage_of_a_lookaround_reaches_the_block(self):
        """The observe contract writes the coverage as 45 deg sector ranges and
        a list of views worth a second look; the STATE block renders it too."""
        observe_coverage = {
            "bin_deg": 45.0,
            "directions_known": True,
            "checked_deg": [[-180.0, -135.0], [-135.0, -90.0], [-90.0, -45.0], [-45.0, 0.0]],
            "unchecked_deg": [[0.0, 45.0], [45.0, 90.0], [90.0, 135.0], [135.0, 180.0]],
            "all_directions_covered": False,
            "worth_looking": [{"observation_id": "obs_2", "unchecked_deg": [0.0, 45.0]}],
            "note": "pick the direction you care about from view_index",
        }

        def lookaround(ctx, args):
            obs_id = ctx.next_obs_id()
            _register_view(ctx, obs_id)
            return ToolResult(
                STATUS_OK,
                {"observation_id": obs_id, "view_index": [], "coverage": observe_coverage},
                [EvidenceRef("observation", obs_id, "lookaround")],
            )

        registry = registry_of(_spec("observe", lookaround))
        vlm = RecordingScriptedVLM(
            [tool_call("observe", mode="lookaround"), tool_call("finish", reason="ok")]
        )
        AgentRuntime(registry, FakeWorld(), None, vlm, max_turns=2).run("look around")

        coverage = state_value(vlm.seen[1]["user"], "searched_coverage")
        self.assertIn("checked=180", coverage)
        self.assertIn("unchecked=180", coverage)
        self.assertIn("worth_looking=yes(1 views)", coverage)

    def test_advice_for_a_fully_covered_lookaround(self):
        advice = recovery_advice(
            "observe",
            {"coverage": {"verified": "all directions covered"}},
            STATUS_OK,
            available=("observe", "propose_viewpoints", "go_to_viewpoint"),
            coverage={
                "directions_known": True,
                "checked_deg": [[-180.0, -90.0], [-90.0, 0.0], [0.0, 90.0], [90.0, 180.0]],
                "unchecked_deg": [],
                "all_directions_covered": True,
                "worth_looking": [],
            },
        )
        text = " ".join(advice)
        self.assertIn("searched out", text)
        self.assertIn("propose_viewpoints", text)
        self.assertIn("expected_info", text)

    def test_a_fruitless_grounding_is_an_open_question_not_an_absence_claim(self):
        registry = registry_of(_spec("ground_target", _grounds_nothing))
        vlm = RecordingScriptedVLM(
            [
                tool_call("ground_target", instruction="the red book"),
                tool_call("finish", reason="ok"),
            ]
        )
        AgentRuntime(registry, FakeWorld(), None, vlm, max_turns=2).run("find the red book")

        block = state_block(vlm.seen[1]["user"])
        self.assertIn("obs_1", block)
        # a view that showed no candidate is not evidence the target is not there
        self.assertIn("not evidence of absence", block)
        for claim in ("does not exist", "occluded", "target is absent", "is not in the scene"):
            with self.subTest(claim=claim):
                self.assertNotIn(claim, block)

    def test_recent_failures_name_the_tool_the_status_and_the_reason(self):
        registry = registry_of(_spec("look", _looks), _spec("look_closely", _fails))
        vlm = RecordingScriptedVLM(
            [
                tool_call("look", sector="north"),
                tool_call("look_closely", observation_id="obs_99"),
                tool_call("look", sector="south"),
                tool_call("finish", reason="ok"),
            ]
        )
        AgentRuntime(registry, FakeWorld(), None, vlm, budget=8, max_turns=4).run("scan")

        failures = state_value(vlm.seen[2]["user"], "recent_failures")
        self.assertIn("look_closely", failures)
        self.assertIn("unknown_observation_id", failures)

    def test_a_repeating_move_reaches_the_state_block(self):
        """The repeat guard's block is a failure the STATE block can read."""
        registry = registry_of(
            _spec("go_to_viewpoint", _repeating_move, side_effects=True)
        )
        vlm = RecordingScriptedVLM(
            [
                tool_call("go_to_viewpoint", viewpoint_id="vp_7"),
                tool_call("finish", reason="blocked"),
            ]
        )
        AgentRuntime(registry, FakeWorld(), None, vlm, max_turns=2).run("reach the doorway")

        failures = state_value(vlm.seen[1]["user"], "recent_failures")
        self.assertIn("repeating_move", failures)
        needed = state_value(vlm.seen[1]["user"], "next_info_needed")
        self.assertIn("propose_viewpoints", needed)

    def test_next_info_needed_follows_the_tools_own_next_step(self):
        registry = registry_of(_spec("go_to_viewpoint", _blocked_leg, side_effects=True))
        vlm = RecordingScriptedVLM(
            [
                tool_call("go_to_viewpoint", viewpoint_id="vp_7"),
                tool_call("finish", reason="blocked"),
            ]
        )
        AgentRuntime(registry, FakeWorld(), None, vlm, max_turns=2).run("reach the doorway")

        needed = state_value(vlm.seen[1]["user"], "next_info_needed")
        self.assertIn("go_to_viewpoint", needed)
        self.assertIn("vp_9", needed)

    def test_the_state_is_readable_outside_the_loop(self):
        vlm = RecordingScriptedVLM(
            [tool_call("observe", mode="forward"), tool_call("finish", reason="ok")]
        )
        runtime = AgentRuntime(make_registry("observe"), FakeWorld(), None, vlm, max_turns=2)
        log = runtime.run("go")

        state = runtime.state()
        self.assertEqual(tuple(state), EXPECTED_STATE_LABELS)
        self.assertEqual(tuple(log.state), EXPECTED_STATE_LABELS)
        payload = log.to_dict()
        self.assertEqual(tuple(payload["state"]), EXPECTED_STATE_LABELS)
        self.assertIn("recoveries", payload)
        json.dumps(payload)  # the log stays JSON-serializable

    def test_the_candidate_line_leads_with_the_id_the_tools_take(self):
        """The candidate line is the id the model copies into its next call.

        It rendered ``cand_1@obs_2`` - the candidate id glued to the frame it
        was grounded in, a spelling no movement tool resolves. In the G1 matrix
        a model pasted it into ``approach_candidate``, got
        ``unknown_candidate_id`` and burned the rest of the episode re-typing
        ids (vlm_nav/runtime.py ``_candidate_text``; G1 §5.7). The frame stays
        on the line, as its own clause.
        """
        from vlm_nav.observations import CANDIDATE_STORE_KEY

        ctx = ToolContext(
            world=FakeWorld(), memory=None, budget={"tool_calls_left": 4}
        )
        ctx.extra[CANDIDATE_STORE_KEY] = {
            "cand_1": {
                "id": "cand_1",
                "source": "ground_target",
                "observation_id": "obs_2",
                "position_flu": [3.0, 0.0, 0.0],
                "confidence": 0.8,
            }
        }
        vlm = RecordingScriptedVLM([tool_call("finish", reason="ok")])
        runtime = AgentRuntime(
            make_registry("observe"), FakeWorld(), None, vlm, budget=4, max_turns=1,
            tool_context=ctx,
        )
        runtime.run("go")

        value = state_value(vlm.seen[0]["user"], "candidate")
        self.assertNotIn("cand_1@obs_2", value)
        self.assertEqual(value.split()[0], "cand_1")
        self.assertIn("obs_2", value)  # the frame is still reported
        self.assertEqual(runtime.state()["candidate"], value)


class ProgressKindTest(unittest.TestCase):
    """Progress is judged on pose + heading + view content + query intent."""

    def test_the_progress_vocabulary(self):
        self.assertEqual(
            tuple(PROGRESS_KINDS),
            (PROGRESS_INFORMATION, PROGRESS_ACTION, PROGRESS_TASK, PROGRESS_NONE),
        )

    def test_a_turn_record_reports_its_progress_kind(self):
        vlm = RecordingScriptedVLM(
            [tool_call("observe", mode="forward"), tool_call("finish", reason="ok")]
        )
        log = AgentRuntime(make_registry("observe"), FakeWorld(), None, vlm, max_turns=2).run("go")

        self.assertEqual(log.turns[0].progress_kind, PROGRESS_INFORMATION)
        self.assertTrue(log.turns[0].new_evidence)
        self.assertEqual(log.turns[1].progress_kind, PROGRESS_TASK)
        for record in log.to_dict()["turns"]:
            self.assertIn(record["progress_kind"], PROGRESS_KINDS)
            self.assertIn("recovery", record)

    def test_the_same_view_under_a_new_id_is_not_new_information(self):
        """The reported bug: a fresh obs_N of the same place/heading/content,
        asked for by the same query, is not progress."""
        registry = registry_of(_spec("reobserve", _publish_current_view, cacheable=False))
        vlm = RecordingScriptedVLM([tool_call("reobserve"), tool_call("reobserve")])
        log = AgentRuntime(
            registry, FakeWorld(), None, vlm, budget=8, max_turns=6, max_stagnant_turns=1
        ).run("look at the doorway")

        self.assertEqual(log.turns[0].progress_kind, PROGRESS_INFORMATION)
        self.assertEqual(log.turns[0].evidence[0]["ref"], "obs_2")
        self.assertEqual(log.turns[1].progress_kind, PROGRESS_NONE)
        self.assertFalse(log.turns[1].new_evidence)
        self.assertEqual(log.finish_reason, REASON_NO_NEW_EVIDENCE)
        self.assertEqual(len(log.turns), 2)

    def test_a_new_heading_is_information_progress_without_moving(self):
        """Zero displacement is not a failure by itself: turning in place and
        seeing something new is information progress."""
        registry = registry_of(
            _spec("turn", _turn_then_publish, side_effects=True, cacheable=False)
        )
        world = FakeWorld()
        vlm = RecordingScriptedVLM([tool_call("turn"), tool_call("turn"), tool_call("finish", reason="checked")])
        log = AgentRuntime(
            registry, world, None, vlm, budget=8, max_turns=6, max_stagnant_turns=1
        ).run("look around")

        self.assertEqual(
            [record.progress_kind for record in log.turns[:2]],
            [PROGRESS_INFORMATION, PROGRESS_INFORMATION],
        )
        self.assertEqual(tuple(float(v) for v in world.position_flu), (0.0, 0.0, 0.0))
        self.assertEqual(tuple(float(v) for v in world.position_flu), tuple(world.trajectory[0]))
        self.assertEqual(log.finish_reason, REASON_FINISH)

    def test_a_different_query_on_the_same_view_is_new_information(self):
        registry = registry_of(_spec("reobserve", _publish_current_view, cacheable=False))
        vlm = RecordingScriptedVLM(
            [
                tool_call("reobserve", mode="forward"),
                tool_call("reobserve", mode="lookaround"),
                tool_call("finish", reason="both queries done"),
            ]
        )
        log = AgentRuntime(
            registry, FakeWorld(), None, vlm, budget=8, max_turns=6, max_stagnant_turns=1
        ).run("look around")

        self.assertEqual(
            [record.progress_kind for record in log.turns[:2]],
            [PROGRESS_INFORMATION, PROGRESS_INFORMATION],
        )
        self.assertEqual(log.finish_reason, REASON_FINISH)

    def test_a_flown_leg_is_action_progress(self):
        registry = registry_of(_spec("go_to_viewpoint", _fly_one_leg, side_effects=True))
        world = FakeWorld()
        vlm = RecordingScriptedVLM(
            [tool_call("go_to_viewpoint", viewpoint_id="vp_1"), tool_call("finish", reason="leg flown")]
        )
        log = AgentRuntime(registry, world, None, vlm, budget=6, max_turns=3).run("fly to vp_1")

        self.assertEqual(log.turns[0].progress_kind, PROGRESS_ACTION)
        self.assertTrue(log.turns[0].new_evidence)
        self.assertEqual(tuple(float(v) for v in world.position_flu), (1.0, 0.0, 0.0))

    def test_a_task_signal_is_task_progress(self):
        for data in ({"verified": True}, {"subgoal_complete": True}, {"verdict": "same"}):
            with self.subTest(data=data):
                registry = registry_of(
                    _spec("check", lambda ctx, args, data=data: ToolResult(STATUS_OK, dict(data)))
                )
                vlm = RecordingScriptedVLM(
                    [tool_call("check"), tool_call("finish", reason="done")]
                )
                log = AgentRuntime(
                    registry, FakeWorld(), None, vlm, budget=4, max_turns=2
                ).run("go")
                self.assertEqual(log.turns[0].progress_kind, PROGRESS_TASK)

    def test_a_tool_can_declare_its_own_progress_kind(self):
        cases = (
            ("task", PROGRESS_TASK),
            ("action", PROGRESS_ACTION),
            ("none", PROGRESS_NONE),
            ("nonsense", PROGRESS_INFORMATION),  # an unknown word is ignored
        )
        for declared, expected in cases:
            with self.subTest(declared=declared):
                def handler(ctx, args, declared=declared):
                    return ToolResult(STATUS_OK, {"progress_kind": declared, "probe": 1})

                registry = registry_of(_spec("probe", handler, cacheable=False))
                vlm = RecordingScriptedVLM(
                    [tool_call("probe"), tool_call("finish", reason="done")]
                )
                log = AgentRuntime(
                    registry, FakeWorld(), None, vlm, budget=4, max_turns=2
                ).run("go")
                self.assertEqual(log.turns[0].progress_kind, expected)
                self.assertEqual(log.turns[0].new_evidence, expected != PROGRESS_NONE)

    def test_regrounding_a_fresh_obs_of_the_same_view_is_not_progress(self):
        """The reported loop: the model blind-grounds the *new* obs_N of every
        turn from the same place. A new id of the same (pose, heading, content)
        view asked with the same query is not progress."""
        registry = registry_of(
            _spec("observe", _publish_current_view, cacheable=False),
            _spec("ground_target", _grounds_current_view, cacheable=False),
        )
        vlm = RecordingScriptedVLM(
            [
                tool_call("ground_target", instruction="the red book"),
                tool_call("ground_target", instruction="the red book"),
                tool_call("finish", reason="nothing found here"),
            ]
        )
        log = AgentRuntime(
            registry, FakeWorld(), None, vlm, budget=8, max_turns=5, max_stagnant_turns=1
        ).run("find the red book")

        self.assertEqual(
            [record.progress_kind for record in log.turns[:2]],
            [PROGRESS_INFORMATION, PROGRESS_NONE],
        )
        # a fresh obs id did not buy the second call any progress: the loop
        # recovers instead of grinding the same view to the budget end
        self.assertEqual(log.recoveries[0]["kind"], RECOVERY_ADVICE)
        self.assertEqual(log.recoveries[0]["stalled_tool"], "ground_target")
        self.assertIn("not evidence of absence", " ".join(log.recoveries[0]["advice"]))
        self.assertEqual(log.finish_reason, REASON_FINISH)

    def test_a_replayed_answer_is_no_progress(self):
        vlm = RecordingScriptedVLM(
            [tool_call("probe", sector="3"), tool_call("probe", sector="3")]
        )
        log = AgentRuntime(
            make_registry("probe"),
            FakeWorld(),
            None,
            vlm,
            budget=8,
            max_turns=4,
            max_stagnant_turns=1,
        ).run("scan")

        self.assertEqual(log.turns[1].status, STATUS_CACHED)
        self.assertEqual(log.turns[1].progress_kind, PROGRESS_NONE)
        self.assertEqual(log.finish_reason, REASON_NO_NEW_EVIDENCE)

    def test_a_volatile_timestamp_is_not_new_information(self):
        """A tool that stamps the same answer with a new time has not made
        progress - "the return value changed" is not evidence."""

        def poll(ctx, args):
            return ToolResult(
                STATUS_OK, {"state": "unchanged", "updated_t": time.time()}
            )

        registry = registry_of(_spec("poll", poll, cacheable=False))
        vlm = RecordingScriptedVLM([tool_call("poll"), tool_call("poll")])
        log = AgentRuntime(
            registry, FakeWorld(), None, vlm, budget=8, max_turns=4, max_stagnant_turns=1
        ).run("poll the state")

        self.assertEqual(log.turns[0].progress_kind, PROGRESS_INFORMATION)
        self.assertEqual(log.turns[1].progress_kind, PROGRESS_NONE)
        self.assertEqual(log.finish_reason, REASON_NO_NEW_EVIDENCE)


class RecoveryTest(unittest.TestCase):
    """Stagnation recovers first and only then terminates (goal: no blind loops)."""

    ADVICE_TOOLS = (
        "observe",
        "inspect_region",
        "propose_viewpoints",
        "go_to_viewpoint",
        "ground_target",
        "compare_candidates",
        "approach_candidate",
    )

    def test_advice_for_a_search_that_hit_its_step_cap_points_at_continuing(self):
        """A capped routine is not an absent target: continue the search.

        The search skill returns what it has when it reaches its inner-step cap
        (``inner_steps_exhausted``); the advice must send the model back into the
        routine or to a new place, not into re-grounding one view.
        """
        advice = recovery_advice(
            "search_target",
            {
                "reason": "inner_steps_exhausted",
                "status": "empty",
                "candidates": [],
                "instruction": "the red book",
                "next_action": {"tool": "search_target",
                                "args": {"instruction": "the red book"}},
            },
            STATUS_EMPTY,
            available=self.ADVICE_TOOLS + ("search_target",),
        )
        text = " ".join(advice)
        self.assertIn("search_target", text)
        self.assertIn("step cap", text)
        for claim in ("does not exist", "occluded", "must be behind"):
            self.assertNotIn(claim, text)

    def test_advice_for_a_fruitless_grounding(self):
        advice = recovery_advice(
            "ground_target",
            {"reason": "no_candidate", "observation_id": "obs_3"},
            STATUS_EMPTY,
            available=self.ADVICE_TOOLS,
        )
        text = " ".join(advice)
        self.assertIn("not evidence of absence", text)
        for tool in ("observe", "inspect_region", "go_to_viewpoint"):
            with self.subTest(tool=tool):
                self.assertIn(tool, text)
        for claim in ("does not exist", "occluded", "must be behind"):
            with self.subTest(claim=claim):
                self.assertNotIn(claim, text)

    def test_advice_for_a_blocked_leg_uses_the_tools_own_next_step(self):
        advice = recovery_advice(
            "go_to_viewpoint",
            {
                "outcome": "planner_failed",
                "reason": "planner_failed",
                "blocked_reason": "planner_failed",
                "next_action": {"tool": "go_to_viewpoint", "args": {"viewpoint_id": "vp_7"}},
            },
            STATUS_BLOCKED,
            available=self.ADVICE_TOOLS,
        )
        text = " ".join(advice)
        self.assertIn("vp_7", text)
        self.assertIn("go_to_viewpoint", text)
        self.assertIn("propose_viewpoints", text)

    def test_advice_does_not_reoffer_the_call_that_just_failed(self):
        """The recovery path must not push the call that just made no progress.

        A tool result whose ``next_action`` is the same call (same tool, same
        target) used to be echoed as "the tool's own next step", which is the
        recovery-side half of the repeating-``search_target`` loop.
        """
        advice = recovery_advice(
            "approach_candidate",
            {
                "outcome": "no_progress",
                "reason": "no_progress",
                "blocked_reason": "no_progress",
                "candidate_id": "cand_1",
                "next_action": {
                    "tool": "approach_candidate",
                    "args": {"candidate_id": "cand_1"},
                },
            },
            STATUS_FAILED,
            available=self.ADVICE_TOOLS,
            args={"candidate_id": "cand_1"},
        )
        text = " ".join(advice)
        self.assertNotIn("the tool's own next step", text)
        self.assertIn("do not repeat", text)
        self.assertIn("propose_viewpoints", text)

    def test_advice_for_a_repeating_move_names_the_strategy_change(self):
        """The tool-level repeat guard feeds the runtime's recovery rules."""
        advice = recovery_advice(
            "approach_candidate",
            {
                "outcome": "repeating_move",
                "reason": "repeating_move",
                "blocked_reason": "repeating_move",
                "candidate_id": "cand_1",
                "next_action": {"tool": "propose_viewpoints", "args": {"count": 3}},
            },
            STATUS_BLOCKED,
            available=self.ADVICE_TOOLS,
            args={"candidate_id": "cand_1"},
        )
        text = " ".join(advice)
        self.assertIn("repeating_move", text)
        self.assertIn("propose_viewpoints", text)
        self.assertIn("go_to_viewpoint", text)

    def test_advice_for_a_distance_uncertain_candidate_does_not_reconfirm(self):
        """The v4 confirm loop: an uncertain range cannot be fixed by confirming.

        ``confirm_candidate`` used to feed its uncertain verdict into the
        generic identity rule, which told the model to confirm the same
        candidate from another heading. The advice now changes the view and
        re-grounds, and never offers another confirmation of that candidate.
        """
        advice = recovery_advice(
            "confirm_candidate",
            {
                "verdict": "uncertain",
                "reason": "candidate_distance_uncertain",
                "candidate_id": "cand_1",
                "distance_source": "conflict",
            },
            STATUS_BLOCKED,
            available=self.ADVICE_TOOLS,
        )
        text = " ".join(advice)
        self.assertIn("ground_target", text)
        self.assertIn("observe", text)
        self.assertIn("do not call confirm_candidate again", text)
        self.assertNotIn('confirm_candidate(candidate_id="cand_1")', text)

    def test_advice_for_a_searched_out_area(self):
        advice = recovery_advice(
            "search",
            {"status": "exhausted"},
            STATUS_OK,
            available=self.ADVICE_TOOLS,
            coverage={"checked_deg": 360.0, "unchecked_deg": 0.0, "worth_looking": False},
        )
        text = " ".join(advice)
        self.assertIn("propose_viewpoints", text)
        self.assertIn("go_to_viewpoint", text)
        self.assertIn("expected_info", text)

    def test_advice_for_a_candidate_identity_conflict(self):
        advice = recovery_advice(
            "compare_candidates",
            {"verdict": "uncertain", "conflicts": ["color"]},
            STATUS_OK,
            available=self.ADVICE_TOOLS,
        )
        text = " ".join(advice)
        self.assertIn("compare_candidates", text)
        self.assertIn("evidence", text)

    def test_advice_for_a_repeated_read_is_a_fresh_query(self):
        advice = recovery_advice(
            "look",
            {"hint": REPEAT_HINT, "reason": "repeat"},
            STATUS_CACHED,
            available=self.ADVICE_TOOLS,
        )
        text = " ".join(advice)
        self.assertIn("observe", text)
        self.assertIn("inspect_region", text)

    def test_advice_names_only_registered_tools(self):
        advice = recovery_advice(
            "ground_target",
            {"reason": "no_candidate", "observation_id": "obs_3"},
            STATUS_EMPTY,
            available=("observe", "ground_target"),
        )
        text = " ".join(advice)
        self.assertIn("observe", text)
        self.assertNotIn("go_to_viewpoint", text)
        self.assertNotIn("inspect_region", text)

    def test_no_advice_when_the_registry_has_nothing_else_to_call(self):
        self.assertEqual(
            recovery_advice("probe", {"checked": "sector"}, STATUS_CACHED, available=("probe",)),
            [],
        )
        # plan bookkeeping is not evidence: revising a plan is not a recovery step
        self.assertEqual(
            recovery_advice(
                "get_plan", {"reason": "no_plan"}, STATUS_EMPTY,
                available=("update_plan", "get_plan"),
            ),
            [],
        )

    def test_stagnation_injects_advice_instead_of_terminating(self):
        registry = registry_of(
            _spec("reobserve", _publish_current_view, cacheable=False),
            _spec("observe", _publish_current_view),
            _spec("propose_viewpoints", _propose_viewpoints),
            _spec("go_to_viewpoint", _fly_one_leg, side_effects=True),
        )
        vlm = RecordingScriptedVLM(
            [
                tool_call("reobserve"),
                tool_call("reobserve"),
                tool_call("reobserve"),
                tool_call("finish", reason="done"),
            ]
        )
        runtime = AgentRuntime(
            registry, FakeWorld(), None, vlm, budget=8, max_turns=6, max_stagnant_turns=1
        )
        log = runtime.run("find the red book")

        self.assertEqual(log.finish_reason, REASON_FINISH)
        self.assertEqual(
            [entry["kind"] for entry in log.recoveries],
            [RECOVERY_ADVICE, RECOVERY_ADVICE],
        )
        self.assertEqual(log.recoveries[0]["stalled_tool"], "reobserve")
        self.assertEqual(log.recoveries[0]["turn"], 2)
        self.assertIn("observe", " ".join(log.recoveries[0]["advice"]))
        self.assertEqual(runtime.last_stall_action, RECOVERY_ADVICE)
        # the advice is in the prompt of the very next turn, as the state it
        # answers and as the instruction it carries
        prompt = vlm.seen[2]["user"]
        self.assertIn("RECOVERY", prompt)
        self.assertIn("observe", prompt)
        # the state block names the recovery step as the next information needed
        self.assertIn("observe", state_value(prompt, "next_info_needed"))

    def test_recovery_is_bounded_and_ends_by_asking_for_an_honest_finish(self):
        registry = registry_of(
            _spec("reobserve", _publish_current_view, cacheable=False),
            _spec("observe", _publish_current_view),
        )
        vlm = RecordingScriptedVLM([tool_call("reobserve")] * 8)
        runtime = AgentRuntime(
            registry,
            FakeWorld(),
            None,
            vlm,
            budget=20,
            max_turns=10,
            max_stagnant_turns=1,
            max_recovery_attempts=1,
        )
        log = runtime.run("find the red book")

        # turn 1 information, turn 2 gets one recovery, turn 3 gets the
        # honest-finish grant, turn 4 still no progress: the loop stops
        self.assertEqual(
            [entry["kind"] for entry in log.recoveries],
            [RECOVERY_ADVICE, RECOVERY_FINAL],
        )
        self.assertEqual(len(log.turns), 4)
        self.assertEqual(log.finish_reason, REASON_NO_NEW_EVIDENCE)
        self.assertFalse(log.success)
        self.assertIn("RECOVERY EXHAUSTED", vlm.seen[3]["user"])
        self.assertEqual(runtime.last_stall_action, RECOVERY_STOP)

    def test_the_final_grant_lets_the_model_close_with_an_unfinished_reason(self):
        registry = registry_of(
            _spec("reobserve", _publish_current_view, cacheable=False),
            _spec("observe", _publish_current_view),
        )
        vlm = RecordingScriptedVLM(
            [
                tool_call("reobserve"),
                tool_call("reobserve"),
                tool_call("reobserve"),
                tool_call("finish", reason="not finished: no candidate yet", success=False),
            ]
        )
        runtime = AgentRuntime(
            registry,
            FakeWorld(),
            None,
            vlm,
            budget=8,
            max_turns=6,
            max_stagnant_turns=1,
            max_recovery_attempts=1,
        )
        log = runtime.run("find the red book")

        self.assertEqual(
            [entry["kind"] for entry in log.recoveries],
            [RECOVERY_ADVICE, RECOVERY_FINAL],
        )
        self.assertEqual(log.finish_reason, REASON_FINISH)
        self.assertFalse(log.success)
        self.assertEqual(log.finish_args["reason"], "not finished: no candidate yet")

    def test_a_recovery_hook_can_execute_the_recovery_call(self):
        requests = []

        def hook(request):
            requests.append(request)
            return {"tool": "observe", "args": {"mode": "lookaround"}}

        registry = registry_of(
            _spec("reobserve", _publish_current_view, cacheable=False),
            _spec("observe", _publish_current_view),
        )
        vlm = RecordingScriptedVLM(
            [tool_call("reobserve"), tool_call("reobserve"), tool_call("finish", reason="done")]
        )
        runtime = AgentRuntime(
            registry,
            FakeWorld(),
            None,
            vlm,
            budget=8,
            max_turns=6,
            max_stagnant_turns=1,
            recovery_hook=hook,
        )
        log = runtime.run("find the red book")

        hook_turns = [record for record in log.turns if record.recovery == RECOVERY_HOOK]
        self.assertEqual(len(hook_turns), 1)
        self.assertEqual(hook_turns[0].tool, "observe")
        self.assertEqual(hook_turns[0].status, STATUS_OK)
        self.assertEqual(requests[0]["stalled_tool"], "reobserve")
        self.assertEqual(requests[0]["attempt"], 1)
        self.assertIn("advice", requests[0])
        self.assertEqual(log.finish_reason, REASON_FINISH)
        # the automatic call really ran and the model did not spend a turn asking
        # for it: its result is in the very next prompt
        self.assertEqual(log.vlm_calls, 3)
        self.assertIn("observe(", vlm.seen[2]["user"])

    def test_a_hook_that_returns_nothing_leaves_the_advice_in_place(self):
        registry = registry_of(
            _spec("reobserve", _publish_current_view, cacheable=False),
            _spec("observe", _publish_current_view),
        )
        vlm = RecordingScriptedVLM(
            [tool_call("reobserve"), tool_call("reobserve"), tool_call("finish", reason="done")]
        )
        log = AgentRuntime(
            registry,
            FakeWorld(),
            None,
            vlm,
            budget=8,
            max_turns=6,
            max_stagnant_turns=1,
            recovery_hook=lambda request: None,
        ).run("find the red book")

        self.assertEqual([record.recovery for record in log.turns], ["", "advice", ""])
        self.assertEqual([entry["kind"] for entry in log.recoveries], [RECOVERY_ADVICE])
        self.assertIn("RECOVERY", vlm.seen[2]["user"])


# ----------------------------------------------------------------------
# P4: the sub-task executor and the decision nodes of a long-horizon plan
# ----------------------------------------------------------------------


def executor_registry(*names):
    """A registry with the runtime test doubles plus the plan tools."""
    registry = make_registry(*names)
    register_task_tools(registry)
    return registry


class SubtaskExecutorTest(unittest.TestCase):
    """The runtime's executor advances the sub-task records after every call, so
    routine progress costs no decision turn; the model is asked to revise the
    sub-plan only at a decision node (P4)."""

    def test_the_state_block_carries_the_subtasks_label(self):
        vlm = RecordingScriptedVLM(
            [
                tool_call("update_plan", steps=["scan the corridor", "check the cabinet"]),
                tool_call("step_ok"),
                tool_call("finish", reason="ok"),
            ]
        )
        AgentRuntime(
            executor_registry("step_ok"), FakeWorld(), None, vlm, budget=6, max_turns=4
        ).run("find the red book")

        # before the plan exists the label says so (it never disappears)
        self.assertEqual(state_value(vlm.seen[0]["user"], "subtasks"), "no plan")
        after_plan = state_value(vlm.seen[1]["user"], "subtasks")
        self.assertIn("st 1/2", after_plan)
        self.assertIn("scan the corridor", after_plan)
        after_step = state_value(vlm.seen[2]["user"], "subtasks")
        self.assertIn("current=st_2[active]", after_step)
        self.assertIn("check the cabinet", after_step)

    def test_a_completed_stage_does_not_spend_the_next_stages_next_step(self):
        """The marker belongs to the sub-task the call served, not to whichever
        sub-task the executor auto-activated while running it."""
        vlm = RecordingScriptedVLM(
            [
                tool_call(
                    "update_plan",
                    steps=[
                        {"goal": "scan the corridor",
                         "next_step": tool_call("step_ok")},
                        {"goal": "check the cabinet",
                         "next_step": tool_call("step_ok")},
                    ],
                ),
                tool_call("observe"),
                tool_call("finish", reason="ok"),
            ]
        )
        runtime = AgentRuntime(
            executor_registry("step_ok", "observe"),
            FakeWorld(),
            None,
            vlm,
            budget=10,
            max_turns=4,
        )
        log = runtime.run("find the red book")

        autos = [entry for entry in log.executor_auto_steps
                 if entry["tool"] == "step_ok"]
        self.assertEqual(len(autos), 2, log.executor_auto_steps)
        subtasks = plan_state(runtime.ctx)["subtasks"]
        self.assertEqual([item["status"] for item in subtasks], ["done", "done"])
        self.assertTrue(all(item["_next_step_spent"] for item in subtasks))

    def test_a_plan_next_step_runs_once_per_subtask(self):
        """A static ``next_step`` is a *step*, not a loop: it is consumed by the
        first automatic execution, so a routine turn cannot re-run it (the
        r5 seed-11 grounding loop hit the stagnant-grounding guard)."""
        vlm = RecordingScriptedVLM(
            [
                tool_call(
                    "update_plan",
                    steps=[{"goal": "scan the corridor",
                            "next_step": tool_call("counter")}],
                ),
                tool_call("observe"),
                tool_call("observe"),
                tool_call("finish", reason="ok"),
            ]
        )
        runtime = AgentRuntime(
            executor_registry("counter", "observe"),
            FakeWorld(),
            None,
            vlm,
            budget=10,
            max_turns=6,
        )
        log = runtime.run("find the red book")

        autos = [entry for entry in log.executor_auto_steps if entry["tool"] == "counter"]
        self.assertEqual(len(autos), 1, log.executor_auto_steps)
        self.assertEqual(autos[0]["origin"], "subtask_next_step")
        subtask = plan_state(runtime.ctx)["subtasks"][0]
        self.assertTrue(subtask["_next_step_spent"])

    def test_a_completed_subtask_advances_without_a_plan_turn(self):
        vlm = RecordingScriptedVLM(
            [
                tool_call("update_plan", steps=["scan the corridor", "check the cabinet"]),
                tool_call("step_ok"),
                tool_call("finish", reason="ok"),
            ]
        )
        runtime = AgentRuntime(
            executor_registry("step_ok"), FakeWorld(), None, vlm, budget=6, max_turns=4
        )
        log = runtime.run("find the red book")

        subtasks = plan_state(runtime.ctx)["subtasks"]
        self.assertEqual([s["status"] for s in subtasks], ["done", "active"])
        self.assertEqual(subtasks[0]["reason"], "done_when_met")
        # the frame the step tool published is the evidence of its sub-task
        self.assertEqual(subtasks[0]["evidence"], ["observation:obs_3"])
        # the plan moved without the model spending a turn on bookkeeping
        self.assertEqual(log.tool_names(), ["update_plan", "step_ok", "finish"])
        # rev 1 = the plan, rev 2/3 = the executor's two transitions (done, active)
        self.assertEqual(plan_state(runtime.ctx)["revision"], 3)
        # the model is told what the executor did (and why it may continue)
        self.assertIn("EXECUTOR", vlm.seen[2]["user"])
        self.assertIn("st_2", vlm.seen[2]["user"])
        # the switch is the decision node of that turn
        self.assertEqual(
            log.decision_nodes,
            [{"turn": 2, "node": "subtask_switch", "subtask": "st_2"}],
        )
        self.assertEqual(log.turns[1].decision, "subtask_switch")
        self.assertEqual(log.plan_transitions[-1]["to"], "active")

    def test_the_failure_branch_leads_the_recovery_advice(self):
        vlm = RecordingScriptedVLM(
            [
                tool_call(
                    "update_plan",
                    steps=[
                        {
                            "goal": "fly to vp_3",
                            "failure_branch": ["propose_viewpoints for a clean approach; go_to_viewpoint vp_9"],
                        },
                        {"goal": "check the cabinet", "depends_on": ["st_1"]},
                    ],
                ),
                tool_call("step_blocked"),
                tool_call("step_blocked"),
                tool_call("step_blocked"),
            ]
        )
        runtime = AgentRuntime(
            executor_registry("step_blocked", "propose_viewpoints"),
            FakeWorld(),
            None,
            vlm,
            budget=8,
            max_turns=4,
            max_stagnant_turns=1,
            max_recovery_attempts=1,
        )
        log = runtime.run("fly to the doorway")

        # the blocked leg blocks the sub-task it belongs to and records its evidence
        subtask = plan_state(runtime.ctx)["subtasks"][0]
        self.assertEqual(subtask["status"], "blocked")
        self.assertEqual(subtask["reason"], "planner_failed")
        self.assertEqual(subtask["evidence"], ["trajectory:trace_9"])
        # its failure branch is the recovery the runtime offers
        advice = log.recoveries[0]["advice"]
        self.assertIn("failure branch", advice[0])
        self.assertIn("st_1", advice[0])
        self.assertIn("vp_9", " ".join(advice))
        self.assertIn("RECOVERY", vlm.seen[2]["user"])
        self.assertIn("vp_9", vlm.seen[2]["user"])

    def test_a_branch_naming_only_missing_tools_does_not_invent_a_recovery(self):
        """A failure branch that names a tool this episode does not have is never
        turned into a recovery step: the runtime quotes it from the plan (the
        revision request) and otherwise keeps the pre-P4 stall behavior."""
        vlm = RecordingScriptedVLM(
            [
                tool_call(
                    "update_plan",
                    steps=[{"goal": "fly to vp_3", "failure_branch": ["go_to_viewpoint vp_9"]}],
                ),
                tool_call("step_blocked"),
                tool_call("step_blocked"),
            ]
        )
        runtime = AgentRuntime(
            executor_registry("step_blocked"),
            FakeWorld(),
            None,
            vlm,
            budget=8,
            max_turns=6,
            max_stagnant_turns=1,
        )
        log = runtime.run("fly to the doorway")

        # the stall ends the episode the pre-P4 way: no advice, no invented call
        self.assertEqual(log.finish_reason, REASON_NO_NEW_EVIDENCE)
        self.assertEqual(log.recoveries, [])
        self.assertEqual([record.recovery for record in log.turns], ["", ""])
        self.assertEqual([record.decision for record in log.turns], ["", "path_failure"])
        for seen in vlm.seen:
            self.assertNotIn("RECOVERY", seen["user"])
        # the plan's own branch is still what the state points the model at
        from vlm_nav.tool_impl_task import subtask_progress

        self.assertIn("failure_branch=go_to_viewpoint vp_9", subtask_progress(runtime.ctx))
        self.assertIn("revision_required=path_failure", runtime.state()["subtasks"])

    def test_a_path_failure_asks_for_a_plan_revision_and_the_revision_clears_it(self):
        vlm = RecordingScriptedVLM(
            [
                tool_call(
                    "update_plan",
                    steps=[{"goal": "fly to vp_3", "failure_branch": ["propose_viewpoints"]}],
                ),
                tool_call("step_blocked"),
                tool_call(
                    "update_plan",
                    patch=[
                        {
                            "id": "st_1",
                            "status": "pending",
                            "goal": "approach the doorway from the south instead",
                            "failure_branch": ["propose_viewpoints"],
                        }
                    ],
                ),
                tool_call("step_ok"),
                tool_call("finish", reason="ok"),
            ]
        )
        runtime = AgentRuntime(
            executor_registry("step_blocked", "step_ok", "propose_viewpoints"),
            FakeWorld(),
            None,
            vlm,
            budget=8,
            max_turns=6,
            max_stagnant_turns=2,
        )
        log = runtime.run("fly to the doorway")

        self.assertEqual(log.turns[1].decision, "path_failure")
        # the model is asked, in the very next prompt, to revise that sub-plan
        self.assertIn("PLAN REVISION NEEDED", vlm.seen[2]["user"])
        self.assertIn("path_failure", vlm.seen[2]["user"])
        self.assertIn("st_1", vlm.seen[2]["user"])
        self.assertIn("revision_required", state_value(vlm.seen[2]["user"], "subtasks"))
        # ... and after the revision the request is gone
        self.assertNotIn("PLAN REVISION NEEDED", vlm.seen[3]["user"])
        subtask = plan_state(runtime.ctx)["subtasks"][0]
        self.assertEqual(subtask["goal"], "approach the doorway from the south instead")
        # the revised sub-plan ran and the executor closed it
        self.assertEqual(subtask["status"], "done")
        self.assertEqual(subtask["reason"], "done_when_met")

    def test_the_revision_request_is_dropped_once_the_subtask_recovers(self):
        vlm = RecordingScriptedVLM(
            [
                tool_call(
                    "update_plan",
                    steps=[{"goal": "fly to vp_3", "failure_branch": ["propose_viewpoints"]}],
                ),
                tool_call("step_blocked"),
                tool_call("step_ok"),
                tool_call("finish", reason="ok"),
            ]
        )
        runtime = AgentRuntime(
            executor_registry("step_blocked", "step_ok", "propose_viewpoints"),
            FakeWorld(),
            None,
            vlm,
            budget=8,
            max_turns=6,
            max_stagnant_turns=2,
        )
        log = runtime.run("fly to the doorway")

        self.assertEqual(log.turns[1].decision, "path_failure")
        self.assertIn("PLAN REVISION NEEDED", vlm.seen[2]["user"])
        self.assertIn("revision_required=path_failure", state_value(vlm.seen[2]["user"], "subtasks"))
        # the sub-task recovered on its own: the request is stale and is gone
        self.assertEqual(plan_state(runtime.ctx)["subtasks"][0]["status"], "done")
        self.assertNotIn("PLAN REVISION NEEDED", vlm.seen[3]["user"])
        self.assertNotIn("revision_required", state_value(vlm.seen[3]["user"], "subtasks"))

    def test_the_decision_nodes_of_an_episode_are_recorded(self):
        vlm = RecordingScriptedVLM(
            [
                tool_call("candidate"),
                tool_call("conflict"),
                tool_call("observe", mode="forward"),
                tool_call("finish", reason="ok"),
            ]
        )
        runtime = AgentRuntime(
            executor_registry("candidate", "conflict", "observe"),
            FakeWorld(),
            None,
            vlm,
            budget=8,
            max_turns=4,
        )
        log = runtime.run("find the red book")

        self.assertEqual(
            [record.decision for record in log.turns],
            ["new_candidate", "evidence_conflict", "", ""],
        )
        self.assertEqual(
            [entry["node"] for entry in log.decision_nodes],
            ["new_candidate", "evidence_conflict"],
        )
        # a routine turn neither moves the plan nor asks for a revision
        self.assertEqual(log.plan_transitions, [])
        self.assertNotIn("PLAN REVISION NEEDED", vlm.seen[2]["user"])

    def test_a_lost_target_is_a_decision_node_that_asks_for_a_revision(self):
        vlm = RecordingScriptedVLM(
            [
                tool_call("update_plan", steps=[{"goal": "approach the candidate"}]),
                tool_call("target_lost"),
                tool_call("finish", reason="ok"),
            ]
        )
        runtime = AgentRuntime(
            executor_registry("target_lost"), FakeWorld(), None, vlm, budget=6, max_turns=4
        )
        log = runtime.run("approach the red book")

        self.assertEqual(log.turns[1].decision, "goal_lost")
        self.assertIn("PLAN REVISION NEEDED", vlm.seen[2]["user"])
        self.assertIn("goal_lost", vlm.seen[2]["user"])

    def test_the_executor_switch_keeps_the_plan_static(self):
        vlm = RecordingScriptedVLM(
            [
                tool_call("update_plan", steps=["scan the corridor", "check the cabinet"]),
                tool_call("step_ok"),
                tool_call("step_blocked"),
                tool_call("finish", reason="ok"),
            ]
        )
        runtime = AgentRuntime(
            executor_registry("step_ok", "step_blocked"),
            FakeWorld(),
            None,
            vlm,
            budget=8,
            max_turns=5,
            max_stagnant_turns=3,
            subtask_executor=False,
        )
        log = runtime.run("find the red book")

        subtasks = plan_state(runtime.ctx)["subtasks"]
        self.assertEqual([s["status"] for s in subtasks], ["active", "pending"])
        self.assertEqual(subtasks[0]["spent"], {"tool_calls": 0, "turns": 0})
        self.assertEqual(log.decision_nodes, [])
        self.assertEqual(log.plan_transitions, [])
        for seen in vlm.seen:
            self.assertNotIn("EXECUTOR", seen["user"])
            self.assertNotIn("PLAN REVISION NEEDED", seen["user"])

    def test_an_interrupted_episode_resumes_from_the_plan_state(self):
        world = FakeWorld()
        ctx = ToolContext(
            world=world, memory=None, episode_id="placeholder",
            budget={"tool_calls_left": 4}, extra={},
        )
        registry = executor_registry("step_ok", "observe")
        first = AgentRuntime(
            registry, world, None,
            RecordingScriptedVLM(
                [
                    tool_call("update_plan", steps=["scan the corridor", "check the cabinet"]),
                    tool_call("step_ok"),
                    tool_call("observe", mode="forward"),
                    tool_call("observe", mode="forward"),
                    tool_call("observe", mode="forward"),
                ]
            ),
            budget=4, max_turns=6, tool_context=ctx,
        )
        log = first.run("find the red book")
        self.assertEqual(log.finish_reason, REASON_BUDGET_EXHAUSTED)
        self.assertEqual(
            [s["status"] for s in plan_state(ctx)["subtasks"]], ["done", "active"]
        )

        ctx.budget["tool_calls_left"] = 4  # the caller hands the new run its budget
        second_vlm = RecordingScriptedVLM([tool_call("finish", reason="ok")])
        second = AgentRuntime(
            registry, world, None, second_vlm, budget=4, max_turns=2, tool_context=ctx
        )
        resumed = second.run("find the red book")

        # the interrupted plan is the plan of the new episode
        self.assertEqual(resumed.plan_resumed["resumed_from"], log.episode_id)
        self.assertEqual(resumed.plan_resumed["plan_id"], "plan_1")
        state = state_value(second_vlm.seen[0]["user"], "subtasks")
        self.assertIn("current=st_2[active]", state)
        self.assertIn("(1 done)", state)
        self.assertIn("check the cabinet", state)
        # the budget the first run consumed is still on the sub-task
        self.assertIn("spent=2 calls", state)
        self.assertEqual(
            [s["status"] for s in plan_state(ctx)["subtasks"]], ["done", "active"]
        )

    def test_plan_resume_can_be_switched_off(self):
        world = FakeWorld()
        ctx = ToolContext(
            world=world, memory=None, episode_id="placeholder",
            budget={"tool_calls_left": 3}, extra={},
        )
        registry = executor_registry("observe")
        first = AgentRuntime(
            registry, world, None,
            RecordingScriptedVLM(
                [
                    tool_call("update_plan", steps=["scan the corridor"]),
                    tool_call("observe", mode="forward"),
                    tool_call("observe", mode="forward"),
                ]
            ),
            budget=3, max_turns=4, tool_context=ctx,
        )
        first.run("find the red book")

        ctx.budget["tool_calls_left"] = 3
        second_vlm = RecordingScriptedVLM([tool_call("finish", reason="ok")])
        second = AgentRuntime(
            registry, world, None, second_vlm, budget=3, max_turns=2,
            tool_context=ctx, plan_resume=False,
        )
        resumed = second.run("find the red book")

        self.assertEqual(resumed.plan_resumed, {})
        self.assertEqual(state_value(second_vlm.seen[0]["user"], "subtasks"), "no plan")
        self.assertIsNone(plan_state(ctx))

    def test_a_new_instruction_does_not_inherit_the_old_plan(self):
        world = FakeWorld()
        ctx = ToolContext(
            world=world, memory=None, episode_id="placeholder",
            budget={"tool_calls_left": 3}, extra={},
        )
        registry = executor_registry("observe")
        first = AgentRuntime(
            registry, world, None,
            RecordingScriptedVLM(
                [
                    tool_call("update_plan", steps=["scan the corridor"]),
                    tool_call("observe", mode="forward"),
                    tool_call("observe", mode="forward"),
                ]
            ),
            budget=3, max_turns=4, tool_context=ctx,
        )
        first.run("find the red book")

        ctx.budget["tool_calls_left"] = 3
        second_vlm = RecordingScriptedVLM([tool_call("finish", reason="ok")])
        second = AgentRuntime(
            registry, world, None, second_vlm, budget=3, max_turns=2, tool_context=ctx
        )
        resumed = second.run("count the windows")

        self.assertEqual(resumed.plan_resumed, {})
        self.assertEqual(state_value(second_vlm.seen[0]["user"], "subtasks"), "no plan")


class MechanismSwitchTest(unittest.TestCase):
    """``recovery_enabled`` / ``state_block`` switch one mechanism each.

    They are the runtime half of the ablation arms the decision-loop comparison
    drives (``--recovery``/``--state-block`` in ``scripts/run_vlm_nav_tools.py``),
    so each switch must change the loop and *only* its own layer.
    """

    STALL_TOOLS = ("probe", "observe")

    def _stalling_vlm(self, turns: int = 6) -> RecordingScriptedVLM:
        return RecordingScriptedVLM([tool_call("probe", sector="3")] * turns)

    def test_the_switches_default_to_the_current_behavior(self):
        runtime = AgentRuntime(
            make_registry(*self.STALL_TOOLS), FakeWorld(), None, RecordingScriptedVLM([])
        )
        self.assertTrue(runtime.recovery_enabled)
        self.assertTrue(runtime.state_block)

    def test_recovery_off_stops_at_the_stagnation_limit(self):
        vlm = self._stalling_vlm()
        runtime = AgentRuntime(
            make_registry(*self.STALL_TOOLS),
            FakeWorld(),
            None,
            vlm,
            budget=20,
            max_turns=6,
            max_stagnant_turns=1,
            recovery_enabled=False,
        )
        log = runtime.run("go")

        self.assertEqual(log.finish_reason, REASON_NO_NEW_EVIDENCE)
        self.assertEqual(len(log.turns), 2)  # the stall ends the loop, no recovery turn
        self.assertEqual(log.recoveries, [])
        self.assertEqual([record.recovery for record in log.turns], ["", ""])
        self.assertEqual(runtime.last_stall_action, RECOVERY_STOP)
        for seen in vlm.seen:
            self.assertNotIn("RECOVERY", seen["user"])

    def test_the_same_stall_recovers_when_the_switch_is_on(self):
        vlm = self._stalling_vlm()
        runtime = AgentRuntime(
            make_registry(*self.STALL_TOOLS),
            FakeWorld(),
            None,
            vlm,
            budget=20,
            max_turns=6,
            max_stagnant_turns=1,
        )
        log = runtime.run("go")

        self.assertTrue(log.recoveries)
        self.assertEqual(log.recoveries[0]["kind"], RECOVERY_ADVICE)
        self.assertGreater(len(log.turns), 2)
        self.assertIn("RECOVERY", vlm.seen[2]["user"])
        self.assertIn("observe", vlm.seen[2]["user"])

    def test_recovery_off_never_calls_the_recovery_hook(self):
        calls: list[dict] = []
        log = AgentRuntime(
            make_registry(*self.STALL_TOOLS),
            FakeWorld(),
            None,
            self._stalling_vlm(),
            budget=20,
            max_turns=6,
            max_stagnant_turns=1,
            recovery_enabled=False,
            recovery_hook=lambda request: calls.append(request),
        ).run("go")

        self.assertEqual(calls, [])
        self.assertEqual(log.recoveries, [])
        self.assertEqual(log.finish_reason, REASON_NO_NEW_EVIDENCE)

    def test_state_block_off_drops_the_fixed_labels_from_every_prompt(self):
        vlm = RecordingScriptedVLM(
            [tool_call("observe"), tool_call("finish", reason="done")]
        )
        runtime = AgentRuntime(
            make_registry("observe"), FakeWorld(), None, vlm, max_turns=2, state_block=False
        )
        runtime.run("go")

        self.assertEqual(len(vlm.seen), 2)
        for seen in vlm.seen:
            self.assertNotIn(STATE_MARKER, seen["user"])
            self.assertEqual(seen["context"]["state"], {})
            # the pre-existing odometry section is not the fixed block
            self.assertIn("Structured state:", seen["user"])
            self.assertIn("pose_flu=", seen["user"])
        # the block stays readable for reports even when it is not injected
        self.assertEqual(tuple(runtime.state().keys()), STATE_LABELS)

    def test_state_block_on_is_the_default(self):
        vlm = RecordingScriptedVLM(
            [tool_call("observe"), tool_call("finish", reason="done")]
        )
        AgentRuntime(make_registry("observe"), FakeWorld(), None, vlm, max_turns=2).run("go")

        for seen in vlm.seen:
            self.assertIn(STATE_MARKER, seen["user"])
            self.assertTrue(seen["context"]["state"])

    def test_the_two_switches_are_independent(self):
        vlm = self._stalling_vlm()
        log = AgentRuntime(
            make_registry(*self.STALL_TOOLS),
            FakeWorld(),
            None,
            vlm,
            budget=20,
            max_turns=6,
            max_stagnant_turns=1,
            recovery_enabled=False,
            state_block=True,
        ).run("go")

        self.assertIn(STATE_MARKER, vlm.seen[1]["user"])
        self.assertEqual(log.finish_reason, REASON_NO_NEW_EVIDENCE)
        self.assertTrue(dict(log.state))  # the episode still reports its state


class PromptContractTest(unittest.TestCase):
    """Examples, schemas and tool behavior are one interface, checked together."""

    @staticmethod
    def full_registry() -> ToolRegistry:
        from vlm_nav.tool_impl_navigation import register_navigation_tools
        from vlm_nav.tool_impl_perception import register_perception_tools
        from vlm_nav.tool_impl_task import register_task_tools

        registry = ToolRegistry()
        register_perception_tools(registry)
        register_navigation_tools(registry)
        register_task_tools(registry)
        return registry

    def test_every_prompt_example_passes_the_schema_of_its_tool(self):
        registry = self.full_registry()
        self.assertGreaterEqual(len(PROMPT_EXAMPLES), 3)
        for example in PROMPT_EXAMPLES:
            with self.subTest(example=example):
                name, args = example["tool"], example["args"]
                if name == FINISH_TOOL:
                    # the runtime answers finish itself; its contract is the reason
                    self.assertIn("reason", args)
                    continue
                self.assertIn(name, registry.names())
                self.assertEqual(
                    schema_problems(registry.get(name).input_schema, args), []
                )

    def test_the_doc_renders_the_same_examples(self):
        for example in PROMPT_EXAMPLES:
            with self.subTest(example=example):
                self.assertIn(json.dumps(example), TOOL_CALL_DOC)

    def test_every_example_call_in_the_prompts_validates(self):
        registry = self.full_registry()
        vlm = RecordingScriptedVLM(
            [tool_call("observe", mode="forward"), tool_call("finish", reason="ok")]
        )
        AgentRuntime(registry, FakeWorld(), None, vlm, max_turns=2).run("find the red book")

        texts = {
            "TOOL_CALL_DOC": TOOL_CALL_DOC,
            "DEFAULT_SYSTEM_PROMPT": DEFAULT_SYSTEM_PROMPT,
            "turn prompt": vlm.seen[0]["user"],
            "system prompt": vlm.seen[0]["system"],
        }
        found = set()
        for where, text in texts.items():
            for call in example_calls_in(text):
                found.add(call["tool"])
                with self.subTest(where=where, call=call):
                    name = call["tool"]
                    if name == FINISH_TOOL:
                        self.assertIn("reason", call["args"])
                        continue
                    self.assertIn(name, registry.names())
                    self.assertEqual(
                        schema_problems(registry.get(name).input_schema, call["args"]), []
                    )
        self.assertIn("observe", found)
        self.assertIn("finish", found)

    def test_schema_problems_reports_missing_and_mistyped_arguments(self):
        schema = {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["forward", "lookaround"]},
                "n_views": {"type": "integer"},
                "spread": {"type": ["number", "string"]},
                "frames": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["mode"],
        }
        self.assertEqual(schema_problems(schema, {"mode": "forward"}), [])
        self.assertTrue(any("mode" in p for p in schema_problems(schema, {})))
        self.assertTrue(any("mode" in p for p in schema_problems(schema, {"mode": "sideways"})))
        self.assertTrue(
            any("n_views" in p for p in schema_problems(schema, {"mode": "forward", "n_views": "8"}))
        )
        self.assertTrue(
            any("frames" in p for p in schema_problems(schema, {"mode": "forward", "frames": [1]}))
        )
        self.assertEqual(schema_problems(schema, {"mode": "forward", "spread": 3}), [])
        # a declared default means the omitted key is not missing
        tolerant = {
            "type": "object",
            "properties": {"mode": {"type": "string", "default": "forward"}},
            "required": ["mode"],
        }
        self.assertEqual(schema_problems(tolerant, {}), [])


class ExecutorAutoStepTest(unittest.TestCase):
    """Routine turns with a deterministic next step are executed by the
    executor: no model call is spent on a step the plan already spells out.

    The sources are, in order, the plan's own next step, the last result's
    ``next_action``, the current sub-task's concrete failure branch (after a
    fruitless result) and a sub-task goal written as a tool call. A decision
    node always goes back to the model, and a chain is capped at
    ``MAX_EXECUTOR_AUTO_STEPS`` (3) consecutive steps.
    """

    def test_a_routine_result_with_an_explicit_next_action_runs_without_a_model_turn(self):
        vlm = RecordingScriptedVLM(
            [
                tool_call("update_plan", steps=["scan the corridor"]),
                tool_call("next_probe"),
                tool_call("finish", reason="ok"),
            ]
        )
        runtime = AgentRuntime(
            executor_registry("next_probe", "counter"),
            FakeWorld(),
            None,
            vlm,
            budget=8,
            max_turns=5,
        )
        log = runtime.run("find the red book")

        self.assertEqual(
            log.tool_names(), ["update_plan", "next_probe", "counter", "finish"]
        )
        # three model turns, one of which executed two calls
        self.assertEqual(log.vlm_calls, 3)
        self.assertEqual([record.turn for record in log.turns], [1, 2, 2, 3])
        auto = log.executor_auto_steps
        self.assertEqual(len(auto), 1)
        self.assertEqual(auto[0]["tool"], "counter")
        self.assertEqual(auto[0]["origin"], "next_action")
        self.assertTrue(log.turns[2].auto)
        self.assertEqual(log.turns[2].auto_origin, "next_action")
        self.assertFalse(log.turns[1].auto)

    def test_a_fruitless_result_does_not_auto_repeat_itself(self):
        """An empty/failed result naming its own call is the P4 self-loop.

        The executor used to accept the result's ``next_action`` even when it
        was exactly the call that had just returned empty, so one model turn
        could spend the rest of the auto-step budget on the same dead call.
        """
        vlm = RecordingScriptedVLM(
            [
                tool_call("update_plan", steps=["scan the corridor"]),
                tool_call("loop_probe"),
                tool_call("finish", reason="ok"),
            ]
        )
        log = AgentRuntime(
            executor_registry("loop_probe", "counter"),
            FakeWorld(),
            None,
            vlm,
            budget=8,
            max_turns=5,
        ).run("find the red book")

        self.assertEqual(
            log.tool_names(), ["update_plan", "loop_probe", "finish"]
        )
        self.assertEqual(log.executor_auto_steps, [])
        self.assertEqual(log.vlm_calls, 3)
        # the dead call is not offered back to the model either
        needed = state_value(vlm.seen[2]["user"], "next_info_needed")
        self.assertNotIn("loop_probe", needed)

    def test_a_plan_step_naming_the_fruitless_call_is_not_repeated(self):
        """The executor's own source is filtered too, not only result actions.

        The plan's ``next_step`` ran once (a legal first step); after it came
        back empty, naming it again is the same self-loop and must not spend
        the remaining auto-step budget.
        """
        vlm = RecordingScriptedVLM(
            [
                tool_call(
                    "update_plan",
                    steps=[
                        {
                            "goal": "scan the corridor",
                            "next_step": {"tool": "empty_probe", "args": {}},
                        }
                    ],
                ),
                tool_call("finish", reason="ok"),
            ]
        )
        log = AgentRuntime(
            executor_registry("empty_probe"),
            FakeWorld(),
            None,
            vlm,
            budget=8,
            max_turns=4,
        ).run("find the red book")

        self.assertEqual(
            log.tool_names(), ["update_plan", "empty_probe", "finish"]
        )
        self.assertEqual(len(log.executor_auto_steps), 1)
        self.assertEqual(log.executor_auto_steps[0]["origin"], "subtask_next_step")
        self.assertEqual(log.vlm_calls, 2)

    def test_a_decision_node_still_goes_back_to_the_model(self):
        vlm = RecordingScriptedVLM(
            [
                tool_call("update_plan", steps=["scan the corridor"]),
                tool_call("candidate_next"),
                tool_call("counter"),
                tool_call("finish", reason="ok"),
            ]
        )
        log = AgentRuntime(
            executor_registry("candidate_next", "counter"),
            FakeWorld(),
            None,
            vlm,
            budget=8,
            max_turns=6,
        ).run("find the red book")

        # the new candidate is a decision node: its next_action is not executed
        self.assertEqual(log.turns[1].decision, "new_candidate")
        self.assertEqual(
            log.tool_names(),
            ["update_plan", "candidate_next", "counter", "finish"],
        )
        self.assertEqual(log.vlm_calls, 4)
        self.assertEqual(log.executor_auto_steps, [])

    def test_the_auto_step_chain_is_capped_at_three_consecutive_steps(self):
        vlm = RecordingScriptedVLM(
            [
                tool_call("update_plan", steps=["scan the corridor"]),
                tool_call("chain_tick"),
                tool_call("finish", reason="ok"),
            ]
        )
        runtime = AgentRuntime(
            executor_registry("chain_tick"), FakeWorld(), None, vlm,
            budget=12, max_turns=6,
        )
        log = runtime.run("find the red book")

        self.assertEqual(runtime.max_executor_auto_steps, 3)
        self.assertEqual(
            log.tool_names(),
            ["update_plan", "chain_tick", "chain_tick", "chain_tick",
             "chain_tick", "finish"],
        )
        self.assertEqual(len(log.executor_auto_steps), 3)
        self.assertEqual(log.vlm_calls, 3)
        # every automatic step ran the model's own continuation
        self.assertTrue(all(step["origin"] == "next_action" for step in log.executor_auto_steps))

    def test_the_switch_keeps_the_model_in_the_loop(self):
        vlm = RecordingScriptedVLM(
            [
                tool_call("update_plan", steps=["scan the corridor"]),
                tool_call("next_probe"),
                tool_call("counter"),
                tool_call("finish", reason="ok"),
            ]
        )
        runtime = AgentRuntime(
            executor_registry("next_probe", "counter"),
            FakeWorld(),
            None,
            vlm,
            budget=8,
            max_turns=6,
            executor_auto_steps=False,
        )
        log = runtime.run("find the red book")

        self.assertFalse(runtime.executor_auto_steps)
        self.assertEqual(log.executor_auto_steps, [])
        self.assertEqual(
            log.tool_names(), ["update_plan", "next_probe", "counter", "finish"]
        )
        self.assertEqual(log.vlm_calls, 4)

    def test_a_concrete_failure_branch_runs_after_a_fruitless_result(self):
        vlm = RecordingScriptedVLM(
            [
                tool_call(
                    "update_plan",
                    steps=[{"goal": "find the door", "failure_branch": ["counter"]}],
                ),
                tool_call("empty_probe"),
                tool_call("finish", reason="ok"),
            ]
        )
        log = AgentRuntime(
            executor_registry("empty_probe", "counter"),
            FakeWorld(),
            None,
            vlm,
            budget=8,
            max_turns=5,
        ).run("find the red book")

        self.assertEqual(
            log.tool_names(), ["update_plan", "empty_probe", "counter", "finish"]
        )
        self.assertEqual(log.vlm_calls, 3)
        self.assertEqual(len(log.executor_auto_steps), 1)
        self.assertEqual(log.executor_auto_steps[0]["origin"], "failure_branch")

    def test_a_subtask_goal_written_as_a_tool_call_is_executed(self):
        vlm = RecordingScriptedVLM(
            [
                tool_call("update_plan", steps=["counter"]),
                tool_call("finish", reason="ok"),
            ]
        )
        log = AgentRuntime(
            executor_registry("counter"), FakeWorld(), None, vlm,
            budget=6, max_turns=4,
        ).run("find the red book")

        self.assertEqual(log.tool_names(), ["update_plan", "counter", "finish"])
        self.assertEqual(log.vlm_calls, 2)
        self.assertEqual(log.executor_auto_steps[0]["origin"], "subtask_goal")

    def test_a_path_failure_is_never_auto_executed(self):
        vlm = RecordingScriptedVLM(
            [
                tool_call(
                    "update_plan",
                    steps=[
                        {
                            "goal": "fly to vp_3",
                            "failure_branch": ["propose_viewpoints"],
                        }
                    ],
                ),
                tool_call("step_blocked"),
                tool_call("propose_viewpoints"),
                tool_call("finish", reason="ok"),
            ]
        )
        log = AgentRuntime(
            executor_registry("step_blocked", "propose_viewpoints"),
            FakeWorld(),
            None,
            vlm,
            budget=8,
            max_turns=6,
        ).run("fly to the doorway")

        self.assertEqual(log.turns[1].decision, "path_failure")
        self.assertEqual(log.executor_auto_steps, [])
        self.assertEqual(log.vlm_calls, 4)

    def test_a_goal_with_json_arguments_is_executed_with_those_arguments(self):
        vlm = RecordingScriptedVLM(
            [
                tool_call(
                    "update_plan",
                    steps=[{"goal": 'counter {"where": "north"}'}],
                ),
                tool_call("finish", reason="ok"),
            ]
        )
        log = AgentRuntime(
            executor_registry("counter"), FakeWorld(), None, vlm,
            budget=8, max_turns=4,
        ).run("find the red book")

        self.assertEqual(log.tool_names(), ["update_plan", "counter", "finish"])
        auto = log.executor_auto_steps[0]
        self.assertEqual(auto["origin"], "subtask_goal")
        self.assertEqual(auto["args"], {"where": "north"})

    def test_a_failure_branch_with_json_arguments_is_executed(self):
        vlm = RecordingScriptedVLM(
            [
                tool_call(
                    "update_plan",
                    steps=[
                        {
                            "goal": "find the door",
                            "failure_branch": ['counter {"where": "south"}'],
                        }
                    ],
                ),
                tool_call("empty_probe"),
                tool_call("finish", reason="ok"),
            ]
        )
        log = AgentRuntime(
            executor_registry("empty_probe", "counter"), FakeWorld(), None, vlm,
            budget=8, max_turns=5,
        ).run("find the red book")

        self.assertEqual(
            log.tool_names(), ["update_plan", "empty_probe", "counter", "finish"]
        )
        auto = log.executor_auto_steps[0]
        self.assertEqual(auto["origin"], "failure_branch")
        self.assertEqual(auto["args"], {"where": "south"})

    def test_a_subtask_next_step_runs_without_a_model_turn(self):
        vlm = RecordingScriptedVLM(
            [
                tool_call(
                    "update_plan",
                    steps=[
                        {
                            "goal": "scan the corridor",
                            "next_step": {"tool": "counter", "args": {"where": "east"}},
                        }
                    ],
                ),
                tool_call("finish", reason="ok"),
            ]
        )
        log = AgentRuntime(
            executor_registry("counter"), FakeWorld(), None, vlm,
            budget=8, max_turns=4, max_executor_auto_steps=1,
        ).run("find the red book")

        self.assertEqual(log.tool_names(), ["update_plan", "counter", "finish"])
        auto = log.executor_auto_steps[0]
        self.assertEqual(auto["origin"], "subtask_next_step")
        self.assertEqual(auto["args"], {"where": "east"})

    def test_a_prose_goal_is_not_executed_as_a_call(self):
        vlm = RecordingScriptedVLM(
            [
                tool_call("update_plan", steps=["counter the north side carefully"]),
                tool_call("finish", reason="ok"),
            ]
        )
        log = AgentRuntime(
            executor_registry("counter"), FakeWorld(), None, vlm,
            budget=8, max_turns=4,
        ).run("find the red book")

        self.assertEqual(log.tool_names(), ["update_plan", "finish"])
        self.assertEqual(log.executor_auto_steps, [])

    def test_the_plan_prompts_ask_for_concrete_next_steps(self):
        vlm = RecordingScriptedVLM(
            [
                tool_call("update_plan", steps=["scan the corridor"]),
                tool_call("step_blocked"),
                tool_call("propose_viewpoints"),
                tool_call("finish", reason="ok"),
            ]
        )
        AgentRuntime(
            executor_registry("step_blocked", "propose_viewpoints"),
            FakeWorld(), None, vlm, budget=8, max_turns=6,
            max_stagnant_turns=2,
        ).run("fly to the doorway")

        # the STATE section after the plan teaches the deterministic source ...
        state_prompt = vlm.seen[1]["user"]
        self.assertIn(PLAN_CALL_HINT, state_prompt)
        self.assertIn("next_step", state_prompt)
        self.assertIn("failure_branch", state_prompt)
        # ... and the revision request asks for the same concrete form
        revision = vlm.seen[2]["user"]
        self.assertIn("PLAN REVISION NEEDED", revision)
        self.assertIn("next_step", revision)
        self.assertIn("failure_branch", revision)


class _PlainRecordingVLM:
    """A backend without ``decide_with_tools``: the runtime puts the catalogue
    and the protocol doc into the system prompt itself."""

    def __init__(self):
        self.seen: list[dict] = []

    def decide(self, frames, system_prompt, user_prompt, context=None):
        self.seen.append({"system": system_prompt, "user": user_prompt})
        return {"tool": "finish", "args": {"reason": "ok"}}


class _LongCatalogRegistry:
    """A registry whose state-aware catalogue is one long text."""

    def __init__(self, text: str):
        self.text = text
        self._names = ["tool_0"]

    def specs_text(self) -> str:
        return self.text

    def names(self) -> list[str]:
        return list(self._names)

    def get(self, name):
        raise KeyError(name)


class PromptBudgetTest(unittest.TestCase):
    """The system prompt is budgeted catalogue-first: the state-aware listing is
    never cut, the protocol/examples are truncated (and say so) instead."""

    def _catalogue(self) -> str:
        lines = ["STATE=start: a test catalogue"]
        for index in range(150):
            lines.append(f"- tool_{index}: description {index} args={{\"type\":\"object\"}}")
        return "\n".join(lines)

    def test_the_default_budget_keeps_the_catalogue_and_truncates_the_doc(self):
        catalogue = self._catalogue()
        vlm = _PlainRecordingVLM()
        runtime = AgentRuntime(
            _LongCatalogRegistry(catalogue),
            FakeWorld(),
            None,
            vlm,
            tool_call_doc="D" * 20000,
        )
        runtime.run("go")

        system = vlm.seen[0]["system"]
        self.assertIn(catalogue, system)
        self.assertNotIn("[truncated", system)
        self.assertLessEqual(len(system), runtime.max_system_chars)
        self.assertIn("protocol/example doc truncated", system)
        budget = runtime.prompt_budget
        self.assertEqual(budget["system_chars"], runtime.max_system_chars)
        self.assertTrue(budget["catalogue_complete"])
        self.assertFalse(budget["catalogue_truncated"])
        self.assertTrue(budget["doc_truncated"])
        self.assertEqual(budget["catalogue_chars"], len(catalogue))

    def test_a_tiny_explicit_cap_still_truncates_the_catalogue_last(self):
        catalogue = self._catalogue()
        runtime = AgentRuntime(
            _LongCatalogRegistry(catalogue),
            FakeWorld(),
            None,
            _PlainRecordingVLM(),
            max_system_chars=500,
            tool_call_doc="D" * 4000,
        )
        system = runtime._system_prompt(True)

        self.assertLessEqual(len(system), 500)
        self.assertTrue(runtime.prompt_budget["catalogue_truncated"])
        self.assertFalse(runtime.prompt_budget["catalogue_complete"])


def example_calls_in(text: str) -> list[dict]:
    """Every literal ``{"tool": ...}`` call a prompt text contains.

    The ``{"tool": "<name>", "args": {...}}`` placeholders of the protocol are
    not valid JSON and are skipped, so what is validated is exactly the set of
    example calls a model can copy out of the prompt.
    """
    decoder = json.JSONDecoder()
    calls: list[dict] = []
    for index, _ in enumerate(text):
        if not text.startswith('{"tool"', index):
            continue
        try:
            payload, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "tool" in payload:
            args = payload.get("args")
            calls.append({"tool": str(payload["tool"]), "args": dict(args or {})})
    return calls


if __name__ == "__main__":
    unittest.main()
