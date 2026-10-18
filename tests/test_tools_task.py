"""Tests for the conditional task-plan tools and the repeated-call cache
(``vlm_nav/tool_impl_task.py`` + the ``ToolRegistry.call`` cache in
``vlm_nav/tools.py``, goal.md section 5).

Everything runs on CPU against tool doubles, fake worlds and scripted mock VLM
backends - no model, no renderer.
"""

import json
import unittest

import numpy as np
import torch

from vlm_nav.runtime import (
    BAD_OUTCOMES,
    REASON_FINISH,
    REASON_NO_NEW_EVIDENCE,
    TASK_SIGNAL_KEYS as RUNTIME_TASK_SIGNAL_KEYS,
    AgentRuntime,
)
from vlm_nav.tool_impl_task import (
    EXECUTOR_SKIP_TOOLS,
    FAILURE_OUTCOMES,
    MAX_PLAN_STEPS,
    MAX_STEP_CHARS,
    MAX_SUBTASK_EVIDENCE,
    PLAN_KEY,
    SUBTASK_ACTIVE,
    SUBTASK_BLOCKED,
    SUBTASK_CLAIMED_DONE,
    SUBTASK_DONE,
    SUBTASK_FAILED,
    SUBTASK_FIELDS,
    SUBTASK_PENDING,
    SUBTASK_STATUSES,
    TASK_SIGNAL_KEYS,
    current_subtask,
    evidence_verdict,
    executor_step,
    get_plan,
    normalize_steps,
    normalize_subtasks,
    parse_call_text,
    plan_progress,
    plan_state,
    plan_subtasks,
    plan_text,
    register_task_tools,
    resume_plan,
    subtask_failure_branch,
    subtask_progress,
)
from vlm_nav.tools import (
    REPEAT_HINT,
    STATUS_BLOCKED,
    STATUS_CACHED,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_OK,
    TOOL_CACHE_KEY,
    EvidenceRef,
    ToolCallCache,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    clear_tool_cache,
    tool_cache,
    tool_cache_stats,
)
from vlm_nav.vlm import MockVLM


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


class FakeWorld:
    """Minimal world for the runtime integration tests."""

    def __init__(self):
        self.position_flu = np.zeros(3)
        self.yaw_rad = 0.0
        self.trajectory = [(0.0, 0.0, 0.0)]

    def render(self):
        return torch.full((3, 4, 4), 0.5)


class RecordingVLM(MockVLM):
    """``MockVLM(mode='tools')`` plus prompt/frame recording."""

    def __init__(self, tool_script):
        super().__init__(mode="tools", tool_script=tool_script)
        self.seen = []

    def decide_with_tools(self, frames, system_prompt, user_prompt, tools_text="", context=None):
        self.seen.append(
            {
                "n_images": len(frames),
                "system": system_prompt,
                "user": user_prompt,
                "tools_text": tools_text,
                "context": dict(context or {}),
            }
        )
        return super().decide_with_tools(frames, system_prompt, user_prompt, tools_text, context)


class ToolDouble:
    """A registered tool: counts handler executions, replies a fixed payload."""

    def __init__(
        self,
        name="read",
        *,
        data=None,
        evidence=None,
        status=STATUS_OK,
        cacheable=True,
        side_effects=False,
    ):
        self.calls = 0
        self.status = status
        self.spec = ToolSpec(
            name=name,
            description=f"{name} test double",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=self._handle,
            side_effects=side_effects,
            cacheable=cacheable,
        )
        self._data = data
        self._evidence = list(evidence or [])

    def _handle(self, ctx, args):
        self.calls += 1
        data = {"value": self.calls} if self._data is None else dict(self._data)
        return ToolResult(self.status, data, [EvidenceRef(*ref) for ref in self._evidence])


def make_registry(*doubles):
    registry = ToolRegistry()
    for double in doubles:
        registry.register(double.spec)
    return registry


def plan_registry():
    registry = ToolRegistry()
    register_task_tools(registry)
    return registry


def make_context(episode_id="ep_task", world=None, budget=20, **extra):
    return ToolContext(
        world=world,
        episode_id=episode_id,
        budget={"tool_calls_left": budget},
        extra=dict(extra),
    )


# ----------------------------------------------------------------------
# plan tools
# ----------------------------------------------------------------------


class ParseCallTextTest(unittest.TestCase):
    """A plan step may spell one concrete call; prose is never executed."""

    def test_a_bare_tool_name_is_a_call_with_no_args(self):
        self.assertEqual(parse_call_text("observe"), ("observe", {}))
        self.assertEqual(parse_call_text("  observe \n"), ("observe", {}))

    def test_a_tool_name_followed_by_json_arguments_is_a_call(self):
        self.assertEqual(
            parse_call_text('observe {"mode": "forward"}'),
            ("observe", {"mode": "forward"}),
        )
        self.assertEqual(
            parse_call_text('go_to_viewpoint({"viewpoint_id": "vp_9"})'),
            ("go_to_viewpoint", {"viewpoint_id": "vp_9"}),
        )
        self.assertEqual(
            parse_call_text('go_to_viewpoint({"viewpoint_id": "vp_9"}).'),
            ("go_to_viewpoint", {"viewpoint_id": "vp_9"}),
        )

    def test_prose_and_malformed_arguments_are_not_calls(self):
        for text in (
            "observe the corridor carefully",
            "counter the north side without stopping",
            "先找到走廊里的木门（地标）",
            "go_to_viewpoint not-json",
            'go_to_viewpoint({"viewpoint_id": "vp_9"} and then scan)',
            'go_to_viewpoint(["vp_9"])',
            "",
        ):
            self.assertIsNone(parse_call_text(text), text)
        self.assertIsNone(parse_call_text({"tool": "observe"}))


class PlanRegistrationTest(unittest.TestCase):
    def test_registers_update_plan_and_get_plan(self):
        registry = plan_registry()
        self.assertEqual(registry.names(), ["get_plan", "update_plan"])
        for name in ("update_plan", "get_plan"):
            text = registry.specs_text()
            self.assertIn(name, text)
        # P4: a step is a sub-task record, and a call may also patch one of them
        # (or only advance the progress), so ``steps`` is declared but no longer
        # the single required argument
        schema = registry.get("update_plan").input_schema
        self.assertIn("steps", schema["properties"])
        self.assertIn("patch", schema["properties"])
        for token in ("done_when", "failure_branch", "budget", "sub-task"):
            self.assertIn(token, registry.get("update_plan").description)

    def test_flags_keep_plan_tools_out_of_the_cache(self):
        registry = plan_registry()
        # the plan is episode state: updating it is a mutation (serialized,
        # never cached) and reading it must always answer the current revision
        self.assertTrue(registry.get("update_plan").side_effects)
        self.assertFalse(registry.get("get_plan").side_effects)
        self.assertFalse(registry.get("get_plan").cacheable)

    def test_probe_aliases_point_at_the_same_entry_point(self):
        import vlm_nav.tool_impl_task as module

        self.assertIs(module.register_tools, register_task_tools)
        self.assertIs(module.register_all, register_task_tools)


class UpdatePlanTest(unittest.TestCase):
    def test_stores_a_structured_plan(self):
        ctx = make_context()
        steps = [
            "go to doorway P3 and check the left side",
            "if a candidate appears, compare its attributes",
            "if there is no target, move on to P4",
            "if the path is blocked, propose alternative viewpoints",
        ]
        result = plan_registry().call("update_plan", {"steps": steps}, ctx)
        self.assertEqual(result.status, STATUS_OK)
        self.assertEqual(result.data["plan_id"], "plan_1")
        self.assertEqual(result.data["revision"], 1)
        self.assertEqual(result.data["steps"], steps)
        self.assertEqual(result.data["n_steps"], 4)
        self.assertTrue(result.data["changed"])
        self.assertEqual(result.evidence[0].kind, "memory")
        self.assertEqual(result.evidence[0].ref, "plan_1")
        # the raw record lives in the context, readable by runtime components
        record = plan_state(ctx)
        self.assertIs(ctx.extra[PLAN_KEY], record)
        self.assertEqual(record["steps"], steps)
        self.assertEqual(record["episode_id"], "ep_task")
        self.assertIn("plan_1", plan_text(ctx))
        self.assertIn("go to doorway P3", plan_text(ctx))

    def test_the_step_hint_teaches_the_intent_tool_vocabulary(self):
        """The plan examples are what a model copies: they must name the movement
        tools that exist (approach_candidate / go_to_viewpoint), not the
        deprecated ``navigate``."""
        ctx = make_context()
        result = plan_registry().call("update_plan", {"steps": []}, ctx)
        self.assertEqual(result.status, STATUS_FAILED)
        self.assertEqual(result.data["error"], "empty_plan")
        hint = result.data["hint"]
        self.assertIn("go_to_viewpoint", hint)
        self.assertIn("approach_candidate", hint)
        self.assertIn("ground_target", hint)
        self.assertNotIn("navigate", hint)

    def test_identical_update_keeps_the_revision(self):
        registry = plan_registry()
        ctx = make_context()
        steps = ["observe forward", "if clear, navigate to vp_2"]
        first = registry.call("update_plan", {"steps": steps}, ctx)
        again = registry.call("update_plan", {"steps": steps}, ctx)
        self.assertEqual(first.data["revision"], 1)
        self.assertTrue(first.data["changed"])
        self.assertEqual(again.data["revision"], 1)
        self.assertFalse(again.data["changed"])
        self.assertEqual(again.data["plan_id"], "plan_1")

    def test_revised_plan_bumps_the_revision_and_records_the_note(self):
        registry = plan_registry()
        ctx = make_context()
        registry.call("update_plan", {"steps": ["scan left"]}, ctx)
        result = registry.call(
            "update_plan",
            {"steps": ["scan left", "if no target, scan right"], "reason": "left side clear"},
            ctx,
        )
        self.assertEqual(result.data["revision"], 2)
        self.assertTrue(result.data["changed"])
        self.assertEqual(result.data["n_steps"], 2)
        self.assertEqual(result.data["note"], "left side clear")
        self.assertEqual(registry.call("get_plan", {}, ctx).data["revision"], 2)

    def test_a_single_string_is_split_into_conditional_steps(self):
        # the goal.md example plan is written as one semicolon-joined sentence
        ctx = make_context()
        plan = (
            "go to doorway P3, check the left side；"
            "if a candidate appears, compare attributes;\n"
            "if there is no target, move on to P4"
        )
        result = plan_registry().call("update_plan", {"steps": plan}, ctx)
        self.assertEqual(
            result.data["steps"],
            [
                "go to doorway P3, check the left side",
                "if a candidate appears, compare attributes",
                "if there is no target, move on to P4",
            ],
        )

    def test_step_objects_and_aliases_are_normalized(self):
        ctx = make_context()
        result = plan_registry().call(
            "update_plan",
            {"plan": [{"text": "go to P3"}, {"step": "check left"}, {"heading": "north"}]},
            ctx,
        )
        self.assertEqual(result.status, STATUS_OK)
        self.assertEqual(result.data["steps"], ["go to P3", "check left", "heading=north"])

    def test_plan_is_capped_in_step_count_and_length(self):
        ctx = make_context()
        steps = ["x" * 500] + [f"step {i}" for i in range(MAX_PLAN_STEPS + 5)]
        result = plan_registry().call("update_plan", {"steps": steps}, ctx)
        self.assertEqual(result.data["n_steps"], MAX_PLAN_STEPS)
        self.assertEqual(len(result.data["steps"][0]), MAX_STEP_CHARS)

    def test_invalid_steps_are_rejected_and_keep_the_old_plan(self):
        registry = plan_registry()
        ctx = make_context()
        registry.call("update_plan", {"steps": ["keep me"]}, ctx)
        for args, reason in (
            ({}, "missing_steps"),
            ({"steps": []}, "empty_plan"),
            ({"steps": "   ; \n "}, "empty_plan"),
            ({"steps": 42}, "invalid_steps"),
            ({"steps": None}, "missing_steps"),
        ):
            result = registry.call("update_plan", args, ctx)
            self.assertEqual(result.status, STATUS_FAILED, args)
            self.assertEqual(result.data["error"], reason, args)
            self.assertIn("hint", result.data)
        self.assertEqual(plan_state(ctx)["revision"], 1)
        self.assertEqual(plan_state(ctx)["steps"], ["keep me"])

    def test_normalize_steps_reports_errors_directly(self):
        self.assertEqual(normalize_steps({"steps": ["a", "b"]}), (["a", "b"], None))
        self.assertEqual(normalize_steps({})[1], "missing_steps")
        self.assertTrue(normalize_steps({"steps": []})[0] is None)

    def test_plan_is_episode_scoped(self):
        registry = plan_registry()
        first = make_context(episode_id="ep_1")
        registry.call("update_plan", {"steps": ["episode one plan"]}, first)
        # the same context (and extra dict) reused for another episode reports
        # no plan for the new episode ...
        second = ToolContext(
            world=None,
            episode_id="ep_2",
            budget={"tool_calls_left": 20},
            extra=first.extra,
        )
        self.assertIsNone(plan_state(second))
        self.assertEqual(registry.call("get_plan", {}, second).status, STATUS_EMPTY)
        # ... and starts a fresh plan instead of editing the previous one
        result = registry.call("update_plan", {"steps": ["episode two plan"]}, second)
        self.assertEqual(result.data["plan_id"], "plan_2")
        self.assertEqual(result.data["revision"], 1)
        self.assertIsNone(plan_state(first))  # the record now belongs to ep_2
        self.assertEqual(plan_state(second)["steps"], ["episode two plan"])


class GetPlanTest(unittest.TestCase):
    def test_empty_before_any_update(self):
        result = plan_registry().call("get_plan", {}, make_context())
        self.assertEqual(result.status, STATUS_EMPTY)
        self.assertEqual(result.data["reason"], "no_plan")
        self.assertIn("update_plan", result.data["hint"])
        self.assertEqual(result.evidence, [])

    def test_returns_the_current_revision_after_updates(self):
        registry = plan_registry()
        ctx = make_context()
        registry.call("update_plan", {"steps": ["first"]}, ctx)
        registry.call("update_plan", {"steps": ["first", "second"]}, ctx)
        result = registry.call("get_plan", {}, ctx)
        self.assertEqual(result.status, STATUS_OK)
        self.assertEqual(result.data["steps"], ["first", "second"])
        self.assertEqual(result.data["revision"], 2)
        self.assertEqual(result.data["plan_id"], "plan_1")
        self.assertEqual(result.evidence[0].ref, "plan_1")

    def test_repeated_get_plan_is_never_replayed_from_the_cache(self):
        registry = plan_registry()
        ctx = make_context()
        registry.call("update_plan", {"steps": ["first"]}, ctx)
        registry.call("update_plan", {"steps": ["first", "second"]}, ctx)
        for _ in range(2):
            result = registry.call("get_plan", {}, ctx)
            self.assertEqual(result.status, STATUS_OK)
            self.assertEqual(result.data["revision"], 2)
            self.assertEqual(result.data["steps"], ["first", "second"])
        # update_plan itself is never served from the cache either
        registry.call("update_plan", {"steps": ["first"]}, ctx)
        repeat = registry.call("update_plan", {"steps": ["first"]}, ctx)
        self.assertEqual(repeat.status, STATUS_OK)
        self.assertEqual(repeat.data["revision"], 3)
        self.assertFalse(repeat.data["changed"])

    def test_plan_text_without_a_plan(self):
        self.assertEqual(plan_text(make_context()), "no plan")


# ----------------------------------------------------------------------
# repeated-call cache
# ----------------------------------------------------------------------


class ToolCallCacheTest(unittest.TestCase):
    def test_repeated_read_call_is_served_from_the_cache(self):
        double = ToolDouble("look", data={"sector": "hallway", "clear": True})
        registry = make_registry(double)
        ctx = make_context()
        first = registry.call("look", {"sector": "north"}, ctx)
        hit = registry.call("look", {"sector": "north"}, ctx)
        self.assertEqual(first.status, STATUS_OK)
        self.assertEqual(hit.status, STATUS_CACHED)
        self.assertEqual(double.calls, 1)
        # the original structured fields survive, plus the duplicate hint
        self.assertEqual(hit.data["sector"], "hallway")
        self.assertTrue(hit.data["clear"])
        self.assertEqual(hit.data["hint"], REPEAT_HINT)
        self.assertEqual(hit.data["cached_from_turn"], 1)
        self.assertEqual([e.to_dict() for e in hit.evidence], [e.to_dict() for e in first.evidence])

    def test_tool_hint_is_kept_on_a_hit(self):
        double = ToolDouble("look", data={"reason": "blocked", "hint": "try vp_4"})
        registry = make_registry(double)
        ctx = make_context()
        registry.call("look", {}, ctx)
        hit = registry.call("look", {}, ctx)
        self.assertEqual(hit.status, STATUS_CACHED)
        self.assertEqual(hit.data["hint"], REPEAT_HINT)
        self.assertEqual(hit.data["tool_hint"], "try vp_4")

    def test_different_args_do_not_hit(self):
        double = ToolDouble("look")
        registry = make_registry(double)
        ctx = make_context()
        registry.call("look", {"sector": "north"}, ctx)
        other = registry.call("look", {"sector": "south"}, ctx)
        self.assertEqual(other.status, STATUS_OK)
        self.assertEqual(double.calls, 2)

    def test_arg_order_does_not_matter(self):
        double = ToolDouble("look")
        registry = make_registry(double)
        ctx = make_context()
        registry.call("look", {"sector": "north", "depth": 2}, ctx)
        hit = registry.call("look", {"depth": 2, "sector": "north"}, ctx)
        self.assertEqual(hit.status, STATUS_CACHED)
        self.assertEqual(double.calls, 1)

    def test_side_effects_tool_is_never_cached_and_clears_the_cache(self):
        read = ToolDouble("look")
        move = ToolDouble("fly", side_effects=True)
        registry = make_registry(read, move)
        ctx = make_context()
        registry.call("look", {}, ctx)
        self.assertEqual(registry.call("look", {}, ctx).status, STATUS_CACHED)
        # a state-changing call runs every time and invalidates earlier reads
        self.assertEqual(registry.call("fly", {"to": "vp_1"}, ctx).status, STATUS_OK)
        self.assertEqual(registry.call("fly", {"to": "vp_1"}, ctx).status, STATUS_OK)
        self.assertEqual(move.calls, 2)
        self.assertEqual(registry.call("look", {}, ctx).status, STATUS_OK)
        self.assertEqual(read.calls, 2)

    def test_new_evidence_clears_earlier_read_answers(self):
        # the M2 case: an empty query must not stay cached after an observation
        query = ToolDouble("query_memory", data={"count": 0})
        look = ToolDouble("observe", evidence=[("observation", "obs_1", "new frame")])
        registry = make_registry(query, look)
        ctx = make_context()
        self.assertEqual(registry.call("query_memory", {"q": "red box"}, ctx).data["count"], 0)
        self.assertEqual(
            registry.call("query_memory", {"q": "red box"}, ctx).status, STATUS_CACHED
        )
        registry.call("observe", {}, ctx)
        after = registry.call("query_memory", {"q": "red box"}, ctx)
        self.assertEqual(after.status, STATUS_OK)
        self.assertEqual(query.calls, 2)

    def test_failed_results_are_not_cached(self):
        double = ToolDouble("look", status=STATUS_FAILED)
        registry = make_registry(double)
        ctx = make_context()
        first = registry.call("look", {}, ctx)
        second = registry.call("look", {}, ctx)
        self.assertEqual((first.status, second.status), (STATUS_FAILED, STATUS_FAILED))
        self.assertEqual(double.calls, 2)
        self.assertEqual(tool_cache_stats(ctx)["entries"], 0)

    def test_budget_exhausted_beats_a_cache_hit(self):
        double = ToolDouble("look")
        registry = make_registry(double)
        ctx = make_context()
        registry.call("look", {}, ctx)
        ctx.budget["tool_calls_left"] = 0
        drained = registry.call("look", {}, ctx)
        self.assertEqual(drained.status, STATUS_FAILED)
        self.assertEqual(drained.data["reason"], "budget_exhausted")
        self.assertEqual(double.calls, 1)

    def test_a_hit_consumes_budget_and_is_logged(self):
        double = ToolDouble("look")
        registry = make_registry(double)
        ctx = make_context(budget=5)
        registry.call("look", {"a": 1}, ctx)
        registry.call("look", {"a": 1}, ctx)
        self.assertEqual(ctx.budget["tool_calls_left"], 3)
        self.assertEqual([entry["status"] for entry in ctx.log], [STATUS_OK, STATUS_CACHED])
        for entry in ctx.log:
            self.assertEqual(entry["tool"], "look")
            self.assertEqual(entry["args"], {"a": 1})
            self.assertGreaterEqual(entry["ms"], 0.0)
            self.assertIsInstance(entry["evidence"], list)

    def test_non_cacheable_spec_is_never_cached(self):
        double = ToolDouble("live", cacheable=False)
        registry = make_registry(double)
        ctx = make_context()
        self.assertEqual(registry.call("live", {}, ctx).status, STATUS_OK)
        self.assertEqual(registry.call("live", {}, ctx).status, STATUS_OK)
        self.assertEqual(double.calls, 2)
        self.assertEqual(tool_cache_stats(ctx)["entries"], 0)

    def test_explicit_limit_and_missing_extra_are_handled(self):
        double = ToolDouble("look")
        registry = make_registry(double)
        ctx = ToolContext(world=None, budget={"tool_calls_left": 5})  # extra defaults to {}
        self.assertEqual(tool_cache(ctx, limit=1).limit, 1)
        ctx.extra = None  # a hand-built context may leave extra unset
        self.assertEqual(registry.call("look", {}, ctx).status, STATUS_OK)
        self.assertEqual(registry.call("look", {}, ctx).status, STATUS_CACHED)
        self.assertIsInstance(ctx.extra[TOOL_CACHE_KEY], ToolCallCache)

    def test_cache_limit_evicts_the_oldest_entry(self):
        double = ToolDouble("look")
        registry = make_registry(double)
        ctx = make_context(tool_cache_limit=2)
        for key in ("a", "b"):
            registry.call("look", {"sector": key}, ctx)
        registry.call("look", {"sector": "c"}, ctx)
        stats = tool_cache_stats(ctx)
        self.assertEqual((stats["entries"], stats["limit"], stats["evictions"]), (2, 2, 1))
        # "a" was evicted: it re-executes, "c" is still cached
        self.assertEqual(registry.call("look", {"sector": "c"}, ctx).status, STATUS_CACHED)
        self.assertEqual(registry.call("look", {"sector": "a"}, ctx).status, STATUS_OK)
        self.assertEqual(double.calls, 4)

    def test_cache_is_context_scoped(self):
        double = ToolDouble("look")
        registry = make_registry(double)
        first, second = make_context(episode_id="ep_1"), make_context(episode_id="ep_2")
        for ctx in (first, second):
            self.assertEqual(registry.call("look", {}, ctx).status, STATUS_OK)
        self.assertEqual(double.calls, 2)
        self.assertEqual(registry.call("look", {}, second).status, STATUS_CACHED)

    def test_cache_stats_and_clear_helper(self):
        double = ToolDouble("look")
        registry = make_registry(double)
        ctx = make_context()
        cache = tool_cache(ctx)
        self.assertIs(cache, tool_cache(ctx))
        self.assertIsInstance(ctx.extra[TOOL_CACHE_KEY], ToolCallCache)
        registry.call("look", {}, ctx)
        registry.call("look", {}, ctx)
        stats = tool_cache_stats(ctx)
        self.assertEqual(stats["entries"], 1)
        self.assertEqual(stats["hits"], 1)
        self.assertGreaterEqual(stats["limit"], stats["entries"])
        self.assertEqual(clear_tool_cache(ctx), 1)
        self.assertTrue(cache.stats()["entries"] == 0)
        self.assertEqual(registry.call("look", {}, ctx).status, STATUS_OK)

    def test_unexpected_cache_value_is_replaced(self):
        double = ToolDouble("look")
        registry = make_registry(double)
        ctx = make_context(tool_cache="not-a-cache")
        self.assertEqual(registry.call("look", {}, ctx).status, STATUS_OK)
        self.assertIsInstance(ctx.extra[TOOL_CACHE_KEY], ToolCallCache)

    def test_cache_entries_serialize_to_the_episode_summary(self):
        double = ToolDouble("look")
        registry = make_registry(double)
        ctx = make_context()
        registry.call("look", {"a": 1}, ctx)
        entry = tool_cache(ctx).get("look", {"a": 1})
        self.assertEqual(entry.tool, "look")
        self.assertEqual(entry.args, {"a": 1})
        self.assertEqual(entry.call, 1)
        self.assertIsInstance(entry.result, ToolResult)
        json.dumps(tool_cache_stats(ctx))  # counters stay plain JSON


class RuntimeCacheTest(unittest.TestCase):
    def test_repeated_query_in_the_tool_loop_returns_cached(self):
        double = ToolDouble("look", data={"sector": "hallway", "clear": True})
        registry = make_registry(double)
        vlm = RecordingVLM(
            [
                {"tool": "look", "args": {"sector": "north"}},
                {"tool": "look", "args": {"sector": "north"}},
                {"tool": "finish", "args": {"reason": "no new information"}},
            ]
        )
        log = AgentRuntime(registry, FakeWorld(), None, vlm, budget=10, max_turns=6).run("scan")
        self.assertEqual(
            [record.status for record in log.turns], [STATUS_OK, STATUS_CACHED, STATUS_OK]
        )
        self.assertEqual(double.calls, 1)
        self.assertTrue(log.turns[0].new_evidence)
        self.assertFalse(log.turns[1].new_evidence)
        self.assertEqual(log.turns[1].data["hint"], REPEAT_HINT)
        self.assertEqual(log.finish_reason, REASON_FINISH)
        # the model sees the duplicate hint in the following turn's context
        self.assertIn(REPEAT_HINT, vlm.seen[2]["user"])

    def test_cache_hits_do_not_stop_a_productive_loop(self):
        # a hit is not progress, but a distinct call after it still is: the
        # stagnation counter has to reset on the fresh result
        double = ToolDouble("look")
        registry = make_registry(double)
        vlm = RecordingVLM(
            [
                {"tool": "look", "args": {"sector": "north"}},
                {"tool": "look", "args": {"sector": "north"}},
                {"tool": "look", "args": {"sector": "south"}},
                {"tool": "finish", "args": {"reason": "checked"}},
            ]
        )
        log = AgentRuntime(registry, FakeWorld(), None, vlm, budget=10, max_turns=6).run("scan")
        self.assertEqual(
            [record.status for record in log.turns],
            [STATUS_OK, STATUS_CACHED, STATUS_OK, STATUS_OK],
        )
        self.assertFalse(log.turns[1].new_evidence)
        self.assertTrue(log.turns[2].new_evidence)
        self.assertEqual(log.finish_reason, REASON_FINISH)

    def test_side_effect_call_in_the_loop_invalidates_reads(self):
        read, move = ToolDouble("look"), ToolDouble("fly", side_effects=True)
        registry = make_registry(read, move)
        vlm = RecordingVLM(
            [
                {"tool": "look", "args": {}},
                {"tool": "fly", "args": {"to": "vp_1"}},
                {"tool": "look", "args": {}},
                {"tool": "finish", "args": {"reason": "checked"}},
            ]
        )
        log = AgentRuntime(registry, FakeWorld(), None, vlm, budget=10, max_turns=6).run("scan")
        self.assertEqual(
            [record.status for record in log.turns],
            [STATUS_OK, STATUS_OK, STATUS_OK, STATUS_OK],
        )
        self.assertEqual(read.calls, 2)


class PlanRuntimeTest(unittest.TestCase):
    def test_plan_is_maintained_and_readable_inside_the_loop(self):
        registry = plan_registry()
        ctx = make_context(episode_id="ep_plan", budget=10)
        steps = [
            "go to doorway P3 and check the left side",
            "if a candidate appears, compare its attributes",
            "if there is no target, move on to P4",
        ]
        vlm = RecordingVLM(
            [
                {"tool": "update_plan", "args": {"steps": steps}},
                {"tool": "get_plan", "args": {}},
                {"tool": "finish", "args": {"reason": "plan recorded"}},
            ]
        )
        log = AgentRuntime(
            registry, FakeWorld(), None, vlm, budget=10, max_turns=6, tool_context=ctx
        ).run("inspect the hallway")

        self.assertEqual(log.tool_names(), ["update_plan", "get_plan", "finish"])
        self.assertEqual(log.finish_reason, REASON_FINISH)
        self.assertEqual(log.turns[0].data["revision"], 1)
        self.assertEqual(log.turns[1].data["plan_id"], log.turns[0].data["plan_id"])
        self.assertEqual(log.turns[1].data["steps"], steps)
        # the plan stays readable outside the loop ...
        self.assertEqual(plan_state(ctx)["steps"], steps)
        self.assertIn("rev=1", plan_text(ctx))
        # ... and reaches the model: the catalogue advertises the tools and the
        # plan rides back in the structured tool feedback
        self.assertIn("- update_plan:", vlm.seen[0]["tools_text"])
        self.assertIn("- get_plan:", vlm.seen[0]["tools_text"])
        self.assertIn("plan_1", vlm.seen[1]["user"])
        self.assertIn("doorway P3", vlm.seen[1]["user"])

    def test_plan_revision_is_visible_to_the_next_turn(self):
        registry = plan_registry()
        vlm = RecordingVLM(
            [
                {"tool": "update_plan", "args": {"steps": ["try vp_1"]}},
                {"tool": "update_plan", "args": {"steps": ["try vp_1", "if blocked, try vp_2"]}},
                {"tool": "get_plan", "args": {}},
                {"tool": "finish", "args": {"reason": "done"}},
            ]
        )
        log = AgentRuntime(registry, FakeWorld(), None, vlm, budget=10, max_turns=6).run("go")
        self.assertEqual(log.turns[0].data["revision"], 1)
        self.assertEqual(log.turns[1].data["revision"], 2)
        self.assertTrue(log.turns[1].data["changed"])
        self.assertEqual(log.turns[2].data["revision"], 2)
        self.assertEqual(log.turns[2].data["n_steps"], 2)

    def test_stagnating_get_plan_stops_the_episode(self):
        registry = plan_registry()
        vlm = RecordingVLM([{"tool": "get_plan", "args": {}}] * 4)
        log = AgentRuntime(
            registry, FakeWorld(), None, vlm, budget=10, max_turns=6, max_stagnant_turns=1
        ).run("go")
        self.assertEqual(log.finish_reason, REASON_NO_NEW_EVIDENCE)
        self.assertEqual([record.status for record in log.turns], [STATUS_EMPTY, STATUS_EMPTY])


class PlanProgressTest(unittest.TestCase):
    """The plan carries its own execution progress and completion condition."""

    def test_update_plan_stores_the_progress_and_the_completion_condition(self):
        registry = plan_registry()
        ctx = make_context()
        result = registry.call(
            "update_plan",
            {
                "steps": ["observe forward", "if clear, go to vp_2", "approach the candidate"],
                "current_step": 2,
                "done": [1],
                "done_when": "the candidate is verified",
            },
            ctx,
        )
        self.assertEqual(result.status, STATUS_OK)
        self.assertEqual(result.data["current_step"], 2)
        self.assertEqual(result.data["current_step_text"], "if clear, go to vp_2")
        self.assertEqual(result.data["done"], [1])
        self.assertEqual(result.data["done_when"], "the candidate is verified")
        self.assertFalse(result.data["complete"])
        record = plan_state(ctx)
        self.assertEqual(record["current_step"], 2)
        self.assertEqual(record["done_when"], "the candidate is verified")

        rendered = plan_progress(ctx)
        self.assertIn("plan_1", rendered)
        self.assertIn("step 2/3", rendered)
        self.assertIn("1 done", rendered)
        self.assertIn("if clear, go to vp_2", rendered)
        self.assertIn("the candidate is verified", rendered)
        self.assertEqual(result.data["progress"], rendered)

    def test_a_progress_only_update_keeps_the_steps_and_moves_the_revision(self):
        registry = plan_registry()
        ctx = make_context()
        registry.call("update_plan", {"steps": ["a", "b", "c"]}, ctx)
        marked = registry.call("update_plan", {"done": [1, 2]}, ctx)
        self.assertEqual(marked.status, STATUS_OK)
        self.assertTrue(marked.data["changed"])
        self.assertEqual(marked.data["revision"], 2)
        self.assertEqual(marked.data["steps"], ["a", "b", "c"])
        self.assertEqual(marked.data["done"], [1, 2])
        # the current step follows the first open one
        self.assertEqual(marked.data["current_step"], 3)
        self.assertEqual(marked.data["current_step_text"], "c")
        # re-sending the same progress is not a change
        again = registry.call("update_plan", {"done": [1, 2]}, ctx)
        self.assertFalse(again.data["changed"])
        self.assertEqual(again.data["revision"], 2)

    def test_a_completed_plan_says_so(self):
        registry = plan_registry()
        ctx = make_context()
        registry.call("update_plan", {"steps": ["a", "b"]}, ctx)
        done = registry.call("update_plan", {"done": [1, 2]}, ctx)
        self.assertTrue(done.data["complete"])
        self.assertIn("COMPLETE", plan_progress(ctx))
        self.assertIn("2 done", plan_progress(ctx))

    def test_step_texts_resolve_and_unusable_progress_entries_are_reported(self):
        registry = plan_registry()
        ctx = make_context()
        registry.call("update_plan", {"steps": ["observe forward", "go to vp_2"]}, ctx)
        result = registry.call(
            "update_plan",
            {
                "current_step": "go to vp_2",
                "done": ["observe forward", 9, "nope"],
            },
            ctx,
        )
        self.assertEqual(result.data["current_step"], 2)
        self.assertEqual(result.data["done"], [1])
        self.assertTrue(result.data["progress_notes"])
        self.assertIn("9", " ".join(result.data["progress_notes"]))

    def test_progress_without_a_plan_still_requires_steps(self):
        result = plan_registry().call("update_plan", {"done": [1]}, make_context())
        self.assertEqual(result.status, STATUS_FAILED)
        self.assertEqual(result.data["error"], "missing_steps")

    def test_plan_progress_without_a_plan(self):
        self.assertEqual(plan_progress(make_context()), "no plan")

    def test_get_plan_answers_with_the_progress(self):
        registry = plan_registry()
        ctx = make_context()
        registry.call(
            "update_plan",
            {"steps": ["a", "b"], "done": [1], "done_when": "b is confirmed"},
            ctx,
        )
        result = registry.call("get_plan", {}, ctx)
        self.assertEqual(result.data["current_step"], 2)
        self.assertEqual(result.data["done"], [1])
        self.assertEqual(result.data["done_when"], "b is confirmed")
        self.assertIn("step 2/2", result.data["progress"])

    def test_plan_text_also_carries_the_progress(self):
        registry = plan_registry()
        ctx = make_context()
        registry.call("update_plan", {"steps": ["a", "b"], "done": [1]}, ctx)
        text = plan_text(ctx)
        self.assertIn("rev=1", text)
        self.assertIn("step 2/2", text)


# ----------------------------------------------------------------------
# P4: the sub-task record, its executor and the resume path
# ----------------------------------------------------------------------


def subtasks_of(ctx):
    """The stored sub-task records of the episode's plan."""
    return (plan_state(ctx) or {}).get("subtasks") or []


def plan_with(registry, ctx, steps, **args):
    result = registry.call("update_plan", {"steps": steps, **args}, ctx)
    assert result.status == STATUS_OK, result.data
    return result


def step_event(
    tool="look",
    *,
    status=STATUS_OK,
    data=None,
    evidence=(),
    kind="",
    turn=1,
    args=None,
):
    """One tool result as the runtime hands it to :func:`executor_step`."""
    refs = [
        ref if isinstance(ref, dict) else {"kind": ref[0], "ref": ref[1], "note": ""}
        for ref in evidence
    ]
    return {
        "tool": tool,
        "args": dict(args or {}),
        "status": status,
        "data": dict(data or {}),
        "evidence": refs,
        "progress_kind": kind,
        "turn": turn,
    }


class SubtaskSchemaTest(unittest.TestCase):
    """A step is a record: goal, preconditions, dependencies, expected
    observation, completion condition, failure branch and budget. A plain text
    step is wrapped into one automatically (P4)."""

    def test_plain_text_steps_are_wrapped_into_subtask_records(self):
        registry, ctx = plan_registry(), make_context()
        result = plan_with(registry, ctx, ["find the corridor", "check the cabinet"])

        subtasks = subtasks_of(ctx)
        self.assertEqual([s["id"] for s in subtasks], ["st_1", "st_2"])
        self.assertEqual(
            [s["goal"] for s in subtasks], ["find the corridor", "check the cabinet"]
        )
        for subtask in subtasks:
            self.assertEqual(subtask["preconditions"], [])
            self.assertEqual(subtask["depends_on"], [])
            self.assertEqual(subtask["expected_observation"], "")
            self.assertEqual(subtask["done_when"], "")
            self.assertEqual(subtask["failure_branch"], [])
            self.assertEqual(subtask["budget"], {"tool_calls": 0, "turns": 0})
            self.assertEqual(subtask["evidence"], [])
            self.assertEqual(subtask["spent"], {"tool_calls": 0, "turns": 0})
        # the first step of a fresh plan is the one being executed
        self.assertEqual(
            [s["status"] for s in subtasks], [SUBTASK_ACTIVE, SUBTASK_PENDING]
        )
        # the legacy progress contract is unchanged: steps/done/current_step
        self.assertEqual(
            plan_state(ctx)["steps"], ["find the corridor", "check the cabinet"]
        )
        self.assertEqual(result.data["current_step"], 1)
        self.assertEqual(result.data["done"], [])

    def test_a_full_subtask_entry_round_trips(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [
                {
                    "id": "corridor",
                    "goal": "find the corridor on the left",
                    "preconditions": ["the doorway P3 is reached"],
                    "depends_on": [],
                    "expected_observation": "a hallway with two doors",
                    "done_when": "the corridor is visible",
                    "failure_branch": ["if blocked, propose_viewpoints then go_to_viewpoint vp_9"],
                    "budget": {"tool_calls": 4, "turns": 2},
                },
                {
                    "id": "cabinet",
                    "goal": "check the cabinet next to the door",
                    "depends_on": ["corridor"],
                    "done_when": "the cabinet is in view",
                    "failure_branch": "look behind the door; check the other side",
                    "budget": 3,
                },
            ],
        )

        first, second = subtasks_of(ctx)
        self.assertEqual(first["id"], "corridor")
        self.assertEqual(first["preconditions"], ["the doorway P3 is reached"])
        self.assertEqual(first["expected_observation"], "a hallway with two doors")
        self.assertEqual(first["done_when"], "the corridor is visible")
        self.assertEqual(
            first["failure_branch"],
            ["if blocked, propose_viewpoints then go_to_viewpoint vp_9"],
        )
        self.assertEqual(first["budget"], {"tool_calls": 4, "turns": 2})
        # a dependency may be named by id; a string branch splits like a plan
        self.assertEqual(second["depends_on"], ["corridor"])
        self.assertEqual(
            second["failure_branch"], ["look behind the door", "check the other side"]
        )
        self.assertEqual(second["budget"], {"tool_calls": 3, "turns": 0})
        self.assertEqual([s["status"] for s in subtasks_of(ctx)],
                         [SUBTASK_ACTIVE, SUBTASK_PENDING])

    def test_next_step_and_the_located_scope_round_trip(self):
        """A sub-task can declare the place its evidence must point at and the
        concrete call that continues it (the executor's deterministic source)."""
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [
                {
                    "id": "door",
                    "goal": "find the wooden door",
                    "done_when": "the door is located within 2 m and reached",
                    "goal_position": [1.2, 3.4, 0.5],
                    "locate_m": 1.5,
                    "next_step": {"tool": "observe", "args": {"mode": "forward"}},
                    "failure_branch": ['go_to_viewpoint({"viewpoint_id": "vp_9"})'],
                }
            ],
        )

        subtask = subtasks_of(ctx)[0]
        self.assertEqual(subtask["goal_position"], [1.2, 3.4, 0.5])
        self.assertEqual(subtask["locate_m"], 1.5)
        self.assertEqual(
            subtask["next_step"], {"tool": "observe", "args": {"mode": "forward"}}
        )
        self.assertEqual(
            subtask["failure_branch"], ['go_to_viewpoint({"viewpoint_id": "vp_9"})']
        )

        # a patch can declare the place later (e.g. once a tool reported it)
        registry.call(
            "update_plan",
            {"patch": [{"id": "door", "goal_position": [2.0, 3.0, 0.0], "locate_m": 2.0}]},
            ctx,
        )
        subtask = subtasks_of(ctx)[0]
        self.assertEqual(subtask["goal_position"], [2.0, 3.0, 0.0])
        self.assertEqual(subtask["locate_m"], 2.0)

    def test_depends_on_accepts_indices_and_the_status_can_be_declared(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [
                {"goal": "scan the room", "done": True},
                {"goal": "report the layout", "depends_on": [1], "status": "blocked"},
            ],
        )
        first, second = subtasks_of(ctx)
        self.assertEqual(first["status"], SUBTASK_DONE)
        self.assertEqual(second["depends_on"], ["st_1"])
        self.assertEqual(second["status"], SUBTASK_BLOCKED)
        # the statuses drive the legacy progress fields
        self.assertEqual(plan_state(ctx)["done"], [1])

    def test_lists_and_budgets_are_capped(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [
                {
                    "goal": "bounded",
                    "preconditions": [f"p{i}" for i in range(20)],
                    "budget": {"tool_calls": 10**6, "turns": -3},
                }
            ],
        )
        subtask = subtasks_of(ctx)[0]
        self.assertLessEqual(len(subtask["preconditions"]), 6)
        self.assertLessEqual(subtask["budget"]["tool_calls"], 999)
        self.assertEqual(subtask["budget"]["turns"], 0)

    def test_normalize_subtasks_reports_the_error_of_a_bad_step_list(self):
        self.assertEqual(normalize_subtasks({"steps": 42})[2], "invalid_steps")
        self.assertEqual(normalize_subtasks({})[2], "missing_steps")
        self.assertIsNone(normalize_subtasks({"steps": []})[0])

    def test_subtask_fields_are_the_documented_schema(self):
        self.assertEqual(
            tuple(SUBTASK_FIELDS),
            (
                "id", "goal", "preconditions", "depends_on", "expected_observation",
                "done_when", "goal_position", "locate_m", "reach_m", "failure_branch",
                "next_step", "budget", "status", "evidence", "spent", "reason",
                "updated_turn",
            ),
        )
        self.assertEqual(
            tuple(SUBTASK_STATUSES),
            ("pending", "active", "claimed_done", "done", "failed", "blocked"),
        )


class SubtaskPatchTest(unittest.TestCase):
    """The model revises one sub-task without re-stating the whole plan."""

    def test_a_patch_argument_updates_one_subtask_and_keeps_the_rest(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [
                {"id": "corridor", "goal": "find the corridor", "failure_branch": ["go to vp_9"]},
                {"id": "cabinet", "goal": "find the cabinet by the door"},
            ],
        )
        result = registry.call(
            "update_plan",
            {"patch": [{"id": "cabinet", "goal": "find the cabinet beside the door"}]},
            ctx,
        )
        self.assertEqual(result.status, STATUS_OK)
        self.assertTrue(result.data["changed"])
        self.assertEqual(result.data["n_steps"], 2)
        first, second = subtasks_of(ctx)
        self.assertEqual(first["id"], "corridor")
        self.assertEqual(first["goal"], "find the corridor")
        self.assertEqual(first["failure_branch"], ["go to vp_9"])
        self.assertEqual(second["id"], "cabinet")
        self.assertEqual(second["goal"], "find the cabinet beside the door")
        # the execution state of the patched sub-task survives the patch
        self.assertEqual(second["status"], SUBTASK_PENDING)
        self.assertEqual(plan_state(ctx)["steps"], ["find the corridor", "find the cabinet beside the door"])

    def test_a_patch_entry_inside_steps_is_recognized(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(registry, ctx, ["find the corridor", "check the cabinet"])
        result = registry.call(
            "update_plan", {"steps": [{"id": "st_2", "done_when": "the cabinet is in view"}]}, ctx
        )
        self.assertEqual(result.status, STATUS_OK)
        self.assertEqual(result.data["n_steps"], 2)
        self.assertEqual(subtasks_of(ctx)[1]["done_when"], "the cabinet is in view")
        self.assertEqual(subtasks_of(ctx)[0]["goal"], "find the corridor")

    def test_a_patch_that_changes_nothing_keeps_the_revision(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(registry, ctx, ["find the corridor", "check the cabinet"])
        revision = plan_state(ctx)["revision"]
        result = registry.call(
            "update_plan", {"patch": [{"id": "st_1", "goal": "find the corridor"}]}, ctx
        )
        self.assertEqual(result.status, STATUS_OK)
        self.assertFalse(result.data["changed"])
        self.assertEqual(plan_state(ctx)["revision"], revision)

    def test_a_patch_of_an_unknown_id_is_reported_and_changes_nothing(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(registry, ctx, ["only step"])
        result = registry.call("update_plan", {"patch": [{"id": "nope", "goal": "x"}]}, ctx)
        self.assertEqual(result.status, STATUS_FAILED)
        self.assertEqual(result.data["error"], "unknown_subtask")
        self.assertIn("nope", result.data["unknown_subtasks"])
        self.assertIn("st_1", result.data["hint"])
        self.assertEqual(plan_state(ctx)["revision"], 1)
        self.assertEqual(subtasks_of(ctx)[0]["goal"], "only step")

    def test_a_replacement_inherits_the_state_of_a_restated_step(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(registry, ctx, ["scan the corridor", "check the cabinet"])
        executor_step(
            ctx,
            step_event(
                "scan",
                status=STATUS_OK,
                data={"subgoal_complete": True},
                evidence=[("observation", "obs_1")],
            ),
        )
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_DONE)

        plan_with(registry, ctx, ["scan the corridor", "check the cabinet", "confirm the target"])
        first, second, third = subtasks_of(ctx)
        self.assertEqual(first["id"], "st_1")
        self.assertEqual(first["status"], SUBTASK_DONE)  # the restated step keeps its state
        self.assertEqual(second["status"], SUBTASK_ACTIVE)
        self.assertEqual(third["status"], SUBTASK_PENDING)


class GetPlanProgressTest(unittest.TestCase):
    """``get_plan`` answers with the execution progress of the plan (P4)."""

    def test_get_plan_reports_the_current_done_and_failed_subtasks(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [
                {"goal": "scan the corridor", "budget": 4},
                {"goal": "check the cabinet", "budget": 2},
                {"goal": "confirm the target"},
            ],
        )
        executor_step(
            ctx,
            step_event("scan", data={"subgoal_complete": True}, evidence=[("observation", "obs_3")]),
        )
        data = registry.call("get_plan", {}, ctx).data

        self.assertEqual(data["current_subtask"]["id"], "st_2")
        self.assertEqual(data["current_subtask"]["status"], SUBTASK_ACTIVE)
        self.assertEqual(data["done_subtasks"], ["st_1"])
        self.assertEqual(data["blocked_subtasks"], [])
        self.assertEqual(data["failed_subtasks"], [])
        # the budget consumption is reported per call and in total
        self.assertEqual(data["budget_used"]["tool_calls"], 1)
        self.assertEqual(data["subtasks"][1]["budget"], {"tool_calls": 2, "turns": 0})
        self.assertEqual(data["subtasks"][0]["status"], SUBTASK_DONE)
        self.assertEqual(data["subtasks"][0]["evidence"], ["observation:obs_3"])

    def test_a_blocked_subtask_reports_its_reason_and_its_failure_branch(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [
                {
                    "goal": "fly to vp_3",
                    "failure_branch": ["go_to_viewpoint vp_9"],
                },
                {"goal": "check the cabinet"},
            ],
        )
        executor_step(
            ctx,
            step_event(
                "go_to_viewpoint",
                status=STATUS_BLOCKED,
                data={"outcome": "planner_failed"},
            ),
        )
        data = registry.call("get_plan", {}, ctx).data
        blocked = data["blocked_subtasks"][0]
        self.assertEqual(blocked["id"], "st_1")
        self.assertEqual(blocked["reason"], "planner_failed")
        self.assertEqual(blocked["failure_branch"], ["go_to_viewpoint vp_9"])
        self.assertEqual(data["current_subtask"]["id"], "st_1")
        self.assertEqual(data["current_subtask"]["status"], SUBTASK_BLOCKED)


class ExecutorStepTest(unittest.TestCase):
    """``executor_step`` advances the sub-tasks from the tool results."""

    def test_a_completion_signal_finishes_the_subtask_and_activates_the_next(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(registry, ctx, ["scan the corridor", "check the cabinet"])
        outcome = executor_step(
            ctx,
            step_event("scan", data={"subgoal_complete": True}, evidence=[("observation", "obs_2")]),
        )
        self.assertTrue(outcome["changed"])
        self.assertEqual(outcome["completed"], "st_1")
        self.assertEqual(outcome["active"], "st_2")
        self.assertEqual(outcome["decision"], "subtask_switch")
        self.assertEqual(
            [t["to"] for t in outcome["transitions"]], [SUBTASK_DONE, SUBTASK_ACTIVE]
        )
        self.assertEqual(
            outcome["transitions"][0],
            {"id": "st_1", "from": SUBTASK_ACTIVE, "to": SUBTASK_DONE,
             "reason": "done_when_met", "turn": 1},
        )
        first, second = subtasks_of(ctx)
        self.assertEqual(first["status"], SUBTASK_DONE)
        self.assertEqual(first["evidence"], ["observation:obs_2"])
        self.assertEqual(first["reason"], "done_when_met")
        self.assertEqual(second["status"], SUBTASK_ACTIVE)

    def test_one_step_advances_a_chain_of_subtasks(self):
        """A result that closes several already-finished steps activates the
        next open one in a single executor step (no extra decision turns)."""
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            ["scan the corridor", "check the cabinet", "confirm the target"],
        )
        # the middle step is declared done by the model, the first one is finished
        # by the tool result: the executor has to walk both in one step
        registry.call("update_plan", {"patch": [{"id": "st_2", "status": "done"}]}, ctx)
        outcome = executor_step(
            ctx,
            step_event(
                "scan",
                data={"subgoal_complete": True},
                evidence=[("observation", "obs_1")],
            ),
        )
        self.assertEqual([t["to"] for t in outcome["transitions"]],
                         [SUBTASK_DONE, SUBTASK_ACTIVE])
        self.assertEqual(outcome["active"], "st_3")
        self.assertEqual([s["status"] for s in subtasks_of(ctx)],
                         [SUBTASK_DONE, SUBTASK_DONE, SUBTASK_ACTIVE])

    def test_a_failed_leg_blocks_the_subtask_and_records_the_evidence(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [
                {"goal": "fly to vp_3", "failure_branch": ["go_to_viewpoint vp_9"]},
                {"goal": "check the cabinet", "depends_on": ["st_1"]},
            ],
        )
        outcome = executor_step(
            ctx,
            step_event(
                "go_to_viewpoint",
                status=STATUS_BLOCKED,
                data={"outcome": "planner_failed"},
                evidence=[("trajectory", "trace_4")],
            ),
        )
        self.assertEqual(outcome["blocked"], "st_1")
        self.assertEqual(outcome["blocked_reason"], "planner_failed")
        self.assertEqual(outcome["failure_subtask"], "st_1")
        self.assertEqual(outcome["failure_branch"], ["go_to_viewpoint vp_9"])
        first, second = subtasks_of(ctx)
        self.assertEqual(first["status"], SUBTASK_BLOCKED)
        self.assertEqual(first["reason"], "planner_failed")
        self.assertEqual(first["evidence"], ["trajectory:trace_4"])
        # a blocked sub-task stays the current one: its failure branch is the way out
        self.assertEqual(second["status"], SUBTASK_PENDING)

    def test_budget_overrun_marks_the_subtask_failed(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(registry, ctx, [{"goal": "scan the corridor", "budget": {"tool_calls": 1}}])
        first = executor_step(ctx, step_event("scan", turn=1))
        self.assertEqual(first["charged"], "st_1")
        self.assertEqual(first["failed"], "")  # the one call it was allowed
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_ACTIVE)

        over = executor_step(ctx, step_event("scan", turn=2))
        self.assertEqual(over["failed"], "st_1")
        self.assertEqual(over["budget_exceeded"], ["st_1"])
        subtask = subtasks_of(ctx)[0]
        self.assertEqual(subtask["status"], SUBTASK_FAILED)
        self.assertEqual(subtask["reason"], "budget_exceeded:tool_calls")

    def test_turn_budget_overrun_marks_the_subtask_failed(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(registry, ctx, [{"goal": "scan the corridor", "budget": {"turns": 1}}])
        executor_step(ctx, step_event("scan", turn=1))
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_ACTIVE)
        # the same turn is not charged twice ...
        executor_step(ctx, step_event("look", turn=1))
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_ACTIVE)
        # ... the next turn is
        over = executor_step(ctx, step_event("look", turn=2))
        self.assertEqual(over["failed"], "st_1")
        self.assertEqual(subtasks_of(ctx)[0]["reason"], "budget_exceeded:turns")

    def test_depends_on_gates_the_activation(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [
                {"id": "corridor", "goal": "find the corridor"},
                {"id": "cabinet", "goal": "check the cabinet", "depends_on": ["corridor"]},
            ],
        )
        # the cabinet may not become active while the corridor step is open
        outcome = executor_step(ctx, step_event("look", data={"checked": "north"}))
        self.assertEqual(outcome["active"], "corridor")
        self.assertEqual(subtasks_of(ctx)[1]["status"], SUBTASK_PENDING)

        outcome = executor_step(
            ctx,
            step_event(
                "ground",
                data={"subgoal_complete": True},
                evidence=[("observation", "obs_1")],
            ),
        )
        self.assertEqual(outcome["active"], "cabinet")

    def test_a_later_step_is_claimed_until_its_dependency_is_done(self):
        """A completion signal for a later stage cannot outrun its predecessor;
        once the predecessor is done the claimed stage is promoted in place."""
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [
                {"id": "door", "goal": "find the door"},
                {"id": "cabinet", "goal": "check the cabinet", "depends_on": ["door"]},
            ],
        )
        early = executor_step(
            ctx,
            step_event(
                "look",
                data={"subtask_id": "cabinet", "subgoal_complete": True},
                evidence=[("observation", "obs_2")],
            ),
        )
        self.assertEqual(early["claimed"], "cabinet")
        self.assertEqual(early["completed"], "")
        self.assertEqual(early["out_of_order"], "cabinet")
        self.assertEqual(early["predecessors"], ["door"])
        door, cabinet = subtasks_of(ctx)
        self.assertEqual(door["status"], SUBTASK_ACTIVE)
        self.assertEqual(cabinet["status"], SUBTASK_CLAIMED_DONE)
        self.assertEqual(cabinet["reason"], "predecessors_open:door")

        closed = executor_step(
            ctx,
            step_event(
                "look",
                turn=2,
                data={"subgoal_complete": True},
                evidence=[("observation", "obs_1")],
            ),
        )
        self.assertEqual(closed["completed"], "door")
        door, cabinet = subtasks_of(ctx)
        self.assertEqual(door["status"], SUBTASK_DONE)
        # the dependency is done: the claim becomes a real completion
        self.assertEqual(cabinet["status"], SUBTASK_DONE)
        self.assertIn("cabinet", closed["promoted"])

    def test_the_ordinal_order_gates_a_later_step_without_depends_on(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(registry, ctx, ["scan the corridor", "check the cabinet", "confirm the box"])
        early = executor_step(
            ctx,
            step_event(
                "verify",
                data={"subtask_id": "st_3", "subgoal_complete": True},
                evidence=[("observation", "obs_3")],
            ),
        )
        self.assertEqual(early["claimed"], "st_3")
        self.assertEqual(early["predecessors"], ["st_1", "st_2"])
        self.assertEqual(subtasks_of(ctx)[2]["status"], SUBTASK_CLAIMED_DONE)

        executor_step(
            ctx,
            step_event("look", turn=2, data={"subgoal_complete": True},
                       evidence=[("observation", "obs_1")]),
        )
        # the claim waiting for st_2 does not stop st_2 from being activated
        self.assertEqual(subtasks_of(ctx)[1]["status"], SUBTASK_ACTIVE)
        self.assertEqual(subtasks_of(ctx)[2]["status"], SUBTASK_CLAIMED_DONE)
        executor_step(
            ctx,
            step_event("look", turn=3, data={"subgoal_complete": True},
                       evidence=[("observation", "obs_2")]),
        )
        self.assertEqual([s["status"] for s in subtasks_of(ctx)],
                         [SUBTASK_DONE, SUBTASK_DONE, SUBTASK_DONE])

    def test_plan_bookkeeping_calls_are_not_execution_steps(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(registry, ctx, [{"goal": "scan", "budget": 1}])
        for tool in EXECUTOR_SKIP_TOOLS:
            outcome = executor_step(ctx, step_event(tool))
            self.assertFalse(outcome["changed"], tool)
            self.assertEqual(outcome["charged"], "", tool)
        self.assertEqual(subtasks_of(ctx)[0]["spent"], {"tool_calls": 0, "turns": 0})

    def test_the_executor_never_charges_the_global_budget(self):
        registry, ctx = plan_registry(), make_context(budget=7)
        plan_with(registry, ctx, ["scan the corridor"])
        left = ctx.budget["tool_calls_left"]
        executor_step(ctx, step_event("scan"))
        self.assertEqual(ctx.budget["tool_calls_left"], left)

    def test_a_step_without_a_plan_changes_nothing(self):
        outcome = executor_step(make_context(), step_event("scan"))
        self.assertFalse(outcome["changed"])
        self.assertEqual(outcome["transitions"], [])
        self.assertEqual(outcome["active"], "")

    def test_transitions_are_recorded_on_the_plan_record(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(registry, ctx, ["scan the corridor", "check the cabinet"])
        executor_step(
            ctx,
            step_event(
                "scan",
                turn=1,
                data={"subgoal_complete": True},
                evidence=[("observation", "obs_1")],
            ),
        )
        transitions = plan_state(ctx)["executor"]["transitions"]
        self.assertEqual(len(transitions), 2)
        self.assertEqual(transitions[-1]["to"], SUBTASK_ACTIVE)
        self.assertEqual(transitions[-1]["turn"], 1)

    def test_the_executor_only_keeps_a_bounded_evidence_list(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(registry, ctx, ["scan the corridor"])
        for turn in range(MAX_SUBTASK_EVIDENCE + 4):
            executor_step(
                ctx,
                step_event(
                    "look",
                    turn=turn + 1,
                    evidence=[("observation", f"obs_{turn + 1}")],
                ),
            )
        evidence = subtasks_of(ctx)[0]["evidence"]
        self.assertEqual(len(evidence), MAX_SUBTASK_EVIDENCE)
        self.assertEqual(evidence[-1], f"observation:obs_{MAX_SUBTASK_EVIDENCE + 4}")

    def test_the_failure_and_signal_vocabularies_match_the_runtime(self):
        """The executor reads the same outcome/signal words the runtime's
        recovery rules and progress scoring read - one vocabulary, two readers -
        plus the sub-task-specific signals a tool may report directly."""
        self.assertEqual(set(FAILURE_OUTCOMES), set(BAD_OUTCOMES))
        self.assertTrue(set(RUNTIME_TASK_SIGNAL_KEYS).issubset(set(TASK_SIGNAL_KEYS)))
        self.assertEqual(
            set(TASK_SIGNAL_KEYS) - set(RUNTIME_TASK_SIGNAL_KEYS),
            {"subtask_complete", "step_complete", "done_when_met"},
        )


#: The landmark stage's real ``done_when`` (verify_long_horizon task file): the
#: stage needs a reported position *and* an arrival, not just a confident word.
DOOR_DONE_WHEN = (
    "木门被定位并到达：某次工具结果给出距木门 2 m 以内的位置，"
    "且无人机飞到距木门 2 m 以内"
)
CABINET_DONE_WHEN = (
    "柜子被定位并到达：某次工具结果给出距柜子 2 m 以内的位置，"
    "且无人机飞到距柜子 2 m 以内"
)
TARGET_DONE_WHEN = (
    "纸箱被确认为目标（verify_goal 通过或定位位置进入成功距离）并飞到它旁边"
)


def ray_only_search_event(*, turn=1):
    """A search skill that "confirmed" a candidate it could only place on a ray.

    This is the shape of the real failure (ep_007/ep_011): ``confirmed=True`` is
    a task signal, the candidate has evidence references, but its position is
    unresolved (``position_source=ray``, no place) - the evidence does not
    satisfy a ``done_when`` that asks for a located and reached object.
    """
    return step_event(
        "search_target",
        turn=turn,
        data={
            "status": "ok",
            "confirmed": True,
            "candidate_id": "cand_1",
            "candidates": [
                {
                    "id": "cand_1",
                    "observation_id": "obs_2",
                    "position_source": "ray",
                    "range_status": "unknown",
                    "ray": {"origin_flu": [0.85, 4.6, -1.6], "bearing_deg": -83.5},
                }
            ],
        },
        evidence=[("memory", "cand_1"), ("observation", "obs_2")],
    )


def located_move_event(*, turn=1, signal=True, arrived=True, resolved=True):
    """A movement call that resolved the candidate's place and flew a leg."""
    data = {
        "candidate_id": "cand_1",
        "outcome": "arrived" if arrived else "partial",
        "task_status": "arrived" if arrived else "running",
        "moved_m": 3.0,
        "position_source": "depth" if resolved else "ray",
        "target_position": [1.2, 1.69, -0.98] if resolved else [],
    }
    if signal:
        data["subgoal_complete"] = True
    return step_event(
        "approach_candidate",
        turn=turn,
        data=data,
        evidence=[("observation", "obs_11"), ("memory", "cand_1")],
    )


class ExecutorEvidenceBindingTest(unittest.TestCase):
    """A sub-task is ``done`` only when its ``evidence[]`` satisfies its
    ``done_when``; a completion signal without that evidence is ``claimed_done``
    (P4 defect A)."""

    def _door_plan(self, registry, ctx, *, budget=0):
        first = {"id": "st_1", "goal": "find the wooden door", "done_when": DOOR_DONE_WHEN}
        if budget:
            first["budget"] = {"tool_calls": budget}
        plan_with(
            registry,
            ctx,
            [first, {"id": "st_2", "goal": "find the cabinet", "depends_on": ["st_1"]}],
        )

    def test_a_signal_without_satisfying_evidence_is_claimed_not_done(self):
        registry, ctx = plan_registry(), make_context()
        self._door_plan(registry, ctx)
        outcome = executor_step(ctx, ray_only_search_event())

        self.assertEqual(outcome["claimed"], "st_1")
        self.assertEqual(outcome["completed"], "")
        self.assertEqual(outcome["evidence_missing"], ["located", "reached"])
        self.assertEqual(
            [entry["to"] for entry in outcome["transitions"]], [SUBTASK_CLAIMED_DONE]
        )
        first, second = subtasks_of(ctx)
        self.assertEqual(first["status"], SUBTASK_CLAIMED_DONE)
        self.assertTrue(first["reason"].startswith("evidence_insufficient"))
        # an unproven step does not hand the plan to the next one
        self.assertEqual(second["status"], SUBTASK_PENDING)
        self.assertEqual(plan_state(ctx)["done"], [])

        data = registry.call("get_plan", {}, ctx).data
        self.assertEqual(data["done_subtasks"], [])
        self.assertEqual(data["claimed_subtasks"][0]["id"], "st_1")
        self.assertEqual(data["claimed_subtasks"][0]["evidence_missing"],
                         ["located", "reached"])
        self.assertEqual(data["current_subtask"]["status"], SUBTASK_CLAIMED_DONE)
        self.assertEqual(data["current_step"], 1)
        self.assertIn("current=st_1[claimed_done]", subtask_progress(ctx))

    def test_a_partial_leg_is_not_reach_evidence(self):
        """A flown leg is ``moved``, not ``reached``: the movement stage stays
        claimed until a call reports an arrival (or a remaining distance inside
        the stage's reach tolerance)."""
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [
                {
                    "id": "st_1",
                    "goal": "approach the wooden door",
                    "done_when": DOOR_DONE_WHEN,
                    "reach_m": 0.8,
                }
            ],
        )
        partial = executor_step(
            ctx,
            step_event(
                "approach_candidate",
                data={
                    "subgoal_complete": True,
                    "outcome": "partial",
                    "task_status": "running",
                    "moved_m": 3.0,
                    "remaining_m": 2.5,
                    "target_position": [1.0, 0.0, 0.0],
                },
                evidence=[("observation", "obs_11")],
            ),
        )
        self.assertEqual(partial["claimed"], "st_1")
        self.assertEqual(partial["completed"], "")
        self.assertEqual(partial["evidence_missing"], ["reached"])
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_CLAIMED_DONE)

        done = executor_step(
            ctx,
            step_event(
                "approach_candidate",
                turn=2,
                data={
                    "outcome": "arrived",
                    "task_status": "arrived",
                    "moved_m": 0.2,
                    "remaining_m": 0.4,
                    "target_position": [1.0, 0.0, 0.0],
                },
                evidence=[("observation", "obs_12")],
            ),
        )
        self.assertEqual(done["completed"], "st_1")
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_DONE)

    def test_reach_evidence_is_scoped_to_a_located_place(self):
        """An arrival somewhere else is not an arrival at *this* stage's object:
        with a located place on record, the movement target has to be near it."""
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [
                {
                    "id": "st_1",
                    "goal": "approach the wooden door",
                    "done_when": "the door is located within 2 m and reached",
                    "reach_m": 1.0,
                }
            ],
        )
        located = executor_step(
            ctx,
            step_event(
                "ground_target",
                data={
                    "subgoal_complete": True,
                    "candidate_id": "cand_1",
                    "target_position": [0.0, 0.0, 0.0],
                },
                evidence=[("observation", "obs_1")],
            ),
        )
        self.assertEqual(located["claimed"], "st_1")
        self.assertEqual(located["evidence_missing"], ["reached"])

        elsewhere = executor_step(
            ctx,
            step_event(
                "go_to_viewpoint",
                turn=2,
                data={
                    "outcome": "arrived",
                    "task_status": "arrived",
                    "remaining_m": 0.0,
                    "target_position": [10.0, 0.0, 0.0],
                },
                evidence=[("observation", "obs_2")],
            ),
        )
        self.assertEqual(elsewhere["completed"], "")
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_CLAIMED_DONE)

        at_target = executor_step(
            ctx,
            step_event(
                "approach_candidate",
                turn=3,
                data={
                    "outcome": "arrived",
                    "task_status": "arrived",
                    "remaining_m": 0.0,
                    "target_position": [0.5, 0.0, 0.0],
                },
                evidence=[("observation", "obs_3")],
            ),
        )
        self.assertEqual(at_target["completed"], "st_1")
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_DONE)

    def test_a_reach_tolerance_turns_a_located_stage_into_a_reach_stage(self):
        """``reach_m`` marks a movement stage even when its ``done_when`` only
        names a place: located-only stays claimed, a remaining distance inside
        the tolerance closes it, one outside does not."""
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [
                {
                    "id": "st_1",
                    "goal": "approach the box",
                    "done_when": "the box is located within 2 m",
                    "reach_m": 0.8,
                }
            ],
        )
        claim = executor_step(
            ctx,
            step_event(
                "ground_target",
                data={
                    "subgoal_complete": True,
                    "candidate_id": "cand_1",
                    "object_position": [2.0, 0.0, 0.0],
                },
                evidence=[("observation", "obs_1")],
            ),
        )
        self.assertEqual(claim["claimed"], "st_1")
        self.assertEqual(claim["evidence_missing"], ["reached"])

        far = executor_step(
            ctx,
            step_event(
                "approach_candidate",
                turn=2,
                data={
                    "subgoal_complete": True,
                    "outcome": "partial",
                    "remaining_m": 1.4,
                    "target_position": [2.0, 0.0, 0.0],
                },
                evidence=[("observation", "obs_2")],
            ),
        )
        self.assertEqual(far["completed"], "")
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_CLAIMED_DONE)

        close = executor_step(
            ctx,
            step_event(
                "approach_candidate",
                turn=3,
                data={
                    "outcome": "partial",
                    "remaining_m": 0.6,
                    "target_position": [2.0, 0.0, 0.0],
                },
                evidence=[("observation", "obs_3")],
            ),
        )
        self.assertEqual(close["completed"], "st_1")
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_DONE)

    def test_a_movement_result_with_the_object_place_closes_the_step(self):
        registry, ctx = plan_registry(), make_context()
        self._door_plan(registry, ctx)
        outcome = executor_step(ctx, located_move_event())

        self.assertEqual(outcome["completed"], "st_1")
        self.assertEqual(outcome["claimed"], "")
        self.assertEqual(outcome["active"], "st_2")
        first, second = subtasks_of(ctx)
        self.assertEqual(first["status"], SUBTASK_DONE)
        self.assertEqual(first["reason"], "done_when_met")
        self.assertEqual(second["status"], SUBTASK_ACTIVE)

    def test_a_claimed_step_is_completed_when_the_missing_evidence_arrives(self):
        registry, ctx = plan_registry(), make_context()
        self._door_plan(registry, ctx)
        executor_step(ctx, ray_only_search_event())
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_CLAIMED_DONE)

        outcome = executor_step(ctx, located_move_event(turn=2))
        self.assertEqual(outcome["completed"], "st_1")
        self.assertEqual(
            outcome["transitions"][0],
            {"id": "st_1", "from": SUBTASK_CLAIMED_DONE, "to": SUBTASK_DONE,
             "reason": "done_when_met", "turn": 2},
        )
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_DONE)

    def test_a_claimed_step_is_promoted_without_a_new_completion_signal(self):
        """ep_007: the missing evidence arrived on an ordinary call - re-running
        the verdict on every charged call removes the lagging status."""
        registry, ctx = plan_registry(), make_context()
        self._door_plan(registry, ctx)
        executor_step(ctx, ray_only_search_event())
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_CLAIMED_DONE)

        outcome = executor_step(ctx, located_move_event(turn=2, signal=False))
        self.assertEqual(outcome["completed"], "st_1")
        self.assertEqual(outcome["claimed"], "")
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_DONE)
        self.assertEqual(subtasks_of(ctx)[0]["reason"], "done_when_met")

    def test_a_verified_claim_needs_a_passing_verification(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [{"id": "st_1", "goal": "confirm the box",
              "done_when": "the box is confirmed as the target (verify_goal verified=true)"}],
        )
        claim = executor_step(
            ctx,
            step_event("search_target", data={"success": True},
                       evidence=[("observation", "obs_9")]),
        )
        self.assertEqual(claim["claimed"], "st_1")
        self.assertIn("confirmation", claim["evidence_missing"])

        done = executor_step(
            ctx,
            step_event("verify_goal", turn=2, data={"verified": True},
                       evidence=[("map", "goal_check")]),
        )
        self.assertEqual(done["claimed"], "")
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_DONE)

    def test_the_real_target_done_when_is_satisfied_by_identity_place_and_arrival(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [{"id": "st_1", "goal": "confirm the box", "done_when": TARGET_DONE_WHEN}],
        )
        outcome = executor_step(ctx, located_move_event())
        self.assertEqual(outcome["completed"], "st_1")
        self.assertEqual(outcome["claimed"], "")
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_DONE)

    def test_budget_exhaustion_with_complete_evidence_marks_the_step_done(self):
        registry, ctx = plan_registry(), make_context()
        self._door_plan(registry, ctx, budget=1)
        first = executor_step(ctx, located_move_event(turn=1, signal=False))
        self.assertEqual(first["charged"], "st_1")
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_ACTIVE)

        over = executor_step(ctx, located_move_event(turn=2, signal=False))
        self.assertEqual(over["completed"], "st_1")
        self.assertEqual(over["failed"], "")
        self.assertEqual(over["budget_with_evidence"], "st_1")
        self.assertEqual(over["budget_exceeded"], [])
        subtask = subtasks_of(ctx)[0]
        self.assertEqual(subtask["status"], SUBTASK_DONE)
        self.assertEqual(subtask["reason"], "budget_with_evidence")
        # the completed step hands the plan on exactly as a signal-driven done does
        self.assertEqual(subtasks_of(ctx)[1]["status"], SUBTASK_ACTIVE)

    def test_budget_exhaustion_without_the_evidence_still_fails(self):
        registry, ctx = plan_registry(), make_context()
        self._door_plan(registry, ctx, budget=1)
        executor_step(
            ctx, step_event("observe", turn=1, evidence=[("observation", "obs_1")])
        )
        over = executor_step(
            ctx, step_event("observe", turn=2, evidence=[("observation", "obs_2")])
        )
        self.assertEqual(over["failed"], "st_1")
        self.assertEqual(over["completed"], "")
        self.assertEqual(over["budget_exceeded"], ["st_1"])
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_FAILED)
        self.assertEqual(subtasks_of(ctx)[0]["reason"], "budget_exceeded:tool_calls")

    def test_budget_with_evidence_rejects_a_place_outside_the_declared_place(self):
        """ep_011: a located mark is not an arrival at *this* stage's place."""
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [
                {
                    "id": "st_1",
                    "goal": "find the wooden door",
                    "done_when": DOOR_DONE_WHEN,
                    "goal_position": [10.0, 0.0, 0.0],
                    "locate_m": 2.0,
                    "budget": {"tool_calls": 1},
                }
            ],
        )
        far = step_event(
            "approach_candidate",
            turn=1,
            data={"target_position": [0.0, 0.0, 0.0], "moved_m": 3.0},
            evidence=[("observation", "obs_11")],
        )
        first = executor_step(ctx, far)
        self.assertEqual(first["charged"], "st_1")
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_ACTIVE)

        over = executor_step(ctx, step_event(
            "approach_candidate",
            turn=2,
            data={"target_position": [0.0, 0.0, 0.0], "moved_m": 3.0},
            evidence=[("observation", "obs_12")],
        ))
        self.assertEqual(over["budget_with_evidence"], "")
        self.assertEqual(over["completed"], "")
        self.assertEqual(over["failed"], "st_1")
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_FAILED)
        self.assertEqual(subtasks_of(ctx)[0]["reason"], "budget_exceeded:tool_calls")

    def test_budget_with_evidence_accepts_a_place_inside_locate_m(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [
                {
                    "id": "st_1",
                    "goal": "find the wooden door",
                    "done_when": DOOR_DONE_WHEN,
                    "goal_position": [10.0, 0.0, 0.0],
                    "locate_m": 2.0,
                    "budget": {"tool_calls": 1},
                }
            ],
        )
        executor_step(
            ctx,
            step_event(
                "approach_candidate",
                turn=1,
                data={"target_position": [9.0, 0.0, 0.0], "moved_m": 3.0},
                evidence=[("observation", "obs_11")],
            ),
        )
        over = executor_step(
            ctx,
            step_event(
                "approach_candidate",
                turn=2,
                data={
                    "outcome": "arrived",
                    "task_status": "arrived",
                    "remaining_m": 0.4,
                    "target_position": [8.5, 0.0, 0.0],
                    "moved_m": 1.0,
                },
                evidence=[("observation", "obs_12")],
            ),
        )
        self.assertEqual(over["budget_with_evidence"], "st_1")
        self.assertEqual(over["failed"], "")
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_DONE)
        self.assertEqual(subtasks_of(ctx)[0]["reason"], "budget_with_evidence")

    def test_a_signal_with_a_place_outside_the_declared_place_is_claimed(self):
        """The signal cannot close the step until a place inside ``locate_m``
        is reported; that later result need not carry a completion signal."""
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [
                {
                    "id": "st_1",
                    "goal": "find the wooden door",
                    "done_when": DOOR_DONE_WHEN,
                    "goal_position": [10.0, 0.0, 0.0],
                    "locate_m": 2.0,
                },
                {"id": "st_2", "goal": "find the cabinet", "depends_on": ["st_1"]},
            ],
        )
        claim = executor_step(
            ctx,
            step_event(
                "approach_candidate",
                data={
                    "subgoal_complete": True,
                    "target_position": [0.0, 0.0, 0.0],
                    "moved_m": 3.0,
                },
                evidence=[("observation", "obs_11")],
            ),
        )
        self.assertEqual(claim["claimed"], "st_1")
        self.assertEqual(claim["completed"], "")
        self.assertIn("located", claim["evidence_missing"])
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_CLAIMED_DONE)
        self.assertEqual(subtasks_of(ctx)[1]["status"], SUBTASK_PENDING)

        done = executor_step(
            ctx,
            step_event(
                "approach_candidate",
                turn=2,
                data={
                    "outcome": "arrived",
                    "task_status": "arrived",
                    "remaining_m": 0.4,
                    "target_position": [9.5, 0.0, 0.0],
                    "moved_m": 1.0,
                },
                evidence=[("observation", "obs_12")],
            ),
        )
        self.assertEqual(done["completed"], "st_1")
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_DONE)
        self.assertEqual(subtasks_of(ctx)[1]["status"], SUBTASK_ACTIVE)

    def test_the_done_when_distance_is_the_tolerance_when_locate_m_is_absent(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [
                {
                    "id": "st_1",
                    "goal": "find the box",
                    "done_when": "the box is located within 1.5 m and reached",
                    "goal_position": [10.0, 0.0, 0.0],
                }
            ],
        )
        verdict = evidence_verdict(subtasks_of(ctx)[0])
        self.assertEqual(verdict["locate_m"], 1.5)

        # 2.0 m away is outside the done_when's 1.5 m: claimed, not done
        claim = executor_step(
            ctx,
            step_event(
                "approach_candidate",
                data={
                    "subgoal_complete": True,
                    "target_position": [8.0, 0.0, 0.0],
                    "moved_m": 2.0,
                },
                evidence=[("observation", "obs_1")],
            ),
        )
        self.assertEqual(claim["claimed"], "st_1")
        self.assertIn("located", claim["evidence_missing"])

        # 1.4 m away, reported by a call without a completion signal, closes it
        done = executor_step(
            ctx,
            step_event(
                "approach_candidate",
                turn=2,
                data={
                    "outcome": "arrived",
                    "task_status": "arrived",
                    "remaining_m": 0.4,
                    "target_position": [8.6, 0.0, 0.0],
                    "moved_m": 0.5,
                },
                evidence=[("observation", "obs_2")],
            ),
        )
        self.assertEqual(done["completed"], "st_1")
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_DONE)

    def test_unscoped_located_evidence_is_marked_weak(self):
        registry, ctx = plan_registry(), make_context()
        self._door_plan(registry, ctx)
        outcome = executor_step(ctx, located_move_event())

        self.assertEqual(outcome["completed"], "st_1")
        self.assertTrue(outcome["evidence_weak"])
        subtask = subtasks_of(ctx)[0]
        verdict = evidence_verdict(subtask)
        self.assertTrue(verdict["ok"])
        self.assertTrue(verdict["evidence_weak"])
        self.assertEqual(verdict["goal_position"], [])
        self.assertTrue(subtask["_evidence_weak"])
        data = registry.call("get_plan", {}, ctx).data
        self.assertTrue(data["subtasks"][0]["evidence_weak"])

    def test_a_declared_done_without_evidence_is_claimed_but_hands_the_plan_on(self):
        registry, ctx = plan_registry(), make_context()
        self._door_plan(registry, ctx)
        result = registry.call("update_plan", {"done": [1]}, ctx)
        self.assertEqual(result.status, STATUS_OK)
        data = result.data
        self.assertEqual(data["done"], [])  # done is evidence-bound
        self.assertEqual(data["claimed_subtasks"][0]["id"], "st_1")
        # the model's declaration still moves the plan to the next step
        self.assertEqual(data["current_step"], 2)
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_CLAIMED_DONE)
        self.assertEqual(subtasks_of(ctx)[1]["status"], SUBTASK_ACTIVE)
        self.assertIn("claimed_done", " ".join(data["progress_notes"]))

    def test_plan_state_keeps_the_derived_progress_in_step_with_the_subtasks(self):
        """Another component may mark a sub-task done directly (the long-horizon
        driver does, from its own evidence chain); the legacy ``done`` /
        ``current_step`` view follows it on the next read."""
        registry, ctx = plan_registry(), make_context()
        plan_with(registry, ctx, ["scan the corridor", "check the cabinet"])
        record = plan_state(ctx)
        record["subtasks"][0]["status"] = SUBTASK_DONE
        state = plan_state(ctx)
        self.assertEqual(state["done"], [1])
        self.assertEqual(state["current_step"], 2)

    def test_the_evidence_marks_survive_a_resume(self):
        registry, ctx = plan_registry(), make_context(episode_id="ep_1")
        self._door_plan(registry, ctx, budget=1)
        executor_step(ctx, located_move_event(turn=1, signal=False))
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_ACTIVE)

        ctx.episode_id = "ep_2"
        self.assertIsNotNone(resume_plan(ctx, "ep_2", "find the wooden door"))
        # the carried evidence still satisfies done_when, so the exhausted budget
        # closes the step instead of failing it
        over = executor_step(ctx, step_event("observe", turn=1))
        self.assertEqual(over["completed"], "st_1")
        self.assertEqual(subtasks_of(ctx)[0]["reason"], "budget_with_evidence")


class FailedSubtaskDeadlockTest(unittest.TestCase):
    """A ``failed`` current step must not hold later evidence hostage.

    ep_011: ``st_1`` failed (``budget_exceeded:tool_calls``) and stayed the
    current sub-task, so every later result - including the object and target
    evidence of ``st_2``/``st_3`` - was charged to the failed step and those
    steps stayed ``pending`` with ``evidence_complete=true``. The executor now
    redirects the result to the first unfinished step its evidence can serve,
    and a failed step is promoted when the missing evidence arrives later.
    """

    def stage_plan(self, registry, ctx):
        plan_with(
            registry,
            ctx,
            [
                {
                    "id": "st_1",
                    "goal": "find the door",
                    "done_when": "the door is located within 2 m",
                    "goal_position": [10.0, 0.0, 0.0],
                    "locate_m": 1.0,
                    "budget": {"tool_calls": 1},
                },
                {
                    "id": "st_2",
                    "goal": "find the cabinet",
                    "done_when": "the cabinet is located within 2 m",
                    "goal_position": [1.0, 2.0, 3.0],
                    "locate_m": 1.0,
                    "depends_on": ["st_1"],
                },
                {
                    "id": "st_3",
                    "goal": "find the cardboard box",
                    "done_when": "the box is located within 2 m",
                    "goal_position": [5.0, 5.0, 5.0],
                    "locate_m": 1.0,
                    "depends_on": ["st_2"],
                },
            ],
        )

    def test_later_evidence_is_charged_to_the_first_feasible_open_step(self):
        registry, ctx = plan_registry(), make_context()
        self.stage_plan(registry, ctx)

        first = executor_step(ctx, step_event("approach_candidate", turn=1))
        self.assertEqual(first["charged"], "st_1")
        self.assertEqual(first["failed"], "")
        failed = executor_step(ctx, step_event("approach_candidate", turn=2))
        self.assertEqual(failed["failed"], "st_1")
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_FAILED)

        object_evidence = executor_step(
            ctx,
            step_event(
                "ground_target",
                turn=3,
                data={"candidate_id": "cand_2", "object_position": [1.0, 2.0, 3.0]},
                evidence=[("observation", "obs_2")],
            ),
        )
        self.assertEqual(object_evidence["charged"], "st_2")
        self.assertNotEqual(object_evidence["charged"], "st_1")
        second = subtasks_of(ctx)[1]
        self.assertEqual(second["status"], SUBTASK_ACTIVE)
        self.assertTrue(evidence_verdict(second)["ok"])
        self.assertIn("observation:obs_2", second["evidence"])

        target_evidence = executor_step(
            ctx,
            step_event(
                "ground_target",
                turn=4,
                data={"candidate_id": "cand_3", "object_position": [5.0, 5.0, 5.0]},
                evidence=[("observation", "obs_3")],
            ),
        )
        self.assertEqual(target_evidence["charged"], "st_3")
        third = subtasks_of(ctx)[2]
        self.assertEqual(third["status"], SUBTASK_ACTIVE)
        self.assertTrue(evidence_verdict(third)["ok"])
        # the lagging rows of the ep_011 plan-vs-evidence audit are gone: the
        # later stages are no longer left ``pending`` with complete evidence
        self.assertEqual(
            [item["status"] for item in subtasks_of(ctx)],
            [SUBTASK_FAILED, SUBTASK_ACTIVE, SUBTASK_ACTIVE],
        )

    def test_a_failed_step_is_promoted_when_its_missing_evidence_arrives(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [
                {
                    "id": "st_1",
                    "goal": "find the wooden door",
                    "done_when": "the door is located within 2 m",
                    "goal_position": [2.0, 0.0, 0.0],
                    "locate_m": 1.0,
                    "budget": {"tool_calls": 1},
                }
            ],
        )
        executor_step(ctx, step_event("observe", turn=1))
        failed = executor_step(ctx, step_event("observe", turn=2))
        self.assertEqual(failed["failed"], "st_1")
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_FAILED)

        # no completion signal: only the place the failed leg never supplied
        recovered = executor_step(
            ctx,
            step_event(
                "ground_target",
                turn=3,
                data={"candidate_id": "cand_1", "object_position": [2.1, 0.0, 0.0]},
                evidence=[("observation", "obs_2")],
            ),
        )
        self.assertEqual(recovered["charged"], "st_1")
        self.assertIn("st_1", recovered["promoted"])
        self.assertEqual(recovered["completed"], "st_1")
        self.assertEqual(subtasks_of(ctx)[0]["status"], SUBTASK_DONE)
        self.assertEqual(subtasks_of(ctx)[0]["reason"], "done_when_met")

    def test_a_failed_step_with_open_predecessors_is_claimed_not_done(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [
                {"id": "st_1", "goal": "scan the corridor"},
                {
                    "id": "st_2",
                    "goal": "find the cabinet",
                    "done_when": "the cabinet is located within 2 m",
                    "goal_position": [1.0, 2.0, 3.0],
                    "locate_m": 1.0,
                    "budget": {"tool_calls": 1},
                    "depends_on": ["st_1"],
                },
            ],
        )
        # st_1 is active and open; st_2 runs out of budget and fails
        executor_step(ctx, step_event("look", turn=1, data={"subtask_id": "st_2"}))
        over = executor_step(
            ctx, step_event("look", turn=2, data={"subtask_id": "st_2"})
        )
        self.assertEqual(over["failed"], "st_2")
        self.assertEqual(subtasks_of(ctx)[1]["status"], SUBTASK_FAILED)

        # the evidence arrives while st_1 is still open: a claim, not a done
        recovered = executor_step(
            ctx,
            step_event(
                "ground_target",
                turn=3,
                data={
                    "subtask_id": "st_2",
                    "candidate_id": "cand_2",
                    "object_position": [1.0, 2.0, 3.0],
                },
                evidence=[("observation", "obs_3")],
            ),
        )
        self.assertEqual(recovered["charged"], "st_2")
        self.assertEqual(subtasks_of(ctx)[1]["status"], SUBTASK_CLAIMED_DONE)
        self.assertFalse(recovered["promoted"])

        # once the predecessor closes, the failed-and-claimed step is promoted
        closed = executor_step(
            ctx,
            step_event(
                "look",
                turn=4,
                data={"subgoal_complete": True},
                evidence=[("observation", "obs_1")],
            ),
        )
        self.assertEqual(closed["completed"], "st_1")
        self.assertIn("st_2", closed["promoted"])
        self.assertEqual(subtasks_of(ctx)[1]["status"], SUBTASK_DONE)


class SubtaskFailureBranchTest(unittest.TestCase):
    def test_the_branch_of_the_current_subtask_is_offered(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [{"goal": "fly to vp_3", "failure_branch": ["go_to_viewpoint vp_9", "check_path"]}],
        )
        info = subtask_failure_branch(ctx)
        self.assertEqual(info["id"], "st_1")
        self.assertEqual(info["goal"], "fly to vp_3")
        self.assertEqual(info["status"], SUBTASK_ACTIVE)
        self.assertEqual(info["branch"], ["go_to_viewpoint vp_9", "check_path"])
        self.assertEqual(current_subtask(ctx)["id"], "st_1")

    def test_no_branch_and_no_plan_answer_none(self):
        registry, ctx = plan_registry(), make_context()
        self.assertIsNone(subtask_failure_branch(ctx))
        self.assertIsNone(current_subtask(ctx))
        plan_with(registry, ctx, ["scan the corridor"])
        self.assertIsNone(subtask_failure_branch(ctx))


class SubtaskProgressTextTest(unittest.TestCase):
    def test_no_plan_says_so(self):
        self.assertEqual(subtask_progress(make_context()), "no plan")

    def test_the_line_names_the_current_subtask_and_its_progress(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [{"goal": "scan the corridor", "budget": 4}, {"goal": "check the cabinet"}],
        )
        executor_step(
            ctx,
            step_event(
                "scan",
                data={"subgoal_complete": True},
                evidence=[("observation", "obs_1")],
            ),
        )
        text = subtask_progress(ctx)
        self.assertIn("plan_1", text)
        self.assertIn("st 2/2", text)
        self.assertIn("current=st_2[active]", text)
        self.assertIn("check the cabinet", text)
        self.assertIn("(1 done)", text)

    def test_a_blocked_current_subtask_shows_its_failure_branch(self):
        registry, ctx = plan_registry(), make_context()
        plan_with(
            registry,
            ctx,
            [{"goal": "fly to vp_3", "failure_branch": ["go_to_viewpoint vp_9"]}],
        )
        executor_step(
            ctx, step_event("go_to_viewpoint", status=STATUS_BLOCKED, data={"outcome": "blocked"})
        )
        text = subtask_progress(ctx)
        self.assertIn("current=st_1[blocked]", text)
        self.assertIn("failure_branch=go_to_viewpoint vp_9", text)


class PlanResumeTest(unittest.TestCase):
    """An interrupted episode leaves its plan in the context; the next run of
    the same context continues from it instead of starting over (P4)."""

    def test_resume_adopts_an_interrupted_plan(self):
        registry, ctx = plan_registry(), make_context(episode_id="ep_1")
        plan_with(
            registry,
            ctx,
            [{"goal": "scan the corridor"}, {"goal": "check the cabinet"}],
        )
        executor_step(ctx, step_event("scan", data={"subgoal_complete": True}, evidence=[("observation", "obs_4")]))

        ctx.episode_id = "ep_2"
        self.assertIsNone(plan_state(ctx))  # the record belongs to ep_1
        record = resume_plan(ctx, "ep_2", "find the red book")
        self.assertIsNotNone(record)
        self.assertEqual(plan_state(ctx), record)
        self.assertEqual(record["plan_id"], "plan_1")
        self.assertEqual(record["resumed_from"], "ep_1")
        self.assertEqual(record["resume_count"], 1)
        # the execution state survives: st_1 done, st_2 active, the spent calls kept
        first, second = record["subtasks"]
        self.assertEqual(first["status"], SUBTASK_DONE)
        self.assertEqual(first["evidence"], ["observation:obs_4"])
        self.assertEqual(second["status"], SUBTASK_ACTIVE)
        self.assertEqual(second["spent"]["tool_calls"], 0)
        # the resumed plan is the plan of the new episode
        self.assertEqual(registry.call("get_plan", {}, ctx).data["plan_id"], "plan_1")
        self.assertIn("st 2/2", subtask_progress(ctx))

    def test_resume_keeps_the_budget_consumption_of_the_first_run(self):
        registry, ctx = plan_registry(), make_context(episode_id="ep_1")
        plan_with(registry, ctx, [{"goal": "scan the corridor"}])
        executor_step(ctx, step_event("scan", turn=1))
        ctx.episode_id = "ep_2"
        record = resume_plan(ctx, "ep_2", "find the red book")
        self.assertEqual(record["subtasks"][0]["spent"], {"tool_calls": 1, "turns": 1})
        # the turn counter of the new run starts fresh
        self.assertEqual(record["executor"]["last_turn"], 0)
        executor_step(ctx, step_event("scan", turn=1))
        self.assertEqual(record["subtasks"][0]["spent"]["turns"], 2)

    def test_resume_refuses_a_plan_written_for_another_instruction(self):
        registry, ctx = plan_registry(), make_context(episode_id="ep_1")
        registry.call("update_plan", {"steps": ["scan the corridor"]}, ctx)
        plan_state(ctx)["instruction"] = "find the red book"
        ctx.episode_id = "ep_2"
        self.assertIsNone(resume_plan(ctx, "ep_2", "count the windows"))
        self.assertIsNone(plan_state(ctx))

    def test_resume_refuses_a_plan_that_is_already_finished(self):
        """A completed plan belongs to the episode that finished it: adopting it
        would tell the next run that the task is over."""
        registry, ctx = plan_registry(), make_context(episode_id="ep_1")
        plan_with(registry, ctx, ["scan the corridor", "check the cabinet"])
        registry.call("update_plan", {"done": [1, 2]}, ctx)
        ctx.episode_id = "ep_2"
        self.assertIsNone(resume_plan(ctx, "ep_2", "find the red book"))
        self.assertIsNone(plan_state(ctx))

    def test_resume_needs_a_plan(self):
        self.assertIsNone(resume_plan(make_context(), "ep_2", "go"))

    def test_resume_of_the_current_episode_is_a_no_op(self):
        registry, ctx = plan_registry(), make_context(episode_id="ep_1")
        plan_with(registry, ctx, ["scan the corridor"])
        record = resume_plan(ctx, "ep_1", "go")
        self.assertEqual(record["resume_count"], 0)
        self.assertEqual(record["resumed_from"], "")


if __name__ == "__main__":
    unittest.main()
