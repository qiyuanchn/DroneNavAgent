"""Tests for the semantic navigation service (workstream C).

No VLM, GPU, world or network: the service only sees a scripted
``FakeRegistry`` that implements ``names`` / ``get`` / ``call``, exactly the
surface the contract allows it to use.
"""

from __future__ import annotations

import pytest

from vlm_nav import semantic
from vlm_nav.semantic import SemanticNavigationService, compose_instruction
from vlm_nav.tools import (
    STATUS_BLOCKED,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_OK,
    ToolContext,
    ToolResult,
)

COMMON_KEYS = {
    "ok", "tool", "status", "candidate_id", "task_id", "arrived", "accepted",
    "verified", "calls", "error", "data",
}


class FakeRegistry:
    """Scripted registry: ``names``/``get``/``call`` only."""

    def __init__(self, *tools: str) -> None:
        self._names = set(tools)
        self._scripted = {name: [] for name in tools}
        self._defaults = {}
        self.calls: list[dict] = []

    def names(self) -> list[str]:
        return sorted(self._names)

    def get(self, name: str):
        if name not in self._names:
            raise KeyError(name)
        return {"name": name}

    def script(self, name: str, *results) -> "FakeRegistry":
        self._names.add(name)
        self._scripted.setdefault(name, []).extend(results)
        return self

    def default(self, name: str, factory) -> "FakeRegistry":
        self._names.add(name)
        self._defaults[name] = factory
        return self

    def call(self, name: str, args: dict, ctx):
        self.calls.append({"tool": name, "args": dict(args or {})})
        if name not in self._names:
            raise KeyError(f"unknown tool: {name}")
        queue = self._scripted.setdefault(name, [])
        if queue:
            item = queue.pop(0)
        elif name in self._defaults:
            item = self._defaults[name]
        else:
            raise AssertionError(f"unexpected call: {name} {args}")
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(name, args, ctx)
        return item

    @property
    def tools(self) -> list[str]:
        return [call["tool"] for call in self.calls]


def make_ctx() -> ToolContext:
    return ToolContext(world=object(), episode_id="ep_semantic")


def assert_contract(out: dict) -> None:
    assert set(out) == COMMON_KEYS
    assert isinstance(out["ok"], bool)
    assert isinstance(out["tool"], str)
    assert isinstance(out["status"], str)
    assert isinstance(out["candidate_id"], str)
    assert isinstance(out["task_id"], str)
    assert out["arrived"] in (True, False)
    assert out["accepted"] in (True, False)
    assert out["verified"] in (True, False)
    assert isinstance(out["calls"], int) and out["calls"] >= 0
    assert isinstance(out["error"], str)
    assert isinstance(out["data"], dict)


def found(candidate: str = "cand_1", confirmed: bool = True, **extra) -> ToolResult:
    data = {
        "candidate_id": candidate,
        "confirmed": confirmed,
        "reason": "found",
        **extra,
    }
    return ToolResult(STATUS_OK, data)


def search_empty(**extra) -> ToolResult:
    data = {"candidate_id": "", "confirmed": False, "reason": "not_found", **extra}
    return ToolResult(STATUS_EMPTY, data)


def movement(
    task_id: str = "task_1",
    task_status: str = "running",
    arrived: bool = False,
    **extra,
) -> ToolResult:
    data = {
        "task_id": task_id,
        "task_status": task_status,
        "arrived": arrived,
        "outcome": task_status,
        **extra,
    }
    status = STATUS_OK if task_status in ("running", "arrived") else STATUS_BLOCKED
    return ToolResult(status, data)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_success_returns_candidate_and_composes_instruction():
    registry = FakeRegistry().script("search_target", found())
    service = SemanticNavigationService(registry, max_retries=2)
    out = service.search(
        make_ctx(), "木椅", relation="窗边", max_steps=4, max_turns=2
    )

    assert_contract(out)
    assert out["ok"] is True
    assert out["tool"] == "search_target"
    assert out["status"] == STATUS_OK
    assert out["candidate_id"] == "cand_1"
    assert out["accepted"] is True  # confirmed=true in the tool payload
    assert out["arrived"] is False
    assert out["verified"] is False
    assert out["calls"] == 1
    assert out["error"] == ""
    assert out["data"]["confirmed"] is True
    assert registry.calls[0]["args"]["instruction"] == "窗边的木椅"
    assert registry.calls[0]["args"]["max_steps"] == 4
    assert registry.calls[0]["args"]["budget_turns"] == 2


def test_search_unconfirmed_candidate_is_returned_but_not_accepted():
    registry = FakeRegistry().script("search_target", found(confirmed=False))
    out = SemanticNavigationService(registry).search(make_ctx(), "chair")

    assert_contract(out)
    assert out["ok"] is True
    assert out["candidate_id"] == "cand_1"
    # the id is usable for approach, but identity is not confirmed
    assert out["accepted"] is False
    assert out["data"]["confirmed"] is False


def test_search_follows_next_action_then_retries_with_original_instruction():
    registry = FakeRegistry()
    registry.script(
        "search_target",
        search_empty(next_action={"tool": "observe", "args": {"mode": "forward"}}),
        found(),
    )
    registry.script("observe", ToolResult(STATUS_OK, {"observation_id": "obs_1"}))
    out = SemanticNavigationService(registry).search(
        make_ctx(), "chair", relation="by the window", max_steps=5
    )

    assert_contract(out)
    assert out["ok"] is True
    assert out["calls"] == 3
    assert registry.tools == ["search_target", "observe", "search_target"]
    assert registry.calls[1]["args"] == {"mode": "forward"}
    # both searches carry the original, composed instruction
    expected = compose_instruction("chair", "by the window")
    assert registry.calls[0]["args"]["instruction"] == expected
    assert registry.calls[2]["args"]["instruction"] == expected


def test_search_follows_strategy_change_option():
    registry = FakeRegistry()
    registry.script(
        "search_target",
        search_empty(
            strategy_change={
                "kind": "change_strategy",
                "options": [
                    {"tool": "propose_viewpoints", "args": {"count": 3}},
                    {"tool": "observe", "args": {"mode": "forward"}},
                ],
            }
        ),
        found(candidate="cand_2"),
    )
    registry.script(
        "propose_viewpoints", ToolResult(STATUS_OK, {"viewpoint_id": "vp_1"})
    )
    out = SemanticNavigationService(registry).search(make_ctx(), "chair")

    assert out["ok"] is True
    assert registry.tools == ["search_target", "propose_viewpoints", "search_target"]
    assert registry.calls[1]["args"] == {"count": 3}


def test_search_ignores_unregistered_recovery_tool():
    registry = FakeRegistry().script(
        "search_target",
        search_empty(next_action={"tool": "teleport", "args": {}}),
    )
    out = SemanticNavigationService(registry).search(make_ctx(), "chair")

    assert_contract(out)
    assert out["ok"] is False
    assert out["status"] == STATUS_EMPTY
    assert out["calls"] == 1
    assert registry.tools == ["search_target"]
    assert out["error"] == "not_found"


def test_search_recovery_is_bounded_by_max_retries():
    registry = FakeRegistry()
    registry.default(
        "search_target",
        lambda name, args, ctx: search_empty(
            next_action={"tool": "observe", "args": {"mode": "forward"}}
        ),
    )
    registry.default(
        "observe", lambda name, args, ctx: ToolResult(STATUS_OK, {"obs": "obs_1"})
    )
    out = SemanticNavigationService(registry, max_retries=1).search(
        make_ctx(), "chair"
    )

    assert_contract(out)
    assert out["ok"] is False
    # search -> recovery -> retry search; the second proposal is not followed
    assert out["calls"] == 3
    assert registry.tools == ["search_target", "observe", "search_target"]


def test_search_requires_a_target_or_instruction():
    registry = FakeRegistry()
    out = SemanticNavigationService(registry).search(make_ctx(), "   ")

    assert_contract(out)
    assert out["ok"] is False
    assert "target or instruction" in out["error"]
    assert registry.calls == []


def test_compose_instruction_rules():
    assert compose_instruction("木椅", "窗边") == "窗边的木椅"
    assert compose_instruction("木椅", "窗边的") == "窗边的木椅"
    assert compose_instruction("chair", "next to the window") == (
        "chair next to the window"
    )
    assert compose_instruction("chair by the window", "by the window") == (
        "chair by the window"
    )
    assert compose_instruction("chair", "") == "chair"


# ---------------------------------------------------------------------------
# approach
# ---------------------------------------------------------------------------


def test_approach_loops_task_id_until_arrived():
    registry = FakeRegistry().script(
        "approach_candidate",
        movement("task_1", "running"),
        movement("task_1", "running"),
        movement("task_1", "arrived", arrived=True),
    )
    out = SemanticNavigationService(registry).approach(
        make_ctx(), "cand_1", standoff_m=0.8
    )

    assert_contract(out)
    assert out["ok"] is True
    assert out["arrived"] is True
    assert out["accepted"] is False
    assert out["tool"] == "approach_candidate"
    assert out["status"] == "arrived"
    assert out["task_id"] == "task_1"
    assert out["calls"] == 3
    assert registry.calls[0]["args"] == {"candidate_id": "cand_1", "standoff_m": 0.8}
    assert registry.calls[1]["args"] == {"task_id": "task_1"}
    assert registry.calls[2]["args"] == {"task_id": "task_1"}


def test_approach_blocked_follows_next_action_once_then_retries():
    registry = FakeRegistry()
    registry.script(
        "approach_candidate",
        movement(
            "task_1",
            "blocked",
            next_action={
                "tool": "go_to_viewpoint",
                "args": {"viewpoint_id": "vp_9"},
            },
        ),
        movement("task_2", "arrived", arrived=True),
    )
    registry.script(
        "go_to_viewpoint", movement("task_9", "arrived", arrived=True)
    )
    out = SemanticNavigationService(registry).approach(make_ctx(), "cand_1")

    assert_contract(out)
    assert out["ok"] is True
    assert out["arrived"] is True
    assert out["task_id"] == "task_2"
    assert registry.tools == [
        "approach_candidate", "go_to_viewpoint", "approach_candidate",
    ]
    assert registry.calls[1]["args"] == {"viewpoint_id": "vp_9"}
    # the retry starts the approach again rather than continuing a blocked task
    assert registry.calls[2]["args"]["candidate_id"] == "cand_1"


def test_approach_blocked_without_registered_recovery_fails():
    registry = FakeRegistry().script(
        "approach_candidate",
        movement(
            "task_1",
            "blocked",
            reason="repeating_move",
            next_action={"tool": "teleport", "args": {}},
        ),
    )
    out = SemanticNavigationService(registry).approach(make_ctx(), "cand_1")

    assert_contract(out)
    assert out["ok"] is False
    assert out["arrived"] is False
    assert out["status"] == "blocked"
    assert out["task_id"] == "task_1"
    assert out["calls"] == 1
    assert "repeating_move" in out["error"]


def test_approach_is_bounded_by_max_legs():
    registry = FakeRegistry().default(
        "approach_candidate",
        lambda name, args, ctx: movement("task_1", "running"),
    )
    out = SemanticNavigationService(registry).approach(
        make_ctx(), "cand_1", max_legs=2
    )

    assert_contract(out)
    assert out["ok"] is False
    assert out["arrived"] is False
    assert out["status"] == "running"
    assert out["task_id"] == "task_1"
    assert out["calls"] == 2
    assert "max_legs_exceeded" in out["error"]


def test_approach_requires_a_candidate():
    registry = FakeRegistry()
    out = SemanticNavigationService(registry).approach(make_ctx())

    assert_contract(out)
    assert out["ok"] is False
    assert "candidate" in out["error"]
    assert out["calls"] == 0
    assert registry.calls == []


def test_approach_uses_scoped_candidate_when_omitted():
    ctx = make_ctx()
    ctx.extra["task_scope"] = {"candidate_id": "cand_7"}
    registry = FakeRegistry().script(
        "approach_candidate", movement("t", "arrived", arrived=True)
    )
    out = SemanticNavigationService(registry).approach(ctx)

    assert out["ok"] is True
    assert out["candidate_id"] == "cand_7"
    assert registry.calls[0]["args"]["candidate_id"] == "cand_7"


# ---------------------------------------------------------------------------
# confirm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verdict", "ok", "status"),
    [
        ("accepted", True, "accepted"),
        ("uncertain", False, "uncertain"),
        ("rejected", False, "rejected"),
    ],
)
def test_confirm_preserves_verdict(verdict, ok, status):
    registry = FakeRegistry().script(
        "confirm_candidate",
        ToolResult(
            STATUS_OK if verdict == "accepted" else STATUS_BLOCKED,
            {
                "candidate_id": "cand_1",
                "verdict": verdict,
                "missing": [] if verdict == "accepted" else ["identity"],
                "reason": f"verdict_{verdict}",
            },
        ),
    )
    out = SemanticNavigationService(registry).confirm(
        make_ctx(), "cand_1", max_views=3
    )

    assert_contract(out)
    assert out["ok"] is ok
    assert out["accepted"] is ok
    assert out["status"] == status
    assert out["calls"] == 1
    assert out["data"]["verdict"] == verdict
    assert registry.calls[0]["args"] == {"candidate_id": "cand_1", "max_views": 3}
    if ok:
        assert out["error"] == ""
    else:
        assert f"verdict_{verdict}" in out["error"]


def test_confirm_does_not_retry_a_rejected_candidate():
    registry = FakeRegistry()
    registry.script(
        "confirm_candidate",
        ToolResult(
            STATUS_BLOCKED,
            {
                "candidate_id": "cand_1",
                "verdict": "rejected",
                "missing": ["identity"],
                "next_action": {
                    "tool": "ground_target",
                    "args": {"instruction": "chair"},
                },
            },
        ),
    )
    registry.script("ground_target", found(candidate="cand_2"))
    out = SemanticNavigationService(registry).confirm(make_ctx(), "cand_1")

    assert out["ok"] is False
    assert out["status"] == "rejected"
    assert registry.tools == ["confirm_candidate"]


def test_confirm_without_verdict_is_not_ok():
    registry = FakeRegistry().script(
        "confirm_candidate", ToolResult(STATUS_OK, {"candidate_id": "cand_1"})
    )
    out = SemanticNavigationService(registry).confirm(make_ctx(), "cand_1")

    assert_contract(out)
    assert out["ok"] is False
    assert out["accepted"] is False
    assert out["status"] == STATUS_FAILED
    assert "verdict" in out["error"]


def test_confirm_passes_instruction_and_relation():
    registry = FakeRegistry().script(
        "confirm_candidate",
        ToolResult(STATUS_OK, {"candidate_id": "cand_1", "verdict": "accepted"}),
    )
    out = SemanticNavigationService(registry).confirm(
        make_ctx(),
        "cand_1",
        instruction="the wooden chair",
        relation="by the window",
    )

    assert out["ok"] is True
    assert registry.calls[0]["args"] == {
        "candidate_id": "cand_1",
        "max_views": 2,
        "instruction": "the wooden chair",
        "relation": "by the window",
    }


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "ok", "status"),
    [
        (True, True, "verified"),
        ("uncertain", False, "uncertain"),
        (False, False, "failed"),
        (None, False, "failed"),
    ],
)
def test_verify_is_ok_only_for_verified_true(raw, ok, status):
    registry = FakeRegistry().script(
        "verify_goal",
        ToolResult(
            STATUS_OK,
            {"verified": raw, "reason": "checked", "candidate_id": "cand_1"},
        ),
    )
    out = SemanticNavigationService(registry).verify(make_ctx(), "cand_1")

    assert_contract(out)
    assert out["ok"] is ok
    assert out["verified"] is ok
    assert out["status"] == status
    assert out["calls"] == 1
    assert out["data"]["verified"] == raw
    if ok:
        assert out["error"] == ""
    else:
        assert out["error"]


def test_verify_passes_success_distance():
    registry = FakeRegistry().script(
        "verify_goal", ToolResult(STATUS_OK, {"verified": True})
    )
    out = SemanticNavigationService(registry).verify(
        make_ctx(), "cand_1", success_dist_m=1.2
    )

    assert out["ok"] is True
    assert registry.calls[0]["args"] == {
        "candidate_id": "cand_1",
        "success_dist_m": 1.2,
    }


# ---------------------------------------------------------------------------
# revisit / goto
# ---------------------------------------------------------------------------


def test_revisit_observation_loops_until_arrived():
    registry = FakeRegistry().script(
        "return_to_observation",
        movement("task_3", "running"),
        movement("task_3", "arrived", arrived=True),
    )
    out = SemanticNavigationService(registry).revisit(
        make_ctx(), observation_id="obs_4"
    )

    assert_contract(out)
    assert out["ok"] is True
    assert out["arrived"] is True
    assert out["tool"] == "return_to_observation"
    assert out["task_id"] == "task_3"
    assert out["calls"] == 2
    assert registry.calls[0]["args"]["observation_id"] == "obs_4"
    assert registry.calls[1]["args"] == {"task_id": "task_3"}


def test_revisit_viewpoint_loops_until_arrived():
    registry = FakeRegistry().script(
        "go_to_viewpoint",
        movement("vp_task", "arrived", arrived=True),
    )
    out = SemanticNavigationService(registry).revisit(
        make_ctx(), viewpoint_id="vp_2"
    )

    assert out["ok"] is True
    assert out["arrived"] is True
    assert out["tool"] == "go_to_viewpoint"
    assert registry.calls[0]["args"]["viewpoint_id"] == "vp_2"


def test_goto_position_proposes_a_viewpoint_then_flies():
    registry = FakeRegistry()
    registry.script(
        "propose_viewpoints",
        ToolResult(
            STATUS_OK,
            {"viewpoint_id": "vp_4", "viewpoints": [{"viewpoint_id": "vp_4"}]},
        ),
    )
    registry.script(
        "go_to_viewpoint", movement("task_8", "arrived", arrived=True)
    )
    out = SemanticNavigationService(registry).goto(
        make_ctx(), [1.0, 2.0, 1.0]
    )

    assert_contract(out)
    assert out["ok"] is True
    assert out["arrived"] is True
    assert out["tool"] == "go_to_viewpoint"
    assert out["calls"] == 2
    assert registry.calls[0]["args"]["target_hypothesis"] == [1.0, 2.0, 1.0]
    assert registry.calls[1]["args"]["viewpoint_id"] == "vp_4"


def test_goto_position_without_a_viewpoint_fails():
    registry = FakeRegistry().script(
        "propose_viewpoints",
        ToolResult(
            STATUS_BLOCKED,
            {"viewpoints": [], "reason": "no_reachable_viewpoint"},
        ),
    )
    out = SemanticNavigationService(registry).goto(make_ctx(), [1.0, 2.0, 1.0])

    assert_contract(out)
    assert out["ok"] is False
    assert out["calls"] == 1
    assert registry.tools == ["propose_viewpoints"]
    assert "no_reachable_viewpoint" in out["error"]


def test_revisit_missing_destination_is_structured():
    registry = FakeRegistry()
    out = SemanticNavigationService(registry).revisit(make_ctx())

    assert_contract(out)
    assert out["ok"] is False
    assert "position, observation_id or viewpoint_id" in out["error"]
    assert registry.calls == []


def test_revisit_invalid_position_is_structured():
    registry = FakeRegistry()
    out = SemanticNavigationService(registry).revisit(
        make_ctx(), position=[1.0, float("nan"), 1.0]
    )

    assert_contract(out)
    assert out["ok"] is False
    assert "finite" in out["error"]
    assert registry.calls == []


def test_travel_loop_is_bounded(monkeypatch):
    monkeypatch.setattr(semantic, "MAX_TRAVEL_LEGS", 2)
    registry = FakeRegistry().default(
        "go_to_viewpoint",
        lambda name, args, ctx: movement("task_1", "running"),
    )
    out = SemanticNavigationService(registry).revisit(
        make_ctx(), viewpoint_id="vp_1"
    )

    assert_contract(out)
    assert out["ok"] is False
    assert out["status"] == "running"
    assert out["calls"] == 2
    assert "max_travel_legs_exceeded" in out["error"]


def test_travel_reports_a_blocked_last_leg_at_the_cap(monkeypatch):
    monkeypatch.setattr(semantic, "MAX_TRAVEL_LEGS", 2)
    registry = FakeRegistry().script(
        "go_to_viewpoint",
        movement("task_1", "running"),
        movement("task_1", "blocked", reason="path_blocked"),
    )
    out = SemanticNavigationService(registry).revisit(
        make_ctx(), viewpoint_id="vp_1"
    )

    assert out["ok"] is False
    assert out["status"] == "blocked"
    assert out["error"] == "path_blocked"
    assert out["calls"] == 2


# ---------------------------------------------------------------------------
# no false success / guards
# ---------------------------------------------------------------------------


def test_approach_does_not_infer_arrival_from_task_status():
    registry = FakeRegistry().script(
        "approach_candidate",
        ToolResult(STATUS_OK, {"task_id": "task_1", "task_status": "arrived"}),
    )
    out = SemanticNavigationService(registry).approach(make_ctx(), "cand_1")

    assert_contract(out)
    assert out["ok"] is False
    assert out["arrived"] is False
    assert "without arrived=true" in out["error"]


def test_revisit_does_not_infer_arrival_from_task_status():
    registry = FakeRegistry().script(
        "go_to_viewpoint",
        ToolResult(STATUS_OK, {"task_id": "task_1", "task_status": "arrived"}),
    )
    out = SemanticNavigationService(registry).revisit(
        make_ctx(), viewpoint_id="vp_1"
    )

    assert out["ok"] is False
    assert out["arrived"] is False
    assert out["task_id"] == "task_1"


def test_missing_registry_is_structured():
    out = SemanticNavigationService(None).search(make_ctx(), "chair")

    assert_contract(out)
    assert out["ok"] is False
    assert "tool registry" in out["error"]


def test_missing_world_is_structured():
    registry = FakeRegistry()
    out = SemanticNavigationService(registry).search(
        ToolContext(world=None), "chair"
    )

    assert_contract(out)
    assert out["ok"] is False
    assert "world backend" in out["error"]
    assert registry.calls == []


def test_registry_exception_is_structured():
    registry = FakeRegistry().script(
        "search_target", RuntimeError("registry exploded")
    )
    out = SemanticNavigationService(registry).search(make_ctx(), "chair")

    assert_contract(out)
    assert out["ok"] is False
    assert out["calls"] == 1
    assert "RuntimeError" in out["error"]
    assert "registry exploded" in out["error"]


def test_contract_reports_tool_availability():
    registry = FakeRegistry("search_target", "verify_goal")
    contract = SemanticNavigationService(registry, max_retries=3).contract()

    assert contract["service"] == "semantic.service@1"
    assert contract["max_retries"] == 3
    assert contract["available"]["search_target"] is True
    assert contract["available"]["verify_goal"] is True
    assert contract["available"]["approach_candidate"] is False
    assert "approach_candidate" in contract["missing"]
    assert contract["methods"] == list(semantic.SEMANTIC_METHODS)
