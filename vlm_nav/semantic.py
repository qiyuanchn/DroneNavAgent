"""Semantic navigation service: search, approach, confirm, verify, revisit.

The service is the task-level contract above the semantic tools. A caller (the
mixed-mission executor, a planner plugin, a script) asks for a *semantic stage*
- "find the chair by the window", "fly next to it", "confirm it is the one",
"verify completion", "go back to the kitchen" - and the service drives the
registered tools to a structured verdict instead of exposing the tool loop.

Design rules (``docs/mixed_mission_contract.md``):

* every tool runs through ``registry.call(name, args, ctx)`` - the service never
  imports or invokes a handler, so tool scope, budget, cache and evidence
  stamping are the registry's, not a second implementation;
* methods are stateless: everything episode-scoped lives in ``ctx`` and every
  answer is a fresh dict;
* recovery is *evidence-driven*: ``next_action`` / ``strategy_change`` are only
  followed when they name a tool the registry currently exposes, and only up to
  ``max_retries`` times;
* success is never inferred from a status: ``search`` is ok with a candidate id,
  ``approach``/``revisit``/``goto`` only with ``arrived is True``, ``confirm``
  only with ``verdict == "accepted"`` and ``verify`` only with
  ``data["verified"] is True`` (the string ``"uncertain"`` is not a pass).

Every method returns the common dict::

    {"ok": bool, "tool": str, "status": str, "candidate_id": str,
     "task_id": str, "arrived": bool, "accepted": bool, "verified": bool,
     "calls": int, "error": str, "data": {...}}

``calls`` counts every registry call the method made, including recovery calls.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .tools import STATUS_BLOCKED, STATUS_EMPTY, STATUS_FAILED, STATUS_OK

SEMANTIC_SERVICE = "semantic.service@1"
SEMANTIC_SCHEMA = "vlm_nav.semantic.service/1"

SEMANTIC_METHODS = ("search", "approach", "confirm", "verify", "revisit", "goto")

SEARCH_TOOL = "search_target"
APPROACH_TOOL = "approach_candidate"
CONFIRM_TOOL = "confirm_candidate"
VERIFY_TOOL = "verify_goal"
VIEWPOINT_TOOL = "go_to_viewpoint"
RETURN_TOOL = "return_to_observation"
PROPOSE_TOOL = "propose_viewpoints"

#: The tools the service owns a loop for; ``semantic_status`` reports which of
#: them the active registry exposes.
SEMANTIC_TOOL_NAMES = (
    SEARCH_TOOL,
    APPROACH_TOOL,
    CONFIRM_TOOL,
    VERIFY_TOOL,
    VIEWPOINT_TOOL,
    RETURN_TOOL,
    PROPOSE_TOOL,
)

#: A goto/revisit task may need many one-leg calls. The cap is only a loop
#: guard; the movement tools own their own segment/timeout budgets.
MAX_TRAVEL_LEGS = 32

STATUS_ARRIVED = "arrived"
STATUS_RUNNING = "running"
STATUS_VERIFIED = "verified"
STATUS_UNCERTAIN = "uncertain"
STATUS_ACCEPTED = "accepted"


def _blank(
    tool: str = "",
    *,
    error: str = "",
    status: str = STATUS_FAILED,
    calls: int = 0,
    data: Mapping | None = None,
) -> dict:
    """The structured failure/empty answer; never raises for bad input."""
    return {
        "ok": False,
        "tool": str(tool or ""),
        "status": str(status or STATUS_FAILED),
        "candidate_id": "",
        "task_id": "",
        "arrived": False,
        "accepted": False,
        "verified": False,
        "calls": max(0, int(calls)),
        "error": str(error or ""),
        "data": dict(data or {}),
    }


def _result_parts(result: Any) -> tuple[str, dict]:
    """``(status, data)`` of a tool result (ToolResult or a mapping lookalike)."""
    status = getattr(result, "status", "")
    data = getattr(result, "data", None)
    if isinstance(result, Mapping):
        status = status or result.get("status", "")
        if data is None:
            data = result.get("data", result)
    return str(status or ""), dict(data) if isinstance(data, Mapping) else {}


def _reason(data: Mapping, fallback: str = "") -> str:
    """The most specific human-readable reason in a tool payload."""
    for key in ("reason", "error", "blocked_reason", "hint"):
        value = data.get(key)
        if value not in (None, ""):
            return str(value)
    return str(fallback or "")


def _action(payload: Any) -> tuple[str, dict] | None:
    """``(tool, args)`` of a ``next_action``-shaped payload, or None."""
    if not isinstance(payload, Mapping):
        return None
    tool = payload.get("tool") or payload.get("name")
    if not isinstance(tool, str) or not tool.strip():
        return None
    args = payload.get("args", payload.get("arguments"))
    if not isinstance(args, Mapping):
        args = {}
    return tool.strip(), dict(args)


def _recovery_action(data: Mapping) -> tuple[str, dict] | None:
    """The registered recovery a result proposes, ``next_action`` first.

    ``strategy_change`` may carry the call directly or a ranked ``options``
    list (the repeat-move guard's shape); the first usable entry wins.
    """
    action = _action(data.get("next_action"))
    if action is not None:
        return action
    change = data.get("strategy_change")
    if not isinstance(change, Mapping):
        return None
    action = _action(change)
    if action is not None:
        return action
    options = change.get("options")
    if isinstance(options, (list, tuple)):
        for option in options:
            action = _action(option)
            if action is not None:
                return action
    return None


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def compose_instruction(target: str, relation: str = "") -> str:
    """Compose a search instruction from a target and an optional relation.

    The relation is a place/anchor phrase ("窗边", "next to the window"). A
    Chinese relation is joined with ``的`` ("窗边" + "木椅" -> "窗边的木椅");
    an English one is appended ("chair" + "next to the window"). A relation
    already contained in the target is not repeated.
    """
    target = str(target or "").strip()
    relation = str(relation or "").strip()
    if not relation or not target or relation in target:
        return target
    if _has_cjk(relation):
        return f"{relation}{target}" if relation.endswith("的") else f"{relation}的{target}"
    return f"{target} {relation}"


def _position(value: Any) -> list[float] | None:
    """A finite ``[x, y, z]`` position, or None."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    try:
        point = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if len(point) < 3:
        return None
    point = point[:3]
    if not all(math.isfinite(item) for item in point):
        return None
    return point


def _extra_of(ctx: Any) -> Mapping:
    extra = getattr(ctx, "extra", None)
    return extra if isinstance(extra, Mapping) else {}


def resolve_candidate(ctx: Any, candidate_id: Any = None) -> str:
    """The explicit candidate id, else the episode's scoped/last candidate."""
    explicit = str(candidate_id or "").strip()
    if explicit:
        return explicit
    extra = _extra_of(ctx)
    scope = extra.get("task_scope")
    if isinstance(scope, Mapping):
        scoped = str(scope.get("candidate_id") or "").strip()
        if scoped:
            return scoped
    for key in ("candidate_id", "last_candidate_id"):
        value = str(extra.get(key) or "").strip()
        if value:
            return value
    return ""


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


class SemanticNavigationService:
    """Drive the semantic tools to a structured verdict (stateless)."""

    def __init__(self, registry: Any, *, max_retries: int = 2) -> None:
        self.registry = registry
        self.max_retries = max(0, _as_int(max_retries, 2))

    # -- contract --------------------------------------------------------- #

    def contract(self) -> dict:
        """Describe the service and its tool requirements (for ``semantic_status``)."""
        names = self._names()
        available = {name: name in names for name in SEMANTIC_TOOL_NAMES}
        return {
            "schema": SEMANTIC_SCHEMA,
            "service": SEMANTIC_SERVICE,
            "methods": list(SEMANTIC_METHODS),
            "tools": list(SEMANTIC_TOOL_NAMES),
            "available": available,
            "missing": [name for name in SEMANTIC_TOOL_NAMES if not available[name]],
            "max_retries": self.max_retries,
            "evidence": {
                "search": "candidate_id",
                "approach": "arrived is true",
                "confirm": 'verdict == "accepted"',
                "verify": "verified is true",
                "revisit/goto": "arrived is true",
            },
        }

    # -- public API ------------------------------------------------------- #

    def search(
        self,
        ctx: Any,
        target: str,
        *,
        relation: str = "",
        instruction: str | None = None,
        max_steps: int = 8,
        max_turns: int = 1,
    ) -> dict:
        """Search for ``target``; ok when a candidate id is returned.

        ``max_steps``/``max_turns`` are passed to ``search_target`` as
        ``max_steps``/``budget_turns``. A result with no candidate may propose
        a recovery (``next_action`` or ``strategy_change``); it is executed at
        most ``max_retries`` times, each followed by a retry of the original
        search. An unconfirmed candidate (``confirmed=false``) is returned for
        ``approach`` but is never reported as identity-accepted.
        """
        problem = self._guard(ctx)
        if problem:
            return _blank(SEARCH_TOOL, error=problem)
        target = str(target or "").strip()
        text = str(instruction or "").strip()
        if not text:
            text = compose_instruction(target, relation)
        if not text:
            return _blank(SEARCH_TOOL, error="search requires a target or instruction")

        args = {
            "instruction": text,
            "max_steps": max(1, _as_int(max_steps, 8)),
            "budget_turns": max(1, _as_int(max_turns, 1)),
        }
        names = self._names()
        calls = 0
        recoveries = 0
        status = STATUS_FAILED
        data: dict = {}
        while True:
            calls += 1
            try:
                result = self.registry.call(SEARCH_TOOL, dict(args), ctx)
            except Exception as exc:  # noqa: BLE001 - the boundary reports, never raises
                return _blank(
                    SEARCH_TOOL,
                    error=f"{type(exc).__name__}: {exc}",
                    status=STATUS_FAILED,
                    calls=calls,
                    data=data,
                )
            status, data = _result_parts(result)
            candidate = str(data.get("candidate_id") or "").strip()
            if candidate:
                return {
                    "ok": True,
                    "tool": SEARCH_TOOL,
                    "status": status or STATUS_OK,
                    "candidate_id": candidate,
                    "task_id": "",
                    "arrived": False,
                    # a found box is not an identity confirmation
                    "accepted": data.get("confirmed") is True,
                    "verified": False,
                    "calls": calls,
                    "error": "",
                    "data": data,
                }
            action = _recovery_action(data)
            if (
                action is None
                or action[0] not in names
                or recoveries >= self.max_retries
            ):
                break
            recoveries += 1
            calls += 1
            try:
                self.registry.call(action[0], dict(action[1]), ctx)
            except Exception:  # noqa: BLE001 - a failed recovery is not a crash
                # The retry of the original search may still succeed; the hit
                # is counted and the search's own reason remains the answer.
                pass
        error = _reason(data, status or "no_candidate")
        return _blank(
            SEARCH_TOOL, error=error, status=status or STATUS_EMPTY,
            calls=calls, data=data,
        )

    def approach(
        self,
        ctx: Any,
        candidate_id: str | None = None,
        *,
        standoff_m: float = 1.2,
        max_legs: int = 8,
    ) -> dict:
        """Fly to a candidate; ok only when the movement reports arrival.

        The first call carries the candidate id and standoff; ``task_status ==
        "running"`` answers are continued with the returned ``task_id``. A
        blocked/failed answer may follow one registered recovery
        (``next_action``/``strategy_change``, bounded by ``max_retries``) and
        then retries the approach.
        """
        problem = self._guard(ctx)
        if problem:
            return _blank(APPROACH_TOOL, error=problem)
        candidate = resolve_candidate(ctx, candidate_id)
        if not candidate:
            return _blank(
                APPROACH_TOOL,
                error="approach requires a candidate id (none is in ctx.extra)",
            )
        legs = max(1, _as_int(max_legs, 8))
        standoff = _as_float(standoff_m, 1.2)
        names = self._names()
        initial = {"candidate_id": candidate, "standoff_m": standoff}
        args = dict(initial)
        calls = 0
        attempts = 0
        recoveries = 0
        task_id = ""
        status = STATUS_FAILED
        data: dict = {}
        arrived = False
        while attempts < legs:
            attempts += 1
            calls += 1
            try:
                result = self.registry.call(APPROACH_TOOL, dict(args), ctx)
            except Exception as exc:  # noqa: BLE001
                return _blank(
                    APPROACH_TOOL,
                    error=f"{type(exc).__name__}: {exc}",
                    status=STATUS_FAILED,
                    calls=calls,
                    data=data,
                )
            status, data = _result_parts(result)
            task_id = str(data.get("task_id") or task_id)
            if data.get("arrived") is True:
                arrived = True
                break
            task_status = str(data.get("task_status") or "").strip().lower()
            if task_status == STATUS_RUNNING:
                if not task_id:
                    break
                args = {"task_id": task_id}
                continue
            action = _recovery_action(data)
            if (
                action is None
                or action[0] not in names
                or recoveries >= self.max_retries
            ):
                break
            recoveries += 1
            calls += 1
            try:
                self.registry.call(action[0], dict(action[1]), ctx)
            except Exception:  # noqa: BLE001 - retry the approach regardless
                pass
            args = dict(initial)
        if arrived:
            return {
                "ok": True,
                "tool": APPROACH_TOOL,
                "status": STATUS_ARRIVED,
                "candidate_id": candidate,
                "task_id": task_id,
                "arrived": True,
                "accepted": False,
                "verified": False,
                "calls": calls,
                "error": "",
                "data": data,
            }
        reported = str(data.get("task_status") or "").strip().lower()
        if attempts >= legs and reported == STATUS_RUNNING:
            error = f"max_legs_exceeded after {attempts} approach call(s)"
            status = STATUS_RUNNING
        else:
            error = _reason(
                data,
                (
                    f"approach ended as {reported or status} "
                    "without arrived=true"
                ),
            )
            if reported in ("blocked", "failed", "cancelled", STATUS_RUNNING):
                status = reported
            elif reported == STATUS_ARRIVED:
                # the lifecycle word is not the evidence boolean
                status = STATUS_FAILED
            elif status not in (STATUS_BLOCKED, STATUS_FAILED, STATUS_EMPTY):
                status = STATUS_BLOCKED
        return _blank(
            APPROACH_TOOL, error=error, status=status or STATUS_BLOCKED,
            calls=calls, data=data,
        ) | {
            "candidate_id": candidate,
            "task_id": task_id,
        }

    def confirm(
        self,
        ctx: Any,
        candidate_id: str | None = None,
        *,
        instruction: str | None = None,
        relation: str | None = None,
        max_views: int = 2,
    ) -> dict:
        """Confirm a candidate; ok only for the ``accepted`` verdict.

        ``uncertain`` and ``rejected`` are preserved in ``status`` (and in
        ``data["verdict"]``). A rejected candidate is not retried, and a
        proposed ``next_action`` is not executed here: confirmation has one
        verdict, the caller decides what to do with it.
        """
        problem = self._guard(ctx)
        if problem:
            return _blank(CONFIRM_TOOL, error=problem)
        candidate = resolve_candidate(ctx, candidate_id)
        if not candidate:
            return _blank(
                CONFIRM_TOOL,
                error="confirm requires a candidate id (none is in ctx.extra)",
            )
        args: dict[str, Any] = {
            "candidate_id": candidate,
            "max_views": max(1, _as_int(max_views, 2)),
        }
        if instruction is not None and str(instruction).strip():
            args["instruction"] = str(instruction).strip()
        if relation is not None and str(relation).strip():
            args["relation"] = str(relation).strip()
        try:
            result = self.registry.call(CONFIRM_TOOL, dict(args), ctx)
        except Exception as exc:  # noqa: BLE001
            return _blank(
                CONFIRM_TOOL, error=f"{type(exc).__name__}: {exc}", calls=1,
            )
        status, data = _result_parts(result)
        verdict = str(data.get("verdict") or "").strip().lower()
        accepted = verdict == STATUS_ACCEPTED
        if verdict:
            out_status = verdict
        elif status in (STATUS_BLOCKED, STATUS_FAILED, STATUS_EMPTY):
            out_status = status
        else:
            out_status = STATUS_FAILED
        return {
            "ok": accepted,
            "tool": CONFIRM_TOOL,
            "status": out_status,
            "candidate_id": candidate,
            "task_id": str(data.get("task_id") or ""),
            "arrived": False,
            "accepted": accepted,
            "verified": False,
            "calls": 1,
            "error": "" if accepted else _reason(
                data, f"confirm_candidate returned verdict={verdict or 'none'}"
            ),
            "data": data,
        }

    def verify(
        self,
        ctx: Any,
        candidate_id: str | None = None,
        *,
        success_dist_m: float | None = None,
    ) -> dict:
        """Verify completion; ok only when ``data["verified"] is True``.

        The string ``"uncertain"`` (and a missing/false verdict) stays not-ok;
        the raw verdict is preserved in ``data`` and reflected in ``status``.
        """
        problem = self._guard(ctx)
        if problem:
            return _blank(VERIFY_TOOL, error=problem)
        candidate = resolve_candidate(ctx, candidate_id)
        args: dict[str, Any] = {}
        if candidate:
            args["candidate_id"] = candidate
        if success_dist_m is not None:
            args["success_dist_m"] = _as_float(success_dist_m, 0.0)
        try:
            result = self.registry.call(VERIFY_TOOL, dict(args), ctx)
        except Exception as exc:  # noqa: BLE001
            return _blank(
                VERIFY_TOOL, error=f"{type(exc).__name__}: {exc}", calls=1,
            )
        status, data = _result_parts(result)
        raw = data.get("verified")
        verified = raw is True
        if verified:
            out_status = STATUS_VERIFIED
        elif isinstance(raw, str) and raw.strip().lower() == STATUS_UNCERTAIN:
            out_status = STATUS_UNCERTAIN
        elif status in (STATUS_BLOCKED, STATUS_FAILED, STATUS_EMPTY):
            out_status = status
        else:
            out_status = STATUS_FAILED
        return {
            "ok": verified,
            "tool": VERIFY_TOOL,
            "status": out_status,
            "candidate_id": candidate or str(data.get("candidate_id") or ""),
            "task_id": str(data.get("task_id") or ""),
            "arrived": False,
            "accepted": False,
            "verified": verified,
            "calls": 1,
            "error": "" if verified else _reason(
                data, f"verify_goal returned verified={raw!r}"
            ),
            "data": data,
        }

    def revisit(
        self,
        ctx: Any,
        *,
        position: Any = None,
        observation_id: str | None = None,
        viewpoint_id: str | None = None,
        standoff_m: float = 0.6,
    ) -> dict:
        """Return to a recorded place; ok only when the movement arrives.

        ``observation_id`` flies back to a recorded camera pose
        (``return_to_observation``); ``viewpoint_id`` flies to a proposed
        viewpoint (``go_to_viewpoint``); ``position`` is turned into a
        viewpoint proposal first (``propose_viewpoints`` with the position as
        ``target_hypothesis``). A running task is continued with its
        ``task_id`` until it arrives or the loop guard trips.
        """
        problem = self._guard(ctx)
        if problem:
            return _blank(error=problem)
        observation = str(observation_id or "").strip()
        viewpoint = str(viewpoint_id or "").strip()
        point = _position(position) if position is not None else None
        calls = 0
        data: dict = {}
        if observation:
            tool = RETURN_TOOL
            args: dict[str, Any] = {
                "observation_id": observation,
                "standoff_m": _as_float(standoff_m, 0.6),
            }
        elif viewpoint:
            tool = VIEWPOINT_TOOL
            args = {
                "viewpoint_id": viewpoint,
                "standoff_m": _as_float(standoff_m, 0.6),
            }
        elif point is not None:
            tool = VIEWPOINT_TOOL
            calls += 1
            try:
                proposal = self.registry.call(
                    PROPOSE_TOOL, {"target_hypothesis": point, "count": 3}, ctx
                )
            except Exception as exc:  # noqa: BLE001
                return _blank(
                    PROPOSE_TOOL,
                    error=f"{type(exc).__name__}: {exc}",
                    calls=calls,
                )
            proposal_status, data = _result_parts(proposal)
            viewpoint = str(data.get("viewpoint_id") or "").strip()
            if not viewpoint:
                entries = data.get("viewpoints")
                if isinstance(entries, (list, tuple)):
                    for entry in entries:
                        if isinstance(entry, Mapping) and entry.get("viewpoint_id"):
                            viewpoint = str(entry["viewpoint_id"])
                            break
            if not viewpoint:
                return _blank(
                    PROPOSE_TOOL,
                    error=_reason(data, "propose_viewpoints returned no viewpoint"),
                    status=proposal_status or STATUS_BLOCKED,
                    calls=calls,
                    data=data,
                )
            args = {
                "viewpoint_id": viewpoint,
                "standoff_m": _as_float(standoff_m, 0.6),
            }
        elif position is not None:
            return _blank(
                error="position must be a finite [x, y, z] sequence",
                status=STATUS_FAILED,
            )
        else:
            return _blank(
                error=(
                    "revisit requires a position, observation_id or viewpoint_id"
                ),
                status=STATUS_FAILED,
            )
        return self._travel(ctx, tool, args, calls=calls, data=data, candidate="")

    def goto(
        self,
        ctx: Any,
        position: Any,
        *,
        standoff_m: float = 0.6,
    ) -> dict:
        """Fly to a world position (via ``propose_viewpoints`` + ``go_to_viewpoint``)."""
        return self.revisit(ctx, position=position, standoff_m=standoff_m)

    # -- internals -------------------------------------------------------- #

    def _names(self) -> set[str]:
        if self.registry is None:
            return set()
        names = getattr(self.registry, "names", None)
        if not callable(names):
            return set()
        try:
            return {str(name) for name in names()}
        except Exception:  # noqa: BLE001 - a broken catalogue is "no recovery"
            return set()

    def _guard(self, ctx: Any) -> str:
        """A structured-error reason for a missing registry/world, else ``""``."""
        if self.registry is None or not callable(getattr(self.registry, "call", None)):
            return "semantic navigation requires a tool registry"
        if ctx is None or getattr(ctx, "world", None) is None:
            return "semantic navigation requires a world backend"
        return ""

    def _travel(
        self,
        ctx: Any,
        tool: str,
        initial_args: dict,
        *,
        calls: int,
        data: Mapping | None = None,
        candidate: str = "",
    ) -> dict:
        """Loop one movement tool on its ``task_id`` until ``arrived`` is true."""
        args = dict(initial_args)
        last_data = dict(data or {})
        status = STATUS_FAILED
        task_id = ""
        arrived = False
        legs = 0
        while legs < MAX_TRAVEL_LEGS:
            legs += 1
            calls += 1
            try:
                result = self.registry.call(tool, dict(args), ctx)
            except Exception as exc:  # noqa: BLE001
                return _blank(
                    tool,
                    error=f"{type(exc).__name__}: {exc}",
                    status=STATUS_FAILED,
                    calls=calls,
                    data=last_data,
                )
            status, last_data = _result_parts(result)
            task_id = str(last_data.get("task_id") or task_id)
            if last_data.get("arrived") is True:
                arrived = True
                break
            task_status = str(last_data.get("task_status") or "").strip().lower()
            if task_status == STATUS_RUNNING and task_id:
                args = {"task_id": task_id}
                continue
            break
        if arrived:
            return {
                "ok": True,
                "tool": tool,
                "status": STATUS_ARRIVED,
                "candidate_id": candidate,
                "task_id": task_id,
                "arrived": True,
                "accepted": False,
                "verified": False,
                "calls": calls,
                "error": "",
                "data": last_data,
            }
        reported = str(last_data.get("task_status") or "").strip().lower()
        if legs >= MAX_TRAVEL_LEGS and reported == STATUS_RUNNING:
            error = f"max_travel_legs_exceeded after {legs} call(s)"
            status = STATUS_RUNNING
        else:
            error = _reason(
                last_data,
                (
                    f"{tool} ended as {reported or status} without "
                    "arrived=true"
                ),
            )
            if reported in ("blocked", "failed", "cancelled", STATUS_RUNNING):
                status = reported
            elif reported == STATUS_ARRIVED:
                status = STATUS_FAILED
            elif status not in (STATUS_BLOCKED, STATUS_FAILED, STATUS_EMPTY):
                status = STATUS_BLOCKED
        return _blank(
            tool, error=error, status=status or STATUS_BLOCKED,
            calls=calls, data=last_data,
        ) | {
            "candidate_id": candidate,
            "task_id": task_id,
        }


__all__ = [
    "MAX_TRAVEL_LEGS",
    "SEMANTIC_METHODS",
    "SEMANTIC_SCHEMA",
    "SEMANTIC_SERVICE",
    "SEMANTIC_TOOL_NAMES",
    "SemanticNavigationService",
    "compose_instruction",
    "resolve_candidate",
]
