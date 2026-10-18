"""Tool registry and shared contracts for the agent runtime (goal.md M1).

Every tool has one input/output contract: ``ToolSpec`` declares name,
description, a JSON input schema and a handler ``fn(ctx, args) -> ToolResult``.
Results carry structured fields plus evidence references (image/observation/
memory/map/trajectory), never bare prose. The same registry backs both the
direct-call adapter and the MCP servers (``mcp_server.py``).

The model requests a tool with a constrained-JSON call (no native tool-calling
required on the model side):

    {"tool": "observe", "args": {"mode": "lookaround"}}
    {"tool": "go_to_viewpoint", "args": {"viewpoint_id": "vp_3"}}
    {"tool": "finish", "args": {"reason": "target verified"}}

``parse_tool_call`` extracts one call from a model reply; the runtime executes
it, formats the ``ToolResult`` and feeds it back. Pose-changing tools are
serialized (``side_effects=True``); read-only tools may run in parallel.

A call that cannot produce anything new must not look like progress: an
identical repeat of a ``cacheable`` read-only tool is served from the episode's
tool cache (``ctx.extra[TOOL_CACHE_KEY]``, bounded by
``DEFAULT_TOOL_CACHE_LIMIT``) as ``STATUS_CACHED`` plus the "repeat query, no
new evidence" hint. The cache is dropped whenever the episode state advances -
a ``side_effects`` call, or a result that carries new evidence - so a replayed
answer is never stale evidence (goal.md section 5).

The same contract covers what a *turn* changed: ``PROGRESS_KINDS``
(``information`` / ``action`` / ``task`` / ``none``) is the vocabulary the
runtime scores every executed call with, and a tool that knows what its own
answer means may declare it by putting one of those words under
``data["progress_kind"]`` (``declared_progress_kind``) - a search skill that
exhausted an area, a verifier that confirmed the target. ``PROMPT_EXAMPLES``
holds the example calls the prompts show the model; ``schema_problems`` checks
a call's arguments against a ``ToolSpec``'s schema, so the examples, the schemas
and the tools they demonstrate are validated as one interface.

Tool *dispatch* is state-aware. A registry that is handed the episode's context
(``ToolRegistry.attach_state``) renders its catalogue for the state the episode
is in: ``tool_state`` derives it from ``ctx.extra`` (a running navigation task,
a verification, a grounded candidate, recorded observations, or nothing yet),
``TOOL_STATE_PRIORITY`` ranks the tools each state calls first, and
``tool_catalog`` renders every registered tool - the state's own ones with their
full description and arguments, the rest degraded to a gist, a schema or a bare
name as the character budget allows, never cut inside an entry. The same state
drives the MCP side (``mcp_server``), so a local call and an MCP client see the
same tool set in the same order.

Two further contracts live here because both adapters share them:

- ``batch_read``: one call may pack several *read-only* checks (observe +
  query_memory, a compare and an inspect) and gets their results merged, so one
  model turn can gather a whole read-instead of one call per turn. Pose-changing
  tools are refused inside a batch (``side_effects`` tools stay serial, each with
  its own ``task_id``/completion state) and the wrapper itself costs no budget:
  a batch of n read-only calls costs n, not n+1.
- the completion evidence chain: ``finish`` with ``"success": true`` is only
  accepted when the episode's own trail backs it - a ``verify_goal`` that
  returned ``verified: true``, or a grounded candidate plus an arrival
  (``completion_evidence``). The trail is recorded in the episode's evidence
  pack (``ctx.extra["evidence_pack"]``) by the tools that produce it, so the
  runtime, the runner and an MCP client judge one pack the same way: an unbacked
  claim is downgraded to an *unverified* claim and is never counted as success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Any, Callable, Mapping

from .observations import CANDIDATE_STORE_KEY, OBSERVATION_STORE_KEY

# status vocabulary shared by every tool
STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_BLOCKED = "blocked"
STATUS_FAILED = "failed"
STATUS_CACHED = "cached"

# ---------------------------------------------------------------------------
# turn progress (goal.md 5: a turn that cannot produce anything new must not
# look like progress)
# ---------------------------------------------------------------------------

#: ``information`` = a new view/query answer (checked a new view, excluded a
#: candidate, sharpened the evidence); ``action`` = the drone moved (a new
#: observation position was reached, a route segment was flown); ``task`` = the
#: task itself advanced (identity confirmed, subgoal completed, a termination
#: condition satisfied); ``none`` = nothing moved (a replay, a failure, or the
#: same view reviewed again).
PROGRESS_INFORMATION = "information"
PROGRESS_ACTION = "action"
PROGRESS_TASK = "task"
PROGRESS_NONE = "none"
PROGRESS_KINDS = (PROGRESS_INFORMATION, PROGRESS_ACTION, PROGRESS_TASK, PROGRESS_NONE)


def declared_progress_kind(data: Any) -> str | None:
    """``data["progress_kind"]`` normalized, or None when nothing is declared.

    A tool that knows what its own answer means (a search skill that exhausted
    an area, a verifier that confirmed the target) can declare it instead of
    leaving the classification to the runtime's heuristics.
    """
    if not isinstance(data, Mapping):
        return None
    value = data.get("progress_kind")
    if isinstance(value, str) and value.strip().lower() in PROGRESS_KINDS:
        return value.strip().lower()
    return None


# ---------------------------------------------------------------------------
# state-aware tool dispatch (goal.md P4)
# ---------------------------------------------------------------------------

#: ``start``: nothing observed yet - look, search, recall, then ground.
#: ``search``: views are on record but no candidate exists - sweep, ground,
#: compare. ``candidate``: something was grounded - compare it, approach it,
#: verify it. ``moving``: a navigation task is running - report it, continue it
#: or cancel it. ``verified``: a completion check passed - finish it.
TOOL_STATE_START = "start"
TOOL_STATE_SEARCH = "search"
TOOL_STATE_CANDIDATE = "candidate"
TOOL_STATE_MOVING = "moving"
TOOL_STATE_VERIFIED = "verified"
TOOL_STATES = (
    TOOL_STATE_START,
    TOOL_STATE_SEARCH,
    TOOL_STATE_CANDIDATE,
    TOOL_STATE_MOVING,
    TOOL_STATE_VERIFIED,
)

#: What each state calls first, in order. Names the registry does not know are
#: skipped, and every registered tool the tuple does not mention keeps its
#: alphabetical place behind these - so a state changes the *order* and the
#: *budget share* of the listing, never the set of tools that can be called.
TOOL_STATE_PRIORITY: dict[str, tuple[str, ...]] = {
    TOOL_STATE_START: (
        "observe", "query_memory", "search_target", "ground_target",
        "propose_viewpoints", "batch_read", "inspect_region", "check_path",
        "update_plan", "get_plan", "verify_goal", "finish",
    ),
    TOOL_STATE_SEARCH: (
        "search_target", "observe", "ground_target", "query_memory",
        "compare_candidates", "inspect_region", "propose_viewpoints", "check_path",
        "batch_read", "update_plan", "get_plan", "verify_goal", "finish",
    ),
    TOOL_STATE_CANDIDATE: (
        "compare_candidates", "ground_target", "approach_candidate", "verify_goal",
        "propose_viewpoints", "check_path", "observe", "query_memory", "batch_read",
        "inspect_region", "task_status", "cancel_task", "update_plan", "get_plan",
        "finish",
    ),
    TOOL_STATE_MOVING: (
        "task_status", "cancel_task", "observe", "verify_goal", "batch_read",
        "compare_candidates", "propose_viewpoints", "ground_target", "inspect_region",
        "query_memory", "check_path", "search_target", "update_plan", "get_plan",
        "finish",
    ),
    TOOL_STATE_VERIFIED: (
        "finish", "verify_goal", "task_status", "cancel_task", "observe",
        "batch_read", "query_memory", "compare_candidates", "ground_target",
        "inspect_region", "propose_viewpoints", "check_path", "search_target",
        "update_plan", "get_plan",
    ),
}

#: Order of every tool the modules of this package register, used behind the
#: state-specific head so the ranking of a name no state mentions is stable.
TOOL_TAIL_ORDER = (
    "approach_candidate", "batch_read", "cancel_task", "check_path",
    "compare_candidates", "finish", "get_plan", "go_to_viewpoint", "ground_target",
    "inspect_region", "navigate", "observe", "propose_viewpoints", "query_memory",
    "return_to_observation", "search_target", "task_status", "update_plan",
    "verify_goal",
)

#: What a state is, in one sentence, when the context carries no detail.
TOOL_STATE_HINTS = {
    TOOL_STATE_START: "nothing observed yet: look, search or recall before acting",
    TOOL_STATE_SEARCH: "views on record, no candidate yet: search, ground, compare",
    TOOL_STATE_CANDIDATE: "candidate grounded: compare, approach and verify it",
    TOOL_STATE_MOVING: "a navigation task is running: report, continue or cancel it",
    TOOL_STATE_VERIFIED: "completion check passed: finish with the evidence",
}

#: A state's one-line hint is clipped to this before it leads the catalogue.
MAX_STATE_HINT_CHARS = 180
#: The dispatch rules the catalogue carries ahead of the tools themselves. One
#: line, because the catalogue is the only tool text that is budgeted for the
#: model (the protocol doc is not re-rendered per state) - and these three rules
#: are about the tool set as a whole: what may be packed into one call, what a
#: repeat means, and what a completion claim needs.
CATALOG_DISPATCH_NOTE = (
    "Dispatch: read-only checks may be packed into one batch_read call "
    "(calls=[{tool, args}, ...]); a pose-changing tool is refused there and stays "
    "serial with its own task_id. A repeated request for a place already reached "
    "answers already_done (nothing flies again). finish with success=true is only "
    "counted with the evidence trail it reports (a verify_goal that returned "
    "verified=true, or a grounded candidate plus an arrival)."
)
#: Default character budget of a state-aware catalogue: enough for the state's
#: own tools with their full arguments plus every other tool's name. The
#: caller's own budget wins (``ctx.extra``-driven adapters pass theirs); the
#: module runtime reserves this catalogue before its protocol doc, and its
#: default system budget (``runtime.DEFAULT_SYSTEM_CHARS``) holds the listing
#: whole. The runner passes the same width as an upper bound
#: (``scripts/run_vlm_nav_tools.py``).
DEFAULT_CATALOG_CHARS = 9000


def _count(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return 0


@dataclass(frozen=True)
class ToolState:
    """The episode state the catalogue is rendered for (``tool_state``)."""

    name: str
    hint: str = ""
    candidates: tuple[str, ...] = ()
    intents: tuple[str, ...] = ()
    tasks: tuple[tuple[str, str], ...] = ()  # (task_id, intent) of running tasks
    observations: int = 0
    viewpoints: int = 0
    verified: bool = False

    def priority(self) -> tuple[str, ...]:
        """The ranked tool names of this state (deduped, intent tools first)."""
        head: list[str] = []
        if self.name == TOOL_STATE_MOVING:
            head = ["task_status", "cancel_task", *self.intents]
        order: list[str] = []
        for group in (head, TOOL_STATE_PRIORITY.get(self.name, ()), TOOL_TAIL_ORDER):
            for name in group:
                if name not in order:
                    order.append(name)
        return tuple(order)


def tool_state(extra: Mapping | None = None, ctx: Any = None) -> ToolState:
    """The state of the episode from ``ctx.extra`` (or a ``ToolContext``).

    The signals are the ones the tools themselves write - the candidate store of
    ``ground_target``, the ``nav_tasks`` records of the movement tools, the
    recorded observations/viewpoints and the evidence pack of the completion
    chain - so the state follows the evidence the episode really has, not a
    counter the runtime keeps beside it.
    """
    if extra is None and ctx is not None:
        extra = getattr(ctx, "extra", None)
    if not isinstance(extra, Mapping):
        extra = {}

    candidates: list[str] = []
    store = extra.get(CANDIDATE_STORE_KEY)
    if isinstance(store, Mapping):
        for key, record in store.items():
            candidate_id = ""
            if isinstance(record, Mapping):
                candidate_id = str(record.get("id") or "")
            candidates.append(candidate_id or str(key))

    tasks: list[tuple[str, str]] = []
    intents: list[str] = []
    store = extra.get("nav_tasks")
    if isinstance(store, Mapping):
        for key, task in store.items():
            if not isinstance(task, Mapping):
                continue
            if str(task.get("status") or "") != "running":
                continue
            task_id = str(task.get("task_id") or key)
            intent = str(task.get("intent") or "")
            tasks.append((task_id, intent))
            if intent and intent not in intents:
                intents.append(intent)

    pack = extra.get(EVIDENCE_PACK_KEY)
    verified = False
    if isinstance(pack, Mapping):
        for item in pack.get("verifications") or ():
            if isinstance(item, Mapping) and item.get("verified") is True:
                verified = True
                break

    observations = _count(extra.get("observations")) or _count(extra.get(OBSERVATION_STORE_KEY))
    viewpoints = _count(extra.get("viewpoints_by_id")) or _count(extra.get("viewpoints"))

    if tasks:
        name = TOOL_STATE_MOVING
        hint = "{} running task(s): {}".format(
            len(tasks),
            ", ".join(f"{task_id} {intent}".strip() for task_id, intent in tasks[:3]),
        )
    elif verified:
        name = TOOL_STATE_VERIFIED
        hint = "a completion check of this episode passed (verify_goal verified=true)"
    elif candidates:
        name = TOOL_STATE_CANDIDATE
        hint = "{} grounded candidate(s): {}".format(
            len(candidates), ", ".join(candidates[:4])
        )
    elif observations:
        name = TOOL_STATE_SEARCH
        hint = f"{observations} observation(s) on record, no candidate grounded yet"
    else:
        name = TOOL_STATE_START
        hint = TOOL_STATE_HINTS[TOOL_STATE_START]
    return ToolState(
        name=name,
        hint=hint,
        candidates=tuple(candidates),
        intents=tuple(intents),
        tasks=tuple(tasks),
        observations=int(observations),
        viewpoints=int(viewpoints),
        verified=bool(verified),
    )


@dataclass(frozen=True)
class ToolCatalog:
    """One rendered tool catalogue: the state, the tiers and the text.

    ``order`` ranks every registered tool for the state, the tiers say how much
    of each entry fits the budget (``full``: description + schema, ``compact``:
    a gist + schema, ``named``: the name in the trailing line), and ``text`` is
    what the model reads. Every registered tool is in exactly one tier, so a
    tight budget degrades the listing instead of hiding tools from it.
    """

    state: ToolState
    order: tuple[str, ...]
    full: tuple[str, ...]
    compact: tuple[str, ...]
    named: tuple[str, ...]
    text: str
    limit: int

    def sections(self) -> dict:
        return {"full": list(self.full), "compact": list(self.compact),
                "named": list(self.named)}

    def has(self, name: str) -> bool:
        return name in self.order


def _gist(description: str, limit: int = 160) -> str:
    """First sentence of a tool description, clipped - kept parseable as text."""
    text = " ".join(str(description or "").split())
    for marker in (". ", ".\n"):
        head, sep, _ = text.partition(marker)
        if sep and len(head) <= limit:
            text = head + "."
            break
    if len(text) > limit:
        text = text[: max(0, limit - 3)].rstrip() + "..."
    return text


def _entry_line(name: str, spec: Any, description: str) -> str:
    schema = getattr(spec, "input_schema", None) or {"type": "object", "properties": {}}
    return f"- {name}: {description} args={json.dumps(schema, default=str, separators=(',', ':'))}"


def _catalog_text(header: str, body: list[str], named: list[str], named_prefix: str) -> str:
    parts = [header]
    if body:
        parts.append("Tools this state calls first - full description and arguments:")
        parts.extend(body)
    if named:
        parts.append(f"{named_prefix}{', '.join(named)}")
    return "\n".join(parts)


def tool_catalog(registry: Any, state: ToolState | None = None,
                 limit: int | None = None) -> ToolCatalog:
    """Render ``registry``'s tools for ``state`` inside ``limit`` characters.

    The tool order is the state's ranking (``ToolState.priority``) with every
    remaining registered name behind it, and every entry is added whole: the
    listing first tries the full entry (description + schema), then a compact one
    (first sentence + schema) and otherwise leaves the name in the trailing line.
    The state line, the dispatch note and the names are the fixed part - a
    ``limit`` too small for an entry still names every tool instead of truncating
    prose into a schema the model cannot parse.
    """
    names = list(registry.names())
    state = state or ToolState(TOOL_STATE_START, TOOL_STATE_HINTS[TOOL_STATE_START])
    limit = DEFAULT_CATALOG_CHARS if limit is None else max(120, int(limit))

    ranked = [name for name in state.priority() if name in names]
    order = ranked + [name for name in names if name not in ranked]

    hint = str(state.hint or TOOL_STATE_HINTS.get(state.name, "")).strip()
    if len(hint) > MAX_STATE_HINT_CHARS:
        hint = hint[: MAX_STATE_HINT_CHARS - 3].rstrip() + "..."
    state_line = f"STATE={state.name}: {hint}" if hint else f"STATE={state.name}"
    prefix = "Other registered tools (callable now, same call format): "

    # the header carries the dispatch rules while the budget allows it: with a
    # budget too small for them the state line alone still leads the listing,
    # because naming the tools matters more than the note about them
    header_options = [f"{state_line}\n{CATALOG_DISPATCH_NOTE}", state_line]
    header = header_options[0]
    for candidate in header_options:
        if len(_catalog_text(candidate, [], list(order), prefix)) <= limit:
            header = candidate
            break

    body: list[str] = []
    named = list(order)
    tier: dict[str, str] = {}
    for name in order:
        spec = registry.get(name)
        full = _entry_line(name, spec, str(getattr(spec, "description", "") or ""))
        compact = _entry_line(name, spec, _gist(str(getattr(spec, "description", "") or "")))
        rest = [item for item in named if item != name]
        for kind, line in (("full", full), ("compact", compact)):
            if len(_catalog_text(header, body + [line], rest, prefix)) <= limit:
                body.append(line)
                named = rest
                tier[name] = kind
                break

    text = _catalog_text(header, body, named, prefix)
    return ToolCatalog(
        state=state,
        order=tuple(order),
        full=tuple(name for name in order if tier.get(name) == "full"),
        compact=tuple(name for name in order if tier.get(name) == "compact"),
        named=tuple(name for name in order if name not in tier),
        text=text,
        limit=int(limit),
    )


# ---------------------------------------------------------------------------
# prompt examples: one source for the doc, the prompts and the schema check
# ---------------------------------------------------------------------------

#: The example calls the prompts show the model. They are rendered verbatim
#: into ``TOOL_CALL_DOC`` and the runtime cites the first one in its turn
#: prompt, and the test suite validates every entry against the schema of the
#: registered tool of the same name - so an example cannot drift away from the
#: schema and the tool it is supposed to demonstrate.
PROMPT_EXAMPLES: tuple[dict, ...] = (
    {"tool": "observe", "args": {"mode": "lookaround"}},
    {"tool": "ground_target", "args": {"instruction": "the red book"}},
    {"tool": "go_to_viewpoint", "args": {"viewpoint_id": "vp_3"}},
    {"tool": "finish", "args": {"reason": "target verified", "success": True}},
)


def schema_problems(schema: Any, args: Any) -> list[str]:
    """Minimal JSON-schema check of one tool call's args (``[]`` == valid).

    Covers the subset every ``ToolSpec`` of this package declares -
    ``type: object`` with ``properties`` / ``required`` / ``enum`` / ``items``
    plus ``type`` lists and ``default`` (an omitted key with a default is not
    missing). It exists so the prompt examples and the schemas can be checked
    against each other as the one interface they are:
    ``schema_problems(spec.input_schema, example["args"])`` must be empty for
    every example a prompt shows.
    """
    if not isinstance(schema, Mapping):
        return []
    problems: list[str] = []
    properties = schema.get("properties")
    properties = dict(properties) if isinstance(properties, Mapping) else {}
    if args is None:
        args = {}
    if not isinstance(args, Mapping):
        return [f"arguments must be an object, got {type(args).__name__}"]
    for key in schema.get("required") or ():
        if key in args:
            continue
        declared = properties.get(key)
        if isinstance(declared, Mapping) and "default" in declared:
            continue  # the schema supplies the value the argument would have had
        problems.append(f"missing required argument: {key}")
    for key, value in args.items():
        declared = properties.get(key)
        if not isinstance(declared, Mapping):
            continue
        problems.extend(_value_problems(str(key), value, declared))
    return problems


def _value_problems(key: str, value: Any, declared: Mapping) -> list[str]:
    """Type/enum/items problems of one argument value (unknown keys pass)."""
    problems: list[str] = []
    types = declared.get("type")
    if isinstance(types, str):
        allowed = (types,)
    elif isinstance(types, (list, tuple)):
        allowed = tuple(str(item) for item in types)
    else:
        allowed = ()
    if allowed and not any(_matches_type(value, name) for name in allowed):
        problems.append(f"{key} must be {allowed}, got {type(value).__name__}")
        return problems  # a value of the wrong type has no meaningful items check
    enum = declared.get("enum")
    if isinstance(enum, (list, tuple)) and enum and value not in enum:
        problems.append(f"{key}={value!r} is not one of {list(enum)}")
    items = declared.get("items")
    if isinstance(value, (list, tuple)) and isinstance(items, Mapping):
        for index, item in enumerate(value):
            problems.extend(_value_problems(f"{key}[{index}]", item, items))
    return problems


def _matches_type(value: Any, name: str) -> bool:
    if name == "string":
        return isinstance(value, str)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "array":
        return isinstance(value, (list, tuple))
    if name == "object":
        return isinstance(value, Mapping)
    if name == "null":
        return value is None
    return True  # an unknown type name constrains nothing


@dataclass(frozen=True)
class EvidenceRef:
    kind: str  # "image" | "observation" | "memory" | "map" | "trajectory"
    ref: str  # e.g. "obs_83", "frames/step_012.png", "hyp_7"
    note: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "ref": self.ref, "note": self.note}


@dataclass
class ToolResult:
    status: str
    data: dict
    evidence: list[EvidenceRef] = field(default_factory=list)
    #: Model-facing rendering override (``to_prompt``). A merged batch uses it to
    #: keep one compact result block instead of inlining every inner payload;
    #: empty means the default structured field rendering.
    prompt_text: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "data": self.data,
            "evidence": [e.to_dict() for e in self.evidence],
        }

    def to_prompt(self) -> str:
        """Compact structured text for the VLM context (goal.md: structured
        fields + evidence references, not prose outcomes)."""
        if self.prompt_text:
            return self.prompt_text
        parts = [f"[{self.status}]"]
        for key, value in self.data.items():
            parts.append(f"{key}={value}")
        if self.evidence:
            refs = ", ".join(f"{e.kind}:{e.ref}" for e in self.evidence[:6])
            parts.append(f"evidence=[{refs}]")
        return " ".join(parts)


@dataclass
class ToolContext:
    """Runtime state threaded through every tool call (episode-scoped)."""

    world: Any
    memory: Any = None  # memory.ObservationLog or None
    episode_id: str = "ep_0"
    observation_seq: int = 0
    budget: dict = field(default_factory=lambda: {"tool_calls_left": 40})
    log: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def next_obs_id(self) -> str:
        """Allocate the episode's next ``obs_N`` id.

        The id is reserved in the observation store
        (``ctx.extra["observation_store"]``, ``vlm_nav.observations``) when one
        exists, so the id this context hands out can never collide with one the
        store minted for another producer - the two share one namespace.
        """
        self.observation_seq += 1
        obs_id = f"obs_{self.observation_seq}"
        store = self.extra.get(OBSERVATION_STORE_KEY) if isinstance(self.extra, dict) else None
        reserve = getattr(store, "reserve_id", None)
        if callable(reserve):
            try:
                reserve(obs_id)
            except Exception:  # noqa: BLE001 - id minting must never fail a tool
                pass
        return obs_id


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict  # JSON-schema subset: {type: object, properties: {...}, required: [...]}
    handler: Callable[[ToolContext, dict], ToolResult]
    side_effects: bool = False  # pose-changing tools are serialized
    cacheable: bool = True  # repeated identical calls may be served from the context cache


# ---------------------------------------------------------------------------
# repeated-call cache (goal.md section 5)
# ---------------------------------------------------------------------------

#: ``ctx.extra`` key holding the episode's ``ToolCallCache``.
TOOL_CACHE_KEY = "tool_cache"
#: Default entry budget of the cache; ``ctx.extra["tool_cache_limit"]`` may
#: tighten or widen it before the first call of the episode.
DEFAULT_TOOL_CACHE_LIMIT = 32
#: ``data["hint"]`` of a cache hit: goal.md requires a repeated call with no new
#: evidence to be recognisable as such instead of looking like fresh evidence.
REPEAT_HINT = "repeat query, no new evidence"


@dataclass
class CacheEntry:
    """One cached tool result plus the call that produced it.

    ``hits`` counts how often this exact call was replayed, so an episode
    summary can name the calls the model kept repeating.
    """

    tool: str
    args: dict
    result: ToolResult
    call: int  # 1-based ordinal of the producing call in ToolContext.log
    hits: int = 0


class ToolCallCache:
    """Bounded, episode-scoped cache of read-only tool results (goal.md 5).

    goal.md asks that a repeated call which cannot produce new evidence either
    hit a cache or come back with a duplicate hint, instead of spending tool
    budget and VLM tokens on the same answer again. The cache lives in
    ``ctx.extra[TOOL_CACHE_KEY]`` so its scope is exactly one ``ToolContext``
    (one episode); ``ToolRegistry.call`` owns the policy - it stores only
    results of ``cacheable``, read-only (``side_effects=False``) tools, never a
    ``failed`` result, and drops the entries as soon as the episode state
    advances (a side-effecting call, or a result carrying new evidence).

    ``limit`` bounds the number of entries (oldest evicted first); the hit and
    eviction counters feed :func:`tool_cache_stats` for the episode summary.
    """

    def __init__(self, limit: int = DEFAULT_TOOL_CACHE_LIMIT):
        self.limit = max(1, int(limit))
        self._entries: dict[str, CacheEntry] = {}
        self.hits = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def key(name: str, args: dict | None) -> str:
        """Canonical key of one call: tool name + order-insensitive JSON args."""
        payload = json.dumps(
            dict(args or {}), sort_keys=True, default=str, separators=(",", ":")
        )
        return f"{name}:{payload}"

    def get(self, name: str, args: dict | None) -> CacheEntry | None:
        """The entry of an identical earlier call, or None (counts the hit)."""
        entry = self._entries.get(self.key(name, args))
        if entry is None:
            return None
        entry.hits += 1
        self.hits += 1
        return entry

    def put(self, name: str, args: dict | None, result: ToolResult, call: int) -> CacheEntry:
        """Store a result as the newest entry, evicting the oldest over budget."""
        key = self.key(name, args)
        entry = CacheEntry(str(name), dict(args or {}), result, int(call))
        self._entries.pop(key, None)  # a re-put becomes the newest entry
        self._entries[key] = entry
        while len(self._entries) > self.limit:
            self._entries.pop(next(iter(self._entries)))
            self.evictions += 1
        return entry

    def clear(self) -> int:
        """Drop every entry (the episode state advanced); returns the count."""
        dropped = len(self._entries)
        self._entries.clear()
        return dropped

    def hit_result(self, entry: CacheEntry) -> ToolResult:
        """Model-facing result of a cache hit: the original structured fields
        plus ``cached_from_turn`` and the repeat hint (goal.md 5).

        The original citations are kept - they are still the references behind
        the answer - and the runtime counts ``STATUS_CACHED`` as "no new
        evidence", so a replay never looks like progress.
        """
        data = dict(entry.result.data)
        if data.get("hint"):
            data["tool_hint"] = data["hint"]  # keep the tool's own advice
        data["cached_from_turn"] = entry.call
        data["hint"] = REPEAT_HINT
        return ToolResult(STATUS_CACHED, data, list(entry.result.evidence))

    def stats(self) -> dict:
        return {
            "entries": len(self._entries),
            "limit": self.limit,
            "hits": self.hits,
            "evictions": self.evictions,
        }


def tool_cache(ctx: ToolContext, limit: int | None = None) -> ToolCallCache:
    """The context's repeated-call cache, created on first use.

    ``ctx.extra[TOOL_CACHE_KEY]`` holds it for the episode; a value of an
    unexpected type (a hand-built or deserialized ``extra``) is replaced instead
    of trusted. ``limit`` - or ``ctx.extra["tool_cache_limit"]`` set before the
    first call - bounds the entry count of this episode.
    """
    if not isinstance(ctx.extra, dict):
        ctx.extra = {}
    cache = ctx.extra.get(TOOL_CACHE_KEY)
    if not isinstance(cache, ToolCallCache):
        if limit is None:
            limit = ctx.extra.get("tool_cache_limit", DEFAULT_TOOL_CACHE_LIMIT)
        cache = ToolCallCache(limit)
        ctx.extra[TOOL_CACHE_KEY] = cache
    return cache


def clear_tool_cache(ctx: ToolContext) -> int:
    """Invalidate the episode's repeated-call cache; returns the dropped count."""
    return tool_cache(ctx).clear()


def tool_cache_stats(ctx: ToolContext) -> dict:
    """Cache counters (entries/limit/hits/evictions) for the episode summary."""
    return tool_cache(ctx).stats()


# ---------------------------------------------------------------------------
# completion evidence: the trail behind "success": true (goal.md P4)
# ---------------------------------------------------------------------------

#: ``ctx.extra`` key of the episode's evidence pack. Tools that produce
#: completion-relevant evidence (a verification, a grounding, an arrival) record
#: it here, and the tools that read the chain (``finish``) judge exactly this
#: pack - so a local call, an MCP client and the runner all see one trail.
EVIDENCE_PACK_KEY = "evidence_pack"
#: How many records of each kind the pack keeps (bounded episode state).
EVIDENCE_PACK_CAP = 16

COMPLETION_VERIFICATION = "verification"
COMPLETION_ARRIVAL = "arrival"
COMPLETION_IDENTITY = "identity"

#: Tools whose answer grounds a target identity (``cand_N``).
IDENTITY_TOOLS = ("ground_target", "search_target")
#: Tools whose answer flies to a place; ``outcome == "arrived"`` is the arrival.
ARRIVAL_TOOLS = ("approach_candidate", "go_to_viewpoint", "return_to_observation")
#: Outcomes that mean "the drone is there".
ARRIVAL_OUTCOMES = ("arrived",)

MISSING_VERIFICATION = "no verify_goal call of this episode returned verified=true"
MISSING_ARRIVAL = "no arrival evidence (a movement call with outcome=arrived)"
MISSING_IDENTITY = "no grounded candidate (cand_N from ground_target/search_target)"
MISSING_REFERENCE = "the evidence carries no reference (kind:ref)"

#: The ``finish`` contract of both adapters: the outcome the model declares.
FINISH_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {
            "type": "string",
            "description": "one short sentence: why the episode ends here",
        },
        "success": {
            "type": "boolean",
            "description": (
                "true only with evidence (a verify_goal that returned verified=true, "
                "or a grounded candidate plus an arrival); false is the honest give-up"
            ),
        },
    },
    "required": ["reason"],
}


def evidence_pack(ctx: ToolContext) -> dict:
    """The episode's evidence pack, repaired into ``{kind: [records]}``.

    The pack is plain episode state (``ctx.extra[EVIDENCE_PACK_KEY]``), never a
    second source of truth: the tools that produce the evidence append here, and
    the completion check reads what they appended.
    """
    if not isinstance(ctx.extra, dict):
        ctx.extra = {}
    pack = ctx.extra.get(EVIDENCE_PACK_KEY)
    if not isinstance(pack, dict):
        pack = {}
        ctx.extra[EVIDENCE_PACK_KEY] = pack
    for key in ("verifications", "arrivals", "candidates"):
        if not isinstance(pack.get(key), list):
            pack[key] = []
    return pack


def _ref_dicts(evidence: Any) -> list[dict]:
    """Evidence references normalized to ``{"kind", "ref", "note"}`` dicts."""
    refs: list[dict] = []
    for item in evidence or ():
        if isinstance(item, EvidenceRef):
            refs.append(item.to_dict())
        elif isinstance(item, Mapping) and item.get("kind") and item.get("ref") is not None:
            refs.append({"kind": str(item["kind"]), "ref": str(item["ref"]),
                         "note": str(item.get("note") or "")})
    return refs


def _append_pack(pack: dict, key: str, record: dict) -> dict:
    items = pack.setdefault(key, [])
    items.append(record)
    del items[:-EVIDENCE_PACK_CAP]
    return record


def record_verification(
    ctx: ToolContext, *, verified: Any, observation_id: str = "", checks: Any = None,
    evidence: Any = (), reason: str = "", call: int | None = None,
) -> dict:
    """Record one completion check (``verify_goal``) in the episode's pack."""
    record = {
        "verified": verified,
        "observation_id": str(observation_id or ""),
        "checks": dict(checks) if isinstance(checks, Mapping) else {},
        "reason": str(reason or ""),
        "evidence": _ref_dicts(evidence),
        "call": int(call if call is not None else len(getattr(ctx, "log", []) or ()) + 1),
    }
    return _append_pack(evidence_pack(ctx), "verifications", record)


def record_candidate(
    ctx: ToolContext, *, candidate_id: str, observation_id: str = "", position: Any = None,
    evidence: Any = (), call: int | None = None,
) -> dict:
    """Record that a tool grounded one candidate identity."""
    record = {
        "candidate_id": str(candidate_id or ""),
        "observation_id": str(observation_id or ""),
        "position": list(position) if position is not None else None,
        "evidence": _ref_dicts(evidence),
        "call": int(call if call is not None else len(getattr(ctx, "log", []) or ()) + 1),
    }
    return _append_pack(evidence_pack(ctx), "candidates", record)


def record_arrival(
    ctx: ToolContext, *, task_id: str = "", target_id: str = "", intent: str = "",
    outcome: str = "arrived", observation_id: str = "", position: Any = None,
    remaining_m: Any = None, evidence: Any = (), call: int | None = None,
) -> dict:
    """Record that a movement call reached its place (``outcome=arrived``)."""
    record = {
        "task_id": str(task_id or ""),
        "target_id": str(target_id or ""),
        "intent": str(intent or ""),
        "outcome": str(outcome or ""),
        "observation_id": str(observation_id or ""),
        "position": list(position) if position is not None else None,
        "remaining_m": None if remaining_m is None else float(remaining_m),
        "evidence": _ref_dicts(evidence),
        "call": int(call if call is not None else len(getattr(ctx, "log", []) or ()) + 1),
    }
    return _append_pack(evidence_pack(ctx), "arrivals", record)


def evidence_records_from_pack(pack: Any) -> list[dict]:
    """The pack as the record shape :func:`completion_evidence` judges."""
    if not isinstance(pack, Mapping):
        return []
    records: list[dict] = []
    for item in pack.get("verifications") or ():
        if not isinstance(item, Mapping):
            continue
        records.append({
            "tool": "verify_goal",
            "status": STATUS_OK,
            "data": {"verified": item.get("verified"), "checks": dict(item.get("checks") or {}),
                     "observation_id": item.get("observation_id", "")},
            "evidence": list(item.get("evidence") or []),
            "call": item.get("call"),
        })
    for item in pack.get("candidates") or ():
        if not isinstance(item, Mapping):
            continue
        records.append({
            "tool": "ground_target",
            "status": STATUS_OK,
            "data": {"candidate_id": item.get("candidate_id", ""),
                     "observation_id": item.get("observation_id", ""),
                     "position_flu": item.get("position")},
            "evidence": list(item.get("evidence") or []),
            "call": item.get("call"),
        })
    for item in pack.get("arrivals") or ():
        if not isinstance(item, Mapping):
            continue
        records.append({
            "tool": str(item.get("intent") or "approach_candidate"),
            "status": STATUS_OK,
            "data": {"outcome": item.get("outcome", ""), "task_id": item.get("task_id", ""),
                     "task_status": "arrived" if item.get("outcome") in ARRIVAL_OUTCOMES else "",
                     "candidate_id": item.get("target_id", ""),
                     "last_observation_id": item.get("observation_id", "")},
            "evidence": list(item.get("evidence") or []),
            "call": item.get("call"),
        })
    return records


def evidence_records_from_trace(entries: Any) -> list[dict]:
    """Trace turns as the record shape :func:`completion_evidence` judges.

    The runner keeps one entry per turn (``tool``, ``status``, ``data``,
    ``evidence``); this adapter is what lets the runner judge the *same* chain
    the tools judged in-loop, on both runtimes.
    """
    records: list[dict] = []
    for entry in entries or ():
        if not isinstance(entry, Mapping):
            continue
        tool = entry.get("tool")
        if not isinstance(tool, str) or not tool:
            continue
        data = entry.get("data")
        records.append({
            "tool": tool,
            "status": str(entry.get("status") or STATUS_OK),
            "data": dict(data) if isinstance(data, Mapping) else {},
            "evidence": _ref_dicts(entry.get("evidence") or ()),
            "call": entry.get("turn"),
        })
    return records


def _chain_of(records: list[dict]) -> dict:
    """One completion chain (verification, or identity + arrival) or an empty one."""
    chain = {"kind": "", "evidence_trail": [], "candidate_ids": [], "arrivals": 0,
             "verifications": 0}
    for record in records:
        tool = str(record.get("tool") or "")
        data = record.get("data") if isinstance(record.get("data"), Mapping) else {}
        refs = _ref_dicts(record.get("evidence") or ())
        if tool == "verify_goal":
            chain["verifications"] += 1
            if data.get("verified") is True:
                chain["kind"] = COMPLETION_VERIFICATION
                chain["evidence_trail"] = refs
                return chain
    candidate_ids: list[str] = []
    candidate_refs: list[dict] = []
    arrivals: list[dict] = []
    for record in records:
        tool = str(record.get("tool") or "")
        data = record.get("data") if isinstance(record.get("data"), Mapping) else {}
        refs = _ref_dicts(record.get("evidence") or ())
        if tool in IDENTITY_TOOLS:
            candidate_id = str(data.get("candidate_id") or "")
            if not candidate_id:
                candidates = data.get("candidates")
                if isinstance(candidates, list) and candidates:
                    top = candidates[0] if isinstance(candidates[0], Mapping) else {}
                    candidate_id = str(top.get("id") or "")
            if candidate_id:
                if candidate_id not in candidate_ids:
                    candidate_ids.append(candidate_id)
                candidate_refs.extend(refs)
        if tool in ARRIVAL_TOOLS and str(data.get("outcome") or "") in ARRIVAL_OUTCOMES:
            arrivals.append({"refs": refs})
        elif tool == "task_status" and data.get("completed") and data.get("arrived"):
            arrivals.append({"refs": refs or _ref_dicts(data.get("arrival_evidence") or ())})
    chain["candidate_ids"] = candidate_ids
    chain["arrivals"] = len(arrivals)
    if candidate_ids and arrivals:
        chain["kind"] = "identity+arrival"
        chain["evidence_trail"] = candidate_refs + [ref for item in arrivals
                                                    for ref in item["refs"]]
    return chain


def completion_evidence(records: Any) -> dict:
    """Judge the completion chain behind a ``"success": true`` claim.

    Returns ``{verified, kind, evidence_trail, missing, ...}``: ``verified`` is
    true when the trail holds a verification that passed, or a grounded
    candidate *and* an arrival - and the chosen chain carries at least one
    evidence reference, because a claim has to point at something. Everything
    else is reported in ``missing``, so the caller can say what to do instead.
    """
    records = [record for record in (records or ()) if isinstance(record, Mapping)]
    chain = _chain_of(records)
    verified = bool(chain["kind"]) and bool(chain["evidence_trail"])
    missing: list[str] = []
    if chain["kind"] and not chain["evidence_trail"]:
        missing.append(MISSING_REFERENCE)
    if not chain["kind"]:
        if not chain["verifications"]:
            missing.append(MISSING_VERIFICATION)
        if not chain["candidate_ids"]:
            missing.append(MISSING_IDENTITY)
        if not chain["arrivals"]:
            missing.append(MISSING_ARRIVAL)
    trail: list[dict] = []
    seen: set[tuple] = set()
    for ref in chain["evidence_trail"]:
        key = (ref.get("kind"), ref.get("ref"))
        if key in seen:
            continue
        seen.add(key)
        trail.append(ref)
    return {
        "verified": bool(verified),
        "kind": str(chain["kind"]),
        "evidence_trail": trail,
        "missing": missing,
        "candidate_ids": list(chain["candidate_ids"]),
        "arrivals": int(chain["arrivals"]),
        "verifications": int(chain["verifications"]),
    }


# ---------------------------------------------------------------------------
# batch_read: several read-only checks in one call (goal.md P4)
# ---------------------------------------------------------------------------

BATCH_TOOL_NAME = "batch_read"
#: Calls one batch may carry (a batch is one turn's read-in, not an episode).
MAX_BATCH_CALLS = 6

BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "description": "a registered read-only tool"},
                    "args": {"type": "object", "description": "that tool's arguments"},
                },
                "required": ["tool"],
            },
            "description": (
                "the read-only checks to run, in order (observe, query_memory, "
                "compare_candidates, inspect_region, check_path, ...)"
            ),
        },
        "reason": {"type": "string", "description": "why these checks belong together"},
    },
    "required": ["calls"],
}

BATCH_DESCRIPTION = (
    "Run several read-only checks in one call and get their results merged "
    "(one model turn instead of one per check): pass calls=[{tool, args}, ...] "
    "with any registered read-only tool. Pose-changing tools are refused here - "
    "movement is serial, so call approach_candidate / go_to_viewpoint / "
    "return_to_observation on its own with its task_id. Each call inside the "
    "batch costs its own tool budget; the batch wrapper costs none."
)


def parse_batch_calls(args: Any) -> tuple[list[tuple[str, dict]], list[str]]:
    """``(calls, problems)`` of one batch request - empty problems mean valid."""
    if not isinstance(args, Mapping) or "calls" not in args:
        return [], ["missing required argument: calls"]
    calls = args.get("calls")
    if not isinstance(calls, (list, tuple)):
        return [], ["calls must be an array of {tool, args} objects"]
    if not calls:
        return [], ["calls is empty: name at least one read-only check"]
    if len(calls) > MAX_BATCH_CALLS:
        return [], [f"calls carries {len(calls)} entries; at most {MAX_BATCH_CALLS}"]
    parsed: list[tuple[str, dict]] = []
    problems: list[str] = []
    for index, item in enumerate(calls):
        if not isinstance(item, Mapping):
            problems.append(f"calls[{index}] must be an object")
            continue
        name = item.get("tool")
        if not isinstance(name, str) or not name.strip():
            problems.append(f"calls[{index}].tool must be a tool name")
            continue
        inner = item.get("args")
        if inner is None:
            inner = {}
        if not isinstance(inner, Mapping):
            problems.append(f"calls[{index}].args must be an object")
            continue
        parsed.append((name.strip(), dict(inner)))
    return parsed, problems


def _scalar_summary(data: Any, limit: int = 2) -> str:
    """``k=v`` of the first scalar fields of a result (for the compact line)."""
    if not isinstance(data, Mapping):
        return ""
    parts: list[str] = []
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool)):
            if isinstance(value, float):
                value = round(value, 2)
            text = str(value)
            if len(text) > 24:
                text = text[:21] + "..."
            parts.append(f"{key}={text}")
            if len(parts) >= limit:
                break
    return ", ".join(parts)


def batch_read_spec(registry: "ToolRegistry") -> ToolSpec:
    """``ToolSpec`` of the batch verb, bound to the registry it runs against."""

    def handler(ctx: ToolContext, args: dict) -> ToolResult:
        return registry.batch(args, ctx)

    return ToolSpec(
        name=BATCH_TOOL_NAME,
        description=BATCH_DESCRIPTION,
        input_schema=BATCH_SCHEMA,
        handler=handler,
        side_effects=False,
        cacheable=False,
    )


def register_batch_tool(registry: "ToolRegistry") -> ToolSpec:
    """Register ``batch_read`` on ``registry`` (idempotent) and return its spec."""
    if BATCH_TOOL_NAME in registry.names():
        return registry.get(BATCH_TOOL_NAME)
    spec = batch_read_spec(registry)
    registry.register(spec)
    return spec


class ToolRegistry:
    def __init__(self, *, state_source: Any = None, catalog_limit: int | None = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        #: ``() -> Mapping``: the episode context the catalogue is rendered for.
        self.state_source = state_source
        self.catalog_limit = catalog_limit

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    # -- state-aware catalogue -------------------------------------------- #
    def attach_state(self, source: Any, limit: int | None = None) -> "ToolRegistry":
        """Bind the episode context the catalogue is rendered for.

        ``source`` is a callable returning the live ``ctx.extra`` mapping (the
        adapters pass ``lambda: ctx.extra``), so the catalogue follows the
        episode without the registry holding the runtime. ``limit`` bounds the
        rendered characters (``catalog_limit``); the runtime's own tools budget
        must be at least as wide, or it truncates the listing again.
        """
        self.state_source = source
        if limit is not None:
            self.catalog_limit = max(120, int(limit))
        return self

    def tool_state(self) -> ToolState | None:
        """The episode state of the attached source, or None without one."""
        source = self.state_source
        if source is None:
            return None
        try:
            extra = source()
        except Exception:  # noqa: BLE001 - a broken source must not break a prompt
            return None
        return tool_state(extra)

    def catalog(self, state: ToolState | None = None,
                limit: int | None = None) -> ToolCatalog:
        """``ToolCatalog`` of this registry (state-aware when one is attached)."""
        state = state or self.tool_state() or ToolState(
            TOOL_STATE_START, TOOL_STATE_HINTS[TOOL_STATE_START]
        )
        return tool_catalog(self, state=state, limit=limit if limit is not None
                            else self.catalog_limit)

    def specs_text(self) -> str:
        """Tool descriptions + schemas for the system prompt.

        Without an attached state source the full catalogue is rendered (the
        historical format); with one, the state-aware listing of
        ``ToolCatalog.text`` is returned instead - ordered by the state and
        bounded by ``catalog_limit``, so the tools a state calls arrive whole.
        """
        if self.state_source is None:
            return "\n".join(
                _entry_line(name, self._tools[name], str(self._tools[name].description))
                for name in self.names()
            )
        return self.catalog().text

    def call(self, name: str, args: dict, ctx: ToolContext) -> ToolResult:
        """Execute one tool with budget enforcement, timing and structured log.

        Repeated-call policy (goal.md 5): an identical name+args call to a
        ``cacheable`` read-only tool is answered from the episode's
        ``ToolCallCache`` as ``STATUS_CACHED`` with the "repeat query, no new
        evidence" hint - recomputing an answer the model already has costs
        budget and VLM tokens without adding evidence. The cache is dropped
        whenever the episode state advances: any ``side_effects=True`` call
        (pose/map/agent-state mutation) and any result carrying evidence (a new
        observation can change what an earlier read meant). Cache hits still
        consume one budget unit and are logged like any other call, so the
        audit trail and the budget see every model request.

        ``batch_read`` is dispatched here rather than through its handler: the
        wrapper itself is free, and every call *inside* the batch pays its own
        budget unit through this same method (so a batch of n read-only checks
        costs n, not n+1 - and each inner call is logged and cached exactly like
        a call the model made on its own).
        """
        if name == BATCH_TOOL_NAME and name in self._tools:
            return self.batch(args, ctx)
        if ctx.budget.get("tool_calls_left", 0) <= 0:
            return ToolResult(STATUS_FAILED, {"reason": "budget_exhausted"})
        spec = self.get(name)
        args = dict(args or {})
        cache = tool_cache(ctx)
        # getattr keeps duck-typed specs (a hand-built stub) working
        cacheable = bool(getattr(spec, "cacheable", True)) and not spec.side_effects
        t0 = time.time()
        if cacheable:
            entry = cache.get(name, args)
            if entry is not None:
                result = cache.hit_result(entry)
                ctx.budget["tool_calls_left"] = ctx.budget.get("tool_calls_left", 1) - 1
                ctx.log.append(self._log_entry(name, args, result, t0))
                return result
        try:
            result = spec.handler(ctx, args)
        except Exception as exc:  # tools never crash the runtime
            result = ToolResult(STATUS_FAILED, {"error": f"{type(exc).__name__}: {exc}"})
        ctx.budget["tool_calls_left"] = ctx.budget.get("tool_calls_left", 1) - 1
        if spec.side_effects or result.evidence:
            cache.clear()  # state advanced: earlier read answers are no longer current
        if cacheable and result.status != STATUS_FAILED:
            cache.put(name, args, result, len(ctx.log) + 1)
        ctx.log.append(self._log_entry(name, args, result, t0))
        return result

    # -- batch_read ------------------------------------------------------- #
    def batch(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Run the read-only calls of one ``batch_read`` request, merged.

        Every call runs through :meth:`call` (budget, cache, audit log), so the
        merged answer is exactly what the model would have got one call at a
        time - minus the turns. A call that is not read-only is *refused* with
        its own serial next step instead of being executed, and a refused entry
        never voids the evidence the other calls produced; the batch status is
        ``failed`` only to say that not everything ran.
        """
        calls, problems = parse_batch_calls(args)
        if problems:
            return ToolResult(
                STATUS_FAILED,
                {
                    "reason": "malformed_batch",
                    "problems": problems,
                    "known_tools": self.names()[:24],
                    "hint": (
                        'pass calls=[{"tool": "<read-only tool>", "args": {...}}, ...] '
                        "with 1..%d entries" % MAX_BATCH_CALLS
                    ),
                },
            )
        if ctx.budget.get("tool_calls_left", 0) <= 0:
            return ToolResult(
                STATUS_FAILED,
                {"reason": "budget_exhausted",
                 "hint": "no tool budget left for the batch's calls"},
            )

        results: list[dict] = []
        evidence: list[EvidenceRef] = []
        seen_refs: set[tuple[str, str]] = set()
        counts = {STATUS_OK: 0, STATUS_EMPTY: 0, STATUS_CACHED: 0, STATUS_FAILED: 0}
        information = False
        # the wrapper is logged where it ran: its entry comes first, the calls
        # inside it follow (they are ordinary calls of this episode)
        log_index = len(ctx.log)
        t0 = time.time()
        for tool, inner_args in calls:
            entry: dict[str, Any] = {"tool": tool, "args": dict(inner_args)}
            spec = self._tools.get(tool)
            if tool == BATCH_TOOL_NAME:
                entry.update(
                    status=STATUS_FAILED,
                    data={"reason": "nested_batch",
                          "hint": "a batch does not contain another batch"},
                    evidence=[],
                )
            elif spec is None:
                entry.update(
                    status=STATUS_FAILED,
                    data={"reason": "unknown_tool", "known_tools": self.names()[:24]},
                    evidence=[],
                )
            elif bool(getattr(spec, "side_effects", False)):
                entry.update(
                    status=STATUS_FAILED,
                    data={
                        "reason": "not_read_only",
                        "next_action": {"tool": tool, "args": dict(inner_args)},
                        "hint": (
                            f"{tool} changes the pose: call it on its own (movement is "
                            "serial, one leg and one task_id per call)"
                        ),
                    },
                    evidence=[],
                )
            else:
                result = self.call(tool, inner_args, ctx)
                entry.update(
                    status=result.status,
                    data=dict(result.data),
                    evidence=[ref.to_dict() for ref in result.evidence],
                )
                if result.status == STATUS_OK and not result.data.get("cached_from_turn"):
                    information = True
                for ref in result.evidence:
                    key = (ref.kind, ref.ref)
                    if key in seen_refs:
                        continue
                    seen_refs.add(key)
                    evidence.append(ref)
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
            results.append(entry)

        failed = counts.get(STATUS_FAILED, 0)
        data: dict[str, Any] = {
            "calls": len(results),
            "count": len(results),
            "ok": counts.get(STATUS_OK, 0),
            "empty": counts.get(STATUS_EMPTY, 0),
            "cached": counts.get(STATUS_CACHED, 0),
            "failed": failed,
            "results": results,
            "tools": [entry["tool"] for entry in results],
            "progress_kind": PROGRESS_INFORMATION if information else PROGRESS_NONE,
        }
        if failed:
            refused = sorted({str(entry.get("data", {}).get("reason") or "failed")
                              for entry in results if entry["status"] == STATUS_FAILED})
            data["hint"] = (
                "not every call ran (" + ", ".join(refused) + "): see results[*].data.reason "
                "- a refused call names the call that can do the work"
            )
        elif counts.get(STATUS_CACHED) == len(results):
            data["hint"] = REPEAT_HINT
        summary = " | ".join(
            f"{entry['tool']}->{entry['status']}"
            + (f"({_scalar_summary(entry.get('data'))})" if entry.get("data") else "")
            for entry in results
        )
        prompt = (
            f"[{STATUS_FAILED if failed else STATUS_OK}] batch_read: {len(results)} calls "
            f"(ok={data['ok']} empty={data['empty']} cached={data['cached']} "
            f"failed={failed}) | {summary}"
        )
        if evidence:
            refs = ", ".join(f"{ref.kind}:{ref.ref}" for ref in evidence[:6])
            prompt += f" evidence=[{refs}]"
        result = ToolResult(
            STATUS_FAILED if failed else STATUS_OK, data, evidence,
            prompt_text=prompt[:600],
        )
        ctx.log.insert(log_index, self._log_entry(BATCH_TOOL_NAME, dict(args or {}), result, t0))
        return result

    @staticmethod
    def _log_entry(name: str, args: dict, result: ToolResult, t0: float) -> dict:
        return {
            "tool": name,
            "args": args,
            "status": result.status,
            "ms": round((time.time() - t0) * 1000, 1),
            "evidence": [e.to_dict() for e in result.evidence],
        }


def parse_tool_call(text: str) -> tuple[str, dict] | None:
    """Extract one {"tool": ..., "args": {...}} call from a model reply.

    Returns None when the reply contains no decodable call (the runtime then
    re-prompts once). Tolerates prose around the JSON object.
    """
    if not text:
        return None
    for i, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[i : j + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(data, dict) and "tool" in data:
                        tool = str(data.get("tool", "")).strip()
                        args = data.get("args") or {}
                        if isinstance(args, dict):
                            return tool, args
                    break
    return None


TOOL_CALL_DOC = """\
Reply with EXACTLY ONE JSON object per turn, either a tool call:
{"tool": "<name>", "args": {...}}
or the finish call:
{"tool": "finish", "args": {"reason": "<short>"}}
Do not narrate; only return the JSON object.
Ids are exact strings copied from a tool result (cand_N, vp_N, obs_N, nav_N,
plan_N); never invent one.
Worked examples - the arguments are exactly what their tool schema accepts:
{examples}

Typed ids - one meaning per type, never mixed:
- observation_id (obs_N): a recorded view - where it was captured, which way it
  faced, the image it holds. Ground that image, or fly back to that pose with
  return_to_observation.
- candidate_id (cand_N): one target instance inside an observation (from
  ground_target). Approach it with approach_candidate; never return_to_observation
  for it ("back to where that frame was shot" is NOT "go to the object in it").
- viewpoint_id (vp_N): an observation POSITION from propose_viewpoints. Reach it
  with go_to_viewpoint.
A wrong-typed id is an error and lists the ids of the type the argument wanted.

Reserved navigation tools - one per intent, now registered, call them by these
names:
- approach_candidate({"candidate_id": "cand_N"}): fly to the OBJECT in a frame;
  its position comes from the candidate, so ground_target first without one.
- go_to_viewpoint({"viewpoint_id": "vp_N"}): fly to an observation position.
- return_to_observation({"observation_id": "obs_N"}): fly back to the CAMERA pose
  that frame was shot from and restore its heading. A recalled observation is a
  navigable target by its own name; the legacy navigate({"target_id": "obs_N"})
  is deprecated, moves nothing and answers with exactly that call.
Each movement call flies at most one leg and answers with task_id, status
(ok|blocked|no_progress), outcome (arrived|partial|collision|stalled|
planner_failed|already_at_pose|...), progress_m, turn_deg, remaining_m,
blocked_reason, last_observation_id. already_at_pose moved nothing (no_progress):
never read it as arrival - do not repeat that call, take the next_action it gives
you instead. Continue with the same tool and its task_id; task_status /
cancel_task report and stop it.
Boxes/regions are [x1, y1, x2, y2] corners (normalized 0-1000), as the schemas
say; a box the tool returns can be fed back.
Call verify_goal before finish: answer {"success": true} only when a verify_goal
call of this episode returned verified=true. When a check is unmet, keep working
(approach_candidate / return_to_observation / go_to_viewpoint / observe /
query_memory) or finish with a reason but no success flag."""

# the worked examples come from PROMPT_EXAMPLES, the same list the tests check
# against the registered schemas and the runtime cites in its turn prompt
TOOL_CALL_DOC = TOOL_CALL_DOC.replace(
    "{examples}",
    "\n".join(json.dumps(example) for example in PROMPT_EXAMPLES),
)
