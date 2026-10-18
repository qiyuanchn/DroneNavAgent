"""MCP servers for the NavAgent tool runtime (goal.md section 6).

Two stdio services share one :class:`NavSession` (tool registry + runtime state
+ resource resolver):

- ``nav-environment`` (``--service env``): ``observe``, ``search_target``,
  ``propose_viewpoints``, ``check_path``, ``approach_candidate``,
  ``go_to_viewpoint``, ``return_to_observation``, ``verify_goal``, ``finish``,
  ``batch_read`` (the movement tools one per intent; the mixed-semantics
  ``navigate`` was retired; ``search_target`` is the search routine over the
  perception tools and may fly the drone itself; ``finish`` ends the episode
  with an outcome whose ``"success": true`` is only accepted with the episode's
  evidence chain, and ``batch_read`` packs several read-only checks into one
  call).
- ``nav-evidence`` (``--service evidence``): ``ground_target``,
  ``query_memory``, ``compare_candidates``, ``inspect_region``.

The MCP tool schema is ``ToolSpec.input_schema`` verbatim, the handler is
``ToolSpec.handler``, and every call goes through ``ToolRegistry.call`` (budget,
timing, ``ctx.log``). The direct-call adapter (``NavSession.call``) and the MCP
adapter therefore share one registry, one budget, one log and one evidence pack -
no second agent behaviour to keep in sync. Tool *discovery* is state-aware: both
``list_tools`` and ``service_inventory`` order the service's tools by the
episode state (``NavSession.catalog``, the same ranking the local catalogue
renders), so the tool a state needs is the one a client sees first.

Run standalone::

    python -m vlm_nav.mcp_server --service env --list
    python -m vlm_nav.mcp_server --service evidence --frames-dir out/frames

or embed it in the harness (same process, same context)::

    session = build_session(ctx)
    start_stdio_thread("env", session)

Tool implementations are merged from the parallel toolset modules
(``DEFAULT_TOOL_MODULES``: ``vlm_nav.tool_impl_perception`` +
``vlm_nav.tool_impl_navigation``, overridable with ``--tools-module`` /
``VLM_NAV_TOOLS_MODULE``). Providers are found by convention: ``register_*tools(
registry, ...)``, ``build_registry(ctx, ...)``, sequence exports such as
``TOOL_SPECS``, or per-tool factories like ``observe_spec``. A tool whose
implementation is not importable yet is registered as a stub: discovery and
schemas stay complete, the call returns
``{"status": "failed", "reason": "tool_not_implemented", "stub": true}``.

Resources are resolved by the runtime into real content - ``nav://obs/<id>/image``
returns PNG bytes (or JPEG/GIF/WebP passthrough), never a bare path::

    nav://obs                              index of recorded observations (json)
    nav://obs/<observation_id>/image       recorded frame (image/*)
    nav://obs/<observation_id>/meta        pose / meta of one observation (json)
    nav://map/bev?size=<n>                 top-down memory map (image/png)
    nav://map/state                        pose, target, distance, subgoal (json)
    nav://traj/<episode_id>                flown trajectory (json)
    nav://traj/<episode_id>/segment/<i>    one trajectory chunk (json)
    nav://candidates                       candidate target evidence (json)
    nav://episode/<episode_id>/log         tool call log + budget (json)

Runtime wiring the resolver reads (first match wins): frames recorded with
``record_observation(ctx, frame, pose=..., **meta)`` / ``ctx.extra["obs_store"]``,
the runtime's ``ctx.extra["images"]``, the perception tools'
``ctx.extra["perception.observations"]``, then ``ctx.memory`` (a mapping, or
``frame(obs_id)`` / ``image_path(obs_id)`` / ``get(obs_id)``), then
``ctx.extra["frames_dir"]`` (``obs_7.png`` or the harness ``step_007.png``);
candidates come from ``ctx.extra["<ns>.candidates"]`` or ``memory.candidates``,
the map/trajectory from ``ctx.world``.

Tool results over MCP carry the compact ``ToolResult.to_prompt()`` text, the
structured ``{status, data, evidence}`` payload, and - when the result refers to
an observation - the resolved frame as MCP image content (up to
``MAX_RESULT_IMAGES``); the direct-call adapter reads the same frames through
``session.resources.read("nav://obs/<id>/image")``.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import functools
import importlib
import importlib.util
import inspect
import io
import json
import math
import os
import struct
import sys
import threading
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import parse_qs, urlparse

import anyio
import numpy as np

from .tools import (
    BATCH_DESCRIPTION,
    BATCH_SCHEMA,
    BATCH_TOOL_NAME,
    DEFAULT_CATALOG_CHARS,
    FINISH_SCHEMA,
    STATUS_FAILED,
    TOOL_STATE_START,
    TOOL_STATE_HINTS,
    ToolCatalog,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    ToolState,
    tool_catalog,
    tool_state,
)

try:  # mcp is only needed by the serving paths, not by the direct-call adapter
    import mcp.types as mcp_types
    from mcp.server.lowlevel import Server
    from mcp.server.lowlevel.helper_types import ReadResourceContents
except ImportError as exc:  # pragma: no cover - depends on the environment
    mcp_types = None
    Server = None
    ReadResourceContents = None
    _MCP_IMPORT_ERROR: Exception | None = exc
else:
    _MCP_IMPORT_ERROR = None

IMAGE_MIME = "image/png"
JSON_MIME = "application/json"
#: The completion verb of the environment service (``vlm_nav.tool_impl_navigation``
#: registers the tool; ``AgentRuntime`` reads its ``success`` field).
FINISH_TOOL_NAME = "finish"

ENV_TOOL_NAMES = (
    "observe",
    "search_target",
    "propose_viewpoints",
    "check_path",
    "approach_candidate",
    "go_to_viewpoint",
    "return_to_observation",
    "verify_goal",
    # the completion claim (with the evidence chain) and the read-only batch are
    # part of the env vocabulary: an MCP-only client must be able to end an
    # episode with an outcome and to gather several read-only checks in one turn
    "finish",
    "batch_read",
)
EVIDENCE_TOOL_NAMES = ("ground_target", "query_memory", "compare_candidates", "inspect_region")

SERVICES: dict[str, tuple[str, tuple[str, ...]]] = {
    "env": ("nav-environment", ENV_TOOL_NAMES),
    "evidence": ("nav-evidence", EVIDENCE_TOOL_NAMES),
}
ALL_TOOL_NAMES = ENV_TOOL_NAMES + EVIDENCE_TOOL_NAMES
TOOL_SERVICE: dict[str, str] = {
    name: service for service, (_, names) in SERVICES.items() for name in names
}

# Resource categories owned by each service (goal.md section 6: state ownership
# over service count). Listing/templates are filtered per service; reads accept
# every URI so a client may follow a reference handed over by the other service.
RESOURCE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "env": ("obs", "map", "traj", "log"),
    "evidence": ("obs", "memory", "log"),
}

MAX_LISTED_OBSERVATIONS = 24
TRAJECTORY_SEGMENT_STEPS = 32
#: How many frames a tool result may carry over MCP. A ``lookaround`` capture
#: hands back one observation per view (8 by default), and a model can only
#: choose between the views it can actually see, so the evidence pack must be
#: able to carry a whole sweep - the same reason ``AgentRuntime.max_images``
#: exists on the local path.
MAX_RESULT_IMAGES = 8
STUB_MARKER = "[stub: implementation pending]"

#: First-version tool contracts (goal.md section 3). These schemas are what the
#: MCP services advertise when the toolset module is not importable yet; when it
#: is, the module's own ``ToolSpec.input_schema`` is used verbatim instead.
TOOL_CONTRACTS: dict[str, dict[str, Any]] = {
    "observe": {
        "description": (
            "Take a fresh observation at the current pose: one forward frame (the default), a "
            "360-degree look-around sweep, a downward look, or a narrow re-acquisition sweep. "
            "Returns the observation id, pose, target bearing/distance and image evidence, "
            "plus the visual context of the capture (view_index, coverage, expansion)."
        ),
        "side_effects": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["forward", "lookaround", "down", "narrow_scan"],
                    "default": "forward",
                    "description": "forward (default): single frame; lookaround: 360 sweep; down: downward look; narrow_scan: re-acquisition sweep",
                },
                "n_views": {"type": "integer", "minimum": 1, "maximum": 16},
                "pitch_down_deg": {"type": "number"},
                "note": {"type": "string", "description": "why this observation is needed"},
            },
            "required": [],
        },
    },
    "search_target": {
        "description": (
            "Search for the target the instruction names, end to end: sweep 360 degrees "
            "(observe lookaround), compare the swept views in one model call before "
            "grounding any of them (the comparison runs by default when the instruction "
            "names the target by a relation), ground the views it picked, magnify a view "
            "for small or distant targets, confirm the best candidate with "
            "compare_candidates - and when a round finds nothing, propose observation "
            "points, state what the new place is expected to add and fly one leg there "
            "(propose_viewpoints + go_to_viewpoint), then sweep again. Refuses to ground "
            "the same evidence twice "
            "(the anti-loop guard) and returns steps=[{step, why, obs_ids, result}] plus "
            "vlm_calls/cost, so every model call it made is visible. candidate_id is set "
            "when a candidate was found and confirmed; otherwise the status is empty and "
            "next_action says how to continue."
        ),
        "side_effects": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "what to find, in the words of the task ('the red backpack')",
                },
                "budget_turns": {
                    "type": "integer",
                    "description": "rounds inside this call (default 1): each round is sweep -> ground -> one leg to a better viewpoint",
                },
                "max_groundings": {
                    "type": "integer",
                    "description": "views grounded per round (default 3, each one a model call)",
                },
                "max_inspects": {
                    "type": "integer",
                    "description": "crops magnified per round for small or distant targets (default 1)",
                },
                "max_steps": {
                    "type": "integer",
                    "description": "steps the whole call may run inside itself (default 8): at the cap the call returns what it has and the episode keeps its budget",
                },
                "min_confidence": {
                    "type": "number",
                    "description": "a candidate below this confidence is reported as not confirmed",
                },
                "allow_move": {
                    "type": "boolean",
                    "description": "fly to the proposed viewpoint when a round finds nothing (default true)",
                },
                "mode": {
                    "type": "string",
                    "enum": ["lookaround", "forward", "pitch_down"],
                    "description": "how each round observes (default lookaround)",
                },
                "n_views": {"type": "integer", "description": "lookaround frames per sweep"},
                "viewpoints": {"type": "integer", "description": "how many observation points to propose"},
                "compare_views": {
                    "type": "string",
                    "enum": ["auto", "on", "off"],
                    "description": "auto (default): compare the swept views in one model call before grounding any of them when the instruction names the target by a relation ('next to the lamp'); on: always; off: ground in the swept order",
                },
                "compare_views_max": {
                    "type": "integer",
                    "description": "views one comparison call may carry (default 8)",
                },
                "compare_picks": {
                    "type": "integer",
                    "description": "compared views the comparison may pick for grounding (default 2)",
                },
                "observation_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "frames already on record to compare instead of sweeping (e.g. the observation ids a query_memory hit reported)",
                },
                "face_picked": {
                    "type": "boolean",
                    "description": "turn to a picked view the drone does not already face and ground the capture taken there (default true): a depth map belongs to the frame at the drone's own pose",
                },
            },
            "required": ["instruction"],
        },
    },
    "propose_viewpoints": {
        "description": (
            "Propose reachable observation viewpoints for the current target or candidate, "
            "with expected coverage and travel cost. Returns viewpoint ids usable with "
            "check_path and go_to_viewpoint."
        ),
        "side_effects": False,
        "input_schema": {
            "type": "object",
            "properties": {
                "target_id": {"type": "string", "description": "candidate id; defaults to the best live candidate"},
                "max_viewpoints": {"type": "integer", "minimum": 1, "maximum": 8},
                "radius_m": {"type": "number", "description": "preferred standoff distance"},
            },
            "required": [],
        },
    },
    "check_path": {
        "description": (
            "Check a candidate viewpoint before committing: returns reachability, path cost, "
            "unknown-region fraction and a blocking reason when unreachable."
        ),
        "side_effects": False,
        "input_schema": {
            "type": "object",
            "properties": {
                "viewpoint_id": {"type": "string"},
                "target_id": {"type": "string"},
            },
            "required": ["viewpoint_id"],
        },
    },
    "approach_candidate": {
        "description": (
            "Approach the target candidate cand_N that ground_target returned: fly AT MOST "
            "ONE leg (<= 8 m) per call and return at once with task_id, status "
            "(ok|blocked|no_progress), outcome, progress_m, turn_deg, remaining_m, "
            "blocked_reason and last_observation_id. The leg ends standoff_m short of the "
            "candidate's world position and calling again with task_id flies the next "
            "segment. This is the way to the OBJECT in an observation - not "
            "return_to_observation (that one flies to the camera pose)."
        ),
        "side_effects": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_id": {
                    "type": "string",
                    "description": "cand_N from ground_target (never pass obs_N here)",
                },
                "standoff_m": {
                    "type": "number",
                    "description": "how close to get to the candidate (default 1.2 m)",
                },
                "max_legs": {"type": "integer", "minimum": 1, "maximum": 64},
                "task_id": {"type": "string", "description": "continue an existing approach task (nav_N)"},
                "timeout_s": {"type": "number"},
            },
            "required": ["candidate_id"],
        },
    },
    "go_to_viewpoint": {
        "description": (
            "Fly to an observation POSITION vp_N proposed by propose_viewpoints: at most one "
            "leg (<= 8 m) per call, returning task_id, status (ok|blocked|no_progress), "
            "outcome, progress_m, turn_deg, remaining_m and blocked_reason. A position, not "
            "an object: to reach the object in a frame use approach_candidate; to fly back to "
            "where a frame was shot use return_to_observation."
        ),
        "side_effects": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "viewpoint_id": {"type": "string", "description": "vp_N from propose_viewpoints"},
                "task_id": {"type": "string", "description": "continue an existing task (nav_N)"},
                "timeout_s": {"type": "number"},
                "max_segments": {"type": "integer", "minimum": 1, "maximum": 64},
            },
            "required": ["viewpoint_id"],
        },
    },
    "return_to_observation": {
        "description": (
            "Fly back to the pose an observation obs_N was captured at and (with "
            "restore_yaw=true, the default) turn to the heading it recorded - the CAMERA's "
            "place, not the object it photographed. One leg (<= 8 m) per call, same progress "
            "contract as the other movement tools; a call that cannot move anything is "
            "status=no_progress / outcome=already_at_pose."
        ),
        "side_effects": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "observation_id": {
                    "type": "string",
                    "description": "obs_N of a recorded view (from observe/query_memory)",
                },
                "restore_yaw": {
                    "type": "boolean",
                    "description": "turn to the heading that observation was captured at (default true)",
                },
                "task_id": {"type": "string", "description": "continue an existing task (nav_N)"},
                "timeout_s": {"type": "number"},
                "max_segments": {"type": "integer", "minimum": 1, "maximum": 64},
            },
            "required": ["observation_id"],
        },
    },
    "verify_goal": {
        "description": (
            "Independently verify the goal from an observation (defaults to the latest one): "
            "returns satisfied and unsatisfied conditions with the evidence used."
        ),
        "side_effects": False,
        "input_schema": {
            "type": "object",
            "properties": {
                "conditions": {"type": "array", "items": {"type": "string"}},
                "candidate_id": {"type": "string"},
                "observation_id": {"type": "string"},
            },
            "required": [],
        },
    },
    FINISH_TOOL_NAME: {
        "description": (
            "End the episode with an explicit outcome and the evidence behind it: "
            "success=true only when this episode left the trail (a verify_goal that "
            "returned verified=true, or a grounded candidate plus an arrival). The "
            "answer reports verified, claim, evidence_trail and what is missing; an "
            "unbacked success claim is downgraded and never counted as success."
        ),
        "side_effects": False,
        "input_schema": FINISH_SCHEMA,
    },
    BATCH_TOOL_NAME: {
        "description": BATCH_DESCRIPTION,
        "side_effects": False,
        "input_schema": BATCH_SCHEMA,
    },
    "ground_target": {
        "description": (
            "Ground an instruction into candidate detections inside one observation: returns "
            "candidate ids, image boxes, attributes and open questions (identification kept "
            "separate from motion)."
        ),
        "side_effects": False,
        "input_schema": {
            "type": "object",
            "properties": {
                "instruction": {"type": "string"},
                "observation_id": {"type": "string", "description": "defaults to the latest observation"},
                "region": {"type": "array", "items": {"type": "number"}, "description": "optional [x0, y0, x1, y1] in normalized [0, 1000] image coordinates"},
                "max_candidates": {"type": "integer", "minimum": 1, "maximum": 8},
            },
            "required": ["instruction"],
        },
    },
    "query_memory": {
        "description": (
            "Query recorded observations and revisable hypotheses for a target, place or "
            "event: returns matching evidence with references to the original images."
        ),
        "side_effects": False,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "kind": {"type": "string", "enum": ["target", "place", "event", "route", "any"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 12},
            },
            "required": ["query"],
        },
    },
    "compare_candidates": {
        "description": (
            "Compare two candidate evidence sets: returns matching attributes, conflicts and "
            "what still needs confirmation (same-instance vs different-instance check)."
        ),
        "side_effects": False,
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                "criteria": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["candidate_ids"],
        },
    },
    "inspect_region": {
        "description": (
            "Zoom into a region of a recorded observation to read fine detail (far or small "
            "targets, discriminative attributes). Returns a crop observation id and image "
            "evidence."
        ),
        "side_effects": False,
        "input_schema": {
            "type": "object",
            "properties": {
                "observation_id": {"type": "string"},
                "region": {"type": "array", "items": {"type": "number"}, "description": "[x0, y0, x1, y1] in normalized [0, 1000] image coordinates"},
                "zoom": {"type": "number", "minimum": 1.0, "description": "resampling factor for the returned crop"},
            },
            "required": ["observation_id", "region"],
        },
    },
}

#: Candidate modules/paths searched for the tool implementations, in order; all
#: importable ones are merged, so a partially landed toolset still works.
DEFAULT_TOOL_MODULES = (
    "vlm_nav.tool_impl_perception",
    "vlm_nav.tool_impl_navigation",
    "vlm_nav.agent_tools",
    "vlm_nav.toolset",
)
_REGISTRY_BUILDERS = (
    "build_registry",
    "make_registry",
    "create_registry",
    "build_tool_registry",
    "build_env_registry",
    "build_evidence_registry",
    "build_environment_registry",
    "default_registry",
)
_REGISTRY_FILLERS = ("register_all", "register_default_tools", "register_tools", "register")
_SPEC_EXPORTS = ("TOOL_SPECS", "TOOLS", "SPECS", "DEFAULT_SPECS")
_TOOL_FACTORY_PATTERNS = ("{name}_spec", "make_{name}", "{name}_tool", "make_{name}_spec")

#: ``ctx.extra`` keys that hold observation-frame records: the runtime publishes
#: rendered frames under ``images``, the perception tools under
#: ``perception.observations``, this module's recorder under ``obs_store``.
_EXTRA_OBSERVATION_KEYS = ("obs_store", "images", "observations", "perception.observations")


# --------------------------------------------------------------------------- #
# image / json helpers
# --------------------------------------------------------------------------- #
def _is_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "shape")


def _to_hwc_uint8(payload: Any) -> np.ndarray:
    """Normalize a tensor/array frame (CHW or HWC, float [0, 1] or uint8) to RGB."""
    if _is_tensor(payload):
        payload = payload.detach().cpu().numpy()
    arr = np.asarray(payload)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"unsupported frame shape for image encoding: {np.asarray(payload).shape}")
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    if arr.dtype == np.bool_:
        return (arr.astype(np.uint8) * 255)
    arr = arr.astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size and float(finite.max()) <= 1.0 + 1e-6 and float(finite.min()) >= -1e-6:
        arr = arr * 255.0
    return np.clip(np.nan_to_num(arr), 0, 255).astype(np.uint8)


def _encode_png(rgb: np.ndarray) -> bytes:
    """Minimal stdlib PNG encoder (no Pillow dependency on the serving path)."""
    height, width = rgb.shape[:2]
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0
        raw += rgb[y].tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def _sniff_image_mime(raw: bytes) -> str:
    for magic, mime in _SIGNATURES:
        if raw.startswith(magic):
            return mime
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError(f"unsupported image bytes (magic={raw[:4]!r})")


def encode_image(payload: Any) -> tuple[bytes, str]:
    """Resolve an observation/map payload into real image bytes + mime type."""
    if isinstance(payload, (bytes, bytearray, memoryview)):
        raw = bytes(payload)
        return raw, _sniff_image_mime(raw)
    if isinstance(payload, (str, Path)):
        raw = Path(payload).read_bytes()
        return raw, _sniff_image_mime(raw)
    save = getattr(payload, "save", None)  # PIL image
    if callable(save):
        buffer = io.BytesIO()
        save(buffer, format="PNG")
        return buffer.getvalue(), IMAGE_MIME
    return _encode_png(_to_hwc_uint8(payload)), IMAGE_MIME


def _array_summary(arr: Any) -> Any:
    arr = np.asarray(arr)
    if arr.size <= 256:
        return arr.tolist()
    return f"<array shape={tuple(int(s) for s in arr.shape)} dtype={arr.dtype}>"


def _jsonable(value: Any, depth: int = 0) -> Any:
    """Make a tool result / resource payload JSON-serializable.

    Structured fields are expected to be scalars, lists and dicts; anything else
    (tensors, numpy arrays, dataclasses, paths) is summarized rather than
    dropped, so images never leak raw arrays into the model context.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value), depth + 1)
    if isinstance(value, Mapping):
        if depth >= 6:
            return str(value)
        return {str(k): _jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        if depth >= 6:
            return str(value)
        return [_jsonable(v, depth + 1) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _array_summary(value)
    if _is_tensor(value):
        return _array_summary(value.detach().cpu().numpy())
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _safe(fn: Callable[[], Any], default: Any = None) -> Any:
    try:
        return fn()
    except Exception:
        return default


def _extra(ctx: ToolContext) -> dict:
    """``ctx.extra`` repaired into a dict (a runtime may leave it None)."""
    if not isinstance(ctx.extra, dict):
        ctx.extra = {}
    return ctx.extra


def _extra_mapping_keys(extra: Mapping, keys: Sequence[str], suffixes: Sequence[str]) -> list[str]:
    """Explicit ``extra`` store keys plus namespaced ones (``<ns>.<suffix>``)."""
    found = [key for key in keys if isinstance(extra.get(key), Mapping)]
    for key, value in extra.items():
        if key in found or not isinstance(value, Mapping):
            continue
        if any(key.endswith(f".{suffix}") or key.endswith(f"_{suffix}") for suffix in suffixes):
            found.append(key)
    return found


# --------------------------------------------------------------------------- #
# observations
# --------------------------------------------------------------------------- #
@dataclass
class Observation:
    observation_id: str
    image: Any = None
    pose: dict | None = None
    meta: dict = field(default_factory=dict)


def _record_id(record: Any) -> str | None:
    for key in ("observation_id", "obs_id", "id"):
        if isinstance(record, Mapping) and key in record:
            return str(record[key])
        if hasattr(record, key):
            return str(getattr(record, key))
    return None


def _pick(mapping: Mapping, *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _obs_id_for_frame_stem(stem: str) -> str | None:
    """Map a frame file name to an observation id.

    Accepts both the runtime naming (``obs_7``) and the harness frame naming
    (``step_007``), so ``nav://obs/obs_007/image`` resolves either way.
    """
    if stem.startswith("obs_"):
        return stem
    if stem.startswith("step_"):
        digits = stem[5:]
        if digits.isdigit():
            return f"obs_{int(digits):03d}"
    return None


def _normalize_record(obs_id: str, record: Any) -> Observation | None:
    if record is None:
        return None
    if isinstance(record, Observation):
        return record
    if isinstance(record, Mapping):
        image = _pick(record, "image", "image_path", "png", "rgb", "frame", "crop", "image_bytes")
        pose = record.get("pose")
        if pose is None:
            pose = {k: record[k] for k in ("position_flu", "yaw_rad", "step") if k in record} or None
        meta = {
            k: _jsonable(v)
            for k, v in record.items()
            if k not in ("image", "image_path", "png", "rgb", "frame", "crop", "image_bytes", "pose")
        }
        return Observation(obs_id, image, _jsonable(pose) if pose else None, meta)
    image = None
    for attr in ("image", "image_path", "png", "rgb", "frame", "crop"):
        candidate = getattr(record, attr, None)
        if isinstance(candidate, str) and candidate.endswith((".png", ".jpg", ".jpeg")):
            image = candidate
            break
        if candidate is not None and not isinstance(candidate, str):
            image = candidate
            break
    if image is None and (
        _is_tensor(record)
        or isinstance(record, (np.ndarray, bytes, bytearray, memoryview, str, Path))
    ):
        image = record  # the record itself is the frame payload
    pose = None
    position = getattr(record, "position_flu", None)
    if position is not None:
        pose = {"position_flu": _jsonable(position)}
        yaw = getattr(record, "yaw_rad", None)
        if yaw is not None:
            pose["yaw_rad"] = float(yaw)
    meta = {}
    for attr in ("note", "tool", "step", "timestamp", "meta"):
        value = getattr(record, attr, None)
        if value is not None:
            meta[attr] = _jsonable(value)
    return Observation(obs_id, image, pose, meta)


class ObservationStore:
    """Read-only view over whatever the runtime uses to keep observations.

    Lookup order: ``ctx.extra["obs_store"]`` / ``ctx.extra["observations"]``,
    then the memory object (a mapping, or attributes/methods such as
    ``observations`` / ``get(obs_id)`` / ``image_for(obs_id)``), then frames on
    disk under ``ctx.extra["frames_dir"]`` (``<obs_id>.png`` or the harness
    naming ``step_<n:03d>.png``).
    """

    FRAME_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".npy")

    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx

    # -- sources ---------------------------------------------------------- #
    def _extra_mappings(self) -> list[Mapping]:
        extra = _extra(self.ctx)
        keys = _extra_mapping_keys(extra, _EXTRA_OBSERVATION_KEYS, ("observations", "frames", "images"))
        return [extra[key] for key in keys]

    def _memory_mapping(self) -> Mapping | None:
        memory = self.ctx.memory
        if isinstance(memory, Mapping):
            return memory
        for attr in ("observations", "frames", "images", "records"):
            value = getattr(memory, attr, None)
            if isinstance(value, Mapping):
                return value
        return None

    def _frames_dir(self) -> Path | None:
        path = _extra(self.ctx).get("frames_dir")
        if not path:
            return None
        path = Path(path)
        return path if path.is_dir() else None

    def _frame_path(self, obs_id: str) -> Path | None:
        directory = self._frames_dir()
        if directory is None:
            return None
        names = [obs_id]
        digits = "".join(ch for ch in obs_id if ch.isdigit())
        if digits:
            number = int(digits)
            names += [f"step_{number:03d}", f"step_{number}", f"{number:03d}", str(number)]
        for stem in names:
            for suffix in self.FRAME_SUFFIXES:
                path = directory / f"{stem}{suffix}"
                if path.is_file():
                    return path
        return None

    # -- reads ------------------------------------------------------------ #
    def ids(self, limit: int | None = None) -> list[str]:
        found: dict[str, None] = {}
        for mapping in [*self._extra_mappings(), self._memory_mapping()]:
            if mapping:
                found.update(dict.fromkeys(str(k) for k in mapping))
        ids_fn = getattr(self.ctx.memory, "ids", None) if self.ctx.memory is not None else None
        if callable(ids_fn):
            for obs_id in _safe(lambda: list(ids_fn()), []) or []:
                found.setdefault(str(obs_id), None)
        directory = self._frames_dir()
        if directory is not None:
            for path in sorted(directory.iterdir()):
                if path.suffix.lower() not in self.FRAME_SUFFIXES:
                    continue
                obs_id = _obs_id_for_frame_stem(path.stem)
                if obs_id:
                    found.setdefault(obs_id, None)
        ids = sorted(found, key=lambda oid: (int("".join(ch for ch in oid if ch.isdigit()) or -1), oid))
        return ids[-limit:] if limit else ids

    def get(self, obs_id: str) -> Observation | None:
        for mapping in self._extra_mappings():
            if obs_id in mapping:
                return _normalize_record(obs_id, mapping[obs_id])
        record = self._from_memory(obs_id)
        if record is not None:
            return _normalize_record(obs_id, record)
        path = self._frame_path(obs_id)
        if path is not None:
            payload: Any = np.load(path) if path.suffix.lower() == ".npy" else path
            return Observation(obs_id, payload, None, {"source": str(path)})
        return None

    def _from_memory(self, obs_id: str) -> Any:
        memory = self.ctx.memory
        if memory is None:
            return None
        mapping = self._memory_mapping()
        if mapping is not None and obs_id in mapping:
            return mapping[obs_id]
        for attr in ("observations", "frames", "images", "records"):
            value = getattr(memory, attr, None)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for item in value:
                    if _record_id(item) == obs_id:
                        return item
        if hasattr(memory, "_records"):
            records = getattr(memory, "_records")
            if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
                for item in records:
                    if _record_id(item) == obs_id:
                        return item
        # memory.ObservationLog and friends: frame() returns the pixels directly,
        # the metadata accessors carry the PNG path.
        for method in ("frame", "get_observation", "get", "image_path", "image_for", "image"):
            fn = getattr(memory, method, None)
            if not callable(fn):
                continue
            record = _safe(lambda: fn(obs_id))
            if record is not None:
                return record
        return None


def record_observation(
    ctx: ToolContext,
    image: Any,
    *,
    observation_id: str | None = None,
    pose: Mapping | None = None,
    **meta: Any,
) -> str:
    """Store a frame in the runtime's observation store and return its id.

    Tool handlers use this so observation images are addressable as
    ``nav://obs/<id>/image`` resources; ``ctx.next_obs_id()`` keeps ids unique
    per episode.
    """
    obs_id = observation_id or ctx.next_obs_id()
    store = _extra(ctx).setdefault("obs_store", {})
    store[obs_id] = {"image": image, "pose": dict(pose) if pose else None, **meta}
    return obs_id


# --------------------------------------------------------------------------- #
# resources
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ResourceInfo:
    uri: str
    name: str
    description: str
    mime_type: str


@dataclass(frozen=True)
class ResourceTemplateInfo:
    uri_template: str
    name: str
    description: str
    mime_type: str


@dataclass(frozen=True)
class ResourceContent:
    uri: str
    content: bytes | str
    mime_type: str


RESOURCE_TEMPLATES: tuple[ResourceTemplateInfo, ...] = (
    ResourceTemplateInfo(
        "nav://obs/{observation_id}/image",
        "observation_image",
        "recorded frame of one observation, resolved to image bytes",
        IMAGE_MIME,
    ),
    ResourceTemplateInfo(
        "nav://obs/{observation_id}/meta",
        "observation_meta",
        "pose / tool / time metadata of one observation",
        JSON_MIME,
    ),
    ResourceTemplateInfo(
        "nav://traj/{episode_id}",
        "trajectory",
        "flown trajectory of an episode",
        JSON_MIME,
    ),
    ResourceTemplateInfo(
        "nav://traj/{episode_id}/segment/{segment_index}",
        "trajectory_segment",
        f"one chunk ({TRAJECTORY_SEGMENT_STEPS} poses) of the flown trajectory",
        JSON_MIME,
    ),
)


class NavResources:
    """Resolves ``nav://`` URIs into real content for the MCP resource layer."""

    def __init__(self, session: "NavSession") -> None:
        self.session = session
        self.store = ObservationStore(session.ctx)

    # -- discovery -------------------------------------------------------- #
    def categories(self, service: str | None = None) -> tuple[str, ...]:
        if service is None:
            ordered: list[str] = []
            for names in RESOURCE_CATEGORIES.values():
                for name in names:
                    if name not in ordered:
                        ordered.append(name)
            return tuple(ordered)
        return RESOURCE_CATEGORIES[_validate_service(service)]

    def templates(self, service: str | None = None) -> list[ResourceTemplateInfo]:
        categories = set(self.categories(service))
        return [t for t in RESOURCE_TEMPLATES if self._category_of_template(t) in categories]

    @staticmethod
    def _category_of_template(template: ResourceTemplateInfo) -> str:
        return "traj" if template.uri_template.startswith("nav://traj/") else "obs"

    def list(self, service: str | None = None) -> list[ResourceInfo]:
        out: list[ResourceInfo] = []
        for category in self.categories(service):
            if category == "obs":
                out.append(ResourceInfo("nav://obs", "observations", "index of recorded observations", JSON_MIME))
                for obs_id in self.store.ids(limit=MAX_LISTED_OBSERVATIONS):
                    out.append(
                        ResourceInfo(
                            f"nav://obs/{obs_id}/image",
                            f"{obs_id} image",
                            f"recorded frame of {obs_id}",
                            IMAGE_MIME,
                        )
                    )
                    out.append(
                        ResourceInfo(
                            f"nav://obs/{obs_id}/meta",
                            f"{obs_id} meta",
                            f"pose/meta of {obs_id}",
                            JSON_MIME,
                        )
                    )
            elif category == "map":
                out.append(ResourceInfo("nav://map/bev", "local map", "top-down memory map", IMAGE_MIME))
                out.append(ResourceInfo("nav://map/state", "map state", "pose, target, distance, subgoal", JSON_MIME))
            elif category == "traj":
                episode_id = self.session.ctx.episode_id
                out.append(ResourceInfo(f"nav://traj/{episode_id}", "trajectory", "flown trajectory", JSON_MIME))
                out.append(
                    ResourceInfo(
                        f"nav://traj/{episode_id}/segment/0",
                        "trajectory segment 0",
                        f"first {TRAJECTORY_SEGMENT_STEPS} poses",
                        JSON_MIME,
                    )
                )
            elif category == "memory":
                out.append(ResourceInfo("nav://candidates", "candidates", "candidate target evidence", JSON_MIME))
            elif category == "log":
                out.append(
                    ResourceInfo(
                        f"nav://episode/{self.session.ctx.episode_id}/log",
                        "tool log",
                        "tool calls, statuses, timing and budget",
                        JSON_MIME,
                    )
                )
        return out

    # -- reads ------------------------------------------------------------ #
    def read(self, uri: str) -> ResourceContent:
        parsed = urlparse(str(uri))
        if parsed.scheme != "nav":
            raise ValueError(f"not a nav:// resource: {uri}")
        parts = [p for p in f"{parsed.netloc}/{parsed.path}".split("/") if p]
        query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}

        if parts == ["obs"]:
            return self._json_content(uri, self._observation_index())
        if len(parts) == 3 and parts[0] == "obs":
            obs_id, kind = parts[1], parts[2]
            if kind == "image":
                return self._image_content(uri, self._observation_image(obs_id))
            if kind == "meta":
                return self._json_content(uri, self._observation_meta(obs_id))
        if parts == ["map", "bev"]:
            size = int(query["size"]) if query.get("size") else None
            return self._image_content(uri, self._map_bev(size))
        if parts == ["map", "state"]:
            return self._json_content(uri, self._map_state())
        if len(parts) == 2 and parts[0] == "traj":
            return self._json_content(uri, self._trajectory(parts[1]))
        if len(parts) == 4 and parts[0] == "traj" and parts[2] == "segment":
            return self._json_content(uri, self._trajectory_segment(parts[1], parts[3]))
        if parts == ["candidates"]:
            return self._json_content(uri, self._candidates())
        if len(parts) == 3 and parts[0] == "episode" and parts[2] == "log":
            return self._json_content(uri, self._tool_log(parts[1]))
        raise ValueError(
            f"unsupported nav:// resource {uri}: expected nav://obs[/<id>[/image|meta]], "
            "nav://map/[bev|state], nav://traj/<episode_id>[/segment/<i>], nav://candidates "
            "or nav://episode/<episode_id>/log"
        )

    @staticmethod
    def _json_content(uri: str, payload: Mapping) -> ResourceContent:
        return ResourceContent(uri, json.dumps(_jsonable(payload), indent=2, sort_keys=False), JSON_MIME)

    @staticmethod
    def _image_content(uri: str, payload: tuple[bytes, str]) -> ResourceContent:
        data, mime = payload
        return ResourceContent(uri, data, mime)

    # -- payload builders -------------------------------------------------- #
    def _observation_image(self, obs_id: str) -> tuple[bytes, str]:
        obs = self.store.get(obs_id)
        if obs is None:
            raise ValueError(f"unknown observation: {obs_id}")
        if obs.image is None:
            raise ValueError(f"observation {obs_id} has no image payload")
        return encode_image(obs.image)

    def _observation_meta(self, obs_id: str) -> dict:
        obs = self.store.get(obs_id)
        if obs is None:
            raise ValueError(f"unknown observation: {obs_id}")
        return {
            "episode_id": self.session.ctx.episode_id,
            "observation_id": obs.observation_id,
            "pose": obs.pose,
            "meta": obs.meta,
        }

    def _observation_index(self) -> dict:
        ids = self.store.ids()
        entries = []
        for obs_id in ids:
            obs = self.store.get(obs_id)
            if obs is None:
                continue
            entries.append(
                {
                    "observation_id": obs_id,
                    "pose": obs.pose,
                    "meta": obs.meta,
                    "image_uri": f"nav://obs/{obs_id}/image",
                }
            )
        return {
            "episode_id": self.session.ctx.episode_id,
            "current_observation": _extra(self.session.ctx).get("current_observation"),
            "count": len(entries),
            "observations": entries,
        }

    def _map_bev(self, size: int | None) -> tuple[bytes, str]:
        world = self.session.ctx.world
        if world is None:
            raise ValueError("no world bound to this session")
        image = None
        if size:
            image = _safe(lambda: world.bev_image(size=size))
        if image is None:
            image = world.bev_image()
        return encode_image(image)

    def _map_state(self) -> dict:
        world = self.session.ctx.world
        if world is None:
            raise ValueError("no world bound to this session")
        position = _safe(lambda: [float(v) for v in np.asarray(world.position_flu).reshape(-1)[:3]])
        yaw = _safe(lambda: float(world.yaw_rad))
        target = _safe(lambda: world.target)
        state = {
            "episode_id": self.session.ctx.episode_id,
            "position_flu": position,
            "yaw_deg": None if yaw is None else math.degrees(yaw),
            "distance_to_target_m": _safe(lambda: round(float(world.distance_to_target()), 3)),
            "target_in_fov": _safe(lambda: bool(world.target_in_fov())),
            "target": None
            if target is None
            else {
                "name": str(getattr(target, "name", "")),
                "position_flu": _jsonable(getattr(target, "position_flu", None)),
                "radius_m": _safe(lambda: float(getattr(target, "radius_m")), None),
            },
            "last_subgoal": _jsonable(getattr(world, "last_subgoal", None)),
            "observation_seq": self.session.ctx.observation_seq,
            "budget": dict(self.session.ctx.budget),
        }
        return state

    def _trajectory(self, episode_id: str) -> dict:
        poses = self._trajectory_poses()
        return {
            "episode_id": episode_id,
            "step_count": len(poses),
            "segment_steps": TRAJECTORY_SEGMENT_STEPS,
            "poses": poses,
        }

    def _trajectory_segment(self, episode_id: str, segment_index: str) -> dict:
        try:
            index = int(segment_index)
        except ValueError as exc:
            raise ValueError(f"trajectory segment index must be an integer: {segment_index}") from exc
        poses = self._trajectory_poses()
        start = index * TRAJECTORY_SEGMENT_STEPS
        end = start + TRAJECTORY_SEGMENT_STEPS
        if index < 0 or start >= len(poses):
            raise ValueError(
                f"trajectory segment {index} out of range: {len(poses)} poses, "
                f"max segment {max(0, (len(poses) - 1) // TRAJECTORY_SEGMENT_STEPS)}"
            )
        return {
            "episode_id": episode_id,
            "segment_index": index,
            "start_step": start,
            "end_step": min(end, len(poses)) - 1,
            "total_steps": len(poses),
            "segment_steps": TRAJECTORY_SEGMENT_STEPS,
            "poses": poses[start:end],
        }

    def _trajectory_poses(self) -> list[dict]:
        world = self.session.ctx.world
        if world is None:
            return []
        raw = getattr(world, "trajectory", None) or []
        poses = []
        for step, pose in enumerate(raw):
            entry: dict[str, Any] = {"step": step}
            if isinstance(pose, Mapping):
                entry.update({str(k): _jsonable(v) for k, v in pose.items()})
            else:
                entry["position_flu"] = _jsonable(np.asarray(pose).reshape(-1)[:3])
            poses.append(entry)
        return poses

    def _candidates(self) -> dict:
        ctx = self.session.ctx
        extra = _extra(ctx)
        sources: list[tuple[str, Any]] = []
        for key, value in extra.items():
            if key == "candidates" or key.endswith(".candidates") or key.endswith("_candidates"):
                if isinstance(value, (Mapping, list, tuple)):
                    sources.append((key, value))
        memory_candidates = getattr(ctx.memory, "candidates", None) if ctx.memory is not None else None
        if isinstance(memory_candidates, (Mapping, list, tuple)):
            sources.append(("memory.candidates", memory_candidates))
        items: list[Any] = []
        labels: list[str] = []
        for label, source in sources:
            labels.append(label)
            if callable(source):
                source = _safe(source, [])
            values = source.values() if isinstance(source, Mapping) else source
            items.extend(_jsonable(item) for item in values)
        return {
            "episode_id": ctx.episode_id,
            "sources": labels,
            "count": len(items),
            "candidates": items,
        }

    def _tool_log(self, episode_id: str) -> dict:
        session = self.session
        return {
            "episode_id": episode_id,
            "session_episode_id": session.ctx.episode_id,
            "provider": session.provider,
            "services": list(session.services),
            "observation_seq": session.ctx.observation_seq,
            "budget": dict(session.ctx.budget),
            "tool_calls": list(session.ctx.log),
        }


# --------------------------------------------------------------------------- #
# tool providers (parallel toolset module) and stubs
# --------------------------------------------------------------------------- #
def _import_module_ref(ref: str):
    path = Path(ref)
    if path.suffix == ".py" or path.exists():
        module_name = f"vlm_nav_toolset_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load tools module from file: {ref}")
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(module_name, module)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(ref)


def _tool_module_refs(explicit: str | None) -> list[str]:
    """Module refs to try in order: explicit, env override, then the defaults."""
    refs: list[str] = []
    for ref in (explicit, os.environ.get("VLM_NAV_TOOLS_MODULE"), *DEFAULT_TOOL_MODULES):
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _invoke(fn: Callable, kwargs: Mapping[str, Any]) -> Any:
    """Call a provider callable with the subset of ``kwargs`` it accepts."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn()
    parameters = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters.values()):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return fn(**kwargs)
    call_kwargs = {name: value for name, value in kwargs.items() if name in parameters}
    required = [
        p.name
        for p in parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]
    if any(name not in call_kwargs for name in required):
        return None
    return fn(**call_kwargs)


def _as_specs(value: Any, label: str, notes: list[str]) -> dict[str, ToolSpec]:
    if value is None:
        return {}
    if isinstance(value, ToolSpec):
        return {value.name: value}
    if isinstance(value, ToolRegistry):
        return {name: value.get(name) for name in value.names()}
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if isinstance(item, ToolSpec):
                out[item.name] = item
            elif callable(item):
                spec = _invoke(item, {})
                if isinstance(spec, ToolSpec):
                    out[spec.name] = spec
            else:
                notes.append(f"{label}: ignored non-ToolSpec value for {key!r}")
        return out
    if isinstance(value, (list, tuple, set)):
        out = {}
        for item in value:
            if isinstance(item, ToolSpec):
                out[item.name] = item
            elif isinstance(item, tuple) and len(item) == 2 and all(isinstance(x, str) for x in item):
                out[item[0]] = _make_spec(item[0], item[1])
        if not out and value:
            notes.append(f"{label}: no ToolSpec found in sequence export")
        return out
    notes.append(f"{label}: ignored unsupported provider return type {type(value).__name__}")
    return {}


def _collect_specs(obj: Any, names: Sequence[str], kwargs: Mapping[str, Any]) -> tuple[dict[str, ToolSpec], list[str]]:
    """Collect ToolSpecs from a provider module/object by convention."""
    found: dict[str, ToolSpec] = {}
    notes: list[str] = []

    def merge(value: Any, label: str) -> None:
        for name, spec in _as_specs(value, label, notes).items():
            if name in names:
                found.setdefault(name, spec)

    fillers = list(_REGISTRY_FILLERS)
    for attr in sorted(dir(obj)):
        if attr.startswith("register") and attr.endswith("tools") and attr not in fillers:
            fillers.append(attr)  # e.g. register_perception_tools(registry, vlm=None)
    for attr in fillers:
        fn = getattr(obj, attr, None)
        if callable(fn):
            registry = ToolRegistry()
            merge(_invoke(fn, {"registry": registry, **kwargs}), attr)
            merge(registry, attr)
    for attr in _REGISTRY_BUILDERS:
        fn = getattr(obj, attr, None)
        if callable(fn):
            merge(_invoke(fn, kwargs), attr)
    for attr in _SPEC_EXPORTS:
        if hasattr(obj, attr):
            merge(getattr(obj, attr), attr)
    for name in names:
        if name in found:
            continue
        for pattern in _TOOL_FACTORY_PATTERNS:
            fn = getattr(obj, pattern.format(name=name), None)
            if callable(fn):
                merge(_invoke(fn, kwargs), pattern.format(name=name))
                if name in found:
                    break
    return found, notes


def _vlm_from_ctx(ctx: ToolContext) -> Any:
    """Backend to hand to a provider's ``register_...(registry, vlm=...)``."""
    extra = _extra(ctx)
    for key in ("perception.vlm", "vlm"):
        backend = extra.get(key)
        if backend is not None:
            return backend
    runtime = extra.get("runtime")
    for attribute in ("vlm", "backend", "vlm_backend"):
        backend = getattr(runtime, attribute, None)
        if backend is not None:
            return backend
    return None


def discover_specs(
    names: Sequence[str],
    ctx: ToolContext,
    *,
    provider: Any = None,
    tools_module: str | None = None,
) -> tuple[dict[str, ToolSpec], str, list[str]]:
    """Find the tool implementations for ``names``; missing ones become stubs.

    Tool implementations may be split across several modules (perception and
    navigation live in separate files), so every importable candidate is merged
    and the search stops once all requested tools are found. An explicitly
    requested module that cannot be imported raises instead of silently
    degrading to stubs.
    """
    kwargs = {
        "ctx": ctx,
        "world": ctx.world,
        "memory": ctx.memory,
        "episode_id": ctx.episode_id,
        "extra": _extra(ctx),
        "vlm": _vlm_from_ctx(ctx),
    }
    if provider is not None:
        obj = getattr(provider, "PROVIDER", provider)
        specs, notes = _collect_specs(obj, names, kwargs)
        return specs, "provider", notes

    notes: list[str] = []
    specs: dict[str, ToolSpec] = {}
    sources: list[str] = []
    for ref in _tool_module_refs(tools_module):
        if all(name in specs for name in names):
            break
        try:
            module = _import_module_ref(ref)
        except Exception as exc:  # a candidate module that crashes must not kill the server
            if ref == tools_module:
                raise
            notes.append(f"toolset module not importable: {ref} ({type(exc).__name__}: {exc})")
            continue
        obj = getattr(module, "PROVIDER", module)
        missing = [name for name in names if name not in specs]
        found, collect_notes = _collect_specs(obj, missing, kwargs)
        notes.extend(collect_notes)
        if found:
            specs.update(found)
            sources.append(ref)
    if not sources:
        notes.append("no toolset module exposed ToolSpecs; serving contract stubs")
        return {}, "stub", notes
    return specs, "+".join(sources), notes


def _stub_handler(name: str) -> Callable[[ToolContext, dict], ToolResult]:
    def handler(ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(
            STATUS_FAILED,
            {"tool": name, "reason": "tool_not_implemented", "stub": True},
        )

    return handler


def _make_spec(name: str, description: str | None = None, input_schema: dict | None = None, handler: Any = None) -> ToolSpec:
    contract = TOOL_CONTRACTS.get(name, {})
    return ToolSpec(
        name=name,
        description=description or contract.get("description", ""),
        input_schema=input_schema or contract.get("input_schema", {"type": "object", "properties": {}}),
        handler=handler or _stub_handler(name),
        side_effects=bool(contract.get("side_effects", False)),
    )


def stub_spec(name: str) -> ToolSpec:
    """Advertised contract for a tool whose implementation is not importable."""
    contract = TOOL_CONTRACTS[name]
    return _make_spec(
        name,
        description=f"{contract['description']} {STUB_MARKER}",
        input_schema=contract["input_schema"],
        handler=_stub_handler(name),
    )


# --------------------------------------------------------------------------- #
# session: one registry + one log shared by both adapters
# --------------------------------------------------------------------------- #
def _validate_service(service: str) -> str:
    if service not in SERVICES:
        raise KeyError(f"unknown service: {service} (expected one of {sorted(SERVICES)})")
    return service


def _service_tool_names(services: str | Sequence[str] | None) -> tuple[str, ...]:
    if services is None:
        return ALL_TOOL_NAMES
    if isinstance(services, str):
        services = (services,)
    ordered: list[str] = []
    for service in services:
        for name in SERVICES[_validate_service(service)][1]:
            if name not in ordered:
                ordered.append(name)
    return tuple(ordered)


class NavSession:
    """One live episode: shared registry, runtime state and resource resolver.

    ``session.call`` is the direct-call adapter; the MCP tool handler calls the
    same method, so both adapters consume the same budget and append to the same
    ``ctx.log``.
    """

    def __init__(
        self,
        ctx: ToolContext,
        registry: ToolRegistry,
        *,
        services: Sequence[str] | None = None,
        provider: str = "stub",
        notes: Sequence[str] | None = None,
        stub_tools: Iterable[str] = (),
        log_sink: Callable[[dict], None] | None = None,
    ) -> None:
        self.ctx = ctx
        self.registry = registry
        self.services = tuple(services or tuple(SERVICES))
        self.provider = provider
        self.notes = list(notes or [])
        self.stub_tools = set(stub_tools)
        self.log_sink = log_sink
        self.resources = NavResources(self)
        self._service_registries: dict[str, ToolRegistry] = {}

    # -- registry views ---------------------------------------------------- #
    def state(self) -> ToolState:
        """The episode state the tool sets are ordered by (``tools.tool_state``)."""
        return tool_state(self.ctx.extra)

    def catalog(self, service: str | None = None,
                limit: int | None = None) -> ToolCatalog:
        """The state-aware catalogue of this session (same order as the local one).

        The tools a service serves are ``SERVICES``' own list, so the catalogue is
        rendered over the *registry* and then restricted to the service's names:
        an MCP client and a local caller see one ranking, one state and one set of
        schemas (the ``ToolSpec`` objects are the same).
        """
        served = set(self.names(service))
        catalog = tool_catalog(self.registry, state=self.state(), limit=limit)
        order = tuple(name for name in catalog.order if name in served)
        return ToolCatalog(
            state=catalog.state,
            order=order,
            full=tuple(name for name in catalog.full if name in served),
            compact=tuple(name for name in catalog.compact if name in served),
            named=tuple(name for name in catalog.named if name in served),
            text=catalog.text,
            limit=catalog.limit,
        )

    def specs(self, service: str | None = None) -> list[ToolSpec]:
        if service is None:
            names: Sequence[str] = ALL_TOOL_NAMES
        else:
            names = SERVICES[_validate_service(service)][1]
        registered = set(self.registry.names())
        return [self.registry.get(name) for name in names if name in registered]

    def ordered_specs(self, service: str | None = None) -> list[ToolSpec]:
        """``specs`` in the episode state's order (the recommended tool first)."""
        catalog = self.catalog(service)
        rank = {name: index for index, name in enumerate(catalog.order)}
        specs = self.specs(service)
        return sorted(specs, key=lambda spec: (rank.get(spec.name, len(rank)), spec.name))

    def names(self, service: str | None = None) -> list[str]:
        return [spec.name for spec in self.specs(service)]

    def registry_for(self, service: str) -> ToolRegistry:
        """Subset registry for one MCP service (same ToolSpec objects)."""
        service = _validate_service(service)
        if service not in self._service_registries:
            registry = ToolRegistry()
            for spec in self.specs(service):
                registry.register(spec)
            self._service_registries[service] = registry
        return self._service_registries[service]

    # -- tool calls -------------------------------------------------------- #
    def call(self, name: str, args: Mapping | None = None) -> ToolResult:
        """Direct-call adapter (identical path to the MCP tool handler)."""
        if name not in set(self.registry.names()):
            raise KeyError(f"tool not registered in this session: {name}")
        before = len(self.ctx.log)
        result = self.registry.call(name, dict(args or {}), self.ctx)
        if self.log_sink is not None and len(self.ctx.log) > before:
            entry = dict(self.ctx.log[-1])
            entry["episode_id"] = self.ctx.episode_id
            self.log_sink(entry)
        return result

    def tool_log(self) -> list[dict]:
        return list(self.ctx.log)


_SESSIONS: dict[str, NavSession] = {}
_CURRENT_SESSION: NavSession | None = None


def register_session(session: NavSession, *, make_current: bool = True) -> NavSession:
    global _CURRENT_SESSION
    _SESSIONS[session.ctx.episode_id] = session
    if make_current or _CURRENT_SESSION is None:
        _CURRENT_SESSION = session
    return session


def get_session(episode_id: str | None = None) -> NavSession | None:
    if episode_id is None:
        return _CURRENT_SESSION
    return _SESSIONS.get(episode_id)


def current_session() -> NavSession | None:
    return _CURRENT_SESSION


def build_session(
    ctx: ToolContext,
    *,
    provider: Any = None,
    tools: Any = None,
    tools_module: str | None = None,
    services: str | Sequence[str] | None = None,
    log_sink: Callable[[dict], None] | None = None,
) -> NavSession:
    """Build (and register) the session backing both MCP services.

    Tool sources, in order of precedence: an explicit ``tools`` collection
    (``ToolSpec`` list/dict, or a ``ToolRegistry`` instance - a registry is
    adopted as-is so MCP and direct calls share the exact object), an explicit
    ``provider`` object, then ``tools_module`` plus the default toolset modules
    - all importable candidates are merged, so perception and navigation can
    live in separate files. Requested tools without an implementation become
    contract stubs.
    """
    names = _service_tool_names(services)
    notes: list[str] = []
    if isinstance(tools, ToolRegistry):
        # adopt the caller's registry object: MCP calls then run through the very
        # same registry (and any subclass override) as the direct-call adapter
        registry = tools
        source = f"registry:{type(tools).__name__}"
    else:
        if tools is not None:
            specs = _as_specs(tools, "tools", notes)
            source = "tools"
        else:
            specs, source, notes = discover_specs(names, ctx, provider=provider, tools_module=tools_module)
        registry = ToolRegistry()
        for name in names:
            spec = specs.get(name)
            if spec is not None:
                registry.register(spec)
    registered = set(registry.names())
    stubs: list[str] = []
    for name in names:
        if name not in registered:
            registry.register(stub_spec(name))
            stubs.append(name)
    # the episode context drives the catalogue order of both adapters: the MCP
    # tool list and the local ``specs_text`` read the same state
    registry.attach_state(lambda: ctx.extra, limit=DEFAULT_CATALOG_CHARS)
    service_list = (services,) if isinstance(services, str) else tuple(services or tuple(SERVICES))
    return register_session(
        NavSession(
            ctx,
            registry,
            services=service_list,
            provider="stub" if source == "stub" else source,
            notes=notes,
            stub_tools=stubs,
            log_sink=log_sink,
        )
    )


# --------------------------------------------------------------------------- #
# MCP server
# --------------------------------------------------------------------------- #
def _require_mcp() -> None:
    if mcp_types is None:  # pragma: no cover - environment dependent
        raise RuntimeError(f"the 'mcp' package is required to serve MCP: {_MCP_IMPORT_ERROR}")


def _structured_result(result: ToolResult, session: NavSession, tool: str) -> dict:
    return {
        "tool": tool,
        "episode_id": session.ctx.episode_id,
        "status": result.status,
        "data": _jsonable(result.data),
        "evidence": [e.to_dict() for e in result.evidence],
    }


def _images_from_data(data: Mapping, session: NavSession | None = None, evidence: Iterable[Any] = ()) -> list:
    """MCP image content for a tool result (at most ``MAX_RESULT_IMAGES``).

    Inline pixels first (tools that hand a crop back directly); otherwise the
    frames the result refers to *by observation id* - ``image_refs``, the
    ``observation_id`` fields and the observation evidence references - resolved
    by the runtime into real image bytes instead of leaving the model a bare
    address. A ``lookaround`` capture therefore arrives with every view of the
    sweep, which is what makes choosing a view possible at all.
    """
    inline = None
    for key in ("image_png_b64", "png_b64", "image_b64", "crop_png_b64"):
        value = data.get(key)
        if isinstance(value, str) and value:
            inline = mcp_types.ImageContent(type="image", data=value, mimeType=IMAGE_MIME)
            break
    if inline is None:
        for key in ("image_png", "image_bytes", "png"):
            value = data.get(key)
            if isinstance(value, (bytes, bytearray)):
                try:
                    raw, mime = encode_image(bytes(value))
                except ValueError:
                    continue
                inline = mcp_types.ImageContent(
                    type="image", data=base64.b64encode(raw).decode(), mimeType=mime
                )
                break
    if inline is None:
        for key in ("image", "crop", "frame"):
            value = data.get(key)
            if value is None:
                continue
            try:
                raw, mime = encode_image(value)
            except (ValueError, OSError):
                continue
            inline = mcp_types.ImageContent(
                type="image", data=base64.b64encode(raw).decode(), mimeType=mime
            )
            break
    if inline is not None:
        return [inline]

    images: list = []
    if session is None:
        return images
    for obs_id in _referenced_observation_ids(data, evidence):
        observation = session.resources.store.get(obs_id)
        if observation is None or observation.image is None:
            continue
        try:
            raw, mime = encode_image(observation.image)
        except (ValueError, OSError):
            continue
        images.append(mcp_types.ImageContent(type="image", data=base64.b64encode(raw).decode(), mimeType=mime))
    return images


def _referenced_observation_ids(data: Mapping, evidence: Iterable[Any] = ()) -> list[str]:
    """Observation ids a tool result points at, in the order it named them.

    The frames of an evidence pack are bound by ``obs_id``: the ids come from
    the result's own ``image_refs`` / ``observation_id`` fields and from its
    evidence references of the observation kind, and each one is resolved
    through the same ``nav://obs/<id>/image`` store the model addresses - so an
    attachment is always the frame the result referred to, never "the second
    array of the payload". Deduplicated and bounded by the image budget.
    """
    refs: list[str] = []
    for key in ("image_refs", "observation_ids", "last_observation_id", "current_observation", "observation_id"):
        value = data.get(key)
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            if isinstance(item, str) and item and item not in refs:
                refs.append(item)
    for ref in evidence or ():
        kind = getattr(ref, "kind", None) if not isinstance(ref, Mapping) else ref.get("kind")
        ref_id = getattr(ref, "ref", None) if not isinstance(ref, Mapping) else ref.get("ref")
        if str(kind) not in ("observation", "image"):
            continue
        text = str(ref_id or "")
        if text and text not in refs:
            refs.append(text)
    return refs[:MAX_RESULT_IMAGES]


def server_name(service: str) -> str:
    return SERVICES[_validate_service(service)][0]


def build_server(service: str, session: NavSession) -> Any:
    """MCP server exposing one service's tools plus the nav:// resource layer."""
    _require_mcp()
    service = _validate_service(service)
    tool_names = tuple(SERVICES[service][1])
    server = Server(server_name(service))

    @server.list_tools()
    async def _list_tools() -> list:
        tools = []
        # the listing follows the episode state: the tools this state calls come
        # first (the same order the local catalogue renders), so an MCP client
        # reads one ranking, not a second one
        for spec in session.ordered_specs(service):
            tools.append(
                mcp_types.Tool(
                    name=spec.name,
                    description=spec.description,
                    inputSchema=spec.input_schema,
                    annotations=mcp_types.ToolAnnotations(
                        readOnlyHint=not spec.side_effects,
                        # a movement tool reached its place once: a repeat of the
                        # same request answers already_done instead of flying again
                        idempotentHint=bool(spec.side_effects),
                        destructiveHint=False,
                    ),
                )
            )
        return tools

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict | None):
        if name not in tool_names:
            return mcp_types.CallToolResult(
                content=[
                    mcp_types.TextContent(
                        type="text",
                        text=f"unknown tool for {server_name(service)}: {name}; available={list(tool_names)}",
                    )
                ],
                isError=True,
            )
        result = await anyio.to_thread.run_sync(functools.partial(session.call, name, dict(arguments or {})))
        content = [mcp_types.TextContent(type="text", text=result.to_prompt())]
        content.extend(_images_from_data(result.data, session, result.evidence))
        return content, _structured_result(result, session, name)

    @server.list_resources()
    async def _list_resources() -> list:
        entries = await anyio.to_thread.run_sync(functools.partial(session.resources.list, service))
        return [
            mcp_types.Resource(
                uri=entry.uri,
                name=entry.name,
                description=entry.description,
                mimeType=entry.mime_type,
            )
            for entry in entries
        ]

    @server.list_resource_templates()
    async def _list_resource_templates() -> list:
        templates = await anyio.to_thread.run_sync(functools.partial(session.resources.templates, service))
        return [
            mcp_types.ResourceTemplate(
                uriTemplate=template.uri_template,
                name=template.name,
                description=template.description,
                mimeType=template.mime_type,
            )
            for template in templates
        ]

    @server.read_resource()
    async def _read_resource(uri) -> list:
        content = await anyio.to_thread.run_sync(session.resources.read, str(uri))
        return [ReadResourceContents(content=content.content, mime_type=content.mime_type)]

    return server


def _stdio_streams() -> tuple[io.TextIOWrapper, io.TextIOWrapper]:
    """Text streams over duplicated std fds.

    ``mcp.server.stdio.stdio_server`` wraps ``sys.stdout.buffer``, and tearing
    that wrapper down closes the process' own stdout. Duplicating the fds keeps
    an embedded server (``start_stdio_thread``) from breaking the host process'
    streams when the MCP session ends.
    """
    stdin = io.TextIOWrapper(io.FileIO(os.dup(0), "rb", closefd=True), encoding="utf-8", errors="replace")
    stdout = io.TextIOWrapper(io.FileIO(os.dup(1), "wb", closefd=True), encoding="utf-8")
    return stdin, stdout


async def serve_stdio(service: str, session: NavSession) -> None:
    from mcp.server.stdio import stdio_server

    server = build_server(service, session)
    stdin, stdout = _stdio_streams()
    try:
        async with stdio_server(anyio.wrap_file(stdin), anyio.wrap_file(stdout)) as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        stdin.close()
        stdout.close()


def run_stdio(service: str, session: NavSession | None = None, *, episode_id: str | None = None) -> None:
    """Serve one service over stdio (blocking)."""
    session = session or get_session(episode_id)
    if session is None:
        raise RuntimeError("no NavSession to serve: build_session(ctx) first or pass a session")
    anyio.run(serve_stdio, service, session)


def start_stdio_thread(
    service: str,
    session: NavSession,
    *,
    name: str | None = None,
    daemon: bool = True,
) -> threading.Thread:
    """Serve ``service`` over stdio in a background thread (in-process runtime)."""
    thread = threading.Thread(
        target=run_stdio,
        args=(service, session),
        name=name or f"mcp-{service}",
        daemon=daemon,
    )
    thread.start()
    return thread


# --------------------------------------------------------------------------- #
# inventory + CLI
# --------------------------------------------------------------------------- #
def service_inventory(service: str, session: NavSession) -> dict:
    """Discovery payload: tools (schemas) and resources as the server exposes them."""
    catalog = session.catalog(service)
    return {
        "service": service,
        "server": server_name(service),
        "episode_id": session.ctx.episode_id,
        "provider": session.provider,
        "stub_tools": sorted(session.stub_tools),
        "provider_notes": session.notes,
        # the episode state the tool order was rendered for, and how the tools
        # are ranked/degraded for it (the same state the local adapter reads)
        "state": {
            "name": catalog.state.name,
            "hint": catalog.state.hint,
            "candidates": list(catalog.state.candidates),
            "running_tasks": [list(item) for item in catalog.state.tasks],
            "observations": catalog.state.observations,
            "verified": catalog.state.verified,
        },
        "catalog": {
            "order": list(catalog.order),
            "full": list(catalog.full),
            "compact": list(catalog.compact),
            "named": list(catalog.named),
            "limit": catalog.limit,
            "chars": len(catalog.text),
        },
        "tools": [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
                "side_effects": spec.side_effects,
                "stub": spec.name in session.stub_tools,
            }
            for spec in session.ordered_specs(service)
        ],
        "resources": [dataclasses.asdict(entry) for entry in session.resources.list(service)],
        "resource_templates": [
            dataclasses.asdict(template) for template in session.resources.templates(service)
        ],
    }


def _load_world(ref: str | None) -> Any:
    if not ref:
        return None
    if ":" in ref:
        module_ref, _, attribute = ref.partition(":")
    else:
        module_ref, attribute = ref, "build_world"
    module = _import_module_ref(module_ref)
    factory = getattr(module, attribute, None)
    if factory is None:
        raise RuntimeError(f"{module_ref} has no callable {attribute!r} (use module:callable)")
    return factory()


def _jsonl_sink(path: str | None) -> Callable[[dict], None] | None:
    if not path:
        return None
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    handle = out.open("a", encoding="utf-8")

    def sink(entry: dict) -> None:
        handle.write(json.dumps(_jsonable(entry)) + "\n")
        handle.flush()

    return sink


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vlm_nav.mcp_server",
        description="MCP stdio servers for the NavAgent tool runtime.",
    )
    parser.add_argument("--service", choices=sorted(SERVICES), required=True)
    parser.add_argument("--episode-id", default="ep_0")
    parser.add_argument("--tools-module", default=None, help="toolset module or path providing ToolSpecs")
    parser.add_argument("--world-module", default=None, help="module:callable building the world for a standalone server")
    parser.add_argument("--frames-dir", default=None, help="directory with recorded observation frames")
    parser.add_argument("--log-file", default=None, help="append every tool call to this JSONL file")
    parser.add_argument("--list", action="store_true", dest="list_inventory", help="print the tool/resource inventory and exit")
    return parser


#: ``diffaero/__init__.py`` prints a banner to stdout at import time (and taichi
#: prints one too), which would corrupt the JSON-RPC stream of a stdio client.
#: ``main`` therefore re-executes the interpreter with fd 1 parked on stderr
#: while the package imports, then restores it before serving. Pass an explicit
#: ``argv`` (an in-process call) to skip the re-exec.
_BOOTSTRAP_SOURCE = (
    "import os, sys\n"
    "real_stdout = os.dup(1)\n"
    "os.dup2(2, 1)\n"
    "import vlm_nav.mcp_server as mcp_server\n"
    "sys.stdout.flush(); sys.stderr.flush()\n"
    "os.dup2(real_stdout, 1); os.close(real_stdout)\n"
    "raise SystemExit(mcp_server.main())\n"
)


def _bootstrap_stdout(argv: Sequence[str]) -> None:
    env = dict(os.environ)
    env["VLM_NAV_MCP_BOOTSTRAPPED"] = "1"
    os.execve(sys.executable, [sys.executable, "-c", _BOOTSTRAP_SOURCE, *argv], env)


def main(argv: Sequence[str] | None = None) -> int:
    program_call = argv is None
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if (
        program_call
        and os.environ.get("VLM_NAV_MCP_BOOTSTRAPPED") != "1"
        and os.environ.get("VLM_NAV_MCP_NO_BOOTSTRAP") != "1"
    ):
        try:
            _bootstrap_stdout(sys.argv[1:])
        except OSError as exc:  # pragma: no cover - exotic environments
            print(f"[mcp_server] stdout bootstrap skipped: {exc}", file=sys.stderr)
    ctx = ToolContext(world=_load_world(args.world_module), episode_id=args.episode_id)
    if args.frames_dir:
        ctx.extra["frames_dir"] = args.frames_dir
    session = build_session(
        ctx,
        tools_module=args.tools_module,
        services=(args.service,),
        log_sink=_jsonl_sink(args.log_file),
    )
    for note in session.notes:
        print(f"[mcp_server] {note}", file=sys.stderr)
    if session.stub_tools:
        print(f"[mcp_server] stub tools: {', '.join(sorted(session.stub_tools))}", file=sys.stderr)
    if args.list_inventory:
        json.dump(service_inventory(args.service, session), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    run_stdio(args.service, session)
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
