"""Pluggable VLM backends for the navigation harness.

- ``MockVLM``: deterministic scripted/oracle policy for harness bring-up and
  tests; needs no model weights.
- ``HFLocalVLM``: in-process Hugging Face model (e.g. local Qwen3-VL).
- ``OpenAIVLM``: any OpenAI-compatible chat-completions endpoint (hosted API
  or a local server such as ``vllm serve``).

Besides the single-shot ``decide`` (one action JSON per call), every backend
exposes ``decide_with_tools`` for the agent runtime: it asks the model for one
tool call of the registry contract in ``vlm_nav.tools`` and returns text that
``tools.parse_tool_call`` understands.

Two interchangeable paths, sharing the same tools, log hook and history:

- constrained JSON (default, works with any text-in/text-out backend): the tool
  schemas and ``TOOL_CALL_DOC`` are appended to the system prompt and the reply
  is the model's text;
- native tool calling (``OpenAIVLM``, and ``HFLocalVLM`` when its chat template
  renders a ``tools`` argument): the API/template carries the schemas and the
  returned ``tool_calls`` are converted back into the canonical
  ``{"tool": <name>, "args": {...}}`` JSON (keeping the API call id under
  ``"id"``). History round-trips through ``context["history"]``, which accepts
  OpenAI-shaped messages as well as the convenience form
  ``{"tool_call": {...}, "result": "<ToolResult text>"}``.
"""

from __future__ import annotations

import base64
from collections import OrderedDict
import hashlib
import io
import json
import math
import os
import threading
import time
from typing import Any, Callable
import urllib.parse
import urllib.request

import numpy as np
import torch
from PIL import Image

from .tools import TOOL_CALL_DOC, parse_tool_call


def frame_to_pil(frame_chw: torch.Tensor):
    arr = (frame_chw.detach().clamp(0, 1) * 255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def _frame_count(frames) -> int:
    if isinstance(frames, (list, tuple)):
        return len(frames)
    return 1


def usage_metrics(usage: Any) -> dict:
    """Normalize an OpenAI/vLLM usage block for logs and summaries.

    vLLM reports prefix-cache reuse in ``prompt_tokens_details.cached_tokens``;
    keeping that field next to prompt/completion counts is what makes a
    prefix-caching or contact-sheet change measurable after the fact.
    """
    if not isinstance(usage, dict):
        return {}
    details = usage.get("prompt_tokens_details")
    cached = None
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
    if cached is None:
        cached = usage.get("cached_tokens")
    metrics = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }
    if cached is not None:
        metrics["cached_tokens"] = cached
    return {key: value for key, value in metrics.items() if value is not None}


def render_tool_reply(calls: list[dict]) -> str:
    """Canonical JSON text for the native tool calls of one reply.

    A single call renders as ``{"tool": ..., "args": ...}`` (plus the API
    ``id`` when the backend reported one). Several parallel calls keep the
    first at the top level - so ``parse_tool_call`` still returns it - and list
    every call under ``"tool_calls"``.
    """
    if not calls:
        return ""
    head = dict(calls[0])
    if len(calls) == 1:
        return json.dumps(head, ensure_ascii=False)
    head["tool_calls"] = [dict(call) for call in calls]
    return json.dumps(head, ensure_ascii=False)


def _iter_json_objects(text: str) -> list[dict]:
    """Top-level ``{...}`` objects of a reply, left to right (best effort)."""
    objects: list[dict] = []
    index, length = 0, len(text or "")
    while index < length:
        if text[index] != "{":
            index += 1
            continue
        depth, end = 0, None
        for j in range(index, length):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is None:
            break
        try:
            data = json.loads(text[index : end + 1])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            objects.append(data)
        index = end + 1
    return objects


def _canonical_from_any(data: dict) -> dict | None:
    """Accept ``{"tool","args"}`` and the native spellings of one tool call."""
    if not isinstance(data, dict):
        return None
    function = data.get("function") if isinstance(data.get("function"), dict) else None
    name = data.get("tool") or data.get("name") or (function or {}).get("name")
    args = data.get(
        "args", data.get("arguments", data.get("parameters", data.get("input")))
    )
    if args is None and function is not None:
        args = function.get("arguments", function.get("parameters"))
    if not isinstance(name, str) or not name.strip():
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_raw": args}
    if args is None:
        args = {}
    if not isinstance(args, dict):
        args = {"value": args}
    call: dict[str, Any] = {"tool": name.strip(), "args": args}
    if isinstance(data.get("id"), str) and data["id"]:
        call["id"] = data["id"]
    return call


def _as_canonical_call(value) -> dict:
    """Coerce a canonical/OpenAI call dict (or its JSON text) to ``{tool,args}``."""
    if isinstance(value, dict):
        return _canonical_from_any(value) or {"tool": "", "args": {}}
    if isinstance(value, str):
        for data in _iter_json_objects(value):
            call = _canonical_from_any(data)
            if call:
                return call
    return {"tool": "", "args": {}}


def openai_tool_calls_to_canonical(tool_calls: list[dict] | None) -> list[dict]:
    """Convert an OpenAI ``message.tool_calls`` list to canonical calls."""
    calls: list[dict] = []
    for item in tool_calls or []:
        call = _canonical_from_any(item if isinstance(item, dict) else {})
        if call is not None and call["tool"]:
            calls.append(call)
    return calls


def extract_native_tool_calls(text: str) -> list[dict]:
    """Pull canonical calls out of a local-model reply.

    Handles the usual template spellings - ``<tool_call>{...}</tool_call>`` and
    bare JSON - with either ``name``/``arguments`` or ``tool``/``args`` keys.
    """
    if not text:
        return []
    calls: list[dict] = []
    for data in _iter_json_objects(text):
        nested = data.get("tool_calls")
        if isinstance(nested, list):
            for item in nested:
                call = _canonical_from_any(item if isinstance(item, dict) else {})
                if call and call["tool"]:
                    calls.append(call)
            continue
        call = _canonical_from_any(data)
        if call and call["tool"]:
            calls.append(call)
    return calls


def tools_from_specs_text(tools_text: str) -> list[dict]:
    """Recover ``[{name, description, parameters}]`` from ``specs_text()``.

    Registry lines look like ``- <name>: <description> args={...json...}``.
    Lines that do not follow that shape are skipped rather than guessed at.
    """
    tools: list[dict] = []
    for line in (tools_text or "").splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        name, sep, rest = line[2:].strip().partition(": ")
        name = name.strip()
        if not sep or not name:
            continue
        description, marker, schema_text = rest.rpartition(" args=")
        if not marker:
            description, schema_text = rest, ""
        parameters: dict = {"type": "object", "properties": {}}
        if schema_text:
            try:
                loaded = json.loads(schema_text.strip())
            except json.JSONDecodeError:
                loaded = None
            if isinstance(loaded, dict):
                parameters = loaded
            else:
                description = rest
        tools.append(
            {"name": name, "description": description.strip(), "parameters": parameters}
        )
    return tools


def resolve_tool_specs(tools_text: str = "", context: dict | None = None) -> list[dict]:
    """Tool specs as ``[{name, description, parameters}]``.

    ``context["tools"]`` wins when the runtime already has the structured specs
    (``ToolSpec``-shaped, OpenAI-shaped or naked ``{name, parameters}``); the
    rendered ``tools_text`` is parsed otherwise.
    """
    raw = None
    if isinstance(context, dict) and context.get("tools"):
        raw = context["tools"]
    if raw is None:
        raw = tools_from_specs_text(tools_text)
    specs: list[dict] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        function = item.get("function") if isinstance(item.get("function"), dict) else item
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        parameters = (
            function.get("parameters")
            or function.get("input_schema")
            or {"type": "object", "properties": {}}
        )
        specs.append(
            {
                "name": name.strip(),
                "description": str(function.get("description", "")),
                "parameters": parameters,
            }
        )
    return specs


def to_openai_tools(specs: list[dict] | None) -> list[dict]:
    """Wrap ``[{name, description, parameters}]`` in the OpenAI tools schema."""
    tools: list[dict] = []
    for spec in specs or []:
        if not isinstance(spec, dict):
            continue
        function = spec.get("function") if isinstance(spec.get("function"), dict) else None
        if function is not None:
            tools.append({"type": "function", "function": function})
            continue
        name = spec.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name.strip(),
                    "description": str(spec.get("description", "")),
                    "parameters": spec.get("parameters")
                    or spec.get("input_schema")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return tools


def specs_to_text(specs: list[dict] | None) -> str:
    """Same rendering as ``ToolRegistry.specs_text`` (schema text for prompts)."""
    lines = []
    for spec in specs or []:
        lines.append(
            f"- {spec['name']}: {spec.get('description', '')} "
            f"args={json.dumps(spec.get('parameters') or {'type': 'object', 'properties': {}})}"
        )
    return "\n".join(lines)


def build_tool_system_prompt(system_prompt: str, tools_text: str = "") -> str:
    """System prompt for the constrained-JSON path: policy + schemas + protocol."""
    parts: list[str] = []
    if system_prompt and system_prompt.strip():
        parts.append(system_prompt.strip())
    if tools_text and tools_text.strip():
        parts.append("Available tools:\n" + tools_text.strip())
    parts.append(TOOL_CALL_DOC)
    return "\n\n".join(parts)


def _openai_call_from_canonical(call: dict, fallback_id: str) -> dict:
    call_id = str(call.get("id") or fallback_id)
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": str(call.get("tool", "")),
            "arguments": json.dumps(call.get("args") or {}, ensure_ascii=False),
        },
    }


def history_to_messages(history: list | None) -> list[dict]:
    """Turn runtime tool history into OpenAI chat messages.

    Accepted item shapes (extra keys are ignored):

    - ``{"role": "assistant", "content": ...}`` - plain model text;
    - ``{"role": "assistant", "tool_calls": [...]}`` - verbatim API shape or
      canonical ``{tool, args}`` dicts;
    - ``{"role": "tool", "tool_call_id": ..., "content": ...}`` - tool result;
    - ``{"role": "user"|"system", "content": ...}`` - text feedback;
    - ``{"tool_call": {...}, "result": "<ToolResult text>"}`` - convenience form
      (a missing call id is filled in so the pair stays consistent).
    """
    messages: list[dict] = []
    for index, item in enumerate(history or []):
        if not isinstance(item, dict):
            continue
        if "tool_call" in item and "role" not in item:
            call = _as_canonical_call(item["tool_call"])
            call_id = str(item.get("tool_call_id") or call.get("id") or f"call_{index}")
            messages.append(
                {
                    "role": "assistant",
                    "content": item.get("content"),
                    "tool_calls": [_openai_call_from_canonical(call, call_id)],
                }
            )
            if "result" in item:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": str(item.get("result") or ""),
                    }
                )
            continue
        role = str(item.get("role") or "assistant")
        if item.get("tool_calls"):
            messages.append(
                {
                    "role": "assistant",
                    "content": item.get("content"),
                    "tool_calls": [
                        _openai_call_from_canonical(_as_canonical_call(call), f"call_{index}_{k}")
                        for k, call in enumerate(item["tool_calls"])
                    ],
                }
            )
        elif role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("tool_call_id") or f"call_{index}"),
                    "content": str(item.get("content") or ""),
                }
            )
        elif item.get("content") is not None:
            messages.append({"role": role, "content": str(item.get("content"))})
    return messages


def history_to_text(history: list | None) -> str:
    """Render tool history as prompt text for the constrained-JSON path."""
    lines: list[str] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        if "tool_call" in item and "role" not in item:
            lines.append("assistant: " + json.dumps(_as_canonical_call(item["tool_call"])))
            if "result" in item:
                lines.append("tool: " + str(item.get("result") or ""))
            continue
        role = str(item.get("role") or "assistant")
        if item.get("tool_calls"):
            for call in item["tool_calls"]:
                lines.append("assistant: " + json.dumps(_as_canonical_call(call)))
        elif role == "tool":
            call_id = item.get("tool_call_id")
            label = f"tool[{call_id}]" if call_id else "tool"
            lines.append(f"{label}: {item.get('content') or ''}")
        elif item.get("content"):
            lines.append(f"{role}: {item['content']}")
    return "\n".join(lines)


class VLMBackend:
    name = "base"
    on_log: Callable[[dict], None] | None = None

    def decide(
        self,
        frames: "torch.Tensor | list[torch.Tensor]",
        system_prompt: str,
        user_prompt: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError

    def decide_with_tools(
        self,
        frames: "torch.Tensor | list[torch.Tensor]",
        system_prompt: str,
        user_prompt: str,
        tools_text: str = "",
        context: dict[str, Any] | None = None,
    ) -> str:
        """Ask the model for one tool call (constrained-JSON path).

        The tool schemas (``tools_text``, or ``context["tools"]``) and
        ``TOOL_CALL_DOC`` are appended to the system prompt; previous tool
        turns in ``context["history"]`` are rendered as text. The model's reply
        is returned verbatim - ``tools.parse_tool_call`` extracts the call.

        Backends with native tool calling override this; ``context["on_log"]``
        and ``self.on_log`` receive one event per call either way.
        """
        tools_text = self._resolve_tools_text(tools_text, context)
        history_text = history_to_text(_context_history(context))
        user = (
            user_prompt
            if not history_text
            else f"{user_prompt}\n\nPrevious tool turns:\n{history_text}"
        )
        started = time.time()
        raw = self.decide(frames, build_tool_system_prompt(system_prompt, tools_text), user, context)
        raw = raw if isinstance(raw, str) else str(raw)
        parsed = parse_tool_call(raw)
        calls = [{"tool": parsed[0], "args": parsed[1]}] if parsed else []
        self._log_tool_call("json", raw, calls, started, context, _frame_count(frames))
        return raw

    @staticmethod
    def _resolve_tools_text(tools_text: str, context: dict | None) -> str:
        tools_text = tools_text or ""
        if not tools_text and isinstance(context, dict) and isinstance(context.get("tools_text"), str):
            tools_text = context["tools_text"]
        if not tools_text:
            tools_text = specs_to_text(resolve_tool_specs("", context))
        return tools_text

    def _log_tool_call(
        self,
        path: str,
        raw: str,
        calls: list[dict],
        started: float,
        context: dict | None,
        n_frames: int | None = None,
        extra: dict | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "backend": self.name,
            "path": path,  # "json" (constrained) | "native" (API tool calls)
            "ms": round((time.time() - started) * 1000.0, 1),
            "raw": raw,
            "tool_calls": [dict(call) for call in calls],
            "n_frames": n_frames,
        }
        if isinstance(context, dict):
            for key in ("episode_id", "step"):
                if context.get(key) is not None:
                    event[key] = context[key]
        if extra:
            event.update(extra)
        self._emit_log(event, context)

    def _emit_log(self, event: dict, context: dict | None) -> None:
        hook = context.get("on_log") if isinstance(context, dict) else None
        if hook is None:
            hook = getattr(self, "on_log", None)
        if not callable(hook):
            return
        try:
            hook(dict(event))
        except Exception:  # a broken log hook must not break the tool loop
            pass

    @staticmethod
    def _as_image_list(frames, image_size) -> list:
        if isinstance(frames, torch.Tensor):
            frames = [frames]
        images = [frame_to_pil(f) for f in frames]
        if image_size:
            images = [im.resize(image_size, Image.BILINEAR) for im in images]
        return images

    def close(self) -> None:
        pass


def _context_history(context: dict | None) -> list:
    if not isinstance(context, dict):
        return []
    history = context.get("history")
    return history if isinstance(history, list) else []


class MockVLM(VLMBackend):
    """Scripted or oracle policy.

    mode="forward": always a small forward step.
    mode="script": consume ``script`` entries (dicts) in order, then hover.
    mode="oracle": fly toward context["target_pos_flu"] using the known pose
        (validates the harness end-to-end without a model).
    mode="tools": consume ``tool_script`` entries, then finish.

    ``tool_script`` (any mode) scripts the tool loop: each entry is a tool-call
    dict (``{"tool": ..., "args": {...}}``) - arbitrary dicts are dumped as-is,
    so malformed/echo replies can be scripted too - or a raw string reply
    returned verbatim. Once the script runs out the mock replies
    ``{"tool": "finish", ...}`` so a tool loop always terminates.
    """

    name = "mock"

    def __init__(
        self,
        mode: str = "oracle",
        script: list[dict] | None = None,
        tool_script: list | None = None,
        on_log: Callable[[dict], None] | None = None,
    ):
        self.mode = mode
        self.script = list(script or [])
        self.tool_script = list(tool_script or [])
        self.on_log = on_log

    def _tool_reply(self) -> str:
        if self.tool_script:
            item = self.tool_script.pop(0)
            return item if isinstance(item, str) else json.dumps(item)
        return json.dumps({"tool": "finish", "args": {"reason": "mock tool script exhausted"}})

    def decide_with_tools(self, frames, system_prompt, user_prompt, tools_text="", context=None):
        if not self.tool_script and self.mode != "tools":
            return super().decide_with_tools(
                frames, system_prompt, user_prompt, tools_text, context
            )
        started = time.time()
        raw = self._tool_reply()
        parsed = parse_tool_call(raw)
        calls = [{"tool": parsed[0], "args": parsed[1]}] if parsed else []
        self._log_tool_call("json", raw, calls, started, context, _frame_count(frames))
        return raw

    def decide(self, frame_chw, system_prompt, user_prompt, context=None):
        # the confirmation step asks for an identity verdict, not a tool call:
        # a scripted model has no vision, so it confirms deterministically (the
        # mechanism, not the mock's eyesight, is what a scripted episode tests)
        if "identity-confirmation" in str(system_prompt or "").lower():
            # a scripted model has no eyesight: draw the box around the pixel
            # the candidate projects to on this view (metadata supplied by the
            # confirmation skill), falling back to the frame centre
            pixel = (context or {}).get("candidate_pixel") if isinstance(context, dict) else None
            if isinstance(pixel, (list, tuple)) and len(pixel) >= 2:
                u, v = float(pixel[0]), float(pixel[1])
                box = [u - 40.0, v - 40.0, u + 40.0, v + 40.0]
            else:
                box = [450, 450, 550, 550]
            return json.dumps({
                "target": True,
                "same_instance": True,
                "attributes_match": True,
                "relation_supported": True,
                "box": box,
                "confidence": 0.9,
                "reason": "mock confirmation",
            })
        if self.mode == "tools":
            return self._tool_reply()
        if self.mode == "script":
            action = self.script.pop(0) if self.script else {"action": "hover"}
            return json.dumps(action)
        if self.mode == "forward":
            return json.dumps({"action": "move", "forward_m": 0.8, "reason": "mock"})
        if self.mode == "oracle":
            ctx = context or {}
            target = np.asarray(ctx["target_pos_flu"], dtype=np.float64)
            pos = np.asarray(ctx["position_flu"], dtype=np.float64)
            yaw = float(ctx["yaw_rad"])
            rel = target - pos
            dist = float(np.linalg.norm(rel))
            if dist < float(ctx.get("done_dist", 1.5)):
                return json.dumps({"action": "done", "reason": "oracle reached"})
            desired = math.atan2(rel[1], rel[0])
            dyaw = (desired - yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(dyaw) > math.radians(20):
                return json.dumps(
                    {"action": "turn", "yaw_deg": math.degrees(dyaw), "reason": "oracle align"}
                )
            step = min(1.0, dist * 0.6)
            horiz = float(np.linalg.norm(rel[:2]))
            up = float(np.clip(rel[2], -step * 0.5, step * 0.5))
            fwd = float(np.clip(np.hypot(step, up), 0.0, 1.0))
            fwd = min(step, horiz)
            return json.dumps(
                {"action": "move", "forward_m": fwd, "up_m": up, "reason": "oracle approach"}
            )
        raise ValueError(f"unknown mock mode {self.mode!r}")


class HFLocalVLM(VLMBackend):
    """In-process Hugging Face vision-language model (e.g. local Qwen3-VL).

    Tool calling follows the chat template: when it renders a ``tools``
    argument the native path is used (schemas go through the template and the
    reply's tool call is converted to canonical JSON), otherwise the backend
    falls back to the constrained-JSON path. ``native_tools`` forces either
    behaviour instead of probing on the first tool call.
    """

    name = "hf"

    _TOOL_PROBE = {
        "type": "function",
        "function": {
            "name": "diffaero_tools_probe",
            "description": "capability probe",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        dtype: str = "bfloat16",
        max_new_tokens: int = 96,
        image_size: tuple[int, int] = (320, 180),
        num_threads: int | None = None,
        template_kwargs: dict | None = None,
        native_tools: bool | None = None,
        on_log: Callable[[dict], None] | None = None,
    ):
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

        if num_threads:
            torch.set_num_threads(int(num_threads))
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.image_size = image_size
        self.template_kwargs = dict(template_kwargs or {})
        self.native_tools = native_tools
        self.on_log = on_log
        self.last_usage: dict = {}
        torch_dtype = getattr(torch, dtype)
        config = AutoConfig.from_pretrained(model_path)
        self.is_qwen3vl = getattr(config, "model_type", "") == "qwen3_vl"
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch_dtype, device_map=self.device
        )
        self.model.eval()

    def _render(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        kwargs = dict(self.template_kwargs)
        if tools:
            kwargs["tools"] = tools
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )

    def _token_budget(self, context: dict | None) -> int:
        """Per-call generation cap: ``context["max_new_tokens"]`` may raise it.

        A self-plan turn needs room for the whole multi-stage JSON object; the
        backend default (96) cuts a four-stage plan mid-object. The runtime
        asks for the larger cap only on that turn, and a caller can never
        lower the configured default through the context.
        """
        configured = int(self.max_new_tokens)
        if not isinstance(context, dict):
            return configured
        try:
            requested = int(context.get("max_new_tokens") or 0)
        except (TypeError, ValueError):
            return configured
        return max(configured, requested)

    def _generate(self, text: str, images: list, max_new_tokens: int | None = None) -> str:
        inputs = self.processor(
            text=[text], images=images, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=(
                    self.max_new_tokens if max_new_tokens is None else max_new_tokens
                ),
                do_sample=False,
            )
        prompt_tokens = int(inputs["input_ids"].shape[1])
        new_tokens = output[0, prompt_tokens:]
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": int(new_tokens.shape[0]),
        }
        return self.processor.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

    @staticmethod
    def _messages(system_prompt: str, user_prompt: str, images: list) -> list[dict]:
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": user_prompt})
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _messages_without_system(system_prompt: str, user_prompt: str, images: list) -> list[dict]:
        # some small VLMs ship templates without a system role
        return [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": system_prompt + "\n\n" + user_prompt},
                ],
            }
        ]

    def _supports_tools_param(self) -> bool:
        """True when the chat template actually renders a ``tools`` argument.

        Jinja ignores unknown kwargs, so the probe checks that the rendered
        prompt really mentions the probe function.
        """
        if self.native_tools is not None:
            return bool(self.native_tools)
        try:
            rendered = self._render(
                [{"role": "user", "content": "ping"}], tools=[self._TOOL_PROBE]
            )
        except Exception:
            self.native_tools = False
            return False
        self.native_tools = "diffaero_tools_probe" in rendered
        return self.native_tools

    def decide(self, frames, system_prompt, user_prompt, context=None):
        started = time.time()
        image_size = (
            context.get("image_size", self.image_size)
            if isinstance(context, dict) else self.image_size
        )
        images = self._as_image_list(frames, image_size)
        try:
            text = self._render(self._messages(system_prompt, user_prompt, images))
        except Exception:
            text = self._render(self._messages_without_system(system_prompt, user_prompt, images))
        raw = self._generate(text, images, self._token_budget(context))
        if not (isinstance(context, dict) and context.get("suppress_decide_log")):
            self._log_tool_call(
                "decide", raw, [], started, context, _frame_count(frames),
                extra={
                    "usage_metrics": dict(self.last_usage),
                    "json_mode": bool(
                        isinstance(context, dict) and context.get("json_mode")
                    ),
                },
            )
        return raw

    def decide_with_tools(self, frames, system_prompt, user_prompt, tools_text="", context=None):
        specs = resolve_tool_specs(tools_text, context)
        if not specs or not self._supports_tools_param():
            fallback = dict(context or {})
            fallback["suppress_decide_log"] = True
            return super().decide_with_tools(
                frames, system_prompt, user_prompt, tools_text, fallback
            )
        images = self._as_image_list(frames, self.image_size)
        messages = [
            *self._messages(system_prompt, user_prompt, images),
            *history_to_messages(_context_history(context)),
        ]
        started = time.time()
        try:
            text = self._render(messages, tools=to_openai_tools(specs))
        except Exception:
            fallback = dict(context or {})
            fallback["suppress_decide_log"] = True
            return super().decide_with_tools(
                frames, system_prompt, user_prompt, tools_text, fallback
            )
        raw = self._generate(text, images, self._token_budget(context))
        calls = extract_native_tool_calls(raw)
        self._log_tool_call(
            "native", raw, calls, started, context, _frame_count(frames),
            extra={"usage_metrics": dict(self.last_usage)},
        )
        return render_tool_reply(calls) if calls else raw


class OpenAIVLM(VLMBackend):
    """OpenAI-compatible chat-completions backend (hosted API or local server).

    ``decide_with_tools`` uses native tool calling: the registry schemas go out
    in ``tools``, ``message.tool_calls`` come back as canonical JSON (call ids
    kept), and ``context["history"]`` - including ``{"role": "tool",
    "tool_call_id": ...}`` result messages - is sent back with the request.
    """

    name = "openai"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_new_tokens: int = 96,
        image_size: tuple[int, int] = (320, 180),
        timeout_s: float = 120.0,
        tool_choice: str = "auto",
        response_format: dict | str | None = None,
        image_format: str = "jpeg",
        jpeg_quality: int = 90,
        image_cache_size: int = 64,
        session: Any = None,
        transport: Callable[[str, dict, dict], dict] | None = None,
        on_log: Callable[[dict], None] | None = None,
    ):
        self.model = model
        base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                    or "https://api.openai.com").rstrip("/")
        # accept both ``http://host:port`` and ``http://host:port/v1``; the
        # request path below is always ``/v1/chat/completions``
        if base_url.endswith("/v1"):
            base_url = base_url[:-3].rstrip("/")
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.max_new_tokens = max_new_tokens
        self.image_size = image_size
        self.timeout_s = timeout_s
        self.tool_choice = tool_choice or "auto"
        self.response_format = (
            {"type": "json_object"} if response_format == "json" else response_format
        )
        fmt = str(image_format or "jpeg").strip().lower()
        self.image_format = fmt if fmt in ("jpeg", "jpg", "png") else "jpeg"
        self.jpeg_quality = max(1, min(100, int(jpeg_quality)))
        self.image_cache_size = max(0, int(image_cache_size))
        self.on_log = on_log
        self._transport = transport
        self._session = (
            None if transport is not None
            else (session if session is not None else self._make_session())
        )
        self._lock = threading.Lock()
        self._image_cache: OrderedDict[str, str] = OrderedDict()
        self._image_cache_hits = 0
        self._image_cache_misses = 0

    @staticmethod
    def _make_session() -> Any:
        """A keep-alive client when requests is installed, else None.

        ``requests`` lives in the ``openai`` extra; the stdlib fallback keeps
        the backend usable without it. A caller may inject ``session`` (or a
        ``transport`` callable in tests).
        """
        try:
            import requests  # noqa: PLC0415 - optional dependency

            return requests.Session()
        except Exception:  # noqa: BLE001 - optional dependency
            return None

    @property
    def _endpoint(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    def _image_data_url(self, image) -> str:
        """Encode one resized image, reusing the bytes for repeated frames."""
        key = (
            f"{image.size[0]}x{image.size[1]}:{self.image_format}:"
            f"{self.jpeg_quality}:{hashlib.sha1(image.tobytes()).hexdigest()}"
        )
        with self._lock:
            cached = self._image_cache.get(key)
            if cached is not None:
                self._image_cache.move_to_end(key)
                self._image_cache_hits += 1
                return cached
        buffer = io.BytesIO()
        if self.image_format == "png":
            image.save(buffer, format="PNG")
            mime = "image/png"
        else:
            image.convert("RGB").save(
                buffer, format="JPEG", quality=self.jpeg_quality, optimize=False
            )
            mime = "image/jpeg"
        data_url = f"data:{mime};base64," + base64.b64encode(
            buffer.getvalue()
        ).decode()
        if self.image_cache_size:
            with self._lock:
                self._image_cache[key] = data_url
                self._image_cache_misses += 1
                while len(self._image_cache) > self.image_cache_size:
                    self._image_cache.popitem(last=False)
        return data_url

    def image_cache_stats(self) -> dict:
        with self._lock:
            return {
                "size": len(self._image_cache),
                "limit": self.image_cache_size,
                "hits": self._image_cache_hits,
                "misses": self._image_cache_misses,
            }

    def _messages(
        self, frames, system_prompt, user_prompt, image_size: Any = "__default__"
    ) -> list[dict]:
        size = self.image_size if image_size == "__default__" else image_size
        images = self._as_image_list(frames, size)
        content: list[dict] = []
        for image in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._image_data_url(image)},
            })
        content.append({"type": "text", "text": user_prompt})
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    def _post(self, payload: dict) -> dict:
        url = self._endpoint
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if (
            self.response_format
            and "response_format" not in payload
            and "tools" not in payload
        ):
            payload = {**payload, "response_format": self.response_format}
        if self._transport is not None:
            return self._transport(url, headers, payload)
        if self._session is not None:
            with self._lock:
                try:
                    response = self._session.post(
                        url, json=payload, headers=headers, timeout=self.timeout_s
                    )
                except Exception:
                    # A pooled connection can go stale between calls; rebuild
                    # the session once and retry the request.
                    try:
                        self._session.close()
                    except Exception:  # noqa: BLE001
                        pass
                    self._session = self._make_session()
                    if self._session is None:
                        raise
                    response = self._session.post(
                        url, json=payload, headers=headers, timeout=self.timeout_s
                    )
            status = getattr(response, "status_code", None)
            if isinstance(status, int) and status >= 400:
                detail = str(getattr(response, "text", "") or "")[:1200]
                raise RuntimeError(
                    f"HTTP {status} from {url}: {detail or 'no error body'}"
                )
            raise_for_status = getattr(response, "raise_for_status", None)
            if callable(raise_for_status):
                raise_for_status()
            return response.json()
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers=headers
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode())

    def _token_budget(self, context: dict | None) -> int:
        """Per-call cap; ``context["max_new_tokens"]`` may only raise it."""
        configured = int(self.max_new_tokens)
        if not isinstance(context, dict):
            return configured
        try:
            requested = int(context.get("max_new_tokens") or 0)
        except (TypeError, ValueError):
            return configured
        return max(configured, requested)

    def decide(self, frames, system_prompt, user_prompt, context=None):
        started = time.time()
        image_size = (
            context.get("image_size", "__default__")
            if isinstance(context, dict) else "__default__"
        )
        payload = {
            "model": self.model,
            "max_tokens": self._token_budget(context),
            "temperature": 0.0,
            "messages": self._messages(
                frames, system_prompt, user_prompt, image_size=image_size
            ),
        }
        guided = (
            context.get("guided_json") if isinstance(context, dict) else None
        )
        if isinstance(guided, dict):
            # vLLM's guided-decoding spelling; harmless on servers that ignore
            # unknown fields, and stricter than json_object where supported.
            payload["guided_json"] = guided
        elif isinstance(context, dict) and context.get("json_mode"):
            payload["response_format"] = {"type": "json_object"}
        body = self._post(payload)
        usage = body.get("usage")
        if not (isinstance(context, dict) and context.get("suppress_decide_log")):
            self._log_tool_call(
                "decide",
                str((body["choices"][0]["message"].get("content") or "").strip()),
                [],
                started,
                context,
                _frame_count(frames),
                extra={
                    "usage": usage,
                    "usage_metrics": usage_metrics(usage),
                    "response_format": payload.get("response_format"),
                    "image_cache": self.image_cache_stats(),
                },
            )
        return (body["choices"][0]["message"].get("content") or "").strip()

    def decide_with_tools(self, frames, system_prompt, user_prompt, tools_text="", context=None):
        specs = resolve_tool_specs(tools_text, context)
        if not specs:
            fallback = dict(context or {})
            fallback["suppress_decide_log"] = True
            return super().decide_with_tools(
                frames, system_prompt, user_prompt, tools_text, fallback
            )
        messages = [
            *self._messages(frames, system_prompt, user_prompt),
            *history_to_messages(_context_history(context)),
        ]
        payload = {
            "model": self.model,
            "max_tokens": self._token_budget(context),
            "temperature": 0.0,
            "messages": messages,
            "tools": to_openai_tools(specs),
            "tool_choice": self.tool_choice,
        }
        started = time.time()
        body = self._post(payload)
        message = body["choices"][0]["message"]
        calls = openai_tool_calls_to_canonical(message.get("tool_calls"))
        raw = render_tool_reply(calls) if calls else (message.get("content") or "").strip()
        usage = body.get("usage")
        self._log_tool_call(
            "native", raw, calls, started, context, _frame_count(frames),
            extra={
                "usage": usage,
                "usage_metrics": usage_metrics(usage),
                "tool_choice": self.tool_choice,
                "image_cache": self.image_cache_stats(),
            },
        )
        return raw

    def close(self) -> None:
        session, self._session = self._session, None
        close = getattr(session, "close", None)
        if callable(close):
            close()


def make_backend(spec: str, **kwargs) -> VLMBackend:
    """spec: 'mock:oracle' | 'mock:forward' | 'mock:script' | 'mock:tools' | 'hf' | 'openai'."""
    kind, _, mode = spec.partition(":")
    if kind == "mock":
        return MockVLM(
            mode=mode or "oracle",
            script=kwargs.get("script"),
            tool_script=kwargs.get("tool_script"),
            on_log=kwargs.get("on_log"),
        )
    if kind == "hf":
        return HFLocalVLM(
            model_path=kwargs["model_path"],
            device=kwargs.get("device", "cpu"),
            max_new_tokens=kwargs.get("max_new_tokens", 96),
            image_size=tuple(kwargs.get("image_size", (320, 180))),
            num_threads=kwargs.get("num_threads"),
            template_kwargs=kwargs.get("template_kwargs"),
            native_tools=kwargs.get("native_tools"),
            on_log=kwargs.get("on_log"),
        )
    if kind == "openai":
        return OpenAIVLM(
            model=kwargs["model"],
            base_url=kwargs.get("base_url"),
            api_key=kwargs.get("api_key"),
            timeout_s=kwargs.get("timeout_s", 120.0),
            max_new_tokens=kwargs.get("max_new_tokens", 96),
            image_size=tuple(kwargs.get("image_size", (320, 180))),
            tool_choice=kwargs.get("tool_choice") or "auto",
            response_format=kwargs.get("response_format"),
            image_format=kwargs.get("image_format", "jpeg"),
            jpeg_quality=kwargs.get("jpeg_quality", 90),
            image_cache_size=kwargs.get("image_cache_size", 64),
            session=kwargs.get("session"),
            transport=kwargs.get("transport"),
            on_log=kwargs.get("on_log"),
        )
    raise ValueError(f"unknown VLM backend {spec!r}")
