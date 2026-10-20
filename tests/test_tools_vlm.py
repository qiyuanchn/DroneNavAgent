"""Tool-calling tests for the VLM backends (goal.md M1 backend upgrade).

Covers the four requirements of ``VLMBackend.decide_with_tools``:

1. constrained-JSON path: tool schemas + ``TOOL_CALL_DOC`` reach the prompt and
   the reply stays a single JSON object;
2. native tool calling: OpenAI ``tools``/``tool_calls`` round trip with call ids
   and tool-result messages (urllib mocked out), local HF native path with a
   fallback to JSON when the chat template has no tool support;
3. scripted MockVLM tool sequences (``list[dict]``);
4. one shared log hook (``on_log``) for both paths.

Everything runs on CPU with fake processors/models - no weights, no network.
"""

import http.server
import json
import threading
import unittest
from unittest import mock

import torch

from vlm_nav.tools import (
    TOOL_CALL_DOC,
    EvidenceRef,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    parse_tool_call,
    reply_looks_truncated,
)
from vlm_nav.vlm import (
    HFLocalVLM,
    MockVLM,
    OpenAIVLM,
    VLMBackend,
    build_tool_system_prompt,
    extract_native_tool_calls,
    history_to_messages,
    history_to_text,
    make_backend,
    openai_tool_calls_to_canonical,
    render_tool_reply,
    resolve_tool_specs,
    to_openai_tools,
    tools_from_specs_text,
)

FRAME = torch.zeros(3, 8, 8)

SPECS_TEXT = (
    '- finish: stop the episode args={"type": "object", "properties": {"reason": {"type": "string"}}}\n'
    '- observe: look around args={"type": "object", "properties": {"mode": {"type": "string"}}, '
    '"required": ["mode"]}'
)


def _observe(ctx, args):
    return ToolResult(
        "ok",
        {"mode": args.get("mode"), "target_visible": False},
        [EvidenceRef("observation", ctx.next_obs_id(), "scan")],
    )


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="observe",
            description="Scan the local area: returns what is visible now",
            input_schema={
                "type": "object",
                "properties": {"mode": {"type": "string"}},
                "required": ["mode"],
            },
            handler=_observe,
        )
    )
    registry.register(
        ToolSpec(
            name="finish",
            description="Stop the episode",
            input_schema={"type": "object", "properties": {"reason": {"type": "string"}}},
            handler=lambda ctx, args: ToolResult("ok", {"reason": args.get("reason", "")}),
        )
    )
    return registry


class ToolCallParsingTest(unittest.TestCase):
    """Template wrappers and generation-cap truncation of the call JSON."""

    def test_tool_call_tag_and_fence_are_stripped(self):
        raw = (
            "```json\n<tool_call>\n"
            '{"name": "update_plan", "arguments": {"steps": ['
            '{"id": "st_1", "goal": "find the door", '
            '"done_when": {"type": "sighted", "description": "the door is seen"}}]}}\n'
            "</tool_call>\n```"
        )
        call = parse_tool_call(raw)
        self.assertIsNotNone(call)
        tool, args = call
        self.assertEqual(tool, "update_plan")
        self.assertEqual(args["steps"][0]["id"], "st_1")
        self.assertFalse(reply_looks_truncated(raw))

    def test_missing_closing_delimiters_are_repaired(self):
        raw = (
            '{"tool": "update_plan", "args": {"steps": ['
            '{"id": "st_1", "goal": "find the door", '
            '"done_when": {"type": "sighted", "description": "the door is seen"}}'
        )
        self.assertTrue(reply_looks_truncated(raw))
        tool, args = parse_tool_call(raw)
        self.assertEqual(tool, "update_plan")
        self.assertEqual(len(args["steps"]), 1)

    def test_cut_inside_a_string_stays_undecodable(self):
        raw = (
            '<tool_call>\n{"tool": "update_plan", "args": {"steps": ['
            '{"id": "st_1", "goal": "find the door", '
            '"done_when": {"type": "sighted", "description": "the do'
        )
        self.assertIsNone(parse_tool_call(raw))
        self.assertTrue(reply_looks_truncated(raw))


class ToolSpecsTextTest(unittest.TestCase):
    def test_resolve_from_registry_specs_text(self):
        registry = _registry()
        specs = resolve_tool_specs(registry.specs_text())
        self.assertEqual([s["name"] for s in specs], registry.names())
        observe = next(s for s in specs if s["name"] == "observe")
        # description keeps its own ": " and the schema survives round trip
        self.assertEqual(observe["description"], "Scan the local area: returns what is visible now")
        self.assertEqual(observe["parameters"], registry.get("observe").input_schema)

    def test_resolve_prefers_structured_context_tools(self):
        specs = resolve_tool_specs(
            SPECS_TEXT,
            {"tools": [{"name": "memory", "description": "recall", "input_schema": {"type": "object"}}]},
        )
        self.assertEqual([s["name"] for s in specs], ["memory"])
        self.assertEqual(specs[0]["parameters"], {"type": "object"})

    def test_resolve_accepts_openai_shaped_tools(self):
        specs = resolve_tool_specs(
            "",
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "navigate", "description": "go", "parameters": {"type": "object"}},
                    }
                ]
            },
        )
        self.assertEqual(specs[0]["name"], "navigate")

    def test_to_openai_tools_wraps_specs(self):
        tools = to_openai_tools(resolve_tool_specs(SPECS_TEXT))
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "finish")
        self.assertEqual(tools[1]["function"]["parameters"]["required"], ["mode"])

    def test_specs_text_without_schema_keeps_tool(self):
        specs = tools_from_specs_text("- weird: no schema here")
        self.assertEqual(specs[0]["name"], "weird")
        self.assertEqual(specs[0]["parameters"], {"type": "object", "properties": {}})

    def test_non_spec_lines_are_ignored(self):
        specs = tools_from_specs_text("just some prose\n  * not a tool: skip\n- observe: x args={}")
        self.assertEqual([s["name"] for s in specs], ["observe"])
        self.assertEqual(specs[0]["description"], "x")


class MockToolScriptTest(unittest.TestCase):
    def test_scripted_sequence_then_finish(self):
        backend = MockVLM(
            mode="tools",
            tool_script=[
                {"tool": "observe", "args": {"mode": "lookaround"}},
                {"tool": "navigate", "args": {"viewpoint_id": "vp_3"}},
            ],
        )
        raw = backend.decide_with_tools(FRAME, "sys", "usr", SPECS_TEXT)
        self.assertEqual(parse_tool_call(raw), ("observe", {"mode": "lookaround"}))
        raw = backend.decide_with_tools(FRAME, "sys", "usr", SPECS_TEXT)
        self.assertEqual(parse_tool_call(raw), ("navigate", {"viewpoint_id": "vp_3"}))
        # script exhausted -> the loop still terminates
        raw = backend.decide_with_tools(FRAME, "sys", "usr", SPECS_TEXT)
        name, args = parse_tool_call(raw)
        self.assertEqual(name, "finish")
        self.assertIn("exhausted", args["reason"])
        self.assertEqual(backend.tool_script, [])

    def test_script_works_without_tools_mode(self):
        backend = MockVLM(mode="oracle", tool_script=[{"tool": "finish", "args": {}}])
        self.assertEqual(
            parse_tool_call(backend.decide_with_tools(FRAME, "sys", "usr", SPECS_TEXT)),
            ("finish", {}),
        )

    def test_raw_string_entries_pass_through(self):
        backend = MockVLM(mode="tools", tool_script=["sorry, no JSON today"])
        raw = backend.decide_with_tools(FRAME, "sys", "usr", SPECS_TEXT)
        self.assertEqual(raw, "sorry, no JSON today")
        self.assertIsNone(parse_tool_call(raw))

    def test_scripted_action_reply_is_not_a_tool_call(self):
        backend = MockVLM(mode="tools", tool_script=[{"action": "move", "forward_m": 1.0}])
        raw = backend.decide_with_tools(FRAME, "sys", "usr", SPECS_TEXT)
        self.assertEqual(json.loads(raw)["action"], "move")
        self.assertIsNone(parse_tool_call(raw))

    def test_tools_mode_through_decide(self):
        backend = MockVLM(mode="tools", tool_script=[{"tool": "observe", "args": {}}])
        self.assertEqual(parse_tool_call(backend.decide(FRAME, "sys", "usr")), ("observe", {}))

    def test_decide_stays_backward_compatible(self):
        backend = MockVLM(mode="oracle")
        context = {
            "position_flu": (0.0, 0.0, 0.0),
            "target_pos_flu": (5.0, 0.0, 0.0),
            "yaw_rad": 0.0,
            "done_dist": 1.5,
        }
        payload = json.loads(backend.decide(FRAME, "sys", "usr", context))
        self.assertEqual(payload["action"], "move")
        self.assertGreater(payload["forward_m"], 0.0)
        # without a tool script the tool path degrades to the action reply
        raw = backend.decide_with_tools(FRAME, "sys", "usr", SPECS_TEXT, context)
        self.assertEqual(json.loads(raw)["action"], "move")

    def test_make_backend_accepts_tool_script(self):
        backend = make_backend("mock:tools", tool_script=[{"tool": "finish", "args": {}}])
        self.assertIsInstance(backend, MockVLM)
        self.assertEqual(parse_tool_call(backend.decide_with_tools(FRAME, "s", "u")), ("finish", {}))


class _RecordingBackend(VLMBackend):
    name = "recording"

    def __init__(self, reply: str, on_log=None):
        self.reply = reply
        self.on_log = on_log
        self.prompts: list[tuple[str, str]] = []

    def decide(self, frames, system_prompt, user_prompt, context=None):
        self.prompts.append((system_prompt, user_prompt))
        return self.reply


class ConstrainedJsonPathTest(unittest.TestCase):
    def test_prompt_carries_tool_doc_and_schemas(self):
        backend = _RecordingBackend('{"tool": "observe", "args": {"mode": "lookaround"}}')
        raw = backend.decide_with_tools(FRAME, "POLICY", "USER", SPECS_TEXT)
        self.assertEqual(parse_tool_call(raw), ("observe", {"mode": "lookaround"}))
        system, user = backend.prompts[-1]
        self.assertTrue(system.startswith("POLICY"))
        self.assertIn(TOOL_CALL_DOC, system)
        self.assertIn("- observe: look around", system)
        self.assertIn('"required": ["mode"]', system)
        self.assertEqual(user, "USER")

    def test_tools_text_falls_back_to_context_tools(self):
        backend = _RecordingBackend('{"tool": "memory", "args": {}}')
        context = {"tools": [{"name": "memory", "description": "recall", "parameters": {"type": "object"}}]}
        backend.decide_with_tools(FRAME, "POLICY", "USER", "", context)
        system, _ = backend.prompts[-1]
        self.assertIn("- memory: recall", system)

    def test_history_is_rendered_as_text(self):
        backend = _RecordingBackend('{"tool": "finish", "args": {"reason": "done"}}')
        context = {
            "history": [
                {"tool_call": {"tool": "observe", "args": {"mode": "lookaround"}},
                 "result": "[ok] target_visible=false evidence=[observation:obs_1]"},
                {"role": "user", "content": "try the north door"},
            ]
        }
        backend.decide_with_tools(FRAME, "POLICY", "USER", SPECS_TEXT, context)
        _, user = backend.prompts[-1]
        self.assertIn("Previous tool turns:", user)
        self.assertIn('{"tool": "observe", "args": {"mode": "lookaround"}}', user)
        self.assertIn("[ok] target_visible=false", user)
        self.assertIn("user: try the north door", user)

    def test_json_path_logs_one_event(self):
        events: list[dict] = []
        backend = _RecordingBackend('{"tool": "finish", "args": {}}', on_log=events.append)
        backend.decide_with_tools(FRAME, "POLICY", "USER", SPECS_TEXT, {"episode_id": "ep_7", "step": 3})
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["backend"], "recording")
        self.assertEqual(event["path"], "json")
        self.assertEqual(event["episode_id"], "ep_7")
        self.assertEqual(event["step"], 3)
        self.assertEqual(event["tool_calls"], [{"tool": "finish", "args": {}}])
        self.assertEqual(event["n_frames"], 1)
        self.assertGreaterEqual(event["ms"], 0.0)
        self.assertIn("finish", event["raw"])

    def test_unparsable_reply_logs_empty_tool_calls(self):
        events: list[dict] = []
        backend = _RecordingBackend("I cannot do that.", on_log=events.append)
        self.assertEqual(backend.decide_with_tools(FRAME, "POLICY", "USER", SPECS_TEXT), "I cannot do that.")
        self.assertEqual(events[0]["tool_calls"], [])

    def test_build_tool_system_prompt_orders_parts(self):
        text = build_tool_system_prompt("POLICY", "- t: d args={}")
        self.assertLess(text.index("POLICY"), text.index("Available tools:"))
        self.assertLess(text.index("Available tools:"), text.index(TOOL_CALL_DOC))


class NativeConversionTest(unittest.TestCase):
    def test_extract_tool_call_block(self):
        reply = '<tool_call>\n{"name": "observe", "arguments": {"mode": "lookaround"}}\n</tool_call>'
        self.assertEqual(extract_native_tool_calls(reply), [{"tool": "observe", "args": {"mode": "lookaround"}}])

    def test_extract_bare_canonical_json(self):
        reply = 'sure\n{"tool": "navigate", "args": {"viewpoint_id": "vp_3"}}'
        self.assertEqual(extract_native_tool_calls(reply), [{"tool": "navigate", "args": {"viewpoint_id": "vp_3"}}])

    def test_extract_string_arguments(self):
        reply = '{"name": "finish", "arguments": "{\\"reason\\": \\"done\\"}"}'
        self.assertEqual(extract_native_tool_calls(reply), [{"tool": "finish", "args": {"reason": "done"}}])

    def test_extract_multiple_calls_and_nested_args(self):
        reply = (
            '{"name": "memory", "arguments": {"query": "where did I see the door", '
            '"filters": {"step": 3}}}\n{"name": "observe", "arguments": {}}'
        )
        calls = extract_native_tool_calls(reply)
        self.assertEqual([c["tool"] for c in calls], ["memory", "observe"])
        self.assertEqual(calls[0]["args"]["filters"], {"step": 3})

    def test_extract_no_calls(self):
        self.assertEqual(extract_native_tool_calls(""), [])
        self.assertEqual(extract_native_tool_calls("hovering"), [])

    def test_render_single_and_parallel_calls(self):
        single = render_tool_reply([{"tool": "observe", "args": {"mode": "lookaround"}, "id": "call_1"}])
        self.assertEqual(
            json.loads(single), {"tool": "observe", "args": {"mode": "lookaround"}, "id": "call_1"}
        )
        self.assertEqual(parse_tool_call(single), ("observe", {"mode": "lookaround"}))
        parallel = render_tool_reply(
            [
                {"tool": "observe", "args": {}, "id": "call_1"},
                {"tool": "memory", "args": {"query": "door"}, "id": "call_2"},
            ]
        )
        # the first call stays parseable, the rest ride along for a parallel runtime
        self.assertEqual(parse_tool_call(parallel), ("observe", {}))
        self.assertEqual([c["id"] for c in json.loads(parallel)["tool_calls"]], ["call_1", "call_2"])
        self.assertEqual(render_tool_reply([]), "")

    def test_openai_tool_calls_to_canonical(self):
        calls = openai_tool_calls_to_canonical(
            [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "observe", "arguments": '{"mode": "lookaround"}'},
                },
                {"type": "function", "function": {"name": "", "arguments": "{}"}},
            ]
        )
        self.assertEqual(calls, [{"tool": "observe", "args": {"mode": "lookaround"}, "id": "call_abc"}])

    def test_history_to_messages_convenience_form(self):
        messages = history_to_messages(
            [{"tool_call": {"tool": "observe", "args": {"mode": "lookaround"}}, "result": "[ok] fresh"}]
        )
        self.assertEqual(len(messages), 2)
        assistant, tool = messages
        self.assertEqual(assistant["role"], "assistant")
        call_id = assistant["tool_calls"][0]["id"]
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "observe")
        self.assertEqual(json.loads(assistant["tool_calls"][0]["function"]["arguments"]), {"mode": "lookaround"})
        self.assertEqual(tool["role"], "tool")
        self.assertEqual(tool["tool_call_id"], call_id)
        self.assertEqual(tool["content"], "[ok] fresh")

    def test_history_to_messages_accepts_canonical_tool_calls(self):
        # the runtime may remember only the canonical reply text + its id
        messages = history_to_messages(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"tool": "observe", "args": {"mode": "lookaround"}, "id": "call_9"}],
                },
                {"role": "tool", "tool_call_id": "call_9", "content": "[ok] evidence=[observation:obs_1]"},
            ]
        )
        self.assertEqual([m["role"] for m in messages], ["assistant", "tool"])
        self.assertEqual(messages[0]["tool_calls"][0]["id"], "call_9")
        self.assertEqual(
            json.loads(messages[0]["tool_calls"][0]["function"]["arguments"]), {"mode": "lookaround"}
        )
        self.assertEqual(messages[1]["tool_call_id"], "call_9")

    def test_history_to_text_shapes(self):
        text = history_to_text(
            [
                {"role": "assistant", "tool_calls": [{"tool": "observe", "args": {}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "[empty] nothing"},
                {"role": "assistant", "content": "thinking"},
            ]
        )
        lines = text.splitlines()
        self.assertEqual(lines[0], 'assistant: {"tool": "observe", "args": {}}')
        self.assertEqual(lines[1], "tool[call_1]: [empty] nothing")
        self.assertEqual(lines[2], "assistant: thinking")


class _FakeResponse:
    def __init__(self, body: dict):
        self._body = body

    def read(self) -> bytes:
        return json.dumps(self._body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _UrlopenRecorder:
    """Stand-in for urllib.request.urlopen that records the JSON payloads."""

    def __init__(self, bodies: list[dict]):
        self.bodies = list(bodies)
        self.payloads: list[dict] = []
        self.urls: list[str] = []
        self.headers: list[dict] = []

    def __call__(self, request, timeout=None):
        self.urls.append(request.full_url)
        self.headers.append(dict(request.header_items()))
        self.payloads.append(json.loads(request.data.decode()))
        body = self.bodies.pop(0) if self.bodies else {"choices": [{"message": {"content": "{}"}}]}
        return _FakeResponse(body)

    @staticmethod
    def tool_call_body(call_id: str, name: str, args: dict) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(args)},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 5},
        }


class OpenAINativeTest(unittest.TestCase):
    def _backend(self, recorder: _UrlopenRecorder, **kwargs) -> OpenAIVLM:
        kwargs.setdefault("model", "test-model")
        kwargs.setdefault("base_url", "http://localhost:9999")
        kwargs.setdefault("image_size", (8, 8))
        return OpenAIVLM(**kwargs)

    def test_decide_payload_unchanged(self):
        recorder = _UrlopenRecorder([{"choices": [{"message": {"content": "  hover  "}}]}])
        backend = self._backend(recorder)
        with mock.patch("urllib.request.urlopen", recorder):
            self.assertEqual(backend.decide(FRAME, "POLICY", "USER"), "hover")
        payload = recorder.payloads[0]
        self.assertEqual(set(payload), {"model", "max_tokens", "temperature", "messages"})
        self.assertEqual(recorder.urls[0], "http://localhost:9999/v1/chat/completions")
        self.assertTrue(recorder.headers[0]["Authorization"].endswith("Bearer "))
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "POLICY"})
        content = payload["messages"][1]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(content[-1], {"type": "text", "text": "USER"})

    def test_native_tool_call_payload_and_reply(self):
        recorder = _UrlopenRecorder(
            [_UrlopenRecorder.tool_call_body("call_abc", "observe", {"mode": "lookaround"})]
        )
        backend = self._backend(recorder)
        with mock.patch("urllib.request.urlopen", recorder):
            raw = backend.decide_with_tools(FRAME, "POLICY", "USER", SPECS_TEXT)
        self.assertEqual(
            json.loads(raw), {"tool": "observe", "args": {"mode": "lookaround"}, "id": "call_abc"}
        )
        self.assertEqual(parse_tool_call(raw), ("observe", {"mode": "lookaround"}))
        payload = recorder.payloads[0]
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(
            [tool["function"]["name"] for tool in payload["tools"]], ["finish", "observe"]
        )
        observe = payload["tools"][1]["function"]
        self.assertEqual(observe["description"], "look around")
        self.assertEqual(observe["parameters"]["required"], ["mode"])
        # native path: schemas travel in "tools", not in the system prompt
        self.assertEqual(payload["messages"][0]["content"], "POLICY")
        self.assertNotIn(TOOL_CALL_DOC, payload["messages"][0]["content"])

    def test_tool_result_round_trip(self):
        recorder = _UrlopenRecorder(
            [
                _UrlopenRecorder.tool_call_body("call_abc", "observe", {"mode": "lookaround"}),
                {"choices": [{"message": {"content": '{"tool": "finish", "args": {"reason": "seen"}}'}}]},
            ]
        )
        backend = self._backend(recorder)
        with mock.patch("urllib.request.urlopen", recorder):
            first = backend.decide_with_tools(FRAME, "POLICY", "USER", SPECS_TEXT)
            call_id = json.loads(first)["id"]
            history = [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "observe", "arguments": '{"mode": "lookaround"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": call_id, "content": "[ok] evidence=[observation:obs_1]"},
            ]
            second = backend.decide_with_tools(
                FRAME, "POLICY", "USER", SPECS_TEXT, {"history": history}
            )
        self.assertEqual(parse_tool_call(second), ("finish", {"reason": "seen"}))
        messages = recorder.payloads[1]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "tool"])
        self.assertEqual(messages[-1]["tool_call_id"], "call_abc")
        self.assertEqual(messages[-2]["tool_calls"][0]["id"], "call_abc")
        self.assertEqual(
            messages[-2]["tool_calls"][0]["function"],
            {"name": "observe", "arguments": '{"mode": "lookaround"}'},
        )

    def test_convenience_history_generates_matching_ids(self):
        recorder = _UrlopenRecorder([{"choices": [{"message": {"content": "ok"}}]}])
        backend = self._backend(recorder)
        history = [
            {"tool_call": {"tool": "observe", "args": {"mode": "lookaround"}}, "result": "[ok] fresh"},
            {"tool_call": {"tool": "memory", "args": {"query": "door"}}, "result": "[empty] no hits"},
        ]
        with mock.patch("urllib.request.urlopen", recorder):
            self.assertEqual(backend.decide_with_tools(FRAME, "POLICY", "USER", SPECS_TEXT, {"history": history}), "ok")
        messages = recorder.payloads[0]["messages"]
        assistant_ids = [m["tool_calls"][0]["id"] for m in messages if m["role"] == "assistant"]
        tool_ids = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
        self.assertEqual(assistant_ids, tool_ids)
        self.assertEqual(len(assistant_ids), 2)
        self.assertEqual(len(set(assistant_ids)), 2)

    def test_content_only_reply_passes_through(self):
        events: list[dict] = []
        recorder = _UrlopenRecorder([{"choices": [{"message": {"content": " no tool needed "}}]}])
        backend = self._backend(recorder, on_log=events.append)
        with mock.patch("urllib.request.urlopen", recorder):
            raw = backend.decide_with_tools(FRAME, "POLICY", "USER", SPECS_TEXT, {"episode_id": "ep_2"})
        self.assertEqual(raw, "no tool needed")
        self.assertEqual(events[0]["path"], "native")
        self.assertEqual(events[0]["tool_calls"], [])
        self.assertEqual(events[0]["episode_id"], "ep_2")

    def test_native_log_reports_calls_and_usage(self):
        events: list[dict] = []
        recorder = _UrlopenRecorder(
            [_UrlopenRecorder.tool_call_body("call_abc", "observe", {"mode": "lookaround"})]
        )
        backend = self._backend(recorder, on_log=events.append)
        with mock.patch("urllib.request.urlopen", recorder):
            backend.decide_with_tools(FRAME, "POLICY", "USER", SPECS_TEXT)
        self.assertEqual(
            events[0]["tool_calls"], [{"tool": "observe", "args": {"mode": "lookaround"}, "id": "call_abc"}]
        )
        self.assertEqual(events[0]["usage"], {"prompt_tokens": 11, "completion_tokens": 5})

    def test_falls_back_to_json_without_tools(self):
        recorder = _UrlopenRecorder([{"choices": [{"message": {"content": '{"tool": "finish", "args": {}}'}}]}])
        backend = self._backend(recorder)
        with mock.patch("urllib.request.urlopen", recorder):
            raw = backend.decide_with_tools(FRAME, "POLICY", "USER")
        self.assertEqual(parse_tool_call(raw), ("finish", {}))
        payload = recorder.payloads[0]
        self.assertNotIn("tools", payload)
        self.assertIn(TOOL_CALL_DOC, payload["messages"][0]["content"])

    def test_make_backend_passes_tool_choice(self):
        backend = make_backend(
            "openai", model="m", base_url="http://x", image_size=(8, 8), tool_choice="required"
        )
        self.assertEqual(backend.tool_choice, "required")


class _ChatHandler(http.server.BaseHTTPRequestHandler):
    """Tiny chat-completions endpoint recording what the client sent."""

    replies: list[dict] = []
    captured: list[dict] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length).decode())
        type(self).captured.append(
            {
                "path": self.path,
                "payload": payload,
                "content_type": self.headers.get("Content-Type"),
                "authorization": self.headers.get("Authorization"),
            }
        )
        reply = (
            type(self).replies.pop(0)
            if type(self).replies
            else {"choices": [{"message": {"content": "{}"}}]}
        )
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # keep the pytest output clean
        pass


class OpenAILoopbackTest(unittest.TestCase):
    """The payload/response assertions again, but over real HTTP + urllib."""

    def setUp(self):
        self.handler = type("_Handler", (_ChatHandler,), {"replies": [], "captured": []})
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), self.handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def test_real_http_tool_call_round_trip(self):
        self.handler.replies.append(
            _UrlopenRecorder.tool_call_body("call_http", "observe", {"mode": "lookaround"})
        )
        backend = OpenAIVLM(
            model="test-model",
            base_url=f"http://127.0.0.1:{self.server.server_port}",
            api_key="k",
            image_size=(8, 8),
        )
        raw = backend.decide_with_tools(FRAME, "POLICY", "USER", SPECS_TEXT)
        self.assertEqual(parse_tool_call(raw), ("observe", {"mode": "lookaround"}))
        self.assertEqual(json.loads(raw)["id"], "call_http")
        captured = self.handler.captured[0]
        self.assertEqual(captured["path"], "/v1/chat/completions")
        self.assertEqual(captured["authorization"], "Bearer k")
        self.assertEqual(captured["content_type"], "application/json")
        self.assertEqual(
            [t["function"]["name"] for t in captured["payload"]["tools"]], ["finish", "observe"]
        )


class _FakeInputs(dict):
    def to(self, device):
        return self


class _FakeProcessor:
    """Minimal AutoProcessor stand-in: template + tokenizer contract."""

    def __init__(self, supports_tools: bool = True, reply: str = "", prompt_tokens: int = 4):
        self.supports_tools = supports_tools
        self.reply = reply
        self.prompt_tokens = prompt_tokens
        self.renders: list[dict] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, tools=None, **kwargs):
        self.renders.append({"messages": messages, "tools": tools})
        if tools and not self.supports_tools:
            raise ValueError("this chat template does not support tools")
        text = "PROMPT"
        if tools:
            names = ",".join(tool["function"]["name"] for tool in tools)
            text += f"\n<tools>{names}</tools>"
        return text

    def __call__(self, text=None, images=None, return_tensors="pt"):
        self.last_images = images
        self.last_text = text[0] if isinstance(text, list) else text
        return _FakeInputs({"input_ids": torch.zeros(1, self.prompt_tokens, dtype=torch.long)})

    def decode(self, tokens, skip_special_tokens=True):
        return self.reply


class _FakeModel:
    def __init__(self, new_tokens: int = 6):
        self.new_tokens = new_tokens
        self.calls: list[dict] = []

    def generate(self, **inputs):
        self.calls.append(inputs)
        input_ids = inputs["input_ids"]
        total = input_ids.shape[1] + self.new_tokens
        return torch.zeros(1, total, dtype=torch.long)


def _hf_backend(processor: _FakeProcessor, native_tools=None, on_log=None) -> HFLocalVLM:
    backend = object.__new__(HFLocalVLM)  # no weights: fakes injected directly
    backend.device = torch.device("cpu")
    backend.max_new_tokens = 32
    backend.image_size = (8, 8)
    backend.template_kwargs = {}
    backend.native_tools = native_tools
    backend.on_log = on_log
    backend.processor = processor
    backend.model = _FakeModel()
    return backend


class HFLocalNativeTest(unittest.TestCase):
    def test_native_path_when_template_supports_tools(self):
        processor = _FakeProcessor(
            reply='<tool_call>\n{"name": "observe", "arguments": {"mode": "lookaround"}}\n</tool_call>'
        )
        backend = _hf_backend(processor)
        raw = backend.decide_with_tools(FRAME, "POLICY", "USER", SPECS_TEXT)
        self.assertEqual(parse_tool_call(raw), ("observe", {"mode": "lookaround"}))
        self.assertTrue(backend.native_tools)
        probe, decision = processor.renders[0], processor.renders[-1]
        self.assertEqual(probe["tools"][0]["function"]["name"], "diffaero_tools_probe")
        self.assertEqual(
            [tool["function"]["name"] for tool in decision["tools"]], ["finish", "observe"]
        )
        self.assertIn("<tools>finish,observe</tools>", processor.last_text)
        # native path keeps the policy prompt free of the JSON protocol text
        self.assertEqual(decision["messages"][0]["content"], "POLICY")
        self.assertNotIn(TOOL_CALL_DOC, decision["messages"][0]["content"])

    def test_context_raises_the_generation_cap_for_one_call_only(self):
        processor = _FakeProcessor(reply='{"tool": "observe", "args": {"mode": "forward"}}')
        backend = _hf_backend(processor)
        backend.decide_with_tools(FRAME, "POLICY", "USER", SPECS_TEXT)
        self.assertEqual(32, backend.model.calls[-1]["max_new_tokens"])
        backend.decide_with_tools(
            FRAME, "POLICY", "USER", SPECS_TEXT, {"max_new_tokens": 512}
        )
        self.assertEqual(512, backend.model.calls[-1]["max_new_tokens"])
        # the context can raise the configured cap, never lower it
        backend.decide_with_tools(
            FRAME, "POLICY", "USER", SPECS_TEXT, {"max_new_tokens": 8}
        )
        self.assertEqual(32, backend.model.calls[-1]["max_new_tokens"])

    def test_native_path_carries_tool_history(self):
        processor = _FakeProcessor(reply='{"name": "finish", "arguments": {"reason": "seen"}}')
        backend = _hf_backend(processor)
        history = [
            {"tool_call": {"tool": "observe", "args": {"mode": "lookaround"}}, "result": "[ok] obs_1"},
        ]
        backend.decide_with_tools(FRAME, "POLICY", "USER", SPECS_TEXT, {"history": history})
        messages = processor.renders[-1]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "tool"])
        self.assertEqual(messages[-1]["content"], "[ok] obs_1")

    def test_fallback_to_json_when_template_rejects_tools(self):
        processor = _FakeProcessor(supports_tools=False, reply='{"tool": "finish", "args": {"reason": "x"}}')
        events: list[dict] = []
        backend = _hf_backend(processor, on_log=events.append)
        raw = backend.decide_with_tools(FRAME, "POLICY", "USER", SPECS_TEXT)
        self.assertEqual(parse_tool_call(raw), ("finish", {"reason": "x"}))
        self.assertFalse(backend.native_tools)
        decision = processor.renders[-1]
        self.assertIsNone(decision["tools"])
        system = decision["messages"][0]["content"]
        self.assertIn(TOOL_CALL_DOC, system)
        self.assertIn("- observe: look around", system)
        self.assertEqual(events[0]["path"], "json")

    def test_native_path_reported_in_log(self):
        processor = _FakeProcessor(reply='{"name": "observe", "arguments": {}}')
        events: list[dict] = []
        backend = _hf_backend(processor, on_log=events.append)
        backend.decide_with_tools(FRAME, "POLICY", "USER", SPECS_TEXT, {"episode_id": "ep_9"})
        self.assertEqual(events[0]["path"], "native")
        self.assertEqual(events[0]["backend"], "hf")
        self.assertEqual(events[0]["episode_id"], "ep_9")
        self.assertEqual(events[0]["tool_calls"], [{"tool": "observe", "args": {}}])

    def test_native_tools_false_forces_json_path(self):
        processor = _FakeProcessor(supports_tools=True, reply='{"tool": "finish", "args": {}}')
        backend = _hf_backend(processor, native_tools=False)
        raw = backend.decide_with_tools(FRAME, "POLICY", "USER", SPECS_TEXT)
        self.assertEqual(parse_tool_call(raw), ("finish", {}))
        # no probe render, no native render: straight to the JSON path
        self.assertEqual(len(processor.renders), 1)
        self.assertIsNone(processor.renders[0]["tools"])
        self.assertIn(TOOL_CALL_DOC, processor.renders[0]["messages"][0]["content"])

    def test_decide_still_plain_generation(self):
        processor = _FakeProcessor(reply='{"action": "hover"}')
        backend = _hf_backend(processor)
        self.assertEqual(backend.decide(FRAME, "POLICY", "USER"), '{"action": "hover"}')
        messages = processor.renders[-1]["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": "POLICY"})
        self.assertEqual(messages[1]["content"][-1], {"type": "text", "text": "USER"})
        self.assertEqual(processor.renders[-1]["tools"], None)


def _raising_hook(event):
    raise RuntimeError("boom")


class LogHookTest(unittest.TestCase):
    def test_context_hook_wins_over_instance_hook(self):
        seen: list[dict] = []
        backend = MockVLM(
            mode="tools",
            tool_script=[{"tool": "finish", "args": {}}],
            on_log=seen.append,
        )
        other: list[dict] = []
        raw = backend.decide_with_tools(
            FRAME, "sys", "usr", SPECS_TEXT, {"on_log": other.append, "episode_id": "ep_1"}
        )
        self.assertEqual(parse_tool_call(raw), ("finish", {}))
        self.assertEqual(other[0]["episode_id"], "ep_1")
        self.assertEqual(seen, [])  # the per-call hook replaces the instance hook

    def test_raising_hook_does_not_propagate(self):
        backend = MockVLM(
            mode="tools", tool_script=[{"tool": "finish", "args": {}}], on_log=_raising_hook
        )
        raw = backend.decide_with_tools(FRAME, "sys", "usr", SPECS_TEXT)
        self.assertEqual(parse_tool_call(raw), ("finish", {}))


class RegistryToolLoopTest(unittest.TestCase):
    """The M1 chain: model replies a tool call -> registry executes -> result
    goes back into the next prompt as structured evidence."""

    def test_mock_driven_tool_loop(self):
        registry = _registry()
        specs_text = registry.specs_text()
        backend = MockVLM(
            mode="tools",
            tool_script=[
                {"tool": "observe", "args": {"mode": "lookaround"}},
                {"tool": "finish", "args": {"reason": "target verified"}},
            ],
        )
        context = ToolContext(world=None, episode_id="ep_1", budget={"tool_calls_left": 5})
        history: list[dict] = []
        for _ in range(3):
            raw = backend.decide_with_tools(
                FRAME, "POLICY", "USER", specs_text,
                {"history": history, "episode_id": context.episode_id},
            )
            call = parse_tool_call(raw)
            self.assertIsNotNone(call)
            name, args = call
            result = registry.call(name, args, context)
            history.append({"tool_call": {"tool": name, "args": args}, "result": result.to_prompt()})
            if name == "finish":
                break
        self.assertEqual([entry["tool"] for entry in context.log], ["observe", "finish"])
        self.assertIn("evidence=[observation:obs_1]", history[0]["result"])
        self.assertEqual(context.budget["tool_calls_left"], 3)


if __name__ == "__main__":
    unittest.main()
