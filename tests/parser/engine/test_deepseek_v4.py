# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DeepSeek V4-specific parser engine semantics."""

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.replay_harness import (
    MockTokenizer,
    _test_request,
    collect_output,
    replay_streaming,
)
from tests.parser.engine.streaming_helpers import (
    collect_content,
    collect_function_name,
    collect_tool_arguments,
    simulate_reasoning_streaming,
    simulate_tool_streaming,
)
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.deepseek_v4 import (
    DSML_INVOKE_END,
    DSML_INVOKE_NAME_END,
    DSML_INVOKE_PREFIX,
    DSML_THINK_END,
    DSML_THINK_START,
    DSML_TOOL_END,
    DSML_TOOL_START,
    INVALID_DSML_TOOL_NAME,
    DeepSeekV4Parser,
    _dsml_arg_converter,
    _unwrap_wrapper_args,
    deepseek_v4_config,
)
from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.registered_adapters import (
    DeepSeekV4ParserReasoningAdapter,
    DeepSeekV4ParserToolAdapter,
)

_THINK_START_ID = 50
_THINK_END_ID = 51

_PARAM_OPEN = '｜DSML｜parameter name="{name}" string="{is_str}">'
_PARAM_CLOSE = "</｜DSML｜parameter>"
_GET_WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "units": {"type": "string"},
                },
            },
        },
    }
]


def _param(name: str, is_str: str, value: str) -> str:
    return f"<{_PARAM_OPEN.format(name=name, is_str=is_str)}{value}{_PARAM_CLOSE}"


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(
        {
            DSML_THINK_START: _THINK_START_ID,
            DSML_THINK_END: _THINK_END_ID,
        }
    )


# ── Arg converter unit tests ─────────────────────────────────────────


class TestArgConverter:
    def _raw(self, *params: tuple[str, str, str]) -> str:
        lines = [_param(n, s, v) for n, s, v in params]
        return "\n" + "\n".join(lines) + "\n"

    def test_string_param(self):
        raw = self._raw(("city", "true", "杭州"))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result == {"city": "杭州"}

    def test_string_with_spaces_and_quotes(self):
        raw = self._raw(("msg", "true", 'He said "hello world"'))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["msg"] == 'He said "hello world"'

    def test_integer_param(self):
        raw = self._raw(("count", "false", "42"))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["count"] == 42
        assert isinstance(result["count"], int)

    def test_float_param(self):
        raw = self._raw(("ratio", "false", "3.14"))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert abs(result["ratio"] - 3.14) < 1e-9

    def test_bool_param(self):
        raw = self._raw(("flag", "false", "true"))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["flag"] is True

    def test_array_param(self):
        raw = self._raw(("items", "false", '["a", "b", "c"]'))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["items"] == ["a", "b", "c"]

    def test_object_param(self):
        raw = self._raw(("opts", "false", '{"key": "val"}'))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["opts"] == {"key": "val"}

    def test_mixed_types(self):
        raw = self._raw(
            ("location", "true", "Tokyo"),
            ("limit", "false", "10"),
            ("active", "false", "false"),
        )
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result == {"location": "Tokyo", "limit": 10, "active": False}

    def test_empty_args(self):
        result = json.loads(_dsml_arg_converter("", partial=False))
        assert result == {}

    def test_invalid_json_fallback(self):
        raw = self._raw(("data", "false", "[broken"))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["data"] == "[broken"

    def test_chinese_chars_preserved_in_json(self):
        raw = self._raw(("query", "true", "你好世界"))
        raw_json = _dsml_arg_converter(raw, partial=False)
        assert "你好世界" in raw_json
        result = json.loads(raw_json)
        assert result["query"] == "你好世界"

    def test_partial_complete_plus_in_progress(self):
        raw = self._raw(("city", "true", "Tokyo"))
        raw += f"<{_PARAM_OPEN.format(name='unit', is_str='true')}celsi"
        result = json.loads(_dsml_arg_converter(raw, partial=True))
        assert result["city"] == "Tokyo"
        assert result["unit"] == "celsi"

    def test_partial_no_in_progress(self):
        raw = self._raw(("city", "true", "Tokyo"))
        result = json.loads(_dsml_arg_converter(raw, partial=True))
        assert result == {"city": "Tokyo"}

    def test_partial_value_with_angle_bracket(self):
        raw = f"<{_PARAM_OPEN.format(name='code', is_str='true')}a<b"
        result = json.loads(_dsml_arg_converter(raw, partial=True))
        assert result == {"code": "a<b"}

    def test_partial_value_with_angle_bracket_and_complete_param(self):
        raw = self._raw(("city", "true", "Tokyo"))
        raw += f"<{_PARAM_OPEN.format(name='expr', is_str='true')}x<5"
        result = json.loads(_dsml_arg_converter(raw, partial=True))
        assert result["city"] == "Tokyo"
        assert result["expr"] == "x<5"

    def test_null_string_false(self):
        raw = self._raw(("val", "false", "null"))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["val"] is None

    def test_string_true_not_json_parsed(self):
        raw = self._raw(("n", "true", "42"))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["n"] == "42"
        assert isinstance(result["n"], str)


# ── Bare </think> absorption and duplicate <think> absorption ─────────


class TestThinkTagAbsorption:
    def test_bare_think_end_not_leaked(self, mock_tokenizer):
        parser = DeepSeekV4Parser(mock_tokenizer)
        chunks = ["</think>", "Here is the direct answer."]
        reasoning, content = simulate_reasoning_streaming(parser, chunks)
        assert reasoning == ""
        assert "</think>" not in content
        assert "Here is the direct answer" in content

    def test_duplicate_think_start_absorbed(self, mock_tokenizer):
        parser = DeepSeekV4Parser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        chunks = [
            "<think>\n",
            "Some reasoning.\n",
            "</think>\n",
            "Answer.",
        ]
        reasoning, content = simulate_reasoning_streaming(parser, chunks)
        assert "Some reasoning" in reasoning
        assert "Answer" in content


# ── Missing </｜DSML｜invoke> before </｜DSML｜tool_calls> ────────────


class TestMissingInvokeEnd:
    def test_non_streaming(self, mock_tokenizer, mock_request):
        parser = DeepSeekV4Parser(mock_tokenizer)
        text = (
            f"{DSML_TOOL_START}"
            f"{DSML_INVOKE_PREFIX}get_weather{DSML_INVOKE_NAME_END}\n"
            f"{_param('location', 'true', 'NYC')}\n"
            f"{DSML_TOOL_END}"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"location": "NYC"}

    def test_streaming_with_trailing_content(self, mock_tokenizer, mock_request):
        parser = DeepSeekV4Parser(mock_tokenizer)
        chunks = [
            DSML_TOOL_START,
            f"{DSML_INVOKE_PREFIX}get_weather{DSML_INVOKE_NAME_END}\n"
            f"{_param('location', 'true', 'NYC')}\n",
            DSML_TOOL_END,
            "Done.",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        assert collect_function_name(results) == "get_weather"
        args = json.loads(collect_tool_arguments(results))
        assert args == {"location": "NYC"}
        assert "Done." in collect_content(results)

    def test_unknown_tool_is_recoverable_invalid_tool_call(
        self, mock_tokenizer, mock_request
    ):
        tools = [_make_tool("get_weather", {"location": {"type": "string"}})]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools
        text = _tool_calls(
            _invoke("missing_tool", ("location", "true", "NYC")),
        )

        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        tool_call = result.tool_calls[0]
        assert tool_call.function.name == "__invalid_dsml_tool_call__"
        assert json.loads(tool_call.function.arguments) == {
            "error": "The model attempted an unknown tool; retry the intended call.",
            "tool_name": "missing_tool",
        }

    def test_valid_name_truncated_at_eos_is_invalid(self, mock_tokenizer, mock_request):
        tools = [_make_tool("get_weather", {"location": {"type": "string"}})]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools
        text = DSML_TOOL_START + _invoke(
            "get_weather", ("location", "true", "NYC")
        ).replace(DSML_INVOKE_END, "")

        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        tool_call = result.tool_calls[0]
        assert tool_call.function.name not in {tool.function.name for tool in tools}
        assert json.loads(tool_call.function.arguments)["tool_name"] == "get_weather"

    def test_empty_name_truncated_at_eos_is_invalid(self, mock_tokenizer, mock_request):
        tools = [_make_tool("get_weather", {"location": {"type": "string"}})]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools
        text = DSML_TOOL_START + DSML_INVOKE_PREFIX

        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        tool_call = result.tool_calls[0]
        assert tool_call.function.name not in {tool.function.name for tool in tools}
        assert "tool_name" not in json.loads(tool_call.function.arguments)

    def test_truncated_tool_prefix_is_recoverable_invalid_tool_call(
        self, mock_tokenizer, mock_request
    ):
        tools = [_make_tool("get_weather", {"location": {"type": "string"}})]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools

        result = parser.extract_tool_calls(DSML_TOOL_START, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        tool_call = result.tool_calls[0]
        assert tool_call.function.name == "__invalid_dsml_tool_call__"
        assert json.loads(tool_call.function.arguments) == {
            "error": (
                "The model ended after an incomplete tool-call prefix; "
                "retry the intended call."
            )
        }
        assert tool_call.function.name not in {tool.function.name for tool in tools}

    def test_sentinel_collision_uses_unrequested_name_and_valid_sentinel_call_survives(
        self, mock_tokenizer, mock_request
    ):
        sentinel_tool = _make_tool(
            INVALID_DSML_TOOL_NAME, {"value": {"type": "string"}}
        )
        tools = [sentinel_tool]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools

        invalid = parser.extract_tool_calls(
            _tool_calls(_invoke("missing_tool", ("value", "true", "bad"))),
            mock_request,
        )
        valid = parser.extract_tool_calls(
            _tool_calls(_invoke(INVALID_DSML_TOOL_NAME, ("value", "true", "good"))),
            mock_request,
        )

        invalid_call = invalid.tool_calls[0]
        assert invalid_call.function.name not in {tool.function.name for tool in tools}
        assert invalid_call.function.name.startswith(INVALID_DSML_TOOL_NAME)
        assert valid.tool_calls[0].function.name == INVALID_DSML_TOOL_NAME
        assert json.loads(valid.tool_calls[0].function.arguments) == {"value": "good"}

    def test_parse_direct_recovers_truncated_prefix(self, mock_tokenizer, mock_request):
        tools = [_make_tool("get_weather", {"location": {"type": "string"}})]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools

        _, _, tool_calls = parser.parse(DSML_TOOL_START, mock_request)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].name not in {tool.function.name for tool in tools}

    def test_parse_delta_finished_recovers_truncated_prefix(
        self, mock_tokenizer, mock_request
    ):
        tools = [_make_tool("get_weather", {"location": {"type": "string"}})]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools

        delta = parser.parse_delta(DSML_TOOL_START, [], mock_request, finished=True)

        assert delta is not None
        assert delta.tool_calls[0].function.name not in {
            tool.function.name for tool in tools
        }

    def test_streaming_final_recovers_lexical_partial_prefix(
        self, mock_tokenizer, mock_request
    ):
        tools = [_make_tool("get_weather", {"location": {"type": "string"}})]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools
        partial = DSML_TOOL_START[:-3]

        assert parser.parse_delta(partial, [], mock_request, finished=False) is None
        delta = parser.parse_delta("", [], mock_request, finished=True)

        assert delta is not None
        assert delta.tool_calls[0].function.name not in {
            tool.function.name for tool in tools
        }

    def test_finish_recovery_preserves_adjacent_content(
        self, mock_tokenizer, mock_request
    ):
        tools = [_make_tool("get_weather", {"location": {"type": "string"}})]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools
        text = (
            _tool_calls(
                _invoke("get_weather", ("location", "true", "NYC")),
            )
            + "Done."
        )

        delta = parser.parse_delta(text, [], mock_request, finished=True)

        assert delta is not None
        assert delta.content == "Done."
        assert delta.tool_calls[0].function.name == "get_weather"
        assert delta._delta_order == ("tool_calls", "content")


# ── Thinking mode initial state ──────────────────────────────────────


class TestAdapterFinishRecovery:
    def test_adapter_non_streaming_recovers_valid_name_truncation(
        self, mock_tokenizer, mock_request
    ):
        tools = [_make_tool("get_weather", {"location": {"type": "string"}})]
        adapter = DeepSeekV4ParserToolAdapter(mock_tokenizer, tools=tools)
        mock_request.tools = tools
        text = DSML_TOOL_START + _invoke(
            "get_weather", ("location", "true", "NYC")
        ).replace(DSML_INVOKE_END, "")

        result = adapter.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name not in {
            tool.function.name for tool in tools
        }


class TestTentativeToolStreaming:
    def _request_with_weather(self, mock_request):
        tool = _make_tool("get_weather", {"location": {"type": "string"}})
        mock_request.tools = [tool]
        return mock_request

    def test_incomplete_call_never_streams_requested_name_or_arguments(
        self, mock_tokenizer, mock_request
    ):
        request = self._request_with_weather(mock_request)
        parser = DeepSeekV4Parser(mock_tokenizer, tools=request.tools)
        partial = (
            _tool_calls(_invoke("get_weather", ("location", "true", "NYC")))
            .replace(DSML_INVOKE_END, "")
            .replace(DSML_TOOL_END, "")
        )

        first = parser.extract_tool_calls_streaming(
            "", partial, partial, [], [], [], request
        )
        final = parser.finish_streaming()

        assert first is None or not first.tool_calls
        assert final is not None
        assert len(final.tool_calls) == 1
        tool_call = final.tool_calls[0]
        assert tool_call.function.name != "get_weather"
        arguments = json.loads(tool_call.function.arguments)
        assert arguments["error"]
        assert arguments["tool_name"] == "get_weather"
        assert "location" not in arguments

    def test_explicit_close_releases_buffered_call_once(
        self, mock_tokenizer, mock_request
    ):
        request = self._request_with_weather(mock_request)
        parser = DeepSeekV4Parser(mock_tokenizer, tools=request.tools)
        body = _invoke("get_weather", ("location", "true", "NYC"))
        partial = (
            _tool_calls(body).replace(DSML_INVOKE_END, "").replace(DSML_TOOL_END, "")
        )

        tentative = parser.extract_tool_calls_streaming(
            "", partial, partial, [], [], [], request
        )
        closed = parser.extract_tool_calls_streaming(
            partial, partial + DSML_INVOKE_END, DSML_INVOKE_END, [], [], [], request
        )
        aggregate = [delta for delta in (tentative, closed) if delta is not None]

        assert tentative is None or not tentative.tool_calls
        assert len(aggregate) == 1
        tool_calls = aggregate[0].tool_calls
        assert tool_calls[0].function.name == "get_weather"
        arguments = "".join(
            tc.function.arguments or "" for tc in tool_calls if tc.function is not None
        )
        assert json.loads(arguments) == {"location": "NYC"}
        assert parser.finish_streaming() is None

    def test_explicit_close_releases_buffer_when_converter_has_no_delta(
        self, mock_tokenizer, mock_request
    ):
        request = self._request_with_weather(mock_request)
        parser = DeepSeekV4Parser(mock_tokenizer, tools=request.tools)
        events = [
            SemanticEvent(EventType.TOOL_CALL_START, tool_index=0),
            SemanticEvent(EventType.TOOL_NAME, "get_weather", tool_index=0),
            SemanticEvent(
                EventType.ARG_VALUE_CHUNK,
                _param("location", "true", "NYC"),
                tool_index=0,
            ),
        ]

        assert parser._events_to_delta(events) is None
        buffered = list(parser._tentative_tool_deltas[0])
        parser._tool_slots[0].streamed_json = parser._convert_args_for_slot(0, False)

        delta = parser._events_to_delta(
            [SemanticEvent(EventType.TOOL_CALL_END, tool_index=0)]
        )

        assert delta is not None
        assert delta.tool_calls == buffered
        assert delta.tool_calls[0].function.name == "get_weather"

    def test_parallel_close_order_interleaves_buffered_and_current_slots(
        self, mock_tokenizer, mock_request
    ):
        weather = _make_tool("get_weather", {"location": {"type": "string"}})
        clock = _make_tool("get_time", {"zone": {"type": "string"}})
        mock_request.tools = [weather, clock]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=mock_request.tools)

        first = [
            SemanticEvent(EventType.TOOL_CALL_START, tool_index=0),
            SemanticEvent(EventType.TOOL_NAME, "get_weather", tool_index=0),
            SemanticEvent(
                EventType.ARG_VALUE_CHUNK,
                _param("location", "true", "NYC"),
                tool_index=0,
            ),
        ]
        assert parser._events_to_delta(first) is None
        parser._tool_slots[0].streamed_json = parser._convert_args_for_slot(0, False)

        delta = parser._events_to_delta(
            [
                SemanticEvent(EventType.TOOL_CALL_END, tool_index=0),
                SemanticEvent(EventType.TOOL_CALL_START, tool_index=1),
                SemanticEvent(EventType.TOOL_NAME, "get_time", tool_index=1),
                SemanticEvent(
                    EventType.ARG_VALUE_CHUNK,
                    _param("zone", "true", "UTC"),
                    tool_index=1,
                ),
                SemanticEvent(EventType.TOOL_CALL_END, tool_index=1),
            ]
        )

        assert delta is not None
        names = [
            tool_call.function.name
            for tool_call in delta.tool_calls
            if tool_call.function is not None and tool_call.function.name
        ]
        assert names == ["get_weather", "get_time"]

    def test_valid_parallel_sibling_survives_other_sibling_eos_invalidation(
        self, mock_tokenizer, mock_request
    ):
        weather = _make_tool("get_weather", {"location": {"type": "string"}})
        clock = _make_tool("get_time", {"zone": {"type": "string"}})
        mock_request.tools = [weather, clock]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=mock_request.tools)
        valid = _invoke("get_weather", ("location", "true", "NYC"))
        incomplete = _invoke("get_time", ("zone", "true", "UTC")).replace(
            DSML_INVOKE_END, ""
        )
        batch = DSML_TOOL_START + valid + incomplete

        streamed = parser.extract_tool_calls_streaming(
            "", batch, batch, [], [], [], mock_request
        )
        finished = parser.finish_streaming()
        aggregate = [delta for delta in (streamed, finished) if delta is not None]
        calls = [tc for delta in aggregate for tc in delta.tool_calls]

        assert [call.function.name for call in calls] == [
            "get_weather",
            "__invalid_dsml_tool_call__",
        ]
        assert json.loads(calls[0].function.arguments) == {"location": "NYC"}
        assert json.loads(calls[1].function.arguments)["tool_name"] == "get_time"


class TestThinkingModeConfig:
    def test_thinking_true_starts_in_reasoning(self):
        cfg = deepseek_v4_config(thinking=True)
        assert cfg.initial_state.name == "REASONING"

    def test_thinking_false_starts_in_content(self):
        cfg = deepseek_v4_config(thinking=False)
        assert cfg.initial_state.name == "CONTENT"

    @pytest.mark.parametrize(
        ("chat_template_kwargs", "expected_state"),
        [
            ({}, "REASONING"),
            ({"thinking": True}, "REASONING"),
            ({"enable_thinking": True}, "REASONING"),
            ({"reasoning_effort": "high"}, "REASONING"),
            ({"thinking": False}, "CONTENT"),
            ({"enable_thinking": False}, "CONTENT"),
            (
                {"enable_thinking": True, "reasoning_effort": "none"},
                "CONTENT",
            ),
        ],
    )
    def test_parser_thinking_mode_matches_tokenizer_default(
        self, mock_tokenizer, chat_template_kwargs, expected_state
    ):
        parser = DeepSeekV4Parser(
            mock_tokenizer,
            chat_template_kwargs=chat_template_kwargs,
        )
        assert parser.parser_engine_config.initial_state.name == expected_state

    def test_thinking_mode_reasoning_without_tags(self, mock_tokenizer):
        parser = DeepSeekV4Parser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        chunks = [
            "\n\nLet me consider ",
            "this carefully.\n",
            "</think>\n",
            "Here is the result.",
        ]
        reasoning, content = simulate_reasoning_streaming(parser, chunks)
        assert "Let me consider" in reasoning
        assert "Here is the result" in content

    def test_thinking_mode_all_reasoning_no_end_tag(self, mock_tokenizer):
        parser = DeepSeekV4Parser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        chunks = ["I'll review ", "the PR."]
        reasoning, content = simulate_reasoning_streaming(parser, chunks)
        assert "review" in reasoning
        assert "the PR" in reasoning
        assert content == ""

    def test_reasoning_effort_none_overrides_enable_thinking(self, mock_tokenizer):
        p = DeepSeekV4Parser(
            mock_tokenizer,
            chat_template_kwargs={
                "enable_thinking": True,
                "reasoning_effort": "none",
            },
        )
        assert p.parser_engine_config.initial_state.name == "CONTENT"


# ── Implicit reasoning end (missing </think> before tool calls) ─────


class TestImplicitReasoningEnd:
    """Tool call markers end reasoning implicitly when </think> is missing.

    DeepSeek V4 models occasionally omit </think> before emitting tool calls.
    The (REASONING, TOOL_START) transition handles this gracefully.
    """

    @pytest.fixture
    def thinking_parser(self, mock_tokenizer):
        return DeepSeekV4Parser(mock_tokenizer, chat_template_kwargs={"thinking": True})

    def _reasoning_then_tool(self, reasoning_text: str) -> str:
        return reasoning_text + _tool_calls(
            _invoke("get_weather", ("location", "true", "NYC")),
        )

    def test_non_streaming_extract_reasoning_implicit_end(self, thinking_parser):
        text = self._reasoning_then_tool("Let me look up the weather.\n\n")
        reasoning, content = thinking_parser.extract_reasoning(text, None)
        assert reasoning == "Let me look up the weather."
        assert DSML_TOOL_START not in reasoning
        assert DSML_INVOKE_PREFIX not in reasoning
        assert content is None

    def test_non_streaming_extract_tool_calls_implicit_end(
        self, thinking_parser, mock_request
    ):
        text = self._reasoning_then_tool("Let me look up the weather.\n\n")
        result = thinking_parser.extract_tool_calls(text, mock_request)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"location": "NYC"}

    def test_non_streaming_parse_implicit_end(self, thinking_parser, mock_request):
        text = self._reasoning_then_tool("Let me look up the weather.\n\n")
        reasoning, content, tool_calls = thinking_parser.parse(text, mock_request)
        assert reasoning == "Let me look up the weather."
        assert content is None
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        args = json.loads(tool_calls[0].arguments)
        assert args == {"location": "NYC"}

    def test_streaming_reasoning_implicit_end(self, thinking_parser):
        chunks = [
            "Let me look up the weather.\n\n",
            DSML_TOOL_START,
            DSML_INVOKE_PREFIX + "get_weather" + DSML_INVOKE_NAME_END,
        ]
        reasoning, content = simulate_reasoning_streaming(thinking_parser, chunks)
        assert reasoning == "Let me look up the weather."
        assert DSML_TOOL_START not in reasoning
        assert DSML_INVOKE_PREFIX not in reasoning

    def test_streaming_tool_extraction_implicit_end(
        self, thinking_parser, mock_request
    ):
        chunks = [
            "Let me check.\n\n",
            DSML_TOOL_START,
            DSML_INVOKE_PREFIX
            + "get_weather"
            + DSML_INVOKE_NAME_END
            + "\n"
            + _param("location", "true", "NYC")
            + "\n"
            + DSML_INVOKE_END,
            DSML_TOOL_END,
        ]
        results = simulate_tool_streaming(thinking_parser, mock_request, chunks)
        assert collect_function_name(results) == "get_weather"
        args = json.loads(collect_tool_arguments(results))
        assert args == {"location": "NYC"}

    def test_thinking_false_explicit_think_then_tool_call(self, mock_tokenizer):
        parser = DeepSeekV4Parser(mock_tokenizer)
        chunks = [
            DSML_THINK_START,
            "Let me check the weather.",
            DSML_TOOL_START,
            DSML_INVOKE_PREFIX + "get_weather" + DSML_INVOKE_NAME_END,
        ]
        reasoning, content = simulate_reasoning_streaming(parser, chunks)
        assert "Let me check the weather" in reasoning
        assert DSML_TOOL_START not in reasoning
        assert DSML_THINK_START not in reasoning

    def test_non_streaming_parallel_tools_after_implicit_end(
        self, thinking_parser, mock_request
    ):
        text = "I need both.\n\n" + _tool_calls(
            _invoke("get_weather", ("location", "true", "NYC")),
            _invoke("get_time", ("timezone", "true", "EST")),
        )
        result = thinking_parser.extract_tool_calls(text, mock_request)
        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "get_time"

    def test_streaming_implicit_end_trailing_whitespace_stripped(self, thinking_parser):
        chunks = [
            "Reasoning.\n\n\n",
            DSML_TOOL_START,
            DSML_INVOKE_PREFIX + "func" + DSML_INVOKE_NAME_END,
        ]
        reasoning, content = simulate_reasoning_streaming(thinking_parser, chunks)
        assert reasoning == "Reasoning."


# ── Wrapper argument unwrapping ──────────────────────────────────────


class TestWrapperUnwrapping:
    def test_unwrap_arguments_wrapper(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        tool = ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        )

        result = _unwrap_wrapper_args(
            '{"arguments": {"location": "Beijing"}}',
            [tool],
            "get_weather",
        )
        assert json.loads(result) == {"location": "Beijing"}

    def test_unwrap_input_wrapper(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        tool = ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        )

        result = _unwrap_wrapper_args(
            '{"input": {"location": "Beijing"}}',
            [tool],
            "get_weather",
        )
        assert json.loads(result) == {"location": "Beijing"}

    def test_no_unwrap_when_key_in_schema(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        tool = ChatCompletionToolsParam(
            type="function",
            function={
                "name": "func",
                "parameters": {
                    "type": "object",
                    "properties": {"arguments": {"type": "string"}},
                },
            },
        )

        result = _unwrap_wrapper_args(
            '{"arguments": "some value"}',
            [tool],
            "func",
        )
        assert json.loads(result) == {"arguments": "some value"}

    def test_no_unwrap_when_no_tools(self):
        result = _unwrap_wrapper_args(
            '{"arguments": {"location": "Beijing"}}',
            None,
            "get_weather",
        )
        assert json.loads(result) == {"arguments": {"location": "Beijing"}}

    def test_unwrap_json_string_inner(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        tool = ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        )

        result = _unwrap_wrapper_args(
            '{"arguments": "{\\"location\\": \\"Beijing\\"}"}',
            [tool],
            "get_weather",
        )
        assert json.loads(result) == {"location": "Beijing"}


# ── Parallel tool call wrapper unwrapping ───────────────────────────


def _make_tool(name, properties):
    from vllm.entrypoints.openai.chat_completion.protocol import (  # noqa: E501
        ChatCompletionToolsParam,
    )

    return ChatCompletionToolsParam(
        type="function",
        function={
            "name": name,
            "parameters": {
                "type": "object",
                "properties": properties,
            },
        },
    )


def _invoke(name, *params):
    body = "\n".join(_param(n, s, v) for n, s, v in params)
    return (
        f"{DSML_INVOKE_PREFIX}{name}{DSML_INVOKE_NAME_END}\n{body}\n{DSML_INVOKE_END}"
    )


def _tool_calls(*invokes):
    return DSML_TOOL_START + "\n".join(invokes) + DSML_TOOL_END


class TestLegacyToolFramingResidue:
    def test_named_request_drops_legacy_suffix_after_v4_calls(
        self, mock_tokenizer, mock_request
    ):
        tool = _make_tool("get_weather", {"location": {"type": "string"}})
        tools = [tool]
        mock_request.tools = tools
        mock_request.tool_choice = {
            "type": "function",
            "function": {"name": "get_weather"},
        }
        calls = _tool_calls(
            _invoke("get_weather", ("location", "true", "NYC")),
            _invoke("get_weather", ("location", "true", "NYC")),
        )
        legacy_suffix = "<｜｜tool▁calls▁end｜｜>\n\n"
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)

        for text in (calls, calls + legacy_suffix):
            _, content, tool_calls = parser.parse(text, mock_request)

            assert [tool_call.name for tool_call in tool_calls or []] == [
                "get_weather",
                "get_weather",
            ]
            assert content in (None, "")


class TestParallelUnwrapping:
    @pytest.fixture
    def weather_tool(self):
        return _make_tool(
            "get_weather",
            {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
        )

    @pytest.fixture
    def time_tool(self):
        return _make_tool(
            "get_time",
            {"timezone": {"type": "string"}},
        )

    @pytest.mark.parametrize(
        "weather_args, expected",
        [
            (
                '{"location": "NYC", "unit": "celsius"}',
                {"location": "NYC", "unit": "celsius"},
            ),
            ('{"location": "NYC"}', {"location": "NYC"}),
        ],
        ids=["all_props", "subset_props"],
    )
    def test_unwrap_parallel_uses_correct_schema(
        self,
        mock_tokenizer,
        mock_request,
        weather_tool,
        time_tool,
        weather_args,
        expected,
    ):
        tools = [weather_tool, time_tool]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools

        text = _tool_calls(
            _invoke("get_weather", ("arguments", "false", weather_args)),
            _invoke("get_time", ("timezone", "true", "EST")),
        )

        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        args0 = json.loads(result.tool_calls[0].function.arguments)
        assert args0 == expected
        assert result.tool_calls[1].function.name == "get_time"
        args1 = json.loads(result.tool_calls[1].function.arguments)
        assert args1 == {"timezone": "EST"}

    def test_duplicate_parallel_wrappers_use_each_call_schema(
        self, mock_tokenizer, mock_request
    ):
        first_tool = _make_tool("first", {"arguments": {"type": "object"}})
        second_tool = _make_tool("second", {"value": {"type": "string"}})
        tools = [first_tool, second_tool]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools
        raw_arguments = '{"value": "payload"}'
        text = _tool_calls(
            _invoke("first", ("arguments", "false", raw_arguments)),
            _invoke("second", ("arguments", "false", raw_arguments)),
        )

        result = parser.extract_tool_calls(text, mock_request)

        assert [call.function.name for call in result.tool_calls] == [
            "first",
            "second",
        ]
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "arguments": {"value": "payload"}
        }
        assert json.loads(result.tool_calls[1].function.arguments) == {
            "value": "payload"
        }

    def test_unwrap_parallel_streaming(
        self, mock_tokenizer, mock_request, weather_tool, time_tool
    ):
        tools = [weather_tool, time_tool]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools

        chunks = [
            DSML_TOOL_START,
            _invoke(
                "get_weather",
                ("arguments", "false", '{"location": "NYC"}'),
            ),
            _invoke("get_time", ("timezone", "true", "EST")),
            DSML_TOOL_END,
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)
        final_delta, _ = results[-1]
        finish_delta = parser.finish_streaming()
        extracted = parser._build_extracted_result(final_delta, finish_delta)

        assert extracted.tools_called is True
        assert len(extracted.tool_calls) == 2
        args0 = json.loads(extracted.tool_calls[0].function.arguments)
        assert args0 == {"location": "NYC"}
        args1 = json.loads(extracted.tool_calls[1].function.arguments)
        assert args1 == {"timezone": "EST"}

    def test_no_unwrap_parallel_when_no_match(
        self, mock_tokenizer, mock_request, weather_tool, time_tool
    ):
        tools = [weather_tool, time_tool]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools

        text = _tool_calls(
            _invoke(
                "get_weather",
                ("arguments", "false", '{"unknown_key": "val"}'),
            ),
            _invoke("get_time", ("timezone", "true", "EST")),
        )

        result = parser.extract_tool_calls(text, mock_request)

        assert len(result.tool_calls) == 2
        args0 = json.loads(result.tool_calls[0].function.arguments)
        assert args0 == {"arguments": {"unknown_key": "val"}}
        args1 = json.loads(result.tool_calls[1].function.arguments)
        assert args1 == {"timezone": "EST"}

    def test_identical_parallel_invokes_preserve_opaque_arguments(
        self, mock_tokenizer, mock_request
    ):
        tool = _make_tool(
            "dispatch",
            {
                "city": {"type": "string"},
                "payload": {"type": "string"},
            },
        )
        tools = [tool]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools
        opaque_payload = (
            '{"script":"printf \'<tag>\\n\'","meta":{"items":[1,{"quoted":"a\\"b"}]}}'
        )
        invoke = _invoke(
            "dispatch",
            ("city", "true", "Boston"),
            ("payload", "true", opaque_payload),
        )

        result = parser.extract_tool_calls(_tool_calls(invoke, invoke), mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        for tool_call in result.tool_calls:
            assert tool_call.function.name == "dispatch"
            assert json.loads(tool_call.function.arguments) == {
                "city": "Boston",
                "payload": opaque_payload,
            }

    def test_identical_parallel_invokes_stream_without_merging(
        self, mock_tokenizer, mock_request
    ):
        tool = _make_tool(
            "dispatch",
            {
                "city": {"type": "string"},
                "payload": {"type": "string"},
            },
        )
        tools = [tool]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools
        opaque_payload = (
            '{"script":"printf \'<tag>\\n\'","meta":{"items":[1,{"quoted":"a\\"b"}]}}'
        )
        invoke = _invoke(
            "dispatch",
            ("city", "true", "Boston"),
            ("payload", "true", opaque_payload),
        )

        results = simulate_tool_streaming(
            parser,
            mock_request,
            [DSML_TOOL_START, invoke, invoke, DSML_TOOL_END],
        )
        final_delta, _ = results[-1]
        finish_delta = parser.finish_streaming()
        extracted = parser._build_extracted_result(final_delta, finish_delta)

        assert extracted.tools_called is True
        assert len(extracted.tool_calls) == 2
        for tool_call in extracted.tool_calls:
            assert tool_call.function.name == "dispatch"
            assert json.loads(tool_call.function.arguments) == {
                "city": "Boston",
                "payload": opaque_payload,
            }

    def test_single_invoke_remains_single(self, mock_tokenizer, mock_request):
        tool = _make_tool("get_weather", {"city": {"type": "string"}})
        tools = [tool]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools

        result = parser.extract_tool_calls(
            _tool_calls(_invoke("get_weather", ("city", "true", "Boston"))),
            mock_request,
        )

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Boston"}

    def test_unwrap_single_tool_still_works(self, mock_tokenizer, mock_request):
        tool = _make_tool("get_weather", {"location": {"type": "string"}})
        tools = [tool]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools

        text = _tool_calls(
            _invoke(
                "get_weather",
                ("arguments", "false", '{"location": "Beijing"}'),
            ),
        )

        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"location": "Beijing"}


# ── Streaming wrapper consistency ─────────────────────────────────────


class TestStreamingWrapperConsistency:
    """Streamed arg deltas must stay consistent with final extraction
    when wrapper params like 'arguments' are unwrapped."""

    def test_streaming_wrapper_unwrap_consistency(self, mock_tokenizer, mock_request):
        tool = _make_tool("get_weather", {"location": {"type": "string"}})
        tools = [tool]
        parser = DeepSeekV4Parser(mock_tokenizer, tools=tools)
        mock_request.tools = tools

        chunks = [
            DSML_TOOL_START,
            _invoke(
                "get_weather",
                ("arguments", "false", '{"location": "NYC"}'),
            ),
            DSML_TOOL_END,
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)
        streamed_args = collect_tool_arguments(results)

        final_delta, _ = results[-1]
        finish_delta = parser.finish_streaming()
        extracted = parser._build_extracted_result(final_delta, finish_delta)

        assert extracted.tools_called is True
        assert len(extracted.tool_calls) == 1

        final_args = extracted.tool_calls[0].function.arguments
        assert json.loads(final_args) == {"location": "NYC"}

        assert '"arguments"' not in streamed_args, (
            f"Streamed args should not contain wrapper key, got: {streamed_args!r}"
        )

        assert final_args.startswith(streamed_args), (
            f"Extracted args {final_args!r} "
            f"should start with streamed args {streamed_args!r}"
        )


# ── DelegatingParser: large delta with </think> + tool calls ─────────

_DSV4_FULL_VOCAB = {
    DSML_THINK_START: 128821,
    DSML_THINK_END: 128822,
    DSML_TOOL_START: 128823,
    DSML_TOOL_END: 128824,
}


class _DeepSeekV4Delegating(DelegatingParser):
    reasoning_parser_cls = DeepSeekV4ParserReasoningAdapter
    tool_parser_cls = DeepSeekV4ParserToolAdapter


def _dsv4_tokens(
    reasoning: str,
    tool_name: str,
    params: list[tuple[str, str, str]],
) -> list[tuple[int, str]]:
    """Build a token sequence: reasoning + </think> + DSML tool block."""
    tokens: list[tuple[int, str]] = []
    tid = 100

    for word in reasoning.split(" "):
        prefix = " " if tokens else ""
        tokens.append((tid, prefix + word))
        tid += 1

    tokens.append((_DSV4_FULL_VOCAB[DSML_THINK_END], DSML_THINK_END))

    tokens.append((tid, "\n\n"))
    tid += 1

    tokens.append((_DSV4_FULL_VOCAB[DSML_TOOL_START], DSML_TOOL_START))

    tokens.append((tid, "\n"))
    tid += 1

    invoke_prefix_text = f"{DSML_INVOKE_PREFIX}{tool_name}{DSML_INVOKE_NAME_END}"
    tokens.append((tid, invoke_prefix_text))
    tid += 1

    tokens.append((tid, "\n"))
    tid += 1

    for name, is_str, value in params:
        param_text = _param(name, is_str, value)
        tokens.append((tid, param_text))
        tid += 1
        tokens.append((tid, "\n"))
        tid += 1

    tokens.append((tid, DSML_INVOKE_END))
    tid += 1

    tokens.append((tid, "\n"))
    tid += 1

    tokens.append((_DSV4_FULL_VOCAB[DSML_TOOL_END], DSML_TOOL_END))

    return tokens


class TestDelegatingParserLargeDelta:
    """Regression: tool calls lost when </think> + DSML arrive in same delta.

    The DelegatingParser used by the serving layer splits reasoning and
    tool parsing across two separate engine instances.  When </think> and
    the entire DSML tool block arrive in a single large streaming delta,
    the content transfer from reasoning adapter to tool adapter must
    preserve the tool call text.
    """

    @pytest.fixture
    def dsv4_tokens(self):
        return _dsv4_tokens(
            reasoning="The user wants the current weather in Berlin.",
            tool_name="get_weather",
            params=[
                ("location", "true", "Berlin"),
                ("units", "true", "celsius"),
            ],
        )

    @pytest.fixture
    def dsv4_tokenizer(self, dsv4_tokens):
        return MockTokenizer(
            vocab=dict(_DSV4_FULL_VOCAB),
            tokens=dsv4_tokens,
        )

    @pytest.mark.parametrize(
        "chunk_size",
        [1, 2, 3, 5, None],
        ids=lambda c: f"chunk={c}",
    )
    def test_tool_calls_extracted_at_all_chunk_sizes(
        self, dsv4_tokenizer, dsv4_tokens, chunk_size
    ):
        parser = _DeepSeekV4Delegating(
            dsv4_tokenizer,
            chat_template_kwargs={"thinking": True},
        )
        deltas = replay_streaming(
            parser,
            dsv4_tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            tools=_GET_WEATHER_TOOLS,
        )
        output = collect_output(deltas)

        assert "The user wants" in output.reasoning
        assert len(output.tool_calls) == 1, (
            f"Expected 1 tool call but got {len(output.tool_calls)}; "
            f"reasoning={output.reasoning!r}, content={output.content!r}"
        )
        assert output.tool_calls[0]["name"] == "get_weather"
        args = json.loads(output.tool_calls[0]["arguments"])
        assert args == {"location": "Berlin", "units": "celsius"}

    def test_default_thinking_extracts_tool_call_without_think_end(self, dsv4_tokens):
        tokens = [
            token
            for token in dsv4_tokens
            if token[0] != _DSV4_FULL_VOCAB[DSML_THINK_END]
        ]
        tokenizer = MockTokenizer(
            vocab=dict(_DSV4_FULL_VOCAB),
            tokens=tokens,
        )
        parser = _DeepSeekV4Delegating(tokenizer)

        deltas = replay_streaming(
            parser,
            tokens,
            chunk_size=1,
            finished_on_last=True,
            tools=_GET_WEATHER_TOOLS,
            prompt_token_ids=[_DSV4_FULL_VOCAB[DSML_THINK_START]],
        )
        output = collect_output(deltas)

        assert "The user wants" in output.reasoning
        assert output.content == ""
        assert len(output.tool_calls) == 1
        assert output.tool_calls[0]["name"] == "get_weather"
        args = json.loads(output.tool_calls[0]["arguments"])
        assert args == {"location": "Berlin", "units": "celsius"}

    def test_verbatim_content_does_not_duplicate_held_terminal_prefix(self):
        eos_text = "<｜end▁of▁sentence｜>"
        token_ids = [
            128822,
            24313,
            19995,
            16562,
            13400,
            10177,
            10499,
            3362,
            21,
            95,
            1,
        ]
        token_texts = [
            DSML_THINK_END,
            '{"',
            "answer",
            '\":\"',
            "beta",
            '\",\"',
            "count",
            '\":',
            "3",
            "}",
            eos_text,
        ]
        parser = _DeepSeekV4Delegating(
            MockTokenizer(
                vocab={DSML_THINK_END: 128822, eos_text: 1},
                tokens=list(zip(token_ids, token_texts)),
            ),
            chat_template_kwargs={"thinking": True},
        )
        request = _test_request()
        chunks = [
            (DSML_THINK_END + '{"answer":"beta","', token_ids[:6]),
            ('count":3}', token_ids[6:]),
        ]

        deltas = [
            parser.parse_delta(
                chunk,
                ids,
                request,
                prompt_token_ids=[] if index == 0 else None,
                finished=index == len(chunks) - 1,
            )
            for index, (chunk, ids) in enumerate(chunks)
        ]

        content = collect_output(deltas).content
        assert content == '{"answer":"beta","count":3}'
        assert json.loads(content) == {"answer": "beta", "count": 3}

    def test_eos_drop_token_does_not_swallow_tool_calls(self):
        """Tool calls must survive when an EOS DROP token's ID is in
        delta_token_ids but its text is absent from delta_text.

        At large stream_interval the EOS token ID arrives in the same
        delta as </think> + tool calls but the detokenizer strips the
        EOS text.  The scanner's _rebuild_from_anchors defers all text
        after </think> when it can't find the EOS anchor text.  The
        reasoning adapter's finish_streaming must flush deferred text
        as content (with skip_tool_parsing), not as tool calls.
        """
        eos_text = "<｜end▁of▁sentence｜>"
        eos_id = 128801
        vocab = {
            DSML_THINK_START: 128821,
            DSML_THINK_END: 128822,
            eos_text: eos_id,
        }

        reasoning = "The user wants weather."
        tool_block = (
            "\n\n"
            + DSML_TOOL_START
            + "\n"
            + DSML_INVOKE_PREFIX
            + "get_weather"
            + DSML_INVOKE_NAME_END
            + "\n"
            + _param("location", "true", "Berlin")
            + "\n"
            + DSML_INVOKE_END
            + "\n"
            + DSML_TOOL_END
        )
        # delta_text does NOT include EOS text (detokenizer strips it)
        full_text = reasoning + DSML_THINK_END + tool_block
        # Build token list: word-split reasoning, then special tokens,
        # then word-split tool block content, then EOS.
        # EOS ID is present but its text is NOT in delta_text.
        tokens: list[tuple[int, str]] = []
        tid = 100
        for word in reasoning.split(" "):
            pfx = " " if tokens else ""
            tokens.append((tid, pfx + word))
            tid += 1
        tokens.append((128822, DSML_THINK_END))
        for ch in tool_block:
            tokens.append((tid, ch))
            tid += 1
        tokens.append((eos_id, eos_text))

        all_ids = [t[0] for t in tokens]
        tokenizer = MockTokenizer(vocab=vocab, tokens=tokens)
        request = _test_request(tools=_GET_WEATHER_TOOLS)

        # All-in-one delta: EOS ID in token_ids but text NOT in
        # delta_text (detokenizer strips EOS).  This is the scenario
        # at large stream_interval.
        parser = _DeepSeekV4Delegating(
            tokenizer,
            chat_template_kwargs={"thinking": True},
        )
        deltas = [
            parser.parse_delta(
                full_text,
                all_ids,
                request,
                prompt_token_ids=[],
                finished=True,
            )
        ]

        output = collect_output(deltas)

        assert "The user wants" in output.reasoning
        assert len(output.tool_calls) == 1, (
            f"Expected 1 tool call but got {len(output.tool_calls)}; "
            f"reasoning={output.reasoning!r}, content={output.content!r}"
        )
        assert output.tool_calls[0]["name"] == "get_weather"
        args = json.loads(output.tool_calls[0]["arguments"])
        assert args == {"location": "Berlin"}

    @pytest.mark.parametrize(
        "chunk_size",
        [1, 2, 3, 5, None],
        ids=lambda c: f"chunk={c}",
    )
    def test_eos_not_leaked_when_reasoning_never_ends(self, chunk_size):
        """EOS must not leak into reasoning_content when the model never
        emits </think> (generation ends while still in REASONING state)."""
        eos_text = "<｜end▁of▁sentence｜>"
        eos_id = 128801
        vocab = {
            **_DSV4_FULL_VOCAB,
            eos_text: eos_id,
        }

        reasoning_text = "Good morning! How can I help you today?"
        tokens: list[tuple[int, str]] = []
        tid = 100
        for word in reasoning_text.split(" "):
            prefix = " " if tokens else ""
            tokens.append((tid, prefix + word))
            tid += 1
        tokens.append((eos_id, eos_text))

        tokenizer = MockTokenizer(vocab=vocab, tokens=tokens)
        parser = _DeepSeekV4Delegating(
            tokenizer,
            chat_template_kwargs={"thinking": True},
        )
        deltas = replay_streaming(
            parser,
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
        )
        output = collect_output(deltas)

        assert reasoning_text in output.reasoning
        assert eos_text not in output.reasoning
        assert output.content == ""
        assert output.tool_calls == []
