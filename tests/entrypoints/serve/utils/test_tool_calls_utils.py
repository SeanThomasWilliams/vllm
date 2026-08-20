# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.serve.utils.tool_calls_utils import (
    maybe_filter_parallel_tool_calls,
)


def _request(parallel_tool_calls: bool | None) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="deepseek-v4-flash-0731",
        messages=[{"role": "user", "content": "Call the tool."}],
        parallel_tool_calls=parallel_tool_calls,
    )


def _tool_call(call_id: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        function=FunctionCall(name="dispatch", arguments='{"opaque":"a<b>"}'),
    )


def test_request_default_preserves_parallel_tool_calls():
    request = ChatCompletionRequest(
        model="deepseek-v4-flash-0731",
        messages=[{"role": "user", "content": "Call the tool."}],
    )
    choice = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(
            role="assistant",
            tool_calls=[_tool_call("call-0"), _tool_call("call-1")],
        ),
        finish_reason="tool_calls",
    )

    expected = choice.model_dump(mode="json")
    result = maybe_filter_parallel_tool_calls(choice, request)

    assert request.parallel_tool_calls is True
    assert result is choice
    assert result.model_dump(mode="json") == expected


@pytest.mark.parametrize("parallel_tool_calls", [False, True, None])
def test_nonstreaming_filters_only_when_parallel_calls_explicitly_disabled(
    parallel_tool_calls: bool | None,
):
    choice = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(
            role="assistant",
            tool_calls=[_tool_call("call-0"), _tool_call("call-1")],
        ),
        finish_reason="tool_calls",
    )

    expected = choice.model_dump(mode="json")
    if parallel_tool_calls is False:
        expected["message"]["tool_calls"] = expected["message"]["tool_calls"][:1]

    result = maybe_filter_parallel_tool_calls(choice, _request(parallel_tool_calls))

    assert result is choice
    assert result.model_dump(mode="json") == expected


@pytest.mark.parametrize("parallel_tool_calls", [False, True, None])
def test_streaming_filters_only_when_parallel_calls_explicitly_disabled(
    parallel_tool_calls: bool | None,
):
    choice = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=index,
                    id=f"call-{index}",
                    type="function",
                    function=DeltaFunctionCall(
                        name="dispatch", arguments='{"opaque":"a<b>"}'
                    ),
                )
                for index in range(2)
            ]
        ),
        finish_reason="tool_calls",
    )

    expected = choice.model_dump(mode="json")
    if parallel_tool_calls is False:
        expected["delta"]["tool_calls"] = [
            tool_call
            for tool_call in expected["delta"]["tool_calls"]
            if tool_call["index"] == 0
        ]

    result = maybe_filter_parallel_tool_calls(choice, _request(parallel_tool_calls))

    assert result is choice
    assert result.model_dump(mode="json") == expected
