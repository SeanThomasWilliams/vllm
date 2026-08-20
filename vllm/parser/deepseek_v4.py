# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 parser: ``<think>``/``</think>``
reasoning plus DSML tool calls in a single state machine.

DeepSeek V4 output format::

    <think>
    ...reasoning...
    </think>
    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="func_name">
    <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
    <｜DSML｜parameter name="count" string="false">5</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
"""

from __future__ import annotations

import contextlib
import functools
import json
from typing import TYPE_CHECKING

import regex as re

from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.tool_parsers.utils import find_tool_name, find_tool_properties

if TYPE_CHECKING:
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage, DeltaToolCall
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

_DSML = "｜DSML｜"

DSML_THINK_START = "<think>"
DSML_THINK_END = "</think>"
DSML_TOOL_START = f"<{_DSML}tool_calls>"
DSML_TOOL_END = f"</{_DSML}tool_calls>"
DSML_FUNCTION_TOOL_START = f"<{_DSML}function_calls>"
DSML_INVOKE_PREFIX = f'<{_DSML}invoke name="'
DSML_INVOKE_NAME_END = '">'
DSML_INVOKE_END = f"</{_DSML}invoke>"
DSML_PARAM_START = f"<{_DSML}parameter"
DSML_PARAM_CLOSE = f"</{_DSML}parameter>"
INVALID_DSML_TOOL_NAME = "__invalid_dsml_tool_call__"

_ESCAPED_DSML = re.escape(_DSML)
_DSML_TAG_RE = re.compile(rf"<(/?){_ESCAPED_DSML}[a-z_]+(?=[\s>])")
_DSML_TOOL_PREAMBLE_RE = re.compile(rf"(?:\s*<{_ESCAPED_DSML}(?:tool_calls)?\s*)+$")
_PARAM_RE = re.compile(
    rf'<{_ESCAPED_DSML}parameter\s+name="([^"]+)"\s+string="(true|false)">'
    rf"(.*?)</{_ESCAPED_DSML}parameter>",
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(
    rf'<{_ESCAPED_DSML}parameter\s+name="([^"]+)"\s+string="(true|false)">'
    rf"(.*)$",
    re.DOTALL,
)


def _escape_unparsed_dsml(text: str) -> str:
    """Neutralize DSML-looking tags that were not accepted as tool syntax."""
    escaped = _DSML_TAG_RE.sub(lambda match: f"&lt;{match.group(0)[1:]}", text)
    escaped = escaped.replace(f"</{_DSML}", f"&lt;/{_DSML}")
    return escaped.replace(f"<{_DSML}", f"&lt;{_DSML}")


def _strip_parsed_tool_preamble(text: str) -> str:
    """Drop malformed DSML framing immediately before parsed tool calls."""
    return _DSML_TOOL_PREAMBLE_RE.sub("", text)


def _strip_parsed_tool_residue(text: str) -> str:
    """Drop DSML framing residue beside an accepted tool call."""
    text = _strip_parsed_tool_preamble(text)
    markers = (
        f"<{_DSML}",
        f"</{_DSML}",
    )
    cutoffs = [text.find(marker) for marker in markers if marker in text]
    if not cutoffs:
        return text
    return text[: min(cutoffs)].rstrip()


def _resolve_stray_tool_framing(
    pending: str | None,
    content: str | None,
    has_tool_calls: bool,
    finished: bool,
) -> tuple[str | None, str | None]:
    """Delay ambiguous DSML framing until tool-call context disambiguates it."""
    if pending is not None:
        if has_tool_calls:
            pending = None
        elif content:
            content = pending + content
            pending = None
        elif finished:
            content = pending
            pending = None

    stripped = content.strip() if content else ""
    partial_tool_preamble = bool(
        stripped
        and any(
            terminal.startswith(stripped)
            for terminal in (DSML_TOOL_START, DSML_INVOKE_PREFIX)
        )
    )
    if content and (
        stripped == "<"
        or _DSML_TOOL_PREAMBLE_RE.fullmatch(content)
        or partial_tool_preamble
    ):
        if has_tool_calls:
            content = None
        elif not finished:
            pending, content = content, None

    return pending, content


def _malformed_name_candidate(raw_name: str) -> tuple[str, str]:
    """Split a missing-quote tool name from the DSML text it consumed."""
    stripped = raw_name.lstrip()
    match = re.match(r'([^\s>"]+)', stripped)
    if match is None:
        return "", ""
    return match.group(1), stripped[match.end() :]


def _verbatim_content_after_reasoning(
    model_output: str,
    starts_in_reasoning: bool,
) -> str | None:
    """Preserve DSML bytes for the separate tool-parser adapter."""
    end_idx = model_output.find(DSML_THINK_END)
    start_idx = model_output.find(DSML_THINK_START)
    if end_idx < 0 or (not starts_in_reasoning and start_idx < 0):
        return None
    prefix = model_output[:start_idx] if 0 <= start_idx < end_idx else ""
    return prefix + model_output[end_idx + len(DSML_THINK_END) :]


def _dsml_arg_converter(raw_args: str, partial: bool) -> str:
    params: dict[str, object] = {}

    last_end = 0
    for m in _PARAM_RE.finditer(raw_args):
        name, is_str, value = m.group(1), m.group(2), m.group(3)
        if is_str == "true":
            params[name] = value
        else:
            try:
                params[name] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                params[name] = value
        last_end = m.end()

    if partial:
        pm = _PARTIAL_PARAM_RE.search(raw_args, last_end)
        if pm:
            name, is_str, value = pm.group(1), pm.group(2), pm.group(3)
            if is_str == "true":
                params[name] = value
            else:
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    params[name] = json.loads(value)

    return json.dumps(params, ensure_ascii=False)


def _unwrap_wrapper_args(
    args_json: str,
    tools: list[Tool] | None,
    func_name: str | None,
) -> str:
    if not tools or not func_name:
        return args_json
    try:
        args = json.loads(args_json)
    except (json.JSONDecodeError, ValueError):
        return args_json
    if not isinstance(args, dict):
        return args_json
    properties = find_tool_properties(tools, func_name)
    if not properties:
        return args_json
    allowed = set(properties.keys())
    for wrapper in ("arguments", "input"):
        if set(args.keys()) != {wrapper} or wrapper in allowed:
            continue
        inner = args[wrapper]
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except json.JSONDecodeError:
                return args_json
        if isinstance(inner, dict) and set(inner.keys()).issubset(allowed):
            return json.dumps(inner, ensure_ascii=False)
    return args_json


@functools.cache
def deepseek_v4_config(thinking: bool = False) -> ParserEngineConfig:
    return ParserEngineConfig(
        name="deepseek_v4",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        terminals={
            "THINK_START": DSML_THINK_START,
            "THINK_END": DSML_THINK_END,
            "TOOL_START": DSML_TOOL_START,
            "TOOL_END": DSML_TOOL_END,
            "INVALID_TOOL_START": DSML_FUNCTION_TOOL_START,
            "INVOKE_PREFIX": DSML_INVOKE_PREFIX,
            "INVOKE_NAME_END": DSML_INVOKE_NAME_END,
            "INVOKE_END": DSML_INVOKE_END,
            "PARAM_CLOSE": DSML_PARAM_CLOSE,
        },
        token_id_terminals={
            "THINK_START": DSML_THINK_START,
            "THINK_END": DSML_THINK_END,
            "TOOL_START": DSML_TOOL_START,
            "TOOL_END": DSML_TOOL_END,
        },
        transitions={
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            # Absorb a bare </think> with no prior <think>
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            # Absorb a duplicate <think> while already reasoning
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            # Tool call beginning while still inside <think>
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
            # Recover a plausible invocation when the model omits the outer
            # tool_calls wrapper. Bare closing wrappers are structural noise.
            # Quarantine an explicitly wrong wrapper as content. Without this
            # state, bare-invoke recovery would accept the invoke nested inside
            # a model-emitted ``function_calls`` block as valid DSML.
            (ParserState.CONTENT, "INVALID_TOOL_START"): Transition(
                ParserState.MESSAGE_HEADER,
                (EventType.TEXT_CHUNK,),
            ),
            (ParserState.REASONING, "INVALID_TOOL_START"): Transition(
                ParserState.MESSAGE_HEADER,
                (EventType.REASONING_END, EventType.TEXT_CHUNK),
            ),
            (ParserState.CONTENT, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.REASONING, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            ),
            (ParserState.CONTENT, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            (ParserState.CONTENT, "INVOKE_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            (ParserState.CONTENT, "PARAM_CLOSE"): Transition(
                ParserState.CONTENT,
                (),
            ),
            (ParserState.TOOL_PREAMBLE, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_NAME, "INVOKE_NAME_END"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
            (ParserState.TOOL_ARGS, "INVOKE_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            # Parallel tool calls
            (ParserState.TOOL_BETWEEN, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
        },
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
        arg_converter=_dsml_arg_converter,
        arg_structural_chars=frozenset(">"),
        skip_tool_parsing_preserve_terminals=frozenset({"PARAM_CLOSE"}),
        strip_content_whitespace_with_tools=False,
        tool_args_json=False,
        validate_tool_names=True,
    )


class DeepSeekV4Parser(ParserEngine):
    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.pop("chat_template_kwargs", None) or {}
        thinking = bool(
            chat_kwargs.get("thinking") or chat_kwargs.get("enable_thinking")
        )
        if "thinking" not in chat_kwargs and "enable_thinking" not in chat_kwargs:
            thinking = True
        thinking = thinking and chat_kwargs.get("reasoning_effort") != "none"
        super().__init__(
            tokenizer,
            tools,
            parser_engine_config=deepseek_v4_config(thinking=thinking),
            **kwargs,
        )
        self._arg_converter = self._convert_args
        self._pending_stray_tool_framing: str | None = None
        self._invalid_tool_names: set[str] = set()
        self._tentative_tool_deltas: dict[int, list[DeltaToolCall]] = {}
        self._finish_invalid_slot: int | None = None
        self._finish_invalid_prefix: str | None = None

    def _reset(self, initial_state: ParserState | None = None) -> None:
        super()._reset(initial_state=initial_state)
        self._pending_stray_tool_framing = None
        self._invalid_tool_names.clear()
        self._tentative_tool_deltas.clear()
        self._finish_invalid_slot = None
        self._finish_invalid_prefix = None

    @staticmethod
    def _invalid_tool_arguments(
        message: str,
        attempted_name: str | None = None,
    ) -> str:
        arguments: dict[str, str] = {"error": message}
        if attempted_name:
            arguments["tool_name"] = attempted_name
        return json.dumps(arguments, ensure_ascii=False)

    def _choose_invalid_tool_name(self) -> str:
        candidate = INVALID_DSML_TOOL_NAME
        suffix = 0
        while candidate in self._invalid_tool_names or find_tool_name(
            self._tools, candidate
        ):
            suffix += 1
            candidate = f"{INVALID_DSML_TOOL_NAME}_{suffix}"
        self._invalid_tool_names.add(candidate)
        return candidate

    def _mark_invalid_tool_slot(self, idx: int, message: str) -> None:
        slot = self._tool_slots[idx]
        if slot.invalid:
            return
        attempted_name = slot.name
        slot.name = self._choose_invalid_tool_name()
        slot.invalid = True
        slot.name_sent = False
        slot.string_keys = None
        slot._args_parts[:] = [self._invalid_tool_arguments(message, attempted_name)]
        slot._args_joined = None
        slot.streamed_json = ""

    def _accept_tool_name(self, name: str) -> bool:
        if name in self._invalid_tool_names:
            return True
        return super()._accept_tool_name(name)

    @staticmethod
    def _partial_dsml_start(text: str | None) -> str | None:
        if not text:
            return None
        stripped = text.lstrip()
        if len(stripped) < 2:
            return None
        for terminal in (DSML_TOOL_START, DSML_INVOKE_PREFIX):
            if terminal.startswith(stripped):
                return text
        return None

    def _before_finish(self) -> None:
        self._finish_invalid_slot = None
        self._finish_invalid_prefix = None
        if self.skip_tool_parsing:
            return
        state = self._engine.state
        if state in (ParserState.TOOL_NAME, ParserState.TOOL_ARGS):
            idx = self._engine.tool_index
            if idx >= 0:
                self._ensure_slot(idx)
                self._mark_invalid_tool_slot(
                    idx,
                    "The model ended during an incomplete tool call; "
                    "retry the intended call.",
                )
                self._finish_invalid_slot = idx
                return

        if state == ParserState.TOOL_PREAMBLE:
            self._finish_invalid_prefix = DSML_TOOL_START
            return

        self._finish_invalid_prefix = self._partial_dsml_start(
            self._engine._lexer.buffer
        )
        if self._finish_invalid_prefix is None:
            self._finish_invalid_prefix = self._partial_dsml_start(
                self._pending_stray_tool_framing
            )

    @staticmethod
    def _delta_has_tool_index(delta: DeltaMessage | None, idx: int) -> bool:
        return bool(delta and any(tc.index == idx for tc in delta.tool_calls))

    def _make_invalid_tool_delta(self, idx: int) -> DeltaMessage:
        from vllm.entrypoints.openai.engine.protocol import (
            DeltaFunctionCall,
            DeltaMessage,
            DeltaToolCall,
        )

        slot = self._tool_slots[idx]
        slot.name_sent = True
        self._ensure_tool_id(slot, slot.name)
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=idx,
                    id=slot.id,
                    type="function",
                    function=DeltaFunctionCall(
                        name=slot.name,
                        arguments=slot.args,
                    ),
                )
            ]
        )

    def _after_finish(self, delta: DeltaMessage | None) -> DeltaMessage | None:
        invalid_slot = self._finish_invalid_slot
        invalid_prefix = self._finish_invalid_prefix
        self._finish_invalid_slot = None
        self._finish_invalid_prefix = None
        self._pending_stray_tool_framing = None

        if invalid_slot is not None:
            if not self._delta_has_tool_index(delta, invalid_slot):
                delta = self._merge_deltas(
                    delta,
                    self._make_invalid_tool_delta(invalid_slot),
                )
            return delta

        if invalid_prefix is None:
            return delta

        idx = len(self._tool_slots)
        self._ensure_slot(idx)
        self._mark_invalid_tool_slot(
            idx,
            "The model ended after an incomplete tool-call prefix; "
            "retry the intended call.",
        )
        if delta is not None and delta.content:
            escaped_prefix = _escape_unparsed_dsml(invalid_prefix)
            content = delta.content
            if content.endswith(escaped_prefix):
                content = content[: -len(escaped_prefix)]
            elif content.endswith(invalid_prefix):
                content = content[: -len(invalid_prefix)]
            delta.content = content or None
        return self._merge_deltas(delta, self._make_invalid_tool_delta(idx))

    def extract_reasoning(
        self,
        model_output: str,
        request,
    ) -> tuple[str | None, str | None]:
        reasoning, content = super().extract_reasoning(model_output, request)
        if not self.skip_tool_parsing:
            return reasoning, content

        starts_in_reasoning = (
            self.parser_engine_config.initial_state == ParserState.REASONING
        )
        exact_content = _verbatim_content_after_reasoning(
            model_output, starts_in_reasoning
        )
        if exact_content is None:
            return reasoning, content
        return reasoning, exact_content or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
    ) -> DeltaMessage | None:
        if self.skip_tool_parsing and self.reasoning_ended:
            if not delta_text:
                return None
            from vllm.entrypoints.openai.engine.protocol import DeltaMessage

            return DeltaMessage(content=delta_text)

        delta = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        if not self.skip_tool_parsing or not self.reasoning_ended:
            return delta

        end_idx = delta_text.find(DSML_THINK_END)
        if end_idx < 0:
            return delta
        exact_content = delta_text[end_idx + len(DSML_THINK_END) :]
        if delta is None and exact_content:
            from vllm.entrypoints.openai.engine.protocol import DeltaMessage

            return DeltaMessage(content=exact_content)
        if delta is not None:
            delta.content = exact_content or None
        return delta

    def _events_to_delta(
        self,
        events: list[SemanticEvent],
        finished: bool = False,
    ) -> DeltaMessage | None:
        explicit_close_order: list[int] = []
        for event in events:
            if (
                event.type == EventType.TOOL_CALL_END
                and event.tool_index not in explicit_close_order
            ):
                explicit_close_order.append(event.tool_index)
        explicit_closes = set(explicit_close_order)
        delta = super()._events_to_delta(events, finished=finished)
        if self.skip_tool_parsing:
            return delta

        if explicit_closes or (delta is not None and delta.tool_calls):
            released: list[DeltaToolCall] = []
            current_tool_calls = delta.tool_calls if delta is not None else []
            current_by_idx: dict[int, list[DeltaToolCall]] = {}
            for tool_call in current_tool_calls:
                current_by_idx.setdefault(tool_call.index, []).append(tool_call)

            # Release every closed slot in event order. Current deltas are
            # grouped first so a close-only slot cannot be overtaken by a
            # later sibling whose final fragment arrived in this batch.
            for idx in explicit_close_order:
                if idx < 0 or idx >= len(self._tool_slots):
                    continue
                slot = self._tool_slots[idx]
                current = current_by_idx.pop(idx, [])
                if slot.invalid:
                    self._tentative_tool_deltas.pop(idx, None)
                else:
                    released.extend(self._tentative_tool_deltas.pop(idx, []))
                if current:
                    released.extend(current)
                elif slot.invalid:
                    released.extend(self._make_invalid_tool_delta(idx).tool_calls)

            # Non-closed current deltas are still tentative, except for an
            # invalid auto-close which is already safe to expose.
            for tool_call in current_tool_calls:
                if tool_call.index not in current_by_idx:
                    continue
                idx = tool_call.index
                current_by_idx[idx].remove(tool_call)
                if not current_by_idx[idx]:
                    del current_by_idx[idx]
                slot = self._tool_slots[idx]
                if slot.invalid:
                    self._tentative_tool_deltas.pop(idx, None)
                    released.append(tool_call)
                else:
                    self._tentative_tool_deltas.setdefault(idx, []).append(tool_call)

            if delta is None and released:
                from vllm.entrypoints.openai.engine.protocol import DeltaMessage

                delta = DeltaMessage(tool_calls=released)
            elif delta is not None:
                delta.tool_calls = released
                if (
                    not delta.tool_calls
                    and delta.content is None
                    and delta.reasoning is None
                ):
                    delta = None

        has_tool_calls = bool(delta is not None and delta.tool_calls)
        content = delta.content if delta is not None else None
        self._pending_stray_tool_framing, content = _resolve_stray_tool_framing(
            self._pending_stray_tool_framing,
            content,
            has_tool_calls,
            finished,
        )
        if delta is None and content is not None:
            from vllm.entrypoints.openai.engine.protocol import DeltaMessage

            delta = DeltaMessage(content=content)
        elif delta is not None:
            delta.content = content

        if delta is not None and delta.content:
            if delta.tool_calls:
                delta.content = _strip_parsed_tool_residue(delta.content) or None
            if delta.content and _DSML in delta.content:
                delta.content = _escape_unparsed_dsml(delta.content)
        return delta

    def _repair_malformed_tool_slot(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._tool_slots):
            return
        slot = self._tool_slots[idx]
        if (
            slot.invalid
            or slot.name_sent
            or not slot.name
            or find_tool_name(self._tools, slot.name)
        ):
            return

        candidate, swallowed = _malformed_name_candidate(slot.name)
        if not candidate or not find_tool_name(self._tools, candidate):
            return

        # The lexer ended TOOL_NAME at the first later `">`, usually the
        # parameter opener's terminator. Restore those consumed bytes before
        # rebuilding the argument envelope.
        combined_args = swallowed + DSML_INVOKE_NAME_END + slot.args
        param_start = combined_args.find(DSML_PARAM_START)
        if param_start >= 0:
            slot._args_parts[:] = [combined_args[param_start:]]
            slot._args_joined = None
        slot.name = candidate

    def _handle_tool_end(
        self,
        event: SemanticEvent,
        deltas: list[DeltaToolCall],
    ) -> None:
        self._repair_malformed_tool_slot(event.tool_index)
        idx = event.tool_index
        if (
            self._tools
            and idx < len(self._tool_slots)
            and not self._tool_slots[idx].invalid
            and self._tool_slots[idx].name
            and not find_tool_name(self._tools, self._tool_slots[idx].name)
        ):
            self._mark_invalid_tool_slot(
                idx,
                "The model attempted an unknown tool; retry the intended call.",
            )
        super()._handle_tool_end(event, deltas)

    def _convert_args_for_slot(self, idx: int, partial: bool) -> str:
        slot = self._tool_slots[idx]
        if slot.invalid:
            return slot.args or "{}"
        return self._convert_args(slot.args, partial, slot.name)

    def _convert_args(
        self,
        raw_args: str,
        partial: bool,
        func_name: str | None = None,
    ) -> str:
        result = _dsml_arg_converter(raw_args, partial)
        if not self._tools:
            return result
        if func_name is None:
            func_name = next(
                (slot.name for slot in self._tool_slots if slot.args == raw_args),
                None,
            )
        return _unwrap_wrapper_args(result, self._tools, func_name)
