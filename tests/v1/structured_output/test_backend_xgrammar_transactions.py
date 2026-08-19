# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transaction and terminal-boundary tests for XGrammar token batches."""

from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar


class _TerminalMatcher:
    """Minimal matcher for the incident-shaped terminal sequence [223, 5]."""

    def __init__(self) -> None:
        self.accepted: list[int] = []
        self.forks: list[_TerminalMatcher] = []
        self.terminated = False

    def accept_token(self, token: int) -> bool:
        assert not self.terminated, "suffix reached a terminated matcher"
        self.accepted.append(token)
        if token == 223:
            self.terminated = True
            return True
        return False

    def is_terminated(self) -> bool:
        return self.terminated

    def fork(self) -> "_TerminalMatcher":
        probe = _TerminalMatcher()
        probe.accepted = list(self.accepted)
        probe.terminated = self.terminated
        self.forks.append(probe)
        return probe

    def reset(self) -> None:
        self.accepted.clear()
        self.terminated = False

    def rollback(self, num_tokens: int) -> None:
        del self.accepted[-num_tokens:]
        self.terminated = False


def _grammar(matcher: _TerminalMatcher) -> XgrammarGrammar:
    return XgrammarGrammar(vocab_size=1024, matcher=matcher, ctx=object())  # type: ignore[arg-type]


def test_accept_tokens_ignores_incident_suffix_after_termination():
    matcher = _TerminalMatcher()
    grammar = _grammar(matcher)

    assert grammar.accept_tokens("incident", [223, 5])
    assert matcher.accepted == [223]
    assert grammar.is_terminated()
    assert grammar.num_processed_tokens == 1

    assert grammar.accept_tokens("incident", [5])
    assert matcher.accepted == [223]
    assert grammar.num_processed_tokens == 1

    grammar.reset()
    assert matcher.accepted == []
    assert not grammar.is_terminated()
    assert grammar.num_processed_tokens == 0


def test_validate_tokens_probes_fork_without_advancing_live_matcher():
    matcher = _TerminalMatcher()
    grammar = _grammar(matcher)

    assert grammar.validate_tokens([223, 5]) == [223]
    assert matcher.accepted == []
    assert not matcher.is_terminated()
    assert grammar.num_processed_tokens == 0
    assert len(matcher.forks) == 1
    assert matcher.forks[0].accepted == [223]
    assert matcher.forks[0].is_terminated()
