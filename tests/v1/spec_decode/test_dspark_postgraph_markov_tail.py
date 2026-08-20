# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.dspark import speculator as speculator_module
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


@pytest.fixture(autouse=True)
def reset_proposal_trace_latch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(speculator_module, "_DSPARK_PROPOSAL_TRACE_EMITTED", False)


def _make_speculator() -> SimpleNamespace:
    head_hidden = torch.randn(8, 16)
    model = SimpleNamespace(
        compute_local_draft_logits=Mock(side_effect=lambda hidden: hidden[:, :11] + 1)
    )
    return SimpleNamespace(
        _run_model=Mock(return_value=head_hidden),
        model=model,
        _markov_outside_cudagraph=True,
        _ensure_captured_markov_buffers=Mock(),
        _captured_markov_hidden=torch.empty(7, 16),
        _captured_base_logits=torch.empty(7, 11),
        sample_indices=torch.arange(7),
        num_query_per_req=7,
        _speculative_steps_for_query_len=lambda query_len: query_len,
        _sample_sequential=Mock(),
        capacity_activation_batch_size=0,
    )


def test_capture_records_backbone_output_without_markov_collectives(monkeypatch):
    speculator = _make_speculator()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    DSparkSpeculator._generate_draft(
        speculator,
        num_reqs=1,
        num_tokens_padded=8,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
    )

    speculator._sample_sequential.assert_not_called()
    speculator._ensure_captured_markov_buffers.assert_called_once_with()
    torch.testing.assert_close(
        speculator._captured_markov_hidden,
        speculator._run_model.return_value[:7],
    )
    torch.testing.assert_close(
        speculator._captured_base_logits,
        speculator._run_model.return_value[:7, :11] + 1,
    )


def test_capture_warmup_does_not_launch_markov_collectives(monkeypatch):
    speculator = _make_speculator()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    DSparkSpeculator._generate_draft(
        speculator,
        num_reqs=1,
        num_tokens_padded=8,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        capture_only=True,
    )

    speculator._sample_sequential.assert_not_called()
    speculator._ensure_captured_markov_buffers.assert_called_once_with()
    torch.testing.assert_close(
        speculator._captured_markov_hidden,
        speculator._run_model.return_value[:7],
    )
    torch.testing.assert_close(
        speculator._captured_base_logits,
        speculator._run_model.return_value[:7, :11] + 1,
    )


def test_eager_generation_samples_complete_markov_tail(monkeypatch):
    speculator = _make_speculator()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    DSparkSpeculator._generate_draft(
        speculator,
        num_reqs=1,
        num_tokens_padded=8,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    speculator._sample_sequential.assert_called_once_with(
        1,
        speculator._run_model.return_value,
        7,
        7,
        is_profile=False,
        use_capacity=True,
    )
    speculator._ensure_captured_markov_buffers.assert_not_called()


def test_graph_replay_finishes_markov_tail_from_stable_buffers():
    speculator = _make_speculator()

    DSparkSpeculator._finish_captured_draft(
        speculator,
        num_reqs=1,
        num_tokens_padded=8,
        num_query_per_req=7,
        is_profile=False,
    )

    args = speculator._sample_sequential.call_args
    assert args.args[:4] == (1, None, 7, 7)
    assert args.kwargs["is_profile"] is False
    assert args.kwargs["use_capacity"] is True
    assert (
        args.kwargs["prepared_sample_hidden"].untyped_storage().data_ptr()
        == speculator._captured_markov_hidden.untyped_storage().data_ptr()
    )
    assert (
        args.kwargs["precomputed_base_logits"].untyped_storage().data_ptr()
        == speculator._captured_base_logits.untyped_storage().data_ptr()
    )


def _trace_sequential_speculator(
    sampled_ids: list[int], trace_enabled: bool = True
) -> SimpleNamespace:
    model = SimpleNamespace(
        compute_draft_logits=lambda hidden: torch.zeros((hidden.shape[0], 12)),
        markov_embed=lambda tokens: torch.zeros((tokens.shape[0], 2)),
        markov_bias=lambda embed: torch.zeros((embed.shape[0], 12)),
    )
    speculator = SimpleNamespace(
        sample_indices=torch.arange(5),
        model=model,
        _use_local_draft_argmax=False,
        draft_logits=None,
        _draft_topk=None,
        sample_idx_mapping=torch.arange(5),
        sample_pos=torch.arange(5),
        draft_token_confidence_logits=torch.empty((1, 5)),
        min_survival_probability=0.0,
        use_draft_token_capacity=False,
        input_buffers=SimpleNamespace(input_ids=torch.zeros(5, dtype=torch.int64)),
        device=torch.device("cpu"),
        vocab_size=12,
        draft_token_valid_lengths=torch.empty((1,), dtype=torch.int32),
        draft_tokens=torch.empty((1, 5), dtype=torch.int64),
        draft_token_capacity=torch.empty((1,), dtype=torch.int32),
        _proposal_trace_enabled=trace_enabled,
        _sample_logits=Mock(
            side_effect=[torch.tensor([sampled_id]) for sampled_id in sampled_ids]
        ),
    )
    if trace_enabled:
        speculator._proposal_trace_raw_ids = torch.empty(5, dtype=torch.int64)
        speculator._proposal_trace_prefix_bits = torch.empty(5, dtype=torch.int32)
    return speculator


def test_trace_keeps_invalid_raw_ids_and_masks_after_first_invalid_token():
    speculator = _trace_sequential_speculator([11, -1, 13, 14, 15])

    DSparkSpeculator._sample_sequential(speculator, 1, torch.randn(5, 2), 5, 5)

    assert speculator._proposal_trace_raw_ids.tolist() == [11, -1, 13, 14, 15]
    assert speculator._proposal_trace_prefix_bits.tolist() == [1, 0, 0, 0, 0]
    assert speculator.draft_tokens.tolist() == [[11, 0, 0, 0, 0]]
    assert speculator.draft_token_valid_lengths.tolist() == [1]
    assert speculator.draft_token_capacity.tolist() == [1]


def test_trace_keeps_all_valid_k5_tokens_and_capacity():
    speculator = _trace_sequential_speculator([11, 10, 9, 8, 7])

    DSparkSpeculator._sample_sequential(speculator, 1, torch.randn(5, 2), 5, 5)

    assert speculator._proposal_trace_raw_ids.tolist() == [11, 10, 9, 8, 7]
    assert speculator._proposal_trace_prefix_bits.tolist() == [1, 1, 1, 1, 1]
    assert speculator.draft_tokens.tolist() == [[11, 10, 9, 8, 7]]
    assert speculator.draft_token_valid_lengths.tolist() == [5]
    assert speculator.draft_token_capacity.tolist() == [5]


@pytest.mark.parametrize(
    ("kwargs", "unaligned"),
    [
        ({"is_profile": True}, False),
        ({"dummy_run": True}, False),
        ({}, True),
        ({"num_speculative_tokens": 4}, False),
    ],
)
def test_trace_filters_nonproduction_unusable_proposals(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    unaligned: bool,
):
    events: list[str] = []
    speculator = DSparkSpeculator.__new__(DSparkSpeculator)
    speculator.use_draft_token_capacity = False
    speculator._proposal_trace_enabled = True
    speculator._proposal_trace_pending = False
    speculator._proposal_trace_steps = 5
    speculator.num_speculative_steps = 5
    speculator.dynamic_physical_depth = True
    speculator._has_unaligned_cached_prefix = Mock(return_value=unaligned)
    speculator._emit_proposal_trace = Mock(side_effect=lambda: events.append("emit"))
    input_batch = SimpleNamespace(num_reqs=1)
    monkeypatch.setattr(
        speculator_module.DFlashSpeculator,
        "propose",
        lambda self, *_args, **_kwargs: events.append("super") or "draft",
    )

    result = DSparkSpeculator.propose(speculator, input_batch, **kwargs)

    assert result == "draft"
    assert events == ["super"]
    assert not speculator._proposal_trace_pending
    assert not speculator_module._DSPARK_PROPOSAL_TRACE_EMITTED
    if kwargs.get("num_speculative_tokens") == 4:
        assert speculator._last_num_speculative_steps == 4


def _trace_proposal_instance() -> DSparkSpeculator:
    speculator = DSparkSpeculator.__new__(DSparkSpeculator)
    speculator.use_draft_token_capacity = False
    speculator._proposal_trace_enabled = True
    speculator._proposal_trace_pending = False
    speculator._proposal_trace_steps = 5
    speculator.num_speculative_steps = 5
    speculator.dynamic_physical_depth = False
    speculator._has_unaligned_cached_prefix = Mock(return_value=False)
    speculator._markov_outside_cudagraph = False
    speculator._capture_sharded_markov = False
    speculator._use_local_draft_argmax = False
    speculator._draft_topk = None
    speculator._proposal_trace_raw_ids = torch.tensor([11, 10, 9, 8, 7])
    speculator._proposal_trace_prefix_bits = torch.ones(5, dtype=torch.int32)
    speculator.draft_tokens = torch.tensor([[11, 10, 9, 8, 7]])
    speculator.draft_token_valid_lengths = torch.tensor([5], dtype=torch.int32)
    speculator.draft_token_capacity = torch.tensor([5], dtype=torch.int32)
    return speculator


def test_fullgraph_post_return_trace_emits_once(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    first_events: list[str] = []
    second_events: list[str] = []
    first = _trace_proposal_instance()
    second = _trace_proposal_instance()
    payloads: list[str] = []

    def replay(self, *_args, **_kwargs):
        events = first_events if self is first else second_events
        events.append("replay")
        return "draft"

    monkeypatch.setattr(speculator_module.DFlashSpeculator, "propose", replay)

    def emit(_message, payload):
        first_events.append("emit")
        payloads.append(payload)

    monkeypatch.setattr(speculator_module.logger, "info", emit)

    assert DSparkSpeculator.propose(first, SimpleNamespace(num_reqs=1)) == "draft"
    assert first_events == ["replay", "emit"]
    assert DSparkSpeculator.propose(second, SimpleNamespace(num_reqs=1)) == "draft"
    assert second_events == ["replay"]
    assert len(payloads) == 1
    assert payloads[0] == json.dumps(
        {
            "rank": 0,
            "K": 5,
            "markov_outside_cudagraph": 0,
            "capture_sharded_markov": 0,
            "local_draft_argmax": 0,
            "draft_topk": -1,
            "raw_ids": [11, 10, 9, 8, 7],
            "prefix_bits": [1, 1, 1, 1, 1],
            "masked_ids": [11, 10, 9, 8, 7],
            "valid_length": 5,
            "capacity": 5,
        },
        separators=(",", ":"),
    )
    assert not second._proposal_trace_pending


def test_trace_does_not_host_read_during_capture(monkeypatch: pytest.MonkeyPatch):
    class HostReadGuard:
        def __getitem__(self, _key):
            return self

        def detach(self):
            return self

        def cpu(self):
            raise AssertionError("trace attempted D2H during capture")

        def item(self):
            raise AssertionError("trace attempted scalar D2H during capture")

        def tolist(self):
            raise AssertionError("trace attempted list D2H during capture")

    speculator = _trace_proposal_instance()
    speculator._proposal_trace_pending = True
    speculator._proposal_trace_raw_ids = HostReadGuard()
    speculator._proposal_trace_prefix_bits = HostReadGuard()
    speculator.draft_tokens = HostReadGuard()
    speculator.draft_token_valid_lengths = HostReadGuard()
    speculator.draft_token_capacity = HostReadGuard()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    DSparkSpeculator._emit_proposal_trace(speculator)


def test_disabled_trace_uses_no_trace_buffers():
    speculator = _trace_sequential_speculator([1, 2, 3, 4, 5], trace_enabled=False)

    DSparkSpeculator._sample_sequential(speculator, 1, torch.randn(5, 2), 5, 5)

    assert not hasattr(speculator, "_proposal_trace_raw_ids")
    assert not hasattr(speculator, "_proposal_trace_prefix_bits")
