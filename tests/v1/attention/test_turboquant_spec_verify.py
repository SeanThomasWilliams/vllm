# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.v1.attention.backends import turboquant_attn as tq_attn
from vllm.v1.attention.backends.turboquant_attn import (
    TurboQuantAttentionImpl,
    TurboQuantMetadata,
    TurboQuantMetadataBuilder,
)


def _make_impl() -> TurboQuantAttentionImpl:
    impl = TurboQuantAttentionImpl.__new__(TurboQuantAttentionImpl)
    impl.num_heads = 2
    impl.head_size = 4
    impl.scale = 1.0
    impl.num_kv_heads = 2
    impl.tq_config = SimpleNamespace(
        key_mse_bits=8,
        key_packed_size=4,
        effective_value_quant_bits=4,
        key_fp8=False,
        norm_correction=False,
        centroid_bits=4,
    )
    impl.max_num_kv_splits = 32
    return impl


def _make_layer(head_size: int) -> SimpleNamespace:
    eye = torch.eye(head_size, dtype=torch.float32)
    return SimpleNamespace(
        _tq_Pi=eye,
        _tq_PiT=eye,
        _tq_centroids=torch.zeros(1, dtype=torch.float32),
        _tq_mid_o_buf=torch.full((2, 2, 32, head_size + 1), 11.0),
        _tq_output_buf=torch.full((2, 2, head_size), 22.0),
        _tq_lse_buf=torch.full((2, 2), 33.0),
    )


def _make_metadata(*, max_query_len: int, max_seq_len: int) -> TurboQuantMetadata:
    seq_lens = torch.tensor([100, 200], dtype=torch.int32)
    block_table = torch.tensor(
        [
            [0, 1, 2, 3],
            [10, 11, 12, 13],
        ],
        dtype=torch.int32,
    )
    return TurboQuantMetadata(
        seq_lens=seq_lens,
        slot_mapping=torch.arange(8, dtype=torch.int32),
        block_table=block_table,
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        num_actual_tokens=8,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        is_prefill=True,
        num_decodes=0,
        num_decode_tokens=0,
    )


def test_spec_verify_routes_through_decode_kernel(monkeypatch):
    impl = _make_impl()
    layer = _make_layer(impl.head_size)
    metadata = _make_metadata(max_query_len=4, max_seq_len=128)

    query = torch.arange(8 * impl.num_heads * impl.head_size, dtype=torch.float32)
    query = query.view(8, impl.num_heads * impl.head_size)
    key = query.clone()
    value = query.clone()
    kv_cache = torch.empty(1, 1, impl.num_kv_heads, impl.head_size)

    called = {}

    def fake_decode_attention(self, q, passed_kv_cache, passed_metadata, *args):
        called["query"] = q
        called["kv_cache"] = passed_kv_cache
        called["metadata"] = passed_metadata
        return torch.full((8, impl.num_heads, impl.head_size), 7.0)

    monkeypatch.setattr(TurboQuantAttentionImpl, "_ensure_on_device", lambda *_: None)
    monkeypatch.setattr(
        TurboQuantAttentionImpl,
        "_decode_attention",
        fake_decode_attention,
    )

    out = impl.forward(layer, query, key, value, kv_cache, metadata)

    assert out.shape == (8, impl.num_heads * impl.head_size)
    assert torch.all(out == 7.0)
    synth_metadata = called["metadata"]
    assert torch.equal(
        synth_metadata.seq_lens,
        torch.tensor([97, 98, 99, 100, 197, 198, 199, 200], dtype=torch.int32),
    )
    assert torch.equal(
        synth_metadata.block_table,
        torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [10, 11, 12, 13],
                [10, 11, 12, 13],
                [10, 11, 12, 13],
                [10, 11, 12, 13],
            ],
            dtype=torch.int32,
        ),
    )
    assert synth_metadata.is_prefill is False
    assert called["kv_cache"] is kv_cache
    assert called["query"].shape == (8, impl.num_heads, impl.head_size)


def test_spec_verify_falls_back_when_not_eligible(monkeypatch):
    impl = _make_impl()
    layer = _make_layer(impl.head_size)
    metadata = _make_metadata(max_query_len=1, max_seq_len=1)

    query = torch.zeros(8, impl.num_heads * impl.head_size)
    key = query.clone()
    value = query.clone()
    kv_cache = torch.empty(1, 1, impl.num_kv_heads, impl.head_size)

    decode_called = False
    prefill_called = False

    def fake_decode_attention(**kwargs):
        nonlocal decode_called
        decode_called = True
        raise AssertionError("spec-verify path should not be used")

    def fake_prefill_attention(self, q, *args, **kwargs):
        nonlocal prefill_called
        prefill_called = True
        return torch.full((q.shape[0], impl.num_heads, impl.head_size), 3.0)

    monkeypatch.setattr(TurboQuantAttentionImpl, "_ensure_on_device", lambda *_: None)
    monkeypatch.setattr(
        tq_attn,
        "triton_turboquant_decode_attention",
        fake_decode_attention,
    )
    monkeypatch.setattr(
        TurboQuantAttentionImpl,
        "_prefill_attention",
        fake_prefill_attention,
    )

    out = impl.forward(layer, query, key, value, kv_cache, metadata)

    assert prefill_called
    assert not decode_called
    assert out.shape == (8, impl.num_heads * impl.head_size)
    assert torch.all(out == 3.0)


def test_metadata_builder_uses_prefill_cpu_max(monkeypatch):
    builder = TurboQuantMetadataBuilder.__new__(TurboQuantMetadataBuilder)
    builder.reorder_batch_threshold = 1

    cam = SimpleNamespace(
        seq_lens=torch.tensor([100, 4, 7], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([100, 4, 7], dtype=torch.int32),
        slot_mapping=torch.arange(6, dtype=torch.int32),
        block_table_tensor=torch.tensor(
            [[0, 1], [10, 11], [20, 21]], dtype=torch.int32
        ),
        query_start_loc=torch.tensor([0, 1, 3, 6], dtype=torch.int32),
        num_actual_tokens=6,
        max_query_len=3,
        max_seq_len=100,
        num_reqs=3,
    )

    monkeypatch.setattr(
        tq_attn,
        "split_decodes_and_prefills",
        lambda *args, **kwargs: (1, 2, 1, None),
    )

    metadata = builder.build(0, cam)

    assert metadata.prefill_max_seq_cpu == 7


def test_prefill_attention_capture_guard_avoids_tolist(monkeypatch):
    impl = _make_impl()
    layer = _make_layer(impl.head_size)
    metadata = _make_metadata(max_query_len=1, max_seq_len=4)

    query = torch.arange(4 * impl.num_heads * impl.head_size, dtype=torch.float32)
    query = query.view(4, impl.num_heads, impl.head_size)
    key = query.clone()
    value = query.clone()
    kv_cache = torch.empty(1, 1, impl.num_kv_heads, impl.head_size)

    monkeypatch.setattr(tq_attn, "_HAS_FLASH_ATTN", True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        torch.Tensor,
        "tolist",
        lambda self: (_ for _ in ()).throw(AssertionError("tolist should not run")),
    )

    flash_called = {}

    def fake_flash_attn(self, **kwargs):
        flash_called.update(kwargs)
        return torch.full((4, impl.num_heads, impl.head_size), 9.0)

    monkeypatch.setattr(TurboQuantAttentionImpl, "_flash_attn_varlen", fake_flash_attn)

    out = impl._prefill_attention(
        query,
        key,
        value,
        kv_cache,
        metadata,
        layer._tq_Pi,
        layer._tq_centroids,
        layer._tq_PiT,
        layer=layer,
    )

    assert torch.all(out == 9.0)
    assert flash_called["cu_seqlens_q"] is metadata.query_start_loc
    assert flash_called["cu_seqlens_k"] is metadata.query_start_loc
