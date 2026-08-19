# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from vllm.v1.worker.gpu.structured_outputs import _build_grammar_mapping


def test_grammar_mapping_uses_scheduled_draft_rows_when_budget_is_zero():
    mapping = _build_grammar_mapping(
        ["low", "high", "prefill"],
        ["low", "high", "prefill"],
        np.array([0, 1, 2, 3], dtype=np.int32),
        np.array([2, 2, 0], dtype=np.int32),
        num_bonus_tokens=1,
        mask_stride=3,
    )

    assert mapping == [0, 1, 2, 3, 4, 5, 6]


def test_grammar_mapping_falls_back_to_compacted_offsets_without_speculation():
    mapping = _build_grammar_mapping(
        ["first", "second"],
        ["second", "first"],
        np.array([0, 2, 3], dtype=np.int32),
        None,
        num_bonus_tokens=1,
        mask_stride=3,
    )

    assert mapping == [3, 0, 1]
