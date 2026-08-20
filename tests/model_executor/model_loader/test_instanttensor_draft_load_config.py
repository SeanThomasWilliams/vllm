# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import LoadConfig
from vllm.model_executor.model_loader import _instanttensor_draft_load_config


def _configs(*, same_source: bool = True):
    target = SimpleNamespace(model="model", revision="revision")
    draft = SimpleNamespace(
        model="model" if same_source else "draft-model",
        revision="revision",
    )
    load_config = LoadConfig(
        load_format="instanttensor", safetensors_load_strategy="eager"
    )
    vllm_config = SimpleNamespace(
        load_config=load_config,
        model_config=target,
        speculative_config=SimpleNamespace(
            target_model_config=target,
            draft_model_config=draft,
        ),
    )
    return vllm_config, target, draft, load_config


def test_auto_keeps_instanttensor_for_target(monkeypatch):
    monkeypatch.delenv("INSTANTTENSOR_DRAFT_LOADER", raising=False)
    vllm_config, target, _, load_config = _configs()

    resolved = _instanttensor_draft_load_config(vllm_config, target, None)

    assert resolved is load_config
    assert resolved.load_format == "instanttensor"


def test_auto_uses_lazy_safetensors_for_same_source_draft(monkeypatch):
    monkeypatch.setenv("INSTANTTENSOR_DRAFT_LOADER", "auto")
    vllm_config, _, draft, load_config = _configs()

    resolved = _instanttensor_draft_load_config(vllm_config, draft, None)

    assert resolved is not load_config
    assert resolved.load_format == "safetensors"
    assert resolved.safetensors_load_strategy == "lazy"
    assert load_config.load_format == "instanttensor"
    assert load_config.safetensors_load_strategy == "eager"


def test_auto_keeps_instanttensor_for_different_source_draft(monkeypatch):
    monkeypatch.setenv("INSTANTTENSOR_DRAFT_LOADER", "auto")
    vllm_config, _, draft, load_config = _configs(same_source=False)

    resolved = _instanttensor_draft_load_config(vllm_config, draft, None)

    assert resolved is load_config
    assert resolved.load_format == "instanttensor"


def test_explicit_safetensors_forces_draft_switch(monkeypatch):
    monkeypatch.setenv("INSTANTTENSOR_DRAFT_LOADER", "safetensors")
    vllm_config, _, draft, _ = _configs(same_source=False)

    resolved = _instanttensor_draft_load_config(vllm_config, draft, None)

    assert resolved.load_format == "safetensors"
    assert resolved.safetensors_load_strategy == "lazy"


def test_explicit_instanttensor_keeps_draft_loader(monkeypatch):
    monkeypatch.setenv("INSTANTTENSOR_DRAFT_LOADER", "instanttensor")
    vllm_config, _, draft, load_config = _configs()

    resolved = _instanttensor_draft_load_config(vllm_config, draft, None)

    assert resolved is load_config
    assert resolved.load_format == "instanttensor"


def test_invalid_draft_loader_mode_fails(monkeypatch):
    monkeypatch.setenv("INSTANTTENSOR_DRAFT_LOADER", "unsafe")
    vllm_config, target, _, _ = _configs()

    with pytest.raises(
        ValueError, match="must be one of auto, safetensors, instanttensor"
    ):
        _instanttensor_draft_load_config(vllm_config, target, None)
