# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.models.deepseek_v4.nvidia.dspark import DSparkDeepseekV4ForCausalLM
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4ForCausalLM
from vllm.models.deepseek_v4.nvidia.mtp import DeepSeekV4MTP


def test_deepseek_v4_post_load_hook_finalizes_b12x_target_weights():
    calls: list[str] = []
    model = SimpleNamespace(
        model=SimpleNamespace(
            finalize_mega_moe_weights=lambda: calls.append("mega_moe"),
            finalize_mhc_broadcast_weights=lambda: calls.append("mhc_broadcast"),
            setup_b12x_wo_projection=lambda: calls.append("wo_projection"),
        )
    )

    DeepseekV4ForCausalLM.process_weights_after_loading(model)

    assert calls == ["mega_moe", "mhc_broadcast", "wo_projection"]


def test_deepseek_v4_post_load_hooks_cover_mtp_and_dspark():
    calls: list[str] = []
    mtp = SimpleNamespace(finalize_mega_moe_weights=lambda: calls.append("mtp"))
    dspark = SimpleNamespace(
        _finalize_moe=lambda: calls.append("dspark_moe"),
        model=SimpleNamespace(finalize_mhc_weights=lambda: calls.append("dspark_mhc")),
    )

    DeepSeekV4MTP.process_weights_after_loading(mtp)
    DSparkDeepseekV4ForCausalLM.process_weights_after_loading(dspark)

    assert calls == ["mtp", "dspark_moe", "dspark_mhc"]
