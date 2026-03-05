# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.config import ModelConfig
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.glm4_moe_lite import Glm4MoeLiteConfig


def _glm4_moe_lite_min_config() -> dict[str, Any]:
    # Minimal config sufficient to validate:
    # - vLLM `_CONFIG_REGISTRY` loading under Transformers 4.x
    # - MLA head_size / num_kv_heads calculations via ModelConfig logic
    return {
        "model_type": "glm4_moe_lite",
        "architectures": ["Glm4MoeLiteForCausalLM"],
        "vocab_size": 154880,
        "hidden_size": 2048,
        "intermediate_size": 10240,
        "moe_intermediate_size": 1536,
        "num_hidden_layers": 47,
        "num_attention_heads": 20,
        "num_key_value_heads": 20,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "attention_dropout": 0.0,
        "attention_bias": False,
        "max_position_embeddings": 202752,
        "rope_theta": 1_000_000.0,
        "rope_scaling": None,
        "partial_rotary_factor": 1.0,
        "q_lora_rank": 768,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "n_routed_experts": 64,
        "n_shared_experts": 1,
        "num_experts_per_tok": 4,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.8,
        "n_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "first_k_dense_replace": 1,
        "tie_word_embeddings": False,
        "dtype": "bfloat16",
        "pad_token_id": 154820,
        # The checkpoint uses a list; vLLM supports list-of-eos semantics.
        "eos_token_id": [154820, 154827, 154829],
    }


class _DummyModelConfig:
    """A tiny harness that reuses ModelConfig descriptor logic.

    We don't want to instantiate the full pydantic ModelConfig in unit tests,
    because it triggers platform detection and may require external assets.

    Instead we attach the relevant descriptors to this class and provide only
    the minimal attributes the code paths expect.
    """

    # Reuse vLLM's implementation directly.
    is_deepseek_mla = ModelConfig.is_deepseek_mla
    use_mla = ModelConfig.use_mla
    get_head_size = ModelConfig.get_head_size
    get_total_num_kv_heads = ModelConfig.get_total_num_kv_heads
    get_num_kv_heads = ModelConfig.get_num_kv_heads

    # Required by get_head_size / get_total_num_kv_heads fast paths.
    is_attention_free = False

    def __init__(self, hf_config: Glm4MoeLiteConfig):
        self.hf_config = hf_config
        self.hf_text_config = hf_config


def test_glm4_moe_lite_config_registry_and_mla(monkeypatch: pytest.MonkeyPatch,
                                               tmp_path):
    # Write a local config.json so the test is offline and hermetic.
    cfg = _glm4_moe_lite_min_config()
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    config = get_config(str(tmp_path), trust_remote_code=False)
    assert isinstance(config, Glm4MoeLiteConfig)

    # Derived field computed by the shim.
    assert config.qk_head_dim == 256

    dummy = _DummyModelConfig(config)
    parallel = SimpleNamespace(tensor_parallel_size=1)

    # Default path: MLA enabled -> KVPE head_size=576, num_kv_heads=1.
    monkeypatch.delenv("VLLM_MLA_DISABLE", raising=False)
    assert dummy.is_deepseek_mla is True
    assert dummy.use_mla is True
    assert dummy.get_head_size() == 576
    assert dummy.get_num_kv_heads(parallel) == 1

    # Debug path: MLA disabled -> full K/V cache head_size=256, num_kv_heads=20.
    monkeypatch.setenv("VLLM_MLA_DISABLE", "1")
    assert dummy.use_mla is False
    assert dummy.get_head_size() == 256
    assert dummy.get_num_kv_heads(parallel) == 20

