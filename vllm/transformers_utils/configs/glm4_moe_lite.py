# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any, Optional, Union

from transformers.configuration_utils import PretrainedConfig


class Glm4MoeLiteConfig(PretrainedConfig):
    """Minimal Transformers config shim for `model_type=glm4_moe_lite`.

    Why this exists:
    - Transformers 4.x does not ship a `Glm4MoeLiteConfig`, so `AutoConfig`
      cannot parse the model.
    - vLLM can register custom configs via `_CONFIG_REGISTRY` and keep the
      serving runtime pinned while still supporting new model types.

    This config is intentionally permissive: we strongly type only the fields
    vLLM/TT consume, and pass through any unknown keys via `**kwargs` so that
    newer checkpoints don't break parsing.
    """

    model_type = "glm4_moe_lite"

    def __init__(
        self,
        architectures: Optional[list[str]] = None,
        vocab_size: int = 154880,
        hidden_size: int = 2048,
        intermediate_size: int = 10240,
        moe_intermediate_size: int = 1536,
        num_hidden_layers: int = 47,
        num_attention_heads: int = 20,
        num_key_value_heads: int = 20,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        max_position_embeddings: int = 202752,
        rope_theta: float = 1_000_000.0,
        rope_scaling: Optional[dict[str, Any]] = None,
        partial_rotary_factor: float = 1.0,
        q_lora_rank: int = 768,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 192,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        n_routed_experts: int = 64,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 4,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.8,
        n_group: int = 1,
        topk_group: int = 1,
        topk_method: str = "noaux_tc",
        first_k_dense_replace: int = 1,
        tie_word_embeddings: bool = False,
        dtype: str = "bfloat16",
        pad_token_id: Optional[int] = 154820,
        eos_token_id: Optional[Union[int, list[int]]] = None,
        bos_token_id: Optional[int] = None,
        **kwargs,
    ) -> None:
        if architectures is None:
            architectures = ["Glm4MoeLiteForCausalLM"]

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps

        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias

        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        # Derived convenience field used by some downstream code.
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.topk_method = topk_method
        self.first_k_dense_replace = first_k_dense_replace

        self.tie_word_embeddings = tie_word_embeddings
        self.dtype = dtype

        # If config.json omits eos_token_id, keep it None and let vLLM use the
        # tokenizer eos token. If it's present and is a list, vLLM will treat
        # it as additional stop token IDs.
        if eos_token_id is not None:
            self.eos_token_id = eos_token_id

        super().__init__(
            architectures=architectures,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        # PretrainedConfig may set architectures from kwargs; ensure it's never
        # None for vLLM registry inspection.
        self.architectures = architectures
