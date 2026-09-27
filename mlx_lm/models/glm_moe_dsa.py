# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import BaseModelArgs
from .deepseek_v32 import Model as DSV32Model


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    index_head_dim: int
    index_n_heads: int
    index_topk: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: Optional[int]
    n_routed_experts: Optional[int]
    routed_scaling_factor: float
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    v_head_dim: int
    qk_nope_head_dim: int
    norm_topk_prob: bool
    n_group: int
    topk_group: int
    num_experts_per_tok: int
    first_k_dense_replace: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_parameters: Dict
    attention_bias: bool
    rope_scaling: Dict = None
    rope_theta: Optional[float] = None
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    moe_layer_freq: int = 1
    # Shared layers reuse the latest full layer's selected positions.
    index_topk_freq: int = 1
    indexer_types: Optional[Any] = None
    index_skip_topk_offset: int = 0
    indexer_rope_interleave: bool = True
    indexer_float32: bool = True
    router_logits_float32: bool = True
    indexer_head_tile: int = 4
    indexer_norm_eps: float = 1e-6
    num_nextn_predict_layers: int = 0

    def __post_init__(self):
        self.rope_scaling = self.rope_parameters
        self.rope_theta = self.rope_parameters["rope_theta"]
        if self.index_topk_freq <= 0 or self.indexer_head_tile <= 0:
            raise ValueError("Indexer frequency and head tile must be positive")
        if self.indexer_types is None:
            self.indexer_types = [
                "full"
                if i == 0
                or max(i - self.index_skip_topk_offset + 1, 0) % self.index_topk_freq
                == 0
                else "shared"
                for i in range(self.num_hidden_layers)
            ]


class Model(DSV32Model):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
