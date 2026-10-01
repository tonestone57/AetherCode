"""
Configuration class for AetherCode-33B Model.
Based on specification in AetherCode-Max8B-V4-Ultra.md
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class AetherCode33BConfig:
    vocab_size: int = 152000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    moe_intermediate_size: int = 1408  # Fine-grained expert intermediate size
    num_hidden_layers: int = 44
    num_dense_layers: int = 6          # Layers 1-6 Dense Anchor Base
    num_attention_heads: int = 32
    num_key_value_heads: int = 32

    # MLA Attention parameters
    kv_lora_rank: int = 512            # d_c = 512
    q_lora_rank: Optional[int] = 1536
    qk_head_dim: int = 128
    v_head_dim: int = 128
    qk_rope_head_dim: int = 64         # d_rope = 64
    attention_logit_softcapping: float = 50.0

    # MoE Routing parameters
    num_routed_experts: int = 72       # 72 Fine-Grained Experts
    num_active_experts: int = 12       # Top-12 Active Routed
    num_shared_experts: int = 2        # 2 Universal Shared Experts
    routed_scaling_factor: float = 1.0
    router_aux_loss_coef: float = 0.0  # Auxiliary-Loss-Free

    # Context & RoPE Parameters (YaRN)
    max_position_embeddings: int = 131072  # 128K context window
    rope_theta: float = 10000.0
    rope_scaling: Dict[str, Any] = field(default_factory=lambda: {
        "type": "yarn",
        "factor": 32.0,
        "original_max_position_embeddings": 4096,
        "ext_factor": 1.0,
        "attn_factor": 1.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
    })

    # Untied Factorized Head Parameters
    factorized_head_rank: int = 512   # 152K x 512 x 4096

    # Quantization target specs
    quantization_type: str = "Q5_K_M"

    def __post_init__(self):
        assert self.num_hidden_layers > self.num_dense_layers, "num_hidden_layers must be greater than num_dense_layers"
