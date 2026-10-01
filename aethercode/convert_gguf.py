"""
GGUF Metadata Calibration & Export Utility for AetherCode-33B.
Calibrates and exports GGUF headers and model tensors matching Section 6 of AetherCode-Max8B-V4-Ultra.md.
"""

from typing import Dict, Any
import gguf
from .config import AetherCode33BConfig


def get_aethercode_gguf_metadata(config: AetherCode33BConfig) -> Dict[str, Any]:
    """
    Constructs the exact GGUF key-value metadata dict specified in the blueprint Section 6.
    """
    file_type = 17 if config.quantization_type == "Q5_K_M" else 18

    metadata = {
        "general.architecture": "llama",
        "general.quantization_version": 2,
        "general.file_type": file_type,
        "llama.block_count": config.num_hidden_layers,
        "llama.expert_routed_count": config.num_routed_experts,
        "llama.expert_active_count": config.num_active_experts,
        "llama.expert_shared_count": config.num_shared_experts,
        "llama.expert_weights_scale": config.routed_scaling_factor,
        "llama.rope.dimension_count": config.qk_rope_head_dim,
        "llama.attention.kv_lora_rank": config.kv_lora_rank,
        "llama.attention.logit_softcapping": config.attention_logit_softcapping,
        "llama.rope.scaling.type": config.rope_scaling.get("type", "yarn"),
        "llama.rope.scaling.factor": float(config.rope_scaling.get("factor", 32.0)),
        "llama.rope.scaling.orig_ctx_len": int(config.rope_scaling.get("original_max_position_embeddings", 4096)),
        "llama.rope.scaling.ext_factor": float(config.rope_scaling.get("ext_factor", 1.0)),
        "llama.rope.scaling.attn_factor": float(config.rope_scaling.get("attn_factor", 1.0)),
        "llama.rope.scaling.beta_fast": float(config.rope_scaling.get("beta_fast", 32.0)),
        "llama.rope.scaling.beta_slow": float(config.rope_scaling.get("beta_slow", 1.0)),
        "llama.context_length": config.max_position_embeddings,
        "llama.embedding_length": config.hidden_size,
        "llama.feed_forward_length": config.intermediate_size,
        "llama.head_count": config.num_attention_heads,
    }
    return metadata


def build_gguf_writer(output_path: str, config: AetherCode33BConfig) -> gguf.GGUFWriter:
    """
    Initializes a GGUFWriter with metadata calibrated according to Section 6.
    """
    writer = gguf.GGUFWriter(output_path, "llama")
    metadata = get_aethercode_gguf_metadata(config)

    for key, val in metadata.items():
        if isinstance(val, int):
            writer.add_uint32(key, val)
        elif isinstance(val, float):
            writer.add_float32(key, val)
        elif isinstance(val, str):
            writer.add_string(key, val)

    return writer


if __name__ == "__main__":
    cfg = AetherCode33BConfig()
    meta = get_aethercode_gguf_metadata(cfg)
    print("Calibrated GGUF Metadata Header Keys:")
    for k, v in meta.items():
        print(f"  {k} = {v}")
