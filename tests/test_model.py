"""
Unit Tests for AetherCode-33B Model Implementation & GGUF Calibration.
"""

import pytest
import torch
from aethercode.config import AetherCode33BConfig
from aethercode.model import (
    RMSNorm,
    YaRNRotaryEmbedding,
    MultiHeadLatentAttention,
    SigmoidRouter,
    FineGrainedMoE,
    AetherCodeDecoderLayer,
    FactorizedOutputHead,
    AetherCode33BForCausalLM,
)
from aethercode.convert_gguf import get_aethercode_gguf_metadata


@pytest.fixture
def mini_config():
    return AetherCode33BConfig(
        vocab_size=500,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=4,
        num_dense_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        kv_lora_rank=32,
        q_lora_rank=64,
        qk_head_dim=16,
        v_head_dim=16,
        qk_rope_head_dim=16,
        attention_logit_softcapping=50.0,
        num_routed_experts=6,
        num_active_experts=2,
        num_shared_experts=1,
        factorized_head_rank=32,
    )


def test_config_defaults():
    config = AetherCode33BConfig()
    assert config.vocab_size == 152000
    assert config.hidden_size == 4096
    assert config.num_hidden_layers == 44
    assert config.num_dense_layers == 6
    assert config.num_routed_experts == 72
    assert config.num_active_experts == 12
    assert config.num_shared_experts == 2
    assert config.kv_lora_rank == 512
    assert config.qk_rope_head_dim == 64
    assert config.attention_logit_softcapping == 50.0


def test_rmsnorm():
    norm = RMSNorm(128)
    x = torch.randn(2, 5, 128)
    out = norm(x)
    assert out.shape == (2, 5, 128)


def test_yarn_rotary_embedding():
    rope = YaRNRotaryEmbedding(dim=16, max_position_embeddings=131072)
    x = torch.randn(2, 4, 10, 16)
    cos, sin = rope(x, seq_len=10)
    assert cos.shape == (10, 16)
    assert sin.shape == (10, 16)


def test_mla_attention(mini_config):
    mla = MultiHeadLatentAttention(mini_config)
    x = torch.randn(2, 8, mini_config.hidden_size)
    out = mla(x)
    assert out.shape == (2, 8, mini_config.hidden_size)


def test_sigmoid_router(mini_config):
    router = SigmoidRouter(mini_config)
    flat_x = torch.randn(16, mini_config.hidden_size)
    weights, indices = router(flat_x)
    assert weights.shape == (16, mini_config.num_active_experts)
    assert indices.shape == (16, mini_config.num_active_experts)
    # Check weights sum to 1.0 per token
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(16), rtol=1e-4, atol=1e-4)


def test_fine_grained_moe(mini_config):
    moe = FineGrainedMoE(mini_config)
    x = torch.randn(2, 6, mini_config.hidden_size)
    out = moe(x)
    assert out.shape == (2, 6, mini_config.hidden_size)


def test_hybrid_decoder_layers(mini_config):
    dense_layer = AetherCodeDecoderLayer(mini_config, layer_idx=0)
    moe_layer = AetherCodeDecoderLayer(mini_config, layer_idx=3)

    assert not hasattr(dense_layer.mlp, "routed_experts")
    assert hasattr(moe_layer.mlp, "routed_experts")

    x = torch.randn(2, 5, mini_config.hidden_size)
    out_dense = dense_layer(x)
    out_moe = moe_layer(x)

    assert out_dense.shape == (2, 5, mini_config.hidden_size)
    assert out_moe.shape == (2, 5, mini_config.hidden_size)


def test_factorized_output_head(mini_config):
    head = FactorizedOutputHead(mini_config)
    x = torch.randn(2, 5, mini_config.hidden_size)
    logits = head(x)
    assert logits.shape == (2, 5, mini_config.vocab_size)


def test_full_model_forward(mini_config):
    model = AetherCode33BForCausalLM(mini_config)
    input_ids = torch.randint(0, mini_config.vocab_size, (2, 7))
    logits = model(input_ids)
    assert logits.shape == (2, 7, mini_config.vocab_size)


def test_gguf_metadata_calibration():
    config = AetherCode33BConfig()
    meta = get_aethercode_gguf_metadata(config)

    assert meta["general.architecture"] == "llama"
    assert meta["general.file_type"] == 17
    assert meta["llama.block_count"] == 44
    assert meta["llama.expert_routed_count"] == 72
    assert meta["llama.expert_active_count"] == 12
    assert meta["llama.expert_shared_count"] == 2
    assert meta["llama.attention.kv_lora_rank"] == 512
    assert meta["llama.attention.logit_softcapping"] == 50.0
    assert meta["llama.rope.scaling.type"] == "yarn"
    assert meta["llama.rope.scaling.factor"] == 32.0
