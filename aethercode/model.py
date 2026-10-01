"""
PyTorch Implementation of AetherCode-33B Model Architecture.
Features:
- Multi-Head Latent Attention (MLA) with Soft-Capping & YaRN RoPE
- Auxiliary-Loss-Free Sigmoid Router with Dynamic Expert Bias (b_e)
- Fine-Grained MoE (72 Routed Experts + 2 Universal Shared Experts)
- Untied Low-Rank Factorized Output Head (152K x 512 x 4096)
- Hybrid Layer Architecture (Layers 1-6 Dense, Layers 7-44 MoE)
"""

import math
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AetherCode33BConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class YaRNRotaryEmbedding(nn.Module):
    """
    YaRN (Yet Another RoPE Extrapolation Patch) Rotary Position Embedding for MLA.
    """
    def __init__(self, dim: int, max_position_embeddings: int = 131072, base: float = 10000.0, rope_scaling: Optional[dict] = None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Default YaRN factors if rope_scaling is given
        if rope_scaling is not None and rope_scaling.get("type") == "yarn":
            scale_factor = rope_scaling.get("factor", 32.0)
            orig_max = rope_scaling.get("original_max_position_embeddings", 4096)
            beta_fast = rope_scaling.get("beta_fast", 32.0)
            beta_slow = rope_scaling.get("beta_slow", 1.0)
            attn_factor = rope_scaling.get("attn_factor", 1.0)
        else:
            scale_factor = 1.0
            orig_max = max_position_embeddings
            beta_fast = 32.0
            beta_slow = 1.0
            attn_factor = 1.0

        self.attn_factor = attn_factor
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))

        if scale_factor > 1.0:
            # Calibrate frequencies using YaRN interpolation/extrapolation blend
            low_freq_wavelen = orig_max / beta_slow
            high_freq_wavelen = orig_max / beta_fast
            wavelen = 2 * math.pi / inv_freq

            inv_freq_interpolated = inv_freq / scale_factor
            smooth = torch.clamp((wavelen - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen), 0.0, 1.0)
            inv_freq = (1 - smooth) * inv_freq + smooth * inv_freq_interpolated

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attn_factor
        sin = emb.sin() * self.attn_factor
        return cos, sin


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x shape: [batch, heads, seq_len, head_dim]
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    rotated_x = torch.cat((-x2, x1), dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, head_dim]
    return (x * cos) + (rotated_x * sin)


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) with Soft-Capping & YaRN RoPE support.
    """
    def __init__(self, config: AetherCode33BConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.softcapping = config.attention_logit_softcapping

        # Query projection
        if self.q_lora_rank is not None:
            self.q_down_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.q_up_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False)

        self.q_rope_proj = nn.Linear(
            self.q_lora_rank if self.q_lora_rank is not None else self.hidden_size,
            self.num_heads * self.qk_rope_head_dim,
            bias=False
        )

        # Key-Value compression and latent projection
        self.kv_down_proj = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.k_up_proj = nn.Linear(self.kv_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        self.v_up_proj = nn.Linear(self.kv_lora_rank, self.num_heads * self.v_head_dim, bias=False)
        self.k_rope_proj = nn.Linear(self.hidden_size, self.qk_rope_head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)

        # RoPE
        self.rotary_emb = YaRNRotaryEmbedding(
            dim=self.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_scaling=config.rope_scaling
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Compress and project Query
        if self.q_lora_rank is not None:
            q_latent = self.q_norm(self.q_down_proj(hidden_states))
            q_nope = self.q_up_proj(q_latent).view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
            q_pe = self.q_rope_proj(q_latent).view(batch_size, seq_len, self.num_heads, self.qk_rope_head_dim)
        else:
            q_nope = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
            q_pe = self.q_rope_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.qk_rope_head_dim)

        q_nope = q_nope.transpose(1, 2)  # [B, N_heads, L, qk_head_dim]
        q_pe = q_pe.transpose(1, 2)      # [B, N_heads, L, qk_rope_head_dim]

        # Compress Key-Value
        kv_latent = self.kv_norm(self.kv_down_proj(hidden_states))
        k_nope = self.k_up_proj(kv_latent).view(batch_size, seq_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        v = self.v_up_proj(kv_latent).view(batch_size, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2)

        # RoPE Keys
        k_pe = self.k_rope_proj(hidden_states).view(batch_size, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        k_pe = k_pe.expand(-1, self.num_heads, -1, -1)  # Expand across heads

        # Apply RoPE
        cos, sin = self.rotary_emb(q_pe, seq_len)
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin)
        k_pe = apply_rotary_pos_emb(k_pe, cos, sin)

        # Combine Non-RoPE and RoPE components for Q and K
        q = torch.cat([q_nope, q_pe], dim=-1)  # [B, N_heads, L, qk_head_dim + qk_rope_head_dim]
        k = torch.cat([k_nope, k_pe], dim=-1)  # [B, N_heads, L, qk_head_dim + qk_rope_head_dim]

        # Scaled Dot-Product Attention Scores
        scale = 1.0 / math.sqrt(self.qk_head_dim + self.qk_rope_head_dim)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B, N_heads, L, L]

        # Attention Logit Soft-Capping
        if self.softcapping is not None and self.softcapping > 0:
            attn_scores = self.softcapping * torch.tanh(attn_scores / self.softcapping)

        # Causal mask
        if attention_mask is None and seq_len > 1:
            causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device)
            causal_mask = torch.triu(causal_mask, diagonal=1)
            attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)
        elif attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(attn_scores, dim=-1)

        # Context output
        context = torch.matmul(attn_weights, v)  # [B, N_heads, L, v_head_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        output = self.o_proj(context)
        return output


class SwiGLUMLP(nn.Module):
    """
    Standard Feed-Forward SwiGLU MLP.
    """
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SigmoidRouter(nn.Module):
    """
    Auxiliary-Loss-Free Sigmoid Router with Dynamic Expert Bias (b_e).
    """
    def __init__(self, config: AetherCode33BConfig):
        super().__init__()
        self.num_experts = config.num_routed_experts
        self.top_k = config.num_active_experts
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        # Dynamic expert bias b_e
        self.b_e = nn.Parameter(torch.zeros(self.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: [B * L, H]
        logits = self.gate(hidden_states)  # [B * L, num_experts]
        scores = torch.sigmoid(logits + self.b_e)  # Auxiliary-loss-free Sigmoid activation with bias

        topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        # Normalize topk weights per token
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)
        return topk_weights, topk_indices


class FineGrainedMoE(nn.Module):
    """
    Fine-Grained MoE with 72 Routed Experts + 2 Universal Shared Experts.
    """
    def __init__(self, config: AetherCode33BConfig):
        super().__init__()
        self.config = config
        self.num_routed = config.num_routed_experts
        self.num_shared = config.num_shared_experts
        self.top_k = config.num_active_experts

        self.router = SigmoidRouter(config)

        # 72 Routed Fine-Grained Experts
        self.routed_experts = nn.ModuleList([
            SwiGLUMLP(config.hidden_size, config.moe_intermediate_size)
            for _ in range(self.num_routed)
        ])

        # 2 Universal Shared Experts
        self.shared_experts = nn.ModuleList([
            SwiGLUMLP(config.hidden_size, config.intermediate_size)
            for _ in range(self.num_shared)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat_states = hidden_states.view(-1, hidden_dim)

        # Shared Experts Computation (Always active)
        shared_output = torch.zeros_like(flat_states)
        for shared_exp in self.shared_experts:
            shared_output = shared_output + shared_exp(flat_states)

        # Routed Experts Computation
        routing_weights, routing_indices = self.router(flat_states)  # [B * L, top_k]

        routed_output = torch.zeros_like(flat_states)
        for k in range(self.top_k):
            exp_indices = routing_indices[:, k]
            exp_weights = routing_weights[:, k].unsqueeze(-1)

            for exp_id in range(self.num_routed):
                mask = (exp_indices == exp_id)
                if mask.any():
                    selected_states = flat_states[mask]
                    exp_out = self.routed_experts[exp_id](selected_states)
                    routed_output[mask] += exp_out * exp_weights[mask]

        final_output = shared_output + routed_output
        return final_output.view(batch_size, seq_len, hidden_dim)


class AetherCodeDecoderLayer(nn.Module):
    """
    Decoder layer supporting both Dense Anchor Base and Fine-Grained MoE.
    """
    def __init__(self, config: AetherCode33BConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = MultiHeadLatentAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

        is_moe = layer_idx >= config.num_dense_layers
        if is_moe:
            self.mlp = FineGrainedMoE(config)
        else:
            self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        normed_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(normed_states, attention_mask=attention_mask)
        hidden_states = hidden_states + attn_out

        normed_states = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed_states)
        hidden_states = hidden_states + mlp_out
        return hidden_states


class FactorizedOutputHead(nn.Module):
    """
    Low-Rank Untied Factorized Output Head (152K x 512 x 4096).
    """
    def __init__(self, config: AetherCode33BConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.rank = config.factorized_head_rank

        self.W_A = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.W_B = nn.Linear(self.rank, self.vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        latent = self.W_A(hidden_states)
        logits = self.W_B(latent)
        return logits


class AetherCode33BModel(nn.Module):
    """
    AetherCode-33B Base Transformer Model.
    """
    def __init__(self, config: AetherCode33BConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            AetherCodeDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class AetherCode33BForCausalLM(nn.Module):
    """
    AetherCode-33B Model with Untied Low-Rank Factorized LM Head.
    """
    def __init__(self, config: AetherCode33BConfig):
        super().__init__()
        self.config = config
        self.model = AetherCode33BModel(config)
        self.lm_head = FactorizedOutputHead(config)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)
        return logits
