# Master Blueprint: AetherCode-35B (Production Q5 + Q8 Cache Edition)

A 35.0 Billion total parameter ultra-sparse hybrid model featuring 6.2 Billion active parameters per token (Top-6 Active Routed + 2 Shared Experts). Scaled down to 35.0B parameters across 46 physical layers, this architecture provides a full 128K `q8_0` MLA KV cache while guaranteeing 4.55 GB of unallocated free System RAM on a strict 32.0 GB system ceiling.

---

## 1. Executive Hardware & Architecture Summary

| Parameter / Dimension | Specification | Implementation Detail |
|---|---|---|
| **Total Parameters** | 35.0 Billion | Scaled down (46 physical layers) to accommodate 128K q8_0 cache |
| **Active Parameters / Token** | 6.2 Billion | Layers 1–6 Dense + 6 Active Routed / 2 Shared Experts (Layers 7–46) |
| **Physical Layer Count** | 46 Layers | 6 Dense Anchor Base + 40 Ultra-Sparse MoE Layers |
| **Quantization Format** | Q5_K_M / Q5_K_S | Base: Q5_K_M (5.5 bpw) \| Experts: Q5_K_S / IQ5_KS (5.15 bpw) |
| **Attention Architecture** | 100% MLA | Multi-Head Latent Attention ($d_c = 512, d_{\text{rope}} = 64$) with QK-Norm |
| **Context Window ($N_{\text{ctx}}$)** | 131,072 Tokens (128K) | YaRN RoPE extrapolation (q8_0 high-precision quantized cache) |
| **BPE Vocabulary Size** | 152,000 Tokens | Indent-aware multi-space merging + Native FIM |
| **Target Hardware Ceiling** | 32.0 GB System RAM | Linux kernel optimized; 0 GPU dependency; standard `llama.cpp` |
| **Est. Generation Speed** | 12–14 tok/s | Dual-Channel DDR5 @ 70 GB/s bandwidth |

---

## 2. Intrinsic Model Architecture & Layer Topology

```
Input Tokens (152K Indent-Aware BPE Vocabulary)
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Layers 1–6: Dense Anchor Base (Q5_K_M Precision)                         │
│ - High-precision syntax, whitespace, and punctuation extraction         │
│ - 6.2B Active Base Parameters (No Expert Routing)                        │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Layers 7–46: Ultra-Sparse MoE + MLA Blocks (40 Layers)                  │
│ - Multi-Head Latent Attention (MLA) with q8_0 Latent Cache               │
│ - YaRN RoPE Extrapolation scaling position embeddings to 128K tokens     │
│ - Auxiliary-Loss-Free Sigmoid Router with Dynamic Expert Bias (b_e)     │
│ - 36 Routed Experts (Q5_K_S / IQ5_KS) -> Top-6 Active + 2 Shared        │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Low-Rank Untied Factorized Output Head (152K × 512 × 4096)              │
│ + Integrated Prompt Lookup Acceleration (Inline Speculative Engine)     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Revised Hardware Memory Budget (32 GB RAM Ceiling)

### Memory Allocation Breakdown (32 GB RAM Ceiling)

```
35B Model Weights (Q5_K_M / Q5_K_S)   [22.95 GB]  ███████████████████████
128K MLA High-Precision Cache (q8_0)  [ 3.70 GB]  ████
GGML Graph & Temp Tensors              [ 0.80 GB]  █
OS / IDE / Tooling Free Cushion       [ 4.55 GB]  █████
```

| Component | Precision / Format | Memory Allocation |
|---|---|---|
| **Dense Base & Attention Weights** | Q5_K_M (5.5 bpw) | 7.45 GB |
| **36 MoE Routed Expert Weights** | Q5_K_S / IQ5_KS (5.15 bpw) | 15.50 GB |
| **128K Context KV Cache** | MLA Latent Cache (q8_0 High Precision) | 3.70 GB |
| **GGML Graph & Temp Tensors** | CPU Vector Scratch Buffer | 0.80 GB |
| **Total Model Operating Footprint** | — | **27.45 GB** |
| **Linux OS / IDE Headroom** | Free System Buffer | **4.55 GB** |

> **Memory Optimization Note:** Lowering the MoE layer count from 48 (38B) to 40 (35B) trims weight memory by 1.73 GB. This freed capacity directly absorbs the additional memory required by the `q8_0` KV cache (up from 1.95 GB in `q4_0` to 3.70 GB in `q8_0`), maintaining 4.55 GB of clean unallocated RAM for background IDE tools, compiler daemons, and system overhead.

---

## 4. Linux Kernel & llama.cpp Production Deployment

### 1. Host OS Configuration

```bash
# Set THP to madvise so GGML manages hugepage allocation cleanly
echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled

# Disable swap aggression to prevent kernel page swapping
sudo sysctl -w vm.swappiness=0

# Expand max memory map count for GGUF files
sudo sysctl -w vm.max_map_count=524288
```

### 2. Validated Upstream llama-server Command

```bash
# Build llama.cpp with Linux CPU vector extensions (AVX-512) & NUMA support
cmake -B build -DGGML_NATIVE=ON -DGGML_AVX512=ON -DGGML_NUMA=ON
cmake --build build --config Release -j$(nproc)

# Execute server instance with 35B model, 128K context, and q8_0 KV cache
./build/bin/llama-server \
  --model ./models/AetherCode-35B-Q5_K_M.gguf \
  --ctx-size 131072 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --threads $(nproc) \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --lookup-ngram-min 2 \
  --draft-max 16 \
  --mlock \
  --numa distribute \
  --host 127.0.0.1 \
  --port 8080
```

### 3. Native Ollama Deployment (Modelfile)

```dockerfile
FROM ./models/AetherCode-35B-Q5_K_M.gguf

# 128K context size for 32GB RAM operation
PARAMETER num_ctx 131072
PARAMETER num_batch 2048
PARAMETER temperature 0.10
PARAMETER top_p 0.90

# Penalty tuned for code syntax stability
PARAMETER repeat_penalty 1.00
PARAMETER presence_penalty 0.10

# FIM and Chat Control Stop Markers
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|file_sep|>"
PARAMETER stop "<|fim_prefix|>"
PARAMETER stop "<|fim_suffix|>"
PARAMETER stop "<|fim_middle|>"
PARAMETER stop "</think>"

# Jinja Chat Template with Reasoning Block Support
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ if .Response }}{{ .Response }}{{ else }}<think>
{{ end }}"""
```

Register and launch:

```bash
ollama create aethercode-35b-q5 -f Modelfile
ollama run aethercode-35b-q5
```

---

## 5. GGUF Metadata Configuration

Ensure these key-value pairs are stored in the GGUF header during quantization:

```ini
[GGUF Metadata Keys]
general.quantization_version = 2
general.file_type = 17   # Q5_K_M
llama.block_count = 46
llama.expert_routed_count = 36
llama.expert_active_count = 6
llama.expert_shared_count = 2
llama.expert_weights_scale  = 1.0
llama.rope.dimension_count = 64
llama.attention.kv_lora_rank = 512
```
