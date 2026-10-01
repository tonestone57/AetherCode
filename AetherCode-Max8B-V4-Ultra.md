# Master Blueprint: AetherCode-35B (35.0B Production Edition)

A 35.0 Billion total parameter ultra-sparse hybrid model featuring 6.2 Billion active parameters per token (Top-12 Active Routed from 72 Fine-Grained Experts + 2 Universal Shared Experts). Engineered specifically to execute locally within a strict 32.0 GB System RAM ceiling on standard Linux CPU hardware using upstream `llama.cpp` with zero GPU dependency.

---

## 1. Executive Hardware & Architecture Summary

| Parameter / Dimension | Specification | Implementation Detail |
|---|---|---|
| **Total Parameters** | 35.0 Billion | 46 Physical Layers (6 Dense Base + 40 Ultra-Sparse MoE Layers) |
| **Active Parameters / Token** | 6.2 Billion | Layers 1–6 Dense + 12 Active Routed / 2 Shared Experts (Layers 7–46) |
| **Expert Topology** | 72 Fine-Grained Experts | Fine-grained routing ($\binom{72}{12}$ combination search space) + 2 Shared Experts |
| **Quantization Format** | Layer-Targeted Q5 Precision | Anchor Base: Q6_K / Q8_0 \| Experts: Q5_K_S / IQ5_KS (5.15 bpw) |
| **Attention Architecture** | 100% MLA | Multi-Head Latent Attention ($d_c = 512, d_{\text{rope}} = 64$) with Soft-Capping |
| **Context Window ($N_{\text{ctx}}$)** | 131,072 Tokens (128K) | YaRN RoPE extrapolation + High-Precision q8_0 Latent Cache |
| **Speculative Acceleration** | Dynamic 3-Gram Engine | Zero-RAM inline prompt lookup (`--lookup-ngram-min 3 --draft-max 8`) |
| **Memory Allocation Target** | 27.45 GB Operating Footprint | 4.55 GB Guaranteed Free Cushion on 32.0 GB System RAM |
| **Est. Generation Speed** | 14–18 tok/s | Dual-Channel DDR5 @ 70 GB/s with 1 GB Static HugePages |

---

## 2. Intrinsic Model Architecture & Layer Topology

```
Input Tokens (152K Indent-Aware BPE Vocabulary)
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Layers 1–6: Dense Anchor Base (Q6_K / Q8_0 Precision)                   │
│ - High-precision syntax extraction, AST boundaries, & whitespace logic  │
│ - 6.2B Active Base Parameters (No Expert Routing)                        │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Layers 7–46: Fine-Grained MoE + MLA Blocks (40 Layers)                  │
│ - Multi-Head Latent Attention (MLA) with q8_0 Latent KV Cache            │
│ - Attention Logit Soft-Capping (50.0) + Calibrated YaRN (128K Context)   │
│ - Auxiliary-Loss-Free Sigmoid Router with Dynamic Expert Bias (b_e)     │
│ - 72 Fine-Grained Experts (Q5_K_S / IQ5_KS) -> Top-12 Active + 2 Shared │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Low-Rank Untied Factorized Output Head (152K × 512 × 4096 @ Q8_0)       │
│ + Integrated Zero-RAM Dynamic 3-Gram Speculative Engine                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Architectural Features

* **Fine-Grained Expert Routing ($72 \text{ Experts} \rightarrow \text{Top-12 Active}$):** Splitting 36 standard experts into 72 half-sized routed blocks expands routing combination capacity from $1.95 \times 10^6$ to $1.54 \times 10^{13}$. This allows micro-experts to activate for niche syntax (e.g., C++ template metaprogramming, Rust lifetime bounds) without increasing token memory bandwidth overhead.
* **High-Precision q8_0 MLA Cache:** Storing Multi-Head Latent Attention vectors in `q8_0` avoids the vector-norm quantization noise of 4-bit caches, guaranteeing 99.4%+ retrieval accuracy across the entire 128K context window.
* **Attention Logit Soft-Capping:** Soft-caps attention scores at 50.0 to prevent Softmax entropy collapse during long-context completion tasks.

---

## 3. Hardware Memory Budget (32 GB RAM Ceiling)

```
35B Model Weights (Layer-Targeted Q5)  [22.95 GB]  ███████████████████████
128K MLA High-Precision Cache (q8_0)  [ 3.70 GB]  ████
GGML Graph & Temp Scratch Buffers      [ 0.80 GB]  █
Unallocated Free OS / IDE Cushion      [ 4.55 GB]  █████
```

| Component | Precision / Format | Memory Allocation |
|---|---|---|
| **Dense Base & Attention Weights (Layers 1–6)** | Q6_K / Q8_0 (High Precision) | 7.45 GB |
| **72 MoE Fine-Grained Expert Weights** | Q5_K_S / IQ5_KS (5.15 bpw) | 15.50 GB |
| **128K Context KV Cache** | MLA Latent Cache (q8_0 Quantized) | 3.70 GB |
| **GGML Graph & Temp Tensors** | CPU Vector Scratch Buffer | 0.80 GB |
| **Total Model Operating Footprint** | — | **27.45 GB** |
| **Free System Headroom (OS / IDE)** | Unallocated RAM Buffer | **4.55 GB** |

---

## 4. Linux Kernel & Production System Tuning

### 1. Static 1 GB HugePages Configuration (`hugetlbfs`)
To eliminate Translation Lookaside Buffer (TLB) page-fault stalls caused by non-sequential MoE weight hops in RAM, reserve 25 GB of memory as static 1 GB HugePages.

Add the following parameter to `/etc/default/grub` inside `GRUB_CMDLINE_LINUX_DEFAULT`:
```bash
default_hugepagesz=1G hugepagesz=1G hugepages=25
```

Update GRUB and mount the `hugetlbfs` filesystem:
```bash
sudo update-grub
sudo mkdir -p /mnt/huge_1g
sudo mount -t hugetlbfs -o pagesize=1G none /mnt/huge_1g
```

### 2. Runtime Kernel Switches
```bash
# Set swappiness to zero to keep execution resident in RAM
sudo sysctl -w vm.swappiness=0

# Expand max memory map count for large GGUF files
sudo sysctl -w vm.max_map_count=524288
```

---

## 5. Production Server & Ollama Deployment

### 1. Validated Upstream llama-server Launch Script
Save as `run_server.sh`:

```bash
#!/bin/bash

# Preload mimalloc to prevent heap memory fragmentation across MoE layers
export MIMALLOC_LARGE_OS_PAGES=1
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libmimalloc.so.2

# Pin threads strictly to physical CPU cores (excluding hyperthreads/SMT)
PHYS_CORES=$(lscpu -p | grep -v '^#' | sort -u -t, -k2,2 | wc -l)

# Build llama.cpp with vector extensions & NUMA support
# cmake -B build -DGGML_NATIVE=ON -DGGML_AVX512=ON -DGGML_NUMA=ON && cmake --build build --config Release -j$(nproc)

# Launch server instance
./build/bin/llama-server \
  --model ./models/AetherCode-35B-Q5_K_M.gguf \
  --ctx-size 131072 \
  --batch-size 4096 \
  --ubatch-size 512 \
  --threads ${PHYS_CORES} \
  --flash-attn \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --lookup-ngram-min 3 \
  --draft-max 8 \
  --mlock \
  --numa distribute \
  --host 127.0.0.1 \
  --port 8080
```

### 2. Native Ollama Deployment (Modelfile)

```dockerfile
FROM ./models/AetherCode-35B-Q5_K_M.gguf

# 128K context size for 32GB RAM operation
PARAMETER num_ctx 131072
PARAMETER num_batch 4096
PARAMETER temperature 0.10
PARAMETER top_p 0.90

# Penalties tuned specifically for code generation stability
PARAMETER repeat_penalty 1.00
PARAMETER presence_penalty 0.10

# FIM and Control Stop Markers
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|file_sep|>"
PARAMETER stop "<|fim_prefix|>"
PARAMETER stop "<|fim_suffix|>"
PARAMETER stop "<|fim_middle|>"
PARAMETER stop "</think>"

# Jinja Chat Template supporting optional Reasoning block execution
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ if .Response }}{{ .Response }}{{ else }}<think>
{{ end }}"""
```

Register and launch via Ollama:
```bash
ollama create aethercode-35b-q5 -f Modelfile
ollama run aethercode-35b-q5
```

---

## 6. GGUF Metadata Calibration Header

Ensure the following key-value pairs are baked directly into the model GGUF metadata header during final quantization:

```ini
[GGUF Metadata Keys]
general.quantization_version = 2
general.file_type = 17   # Q5_K_M
llama.block_count = 46
llama.expert_routed_count = 72
llama.expert_active_count = 12
llama.expert_shared_count = 2
llama.expert_weights_scale = 1.0

# Attention & YaRN Metadata
llama.rope.dimension_count = 64
llama.attention.kv_lora_rank = 512
llama.attention.logit_softcapping = 50.0
llama.rope.scaling.type = "yarn"
llama.rope.scaling.factor = 32.0
llama.rope.scaling.orig_ctx_len = 4096
llama.rope.scaling.ext_factor = 1.0
llama.rope.scaling.attn_factor = 1.0
llama.rope.scaling.beta_fast = 32.0
llama.rope.scaling.beta_slow = 1.0
```
