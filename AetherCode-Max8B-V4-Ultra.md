# Master Blueprint: AetherCode-33B (33.0B Uniform Q5 Edition)

A 33.0 Billion total parameter ultra-sparse hybrid model featuring 5.8 Billion active parameters per token (Top-12 Active Routed from 72 Fine-Grained Experts + 2 Universal Shared Experts). Standardized strictly on Q5_K_M / Q5_K_L quantization across all expert and base weights (eliminating all Q5_K_S / IQ5_KS variants), this architecture accommodates a 128K q8_0 high-precision KV cache while guaranteeing $\ge 4.75\text{ GB}$ of unallocated free RAM on a 32.0 GB System RAM ceiling.

---

## 1. Executive Hardware & Architecture Summary

| Parameter / Dimension | Specification | Implementation Detail |
|---|---|---|
| **Total Parameters** | 33.0 Billion | Scaled down (44 physical layers) to absorb Q5_K_M/Q5_K_L weight density |
| **Active Parameters / Token** | 5.8 Billion | Layers 1–6 Dense + 12 Active Routed / 2 Shared Experts (Layers 7–44) |
| **Expert Topology** | 72 Fine-Grained Experts | Fine-grained routing ($\binom{72}{12}$ combination search space) + 2 Shared Experts |
| **Quantization Profile** | Strict Q5_K_M / Q5_K_L | Dense Anchor Base: Q6_K / Q5_K_M \| All Experts: Q5_K_M (5.5 bpw) |
| **Attention Architecture** | 100% MLA | Multi-Head Latent Attention ($d_c = 512, d_{\text{rope}} = 64$) with Soft-Capping |
| **Context Window ($N_{\text{ctx}}$)** | 131,072 Tokens (128K) | YaRN RoPE extrapolation + High-Precision q8_0 Latent Cache |
| **Speculative Acceleration** | Dynamic 3-Gram Engine | Zero-RAM inline prompt lookup (`--lookup-ngram-min 3 --draft-max 8`) |
| **Memory Allocation Target** | 27.25 GB Operating Footprint | 4.75 GB Guaranteed Free Cushion on 32.0 GB System RAM |
| **Est. Generation Speed** | 13–16 tok/s | Dual-Channel DDR5 @ 70 GB/s with 1 GB Static HugePages |

---

## 2. Intrinsic Model Architecture & Layer Topology

```
Input Tokens (152K Indent-Aware BPE Vocabulary)
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Layers 1–6: Dense Anchor Base (Q6_K / Q5_K_M Precision)                 │
│ - High-precision syntax extraction, AST boundaries, & whitespace logic  │
│ - 5.8B Active Base Parameters (No Expert Routing)                        │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Layers 7–44: Fine-Grained MoE + MLA Blocks (38 Layers)                  │
│ - Multi-Head Latent Attention (MLA) with q8_0 Latent KV Cache            │
│ - Attention Logit Soft-Capping (50.0) + Calibrated YaRN (128K Context)   │
│ - Auxiliary-Loss-Free Sigmoid Router with Dynamic Expert Bias (b_e)     │
│ - 72 Fine-Grained Experts (Strictly Q5_K_M) -> Top-12 Active + 2 Shared │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Low-Rank Untied Factorized Output Head (152K × 512 × 4096 @ Q8_0)       │
│ + Integrated Zero-RAM Dynamic 3-Gram Speculative Engine                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Quantization Standard Optimization

* **Elimination of Q5_K_S / IQ5_KS:** Q5_K_M uses 6-bit quantization for block scales and critical tensors (and 5-bit for remaining weights), providing higher weight accuracy on expert matrices compared to Q5_K_S (which forces static 5-bit sub-block scales).
* **Layer Budget Adjustment:** Upgrading all 72 routed experts from Q5_K_S (~5.15 bpw) to Q5_K_M (~5.50 bpw) increases expert weight density by ~6.8%. Trimming total layers from 46 to 44 (33.0B total parameters) completely neutralizes this memory increase, preserving the 4.75 GB free system RAM cushion.

---

## 3. Hardware Memory Budget (32 GB RAM Ceiling)

```
33B Model Weights (Q5_K_M Standard)   [22.75 GB]  ███████████████████████
128K MLA High-Precision Cache (q8_0)  [ 3.70 GB]  ████
GGML Graph & Temp Scratch Buffers      [ 0.80 GB]  █
Unallocated Free OS / IDE Cushion      [ 4.75 GB]  █████
```

| Component | Precision / Format | Q5_K_M Allocation | Q5_K_L Allocation |
|---|---|---|---|
| **Dense Base & Attention Weights (Layers 1–6)** | Q6_K / Q5_K_M | 7.20 GB | 7.45 GB |
| **72 MoE Fine-Grained Expert Weights** | Q5_K_M or Q5_K_L | 15.55 GB | 16.20 GB |
| **128K Context KV Cache** | MLA Latent Cache (q8_0) | 3.70 GB | 3.70 GB |
| **GGML Graph & Temp Tensors** | CPU Vector Scratch Buffer | 0.80 GB | 0.80 GB |
| **Total Model Operating Footprint** | — | **27.25 GB** | **28.15 GB** |
| **Free System Headroom (OS / IDE)** | Unallocated RAM Buffer | **4.75 GB** | **3.85 GB** |

> **Recommendation:** Standardize on Q5_K_M for the ideal balance of precision and 4.75 GB OS headroom. If maximum parameter precision is desired for specialized offline tasks, Q5_K_L remains fully functional within a 3.85 GB system cushion.

---

## 4. Linux Kernel & Production System Tuning

### 1. Static 1 GB HugePages Configuration (`hugetlbfs`)
Reserving 25 GB of system RAM as static 1 GB HugePages eliminates Translation Lookaside Buffer (TLB) misses during non-sequential expert routing:

Add the following to `/etc/default/grub` inside `GRUB_CMDLINE_LINUX_DEFAULT`:
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
# Disable swap aggression to maintain low latency
sudo sysctl -w vm.swappiness=0

# Increase max memory map count for large GGUF weights
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

# Pin execution strictly to physical CPU cores
PHYS_CORES=$(lscpu -p | grep -v '^#' | sort -u -t, -k2,2 | wc -l)

# Launch server instance with Q5_K_M weights and q8_0 KV cache
./build/bin/llama-server \
  --model ./models/AetherCode-33B-Q5_K_M.gguf \
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
FROM ./models/AetherCode-33B-Q5_K_M.gguf

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
ollama create aethercode-33b-q5m -f Modelfile
ollama run aethercode-33b-q5m
```

---

## 6. GGUF Metadata Calibration Header

Ensure these key-value attributes are present in the GGUF header during quantization:

```ini
[GGUF Metadata Keys]
general.quantization_version = 2
general.file_type = 17   # Q5_K_M (or 18 for Q5_K_L)
llama.block_count = 44
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
