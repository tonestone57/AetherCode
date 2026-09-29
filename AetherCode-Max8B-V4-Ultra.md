# Master Blueprint: AetherCode-Max8B (Qwen4 / Flash-Next Framework)

This updated blueprint transitions the model to the Qwen4 / Qwen 3.8 Flash-Next framework and strictly enforces Q4_K_M quantization or better across all weights. By leveraging Hybrid GDN + QSA Attention and Gated Residuals, the KV cache footprint is drastically reduced, allowing a 262,144-token native context window to run smoothly alongside a 48B total / 8.0B active parameter MoE structure within a 32 GB RAM budget.

---

## 1. System Memory Allocation Budget (Q4_K_M Target)

All weights use GGML Q4_K_M (4-bit block quantization with 6-bit scales). Low-bit Q3 quants are completely removed.

| Memory Component | Technical Specification | Memory Footprint |
|---|---|---|
| **Model Weights (48B Total / 8.0B Active)** | Q4_K_M GGUF (60 Layers, Ultra-Sparse MoE) | 24.80 GB |
| **GDN Recurrent State + QSA Sparse Context** | 262K Token Window (Gated DeltaNet + QSA Indexing) | 0.85 GB |
| **N-Gram Offloaded Table** | Bigram/Trigram Lookup Table (mmap System RAM) | 0.85 GB |
| **System & Engine Overhead** | OS, llama.cpp / Ollama Server, IDE Extensions | 3.80 GB |
| **Total Memory Footprint** | Target Ceiling: 32.00 GB | 30.30 GB |
| **Free System RAM Headroom** | Safe cushion for compilers, IDEs, and OS | 1.70 GB |

---

## 2. Updated Model Topology

```
                                  [ Standard 152K BPE Tokenizer ]
                                                 │
                                                 ▼
                          [ 60 Distinct Layers w/ Gated Residuals (GR) ]
                                                 │
             ┌───────────────────────────────────┴───────────────────────────────────┐
             │ (75% of Layers)                                       │ (25% of Layers)
             ▼                                                       ▼
  [ Gated DeltaNet (GDN) ]                                [ Qwen Sparse Attention (QSA) ]
  (Recurrent O(1) Memory Compression)                     (Micro-block Granularity Retrieval)
             │                                                       │
             └───────────────────────────────────┬───────────────────┘
                                                 │
                                                 ▼
                               [ Ultra-Sparse MoE Routing Layer ]
                             (48 Fine-Grained + 2 Shared Experts)
                                                 │
                                                 ▼
                              [ Active Compute per Token: 8.0B Params ]
                              (8 Routed Experts Activated + 2 Shared Experts)
                                                 │
                                                 ▼
                              [ Offloaded N-Gram Prefetching ]
                              (20M Entries / Local CPU System RAM)
```

---

## 3. Core Architectural Pillars (Qwen4 Framework)

### A. Hybrid GDN + QSA Attention Layering
* **75% Gated DeltaNet (GDN):** Three out of every four layers replace traditional matrix attention with Gated DeltaNet linear recurrent layers. Past conversation context is compressed into a fixed-size recurrent state, slashing prefill processing times by up to 8.6x.
* **25% Qwen Sparse Attention (QSA):** The remaining layers use QSA, utilizing a lightweight indexer to retrieve specific key tokens at micro-block granularity during complex multi-file reasoning.

### B. Ultra-Sparse Mixture-of-Experts (MoE)
* **Parameter Distribution:** 48 fine-grained routed experts + 2 permanently active shared experts per layer.
* **Token Activation:** Only 8 routed experts + 2 shared experts activate per token pass, keeping inference compute fixed at 8.0B active parameters while maintaining the broad domain coverage of a 48B model.

### C. Gated Residuals (GR)
* **Multi-Branch Flow:** Replaces standard single-path residual skip connections by splitting internal residual streams into 4 branches managed by dynamic read/write gates.
* **Stability:** Prevents representation collapse and gradient explosion during long-context generation sequences.

### D. N-Gram Offloaded Memory Scaling
* **RAM Offloading:** Houses a lookup table for local bigrams and trigrams (20M entries / 0.85 GB) in CPU system RAM using asynchronous prefetching (`mmap`).
* **Zero-FLOP Accelerations:** Offloads common code syntax patterns directly to memory lookups without executing tensor multiplications.

### E. 262K Native Context & Hybrid Reasoning Controls
* **Context Capacity:** 262,144 tokens native out of the box (scalable up to 1,000,000 using YaRN).
* **Dynamic reasoning_effort Toggles:** Supports hybrid reasoning depth control. Reasoning traces can be adjusted on the fly using runtime chat template kwargs:
  * `none` / `low`: Fast inline code completions and simple refactoring.
  * `medium`: Default balanced logical reasoning.
  * `high` / `xhigh`: Deep step-by-step `<think>` reasoning for multi-file bug diagnosis and architecture planning.

---

## 4. Native llama.cpp & Ollama Deployment Commands

### Option A: Upstream llama.cpp (llama-server)
To run the model with Q4_K_M GGUF and hybrid thinking controls in `llama.cpp`:

```bash
# Build llama.cpp with native AVX-512 / Metal / CUDA support
cmake -B build -DGGML_NATIVE=ON -DGGML_METAL=ON
cmake --build build --config Release -j

# Launch llama-server with 262K context and Q4_K_M weights
./build/bin/llama-server \
    -m ./models/AetherCode-Max8B-Q4_K_M.gguf \
    --host 127.0.0.1 \
    --port 8080 \
    -c 262144 \
    -t 8 \
    --mlock \
    --mmap \
    --chat-template-kwargs '{"reasoning_effort":"medium"}'
```

> **Reasoning Effort Control:** To increase or decrease thinking traces during runtime, adjust `--chat-template-kwargs '{"reasoning_effort":"xhigh"}'` or `'{"reasoning_effort":"low"}'`.

### Option B: Ollama Deployment
* Create a `Modelfile` referencing the Q4_K_M model:

```dockerfile
FROM ./models/AetherCode-Max8B-Q4_K_M.gguf

PARAMETER num_ctx 262144
PARAMETER num_thread 8
PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|im_end|>"

# Set default hybrid reasoning effort to medium
SYSTEM """ You are AetherCode, an expert coding assistant built on the Qwen4 architecture.
Write clean, memory-safe, fully typed code.
"""
```

* Build and run in Ollama:

```bash
ollama create aethercode-q4 -f Modelfile
ollama run aethercode-q4 "Write a thread-safe MPMC queue in Rust"
```

* Passing `reasoning_effort` via API requests:

```json
POST /api/chat
{
  "model": "aethercode-q4",
  "messages": [
    { "role": "user", "content": "Refactor this database driver to use async connection pooling." }
  ],
  "options": {
    "chat_template_kwargs": {
      "reasoning_effort": "high"
    }
  }
}
```

---

## 5. Performance Expectations (Q4_K_M)

| Operating Mode | Active Parameters | Quantization | Expected Speed (CPU + Unified Memory) |
|---|---|---|---|
| **Architect / Thinking Mode (xhigh)** | 8.0 Billion | Q4_K_M | 24–30 tok/sec |
| **Standard Response (medium)** | 8.0 Billion | Q4_K_M | 35–42 tok/sec |
| **Inline Autocomplete (none / low)** | 8.0 Billion | Q4_K_M | 50–60+ tok/sec |
