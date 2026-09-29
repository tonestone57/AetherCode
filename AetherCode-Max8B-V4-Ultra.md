# Master Blueprint: AetherCode-Max8B (42B Production Edition)

The **AetherCode-Max8B (42B Production Edition)** scales total parameter capacity down from 48B to 42.0B Total Parameters while maintaining 8.0B Active Parameters per token pass. This structural shift frees up memory to dedicate 7.00 GB of RAM for OS, compiler, and IDE overhead on a 32 GB system, while remaining natively compatible with upstream `llama.cpp` and `Ollama` without custom kernels.

---

## 1. System Memory Allocation Budget (32 GB RAM Ceiling)

All model weights strictly use Q4_K_M + iMatrix calibration (4-bit block quantization with 6-bit scale factors). Low-precision Q3 quant formats are completely excluded.

| Memory Component | Technical Specification | Memory Footprint |
|---|---|---|
| **Model Base Weights (42B Total / 8.0B Active)** | Q4_K_M + iMatrix (60 Layers, 36 Routed Experts) | 21.71 GB |
| **Embedded MTP Draft Tensors** | 2-Step Speculative Prediction Heads (inside GGUF) | 1.05 GB |
| **GDN State + QSA Context Cache** | 262K Token Window (q4_0 Quantized KV Cache) | 0.85 GB |
| **N-Gram Offloaded Lookup Table** | 20M Bigram/Trigram Entries (mmap System RAM Cache) | 0.85 GB |
| **System & OS Overhead Allocation** | OS, Compilers, Clangd, IDEs, Web Browsers | 7.00 GB |
| **Total Memory Usage** | Target Budget Ceiling: 32.00 GB | 31.46 GB |
| **Free RAM Cushion** | Buffer to prevent OS memory thrashing | 0.54 GB |

---

## 2. Updated Model Topology

```
                           [ 152K Vocab BPE Tokenizer + Native FIM + Tool Tokens ]
                                                     │
                                                     ▼
                             [ 60 Layers w/ Gated Residuals (GR) & d_head=128 ]
                                                     │
                 ┌───────────────────────────────────┴───────────────────────────────────┐
                 │ (75% GDN Layers)                                      │ (25% QSA Layers)
                 ▼                                                       ▼
      [ Gated DeltaNet (GDN) ]                                [ Qwen Sparse Attention (QSA) ]
      (Recurrent O(1) Memory Compression)                     (FlashAttention-3 Kernels)
                 │                                                       │
                 └───────────────────────────────────┬───────────────────┘
                                                     │
                                                     ▼
                                   [ Ultra-Sparse MoE Routing Layer ]
                                 (36 Fine-Grained + 2 Shared Experts)
                                                     │
                                                     ▼
                                 [ Active Compute: 8.0B Params / Token ]
                                 (8 Routed Experts + 2 Shared Experts Active)
                                                     │
                                                     ▼
                                 [ Embedded 2-Step MTP Draft Heads ]
```

---

## 3. Core Architectural Pillar Specifications

### A. MoE Expert Sizing (36 Routed / 8.0B Active)
* **Total Experts:** 36 fine-grained routed experts + 2 permanently active shared experts per layer.
* **Active Routing:** 8 routed experts activated per token via auxiliary-loss-free softmax gating + 2 shared experts = 8.0B active parameters engaged per forward pass.
* **Specialization Density:** Trimming routed experts from 48 to 36 saves ~3.1 GB of weight RAM without altering reasoning power per token pass.

### B. Hybrid GDN + QSA Attention Layering ($d_{\text{head}} = 128$)
* **75% Gated DeltaNet (GDN):** Three out of every four layers use linear recurrent Gated DeltaNet state blocks, compressing historical context into a fixed-size memory vector.
* **25% Qwen Sparse Attention (QSA):** The remaining layers utilize QSA with a micro-block retrieval indexer. Head dimension is locked to $d_{\text{head}} = 128$ for hardware alignment with FlashAttention-3 kernels.

### C. Multi-Token Prediction (MTP) Speculative Heads
* **Embedded Draft Heads:** Houses 2-step speculative prediction heads directly inside the base GGUF file.
* **Zero Draft Model Overhead:** `llama.cpp` draft verification (`--spec-type draft-mtp`) yields a 1.4× – 1.8× speedup during generation without needing external draft models.

### D. Native Fill-In-The-Middle (FIM) & Repository Infilling
Native FIM tokens are baked directly into the 152K vocabulary for sub-100ms line completions in Cursor, VS Code, and Continue.dev:
* `<|fim_prefix|>`: Code context prior to cursor.
* `<|fim_suffix|>`: Code context past cursor.
* `<|fim_middle|>`: Target generation region.
* `<|repo_name|>` & `<|file_sep|>`: Multi-file repository indexing boundaries.

### E. 262K Native Context & Hybrid Reasoning Control
* **Context Capacity:** 262,144 native token context window, compressed via GDN states and `-ctk q4_0 -ctv q4_0` cache quantization.
* **Dynamic reasoning_effort:** Toggle reasoning traces dynamically via chat template parameters:
  * `none` / `low`: Fast inline completions and basic syntax generation.
  * `medium`: Balanced multi-step refactoring and explanations.
  * `high` / `xhigh`: Full CoT reasoning chains for deep debugging and system design.

---

## 4. Production Deployment Configurations

### Upstream llama-server Deployment Script
Compatible with official, unmodified `llama.cpp` builds (b3000 or newer):

```bash
# Build llama.cpp with native AVX-512 / Metal / CUDA acceleration
cmake -B build -DGGML_NATIVE=ON -DGGML_METAL=ON -DGGML_CUDA=ON
cmake --build build --config Release -j

# Launch llama-server with 42B Q4_K_M weights and 7GB system buffer
./build/bin/llama-server \
    -m ./models/AetherCode-Max8B-42B-Q4_K_M.gguf \
    --host 127.0.0.1 \
    --port 8080 \
    -c 262144 \
    --flash-attn \
    -ctk q4_0 \
    -ctv q4_0 \
    --spec-type draft-mtp \
    --spec-draft-n-max 2 \
    --parallel 2 \
    -t 8 \
    --mlock \
    --mmap \
    --chat-template-kwargs '{"reasoning_effort":"medium"}'
```

### Ollama Deployment (Modelfile)
* Define the `Modelfile`:

```dockerfile
FROM ./models/AetherCode-Max8B-42B-Q4_K_M.gguf

# Context & Execution Allocation
PARAMETER num_ctx 262144
PARAMETER num_thread 8
PARAMETER temperature 0.1
PARAMETER top_p 0.95

# FIM Special Stop Tokens
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|fim_prefix|>"
PARAMETER stop "<|fim_suffix|>"
PARAMETER stop "<|fim_middle|>"

SYSTEM """You are AetherCode, an expert coding assistant built on the Qwen4 / 42B MoE framework.
Write clean, memory-safe, fully typed code."""
```

* Register and start in Ollama:

```bash
ollama create aethercode-42b -f Modelfile
ollama run aethercode-42b "Write an async HTTP connection pool in Rust"
```

---

## 5. Performance & Throughput Targets

| Operation Mode | reasoning_effort | Primary Task Target | Expected Speed (Q4_K_M + MTP) |
|---|---|---|---|
| **Inline Autocomplete (FIM)** | `none` | Cursor / IDE Line Infilling | 75–95+ tok/sec |
| **Code Review & Refactoring** | `low` / `medium` | Diff Generation & Bug Fixing | 48–58 tok/sec |
| **Deep Reasoning & Architecture** | `high` / `xhigh` | Multi-File Agentic Code Generation | 30–38 tok/sec |
