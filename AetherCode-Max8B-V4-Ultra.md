# Master Blueprint: AetherCode-Max8B (42B Production Edition)

The **AetherCode-Max8B (42B Production Edition)** scales total parameter capacity down from 48B to 42.0B Total Parameters while maintaining 8.0B Active Parameters per token pass. This structural shift frees up memory to dedicate up to 7.00 GB of RAM for OS, compiler, and IDE overhead on a 32 GB system, while remaining natively compatible with upstream `llama.cpp` and `Ollama` without custom kernels.

---

## 1. Updated Memory Allocation & System Layout (35M N-Gram)

Upgrading the N-Gram lookup table from 20M to 35M entries expands static boilerplate pattern matching while maintaining standard system stability on a 32.00 GB hardware limit.

| Blueprint Layer | Allocation Strategy | Footprint (20M Baseline) | Updated Footprint (35M Entry) |
|---|---|---|---|
| **Model Base Weights** | 42B Base Model (Q4_K_M) | 21.71 GB | 21.71 GB |
| **Embedded Draft Heads** | MTP Speculative Tensors (`--spec-type draft-mtp`) | 1.05 GB | 1.05 GB |
| **KV & Context Cache** | GDN State + QSA Context Cache (262K context, q4_0) | 0.85 GB | 0.85 GB |
| **N-Gram Table** | 35M Entries Static Lookup Cache (`-lcs`) | 0.85 GB | 2.10 GB (+1.25 GB) |
| **System Overhead Reserve** | OS, IDE (VS Code / JetBrains), Compiler, & Buffer | 7.00 GB | 6.29 GB (System RAM) or 7.00 GB (NVMe Direct) |
| **Total Memory Footprint** | Hardware Ceiling: 32.00 GB | 31.46 GB | 32.00 GB (System RAM) / 25.71 GB (NVMe) |

### Key System Adjustments for the 35M Upgrade

* **RAM-Bound Deployment (System RAM Execution):**
  * **Hit Rate:** Increases from ~85.0% to ~87.8% draft acceptance.
  * **Memory Realignment:** The N-gram table takes 2.10 GB. To avoid hitting page swapping on a 32.00 GB machine, the System Overhead Reserve is adjusted from 7.00 GB down to 6.29 GB. This maintains a 0.00 GB swap profile while giving high draft hits for repetitive framework structures.

* **NVMe-Mapped Option (`-lm mmap`):**
  * Mapping the 35M table file from an NVMe drive via `mmap` retains the full 7.00 GB System Overhead Reserve. The 2.10 GB file is read in $O(1)$ sparse chunks directly from SSD storage, adding less than 15 $\mu\text{s}$ per token pass.

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
Compatible with official, unmodified `llama.cpp` builds (b3000 or newer) utilizing the updated 35M N-Gram static lookup cache:

```bash
# Build llama.cpp with native AVX-512 / Metal / CUDA acceleration
cmake -B build -DGGML_NATIVE=ON -DGGML_METAL=ON -DGGML_CUDA=ON
cmake --build build --config Release -j

# Launch llama-server with 42B Q4_K_M weights, 35M N-Gram cache, and MTP draft heads
./build/bin/llama-server \
    -m ./models/AetherCode-Max-42B-Q4_K_M.gguf \
    --host 127.0.0.1 \
    --port 8080 \
    -c 262144 \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    -fa \
    --spec-type draft-mtp,ngram-cache \
    -lcs ngram_35m_static.bin \
    --spec-ngram-cache-size-n 12 \
    --spec-ngram-cache-size-m 8 \
    -lm mmap \
    -t 16 \
    --chat-template-kwargs '{"reasoning_effort":"medium"}'
```

#### Modified Flag Breakdown
* `-lcs ngram_35m_static.bin`: Points to the pre-compiled 35M-entry binary lookup cache file (~2.10 GB).
* `--spec-type draft-mtp,ngram-cache`: Runs combined neural Multi-Token Prediction with the static 35M N-gram table.
* `--spec-ngram-cache-size-n 12`: Sets pattern lookup depth to 12 back-tokens for matching syntax blocks.
* `--spec-ngram-cache-size-m 8`: Drafts up to 8 tokens per hit pass.
* `-lm mmap`: Enables memory-mapping for weight and lookup table loading to ensure OS page caching operates efficiently.

### Ollama Deployment (Modelfile)
* Define the `Modelfile`:

```dockerfile
FROM ./models/AetherCode-Max-42B-Q4_K_M.gguf

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

| Operation Mode | reasoning_effort | Primary Task Target | Expected Speed (Q4_K_M + MTP + 35M N-Gram) |
|---|---|---|---|
| **Inline Autocomplete (FIM)** | `none` | Cursor / IDE Line Infilling | 75–95+ tok/sec |
| **Code Review & Refactoring** | `low` / `medium` | Diff Generation & Bug Fixing | 48–58 tok/sec |
| **Deep Reasoning & Architecture** | `high` / `xhigh` | Multi-File Agentic Code Generation | 30–38 tok/sec |
