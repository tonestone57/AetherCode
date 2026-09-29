To transform this from a pure model architecture into a production-grade, local IDE/Agent engine, five critical production features were missing from your setup:
* **Fill-In-The-Middle (FIM) Special Token Architecture:** Essential for IDE inline autocomplete (Cursor, VS Code, Continue.dev) to work in the middle of code blocks.
* **Embedded GGUF Tool-Calling & JSON-Schema Chat Template:** Native function calling and structured JSON output handling for local agent frameworks.
* **FlashAttention-3 Hardware Alignment ($d_{\text{head}} = 128$):** Fixed key/value head dimensions so `llama-server --flash-attn` uses fast CUDA/Metal matrix kernels.
* **Embedded RoPE Scaling Metadata (YaRN):** Native GGUF metadata for smooth context scaling beyond 262K up to 1,000,000 tokens without manual CLI flags.
* **Multi-Slot Parallel Serving Configuration:** Optimization flags enabling concurrent IDE background indexing alongside active chat sessions.

---

# Complete Unified Blueprint: AetherCode-Max8B (Production Edition)

## 1. Model Architecture & Vocabulary Extensions

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
                                 (48 Fine-Grained + 2 Shared Experts)
                                                     │
                                                     ▼
                                 [ Active Compute: 8.0B Params / Token ]
                                                     │
                                                     ▼
                                 [ Embedded 2-Step MTP Draft Heads ]
```

### FIM Special Tokens (IDE Infilling Protocol)
* `<|fim_prefix|>`: Marks code above the cursor
* `<|fim_suffix|>`: Marks code below the cursor
* `<|fim_middle|>`: Generates the code between prefix and suffix
* `<|repo_name|>`: Multi-file repository context tagging
* `<|file_sep|>`: File boundary demarcation for repository indexing

---

## 2. Comprehensive System RAM & VRAM Budget (32 GB RAM Target)

| Component | Technical Specification | RAM Footprint |
|---|---|---|
| **Model Base Weights** | 48B Total / 8.0B Active (Q4_K_M + iMatrix Calibrated) | 24.80 GB |
| **Embedded MTP Draft Tensors** | 2-Step Speculative Draft Heads (built-in GGUF) | 1.20 GB |
| **GDN + QSA Context Cache** | 262K Context Window (q4_0 Quantized KV Cache) | 0.85 GB |
| **N-Gram Lookup Table** | 20M Bigram/Trigram Entries (mmap CPU RAM System Cache) | 0.85 GB |
| **Multi-Slot Runtime Overhead** | 4 Parallel IDE/Agent Request Slots (`--parallel 4`) | 1.60 GB |
| **System Overhead** | OS, llama.cpp Engine, IDE Extensions | 2.20 GB |
| **Total Memory Usage** | Target Budget Ceiling: 32.00 GB | 31.50 GB |
| **Free RAM Headroom** | Safe Cushion for Compilers, Clangd, and OS | 0.50 GB |

---

## 3. Production Deployment Commands (llama.cpp & Ollama)

### A. Upstream llama-server Deployment
This launch command activates FlashAttention-3, Multi-Token Prediction (MTP), KV-Cache Quantization, 4-Slot Parallel Decoding, and Dynamic Reasoning Control:

```bash
# Build llama.cpp with native hardware optimizations
cmake -B build -DGGML_NATIVE=ON -DGGML_METAL=ON -DGGML_CUDA=ON
cmake --build build --config Release -j

# Launch high-concurrency production llama-server
./build/bin/llama-server \
    -m ./models/AetherCode-Max8B-Prod-Q4_K_M.gguf \
    --host 127.0.0.1 \
    --port 8080 \
    -c 262144 \
    --flash-attn \
    -ctk q4_0 \
    -ctv q4_0 \
    --spec-type draft-mtp \
    --spec-draft-n-max 2 \
    --parallel 4 \
    -t 8 \
    --mlock \
    --mmap \
    --chat-template-kwargs '{"reasoning_effort":"medium"}'
```

### B. Ollama Modelfile with Native FIM & Tool-Calling Support

```dockerfile
FROM ./models/AetherCode-Max8B-Prod-Q4_K_M.gguf

# Context & Hardware Allocations
PARAMETER num_ctx 262144
PARAMETER num_thread 8
PARAMETER temperature 0.1
PARAMETER top_p 0.95

# FIM Infill Stop Tokens
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|fim_prefix|>"
PARAMETER stop "<|fim_suffix|>"
PARAMETER stop "<|fim_middle|>"

# OpenAI / Cursor Compatible System Prompt
SYSTEM """You are AetherCode, an expert coding assistant built on the Qwen4 architecture.
You support structured JSON tool execution and high-speed code completion.
When writing code, produce clean, robust, memory-safe, fully typed solutions."""
```

Create and run:

```bash
ollama create aethercode-prod -f Modelfile
ollama run aethercode-prod "Write a lock-free SPMC queue in C++20"
```

---

## 4. Real-Time IDE Autocomplete Query Protocol (FIM Payload)

When your IDE (Cursor, VS Code, or Continue.dev) triggers inline autocomplete via the OpenAI/Ollama API, it passes FIM tokens directly to the backend:

```json
POST /v1/completions
{
  "model": "aethercode-prod",
  "prompt": "<|fim_prefix|>struct RingBuffer {\n    data: Vec<u8>,\n<|fim_suffix|>\n    pub fn new(capacity: usize) -> Self {\n        Self { data: vec![0; capacity] }\n    }\n}<|fim_middle|>",
  "max_tokens": 128,
  "temperature": 0.0,
  "stop": ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "\n\n"]
}
```

---

## 5. Final Performance Benchmarks across Modes

| Operation Mode | reasoning_effort | Use Case Target | Target Speed |
|---|---|---|---|
| **Inline Autocomplete (FIM)** | `none` | Cursor / IDE Line Infilling | 70–90+ tok/sec |
| **Code Review & Refactoring** | `low` / `medium` | Diff Generation & Bug Fixing | 45–55 tok/sec |
| **Deep Reasoning & Architecture** | `high` / `xhigh` | Multi-File Agent Code Generation | 28–35 tok/sec |
