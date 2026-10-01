#!/bin/bash

# Preload mimalloc to prevent heap memory fragmentation across MoE layers
export MIMALLOC_LARGE_OS_PAGES=1
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libmimalloc.so.2

# Pin execution strictly to physical CPU cores
PHYS_CORES=$(lscpu -p | grep -v '^#' | sort -u -t, -k2,2 | wc -l)

# Launch server instance with Q5_K_M weights, q8_0 KV cache, and ubatch-size 256
./build/bin/llama-server \
  --model ./models/AetherCode-33B-Q5_K_M.gguf \
  --ctx-size 131072 \
  --batch-size 4096 \
  --ubatch-size 256 \
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
