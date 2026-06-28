# Lab 01: LLM Profiling

> Profile model performance across quantization levels. Build your first cost-per-token number.

**Phase:** 1 — Mental Models
**GPU required:** No — runs entirely on CPU. GPU dramatically speeds it up but is not required.
**Time:** 2–3 hours
**Cost:** Free (local compute only)

---

## Objective

By the end of this lab you will have:
- A working local LLM inference setup via llama.cpp
- Benchmark numbers for VRAM/RAM usage at Q4, Q6, Q8, and F16 quant levels
- TTFT and tokens/sec measurements for each
- A cost-per-1M-tokens estimate mapped to real AWS instance pricing

This lab is intentionally low-infrastructure. The goal is to build intuition about quantization and inference performance before you touch Kubernetes or cloud GPU instances.

---

## Prerequisites

```bash
# Check you have these
cmake --version     # >= 3.14
python3 --version   # >= 3.9
pip3 show huggingface-hub 2>/dev/null || pip3 install huggingface-hub
```

---

## Part 1: Install llama.cpp

llama.cpp is a CPU (and GPU) inference engine for GGUF-format models. It's not what you'd run in production, but it's the fastest way to profile quantization on any hardware.

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# CPU-only build (works on any machine)
cmake -B build -DLLAMA_CURL=ON
cmake --build build --config Release -j$(nproc)

# Verify build
./build/bin/llama-cli --version
```

> **If you have a GPU:** enable it at build time for 10–50× faster inference:
> - NVIDIA: `cmake -B build -DGGML_CUDA=ON`
> - Apple Silicon: `cmake -B build -DGGML_METAL=ON`
> - The rest of this lab works identically either way.

---

## Part 2: Download the model at multiple quant levels

We'll use **Llama-3.2-1B-Instruct** — small enough to run on any laptop CPU in reasonable time, but representative of the quantization tradeoffs you'll see at larger scales.

```bash
mkdir -p ~/models && cd ~/models

# Download 4 quantization levels of the same model
# Source: Hugging Face Hub (bartowski's quantizations are the community standard)

pip3 install huggingface-hub

python3 - <<'EOF'
from huggingface_hub import hf_hub_download

models = [
    # (repo,                                        filename,                          label)
    ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Q4_K_M"),
    ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q6_K.gguf",   "Q6_K"),
    ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf",   "Q8_0"),
    ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-F16.gguf",    "F16"),
]

for repo, filename, label in models:
    print(f"Downloading {label}...")
    path = hf_hub_download(repo_id=repo, filename=filename, local_dir=".")
    print(f"  → {path}")
EOF

ls -lh ~/models/*.gguf
```

Expected file sizes (1B model — scale these by 8–70× for larger models):

```
Llama-3.2-1B-Instruct-Q4_K_M.gguf   ~700 MB
Llama-3.2-1B-Instruct-Q6_K.gguf     ~900 MB
Llama-3.2-1B-Instruct-Q8_0.gguf    ~1.1 GB
Llama-3.2-1B-Instruct-F16.gguf     ~2.1 GB
```

---

## Part 3: Benchmark each quant level

Save as `benchmark.sh` in your llama.cpp directory:

```bash
#!/usr/bin/env bash
set -euo pipefail

LLAMA_CLI="./build/bin/llama-cli"
MODELS_DIR="$HOME/models"
PROMPT="Explain the CAP theorem in distributed systems. Cover consistency, availability, and partition tolerance in detail."
N_TOKENS=256   # output tokens to generate

declare -A QUANT_FILES=(
    ["Q4_K_M"]="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    ["Q6_K"]="Llama-3.2-1B-Instruct-Q6_K.gguf"
    ["Q8_0"]="Llama-3.2-1B-Instruct-Q8_0.gguf"
    ["F16"]="Llama-3.2-1B-Instruct-F16.gguf"
)

echo "================================================================"
echo "  llama.cpp Quantization Benchmark"
echo "  Model: Llama-3.2-1B-Instruct"
echo "  Output tokens: $N_TOKENS"
echo "================================================================"
printf "%-10s %-12s %-14s %-14s %-12s\n" "Quant" "File Size" "Load Time(s)" "Tokens/sec" "RAM Used"
echo "----------------------------------------------------------------"

for QUANT in Q4_K_M Q6_K Q8_0 F16; do
    MODEL_FILE="$MODELS_DIR/${QUANT_FILES[$QUANT]}"
    SIZE=$(du -sh "$MODEL_FILE" | cut -f1)

    # Measure RAM before
    RAM_BEFORE=$(ps -o rss= $$ 2>/dev/null || echo 0)

    # Run benchmark — llama-bench gives us clean numbers
    OUTPUT=$($LLAMA_CLI \
        --model "$MODEL_FILE" \
        --prompt "$PROMPT" \
        --n-predict $N_TOKENS \
        --threads $(nproc) \
        --ctx-size 2048 \
        --no-display-prompt \
        2>&1)

    LOAD_TIME=$(echo "$OUTPUT" | grep -oP 'load time\s*=\s*\K[\d.]+' || echo "?")
    TOK_SEC=$(echo "$OUTPUT"  | grep -oP 'eval time.*\K[\d.]+(?= tokens per second)' | tail -1 || echo "?")
    RAM_MB=$(echo "$OUTPUT"   | grep -oP 'system RAM used.*\K[\d.]+(?= MiB)' | tail -1 || echo "?")

    printf "%-10s %-12s %-14s %-14s %-12s\n" \
        "$QUANT" "$SIZE" "${LOAD_TIME}ms" "${TOK_SEC}" "${RAM_MB} MiB"
done
```

```bash
chmod +x benchmark.sh && ./benchmark.sh
```

---

## Part 4: Run the precision quality test

Numbers are only meaningful alongside quality. Run the same prompt through each quant and compare outputs:

```bash
#!/usr/bin/env bash
# quality_test.sh

LLAMA_CLI="./build/bin/llama-cli"
MODELS_DIR="$HOME/models"
PROMPT="What is 17 multiplied by 43? Show your working."

for QUANT in Q4_K_M Q6_K Q8_0 F16; do
    echo ""
    echo "════════════════════════════════════════"
    echo "  $QUANT"
    echo "════════════════════════════════════════"
    $LLAMA_CLI \
        --model "$MODELS_DIR/Llama-3.2-1B-Instruct-${QUANT}.gguf" \
        --prompt "<|begin_of_text|><|start_header_id|>user<|end_header_id|>
${PROMPT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>" \
        --n-predict 128 \
        --threads $(nproc) \
        --no-display-prompt \
        2>/dev/null
done
```

```bash
chmod +x quality_test.sh && ./quality_test.sh
```

**What to look for:** at 1B scale, you'll likely see degradation at Q4_K_M on reasoning tasks. This is more pronounced at 7B+ where quantization tradeoffs are more meaningful.

---

## Part 5: Fill in your results table

```
Model: Llama-3.2-1B-Instruct
Hardware: ___________________

Quant    File Size   Tokens/sec   RAM Used   Quality (subjective)
────────────────────────────────────────────────────────────────
Q4_K_M
Q6_K
Q8_0
F16
```

---

## Part 6: Calculate cost-per-1M-tokens for AWS

```python
#!/usr/bin/env python3
# cost_estimate.py
# Fill in YOUR benchmark numbers from Part 3

results = {
    "Q4_K_M": {"tok_per_sec": 0, "notes": "fill in"},
    "Q6_K":   {"tok_per_sec": 0, "notes": "fill in"},
    "Q8_0":   {"tok_per_sec": 0, "notes": "fill in"},
    "F16":    {"tok_per_sec": 0, "notes": "fill in"},
}

# AWS GPU instance costs (on-demand, us-east-1)
# CPU numbers from your laptop need GPU multiplier (rough: 10-50x faster on GPU)
instances = {
    "g5.xlarge (A10G, 24GB)":   1.01,
    "g5.2xlarge (A10G, 24GB)":  1.21,
    "g4dn.xlarge (T4, 16GB)":   0.53,
}

print(f"{'Quant':<10} {'Tok/s (CPU)':>12}  {'Estimated GPU tok/s':>20}")
print("-" * 50)
for quant, data in results.items():
    cpu_tps = data["tok_per_sec"]
    # GPU is roughly 15-30x faster than laptop CPU for 1B model
    est_gpu_tps = cpu_tps * 20
    print(f"{quant:<10} {cpu_tps:>12.1f}  {est_gpu_tps:>20.1f}")

print()
print(f"{'Instance':<30} {'$/hr':>6}  {'$/1M tokens (est)':>18}")
print("-" * 60)
for instance, hourly in instances.items():
    # Use Q4_K_M GPU estimate for cost (most common production quant)
    q4_cpu = results["Q4_K_M"]["tok_per_sec"]
    est_gpu_tps = q4_cpu * 20
    tok_per_hr = est_gpu_tps * 3600
    cost_per_1m = (hourly / tok_per_hr) * 1_000_000
    print(f"{instance:<30} ${hourly:>5.2f}  ${cost_per_1m:>17.3f}")
```

```bash
python3 cost_estimate.py
```

---

## Expected results (reference — your numbers will vary)

On a modern laptop CPU (8-core, Llama-3.2-1B):

```
Quant    Tokens/sec   RAM       Quality notes
──────────────────────────────────────────────────────────
Q4_K_M   15–25        ~750 MB   Minor degradation on math
Q6_K     12–18        ~950 MB   Near-lossless
Q8_0     10–15        ~1.1 GB   Effectively lossless
F16       5–10        ~2.1 GB   Baseline (no quantization)
```

**Key observation:** Q6_K gives you 90% of F16 quality at 45% of the memory. This pattern holds at larger model scales and is why Q6_K or AWQ-4bit are common production choices.

---

## Cleanup

```bash
# Remove model files when done (free up disk space)
rm ~/models/*.gguf
```
