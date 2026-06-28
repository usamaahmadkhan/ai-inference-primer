# Phase 1: Mental Models

> Before you operate it, understand it.

**Duration:** 2–3 weeks
**Lab:** [01-llm-profiling](../../labs/01-llm-profiling/)

---

## Why this phase exists

Most SREs jump straight to deploying vLLM and then spend weeks debugging things they don't understand. GPU memory pressure, latency spikes when batch size grows, the cost jump when you move from INT4 to FP16 — all of these are explained by what happens inside the model, not in your infrastructure.

Spend 2–3 weeks here. You will recoup that time a hundred times over when you're on-call.

---

## Core concepts

### 1. GPU Architecture

The physical hardware you're scheduling on. Knowing what's inside the box makes you a better debugger.

```
A Single GPU (e.g., NVIDIA H100 80GB SXM)
┌─────────────────────────────────────────────────────┐
│  132 Streaming Multiprocessors (SMs)                │
│  ┌──────────────────┐  ┌──────────────────┐         │
│  │   CUDA Cores     │  │  Tensor Cores    │         │
│  │  (FP32 compute)  │  │ (matrix math,    │         │
│  │                  │  │  the AI work)    │         │
│  └──────────────────┘  └──────────────────┘         │
│                                                     │
│  80 GB HBM3 (High Bandwidth Memory)                 │
│  Memory Bandwidth: 3.35 TB/s                        │
│  ┌─────────────────────────────────────────┐        │
│  │  Model weights live here                │        │
│  │  KV cache lives here (the hot resource) │        │
│  └─────────────────────────────────────────┘        │
│                                                     │
│  NVLink (chip-to-chip, 900 GB/s bidirectional)      │
│  (This is why multi-GPU tensor parallelism works)   │
└─────────────────────────────────────────────────────┘
```

**What you need to know:**
- VRAM is your primary constraint. Run out of it and the process crashes, not gracefully degrades.
- Memory bandwidth (TB/s), not compute (TFLOPS), is usually the bottleneck for inference.
- Tensor cores handle the matrix multiplications. They're why modern GPUs are so fast for transformers.
- NVLink enables high-bandwidth GPU-to-GPU communication. Without it (e.g., across PCIe), tensor parallelism gets expensive.

**AWS GPU families mapped to chips:**

| Instance | GPU | VRAM | Use Case |
|---|---|---|---|
| g4dn | NVIDIA T4 | 16 GB | Dev, small models |
| g5 | NVIDIA A10G | 24 GB | 7B–13B models |
| g6 | NVIDIA L4 | 24 GB | Inference-optimized, cheaper |
| p3 | NVIDIA V100 | 16/32 GB | Legacy, avoid for new deployments |
| p4d | NVIDIA A100 | 40 GB | 30B–70B models |
| p4de | NVIDIA A100 | 80 GB | Large models, fine-tuning |
| p5 | NVIDIA H100 | 80 GB | Largest models, highest throughput |

---

### 2. LLM Inference Mechanics

What actually happens when a request hits your endpoint.

```
Inference Request Lifecycle
═══════════════════════════

User prompt: "Explain Kubernetes to me"
           │
           ▼
┌──────────────────────┐
│   TOKENIZATION       │  "Explain" → [32354]
│                      │  "Kubernetes" → [42]
│  Text → Token IDs    │  "to" → [311]
│                      │  "me" → [757]
└──────────┬───────────┘
           │  Input: [32354, 42, 311, 757]
           ▼
┌──────────────────────────────────────────────────┐
│   PREFILL PHASE  (compute-bound)                 │
│                                                  │
│   Process ALL input tokens in parallel           │
│   Compute attention for every token              │
│   Store Key/Value pairs in KV Cache              │
│                                                  │
│   Duration: proportional to prompt length        │
│   This is your TTFT (Time to First Token)        │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│   DECODE PHASE  (memory bandwidth-bound)         │
│                                                  │
│   Generate ONE token at a time                   │
│   Each token attends to all previous tokens      │
│   (reads from KV Cache each step)                │
│                                                  │
│   Token 1: "Kubernetes"                          │
│   Token 2: "is"                                  │
│   Token 3: "a"          ◄── each step reads      │
│   Token 4: "container"      full KV cache        │
│   ...                                            │
│                                                  │
│   Duration: proportional to output length        │
│   This determines your TBT (Time Between Tokens) │
└──────────────────────────────────────────────────┘
```

**The KV Cache** is the central resource in LLM serving. It stores intermediate attention states so you don't recompute them on every decode step. It lives in VRAM. When it's full, throughput collapses.

This is what PagedAttention (vLLM's core innovation) solves.

---

### 3. Quantization

Reducing model precision to save VRAM and increase throughput. You'll make this decision constantly.

```
Model Size vs Precision Trade-off (Llama-3 70B example)
════════════════════════════════════════════════════════

FP32  (4 bytes/param)  │████████████████████████│ ~280 GB  ← never used for inference
FP16  (2 bytes/param)  │████████████│            ~140 GB  ← baseline
BF16  (2 bytes/param)  │████████████│            ~140 GB  ← same size, better training
INT8  (1 byte/param)   │██████│                  ~70 GB   ← small quality loss
INT4  (0.5 bytes/param)│███│                     ~35 GB   ← fits on 2x A10G
GPTQ  (4-bit grouped)  │███│                     ~35 GB   ← calibrated, better quality
AWQ   (4-bit act-aware) │███│                    ~35 GB   ← currently best 4-bit method

                              Quality loss
    FP16 ──────────────────────────────── INT4
    (none)                               (noticeable on complex reasoning)
```

**How to choose:**
- `FP16/BF16` — when quality matters most and you have the VRAM
- `INT8` — good default for most production use cases
- `AWQ` or `GPTQ` — when you need to fit on fewer/smaller GPUs without major quality loss
- Never use `FP32` for inference

---

### 4. Key Inference Metrics

These replace your RPS/p99 vocabulary. Internalize them before you write a single SLO.

```
Inference Latency Breakdown
═══════════════════════════

Request arrives
     │
     │◄── Queue time (request waited in batch queue)
     │
     ├── PREFILL begins
     │        │
     │        │  (processing input tokens)
     │        │
     │◄────── TTFT (Time to First Token)
     │
     ├── DECODE begins
     │        │
     │        │◄── TBT (Time Between Tokens)
     │        │◄── TBT
     │        │◄── TBT
     │        │     ... repeated for each output token
     │
     ▼
  Last token arrives
     │
     │◄── E2E Latency = TTFT + (N_tokens × TBT)
```

| Metric | What it measures | Typical SLO target |
|---|---|---|
| TTFT | Time until first token appears (prefill speed) | p50 < 500ms, p99 < 2s |
| TBT | Time between each subsequent token (decode speed) | < 50ms for real-time UX |
| E2E Latency | Full response time | Depends on output length |
| Throughput | Tokens generated per second (system-wide) | Maximize this |
| Request Queue Depth | Pending requests waiting for a GPU | Alert > 10 |

---

### 5. Training vs Inference Infrastructure

These are fundamentally different problems. Know which you're solving.

```
Training Infrastructure          Inference Infrastructure
════════════════════════         ════════════════════════

Goal: minimize time to           Goal: minimize latency,
      train a model                    maximize throughput

Compute: sustained 100%          Compute: bursty, often <40%
         GPU utilization                  avg utilization

Memory: gradient storage +       Memory: weights + KV cache
        optimizer states                  (no gradients)
        (10-20x model size)

Networking: critical             Networking: moderate
            (all-reduce ops,               (model load at startup,
             EFA/InfiniBand)               light after)

Failure: tolerable with          Failure: user-facing, 
         checkpointing                    needs HA/fallback

Duration: hours to weeks         Duration: milliseconds
          (batch jobs)                    (latency-sensitive)

Your involvement: mostly         Your involvement: constant
                  setup                            on-call
```

---

## Resources

See [resources.md](./resources.md) for the full curated list with descriptions.

**Start with:**
1. Andrej Karpathy's "Let's Build GPT" — watch this before anything else
2. The Illustrated Transformer (Jay Alammar) — visual, intuitive
3. Tim Dettmers' GPU guide — the most honest breakdown of GPU memory math

---

## Lab

**[→ Lab 01: LLM Profiling](../../labs/01-llm-profiling/)**

Deploy llama.cpp on a GPU instance. Profile VRAM consumption at different quantization levels (Q4_K_M, Q6_K, Q8_0, F16). Measure TTFT and tokens/second for each. Convert to cost-per-1M-tokens. Build your first mental model with real numbers.
