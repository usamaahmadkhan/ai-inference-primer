# Phase 0: Jargon

> Read this before Phase 1. You will encounter every term here within your first week of working on AI infrastructure.

This is a reference, not a reading assignment. Skim it once to orient yourself. Return to specific entries when something in a later phase doesn't click.

---

## How to use this section

Terms are grouped by domain. Within each group, they build on each other — read top to bottom the first time.

- **Bold** = the term being defined
- `monospace` = code, config key, or exact CLI value
- ← = "this is caused by" or "this comes from"
- → = "this leads to" or "this affects"

---

## 1. Model Fundamentals

### Token

The atomic unit of text that LLMs process. Not a word — a sub-word chunk.

```
"Kubernetes networking" → ["Ku", "bern", "etes", " networking"]
                            tok1   tok2    tok3       tok4

"Hello" → ["Hello"]                          1 token
"antidisestablishmentarianism" → 7-9 tokens  (rare word = more tokens)
```

Rule of thumb: **1 token ≈ 0.75 English words ≈ 4 characters**. This matters for cost (you pay per token) and capacity (context windows are measured in tokens, not words).

---

### Context Window / Context Length

The maximum number of tokens a model can hold in "working memory" at once — both input and output combined.

```
Context window: 8,192 tokens
                │
                ├── Prompt (input):   1,000 tokens
                ├── History:          5,000 tokens
                └── Max output:       2,192 tokens remaining
```

**Why it matters for infra:** larger context windows require more VRAM for the KV cache. A 70B model at 128K context needs dramatically more memory than the same model at 4K context. This is one of the primary knobs you tune in vLLM.

---

### Parameters / Weights

The numerical values inside a model that encode its "knowledge." When someone says "a 7B model," they mean 7 billion parameters.

```
Model size in memory (approximate):
  parameters × bytes_per_parameter = VRAM needed

  7B model in FP16 (2 bytes/param):   7B × 2 = ~14 GB
  70B model in FP16 (2 bytes/param): 70B × 2 = ~140 GB
```

Parameters are stored as tensors. They're loaded from disk into GPU VRAM at startup. **They don't change during inference** — only during training/fine-tuning.

---

### Transformer

The neural network architecture that all modern LLMs use. Not important to understand mathematically — but you need to know its structure because it maps directly to what's happening in VRAM.

```
Transformer: repeated stacks of "layers"
═════════════════════════════════════════

Input tokens
     │
     ▼
┌──────────────────────┐
│   Embedding Layer    │  Converts token IDs → vectors
└──────────┬───────────┘
           │
           ▼ (repeated N times, e.g., 32 layers for a 7B model)
┌──────────────────────┐
│   Transformer Block  │
│                      │
│  ┌────────────────┐  │
│  │ Self-Attention │  │  ← where KV cache lives
│  └────────────────┘  │
│  ┌────────────────┐  │
│  │   FFN Layer    │  │  ← most compute happens here
│  └────────────────┘  │
└──────────┬───────────┘
           │
     (next layer)
           │
           ▼
┌──────────────────────┐
│   Output Layer       │  Converts final vectors → token probabilities
└──────────────────────┘
```

More layers = larger model = more parameters = more VRAM.

---

### Attention / Self-Attention

The mechanism inside each Transformer block that lets every token "look at" every other token in the context. This is what makes transformers so powerful — and so memory-hungry at long contexts.

```
Self-attention cost scales quadratically with sequence length:
  sequence length 1K  →  cost = 1K²  = 1M  operations
  sequence length 4K  →  cost = 4K²  = 16M operations (16× more)
  sequence length 32K →  cost = 32K² = 1B  operations (1000× more)
```

**Why it matters for infra:** this quadratic scaling is why long-context workloads are expensive, and why techniques like Flash Attention exist.

---

### Embedding

A fixed-length vector (list of numbers) that represents a token, word, sentence, or document. Used in two ways:

1. **Inside LLMs** — how tokens are represented internally as they flow through the model
2. **Embedding models** — standalone models (BERT, E5, BGE) that encode text into vectors for vector database storage (RAG use case)

```
"Kubernetes" → [0.23, -0.87, 0.41, 0.09, ... ] (768 or 1536 numbers)
"k8s"        → [0.22, -0.85, 0.43, 0.08, ... ] (similar vector = similar meaning)
"pizza"      → [0.91,  0.12, -0.33, 0.77, ... ] (very different vector)
```

---

## 2. Inference Mechanics

### Inference vs Training

| | Training | Inference |
|---|---|---|
| Goal | Update model weights | Generate output from fixed weights |
| Compute | Extremely high (forward + backward pass) | High (forward pass only) |
| Memory | 4–10× model size (gradients, optimizer state) | 1–1.5× model size (weights + KV cache) |
| Duration | Hours to weeks | Milliseconds per request |
| Your involvement | Setup, orchestrate | Constant on-call |

**Your job is almost entirely inference.** Training is a batch job you run occasionally. Inference is a latency-sensitive, always-on service.

---

### Prefill vs Decode

The two distinct phases of a single inference request. They have completely different performance characteristics.

```
Request: "Explain distributed systems in 3 sentences"
          └── 8 input tokens

PREFILL PHASE (compute-bound)
═════════════════════════════
Process all 8 input tokens in one parallel matrix operation.
Compute and store the KV cache entries for all input tokens.

Duration ∝ prompt length
GPU behavior: high compute utilization
This phase ends when → first output token is generated (TTFT)

DECODE PHASE (memory bandwidth-bound)
══════════════════════════════════════
Generate output tokens one at a time.
Each step: read entire KV cache + compute one new token.

Token 1: "Distributed"
Token 2: "systems"       } each token: one forward pass,
Token 3: "are"           } reads entire KV cache
Token 4: "networks"
...

Duration ∝ output length
GPU behavior: memory bandwidth saturated, compute underutilized
This phase ends when → EOS token generated or max_tokens reached
```

**Why it matters:** most GPU time is spent in decode. Throughput optimization (tokens/sec) is primarily a decode problem.

---

### KV Cache

The most important data structure in LLM serving. Stores intermediate attention computation results so they don't need to be recomputed every decode step.

```
Without KV cache:
  For each new token, recompute attention over ALL previous tokens
  Cost grows quadratically with sequence length → unusably slow

With KV cache:
  Compute attention for new token only
  Read cached K,V from previous tokens from VRAM
  Cost is linear in sequence length → practical

KV cache lives in VRAM. Size formula:
  2 × num_layers × num_heads × head_dim × sequence_length × bytes_per_element
  
  For Llama-3-8B at 4K context, FP16:
  2 × 32 × 8 × 128 × 4096 × 2 = ~536 MB per request

  At 100 concurrent requests: 53 GB — easily larger than the model itself
```

**This is why you run out of VRAM under load even when the model fits fine at idle.** PagedAttention (vLLM) exists to manage this efficiently.

---

### PagedAttention

vLLM's core innovation. Manages KV cache like an OS manages virtual memory — in fixed-size pages, allocated on demand, not pre-allocated per request.

```
Problem: pre-allocating KV cache per request wastes VRAM
          because you don't know output length in advance.

PagedAttention solution:
  KV cache split into fixed 16-token "pages"
  Pages allocated only as tokens are generated
  Non-contiguous pages linked like a page table

  Request A (short output):  uses 3 pages → returns 3 pages to pool
  Request B (long output):   gets those 3 pages → uses 12 pages total

Result: ~2–4× throughput improvement over naive KV cache management
```

---

### Continuous Batching

Serving technique where new requests are added to the active batch as previous requests finish — without waiting for the whole batch to complete.

```
Static batching (old):              Continuous batching (vLLM default):
Batch[A,B,C] → all finish           A finishes → D joins immediately
→ Batch[D,E,F]                      E finishes → F joins immediately
→ GPU idle between batches          → GPU never idle between requests
```

**You don't configure this** — vLLM does it automatically. But you need to know it exists so you understand why vLLM's throughput is so much higher than naive deployments.

---

### Quantization

Reducing the numerical precision of model weights to save VRAM and increase throughput at the cost of some accuracy.

```
Precision   Bytes/param   70B model VRAM   Quality loss
─────────────────────────────────────────────────────────
FP32           4            280 GB          None (baseline)
FP16           2            140 GB          Negligible
BF16           2            140 GB          Negligible (better for training)
INT8           1             70 GB          Small (most tasks: unnoticeable)
GPTQ (4-bit)   0.5           35 GB          Moderate (task-dependent)
AWQ  (4-bit)   0.5           35 GB          Small (better than GPTQ at 4-bit)
GGUF (varies)  0.5–1         35–70 GB        Varies by quant level
```

**FP16/BF16** — default for production when you have the VRAM.
**AWQ** — current best 4-bit method. Use when you need to fit on smaller GPUs.
**GGUF** — llama.cpp format. Dev/edge use. Not the format vLLM uses.

---

### TTFT (Time to First Token)

The latency from when the request arrives to when the first output token is generated. Measures prefill speed.

```
User sends request
      │
      │← Queue wait (request waited for a free slot)
      │
      Prefill starts (processing input tokens)
      │
      │← TTFT ends here
      │
      First token arrives at client ← user sees something
      │
      Second token...
      Third token...
```

**This is your streaming UX metric.** Users tolerate high E2E latency if TTFT is fast — the response "feels" fast even if it takes time to complete. Target: p50 < 500ms, p99 < 2s for conversational UX.

---

### TBT / ITL (Time Between Tokens / Inter-Token Latency)

Time between consecutive output tokens during the decode phase. Determines perceived streaming smoothness.

Target: < 30–50ms for real-time streaming UX. Above 100ms and users notice stutter.

---

### Throughput

Tokens generated per second across all concurrent requests. The system-level productivity metric.

```
Throughput = total_output_tokens / time_window

High throughput + high TTFT = saturated GPU (need to scale out)
Low throughput + low TTFT   = underutilized GPU (cost waste)
```

---

## 3. GPU Hardware

### VRAM (Video RAM / HBM)

GPU memory. Where model weights, KV cache, and activations live during inference. The primary constraint in LLM serving.

```
VRAM budget for a g5.2xlarge (A10G, 24GB):
  Model weights (Llama-3-8B FP16): ~16 GB
  GPU OS / CUDA overhead:          ~0.5 GB
  Available for KV cache:          ~7.5 GB   ← limits max concurrent requests
```

When VRAM is exhausted: **CUDA OOM error**, process crashes. Not graceful degradation — hard crash.

---

### HBM (High Bandwidth Memory)

The specific DRAM technology used in data center GPUs (A100, H100). Much faster than consumer GPU GDDR6.

```
A100 HBM2e:  2 TB/s bandwidth
H100 HBM3:   3.35 TB/s bandwidth
A10G GDDR6:  600 GB/s bandwidth   ← 5× slower than A100
```

Memory bandwidth, not raw compute (TFLOPS), is usually the bottleneck for inference decode.

---

### CUDA / CUDA Cores

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform. CUDA cores are the GPU's general-purpose compute units. Every GPU operation in deep learning runs through CUDA.

When you see "CUDA OOM" — it means the GPU ran out of VRAM, not CPU RAM.

---

### Tensor Cores

Specialized hardware inside NVIDIA GPUs designed for matrix multiplication — the core operation in deep learning. Much faster than CUDA cores for this specific operation.

```
A100 Tensor Core performance:
  FP16: 312 TFLOPS   (with Tensor Cores)
  FP32:  19 TFLOPS   (CUDA cores only)

Speedup: ~16× for matrix multiply operations
```

All modern inference frameworks use Tensor Cores by default with FP16/BF16 precision.

---

### SM (Streaming Multiprocessor)

The basic processing unit of a GPU. A GPU contains many SMs; each SM contains CUDA cores, Tensor Cores, and shared memory.

```
H100 SXM: 132 SMs
  Each SM: 128 CUDA cores + 4 Tensor Core groups + 256KB L1/shared memory

SM Occupancy: how many threads are actively running vs max capacity
  Low occupancy (<50%): GPU is underutilized — small batch sizes
  High occupancy (>80%): GPU is well-utilized
```

---

### NVLink

NVIDIA's high-bandwidth GPU-to-GPU interconnect. Required for efficient tensor parallelism across multiple GPUs.

```
NVLink vs PCIe (for multi-GPU communication):
  NVLink (A100):  600 GB/s bidirectional  ← tensor parallel works well
  PCIe Gen4:       64 GB/s bidirectional  ← tensor parallel bottlenecked

Multi-GPU tensor parallelism on PCIe: ~2–3× slower than NVLink
p4d/p5 instances use NVLink. g5 instances use PCIe.
```

---

### MIG (Multi-Instance GPU)

Hardware partitioning of A100/H100 GPUs into isolated slices. Each slice has its own VRAM, compute, and memory bandwidth — full hardware isolation.

```
A100 80GB → up to 7× MIG slices:
  7× 1g.10gb  (10 GB VRAM each)  ← 7 isolated tenants
  3× 2g.20gb  (20 GB VRAM each)
  2× 3g.40gb  (40 GB VRAM each)
  1× 7g.80gb  (full GPU, no MIG)
```

Different from time-slicing (software sharing). MIG is true hardware isolation — one tenant cannot affect another's performance or crash their workload.

---

## 4. Serving Infrastructure

### Inference Server

Software that loads model weights and exposes them via an API. Your primary tool is vLLM.

```
Model weights (files on disk / S3)
        │  loaded at startup (~minutes)
        ▼
Inference Server (vLLM, TGI, Triton)
        │  HTTP / gRPC
        ▼
Your application
```

---

### Tensor Parallelism

Splitting a single model's weight matrices across multiple GPUs so a model that doesn't fit on one GPU can fit across several.

```
Without tensor parallel:           With tensor parallel (2 GPUs):
One GPU needs 140GB for 70B        GPU 0: half of each weight matrix
→ impossible on single A100        GPU 1: other half
                                   → 70GB each, feasible on A100 80GB

vLLM config: tensor_parallel_size: 2
Requirement: GPUs must be on same node (NVLink preferred)
```

---

### DCGM (Data Center GPU Manager)

NVIDIA's toolkit for monitoring GPU health and performance metrics. DCGM Exporter exposes these metrics in Prometheus format.

Think of it as `node_exporter` for GPUs.

---

### GPU Operator

Kubernetes operator that automates the installation and lifecycle management of NVIDIA drivers, container toolkit, device plugin, and DCGM on K8s GPU nodes.

**Without it:** you manually install drivers on every GPU node, manage upgrades, configure containerd. Painful.
**With it:** one Helm install, everything managed as DaemonSets.

---

### Device Plugin

Kubernetes component (DaemonSet) that advertises non-standard resources to the scheduler. For GPUs: advertises `nvidia.com/gpu` so pods can request GPU resources.

```yaml
resources:
  limits:
    nvidia.com/gpu: 1   # ← only works because device plugin is running
```

---

## 5. LLMOps

### RAG (Retrieval-Augmented Generation)

Architecture that gives an LLM access to external knowledge by retrieving relevant documents and injecting them into the prompt.

```
Standard LLM:           RAG:
User question           User question
      │                       │
      ▼                       ▼
     LLM                Vector DB search
      │                       │ top-K relevant docs
      ▼                       ▼
   Answer              LLM + retrieved docs
 (may hallucinate)           │
                             ▼
                      Answer grounded in real data
```

Infrastructure components: embedding model, vector database, retrieval pipeline.

---

### Fine-tuning

Continuing to train a pre-trained model on domain-specific data so it learns new behaviors or knowledge.

```
Pre-trained model (general knowledge)
        │
        + your training data (domain specific)
        │
        ▼
Fine-tuned model (general + domain)
```

**LoRA (Low-Rank Adaptation)** — fine-tuning technique that trains only a small adapter (~50–200MB) instead of updating all weights. Makes fine-tuning practical on 1–2 GPUs instead of dozens.

---

### LoRA / QLoRA

- **LoRA** — fine-tuning by training small adapter matrices, leaving base model frozen. Result: base model + adapter file. Can hot-swap adapters without reloading base model.
- **QLoRA** — LoRA with the base model quantized to 4-bit during training. Fits 70B fine-tuning on 2× A100.

---

### Model Registry

Versioned artifact store for trained model files. Same concept as a container registry, but for model weights.

```
Container registry: image → tag → digest
Model registry:     model → version → artifact (weights file + metadata)

MLflow, Weights & Biases Registry, HuggingFace Hub are common choices.
```

---

### Prompt

The input text (plus any system instructions) sent to an LLM. In an API context:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "Explain Kubernetes"}
  ]
}
```

The "system" message is prepended to every request and is invisible to the user. Prompt length directly affects TTFT and KV cache usage.

---

### Guardrails

Input/output filters applied around an LLM to prevent harmful, off-topic, or policy-violating content. From an infra perspective: another latency-adding step in your serving pipeline.

---

## 6. AWS Specific

### DLAMI (Deep Learning AMI)

AWS-provided EC2 AMI pre-configured with NVIDIA drivers, CUDA, PyTorch, and other ML frameworks. Use this as your base AMI for GPU EC2 nodes to avoid manual driver installation.

---

### DLC (Deep Learning Containers)

AWS-provided Docker images pre-built with ML frameworks optimized for AWS hardware. HuggingFace TGI on SageMaker uses DLC images. Saves you maintaining your own CUDA-compatible base images.

---

### EFA (Elastic Fabric Adapter)

AWS's high-performance network interface for HPC and ML workloads. Required for multi-node distributed training at scale.

```
Without EFA: standard Ethernet (10–25 Gbps)  → training bottlenecked on network
With EFA:    100 Gbps+ OS-bypass network      → near-linear multi-node scaling

Available on: p3dn, p4d, p5 instances
Not available on: g4dn, g5, g6 instances
```

---

### Trainium / Inferentia

AWS custom silicon for ML workloads. Alternative to NVIDIA GPUs.

- **Trainium (Trn1)** — optimized for training
- **Inferentia (Inf2)** — optimized for inference, ~30–40% cheaper than equivalent GPU inference

Catch: requires compiling your model with the AWS Neuron SDK. Not a drop-in replacement.

---

## Quick Reference Card

```
Term                    One-line definition
────────────────────────────────────────────────────────────────────
Token                   Sub-word unit; ~0.75 words; unit of cost
Context window          Max tokens (input + output) model can process
Parameters              Model's learned numerical values; "weights"
VRAM                    GPU memory; primary constraint in inference
Prefill                 Processing input tokens; compute-bound
Decode                  Generating output tokens one at a time; memory-bound
KV Cache                Stored attention states; lives in VRAM; grows with requests
PagedAttention          vLLM's virtual-memory-style KV cache management
Continuous batching     Adding new requests mid-batch as others finish
Quantization            Reducing weight precision to save VRAM
TTFT                    Latency to first output token; UX metric
TBT / ITL               Time between tokens; streaming smoothness
Throughput              Tokens/sec system-wide; capacity metric
Tensor Parallel         Splitting model across multiple GPUs
MIG                     Hardware GPU partitioning (A100/H100 only)
NVLink                  High-bandwidth GPU-GPU interconnect
DCGM                    GPU hardware monitoring; feeds Prometheus
GPU Operator            K8s operator for NVIDIA driver lifecycle
LoRA                    Efficient fine-tuning via small adapter weights
RAG                     Retrieval-augmented generation; LLM + vector search
DLAMI                   AWS AMI with ML dependencies pre-installed
EFA                     AWS high-performance network for multi-node training
Inferentia              AWS custom chip; cheaper inference than GPU
```
