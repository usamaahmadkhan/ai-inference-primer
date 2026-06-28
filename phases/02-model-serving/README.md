# Phase 2: Model Serving

> The runtime between your infrastructure and the model weights.

**Duration:** 3–4 weeks
**Lab:** [02-vllm-deployment](../../labs/02-vllm-deployment/)

---

## The landscape

```
Model Serving Ecosystem (2025)
══════════════════════════════

               ┌─────────────────────────────────────┐
               │          YOUR APPLICATION            │
               │   (OpenAI-compatible REST API)       │
               └────────────────┬────────────────────┘
                                │  HTTP/gRPC
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
    ┌────┴─────┐          ┌─────┴────┐          ┌──────┴────┐
    │  vLLM    │          │   TGI    │          │  Triton   │
    │          │          │          │          │  Server   │
    │ PagedAttn│          │ HuggingF │          │  NVIDIA   │
    │ Cont.    │          │ ace      │          │  Generic  │
    │ Batching │          │          │          │           │
    └────┬─────┘          └─────┬────┘          └──────┬────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │      GPU + VRAM       │
                    │   (model weights +    │
                    │     KV cache)         │
                    └───────────────────────┘

Use vLLM by default. Consider TGI for HuggingFace-native workflows.
Consider Triton for non-LLM model serving (CV, recommenders).
```

---

## vLLM (your primary tool)

vLLM is the current industry standard for LLM inference serving. If you only learn one serving framework, make it this one.

### Why vLLM won

The problem with naive LLM serving: each request pre-allocates a fixed VRAM block for its KV cache, even though you don't know output length in advance. This wastes memory, kills throughput, and causes OOM crashes on variable-length workloads.

**PagedAttention** (vLLM's core innovation) treats KV cache like virtual memory — paging it in/out in fixed-size blocks, allocated on demand.

```
Naive KV Cache (before vLLM)        PagedAttention (vLLM)
══════════════════════════════       ══════════════════════════════

VRAM                                 VRAM
┌─────────────────────────┐          ┌─────────────────────────┐
│ Request A  [████████░░] │ ← waste  │ Block 0  [Req A token 1]│
│ Request B  [███░░░░░░░] │ ← waste  │ Block 1  [Req A token 2]│
│ Request C  [████████████│ ← full   │ Block 2  [Req B token 1]│
│            │            │          │ Block 3  [Req C token 1]│
│ (request D │ rejected)  │          │ Block 4  [Req D token 1]│ ← fits!
└─────────────────────────┘          └─────────────────────────┘

Result: low GPU utilization          Result: 2-4x higher throughput
        frequent OOM failures                near-zero memory waste
```

### Continuous Batching

```
Static Batching (old way)           Continuous Batching (vLLM)
═══════════════════════════         ════════════════════════════

Batch starts → all finish           Requests join/leave mid-batch
→ next batch                        as they complete

Time ──────────────────────►        Time ──────────────────────►

[A████████████████]  wait           [A████████████████]
[B██████] wait wait  wait           [B██████][D████████████]
[C████████████] wait wait     [C████████████]
                   [D...]              [E██][F████]

GPU: ████░░░░████░░░░████           GPU: ████████████████████
     (idle between batches)              (near-continuous)
```

### Key vLLM configuration knobs

```yaml
# The settings that matter most

engine_args:
  model: "meta-llama/Llama-3.1-8B-Instruct"

  # GPU memory to reserve for KV cache (rest goes to model weights)
  # Tune this first when you hit OOM or low throughput
  gpu_memory_utilization: 0.90  # 0.85–0.95 is the practical range

  # Tensor parallelism: split model across N GPUs
  # Required when model doesn't fit on one GPU
  tensor_parallel_size: 2  # for 70B on 2x A100 80GB

  # Maximum sequence length (prompt + output)
  # Higher = more VRAM for KV cache
  max_model_len: 8192

  # Maximum concurrent sequences
  # vLLM manages this automatically, but you can cap it
  max_num_seqs: 256

  # Quantization
  quantization: "awq"  # null | "awq" | "gptq" | "squeezellm"
  dtype: "auto"         # auto | float16 | bfloat16
```

### Tensor Parallelism: when and how

```
Single GPU (fits in VRAM)           Tensor Parallel (too big for 1 GPU)
══════════════════════════          ════════════════════════════════════

     GPU 0                               GPU 0        GPU 1
  ┌─────────┐                         ┌─────────┐  ┌─────────┐
  │  Full   │                         │ Layer   │  │ Layer   │
  │  Model  │  ← works fine           │ shard   │  │ shard   │
  │         │                         │  (½)    │  │  (½)    │
  └─────────┘                         └────┬────┘  └────┬────┘
                                           │  NVLink    │
                                           └─────┬──────┘
                                                 │
                                          combined output

# vLLM does this automatically:
tensor_parallel_size: 2

# Requires NVLink for good performance.
# PCIe multi-GPU tensor parallelism works but ~2x slower.
```

---

## Text Generation Inference (TGI)

HuggingFace's serving stack. Solid choice when:
- Your team is already in the HuggingFace ecosystem
- You need tight integration with HuggingFace Hub model management
- You're running on SageMaker (AWS DLC includes TGI)

```bash
# TGI is a single Docker container, dead simple to start
docker run --gpus all \
  -e MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  -e NUM_SHARD=1 \
  -e MAX_INPUT_LENGTH=4096 \
  -e MAX_TOTAL_TOKENS=8192 \
  -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest
```

**TGI vs vLLM — when to choose which:**

| Factor | vLLM | TGI |
|---|---|---|
| Raw throughput | Slightly higher | Slightly lower |
| HuggingFace Hub integration | Good | Native |
| OpenAI API compatibility | Native | Requires adapter |
| SageMaker DLC support | DIY | AWS-native |
| Community/ecosystem | Larger | HF-backed |
| Active development pace | Very fast | Fast |

---

## NVIDIA Triton Inference Server

Use Triton when you're serving a pipeline of models, not just a single LLM.

```
Multi-model serving with Triton
════════════════════════════════

Incoming request
     │
     ▼
┌────────────────┐
│  Ensemble      │  Triton orchestrates
│  Pipeline      │  multiple models
└────────────────┘
     │
     ├──► [Text Embedding Model]  (BERT, E5, etc.)
     │
     ├──► [Reranker Model]        (cross-encoder)
     │
     ├──► [LLM]                   (Llama, Mistral, etc.)
     │
     └──► [Post-processor]        (classifier, filter)
```

Triton is overkill for pure LLM serving. Use it if you have multi-model inference pipelines or need to serve TensorRT, ONNX, and PyTorch models side by side.

---

## Ray Serve

When you need programmatic routing, dynamic model loading, or heterogeneous serving at scale.

```python
# Ray Serve: useful for complex routing logic
@serve.deployment
class LLMRouter:
    def __init__(self):
        self.small_model = serve.get_deployment("llama-8b")
        self.large_model = serve.get_deployment("llama-70b")

    async def __call__(self, request):
        # Route based on request complexity, user tier, cost budget
        if request.json()["tier"] == "premium":
            return await self.large_model.remote(request)
        return await self.small_model.remote(request)
```

Use Ray Serve when you need:
- A/B routing between model versions
- Dynamic model loading/unloading
- Complex multi-step pipelines with Python logic between steps
- Heterogeneous hardware routing (different requests to different GPU types)

---

## Ollama (dev only)

Dead simple local deployment. Use it to prototype, not to serve production traffic.

```bash
ollama run llama3.1:8b
```

Why not production: no batching, no tensor parallelism, no SLA-grade observability, no horizontal scaling story. The simplicity is its value and its limitation.

---

## Resources

See [resources.md](./resources.md) for the full curated list.

**Must-read:**
1. The vLLM PagedAttention paper — read it, it's short and explains everything
2. vLLM docs: `gpu_memory_utilization` and `max_num_seqs` sections specifically

---

## Lab

**[→ Lab 02: vLLM Deployment](../../labs/02-vllm-deployment/)**

Deploy vLLM serving Llama-3.1-8B. Run a load test. Find the batch size where throughput peaks before TTFT degrades. Document your saturation point. This number will define your capacity planning for everything that comes after.
