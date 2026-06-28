# Phase 4: AWS AI Stack

> Cloud-native AI infrastructure patterns on AWS. Know what you're paying for and when the managed services are actually worth it.

**Duration:** 3–4 weeks
**Lab:** [04-aws-comparison](../../labs/04-aws-comparison/)

---

## The AWS AI infrastructure decision tree

```
New AI workload arrives
         │
         ▼
  ┌──────────────────────────────────────┐
  │ Do you need to control the model?    │
  │ (fine-tuning, custom weights,        │
  │  data privacy requirements)          │
  └──────────────────────────────────────┘
         │ Yes                     │ No
         ▼                         ▼
  ┌─────────────────┐      ┌──────────────────┐
  │ Self-hosted on  │      │ Amazon Bedrock   │
  │ EKS + vLLM      │      │ (API only, no    │
  │                 │      │  infra to run)   │
  │ Full control    │      └──────────────────┘
  │ Full ops burden │
  └─────────────────┘
         │
         ▼
  ┌──────────────────────────────────────┐
  │ Do you want managed compute?         │
  │ (no K8s, no GPU driver ops,          │
  │  willing to trade control for UX)    │
  └──────────────────────────────────────┘
         │ Yes                     │ No
         ▼                         ▼
  ┌─────────────────┐      ┌──────────────────┐
  │  SageMaker      │      │  EKS + vLLM      │
  │  Endpoint       │      │  (your stack,    │
  │                 │      │   full control)  │
  │  Easier ops     │      └──────────────────┘
  │  Vendor lock-in │
  │  Cost premium   │
  └─────────────────┘
```

---

## EC2 GPU Instance Families

Know your hardware before you provision it. GPU availability, pricing, and appropriate workloads vary significantly across families.

```
AWS GPU Instance Decision Matrix (2025)
════════════════════════════════════════════════════════════════

Instance       GPU           VRAM    Use Case
───────────────────────────────────────────────────────────────
g4dn.xlarge    T4 (1x)       16 GB   Dev, small models <7B
g4dn.12xlarge  T4 (4x)       64 GB   Multi-model dev
g5.xlarge      A10G (1x)     24 GB   7B–13B production, cheap
g5.12xlarge    A10G (4x)     96 GB   30B or 4x parallel 7B
g5.48xlarge    A10G (8x)    192 GB   70B model, tensor parallel
g6.xlarge      L4 (1x)       24 GB   Inference-optimized, ~20% 
                                      cheaper than g5, newer
p4d.24xlarge   A100 (8x)    320 GB   Large models, high throughput
p4de.24xlarge  A100 (8x)    640 GB   Full 80GB A100, fine-tuning
p5.48xlarge    H100 (8x)    640 GB   Frontier models, max perf
               ↑
               Hardest to get — 3–6mo capacity reservation wait
               $98/hr on-demand, $30–40/hr spot

Spot availability (approximate):
g4dn: ████████░░ 80%    g5: ██████░░░░ 60%    p4d: ████░░░░░░ 40%
p5:   ██░░░░░░░░ 20%    ← plan for on-demand for p5 baselines
```

### VRAM sizing rule of thumb

```
Model VRAM requirements (FP16, no quantization)
═══════════════════════════════════════════════
7B  model  →  ~14 GB  (fits g4dn, g5, g6 with room for KV cache)
13B model  →  ~26 GB  (needs g5 or larger; g4dn won't work)
30B model  →  ~60 GB  (needs 2x A10G or 1x A100 40GB)
70B model  →  ~140 GB (needs 2x A100 80GB or 4x A10G with tensor parallel)
405B model →  ~810 GB (needs 8x H100 or 16x A100)

KV cache uses remaining VRAM. More concurrent requests = more VRAM for KV.
Underprovision VRAM → OOM crashes under load.
```

---

## Amazon SageMaker

SageMaker abstracts GPU node management in exchange for less control and higher cost. Know the tradeoffs before committing.

```
SageMaker Inference Modes
══════════════════════════

Real-time Endpoint          Serverless Inference
────────────────────         ────────────────────
Always-on instance           Pay per invocation
Low latency (<100ms)         Cold start: 5–30s
Best for: prod APIs          Best for: bursty, 
Predictable cost             low-traffic workloads

Async Inference              Batch Transform
────────────────────         ────────────────────
Request → S3 queue           Dataset → S3 → process
→ process → S3 result        → S3 results
Best for: long-running       Best for: offline
          inference jobs               scoring jobs
No timeout limits            Very cost-effective
```

### Real-time endpoint deployment

```python
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

# Deploy via HuggingFace TGI DLC (Deep Learning Container)
hub = {
    'HF_MODEL_ID': 'meta-llama/Llama-3.1-8B-Instruct',
    'SM_NUM_GPUS': json.dumps(1),
    'MAX_INPUT_LENGTH': json.dumps(4096),
    'MAX_TOTAL_TOKENS': json.dumps(8192),
}

huggingface_model = HuggingFaceModel(
    image_uri=sagemaker.image_uris.retrieve(
        framework="huggingface-llm",
        region="us-east-1",
        version="2.0.1"
    ),
    env=hub,
    role=sagemaker.get_execution_role(),
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge",
    container_startup_health_check_timeout=300,
)
```

### SageMaker cost vs EKS

```
Cost comparison: serving Llama-3.1-8B at 100 req/min
══════════════════════════════════════════════════════

EKS (g5.2xlarge, on-demand):
  EC2: $1.21/hr
  EKS cluster overhead: ~$0.10/hr
  Total: ~$1.31/hr (~$960/mo)
  ✓ Full control, observable, portable

SageMaker (ml.g5.2xlarge):
  Endpoint: $1.83/hr  (51% premium over raw EC2)
  Total: ~$1.83/hr (~$1,340/mo)
  ✓ Managed, less ops
  ✗ Vendor lock-in, less observable

Bedrock (Claude 3 Haiku, equivalent quality):
  ~$0.00025 per 1K input tokens
  ~$0.00125 per 1K output tokens
  At 100 req/min × 500 avg tokens:
  ~$3.75/hr (~$2,750/mo)  ← most expensive per token
  ✓ Zero ops, no GPU management
  ✗ No model control, highest cost at scale

Break-even: EKS wins above ~20 req/min sustained.
            Bedrock wins below ~5 req/min.
```

---

## Amazon Bedrock

Bedrock is the right choice when you don't want to run infrastructure and don't need model control.

```
When to use Bedrock
═══════════════════

✓ Prototyping / early-stage products
✓ Infrequent or unpredictable traffic
✓ You need Claude, GPT-4-class models (not self-hostable)
✓ Compliance requirements (AWS handles the model infra)
✓ Team without GPU infra expertise

✗ High-volume, sustained traffic (cost)
✗ Need to fine-tune the model
✗ Strict data residency for inference payload
✗ Need sub-100ms TTFT
✗ Custom model architectures
```

---

## AWS Neuron: Trainium and Inferentia

AWS custom silicon. Often overlooked, worth knowing.

```
Neuron Instance Family
═══════════════════════

trn1.2xlarge    (Trainium)   →  Fine-tuning, training jobs
trn1.32xlarge   (Trainium)   →  Large model fine-tuning

inf2.xlarge     (Inferentia) →  Inference, small models
inf2.48xlarge   (Inferentia) →  Large model inference

Cost advantage: ~30–40% cheaper than GPU equivalents for inference
Catch: Must compile your model with AWS Neuron SDK first

# Compilation adds a one-time cost
# Compiled model is hardware-specific
# Not a drop-in replacement for GPU workloads
```

The Neuron SDK compiles PyTorch models to run on the custom chips. If you're cost-sensitive and running high sustained inference volumes, Inferentia is worth the porting effort. Start with GPU until you've validated your workload.

---

## Storage Patterns for Models

Model loading time is a frequently ignored bottleneck. A 70B model at FP16 is ~140GB. Loading from S3 to GPU VRAM on startup affects cold start time.

```
Storage Tier Decision
═════════════════════

S3
├── Use for: model artifact storage, versioning
├── Load time: 140GB at ~200MB/s = ~12 min cold start
└── Cost: ~$0.023/GB/mo (cheapest)

EFS (NFS)
├── Use for: shared model cache across pods
├── Load time: 140GB at ~500MB/s = ~5 min
├── Benefit: pod restarts reuse warm cache
└── Cost: ~$0.30/GB/mo

FSx for Lustre (high-perf parallel FS)
├── Use for: training data, high-throughput fine-tuning
├── Load time: 140GB at ~1-4GB/s = ~1 min
├── Can be backed by S3 with lazy load
└── Cost: ~$0.14/GB/mo

Local NVMe SSD (on instance)
├── Use for: model cache for single-node deployments
├── Load time: 140GB at ~3GB/s = ~45 sec
├── Lost on node termination
└── Cost: included in instance price

Recommended pattern:
  S3 (source of truth) → EFS (shared cache) → GPU VRAM
  Pods mount EFS; first load populates cache; subsequent pods reuse it
```

---

## Resources

See [resources.md](./resources.md) for the full curated list.

**Essential:**
1. AWS ML Blog — the canonical source for AWS-specific AI infra patterns
2. SageMaker MLOps Workshop — hands-on, follows real production patterns
3. AWS Neuron SDK Docs — if cost optimization is a priority

---

## Lab

**[→ Lab 04: AWS Comparison](../../labs/04-aws-comparison/)**

Deploy the same 8B model three ways: EKS + vLLM, SageMaker real-time endpoint, Bedrock. Measure cold start, TTFT p50/p99, cost per 1M tokens, and operational burden. Build a decision matrix you'd actually use in a design review. This exercise removes abstract arguments from infrastructure decisions — you'll have real numbers.
