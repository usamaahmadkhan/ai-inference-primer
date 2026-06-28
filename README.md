# AI Inference Primer

> A field guide for Site Reliability Engineers stepping into AI infrastructure.
> Written for engineers who already know Kubernetes, AWS, and distributed systems —
> and want to operate AI models in production without starting from zero.

*Inspired by the [System Design Primer](https://github.com/donnemartin/system-design-primer). Same philosophy: deep, practical, no hand-holding.*

---

## Why this exists

Every company is deploying LLMs. Most are doing it wrong — treating model serving like a stateless API, ignoring GPU memory pressure, and discovering their SLOs don't map to inference metrics until 3am.

SREs are the right people to fix this. You already understand failure modes, capacity planning, and the difference between what a dashboard shows and what's actually happening. You just need the AI-specific vocabulary and tooling layer.

This guide gives you that, in the order it matters.

---

## How to use this guide

**If you have 2–3 hours/week** → Work through phases sequentially. Do the lab at the end of each phase before moving on. The lab is where it sticks.

**If you're preparing for a role** → Start with the [cheatsheets](./cheatsheets/) to get the vocabulary, then go deep on phases 2, 3, and 6.

**If you're debugging a production incident** → Jump directly to [Phase 6: AI Observability](./phases/06-ai-observability/).

**If you're designing a new platform** → Read phases 4 and 7, then the [configs](./configs/) directory.

---

## Study guide by time budget

| Time Available | Focus |
|---|---|
| 30 min | [Cheatsheets](./cheatsheets/) — vocabulary and quick reference |
| 1 week | Phases 1–2: Mental models + vLLM |
| 1 month | Phases 1–4: Full serving + K8s + AWS |
| 3 months | All 7 phases with labs |
| Ongoing | Phase 7 + community links |

---

## The Learning Path

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI INFERENCE PRIMER                          │
│                  SRE → AI Infrastructure                        │
└─────────────────────────────────────────────────────────────────┘

Phase 1          Phase 2          Phase 3          Phase 4
Mental           Model            K8s for          AWS AI
Models    ──►    Serving   ──►    AI        ──►    Stack
(2-3w)           (3-4w)           (4-5w)           (3-4w)
                                                      │
                                                      ▼
Phase 7          Phase 6          Phase 5          Phase 4
Advanced  ◄──    AI Obs    ◄──    LLMOps    ◄──   AWS AI
Topics           (2-3w)           (3-4w)           Stack
(ongoing)                                          (3-4w)

Total: ~22–30 weeks at serious part-time pace
```

---

## Phase Index

### [Phase 1: Mental Models](./phases/01-mental-models/) `2–3 weeks`
Understand what you're operating before you operate it. GPU architecture, inference mechanics, the KV cache, quantization. If you skip this phase, you'll be debugging blindly.

### [Phase 2: Model Serving](./phases/02-model-serving/) `3–4 weeks`
vLLM, TGI, Triton, Ray Serve. The runtime layer between your infrastructure and the model weights. vLLM is the current standard — go deep on it.

### [Phase 3: Kubernetes for AI](./phases/03-kubernetes-for-ai/) `4–5 weeks`
GPU Operator, device plugins, MIG partitioning, Karpenter with GPU nodes, KubeRay, DCGM Exporter. Running GPU workloads on K8s is materially different from running CPU workloads.

### [Phase 4: AWS AI Stack](./phases/04-aws-ai-stack/) `3–4 weeks`
EC2 GPU families, SageMaker, Bedrock, Trainium/Inferentia, storage patterns. Know the managed vs self-managed tradeoffs in AWS specifically.

### [Phase 5: LLMOps](./phases/05-llmops/) `3–4 weeks`
Model registries, deployment strategies, fine-tuning pipelines, CI/CD for models. The lifecycle management layer that most teams build last and should build first.

### [Phase 6: AI Observability](./phases/06-ai-observability/) `2–3 weeks`
DCGM metrics, inference SLIs, LLM-specific tracing, cost attribution. Your existing Prometheus/Grafana stack gets extended, not replaced.

### [Phase 7: Advanced Topics](./phases/07-advanced-topics/) `Ongoing`
Vector databases, distributed training infra, speculative decoding, AI agent infrastructure, multi-tenant LLM platforms.

---

## Labs

Each phase has a corresponding lab. Don't skip them.

| Lab | What You Build | Phase |
|---|---|---|
| [01: LLM Profiling](./labs/01-llm-profiling/) | llama.cpp on GPU, quant benchmarking | 1 |
| [02: vLLM Deployment](./labs/02-vllm-deployment/) | vLLM + load test + saturation analysis | 2 |
| [03: EKS GPU Cluster](./labs/03-eks-gpu-cluster/) | EKS + GPU Operator + DCGM + Karpenter | 3 |
| [04: AWS Comparison](./labs/04-aws-comparison/) | EKS vs SageMaker vs Bedrock decision matrix | 4 |
| [05: Fine-Tuning Pipeline](./labs/05-fine-tuning-pipeline/) | LoRA → eval gate → MLflow → auto-deploy | 5 |
| [06: Observability Stack](./labs/06-observability-stack/) | GPU metrics + LLM tracing + cost dashboard | 6 |
| [07: LLM Gateway](./labs/07-llm-gateway/) | Multi-tenant gateway with rate limiting + chargeback | 7 |

---

## Configs

Production-ready configuration templates. Not toy examples.

```
configs/
├── vllm/
│   ├── deployment.yaml        # K8s Deployment with GPU resources + probes
│   └── hpa.yaml               # HPA on custom inference queue metric
├── kubernetes/
│   ├── gpu-operator/
│   │   └── values.yaml        # GPU Operator Helm values (EKS-tuned)
│   ├── dcgm/
│   │   └── values.yaml        # DCGM Exporter with ServiceMonitor
│   └── karpenter/
│       └── gpu-nodepool.yaml  # GPU NodePool for mixed on-demand/spot
└── terraform/
    └── eks-gpu/               # EKS cluster with GPU node groups
        ├── main.tf
        ├── variables.tf
        └── outputs.tf
```

---

## Cheatsheets

Quick references for when you need an answer in 60 seconds.

- [GPU Instances](./cheatsheets/gpu-instances.md) — AWS GPU families, VRAM, use cases, cost
- [Inference Metrics](./cheatsheets/inference-metrics.md) — TTFT, TBT, throughput, what to SLO on
- [kubectl for GPU](./cheatsheets/kubectl-gpu.md) — Commands you'll run constantly

---

## Prerequisite knowledge

This guide assumes you're already comfortable with:

- Kubernetes (deployments, services, resource requests/limits, HPA, node affinity)
- AWS (EKS, EC2, IAM, S3, VPC)
- Terraform (modules, state, workspaces)
- Prometheus + Grafana (metrics, alerting, dashboards)
- Linux/Bash (you live here)

If you need a refresher on any of these, do that first. This guide does not cover them.

---

## What this guide does NOT cover

- ML theory or model training mathematics
- Python/PyTorch for data scientists
- Prompt engineering
- Fine-tuning from scratch on foundation models (the infra for it, yes — the ML, no)

---

## Contributing

Found an error, a better pattern, or something that's gone stale? Open a PR.
See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

Use the issue templates:
- [Suggest a resource](./.github/ISSUE_TEMPLATE/resource-suggestion.md)
- [Lab feedback](./.github/ISSUE_TEMPLATE/lab-feedback.md)

---

## License

MIT. Use it, fork it, teach with it.
