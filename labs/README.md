# Labs

> Hands-on projects for each phase. Don't skip them — the lab is where it sticks.

Each lab is self-contained. You can run them independently without completing previous labs, though some reference configs from earlier phases.

---

## GPU requirements at a glance

| Lab | Title | GPU Required | Est. Cost |
|---|---|---|---|
| [01](./01-llm-profiling/) | LLM Profiling | No (CPU works) | Free |
| [02](./02-vllm-deployment/) | vLLM Deployment | **Yes** (min g5.xlarge) | ~$1–5 |
| [03](./03-eks-gpu-cluster/) | EKS GPU Cluster | Parts 1–3: No / Parts 4–6: Yes | Free or ~$5–15 |
| [04](./04-aws-comparison/) | AWS Comparison | No (Bedrock + SM Serverless) | ~$1–5 |
| [05](./05-fine-tuning-pipeline/) | Fine-Tuning Pipeline | No (CPU or Colab) | Free |
| [06](./06-observability-stack/) | Observability Stack | No (Docker Compose) | Free |
| [07](./07-llm-gateway/) | LLM Gateway | No (Ollama on CPU) | Free |

**Start with Lab 01 or Lab 06** if you want a fast win on any hardware. Labs 01 and 06 require nothing beyond a laptop and Docker.

---

## Lab summaries

### [Lab 01 — LLM Profiling](./01-llm-profiling/)
Install llama.cpp, download Llama-3.2-1B at 4 quantization levels (Q4_K_M, Q6_K, Q8_0, F16), benchmark tokens/sec and RAM for each, calculate cost-per-1M-tokens. Builds intuition before you touch K8s or cloud.

### [Lab 02 — vLLM Deployment](./02-vllm-deployment/)
Deploy vLLM with Llama-3.1-8B. Run an async load test in Python. Find your saturation point — the concurrency level where throughput peaks before TTFT degrades. GPU required.

### [Lab 03 — EKS GPU Cluster](./03-eks-gpu-cluster/)
Parts 1–3 use kind (local, free): deploy GPU Operator, Prometheus, Grafana, and a DCGM metric simulator. Parts 4–6 use real EKS + Karpenter: provision GPU nodes on demand, run vLLM, wire real DCGM metrics.

### [Lab 04 — AWS Comparison](./04-aws-comparison/)
Call an LLM via Bedrock (zero infra). Deploy on SageMaker Serverless (no GPU). Optionally deploy on a real GPU endpoint. Compare cold start, latency, and cost per 1M tokens. Build a decision matrix.

### [Lab 05 — Fine-Tuning Pipeline](./05-fine-tuning-pipeline/)
LoRA fine-tune GPT-2 (runs on CPU). Track with MLflow. Run an eval gate. Register to MLflow Registry. Chain it into a shell pipeline that blocks bad models. Optional: wire into Argo Workflows.

### [Lab 06 — Observability Stack](./06-observability-stack/)
Full stack via Docker Compose: GPU metric simulator (DCGM-format) → Prometheus → Grafana dashboards, plus Langfuse for LLM request tracing. Generate load and watch metrics update live. No cloud required.

### [Lab 07 — LLM Gateway](./07-llm-gateway/)
Build a multi-tenant LLM gateway in FastAPI: JWT auth, per-tenant token-bucket rate limiting (Redis), model routing by tier, cost attribution, Prometheus metrics. Capstone lab — combines infra skills from all phases.
