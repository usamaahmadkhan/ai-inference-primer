# Phase 6: AI Observability

> New metrics, new failure modes. Your Prometheus stack gets extended, not replaced.

**Duration:** 2–3 weeks
**Lab:** [06-observability-stack](../../labs/06-observability-stack/)

---

## The full AI observability stack

```
AI Observability: Two layers you need
═══════════════════════════════════════════════════════════════

Layer 1: Infrastructure (you already know this)
─────────────────────────────────────────────────────────────
  GPU Hardware
      │
      ▼ DCGM Exporter
  GPU Metrics (utilization, VRAM, temperature, power)
      │
      ▼ Prometheus
  Time-series storage
      │
      ▼ Grafana
  Dashboards + Alerts

Layer 2: LLM Application (new territory)
─────────────────────────────────────────────────────────────
  LLM Requests (prompts, responses, tokens, latency)
      │
      ▼ Langfuse / LangSmith SDK
  Trace storage (structured, queryable)
      │
      ▼ Langfuse UI / Grafana
  Prompt analysis, cost attribution, quality monitoring

Both layers are required. Neither covers the other.
```

---

## GPU Metrics (DCGM)

Your GPU metrics vocabulary. Every SRE on a GPU-serving team needs these internalized.

```
DCGM Metric Reference
═════════════════════════════════════════════════════════

Metric Name                              │ What it means
─────────────────────────────────────────┼─────────────────────────────────────
nvidia_dcgm_fi_dev_gpu_util              │ GPU compute utilization (%)
                                         │ Target: >70% in production
                                         │ Low: idle or model loading
                                         │ Sustained 100%: may be bottlenecked

nvidia_dcgm_fi_dev_fb_used               │ VRAM used (MiB)
nvidia_dcgm_fi_dev_fb_free               │ VRAM free (MiB)
                                         │ (fb = framebuffer, historical name)
                                         │ Watch ratio: fb_used/(fb_used+fb_free)
                                         │ Alert at >90%

nvidia_dcgm_fi_dev_power_usage           │ GPU power draw (watts)
                                         │ A100 TDP: 400W, H100: 700W
                                         │ Sustained near TDP = expected
                                         │ Sudden drop = GPU throttling

nvidia_dcgm_fi_dev_gpu_temp              │ GPU core temperature (°C)
                                         │ Alert at >85°C
                                         │ Throttling starts ~83°C on A100

nvidia_dcgm_fi_dev_sm_clock              │ Streaming multiprocessor clock (MHz)
                                         │ Drops when thermal throttling

nvidia_dcgm_fi_dev_nvlink_bandwidth_c_tx │ NVLink transmit bandwidth (bytes/s)
nvidia_dcgm_fi_dev_nvlink_bandwidth_c_rx │ NVLink receive bandwidth (bytes/s)
                                         │ Critical for tensor parallel workloads
                                         │ Low here = inter-GPU bottleneck

nvidia_dcgm_fi_dev_pcie_tx_bytes         │ PCIe host→GPU bandwidth
nvidia_dcgm_fi_dev_pcie_rx_bytes         │ PCIe GPU→host bandwidth
                                         │ High = model loading or CPU-GPU transfer
```

### GPU utilization: reading it correctly

```
GPU Utilization Interpretation
════════════════════════════════

High util + normal TTFT      → healthy, well-loaded
High util + high TTFT        → saturated, need more capacity
Low util + normal TTFT       → underloaded (cost inefficiency)
Low util + high TTFT         → stuck/deadlocked workload — investigate
Low util + OOM crash         → KV cache exhausted before compute saturates

VRAM pressure zones:
  0–70%  ████████████░░░░░░░░  Healthy
  70–85% ████████████████░░░░  Monitor
  85–95% ██████████████████░░  Alert (page SRE)
  95%+   ████████████████████  Critical (OOM imminent)
```

---

## LLM Inference SLIs

These are the metrics you define SLOs on. Define them before an incident forces you to.

```
SLI Reference for LLM Serving
═══════════════════════════════════════════════════════════════

Metric          │ Formula                      │ Typical SLO
────────────────┼──────────────────────────────┼────────────────────────
TTFT p50        │ median(time_to_first_token)   │ < 300ms
TTFT p99        │ 99th pct(time_to_first_token) │ < 2s
                │                              │
TBT p50         │ median(time_between_tokens)   │ < 30ms (streaming UX)
TBT p99         │ 99th pct(time_between_tokens) │ < 100ms
                │                              │
E2E Latency     │ TTFT + (n_tokens × TBT)       │ Depends on output len
                │                              │
Throughput      │ sum(output_tokens) / time     │ Maximize (no SLO needed)
                │                              │
Queue Depth     │ pending_requests gauge        │ Alert at > 10
                │                              │
Error Rate      │ 5xx / total requests          │ < 0.1%
```

### vLLM exposes these natively

```bash
# vLLM metrics endpoint (Prometheus format)
curl http://vllm-service:8000/metrics

# Key metrics from vLLM:
# vllm:e2e_request_latency_seconds      - histogram of E2E latency
# vllm:time_to_first_token_seconds      - histogram of TTFT
# vllm:time_per_output_token_seconds    - histogram of TBT
# vllm:request_queue_length             - current queue depth
# vllm:gpu_cache_usage_perc             - KV cache utilization (%)
# vllm:num_requests_running             - currently processing
# vllm:num_requests_waiting             - in queue
# vllm:prompt_tokens_total              - counter for billing
# vllm:generation_tokens_total          - counter for billing
```

### Prometheus recording rules

```yaml
# Pre-compute expensive queries as recording rules
groups:
- name: vllm_sli_recording
  interval: 30s
  rules:

  # TTFT SLO compliance (% of requests under 2s)
  - record: vllm:ttft_slo_compliance_ratio
    expr: |
      sum(rate(vllm:time_to_first_token_seconds_bucket{le="2.0"}[5m]))
      /
      sum(rate(vllm:time_to_first_token_seconds_count[5m]))

  # KV cache pressure (approaching full = throughput will drop)
  - record: vllm:kv_cache_pressure
    expr: vllm:gpu_cache_usage_perc > 0.85

  # Token throughput (output tokens/sec)
  - record: vllm:output_token_throughput
    expr: sum(rate(vllm:generation_tokens_total[5m]))
```

---

## LLM-Specific Tracing with Langfuse

GPU metrics tell you the infrastructure is saturated. Langfuse tells you *why* — which prompts are slow, which users are expensive, which model versions are worse.

```
Langfuse Trace Structure
═════════════════════════

Trace (one user turn)
├── id: "tr-abc123"
├── user_id: "user-456"
├── session_id: "sess-789"
├── input: "Explain Kubernetes networking"
├── output: "Kubernetes networking works by..."
├── latency: 3421ms
├── cost: $0.0023
│
└── Spans (sub-steps)
    ├── span: "prompt-formatting"
    │   └── latency: 12ms
    ├── span: "llm-call"
    │   ├── model: "llama-3.1-8b"
    │   ├── prompt_tokens: 234
    │   ├── completion_tokens: 891
    │   ├── ttft: 412ms
    │   └── latency: 3380ms
    └── span: "post-processing"
        └── latency: 29ms
```

### Instrumenting your application

```python
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse()  # reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

@observe()  # auto-traces this function
def generate_response(user_message: str, user_id: str) -> str:
    langfuse_context.update_current_trace(
        user_id=user_id,
        metadata={"model_version": "v3.2", "env": "production"}
    )

    response = vllm_client.chat.completions.create(
        model="llama-3.1-8b",
        messages=[{"role": "user", "content": user_message}]
    )

    # Langfuse captures: prompt, response, tokens, latency automatically
    return response.choices[0].message.content
```

---

## Cost Attribution

Finance will ask for this. Build it before they do.

```
Cost Attribution Model
═══════════════════════

Per-request cost:
  cost = (prompt_tokens × input_price) + (completion_tokens × output_price)

For self-hosted GPU:
  input_price  = (gpu_cost_per_hour) / (throughput_tokens_per_hour)
  output_price = same (or 2–4x if decode is slower than prefill)

Example (g5.2xlarge, Llama-3.1-8B):
  GPU cost: $1.21/hr = $0.0000003/ms
  Throughput: ~2000 tokens/sec = 7.2M tokens/hr
  Cost per 1M tokens: $1.21 / 7.2 = $0.168 / 1M tokens

Grafana dashboard breakdown by:
  ┌─────────────────────────────────────────────┐
  │  Cost by Team     │  Cost by User           │
  │  ████ eng: 45%    │  ██ user-123: 12%       │
  │  ███ product: 30% │  █ user-456: 8%         │
  │  ██ data: 15%     │  ...                    │
  │  █ other: 10%     │                         │
  │                   │                         │
  │  Cost by Model    │  Cost by Prompt Type    │
  │  ████ 70b: 60%    │  ████ summarize: 40%    │
  │  ██ 8b: 25%       │  ██ classify: 20%       │
  │  █ 13b: 15%       │  ...                    │
  └─────────────────────────────────────────────┘
```

---

## Resources

See [resources.md](./resources.md) for the full curated list.

**Essential:**
1. DCGM Exporter GitHub — includes example Grafana dashboards
2. vLLM Metrics docs — full reference for vLLM Prometheus metrics
3. Langfuse self-hosted docs — run it in your cluster, don't send prod data out

---

## Lab

**[→ Lab 06: Observability Stack](../../labs/06-observability-stack/)**

Build the full stack: DCGM Exporter → Prometheus for GPU metrics, vLLM metrics via ServiceMonitor, Langfuse for LLM request tracing, Grafana dashboard showing GPU utilization, TTFT p99, queue depth, and cost per request by team. Add the VRAM pressure and TTFT SLO alerts. This is the deliverable you put in front of your team as "this is how we run AI in production."
