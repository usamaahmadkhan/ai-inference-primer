# Inference Metrics Cheatsheet

> The vocabulary of LLM serving. Know these before you write a single SLO.

---

## Core Metrics

```
Metric          │ Full Name                    │ Unit   │ Formula
────────────────┼──────────────────────────────┼────────┼────────────────────────────
TTFT            │ Time to First Token          │ ms     │ time(first_token) - time(request)
TBT             │ Time Between Tokens          │ ms     │ avg time between consecutive tokens
ITL             │ Inter-Token Latency          │ ms     │ synonym for TBT
E2E             │ End-to-End Latency           │ ms     │ TTFT + (n_output_tokens × TBT)
TPOT            │ Time Per Output Token        │ ms     │ synonym for TBT (OpenAI terminology)
Throughput      │ Output Token Throughput      │ tok/s  │ total_output_tokens / time_window
RPS             │ Requests Per Second          │ req/s  │ total_requests / time_window
```

---

## Prometheus Metric Names

### vLLM
```
vllm:e2e_request_latency_seconds         E2E latency histogram
vllm:time_to_first_token_seconds         TTFT histogram
vllm:time_per_output_token_seconds       TBT histogram
vllm:request_queue_length                Current queue depth (gauge)
vllm:gpu_cache_usage_perc               KV cache utilization 0.0–1.0
vllm:num_requests_running               Requests currently being processed
vllm:num_requests_waiting               Requests in queue
vllm:prompt_tokens_total                Input tokens (counter, use for billing)
vllm:generation_tokens_total            Output tokens (counter, use for billing)
vllm:num_preemptions_total              KV cache evictions (high = VRAM pressure)
```

### DCGM (GPU hardware)
```
nvidia_dcgm_fi_dev_gpu_util              GPU compute utilization (%)
nvidia_dcgm_fi_dev_mem_copy_util         GPU memory bandwidth utilization (%)
nvidia_dcgm_fi_dev_fb_used               VRAM used (MiB)
nvidia_dcgm_fi_dev_fb_free               VRAM free (MiB)
nvidia_dcgm_fi_dev_power_usage           Power draw (W)
nvidia_dcgm_fi_dev_gpu_temp              Temperature (°C)
nvidia_dcgm_fi_dev_sm_clock              SM clock speed (MHz)
nvidia_dcgm_fi_dev_nvlink_bandwidth_*    NVLink bandwidth (bytes/s)
nvidia_dcgm_fi_dev_pcie_tx_bytes         PCIe host→GPU transfer
nvidia_dcgm_fi_dev_pcie_rx_bytes         PCIe GPU→host transfer
```

---

## Recommended SLO Targets

| Metric | Chatbot / Streaming | Internal Tool | Batch / Async |
|---|---|---|---|
| TTFT p50 | < 300ms | < 1s | N/A |
| TTFT p99 | < 2s | < 5s | N/A |
| TBT p50 | < 30ms | < 100ms | N/A |
| TBT p99 | < 100ms | < 500ms | N/A |
| E2E p99 (500 tok) | < 18s | < 55s | < 5min |
| Error Rate | < 0.1% | < 0.5% | < 1% |
| Queue Depth | Alert > 5 | Alert > 20 | Alert > 100 |
| VRAM usage | Alert > 90% | Alert > 90% | Alert > 85% |

*Adjust to your actual workload. These are starting points.*

---

## Useful PromQL Queries

```promql
# TTFT SLO compliance — % of requests under 2 seconds
sum(rate(vllm:time_to_first_token_seconds_bucket{le="2.0"}[5m]))
/
sum(rate(vllm:time_to_first_token_seconds_count[5m]))

# 99th percentile TTFT
histogram_quantile(0.99,
  sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m]))
)

# Output token throughput (tokens/sec)
sum(rate(vllm:generation_tokens_total[5m]))

# VRAM pressure ratio
(nvidia_dcgm_fi_dev_fb_used)
/
(nvidia_dcgm_fi_dev_fb_used + nvidia_dcgm_fi_dev_fb_free)

# KV cache eviction rate (high = VRAM constrained)
rate(vllm:num_preemptions_total[5m])

# Cost per request (adjust GPU $/hr)
(rate(vllm:prompt_tokens_total[1h]) + rate(vllm:generation_tokens_total[1h]))
* (1.21 / 3600)  # $1.21/hr g5.2xlarge
/ rate(vllm:e2e_request_latency_seconds_count[1h])

# Queue depth (active alert source)
vllm:request_queue_length > 10

# GPU temperature alert
nvidia_dcgm_fi_dev_gpu_temp > 85
```

---

## Reading Saturation: What the Numbers Mean

```
Scenario                          │ GPU Util │ VRAM % │ Queue │ Diagnosis
──────────────────────────────────┼──────────┼────────┼───────┼──────────────────────
Healthy, well-loaded              │ 70–90%   │ 60–80% │ 0–5   │ ✓ Good
Underloaded / idle                │ <20%     │ <50%   │ 0     │ Cost waste
Memory-saturated (small batches)  │ 30–60%   │ >90%   │ >10   │ Need more VRAM
Compute-saturated (large batches) │ >95%     │ 60–80% │ >20   │ Need more GPUs
OOM crash pattern                 │ spike→0  │ 100%   │ >50   │ VRAM exhausted
Thermal throttling                │ 70–80%   │ normal │ 0     │ Check temp, clocks
Stuck / deadlock                  │ <5%      │ normal │ >100  │ Check logs/traces
```

---

## Token Economics

```
Cost per 1M tokens formula (self-hosted)
═════════════════════════════════════════

GPU hourly cost
────────────────────────────────  =  $ per token
Tokens generated per hour

Example: g5.2xlarge running Llama-3.1-8B

  GPU cost:   $1.21/hr
  Throughput: ~2,000 tok/s = 7,200,000 tok/hr

  Cost per 1M tokens = $1.21 / 7.2 = $0.168 / 1M tokens

Compare to:
  Bedrock Claude Haiku:   $0.25 / 1M input, $1.25 / 1M output
  OpenAI GPT-4o mini:     $0.15 / 1M input, $0.60 / 1M output
  Self-hosted 8B:         ~$0.17 / 1M (input + output combined)

Self-hosted wins above ~10 req/min sustained.
Below that, managed APIs are cheaper (no idle GPU cost).
```
