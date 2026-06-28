# Lab 02: vLLM Deployment and Saturation Analysis

> Deploy vLLM, load test it, and find your saturation point. This number defines all future capacity planning.

**Phase:** 2 — Model Serving
**Time:** 3–4 hours
**Prerequisites:** GPU instance (g5.xlarge minimum), Docker, `hey` or `wrk` for load testing

---

## Objective

By the end of this lab you will have:
- vLLM serving Llama-3.1-8B via OpenAI-compatible API
- Measured TTFT and throughput at multiple concurrency levels
- Identified your saturation point (where adding load stops adding throughput)
- A cost-per-1M-tokens number for this hardware

---

## Part 1: Deploy vLLM with Docker

```bash
# Get a HuggingFace token for gated models (Llama-3 requires acceptance of terms)
# https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

export HF_TOKEN="hf_your_token_here"
export MODEL="meta-llama/Llama-3.1-8B-Instruct"

# Run vLLM — this pulls the model on first start (~16GB download)
docker run --gpus all \
  --name vllm-inference \
  -p 8000:8000 \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model $MODEL \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --dtype auto

# Verify it started
curl http://localhost:8000/health
curl http://localhost:8000/v1/models | jq
```

---

## Part 2: Baseline Test

```bash
# Single request — establish baseline
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Write a 200-word summary of Kubernetes networking."}],
    "max_tokens": 256,
    "stream": false
  }' | jq '{
    ttft_ms: "measure manually",
    total_tokens: .usage.total_tokens,
    output_tokens: .usage.completion_tokens
  }'
```

---

## Part 3: Load Test Script

Save as `load_test.py`:

```python
#!/usr/bin/env python3
"""
vLLM Saturation Analysis
Sweeps concurrency levels and measures throughput + TTFT.
"""

import asyncio
import aiohttp
import time
import statistics
import json

ENDPOINT = "http://localhost:8000/v1/chat/completions"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

PROMPT = "Explain the CAP theorem in distributed systems. Be thorough."

async def single_request(session: aiohttp.ClientSession) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 200,
        "stream": True,
    }

    start = time.perf_counter()
    first_token_time = None
    token_times = []

    async with session.post(ENDPOINT, json=payload) as resp:
        async for line in resp.content:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk["choices"][0]["delta"].get("content"):
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                else:
                    token_times.append(now)

    end = time.perf_counter()
    ttft = (first_token_time - start) * 1000 if first_token_time else None
    tbt = statistics.mean([(token_times[i] - token_times[i-1]) * 1000
                           for i in range(1, len(token_times))]) if len(token_times) > 1 else None

    return {
        "e2e_ms": (end - start) * 1000,
        "ttft_ms": ttft,
        "tbt_ms": tbt,
        "output_tokens": len(token_times) + 1,
    }

async def run_concurrent(concurrency: int, total_requests: int) -> dict:
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def bounded_request(session):
        async with semaphore:
            return await single_request(session)

    async with aiohttp.ClientSession() as session:
        start = time.perf_counter()
        tasks = [bounded_request(session) for _ in range(total_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.perf_counter() - start

    valid = [r for r in results if isinstance(r, dict)]
    total_tokens = sum(r["output_tokens"] for r in valid)

    return {
        "concurrency": concurrency,
        "requests": len(valid),
        "duration_s": duration,
        "throughput_tok_s": total_tokens / duration,
        "throughput_req_s": len(valid) / duration,
        "ttft_p50_ms": statistics.median(r["ttft_ms"] for r in valid if r["ttft_ms"]),
        "ttft_p99_ms": sorted(r["ttft_ms"] for r in valid if r["ttft_ms"])[int(len(valid)*0.99)-1],
        "tbt_p50_ms": statistics.median(r["tbt_ms"] for r in valid if r["tbt_ms"]),
    }

async def main():
    concurrency_levels = [1, 2, 4, 8, 16, 32, 64]
    print(f"{'Concurrency':>12} {'Throughput (tok/s)':>20} {'TTFT p50 (ms)':>15} {'TTFT p99 (ms)':>15} {'TBT p50 (ms)':>14}")
    print("-" * 80)

    for concurrency in concurrency_levels:
        result = await run_concurrent(concurrency, total_requests=concurrency * 3)
        print(f"{result['concurrency']:>12} {result['throughput_tok_s']:>20.1f} "
              f"{result['ttft_p50_ms']:>15.0f} {result['ttft_p99_ms']:>15.0f} "
              f"{result['tbt_p50_ms']:>14.1f}")

if __name__ == "__main__":
    asyncio.run(main())
```

```bash
pip install aiohttp
python load_test.py
```

---

## Part 4: Interpret Your Results

Record your numbers in this table:

```
Concurrency | Throughput (tok/s) | TTFT p50 | TTFT p99 | Notes
────────────┼────────────────────┼──────────┼──────────┼──────
1           |                    |          |          |
2           |                    |          |          |
4           |                    |          |          |
8           |                    |          |          |
16          |                    |          |          |
32          |                    |          |          |
64          |                    |          |          |
```

**What to look for:**
- Throughput should increase with concurrency — up to a point
- TTFT p50 starts rising as GPU is saturated
- TTFT p99 diverges from p50 under pressure (queue effects)
- The concurrency where throughput plateaus = your saturation point

---

## Part 5: Calculate Cost per 1M Tokens

```bash
# Get your throughput at the saturation point (replace X with your number)
THROUGHPUT_TOK_S=X
GPU_COST_PER_HOUR=1.21  # g5.xlarge on-demand us-east-1

TOKENS_PER_HOUR=$(echo "$THROUGHPUT_TOK_S * 3600" | bc)
COST_PER_1M=$(echo "scale=4; $GPU_COST_PER_HOUR / ($TOKENS_PER_HOUR / 1000000)" | bc)

echo "Throughput: ${THROUGHPUT_TOK_S} tok/s"
echo "Tokens/hour: ${TOKENS_PER_HOUR}"
echo "Cost per 1M tokens: \$${COST_PER_1M}"
```

---

## Expected Results (g5.xlarge, Llama-3.1-8B FP16)

Your numbers will vary. Rough ballpark:
- Throughput peak: ~1,500–2,500 tok/s
- TTFT p50 at saturation: ~400–800ms
- Saturation point: concurrency 16–32
- Cost per 1M tokens: ~$0.15–$0.30

---

## Cleanup

```bash
docker stop vllm-inference && docker rm vllm-inference
```
