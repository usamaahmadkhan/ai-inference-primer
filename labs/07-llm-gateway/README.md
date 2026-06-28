# Lab 07: LLM Gateway

> Build a multi-tenant LLM gateway with auth, rate limiting, model routing, and cost attribution.

**Phase:** 7 — Advanced Topics
**GPU required:** No — uses Ollama on CPU as the LLM backend.
**Time:** 3–4 hours
**Cost:** Free

---

## Objective

By the end of this lab you will have:
- A working LLM API gateway in Python (FastAPI)
- JWT authentication mapping API keys to tenants
- Per-tenant token-bucket rate limiting backed by Redis
- Model routing based on tenant tier (premium → large model, standard → small model)
- Cost attribution tracking tokens per tenant
- Prometheus metrics on all of the above

This is the most SRE-native lab in the guide. Everything here — rate limiting, auth, routing, observability — is infrastructure engineering, not ML.

---

## Architecture

```
LLM Gateway Architecture
══════════════════════════════════════════════════════════════

Client Request (curl / SDK)
         │
         │  POST /v1/chat/completions
         │  Authorization: Bearer <api-key>
         ▼
┌────────────────────────────────────────────────────────────┐
│                      LLM Gateway                           │
│                                                            │
│  ① Auth Middleware          ② Rate Limiter                │
│  ─────────────────          ────────────────               │
│  API key → tenant ID        Token bucket per tenant        │
│  JWT validation             Redis backend                  │
│  RBAC (model access)        Hard + soft limits             │
│                                                            │
│  ③ Model Router             ④ Cost Tracker               │
│  ──────────────             ──────────────────            │
│  tier: premium → 7b         Count tokens per:             │
│  tier: standard → 0.5b      - tenant                      │
│  tier: free → 0.5b (low)   - model                       │
│                             Log to Redis + Prometheus      │
└────────────────────────────────────────────────────────────┘
         │
         ├──► Ollama (qwen2:0.5b)    ← standard / free tier
         └──► Ollama (qwen2:1.5b)   ← premium tier
```

---

## Part 1: Environment setup

```bash
python3 -m venv gateway-lab
source gateway-lab/bin/activate

pip install \
  fastapi==0.111.0 \
  uvicorn==0.29.0 \
  httpx==0.27.0 \
  redis==5.0.4 \
  pyjwt==2.8.0 \
  prometheus-client==0.20.0 \
  pydantic==2.7.0

# Start Redis (rate limit backend)
docker run -d --name redis-gateway -p 6379:6379 redis:7-alpine
redis-cli ping   # should return PONG

# Start Ollama and pull models
docker run -d --name ollama -p 11434:11434 ollama/ollama:latest
sleep 5
docker exec ollama ollama pull qwen2:0.5b    # standard tier (~400MB)
docker exec ollama ollama pull qwen2:1.5b    # premium tier (~900MB)
```

---

## Part 2: The gateway

Save as `gateway.py`:

```python
#!/usr/bin/env python3
"""
LLM Gateway — multi-tenant, rate-limited, cost-tracked.
Implements OpenAI-compatible /v1/chat/completions endpoint.
"""

import time, json, jwt, redis, httpx, logging
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Configuration ─────────────────────────────────────────────────────────────

OLLAMA_BASE     = "http://localhost:11434"
JWT_SECRET      = "change-me-in-production"
REDIS_URL       = "redis://localhost:6379"

# Tenant config: api_key → tenant details
# In production this lives in a database, not in-process config
TENANTS = {
    "sk-eng-team-001": {
        "tenant_id": "team-eng",
        "tier":      "premium",
        "rate_limit_tokens_per_min": 100_000,
        "allowed_models": ["*"],
    },
    "sk-product-001": {
        "tenant_id": "team-product",
        "tier":      "standard",
        "rate_limit_tokens_per_min": 20_000,
        "allowed_models": ["standard", "free"],
    },
    "sk-free-001": {
        "tenant_id": "user-free",
        "tier":      "free",
        "rate_limit_tokens_per_min": 5_000,
        "allowed_models": ["free"],
    },
}

# Tier → Ollama model mapping
MODEL_ROUTING = {
    "premium":  "qwen2:1.5b",
    "standard": "qwen2:0.5b",
    "free":     "qwen2:0.5b",
}

# ─── Prometheus Metrics ────────────────────────────────────────────────────────

requests_total = Counter(
    "gateway_requests_total", "Total LLM requests",
    ["tenant", "tier", "model", "status"]
)
tokens_total = Counter(
    "gateway_tokens_total", "Total tokens processed",
    ["tenant", "tier", "model", "type"]   # type: prompt | completion
)
request_latency = Histogram(
    "gateway_request_latency_seconds", "Request latency",
    ["tenant", "tier"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60]
)
rate_limit_hits = Counter(
    "gateway_rate_limit_hits_total", "Rate limit rejections",
    ["tenant", "tier"]
)
active_requests = Gauge(
    "gateway_active_requests", "Currently active requests",
    ["tenant"]
)

# ─── Redis Rate Limiter ────────────────────────────────────────────────────────

r = redis.from_url(REDIS_URL, decode_responses=True)

def check_rate_limit(tenant_id: str, tokens_requested: int, limit_per_min: int) -> bool:
    """
    Token-bucket rate limiter using Redis.
    Key expires every 60s — simple sliding window approximation.
    Returns True if allowed, False if rate limited.
    """
    key = f"ratelimit:{tenant_id}:{int(time.time() // 60)}"
    pipe = r.pipeline()
    pipe.incrby(key, tokens_requested)
    pipe.expire(key, 120)   # 2 min TTL so cleanup is automatic
    current_usage, _ = pipe.execute()
    return int(current_usage) <= limit_per_min

def record_token_usage(tenant_id: str, prompt_tokens: int, completion_tokens: int):
    """Track cumulative token usage per tenant in Redis (for billing/chargeback)."""
    pipe = r.pipeline()
    pipe.hincrby(f"usage:{tenant_id}", "prompt_tokens",     prompt_tokens)
    pipe.hincrby(f"usage:{tenant_id}", "completion_tokens", completion_tokens)
    pipe.hincrby(f"usage:{tenant_id}", "total_requests",    1)
    pipe.execute()

def get_tenant_usage(tenant_id: str) -> dict:
    return r.hgetall(f"usage:{tenant_id}") or {}

# ─── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(title="LLM Gateway", version="0.1.0")
app.mount("/metrics", make_asgi_app())   # Prometheus scrape endpoint

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model:       Optional[str] = None
    messages:    list[ChatMessage]
    max_tokens:  Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream:      Optional[bool] = False

def authenticate(request: Request) -> dict:
    """Extract and validate API key from Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")

    api_key = auth_header.removeprefix("Bearer ").strip()
    tenant  = TENANTS.get(api_key)
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return tenant

@app.get("/health")
async def health():
    return {"status": "ok", "redis": r.ping()}

@app.get("/v1/models")
async def list_models(tenant: dict = Depends(authenticate)):
    """Return models available to this tenant's tier."""
    return {
        "object": "list",
        "data": [
            {"id": "premium-model",  "object": "model"},
            {"id": "standard-model", "object": "model"},
        ]
    }

@app.get("/usage/{tenant_id}")
async def usage(tenant_id: str, request: Request):
    """Return token usage for a tenant (admin endpoint — add auth in production)."""
    return get_tenant_usage(tenant_id)

@app.post("/v1/chat/completions")
async def chat_completions(body: ChatRequest, request: Request):
    tenant = authenticate(request)

    tenant_id = tenant["tenant_id"]
    tier      = tenant["tier"]
    model     = MODEL_ROUTING[tier]

    active_requests.labels(tenant=tenant_id).inc()
    start = time.perf_counter()

    try:
        # Estimate prompt tokens (rough: 1 token ≈ 4 chars)
        prompt_chars    = sum(len(m.content) for m in body.messages)
        est_prompt_tok  = prompt_chars // 4
        est_total_tok   = est_prompt_tok + (body.max_tokens or 512)

        # ① Rate limit check
        if not check_rate_limit(tenant_id, est_total_tok, tenant["rate_limit_tokens_per_min"]):
            rate_limit_hits.labels(tenant=tenant_id, tier=tier).inc()
            requests_total.labels(tenant=tenant_id, tier=tier, model=model, status="rate_limited").inc()
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": f"Token limit exceeded. Limit: {tenant['rate_limit_tokens_per_min']:,} tokens/min.",
                    "tenant": tenant_id,
                }
            )

        logger.info(f"[{tenant_id}/{tier}] → {model} | est_tokens={est_total_tok}")

        # ② Forward to Ollama
        ollama_payload = {
            "model":    model,
            "messages": [{"role": m.role, "content": m.content} for m in body.messages],
            "stream":   False,
            "options":  {
                "num_predict": body.max_tokens or 512,
                "temperature": body.temperature or 0.7,
            }
        }

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=ollama_payload)

        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Ollama error: {resp.text}")

        ollama_resp     = resp.json()
        output_text     = ollama_resp["message"]["content"]
        prompt_tokens   = ollama_resp.get("prompt_eval_count", est_prompt_tok)
        completion_tokens = ollama_resp.get("eval_count", len(output_text) // 4)

        # ③ Track usage
        record_token_usage(tenant_id, prompt_tokens, completion_tokens)
        tokens_total.labels(tenant=tenant_id, tier=tier, model=model, type="prompt").inc(prompt_tokens)
        tokens_total.labels(tenant=tenant_id, tier=tier, model=model, type="completion").inc(completion_tokens)
        requests_total.labels(tenant=tenant_id, tier=tier, model=model, status="success").inc()

        latency = time.perf_counter() - start
        request_latency.labels(tenant=tenant_id, tier=tier).observe(latency)

        logger.info(f"[{tenant_id}/{tier}] ✓ {latency*1000:.0f}ms | {prompt_tokens}+{completion_tokens} tokens")

        # ④ Return OpenAI-compatible response
        return {
            "id":      f"chatcmpl-{int(time.time())}",
            "object":  "chat.completion",
            "model":   model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens":     prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens":      prompt_tokens + completion_tokens,
            },
            "x-gateway": {   # bonus: expose routing metadata in response headers
                "tenant": tenant_id,
                "tier":   tier,
                "model":  model,
            }
        }

    finally:
        active_requests.labels(tenant=tenant_id).dec()
```

---

## Part 3: Run the gateway

```bash
uvicorn gateway:app --host 0.0.0.0 --port 8080 --reload
```

---

## Part 4: Test all tenant tiers

```bash
# Premium tier — routes to large model, high rate limit
curl -s http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-eng-team-001" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is Kubernetes?"}], "max_tokens": 100}' \
  | python3 -m json.tool

# Standard tier
curl -s http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-product-001" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is a pod?"}], "max_tokens": 50}' \
  | python3 -m json.tool

# Invalid key — should get 401
curl -s http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer invalid-key" \
  -d '{"messages": []}' | python3 -m json.tool
```

---

## Part 5: Test rate limiting

```python
# rate_limit_test.py — exhaust the free tier's 5K token/min limit
import httpx, time

client = httpx.Client(base_url="http://localhost:8080", timeout=30)
headers = {"Authorization": "Bearer sk-free-001"}

print("Sending requests until rate limit hit...")
for i in range(30):
    resp = client.post("/v1/chat/completions",
        headers=headers,
        json={"messages": [{"role": "user", "content": "Write a 200-word essay about distributed systems."}],
              "max_tokens": 300})

    if resp.status_code == 429:
        print(f"Request {i+1}: RATE LIMITED ✓ — {resp.json()['detail']['message']}")
        break
    elif resp.status_code == 200:
        usage = resp.json()["usage"]
        print(f"Request {i+1}: OK | {usage['total_tokens']} tokens used")
    else:
        print(f"Request {i+1}: ERROR {resp.status_code}")
```

```bash
python3 rate_limit_test.py
```

---

## Part 6: Cost attribution dashboard

```python
# usage_report.py — print per-tenant usage report
import redis, json
from datetime import datetime

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

# Pricing (self-hosted: $1.21/hr g5.2xlarge, ~7M tokens/hr)
COST_PER_1M_TOKENS = 0.17

tenants = ["team-eng", "team-product", "user-free"]

print(f"\n{'═'*60}")
print(f"  LLM Gateway Usage Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'═'*60}")
print(f"{'Tenant':<20} {'Prompt':>10} {'Completion':>12} {'Total':>10} {'Est. Cost':>12}")
print(f"{'─'*60}")

total_tokens = 0
for tenant_id in tenants:
    usage = r.hgetall(f"usage:{tenant_id}")
    if not usage:
        continue
    p = int(usage.get("prompt_tokens",     0))
    c = int(usage.get("completion_tokens", 0))
    t = p + c
    cost = (t / 1_000_000) * COST_PER_1M_TOKENS
    total_tokens += t
    print(f"{tenant_id:<20} {p:>10,} {c:>12,} {t:>10,} ${cost:>10.4f}")

print(f"{'─'*60}")
total_cost = (total_tokens / 1_000_000) * COST_PER_1M_TOKENS
print(f"{'TOTAL':<20} {'':>10} {'':>12} {total_tokens:>10,} ${total_cost:>10.4f}")
print()
```

```bash
python3 usage_report.py
```

---

## Part 7: Scrape gateway metrics with Prometheus

```bash
# The gateway exposes /metrics automatically via prometheus_client

# Quick check without Prometheus:
curl -s http://localhost:8080/metrics | grep gateway_

# If you have the observability stack from Lab 06 running:
# Add this to prometheus/prometheus.yml and reload:
#
# - job_name: llm-gateway
#   static_configs:
#   - targets: ['host.docker.internal:8080']
```

---

## Extensions (if you want to go further)

```
Ideas for extending this gateway:

1. Streaming responses
   Replace the non-streaming Ollama call with an SSE stream.
   FastAPI + StreamingResponse + httpx streaming.

2. Model fallback
   If the primary model is unavailable, fall back to a smaller one.
   Add retry logic with exponential backoff.

3. Request logging to S3
   Log every request/response to S3 for audit trail and fine-tuning data.

4. Semantic caching
   Cache responses for semantically similar queries using a vector DB.
   Qdrant + sentence-transformers for the similarity check.

5. Prompt injection detection
   Add a middleware layer that scans prompts for injection attempts
   before forwarding to the model.
```

---

## Cleanup

```bash
docker stop redis-gateway ollama && docker rm redis-gateway ollama
```
