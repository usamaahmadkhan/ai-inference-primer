# Lab 06: AI Observability Stack

> Build the full GPU + LLM observability stack locally with Docker Compose.

**Phase:** 6 — AI Observability
**GPU required:** No — the entire stack runs on CPU with a metric simulator.
**Time:** 2–3 hours
**Cost:** Free

---

## Objective

By the end of this lab you will have:
- Prometheus scraping simulated GPU metrics (DCGM-format) and real vLLM metrics
- Grafana dashboards for GPU utilisation, VRAM pressure, TTFT, queue depth
- Langfuse capturing LLM request traces with token counts and costs
- A working alert rule that fires on VRAM pressure
- A cost-per-request dashboard breakdown by model

This entire stack runs on your laptop. No cloud account required.

---

## Architecture

```
Full AI Observability Stack (local)
════════════════════════════════════════════════════════════

  ┌─────────────────────┐     ┌──────────────────────────┐
  │   GPU Simulator     │     │      Ollama (CPU)        │
  │   :9400/metrics     │     │      :11434              │
  │   (DCGM-format)     │     │  (LLM inference, no GPU) │
  └──────────┬──────────┘     └────────────┬─────────────┘
             │  scrape                      │  requests
             ▼                             ▼
  ┌──────────────────┐         ┌───────────────────────┐
  │    Prometheus    │◄────────│   LLM Proxy / App     │
  │      :9090       │  scrape │   :8080               │
  └────────┬─────────┘         │   (traces → Langfuse) │
           │                   └───────────┬───────────┘
           ▼                               │ traces
  ┌──────────────────┐         ┌───────────▼───────────┐
  │     Grafana      │         │       Langfuse        │
  │      :3000       │         │        :3001          │
  │  GPU + LLM dash  │         │  Prompt tracing + cost│
  └──────────────────┘         └───────────────────────┘
```

---

## Part 1: Docker Compose stack

Save as `docker-compose.yml`:

```yaml
version: "3.8"

services:

  # ── LLM Inference ──────────────────────────────────────────────
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434"]
      interval: 10s
      timeout: 5s
      retries: 10

  # ── GPU Metrics Simulator ───────────────────────────────────────
  # Emits DCGM-compatible Prometheus metrics without real GPUs
  dcgm-simulator:
    image: python:3.11-slim
    ports:
      - "9400:9400"
    command: >
      sh -c "pip install prometheus-client -q && python3 /app/simulate.py"
    volumes:
      - ./simulator:/app
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9400/metrics"]
      interval: 10s
      retries: 5

  # ── Prometheus ──────────────────────────────────────────────────
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus:/etc/prometheus
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=24h'
    depends_on:
      - dcgm-simulator

  # ── Grafana ─────────────────────────────────────────────────────
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning
      - grafana-data:/var/lib/grafana
    depends_on:
      - prometheus

  # ── Langfuse (LLM tracing) ──────────────────────────────────────
  langfuse:
    image: langfuse/langfuse:latest
    ports:
      - "3001:3000"
    environment:
      DATABASE_URL:            "postgresql://langfuse:langfuse@langfuse-db:5432/langfuse"
      NEXTAUTH_URL:            "http://localhost:3001"
      NEXTAUTH_SECRET:         "super-secret-change-in-prod"
      SALT:                    "super-secret-salt"
      LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES: "true"
    depends_on:
      langfuse-db:
        condition: service_healthy

  langfuse-db:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER:     langfuse
      POSTGRES_PASSWORD: langfuse
      POSTGRES_DB:       langfuse
    volumes:
      - langfuse-db-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U langfuse"]
      interval: 5s
      timeout: 5s
      retries: 10

volumes:
  ollama-data:
  prometheus-data:
  grafana-data:
  langfuse-db-data:
```

---

## Part 2: GPU metrics simulator

```bash
mkdir -p simulator
```

```python
# simulator/simulate.py
"""
Emits DCGM-format GPU metrics for Prometheus.
Simulates realistic load patterns: ramp-up, saturation, cool-down.
"""
import time, math, random
from prometheus_client import start_http_server, Gauge, Counter

# Mirror exact DCGM metric names so real dashboards work unchanged
METRICS = {
    "gpu_util":   Gauge("nvidia_dcgm_fi_dev_gpu_util",    "GPU utilization %",    ["gpu","node","modelName"]),
    "fb_used":    Gauge("nvidia_dcgm_fi_dev_fb_used",     "VRAM used MiB",        ["gpu","node","modelName"]),
    "fb_free":    Gauge("nvidia_dcgm_fi_dev_fb_free",     "VRAM free MiB",        ["gpu","node","modelName"]),
    "power":      Gauge("nvidia_dcgm_fi_dev_power_usage", "Power draw W",         ["gpu","node","modelName"]),
    "temp":       Gauge("nvidia_dcgm_fi_dev_gpu_temp",    "Temperature C",        ["gpu","node","modelName"]),
    "sm_clock":   Gauge("nvidia_dcgm_fi_dev_sm_clock",    "SM clock MHz",         ["gpu","node","modelName"]),
    "nvlink_tx":  Gauge("nvidia_dcgm_fi_dev_nvlink_bandwidth_c_tx", "NVLink TX bytes/s", ["gpu","node"]),
}

# Simulate vLLM metrics too
VLLM = {
    "queue_len":       Gauge("vllm_request_queue_length",          "Queue depth",          ["model"]),
    "running":         Gauge("vllm_num_requests_running",          "Active requests",      ["model"]),
    "cache_usage":     Gauge("vllm_gpu_cache_usage_perc",          "KV cache usage",       ["model"]),
    "prompt_tokens":   Counter("vllm_prompt_tokens_total",         "Input tokens",         ["model"]),
    "gen_tokens":      Counter("vllm_generation_tokens_total",     "Output tokens",        ["model"]),
}

VRAM_TOTAL = 24576   # 24 GB in MiB (A10G)
GPUS = [("0", "node-gpu-0", "Llama-3.1-8B"), ("1", "node-gpu-1", "Llama-3.1-8B")]

start_http_server(9400)
print("DCGM simulator running on :9400/metrics")

t = 0
while True:
    # Simulate a realistic load cycle: ramp up over 5 min, sustain, cool down
    cycle = (t % 300) / 300           # 0→1 over 5 min
    base_util = 30 + 55 * math.sin(cycle * math.pi) + random.gauss(0, 4)
    util = max(5, min(98, base_util))

    # Simulate VRAM: model weights take base 15GB, KV cache grows with load
    model_vram  = 15000
    kvcache_vram = (util / 100) * 7000 + random.gauss(0, 200)
    used  = model_vram + kvcache_vram
    free  = max(0, VRAM_TOTAL - used)
    cache_pct = kvcache_vram / (VRAM_TOTAL - model_vram)

    for gpu_id, node, model in GPUS:
        g = {"gpu": gpu_id, "node": node, "modelName": model}
        METRICS["gpu_util"].labels(**g).set(util + random.gauss(0, 2))
        METRICS["fb_used"].labels(**g).set(used  + random.gauss(0, 100))
        METRICS["fb_free"].labels(**g).set(free)
        METRICS["power"].labels(**g).set(120 + util * 2.8 + random.gauss(0, 5))
        METRICS["temp"].labels(**g).set(45 + util * 0.35 + random.gauss(0, 1))
        METRICS["sm_clock"].labels(**g).set(1600 if util > 20 else 600)
        METRICS["nvlink_tx"].labels(gpu=gpu_id, node=node).set(util * 1e8)

    # vLLM metrics
    queue = max(0, int((util - 70) / 5) + random.randint(0, 2)) if util > 70 else 0
    VLLM["queue_len"].labels(model="llama-3.1-8b").set(queue)
    VLLM["running"].labels(model="llama-3.1-8b").set(max(1, int(util / 20)))
    VLLM["cache_usage"].labels(model="llama-3.1-8b").set(min(1.0, cache_pct))
    VLLM["prompt_tokens"].labels(model="llama-3.1-8b").inc(random.randint(50, 200))
    VLLM["gen_tokens"].labels(model="llama-3.1-8b").inc(random.randint(100, 400))

    t += 1
    time.sleep(2)
```

---

## Part 3: Prometheus config

```bash
mkdir -p prometheus
```

```yaml
# prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerts.yml"

alerting:
  alertmanagers:
  - static_configs:
    - targets: []   # wire up AlertManager if you want actual notifications

scrape_configs:
- job_name: dcgm-simulator
  static_configs:
  - targets: ['dcgm-simulator:9400']
  relabel_configs:
  - source_labels: [__address__]
    target_label: instance

- job_name: vllm
  static_configs:
  - targets: ['dcgm-simulator:9400']   # simulator emits vLLM metrics too
  metric_relabel_configs:
  - source_labels: [__name__]
    regex: 'vllm_.*'
    action: keep
```

```yaml
# prometheus/alerts.yml
groups:
- name: gpu-inference
  rules:

  - alert: VRAMPressureCritical
    expr: |
      (nvidia_dcgm_fi_dev_fb_used /
      (nvidia_dcgm_fi_dev_fb_used + nvidia_dcgm_fi_dev_fb_free)) > 0.90
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "VRAM > 90% on {{ $labels.node }}"
      description: "GPU {{ $labels.gpu }} VRAM usage is {{ $value | humanizePercentage }}. OOM crash risk."

  - alert: InferenceQueueBuilding
    expr: vllm_request_queue_length > 10
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "LLM request queue > 10 (currently {{ $value }})"
      description: "Requests are backing up. Consider scaling out replicas."

  - alert: GPUUnderutilized
    expr: nvidia_dcgm_fi_dev_gpu_util < 15
    for: 10m
    labels:
      severity: info
    annotations:
      summary: "GPU utilisation < 15% for 10 minutes — potential idle cost"
```

---

## Part 4: Grafana dashboard provisioning

```bash
mkdir -p grafana/provisioning/datasources grafana/provisioning/dashboards
```

```yaml
# grafana/provisioning/datasources/prometheus.yaml
apiVersion: 1
datasources:
- name: Prometheus
  type: prometheus
  url: http://prometheus:9090
  isDefault: true
  editable: true
```

```yaml
# grafana/provisioning/dashboards/dashboards.yaml
apiVersion: 1
providers:
- name: default
  folder: AI Inference
  type: file
  options:
    path: /etc/grafana/provisioning/dashboards
```

Save the dashboard JSON at `grafana/provisioning/dashboards/gpu-inference.json`:

```bash
cat > grafana/provisioning/dashboards/gpu-inference.json << 'EOF'
{
  "title": "AI Inference Overview",
  "panels": [
    {
      "title": "GPU Utilization %",
      "type": "timeseries",
      "targets": [{"expr": "nvidia_dcgm_fi_dev_gpu_util", "legendFormat": "GPU {{gpu}} ({{node}})"}],
      "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8}
    },
    {
      "title": "VRAM Usage %",
      "type": "gauge",
      "targets": [{"expr": "avg(nvidia_dcgm_fi_dev_fb_used / (nvidia_dcgm_fi_dev_fb_used + nvidia_dcgm_fi_dev_fb_free)) * 100"}],
      "fieldConfig": {"defaults": {"thresholds": {"steps": [{"value": 0, "color": "green"}, {"value": 80, "color": "yellow"}, {"value": 90, "color": "red"}]}}},
      "gridPos": {"x": 12, "y": 0, "w": 6, "h": 8}
    },
    {
      "title": "Request Queue Depth",
      "type": "timeseries",
      "targets": [{"expr": "vllm_request_queue_length", "legendFormat": "Queue depth"}],
      "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8}
    },
    {
      "title": "GPU Temperature °C",
      "type": "timeseries",
      "targets": [{"expr": "nvidia_dcgm_fi_dev_gpu_temp", "legendFormat": "GPU {{gpu}} temp"}],
      "gridPos": {"x": 12, "y": 8, "w": 12, "h": 8}
    },
    {
      "title": "Token Throughput (tokens/sec)",
      "type": "timeseries",
      "targets": [{"expr": "rate(vllm_generation_tokens_total[1m])", "legendFormat": "Output tokens/sec"}],
      "gridPos": {"x": 0, "y": 16, "w": 12, "h": 8}
    },
    {
      "title": "KV Cache Usage %",
      "type": "timeseries",
      "targets": [{"expr": "vllm_gpu_cache_usage_perc * 100", "legendFormat": "KV cache %"}],
      "gridPos": {"x": 12, "y": 16, "w": 12, "h": 8}
    }
  ],
  "schemaVersion": 38,
  "version": 1
}
EOF
```

---

## Part 5: Start the stack

```bash
docker compose up -d
docker compose ps   # all services should be healthy within ~60s
```

```bash
# Pull a small model into Ollama (CPU-compatible)
docker compose exec ollama ollama pull qwen2:0.5b   # 400MB, runs on any CPU

# Verify model available
curl http://localhost:11434/api/tags | python3 -m json.tool
```

---

## Part 6: Generate traces with Langfuse instrumentation

```bash
pip install langfuse openai
```

```python
# trace_requests.py — send requests and capture traces in Langfuse
import os, time, random
from openai import OpenAI
from langfuse import Langfuse
from langfuse.openai import openai as langfuse_openai

# Setup Langfuse
# First: go to http://localhost:3001, create account, create project, get keys
LANGFUSE_PUBLIC_KEY = "pk-lf-..."   # from Langfuse UI
LANGFUSE_SECRET_KEY = "sk-lf-..."   # from Langfuse UI

langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host="http://localhost:3001",
)

# Patch OpenAI client to auto-trace via Langfuse
client = langfuse_openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",               # Ollama doesn't need a real key
)

PROMPTS = [
    ("support",  "Explain what a Kubernetes pod is in simple terms."),
    ("support",  "What does OOMKilled mean in Kubernetes?"),
    ("devtools", "Write a kubectl command to list all pods in all namespaces."),
    ("devtools", "How do I port-forward to a pod in Kubernetes?"),
    ("support",  "What is the difference between a Deployment and a StatefulSet?"),
]

for i in range(20):
    category, prompt = random.choice(PROMPTS)
    tenant = random.choice(["team-eng", "team-product", "team-data"])

    start = time.perf_counter()
    response = client.chat.completions.create(
        model="qwen2:0.5b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        metadata={             # Langfuse captures these as trace metadata
            "category": category,
            "tenant":   tenant,
        }
    )
    latency_ms = (time.perf_counter() - start) * 1000

    print(f"[{tenant}/{category}] {latency_ms:.0f}ms | "
          f"{response.usage.prompt_tokens}+{response.usage.completion_tokens} tokens")
    time.sleep(0.5)

langfuse.flush()
print("\nTraces visible at: http://localhost:3001")
```

```bash
# Set your Langfuse keys first (from http://localhost:3001 UI)
python3 trace_requests.py
```

---

## Part 7: Verify everything is working

```bash
# Prometheus: check targets are UP
curl -s http://localhost:9090/api/v1/targets | \
  python3 -c "import json,sys; [print(t['labels']['job'], t['health']) for t in json.load(sys.stdin)['data']['activeTargets']]"

# Check a metric exists
curl -s 'http://localhost:9090/api/v1/query?query=nvidia_dcgm_fi_dev_gpu_util' | \
  python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Got {len(d[\"data\"][\"result\"])} GPU metric series')"

# Check alerts
curl -s http://localhost:9090/api/v1/alerts | \
  python3 -c "import json,sys; [print(a['labels']['alertname'], a['state']) for a in json.load(sys.stdin)['data']['alerts']]"
```

**Access the UIs:**
- Grafana:   http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- Langfuse:  http://localhost:3001

---

## Cleanup

```bash
docker compose down -v   # -v removes volumes too
```
