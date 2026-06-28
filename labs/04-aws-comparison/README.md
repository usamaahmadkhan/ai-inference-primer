# Lab 04: AWS Comparison

> Deploy the same model three ways. Build a decision matrix with real numbers.

**Phase:** 4 — AWS AI Stack
**GPU required:** No — Bedrock and SageMaker Serverless work without provisioning GPU instances.
**Time:** 2–3 hours
**Cost:** ~$1–5 (Bedrock and SageMaker Serverless have minimal cost at lab scale; avoid the EC2 GPU path if cost-sensitive)

---

## Objective

By the end of this lab you will have:
- Called an LLM via Amazon Bedrock (zero infra, API-only)
- Deployed a model on SageMaker Serverless Inference (managed, no GPU provisioning)
- *(Optional, ~$2/hr)* Deployed on a SageMaker Real-time endpoint (GPU)
- A side-by-side cost and latency comparison table
- A decision matrix you'd actually use in a design review

---

## Prerequisites

```bash
# AWS CLI configured with a profile that has these permissions:
#   bedrock:InvokeModel
#   sagemaker:CreateModel, CreateEndpointConfig, CreateEndpoint
#   iam:PassRole (for SageMaker execution role)
aws sts get-caller-identity

pip3 install boto3 anthropic
```

---

## Part 1: Amazon Bedrock

Bedrock is the zero-infra path. No instances, no containers — just an API call.

```bash
# First, enable model access in the AWS Console:
# Bedrock → Model access → Request access to: Anthropic Claude Haiku 3
# (Takes ~1 minute, free to enable)
aws bedrock list-foundation-models \
  --region us-east-1 \
  --query 'modelSummaries[?contains(modelId, `anthropic`)].modelId' \
  --output table
```

```python
#!/usr/bin/env python3
# bedrock_test.py
import boto3, json, time

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

PROMPT = "Explain Kubernetes resource requests and limits in 3 sentences."

def invoke_bedrock(prompt: str, model_id: str = "anthropic.claude-3-haiku-20240307-v1:0") -> dict:
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": prompt}]
    })

    start = time.perf_counter()
    response = bedrock.invoke_model(body=body, modelId=model_id)
    first_byte = time.perf_counter()

    result = json.loads(response["body"].read())
    end = time.perf_counter()

    return {
        "output": result["content"][0]["text"],
        "input_tokens":  result["usage"]["input_tokens"],
        "output_tokens": result["usage"]["output_tokens"],
        "e2e_latency_ms": (end - start) * 1000,
        "model": model_id,
    }

# Run 5 times to get stable latency numbers
latencies = []
for i in range(5):
    result = invoke_bedrock(PROMPT)
    latencies.append(result["e2e_latency_ms"])
    print(f"Run {i+1}: {result['e2e_latency_ms']:.0f}ms | "
          f"{result['output_tokens']} output tokens")

print(f"\nMedian latency: {sorted(latencies)[2]:.0f}ms")
print(f"p99 latency:    {max(latencies):.0f}ms")

# Cost calculation
INPUT_PRICE  = 0.00025 / 1000  # Claude Haiku: $0.00025 per 1K input tokens
OUTPUT_PRICE = 0.00125 / 1000  # Claude Haiku: $0.00125 per 1K output tokens
cost = result["input_tokens"] * INPUT_PRICE + result["output_tokens"] * OUTPUT_PRICE
print(f"Cost per request: ${cost:.6f}")
print(f"Cost per 1M output tokens: ${OUTPUT_PRICE * 1_000_000:.2f}")
```

```bash
python3 bedrock_test.py
```

---

## Part 2: SageMaker Serverless Inference

Serverless inference: no always-on GPU, pay per invocation, cold start on first request.

```python
#!/usr/bin/env python3
# sagemaker_serverless.py
import boto3, json, time, sagemaker
from sagemaker.huggingface import HuggingFaceModel

sess   = sagemaker.Session()
region = sess.boto_region_name
role   = sagemaker.get_execution_role()   # needs AmazonSageMakerFullAccess

# Deploy TGI with a small model on serverless
# distilgpt2 is tiny (300MB) — good for lab purposes without GPU cost
hub_env = {
    "HF_MODEL_ID":                 "distilgpt2",
    "HF_TASK":                     "text-generation",
    "SM_NUM_GPUS":                 json.dumps(0),   # CPU inference for serverless
    "MAX_INPUT_LENGTH":            json.dumps(512),
    "MAX_TOTAL_TOKENS":            json.dumps(768),
}

print("Creating SageMaker model...")
model = HuggingFaceModel(
    env=hub_env,
    role=role,
    transformers_version="4.37",
    pytorch_version="2.1",
    py_version="py310",
)

print("Deploying serverless endpoint (2–5 min)...")
predictor = model.deploy(
    serverless_inference_config=sagemaker.serverless.ServerlessInferenceConfig(
        memory_size_in_mb=4096,    # 1024, 2048, 3072, 4096, 6144 options
        max_concurrency=5,
    )
)
print(f"Endpoint: {predictor.endpoint_name}")

# Test with cold start and warm requests
PROMPT = "Kubernetes is a container orchestration system that"

print("\n--- Cold start (first request) ---")
start = time.perf_counter()
response = predictor.predict({"inputs": PROMPT, "parameters": {"max_new_tokens": 50}})
cold_latency = (time.perf_counter() - start) * 1000
print(f"Cold start latency: {cold_latency:.0f}ms")
print(f"Output: {response[0]['generated_text'][:100]}...")

print("\n--- Warm requests ---")
warm_latencies = []
for i in range(5):
    start = time.perf_counter()
    predictor.predict({"inputs": PROMPT, "parameters": {"max_new_tokens": 50}})
    warm_latencies.append((time.perf_counter() - start) * 1000)
    print(f"  Run {i+1}: {warm_latencies[-1]:.0f}ms")

print(f"\nMedian warm latency: {sorted(warm_latencies)[2]:.0f}ms")
print(f"Cold start overhead: {cold_latency - sorted(warm_latencies)[2]:.0f}ms")
```

```bash
python3 sagemaker_serverless.py
```

**⚠️ Delete the endpoint when done:**

```bash
python3 -c "
import boto3
sm = boto3.client('sagemaker')
# List your endpoints to find the name
endpoints = sm.list_endpoints(StatusEquals='InService')
for ep in endpoints['Endpoints']:
    print(ep['EndpointName'])
"
# Then:
# aws sagemaker delete-endpoint --endpoint-name <name>
```

---

## Part 3 (Optional, ~$2): SageMaker Real-Time GPU Endpoint

> **GPU required:** This part uses ml.g5.2xlarge (~$1.83/hr). Skip if cost-sensitive.

```python
#!/usr/bin/env python3
# sagemaker_realtime.py
import boto3, json, time, sagemaker
from sagemaker.huggingface import HuggingFaceModel

sess   = sagemaker.Session()
role   = sagemaker.get_execution_role()

hub_env = {
    "HF_MODEL_ID":      "microsoft/phi-2",   # ~3B, fits on g5.2xlarge
    "HF_TASK":          "text-generation",
    "SM_NUM_GPUS":      json.dumps(1),
    "MAX_INPUT_LENGTH": json.dumps(2048),
    "MAX_TOTAL_TOKENS": json.dumps(4096),
}

model = HuggingFaceModel(
    env=hub_env,
    role=role,
    image_uri=sagemaker.image_uris.retrieve(
        framework="huggingface-llm",
        region=sess.boto_region_name,
        version="2.0.2"
    ),
)

print("Deploying real-time GPU endpoint (5–10 min)...")
start_deploy = time.perf_counter()
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge",
    container_startup_health_check_timeout=300,
)
deploy_time = time.perf_counter() - start_deploy
print(f"Deployment time: {deploy_time:.0f}s")

# Benchmark
PROMPT = "Explain Kubernetes resource requests and limits."
latencies, token_counts = [], []

for i in range(10):
    start = time.perf_counter()
    resp  = predictor.predict({
        "inputs": PROMPT,
        "parameters": {"max_new_tokens": 150, "do_sample": False}
    })
    latency = (time.perf_counter() - start) * 1000
    output  = resp[0]["generated_text"].replace(PROMPT, "")
    tokens  = len(output.split())  # rough word count as proxy
    latencies.append(latency)
    token_counts.append(tokens)
    print(f"Run {i+1}: {latency:.0f}ms | ~{tokens} output words")

print(f"\nMedian E2E latency: {sorted(latencies)[5]:.0f}ms")
print(f"p99 E2E latency:    {max(latencies):.0f}ms")
print(f"Cost per hour:      $1.83 (ml.g5.2xlarge on-demand)")
```

---

## Part 4: Fill in your decision matrix

Record your measurements and complete this matrix:

```
Deployment Comparison: <your model> on AWS
═══════════════════════════════════════════════════════════════════════════════

Metric                  │ Bedrock          │ SM Serverless    │ SM Real-time GPU
────────────────────────┼──────────────────┼──────────────────┼──────────────────
E2E Latency p50         │                  │                  │
E2E Latency p99         │                  │                  │
Cold Start Latency      │ N/A (always on)  │                  │ N/A (always on)
Cost per 1M tokens      │                  │                  │
Cost per hour (idle)    │ $0               │ $0               │ ~$1.83+
GPU provisioning needed │ No               │ No               │ Yes
Model control           │ No               │ Limited          │ Yes
Custom fine-tune        │ No               │ No               │ Yes
Setup time              │ ~10 min          │ ~15 min          │ ~15 min
Ops burden              │ None             │ Low              │ Medium

When to choose:
  Bedrock:        ____________________________________________
  SM Serverless:  ____________________________________________
  SM Real-time:   ____________________________________________
  EKS + vLLM:     ____________________________________________
```

---

## Part 5: The break-even calculation

```python
#!/usr/bin/env python3
# breakeven.py — when does self-hosted beat managed?

# Fill in your numbers from the lab
bedrock_cost_per_1m_output = 1.25   # Claude Haiku output tokens
eks_hourly_cost            = 1.21   # g5.2xlarge on-demand
eks_throughput_tok_per_hr  = 7_200_000  # ~2000 tok/s × 3600

eks_cost_per_1m = (eks_hourly_cost / eks_throughput_tok_per_hr) * 1_000_000

print(f"Bedrock (Haiku) cost/1M output tokens: ${bedrock_cost_per_1m_output:.2f}")
print(f"EKS + vLLM cost/1M output tokens:      ${eks_cost_per_1m:.3f}")
print(f"EKS is {bedrock_cost_per_1m_output / eks_cost_per_1m:.1f}× cheaper per token at full utilisation")
print()

# Break-even: what request rate makes EKS worth it?
# Below break-even: GPU is idle, Bedrock is cheaper
# Above break-even: GPU utilised enough to beat per-token pricing

for utilisation_pct in [10, 25, 50, 75, 90, 100]:
    effective_tok_hr = eks_throughput_tok_per_hr * (utilisation_pct / 100)
    effective_cost   = (eks_hourly_cost / effective_tok_hr) * 1_000_000 if effective_tok_hr > 0 else float('inf')
    cheaper = "EKS ✓" if effective_cost < bedrock_cost_per_1m_output else "Bedrock ✓"
    print(f"GPU utilisation {utilisation_pct:>3}%: EKS = ${effective_cost:>7.3f}/1M → {cheaper}")
```

```bash
python3 breakeven.py
```

---

## Cleanup

```bash
# Delete any SageMaker endpoints (they cost money when idle)
aws sagemaker list-endpoints --status-filter InService \
  --query 'Endpoints[].EndpointName' --output text | \
  tr '\t' '\n' | xargs -I{} aws sagemaker delete-endpoint --endpoint-name {}
```
