# Phase 5: LLMOps

> The lifecycle management layer. Most teams build this last. You should build it first.

**Duration:** 3–4 weeks
**Lab:** [05-fine-tuning-pipeline](../../labs/05-fine-tuning-pipeline/)

---

## The LLMOps stack

```
LLMOps: What you're building
══════════════════════════════════════════════════════════

  Data / Prompts          Model Lifecycle         Serving
  ──────────────          ───────────────         ───────
  ┌──────────┐            ┌─────────────┐         ┌─────┐
  │ Training │            │   Model     │         │     │
  │  Data    │──►train──► │  Registry   │──────►  │ Prod│
  │          │            │  (MLflow,   │ deploy  │     │
  │ Prompts  │──►eval──►  │   W&B, HF   │         │     │
  │ (version │            │   Hub)      │◄──────  │     │
  │  control)│            └─────────────┘ rollback└─────┘
  └──────────┘                   │
                                 │ eval gate
                                 ▼
                           ┌─────────────┐
                           │  Evaluation │
                           │  Framework  │
                           │  (pass/fail)│
                           └─────────────┘

  Observability                CI/CD
  ─────────────                ─────
  ┌──────────────┐             ┌──────────────────────────┐
  │  Langfuse /  │             │  PR → train → eval →     │
  │  LangSmith   │             │  registry → auto-deploy  │
  │  (tracing)   │             │  if eval passes          │
  └──────────────┘             └──────────────────────────┘
```

---

## Model Registries

Models need the same artifact management rigor as code. You need versioning, staging environments, and promotion workflows.

### MLflow Model Registry

```python
import mlflow
import mlflow.pyfunc

# Log a model after training or evaluation
with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        "base_model": "meta-llama/Llama-3.1-8B",
        "lora_rank": 16,
        "training_steps": 1000,
        "eval_score": 0.87
    })

    # Log the model artifact
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=YourModelWrapper(),
        registered_model_name="llama-3-finetuned-support"
    )

# Promote to production via Registry API
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="llama-3-finetuned-support",
    version=3,
    stage="Production"  # None → Staging → Production → Archived
)
```

### Promotion workflow

```
Model Lifecycle Stages
═══════════════════════

  ┌───────────────────────────────────────────────────────────┐
  │                  MLflow Model Registry                    │
  │                                                           │
  │  version-1 [Archived]                                     │
  │  version-2 [Archived]                                     │
  │  version-3 [Production]  ◄── currently serving traffic   │
  │  version-4 [Staging]     ◄── shadow testing              │
  │  version-5 [None]        ◄── just registered             │
  │                                                           │
  │  Promotion path:                                          │
  │  None → Staging → Production                              │
  │                     │                                     │
  │                 eval gate (automated)                     │
  │                 human approval (optional)                 │
  └───────────────────────────────────────────────────────────┘
```

---

## Deployment Strategies for Models

Standard K8s rollout strategies aren't enough for LLMs. A model quality regression doesn't show up as a HTTP 500 — it shows up as subtly worse outputs that your health check can't detect.

```
Model Deployment Strategies
═════════════════════════════

Rolling Update (not enough)         Shadow Deployment (better)
───────────────────────────         ──────────────────────────
Traffic:                            Traffic:
  Old ████████                        Old ████████  ◄── serves users
  New ░░░░████                        New ████████  ◄── receives same
       slowly replaces                              requests, responses
       old version                                  NOT returned to user

Problem: bad model affects          Benefit: compare outputs,
users before you catch it           latency, cost in prod
                                    before cutover

Canary (for model rollouts)         A/B Testing (for evaluation)
───────────────────────────         ────────────────────────────
Traffic:                            Traffic:
  Old ████████████████               Group A ████  ◄── model version 1
  New ████                           Group B ████  ◄── model version 2

5–10% traffic to new model          Measure quality metrics
Monitor quality metrics             across groups, not just
Expand if metrics hold              error rates

Rollback on:                        Use for: model comparison,
  - Quality drop > threshold           prompt experiments,
  - Latency SLO breach                 feature rollouts
  - Error rate increase
```

### Implementing canary with Argo Rollouts

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: llm-inference
spec:
  replicas: 10
  strategy:
    canary:
      steps:
      - setWeight: 10        # 10% to new model version
      - pause: {duration: 30m}  # wait and observe
      - analysis:            # automated quality check
          templates:
          - templateName: llm-quality-check
      - setWeight: 50
      - pause: {duration: 1h}
      - setWeight: 100
      canaryService: llm-inference-canary
      stableService: llm-inference-stable
      trafficRouting:
        istio:
          virtualService:
            name: llm-inference-vsvc
```

---

## Fine-Tuning Pipelines

LoRA (Low-Rank Adaptation) lets you fine-tune a model by training only a small adapter layer, not the full weights. The infrastructure cost is dramatically lower than full fine-tuning.

```
LoRA Fine-Tuning: What changes
════════════════════════════════

Full Fine-tuning (expensive):      LoRA Fine-tuning (practical):
──────────────────────────────     ─────────────────────────────
Training: all 7B parameters        Training: ~20M adapter params
VRAM: 4–10x model size             VRAM: base model + small delta
Storage: full model copy           Storage: base model + 50–200MB
Time: hours–days                   Time: 30min–4hrs

Result: new 14GB model file        Result: 14GB base + 50MB LoRA
                                   (can hot-swap adapters!)
```

### Pipeline architecture

```
Fine-Tuning Pipeline (Argo Workflows + Kubeflow)
═════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────┐
  │                   Argo Workflow                          │
  │                                                          │
  │  Step 1: Data Prep          Step 2: Training             │
  │  ┌─────────────────┐        ┌──────────────────────┐    │
  │  │ Pull dataset    │──────► │ Kubeflow TrainingJob  │    │
  │  │ Tokenize        │        │ (PyTorchJob)          │    │
  │  │ Train/eval split│        │                      │    │
  │  └─────────────────┘        │ nvidia.com/gpu: 2     │    │
  │                             │ LoRA rank: 16         │    │
  │                             │ epochs: 3             │    │
  │                             └──────────┬───────────┘    │
  │                                        │                 │
  │  Step 3: Evaluation         Step 4: Gate                 │
  │  ┌─────────────────┐        ┌──────────────────────┐    │
  │  │ Run eval suite  │◄───────│ Load model artifact  │    │
  │  │ Accuracy        │        │ from S3              │    │
  │  │ ROUGE score     │        └──────────────────────┘    │
  │  │ Latency bench   │                                     │
  │  └────────┬────────┘                                     │
  │           │ score ≥ threshold?                           │
  │    ┌──────┴──────┐                                       │
  │    │ Yes         │ No                                    │
  │    ▼             ▼                                       │
  │  Register      Open GitHub                               │
  │  to MLflow     Issue (block                              │
  │  + deploy      auto-deploy)                              │
  └──────────────────────────────────────────────────────────┘
```

---

## Prompt Version Control

Prompts are code. Version them, test them, gate on them.

```
Prompt Lifecycle with Langfuse
════════════════════════════════

Developer edits prompt
         │
         ▼
┌─────────────────┐
│  Langfuse       │  prompts stored as versioned artifacts
│  Prompt         │  with metadata (author, date, eval scores)
│  Management     │
└────────┬────────┘
         │  SDK fetch
         ▼
┌─────────────────┐
│  Application    │  prompt = langfuse.get_prompt("support-v3")
│  fetches prompt │  response = llm.complete(prompt.compile(user_input))
│  at runtime     │
└────────┬────────┘
         │  logs request + response + latency + cost
         ▼
┌─────────────────┐
│  Langfuse       │  full trace visible in UI
│  Tracing        │  link trace back to prompt version
└─────────────────┘
```

---

## CI/CD for Models

```
Model CI/CD Pipeline
═════════════════════

  Code PR with model config change
           │
           ▼
  ┌────────────────────────────────────────────────────┐
  │  GitHub Actions / ArgoCD                           │
  │                                                    │
  │  1. Lint & validate config                         │
  │  2. Trigger fine-tune (if training data changed)   │
  │  3. Run eval suite against new model               │
  │  4. Check eval score vs baseline in MLflow         │
  │  5a. PASS → push to registry → trigger Argo        │
  │            Rollout (canary deploy)                 │
  │  5b. FAIL → block merge, post results to PR        │
  └────────────────────────────────────────────────────┘

# Key principle: the same gate that blocks bad code
# should block bad models. Automate the boring part.
# Require human review only for major version changes.
```

---

## Resources

See [resources.md](./resources.md) for the full curated list.

**Essential:**
1. Full Stack LLM Bootcamp — free on YouTube, best practical LLMOps content available
2. Weights & Biases free courses — MLflow and W&B are interchangeable concepts
3. Langfuse docs — self-host it, don't send prod traces to SaaS

---

## Lab

**[→ Lab 05: Fine-Tuning Pipeline](../../labs/05-fine-tuning-pipeline/)**

Build an end-to-end LoRA fine-tuning pipeline: Argo Workflow → Kubeflow TrainingJob → automated eval gate → MLflow registry → auto-deploy to vLLM on pass, open GitHub issue on fail. The pipeline itself is the deliverable, not the model quality.
