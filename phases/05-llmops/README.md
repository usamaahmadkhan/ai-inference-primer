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

## Evaluation Frameworks

The eval gate in the fine-tuning pipeline above is doing real work, but "run eval suite" glosses over a real engineering discipline. Evals are the test suite for a system that doesn't fail loudly — a regressed model still returns 200 OK with confident, plausible, wrong output. Without evals you're flying blind between deploys.

```
Why evals are not optional
══════════════════════════════════════════════════════════

Traditional software regression          Model regression
─────────────────────────────             ─────────────────────────────
Unit test fails                          Output looks fine, reads fine,
→ CI blocks the merge                    is subtly worse
→ loud, deterministic signal             → no exception, no failed
                                            health check, no alert
                                          → ships straight to prod
                                          → shows up as a drop in a
                                            product metric weeks later
```

### Open-source eval tooling

Two repos come up constantly and solve different problems — know which one you need.

**[OpenAI Evals](https://github.com/openai/evals)** is a framework plus a registry of benchmark specs. An eval is a YAML file: an input, a grading method (string match, model-graded, or custom), and a dataset. It supports both single-turn completions and, via its Completion Function Protocol, prompt chains and tool-using agents. This is the tool to reach for when you need a CI-style harness that runs a battery of input/output checks against every model or prompt change — directly pluggable into the gate shown above.

**[Anthropic evals](https://github.com/anthropics/evals)** is a different category of artifact: model-written behavioral datasets from the "Discovering Language Model Behaviors with Model-Written Evaluations" paper. It's not a serving harness — it's a dataset collection for probing things like sycophancy, persona consistency, and advanced-AI-risk-adjacent behaviors (power-seeking tendencies, self-preservation framing). Useful as a reference for behavioral/safety regression testing, not as your primary CI eval engine.

```
Tool                        Category                  Use it for
──────────────────────────────────────────────────────────────────────────
OpenAI Evals                Framework + registry       Your CI eval gate;
                                                        custom task definitions;
                                                        agent/chain evaluation
                                                        via Completion Functions

Anthropic evals             Behavioral dataset          Safety/behavioral
                                                        regression checks;
                                                        sycophancy, persona,
                                                        risk-indicator probes

MLflow eval (built-in)      Tracking-integrated         Lightweight checks tied
                                                        directly to a registered
                                                        model version (what the
                                                        Fine-Tuning Pipelines
                                                        section above uses)
```

### Agent evals are a harder problem

Everything above assumes single-turn input → output → grade. The moment you're evaluating an agent — something that calls tools, modifies state across multiple turns, and can find creative solutions a rigid grader didn't anticipate — the eval design gets materially harder. Anthropic's [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) is the most useful field guide on this available right now: it defines the vocabulary you need (task, trial, grader, transcript, outcome, eval harness vs agent harness), covers the three grader types (code-based, model-based, human) and when to combine them, and introduces `pass@k` vs `pass^k` for handling the non-determinism that agents introduce into your eval scores.

This matters even if you're not building agents yet, because the LLMOps gate described in this phase — eval suite → score vs baseline → pass/fail — is the single-turn special case of exactly this framework. See [Phase 7: Advanced Topics](../07-advanced-topics/) for where agent evaluation connects to agent infrastructure specifically.

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

See [resources.md](./resources.md) for the full curated list with descriptions.

**Essential:**
1. Full Stack LLM Bootcamp — free on YouTube, best practical LLMOps content available
2. Weights & Biases free courses — MLflow and W&B are interchangeable concepts
3. Langfuse docs — self-host it, don't send prod traces to SaaS
4. [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) — Anthropic, Jan 2026. The eval vocabulary and grader taxonomy in this phase's gate design comes directly from this framework.
5. [OpenAI Evals](https://github.com/openai/evals) — clone it, read `docs/build-eval.md`, run the existing registry before writing your own YAML specs.

---

## Lab

**[→ Lab 05: Fine-Tuning Pipeline](../../labs/05-fine-tuning-pipeline/)**

Build an end-to-end LoRA fine-tuning pipeline: Argo Workflow → Kubeflow TrainingJob → automated eval gate → MLflow registry → auto-deploy to vLLM on pass, open GitHub issue on fail. The pipeline itself is the deliverable, not the model quality.
