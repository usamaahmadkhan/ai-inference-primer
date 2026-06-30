# Phase 5 Resources: LLMOps

Curated, in the order we'd actually read/use them.

---

## Evaluation frameworks and articles

**[Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)** — Anthropic Engineering, Jan 2026
The clearest available framework for eval vocabulary and design: task, trial, grader, transcript, outcome, eval harness vs agent harness. Covers the three grader types (code-based, model-based, human), capability vs regression evals, and `pass@k` vs `pass^k` for handling non-determinism. Written for agents specifically, but the grader taxonomy applies directly to the single-turn eval gate in this phase's Fine-Tuning Pipelines section.

**[OpenAI Evals](https://github.com/openai/evals)** — GitHub
Open-source framework and registry of benchmark specs. An eval is a YAML file with an input, dataset, and grading method (string match, model-graded, or custom). Supports prompt chains and tool-using agents via the Completion Function Protocol. This is the most direct path to a working CI eval gate — start with `docs/build-eval.md`.

**[Anthropic evals](https://github.com/anthropics/evals)** — GitHub
Model-written behavioral datasets from the "Discovering Language Model Behaviors with Model-Written Evaluations" paper (Perez et al., 2022). Datasets cover persona consistency, sycophancy, advanced-AI-risk-adjacent behaviors, and gender bias (Winogender). This is a dataset collection, not a serving framework — use it for behavioral/safety regression checks alongside your primary eval harness, not as a replacement for one.

**[MLflow Evaluate docs](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)**
Built-in evaluation tightly coupled to the MLflow Model Registry workflow already covered in this phase. Lower ceiling than OpenAI Evals for complex grading, but zero extra infrastructure if you're already using MLflow for registry and tracking.

---

## LLMOps fundamentals

**Full Stack LLM Bootcamp** (free, YouTube — search "Full Stack LLM Bootcamp")
The most practical, infrastructure-aware LLMOps content publicly available. Built by the team behind Weights & Biases. Covers the full lifecycle from prompt engineering through deployment and monitoring, with a production engineer's framing rather than a research framing.

**Weights & Biases free courses** — wandb.courses
MLflow and W&B solve the same problem (experiment tracking, model registry) with different UX. The concepts transfer directly regardless of which one your team has standardized on.

**[Langfuse self-hosted docs](https://langfuse.com/docs/deployment/self-host)**
Self-host this. Do not send production LLM traces to a third-party SaaS unless you've explicitly cleared that with security/compliance. The self-hosted deployment runs comfortably on the same K8s cluster as everything else in this guide.

**[Kubeflow Pipelines docs](https://www.kubeflow.org/docs/components/pipelines/)**
If you're orchestrating fine-tuning jobs on Kubernetes (per the Fine-Tuning Pipelines section), this is the most K8s-native option. Argo Workflows is the lighter-weight alternative used in Lab 05.

---

## Reading order

If you're going through this phase for the first time: start with the Demystifying Evals article for vocabulary, then skim OpenAI Evals' `build-eval.md` to see the grader taxonomy implemented in actual YAML, then do Lab 05 to build the pipeline end to end. Come back to the Full Stack LLM Bootcamp once you've built one real pipeline — it'll click faster with hands-on context already in place.
