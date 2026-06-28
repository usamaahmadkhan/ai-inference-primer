# Lab 05: Fine-Tuning Pipeline

> Build a LoRA fine-tuning pipeline with eval gate, MLflow tracking, and auto-deploy.

**Phase:** 5 — LLMOps
**GPU required:** No for the pipeline skeleton (Parts 1–3). Recommended for Part 4 (actual training). Google Colab free tier works.
**Time:** 3–4 hours
**Cost:** Free (CPU training is slow but works for a tiny demo model; Colab T4 is free)

---

## Objective

By the end of this lab you will have:
- A LoRA fine-tuning script that runs on CPU (slow) or GPU (fast)
- MLflow tracking experiment results and registering model versions
- An automated evaluation gate that blocks bad models
- An Argo Workflow that chains train → eval → register → deploy

The model quality doesn't matter. The pipeline is the deliverable.

---

## Architecture

```
Fine-Tuning Pipeline
══════════════════════════════════════════════════════════════

  [Training Data]
        │
        ▼
  ┌─────────────┐   fails   ┌─────────────────────┐
  │  LoRA Train │ ─────────►│ GitHub Issue created │
  │  (PyTorch)  │           │ Pipeline blocked     │
  └──────┬──────┘           └─────────────────────┘
         │ model artifact
         ▼
  ┌─────────────┐   score < threshold
  │  Eval Gate  │ ──────────────────────────────────────►  BLOCK
  │  (accuracy, │
  │   latency)  │   score ≥ threshold
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │   MLflow    │  registers version, tags as "Staging"
  │  Registry   │
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ Auto-deploy │  patches vLLM Deployment image tag
  │  to vLLM    │
  └─────────────┘
```

---

## Part 1: Environment setup

```bash
# Create isolated environment
python3 -m venv llmops-lab
source llmops-lab/bin/activate

pip install \
  transformers==4.44.0 \
  peft==0.12.0 \
  datasets==2.20.0 \
  torch==2.3.0 \
  mlflow==2.15.0 \
  accelerate==0.33.0 \
  evaluate==0.4.2 \
  rouge-score==0.1.2

# Start MLflow tracking server (local)
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000 &

echo "MLflow UI: http://localhost:5000"
```

---

## Part 2: LoRA fine-tuning script

```python
#!/usr/bin/env python3
# train.py — LoRA fine-tuning with MLflow tracking
"""
Fine-tunes a small model (GPT-2) with LoRA on a simple instruction dataset.
GPT-2 is not a great chat model, but it's 500MB and runs on any CPU.
The patterns here transfer directly to Llama-3, Mistral, etc.
"""

import mlflow
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

parser = argparse.ArgumentParser()
parser.add_argument("--model",         default="gpt2")
parser.add_argument("--lora-rank",     type=int, default=8)
parser.add_argument("--lora-alpha",    type=int, default=16)
parser.add_argument("--epochs",        type=int, default=1)
parser.add_argument("--batch-size",    type=int, default=4)
parser.add_argument("--max-steps",     type=int, default=50)   # short for lab
parser.add_argument("--output-dir",    default="./lora-output")
parser.add_argument("--mlflow-uri",    default="http://localhost:5000")
args = parser.parse_args()

mlflow.set_tracking_uri(args.mlflow_uri)
mlflow.set_experiment("lora-finetuning-lab")

with mlflow.start_run() as run:
    # Log hyperparameters
    mlflow.log_params({
        "base_model":   args.model,
        "lora_rank":    args.lora_rank,
        "lora_alpha":   args.lora_alpha,
        "epochs":       args.epochs,
        "max_steps":    args.max_steps,
    })

    print(f"MLflow run ID: {run.info.run_id}")

    # Load tokenizer and model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,   # CPU: use float32; GPU: use float16
    )

    # Apply LoRA — only train adapter, not full model
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,             # rank: higher = more params = more expressive
        lora_alpha=args.lora_alpha,   # scaling factor
        lora_dropout=0.1,
        target_modules=["c_attn"],    # GPT-2 attention projection layers
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    trainable_pct = 100 * trainable / total
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable_pct:.2f}%)")
    mlflow.log_metric("trainable_params_pct", trainable_pct)

    # Dataset — tiny instruction dataset from HuggingFace Hub
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:200]")   # 200 examples only

    def tokenize(examples):
        prompts = [
            f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
            for inst, inp, out in zip(
                examples["instruction"], examples["input"], examples["output"]
            )
        ]
        tokens = tokenizer(prompts, truncation=True, max_length=256, padding="max_length")
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    split = tokenized.train_test_split(test_size=0.1, seed=42)

    # Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        logging_steps=10,
        save_steps=args.max_steps,
        evaluation_strategy="steps",
        eval_steps=25,
        report_to="none",           # we handle MLflow manually
        no_cuda=not torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Training...")
    train_result = trainer.train()

    # Log metrics
    mlflow.log_metrics({
        "train_loss":        train_result.training_loss,
        "train_runtime_sec": train_result.metrics["train_runtime"],
    })

    # Save adapter (not full model — just the LoRA weights)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Log artifact to MLflow
    mlflow.log_artifacts(args.output_dir, artifact_path="lora-adapter")

    print(f"Training complete. Loss: {train_result.training_loss:.4f}")
    print(f"Artifacts logged to run: {run.info.run_id}")
```

```bash
python3 train.py --max-steps 50 --epochs 1
# On CPU: ~5–10 minutes for 50 steps
# On GPU: ~1–2 minutes
```

---

## Part 3: Evaluation gate

```python
#!/usr/bin/env python3
# eval_gate.py — automated quality check before model promotion
"""
Runs a simple eval suite against the fine-tuned model.
Compares against a baseline score.
Exits 0 (pass) or 1 (fail) — suitable for CI/CD gates.
"""

import sys, json, mlflow, torch, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--adapter-path",  default="./lora-output")
parser.add_argument("--base-model",    default="gpt2")
parser.add_argument("--run-id",        required=True, help="MLflow run ID from training")
parser.add_argument("--min-score",     type=float, default=0.3)
parser.add_argument("--mlflow-uri",    default="http://localhost:5000")
args = parser.parse_args()

mlflow.set_tracking_uri(args.mlflow_uri)

# Load model
print("Loading model for eval...")
tokenizer = AutoTokenizer.from_pretrained(args.base_model)
tokenizer.pad_token = tokenizer.eos_token
base  = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(base, args.adapter_path)
model.eval()

# Simple eval: measure perplexity on held-out prompts
eval_prompts = [
    "### Instruction:\nExplain what a Kubernetes pod is.\n\n### Input:\n\n### Response:\n",
    "### Instruction:\nWhat is the difference between CPU and memory limits?\n\n### Input:\n\n### Response:\n",
    "### Instruction:\nDescribe how a Dockerfile works.\n\n### Input:\n\n### Response:\n",
]

losses = []
with torch.no_grad():
    for prompt in eval_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
        outputs = model(**inputs, labels=inputs["input_ids"])
        losses.append(outputs.loss.item())

avg_loss  = sum(losses) / len(losses)
# Invert loss to a 0-1 score (lower loss = better = higher score)
eval_score = max(0.0, 1.0 - (avg_loss / 10.0))

print(f"Eval loss (avg):  {avg_loss:.4f}")
print(f"Eval score:       {eval_score:.4f}")
print(f"Min score:        {args.min_score}")

# Log eval results back to the MLflow run
with mlflow.start_run(run_id=args.run_id):
    mlflow.log_metrics({
        "eval_loss":  avg_loss,
        "eval_score": eval_score,
    })
    mlflow.set_tag("eval_status", "passed" if eval_score >= args.min_score else "failed")

if eval_score >= args.min_score:
    print("✓ EVAL PASSED — model is eligible for promotion")
    sys.exit(0)
else:
    print(f"✗ EVAL FAILED — score {eval_score:.4f} below threshold {args.min_score}")
    sys.exit(1)
```

```bash
# Get the run ID from MLflow UI (http://localhost:5000)
# or from the train.py output
MLFLOW_RUN_ID="<your-run-id>"
python3 eval_gate.py --run-id $MLFLOW_RUN_ID
echo "Exit code: $?"
```

---

## Part 4: Register model to MLflow Registry

```python
#!/usr/bin/env python3
# register.py — promote model to MLflow Registry
import mlflow, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run-id",        required=True)
parser.add_argument("--model-name",    default="lora-finetuned-lab")
parser.add_argument("--mlflow-uri",    default="http://localhost:5000")
args = parser.parse_args()

mlflow.set_tracking_uri(args.mlflow_uri)
client = mlflow.tracking.MlflowClient()

# Register model version from the run's artifact
model_uri = f"runs:/{args.run_id}/lora-adapter"
mv = mlflow.register_model(model_uri, args.model_name)

print(f"Registered model: {args.model_name} version {mv.version}")

# Transition to Staging
client.transition_model_version_stage(
    name=args.model_name,
    version=mv.version,
    stage="Staging",
)
print(f"Promoted to Staging")

# Add description
client.update_model_version(
    name=args.model_name,
    version=mv.version,
    description=f"LoRA adapter trained from run {args.run_id}. Eval gate passed.",
)
```

---

## Part 5: Wire it together with a shell pipeline

```bash
#!/usr/bin/env bash
# pipeline.sh — simulates what Argo Workflows would orchestrate

set -euo pipefail
MLFLOW_URI="http://localhost:5000"
MODEL_NAME="lora-finetuned-lab"

echo "═══════════════════════════════════"
echo "  Step 1: Train"
echo "═══════════════════════════════════"
RUN_ID=$(python3 train.py --max-steps 50 2>&1 | grep "MLflow run ID:" | awk '{print $NF}')
echo "Run ID: $RUN_ID"

echo ""
echo "═══════════════════════════════════"
echo "  Step 2: Evaluate"
echo "═══════════════════════════════════"
if python3 eval_gate.py --run-id "$RUN_ID" --mlflow-uri "$MLFLOW_URI"; then
    echo "Gate: PASSED"
else
    echo "Gate: FAILED — pipeline blocked"
    exit 1
fi

echo ""
echo "═══════════════════════════════════"
echo "  Step 3: Register"
echo "═══════════════════════════════════"
python3 register.py --run-id "$RUN_ID" --model-name "$MODEL_NAME" --mlflow-uri "$MLFLOW_URI"

echo ""
echo "═══════════════════════════════════"
echo "  Pipeline complete"
echo "  View at: $MLFLOW_URI"
echo "═══════════════════════════════════"
```

```bash
chmod +x pipeline.sh && ./pipeline.sh
```

Open MLflow at http://localhost:5000 — you should see the experiment run with metrics, the artifact, and the registered model version in Staging.

---

## Part 6 (Optional): Argo Workflows definition

If you have the kind cluster from Lab 03 running, apply this to see the pipeline as a real K8s workflow:

```yaml
# argo-finetune-pipeline.yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  name: lora-pipeline
  namespace: default
spec:
  entrypoint: finetune-pipeline
  arguments:
    parameters:
    - name: base-model
      value: "gpt2"
    - name: max-steps
      value: "50"
    - name: min-eval-score
      value: "0.3"

  templates:
  - name: finetune-pipeline
    dag:
      tasks:
      - name: train
        template: train-step
      - name: evaluate
        template: eval-step
        dependencies: [train]
        arguments:
          parameters:
          - name: run-id
            value: "{{tasks.train.outputs.parameters.run-id}}"
      - name: register
        template: register-step
        dependencies: [evaluate]
        arguments:
          parameters:
          - name: run-id
            value: "{{tasks.train.outputs.parameters.run-id}}"

  - name: train-step
    container:
      image: python:3.11-slim
      command: [sh, -c]
      args: ["pip install transformers peft datasets torch mlflow -q && python3 /scripts/train.py"]
    outputs:
      parameters:
      - name: run-id
        valueFrom:
          path: /tmp/run-id.txt

  - name: eval-step
    inputs:
      parameters:
      - name: run-id
    container:
      image: python:3.11-slim
      command: [sh, -c]
      args: ["python3 /scripts/eval_gate.py --run-id {{inputs.parameters.run-id}}"]

  - name: register-step
    inputs:
      parameters:
      - name: run-id
    container:
      image: python:3.11-slim
      command: [sh, -c]
      args: ["python3 /scripts/register.py --run-id {{inputs.parameters.run-id}}"]
```

```bash
# Install Argo Workflows on kind cluster
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/latest/download/install.yaml
kubectl apply -f argo-finetune-pipeline.yaml
argo watch lora-pipeline
```
