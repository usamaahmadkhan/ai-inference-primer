# Lab 03: EKS GPU Cluster

> Stand up the GPU infrastructure baseline. Everything else in this guide runs on top of this.

**Phase:** 3 — Kubernetes for AI
**GPU required:** No for Parts 1–3 (local simulation with kind). Yes for Parts 4–6 (real EKS + GPU).
**Time:** Part 1–3: 2–3 hours (free). Part 4–6: 3–4 hours (~$5–15 in AWS costs).
**Cost:** Free for local simulation. Parts 4–6 require an AWS account.

---

## Objective

By the end of this lab you will have:
- Understood GPU Operator component architecture by running it locally (sans real GPU)
- Deployed DCGM Exporter and wired it to Prometheus
- *(GPU path)* A real EKS cluster with Karpenter provisioning GPU nodes on demand
- *(GPU path)* vLLM running on a Kubernetes GPU node with HPA on queue depth

---

## Architecture

```
What you're building
═════════════════════

Local (Parts 1–3)                 AWS (Parts 4–6)
──────────────────                ────────────────────────────────
kind cluster                      EKS cluster
  └── GPU Operator (no GPU)         ├── System node group (CPU)
  └── Prometheus                    ├── Karpenter
  └── Grafana                       │     └── GPU NodePool (g5.xlarge)
  └── Fake DCGM metrics             ├── GPU Operator
                                    ├── DCGM Exporter → Prometheus
                                    └── vLLM Deployment + HPA
```

---

## Part 1: Local cluster with kind

```bash
# Install kind if needed
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.23.0/kind-linux-amd64
chmod +x kind && sudo mv kind /usr/local/bin/

# Install kubectl if needed
curl -LO "https://dl.k8s.io/release/$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Create a 3-node kind cluster
cat <<EOF | kind create cluster --name ai-infra-lab --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
- role: worker
  labels:
    workload-type: gpu-simulation
- role: worker
  labels:
    workload-type: system
EOF

kubectl get nodes
```

---

## Part 2: Deploy GPU Operator (simulation mode)

Without real GPUs, the GPU Operator's driver and device plugin DaemonSets will fail on their GPU checks — but deploying them teaches you the component architecture and lets you verify everything else (DCGM, toolkit, NFD) is wired correctly.

```bash
# Add NVIDIA Helm repo
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Node Feature Discovery (GPU Operator dependency)
helm install nfd node-feature-discovery/node-feature-discovery \
  --namespace node-feature-discovery \
  --create-namespace \
  --version 0.15.4

# Install GPU Operator with driver disabled (no GPU nodes)
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=false \
  --set toolkit.enabled=false \
  --set devicePlugin.enabled=false \
  --set dcgmExporter.enabled=false \
  --set migManager.enabled=false \
  --wait --timeout=120s

kubectl get pods -n gpu-operator
```

**What each component does:**

```
gpu-operator pod            → Watches for GPU nodes, manages lifecycle
nvidia-driver-daemonset     → Installs NVIDIA kernel driver (needs real GPU)
nvidia-container-toolkit    → Configures containerd to mount GPUs
nvidia-device-plugin        → Advertises nvidia.com/gpu to K8s scheduler
nvidia-dcgm-exporter        → Exports GPU metrics to Prometheus
```

---

## Part 3: Observability stack with simulated GPU metrics

Real DCGM requires real GPUs. For local learning, deploy a metrics simulator alongside the real Prometheus + Grafana stack.

```bash
# Deploy kube-prometheus-stack (Prometheus + Grafana + AlertManager)
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set grafana.adminPassword=admin \
  --set prometheus.prometheusSpec.retention=2h \
  --wait --timeout=300s

kubectl get pods -n monitoring
```

Deploy the GPU metrics simulator:

```bash
cat <<'EOF' | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dcgm-simulator
  namespace: monitoring
  labels:
    app: dcgm-simulator
spec:
  replicas: 2
  selector:
    matchLabels:
      app: dcgm-simulator
  template:
    metadata:
      labels:
        app: dcgm-simulator
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9400"
    spec:
      containers:
      - name: simulator
        image: python:3.11-slim
        command: ["/bin/sh", "-c"]
        args:
        - |
          pip install prometheus-client -q
          python3 - <<'PYEOF'
          import time, random, math
          from prometheus_client import start_http_server, Gauge

          # Simulate realistic GPU metrics
          gpu_util    = Gauge('nvidia_dcgm_fi_dev_gpu_util',    'GPU utilization',    ['gpu', 'node'])
          fb_used     = Gauge('nvidia_dcgm_fi_dev_fb_used',     'VRAM used MiB',      ['gpu', 'node'])
          fb_free     = Gauge('nvidia_dcgm_fi_dev_fb_free',     'VRAM free MiB',      ['gpu', 'node'])
          power       = Gauge('nvidia_dcgm_fi_dev_power_usage', 'Power draw W',       ['gpu', 'node'])
          temp        = Gauge('nvidia_dcgm_fi_dev_gpu_temp',    'Temperature C',      ['gpu', 'node'])

          start_http_server(9400)
          t = 0
          while True:
              # Simulate realistic load pattern with noise
              base_util = 65 + 20 * math.sin(t / 30) + random.gauss(0, 5)
              util      = max(0, min(100, base_util))
              vram_used = 14000 + util * 60 + random.gauss(0, 200)
              vram_free = 24576 - vram_used

              for gpu_id in ['0', '1']:
                  node = f'node-{gpu_id}'
                  gpu_util.labels(gpu=gpu_id, node=node).set(util)
                  fb_used.labels(gpu=gpu_id, node=node).set(vram_used)
                  fb_free.labels(gpu=gpu_id, node=node).set(max(0, vram_free))
                  power.labels(gpu=gpu_id, node=node).set(180 + util * 2)
                  temp.labels(gpu=gpu_id, node=node).set(55 + util * 0.3)

              t += 1
              time.sleep(1)
          PYEOF
        ports:
        - containerPort: 9400
---
apiVersion: v1
kind: Service
metadata:
  name: dcgm-simulator
  namespace: monitoring
  labels:
    app: dcgm-simulator
spec:
  selector:
    app: dcgm-simulator
  ports:
  - port: 9400
    name: metrics
EOF

# Create ServiceMonitor so Prometheus scrapes the simulator
cat <<'EOF' | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: dcgm-simulator
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: dcgm-simulator
  endpoints:
  - port: metrics
    interval: 15s
EOF
```

### Access Grafana

```bash
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80 &
open http://localhost:3000   # admin / admin
```

Import dashboard ID **12239** (NVIDIA DCGM Exporter Dashboard) from grafana.com — it works with both real DCGM metrics and the simulator's metrics since they use identical metric names.

**Verify metrics are flowing:**

```bash
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090 &
# Open http://localhost:9090 and query:
# nvidia_dcgm_fi_dev_gpu_util
# nvidia_dcgm_fi_dev_fb_used
```

---

## Part 4: Real EKS cluster (AWS, GPU optional)

> **Cost warning:** The EKS control plane costs ~$0.10/hr. System nodes ~$0.05/hr. GPU nodes only spin up when you deploy a GPU workload (Karpenter). Estimated cost for this lab: $5–15 depending on how long GPU nodes run.

```bash
# Prerequisites
aws --version          # AWS CLI v2
eksctl version         # >= 0.180
kubectl version --client

# Set your region
export AWS_REGION=us-east-1
export CLUSTER_NAME=ai-infra-lab

# Create cluster with system nodes only (no GPU yet — Karpenter handles that)
eksctl create cluster \
  --name $CLUSTER_NAME \
  --region $AWS_REGION \
  --version 1.30 \
  --nodegroup-name system \
  --node-type m5.xlarge \
  --nodes 2 \
  --nodes-min 2 \
  --nodes-max 4 \
  --managed

kubectl get nodes
```

---

## Part 5: Install Karpenter + GPU Operator on EKS

```bash
# Install Karpenter (using eksctl for IAM setup)
eksctl create iamserviceaccount \
  --cluster $CLUSTER_NAME \
  --name karpenter \
  --namespace karpenter \
  --role-name KarpenterControllerRole-$CLUSTER_NAME \
  --attach-policy-arn arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):policy/KarpenterControllerPolicy-$CLUSTER_NAME \
  --approve

helm install karpenter oci://public.ecr.aws/karpenter/karpenter \
  --version "0.37.0" \
  --namespace karpenter \
  --create-namespace \
  --set settings.clusterName=$CLUSTER_NAME \
  --set settings.interruptionQueue=$CLUSTER_NAME

# Apply the GPU NodePool from configs/
kubectl apply -f ../../configs/kubernetes/karpenter/gpu-nodepool.yaml

# Install GPU Operator
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  -f ../../configs/kubernetes/gpu-operator/values.yaml

# Install full observability stack
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set grafana.adminPassword=admin
```

---

## Part 6: Deploy vLLM and trigger GPU node provisioning

> **GPU required:** This part provisions a real GPU instance (~$1–2/hr for g5.xlarge).

```bash
# Create namespace
kubectl create namespace inference

# Store your HuggingFace token
kubectl create secret generic hf-token \
  --namespace inference \
  --from-literal=token=$HF_TOKEN

# Deploy vLLM (triggers Karpenter to provision a g5.xlarge)
kubectl apply -f ../../configs/vllm/deployment.yaml
kubectl apply -f ../../configs/vllm/hpa.yaml

# Watch Karpenter provision the GPU node
kubectl get nodeclaims -w &
kubectl get pods -n inference -w

# Once running, test it
kubectl port-forward -n inference svc/vllm-inference 8000:8000 &
curl http://localhost:8000/health
curl http://localhost:8000/v1/models | jq
```

**Verify GPU metrics flowing from real hardware:**

```bash
kubectl get pods -n gpu-operator
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090 &
# Query: nvidia_dcgm_fi_dev_gpu_util — should show real values (not simulated)
```

---

## Cleanup

```bash
# Local kind cluster
kind delete cluster --name ai-infra-lab

# AWS (IMPORTANT — avoid surprise charges)
kubectl delete -f ../../configs/vllm/deployment.yaml   # removes GPU node via Karpenter
eksctl delete cluster --name $CLUSTER_NAME --region $AWS_REGION
```
