# kubectl for GPU Workloads

> Commands you'll run constantly when operating GPU infrastructure on Kubernetes.

---

## Node Inspection

```bash
# List all GPU nodes and their GPU count
kubectl get nodes -l nvidia.com/gpu.present=true \
  -o custom-columns='NAME:.metadata.name,TYPE:.metadata.labels.node\.kubernetes\.io/instance-type,GPU:.status.allocatable.nvidia\.com/gpu'

# Describe a GPU node — check capacity, allocatable, conditions
kubectl describe node <node-name> | grep -A5 "Capacity:\|Allocatable:\|nvidia"

# Check GPU resources across all nodes
kubectl get nodes -o json | jq '.items[] | {
  name: .metadata.name,
  gpu_capacity: .status.capacity["nvidia.com/gpu"],
  gpu_allocatable: .status.allocatable["nvidia.com/gpu"]
}'

# Which pods are consuming GPUs on a specific node
kubectl get pods --all-namespaces --field-selector spec.nodeName=<node-name> \
  -o custom-columns='NAMESPACE:.metadata.namespace,POD:.metadata.name,GPU:.spec.containers[*].resources.limits.nvidia\.com/gpu'
```

---

## GPU Utilization (without SSH)

```bash
# Run nvidia-smi on a specific GPU node (no SSH needed)
kubectl run -it --rm gpu-debug \
  --image=nvidia/cuda:12.2.0-base-ubuntu22.04 \
  --overrides='{"spec": {"nodeName": "<node-name>", "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists"}]}}' \
  --restart=Never \
  -- nvidia-smi

# nvidia-smi with continuous refresh (watch mode)
kubectl run -it --rm gpu-debug \
  --image=nvidia/cuda:12.2.0-base-ubuntu22.04 \
  --overrides='{"spec": {"nodeName": "<node-name>", "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists"}]}}' \
  --restart=Never \
  -- nvidia-smi dmon -s u  # streaming GPU utilization

# Check GPU memory specifically
# nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```

---

## GPU Operator

```bash
# Check GPU Operator status
kubectl get pods -n gpu-operator

# Expected pods (varies by config):
# gpu-operator-*                        (controller)
# nvidia-driver-daemonset-*             (driver install, one per node)
# nvidia-container-toolkit-daemonset-*  (container runtime config)
# nvidia-device-plugin-daemonset-*      (exposes nvidia.com/gpu resource)
# nvidia-dcgm-exporter-*               (Prometheus metrics)
# nvidia-mig-manager-*                 (if MIG enabled)

# Tail GPU Operator controller logs
kubectl logs -n gpu-operator -l app=gpu-operator -f

# Check driver install status on a specific node
kubectl logs -n gpu-operator -l app=nvidia-driver-daemonset \
  --field-selector spec.nodeName=<node-name>

# Force GPU Operator to re-validate a node
kubectl annotate node <node-name> \
  nvidia.com/gpu.deploy.driver=true --overwrite
```

---

## vLLM Debugging

```bash
# Get vLLM pod logs (follow)
kubectl logs -f deployment/vllm-inference -c vllm

# Check vLLM metrics endpoint
kubectl port-forward svc/vllm-inference 8000:8000
curl http://localhost:8000/metrics | grep -E "vllm:|nvida_"

# Check vLLM health
curl http://localhost:8000/health

# List models available
curl http://localhost:8000/v1/models | jq

# Quick inference test
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 50
  }'

# Check vLLM stats (queue, running, waiting)
curl http://localhost:8000/metrics | grep -E "num_requests|queue_length|cache_usage"
```

---

## Karpenter GPU Nodes

```bash
# List all Karpenter-managed nodes
kubectl get nodes -l karpenter.sh/nodepool -o wide

# Check which NodePool a node belongs to
kubectl get node <node-name> -o jsonpath='{.metadata.labels.karpenter\.sh/nodepool}'

# Check NodePool status (GPU limits, usage)
kubectl get nodepool gpu-inference -o yaml | grep -A10 "status:"

# Force Karpenter to evict and replace a GPU node
kubectl annotate node <node-name> karpenter.sh/do-not-disrupt-

# List NodeClaims (Karpenter's node requests)
kubectl get nodeclaims

# Manually trigger consolidation check
kubectl annotate nodepool gpu-inference karpenter.sh/do-not-disrupt-
```

---

## DCGM Exporter

```bash
# Check DCGM exporter is running on GPU nodes
kubectl get pods -n gpu-operator -l app=nvidia-dcgm-exporter

# Port-forward and check metrics
DCGM_POD=$(kubectl get pods -n gpu-operator -l app=nvidia-dcgm-exporter \
  -o jsonpath='{.items[0].metadata.name}')
kubectl port-forward -n gpu-operator $DCGM_POD 9400:9400
curl http://localhost:9400/metrics | grep nvidia_dcgm_fi_dev_gpu_util

# Check DCGM exporter logs
kubectl logs -n gpu-operator -l app=nvidia-dcgm-exporter -f
```

---

## Useful Label Selectors

```bash
# All GPU nodes (device plugin label)
-l nvidia.com/gpu.present=true

# Nodes with specific GPU type
-l nvidia.com/gpu.product=A10G

# Karpenter GPU node pools
-l karpenter.sh/nodepool=gpu-inference

# vLLM pods
-l app=vllm-inference

# GPU Operator components
-n gpu-operator
```

---

## Troubleshooting

```bash
# Pod stuck in Pending — check if GPU available
kubectl describe pod <pod-name> | grep -A5 "Events:"
# Look for: "Insufficient nvidia.com/gpu"

# Check GPU resource pressure across cluster
kubectl describe nodes | grep -A3 "nvidia.com/gpu"

# Pod crashes with CUDA OOM — check VRAM config
kubectl logs <pod-name> --previous | grep -i "cuda\|oom\|memory"

# Check if GPU Operator installed drivers correctly
kubectl get pods -n gpu-operator -l app=nvidia-driver-daemonset
# Status should be Running, not CrashLoopBackOff

# Reset a stuck GPU node (last resort)
kubectl cordon <node-name>
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data
kubectl delete node <node-name>
# Karpenter will provision a replacement

# Check MIG partitioning (if enabled)
kubectl run -it --rm mig-check \
  --image=nvidia/cuda:12.2.0-base-ubuntu22.04 \
  --overrides='{"spec": {"nodeName": "<node-name>"}}' \
  --restart=Never -- nvidia-smi mig -lgi
```
