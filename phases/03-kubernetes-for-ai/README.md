# Phase 3: Kubernetes for AI

> GPU workloads on Kubernetes are materially different from CPU workloads. This phase covers what changes.

**Duration:** 4–5 weeks
**Lab:** [03-eks-gpu-cluster](../../labs/03-eks-gpu-cluster/)

---

## What changes when you add GPUs

```
Standard K8s Workload              GPU Workload
══════════════════════             ══════════════════════════

Resources:                         Resources:
  cpu: "4"                           cpu: "8"
  memory: "8Gi"                      memory: "32Gi"
                                     nvidia.com/gpu: "1"  ← device plugin
                                                              required

Scheduling:                        Scheduling:
  Default scheduler                  GPU Operator installs
  works fine                         device plugin that
                                     makes GPUs schedulable

Node provisioning:                 Node provisioning:
  Karpenter spins up                 Karpenter must select
  right-sized instance               GPU instance family
  in seconds                         (minutes to start,
                                     capacity often scarce)

Failures:                          Failures:
  Pod restart fixes                  CUDA OOM crashes pod
  most issues                        GPU process isolation
                                     incomplete — one bad
                                     tenant can affect others

Observability:                     Observability:
  node_exporter covers               DCGM Exporter required
  everything                         for GPU metrics
```

---

## NVIDIA GPU Operator

The GPU Operator is the foundation of GPU on Kubernetes. It automates the entire driver and plugin lifecycle that you'd otherwise manage manually.

```
GPU Operator Component Stack
═════════════════════════════

  ┌─────────────────────────────────────────────┐
  │           Your Application Pod              │
  │   (requests nvidia.com/gpu: 1)              │
  └────────────────────┬────────────────────────┘
                       │
  ┌────────────────────▼────────────────────────┐
  │         NVIDIA Device Plugin                │
  │  Exposes GPU resources to K8s scheduler     │
  │  Reports GPU capacity to API server         │
  └────────────────────┬────────────────────────┘
                       │
  ┌────────────────────▼────────────────────────┐
  │         NVIDIA Container Toolkit            │
  │  Mounts GPU into container namespace        │
  │  Configures container runtime (containerd)  │
  └────────────────────┬────────────────────────┘
                       │
  ┌────────────────────▼────────────────────────┐
  │           NVIDIA Drivers (kernel)           │
  │  GPU Operator installs and manages these    │
  │  on every GPU node automatically            │
  └────────────────────┬────────────────────────┘
                       │
  ┌────────────────────▼────────────────────────┐
  │             Physical GPU                    │
  └─────────────────────────────────────────────┘

  All layers above the physical GPU are managed
  by the GPU Operator DaemonSet. Don't install
  drivers manually on K8s GPU nodes.
```

### Install via Helm

```bash
# Add NVIDIA Helm repo
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# Install GPU Operator (see configs/kubernetes/gpu-operator/values.yaml
# for EKS-tuned values)
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  -f configs/kubernetes/gpu-operator/values.yaml
```

### Verify it's working

```bash
# Check all GPU Operator pods are running
kubectl get pods -n gpu-operator

# Verify GPU is schedulable on a node
kubectl describe node <gpu-node> | grep nvidia.com/gpu

# Run a test pod
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  restartPolicy: Never
  containers:
  - name: cuda-vector-add
    image: nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda10.2
    resources:
      limits:
        nvidia.com/gpu: 1
EOF

kubectl logs gpu-test
# Should print: Test PASSED
```

---

## GPU Scheduling Patterns

### Basic resource request

```yaml
# This is the minimum to schedule on a GPU node
resources:
  limits:
    nvidia.com/gpu: 1    # Whole GPU allocation
    memory: "24Gi"
    cpu: "8"
  requests:
    nvidia.com/gpu: 1    # Must match limits for GPU
    memory: "20Gi"
    cpu: "4"

# Always set nvidia.com/gpu in BOTH limits and requests.
# K8s does not allow fractional GPU requests natively
# (use time-slicing or MIG for sharing — see below).
```

### Node affinity for GPU type

```yaml
# Target specific GPU hardware
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: node.kubernetes.io/instance-type
          operator: In
          values:
          - p4d.24xlarge    # A100 40GB
          - p4de.24xlarge   # A100 80GB
        - key: nvidia.com/gpu.product
          operator: In
          values:
          - A100-SXM4-80GB

tolerations:
- key: nvidia.com/gpu
  operator: Exists
  effect: NoSchedule
```

---

## MIG: Multi-Instance GPU

A100 and H100 GPUs support MIG (Multi-Instance GPU) — hardware-level partitioning into isolated GPU slices. Use this for multi-tenant inference where you want true isolation, not just resource limits.

```
A100 80GB MIG Profiles
════════════════════════

Full GPU                    1g.10gb (smallest)
┌──────────────────┐        ┌──┐
│                  │        │  │  ← 1/7th of A100
│                  │   or   ├──┤  7 instances per GPU
│   80 GB VRAM     │        │  │
│   Full compute   │        ├──┤
│                  │        │  │
│                  │        ├──┤
│                  │        │  │
└──────────────────┘        └──┘

Common MIG profiles on A100 80GB:
┌─────────────────┬──────────────┬──────────────────────────┐
│ Profile         │ VRAM per inst│ Max instances            │
├─────────────────┼──────────────┼──────────────────────────┤
│ 1g.10gb         │ 10 GB        │ 7 (smallest, for <7B)    │
│ 2g.20gb         │ 20 GB        │ 3                        │
│ 3g.40gb         │ 40 GB        │ 2 (for 13B–30B)          │
│ 4g.40gb         │ 40 GB        │ 1                        │
│ 7g.80gb         │ 80 GB        │ 1 (full GPU, no MIG)     │
└─────────────────┴──────────────┴──────────────────────────┘
```

```yaml
# Request a MIG slice instead of a full GPU
resources:
  limits:
    nvidia.com/mig-3g.40gb: 1  # 40GB slice of A100
```

**When to use MIG:**
- Multiple smaller models on one expensive GPU
- Strict tenant isolation (each tenant gets a hardware slice)
- Dev/test environments sharing production hardware

**When NOT to use MIG:**
- Single large model that needs the full GPU
- Models requiring tensor parallelism across the full GPU

---

## GPU Time-Slicing

Lighter alternative to MIG. Multiple pods share a GPU through time-slicing. No memory isolation — pods can still OOM each other.

```yaml
# GPU Operator ConfigMap to enable time-slicing
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: gpu-operator
data:
  any: |-
    version: v1
    flags:
      migStrategy: none
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 4    # 4 pods can share 1 GPU
```

Use time-slicing for: development environments, low-traffic inference, batch jobs that don't need dedicated GPU time.

---

## Karpenter with GPU Nodes

GPU instances need specific configuration in Karpenter. The default NodePool won't select GPU instances.

```yaml
# NodePool that provisions GPU instances
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu-inference
spec:
  template:
    metadata:
      labels:
        workload-type: gpu-inference
    spec:
      nodeClassRef:
        apiVersion: karpenter.k8s.aws/v1
        kind: EC2NodeClass
        name: gpu-inference
      requirements:
      - key: karpenter.sh/capacity-type
        operator: In
        values: ["on-demand", "spot"]     # GPU spot = 60–70% cheaper, ~5% interruption
      - key: node.kubernetes.io/instance-type
        operator: In
        values:
        - g5.xlarge      # 1x A10G, 24GB  — small models
        - g5.2xlarge     # 1x A10G, 24GB  — higher CPU
        - g5.12xlarge    # 4x A10G, 96GB  — tensor parallel
        - p4d.24xlarge   # 8x A100, 320GB — large models
      taints:
      - key: nvidia.com/gpu
        effect: NoSchedule   # Only GPU-aware pods schedule here
  limits:
    nvidia.com/gpu: 20       # Hard cap on GPU consumption
  disruption:
    consolidationPolicy: WhenEmpty   # Don't evict running GPU workloads
    expireAfter: 720h
```

### GPU spot instance strategy

```
Spot interruption handling for GPU inference
════════════════════════════════════════════

Problem: GPU spot can be interrupted with 2-min notice.
         In-flight requests will fail.

Mitigation stack:

1. Multiple replicas + PodDisruptionBudget
   └── Always 1 replica running during node drain

2. vLLM graceful shutdown
   └── SIGTERM handler: drain queue, finish in-flight requests

3. Request retries at the gateway layer
   └── Retry on 503 with exponential backoff

4. Karpenter consolidation: WhenEmpty
   └── Karpenter only removes GPU nodes when fully idle

5. Reserved capacity for baseline
   └── Keep 1 on-demand replica, scale with spot
```

---

## KubeRay: Ray on Kubernetes

For distributed inference and workloads that need Ray's actor model.

```
KubeRay Architecture
═════════════════════

┌─────────────────────────────────────────────┐
│              RayCluster CR                  │
│                                             │
│  ┌────────────┐    ┌─────────────────────┐  │
│  │  Head Node │    │   Worker Nodes      │  │
│  │            │    │  ┌───┐ ┌───┐ ┌───┐ │  │
│  │ Ray GCS    │◄──►│  │GPU│ │GPU│ │GPU│ │  │
│  │ Dashboard  │    │  │ 0 │ │ 1 │ │ 2 │ │  │
│  │ Autoscaler │    │  └───┘ └───┘ └───┘ │  │
│  └────────────┘    └─────────────────────┘  │
│                                             │
│  Managed by KubeRay Operator                │
└─────────────────────────────────────────────┘

Use RayService for:
  - vLLM with tensor parallelism across pods
  - Multiple models in one Ray cluster
  - Dynamic model loading/unloading

Use RayJob for:
  - Batch inference jobs
  - Fine-tuning runs
  - One-shot distributed compute
```

---

## DCGM Exporter

DCGM (Data Center GPU Manager) exposes GPU metrics to Prometheus. This is your `node_exporter` for GPUs.

```
DCGM Exporter → Prometheus → Grafana
══════════════════════════════════════

GPU Hardware
    │
    ├── GPU Utilization (%)          ← nvidia_dcgm_fi_dev_gpu_util
    ├── VRAM Used (bytes)            ← nvidia_dcgm_fi_dev_fb_used
    ├── VRAM Free (bytes)            ← nvidia_dcgm_fi_dev_fb_free
    ├── Power Draw (watts)           ← nvidia_dcgm_fi_dev_power_usage
    ├── Temperature (°C)             ← nvidia_dcgm_fi_dev_gpu_temp
    ├── SM Clock (MHz)               ← nvidia_dcgm_fi_dev_sm_clock
    └── NVLink Bandwidth (bytes/s)   ← nvidia_dcgm_fi_dev_nvlink_bandwidth_total

DCGM DaemonSet (runs on every GPU node)
    │
    │  scrapes hardware counters
    │
    ▼
Prometheus metrics endpoint (:9400/metrics)
    │
    │  scraped by Prometheus (via ServiceMonitor)
    │
    ▼
Grafana Dashboard
```

### Key alerts to set

```yaml
# These are the alerts that matter for GPU inference

groups:
- name: gpu-inference
  rules:

  # VRAM pressure — model may OOM soon
  - alert: GPUMemoryPressure
    expr: |
      (nvidia_dcgm_fi_dev_fb_used / 
       (nvidia_dcgm_fi_dev_fb_used + nvidia_dcgm_fi_dev_fb_free)) > 0.90
    for: 5m
    annotations:
      summary: "GPU VRAM > 90% on {{ $labels.instance }}"

  # GPU underutilized — wasted capacity (or workload stuck)
  - alert: GPUUnderutilized
    expr: nvidia_dcgm_fi_dev_gpu_util < 20
    for: 15m
    annotations:
      summary: "GPU utilization < 20% — check for stalled workload"

  # Temperature critical
  - alert: GPUTemperatureCritical
    expr: nvidia_dcgm_fi_dev_gpu_temp > 85
    for: 2m
    annotations:
      summary: "GPU temperature > 85°C — thermal throttling likely"
```

---

## Resources

See [resources.md](./resources.md) for the full curated list.

**Essential:**
1. NVIDIA GPU Operator docs — especially the EKS-specific installation guide
2. AWS EKS GPU Workshop — hands-on, follows the actual production path
3. KubeRay docs — start with RayService, that's the inference pattern

---

## Lab

**[→ Lab 03: EKS GPU Cluster](../../labs/03-eks-gpu-cluster/)**

Build an EKS cluster with Karpenter GPU node pools (on-demand + spot). Deploy GPU Operator. Get DCGM metrics into Prometheus. Deploy vLLM as a K8s Deployment with GPU resource requests. Set up HPA on request queue depth using a custom metric from vLLM. This is the production baseline everything else builds on.
