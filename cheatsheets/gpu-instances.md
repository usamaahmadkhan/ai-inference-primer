# GPU Instances Cheatsheet

> Quick reference for AWS GPU instance selection. Bookmark this.

---

## Instance Selection by Model Size

```
Model Size → Minimum Instance
══════════════════════════════════════════════════════════════

Model   VRAM (FP16)  Min Instance        VRAM Available  Notes
──────────────────────────────────────────────────────────────
3B      ~6 GB        g4dn.xlarge (16GB)  16 GB           Comfortable
7B      ~14 GB       g4dn.xlarge (16GB)  16 GB           Tight, use INT8
8B      ~16 GB       g5.xlarge (24GB)    24 GB           Comfortable
13B     ~26 GB       g5.2xlarge (24GB)   24 GB           Needs INT8/INT4
30B     ~60 GB       p4d.24xlarge        40GB ×8         Tensor parallel
70B     ~140 GB      p4de.24xlarge       80GB ×8         Tensor parallel
405B    ~810 GB      p5.48xlarge ×2      80GB ×16        Multi-node
```

---

## Full Instance Reference

| Instance | GPU | Count | VRAM | vCPU | RAM | Network | On-Demand $/hr |
|---|---|---|---|---|---|---|---|
| g4dn.xlarge | T4 | 1 | 16 GB | 4 | 16 GB | Up to 25 Gbps | ~$0.53 |
| g4dn.2xlarge | T4 | 1 | 16 GB | 8 | 32 GB | Up to 25 Gbps | ~$0.75 |
| g4dn.4xlarge | T4 | 1 | 16 GB | 16 | 64 GB | Up to 25 Gbps | ~$1.20 |
| g4dn.8xlarge | T4 | 1 | 16 GB | 32 | 128 GB | 50 Gbps | ~$2.26 |
| g4dn.12xlarge | T4 | 4 | 64 GB | 48 | 192 GB | 50 Gbps | ~$3.91 |
| g4dn.16xlarge | T4 | 1 | 16 GB | 64 | 256 GB | 50 Gbps | ~$4.52 |
| g5.xlarge | A10G | 1 | 24 GB | 4 | 16 GB | Up to 10 Gbps | ~$1.01 |
| g5.2xlarge | A10G | 1 | 24 GB | 8 | 32 GB | Up to 10 Gbps | ~$1.21 |
| g5.4xlarge | A10G | 1 | 24 GB | 16 | 64 GB | Up to 25 Gbps | ~$1.62 |
| g5.8xlarge | A10G | 1 | 24 GB | 32 | 128 GB | 25 Gbps | ~$2.42 |
| g5.12xlarge | A10G | 4 | 96 GB | 48 | 192 GB | 40 Gbps | ~$5.67 |
| g5.48xlarge | A10G | 8 | 192 GB | 192 | 768 GB | 100 Gbps | ~$16.29 |
| g6.xlarge | L4 | 1 | 24 GB | 4 | 16 GB | Up to 25 Gbps | ~$0.80 |
| g6.2xlarge | L4 | 1 | 24 GB | 8 | 32 GB | Up to 25 Gbps | ~$0.97 |
| g6.12xlarge | L4 | 4 | 96 GB | 48 | 192 GB | 40 Gbps | ~$4.60 |
| p3.2xlarge | V100 | 1 | 16 GB | 8 | 61 GB | Up to 10 Gbps | ~$3.06 |
| p3.8xlarge | V100 | 4 | 64 GB | 32 | 244 GB | 10 Gbps | ~$12.24 |
| p3.16xlarge | V100 | 8 | 128 GB | 64 | 488 GB | 25 Gbps | ~$24.48 |
| p4d.24xlarge | A100 40GB | 8 | 320 GB | 96 | 1152 GB | 400 Gbps | ~$32.77 |
| p4de.24xlarge | A100 80GB | 8 | 640 GB | 96 | 1152 GB | 400 Gbps | ~$40.97 |
| p5.48xlarge | H100 80GB | 8 | 640 GB | 192 | 2048 GB | 3200 Gbps | ~$98.32 |

*Prices approximate us-east-1 on-demand. Check AWS pricing page for current.*

---

## Spot Discount and Availability

| Family | Typical Spot Discount | Interruption Rate | Strategy |
|---|---|---|---|
| g4dn | 60–70% off | Low | Good for spot baseline |
| g5 | 50–65% off | Medium | Mix spot/on-demand |
| g6 | 50–60% off | Medium | Good newer alternative |
| p4d | 40–55% off | High | On-demand baseline + spot burst |
| p5 | 20–40% off | Very High | Mostly on-demand |

**Rule:** Use on-demand for your minimum viable capacity, spot for burst. Set `WhenEmpty` consolidation in Karpenter so spot nodes aren't evicted mid-request.

---

## GPU Hardware Generations

```
NVIDIA GPU Generations (relevant to AWS)
═════════════════════════════════════════

V100 (Volta, 2017)      → p3 instances
  FP16: 125 TFLOPS | FP32: 14 TFLOPS | VRAM: 16/32 GB
  Legacy. Avoid for new workloads.

T4 (Turing, 2018)       → g4dn instances
  FP16: 65 TFLOPS | INT8: 130 TOPS | VRAM: 16 GB
  Best cost for small inference workloads.

A10G (Ampere, 2021)     → g5 instances
  FP16: 125 TFLOPS | VRAM: 24 GB
  Current sweet spot for 7B–13B serving.

L4 (Ada Lovelace, 2023) → g6 instances
  FP16: 121 TFLOPS | VRAM: 24 GB
  Newer, ~20% cheaper than A10G, similar perf.

A100 (Ampere, 2020)     → p4d/p4de instances
  FP16: 312 TFLOPS | BF16: 312 TFLOPS | VRAM: 40/80 GB
  Gold standard for 30B–70B models.

H100 (Hopper, 2022)     → p5 instances
  FP16: 989 TFLOPS | VRAM: 80 GB | NVLink 900GB/s
  Frontier models. Hard to get. Expensive.
```

---

## Karpenter Instance Selectors

```yaml
# Common instance type lists for Karpenter NodePools

# Small inference (7B models)
values: [g5.xlarge, g5.2xlarge, g6.xlarge, g6.2xlarge]

# Medium inference (13B–30B models)
values: [g5.12xlarge, g5.48xlarge, g6.12xlarge]

# Large inference (70B models)
values: [p4d.24xlarge, p4de.24xlarge]

# Dev/test (cheapest GPU)
values: [g4dn.xlarge, g4dn.2xlarge, g6.xlarge]

# Maximum performance
values: [p5.48xlarge]
```
