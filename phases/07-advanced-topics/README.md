# Phase 7: Advanced Topics

> The frontier. This phase has no end date — it evolves as the field evolves.

**Duration:** Ongoing
**Lab:** [07-llm-gateway](../../labs/07-llm-gateway/)

---

## Vector Databases and RAG Infrastructure

RAG (Retrieval-Augmented Generation) is the dominant production pattern for LLMs. The vector database is its core infrastructure component.

```
RAG Architecture
═════════════════

Indexing Pipeline (offline)         Query Pipeline (online)
────────────────────────────        ──────────────────────────────
Documents                           User query
    │                                   │
    ▼                                   ▼
Chunking                          Embed query
    │                             (same embedding model)
    ▼                                   │
Embedding model                         ▼
(e5-large, bge-m3, etc.)          Vector DB search
    │                             (ANN: HNSW or IVF)
    ▼                                   │
Vector DB (store vectors               ▼
+ original text)                  Top-K chunks retrieved
    │                                   │
    ▼                                   ▼
Done. Incremental updates          Inject into LLM prompt
as docs change.                         │
                                        ▼
                                   LLM generates response
                                   grounded in retrieved docs
```

### Vector DB comparison

```
Vector Database Decision Matrix
═══════════════════════════════════════════════════════════════

           │ Qdrant    │ Weaviate  │ Pinecone  │ pgvector
───────────┼───────────┼───────────┼───────────┼───────────
Self-host  │ ✓ easy    │ ✓ easy    │ ✗ SaaS    │ ✓ Postgres
SaaS       │ ✓         │ ✓         │ ✓ only    │ ✗
           │           │           │           │
Scale      │ Billions  │ Billions  │ Billions  │ ~10M rows
           │           │           │           │ (Postgres limit)
           │           │           │           │
Filtering  │ Strong    │ Strong    │ Moderate  │ SQL (native)
           │           │           │           │
Perf/cost  │ High/low  │ High/mod  │ High/high │ Good/free
           │           │           │           │
K8s native │ ✓ Helm    │ ✓ Helm    │ N/A       │ via CloudNativePG
           │           │           │           │
Best for   │ Most      │ Multi-    │ No infra  │ Already on
           │ use cases │ modal,    │ teams     │ Postgres
           │           │ GraphQL   │           │

Recommendation: Start with Qdrant. Migrate to pgvector
if you're already on Postgres and scale < 10M vectors.
```

### ANN Index Tradeoffs

```
Approximate Nearest Neighbor Indexes
══════════════════════════════════════

HNSW (Hierarchical Navigable Small World)
  Build time:  Slow (especially for large datasets)
  Query time:  Fast (logarithmic)
  Memory:      High (graph structure in RAM)
  Best for:    Real-time, latency-sensitive search

IVF (Inverted File Index)
  Build time:  Fast
  Query time:  Moderate (scans clusters)
  Memory:      Lower than HNSW
  Best for:    Large datasets where memory is constrained

Flat (brute force)
  Build time:  Instant
  Query time:  Slow (linear scan)
  Memory:      Lowest
  Best for:    < 100K vectors, testing, ground truth

Production default: HNSW with ef_construction=200, m=16
Tune ef (search parameter) for latency vs recall tradeoff.
```

---

## Distributed Training Infrastructure

You probably won't train foundation models. But you'll support teams that fine-tune, and fine-tuning at scale requires distributed training.

```
Distributed Training Strategies
═════════════════════════════════

Data Parallel (DDP)                 Model Parallel
──────────────────                  ────────────────────────
GPU 0: full model copy              GPU 0: layers 1–12
GPU 1: full model copy              GPU 1: layers 13–24
GPU 2: full model copy              GPU 2: layers 25–32

Each GPU gets different             Model too large for
data batch. Gradients               one GPU. Layers split
averaged across GPUs.               across GPUs (pipeline).

When to use: model fits             When to use: model doesn't
on one GPU.                         fit on one GPU.

Tensor Parallel (Megatron)          FSDP (Fully Sharded DP)
──────────────────────────          ─────────────────────────
GPU 0: first half of                GPU 0–N: model sharded
        each weight matrix          across all GPUs.
GPU 1: second half                  Each GPU owns fraction
                                    of parameters.
Inter-GPU comm every
forward pass (NVLink                Memory efficient.
critical here).                     Scales to very large models.

When to use: transformer            When to use: largest models,
layers specifically.                memory-constrained.
```

### EFA networking for multi-node training

```
Multi-Node Training Network Requirements
══════════════════════════════════════════

Single-node (NVLink):
  GPU-GPU: 900 GB/s bidirectional (NVLink)
  ✓ All-reduce fast, no bottleneck

Multi-node (EFA required):
  Node A GPUs ←─── EFA ────► Node B GPUs
              100 Gb/s (EFA)

Without EFA: 10–25 Gb/s Ethernet = training slowdown 5–10x
With EFA:    100 Gb/s+ = near-linear scaling

EFA instance requirement:
  p3dn.24xlarge, p4d.24xlarge, p5.48xlarge
  (EFA not available on g4/g5 families)

AWS Placement Group (cluster):
  Ensures instances are physically co-located.
  Required for EFA workloads.
```

---

## Inference Optimization Internals

Understanding these lets you evaluate serving framework benchmarks critically rather than trusting marketing numbers.

```
Flash Attention
════════════════

Standard Attention (slow):          Flash Attention:
──────────────────────────          ─────────────────────────
1. Load Q, K matrices to VRAM       1. Tile Q, K, V into blocks
2. Compute Q×K^T (huge matrix)      2. Load ONE tile at a time
3. Store result in VRAM             3. Compute attention on tile
4. Load again for softmax           4. Accumulate result
5. Load again for ×V                (No large intermediate matrix
                                    ever in VRAM)

Memory: O(n²) with n=seq_len        Memory: O(n) — massive saving
VRAM: bottleneck at long seqs       VRAM: enables long contexts
```

```
Speculative Decoding
═════════════════════

Problem: large model generates one token per forward pass.
         Slow for long outputs.

Solution: use a small "draft" model to generate k tokens cheaply,
          then verify all k tokens with the large model in ONE pass.
          Large model's parallelism does the work.

Draft model generates:   [token1, token2, token3, token4, token5]
Large model verifies:    [✓, ✓, ✓, ✗]  (accepts 3, rejects at 4)

Net effect: 2–3x throughput improvement for long outputs
            with identical quality to large model alone.

Infrastructure implication:
  You need BOTH models loaded in VRAM simultaneously.
  Draft model: small (1B–3B params)
  Target model: your main model
```

---

## AI Agent Infrastructure

Agents are LLMs that can use tools, maintain state, and run multi-step workflows. The infrastructure complexity is significantly higher than single-turn inference.

```
Agent Infrastructure Stack
═══════════════════════════

User Request
     │
     ▼
┌─────────────────────────────────────────────────────┐
│              Agent Orchestrator                     │
│  (LangGraph, AutoGen, custom)                       │
│                                                     │
│  Loop:                                              │
│  1. LLM generates next action (tool call or answer) │
│  2. Execute tool if action is a tool call           │
│  3. Add result to context                           │
│  4. Repeat until done                               │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   ┌─────────┐  ┌─────────┐  ┌─────────┐
   │  Tools  │  │  State  │  │   LLM   │
   │         │  │         │  │         │
   │ Web     │  │ Redis   │  │ vLLM /  │
   │ Search  │  │ (short  │  │ Bedrock │
   │ Code    │  │  term)  │  │         │
   │ Execute │  │ Postgres│  └─────────┘
   │ DB Query│  │ (long   │
   │ APIs    │  │  term)  │
   └─────────┘  └─────────┘

Infrastructure concerns:
  - Long-running requests (minutes, not ms) → async pattern required
  - State management across turns → Redis/Postgres
  - Tool execution isolation → sandboxed containers
  - Token context growing with each tool call → context management
  - Cost is unpredictable → hard per-request token budget needed
```

---

## Multi-Tenant LLM Platforms

This is where your SRE background pays off hardest. ML engineers are bad at this. You're not.

```
Internal LLM Platform Architecture
════════════════════════════════════

  External Clients
  (internal teams, apps)
         │
         ▼ HTTPS + JWT
  ┌─────────────────────────────────────────────────┐
  │              LLM Gateway                        │
  │                                                 │
  │  Auth & AuthZ          Rate Limiting            │
  │  ─────────────         ─────────────────────    │
  │  JWT validation        Token-bucket per tenant  │
  │  API key → tenant      Hard limit: 100K tok/min │
  │  RBAC (model access)   Soft limit: alert at 80% │
  │                                                 │
  │  Model Routing         Cost Attribution         │
  │  ─────────────         ─────────────────────    │
  │  tier: premium → 70B  Track tokens per:         │
  │  tier: standard → 8B  - tenant                 │
  │  cost_sensitive → 3B   - user                  │
  │  Route by: task type,  - model                 │
  │  user tier, load       - endpoint              │
  └────────────┬────────────────────────────────────┘
               │
    ┌──────────┼───────────┐
    ▼          ▼           ▼
  ┌────┐    ┌────┐      ┌────┐
  │ 8B │    │ 70B│      │ 3B │
  │vLLM│    │vLLM│      │vLLM│
  └────┘    └────┘      └────┘

  Built with: Envoy / Kong / custom Go service
```

---

## Resources

See [resources.md](./resources.md) for the full curated list.

**Essential:**
1. Qdrant docs — Architecture section specifically
2. Anthropic MCP Specification — the emerging standard for agent tool use
3. AI Infrastructure Alliance — community, stay current

---

## Lab

**[→ Lab 07: LLM Gateway](../../labs/07-llm-gateway/)**

Build an internal LLM gateway: JWT auth, per-tenant rate limiting (token-bucket over Redis), model routing by tier, cost attribution per team, full Prometheus metrics. This is the capstone. It combines everything from all previous phases into one production-grade service. It's also the exact thing companies are hiring for right now.
