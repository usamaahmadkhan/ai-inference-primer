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

## Designing and Evaluating Agents

The diagram above shows the infrastructure shape. It doesn't tell you when an agent is the right architecture, how to design the tools that orchestrator calls, or how to know whether the thing you built actually works. Three Anthropic engineering posts cover exactly that gap, and they're written for builders, not researchers — worth reading in full, not just skimming the summary below.

### When to build an agent at all (and the 5 patterns when you do)

**[Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)** (Schluntz & Zhang) draws the architectural line that the "Agent Orchestrator" box in the diagram above glosses over: **workflows** (LLMs and tools orchestrated through code paths you write) versus **agents** (the LLM dynamically directs its own tool use and control flow). Workflows give you predictability and debuggability. Agents give you flexibility at the cost of latency, cost, and compounding error risk. The core recommendation is to find the simplest pattern that works and only add agentic autonomy when the task genuinely can't be hardcoded into a fixed path.

```
The five workflow patterns (use before reaching for a full agent)
═══════════════════════════════════════════════════════════════════

Prompt Chaining          Routing                Parallelization
──────────────────       ──────────────────     ──────────────────
Step 1 → Step 2 → ...    Classify input →       Run N calls at once
Each step's output       send to specialized    → aggregate results
feeds the next            prompt/model path     (voting, sectioning)

Orchestrator-Workers     Evaluator-Optimizer
──────────────────       ──────────────────
One LLM breaks down      Generator produces,
task, delegates to       evaluator critiques,
workers dynamically      loop until criteria met

  Use these for:                    Reach for a full autonomous
  predictable subtask structure,     agent only when:
  bounded complexity,                the path can't be predicted
  need for debuggability             in advance, and the cost of
                                    autonomy is justified by the
                                    task's value
```

The post's Appendix 2 ("Prompt Engineering your Tools") is the seed that the next article expands into a full standalone piece.

### Designing the tools your agent calls

**[Writing effective tools for AI agents — using AI agents](https://www.anthropic.com/engineering/writing-tools-for-agents)** treats tool design as a distinct engineering discipline, not an afterthought to API wrapping. The framing that matters most: a tool definition is a contract between a deterministic system and a non-deterministic caller, and every word in the name, description, and parameter docs is effectively a prompt that shapes how reliably the agent calls it correctly.

```
Tool design checklist (from the article)
══════════════════════════════════════════════════════════

1. High leverage           Don't wrap every API 1:1. Build tools that
                           collapse multi-step API choreography into
                           one call the agent can reason about simply.

2. Clear namespacing       Distinct, unambiguous names. An agent
                           choosing between similarly-named tools is
                           choosing somewhat randomly.

3. Meaningful responses    Return human-readable fields, not raw IDs.
                           The agent reasons over what you give back —
                           give it something reasoning-friendly.

4. Token efficiency        Paginate, truncate, filter by default.
                           Claude Code caps tool responses at 25K
                           tokens for exactly this reason.

5. Prompt-engineer specs   Treat the tool description like you'd brief
                           a new hire: state the implicit context
                           (formats, terminology, relationships)
                           explicitly. Iterate on it like a prompt,
                           because it is one.
```

The "using AI agents" half of the title is the other half of the point: the article's recommended loop is Prototype → Evaluate → Collaborate, where Claude itself is used to read transcripts of its own tool-calling failures and suggest description fixes — a genuinely useful technique once you have an eval harness in place to measure whether the fix helped.

### Evaluating what you built

This is where Phase 5's evaluation framework gets agent-specific teeth. **[Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)** explains precisely why the "Infrastructure concerns" list above includes unpredictable cost and growing context: agents act over many turns, modify state, and can find valid solutions a rigid grader didn't anticipate (the article's example: Claude Opus 4.5 "failed" a flight-booking benchmark by finding a legitimate policy loophole that produced a better outcome for the user than the intended answer).

The practical takeaway for infrastructure engineers specifically: build your eval harness to verify **outcome** (the actual end-state in the environment — did the database row get written, not just did the transcript say it did) separately from **process** (did it use a reasonable number of tool calls, stay within turn limits, follow expected patterns). Grading the path an agent took too strictly produces brittle evals that punish creativity; grading only the outcome risks missing process problems like runaway tool-calling loops that blow your cost budget. Most production agent evals combine code-based graders (state checks, tool-call verification) with model-based graders (rubric scoring) for exactly this reason — see [Phase 5: Evaluation Frameworks](../05-llmops/#evaluation-frameworks) for the grader taxonomy this builds on.

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

See [resources.md](./resources.md) for the full curated list with descriptions.

**Essential:**
1. [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) — Anthropic. Read before writing any agent orchestration code. The workflows-vs-agents distinction will save you from over-building.
2. [Writing effective tools for AI agents — using AI agents](https://www.anthropic.com/engineering/writing-tools-for-agents) — Anthropic. Tool descriptions are prompts; treat them that way.
3. [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) — Anthropic. How to know if the agent you built actually works.
4. Qdrant docs — Architecture section specifically
5. Anthropic MCP Specification — the emerging standard for agent tool use
6. AI Infrastructure Alliance — community, stay current

---

## Lab

**[→ Lab 07: LLM Gateway](../../labs/07-llm-gateway/)**

Build an internal LLM gateway: JWT auth, per-tenant rate limiting (token-bucket over Redis), model routing by tier, cost attribution per team, full Prometheus metrics. This is the capstone. It combines everything from all previous phases into one production-grade service. It's also the exact thing companies are hiring for right now.
