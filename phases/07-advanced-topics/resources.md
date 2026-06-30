# Phase 7 Resources: Advanced Topics

Curated, in the order we'd actually read/use them.

---

## Building and evaluating agents

**[Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)** — Anthropic Engineering, Dec 2024 (Erik Schluntz & Barry Zhang)
The foundational read for anyone building agent orchestration. Draws the architectural distinction between workflows (LLM + tools through code paths you write) and agents (LLM dynamically directs its own tool use). Covers five workflow patterns — prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer — and when each beats reaching for a full autonomous agent. Read this before writing a single line of orchestration code; it'll change which pattern you reach for first.

**[Writing effective tools for AI agents — using AI agents](https://www.anthropic.com/engineering/writing-tools-for-agents)** — Anthropic Engineering, Sept 2025
Tool design as a discipline distinct from API wrapping. Five concrete principles (high-leverage tools, clear namespacing, meaningful human-readable responses, token-efficient defaults, prompt-engineered descriptions) plus a Prototype → Evaluate → Collaborate loop that uses Claude itself to critique and improve its own tool descriptions. Directly extends the "Tools" box in this phase's Agent Infrastructure diagram.

**[Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)** — Anthropic Engineering, Jan 2026
The field guide for evaluating agents specifically, as distinct from single-turn LLM evals. Defines task/trial/grader/transcript/outcome vocabulary, the three grader types and when to combine them, capability vs regression evals, and `pass@k` vs `pass^k` for handling agent non-determinism. The appendix lists open-source eval frameworks (Harbor, Braintrust, LangSmith, Langfuse, Arize Phoenix) if you need infrastructure beyond a custom harness. Cross-references [Phase 5: Evaluation Frameworks](../05-llmops/resources.md) — read that first if you haven't built any eval gate yet.

---

## RAG and vector infrastructure

**[Qdrant docs — Architecture concepts](https://qdrant.tech/documentation/concepts/)**
Read the Architecture section specifically before deploying. Covers collections, segments, and the HNSW index internals that determine your latency/recall tradeoff at scale.

**[LlamaIndex production RAG guide](https://docs.llamaindex.ai/en/stable/)**
The most complete framework-level documentation for production retrieval pipelines: chunking strategies, retrieval evaluation, and the indexing patterns that actually hold up past a proof-of-concept.

---

## Agent protocols and community

**[Anthropic MCP Specification](https://modelcontextprotocol.io)**
The emerging standard for how agents discover and call tools across systems. If you're building any multi-tool agent infrastructure today, build against this rather than a bespoke protocol.

**AI Infrastructure Alliance** (ai-infrastructure.org)
Community tracking the broader AI infra landscape — useful for staying current on tooling churn in a field that moves monthly, not yearly.

**[LMSYS Blog](https://lmsys.org/blog/)**
Research-adjacent but practitioner-readable. The team behind vLLM and Chatbot Arena; their posts are usually the first public writeup of a serving technique before it shows up in mainstream tooling.

---

## Reading order

Read Building Effective Agents first regardless of what you're building — it reframes "should this be an agent" as a real engineering decision rather than a default. Writing Tools for Agents next if you're past the design stage and actually implementing tool calls. Demystifying Evals last, once you have something running that you need to actually validate — reading it before you have an agent to evaluate makes the grader taxonomy abstract instead of useful.
