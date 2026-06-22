# Architecture — Enervera Pulmonology Assistant

A pulmonology-specialised medical assistant built on **Hybrid GraphRAG**. The
project is two pipelines joined by three data stores:

1. **Offline knowledge pipeline** — reference PDFs → validated, graph-ready
   micro-chunks → Pinecone (vectors) + Neo4j (knowledge graph).
2. **Online GraphRAG runtime** — a user query flows through a safety gatekeeper,
   hybrid retrieval (vectors + graph + memory), and a streamed clinical answer.

Both are **retargetable to another specialty by editing `domain/` packages
only** — the orchestration, loaders, retrievers, and memory engine are
domain-agnostic.

Companion docs: [`HANDOFF.md`](HANDOFF.md) (run/deploy/production roadmap),
[`RETARGETING.md`](RETARGETING.md) (specialty swap playbook), root
[`README.md`](../README.md) (the chunker in detail).

---

## 1. System map

```
                       OFFLINE (build the knowledge base)
 ┌──────────────────────────────────────────────────────────────────────┐
 │  dataset/*.pdf                                                         │
 │      │  chunker.py  → chunking/ pipeline (load→clean→segment→LLM       │
 │      ▼                 extract→validate→store)                         │
 │  chunks/v1/*.json   (MicroChunk: text + entities + relations + summary)│
 │      │                                                                 │
 │      ├─ ingest_pinecone.py  ──►  Pinecone  index ns=pulmonology_v1     │
 │      └─ ingest_neo4j.py     ──►  Neo4j  (:Entity / :Chunk + relations) │
 └──────────────────────────────────────────────────────────────────────┘
                                    │
                       shared data stores
                                    │
                       ONLINE (answer the user)
 ┌──────────────────────────────────────────────────────────────────────┐
 │  run_graphrag.py (CLI)  /  api.py (FastAPI /chat)                      │
 │      │                                                                 │
 │      ▼  GraphRAGPipeline.run()                                         │
 │  memory load → gatekeeper → routing → vector+rerank → entities →       │
 │  graph traversal → (episodic) → answer stream → memory update         │
 │      │            │                │                                   │
 │   Redis/RAM    Pinecone          Neo4j        Pinecone(episodic)       │
 │   (session)    (pulmonology_v1)  (graph)      (per-user, opt)          │
 └──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Design principles

1. **Domain knowledge is isolated from orchestration.** All specialty/clinical
   facts live in three `domain` surfaces:
   [`chunking/domain.py`](../chunking/domain.py) (offline),
   [`graphrag/domain/`](../graphrag/domain/) and
   [`Memory_Layer/session_memory/domain/`](../Memory_Layer/session_memory/domain/)
   (online). A specialty swap edits those plus a data re-ingest — not a rewrite.
2. **Safety is layered, not single-point.** The LLM gatekeeper estimates risk,
   but a **deterministic red-flag backstop** catches cardiopulmonary emergencies
   even if the LLM misses, fails, or is skipped.
3. **Degrade, don't crash.** If Pinecone / Neo4j / Gemini / Redis are down, the
   affected stage returns empty and the pipeline continues (answers fall back to
   model-only knowledge). Episodic memory is best-effort — it warns, never breaks.
4. **Sequential, observable stages.** The runtime flows through numbered stages
   (`STAGE -2` … `STAGE 5`), each emitting a log banner, so a transcript is
   self-documenting.

---

## 3. Offline pipeline — PDF → knowledge base

Entry point [`chunker.py`](../chunker.py) processes every PDF in `dataset/`
through [`chunking/`](../chunking/):

| Stage | Package | Responsibility |
|---|---|---|
| Load | `chunking/loaders/` | PDF (PyMuPDF) + CSV → raw text with page tracking |
| Clean | `chunking/cleaners/` | normalize text; strip OCR glyphs, headers/footers |
| Segment | `chunking/detectors/` + `extractors/` | section detection → ~350-token semantic blocks with page ranges |
| Extract | `chunking/llm/` (Gemini) | per-block entities, directional relations, summary, significance note |
| Validate | `chunking/validators/` + `schemas/` | strict Pydantic schema gate, partial-recovery on parse failure |
| Store | `chunking/storage/` | versioned JSON (`chunks/v1/…`) |

Orchestration (`chunking/pipeline/`) handles batching, parallel workers, resume,
and a run summary. The output unit is a **`MicroChunk`**: text + entities +
relations + summary + significance — the shape both ingest scripts consume.

`chunk_pages_27_845.py` is an example driver (plan / smoke / full run over a page
range); `test_pipeline.py` is the offline chunking smoke test (`--live` for real).

**Ingest bridges** (chunks → stores):
- [`ingest_pinecone.py`](../ingest_pinecone.py) `--namespace pulmonology_v1` —
  embeds chunk text into the main vector index. **Must** use that namespace;
  retrieval is locked to it.
- [`ingest_neo4j.py`](../ingest_neo4j.py) `--version v1` — writes `:Entity` /
  `:Chunk` nodes and relations; entity names are canonical/lowercase to match
  query-time lookups.

[`live_check.py`](../live_check.py) verifies the stores are populated (Pinecone
namespace vector count + Neo4j node counts).

---

## 4. Online runtime — query → answer

Orchestrated by
[`graphrag/pipeline/graphrag_pipeline.py`](../graphrag/pipeline/graphrag_pipeline.py)
`:: GraphRAGPipeline.run()`:

```
user query
 │
 ├─ STAGE -2  Session memory load            (memory.SessionMemoryAdapter → Memory_Layer)
 │              → builds a memory-aware "retrieval query" from prior turns + summary
 │
 ├─ STAGE -1  Gatekeeper / analyzer (LLM)    (query_understanding.analyzer + domain.prompts)
 │              → intent, risk, pulmonology_relevance, medical_entities, follow-ups
 │              → trivial acks in an established session skip the LLM entirely
 │   ├─ Deterministic red-flag detection     (domain.clinical_policy.detect_red_flags)
 │   │     → conservative regex; on hit, ESCALATE (critical risk + full retrieval)
 │   ├─ Refuse                                (non-medical → REFUSAL_MESSAGE)
 │   └─ Scope-gate                            (pulmonology_relevance < 75 → OUT_OF_SCOPE_MESSAGE)
 │
 ├─ STAGE  0  Routing                         (query_understanding.routing / query_config)
 │              → NO_RETRIEVAL | MEMORY_FIRST | HYBRID_RAG
 │              → emergencies forced to HYBRID + critical
 │
 ├─ STAGE  1  Vector retrieval + rerank       (retrievers.pinecone_retriever, ns=pulmonology_v1)
 ├─ STAGE  2  Entity extraction               (processors.entity_processor — from chunks)
 ├─ STAGE  3  Graph traversal                 (retrievers.neo4j_retriever)
 │              entities = chunk + query + memory  (hybrid, normalized, deduped, ≤40)
 ├─ STAGE 3.5 Episodic retrieval (opt)        (episodic.* — only with --user-id)
 │
 ├─ STAGE  4  Answer generation (LLM stream)  (llm.gemini_llm + domain.answer_prompt)
 │              risk_level drives urgency; critical → structured emergency answer
 │
 ├─ STAGE  5  Episodic ingest (opt)           (episodic.* — written at session close)
 └─ memory update                             (state extract → summarize → save)
```

Stage numbering is intentional: retrieval is `STAGE 1`, so the two pre-retrieval
stages (memory load, gatekeeper) are `-2` and `-1`.

### 4.1 Query understanding — [`graphrag/query_understanding/`](../graphrag/query_understanding/)
- **`analyzer.py`** — the **gatekeeper LLM**. Runs `GATEKEEPER_SYSTEM_PROMPT`
  (from `domain/prompts.py`) and returns a JSON analysis: `intent`, `risk`,
  `pulmonology_relevance`, `medical_entities`, triage `follow_ups`.
- **`routing.py`** — pure `decide_routing()` maps the analysis to a `RoutingMode`
  (`NO_RETRIEVAL` / `MEMORY_FIRST` / `HYBRID_RAG`). Trivial acks and conversational
  continuations short-circuit; a failed gatekeeper degrades to the cheap path;
  emergencies are forced to full hybrid retrieval.
- **`query_config.py`** — per-`QueryType` retrieval tuning (top-k, rerank, hops).

### 4.2 Retrieval
- **`retrievers/pinecone_retriever.py`** — vector search + rerank, locked to
  namespace `pulmonology_v1` (`domain/vocabulary.py`).
- **`retrievers/neo4j_retriever.py`** — graph traversal over `:Entity`/`:Chunk`.
  Seed entities are a **hybrid merge** of chunk + query + memory terms, normalized
  (`lower`, `_`→space) and deduped to ≤40 (`_merge_graph_entities`).
- **`processors/entity_processor.py`** — extracts entities from retrieved chunks.

### 4.3 Answer generation
- **`llm/gemini_llm.py`** + **`domain/answer_prompt.py`** — streamed answer with a
  layered prompt: `SPECIALTY` persona knob, per-intent guidance, and
  `risk_level`-driven urgency (critical → a structured emergency response).

### 4.4 Session memory — [`Memory_Layer/session_memory/`](../Memory_Layer/session_memory/)
Multi-turn triage continuity. `state_extractor` pulls symptoms/conditions/drugs
via regex and computes a **symptom-weighted risk** (max of gatekeeper risk and
symptom-derived risk); `summarizer` keeps a deterministic rolling summary;
`session_manager` persists to **Redis with a RAM fallback**. Reached from the
pipeline via the sync facade `memory/session_adapter.py`.

### 4.5 Episodic memory — [`episodic/`](../episodic/)
Optional long-term, per-user recall (Pinecone, index `episodicmemory`). Activated
only with `--user-id`; embedded via `episodic/api/dependencies.py::build_container()`.
**Written once at session close** (`POST /session/end`), not per-turn.

---

## 5. The domain layer (retarget surface)

| File | Holds |
|---|---|
| `chunking/domain.py` | offline: entity/relation types, extraction prompt, segmentation patterns, validation thresholds |
| `graphrag/domain/prompts.py` | gatekeeper system prompt (red flags, relevance rubric) |
| `graphrag/domain/answer_prompt.py` | answer prompt, `SPECIALTY` knob, risk layers |
| `graphrag/domain/clinical_policy.py` | red-flag regex, high-signal symptoms, triage policy, `MAX_FOLLOWUP_QUESTIONS` |
| `graphrag/domain/query_taxonomy.py` | query types, per-type tuning, intent→type map |
| `graphrag/domain/vocabulary.py` | `PINECONE_NAMESPACE`, relevance threshold (75), graph node label |
| `graphrag/domain/entity_rules.py` / `messages.py` | drug-pair heuristics / refusal & emergency copy |
| `Memory_Layer/session_memory/domain/` | extraction patterns, risk rules, render labels |

---

## 6. Safety architecture

Three independent guardrails, in execution order:

1. **Deterministic red-flag backstop** (`clinical_policy.detect_red_flags`) — runs
   regardless of the LLM. A hit escalates to **critical risk + full retrieval** and
   produces a reasoned, structured emergency answer (not a bare alarm).
2. **Scope gate** — queries below the **75% pulmonology-relevance** threshold get
   the out-of-scope message; non-medical queries are refused.
3. **Terminal triage state** — `assessment_ready` (or the gatekeeper no longer
   needing follow-ups, or exceeding `MAX_DIAGNOSTIC_TURNS`) ends the follow-up loop
   and forces a final assessment, preventing endless question loops.

These are guardrails, not a substitute for clinical sign-off.

---

## 7. Delivery surfaces, config & dependencies

**Entrypoints:**
- `run_graphrag.py` — CLI / REPL (`--query`, `--session-id`, `--user-id`, `--quiet`).
- `api.py` — FastAPI (`GET /`, `GET /health`, `POST /chat`). `/chat` is a sync
  `def` so FastAPI threadpools it; calls are serialized with a lock (the shared
  pipeline isn't thread-safe). Deployed via `render.yaml` (`uvicorn api:app`);
  API deps in `requirements-api.txt`.

**External stores:**

| Store | Used by | Failure mode |
|---|---|---|
| Pinecone (`pulmonology_v1`) | vector retrieval | empty results → model-only answer |
| Neo4j | graph traversal | `⏭️ Graph skipped` → degraded recall |
| Pinecone (`episodicmemory`) | per-user memory (opt) | best-effort; warns, never breaks |
| Redis | shared session memory | RAM fallback (sessions lost on restart) |
| Gemini API | chunk extraction, gatekeeper, answer, episodic | retrieval/answer degrade or error |

**Config:** environment variables read by `graphrag/config/settings.py` (required:
`GEMINI_API_KEY`, `PINECONE_API_KEY`, `NEO4J_PASSWORD` — fails fast otherwise).
Specialty "domain knobs" live in code (not env) so a swap is one edit:
`PINECONE_NAMESPACE`, relevance threshold, graph node label
(`domain/vocabulary.py`), `SPECIALTY` (`answer_prompt.py`),
`MAX_FOLLOWUP_QUESTIONS` (`clinical_policy.py`).

---

## 8. Cross-cutting concerns

- **Cost:** Gemini calls per turn = gatekeeper (every non-trivial turn) + answer
  (stream); episodic adds extract/contradiction/clarify on ingest; the offline
  chunker is the largest one-time LLM cost. Output tokens dominate.
- **Latency:** runtime stages are sequential (gatekeeper → retrieval → rerank →
  graph → answer). Emergencies run full retrieval — slightly slower but safer.
- **Scaling:** the runtime is stateless except for store connections →
  horizontally scalable once Redis is shared and multiple uvicorn workers run
  (each gets its own pipeline instance).
- **Testing:** chunking has `test_pipeline.py`; the GraphRAG runtime is verified
  by `smoke_test.py` (stubs externals). Quick sanity:
  `python -m compileall graphrag Memory_Layer episodic`.

See [`HANDOFF.md` §9](HANDOFF.md) for the prioritized hardening roadmap (HTTP API
scale, real Redis, observability, `graph_enabled` wiring, compliance).
