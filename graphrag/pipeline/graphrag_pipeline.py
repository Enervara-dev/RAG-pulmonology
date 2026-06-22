import asyncio
import json

from graphrag.config.settings import settings
from graphrag.domain import (
    PULMONOLOGY_RELEVANCE_THRESHOLD,
    detect_red_flags,
    emergency_blocks,
    out_of_scope_blocks,
    refusal_blocks,
)
from graphrag.domain.clinical_policy import (
    ASSESSMENT_READY_INTENT,
    MAX_DIAGNOSTIC_TURNS,
)
from Memory_Layer.session_memory.models import Role
from graphrag.retrievers.pinecone_retriever import PineconeRetriever
from graphrag.retrievers.neo4j_retriever import Neo4jRetriever
from graphrag.processors.entity_processor import EntityProcessor
from graphrag.llm.gemini_llm import GeminiLLM
from graphrag.memory import SessionMemoryAdapter
from graphrag.query_understanding import (
    QueryType,
    RoutingMode,
    decide_routing,
    get_config,
    is_trivial_input,
)
from graphrag.query_understanding.analyzer import MedicalQueryAnalyzer
from graphrag.utils.logger import get_logger
from graphrag.validators.answer_validator import blocks_to_text

logger = get_logger(__name__)

_SEPARATOR = "─" * 72

# Max entities handed to graph traversal (chunk + query + memory combined).
_MAX_GRAPH_ENTITIES = 40


def _normalize_entity(value: str) -> str:
    """Canonicalise an entity term to match :Entity.name in the graph."""
    return str(value).strip().lower().replace("_", " ")


def _entities_from_analysis(analysis) -> list[str]:
    """Medical entities the gatekeeper extracted from the user's query."""
    if not isinstance(analysis, dict):
        return []
    ents = analysis.get("medical_entities") or {}
    out: list[str] = []
    for key in ("symptoms", "conditions", "drugs"):
        out.extend(ents.get(key) or [])
    return out


def _entities_from_memory(wm) -> list[str]:
    """Clinical entities already accumulated in session memory."""
    state = wm.state
    out: list[str] = []
    for vals in (state.symptoms, state.conditions, state.chronic_conditions,
                 state.drugs, state.discussed_entities):
        out.extend(vals or [])
    return out


def _merge_graph_entities(*sources: list[str]) -> list[str]:
    """Normalise + de-dupe entity sources, preserving order (first source wins)."""
    seen: set[str] = set()
    merged: list[str] = []
    for src in sources:
        for e in src:
            n = _normalize_entity(e)
            if n and n not in seen:
                seen.add(n)
                merged.append(n)
                if len(merged) >= _MAX_GRAPH_ENTITIES:
                    return merged
    return merged


class GraphRAGPipeline:
    def __init__(self, redis_url: str | None = None):
        try:
            self.pinecone_retriever = PineconeRetriever()
            self.neo4j_retriever    = Neo4jRetriever()
            self.llm                = GeminiLLM()
            self.entity_processor   = EntityProcessor()
            self.query_analyzer     = MedicalQueryAnalyzer()
            self.memory_adapter     = SessionMemoryAdapter(redis_url=redis_url)

            self._episodic = None
            # Persistent event loop for the episodic async calls. Created lazily
            # on first use so we don't pay startup cost when --user-id is unused.
            # Using one loop for the whole pipeline lifetime (vs asyncio.run per
            # call) prevents the 'Event loop is closed' error from the genai
            # SDK's cached AsyncClient binding to a now-dead loop.
            self._loop = None
            if settings.EPISODIC_MEMORY_ENABLED:
                try:
                    from episodic.api.dependencies import build_container
                    self._episodic = build_container()
                    logger.info("📚 Episodic memory layer ACTIVE (pass --user-id to use it)")
                except Exception as exc:
                    logger.warning(
                        "Episodic memory layer disabled — failed to initialize: %s", exc
                    )

            logger.info("\n" + "★" * 80)
            logger.info("★  GRAPH-RAG ENGINE  ·  QUERY UNDERSTANDING LAYER ACTIVE")
            logger.info("★  Stack: Classify → Vector → Rerank → Graph → LLM")
            logger.info("★" * 80 + "\n")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise

    # ------------------------------------------------------------------

    def _emit_and_store(self, blocks, *, session, user_query, analysis, query_type):
        """Persist a canned/short-circuit block response to memory, then yield it.

        Used by the no-LLM paths (refuse / out-of-scope) so they stream typed
        blocks over the SAME NDJSON transport as a generated answer.
        """
        self.memory_adapter.update_after_interaction(
            session=session,
            user_query=user_query,
            assistant_answer=blocks_to_text(blocks),
            analysis=analysis,
            query_type=query_type,
        )
        yield from blocks

    def run(self, query_text: str, session_id: str = "default", user_id: str | None = None):
        """Run one turn and STREAM the answer as validated NDJSON block dicts.

        This is a generator: it yields one block `dict` at a time (the canned
        short-circuits and the streamed Stage-4 answer alike). Session memory is
        updated as the generator runs; episodic memory is written only at
        `end_session`. Callers serialize blocks to NDJSON (`json.dumps + "\\n"`).
        """
        original_query_text = query_text
        logger.info(f"\n{'═' * 72}")
        logger.info(f"📝 Original Query: {query_text}")
        logger.info(f"{'═' * 72}")

        logger.info(f"\n{_SEPARATOR}")
        logger.info("STAGE -2 -> Session Memory Load")
        logger.info(_SEPARATOR)

        memory_bundle = self.memory_adapter.load(session_id)
        session = memory_bundle.session
        working_memory = memory_bundle.working_memory
        memory_query_text = self.memory_adapter.build_retrieval_query(
            query_text=query_text,
            wm=working_memory,
        )
        analyzer_query_text = (
            memory_query_text
            if working_memory.turn_count or working_memory.has_summary
            else query_text
        )

        # ── Stage -1: Medical Gatekeeper / Query Analyzer ─────────────────
        logger.info(f"\n{_SEPARATOR}")
        logger.info("🛡️   STAGE -1 → Medical Gatekeeper & Analyzer")
        logger.info(_SEPARATOR)

        trivial_skip = is_trivial_input(original_query_text) and working_memory.turn_count > 0
        if trivial_skip:
            logger.info("⏭️  Trivial acknowledgment in established session — skipping gatekeeper LLM.")
            analysis = {}
        else:
            analysis = self.query_analyzer.analyze(analyzer_query_text)

        # ── Deterministic emergency detection ─────────────────────────────
        # Flag clear cardiopulmonary red flags even if the gatekeeper LLM missed
        # them, failed, or was skipped. Conservative phrase matching — see
        # graphrag/domain/clinical_policy.py::RED_FLAG_PATTERNS. Instead of
        # returning a bare alarm, we ESCALATE the turn (critical risk + full
        # retrieval) and let the answer LLM produce a structured, reasoned
        # emergency response. The canned emergency_blocks() are the fallback only.
        red_flags = detect_red_flags(original_query_text)
        emergency = bool(red_flags)
        if red_flags:
            logger.info("🚨 Red-flag detected: %s — escalating to emergency response.", ", ".join(red_flags))

        if analysis and "error" not in analysis and analysis.get("final_action"):
            logger.info(f"🧠 Analysis Results:\n{json.dumps(analysis, indent=2)}")

            final_action = analysis.get("final_action")
            if final_action == "refuse" and not emergency:
                logger.info("⛔ Non-medical query refused.")
                yield from self._emit_and_store(
                    refusal_blocks(),
                    session=session,
                    user_query=original_query_text,
                    analysis=analysis,
                    query_type="unknown",
                )
                return
            elif final_action == "emergency_redirect":
                logger.info("🚨 Gatekeeper flagged emergency — escalating to emergency response.")
                emergency = True

            # ── Pulmonology scope gate ────────────────────────────────────
            # Restrict medical-but-out-of-specialty queries. Greetings,
            # conversational follow-ups, and emergencies are exempt (the latter
            # must always be answered).
            gk_intent = analysis.get("intent")
            relevance = analysis.get("pulmonology_relevance")
            if (
                not emergency
                and isinstance(relevance, (int, float))
                and relevance < PULMONOLOGY_RELEVANCE_THRESHOLD
                and gk_intent not in ("greeting", "followup_query", "emergency")
                and final_action != "route_to_followup"
            ):
                logger.info(
                    "⛔ Out of pulmonology scope (relevance=%s < %s) — restricting.",
                    relevance, PULMONOLOGY_RELEVANCE_THRESHOLD,
                )
                yield from self._emit_and_store(
                    out_of_scope_blocks(),
                    session=session,
                    user_query=original_query_text,
                    analysis=analysis,
                    query_type="out_of_scope",
                )
                return

            # ── Terminal-state gate: stop the follow-up loop ──────────────
            # The session carries a turn counter (working_memory). Once the user
            # has taken more than MAX_DIAGNOSTIC_TURNS turns in a diagnostic
            # exchange, OR the gatekeeper no longer needs a follow-up, force the
            # terminal `assessment_ready` intent so Stage 4 concludes with a
            # final assessment instead of asking again.
            gk_intent = analysis.get("intent")
            prior_user_turns = sum(
                1 for t in working_memory.recent_turns if t.role == Role.USER
            )
            current_user_turn = prior_user_turns + 1   # current message not yet stored
            in_diagnostic_loop = (
                gk_intent in ("followup_query", ASSESSMENT_READY_INTENT)
                or bool(working_memory.state.symptoms or working_memory.state.conditions)
            )
            if in_diagnostic_loop and (
                current_user_turn > MAX_DIAGNOSTIC_TURNS
                or analysis.get("needs_followup") is False
            ):
                logger.info(
                    "🧭 Terminal state → forcing intent=assessment_ready "
                    "(user_turn=%d > %d? | needs_followup=%s)",
                    current_user_turn, MAX_DIAGNOSTIC_TURNS, analysis.get("needs_followup"),
                )
                analysis["intent"] = ASSESSMENT_READY_INTENT
                analysis["needs_followup"] = False
                analysis["followup_questions"] = []
                # Retrieve (don't route to follow-up) so the final assessment
                # gets clinical backfill rather than a memory-only deflection.
                analysis["final_action"] = "retrieve"

            # NOTE: triage follow-up questions are no longer appended
            # deterministically here. The answer model emits them as a
            # `follow_up_questions` block (guided by QUESTIONING_POLICY and the
            # per-intent guidance), and the per-line validator drops that block
            # when the turn is terminal. See answer_prompt.py / answer_validator.

            rewritten = analysis.get("rewritten_query")
            if rewritten and rewritten.strip() and rewritten != query_text:
                logger.info(f"🔄 Query optimized: '{rewritten}'")
                query_text = rewritten
        elif not trivial_skip:
            logger.warning("⚠️ Query Analyzer returned no valid result. Proceeding with original query.")
            analysis = {}

        # ── Stage 0: Query Understanding & Routing ────────────────────────
        routing_mode, query_type = decide_routing(
            analysis=analysis,
            wm=working_memory,
            raw_query=original_query_text,
        )

        # Emergencies always get full retrieval + a critical-risk answer so the
        # response can name possible serious causes and a grounded next step.
        if emergency:
            routing_mode = RoutingMode.HYBRID_RAG
            if query_type in (QueryType.UNKNOWN, QueryType.OUT_OF_CONTEXT):
                query_type = QueryType.SYMPTOM_QUERY

        config = get_config(query_type)
        intent_str = (analysis or {}).get("intent") or "unknown"
        answer_risk_level = "critical" if emergency else ((analysis or {}).get("risk_level") or "none")

        if routing_mode == RoutingMode.NO_RETRIEVAL:
            logger.info("⏭️  ROUTING: NO_RETRIEVAL (memory-only response)")
            vector_top_k = 0
            reranker_top_k = 0
            graph_hops = 0
        elif routing_mode == RoutingMode.MEMORY_FIRST:
            logger.info("🧠 ROUTING: MEMORY_FIRST (small clinical backfill, no graph)")
            vector_top_k = 3
            reranker_top_k = 3
            graph_hops = 0
        else:  # HYBRID_RAG
            logger.info("🔍 ROUTING: HYBRID_RAG (full retrieval active)")
            vector_top_k = config.vector_top_k
            reranker_top_k = config.reranker_top_k
            graph_hops = config.graph_hops

        # Stage-4 closure flags — drive the terminal / NO_RETRIEVAL-conclude
        # constraint appended to the answer system prompt.
        memory_only = routing_mode == RoutingMode.NO_RETRIEVAL
        needs_followup_flag = bool((analysis or {}).get("needs_followup"))
        has_findings = bool(
            working_memory.state.symptoms
            or working_memory.state.conditions
            or _entities_from_analysis(analysis)
        )

        retrieval_query_text = self.memory_adapter.build_retrieval_query(
            query_text=query_text,
            wm=working_memory,
        )

        logger.info(f"   Intent  : {intent_str.upper()}")
        logger.info(f"   Mode    : {routing_mode.value.upper()}")
        logger.info(f"   top_k   : {vector_top_k}")
        logger.info(f"   Graph   : {'enabled' if graph_hops > 0 else 'GATED/DISABLED'}")

        # ── Stage 1: Vector Retrieval + Reranking ─────────────────────────
        logger.info(f"\n{_SEPARATOR}")
        logger.info("⚙️   STAGE 1 → Vector Retrieval + Reranking")
        logger.info(_SEPARATOR)

        if vector_top_k > 0:
            matches = self.pinecone_retriever.retrieve(
                retrieval_query_text,
                vector_top_k=vector_top_k,
                reranker_top_k=reranker_top_k,
            )
        else:
            logger.info("❌ Vector retrieval skipped.")
            matches = []

        # ── Stage 2: Entity Extraction ────────────────────────────────────
        logger.info(f"\n{_SEPARATOR}")
        logger.info("⚙️   STAGE 2 → Entity Extraction")
        logger.info(_SEPARATOR)

        vector_context_str, extracted_entities, _ = self.entity_processor.process_matches(
            matches,
            priority_entity_types = config.priority_entity_types,
            boost_drug_pairs      = config.boost_drug_pairs,
            query                 = retrieval_query_text,
        )

        # ── Stage 3: Graph Retrieval ──────────────────────────────────────
        logger.info(f"\n{_SEPARATOR}")
        logger.info("⚙️   STAGE 3 → Knowledge Graph Traversal")
        logger.info(_SEPARATOR)

        # Hybrid entity set for traversal: entities from the retrieved chunks
        # (most reliable — same corpus/canonicalisation as the graph), plus the
        # entities the gatekeeper pulled from the query and the clinical state in
        # memory. Keeps the graph live even when chunk metadata is sparse.
        graph_entities = _merge_graph_entities(
            extracted_entities,
            _entities_from_analysis(analysis),
            _entities_from_memory(working_memory),
        )

        if graph_hops > 0 and graph_entities:
            logger.info(
                "🔗 Graph entities (%d): %s",
                len(graph_entities), ", ".join(graph_entities[:15]),
            )
            graph_context_list = self.neo4j_retriever.retrieve_relations(
                graph_entities,
                hops  = graph_hops,
                limit = 20,
            )
            graph_context_str = "\n".join([f"- {g}" for g in graph_context_list]) if graph_context_list else "No relevant relations found."
        else:
            logger.info(f"⏭️   Graph skipped (Gated or no entities).")
            graph_context_str = ""

        # ── Stage 3.5: Episodic Memory Retrieval ─────────────────────────
        # Only when a --user-id is supplied AND the episodic container
        # initialized cleanly. Best-effort: a failure here degrades to no
        # episodic context, never breaks the pipeline.
        episodic_context_str = ""
        if user_id and self._episodic is not None:
            logger.info(f"\n{_SEPARATOR}")
            logger.info("🧠  STAGE 3.5 → Episodic Memory Retrieval")
            logger.info(_SEPARATOR)
            episodic_context_str = self._load_episodic_context(
                user_id=user_id, query_text=retrieval_query_text
            )
            if episodic_context_str:
                logger.info(f"   Episodic context: {len(episodic_context_str)} chars")
            else:
                logger.info("   Episodic context: empty")

        # ── Stage 4: LLM ─────────────────────────────────────────────────
        logger.info(f"\n{_SEPARATOR}")
        logger.info("⚙️   STAGE 4 → LLM Response Generation")
        logger.info(_SEPARATOR)

        # Assemble the full conversational context
        memory_payload = self.memory_adapter.assemble_payload(
            wm=working_memory,
            user_query=original_query_text,
            query_type=intent_str,
            goal=config.goal,
            vector_context=vector_context_str,
            graph_context=graph_context_str,
        )

        # Concatenate episodic memory in front of the Redis session block so
        # the answer LLM sees long-term patient history before short-term turns.
        combined_memory_context = memory_payload.memory_context
        if episodic_context_str:
            combined_memory_context = (
                episodic_context_str.strip() + "\n\n" + combined_memory_context
            )

        # Stream the answer as validated NDJSON blocks. risk_level drives the
        # urgency layer of the system prompt (critical → emergency block sequence:
        # warning → summary → condition_list → next_steps). Each block is yielded
        # to the caller AS IT STREAMS and accumulated for the memory write.
        answer_blocks: list[dict] = []
        for block in self.llm.generate_response(
            query_text      = original_query_text,
            vector_context  = vector_context_str,
            graph_context   = graph_context_str,
            memory_context  = combined_memory_context,
            conversation_history = memory_payload.conversation_context,
            query_type      = intent_str,
            goal            = config.goal,
            risk_level      = answer_risk_level,
            needs_followup  = needs_followup_flag,
            memory_only     = memory_only,
            has_findings    = has_findings,
        ):
            answer_blocks.append(block)
            yield block

        # Safety net: if generation produced nothing during an emergency, never
        # leave the patient with nothing — emit the canned emergency blocks.
        if emergency and not answer_blocks:
            logger.warning("Emergency generation produced no blocks — using canned emergency blocks.")
            for block in emergency_blocks():
                answer_blocks.append(block)
                yield block

        # Persist a plain-text rendering of the streamed blocks to session memory.
        self.memory_adapter.update_after_interaction(
            session=session,
            user_query=original_query_text,
            assistant_answer=blocks_to_text(answer_blocks),
            analysis=analysis,
            query_type=("emergency" if emergency else query_type.value),
        )

        # NOTE: episodic memory is NOT written per-turn. It is consolidated and
        # written ONCE when the conversation closes — call `end_session(...)`
        # (exposed as POST /session/end). This keeps long-term memory to one
        # coherent episode per consultation instead of fragmented per-message ones.

    # ------------------------------------------------------------------
    # Episodic memory helpers (sync wrappers around the async services)
    # ------------------------------------------------------------------

    def _run_async(self, coro):
        """
        Run a coroutine on the pipeline's persistent event loop.

        Pinned to one loop for the pipeline's lifetime so the genai SDK's
        cached AsyncClient (and any other async http session it holds onto)
        stays bound to a live loop across multiple turns.
        """
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def _load_episodic_context(self, *, user_id: str, query_text: str) -> str:
        """Return a prompt-ready episodic context block. Returns '' on any failure."""
        try:
            from episodic.schemas.retrieval import RetrievalRequest
            req = RetrievalRequest(user_id=user_id, query_text=query_text)
            block = self._run_async(self._episodic.context_pipeline.build(req))
            return block.rendered_prompt or ""
        except Exception as exc:
            logger.warning("Episodic context load failed: %s", exc)
            return ""

    def end_session(self, *, user_id: str, session_id: str = "default") -> dict:
        """
        Close a conversation and write ONE consolidated episode to episodic
        memory. Call this when the chat ends (exposed as POST /session/end).

        Loads the session's accumulated state + rolling summary + patient
        statements, builds a single digest, and runs it through the episodic
        ingest pipeline (extract → contradiction check → store). Best-effort and
        safe to call multiple times. Returns a small status dict for the API.
        """
        if not user_id:
            return {"stored": False, "reason": "no user_id — episodic memory is per-user only"}
        if self._episodic is None:
            return {"stored": False, "reason": "episodic memory is not active"}

        try:
            bundle = self.memory_adapter.load(session_id)
            digest = self.memory_adapter.build_session_digest(bundle.working_memory)
        except Exception as exc:
            logger.warning("end_session: could not load/condense session %s: %s", session_id, exc)
            return {"stored": False, "reason": f"session load failed: {exc}"}

        if not digest.strip():
            return {"stored": False, "reason": "empty session — nothing clinical to store"}

        return self._run_episodic_ingest(user_id=user_id, text=digest)

    def _run_episodic_ingest(self, *, user_id: str, text: str) -> dict:
        """
        Run one text through the episodic ingest pipeline. Best-effort: any LLM
        rate-limit / network error is logged and swallowed. Returns a status dict.
        """
        try:
            result = self._run_async(
                self._episodic.ingest_pipeline.run(user_id=user_id, utterance=text)
            )
        except Exception as exc:
            logger.warning("Episodic ingest failed: %s", exc)
            return {"stored": False, "reason": f"ingest error: {exc}"}

        if result.stored is not None:
            logger.info(
                "📥 Episodic memory ingested: episode_id=%s category=%s priority=%s",
                result.stored.episode_id,
                result.stored.category.value,
                result.stored.clinical_priority.value,
            )
            status = {"stored": True, "episode_id": str(result.stored.episode_id),
                      "category": result.stored.category.value}
        elif result.clarification.needs_clarification:
            qs = "; ".join(q.question for q in result.clarification.questions)
            logger.info("📝 Episodic ingest deferred — clarification needed: %s", qs)
            status = {"stored": False, "reason": "clarification needed", "questions": qs}
        else:
            logger.info("📭 Episodic ingest skipped (no clinical content extracted).")
            status = {"stored": False, "reason": "no clinical content extracted"}

        if result.contradictions.has_contradictions:
            logger.info(
                "⚠️  Episodic contradiction signal: %d item(s), penalty=%.2f, triggers_clarification=%s",
                len(result.contradictions.contradictions),
                result.contradictions.confidence_penalty,
                result.contradictions.triggers_clarification,
            )
            status["contradictions"] = len(result.contradictions.contradictions)

        return status

    # ------------------------------------------------------------------

    def close(self):
        if hasattr(self, "neo4j_retriever"):
            self.neo4j_retriever.close()
        loop = getattr(self, "_loop", None)
        if loop is not None and not loop.is_closed():
            try:
                loop.close()
            except Exception as exc:
                logger.debug("Pipeline loop close raised: %s", exc)
