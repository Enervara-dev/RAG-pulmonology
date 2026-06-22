"""
Offline end-to-end smoke test for the GraphRAG pulmonology assistant.

Runs WITHOUT network or paid calls: Pinecone, Neo4j, and Gemini are stubbed.
It exercises every subsystem and asserts behavior:

  - imports / wiring          (all packages import)
  - domain layer              (namespace, scope threshold, prompts, red flags)
  - session memory            (extraction, symptom-weighted risk, triggers, continuity)
  - entity processor          (plain-name parse, dedup, hybrid merge)
  - full pipeline scenarios   (in-scope, out-of-scope, emergency, terminal state,
                               NO_RETRIEVAL conclude, greeting)
  - Stage-4 prompt injection  (real gemini_llm, generate_stream patched)
  - HTTP API wiring           (best-effort: routes registered)

A LIVE run (real Pinecone/Neo4j/Gemini) still has to be done on a networked
machine — this validates the wiring and decision logic, not the data stores.

Usage:
    python smoke_test.py          # exits 0 if all pass, 1 otherwise
"""

from __future__ import annotations

import io
import logging
import os
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.ERROR)
# Silence the pipeline's own loggers (their handlers bind to the real stdout and
# bypass redirect_stdout) so the smoke-test report stays readable.
logging.disable(logging.CRITICAL)


# ── tiny harness ──────────────────────────────────────────────────────────────
class Report:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def section(self, title: str) -> None:
        print(f"\n=== {title} ===")

    def check(self, name: str, cond: bool, detail: str = "") -> bool:
        ok = bool(cond)
        self.passed += ok
        self.failed += (not ok)
        mark = "PASS" if ok else "FAIL"
        line = f"  [{mark}] {name}"
        if detail:
            line += f"  → {detail}"
        print(line)
        return ok

    def skip(self, name: str, reason: str) -> None:
        print(f"  [SKIP] {name}  → {reason}")


R = Report()


@contextmanager
def silent():
    """Swallow the pipeline's stdout (stage logs / streamed prints) for a clean report."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        yield


# ── stub pipeline builder ─────────────────────────────────────────────────────
def build_stub_pipeline(analyses, *, graph=("asthma -[manifests_as]-> wheeze",),
                        chunk_entities=("asthma", "wheeze"),
                        llm_blocks=({"type": "summary", "data": {"text": "clinical guidance"}},)):
    """
    A GraphRAGPipeline with external services stubbed but real in-process
    components (entity_processor, memory_adapter → RAM). `analyses` is a list of
    gatekeeper-analysis dicts returned in order, one per .run() call. The stub LLM
    STREAMS `llm_blocks` (block dicts), mirroring the real NDJSON contract.
    Returns (pipeline, calls, flags) where `calls` records each generate_response kwargs.
    """
    from graphrag.pipeline.graphrag_pipeline import GraphRAGPipeline
    from graphrag.processors.entity_processor import EntityProcessor
    from graphrag.memory import SessionMemoryAdapter

    p = GraphRAGPipeline.__new__(GraphRAGPipeline)
    p._episodic = None
    p._loop = None
    p.entity_processor = EntityProcessor()
    p.memory_adapter = SessionMemoryAdapter()
    calls: list[dict] = []
    flags = {"retrieved": False}

    class A:
        def __init__(self): self.q = list(analyses)
        def analyze(self, q): return self.q.pop(0) if self.q else {}

    class PC:
        def retrieve(self, *a, **k):
            flags["retrieved"] = True
            return [{"id": "c1", "metadata": {
                "summary": "Asthma is an airway disease that causes wheeze.",
                "entities": list(chunk_entities)}}]

    class N:
        def retrieve_relations(self, *a, **k): return list(graph)
        def close(self): pass

    class L:
        def generate_response(self, **k):
            calls.append(k)
            for b in llm_blocks:
                yield b

    p.query_analyzer = A()
    p.pinecone_retriever = PC()
    p.neo4j_retriever = N()
    p.llm = L()
    return p, calls, flags


def run_blocks(p, *args, **kwargs) -> list[dict]:
    """Consume the pipeline's NDJSON block generator into a list (stdout silenced)."""
    with silent():
        return list(p.run(*args, **kwargs))


def analysis(intent="symptom_query", *, needs_followup=False, relevance=95,
             risk="low", action="retrieve", symptoms=("cough",)):
    return {
        "domain": "health", "intent": intent, "risk_level": risk,
        "pulmonology_relevance": relevance,
        "medical_entities": {"symptoms": list(symptoms), "drugs": [], "conditions": []},
        "rewritten_query": "", "needs_followup": needs_followup,
        "followup_questions": (["q"] if needs_followup else []), "final_action": action,
    }


# ── 1. imports / wiring ───────────────────────────────────────────────────────
def test_imports():
    R.section("1. Imports / wiring")
    import importlib
    mods = [
        "graphrag.pipeline.graphrag_pipeline",
        "graphrag.query_understanding.analyzer",
        "graphrag.query_understanding.routing",
        "graphrag.retrievers.pinecone_retriever",
        "graphrag.retrievers.neo4j_retriever",
        "graphrag.processors.entity_processor",
        "graphrag.llm.gemini_llm",
        "graphrag.domain",
        "Memory_Layer.session_memory",
        "Memory_Layer.session_memory.domain",
        "episodic.api.dependencies",
        "episodic.schemas.retrieval",
    ]
    for m in mods:
        try:
            importlib.import_module(m)
            R.check(f"import {m}", True)
        except Exception as e:
            R.check(f"import {m}", False, repr(e))


# ── 2. domain layer ───────────────────────────────────────────────────────────
def test_domain():
    R.section("2. Domain layer")
    from graphrag.domain import (
        PINECONE_NAMESPACE, PULMONOLOGY_RELEVANCE_THRESHOLD,
        GATEKEEPER_SYSTEM_PROMPT, compose_system_prompt, detect_red_flags,
    )
    from graphrag.domain.clinical_policy import (
        closure_directive, ASSESSMENT_READY_INSTRUCTION, NO_RETRIEVAL_CONCLUDE_INSTRUCTION,
        MAX_DIAGNOSTIC_TURNS,
    )
    R.check("retrieval namespace = pulmonology_v1", PINECONE_NAMESPACE == "pulmonology_v1", PINECONE_NAMESPACE)
    R.check("scope threshold = 75", PULMONOLOGY_RELEVANCE_THRESHOLD == 75)
    R.check("max diagnostic turns = 2", MAX_DIAGNOSTIC_TURNS == 2)
    R.check("gatekeeper prompt has relevance rubric + red flags + terminal state",
            all(s in GATEKEEPER_SYSTEM_PROMPT for s in
                ("pulmonology_relevance", "RESPIRATORY / CARDIOPULMONARY RED FLAGS", "assessment_ready")))
    from graphrag.domain.answer_prompt import OUTPUT_CONTRACT
    crit = compose_system_prompt(query_type="symptom_query", risk_level="critical", has_name=False)
    R.check("critical answer prompt = emergency block sequence",
            all(s in crit for s in ("EMERGENCY RESPONSE AS BLOCKS", "condition_list", "next_steps")))
    R.check("answer prompt ends with the NDJSON output contract (last layer)",
            crit.rstrip().endswith(OUTPUT_CONTRACT.rstrip())
            and crit.index("OUTPUT FORMAT") > crit.index("STYLE & TONE"))
    R.check("output contract enforces compact one-line JSON",
            "COMPACT JSON" in crit and "WRONG" in crit)
    term = compose_system_prompt(query_type="assessment_ready", risk_level="none",
                                 has_name=False, terminal=True)
    R.check("terminal prompt forbids a follow_up_questions block",
            "TERMINAL TURN" in term and "follow_up_questions" in term)
    pulm = compose_system_prompt(query_type="symptom_query", risk_level="none", has_name=False)
    R.check("answer prompt is pulmonology-tuned", "pulmonology" in pulm.lower())
    # red flag detection
    R.check("red flag: coughing up blood", detect_red_flags("I am coughing up blood") == ["haemoptysis"])
    R.check("red flag: NOT tripped by mild 'cant breathe properly'",
            detect_red_flags("i cant breath properly, sinus") == [])
    # closure directive matrix
    R.check("closure: greeting (no findings) → none",
            closure_directive(intent="greeting", needs_followup=False, memory_only=True, has_findings=False) is None)
    R.check("closure: assessment_ready → terminal instruction",
            closure_directive(intent="assessment_ready", needs_followup=False, memory_only=False, has_findings=True) == ASSESSMENT_READY_INSTRUCTION)
    R.check("closure: NO_RETRIEVAL medical → conclude instruction",
            closure_directive(intent="followup_query", needs_followup=True, memory_only=True, has_findings=True) == NO_RETRIEVAL_CONCLUDE_INSTRUCTION)
    R.check("closure: mid-triage (needs_followup) → none",
            closure_directive(intent="symptom_query", needs_followup=True, memory_only=False, has_findings=True) is None)


# ── 3. session memory ─────────────────────────────────────────────────────────
def test_memory():
    R.section("3. Session memory")
    from Memory_Layer.session_memory import SessionMemory, Message, Role, extract_state, get_working_memory
    from Memory_Layer.session_memory.state_extractor import extract_entities

    raw = extract_entities("chest pain and wheezing, worse in the morning, coughing up blood")
    R.check("respiratory symptom extraction", {"chest_pain", "wheezing", "haemoptysis"} <= set(raw.symptoms), str(raw.symptoms))
    R.check("trigger extraction", "morning" in raw.triggers, str(raw.triggers))

    # symptom-weighted risk in the live path
    s = SessionMemory(session_id="m1")
    s.state = extract_state(s, Message(role=Role.USER, content="I have chest pain", risk_level="low"))
    R.check("critical symptom escalates risk → critical", str(s.state.risk_level) in ("critical", "RiskLevel.CRITICAL"), str(s.state.risk_level))

    # continuity across turns
    s2 = SessionMemory(session_id="m2")
    s2.state = extract_state(s2, Message(role=Role.USER, content="cough for 3 days"))
    s2.add_turn(Message(role=Role.USER, content="cough for 3 days"))
    s2.state = extract_state(s2, Message(role=Role.USER, content="now also wheezing"))
    R.check("symptoms accumulate across turns", {"cough", "wheezing"} <= set(s2.state.symptoms), str(s2.state.symptoms))


# ── 4. entity processor ───────────────────────────────────────────────────────
def test_entities():
    R.section("4. Entity processor")
    from graphrag.processors.entity_processor import EntityProcessor
    from graphrag.pipeline.graphrag_pipeline import _merge_graph_entities, _entities_from_analysis

    # plain-name metadata (real Pinecone format) — the bug we fixed
    _, ents, _ = EntityProcessor.process_matches(
        [{"id": "1", "metadata": {"summary": "s", "entities": ["asthma", "wheeze", "asthma"]}}],
        priority_entity_types=["disease"], query="")
    R.check("plain-name entities extracted + deduped", ents == ["asthma", "wheeze"], str(ents))

    merged = _merge_graph_entities(["asthma"], _entities_from_analysis(
        {"medical_entities": {"symptoms": ["chest_pain"], "drugs": [], "conditions": ["copd"]}}), ["wheeze"])
    R.check("hybrid graph entities (chunk+query+memory, normalized)",
            merged[0] == "asthma" and "chest pain" in merged, str(merged))


# ── 5. full pipeline scenarios ────────────────────────────────────────────────
def test_pipeline_scenarios():
    R.section("5. Full pipeline (stubbed services, NDJSON blocks)")
    from graphrag.domain import out_of_scope_blocks

    def types(blocks):
        return [b["type"] for b in blocks]

    # a) in-scope → retrieval + graph + streamed blocks, graph entities are hybrid
    p, calls, flags = build_stub_pipeline([analysis("symptom_query", needs_followup=True)])
    blocks = run_blocks(p, "breathless and wheezing", session_id="sc_in")
    gctx = calls[-1]["graph_context"]
    R.check("in-scope streamed blocks + retrieval ran",
            "summary" in types(blocks) and flags["retrieved"], str(types(blocks)))
    R.check("graph traversal produced relations", "manifests_as" in gctx, gctx[:60])

    # b) out-of-scope → restricted (canned blocks), retrieval skipped
    p, calls, flags = build_stub_pipeline([analysis("symptom_query", relevance=20, needs_followup=False)])
    blocks = run_blocks(p, "itchy skin rash on my arm", session_id="sc_oos")
    R.check("out-of-scope streams canned blocks (not a string)", blocks == out_of_scope_blocks(), str(blocks))
    R.check("out-of-scope skipped retrieval", flags["retrieved"] is False and not calls)

    # c) emergency (red flag) → streamed answer at critical risk, retrieval ran
    p, calls, flags = build_stub_pipeline([analysis("symptom_query", needs_followup=False)])
    blocks = run_blocks(p, "I am coughing up blood and struggling to breathe", session_id="sc_er")
    R.check("emergency → streamed LLM blocks (not static)", "summary" in types(blocks))
    R.check("emergency → critical risk + retrieval ran",
            calls and calls[-1]["risk_level"] == "critical" and flags["retrieved"],
            str(calls[-1]["risk_level"]) if calls else "no-call")

    # c2) emergency with an LLM that yields NOTHING → canned emergency blocks fallback
    p, calls, _ = build_stub_pipeline([analysis("symptom_query", needs_followup=False)], llm_blocks=())
    blocks = run_blocks(p, "I am coughing up blood and struggling to breathe", session_id="sc_er2")
    R.check("emergency + empty stream → canned emergency blocks",
            types(blocks) == ["warning", "next_steps"] and blocks[0]["data"]["severity"] == "critical",
            str(types(blocks)))

    # d) terminal state: 3 follow-needed turns → 3rd flips to assessment_ready
    p, calls, _ = build_stub_pipeline([analysis("symptom_query", needs_followup=True)] * 3)
    for _ in range(3):
        run_blocks(p, "I have a cough", session_id="sc_turns")
    R.check("turn 1 not terminal", calls[0]["query_type"] == "symptom_query")
    R.check("turn 3 forced → assessment_ready", calls[2]["query_type"] == "assessment_ready" and calls[2]["needs_followup"] is False)

    # e) needs_followup False mid-loop → terminal
    p, calls, _ = build_stub_pipeline([analysis("symptom_query", needs_followup=True),
                                       analysis("symptom_query", needs_followup=False)])
    run_blocks(p, "I have a cough", session_id="sc_nf")
    run_blocks(p, "still coughing", session_id="sc_nf")
    R.check("needs_followup False → assessment_ready", calls[1]["query_type"] == "assessment_ready")

    # f) NO_RETRIEVAL medical follow-up → memory_only + findings (conclude)
    p, calls, _ = build_stub_pipeline([analysis("symptom_query", needs_followup=True),
                                       analysis("followup_query", needs_followup=True, action="route_to_followup")])
    run_blocks(p, "I have a cough", session_id="sc_nr")
    run_blocks(p, "is it serious?", session_id="sc_nr")
    R.check("NO_RETRIEVAL follow-up → memory_only + has_findings",
            calls[1]["memory_only"] is True and calls[1]["has_findings"] is True, str({k: calls[1][k] for k in ("memory_only", "has_findings")}))

    # g) greeting → exempt from scope gate, answered
    p, calls, _ = build_stub_pipeline([analysis("greeting", relevance=5, needs_followup=False)])
    blocks = run_blocks(p, "hello", session_id="sc_hi")
    R.check("greeting exempt → answered", "summary" in types(blocks))


# ── 6. Stage-4 prompt injection (real gemini_llm, generate_stream patched) ─────
def test_stage4_injection():
    R.section("6. Stage-4 prompt injection (real gemini_llm)")
    try:
        import graphrag.llm.gemini_llm as gl
        from graphrag.domain.clinical_policy import ASSESSMENT_READY_INSTRUCTION, NO_RETRIEVAL_CONCLUDE_INSTRUCTION
    except Exception as e:
        R.skip("gemini_llm injection", repr(e))
        return

    cap: dict = {}

    def fake_stream(*, user_prompt, model, system_instruction=None, temperature=None):
        cap["sys"] = system_instruction
        # Emit one valid NDJSON block so the validator yields something real.
        yield '{"type":"summary","data":{"text":"ok"}}'

    gl.generate_stream = fake_stream
    try:
        llm = gl.GeminiLLM()
    except Exception as e:
        R.skip("gemini_llm injection (needs GEMINI_API_KEY for client init)", repr(e))
        return

    def call(**kw):
        # generate_response is a generator now — consume it to drive the stream.
        with silent():
            out = list(llm.generate_response(query_text="q", vector_context="", graph_context="",
                                             memory_context="", conversation_history="", **kw))
        return cap.get("sys", ""), out

    s, out = call(query_type="assessment_ready", needs_followup=False, memory_only=False, has_findings=True)
    R.check("assessment_ready → terminal constraint injected", ASSESSMENT_READY_INSTRUCTION in s)
    R.check("stream yields validated block dicts", out and out[0]["type"] == "summary", str(out[:1]))
    s, _ = call(query_type="followup_query", needs_followup=True, memory_only=True, has_findings=True)
    R.check("NO_RETRIEVAL → conclude constraint injected", NO_RETRIEVAL_CONCLUDE_INSTRUCTION in s)
    s, _ = call(query_type="symptom_query", needs_followup=True, memory_only=False, has_findings=True)
    R.check("mid-triage → NO constraint (follow-up allowed)",
            ASSESSMENT_READY_INSTRUCTION not in s and NO_RETRIEVAL_CONCLUDE_INSTRUCTION not in s)


# ── 8. Episodic memory: session-end only (not per-turn) ───────────────────────
def test_episodic_session_end():
    R.section("8. Episodic — written only at session end")
    from graphrag.pipeline.graphrag_pipeline import GraphRAGPipeline
    from graphrag.processors.entity_processor import EntityProcessor
    from graphrag.memory import SessionMemoryAdapter

    captured: list[str] = []

    class _Stored:
        episode_id = "ep-1"
        class category: value = "symptom"
        class clinical_priority: value = "normal"

    class _Result:
        stored = _Stored()
        class clarification:
            needs_clarification = False
            questions: list = []
        class contradictions:
            has_contradictions = False
            contradictions: list = []
            confidence_penalty = 0.0
            triggers_clarification = False

    class _Ingest:
        async def run(self, *, user_id, utterance):
            captured.append(utterance)
            return _Result()

    class _CtxBlock:
        rendered_prompt = ""

    class _Context:
        async def build(self, req):
            return _CtxBlock()

    class _Episodic:
        ingest_pipeline = _Ingest()
        context_pipeline = _Context()

    p = GraphRAGPipeline.__new__(GraphRAGPipeline)
    p._episodic = _Episodic()
    p._loop = None
    p.entity_processor = EntityProcessor()
    p.memory_adapter = SessionMemoryAdapter()

    class A:
        def analyze(self, q): return analysis("symptom_query", needs_followup=False)
    class PC:
        def retrieve(self, *a, **k): return [{"id": "c", "metadata": {"summary": "s", "entities": ["asthma"]}}]
    class N:
        def retrieve_relations(self, *a, **k): return []
        def close(self): pass
    class L:
        def generate_response(self, **k):
            yield {"type": "summary", "data": {"text": "ok"}}
    p.query_analyzer = A(); p.pinecone_retriever = PC(); p.neo4j_retriever = N(); p.llm = L()

    # A chat turn WITH a user_id must NOT write episodic memory (no per-turn ingest).
    with silent():
        list(p.run("I have a cough and chest pain", session_id="ep_s", user_id="u1"))
    R.check("no per-turn episodic write during /chat", captured == [])

    # Closing the session writes exactly ONE consolidated episode.
    with silent():
        status = p.end_session(user_id="u1", session_id="ep_s")
    R.check("end_session stores one episode", status.get("stored") is True and len(captured) == 1, str(status))
    digest = captured[0] if captured else ""
    R.check("digest consolidates the session", "cough" in digest and "chest_pain" in digest, digest[:80])

    # No user_id → nothing stored.
    with silent():
        st2 = p.end_session(user_id="", session_id="ep_s")
    R.check("end_session is a no-op without user_id", st2.get("stored") is False, str(st2))


# ── 9. NDJSON answer streaming (schema + validator + partial recovery) ────────
def test_answer_streaming():
    R.section("9. NDJSON answer streaming")
    import json as _json
    from graphrag.schemas.blocks import BLOCK_TYPES, AnswerResponse
    from graphrag.validators.answer_validator import validate_line, iter_blocks, blocks_to_text
    from graphrag.domain.messages import refusal_blocks, out_of_scope_blocks, emergency_blocks

    R.check("BLOCK_TYPES = the 7-type contract",
            len(BLOCK_TYPES) == 7 and "condition_list" in BLOCK_TYPES and "follow_up_questions" in BLOCK_TYPES)

    # validate_line: accept good, drop bad
    R.check("validate_line accepts a valid block",
            validate_line('{"type":"summary","data":{"text":"hi"}}').type == "summary")
    R.check("validate_line drops malformed JSON", validate_line("{not json") is None)
    R.check("validate_line drops empty line", validate_line("   ") is None)
    R.check("validate_line drops unknown type", validate_line('{"type":"foo","data":{}}') is None)
    R.check("validate_line drops empty required list",
            validate_line('{"type":"key_points","data":{"points":[]}}') is None)
    R.check("validate_line drops extra keys (extra=forbid)",
            validate_line('{"type":"summary","data":{"text":"x"},"z":1}') is None)

    # partial recovery: a malformed middle line is dropped; neighbours still stream
    toks = ['{"type":"summary",', '"data":{"text":"a"}}\n', 'GARBAGE LINE\n',
            '{"type":"next_steps","data":{"steps":["see a doctor"]}}']
    out = list(iter_blocks(iter(toks), terminal=False))
    R.check("malformed line dropped, valid lines before/after still stream",
            [b["type"] for b in out] == ["summary", "next_steps"], str([b["type"] for b in out]))
    R.check("trailing line (no final newline) is flushed", out[-1]["data"]["steps"] == ["see a doctor"])
    R.check("blocks_to_text renders streamed blocks", "see a doctor" in blocks_to_text(out))

    # terminal=True drops follow_up_questions
    fl = ('{"type":"follow_up_questions","data":{"questions":["q?"]}}\n'
          '{"type":"summary","data":{"text":"s"}}')
    R.check("terminal=True drops follow_up_questions",
            [b["type"] for b in iter_blocks(iter([fl]), terminal=True)] == ["summary"])
    R.check("terminal=False keeps follow_up_questions",
            "follow_up_questions" in [b["type"] for b in iter_blocks(iter([fl]), terminal=False)])

    # critical emergency sequence validates as warning+summary+condition_list+next_steps
    crit = "\n".join([
        '{"type":"warning","data":{"text":"Seek emergency care now.","severity":"critical"}}',
        '{"type":"summary","data":{"text":"why this is concerning"}}',
        '{"type":"condition_list","data":{"conditions":[{"name":"PE","likelihood":null,"description":"can sometimes be a clot"}]}}',
        '{"type":"next_steps","data":{"steps":["Call 911"]}}',
    ])
    R.check("critical → warning+summary+condition_list+next_steps",
            [b["type"] for b in iter_blocks(iter([crit]), terminal=True)]
            == ["warning", "summary", "condition_list", "next_steps"])

    # PRETTY-PRINTED / multi-line blocks must reassemble (depth-aware framing).
    # gemini-2.5-flash-lite splits array-bearing blocks across lines — the old
    # newline framing dropped them all.
    pretty = (
        '{"type":"summary","data":{"text":"Night-time breathing trouble has causes."}}\n'
        '{\n  "type": "condition_list",\n  "data": {\n    "conditions": [\n'
        '      {"name": "OSA", "likelihood": "high", "description": "apnea"},\n'
        '      {"name": "GERD", "likelihood": "medium", "description": "reflux"},\n'
        '      {"name": "Asthma", "likelihood": "medium", "description": "nocturnal"}\n'
        '    ]\n  }\n}\n'
        '{"type":"warning","data":{"text":"Seek care if you wake gasping.","severity":"caution"}}\n'
        '{\n  "type": "follow_up_questions",\n  "data": {"questions": ["Snore?", "Heartburn?"]}\n}\n'
        '{"type":"next_steps","data":{"steps":["Sleep diary","See a clinician"]}}'
    )
    # fed in awkward 5-char chunks to prove framing is stream-position-agnostic
    chunks = [pretty[i:i + 5] for i in range(0, len(pretty), 5)]
    pp = list(iter_blocks(iter(chunks), terminal=False))
    R.check("pretty-printed multi-line blocks reassemble (none dropped)",
            [b["type"] for b in pp]
            == ["summary", "condition_list", "warning", "follow_up_questions", "next_steps"],
            str([b["type"] for b in pp]))
    R.check("multi-line condition_list array intact",
            pp[1]["data"]["conditions"][0]["name"] == "OSA" and len(pp[1]["data"]["conditions"]) == 3)
    R.check("terminal=True drops a pretty-printed follow_up_questions",
            "follow_up_questions" not in [b["type"] for b in iter_blocks(iter(chunks), terminal=True)])

    # A brace-balanced but schema-invalid object is dropped; neighbours survive.
    mix = ('{"type":"summary","data":{"text":"ok"}}'
           '{"type":"warning","data":{"severity":"nope"}}'
           '{"type":"next_steps","data":{"steps":["go"]}}')
    R.check("brace-balanced malformed object dropped; neighbours survive",
            [b["type"] for b in iter_blocks(iter([mix]), terminal=False)] == ["summary", "next_steps"])

    # Structural unwrapping: a top-level array and a {"blocks":[...]} wrapper.
    arr = '[{"type":"summary","data":{"text":"a"}},{"type":"next_steps","data":{"steps":["x"]}}]'
    R.check("top-level array unpacked into element blocks",
            [b["type"] for b in iter_blocks(iter([arr]), terminal=False)] == ["summary", "next_steps"])
    R.check("stray {\"blocks\":[...]} wrapper unwrapped",
            [b["type"] for b in iter_blocks(iter(['{"blocks":[{"type":"summary","data":{"text":"w"}}]}']),
                                            terminal=False)] == ["summary"])

    # LAZY: first block reaches the client before the stream is exhausted
    pulled = {"n": 0}
    def lazy_tokens():
        for t in ['{"type":"summary","data":{"text":"first"}}\n',
                  '{"type":"next_steps","data":{"steps":["x"]}}']:
            pulled["n"] += 1
            yield t
    gen = iter_blocks(lazy_tokens(), terminal=False)
    first = next(gen)
    R.check("first block emitted before whole response is buffered",
            first["type"] == "summary" and pulled["n"] == 1, f"tokens pulled={pulled['n']}")

    # canned no-LLM paths are blocks, not strings — and they validate
    R.check("refusal_blocks → summary block", [b["type"] for b in refusal_blocks()] == ["summary"])
    R.check("out_of_scope_blocks → summary block", [b["type"] for b in out_of_scope_blocks()] == ["summary"])
    eb = emergency_blocks()
    R.check("emergency_blocks → warning(critical)+next_steps",
            [b["type"] for b in eb] == ["warning", "next_steps"] and eb[0]["data"]["severity"] == "critical")
    canned = refusal_blocks() + out_of_scope_blocks() + emergency_blocks()
    R.check("all canned blocks pass the schema",
            all(validate_line(_json.dumps(b)) is not None for b in canned))
    R.check("AnswerResponse wraps a block list (non-stream consumers)",
            len(AnswerResponse(blocks=canned).blocks) == len(canned))


# ── 10. Pipeline refuse path streams blocks ───────────────────────────────────
def test_pipeline_refuse():
    R.section("10. Pipeline refuse path (blocks)")
    from graphrag.domain.messages import refusal_blocks
    p, calls, flags = build_stub_pipeline(
        [analysis("out_of_context", relevance=0, needs_followup=False, action="refuse")])
    blocks = run_blocks(p, "what's the capital of France?", session_id="sc_refuse")
    R.check("refuse streams canned blocks (not a string)", blocks == refusal_blocks(), str(blocks))
    R.check("refuse skipped retrieval + LLM", flags["retrieved"] is False and not calls)


# ── 7. HTTP API wiring (best-effort) ──────────────────────────────────────────
def test_api_wiring():
    R.section("7. HTTP API wiring")
    try:
        import fastapi  # noqa: F401
    except Exception:
        R.skip("FastAPI app", "fastapi not installed in this interpreter")
        return
    try:
        from api import app
        paths = {getattr(r, "path", None) for r in app.routes}
        R.check("/health route registered", "/health" in paths)
        R.check("/chat route registered", "/chat" in paths)
        R.check("/session/end route registered", "/session/end" in paths)
    except Exception as e:
        R.check("app.main imports", False, repr(e))


def main() -> None:
    print("=" * 64)
    print("  OFFLINE SMOKE TEST — GraphRAG pulmonology assistant")
    print("  (Pinecone / Neo4j / Gemini stubbed — no network, no cost)")
    print("=" * 64)
    test_imports()
    test_domain()
    test_memory()
    test_entities()
    test_pipeline_scenarios()
    test_stage4_injection()
    test_episodic_session_end()
    test_answer_streaming()
    test_pipeline_refuse()
    test_api_wiring()
    print("\n" + "=" * 64)
    total = R.passed + R.failed
    print(f"  RESULT: {R.passed}/{total} checks passed, {R.failed} failed")
    print("=" * 64)
    sys.exit(1 if R.failed else 0)


if __name__ == "__main__":
    main()
