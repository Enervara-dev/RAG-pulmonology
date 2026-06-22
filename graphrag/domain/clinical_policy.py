"""
graphrag.domain.clinical_policy
─────────────────────────────────
Structured clinical decision policy for the triage + answer layers — the
guideline-aligned rules that shape ranking, urgency, questioning, escalation,
and patient-facing tone.

⭐ EDIT FOR A NEW SPECIALTY/USE CASE. This is where triage behaviour is tuned:
which symptoms carry the most weight, which red flags force escalation, how many
clarifying questions are allowed, and the prose policies woven into the answer
prompt. The code (pipeline + answer_prompt) reads from here; nothing clinical is
hardcoded in the logic.
"""

from __future__ import annotations

import re

# ── Follow-up questioning budget ──────────────────────────────────────────────
# The triage layer may ask up to this many clarifying questions in one turn when
# severity or ambiguity warrants it (was effectively 1 before).
MAX_FOLLOWUP_QUESTIONS: int = 3

# ── Diagnostic loop termination ───────────────────────────────────────────────
# Terminal state for the diagnostic process. The session already carries a turn
# counter (SessionMemory.turn_count / WorkingMemory.turn_count); once the user
# has taken more than MAX_DIAGNOSTIC_TURNS turns — OR the gatekeeper stops needing
# follow-ups — the pipeline forces the intent to ASSESSMENT_READY so Stage 4
# concludes with a final assessment instead of looping on more questions.
MAX_DIAGNOSTIC_TURNS: int = 2
ASSESSMENT_READY_INTENT: str = "assessment_ready"

# Appended to the answer system prompt (Stage 4) when the diagnostic process is
# terminal — exact wording per the loop-prevention contract.
ASSESSMENT_READY_INSTRUCTION: str = (
    "CRITICAL INSTRUCTION: You have collected enough symptoms. Do NOT ask any "
    "further follow-up questions. Provide your final assessment and recommendations "
    "strictly based on the provided context."
)

# Appended when routing falls to NO_RETRIEVAL during a medical interaction — the
# model must wrap up from memory instead of defaulting to open-ended chat.
NO_RETRIEVAL_CONCLUDE_INSTRUCTION: str = (
    "CRITICAL INSTRUCTION: No new clinical information is being retrieved. "
    "Summarize the findings already gathered in this conversation, give your best "
    "assessment and clear next-step recommendations, and conclude — do NOT ask "
    "further follow-up questions or prolong the interaction."
)


def closure_directive(
    *,
    intent: str,
    needs_followup: bool,
    memory_only: bool,
    has_findings: bool,
) -> str | None:
    """
    Resolve the terminal/closure constraint to append at Stage 4, or None.

    - NO_RETRIEVAL during a medical interaction → conclude from memory.
    - assessment_ready (or the gatekeeper needing no more follow-ups) → final
      assessment, no further questions.
    Gated on `has_findings` so greetings / non-clinical turns are never forced
    to "conclude".
    """
    if not has_findings:
        return None
    if memory_only:
        return NO_RETRIEVAL_CONCLUDE_INSTRUCTION
    if intent == ASSESSMENT_READY_INTENT or not needs_followup:
        return ASSESSMENT_READY_INSTRUCTION
    return None

# ── High-signal symptoms (drive ranking + urgency) ────────────────────────────
# Prose, for prompt injection. Mirrors the canonical risk keys in the memory
# layer (session_memory/domain/risk_rules.py) but is phrased for the LLM.
# NOTE: these features escalate risk only when DESCRIBED AS PRESENT and severe —
# not when merely named, mild, historical, or hypothetical ("could this be…").
HIGH_SIGNAL_SYMPTOMS_TEXT = (
    "chest pain, coughing up blood (haemoptysis), wheeze, smoking history, signs of "
    "low oxygen (bluish lips/fingertips, marked breathlessness at rest), severe "
    "weakness, fast breathing (tachypnea), and persistent or high fever"
)

# ── Emergency red flags (respiratory / cardiopulmonary) ───────────────────────
# Prose list for the gatekeeper emergency section.
RED_FLAGS_TEXT = (
    "severe shortness of breath or breathlessness at rest; bluish/grey lips, face, "
    "or fingertips (cyanosis); too breathless to speak in full sentences; new "
    "confusion or drowsiness; persistent or crushing chest pain; coughing up blood; "
    "fainting or loss of consciousness (syncope); or signs of dangerously low oxygen"
)

# Deterministic backstop — STRONG, present-tense red-flag phrases. The pipeline
# escalates to the emergency message when any of these match the user's message,
# even if the LLM gatekeeper missed it. Patterns are intentionally conservative
# (they require explicit severity) so ordinary complaints like "can't breathe
# properly" or a past "chest pain last week" do NOT trip them.
RED_FLAG_PATTERNS: dict[str, re.Pattern[str]] = {
    "severe_breathlessness": re.compile(
        r"\b(can'?t|cannot|unable to)\s+breathe?\s+at all\b"
        r"|\bstruggl\w*\s+to\s+breathe?\b"
        r"|\bgasping\s+(for\s+)?(air|breath)\b"
        r"|\bfighting\s+(for|to)\s+breathe?\b"
        r"|\bsuffocat\w*"
        r"|\bsever(e|ely)\s+(short(ness)?\s+of\s+breath|breathless|difficulty\s+breathing|dyspn\w*)",
        re.IGNORECASE,
    ),
    "cyanosis": re.compile(
        r"\b(blue|bluish|grey|gray|purple)\s+(lips?|fingers?|fingertips?|face|skin|nails?)\b"
        r"|\blips?\s+(are\s+)?(turning\s+)?(blue|bluish|grey|gray|purple)\b",
        re.IGNORECASE,
    ),
    "cannot_speak_full_sentences": re.compile(
        r"\b(can'?t|cannot|too\s+breathless\s+to)\s+(speak|talk|finish)\b.*\b(sentence|sentences|words)\b"
        r"|\bonly\s+(say|speak)\s+(a\s+)?few\s+words\b"
        r"|\bcan'?t\s+(complete|finish)\s+(a\s+)?sentence",
        re.IGNORECASE,
    ),
    "confusion": re.compile(
        r"\b(confused|confusion|disorient\w+|not\s+making\s+sense|very\s+drowsy|hard\s+to\s+wake)\b",
        re.IGNORECASE,
    ),
    "persistent_chest_pain": re.compile(
        r"\b(persistent|constant|severe|crushing|tight)\b.{0,20}\bchest\s+pain\b"
        r"|\bchest\s+pain\b.{0,30}\b(won'?t|doesn'?t|wont)\s+(go\s+away|stop|ease)\b"
        r"|\bchest\s+pain\b.{0,20}\bfor\s+(hours|the\s+last\s+hour)\b"
        r"|\bcrushing\b.{0,15}\bchest\b",
        re.IGNORECASE,
    ),
    "haemoptysis": re.compile(
        r"\bcough(ing)?\s+up\s+blood\b"
        r"|\bblood\s+(in|when|while)\b.{0,20}\b(cough|sputum|phlegm|mucus)\b"
        r"|\b(haemoptysis|hemoptysis)\b"
        r"|\bspitting\s+blood\b",
        re.IGNORECASE,
    ),
    "syncope": re.compile(
        r"\b(faint(ed|ing)?|passed\s+out|black(ed)?\s+out|collaps(e|ed|ing)|lost\s+consciousness)\b",
        re.IGNORECASE,
    ),
    "oxygen_distress": re.compile(
        r"\boxygen\b.{0,15}\b(low|drop|dropping|falling|below)\b"
        r"|\b(o2|spo2|sats?|saturation)\b.{0,12}\b(low|drop\w*|falling|\d{1,2}\s*%)\b"
        r"|\blow\s+oxygen\b",
        re.IGNORECASE,
    ),
}


def detect_red_flags(text: str) -> list[str]:
    """Return the names of any emergency red flags present in `text`."""
    if not text:
        return []
    return [name for name, pat in RED_FLAG_PATTERNS.items() if pat.search(text)]


# ── Answer-layer policy blocks (woven into the answer system prompt) ───────────

DIFFERENTIAL_POLICY = f"""CLINICAL REASONING & DIFFERENTIAL
- Lead with the 1–3 MOST CLINICALLY LIKELY explanations for THIS patient, each with a \
one-line rationale tied to their specific features. Do not enumerate long lists of \
low-probability possibilities.
- Weight high-signal features heavily when ranking and when judging urgency: \
{HIGH_SIGNAL_SYMPTOMS_TEXT}.
- Do NOT surface rare or exotic conditions unless the symptoms strongly support them \
or the patient explicitly asks. Mention a "can't-miss" serious cause only when its \
red flags are plausibly present — and then say what would confirm or exclude it.
- Synthesise the retrieved context into coherent clinical reasoning (why these causes, \
what links the findings) — do not just summarise the source text."""

UNCERTAINTY_POLICY = """HANDLING UNCERTAINTY
- If the picture is uncertain but LOW risk: say so plainly, give sensible self-care and \
clear "see a clinician if…" criteria, and offer to narrow it down with one or two questions.
- If the picture is uncertain AND any severe/high-signal feature is present: do NOT \
reassure. Err toward caution — recommend timely or urgent assessment and state the \
specific red flags that mean "seek care now"."""

QUESTIONING_POLICY = f"""TRIAGE QUESTIONING
- Actively ask the clinically important triage questions when symptoms are ambiguous \
or potentially serious — do not default to "no questions". Good triage questions probe \
onset/duration, progression, severity, triggers/relievers, associated red-flag symptoms, \
and relevant history (e.g. smoking, known lung disease).
- Ask only what changes management. Ask at most {MAX_FOLLOWUP_QUESTIONS} questions, \
fewest possible, the most decision-relevant first. If you already have enough to answer \
safely, don't ask."""

SAFEGUARDS = """GENERATION SAFEGUARDS
- No FALSE REASSURANCE: never imply something is harmless when red flags or high-signal \
features are present.
- No PANIC: stay calm and measured; avoid alarming language for low-risk situations.
- No DIAGNOSTIC DUMPING: don't overwhelm with exhaustive lists, jargon, or encyclopedic \
detail. Keep it concise and readable.
- ALWAYS end with clear, concrete next steps (self-care, what to monitor, and exactly \
when/where to seek care)."""
