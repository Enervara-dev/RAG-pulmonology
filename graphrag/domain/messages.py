"""
graphrag.domain.messages
──────────────────────────
User-facing canned responses emitted directly by the pipeline (no LLM call).

⭐ EDIT FOR A NEW SPECIALTY/USE CASE. These are returned when the gatekeeper
refuses a non-medical query, redirects a detected emergency, or restricts an
out-of-specialty question.

Transport is uniform NDJSON: even these no-LLM paths emit typed blocks (see
`graphrag/schemas/blocks.py`), not raw strings, so the frontend renders them the
same way as a streamed answer. The `*_blocks()` builders below are the canonical
form; the plain-string constants are kept for logging / fallback only.
"""

from __future__ import annotations

# ── Out-of-domain (non-medical) ───────────────────────────────────────────────
REFUSAL_TEXT: str = (
    "I can only answer healthcare-related questions. Please ask a medical question."
)

# ── Medical but below the pulmonology relevance threshold ─────────────────────
OUT_OF_SCOPE_TEXT: str = (
    "I'm focused on pulmonology and respiratory medicine, so I can't help with "
    "that one. Please ask about a lung or breathing-related concern (for example "
    "cough, breathlessness, asthma, COPD, or chest infections)."
)

# ── Emergency red-flag redirect ───────────────────────────────────────────────
EMERGENCY_WARNING_TEXT: str = (
    "Your symptoms may signal a serious, time-sensitive emergency. Please seek "
    "emergency care now."
)
EMERGENCY_STEPS: list[str] = [
    "Call your local emergency number (112 / 911) immediately, or go to the "
    "nearest emergency department.",
    "If you feel faint or are severely breathless, do not drive yourself — have "
    "someone take you or call an ambulance.",
    "Stay as calm and still as you can while help is on the way.",
]

# ── Backward-compatible plain-string forms (logging / last-resort fallback) ────
REFUSAL_MESSAGE: str = REFUSAL_TEXT
OUT_OF_SCOPE_MESSAGE: str = OUT_OF_SCOPE_TEXT
EMERGENCY_MESSAGE: str = (
    EMERGENCY_WARNING_TEXT + " Call emergency services (112 / 911) immediately or "
    "go to the nearest hospital."
)


# ── Block builders (the canonical NDJSON form) ────────────────────────────────
def refusal_blocks() -> list[dict]:
    """Non-medical query → a single summary block."""
    return [{"type": "summary", "data": {"text": REFUSAL_TEXT}}]


def out_of_scope_blocks() -> list[dict]:
    """Out-of-specialty medical query → a single summary block."""
    return [{"type": "summary", "data": {"text": OUT_OF_SCOPE_TEXT}}]


def emergency_blocks() -> list[dict]:
    """Detected emergency → a critical warning followed by next steps."""
    return [
        {"type": "warning", "data": {"text": EMERGENCY_WARNING_TEXT, "severity": "critical"}},
        {"type": "next_steps", "data": {"steps": list(EMERGENCY_STEPS)}},
    ]
