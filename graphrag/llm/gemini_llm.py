"""
Streaming Gemini-backed answer generator used by the GraphRAG pipeline.

STAGE 4 emits the answer as a stream of validated UI blocks (NDJSON). The model
is instructed (via `compose_system_prompt`'s OUTPUT CONTRACT) to produce one JSON
block per line; we keep TEXT streaming (no JSON mime/schema — that would force a
single object and break per-line streaming) and validate each line as it arrives
through `answer_validator.iter_blocks`. Valid blocks are yielded downstream as
plain dicts; malformed lines are logged and dropped without aborting the stream.
"""


from __future__ import annotations

import time
from typing import Iterator

from graphrag.config.settings import settings
from graphrag.domain.vocabulary import DEFAULT_ANSWER_GOAL
from graphrag.llm.gemini_client import (
    DEFAULT_MODEL,
    generate_stream,
    get_client,
)
from graphrag.utils.logger import get_logger
from graphrag.validators.answer_validator import iter_blocks

logger = get_logger(__name__)


class GeminiLLM:
    def __init__(self):
        # Fail fast if the API key is missing.
        get_client()
        self._model = settings.ANSWER_MODEL or DEFAULT_MODEL

    def generate_from_messages(self, messages: list[dict[str, str]]) -> Iterator[dict]:
        logger.info("[3/3] Sending memory-aware structured context to LLM Engine...")
        system_instruction, user_prompt = _split_messages(messages)
        return self._stream_blocks(
            system_instruction=system_instruction, user_prompt=user_prompt, terminal=False
        )

    def generate_response(
        self,
        query_text: str,
        vector_context: str,
        graph_context: str,
        memory_context: str = "",
        conversation_history: str = "",
        query_type: str = "unknown",
        goal: str = DEFAULT_ANSWER_GOAL,
        risk_level: str = "none",
        needs_followup: bool = True,
        memory_only: bool = False,
        has_findings: bool = False,
    ) -> Iterator[dict]:
        """
        Stream the Stage-4 answer as validated block dicts.

        The system prompt is built via the layered composer in the graphrag
        domain layer — `graphrag.domain.answer_prompt.compose_system_prompt` —
        so the assistant's persona/safety/grounding rules (and the NDJSON output
        contract) live in one retargetable place.

        Yields one validated block `dict` per emitted line, as it streams.
        """
        from graphrag.domain.answer_prompt import compose_system_prompt
        from graphrag.domain.clinical_policy import closure_directive

        logger.info("[3/3] Sending structured context to LLM Engine...")

        # ── Terminal/closure resolution ───────────────────────────────────────
        # If the diagnostic process is terminal (assessment_ready / no more
        # follow-ups) or this is a NO_RETRIEVAL medical interaction, the model
        # must conclude instead of looping on follow-up questions. `terminal`
        # threads into BOTH the prompt (don't emit follow_up_questions) and the
        # per-line validator (drop a follow_up_questions block if one slips out).
        constraint = closure_directive(
            intent=query_type,
            needs_followup=needs_followup,
            memory_only=memory_only,
            has_findings=has_findings,
        )
        terminal = constraint is not None

        has_name = "Patient name:" in memory_context
        system_prompt = compose_system_prompt(
            query_type=query_type,
            risk_level=risk_level,
            has_name=has_name,
            terminal=terminal,
        )
        if constraint:
            system_prompt = f"{system_prompt}\n\n{constraint}"
            logger.info(
                "🧭 Stage-4 closure constraint injected (intent=%s, needs_followup=%s, memory_only=%s).",
                query_type, needs_followup, memory_only,
            )

        user_prompt = f"""
USER QUESTION: {query_text}

=== STRUCTURED CLINICAL MEMORY ===
{memory_context}

=== RECENT CONVERSATION ===
{conversation_history}

=== RETRIEVED MEDICAL CONTEXT ===
{vector_context}

=== GRAPH RELATIONS ===
{graph_context}
"""

        return self._stream_blocks(
            system_instruction=system_prompt, user_prompt=user_prompt, terminal=terminal
        )

    def _stream_blocks(
        self, *, system_instruction: str | None, user_prompt: str, terminal: bool
    ) -> Iterator[dict]:
        """
        Run the raw token stream through the per-line block validator and yield
        validated block dicts. Transport/SDK errors are caught and logged; the
        per-line validator drops bad lines without failing the whole stream.
        """
        t_start = time.monotonic()
        logger.info("\n" + "=" * 80)
        logger.info("AI RESPONSE (NDJSON blocks)")
        logger.info("=" * 80)

        count = 0
        try:
            for block in iter_blocks(
                self._raw_tokens(system_instruction, user_prompt, t_start),
                terminal=terminal,
            ):
                count += 1
                yield block
        except Exception as e:  # transport/SDK error mid-stream
            logger.error(f"LLM Error: {e}")

        t_end = time.monotonic()
        logger.info(
            "⏱️  Stream complete in %.0fms (%d block(s))",
            (t_end - t_start) * 1000, count,
        )

    def _raw_tokens(
        self, system_instruction: str | None, user_prompt: str, t_start: float
    ) -> Iterator[str]:
        """Yield raw text chunks from Gemini, logging time-to-first-token."""
        first: float | None = None
        for piece in generate_stream(
            model=self._model,
            system_instruction=system_instruction,
            user_prompt=user_prompt,
        ):
            if first is None:
                first = time.monotonic()
                logger.info(
                    "⏱️  Time-to-first-token: %.0fms", (first - t_start) * 1000
                )
            yield piece


def _split_messages(messages: list[dict[str, str]]) -> tuple[str | None, str]:
    """
    Collapse an OpenAI-style messages array into (system_instruction, user_prompt)
    that Gemini's generate_content API expects.

    System messages are concatenated into the system_instruction. The remaining
    user/assistant turns are joined into a single user prompt with role prefixes
    so multi-turn context is preserved.
    """
    system_parts: list[str] = []
    body_parts: list[str] = []
    for msg in messages:
        role = (msg.get("role") or "").lower()
        content = msg.get("content") or ""
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            body_parts.append(f"Assistant: {content}")
        else:
            body_parts.append(f"User: {content}")
    system_instruction = "\n\n".join(p for p in system_parts if p).strip() or None
    user_prompt = "\n\n".join(p for p in body_parts if p).strip()
    return system_instruction, user_prompt
