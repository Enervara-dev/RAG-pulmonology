"""
graphrag.validators.answer_validator
───────────────────────────────────────
Per-line validation + partial recovery for the answer stage's NDJSON stream —
the read side of the block contract in `graphrag/schemas/blocks.py`.

Mirrors the chunker's tolerant validator (`chunking/validators/schema_validator.py`):
the model streams text; we validate each completed JSON value independently and
forward only the ones that parse as a valid `Block`. A single malformed value is
logged and dropped — it never aborts the stream, and valid blocks before/after
still reach the client.

Framing is by STRUCTURE, not by newline. `iter_blocks` runs a depth-aware scanner
(tracking {}/[] nesting, ignoring braces inside JSON strings) so a block that the
model pretty-prints across several lines — common for `gemini-2.5-flash-lite` on
array-bearing blocks like condition_list / follow_up_questions / next_steps —
reassembles correctly instead of being shattered on '\\n'. A top-level array is
unpacked into its element blocks, and a stray ``{"blocks": [...]}`` wrapper is
unwrapped.

Key property: streaming stays LAZY. Each block is emitted the moment its
top-level value closes, so the first block reaches the client long before the
model finishes — the whole answer is never buffered.
"""

from __future__ import annotations

import json
from typing import Iterable, Iterator, Optional

from pydantic import TypeAdapter, ValidationError

from graphrag.schemas.blocks import Block
from graphrag.utils.logger import get_logger

logger = get_logger(__name__)

# Built once — validates an arbitrary dict against the discriminated Block union.
_BLOCK_ADAPTER: TypeAdapter = TypeAdapter(Block)


def validate_line(line: str):
    """
    Validate ONE NDJSON line as a single `Block`.

    Returns the parsed Block model on success, or ``None`` (after logging the
    reason) for an empty line, non-JSON, a non-object, or a schema violation.
    Never raises — bad lines are dropped, not fatal.
    """
    s = (line or "").strip()
    if not s:
        return None
    # A stray markdown fence (the prompt forbids them, but be defensive) is noise.
    if s.startswith("```"):
        logger.warning("Dropping markdown-fence answer line: %r", s[:80])
        return None
    try:
        data = json.loads(s)
    except json.JSONDecodeError as exc:
        logger.warning("Dropping non-JSON answer line: %s | %r", exc, s[:120])
        return None
    if not isinstance(data, dict):
        logger.warning("Dropping non-object answer line: %r", s[:120])
        return None
    try:
        return _BLOCK_ADAPTER.validate_python(data)
    except ValidationError as exc:
        first = exc.errors()[:1]
        logger.warning("Dropping invalid block: %s | %r", first, s[:120])
        return None


def _validate_obj(data) -> Optional["object"]:
    """Validate one parsed JSON value as a `Block`; log + drop on failure."""
    if not isinstance(data, dict):
        logger.warning("Dropping non-object block: %r", str(data)[:120])
        return None
    try:
        return _BLOCK_ADAPTER.validate_python(data)
    except ValidationError as exc:
        logger.warning("Dropping invalid block: %s | %r", exc.errors()[:1], str(data)[:120])
        return None


def _emit_value(data, *, terminal: bool) -> Iterator[dict]:
    """
    Yield validated block dicts from one parsed top-level JSON value.

    - object              → one block
    - array               → one block per element
    - {"blocks": [...]}   → unwrap, one block per element
    Drops a `follow_up_questions` block when the turn is terminal.
    """
    if isinstance(data, dict) and list(data.keys()) == ["blocks"] and isinstance(data["blocks"], list):
        items = data["blocks"]
    elif isinstance(data, list):
        items = data
    else:
        items = [data]

    for item in items:
        block = _validate_obj(item)
        if block is None:
            continue
        if terminal and block.type == "follow_up_questions":
            logger.info("Terminal turn — dropping follow_up_questions block.")
            continue
        yield block.model_dump()


def _frame_emit(raw: str, *, terminal: bool) -> Iterator[dict]:
    """Parse one sliced top-level value and emit its blocks (partial recovery)."""
    raw = raw.strip()
    if not raw:
        return
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Dropping non-JSON answer value: %s | %r", exc, raw[:120])
        return
    yield from _emit_value(data, terminal=terminal)


def iter_blocks(token_stream: Iterable[str], *, terminal: bool) -> Iterator[dict]:
    """
    Turn a stream of raw text tokens into a stream of validated block dicts.

    Frames by STRUCTURE, not by newline: a depth-aware scanner walks the buffered
    characters tracking {}/[] nesting (ignoring braces inside JSON strings, with
    " and \\ escape handling). When a top-level value closes (depth returns to 0)
    it is sliced out, parsed, validated, and emitted — so pretty-printed /
    multi-line blocks reassemble instead of being dropped. Whitespace, commas, and
    other junk BETWEEN top-level values are skipped.

    When ``terminal`` is True the diagnostic turn is concluding, so any
    ``follow_up_questions`` block is dropped — the assistant must not ask more
    questions once the assessment is final.
    """
    buffer = ""
    i = 0            # scan cursor into buffer
    start = None     # index where the current top-level value began
    depth = 0
    in_str = False
    esc = False

    for token in token_stream:
        if not token:
            continue
        buffer += token
        while i < len(buffer):
            ch = buffer[i]

            if start is None:
                # Between values: skip until the next value opens.
                if ch in "{[":
                    start = i
                    depth = 0
                    in_str = False
                    esc = False
                    # fall through to process this opening bracket for depth
                else:
                    i += 1
                    continue

            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch in "{[":
                    depth += 1
                elif ch in "}]":
                    depth -= 1
                    if depth == 0:
                        raw = buffer[start : i + 1]
                        yield from _frame_emit(raw, terminal=terminal)
                        buffer = buffer[i + 1 :]
                        i = 0
                        start = None
                        continue  # restart scan on the trimmed buffer

            i += 1

    # End of stream: a still-open value is truncated output — try it, then drop.
    if start is not None:
        yield from _frame_emit(buffer[start:], terminal=terminal)


def blocks_to_text(blocks: Iterable[dict]) -> str:
    """
    Flatten streamed block dicts into a plain-text answer for memory storage.

    The session/episodic memory layers store the assistant's reply as a string;
    this renders the structured blocks back to readable text (no markdown).
    """
    lines: list[str] = []
    for b in blocks:
        btype = b.get("type")
        data = b.get("data") or {}
        if btype == "summary":
            lines.append(data.get("text", ""))
        elif btype == "key_points":
            lines.extend(f"- {p}" for p in data.get("points", []))
        elif btype == "bullet_list":
            if data.get("title"):
                lines.append(str(data["title"]))
            lines.extend(f"- {i}" for i in data.get("items", []))
        elif btype == "follow_up_questions":
            lines.extend(f"- {q}" for q in data.get("questions", []))
        elif btype == "warning":
            lines.append(data.get("text", ""))
        elif btype == "next_steps":
            lines.extend(f"- {s}" for s in data.get("steps", []))
        elif btype == "condition_list":
            for c in data.get("conditions", []):
                seg = str(c.get("name", "")).strip()
                if c.get("likelihood"):
                    seg += f" ({c['likelihood']})"
                if c.get("description"):
                    seg += f": {c['description']}"
                if seg:
                    lines.append(seg)
    return "\n".join(x for x in lines if x).strip()


__all__ = ["validate_line", "iter_blocks", "blocks_to_text"]
