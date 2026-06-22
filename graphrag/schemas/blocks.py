"""
graphrag.schemas.blocks
─────────────────────────
Typed UI blocks the answer stage emits as NDJSON — one block object per line —
for the frontend to render incrementally.

Transport contract (locked): each line on the wire is exactly ONE JSON object of
the shape ``{"type": <block type>, "data": {...}}``. There is NO wrapping array
or object on the stream. See `graphrag/validators/answer_validator.py` for the
per-line validation + partial-recovery reader, mirrored on the chunker's
`chunking/validators/` pattern.

This module is the schema single-source-of-truth:
- one model per block type (+ `Condition`),
- `Block` = discriminated union on the `type` literal,
- `BLOCK_TYPES` = the tuple of valid type strings, derived from the models,
- `AnswerResponse` = `{blocks: [Block]}` for NON-streaming consumers only.

All models are strict (`extra="forbid"`); non-empty lists use `min_length=1`.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union, get_args

from pydantic import BaseModel, ConfigDict, Field

_STRICT = ConfigDict(extra="forbid")


class _Strict(BaseModel):
    model_config = _STRICT


# ── summary ───────────────────────────────────────────────────────────────────
class SummaryData(_Strict):
    text: str


class SummaryBlock(_Strict):
    type: Literal["summary"]
    data: SummaryData


# ── key_points ────────────────────────────────────────────────────────────────
class KeyPointsData(_Strict):
    points: list[str] = Field(..., min_length=1)


class KeyPointsBlock(_Strict):
    type: Literal["key_points"]
    data: KeyPointsData


# ── bullet_list ───────────────────────────────────────────────────────────────
class BulletListData(_Strict):
    title: Optional[str] = None
    items: list[str] = Field(..., min_length=1)


class BulletListBlock(_Strict):
    type: Literal["bullet_list"]
    data: BulletListData


# ── follow_up_questions ───────────────────────────────────────────────────────
class FollowUpQuestionsData(_Strict):
    questions: list[str] = Field(..., min_length=1)


class FollowUpQuestionsBlock(_Strict):
    type: Literal["follow_up_questions"]
    data: FollowUpQuestionsData


# ── warning ───────────────────────────────────────────────────────────────────
class WarningData(_Strict):
    text: str
    severity: Literal["info", "caution", "critical"]


class WarningBlock(_Strict):
    type: Literal["warning"]
    data: WarningData


# ── next_steps ────────────────────────────────────────────────────────────────
class NextStepsData(_Strict):
    steps: list[str] = Field(..., min_length=1)


class NextStepsBlock(_Strict):
    type: Literal["next_steps"]
    data: NextStepsData


# ── condition_list ────────────────────────────────────────────────────────────
class Condition(_Strict):
    name: str
    likelihood: Optional[str] = None
    description: Optional[str] = None


class ConditionListData(_Strict):
    conditions: list[Condition] = Field(..., min_length=1)


class ConditionListBlock(_Strict):
    type: Literal["condition_list"]
    data: ConditionListData


# ── the discriminated union ───────────────────────────────────────────────────
_BLOCK_MODELS = (
    SummaryBlock,
    KeyPointsBlock,
    BulletListBlock,
    FollowUpQuestionsBlock,
    WarningBlock,
    NextStepsBlock,
    ConditionListBlock,
)

Block = Annotated[
    Union[
        SummaryBlock,
        KeyPointsBlock,
        BulletListBlock,
        FollowUpQuestionsBlock,
        WarningBlock,
        NextStepsBlock,
        ConditionListBlock,
    ],
    Field(discriminator="type"),
]

# Single source of truth for the valid `type` strings — derived from the models
# so it can never drift from the union above.
BLOCK_TYPES: tuple[str, ...] = tuple(
    get_args(model.model_fields["type"].annotation)[0] for model in _BLOCK_MODELS
)


class AnswerResponse(_Strict):
    """A whole answer as a list of blocks — for NON-streaming consumers only.

    The wire format is NDJSON (one `Block` per line); this wrapper exists for
    callers that want the complete answer as a single validated object.
    """

    blocks: list[Block]


__all__ = [
    "SummaryBlock",
    "KeyPointsBlock",
    "BulletListBlock",
    "FollowUpQuestionsBlock",
    "WarningBlock",
    "NextStepsBlock",
    "ConditionListBlock",
    "Condition",
    "Block",
    "BLOCK_TYPES",
    "AnswerResponse",
]
