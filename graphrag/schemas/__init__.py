"""
graphrag.schemas
──────────────────
Typed answer-stage block schemas (the NDJSON UI contract). See `blocks.py`.
"""

from .blocks import (
    BLOCK_TYPES,
    AnswerResponse,
    Block,
    BulletListBlock,
    Condition,
    ConditionListBlock,
    FollowUpQuestionsBlock,
    KeyPointsBlock,
    NextStepsBlock,
    SummaryBlock,
    WarningBlock,
)

__all__ = [
    "Block",
    "BLOCK_TYPES",
    "AnswerResponse",
    "SummaryBlock",
    "KeyPointsBlock",
    "BulletListBlock",
    "FollowUpQuestionsBlock",
    "WarningBlock",
    "NextStepsBlock",
    "ConditionListBlock",
    "Condition",
]
