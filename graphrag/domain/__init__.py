"""
graphrag.domain
────────────────
╔══════════════════════════════════════════════════════════════════════════════╗
║  THE PACKAGE TO EDIT FOR A NEW SPECIALTY / USE CASE (graphrag side).           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  All domain-specific knowledge for the retrieval + query-understanding layer   ║
║  lives here. The rest of graphrag (retrievers, routing logic, the QueryConfig  ║
║  dataclass, the Gemini client) is domain-agnostic and reads FROM this package. ║
║                                                                                ║
║    prompts.py        → gatekeeper system prompt                                ║
║    query_taxonomy.py → query types, per-type retrieval tuning, intent map      ║
║    vocabulary.py     → clinical-state keys, graph node label, default goal     ║
║    entity_rules.py   → entity post-processing heuristics (drug-pair boost)     ║
║    messages.py       → refusal / emergency canned responses                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from .answer_prompt import compose_system_prompt
from .clinical_policy import (
    MAX_FOLLOWUP_QUESTIONS,
    RED_FLAG_PATTERNS,
    detect_red_flags,
)
from .prompts import GATEKEEPER_SYSTEM_PROMPT
from .query_taxonomy import (
    DEFAULT_QUERY_TYPE,
    INTENT_TO_QUERYTYPE,
    QUERY_TUNING,
    QUERY_TYPES,
)
from .vocabulary import (
    CLINICAL_STATE_KEYS,
    DEFAULT_ANSWER_GOAL,
    GRAPH_NODE_LABEL,
    PINECONE_NAMESPACE,
    PULMONOLOGY_RELEVANCE_THRESHOLD,
)
from .entity_rules import DRUG_NAME_PATTERN, DRUG_NAME_STOPWORDS
from .messages import (
    EMERGENCY_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    REFUSAL_MESSAGE,
    emergency_blocks,
    out_of_scope_blocks,
    refusal_blocks,
)

__all__ = [
    "GATEKEEPER_SYSTEM_PROMPT",
    "compose_system_prompt",
    "MAX_FOLLOWUP_QUESTIONS",
    "RED_FLAG_PATTERNS",
    "detect_red_flags",
    "QUERY_TYPES",
    "QUERY_TUNING",
    "INTENT_TO_QUERYTYPE",
    "DEFAULT_QUERY_TYPE",
    "CLINICAL_STATE_KEYS",
    "GRAPH_NODE_LABEL",
    "PINECONE_NAMESPACE",
    "PULMONOLOGY_RELEVANCE_THRESHOLD",
    "DEFAULT_ANSWER_GOAL",
    "DRUG_NAME_PATTERN",
    "DRUG_NAME_STOPWORDS",
    "EMERGENCY_MESSAGE",
    "OUT_OF_SCOPE_MESSAGE",
    "REFUSAL_MESSAGE",
    "refusal_blocks",
    "out_of_scope_blocks",
    "emergency_blocks",
]
