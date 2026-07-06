"""
states.py — Expanded graph state for the advanced RAG pipeline.

Every field is Optional or has a safe default so nodes can be
added/skipped without breaking the TypedDict schema.
"""

from typing import List, Optional
from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────────
    question: str                      # Original user question

    # ── Query Analysis ────────────────────────────────────────────────────────
    query_analysis: dict               # QueryAnalysis model serialised as dict
    retrieval_strategy: str            # Name of chosen strategy
    search_queries: List[str]          # Primary + variant queries

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieved_documents: List[dict]    # Raw retrieved docs (serialised)
    graded_documents: List[dict]       # Relevance-filtered docs
    context: str                       # Formatted context string for the LLM

    # ── Generation ────────────────────────────────────────────────────────────
    answer: str                        # Final answer string
    faithfulness_score: float          # 0.0 – 1.0 grounding score

    # ── Control Flow ──────────────────────────────────────────────────────────
    retry_count: int                   # Self-correction loop counter
    blocked: bool                      # True when input guard blocks the query

    # ── Observability ─────────────────────────────────────────────────────────
    retrieval_trace: List[str]         # Human-readable step-by-step trace
    guardrail_flags: List[str]         # Triggered guardrail names
    node_timings: dict                 # node_name -> elapsed_ms
