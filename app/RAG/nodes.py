"""
nodes.py — All LangGraph node implementations for the advanced RAG pipeline.

Node registry:
  input_guard          — Validates and sanitises the incoming query
  analyse_query_node   — Classifies intent and selects retrieval strategy
  adaptive_retrieve    — Executes selected retrieval strategy
  relevance_grader     — Grades each doc for relevance (parallel)
  query_transformer    — Rewrites query if retrieval quality is poor
  generate_answer      — Generates the grounded answer
  faithfulness_checker — Verifies answer is supported by context
  output_guard         — Validates and sanitises the generated answer
  fallback_response    — Returns graceful "not enough info" message
  blocked_response     — Returns message when input guard blocks query
"""

import asyncio
import json
import logging
import os
import time
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from app.guardrails.input_guard import check_input
from app.guardrails.output_guard import check_output
from app.RAG.query_rewriting import rewrite_basic, rewrite_step_back, rewrite_sub_queries
from app.RAG.prompts import (
    generation_prompt,
    relevance_grading_prompt,
    faithfulness_check_prompt,
    query_transform_prompt,
)
from app.RAG.query_analyzer import analyse_query
from app.RAG.retrieval_strategies import execute_strategy
from app.RAG.reranker import rerank
from app.services import storage as _storage
from app.services.llm_config import (
    gen_llm, fast_llm,
    GEN_LLM_LIMITER, FAST_LLM_LIMITER,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES    = int(os.getenv("MAX_RETRY_COUNT", "2"))
_ENABLE_RERANK  = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
_RERANK_TOP_N   = int(os.getenv("RERANK_TOP_N", "5"))
_RETRIEVAL_K    = int(os.getenv("RETRIEVAL_TOP_K", "10"))
_FAITH_THRESHOLD = 0.6  # Below this → retry generation


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_context(docs: List[Document]) -> str:
    """Format docs into a numbered context block for the generation prompt."""
    blocks = []
    for i, doc in enumerate(docs, 1):
        el_type = doc.metadata.get("element_type", "text")
        source  = doc.metadata.get("source") or doc.metadata.get("title") or "page"
        img_url = doc.metadata.get("image_url", "")

        header = f"[{i}] ({el_type}) Source: {source}"
        if img_url:
            header += f" | Image URL: {img_url}"

        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def _docs_to_dicts(docs: List[Document]) -> List[dict]:
    return [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]


def _dicts_to_docs(dicts: List[dict]) -> List[Document]:
    return [Document(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in dicts]


def _timed(state: dict, node_name: str, start: float) -> dict:
    """Append node timing to the state's node_timings dict."""
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    timings = dict(state.get("node_timings", {}))
    timings[node_name] = elapsed_ms
    return timings


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def input_guard(state: dict) -> dict:
    """
    Input validation node.
    Checks for prompt injection, query length, PII, and relevance.
    Sets state['blocked'] = True if the query should not proceed.
    """
    t0 = time.perf_counter()
    question = state.get("question", "")
    trace = list(state.get("retrieval_trace", []))
    flags = list(state.get("guardrail_flags", []))

    result = await check_input(question)

    if result["blocked"]:
        flags.extend(result["flags"])
        trace.append(f"INPUT_GUARD: BLOCKED — {', '.join(result['flags'])}")
        logger.warning("Input guard blocked query: %s", result["flags"])
        return {
            "blocked": True,
            "guardrail_flags": flags,
            "retrieval_trace": trace,
            "answer": result.get("message", "This query cannot be processed."),
            "node_timings": _timed(state, "input_guard", t0),
        }

    trace.append("INPUT_GUARD: Passed.")
    return {
        "blocked": False,
        "guardrail_flags": flags,
        "retrieval_trace": trace,
        "node_timings": _timed(state, "input_guard", t0),
    }


async def analyse_query_node(state: dict) -> dict:
    """Query analysis — classifies intent and selects retrieval strategy."""
    t0 = time.perf_counter()
    question = state.get("question", "")
    trace = list(state.get("retrieval_trace", []))

    analysis = await analyse_query(question)

    trace.append(
        f"QUERY_ANALYSIS: intent={analysis.intent} | strategy={analysis.strategy} | "
        f"queries={analysis.search_queries}"
    )
    logger.info("Query analysis: %s", analysis.model_dump())

    return {
        "query_analysis": analysis.model_dump(),
        "retrieval_strategy": analysis.strategy,
        "search_queries": analysis.search_queries,
        "retrieval_trace": trace,
        "node_timings": _timed(state, "analyse_query", t0),
    }


async def adaptive_retrieve(state: dict) -> dict:
    """
    Adaptive retrieval node.
    Executes the strategy selected by the query analyser.
    Optionally re-ranks results before grading.
    """
    t0 = time.perf_counter()
    question  = state.get("question", "")
    strategy  = state.get("retrieval_strategy", "hybrid")
    queries   = state.get("search_queries", [question])
    trace     = list(state.get("retrieval_trace", []))
    analysis  = state.get("query_analysis", {})

    # Determine element type filter for metadata_filter strategy
    element_types = []
    if analysis.get("references_tables"):
        element_types.append("table")
    if analysis.get("references_images"):
        element_types.append("image")

    docs = await execute_strategy(
        strategy=strategy,
        queries=queries,
        store_manager=_storage.store_manager,
        top_k=_RETRIEVAL_K,
        element_types=element_types or None,
    )

    trace.append(f"RETRIEVE ({strategy.upper()}): {len(docs)} documents retrieved.")

    # Optional re-ranking
    rerank_scores = []
    if _ENABLE_RERANK and docs:
        docs, rerank_scores = await rerank(question, docs, top_n=_RERANK_TOP_N)
        trace.append(f"RERANK: Top {len(docs)} documents after re-ranking.")

    return {
        "retrieved_documents": _docs_to_dicts(docs),
        "retrieval_trace": trace,
        "node_timings": _timed(state, "adaptive_retrieve", t0),
    }


async def relevance_grader(state: dict) -> dict:
    """
    Relevance grading node.
    Grades all retrieved docs concurrently using the fast LLM.
    Filters out irrelevant docs to improve precision.
    """
    t0 = time.perf_counter()
    question  = state.get("question", "")
    raw_docs  = state.get("retrieved_documents", [])
    trace     = list(state.get("retrieval_trace", []))
    docs      = _dicts_to_docs(raw_docs)

    if not docs:
        trace.append("RELEVANCE_GRADER: No documents to grade.")
        return {
            "graded_documents": [],
            "context": "",
            "retrieval_trace": trace,
            "node_timings": _timed(state, "relevance_grader", t0),
        }

    async def _grade_one(doc: Document) -> tuple[Document, bool]:
        try:
            await FAST_LLM_LIMITER.acquire(100)
            chain = relevance_grading_prompt | fast_llm | StrOutputParser()
            raw = await chain.ainvoke({
                "query": question,
                "document": doc.page_content[:600],
            })
            # Strip code fences
            if raw.strip().startswith("```"):
                raw = raw.strip().split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw.strip())
            relevant = bool(result.get("relevant", True))
        except Exception as e:
            logger.warning("Grading failed for doc (%s) — keeping it.", e)
            relevant = True  # Default: keep on failure
        return doc, relevant

    # Grade all docs concurrently
    grade_results = await asyncio.gather(*[_grade_one(d) for d in docs])

    graded = [doc for doc, rel in grade_results if rel]
    rejected = len(docs) - len(graded)

    trace.append(
        f"RELEVANCE_GRADER: {len(graded)}/{len(docs)} docs passed "
        f"({rejected} filtered as irrelevant)."
    )

    context = _format_context(graded) if graded else ""

    return {
        "graded_documents": _docs_to_dicts(graded),
        "context": context,
        "retrieval_trace": trace,
        "node_timings": _timed(state, "relevance_grader", t0),
    }


async def query_transformer(state: dict) -> dict:
    """
    Query transformation node (retry path).
    Rewrites the query when retrieval quality was insufficient using dynamic strategies.
    """
    t0 = time.perf_counter()
    question    = state.get("question", "")
    original    = state.get("original_question") or question
    retry_count = state.get("retry_count", 0)
    trace       = list(state.get("retrieval_trace", []))

    # Determine strategy based on retry count
    if retry_count == 0:
        strategy_name = "step-back"
        new_queries = await rewrite_step_back(original)
        next_retrieval_strategy = "hybrid"
    elif retry_count == 1:
        strategy_name = "sub-queries"
        new_queries = await rewrite_sub_queries(original)
        next_retrieval_strategy = "multi_query"
    else:
        strategy_name = "basic"
        new_queries = await rewrite_basic(original)
        next_retrieval_strategy = "hybrid"

    trace.append(f"QUERY_TRANSFORM ({strategy_name}): '{original}' → {new_queries}")
    logger.info("Query transformed (%s): %s → %s", strategy_name, original, new_queries)

    return {
        "original_question": original,  # Preserve the original
        "search_queries": new_queries,
        "retrieval_strategy": next_retrieval_strategy,
        "retry_count": retry_count + 1,
        "retrieval_trace": trace,
        "node_timings": _timed(state, "query_transformer", t0),
    }


async def generate_answer(state: dict) -> dict:
    """
    Answer generation node.
    Uses the generation LLM with the graded context.
    """
    t0 = time.perf_counter()
    question = state.get("question", "")
    context  = state.get("context", "")
    trace    = list(state.get("retrieval_trace", []))

    if not context:
        trace.append("GENERATE: No context available — returning fallback message.")
        return {
            "answer": "I don't have enough information in the retrieved context to answer this question.",
            "retrieval_trace": trace,
            "node_timings": _timed(state, "generate_answer", t0),
        }

    token_est = (len(question) + len(context)) // 4
    await GEN_LLM_LIMITER.acquire(token_est)

    chain = generation_prompt | gen_llm | StrOutputParser()
    answer = await chain.ainvoke({"context": context, "question": question})

    trace.append(f"GENERATE: Answer produced ({len(answer)} chars).")
    return {
        "answer": answer,
        "retrieval_trace": trace,
        "node_timings": _timed(state, "generate_answer", t0),
    }


async def faithfulness_checker(state: dict) -> dict:
    """
    Faithfulness verification node.
    Checks if the generated answer is grounded in the retrieved context.
    Scores 0.0 (fully hallucinated) → 1.0 (fully grounded).
    """
    t0 = time.perf_counter()
    context = state.get("context", "")
    answer  = state.get("answer", "")
    trace   = list(state.get("retrieval_trace", []))

    if not context or not answer:
        return {
            "faithfulness_score": 1.0,
            "retrieval_trace": trace,
            "node_timings": _timed(state, "faithfulness_checker", t0),
        }

    # Skip faithfulness check if answer is the canonical "not enough info" reply
    if "don't have enough information" in answer.lower():
        return {
            "faithfulness_score": 1.0,
            "retrieval_trace": trace,
            "node_timings": _timed(state, "faithfulness_checker", t0),
        }

    try:
        await FAST_LLM_LIMITER.acquire(300)
        chain = faithfulness_check_prompt | fast_llm | StrOutputParser()
        raw = await chain.ainvoke({
            "context": context[:4000],
            "answer": answer[:2000],
        })

        # Strip code fences
        if raw.strip().startswith("```"):
            raw = raw.strip().split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw.strip())
        score   = float(result.get("faithfulness_score", 1.0))
        verdict = result.get("verdict", "faithful")
        unsupported = result.get("unsupported_claims", [])

        trace.append(
            f"FAITHFULNESS: score={score:.2f} | verdict={verdict} | "
            f"unsupported claims: {unsupported[:2]}"
        )
        logger.info("Faithfulness: %.2f (%s)", score, verdict)

        return {
            "faithfulness_score": score,
            "retrieval_trace": trace,
            "node_timings": _timed(state, "faithfulness_checker", t0),
        }
    except Exception as e:
        logger.warning("Faithfulness check failed (%s) — assuming faithful.", e)
        return {
            "faithfulness_score": 1.0,
            "retrieval_trace": trace,
            "node_timings": _timed(state, "faithfulness_checker", t0),
        }


async def output_guard(state: dict) -> dict:
    """
    Output validation node.
    Sanitises the answer and adds low-confidence disclaimers when needed.
    """
    t0 = time.perf_counter()
    answer      = state.get("answer", "")
    faith_score = state.get("faithfulness_score", 1.0)
    trace       = list(state.get("retrieval_trace", []))
    flags       = list(state.get("guardrail_flags", []))

    result = await check_output(answer, faith_score)

    if result.get("flags"):
        flags.extend(result["flags"])

    final_answer = result.get("answer", answer)
    trace.append(f"OUTPUT_GUARD: Passed. flags={result.get('flags', [])}")

    return {
        "answer": final_answer,
        "guardrail_flags": flags,
        "retrieval_trace": trace,
        "node_timings": _timed(state, "output_guard", t0),
    }


async def fallback_response(state: dict) -> dict:
    """Graceful fallback when retrieval quality is persistently poor."""
    trace = list(state.get("retrieval_trace", []))
    trace.append("FALLBACK: Returning graceful no-information response.")
    return {
        "answer": (
            "I wasn't able to find sufficient relevant information on this page "
            "to answer your question. Please try rephrasing or check that the "
            "page has been indexed correctly."
        ),
        "faithfulness_score": 1.0,
        "retrieval_trace": trace,
    }


async def blocked_response(state: dict) -> dict:
    """Response node when the input guard has blocked the query."""
    # The answer was already set by input_guard; just pass through
    return {
        "answer": state.get(
            "answer",
            "This query could not be processed due to content policy restrictions.",
        )
    }


# ── Edge Condition Functions ──────────────────────────────────────────────────

def route_after_input_guard(state: dict) -> str:
    """Route to blocked_response or analyse_query_node."""
    return "blocked" if state.get("blocked") else "analyse"


def route_after_grading(state: dict) -> str:
    """
    If enough relevant docs found → generate.
    If not enough and retries remain → transform query.
    If exhausted retries → fallback.
    """
    graded      = state.get("graded_documents", [])
    retry_count = state.get("retry_count", 0)

    if len(graded) >= 1:
        return "generate"
    if retry_count < _MAX_RETRIES:
        return "transform"
    return "fallback"


def route_after_faithfulness(state: dict) -> str:
    """
    If answer is faithful → output guard.
    If score is low and retries remain → regenerate.
    Otherwise → output guard (best we can do).
    """
    score       = state.get("faithfulness_score", 1.0)
    retry_count = state.get("retry_count", 0)

    if score >= _FAITH_THRESHOLD:
        return "output_guard"
    if retry_count < _MAX_RETRIES:
        return "regenerate"
    return "output_guard"
