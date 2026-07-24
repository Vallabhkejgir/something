"""
tracing.py — Enhanced LangSmith tracing and structured logging utilities.

Provides:
  - trace_node()         : Context manager to time and log individual graph nodes
  - format_trace_report(): Format retrieval_trace into a human-readable string
  - log_final_state()    : Log a structured summary of the completed pipeline run
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import List

logger = logging.getLogger("rag.trace")


@asynccontextmanager
async def trace_node(node_name: str, state: dict):
    """
    Async context manager that times a node's execution and appends
    the result to the retrieval_trace.

    Usage:
        async with trace_node("my_node", state) as t:
            result = await do_work()
            t["trace"].append("MY_NODE: done")
            yield result
    """
    t0 = time.perf_counter()
    trace_entries: List[str] = []
    ctx = {"trace": trace_entries}
    try:
        yield ctx
    finally:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.debug("[%s] completed in %dms", node_name, elapsed_ms)


def format_trace_report(trace: List[str]) -> str:
    """
    Convert the retrieval_trace list into a readable step-by-step report.

    Example output:
        Step 1: INPUT_GUARD: Passed.
        Step 2: QUERY_ANALYSIS: intent=factual | strategy=hybrid
        ...
    """
    lines = ["=== RAG Pipeline Trace ==="]
    for i, entry in enumerate(trace, 1):
        lines.append(f"  Step {i:02d}: {entry}")
    lines.append("=" * 30)
    return "\n".join(lines)


def log_final_state(state: dict) -> None:
    """
    Emit a structured INFO log summarising a completed pipeline run.
    Useful for LangSmith custom metadata and local debugging.
    """
    question = state.get("question", "")[:80]
    strategy = state.get("retrieval_strategy", "unknown")
    faith    = state.get("faithfulness_score", 1.0)
    retries  = state.get("retry_count", 0)
    blocked  = state.get("blocked", False)
    flags    = state.get("guardrail_flags", [])
    timings  = state.get("node_timings", {})
    answer_len = len(state.get("answer", ""))

    logger.info(
        "PIPELINE COMPLETE | q='%s...' | strategy=%s | faith=%.2f | "
        "retries=%d | blocked=%s | flags=%s | answer_len=%d | timings=%s",
        question, strategy, faith, retries, blocked, flags, answer_len, timings,
    )

    # Emit trace at DEBUG level
    trace = state.get("retrieval_trace", [])
    if trace:
        logger.debug("\n%s", format_trace_report(trace))
