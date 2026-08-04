"""
reranker.py — LLM-based re-ranking of retrieved documents.

After initial retrieval, this module scores each (query, document) pair
using the fast LLM to improve precision before passing context to the generator.

Design:
  - Scores are obtained in a single batched LLM call to minimise latency.
  - Each document gets a relevance score 0-10.
  - Documents below a threshold are filtered out.
  - Falls back to original order if the LLM call fails.
"""

import asyncio
import json
import logging
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.services.llm_config import fast_llm, FAST_LLM_LIMITER

logger = logging.getLogger(__name__)

_RERANK_THRESHOLD = 3  # Documents scoring below this are filtered
_RERANK_PROMPT = """
You are a relevance judge for a retrieval system.

QUERY: {query}

Below are {n} retrieved document excerpts. Rate each one's relevance to the query
on a scale of 0-10, where:
  10 = directly answers the query
   7 = highly relevant, contains useful information
   4 = somewhat relevant, tangentially related
   0 = completely irrelevant

DOCUMENTS:
{docs_block}

Respond with a JSON array of objects in this exact format (no other text):
[{{"id": 0, "score": <int 0-10>, "reason": "<5 words max>"}}, ...]

Return an object for every document indexed 0 to {last_idx}.
""".strip()


async def rerank(
    query: str,
    docs: List[Document],
    top_n: int = 5,
    threshold: int = _RERANK_THRESHOLD,
) -> Tuple[List[Document], List[dict]]:
    """
    Re-rank `docs` by relevance to `query` using the fast LLM.

    Returns:
        (reranked_docs, scores_list)
        reranked_docs: top_n documents above threshold, sorted by score desc
        scores_list:   list of {id, score, reason} dicts for tracing
    """
    if not docs:
        return [], []

    if len(docs) == 1:
        return docs, [{"id": 0, "score": 10, "reason": "only doc"}]

    # Build the document block for the prompt
    docs_block_lines = []
    for i, doc in enumerate(docs):
        snippet = doc.page_content[:400].replace("\n", " ")
        source  = doc.metadata.get("source", "")
        el_type = doc.metadata.get("element_type", "text")
        docs_block_lines.append(f"[{i}] ({el_type}) {snippet}...")
    docs_block = "\n".join(docs_block_lines)

    prompt = _RERANK_PROMPT.format(
        query=query,
        n=len(docs),
        docs_block=docs_block,
        last_idx=len(docs) - 1,
    )

    try:
        # Estimate token cost: ~prompt_chars / 4 + some overhead
        token_est = len(prompt) // 4 + 100
        await FAST_LLM_LIMITER.acquire(token_est)

        response = await (fast_llm | StrOutputParser()).ainvoke([HumanMessage(content=prompt)])
        raw = response.strip()

        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        scores_list = json.loads(raw)

        # Build score map
        score_map: dict[int, int] = {}
        for item in scores_list:
            idx = int(item.get("id", -1))
            score = int(item.get("score", 0))
            if 0 <= idx < len(docs):
                score_map[idx] = score

        # Attach scores to docs and filter
        scored_pairs: List[Tuple[int, Document]] = []
        for i, doc in enumerate(docs):
            score = score_map.get(i, 0)
            doc.metadata["rerank_score"] = score
            if score >= threshold:
                scored_pairs.append((score, doc))

        # Sort descending by score
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        reranked = [doc for _, doc in scored_pairs[:top_n]]

        logger.debug(
            "Reranker: %d docs → %d above threshold (≥%d) → top %d returned",
            len(docs), len(scored_pairs), threshold, len(reranked),
        )
        return reranked, scores_list

    except Exception as e:
        logger.warning("Reranking failed (%s) — returning original order.", e)
        # Return original order, truncated to top_n
        return docs[:top_n], []
