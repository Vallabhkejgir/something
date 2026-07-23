"""
query_rewriting.py — Query rewriting and transformation strategies.

This module provides different strategies to rewrite user queries when initial
retrieval fails, enabling the adaptive RAG to self-correct effectively.
"""

import json
import logging
from typing import List

from langchain_core.output_parsers import StrOutputParser

from app.RAG.prompts import (
    query_transform_prompt,
    step_back_prompt,
    sub_query_prompt,
)
from app.services.llm_config import fast_llm, FAST_LLM_LIMITER

logger = logging.getLogger(__name__)


async def rewrite_basic(query: str) -> List[str]:
    """
    Basic rewrite: Asks the LLM to write a better search query.
    Returns a list with a single rewritten query.
    """
    try:
        await FAST_LLM_LIMITER.acquire(150)
        chain = query_transform_prompt | fast_llm | StrOutputParser()
        new_question = await chain.ainvoke({"question": query})
        new_question = new_question.strip().strip('"').strip("'")
        return [new_question]
    except Exception as e:
        logger.warning("Basic query transform failed (%s) — keeping original.", e)
        return [query]


async def rewrite_step_back(query: str) -> List[str]:
    """
    Step-back prompting: Asks the LLM to generate a broader, more generic
    high-level question to capture wider context.
    """
    try:
        await FAST_LLM_LIMITER.acquire(150)
        chain = step_back_prompt | fast_llm | StrOutputParser()
        new_question = await chain.ainvoke({"question": query})
        new_question = new_question.strip().strip('"').strip("'")
        return [new_question]
    except Exception as e:
        logger.warning("Step-back transform failed (%s) — keeping original.", e)
        return [query]


async def rewrite_sub_queries(query: str) -> List[str]:
    """
    Sub-query decomposition: Asks the LLM to break down a complex query
    into 2-3 simpler, self-contained sub-queries.
    """
    try:
        await FAST_LLM_LIMITER.acquire(250)
        chain = sub_query_prompt | fast_llm | StrOutputParser()
        raw = await chain.ainvoke({"question": query})
        
        # Strip markdown code fences if present
        if raw.strip().startswith("```"):
            raw = raw.strip().split("```")[1]
            if raw.lower().startswith("json"):
                raw = raw[4:]
                
        queries = json.loads(raw.strip())
        if not isinstance(queries, list) or not queries:
            return [query]
            
        return [str(q).strip() for q in queries[:3]]
    except Exception as e:
        logger.warning("Sub-query transform failed (%s) — keeping original.", e)
        return [query]
