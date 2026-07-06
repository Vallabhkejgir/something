"""
query_analyzer.py — LLM-based query analysis and retrieval strategy selection.

Uses the fast LLM to:
  1. Classify query intent (factual, tabular, visual, conceptual, procedural, comparative)
  2. Detect referenced content types (tables, images)
  3. Select the optimal retrieval strategy
  4. Generate query variants for multi-query expansion

Returns a QueryAnalysis Pydantic model consumed by the graph nodes.
"""

import logging
from typing import List

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.services.llm_config import fast_llm, FAST_LLM_LIMITER

logger = logging.getLogger(__name__)

# ── Output Schema ─────────────────────────────────────────────────────────────

VALID_STRATEGIES = {
    "dense",        # Pure semantic / embedding search
    "sparse",       # BM25 keyword search
    "hybrid",       # Dense + sparse with RRF (default)
    "multi_query",  # LLM generates multiple query variants
    "hyde",         # Hypothetical Document Embeddings
    "parent_child", # Retrieve children, return parents
    "metadata_filter",  # Filter by element_type first
}

VALID_INTENTS = {
    "factual",      # "What is X?" — straightforward lookup
    "tabular",      # "Show me the table of Y" / "compare A and B"
    "visual",       # "What does the image/chart show?"
    "conceptual",   # "Explain why..." — needs synthesis
    "procedural",   # "How do I..." — steps/sequence
    "comparative",  # "What's the difference between..."
    "conversational",  # Greetings / chitchat
}


class QueryAnalysis(BaseModel):
    intent: str = Field(
        description=f"Query intent. One of: {', '.join(sorted(VALID_INTENTS))}"
    )
    references_tables: bool = Field(
        description="True if query explicitly asks about tabular data."
    )
    references_images: bool = Field(
        description="True if query explicitly asks about images, charts, diagrams."
    )
    strategy: str = Field(
        description=f"Best retrieval strategy. One of: {', '.join(sorted(VALID_STRATEGIES))}"
    )
    search_queries: List[str] = Field(
        description="1-3 search query variants to use for retrieval. "
                    "Include the original and any rephrased versions."
    )
    reasoning: str = Field(
        description="One sentence explaining why this strategy was chosen."
    )


# ── Prompt ────────────────────────────────────────────────────────────────────

_ANALYSIS_PROMPT = """
You are a query analysis expert for a RAG (Retrieval-Augmented Generation) system.

Analyse the user's query and select the best retrieval strategy.

RETRIEVAL STRATEGIES:
- dense: semantic embedding search. Best for conceptual/meaning-based queries.
- sparse: BM25 keyword search. Best for exact names, codes, acronyms.
- hybrid: dense + sparse combined (default, good for most queries).
- multi_query: generate 2-3 query variants and merge results. Best for ambiguous/complex queries.
- hyde: generate a hypothetical answer and embed it for retrieval. Best for abstract/conceptual queries.
- parent_child: retrieve specific chunks then expand to broader context. Best when detail + context both needed.
- metadata_filter: restrict search to specific element types (tables/images). Best when user explicitly asks for a table or image.

USER QUERY: {query}

Respond in valid JSON matching this schema exactly:
{{
  "intent": "<one of: factual|tabular|visual|conceptual|procedural|comparative|conversational>",
  "references_tables": <true|false>,
  "references_images": <true|false>,
  "strategy": "<one of: dense|sparse|hybrid|multi_query|hyde|parent_child|metadata_filter>",
  "search_queries": ["<query1>", "<optional query2>", "<optional query3>"],
  "reasoning": "<one sentence>"
}}
""".strip()


# ── Analyser ──────────────────────────────────────────────────────────────────

async def analyse_query(query: str) -> QueryAnalysis:
    """
    Analyse the user query and return a QueryAnalysis.

    Falls back to a safe default (hybrid strategy) if the LLM call fails
    or returns malformed JSON — ensuring the pipeline never breaks.
    """
    default = QueryAnalysis(
        intent="factual",
        references_tables=False,
        references_images=False,
        strategy="hybrid",
        search_queries=[query],
        reasoning="Defaulting to hybrid retrieval (analysis unavailable).",
    )

    # Quick heuristic: very short or purely conversational queries skip LLM analysis
    if len(query.strip()) < 5:
        return default

    try:
        await FAST_LLM_LIMITER.acquire(150)

        prompt = _ANALYSIS_PROMPT.format(query=query)
        response = await fast_llm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        import json
        data = json.loads(raw)

        # Validate and sanitise fields
        intent   = data.get("intent", "factual")
        strategy = data.get("strategy", "hybrid")
        if intent not in VALID_INTENTS:
            intent = "factual"
        if strategy not in VALID_STRATEGIES:
            strategy = "hybrid"

        search_queries = data.get("search_queries", [query])
        if not isinstance(search_queries, list) or not search_queries:
            search_queries = [query]
        # Ensure the original query is always present
        if query not in search_queries:
            search_queries.insert(0, query)
        # Cap at 3 variants
        search_queries = search_queries[:3]

        return QueryAnalysis(
            intent=intent,
            references_tables=bool(data.get("references_tables", False)),
            references_images=bool(data.get("references_images", False)),
            strategy=strategy,
            search_queries=search_queries,
            reasoning=data.get("reasoning", ""),
        )

    except Exception as e:
        logger.warning("Query analysis failed (%s) — using default hybrid strategy.", e)
        return default
