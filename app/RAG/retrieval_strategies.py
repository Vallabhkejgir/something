"""
retrieval_strategies.py — Multi-strategy retrieval engine.

Implements all retrieval strategies selected by the query analyser:

  dense          — FAISS semantic search
  sparse         — BM25 keyword search
  hybrid         — dense + sparse merged via Reciprocal Rank Fusion (default)
  multi_query    — run multiple query variants in parallel, merge with RRF
  hyde           — embed a hypothetical answer, then retrieve semantically
  parent_child   — retrieve children, expand to parent documents
  metadata_filter— pre-filter by element_type before vector search

All strategies are async and accept the same interface:
    docs = await strategy_fn(queries, store_manager, top_k)
"""

import asyncio
import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from app.services.llm_config import fast_llm, FAST_LLM_LIMITER, query_embeddings

logger = logging.getLogger(__name__)


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    *doc_lists: List[Document], k: int = 60, top_n: int = 8
) -> List[Document]:
    """
    Merge multiple ranked document lists using RRF.
    k=60 is the standard constant from the original RRF paper.
    """
    scores: dict[str, dict] = {}

    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list):
            key = doc.page_content.strip()
            if key not in scores:
                scores[key] = {"doc": doc, "score": 0.0}
            scores[key]["score"] += 1.0 / (rank + 1 + k)

    sorted_items = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_items[:top_n]]


# ── Individual Retrieval Strategies ──────────────────────────────────────────

async def retrieve_dense(query: str, store_manager, top_k: int = 10) -> List[Document]:
    """Semantic dense retrieval via FAISS."""
    if store_manager.vector_store is None:
        return []
    retriever = store_manager.vector_store.as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance for diversity
        search_kwargs={"k": top_k, "fetch_k": top_k * 2, "lambda_mult": 0.7},
    )
    try:
        return await retriever.ainvoke(query)
    except Exception as e:
        logger.warning("Dense retrieval error: %s", e)
        return []


async def retrieve_sparse(query: str, store_manager, top_k: int = 10) -> List[Document]:
    """BM25 sparse retrieval."""
    if store_manager.bm25_retriever is None:
        return []
    store_manager.bm25_retriever.k = top_k
    try:
        return await store_manager.bm25_retriever.ainvoke(query)
    except Exception as e:
        logger.warning("Sparse retrieval error: %s", e)
        return []


async def retrieve_hybrid(
    queries: List[str], store_manager, top_k: int = 10
) -> List[Document]:
    """
    Hybrid retrieval: run dense and sparse concurrently for the primary query,
    then fuse results via RRF.
    """
    primary = queries[0] if queries else ""
    dense_docs, sparse_docs = await asyncio.gather(
        retrieve_dense(primary, store_manager, top_k),
        retrieve_sparse(primary, store_manager, top_k),
    )
    fused = reciprocal_rank_fusion(dense_docs, sparse_docs, top_n=top_k)
    logger.debug("Hybrid: %d dense + %d sparse → %d fused", len(dense_docs), len(sparse_docs), len(fused))
    return fused


async def retrieve_multi_query(
    queries: List[str], store_manager, top_k: int = 10
) -> List[Document]:
    """
    Multi-query: run hybrid retrieval for each query variant in parallel,
    then fuse all result lists with RRF.
    """
    tasks = [retrieve_hybrid([q], store_manager, top_k) for q in queries]
    per_query_results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_lists = [r for r in per_query_results if isinstance(r, list)]
    if not valid_lists:
        return []

    fused = reciprocal_rank_fusion(*valid_lists, top_n=top_k)
    logger.debug(
        "MultiQuery: %d queries × ~%d docs → %d fused",
        len(queries), top_k, len(fused),
    )
    return fused


async def retrieve_hyde(
    query: str, store_manager, top_k: int = 10
) -> List[Document]:
    """
    Hypothetical Document Embedding (HyDE):
    1. LLM generates a hypothetical answer to the query.
    2. Embed the hypothetical answer.
    3. Use the embedding to search FAISS (semantic proximity to real docs).
    """
    if store_manager.vector_store is None:
        return []

    try:
        await FAST_LLM_LIMITER.acquire(200)
        hypo_prompt = (
            "Write a short, factual paragraph (3-5 sentences) that would directly "
            f"answer the following question. Be specific and use domain language.\n\nQuestion: {query}"
        )
        resp = await fast_llm.ainvoke([HumanMessage(content=hypo_prompt)])
        hypothetical_answer = resp.content.strip()
        logger.debug("HyDE hypothetical answer: %s...", hypothetical_answer[:80])
    except Exception as e:
        logger.warning("HyDE generation failed (%s) — falling back to dense.", e)
        return await retrieve_dense(query, store_manager, top_k)

    # Search using the hypothetical answer as the query
    try:
        retriever = store_manager.vector_store.as_retriever(
            search_kwargs={"k": top_k}
        )
        return await retriever.ainvoke(hypothetical_answer)
    except Exception as e:
        logger.warning("HyDE dense retrieval failed: %s", e)
        return []


async def retrieve_parent_child(
    queries: List[str], store_manager, top_k: int = 10
) -> List[Document]:
    """
    Parent-Child retrieval:
    1. Retrieve child chunks (small, precise).
    2. Map each child back to its parent (large, coherent context).
    3. Deduplicate parents.
    """
    child_docs = await retrieve_hybrid(queries, store_manager, top_k)

    seen_parent_ids: set[str] = set()
    expanded_docs: List[Document] = []

    for child in child_docs:
        parent_id = child.metadata.get("parent_chunk_id")
        if parent_id and parent_id not in seen_parent_ids:
            parent_doc = store_manager.get_parent(child)
            expanded_docs.append(parent_doc)
            seen_parent_ids.add(parent_id)
        elif not parent_id:
            # Standalone doc (e.g. image, table child without parent)
            expanded_docs.append(child)

    logger.debug(
        "ParentChild: %d children → %d unique parents",
        len(child_docs), len(expanded_docs),
    )
    return expanded_docs[:top_k]


async def retrieve_metadata_filtered(
    query: str,
    store_manager,
    element_types: List[str],
    top_k: int = 10,
) -> List[Document]:
    """
    Metadata-filtered retrieval: restrict search to specific element_types.
    Falls back to hybrid retrieval if filtered set is empty.
    """
    if store_manager.vector_store is None:
        return []

    # Filter the accumulated index docs by element_type
    filtered_docs = [
        d for d in store_manager._all_index_docs
        if d.metadata.get("element_type") in element_types
    ]

    if not filtered_docs:
        logger.debug("MetadataFilter: no docs of types %s — falling back to hybrid.", element_types)
        return await retrieve_hybrid([query], store_manager, top_k)

    # Build a temporary FAISS index over the filtered subset
    from langchain_community.vectorstores import FAISS as _FAISS
    from app.services.llm_config import embeddings as _embeddings
    try:
        temp_store = await asyncio.to_thread(_FAISS.from_documents, filtered_docs, _embeddings)
        retriever = temp_store.as_retriever(search_kwargs={"k": min(top_k, len(filtered_docs))})
        results = await retriever.ainvoke(query)
        logger.debug(
            "MetadataFilter: %d docs of types %s → %d results",
            len(filtered_docs), element_types, len(results),
        )
        return results
    except Exception as e:
        logger.warning("MetadataFilter retrieval failed (%s) — falling back to hybrid.", e)
        return await retrieve_hybrid([query], store_manager, top_k)


# ── Strategy Dispatcher ───────────────────────────────────────────────────────

async def execute_strategy(
    strategy: str,
    queries: List[str],
    store_manager,
    top_k: int = 10,
    element_types: List[str] = None,
) -> List[Document]:
    """
    Dispatch to the appropriate retrieval strategy.
    Always returns a (possibly empty) list of Documents.
    """
    primary = queries[0] if queries else ""

    if strategy == "dense":
        return await retrieve_dense(primary, store_manager, top_k)

    elif strategy == "sparse":
        return await retrieve_sparse(primary, store_manager, top_k)

    elif strategy == "multi_query":
        return await retrieve_multi_query(queries, store_manager, top_k)

    elif strategy == "hyde":
        return await retrieve_hyde(primary, store_manager, top_k)

    elif strategy == "parent_child":
        return await retrieve_parent_child(queries, store_manager, top_k)

    elif strategy == "metadata_filter":
        types = element_types or (["table"] if "table" in primary.lower() else ["image"])
        return await retrieve_metadata_filtered(primary, store_manager, types, top_k)

    else:  # default: hybrid
        return await retrieve_hybrid(queries, store_manager, top_k)
