import json
import asyncio
from typing import List
from langchain_core.output_parsers import StrOutputParser
from app.services.llm_config import llm, GEN_LLM_LIMITER
from app.RAG.reranker import rerank
from app.RAG.prompts import (
    prompt,
    rewrite_prompt,
    decompose_prompt,
    categorize_prompt,
    relevance_prompt,
    faithfulness_prompt,
)
from app.services import storage

_MAX_RETRIES = 3


def parse_json_bool_array(text: str, default_length: int) -> List[bool]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            result = []
            for item in parsed:
                if isinstance(item, bool):
                    result.append(item)
                elif isinstance(item, str):
                    result.append(item.lower() in ("true", "yes", "1"))
                else:
                    result.append(bool(item))
            if len(result) == default_length:
                return result
            elif len(result) > 0:
                if len(result) < default_length:
                    result.extend([True] * (default_length - len(result)))
                return result[:default_length]
    except Exception:
        pass
    return [True] * default_length


async def rewrite_query(state):
    print("---NODE: REWRITE---")
    res = await (rewrite_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    return {"rewritten_queries": [res.strip()]}


async def decompose_query(state):
    print("---NODE: DECOMPOSE---")
    res = await (decompose_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    return {"sub_queries": [q.strip() for q in res.split("\n") if q.strip()]}


async def retrieve_context(state):
    print("---NODE: RETRIEVE---")
    store = storage.store_manager.get_vector_store()
    if store is None:
        raise ValueError("Vector Store not initialized")

    bm25_retriever = storage.store_manager.bm25_retriever

    retriever = store.as_retriever(search_kwargs={"k": 10})

    queries = state.get("rewritten_queries", []) + state.get("sub_queries", [])
    if not queries:
        queries = [state["question"]]

    all_docs = []

    tasks = []
    for q in queries:
        tasks.append(retriever.ainvoke(q))
        if bm25_retriever:
            tasks.append(bm25_retriever.ainvoke(q))

    results = await asyncio.gather(*tasks)

    idx = 0
    for q in queries:
        dense_docs = results[idx]
        idx += 1
        
        sparse_results = []
        if bm25_retriever:
            sparse_results = results[idx]
            idx += 1
            
        fused_docs = storage.store_manager.reciprocal_rank_fusion(dense_docs, sparse_results)
        all_docs.extend(fused_docs[:10])

    unique_docs = {}
    for d in all_docs:
        chunk_id = d.metadata.get("chunk_id", d.page_content)
        if chunk_id not in unique_docs:
            unique_docs[chunk_id] = d

    reranked_docs, _ = await rerank(
        query=state["question"],
        docs=list(unique_docs.values()),
        top_n=7
    )

    unique_contents = []
    for d in reranked_docs:
        meta = d.metadata
        url = meta.get("source_url", "N/A")
        title = meta.get("document_title", "N/A")
        heading = meta.get("section_heading", "N/A")
        content = f"Source: {url}\nTitle: {title}\nHeading: {heading}\nContent: {d.page_content}"
        unique_contents.append(content)

    context = "\n\n".join(unique_contents)
    return {"context": context, "retrieved_chunks": unique_contents}


async def relevance_grader(state):
    print("---NODE: RELEVANCE GRADER---")
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        chunks = [c for c in state.get("context", "").split("\n\n") if c.strip()]

    if not chunks:
        return {"relevance_scores": [], "context": ""}

    formatted_chunks = "\n---\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)])

    res = await (relevance_prompt | llm | StrOutputParser()).ainvoke({
        "question": state["question"],
        "chunks": formatted_chunks,
    })

    scores = parse_json_bool_array(res, len(chunks))

    relevant_chunks = [chunk for chunk, is_rel in zip(chunks, scores) if is_rel]
    if not relevant_chunks:
        relevant_chunks = chunks[:3]
        scores = [True] * len(relevant_chunks) + [False] * (len(chunks) - len(relevant_chunks))

    if not relevant_chunks:
        filtered_context = ""
    else:
        filtered_context = "\n\n".join(relevant_chunks)

    return {"context": filtered_context, "relevance_scores": scores}


async def generate_answer(state):
    print("---NODE: GENERATE---")
    tokens = (len(state["question"]) + len(state.get("context", ""))) // 4
    await GEN_LLM_LIMITER.acquire(max(tokens, 1))

    ans = await (prompt | llm | StrOutputParser()).ainvoke({
        "context": state.get("context", ""),
        "question": state["question"],
    })
    return {"answer": ans}


async def categorize_question(state):
    print("---NODE: CATEGORIZE---")
    res = await (categorize_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    category = res.strip().lower()
    return {"category": category}


async def faithfulness_checker(state):
    print("---NODE: FAITHFULNESS CHECKER---")
    res = await (faithfulness_prompt | llm | StrOutputParser()).ainvoke({
        "context": state.get("context", ""),
        "answer": state.get("answer", ""),
    })

    is_faithful = "yes" in res.strip().lower()
    retry_count = state.get("retry_count", 0)

    if not is_faithful:
        retry_count += 1
        print(f"---UNFAITHFUL ANSWER DETECTED (retry_count={retry_count})---")
        return {"is_faithful": False, "faithfulness": "unfaithful", "retry_count": retry_count}
    else:
        print("---FAITHFUL ANSWER---")
        return {"is_faithful": True, "faithfulness": "faithful", "retry_count": retry_count}
