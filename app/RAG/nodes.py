import json
import asyncio
from typing import List
from langchain_core.output_parsers import StrOutputParser
from app.services.llm_config import llm, GEN_LLM_LIMITER
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
    speculative = state.get("speculative_rewritten_queries")
    retry_count = state.get("retry_count", 0)
    if speculative and retry_count == 0:
        print("---USING SPECULATIVE REWRITTEN QUERIES---")
        return {"rewritten_queries": speculative}
    res = await (rewrite_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    return {"rewritten_queries": [q.strip() for q in res.split("\n") if q.strip()]}


async def decompose_query(state):
    print("---NODE: DECOMPOSE---")
    speculative = state.get("speculative_sub_queries")
    retry_count = state.get("retry_count", 0)
    if speculative and retry_count == 0:
        print("---USING SPECULATIVE SUB-QUERIES---")
        return {"sub_queries": speculative}
    res = await (decompose_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    return {"sub_queries": [q.strip() for q in res.split("\n") if q.strip()]}



async def _do_retrieve(queries):
    store = storage.store_manager.get_vector_store()
    if store is None:
        raise ValueError("Vector Store not initialized")

    bm25_retriever = storage.store_manager.bm25_retriever
    retriever = store.as_retriever(search_kwargs={"k": 5})

    all_docs = []

    async def process_query(q):
        dense_coro = retriever.ainvoke(q)
        if bm25_retriever:
            sparse_coro = bm25_retriever.ainvoke(q)
            dense_docs, sparse_results = await asyncio.gather(dense_coro, sparse_coro)
        else:
            dense_docs = await dense_coro
            sparse_results = []
        fused_docs = storage.store_manager.reciprocal_rank_fusion(dense_docs, sparse_results)
        return fused_docs[:5]

    results = await asyncio.gather(*(process_query(q) for q in queries))
    for fused_docs in results:
        all_docs.extend(fused_docs)

    unique_docs = {}
    for d in all_docs:
        chunk_id = d.metadata.get("chunk_id", d.page_content)
        if chunk_id not in unique_docs:
            unique_docs[chunk_id] = d

    unique_contents = []
    for d in unique_docs.values():
        meta = d.metadata
        url = meta.get("source_url", "N/A")
        title = meta.get("document_title", "N/A")
        heading = meta.get("section_heading", "N/A")
        content = f"Source: {url}\nTitle: {title}\nHeading: {heading}\nContent: {d.page_content}"
        unique_contents.append(content)

    context = "\n\n".join(unique_contents)
    return {"context": context, "retrieved_chunks": unique_contents}

async def retrieve_context(state):
    print("---NODE: RETRIEVE---")
    spec_context = state.get("speculative_context")
    spec_chunks = state.get("speculative_retrieved_chunks")
    retry_count = state.get("retry_count", 0)
    
    if spec_context and spec_chunks and retry_count == 0:
        print("---USING SPECULATIVE RETRIEVAL---")
        return {"context": spec_context, "retrieved_chunks": spec_chunks}

    queries = state.get("rewritten_queries", []) + state.get("sub_queries", [])
    if not queries:
        queries = [state["question"]]
    return await _do_retrieve(queries)


async def relevance_grader(state):
    print("---NODE: RELEVANCE GRADER---")
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        chunks = [c for c in state.get("context", "").split("\n\n") if c.strip()]

    if not chunks:
        return {"relevance_scores": [], "context": "", "speculative_answer": ""}

    formatted_chunks = "\n---\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)])

    # Fire both concurrently! We want to speculatively generate an answer assuming all chunks are relevant
    grade_coro = (relevance_prompt | llm | StrOutputParser()).ainvoke({
        "question": state["question"],
        "chunks": formatted_chunks,
    })
    
    unfiltered_context = "\n\n".join(chunks)
    
    async def _speculative_generate():
        tokens = (len(state["question"]) + len(unfiltered_context)) // 4
        await GEN_LLM_LIMITER.acquire(max(tokens, 1))
        return await (prompt | llm | StrOutputParser()).ainvoke({
            "context": unfiltered_context,
            "question": state["question"],
        })

    grade_task = asyncio.create_task(grade_coro)
    generate_task = asyncio.create_task(_speculative_generate())
    
    # We must wait for grade_task no matter what
    res = await grade_task
    scores = parse_json_bool_array(res, len(chunks))

    relevant_chunks = [chunk for chunk, is_rel in zip(chunks, scores) if is_rel]
    if not relevant_chunks:
        filtered_context = ""
    else:
        filtered_context = "\n\n".join(relevant_chunks)

    # If ALL chunks were relevant, the speculative answer is perfectly valid!
    if all(scores) and len(scores) == len(chunks):
        # We need the speculative answer, so we wait for it
        speculative_ans = await generate_task
        return {"context": filtered_context, "relevance_scores": scores, "speculative_answer": speculative_ans}
    else:
        # We don't need it. We can cancel it to save resources!
        generate_task.cancel()
        return {"context": filtered_context, "relevance_scores": scores, "speculative_answer": ""}


async def generate_answer(state):
    print("---NODE: GENERATE---")
    
    speculative_ans = state.get("speculative_answer", "")
    if speculative_ans:
        print("---USING SPECULATIVE ANSWER---")
        return {"answer": speculative_ans}
        
    tokens = (len(state["question"]) + len(state.get("context", ""))) // 4
    await GEN_LLM_LIMITER.acquire(max(tokens, 1))

    ans = await (prompt | llm | StrOutputParser()).ainvoke({
        "context": state.get("context", ""),
        "question": state["question"],
    })
    return {"answer": ans}


async def categorize_question(state):
    print("---NODE: CATEGORIZE---")

    async def get_category():
        res = await (categorize_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
        return res.strip().lower()

    async def get_rewritten_and_retrieve():
        res = await (rewrite_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
        queries = [q.strip() for q in res.split("\n") if q.strip()]
        retrieved_data = await _do_retrieve(queries)
        return queries, retrieved_data

    async def get_decomposed_and_retrieve():
        res = await (decompose_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
        queries = [q.strip() for q in res.split("\n") if q.strip()]
        retrieved_data = await _do_retrieve(queries)
        return queries, retrieved_data

    rewrite_task = asyncio.create_task(get_rewritten_and_retrieve())
    decompose_task = asyncio.create_task(get_decomposed_and_retrieve())

    category, retrieved_data = await asyncio.gather(
        get_category(),
        _do_retrieve([state["question"]])
    )

    if category == "vague":
        queries, ret_data = await rewrite_task
        decompose_task.cancel()
        return {
            "category": category,
            "context": retrieved_data["context"],
            "retrieved_chunks": retrieved_data["retrieved_chunks"],
            "speculative_rewritten_queries": queries,
            "speculative_context": ret_data["context"],
            "speculative_retrieved_chunks": ret_data["retrieved_chunks"]
        }
    elif category == "complex":
        queries, ret_data = await decompose_task
        rewrite_task.cancel()
        return {
            "category": category,
            "context": retrieved_data["context"],
            "retrieved_chunks": retrieved_data["retrieved_chunks"],
            "speculative_sub_queries": queries,
            "speculative_context": ret_data["context"],
            "speculative_retrieved_chunks": ret_data["retrieved_chunks"]
        }
    else:
        rewrite_task.cancel()
        decompose_task.cancel()
        return {
            "category": category,
            "context": retrieved_data["context"],
            "retrieved_chunks": retrieved_data["retrieved_chunks"]
        }


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
