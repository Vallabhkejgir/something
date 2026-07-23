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
    res = await (rewrite_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    return {"rewritten_queries": [q.strip() for q in res.split("\n") if q.strip()]}


async def decompose_query(state):
    print("---NODE: DECOMPOSE---")
    res = await (decompose_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    return {"sub_queries": [q.strip() for q in res.split("\n") if q.strip()]}


async def retrieve_context(state):
    print("---NODE: RETRIEVE---")
    if storage.vector_store is None:
        raise ValueError("Vector Store not initialized")

    retriever = storage.vector_store.as_retriever(search_kwargs={"k": 5})

    queries = state.get("rewritten_queries", []) + state.get("sub_queries", [])
    if not queries:
        queries = [state["question"]]

    all_docs = []
    results = await asyncio.gather(*[retriever.ainvoke(q) for q in queries])
    for docs in results:
        all_docs.extend(docs)

    unique_contents = list(dict.fromkeys([d.page_content for d in all_docs]))
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
