import json
import re

from langchain_core.output_parsers import StrOutputParser
from app.services.llm_config import llm, GEN_LLM_LIMITER
from app.RAG.prompts import (
    adaptive_planner_prompt,
    prompt,
    rewrite_prompt,
    decompose_prompt,
    categorize_prompt,
)
from app.services import storage


ALLOWED_STRATEGIES = {
    "multi_query",
    "decompose",
    "grapho1",
    "tv_rag",
    "hifi_rag",
    "affordance_rag",
}


def _clean_query(text):
    text = re.sub(r"^\s*[-*]?\s*\d+[\).\s-]*", "", text).strip()
    return text.strip("\"'` ")


def _unique(items, limit=None):
    seen = set()
    out = []
    for item in items:
        if not item:
            continue
        cleaned = _clean_query(str(item))
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            out.append(cleaned)
        if limit and len(out) >= limit:
            break
    return out


def _as_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return value.splitlines()
    return []


def _json_from_text(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _queries_from_text(text, key="queries"):
    parsed = _json_from_text(text)
    if parsed.get(key):
        return _unique(_as_list(parsed.get(key)))
    return _unique(text.splitlines())


async def plan_adaptive_rag(state):
    print("---NODE: ADAPTIVE PLAN---")
    res = await (adaptive_planner_prompt | llm | StrOutputParser()).ainvoke({
        "question": state["question"]
    })
    plan = _json_from_text(res)

    category = str(plan.get("category") or "concise").strip().lower()
    if category not in {"vague", "complex", "concise"}:
        category = "concise"

    strategies = [
        str(strategy).strip().lower()
        for strategy in _as_list(plan.get("strategies", []))
        if str(strategy).strip().lower() in ALLOWED_STRATEGIES
    ]

    if category == "complex" and "decompose" not in strategies:
        strategies.append("decompose")
    if category == "vague" and "multi_query" not in strategies:
        strategies.append("multi_query")
    if "hifi_rag" not in strategies:
        strategies.append("hifi_rag")

    retrieval_queries = _unique(_as_list(plan.get("retrieval_queries", [])), limit=8)
    if not retrieval_queries:
        retrieval_queries = [state["question"]]
    elif state["question"].lower() not in {q.lower() for q in retrieval_queries}:
        retrieval_queries.insert(0, state["question"])
        retrieval_queries = _unique(retrieval_queries, limit=8)

    return {
        "category": category,
        "canonical_question": str(plan.get("canonical_question") or state["question"]).strip(),
        "adaptive_strategies": _unique(strategies),
        "retrieval_queries": retrieval_queries,
        "graph_focus": _unique(_as_list(plan.get("graph_focus", [])), limit=5),
        "temporal_focus": _unique(_as_list(plan.get("temporal_focus", [])), limit=5),
        "affordance_focus": _unique(_as_list(plan.get("affordance_focus", [])), limit=5),
    }


async def rewrite_query(state):
    print("---NODE: REWRITE---")
    res = await (rewrite_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    return {"rewritten_queries": _queries_from_text(res)}

async def decompose_query(state):
    print("---NODE: DECOMPOSE---")
    res = await (decompose_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    return {"sub_queries": _queries_from_text(res)}

async def retrieve_context(state):
    print("---NODE: RETRIEVE---")
    if storage.vector_store is None:
        raise ValueError("Vector Store not initialized")
    
    retriever = storage.vector_store.as_retriever(search_kwargs={"k": 5})
    
    queries = (
        state.get("retrieval_queries", [])
        + state.get("rewritten_queries", [])
        + state.get("sub_queries", [])
    )
    if not queries:
        queries = [state["question"]]
    queries = _unique(queries, limit=10)
    
    all_docs = []
    trace = []
    for q in queries:
        docs = await retriever.ainvoke(q)
        trace.append(f"{q} -> {len(docs)} docs")
        all_docs.extend(docs)
    
    unique_docs = []
    seen_docs = set()
    for doc in all_docs:
        content = doc.page_content.strip()
        if not content or content in seen_docs:
            continue
        seen_docs.add(content)
        unique_docs.append(doc)

    context_blocks = []
    for index, doc in enumerate(unique_docs[:12], start=1):
        source = doc.metadata.get("source") or doc.metadata.get("title") or "retrieved document"
        context_blocks.append(f"[{index}] Source: {source}\n{doc.page_content}")

    context = "\n\n".join(context_blocks)
    return {"context": context, "retrieval_trace": trace}

async def generate_answer(state):
    print("---NODE: GENERATE---")
    tokens = (len(state["question"]) + len(state["context"])) // 4
    await GEN_LLM_LIMITER.acquire(tokens)
    
    ans = await (prompt | llm | StrOutputParser()).ainvoke({
        "context": state["context"], 
        "question": state.get("canonical_question") or state["question"]
    })
    return {"answer": ans}


async def categorize_question(state):
    print("---NODE: CATEGORIZE---")
    # Invoke the LLM to get the category
    res = await (categorize_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    category = res.strip().lower()
    # We store the category in the state (update states.py if you want strict typing)
    return {"category": category}

