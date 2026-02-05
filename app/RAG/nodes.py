from langchain_core.output_parsers import StrOutputParser
from app.services.llm_config import llm, GEN_LLM_LIMITER
from app.RAG.prompts import prompt, rewrite_prompt, decompose_prompt, categorize_prompt
from app.services import storage

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
    for q in queries:
        docs = await retriever.ainvoke(q)
        all_docs.extend(docs)
    
    context = "\n\n".join(set([d.page_content for d in all_docs]))
    return {"context": context}

async def generate_answer(state):
    print("---NODE: GENERATE---")
    tokens = (len(state["question"]) + len(state["context"])) // 4
    await GEN_LLM_LIMITER.acquire(tokens)
    
    ans = await (prompt | llm | StrOutputParser()).ainvoke({
        "context": state["context"], 
        "question": state["question"]
    })
    return {"answer": ans}

async def categorize_question(state):
    print("---NODE: CATEGORIZE---")
    res = await (categorize_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    category = res.strip().lower()
    return {"category": category}