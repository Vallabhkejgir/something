import re

with open("app/RAG/nodes.py", "r") as f:
    content = f.read()

# Refactor retrieve logic into _do_retrieve
retrieve_logic = """
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
        content = f"Source: {url}\\nTitle: {title}\\nHeading: {heading}\\nContent: {d.page_content}"
        unique_contents.append(content)

    context = "\\n\\n".join(unique_contents)
    return {"context": context, "retrieved_chunks": unique_contents}

async def retrieve_context(state):
    print("---NODE: RETRIEVE---")
    queries = state.get("rewritten_queries", []) + state.get("sub_queries", [])
    if not queries:
        queries = [state["question"]]
    return await _do_retrieve(queries)
"""

# Replace old retrieve_context
old_retrieve_start = content.find("async def retrieve_context(state):")
old_retrieve_end = content.find("async def relevance_grader(state):")
content = content[:old_retrieve_start] + retrieve_logic + "\n\n" + content[old_retrieve_end:]

# Replace categorize_question
old_categorize = """async def categorize_question(state):
    print("---NODE: CATEGORIZE---")
    res = await (categorize_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    category = res.strip().lower()
    return {"category": category}"""

new_categorize = """async def categorize_question(state):
    print("---NODE: CATEGORIZE---")
    
    async def get_category():
        res = await (categorize_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
        return res.strip().lower()
        
    category, retrieved_data = await asyncio.gather(
        get_category(),
        _do_retrieve([state["question"]])
    )
    
    return {
        "category": category,
        "context": retrieved_data["context"],
        "retrieved_chunks": retrieved_data["retrieved_chunks"]
    }"""

content = content.replace(old_categorize, new_categorize)

with open("app/RAG/nodes.py", "w") as f:
    f.write(content)
