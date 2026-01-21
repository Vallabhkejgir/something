from storage import vector_store
from graph_types import GraphState

retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 most relevant chunks


async def retrieve_context(state: GraphState):
    print("---NODE: RETRIEVE_CONTEXT---")

    queries = (
        state.get("rewritten_queries", [])
        + state.get("sub_queries", [])
    )

    all_docs = []

    for q in queries:
        docs = await retriever.ainvoke(q)
        all_docs.extend(docs)

    # Deduplicate by content
    seen, unique_docs = set(), []
    for d in all_docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            unique_docs.append(d)

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source','N/A')}\n{doc.page_content}"
        for doc in unique_docs
    )

    return {
        "context": context,
        "question": state["question"]
    }
