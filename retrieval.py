# Import the storage MODULE, not the variable. 
# This ensures we see the updates made by app.py later.
import storage 
from graph_types import GraphState

async def retrieve_context(state: GraphState):
    print("---NODE: RETRIEVE_CONTEXT---")

    # --- THE FIX ---
    # We check the vector store HERE, inside the function.
    # By the time this runs, app.py will have filled 'storage.vector_store'.
    if storage.vector_store is None:
        raise ValueError("Vector Store is empty! Make sure 'store_chunks' ran successfully.")
        
    # Create the retriever dynamically now that data exists
    retriever = storage.vector_store.as_retriever(search_kwargs={"k": 5})
    # ---------------

    queries = (
        state.get("rewritten_queries", [])
        + state.get("sub_queries", [])
    )

    all_docs = []

    for q in queries:
        # FAISS retrievers in LangChain are wrapped to be async-compatible.
        # We use await here.
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