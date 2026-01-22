import asyncio
from loader import Doc_loader
from chunks import chunk_texts
from Graph import app
from output import Output
from storage import store_chunks
from graph_types import GraphState

async def main():
    # 1. Load and chunk documents (Synchronous)
    docs = Doc_loader()
    chunks = chunk_texts(docs)

    # 2. Store chunks (Synchronous FAISS)
    try:
        # This is sync, so we just call it normally
        store_chunks(chunks)
        print("✅ Chunks stored successfully")
    except Exception as e:
        print(f"⚠️ Warning: Failed to store chunks - {str(e)}")
        return

    query = "Considering the functionalities of RunnableParallel for querying multiple sources and RunnableLambda for custom function integration, design a conceptual LCEL chain..."

    inputs: GraphState = {
        "question": query,
        "rewritten_queries": [],   
        "sub_queries": [],         
        "context": [],             
        "answer": ""               
    }

    print("🚀 Starting graph execution...\n")

    # 3. Stream Output (Async)
    # This executes the graph and prints updates
    # await Output(inputs)
    
    # 4. If you want to access the final state AFTER the stream
    # Note: app.astream yields updates. To get the final state without re-running,
    # you typically rely on the last yield of the stream.
    # However, if you simply want to print the answer again clearly:
    
    # WARNING: Calling await app.ainvoke(inputs) here would run the graph A SECOND TIME.
    # To save money/time, we just trust Output() to show us the progress.
    # If you really need the final object, you can do this instead of Output():
    
    final_state = await app.ainvoke(inputs)
    print(final_state['answer'])

if __name__ == "__main__":
    # Run the async main loop
    asyncio.run(main())

