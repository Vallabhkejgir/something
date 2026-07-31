import asyncio
from app.RAG.graph import rag_app, GraphState
import os

os.environ["GOOGLE_API_KEY"] = "dummy"

async def main():
    inputs = GraphState(
        question="What is this?",
        category="",
        rewritten_queries=[],
        sub_queries=[],
        context="",
        answer="",
        retry_count=0,
        is_faithful=True,
        relevance_scores=[],
        retrieved_chunks=[],
    )
    try:
        # We can't really run it because of dummy api key but we can check it imports and compiles fine
        print("Graph compiled successfully")
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(main())
