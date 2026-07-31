import asyncio
import time
from app.RAG.nodes import relevance_grader
from app.services.llm_config import llm

async def main():
    state = {
        "question": "What is the capital of France?",
        "context": "Paris is the capital of France.\n\nLondon is the capital of UK.\n\nBerlin is in Germany.",
        "retrieved_chunks": [
            "Paris is the capital of France.",
            "London is the capital of UK.",
            "Berlin is in Germany."
        ]
    }
    
    start = time.time()
    res = await relevance_grader(state)
    print(f"Time: {time.time() - start:.2f}s")
    print(res)

asyncio.run(main())
