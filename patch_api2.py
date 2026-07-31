import re

with open("app/api.py", "r") as f:
    content = f.read()

new_query = """@app.post("/api/query")
async def query(req: QueryRequest):
    if not initialized:
        return JSONResponse(status_code=400, content={"error": "Load docs first"})

    user_prompt = req.prompt
    
    # Try to fetch from cache first
    cached_result = await query_cache.get(user_prompt, store_manager.index_version)
    if cached_result:
        return {"answer": cached_result["answer"]}

    inputs = GraphState(
        question=user_prompt,
        category="",
        rewritten_queries=[],
        sub_queries=[],
        context="",
        answer="",
        retry_count=0,
        is_faithful=True,
        relevance_scores=[],
        retrieved_chunks=[],
        speculative_answer="",
    )

    try:
        final_state = await rag_app.ainvoke(inputs)
        
        # Save successful result to cache in background to avoid blocking response
        import asyncio
        asyncio.create_task(query_cache.set(user_prompt, {"answer": final_state["answer"]}, store_manager.index_version))
        
        return {"answer": final_state["answer"]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})"""

content = re.sub(r'@app\.post\("/api/query"\).*?except Exception as e:\n        return JSONResponse\(status_code=500, content=\{"error": str\(e\)\}\)', new_query, content, flags=re.DOTALL)

with open("app/api.py", "w") as f:
    f.write(content)
