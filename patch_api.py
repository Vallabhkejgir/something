with open("app/api.py", "r") as f:
    content = f.read()

content = content.replace("from fastapi import FastAPI, Request", "from fastapi import FastAPI, Request, BackgroundTasks")
content = content.replace("async def query(req: QueryRequest):", "async def query(req: QueryRequest, background_tasks: BackgroundTasks):")
content = content.replace(
"""        # Save successful result to cache in background to avoid blocking response
        import asyncio
        asyncio.create_task(query_cache.set(user_prompt, {"answer": final_state["answer"]}, store_manager.index_version))""",
"""        # Save successful result to cache in background to avoid blocking response
        background_tasks.add_task(query_cache.set, user_prompt, {"answer": final_state["answer"]}, store_manager.index_version)""")

with open("app/api.py", "w") as f:
    f.write(content)
