import re

with open("app/RAG/nodes.py", "r") as f:
    content = f.read()

# relevance_grader unhandled exception fix
old_grader_await = """    res = await grade_task
    scores = parse_json_bool_array(res, len(chunks))"""
new_grader_await = """    try:
        res = await grade_task
    except Exception as e:
        print(f"---SPECULATIVE GRADE FAILED: {e}---")
        formatted_chunks = "\\n---\\n".join([f"Chunk {i+1}:\\n{chunk}" for i, chunk in enumerate(chunks)])
        res = await (relevance_prompt | llm | StrOutputParser()).ainvoke({
            "question": state["question"],
            "chunks": formatted_chunks,
        })
    scores = parse_json_bool_array(res, len(chunks))"""
content = content.replace(old_grader_await, new_grader_await)

# generate_answer unhandled exception fix
old_gen_await = """    spec_gen = state.get("speculative_generate_task")
    if spec_gen:
        print("---USING SPECULATIVE GENERATE TASK---")
        ans = await spec_gen
        return {"answer": ans}"""
new_gen_await = """    spec_gen = state.get("speculative_generate_task")
    if spec_gen:
        print("---USING SPECULATIVE GENERATE TASK---")
        try:
            ans = await spec_gen
            return {"answer": ans}
        except Exception as e:
            print(f"---SPECULATIVE GENERATE TASK FAILED: {e}---")"""
content = content.replace(old_gen_await, new_gen_await)

# relevance_grader canceling fallback
old_grader_start = """async def relevance_grader(state):
    print("---NODE: RELEVANCE GRADER---")
    
    chunks = state.get("retrieved_chunks", [])"""
new_grader_start = """async def relevance_grader(state):
    print("---NODE: RELEVANCE GRADER---")
    
    fallback = state.get("speculative_rewrite_fallback_task")
    if fallback:
        fallback.cancel()

    chunks = state.get("retrieved_chunks", [])"""
content = content.replace(old_grader_start, new_grader_start)

with open("app/RAG/nodes.py", "w") as f:
    f.write(content)
