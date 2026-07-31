with open("app/RAG/nodes.py", "r") as f:
    content = f.read()

# 1. Cancel in rewrite_query
old_rewrite_1 = """    if speculative_vague and retry_count == 0:
        print("---USING SPECULATIVE VAGUE TASK---")
        try:
            queries, retrieve_task, grade_task, gen_task, faith_task = await speculative_vague
            return {"""
new_rewrite_1 = """    if speculative_vague and retry_count == 0:
        print("---USING SPECULATIVE VAGUE TASK---")
        try:
            queries, retrieve_task, grade_task, gen_task, faith_task = await speculative_vague
            if speculative_fallback:
                speculative_fallback.cancel()
            return {"""
content = content.replace(old_rewrite_1, new_rewrite_1)

old_rewrite_2 = """    res = await (rewrite_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    return {"""
new_rewrite_2 = """    if speculative_fallback:
        speculative_fallback.cancel()
    res = await (rewrite_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    return {"""
content = content.replace(old_rewrite_2, new_rewrite_2)

# 2. Cancel in decompose_query
old_decompose_1 = """async def decompose_query(state):
    print("---NODE: DECOMPOSE---")
    speculative_complex = state.get("speculative_complex_task")
    retry_count = state.get("retry_count", 0)"""
new_decompose_1 = """async def decompose_query(state):
    print("---NODE: DECOMPOSE---")
    speculative_complex = state.get("speculative_complex_task")
    speculative_fallback = state.get("speculative_rewrite_fallback_task")
    if speculative_fallback:
        speculative_fallback.cancel()
    retry_count = state.get("retry_count", 0)"""
content = content.replace(old_decompose_1, new_decompose_1)

# 3. Cancel in faithfulness_checker when faithful
old_faith_1 = """    else:
        print("---FAITHFUL ANSWER---")
        return {"is_faithful": True, "faithfulness": "faithful", "retry_count": retry_count}"""
new_faith_1 = """    else:
        print("---FAITHFUL ANSWER---")
        fallback = state.get("speculative_rewrite_fallback_task")
        if fallback:
            fallback.cancel()
        return {"is_faithful": True, "faithfulness": "faithful", "retry_count": retry_count, "speculative_rewrite_fallback_task": None}"""
content = content.replace(old_faith_1, new_faith_1)

with open("app/RAG/nodes.py", "w") as f:
    f.write(content)
