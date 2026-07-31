import re
with open("app/RAG/nodes.py", "r") as f:
    content = f.read()

# Replace the relevance_grader else block
old_else = """    else:
        # We don't need it. We can cancel it to save resources!
        generate_task.cancel()
        faith_task.cancel()
        return {"context": filtered_context, "relevance_scores": scores, "speculative_answer": "", "speculative_faithfulness_task": None}"""

new_else = """    else:
        # We don't need the unfiltered ones. Cancel them!
        generate_task.cancel()
        faith_task.cancel()
        
        # Pipeline the filtered ones!
        async def run_filtered_generate():
            tokens = (len(state["question"]) + len(filtered_context)) // 4
            await GEN_LLM_LIMITER.acquire(max(tokens, 1))
            return await (prompt | llm | StrOutputParser()).ainvoke({
                "context": filtered_context,
                "question": state["question"],
            })
            
        new_gen = asyncio.create_task(run_filtered_generate())
        
        async def run_filtered_faith():
            ans = await new_gen
            return await (faithfulness_prompt | llm | StrOutputParser()).ainvoke({
                "context": filtered_context,
                "answer": ans,
            })
            
        new_faith = asyncio.create_task(run_filtered_faith())
        
        return {
            "context": filtered_context, 
            "relevance_scores": scores, 
            "speculative_answer": "", 
            "speculative_generate_task": new_gen,
            "speculative_faithfulness_task": new_faith
        }"""
content = content.replace(old_else, new_else)

# Replace generate_answer
old_gen = """async def generate_answer(state):
    print("---NODE: GENERATE---")
    
    speculative_ans = state.get("speculative_answer", "")
    if speculative_ans:
        print("---USING SPECULATIVE ANSWER---")
        return {"answer": speculative_ans}
        
    tokens = (len(state["question"]) + len(state.get("context", ""))) // 4
    await GEN_LLM_LIMITER.acquire(max(tokens, 1))

    ans = await (prompt | llm | StrOutputParser()).ainvoke({
        "context": state.get("context", ""),
        "question": state["question"],
    })
    return {"answer": ans}"""

new_gen = """async def generate_answer(state):
    print("---NODE: GENERATE---")
    
    speculative_ans = state.get("speculative_answer", "")
    if speculative_ans:
        print("---USING SPECULATIVE ANSWER---")
        return {"answer": speculative_ans}
        
    spec_gen = state.get("speculative_generate_task")
    if spec_gen:
        print("---USING SPECULATIVE GENERATE TASK---")
        ans = await spec_gen
        return {"answer": ans}
        
    tokens = (len(state["question"]) + len(state.get("context", ""))) // 4
    await GEN_LLM_LIMITER.acquire(max(tokens, 1))

    ans = await (prompt | llm | StrOutputParser()).ainvoke({
        "context": state.get("context", ""),
        "question": state["question"],
    })
    return {"answer": ans}"""
content = content.replace(old_gen, new_gen)

# In categorize_question, fix the vague fallback
old_vague = """    if category == "vague":
        queries, ret_data, g_task, gen_task, f_task = await rewrite_task
        decompose_task.cancel()
        grade_task.cancel()
        generate_task.cancel()
        faith_task.cancel()
        # We need a new rewrite_task running as a fallback for vague!
        new_fallback_task = asyncio.create_task(get_rewritten_and_retrieve())
        return {
            "category": category,
            "context": retrieved_data["context"],
            "retrieved_chunks": retrieved_data["retrieved_chunks"],
            "speculative_rewritten_queries": queries,
            "speculative_context": ret_data["context"],
            "speculative_retrieved_chunks": ret_data["retrieved_chunks"],
            "speculative_grade_task": g_task,
            "speculative_generate_task": gen_task,
            "speculative_faithfulness_task": f_task,
            "speculative_rewrite_fallback_task": new_fallback_task,
        }"""

new_vague = """    if category == "vague":
        queries, ret_data, g_task, gen_task, f_task = await rewrite_task
        # Keep decompose_task running as a diverse fallback for vague!
        grade_task.cancel()
        generate_task.cancel()
        faith_task.cancel()
        return {
            "category": category,
            "context": retrieved_data["context"],
            "retrieved_chunks": retrieved_data["retrieved_chunks"],
            "speculative_rewritten_queries": queries,
            "speculative_context": ret_data["context"],
            "speculative_retrieved_chunks": ret_data["retrieved_chunks"],
            "speculative_grade_task": g_task,
            "speculative_generate_task": gen_task,
            "speculative_faithfulness_task": f_task,
            "speculative_rewrite_fallback_task": decompose_task,
        }"""
content = content.replace(old_vague, new_vague)

with open("app/RAG/nodes.py", "w") as f:
    f.write(content)
