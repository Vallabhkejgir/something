import re

with open('app/RAG/nodes.py', 'r') as f:
    content = f.read()

# 1. Update rewrite_query
old_rq = """async def rewrite_query(state):
    print("---NODE: REWRITE---")
    speculative = state.get("speculative_rewritten_queries")
    retry_count = state.get("retry_count", 0)
    if speculative and retry_count == 0:
        print("---USING SPECULATIVE REWRITTEN QUERIES---")
        return {"rewritten_queries": speculative}
    res = await (rewrite_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    return {"rewritten_queries": [q.strip() for q in res.split("\\n") if q.strip()]}"""

new_rq = """async def rewrite_query(state):
    print("---NODE: REWRITE---")
    speculative = state.get("speculative_rewritten_queries")
    speculative_fallback = state.get("speculative_rewrite_fallback_task")
    retry_count = state.get("retry_count", 0)
    
    if speculative and retry_count == 0:
        print("---USING SPECULATIVE REWRITTEN QUERIES---")
        return {"rewritten_queries": speculative}
        
    if speculative_fallback:
        print("---USING SPECULATIVE REWRITE FALLBACK---")
        try:
            queries, ret_data, grade_task, gen_task, faith_task = await speculative_fallback
            return {
                "rewritten_queries": queries,
                "speculative_context": ret_data["context"],
                "speculative_retrieved_chunks": ret_data["retrieved_chunks"],
                "speculative_grade_task": grade_task,
                "speculative_generate_task": gen_task,
                "speculative_faithfulness_task": faith_task,
                "speculative_rewrite_fallback_task": None,
            }
        except Exception as e:
            print(f"---SPECULATIVE FALLBACK FAILED: {e}---")

    res = await (rewrite_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    return {
        "rewritten_queries": [q.strip() for q in res.split("\\n") if q.strip()],
        "speculative_context": "",
        "speculative_retrieved_chunks": [],
        "speculative_grade_task": None,
        "speculative_generate_task": None,
        "speculative_faithfulness_task": None,
        "speculative_rewrite_fallback_task": None,
    }"""
content = content.replace(old_rq, new_rq)

# 2. Update decompose_query
old_dq = """async def decompose_query(state):
    print("---NODE: DECOMPOSE---")
    speculative = state.get("speculative_sub_queries")
    retry_count = state.get("retry_count", 0)
    if speculative and retry_count == 0:
        print("---USING SPECULATIVE SUB-QUERIES---")
        return {"sub_queries": speculative}
    res = await (decompose_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    return {"sub_queries": [q.strip() for q in res.split("\\n") if q.strip()]}"""

new_dq = """async def decompose_query(state):
    print("---NODE: DECOMPOSE---")
    speculative = state.get("speculative_sub_queries")
    retry_count = state.get("retry_count", 0)
    if speculative and retry_count == 0:
        print("---USING SPECULATIVE SUB-QUERIES---")
        return {"sub_queries": speculative}
    res = await (decompose_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    return {
        "sub_queries": [q.strip() for q in res.split("\\n") if q.strip()],
        "speculative_context": "",
        "speculative_retrieved_chunks": [],
        "speculative_grade_task": None,
        "speculative_generate_task": None,
        "speculative_faithfulness_task": None,
        "speculative_rewrite_fallback_task": None,
    }"""
content = content.replace(old_dq, new_dq)

# 3. Update categorize_question
old_cq = """    if category == "vague":
        queries, ret_data, g_task, gen_task, f_task = await rewrite_task
        decompose_task.cancel()
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
        }
    elif category == "complex":
        queries, ret_data, g_task, gen_task, f_task = await decompose_task
        rewrite_task.cancel()
        grade_task.cancel()
        generate_task.cancel()
        faith_task.cancel()
        return {
            "category": category,
            "context": retrieved_data["context"],
            "retrieved_chunks": retrieved_data["retrieved_chunks"],
            "speculative_sub_queries": queries,
            "speculative_context": ret_data["context"],
            "speculative_retrieved_chunks": ret_data["retrieved_chunks"],
            "speculative_grade_task": g_task,
            "speculative_generate_task": gen_task,
            "speculative_faithfulness_task": f_task,
        }
    else:
        rewrite_task.cancel()
        decompose_task.cancel()
        return {
            "category": category,
            "context": retrieved_data["context"],
            "retrieved_chunks": retrieved_data["retrieved_chunks"],
            "speculative_grade_task": grade_task,
            "speculative_generate_task": generate_task,
            "speculative_faithfulness_task": faith_task,
        }"""

new_cq = """    if category == "vague":
        queries, ret_data, g_task, gen_task, f_task = await rewrite_task
        decompose_task.cancel()
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
            "speculative_rewrite_fallback_task": None,
        }
    elif category == "complex":
        queries, ret_data, g_task, gen_task, f_task = await decompose_task
        # We KEEP rewrite_task running as a fallback!
        grade_task.cancel()
        generate_task.cancel()
        faith_task.cancel()
        return {
            "category": category,
            "context": retrieved_data["context"],
            "retrieved_chunks": retrieved_data["retrieved_chunks"],
            "speculative_sub_queries": queries,
            "speculative_context": ret_data["context"],
            "speculative_retrieved_chunks": ret_data["retrieved_chunks"],
            "speculative_grade_task": g_task,
            "speculative_generate_task": gen_task,
            "speculative_faithfulness_task": f_task,
            "speculative_rewrite_fallback_task": rewrite_task,
        }
    else:
        # We KEEP rewrite_task running as a fallback!
        decompose_task.cancel()
        return {
            "category": category,
            "context": retrieved_data["context"],
            "retrieved_chunks": retrieved_data["retrieved_chunks"],
            "speculative_grade_task": grade_task,
            "speculative_generate_task": generate_task,
            "speculative_faithfulness_task": faith_task,
            "speculative_rewrite_fallback_task": rewrite_task,
        }"""
content = content.replace(old_cq, new_cq)

# 4. Update retrieve_context
old_rc = """async def retrieve_context(state):
    print("---NODE: RETRIEVE---")
    spec_context = state.get("speculative_context")
    spec_chunks = state.get("speculative_retrieved_chunks")
    retry_count = state.get("retry_count", 0)
    
    if spec_context and spec_chunks and retry_count == 0:
        print("---USING SPECULATIVE RETRIEVAL---")
        return {"context": spec_context, "retrieved_chunks": spec_chunks}"""

new_rc = """async def retrieve_context(state):
    print("---NODE: RETRIEVE---")
    spec_context = state.get("speculative_context")
    spec_chunks = state.get("speculative_retrieved_chunks")
    
    if spec_context and spec_chunks:
        print("---USING SPECULATIVE RETRIEVAL---")
        return {"context": spec_context, "retrieved_chunks": spec_chunks}"""
content = content.replace(old_rc, new_rc)

# 5. Update relevance_grader
old_rg = """    spec_grade = state.get("speculative_grade_task")
    spec_gen = state.get("speculative_generate_task")
    spec_faith = state.get("speculative_faithfulness_task")
    retry_count = state.get("retry_count", 0)

    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        chunks = [c for c in state.get("context", "").split("\\n\\n") if c.strip()]

    if not chunks:
        return {"relevance_scores": [], "context": "", "speculative_answer": "", "speculative_faithfulness_task": None}

    if spec_grade and spec_gen and spec_faith and retry_count == 0:
        print("---USING SPECULATIVE GRADE & GENERATE TASKS---")"""

new_rg = """    spec_grade = state.get("speculative_grade_task")
    spec_gen = state.get("speculative_generate_task")
    spec_faith = state.get("speculative_faithfulness_task")

    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        chunks = [c for c in state.get("context", "").split("\\n\\n") if c.strip()]

    if not chunks:
        return {"relevance_scores": [], "context": "", "speculative_answer": "", "speculative_faithfulness_task": None}

    if spec_grade and spec_gen and spec_faith:
        print("---USING SPECULATIVE GRADE & GENERATE TASKS---")"""
content = content.replace(old_rg, new_rg)

# 6. Update faithfulness_checker
old_fc = """async def faithfulness_checker(state):
    print("---NODE: FAITHFULNESS CHECKER---")
    
    spec_faith = state.get("speculative_faithfulness_task")
    retry_count = state.get("retry_count", 0)
    
    if spec_faith and retry_count == 0:
        print("---USING SPECULATIVE FAITHFULNESS---")"""

new_fc = """async def faithfulness_checker(state):
    print("---NODE: FAITHFULNESS CHECKER---")
    
    spec_faith = state.get("speculative_faithfulness_task")
    retry_count = state.get("retry_count", 0)
    
    if spec_faith:
        print("---USING SPECULATIVE FAITHFULNESS---")"""
content = content.replace(old_fc, new_fc)

with open('app/RAG/nodes.py', 'w') as f:
    f.write(content)
