import re

with open('app/RAG/nodes.py', 'r') as f:
    content = f.read()

# 1. Update _do_retrieve_grade_generate to create faith_task
old_drgg = """    if not chunks:
        async def empty_grade(): return "[]"
        async def empty_gen(): return ""
        return retrieved_data, asyncio.create_task(empty_grade()), asyncio.create_task(empty_gen())
        
    formatted_chunks = "\\n---\\n".join([f"Chunk {i+1}:\\n{chunk}" for i, chunk in enumerate(chunks)])
    unfiltered_context = "\\n\\n".join(chunks)
    
    async def run_grade():
        return await (relevance_prompt | llm | StrOutputParser()).ainvoke({
            "question": question,
            "chunks": formatted_chunks,
        })
        
    async def run_generate():
        tokens = (len(question) + len(unfiltered_context)) // 4
        await GEN_LLM_LIMITER.acquire(max(tokens, 1))
        return await (prompt | llm | StrOutputParser()).ainvoke({
            "context": unfiltered_context,
            "question": question,
        })
        
    grade_task = asyncio.create_task(run_grade())
    generate_task = asyncio.create_task(run_generate())
    
    return retrieved_data, grade_task, generate_task"""

new_drgg = """    if not chunks:
        async def empty_grade(): return "[]"
        async def empty_gen(): return ""
        async def empty_faith(): return "yes"
        return retrieved_data, asyncio.create_task(empty_grade()), asyncio.create_task(empty_gen()), asyncio.create_task(empty_faith())
        
    formatted_chunks = "\\n---\\n".join([f"Chunk {i+1}:\\n{chunk}" for i, chunk in enumerate(chunks)])
    unfiltered_context = "\\n\\n".join(chunks)
    
    async def run_grade():
        return await (relevance_prompt | llm | StrOutputParser()).ainvoke({
            "question": question,
            "chunks": formatted_chunks,
        })
        
    async def run_generate():
        tokens = (len(question) + len(unfiltered_context)) // 4
        await GEN_LLM_LIMITER.acquire(max(tokens, 1))
        return await (prompt | llm | StrOutputParser()).ainvoke({
            "context": unfiltered_context,
            "question": question,
        })
        
    grade_task = asyncio.create_task(run_grade())
    generate_task = asyncio.create_task(run_generate())
    
    async def run_faithfulness():
        try:
            ans = await generate_task
        except asyncio.CancelledError:
            raise
        return await (faithfulness_prompt | llm | StrOutputParser()).ainvoke({
            "context": unfiltered_context,
            "answer": ans,
        })
        
    faith_task = asyncio.create_task(run_faithfulness())
    
    return retrieved_data, grade_task, generate_task, faith_task"""

content = content.replace(old_drgg, new_drgg)

# 2. Update relevance_grader
old_rg = """    spec_grade = state.get("speculative_grade_task")
    spec_gen = state.get("speculative_generate_task")
    retry_count = state.get("retry_count", 0)

    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        chunks = [c for c in state.get("context", "").split("\\n\\n") if c.strip()]

    if not chunks:
        return {"relevance_scores": [], "context": "", "speculative_answer": ""}

    if spec_grade and spec_gen and retry_count == 0:
        print("---USING SPECULATIVE GRADE & GENERATE TASKS---")
        grade_task = spec_grade
        generate_task = spec_gen
    else:
        formatted_chunks = "\\n---\\n".join([f"Chunk {i+1}:\\n{chunk}" for i, chunk in enumerate(chunks)])
        async def run_grade():
            return await (relevance_prompt | llm | StrOutputParser()).ainvoke({
                "question": state["question"],
                "chunks": formatted_chunks,
            })
        grade_task = asyncio.create_task(run_grade())
        
        unfiltered_context = "\\n\\n".join(chunks)
        async def _speculative_generate():
            tokens = (len(state["question"]) + len(unfiltered_context)) // 4
            await GEN_LLM_LIMITER.acquire(max(tokens, 1))
            return await (prompt | llm | StrOutputParser()).ainvoke({
                "context": unfiltered_context,
                "question": state["question"],
            })
        generate_task = asyncio.create_task(_speculative_generate())

    res = await grade_task
    scores = parse_json_bool_array(res, len(chunks))

    relevant_chunks = [chunk for chunk, is_rel in zip(chunks, scores) if is_rel]
    if not relevant_chunks:
        filtered_context = ""
    else:
        filtered_context = "\\n\\n".join(relevant_chunks)

    # If ALL chunks were relevant, the speculative answer is perfectly valid!
    if all(scores) and len(scores) == len(chunks):
        # We need the speculative answer, so we wait for it
        speculative_ans = await generate_task
        return {"context": filtered_context, "relevance_scores": scores, "speculative_answer": speculative_ans}
    else:
        # We don't need it. We can cancel it to save resources!
        generate_task.cancel()
        return {"context": filtered_context, "relevance_scores": scores, "speculative_answer": ""}"""

new_rg = """    spec_grade = state.get("speculative_grade_task")
    spec_gen = state.get("speculative_generate_task")
    spec_faith = state.get("speculative_faithfulness_task")
    retry_count = state.get("retry_count", 0)

    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        chunks = [c for c in state.get("context", "").split("\\n\\n") if c.strip()]

    if not chunks:
        return {"relevance_scores": [], "context": "", "speculative_answer": "", "speculative_faithfulness_task": None}

    if spec_grade and spec_gen and spec_faith and retry_count == 0:
        print("---USING SPECULATIVE GRADE & GENERATE TASKS---")
        grade_task = spec_grade
        generate_task = spec_gen
        faith_task = spec_faith
    else:
        formatted_chunks = "\\n---\\n".join([f"Chunk {i+1}:\\n{chunk}" for i, chunk in enumerate(chunks)])
        async def run_grade():
            return await (relevance_prompt | llm | StrOutputParser()).ainvoke({
                "question": state["question"],
                "chunks": formatted_chunks,
            })
        grade_task = asyncio.create_task(run_grade())
        
        unfiltered_context = "\\n\\n".join(chunks)
        async def _speculative_generate():
            tokens = (len(state["question"]) + len(unfiltered_context)) // 4
            await GEN_LLM_LIMITER.acquire(max(tokens, 1))
            return await (prompt | llm | StrOutputParser()).ainvoke({
                "context": unfiltered_context,
                "question": state["question"],
            })
        generate_task = asyncio.create_task(_speculative_generate())
        
        async def _speculative_faithfulness():
            try:
                ans = await generate_task
            except asyncio.CancelledError:
                raise
            return await (faithfulness_prompt | llm | StrOutputParser()).ainvoke({
                "context": unfiltered_context,
                "answer": ans,
            })
        faith_task = asyncio.create_task(_speculative_faithfulness())

    res = await grade_task
    scores = parse_json_bool_array(res, len(chunks))

    relevant_chunks = [chunk for chunk, is_rel in zip(chunks, scores) if is_rel]
    if not relevant_chunks:
        filtered_context = ""
    else:
        filtered_context = "\\n\\n".join(relevant_chunks)

    # If ALL chunks were relevant, the speculative answer is perfectly valid!
    if all(scores) and len(scores) == len(chunks):
        # We need the speculative answer, so we wait for it
        speculative_ans = await generate_task
        return {"context": filtered_context, "relevance_scores": scores, "speculative_answer": speculative_ans, "speculative_faithfulness_task": faith_task}
    else:
        # We don't need it. We can cancel it to save resources!
        generate_task.cancel()
        faith_task.cancel()
        return {"context": filtered_context, "relevance_scores": scores, "speculative_answer": "", "speculative_faithfulness_task": None}"""

content = content.replace(old_rg, new_rg)


# 3. Update categorize_question
old_cq = """    async def get_rewritten_and_retrieve():
        res = await (rewrite_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
        queries = [q.strip() for q in res.split("\\n") if q.strip()]
        ret_data, grade_task, gen_task = await _do_retrieve_grade_generate(queries, state["question"])
        return queries, ret_data, grade_task, gen_task

    async def get_decomposed_and_retrieve():
        res = await (decompose_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
        queries = [q.strip() for q in res.split("\\n") if q.strip()]
        ret_data, grade_task, gen_task = await _do_retrieve_grade_generate(queries, state["question"])
        return queries, ret_data, grade_task, gen_task

    rewrite_task = asyncio.create_task(get_rewritten_and_retrieve())
    decompose_task = asyncio.create_task(get_decomposed_and_retrieve())

    category, (retrieved_data, grade_task, generate_task) = await asyncio.gather(
        get_category(),
        _do_retrieve_grade_generate([state["question"]], state["question"])
    )

    if category == "vague":
        queries, ret_data, g_task, gen_task = await rewrite_task
        decompose_task.cancel()
        grade_task.cancel()
        generate_task.cancel()
        return {
            "category": category,
            "context": retrieved_data["context"],
            "retrieved_chunks": retrieved_data["retrieved_chunks"],
            "speculative_rewritten_queries": queries,
            "speculative_context": ret_data["context"],
            "speculative_retrieved_chunks": ret_data["retrieved_chunks"],
            "speculative_grade_task": g_task,
            "speculative_generate_task": gen_task,
        }
    elif category == "complex":
        queries, ret_data, g_task, gen_task = await decompose_task
        rewrite_task.cancel()
        grade_task.cancel()
        generate_task.cancel()
        return {
            "category": category,
            "context": retrieved_data["context"],
            "retrieved_chunks": retrieved_data["retrieved_chunks"],
            "speculative_sub_queries": queries,
            "speculative_context": ret_data["context"],
            "speculative_retrieved_chunks": ret_data["retrieved_chunks"],
            "speculative_grade_task": g_task,
            "speculative_generate_task": gen_task,
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
        }"""

new_cq = """    async def get_rewritten_and_retrieve():
        res = await (rewrite_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
        queries = [q.strip() for q in res.split("\\n") if q.strip()]
        ret_data, grade_task, gen_task, faith_task = await _do_retrieve_grade_generate(queries, state["question"])
        return queries, ret_data, grade_task, gen_task, faith_task

    async def get_decomposed_and_retrieve():
        res = await (decompose_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
        queries = [q.strip() for q in res.split("\\n") if q.strip()]
        ret_data, grade_task, gen_task, faith_task = await _do_retrieve_grade_generate(queries, state["question"])
        return queries, ret_data, grade_task, gen_task, faith_task

    rewrite_task = asyncio.create_task(get_rewritten_and_retrieve())
    decompose_task = asyncio.create_task(get_decomposed_and_retrieve())

    category, (retrieved_data, grade_task, generate_task, faith_task) = await asyncio.gather(
        get_category(),
        _do_retrieve_grade_generate([state["question"]], state["question"])
    )

    if category == "vague":
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

content = content.replace(old_cq, new_cq)


# 4. Update faithfulness_checker
old_fc = """async def faithfulness_checker(state):
    print("---NODE: FAITHFULNESS CHECKER---")
    res = await (faithfulness_prompt | llm | StrOutputParser()).ainvoke({
        "context": state.get("context", ""),
        "answer": state.get("answer", ""),
    })

    is_faithful = "yes" in res.strip().lower()
    retry_count = state.get("retry_count", 0)

    if not is_faithful:
        retry_count += 1
        print(f"---UNFAITHFUL ANSWER DETECTED (retry_count={retry_count})---")
        return {"is_faithful": False, "faithfulness": "unfaithful", "retry_count": retry_count}
    else:
        print("---FAITHFUL ANSWER---")
        return {"is_faithful": True, "faithfulness": "faithful", "retry_count": retry_count}"""

new_fc = """async def faithfulness_checker(state):
    print("---NODE: FAITHFULNESS CHECKER---")
    
    spec_faith = state.get("speculative_faithfulness_task")
    retry_count = state.get("retry_count", 0)
    
    if spec_faith and retry_count == 0:
        print("---USING SPECULATIVE FAITHFULNESS---")
        try:
            res = await spec_faith
        except asyncio.CancelledError:
            res = await (faithfulness_prompt | llm | StrOutputParser()).ainvoke({
                "context": state.get("context", ""),
                "answer": state.get("answer", ""),
            })
    else:
        res = await (faithfulness_prompt | llm | StrOutputParser()).ainvoke({
            "context": state.get("context", ""),
            "answer": state.get("answer", ""),
        })

    is_faithful = "yes" in res.strip().lower()

    if not is_faithful:
        retry_count += 1
        print(f"---UNFAITHFUL ANSWER DETECTED (retry_count={retry_count})---")
        return {"is_faithful": False, "faithfulness": "unfaithful", "retry_count": retry_count}
    else:
        print("---FAITHFUL ANSWER---")
        return {"is_faithful": True, "faithfulness": "faithful", "retry_count": retry_count}"""

content = content.replace(old_fc, new_fc)

with open('app/RAG/nodes.py', 'w') as f:
    f.write(content)
