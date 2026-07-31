import re

with open("app/RAG/nodes.py", "r") as f:
    content = f.read()

# 1. Update categorize_question
old_cat = """async def categorize_question(state):
    print("---NODE: CATEGORIZE---")

    async def get_category():
        res = await (categorize_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
        return res.strip().lower()

    async def get_rewritten_and_retrieve():
        return await _do_rewrite_and_retrieve_grade_generate(state["question"])

    async def get_decomposed_and_retrieve():
        res = await (decompose_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
        queries = [q.strip() for q in res.split("\n") if q.strip()]
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

new_cat = """async def categorize_question(state):
    print("---NODE: CATEGORIZE---")

    async def get_category():
        res = await (categorize_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
        return res.strip().lower()

    async def get_rewritten_and_retrieve():
        return await _do_rewrite_and_retrieve_grade_generate(state["question"])

    async def get_decomposed_and_retrieve():
        res = await (decompose_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
        queries = [q.strip() for q in res.split("\n") if q.strip()]
        ret_data, grade_task, gen_task, faith_task = await _do_retrieve_grade_generate(queries, state["question"])
        return queries, ret_data, grade_task, gen_task, faith_task

    rewrite_task = asyncio.create_task(get_rewritten_and_retrieve())
    decompose_task = asyncio.create_task(get_decomposed_and_retrieve())
    concise_task = asyncio.create_task(_do_retrieve_grade_generate([state["question"]], state["question"]))
    cat_task = asyncio.create_task(get_category())

    category = await cat_task

    def _cancel_concise():
        if not concise_task.done():
            concise_task.cancel()
        else:
            try:
                _, gt, gent, ft = concise_task.result()
                gt.cancel()
                gent.cancel()
                ft.cancel()
            except Exception:
                pass

    if category == "vague":
        _cancel_concise()
        queries, ret_data, g_task, gen_task, f_task = await rewrite_task
        # Keep decompose_task running as a diverse fallback for vague!
        return {
            "category": category,
            "speculative_rewritten_queries": queries,
            "speculative_context": ret_data["context"],
            "speculative_retrieved_chunks": ret_data["retrieved_chunks"],
            "speculative_grade_task": g_task,
            "speculative_generate_task": gen_task,
            "speculative_faithfulness_task": f_task,
            "speculative_rewrite_fallback_task": decompose_task,
        }
    elif category == "complex":
        _cancel_concise()
        queries, ret_data, g_task, gen_task, f_task = await decompose_task
        # We KEEP rewrite_task running as a fallback!
        return {
            "category": category,
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
        retrieved_data, grade_task, generate_task, faith_task = await concise_task
        return {
            "category": category,
            "context": retrieved_data["context"],
            "retrieved_chunks": retrieved_data["retrieved_chunks"],
            "speculative_grade_task": grade_task,
            "speculative_generate_task": generate_task,
            "speculative_faithfulness_task": faith_task,
            "speculative_rewrite_fallback_task": rewrite_task,
        }"""
content = content.replace(old_cat, new_cat)

# 2. Update _do_retrieve_grade_generate
old_drgg = """    async def run_generate():
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
        })"""

new_drgg = """    async def run_generate():
        if not unfiltered_context.strip():
            return "I don't have enough information in the retrieved context."
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
        if not unfiltered_context.strip():
            return "yes"
        return await (faithfulness_prompt | llm | StrOutputParser()).ainvoke({
            "context": unfiltered_context,
            "answer": ans,
        })"""
content = content.replace(old_drgg, new_drgg)

# 3. Update relevance_grader
old_rg = """        # Pipeline the filtered ones!
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
            })"""

new_rg = """        # Pipeline the filtered ones!
        async def run_filtered_generate():
            if not filtered_context.strip():
                return "I don't have enough information in the retrieved context."
            tokens = (len(state["question"]) + len(filtered_context)) // 4
            await GEN_LLM_LIMITER.acquire(max(tokens, 1))
            return await (prompt | llm | StrOutputParser()).ainvoke({
                "context": filtered_context,
                "question": state["question"],
            })
            
        new_gen = asyncio.create_task(run_filtered_generate())
        
        async def run_filtered_faith():
            ans = await new_gen
            if not filtered_context.strip():
                return "yes"
            return await (faithfulness_prompt | llm | StrOutputParser()).ainvoke({
                "context": filtered_context,
                "answer": ans,
            })"""
content = content.replace(old_rg, new_rg)

# 4. Update generate_answer
old_ga = """    spec_gen = state.get("speculative_generate_task")
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

new_ga = """    spec_gen = state.get("speculative_generate_task")
    if spec_gen:
        print("---USING SPECULATIVE GENERATE TASK---")
        ans = await spec_gen
        return {"answer": ans}
        
    if not state.get("context", "").strip():
        return {"answer": "I don't have enough information in the retrieved context."}

    tokens = (len(state["question"]) + len(state.get("context", ""))) // 4
    await GEN_LLM_LIMITER.acquire(max(tokens, 1))

    ans = await (prompt | llm | StrOutputParser()).ainvoke({
        "context": state.get("context", ""),
        "question": state["question"],
    })
    return {"answer": ans}"""
content = content.replace(old_ga, new_ga)

# 5. Update faithfulness_checker
old_fc = """    if spec_faith:
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
        })"""

new_fc = """    if spec_faith:
        print("---USING SPECULATIVE FAITHFULNESS---")
        try:
            res = await spec_faith
        except asyncio.CancelledError:
            if not state.get("context", "").strip():
                res = "yes"
            else:
                res = await (faithfulness_prompt | llm | StrOutputParser()).ainvoke({
                    "context": state.get("context", ""),
                    "answer": state.get("answer", ""),
                })
    else:
        if not state.get("context", "").strip():
            res = "yes"
        else:
            res = await (faithfulness_prompt | llm | StrOutputParser()).ainvoke({
                "context": state.get("context", ""),
                "answer": state.get("answer", ""),
            })"""
content = content.replace(old_fc, new_fc)

with open("app/RAG/nodes.py", "w") as f:
    f.write(content)
