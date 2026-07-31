import re

with open("app/RAG/nodes.py", "r") as f:
    content = f.read()

new_func = """
async def _do_retrieve_grade_generate(queries, question):
    retrieved_data = await _do_retrieve(queries)
    chunks = retrieved_data["retrieved_chunks"]
    
    if not chunks:
        async def empty_grade(): return "[]"
        async def empty_gen(): return ""
        return retrieved_data, asyncio.create_task(empty_grade()), asyncio.create_task(empty_gen())
        
    formatted_chunks = "\n---\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)])
    unfiltered_context = "\n\n".join(chunks)
    
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
    
    return retrieved_data, grade_task, generate_task
"""

# insert after _do_retrieve
content = content.replace("async def retrieve_context(state):", new_func + "\nasync def retrieve_context(state):")

# update relevance_grader
old_relevance = """async def relevance_grader(state):
    print("---NODE: RELEVANCE GRADER---")
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        chunks = [c for c in state.get("context", "").split("\\n\\n") if c.strip()]

    if not chunks:
        return {"relevance_scores": [], "context": "", "speculative_answer": ""}

    formatted_chunks = "\\n---\\n".join([f"Chunk {i+1}:\\n{chunk}" for i, chunk in enumerate(chunks)])

    # Fire both concurrently! We want to speculatively generate an answer assuming all chunks are relevant
    grade_coro = (relevance_prompt | llm | StrOutputParser()).ainvoke({
        "question": state["question"],
        "chunks": formatted_chunks,
    })
    
    unfiltered_context = "\\n\\n".join(chunks)
    
    async def _speculative_generate():
        tokens = (len(state["question"]) + len(unfiltered_context)) // 4
        await GEN_LLM_LIMITER.acquire(max(tokens, 1))
        return await (prompt | llm | StrOutputParser()).ainvoke({
            "context": unfiltered_context,
            "question": state["question"],
        })

    grade_task = asyncio.create_task(grade_coro)
    generate_task = asyncio.create_task(_speculative_generate())
    
    # We must wait for grade_task no matter what
    res = await grade_task"""

new_relevance = """async def relevance_grader(state):
    print("---NODE: RELEVANCE GRADER---")
    
    spec_grade = state.get("speculative_grade_task")
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

    res = await grade_task"""

content = content.replace(old_relevance, new_relevance)

# update categorize_question
old_categorize = """async def categorize_question(state):
    print("---NODE: CATEGORIZE---")

    async def get_category():
        res = await (categorize_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
        return res.strip().lower()

    async def get_rewritten_and_retrieve():
        res = await (rewrite_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
        queries = [q.strip() for q in res.split("\\n") if q.strip()]
        retrieved_data = await _do_retrieve(queries)
        return queries, retrieved_data

    async def get_decomposed_and_retrieve():
        res = await (decompose_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
        queries = [q.strip() for q in res.split("\\n") if q.strip()]
        retrieved_data = await _do_retrieve(queries)
        return queries, retrieved_data

    rewrite_task = asyncio.create_task(get_rewritten_and_retrieve())
    decompose_task = asyncio.create_task(get_decomposed_and_retrieve())

    category, retrieved_data = await asyncio.gather(
        get_category(),
        _do_retrieve([state["question"]])
    )

    if category == "vague":
        queries, ret_data = await rewrite_task
        decompose_task.cancel()
        return {
            "category": category,
            "context": retrieved_data["context"],
            "retrieved_chunks": retrieved_data["retrieved_chunks"],
            "speculative_rewritten_queries": queries,
            "speculative_context": ret_data["context"],
            "speculative_retrieved_chunks": ret_data["retrieved_chunks"]
        }
    elif category == "complex":
        queries, ret_data = await decompose_task
        rewrite_task.cancel()
        return {
            "category": category,
            "context": retrieved_data["context"],
            "retrieved_chunks": retrieved_data["retrieved_chunks"],
            "speculative_sub_queries": queries,
            "speculative_context": ret_data["context"],
            "speculative_retrieved_chunks": ret_data["retrieved_chunks"]
        }
    else:
        rewrite_task.cancel()
        decompose_task.cancel()
        return {
            "category": category,
            "context": retrieved_data["context"],
            "retrieved_chunks": retrieved_data["retrieved_chunks"]
        }"""

new_categorize = """async def categorize_question(state):
    print("---NODE: CATEGORIZE---")

    async def get_category():
        res = await (categorize_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
        return res.strip().lower()

    async def get_rewritten_and_retrieve():
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

content = content.replace(old_categorize, new_categorize)

with open("app/RAG/nodes.py", "w") as f:
    f.write(content)
