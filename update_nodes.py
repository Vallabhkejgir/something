import re

with open('app/RAG/nodes.py', 'r') as f:
    content = f.read()

# Replace _do_retrieve_grade_generate with _start_retrieve_grade_generate
new_start_retrieve = """
def _start_retrieve_grade_generate(queries, question):
    retrieve_task = asyncio.create_task(_do_retrieve(queries))
    
    async def run_grade():
        retrieved_data = await retrieve_task
        chunks = retrieved_data["retrieved_chunks"]
        if not chunks:
            return "[]"
        formatted_chunks = "\\n---\\n".join([f"Chunk {i+1}:\\n{chunk}" for i, chunk in enumerate(chunks)])
        return await (relevance_prompt | llm | StrOutputParser()).ainvoke({
            "question": question,
            "chunks": formatted_chunks,
        })
        
    grade_task = asyncio.create_task(run_grade())
    
    async def run_generate():
        retrieved_data = await retrieve_task
        chunks = retrieved_data["retrieved_chunks"]
        if not chunks:
            return ""
        unfiltered_context = "\\n\\n".join(chunks)
        tokens = (len(question) + len(unfiltered_context)) // 4
        await GEN_LLM_LIMITER.acquire(max(tokens, 1))
        return await (prompt | llm | StrOutputParser()).ainvoke({
            "context": unfiltered_context,
            "question": question,
        })
        
    generate_task = asyncio.create_task(run_generate())
    
    async def run_faithfulness():
        try:
            ans = await generate_task
        except asyncio.CancelledError:
            raise
        retrieved_data = await retrieve_task
        chunks = retrieved_data["retrieved_chunks"]
        if not chunks:
            return "yes"
        unfiltered_context = "\\n\\n".join(chunks)
        return await (faithfulness_prompt | llm | StrOutputParser()).ainvoke({
            "context": unfiltered_context,
            "answer": ans,
        })
        
    faith_task = asyncio.create_task(run_faithfulness())
    
    return retrieve_task, grade_task, generate_task, faith_task

async def _start_rewrite_and_retrieve_grade_generate(question):
    res = await (rewrite_prompt | llm | StrOutputParser()).ainvoke({"question": question})
    queries = [q.strip() for q in res.split("\\n") if q.strip()]
    retrieve_task, grade_task, gen_task, faith_task = _start_retrieve_grade_generate(queries, question)
    return queries, retrieve_task, grade_task, gen_task, faith_task
"""

content = re.sub(r'async def _do_retrieve_grade_generate.*?async def _do_rewrite_and_retrieve_grade_generate.*?return queries, ret_data, grade_task, gen_task, faith_task', new_start_retrieve, content, flags=re.DOTALL)

# Update rewrite_query
old_rewrite = """
    if speculative_vague and retry_count == 0:
        print("---USING SPECULATIVE VAGUE TASK---")
        try:
            queries, ret_data, grade_task, gen_task, faith_task = await speculative_vague
            return {
                "rewritten_queries": queries,
                "speculative_grade_task": grade_task,
                "speculative_generate_task": gen_task,
                "speculative_faithfulness_task": faith_task,
                "retrieved_chunks": ret_data["retrieved_chunks"],
                "context": ret_data["context"],
            }
        except Exception as e:
            print(f"---SPECULATIVE VAGUE FAILED: {e}---")

    if speculative_fallback:
        print("---USING SPECULATIVE REWRITE FALLBACK---")
        try:
            queries, ret_data, grade_task, gen_task, faith_task = await speculative_fallback
            new_fallback_task = asyncio.create_task(_do_rewrite_and_retrieve_grade_generate(state["question"]))
            return {
                "rewritten_queries": queries,
                "speculative_grade_task": grade_task,
                "speculative_generate_task": gen_task,
                "speculative_faithfulness_task": faith_task,
                "speculative_rewrite_fallback_task": new_fallback_task,
                "retrieved_chunks": ret_data["retrieved_chunks"],
                "context": ret_data["context"],
            }
        except Exception as e:
            print(f"---SPECULATIVE FALLBACK FAILED: {e}---")

    res = await (rewrite_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    return {
        "rewritten_queries": [q.strip() for q in res.split("\\n") if q.strip()],
        "speculative_grade_task": None,
        "speculative_generate_task": None,
        "speculative_faithfulness_task": None,
        "speculative_rewrite_fallback_task": None,
        "retrieved_chunks": [],
        "context": "",
    }"""

new_rewrite = """
    if speculative_vague and retry_count == 0:
        print("---USING SPECULATIVE VAGUE TASK---")
        try:
            queries, retrieve_task, grade_task, gen_task, faith_task = await speculative_vague
            return {
                "rewritten_queries": queries,
                "speculative_retrieve_task": retrieve_task,
                "speculative_grade_task": grade_task,
                "speculative_generate_task": gen_task,
                "speculative_faithfulness_task": faith_task,
            }
        except Exception as e:
            print(f"---SPECULATIVE VAGUE FAILED: {e}---")

    if speculative_fallback:
        print("---USING SPECULATIVE REWRITE FALLBACK---")
        try:
            queries, retrieve_task, grade_task, gen_task, faith_task = await speculative_fallback
            new_fallback_task = asyncio.create_task(_start_rewrite_and_retrieve_grade_generate(state["question"]))
            return {
                "rewritten_queries": queries,
                "speculative_retrieve_task": retrieve_task,
                "speculative_grade_task": grade_task,
                "speculative_generate_task": gen_task,
                "speculative_faithfulness_task": faith_task,
                "speculative_rewrite_fallback_task": new_fallback_task,
            }
        except Exception as e:
            print(f"---SPECULATIVE FALLBACK FAILED: {e}---")

    res = await (rewrite_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    return {
        "rewritten_queries": [q.strip() for q in res.split("\\n") if q.strip()],
        "speculative_retrieve_task": None,
        "speculative_grade_task": None,
        "speculative_generate_task": None,
        "speculative_faithfulness_task": None,
        "speculative_rewrite_fallback_task": None,
        "retrieved_chunks": [],
        "context": "",
    }"""

content = content.replace(old_rewrite, new_rewrite)

# Update decompose_query
old_decompose = """
    if speculative_complex and retry_count == 0:
        print("---USING SPECULATIVE COMPLEX TASK---")
        try:
            queries, ret_data, grade_task, gen_task, faith_task = await speculative_complex
            return {
                "sub_queries": queries,
                "speculative_grade_task": grade_task,
                "speculative_generate_task": gen_task,
                "speculative_faithfulness_task": faith_task,
                "retrieved_chunks": ret_data["retrieved_chunks"],
                "context": ret_data["context"],
            }
        except Exception as e:
            print(f"---SPECULATIVE COMPLEX FAILED: {e}---")

    res = await (decompose_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    return {
        "sub_queries": [q.strip() for q in res.split("\\n") if q.strip()],
        "speculative_grade_task": None,
        "speculative_generate_task": None,
        "speculative_faithfulness_task": None,
        "speculative_rewrite_fallback_task": None,
        "retrieved_chunks": [],
        "context": "",
    }"""

new_decompose = """
    if speculative_complex and retry_count == 0:
        print("---USING SPECULATIVE COMPLEX TASK---")
        try:
            queries, retrieve_task, grade_task, gen_task, faith_task = await speculative_complex
            return {
                "sub_queries": queries,
                "speculative_retrieve_task": retrieve_task,
                "speculative_grade_task": grade_task,
                "speculative_generate_task": gen_task,
                "speculative_faithfulness_task": faith_task,
            }
        except Exception as e:
            print(f"---SPECULATIVE COMPLEX FAILED: {e}---")

    res = await (decompose_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    return {
        "sub_queries": [q.strip() for q in res.split("\\n") if q.strip()],
        "speculative_retrieve_task": None,
        "speculative_grade_task": None,
        "speculative_generate_task": None,
        "speculative_faithfulness_task": None,
        "speculative_rewrite_fallback_task": None,
        "retrieved_chunks": [],
        "context": "",
    }"""

content = content.replace(old_decompose, new_decompose)

# Update categorize_question
old_cat = """    async def get_category():
        res = await (categorize_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
        return res.strip().lower()

    async def get_rewritten_and_retrieve():
        return await _do_rewrite_and_retrieve_grade_generate(state["question"])

    async def get_decomposed_and_retrieve():
        res = await (decompose_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
        queries = [q.strip() for q in res.split("\\n") if q.strip()]
        ret_data, grade_task, gen_task, faith_task = await _do_retrieve_grade_generate(queries, state["question"])
        return queries, ret_data, grade_task, gen_task, faith_task

    rewrite_task = asyncio.create_task(get_rewritten_and_retrieve())
    decompose_task = asyncio.create_task(get_decomposed_and_retrieve())
    concise_task = asyncio.create_task(_do_retrieve_grade_generate([state["question"]], state["question"]))

    category = await get_category()

    def cancel_concise_task():
        if concise_task.done():
            try:
                _, g_task, gen_task, f_task = concise_task.result()
                g_task.cancel()
                gen_task.cancel()
                f_task.cancel()
            except Exception:
                pass
        else:
            concise_task.cancel()

    if category == "vague":
        cancel_concise_task()
        return {
            "category": category,
            "speculative_vague_task": rewrite_task,
            "speculative_rewrite_fallback_task": decompose_task,
        }
    elif category == "complex":
        cancel_concise_task()
        return {
            "category": category,
            "speculative_complex_task": decompose_task,
            "speculative_rewrite_fallback_task": rewrite_task,
        }
    else:
        decompose_task.cancel()
        return {
            "category": category,
            "speculative_concise_task": concise_task,
            "speculative_rewrite_fallback_task": rewrite_task,
        }"""

new_cat = """    async def get_category():
        res = await (categorize_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
        return res.strip().lower()

    async def get_rewritten_and_retrieve():
        return await _start_rewrite_and_retrieve_grade_generate(state["question"])

    async def get_decomposed_and_retrieve():
        res = await (decompose_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
        queries = [q.strip() for q in res.split("\\n") if q.strip()]
        retrieve_task, grade_task, gen_task, faith_task = _start_retrieve_grade_generate(queries, state["question"])
        return queries, retrieve_task, grade_task, gen_task, faith_task

    rewrite_task = asyncio.create_task(get_rewritten_and_retrieve())
    decompose_task = asyncio.create_task(get_decomposed_and_retrieve())
    c_ret, c_grade, c_gen, c_faith = _start_retrieve_grade_generate([state["question"]], state["question"])

    category = await get_category()

    def cancel_concise_tasks():
        c_ret.cancel()
        c_grade.cancel()
        c_gen.cancel()
        c_faith.cancel()

    if category == "vague":
        cancel_concise_tasks()
        return {
            "category": category,
            "speculative_vague_task": rewrite_task,
            "speculative_rewrite_fallback_task": decompose_task,
        }
    elif category == "complex":
        cancel_concise_tasks()
        return {
            "category": category,
            "speculative_complex_task": decompose_task,
            "speculative_rewrite_fallback_task": rewrite_task,
        }
    else:
        decompose_task.cancel()
        return {
            "category": category,
            "speculative_retrieve_task": c_ret,
            "speculative_grade_task": c_grade,
            "speculative_generate_task": c_gen,
            "speculative_faithfulness_task": c_faith,
            "speculative_rewrite_fallback_task": rewrite_task,
        }"""

content = content.replace(old_cat, new_cat)


# Update retrieve_context
old_retrieve_context = """async def retrieve_context(state):
    print("---NODE: RETRIEVE---")
    
    chunks = state.get("retrieved_chunks", [])
    context = state.get("context", "")
    
    if chunks and context:
        print("---USING SPECULATIVE RETRIEVAL---")
        return {"context": context, "retrieved_chunks": chunks}

    queries = state.get("rewritten_queries", []) + state.get("sub_queries", [])
    if not queries:
        queries = [state["question"]]
    return await _do_retrieve(queries)"""

new_retrieve_context = """async def retrieve_context(state):
    print("---NODE: RETRIEVE---")
    
    spec_ret = state.get("speculative_retrieve_task")
    if spec_ret:
        print("---USING SPECULATIVE RETRIEVAL TASK---")
        ret_data = await spec_ret
        return {"context": ret_data["context"], "retrieved_chunks": ret_data["retrieved_chunks"]}
        
    chunks = state.get("retrieved_chunks", [])
    context = state.get("context", "")
    
    if chunks and context:
        print("---USING SPECULATIVE RETRIEVAL---")
        return {"context": context, "retrieved_chunks": chunks}

    queries = state.get("rewritten_queries", []) + state.get("sub_queries", [])
    if not queries:
        queries = [state["question"]]
    return await _do_retrieve(queries)"""

content = content.replace(old_retrieve_context, new_retrieve_context)

# Update relevance_grader
old_grader = """async def relevance_grader(state):
    print("---NODE: RELEVANCE GRADER---")
    
    spec_concise = state.get("speculative_concise_task")
    retry_count = state.get("retry_count", 0)

    if spec_concise and retry_count == 0:
        print("---USING SPECULATIVE CONCISE TASK---")
        retrieved_data, grade_task, generate_task, faith_task = await spec_concise
        chunks = retrieved_data["retrieved_chunks"]
    else:
        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            chunks = [c for c in state.get("context", "").split("\\n\\n") if c.strip()]
            
        if state.get("speculative_grade_task") and state.get("speculative_generate_task") and state.get("speculative_faithfulness_task"):
            print("---USING UPSTREAM SPECULATIVE TASKS---")
            grade_task = state["speculative_grade_task"]
            generate_task = state["speculative_generate_task"]
            faith_task = state["speculative_faithfulness_task"]
        else:
            if not chunks:
                return {"relevance_scores": [], "context": "", "speculative_answer": "", "speculative_faithfulness_task": None}
                
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
            faith_task = asyncio.create_task(_speculative_faithfulness())"""

new_grader = """async def relevance_grader(state):
    print("---NODE: RELEVANCE GRADER---")
    
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        spec_ret = state.get("speculative_retrieve_task")
        if spec_ret:
            print("---USING SPECULATIVE RETRIEVAL TASK IN GRADER---")
            ret_data = await spec_ret
            chunks = ret_data["retrieved_chunks"]
        else:
            chunks = [c for c in state.get("context", "").split("\\n\\n") if c.strip()]
            
    if state.get("speculative_grade_task") and state.get("speculative_generate_task") and state.get("speculative_faithfulness_task"):
        print("---USING UPSTREAM SPECULATIVE TASKS---")
        grade_task = state["speculative_grade_task"]
        generate_task = state["speculative_generate_task"]
        faith_task = state["speculative_faithfulness_task"]
    else:
        if not chunks:
            return {"relevance_scores": [], "context": "", "speculative_answer": "", "speculative_faithfulness_task": None}
            
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
        faith_task = asyncio.create_task(_speculative_faithfulness())"""

content = content.replace(old_grader, new_grader)

with open('app/RAG/nodes.py', 'w') as f:
    f.write(content)
