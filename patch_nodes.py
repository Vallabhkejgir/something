import re

with open("app/RAG/nodes.py", "r") as f:
    content = f.read()

relevance_grader_str = """
async def relevance_grader(state):
    print("---NODE: RELEVANCE GRADER---")
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        chunks = [c for c in state.get("context", "").split("\n\n") if c.strip()]

    if not chunks:
        return {"relevance_scores": [], "context": "", "speculative_answer": ""}

    formatted_chunks = "\\n---\\n".join([f"Chunk {i+1}:\\n{chunk}" for i, chunk in enumerate(chunks)])

    # Fire both concurrently! We want to speculatively generate an answer assuming all chunks are relevant
    grade_coro = (relevance_prompt | llm | StrOutputParser()).ainvoke({
        "question": state["question"],
        "chunks": formatted_chunks,
    })
    
    unfiltered_context = "\\n\\n".join(chunks)
    generate_coro = (prompt | llm | StrOutputParser()).ainvoke({
        "context": unfiltered_context,
        "question": state["question"],
    })

    grade_task = asyncio.create_task(grade_coro)
    generate_task = asyncio.create_task(generate_coro)
    
    # We must wait for grade_task no matter what
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
        return {"context": filtered_context, "relevance_scores": scores, "speculative_answer": ""}
"""

generate_answer_str = """
async def generate_answer(state):
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
    return {"answer": ans}
"""

old_relevance = re.search(r'async def relevance_grader.*?return \{"context": filtered_context, "relevance_scores": scores\}', content, re.DOTALL).group(0)
old_generate = re.search(r'async def generate_answer.*?return \{"answer": ans\}', content, re.DOTALL).group(0)

content = content.replace(old_relevance, relevance_grader_str.strip())
content = content.replace(old_generate, generate_answer_str.strip())

with open("app/RAG/nodes.py", "w") as f:
    f.write(content)
