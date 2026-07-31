import re

with open("app/RAG/nodes.py", "r") as f:
    content = f.read()

relevance_grader_str = """
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
"""

# Replace the specific block
old_block = """
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
"""

content = content.replace(old_block.strip(), relevance_grader_str.strip())

with open("app/RAG/nodes.py", "w") as f:
    f.write(content)
