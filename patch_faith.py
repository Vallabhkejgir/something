import re

with open("app/RAG/nodes.py", "r") as f:
    content = f.read()

old_faith = """        try:
            res = await spec_faith
        except asyncio.CancelledError:
            if not state.get("context", "").strip():
                res = "yes"
            else:
                res = await (faithfulness_prompt | llm | StrOutputParser()).ainvoke({
                    "context": state.get("context", ""),
                    "answer": state.get("answer", ""),
                })"""

new_faith = """        try:
            res = await spec_faith
        except (asyncio.CancelledError, Exception) as e:
            if isinstance(e, Exception) and not isinstance(e, asyncio.CancelledError):
                print(f"---SPECULATIVE FAITHFULNESS FAILED: {e}---")
            if not state.get("context", "").strip():
                res = "yes"
            else:
                res = await (faithfulness_prompt | llm | StrOutputParser()).ainvoke({
                    "context": state.get("context", ""),
                    "answer": state.get("answer", ""),
                })"""

content = content.replace(old_faith, new_faith)

with open("app/RAG/nodes.py", "w") as f:
    f.write(content)
