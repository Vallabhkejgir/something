with open("app/RAG/nodes.py", "r") as f:
    content = f.read()

content = content.replace("formatted_chunks = \"\n---\n\".join([f\"Chunk {i+1}:\n{chunk}\" for i, chunk in enumerate(chunks)])\n    unfiltered_context = \"\n\n\".join(chunks)", "formatted_chunks = \"\\n---\\n\".join([f\"Chunk {i+1}:\\n{chunk}\" for i, chunk in enumerate(chunks)])\n    unfiltered_context = \"\\n\\n\".join(chunks)")

with open("app/RAG/nodes.py", "w") as f:
    f.write(content)
