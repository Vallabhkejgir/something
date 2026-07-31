import re

with open("app/RAG/graph.py", "r") as f:
    content = f.read()

# Replace the conditional edge for categorize to point "concise" to "relevance_grader" instead of "retrieve"
old_edges = """workflow.add_conditional_edges(
    "categorize",
    route_question,
    {
        "vague": "rewrite",
        "complex": "decompose",
        "concise": "retrieve",
    },
)"""

new_edges = """workflow.add_conditional_edges(
    "categorize",
    route_question,
    {
        "vague": "rewrite",
        "complex": "decompose",
        "concise": "relevance_grader",
    },
)"""

content = content.replace(old_edges, new_edges)

with open("app/RAG/graph.py", "w") as f:
    f.write(content)
