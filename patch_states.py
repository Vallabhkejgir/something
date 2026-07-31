with open('app/RAG/states.py', 'r') as f:
    content = f.read()

content = content.replace('speculative_concise_task: Any', 'speculative_retrieve_task: Any')

with open('app/RAG/states.py', 'w') as f:
    f.write(content)
