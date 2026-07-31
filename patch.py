import re

with open("app/RAG/nodes.py", "r") as f:
    content = f.read()

content = content.replace('res.split("\n") if q.strip()]', r'res.split("\n") if q.strip()]')

with open("app/RAG/nodes.py", "w") as f:
    f.write(content)
