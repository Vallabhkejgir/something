with open("app/RAG/nodes.py", "r") as f:
    content = f.read()

# Fix the broken string splitting
import re
new_content = re.sub(r'split\("\n\n"\)', r'split("\\n\\n")', content)

with open("app/RAG/nodes.py", "w") as f:
    f.write(new_content)
