import re

with open("app/services/cache.py", "r") as f:
    content = f.read()

content = content.replace("await asyncio.to_thread(_qe.embed_query, query)", "await _qe.aembed_query(query)")

with open("app/services/cache.py", "w") as f:
    f.write(content)

