import re

with open("app/services/storage.py", "r") as f:
    content = f.read()

content = content.replace("await asyncio.to_thread(self.vector_store.add_documents, index_docs)", "await self.vector_store.aadd_documents(index_docs)")

with open("app/services/storage.py", "w") as f:
    f.write(content)

