import re

with open("tests/test_api.py", "r") as f:
    content = f.read()

content = content.replace(
    "@patch(\"app.api.store_chunks\")",
    "@patch(\"app.api.store_manager.add_documents\", new_callable=AsyncMock)"
)
content = content.replace("mock_store, mock_chunk", "mock_add_docs, mock_chunk")

with open("tests/test_api.py", "w") as f:
    f.write(content)
