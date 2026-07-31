import re

with open("tests/test_api.py", "r") as f:
    content = f.read()

content = content.replace("@patch(\"app.api.process_elements\")", "@patch(\"app.api.process_elements\", new_callable=AsyncMock)")

with open("tests/test_api.py", "w") as f:
    f.write(content)
