import re

with open("tests/test_api.py", "r") as f:
    content = f.read()

mock_setup = """
    mock_doc = MagicMock()
    mock_doc.page_content = "content"
    mock_doc.metadata = {"section_heading": "heading"}
    mock_loader.return_value = [mock_doc]
"""
content = content.replace("    mock_loader.return_value = [\"doc\"]", mock_setup)
content = "from unittest.mock import MagicMock\n" + content

with open("tests/test_api.py", "w") as f:
    f.write(content)
