from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from app.api import app

client = TestClient(app)

@patch("app.api.Doc_loader")
@patch("app.api.process_elements", new_callable=AsyncMock)
@patch("app.api.store_manager.add_documents", new_callable=AsyncMock)
def test_debug(mock_add_docs, mock_chunk, mock_loader):
    # Setup mock doc with page_content and metadata
    mock_doc = MagicMock()
    mock_doc.page_content = "content"
    mock_doc.metadata = {"section_heading": "heading"}
    mock_loader.return_value = [mock_doc]
    mock_chunk.return_value = ["chunk"]

    init_response = client.post("/api/initialize", json={"doc_url": "https://example.com"})
    print(init_response.json())

test_debug()
