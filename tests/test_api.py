import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from app.api import app

client = TestClient(app)

def test_index_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>RAG Query Interface</title>" in response.text

def test_status_endpoint():
    response = client.get("/api/status")
    assert response.status_code == 200
    assert "initialized" in response.json()

def test_query_uninitialized():
    response = client.post("/api/query", json={"prompt": "Hello"})
    assert response.status_code == 400
    assert response.json()["error"] == "Load docs first"

@patch("app.api.Doc_loader")
@patch("app.api.chunk_texts")
@patch("app.api.store_chunks")
def test_initialize_and_query(mock_store, mock_chunk, mock_loader):
    mock_loader.return_value = ["doc"]
    mock_chunk.return_value = ["chunk"]

    init_response = client.post("/api/initialize", json={"doc_url": "https://example.com"})
    assert init_response.status_code == 200
    assert init_response.json()["status"] == "success"

    status_response = client.get("/api/status")
    assert status_response.json()["initialized"] is True

    with patch("app.api.rag_app.ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.return_value = {"answer": "Test answer"}
        query_response = client.post("/api/query", json={"prompt": "Test question"})
        assert query_response.status_code == 200
        assert query_response.json()["answer"] == "Test answer"
