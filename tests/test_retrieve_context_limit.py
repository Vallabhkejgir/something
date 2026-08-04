import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.documents import Document

from app.RAG.nodes import retrieve_context

@pytest.mark.anyio
async def test_retrieve_context_limits_to_7_chunks():
    mock_docs = [Document(page_content=f"Doc {i}", metadata={"chunk_id": str(i)}) for i in range(10)]
    
    with patch("app.RAG.nodes.storage.store_manager") as mock_manager, \
         patch("app.RAG.nodes.rerank") as mock_rerank:
        
        mock_retriever = AsyncMock()
        mock_retriever.ainvoke.return_value = mock_docs
        
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever
        mock_manager.get_vector_store.return_value = mock_store
        
        mock_manager.bm25_retriever = None
        mock_manager.reciprocal_rank_fusion.return_value = mock_docs
        
        mock_rerank.return_value = (mock_docs[:7], [1.0] * 7)
        
        state = {"question": "What is life?"}
        res = await retrieve_context(state)
        
        mock_rerank.assert_called_once()
        kwargs = mock_rerank.call_args.kwargs
        assert kwargs.get("top_n") == 7, "Should pass top_n=7 to rerank"
        
        assert len(res["retrieved_chunks"]) == 7, "Output should be limited to 7 chunks"
