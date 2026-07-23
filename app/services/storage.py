from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from app.services.llm_config import embeddings
from rank_bm25 import BM25Okapi
import pickle
import os

# Persistent Qdrant local client
qdrant_client = QdrantClient(path="app/services/qdrant_data")
COLLECTION_NAME = "employee_knowledge_base"

# Global references (will be initialized if exists or created)
vector_store = None
sparse_index = None # For BM25
sparse_docs = None # Store docs for retrieval by BM25

def store_chunks(splits):
    global vector_store, sparse_index, sparse_docs
    print("🚀 Indexing Qdrant and BM25...")

    # Dense Vector Store
    vector_store = Qdrant.from_documents(
        splits,
        embeddings,
        location="app/services/qdrant_data",  # Path to local SQLite/disk persist
        collection_name=COLLECTION_NAME,
        force_recreate=True, # For simplicity of overwriting in this iteration
    )

    # Sparse Vector Store (BM25)
    tokenized_corpus = [doc.page_content.lower().split(" ") for doc in splits]
    sparse_index = BM25Okapi(tokenized_corpus)
    sparse_docs = splits

    # Save BM25 index and docs to disk
    with open("app/services/qdrant_data/bm25_index.pkl", "wb") as f:
        pickle.dump(sparse_index, f)
    with open("app/services/qdrant_data/bm25_docs.pkl", "wb") as f:
        pickle.dump(sparse_docs, f)

    return vector_store

def get_vector_store():
    global vector_store
    if vector_store is None:
        try:
            # Attempt to load existing
            vector_store = Qdrant(
                client=qdrant_client,
                collection_name=COLLECTION_NAME,
                embeddings=embeddings,
            )
        except Exception:
            pass
    return vector_store

def get_bm25_index():
    global sparse_index, sparse_docs
    if sparse_index is None:
        if os.path.exists("app/services/qdrant_data/bm25_index.pkl") and os.path.exists("app/services/qdrant_data/bm25_docs.pkl"):
            with open("app/services/qdrant_data/bm25_index.pkl", "rb") as f:
                sparse_index = pickle.load(f)
            with open("app/services/qdrant_data/bm25_docs.pkl", "rb") as f:
                sparse_docs = pickle.load(f)
    return sparse_index, sparse_docs

def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    """
    Combines dense and sparse search results using Reciprocal Rank Fusion.
    k is a constant usually set to 60.
    """
    rrf_scores = {}

    # Score dense results
    for rank, doc in enumerate(dense_results):
        doc_content = doc.page_content
        if doc_content not in rrf_scores:
            rrf_scores[doc_content] = {"score": 0.0, "doc": doc}
        rrf_scores[doc_content]["score"] += 1.0 / (rank + 1 + k)

    # Score sparse results
    for rank, doc in enumerate(sparse_results):
        doc_content = doc.page_content
        if doc_content not in rrf_scores:
            rrf_scores[doc_content] = {"score": 0.0, "doc": doc}
        rrf_scores[doc_content]["score"] += 1.0 / (rank + 1 + k)

    # Sort by RRF score descending
    sorted_results = sorted(list(rrf_scores.values()), key=lambda x: x["score"], reverse=True)

    # Return documents
    return [item["doc"] for item in sorted_results]
