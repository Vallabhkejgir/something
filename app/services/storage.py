from langchain_community.vectorstores import FAISS
from app.services.llm_config import embeddings

vector_store = None

def store_chunks(splits):
    global vector_store
    print("🚀 Indexing FAISS...")
    vector_store = FAISS.from_documents(splits, embeddings)
    return vector_store
