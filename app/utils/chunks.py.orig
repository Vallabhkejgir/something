import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def chunk_texts(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = splitter.split_documents(docs)

    # Enrich chunks with metadata
    for i, chunk in enumerate(splits):
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        # Default metadata fields if missing
        if "document_title" not in chunk.metadata:
            chunk.metadata["document_title"] = chunk.metadata.get("source", chunk.metadata.get("source_url", "Unknown Document"))
        if "source_url" not in chunk.metadata:
            chunk.metadata["source_url"] = chunk.metadata.get("source", "Unknown URL")
        if "section_heading" not in chunk.metadata:
            chunk.metadata["section_heading"] = "General"

    print(f"Split into {len(splits)} chunks.")
    return splits
