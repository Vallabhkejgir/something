"""
metadata.py — Centralised metadata enrichment for all document chunks.

Every chunk that enters the vector store is stamped with:
  - chunk_id          : UUID unique per chunk
  - content_hash      : SHA-256 of page_content (for deduplication)
  - element_type      : text | table | image
  - source_url        : origin URL
  - page_title        : page <title>
  - indexed_at        : ISO-8601 timestamp
  - token_count       : approximate token count (char / 4)
  - parent_chunk_id   : UUID linking a child chunk back to its parent (optional)
"""

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Optional
from langchain_core.documents import Document


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def enrich(
    doc: Document,
    source_url: str,
    page_title: str,
    element_type: str,
    parent_chunk_id: Optional[str] = None,
    extra: Optional[dict] = None,
) -> Document:
    """
    Stamp a Document with standardised metadata.
    Returns the same Document object with metadata updated (mutates in place).
    """
    chunk_id = str(uuid.uuid4())
    doc.metadata.update(
        {
            "chunk_id": chunk_id,
            "content_hash": _content_hash(doc.page_content),
            "element_type": element_type,
            "source": source_url,
            "title": page_title,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "token_count": max(1, len(doc.page_content) // 4),
            **({"parent_chunk_id": parent_chunk_id} if parent_chunk_id else {}),
            **(extra or {}),
        }
    )
    return doc


def enrich_batch(
    docs: list[Document],
    source_url: str,
    page_title: str,
    element_type: str,
    parent_chunk_id: Optional[str] = None,
    extra: Optional[dict] = None,
) -> list[Document]:
    """Enrich a list of Documents sharing the same provenance."""
    return [
        enrich(d, source_url, page_title, element_type, parent_chunk_id, extra)
        for d in docs
    ]
