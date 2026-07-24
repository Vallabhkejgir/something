"""
chunks.py — Main entry point for processing page elements into indexable documents.

Routes each element type to the appropriate chunking strategy and runs all
async tasks concurrently for minimum indexing latency.
"""

import asyncio
import logging
from typing import List

from langchain_core.documents import Document

from app.services.llm_config import llm as fast_llm, GEN_LLM_LIMITER as FAST_LLM_LIMITER
from app.utils.chunking_strategies import chunk_text, chunk_table, chunk_image

logger = logging.getLogger(__name__)


async def process_elements(url: str, title: str, elements: list) -> List[Document]:
    """
    Convert raw page elements (from the Chrome extension) into a list of
    enriched Document objects ready for indexing.

    Processing is parallelised:
      - All image descriptions and table summaries run concurrently via asyncio.gather().
      - Text chunking is synchronous (CPU-bound, fast) and done inline.

    Returns ALL documents (parents + children). The storage layer separates them
    by the `is_parent` metadata flag.
    """
    all_docs: List[Document] = []

    # ── Separate elements by type ─────────────────────────────────────────────
    text_elements  = [el for el in elements if el.get("type") == "text"]
    table_elements = [el for el in elements if el.get("type") == "table"]
    image_elements = [el for el in elements if el.get("type") == "image"]

    logger.info(
        "Processing elements for '%s': %d text, %d tables, %d images",
        title, len(text_elements), len(table_elements), len(image_elements),
    )

    # ── Text: synchronous, fast ───────────────────────────────────────────────
    text_chunks_count = 0
    for el in text_elements:
        content = el.get("content", "").strip()
        heading = el.get("heading", "").strip()
        if content:
            text_docs = chunk_text(
                text=content,
                source_url=url,
                page_title=title,
                section_heading=heading,
            )
            all_docs.extend(text_docs)
            text_chunks_count += len(text_docs)

    if text_chunks_count > 0:
        logger.debug("Text chunks produced: %d", text_chunks_count)

    # ── Tables & Images: concurrent async tasks ───────────────────────────────
    async_tasks = []

    for idx, el in enumerate(table_elements):
        md = el.get("content", "")
        if md.strip():
            async_tasks.append(
                chunk_table(
                    table_markdown=md,
                    source_url=url,
                    page_title=title,
                    table_index=idx,
                    fast_llm=fast_llm,
                    fast_limiter=FAST_LLM_LIMITER,
                )
            )

    for el in image_elements:
        img_url   = el.get("url", "")
        alt_text  = el.get("alt", "")
        base64    = el.get("base64", "")
        if base64 or img_url:
            async_tasks.append(
                chunk_image(
                    base64_data=base64,
                    img_url=img_url,
                    alt_text=alt_text,
                    source_url=url,
                    page_title=title,
                    fast_llm=fast_llm,
                    fast_limiter=FAST_LLM_LIMITER,
                )
            )

    if async_tasks:
        results = await asyncio.gather(*async_tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                logger.warning("Async chunking task failed: %s", res)
                continue
            if res is None:
                continue
            if isinstance(res, list):
                all_docs.extend(res)
            elif isinstance(res, Document):
                all_docs.append(res)

    logger.info(
        "process_elements complete: %d total chunks for '%s'",
        len(all_docs), title,
    )
    return all_docs
