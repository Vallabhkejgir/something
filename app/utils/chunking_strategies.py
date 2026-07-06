"""
chunking_strategies.py — Structure-aware chunking strategies for text, tables, and images.

Strategies:
  - SemanticTextChunker   : Groups sentences by embedding similarity for coherent paragraphs.
                            Falls back to RecursiveCharacterTextSplitter for short texts.
  - ParentChildChunker    : Produces large parent + small child pairs for two-level retrieval.
  - TableChunker          : Preserves whole tables; generates LLM summary for retrieval.
  - ImageChunker          : Enhances image descriptions and classifies image type.
"""

import asyncio
import logging
import uuid
from typing import Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.metadata import enrich, enrich_batch

logger = logging.getLogger(__name__)

# ── Splitter Configuration ────────────────────────────────────────────────────
_PARENT_CHUNK_SIZE = 1500
_CHILD_CHUNK_SIZE  = 400
_CHILD_OVERLAP     = 80

_MARKDOWN_SEPARATORS = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "]

_parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_PARENT_CHUNK_SIZE,
    chunk_overlap=200,
    separators=_MARKDOWN_SEPARATORS,
)

_child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_CHILD_CHUNK_SIZE,
    chunk_overlap=_CHILD_OVERLAP,
    separators=_MARKDOWN_SEPARATORS,
)


# ── Text Chunker ──────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    source_url: str,
    page_title: str,
    section_heading: str = "",
) -> list[Document]:
    """
    Parent-Child chunking for text content.

    For each parent chunk, produce N child chunks.
    The parent is stored in the doc store for context expansion.
    The children are stored in the vector index for retrieval.

    Both are returned; callers should route them appropriately.
    Each child has `parent_chunk_id` linking to the parent's `chunk_id`.
    """
    if not text or not text.strip():
        return []

    # Prepend heading context so every chunk carries its section identity
    context_prefix = f"[Source: {page_title}]"
    if section_heading:
        context_prefix += f" [{section_heading}]"

    parent_texts = _parent_splitter.split_text(text)
    all_docs: list[Document] = []

    for p_text in parent_texts:
        # Create parent document
        parent_content = f"{context_prefix}\n\n{p_text}"
        parent_doc = Document(page_content=parent_content)
        parent_id = str(uuid.uuid4())
        enrich(
            parent_doc,
            source_url=source_url,
            page_title=page_title,
            element_type="text_parent",
            extra={"chunk_id": parent_id, "is_parent": True},
        )
        # Override chunk_id with pre-assigned parent_id
        parent_doc.metadata["chunk_id"] = parent_id
        all_docs.append(parent_doc)

        # Create child documents for this parent
        child_texts = _child_splitter.split_text(p_text)
        for c_text in child_texts:
            child_content = f"{context_prefix}\n\n{c_text}"
            child_doc = Document(page_content=child_content)
            enrich(
                child_doc,
                source_url=source_url,
                page_title=page_title,
                element_type="text",
                parent_chunk_id=parent_id,
                extra={"is_parent": False},
            )
            all_docs.append(child_doc)

    logger.debug(
        "TextChunker: %d parent(s), %d total chunk(s) for '%s'",
        len(parent_texts), len(all_docs), page_title,
    )
    return all_docs


# ── Table Chunker ─────────────────────────────────────────────────────────────

async def chunk_table(
    table_markdown: str,
    source_url: str,
    page_title: str,
    table_index: int,
    fast_llm,
    fast_limiter,
) -> list[Document]:
    """
    Dual-index a table:
      1. A 'retrieval' document containing a plain-language summary of the table.
      2. A 'display'   document containing the full Markdown table (returned as parent).

    The summary chunk is indexed in the vector store.
    The full-table chunk is stored as the parent for context expansion.
    """
    if not table_markdown or not table_markdown.strip():
        return []

    # Parse column names from the first markdown row
    lines = [l.strip() for l in table_markdown.strip().split("\n") if l.strip()]
    header_line = lines[0] if lines else ""
    columns = [c.strip() for c in header_line.split("|") if c.strip()] if header_line else []
    row_count = max(0, len(lines) - 2)  # exclude header + separator rows

    # Generate LLM-based summary for retrieval
    summary = await _summarise_table(table_markdown, fast_llm, fast_limiter)

    parent_id = str(uuid.uuid4())

    # Parent: full table (stored, not indexed for vector search directly)
    parent_content = f"[Table {table_index} from '{page_title}']\n\n{table_markdown}"
    parent_doc = Document(page_content=parent_content)
    enrich(
        parent_doc,
        source_url=source_url,
        page_title=page_title,
        element_type="table_parent",
        extra={
            "chunk_id": parent_id,
            "is_parent": True,
            "table_index": table_index,
            "column_names": columns,
            "row_count": row_count,
        },
    )
    parent_doc.metadata["chunk_id"] = parent_id

    # Child: LLM summary (indexed in vector store for semantic retrieval)
    retrieval_content = (
        f"[Table {table_index} Summary from '{page_title}']\n\n"
        f"{summary}\n\n"
        f"Columns: {', '.join(columns) if columns else 'N/A'} | Rows: {row_count}"
    )
    child_doc = Document(page_content=retrieval_content)
    enrich(
        child_doc,
        source_url=source_url,
        page_title=page_title,
        element_type="table",
        parent_chunk_id=parent_id,
        extra={
            "is_parent": False,
            "table_index": table_index,
            "column_names": columns,
            "row_count": row_count,
            "table_summary": summary,
        },
    )

    logger.debug("TableChunker: table_%d from '%s' — %d rows, %d cols", table_index, page_title, row_count, len(columns))
    return [parent_doc, child_doc]


async def _summarise_table(markdown: str, fast_llm, fast_limiter) -> str:
    """Ask the fast LLM to produce a 1-2 sentence summary of the table."""
    try:
        await fast_limiter.acquire(300)
        msg = HumanMessage(content=(
            "Summarise this table in 1-2 sentences, describing what it shows "
            "and the key insights or comparisons it contains.\n\n"
            f"Table:\n{markdown[:3000]}"  # Guard against huge tables
        ))
        resp = await fast_llm.ainvoke([msg])
        return resp.content.strip()
    except Exception as e:
        logger.warning("Table summarisation failed: %s", e)
        return f"A data table from the page."


# ── Image Chunker ─────────────────────────────────────────────────────────────

async def chunk_image(
    base64_data: str,
    img_url: str,
    alt_text: str,
    source_url: str,
    page_title: str,
    fast_llm,
    fast_limiter,
) -> Optional[Document]:
    """
    Generate a rich image description and classify the image type.
    Returns a single Document with full metadata for vector indexing.
    """
    if not base64_data and not img_url:
        return None

    description, img_type = await _describe_and_classify_image(
        base64_data, alt_text, fast_llm, fast_limiter
    )

    content = (
        f"[Image from '{page_title}']\n"
        f"Type: {img_type}\n"
        f"Alt text: {alt_text or 'None'}\n"
        f"Description: {description}"
    )

    doc = Document(page_content=content)
    enrich(
        doc,
        source_url=source_url,
        page_title=page_title,
        element_type="image",
        extra={
            "image_url": img_url,
            "image_type": img_type,
            "caption": description,
            "alt_text": alt_text,
        },
    )
    logger.debug("ImageChunker: %s (%s) from '%s'", img_url[:60], img_type, page_title)
    return doc


async def _describe_and_classify_image(
    base64_data: str, alt_text: str, fast_llm, fast_limiter
) -> tuple[str, str]:
    """
    Returns (description, image_type).
    image_type is one of: photo | chart | diagram | screenshot | table_image | icon | other
    """
    if not base64_data:
        return alt_text or "No description available.", "other"

    try:
        await fast_limiter.acquire(600)
        prompt_text = (
            "Analyse this image carefully and do TWO things:\n"
            "1. Classify it as exactly ONE of: photo | chart | diagram | screenshot | table_image | icon | other\n"
            "2. Describe it in detail. If it's a chart/graph, explain the data, axes, trends, and key insights. "
            "   If it's a diagram, explain the relationships and flow. "
            "   If it contains text, include the text verbatim. "
            "   If it's a screenshot, describe what UI or content is shown.\n\n"
            f"Original alt text (may be empty): {alt_text}\n\n"
            "Respond in this exact format:\n"
            "TYPE: <classification>\n"
            "DESCRIPTION: <detailed description>"
        )
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": base64_data}},
            ]
        )
        resp = await fast_llm.ainvoke([message])
        raw = resp.content.strip()

        img_type = "other"
        description = raw

        for line in raw.split("\n"):
            if line.startswith("TYPE:"):
                img_type = line.replace("TYPE:", "").strip().lower()
            elif line.startswith("DESCRIPTION:"):
                description = line.replace("DESCRIPTION:", "").strip()
                # Grab remaining lines too
                idx = raw.find("DESCRIPTION:")
                description = raw[idx + len("DESCRIPTION:"):].strip()
                break

        return description, img_type
    except Exception as e:
        logger.warning("Image description failed: %s", e)
        return alt_text or "Image could not be described.", "other"
