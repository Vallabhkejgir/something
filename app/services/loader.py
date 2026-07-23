import os
import tempfile
import urllib.request
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader

def _extract_metadata(element) -> dict:
    metadata = {}
    if hasattr(element, "metadata"):
        m = element.metadata
        if hasattr(m, "to_dict"):
            m = m.to_dict()
        if m.get("filename"):
            metadata["document_title"] = m["filename"]
        if m.get("page_number"):
            metadata["page_number"] = m["page_number"]
        if m.get("url"):
            metadata["source_url"] = m["url"]
    return metadata

def parse_with_unstructured(file_path: str, url: str = None) -> List[Document]:
    from unstructured.partition.auto import partition
    elements = partition(filename=file_path)

    docs = []
    current_section = None

    for el in elements:
        # Check if it's a heading
        el_type = type(el).__name__
        if "Title" in el_type or "Heading" in el_type:
            current_section = str(el)

        text = str(el).strip()
        if not text:
            continue

        metadata = _extract_metadata(el)
        if current_section:
            metadata["section_heading"] = current_section
        if url:
            metadata["source_url"] = url

        docs.append(Document(page_content=text, metadata=metadata))

    return docs

def Doc_loader(url: str = None) -> List[Document]:
    """
    Load documents from a URL or local file path.
    Supports HTTP/HTTPS URLs (via WebBaseLoader or unstructured)
    and local paths (PDF, DOCX, Markdown, etc.) using unstructured.
    """
    if not url:
        return []

    print(f"📄 Loading documents from: {url}")
    try:
        if url.startswith("http://") or url.startswith("https://"):
            if url.endswith((".pdf", ".docx", ".txt", ".md")):
                # Download to temp file and parse with unstructured
                fd, tmp_name = tempfile.mkstemp(suffix=os.path.splitext(url)[1])
                os.close(fd)
                try:
                    urllib.request.urlretrieve(url, tmp_name)
                    docs = parse_with_unstructured(tmp_name, url)
                finally:
                    if os.path.exists(tmp_name):
                        os.unlink(tmp_name)
                print(f"✅ Loaded {len(docs)} document elements via unstructured.")
                return docs
            else:
                loader = WebBaseLoader(url)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source_url"] = url
                print(f"✅ Loaded {len(docs)} document(s) via WebBaseLoader.")
                return docs
        else:
            # Assume it's a local file path
            docs = parse_with_unstructured(url, url)
            print(f"✅ Loaded {len(docs)} document elements via unstructured.")
            return docs

    except Exception as e:
        print(f"❌ Error loading documents: {str(e)}")
        raise