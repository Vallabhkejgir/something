from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader

def Doc_loader(source: str | None = None, source_type: str = "url"):
    """
    Load documents from various sources.
    source: URL or local file path.
    source_type: "url", "pdf", or "txt".
    """
    if source is None:
        raise ValueError("source cannot be None")
    print(f"📄 Loading documents from {source_type}: {source}")
    try:
        if source_type == "url":
            loader = WebBaseLoader(source)
        elif source_type == "pdf":
            loader = PyPDFLoader(source)
        elif source_type == "txt":
            loader = TextLoader(source)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        docs = loader.load()
        print(f"✅ Loaded {len(docs)} document(s).")
        return docs
    except Exception as e:
        print(f"❌ Error loading documents: {str(e)}")
        raise