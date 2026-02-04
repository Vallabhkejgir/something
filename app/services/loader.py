from langchain_community.document_loaders import WebBaseLoader

def Doc_loader(url=None):
    """
    Load documents from a URL using WebBaseLoader.
    If no URL is provided, uses the default example URL.
    """
    if url is None:
        url = ""
    
    print(f"📄 Loading documents from: {url}")
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        print(f"✅ Loaded {len(docs)} document(s).")
        return docs
    except Exception as e:
        print(f"❌ Error loading documents: {str(e)}")
        raise