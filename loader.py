from langchain_community.document_loaders import WebBaseLoader

def Doc_loader():
    # print("Loading documents...")
    loader = WebBaseLoader("https://www.pinecone.io/learn/series/langchain/langchain-expression-language/")
    docs = loader.load()
    # print(f"Loaded {len(docs)} document(s).")
    return docs