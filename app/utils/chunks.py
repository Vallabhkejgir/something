from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_texts(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks.")
    return splits
