from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_texts(docs):
    # print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
    splits = text_splitter.split_documents(docs)
    print(f"Split documents into {len(splits)} chunks.")
    return splits