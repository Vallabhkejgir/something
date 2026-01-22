import asyncio
from langchain_chroma import Chroma
from Classes.TokenBucket import TokenBucket
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from Global_var import embeddings
from langchain_community.vectorstores import FAISS
from Global_var import embeddings

# We define this globally so retrieval.py can access it later
vector_store = None

def store_chunks(splits):
    """
    Takes a list of document chunks and indexes them into a vector store.
    Using FAISS (in-memory) to avoid Windows/SQLite crashes.
    """
    global vector_store
    print("🚀 Starting FAISS Ingestion...")

    if not splits:
        print("⚠️ Warning: No chunks found to store.")
        return

    print(f"   -> Indexing {len(splits)} chunks...")

    # Create the Vector Store in one go (Synchronous)
    # This builds the index in RAM, which is very fast and stable
    vector_store = FAISS.from_documents(splits, embeddings)

    print("   ✅ Indexing complete.")
    print("🎉 All documents stored successfully in memory!")












# ---1. Initialize Vector Store (Same as before)
# vector_store = Chroma(
#     embedding_function=embeddings,
#     collection_name="rag-chroma"
#     # persist_directory="chroma_db"
# )

# # --- 2. The Async Processing Function ---
# async def store_chunks_async(splits):
#     print(" Starting Async Ingestion...")

#     # Initialize our Rate Limiter
#     # Limits: 30,000 TPM, 100 RPM
#     limiter = TokenBucket(max_tokens_per_min=30000, max_requests_per_min=100)
    
#     # Batch Size Strategy: 
#     # Keep it 50 to save your Daily Requests (1,000 RPD limit)
#     BATCH_SIZE = 50 
    
#     tasks = []
    
#     # Create a Semaphore to prevent opening too many connections at once
#     # (Even if we have tokens, we don't want to spawn 100 threads instantly)
#     sem = asyncio.Semaphore(5) 

#     async def process_batch(batch_splits, batch_id):
#         async with sem: # Only allow 5 active uploads at a time
#             # 1. Estimate tokens (4 chars approx 1 token)
#             text_content = "".join([d.page_content for d in batch_splits])
#             token_count = len(text_content) // 4
            
#             # 2. Wait for permission from Rate Limiter
#             await limiter.acquire(token_count)
            
#             # 3. Call the API asynchronously
#             # Note: We use 'aadd_documents' which is the Async version
#             print(f" Sending Batch {batch_id} ({token_count} tokens)...")
#             await vector_store.aadd_documents(batch_splits)
#             print(f" Batch {batch_id} complete.")

#     # Create all tasks
#     total_batches = (len(splits) + BATCH_SIZE - 1) // BATCH_SIZE
#     for i in range(0, len(splits), BATCH_SIZE):
#         batch = splits[i:i+BATCH_SIZE]
#         batch_id = i // BATCH_SIZE + 1
#         tasks.append(process_batch(batch, batch_id))

#     # Run them all
#     await asyncio.gather(*tasks)
#     print(" All documents stored successfully!")
#     print(f"   Total batch stored: {total_batches}")


    
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# import time
# from langchain_chroma import Chroma
# from Global_var import embeddings

# # --- 1. Initialize Vector Store ---
# vector_store = Chroma(
#     embedding_function=embeddings,
#     collection_name="rag-chroma",
#     persist_directory="chroma_db"
# )

# # --- 2. The Synchronous Processing Function ---
# def store_chunks(splits):
#     print("Starting Synchronous Ingestion...")

#     # Batch strategy
#     BATCH_SIZE = 50 
    
#     # Simple Rate Limiting (1 request per second is safe for Google GenAI)
#     REQUEST_DELAY = 1.0 

#     total_batches = (len(splits) + BATCH_SIZE - 1) // BATCH_SIZE
    
#     for i in range(0, len(splits), BATCH_SIZE):
#         batch = splits[i:i+BATCH_SIZE]
#         batch_id = (i // BATCH_SIZE) + 1
        
#         # Calculate tokens for logging
#         text_content = "".join([d.page_content for d in batch])
#         token_count = len(text_content) // 4
        
#         print(f"Sending Batch {batch_id}/{total_batches} ({token_count} tokens)...")
        
#         # --- CRITICAL FIX: Use synchronous 'add_documents' instead of 'aadd_documents' ---
#         # This prevents the Windows Async/GRPC crash
#         vector_store.add_documents(batch)
        
#         print(f"Batch {batch_id} complete.")
        
#         # Sleep briefly to respect rate limits
#         if i + BATCH_SIZE < len(splits):
#             time.sleep(REQUEST_DELAY)

#     print("All documents stored successfully!")

