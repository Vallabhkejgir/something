import asyncio
import os
from langchain_chroma import Chroma
from Classes.TokenBucket import TokenBucket
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from Global_var import embeddings

# ---1. Initialize Vector Store (Same as before)
vector_store = Chroma(
    embedding_function=embeddings,
    collection_name="rag-chroma",
    persist_directory="chroma_db"
)

# --- 2. The Async Processing Function ---
async def store_chunks_async(splits, embeddings):
    print("🚀 Starting Async Ingestion...")

    # Initialize our Rate Limiter
    # Limits: 30,000 TPM, 100 RPM
    limiter = TokenBucket(max_tokens_per_min=30000, max_requests_per_min=100)
    
    # Batch Size Strategy: 
    # Keep it 50 to save your Daily Requests (1,000 RPD limit)
    BATCH_SIZE = 50 
    
    tasks = []
    
    # Create a Semaphore to prevent opening too many connections at once
    # (Even if we have tokens, we don't want to spawn 100 threads instantly)
    sem = asyncio.Semaphore(5) 

    async def process_batch(batch_splits, batch_id):
        async with sem: # Only allow 5 active uploads at a time
            # 1. Estimate tokens (4 chars approx 1 token)
            text_content = "".join([d.page_content for d in batch_splits])
            token_count = len(text_content) // 4
            
            # 2. Wait for permission from Rate Limiter
            await limiter.acquire(token_count)
            
            # 3. Call the API asynchronously
            # Note: We use 'aadd_documents' which is the Async version
            print(f"   -> Sending Batch {batch_id} ({token_count} tokens)...")
            await vector_store.aadd_documents(batch_splits)
            print(f"   ✅ Batch {batch_id} complete.")

    # Create all tasks
    total_batches = (len(splits) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(splits), BATCH_SIZE):
        batch = splits[i:i+BATCH_SIZE]
        batch_id = i // BATCH_SIZE + 1
        tasks.append(process_batch(batch, batch_id))

    # Run them all
    await asyncio.gather(*tasks)
    print("🎉 All documents stored successfully!")




# def store_chunks_in_chroma(splits, embeddings):
#     print("Storing chunks in Chroma vector store (in batches)...")

#     vector_store = Chroma(
#         embedding_function=embeddings,
#         collection_name="rag-chroma",
#         persist_directory="chroma_db"  # <--- This saves your DB here
#     )

#     # Process documents in batches
#     batch_size = 15 
#     for i in range(0, len(splits), batch_size):
#         batch = splits[i:i+batch_size]
#         print(f"Adding batch {i//batch_size + 1}/{(len(splits)-1)//batch_size + 1}...")
#         vector_store.add_documents(batch)
#         # The free tier is often limited to ~15 requests per minute.
#         # A 5-second delay is a safe choice to stay under the limit.
#         time.sleep(10)

#     print("Storing complete.")
# ###