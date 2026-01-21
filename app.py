import asyncio
from Global_var import llm, embeddings, GEN_LLM_LIMITER
from loader import Doc_loader
from chunks import chunk_texts
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from Graph import app

# from langchain_chroma import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.documents import Document
# from typing_extensions import List, TypedDict
# from langgraph.graph import START, END, StateGraph


async def main():
    # Load and chunk documents
    docs = Doc_loader()
    chunks = chunk_texts(docs)

    # Try to store chunks, but continue even if it fails (e.g., due to API key issues)
    try:
        from storage import store_chunks_async
        await store_chunks_async(chunks, embeddings)
        print("✅ Chunks stored successfully")
    except Exception as e:
        print(f"⚠️ Warning: Failed to store chunks - {str(e)}")
        print("Continuing to retrieval and generation...\n")

    query = "Considering the functionalities of RunnableParallel for querying multiple sources and RunnableLambda for custom function integration, design a conceptual LCEL chain that first retrieves a person's birth year from one vector store and their birth month and day from a second, separate vector store. Then, use a RunnableLambda to combine these two pieces of information into a single, complete birthdate string. Finally, analyze the primary advantages and disadvantages of this LCEL approach, particularly regarding code readability and scalability, compared to writing a standard Python function that would sequentially call each retriever and then format the string."

    inputs = {"question": query}

    print("🚀 Starting graph execution...\n")
    
    final_state = await app.ainvoke(inputs)
    print("\n---FINAL ANSWER---")
    print(final_state['answer'])

asyncio.run(main())



