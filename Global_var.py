import os
from dotenv import load_dotenv
from Classes.TokenBucket import TokenBucket
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


load_dotenv()

if os.getenv("GOOGLE_API_KEY"):
    print("✅ Google API Key loaded")
else:
    print("❌ Google API Key missing")

# Global Rate Limiters for Google Generative AI API
# Gemini Embeddings

model_name = os.getenv("LLM_MODEL")
llm = ChatGoogleGenerativeAI(model=model_name)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")



# Gemini Flash (generation)
GEN_LLM_LIMITER = TokenBucket(
    max_tokens_per_min=250_000,
    max_requests_per_min=5
)

