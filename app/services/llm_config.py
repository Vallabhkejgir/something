import os
from dotenv import load_dotenv
from app.utils.token_bucket import TokenBucket
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "dummy_key_for_init"
model_name = os.getenv("LLM_MODEL", "gemini-flash-latest")

llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=api_key)

# Global Rate Limiters
GEN_LLM_LIMITER = TokenBucket(max_tokens_per_min=250_000, max_requests_per_min=5)
