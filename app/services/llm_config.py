import os
from dotenv import load_dotenv
from app.utils.token_bucket import TokenBucket
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

# Models
model_name = os.getenv("LLM_MODEL", "gemini-1.5-flash")
llm = ChatGoogleGenerativeAI(model=model_name)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Global Rate Limiters
GEN_LLM_LIMITER = TokenBucket(max_tokens_per_min=250_000, max_requests_per_min=5)