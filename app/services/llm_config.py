import os
from dotenv import load_dotenv
from app.utils.token_bucket import TokenBucket
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

# ── Model names ───────────────────────────────────────────────────────────────
FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", "gemini-3.5-flash")
GEN_LLM_MODEL  = os.getenv("GEN_LLM_MODEL",  "gemini-3.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# ── LLM instances ─────────────────────────────────────────────────────────────
# fast_llm: used for guardrails, relevance grading, faithfulness checks
# High throughput, low latency, temperature=0 for determinism
fast_llm = ChatGoogleGenerativeAI(
    model=FAST_LLM_MODEL,
    temperature=0,
    max_output_tokens=512,
)

# gen_llm: used for final answer generation
# Slightly higher temperature for natural language quality
gen_llm = ChatGoogleGenerativeAI(
    model=GEN_LLM_MODEL,
    temperature=0.3,
    max_output_tokens=2048,
)

# Legacy alias so existing code doesn't break during transition
llm = gen_llm

# Embeddings — used for dense retrieval and semantic cache
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    task_type="retrieval_document",  # Optimised for document indexing
)

query_embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    task_type="retrieval_query",  # Optimised for query-side retrieval
)

# ── Rate Limiters ─────────────────────────────────────────────────────────────
GEN_MAX_TOKENS  = int(os.getenv("GEN_MAX_TOKENS_PER_MIN",  "250000"))
GEN_MAX_REQS    = int(os.getenv("GEN_MAX_REQUESTS_PER_MIN", "15"))
FAST_MAX_TOKENS = int(os.getenv("FAST_MAX_TOKENS_PER_MIN", "500000"))
FAST_MAX_REQS   = int(os.getenv("FAST_MAX_REQUESTS_PER_MIN", "30"))

GEN_LLM_LIMITER  = TokenBucket(max_tokens_per_min=GEN_MAX_TOKENS,  max_requests_per_min=GEN_MAX_REQS)
FAST_LLM_LIMITER = TokenBucket(max_tokens_per_min=FAST_MAX_TOKENS, max_requests_per_min=FAST_MAX_REQS)