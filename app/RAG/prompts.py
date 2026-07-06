"""
prompts.py — All prompt templates for the RAG pipeline nodes.

Prompts are grouped by node and designed for specific objectives:
  RELEVANCE_GRADING_PROMPT  — binary relevance classification
  GENERATION_PROMPT         — grounded answer generation with citation
  FAITHFULNESS_CHECK_PROMPT — claim verification against context
  QUERY_TRANSFORM_PROMPT    — query rewriting for retry
"""

from langchain_core.prompts import ChatPromptTemplate

# ── Relevance Grading ─────────────────────────────────────────────────────────
# Used by: relevance_grader node (fast_llm, temperature=0)
RELEVANCE_GRADING_PROMPT_STR = """
You are a relevance classifier. Determine whether the document excerpt is relevant to the query.

QUERY: {query}

DOCUMENT:
{document}

Respond with a single JSON object:
{{"relevant": true|false, "reason": "<10 words max>"}}

Rules:
- relevant=true if the document contains information that directly or partially helps answer the query.
- relevant=false only if the document is completely off-topic.
""".strip()

relevance_grading_prompt = ChatPromptTemplate.from_template(RELEVANCE_GRADING_PROMPT_STR)

# ── Answer Generation ─────────────────────────────────────────────────────────
# Used by: generate_answer node (gen_llm, temperature=0.3)
GENERATION_PROMPT_STR = """
You are an expert question-answering assistant for the current webpage.
Your answers are based exclusively on the retrieved context provided below.

RULES:
1. Answer ONLY from the provided context. Never use external knowledge.
2. If the answer is not in the context, respond: "I don't have enough information in the retrieved context to answer this."
3. Cite sources: reference the [number] at the start of each context block when you use it.
4. If the context contains a table (Markdown format), present it as a Markdown table in your answer.
5. If the context contains an image chunk, embed it using: ![Caption](image_url)
6. Be clear, concise, and structured. Use bullet points or numbered lists when appropriate.
7. Do NOT fabricate details, statistics, names, or quotes not present in the context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

generation_prompt = ChatPromptTemplate.from_template(GENERATION_PROMPT_STR)

# ── Faithfulness Check ────────────────────────────────────────────────────────
# Used by: faithfulness_checker node (fast_llm, temperature=0)
FAITHFULNESS_CHECK_PROMPT_STR = """
You are a faithfulness auditor. Verify whether the answer is fully grounded in the context.

CONTEXT (ground truth):
{context}

ANSWER TO VERIFY:
{answer}

Instructions:
1. Extract the key factual claims from the ANSWER.
2. For each claim, determine if it is supported by the CONTEXT.
3. Calculate a faithfulness score: (supported_claims / total_claims).

Respond with a JSON object:
{{
  "supported_claims": ["<claim 1>", ...],
  "unsupported_claims": ["<claim 1>", ...],
  "faithfulness_score": <float 0.0 to 1.0>,
  "verdict": "faithful" | "partially_faithful" | "hallucinated"
}}

If there are no factual claims (e.g., the answer says "I don't know"), set score=1.0 and verdict="faithful".
""".strip()

faithfulness_check_prompt = ChatPromptTemplate.from_template(FAITHFULNESS_CHECK_PROMPT_STR)

# ── Query Transform ───────────────────────────────────────────────────────────
# Used by: query_transformer node (fast_llm, temperature=0.5)
QUERY_TRANSFORM_PROMPT_STR = """
A retrieval system failed to find relevant documents for the query below.
Rewrite it to be more likely to retrieve useful information from a webpage.

ORIGINAL QUERY: {question}

RETRIEVAL FAILURE REASON: The retrieved documents did not contain relevant information.

Write a better search query. Be more specific, use different terminology, or break it into key concepts.
Respond with ONLY the rewritten query — no explanation, no quotes.
""".strip()

query_transform_prompt = ChatPromptTemplate.from_template(QUERY_TRANSFORM_PROMPT_STR)

# ── Legacy alias (used by old tests/imports) ──────────────────────────────────
prompt = generation_prompt
