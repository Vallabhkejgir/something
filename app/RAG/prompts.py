from langchain_core.prompts import ChatPromptTemplate

# prompt_template = """You are an assistant for question-answering tasks.
# Use the following pieces of retrieved context to answer the question.
# Context: {context}
# Question: {question}
# Answer:"""

prompt_template = """
You are a question-answering assistant.

Use ONLY the provided context to answer the question.
If the answer cannot be found in the context, say:
"I don't have enough information in the retrieved context."

Context:
{context}

Question:
{question}

Answer in a clear and concise manner.
Do not add external knowledge.
"""


# rewrite_template = """Generate 3 search queries for: {question}. Newline separated."""
# rewrite_template = """
# You are an AI assistant that helps users by generating multiple search queries based on their original question.
# Your goal is to generate 3 diverse versions of the given question to retrieve relevant documents from a vector database.
# Provide these queries as a newline-separated list.

# Original question: {question}
# """

adaptive_planner_template = """
You are the query planner for an Adaptive RAG pipeline.

Create a retrieval plan that preserves the user's exact intent. Do not answer
the question. Return valid JSON only.

Supported strategies:
- multi_query: use when paraphrases or alternate terminology improve recall.
- decompose: use when the question has multiple parts or needs multi-hop lookup.
- grapho1: use when entities, relationships, causes, dependencies, citations,
  workflows, or comparisons matter. Generate entity/relation-focused queries.
- tv_rag: use when time, version, sequence, chronology, validity window, or
  changing facts matter. Generate temporal/version-focused queries.
- hifi_rag: use when the answer must be high precision, grounded, or filtered
  against noisy context. Generate exact-evidence queries.
- affordance_rag: use when the user asks for actions, steps, implementation,
  configuration, code, decisions, or what can be done with the retrieved material.

Rules for retrieval_queries:
- Generate 3 to 6 queries.
- Every query must be a faithful search query for the original question.
- Include the original question as the first query unless it is too vague.
- Prefer specific nouns, entities, constraints, dates, versions, and actions.
- Do not invent facts, names, dates, or technologies not present in the question.
- Avoid generic queries like "overview", "introduction", or "more information".

Return exactly this JSON shape:
{{
  "category": "vague|complex|concise",
  "strategies": ["multi_query"],
  "canonical_question": "clear standalone version of the question",
  "retrieval_queries": ["query one", "query two"],
  "graph_focus": ["entity or relation focus"],
  "temporal_focus": ["time/version focus"],
  "affordance_focus": ["action/capability focus"]
}}

Question:
{question}
"""

rewrite_template = """
Rewrite the user question into focused vector-search queries.

Return valid JSON only:
{{"queries": ["query one", "query two", "query three"]}}

Rules:
- Generate exactly 3 queries.
- Preserve the original meaning and constraints.
- Include specific nouns and entities from the question.
- Do not introduce unrelated topics.
- Do not number the queries.

Question:
{question}
"""


# decompose_template = """Break down into 2-4 sub-questions if complex: {question}"""
# decompose_template = """
# You are an assistant that helps break down complex questions into smaller,
# more manageable sub-questions.

# If it is complex, return 2-4 sub-questions that together can answer the original.

# Original Question: {question}
# """

decompose_template = """
You are an assistant that helps break down complex questions into smaller, more manageable sub-questions.

If the question is complex:
- Generate 2 to 4 independent sub-questions
- Each sub-question should be answerable separately but not too broad, featuring specific focus on parts of the original question.
- Together they must fully answer the original question

If the question is simple, return the original question unchanged.

Return ONLY the list of sub-questions, one per line.

Original question:
{question}
"""



# categorize_template = """You are an assistant that categorizes user questions for a RAG system.
# Analyze the following question and classify it into one of three categories:

# 1. "vague": The request is unclear, too short, or lacks context (e.g., "tell me more", "how does it work"). Needs rewriting.
# 2. "complex": The question has multiple parts or requires a multi-step explanation. Needs decomposition.
# 3. "concise": The question is specific, clear, and can be answered directly with a single search.

# Question: {question}

# Return only one word: "vague", "complex", or "concise"."""

categorize_template = """
Classify the user question for a RAG pipeline.

Categories:
- vague: unclear, missing subject or lacks context, or too short to retrieve documents. Needs rewriting
- complex: requires multiple steps explanation or has multiple parts. Needs decomposition
- concise: clear, focused, single-intent question

Question:
{question}

Respond with exactly one word:
vague OR complex OR concise
"""


categorize_prompt = ChatPromptTemplate.from_template(categorize_template)


prompt = ChatPromptTemplate.from_template(prompt_template)
adaptive_planner_prompt = ChatPromptTemplate.from_template(adaptive_planner_template)
rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)
decompose_prompt = ChatPromptTemplate.from_template(decompose_template)
