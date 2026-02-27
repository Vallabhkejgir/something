from langchain_core.prompts import ChatPromptTemplate

# prompt_template = """You are an assistant for question-answering tasks.
# Use the following pieces of retrieved context to answer the question.
# Context: {context}
# Question: {question}
# Answer:"""

prompt_template = """
You are a conversational question-answering assistant.
Use the provided context AND the conversation history to answer the current question.

If the answer cannot be found in the context or history, say:
"I don't have enough information in the retrieved context."

History:
{history}

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

rewrite_template = """
You are an AI assistant that rephrases user questions to be standalone search queries.
Given the conversation history and a follow-up question, rephrase the follow-up question 
into 3 distinct standalone search queries for a vector database.

History:
{history}

Follow-up Question: {question}

Return ONLY a newline-separated list of 3 standalone queries.
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
rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)
decompose_prompt = ChatPromptTemplate.from_template(decompose_template)
