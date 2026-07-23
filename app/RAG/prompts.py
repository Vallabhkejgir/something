from langchain_core.prompts import ChatPromptTemplate

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

rewrite_template = """
You generate search queries for a vector database.

Generate exactly 3 alternative search queries that:
- Preserve the original meaning
- Use different wording or focus on different aspects
- Are concise (max 12 words each)

Return ONLY a newline-separated list. No numbering. No explanations.

Original question:
{question}
"""

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

relevance_template = """
You are a grader assessing relevance of retrieved document chunks to a user question.

User Question:
{question}

Retrieved Document Chunks:
{chunks}

For each chunk, determine if it is relevant to answering the user question.
Return ONLY a JSON array of boolean values (true or false) corresponding to each chunk in order (e.g. [true, false, true]).
Do NOT include markdown formatting, explanations, or any other text.
"""

faithfulness_template = """
You are a grader assessing whether an answer is grounded in / faithful to a given context.

Context:
{context}

Answer:
{answer}

Does the answer rely ONLY on the provided context without introducing fabricated facts or hallucinated information?
Respond with exactly one word: "yes" if faithful, or "no" if unfaithful.
"""

categorize_prompt = ChatPromptTemplate.from_template(categorize_template)
prompt = ChatPromptTemplate.from_template(prompt_template)
rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)
decompose_prompt = ChatPromptTemplate.from_template(decompose_template)
relevance_prompt = ChatPromptTemplate.from_template(relevance_template)
faithfulness_prompt = ChatPromptTemplate.from_template(faithfulness_template)
