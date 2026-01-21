from langchain_core.prompts import ChatPromptTemplate

prompt_template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}

Answer:"""

rewrite_prompt_template = """
You are an AI assistant that helps users by generating multiple search queries based on their original question.
Your goal is to generate 3 diverse versions of the given question to retrieve relevant documents from a vector database.
Provide these queries as a newline-separated list.

Original question: {question}
"""

decompose_prompt_template = """
You are an assistant that helps break down complex questions into smaller,
more manageable sub-questions.

If the question is simple, just return the same question.
If it is complex, return 2-4 sub-questions that together can answer the original.

Original Question: {question}
"""

rewrite_prompt = ChatPromptTemplate.from_template(rewrite_prompt_template)
decompose_prompt = ChatPromptTemplate.from_template(decompose_prompt_template)
prompt = ChatPromptTemplate.from_template(prompt_template)

# query_rewriter_chain = rewrite_prompt | llm | StrOutputParser()
# decomposer_chain = decompose_prompt | llm | StrOutputParser()
