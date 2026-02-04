from langchain_core.prompts import ChatPromptTemplate

prompt_template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
Context: {context}
Question: {question}
Answer:"""

rewrite_template = """Generate 3 search queries for: {question}. Newline separated."""
decompose_template = """Break down into 2-4 sub-questions if complex: {question}"""

prompt = ChatPromptTemplate.from_template(prompt_template)
rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)
decompose_prompt = ChatPromptTemplate.from_template(decompose_template)
