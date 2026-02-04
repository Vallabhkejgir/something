from langgraph.graph import START, END, StateGraph
from app.RAG.states import GraphState
from app.RAG.nodes import rewrite_query, decompose_query, retrieve_context, generate_answer

workflow = StateGraph(GraphState)

workflow.add_node("decompose", decompose_query)
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("retrieve", retrieve_context)
workflow.add_node("generate", generate_answer)

workflow.set_entry_point("decompose")
workflow.add_edge("decompose", "rewrite")
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

rag_app = workflow.compile()