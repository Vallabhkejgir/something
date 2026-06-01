from langgraph.graph import END, StateGraph
from app.RAG.states import GraphState
from app.RAG.nodes import plan_adaptive_rag, retrieve_context, generate_answer


workflow = StateGraph(GraphState)

workflow.add_node("plan", plan_adaptive_rag)
workflow.add_node("retrieve", retrieve_context)
workflow.add_node("generate", generate_answer)

workflow.set_entry_point("plan")

workflow.add_edge("plan", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

rag_app = workflow.compile()
