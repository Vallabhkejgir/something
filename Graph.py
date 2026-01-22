from langgraph.graph import START, END, StateGraph
from graph_types import GraphState
from Query_Rewrite import rewrite_query, decompose_query
from retrieval import retrieve_context
from generate import generate_answer

print("\nBuilding graph...")
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


app = workflow.compile()
print("Graph ready")