from langgraph.graph import START, END, StateGraph
from app.RAG.states import GraphState
from app.RAG.nodes import (
    rewrite_query,
    decompose_query,
    retrieve_context,
    relevance_grader,
    generate_answer,
    categorize_question,
    faithfulness_checker,
    _MAX_RETRIES,
)


def route_question(state: GraphState):
    category = state.get("category")

    if category == "vague":
        return "vague"
    elif category == "complex":
        return "complex"
    else:
        return "concise"


def route_faithfulness(state: GraphState):
    is_faithful = state.get("is_faithful", True)
    retry_count = state.get("retry_count", 0)

    if is_faithful or retry_count >= _MAX_RETRIES:
        return "end"
    return "rewrite"


workflow = StateGraph(GraphState)

workflow.add_node("categorize", categorize_question)
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("decompose", decompose_query)
workflow.add_node("retrieve", retrieve_context)
workflow.add_node("relevance_grader", relevance_grader)
workflow.add_node("generate", generate_answer)
workflow.add_node("faithfulness_checker", faithfulness_checker)

workflow.set_entry_point("categorize")

workflow.add_conditional_edges(
    "categorize",
    route_question,
    {
        "vague": "rewrite",
        "complex": "decompose",
        "concise": "retrieve",
    },
)

workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("decompose", "retrieve")
workflow.add_edge("retrieve", "relevance_grader")
workflow.add_edge("relevance_grader", "generate")
workflow.add_edge("generate", "faithfulness_checker")

workflow.add_conditional_edges(
    "faithfulness_checker",
    route_faithfulness,
    {
        "end": END,
        "rewrite": "rewrite",
    },
)

rag_app = workflow.compile()
