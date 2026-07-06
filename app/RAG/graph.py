"""
graph.py — LangGraph pipeline definition for the advanced RAG system.

Graph Topology:
    [input_guard] ─── blocked ──→ [blocked_response] ──→ END
         │
         └── analyse ──→ [analyse_query] ──→ [adaptive_retrieve]
                                                     │
                                              [relevance_grader]
                                                     │
                         ┌───── generate ────────────┤
                         │                           ├── transform ──→ [query_transformer] ──┐
                         │                           │                                       │
                         │                      fallback ──→ [fallback_response] ──→ END    │
                         │                                                                   │
                    [generate_answer] ◄───────────────────────────────────────────────────────┘
                         │
                   [faithfulness_checker]
                         │
              ┌── output_guard ──→ [output_guard] ──→ END
              │
              └── regenerate ──→ [generate_answer] (retry with incremented count)
"""

from langgraph.graph import END, StateGraph

from app.RAG.states import GraphState
from app.RAG.nodes import (
    input_guard,
    analyse_query_node,
    adaptive_retrieve,
    relevance_grader,
    query_transformer,
    generate_answer,
    faithfulness_checker,
    output_guard,
    fallback_response,
    blocked_response,
    route_after_input_guard,
    route_after_grading,
    route_after_faithfulness,
)

# ── Build Graph ───────────────────────────────────────────────────────────────

workflow = StateGraph(GraphState)

# Register all nodes
workflow.add_node("input_guard",          input_guard)
workflow.add_node("analyse_query",        analyse_query_node)
workflow.add_node("adaptive_retrieve",    adaptive_retrieve)
workflow.add_node("relevance_grader",     relevance_grader)
workflow.add_node("query_transformer",    query_transformer)
workflow.add_node("generate_answer",      generate_answer)
workflow.add_node("faithfulness_checker", faithfulness_checker)
workflow.add_node("output_guard",         output_guard)
workflow.add_node("fallback_response",    fallback_response)
workflow.add_node("blocked_response",     blocked_response)

# ── Entry point ───────────────────────────────────────────────────────────────
workflow.set_entry_point("input_guard")

# ── Edges ─────────────────────────────────────────────────────────────────────

# After input guard: branch to analyse or block
workflow.add_conditional_edges(
    "input_guard",
    route_after_input_guard,
    {"blocked": "blocked_response", "analyse": "analyse_query"},
)

# Linear: analyse → retrieve → grade
workflow.add_edge("analyse_query",     "adaptive_retrieve")
workflow.add_edge("adaptive_retrieve", "relevance_grader")

# After grading: generate, transform (retry), or fallback
workflow.add_conditional_edges(
    "relevance_grader",
    route_after_grading,
    {"generate": "generate_answer", "transform": "query_transformer", "fallback": "fallback_response"},
)

# After query transform: loop back to retrieve with new query
workflow.add_edge("query_transformer", "adaptive_retrieve")

# Linear: generate → faithfulness check
workflow.add_edge("generate_answer", "faithfulness_checker")

# After faithfulness: output guard (pass) or regenerate (retry)
workflow.add_conditional_edges(
    "faithfulness_checker",
    route_after_faithfulness,
    {"output_guard": "output_guard", "regenerate": "generate_answer"},
)

# Termination edges
workflow.add_edge("output_guard",      END)
workflow.add_edge("fallback_response", END)
workflow.add_edge("blocked_response",  END)

# ── Compile ───────────────────────────────────────────────────────────────────
rag_app = workflow.compile()
