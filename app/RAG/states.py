from typing_extensions import List, TypedDict

class GraphState(TypedDict):
    question: str
    category: str           # Added for routing
    canonical_question: str
    adaptive_strategies: List[str]
    retrieval_queries: List[str]
    rewritten_queries: List[str]
    sub_queries: List[str]
    graph_focus: List[str]
    temporal_focus: List[str]
    affordance_focus: List[str]
    retrieval_trace: List[str]
    context: str
    answer: str
