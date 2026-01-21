from typing_extensions import List, TypedDict

# --- GRAPH STATE ---
class GraphState(TypedDict):
    question: str
    rewritten_queries: List[str]
    sub_queries: List[str]
    context: List[str]
    answer: str
