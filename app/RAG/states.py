from typing_extensions import List, TypedDict

class GraphState(TypedDict):
    question: str
    category: str           # Added for routing
    rewritten_queries: List[str]
    sub_queries: List[str]
    context: str
    answer: str
    