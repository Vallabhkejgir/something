from typing_extensions import List, TypedDict

class GraphState(TypedDict):
    question: str
    rewritten_queries: List[str]
    sub_queries: List[str]
    context: str
    answer: str
    