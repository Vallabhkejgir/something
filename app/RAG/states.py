from typing import List, TypedDict

class GraphState(TypedDict):
    question: str
    category: str           # Added for routing
    rewritten_queries: List[str]
    sub_queries: List[str]
    context: str
    answer: str
    retry_count: int
    is_faithful: bool
    relevance_scores: List[bool]
    retrieved_chunks: List[str]
    speculative_answer: str
    speculative_rewritten_queries: List[str]
    speculative_sub_queries: List[str]
    speculative_context: str
    speculative_retrieved_chunks: List[str]
