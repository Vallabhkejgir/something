from typing import List, TypedDict, Any

class GraphState(TypedDict, total=False):
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
    speculative_vague_task: Any
    speculative_complex_task: Any
    speculative_concise_task: Any
    speculative_grade_task: Any
    speculative_generate_task: Any
    speculative_faithfulness_task: Any
    speculative_rewrite_fallback_task: Any
