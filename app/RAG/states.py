from typing_extensions import List, TypedDict, Annotated
import operator

class GraphState(TypedDict):
    question: str
    category: str
    rewritten_queries: List[str]
    sub_queries: List[str]
    context: str
    answer: str
    # Annotated with operator.add allows the history to append rather than overwrite
    history: Annotated[List[dict], operator.add]