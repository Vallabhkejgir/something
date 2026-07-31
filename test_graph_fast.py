import asyncio
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class MyState(TypedDict):
    question: str
    category: str
    rewritten_queries: List[str]
    sub_queries: List[str]
    context: str
    answer: str
    retry_count: int
    is_faithful: bool
    relevance_scores: List[bool]
    retrieved_chunks: List[str]

async def categorize_question(state):
    print("categorize")
    # Simulate both categorizing and initial retrieving
    return {"category": "concise", "context": "initial context", "retrieved_chunks": ["initial chunk"]}

async def rewrite_query(state):
    print("rewrite")
    return {"rewritten_queries": ["rw1", "rw2"]}

async def decompose_query(state):
    print("decompose")
    return {"sub_queries": ["sub1", "sub2"]}

async def retrieve_context(state):
    print("retrieve")
    return {"context": "new context", "retrieved_chunks": ["new chunk 1", "new chunk 2"]}

async def relevance_grader(state):
    print("relevance")
    return {"context": state["context"], "relevance_scores": [True] * len(state.get("retrieved_chunks", []))}

async def generate_answer(state):
    print("generate")
    return {"answer": "final answer"}

async def faithfulness_checker(state):
    print("faithfulness")
    r = state.get("retry_count", 0)
    if r == 0:
        return {"is_faithful": False, "retry_count": 1}
    return {"is_faithful": True, "retry_count": r}

def route_question(state):
    if state["category"] == "vague": return "vague"
    if state["category"] == "complex": return "complex"
    return "concise"

def route_faithfulness(state):
    if state.get("is_faithful", True) or state.get("retry_count", 0) >= 3:
        return "end"
    return "rewrite"

workflow = StateGraph(MyState)
workflow.add_node("categorize", categorize_question)
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("decompose", decompose_query)
workflow.add_node("retrieve", retrieve_context)
workflow.add_node("relevance_grader", relevance_grader)
workflow.add_node("generate", generate_answer)
workflow.add_node("faithfulness_checker", faithfulness_checker)

workflow.set_entry_point("categorize")

workflow.add_conditional_edges("categorize", route_question, {
    "vague": "rewrite",
    "complex": "decompose",
    "concise": "relevance_grader", # Skip retrieve since categorize already did it
})

workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("decompose", "retrieve")
workflow.add_edge("retrieve", "relevance_grader")
workflow.add_edge("relevance_grader", "generate")
workflow.add_edge("generate", "faithfulness_checker")

workflow.add_conditional_edges("faithfulness_checker", route_faithfulness, {
    "end": END,
    "rewrite": "rewrite",
})

app = workflow.compile()

async def main():
    res = await app.ainvoke({
        "question": "test", "category": "", "rewritten_queries": [], 
        "sub_queries": [], "context": "", "answer": "", 
        "retry_count": 0, "is_faithful": True, "relevance_scores": [], "retrieved_chunks": []
    })
    print(res)

asyncio.run(main())
