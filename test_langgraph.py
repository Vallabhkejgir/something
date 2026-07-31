import asyncio
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class MyState(TypedDict):
    a: str
    b: List[str]
    c: List[str]

async def start(state): return {}
async def get_a(state): return {"a": "vague"}
async def get_b(state): return {"b": ["b1", "b2"]}
async def get_c(state): return {"c": ["c1", "c2"]}
async def merge(state):
    if state["a"] == "vague":
        return {"c": []}
    return {}

workflow = StateGraph(MyState)
workflow.add_node("start", start)
workflow.add_node("get_a", get_a)
workflow.add_node("get_b", get_b)
workflow.add_node("get_c", get_c)
workflow.add_node("merge", merge)

workflow.set_entry_point("start")
workflow.add_edge("start", "get_a")
workflow.add_edge("start", "get_b")
workflow.add_edge("start", "get_c")

workflow.add_edge("get_a", "merge")
workflow.add_edge("get_b", "merge")
workflow.add_edge("get_c", "merge")
workflow.add_edge("merge", END)

app = workflow.compile()
print(asyncio.run(app.ainvoke({"a": "", "b": [], "c": []})))
