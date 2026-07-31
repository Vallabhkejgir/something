import asyncio
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class MyState(TypedDict):
    a: str
    b: List[str]
    c: List[str]
    retry: int

async def start(state): return {"retry": state.get("retry", 0)}
async def get_a(state): return {"a": "vague"}
async def get_b(state): return {"b": ["b1", "b2"]}
async def get_c(state): return {"c": ["c1", "c2"]}
async def merge(state):
    print("merge running, retry:", state.get("retry"))
    if state.get("retry", 0) > 0:
        return {"c": []}
    if state["a"] == "vague":
        return {"c": []}
    return {}
async def check(state):
    r = state.get("retry", 0)
    if r == 0:
        return {"retry": 1}
    return {}

def route_check(state):
    if state.get("retry", 0) == 1 and state.get("a") != "done":
        # Hack to only retry once
        state["a"] = "done" # this won't mutate state properly but for routing logic it's fine
        return "b"
    return "end"

workflow = StateGraph(MyState)
workflow.add_node("start", start)
workflow.add_node("get_a", get_a)
workflow.add_node("get_b", get_b)
workflow.add_node("get_c", get_c)
workflow.add_node("merge", merge)
workflow.add_node("check", check)

workflow.set_entry_point("start")
workflow.add_edge("start", "get_a")
workflow.add_edge("start", "get_b")
workflow.add_edge("start", "get_c")

workflow.add_edge("get_a", "merge")
workflow.add_edge("get_b", "merge")
workflow.add_edge("get_c", "merge")

workflow.add_edge("merge", "check")
workflow.add_conditional_edges("check", route_check, {"b": "get_b", "end": END})

app = workflow.compile()
print(asyncio.run(app.ainvoke({"a": "", "b": [], "c": [], "retry": 0})))
