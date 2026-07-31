import asyncio
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END

class State(TypedDict):
    task: Any
    result: str

async def node_a(state):
    async def background():
        await asyncio.sleep(1)
        return "hello from task"
    
    task = asyncio.create_task(background())
    return {"task": task}

async def node_b(state):
    task = state["task"]
    res = await task
    return {"result": res}

workflow = StateGraph(State)
workflow.add_node("a", node_a)
workflow.add_node("b", node_b)
workflow.add_edge("a", "b")
workflow.set_entry_point("a")
app = workflow.compile()

async def main():
    res = await app.ainvoke({"task": None, "result": ""})
    print("Result:", res)

asyncio.run(main())
