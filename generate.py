from graph_types import GraphState
from langchain_core.output_parsers import StrOutputParser
from prompt import prompt
from Global_var import llm, GEN_LLM_LIMITER

async def generate_answer(state: GraphState):
    print("---NODE: GENERATE_ANSWER---")

    prompt_tokens = (
        len(state["question"]) + len(state["context"])
    ) // 4

    await GEN_LLM_LIMITER.acquire(prompt_tokens)

    answer = await (
        prompt
        | llm
        | StrOutputParser()
    ).ainvoke({
        "context": state["context"],
        "question": state["question"]
    })

    return {"answer": answer}
