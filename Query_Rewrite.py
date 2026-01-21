from langgraph.graph import START, END, StateGraph
from graph_types import GraphState
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from prompt import rewrite_prompt, decompose_prompt
from Global_var import llm, GEN_LLM_LIMITER


# def decompose_query(state: GraphState):
#     print("---NODE: DECOMPOSE_QUERY---")
#     q = state["question"]

#     result = decomposer_chain.invoke({"question": q})
#     sub_queries = [s.strip() for s in result.split("\n") if s.strip()]

#     return {"sub_queries": sub_queries, "question": q}

# def rewrite_query(state: GraphState):
#     print("---NODE: REWRITE_QUERY---")
#     original_q = state["question"]

#     rewritten = query_rewriter_chain.invoke({"question": original_q})
#     rewritten_list = [q.strip() for q in rewritten.split("\n") if q.strip()]

#     return {"rewritten_queries": rewritten_list, "question": original_q}


async def rewrite_query(state: GraphState):
    print("---NODE: REWRITE_QUERY---")

    question = state["question"]

    # Rough token estimate (safe side)
    token_estimate = len(question) // 4

    await GEN_LLM_LIMITER.acquire(token_estimate)

    rewritten = await (
        rewrite_prompt
        | llm
        | StrOutputParser()
    ).ainvoke({"question": question})

    rewritten_queries = [
        q.strip() for q in rewritten.split("\n") if q.strip()
    ]

    return {
        "rewritten_queries": rewritten_queries,
        "question": question
    }


async def decompose_query(state: GraphState):
    print("---NODE: DECOMPOSE_QUERY---")

    question = state["question"]
    token_estimate = len(question) // 4

    await GEN_LLM_LIMITER.acquire(token_estimate)

    result = await (
        decompose_prompt
        | llm
        | StrOutputParser()
    ).ainvoke({"question": question})

    sub_queries = [
        s.strip() for s in result.split("\n") if s.strip()
    ]

    return {
        "sub_queries": sub_queries,
        "question": question
    }
