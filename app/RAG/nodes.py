import json
import asyncio
from typing import List
from langchain_core.output_parsers import StrOutputParser
from app.services.llm_config import llm, GEN_LLM_LIMITER
from app.RAG.prompts import (
    prompt,
    rewrite_prompt,
    decompose_prompt,
    categorize_prompt,
    relevance_prompt,
    faithfulness_prompt,
)
from app.services import storage

_MAX_RETRIES = 3


def parse_json_bool_array(text: str, default_length: int) -> List[bool]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            result = []
            for item in parsed:
                if isinstance(item, bool):
                    result.append(item)
                elif isinstance(item, str):
                    result.append(item.lower() in ("true", "yes", "1"))
                else:
                    result.append(bool(item))
            if len(result) == default_length:
                return result
            elif len(result) > 0:
                if len(result) < default_length:
                    result.extend([True] * (default_length - len(result)))
                return result[:default_length]
    except Exception:
        pass
    return [True] * default_length


async def rewrite_query(state):
    print("---NODE: REWRITE---")
    speculative_vague = state.get("speculative_vague_task")
    speculative_fallback = state.get("speculative_rewrite_fallback_task")
    retry_count = state.get("retry_count", 0)
    
    if speculative_vague and retry_count == 0:
        print("---USING SPECULATIVE VAGUE TASK---")
        try:
            queries, retrieve_task, grade_task, gen_task, faith_task = await speculative_vague
            return {
                "rewritten_queries": queries,
                "speculative_retrieve_task": retrieve_task,
                "speculative_grade_task": grade_task,
                "speculative_generate_task": gen_task,
                "speculative_faithfulness_task": faith_task,
            }
        except Exception as e:
            print(f"---SPECULATIVE VAGUE FAILED: {e}---")

    if speculative_fallback:
        print("---USING SPECULATIVE REWRITE FALLBACK---")
        try:
            queries, retrieve_task, grade_task, gen_task, faith_task = await speculative_fallback
            new_fallback_task = asyncio.create_task(_start_rewrite_and_retrieve_grade_generate(state["question"]))
            return {
                "rewritten_queries": queries,
                "speculative_retrieve_task": retrieve_task,
                "speculative_grade_task": grade_task,
                "speculative_generate_task": gen_task,
                "speculative_faithfulness_task": faith_task,
                "speculative_rewrite_fallback_task": new_fallback_task,
            }
        except Exception as e:
            print(f"---SPECULATIVE FALLBACK FAILED: {e}---")

    res = await (rewrite_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    return {
        "rewritten_queries": [q.strip() for q in res.split("\n") if q.strip()],
        "speculative_retrieve_task": None,
        "speculative_grade_task": None,
        "speculative_generate_task": None,
        "speculative_faithfulness_task": None,
        "speculative_rewrite_fallback_task": None,
        "retrieved_chunks": [],
        "context": "",
    }


async def decompose_query(state):
    print("---NODE: DECOMPOSE---")
    speculative_complex = state.get("speculative_complex_task")
    retry_count = state.get("retry_count", 0)
    
    if speculative_complex and retry_count == 0:
        print("---USING SPECULATIVE COMPLEX TASK---")
        try:
            queries, retrieve_task, grade_task, gen_task, faith_task = await speculative_complex
            return {
                "sub_queries": queries,
                "speculative_retrieve_task": retrieve_task,
                "speculative_grade_task": grade_task,
                "speculative_generate_task": gen_task,
                "speculative_faithfulness_task": faith_task,
            }
        except Exception as e:
            print(f"---SPECULATIVE COMPLEX FAILED: {e}---")

    res = await (decompose_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
    return {
        "sub_queries": [q.strip() for q in res.split("\n") if q.strip()],
        "speculative_retrieve_task": None,
        "speculative_grade_task": None,
        "speculative_generate_task": None,
        "speculative_faithfulness_task": None,
        "speculative_rewrite_fallback_task": None,
        "retrieved_chunks": [],
        "context": "",
    }



async def _do_retrieve(queries):
    store = storage.store_manager.get_vector_store()
    if store is None:
        raise ValueError("Vector Store not initialized")

    bm25_retriever = storage.store_manager.bm25_retriever
    retriever = store.as_retriever(search_kwargs={"k": 5})

    all_docs = []

    async def process_query(q):
        dense_coro = retriever.ainvoke(q)
        if bm25_retriever:
            sparse_coro = bm25_retriever.ainvoke(q)
            dense_docs, sparse_results = await asyncio.gather(dense_coro, sparse_coro)
        else:
            dense_docs = await dense_coro
            sparse_results = []
        fused_docs = storage.store_manager.reciprocal_rank_fusion(dense_docs, sparse_results)
        return fused_docs[:5]

    results = await asyncio.gather(*(process_query(q) for q in queries))
    for fused_docs in results:
        all_docs.extend(fused_docs)

    unique_docs = {}
    for d in all_docs:
        chunk_id = d.metadata.get("chunk_id", d.page_content)
        if chunk_id not in unique_docs:
            unique_docs[chunk_id] = d

    unique_contents = []
    for d in unique_docs.values():
        meta = d.metadata
        url = meta.get("source_url", "N/A")
        title = meta.get("document_title", "N/A")
        heading = meta.get("section_heading", "N/A")
        content = f"Source: {url}\nTitle: {title}\nHeading: {heading}\nContent: {d.page_content}"
        unique_contents.append(content)

    context = "\n\n".join(unique_contents)
    return {"context": context, "retrieved_chunks": unique_contents}


def _start_retrieve_grade_generate(queries, question):
    retrieve_task = asyncio.create_task(_do_retrieve(queries))
    
    async def run_grade():
        retrieved_data = await retrieve_task
        chunks = retrieved_data["retrieved_chunks"]
        if not chunks:
            return "[]"
        formatted_chunks = "\n---\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)])
        return await (relevance_prompt | llm | StrOutputParser()).ainvoke({
            "question": question,
            "chunks": formatted_chunks,
        })
        
    grade_task = asyncio.create_task(run_grade())
    
    async def run_generate():
        retrieved_data = await retrieve_task
        chunks = retrieved_data["retrieved_chunks"]
        if not chunks:
            return ""
        unfiltered_context = "\n\n".join(chunks)
        tokens = (len(question) + len(unfiltered_context)) // 4
        await GEN_LLM_LIMITER.acquire(max(tokens, 1))
        return await (prompt | llm | StrOutputParser()).ainvoke({
            "context": unfiltered_context,
            "question": question,
        })
        
    generate_task = asyncio.create_task(run_generate())
    
    async def run_faithfulness():
        try:
            ans = await generate_task
        except asyncio.CancelledError:
            raise
        retrieved_data = await retrieve_task
        chunks = retrieved_data["retrieved_chunks"]
        if not chunks:
            return "yes"
        unfiltered_context = "\n\n".join(chunks)
        return await (faithfulness_prompt | llm | StrOutputParser()).ainvoke({
            "context": unfiltered_context,
            "answer": ans,
        })
        
    faith_task = asyncio.create_task(run_faithfulness())
    
    return retrieve_task, grade_task, generate_task, faith_task

async def _start_rewrite_and_retrieve_grade_generate(question):
    res = await (rewrite_prompt | llm | StrOutputParser()).ainvoke({"question": question})
    queries = [q.strip() for q in res.split("\n") if q.strip()]
    retrieve_task, grade_task, gen_task, faith_task = _start_retrieve_grade_generate(queries, question)
    return queries, retrieve_task, grade_task, gen_task, faith_task

async def retrieve_context(state):
    print("---NODE: RETRIEVE---")
    
    spec_ret = state.get("speculative_retrieve_task")
    if spec_ret:
        print("---USING SPECULATIVE RETRIEVAL TASK---")
        try:
            ret_data = await spec_ret
            return {"context": ret_data["context"], "retrieved_chunks": ret_data["retrieved_chunks"]}
        except Exception as e:
            print(f"---SPECULATIVE RETRIEVAL TASK FAILED: {e}---")
        
    chunks = state.get("retrieved_chunks", [])
    context = state.get("context", "")
    
    if chunks and context:
        print("---USING SPECULATIVE RETRIEVAL---")
        return {"context": context, "retrieved_chunks": chunks}

    queries = state.get("rewritten_queries", []) + state.get("sub_queries", [])
    if not queries:
        queries = [state["question"]]
    return await _do_retrieve(queries)


async def relevance_grader(state):
    print("---NODE: RELEVANCE GRADER---")
    
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        spec_ret = state.get("speculative_retrieve_task")
        if spec_ret:
            print("---USING SPECULATIVE RETRIEVAL TASK IN GRADER---")
            try:
                ret_data = await spec_ret
                chunks = ret_data["retrieved_chunks"]
            except Exception as e:
                print(f"---SPECULATIVE RETRIEVAL TASK FAILED IN GRADER: {e}---")
        if not chunks:
            chunks = [c for c in state.get("context", "").split("\n\n") if c.strip()]
            
    if state.get("speculative_grade_task") and state.get("speculative_generate_task") and state.get("speculative_faithfulness_task"):
        print("---USING UPSTREAM SPECULATIVE TASKS---")
        grade_task = state["speculative_grade_task"]
        generate_task = state["speculative_generate_task"]
        faith_task = state["speculative_faithfulness_task"]
    else:
        if not chunks:
            return {"relevance_scores": [], "context": "", "speculative_answer": "", "speculative_faithfulness_task": None}
            
        formatted_chunks = "\n---\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)])
        async def run_grade():
            return await (relevance_prompt | llm | StrOutputParser()).ainvoke({
                "question": state["question"],
                "chunks": formatted_chunks,
            })
        grade_task = asyncio.create_task(run_grade())
        
        unfiltered_context = "\n\n".join(chunks)
        async def _speculative_generate():
            tokens = (len(state["question"]) + len(unfiltered_context)) // 4
            await GEN_LLM_LIMITER.acquire(max(tokens, 1))
            return await (prompt | llm | StrOutputParser()).ainvoke({
                "context": unfiltered_context,
                "question": state["question"],
            })
        generate_task = asyncio.create_task(_speculative_generate())
        
        async def _speculative_faithfulness():
            try:
                ans = await generate_task
            except asyncio.CancelledError:
                raise
            return await (faithfulness_prompt | llm | StrOutputParser()).ainvoke({
                "context": unfiltered_context,
                "answer": ans,
            })
        faith_task = asyncio.create_task(_speculative_faithfulness())

    if not chunks:
        return {"relevance_scores": [], "context": "", "speculative_answer": "", "speculative_faithfulness_task": None}

    res = await grade_task
    scores = parse_json_bool_array(res, len(chunks))

    relevant_chunks = [chunk for chunk, is_rel in zip(chunks, scores) if is_rel]
    if not relevant_chunks:
        filtered_context = ""
    else:
        filtered_context = "\n\n".join(relevant_chunks)

    if all(scores) and len(scores) == len(chunks):
        # We don't await generation here anymore!
        return {
            "context": filtered_context, 
            "relevance_scores": scores, 
            "speculative_answer": "", 
            "speculative_generate_task": generate_task,
            "speculative_faithfulness_task": faith_task
        }
    else:
        generate_task.cancel()
        faith_task.cancel()
        
        async def run_filtered_generate():
            if not filtered_context.strip():
                return "I don't have enough information in the retrieved context."
            tokens = (len(state["question"]) + len(filtered_context)) // 4
            await GEN_LLM_LIMITER.acquire(max(tokens, 1))
            return await (prompt | llm | StrOutputParser()).ainvoke({
                "context": filtered_context,
                "question": state["question"],
            })
            
        new_gen = asyncio.create_task(run_filtered_generate())
        
        async def run_filtered_faith():
            ans = await new_gen
            if not filtered_context.strip():
                return "yes"
            return await (faithfulness_prompt | llm | StrOutputParser()).ainvoke({
                "context": filtered_context,
                "answer": ans,
            })
            
        new_faith = asyncio.create_task(run_filtered_faith())
        
        return {
            "context": filtered_context, 
            "relevance_scores": scores, 
            "speculative_answer": "", 
            "speculative_generate_task": new_gen,
            "speculative_faithfulness_task": new_faith
        }


async def generate_answer(state):
    print("---NODE: GENERATE---")
    
    speculative_ans = state.get("speculative_answer", "")
    if speculative_ans:
        print("---USING SPECULATIVE ANSWER---")
        return {"answer": speculative_ans}
        
    spec_gen = state.get("speculative_generate_task")
    if spec_gen:
        print("---USING SPECULATIVE GENERATE TASK---")
        ans = await spec_gen
        return {"answer": ans}
        
    if not state.get("context", "").strip():
        return {"answer": "I don't have enough information in the retrieved context."}

    tokens = (len(state["question"]) + len(state.get("context", ""))) // 4
    await GEN_LLM_LIMITER.acquire(max(tokens, 1))

    ans = await (prompt | llm | StrOutputParser()).ainvoke({
        "context": state.get("context", ""),
        "question": state["question"],
    })
    return {"answer": ans}


async def categorize_question(state):
    print("---NODE: CATEGORIZE---")

    async def get_category():
        res = await (categorize_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
        return res.strip().lower()

    async def get_rewritten_and_retrieve():
        return await _start_rewrite_and_retrieve_grade_generate(state["question"])

    async def get_decomposed_and_retrieve():
        res = await (decompose_prompt | llm | StrOutputParser()).ainvoke({"question": state["question"]})
        queries = [q.strip() for q in res.split("\n") if q.strip()]
        retrieve_task, grade_task, gen_task, faith_task = _start_retrieve_grade_generate(queries, state["question"])
        return queries, retrieve_task, grade_task, gen_task, faith_task

    rewrite_task = asyncio.create_task(get_rewritten_and_retrieve())
    decompose_task = asyncio.create_task(get_decomposed_and_retrieve())
    c_ret, c_grade, c_gen, c_faith = _start_retrieve_grade_generate([state["question"]], state["question"])

    category = await get_category()

    def cancel_concise_tasks():
        c_ret.cancel()
        c_grade.cancel()
        c_gen.cancel()
        c_faith.cancel()

    if category == "vague":
        cancel_concise_tasks()
        return {
            "category": category,
            "speculative_vague_task": rewrite_task,
            "speculative_rewrite_fallback_task": decompose_task,
        }
    elif category == "complex":
        cancel_concise_tasks()
        return {
            "category": category,
            "speculative_complex_task": decompose_task,
            "speculative_rewrite_fallback_task": rewrite_task,
        }
    else:
        decompose_task.cancel()
        return {
            "category": category,
            "speculative_retrieve_task": c_ret,
            "speculative_grade_task": c_grade,
            "speculative_generate_task": c_gen,
            "speculative_faithfulness_task": c_faith,
            "speculative_rewrite_fallback_task": rewrite_task,
        }


async def faithfulness_checker(state):
    print("---NODE: FAITHFULNESS CHECKER---")
    
    spec_faith = state.get("speculative_faithfulness_task")
    retry_count = state.get("retry_count", 0)
    
    if spec_faith:
        print("---USING SPECULATIVE FAITHFULNESS---")
        try:
            res = await spec_faith
        except asyncio.CancelledError:
            if not state.get("context", "").strip():
                res = "yes"
            else:
                res = await (faithfulness_prompt | llm | StrOutputParser()).ainvoke({
                    "context": state.get("context", ""),
                    "answer": state.get("answer", ""),
                })
    else:
        if not state.get("context", "").strip():
            res = "yes"
        else:
            res = await (faithfulness_prompt | llm | StrOutputParser()).ainvoke({
                "context": state.get("context", ""),
                "answer": state.get("answer", ""),
            })

    is_faithful = "yes" in res.strip().lower()

    if not is_faithful:
        retry_count += 1
        print(f"---UNFAITHFUL ANSWER DETECTED (retry_count={retry_count})---")
        return {"is_faithful": False, "faithfulness": "unfaithful", "retry_count": retry_count}
    else:
        print("---FAITHFUL ANSWER---")
        return {"is_faithful": True, "faithfulness": "faithful", "retry_count": retry_count}
