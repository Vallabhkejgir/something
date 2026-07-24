from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.services.loader import Doc_loader
from app.utils.chunks import process_elements
from app.services.storage import store_chunks
from app.RAG.graph import rag_app, GraphState

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="app/templates")

initialized = False


class InitRequest(BaseModel):
    doc_url: Optional[str] = ""


class QueryRequest(BaseModel):
    prompt: str


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/status")
async def status():
    return {"initialized": initialized}


@app.post("/api/initialize")
async def initialize(req: InitRequest):
    global initialized
    try:
        url = req.doc_url
        docs = Doc_loader(url)
        chunks = await process_elements(url, "Document Title", docs)
        store_chunks(chunks)
        initialized = True
        return {"status": "success"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/query")
async def query(req: QueryRequest):
    if not initialized:
        return JSONResponse(status_code=400, content={"error": "Load docs first"})

    user_prompt = req.prompt
    inputs = GraphState(
        question=user_prompt,
        category="",
        rewritten_queries=[],
        sub_queries=[],
        context="",
        answer="",
        retry_count=0,
        is_faithful=True,
        relevance_scores=[],
        retrieved_chunks=[],
    )

    try:
        final_state = await rag_app.ainvoke(inputs)
        return {"answer": final_state["answer"]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=5000)
