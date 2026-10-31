"""FastAPI stub for Phase 7: POST /ask with the same interface as graphrag.answer()."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import STRATEGIES, answer

app = FastAPI(title="KG GraphRAG", version="0.1")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    strategy: str = Field(default="B", pattern="^[ABCabc]$")
    kg_version: str = Field(default="resolved", pattern="^(raw|resolved)$")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "strategies": STRATEGIES}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    try:
        return answer(req.question, req.strategy, req.kg_version)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
