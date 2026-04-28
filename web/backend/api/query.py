"""Query API: RAG query endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from api.upload import get_project_dir, is_project_indexed

router = APIRouter()


class QueryRequest(BaseModel):
    project_id: str
    query: str
    top_k: int = 5
    filter_language: str = ""
    filter_chunk_type: str = ""
    project_top_k: int = 3
    embedding_model: str = "Qwen3-VL-Embedding-8B"
    llm_model: str = "Qwen3.5-8B"
    llm_provider: str = "auto"
    api_base_url: str = "http://localhost:8000"
    dry_run: bool = False


class ChunkResult(BaseModel):
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float


class QueryResponse(BaseModel):
    answer: str
    chunks: list[ChunkResult]
    total_prompt_tokens: int
    total_chunk_tokens: int
    candidate_project_ids: list[str]


@router.post("/rag", response_model=QueryResponse)
def query_rag(req: QueryRequest) -> QueryResponse:
    """Execute a RAG query against an indexed project."""
    if not is_project_indexed(req.project_id):
        raise HTTPException(status_code=400, detail="Project not indexed yet")

    project_dir = get_project_dir(req.project_id)
    index_dir = str(project_dir / ".rag_index")

    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from sec_rag.index_store import IndexStore
    from sec_rag.query_pipeline import QueryPipeline, build_embedding_function, build_llm

    try:
        _, embedding_client = build_embedding_function(req.embedding_model)
        llm = None if req.dry_run else build_llm(
            provider=req.llm_provider,
            model=req.llm_model,
            base_url=req.api_base_url,
        )
    except Exception as exc:
        logger.error("Failed to initialize models: {}", exc)
        raise HTTPException(status_code=500, detail=f"Model initialization failed: {exc}")

    index_store = IndexStore(persist_dir=index_dir)
    pipeline = QueryPipeline(index_store, embedding_client, llm)

    try:
        result = pipeline.run(
            query=req.query,
            top_k=req.top_k,
            filter_language=req.filter_language or None,
            filter_chunk_type=req.filter_chunk_type or None,
            project_top_k=req.project_top_k,
        )
    except Exception as exc:
        logger.error("Query failed: {}", exc)
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")

    chunks = [
        ChunkResult(
            chunk_id=c.get("chunk_id", ""),
            text=c.get("text", ""),
            metadata=c.get("metadata", {}),
            score=float(c.get("score", 0.0)),
        )
        for c in result.chunks
    ]

    return QueryResponse(
        answer=result.answer,
        chunks=chunks,
        total_prompt_tokens=result.total_prompt_tokens,
        total_chunk_tokens=result.total_chunk_tokens,
        candidate_project_ids=[req.project_id],
    )


@router.post("/retrieve")
def retrieve_only(req: QueryRequest) -> dict[str, Any]:
    """Retrieve chunks without LLM generation (dry-run)."""
    req.dry_run = True
    result = query_rag(req)
    return {
        "chunks": [c.model_dump() for c in result.chunks],
        "total_prompt_tokens": result.total_prompt_tokens,
        "total_chunk_tokens": result.total_chunk_tokens,
    }
