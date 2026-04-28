"""Config API: manage RAG configuration."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# Default configuration
_default_config: dict[str, Any] = {
    "match_mode": "vector",  # vector | graph | hybrid
    "top_k": 5,
    "project_top_k": 3,
    "chunk_backend": "ast",  # ast | llm | hybrid
    "chunk_naive_lines": 60,
    "chunk_overlap_ratio": 0.2,
    "chunk_max_tokens": 600,
    "embedding_model": "Qwen3-VL-Embedding-8B",
    "llm_model": "Qwen3.5-8B",
    "llm_provider": "auto",
    "api_base_url": "http://localhost:8000",
    "enable_rerank": False,
    "rerank_model": "BAAI/bge-reranker-base",
}

_current_config = _default_config.copy()


class ConfigModel(BaseModel):
    match_mode: str = "vector"
    top_k: int = 5
    project_top_k: int = 3
    chunk_backend: str = "ast"
    chunk_naive_lines: int = 60
    chunk_overlap_ratio: float = 0.2
    chunk_max_tokens: int = 600
    embedding_model: str = "Qwen3-VL-Embedding-8B"
    llm_model: str = "Qwen3.5-8B"
    llm_provider: str = "auto"
    api_base_url: str = "http://localhost:8000"
    enable_rerank: bool = False
    rerank_model: str = "BAAI/bge-reranker-base"


@router.get("/")
def get_config() -> dict[str, Any]:
    return _current_config.copy()


@router.post("/")
def update_config(cfg: ConfigModel) -> dict[str, Any]:
    global _current_config
    _current_config = cfg.model_dump()
    return _current_config


@router.post("/reset")
def reset_config() -> dict[str, Any]:
    global _current_config
    _current_config = _default_config.copy()
    return _current_config
