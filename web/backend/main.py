"""FastAPI backend for sec-rag Web UI.

Provides REST APIs and SSE endpoints for:
- File upload and extraction
- Index building with progress
- RAG querying
- Configuration management
- Evaluation visualization
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to Python path so we can import sec_rag
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Add backend directory to path for api imports
BACKEND_DIR = Path(__file__).parent.resolve()
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.config import router as config_router
from api.eval import router as eval_router
from api.index import router as index_router
from api.query import router as query_router
from api.upload import router as upload_router

app = FastAPI(
    title="sec-rag Web UI",
    description="Web interface for subjective testing of sec-rag",
    version="1.0.0",
)

# CORS for Vue frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload_router, prefix="/api/upload", tags=["upload"])
app.include_router(index_router, prefix="/api/index", tags=["index"])
app.include_router(query_router, prefix="/api/query", tags=["query"])
app.include_router(config_router, prefix="/api/config", tags=["config"])
app.include_router(eval_router, prefix="/api/eval", tags=["eval"])


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting sec-rag Web UI backend on http://localhost:8001")
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
