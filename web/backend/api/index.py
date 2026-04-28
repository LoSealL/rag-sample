"""Index API: build RAG index from uploaded project with SSE progress."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from api.upload import get_project_dir, is_project_indexed, mark_project_indexed

router = APIRouter()


class IndexRequest(BaseModel):
    project_id: str
    languages: str = "python,c,cpp"
    chunk_backend: str = "ast"
    chunk_naive_lines: int = 60
    chunk_overlap_ratio: float = 0.2
    chunk_max_tokens: int = 600
    embedding_model: str = "Qwen3-VL-Embedding-8B"
    llm_model: str = "Qwen3.5-8B"
    llm_provider: str = "auto"
    api_base_url: str = "http://localhost:8000"
    verbose: bool = False


async def _index_progress_generator(req: IndexRequest):
    """Generator that yields SSE events during indexing."""
    import asyncio

    project_dir = get_project_dir(req.project_id)

    # Emit initial status
    yield {"event": "status", "data": json.dumps({"stage": "init", "message": "Initializing..."})}
    await asyncio.sleep(0.1)

    # Check if already indexed
    if is_project_indexed(req.project_id):
        yield {"event": "status", "data": json.dumps({"stage": "done", "message": "Already indexed"})}
        return

    try:
        yield {"event": "status", "data": json.dumps({"stage": "config", "message": "Loading configuration..."})}
        await asyncio.sleep(0.1)

        # Import here to avoid loading heavy deps at startup
        import sys
        project_root = Path(__file__).parent.parent.parent.parent.resolve()
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from sec_rag.cli import configure_logging
        from sec_rag.chunkers.chunker_factory import ChunkerFactory
        from sec_rag.index_store import IndexStore
        from sec_rag.project_processing import (
            SensitiveInfoChecker,
            annotate_chunk,
            build_project_summary_chunk,
            create_project_context,
            sanitize_code_chunk,
            summarize_project,
        )
        from sec_rag.query_pipeline import build_embedding_function, build_llm

        configure_logging(req.verbose)

        index_dir = str(project_dir / ".rag_index")
        index_store = IndexStore(persist_dir=index_dir)
        checker = SensitiveInfoChecker()

        yield {"event": "status", "data": json.dumps({"stage": "embeddings", "message": "Loading embedding model..."})}
        embed_fn, _ = build_embedding_function(req.embedding_model)
        await asyncio.sleep(0.1)

        yield {"event": "status", "data": json.dumps({"stage": "llm", "message": "Loading LLM..."})}
        llm = build_llm(
            provider=req.llm_provider,
            model=req.llm_model,
            base_url=req.api_base_url,
        )
        await asyncio.sleep(0.1)

        chunker = ChunkerFactory(
            llm=llm,
            chunk_backend=req.chunk_backend,
            naive_chunk_lines=req.chunk_naive_lines,
            naive_overlap_ratio=req.chunk_overlap_ratio,
            max_tokens_before_split=req.chunk_max_tokens,
        )

        project = create_project_context(str(project_dir))
        logger.info("Indexing project {} in {}", project.project_name, project.project_root)

        # Discover files
        yield {"event": "status", "data": json.dumps({"stage": "discover", "message": "Discovering files..."})}
        lang_map = {"python": {".py"}, "c": {".c", ".h"}, "cpp": {".cpp", ".cc", ".cxx", ".hpp", ".hh"}}
        selected_exts = set()
        for lang in req.languages.split(","):
            selected_exts.update(lang_map.get(lang.strip(), set()))

        files = [p for p in project_dir.rglob("*") if p.is_file() and p.suffix in selected_exts]
        total_files = len(files)

        yield {"event": "progress", "data": json.dumps({"current": 0, "total": total_files, "message": f"Found {total_files} files"})}
        await asyncio.sleep(0.1)

        batch_size = 10
        total_chunks = 0
        project_documents = []

        for i in range(0, total_files, batch_size):
            batch = files[i:i + batch_size]
            batch_chunks = []

            for file_path in batch:
                try:
                    if not chunker.supports_file(str(file_path)):
                        continue
                    chunks = chunker.chunk_file(str(file_path))
                    if not chunks:
                        continue
                    kind = chunker.document_kind_for_file(str(file_path))
                    if chunker.is_cpp_file(str(file_path)):
                        chunks = [sanitize_code_chunk(chunk, llm, project, checker) for chunk in chunks]
                    for chunk in chunks:
                        annotate_chunk(chunk, project, document_kind=kind)
                    batch_chunks.extend(chunks)
                    project_documents.extend(chunks)
                except Exception as e:
                    logger.warning("Failed to chunk {}: {}", file_path, e)

            if batch_chunks:
                index_store.add_chunks("rag_index", batch_chunks, embed_fn=embed_fn)
                total_chunks += len(batch_chunks)

            current = min(i + len(batch), total_files)
            yield {"event": "progress", "data": json.dumps({"current": current, "total": total_files, "message": f"Processed {current}/{total_files} files"})}
            await asyncio.sleep(0.01)

        if project_documents:
            summary_text = summarize_project(project, project_documents, llm)
            index_store.add_chunks(
                "project_index",
                [build_project_summary_chunk(project, summary_text)],
                embed_fn=embed_fn,
            )
            total_chunks += 1

        mark_project_indexed(req.project_id)

        yield {"event": "done", "data": json.dumps({"total_files": total_files, "total_chunks": total_chunks, "message": "Indexing complete"})}

    except Exception as exc:
        logger.error("Indexing failed: {}", exc)
        yield {"event": "error", "data": json.dumps({"message": str(exc)})}


@router.post("/build")
def build_index(req: IndexRequest):
    """Start index building. Returns SSE stream with progress updates."""
    return EventSourceResponse(_index_progress_generator(req))


@router.get("/build-sse")
def build_index_sse(
    project_id: str,
    languages: str = "python,c,cpp",
    chunk_backend: str = "ast",
):
    """SSE endpoint for index building (GET for EventSource compatibility)."""
    req = IndexRequest(
        project_id=project_id,
        languages=languages,
        chunk_backend=chunk_backend,
    )
    return EventSourceResponse(_index_progress_generator(req))


@router.get("/status/{project_id}")
def index_status(project_id: str) -> dict[str, Any]:
    """Check if a project has been indexed."""
    return {
        "project_id": project_id,
        "indexed": is_project_indexed(project_id),
    }
