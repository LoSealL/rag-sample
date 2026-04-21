#!/usr/bin/env python3
# ruff: noqa: E402
"""End-to-end test of sec-rag with llama-server embedding service."""

from __future__ import annotations

__test__ = False

import os
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loguru import logger

from sec_rag.chunkers.code_chunker import CodeChunker
from sec_rag.chunkers.verilog_chunker import VerilogChunker
from sec_rag.index_store import IndexStore
from sec_rag.project_processing import annotate_chunk, create_project_context
from sec_rag.query_pipeline import build_embedding_function

logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}\n",
)

EMBEDDING_API_URL = "http://192.168.3.91:8081"
INDEX_DIR = ROOT_DIR / ".rag_index_test"
CODEBASE_DIR = ROOT_DIR / ".codebase"


def cleanup_index() -> None:
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
        logger.info("Cleaned up old index: {}", INDEX_DIR)


def build_safe_embed_fn():
    embed_fn, _ = build_embedding_function(
        embedding_model_name="nomic-embed-text",
        base_url=EMBEDDING_API_URL,
    )

    def safe_embed_fn(text: str, max_tokens: int = 400) -> list[float]:
        if len(text) > max_tokens * 4:
            import numpy as np

            parts = []
            step = max_tokens * 4
            for offset in range(0, len(text), step):
                parts.append(embed_fn(text[offset : offset + step]))
            return np.mean(parts, axis=0).tolist()
        return embed_fn(text)

    return embed_fn, safe_embed_fn


def test_embedding_function() -> None:
    logger.info("{}", "=" * 70)
    logger.info("TEST 1: Embedding Function")
    logger.info("{}", "=" * 70)

    embed_fn, _ = build_safe_embed_fn()
    for text in [
        "GPIO interrupt handling",
        "FFT computation",
        "SystemC TLM interconnect",
    ]:
        embedding = embed_fn(text)
        logger.info("  OK Embedded '{}' -> dim={}", text, len(embedding))
    logger.info("")


def collect_project_chunks(project_name: str):
    project_root = CODEBASE_DIR / project_name
    context = create_project_context(str(project_root))
    chunks = []

    for root, _, files in os.walk(project_root):
        for filename in sorted(files):
            file_path = Path(root) / filename
            rel_path = file_path.relative_to(project_root)

            if filename.endswith(".cpp"):
                file_chunks = CodeChunker().chunk_file(str(file_path))
            elif filename.endswith(".v"):
                file_chunks = VerilogChunker().chunk_file(str(file_path))
            else:
                continue

            for chunk in file_chunks:
                annotate_chunk(chunk, context, document_kind="code")
            chunks.extend(file_chunks)
            logger.info("    {} -> {} chunks", rel_path.as_posix(), len(file_chunks))

    return context, chunks


def test_indexing():
    logger.info("{}", "=" * 70)
    logger.info("TEST 2: Indexing Projects")
    logger.info("{}", "=" * 70)

    cleanup_index()
    store = IndexStore(str(INDEX_DIR))
    _, safe_embed_fn = build_safe_embed_fn()

    logger.info("Indexing firmware project...")
    firmware_context, firmware_chunks = collect_project_chunks("firmware")
    store.add_chunks("code_index", firmware_chunks, embed_fn=safe_embed_fn)
    logger.info("  OK Indexed {} firmware chunks\n", len(firmware_chunks))

    logger.info("Indexing dsp-core project...")
    dsp_context, dsp_chunks = collect_project_chunks("dsp-core")
    store.add_chunks("code_index", dsp_chunks, embed_fn=safe_embed_fn)
    logger.info("  OK Indexed {} dsp-core chunks\n", len(dsp_chunks))

    return store, firmware_chunks, dsp_chunks, firmware_context, dsp_context


def test_retrieval(store, firmware_context, dsp_context) -> None:
    logger.info("{}", "=" * 70)
    logger.info("TEST 3: Retrieval by Project")
    logger.info("{}", "=" * 70)

    embed_fn, _ = build_safe_embed_fn()
    queries = [
        ("GPIO interrupt handling", firmware_context.project_id),
        ("FFT computation", dsp_context.project_id),
        ("filter bank design", dsp_context.project_id),
    ]

    for query_text, project_id in queries:
        logger.info("Query: '{}' (project: {})", query_text, project_id)
        results = store.query(
            collection_name="code_index",
            query_text=query_text,
            embedding_function=embed_fn,
            top_k=3,
            filter_project_id=project_id,
        )
        if not results:
            logger.info("  (no results)")
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            logger.info(
                "  [{}] {} ({})",
                index,
                metadata.get("name", "unnamed"),
                Path(metadata.get("file_path", "unknown")).name,
            )
        logger.info("")


def test_statistics(firmware_chunks, dsp_chunks) -> None:
    logger.info("{}", "=" * 70)
    logger.info("TEST 4: Indexing Statistics")
    logger.info("{}", "=" * 70)
    logger.info("Firmware total chunks: {}", len(firmware_chunks))
    logger.info("DSP-Core total chunks: {}", len(dsp_chunks))
    logger.info("Total indexed chunks: {}\n", len(firmware_chunks) + len(dsp_chunks))


def main() -> None:
    logger.info("\nStarting end-to-end tests with llama-server embeddings\n")
    test_embedding_function()
    store, fw_chunks, dsp_chunks, fw_ctx, dsp_ctx = test_indexing()
    test_statistics(fw_chunks, dsp_chunks)
    test_retrieval(store, fw_ctx, dsp_ctx)
    logger.info("{}", "=" * 70)
    logger.info("All tests completed successfully")
    logger.info("{}", "=" * 70)
    logger.info("Index stored at: {}", INDEX_DIR)


if __name__ == "__main__":
    main()
