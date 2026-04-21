#!/usr/bin/env python3
# ruff: noqa: E402, I001, E501
"""Expanded end-to-end test for .codebase indexing and retrieval."""

from __future__ import annotations

__test__ = False

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loguru import logger

from sec_rag.query_pipeline import build_embedding_function
from test_e2e import (
    CODEBASE_DIR,
    INDEX_DIR,
    cleanup_index,
    collect_project_chunks,
)
from sec_rag.index_store import IndexStore

logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}\n",
)

EMBEDDING_API_URL = "http://192.168.3.91:8081"


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


def verify_embedding_service() -> None:
    logger.info("{}", "=" * 60)
    logger.info("TEST 1: Embedding Function")
    logger.info("{}", "=" * 60)
    embed_fn, _ = build_safe_embed_fn()
    for text in ["GPIO interrupt handling", "FFT computation", "SystemC TLM interconnect"]:
        embedding = embed_fn(text)
        logger.info("  OK Embedded '{}' -> dim={}", text, len(embedding))
    logger.info("")


def build_index():
    logger.info("{}", "=" * 60)
    logger.info("TEST 2: Indexing Projects")
    logger.info("{}", "=" * 60)

    cleanup_index()
    store = IndexStore(str(INDEX_DIR))
    _, safe_embed_fn = build_safe_embed_fn()

    logger.info("Indexing firmware project...")
    firmware_context, firmware_chunks = collect_project_chunks("firmware")
    store.add_chunks("code_index", firmware_chunks, embed_fn=safe_embed_fn)
    logger.info("OK Indexed firmware: {} chunks\n", len(firmware_chunks))

    logger.info("Indexing dsp-core project...")
    dsp_context, dsp_chunks = collect_project_chunks("dsp-core")
    store.add_chunks("code_index", dsp_chunks, embed_fn=safe_embed_fn)
    logger.info("OK Indexed dsp-core: {} chunks\n", len(dsp_chunks))

    return store, firmware_context, dsp_context, firmware_chunks, dsp_chunks


def verify_retrieval(store, firmware_context, dsp_context) -> None:
    logger.info("{}", "=" * 60)
    logger.info("TEST 3: Retrieval by Project")
    logger.info("{}", "=" * 60)

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
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            logger.info(
                "  [{}] {} (file: {})",
                index,
                metadata.get("name", "unnamed"),
                metadata.get("file_path", "unknown"),
            )
        logger.info("")


def verify_statistics(firmware_chunks, dsp_chunks) -> None:
    logger.info("{}", "=" * 60)
    logger.info("TEST 4: Indexing Statistics")
    logger.info("{}", "=" * 60)
    logger.info("Firmware project:")
    logger.info("  Total chunks: {}", len(firmware_chunks))
    logger.info("  Sanitized: {}", sum(1 for c in firmware_chunks if c.metadata.sanitized))
    logger.info("  Raw: {}", sum(1 for c in firmware_chunks if not c.metadata.sanitized))
    logger.info("DSP-Core project:")
    logger.info("  Total chunks: {}", len(dsp_chunks))
    logger.info("  Sanitized: {}", sum(1 for c in dsp_chunks if c.metadata.sanitized))
    logger.info("  Raw: {}", sum(1 for c in dsp_chunks if not c.metadata.sanitized))
    logger.info("Total indexed chunks: {}", len(firmware_chunks) + len(dsp_chunks))
    logger.info("")


def main() -> None:
    logger.info("\nStarting end-to-end tests with llama-server embeddings\n")
    verify_embedding_service()
    store, firmware_context, dsp_context, firmware_chunks, dsp_chunks = build_index()
    verify_statistics(firmware_chunks, dsp_chunks)
    verify_retrieval(store, firmware_context, dsp_context)
    logger.info("{}", "=" * 60)
    logger.info("All tests completed successfully")
    logger.info("{}", "=" * 60)
    logger.info("Index stored at: {}", INDEX_DIR)
    logger.info("Codebase root: {}", CODEBASE_DIR)


if __name__ == "__main__":
    main()
