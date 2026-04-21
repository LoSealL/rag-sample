#!/usr/bin/env python3
# ruff: noqa: E402
"""快速测试脚本 - 验证 .codebase 与 embedding 服务集成。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sec_rag.chunkers.code_chunker import CodeChunker
from sec_rag.chunkers.verilog_chunker import VerilogChunker
from sec_rag.index_store import IndexStore
from sec_rag.project_processing import annotate_chunk, create_project_context
from sec_rag.query_pipeline import build_embedding_function

EMBEDDING_URL = "http://192.168.3.91:8081"
INDEX_DIR = ROOT_DIR / ".rag_index"
CODEBASE_DIR = ROOT_DIR / ".codebase"


def quick_embed_test() -> bool:
    """测试 embedding 服务。"""
    print("\n测试 embedding 服务...")
    try:
        embed_fn, _ = build_embedding_function(
            embedding_model_name="nomic-embed-text",
            base_url=EMBEDDING_URL,
        )
        embedding = embed_fn("Hello, this is a test embedding")
        print(f"  OK embedding 维度: {len(embedding)}")
        print(f"  OK 服务 URL: {EMBEDDING_URL}")
        print("  OK 模型: nomic-embed-text")
        return True
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


def quick_index() -> IndexStore:
    """快速索引 .codebase 中的两个项目。"""
    print("\n索引 .codebase 测试项目...")

    store = IndexStore(str(INDEX_DIR))
    embed_fn, _ = build_embedding_function(
        embedding_model_name="nomic-embed-text",
        base_url=EMBEDDING_URL,
    )

    def safe_embed(text: str, max_tokens: int = 400) -> list[float]:
        if len(text) > max_tokens * 4:
            import numpy as np

            parts = []
            for offset in range(0, len(text), max_tokens * 4):
                parts.append(embed_fn(text[offset : offset + max_tokens * 4]))
            return np.mean(parts, axis=0).tolist()
        return embed_fn(text)

    total_chunks = 0
    for project_name in ["firmware", "dsp-core"]:
        project_path = CODEBASE_DIR / project_name
        if not project_path.exists():
            continue

        context = create_project_context(str(project_path))
        chunks = []

        for root, _, files in os.walk(project_path):
            for filename in files:
                file_path = Path(root) / filename
                if filename.endswith(".cpp"):
                    file_chunks = CodeChunker().chunk_file(str(file_path))
                elif filename.endswith(".v"):
                    file_chunks = VerilogChunker().chunk_file(str(file_path))
                else:
                    continue

                for chunk in file_chunks:
                    annotate_chunk(chunk, context, document_kind="code")
                chunks.extend(file_chunks)

        if chunks:
            store.add_chunks("code_index", chunks, embed_fn=safe_embed)
            total_chunks += len(chunks)
            print(f"  OK {project_name}: {len(chunks)} chunks")

    print(f"OK 总共索引 {total_chunks} chunks")
    return store


def quick_query(store: IndexStore) -> None:
    """执行示例查询。"""
    print("\n执行示例查询...")

    embed_fn, _ = build_embedding_function(
        embedding_model_name="nomic-embed-text",
        base_url=EMBEDDING_URL,
    )

    for query in ["GPIO interrupt", "FFT algorithm", "filter design"]:
        results = store.query(
            collection_name="code_index",
            query_text=query,
            embedding_function=embed_fn,
            top_k=2,
        )
        print(f"  {query!r} -> {len(results)} 结果")


def main() -> None:
    print("\n" + "=" * 60)
    print("sec-rag .codebase 快速测试")
    print("=" * 60)

    if not quick_embed_test():
        raise SystemExit(1)

    store = quick_index()
    quick_query(store)

    print("\n" + "=" * 60)
    print("快速测试完成")
    print("=" * 60)
    print(f"\n索引位置: {INDEX_DIR}")


if __name__ == "__main__":
    main()
