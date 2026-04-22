"""Backward-compatible facade chunker for Python and C/C++/SystemC."""

from __future__ import annotations

from pathlib import Path

from sec_rag.chunkers.base_chunker import Chunk, ChunkMetadata
from sec_rag.chunkers.cpp_chunker import CppChunker
from sec_rag.chunkers.python_chunker import PythonChunker

__all__ = ["CodeChunker", "Chunk", "ChunkMetadata"]


class CodeChunker:
    """Facade that routes files to language-specific chunkers."""

    SUPPORTED_LANGUAGES = {"python", "cpp"}

    def __init__(
        self,
        naive_chunk_lines: int = 60,
        naive_overlap_ratio: float = 0.2,
        max_tokens_before_split: int = 600,
    ):
        self._python = PythonChunker(
            naive_chunk_lines=naive_chunk_lines,
            naive_overlap_ratio=naive_overlap_ratio,
            max_tokens_before_split=max_tokens_before_split,
        )
        self._cpp = CppChunker(
            naive_chunk_lines=naive_chunk_lines,
            naive_overlap_ratio=naive_overlap_ratio,
            max_tokens_before_split=max_tokens_before_split,
        )

    def chunk_file(self, file_path: str) -> list[Chunk]:
        ext = Path(file_path).suffix.lower()
        if ext == ".py":
            return self._python.chunk_file(file_path)
        if ext in CppChunker.SUPPORTED_EXTENSIONS:
            return self._cpp.chunk_file(file_path)
        return []
