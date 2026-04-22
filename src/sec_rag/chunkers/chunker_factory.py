"""ChunkerFactory: unified constructor that internally routes to all chunkers."""

from pathlib import Path

from sec_rag.chunkers.base_chunker import BaseChunker, Chunk
from sec_rag.chunkers.cpp_chunker import CppChunker
from sec_rag.chunkers.doc_chunker import DocChunker
from sec_rag.chunkers.llm_chunker import LLMChunker
from sec_rag.chunkers.python_chunker import PythonChunker
from sec_rag.chunkers.verilog_chunker import VerilogChunker

_DOC_EXTENSIONS = {".md", ".markdown", ".docx"}
_CPP_EXTENSIONS = {".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh"}


class ChunkerFactory(BaseChunker):
    """
    Unified chunker facade.

    A single constructor accepts all tuning knobs for every backend
    (AST code chunkers, document chunker, LLM chunker).  Backend selection
    and per-file routing are handled internally.

    Parameters
    ----------
    llm:
        LLM client passed to :class:`LLMChunker`.  Required when
        ``chunk_backend`` is ``"llm"`` or ``"hybrid"``.
    chunk_backend:
        ``"ast"`` — rule-based AST chunking (default).
        ``"llm"`` — LLM semantic chunking for all supported files.
        ``"hybrid"`` — AST first; fall back to LLM when result is empty.
    naive_chunk_lines:
        Lines per naive/fallback chunk (code chunkers).
    naive_overlap_ratio:
        Overlap fraction between consecutive naive chunks.
    max_tokens_before_split:
        Token count above which an AST chunk is split further.
    doc_target_tokens:
        Target tokens per paragraph/section for :class:`DocChunker`.
    llm_max_chunks:
        Max chunks :class:`LLMChunker` may emit per file.
    """

    SUPPORTED_EXTENSIONS = (
        PythonChunker.SUPPORTED_EXTENSIONS
        | CppChunker.SUPPORTED_EXTENSIONS
        | VerilogChunker.SUPPORTED_EXTENSIONS
        | DocChunker.SUPPORTED_EXTENSIONS
    )

    def __init__(
        self,
        *,
        llm=None,
        chunk_backend: str = "ast",
        naive_chunk_lines: int = 60,
        naive_overlap_ratio: float = 0.2,
        max_tokens_before_split: int = 600,
        doc_target_tokens: int = 512,
        llm_max_chunks: int = 40,
    ) -> None:
        self._backend = chunk_backend
        common = dict(
            naive_chunk_lines=naive_chunk_lines,
            naive_overlap_ratio=naive_overlap_ratio,
            max_tokens_before_split=max_tokens_before_split,
        )
        self._ast_chunkers: list[BaseChunker] = [
            PythonChunker(**common),
            CppChunker(**common),
            VerilogChunker(),
            DocChunker(doc_target_tokens=doc_target_tokens),
        ]
        self._llm_chunker: LLMChunker | None = (
            LLMChunker(
                llm=llm,
                target_chunk_lines=naive_chunk_lines,
                overlap_lines=max(1, int(naive_chunk_lines * naive_overlap_ratio)),
                max_chunks=llm_max_chunks,
            )
            if llm is not None
            else None
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def document_kind_for_file(self, file_path: str) -> str:
        """Return ``"doc"`` for document files, ``"code"`` otherwise."""
        return "doc" if Path(file_path).suffix.lower() in _DOC_EXTENSIONS else "code"

    def is_cpp_file(self, file_path: str) -> bool:
        """Return True for C/C++ files that need sanitization."""
        return Path(file_path).suffix.lower() in _CPP_EXTENSIONS

    def _ast_chunk_file(self, file_path: str) -> list[Chunk]:
        for chunker in self._ast_chunkers:
            if chunker.supports_file(file_path):
                return chunker.chunk_file(file_path)
        return []

    # ------------------------------------------------------------------
    # BaseChunker abstract interface
    # ------------------------------------------------------------------

    def language_for_file(self, file_path: str, source: str = "") -> str:
        for chunker in self._ast_chunkers:
            if chunker.supports_file(file_path):
                return chunker.language_for_file(file_path, source)
        return "unknown"

    def chunk_source(
        self, source: str, file_path: str = "", language: str = ""
    ) -> list[Chunk]:
        for chunker in self._ast_chunkers:
            if chunker.supports_file(file_path):
                return chunker.chunk_source(source, file_path, language)
        return []

    def chunk_file(self, file_path: str) -> list[Chunk]:
        if self._backend == "llm" and self._llm_chunker is not None:
            return self._llm_chunker.chunk_file(file_path)
        if self._backend == "hybrid" and self._llm_chunker is not None:
            chunks = self._ast_chunk_file(file_path)
            return chunks if chunks else self._llm_chunker.chunk_file(file_path)
        return self._ast_chunk_file(file_path)
