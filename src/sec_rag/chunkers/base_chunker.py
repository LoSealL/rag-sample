"""Base chunker abstractions and shared chunk data structures."""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChunkMetadata:
    file_path: str
    language: str
    chunk_type: str
    name: str
    chunk_id: str
    line_start: int
    line_end: int
    project_id: str = ""
    project_name: str = ""
    project_root: str = ""
    document_kind: str = "code"
    sanitized: bool = False
    source_language_family: str = ""
    declaration: str = ""
    semantic_summary: str = ""


@dataclass
class Chunk:
    """A chunk of source/document text with metadata."""

    metadata: ChunkMetadata
    text: str

    def to_dict(self) -> dict:
        return {
            "id": self.metadata.chunk_id,
            "document": self.text,
            "metadata": {
                "file_path": self.metadata.file_path,
                "language": self.metadata.language,
                "chunk_type": self.metadata.chunk_type,
                "name": self.metadata.name,
                "chunk_id": self.metadata.chunk_id,
                "line_start": self.metadata.line_start,
                "line_end": self.metadata.line_end,
                "project_id": self.metadata.project_id,
                "project_name": self.metadata.project_name,
                "project_root": self.metadata.project_root,
                "document_kind": self.metadata.document_kind,
                "sanitized": self.metadata.sanitized,
                "source_language_family": self.metadata.source_language_family,
                "declaration": self.metadata.declaration,
                "semantic_summary": self.metadata.semantic_summary,
            },
        }


class BaseChunker(ABC):
    """Unified chunker API for all chunking implementations."""

    SUPPORTED_EXTENSIONS: set[str] = set()

    def chunk_file(self, file_path: str) -> list[Chunk]:
        """Read a file and chunk it according to the concrete implementation."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.supports_file(file_path):
            return []

        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                source = f.read()
        except Exception:
            return []

        if not source.strip():
            return []

        language = self.language_for_file(file_path, source)
        return self.chunk_source(source, file_path, language)

    def supports_file(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    @abstractmethod
    def language_for_file(self, file_path: str, source: str) -> str:
        """Resolve the metadata language for this file."""

    @abstractmethod
    def chunk_source(self, source: str, file_path: str, language: str) -> list[Chunk]:
        """Chunk an already loaded source string."""

    @staticmethod
    def make_chunk_id(
        file_path: str,
        language: str,
        chunk_type: str,
        name: str,
        index: int = 0,
    ) -> str:
        lang_prefix = {
            "python": "py",
            "cpp": "cpp",
            "verilog": "verilog",
            "systemverilog": "verilog",
            "markdown": "doc",
            "docx": "doc",
        }.get(language, language[:3])
        safe_name = name.replace(":", "_").replace("/", "_")
        path_hash = hashlib.sha256(file_path.encode()).hexdigest()[:8]
        return f"{lang_prefix}:{path_hash}:{chunk_type}:{safe_name}:{index}"

    @classmethod
    def make_chunk(
        cls,
        text: str,
        file_path: str,
        language: str,
        chunk_type: str,
        name: str,
        line_start: int,
        line_end: int,
        index: int = 0,
    ) -> Chunk:
        metadata = ChunkMetadata(
            file_path=file_path,
            language=language,
            chunk_type=chunk_type,
            name=name,
            chunk_id=cls.make_chunk_id(file_path, language, chunk_type, name, index),
            line_start=line_start,
            line_end=line_end,
        )
        return Chunk(metadata=metadata, text=text)
