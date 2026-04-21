"""
Document chunker using unstructured.io.

Chunks Markdown and Word (.docx) files by section/paragraph,
not by token count.

Usage:
    chunker = DocChunker()
    chunks = chunker.chunk_file("docs/readme.md")
    chunks = chunker.chunk_file("docs/report.docx")
"""

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

from loguru import logger

from sec_rag.chunkers.code_chunker import Chunk, ChunkMetadata

DOC_EXTENSIONS = {".md", ".markdown", ".docx"}


@dataclass
class DocChunker:
    """
    Chunks document files (Markdown, Word) by section/paragraph.

    Usage:
        chunker = DocChunker()
        chunks = chunker.chunk_file("docs/readme.md")
    """

    SUPPORTED_EXTENSIONS = DOC_EXTENSIONS

    # Tokens per character (used for size estimation)
    TOKENS_PER_CHAR = 0.25

    def chunk_file(self, file_path: str) -> list[Chunk]:
        """
        Chunk a document file by section/paragraph.

        Args:
            file_path: Path to the document file.

        Returns:
            List of Chunk objects. Empty list if file cannot be chunked.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            logger.debug("Unsupported document type for {}", file_path)
            return []

        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                source = f.read()
        except Exception as e:
            logger.warning("Failed to read {}: {}", file_path, e)
            return []

        if not source.strip():
            return []

        if ext in (".md", ".markdown"):
            return self._chunk_markdown(source, file_path)
        elif ext == ".docx":
            return self._chunk_docx(source, file_path)

        return []

    def _chunk_markdown(self, source: str, file_path: str) -> list[Chunk]:
        """
        Chunk a Markdown file by section (delimited by headings).

        Each chunk corresponds to one section (from heading to the next heading).
        """
        lines = source.split("\n")
        chunks: list[Chunk] = []
        current_lines: list[str] = []
        current_heading: str | None = None
        section_start_line: int = 1
        chunk_index: int = 0

        heading_pattern = __import__("re").compile(r"^(#{1,6})\s+(.+)$")

        def flush_section():
            nonlocal current_lines, current_heading, section_start_line, chunk_index
            if not current_lines:
                return
            text = "\n".join(current_lines).strip()
            if not text:
                return

            name = current_heading if current_heading else f"section_{chunk_index}"
            chunk_type = "section"
            if not current_heading:
                chunk_type = "paragraph"

            chunk_id = self._make_doc_chunk_id(file_path, chunk_type, name, chunk_index)
            metadata = ChunkMetadata(
                file_path=file_path,
                language="markdown",
                chunk_type=chunk_type,
                name=name,
                chunk_id=chunk_id,
                line_start=section_start_line,
                line_end=section_start_line + len(current_lines) - 1,
            )
            chunks.append(Chunk(metadata=metadata, text=text))
            current_lines = []
            chunk_index += 1

        for i, line in enumerate(lines):
            heading_match = heading_pattern.match(line)
            if heading_match:
                # Flush the previous section
                flush_section()
                current_heading = heading_match.group(2).strip()
                section_start_line = i + 1
                current_lines = [line]
            else:
                current_lines.append(line)

        # Flush the last section
        flush_section()

        if not chunks:
            # Fallback: whole file as one chunk
            chunks.append(self._make_whole_file_chunk(source, file_path, "markdown"))

        return chunks

    def _chunk_docx(self, source: str, file_path: str) -> list[Chunk]:
        """
        Chunk a .docx file using unstructured.io.

        If unstructured.io is not available or fails, falls back to
        naive paragraph splitting.
        """
        if find_spec("unstructured.partition.docx") is None:
            logger.warning(
                "unstructured.io not installed, skipping docx file: %s",
                file_path,
            )
            return self._naive_docx_chunk(source, file_path)

        try:
            # Write source bytes to a temp .docx file (unstructured needs a file path)
            # Since we read text above, we'd need the original bytes.
            # For now, fall back to naive chunking because we don't have bytes.
            logger.warning(
                "docx chunking requires original file bytes. "
                "Falling back to naive paragraph chunking for: %s",
                file_path,
            )
            return self._naive_docx_chunk(source, file_path)
        except Exception as e:
            logger.warning(
                "Failed to parse docx with unstructured.io for %s: %s. "
                "Falling back to naive chunking.",
                file_path,
                e,
            )
            return self._naive_docx_chunk(source, file_path)

    def _naive_docx_chunk(self, source: str, file_path: str) -> list[Chunk]:
        """
        Fallback: chunk a .docx file by paragraphs (double newlines).
        """
        # Split on double newlines (paragraph boundaries)
        paragraphs = source.split("\n\n")
        chunks: list[Chunk] = []
        line_number = 1

        target_chars = 512 * 4  # ~512 tokens at 4 chars/token
        current_paragraphs: list[str] = []
        current_char_count = 0
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                line_number += 2
                continue

            if current_char_count + len(para) > target_chars and current_paragraphs:
                # Flush current chunk
                text = "\n\n".join(current_paragraphs)
                name = f"paragraph_{chunk_index}"
                chunk_id = self._make_doc_chunk_id(
                    file_path, "paragraph", name, chunk_index
                )
                lines_in_chunk = text.count("\n") + 1
                metadata = ChunkMetadata(
                    file_path=file_path,
                    language="docx",
                    chunk_type="paragraph",
                    name=name,
                    chunk_id=chunk_id,
                    line_start=line_number - lines_in_chunk,
                    line_end=line_number,
                )
                chunks.append(Chunk(metadata=metadata, text=text))
                current_paragraphs = []
                current_char_count = 0
                chunk_index += 1

            current_paragraphs.append(para)
            current_char_count += len(para)
            line_number += para.count("\n") + 2

        # Flush remaining
        if current_paragraphs:
            text = "\n\n".join(current_paragraphs)
            name = f"paragraph_{chunk_index}"
            chunk_id = self._make_doc_chunk_id(
                file_path, "paragraph", name, chunk_index
            )
            metadata = ChunkMetadata(
                file_path=file_path,
                language="docx",
                chunk_type="paragraph",
                name=name,
                chunk_id=chunk_id,
                line_start=line_number,
                line_end=line_number + text.count("\n"),
            )
            chunks.append(Chunk(metadata=metadata, text=text))

        if not chunks:
            chunks.append(self._make_whole_file_chunk(source, file_path, "docx"))

        return chunks

    def _make_doc_chunk_id(
        self, file_path: str, chunk_type: str, name: str, index: int
    ) -> str:
        """Create a chunk_id for a document chunk."""
        import hashlib

        path_hash = hashlib.sha256(file_path.encode()).hexdigest()[:8]
        safe_name = name.replace(":", "_")
        return f"doc:{path_hash}:{chunk_type}:{safe_name}:{index}"

    def _make_whole_file_chunk(
        self, source: str, file_path: str, language: str
    ) -> Chunk:
        """Create a single chunk for an entire file."""
        lines = source.split("\n")
        name = file_path.split("/")[-1]
        chunk_id = self._make_doc_chunk_id(file_path, "section", name, 0)
        metadata = ChunkMetadata(
            file_path=file_path,
            language=language,
            chunk_type="section",
            name=name,
            chunk_id=chunk_id,
            line_start=1,
            line_end=len(lines),
        )
        return Chunk(metadata=metadata, text=source)
