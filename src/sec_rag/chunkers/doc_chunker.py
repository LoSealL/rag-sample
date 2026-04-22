"""Document chunker for Markdown and docx."""

import re
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

from sec_rag.chunkers.base_chunker import BaseChunker, Chunk

DOC_EXTENSIONS = {".md", ".markdown", ".docx"}


@dataclass
class DocChunker(BaseChunker):
    SUPPORTED_EXTENSIONS = DOC_EXTENSIONS
    TOKENS_PER_CHAR = 0.25
    doc_target_tokens: int = 512

    def language_for_file(self, file_path: str, source: str) -> str:
        ext = Path(file_path).suffix.lower()
        if ext in {".md", ".markdown"}:
            return "markdown"
        if ext == ".docx":
            return "docx"
        return "doc"

    def chunk_source(self, source: str, file_path: str, language: str) -> list[Chunk]:
        ext = Path(file_path).suffix.lower()
        if ext in {".md", ".markdown"}:
            return self._chunk_markdown(source, file_path)
        if ext == ".docx":
            return self._chunk_docx(source, file_path)
        return []

    def _chunk_markdown(self, source: str, file_path: str) -> list[Chunk]:
        lines = source.split("\n")
        chunks: list[Chunk] = []
        current_lines: list[str] = []
        current_heading: str | None = None
        section_start_line = 1
        chunk_index = 0

        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

        def flush_section() -> None:
            nonlocal current_lines, current_heading, section_start_line, chunk_index
            if not current_lines:
                return
            text = "\n".join(current_lines).strip()
            if not text:
                return

            name = current_heading if current_heading else f"section_{chunk_index}"
            chunk_type = "section" if current_heading else "paragraph"
            chunks.append(
                self.make_chunk(
                    text,
                    file_path,
                    "markdown",
                    chunk_type,
                    name,
                    section_start_line,
                    section_start_line + len(current_lines) - 1,
                    chunk_index,
                )
            )
            current_lines = []
            chunk_index += 1

        for i, line in enumerate(lines):
            heading_match = heading_pattern.match(line)
            if heading_match:
                flush_section()
                current_heading = heading_match.group(2).strip()
                section_start_line = i + 1
                current_lines = [line]
            else:
                current_lines.append(line)

        flush_section()

        if not chunks:
            chunks.append(
                self.make_chunk(
                    source,
                    file_path,
                    "markdown",
                    "section",
                    Path(file_path).name,
                    1,
                    len(lines),
                )
            )

        return chunks

    def _chunk_docx(self, source: str, file_path: str) -> list[Chunk]:
        if find_spec("unstructured.partition.docx") is None:
            return self._naive_docx_chunk(source, file_path)

        try:
            return self._naive_docx_chunk(source, file_path)
        except Exception:
            return self._naive_docx_chunk(source, file_path)

    def _naive_docx_chunk(self, source: str, file_path: str) -> list[Chunk]:
        paragraphs = source.split("\n\n")
        chunks: list[Chunk] = []
        line_number = 1

        target_chars = self.doc_target_tokens * 4
        current_paragraphs: list[str] = []
        current_char_count = 0
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                line_number += 2
                continue

            if current_char_count + len(para) > target_chars and current_paragraphs:
                text = "\n\n".join(current_paragraphs)
                lines_in_chunk = text.count("\n") + 1
                chunks.append(
                    self.make_chunk(
                        text,
                        file_path,
                        "docx",
                        "paragraph",
                        f"paragraph_{chunk_index}",
                        line_number - lines_in_chunk,
                        line_number,
                        chunk_index,
                    )
                )
                current_paragraphs = []
                current_char_count = 0
                chunk_index += 1

            current_paragraphs.append(para)
            current_char_count += len(para)
            line_number += para.count("\n") + 2

        if current_paragraphs:
            text = "\n\n".join(current_paragraphs)
            chunks.append(
                self.make_chunk(
                    text,
                    file_path,
                    "docx",
                    "paragraph",
                    f"paragraph_{chunk_index}",
                    line_number,
                    line_number + text.count("\n"),
                    chunk_index,
                )
            )

        if not chunks:
            lines = source.split("\n")
            chunks.append(
                self.make_chunk(
                    source,
                    file_path,
                    "docx",
                    "section",
                    Path(file_path).name,
                    1,
                    len(lines),
                )
            )

        return chunks
