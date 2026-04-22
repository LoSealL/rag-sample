"""Python AST chunker based on tree-sitter."""

from pathlib import Path

import tree_sitter
import tree_sitter_python

from sec_rag.chunkers.base_chunker import BaseChunker, Chunk


class PythonChunker(BaseChunker):
    SUPPORTED_EXTENSIONS = {".py"}

    def __init__(
        self,
        naive_chunk_lines: int = 60,
        naive_overlap_ratio: float = 0.2,
        max_tokens_before_split: int = 600,
    ):
        self._naive_chunk_lines = naive_chunk_lines
        self._naive_overlap_ratio = naive_overlap_ratio
        self._max_tokens_before_split = max_tokens_before_split
        language = tree_sitter.Language(tree_sitter_python.language())
        self._parser = tree_sitter.Parser(language)

    def language_for_file(self, file_path: str, source: str) -> str:
        return "python"

    def chunk_source(self, source: str, file_path: str, language: str) -> list[Chunk]:
        try:
            tree = self._parser.parse(bytes(source, "utf-8"))
        except Exception:
            return self._naive_chunk(source, file_path, language)

        root = tree.root_node
        if root is None:
            return self._naive_chunk(source, file_path, language)

        chunks: list[Chunk] = []
        for node in root.children:
            if node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                name = (
                    source[name_node.start_byte : name_node.end_byte]
                    if name_node
                    else "unnamed_func"
                )
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                chunk_text = source[node.start_byte : node.end_byte]

                if len(chunk_text) // 4 > self._max_tokens_before_split:
                    chunks.extend(
                        self._split_large_chunk(
                            chunk_text,
                            file_path,
                            language,
                            "function",
                            name,
                            start_line,
                            end_line,
                        )
                    )
                else:
                    chunks.append(
                        self.make_chunk(
                            chunk_text,
                            file_path,
                            language,
                            "function",
                            name,
                            start_line,
                            end_line,
                        )
                    )
            elif node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                name = (
                    source[name_node.start_byte : name_node.end_byte]
                    if name_node
                    else "UnnamedClass"
                )
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                chunk_text = source[node.start_byte : node.end_byte]
                chunks.append(
                    self.make_chunk(
                        chunk_text,
                        file_path,
                        language,
                        "class",
                        name,
                        start_line,
                        end_line,
                    )
                )

        if not chunks:
            line_count = source.count("\n") + 1
            chunks.append(
                self.make_chunk(
                    source,
                    file_path,
                    language,
                    "module_docstring",
                    Path(file_path).name,
                    1,
                    line_count,
                )
            )

        chunks.sort(key=lambda c: c.metadata.line_start)
        return chunks

    def _split_large_chunk(
        self,
        text: str,
        file_path: str,
        language: str,
        chunk_type: str,
        name: str,
        start_line: int,
        end_line: int,
    ) -> list[Chunk]:
        lines = text.split("\n")
        total_lines = len(lines)
        target_lines = self._naive_chunk_lines
        overlap_lines = int(target_lines * self._naive_overlap_ratio)

        chunks: list[Chunk] = []
        step = max(1, target_lines - overlap_lines)
        for i in range(0, total_lines, step):
            chunk_lines = lines[i : i + target_lines]
            if not chunk_lines:
                continue
            chunks.append(
                self.make_chunk(
                    "\n".join(chunk_lines),
                    file_path,
                    language,
                    chunk_type,
                    f"{name}_part_{i // step + 1}",
                    start_line + i,
                    min(start_line + i + target_lines - 1, end_line),
                )
            )
        return chunks

    def _naive_chunk(self, source: str, file_path: str, language: str) -> list[Chunk]:
        lines = source.split("\n")
        total_lines = len(lines)
        target_lines = self._naive_chunk_lines
        overlap_lines = int(target_lines * self._naive_overlap_ratio)
        chunks: list[Chunk] = []

        step = max(1, target_lines - overlap_lines)
        for i in range(0, total_lines, step):
            chunk_lines = lines[i : i + target_lines]
            if not chunk_lines:
                continue
            chunks.append(
                self.make_chunk(
                    "\n".join(chunk_lines),
                    file_path,
                    language,
                    "function",
                    f"lines_{i + 1}_{min(i + target_lines, total_lines)}",
                    i + 1,
                    min(i + target_lines, total_lines),
                )
            )

        return chunks
