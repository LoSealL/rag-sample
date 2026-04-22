"""C/C++/SystemC chunker (unified as cpp language)."""

import re

import tree_sitter
import tree_sitter_c
import tree_sitter_cpp

from sec_rag.chunkers.base_chunker import BaseChunker, Chunk

_SYSTEMC_INCLUDES = re.compile(
    r'#\s*include\s+[<"](?:systemc|tlm|tlm_utils)[./]',
    re.IGNORECASE,
)
_SYSTEMC_KEYWORDS = re.compile(
    r'\b(?:SC_MODULE|SC_THREAD|SC_METHOD|SC_CTHREAD|sc_in|sc_out|sc_inout'
    r'|sc_signal|sc_port|tlm_initiator_socket|tlm_target_socket)\b',
)


class CppChunker(BaseChunker):
    SUPPORTED_EXTENSIONS = {".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh"}

    def __init__(
        self,
        naive_chunk_lines: int = 60,
        naive_overlap_ratio: float = 0.2,
        max_tokens_before_split: int = 600,
    ):
        self._naive_chunk_lines = naive_chunk_lines
        self._naive_overlap_ratio = naive_overlap_ratio
        self._max_tokens_before_split = max_tokens_before_split

        c_lang = tree_sitter.Language(tree_sitter_c.language())
        cpp_lang = tree_sitter.Language(tree_sitter_cpp.language())
        self._c_parser = tree_sitter.Parser(c_lang)
        self._cpp_parser = tree_sitter.Parser(cpp_lang)

    def language_for_file(self, file_path: str, source: str) -> str:
        return "cpp"

    def chunk_source(self, source: str, file_path: str, language: str) -> list[Chunk]:
        parser = self._cpp_parser
        if file_path.endswith((".c", ".h")) and not self._is_systemc(source):
            parser = self._c_parser

        try:
            tree = parser.parse(bytes(source, "utf-8"))
        except Exception:
            return self._naive_chunk(source, file_path, language)

        root = tree.root_node
        if root is None:
            return self._naive_chunk(source, file_path, language)

        chunks: list[Chunk] = []
        for node in root.children:
            if node.type in ("function_definition", "preproc_function_def"):
                name = self._c_function_name(source, node)
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
            elif node.type == "class_specifier":
                name_node = node.child_by_field_name("name")
                name = (
                    source[name_node.start_byte : name_node.end_byte]
                    if name_node
                    else "UnnamedClass"
                )
                chunks.append(
                    self.make_chunk(
                        source[node.start_byte : node.end_byte],
                        file_path,
                        language,
                        "class",
                        name,
                        node.start_point[0] + 1,
                        node.end_point[0] + 1,
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
                    file_path.split("/")[-1],
                    1,
                    line_count,
                )
            )

        chunks.sort(key=lambda c: c.metadata.line_start)
        return chunks

    @staticmethod
    def _is_systemc(source: str) -> bool:
        head = "\n".join(source.splitlines()[:60])
        return bool(_SYSTEMC_INCLUDES.search(head) or _SYSTEMC_KEYWORDS.search(head))

    @staticmethod
    def _c_function_name(source: str, node) -> str:
        cursor = node.walk()
        while cursor.goto_first_child():
            if cursor.node.type == "identifier":
                return source[cursor.node.start_byte : cursor.node.end_byte]
            if not cursor.goto_next_sibling():
                break
        return "unnamed_func"

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
