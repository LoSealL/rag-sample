"""
Code chunker using tree-sitter.

Chunks Python, C, and C++ source files by function/class, not by token count.
Falls back to naive line-range chunking if tree-sitter fails to parse.

Usage:
    chunker = CodeChunker()
    chunks = chunker.chunk_file("src/auth/login.py")
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import tree_sitter
import tree_sitter_c
import tree_sitter_cpp
import tree_sitter_python
from loguru import logger

LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
}

# SystemC/TLM detection: if any of these patterns appear in the first 60 lines
# of a C/C++ file, the file is classified as "systemc" instead of "cpp"/"c".
_SYSTEMC_INCLUDES = re.compile(
    r'#\s*include\s+[<"](?:systemc|tlm|tlm_utils)[./]',
    re.IGNORECASE,
)
_SYSTEMC_KEYWORDS = re.compile(
    r'\b(?:SC_MODULE|SC_THREAD|SC_METHOD|SC_CTHREAD|sc_in|sc_out|sc_inout'
    r'|sc_signal|sc_port|tlm_initiator_socket|tlm_target_socket)\b',
)


@dataclass
class ChunkMetadata:
    file_path: str
    language: str
    chunk_type: str  # "function" | "class" | "module_docstring" | "header"
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
    """A chunk of code with metadata."""

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


class CodeChunker:
    """
    Chunks source code files by function/class using tree-sitter AST.

    Usage:
        chunker = CodeChunker()
        chunks = chunker.chunk_file("path/to/file.py")
    """

    # Languages supported by this chunker
    SUPPORTED_LANGUAGES = {"python", "c", "cpp"}

    # Approximate tokens per line (used for size estimation)
    TOKENS_PER_LINE = 4

    def __init__(self):
        self._parsers: dict[str, tree_sitter.Parser] = {}
        self._load_parsers()

    def _load_parsers(self) -> None:
        """Load tree-sitter parsers for each supported language."""
        for lang, module in [
            ("python", tree_sitter_python),
            ("c", tree_sitter_c),
            ("cpp", tree_sitter_cpp),
        ]:
            language = tree_sitter.Language(module.language())
            parser = tree_sitter.Parser(language)
            self._parsers[lang] = parser
            logger.debug("Loaded tree-sitter parser for {}", lang)

    def chunk_file(self, file_path: str) -> list[Chunk]:
        """
        Chunk a source code file by function/class.

        Args:
            file_path: Path to the source file.

        Returns:
            List of Chunk objects. Empty list if file cannot be chunked.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        language = self._language_from_path(file_path)
        if language is None:
            logger.debug("Unsupported language for {}", file_path)
            return []

        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                source = f.read()
        except Exception as e:
            logger.warning("Failed to read {}: {}", file_path, e)
            return []

        if not source.strip():
            return []

        # Promote C/C++ files that contain SystemC/TLM patterns to "systemc"
        if language in ("c", "cpp") and self._is_systemc(source):
            language = "systemc"

        return self._chunk_source(source, file_path, language)

    @staticmethod
    def _is_systemc(source: str) -> bool:
        """Return True if the source appears to be a SystemC/TLM file."""
        head = "\n".join(source.splitlines()[:60])
        return bool(_SYSTEMC_INCLUDES.search(head) or _SYSTEMC_KEYWORDS.search(head))

    def _chunk_source(self, source: str, file_path: str, language: str) -> list[Chunk]:
        """Parse and chunk source code."""
        # "systemc" files use the C++ tree-sitter parser
        parser_key = "cpp" if language == "systemc" else language
        parser = self._parsers.get(parser_key)
        if parser is None:
            return []

        try:
            tree = parser.parse(bytes(source, "utf-8"))
        except Exception as e:
            # tree_sitter stubs may not expose ParseError; fall back on parse failures.
            logger.warning(
                "tree-sitter parse error for %s: %s. Falling back to naive chunking.",
                file_path,
                e,
            )
            return self._naive_chunk(source, file_path, language)

        if tree.root_node is None:
            logger.warning(
                "tree-sitter returned empty tree for %s. "
                "Falling back to naive chunking.",
                file_path,
            )
            return self._naive_chunk(source, file_path, language)

        # Count lines in source for overlap calculation
        line_count = source.count("\n")
        chunks: list[Chunk] = []

        # Extract top-level definitions
        root = tree.root_node

        # Python: function, class, module_docstring
        if language == "python":
            chunks.extend(
                self._extract_python_defs(source, file_path, root, line_count)
            )

        # C/C++/SystemC: function declarations and definitions
        elif language in ("c", "cpp", "systemc"):
            chunks.extend(self._extract_c_defs(source, file_path, root, line_count))

        # If no chunks found, try module-level docstring or entire file
        if not chunks:
            if language == "python":
                docstring = self._extract_python_docstring(source, file_path, root)
                if docstring:
                    chunks.append(docstring)
            if not chunks:
                # Last resort: whole file as one chunk
                chunks.append(
                    self._make_chunk(
                        source,
                        file_path,
                        language,
                        "module_docstring",
                        file_path.split("/")[-1],
                        1,
                        line_count,
                    )
                )

        # Sort by line number for deterministic ordering
        chunks.sort(key=lambda c: c.metadata.line_start)
        return chunks

    def _extract_python_defs(
        self, source: str, file_path: str, root, line_count: int
    ) -> list[Chunk]:
        """Extract function and class definitions from Python source."""
        chunks: list[Chunk] = []
        seen_names: set[str] = set()

        for node in root.children:
            if node.type == "function_definition":
                name = self._node_text(source, node)
                # Get the function name from the name node
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = source[name_node.start_byte : name_node.end_byte]

                chunk_type = "function"
                start_line = node.start_point[0] + 1  # 1-indexed
                end_line = node.end_point[0] + 1

                chunk_text = self._node_text(source, node)
                estimated_tokens = len(chunk_text) // 4

                if estimated_tokens > 600:
                    sub_chunks = self._split_large_chunk(
                        chunk_text,
                        file_path,
                        "python",
                        chunk_type,
                        name,
                        start_line,
                        end_line,
                    )
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(
                        self._make_chunk(
                            chunk_text,
                            file_path,
                            "python",
                            chunk_type,
                            name,
                            start_line,
                            end_line,
                        )
                    )
                seen_names.add(name)

            elif node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                name = (
                    source[name_node.start_byte : name_node.end_byte]
                    if name_node
                    else "UnnamedClass"
                )

                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                chunk_text = self._node_text(source, node)
                chunks.append(
                    self._make_chunk(
                        chunk_text,
                        file_path,
                        "python",
                        "class",
                        name,
                        start_line,
                        end_line,
                    )
                )
                seen_names.add(name)

        return chunks

    def _extract_python_docstring(
        self, source: str, file_path: str, root
    ) -> Chunk | None:
        """Extract module-level docstring from Python source."""
        # A module docstring is typically the first expression statement
        # in the file that is a string
        first_node = root.children[0] if root.children else None
        if first_node and first_node.type == "expression_statement":
            expr = first_node
            if expr.children and expr.children[0].type == "string":
                string_node = expr.children[0]
                text = self._node_text(source, string_node)
                start_line = string_node.start_point[0] + 1
                end_line = string_node.end_point[0] + 1
                return self._make_chunk(
                    text,
                    file_path,
                    "python",
                    "module_docstring",
                    "__module__",
                    start_line,
                    end_line,
                )
        return None

    def _extract_c_defs(
        self, source: str, file_path: str, root, line_count: int
    ) -> list[Chunk]:
        """Extract function definitions from C/C++ source."""
        chunks: list[Chunk] = []
        seen_names: set[str] = set()

        for node in root.children:
            # C: function_definition, C++: function_definition
            if node.type in ("function_definition", "preproc_function_def"):
                name = self._c_function_name(source, node)
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                chunk_text = self._node_text(source, node)
                estimated_tokens = len(chunk_text) // 4

                if estimated_tokens > 600:
                    sub_chunks = self._split_large_chunk(
                        chunk_text,
                        file_path,
                        "c"
                        if file_path.endswith(".c") or file_path.endswith(".h")
                        else "cpp",
                        "function",
                        name,
                        start_line,
                        end_line,
                    )
                    chunks.extend(sub_chunks)
                else:
                    lang = (
                        "c"
                        if file_path.endswith(".c") or file_path.endswith(".h")
                        else "cpp"
                    )
                    chunks.append(
                        self._make_chunk(
                            chunk_text,
                            file_path,
                            lang,
                            "function",
                            name,
                            start_line,
                            end_line,
                        )
                    )
                seen_names.add(name)

            # C++ class definitions
            elif node.type == "class_specifier":
                name_node = node.child_by_field_name("name")
                name = (
                    source[name_node.start_byte : name_node.end_byte]
                    if name_node
                    else "UnnamedClass"
                )
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                chunk_text = self._node_text(source, node)
                chunks.append(
                    self._make_chunk(
                        chunk_text,
                        file_path,
                        "cpp",
                        "class",
                        name,
                        start_line,
                        end_line,
                    )
                )
                seen_names.add(name)

        return chunks

    def _c_function_name(self, source: str, node) -> str:
        """Extract function name from a C/C++ function definition node."""
        # Try to find the declarator with the function name
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
        """
        Split a large chunk (>600 tokens) at logical sub-boundaries.

        For Python: tries inner class definitions first, then blank-line gaps.
        Labels sub-chunks with parent context: outer_func_part_1, outer_func_part_2.
        """
        lines = text.split("\n")
        total_lines = len(lines)
        # Target: ~300 tokens per chunk = ~60 lines at 5 tokens/line
        target_lines = 60
        overlap_lines = int(target_lines * 0.2)  # 20% overlap = 12 lines

        sub_chunks: list[Chunk] = []
        # Find blank-line gaps as split points
        split_points = []
        for i, line in enumerate(lines):
            if i > 0 and not line.strip():
                split_points.append(i)

        # If we found logical split points, use them
        if len(split_points) >= 2:
            split_points = [
                p
                for p in split_points
                if target_lines <= p <= total_lines - target_lines
            ]
            if not split_points:
                split_points = [total_lines // 2]

            # Build sub-chunks along split points
            prev = 0
            for idx, sp in enumerate(split_points):
                sub_text = "\n".join(lines[prev:sp])
                sub_name = f"{name}_part_{idx + 1}"
                sub_start = start_line + prev
                sub_end = start_line + sp - 1
                sub_chunks.append(
                    self._make_chunk(
                        sub_text,
                        file_path,
                        language,
                        chunk_type,
                        sub_name,
                        sub_start,
                        sub_end,
                    )
                )
                prev = sp - overlap_lines  # overlap
                if prev < 0:
                    prev = 0

            # Last segment
            if prev < total_lines:
                sub_text = "\n".join(lines[prev:])
                sub_name = f"{name}_part_{len(split_points) + 1}"
                sub_start = start_line + prev
                sub_end = end_line
                sub_chunks.append(
                    self._make_chunk(
                        sub_text,
                        file_path,
                        language,
                        chunk_type,
                        sub_name,
                        sub_start,
                        sub_end,
                    )
                )
        else:
            # No logical split points found — split evenly by line count
            for i in range(0, total_lines, target_lines - overlap_lines):
                chunk_lines = lines[i : i + target_lines]
                if not chunk_lines:
                    continue
                sub_text = "\n".join(chunk_lines)
                sub_name = f"{name}_part_{i // (target_lines - overlap_lines) + 1}"
                sub_start = start_line + i
                sub_end = min(start_line + i + target_lines - 1, end_line)
                sub_chunks.append(
                    self._make_chunk(
                        sub_text,
                        file_path,
                        language,
                        chunk_type,
                        sub_name,
                        sub_start,
                        sub_end,
                    )
                )

        return sub_chunks

    def _naive_chunk(self, source: str, file_path: str, language: str) -> list[Chunk]:
        """
        Fallback: chunk by line ranges when tree-sitter fails.

        Chunks of ~60 lines with 20% overlap.
        """
        lines = source.split("\n")
        total_lines = len(lines)
        target_lines = 60
        overlap_lines = int(target_lines * 0.2)
        chunks: list[Chunk] = []

        for i in range(0, total_lines, target_lines - overlap_lines):
            chunk_lines = lines[i : i + target_lines]
            if not chunk_lines:
                continue
            chunk_text = "\n".join(chunk_lines)
            name = f"lines_{i + 1}_{min(i + target_lines, total_lines)}"
            start_line = i + 1
            end_line = min(i + target_lines, total_lines)
            chunks.append(
                self._make_chunk(
                    chunk_text,
                    file_path,
                    language,
                    "function",
                    name,
                    start_line,
                    end_line,
                )
            )

        return chunks

    def _make_chunk(
        self,
        text: str,
        file_path: str,
        language: str,
        chunk_type: str,
        name: str,
        line_start: int,
        line_end: int,
    ) -> Chunk:
        """Create a Chunk with a properly formatted chunk_id."""
        # chunk_id format: "py:src/auth/login.py:func:authenticate:0"
        # Language prefix: py, c, cpp
        lang_prefix = {"python": "py", "c": "c", "cpp": "cpp"}.get(
            language, language[:2]
        )
        # Sanitize name: replace colons with underscores (colons are the ID separator)
        safe_name = name.replace(":", "_").replace("/", "_")
        chunk_id = f"{lang_prefix}:{file_path}:{chunk_type}:{safe_name}:0"

        # If chunk_id would contain colons from the path... handle that
        # The chunk_id uses colons as separators, so colons in paths break parsing.
        # Use a hash of the path instead of embedding it directly.
        import hashlib

        path_hash = hashlib.sha256(file_path.encode()).hexdigest()[:8]
        chunk_id = f"{lang_prefix}:{path_hash}:{chunk_type}:{safe_name}:0"

        metadata = ChunkMetadata(
            file_path=file_path,
            language=language,
            chunk_type=chunk_type,
            name=name,
            chunk_id=chunk_id,
            line_start=line_start,
            line_end=line_end,
        )
        return Chunk(metadata=metadata, text=text)

    @staticmethod
    def _node_text(source: str, node) -> str:
        """Get the text content of a tree-sitter node."""
        return source[node.start_byte : node.end_byte]

    @staticmethod
    def _language_from_path(file_path: str) -> str | None:
        """Determine language from file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        return LANGUAGE_EXTENSIONS.get(ext)
