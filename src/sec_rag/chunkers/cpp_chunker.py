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
        self._collect_chunks(root, source, file_path, language, chunks)

        # Deduplicate: if a chunk is fully contained within another chunk
        # with the same name, keep the inner one (more precise)
        filtered = []
        for i, chunk in enumerate(chunks):
            is_contained = False
            for j, other in enumerate(chunks):
                if i == j:
                    continue
                if (chunk.metadata.line_start >= other.metadata.line_start and
                    chunk.metadata.line_end <= other.metadata.line_end and
                    chunk.metadata.name == other.metadata.name):
                    # Same name, fully contained — only keep if this is the inner one
                    if chunk.metadata.line_start > other.metadata.line_start:
                        is_contained = True
                        break
            if not is_contained:
                filtered.append(chunk)

        if not filtered:
            line_count = source.count("\n") + 1
            filtered.append(
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

        filtered.sort(key=lambda c: c.metadata.line_start)
        return filtered

    def _collect_chunks(self, node, source: str, file_path: str, language: str, chunks: list[Chunk]):
        """Recursively collect semantic units from AST."""
        chunk = self._node_to_chunk(node, source, file_path, language)
        if chunk:
            if chunk.metadata.chunk_type == "namespace":
                # Namespace with inner units: skip namespace, recurse into children
                if self._has_inner_chunks(node, source, file_path, language):
                    for child in node.children:
                        self._collect_chunks(child, source, file_path, language, chunks)
                    return
                # Empty namespace: keep it
                chunks.append(chunk)
                return
            else:
                # Non-namespace semantic unit: add it, don't recurse
                chunks.append(chunk)
                return
        
        # Not a semantic unit: recurse into children
        for child in node.children:
            self._collect_chunks(child, source, file_path, language, chunks)

    def _has_inner_chunks(self, node, source: str, file_path: str, language: str) -> bool:
        """Check if a node contains any inner semantic units."""
        for child in node.children:
            if self._node_to_chunk(child, source, file_path, language) is not None:
                return True
            if self._has_inner_chunks(child, source, file_path, language):
                return True
        return False

    def _node_to_chunk(self, node, source: str, file_path: str, language: str) -> Chunk | None:
        """Convert a single AST node to a Chunk if it's a semantic unit."""
        chunk_text = self._text_from_node(source, node)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        
        if node.type == "function_definition":
            name = self._extract_name(node, source, "identifier")
            if not name:
                name = "unnamed_func"
            if len(chunk_text) // 4 > self._max_tokens_before_split:
                return None  # Will be handled by _collect_chunks recursing into it
            return self.make_chunk(
                chunk_text, file_path, language, "function", name,
                start_line, end_line,
            )
        
        elif node.type == "class_specifier":
            # Skip forward declarations and explicit instantiations (no body)
            has_body = any(c.type == "field_declaration_list" for c in node.children)
            if not has_body:
                return None
            name = self._extract_name(node, source, "type_identifier")
            if not name:
                name = "UnnamedClass"
            return self.make_chunk(
                chunk_text, file_path, language, "class", name,
                start_line, end_line,
            )
        
        elif node.type == "struct_specifier":
            name = self._extract_name(node, source, "type_identifier")
            if not name:
                name = "UnnamedStruct"
            return self.make_chunk(
                chunk_text, file_path, language, "struct", name,
                start_line, end_line,
            )
        
        elif node.type == "alias_declaration":
            name = self._extract_name(node, source, "type_identifier")
            if not name:
                name = "UnnamedAlias"
            return self.make_chunk(
                chunk_text, file_path, language, "type_alias", name,
                start_line, end_line,
            )
        
        elif node.type == "template_declaration":
            # Template wraps another declaration — extract the inner one
            inner = self._find_template_inner(node, source, file_path, language)
            if inner:
                return inner
            return None
        
        elif node.type == "namespace_definition":
            name = self._extract_name(node, source, "namespace_identifier")
            if not name:
                name = "unnamed_namespace"
            return self.make_chunk(
                chunk_text, file_path, language, "namespace", name,
                start_line, end_line,
            )
        
        elif node.type == "concept_definition":
            name = self._extract_name(node, source, "identifier")
            if not name:
                name = "UnnamedConcept"
            return self.make_chunk(
                chunk_text, file_path, language, "concept", name,
                start_line, end_line,
            )
        
        return None

    @staticmethod
    def _text_from_node(source: str, node) -> str:
        """Extract text from node using byte offsets (handles non-ASCII)."""
        source_bytes = source.encode("utf-8")
        return source_bytes[node.start_byte : node.end_byte].decode("utf-8")

    def _find_template_inner(self, node, source: str, file_path: str, language: str) -> Chunk | None:
        """Find the inner declaration of a template and mark it as template."""
        for child in node.children:
            chunk = self._node_to_chunk(child, source, file_path, language)
            if chunk:
                # Mark as template variant
                if chunk.metadata.chunk_type == "class":
                    chunk.metadata.chunk_type = "class_template"
                elif chunk.metadata.chunk_type == "function":
                    chunk.metadata.chunk_type = "function_template"
                elif chunk.metadata.chunk_type == "struct":
                    chunk.metadata.chunk_type = "struct_template"
                return chunk
        return None

    def _extract_name(self, node, source: str, name_field: str) -> str:
        """Extract identifier name from node."""
        # Try child_by_field_name first (tree-sitter 0.20+)
        try:
            name_node = node.child_by_field_name("name")
            if name_node:
                return self._text_from_node(source, name_node)
        except Exception:
            pass
        
        # Fallback: search children recursively for the name type
        for child in node.children:
            if child.type == name_field:
                return self._text_from_node(source, child)
            if child.type == "identifier":
                return self._text_from_node(source, child)
            # For function definitions, name is inside function_declarator
            if child.type == "function_declarator":
                for sub in child.children:
                    if sub.type in ("identifier", "operator_name", "destructor_name"):
                        return self._text_from_node(source, sub)
            # For template declarations, recurse into inner declaration
            if child.type in ("class_specifier", "struct_specifier", "function_definition"):
                return self._extract_name(child, source, name_field)
        return ""

    @staticmethod
    def _is_systemc(source: str) -> bool:
        head = "\n".join(source.splitlines()[:60])
        return bool(_SYSTEMC_INCLUDES.search(head) or _SYSTEMC_KEYWORDS.search(head))

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
