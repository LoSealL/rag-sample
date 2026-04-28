"""LLM-driven semantic chunker with AST fallback and boundary correction."""

import json
import re
from pathlib import Path

from sec_rag.chunkers.base_chunker import BaseChunker, Chunk


class LLMChunker(BaseChunker):
    SUPPORTED_EXTENSIONS = {
        ".py",
        ".c",
        ".h",
        ".cpp",
        ".cc",
        ".cxx",
        ".hpp",
        ".hh",
        ".v",
        ".sv",
        ".md",
        ".markdown",
        ".docx",
        ".txt",
    }

    # C++ declaration keywords — used for boundary correction
    _CPP_DECL_KEYWORDS = re.compile(
        r'^\s*(?:template\s*<[^>]*>\s*)?(?:'
        r'class|struct|using|concept|namespace|'
        r'(?:[\w:<>,\s&*]+)\s+[A-Za-z_]\w*\s*\('
        r')',
        re.MULTILINE
    )
    
    # Template detection
    _TEMPLATE_RE = re.compile(r'^\s*template\s*<', re.MULTILINE)

    def __init__(
        self,
        llm,
        target_chunk_lines: int = 80,
        overlap_lines: int = 8,
        max_chunks: int = 40,
    ):
        self._llm = llm
        self._target_chunk_lines = target_chunk_lines
        self._overlap_lines = overlap_lines
        self._max_chunks = max_chunks

    def language_for_file(self, file_path: str, source: str) -> str:
        ext = Path(file_path).suffix.lower()
        if ext == ".py":
            return "python"
        if ext in {".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh"}:
            return "cpp"
        if ext == ".v":
            return "verilog"
        if ext == ".sv":
            return "systemverilog"
        if ext in {".md", ".markdown"}:
            return "markdown"
        if ext == ".docx":
            return "docx"
        return "text"

    def chunk_source(self, source: str, file_path: str, language: str) -> list[Chunk]:
        lines = source.split("\n")
        ranges = self._infer_ranges_with_llm(lines, language, source)
        if not ranges:
            return self._naive_chunks(lines, file_path, language)

        chunks: list[Chunk] = []
        for idx, item in enumerate(ranges):
            start_line = int(item.get("start_line", 0))
            end_line = int(item.get("end_line", 0))
            if start_line <= 0 or end_line <= 0 or start_line > end_line:
                continue
            start_line = max(1, start_line)
            end_line = min(len(lines), end_line)
            text = "\n".join(lines[start_line - 1 : end_line]).strip()
            if not text:
                continue
            name = str(item.get("name") or f"chunk_{idx + 1}")
            chunk_type = str(item.get("chunk_type") or "semantic")
            chunks.append(
                self.make_chunk(
                    text,
                    file_path,
                    language,
                    chunk_type,
                    name,
                    start_line,
                    end_line,
                    idx,
                )
            )

        if not chunks:
            return self._naive_chunks(lines, file_path, language)

        chunks.sort(key=lambda c: c.metadata.line_start)
        return chunks

    def _infer_ranges_with_llm(self, lines: list[str], language: str, source: str) -> list[dict]:
        numbered = "\n".join(f"{idx + 1:04d}: {line}" for idx, line in enumerate(lines))
        
        # Build language-specific instructions
        if language == "cpp":
            semantic_types = (
                "class, class_template, struct, struct_template, "
                "function, function_template, concept, type_alias, namespace"
            )
            rules = (
                "CRITICAL RULES for C/C++:\n"
                "1. Each chunk MUST contain exactly ONE complete semantic unit.\n"
                "2. NEVER split a semantic unit across chunks.\n"
                "3. start_line MUST be the line containing the DECLARATION KEYWORD:\n"
                "   - For classes/structs: the 'class' or 'struct' keyword line\n"
                "   - For functions: the return-type + name line\n"
                "   - For templates: the 'template<...>' line\n"
                "   - For type aliases: the 'using' line\n"
                "   - For concepts: the 'template<...> concept' line\n"
                "   - For namespaces: the 'namespace' line\n"
                "   Do NOT include preceding Doxygen comments in start_line!\n"
                "4. end_line MUST be the line where the unit ends (closing brace '}' or semicolon ';').\n"
                "5. For templates: chunk_type must be class_template/function_template (NOT class/function).\n"
                "6. For type aliases: 'using X = Y;' is a complete unit.\n"
                "7. For concepts: 'template<...> concept X = ...;' is a complete unit.\n"
                "8. Do NOT wrap entire namespaces in one chunk; extract inner units instead.\n"
                "9. Do NOT include explicit template instantiations like 'template class X<Y>;'.\n"
            )
        elif language == "python":
            semantic_types = "function, class, module"
            rules = (
                "CRITICAL RULES for Python:\n"
                "1. Each chunk MUST contain exactly ONE complete function or class.\n"
                "2. Include the docstring and decorators belonging to the function/class.\n"
                "3. NEVER split a function or class across chunks.\n"
                "4. start_line must be the 'def' or 'class' line.\n"
            )
        else:
            semantic_types = "section, semantic"
            rules = (
                "CRITICAL RULES:\n"
                "1. Split at logical boundaries (sections, paragraphs).\n"
                "2. Never split in the middle of a sentence or code block.\n"
            )

        prompt = (
            "You are an expert C++ code analyst specialized in semantic chunking for RAG systems.\n"
            "Your task is to identify every semantic unit in the source code and output\n"
            "its exact line range. Precision is critical — a single line error makes the chunk useless.\n\n"
            f"Language: {language}\n"
            f"Supported semantic types: {semantic_types}\n"
            f"Max chunks per file: {self._max_chunks}\n\n"
            f"{rules}\n"
            "OUTPUT FORMAT:\n"
            "Return STRICT JSON only — no markdown, no explanations, no code fences.\n"
            "Schema: {\"chunks\":[{\"start_line\":1,\"end_line\":10,"
            "\"chunk_type\":\"class\",\"name\":\"MyClass\"}]}\n\n"
            "chunk_type must be one of the supported semantic types.\n"
            "name must be the exact symbol name.\n"
            "start_line and end_line must be 1-based and inclusive.\n"
            "Chunks must be sorted by start_line and must not overlap.\n\n"
            "EXAMPLE (C++):\n"
            "001: /**\n"
            "002:  * @brief A point in 2D space.\n"
            "003:  */\n"
            "004: struct Point {\n"
            "005:     float x;\n"
            "006:     float y;\n"
            "007: };\n"
            "008: \n"
            "009: float distance(Point a, Point b) {\n"
            "010:     return std::sqrt(a.x*a.x + a.y*a.y);\n"
            "011: }\n"
            "\n"
            "Expected output:\n"
            '{"chunks":['
            '{"start_line":4,"end_line":7,"chunk_type":"struct","name":"Point"},'
            '{"start_line":9,"end_line":11,"chunk_type":"function","name":"distance"}'
            ']}\n\n'
            "Notice: start_line=4 (the 'struct' keyword), NOT 1 (the comment).\n\n"
            "Input with line numbers:\n"
            f"{numbered}"
        )

        try:
            raw = self._llm.complete(prompt)
        except Exception:
            return []

        payload = self._extract_json(raw)
        if not isinstance(payload, dict):
            return []

        chunks = payload.get("chunks")
        if not isinstance(chunks, list):
            return []
        
        # Post-process: validate, clamp, and correct boundaries
        valid_chunks = self._post_process_ranges(chunks, lines, source)
        
        return valid_chunks[: self._max_chunks]

    def _post_process_ranges(self, chunks: list[dict], lines: list[str], source: str) -> list[dict]:
        """Validate and correct chunk boundaries using source heuristics."""
        total_lines = len(lines)
        valid = []
        
        for item in chunks:
            if not isinstance(item, dict):
                continue
            start = int(item.get("start_line", 0))
            end = int(item.get("end_line", 0))
            if start < 1 or end < 1 or start > end or start > total_lines:
                continue
            
            end = min(end, total_lines)
            chunk_type = str(item.get("chunk_type", ""))
            name = str(item.get("name", ""))
            
            # Correct start_line: move forward from comment/template to declaration keyword
            corrected_start = self._find_declaration_line(lines, start - 1, chunk_type)
            if corrected_start > 0:
                start = corrected_start
            else:
                # Fallback: if LLM gave a comment line, try to find the next declaration
                for i in range(start - 1, min(len(lines), start - 1 + 10)):
                    line = lines[i].strip()
                    if line and not line.startswith('//') and not line.startswith('/*') and not line.startswith('*'):
                        if re.search(r'^\s*(?:template\s*<[^>]*>\s*)?(?:class|struct|using|concept|namespace|(?:[\w:<>,\s&*]+)\s+[A-Za-z_]\w*\s*\()', line):
                            start = i + 1
                            break
            
            # Correct end_line: ensure it ends at closing brace or semicolon
            corrected_end = self._find_unit_end(lines, start - 1, end - 1)
            if corrected_end > 0:
                end = corrected_end
            
            # Correct chunk_type for templates
            if chunk_type in ("class", "function", "struct"):
                # Check if there's a template declaration before this line
                text_above = "\n".join(lines[max(0, start - 5) : start])
                if self._TEMPLATE_RE.search(text_above):
                    chunk_type = chunk_type + "_template"
            
            item["start_line"] = start
            item["end_line"] = end
            item["chunk_type"] = chunk_type
            
            # Skip explicit template instantiations
            chunk_text = "\n".join(lines[start - 1 : end]).strip()
            if re.search(r'^\s*template\s+class\s+\w+<[^;>]*>\s*;\s*$', chunk_text, re.MULTILINE):
                continue
            
            valid.append(item)
        
        # Filter out namespace chunks that contain inner units
        namespace_ranges = [
            (item["start_line"], item["end_line"])
            for item in valid
            if item.get("chunk_type") == "namespace"
        ]
        filtered = []
        for item in valid:
            if item.get("chunk_type") == "namespace":
                # Check if this namespace contains any other chunk
                inner = any(
                    other["start_line"] > item["start_line"] and other["end_line"] < item["end_line"]
                    for other in valid
                    if other is not item
                )
                if inner:
                    continue
            filtered.append(item)
        
        # Sort and deduplicate overlapping chunks
        filtered.sort(key=lambda x: (x["start_line"], -x["end_line"]))
        deduped = []
        for item in filtered:
            if deduped and item["start_line"] <= deduped[-1]["end_line"]:
                # Overlapping — extend previous if this one ends later
                if item["end_line"] > deduped[-1]["end_line"]:
                    deduped[-1]["end_line"] = item["end_line"]
                    if item.get("name"):
                        deduped[-1]["name"] = item["name"]
                    if item.get("chunk_type"):
                        deduped[-1]["chunk_type"] = item["chunk_type"]
                continue
            deduped.append(item)
        
        return deduped

    def _find_declaration_line(self, lines: list[str], start_idx: int, chunk_type: str) -> int:
        """Find the actual declaration keyword line starting from start_idx."""
        # For templates, we want the class/struct/function/concept line, not template<...> line
        keywords = {
            "class": r'^\s*class\s+',
            "class_template": r'^\s*(?:template\s*<[^>]*>\s*)?class\s+',
            "struct": r'^\s*struct\s+',
            "struct_template": r'^\s*(?:template\s*<[^>]*>\s*)?struct\s+',
            "function": r'^\s*(?:[\w:<>,\s&*]+)\s+[A-Za-z_]\w*\s*\(',
            "function_template": r'^\s*(?:template\s*<[^>]*>\s*)?(?:[\w:<>,\s&*]+)\s+[A-Za-z_]\w*\s*\(',
            "concept": r'^\s*(?:template\s*<[^>]*>\s*)?concept\s+',
            "type_alias": r'^\s*using\s+',
            "namespace": r'^\s*namespace\s+',
        }
        
        pattern = keywords.get(chunk_type)
        if not pattern:
            return -1
        
        # Search within a window of 5 lines forward and 2 lines backward
        for i in range(max(0, start_idx - 2), min(len(lines), start_idx + 6)):
            if re.search(pattern, lines[i]):
                return i + 1  # 1-based
        return -1

    def _find_unit_end(self, lines: list[str], start_idx: int, end_idx: int) -> int:
        """Find the actual end of a semantic unit (closing brace or semicolon)."""
        # If end_idx already looks correct (ends with }; or }), trust it
        last_line = lines[end_idx].strip()
        if last_line.endswith('};') or last_line.endswith('}'):
            return end_idx + 1
        
        # Search forward for closing brace or semicolon
        brace_depth = 0
        for i in range(start_idx, min(len(lines), end_idx + 5)):
            line = lines[i]
            # Simple brace counting (ignores strings/comments)
            brace_depth += line.count('{') - line.count('}')
            if brace_depth == 0 and (';' in line or '}' in line):
                return i + 1
        
        return -1

    @staticmethod
    def _extract_json(raw: str):
        raw = raw.strip()
        
        # Try direct parse first
        try:
            return json.loads(raw)
        except Exception:
            pass
        
        # Remove markdown code fences
        cleaned = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()
        
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # Extract JSON object from text
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    def _naive_chunks(self, lines: list[str], file_path: str, language: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        total = len(lines)
        target = self._target_chunk_lines
        step = max(1, target - self._overlap_lines)
        index = 0
        for start in range(0, total, step):
            part = lines[start : start + target]
            if not part:
                continue
            text = "\n".join(part).strip()
            if not text:
                continue
            line_start = start + 1
            line_end = min(total, start + target)
            chunks.append(
                self.make_chunk(
                    text,
                    file_path,
                    language,
                    "semantic",
                    f"chunk_{index + 1}",
                    line_start,
                    line_end,
                    index,
                )
            )
            index += 1
        return chunks
