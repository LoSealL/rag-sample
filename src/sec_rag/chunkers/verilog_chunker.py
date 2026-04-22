"""Verilog/SystemVerilog chunker using regex-based extraction."""

import re
from pathlib import Path

from sec_rag.chunkers.base_chunker import BaseChunker, Chunk

LANGUAGE_EXTENSIONS = {
    ".v": "verilog",
    ".sv": "systemverilog",
}

MODULE_PATTERN = re.compile(
    r"\b(?:module|interface|program)\s+(\w+)\s*\(",
    re.MULTILINE,
)
BLOCK_PATTERN = re.compile(
    r"\b(?:always|initial|always_comb|always_ff|always_latch)\b",
    re.MULTILINE,
)


class VerilogChunker(BaseChunker):
    """Extracts Verilog/SystemVerilog constructs using regex patterns."""

    SUPPORTED_EXTENSIONS = {".v", ".sv"}

    def language_for_file(self, file_path: str, source: str) -> str:
        return LANGUAGE_EXTENSIONS.get(Path(file_path).suffix.lower(), "verilog")

    def chunk_source(self, source: str, file_path: str, language: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        line_count = source.count("\n") + 1

        for match in MODULE_PATTERN.finditer(source):
            module_name = match.group(1)
            start_byte = match.start()
            start_line = source[:start_byte].count("\n") + 1

            endmodule_match = re.search(
                rf"\bendmodule\b\s*(:\s*{re.escape(module_name)})?\b",
                source[start_byte:],
            )
            if endmodule_match:
                end_byte = start_byte + endmodule_match.end()
                end_line = source[:end_byte].count("\n") + 1
            else:
                end_byte = len(source)
                end_line = line_count

            chunk_text = source[start_byte:end_byte]
            chunks.append(
                self.make_chunk(
                    chunk_text,
                    file_path,
                    language,
                    "module",
                    module_name,
                    start_line,
                    end_line,
                )
            )

        if not chunks:
            block_num = 0
            for match in BLOCK_PATTERN.finditer(source):
                block_type = match.group(0)
                start_byte = match.start()
                start_line = source[:start_byte].count("\n") + 1

                end_match = re.search(r"\bend\b", source[start_byte:])
                if end_match:
                    end_byte = start_byte + end_match.end()
                    end_line = source[:end_byte].count("\n") + 1
                    chunk_text = source[start_byte:end_byte]
                else:
                    lines = source[start_byte:].split("\n")[:50]
                    chunk_text = "\n".join(lines)
                    end_line = start_line + len(lines) - 1

                block_num += 1
                chunks.append(
                    self.make_chunk(
                        chunk_text,
                        file_path,
                        language,
                        block_type,
                        f"{block_type}_{block_num}",
                        start_line,
                        end_line,
                        block_num - 1,
                    )
                )

        if not chunks:
            chunks.append(
                self.make_chunk(
                    source,
                    file_path,
                    language,
                    "module",
                    Path(file_path).stem,
                    1,
                    line_count,
                )
            )

        return chunks
