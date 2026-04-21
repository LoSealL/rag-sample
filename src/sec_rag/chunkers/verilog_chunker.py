"""
Verilog/SystemVerilog chunker using regex-based extraction.

Extracts modules, ports, parameters, and behavioral blocks from Verilog/SystemVerilog.
Falls back to naive line-range chunking if regex extraction fails.

Usage:
    chunker = VerilogChunker()
    chunks = chunker.chunk_file("src/rtl/alu.v")
"""

import re
from pathlib import Path

from loguru import logger

from sec_rag.chunkers.code_chunker import Chunk, ChunkMetadata

LANGUAGE_EXTENSIONS = {
    ".v": "verilog",
    ".sv": "systemverilog",
}

# Regex patterns for Verilog/SystemVerilog extraction
# Module/interface/program definition
MODULE_PATTERN = re.compile(
    r"\b(?:module|interface|program)\s+(\w+)\s*\(",
    re.MULTILINE,
)
# Module port list: input, output, inout, modport
PORT_PATTERN = re.compile(
    r"\b(?:input|output|inout|ref)\b\s*(?:\[[\d:]+\])?\s*(\w+)",
    re.MULTILINE,
)
# Parameter declaration
PARAM_PATTERN = re.compile(
    r"\bparameter\b(?:\s+\w+)?\s+(\w+)\s*=",
    re.MULTILINE,
)
# Always/initial/always_comb/always_ff block
BLOCK_PATTERN = re.compile(
    r"\b(?:always|initial|always_comb|always_ff|always_latch)\b",
    re.MULTILINE,
)


class VerilogChunker:
    """Extracts Verilog/SystemVerilog constructs using regex patterns."""

    SUPPORTED_LANGUAGES = {"verilog", "systemverilog"}
    TOKENS_PER_LINE = 3  # Verilog tends to be more concise than C

    def chunk_file(self, file_path: str) -> list[Chunk]:
        """
        Chunk a Verilog/SystemVerilog file by module/port/parameter/block.

        Args:
            file_path: Path to the Verilog file.

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
            logger.debug("Unsupported Verilog language for {}", file_path)
            return []

        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                source = f.read()
        except Exception as e:
            logger.warning("Failed to read {}: {}", file_path, e)
            return []

        if not source.strip():
            return []

        return self._chunk_source(source, file_path, language)

    @staticmethod
    def _language_from_path(file_path: str) -> str | None:
        """Determine language from file extension."""
        ext = Path(file_path).suffix.lower()
        return LANGUAGE_EXTENSIONS.get(ext)

    def _chunk_source(
        self, source: str, file_path: str, language: str
    ) -> list[Chunk]:
        """Parse and chunk Verilog/SystemVerilog source."""
        chunks: list[Chunk] = []
        line_count = source.count("\n")

        # Extract modules
        for match in MODULE_PATTERN.finditer(source):
            module_name = match.group(1)
            start_byte = match.start()
            start_line = source[:start_byte].count("\n") + 1

            # Find the matching 'endmodule'
            endmodule_match = re.search(
                rf"\bendmodule\b\s*(:\s*{re.escape(module_name)})?\b",
                source[start_byte:],
            )
            if endmodule_match:
                end_byte = start_byte + endmodule_match.end()
                end_line = source[:end_byte].count("\n") + 1
                chunk_text = source[start_byte:end_byte]
            else:
                end_byte = len(source)
                end_line = line_count
                chunk_text = source[start_byte:end_byte]

            chunk_id = f"verilog:{file_path}:module:{module_name}:0"
            chunks.append(
                Chunk(
                    metadata=ChunkMetadata(
                        file_path=file_path,
                        language=language,
                        chunk_type="module",
                        name=module_name,
                        chunk_id=chunk_id,
                        line_start=start_line,
                        line_end=end_line,
                    ),
                    text=chunk_text,
                )
            )

        # If no modules found, try to find always/initial blocks
        if not chunks:
            block_num = 0
            for match in BLOCK_PATTERN.finditer(source):
                block_type = match.group(0)
                start_byte = match.start()
                start_line = source[:start_byte].count("\n") + 1

                # Find matching 'end' (rough heuristic: next 'end' keyword)
                end_match = re.search(r"\bend\b", source[start_byte:])
                if end_match:
                    end_byte = start_byte + end_match.end()
                    end_line = source[:end_byte].count("\n") + 1
                    chunk_text = source[start_byte:end_byte]
                else:
                    # Take next 50 lines as fallback
                    lines = source[start_byte:].split("\n")[:50]
                    chunk_text = "\n".join(lines)
                    end_line = start_line + len(lines) - 1

                block_num += 1
                chunk_id = f"verilog:{file_path}:{block_type}:{block_num}:0"
                chunks.append(
                    Chunk(
                        metadata=ChunkMetadata(
                            file_path=file_path,
                            language=language,
                            chunk_type=block_type,
                            name=f"{block_type}_{block_num}",
                            chunk_id=chunk_id,
                            line_start=start_line,
                            line_end=end_line,
                        ),
                        text=chunk_text,
                    )
                )

        # If still no chunks, use the whole file
        if not chunks:
            chunks.append(
                Chunk(
                    metadata=ChunkMetadata(
                        file_path=file_path,
                        language=language,
                        chunk_type="module",
                        name=Path(file_path).stem,
                        chunk_id=f"verilog:{file_path}:whole:0",
                        line_start=1,
                        line_end=line_count,
                    ),
                    text=source,
                )
            )

        return chunks
