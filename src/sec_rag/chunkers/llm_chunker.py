"""LLM-driven semantic chunker."""

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
        ranges = self._infer_ranges_with_llm(lines, language)
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

    def _infer_ranges_with_llm(self, lines: list[str], language: str) -> list[dict]:
        numbered = "\n".join(f"{idx + 1:04d}: {line}" for idx, line in enumerate(lines))
        prompt = (
            "You are a code/document chunk planner.\n"
            "Split the input into coherent retrieval chunks.\n"
            f"Language: {language}\n"
            f"Max chunks: {self._max_chunks}\n"
            "Return STRICT JSON only, no markdown.\n"
            "Schema: {\"chunks\":[{\"start_line\":1,\"end_line\":10,"
            "\"chunk_type\":\"function|class|module|section|semantic\","
            "\"name\":\"short_name\"}]}\n"
            "Rules: start_line/end_line must be valid and non-overlapping.\n"
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
        return chunks[: self._max_chunks]

    @staticmethod
    def _extract_json(raw: str):
        raw = raw.strip()
        try:
            return json.loads(raw)
        except Exception:
            pass

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
