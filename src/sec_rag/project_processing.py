"""Project-level indexing helpers for summaries, sanitization, and checks."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from sec_rag.chunkers.base_chunker import Chunk, ChunkMetadata


@dataclass(frozen=True)
class ProjectContext:
    """Logical project unit derived from an input folder."""

    project_id: str
    project_name: str
    project_root: str


class SensitiveInfoError(ValueError):
    """Raised when sensitive information is detected in sanitized text."""


class SensitiveInfoChecker:
    """Rule-based validator for sensitive information in sanitized documents."""

    PATTERNS: dict[str, re.Pattern[str]] = {
        "password": re.compile(
            r"(?i)(password|passwd|pwd|secret|token|api[_-]?key)\s*[:=]\s*\S+"
        ),
        "email": re.compile(
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            re.IGNORECASE,
        ),
        "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "url_credentials": re.compile(
            r"\b(?:https?|ssh)://[^\s:@/]+:[^\s@/]+@",
            re.IGNORECASE,
        ),
    }
    PHONE_CANDIDATE = re.compile(
        r"(?<![0-9A-Za-z_])(?:\+?\d[\d\s\-().]{6,}\d)(?![0-9A-Za-z_])"
    )

    def check_text(self, text: str, *, context: str) -> None:
        """Raise if sensitive information patterns are found."""
        hits = [name for name, pattern in self.PATTERNS.items() if pattern.search(text)]
        if self._contains_phone(text):
            hits.append("phone")
        if hits:
            raise SensitiveInfoError(
                "Sensitive information detected in "
                f"{context}: {', '.join(sorted(hits))}"
            )

    def _contains_phone(self, text: str) -> bool:
        """Detect phone-like strings while avoiding numeric constants in code."""
        for match in self.PHONE_CANDIDATE.finditer(text):
            candidate = match.group(0)
            digits = re.sub(r"\D", "", candidate)
            if len(digits) < 7 or len(digits) > 15:
                continue

            # Require explicit formatting separators so plain constants like
            # 100000000UL and 115200 are not treated as phone numbers.
            separator_count = len(re.findall(r"[\s\-().]", candidate))
            if separator_count < 1:
                continue

            return True
        return False


def create_project_context(project_path: str) -> ProjectContext:
    """Create a stable project context from an input folder path."""
    resolved = str(Path(project_path).resolve())
    project_name = Path(resolved).name or "project"
    project_id = hashlib.sha256(resolved.encode()).hexdigest()[:12]
    return ProjectContext(
        project_id=project_id,
        project_name=project_name,
        project_root=resolved,
    )


def annotate_chunk(
    chunk: Chunk,
    project: ProjectContext,
    *,
    document_kind: str,
    sanitized: bool = False,
    declaration: str = "",
    semantic_summary: str = "",
    is_private: bool = False,
    privacy_summary: str = "",
) -> Chunk:
    """Attach project metadata to a chunk."""
    chunk.metadata.project_id = project.project_id
    chunk.metadata.project_name = project.project_name
    chunk.metadata.project_root = project.project_root
    chunk.metadata.document_kind = document_kind
    chunk.metadata.sanitized = sanitized
    chunk.metadata.source_language_family = _language_family(chunk.metadata.language)
    chunk.metadata.declaration = declaration
    chunk.metadata.semantic_summary = semantic_summary
    chunk.metadata.is_private = is_private
    chunk.metadata.privacy_summary = privacy_summary
    return chunk


def sanitize_code_chunk(
    chunk: Chunk,
    llm,
    project: ProjectContext,
    checker: SensitiveInfoChecker,
) -> Chunk:
    """Create a sanitized query document for C/C++ style chunks."""
    declaration = extract_declaration(chunk)
    semantic_summary = summarize_chunk_behavior(chunk, llm, declaration)
    sanitized_text = build_sanitized_document(
        chunk=chunk,
        project=project,
        declaration=declaration,
        semantic_summary=semantic_summary,
    )
    checker.check_text(
        sanitized_text,
        context=f"{project.project_name}:{chunk.metadata.file_path}:{chunk.metadata.name}",
    )
    chunk.text = sanitized_text
    return annotate_chunk(
        chunk,
        project,
        document_kind="sanitized_code",
        sanitized=True,
        declaration=declaration,
        semantic_summary=semantic_summary,
    )


def summarize_project(project: ProjectContext, chunks: list[Chunk], llm) -> str:
    """Generate a short project summary used as the first retrieval hop."""
    sample_lines = []
    for chunk in chunks[:8]:
        summary_line = chunk.metadata.semantic_summary or chunk.metadata.name
        sample_lines.append(
            f"- {chunk.metadata.language}:{chunk.metadata.name} -> {summary_line[:140]}"
        )

    if not sample_lines:
        return f"Project {project.project_name} contains no indexable documents."

    if llm is None:
        return (
            f"Project {project.project_name} focuses on "
            f"{len(chunks)} indexed documents. "
            "Representative symbols include "
            f"{', '.join(chunk.metadata.name for chunk in chunks[:3])}."
        )

    prompt = (
        "Summarize this project in at most three concise sentences. "
        "Do not copy code. Focus on responsibilities, modules, and retrieval hints.\n\n"
        f"Project: {project.project_name}\n"
        f"Root: {project.project_root}\n"
        "Documents:\n"
        + "\n".join(sample_lines)
    )
    try:
        return str(llm.complete(prompt)).strip()
    except Exception as exc:
        logger.warning(
            "Project summary generation failed for {}: {}",
            project.project_name,
            exc,
        )
        return (
            f"Project {project.project_name} includes "
            f"{len(chunks)} indexed documents and "
            f"symbols such as {', '.join(chunk.metadata.name for chunk in chunks[:3])}."
        )


def build_project_summary_chunk(project: ProjectContext, summary_text: str) -> Chunk:
    """Create a chunk stored in the project summary index."""
    metadata = ChunkMetadata(
        file_path=project.project_root,
        language="project",
        chunk_type="project_summary",
        name=project.project_name,
        chunk_id=f"project:{project.project_id}:summary",
        line_start=0,
        line_end=0,
        project_id=project.project_id,
        project_name=project.project_name,
        project_root=project.project_root,
        document_kind="project_summary",
        sanitized=True,
        source_language_family="project",
        declaration=project.project_name,
        semantic_summary=summary_text,
    )
    return Chunk(metadata=metadata, text=summary_text)


def extract_declaration(chunk: Chunk) -> str:
    """Build a concise symbol declaration without preserving implementation."""
    text = chunk.text.strip()
    if chunk.metadata.language in {"c", "cpp"}:
        prefix = text.split("{", 1)[0].strip()
        declaration = " ".join(
            line.strip() for line in prefix.splitlines() if line.strip()
        )
        if declaration and not declaration.endswith(";"):
            declaration = f"{declaration};"
        return declaration or f"{chunk.metadata.chunk_type} {chunk.metadata.name};"
    return f"{chunk.metadata.chunk_type} {chunk.metadata.name}"


def summarize_chunk_behavior(chunk: Chunk, llm, declaration: str) -> str:
    """Turn implementation text into a non-code behavioral description."""
    if llm is None:
        return (
            f"{chunk.metadata.name} is a {chunk.metadata.chunk_type} in "
            f"{chunk.metadata.language} "
            f"defined at lines {chunk.metadata.line_start}-{chunk.metadata.line_end}."
        )

    snippet = chunk.text[:2000]
    prompt = (
        "Convert the following implementation into a concise natural-language "
        "description. Do not copy code, string literals, IPs, credentials, or "
        "contact details. Summarize responsibilities, inputs/outputs, state "
        "transitions, and dependencies.\n\n"
        f"Declaration: {declaration}\n"
        f"Language: {chunk.metadata.language}\n"
        f"Implementation snippet:\n{snippet}"
    )
    try:
        return str(llm.complete(prompt)).strip()
    except Exception as exc:
        logger.warning(
            "Implementation summarization failed for {}: {}",
            chunk.metadata.name,
            exc,
        )
        return (
            f"{chunk.metadata.name} performs {chunk.metadata.chunk_type} logic in "
            f"{chunk.metadata.language} and is defined at lines "
            f"{chunk.metadata.line_start}-{chunk.metadata.line_end}."
        )


def build_sanitized_document(
    *,
    chunk: Chunk,
    project: ProjectContext,
    declaration: str,
    semantic_summary: str,
) -> str:
    """Format the sanitized document that becomes the indexed text."""
    return "\n".join(
        [
            f"project: {project.project_name}",
            f"project_root: {project.project_root}",
            f"file: {chunk.metadata.file_path}",
            f"language: {chunk.metadata.language}",
            f"symbol_type: {chunk.metadata.chunk_type}",
            f"symbol_name: {chunk.metadata.name}",
            f"line_range: {chunk.metadata.line_start}-{chunk.metadata.line_end}",
            f"declaration: {declaration}",
            f"summary: {semantic_summary}",
        ]
    )


def process_privacy_chunk(
    chunk: Chunk,
    llm,
    project: ProjectContext,
) -> Chunk:
    """Mark a chunk as private and generate a privacy-safe summary.

    The original text is preserved in the index for retrieval relevance,
    but a privacy-safe summary is generated for LLM prompt substitution.
    """
    declaration = extract_declaration(chunk)
    from sec_rag.privacy import generate_privacy_summary

    privacy_summary = generate_privacy_summary(
        chunk_text=chunk.text,
        file_path=chunk.metadata.file_path,
        llm=llm,
        declaration=declaration,
    )
    return annotate_chunk(
        chunk,
        project,
        document_kind=chunk.metadata.document_kind,
        sanitized=chunk.metadata.sanitized,
        declaration=declaration,
        semantic_summary=chunk.metadata.semantic_summary,
        is_private=True,
        privacy_summary=privacy_summary,
    )


def _language_family(language: str) -> str:
    if language in {"c", "cpp"}:
        return "c_family"
    if language == "systemc":
        return "systemc_tlm"
    if language == "python":
        return "python"
    if language in {"verilog", "systemverilog"}:
        return "hdl"
    return language