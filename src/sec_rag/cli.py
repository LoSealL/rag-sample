"""
CLI entry points for sec-rag.

Usage:
    sec-rag-index --path ./src --languages python,c
    sec-rag-query "how does auth work?" --top-k 5
    sec-rag-stats
"""

import argparse
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from loguru import logger

from .chunkers.code_chunker import CodeChunker
from .chunkers.doc_chunker import DocChunker
from .chunkers.verilog_chunker import VerilogChunker
from .index_store import IndexStore
from .project_processing import (
    SensitiveInfoChecker,
    annotate_chunk,
    build_project_summary_chunk,
    create_project_context,
    sanitize_code_chunk,
    summarize_project,
)
from .query_pipeline import QueryPipeline, build_embedding_function, build_llm

# Load .env file if it exists
load_dotenv()


def configure_logging(verbose: bool = False) -> None:
    """Configure loguru sinks and level for CLI output."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


configure_logging()


# Default directories and settings
DEFAULT_INDEX_DIR = os.environ.get("RAG_INDEX_DIR", ".rag_index")
DEFAULT_API_BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:8000")
DEFAULT_EMBEDDING_BASE_URL = (
    os.environ.get("EMBEDDING_BASE_URL")
    or os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:8000")
)
DEFAULT_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "Qwen3-VL-Embedding-8B")
DEFAULT_LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "auto")
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen3.5-8B")
DEFAULT_LLM_TIMEOUT_SECONDS = int(os.environ.get("LOCAL_LLM_TIMEOUT_SECONDS", "300"))
PROJECT_SUMMARY_COLLECTION = "project_index"
SUPPORTED_CODE_LANGUAGES = {"python", "c", "cpp", "verilog", "systemverilog"}


def _add_index_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--index-dir",
        type=str,
        default=DEFAULT_INDEX_DIR,
        help=f"Directory for ChromaDB index (default: {DEFAULT_INDEX_DIR})",
    )


def _add_embedding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--embedding-base-url",
        type=str,
        default=DEFAULT_EMBEDDING_BASE_URL,
        help=f"Embedding service base URL (default: {DEFAULT_EMBEDDING_BASE_URL})",
    )


def _add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-base-url",
        type=str,
        default=DEFAULT_API_BASE_URL,
        help=f"LLM service base URL (default: {DEFAULT_API_BASE_URL})",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=DEFAULT_LLM_PROVIDER,
        choices=["auto", "openai", "anthropic"],
        help=f"API style for local LLM (default: {DEFAULT_LLM_PROVIDER})",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=f"LLM model name (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=DEFAULT_LLM_TIMEOUT_SECONDS,
        help=(
            "LLM request timeout in seconds "
            f"(default: {DEFAULT_LLM_TIMEOUT_SECONDS})"
        ),
    )


def _add_verbose_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )


# File exclusion patterns
EXCLUDE_DIRS = {
    "__pycache__",
    "node_modules",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "build",
    "dist",
    ".tox",
    ".eggs",
    "*.egg-info",
}
EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".so",
    ".o",
    ".obj",
    ".exe",
    ".dll",
    ".dylib",
}
EXCLUDE_NAMES = {".DS_Store", "Thumbs.db"}


def is_excluded(path: Path) -> bool:
    """Check if a path should be excluded from indexing."""
    parts = path.parts
    # Check if any directory in the path is excluded
    for part in parts:
        if part in EXCLUDE_DIRS:
            return True
        if part.startswith("."):
            return True
    # Check file extension
    if path.suffix.lower() in EXCLUDE_EXTENSIONS:
        return True
    if path.name in EXCLUDE_NAMES:
        return True
    return False


def discover_files(
    root_path: str,
    languages: set[str],
) -> list[Path]:
    """
    Discover all source files to index under root_path.

    Args:
        root_path: Root directory to search.
        languages: Set of language names ("python", "c", "cpp")

    Returns:
        List of Path objects for files to index.
    """
    lang_extensions: dict[str, list[str]] = {
        "python": [".py"],
        "c": [".c", ".h"],
        "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".hh"],
        "verilog": [".v"],
        "systemverilog": [".sv"],
    }
    doc_extensions = {".md", ".markdown", ".docx"}

    allowed_extensions: set[str] = set()
    for lang in languages:
        if lang in lang_extensions:
            allowed_extensions.update(lang_extensions[lang])
        elif lang == "doc":
            allowed_extensions.update(doc_extensions)

    root = Path(root_path).resolve()
    files: list[Path] = []

    for ext in allowed_extensions:
        for path in root.rglob(f"*{ext}"):
            if is_excluded(path.relative_to(root)):
                continue
            files.append(path)

    return files


def main_index(args: list[str] | None = None) -> int:
    """
    sec-rag-index: Index a directory of code and document files.

    Usage:
        sec-rag-index --path ./src --languages python,c
        sec-rag-index --path ./docs --languages doc
        sec-rag-index --path . --languages python,c,doc
    """
    parser = argparse.ArgumentParser(
        description="Index code and documents for RAG retrieval.",
    )
    parser.add_argument(
        "--path",
        type=str,
        action="append",
        required=True,
        help="Root directory to index",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="python,c,cpp,verilog,systemverilog,doc",
        help=(
            "Comma-separated languages to index: python,c,cpp,verilog,"
            "systemverilog,doc (default: all)"
        ),
    )
    _add_index_dir_arg(parser)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of files to process per batch (default: 10)",
    )
    _add_verbose_arg(parser)
    _add_embedding_args(parser)
    _add_llm_args(parser)

    parsed = parser.parse_args(args)
    configure_logging(parsed.verbose)

    # Parse languages
    lang_map: dict[str, set[str]] = {
        "python": {"python"},
        "c": {"c"},
        "cpp": {"cpp"},
        "verilog": {"verilog"},
        "systemverilog": {"systemverilog"},
        "doc": {"doc"},
    }
    languages: set[str] = set()
    for lang in parsed.languages.split(","):
        lang = lang.strip().lower()
        if lang in lang_map:
            languages.update(lang_map[lang])
        elif lang in SUPPORTED_CODE_LANGUAGES:
            languages.add(lang)

    if not languages:
        logger.error("No valid languages specified")
        return 1

    # Initialize chunkers
    code_chunker = CodeChunker()
    doc_chunker = DocChunker()
    verilog_chunker = VerilogChunker()

    # Initialize index store
    index_store = IndexStore(persist_dir=parsed.index_dir)
    checker = SensitiveInfoChecker()
    try:
        embed_fn, _ = build_embedding_function(
            parsed.embedding_model,
            base_url=parsed.embedding_base_url,
        )
    except Exception as exc:
        logger.error(
            "Failed to initialize local embedding client for indexing: {}", exc
        )
        return 1
    try:
        llm = build_llm(
            provider=parsed.llm_provider,
            model=parsed.llm_model,
            base_url=parsed.api_base_url,
            timeout=parsed.llm_timeout,
        )
    except Exception as exc:
        logger.error("Failed to initialize local LLM for indexing: {}", exc)
        return 1

    total_chunks = 0
    code_files_processed = 0
    doc_files_processed = 0

    for project_path in parsed.path:
        project = create_project_context(project_path)
        logger.info(
            "Discovering files for project {} in {}...",
            project.project_name,
            project.project_root,
        )
        files = discover_files(project.project_root, languages)
        logger.info(
            "Found {} files to index for project {}",
            len(files),
            project.project_name,
        )

        if not files:
            logger.warning(
                "No files found to index for project {}",
                project.project_name,
            )
            continue

        project_documents: list = []
        batch_size = parsed.batch_size
        for i in range(0, len(files), batch_size):
            batch = files[i : i + batch_size]
            logger.info(
                "Processing project {} batch {}/{} ({} files)...",
                project.project_name,
                i // batch_size + 1,
                (len(files) + batch_size - 1) // batch_size,
                len(batch),
            )

            code_chunks = []
            doc_chunks = []

            for file_path in batch:
                try:
                    ext = file_path.suffix.lower()
                    if ext in (".py",):
                        chunks = code_chunker.chunk_file(str(file_path))
                        for chunk in chunks:
                            annotate_chunk(chunk, project, document_kind="code")
                        if chunks:
                            code_chunks.extend(chunks)
                            project_documents.extend(chunks)
                            code_files_processed += 1
                    elif ext in (".c", ".h", ".cpp", ".cc", ".cxx", ".hpp"):
                        chunks = code_chunker.chunk_file(str(file_path))
                        sanitized_chunks = [
                            sanitize_code_chunk(chunk, llm, project, checker)
                            for chunk in chunks
                        ]
                        if sanitized_chunks:
                            code_chunks.extend(sanitized_chunks)
                            project_documents.extend(sanitized_chunks)
                            code_files_processed += 1
                    elif ext in (".v", ".sv"):
                        chunks = verilog_chunker.chunk_file(str(file_path))
                        for chunk in chunks:
                            annotate_chunk(chunk, project, document_kind="code")
                        if chunks:
                            code_chunks.extend(chunks)
                            project_documents.extend(chunks)
                            code_files_processed += 1
                    elif ext in (".md", ".markdown", ".docx"):
                        chunks = doc_chunker.chunk_file(str(file_path))
                        for chunk in chunks:
                            annotate_chunk(chunk, project, document_kind="doc")
                        if chunks:
                            doc_chunks.extend(chunks)
                            project_documents.extend(chunks)
                            doc_files_processed += 1
                except Exception as e:
                    logger.warning("Failed to chunk {}: {}", file_path, e)
                    continue

            if code_chunks:
                index_store.add_chunks("code_index", code_chunks, embed_fn=embed_fn)
                total_chunks += len(code_chunks)
            if doc_chunks:
                index_store.add_chunks("doc_index", doc_chunks, embed_fn=embed_fn)
                total_chunks += len(doc_chunks)

            logger.info(
                "Batch done: {} code chunks, {} doc chunks",
                len(code_chunks),
                len(doc_chunks),
            )

        if project_documents:
            summary_text = summarize_project(project, project_documents, llm)
            checker.check_text(
                summary_text,
                context=f"project summary:{project.project_name}",
            )
            index_store.add_chunks(
                PROJECT_SUMMARY_COLLECTION,
                [build_project_summary_chunk(project, summary_text)],
                embed_fn=embed_fn,
            )
            total_chunks += 1

    logger.info(
        "Indexing complete: {} files ({} code, {} doc), {} total chunks",
        code_files_processed + doc_files_processed,
        code_files_processed,
        doc_files_processed,
        total_chunks,
    )
    return 0


def main_query(args: list[str] | None = None) -> int:
    """
    sec-rag-query: Ask a question about the indexed codebase.

    Usage:
        sec-rag-query "how does auth work?"
        sec-rag-query "how does auth work?" --top-k 5
        sec-rag-query "..." --filter-lang python
    """
    parser = argparse.ArgumentParser(
        description="Query the RAG index.",
    )
    parser.add_argument(
        "query",
        type=str,
        help="Question to ask about the indexed code/docs",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--filter-type",
        type=str,
        choices=["code", "doc"],
        help="Filter by type: code or doc",
    )
    parser.add_argument(
        "--filter-lang",
        type=str,
        choices=["python", "c", "cpp", "markdown", "docx"],
        help="Filter by language",
    )
    _add_index_dir_arg(parser)
    _add_embedding_args(parser)
    _add_llm_args(parser)
    _add_verbose_arg(parser)
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Show retrieved chunks before the answer",
    )

    parsed = parser.parse_args(args)
    configure_logging(parsed.verbose)

    # Build embedding function and LLM
    _, embedding_model = build_embedding_function(
        parsed.embedding_model,
        base_url=parsed.embedding_base_url,
    )
    llm = build_llm(
        provider=parsed.llm_provider,
        model=parsed.llm_model,
        base_url=parsed.api_base_url,
        timeout=parsed.llm_timeout,
    )

    # Build pipeline
    index_store = IndexStore(persist_dir=parsed.index_dir)
    pipeline = QueryPipeline(index_store, embedding_model, llm)

    # Build filter kwargs
    filter_lang = parsed.filter_lang
    filter_type = parsed.filter_type

    logger.info("Query: {}", parsed.query)
    if filter_lang:
        logger.info("Language filter: {}", filter_lang)
    if filter_type:
        logger.info("Type filter: {}", filter_type)

    try:
        result = pipeline.run(
            parsed.query,
            top_k=parsed.top_k,
            filter_language=filter_lang,
            filter_chunk_type=filter_type if filter_type else None,
        )
    except Exception as e:
        logger.error("Query failed: {}", e)
        return 1

    # Show chunks
    if parsed.show_chunks or parsed.verbose:
        logger.info("--- Retrieved Chunks ---")
        for i, chunk in enumerate(result.chunks, 1):
            metadata = chunk["metadata"]
            logger.info("[{}] {}", i, metadata.get("file_path", "?"))
            logger.info(
                "    Lines {}-{}",
                metadata.get("line_start", "?"),
                metadata.get("line_end", "?"),
            )
            logger.info(
                "    Type: {} | Score: {:.4f}",
                metadata.get("chunk_type", "?"),
                chunk["score"],
            )
            text = chunk["text"]
            logger.info("    ---")
            logger.info("    {}", f"{text[:300]}{'...' if len(text) > 300 else ''}")

    # Show answer
    logger.success("--- Answer ---")
    logger.success("{}", result.answer)

    # Show token count
    logger.info("Tokens: {} prompt (approx)", result.total_prompt_tokens)

    return 0


def main_stats(args: list[str] | None = None) -> int:
    """
    sec-rag-stats: Show statistics about the RAG index.

    Usage:
        sec-rag-stats
        sec-rag-stats --index-dir .rag_index
    """
    parser = argparse.ArgumentParser(
        description="Show RAG index statistics.",
    )
    _add_index_dir_arg(parser)
    parsed = parser.parse_args(args)

    index_store = IndexStore(persist_dir=parsed.index_dir)
    stats = index_store.get_stats()

    if not stats:
        logger.info("No collections found in index.")
        return 0

    logger.info("Index directory: {}", os.path.abspath(parsed.index_dir))
    logger.info("Collections:")
    total = 0
    for coll_name, info in stats.items():
        count = info.get("count", 0)
        total += count
        logger.info("  {}: {} chunks", coll_name, count)

    logger.info("Total: {} chunks", total)

    # Index size on disk
    index_path = Path(parsed.index_dir)
    if index_path.exists():
        size_bytes = sum(f.stat().st_size for f in index_path.rglob("*") if f.is_file())
        size_mb = size_bytes / (1024 * 1024)
        logger.info("Index size: {:.1f} MB", size_mb)

    return 0


def _exit_on_code(code: int) -> None:
    """Convert return codes to click exits."""
    if code != 0:
        raise click.exceptions.Exit(code=code)


@click.group(help="sec-rag command group.")
def cli() -> None:
    """Unified CLI entrypoint for index/query/stats subcommands."""


@cli.command("index")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def click_index(args: tuple[str, ...]) -> None:
    """Run the index command using argparse backend."""
    _exit_on_code(main_index(list(args)))


@cli.command("query")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def click_query(args: tuple[str, ...]) -> None:
    """Run the query command using argparse backend."""
    _exit_on_code(main_query(list(args)))


@cli.command("stats")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def click_stats(args: tuple[str, ...]) -> None:
    """Run the stats command using argparse backend."""
    _exit_on_code(main_stats(list(args)))


if __name__ == "__main__":
    cli(prog_name="sec-rag")
