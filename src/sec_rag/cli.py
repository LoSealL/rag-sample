"""
CLI entry points for sec-rag.

Usage:
    sec-rag index --path ./src --languages python,c
    sec-rag query "how does auth work?" --top-k 5
    sec-rag stats
"""

import os
import sys
from collections import Counter
from pathlib import Path

import click
from dotenv import load_dotenv
from loguru import logger

from .chunkers.chunker_factory import ChunkerFactory
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
DEFAULT_NAIVE_CHUNK_LINES = int(os.environ.get("CHUNK_NAIVE_LINES", "60"))
DEFAULT_NAIVE_OVERLAP_RATIO = float(os.environ.get("CHUNK_OVERLAP_RATIO", "0.2"))
DEFAULT_MAX_TOKENS_BEFORE_SPLIT = int(os.environ.get("CHUNK_MAX_TOKENS", "600"))
DEFAULT_DOC_TARGET_TOKENS = int(os.environ.get("CHUNK_DOC_TARGET_TOKENS", "512"))
DEFAULT_CHUNK_BACKEND = os.environ.get("CHUNK_BACKEND", "ast")
DEFAULT_CHUNK_LLM_MAX_CHUNKS = int(os.environ.get("CHUNK_LLM_MAX_CHUNKS", "40"))
PROJECT_SUMMARY_COLLECTION = "project_index"
SUPPORTED_CODE_LANGUAGES = {"python", "c", "cpp", "verilog", "systemverilog"}


def _index_options(func):
    func = click.option(
        "--dry-run",
        is_flag=True,
        help="Analyze files and print stats without indexing or LLM calls",
    )(func)
    func = click.option(
        "--path",
        "paths",
        multiple=True,
        required=True,
        help="Root directory to index (repeatable)",
    )(func)
    func = click.option(
        "--languages",
        default="python,c,cpp,verilog,systemverilog,doc",
        show_default=True,
        help="Comma-separated languages: python,c,cpp,verilog,systemverilog,doc",
    )(func)
    func = click.option(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
        show_default=True,
        help="Directory for ChromaDB index",
    )(func)
    func = click.option(
        "--batch-size",
        type=int,
        default=10,
        show_default=True,
        help="Number of files to process per batch",
    )(func)
    func = click.option("--verbose", "-v", is_flag=True, help="Verbose output")(func)
    func = click.option(
        "--chunk-backend",
        type=click.Choice(["ast", "llm", "hybrid"]),
        default=DEFAULT_CHUNK_BACKEND,
        show_default=True,
        help="Chunking backend",
    )(func)
    func = click.option(
        "--chunk-naive-lines",
        type=int,
        default=DEFAULT_NAIVE_CHUNK_LINES,
        show_default=True,
        help="Lines per fallback/naive chunk",
    )(func)
    func = click.option(
        "--chunk-overlap-ratio",
        type=float,
        default=DEFAULT_NAIVE_OVERLAP_RATIO,
        show_default=True,
        help="Overlap ratio between fallback chunks",
    )(func)
    func = click.option(
        "--chunk-max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS_BEFORE_SPLIT,
        show_default=True,
        help="Token threshold for splitting large code chunks",
    )(func)
    func = click.option(
        "--chunk-doc-target-tokens",
        type=int,
        default=DEFAULT_DOC_TARGET_TOKENS,
        show_default=True,
        help="Target tokens per document chunk",
    )(func)
    func = click.option(
        "--chunk-llm-max-chunks",
        type=int,
        default=DEFAULT_CHUNK_LLM_MAX_CHUNKS,
        show_default=True,
        help="Max chunks emitted by LLM backend",
    )(func)
    func = click.option(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        show_default=True,
        help="Embedding model",
    )(func)
    func = click.option(
        "--embedding-base-url",
        default=DEFAULT_EMBEDDING_BASE_URL,
        show_default=True,
        help="Embedding service base URL",
    )(func)
    func = click.option(
        "--api-base-url",
        default=DEFAULT_API_BASE_URL,
        show_default=True,
        help="LLM service base URL",
    )(func)
    func = click.option(
        "--llm-provider",
        type=click.Choice(["auto", "openai", "anthropic"]),
        default=DEFAULT_LLM_PROVIDER,
        show_default=True,
        help="API style for local LLM",
    )(func)
    func = click.option(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        show_default=True,
        help="LLM model name",
    )(func)
    func = click.option(
        "--llm-timeout",
        type=int,
        default=DEFAULT_LLM_TIMEOUT_SECONDS,
        show_default=True,
        help="LLM request timeout in seconds",
    )(func)
    return func


def _query_options(func):
    func = click.argument("query", type=str)(func)
    func = click.option(
        "--dry-run",
        is_flag=True,
        help="Retrieve and print stats only, without generating final answer",
    )(func)
    func = click.option(
        "--top-k",
        type=int,
        default=5,
        show_default=True,
        help="Number of chunks to retrieve",
    )(func)
    func = click.option(
        "--filter-type",
        type=click.Choice(["code", "doc"]),
        help="Filter by type",
    )(func)
    func = click.option(
        "--filter-lang",
        type=click.Choice(["python", "c", "cpp", "markdown", "docx"]),
        help="Filter by language",
    )(func)
    func = click.option(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
        show_default=True,
        help="Directory for ChromaDB index",
    )(func)
    func = click.option(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        show_default=True,
        help="Embedding model",
    )(func)
    func = click.option(
        "--embedding-base-url",
        default=DEFAULT_EMBEDDING_BASE_URL,
        show_default=True,
        help="Embedding service base URL",
    )(func)
    func = click.option(
        "--api-base-url",
        default=DEFAULT_API_BASE_URL,
        show_default=True,
        help="LLM service base URL",
    )(func)
    func = click.option(
        "--llm-provider",
        type=click.Choice(["auto", "openai", "anthropic"]),
        default=DEFAULT_LLM_PROVIDER,
        show_default=True,
        help="API style for local LLM",
    )(func)
    func = click.option(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        show_default=True,
        help="LLM model name",
    )(func)
    func = click.option(
        "--llm-timeout",
        type=int,
        default=DEFAULT_LLM_TIMEOUT_SECONDS,
        show_default=True,
        help="LLM request timeout in seconds",
    )(func)
    func = click.option("--verbose", "-v", is_flag=True, help="Verbose output")(func)
    func = click.option(
        "--show-chunks",
        is_flag=True,
        help="Show retrieved chunks before the answer",
    )(func)
    return func


def _stats_options(func):
    func = click.option(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
        show_default=True,
        help="Directory for ChromaDB index",
    )(func)
    return func


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


def _run_index(
    paths: tuple[str, ...],
    dry_run: bool,
    languages: str,
    index_dir: str,
    batch_size: int,
    verbose: bool,
    chunk_backend: str,
    chunk_naive_lines: int,
    chunk_overlap_ratio: float,
    chunk_max_tokens: int,
    chunk_doc_target_tokens: int,
    chunk_llm_max_chunks: int,
    embedding_model: str,
    embedding_base_url: str,
    api_base_url: str,
    llm_provider: str,
    llm_model: str,
    llm_timeout: int,
) -> int:
    configure_logging(verbose)

    # Parse languages
    lang_map: dict[str, set[str]] = {
        "python": {"python"},
        "c": {"c"},
        "cpp": {"cpp"},
        "verilog": {"verilog"},
        "systemverilog": {"systemverilog"},
        "doc": {"doc"},
    }
    selected_languages: set[str] = set()
    for lang in languages.split(","):
        lang = lang.strip().lower()
        if lang in lang_map:
            selected_languages.update(lang_map[lang])
        elif lang in SUPPORTED_CODE_LANGUAGES:
            selected_languages.add(lang)

    if not selected_languages:
        logger.error("No valid languages specified")
        return 1

    if dry_run:
        files_total = 0
        lines_total = 0
        est_tokens_total = 0
        ext_counter: Counter[str] = Counter()

        logger.info("[DRY-RUN] Index analysis started")
        for project_path in paths:
            project = create_project_context(project_path)
            files = discover_files(project.project_root, selected_languages)
            logger.info(
                "[DRY-RUN] Project {}: {} candidate files",
                project.project_name,
                len(files),
            )
            for file_path in files:
                try:
                    text = file_path.read_text(encoding="utf-8", errors="replace")
                except Exception as exc:
                    logger.warning(
                        "[DRY-RUN] Skip unreadable file {}: {}",
                        file_path,
                        exc,
                    )
                    continue
                line_count = text.count("\n") + (1 if text else 0)
                est_tokens = max(1, len(text) // 4) if text else 0
                logger.info(
                    "[DRY-RUN] file={} lines={} est_tokens={}",
                    file_path,
                    line_count,
                    est_tokens,
                )
                files_total += 1
                lines_total += line_count
                est_tokens_total += est_tokens
                ext_counter[file_path.suffix.lower() or "<noext>"] += 1

            logger.info(
                (
                    "[DRY-RUN] Project {} aggregation: "
                    "summary_chunk={} (if chunking yields docs)"
                ),
                project.project_name,
                "would-generate" if files else "none",
            )

        logger.info("[DRY-RUN] --- Aggregate ---")
        logger.info("[DRY-RUN] total_files={}", files_total)
        logger.info("[DRY-RUN] total_lines={}", lines_total)
        logger.info("[DRY-RUN] estimated_tokens={}", est_tokens_total)
        if ext_counter:
            logger.info("[DRY-RUN] file distribution by extension:")
            for ext, cnt in sorted(ext_counter.items(), key=lambda x: (-x[1], x[0])):
                logger.info("[DRY-RUN]   {}: {}", ext, cnt)
        return 0

    # Initialize index store
    index_store = IndexStore(persist_dir=index_dir)
    checker = SensitiveInfoChecker()
    try:
        embed_fn, _ = build_embedding_function(
            embedding_model,
            base_url=embedding_base_url,
        )
    except Exception as exc:
        logger.error(
            "Failed to initialize local embedding client for indexing: {}", exc
        )
        return 1
    try:
        llm = build_llm(
            provider=llm_provider,
            model=llm_model,
            base_url=api_base_url,
            timeout=llm_timeout,
        )
    except Exception as exc:
        logger.error("Failed to initialize local LLM for indexing: {}", exc)
        return 1

    # Initialize unified chunker
    chunker = ChunkerFactory(
        llm=llm,
        chunk_backend=chunk_backend,
        naive_chunk_lines=chunk_naive_lines,
        naive_overlap_ratio=chunk_overlap_ratio,
        max_tokens_before_split=chunk_max_tokens,
        doc_target_tokens=chunk_doc_target_tokens,
        llm_max_chunks=chunk_llm_max_chunks,
    )

    total_chunks = 0
    files_processed = 0

    for project_path in paths:
        project = create_project_context(project_path)
        logger.info(
            "Discovering files for project {} in {}...",
            project.project_name,
            project.project_root,
        )
        files = discover_files(project.project_root, selected_languages)
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
        for i in range(0, len(files), batch_size):
            batch = files[i : i + batch_size]
            logger.info(
                "Processing project {} batch {}/{} ({} files)...",
                project.project_name,
                i // batch_size + 1,
                (len(files) + batch_size - 1) // batch_size,
                len(batch),
            )

            batch_chunks = []

            for file_path in batch:
                try:
                    if not chunker.supports_file(str(file_path)):
                        continue
                    chunks = chunker.chunk_file(str(file_path))
                    if not chunks:
                        continue
                    kind = chunker.document_kind_for_file(str(file_path))
                    if chunker.is_cpp_file(str(file_path)):
                        chunks = [
                            sanitize_code_chunk(chunk, llm, project, checker)
                            for chunk in chunks
                        ]
                    for chunk in chunks:
                        annotate_chunk(chunk, project, document_kind=kind)
                    batch_chunks.extend(chunks)
                    project_documents.extend(chunks)
                    files_processed += 1
                except Exception as e:
                    logger.warning("Failed to chunk {}: {}", file_path, e)
                    continue

            if batch_chunks:
                index_store.add_chunks("rag_index", batch_chunks, embed_fn=embed_fn)
                total_chunks += len(batch_chunks)

            logger.info("Batch done: {} chunks", len(batch_chunks))

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
        "Indexing complete: {} files, {} total chunks",
        files_processed,
        total_chunks,
    )
    return 0


def _run_query(
    query: str,
    dry_run: bool,
    top_k: int,
    filter_type: str | None,
    filter_lang: str | None,
    index_dir: str,
    embedding_model: str,
    embedding_base_url: str,
    api_base_url: str,
    llm_provider: str,
    llm_model: str,
    llm_timeout: int,
    verbose: bool,
    show_chunks: bool,
) -> int:
    configure_logging(verbose)

    # Build embedding function
    _, embedding_client = build_embedding_function(
        embedding_model,
        base_url=embedding_base_url,
    )

    llm = None
    if not dry_run:
        llm = build_llm(
            provider=llm_provider,
            model=llm_model,
            base_url=api_base_url,
            timeout=llm_timeout,
        )

    # Build pipeline
    index_store = IndexStore(persist_dir=index_dir)
    pipeline = QueryPipeline(index_store, embedding_client, llm)

    # Build filter kwargs
    filter_lang = "cpp" if filter_lang == "c" else filter_lang

    logger.info("Query: {}", query)
    if filter_lang:
        logger.info("Language filter: {}", filter_lang)
    if filter_type:
        logger.info("Type filter: {}", filter_type)

    if dry_run:
        try:
            retrieval = pipeline.retrieve(
                query,
                top_k=top_k,
                filter_language=filter_lang,
                filter_chunk_type=filter_type if filter_type else None,
            )
        except Exception as e:
            logger.error("Query dry-run failed: {}", e)
            return 1

        logger.info("[DRY-RUN] --- Retrieved Chunks ---")
        for i, chunk in enumerate(retrieval.chunks, 1):
            metadata = chunk.get("metadata", {})
            logger.info(
                "[{}] score={:.4f} file={} type={} lang={} lines={}-{}",
                i,
                float(chunk.get("score", 0.0)),
                metadata.get("file_path", "?"),
                metadata.get("chunk_type", "?"),
                metadata.get("language", "?"),
                metadata.get("line_start", "?"),
                metadata.get("line_end", "?"),
            )

        lang_counter: Counter[str] = Counter()
        type_counter: Counter[str] = Counter()
        project_counter: Counter[str] = Counter()
        scores: list[float] = []
        for chunk in retrieval.chunks:
            md = chunk.get("metadata", {})
            lang_counter[str(md.get("language", "unknown"))] += 1
            type_counter[str(md.get("chunk_type", "unknown"))] += 1
            project_name = str(md.get("project_name", md.get("project_id", "unknown")))
            project_counter[project_name] += 1
            try:
                scores.append(float(chunk.get("score", 0.0)))
            except Exception:
                pass

        logger.info("[DRY-RUN] --- Distribution ---")
        logger.info("[DRY-RUN] hits={}", len(retrieval.chunks))
        logger.info("[DRY-RUN] candidate_projects={}", retrieval.candidate_project_ids)
        if scores:
            logger.info(
                "[DRY-RUN] score min={:.4f} max={:.4f} avg={:.4f}",
                min(scores),
                max(scores),
                sum(scores) / len(scores),
            )
        logger.info("[DRY-RUN] by_language={}", dict(lang_counter))
        logger.info("[DRY-RUN] by_chunk_type={}", dict(type_counter))
        logger.info("[DRY-RUN] by_project={}", dict(project_counter))
        logger.info("[DRY-RUN] prompt_tokens_est={}", retrieval.total_prompt_tokens)
        return 0

    try:
        result = pipeline.run(
            query,
            top_k=top_k,
            filter_language=filter_lang,
            filter_chunk_type=filter_type if filter_type else None,
        )
    except Exception as e:
        logger.error("Query failed: {}", e)
        return 1

    # Show chunks
    if show_chunks or verbose:
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


def _run_stats(index_dir: str) -> int:
    index_store = IndexStore(persist_dir=index_dir)
    stats = index_store.get_stats()

    if not stats:
        logger.info("No collections found in index.")
        return 0

    logger.info("Index directory: {}", os.path.abspath(index_dir))
    logger.info("Collections:")
    total = 0
    for coll_name, info in stats.items():
        count = info.get("count", 0)
        total += count
        logger.info("  {}: {} chunks", coll_name, count)

    logger.info("Total: {} chunks", total)

    # Index size on disk
    index_path = Path(index_dir)
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
@_index_options
def index(**kwargs) -> None:
    """Index code and documents for RAG retrieval."""
    _exit_on_code(_run_index(**kwargs))


@cli.command("query")
@_query_options
def query(**kwargs) -> None:
    """Ask a question about the indexed codebase."""
    _exit_on_code(_run_query(**kwargs))


@cli.command("stats")
@_stats_options
def stats(**kwargs) -> None:
    """Show statistics about the RAG index."""
    _exit_on_code(_run_stats(**kwargs))


if __name__ == "__main__":
    cli(prog_name="sec-rag")
