"""
Query pipeline — orchestrates embedding, retrieval, and LLM generation.

Usage:
    pipeline = QueryPipeline(index_store, embedding_model, llm)
    result = pipeline.run("how does auth work?", top_k=5)
"""

import importlib
import json
import os
import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Any
from urllib import error, request

import tiktoken
from loguru import logger

# Default tokenizer for token counting (cl100k_base = GPT-4/GPT-4o / Anthropic)
DEFAULT_TOKENIZER = "cl100k_base"
DEFAULT_LOCAL_BASE_URL = "http://localhost:8000"
DEFAULT_LOCAL_LLM_MODEL = "Qwen3.5-8B"
DEFAULT_LOCAL_EMBEDDING_MODEL = "Qwen3-VL-Embedding-8B"
DEFAULT_LOCAL_LLM_TIMEOUT_SECONDS = 300
# Read max_tokens from env, default to None (unlimited)
_max_tokens_env = os.environ.get("LOCAL_LLM_MAX_TOKENS")
DEFAULT_LOCAL_LLM_MAX_TOKENS = int(_max_tokens_env) if _max_tokens_env else None
DEFAULT_LOG_LLM_PAYLOAD = (
    os.environ.get("SEC_RAG_LOG_LLM_PAYLOAD", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)
DEFAULT_LOG_LLM_PAYLOAD_PREVIEW_CHARS = int(
    os.environ.get("SEC_RAG_LOG_LLM_PAYLOAD_PREVIEW_CHARS", "800")
)
DEFAULT_ENABLE_RERANK = (
    os.environ.get("SEC_RAG_ENABLE_RERANK", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)
DEFAULT_RERANK_MODEL = os.environ.get(
    "SEC_RAG_RERANK_MODEL", "BAAI/bge-reranker-base"
)
DEFAULT_RERANK_CANDIDATE_MULTIPLIER = int(
    os.environ.get("SEC_RAG_RERANK_CANDIDATE_MULTIPLIER", "4")
)
PROJECT_SUMMARY_COLLECTION = "project_index"
PRIMARY_COLLECTION = "rag_index"
LEGACY_COLLECTIONS = ["code_index", "doc_index"]

SOURCE_EXPOSURE_BLOCK_MESSAGE = (
    "基于安全策略，我不能直接返回源码原文。"
    "我可以提供功能说明、模块职责、调用关系和关键风险点的摘要。"
)


@dataclass
class QueryResult:
    """Result of a RAG query."""

    answer: str
    chunks: list[dict]
    total_prompt_tokens: int
    total_chunk_tokens: int


@dataclass
class RetrievalResult:
    """Retrieval-only result without LLM generation."""

    chunks: list[dict]
    total_prompt_tokens: int
    total_chunk_tokens: int
    candidate_project_ids: list[str]


class QueryPipeline:
    """
    End-to-end RAG query pipeline.

    Orchestrates: embed query → ChromaDB retrieval → assemble prompt → LLM generation.

    Usage:
        pipeline = QueryPipeline(index_store, embedding_model, llm)
        result = pipeline.run("how does auth work?", top_k=5)
    """

    def __init__(
        self,
        index_store,
        embedding_model,  # sentence-transformers model
        llm,  # llama-index LLM
        tokenizer_name: str = DEFAULT_TOKENIZER,
        enable_rerank: bool = DEFAULT_ENABLE_RERANK,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        rerank_candidate_multiplier: int = DEFAULT_RERANK_CANDIDATE_MULTIPLIER,
    ):
        """
        Initialize the query pipeline.

        Args:
            index_store: IndexStore instance.
            embedding_model: sentence-transformers model instance.
            llm: llama-index LLM instance (OpenAI, Anthropic, etc.)
            tokenizer_name: tiktoken encoder to use for token counting.
        """
        self.index_store = index_store
        self.embedding_model = embedding_model
        self.llm = llm
        self.enable_rerank = enable_rerank
        self.rerank_model = rerank_model
        self.rerank_candidate_multiplier = max(1, rerank_candidate_multiplier)
        self._reranker: Any | None = None
        self._rerank_unavailable = False
        try:
            self._encoder = tiktoken.get_encoding(tokenizer_name)
        except Exception:
            logger.warning(
                "Failed to load tiktoken encoder '{}', token counting disabled",
                tokenizer_name,
            )
            self._encoder = None

    def run(
        self,
        query: str,
        top_k: int = 5,
        filter_language: str | None = None,
        filter_chunk_type: str | None = None,
        project_top_k: int = 3,
    ) -> QueryResult:
        """
        Run a RAG query end-to-end.

        Args:
            query: Natural language question.
            top_k: Number of chunks to retrieve per collection.
            filter_language: Optional language filter.
            filter_chunk_type: Optional chunk type filter.
            project_top_k: Number of candidate projects to fetch from summary index.

        Returns:
            QueryResult with the LLM answer, retrieved chunks, and token counts.
        """

        retrieval = self.retrieve(
            query=query,
            top_k=top_k,
            filter_language=filter_language,
            filter_chunk_type=filter_chunk_type,
            project_top_k=project_top_k,
        )

        # Build the prompt and call LLM
        prompt = self._build_prompt(retrieval.chunks, query)
        answer = self._call_llm(prompt)

        return QueryResult(
            answer=answer,
            chunks=retrieval.chunks,
            total_prompt_tokens=retrieval.total_prompt_tokens,
            total_chunk_tokens=retrieval.total_chunk_tokens,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_language: str | None = None,
        filter_chunk_type: str | None = None,
        project_top_k: int = 3,
    ) -> RetrievalResult:
        """Retrieve top chunks and prompt token estimate without calling LLM."""

        def embed(query_text: str) -> list[float]:
            return self.embedding_model.encode(query_text).tolist()

        candidate_project_ids = self._query_project_candidates(
            query,
            embed,
            project_top_k,
        )

        all_chunks: list[dict] = []
        if candidate_project_ids:
            for project_id in candidate_project_ids:
                all_chunks.extend(
                    self.index_store.query(
                        PRIMARY_COLLECTION,
                        query,
                        embed,
                        top_k=top_k,
                        filter_language=filter_language,
                        filter_chunk_type=filter_chunk_type,
                        filter_project_id=project_id,
                    )
                )
        else:
            all_chunks = self.index_store.query(
                PRIMARY_COLLECTION,
                query,
                embed,
                top_k=top_k,
                filter_language=filter_language,
                filter_chunk_type=filter_chunk_type,
            )

        # Backward-compatible fallback: if unified collection has no hits,
        # query legacy code/doc collections and merge results.
        if not all_chunks:
            logger.info(
                "No hits in {}. Fallback to legacy collections: {}",
                PRIMARY_COLLECTION,
                ", ".join(LEGACY_COLLECTIONS),
            )
            for collection in LEGACY_COLLECTIONS:
                try:
                    all_chunks.extend(
                        self.index_store.query(
                            collection,
                            query,
                            embed,
                            top_k=top_k,
                            filter_language=filter_language,
                            filter_chunk_type=filter_chunk_type,
                        )
                    )
                except Exception as exc:
                    logger.debug(
                        "Legacy fallback query skipped for {}: {}",
                        collection,
                        exc,
                    )

        all_chunks.sort(key=lambda x: x["score"])

        # Deduplicate by chunk_id; keep only the top-scoring duplicate.
        seen_ids: set[str] = set()
        unique_chunks: list[dict] = []
        for chunk in all_chunks:
            chunk_id = chunk["chunk_id"]
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_chunks.append(chunk)

        if self.enable_rerank:
            top_chunks = self._rerank_chunks(query, unique_chunks, top_k)
        else:
            top_chunks = unique_chunks[:top_k]

        # Privacy detection: log warning if any private chunks are retrieved
        private_chunks = [
            c for c in top_chunks
            if c.get("metadata", {}).get("is_private")
        ]
        if private_chunks:
            logger.opt(colors=True).warning(
                "<yellow>[PRIVACY ALERT]</yellow> Query retrieved {} high-confidential chunk(s): {}",
                len(private_chunks),
                ", ".join(
                    f"{c['metadata'].get('file_path', '?')}:{c['metadata'].get('name', '?')}"
                    for c in private_chunks
                ),
            )

        # Build prompt for token accounting only.
        prompt = self._build_prompt(top_chunks, query)
        chunk_tokens = self._count_tokens(prompt)
        total_tokens = chunk_tokens + 60  # ~60 tokens for system/query framing

        return RetrievalResult(
            chunks=top_chunks,
            total_prompt_tokens=total_tokens,
            total_chunk_tokens=chunk_tokens,
            candidate_project_ids=candidate_project_ids,
        )

    def _get_reranker(self):
        if self._rerank_unavailable:
            return None
        if self._reranker is not None:
            return self._reranker

        try:
            postprocessor_module = importlib.import_module(
                "llama_index.core.postprocessor"
            )
            sentence_transformer_rerank = getattr(
                postprocessor_module,
                "SentenceTransformerRerank",
            )

            self._reranker = sentence_transformer_rerank(
                model=self.rerank_model,
                top_n=None,
            )
            logger.info(
                "SentenceTransformerRerank enabled with model {}",
                self.rerank_model,
            )
            return self._reranker
        except Exception as exc:
            logger.warning(
                (
                    "SentenceTransformerRerank unavailable ({}); "
                    "fallback to vector ranking only"
                ),
                exc,
            )
            self._rerank_unavailable = True
            return None

    def _rerank_chunks(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        reranker = self._get_reranker()
        if reranker is None:
            return chunks[:top_k]

        candidate_k = min(
            len(chunks),
            max(top_k, top_k * self.rerank_candidate_multiplier),
        )
        candidates = chunks[:candidate_k]

        try:
            schema_module = importlib.import_module("llama_index.core.schema")
            node_with_score_cls = getattr(schema_module, "NodeWithScore")
            text_node_cls = getattr(schema_module, "TextNode")

            nodes = [
                node_with_score_cls(
                    node=text_node_cls(
                        id_=chunk["chunk_id"],
                        text=chunk.get("text", ""),
                        metadata=chunk.get("metadata", {}),
                    ),
                    score=float(chunk.get("score", 0.0)),
                )
                for chunk in candidates
            ]
            reranked_nodes = reranker.postprocess_nodes(nodes, query_str=query)

            ranked: list[dict] = []
            by_id = {chunk["chunk_id"]: chunk for chunk in candidates}
            for node in reranked_nodes:
                node_id = getattr(node.node, "node_id", None)
                if not node_id:
                    continue
                chunk = by_id.get(node_id)
                if not chunk:
                    continue
                score = getattr(node, "score", None)
                if score is not None:
                    chunk = {**chunk, "score": score}
                ranked.append(chunk)

            if not ranked:
                return candidates[:top_k]
            return ranked[:top_k]
        except Exception as exc:
            logger.warning("Rerank failed ({}); fallback to vector ranking only", exc)
            return candidates[:top_k]

    def _query_project_candidates(
        self,
        query: str,
        embedding_function,
        project_top_k: int,
    ) -> list[str]:
        """Resolve project candidates from the summary collection before code lookup."""
        try:
            results = self.index_store.query(
                PROJECT_SUMMARY_COLLECTION,
                query,
                embedding_function,
                top_k=project_top_k,
                filter_document_kind="project_summary",
            )
        except Exception as exc:
            logger.debug("Project summary retrieval skipped: {}", exc)
            return []

        project_ids: list[str] = []
        for result in results:
            project_id = result.get("metadata", {}).get("project_id")
            if project_id and project_id not in project_ids:
                project_ids.append(project_id)
        return project_ids

    def _build_prompt(self, chunks: list[dict], query: str) -> str:
        """
        Build the prompt from retrieved chunks.

        Format:
        System: You are a helpful coding assistant. Answer based only on the
        provided context. If the context does not contain the answer, say so.

        Context:
        ---
        File: {file_path} (lines {line_start}-{line_end}), type={chunk_type}
        ```{language}
        {chunk_text}
        ```
        ---
        [... more chunks ...]

        User: {query}
        """
        system_prompt = (
            "You are a helpful coding assistant. Answer the user's question "
            "based only on the provided context. If the context does not "
            "contain the answer, say so. Do not invent code or concepts not "
            "present in the context. Never return verbatim source code from the "
            "context, including fenced code blocks or long line-by-line snippets; "
            "provide concise summaries instead."
        )

        context_parts = []
        for chunk in chunks:
            metadata = chunk["metadata"]
            file_path = metadata.get("file_path", "?")
            line_start = metadata.get("line_start", "?")
            line_end = metadata.get("line_end", "?")
            chunk_type = metadata.get("chunk_type", "?")
            project_name = metadata.get("project_name")
            language = metadata.get("language", "")
            lang_block = (
                "python"
                if language == "python"
                else ("c" if language == "c" else language)
            )
            is_private = metadata.get("is_private", False)
            privacy_summary = metadata.get("privacy_summary", "")

            # Substitute privacy-safe summary for private chunks
            if is_private:
                text = (
                    f"[PRIVACY-SAFE SUMMARY] {privacy_summary}\n\n"
                    f"[NOTE: The raw implementation of this high-confidential "
                    f"component is withheld per privacy policy. "
                    f"Only the summary above is provided.]"
                )
            else:
                text = chunk["text"]

            header = (
                f"Project: {project_name}\n" if project_name else ""
            ) + f"File: {file_path} (lines {line_start}-{line_end}), type={chunk_type}"
            if is_private:
                header += " [HIGH-CONFIDENTIAL]"

            context_parts.append(f"{header}\n```{lang_block}\n{text}\n```")

        context = "\n---\n".join(context_parts)

        full_prompt = (
            f"System: {system_prompt}\n\nContext:\n---\n{context}\n---\n\nUser: {query}"
        )
        return full_prompt

    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM with the prompt.

        Uses llama-index's streaming or direct call interface.
        """
        try:
            # llama-index 0.10+ interface
            response = self.llm.complete(prompt)
            return self._sanitize_answer(str(response))
        except Exception as e:
            logger.error("LLM call failed: {}", e)
            raise

    def _sanitize_answer(self, answer: str) -> str:
        """Block likely verbatim code leakage in model output."""
        if "```" in answer:
            return SOURCE_EXPOSURE_BLOCK_MESSAGE

        if re.search(
            r"\b(module\s+\w+|endmodule|always\s*@|#include\s*[<\"]|"
            r"import\s+\w+|def\s+\w+\s*\(|class\s+\w+|\w+\s*\([^)]*\)\s*\{)",
            answer,
        ):
            return SOURCE_EXPOSURE_BLOCK_MESSAGE

        lines = [line for line in answer.splitlines() if line.strip()]
        if not lines:
            return answer

        code_like = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith((
                "#include",
                "import ",
                "from ",
                "def ",
                "class ",
            )) or re.search(
                r"[{};]|\b(return|if|else|for|while|switch|always|endmodule)\b",
                stripped,
            ):
                code_like += 1

        if code_like >= 2 and code_like / len(lines) >= 0.4:
            return SOURCE_EXPOSURE_BLOCK_MESSAGE

        return answer

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken (cl100k_base)."""
        if self._encoder is None:
            return len(text) // 4  # Rough estimate
        try:
            return len(self._encoder.encode(text))
        except Exception:
            return len(text) // 4

    def benchmark_token_reduction(
        self,
        query: str,
        full_context: str,
        top_k: int = 5,
    ) -> dict[str, int | float]:
        """
        Benchmark token reduction for a query.

        Compares tokens in the full context vs. RAG-retrieved chunks.

        Args:
            query: The query string.
            full_context: The full codebase context as a single string.
            top_k: Number of chunks to retrieve.

        Returns:
            Dict with 'full_tokens' and 'rag_tokens' counts.
        """
        result = self.run(query, top_k=top_k)

        full_tokens = self._count_tokens(full_context)
        rag_tokens = result.total_prompt_tokens

        return {
            "full_tokens": full_tokens,
            "rag_tokens": rag_tokens,
            "reduction_pct": (
                (1 - rag_tokens / full_tokens) * 100 if full_tokens > 0 else 0
            ),
        }


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def _request_json(
    method: str,
    url: str,
    payload: dict,
    headers: dict[str, str],
    timeout: int = 30,
) -> tuple[int, dict | None, str]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, data=body, method=method)
    for key, value in headers.items():
        req.add_header(key, value)
    req.add_header("Content-Type", "application/json")

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(text), text
            except json.JSONDecodeError:
                return resp.status, None, text
    except error.HTTPError as http_err:
        text = http_err.read().decode("utf-8", errors="replace")
        try:
            return http_err.code, json.loads(text), text
        except json.JSONDecodeError:
            return http_err.code, None, text
    except Exception as e:
        raise RuntimeError(f"HTTP request failed for {url}: {e}") from e


def _extract_openai_chat_content(data: dict) -> str:
    choices = data.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        return "".join(parts).strip()
    return str(content)


def _extract_anthropic_content(data: dict) -> str:
    content = data.get("content", [])
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        return "".join(parts).strip()
    return ""


def _detect_api_style(
    base_url: str,
    model: str,
    api_key: str | None,
    preferred_style: str = "auto",
    timeout: int = 10,
) -> str:
    if preferred_style in {"openai", "anthropic"}:
        return preferred_style

    openai_headers: dict[str, str] = {}
    if api_key:
        openai_headers["Authorization"] = f"Bearer {api_key}"

    openai_status, openai_data, openai_text = _request_json(
        method="POST",
        url=_join_url(base_url, "/v1/chat/completions"),
        payload={
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 8,
        },
        headers=openai_headers,
        timeout=timeout,
    )
    if openai_data and "choices" in openai_data:
        logger.info("Detected OpenAI-style API at {}", base_url)
        return "openai"
    if openai_status not in {404, 405}:
        if openai_text and any(k in openai_text.lower() for k in ["messages", "model"]):
            logger.info("Detected OpenAI-style API at {}", base_url)
            return "openai"

    anthropic_headers: dict[str, str] = {
        "anthropic-version": "2023-06-01",
    }
    if api_key:
        anthropic_headers["x-api-key"] = api_key

    anthropic_status, anthropic_data, anthropic_text = _request_json(
        method="POST",
        url=_join_url(base_url, "/v1/messages"),
        payload={
            "model": model,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "ping"}],
        },
        headers=anthropic_headers,
        timeout=timeout,
    )
    if anthropic_data and "content" in anthropic_data:
        logger.info("Detected Anthropic-style API at {}", base_url)
        return "anthropic"
    if anthropic_status not in {404, 405}:
        if anthropic_text and any(
            k in anthropic_text.lower() for k in ["anthropic", "messages", "max_tokens"]
        ):
            logger.info("Detected Anthropic-style API at {}", base_url)
            return "anthropic"

    raise RuntimeError(
        "Failed to detect API style on localhost. "
        "Expected OpenAI (/v1/chat/completions) or Anthropic (/v1/messages)."
    )


class _EmbeddingVector(list[float]):
    def tolist(self) -> list[float]:
        return list(self)


class LocalEmbeddingClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def encode(self, text: str) -> _EmbeddingVector:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["x-api-key"] = self.api_key

        vector = self._encode_single(text, headers)
        if vector is not None:
            return _EmbeddingVector(vector)

        parts = self._split_text_for_embedding(text)
        if len(parts) <= 1:
            raise RuntimeError("Embedding response format is invalid (empty embedding)")

        logger.warning(
            "Embedding input too large; falling back to chunked embedding ({} parts)",
            len(parts),
        )
        vectors: list[list[float]] = []
        for part in parts:
            part_vector = self._encode_single(part, headers)
            if part_vector is None:
                raise RuntimeError(
                    "Embedding response format is invalid (empty embedding)"
                )
            vectors.append(part_vector)

        return _EmbeddingVector(self._average_vectors(vectors))

    def _encode_single(self, text: str, headers: dict[str, str]) -> list[float] | None:
        status, data, raw_text = _request_json(
            method="POST",
            url=_join_url(self.base_url, "/v1/embeddings"),
            payload={"model": self.model, "input": text},
            headers=headers,
            timeout=self.timeout,
        )
        if not data or "data" not in data:
            if self._is_input_too_large_error(status, data, raw_text):
                return None
            raise RuntimeError(
                f"Embedding request failed with status={status}, body={raw_text[:300]}"
            )

        embedding = data.get("data", [{}])[0].get("embedding")
        if not isinstance(embedding, list):
            if self._is_input_too_large_error(status, data, raw_text):
                return None
            raise RuntimeError(
                "Embedding response format is invalid (missing embedding array)"
            )
        return [float(v) for v in embedding]

    def _is_input_too_large_error(
        self,
        status: int,
        data: dict | None,
        raw_text: str,
    ) -> bool:
        if status < 400:
            return False

        error_message = ""
        if isinstance(data, dict):
            error_data = data.get("error")
            if isinstance(error_data, dict):
                error_message = str(error_data.get("message", ""))
            elif error_data is not None:
                error_message = str(error_data)

        haystack = f"{error_message} {raw_text}".lower()
        return (
            "too large" in haystack
            or "too long" in haystack
            or "maximum context" in haystack
            or "batch size" in haystack
            or "token" in haystack
            and "limit" in haystack
        )

    def _split_text_for_embedding(
        self,
        text: str,
        max_chars: int = 900,
        overlap_chars: int = 120,
    ) -> list[str]:
        normalized = text.strip()
        if len(normalized) <= max_chars:
            return [normalized] if normalized else []

        chunks: list[str] = []
        start = 0
        step = max_chars - overlap_chars
        while start < len(normalized):
            end = min(len(normalized), start + max_chars)
            chunks.append(normalized[start:end])
            if end >= len(normalized):
                break
            start += step
        return chunks

    def _average_vectors(self, vectors: list[list[float]]) -> list[float]:
        if not vectors:
            raise RuntimeError("No vectors to average")

        dim = len(vectors[0])
        if dim == 0:
            raise RuntimeError("Cannot average empty embedding vectors")

        for vector in vectors:
            if len(vector) != dim:
                raise RuntimeError(
                    "Embedding dimensions are inconsistent across chunks"
                )

        sums = [0.0] * dim
        for vector in vectors:
            for i, value in enumerate(vector):
                sums[i] += float(value)

        return [value / len(vectors) for value in sums]


class LocalLLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_style: str = "auto",
        api_key: str | None = None,
        timeout: int = DEFAULT_LOCAL_LLM_TIMEOUT_SECONDS,
        max_tokens: int | None = DEFAULT_LOCAL_LLM_MAX_TOKENS,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.api_style = _detect_api_style(
            base_url=self.base_url,
            model=self.model,
            api_key=self.api_key,
            preferred_style=api_style,
        )

    def complete(self, prompt: str) -> str:
        if self.api_style == "openai":
            headers: dict[str, str] = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            }
            if self.max_tokens is not None:
                payload["max_tokens"] = self.max_tokens
            logger.debug(
                "LLM POST payload | prompt_chars={} | max_tokens={}",
                len(prompt),
                self.max_tokens if self.max_tokens is not None else "unlimited",
            )
            payload_json = json.dumps(payload, ensure_ascii=False)
            logger.debug(
                "LLM POST payload | body_sha256={} | body_chars={}",
                sha256(payload_json.encode("utf-8")).hexdigest()[:16],
                len(payload_json),
            )
            if True:
                logger.debug(
                    "LLM POST payload | body_preview={}...",
                    payload_json[:9999],
                )
            status, data, raw_text = _request_json(
                method="POST",
                url=_join_url(self.base_url, "/v1/chat/completions"),
                payload=payload,
                headers=headers,
                timeout=self.timeout,
            )
            logger.debug(
                "LLM response | status={} | finish_reason={} | response_preview={}...",
                status,
                (
                    data.get("choices", [{}])[0].get("finish_reason", "?")
                    if isinstance(data, dict) and data.get("choices")
                    else "?"
                ),
                raw_text[:400],
            )
            if not data:
                raise RuntimeError(
                    "OpenAI-style completion failed with "
                    f"status={status}, body={raw_text[:300]}"
                )
            text = _extract_openai_chat_content(data)
            if not text:
                finish_reason = ""
                choices = data.get("choices", []) if isinstance(data, dict) else []
                if choices and isinstance(choices[0], dict):
                    finish_reason = str(choices[0].get("finish_reason", ""))
                raise RuntimeError(
                    "OpenAI-style completion returned empty content"
                    f" (finish_reason={finish_reason or 'unknown'})"
                )
            return text

        headers = {
            "anthropic-version": "2023-06-01",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key

        status, data, raw_text = _request_json(
            method="POST",
            url=_join_url(self.base_url, "/v1/messages"),
            payload={
                "model": self.model,
                "max_tokens": 1024,
                "temperature": 0.2,
                "messages": [{"role": "user", "content": prompt}],
            },
            headers=headers,
            timeout=self.timeout,
        )
        if not data:
            raise RuntimeError(
                "Anthropic-style completion failed with "
                f"status={status}, body={raw_text[:300]}"
            )
        text = _extract_anthropic_content(data)
        if not text:
            raise RuntimeError("Anthropic-style completion returned empty content")
        return text


def build_embedding_function(
    embedding_model_name: str = DEFAULT_LOCAL_EMBEDDING_MODEL,
    base_url: str | None = None,
):
    """
    Build an embedding function from a localhost embedding API.

    Returns a function: str -> list[float]
    """
    api_base_url = (
        base_url
        or os.environ.get("EMBEDDING_BASE_URL")
        or os.environ.get("LOCAL_LLM_BASE_URL", DEFAULT_LOCAL_BASE_URL)
    )
    api_key = os.environ.get("LOCAL_API_KEY")
    model = LocalEmbeddingClient(
        base_url=api_base_url,
        model=embedding_model_name,
        api_key=api_key,
    )

    def embed(text: str) -> list[float]:
        return model.encode(text).tolist()

    return embed, model


def build_llm(
    provider: str = "auto",
    model: str = DEFAULT_LOCAL_LLM_MODEL,
    base_url: str | None = None,
    timeout: int = DEFAULT_LOCAL_LLM_TIMEOUT_SECONDS,
    max_tokens: int | None = DEFAULT_LOCAL_LLM_MAX_TOKENS,
):
    """
    Build an LLM client from localhost API.

    Args:
        provider: API style preference: "auto", "openai", or "anthropic"
        model: Model name to use.
        timeout: Request timeout in seconds.
        max_tokens: Max tokens for generation. None = no limit.

    Returns:
        LocalLLMClient instance.
    """
    if provider not in {"auto", "openai", "anthropic"}:
        raise ValueError(f"Unknown provider/style: {provider}")

    api_base_url = base_url or os.environ.get(
        "LOCAL_LLM_BASE_URL", DEFAULT_LOCAL_BASE_URL
    )
    api_key = os.environ.get("LOCAL_API_KEY")

    return LocalLLMClient(
        base_url=api_base_url,
        model=model,
        api_style=provider,
        api_key=api_key,
        timeout=timeout,
        max_tokens=max_tokens,
    )
