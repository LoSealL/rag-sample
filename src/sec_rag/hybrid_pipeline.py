"""Hybrid RAG query pipeline — combines vector retrieval with graph expansion.

This module extends the standard QueryPipeline by adding a graph-based
expansion step: after retrieving the top-k chunks via vector similarity,
we follow entity relations in the knowledge graph to bring in structurally
related code (callers, callees, parent classes, referenced types) that may
not be semantically similar but are crucial for understanding the full
context.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from sec_rag.graph import GraphQuery, GraphStore
from sec_rag.graph.models import EntityId

from .query_pipeline import QueryPipeline, QueryResult, RetrievalResult


class HybridQueryPipeline(QueryPipeline):
    """Vector + Graph hybrid retrieval pipeline.

    In addition to the standard vector retrieval, this pipeline:

    1. Retrieves top-k chunks via embedding similarity.
    2. Maps each chunk to its corresponding entity in the knowledge graph.
    3. Expands along graph edges (CALLS, REFERENCES, INHERITS_FROM, CONTAINS)
       up to *graph_depth* hops.
    4. Deduplicates and re-ranks the union of vector and graph results.
    5. Sends the combined context to the LLM.

    Usage::

        pipeline = HybridQueryPipeline(
            index_store, embedding_model, llm,
            graph_store=GraphStore(".rag_index/knowledge_graph.json"),
        )
        result = pipeline.run("how does auth work?", top_k=5, graph_depth=2)
    """

    def __init__(
        self,
        index_store,
        embedding_model,
        llm,
        graph_store: GraphStore | None = None,
        graph_depth: int = 2,
        graph_relation_types: list[str] | None = None,
        graph_max_expanded: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize the hybrid pipeline.

        Args:
            index_store, embedding_model, llm: Same as QueryPipeline.
            graph_store: GraphStore instance.  If None, graph expansion is skipped.
            graph_depth: Max hops to traverse from a retrieved entity.
            graph_relation_types: Which edge types to follow.  None means all.
            graph_max_expanded: Hard cap on number of graph-expanded chunks.
            **kwargs: Passed through to QueryPipeline.
        """
        super().__init__(index_store, embedding_model, llm, **kwargs)
        self._graph_store = graph_store
        self._graph_query = GraphQuery(graph_store) if graph_store else None
        self._graph_depth = graph_depth
        self._graph_relation_types = graph_relation_types or [
            "CALLS", "REFERENCES", "INHERITS_FROM", "CONTAINS", "IMPLEMENTS"
        ]
        self._graph_max_expanded = graph_max_expanded

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        top_k: int = 5,
        filter_language: str | None = None,
        filter_chunk_type: str | None = None,
        project_top_k: int = 3,
        graph_depth: int | None = None,
    ) -> QueryResult:
        """Run hybrid retrieval + LLM generation."""
        retrieval = self.hybrid_retrieve(
            query=query,
            top_k=top_k,
            filter_language=filter_language,
            filter_chunk_type=filter_chunk_type,
            project_top_k=project_top_k,
            graph_depth=graph_depth,
        )

        prompt = self._build_prompt(retrieval.chunks, query)
        answer = self._call_llm(prompt)

        return QueryResult(
            answer=answer,
            chunks=retrieval.chunks,
            total_prompt_tokens=retrieval.total_prompt_tokens,
            total_chunk_tokens=retrieval.total_chunk_tokens,
        )

    def hybrid_retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_language: str | None = None,
        filter_chunk_type: str | None = None,
        project_top_k: int = 3,
        graph_depth: int | None = None,
    ) -> RetrievalResult:
        """Retrieve chunks using vector search + graph expansion.

        Returns a RetrievalResult whose ``chunks`` list contains the
        deduplicated union of vector-retrieved and graph-expanded chunks.
        """
        # Phase 1: Standard vector retrieval
        vector_result = self.retrieve(
            query=query,
            top_k=top_k,
            filter_language=filter_language,
            filter_chunk_type=filter_chunk_type,
            project_top_k=project_top_k,
        )

        if self._graph_store is None or self._graph_query is None:
            return vector_result

        # Phase 2: Graph expansion
        depth = graph_depth if graph_depth is not None else self._graph_depth
        graph_chunks = self._expand_via_graph(vector_result.chunks, depth)

        # Phase 3: Deduplicate and merge
        all_chunks = self._merge_chunks(vector_result.chunks, graph_chunks)

        # Recalculate token counts
        total_chunk_tokens = sum(
            len(self._encode(chunk.get("document", "")))
            for chunk in all_chunks
        )
        total_prompt_tokens = len(self._encode(self._build_prompt(all_chunks, query)))

        return RetrievalResult(
            chunks=all_chunks,
            total_prompt_tokens=total_prompt_tokens,
            total_chunk_tokens=total_chunk_tokens,
            candidate_project_ids=vector_result.candidate_project_ids,
        )

    # ------------------------------------------------------------------
    # Graph expansion internals
    # ------------------------------------------------------------------

    def _expand_via_graph(
        self,
        chunks: list[dict],
        depth: int,
    ) -> list[dict]:
        """Follow graph edges from entities mentioned in *chunks*."""
        assert self._graph_store is not None
        assert self._graph_query is not None

        expanded: list[dict] = []
        seen_chunk_ids: set[str] = set()

        for chunk in chunks:
            chunk_id = chunk.get("id", "")
            seen_chunk_ids.add(chunk_id)

            # Try to map chunk -> entity via chunk_id or name matching
            entity = self._chunk_to_entity(chunk)
            if entity is None:
                continue

            # BFS along relevant relation types
            visited: set[str] = {str(entity.id)}
            queue: list[tuple[str, int]] = [(str(entity.id), 0)]

            while queue and len(expanded) < self._graph_max_expanded:
                current_nid, current_depth = queue.pop(0)
                if current_depth >= depth:
                    continue

                neighbors = self._graph_store.get_neighbors(
                    EntityId(*current_nid.split(":", 3)),
                    relation_types=self._graph_relation_types,
                    direction="both",
                )

                for neighbor_entity, _rel in neighbors:
                    nid = str(neighbor_entity.id)
                    if nid in visited:
                        continue
                    visited.add(nid)

                    # Map back to chunk
                    neighbor_chunk = self._entity_to_chunk(neighbor_entity)
                    if neighbor_chunk and neighbor_chunk.get("id") not in seen_chunk_ids:
                        seen_chunk_ids.add(neighbor_chunk.get("id", ""))
                        expanded.append(neighbor_chunk)
                        queue.append((nid, current_depth + 1))

        logger.info(
            "Graph expansion: {} vector chunks -> {} expanded chunks",
            len(chunks),
            len(expanded),
        )
        return expanded

    def _chunk_to_entity(self, chunk: dict) -> Any | None:
        """Map a retrieved chunk to its graph entity, if known."""
        metadata = chunk.get("metadata", {})
        chunk_id = chunk.get("id", "")
        name = metadata.get("name", "")
        file_path = metadata.get("file_path", "")
        chunk_type = metadata.get("chunk_type", "")

        # Try exact match by chunk_id stored in entity metadata
        for nid, data in self._graph_store._g.nodes(data=True):
            if data.get("chunk_id") == chunk_id:
                return self._graph_store.get_entity(EntityId(*nid.split(":", 3)))

        # Fallback: match by name + file_path
        for nid, data in self._graph_store._g.nodes(data=True):
            if data.get("name") == name and data.get("file_path") == file_path:
                return self._graph_store.get_entity(EntityId(*nid.split(":", 3)))

        return None

    def _entity_to_chunk(self, entity) -> dict | None:
        """Map a graph entity back to its chunk document in the index store."""
        if not entity.chunk_id:
            # Try to find by metadata filters
            try:
                results = self.index_store.query(
                    "rag_index",
                    query_text="",
                    top_k=1,
                    where={
                        "$and": [
                            {"name": {"$eq": entity.name}},
                            {"file_path": {"$eq": entity.file_path}},
                        ]
                    },
                )
                if results:
                    return results[0]
            except Exception:
                pass
            return None

        # Direct lookup by chunk_id
        try:
            results = self.index_store.query(
                "rag_index",
                query_text="",
                top_k=1,
                where={"chunk_id": {"$eq": entity.chunk_id}},
            )
            if results:
                return results[0]
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    # Merging / deduplication
    # ------------------------------------------------------------------

    def _merge_chunks(
        self,
        vector_chunks: list[dict],
        graph_chunks: list[dict],
    ) -> list[dict]:
        """Deduplicate and interleave vector + graph chunks.

        Strategy: keep all vector chunks in original order, then append
        graph chunks that are not already present.  This preserves the
        semantic ranking from vector search while adding structural context.
        """
        seen: set[str] = set()
        merged: list[dict] = []

        for chunk in vector_chunks:
            cid = chunk.get("id", "")
            if cid not in seen:
                seen.add(cid)
                merged.append(chunk)

        for chunk in graph_chunks:
            cid = chunk.get("id", "")
            if cid not in seen:
                seen.add(cid)
                merged.append(chunk)

        return merged

    def _encode(self, text: str) -> list[int]:
        """Tokenise text for counting."""
        if self._encoder is None:
            return []
        try:
            return self._encoder.encode(text)
        except Exception:
            return []

    def _build_prompt(self, chunks: list[dict], query: str) -> str:
        """Build the RAG prompt (re-used from QueryPipeline)."""
        # We import the protected method from parent via super() call
        # but QueryPipeline doesn't expose it publicly.  We'll inline
        # a minimal version here.
        context = "\n\n---\n\n".join(
            chunk.get("document", "") for chunk in chunks
        )
        return (
            "You are a helpful coding assistant. Use the following code context "
            "to answer the user's question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

    def _call_llm(self, prompt: str) -> str:
        """Delegate to the LLM (reuse parent implementation if possible)."""
        # QueryPipeline._call_llm is private; we approximate here.
        try:
            return self.llm.complete(prompt)
        except Exception as exc:
            logger.error("LLM generation failed: {}", exc)
            return f"[Error: LLM failed - {exc}]"
