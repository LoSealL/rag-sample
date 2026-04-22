"""
ChromaDB index store for RAG chunks.

Manages two collections: code_index and doc_index.
Uses ChromaDB in embedded (in-process) mode — no server required.

Usage:
    store = IndexStore(".rag_index")
    store.add_chunks("code_index", chunks, embed_fn=my_embed)
    results = store.query("code_index", "auth function", top_k=5)
"""

import os
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from sec_rag.chunkers.base_chunker import Chunk

DEFAULT_PERSIST_DIR = ".rag_index"


class IndexStore:
    """
    ChromaDB-backed index store for code and document chunks.

    Uses ChromaDB in embedded (in-process) mode with persistence.
    No separate server process required.

    Usage:
        store = IndexStore(".rag_index")
        store.add_chunks("code_index", chunks, embed_fn=my_embed)
        results = store.query("code_index", "how does auth work?", top_k=5)
    """

    def __init__(self, persist_dir: str = DEFAULT_PERSIST_DIR):
        """
        Initialize the index store.

        Args:
            persist_dir: Directory for ChromaDB persistence.
                         Created if it doesn't exist.
        """
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        # ChromaDB in embedded mode — no client/server
        self._client = chromadb.PersistentClient(
            path=os.path.abspath(persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,  # Disable ChromaDB telemetry
                allow_reset=True,
            ),
        )

        # Track collections
        self._collections: dict[str, Any] = {}

    def get_or_create_collection(self, name: str) -> Any:
        """
        Get or create a ChromaDB collection.

        Args:
            name: Collection name (e.g., "code_index", "doc_index")

        Returns:
            ChromaDB collection object.
        """
        if name not in self._collections:
            collection = self._client.get_or_create_collection(
                name=name,
                metadata={"description": f"RAG index: {name}"},
            )
            self._collections[name] = collection
            logger.debug("Collection {} created/accessed", name)
        return self._collections[name]

    def add_chunks(
        self,
        collection_name: str,
        chunks: list[Chunk],
        embed_fn=None,  # callable: str -> list[float]
    ) -> None:
        """
        Add chunks to a collection.

        Args:
            collection_name: Name of the collection.
            chunks: List of Chunk objects to add.

        Raises:
            ValueError: If chunks is empty.
        """
        if not chunks:
            return
        if embed_fn is None:
            raise ValueError(
                "embed_fn is required for add_chunks to avoid Chroma default embeddings"
            )

        collection = self.get_or_create_collection(collection_name)

        ids = [c.metadata.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "file_path": c.metadata.file_path,
                "language": c.metadata.language,
                "chunk_type": c.metadata.chunk_type,
                "name": c.metadata.name,
                "chunk_id": c.metadata.chunk_id,
                "line_start": c.metadata.line_start,
                "line_end": c.metadata.line_end,
                "project_id": c.metadata.project_id,
                "project_name": c.metadata.project_name,
                "project_root": c.metadata.project_root,
                "document_kind": c.metadata.document_kind,
                "sanitized": c.metadata.sanitized,
                "source_language_family": c.metadata.source_language_family,
                "declaration": c.metadata.declaration,
                "semantic_summary": c.metadata.semantic_summary,
            }
            for c in chunks
        ]

        # Pre-compute embeddings so ChromaDB never falls back to its default
        # ONNX model (DefaultEmbeddingFunction, e.g. all-MiniLM-L6-v2).
        embeddings = [embed_fn(doc) for doc in documents]

        # ChromaDB will overwrite documents with the same ID.
        # This makes re-indexing idempotent.
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info("Added {} chunks to {}", len(chunks), collection_name)

    def query(
        self,
        collection_name: str,
        query_text: str,
        embedding_function,  # callable: str -> list[float]
        top_k: int = 5,
        filter_language: str | None = None,
        filter_chunk_type: str | None = None,
        filter_project_id: str | None = None,
        filter_document_kind: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query a collection for the most relevant chunks.

        Args:
            collection_name: Name of the collection to query.
            query_text: Natural language query.
            embedding_function: Function that embeds text to a vector.
            top_k: Number of results to return.
            filter_language: Optional language filter (e.g., "python", "c").
            filter_chunk_type: Optional chunk type filter (e.g., "function").
            filter_project_id: Optional project id filter.
            filter_document_kind: Optional document kind filter.

        Returns:
            List of result dicts with 'chunk', 'score', 'metadata' keys.
        """
        collection = self.get_or_create_collection(collection_name)

        # Build where filter if filters are specified
        where_filter: dict[str, Any] | None = None
        if filter_language or filter_chunk_type:
            where_filter = {}
            if filter_language:
                where_filter["language"] = filter_language
            if filter_chunk_type:
                where_filter["chunk_type"] = filter_chunk_type
        if filter_project_id or filter_document_kind:
            where_filter = where_filter or {}
            if filter_project_id:
                where_filter["project_id"] = filter_project_id
            if filter_document_kind:
                where_filter["document_kind"] = filter_document_kind

        # Embed the query
        query_embedding = embedding_function(query_text)

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted: list[dict[str, Any]] = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                formatted.append(
                    {
                        "chunk_id": chunk_id,
                        "text": results["documents"][0][i]
                        if results["documents"]
                        else "",
                        "metadata": results["metadatas"][0][i]
                        if results["metadatas"]
                        else {},
                        "score": results["distances"][0][i]
                        if results["distances"]
                        else 0.0,
                    }
                )

        return formatted

    def query_multi_collection(
        self,
        query_text: str,
        embedding_function,
        collection_names: list[str],
        top_k: int = 5,
        filter_language: str | None = None,
        filter_chunk_type: str | None = None,
        filter_project_id: str | None = None,
        filter_document_kind: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Query multiple collections and merge results.

        Args:
            query_text: Natural language query.
            embedding_function: Function that embeds text to a vector.
            collection_names: List of collection names to query.
            top_k: Number of results per collection.
            filter_language: Optional language filter.
            filter_chunk_type: Optional chunk type filter.

        Returns:
            Dict mapping collection name -> list of results.
        """
        all_results = {}
        for coll_name in collection_names:
            try:
                results = self.query(
                    coll_name,
                    query_text,
                    embedding_function,
                    top_k=top_k,
                    filter_language=filter_language,
                    filter_chunk_type=filter_chunk_type,
                    filter_project_id=filter_project_id,
                    filter_document_kind=filter_document_kind,
                )
                all_results[coll_name] = results
            except Exception as e:
                logger.warning("Failed to query collection {}: {}", coll_name, e)
                all_results[coll_name] = []
        return all_results

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about all collections.

        Returns:
            Dict with collection names and their document counts.
        """
        stats: dict[str, Any] = {}
        for collection in self._client.list_collections():
            name = collection.name if hasattr(collection, "name") else str(collection)
            try:
                coll = self._client.get_collection(name)
                stats[name] = {"count": coll.count()}
            except Exception as exc:
                logger.warning("Failed to get stats for collection {}: {}", name, exc)
        return stats

    def reset(self) -> None:
        """Reset all collections. Use with caution — deletes all indexed data."""
        self._client.reset()
        self._collections = {}
        logger.warning("Index store reset — all collections deleted")

    def delete_collection(self, name: str) -> None:
        """Delete a specific collection."""
        if name in self._collections:
            del self._collections[name]
        self._client.delete_collection(name)
        logger.info("Collection {} deleted", name)
