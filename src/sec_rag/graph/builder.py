"""Graph builder — orchestrates extraction, storage and cross-file resolution.

Usage::

    builder = GraphBuilder(CppAstExtractor(), GraphStore())
    builder.add_file(source, "src/main.cpp", "my-project")
    builder.resolve_cross_file_relations()   # Phase 2
    builder.save()

    # Phase 3: incremental update
    builder.update_file(new_source, "src/main.cpp", "my-project")
    builder.save()
"""

from __future__ import annotations

from .ast_extractor import CppAstExtractor, SymbolTable
from .file_hash_index import FileHashIndex
from .llm_extractor import LLMGraphExtractor
from .models import Entity, EntityId, Relation
from .store import GraphStore


class GraphBuilder:
    """Incrementally build a knowledge graph from source files.

    Supports full rebuilds as well as incremental per-file updates.
    """

    def __init__(
        self,
        extractor: CppAstExtractor | None = None,
        llm_extractor: LLMGraphExtractor | None = None,
        store: GraphStore | None = None,
        hash_index: FileHashIndex | None = None,
    ) -> None:
        self._extractor = extractor or CppAstExtractor()
        self._llm_extractor = llm_extractor
        self._store = store or GraphStore()
        self._hash_index = hash_index or FileHashIndex()
        self._pending_relations: list[Relation] = []
        # Track which entities belong to which file for incremental updates
        self._file_entities: dict[str, list[EntityId]] = {}

    # ------------------------------------------------------------------
    # Add / Update
    # ------------------------------------------------------------------

    def add_file(self, source: str, file_path: str, project_id: str) -> None:
        """Parse *source* and add all extracted entities and relations."""
        entities, relations = self._extractor.extract(source, file_path, project_id)
        self._ingest(file_path, entities, relations)

    def update_file(self, source: str, file_path: str, project_id: str) -> None:
        """Incrementally update the graph for *file_path*.

        Removes all entities and relations previously extracted from this
        file, then re-parses *source* and inserts the new graph fragment.
        Cross-file relations that pointed to now-deleted entities are
        automatically dropped by the backend.
        """
        self._remove_file(file_path)
        self.add_file(source, file_path, project_id)

    def add_file_with_llm_fallback(
        self,
        source: str,
        file_path: str,
        project_id: str,
    ) -> None:
        """Try AST extraction first; on failure fall back to LLM extractor."""
        entities, relations = self._extractor.extract(source, file_path, project_id)
        if not entities and self._llm_extractor is not None:
            entities, relations = self._llm_extractor.extract(source, file_path, project_id)
        self._ingest(file_path, entities, relations, track_llm_pending=True)

    def _ingest(
        self,
        file_path: str,
        entities: list[Entity],
        relations: list[Relation],
        track_llm_pending: bool = False,
    ) -> None:
        """Internal helper: add entities/relations and book-keep file mapping."""
        # Record file -> entity mapping
        ids = self._file_entities.setdefault(file_path, [])
        for entity in entities:
            self._store.add_entity(entity)
            ids.append(entity.id)

        for relation in relations:
            if track_llm_pending and not relation.target.file_path:
                self._pending_relations.append(relation)
            else:
                self._store.add_relation(relation)

    def _remove_file(self, file_path: str) -> None:
        """Remove all entities (and their incident edges) for *file_path*."""
        ids = self._file_entities.pop(file_path, [])
        for eid in ids:
            self._store.remove_entity(eid)
        # Also clean up any pending relations whose source or target
        # belonged to this file.
        self._pending_relations = [
            r for r in self._pending_relations
            if not (str(r.source).startswith(f"{eid.project_id}:{file_path}")
                    or str(r.target).startswith(f"{eid.project_id}:{file_path}"))
        ]

    # ------------------------------------------------------------------
    # Cross-file resolution
    # ------------------------------------------------------------------

    def resolve_cross_file_relations(self) -> None:
        """Resolve pending relations whose targets were initially unknown.

        Uses the symbol table built during extraction to map bare symbol
        names to globally unique EntityIds.
        """
        resolved: list[Relation] = []
        still_pending: list[Relation] = []

        for relation in self._pending_relations:
            candidates = self._extractor._symbol_table.lookup(relation.target.name)
            if len(candidates) == 1:
                new_relation = Relation(
                    source=relation.source,
                    target=candidates[0],
                    relation_type=relation.relation_type,
                    file_path=relation.file_path,
                    line=relation.line,
                    metadata=relation.metadata,
                )
                resolved.append(new_relation)
            else:
                still_pending.append(relation)

        for relation in resolved:
            self._store.add_relation(relation)

        self._pending_relations = still_pending

    # ------------------------------------------------------------------
    # Chunk convenience
    # ------------------------------------------------------------------

    def add_chunk(self, chunk, project_id: str) -> None:
        """Convenience wrapper that extracts from a Chunk object."""
        self.add_file(chunk.text, chunk.metadata.file_path, project_id)

    def update_chunk(self, chunk, project_id: str) -> None:
        """Convenience wrapper for incremental chunk update."""
        self.update_file(chunk.text, chunk.metadata.file_path, project_id)

    # ------------------------------------------------------------------
    # Persistence / stats
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Project-level incremental build
    # ------------------------------------------------------------------

    def build_project(
        self,
        files: list[tuple[str, str]],
        project_id: str,
        progress_callback=None,
    ) -> dict[str, int]:
        """Incrementally build/update the graph for an entire project.

        Args:
            files: List of (file_path, source_text) tuples.
            project_id: Project identifier.
            progress_callback: Optional callable(file_path, action) for UI
                feedback.  *action* is one of "add", "update", "skip", "remove".

        Returns:
            Stats dict with keys ``added``, ``updated``, "skipped", "removed".
        """
        stats = {"added": 0, "updated": 0, "skipped": 0, "removed": 0}
        current_files = [fp for fp, _src in files]

        # 1. Remove stale files (deleted since last index)
        for stale_path in self._hash_index.get_stale_files(current_files):
            self._remove_file(stale_path)
            self._hash_index.remove(stale_path)
            stats["removed"] += 1
            if progress_callback:
                progress_callback(stale_path, "remove")

        # 2. Process current files
        for file_path, source in files:
            if self._hash_index.is_changed(file_path, source):
                # File is new or changed
                if file_path in self._file_entities:
                    # Existing file: incremental update
                    self.update_file(source, file_path, project_id)
                    stats["updated"] += 1
                    action = "update"
                else:
                    # New file
                    self.add_file(source, file_path, project_id)
                    stats["added"] += 1
                    action = "add"
                self._hash_index.update(file_path, source)
            else:
                stats["skipped"] += 1
                action = "skip"

            if progress_callback:
                progress_callback(file_path, action)

        return stats

    # ------------------------------------------------------------------
    # Persistence / stats
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the graph and hash index to disk."""
        self._store.save()
        self._hash_index.save()

    def clear(self) -> None:
        """Remove all nodes and edges."""
        self._store.clear()
        self._pending_relations.clear()
        self._file_entities.clear()
        self._hash_index._hashes.clear()

    def stats(self) -> dict:
        """Return graph statistics."""
        return self._store.stats()
