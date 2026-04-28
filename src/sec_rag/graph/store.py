"""Graph storage facade.

Provides a unified interface over any GraphBackend implementation.
By default uses NetworkXBackend; for very large codebases this can be
swapped for Neo4jBackend or KuzuBackend without changing upstream code.
"""

from __future__ import annotations

from typing import Any

from .backend import GraphBackend
from .models import Entity, EntityId, Relation
from .networkx_backend import NetworkXBackend


class GraphStore:
    """Unified graph storage API.

    Delegates all operations to a configurable GraphBackend.  The default
    backend is NetworkX (pure-Python, zero external services).
    """

    def __init__(
        self,
        persist_path: str = ".rag_index/knowledge_graph.json",
        backend: GraphBackend | None = None,
    ) -> None:
        self._backend = backend or NetworkXBackend(persist_path=persist_path)

    # ------------------------------------------------------------------
    # Delegated methods
    # ------------------------------------------------------------------

    def add_entity(self, entity: Entity) -> None:
        self._backend.add_entity(entity)

    def remove_entity(self, entity_id: EntityId) -> None:
        self._backend.remove_entity(entity_id)

    def add_relation(self, relation: Relation) -> None:
        self._backend.add_relation(relation)

    def remove_relation(self, source: EntityId, target: EntityId, relation_type: str | None = None) -> None:
        self._backend.remove_relation(source, target, relation_type)

    def get_entity(self, entity_id: EntityId) -> Entity | None:
        return self._backend.get_entity(entity_id)

    def get_neighbors(
        self,
        entity_id: EntityId,
        relation_types: list[str] | None = None,
        direction: str = "both",
    ) -> list[tuple[Entity, Relation]]:
        return self._backend.get_neighbors(entity_id, relation_types, direction)

    def get_entities_in_file(self, file_path: str, project_id: str | None = None) -> list[Entity]:
        return self._backend.get_entities_in_file(file_path, project_id)

    def get_paths(
        self,
        source: EntityId,
        target: EntityId,
        max_depth: int = 5,
        relation_types: list[str] | None = None,
    ) -> list[list[Entity]]:
        return self._backend.get_paths(source, target, max_depth, relation_types)

    def get_impact_scope(
        self,
        entity_id: EntityId,
        relation_types: list[str] | None = None,
        max_depth: int = 3,
    ) -> list[Entity]:
        return self._backend.get_impact_scope(entity_id, relation_types, max_depth)

    def stats(self) -> dict[str, Any]:
        return self._backend.stats()

    def clear(self) -> None:
        self._backend.clear()

    def save(self) -> None:
        self._backend.save()

    # ------------------------------------------------------------------
    # Raw access (exporter, visualisation)
    # ------------------------------------------------------------------

    @property
    def _g(self):
        """Expose the underlying graph object for advanced use.

        This is a backdoor used by exporters and debug tools.  Prefer the
        public API for normal operations.
        """
        if hasattr(self._backend, "_g"):
            return self._backend._g
        raise AttributeError("Current backend does not expose an internal graph object")

    def nodes(self) -> list[tuple[str, dict[str, Any]]]:
        return self._backend.nodes()

    def edges(self) -> list[tuple[str, str, dict[str, Any]]]:
        return self._backend.edges()
