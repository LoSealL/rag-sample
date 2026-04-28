"""Graph backend abstraction layer.

Defines the GraphBackend protocol that all storage implementations must
satisfy.  This allows NetworkX (default), Neo4j, Kùzu, or any other
graph database to be plugged in without changing upstream code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .models import Entity, EntityId, Relation


class GraphBackend(ABC):
    """Abstract interface for graph storage backends.

    Implementations must support directed graphs with node/edge attributes.
    """

    @abstractmethod
    def add_entity(self, entity: Entity) -> None:
        """Add or update a node representing *entity*."""

    @abstractmethod
    def remove_entity(self, entity_id: EntityId) -> None:
        """Remove a node and all its incident edges."""

    @abstractmethod
    def add_relation(self, relation: Relation) -> None:
        """Add a directed edge from *relation.source* to *relation.target*."""

    @abstractmethod
    def remove_relation(self, source: EntityId, target: EntityId, relation_type: str | None = None) -> None:
        """Remove edge(s) between *source* and *target*.

        If *relation_type* is given, only remove edges of that type.
        """

    @abstractmethod
    def get_entity(self, entity_id: EntityId) -> Entity | None:
        """Fetch a single entity by id."""

    @abstractmethod
    def get_neighbors(
        self,
        entity_id: EntityId,
        relation_types: list[str] | None = None,
        direction: str = "both",
    ) -> list[tuple[Entity, Relation]]:
        """Return neighbours of *entity_id* together with the connecting relation."""

    @abstractmethod
    def get_entities_in_file(self, file_path: str, project_id: str | None = None) -> list[Entity]:
        """Return all entities defined in *file_path*."""

    @abstractmethod
    def get_paths(
        self,
        source: EntityId,
        target: EntityId,
        max_depth: int = 5,
        relation_types: list[str] | None = None,
    ) -> list[list[Entity]]:
        """Find all simple paths from *source* to *target* up to *max_depth* hops."""

    @abstractmethod
    def get_impact_scope(
        self,
        entity_id: EntityId,
        relation_types: list[str] | None = None,
        max_depth: int = 3,
    ) -> list[Entity]:
        """Return all entities reachable from *entity_id* within *max_depth* hops."""

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return basic graph statistics."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all nodes and edges."""

    @abstractmethod
    def save(self) -> None:
        """Persist the graph to disk (if applicable)."""

    @abstractmethod
    def nodes(self) -> list[tuple[str, dict[str, Any]]]:
        """Iterate over all nodes yielding (node_id, attributes)."""

    @abstractmethod
    def edges(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Iterate over all edges yielding (source_id, target_id, attributes)."""
