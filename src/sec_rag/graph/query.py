"""High-level graph query API.

Provides convenient methods for common code-analysis questions such as
"who calls this function?", "what is the class hierarchy?", and
"what is the impact scope of modifying X?".
"""

from __future__ import annotations

from .models import Entity, EntityId
from .store import GraphStore


class GraphQuery:
    """Query facade over a GraphStore."""

    def __init__(self, store: GraphStore) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Call-graph queries
    # ------------------------------------------------------------------

    def find_callers(
        self,
        function_name: str,
        project_id: str | None = None,
        max_depth: int = 3,
    ) -> list[Entity]:
        """Return all functions that call *function_name* (upstream)."""
        entity = self._find_entity(function_name, "function", project_id)
        if not entity:
            return []
        # Use BFS along incoming edges to find transitive callers
        visited: set[str] = set()
        queue: list[tuple[Entity, int]] = [(entity, 0)]
        result: list[Entity] = []
        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for neighbor, _rel in self._store.get_neighbors(
                current.id,
                relation_types=["CALLS"],
                direction="in",
            ):
                nid = str(neighbor.id)
                if nid in visited:
                    continue
                visited.add(nid)
                result.append(neighbor)
                queue.append((neighbor, depth + 1))
        return result

    def find_callees(
        self,
        function_name: str,
        project_id: str | None = None,
        max_depth: int = 3,
    ) -> list[Entity]:
        """Return all functions called by *function_name* (downstream)."""
        entity = self._find_entity(function_name, "function", project_id)
        if not entity:
            return []
        # Use BFS along outgoing edges to find transitive callees
        visited: set[str] = set()
        queue: list[tuple[Entity, int]] = [(entity, 0)]
        result: list[Entity] = []
        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for neighbor, _rel in self._store.get_neighbors(
                current.id,
                relation_types=["CALLS"],
                direction="out",
            ):
                nid = str(neighbor.id)
                if nid in visited:
                    continue
                visited.add(nid)
                result.append(neighbor)
                queue.append((neighbor, depth + 1))
        return result

    # ------------------------------------------------------------------
    # Class hierarchy queries
    # ------------------------------------------------------------------

    def class_hierarchy(
        self,
        class_name: str,
        project_id: str | None = None,
    ) -> dict[str, list[Entity]]:
        """Return the inheritance hierarchy for *class_name*.

        Returns a dict with keys ``parents``, ``children``, ``siblings``.
        """
        entity = self._find_entity(class_name, ("class", "struct", "class_template", "struct_template"), project_id)
        if not entity:
            return {"parents": [], "children": [], "siblings": []}

        parents = self._store.get_neighbors(
            entity.id,
            relation_types=["INHERITS_FROM"],
            direction="out",
        )
        children = self._store.get_neighbors(
            entity.id,
            relation_types=["INHERITS_FROM"],
            direction="in",
        )

        # Siblings = classes that share at least one parent
        parent_ids = {str(p[0].id) for p in parents}
        siblings: list[Entity] = []
        for parent_id in parent_ids:
            for child, _rel in self._store.get_neighbors(
                EntityId(*parent_id.split(":", 3)),
                relation_types=["INHERITS_FROM"],
                direction="in",
            ):
                if child.id != entity.id:
                    siblings.append(child)

        return {
            "parents": [p[0] for p in parents],
            "children": [c[0] for c in children],
            "siblings": siblings,
        }

    # ------------------------------------------------------------------
    # File dependency queries
    # ------------------------------------------------------------------

    def file_dependencies(
        self,
        file_path: str,
        project_id: str | None = None,
    ) -> list[Entity]:
        """Return all files directly included by *file_path*."""
        entity = self._find_entity_by_path(file_path, "file", project_id)
        if not entity:
            return []
        neighbors = self._store.get_neighbors(
            entity.id,
            relation_types=["INCLUDES"],
            direction="out",
        )
        return [n[0] for n in neighbors]

    def file_dependents(
        self,
        file_path: str,
        project_id: str | None = None,
    ) -> list[Entity]:
        """Return all files that include *file_path*."""
        entity = self._find_entity_by_path(file_path, "file", project_id)
        if not entity:
            return []
        neighbors = self._store.get_neighbors(
            entity.id,
            relation_types=["INCLUDES"],
            direction="in",
        )
        return [n[0] for n in neighbors]

    # ------------------------------------------------------------------
    # Concept / template queries
    # ------------------------------------------------------------------

    def concept_implementations(
        self,
        concept_name: str,
        project_id: str | None = None,
    ) -> list[Entity]:
        """Return all classes/structs that implement *concept_name*."""
        entity = self._find_entity(concept_name, "concept", project_id)
        if not entity:
            return []
        neighbors = self._store.get_neighbors(
            entity.id,
            relation_types=["IMPLEMENTS", "USES_CONCEPT"],
            direction="in",
        )
        return [n[0] for n in neighbors]

    # ------------------------------------------------------------------
    # Impact analysis
    # ------------------------------------------------------------------

    def impact_analysis(
        self,
        entity_name: str,
        project_id: str | None = None,
        max_depth: int = 3,
    ) -> dict[str, list[Entity]]:
        """Analyse the impact of modifying *entity_name*.

        Returns a dict with keys:
        - ``direct_callers``: functions that call this entity
        - ``direct_callees``: functions called by this entity
        - ``transitive_deps``: all entities reachable via CALLS/REFERENCES
        - ``file_deps``: files that include the file containing this entity
        """
        entity = self._find_entity(entity_name, None, project_id)
        if not entity:
            return {
                "direct_callers": [],
                "direct_callees": [],
                "transitive_deps": [],
                "file_deps": [],
            }

        direct_callers = self._store.get_neighbors(
            entity.id,
            relation_types=["CALLS"],
            direction="in",
        )
        direct_callees = self._store.get_neighbors(
            entity.id,
            relation_types=["CALLS"],
            direction="out",
        )
        transitive_deps = self._store.get_impact_scope(
            entity.id,
            relation_types=["CALLS", "REFERENCES", "INHERITS_FROM"],
            max_depth=max_depth,
        )
        file_deps = self._store.get_neighbors(
            EntityId.from_raw(
                entity.project_id, entity.file_path, "file", entity.file_path
            ),
            relation_types=["INCLUDES"],
            direction="in",
        )

        return {
            "direct_callers": [e[0] for e in direct_callers],
            "direct_callees": [e[0] for e in direct_callees],
            "transitive_deps": transitive_deps,
            "file_deps": [e[0] for e in file_deps],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_entity(
        self,
        name: str,
        entity_type: str | tuple[str, ...] | None,
        project_id: str | None,
    ) -> Entity | None:
        """Find an entity by name, optionally filtering by type and project."""
        types = (entity_type,) if isinstance(entity_type, str) else entity_type
        for nid, data in self._store._g.nodes(data=True):
            if data.get("name") != name:
                continue
            if types and data.get("entity_type") not in types:
                continue
            if project_id and data.get("project_id") != project_id:
                continue
            return self._store.get_entity(EntityId(*nid.split(":", 3)))
        return None

    def _find_entity_by_path(
        self,
        file_path: str,
        entity_type: str,
        project_id: str | None,
    ) -> Entity | None:
        """Find a file entity by its path."""
        for nid, data in self._store._g.nodes(data=True):
            if data.get("entity_type") != entity_type:
                continue
            if data.get("file_path") != file_path:
                continue
            if project_id and data.get("project_id") != project_id:
                continue
            return self._store.get_entity(EntityId(*nid.split(":", 3)))
        return None
