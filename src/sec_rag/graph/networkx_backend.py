"""NetworkX graph backend implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None  # type: ignore[assignment]

from .backend import GraphBackend
from .models import Entity, EntityId, Relation


class NetworkXBackend(GraphBackend):
    """Directed graph backend using NetworkX with JSON persistence."""

    def __init__(self, persist_path: str = ".rag_index/knowledge_graph.json") -> None:
        if nx is None:
            raise RuntimeError(
                "networkx is required for NetworkXBackend. "
                "Install it with: uv pip install networkx"
            )
        self._persist_path = Path(persist_path)
        self._g: nx.DiGraph = nx.DiGraph()
        self._load()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_entity(self, entity: Entity) -> None:
        nid = str(entity.id)
        self._g.add_node(
            nid,
            name=entity.name,
            entity_type=entity.entity_type,
            file_path=entity.file_path,
            line_start=entity.line_start,
            line_end=entity.line_end,
            chunk_id=entity.chunk_id,
            declaration=entity.declaration,
            project_id=entity.project_id,
            metadata=entity.metadata,
        )

    def remove_entity(self, entity_id: EntityId) -> None:
        nid = str(entity_id)
        if self._g.has_node(nid):
            self._g.remove_node(nid)

    def add_relation(self, relation: Relation) -> None:
        src = str(relation.source)
        dst = str(relation.target)
        if not self._g.has_node(src):
            self._g.add_node(src, name=relation.source.name, entity_type=relation.source.entity_type)
        if not self._g.has_node(dst):
            self._g.add_node(dst, name=relation.target.name, entity_type=relation.target.entity_type)
        self._g.add_edge(
            src,
            dst,
            relation_type=relation.relation_type,
            file_path=relation.file_path,
            line=relation.line,
            **relation.metadata,
        )

    def remove_relation(self, source: EntityId, target: EntityId, relation_type: str | None = None) -> None:
        src = str(source)
        dst = str(target)
        if not self._g.has_edge(src, dst):
            return
        if relation_type is None:
            self._g.remove_edge(src, dst)
        else:
            # NetworkX doesn't support typed multi-edges easily,
            # so we only remove if the edge's relation_type matches.
            data = self._g.edges[src, dst]
            if data.get("relation_type") == relation_type:
                self._g.remove_edge(src, dst)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_entity(self, entity_id: EntityId) -> Entity | None:
        nid = str(entity_id)
        if nid not in self._g:
            return None
        data = self._g.nodes[nid]
        return Entity(
            id=entity_id,
            name=data.get("name", ""),
            entity_type=data.get("entity_type", ""),
            file_path=data.get("file_path", ""),
            line_start=data.get("line_start", 0),
            line_end=data.get("line_end", 0),
            chunk_id=data.get("chunk_id", ""),
            declaration=data.get("declaration", ""),
            project_id=data.get("project_id", ""),
            metadata=data.get("metadata", {}),
        )

    def get_neighbors(
        self,
        entity_id: EntityId,
        relation_types: list[str] | None = None,
        direction: str = "both",
    ) -> list[tuple[Entity, Relation]]:
        nid = str(entity_id)
        if nid not in self._g:
            return []

        results: list[tuple[Entity, Relation]] = []

        def _collect(u: str, v: str, edge_data: dict[str, Any], forward: bool) -> None:
            rtype = edge_data.get("relation_type", "")
            if relation_types is not None and rtype not in relation_types:
                return
            other_nid = v if forward else u
            other_id = EntityId(*other_nid.split(":", 3))
            other = self.get_entity(other_id)
            if other is None:
                return
            rel = Relation(
                source=EntityId(*u.split(":", 3)),
                target=EntityId(*v.split(":", 3)),
                relation_type=rtype,
                file_path=edge_data.get("file_path", ""),
                line=edge_data.get("line", 0),
                metadata={k: v for k, v in edge_data.items() if k not in ("relation_type", "file_path", "line")},
            )
            results.append((other, rel))

        if direction in ("out", "both"):
            for succ in self._g.successors(nid):
                data = self._g.edges[nid, succ]
                _collect(nid, succ, data, forward=True)

        if direction in ("in", "both"):
            for pred in self._g.predecessors(nid):
                data = self._g.edges[pred, nid]
                _collect(pred, nid, data, forward=False)

        return results

    def get_entities_in_file(self, file_path: str, project_id: str | None = None) -> list[Entity]:
        results: list[Entity] = []
        for nid, data in self._g.nodes(data=True):
            if data.get("file_path") != file_path:
                continue
            if project_id is not None and data.get("project_id") != project_id:
                continue
            entity = self.get_entity(EntityId(*nid.split(":", 3)))
            if entity:
                results.append(entity)
        return results

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def get_paths(
        self,
        source: EntityId,
        target: EntityId,
        max_depth: int = 5,
        relation_types: list[str] | None = None,
    ) -> list[list[Entity]]:
        src = str(source)
        dst = str(target)
        if src not in self._g or dst not in self._g:
            return []

        if relation_types is None:
            paths = list(nx.all_simple_paths(self._g, src, dst, cutoff=max_depth))
        else:
            view = nx.DiGraph()
            view.add_nodes_from(self._g.nodes(data=True))
            for u, v, d in self._g.edges(data=True):
                if d.get("relation_type") in relation_types:
                    view.add_edge(u, v, **d)
            paths = list(nx.all_simple_paths(view, src, dst, cutoff=max_depth))

        result: list[list[Entity]] = []
        for path in paths:
            entity_path = []
            for node_id in path:
                e = self.get_entity(EntityId(*node_id.split(":", 3)))
                if e:
                    entity_path.append(e)
            if len(entity_path) == len(path):
                result.append(entity_path)
        return result

    def get_impact_scope(
        self,
        entity_id: EntityId,
        relation_types: list[str] | None = None,
        max_depth: int = 3,
    ) -> list[Entity]:
        nid = str(entity_id)
        if nid not in self._g:
            return []

        if relation_types is None:
            nodes = nx.single_source_shortest_path_length(self._g, nid, cutoff=max_depth)
        else:
            view = nx.DiGraph()
            view.add_nodes_from(self._g.nodes(data=True))
            for u, v, d in self._g.edges(data=True):
                if d.get("relation_type") in relation_types:
                    view.add_edge(u, v, **d)
            nodes = nx.single_source_shortest_path_length(view, nid, cutoff=max_depth)

        result: list[Entity] = []
        for node_id in nodes:
            if node_id == nid:
                continue
            e = self.get_entity(EntityId(*node_id.split(":", 3)))
            if e:
                result.append(e)
        return result

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "nodes": self._g.number_of_nodes(),
            "edges": self._g.number_of_edges(),
            "entities_by_type": {
                t: sum(1 for _, d in self._g.nodes(data=True) if d.get("entity_type") == t)
                for t in set(d.get("entity_type", "") for _, d in self._g.nodes(data=True))
            },
        }

    def clear(self) -> None:
        self._g.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self._g, edges="edges")
        with open(self._persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _load(self) -> None:
        if not self._persist_path.exists():
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._g = nx.node_link_graph(data, edges="edges")
        except Exception:
            self._g = nx.DiGraph()

    # ------------------------------------------------------------------
    # Raw access (for export, visualisation, etc.)
    # ------------------------------------------------------------------

    def nodes(self) -> list[tuple[str, dict[str, Any]]]:
        return list(self._g.nodes(data=True))

    def edges(self) -> list[tuple[str, str, dict[str, Any]]]:
        return list(self._g.edges(data=True))
