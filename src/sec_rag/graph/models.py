"""Data models for the Code Knowledge Graph."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EntityId:
    """Globally unique identifier for a code entity.

    Composed of project_id, file_path, entity_type and name so that
    two symbols with the same name in different files or projects do
    not collide.
    """

    project_id: str
    file_path: str
    entity_type: str
    name: str

    def __str__(self) -> str:
        return f"{self.project_id}:{self.file_path}:{self.entity_type}:{self.name}"

    def __hash__(self) -> int:
        return hash(str(self))

    @classmethod
    def from_raw(
        cls,
        project_id: str,
        file_path: str,
        entity_type: str,
        name: str,
    ) -> EntityId:
        """Create an EntityId, normalising inputs."""
        return cls(
            project_id=project_id,
            file_path=file_path,
            entity_type=entity_type,
            name=name,
        )


@dataclass
class Entity:
    """A node in the code knowledge graph — a concrete code symbol."""

    id: EntityId
    name: str
    entity_type: str
    file_path: str
    line_start: int
    line_end: int
    chunk_id: str = ""
    declaration: str = ""
    project_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "entity_type": self.entity_type,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "chunk_id": self.chunk_id,
            "declaration": self.declaration,
            "project_id": self.project_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Entity:
        eid = EntityId(*data["id"].split(":", 3))
        return cls(
            id=eid,
            name=data["name"],
            entity_type=data["entity_type"],
            file_path=data["file_path"],
            line_start=data["line_start"],
            line_end=data["line_end"],
            chunk_id=data.get("chunk_id", ""),
            declaration=data.get("declaration", ""),
            project_id=data.get("project_id", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Relation:
    """A directed edge between two entities in the knowledge graph."""

    source: EntityId
    target: EntityId
    relation_type: str
    file_path: str = ""
    line: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "target": str(self.target),
            "relation_type": self.relation_type,
            "file_path": self.file_path,
            "line": self.line,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Relation:
        return cls(
            source=EntityId(*data["source"].split(":", 3)),
            target=EntityId(*data["target"].split(":", 3)),
            relation_type=data["relation_type"],
            file_path=data.get("file_path", ""),
            line=data.get("line", 0),
            metadata=data.get("metadata", {}),
        )
