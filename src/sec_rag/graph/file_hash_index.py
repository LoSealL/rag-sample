"""File hash tracker for incremental graph updates.

Tracks SHA-256 hashes of source files so that only changed files need to
be re-parsed when re-indexing a project.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class FileHashIndex:
    """Persistent file hash index for incremental updates.

    Stores a mapping from file_path -> content_hash.  When a project is
    re-indexed, only files whose hash has changed (or which are new) are
    re-processed.
    """

    def __init__(self, persist_path: str = ".rag_index/file_hashes.json") -> None:
        self._persist_path = Path(persist_path)
        self._hashes: dict[str, str] = {}
        self._load()

    def is_changed(self, file_path: str, source: str) -> bool:
        """Return True if *file_path* content has changed since last index."""
        current_hash = self._compute_hash(source)
        return self._hashes.get(file_path) != current_hash

    def update(self, file_path: str, source: str) -> None:
        """Record the current hash for *file_path*."""
        self._hashes[file_path] = self._compute_hash(source)

    def remove(self, file_path: str) -> None:
        """Remove *file_path* from the index (file deleted)."""
        self._hashes.pop(file_path, None)

    def get_stale_files(self, current_files: list[str]) -> list[str]:
        """Return files that were indexed previously but no longer exist."""
        current = set(current_files)
        return [f for f in self._hashes if f not in current]

    def save(self) -> None:
        """Persist the hash index to disk."""
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "w", encoding="utf-8") as f:
            json.dump(self._hashes, f, indent=2, ensure_ascii=False)

    def _load(self) -> None:
        if self._persist_path.exists():
            try:
                with open(self._persist_path, "r", encoding="utf-8") as f:
                    self._hashes = json.load(f)
            except Exception:
                self._hashes = {}

    @staticmethod
    def _compute_hash(source: str) -> str:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()
