"""Privacy mode configuration and file matching for sec-rag.

Users can designate certain files as high-confidential. These files are indexed
but their raw content is never sent to the LLM. Instead, a privacy-safe summary
is generated at indexing time and substituted at query time.

Configuration formats supported:
    - .privacy_config.json (recommended)
    - Environment variable PRIVACY_CONFIG_PATH
    - CLI flag --privacy-config

Example .privacy_config.json:
    {
        "exact_paths": ["src/secrets/api_keys.h", "config/production.yml"],
        "directories": ["src/crypto/", "internal/"],
        "patterns": [".*_secret\\.py$", ".*\\.key$"],
        "extensions": [".pem", ".p12"],
        "enabled": true
    }
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


DEFAULT_CONFIG_FILENAME = ".privacy_config.json"


@dataclass
class PrivacyConfig:
    """User-supplied privacy rules."""

    exact_paths: list[str] = field(default_factory=list)
    directories: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)  # regex
    extensions: list[str] = field(default_factory=list)
    enabled: bool = True

    # Compiled regex patterns (populated lazily)
    _compiled: list[re.Pattern[str]] = field(default_factory=list, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrivacyConfig:
        """Load from a dictionary (e.g. parsed JSON)."""
        return cls(
            exact_paths=[str(p) for p in data.get("exact_paths", [])],
            directories=[str(p) for p in data.get("directories", [])],
            patterns=[str(p) for p in data.get("patterns", [])],
            extensions=[str(e) for e in data.get("extensions", [])],
            enabled=bool(data.get("enabled", True)),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> PrivacyConfig:
        """Load from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def auto_load(cls, project_roots: list[str]) -> PrivacyConfig | None:
        """Search for .privacy_config.json in project roots or env var."""
        env_path = os.environ.get("PRIVACY_CONFIG_PATH")
        if env_path and Path(env_path).exists():
            logger.info("Loading privacy config from env: {}", env_path)
            return cls.from_file(env_path)

        for root in project_roots:
            candidate = Path(root) / DEFAULT_CONFIG_FILENAME
            if candidate.exists():
                logger.info("Loading privacy config from {}", candidate)
                return cls.from_file(candidate)

        return None

    def _ensure_compiled(self) -> None:
        if self._compiled:
            return
        self._compiled = []
        for pat in self.patterns:
            try:
                self._compiled.append(re.compile(pat))
            except re.error as exc:
                logger.warning("Invalid privacy pattern '{}': {}", pat, exc)

    def is_private(self, file_path: str) -> bool:
        """Check whether a file path matches any privacy rule."""
        if not self.enabled:
            return False

        # Normalise to forward slashes for cross-platform consistency
        normalised = file_path.replace("\\", "/")

        # Exact match
        for exact in self.exact_paths:
            if normalised.endswith(exact.replace("\\", "/")):
                return True

        # Directory containment
        for directory in self.directories:
            dir_norm = directory.replace("\\", "/").rstrip("/") + "/"
            if dir_norm in normalised + "/":
                return True

        # Extension
        suffix = Path(file_path).suffix.lower()
        for ext in self.extensions:
            if suffix == ext.lower():
                return True

        # Regex patterns
        self._ensure_compiled()
        for compiled in self._compiled:
            if compiled.search(normalised):
                return True

        return False

    def summary(self) -> str:
        total = (
            len(self.exact_paths)
            + len(self.directories)
            + len(self.patterns)
            + len(self.extensions)
        )
        return (
            f"PrivacyConfig(enabled={self.enabled}, "
            f"exact={len(self.exact_paths)}, dirs={len(self.directories)}, "
            f"patterns={len(self.patterns)}, exts={len(self.extensions)}, total={total})"
        )


def generate_privacy_summary(
    chunk_text: str,
    file_path: str,
    llm,
    declaration: str = "",
) -> str:
    """Generate a privacy-safe summary for a high-confidential file chunk.

    The summary must describe *what* the code does without revealing
    implementation details, literal strings, or sensitive constants.
    """
    if llm is None:
        return (
            f"[PRIVACY-SAFE SUMMARY] This is a high-confidential component in "
            f"{file_path}. The raw implementation is withheld per privacy policy."
        )

    snippet = chunk_text[:2000]
    prompt = (
        "Generate a concise privacy-safe summary of the following code. "
        "DO NOT include any source code, literal strings, numeric constants, "
        "IP addresses, credentials, or algorithmic details. "
        "Describe ONLY the high-level purpose, inputs, outputs, and "
        "security relevance in plain natural language.\n\n"
        f"File: {file_path}\n"
        f"Declaration: {declaration}\n"
        f"Code snippet:\n{snippet}\n\n"
        "Privacy-safe summary:"
    )
    try:
        summary = str(llm.complete(prompt)).strip()
        if not summary:
            raise ValueError("Empty summary")
        return summary
    except Exception as exc:
        logger.warning(
            "Privacy summary generation failed for {}: {}", file_path, exc
        )
        return (
            f"[PRIVACY-SAFE SUMMARY] High-confidential component in {file_path}. "
            f"Raw implementation withheld per privacy policy."
        )
