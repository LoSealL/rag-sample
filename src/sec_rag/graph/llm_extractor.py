"""LLM-based fallback extractor for complex cases.

When tree-sitter AST parsing fails (macros, heavy template metaprogramming,
preprocessor conditionals) or when cross-file indirect references are too
ambiguous, this extractor asks the local LLM to identify entities and
relations from raw source text.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .models import Entity, EntityId, Relation


class LLMGraphExtractor:
    """Fallback extractor that uses an LLM to identify entities and relations."""

    def __init__(self, llm_client) -> None:
        self._llm = llm_client

    def extract(
        self,
        source: str,
        file_path: str,
        project_id: str,
    ) -> tuple[list[Entity], list[Relation]]:
        """Extract entities and relations using LLM.

        Returns the same (entities, relations) tuple as CppAstExtractor so
        the two are interchangeable.
        """
        prompt = self._build_extraction_prompt(source, file_path)
        try:
            raw = self._llm.complete(prompt)
        except Exception:
            return [], []

        payload = self._extract_json(raw)
        if not isinstance(payload, dict):
            return [], []

        entities = self._parse_entities(payload.get("entities", []), file_path, project_id)
        relations = self._parse_relations(payload.get("relations", []), project_id)
        return entities, relations

    # ------------------------------------------------------------------
    # Prompt engineering
    # ------------------------------------------------------------------

    def _build_extraction_prompt(self, source: str, file_path: str) -> str:
        lines = source.split("\n")
        numbered = "\n".join(f"{i+1:04d}: {line}" for i, line in enumerate(lines))
        return (
            "You are a C++ code analysis expert. Analyse the following source file and "
            "output a JSON object describing all semantic units (entities) and their "
            "relationships.\n\n"
            f"File: {file_path}\n\n"
            "Entity types: file, namespace, class, struct, function, type_alias, concept, template\n"
            "Relation types: INCLUDES, CONTAINS, CALLS, REFERENCES, INHERITS_FROM, IMPLEMENTS\n\n"
            "Output schema (STRICT JSON only):\n"
            '{\n'
            '  "entities": [\n'
            '    {\n'
            '      "name": "MyClass",\n'
            '      "entity_type": "class",\n'
            '      "line_start": 10,\n'
            '      "line_end": 25\n'
            '    }\n'
            '  ],\n'
            '  "relations": [\n'
            '    {\n'
            '      "source_name": "main",\n'
            '      "source_type": "function",\n'
            '      "target_name": "foo",\n'
            '      "target_type": "function",\n'
            '      "relation_type": "CALLS",\n'
            '      "line": 42\n'
            '    }\n'
            '  ]\n'
            '}\n\n'
            "Rules:\n"
            "1. Only include entities with definitions (not forward declarations).\n"
            "2. For member functions, prefix with class name: 'MyClass::foo'.\n"
            "3. For templates, use entity_type 'template' and name the class/function.\n"
            "4. line_start/line_end are 1-based and inclusive.\n"
            "5. Return ONLY the JSON object, no markdown.\n\n"
            f"Source code:\n{numbered}"
        )

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any] | None:
        raw = raw.strip()
        try:
            return json.loads(raw)
        except Exception:
            pass
        # Remove markdown fences
        cleaned = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except Exception:
            pass
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return None

    def _parse_entities(
        self,
        raw_entities: list[dict[str, Any]],
        file_path: str,
        project_id: str,
    ) -> list[Entity]:
        entities: list[Entity] = []
        for item in raw_entities:
            name = str(item.get("name", ""))
            etype = str(item.get("entity_type", ""))
            if not name or not etype:
                continue
            start = int(item.get("line_start", 0))
            end = int(item.get("line_end", 0))
            eid = EntityId.from_raw(project_id, file_path, etype, name)
            entities.append(
                Entity(
                    id=eid,
                    name=name,
                    entity_type=etype,
                    file_path=file_path,
                    line_start=start,
                    line_end=end,
                    project_id=project_id,
                )
            )
        return entities

    def _parse_relations(
        self,
        raw_relations: list[dict[str, Any]],
        project_id: str,
    ) -> list[Relation]:
        relations: list[Relation] = []
        for item in raw_relations:
            src_name = str(item.get("source_name", ""))
            src_type = str(item.get("source_type", ""))
            dst_name = str(item.get("target_name", ""))
            dst_type = str(item.get("target_type", ""))
            rtype = str(item.get("relation_type", ""))
            if not all((src_name, src_type, dst_name, dst_type, rtype)):
                continue
            # We don't know the file paths here, so we use empty file_path
            # and rely on cross-file resolution later
            src_id = EntityId.from_raw(project_id, "", src_type, src_name)
            dst_id = EntityId.from_raw(project_id, "", dst_type, dst_name)
            relations.append(
                Relation(
                    source=src_id,
                    target=dst_id,
                    relation_type=rtype,
                    line=int(item.get("line", 0)),
                )
            )
        return relations
