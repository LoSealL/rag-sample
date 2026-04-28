"""AST-based entity and relation extractor for C/C++ code.

Uses tree-sitter to precisely extract semantic units and their relationships
from C++ source code.  Phase 2 adds CALLS, REFERENCES, INHERITS_FROM and
IMPLEMENTS relation extraction.
"""

from __future__ import annotations

import re
from pathlib import Path

import tree_sitter
import tree_sitter_cpp

from .models import Entity, EntityId, Relation


def _cpp_parser() -> tree_sitter.Parser:
    """Create a tree-sitter C++ parser (reusable across files)."""
    lang = tree_sitter.Language(tree_sitter_cpp.language())
    return tree_sitter.Parser(lang)


_INCLUDE_RE = re.compile(r'#\s*include\s+["<]([^">]+)[">]')


class SymbolTable:
    """Project-wide symbol table for cross-file resolution.

    As files are processed, their exported symbols are registered here.
    Later, when extracting relations inside a function body, unresolved
    identifiers can be looked up in this table to create cross-file edges.
    """

    def __init__(self) -> None:
        # name -> list[EntityId] (multiple overloads / specialisations possible)
        self._symbols: dict[str, list[EntityId]] = {}

    def register(self, entity: Entity) -> None:
        """Register an entity so it can be resolved by name later."""
        ids = self._symbols.setdefault(entity.name, [])
        if entity.id not in ids:
            ids.append(entity.id)

    def lookup(self, name: str) -> list[EntityId]:
        """Return all EntityIds matching *name* (may be ambiguous)."""
        return self._symbols.get(name, [])

    def lookup_unique(self, name: str) -> EntityId | None:
        """Return the unique EntityId for *name*, or None if ambiguous/absent."""
        ids = self._symbols.get(name)
        if ids and len(ids) == 1:
            return ids[0]
        return None


class CppAstExtractor:
    """Extract entities and relations from C++ source via tree-sitter AST."""

    def __init__(self, symbol_table: SymbolTable | None = None) -> None:
        self._parser = _cpp_parser()
        self._symbol_table = symbol_table or SymbolTable()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        source: str,
        file_path: str,
        project_id: str,
    ) -> tuple[list[Entity], list[Relation]]:
        """Extract all entities and relations from a single source file."""
        entities: list[Entity] = []
        relations: list[Relation] = []

        file_entity = self._make_file_entity(file_path, project_id)
        entities.append(file_entity)

        try:
            tree = self._parser.parse(source.encode("utf-8"))
        except Exception:
            relations.extend(self._extract_includes(source, file_path, project_id))
            return entities, relations

        root = tree.root_node
        if root is None:
            relations.extend(self._extract_includes(source, file_path, project_id))
            return entities, relations

        # Phase 1: extract all entities
        namespace_stack: list[Entity] = []
        self._walk_ast(
            root, source, file_path, project_id,
            namespace_stack, entities, relations, parent_class=None,
        )

        # Register entities in symbol table for cross-file resolution
        for entity in entities:
            self._symbol_table.register(entity)

        # Phase 2: extract deep relations (CALLS, REFERENCES, INHERITS_FROM)
        relations.extend(self._extract_semantic_relations(root, source, file_path, project_id, entities))

        # Phase 3: includes + nesting
        relations.extend(self._extract_includes(source, file_path, project_id))
        relations.extend(self._extract_nesting_relations(entities))

        return entities, relations

    # ------------------------------------------------------------------
    # AST walking (entity extraction)
    # ------------------------------------------------------------------

    def _walk_ast(
        self,
        node,
        source: str,
        file_path: str,
        project_id: str,
        namespace_stack: list[Entity],
        entities: list[Entity],
        relations: list[Relation],
        parent_class: Entity | None,
    ) -> None:
        entity = self._node_to_entity(node, source, file_path, project_id, parent_class)
        if entity:
            entities.append(entity)
            if entity.entity_type == "namespace":
                namespace_stack.append(entity)
            if entity.entity_type in ("class", "struct", "class_template", "struct_template"):
                parent_class = entity

            for child in node.children:
                self._walk_ast(
                    child, source, file_path, project_id,
                    namespace_stack, entities, relations, parent_class,
                )

            if entity.entity_type == "namespace":
                namespace_stack.pop()
        else:
            for child in node.children:
                self._walk_ast(
                    child, source, file_path, project_id,
                    namespace_stack, entities, relations, parent_class,
                )

    def _node_to_entity(
        self,
        node,
        source: str,
        file_path: str,
        project_id: str,
        parent_class: Entity | None,
    ) -> Entity | None:
        entity_type = self._node_type_to_entity_type(node.type)
        if not entity_type:
            return None

        name = self._extract_name(node, source)
        if not name:
            return None

        if entity_type in ("class", "struct", "class_template", "struct_template"):
            has_body = any(c.type == "field_declaration_list" for c in node.children)
            if not has_body:
                return None

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        eid = EntityId.from_raw(project_id, file_path, entity_type, name)

        declaration = self._text_from_node(source, node)
        declaration = declaration.split("\n")[0][:200]

        entity = Entity(
            id=eid,
            name=name,
            entity_type=entity_type,
            file_path=file_path,
            line_start=start_line,
            line_end=end_line,
            declaration=declaration,
            project_id=project_id,
        )

        if parent_class and entity_type == "function":
            entity.name = f"{parent_class.name}::{name}"
            entity.id = EntityId.from_raw(project_id, file_path, entity_type, entity.name)

        return entity

    # ------------------------------------------------------------------
    # Semantic relation extraction (Phase 2)
    # ------------------------------------------------------------------

    def _extract_semantic_relations(
        self,
        root,
        source: str,
        file_path: str,
        project_id: str,
        entities: list[Entity],
    ) -> list[Relation]:
        """Extract CALLS, REFERENCES, INHERITS_FROM from AST."""
        relations: list[Relation] = []

        # Build a quick lookup: node -> entity (for functions/classes)
        entity_by_line: dict[int, Entity] = {}
        for e in entities:
            if e.entity_type in ("function", "class", "struct", "class_template", "struct_template"):
                entity_by_line[e.line_start] = e

        for node in self._iter_nodes(root):
            if node.type == "function_definition":
                func_entity = entity_by_line.get(node.start_point[0] + 1)
                if func_entity:
                    relations.extend(
                        self._extract_calls(node, func_entity, source, file_path, project_id)
                    )
                    relations.extend(
                        self._extract_type_references(node, func_entity, source, file_path, project_id)
                    )
            elif node.type in ("class_specifier", "struct_specifier"):
                class_entity = entity_by_line.get(node.start_point[0] + 1)
                if class_entity:
                    relations.extend(
                        self._extract_inheritance(node, class_entity, source, file_path, project_id)
                    )
                    relations.extend(
                        self._extract_member_references(node, class_entity, source, file_path, project_id)
                    )

        return relations

    def _extract_calls(
        self,
        func_node,
        func_entity: Entity,
        source: str,
        file_path: str,
        project_id: str,
    ) -> list[Relation]:
        """Find call_expression nodes inside a function body."""
        relations: list[Relation] = []
        seen: set[str] = set()

        for node in self._iter_nodes(func_node):
            if node.type == "call_expression":
                callee_name = self._extract_callee_name(node, source)
                if callee_name and callee_name not in seen:
                    seen.add(callee_name)
                    # Try to resolve via symbol table
                    callee_id = self._symbol_table.lookup_unique(callee_name)
                    if callee_id:
                        relations.append(
                            Relation(
                                source=func_entity.id,
                                target=callee_id,
                                relation_type="CALLS",
                                file_path=file_path,
                                line=node.start_point[0] + 1,
                            )
                        )

        return relations

    def _extract_type_references(
        self,
        func_node,
        func_entity: Entity,
        source: str,
        file_path: str,
        project_id: str,
    ) -> list[Relation]:
        """Extract type references from function parameters and return type."""
        relations: list[Relation] = []
        seen: set[str] = set()

        # Look at function_declarator children for parameter_list
        for child in func_node.children:
            if child.type == "function_declarator":
                for sub in self._iter_nodes(child):
                    if sub.type in ("type_identifier", "primitive_type", "qualified_identifier"):
                        type_name = self._text_from_node(source, sub)
                        if type_name and type_name not in seen:
                            seen.add(type_name)
                            ref_id = self._symbol_table.lookup_unique(type_name)
                            if ref_id:
                                relations.append(
                                    Relation(
                                        source=func_entity.id,
                                        target=ref_id,
                                        relation_type="REFERENCES",
                                        file_path=file_path,
                                        line=sub.start_point[0] + 1,
                                    )
                                )

        return relations

    def _extract_inheritance(
        self,
        class_node,
        class_entity: Entity,
        source: str,
        file_path: str,
        project_id: str,
    ) -> list[Relation]:
        """Extract base_class_clause from class/struct definition."""
        relations: list[Relation] = []

        for child in class_node.children:
            if child.type == "base_class_clause":
                for base in self._iter_nodes(child):
                    if base.type in ("type_identifier", "qualified_identifier"):
                        base_name = self._text_from_node(source, base)
                        base_id = self._symbol_table.lookup_unique(base_name)
                        if base_id:
                            relations.append(
                                Relation(
                                    source=class_entity.id,
                                    target=base_id,
                                    relation_type="INHERITS_FROM",
                                    file_path=file_path,
                                    line=base.start_point[0] + 1,
                                )
                            )

        return relations

    def _extract_member_references(
        self,
        class_node,
        class_entity: Entity,
        source: str,
        file_path: str,
        project_id: str,
    ) -> list[Relation]:
        """Extract type references from member variable declarations."""
        relations: list[Relation] = []
        seen: set[str] = set()

        for node in self._iter_nodes(class_node):
            if node.type == "field_declaration":
                for sub in node.children:
                    if sub.type in ("type_identifier", "qualified_identifier", "template_type"):
                        type_name = self._text_from_node(source, sub)
                        if type_name and type_name not in seen:
                            seen.add(type_name)
                            ref_id = self._symbol_table.lookup_unique(type_name)
                            if ref_id:
                                relations.append(
                                    Relation(
                                        source=class_entity.id,
                                        target=ref_id,
                                        relation_type="REFERENCES",
                                        file_path=file_path,
                                        line=sub.start_point[0] + 1,
                                    )
                                )

        return relations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_nodes(root):
        """Yield all nodes in the subtree (pre-order)."""
        stack = [root]
        while stack:
            node = stack.pop()
            yield node
            # Reverse children so they are processed left-to-right
            for child in reversed(node.children):
                stack.append(child)

    def _extract_callee_name(self, call_node, source: str) -> str:
        """Extract the function name from a call_expression."""
        # call_expression usually has: function (identifier/field_expression) + arguments
        for child in call_node.children:
            if child.type == "identifier":
                return self._text_from_node(source, child)
            if child.type == "field_expression":
                # object.method() -> return method name
                for sub in child.children:
                    if sub.type == "field_identifier":
                        return self._text_from_node(source, sub)
            if child.type == "qualified_identifier":
                return self._text_from_node(source, child)
        return ""

    # ------------------------------------------------------------------
    # Static helpers (entity naming, etc.)
    # ------------------------------------------------------------------

    @staticmethod
    def _node_type_to_entity_type(node_type: str) -> str:
        mapping = {
            "class_specifier": "class",
            "struct_specifier": "struct",
            "function_definition": "function",
            "alias_declaration": "type_alias",
            "namespace_definition": "namespace",
            "concept_definition": "concept",
            "template_declaration": "template",
        }
        return mapping.get(node_type, "")

    @staticmethod
    def _extract_name(node, source: str) -> str:
        try:
            name_node = node.child_by_field_name("name")
            if name_node:
                return CppAstExtractor._text_from_node(source, name_node)
        except Exception:
            pass

        type_map = {
            "class_specifier": "type_identifier",
            "struct_specifier": "type_identifier",
            "function_definition": "identifier",
            "alias_declaration": "type_identifier",
            "namespace_definition": "namespace_identifier",
            "concept_definition": "identifier",
        }

        expected = type_map.get(node.type)
        for child in node.children:
            if child.type == expected:
                return CppAstExtractor._text_from_node(source, child)
            if child.type == "identifier":
                return CppAstExtractor._text_from_node(source, child)
            if child.type == "function_declarator":
                for sub in child.children:
                    if sub.type in ("identifier", "field_identifier", "operator_name", "destructor_name"):
                        return CppAstExtractor._text_from_node(source, sub)
            if child.type in (
                "class_specifier", "struct_specifier", "function_definition", "concept_definition",
            ):
                return CppAstExtractor._extract_name(child, source)

        return ""

    @staticmethod
    def _text_from_node(source: str, node) -> str:
        source_bytes = source.encode("utf-8")
        return source_bytes[node.start_byte : node.end_byte].decode("utf-8")

    def _extract_includes(self, source: str, file_path: str, project_id: str) -> list[Relation]:
        relations: list[Relation] = []
        source_file_id = EntityId.from_raw(project_id, file_path, "file", Path(file_path).name)
        for match in _INCLUDE_RE.finditer(source):
            included_path = match.group(1)
            included_file_id = EntityId.from_raw(
                project_id, included_path, "file", Path(included_path).name,
            )
            relations.append(
                Relation(
                    source=source_file_id,
                    target=included_file_id,
                    relation_type="INCLUDES",
                    file_path=file_path,
                    line=source[: match.start()].count("\n") + 1,
                )
            )
        return relations

    def _extract_nesting_relations(self, entities: list[Entity]) -> list[Relation]:
        relations: list[Relation] = []
        containers = [e for e in entities if e.entity_type in ("namespace", "class", "struct", "class_template", "struct_template")]
        contained = [e for e in entities if e.entity_type not in ("file", "namespace")]
        for outer in containers:
            for inner in contained:
                if outer.id == inner.id:
                    continue
                if (
                    outer.file_path == inner.file_path
                    and outer.line_start <= inner.line_start
                    and outer.line_end >= inner.line_end
                ):
                    relations.append(
                        Relation(
                            source=outer.id,
                            target=inner.id,
                            relation_type="CONTAINS",
                            file_path=outer.file_path,
                        )
                    )
        return relations

    @staticmethod
    def _make_file_entity(file_path: str, project_id: str) -> Entity:
        name = Path(file_path).name
        eid = EntityId.from_raw(project_id, file_path, "file", name)
        return Entity(
            id=eid,
            name=name,
            entity_type="file",
            file_path=file_path,
            line_start=1,
            line_end=1,
            project_id=project_id,
        )
