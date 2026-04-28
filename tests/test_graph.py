"""Tests for the Code Knowledge Graph module."""

import json
import tempfile
from pathlib import Path

import pytest

from sec_rag.graph import (
    CppAstExtractor,
    Entity,
    EntityId,
    GraphBuilder,
    GraphQuery,
    GraphStore,
    Relation,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def sample_cpp_source():
    return '''
/**
 * @file types.hpp
 */
#pragma once

namespace rag {

using scalar_t = float;

struct Document {
    std::string id;
    std::string text;
};

class Pipeline {
public:
    void run() {}
};

} // namespace rag
'''


@pytest.fixture
def sample_cpp_with_includes():
    return '''
#include "rag/core/types.hpp"
#include <vector>

namespace rag {

float compute(const std::vector<float>& data) {
    return data[0];
}

} // namespace rag
'''


@pytest.fixture
def empty_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield GraphStore(persist_path=f"{tmpdir}/graph.json")


# ------------------------------------------------------------------
# Model tests
# ------------------------------------------------------------------

class TestEntityId:
    def test_string_representation(self):
        eid = EntityId("proj", "file.cpp", "class", "Foo")
        assert str(eid) == "proj:file.cpp:class:Foo"

    def test_from_raw(self):
        eid = EntityId.from_raw("p", "f", "function", "bar")
        assert eid.project_id == "p"
        assert eid.name == "bar"


class TestEntity:
    def test_round_trip_dict(self):
        eid = EntityId.from_raw("p", "f.cpp", "class", "Foo")
        entity = Entity(
            id=eid,
            name="Foo",
            entity_type="class",
            file_path="f.cpp",
            line_start=10,
            line_end=20,
            declaration="class Foo {};",
        )
        d = entity.to_dict()
        restored = Entity.from_dict(d)
        assert restored.name == "Foo"
        assert restored.line_start == 10


class TestRelation:
    def test_round_trip_dict(self):
        src = EntityId.from_raw("p", "a.cpp", "function", "main")
        dst = EntityId.from_raw("p", "b.cpp", "function", "foo")
        rel = Relation(source=src, target=dst, relation_type="CALLS")
        d = rel.to_dict()
        restored = Relation.from_dict(d)
        assert restored.relation_type == "CALLS"
        assert str(restored.source) == str(src)


# ------------------------------------------------------------------
# AST Extractor tests
# ------------------------------------------------------------------

class TestCppAstExtractor:
    def test_extract_entities(self, sample_cpp_source):
        extractor = CppAstExtractor()
        entities, _ = extractor.extract(sample_cpp_source, "types.hpp", "proj")

        # Should have: file, namespace, type_alias, struct, class
        assert len(entities) >= 5

        types = {e.entity_type for e in entities}
        assert "file" in types
        assert "namespace" in types
        assert "type_alias" in types
        assert "struct" in types
        assert "class" in types

    def test_extract_names(self, sample_cpp_source):
        extractor = CppAstExtractor()
        entities, _ = extractor.extract(sample_cpp_source, "types.hpp", "proj")

        names = {e.name for e in entities}
        assert "rag" in names
        assert "scalar_t" in names
        assert "Document" in names
        assert "Pipeline" in names

    def test_extract_includes(self, sample_cpp_with_includes):
        extractor = CppAstExtractor()
        entities, relations = extractor.extract(
            sample_cpp_with_includes, "main.cpp", "proj"
        )

        include_rels = [r for r in relations if r.relation_type == "INCLUDES"]
        assert len(include_rels) == 2

        targets = {r.target.name for r in include_rels}
        assert "types.hpp" in targets
        assert "vector" in targets

    def test_contains_relations(self, sample_cpp_source):
        extractor = CppAstExtractor()
        entities, relations = extractor.extract(sample_cpp_source, "types.hpp", "proj")

        contains = [r for r in relations if r.relation_type == "CONTAINS"]
        assert len(contains) > 0

        # namespace rag should contain struct Document and class Pipeline
        rag_id = EntityId.from_raw("proj", "types.hpp", "namespace", "rag")
        rag_contains = [r for r in contains if str(r.source) == str(rag_id)]
        targets = {r.target.name for r in rag_contains}
        assert "Document" in targets or "scalar_t" in targets

    def test_forward_declaration_not_extracted(self):
        source = "class Forward;\n"
        extractor = CppAstExtractor()
        entities, _ = extractor.extract(source, "fwd.hpp", "proj")
        # File entity + no class entity because it has no body
        assert len(entities) == 1
        assert entities[0].entity_type == "file"


# ------------------------------------------------------------------
# Graph Store tests
# ------------------------------------------------------------------

class TestGraphStore:
    def test_add_and_get_entity(self, empty_store):
        eid = EntityId.from_raw("p", "f.cpp", "class", "Foo")
        entity = Entity(
            id=eid,
            name="Foo",
            entity_type="class",
            file_path="f.cpp",
            line_start=1,
            line_end=10,
        )
        empty_store.add_entity(entity)
        retrieved = empty_store.get_entity(eid)
        assert retrieved is not None
        assert retrieved.name == "Foo"

    def test_add_relation(self, empty_store):
        src = EntityId.from_raw("p", "a.cpp", "function", "main")
        dst = EntityId.from_raw("p", "b.cpp", "function", "foo")
        rel = Relation(source=src, target=dst, relation_type="CALLS")
        empty_store.add_relation(rel)

        neighbors = empty_store.get_neighbors(src, relation_types=["CALLS"], direction="out")
        assert len(neighbors) == 1
        assert neighbors[0][0].name == "foo"

    def test_get_neighbors_filter_by_type(self, empty_store):
        src = EntityId.from_raw("p", "a.cpp", "function", "main")
        dst1 = EntityId.from_raw("p", "b.cpp", "function", "foo")
        dst2 = EntityId.from_raw("p", "c.cpp", "class", "Bar")

        empty_store.add_relation(Relation(source=src, target=dst1, relation_type="CALLS"))
        empty_store.add_relation(Relation(source=src, target=dst2, relation_type="REFERENCES"))

        calls_only = empty_store.get_neighbors(src, relation_types=["CALLS"], direction="out")
        assert len(calls_only) == 1
        assert calls_only[0][0].entity_type == "function"

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/graph.json"
            store = GraphStore(persist_path=path)
            eid = EntityId.from_raw("p", "f.cpp", "class", "Foo")
            store.add_entity(Entity(
                id=eid,
                name="Foo",
                entity_type="class",
                file_path="f.cpp",
                line_start=1,
                line_end=10,
            ))
            store.save()

            # Load into a new store
            store2 = GraphStore(persist_path=path)
            retrieved = store2.get_entity(eid)
            assert retrieved is not None
            assert retrieved.name == "Foo"

    def test_stats(self, empty_store):
        eid = EntityId.from_raw("p", "f.cpp", "class", "Foo")
        empty_store.add_entity(Entity(
            id=eid,
            name="Foo",
            entity_type="class",
            file_path="f.cpp",
            line_start=1,
            line_end=10,
        ))
        stats = empty_store.stats()
        assert stats["nodes"] == 1
        assert stats["edges"] == 0


# ------------------------------------------------------------------
# Graph Builder tests
# ------------------------------------------------------------------

class TestGraphBuilder:
    def test_add_file(self, sample_cpp_source):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(persist_path=f"{tmpdir}/graph.json")
            builder = GraphBuilder(store=store)
            builder.add_file(sample_cpp_source, "types.hpp", "proj")

            stats = builder.stats()
            assert stats["nodes"] > 0
            assert stats["edges"] >= 0

    def test_clear(self, sample_cpp_source):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(persist_path=f"{tmpdir}/graph.json")
            builder = GraphBuilder(store=store)
            builder.add_file(sample_cpp_source, "types.hpp", "proj")
            builder.clear()
            stats = builder.stats()
            assert stats["nodes"] == 0


# ------------------------------------------------------------------
# Graph Query tests
# ------------------------------------------------------------------

class TestGraphQuery:
    def test_find_callers(self, empty_store):
        # main -> foo (main calls foo)
        main_id = EntityId.from_raw("p", "a.cpp", "function", "main")
        foo_id = EntityId.from_raw("p", "b.cpp", "function", "foo")
        empty_store.add_entity(Entity(id=main_id, name="main", entity_type="function", file_path="a.cpp", line_start=1, line_end=5))
        empty_store.add_entity(Entity(id=foo_id, name="foo", entity_type="function", file_path="b.cpp", line_start=1, line_end=5))
        empty_store.add_relation(Relation(source=main_id, target=foo_id, relation_type="CALLS"))

        query = GraphQuery(empty_store)
        callers = query.find_callers("foo")
        assert len(callers) == 1
        assert callers[0].name == "main"

    def test_file_dependencies(self, empty_store):
        a_id = EntityId.from_raw("p", "a.cpp", "file", "a.cpp")
        b_id = EntityId.from_raw("p", "b.hpp", "file", "b.hpp")
        empty_store.add_entity(Entity(id=a_id, name="a.cpp", entity_type="file", file_path="a.cpp", line_start=1, line_end=1))
        empty_store.add_entity(Entity(id=b_id, name="b.hpp", entity_type="file", file_path="b.hpp", line_start=1, line_end=1))
        empty_store.add_relation(Relation(source=a_id, target=b_id, relation_type="INCLUDES"))

        query = GraphQuery(empty_store)
        deps = query.file_dependencies("a.cpp")
        assert len(deps) == 1
        assert deps[0].name == "b.hpp"

    def test_impact_analysis(self, empty_store):
        a = EntityId.from_raw("p", "a.cpp", "function", "main")
        b = EntityId.from_raw("p", "b.cpp", "function", "foo")
        c = EntityId.from_raw("p", "c.cpp", "class", "Bar")
        empty_store.add_entity(Entity(id=a, name="main", entity_type="function", file_path="a.cpp", line_start=1, line_end=5))
        empty_store.add_entity(Entity(id=b, name="foo", entity_type="function", file_path="b.cpp", line_start=1, line_end=5))
        empty_store.add_entity(Entity(id=c, name="Bar", entity_type="class", file_path="c.cpp", line_start=1, line_end=5))
        empty_store.add_relation(Relation(source=a, target=b, relation_type="CALLS"))
        empty_store.add_relation(Relation(source=b, target=c, relation_type="REFERENCES"))

        query = GraphQuery(empty_store)
        impact = query.impact_analysis("foo")
        assert len(impact["direct_callers"]) == 1
        assert impact["direct_callers"][0].name == "main"
        assert len(impact["transitive_deps"]) >= 1
