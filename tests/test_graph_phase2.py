"""Phase 2 tests for the Code Knowledge Graph module."""

import tempfile
from pathlib import Path

import pytest

from sec_rag.graph import (
    CppAstExtractor,
    GraphBuilder,
    GraphQuery,
    GraphStore,
    SymbolTable,
)
from sec_rag.graph.models import Entity, EntityId, Relation


# ------------------------------------------------------------------
# Phase 2: deep semantic relation extraction
# ------------------------------------------------------------------

class TestDeepRelations:
    def test_calls_relation(self):
        source = '''
namespace rag {

float helper(float x) { return x * 2; }

float compute(float x) {
    return helper(x) + 1.0f;
}

} // namespace rag
'''
        extractor = CppAstExtractor()
        entities, relations = extractor.extract(source, "test.cpp", "proj")

        calls = [r for r in relations if r.relation_type == "CALLS"]
        assert len(calls) >= 1
        # compute should call helper
        compute_calls = [r for r in calls if r.source.name.endswith("compute")]
        assert len(compute_calls) >= 1
        assert compute_calls[0].target.name.endswith("helper")

    def test_inherits_from_relation(self):
        source = '''
class Base {};

class Derived : public Base {};
'''
        extractor = CppAstExtractor()
        entities, relations = extractor.extract(source, "test.cpp", "proj")

        inherits = [r for r in relations if r.relation_type == "INHERITS_FROM"]
        assert len(inherits) == 1
        assert inherits[0].source.name == "Derived"
        assert inherits[0].target.name == "Base"

    def test_references_from_function_params(self):
        source = '''
struct Point { float x, y; };

float distance(Point a, Point b) {
    return 0.0f;
}
'''
        extractor = CppAstExtractor()
        entities, relations = extractor.extract(source, "test.cpp", "proj")

        refs = [r for r in relations if r.relation_type == "REFERENCES"]
        # distance should reference Point
        distance_refs = [r for r in refs if r.source.name.endswith("distance")]
        assert any(r.target.name == "Point" for r in distance_refs)

    def test_member_references(self):
        source = '''
struct Config { int value; };

class Engine {
    Config cfg_;
public:
    void start() {}
};
'''
        extractor = CppAstExtractor()
        entities, relations = extractor.extract(source, "test.cpp", "proj")

        refs = [r for r in relations if r.relation_type == "REFERENCES"]
        engine_refs = [r for r in refs if r.source.name == "Engine"]
        assert any(r.target.name == "Config" for r in engine_refs)


# ------------------------------------------------------------------
# Symbol table tests
# ------------------------------------------------------------------

class TestSymbolTable:
    def test_lookup_unique(self):
        st = SymbolTable()
        eid = EntityId.from_raw("p", "f.cpp", "function", "foo")
        st.register(Entity(id=eid, name="foo", entity_type="function", file_path="f.cpp", line_start=1, line_end=1))
        assert st.lookup_unique("foo") == eid

    def test_lookup_ambiguous(self):
        st = SymbolTable()
        st.register(Entity(id=EntityId.from_raw("p", "a.cpp", "function", "foo"), name="foo", entity_type="function", file_path="a.cpp", line_start=1, line_end=1))
        st.register(Entity(id=EntityId.from_raw("p", "b.cpp", "function", "foo"), name="foo", entity_type="function", file_path="b.cpp", line_start=1, line_end=1))
        assert st.lookup_unique("foo") is None  # ambiguous
        assert len(st.lookup("foo")) == 2


# ------------------------------------------------------------------
# Cross-file resolution tests
# ------------------------------------------------------------------

class TestCrossFileResolution:
    def test_resolve_cross_file_calls(self):
        """Simulate two files where main.cpp calls foo() defined in foo.cpp."""
        st = SymbolTable()
        extractor = CppAstExtractor(symbol_table=st)

        # File 1: foo.cpp
        source1 = '''
namespace rag {
float foo(float x) { return x * 2; }
} // namespace rag
'''
        entities1, relations1 = extractor.extract(source1, "foo.cpp", "proj")
        for e in entities1:
            st.register(e)

        # File 2: main.cpp
        source2 = '''
#include "foo.hpp"

float main() {
    return foo(42.0f);
}
'''
        entities2, relations2 = extractor.extract(source2, "main.cpp", "proj")
        for e in entities2:
            st.register(e)

        # Now main should have a CALLS relation to foo
        calls = [r for r in relations2 if r.relation_type == "CALLS"]
        main_calls = [r for r in calls if r.source.name.endswith("main")]
        assert len(main_calls) >= 1
        # The target should be resolved to foo.cpp's foo
        assert "foo" in main_calls[0].target.name

    def test_builder_resolve_pending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(persist_path=f"{tmpdir}/graph.json")
            builder = GraphBuilder(store=store)

            # Add file with a function
            builder.add_file(
                "float foo() { return 1.0f; }\n",
                "foo.cpp",
                "proj",
            )

            # Manually add a pending relation (simulating LLM extractor output)
            pending = Relation(
                source=EntityId.from_raw("proj", "main.cpp", "function", "main"),
                target=EntityId.from_raw("proj", "", "function", "foo"),  # unresolved
                relation_type="CALLS",
            )
            builder._pending_relations.append(pending)

            # Resolve
            builder.resolve_cross_file_relations()

            # Check that the relation was resolved
            neighbors = store.get_neighbors(
                EntityId.from_raw("proj", "main.cpp", "function", "main"),
                relation_types=["CALLS"],
                direction="out",
            )
            assert len(neighbors) == 1
            assert neighbors[0][0].name == "foo"


# ------------------------------------------------------------------
# Hybrid pipeline tests (smoke tests with mocks)
# ------------------------------------------------------------------

class TestHybridPipeline:
    def test_hybrid_retrieve_without_graph_store(self):
        """When no graph store is given, hybrid should degrade to vector-only."""
        from sec_rag.hybrid_pipeline import HybridQueryPipeline

        # Minimal mock objects
        class MockIndexStore:
            def query(self, *a, **k):
                return []

        class MockEmbedder:
            def encode(self, text):
                import numpy as np
                return np.zeros(384)

        class MockLLM:
            def complete(self, prompt):
                return "mock answer"

        pipeline = HybridQueryPipeline(
            index_store=MockIndexStore(),
            embedding_model=MockEmbedder(),
            llm=MockLLM(),
            graph_store=None,
        )

        result = pipeline.hybrid_retrieve("test query", top_k=3)
        assert isinstance(result.chunks, list)

    def test_merge_chunks(self):
        from sec_rag.hybrid_pipeline import HybridQueryPipeline

        class MockIndexStore:
            def query(self, *a, **k):
                return []

        pipeline = HybridQueryPipeline(
            index_store=MockIndexStore(),
            embedding_model=None,
            llm=None,
            graph_store=None,
        )

        vector = [{"id": "a"}, {"id": "b"}]
        graph = [{"id": "b"}, {"id": "c"}]
        merged = pipeline._merge_chunks(vector, graph)
        ids = [c["id"] for c in merged]
        assert ids == ["a", "b", "c"]
