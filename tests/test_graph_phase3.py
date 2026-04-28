"""Phase 3 tests for the Code Knowledge Graph module."""

import tempfile
from pathlib import Path

import pytest

from sec_rag.graph import (
    CppAstExtractor,
    GraphBuilder,
    GraphQuery,
    GraphStore,
    NetworkXBackend,
)
from sec_rag.graph.backend import GraphBackend
from sec_rag.graph.models import Entity, EntityId


# ------------------------------------------------------------------
# Backend abstraction tests
# ------------------------------------------------------------------

class TestBackendAbstraction:
    def test_networkx_backend_is_backend(self):
        backend = NetworkXBackend()
        assert isinstance(backend, GraphBackend)

    def test_networkx_backend_remove_entity(self):
        backend = NetworkXBackend()
        eid = EntityId.from_raw("p", "f.cpp", "class", "Foo")
        backend.add_entity(Entity(id=eid, name="Foo", entity_type="class", file_path="f.cpp", line_start=1, line_end=10))
        assert backend.get_entity(eid) is not None
        backend.remove_entity(eid)
        assert backend.get_entity(eid) is None

    def test_networkx_backend_remove_relation(self):
        backend = NetworkXBackend()
        src = EntityId.from_raw("p", "a.cpp", "function", "main")
        dst = EntityId.from_raw("p", "b.cpp", "function", "foo")
        from sec_rag.graph.models import Relation
        backend.add_relation(Relation(source=src, target=dst, relation_type="CALLS"))
        assert len(backend.get_neighbors(src, relation_types=["CALLS"], direction="out")) == 1
        backend.remove_relation(src, dst, relation_type="CALLS")
        assert len(backend.get_neighbors(src, relation_types=["CALLS"], direction="out")) == 0

    def test_graph_store_pluggable_backend(self):
        backend = NetworkXBackend()
        store = GraphStore(backend=backend)
        eid = EntityId.from_raw("p", "f.cpp", "class", "Foo")
        store.add_entity(Entity(id=eid, name="Foo", entity_type="class", file_path="f.cpp", line_start=1, line_end=10))
        assert store.get_entity(eid) is not None


# ------------------------------------------------------------------
# Incremental update tests
# ------------------------------------------------------------------

class TestIncrementalUpdate:
    def test_update_file_removes_old_entities(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(backend=NetworkXBackend(persist_path=f"{tmpdir}/g.json"))
            builder = GraphBuilder(store=store)

            # Initial file
            builder.add_file(
                "class Foo { void bar() {} };\n",
                "foo.hpp",
                "proj",
            )
            stats_before = builder.stats()
            assert stats_before["nodes"] == 3  # file, class, function

            # Update file: Foo is removed, Baz is added
            builder.update_file(
                "class Baz { void qux() {} };\n",
                "foo.hpp",
                "proj",
            )
            stats_after = builder.stats()
            assert stats_after["nodes"] == 3  # still 3, but different ones

            # Old entities should be gone
            query = GraphQuery(store)
            old = query._find_entity("Foo", "class", "proj")
            assert old is None
            new = query._find_entity("Baz", "class", "proj")
            assert new is not None

    def test_update_file_preserves_other_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(backend=NetworkXBackend(persist_path=f"{tmpdir}/g.json"))
            builder = GraphBuilder(store=store)

            builder.add_file("class A {};\n", "a.hpp", "proj")
            builder.add_file("class B {};\n", "b.hpp", "proj")

            assert builder.stats()["nodes"] == 4  # 2 files + 2 classes

            # Update only a.hpp
            builder.update_file("class A2 {};\n", "a.hpp", "proj")

            stats = builder.stats()
            assert stats["nodes"] == 4  # still 4

            query = GraphQuery(store)
            assert query._find_entity("A2", "class", "proj") is not None
            assert query._find_entity("B", "class", "proj") is not None
            assert query._find_entity("A", "class", "proj") is None

    def test_update_file_cleans_cross_file_edges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(backend=NetworkXBackend(persist_path=f"{tmpdir}/g.json"))
            builder = GraphBuilder(store=store)

            # File 1 defines Foo
            builder.add_file("class Foo {};\n", "foo.hpp", "proj")
            # File 2 references Foo
            builder.add_file("class Bar : public Foo {};\n", "bar.hpp", "proj")

            query = GraphQuery(store)
            foo = query._find_entity("Foo", "class", "proj")
            bar = query._find_entity("Bar", "class", "proj")

            # Should have INHERITS_FROM edge
            edges_before = store.edges()
            inherits = [e for e in edges_before if e[2].get("relation_type") == "INHERITS_FROM"]
            assert len(inherits) == 1

            # Update foo.hpp: remove Foo, add Foo2
            builder.update_file("class Foo2 {};\n", "foo.hpp", "proj")

            # Old Foo edge should be gone (Foo node deleted, edge auto-removed)
            edges_after = store.edges()
            inherits_after = [e for e in edges_after if e[2].get("relation_type") == "INHERITS_FROM"]
            assert len(inherits_after) == 0


# ------------------------------------------------------------------
# Enhanced export tests
# ------------------------------------------------------------------

class TestEnhancedExport:
    def test_dot_export_with_clustering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(backend=NetworkXBackend(persist_path=f"{tmpdir}/g.json"))
            builder = GraphBuilder(store=store)
            builder.add_file(
                '''namespace ns {
class Foo {};
class Bar {};
} // namespace ns
''',
                "test.hpp",
                "proj",
            )

            output = f"{tmpdir}/out.dot"
            from sec_rag.graph.exporter import export_to_dot
            export_to_dot(store, output, cluster_by_namespace=True)

            content = Path(output).read_text()
            assert "subgraph cluster_" in content
            assert 'label="ns"' in content
            assert "Foo" in content
            assert "Bar" in content

    def test_dot_export_without_clustering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(backend=NetworkXBackend(persist_path=f"{tmpdir}/g.json"))
            builder = GraphBuilder(store=store)
            builder.add_file("class Foo {};\n", "test.hpp", "proj")

            output = f"{tmpdir}/out.dot"
            from sec_rag.graph.exporter import export_to_dot
            export_to_dot(store, output, cluster_by_namespace=False)

            content = Path(output).read_text()
            assert "subgraph cluster_" not in content
            assert "Foo" in content

    def test_json_export_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(backend=NetworkXBackend(persist_path=f"{tmpdir}/g.json"))
            builder = GraphBuilder(store=store)
            builder.add_file("class Foo {};\n", "test.hpp", "proj")

            output = f"{tmpdir}/out.json"
            from sec_rag.graph.exporter import export_to_json
            export_to_json(store, output)

            data = json.loads(Path(output).read_text())
            assert "nodes" in data
            assert "edges" in data
            assert any(n.get("name") == "Foo" for n in data["nodes"])


# Need json import
import json
