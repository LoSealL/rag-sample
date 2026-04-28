"""Phase 4 tests for the Code Knowledge Graph module."""

import tempfile
from pathlib import Path

import pytest

from sec_rag.graph import (
    GraphBuilder,
    GraphStore,
    NetworkXBackend,
    FileHashIndex,
)
from sec_rag.graph.models import Entity, EntityId


# ------------------------------------------------------------------
# File hash index tests
# ------------------------------------------------------------------

class TestFileHashIndex:
    def test_detect_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = FileHashIndex(persist_path=f"{tmpdir}/hashes.json")
            assert idx.is_changed("a.cpp", "hello") is True
            idx.update("a.cpp", "hello")
            assert idx.is_changed("a.cpp", "hello") is False
            assert idx.is_changed("a.cpp", "world") is True

    def test_get_stale_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = FileHashIndex(persist_path=f"{tmpdir}/hashes.json")
            idx.update("a.cpp", "content_a")
            idx.update("b.cpp", "content_b")
            stale = idx.get_stale_files(["a.cpp"])
            assert stale == ["b.cpp"]

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/hashes.json"
            idx = FileHashIndex(persist_path=path)
            idx.update("x.cpp", "code")
            idx.save()

            idx2 = FileHashIndex(persist_path=path)
            assert idx2.is_changed("x.cpp", "code") is False
            assert idx2.is_changed("x.cpp", "other") is True


# ------------------------------------------------------------------
# Incremental project build tests
# ------------------------------------------------------------------

class TestIncrementalProjectBuild:
    def test_first_build(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(backend=NetworkXBackend(persist_path=f"{tmpdir}/g.json"))
            hash_index = FileHashIndex(persist_path=f"{tmpdir}/hashes.json")
            builder = GraphBuilder(store=store, hash_index=hash_index)

            files = [
                ("a.hpp", "class A {};\n"),
                ("b.hpp", "class B {};\n"),
            ]
            stats = builder.build_project(files, "proj")
            assert stats["added"] == 2
            assert stats["skipped"] == 0
            assert store.stats()["nodes"] == 4  # 2 files + 2 classes

    def test_rebuild_no_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(backend=NetworkXBackend(persist_path=f"{tmpdir}/g.json"))
            hash_index = FileHashIndex(persist_path=f"{tmpdir}/hashes.json")
            builder = GraphBuilder(store=store, hash_index=hash_index)

            files = [("a.hpp", "class A {};\n")]
            builder.build_project(files, "proj")
            builder.save()

            # Rebuild with same content
            stats = builder.build_project(files, "proj")
            assert stats["skipped"] == 1
            assert stats["added"] == 0
            assert stats["updated"] == 0

    def test_update_changed_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(backend=NetworkXBackend(persist_path=f"{tmpdir}/g.json"))
            hash_index = FileHashIndex(persist_path=f"{tmpdir}/hashes.json")
            builder = GraphBuilder(store=store, hash_index=hash_index)

            files = [("a.hpp", "class A {};\n")]
            builder.build_project(files, "proj")
            builder.save()

            # Change content
            files = [("a.hpp", "class A2 {};\n")]
            stats = builder.build_project(files, "proj")
            assert stats["updated"] == 1
            assert stats["skipped"] == 0

            # Old entity gone, new entity present
            from sec_rag.graph.query import GraphQuery
            query = GraphQuery(store)
            assert query._find_entity("A", "class", "proj") is None
            assert query._find_entity("A2", "class", "proj") is not None

    def test_remove_deleted_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(backend=NetworkXBackend(persist_path=f"{tmpdir}/g.json"))
            hash_index = FileHashIndex(persist_path=f"{tmpdir}/hashes.json")
            builder = GraphBuilder(store=store, hash_index=hash_index)

            files = [
                ("a.hpp", "class A {};\n"),
                ("b.hpp", "class B {};\n"),
            ]
            builder.build_project(files, "proj")
            builder.save()

            # Delete a.hpp
            files = [("b.hpp", "class B {};\n")]
            stats = builder.build_project(files, "proj")
            assert stats["removed"] == 1

            from sec_rag.graph.query import GraphQuery
            query = GraphQuery(store)
            assert query._find_entity("A", "class", "proj") is None
            assert query._find_entity("B", "class", "proj") is not None


# ------------------------------------------------------------------
# HTML export tests
# ------------------------------------------------------------------

class TestHtmlExport:
    def test_html_export_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(backend=NetworkXBackend(persist_path=f"{tmpdir}/g.json"))
            builder = GraphBuilder(store=store)
            builder.add_file("class Foo {};\n", "test.hpp", "proj")

            output = f"{tmpdir}/graph.html"
            from sec_rag.graph.html_exporter import export_to_html
            export_to_html(store, output)

            content = Path(output).read_text()
            assert "<!DOCTYPE html>" in content
            assert "d3" in content.lower()
            assert "Foo" in content
            assert "class" in content

    def test_html_export_with_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphStore(backend=NetworkXBackend(persist_path=f"{tmpdir}/g.json"))
            builder = GraphBuilder(store=store)
            builder.add_file("class Foo {};\n", "test.hpp", "proj1")
            builder.add_file("class Bar {};\n", "test2.hpp", "proj2")

            output = f"{tmpdir}/graph.html"
            from sec_rag.graph.html_exporter import export_to_html
            export_to_html(store, output, project_id="proj1")

            content = Path(output).read_text()
            assert "Foo" in content
            assert "Bar" not in content
