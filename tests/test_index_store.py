# tests/test_index_store.py
"""Tests for the ChromaDB index store."""

from sec_rag.chunkers.base_chunker import Chunk, ChunkMetadata


def _dummy_embed(text: str) -> list[float]:
    import hashlib
    h = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    return [(h % 100) / 100.0] * 384


def make_test_chunk(
    chunk_id: str,
    text: str,
    file_path: str = "/test/sample.py",
    language: str = "python",
    chunk_type: str = "function",
    name: str = "test_func",
    project_id: str = "project-a",
    project_name: str = "project-a",
) -> Chunk:
    """Helper to create a test Chunk."""
    metadata = ChunkMetadata(
        file_path=file_path,
        language=language,
        chunk_type=chunk_type,
        name=name,
        chunk_id=chunk_id,
        line_start=1,
        line_end=10,
        project_id=project_id,
        project_name=project_name,
        project_root=f"/projects/{project_name}",
    )
    return Chunk(metadata=metadata, text=text)


class TestAddAndQuery:
    """test_add_and_query — adds 3 chunks, queries for top-k=2."""

    def test_add_and_query(self, index_store):
        """Verify chunks can be added and retrieved by query."""
        chunks = [
            make_test_chunk(
                "test1", "def authenticate_user(username, password): return True"
            ),
            make_test_chunk(
                "test2", "def get_user_profile(user_id): return {'id': user_id}"
            ),
            make_test_chunk("test3", "def calculate(x, y): return x + y"),
        ]
        index_store.add_chunks("code_index", chunks, embed_fn=_dummy_embed)

        results = index_store.query(
            "code_index",
            "authenticate user function",
            _dummy_embed,
            top_k=2,
        )

        # Should get 2 results
        assert len(results) == 2
        # Results should have required fields
        for r in results:
            assert "chunk_id" in r
            assert "text" in r
            assert "metadata" in r
            assert "score" in r

    def test_metadata_filtering(self, index_store):
        """Verify language metadata filtering works."""
        chunks = [
            make_test_chunk(
                "py1",
                "def foo(): pass",
                file_path="/test/a.py",
                language="python",
            ),
            make_test_chunk(
                "c1",
                "int bar() { return 0; }",
                file_path="/test/b.c",
                language="c",
            ),
        ]
        index_store.add_chunks("code_index", chunks, embed_fn=_dummy_embed)

        # Filter to Python only
        py_results = index_store.query(
            "code_index",
            "any query",
            _dummy_embed,
            top_k=5,
            filter_language="python",
        )
        for r in py_results:
            assert r["metadata"]["language"] == "python"

        # Filter to C only
        c_results = index_store.query(
            "code_index",
            "any query",
            _dummy_embed,
            top_k=5,
            filter_language="c",
        )
        for r in c_results:
            assert r["metadata"]["language"] == "c"

    def test_project_filtering(self, index_store):
        """Verify project metadata filtering works."""
        chunks = [
            make_test_chunk(
                "p1",
                "def foo(): pass",
                project_id="project-a",
                project_name="project-a",
            ),
            make_test_chunk(
                "p2",
                "def bar(): pass",
                project_id="project-b",
                project_name="project-b",
            ),
        ]
        index_store.add_chunks("code_index", chunks, embed_fn=_dummy_embed)

        results = index_store.query(
            "code_index",
            "any query",
            _dummy_embed,
            top_k=5,
            filter_project_id="project-a",
        )

        assert results
        for result in results:
            assert result["metadata"]["project_id"] == "project-a"
