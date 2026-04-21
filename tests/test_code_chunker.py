# tests/test_code_chunker.py
"""Tests for the code chunker using tree-sitter."""

import pytest

from sec_rag.chunkers.code_chunker import CodeChunker


class TestPythonFunctionExtraction:
    """test_python_function_extraction — parses a real Python file with 2 functions."""

    def test_python_function_extraction(self, sample_python_file):
        """Verify a Python file with 2 functions returns 2 chunks."""
        chunker = CodeChunker()
        chunks = chunker.chunk_file(sample_python_file)

        # Should get 2 function chunks
        function_chunks = [c for c in chunks if c.metadata.chunk_type == "function"]
        assert len(function_chunks) == 2, (
            f"Expected 2 function chunks, got {len(function_chunks)}"
        )

        # Verify names
        names = {c.metadata.name for c in function_chunks}
        assert "authenticate_user" in names
        assert "get_user_profile" in names

        # Verify line ranges are positive integers
        for chunk in function_chunks:
            assert chunk.metadata.line_start > 0
            assert chunk.metadata.line_end >= chunk.metadata.line_start
            assert chunk.metadata.language == "python"
            assert chunk.metadata.chunk_id  # not empty

        # Verify chunk text contains the function body
        texts = {c.text for c in function_chunks}
        assert "def authenticate_user" in "".join(texts)
        assert "def get_user_profile" in "".join(texts)

    def test_parse_error_fallback(self, malformed_python_file):
        """Verify malformed Python falls back to naive chunking, no crash."""
        chunker = CodeChunker()
        # Should not raise — falls back gracefully
        chunks = chunker.chunk_file(malformed_python_file)
        # Returns chunks from naive fallback
        assert isinstance(chunks, list)

    def test_module_docstring_only(self, docstring_only_python_file):
        """Verify a file with only a docstring returns 1 module_docstring chunk."""
        chunker = CodeChunker()
        chunks = chunker.chunk_file(docstring_only_python_file)
        assert len(chunks) >= 1
        docstring_chunks = [
            c for c in chunks if c.metadata.chunk_type == "module_docstring"
        ]
        assert len(docstring_chunks) == 1

    def test_empty_file(self, empty_python_file):
        """Verify an empty Python file returns empty list."""
        chunker = CodeChunker()
        chunks = chunker.chunk_file(empty_python_file)
        assert chunks == []

    def test_file_not_found(self):
        """Verify FileNotFoundError is raised for missing files."""
        chunker = CodeChunker()
        with pytest.raises(FileNotFoundError):
            chunker.chunk_file("/nonexistent/file.py")
