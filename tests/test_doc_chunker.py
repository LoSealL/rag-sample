# tests/test_doc_chunker.py
"""Tests for the document chunker using unstructured.io."""

from sec_rag.chunkers.doc_chunker import DocChunker


class TestMarkdownExtraction:
    """Tests for Markdown document chunking."""

    def test_markdown_section_extraction(self, sample_markdown_file):
        """Verify a Markdown file with 3 headings returns 3 section chunks."""
        chunker = DocChunker()
        chunks = chunker.chunk_file(sample_markdown_file)

        # Should get at least 3 section chunks (one per heading)
        assert len(chunks) >= 3, f"Expected >= 3 chunks, got {len(chunks)}"

        # Check language
        for chunk in chunks:
            assert chunk.metadata.language == "markdown"
            assert chunk.metadata.file_path == sample_markdown_file
            assert chunk.metadata.chunk_id  # not empty

        # At least one chunk should have "Authentication" in the name
        names = {c.metadata.name for c in chunks}
        assert any("Authentication" in n for n in names)

    def test_empty_markdown(self, empty_markdown_file):
        """Verify an empty Markdown file returns empty list."""
        chunker = DocChunker()
        chunks = chunker.chunk_file(empty_markdown_file)
        assert chunks == []
