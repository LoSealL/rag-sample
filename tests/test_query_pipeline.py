# tests/test_query_pipeline.py
"""Tests for the query pipeline."""

from unittest.mock import MagicMock, patch

from sec_rag.query_pipeline import (
    SOURCE_EXPOSURE_BLOCK_MESSAGE,
    LocalEmbeddingClient,
    QueryPipeline,
    QueryResult,
)


class TestFullPipelineMockedLlm:
    """test_full_pipeline_mocked_llm — mocked LLM, verifies prompt and response."""

    def test_full_pipeline_mocked_llm(self, index_store):
        """Verify pipeline prompt construction and response formatting."""
        # Skip if OPENAI_API_KEY is not set (we'll mock the LLM)

        # Add some chunks to the index
        from sec_rag.chunkers.base_chunker import Chunk, ChunkMetadata

        metadata = ChunkMetadata(
            file_path="/test/sample.py",
            language="python",
            chunk_type="function",
            name="authenticate_user",
            chunk_id="py:test:func:authenticate_user:0",
            line_start=1,
            line_end=10,
        )
        chunk = Chunk(
            metadata=metadata,
            text="def authenticate_user(username, password): return True",
        )

        def dummy_embed(text: str) -> list[float]:
            import hashlib
            h = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            return [(h % 100) / 100.0] * 384

        index_store.add_chunks("rag_index", [chunk], embed_fn=dummy_embed)

        # Create a mock embedding model
        mock_model = MagicMock()
        mock_model.encode.return_value.tolist.return_value = dummy_embed("query")

        # Create a mock LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.__str__ = MagicMock(return_value="Mocked LLM response")
        mock_llm.complete.return_value = mock_response

        # Build pipeline
        pipeline = QueryPipeline(index_store, mock_model, mock_llm)

        # Run query
        result = pipeline.run("how does authenticate work?", top_k=1)

        # Verify LLM.complete was called
        assert mock_llm.complete.called, "LLM.complete should have been called"

        # Get the prompt passed to the LLM
        call_args = mock_llm.complete.call_args
        prompt = call_args[0][0] if call_args[0] else ""

        # Verify prompt contains the chunk text
        assert "authenticate_user" in prompt or "def authenticate" in prompt
        # Verify prompt has the expected structure
        assert "Context:" in prompt
        assert "User:" in prompt or "User:" in prompt

        # Verify QueryResult
        assert isinstance(result, QueryResult)
        assert result.answer == "Mocked LLM response"
        assert len(result.chunks) == 1
        assert result.total_prompt_tokens > 0

    def test_build_prompt_format(self, index_store):
        """Verify the prompt template is formatted correctly."""
        mock_model = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [0.1] * 384
        mock_llm = MagicMock()

        pipeline = QueryPipeline(index_store, mock_model, mock_llm)

        chunks = [
            {
                "chunk_id": "py:test:func:foo:0",
                "text": "def foo(): pass",
                "metadata": {
                    "file_path": "/test/sample.py",
                    "language": "python",
                    "chunk_type": "function",
                    "name": "foo",
                    "line_start": 1,
                    "line_end": 2,
                },
                "score": 0.1,
            }
        ]

        prompt = pipeline._build_prompt(chunks, "what does foo do?")

        # Check structure
        assert "System:" in prompt
        assert "Context:" in prompt
        assert "/test/sample.py" in prompt
        assert "lines 1-2" in prompt
        assert "type=function" in prompt
        assert "def foo(): pass" in prompt
        assert "what does foo do?" in prompt

    def test_project_summary_guides_followup_queries(self):
        """Verify project summaries are queried before code/doc collections."""
        mock_model = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [0.1] * 384

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.__str__ = MagicMock(return_value="Project-scoped answer")
        mock_llm.complete.return_value = mock_response

        mock_store = MagicMock()
        mock_store.query.side_effect = [
            [
                {
                    "chunk_id": "project:alpha:summary",
                    "text": "Alpha project summary",
                    "metadata": {"project_id": "alpha", "project_name": "alpha"},
                    "score": 0.1,
                }
            ],
            [
                {
                    "chunk_id": "alpha:code:1",
                    "text": "summary: handles authentication state transitions",
                    "metadata": {
                        "project_id": "alpha",
                        "project_name": "alpha",
                        "file_path": "/alpha/auth.cpp",
                        "language": "cpp",
                        "chunk_type": "function",
                        "name": "authenticate",
                        "line_start": 10,
                        "line_end": 42,
                    },
                    "score": 0.2,
                }
            ],
            [],
        ]

        pipeline = QueryPipeline(mock_store, mock_model, mock_llm)
        result = pipeline.run("how does auth work?", top_k=2)

        assert result.answer == "Project-scoped answer"
        assert mock_store.query.call_args_list[0].args[0] == "project_index"
        assert mock_store.query.call_args_list[1].kwargs["filter_project_id"] == "alpha"

    def test_blocks_verbatim_code_output(self, index_store):
        """Pipeline should block responses that look like raw source dumps."""
        from sec_rag.chunkers.base_chunker import Chunk, ChunkMetadata

        metadata = ChunkMetadata(
            file_path="/test/sample.cpp",
            language="cpp",
            chunk_type="function",
            name="foo",
            chunk_id="cpp:test:func:foo:0",
            line_start=1,
            line_end=8,
        )
        chunk = Chunk(
            metadata=metadata,
            text="int foo() { return 42; }",
        )

        def dummy_embed(text: str) -> list[float]:
            import hashlib

            h = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            return [(h % 100) / 100.0] * 384

        index_store.add_chunks("rag_index", [chunk], embed_fn=dummy_embed)

        mock_model = MagicMock()
        mock_model.encode.return_value.tolist.return_value = dummy_embed("query")

        mock_llm = MagicMock()
        mock_llm.complete.return_value = """```cpp
int foo() {
  return 42;
}
```"""

        pipeline = QueryPipeline(index_store, mock_model, mock_llm)
        result = pipeline.run("show me the exact source", top_k=1)

        assert result.answer == SOURCE_EXPOSURE_BLOCK_MESSAGE

    def test_blocks_single_line_hdl_or_code_snippet(self, index_store):
        """Block short single-line code snippets without fenced blocks."""
        from sec_rag.chunkers.base_chunker import Chunk, ChunkMetadata

        metadata = ChunkMetadata(
            file_path="/test/sample.v",
            language="verilog",
            chunk_type="module",
            name="m",
            chunk_id="verilog:test:module:m:0",
            line_start=1,
            line_end=3,
        )
        chunk = Chunk(metadata=metadata, text="module m(input a); endmodule")

        def dummy_embed(text: str) -> list[float]:
            import hashlib

            h = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            return [(h % 100) / 100.0] * 384

        index_store.add_chunks("rag_index", [chunk], embed_fn=dummy_embed)

        mock_model = MagicMock()
        mock_model.encode.return_value.tolist.return_value = dummy_embed("query")

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "module m(input a); always @(*) x=a; endmodule"

        pipeline = QueryPipeline(index_store, mock_model, mock_llm)
        result = pipeline.run("print the module source", top_k=1)

        assert result.answer == SOURCE_EXPOSURE_BLOCK_MESSAGE


def test_embedding_fallback_on_too_large_input():
    """LocalEmbeddingClient should split and average when server rejects long input."""
    client = LocalEmbeddingClient("http://localhost:8081", "dummy-model")
    long_text = "A" * 3000

    responses = [
        (
            500,
            {
                "error": {
                    "message": "input (595 tokens) is too large to process. "
                    "increase the physical batch size (current batch size: 512)"
                }
            },
            "",
        ),
        (200, {"data": [{"embedding": [1.0, 3.0]}]}, ""),
        (200, {"data": [{"embedding": [3.0, 5.0]}]}, ""),
    ]

    with (
        patch("sec_rag.query_pipeline._request_json", side_effect=responses),
        patch.object(
            LocalEmbeddingClient,
            "_split_text_for_embedding",
            return_value=["chunk-1", "chunk-2"],
        ),
    ):
        result = client.encode(long_text)

    assert result == [2.0, 4.0]
