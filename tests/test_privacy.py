"""Tests for privacy mode configuration and enforcement."""

import pytest

from sec_rag.chunkers.base_chunker import Chunk, ChunkMetadata
from sec_rag.privacy import PrivacyConfig, generate_privacy_summary
from sec_rag.project_processing import process_privacy_chunk, create_project_context


class TestPrivacyConfig:
    """Test privacy configuration matching."""

    def test_exact_path_match(self):
        cfg = PrivacyConfig(exact_paths=["src/secrets/api_keys.h"])
        assert cfg.is_private("src/secrets/api_keys.h")
        assert cfg.is_private("project/src/secrets/api_keys.h")
        assert not cfg.is_private("src/other/file.h")

    def test_directory_match(self):
        cfg = PrivacyConfig(directories=["src/crypto/"])
        assert cfg.is_private("src/crypto/aes.cpp")
        assert cfg.is_private("src/crypto/internal/keys.h")
        assert not cfg.is_private("src/utils/helper.cpp")

    def test_extension_match(self):
        cfg = PrivacyConfig(extensions=[".pem", ".key"])
        assert cfg.is_private("server.pem")
        assert cfg.is_private("private.key")
        assert not cfg.is_private("main.cpp")

    def test_regex_pattern_match(self):
        cfg = PrivacyConfig(patterns=[".*_secret\\.py$", ".*\\.token$"])
        assert cfg.is_private("config_secret.py")
        assert cfg.is_private("auth.token")
        assert not cfg.is_private("normal.py")

    def test_disabled_config(self):
        cfg = PrivacyConfig(
            exact_paths=["secret.h"],
            enabled=False,
        )
        assert not cfg.is_private("secret.h")

    def test_from_dict(self):
        data = {
            "exact_paths": ["a.h"],
            "directories": ["crypto/"],
            "patterns": [".*_secret.*"],
            "extensions": [".key"],
            "enabled": True,
        }
        cfg = PrivacyConfig.from_dict(data)
        assert cfg.is_private("a.h")
        assert cfg.is_private("crypto/cipher.cpp")
        assert cfg.is_private("my_secret.py")
        assert cfg.is_private("id.key")

    def test_cross_platform_paths(self):
        cfg = PrivacyConfig(directories=["src/crypto"])
        assert cfg.is_private("src/crypto/keys.h")
        assert cfg.is_private("src\\crypto\\keys.h")

    def test_summary(self):
        cfg = PrivacyConfig(
            exact_paths=["a.h"],
            directories=["crypto/"],
            patterns=[".*secret.*"],
            extensions=[".key"],
        )
        summary = cfg.summary()
        assert "exact=1" in summary
        assert "dirs=1" in summary
        assert "patterns=1" in summary
        assert "exts=1" in summary
        assert "total=4" in summary


class TestPrivacySummaryGeneration:
    """Test privacy-safe summary generation."""

    def test_generate_with_llm(self):
        class FakeLLM:
            def complete(self, prompt: str) -> str:
                return "This function handles authentication securely."

        llm = FakeLLM()
        summary = generate_privacy_summary(
            chunk_text="int auth(const char* pwd) { ... }",
            file_path="src/auth.cpp",
            llm=llm,
            declaration="int auth(const char* pwd);",
        )
        assert "authentication" in summary
        assert "src/auth.cpp" not in summary  # Summary should not leak path

    def test_generate_without_llm(self):
        summary = generate_privacy_summary(
            chunk_text="secret code",
            file_path="src/secret.cpp",
            llm=None,
        )
        assert "[PRIVACY-SAFE SUMMARY]" in summary
        assert "high-confidential" in summary

    def test_generate_llm_failure(self):
        class BadLLM:
            def complete(self, prompt: str) -> str:
                raise RuntimeError("API down")

        summary = generate_privacy_summary(
            chunk_text="secret",
            file_path="src/secret.cpp",
            llm=BadLLM(),
        )
        assert "[PRIVACY-SAFE SUMMARY]" in summary


class TestProcessPrivacyChunk:
    """Test privacy chunk processing."""

    def test_mark_private_and_summary(self):
        chunk = Chunk(
            metadata=ChunkMetadata(
                file_path="src/secret.cpp",
                language="cpp",
                chunk_type="function",
                name="get_api_key",
                chunk_id="test:1",
                line_start=10,
                line_end=20,
            ),
            text="const char* get_api_key() { return 'sk-12345'; }",
        )

        class FakeLLM:
            def complete(self, prompt: str) -> str:
                return "Retrieves sensitive API credentials."

        project = create_project_context("./test-project")
        result = process_privacy_chunk(chunk, FakeLLM(), project)

        assert result.metadata.is_private is True
        assert "credentials" in result.metadata.privacy_summary
        assert result.metadata.project_id == project.project_id

    def test_preserves_existing_metadata(self):
        chunk = Chunk(
            metadata=ChunkMetadata(
                file_path="src/secret.cpp",
                language="cpp",
                chunk_type="function",
                name="encrypt",
                chunk_id="test:2",
                line_start=1,
                line_end=5,
                sanitized=True,
                semantic_summary="Encrypts data",
            ),
            text="void encrypt() { ... }",
        )

        project = create_project_context("./proj")
        result = process_privacy_chunk(chunk, None, project)

        assert result.metadata.is_private is True
        assert result.metadata.sanitized is True
        assert result.metadata.semantic_summary == "Encrypts data"


class TestPrivacyPromptSubstitution:
    """Test that privacy chunks are substituted in prompts."""

    def test_private_chunk_substituted(self):
        from sec_rag.query_pipeline import QueryPipeline

        class FakeStore:
            def query(self, *args, **kwargs):
                return [
                    {
                        "chunk_id": "private:1",
                        "text": "SECRET_PASSWORD = '12345'",
                        "metadata": {
                            "file_path": "src/secrets.h",
                            "language": "c",
                            "chunk_type": "variable",
                            "name": "SECRET_PASSWORD",
                            "line_start": 1,
                            "line_end": 1,
                            "project_name": "test",
                            "is_private": True,
                            "privacy_summary": "Stores authentication credentials.",
                        },
                        "score": 0.9,
                    }
                ]

        class FakeEmbed:
            def encode(self, text: str):
                return [0.1] * 128

        captured_prompt = ""

        class FakeLLM:
            def complete(self, prompt: str) -> str:
                nonlocal captured_prompt
                captured_prompt = prompt
                return "answer"

        pipeline = QueryPipeline(FakeStore(), FakeEmbed(), FakeLLM())
        result = pipeline.run("what is the password?", top_k=1)

        # The prompt should NOT contain the raw secret
        assert "SECRET_PASSWORD" not in captured_prompt
        # It should contain the privacy summary instead
        assert "Stores authentication credentials" in captured_prompt
        assert "[PRIVACY-SAFE SUMMARY]" in captured_prompt
        assert "HIGH-CONFIDENTIAL" in captured_prompt

    def test_non_private_chunk_untouched(self):
        from sec_rag.query_pipeline import QueryPipeline

        class FakeStore:
            def query(self, *args, **kwargs):
                return [
                    {
                        "chunk_id": "public:1",
                        "text": "int add(int a, int b) { return a + b; }",
                        "metadata": {
                            "file_path": "src/math.cpp",
                            "language": "cpp",
                            "chunk_type": "function",
                            "name": "add",
                            "line_start": 1,
                            "line_end": 3,
                            "project_name": "test",
                            "is_private": False,
                            "privacy_summary": "",
                        },
                        "score": 0.9,
                    }
                ]

        class FakeEmbed:
            def encode(self, text: str):
                return [0.1] * 128

        class FakeLLM:
            def complete(self, prompt: str) -> str:
                return "answer"

        pipeline = QueryPipeline(FakeStore(), FakeEmbed(), FakeLLM())
        result = pipeline.run("how to add?", top_k=1)

        # Non-private chunk should have its original text
        assert "int add" in result.chunks[0]["text"]
