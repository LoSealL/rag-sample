# tests/conftest.py
"""Pytest fixtures for rag-sample tests."""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hashlib

from sec_rag.index_store import IndexStore


def _dummy_embed(text: str) -> list[float]:
    """Deterministic 384-dim dummy embedding for tests (no ONNX model needed)."""
    h = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    return [(h % 100) / 100.0] * 384


@pytest.fixture
def dummy_embed():
    """Return a deterministic dummy embedding function for tests."""
    return _dummy_embed


@pytest.fixture
def temp_index_dir(tmp_path):
    """Provide a temporary directory for ChromaDB index."""
    index_dir = tmp_path / ".rag_index"
    index_dir.mkdir()
    yield str(index_dir)


@pytest.fixture
def index_store(temp_index_dir):
    """Provide a fresh IndexStore with temporary directory."""
    return IndexStore(persist_dir=temp_index_dir)


@pytest.fixture
def sample_python_file(tmp_path):
    """Provide a temporary Python file with 2 functions."""
    file_path = tmp_path / "sample.py"
    content = '''"""Sample Python module for testing."""

def authenticate_user(username: str, password: str) -> bool:
    """Authenticate a user with username and password.

    Args:
        username: The user's username.
        password: The user's password.

    Returns:
        True if authentication succeeds, False otherwise.
    """
    # In a real system, this would check against a database
    if username == "admin" and password == "secret":
        return True
    return False


def get_user_profile(user_id: int) -> dict:
    """Get a user's profile by their ID.

    Args:
        user_id: The user's unique identifier.

    Returns:
        A dict with user profile fields.
    """
    return {
        "id": user_id,
        "name": "Test User",
        "email": f"user{user_id}@example.com",
    }
'''
    file_path.write_text(content)
    return str(file_path)


@pytest.fixture
def sample_c_file(tmp_path):
    """Provide a temporary C file with 2 functions."""
    file_path = tmp_path / "sample.c"
    content = """#include <stdio.h>
#include <stdbool.h>

/**
 * Authenticate a user (C implementation)
 * @param username The username string
 * @param password The password string
 * @return true if authenticated, false otherwise
 */
bool authenticate_user(const char* username, const char* password) {
    // Simple hardcoded auth for demonstration
    if (strcmp(username, "admin") == 0 && strcmp(password, "secret") == 0) {
        return true;
    }
    return false;
}


/**
 * Get the current system timestamp
 * @return Unix timestamp as long
 */
long get_system_time(void) {
    return (long)time(NULL);
}
"""
    file_path.write_text(content)
    return str(file_path)


@pytest.fixture
def sample_markdown_file(tmp_path):
    """Provide a temporary Markdown file with 3 sections."""
    file_path = tmp_path / "sample.md"
    content = """# Authentication System

This document describes the authentication system.

## Overview

The authentication system verifies user credentials before granting access.

## User Authentication

The `authenticate_user` function checks username/password pairs.

## Error Handling

All authentication errors return false and log to stderr.
"""
    file_path.write_text(content)
    return str(file_path)


@pytest.fixture
def malformed_python_file(tmp_path):
    """Provide a malformed Python file (invalid syntax)."""
    file_path = tmp_path / "malformed.py"
    file_path.write_text("""def broken_function(
    # Missing closing parenthesis and body
    x: int
        return x + 1
""")
    return str(file_path)


@pytest.fixture
def docstring_only_python_file(tmp_path):
    """Provide a Python file with only a docstring, no functions."""
    file_path = tmp_path / "docstring_only.py"
    file_path.write_text('''"""This module contains only a docstring.

No functions or classes are defined here.
This is used to test module_docstring chunk extraction.
"""
''')
    return str(file_path)


@pytest.fixture
def empty_python_file(tmp_path):
    """Provide an empty Python file."""
    file_path = tmp_path / "empty.py"
    file_path.write_text("")
    return str(file_path)


@pytest.fixture
def empty_markdown_file(tmp_path):
    """Provide an empty Markdown file."""
    file_path = tmp_path / "empty.md"
    file_path.write_text("")
    return str(file_path)
