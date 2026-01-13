"""Pytest configuration for ingestion tests."""

import pytest


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a sample text file for testing."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is sample text for ingestion testing.")
    return file_path


@pytest.fixture
def sample_markdown_file(tmp_path):
    """Create a sample markdown file for testing."""
    file_path = tmp_path / "sample.md"
    content = """# Test Document

This is a test document with markdown content.

## Section 1
Some content here.
"""
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_html_file(tmp_path):
    """Create a sample HTML file for testing."""
    file_path = tmp_path / "sample.html"
    content = """<html>
<head><title>Test</title></head>
<body><p>Test content</p></body>
</html>"""
    file_path.write_text(content)
    return file_path


@pytest.fixture
def chunking_config():
    """Provide chunking configuration."""
    return {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "strategy": "recursive",
    }
