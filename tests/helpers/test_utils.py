"""Test helper utilities and common functions."""

import json
from pathlib import Path
from typing import Any

import numpy as np


class TestDataBuilder:
    """Builder class for creating test data."""

    @staticmethod
    def create_document(
        doc_id: str = "test_doc",
        text: str = "Sample document text",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a test document."""
        return {
            "id": doc_id,
            "text": text,
            "metadata": metadata or {"source": "test"},
        }

    @staticmethod
    def create_documents(count: int = 3) -> list[dict[str, Any]]:
        """Create multiple test documents."""
        return [
            TestDataBuilder.create_document(
                doc_id=f"doc_{i}",
                text=f"This is test document number {i}",
                metadata={"source": "test", "index": i},
            )
            for i in range(count)
        ]

    @staticmethod
    def create_embedding(dimension: int = 384) -> list[float]:
        """Create a test embedding vector."""
        return np.random.rand(dimension).tolist()

    @staticmethod
    def create_search_result(
        doc_id: str = "doc1",
        score: float = 0.95,
        text: str = "Sample text",
    ) -> dict[str, Any]:
        """Create a test search result."""
        return {
            "id": doc_id,
            "score": score,
            "text": text,
            "metadata": {"source": "test"},
        }


class MockResponseBuilder:
    """Builder class for creating mock responses."""

    @staticmethod
    def llm_response(
        content: str = "Test response",
        role: str = "assistant",
        finish_reason: str = "stop",
    ) -> dict[str, Any]:
        """Create a mock LLM response."""
        return {
            "content": content,
            "role": role,
            "finish_reason": finish_reason,
        }

    @staticmethod
    def api_response(
        data: Any = None,
        status: str = "success",
        error: str | None = None,
    ) -> dict[str, Any]:
        """Create a mock API response."""
        response = {"status": status}
        if data is not None:
            response["data"] = data
        if error:
            response["error"] = error
        return response


class FileTestHelper:
    """Helper for file-based testing."""

    @staticmethod
    def create_temp_file(
        tmp_path: Path,
        filename: str,
        content: str,
    ) -> Path:
        """Create a temporary file for testing."""
        file_path = tmp_path / filename
        file_path.write_text(content)
        return file_path

    @staticmethod
    def create_temp_json(
        tmp_path: Path,
        filename: str,
        data: dict[str, Any],
    ) -> Path:
        """Create a temporary JSON file for testing."""
        file_path = tmp_path / filename
        file_path.write_text(json.dumps(data, indent=2))
        return file_path

    @staticmethod
    def load_fixture(fixture_name: str) -> Any:
        """Load data from fixtures directory."""
        fixture_path = Path(__file__).parent.parent / "fixtures" / fixture_name
        if fixture_path.suffix == ".json":
            return json.loads(fixture_path.read_text())
        return fixture_path.read_text()


class AssertionHelper:
    """Helper for common assertions."""

    @staticmethod
    def assert_valid_embedding(embedding: list[float], expected_dim: int = 384):
        """Assert that an embedding is valid."""
        assert isinstance(embedding, list), "Embedding must be a list"
        assert len(embedding) == expected_dim, f"Expected {expected_dim} dimensions"
        assert all(isinstance(x, (int, float)) for x in embedding), "All values must be numeric"

    @staticmethod
    def assert_valid_document(doc: dict[str, Any]):
        """Assert that a document has required fields."""
        assert "id" in doc, "Document must have an id"
        assert "text" in doc, "Document must have text"
        assert isinstance(doc.get("metadata", {}), dict), "Metadata must be a dict"

    @staticmethod
    def assert_search_results(results: list[dict[str, Any]], min_score: float = 0.0):
        """Assert that search results are valid."""
        assert isinstance(results, list), "Results must be a list"
        for result in results:
            assert "id" in result, "Result must have an id"
            assert "score" in result, "Result must have a score"
            assert result["score"] >= min_score, f"Score must be >= {min_score}"


def compare_floats(a: float, b: float, tolerance: float = 1e-6) -> bool:
    """Compare two floats with tolerance."""
    return abs(a - b) < tolerance


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return " ".join(text.lower().split())
