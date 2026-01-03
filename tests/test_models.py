"""
Unit tests for Pydantic models.
"""

import pytest
from pydantic import ValidationError
from src.models.rag_request import GenerateRequest, GenerateResponse


class TestGenerateRequest:
    """Test GenerateRequest model."""

    def test_valid_request_minimal(self):
        """Test valid request with minimal fields."""
        request = GenerateRequest(prompt="What is RAG?")
        
        assert request.prompt == "What is RAG?"
        assert request.top_k == 5  # default
        assert request.max_length is None
        assert request.metadata_filters is None

    def test_valid_request_all_fields(self):
        """Test valid request with all fields."""
        request = GenerateRequest(
            prompt="Test prompt",
            top_k=10,
            max_length=256,
            metadata_filters={"source": "doc1"}
        )
        
        assert request.prompt == "Test prompt"
        assert request.top_k == 10
        assert request.max_length == 256
        assert request.metadata_filters == {"source": "doc1"}

    def test_missing_required_prompt(self):
        """Test that missing prompt raises validation error."""
        with pytest.raises(ValidationError):
            GenerateRequest()

    def test_default_top_k(self):
        """Test default top_k value."""
        request = GenerateRequest(prompt="test")
        assert request.top_k == 5

    def test_custom_top_k(self):
        """Test custom top_k value."""
        request = GenerateRequest(prompt="test", top_k=15)
        assert request.top_k == 15

    def test_zero_top_k(self):
        """Test zero top_k value."""
        request = GenerateRequest(prompt="test", top_k=0)
        assert request.top_k == 0

    def test_negative_top_k(self):
        """Test negative top_k value (allowed by model)."""
        request = GenerateRequest(prompt="test", top_k=-1)
        assert request.top_k == -1

    def test_empty_prompt(self):
        """Test empty prompt string."""
        request = GenerateRequest(prompt="")
        assert request.prompt == ""

    def test_whitespace_prompt(self):
        """Test whitespace-only prompt."""
        request = GenerateRequest(prompt="   ")
        assert request.prompt == "   "

    def test_unicode_prompt(self):
        """Test unicode in prompt."""
        request = GenerateRequest(prompt="Hello 世界 🌍")
        assert request.prompt == "Hello 世界 🌍"

    def test_multiline_prompt(self):
        """Test multiline prompt."""
        prompt = "Line 1\nLine 2\nLine 3"
        request = GenerateRequest(prompt=prompt)
        assert request.prompt == prompt

    def test_metadata_filters_empty_dict(self):
        """Test empty metadata filters."""
        request = GenerateRequest(prompt="test", metadata_filters={})
        assert request.metadata_filters == {}

    def test_metadata_filters_complex(self):
        """Test complex metadata filters."""
        filters = {
            "source": "doc1",
            "type": "research",
            "tags": ["ml", "nlp"]
        }
        request = GenerateRequest(prompt="test", metadata_filters=filters)
        assert request.metadata_filters == filters

    def test_max_length_zero(self):
        """Test max_length of zero."""
        request = GenerateRequest(prompt="test", max_length=0)
        assert request.max_length == 0

    def test_max_length_large(self):
        """Test large max_length value."""
        request = GenerateRequest(prompt="test", max_length=10000)
        assert request.max_length == 10000

    def test_json_serialization(self):
        """Test model can be serialized to JSON."""
        request = GenerateRequest(
            prompt="test",
            top_k=3,
            max_length=100,
            metadata_filters={"key": "value"}
        )
        json_data = request.model_dump()
        
        assert json_data["prompt"] == "test"
        assert json_data["top_k"] == 3
        assert json_data["max_length"] == 100
        assert json_data["metadata_filters"] == {"key": "value"}

    def test_from_dict(self):
        """Test creating model from dictionary."""
        data = {
            "prompt": "test prompt",
            "top_k": 7,
            "max_length": 150
        }
        request = GenerateRequest(**data)
        
        assert request.prompt == "test prompt"
        assert request.top_k == 7
        assert request.max_length == 150


class TestGenerateResponse:
    """Test GenerateResponse model."""

    def test_valid_response_minimal(self):
        """Test valid response with minimal fields."""
        response = GenerateResponse(
            prompt="Test",
            answer="Response"
        )
        
        assert response.prompt == "Test"
        assert response.answer == "Response"
        assert response.context is None
        assert response.sources is None

    def test_valid_response_all_fields(self):
        """Test valid response with all fields."""
        response = GenerateResponse(
            prompt="Test",
            answer="Response",
            context="Context text",
            sources=["source1", "source2"]
        )
        
        assert response.prompt == "Test"
        assert response.answer == "Response"
        assert response.context == "Context text"
        assert response.sources == ["source1", "source2"]

    def test_missing_prompt(self):
        """Test that missing prompt raises validation error."""
        with pytest.raises(ValidationError):
            GenerateResponse(answer="Response")

    def test_missing_answer(self):
        """Test that missing answer raises validation error."""
        with pytest.raises(ValidationError):
            GenerateResponse(prompt="Test")

    def test_empty_strings(self):
        """Test with empty strings."""
        response = GenerateResponse(
            prompt="",
            answer="",
            context=""
        )
        
        assert response.prompt == ""
        assert response.answer == ""
        assert response.context == ""

    def test_empty_sources_list(self):
        """Test with empty sources list."""
        response = GenerateResponse(
            prompt="Test",
            answer="Response",
            sources=[]
        )
        
        assert response.sources == []

    def test_sources_with_multiple_items(self):
        """Test sources with multiple items."""
        sources = ["source1", "source2", "source3", "source4"]
        response = GenerateResponse(
            prompt="Test",
            answer="Response",
            sources=sources
        )
        
        assert response.sources == sources
        assert len(response.sources) == 4

    def test_unicode_in_response(self):
        """Test unicode in response fields."""
        response = GenerateResponse(
            prompt="世界",
            answer="🌍 response",
            context="Context 文本"
        )
        
        assert "世界" in response.prompt
        assert "🌍" in response.answer
        assert "文本" in response.context

    def test_multiline_in_fields(self):
        """Test multiline content in fields."""
        response = GenerateResponse(
            prompt="Line1\nLine2",
            answer="Answer1\nAnswer2",
            context="Context1\nContext2"
        )
        
        assert "\n" in response.prompt
        assert "\n" in response.answer
        assert "\n" in response.context

    def test_long_context(self):
        """Test with very long context."""
        long_context = "word " * 10000
        response = GenerateResponse(
            prompt="Test",
            answer="Response",
            context=long_context
        )
        
        assert len(response.context) == 50000  # "word " is 5 chars * 10000

    def test_json_serialization(self):
        """Test model can be serialized to JSON."""
        response = GenerateResponse(
            prompt="test",
            answer="answer",
            context="context",
            sources=["s1", "s2"]
        )
        json_data = response.model_dump()
        
        assert json_data["prompt"] == "test"
        assert json_data["answer"] == "answer"
        assert json_data["context"] == "context"
        assert json_data["sources"] == ["s1", "s2"]

    def test_from_dict(self):
        """Test creating response from dictionary."""
        data = {
            "prompt": "test",
            "answer": "answer",
            "context": "ctx",
            "sources": ["src"]
        }
        response = GenerateResponse(**data)
        
        assert response.prompt == "test"
        assert response.answer == "answer"
        assert response.context == "ctx"
        assert response.sources == ["src"]

    def test_sources_with_empty_strings(self):
        """Test sources containing empty strings."""
        response = GenerateResponse(
            prompt="Test",
            answer="Response",
            sources=["", "source1", ""]
        )
        
        assert len(response.sources) == 3
        assert response.sources[0] == ""
        assert response.sources[1] == "source1"

    def test_none_values_for_optional_fields(self):
        """Test explicit None for optional fields."""
        response = GenerateResponse(
            prompt="Test",
            answer="Response",
            context=None,
            sources=None
        )
        
        assert response.context is None
        assert response.sources is None
