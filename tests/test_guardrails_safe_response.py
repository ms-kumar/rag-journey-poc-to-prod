"""
Unit tests for safe response templates.
"""

import pytest

from src.services.guardrails.safe_response import (
    ResponseBuilder,
    ResponseType,
    SafeResponseTemplate,
)


class TestSafeResponseTemplate:
    """Tests for SafeResponseTemplate class."""

    def test_get_pii_response(self):
        """Test getting PII response."""
        template = SafeResponseTemplate()
        response = template.get_response(ResponseType.PII_DETECTED)

        assert "personal information" in response.lower() or "pii" in response.lower()
        assert len(response) > 0

    def test_get_toxicity_response(self):
        """Test getting toxicity response."""
        template = SafeResponseTemplate()
        response = template.get_response(ResponseType.TOXIC_CONTENT)

        assert "content policy" in response.lower() or "toxic" in response.lower() or "language" in response.lower()
        assert len(response) > 0

    def test_get_unsafe_query_response(self):
        """Test getting unsafe query response."""
        template = SafeResponseTemplate()
        response = template.get_response(ResponseType.UNSAFE_QUERY)

        assert len(response) > 0

    def test_get_error_response(self):
        """Test getting error response."""
        template = SafeResponseTemplate()
        response = template.get_response(ResponseType.ERROR)

        assert "error" in response.lower()
        assert len(response) > 0

    def test_get_unauthorized_response(self):
        """Test getting unauthorized response."""
        template = SafeResponseTemplate()
        response = template.get_response(ResponseType.UNAUTHORIZED)

        assert "permission" in response.lower() or "unauthorized" in response.lower()
        assert len(response) > 0

    def test_custom_templates(self):
        """Test custom template override."""
        custom_templates = {
            ResponseType.ERROR: "Custom error message: {error_code}"
        }
        template = SafeResponseTemplate(custom_templates=custom_templates)

        response = template.get_response(ResponseType.ERROR)
        assert "Custom error message" in response

    def test_set_template(self):
        """Test setting template dynamically."""
        template = SafeResponseTemplate()
        custom_message = "This is a custom PII message."

        template.set_template(ResponseType.PII_DETECTED, custom_message)
        response = template.get_response(ResponseType.PII_DETECTED)

        assert response == custom_message

    def test_get_pii_response_with_types(self):
        """Test getting PII response with specific types."""
        template = SafeResponseTemplate()
        pii_types = ["email", "phone"]
        response = template.get_pii_response(pii_types)

        assert "email" in response
        assert "phone" in response

    def test_get_toxicity_response_severe(self):
        """Test getting toxicity response for severe content."""
        template = SafeResponseTemplate()
        response = template.get_toxicity_response(
            severity="severe",
            categories=["threat", "violence"]
        )

        assert "harmful" in response.lower() or "cannot" in response.lower()
        assert len(response) > 0

    def test_get_toxicity_response_moderate(self):
        """Test getting toxicity response for moderate content."""
        template = SafeResponseTemplate()
        response = template.get_toxicity_response(
            severity="medium",
            categories=["profanity"]
        )

        assert len(response) > 0

    def test_get_fallback_response(self):
        """Test fallback response."""
        template = SafeResponseTemplate()
        response = template.get_fallback_response()

        assert len(response) > 0
        assert "unable" in response.lower() or "cannot" in response.lower()

    def test_add_helpful_context(self):
        """Test adding helpful context to response."""
        template = SafeResponseTemplate()
        base_response = "Your request cannot be processed."
        suggestions = [
            "Try rephrasing your question",
            "Remove personal information"
        ]

        enhanced = template.add_helpful_context(base_response, suggestions)

        assert base_response in enhanced
        assert "Try rephrasing" in enhanced
        assert "Remove personal information" in enhanced

    def test_format_with_support_info(self):
        """Test adding support information."""
        template = SafeResponseTemplate()
        base_response = "An error occurred."
        support_email = "support@example.com"
        support_url = "https://support.example.com"

        formatted = template.format_with_support_info(
            base_response,
            support_email=support_email,
            support_url=support_url
        )

        assert base_response in formatted
        assert support_email in formatted
        assert support_url in formatted

    def test_response_interpolation(self):
        """Test response template interpolation."""
        custom_template = "Error code: {code}, Message: {message}"
        template = SafeResponseTemplate(
            custom_templates={ResponseType.ERROR: custom_template}
        )

        context = {"code": "500", "message": "Internal error"}
        response = template.get_response(ResponseType.ERROR, context=context)

        assert "500" in response
        assert "Internal error" in response


class TestResponseBuilder:
    """Tests for ResponseBuilder class."""

    def test_build_simple_response(self):
        """Test building simple response."""
        builder = ResponseBuilder()
        response = builder.add_base_message(ResponseType.ERROR).build()

        assert len(response) > 0
        assert "error" in response.lower()

    def test_build_with_custom_message(self):
        """Test building response with custom message."""
        builder = ResponseBuilder()
        response = (
            builder
            .add_custom_message("Custom error occurred.")
            .build()
        )

        assert "Custom error occurred." in response

    def test_build_with_explanation(self):
        """Test building response with explanation."""
        builder = ResponseBuilder()
        response = (
            builder
            .add_base_message(ResponseType.ERROR)
            .add_explanation("The server is temporarily unavailable.")
            .build()
        )

        assert "Explanation:" in response
        assert "temporarily unavailable" in response

    def test_build_with_suggestions(self):
        """Test building response with suggestions."""
        builder = ResponseBuilder()
        suggestions = ["Try again later", "Contact support"]
        response = (
            builder
            .add_base_message(ResponseType.ERROR)
            .add_suggestions(suggestions)
            .build()
        )

        assert "Suggestions:" in response
        assert "Try again later" in response
        assert "Contact support" in response

    def test_build_complete_response(self):
        """Test building complete response with all components."""
        builder = ResponseBuilder()
        response = (
            builder
            .add_base_message(ResponseType.ERROR)
            .add_explanation("Network timeout occurred.")
            .add_suggestions(["Check your connection", "Retry the request"])
            .build()
        )

        assert "error" in response.lower()
        assert "Explanation:" in response
        assert "Network timeout" in response
        assert "Suggestions:" in response
        assert "Check your connection" in response

    def test_builder_reset(self):
        """Test builder reset functionality."""
        builder = ResponseBuilder()

        # Build first response
        response1 = (
            builder
            .add_custom_message("First message")
            .build()
        )

        # Reset and build second response
        response2 = (
            builder
            .reset()
            .add_custom_message("Second message")
            .build()
        )

        assert "First message" in response1
        assert "First message" not in response2
        assert "Second message" in response2

    def test_multiple_custom_messages(self):
        """Test adding multiple custom messages."""
        builder = ResponseBuilder()
        response = (
            builder
            .add_custom_message("First part.")
            .add_custom_message("Second part.")
            .build()
        )

        assert "First part." in response
        assert "Second part." in response


class TestResponseTemplateIntegration:
    """Integration tests for response templates."""

    def test_pii_workflow(self):
        """Test complete PII detection response workflow."""
        template = SafeResponseTemplate()

        # Simulate PII detection
        pii_types = ["email", "phone", "ssn"]
        response = template.get_pii_response(pii_types)

        assert all(pii_type in response for pii_type in pii_types)
        assert "privacy" in response.lower() or "personal" in response.lower()

    def test_toxicity_workflow(self):
        """Test complete toxicity detection response workflow."""
        template = SafeResponseTemplate()
        builder = ResponseBuilder(template)

        # Moderate toxicity
        response = (
            builder
            .add_base_message(ResponseType.TOXIC_CONTENT)
            .add_suggestions(["Rephrase politely", "Follow community guidelines"])
            .build()
        )

        assert "content policy" in response.lower() or "language" in response.lower()
        assert "Suggestions:" in response

    def test_error_with_support(self):
        """Test error response with support information."""
        template = SafeResponseTemplate()
        base_response = template.get_response(ResponseType.ERROR)

        response_with_support = template.format_with_support_info(
            base_response,
            support_email="help@example.com",
            support_url="https://help.example.com"
        )

        assert base_response in response_with_support
        assert "help@example.com" in response_with_support
        assert "https://help.example.com" in response_with_support
