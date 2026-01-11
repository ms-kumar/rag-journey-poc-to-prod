"""
Safe Response Templates for handling guardrail violations.

Provides pre-defined safe responses for various scenarios:
- PII detection
- Toxic content detection
- General safety violations
- Error states
"""

from src.schemas.services.guardrails import ResponseType


class SafeResponseTemplate:
    """Manages safe response templates for guardrail violations."""

    DEFAULT_TEMPLATES = {
        ResponseType.PII_DETECTED: (
            "I noticed your message contains sensitive personal information. "
            "For your privacy and security, I cannot process requests containing "
            "personal data such as email addresses, phone numbers, or identification numbers. "
            "Please rephrase your question without including personal information."
        ),
        ResponseType.TOXIC_CONTENT: (
            "I'm unable to process your request as it contains language that "
            "violates our content policy. Please rephrase your message in a "
            "respectful manner, and I'll be happy to help."
        ),
        ResponseType.UNSAFE_QUERY: (
            "I cannot provide information on this topic as it may be harmful or unsafe. "
            "If you're experiencing a crisis, please reach out to appropriate support services. "
            "I'm here to help with other questions you may have."
        ),
        ResponseType.RATE_LIMIT: (
            "You've reached the rate limit for requests. Please wait a moment before trying again."
        ),
        ResponseType.ERROR: (
            "I encountered an error while processing your request. "
            "Please try again later or contact support if the problem persists."
        ),
        ResponseType.UNAUTHORIZED: (
            "You don't have permission to access this resource. "
            "Please check your credentials or contact an administrator."
        ),
        ResponseType.CONTENT_POLICY_VIOLATION: (
            "Your request violates our content policy. "
            "Please review our usage guidelines and try again with an appropriate query."
        ),
    }

    def __init__(self, custom_templates: dict[ResponseType, str] | None = None):
        """
        Initialize safe response template manager.

        Args:
            custom_templates: Custom templates to override defaults.
        """
        self.templates = self.DEFAULT_TEMPLATES.copy()
        if custom_templates:
            self.templates.update(custom_templates)

    def get_response(
        self,
        response_type: ResponseType,
        context: dict | None = None,
    ) -> str:
        """
        Get a safe response for the given type.

        Args:
            response_type: Type of response needed.
            context: Optional context for template interpolation.

        Returns:
            Safe response string.
        """
        template = self.templates.get(response_type, self.templates[ResponseType.ERROR])

        if context:
            try:
                return template.format(**context)
            except KeyError:
                # If formatting fails, return template as-is
                return template

        return template

    def set_template(self, response_type: ResponseType, template: str) -> None:
        """
        Set or override a response template.

        Args:
            response_type: Type of response to set.
            template: Template string (can include {placeholders}).
        """
        self.templates[response_type] = template

    def get_pii_response(self, pii_types: list[str] | None = None) -> str:
        """
        Get a response for PII detection.

        Args:
            pii_types: Optional list of detected PII types.

        Returns:
            Safe response string.
        """
        if pii_types:
            pii_list = ", ".join(pii_types)
            return (
                f"I detected the following types of sensitive information in your message: {pii_list}. "
                "For your privacy and security, I cannot process this request. "
                "Please remove any personal information and try again."
            )
        return self.get_response(ResponseType.PII_DETECTED)

    def get_toxicity_response(
        self,
        severity: str | None = None,
        categories: list[str] | None = None,
    ) -> str:
        """
        Get a response for toxic content detection.

        Args:
            severity: Severity level of toxicity.
            categories: Categories of toxic content detected.

        Returns:
            Safe response string.
        """
        if severity == "severe" or (
            categories and any(cat in ["threat", "self_harm", "violence"] for cat in categories)
        ):
            return (
                "I cannot process this request as it contains content that may be harmful. "
                "If you're in crisis or need help, please contact emergency services or "
                "appropriate support resources. I'm here to assist with other questions."
            )
        return self.get_response(ResponseType.TOXIC_CONTENT)

    def get_fallback_response(self) -> str:
        """
        Get a generic fallback response.

        Returns:
            Generic safe response string.
        """
        return (
            "I'm unable to process your request at this time. "
            "Please try rephrasing your question or contact support for assistance."
        )

    def add_helpful_context(
        self,
        base_response: str,
        suggestions: list[str] | None = None,
    ) -> str:
        """
        Add helpful suggestions to a base response.

        Args:
            base_response: Base response text.
            suggestions: Optional list of suggestions.

        Returns:
            Enhanced response with suggestions.
        """
        if not suggestions:
            return base_response

        suggestions_text = "\n\nHere are some alternatives:\n" + "\n".join(
            f"- {suggestion}" for suggestion in suggestions
        )
        return base_response + suggestions_text

    def format_with_support_info(
        self,
        base_response: str,
        support_email: str | None = None,
        support_url: str | None = None,
    ) -> str:
        """
        Add support contact information to a response.

        Args:
            base_response: Base response text.
            support_email: Support email address.
            support_url: Support URL.

        Returns:
            Response with support information.
        """
        support_info = []
        if support_email:
            support_info.append(f"Email: {support_email}")
        if support_url:
            support_info.append(f"Support: {support_url}")

        if support_info:
            support_text = "\n\nFor assistance, contact us:\n" + "\n".join(support_info)
            return base_response + support_text

        return base_response


class ResponseBuilder:
    """Builder for constructing detailed safe responses."""

    def __init__(self, template_manager: SafeResponseTemplate | None = None):
        """
        Initialize response builder.

        Args:
            template_manager: SafeResponseTemplate instance.
        """
        self.template_manager = template_manager or SafeResponseTemplate()
        self._response_parts: list[str] = []

    def add_base_message(self, response_type: ResponseType) -> "ResponseBuilder":
        """Add base message from template."""
        self._response_parts.append(self.template_manager.get_response(response_type))
        return self

    def add_custom_message(self, message: str) -> "ResponseBuilder":
        """Add custom message."""
        self._response_parts.append(message)
        return self

    def add_explanation(self, explanation: str) -> "ResponseBuilder":
        """Add explanation."""
        self._response_parts.append(f"\n\nExplanation: {explanation}")
        return self

    def add_suggestions(self, suggestions: list[str]) -> "ResponseBuilder":
        """Add suggestions."""
        if suggestions:
            suggestions_text = "\n\nSuggestions:\n" + "\n".join(f"- {s}" for s in suggestions)
            self._response_parts.append(suggestions_text)
        return self

    def build(self) -> str:
        """Build final response."""
        return " ".join(self._response_parts)

    def reset(self) -> "ResponseBuilder":
        """Reset builder."""
        self._response_parts = []
        return self
