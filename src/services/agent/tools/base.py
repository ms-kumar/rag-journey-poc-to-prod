"""Base tool interface for Agentic RAG."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    """Tool categories for routing."""

    LOCAL = "local"
    EXTERNAL = "external"
    HYBRID = "hybrid"


@dataclass
class ToolMetadata:
    """Metadata for a tool.

    Attributes:
        name: Unique tool name
        description: Human-readable description
        category: Tool category (local/external/hybrid)
        capabilities: List of capabilities
        cost_per_call: Estimated cost per invocation
        avg_latency_ms: Average latency in milliseconds
        success_rate: Historical success rate (0.0 to 1.0)
        requires_api_key: Whether tool requires API key
        version: Tool version
    """

    name: str
    description: str
    category: ToolCategory
    capabilities: list[str] = field(default_factory=list)
    cost_per_call: float = 0.0
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    requires_api_key: bool = False
    version: str = "1.0.0"


class BaseTool(ABC):
    """Base interface for all agent tools."""

    def __init__(self, metadata: ToolMetadata):
        """Initialize the tool with metadata.

        Args:
            metadata: Tool metadata
        """
        self.metadata = metadata
        self.logger = logging.getLogger(f"{__name__}.{metadata.name}")

    @abstractmethod
    async def execute(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool with the given query.

        Args:
            query: User query or task
            **kwargs: Additional tool-specific parameters

        Returns:
            Dictionary containing:
                - success: bool
                - result: Any (tool output)
                - error: str (if failed)
                - metadata: dict (execution metadata)
        """
        pass

    def validate_input(self, query: str, **kwargs: Any) -> bool:
        """Validate input parameters.

        Args:
            query: User query
            **kwargs: Additional parameters

        Returns:
            True if valid, False otherwise
        """
        if not query or not isinstance(query, str):
            self.logger.error("Invalid query: must be a non-empty string")
            return False
        return True

    def get_metadata(self) -> ToolMetadata:
        """Get tool metadata.

        Returns:
            Tool metadata
        """
        return self.metadata

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.metadata.name}, category={self.metadata.category})"
