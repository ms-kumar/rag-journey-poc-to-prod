"""Unit tests for agent tools base classes."""

import pytest

from src.services.agent.tools.base import BaseTool, ToolCategory, ToolMetadata


class DummyTool(BaseTool):
    """Dummy tool for testing."""

    async def execute(self, query: str, **kwargs):
        """Execute dummy operation."""
        if not self.validate_input(query, **kwargs):
            return {
                "success": False,
                "result": None,
                "error": "Invalid input",
                "metadata": {},
            }

        return {
            "success": True,
            "result": {"output": f"Processed: {query}"},
            "error": None,
            "metadata": {"test": True},
        }


class TestToolMetadata:
    """Test ToolMetadata dataclass."""

    def test_metadata_creation(self):
        """Test creating tool metadata."""
        metadata = ToolMetadata(
            name="test_tool",
            description="Test tool for testing",
            category=ToolCategory.LOCAL,
            capabilities=["testing", "validation"],
            cost_per_call=0.0,
            avg_latency_ms=100.0,
            success_rate=0.95,
            requires_api_key=False,
        )

        assert metadata.name == "test_tool"
        assert metadata.category == ToolCategory.LOCAL
        assert len(metadata.capabilities) == 2
        assert metadata.success_rate == 0.95

    def test_metadata_defaults(self):
        """Test metadata with default values."""
        metadata = ToolMetadata(
            name="minimal_tool",
            description="Minimal tool",
            category=ToolCategory.EXTERNAL,
        )

        assert metadata.capabilities == []
        assert metadata.cost_per_call == 0.0
        assert metadata.avg_latency_ms == 0.0
        assert metadata.success_rate == 1.0
        assert metadata.requires_api_key is False
        assert metadata.version == "1.0.0"


class TestBaseTool:
    """Test BaseTool abstract class."""

    def test_tool_creation(self):
        """Test creating a tool instance."""
        metadata = ToolMetadata(
            name="dummy_tool",
            description="Dummy test tool",
            category=ToolCategory.LOCAL,
        )
        tool = DummyTool(metadata)

        assert tool.metadata.name == "dummy_tool"
        assert tool.metadata.category == ToolCategory.LOCAL

    @pytest.mark.asyncio
    async def test_tool_execute_success(self):
        """Test successful tool execution."""
        metadata = ToolMetadata(
            name="dummy_tool",
            description="Dummy test tool",
            category=ToolCategory.LOCAL,
        )
        tool = DummyTool(metadata)

        result = await tool.execute("test query")

        assert result["success"] is True
        assert result["error"] is None
        assert "output" in result["result"]
        assert result["result"]["output"] == "Processed: test query"

    @pytest.mark.asyncio
    async def test_tool_execute_invalid_input(self):
        """Test tool execution with invalid input."""
        metadata = ToolMetadata(
            name="dummy_tool",
            description="Dummy test tool",
            category=ToolCategory.LOCAL,
        )
        tool = DummyTool(metadata)

        # Empty query should fail validation
        result = await tool.execute("")

        assert result["success"] is False
        assert result["error"] == "Invalid input"

    def test_validate_input_valid(self):
        """Test input validation with valid input."""
        metadata = ToolMetadata(
            name="dummy_tool",
            description="Dummy test tool",
            category=ToolCategory.LOCAL,
        )
        tool = DummyTool(metadata)

        assert tool.validate_input("valid query") is True
        assert tool.validate_input("test", param1="value") is True

    def test_validate_input_invalid(self):
        """Test input validation with invalid input."""
        metadata = ToolMetadata(
            name="dummy_tool",
            description="Dummy test tool",
            category=ToolCategory.LOCAL,
        )
        tool = DummyTool(metadata)

        assert tool.validate_input("") is False
        assert tool.validate_input(None) is False
        assert tool.validate_input(123) is False

    def test_get_metadata(self):
        """Test getting tool metadata."""
        metadata = ToolMetadata(
            name="dummy_tool",
            description="Dummy test tool",
            category=ToolCategory.LOCAL,
        )
        tool = DummyTool(metadata)

        retrieved_metadata = tool.get_metadata()
        assert retrieved_metadata.name == "dummy_tool"
        assert retrieved_metadata.category == ToolCategory.LOCAL

    def test_tool_repr(self):
        """Test tool string representation."""
        metadata = ToolMetadata(
            name="dummy_tool",
            description="Dummy test tool",
            category=ToolCategory.LOCAL,
        )
        tool = DummyTool(metadata)

        repr_str = repr(tool)
        assert "DummyTool" in repr_str
        assert "dummy_tool" in repr_str
        assert "LOCAL" in repr_str


class TestToolCategory:
    """Test ToolCategory enum."""

    def test_category_values(self):
        """Test tool category enum values."""
        assert ToolCategory.LOCAL == "local"
        assert ToolCategory.EXTERNAL == "external"
        assert ToolCategory.HYBRID == "hybrid"

    def test_category_comparison(self):
        """Test category comparison."""
        assert ToolCategory.LOCAL != ToolCategory.EXTERNAL
        assert ToolCategory.LOCAL == ToolCategory.LOCAL
