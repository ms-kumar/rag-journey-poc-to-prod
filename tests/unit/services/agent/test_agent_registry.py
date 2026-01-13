"""Unit tests for agent tool registry."""

import pytest

from src.services.agent.tools.base import BaseTool, ToolCategory, ToolMetadata
from src.services.agent.tools.registry import ToolRegistry, get_tool_registry


class MockTool(BaseTool):
    """Mock tool for testing."""

    async def execute(self, query: str, **kwargs):
        return {"success": True, "result": {"mock": True}, "error": None, "metadata": {}}


class TestToolRegistry:
    """Test ToolRegistry class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ToolRegistry()

    def test_registry_initialization(self):
        """Test registry initialization."""
        assert len(self.registry) == 0
        assert self.registry.get_all_tool_names() == []

    def test_register_tool(self):
        """Test registering a tool."""
        metadata = ToolMetadata(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.LOCAL,
        )
        tool = MockTool(metadata)

        self.registry.register_tool(tool)

        assert len(self.registry) == 1
        assert "test_tool" in self.registry.get_all_tool_names()

    def test_register_duplicate_tool(self):
        """Test registering a duplicate tool raises error."""
        metadata = ToolMetadata(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.LOCAL,
        )
        tool1 = MockTool(metadata)
        tool2 = MockTool(metadata)

        self.registry.register_tool(tool1)

        with pytest.raises(ValueError, match="already registered"):
            self.registry.register_tool(tool2)

    def test_get_tool(self):
        """Test getting a tool by name."""
        metadata = ToolMetadata(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.LOCAL,
        )
        tool = MockTool(metadata)
        self.registry.register_tool(tool)

        retrieved_tool = self.registry.get_tool("test_tool")

        assert retrieved_tool is not None
        assert retrieved_tool.metadata.name == "test_tool"

    def test_get_nonexistent_tool(self):
        """Test getting a nonexistent tool returns None."""
        result = self.registry.get_tool("nonexistent")
        assert result is None

    def test_list_tools_all(self):
        """Test listing all tools."""
        # Register multiple tools
        for i in range(3):
            metadata = ToolMetadata(
                name=f"tool_{i}",
                description=f"Tool {i}",
                category=ToolCategory.LOCAL,
            )
            tool = MockTool(metadata)
            self.registry.register_tool(tool)

        tools = self.registry.list_tools()
        assert len(tools) == 3

    def test_list_tools_by_category(self):
        """Test listing tools filtered by category."""
        # Register tools with different categories
        for category in [ToolCategory.LOCAL, ToolCategory.EXTERNAL, ToolCategory.HYBRID]:
            metadata = ToolMetadata(
                name=f"tool_{category.value}",
                description=f"Tool {category.value}",
                category=category,
            )
            tool = MockTool(metadata)
            self.registry.register_tool(tool)

        local_tools = self.registry.list_tools(category=ToolCategory.LOCAL)
        assert len(local_tools) == 1
        assert local_tools[0].metadata.category == ToolCategory.LOCAL

        external_tools = self.registry.list_tools(category=ToolCategory.EXTERNAL)
        assert len(external_tools) == 1

    def test_get_tool_by_capability(self):
        """Test finding tools by capability."""
        metadata1 = ToolMetadata(
            name="search_tool",
            description="Search tool",
            category=ToolCategory.LOCAL,
            capabilities=["search", "retrieval", "vector"],
        )
        metadata2 = ToolMetadata(
            name="generate_tool",
            description="Generate tool",
            category=ToolCategory.LOCAL,
            capabilities=["generation", "text", "llm"],
        )

        self.registry.register_tool(MockTool(metadata1))
        self.registry.register_tool(MockTool(metadata2))

        search_tools = self.registry.get_tool_by_capability("search")
        assert len(search_tools) == 1
        assert search_tools[0].metadata.name == "search_tool"

        gen_tools = self.registry.get_tool_by_capability("generation")
        assert len(gen_tools) == 1
        assert gen_tools[0].metadata.name == "generate_tool"

    def test_get_tool_by_capability_case_insensitive(self):
        """Test capability search is case insensitive."""
        metadata = ToolMetadata(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.LOCAL,
            capabilities=["Search", "Retrieval"],
        )
        self.registry.register_tool(MockTool(metadata))

        tools = self.registry.get_tool_by_capability("search")
        assert len(tools) == 1

        tools = self.registry.get_tool_by_capability("RETRIEVAL")
        assert len(tools) == 1

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        metadata = ToolMetadata(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.LOCAL,
        )
        tool = MockTool(metadata)
        self.registry.register_tool(tool)

        assert len(self.registry) == 1

        result = self.registry.unregister_tool("test_tool")
        assert result is True
        assert len(self.registry) == 0

    def test_unregister_nonexistent_tool(self):
        """Test unregistering a nonexistent tool returns False."""
        result = self.registry.unregister_tool("nonexistent")
        assert result is False

    def test_get_all_tool_names(self):
        """Test getting all tool names."""
        names = ["tool_a", "tool_b", "tool_c"]
        for name in names:
            metadata = ToolMetadata(
                name=name,
                description=f"Tool {name}",
                category=ToolCategory.LOCAL,
            )
            self.registry.register_tool(MockTool(metadata))

        all_names = self.registry.get_all_tool_names()
        assert len(all_names) == 3
        assert set(all_names) == set(names)

    def test_registry_len(self):
        """Test registry length."""
        assert len(self.registry) == 0

        for i in range(5):
            metadata = ToolMetadata(
                name=f"tool_{i}",
                description=f"Tool {i}",
                category=ToolCategory.LOCAL,
            )
            self.registry.register_tool(MockTool(metadata))

        assert len(self.registry) == 5

    def test_registry_repr(self):
        """Test registry string representation."""
        repr_str = repr(self.registry)
        assert "ToolRegistry" in repr_str
        assert "tools=0" in repr_str


class TestGetToolRegistry:
    """Test global tool registry singleton."""

    def test_get_tool_registry_singleton(self):
        """Test get_tool_registry returns singleton."""
        registry1 = get_tool_registry()
        registry2 = get_tool_registry()

        assert registry1 is registry2

    def test_global_registry_persists(self):
        """Test global registry persists across calls."""
        registry = get_tool_registry()

        metadata = ToolMetadata(
            name="persistent_tool",
            description="Persistent tool",
            category=ToolCategory.LOCAL,
        )
        registry.register_tool(MockTool(metadata))

        # Get registry again
        registry2 = get_tool_registry()
        tool = registry2.get_tool("persistent_tool")

        assert tool is not None
        assert tool.metadata.name == "persistent_tool"
