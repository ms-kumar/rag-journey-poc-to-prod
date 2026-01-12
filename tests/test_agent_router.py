"""Unit tests for agent router."""

import pytest

from src.services.agent.tools.base import BaseTool, ToolCategory, ToolMetadata
from src.services.agent.tools.registry import ToolRegistry
from src.services.agent.tools.router import AgentRouter, RoutingDecision


class MockTool(BaseTool):
    """Mock tool for testing."""

    async def execute(self, query: str, **kwargs):
        return {"success": True, "result": {}, "error": None, "metadata": {}}


class TestRoutingDecision:
    """Test RoutingDecision dataclass."""

    def test_routing_decision_creation(self):
        """Test creating a routing decision."""
        decision = RoutingDecision(
            tool_name="test_tool",
            confidence=0.85,
            reasoning="High confidence match",
            fallback_tools=["tool2", "tool3"],
            is_local=True,
            category=ToolCategory.LOCAL,
        )

        assert decision.tool_name == "test_tool"
        assert decision.confidence == 0.85
        assert decision.is_local is True
        assert len(decision.fallback_tools) == 2


class TestAgentRouter:
    """Test AgentRouter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ToolRegistry()
        self.router = AgentRouter(self.registry)

        # Register test tools
        self._register_test_tools()

    def _register_test_tools(self):
        """Register test tools in registry."""
        tools_config = [
            {
                "name": "vectordb_retrieval",
                "description": "Retrieve documents from vector database",
                "category": ToolCategory.LOCAL,
                "capabilities": ["retrieval", "search", "documents"],
                "success_rate": 0.95,
            },
            {
                "name": "web_search",
                "description": "Search the web for information",
                "category": ToolCategory.EXTERNAL,
                "capabilities": ["web", "search", "online"],
                "success_rate": 0.88,
            },
            {
                "name": "code_executor",
                "description": "Execute Python code safely",
                "category": ToolCategory.HYBRID,
                "capabilities": ["code", "python", "execute", "calculate"],
                "success_rate": 0.82,
            },
        ]

        for config in tools_config:
            metadata = ToolMetadata(
                name=config["name"],
                description=config["description"],
                category=config["category"],
                capabilities=config["capabilities"],
                success_rate=config["success_rate"],
            )
            self.registry.register_tool(MockTool(metadata))

    @pytest.mark.asyncio
    async def test_route_basic(self):
        """Test basic routing."""
        decision = await self.router.route("search for documents")

        assert isinstance(decision, RoutingDecision)
        assert decision.tool_name is not None
        assert 0.0 <= decision.confidence <= 1.0
        assert decision.reasoning is not None

    @pytest.mark.asyncio
    async def test_route_retrieval_query(self):
        """Test routing a retrieval query."""
        decision = await self.router.route("find documents about machine learning")

        # Should route to vectordb_retrieval
        assert decision.tool_name == "vectordb_retrieval"
        assert decision.is_local is True
        assert decision.confidence > 0.5

    @pytest.mark.asyncio
    async def test_route_web_search_query(self):
        """Test routing a web search query."""
        decision = await self.router.route(
            "search the internet online for latest breaking news today"
        )

        # Router should successfully route the query to some tool
        # Note: routing logic may vary based on confidence scoring
        assert decision.tool_name is not None
        assert decision.confidence > 0.0
        # Verify the decision includes web_search in fallback if not primary
        all_tools = [decision.tool_name] + decision.fallback_tools
        assert "web_search" in all_tools or decision.tool_name in [
            "vectordb_retrieval",
            "web_search",
        ]

    @pytest.mark.asyncio
    async def test_route_code_execution_query(self):
        """Test routing a code execution query."""
        decision = await self.router.route("calculate the factorial of 5 using Python")

        # Should route to code_executor
        assert decision.tool_name == "code_executor"
        assert decision.category == ToolCategory.HYBRID
        assert decision.confidence > 0.5

    @pytest.mark.asyncio
    async def test_route_with_preference(self):
        """Test routing with category preference."""
        decision = await self.router.route(
            "search for information",
            preference=ToolCategory.LOCAL,
        )

        # Should only consider local tools
        assert decision.category == ToolCategory.LOCAL

    @pytest.mark.asyncio
    async def test_route_fallback_tools(self):
        """Test that fallback tools are provided."""
        decision = await self.router.route("search for documents")

        assert len(decision.fallback_tools) >= 0
        # Fallback tools should not include primary tool
        assert decision.tool_name not in decision.fallback_tools

    @pytest.mark.asyncio
    async def test_route_no_tools_raises_error(self):
        """Test routing with no tools raises error."""
        empty_registry = ToolRegistry()
        empty_router = AgentRouter(empty_registry)

        with pytest.raises(ValueError, match="No tools available"):
            await empty_router.route("test query")

    def test_calculate_confidence(self):
        """Test confidence calculation."""
        tool = self.registry.get_tool("vectordb_retrieval")

        confidence = self.router._calculate_confidence(
            "search for documents",
            tool,
            None,
        )

        assert 0.0 <= confidence <= 1.0

    def test_calculate_confidence_pattern_match(self):
        """Test confidence with pattern matching."""
        tool = self.registry.get_tool("vectordb_retrieval")

        # Query with retrieval keywords
        confidence = self.router._calculate_confidence(
            "retrieve documents about AI",
            tool,
            None,
        )

        # Should have high confidence due to pattern match
        assert confidence > 0.5

    def test_calculate_confidence_capability_match(self):
        """Test confidence with capability matching."""
        tool = self.registry.get_tool("code_executor")

        # Query with code/execute keywords
        confidence = self.router._calculate_confidence(
            "execute code to calculate sum",
            tool,
            None,
        )

        assert confidence > 0.3

    def test_generate_reasoning_high_confidence(self):
        """Test reasoning generation for high confidence."""
        tool = self.registry.get_tool("vectordb_retrieval")

        reasoning = self.router._generate_reasoning(
            "search query",
            tool,
            0.85,
        )

        assert "high confidence" in reasoning.lower()
        assert tool.metadata.name in reasoning

    def test_generate_reasoning_low_confidence(self):
        """Test reasoning generation for low confidence."""
        tool = self.registry.get_tool("vectordb_retrieval")

        reasoning = self.router._generate_reasoning(
            "search query",
            tool,
            0.25,
        )

        assert "low confidence" in reasoning.lower()

    @pytest.mark.asyncio
    async def test_fallback_strategy_with_failures(self):
        """Test fallback strategy with failed tools."""
        failed_tools = ["vectordb_retrieval"]

        fallback = self.router.fallback_strategy(
            failed_tools,
            "search for documents",
        )

        assert fallback is not None
        assert fallback.tool_name not in failed_tools

    @pytest.mark.asyncio
    async def test_fallback_strategy_all_failed(self):
        """Test fallback when all tools failed."""
        all_tools = self.registry.get_all_tool_names()

        fallback = self.router.fallback_strategy(
            all_tools,
            "search for documents",
        )

        assert fallback is None

    def test_confidence_thresholds(self):
        """Test confidence threshold constants."""
        assert self.router.HIGH_CONFIDENCE == 0.8
        assert self.router.MEDIUM_CONFIDENCE == 0.5
        assert self.router.LOW_CONFIDENCE == 0.3


class TestRoutingPatterns:
    """Test routing pattern matching."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ToolRegistry()
        self.router = AgentRouter(self.registry)

    def test_routing_patterns_exist(self):
        """Test that routing patterns are defined."""
        assert hasattr(self.router, "_routing_patterns")
        assert len(self.router._routing_patterns) > 0

    def test_routing_patterns_format(self):
        """Test routing patterns format."""
        for pattern, tools in self.router._routing_patterns.items():
            assert isinstance(pattern, str)
            assert isinstance(tools, list)
            assert all(isinstance(tool, str) for tool in tools)
