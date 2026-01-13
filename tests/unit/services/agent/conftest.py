"""Pytest configuration for agent tests."""

import pytest


# Agent-specific fixtures
@pytest.fixture
def mock_tool():
    """Provide a mock tool for testing."""
    from src.services.agent.tools.base import BaseTool, ToolCategory, ToolMetadata

    class MockTool(BaseTool):
        def __init__(self):
            super().__init__(
                metadata=ToolMetadata(
                    name="mock_tool",
                    description="A mock tool for testing",
                    category=ToolCategory.GENERAL,
                )
            )

        def execute(self, **kwargs):
            return {"status": "success", "result": "mock_result"}

    return MockTool()


@pytest.fixture
def sample_agent_config():
    """Provide sample agent configuration."""
    return {
        "name": "test_agent",
        "description": "Test agent for unit tests",
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_iterations": 5,
    }


@pytest.fixture
def mock_llm_response():
    """Provide mock LLM response."""
    return {
        "content": "This is a test response",
        "role": "assistant",
        "finish_reason": "stop",
    }
