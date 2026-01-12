"""Schemas for agent service."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """Request for agent RAG."""
    
    query: str = Field(..., description="User query", min_length=1)
    max_iterations: int = Field(default=5, description="Maximum agent iterations", ge=1, le=10)
    enable_web_search: bool = Field(default=False, description="Enable web search tool")
    enable_code_execution: bool = Field(default=False, description="Enable code execution tool")


class ToolExecutionInfo(BaseModel):
    """Information about a tool execution."""
    
    tool_name: str
    task: str
    confidence: float
    status: str  # pending, success, failed
    reasoning: Optional[str] = None
    error: Optional[str] = None


class AgentResponse(BaseModel):
    """Response from agent RAG."""
    
    query: str
    answer: str
    plan: list[str]
    tool_history: list[ToolExecutionInfo]
    confidence: float
    iterations: int
    total_latency_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricsSummary(BaseModel):
    """Summary of agent metrics."""
    
    total_tools: int
    tools: dict[str, dict[str, Any]]


class ToolInfo(BaseModel):
    """Information about a registered tool."""
    
    name: str
    description: str
    category: str
    capabilities: list[str]
    cost_per_call: float
    avg_latency_ms: float
    success_rate: float
    requires_api_key: bool
