"""Smart router for tool selection with confidence scoring."""

import logging
import re
from dataclasses import dataclass

from src.services.agent.tools.base import BaseTool, ToolCategory
from src.services.agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Routing decision with confidence score.

    Attributes:
        tool_name: Selected tool name
        confidence: Confidence score (0.0 to 1.0)
        reasoning: Explanation for the decision
        fallback_tools: Alternative tools (ordered by preference)
        is_local: Whether the tool is local
        category: Tool category
    """

    tool_name: str
    confidence: float
    reasoning: str
    fallback_tools: list[str]
    is_local: bool
    category: ToolCategory


class AgentRouter:
    """Smart router for selecting appropriate tools."""

    # Confidence thresholds
    HIGH_CONFIDENCE = 0.8
    MEDIUM_CONFIDENCE = 0.5
    LOW_CONFIDENCE = 0.3

    def __init__(self, registry: ToolRegistry):
        """Initialize router with tool registry.

        Args:
            registry: Tool registry instance
        """
        self.registry = registry
        self.logger = logging.getLogger(__name__)

        # Rule-based patterns for quick routing
        self._routing_patterns = {
            r"\b(retrieve|search|find|lookup|documents?|knowledge)\b": ["vectordb_retrieval"],
            r"\b(rank|rerank|score|relevance)\b": ["reranker"],
            r"\b(generate|answer|respond|create|write)\b": ["generator"],
            r"\b(web|internet|online|latest|current|news)\b": ["web_search"],
            r"\b(wikipedia|wiki|encyclopedia)\b": ["wikipedia"],
            r"\b(calculate|compute|code|python|execute|run)\b": ["code_executor"],
            r"\b(weather|temperature|forecast)\b": ["weather"],
        }

    async def route(
        self,
        query: str,
        context: dict | None = None,
        preference: ToolCategory | None = None,
    ) -> RoutingDecision:
        """Route query to appropriate tool with confidence scoring.

        Args:
            query: User query
            context: Optional context (previous results, etc.)
            preference: Optional category preference

        Returns:
            Routing decision with confidence score
        """
        self.logger.info(f"Routing query: {query[:100]}...")

        # Get all available tools
        available_tools = self.registry.list_tools()
        if not available_tools:
            raise ValueError("No tools available in registry")

        # Apply preference filter if specified
        if preference:
            available_tools = [t for t in available_tools if t.metadata.category == preference]

        # Calculate scores for all tools
        scored_tools = []
        for tool in available_tools:
            score = self._calculate_confidence(query, tool, context)
            scored_tools.append((tool, score))

        # Sort by score (descending)
        scored_tools.sort(key=lambda x: x[1], reverse=True)

        # Select best tool
        best_tool, best_score = scored_tools[0]

        # Get fallback tools (next 2 highest scores)
        fallback_tools = [t.metadata.name for t, _ in scored_tools[1:3]]

        # Generate reasoning
        reasoning = self._generate_reasoning(query, best_tool, best_score)

        decision = RoutingDecision(
            tool_name=best_tool.metadata.name,
            confidence=best_score,
            reasoning=reasoning,
            fallback_tools=fallback_tools,
            is_local=best_tool.metadata.category == ToolCategory.LOCAL,
            category=best_tool.metadata.category,
        )

        self.logger.info(
            f"Routed to {decision.tool_name} with confidence {decision.confidence:.2f}"
        )
        return decision

    def _calculate_confidence(
        self,
        query: str,
        tool: BaseTool,
        context: dict | None = None,
    ) -> float:
        """Calculate confidence score for a tool.

        Scoring factors:
        - Rule-based pattern matching (40%)
        - Capability matching (30%)
        - Tool success rate (20%)
        - Cost/latency penalty (10%)

        Args:
            query: User query
            tool: Tool to score
            context: Optional context

        Returns:
            Confidence score (0.0 to 1.0)
        """
        query_lower = query.lower()
        score = 0.0

        # 1. Rule-based pattern matching (40%)
        pattern_score = 0.0
        for pattern, tool_names in self._routing_patterns.items():
            if re.search(pattern, query_lower, re.IGNORECASE) and tool.metadata.name in tool_names:
                pattern_score = 1.0
                break
        score += pattern_score * 0.4

        # 2. Capability matching (30%)
        capability_score = 0.0
        query_words = set(query_lower.split())
        for capability in tool.metadata.capabilities:
            capability_words = set(capability.lower().split())
            overlap = len(query_words & capability_words)
            if overlap > 0:
                capability_score = max(capability_score, overlap / len(query_words))
        score += capability_score * 0.3

        # 3. Tool success rate (20%)
        score += tool.metadata.success_rate * 0.2

        # 4. Cost/latency penalty (10%)
        # Prefer faster, cheaper tools (slight preference)
        cost_penalty = (
            min(tool.metadata.cost_per_call / 0.01, 1.0) if tool.metadata.cost_per_call > 0 else 0
        )
        latency_penalty = min(tool.metadata.avg_latency_ms / 1000, 1.0)
        efficiency_score = 1.0 - (cost_penalty * 0.5 + latency_penalty * 0.5)
        score += efficiency_score * 0.1

        return min(score, 1.0)

    def _generate_reasoning(self, query: str, tool: BaseTool, score: float) -> str:
        """Generate human-readable reasoning for routing decision.

        Args:
            query: User query
            tool: Selected tool
            score: Confidence score

        Returns:
            Reasoning string
        """
        if score >= self.HIGH_CONFIDENCE:
            confidence_level = "high confidence"
        elif score >= self.MEDIUM_CONFIDENCE:
            confidence_level = "medium confidence"
        else:
            confidence_level = "low confidence"

        reasoning = (
            f"Selected '{tool.metadata.name}' with {confidence_level} ({score:.2f}). "
            f"Tool category: {tool.metadata.category.value}. "
            f"Capabilities: {', '.join(tool.metadata.capabilities[:3])}."
        )

        return reasoning

    def fallback_strategy(
        self,
        failed_tools: list[str],
        query: str,
    ) -> RoutingDecision | None:
        """Fallback strategy when primary tools fail.

        Args:
            failed_tools: List of tools that already failed
            query: Original query

        Returns:
            Alternative routing decision or None
        """
        self.logger.warning(f"Applying fallback strategy. Failed tools: {failed_tools}")

        # Get tools that haven't failed yet
        available_tools = [
            t for t in self.registry.list_tools() if t.metadata.name not in failed_tools
        ]

        if not available_tools:
            self.logger.error("No fallback tools available")
            return None

        # Route among remaining tools
        scored_tools = [(t, self._calculate_confidence(query, t, None)) for t in available_tools]
        scored_tools.sort(key=lambda x: x[1], reverse=True)

        tool, score = scored_tools[0]

        return RoutingDecision(
            tool_name=tool.metadata.name,
            confidence=score,
            reasoning=f"Fallback strategy: Selected {tool.metadata.name} after failures",
            fallback_tools=[t.metadata.name for t, _ in scored_tools[1:3]],
            is_local=tool.metadata.category == ToolCategory.LOCAL,
            category=tool.metadata.category,
        )
