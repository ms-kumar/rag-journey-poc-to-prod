"""Confidence scoring for tool selection."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Calculate confidence scores for tool selection."""
    
    def __init__(self):
        """Initialize confidence scorer."""
        self.logger = logging.getLogger(__name__)
    
    def score_tool_selection(
        self,
        query: str,
        tool_name: str,
        tool_description: str,
        capabilities: list[str],
        success_rate: float = 1.0,
        cost: float = 0.0,
        latency_ms: float = 0.0,
    ) -> float:
        """Calculate confidence score for tool selection.
        
        Scoring factors:
        - Query-tool semantic match (40%)
        - Historical success rate (30%)
        - Tool capabilities match (20%)
        - Cost/latency efficiency (10%)
        
        Args:
            query: User query
            tool_name: Tool name
            tool_description: Tool description
            capabilities: List of tool capabilities
            success_rate: Historical success rate (0.0 to 1.0)
            cost: Cost per call
            latency_ms: Average latency in milliseconds
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        score = 0.0
        query_lower = query.lower()
        
        # 1. Semantic match with description (40%)
        description_words = set(tool_description.lower().split())
        query_words = set(query_lower.split())
        
        overlap = len(description_words & query_words)
        if overlap > 0 and len(query_words) > 0:
            semantic_score = min(overlap / len(query_words), 1.0)
        else:
            semantic_score = 0.0
        
        score += semantic_score * 0.4
        
        # 2. Historical success rate (30%)
        score += success_rate * 0.3
        
        # 3. Capabilities match (20%)
        capability_score = 0.0
        for capability in capabilities:
            capability_words = set(capability.lower().split())
            overlap = len(capability_words & query_words)
            if overlap > 0:
                capability_score = max(
                    capability_score,
                    overlap / len(query_words),
                )
        score += capability_score * 0.2
        
        # 4. Cost/latency efficiency (10%)
        # Prefer faster, cheaper tools
        cost_penalty = min(cost / 0.01, 1.0) if cost > 0 else 0
        latency_penalty = min(latency_ms / 1000, 1.0)
        efficiency_score = 1.0 - (cost_penalty * 0.5 + latency_penalty * 0.5)
        score += efficiency_score * 0.1
        
        return min(score, 1.0)
    
    def score_result_quality(
        self,
        result: dict,
        expected_fields: Optional[list[str]] = None,
    ) -> float:
        """Score the quality of a tool result.
        
        Args:
            result: Tool execution result
            expected_fields: Optional list of expected fields
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        if not result.get("success"):
            return 0.0
        
        score = 0.5  # Base score for success
        
        result_data = result.get("result", {})
        
        # Check if result has content
        if result_data:
            score += 0.2
        
        # Check for expected fields
        if expected_fields:
            present_fields = sum(1 for f in expected_fields if f in result_data)
            score += (present_fields / len(expected_fields)) * 0.3
        else:
            score += 0.3  # No expectations, full score
        
        return min(score, 1.0)
