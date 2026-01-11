"""
Main query understanding orchestrator.

Combines query rewriting, synonym expansion, and optional intent
classification to preprocess queries before retrieval.
"""

import logging
import time
from typing import TYPE_CHECKING, Literal

from src.services.query_understanding.rewriter import QueryRewriter, QueryRewriterConfig
from src.services.query_understanding.synonym_expander import (
    SynonymExpander,
    SynonymExpanderConfig,
)

if TYPE_CHECKING:
    from src.config import Settings

logger = logging.getLogger(__name__)

IntentType = Literal["factual", "howto", "comparison", "troubleshooting", "exploratory"]


class QueryUnderstandingClient:
    """
    Query understanding orchestrator.

    Preprocesses queries through multiple stages:
    1. Query rewriting (typos, acronyms, context)
    2. Synonym expansion (improve recall)
    3. Intent classification (optional)
    """

    def __init__(self, settings: "Settings"):
        """
        Initialize query understanding.

        Args:
            settings: Application settings containing query understanding configuration
        """
        self.settings = settings
        self.query_settings = settings.query_understanding

        # Initialize components
        self.rewriter = None
        self.expander = None

        if self.query_settings.enable_rewriting:
            rewriter_config = QueryRewriterConfig.from_settings(settings)
            self.rewriter = QueryRewriter(rewriter_config)

        if self.query_settings.enable_synonyms:
            expander_config = SynonymExpanderConfig.from_settings(settings)
            self.expander = SynonymExpander(expander_config)

    def process(self, query: str) -> dict:
        """
        Process query through all understanding stages.

        Args:
            query: Original query string

        Returns:
            Dictionary containing:
                - original_query: Original query
                - processed_query: Final processed query
                - rewritten_query: Query after rewriting (if enabled)
                - expanded_query: Query after expansion (if enabled)
                - intent: Classified intent (if enabled)
                - metadata: Latency and stage info

        Example:
            >>> qu = QueryUnderstandingClient()
            >>> result = qu.process("what is ML?")
            >>> print(result["processed_query"])
            "what is machine learning? ml statistical learning"
            >>> print(result["metadata"]["total_latency_ms"])
            1.2
        """
        start_time = time.perf_counter()

        metadata: dict[str, object] = {
            "rewrite_latency_ms": 0.0,
            "expansion_latency_ms": 0.0,
            "intent_latency_ms": 0.0,
            "total_latency_ms": 0.0,
        }

        result: dict[str, object] = {
            "original_query": query,
            "processed_query": query,
            "rewritten_query": None,
            "expanded_query": None,
            "intent": None,
            "metadata": metadata,
        }

        current_query = query

        # Stage 1: Rewrite query
        if self.query_settings.enable_rewriting and self.rewriter:
            rewritten, rewrite_meta = self.rewriter.rewrite(current_query)
            result["rewritten_query"] = rewritten
            metadata["rewrite_latency_ms"] = rewrite_meta["latency_ms"]
            metadata["rewrites_applied"] = rewrite_meta["rewrites_applied"]
            current_query = rewritten

        # Stage 2: Expand with synonyms
        if self.query_settings.enable_synonyms and self.expander:
            expanded, expand_meta = self.expander.expand(current_query)
            result["expanded_query"] = expanded
            metadata["expansion_latency_ms"] = expand_meta["latency_ms"]
            metadata["terms_expanded"] = expand_meta["terms_expanded"]
            metadata["synonyms_added"] = expand_meta["synonyms_added"]
            current_query = expanded

        # Stage 3: Classify intent (optional)
        if self.query_settings.enable_intent_classification:
            intent_start = time.perf_counter()
            intent = self._classify_intent(query)
            intent_latency = (time.perf_counter() - intent_start) * 1000
            result["intent"] = intent
            metadata["intent_latency_ms"] = intent_latency

        result["processed_query"] = current_query

        total_latency = (time.perf_counter() - start_time) * 1000
        metadata["total_latency_ms"] = total_latency

        logger.info(f"Query processed: '{query}' → '{current_query}' ({total_latency:.2f}ms)")

        return result

    def _classify_intent(self, query: str) -> IntentType:
        """
        Classify query intent (simple rule-based).

        Args:
            query: Query string

        Returns:
            Intent classification

        Note:
            This is a simple rule-based classifier. For production,
            consider using a trained model.
        """
        query_lower = query.lower()

        # How-to queries
        if any(
            phrase in query_lower
            for phrase in ["how to", "how do", "how can", "steps to", "guide to"]
        ):
            return "howto"

        # Comparison queries
        if any(
            phrase in query_lower
            for phrase in [
                "vs",
                "versus",
                "compare",
                "difference between",
                "better than",
            ]
        ):
            return "comparison"

        # Troubleshooting queries
        if any(
            phrase in query_lower
            for phrase in [
                "error",
                "not working",
                "fix",
                "problem",
                "issue",
                "debug",
                "troubleshoot",
            ]
        ):
            return "troubleshooting"

        # Factual queries (what, when, where, who)
        if any(
            phrase in query_lower
            for phrase in ["what is", "what are", "when", "where", "who is", "define"]
        ):
            return "factual"

        # Default: exploratory
        return "exploratory"

    def get_all_variations(self, query: str) -> list[str]:
        """
        Generate all query variations for multi-query retrieval.

        Args:
            query: Original query

        Returns:
            List of query variations (original + rewrites + expansions)

        Example:
            >>> qu = QueryUnderstandingClient()
            >>> variations = qu.get_all_variations("what is ML?")
            >>> for v in variations:
            ...     print(v)
            what is ML?
            what is machine learning?
            machine learning definition explanation
        """
        variations = [query]

        # Add rewrite variations
        if self.rewriter:
            rewrites = self.rewriter.get_rewrites(query)
            for rewrite in rewrites:
                if rewrite not in variations:
                    variations.append(rewrite)

        # Add expanded variations
        if self.expander:
            for base_query in variations.copy():
                expanded, _ = self.expander.expand(base_query)
                if expanded != base_query and expanded not in variations:
                    variations.append(expanded)

        return variations
