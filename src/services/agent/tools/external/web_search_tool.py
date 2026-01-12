"""Web search tool using Tavily or DuckDuckGo."""

import logging
from typing import Any

from src.services.agent.tools.base import BaseTool, ToolCategory, ToolMetadata

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """Tool for searching the web."""

    def __init__(self, api_key: str | None = None, max_results: int = 5):
        """Initialize web search tool.

        Args:
            api_key: Optional Tavily API key (uses DuckDuckGo if not provided)
            max_results: Maximum number of results to return
        """
        metadata = ToolMetadata(
            name="web_search",
            description="Search the web for current information, news, and real-time data",
            category=ToolCategory.EXTERNAL,
            capabilities=[
                "web search",
                "current events",
                "real-time information",
                "news search",
                "internet lookup",
            ],
            cost_per_call=0.001,  # Tavily pricing
            avg_latency_ms=800.0,
            success_rate=0.88,
            requires_api_key=api_key is not None,
        )
        super().__init__(metadata)
        self.api_key = api_key
        self.max_results = max_results
        self._search_client = None

    def _get_search_client(self):
        """Lazy initialize search client."""
        if self._search_client is not None:
            return self._search_client

        if self.api_key:
            # Use Tavily if API key provided
            try:
                from tavily import TavilyClient

                self._search_client = TavilyClient(api_key=self.api_key)
                self.logger.info("Initialized Tavily search client")
                return self._search_client
            except ImportError:
                self.logger.warning("Tavily not installed, falling back to DuckDuckGo")

        # Fallback to DuckDuckGo (free, no API key)
        try:
            from langchain_community.tools import DuckDuckGoSearchResults

            self._search_client = DuckDuckGoSearchResults(max_results=self.max_results)
            self.logger.info("Initialized DuckDuckGo search client")
            return self._search_client
        except ImportError:
            raise ImportError(
                "Neither Tavily nor DuckDuckGo search is available. "
                "Install with: pip install tavily-python or duckduckgo-search"
            )

    async def execute(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Execute web search.

        Args:
            query: Search query
            **kwargs: Optional parameters:
                - max_results: Override default max_results
                - search_depth: "basic" or "advanced" (Tavily only)

        Returns:
            Dictionary with search results
        """
        try:
            if not self.validate_input(query, **kwargs):
                return {
                    "success": False,
                    "result": None,
                    "error": "Invalid input parameters",
                    "metadata": {},
                }

            max_results = kwargs.get("max_results", self.max_results)
            search_depth = kwargs.get("search_depth", "basic")

            self.logger.info(f"Searching web for: {query[:100]}...")

            search_client = self._get_search_client()

            # Execute search based on client type
            if hasattr(search_client, "search"):
                # Tavily client
                response = search_client.search(
                    query=query,
                    max_results=max_results,
                    search_depth=search_depth,
                )
                results = response.get("results", [])

                # Format Tavily results
                formatted_results = [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                        "score": r.get("score", 1.0),
                    }
                    for r in results
                ]
            else:
                # DuckDuckGo client
                response = search_client.run(query)

                # Parse DuckDuckGo response (format varies)
                if isinstance(response, str):
                    # Simple text response
                    formatted_results = [
                        {
                            "title": "DuckDuckGo Search Result",
                            "url": "",
                            "content": response,
                            "score": 1.0,
                        }
                    ]
                elif isinstance(response, list):
                    # List of results
                    formatted_results = [
                        {
                            "title": r.get("title", "") if isinstance(r, dict) else "",
                            "url": r.get("link", "") if isinstance(r, dict) else "",
                            "content": r.get("snippet", str(r)) if isinstance(r, dict) else str(r),
                            "score": 1.0,
                        }
                        for r in response[:max_results]
                    ]
                else:
                    formatted_results = [
                        {"title": "", "url": "", "content": str(response), "score": 1.0}
                    ]

            self.logger.info(f"Found {len(formatted_results)} web results")

            return {
                "success": True,
                "result": {
                    "results": formatted_results,
                    "count": len(formatted_results),
                    "query": query,
                },
                "error": None,
                "metadata": {
                    "max_results": max_results,
                    "search_engine": "tavily" if self.api_key else "duckduckgo",
                },
            }

        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "metadata": {},
            }
