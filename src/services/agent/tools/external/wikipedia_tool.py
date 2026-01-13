"""Wikipedia search tool."""

import logging
from typing import Any

from src.services.agent.tools.base import BaseTool, ToolCategory, ToolMetadata

logger = logging.getLogger(__name__)


class WikipediaTool(BaseTool):
    """Tool for searching Wikipedia."""

    def __init__(self, max_results: int = 3):
        """Initialize Wikipedia tool.

        Args:
            max_results: Maximum number of results to return
        """
        metadata = ToolMetadata(
            name="wikipedia",
            description="Search Wikipedia for factual information and encyclopedic knowledge",
            category=ToolCategory.EXTERNAL,
            capabilities=[
                "wikipedia search",
                "factual information",
                "encyclopedic knowledge",
                "definitions",
                "general knowledge",
            ],
            cost_per_call=0.0,  # Free API
            avg_latency_ms=600.0,
            success_rate=0.85,
            requires_api_key=False,
        )
        super().__init__(metadata)
        self.max_results = max_results
        self._wikipedia_client = None

    def _get_wikipedia_client(self):
        """Lazy initialize Wikipedia client."""
        if self._wikipedia_client is not None:
            return self._wikipedia_client

        try:
            import wikipedia  # type: ignore[import-untyped]

            self._wikipedia_client = wikipedia
            self.logger.info("Initialized Wikipedia client")
            return self._wikipedia_client
        except ImportError:
            raise ImportError(
                "Wikipedia library not installed. Install with: pip install wikipedia"
            )

    async def execute(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Execute Wikipedia search.

        Args:
            query: Search query
            **kwargs: Optional parameters:
                - max_results: Override default max_results
                - sentences: Number of sentences to return from summary

        Returns:
            Dictionary with Wikipedia results
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
            sentences = kwargs.get("sentences", 3)

            self.logger.info(f"Searching Wikipedia for: {query[:100]}...")

            wikipedia = self._get_wikipedia_client()

            # Search for pages
            search_results = wikipedia.search(query, results=max_results)

            if not search_results:
                return {
                    "success": True,
                    "result": {
                        "results": [],
                        "count": 0,
                        "query": query,
                    },
                    "error": None,
                    "metadata": {"message": "No Wikipedia results found"},
                }

            # Get summaries for top results
            results = []
            for title in search_results[:max_results]:
                try:
                    # Get page summary
                    summary = wikipedia.summary(title, sentences=sentences)
                    page = wikipedia.page(title)

                    results.append(
                        {
                            "title": title,
                            "url": page.url,
                            "summary": summary,
                            "full_text": page.content[:2000],  # First 2000 chars
                        }
                    )
                except wikipedia.exceptions.DisambiguationError as e:
                    # Handle disambiguation pages
                    self.logger.warning(f"Disambiguation page for '{title}': {e.options[:3]}")
                    # Try first option
                    if e.options:
                        try:
                            summary = wikipedia.summary(e.options[0], sentences=sentences)
                            page = wikipedia.page(e.options[0])
                            results.append(
                                {
                                    "title": e.options[0],
                                    "url": page.url,
                                    "summary": summary,
                                    "full_text": page.content[:2000],
                                }
                            )
                        except Exception:  # nosec B112
                            continue
                except wikipedia.exceptions.PageError:
                    self.logger.warning(f"Page not found: {title}")
                    continue
                except Exception as e:
                    self.logger.warning(f"Error fetching page '{title}': {e}")
                    continue

            self.logger.info(f"Found {len(results)} Wikipedia articles")

            return {
                "success": True,
                "result": {
                    "results": results,
                    "count": len(results),
                    "query": query,
                },
                "error": None,
                "metadata": {
                    "max_results": max_results,
                    "sentences": sentences,
                },
            }

        except Exception as e:
            self.logger.error(f"Wikipedia search failed: {e}")
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "metadata": {},
            }
