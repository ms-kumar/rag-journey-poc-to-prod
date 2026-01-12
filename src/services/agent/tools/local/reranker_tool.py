"""Reranker tool for improving retrieval relevance."""

import logging
from typing import Any

from src.services.agent.tools.base import BaseTool, ToolCategory, ToolMetadata

logger = logging.getLogger(__name__)


class RerankerTool(BaseTool):
    """Tool for reranking retrieved documents."""

    def __init__(self, reranker_client, top_k: int = 3):
        """Initialize reranker tool.

        Args:
            reranker_client: Reranker client instance
            top_k: Number of documents to return after reranking
        """
        metadata = ToolMetadata(
            name="reranker",
            description="Rerank retrieved documents to improve relevance using cross-encoder models",
            category=ToolCategory.LOCAL,
            capabilities=[
                "document reranking",
                "relevance scoring",
                "result refinement",
                "cross-encoder",
            ],
            cost_per_call=0.0,  # Local, no cost
            avg_latency_ms=200.0,
            success_rate=0.93,
            requires_api_key=False,
        )
        super().__init__(metadata)
        self.reranker = reranker_client
        self.top_k = top_k

    async def execute(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Execute reranking on documents.

        Args:
            query: User query
            **kwargs: Required parameters:
                - documents: List of documents to rerank
                Optional:
                - top_k: Override default top_k

        Returns:
            Dictionary with reranked results
        """
        try:
            if not self.validate_input(query, **kwargs):
                return {
                    "success": False,
                    "result": None,
                    "error": "Invalid input parameters",
                    "metadata": {},
                }

            # Extract documents
            documents = kwargs.get("documents")
            if not documents:
                return {
                    "success": False,
                    "result": None,
                    "error": "No documents provided for reranking",
                    "metadata": {},
                }

            top_k = kwargs.get("top_k", self.top_k)

            self.logger.info(f"Reranking {len(documents)} documents for query: {query[:100]}...")

            # Extract text content from documents
            if isinstance(documents[0], dict):
                texts = [doc.get("content", str(doc)) for doc in documents]
            elif hasattr(documents[0], "page_content"):
                texts = [doc.page_content for doc in documents]
            else:
                texts = [str(doc) for doc in documents]

            # Perform reranking
            if hasattr(self.reranker, "rerank"):
                # Use reranker's rerank method
                reranked_results = self.reranker.rerank(
                    query=query,
                    documents=texts,
                    top_k=top_k,
                )
            elif hasattr(self.reranker, "rank"):
                # Alternative method name
                reranked_results = self.reranker.rank(
                    query=query,
                    documents=texts,
                    top_k=top_k,
                )
            else:
                raise AttributeError("Reranker client does not have rerank or rank method")

            # Format results
            reranked_documents = []
            for idx, result in enumerate(reranked_results[:top_k]):
                if isinstance(result, dict):
                    score = result.get("score", result.get("relevance_score", 1.0))
                    doc_idx = result.get("index", idx)
                else:
                    score = 1.0 - (idx / len(reranked_results))  # Fallback scoring
                    doc_idx = idx

                # Get original document
                original_doc = documents[doc_idx] if doc_idx < len(documents) else documents[idx]

                if isinstance(original_doc, dict):
                    reranked_documents.append(
                        {
                            "content": original_doc.get("content", texts[doc_idx]),
                            "metadata": original_doc.get("metadata", {}),
                            "rerank_score": float(score),
                            "original_rank": doc_idx,
                        }
                    )
                else:
                    reranked_documents.append(
                        {
                            "content": texts[doc_idx],
                            "metadata": {},
                            "rerank_score": float(score),
                            "original_rank": doc_idx,
                        }
                    )

            self.logger.info(f"Reranked to top {len(reranked_documents)} documents")

            return {
                "success": True,
                "result": {
                    "documents": reranked_documents,
                    "count": len(reranked_documents),
                    "query": query,
                },
                "error": None,
                "metadata": {
                    "original_count": len(documents),
                    "top_k": top_k,
                },
            }

        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "metadata": {},
            }
