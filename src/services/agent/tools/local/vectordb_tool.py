"""VectorDB retrieval tool for RAG."""

import logging
from typing import Any

from src.services.agent.tools.base import BaseTool, ToolCategory, ToolMetadata

logger = logging.getLogger(__name__)


class VectorDBTool(BaseTool):
    """Tool for retrieving documents from vector database."""
    
    def __init__(self, vectorstore_client, top_k: int = 5):
        """Initialize VectorDB tool.
        
        Args:
            vectorstore_client: Vectorstore client instance
            top_k: Number of documents to retrieve
        """
        metadata = ToolMetadata(
            name="vectordb_retrieval",
            description="Retrieve relevant documents from the local knowledge base using vector similarity search",
            category=ToolCategory.LOCAL,
            capabilities=[
                "document retrieval",
                "semantic search",
                "knowledge base lookup",
                "vector similarity",
                "metadata filtering",
            ],
            cost_per_call=0.0,  # Local, no cost
            avg_latency_ms=150.0,
            success_rate=0.95,
            requires_api_key=False,
        )
        super().__init__(metadata)
        self.vectorstore = vectorstore_client
        self.top_k = top_k
    
    async def execute(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Execute vector similarity search.
        
        Args:
            query: Search query
            **kwargs: Optional parameters:
                - top_k: Override default top_k
                - filters: Metadata filters
                - score_threshold: Minimum similarity score
                
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
            
            # Extract parameters
            top_k = kwargs.get("top_k", self.top_k)
            filters = kwargs.get("filters")
            score_threshold = kwargs.get("score_threshold", 0.0)
            
            self.logger.info(f"Retrieving documents for query: {query[:100]}...")
            
            # Perform retrieval
            if hasattr(self.vectorstore, "similarity_search_with_score"):
                # Use score-based search if available
                results = self.vectorstore.similarity_search_with_score(
                    query,
                    k=top_k,
                    filter=filters,
                )
                
                # Filter by score threshold
                results = [(doc, score) for doc, score in results if score >= score_threshold]
                
                # Format results
                documents = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score),
                    }
                    for doc, score in results
                ]
            else:
                # Fallback to regular search
                docs = self.vectorstore.similarity_search(
                    query,
                    k=top_k,
                    filter=filters,
                )
                documents = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": 1.0,  # No score available
                    }
                    for doc in docs
                ]
            
            self.logger.info(f"Retrieved {len(documents)} documents")
            
            return {
                "success": True,
                "result": {
                    "documents": documents,
                    "count": len(documents),
                    "query": query,
                },
                "error": None,
                "metadata": {
                    "top_k": top_k,
                    "has_filters": filters is not None,
                    "score_threshold": score_threshold,
                },
            }
            
        except Exception as e:
            self.logger.error(f"VectorDB retrieval failed: {e}")
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "metadata": {},
            }
