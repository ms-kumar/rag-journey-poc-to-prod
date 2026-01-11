"""
Integration tests for pipeline with cross-encoder re-ranking.
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from src.schemas.api.rag_request import GenerateRequest
from src.services.pipeline.naive_pipeline.client import NaivePipeline, NaivePipelineConfig


class TestPipelineIntegration:
    """Test pipeline integration with re-ranking."""

    @pytest.fixture
    def mock_components(self):
        """Mock all pipeline components."""
        with (
            patch(
                "src.services.pipeline.naive_pipeline.client.get_ingestion_client"
            ) as mock_ingest,
            patch("src.services.pipeline.naive_pipeline.client.get_chunking_client") as mock_chunk,
            patch("src.services.pipeline.naive_pipeline.client.get_embed_client") as mock_embed,
            patch(
                "src.services.pipeline.naive_pipeline.client.get_langchain_embeddings_adapter"
            ) as mock_adapter,
            patch(
                "src.services.pipeline.naive_pipeline.client.get_vectorstore_client"
            ) as mock_vectorstore,
            patch("src.services.pipeline.naive_pipeline.client.get_generator") as mock_gen,
            patch("src.services.pipeline.naive_pipeline.client.get_reranker") as mock_reranker,
        ):
            # Setup mock returns
            mock_ingest.return_value = Mock()
            mock_chunk.return_value = Mock()
            mock_embed.return_value = Mock()
            mock_adapter.return_value = Mock()
            mock_vectorstore.return_value = Mock()
            mock_gen.return_value = Mock()
            mock_reranker.return_value = Mock()

            yield {
                "ingest": mock_ingest.return_value,
                "chunk": mock_chunk.return_value,
                "embed": mock_embed.return_value,
                "adapter": mock_adapter.return_value,
                "vectorstore": mock_vectorstore.return_value,
                "generator": mock_gen.return_value,
                "reranker": mock_reranker.return_value,
            }

    def test_pipeline_without_reranker(self, mock_components):
        """Test pipeline initialization without re-ranker."""
        config = NaivePipelineConfig(enable_reranker=False)
        pipeline = NaivePipeline(config)

        assert pipeline.reranker is None
        assert pipeline.config.enable_reranker is False

    def test_pipeline_with_reranker(self, mock_components):
        """Test pipeline initialization with re-ranker enabled."""
        config = NaivePipelineConfig(
            enable_reranker=True, reranker_model="test-model", reranker_batch_size=16
        )
        pipeline = NaivePipeline(config)

        assert pipeline.reranker is not None
        assert pipeline.config.enable_reranker is True

    def test_retrieve_without_reranking(self, mock_components):
        """Test retrieval without re-ranking."""
        sample_docs = [
            Document(page_content="doc1", metadata={"id": "1", "score": 0.8}),
            Document(page_content="doc2", metadata={"id": "2", "score": 0.7}),
        ]

        config = NaivePipelineConfig(enable_reranker=False)
        pipeline = NaivePipeline(config)

        # Mock vectorstore to return sample docs
        mock_components["vectorstore"].similarity_search.return_value = sample_docs

        result = pipeline.retrieve("test query", k=2)

        assert len(result) == 2
        assert result == sample_docs

        # Verify reranker was not called
        mock_components["reranker"].rerank.assert_not_called()

    def test_retrieve_with_reranking(self, mock_components):
        """Test retrieval with re-ranking enabled."""
        original_docs = [
            Document(page_content="doc1", metadata={"id": "1", "score": 0.8}),
            Document(page_content="doc2", metadata={"id": "2", "score": 0.7}),
            Document(page_content="doc3", metadata={"id": "3", "score": 0.6}),
        ]

        reranked_docs = [
            Document(page_content="doc2", metadata={"id": "2", "score": 0.7, "reranked": True}),
            Document(page_content="doc1", metadata={"id": "1", "score": 0.8, "reranked": True}),
        ]

        config = NaivePipelineConfig(enable_reranker=True)
        pipeline = NaivePipeline(config)

        # Mock vectorstore to return more docs for reranking
        mock_components["vectorstore"].similarity_search.return_value = original_docs

        # Mock reranker result
        from src.services.reranker.client import RerankResult

        mock_rerank_result = RerankResult(
            documents=reranked_docs,
            scores=[0.95, 0.90],
            original_ranks=[1, 0],
            execution_time=0.1,
            model_used="test-model",
            fallback_used=False,
        )
        mock_components["reranker"].rerank.return_value = mock_rerank_result

        result = pipeline.retrieve("test query", k=2)

        assert len(result) == 2

        # Verify reranker was called with more documents
        mock_components["reranker"].rerank.assert_called_once()
        call_args = mock_components["reranker"].rerank.call_args
        assert call_args[1]["query"] == "test query"
        assert len(call_args[1]["documents"]) == 3  # Retrieved more for reranking
        assert call_args[1]["top_k"] == 2

        # Check that returned documents have reranking metadata
        for doc in result:
            assert doc.metadata.get("reranked") is True

    def test_retrieve_with_reranking_override(self, mock_components):
        """Test retrieval with re-ranking override parameter."""
        sample_docs = [
            Document(page_content="doc1", metadata={"id": "1", "score": 0.8}),
        ]

        config = NaivePipelineConfig(enable_reranker=False)  # Disabled by default
        pipeline = NaivePipeline(config)

        mock_components["vectorstore"].similarity_search.return_value = sample_docs

        # Override to enable reranking for this query
        result = pipeline.retrieve("test query", k=1, enable_reranking=True)

        # Should still not rerank because reranker is None
        assert len(result) == 1
        mock_components["reranker"].rerank.assert_not_called()

    def test_retrieve_different_search_types_with_reranking(self, mock_components):
        """Test re-ranking works with different search types."""
        sample_docs = [Document(page_content="doc1", metadata={"id": "1"})]

        config = NaivePipelineConfig(enable_reranker=True)
        pipeline = NaivePipeline(config)

        # Mock all search methods
        mock_components["vectorstore"].similarity_search.return_value = sample_docs
        mock_components["vectorstore"].bm25_search.return_value = sample_docs
        mock_components["vectorstore"].hybrid_search.return_value = sample_docs
        mock_components["vectorstore"].sparse_search.return_value = sample_docs

        # Mock reranker
        from src.services.reranker.client import RerankResult

        mock_result = RerankResult(
            documents=sample_docs,
            scores=[0.9],
            original_ranks=[0],
            execution_time=0.1,
            model_used="test-model",
            fallback_used=False,
        )
        mock_components["reranker"].rerank.return_value = mock_result

        # Test each search type
        search_types = ["vector", "bm25", "hybrid", "sparse"]

        for search_type in search_types:
            result = pipeline.retrieve("test query", k=1, search_type=search_type)
            assert len(result) == 1

        # Verify reranker was called for each search type
        assert mock_components["reranker"].rerank.call_count == len(search_types)


class TestAPIIntegration:
    """Test API integration with re-ranking."""

    @patch("src.api.v1.endpoints.rag._get_pipeline")
    def test_api_generate_with_reranking(self, mock_get_pipeline):
        """Test API generate endpoint with re-ranking."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.retrieve.return_value = [
            Document(page_content="test content", metadata={"reranked": True})
        ]
        mock_pipeline.generate.return_value = "test answer"
        mock_get_pipeline.return_value = mock_pipeline

        # Import here to avoid import issues
        from src.api.v1.endpoints.rag import generate

        # Create request with re-ranking enabled
        request = GenerateRequest(prompt="test prompt", top_k=5, enable_reranking=True)

        # Mock the async context
        import asyncio

        async def test_generate():
            response = await generate(request)
            return response

        # Run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(test_generate())

            # Verify pipeline.retrieve was called with reranking enabled
            mock_pipeline.retrieve.assert_called_once()
            call_kwargs = mock_pipeline.retrieve.call_args[1]
            assert call_kwargs["enable_reranking"] is True

            assert response.answer == "test answer"

        finally:
            loop.close()


if __name__ == "__main__":
    pytest.main([__file__])
