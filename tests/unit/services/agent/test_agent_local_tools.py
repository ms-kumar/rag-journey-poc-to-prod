"""Unit tests for agent local tools."""

import pytest

from src.services.agent.tools.local.generator_tool import GeneratorTool
from src.services.agent.tools.local.reranker_tool import RerankerTool
from src.services.agent.tools.local.vectordb_tool import VectorDBTool


class MockVectorStore:
    """Mock vectorstore for testing."""

    def similarity_search_with_score(self, query, k=5, filter=None):
        """Mock similarity search with scores."""
        # Return mock documents
        docs = []
        for i in range(k):
            doc = type(
                "Document",
                (),
                {"page_content": f"Document {i} about {query}", "metadata": {"source": f"doc_{i}"}},
            )()
            docs.append((doc, 0.9 - (i * 0.1)))
        return docs

    def similarity_search(self, query, k=5, filter=None):
        """Mock similarity search without scores."""
        docs = []
        for i in range(k):
            doc = type(
                "Document",
                (),
                {"page_content": f"Document {i} about {query}", "metadata": {"source": f"doc_{i}"}},
            )()
            docs.append(doc)
        return docs


class MockReranker:
    """Mock reranker for testing."""

    def rerank(self, query, documents, top_k=3):
        """Mock reranking."""
        # Return top_k documents with scores
        results = []
        for i in range(min(top_k, len(documents))):
            results.append(
                {
                    "index": i,
                    "score": 0.95 - (i * 0.05),
                }
            )
        return results


class MockGenerator:
    """Mock generator for testing."""

    def generate(self, prompt, max_length=512, temperature=0.7):
        """Mock generation."""
        return f"Generated response for: {prompt[:50]}"

    def __call__(self, prompt, max_length=512, temperature=0.7, do_sample=True):
        """Mock callable generation."""
        return [{"generated_text": f"{prompt}\n\nGenerated response."}]


class TestVectorDBTool:
    """Test VectorDBTool."""

    @pytest.mark.asyncio
    async def test_vectordb_tool_creation(self):
        """Test creating VectorDB tool."""
        vectorstore = MockVectorStore()
        tool = VectorDBTool(vectorstore, top_k=5)

        assert tool.metadata.name == "vectordb_retrieval"
        assert tool.metadata.category.value == "local"
        assert tool.top_k == 5

    @pytest.mark.asyncio
    async def test_vectordb_execute_success(self):
        """Test successful VectorDB execution."""
        vectorstore = MockVectorStore()
        tool = VectorDBTool(vectorstore, top_k=3)

        result = await tool.execute("machine learning")

        assert result["success"] is True
        assert result["error"] is None
        assert "documents" in result["result"]
        assert len(result["result"]["documents"]) == 3

    @pytest.mark.asyncio
    async def test_vectordb_execute_with_filters(self):
        """Test VectorDB execution with filters."""
        vectorstore = MockVectorStore()
        tool = VectorDBTool(vectorstore)

        result = await tool.execute(
            "test query",
            filters={"source": "test"},
        )

        assert result["success"] is True
        assert result["metadata"]["has_filters"] is True

    @pytest.mark.asyncio
    async def test_vectordb_execute_custom_top_k(self):
        """Test VectorDB execution with custom top_k."""
        vectorstore = MockVectorStore()
        tool = VectorDBTool(vectorstore, top_k=5)

        result = await tool.execute("test query", top_k=2)

        assert len(result["result"]["documents"]) == 2

    @pytest.mark.asyncio
    async def test_vectordb_execute_invalid_input(self):
        """Test VectorDB execution with invalid input."""
        vectorstore = MockVectorStore()
        tool = VectorDBTool(vectorstore)

        result = await tool.execute("")

        assert result["success"] is False
        assert "Invalid input" in result["error"]


class TestRerankerTool:
    """Test RerankerTool."""

    @pytest.mark.asyncio
    async def test_reranker_tool_creation(self):
        """Test creating Reranker tool."""
        reranker = MockReranker()
        tool = RerankerTool(reranker, top_k=3)

        assert tool.metadata.name == "reranker"
        assert tool.metadata.category.value == "local"
        assert tool.top_k == 3

    @pytest.mark.asyncio
    async def test_reranker_execute_success(self):
        """Test successful reranker execution."""
        reranker = MockReranker()
        tool = RerankerTool(reranker, top_k=2)

        documents = [
            {"content": "Doc 1"},
            {"content": "Doc 2"},
            {"content": "Doc 3"},
        ]

        result = await tool.execute("test query", documents=documents)

        assert result["success"] is True
        assert "documents" in result["result"]
        assert len(result["result"]["documents"]) == 2

    @pytest.mark.asyncio
    async def test_reranker_execute_no_documents(self):
        """Test reranker execution without documents."""
        reranker = MockReranker()
        tool = RerankerTool(reranker)

        result = await tool.execute("test query")

        assert result["success"] is False
        assert "No documents provided" in result["error"]

    @pytest.mark.asyncio
    async def test_reranker_execute_with_doc_objects(self):
        """Test reranker with document objects."""
        reranker = MockReranker()
        tool = RerankerTool(reranker, top_k=2)

        # Create mock document objects
        docs = []
        for i in range(3):
            doc = type("Document", (), {"page_content": f"Document {i}", "metadata": {}})()
            docs.append(doc)

        result = await tool.execute("test query", documents=docs)

        assert result["success"] is True
        assert len(result["result"]["documents"]) == 2


class TestGeneratorTool:
    """Test GeneratorTool."""

    @pytest.mark.asyncio
    async def test_generator_tool_creation(self):
        """Test creating Generator tool."""
        generator = MockGenerator()
        tool = GeneratorTool(generator)

        assert tool.metadata.name == "generator"
        assert tool.metadata.category.value == "local"

    @pytest.mark.asyncio
    async def test_generator_execute_success(self):
        """Test successful generator execution."""
        generator = MockGenerator()
        tool = GeneratorTool(generator)

        result = await tool.execute("What is machine learning?")

        assert result["success"] is True
        assert "response" in result["result"]
        assert len(result["result"]["response"]) > 0

    @pytest.mark.asyncio
    async def test_generator_execute_with_context(self):
        """Test generator execution with context."""
        generator = MockGenerator()
        tool = GeneratorTool(generator)

        context = [
            {"content": "Machine learning is..."},
            {"content": "Deep learning is..."},
        ]

        result = await tool.execute(
            "What is machine learning?",
            context=context,
        )

        assert result["success"] is True
        assert result["result"]["has_context"] is True

    @pytest.mark.asyncio
    async def test_generator_execute_custom_params(self):
        """Test generator with custom parameters."""
        generator = MockGenerator()
        tool = GeneratorTool(generator)

        result = await tool.execute(
            "Test query",
            max_length=256,
            temperature=0.5,
        )

        assert result["success"] is True
        assert result["metadata"]["max_length"] == 256
        assert result["metadata"]["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_generator_execute_invalid_input(self):
        """Test generator execution with invalid input."""
        generator = MockGenerator()
        tool = GeneratorTool(generator)

        result = await tool.execute("")

        assert result["success"] is False
        assert "Invalid input" in result["error"]
