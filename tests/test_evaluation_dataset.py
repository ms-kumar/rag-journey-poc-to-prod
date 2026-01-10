"""
Tests for evaluation dataset management.
"""

from src.services.evaluation.dataset import DatasetBuilder, EvalDataset, EvalExample


class TestEvalExample:
    """Test EvalExample class."""

    def test_create_example(self):
        """Test creating an evaluation example."""
        example = EvalExample(
            query="What is RAG?",
            relevant_doc_ids=["doc1", "doc2"],
            expected_answer="RAG is...",
        )

        assert example.query == "What is RAG?"
        assert len(example.relevant_doc_ids) == 2
        assert example.expected_answer == "RAG is..."
        assert example.query_id  # Auto-generated

    def test_to_dict(self):
        """Test conversion to dictionary."""
        example = EvalExample(
            query="Test query",
            relevant_doc_ids=["doc1"],
            metadata={"category": "test"},
        )

        result = example.to_dict()

        assert result["query"] == "Test query"
        assert result["relevant_doc_ids"] == ["doc1"]
        assert result["metadata"]["category"] == "test"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "query": "Test query",
            "relevant_doc_ids": ["doc1"],
            "query_id": "q_123",
            "metadata": {},
            "expected_answer": None,
            "context_docs": [],
        }

        example = EvalExample.from_dict(data)

        assert example.query == "Test query"
        assert example.query_id == "q_123"


class TestEvalDataset:
    """Test EvalDataset class."""

    def test_create_empty_dataset(self):
        """Test creating an empty dataset."""
        dataset = EvalDataset(name="test", description="Test dataset")

        assert len(dataset) == 0
        assert dataset.name == "test"

    def test_add_example(self):
        """Test adding examples to dataset."""
        dataset = EvalDataset()
        dataset.add_example(
            query="Test query",
            relevant_doc_ids=["doc1"],
        )

        assert len(dataset) == 1
        assert dataset.examples[0].query == "Test query"

    def test_save_and_load(self, tmp_path):
        """Test saving and loading dataset."""
        dataset = EvalDataset(name="test")
        dataset.add_example("Query 1", ["doc1"])
        dataset.add_example("Query 2", ["doc2"])

        # Save
        filepath = tmp_path / "test_dataset.json"
        dataset.save(filepath)

        assert filepath.exists()

        # Load
        loaded = EvalDataset.load(filepath)

        assert len(loaded) == 2
        assert loaded.name == "test"
        assert loaded.examples[0].query == "Query 1"

    def test_split_dataset(self):
        """Test splitting dataset into train/test."""
        dataset = EvalDataset(name="test")
        for i in range(10):
            dataset.add_example(f"Query {i}", [f"doc{i}"])

        train, test = dataset.split(train_ratio=0.8)

        assert len(train) + len(test) == 10
        assert len(train) == 8
        assert len(test) == 2

    def test_get_statistics(self):
        """Test dataset statistics."""
        dataset = EvalDataset()
        dataset.add_example("Q1", ["doc1", "doc2"])
        dataset.add_example("Q2", ["doc3"], expected_answer="Answer")

        stats = dataset.get_statistics()

        assert stats["num_examples"] == 2
        assert stats["avg_relevant_docs"] == 1.5
        assert stats["queries_with_answers"] == 1


class TestDatasetBuilder:
    """Test DatasetBuilder class."""

    def test_create_builder(self):
        """Test creating a dataset builder."""
        builder = DatasetBuilder(name="test", description="Test dataset")

        assert builder.dataset.name == "test"

    def test_add_synthetic_queries(self):
        """Test adding synthetic queries."""
        builder = DatasetBuilder()
        builder.add_synthetic_queries(num_queries=3)

        assert len(builder.dataset) >= 3

    def test_build(self):
        """Test building the dataset."""
        builder = DatasetBuilder(name="test")
        builder.add_synthetic_queries(num_queries=5)

        dataset = builder.build()

        assert isinstance(dataset, EvalDataset)
        assert dataset.name == "test"
        assert len(dataset) >= 5
