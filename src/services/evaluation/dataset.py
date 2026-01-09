"""
Evaluation dataset management for RAG system.

Handles loading, creating, and managing evaluation datasets with
query-document relevance judgments.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvalExample:
    """Single evaluation example with query and relevance judgments."""

    query: str
    relevant_doc_ids: list[str]
    query_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    expected_answer: str | None = None
    context_docs: list[str] = field(default_factory=list)  # Optional context for generation eval

    def __post_init__(self):
        """Generate query_id if not provided."""
        if not self.query_id:
            self.query_id = f"q_{hash(self.query) % 1000000:06d}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalExample":
        """Create from dictionary format."""
        return cls(**data)


@dataclass
class EvalDataset:
    """Collection of evaluation examples."""

    examples: list[EvalExample] = field(default_factory=list)
    name: str = "default"
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def add_example(
        self,
        query: str,
        relevant_doc_ids: list[str],
        expected_answer: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add an evaluation example to the dataset.

        Args:
            query: The query text
            relevant_doc_ids: List of relevant document IDs
            expected_answer: Optional expected answer for generation eval
            metadata: Optional metadata for the example
        """
        example = EvalExample(
            query=query,
            relevant_doc_ids=relevant_doc_ids,
            expected_answer=expected_answer,
            metadata=metadata or {},
        )
        self.examples.append(example)

    def save(self, filepath: Path | str) -> None:
        """
        Save dataset to JSON file.

        Args:
            filepath: Path to save the dataset
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "examples": [ex.to_dict() for ex in self.examples],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved dataset with {len(self.examples)} examples to {filepath}")

    @classmethod
    def load(cls, filepath: Path | str) -> "EvalDataset":
        """
        Load dataset from JSON file.

        Args:
            filepath: Path to the dataset file

        Returns:
            Loaded EvalDataset
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        with open(filepath) as f:
            data = json.load(f)

        examples = [EvalExample.from_dict(ex) for ex in data.get("examples", [])]

        return cls(
            examples=examples,
            name=data.get("name", "default"),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )

    def split(self, train_ratio: float = 0.8) -> tuple["EvalDataset", "EvalDataset"]:
        """
        Split dataset into train and test sets.

        Args:
            train_ratio: Ratio of examples for training (0-1)

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        import random

        examples_copy = self.examples.copy()
        random.shuffle(examples_copy)

        split_idx = int(len(examples_copy) * train_ratio)
        train_examples = examples_copy[:split_idx]
        test_examples = examples_copy[split_idx:]

        train_dataset = EvalDataset(
            examples=train_examples,
            name=f"{self.name}_train",
            description=f"Train split of {self.name}",
            metadata={**self.metadata, "split": "train"},
        )

        test_dataset = EvalDataset(
            examples=test_examples,
            name=f"{self.name}_test",
            description=f"Test split of {self.name}",
            metadata={**self.metadata, "split": "test"},
        )

        return train_dataset, test_dataset

    def get_statistics(self) -> dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dict with dataset statistics
        """
        if not self.examples:
            return {
                "num_examples": 0,
                "avg_relevant_docs": 0.0,
                "queries_with_answers": 0,
            }

        num_relevant = [len(ex.relevant_doc_ids) for ex in self.examples]
        queries_with_answers = sum(
            1 for ex in self.examples if ex.expected_answer is not None
        )

        return {
            "num_examples": len(self.examples),
            "avg_relevant_docs": sum(num_relevant) / len(num_relevant),
            "min_relevant_docs": min(num_relevant),
            "max_relevant_docs": max(num_relevant),
            "queries_with_answers": queries_with_answers,
        }


class DatasetBuilder:
    """Builder for creating evaluation datasets programmatically."""

    def __init__(self, name: str = "default", description: str = ""):
        self.dataset = EvalDataset(name=name, description=description)

    def add_from_logs(
        self,
        log_filepath: Path | str,
        relevance_threshold: float = 0.8,
    ) -> "DatasetBuilder":
        """
        Add examples from query logs with relevance feedback.

        Args:
            log_filepath: Path to query log file
            relevance_threshold: Threshold for considering a document relevant

        Returns:
            Self for chaining
        """
        # This is a placeholder for log parsing logic
        # In practice, you'd parse actual log files
        logger.info(f"Loading examples from logs: {log_filepath}")
        return self

    def add_synthetic_queries(
        self,
        num_queries: int = 100,
        topics: list[str] | None = None,
    ) -> "DatasetBuilder":
        """
        Generate synthetic evaluation queries for testing.

        Args:
            num_queries: Number of queries to generate
            topics: Optional list of topics to focus on

        Returns:
            Self for chaining
        """
        topics = topics or ["python", "machine learning", "rag", "embeddings"]

        synthetic_queries = [
            ("What is RAG?", ["doc_rag_basics"]),
            ("Explain vector embeddings", ["doc_embeddings"]),
            ("How to implement BM25?", ["doc_bm25", "doc_sparse"]),
            ("FastAPI best practices", ["doc_fastapi"]),
            ("Python async programming", ["doc_python_async"]),
        ]

        for query, relevant_ids in synthetic_queries[:num_queries]:
            self.dataset.add_example(query=query, relevant_doc_ids=relevant_ids)

        logger.info(f"Added {len(synthetic_queries)} synthetic examples")
        return self

    def build(self) -> EvalDataset:
        """
        Build and return the dataset.

        Returns:
            Completed EvalDataset
        """
        return self.dataset
