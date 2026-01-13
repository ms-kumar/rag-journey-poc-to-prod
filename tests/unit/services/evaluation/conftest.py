"""Pytest configuration for evaluation tests."""

import pytest


@pytest.fixture
def sample_query():
    """Provide a sample query."""
    return "What is machine learning?"


@pytest.fixture
def sample_context():
    """Provide sample context documents."""
    return [
        "Machine learning is a subset of AI that learns from data.",
        "It uses statistical techniques to improve performance.",
        "Applications include image recognition and NLP.",
    ]


@pytest.fixture
def sample_response():
    """Provide a sample response."""
    return "Machine learning is a subset of artificial intelligence that enables systems to learn from data."


@pytest.fixture
def sample_ground_truth():
    """Provide ground truth answer."""
    return "Machine learning is a branch of AI focused on learning from data."


@pytest.fixture
def evaluation_dataset():
    """Provide an evaluation dataset."""
    return [
        {
            "query": "What is Python?",
            "context": ["Python is a programming language."],
            "response": "Python is a high-level programming language.",
            "ground_truth": "Python is a programming language.",
        },
        {
            "query": "What is FastAPI?",
            "context": ["FastAPI is a web framework."],
            "response": "FastAPI is a modern web framework.",
            "ground_truth": "FastAPI is a web framework for APIs.",
        },
    ]
