"""Evaluation service schemas."""

from pydantic import BaseModel, Field


class EvaluationMetrics(BaseModel):
    """Evaluation metrics for RAG system."""

    retrieval_precision: float = Field(..., description="Retrieval precision", ge=0.0, le=1.0)
    retrieval_recall: float = Field(..., description="Retrieval recall", ge=0.0, le=1.0)
    retrieval_f1: float = Field(..., description="Retrieval F1 score", ge=0.0, le=1.0)
    generation_quality: float = Field(..., description="Generation quality score", ge=0.0, le=1.0)
    answer_relevance: float = Field(..., description="Answer relevance score", ge=0.0, le=1.0)
    faithfulness: float = Field(..., description="Faithfulness to context", ge=0.0, le=1.0)


class EvaluationResult(BaseModel):
    """Single evaluation result."""

    query: str = Field(..., description="Query that was evaluated")
    expected_answer: str | None = Field(None, description="Expected answer if available")
    actual_answer: str = Field(..., description="Generated answer")
    retrieved_docs: list[str] = Field(..., description="Retrieved document IDs")
    metrics: EvaluationMetrics = Field(..., description="Evaluation metrics")
    passed: bool = Field(..., description="Whether evaluation passed thresholds")


class EvaluationReport(BaseModel):
    """Complete evaluation report."""

    total_queries: int = Field(..., description="Total queries evaluated", ge=0)
    passed: int = Field(..., description="Number of queries that passed", ge=0)
    failed: int = Field(..., description="Number of queries that failed", ge=0)
    avg_metrics: EvaluationMetrics = Field(..., description="Average metrics across all queries")
    results: list[EvaluationResult] = Field(..., description="Individual evaluation results")


class ThresholdConfig(BaseModel):
    """Threshold configuration for evaluation."""

    min_retrieval_precision: float = Field(
        0.7, description="Minimum retrieval precision", ge=0.0, le=1.0
    )
    min_retrieval_recall: float = Field(0.7, description="Minimum retrieval recall", ge=0.0, le=1.0)
    min_generation_quality: float = Field(
        0.7, description="Minimum generation quality", ge=0.0, le=1.0
    )
    min_answer_relevance: float = Field(0.7, description="Minimum answer relevance", ge=0.0, le=1.0)
    min_faithfulness: float = Field(0.7, description="Minimum faithfulness", ge=0.0, le=1.0)
