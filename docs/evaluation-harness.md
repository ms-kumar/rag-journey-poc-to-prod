# Evaluation Harness

Comprehensive evaluation framework for RAG system quality assurance with automated testing, metric tracking, and quality gates.

## Features

- 📊 **Comprehensive Metrics**: Precision@k, Recall@k, MRR, NDCG, MAP, and latency tracking
- 📝 **Dataset Management**: Create, save, load, and split evaluation datasets
- 🎯 **Quality Gates**: Configurable thresholds with pass/fail evaluation
- 🔄 **CI Integration**: GitHub Actions workflow with automatic PR comments
- 📈 **Weekly Dashboard**: HTML dashboard with trends and visualizations
- ✅ **Test Coverage**: Comprehensive unit tests for all components

## Quick Start

### 1. Create Evaluation Datasets

```bash
# Create default datasets
python scripts/create_eval_datasets.py

# Or using Make
make eval-datasets
```

This creates:
- `data/eval/rag_default_eval.json` - 20 queries covering all topics
- `data/eval/rag_test_small.json` - 5 queries for quick testing

### 2. Run Evaluation

```bash
# Run with default dataset
make eval

# Run with small test dataset (fast)
make eval-ci

# Run with custom dataset
python scripts/ci_eval_gate.py --dataset data/eval/my_dataset.json
```

### 3. Generate Dashboard

```bash
# Generate HTML dashboard
make dashboard

# Open dashboard
open results/dashboard.html
```

### 4. Full Workflow

```bash
# Run complete evaluation workflow
make eval-full
```

## Usage

### Basic Evaluation

```python
from src.services.evaluation import (
    EvaluationHarness,
    EvalDataset,
    ThresholdConfig
)

# Load dataset
dataset = EvalDataset.load("data/eval/rag_eval.json")

# Configure thresholds
thresholds = ThresholdConfig(
    min_precision_at_5=0.7,
    min_recall_at_10=0.8,
    min_mrr=0.6,
    max_latency_p95=2000.0
)

# Create harness
harness = EvaluationHarness(
    retrieval_function=my_retrieval_fn,
    thresholds=thresholds
)

# Run evaluation
result = harness.evaluate(dataset, k_values=[1, 3, 5, 10])

# Check results
if result.passed:
    print("✅ Evaluation passed!")
    print(result.metrics.get_summary())
else:
    print("❌ Failed:", result.failed_checks)
```

### Create Custom Dataset

```python
from src.services.evaluation.dataset import EvalDataset

# Create dataset
dataset = EvalDataset(
    name="my_eval",
    description="My custom evaluation dataset"
)

# Add examples
dataset.add_example(
    query="What is RAG?",
    relevant_doc_ids=["doc1", "doc2"],
    expected_answer="RAG is...",
    metadata={"category": "basics"}
)

# Save
dataset.save("data/eval/my_eval.json")
```

### Calculate Metrics

```python
from src.services.evaluation.metrics import MetricsCalculator

calc = MetricsCalculator()

# Calculate precision@5
precision = calc.precision_at_k(
    retrieved_ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
    relevant_ids={"doc1", "doc3", "doc5"},
    k=5
)
# Result: 0.6 (3 relevant out of 5)

# Calculate MRR
mrr = calc.mean_reciprocal_rank(
    retrieved_ids=["doc1", "doc2", "doc3"],
    relevant_ids={"doc2"}
)
# Result: 0.5 (relevant at position 2)

# Calculate NDCG@10
ndcg = calc.ndcg_at_k(
    retrieved_ids=retrieved,
    relevant_ids=relevant,
    k=10
)
```

## Metrics Reference

### Retrieval Metrics

| Metric | Description | Range | Interpretation |
|--------|-------------|-------|----------------|
| **Precision@k** | Fraction of top-k results that are relevant | 0-1 | Higher is better |
| **Recall@k** | Fraction of relevant docs retrieved in top-k | 0-1 | Higher is better |
| **MRR** | Reciprocal rank of first relevant result | 0-1 | Higher is better |
| **NDCG@k** | Ranking quality with position discount | 0-1 | Higher is better |
| **MAP** | Mean average precision across queries | 0-1 | Higher is better |

### Performance Metrics

| Metric | Description | Unit | Interpretation |
|--------|-------------|------|----------------|
| **Latency P50** | Median query latency | ms | Lower is better |
| **Latency P95** | 95th percentile latency | ms | Lower is better |
| **Latency P99** | 99th percentile latency | ms | Lower is better |

## Configuration

### Threshold Configuration

Edit `config/eval_thresholds.json`:

```json
{
  "retrieval_thresholds": {
    "min_precision_at_5": 0.6,
    "min_recall_at_10": 0.7,
    "min_mrr": 0.5,
    "min_ndcg_at_10": 0.65,
    "min_map": 0.6
  },
  "performance_thresholds": {
    "max_latency_p95": 2000.0,
    "max_latency_p99": 3000.0
  }
}
```

### Dataset Format

Example `eval_dataset.json`:

```json
{
  "name": "rag_eval",
  "description": "RAG evaluation dataset",
  "examples": [
    {
      "query": "What is RAG?",
      "relevant_doc_ids": ["doc1", "doc2"],
      "query_id": "q_001",
      "expected_answer": "RAG is...",
      "metadata": {
        "category": "basics",
        "difficulty": "easy"
      }
    }
  ]
}
```

## CI Integration

The evaluation gate runs automatically on:
- Push to `main` or `develop`
- Pull requests to `main` or `develop`
- Manual workflow dispatch

### Workflow Features

- ✅ Automatic dataset creation
- ✅ Qdrant service setup
- ✅ Metric calculation
- ✅ PR comments with results
- ✅ Artifact upload
- ✅ Fail on threshold violations

### Example PR Comment

```
## RAG Evaluation Results ✅ PASSED

**Duration:** 12.34s

### Retrieval Metrics
- Precision@5: 0.723
- Recall@10: 0.801
- MRR: 0.678
- NDCG@10: 0.712
- MAP: 0.689

### Performance
- Latency P50: 123.4ms
- Latency P95: 456.7ms
- Latency P99: 789.0ms
```

## Dashboard

The weekly dashboard provides:

- 📊 Latest evaluation metrics
- 📈 Historical trends with percentage changes
- 📋 Result history table
- ✅ Pass/fail status
- 🎨 Professional, responsive UI

**Generate:**
```bash
make dashboard
```

**View:**
```bash
open results/dashboard.html
```

## Scripts

### `scripts/create_eval_datasets.py`

Creates default evaluation datasets.

```bash
python scripts/create_eval_datasets.py
```

### `scripts/ci_eval_gate.py`

Runs evaluation and exits with error code if thresholds not met.

```bash
# Default
python scripts/ci_eval_gate.py

# Custom dataset
python scripts/ci_eval_gate.py --dataset data/eval/my_dataset.json

# Strict mode
python scripts/ci_eval_gate.py --strict

# Custom output
python scripts/ci_eval_gate.py --output results/my_result.json
```

### `scripts/generate_dashboard.py`

Generates HTML dashboard from evaluation results.

```bash
# Default
python scripts/generate_dashboard.py

# Custom directories
python scripts/generate_dashboard.py \
  --results-dir my_results \
  --output reports/dashboard.html
```

## Testing

Run tests for evaluation components:

```bash
# All evaluation tests
pytest tests/test_evaluation_*.py -v

# With coverage
pytest tests/test_evaluation_*.py --cov=src/services/evaluation

# Specific test file
pytest tests/test_evaluation_metrics.py -v
```

## Examples

### Run Demo

```bash
python examples/evaluation_demo.py
```

The demo shows:
- ✅ Metric calculations
- ✅ Dataset creation
- ✅ Running evaluation
- ✅ Result comparison

## Architecture

```
src/services/evaluation/
├── __init__.py          # Public API
├── metrics.py           # Metric calculations
├── dataset.py           # Dataset management
└── harness.py           # Evaluation orchestration

scripts/
├── create_eval_datasets.py    # Dataset creation
├── ci_eval_gate.py             # CI evaluation gate
└── generate_dashboard.py       # Dashboard generation

config/
└── eval_thresholds.json        # Threshold config

data/eval/
├── rag_default_eval.json       # Default dataset
└── rag_test_small.json         # Test dataset

tests/
├── test_evaluation_metrics.py
└── test_evaluation_dataset.py
```

## Best Practices

### 1. Dataset Quality

- ✅ Keep datasets version controlled
- ✅ Add diverse query types and difficulties
- ✅ Maintain high-quality relevance judgments
- ✅ Update datasets as system evolves

### 2. Threshold Management

- ✅ Start conservative, increase gradually
- ✅ Monitor trends over time
- ✅ Adjust based on business requirements
- ✅ Document all threshold changes

### 3. CI Usage

- ✅ Use small dataset for fast PR feedback
- ✅ Run full evaluation nightly/weekly
- ✅ Block merges that fail quality gates
- ✅ Review failed evaluations promptly

### 4. Continuous Improvement

- ✅ Generate weekly dashboards
- ✅ Track metrics over time
- ✅ Investigate drops immediately
- ✅ Celebrate improvements

## Troubleshooting

### Issue: Datasets not found

```bash
# Create datasets first
python scripts/create_eval_datasets.py
```

### Issue: Low metric scores

- Check document ingestion
- Verify embedding quality
- Review query formulation
- Adjust chunk size/overlap

### Issue: CI fails

- Check Qdrant service status
- Verify environment variables
- Review dataset quality
- Adjust thresholds if needed

### Issue: Dashboard not generating

```bash
# Ensure results exist
ls results/eval_*.json

# Run evaluation first
python scripts/ci_eval_gate.py

# Then generate dashboard
python scripts/generate_dashboard.py
```

## API Reference

### EvaluationHarness

```python
class EvaluationHarness:
    def __init__(
        self,
        retrieval_function: Callable,
        generation_function: Callable | None = None,
        thresholds: ThresholdConfig | None = None
    )
    
    def evaluate(
        self,
        dataset: EvalDataset,
        k_values: list[int] | None = None,
        include_generation: bool = False
    ) -> EvalResult
    
    def save_results(self, result: EvalResult, filepath: Path) -> None
    
    def compare_results(
        self,
        baseline_result: EvalResult,
        current_result: EvalResult
    ) -> dict[str, Any]
```

### EvalDataset

```python
class EvalDataset:
    def __init__(
        self,
        examples: list[EvalExample] = [],
        name: str = "default",
        description: str = ""
    )
    
    def add_example(
        self,
        query: str,
        relevant_doc_ids: list[str],
        expected_answer: str | None = None,
        metadata: dict | None = None
    ) -> None
    
    def save(self, filepath: Path) -> None
    
    @classmethod
    def load(cls, filepath: Path) -> "EvalDataset"
    
    def split(self, train_ratio: float = 0.8) -> tuple["EvalDataset", "EvalDataset"]
    
    def get_statistics(self) -> dict[str, Any]
```

### MetricsCalculator

```python
class MetricsCalculator:
    @staticmethod
    def precision_at_k(
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int
    ) -> float
    
    @staticmethod
    def recall_at_k(
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int
    ) -> float
    
    @staticmethod
    def mean_reciprocal_rank(
        retrieved_ids: list[str],
        relevant_ids: set[str]
    ) -> float
    
    @staticmethod
    def ndcg_at_k(
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int
    ) -> float
    
    @staticmethod
    def average_precision(
        retrieved_ids: list[str],
        relevant_ids: set[str]
    ) -> float
```

## Future Enhancements

- [ ] LLM-as-judge for generation metrics
- [ ] NLI-based faithfulness checking
- [ ] A/B testing framework
- [ ] Real-time monitoring dashboard
- [ ] Multi-language support
- [ ] Domain-specific metrics

## License

Part of the RAG Journey project.

## Support

For issues or questions:
- Check documentation in `docs/week-plans/week-5.md`
- Run demo: `python examples/evaluation_demo.py`
- Review tests: `pytest tests/test_evaluation_*.py -v`
