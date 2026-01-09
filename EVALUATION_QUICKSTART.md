# Week 5: Evaluation Harness - Quick Start Guide

## 🎯 Overview

Complete evaluation framework for RAG system with:
- ✅ Comprehensive metrics (Precision, Recall, MRR, NDCG, MAP)
- ✅ Evaluation datasets (20 default + 5 test queries)
- ✅ CI integration with GitHub Actions
- ✅ Configurable quality thresholds
- ✅ Weekly HTML dashboard with trends

## 🚀 Quick Start

### 1. Create Evaluation Datasets

```bash
# Using Make
make eval-datasets

# Or directly
python scripts/create_eval_datasets.py
```

**Output:**
- `data/eval/rag_default_eval.json` - 20 queries
- `data/eval/rag_test_small.json` - 5 queries

### 2. Run Evaluation

```bash
# Full evaluation (20 queries)
make eval

# Quick CI test (5 queries)
make eval-ci

# Custom dataset
python scripts/ci_eval_gate.py --dataset data/eval/my_dataset.json
```

### 3. Generate Dashboard

```bash
# Generate HTML dashboard
make dashboard

# View dashboard
open results/dashboard.html  # macOS
xdg-open results/dashboard.html  # Linux
```

### 4. Full Workflow

```bash
# Run everything
make eval-full
```

## 📊 What Was Built

### Core Modules

```
src/services/evaluation/
├── __init__.py       # Public API
├── metrics.py        # Metric calculations (320 lines)
├── dataset.py        # Dataset management (240 lines)
└── harness.py        # Evaluation orchestration (470 lines)
```

### Scripts

```
scripts/
├── create_eval_datasets.py     # Create datasets
├── ci_eval_gate.py              # CI evaluation
├── generate_dashboard.py        # Dashboard generator
├── verify_evaluation.py         # Verification
└── test_metrics_standalone.py   # Standalone test
```

### Configuration

```
config/
└── eval_thresholds.json  # Quality thresholds
```

### CI/CD

```
.github/workflows/
└── eval_gate.yml  # GitHub Actions workflow
```

## 📈 Metrics Reference

| Metric | What It Measures | Range | Good Value |
|--------|------------------|-------|------------|
| Precision@5 | % of top-5 results that are relevant | 0-1 | > 0.6 |
| Recall@10 | % of relevant docs in top-10 | 0-1 | > 0.7 |
| MRR | Position of first relevant result | 0-1 | > 0.5 |
| NDCG@10 | Ranking quality with discount | 0-1 | > 0.65 |
| MAP | Average precision across queries | 0-1 | > 0.6 |
| Latency P95 | 95th percentile query time | ms | < 2000 |

## 💻 Code Examples

### Basic Usage

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
    min_mrr=0.6
)

# Run evaluation
harness = EvaluationHarness(
    retrieval_function=my_retrieval_fn,
    thresholds=thresholds
)

result = harness.evaluate(dataset)

# Check results
if result.passed:
    print("✅ Evaluation passed!")
else:
    print("❌ Failed checks:", result.failed_checks)
```

### Create Custom Dataset

```python
from src.services.evaluation.dataset import EvalDataset

dataset = EvalDataset(name="my_eval")

dataset.add_example(
    query="What is RAG?",
    relevant_doc_ids=["doc1", "doc2"],
    expected_answer="RAG is...",
    metadata={"category": "basics"}
)

dataset.save("data/eval/my_eval.json")
```

### Calculate Metrics

```python
from src.services.evaluation.metrics import MetricsCalculator

calc = MetricsCalculator()

precision = calc.precision_at_k(
    retrieved_ids=["doc1", "doc2", "doc3"],
    relevant_ids={"doc1", "doc3"},
    k=3
)
# Result: 0.667
```

## 🔧 Makefile Commands

```bash
make eval-datasets   # Create evaluation datasets
make eval            # Run full evaluation (20 queries)
make eval-ci         # Quick CI test (5 queries)
make dashboard       # Generate HTML dashboard
make eval-full       # Complete workflow (datasets + eval + dashboard)
```

## 🎭 CI Integration

Evaluation runs automatically on:
- ✅ Push to `main` or `develop`
- ✅ Pull requests
- ✅ Manual trigger

**PR Comment Example:**
```
## RAG Evaluation Results ✅ PASSED

**Duration:** 12.34s

### Retrieval Metrics
- Precision@5: 0.723
- Recall@10: 0.801
- MRR: 0.678

### Performance
- Latency P95: 456.7ms
```

## 📝 Files Created

### Production Code (15+ files, 3000+ lines)
- ✅ 3 evaluation modules (metrics, dataset, harness)
- ✅ 4 executable scripts
- ✅ 1 GitHub Actions workflow
- ✅ 1 configuration file
- ✅ 2 default datasets

### Tests & Docs
- ✅ 2 test files with comprehensive coverage
- ✅ 3 documentation files (1000+ lines)
- ✅ 1 example/demo script
- ✅ Updated Makefile

## ✅ Verification

All components verified:
```bash
# Test metric calculations
python scripts/test_metrics_standalone.py
# ✅ All metric calculations work correctly!

# Verify imports
python scripts/verify_evaluation.py
# ✅ All verification checks passed!

# Check compilation
python -m py_compile src/services/evaluation/*.py
# ✅ All evaluation modules compile successfully
```

## 📚 Documentation

- **[Week 5 Plan](docs/week-plans/week-5.md)** - Complete implementation details
- **[Evaluation Harness Guide](docs/evaluation-harness.md)** - API reference
- **[Week 5 Summary](docs/week-plans/week-5-summary.md)** - Implementation summary

## 🎯 Default Thresholds

Located in `config/eval_thresholds.json`:

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
    "max_latency_p95": 2000.0
  }
}
```

## 🎨 Dashboard Features

- 📊 Latest metrics display
- 📈 Week-over-week trend analysis
- 📋 Historical results (last 10 runs)
- ✅ Pass/fail status indicators
- 🎯 Threshold comparison
- 📱 Responsive design

## 🔍 Example Datasets

### Default Dataset (20 queries)
- RAG basics (2 queries)
- Embeddings (2 queries)
- BM25/sparse retrieval (2 queries)
- Qdrant vector DB (2 queries)
- FastAPI (2 queries)
- Machine learning (2 queries)
- Reranking (2 queries)
- Query understanding (2 queries)
- Text chunking (2 queries)
- Performance optimization (2 queries)

### Small Test Dataset (5 queries)
- RAG basics
- Embeddings
- BM25
- FastAPI
- Machine learning

## 🚀 Next Steps

1. **Run the demo:**
   ```bash
   python examples/evaluation_demo.py
   ```

2. **Create your datasets:**
   ```bash
   make eval-datasets
   ```

3. **Run evaluation:**
   ```bash
   make eval
   ```

4. **View dashboard:**
   ```bash
   make dashboard
   open results/dashboard.html
   ```

5. **Set up CI:** GitHub Actions workflow already configured!

## 💡 Tips

- Start with `make eval-ci` for quick feedback
- Use `make eval` for comprehensive testing
- Generate dashboards weekly for tracking
- Adjust thresholds in `config/eval_thresholds.json`
- Add custom queries to datasets as needed

## 🎉 Summary

Week 5 delivers a **production-ready evaluation framework** with:
- ✅ 5 retrieval metrics + latency tracking
- ✅ 25 evaluation queries (20 default + 5 test)
- ✅ Automated CI/CD integration
- ✅ Configurable quality gates
- ✅ Visual dashboard with trends
- ✅ Complete documentation
- ✅ Comprehensive tests

**All tasks completed successfully! 🎯**
