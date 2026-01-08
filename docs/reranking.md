# Cross-Encoder Re-ranking

Cross-encoder re-ranking improves retrieval precision by using a trained model to score query-document pairs more accurately than the initial embedding-based similarity.

## Overview

The re-ranker takes an initial set of candidates from the retrieval stage and re-scores them using a cross-encoder model (e.g., MS MARCO trained models). This typically improves precision@k metrics, especially for the top results.

### How It Works

1. **Initial Retrieval**: Retrieve more candidates than needed (e.g., 3x the target k)
2. **Cross-Encoder Scoring**: Score each query-document pair with a trained model
3. **Re-ranking**: Sort by cross-encoder scores (descending)
4. **Top-K Selection**: Return the top k documents

### Benefits

- **Improved Precision@k**: Typical improvements of 0.1-0.3 for precision@1
- **Better Relevance**: More semantically relevant documents in top positions
- **Search Type Agnostic**: Works with vector, BM25, hybrid, and sparse search

## Configuration

### Pipeline Configuration

```python
from src.services.pipeline.naive_pipeline.client import NaivePipelineConfig

config = NaivePipelineConfig(
    # Enable re-ranking
    enable_reranker=True,
    
    # Model configuration
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    reranker_batch_size=32,
    reranker_timeout=30.0,
    reranker_top_k=None,  # None = rerank all candidates
)
```

### Direct Reranker Usage

```python
from src.services.reranker.factory import get_reranker
from langchain_core.documents import Document

# Initialize reranker
reranker = get_reranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    batch_size=16,
    timeout_seconds=15.0,
    fallback_enabled=True
)

# Re-rank documents
documents = [Document(page_content="...", metadata={"id": "1"}), ...]
query = "What is machine learning?"

result = reranker.rerank(query, documents, top_k=5)

# Access results
reranked_docs = result.documents
scores = result.scores
execution_time = result.execution_time
fallback_used = result.fallback_used
```

## API Usage

### Enable Re-ranking in API Requests

```python
import requests

response = requests.post("http://localhost:8000/api/v1/rag/generate", json={
    "prompt": "What is machine learning?",
    "top_k": 5,
    "search_type": "hybrid",
    "enable_reranking": True  # Enable re-ranking for this request
})
```

### Response with Re-ranking Metadata

When re-ranking is enabled, documents in the response will include additional metadata:

```json
{
  "documents": [
    {
      "page_content": "Machine learning is...",
      "metadata": {
        "id": "doc1",
        "score": 0.85,
        "reranked": true,
        "original_rank": 2,
        "rerank_score": 0.92,
        "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "rerank_fallback": false
      }
    }
  ]
}
```

## Performance Considerations

### Latency Trade-offs

- **Additional Latency**: 50-300ms depending on:
  - Number of candidates to re-rank
  - Model size (MiniLM vs full BERT)
  - Batch size
  - Hardware (CPU vs GPU)

- **Optimization Strategies**:
  - Use smaller models (MiniLM, TinyBERT) for faster inference
  - Increase batch size for better throughput
  - Limit candidates (e.g., re-rank only top 20-50)
  - Enable GPU acceleration when available

### Memory Usage

- **Model Loading**: 100-500MB depending on model size
- **Batch Processing**: ~1-10MB per batch depending on sequence length
- **Caching**: Models are loaded once and reused

## Model Selection

### Recommended Models

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `cross-encoder/ms-marco-TinyBERT-L-2` | 50MB | Fast | Good | Low latency requirements |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 90MB | Medium | Very Good | Balanced performance |
| `cross-encoder/ms-marco-electra-base` | 400MB | Slow | Excellent | Quality-first scenarios |

### Domain-Specific Models

- **General Web**: `ms-marco-*` models
- **Scientific**: `cross-encoder/stsb-*` for semantic similarity
- **News**: Fine-tune on domain data or use general models

## Evaluation Metrics

### Precision@k Improvements

```python
from src.services.reranker.evaluation import RerankingEvaluator

evaluator = RerankingEvaluator(reranker)
comparison = evaluator.compare_rankings(
    query=query,
    baseline_docs=original_docs,
    relevant_doc_ids={"doc1", "doc3", "doc5"}
)

# Check improvements
for k in [1, 3, 5, 10]:
    improvement = comparison.improvement[k]
    print(f"Precision@{k} improvement: {improvement:+.3f}")
```

### Benchmark Multiple Queries

```python
results = evaluator.benchmark_multiple_queries(
    queries=["query1", "query2", ...],
    document_lists=[docs1, docs2, ...],
    relevant_doc_sets=[rel1, rel2, ...]
)

print(f"Average precision@5 improvement: {results['avg_improvement'][5]:.3f}")
print(f"Average latency: {results['avg_latency_ms']:.2f}ms")
```

## Timeouts and Fallbacks

### Timeout Configuration

```python
reranker = get_reranker(
    timeout_seconds=10.0,        # Max time per query
    fallback_enabled=True,       # Enable fallback on timeout
    fallback_strategy="original_order"  # or "score_descending"
)
```

### Fallback Strategies

- **`original_order`**: Return documents in their original retrieval order
- **`score_descending`**: Sort by existing scores (from initial retrieval)

### Error Handling

```python
result = reranker.rerank(query, documents)

if result.fallback_used:
    print(f"Fallback used: {result.model_used}")
    print(f"Execution time: {result.execution_time:.3f}s")
else:
    print(f"Re-ranking successful: {len(result.scores)} scores computed")
```

## Best Practices

### When to Use Re-ranking

✅ **Good candidates**:
- High-precision requirements (top-k results matter most)
- Sufficient compute budget (50-300ms acceptable)
- Complex queries where embedding similarity isn't sufficient
- Hybrid search pipelines

❌ **Avoid when**:
- Ultra-low latency requirements (<50ms end-to-end)
- Very large result sets (>100 candidates)
- Simple exact-match queries
- Resource-constrained environments

### Configuration Recommendations

1. **Development**: Use TinyBERT for fast iteration
2. **Production**: Use MiniLM for balanced performance
3. **High-precision**: Use larger models with GPU acceleration
4. **Batch Size**: Start with 16-32, increase if memory allows
5. **Candidates**: Re-rank 2-3x your target k (e.g., rerank 15 to return top 5)

### Monitoring

```python
# Health check
health = reranker.health_check()
print(f"Model loaded: {health['model_loaded']}")
print(f"Device: {health['device']}")

# Performance tracking
if result.execution_time > 1.0:  # Log slow queries
    logger.warning(f"Slow re-ranking: {result.execution_time:.2f}s")
```

## Examples

See the `examples/` directory for comprehensive demonstrations:

- `examples/reranker_demo.py`: Basic usage and evaluation
- `examples/reranker_benchmark.py`: Performance benchmarking

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check internet connection (models download from HuggingFace)
   - Verify model name exists on HuggingFace Hub
   - Check disk space for model caching

2. **Memory Issues**
   - Reduce batch size
   - Use smaller models
   - Enable model quantization if available

3. **Slow Performance**
   - Enable GPU if available
   - Increase batch size
   - Use smaller/faster models
   - Reduce number of candidates

4. **Timeout Issues**
   - Increase timeout value
   - Reduce batch size
   - Check system resources
   - Enable fallback for robustness