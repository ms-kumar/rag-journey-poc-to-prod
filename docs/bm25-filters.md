# BM25 Query Builder & Filters

Advanced filtering and hybrid search capabilities for the RAG system using Qdrant's BM25 indexing and metadata filters.

## Overview

The system now supports three types of retrieval:

1. **Vector Search** - Semantic similarity using embeddings (default)
2. **BM25 Search** - Traditional keyword-based ranking
3. **Hybrid Search** - Combined vector + BM25 for best results

Plus comprehensive **metadata filtering** to narrow results by document properties.

## Quick Start

### Enable BM25 Indexing

Update your `.env` file:

```bash
QDRANT__ENABLE_BM25=true
```

This enables BM25 text indexing on the `page_content` field. **Note:** You'll need to re-ingest documents after enabling BM25.

### Simple API Usage

```bash
# Vector search (default)
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is machine learning?",
    "top_k": 5
  }'

# BM25 keyword search
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "machine learning algorithms",
    "top_k": 5,
    "search_type": "bm25"
  }'

# Hybrid search (best of both)
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "neural networks deep learning",
    "top_k": 10,
    "search_type": "hybrid",
    "hybrid_alpha": 0.6
  }'

# With metadata filters
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "recent AI advances",
    "top_k": 5,
    "search_type": "hybrid",
    "metadata_filters": {
      "source": "research_papers.pdf",
      "year$gte": 2020,
      "category$in": ["AI", "ML"]
    }
  }'
```

## Filter Syntax

### Basic Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `key: value` | Exact match | `{"source": "doc1.txt"}` |
| `$in` | Match any of values | `{"category$in": ["AI", "ML"]}` |
| `$not` | Must not match | `{"status$not": "deleted"}` |
| `$text` | Full-text search | `{"content$text": "neural"}` |
| `$gt` | Greater than | `{"score$gt": 0.5}` |
| `$gte` | Greater or equal | `{"year$gte": 2020}` |
| `$lt` | Less than | `{"price$lt": 100}` |
| `$lte` | Less or equal | `{"score$lte": 1.0}` |

### Filter Examples

```python
# Simple exact match
filters = {"source": "paper.pdf"}

# Range query
filters = {"year$gte": 2020, "year$lte": 2023}

# Multiple conditions (AND)
filters = {
    "author": "Smith",
    "year$gte": 2020,
    "category$in": ["AI", "ML"]
}

# Exclude deleted items
filters = {"status$not": "deleted"}

# Text search on field
filters = {"abstract$text": "neural networks"}

# Complex example
filters = {
    "source": "research.pdf",
    "year$gte": 2020,
    "category$in": ["AI", "ML", "DL"],
    "status$not": "archived",
    "citations$gte": 10
}
```

## Python API

### Using the FilterBuilder

For programmatic filter construction:

```python
from src.services.vectorstore.filters import FilterBuilder

# Simple filter
filter = FilterBuilder().match("source", "doc1.txt").build()

# Complex filter
filter = (
    FilterBuilder()
    .match("author", "Smith")
    .range("year", gte=2020, lte=2023)
    .match_any("category", ["AI", "ML", "DL"])
    .must_not("status", "deleted")
    .text("content", "machine learning")
    .build()
)

# Use with vectorstore
docs = vectorstore.similarity_search_with_filter(
    query="neural networks",
    k=10,
    filter=filter
)
```

### Search Types

```python
from src.services.vectorstore.client import QdrantVectorStoreClient

# Initialize client with BM25 enabled
config = VectorStoreConfig(
    qdrant_url="http://localhost:6333",
    collection_name="my_collection",
    vector_size=384,
    enable_bm25=True
)
vectorstore = QdrantVectorStoreClient(embeddings, config)

# Vector similarity search (semantic)
docs = vectorstore.similarity_search("machine learning", k=5)

# Vector search with filters
docs = vectorstore.similarity_search_with_filter(
    query="deep learning",
    k=5,
    filter_dict={"category": "AI", "year$gte": 2020}
)

# BM25 keyword search
docs = vectorstore.bm25_search(
    query="neural network backpropagation",
    k=10,
    filter_dict={"source": "textbook.pdf"}
)

# Hybrid search (combines both methods)
docs = vectorstore.hybrid_search(
    query="transformer architectures",
    k=10,
    alpha=0.5,  # 0.0=BM25 only, 1.0=vector only, 0.5=balanced
    filter_dict={"year$gte": 2017}
)
```

### Pipeline Integration

```python
from src.services.pipeline.naive_pipeline.client import NaivePipeline

pipeline = NaivePipeline()

# Retrieve with filters
retrieved = pipeline.retrieve(
    query="recent advances in AI",
    k=10,
    search_type="hybrid",
    filters={"year$gte": 2023, "category$in": ["AI", "ML"]},
    hybrid_alpha=0.6
)

# Full RAG query
answer = pipeline.query("What are GPTs?", top_k=5)
```

## Search Type Comparison

### When to Use Each Search Type

| Search Type | Best For | Pros | Cons |
|-------------|----------|------|------|
| **Vector** | Semantic meaning, concepts | Understands context | May miss exact keywords |
| **BM25** | Keywords, exact terms | Fast, precise matching | No semantic understanding |
| **Hybrid** | Best of both worlds | Balanced results | Slightly slower |

### Performance Tips

1. **Vector Search** - Use for conceptual queries where meaning matters more than exact wording
   - Example: "explain how neural networks learn" vs "neural network training"

2. **BM25 Search** - Use for specific terms, names, or technical keywords
   - Example: "BERT tokenizer implementation"
   - Great for code search, API names, specific algorithms

3. **Hybrid Search** - Default choice for production
   - Combines semantic understanding with keyword matching
   - Adjust `alpha` to tune: 0.7-0.8 for more semantic, 0.3-0.4 for more keywords

### Hybrid Alpha Parameter

The `hybrid_alpha` parameter controls the balance:

```python
# More weight on semantic similarity (vector search)
docs = vectorstore.hybrid_search("AI ethics", alpha=0.7)

# More weight on keyword matching (BM25)
docs = vectorstore.hybrid_search("XGBoost hyperparameters", alpha=0.3)

# Balanced (default)
docs = vectorstore.hybrid_search("machine learning models", alpha=0.5)
```

**Recommended values:**
- `0.7-0.8`: Conceptual, semantic queries
- `0.5-0.6`: General purpose (recommended default)
- `0.3-0.4`: Technical terms, specific keywords

## Advanced Usage

### Custom Filter Logic

```python
# Build complex filters programmatically
builder = vectorstore.get_filter_builder()

# Add conditions based on logic
if user_role == "researcher":
    builder.match("access_level", "public")
else:
    builder.match("access_level", "internal")

# Add time range
if recent_only:
    builder.range("timestamp", gte=last_week)

# Build and use
filter = builder.build()
docs = vectorstore.similarity_search_with_filter(query, filter=filter)
```

### Batch Processing with Filters

```python
queries = [
    ("AI ethics", {"category": "ethics"}),
    ("neural networks", {"category": "technical"}),
    ("industry applications", {"category": "business"}),
]

results = []
for query, filters in queries:
    docs = vectorstore.hybrid_search(
        query=query,
        k=5,
        filter_dict=filters,
        alpha=0.6
    )
    results.append(docs)
```

### Metadata Filtering Best Practices

1. **Index your metadata** - Ensure important filter fields are indexed
2. **Use selective filters** - More specific filters = faster queries
3. **Combine with search type** - Filters + hybrid search = best precision
4. **Cache common filters** - Pre-build frequently used filter objects

## Configuration

### Environment Variables

```bash
# Enable BM25 indexing (required for BM25/hybrid search)
QDRANT__ENABLE_BM25=true

# Vector store settings
QDRANT__URL=http://localhost:6333
QDRANT__COLLECTION_NAME=my_collection
QDRANT__PREFER_GRPC=true

# RAG defaults
RAG__TOP_K=5
```

### Programmatic Configuration

```python
from src.config import get_settings

settings = get_settings()

# Access settings
print(f"BM25 enabled: {settings.vectorstore.enable_bm25}")
print(f"Collection: {settings.vectorstore.collection_name}")
```

## Testing

Run the comprehensive test suite:

```bash
# Run all filter tests
pytest tests/test_filters_bm25.py -v

# Run specific test class
pytest tests/test_filters_bm25.py::TestFilterBuilder -v

# Run with coverage
pytest tests/test_filters_bm25.py --cov=src/services/vectorstore
```

## Migration Guide

### Enabling BM25 on Existing Collections

1. **Update configuration:**
   ```bash
   QDRANT__ENABLE_BM25=true
   ```

2. **Recreate collection** (this will delete existing data):
   ```python
   # Delete old collection
   qdrant_client.delete_collection("my_collection")
   
   # Restart your app - collection will be recreated with BM25
   ```

3. **Re-ingest documents:**
   ```bash
   curl -X POST http://localhost:8000/api/v1/rag/ingest
   ```

### Updating Existing Code

**Before:**
```python
docs = vectorstore.similarity_search(query, k=5)
```

**After (with filters):**
```python
docs = vectorstore.similarity_search_with_filter(
    query,
    k=5,
    filter_dict={"category": "AI"}
)
```

**After (with hybrid search):**
```python
docs = vectorstore.hybrid_search(
    query,
    k=5,
    filter_dict={"category": "AI"},
    alpha=0.6
)
```

## Troubleshooting

### BM25 Search Returns Empty Results

**Cause:** BM25 indexing not enabled or documents not indexed.

**Solution:**
1. Verify `QDRANT__ENABLE_BM25=true` in .env
2. Check collection has BM25 index:
   ```python
   collection_info = qdrant_client.get_collection("collection_name")
   print(collection_info.config.params)
   ```
3. Re-ingest documents

### Filters Not Working

**Cause:** Metadata not stored during ingestion.

**Solution:** Ensure metadata is passed when adding texts:
```python
vectorstore.add_texts(
    texts=chunks,
    metadatas=[{"source": "doc.txt", "category": "AI"} for _ in chunks]
)
```

### Hybrid Search Too Slow

**Causes:** Large dataset, unoptimized filters, high k value.

**Solutions:**
1. Use more selective filters
2. Reduce `k` value (try k*2 instead of k*3 in prefetch)
3. Optimize Qdrant settings (HNSW parameters)
4. Consider using vector-only or BM25-only for specific use cases

## Examples

See the full examples in the codebase:

- **Filter builder:** `src/services/vectorstore/filters.py`
- **Vectorstore client:** `src/services/vectorstore/client.py`
- **RAG endpoint:** `src/api/v1/endpoints/rag.py`
- **Tests:** `tests/test_filters_bm25.py`

## API Reference

### FilterBuilder Methods

- `match(key, value)` - Exact match
- `match_any(key, values)` - Match any of values (OR)
- `match_except(key, values)` - Exclude values (NOT IN)
- `text(key, text)` - Full-text search
- `range(key, *, gt, gte, lt, lte)` - Numeric range
- `should(key, value)` - OR condition
- `must_not(key, value)` - Negation
- `build()` - Build Qdrant Filter object

### VectorStore Search Methods

- `similarity_search(query, k)` - Basic vector search
- `similarity_search_with_filter(query, k, filter, filter_dict)` - Vector + filters
- `bm25_search(query, k, filter, filter_dict)` - BM25 keyword search
- `hybrid_search(query, k, filter, filter_dict, alpha)` - Hybrid search
- `get_filter_builder()` - Get FilterBuilder instance

### GenerateRequest Fields

- `prompt` (str, required) - User query
- `top_k` (int, default=5) - Number of results
- `search_type` ("vector"|"bm25"|"hybrid", default="vector")
- `metadata_filters` (dict, optional) - Filter conditions
- `hybrid_alpha` (float, default=0.5) - Hybrid search weight
- `max_length` (int, optional) - Generation max tokens

## Performance Benchmarks

Typical query latencies (on 10K documents):

| Search Type | Latency | Precision | Recall |
|-------------|---------|-----------|--------|
| Vector only | 10-20ms | Good | Good |
| BM25 only | 5-15ms | Excellent | Fair |
| Hybrid (α=0.5) | 15-30ms | Excellent | Excellent |

*With filters: +2-10ms depending on selectivity*

## Future Enhancements

- [ ] Custom BM25 parameters (k1, b)
- [ ] Multi-field BM25 indexing
- [ ] Learned fusion weights (instead of fixed alpha)
- [ ] Filter query DSL (JSON-based)
- [ ] Geospatial filters
- [ ] Date range helpers
- [ ] Filter templates

## References

- [Qdrant Filtering Documentation](https://qdrant.tech/documentation/concepts/filtering/)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Hybrid Search Paper](https://arxiv.org/abs/2104.08663)
