# Metadata Filtering

Comprehensive guide to filtering RAG retrieval results using metadata fields like source, date, tags, and custom attributes.

## Overview

Metadata filtering allows you to narrow down document retrieval to specific subsets based on document attributes. This is essential for:

- **Source-based filtering**: Retrieve from specific documents or collections
- **Time-based filtering**: Focus on recent or historical documents
- **Tag-based filtering**: Filter by categories, topics, or labels
- **Author filtering**: Retrieve documents by specific authors
- **Custom metadata**: Filter by any custom field in your documents

## Quick Start

### Simple Source Filter

Filter by a single document source:

```python
from src.services.vectorstore.filters import FilterBuilder

# Using builder pattern
filter = FilterBuilder().source("research_paper.pdf").build()

# Using dictionary
filter_dict = {"source": "research_paper.pdf"}
```

```bash
# curl example
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are the key findings?",
    "top_k": 5,
    "metadata_filters": {"source": "research_paper.pdf"}
  }'
```

### Multiple Sources

Filter by multiple document sources (match any):

```python
filter = FilterBuilder().sources(["paper1.pdf", "paper2.pdf", "paper3.pdf"]).build()

# Or using dictionary
filter_dict = {"sources": ["paper1.pdf", "paper2.pdf", "paper3.pdf"]}
```

```bash
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Summarize the main points",
    "metadata_filters": {"sources": ["paper1.pdf", "paper2.pdf"]}
  }'
```

## Date Range Filtering

### Date Range with ISO Strings

```python
from src.services.vectorstore.filters import FilterBuilder

# Documents from 2024
filter = FilterBuilder().date_range(
    after="2024-01-01",
    before="2024-12-31"
).build()
```

```bash
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What happened this year?",
    "metadata_filters": {
      "date_after": "2024-01-01",
      "date_before": "2024-12-31"
    }
  }'
```

### Recent Documents Only

```python
# Documents created after a specific date
filter = FilterBuilder().created_after("2024-06-01").build()

# Using dictionary
filter_dict = {"created_after": "2024-06-01"}
```

```bash
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Recent developments?",
    "metadata_filters": {"date_after": "2024-06-01"}
  }'
```

### Historical Documents

```python
# Documents created before a specific date
filter = FilterBuilder().created_before("2023-12-31").build()
```

```bash
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What did we know before 2024?",
    "metadata_filters": {"date_before": "2023-12-31"}
  }'
```

## Tag-Based Filtering

### Single Tag

```python
filter = FilterBuilder().tag("machine-learning").build()

# Dictionary
filter_dict = {"tag": "machine-learning"}
```

```bash
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "ML techniques",
    "metadata_filters": {"tag": "machine-learning"}
  }'
```

### Multiple Tags (Match Any)

```python
# Documents with any of these tags
filter = FilterBuilder().tags(["ai", "ml", "deep-learning"]).build()

# Dictionary
filter_dict = {"tags": ["ai", "ml", "deep-learning"]}
```

```bash
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "AI and ML overview",
    "metadata_filters": {"tags": ["ai", "ml", "deep-learning"]}
  }'
```

## Author Filtering

### Single Author

```python
filter = FilterBuilder().author("John Smith").build()

# Dictionary
filter_dict = {"author": "John Smith"}
```

```bash
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "John Smith papers",
    "metadata_filters": {"author": "John Smith"}
  }'
```

### Multiple Authors

```python
filter = FilterBuilder().authors(["Smith", "Johnson", "Williams"]).build()

# Dictionary
filter_dict = {"authors": ["Smith", "Johnson", "Williams"]}
```

```bash
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Research by team members",
    "metadata_filters": {"authors": ["Smith", "Johnson"]}
  }'
```

## Complex Filters

### Combining Multiple Criteria

```python
# Research papers by specific authors from 2024 with AI tags
filter = (
    FilterBuilder()
    .authors(["Smith", "Johnson"])
    .tags(["ai", "machine-learning"])
    .date_range(after="2024-01-01")
    .source("research_collection")
    .build()
)
```

```bash
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Recent AI research findings",
    "metadata_filters": {
      "authors": ["Smith", "Johnson"],
      "tags": ["ai", "machine-learning"],
      "date_after": "2024-01-01",
      "source": "research_collection"
    }
  }'
```

### With Exclusions

```python
# Documents except drafts, from specific sources
filter = (
    FilterBuilder()
    .sources(["final_report.pdf", "analysis.pdf"])
    .date_range(after="2024-01-01")
    .must_not("status", "draft")
    .exclude_source("temporary.txt")
    .build()
)
```

```bash
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Final reports analysis",
    "metadata_filters": {
      "sources": ["final_report.pdf", "analysis.pdf"],
      "date_after": "2024-01-01",
      "status$not": "draft"
    }
  }'
```

## Advanced Operators

### Range Operators

For numeric fields:

```python
# Documents with citations >= 100
filter_dict = {"citations$gte": 100}

# Published between 2020-2024
filter_dict = {"year$gte": 2020, "year$lte": 2024}

# High relevance scores
filter_dict = {"relevance$gt": 0.8}
```

```bash
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Highly cited papers",
    "metadata_filters": {
      "citations$gte": 100,
      "year$gte": 2020
    }
  }'
```

### Match Any ($in)

```python
# Documents in any of these categories
filter_dict = {"category$in": ["AI", "ML", "DL", "NLP"]}
```

```bash
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "AI research overview",
    "metadata_filters": {"category$in": ["AI", "ML", "NLP"]}
  }'
```

### Negation ($not)

```python
# Exclude deleted or archived documents
filter_dict = {"status$not": "deleted"}
```

```bash
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Active documents only",
    "metadata_filters": {"status$not": "deleted"}
  }'
```

### Text Search ($text)

```python
# Full-text search in content field
filter_dict = {"content$text": "neural networks"}
```

```bash
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Neural network architectures",
    "metadata_filters": {"content$text": "neural networks"}
  }'
```

## Real-World Examples

### Academic Research Use Case

Find recent highly-cited papers by specific researchers:

```bash
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are the recent breakthroughs?",
    "top_k": 10,
    "metadata_filters": {
      "authors": ["LeCun", "Hinton", "Bengio"],
      "tags": ["deep-learning", "neural-networks"],
      "date_after": "2023-01-01",
      "citations$gte": 50,
      "venue$not": "workshop"
    },
    "enable_reranking": true
  }'
```

### Document Management Use Case

Retrieve Q1 2024 financial reports:

```bash
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Summarize Q1 financial performance",
    "metadata_filters": {
      "tag": "quarterly-report",
      "date_after": "2024-01-01",
      "date_before": "2024-03-31",
      "author": "Finance Team",
      "status$not": "draft"
    }
  }'
```

### News/Content Use Case

Find recent articles about specific topics:

```bash
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Latest AI developments",
    "top_k": 5,
    "metadata_filters": {
      "tags": ["artificial-intelligence", "machine-learning"],
      "date_after": "2024-11-01",
      "sources": ["techcrunch.com", "mit-news.com", "nature.com"]
    },
    "search_type": "hybrid",
    "enable_reranking": true
  }'
```

## Python API Examples

### Using FilterBuilder (Programmatic)

```python
from src.services.vectorstore.filters import FilterBuilder
from src.services.vectorstore.client import QdrantVectorStore

# Create vectorstore
vectorstore = QdrantVectorStore(collection_name="documents")

# Build complex filter
filter_obj = (
    FilterBuilder()
    .sources(["paper1.pdf", "paper2.pdf"])
    .tags(["ml", "ai"])
    .date_range(after="2024-01-01", before="2024-12-31")
    .author("Smith")
    .must_not("status", "draft")
    .build()
)

# Use in search
results = vectorstore.search(
    query="machine learning advances",
    k=10,
    filter=filter_obj
)
```

### Using Dictionary Format (API-friendly)

```python
from src.services.vectorstore.filters import build_filter_from_dict
from src.services.pipeline.naive_pipeline import NaivePipeline

# Create pipeline
pipeline = NaivePipeline()

# Define filter as dictionary
filter_dict = {
    "sources": ["paper1.pdf", "paper2.pdf"],
    "tags": ["ml", "ai"],
    "date_after": "2024-01-01",
    "date_before": "2024-12-31",
    "author": "Smith",
    "status$not": "draft"
}

# Retrieve with filters
documents = pipeline.retrieve(
    query="machine learning advances",
    k=10,
    filters=filter_dict
)
```

## Filter Dictionary Reference

### Convenience Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `source` | string | Single source exact match | `{"source": "paper.pdf"}` |
| `sources` | list[string] | Multiple sources (match any) | `{"sources": ["doc1.txt", "doc2.txt"]}` |
| `tag` | string | Single tag exact match | `{"tag": "ai"}` |
| `tags` | list[string] | Multiple tags (match any) | `{"tags": ["ai", "ml"]}` |
| `author` | string | Single author exact match | `{"author": "Smith"}` |
| `authors` | list[string] | Multiple authors (match any) | `{"authors": ["Smith", "Lee"]}` |
| `date_after` | string (ISO) | Documents after date | `{"date_after": "2024-01-01"}` |
| `date_before` | string (ISO) | Documents before date | `{"date_before": "2024-12-31"}` |
| `created_after` | string (ISO) | Created after date | `{"created_after": "2024-01-01"}` |
| `created_before` | string (ISO) | Created before date | `{"created_before": "2024-12-31"}` |

### Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$in` | Match any value in list | `{"category$in": ["AI", "ML"]}` |
| `$not` | Must not match | `{"status$not": "deleted"}` |
| `$gt` | Greater than | `{"score$gt": 0.8}` |
| `$gte` | Greater than or equal | `{"year$gte": 2020}` |
| `$lt` | Less than | `{"price$lt": 100}` |
| `$lte` | Less than or equal | `{"year$lte": 2024}` |
| `$text` | Full-text search | `{"content$text": "neural networks"}` |
| `$after` | Date after (custom field) | `{"published$after": "2024-01-01"}` |
| `$before` | Date before (custom field) | `{"published$before": "2024-12-31"}` |

## Best Practices

### 1. Use Convenience Fields

Prefer convenience fields for common patterns:

```python
# Good - clear and concise
{"sources": ["doc1.txt", "doc2.txt"], "tags": ["ai", "ml"]}

# Also works, but more verbose
{"source$in": ["doc1.txt", "doc2.txt"], "tags$in": ["ai", "ml"]}
```

### 2. Combine Filters for Precision

Use multiple criteria to narrow results:

```python
{
    "sources": ["research_papers/"],
    "date_after": "2024-01-01",
    "tags": ["peer-reviewed"],
    "citations$gte": 10
}
```

### 3. Date Format

Always use ISO 8601 format for dates:

```python
# Good
{"date_after": "2024-01-01"}
{"date_after": "2024-01-01T12:00:00"}

# Bad (will fail)
{"date_after": "01/01/2024"}
{"date_after": "January 1, 2024"}
```

### 4. Combine with Re-ranking

For best results, combine filters with re-ranking:

```python
{
    "metadata_filters": {
        "sources": ["authoritative_sources/"],
        "date_after": "2024-01-01"
    },
    "enable_reranking": true,
    "top_k": 10
}
```

### 5. Test Filter Results

Verify filters return expected documents:

```python
# Check number of results
results = vectorstore.search(query, k=100, filter_dict=filters)
print(f"Found {len(results)} documents matching filters")

# Inspect metadata
for doc in results[:5]:
    print(f"Source: {doc.metadata.get('source')}")
    print(f"Tags: {doc.metadata.get('tags')}")
    print(f"Date: {doc.metadata.get('date')}")
```

## Performance Considerations

### Index Your Metadata

Ensure frequently filtered fields are indexed in Qdrant:

```python
from qdrant_client.models import PayloadSchemaType

vectorstore.client.create_payload_index(
    collection_name="documents",
    field_name="source",
    field_schema=PayloadSchemaType.KEYWORD
)
```

### Filter Before Search

Applying filters reduces search space and improves performance:

```python
# Efficient - filter narrows search space
results = vectorstore.search(query, k=10, filter_dict={"source": "large_doc.pdf"})

# Less efficient - searches entire collection
all_results = vectorstore.search(query, k=1000)
filtered = [r for r in all_results if r.metadata["source"] == "large_doc.pdf"]
```

### Avoid Over-filtering

Balance precision with recall:

```python
# May be too restrictive
filter_dict = {
    "source": "exact_file.txt",
    "author": "Exact Name",
    "date_after": "2024-12-01",
    "date_before": "2024-12-01",
    "tags": ["very-specific-tag"]
}

# Better - allows more relevant results
filter_dict = {
    "sources": ["collection1/", "collection2/"],
    "date_after": "2024-01-01",
    "tags": ["ai", "ml", "nlp"]
}
```

## Troubleshooting

### No Results Returned

1. **Check filter values match metadata exactly**:
   ```python
   # Inspect actual metadata
   doc = vectorstore.get_document(doc_id)
   print(doc.metadata)
   ```

2. **Try relaxing filters**:
   ```python
   # Instead of exact match
   {"source": "exact_name.pdf"}
   
   # Try matching multiple
   {"sources": ["name1.pdf", "name2.pdf", "name3.pdf"]}
   ```

3. **Verify date format**:
   ```python
   # Ensure ISO format
   from datetime import datetime
   date_str = datetime.now().isoformat()  # "2024-12-19T10:30:00"
   ```

### Filters Not Applied

1. **Check API parameter name**: Use `metadata_filters` not `filters`
2. **Verify JSON format**: Ensure proper JSON in curl requests
3. **Check collection has metadata**: Not all documents may have metadata fields

### Unexpected Results

1. **Understand "match any" behavior**: Multiple values in lists use OR logic
2. **Check filter combination**: Multiple filters use AND logic
3. **Inspect returned metadata**: Verify documents match expected criteria

## See Also

- [BM25 Filters](./bm25-filters.md) - Filtering with BM25 search
- [Index Mappings](./index-mappings.md) - Managing metadata indices
- [API Documentation](../README.md) - Full API reference
