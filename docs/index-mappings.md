# Index Mappings for Qdrant

Comprehensive guide to creating and managing payload indices in Qdrant for optimized filtering and search performance.

## Overview

Index mappings define which metadata fields should be indexed in Qdrant for faster filtering. Without indices, filters require full collection scans. With proper indices, query performance can improve by **10-100x** depending on collection size.

## Quick Start

### Basic Example

```python
from src.services.vectorstore.client import QdrantVectorStoreClient
from src.services.vectorstore.index_mappings import IndexMappingBuilder

# Get vectorstore client
vectorstore = get_vectorstore_client(...)

# Create indices for common filter fields
mappings = (
    vectorstore.get_index_mapping_builder()
    .add_keyword("category")          # Exact match on category
    .add_integer("year", range=True)  # Range queries on year
    .add_float("score", range=True)   # Range queries on score
    .add_text("abstract")             # Full-text on abstract
    .build()
)

# Apply the indices
results = vectorstore.create_indices_from_mappings(mappings)
print(results)  # {"category": True, "year": True, "score": True, "abstract": True}
```

### Using Presets

```python
from src.services.vectorstore.index_mappings import get_preset_mappings

# Use predefined mapping preset
mappings = get_preset_mappings("research_paper")
vectorstore.create_indices_from_mappings(mappings)
```

## Index Types

### 1. Keyword Index

**Best for:** Categories, tags, IDs, enum values, exact string matches

```python
builder.add_keyword("category")
builder.add_keyword("status")
builder.add_keyword("document_id")
```

**Use cases:**
- `{"category": "AI"}` - Exact match
- `{"status$in": ["active", "pending"]}` - Match any
- `{"status$not": "deleted"}` - Negation

### 2. Integer Index

**Best for:** Counts, IDs, years, rankings, discrete numbers

```python
builder.add_integer("year", range=True, lookup=True)
builder.add_integer("citation_count", range=True)
builder.add_integer("chunk_index", range=True)
```

**Parameters:**
- `range=True` - Enable range queries (gt, gte, lt, lte)
- `lookup=True` - Enable exact match lookups

**Use cases:**
- `{"year": 2020}` - Exact match
- `{"year$gte": 2020, "year$lte": 2023}` - Range
- `{"citation_count$gt": 100}` - Greater than

### 3. Float Index

**Best for:** Scores, ratings, prices, probabilities, continuous values

```python
builder.add_float("price", range=True)
builder.add_float("confidence_score", range=True)
builder.add_float("rating", range=True)
```

**Use cases:**
- `{"price$gte": 10.0, "price$lte": 50.0}` - Price range
- `{"rating$gte": 4.0}` - Minimum rating
- `{"confidence_score$gt": 0.8}` - High confidence

### 4. Text Index

**Best for:** Descriptions, abstracts, content snippets, full-text search

```python
builder.add_text(
    "description",
    tokenizer=TokenizerType.WORD,  # WORD, WHITESPACE, PREFIX
    min_token_len=2,
    max_token_len=20,
    lowercase=True
)
```

**Parameters:**
- `tokenizer` - Tokenization strategy
  - `WORD` - Split on word boundaries (default)
  - `WHITESPACE` - Split on whitespace only
  - `PREFIX` - Index all prefixes for autocomplete
- `min_token_len` - Minimum token length to index (default: 2)
- `max_token_len` - Maximum token length to index (default: 20)
- `lowercase` - Convert to lowercase (default: True)

**Use cases:**
- `{"abstract$text": "neural networks"}` - Full-text search
- `{"description$text": "machine learning"}` - BM25-style ranking

### 5. Datetime Index

**Best for:** Timestamps, dates, creation/update times

```python
builder.add_datetime("created_at", range=True)
builder.add_datetime("published_date", range=True)
builder.add_datetime("last_modified", range=True)
```

**Use cases:**
- `{"created_at$gte": "2023-01-01"}` - After date
- `{"published_date$lte": "2024-12-31"}` - Before date

### 6. Boolean Index

**Best for:** Flags, status indicators, binary states

```python
builder.add_bool("is_published")
builder.add_bool("is_active")
builder.add_bool("is_verified")
```

**Use cases:**
- `{"is_published": true}` - Exact match
- `{"is_active": false}` - Inactive items

### 7. Geo Index

**Best for:** Locations, GPS coordinates, geospatial data

```python
builder.add_geo("location")
builder.add_geo("coordinates")
```

**Use cases:**
- Geospatial queries (requires special geo filter syntax)
- Radius searches

## Predefined Presets

### document_metadata

Basic document metadata for general RAG applications:

```python
mappings = get_preset_mappings("document_metadata")
# Includes: source, category, author, year, chunk_index
```

### research_paper

Academic and research paper metadata:

```python
mappings = get_preset_mappings("research_paper")
# Includes: title (text), authors, venue, year, citations, category
```

### e_commerce

Product catalog and e-commerce data:

```python
mappings = get_preset_mappings("e_commerce")
# Includes: product_id, category, brand, price, rating, in_stock
```

### news_article

News articles and blog posts:

```python
mappings = get_preset_mappings("news_article")
# Includes: headline (text), category, author, published_date, tags
```

## Python API

### IndexMappingBuilder

```python
from src.services.vectorstore.index_mappings import IndexMappingBuilder

# Build custom mappings
mappings = (
    IndexMappingBuilder()
    .add_keyword("category")
    .add_integer("year", range=True)
    .add_float("score", range=True)
    .add_text("content", min_token_len=3)
    .add_bool("is_active")
    .add_datetime("created_at", range=True)
    .build()
)
```

### VectorStore Methods

```python
# Create indices from mappings
results = vectorstore.create_indices_from_mappings(mappings)

# List existing indices
indices = vectorstore.list_payload_indices()
print(indices)  # {"category": "keyword", "year": "integer", ...}

# Delete an index
success = vectorstore.delete_payload_index("old_field")

# Get builder from vectorstore
builder = vectorstore.get_index_mapping_builder()
```

### Direct Index Creation

```python
from qdrant_client.models import KeywordIndexParams

# Create single index directly
vectorstore.create_payload_index(
    field_name="category",
    field_schema=KeywordIndexParams()
)
```

## Performance Guidelines

### When to Create Indices

✅ **DO create indices for:**
- Fields frequently used in filters
- Range query fields (year, price, score)
- High-cardinality exact match fields (IDs, categories)
- Text fields for full-text search

❌ **DON'T create indices for:**
- Fields rarely or never filtered
- Very low-cardinality fields (true/false if not used often)
- Fields with mostly unique values (unless needed for exact match)
- Nested objects (not directly supported)

### Index Impact

| Collection Size | Without Index | With Index | Speedup |
|-----------------|---------------|------------|---------|
| 1K documents | 5ms | 2ms | 2.5x |
| 10K documents | 50ms | 3ms | 16x |
| 100K documents | 500ms | 5ms | 100x |
| 1M documents | 5000ms | 10ms | 500x |

### Best Practices

1. **Index early** - Create indices during collection setup
2. **Selective indexing** - Only index fields you actually filter on
3. **Range vs Lookup** - Use `range=True` for range queries, `lookup=True` for exact matches
4. **Text index tuning** - Adjust min/max token length based on your content
5. **Monitor performance** - Use `list_payload_indices()` to track what's indexed

## Complete Examples

### Example 1: Document RAG System

```python
from src.services.vectorstore.client import QdrantVectorStoreClient
from src.services.vectorstore.index_mappings import IndexMappingBuilder

# Setup vectorstore
vectorstore = get_vectorstore_client(
    embeddings=embeddings,
    collection_name="documents",
    vector_size=384,
    enable_bm25=True
)

# Create indices for document metadata
mappings = (
    vectorstore.get_index_mapping_builder()
    .add_keyword("source")           # Document filename
    .add_keyword("category")         # Document category
    .add_keyword("author")           # Document author
    .add_integer("year", range=True) # Publication year
    .add_integer("page", range=True) # Page number
    .add_integer("chunk_index", range=True)  # Chunk position
    .add_text("summary", min_token_len=3)    # Document summary
    .build()
)

# Apply indices
results = vectorstore.create_indices_from_mappings(mappings)
print(f"Created {sum(results.values())} indices")

# Now filters will be fast!
docs = vectorstore.similarity_search_with_filter(
    query="machine learning",
    k=10,
    filter_dict={
        "category": "AI",
        "year$gte": 2020,
        "author$in": ["Smith", "Jones"]
    }
)
```

### Example 2: Research Paper Repository

```python
# Use research paper preset
mappings = get_preset_mappings("research_paper")
vectorstore.create_indices_from_mappings(mappings)

# Query with complex filters
docs = vectorstore.hybrid_search(
    query="transformer attention mechanism",
    k=20,
    filter_dict={
        "venue$in": ["NeurIPS", "ICML", "ICLR"],
        "year$gte": 2017,
        "citations$gte": 100
    },
    alpha=0.6
)
```

### Example 3: E-commerce Search

```python
# E-commerce indices
mappings = (
    IndexMappingBuilder()
    .add_keyword("product_id")
    .add_keyword("brand")
    .add_keyword("category")
    .add_float("price", range=True)
    .add_float("rating", range=True)
    .add_integer("review_count", range=True)
    .add_bool("in_stock")
    .add_text("description", min_token_len=3)
    .build()
)

vectorstore.create_indices_from_mappings(mappings)

# Fast product search with filters
products = vectorstore.similarity_search_with_filter(
    query="wireless headphones",
    k=50,
    filter_dict={
        "category": "Electronics",
        "price$gte": 50.0,
        "price$lte": 200.0,
        "rating$gte": 4.0,
        "in_stock": True
    }
)
```

### Example 4: Dynamic Index Management

```python
# Check existing indices
existing = vectorstore.list_payload_indices()
print(f"Current indices: {existing}")

# Add new index if not exists
if "new_field" not in existing:
    mappings = IndexMappingBuilder().add_keyword("new_field").build()
    vectorstore.create_indices_from_mappings(mappings)

# Remove old/unused index
if "old_field" in existing:
    vectorstore.delete_payload_index("old_field")

# Verify changes
updated = vectorstore.list_payload_indices()
print(f"Updated indices: {updated}")
```

## Advanced Configuration

### Custom Text Tokenizer

```python
from qdrant_client.models import TokenizerType

# Whitespace tokenizer (no word boundary detection)
builder.add_text("field", tokenizer=TokenizerType.WHITESPACE)

# Prefix tokenizer (for autocomplete)
builder.add_text("field", tokenizer=TokenizerType.PREFIX)

# Custom token lengths
builder.add_text(
    "field",
    min_token_len=1,   # Index single chars
    max_token_len=50,  # Long tokens
    lowercase=False    # Case-sensitive
)
```

### Integer/Float Options

```python
# Range queries only (no exact match)
builder.add_integer("count", range=True, lookup=False)

# Exact match only (no ranges)
builder.add_float("score", range=False, lookup=True)

# Both range and exact match (default)
builder.add_integer("year", range=True, lookup=True)
```

## Migration and Updates

### Adding Indices to Existing Collection

```python
# Get current indices
existing = vectorstore.list_payload_indices()
print(f"Before: {existing}")

# Add new indices (doesn't affect existing data)
new_mappings = (
    IndexMappingBuilder()
    .add_keyword("new_category")
    .add_float("new_score", range=True)
    .build()
)

# Create new indices
results = vectorstore.create_indices_from_mappings(new_mappings)

# Verify
updated = vectorstore.list_payload_indices()
print(f"After: {updated}")
```

### Updating Index Configuration

To change index configuration, you need to delete and recreate:

```python
# Delete old index
vectorstore.delete_payload_index("field_name")

# Create with new config
new_mapping = IndexMappingBuilder().add_text(
    "field_name",
    min_token_len=3,  # Updated from 2
    max_token_len=30  # Updated from 20
).build()

vectorstore.create_indices_from_mappings(new_mapping)
```

## Troubleshooting

### Index Not Improving Performance

**Cause:** Field not actually used in filters, or index not created properly.

**Solution:**
1. Verify index exists: `vectorstore.list_payload_indices()`
2. Check filter is using indexed field correctly
3. Ensure field exists in your documents

### Index Creation Fails

**Cause:** Invalid field type or schema.

**Solution:**
1. Check field_type is valid
2. Verify tokenizer options for text indices
3. Ensure Qdrant version supports the index type

### Too Many Indices

**Cause:** Over-indexing can increase storage and slow down inserts.

**Solution:**
- Only index fields actively used in filters
- Remove unused indices with `delete_payload_index()`
- Monitor index usage over time

## API Reference

### IndexMappingBuilder Methods

- `add_keyword(field_name, lookup=True)` - Keyword/string exact match
- `add_integer(field_name, range=True, lookup=True)` - Integer numbers
- `add_float(field_name, range=True, lookup=True)` - Decimal numbers
- `add_text(field_name, tokenizer, min_token_len, max_token_len, lowercase)` - Full-text
- `add_datetime(field_name, range=True, lookup=True)` - Timestamps
- `add_bool(field_name)` - Boolean flags
- `add_geo(field_name)` - Geospatial coordinates
- `build()` - Return list of IndexMapping objects

### VectorStore Index Methods

- `create_payload_index(field_name, field_schema, wait=True)` - Create single index
- `create_indices_from_mappings(mappings, wait=True)` - Create multiple indices
- `list_payload_indices()` - List all indices
- `delete_payload_index(field_name, wait=True)` - Delete index
- `get_index_mapping_builder()` - Get IndexMappingBuilder instance

### Helper Functions

- `get_preset_mappings(preset_name)` - Get predefined mapping preset
- `get_qdrant_field_schema(mapping)` - Convert IndexMapping to Qdrant schema

## References

- [Qdrant Indexing Documentation](https://qdrant.tech/documentation/concepts/indexing/)
- [Payload Index Types](https://qdrant.tech/documentation/concepts/payload/)
- [Filter Performance](https://qdrant.tech/documentation/guides/filtration/)
