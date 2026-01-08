# Query Understanding

Query understanding preprocesses user queries before retrieval to improve search quality and recall. It includes query rewriting, synonym expansion, and optional intent classification.

## Features

### 🔄 Query Rewriting
- **Acronym Expansion**: `ML` → `machine learning`, `AI` → `artificial intelligence`
- **Typo Correction**: `machien learing` → `machine learning`
- **Context Addition**: `what is X?` → `X definition explanation`
- **Question Reformulation**: `how to X?` → `X tutorial guide steps`

### 📚 Synonym Expansion
- **Term Expansion**: `model` → `algorithm predictor classifier`
- **Multi-word Phrases**: `machine learning` → `ml statistical learning`
- **Custom Synonyms**: Add domain-specific synonyms
- **Stopword Filtering**: Avoids expanding common words

### 🎯 Intent Classification
- **Factual**: `what is X?`, `define X`
- **How-to**: `how to X?`, `steps to X`
- **Comparison**: `X vs Y`, `difference between X and Y`
- **Troubleshooting**: `error in X`, `X not working`
- **Exploratory**: General queries

## Quick Start

### Basic Usage

```python
from src.services.query_understanding import QueryUnderstanding

# Initialize with defaults
qu = QueryUnderstanding()

# Process a query
result = qu.process("what is ML?")
print(result["processed_query"])
# Output: "Machine learning definition explanation ml statistical learning predictive modeling"
```

### Query Rewriting Only

```python
from src.services.query_understanding import QueryRewriter

rewriter = QueryRewriter()
rewritten, meta = rewriter.rewrite("how to fix machien learing error?")
print(rewritten)
# Output: "fix machine learning error tutorial guide steps"
```

### Synonym Expansion Only

```python
from src.services.query_understanding import SynonymExpander

expander = SynonymExpander()
expanded, meta = expander.expand("neural network training")
print(expanded)
# Output: "neural network training nn artificial neural network learning fitting optimization"
```

## Configuration

### QueryUnderstandingConfig

```python
from src.services.query_understanding import (
    QueryUnderstanding,
    QueryUnderstandingConfig,
)

config = QueryUnderstandingConfig(
    enable_rewriting=True,         # Enable query rewriting
    enable_synonyms=True,           # Enable synonym expansion
    enable_intent_classification=True,  # Enable intent classification
)

qu = QueryUnderstanding(config)
```

### QueryRewriterConfig

```python
from src.services.query_understanding import (
    QueryRewriter,
    QueryRewriterConfig,
)

config = QueryRewriterConfig(
    expand_acronyms=True,      # Expand acronyms (ML → machine learning)
    fix_typos=True,            # Fix common typos
    add_context=True,          # Add context to questions
    max_rewrites=3,            # Maximum number of rewrite operations
    min_query_length=3,        # Minimum query length to process
)

rewriter = QueryRewriter(config)
```

### SynonymExpanderConfig

```python
from src.services.query_understanding import (
    SynonymExpander,
    SynonymExpanderConfig,
)

config = SynonymExpanderConfig(
    max_synonyms_per_term=3,   # Max synonyms to add per term
    min_term_length=3,         # Minimum term length to expand
    expand_all_terms=False,    # If True, expand all terms (not just key terms)
)

expander = SynonymExpander(config)
```

## Advanced Usage

### Full Pipeline with Custom Configs

```python
from src.services.query_understanding import (
    QueryUnderstanding,
    QueryUnderstandingConfig,
    QueryRewriterConfig,
    SynonymExpanderConfig,
)

# Custom rewriter config
rewriter_config = QueryRewriterConfig(
    expand_acronyms=True,
    fix_typos=True,
    add_context=False,  # Disable context addition
    max_rewrites=2,
)

# Custom expander config
expander_config = SynonymExpanderConfig(
    max_synonyms_per_term=2,
    expand_all_terms=False,
)

# Combined config
config = QueryUnderstandingConfig(
    enable_rewriting=True,
    enable_synonyms=True,
    enable_intent_classification=True,
    rewriter_config=rewriter_config,
    expander_config=expander_config,
)

qu = QueryUnderstanding(config)
result = qu.process("what is ML?")
```

### Query Variations for Multi-Query Retrieval

```python
qu = QueryUnderstanding()

# Generate multiple query variations
variations = qu.get_all_variations("what is ML?")

# Use variations for improved recall
for variation in variations:
    results = vectorstore.search(variation, k=10)
    # Combine results with fusion
```

### Custom Synonyms

```python
expander = SynonymExpander()

# Add domain-specific synonyms
expander.add_synonym("rag", [
    "retrieval augmented generation",
    "retrieval-based AI",
    "context-enhanced generation"
])

expanded, _ = expander.expand("explain rag system")
print(expanded)
# Output includes custom synonyms
```

### Intent-Based Processing

```python
config = QueryUnderstandingConfig(enable_intent_classification=True)
qu = QueryUnderstanding(config)

result = qu.process("how to train a model?")

# Use intent to optimize retrieval strategy
if result["intent"] == "howto":
    # Prioritize tutorial/guide documents
    search_strategy = "prioritize_tutorials"
elif result["intent"] == "factual":
    # Prioritize definition/explanation documents
    search_strategy = "prioritize_definitions"
```

## Performance

### Latency Characteristics

From ablation study with 19 test queries:

| Configuration | Avg Latency | P95 Latency | Query Expansion |
|--------------|-------------|-------------|-----------------|
| Baseline (no processing) | 0.01ms | 0.03ms | 1.0x |
| Rewriting only | 0.02ms | 0.06ms | 2.16x |
| Synonyms only | 0.03ms | 0.07ms | 1.88x |
| Rewriting + Synonyms | 0.04ms | 0.09ms | 3.63x |
| **Full pipeline** | **0.08ms** | **0.25ms** | **3.63x** |

**Key Findings:**
- ✅ Full pipeline adds only ~0.08ms overhead (excellent)
- ✅ 3.63x average query expansion improves recall
- ✅ P95 latency < 0.3ms (imperceptible to users)
- ✅ 262.9% longer queries → better matching opportunities

### Optimization Tips

1. **Disable unused features** to minimize latency:
   ```python
   config = QueryUnderstandingConfig(
       enable_rewriting=True,
       enable_synonyms=False,  # Disable if not needed
       enable_intent_classification=False,  # Disable if not needed
   )
   ```

2. **Limit synonym expansion**:
   ```python
   config = SynonymExpanderConfig(
       max_synonyms_per_term=1,  # Reduce from default 3
   )
   ```

3. **Reduce max rewrites**:
   ```python
   config = QueryRewriterConfig(
       max_rewrites=2,  # Reduce from default 3
   )
   ```

## API Examples

### curl Examples

```bash
# Example POST request with query understanding
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "what is ML?",
    "top_k": 5,
    "enable_query_understanding": true
  }'
```

## Use Cases

### 1. Fixing User Typos

```python
rewriter = QueryRewriter()
rewritten, _ = rewriter.rewrite("machien learing algorithim")
# Output: "machine learning algorithm"
```

### 2. Expanding Technical Acronyms

```python
rewriter = QueryRewriter()
rewritten, _ = rewriter.rewrite("what is NLP in AI?")
# Output: "what is Natural language processing in Artificial intelligence?"
```

### 3. Improving Recall with Synonyms

```python
expander = SynonymExpander()
expanded, _ = expander.expand("fast database query")
# Output: "fast database query quick rapid speedy db datastore search request lookup"
```

### 4. Question to Statement Conversion

```python
rewriter = QueryRewriter()
rewritten, _ = rewriter.rewrite("what is machine learning?")
# Output: "machine learning definition explanation"
```

### 5. Multi-Query Retrieval

```python
qu = QueryUnderstanding()
variations = qu.get_all_variations("what is ML?")

# Retrieve with each variation
all_results = []
for query_variant in variations:
    results = vectorstore.search(query_variant, k=20)
    all_results.append(results)

# Fuse results
from src.services.vectorstore.fusion import reciprocal_rank_fusion
fused = reciprocal_rank_fusion({
    f"variant_{i}": results 
    for i, results in enumerate(all_results)
})
```

## Best Practices

1. **Enable rewriting for user-facing queries**:
   - Handles typos and informal language
   - Expands acronyms for better matching

2. **Use synonym expansion for recall**:
   - Adds semantically similar terms
   - Particularly effective for technical domains

3. **Add domain-specific synonyms**:
   ```python
   expander.add_synonym("k8s", ["kubernetes", "container orchestration"])
   ```

4. **Use intent classification to optimize retrieval**:
   - Route how-to queries to tutorial documents
   - Route factual queries to definition documents

5. **Combine with fusion for best results**:
   - Generate query variations
   - Retrieve with each variation
   - Fuse results with RRF or weighted fusion

6. **Monitor latency in production**:
   ```python
   result = qu.process(query)
   latency = result["metadata"]["total_latency_ms"]
   if latency > 10:
       logger.warning(f"High query understanding latency: {latency}ms")
   ```

## Ablation Study

Run the ablation study to measure impact:

```bash
python examples/query_understanding_ablation.py
```

This measures:
- Latency overhead per configuration
- Query expansion ratio
- P50/P95 latency percentiles
- Impact analysis and recommendations

## Demo

Run the demo to see features in action:

```bash
python examples/query_understanding_demo.py
```

Demonstrates:
- Query rewriting examples
- Synonym expansion examples
- Full pipeline processing
- Query variations generation
- Configuration options
- Latency analysis

## Testing

Run tests:

```bash
pytest tests/test_query_understanding.py -v
```

**Test coverage:**
- 47 tests covering all components
- 96% coverage on rewriter (118 statements)
- 92% coverage on expander (64 statements)
- 100% coverage on orchestrator (78 statements)

## Architecture

```
Query → QueryUnderstanding
         │
         ├─→ QueryRewriter
         │    ├─ Fix typos
         │    ├─ Expand acronyms
         │    └─ Add context
         │
         ├─→ SynonymExpander
         │    ├─ Match multi-word phrases
         │    ├─ Add synonyms
         │    └─ Filter stopwords
         │
         └─→ Intent Classifier (optional)
              ├─ Factual
              ├─ How-to
              ├─ Comparison
              ├─ Troubleshooting
              └─ Exploratory
```

## Limitations

1. **Rule-based approach**: Uses predefined rules and dictionaries (no ML models)
2. **English only**: Acronyms, typos, and synonyms are English-focused
3. **Domain coverage**: Synonym dictionary covers ML/AI/programming domains
4. **Context-free**: Does not consider conversation history

## Future Enhancements

- [ ] ML-based query rewriting with T5 or BART
- [ ] Learned synonym expansion from corpus
- [ ] Multi-language support
- [ ] Contextual query rewriting (conversation-aware)
- [ ] Query performance prediction
- [ ] Automatic A/B testing framework
- [ ] Query logs analysis for improvement

## References

- Query rewriting techniques: [Google Search Quality Blog](https://searchquality.google)
- Synonym expansion: [WordNet](https://wordnet.princeton.edu/)
- Intent classification: [BERT for Query Classification](https://arxiv.org/abs/1810.04805)
