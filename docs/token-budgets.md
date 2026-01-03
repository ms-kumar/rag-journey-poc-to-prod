# Token Budget Management

This document describes token budget definitions and cost management for embedding and generation models in the RAG system.

## Overview

Token budgets define the operational constraints and costs for different models:

- **Context windows**: Maximum input tokens supported
- **Output limits**: Maximum tokens the model can generate
- **API costs**: Per-token pricing for commercial models
- **Batch recommendations**: Optimal batch sizes for performance

## Architecture

```python
from src.models.token_budgets import (
    get_embedding_budget,
    get_generation_budget,
    estimate_cost
)

# Get budget for a model
budget = get_embedding_budget("text-embedding-3-small")
print(f"Max tokens: {budget.max_input_tokens}")
print(f"Cost per 1K: ${budget.cost_per_1k_input}")

# Estimate operation cost
cost = estimate_cost("gpt-4-turbo", input_tokens=5000, output_tokens=1000)
print(f"Estimated cost: ${cost:.4f}")
```

## Embedding Models

### OpenAI Embeddings

| Model | Max Input | Context Window | Cost per 1M tokens | Batch Size |
|-------|-----------|----------------|-------------------|------------|
| `text-embedding-3-small` | 8,191 | 8,191 | $0.02 | 2,048 |
| `text-embedding-3-large` | 8,191 | 8,191 | $0.13 | 2,048 |
| `text-embedding-ada-002` | 8,191 | 8,191 | $0.10 | 2,048 |

### Cohere Embeddings

| Model | Max Input | Context Window | Cost per 1M tokens | Batch Size |
|-------|-----------|----------------|-------------------|------------|
| `embed-english-v3.0` | 512 | 512 | $0.10 | 96 |
| `embed-multilingual-v3.0` | 512 | 512 | $0.10 | 96 |

### Local HuggingFace Models

| Model | Max Input | Context Window | Cost | Batch Size |
|-------|-----------|----------------|------|------------|
| `intfloat/e5-small-v2` | 512 | 512 | Free | 32 |
| `intfloat/e5-base-v2` | 512 | 512 | Free | 32 |
| `intfloat/e5-large-v2` | 512 | 512 | Free | 16 |
| `BAAI/bge-small-en-v1.5` | 512 | 512 | Free | 32 |
| `BAAI/bge-base-en-v1.5` | 512 | 512 | Free | 32 |
| `BAAI/bge-large-en-v1.5` | 512 | 512 | Free | 16 |
| `simple-hash` | 100,000 | 100,000 | Free | 1,000 |

## Generation Models

### OpenAI Models

| Model | Context Window | Max Output | Cost (Input) | Cost (Output) |
|-------|----------------|------------|--------------|---------------|
| `gpt-4-turbo` | 128,000 | 4,096 | $10/1M | $30/1M |
| `gpt-4` | 8,192 | 8,192 | $30/1M | $60/1M |
| `gpt-3.5-turbo` | 16,385 | 4,096 | $1.50/1M | $2.00/1M |

### Anthropic Claude

| Model | Context Window | Max Output | Cost (Input) | Cost (Output) |
|-------|----------------|------------|--------------|---------------|
| `claude-3-opus` | 200,000 | 4,096 | $15/1M | $75/1M |
| `claude-3-sonnet` | 200,000 | 4,096 | $3/1M | $15/1M |
| `claude-3-haiku` | 200,000 | 4,096 | $0.25/1M | $1.25/1M |

### Local HuggingFace Models

| Model | Context Window | Max Output | Cost | Batch Size |
|-------|----------------|------------|------|------------|
| `gpt2` | 1,024 | 1,024 | Free | 8 |
| `gpt2-medium` | 1,024 | 1,024 | Free | 4 |
| `gpt2-large` | 1,024 | 1,024 | Free | 2 |
| `meta-llama/Llama-2-7b-hf` | 4,096 | 4,096 | Free | 4 |
| `meta-llama/Llama-2-13b-hf` | 4,096 | 4,096 | Free | 2 |
| `mistralai/Mistral-7B-v0.1` | 8,192 | 8,192 | Free | 4 |

## Usage Examples

### Check Model Budget

```python
from src.models.token_budgets import get_embedding_budget, get_generation_budget

# Get embedding model budget
budget = get_embedding_budget("text-embedding-3-small")
print(f"Max input tokens: {budget.max_input_tokens}")
print(f"Recommended batch size: {budget.recommended_batch_size}")

# Get generation model budget
gen_budget = get_generation_budget("gpt-4-turbo")
print(f"Context window: {gen_budget.max_context_window}")
print(f"Max output: {gen_budget.max_output_tokens}")
```

### Estimate Costs

```python
from src.models.token_budgets import estimate_cost

# Embedding cost
embedding_cost = estimate_cost(
    model_name="text-embedding-3-small",
    input_tokens=10000,
    is_embedding=True
)
print(f"Embedding 10K tokens: ${embedding_cost:.4f}")

# Generation cost
generation_cost = estimate_cost(
    model_name="gpt-4-turbo",
    input_tokens=5000,
    output_tokens=1000,
    is_embedding=False
)
print(f"Generating 1K tokens: ${generation_cost:.4f}")
```

### Cost Estimation Example

```python
# Scenario: RAG pipeline with 100 queries
queries = 100
avg_chunks_per_query = 5
chunk_size = 200  # tokens
output_tokens = 200  # per response

# Embedding cost (assuming E5 local model)
embedding_cost = estimate_cost(
    "intfloat/e5-small-v2",
    input_tokens=queries * chunk_size,
    is_embedding=True
)

# Generation cost (GPT-3.5-turbo)
gen_cost = estimate_cost(
    "gpt-3.5-turbo",
    input_tokens=queries * avg_chunks_per_query * chunk_size,
    output_tokens=queries * output_tokens,
    is_embedding=False
)

total_cost = embedding_cost + gen_cost
print(f"Total cost for 100 queries: ${total_cost:.2f}")
# Output: ~$0.35 (free embeddings + $0.35 for generation)
```

## Budget Enforcement

### Automatic Validation

The system automatically validates inputs against token budgets:

```python
from src.models.token_budgets import get_embedding_budget

budget = get_embedding_budget("text-embedding-3-small")

# Check if text exceeds budget
text_length = 10000  # tokens
if text_length > budget.max_input_tokens:
    # Split into chunks or truncate
    chunks = split_into_chunks(text, budget.max_input_tokens)
```

### Batch Size Optimization

Use recommended batch sizes for optimal performance:

```python
budget = get_embedding_budget("intfloat/e5-base-v2")
batch_size = budget.recommended_batch_size  # 32

# Process in optimal batches
for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    embeddings = embed_client.embed(batch)
```

## Cost Optimization Strategies

### 1. Use Local Models When Possible

```python
# Instead of: EMBED_PROVIDER=openai EMBED_MODEL=text-embedding-3-small
# Use: EMBED_PROVIDER=e5 EMBED_MODEL=intfloat/e5-base-v2
# Savings: $0.02 per 1M tokens → Free
```

### 2. Cache Embeddings Aggressively

```python
# Enable caching (already built-in)
EMBED_CACHE_ENABLED=true
EMBED_CACHE_MAX_SIZE=10000

# With 80% cache hit rate, reduce costs by 80%
```

### 3. Optimize Chunk Sizes

```python
# Balance context vs cost
# Smaller chunks = more embeddings = higher cost
# Larger chunks = better context but may exceed limits

CHUNK_SIZE=512  # Good balance for most models
```

### 4. Use Cheaper Generation Models

```python
# For simple tasks:
# GPT-3.5-turbo ($1.50/1M) vs GPT-4-turbo ($10/1M)
# Claude-3-haiku ($0.25/1M) vs Claude-3-opus ($15/1M)
```

## Adding New Models

To add a new model budget:

1. **Define the budget** in `src/models/token_budgets.py`:

```python
class GenerationModelBudgets:
    MY_NEW_MODEL: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=4096,
        max_output_tokens=4096,
        max_context_window=4096,
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.002,
        supports_batching=True,
        recommended_batch_size=8,
    )
```

2. **Register in mapping**:

```python
GENERATION_MODEL_BUDGETS = {
    # ... existing models
    "my-new-model": GenerationModelBudgets.MY_NEW_MODEL,
}
```

3. **Add tests** in `tests/test_token_budgets.py`:

```python
def test_new_model_budget(self):
    budget = get_generation_budget("my-new-model")
    assert budget.max_input_tokens == 4096
```

## Real-World Cost Examples

### Scenario 1: Document Ingestion
- 1,000 documents
- 500 tokens per document
- E5 embeddings (local)
- **Cost: $0.00** (free local model)

### Scenario 2: Production RAG (1M queries/month)
- 1M queries
- 3 documents retrieved per query @ 200 tokens each
- GPT-3.5-turbo generation @ 150 tokens output
- Text-embedding-3-small for embeddings

**Costs:**
- Embeddings: 1M × 200 tokens = $0.004
- Generation input: 1M × 600 tokens = $0.90
- Generation output: 1M × 150 tokens = $0.30
- **Total: ~$1.20/month**

### Scenario 3: High-Quality RAG
- Same as Scenario 2, but with GPT-4-turbo

**Costs:**
- Embeddings: $0.004
- Generation input: 1M × 600 tokens = $6.00
- Generation output: 1M × 150 tokens = $4.50
- **Total: ~$10.50/month**

## Monitoring Costs

### Track Usage

```python
from src.models.token_budgets import estimate_cost

# Log costs for each operation
total_cost = 0.0

for query in queries:
    # Embedding cost
    emb_cost = estimate_cost("text-embedding-3-small", len(query), is_embedding=True)
    
    # Generation cost
    gen_cost = estimate_cost("gpt-3.5-turbo", context_tokens, output_tokens, is_embedding=False)
    
    total_cost += emb_cost + gen_cost
    
print(f"Total cost: ${total_cost:.2f}")
```

### Set Budget Alerts

```python
MONTHLY_BUDGET = 100.0  # USD
current_spend = calculate_current_spend()

if current_spend > MONTHLY_BUDGET * 0.8:
    send_alert("Approaching budget limit")
```

## References

- [OpenAI Pricing](https://openai.com/pricing)
- [Anthropic Pricing](https://www.anthropic.com/pricing)
- [Cohere Pricing](https://cohere.com/pricing)
- [HuggingFace Models](https://huggingface.co/models)
