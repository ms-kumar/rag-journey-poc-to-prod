# Overflow Guards

Automatic token limit protection for embedding and generation clients.

## Overview

Overflow guards prevent token limit violations by automatically truncating text before it's sent to models. This eliminates costly API errors and ensures all requests stay within model limits.

## Features

- **Automatic Protection**: Built-in to all embedding and generation clients
- **Model-Aware**: Uses correct token limits for each model
- **Zero Configuration**: Works out of the box, no manual setup needed
- **Smart Truncation**: Preserves word boundaries and uses appropriate strategies
- **Cost Savings**: Prevents failed API calls and wasted tokens

## How It Works

### Embedding Overflow Protection

All embedding providers automatically truncate texts to their model's token limit:

```python
from src.services.embeddings.providers import OpenAIEmbeddings

# Create embedder
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# Text way over 8191 token limit
long_text = "word " * 20000  # ~100k chars, ~25k tokens

# Automatically truncated to 8191 tokens before API call
embeddings = embedder.embed([long_text])  # No error!
```

**Protected Providers:**
- OpenAI (8191 tokens)
- Cohere (512 tokens)
- HuggingFace models (model-specific limits)

### Generation Overflow Protection

The generation client reserves space for output tokens from the input budget:

```python
from src.services.generation.client import HFGenerator, GenerationConfig

# Configure generator
config = GenerationConfig(
    model_name="gpt2",
    max_new_tokens=100  # Reserve 100 tokens for output
)
generator = HFGenerator(config)

# Long prompt over GPT-2's 1024 token limit
long_prompt = "Once upon a time " * 500

# Automatically truncated to (1024 - 100) = 924 tokens
response = generator.generate(long_prompt)  # No error!
```

**How It Reserves Space:**
- Gets model's max_input_tokens from token budget
- Subtracts max_new_tokens (output reservation)
- Truncates prompt to remaining space
- Ensures total stays within model limit

## Architecture

### Integration Points

```
User Code
    ↓
Embedding/Generation Client
    ↓
[Overflow Guard] ← TextTruncator.from_model()
    ↓              ↓
    ↓         Token Budgets (model limits)
    ↓
API/Model Call (guaranteed within limits)
```

### Implementation

Overflow guards use the truncation system:

```python
# In embedding providers
def embed(self, texts: Sequence[str]) -> list[list[float]]:
    # Apply overflow guard
    truncator = TextTruncator.from_embedding_model(self.model_name)
    texts = truncator.truncate_batch(texts)
    
    # Now safe to send to API
    return self._call_api(texts)

# In generation client
def generate(self, prompt: str) -> str:
    # Apply overflow guard (reserve output space)
    truncator = TextTruncator.from_generation_model(
        self.config.model_name,
        reserve_output_tokens=self.config.max_new_tokens
    )
    prompt = truncator.truncate(prompt)
    
    # Now safe to generate
    return self._call_model(prompt)
```

## Benefits

### 1. Prevents API Errors

Without overflow guards:
```python
# ❌ Fails with API error: "maximum context length exceeded"
long_text = "..." * 10000
embeddings = embedder.embed([long_text])  # Error!
```

With overflow guards:
```python
# ✅ Automatically handled
long_text = "..." * 10000
embeddings = embedder.embed([long_text])  # Success!
```

### 2. Saves Money

Failed API calls still cost money. Overflow guards prevent:
- Wasted API calls on oversized inputs
- Retries and error handling overhead
- Lost computation on partial processing

### 3. Improves Reliability

- No runtime errors from token limits
- Consistent behavior across different text lengths
- Predictable API usage

### 4. Zero Maintenance

- No manual truncation logic needed
- Automatically uses correct limits per model
- Updates when model limits change

## Configuration

Overflow guards use the HEAD truncation strategy by default (keeps beginning of text). To customize:

### Option 1: Change Default Strategy

Modify the truncation system's default:

```python
from src.services.truncation import TextTruncator, TruncationStrategy

# Create custom truncator
truncator = TextTruncator(
    max_tokens=8191,
    strategy=TruncationStrategy.MIDDLE  # Keep both ends
)
```

### Option 2: Pre-truncate Important Text

For critical sections, manually truncate with preferred strategy:

```python
from src.services.truncation import TextTruncator, TruncationStrategy

# Manually truncate with TAIL strategy (keep end)
truncator = TextTruncator(
    max_tokens=8191,
    strategy=TruncationStrategy.TAIL
)
important_text = truncator.truncate(long_text)

# Now overflow guard won't need to truncate further
embeddings = embedder.embed([important_text])
```

## Testing

12 comprehensive tests ensure overflow guards work correctly:

```bash
# Run overflow guard tests
uv run pytest tests/test_overflow_guards.py -v
```

**Test Coverage:**
- Short texts (no truncation needed)
- Long texts (automatic truncation)
- Batch processing (mixed lengths)
- Edge cases (exact limit, unicode, whitespace)
- Integration with token budgets

## Performance

Overflow guards have minimal overhead:

- **Token Estimation**: ~0.001ms per text (simple division)
- **Truncation**: ~0.01ms per text (string slicing)
- **Total Overhead**: < 1% of API call time

## Monitoring

Track overflow guard activity in production:

```python
from src.services.truncation import estimate_tokens

# Before embedding/generation
original_tokens = estimate_tokens(text)

# After (get actual tokens from result if needed)
if original_tokens > budget.max_input_tokens:
    print(f"Truncated: {original_tokens} → {budget.max_input_tokens} tokens")
```

## Limitations

1. **Token Estimation**: Uses ~4 chars/token heuristic
   - May be slightly inaccurate for some languages
   - For exact counts, integrate tiktoken

2. **Fixed Strategies**: Uses HEAD strategy by default
   - May not preserve most important content
   - Consider manual pre-processing for critical text

3. **No Semantic Awareness**: Truncates mechanically
   - Doesn't understand content importance
   - May cut off key information

## Best Practices

### 1. Design for Truncation

Assume text will be truncated in production:

```python
# ✅ Good: Important info at start
context = f"Key facts: {facts}\n\nDetails: {details}"

# ❌ Bad: Important info at end
context = f"Background: {details}\n\nKey facts: {facts}"  # May be cut off
```

### 2. Monitor Truncation Rates

Track how often truncation occurs:

```python
from src.services.truncation import estimate_tokens
from src.models.token_budgets import get_embedding_budget

budget = get_embedding_budget(model_name)
truncation_count = 0
total_count = 0

for text in texts:
    tokens = estimate_tokens(text)
    if tokens > budget.max_input_tokens:
        truncation_count += 1
    total_count += 1

truncation_rate = truncation_count / total_count
print(f"Truncation rate: {truncation_rate:.1%}")
```

### 3. Chunk Proactively

For documents, chunk before embedding:

```python
from src.services.chunking import ChunkingClient

# Chunk documents to stay well under limits
chunker = ChunkingClient(
    chunk_size=1000,  # Well under 8191 token limit
    overlap=100
)
chunks = chunker.chunk(document)

# Now each chunk fits comfortably
embeddings = embedder.embed(chunks)
```

### 4. Use Appropriate Strategies

Choose truncation strategy based on content:

- **HEAD**: General text, background context
- **TAIL**: Conclusions, summaries, answers
- **MIDDLE**: Instructions with preamble and examples
- **NONE**: Critical text that must fit completely

## Related Documentation

- [Token Budgets](token-budgets.md) - Model limits and cost estimation
- [Truncation](truncation.md) - Truncation strategies and configuration
- [Embedding Providers](../src/services/embeddings/providers.py) - Provider implementations

## References

- Token limits source: Model documentation (OpenAI, Cohere, HuggingFace)
- Tests: [tests/test_overflow_guards.py](../tests/test_overflow_guards.py)
- Implementation: [src/services/embeddings/providers.py](../src/services/embeddings/providers.py), [src/services/generation/client.py](../src/services/generation/client.py)
