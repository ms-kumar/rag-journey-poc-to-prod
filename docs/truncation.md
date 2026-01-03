# Text Truncation Rules

This document describes text truncation strategies and rules for managing token limits in the RAG system.

## Overview

The truncation system ensures text stays within model token budgets by automatically truncating when necessary. It provides multiple strategies to preserve the most important content.

## Quick Start

```python
from src.services.truncation import TextTruncator, TruncationStrategy

# Create truncator for a specific model
truncator = TextTruncator.from_embedding_model("text-embedding-3-small")

# Truncate text
long_text = "..." * 10000
truncated = truncator.truncate(long_text)
```

## Truncation Strategies

### 1. HEAD (Keep Beginning)
Preserves the start of the text, truncates the end.

**Use when:**
- Context is frontloaded (summaries, introductions)
- Most important information is at the beginning
- Processing search results or abstracts

```python
truncator = TextTruncator(max_tokens=100, strategy=TruncationStrategy.HEAD)
result = truncator.truncate("Important intro... less important ending...")
# Result: "Important intro... less important..."
```

### 2. TAIL (Keep End)
Preserves the end of the text, truncates the beginning.

**Use when:**
- Conclusions or final statements are most important
- Processing chat histories (keep recent messages)
- Log analysis (recent entries matter most)

```python
truncator = TextTruncator(max_tokens=100, strategy=TruncationStrategy.TAIL)
result = truncator.truncate("Old context... important recent information")
# Result: "...important recent information"
```

### 3. MIDDLE (Keep Both Ends)
Preserves beginning and end, removes the middle.

**Use when:**
- Both introduction and conclusion are important
- Need context from start and end
- Processing documents where middle is repetitive

```python
truncator = TextTruncator(max_tokens=100, strategy=TruncationStrategy.MIDDLE)
result = truncator.truncate("Intro... repetitive middle content... Conclusion")
# Result: "Intro ... Conclusion"
```

### 4. NONE (Error on Exceed)
Raises an error if text exceeds limit.

**Use when:**
- Token limits are strict requirements
- Want to catch oversized inputs early
- Need explicit handling of limit violations

```python
truncator = TextTruncator(max_tokens=100, strategy=TruncationStrategy.NONE)
try:
    result = truncator.truncate(very_long_text)
except ValueError as e:
    # Handle oversized input
    pass
```

## Token Estimation

The system uses a conservative estimation: **~4 characters per token**.

```python
from src.services.truncation import estimate_tokens, chars_from_tokens

# Estimate tokens in text
token_count = estimate_tokens("Hello, world!")  # ~3 tokens

# Convert tokens to characters
char_count = chars_from_tokens(100)  # 400 characters
```

**Note**: For production systems, consider using [tiktoken](https://github.com/openai/tiktoken) for accurate token counting with specific tokenizers.

## Usage Examples

### With Embedding Models

```python
from src.services.truncation import TextTruncator, TruncationStrategy

# Auto-configure for embedding model
truncator = TextTruncator.from_embedding_model(
    "text-embedding-3-small",
    strategy=TruncationStrategy.HEAD,
    preserve_words=True
)

# Truncate document
doc = "Very long document..." * 1000
truncated_doc = truncator.truncate(doc)
```

### With Generation Models

```python
# Reserve tokens for output
truncator = TextTruncator.from_generation_model(
    "gpt-3.5-turbo",
    strategy=TruncationStrategy.MIDDLE,
    reserve_output_tokens=500  # Reserve 500 tokens for response
)

# Truncate prompt
prompt = f"Context: {long_context}\n\nQuestion: {question}"
truncated_prompt = truncator.truncate(prompt)
```

### Batch Processing

```python
# Truncate multiple texts at once
texts = [doc1, doc2, doc3, ...]
truncated_texts = truncator.truncate_batch(texts)
```

### Direct Budget Truncation

```python
from src.models.token_budgets import get_embedding_budget
from src.services.truncation import truncate_to_budget

budget = get_embedding_budget("text-embedding-3-small")
truncated = truncate_to_budget(
    text,
    budget,
    strategy=TruncationStrategy.HEAD,
    reserve_output_tokens=0
)
```

## Word Boundary Preservation

By default, truncation preserves word boundaries to avoid breaking words.

```python
# With word preservation (default)
truncator = TextTruncator(max_tokens=10, preserve_words=True)
result = truncator.truncate("This is a very long sentence")
# Result: "This is a very..."  (breaks at space)

# Without word preservation
truncator = TextTruncator(max_tokens=10, preserve_words=False)
result = truncator.truncate("This is a very long sentence")
# Result: "This is a ve..."  (may break mid-word)
```

## Splitting with Overlap

For chunking documents into multiple parts:

```python
from src.services.truncation import split_with_overlap

text = "Long document..." * 1000

# Split into chunks with overlap
chunks = split_with_overlap(
    text,
    max_tokens=512,    # Max tokens per chunk
    overlap_tokens=50  # Overlap between chunks
)

# Process each chunk
for chunk in chunks:
    process_chunk(chunk)
```

**Use cases:**
- Document indexing for RAG
- Processing long texts in smaller batches
- Maintaining context across chunks

## Configuration Guidelines

### Embedding Models

| Model | Max Tokens | Recommended Strategy |
|-------|-----------|---------------------|
| OpenAI embeddings | 8,191 | HEAD (keep beginning) |
| Cohere embeddings | 512 | HEAD or MIDDLE |
| E5/BGE models | 512 | HEAD or MIDDLE |

### Generation Models

| Model | Context Window | Reserve Output | Strategy |
|-------|---------------|----------------|----------|
| GPT-4 Turbo | 128,000 | 4,096 | MIDDLE |
| GPT-3.5 Turbo | 16,385 | 2,048 | MIDDLE |
| Claude 3 | 200,000 | 4,096 | MIDDLE |
| Local models | 1,024-8,192 | 512-1,024 | HEAD |

## Best Practices

### 1. Choose the Right Strategy

```python
# Search queries: keep beginning (main question)
search_truncator = TextTruncator(
    max_tokens=512,
    strategy=TruncationStrategy.HEAD
)

# Chat history: keep recent (tail)
chat_truncator = TextTruncator(
    max_tokens=2048,
    strategy=TruncationStrategy.TAIL
)

# Documents: keep intro and conclusion (middle)
doc_truncator = TextTruncator(
    max_tokens=1024,
    strategy=TruncationStrategy.MIDDLE
)
```

### 2. Reserve Output Tokens

Always reserve tokens for model output when using generation models:

```python
# Reserve 25% for output
context_window = 4096
reserve = context_window // 4  # 1024 tokens

truncator = TextTruncator(
    max_tokens=context_window - reserve,
    strategy=TruncationStrategy.MIDDLE
)
```

### 3. Preserve Word Boundaries

Enable word preservation for better readability:

```python
truncator = TextTruncator(
    max_tokens=512,
    preserve_words=True  # Default
)
```

### 4. Handle Edge Cases

```python
# Empty text
truncator.truncate("")  # Returns ""

# Text within limit
truncator.truncate("short")  # Returns "short"

# Very long text
truncator.truncate("x" * 100000)  # Truncated to limit
```

## Integration with RAG Pipeline

### Pre-Embedding Truncation

```python
from src.services.truncation import TextTruncator
from src.services.embeddings.factory import get_embed_client

# Create truncator for embedding model
truncator = TextTruncator.from_embedding_model("intfloat/e5-base-v2")
embed_client = get_embed_client(config)

# Truncate before embedding
chunks = ["chunk1...", "chunk2...", ...]
truncated_chunks = truncator.truncate_batch(chunks)
embeddings = embed_client.embed(truncated_chunks)
```

### Pre-Generation Truncation

```python
from src.services.truncation import TextTruncator
from src.services.generation.factory import get_generator

# Create truncator with output reservation
truncator = TextTruncator.from_generation_model(
    "gpt-3.5-turbo",
    strategy=TruncationStrategy.MIDDLE,
    reserve_output_tokens=512
)

# Build and truncate prompt
retrieved_docs = [doc1, doc2, doc3]
context = "\n\n".join(retrieved_docs)
prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

truncated_prompt = truncator.truncate(prompt)
generator = get_generator()
answer = generator.generate(truncated_prompt)
```

## Performance Considerations

### Token Estimation Speed

Token estimation is **O(n)** where n is text length:

```python
# Fast for small texts
estimate_tokens("short text")  # ~0.001ms

# Scales linearly
estimate_tokens("text" * 10000)  # ~0.1ms
```

### Truncation Speed

Truncation is **O(n)** for string operations:

```python
# HEAD/TAIL: Fast (single slice)
truncator.truncate(text)  # ~0.01ms

# MIDDLE: Slightly slower (two slices)
truncator.truncate(text)  # ~0.02ms
```

### Optimization Tips

1. **Cache truncators**: Reuse instances for multiple truncations
2. **Batch processing**: Use `truncate_batch()` for multiple texts
3. **Early truncation**: Truncate before expensive operations (embedding, generation)

## Monitoring & Debugging

### Track Truncation Events

```python
class TruncationMonitor:
    def __init__(self):
        self.truncation_count = 0
        self.total_chars_removed = 0
    
    def track(self, original: str, truncated: str):
        if len(original) > len(truncated):
            self.truncation_count += 1
            self.total_chars_removed += len(original) - len(truncated)

monitor = TruncationMonitor()
result = truncator.truncate(text)
monitor.track(text, result)
```

### Log Truncation Warnings

```python
import logging

logger = logging.getLogger(__name__)

original_length = estimate_tokens(text)
if original_length > truncator.max_tokens:
    logger.warning(
        f"Truncating text: {original_length} tokens > {truncator.max_tokens} limit"
    )
    truncated = truncator.truncate(text)
```

## Limitations

1. **Token Estimation**: Uses character-based heuristic (~4 chars/token)
   - May be inaccurate for non-English text
   - Consider using tiktoken for exact counts

2. **Context Loss**: Truncation always loses information
   - Use chunking with overlap for better coverage
   - Consider summarization for extreme cases

3. **Word Boundaries**: Preservation may slightly exceed limit
   - Usually by 1-2 tokens
   - Acceptable trade-off for readability

## Future Enhancements

Planned improvements:

- [ ] Tiktoken integration for exact token counts
- [ ] Smart truncation using sentence boundaries
- [ ] Importance-based truncation (keep key sentences)
- [ ] Configurable ellipsis markers
- [ ] Truncation statistics tracking
- [ ] Multi-language token estimation

## References

- [Token Budgets Documentation](./token-budgets.md)
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer)
- [tiktoken Library](https://github.com/openai/tiktoken)
- [HuggingFace Tokenizers](https://huggingface.co/docs/tokenizers/)
