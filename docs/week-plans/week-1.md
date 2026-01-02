# Week 1: Naive RAG Pipeline

## Goals

Build a foundational RAG (Retrieval-Augmented Generation) pipeline with basic components to understand the end-to-end flow.

## Completed Tasks

### 1. Project Setup
- [x] Initialize project structure
- [x] Set up FastAPI application
- [x] Configure logging and settings
- [x] Create Makefile for common tasks
- [x] Set up `.gitignore` and `README.md`

### 2. Document Ingestion
- [x] Create ingestion client to load `.txt` files from `data/` directory
- [x] Factory pattern for client instantiation

### 3. Chunking
- [x] Implement fixed-size chunking (512 characters)
- [x] Handle single string and list inputs
- [x] Skip empty chunks

### 4. Embeddings
- [x] Hash-based deterministic embeddings (64 dimensions)
- [x] LangChain-compatible adapter (`embed_documents`, `embed_query`)

### 5. Vector Store (Qdrant)
- [x] Qdrant client integration
- [x] Auto-create collection if not exists
- [x] Handle dimension mismatch (recreate collection)
- [x] `add_texts` for indexing
- [x] `similarity_search` using `query_points`

### 6. Generation
- [x] HuggingFace text-generation pipeline (GPT-2)
- [x] Configurable generation parameters
- [x] Fix deprecation warnings (`max_new_tokens`, `truncation`)

### 7. API Endpoints
- [x] `POST /api/v1/rag/ingest` - Ingest and index documents
- [x] `POST /api/v1/rag/generate` - RAG query endpoint
- [x] `GET /health` - Health check

### 8. Pipeline Orchestration
- [x] `NaivePipeline` class orchestrating all components
- [x] `ingest_and_index()` - Full ingestion flow
- [x] `retrieve()` - Similarity search
- [x] `generate()` - LLM generation with context

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Ingestion  │────▶│  Chunking   │────▶│ Embeddings  │────▶│   Qdrant    │
│  (txt files)│     │ (512 chars) │     │ (64 dim)    │     │ (vectors)   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
                                                                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Answer    │◀────│   GPT-2     │◀────│  Context    │◀────│  Retrieval  │
│             │     │ (generate)  │     │ (top-k docs)│     │ (query)     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `src/main.py` | FastAPI app with lifespan events |
| `src/api/v1/endpoints/rag.py` | RAG API endpoints |
| `src/services/pipeline/naive_pipeline/client.py` | Pipeline orchestration |
| `src/services/vectorstore/client.py` | Qdrant integration |
| `src/services/embeddings/client.py` | Hash-based embeddings |
| `src/services/generation/client.py` | HuggingFace generation |
| `src/services/chunking/client.py` | Document chunking |
| `src/services/ingestion/client.py` | File ingestion |

## Bugs Fixed

1. **Deprecated `on_event`** → Replaced with `lifespan` context manager
2. **Collection not found** → Auto-create collection in Qdrant
3. **Vector dimension mismatch** → Detect and recreate collection
4. **Deprecated LangChain Qdrant** → Use `qdrant-client` directly
5. **`search` method removed** → Use `query_points` instead
6. **`max_length` warning** → Use `max_new_tokens` instead
7. **Truncation warning** → Add `truncation=True`
8. **Chunking single string** → Handle both `str` and `List[str]`

## Known Limitations

- **Hash-based embeddings**: Not semantic, based on text hash
- **GPT-2**: Base model, not instruction-tuned, produces repetitive output
- **Fixed chunk size**: No overlap, may split context awkwardly
- **No re-ranking**: Raw similarity scores only

## API Usage

```bash
# Start Qdrant
docker compose -f infra/docker/compose.yml up -d

# Run server
make run

# Ingest documents
curl -X POST http://localhost:8000/api/v1/rag/ingest

# Query
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is RAG?", "top_k": 3}'
```

## Next Week (Week 2)

- [ ] Replace hash embeddings with sentence-transformers
- [ ] Integrate better LLM (OpenAI API or local Llama)
- [ ] Add chunk overlap for better context
- [ ] Implement re-ranking
- [ ] Add evaluation metrics

---

## Testing Results (10 Queries)

### Test Commands

```bash
# Ensure server is running
make run

# Ingest documents first
curl -X POST http://localhost:8000/api/v1/rag/ingest

# Run queries
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "YOUR_QUERY", "top_k": 3}'
```

### Query Results

| # | Query | Retrieved Chunks | Answer Quality | Failure Mode |
|---|-------|------------------|----------------|--------------|
| 1 | "What is RAG?" | RAG basics chunk | ⚠️ Poor | Repetitive output, GPT-2 echoes question |
| 2 | "How does machine learning work?" | ML short chunk | ⚠️ Poor | Hallucination, generic ML text |
| 3 | "What is FastAPI?" | FastAPI chunk | ⚠️ Poor | Incomplete answer, repetitive |
| 4 | "How to use Qdrant?" | Qdrant notes chunk | ⚠️ Poor | GPT-2 generates irrelevant code |
| 5 | "What is chunk_size?" | RAG basics chunk | ✅ Partial | Found relevant chunk, but poor generation |
| 6 | "Explain supervised learning" | ML short chunk | ⚠️ Poor | Hallucination beyond source |
| 7 | "How to add health endpoint?" | FastAPI chunk | ⚠️ Poor | Retrieval OK, generation fails |
| 8 | "What is a vector database?" | Qdrant notes chunk | ⚠️ Poor | Repetitive "Answer: Answer:" pattern |
| 9 | "What is top_k in RAG?" | RAG basics chunk | ✅ Partial | Correct chunk retrieved |
| 10 | "Who invented Python?" | Random/irrelevant | ❌ Fail | **Hallucination** - not in corpus |

### Observed Failure Modes

#### 1. **Hallucination** (Critical)
- **Description**: Model generates facts not present in retrieved context
- **Example**: Query "Who invented Python?" returns invented answers
- **Cause**: GPT-2 generates based on pre-training, ignores context
- **Frequency**: ~70% of queries

#### 2. **Repetitive Output Pattern**
- **Description**: Output contains "Answer:\n\nAnswer:\n\nn\n\n" patterns
- **Example**: Most queries show this pattern
- **Cause**: GPT-2 base model has no instruction tuning
- **Frequency**: ~80% of queries

#### 3. **Irrelevant Chunk Retrieval**
- **Description**: Retrieved chunks don't match query semantically
- **Example**: Query about Python returns Qdrant chunks
- **Cause**: Hash-based embeddings are not semantic
- **Frequency**: ~40% of queries

#### 4. **Context Ignored**
- **Description**: Model generates answer without using retrieved context
- **Example**: Answer doesn't reference information in chunks
- **Cause**: No prompt engineering to force context usage
- **Frequency**: ~60% of queries

#### 5. **Incomplete Answers**
- **Description**: Generation stops mid-sentence or is too short
- **Example**: Truncated explanations
- **Cause**: Token limits and generation parameters
- **Frequency**: ~30% of queries

### Root Cause Analysis

| Issue | Root Cause | Fix (Week 2+) |
|-------|------------|---------------|
| Irrelevant retrieval | Hash embeddings not semantic | Use sentence-transformers |
| Hallucination | GPT-2 base model | Use instruction-tuned LLM |
| Repetitive output | No prompt template | Add RAG prompt template |
| Context ignored | No context injection | Format prompt with context |
| Poor quality | All of the above | Full pipeline upgrade |

### Recommendations for Week 2

1. **Embeddings**: Replace hash-based with `all-MiniLM-L6-v2` (384 dim)
2. **LLM**: Use OpenAI GPT-3.5/4 or local Llama-2-7B-chat
3. **Prompt Template**: 
   ```
   Context: {retrieved_chunks}
   
   Question: {query}
   
   Answer based only on the context above:
   ```
4. **Re-ranking**: Add cross-encoder for better precision
5. **Evaluation**: Add RAGAS or similar metrics
