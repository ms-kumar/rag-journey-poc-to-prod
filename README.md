# Advanced RAG

A Retrieval-Augmented Generation (RAG) system built with FastAPI, Qdrant, and HuggingFace.

## Week 1: Naive RAG Pipeline

This week focuses on building a foundational RAG pipeline with the following components:

### Architecture

```
Documents → Chunking → Embeddings → Vector Store (Qdrant)
                                          ↓
Query → Embedding → Similarity Search → Retrieved Chunks → LLM → Answer
```

### Components Built

| Component | Description |
|-----------|-------------|
| **Ingestion** | Load `.txt` files from `data/` directory |
| **Chunking** | Split documents into 512-character chunks |
| **Embeddings** | Hash-based deterministic embeddings (64 dim) |
| **Vector Store** | Qdrant for similarity search |
| **Generation** | GPT-2 for text generation |
| **API** | FastAPI endpoints for ingest and generate |

### Project Structure

```
src/
├── main.py                 # FastAPI app entry point
├── config.py               # App configuration
├── api/
│   └── v1/endpoints/
│       └── rag.py          # RAG endpoints (ingest, generate)
├── models/
│   └── rag_request.py      # Pydantic models
└── services/
    ├── chunking/           # Document chunking
    ├── embeddings/         # Text embeddings
    ├── generation/         # LLM generation
    ├── ingestion/          # Document loading
    ├── pipeline/           # RAG orchestration
    └── vectorstore/        # Qdrant integration
```

### Quick Start

1. **Start Qdrant** (using Docker):
   ```bash
   docker compose -f infra/docker/compose.yml up -d
   ```

2. **Install dependencies**:
   ```bash
   pip install -e .
   ```

3. **Run the server**:
   ```bash
   make run
   # or
   python -m src.main
   ```

4. **Ingest documents**:
   ```bash
   make ingest
   # or
   curl -X POST http://localhost:8000/api/v1/rag/ingest
   ```

5. **Query the RAG system**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/rag/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "What is RAG?", "top_k": 3}'
   ```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/rag/ingest` | POST | Ingest and index documents |
| `/api/v1/rag/generate` | POST | Generate answer using RAG |

### Development

```bash
# Format code
make format

# Check formatting
make check

# Lint code
make lint

# Clean cache files
make clean

# Run tests
make test
```

### Week 1 Limitations

- Simple hash-based embeddings (not semantic)
- Basic GPT-2 model (not instruction-tuned)
- Fixed chunk size without overlap
- No re-ranking or hybrid search

### Next Steps (Week 2+)

- [ ] Semantic embeddings (sentence-transformers)
- [ ] Better LLM integration (OpenAI, Anthropic)
- [ ] Chunk overlap and smarter splitting
- [ ] Re-ranking for improved precision
- [ ] Hybrid search (dense + sparse)
