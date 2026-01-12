# Agentic RAG System - Quick Start

## Overview

Intelligent RAG system with autonomous tool routing, self-reflection, and multi-step planning powered by LangGraph.

## Key Features

✅ **Tool Registry & Router** - Intelligent tool selection with confidence scoring  
✅ **6 Integrated Tools** - Local (3) + External (2) + Hybrid (1)  
✅ **LangGraph Workflow** - Plan → Route → Execute → Reflect  
✅ **Confidence Scoring** - Multi-factor routing algorithm  
✅ **Metrics Tracking** - Complete tool performance monitoring  
✅ **Self-Reflection** - Automatic evaluation and replanning  
✅ **Safe Code Execution** - RestrictedPython sandbox  

## Quick Start

### 1. Install Dependencies

```bash
pip install langgraph langchain-openai tavily-python wikipedia RestrictedPython
```

### 2. Run Demo

```bash
python examples/agent_demo.py
```

### 3. Start API Server

```bash
uvicorn src.main:app --reload
```

### 4. Test the Agent

```bash
# Simple query
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'

# Complex multi-step query
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python and calculate factorial of 5?"}'

# List available tools
curl http://localhost:8000/api/v1/agent/tools

# Get metrics
curl http://localhost:8000/api/v1/agent/metrics
```

## Available Tools

| Tool | Category | Description |
|------|----------|-------------|
| **vectordb_retrieval** | Local | Retrieve documents from knowledge base |
| **reranker** | Local | Rerank documents for relevance |
| **generator** | Local | Generate natural language responses |
| **web_search** | External | Search the web (Tavily/DuckDuckGo) |
| **wikipedia** | External | Search Wikipedia articles |
| **code_executor** | Hybrid | Safe Python code execution |

## Agent Workflow

```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │    PLAN     │  Decompose query
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │    ROUTE    │  Select tool (confidence scoring)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   EXECUTE   │  Run tool
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   REFLECT   │  Evaluate results
                    └──────┬──────┘
                           │
                  ┌────────┴────────┐
                  │                 │
            More tasks?          All done?
                  │                 │
                  ▼                 ▼
              [ROUTE]             [END]
```

## Example Queries

**Knowledge Base Query**:
```bash
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key concepts in RAG?"}'
```
→ Uses: VectorDB → Reranker → Generator

**Real-time Information**:
```bash
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest developments in AI?"}'
```
→ Uses: Web Search → Generator

**Factual Query**:
```bash
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Who invented Python programming language?"}'
```
→ Uses: Wikipedia → Generator

**Computation**:
```bash
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Calculate the sum of numbers from 1 to 100"}'
```
→ Uses: Code Executor

**Complex Multi-step**:
```bash
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning and calculate 5 factorial?"}'
```
→ Uses: VectorDB → Generator → Code Executor

## Configuration

### Optional: Tavily API Key (for better web search)

```bash
export TAVILY_API_KEY="your-api-key-here"
```

Without API key, falls back to free DuckDuckGo search.

### Customize Agent Behavior

Edit `src/services/agent/factory.py`:

```python
create_agent_system(
    enable_web_search=True,      # Enable/disable web search
    enable_wikipedia=True,       # Enable/disable Wikipedia
    enable_code_executor=True,   # Enable/disable code execution
    metrics_storage_path="./logs/agent_metrics.json",
)
```

## Monitoring

View metrics:
```bash
curl http://localhost:8000/api/v1/agent/metrics
```

Example output:
```json
{
  "total_tools": 6,
  "tools": {
    "vectordb_retrieval": {
      "invocations": 42,
      "success_rate": 0.952,
      "avg_latency_ms": 145.3,
      "avg_confidence": 0.82
    }
  }
}
```

## Week 7 Checklist

- [x] Tool registry & router implementation
- [x] Confidence scoring for tool selection
- [x] Local vs external routing
- [x] ≥3 tools integrated (6 total)
- [x] Router success tracking with metrics
- [x] Self-reflection node
- [x] Planning node
- [x] Safe code execution

## Documentation

See [docs/week-plans/week-7-agentic-rag.md](../docs/week-plans/week-7-agentic-rag.md) for complete documentation.

## Troubleshooting

**Issue**: No tools registered  
**Solution**: Ensure RAG pipeline is initialized before agent

**Issue**: Web search not working  
**Solution**: Set TAVILY_API_KEY or use free DuckDuckGo fallback

**Issue**: Code execution fails  
**Solution**: Check RestrictedPython installation: `pip install RestrictedPython`

## Next Steps

1. Run `python examples/agent_demo.py` to see it in action
2. Test API endpoints with sample queries
3. Monitor metrics to optimize routing
4. Add custom tools by extending `BaseTool`
5. Tune confidence thresholds in `router.py`

---

**Built with**: LangGraph, LangChain, FastAPI, RestrictedPython
