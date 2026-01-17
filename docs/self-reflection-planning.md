# Self-Reflection & Planning for Agentic RAG

This implementation adds comprehensive self-reflection and planning capabilities to the Agentic RAG system, enabling continuous improvement and higher quality answers.

## 🎯 Features Implemented

### 1. Advanced Query Planning (`planning.py`)
**Purpose**: Intelligently decompose complex queries into executable subtasks

**Key Components**:
- `QueryPlanner`: Analyzes query complexity and creates execution plans
- `Task`: Represents individual subtasks with dependencies and metadata
- `ExecutionPlan`: Complete plan with tasks, strategy, and rationale

**Capabilities**:
- ✅ Automatic complexity detection (simple/moderate/complex)
- ✅ Smart task decomposition with dependencies
- ✅ Task type identification (retrieval, comparison, computation, synthesis, etc.)
- ✅ Tool suggestions for each task
- ✅ Execution strategy selection (sequential/parallel/hybrid)
- ✅ Plan revision based on feedback
- ✅ Failure analysis and remediation

**Example**:
```python
from src.services.agent.planning import QueryPlanner

planner = QueryPlanner()
plan = planner.create_plan("Compare Python and Java for data science")

# Output: 3-4 tasks with dependencies, tool hints, and execution strategy
for task in plan.tasks:
    print(f"{task.id}: {task.description}")
    print(f"  Tools: {task.tool_hints}")
    print(f"  Type: {task.type}")
```

---

### 2. Answer Critique System (`reflection.py`)
**Purpose**: Evaluate answer quality across multiple dimensions

**Key Components**:
- `AnswerCritic`: Critiques answers for quality and completeness
- `AnswerCritique`: Detailed assessment with scores and suggestions
- `SourceVerifier`: Verifies sources and detects hallucinations
- `SourceVerification`: Source quality and hallucination risk assessment

**Quality Dimensions**:
1. **Completeness** (0.0-1.0): Does it fully answer the query?
2. **Accuracy** (0.0-1.0): Are claims supported by sources?
3. **Relevance** (0.0-1.0): Is it on-topic?
4. **Clarity** (0.0-1.0): Is it well-structured and clear?
5. **Source Quality** (0.0-1.0): Are sources reliable?

**Capabilities**:
- ✅ Multi-dimensional quality scoring
- ✅ Issue identification (incomplete, unclear, unsupported claims)
- ✅ Improvement suggestions
- ✅ Missing aspect detection
- ✅ Source diversity assessment
- ✅ Hallucination risk estimation
- ✅ Questionable claim identification

**Example**:
```python
from src.services.agent.reflection import AnswerCritic, SourceVerifier

critic = AnswerCritic(quality_threshold=0.7)
critique = critic.critique_answer(
    answer="Python is a high-level programming language...",
    query="What is Python?",
    sources=retrieval_sources
)

print(f"Overall Quality: {critique.overall_score:.2f}")
print(f"Needs Revision: {critique.needs_revision}")
print(f"Issues: {critique.issues}")
print(f"Suggestions: {critique.suggestions}")

# Verify sources
verifier = SourceVerifier()
verification = verifier.verify_sources(answer, sources)
print(f"Hallucination Risk: {verification.hallucination_risk:.2f}")
print(f"Questionable Claims: {verification.questionable_claims}")
```

---

### 3. User Feedback System (`feedback.py`)
**Purpose**: Collect and analyze user feedback for continuous improvement

**Key Components**:
- `FeedbackLogger`: Logs and stores user feedback
- `UserFeedback`: Structured feedback with ratings and issues
- `FeedbackAnalytics`: Analytics derived from feedback data

**Capabilities**:
- ✅ Structured feedback logging (1-5 star ratings)
- ✅ Issue tracking (incomplete, unclear, wrong, etc.)
- ✅ Persistent storage (JSON format)
- ✅ Analytics and trends (average rating, positive rate)
- ✅ Common issue identification
- ✅ Success pattern detection
- ✅ Low-rated query tracking
- ✅ Export to JSON/CSV

**Example**:
```python
from src.services.agent.feedback import FeedbackLogger, UserFeedback

logger = FeedbackLogger(storage_path="logs/user_feedback.json")

# Log feedback
feedback = UserFeedback(
    query="What is machine learning?",
    answer="ML is...",
    rating=4,
    feedback_text="Good but needs more examples",
    issues=["incomplete"]
)
logger.log_feedback(feedback)

# Get analytics
analytics = logger.get_analytics(days=7)
print(f"Average Rating: {analytics.average_rating:.2f}")
print(f"Positive Rate: {analytics.positive_feedback_rate:.1%}")
print(f"Common Issues: {analytics.common_issues}")
print(f"Success Patterns: {analytics.success_patterns}")

# Find problem queries
low_rated = logger.get_low_rated_queries(threshold=2)
```

---

### 4. Task Benchmarking (`benchmarking.py`)
**Purpose**: Measure and optimize complex task performance

**Key Components**:
- `TaskBenchmarker`: Benchmarks task execution
- `TaskBenchmark`: Individual task performance metrics
- `ComplexQueryBenchmark`: Complete query execution benchmark

**Metrics Tracked**:
- ⏱️ Execution time per task
- ✅ Success/failure rates
- 🎯 Quality scores
- 🔄 Retry counts
- 🔧 Tool selection confidence
- 📊 Overall query performance

**Capabilities**:
- ✅ Per-task execution timing
- ✅ Success rate tracking
- ✅ Quality score measurement
- ✅ Tool usage analytics
- ✅ Slow query identification
- ✅ Performance bottleneck detection
- ✅ Improvement area identification
- ✅ Export to JSON

**Example**:
```python
from src.services.agent.benchmarking import TaskBenchmarker

benchmarker = TaskBenchmarker(storage_path="logs/benchmarks.jsonl")

# Start benchmark
benchmark = benchmarker.start_benchmark(
    query="Complex query...",
    plan_complexity="complex",
    num_tasks=3
)

# Record each task
benchmarker.record_task(
    benchmark=benchmark,
    task_id="task_1",
    task_description="Retrieve data",
    execution_time=1.2,
    success=True,
    tool_used="vectordb_retrieval",
    quality_score=0.85
)

# Complete benchmark
benchmarker.complete_benchmark(benchmark, final_quality_score=0.88)

# Get statistics
stats = benchmarker.get_statistics()
print(f"Success Rate: {stats['success_rate']:.1%}")
print(f"Avg Time: {stats['average_execution_time']:.2f}s")
print(f"Tool Usage: {stats['tool_usage']}")
```

---

## 🔄 Integration with Agent Workflow

The enhanced `AgentNodes` class now supports all self-reflection features:

### Enhanced Agent Initialization

```python
from src.services.agent.nodes import AgentNodes
from src.services.agent.tools.registry import get_tool_registry
from src.services.agent.tools.router import AgentRouter

nodes = AgentNodes(
    registry=get_tool_registry(),
    router=AgentRouter(get_tool_registry()),
    llm=your_llm,
    enable_reflection=True,      # Enable answer critique
    enable_planning=True,         # Enable advanced planning
    enable_benchmarking=True,     # Enable performance tracking
)
```

### Workflow Changes

**Planning Node**:
- Uses `QueryPlanner` to create sophisticated execution plans
- Provides complexity analysis and task decomposition
- Starts benchmarking if enabled

**Execute Node**:
- Tracks execution time per task
- Records metrics in benchmark
- Provides detailed error information

**Reflect Node**:
- Critiques answer quality
- Verifies sources and checks hallucination risk
- Decides whether revision is needed
- Completes benchmark with final quality score
- Triggers replanning if quality is insufficient

---

## 📁 Project Structure

```
src/services/agent/
├── __init__.py
├── nodes.py                 # Enhanced with reflection & planning
├── planning.py              # Query planning & decomposition
├── reflection.py            # Answer critique & source verification
├── feedback.py              # User feedback logging & analytics
├── benchmarking.py          # Performance benchmarking
├── state.py                 # Agent state (existing)
├── graph.py                 # LangGraph workflow (existing)
├── factory.py               # Factory methods (existing)
├── metrics/                 # Tool metrics (existing)
└── tools/                   # Tool implementations (existing)

examples/
└── self_reflection_demo.py  # Complete demonstration
```

---

## 🚀 Quick Start

### 1. Run the Demo

```bash
cd /home/kumar-shiv/kumar/rag-journey-poc-to-prod
python examples/self_reflection_demo.py
```

### 2. Use in Your Agent

```python
from src.services.agent.factory import create_agent_system
from src.services.agent.nodes import AgentNodes

# Create agent with reflection enabled
registry, router, graph, nodes = create_agent_system(
    enable_reflection=True,
    enable_planning=True,
    enable_benchmarking=True
)

# Execute query
result = await graph.ainvoke({
    "query": "Your complex query here",
    "plan": [],
    "current_task": "",
    "tool_history": [],
    "results": [],
    "final_answer": "",
    "confidence": 0.0,
    "needs_replanning": False,
    "max_iterations": 5,
    "iteration_count": 0,
    "messages": [],
})

# Access quality metrics
print(f"Answer: {result['final_answer']}")
print(f"Quality Score: {result.get('quality_score', 0):.2f}")
print(f"Issues: {result.get('issues', [])}")
print(f"Hallucination Risk: {result.get('hallucination_risk', 0):.2f}")
```

### 3. Collect and Analyze Feedback

```python
from src.services.agent.feedback import FeedbackLogger, UserFeedback

logger = FeedbackLogger(storage_path="logs/user_feedback.json")

# After user interaction
feedback = UserFeedback(
    query=user_query,
    answer=agent_answer,
    rating=user_rating,  # 1-5
    feedback_text=user_comments,
    issues=identified_issues,
    session_id=session_id
)
logger.log_feedback(feedback)

# Periodically analyze
analytics = logger.get_analytics(days=7)
print(f"Recent performance: {analytics.average_rating:.2f}/5")
print(f"Areas to improve: {analytics.improvement_areas}")
```

---

## 📊 Monitoring & Improvement

### Key Metrics to Track

1. **Answer Quality** (from critique):
   - Overall quality score trend
   - Dimension breakdown (completeness, accuracy, etc.)
   - Percentage needing revision

2. **User Satisfaction** (from feedback):
   - Average rating trend
   - Positive feedback rate (4-5 stars)
   - Common issues frequency

3. **Performance** (from benchmarking):
   - Average execution time
   - Success rate trend
   - Tool effectiveness
   - Slow query patterns

4. **Source Quality**:
   - Source verification rate
   - Hallucination risk trend
   - Questionable claim frequency

### Continuous Improvement Cycle

```
┌─────────────────────────────────────────┐
│  1. Execute Query                       │
│     - Plan with QueryPlanner            │
│     - Execute tasks                     │
│     - Generate answer                   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  2. Self-Reflect                        │
│     - Critique answer quality           │
│     - Verify sources                    │
│     - Detect hallucinations             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  3. Decide & Act                        │
│     - If quality low: Revise            │
│     - If sources poor: Re-retrieve      │
│     - If hallucination risk: Verify     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  4. Collect Feedback                    │
│     - Log user rating                   │
│     - Track issues                      │
│     - Analyze patterns                  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  5. Analyze & Improve                   │
│     - Review benchmarks                 │
│     - Identify slow queries             │
│     - Optimize tool selection           │
│     - Adjust quality thresholds         │
└─────────────────────────────────────────┘
```

---

## 🎓 Best Practices

1. **Set Appropriate Thresholds**:
   - Quality threshold: 0.7 (adjust based on your domain)
   - Max iterations: 5 (prevent infinite loops)
   - Hallucination risk threshold: 0.6 (flag for review)

2. **Enable Features Selectively**:
   - Use `enable_reflection` for production (quality matters)
   - Use `enable_planning` for complex queries
   - Use `enable_benchmarking` during development/optimization

3. **Monitor Regularly**:
   - Review feedback analytics weekly
   - Check benchmark statistics daily
   - Analyze low-quality answers immediately

4. **Act on Insights**:
   - Adjust tool selection based on success rates
   - Improve prompts for low-rated query types
   - Add training data for common failure patterns

5. **Balance Quality vs Speed**:
   - Use quality thresholds to trigger revision
   - Set reasonable max iterations
   - Cache frequently used queries

---

## 🔧 Configuration

### Environment Variables

```bash
# Optional LLM for advanced planning
export OPENAI_API_KEY="your-key"

# Logging paths
export AGENT_FEEDBACK_PATH="logs/user_feedback.json"
export AGENT_BENCHMARK_PATH="logs/agent_benchmarks.jsonl"
```

### Quality Thresholds

Customize in `config/agent_config.json`:

```json
{
  "quality_threshold": 0.7,
  "hallucination_risk_threshold": 0.6,
  "max_iterations": 5,
  "enable_reflection": true,
  "enable_planning": true,
  "enable_benchmarking": false
}
```

---

## 📝 Next Steps

1. **Run the demo**: `python examples/self_reflection_demo.py`
2. **Integrate with your agent**: Enable reflection in factory
3. **Start collecting feedback**: Add feedback endpoints to API
4. **Monitor metrics**: Set up dashboards for key metrics
5. **Iterate and improve**: Use insights to optimize

---

## 🤝 Contributing

To add new reflection capabilities:

1. Add new metrics to `AnswerCritique` or `SourceVerification`
2. Implement assessment logic in `AnswerCritic` or `SourceVerifier`
3. Update `AgentNodes._critique_and_verify()` to use new metrics
4. Add examples to the demo

---

## 📚 References

- Planning: Based on ReACT and tree-of-thought approaches
- Reflection: Inspired by self-consistency and chain-of-verification
- Benchmarking: Standard ML evaluation practices
- Feedback: User-centered design principles

---

**Built with ❤️ for production-grade Agentic RAG systems**
