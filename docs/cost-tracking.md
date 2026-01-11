# Cost Tracking & Model Selection

Comprehensive system for tracking costs, selecting models based on quality/latency/cost tradeoffs, and managing autoscaling policies.

## Features

### 1. Cost Tracking
- Per-model cost tracking
- Cost per 1k queries calculation
- Latency and quality metrics
- Token usage tracking
- Error rate monitoring
- Efficiency rankings

### 2. Tiered Model Selection
- **Budget Tier**: Low cost, acceptable quality
- **Balanced Tier**: Good quality/cost tradeoff
- **Premium Tier**: High quality, higher cost
- **Ultra Tier**: Highest quality, highest cost

### 3. Selection Strategies
- **MINIMIZE_COST**: Choose cheapest model meeting requirements
- **MINIMIZE_LATENCY**: Choose fastest model
- **MAXIMIZE_QUALITY**: Choose highest quality model
- **BALANCED**: Optimize quality/cost ratio
- **ADAPTIVE**: Adapt based on load and budget

### 4. Autoscaling & Concurrency
- CPU-based scaling
- Queue-based scaling
- Latency-aware scaling
- Budget-constrained scaling
- Cooldown periods
- Quality SLA enforcement

## Usage

### Cost Tracking

```python
from src.services.cost import CostTracker

# Initialize tracker
tracker = CostTracker()

# Record requests
tracker.record_request(
    model_name="gpt-3.5-turbo",
    cost=0.002,  # Cost in dollars
    latency=0.5,  # Latency in seconds
    tokens=500,
    quality_score=0.9,
    error=False,
)

# Get report
report = tracker.get_report()
print(f"Total cost: ${report.total_cost:.2f}")
print(f"Cost per 1k queries: ${report.cost_per_1k:.2f}")
print(f"Total requests: {report.total_requests}")

# Show summary
tracker.print_summary()

# Get efficiency rankings
for name, efficiency in tracker.get_models_by_efficiency():
    print(f"{name}: {efficiency:.2f}")
```

### Model Selection

```python
from src.services.cost import ModelSelector, SelectionStrategy, ModelTier

# Initialize selector
selector = ModelSelector()

# View comparison matrix
selector.print_comparison_matrix()

# Select model - minimize cost
model = selector.select_model(
    strategy=SelectionStrategy.MINIMIZE_COST,
    min_quality=0.7,
)
print(f"Selected: {model.name} (${model.cost_per_1k:.2f}/1k)")

# Select model - balanced
model = selector.select_model(
    strategy=SelectionStrategy.BALANCED,
    min_quality=0.75,
    max_latency_ms=500,
)

# Select from specific tier
model = selector.select_model(
    strategy=SelectionStrategy.BALANCED,
    tier=ModelTier.PREMIUM,
)

# Budget-aware selection
selector.set_budget(100.0)
model = selector.select_model(
    strategy=SelectionStrategy.ADAPTIVE,
    min_quality=0.7,
)
selector.record_cost(model.cost_per_1k / 1000)
print(f"Remaining budget: ${selector.remaining_budget:.2f}")
```

### Autoscaling

```python
from src.services.cost import Autoscaler, AutoscalingPolicy, LoadMetrics

# Create policy
policy = AutoscalingPolicy(
    min_instances=1,
    max_instances=10,
    scale_up_cpu_threshold=70.0,
    scale_down_cpu_threshold=30.0,
    cooldown_period=60,
    max_cost_per_hour=50.0,
)

# Initialize autoscaler
autoscaler = Autoscaler(policy)

# Check if scaling needed
metrics = LoadMetrics(
    cpu_usage=85.0,
    queue_size=60,
    active_requests=30,
    avg_latency_ms=500,
    p95_latency_ms=800,
    error_rate=2.0,
    requests_per_second=100,
)

decision, new_instances = autoscaler.auto_scale(metrics)
print(f"Decision: {decision.value}, Instances: {new_instances}")

# Get current capacity
capacity = autoscaler.get_current_capacity()
print(f"Total capacity: {capacity['total_capacity']}")

# Record costs
autoscaler.record_cost(10.0)

# View status
autoscaler.print_status()
```

### Integrated System

```python
from src.services.cost import (
    CostTracker,
    ModelSelector,
    Autoscaler,
    SelectionStrategy,
)

# Initialize components
tracker = CostTracker()
selector = ModelSelector()
autoscaler = Autoscaler()

# Set budget
selector.set_budget(100.0)

# Process requests
for i in range(100):
    # Select model based on current conditions
    if autoscaler.current_instances > 5:
        strategy = SelectionStrategy.MINIMIZE_COST
    else:
        strategy = SelectionStrategy.BALANCED
    
    model = selector.select_model(
        strategy=strategy,
        min_quality=0.7,
    )
    
    # Simulate request
    cost = model.cost_per_1k / 1000
    latency = model.avg_latency_ms / 1000
    
    # Track cost
    tracker.record_request(
        model_name=model.name,
        cost=cost,
        latency=latency,
        quality_score=model.quality_score,
    )
    
    # Update budget
    selector.record_cost(cost)
    
    # Check autoscaling
    if i % 10 == 0:
        metrics = LoadMetrics(...)
        autoscaler.auto_scale(metrics)

# Show results
tracker.print_summary()
selector.print_comparison_matrix()
autoscaler.print_status()
```

## Quality/Latency/Cost Matrix

| Model | Tier | Quality | Latency (ms) | Cost/1k | Efficiency |
|-------|------|---------|--------------|---------|------------|
| distilgpt2 | Budget | 0.60 | 30 | $0.05 | 12.00 |
| gpt2 | Budget | 0.65 | 50 | $0.10 | 6.50 |
| flan-t5-base | Balanced | 0.75 | 100 | $0.50 | 1.50 |
| llama-2-7b | Balanced | 0.80 | 150 | $0.80 | 1.00 |
| gpt-3.5-turbo | Premium | 0.90 | 500 | $2.00 | 0.45 |
| claude-2 | Premium | 0.92 | 600 | $2.50 | 0.37 |
| gpt-4 | Ultra | 0.98 | 2000 | $30.00 | 0.03 |

**Efficiency** = Quality Score / Cost per 1k

## Autoscaling Policies

### CPU-Based Scaling
- Scale up when CPU > 70%
- Scale down when CPU < 30%

### Queue-Based Scaling
- Scale up when queue > 50 items
- Scale down when queue < 5 items

### Latency-Based Scaling
- Scale up when P95 latency > threshold
- Ensures quality SLAs are met

### Budget-Aware Scaling
- Prevents scaling when budget exhausted
- Monitors cost per hour limits

## Configuration

### Custom Models

```python
from src.services.cost import ModelCandidate, ModelTier, ModelSelector

custom_models = [
    ModelCandidate(
        name="my-model",
        tier=ModelTier.BALANCED,
        cost_per_1k=1.0,
        avg_latency_ms=200,
        quality_score=0.85,
        max_concurrency=15,
    ),
]

selector = ModelSelector(models=custom_models)
```

### Custom Autoscaling Policy

```python
from src.services.cost import AutoscalingPolicy

policy = AutoscalingPolicy(
    min_instances=2,
    max_instances=20,
    scale_up_cpu_threshold=80.0,
    scale_down_cpu_threshold=25.0,
    scale_up_queue_threshold=100,
    cooldown_period=30,
    max_cost_per_hour=100.0,
    budget_limit=1000.0,
    min_quality_score=0.8,
    target_p95_latency_ms=500.0,
)
```

## Demo

Run the comprehensive demo:

```bash
python examples/cost_tracking_demo.py
```

The demo includes:
1. Cost tracking with multiple models
2. Model selection scenarios
3. Autoscaling simulation
4. Integrated system demonstration

## API Reference

See individual module documentation:
- [tracker.py](../src/services/cost/tracker.py) - Cost tracking
- [model_selector.py](../src/services/cost/model_selector.py) - Model selection
- [autoscaler.py](../src/services/cost/autoscaler.py) - Autoscaling

## Testing

```bash
# Run cost tracking tests
pytest tests/test_cost_tracker.py -v

# Run model selection tests
pytest tests/test_model_selector.py -v

# Run autoscaling tests
pytest tests/test_autoscaler.py -v

# Run all cost service tests
pytest tests/test_cost*.py -v
```

## Metrics

### Cost Metrics
- Total cost
- Cost per 1k queries
- Cost per token
- Cost per hour

### Performance Metrics
- Average latency
- P95 latency
- Requests per second
- Error rate

### Quality Metrics
- Average quality score
- Quality/cost ratio (efficiency)
- Quality/latency ratio

### Capacity Metrics
- Active instances
- Concurrency level
- Total capacity
- Utilization rate

## Best Practices

1. **Set Budgets**: Always set budgets when using adaptive selection
2. **Monitor Efficiency**: Track quality/cost ratios to optimize model selection
3. **Use Cooldown Periods**: Prevent thrashing with appropriate cooldown periods
4. **Track All Metrics**: Record quality scores, tokens, and errors for accurate analysis
5. **Test Autoscaling**: Simulate load patterns before production deployment
6. **Review Reports**: Regularly review cost reports to identify optimization opportunities
