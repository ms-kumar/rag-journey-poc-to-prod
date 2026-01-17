# A/B Experimentation & Feature Flags

This document describes the experimentation framework for running A/B tests and managing feature flags in the RAG pipeline.

## Table of Contents

1. [Overview](#overview)
2. [Experiments](#experiments)
3. [Feature Flags](#feature-flags)
4. [Statistical Analysis](#statistical-analysis)
5. [Canary Deployments](#canary-deployments)
6. [Reports](#reports)
7. [Best Practices](#best-practices)

---

## Overview

The experimentation module provides:

- **A/B Experiments**: Test different prompts, models, or configurations
- **Feature Flags**: Gradual rollout with targeting rules
- **Statistical Analysis**: T-tests, chi-square, confidence intervals
- **Canary Support**: Progressive traffic routing with health checks
- **Automated Reports**: Generate insights and recommendations

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Experimentation Layer                     │
├─────────────┬─────────────┬──────────────┬─────────────────┤
│  Experiments │ Feature     │ Analysis     │ Canary          │
│  Manager     │ Flags       │ Engine       │ Manager         │
├─────────────┴─────────────┴──────────────┴─────────────────┤
│                    Storage Layer                             │
│              (In-memory / Redis / Database)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Experiments

### Creating an Experiment

```python
from src.services.experimentation import Experiment, ExperimentManager, Variant

manager = ExperimentManager()

# Define variants
variants = [
    Variant(id="control", name="GPT-3.5", weight=50),
    Variant(id="treatment", name="GPT-4", weight=50),
]

# Create experiment
experiment = Experiment(
    id="llm-comparison-001",
    name="LLM Model Comparison",
    description="Compare GPT-3.5 vs GPT-4 for answer quality",
    variants=variants,
    metrics=["latency_ms", "quality_score", "user_satisfaction"],
)

manager.register_experiment(experiment)
```

### Assigning Users to Variants

```python
# Deterministic assignment based on user_id
variant = manager.get_variant("llm-comparison-001", user_id="user-123")
print(f"User assigned to: {variant.name}")

# Use the variant
if variant.id == "control":
    response = call_gpt35(query)
else:
    response = call_gpt4(query)
```

### Recording Results

```python
# Record exposure (when user sees the experiment)
manager.record_exposure("llm-comparison-001", user_id="user-123")

# Record outcome metrics
manager.record_outcome(
    experiment_id="llm-comparison-001",
    user_id="user-123",
    metrics={
        "latency_ms": 245.0,
        "quality_score": 0.87,
        "user_satisfaction": 4,
    }
)
```

### Experiment Lifecycle

```
DRAFT → RUNNING → COMPLETED
           ↓
        PAUSED → RUNNING (resume)
           ↓
       CANCELLED
```

---

## Feature Flags

### Creating a Feature Flag

```python
from src.services.experimentation import FeatureFlag, FlagManager, RolloutConfig

manager = FlagManager()

# Simple boolean flag
flag = FeatureFlag(
    id="new-reranker",
    name="New Reranker Model",
    enabled=True,
    rollout_percentage=25,  # 25% of users
)

manager.register_flag(flag)
```

### Using Feature Flags

```python
# Check if flag is enabled for a user
if manager.is_enabled("new-reranker", user_id="user-123"):
    use_new_reranker()
else:
    use_old_reranker()

# Get flag value (for non-boolean flags)
chunk_size = manager.get_value("chunk-size-experiment", default=512)
```

### Targeting Rules

```python
flag = FeatureFlag(
    id="beta-features",
    enabled=True,
    rollout_percentage=100,  # 100% of targeted users
    targeting_rules={
        "user_groups": ["beta_testers", "internal"],
        "user_ids": ["user-admin-001"],
        "attributes": {
            "plan": ["enterprise", "pro"],
            "region": ["us-west", "us-east"],
        }
    }
)
```

### Kill Switches

```python
# Emergency disable
manager.disable_flag("problematic-feature")

# Re-enable
manager.enable_flag("problematic-feature")
```

---

## Statistical Analysis

### Running Analysis

```python
from src.services.experimentation import analyze_experiment, StatisticalTest

# Analyze continuous metric (e.g., latency)
result = analyze_experiment(
    experiment_id="llm-comparison-001",
    metric="latency_ms",
    test_type=StatisticalTest.T_TEST,
    confidence_level=0.95,
)

print(f"P-value: {result.p_value:.4f}")
print(f"Significant: {result.is_significant}")
print(f"Effect size: {result.effect_size:.3f}")
```

### Available Tests

| Test | Use Case |
|------|----------|
| `T_TEST` | Continuous metrics (latency, scores) |
| `CHI_SQUARE` | Categorical outcomes (conversion) |
| `MANN_WHITNEY` | Non-normal distributions |

### Sample Size Calculator

```python
from src.services.experimentation import calculate_sample_size

required_n = calculate_sample_size(
    baseline_rate=0.10,  # 10% baseline conversion
    minimum_detectable_effect=0.02,  # Detect 2% absolute change
    power=0.80,
    significance_level=0.05,
)

print(f"Required sample size per variant: {required_n}")
```

---

## Canary Deployments

### Creating a Canary

```python
from src.services.experimentation import CanaryManager, CanaryConfig

manager = CanaryManager()

config = CanaryConfig(
    deployment_id="v2.1.0-canary",
    traffic_percentage=5,
    health_thresholds={
        "error_rate": 0.05,
        "latency_p99_ms": 500,
        "min_requests": 100,
    },
    auto_promote=True,
    auto_rollback=True,
)

canary = manager.create_canary(config)
```

### Monitoring Canary Health

```python
# Check canary status
status = manager.get_status("v2.1.0-canary")

print(f"Traffic: {status.traffic_percentage}%")
print(f"Error rate: {status.error_rate:.2%}")
print(f"Healthy: {status.is_healthy}")

# Manual promotion
if status.is_healthy and status.min_duration_met:
    manager.promote_canary("v2.1.0-canary", target_percentage=25)

# Manual rollback
if status.error_rate > 0.10:
    manager.rollback_canary("v2.1.0-canary")
```

### Traffic Ramping

```
5% → (5 min) → 25% → (10 min) → 50% → (10 min) → 100%
  ↓              ↓               ↓
 Health       Health          Health
 Check        Check           Check
```

---

## Reports

### Generating Reports

```python
from src.services.experimentation import ReportGenerator, ReportFormat

generator = ReportGenerator()

# Generate markdown report
report = generator.generate(
    experiment_id="llm-comparison-001",
    format=ReportFormat.MARKDOWN,
)

# Save to file
with open("reports/llm-comparison.md", "w") as f:
    f.write(report)

# Generate JSON for APIs
json_report = generator.generate(
    experiment_id="llm-comparison-001",
    format=ReportFormat.JSON,
)
```

### Report Contents

- **Summary**: Winner, confidence, key findings
- **Metrics**: Per-metric analysis with p-values
- **Sample Sizes**: Control vs treatment counts
- **Duration**: Experiment run time
- **Recommendations**: Next steps based on results

---

## Best Practices

### Experiment Design

1. **Clear Hypothesis**: Define what you're testing and expected outcomes
2. **Single Variable**: Change one thing at a time
3. **Sufficient Power**: Calculate sample size before starting
4. **Pre-registration**: Document experiment plan before running

### Statistical Rigor

1. **Confidence Level**: Use 95% (α=0.05) as default
2. **Multiple Testing**: Apply Bonferroni correction for multiple metrics
3. **Peeking**: Avoid early stopping based on intermediate results
4. **Effect Size**: Report practical significance, not just p-values

### Feature Flag Hygiene

1. **Short-lived**: Remove flags after experiments conclude
2. **Named Clearly**: Use descriptive, consistent naming
3. **Documented**: Track flag purpose and owner
4. **Monitored**: Alert on flag state changes

### Canary Safety

1. **Start Small**: Begin with 1-5% traffic
2. **Monitor Closely**: Watch error rates and latency
3. **Auto-rollback**: Configure automatic rollback thresholds
4. **Bake Time**: Allow sufficient time at each traffic level

---

## API Reference

### ExperimentManager

| Method | Description |
|--------|-------------|
| `register_experiment(exp)` | Register a new experiment |
| `get_variant(exp_id, user_id)` | Get variant for user |
| `record_exposure(exp_id, user_id)` | Log exposure event |
| `record_outcome(exp_id, user_id, metrics)` | Log outcome metrics |
| `pause_experiment(exp_id)` | Pause experiment |
| `complete_experiment(exp_id)` | Mark as complete |

### FlagManager

| Method | Description |
|--------|-------------|
| `register_flag(flag)` | Register a new flag |
| `is_enabled(flag_id, user_id)` | Check if enabled for user |
| `get_value(flag_id, default)` | Get flag value |
| `enable_flag(flag_id)` | Enable flag |
| `disable_flag(flag_id)` | Disable flag (kill switch) |

### CanaryManager

| Method | Description |
|--------|-------------|
| `create_canary(config)` | Create canary deployment |
| `get_status(deployment_id)` | Get canary health status |
| `promote_canary(id, percentage)` | Increase traffic |
| `rollback_canary(id)` | Rollback to stable |

---

## Related Documentation

- [CI/CD Pipeline](./ci-cd-pipeline.md) - Deployment integration
- [Observability](./observability.md) - Metrics and tracing
- [Evaluation Harness](./evaluation-harness.md) - Quality metrics
