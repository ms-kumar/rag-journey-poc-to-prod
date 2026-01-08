# Health Check API

Comprehensive health monitoring endpoints for the RAG system.

## Overview

The Health Check API provides multiple endpoints for monitoring system status, dependency health, and component availability. Designed for both manual monitoring and automated orchestration systems (Kubernetes, Docker, etc.).

## Features

- ✅ **Multiple Health Endpoints**: Basic, detailed, readiness, and liveness checks
- ✅ **Component Health Checks**: Individual status for vectorstore, embeddings, generation
- ✅ **System Information**: OS, Python version, architecture details
- ✅ **Dependency Versions**: Installed package versions
- ✅ **Response Time Tracking**: Latency metrics for each component
- ✅ **Async Checks**: Parallel health checks for fast responses
- ✅ **Kubernetes Ready**: Readiness and liveness probes included

## Quick Start

### Basic Health Check

```bash
# Simple health check
curl http://localhost:8000/api/v1/health

# Response
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-01-08T12:00:00Z",
  "uptime_seconds": 3600.5
}
```

### Detailed Health Check

```bash
# Full health check with all information
curl "http://localhost:8000/api/v1/health/detailed?check_components=true"

# Response
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-01-08T12:00:00Z",
  "components": [
    {
      "name": "vectorstore",
      "status": "healthy",
      "message": "Collection 'documents' accessible",
      "details": {
        "collection": "documents",
        "points_count": 1000,
        "vector_size": 384
      },
      "response_time_ms": 15.2
    },
    {
      "name": "embeddings",
      "status": "healthy",
      "message": "Embeddings service responsive",
      "details": {
        "embedding_dim": 384,
        "test_query": "health check"
      },
      "response_time_ms": 45.8
    },
    {
      "name": "generation",
      "status": "healthy",
      "message": "Generation service responsive",
      "details": {
        "model": "gpt2",
        "test_output_length": 25
      },
      "response_time_ms": 120.3
    }
  ],
  "uptime_seconds": 3600.5,
  "system_info": {
    "platform": "Linux-5.15.0-generic",
    "python_version": "3.11.0",
    "architecture": "x86_64",
    "processor": "Intel(R) Xeon(R)"
  },
  "dependencies": {
    "fastapi": "0.128.0",
    "qdrant-client": "1.16.2",
    "langchain": "1.2.0",
    "transformers": "4.57.3",
    "torch": "2.9.1",
    "pydantic": "2.0.0"
  }
}
```

## Endpoints

### 1. Basic Health Check

**GET** `/api/v1/health`

Simple health check without dependency verification. Fast response suitable for load balancers.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-01-08T12:00:00Z",
  "uptime_seconds": 3600.5
}
```

**Use Cases:**
- Load balancer health checks
- Quick availability verification
- Monitoring dashboard status

### 2. Detailed Health Check

**GET** `/api/v1/health/detailed`

Comprehensive health check with optional component verification.

**Query Parameters:**
- `include_system_info` (bool, default: true) - Include OS/system information
- `include_dependencies` (bool, default: true) - Include dependency versions
- `check_components` (bool, default: false) - Check individual component health

**Response:** See detailed example above

**Use Cases:**
- Deep health verification
- Debugging connectivity issues
- System information gathering
- Component-level monitoring

**Performance:**
- Without `check_components`: ~5ms
- With `check_components`: ~200ms (runs actual health tests)

### 3. Readiness Check

**GET** `/api/v1/health/ready`

Kubernetes-style readiness probe. Verifies application is ready to serve traffic.

**Response (Ready):**
```json
{
  "ready": true,
  "status": "healthy",
  "timestamp": "2026-01-08T12:00:00Z"
}
```

**Response (Not Ready):**
```json
{
  "ready": false,
  "status": "unhealthy",
  "timestamp": "2026-01-08T12:00:00Z",
  "components": [
    {
      "name": "vectorstore",
      "status": "unhealthy",
      "message": "Connection refused"
    }
  ]
}
```

**Use Cases:**
- Kubernetes readiness probes
- Traffic routing decisions
- Deployment verification

### 4. Liveness Check

**GET** `/api/v1/health/live`

Kubernetes-style liveness probe. Simple check that process is alive.

**Response:**
```json
{
  "alive": true,
  "timestamp": "2026-01-08T12:00:00Z",
  "uptime_seconds": 3600.5
}
```

**Use Cases:**
- Kubernetes liveness probes
- Process monitoring
- Restart detection

## Status Values

### Service Status Enum

```python
class ServiceStatus(str, Enum):
    HEALTHY = "healthy"      # All systems operational
    DEGRADED = "degraded"    # Partial functionality
    UNHEALTHY = "unhealthy"  # Service unavailable
    UNKNOWN = "unknown"      # Status cannot be determined
```

### Status Determination Rules

**Overall Status Logic:**
- `UNHEALTHY`: Any component is unhealthy
- `DEGRADED`: Any component is degraded or unknown
- `HEALTHY`: All components are healthy
- `UNKNOWN`: No components checked

## Component Health Checks

### Vectorstore Health

Checks Qdrant connection and collection accessibility.

**Checks:**
- Connection to Qdrant server
- Collection exists and is accessible
- Retrieves collection metadata

**Success Criteria:**
- Connection established
- Collection info retrieved
- Points count available

**Failure Scenarios:**
- Connection refused
- Collection not found
- Timeout

### Embeddings Health

Checks embedding service functionality.

**Checks:**
- Embeddings client initialized
- Can generate test embedding
- Returns expected dimension

**Success Criteria:**
- Test query embedded successfully
- Embedding dimension correct

**Failure Scenarios:**
- Model not loaded
- CUDA out of memory
- Service timeout

### Generation Health

Checks text generation service.

**Checks:**
- Generator client initialized
- Can generate text from test prompt
- Returns valid output

**Success Criteria:**
- Generation completes
- Output is non-empty

**Failure Scenarios:**
- Model not loaded
- GPU unavailable
- Service timeout

## Response Time Tracking

Each component health check includes response time:

```json
{
  "name": "vectorstore",
  "status": "healthy",
  "response_time_ms": 15.2
}
```

**Performance Thresholds:**

| Component | Good | Warning | Critical |
|-----------|------|---------|----------|
| Vectorstore | < 50ms | 50-200ms | > 200ms |
| Embeddings | < 100ms | 100-500ms | > 500ms |
| Generation | < 200ms | 200-1000ms | > 1000ms |

## Kubernetes Integration

### Deployment Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: rag-api
        image: rag-api:latest
        ports:
        - containerPort: 8000
        
        # Liveness probe - restart if unhealthy
        livenessProbe:
          httpGet:
            path: /api/v1/health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        # Readiness probe - remove from load balancer if not ready
        readinessProbe:
          httpGet:
            path: /api/v1/health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
```

### Probe Configuration Guidelines

**Liveness Probe:**
- Use `/api/v1/health/live` (fast, doesn't check dependencies)
- `initialDelaySeconds`: 30s (allow startup time)
- `periodSeconds`: 10s (check every 10 seconds)
- `failureThreshold`: 3 (restart after 3 failures)

**Readiness Probe:**
- Use `/api/v1/health/ready` (checks dependencies)
- `initialDelaySeconds`: 10s (wait for initialization)
- `periodSeconds`: 5s (check frequently)
- `failureThreshold`: 2 (remove from traffic quickly)

## Docker Health Check

### Dockerfile Configuration

```dockerfile
FROM python:3.11-slim

# ... application setup ...

# Health check configuration
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health/live || exit 1
```

### Docker Compose

```yaml
version: '3.8'

services:
  rag-api:
    image: rag-api:latest
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health/live"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 60s
```

## Monitoring Integration

### Prometheus Metrics

Example metrics to expose:

```python
from prometheus_client import Counter, Histogram

# Health check metrics
health_check_requests = Counter(
    'health_check_requests_total',
    'Total health check requests',
    ['endpoint', 'status']
)

component_health_duration = Histogram(
    'component_health_check_duration_seconds',
    'Time spent checking component health',
    ['component']
)
```

### Grafana Dashboard

Example queries:

```promql
# Overall health status
health_check_requests_total{endpoint="/api/v1/health/detailed", status="healthy"}

# Component response times
histogram_quantile(0.95, component_health_check_duration_seconds{component="vectorstore"})

# Unhealthy components
health_check_requests_total{status="unhealthy"}
```

## Testing

### Manual Testing

```bash
# Basic health
curl http://localhost:8000/api/v1/health

# Detailed with all checks
curl "http://localhost:8000/api/v1/health/detailed?check_components=true&include_system_info=true&include_dependencies=true"

# Readiness
curl http://localhost:8000/api/v1/health/ready

# Liveness
curl http://localhost:8000/api/v1/health/live
```

### Automated Testing

```bash
# Run health check tests
pytest tests/test_health_check.py -v

# Test coverage
pytest tests/test_health_check.py --cov=src.services.health_check --cov-report=term-missing
```

### Load Testing

```bash
# Test health endpoint under load
ab -n 1000 -c 10 http://localhost:8000/api/v1/health

# Test detailed health endpoint
ab -n 100 -c 5 "http://localhost:8000/api/v1/health/detailed?check_components=true"
```

## Best Practices

### ✅ DO

1. **Use appropriate endpoints**:
   - Load balancers: `/health/live`
   - Readiness checks: `/health/ready`
   - Debugging: `/health/detailed?check_components=true`

2. **Configure timeouts properly**:
   - Basic health: 1-2 seconds
   - Readiness: 3-5 seconds
   - Detailed: 5-10 seconds

3. **Monitor health metrics**:
   - Track failure rates
   - Alert on unhealthy components
   - Monitor response times

4. **Cache detailed checks**:
   - Full component checks are expensive
   - Cache results for 30-60 seconds
   - Use basic checks for frequent polling

### ❌ DON'T

1. **Don't use detailed checks for liveness**:
   - Component checks are slow
   - Can cause false positives
   - Use simple `/health/live` instead

2. **Don't check too frequently**:
   - Adds unnecessary load
   - 10-30 second intervals sufficient
   - Use adaptive intervals

3. **Don't ignore component status**:
   - DEGRADED means issues exist
   - Investigate before full failure
   - Monitor trends

4. **Don't restart immediately**:
   - Allow time for transient failures
   - Use appropriate `failureThreshold`
   - Configure retry logic

## API Reference

### Models

#### ComponentHealth

Component health status information.

```python
class ComponentHealth(BaseModel):
    name: str                           # Component name
    status: ServiceStatus               # Current status
    message: str | None = None          # Status message
    details: dict[str, Any] | None = None  # Additional details
    response_time_ms: float | None = None  # Response time
```

#### HealthCheckResponse

Basic health check response.

```python
class HealthCheckResponse(BaseModel):
    status: ServiceStatus               # Overall status
    version: str                        # API version
    timestamp: str                      # ISO timestamp
    components: list[ComponentHealth] = []  # Component statuses
    uptime_seconds: float | None = None # System uptime
```

#### DetailedHealthResponse

Extended health response with system info.

```python
class DetailedHealthResponse(HealthCheckResponse):
    system_info: dict[str, Any] | None = None  # System information
    dependencies: dict[str, str] | None = None # Dependency versions
```

### Functions

#### `check_vectorstore_health(client)`

Check vectorstore connection health.

**Returns:** ComponentHealth

#### `check_embeddings_health(client)`

Check embeddings service health.

**Returns:** ComponentHealth

#### `check_generation_health(client)`

Check generation service health.

**Returns:** ComponentHealth

#### `check_all_components(vectorstore, embeddings, generator)`

Check all components in parallel.

**Returns:** list[ComponentHealth]

#### `determine_overall_status(components)`

Determine overall status from component statuses.

**Returns:** ServiceStatus

## Troubleshooting

### Issue: Health Checks Timing Out

**Symptom:** Health endpoints return 504 or timeout

**Solution:**
1. Check component connectivity
2. Increase timeout values
3. Use basic health check instead of detailed
4. Verify service dependencies are running

### Issue: False Positive Failures

**Symptom:** Health checks fail intermittently

**Solution:**
1. Increase `failureThreshold` in probes
2. Add retry logic to health checks
3. Use jitter in check intervals
4. Review transient error patterns

### Issue: Slow Health Checks

**Symptom:** Detailed health checks take too long

**Solution:**
1. Disable `check_components` for frequent checks
2. Cache component health results
3. Check components in parallel (already implemented)
4. Optimize component check logic

## Examples

### Example 1: Custom Health Dashboard

```python
import httpx
import asyncio

async def get_system_health():
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:8000/api/v1/health/detailed",
            params={"check_components": True}
        )
        return response.json()

# Display health dashboard
health = await get_system_health()
print(f"Status: {health['status']}")
print(f"Uptime: {health['uptime_seconds']}s")

for component in health['components']:
    print(f"{component['name']}: {component['status']} ({component['response_time_ms']}ms)")
```

### Example 2: Automated Alerting

```python
import asyncio
from datetime import datetime

async def monitor_health():
    while True:
        health = await get_system_health()
        
        if health['status'] == 'unhealthy':
            # Send alert
            print(f"ALERT: System unhealthy at {datetime.now()}")
            for comp in health['components']:
                if comp['status'] == 'unhealthy':
                    print(f"  - {comp['name']}: {comp['message']}")
        
        await asyncio.sleep(30)  # Check every 30 seconds
```

### Example 3: Component-Specific Checks

```python
async def check_specific_component(component_name: str):
    health = await get_system_health()
    
    for comp in health['components']:
        if comp['name'] == component_name:
            return comp
    
    return None

# Check vectorstore status
vectorstore_health = await check_specific_component('vectorstore')
if vectorstore_health and vectorstore_health['status'] == 'healthy':
    print(f"Vectorstore OK: {vectorstore_health['details']['points_count']} points")
```

## See Also

- [Retry and Backoff Documentation](./retry-backoff.md)
- [VectorStore Client](./vectorstore-client.md)
- [Generation Client](./generation-client.md)
- [Deployment Guide](./deployment.md)
