# CI/CD Pipeline Documentation

This document describes the CI/CD pipeline for the RAG API project.

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Stages](#pipeline-stages)
3. [Workflows](#workflows)
4. [Deployment Strategy](#deployment-strategy)
5. [Environment Configuration](#environment-configuration)
6. [Rollback Procedures](#rollback-procedures)
7. [Monitoring & Alerting](#monitoring--alerting)

---

## Overview

The CI/CD pipeline implements a progressive delivery strategy:

```
┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐    ┌────────────┐
│  Build  │───►│  Test    │───►│ Eval    │───►│ Staging  │───►│  Canary    │
│         │    │          │    │ Gate    │    │          │    │  (5-25%)   │
└─────────┘    └──────────┘    └─────────┘    └─────────┘    └──────────┘
                                                                    │
                                                                    ▼
                                                              ┌────────────┐
                                                              │ Production │
                                                              │  (100%)    │
                                                              └────────────┘
```

### Key Principles

1. **Automated Quality Gates**: Every stage has quality gates that must pass
2. **Progressive Delivery**: Traffic gradually shifts to new versions
3. **Automatic Rollback**: Failed canaries trigger automatic rollback
4. **Manual Production Gate**: Production deployment requires approval

---

## Pipeline Stages

### 1. Build Stage

**Trigger**: Push to `main` or `develop` branch, or manual dispatch

**Steps**:
1. Checkout code
2. Set up Python 3.11 with uv
3. Build Docker image
4. Push to GitHub Container Registry (ghcr.io)

**Outputs**:
- Docker image tagged with commit SHA
- Docker image tagged as `latest` (for main branch)

### 2. Test Stage

**Steps**:
1. Run unit tests
2. Run integration tests (with Qdrant and Redis)
3. Generate coverage report
4. Run quality checks (ruff, mypy, bandit)

**Quality Gates**:
- All tests must pass
- Coverage must be > 80%
- No critical linting errors
- No security vulnerabilities (high/critical)

### 3. Evaluation Gate

**Steps**:
1. Create evaluation datasets
2. Run RAG quality evaluation
3. Check metrics against thresholds

**Quality Thresholds** (from `config/eval_thresholds.json`):
| Metric | Minimum |
|--------|---------|
| NDCG@5 | 0.70 |
| Recall@5 | 0.80 |
| MRR | 0.65 |
| Context Relevance | 0.70 |
| Answer Correctness | 0.70 |

### 4. Staging Deployment

**Trigger**: Successful build + test + eval gate

**Steps**:
1. Deploy to staging namespace
2. Run smoke tests
3. Wait for pod health

**Validation**:
- Health endpoint returns 200
- Basic RAG query succeeds
- No error logs in first 2 minutes

### 5. Canary Deployment

**Trigger**: Staging validation passed (main branch only)

**Progressive Traffic**:
1. **Phase 1**: 5% traffic for 5 minutes
2. **Phase 2**: 25% traffic for 5 minutes (if Phase 1 healthy)

**Health Checks**:
- Error rate < 5%
- P99 latency < 500ms
- Minimum 100 requests sampled

### 6. Production Deployment

**Trigger**: Canary healthy + manual approval

**Steps**:
1. Wait for approval (production environment)
2. Scale down canary
3. Deploy to all production pods
4. Monitor for 10 minutes

---

## Workflows

### ci.yml - Continuous Integration

```yaml
Trigger: Push to main/develop, Pull requests
Jobs:
  - test: Run tests with coverage
  - lint: Run ruff, mypy
  - security: Run bandit
```

### eval_gate.yml - Evaluation Gate

```yaml
Trigger: Pull requests
Jobs:
  - eval: Run RAG evaluation
  - comment: Post results to PR
```

### deploy.yml - Deployment Pipeline

```yaml
Trigger: Push to main, manual dispatch
Jobs:
  - build: Build and push Docker image
  - eval-gate: Run evaluation quality gate
  - deploy-staging: Deploy to staging
  - deploy-canary: Progressive canary deployment
  - deploy-production: Full production rollout
```

### rollback.yml - Manual Rollback

```yaml
Trigger: Manual dispatch only
Inputs:
  - environment: staging | production
  - target_version: Version to rollback to
  - skip_confirmation: Skip approval gate
Jobs:
  - validate: Verify versions
  - confirm: Wait for approval (optional)
  - rollback: Execute rollback
  - verify: Run health checks
  - notify: Send notifications
```

---

## Deployment Strategy

### Blue-Green Deployments

For staging, we use blue-green deployments:
- Deploy new version alongside existing
- Switch traffic atomically
- Keep old version for quick rollback

### Canary Deployments

For production, we use canary deployments:

```
┌─────────────────────────────────────────────┐
│                  Ingress                    │
└─────────────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
    ┌────▼────┐              ┌─────▼────┐
    │ Stable  │              │  Canary  │
    │  (95%)  │              │  (5%)    │
    │ v1.2.3  │              │  v1.2.4  │
    └─────────┘              └──────────┘
```

### Traffic Splitting

Traffic splitting is managed via:
- **Kubernetes**: Native traffic splitting with weights
- **Istio** (optional): VirtualService configuration
- **NGINX Ingress**: Canary annotations

---

## Environment Configuration

### Staging Environment

```yaml
Namespace: staging
Replicas: 2
Resources:
  CPU: 500m - 1000m
  Memory: 512Mi - 1Gi
```

### Production Environment

```yaml
Namespace: production
Replicas: 5 (minimum)
Resources:
  CPU: 1000m - 2000m
  Memory: 1Gi - 2Gi
```

### Environment Variables

| Variable | Staging | Production |
|----------|---------|------------|
| `LOG_LEVEL` | DEBUG | INFO |
| `CACHE_TTL` | 300 | 3600 |
| `MAX_TOKENS` | 1000 | 2000 |

### Secrets

Managed via GitHub Secrets and Kubernetes Secrets:

- `OPENAI_API_KEY`
- `QDRANT_API_KEY`
- `REDIS_PASSWORD`
- `KUBE_CONFIG`

---

## Rollback Procedures

### Automatic Rollback

Automatic rollback is triggered when:

1. **Canary health check fails**
   - Error rate > 5%
   - P99 latency > 500ms
   - Health endpoint fails

2. **Deployment timeout**
   - Pods don't become ready within 10 minutes

### Manual Rollback

For manual rollback, use:

```bash
# Via GitHub Actions
gh workflow run rollback.yml \
  -f environment=production \
  -f target_version=v1.2.3

# Via Makefile
make rollback ENV=production

# Via kubectl
kubectl -n production rollout undo deployment/rag-api
```

### Rollback Verification

After rollback:
1. Verify pod status
2. Check health endpoint
3. Monitor error rate
4. Verify version endpoint

See [Rollback Playbook](./rollback-playbook.md) for detailed procedures.

---

## Monitoring & Alerting

### Key Metrics

| Metric | Warning | Critical |
|--------|---------|----------|
| Error Rate | > 1% | > 5% |
| P99 Latency | > 300ms | > 500ms |
| Pod Restarts | > 2/hour | > 5/hour |
| Memory Usage | > 80% | > 90% |

### Alerts

Alerts are sent via:
- **Slack**: `#rag-api-alerts` channel
- **PagerDuty**: For critical issues

### Dashboards

- **Grafana**: Application metrics
- **GitHub Actions**: Pipeline status
- **Kubernetes Dashboard**: Cluster health

---

## Quick Reference

### Common Commands

```bash
# Build and push image
make docker-push

# Deploy to staging
make deploy-staging

# Deploy canary
make deploy-canary

# Check canary health
make canary-health

# Deploy to production
make deploy-prod

# Rollback
make rollback ENV=production

# Check deployment status
make deploy-status

# View deployment history
make deploy-history ENV=production

# Run rollback rehearsal
make rehearse-rollback
```

### Workflow Triggers

| Workflow | Trigger | Manual |
|----------|---------|--------|
| ci.yml | push, PR | No |
| eval_gate.yml | PR | No |
| deploy.yml | push (main) | Yes |
| rollback.yml | - | Yes |

---

## Related Documentation

- [Rollback Playbook](./rollback-playbook.md)
- [Evaluation Harness](./evaluation-harness.md)
- [Health Check](./health-check.md)
- [Performance Profiling](./performance-profiling.md)
