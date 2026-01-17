# Rollback Playbook

This document provides step-by-step procedures for rolling back RAG API deployments.

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [When to Rollback](#when-to-rollback)
3. [Rollback Procedures](#rollback-procedures)
4. [Post-Rollback Verification](#post-rollback-verification)
5. [Communication Templates](#communication-templates)
6. [Troubleshooting](#troubleshooting)

---

## Quick Reference

### Emergency Rollback Commands

```bash
# Immediate rollback to last known good version
make rollback-prod VERSION=v1.2.3

# Or via kubectl
kubectl rollout undo deployment/rag-api -n production

# Or trigger GitHub workflow
gh workflow run rollback.yml -f environment=production -f target_version=v1.2.3
```

### Key Contacts

| Role | Contact | When to Notify |
|------|---------|----------------|
| On-call Engineer | @oncall | All rollbacks |
| Platform Team | @platform | Infrastructure issues |
| ML Team | @ml-team | Model-related issues |
| Product Owner | @product | User-facing impact |

---

## When to Rollback

### Decision Matrix

| Metric | Warning | Critical (Rollback) |
|--------|---------|---------------------|
| Error Rate | > 1% | > 5% |
| P99 Latency | > 500ms | > 1000ms |
| Availability | < 99.5% | < 99% |
| RAG Quality (NDCG) | < 0.7 | < 0.6 |
| User Complaints | 3+ in 10min | 10+ in 10min |

### Automatic Rollback Triggers

The CI/CD pipeline will automatically rollback if:

1. **Canary health check fails** after 10 minutes
2. **Error rate exceeds 5%** during canary phase
3. **P99 latency exceeds 500ms** during canary phase
4. **Health endpoint returns unhealthy** status

### Manual Rollback Indicators

Consider manual rollback when:

- [ ] Users report degraded experience
- [ ] Monitoring shows elevated errors
- [ ] New feature causes unexpected behavior
- [ ] Security vulnerability discovered
- [ ] Data integrity concerns identified
- [ ] Downstream service failures

---

## Rollback Procedures

### Procedure 1: Automated Rollback (Recommended)

**Time to complete: 5-10 minutes**

1. **Trigger rollback workflow**
   ```bash
   gh workflow run rollback.yml \
     -f environment=production \
     -f target_version=v1.2.3 \
     -f skip_confirmation=false
   ```

2. **Monitor workflow progress**
   ```bash
   gh run watch
   ```

3. **Verify rollback completion**
   - Check workflow status in GitHub Actions
   - Verify deployment version matches target

### Procedure 2: Manual Kubernetes Rollback

**Time to complete: 2-5 minutes**

1. **Check current deployment status**
   ```bash
   kubectl -n production get deployment rag-api -o wide
   kubectl -n production rollout history deployment/rag-api
   ```

2. **Rollback to previous version**
   ```bash
   # Rollback to immediate previous version
   kubectl -n production rollout undo deployment/rag-api
   
   # OR rollback to specific revision
   kubectl -n production rollout undo deployment/rag-api --to-revision=3
   ```

3. **Monitor rollback progress**
   ```bash
   kubectl -n production rollout status deployment/rag-api --watch
   ```

4. **Verify pods are healthy**
   ```bash
   kubectl -n production get pods -l app=rag-api
   kubectl -n production logs -l app=rag-api --tail=50
   ```

### Procedure 3: Docker/Container Registry Rollback

**Time to complete: 5-10 minutes**

1. **Identify target version tag**
   ```bash
   # List available tags
   gh api repos/{owner}/{repo}/packages/container/rag-api/versions \
     --jq '.[].metadata.container.tags[]'
   ```

2. **Update deployment to use previous image**
   ```bash
   kubectl -n production set image deployment/rag-api \
     rag-api=ghcr.io/{owner}/rag-api:v1.2.3
   ```

3. **Wait for rollout to complete**
   ```bash
   kubectl -n production rollout status deployment/rag-api
   ```

### Procedure 4: Canary Rollback (During Canary Phase)

**Time to complete: 2-3 minutes**

1. **Scale down canary deployment**
   ```bash
   kubectl -n production scale deployment/rag-api-canary --replicas=0
   ```

2. **Remove canary traffic split**
   ```bash
   kubectl -n production patch virtualservice rag-api \
     --type=json -p='[{"op": "remove", "path": "/spec/http/0"}]'
   ```

3. **Verify all traffic goes to stable**
   ```bash
   kubectl -n production get virtualservice rag-api -o yaml
   ```

---

## Post-Rollback Verification

### Immediate Checks (0-5 minutes)

- [ ] All pods running and ready
  ```bash
  kubectl -n production get pods -l app=rag-api
  ```

- [ ] Health endpoint returns 200
  ```bash
  curl https://api.example.com/health
  ```

- [ ] Version endpoint shows expected version
  ```bash
  curl https://api.example.com/version
  ```

### Short-term Monitoring (5-30 minutes)

- [ ] Error rate returning to baseline
- [ ] Latency returning to baseline
- [ ] No new error patterns in logs
- [ ] User reports decreasing

### Monitoring Commands

```bash
# Check recent logs for errors
kubectl -n production logs -l app=rag-api --since=5m | grep -i error

# Check resource usage
kubectl -n production top pods -l app=rag-api

# Check recent events
kubectl -n production get events --sort-by=.lastTimestamp | head -20
```

### Verification Checklist

| Check | Expected | Command |
|-------|----------|---------|
| Pod Count | >= 3 | `kubectl get pods -l app=rag-api \| wc -l` |
| Ready Status | All Ready | `kubectl get pods -l app=rag-api` |
| Error Rate | < 1% | Check Grafana dashboard |
| P99 Latency | < 300ms | Check Grafana dashboard |
| Health Check | 200 OK | `curl /health` |

---

## Communication Templates

### Incident Start Notification

```
🚨 INCIDENT: Production Rollback Initiated

Environment: Production
Time: [TIMESTAMP]
Reason: [BRIEF DESCRIPTION]
Action: Rolling back from v[CURRENT] to v[TARGET]
Impact: [EXPECTED IMPACT]

Status: In Progress
Lead: @[ONCALL_ENGINEER]

Updates to follow.
```

### Incident Resolution Notification

```
✅ RESOLVED: Production Rollback Complete

Environment: Production
Time Started: [START_TIME]
Time Resolved: [END_TIME]
Duration: [DURATION]

Action Taken: Rolled back from v[CURRENT] to v[TARGET]
Root Cause: [BRIEF ROOT CAUSE or "Under Investigation"]
Impact: [ACTUAL IMPACT]

Next Steps:
- [ ] Post-incident review scheduled for [DATE]
- [ ] Fix to be deployed after review

Questions? Contact @[ONCALL_ENGINEER]
```

### Post-Incident Review Template

```markdown
## Post-Incident Review: [DATE] Rollback

### Summary
- **Date/Time**: 
- **Duration**: 
- **Impact**: 
- **Root Cause**: 

### Timeline
| Time | Event |
|------|-------|
| HH:MM | Issue detected |
| HH:MM | Rollback initiated |
| HH:MM | Rollback complete |
| HH:MM | Verification complete |

### What Went Well
- 

### What Could Be Improved
- 

### Action Items
- [ ] 
- [ ] 

### Attendees
- 
```

---

## Troubleshooting

### Common Issues

#### Issue: Rollback stuck in "Progressing" state

**Symptoms:**
```
NAME      READY   UP-TO-DATE   AVAILABLE   AGE
rag-api   2/5     3            2           10m
```

**Solution:**
```bash
# Check pod status
kubectl -n production describe pods -l app=rag-api

# Check events
kubectl -n production get events --sort-by=.lastTimestamp

# If pods are failing to start, check logs
kubectl -n production logs -l app=rag-api --previous
```

#### Issue: Image pull errors

**Symptoms:**
```
Failed to pull image "ghcr.io/...": rpc error: code = NotFound
```

**Solution:**
```bash
# Verify image exists
gh api repos/{owner}/{repo}/packages/container/rag-api/versions

# Check image pull secret
kubectl -n production get secret regcred -o yaml

# Recreate pull secret if needed
kubectl -n production delete secret regcred
kubectl -n production create secret docker-registry regcred \
  --docker-server=ghcr.io \
  --docker-username=$GITHUB_USER \
  --docker-password=$GITHUB_TOKEN
```

#### Issue: Health checks failing after rollback

**Symptoms:**
- Pods running but not ready
- Liveness/readiness probes failing

**Solution:**
```bash
# Check probe configuration
kubectl -n production describe deployment rag-api | grep -A5 "Liveness\|Readiness"

# Check if dependencies are available
kubectl -n production exec -it <pod> -- curl localhost:8000/health

# Check environment variables
kubectl -n production exec -it <pod> -- env | grep -E "QDRANT|REDIS|OPENAI"
```

#### Issue: Traffic still going to bad version

**Symptoms:**
- Rollback complete but errors continue
- Service mesh not routing correctly

**Solution:**
```bash
# Force endpoint update
kubectl -n production rollout restart deployment/rag-api

# Check service endpoints
kubectl -n production get endpoints rag-api

# Verify pod labels match service selector
kubectl -n production get svc rag-api -o yaml | grep selector -A5
kubectl -n production get pods -l app=rag-api --show-labels
```

---

## Related Documents

- [CI/CD Pipeline Documentation](./ci-cd-pipeline.md)
- [Deployment Architecture](./deployment-architecture.md)
- [Monitoring & Alerting](./monitoring.md)
- [Incident Response Process](./incident-response.md)

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | YYYY-MM-DD | DevOps | Initial version |
