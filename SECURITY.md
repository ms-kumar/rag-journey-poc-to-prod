# Security Policy

## Supported Versions

We release security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please report security vulnerabilities by emailing:

📧 **[INSERT YOUR SECURITY EMAIL HERE]**

### What to Include

Please include as much information as possible:

1. **Description** of the vulnerability
2. **Steps to reproduce** the issue
3. **Potential impact** of the vulnerability
4. **Affected versions**
5. **Suggested fix** (if you have one)
6. **Your contact information** for follow-up

### Example Report

```
Subject: [SECURITY] SQL Injection in Query Parser

Description:
The query parser in src/core/query_understanding.py does not properly
sanitize user input, allowing SQL injection attacks.

Steps to Reproduce:
1. Send a POST request to /api/v1/query with payload:
   {"query": "'; DROP TABLE users; --"}
2. Observe that the table is dropped

Impact:
- Severity: Critical
- Allows arbitrary SQL execution
- Could lead to data breach or deletion

Affected Versions:
- v0.1.0 and earlier

Suggested Fix:
Use parameterized queries instead of string concatenation.
```

## Response Timeline

We aim to respond to security reports according to the following timeline:

| Timeframe | Action |
|-----------|--------|
| **24 hours** | Initial response acknowledging receipt |
| **3 days** | Assessment of severity and impact |
| **7 days** | Action plan and timeline for fix |
| **30 days** | Security patch released (for critical issues) |

Actual timelines may vary based on severity and complexity.

## Security Update Process

1. **Verification** - We verify the reported vulnerability
2. **Assessment** - We assess severity using CVSS scoring
3. **Fix Development** - We develop and test a fix
4. **Disclosure** - We coordinate disclosure with the reporter
5. **Release** - We release a security patch
6. **Announcement** - We publish a security advisory

## Severity Levels

We use the following severity classifications:

### Critical (CVSS 9.0-10.0)
- Remote code execution
- Authentication bypass
- Data breach of sensitive information

**Response**: Immediate patch within 7 days

### High (CVSS 7.0-8.9)
- SQL injection
- Cross-site scripting (XSS)
- Privilege escalation

**Response**: Patch within 14 days

### Medium (CVSS 4.0-6.9)
- Information disclosure
- Denial of service
- CSRF vulnerabilities

**Response**: Patch within 30 days

### Low (CVSS 0.1-3.9)
- Minor information leaks
- Low-impact vulnerabilities

**Response**: Patch in next regular release

## Security Best Practices

When using this system in production:

### 1. Authentication & Authorization
- ✅ Use API keys or OAuth for authentication
- ✅ Implement rate limiting on endpoints
- ✅ Validate all user inputs
- ❌ Never expose internal APIs publicly

### 2. Data Protection
- ✅ Enable guardrails for PII detection
- ✅ Use prompt injection detection
- ✅ Sanitize all outputs
- ❌ Never log sensitive information

### 3. Infrastructure
- ✅ Run behind a reverse proxy (nginx, Traefik)
- ✅ Use HTTPS/TLS for all connections
- ✅ Keep dependencies updated
- ✅ Use security scanning in CI/CD
- ❌ Never run as root in containers

### 4. Configuration
- ✅ Use environment variables for secrets
- ✅ Rotate API keys regularly
- ✅ Enable audit logging
- ❌ Never commit secrets to version control

### 5. Dependencies
- ✅ Pin dependency versions in production
- ✅ Scan for vulnerabilities with `pip-audit`
- ✅ Update dependencies regularly
- ❌ Never use untrusted packages

### Security Checklist

```bash
# Scan dependencies for vulnerabilities
pip-audit

# Check Docker image for vulnerabilities
docker scan ghcr.io/ms-kumar/rag-journey-poc-to-prod:latest

# Run security linters
bandit -r src/
safety check
```

## Known Security Features

This project includes built-in security features:

### ✅ Guardrails
- **Prompt Injection Detection** - Blocks malicious prompts
- **PII Detection** - Identifies and masks sensitive data
- **Jailbreak Detection** - Prevents model abuse
- **Content Filtering** - Blocks inappropriate content

### ✅ Input Validation
- **Schema Validation** - Validates all API inputs
- **Size Limits** - Enforces maximum payload sizes
- **Type Checking** - Strong type validation

### ✅ Sandboxing
- **Code Execution Sandbox** - Isolated tool execution
- **Resource Limits** - CPU/memory constraints
- **Timeout Protection** - Prevents infinite loops

### ✅ Monitoring
- **Structured Logging** - Comprehensive audit trails
- **Health Checks** - Monitors service health
- **Cost Tracking** - Tracks API usage

## Security Advisories

Security advisories will be published at:
- GitHub Security Advisories: https://github.com/ms-kumar/rag-journey-poc-to-prod/security/advisories
- CHANGELOG.md with `[SECURITY]` tag

Subscribe to repository notifications to receive security updates.

## Responsible Disclosure

We follow responsible disclosure principles:

1. **Report Privately** - Don't disclose publicly until patched
2. **Coordinate Disclosure** - We'll work with you on timing
3. **Recognition** - We'll credit you in the security advisory (if desired)
4. **No Retaliation** - We won't take legal action against good-faith researchers

## Hall of Fame

We recognize security researchers who responsibly disclose vulnerabilities:

<!-- Will be updated as researchers report issues -->

---

Thank you for helping keep Advanced RAG System secure! 🔒
