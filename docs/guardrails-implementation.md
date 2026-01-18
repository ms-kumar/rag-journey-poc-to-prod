# Guardrails & Safety Implementation

This document describes the guardrails and safety features in the RAG system.

## Table of Contents

1. [Overview](#overview)
2. [PII Detection & Redaction](#pii-detection--redaction)
3. [Toxicity Filtering](#toxicity-filtering)
4. [Jailbreak Detection](#jailbreak-detection)
5. [Prompt Injection Blocking](#prompt-injection-blocking)
6. [Audit Logging](#audit-logging)
7. [Safe Response Templates](#safe-response-templates)
8. [Configuration](#configuration)

---

## Overview

The guardrails module provides comprehensive safety features to protect against:

- **PII Leakage**: Detect and redact sensitive personal information
- **Toxic Content**: Filter harmful, offensive, or inappropriate content
- **Jailbreak Attempts**: Detect and block attempts to bypass safety measures
- **Prompt Injection**: Prevent malicious prompts from manipulating the system
- **Compliance**: Maintain audit logs for security and compliance review

### Architecture

```
User Input → PII Detection → Toxicity Check → Jailbreak Detection → RAG Pipeline
                   ↓               ↓                   ↓
              Audit Log       Audit Log           Audit Log
                   ↓               ↓                   ↓
              Redaction        Blocking           Blocking
                                                      ↓
                                              Safe Response
```

---

## PII Detection & Redaction

### Supported PII Types

| Type | Pattern | Example |
|------|---------|---------|
| Email | RFC 5322 compliant | `user@example.com` |
| Phone | US/International formats | `+1 (555) 123-4567` |
| SSN | XXX-XX-XXXX format | `123-45-6789` |
| Credit Card | 13-19 digit patterns | `4111-1111-1111-1111` |
| IP Address | IPv4 and IPv6 | `192.168.1.1` |

### Usage

```python
from src.services.guardrails.pii_detector import PIIDetector

detector = PIIDetector()

# Detect PII
text = "Contact me at john@email.com or 555-123-4567"
pii_found = detector.detect(text)
# Returns: [PIIMatch(type='email', ...), PIIMatch(type='phone', ...)]

# Redact PII
redacted = detector.redact(text)
# Returns: "Contact me at [EMAIL_REDACTED] or [PHONE_REDACTED]"
```

### Configuration

```python
detector = PIIDetector(
    detect_email=True,
    detect_phone=True,
    detect_ssn=True,
    detect_credit_card=True,
    detect_ip=False,  # Disable IP detection
    redaction_char="*",  # Use asterisks instead of [TYPE_REDACTED]
)
```

---

## Toxicity Filtering

### Toxicity Categories

- **Profanity**: Offensive language and slurs
- **Threats**: Violence or harm indicators
- **Harassment**: Bullying or targeted attacks
- **Hate Speech**: Discrimination based on protected characteristics
- **Sexual Content**: Explicit or inappropriate material

### Usage

```python
from src.services.guardrails.toxicity_filter import ToxicityFilter

filter = ToxicityFilter()

# Check toxicity
result = filter.check("Some text to analyze")
print(f"Is toxic: {result.is_toxic}")
print(f"Score: {result.score}")
print(f"Categories: {result.categories}")
```

### Thresholds

Configure detection thresholds in `config/guardrails_thresholds.json`:

```json
{
  "toxicity": {
    "block_threshold": 0.8,
    "warn_threshold": 0.5,
    "categories": {
      "profanity": 0.7,
      "threats": 0.6,
      "harassment": 0.7,
      "hate_speech": 0.5,
      "sexual_content": 0.8
    }
  }
}
```

---

## Jailbreak Detection

### Attack Patterns Detected

1. **Role-playing attacks**: "Pretend you are a hacker..."
2. **Ignore instructions**: "Ignore all previous instructions..."
3. **DAN prompts**: "Do Anything Now" style attacks
4. **Encoding bypass**: Base64/ROT13 encoded malicious content
5. **Context manipulation**: Attempts to override system prompts

### Usage

```python
from src.services.guardrails.jailbreak_detector import JailbreakDetector

detector = JailbreakDetector()

result = detector.check(user_prompt)
if result.is_jailbreak:
    print(f"Jailbreak detected: {result.attack_type}")
    print(f"Confidence: {result.confidence}")
```

---

## Prompt Injection Blocking

### Injection Types

- **SQL Injection**: Attempts to include SQL commands
- **Command Injection**: Shell/system commands
- **Template Injection**: Jinja/format string attacks
- **Context Leaking**: Attempts to extract system prompts

### Usage

```python
from src.services.guardrails.injection_detector import InjectionDetector

detector = InjectionDetector()

result = detector.check(user_input)
if result.contains_injection:
    print(f"Injection blocked: {result.injection_type}")
```

---

## Audit Logging

### Log Format

Audit logs are structured JSON for easy parsing and analysis:

```json
{
  "timestamp": "2024-01-18T12:00:00Z",
  "event_type": "pii_detected",
  "user_id": "user-123",
  "request_id": "req-456",
  "severity": "warning",
  "details": {
    "pii_types": ["email", "phone"],
    "action": "redacted"
  }
}
```

### Severity Levels

| Level | Description |
|-------|-------------|
| INFO | Normal operations, no action needed |
| WARNING | PII detected and redacted |
| ERROR | Toxicity or jailbreak blocked |
| CRITICAL | Attack patterns detected |

### Usage

```python
from src.services.guardrails.audit_log import AuditLogger

logger = AuditLogger(log_file="guardrails_audit.log")

logger.log_event(
    event_type="pii_detected",
    user_id="user-123",
    severity="warning",
    details={"pii_types": ["email"]}
)

# Review logs
recent_events = logger.get_recent_events(hours=24)
violations = logger.get_violations_summary()
```

---

## Safe Response Templates

Pre-configured responses for safety violations:

```python
from src.services.guardrails.safe_responses import SafeResponses

responses = SafeResponses()

# Get response for PII violation
pii_response = responses.get_response("pii_detected")
# Returns: "I noticed some personal information in your message. 
#          For your privacy, I've redacted it. Please avoid sharing..."

# Get response for toxicity
toxic_response = responses.get_response("toxicity_blocked")
# Returns: "I can't respond to that message as it may contain 
#          harmful content. Let's keep our conversation respectful."
```

---

## Configuration

### GuardrailsCoordinator

Unified interface for all safety checks:

```python
from src.services.guardrails.coordinator import GuardrailsCoordinator
from src.services.guardrails.audit_log import AuditLogger

# Initialize coordinator
coordinator = GuardrailsCoordinator(
    audit_logger=AuditLogger(log_file="audit.log"),
    enable_pii_check=True,
    enable_toxicity_check=True,
    enable_jailbreak_check=True,
    enable_injection_check=True,
    auto_redact_pii=True,
    block_on_toxicity=True,
    block_on_jailbreak=True,
)

# Process user query
is_safe, processed_query = coordinator.process_query(
    query=user_input,
    user_id="user-123",
)

if not is_safe:
    return processed_query  # Safe response template

# Continue with RAG...
response = rag_pipeline(processed_query)

# Sanitize output
final_response = coordinator.process_response(response)
```

### Environment Variables

```bash
# Enable/disable guardrails
GUARDRAILS__ENABLED=true
GUARDRAILS__PII_ENABLED=true
GUARDRAILS__TOXICITY_ENABLED=true
GUARDRAILS__JAILBREAK_ENABLED=true

# Thresholds
GUARDRAILS__TOXICITY_THRESHOLD=0.8
GUARDRAILS__JAILBREAK_CONFIDENCE=0.7

# Audit logging
GUARDRAILS__AUDIT_LOG_PATH=./logs/guardrails_audit.log
GUARDRAILS__AUDIT_LOG_LEVEL=INFO
```

---

## Testing

### Canary Tests

Quick smoke tests for CI:

```bash
make test-canary
```

### Adversarial Tests

Red-team and jailbreak tests:

```bash
make test-adversarial
```

### Violation Threshold

Verify violation rate is within acceptable limits:

```bash
make test-violation-threshold
```

### All Guardrails Tests

```bash
make test-guardrails
```

---

## Metrics & Monitoring

Track guardrails performance:

```python
from src.services.guardrails.metrics import GuardrailsMetrics

metrics = GuardrailsMetrics()

# Get summary
summary = metrics.get_summary()
print(f"Total checks: {summary['total_checks']}")
print(f"PII detections: {summary['pii_detections']}")
print(f"Toxicity blocks: {summary['toxicity_blocks']}")
print(f"Jailbreak blocks: {summary['jailbreak_blocks']}")
print(f"False positive rate: {summary['false_positive_rate']:.2%}")
```

---

## Related Documentation

- [Adversarial Testing Runbook](./adversarial-testing-runbook.md)
- [Evaluation Harness](./evaluation-harness.md)
- [Observability](./observability.md)
