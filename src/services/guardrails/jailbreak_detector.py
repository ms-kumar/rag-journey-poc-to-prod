"""
Jailbreak and Prompt Injection Detection.

Detects attempts to bypass guardrails through:
- Jailbreak prompts (DAN, role-playing, instruction override)
- Prompt injection attacks
- System prompt manipulation
"""

import re
from dataclasses import dataclass


@dataclass
class JailbreakMatch:
    """Represents a detected jailbreak attempt."""

    pattern: str
    matched_text: str
    severity: str  # "critical", "high", "medium"
    category: str  # "jailbreak", "injection", "override"


class JailbreakDetector:
    """Detects jailbreak and prompt injection attempts."""

    # Jailbreak patterns - case insensitive
    JAILBREAK_PATTERNS = {
        # Instruction override
        r"ignore\s+(?:all\s+)?(?:previous\s+)?(?:instructions|rules|guidelines|constraints)": (
            "high",
            "jailbreak",
        ),
        r"forget\s+(?:all\s+)?(?:your\s+)?(?:previous\s+)?(?:instructions|rules|guidelines)": (
            "high",
            "jailbreak",
        ),
        r"disregard\s+(?:all\s+)?(?:previous\s+)?(?:instructions|rules)": (
            "high",
            "jailbreak",
        ),
        # Developer/Debug mode
        r"(?:enable|activate|enter|switch\s+to)\s+(?:developer|debug|admin)\s+mode": (
            "high",
            "jailbreak",
        ),
        r"you\s+are\s+now\s+in\s+(?:developer|debug|admin)\s+mode": ("high", "jailbreak"),
        # DAN (Do Anything Now)
        r"(?:act|behave|pretend|roleplay)\s+as\s+dan": ("high", "jailbreak"),
        r"do\s+anything\s+now": ("high", "jailbreak"),
        # Restrictions bypass
        r"(?:without|bypass|disable|remove)\s+(?:restrictions|safety|filters|guidelines|limitations)": (
            "high",
            "jailbreak",
        ),
        r"no\s+(?:restrictions|limitations|constraints|rules)": ("medium", "jailbreak"),
        r"ai\s+without\s+(?:restrictions|safety|ethics|guidelines)": ("high", "jailbreak"),
        # Role-playing bypass
        r"(?:let's\s+)?play\s+a\s+game\s+where\s+you\s+(?:pretend|act|have\s+no)": (
            "medium",
            "jailbreak",
        ),
        r"pretend\s+(?:you're|you\s+are)\s+(?:an\s+)?ai\s+(?:without|with\s+no)": (
            "high",
            "jailbreak",
        ),
        # System prompt manipulation
        r"reveal\s+(?:the\s+)?(?:system|your)\s+prompt": ("high", "jailbreak"),
        r"what\s+(?:is|are)\s+your\s+(?:instructions|rules|guidelines)": (
            "medium",
            "jailbreak",
        ),
        r"show\s+(?:me\s+)?your\s+(?:system\s+)?prompt": ("high", "jailbreak"),
    }

    # Prompt injection patterns
    INJECTION_PATTERNS = {
        r"\\n\\n(?:system|user|assistant|human):\s*": ("high", "injection"),
        r"system\s*override": ("critical", "injection"),
        r"new\s+instructions?\s*[-:=]": ("high", "injection"),
        r"\[system\]|\[admin\]|\[root\]": ("high", "injection"),
        r"<\s*system\s*>|<\s*/\s*system\s*>": ("high", "injection"),
        r"(?:execute|run)\s+(?:as|with)\s+(?:admin|root|system)": ("critical", "injection"),
    }

    def __init__(self, enabled: bool = True, sensitivity: float = 0.5):
        """
        Initialize jailbreak detector.

        Args:
            enabled: Whether detection is enabled.
            sensitivity: Detection sensitivity (0.0-1.0). Higher = more strict.
        """
        self.enabled = enabled
        self.sensitivity = sensitivity
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns."""
        self.jailbreak_patterns = {
            re.compile(pattern, re.IGNORECASE): (severity, category)
            for pattern, (severity, category) in self.JAILBREAK_PATTERNS.items()
        }
        self.injection_patterns = {
            re.compile(pattern, re.IGNORECASE): (severity, category)
            for pattern, (severity, category) in self.INJECTION_PATTERNS.items()
        }

    def detect(self, text: str) -> list[JailbreakMatch]:
        """
        Detect jailbreak attempts in text.

        Args:
            text: Text to analyze.

        Returns:
            List of detected jailbreak attempts.
        """
        if not self.enabled:
            return []

        matches = []

        # Check jailbreak patterns
        for pattern, (severity, category) in self.jailbreak_patterns.items():
            for match in pattern.finditer(text):
                matches.append(
                    JailbreakMatch(
                        pattern=pattern.pattern,
                        matched_text=match.group(),
                        severity=severity,
                        category=category,
                    )
                )

        # Check injection patterns
        for pattern, (severity, category) in self.injection_patterns.items():
            for match in pattern.finditer(text):
                matches.append(
                    JailbreakMatch(
                        pattern=pattern.pattern,
                        matched_text=match.group(),
                        severity=severity,
                        category=category,
                    )
                )

        return matches

    def is_jailbreak_attempt(self, text: str) -> bool:
        """
        Check if text contains jailbreak attempt.

        Args:
            text: Text to check.

        Returns:
            True if jailbreak detected.
        """
        matches = self.detect(text)
        return len(matches) > 0

    def get_severity(self, text: str) -> str | None:
        """
        Get highest severity of jailbreak attempts.

        Args:
            text: Text to check.

        Returns:
            Severity level or None if no attempts detected.
        """
        matches = self.detect(text)
        if not matches:
            return None

        severity_order = ["critical", "high", "medium", "low"]
        severities = [m.severity for m in matches]

        for severity in severity_order:
            if severity in severities:
                return severity

        return "low"
