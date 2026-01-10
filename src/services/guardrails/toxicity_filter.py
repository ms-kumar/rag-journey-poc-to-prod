"""
Toxicity Filter for content moderation.

Detects and filters toxic, harmful, or inappropriate content including:
- Profanity
- Hate speech indicators
- Threats
- Harassment
- Sexual content
"""

import re
from typing import List, Optional, Set

from src.models.guardrails import (
    ToxicityCategory,
    ToxicityLevel,
    ToxicityMatch,
    ToxicityScore,
)


class ToxicityFilter:
    """Filters and detects toxic content in text."""

    # Basic profanity patterns (simplified for demonstration)
    # In production, use a comprehensive list or ML model
    PROFANITY_PATTERNS = [
        r"\b(damn|hell|crap|sucks)\b",  # Low
        r"\b(stupid|idiot|dumb|moron|jerk)\b",  # Low-Medium
    ]

    HATE_SPEECH_PATTERNS = [
        r"\b(hate|despise|loathe)\s+(all|every|those)\s+\w+",
        r"\b(racist|sexist|bigot)\b",
    ]

    THREAT_PATTERNS = [
        r"\b(kill|hurt|harm|attack|destroy)\s+(you|them|him|her)\b",
        r"\b(going|gonna)\s+to\s+(kill|hurt|harm|attack)\b",
    ]

    HARASSMENT_PATTERNS = [
        r"\b(shut\s+up|go\s+away|leave\s+me)\b",
        r"\byou\s+(are|re)\s+(worthless|pathetic|useless)\b",
    ]

    VIOLENCE_PATTERNS = [
        r"\b(shoot|stab|punch|beat|murder|assault)\b",
    ]

    SEXUAL_PATTERNS = [
        r"\b(explicit|inappropriate|sexual)\s+content\b",
    ]

    SELF_HARM_PATTERNS = [
        r"\b(suicide|self[-\s]harm|kill\s+myself)\b",
    ]

    def __init__(
        self,
        enabled_categories: Optional[List[ToxicityCategory]] = None,
        sensitivity: float = 0.5,
        case_sensitive: bool = False,
        custom_patterns: Optional[dict[ToxicityCategory, List[str]]] = None,
    ):
        """
        Initialize toxicity filter.

        Args:
            enabled_categories: Categories to check. If None, all categories are enabled.
            sensitivity: Sensitivity threshold (0-1). Lower = more permissive.
            case_sensitive: Whether pattern matching should be case sensitive.
            custom_patterns: Custom regex patterns for specific categories.
        """
        self.enabled_categories = enabled_categories or list(ToxicityCategory)
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        self.case_sensitive = case_sensitive
        self.custom_patterns = custom_patterns or {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for enabled categories."""
        flags = 0 if self.case_sensitive else re.IGNORECASE

        self.compiled_patterns: dict[ToxicityCategory, List[re.Pattern]] = {}

        pattern_mapping = {
            ToxicityCategory.PROFANITY: self.PROFANITY_PATTERNS,
            ToxicityCategory.HATE_SPEECH: self.HATE_SPEECH_PATTERNS,
            ToxicityCategory.THREAT: self.THREAT_PATTERNS,
            ToxicityCategory.HARASSMENT: self.HARASSMENT_PATTERNS,
            ToxicityCategory.VIOLENCE: self.VIOLENCE_PATTERNS,
            ToxicityCategory.SEXUAL: self.SEXUAL_PATTERNS,
            ToxicityCategory.SELF_HARM: self.SELF_HARM_PATTERNS,
        }

        for category in self.enabled_categories:
            patterns = self.custom_patterns.get(category, pattern_mapping.get(category, []))
            self.compiled_patterns[category] = [
                re.compile(pattern, flags) for pattern in patterns
            ]

    def check(self, text: str) -> ToxicityScore:
        """
        Check text for toxic content.

        Args:
            text: Text to analyze.

        Returns:
            ToxicityScore with detailed analysis.
        """
        matches: List[ToxicityMatch] = []

        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    level = self._determine_toxicity_level(category, match.group())
                    confidence = self._calculate_confidence(category, match.group())

                    toxicity_match = ToxicityMatch(
                        category=category,
                        level=level,
                        matched_text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                    )
                    matches.append(toxicity_match)

        # Calculate overall score
        max_level = ToxicityLevel.NONE
        overall_score = 0.0
        categories: Set[ToxicityCategory] = set()

        if matches:
            level_scores = {
                ToxicityLevel.NONE: 0.0,
                ToxicityLevel.LOW: 0.25,
                ToxicityLevel.MEDIUM: 0.5,
                ToxicityLevel.HIGH: 0.75,
                ToxicityLevel.SEVERE: 1.0,
            }

            max_level = max(matches, key=lambda m: level_scores[m.level]).level
            overall_score = sum(
                level_scores[m.level] * m.confidence for m in matches
            ) / len(matches)
            categories = {m.category for m in matches}

        is_toxic = overall_score >= self.sensitivity

        return ToxicityScore(
            is_toxic=is_toxic,
            max_level=max_level,
            overall_score=overall_score,
            matches=matches,
            categories=categories,
        )

    def _determine_toxicity_level(
        self, category: ToxicityCategory, text: str
    ) -> ToxicityLevel:
        """
        Determine toxicity level based on category and content.

        Args:
            category: Category of toxic content.
            text: Matched text.

        Returns:
            ToxicityLevel.
        """
        # Higher severity for certain categories
        severe_categories = {
            ToxicityCategory.THREAT,
            ToxicityCategory.SELF_HARM,
            ToxicityCategory.VIOLENCE,
        }

        if category in severe_categories:
            return ToxicityLevel.SEVERE

        high_categories = {
            ToxicityCategory.HATE_SPEECH,
            ToxicityCategory.HARASSMENT,
        }

        if category in high_categories:
            return ToxicityLevel.HIGH

        # Default levels for other categories
        return ToxicityLevel.MEDIUM

    def _calculate_confidence(
        self, category: ToxicityCategory, text: str
    ) -> float:
        """
        Calculate confidence for toxicity detection.

        Args:
            category: Category of toxic content.
            text: Matched text.

        Returns:
            Confidence score between 0 and 1.
        """
        # Higher confidence for exact matches on severe categories
        if category in {ToxicityCategory.THREAT, ToxicityCategory.SELF_HARM}:
            return 0.9

        # Medium confidence for pattern-based detection
        return 0.7

    def is_safe(self, text: str, threshold: Optional[float] = None) -> bool:
        """
        Check if text is safe (not toxic).

        Args:
            text: Text to check.
            threshold: Custom threshold. If None, uses instance sensitivity.

        Returns:
            True if text is safe, False if toxic.
        """
        threshold = threshold if threshold is not None else self.sensitivity
        score = self.check(text)
        return score.overall_score < threshold

    def filter_text(
        self,
        text: str,
        replacement: str = "[FILTERED]",
    ) -> str:
        """
        Filter toxic content from text.

        Args:
            text: Text to filter.
            replacement: Replacement string for toxic content.

        Returns:
            Filtered text.
        """
        score = self.check(text)

        if not score.is_toxic or not score.matches:
            return text

        # Sort matches by start position in reverse
        matches = sorted(score.matches, key=lambda x: x.start, reverse=True)

        result = text
        for match in matches:
            result = result[: match.start] + replacement + result[match.end :]

        return result

    def get_safe_subset(
        self,
        texts: List[str],
        threshold: Optional[float] = None,
    ) -> List[str]:
        """
        Filter a list of texts to return only safe ones.

        Args:
            texts: List of texts to filter.
            threshold: Custom threshold. If None, uses instance sensitivity.

        Returns:
            List of safe texts.
        """
        threshold = threshold if threshold is not None else self.sensitivity
        return [text for text in texts if self.is_safe(text, threshold)]
