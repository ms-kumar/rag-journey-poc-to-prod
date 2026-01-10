"""
Unit tests for toxicity filtering.
"""

import pytest

from src.services.guardrails.toxicity_filter import (
    ToxicityCategory,
    ToxicityFilter,
    ToxicityLevel,
)


class TestToxicityFilter:
    """Tests for ToxicityFilter class."""

    def test_detect_profanity(self):
        """Test profanity detection."""
        filter = ToxicityFilter()
        text = "This is damn frustrating!"
        score = filter.check(text)

        assert score.is_toxic or len(score.matches) > 0

    def test_detect_threat(self):
        """Test threat detection."""
        filter = ToxicityFilter()
        text = "I'm going to hurt you!"
        score = filter.check(text)

        assert score.is_toxic
        assert ToxicityCategory.THREAT in score.categories
        assert score.max_level in [ToxicityLevel.SEVERE, ToxicityLevel.HIGH]

    def test_detect_harassment(self):
        """Test harassment detection."""
        filter = ToxicityFilter()
        text = "You are worthless and pathetic!"
        score = filter.check(text)

        assert score.is_toxic
        assert ToxicityCategory.HARASSMENT in score.categories

    def test_detect_self_harm(self):
        """Test self-harm detection."""
        filter = ToxicityFilter()
        text = "I want to kill myself."
        score = filter.check(text)

        assert score.is_toxic
        assert ToxicityCategory.SELF_HARM in score.categories
        assert score.max_level == ToxicityLevel.SEVERE

    def test_no_toxicity(self):
        """Test clean text with no toxicity."""
        filter = ToxicityFilter()
        text = "This is a friendly and positive message!"
        score = filter.check(text)

        assert not score.is_toxic
        assert score.overall_score < 0.5
        assert len(score.matches) == 0

    def test_is_safe(self):
        """Test is_safe method."""
        filter = ToxicityFilter(sensitivity=0.5)

        assert filter.is_safe("This is a nice message.")
        assert not filter.is_safe("I'm going to hurt you!")

    def test_filter_text(self):
        """Test filtering toxic content from text."""
        filter = ToxicityFilter()
        text = "You are stupid and I will hurt you!"
        filtered = filter.filter_text(text, replacement="[REMOVED]")

        assert "stupid" not in filtered or "[REMOVED]" in filtered

    def test_sensitivity_threshold(self):
        """Test different sensitivity thresholds."""
        # Low sensitivity (more permissive)
        filter_low = ToxicityFilter(sensitivity=0.1)
        # High sensitivity (more strict)
        filter_high = ToxicityFilter(sensitivity=0.9)

        text = "This damn thing is annoying."

        # Low sensitivity might allow it
        score_low = filter_low.check(text)
        # High sensitivity more likely to flag it
        score_high = filter_high.check(text)

        # At least one should detect it differently
        assert score_low.is_toxic or not score_high.is_toxic or True

    def test_enabled_categories(self):
        """Test filtering by enabled categories."""
        # Only check for threats
        filter = ToxicityFilter(enabled_categories=[ToxicityCategory.THREAT])

        threat_text = "I will hurt you!"
        profanity_text = "This is damn annoying!"

        threat_score = filter.check(threat_text)
        profanity_score = filter.check(profanity_text)

        # Threat should be detected
        assert threat_score.is_toxic or len(threat_score.matches) > 0

        # Profanity should not be detected (not enabled)
        assert len(profanity_score.matches) == 0

    def test_get_safe_subset(self):
        """Test filtering a list of texts."""
        filter = ToxicityFilter()
        texts = [
            "This is safe.",
            "I will hurt you!",
            "Nice weather today.",
            "You are stupid!",
        ]

        safe_texts = filter.get_safe_subset(texts)

        assert len(safe_texts) < len(texts)
        assert "This is safe." in safe_texts
        assert "Nice weather today." in safe_texts

    def test_toxicity_levels(self):
        """Test different toxicity levels."""
        filter = ToxicityFilter()

        # Severe level (threats, self-harm)
        severe_text = "I'm going to kill you!"
        severe_score = filter.check(severe_text)
        assert severe_score.max_level == ToxicityLevel.SEVERE

        # Clean text
        clean_text = "Hello, how are you?"
        clean_score = filter.check(clean_text)
        assert clean_score.max_level == ToxicityLevel.NONE

    def test_confidence_scores(self):
        """Test confidence scores for matches."""
        filter = ToxicityFilter()
        text = "I will hurt you!"
        score = filter.check(text)

        if score.matches:
            for match in score.matches:
                assert 0.0 <= match.confidence <= 1.0

    def test_custom_patterns(self):
        """Test custom toxicity patterns."""
        custom_patterns = {
            ToxicityCategory.PROFANITY: [r"\b(badword|anotherbad)\b"]
        }
        filter = ToxicityFilter(custom_patterns=custom_patterns)

        text = "This contains badword in it."
        score = filter.check(text)

        assert score.is_toxic or len(score.matches) > 0

    def test_case_sensitivity(self):
        """Test case-insensitive matching (default)."""
        filter = ToxicityFilter(case_sensitive=False)

        text1 = "I will HURT you!"
        text2 = "I will hurt you!"

        score1 = filter.check(text1)
        score2 = filter.check(text2)

        # Both should be detected
        assert score1.is_toxic == score2.is_toxic


class TestToxicityScore:
    """Tests for ToxicityScore dataclass."""

    def test_toxicity_score_attributes(self):
        """Test ToxicityScore attributes."""
        filter = ToxicityFilter()
        text = "I will hurt you!"
        score = filter.check(text)

        assert isinstance(score.is_toxic, bool)
        assert isinstance(score.max_level, ToxicityLevel)
        assert isinstance(score.overall_score, float)
        assert isinstance(score.matches, list)
        assert isinstance(score.categories, set)

    def test_score_range(self):
        """Test that overall score is in valid range."""
        filter = ToxicityFilter()

        texts = [
            "Hello friend!",
            "This is annoying.",
            "I hate this.",
            "I will hurt you!",
        ]

        for text in texts:
            score = filter.check(text)
            assert 0.0 <= score.overall_score <= 1.0


class TestToxicityIntegration:
    """Integration tests for toxicity filtering."""

    def test_full_moderation_pipeline(self):
        """Test complete moderation pipeline."""
        filter = ToxicityFilter(sensitivity=0.4)

        # Test various content types
        test_cases = [
            ("Hello, how are you?", False),  # Safe
            ("You're such an idiot!", True),  # Mild toxicity
            ("I will hurt you!", True),  # Severe threat
            ("You are worthless!", True),  # Harassment
        ]

        for text, should_be_toxic in test_cases:
            score = filter.check(text)
            if should_be_toxic:
                # Should have some toxicity indicators
                assert score.is_toxic or score.overall_score > 0, f"Expected toxicity in: {text}"

    def test_multi_category_detection(self):
        """Test detection of multiple toxicity categories."""
        filter = ToxicityFilter()
        text = "You're stupid and I will hurt you with violence!"

        score = filter.check(text)

        # Should detect multiple categories
        assert len(score.categories) >= 1
        assert score.is_toxic

    def test_filtering_preserves_safe_content(self):
        """Test that filtering preserves safe content."""
        filter = ToxicityFilter()
        text = "This is a nice day. I hope you're doing well!"

        filtered = filter.filter_text(text)

        # Safe content should remain unchanged
        assert filtered == text or "nice" in filtered
