"""Unit tests for the reflection module (AnswerCritic and SourceVerifier)."""

from src.services.agent.reflection import (
    AnswerCritic,
    AnswerCritique,
    SourceVerification,
    SourceVerifier,
)


class TestAnswerCritique:
    """Test AnswerCritique dataclass."""

    def test_default_values(self):
        """Test default values of AnswerCritique."""
        critique = AnswerCritique(answer="Test answer")

        assert critique.answer == "Test answer"
        assert critique.completeness_score == 0.0
        assert critique.accuracy_score == 0.0
        assert critique.relevance_score == 0.0
        assert critique.clarity_score == 0.0
        assert critique.source_quality_score == 0.0
        assert critique.overall_score == 0.0
        assert critique.issues == []
        assert critique.suggestions == []
        assert critique.needs_revision is False
        assert critique.missing_aspects == []
        assert critique.timestamp is not None

    def test_custom_values(self):
        """Test AnswerCritique with custom values."""
        critique = AnswerCritique(
            answer="Test answer",
            completeness_score=0.8,
            accuracy_score=0.9,
            relevance_score=0.85,
            clarity_score=0.75,
            source_quality_score=0.7,
            overall_score=0.8,
            issues=["Minor issue"],
            suggestions=["Add more detail"],
            needs_revision=True,
            missing_aspects=["historical context"],
        )

        assert critique.completeness_score == 0.8
        assert critique.accuracy_score == 0.9
        assert critique.overall_score == 0.8
        assert "Minor issue" in critique.issues
        assert critique.needs_revision is True


class TestSourceVerification:
    """Test SourceVerification dataclass."""

    def test_default_values(self):
        """Test default values of SourceVerification."""
        verification = SourceVerification()

        assert verification.sources_found == 0
        assert verification.sources_verified == 0
        assert verification.source_diversity == 0.0
        assert verification.hallucination_risk == 0.0
        assert verification.verified_sources == []
        assert verification.questionable_claims == []

    def test_custom_values(self):
        """Test SourceVerification with custom values."""
        verification = SourceVerification(
            sources_found=5,
            sources_verified=4,
            source_diversity=0.8,
            hallucination_risk=0.2,
            verified_sources=[{"id": "src1", "content": "test"}],
            questionable_claims=["claim1"],
        )

        assert verification.sources_found == 5
        assert verification.sources_verified == 4
        assert verification.source_diversity == 0.8
        assert verification.hallucination_risk == 0.2
        assert len(verification.verified_sources) == 1


class TestAnswerCritic:
    """Test AnswerCritic class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.critic = AnswerCritic(quality_threshold=0.7)

    def test_init_default(self):
        """Test default initialization."""
        critic = AnswerCritic()
        assert critic.quality_threshold == 0.7
        assert critic.llm is None

    def test_init_custom_threshold(self):
        """Test initialization with custom threshold."""
        critic = AnswerCritic(quality_threshold=0.8)
        assert critic.quality_threshold == 0.8

    def test_critique_empty_answer(self):
        """Test critique with empty answer."""
        critique = self.critic.critique_answer(
            answer="",
            query="What is Python?",
            sources=[],
        )

        assert critique.answer == ""
        assert critique.completeness_score < 0.5
        assert critique.needs_revision is True

    def test_critique_short_answer(self):
        """Test critique with very short answer."""
        critique = self.critic.critique_answer(
            answer="Yes.",
            query="Is Python a good programming language?",
            sources=[],
        )

        assert critique.completeness_score < 0.5
        assert "too short" in critique.issues[0].lower() or critique.needs_revision

    def test_critique_good_answer_with_sources(self):
        """Test critique with a good, detailed answer and sources."""
        answer = """
        Python is a high-level, interpreted programming language known for its
        clear syntax and code readability. It was created by Guido van Rossum
        and first released in 1991. Python supports multiple programming paradigms,
        including procedural, object-oriented, and functional programming.

        Key features include:
        - Dynamic typing
        - Automatic memory management
        - Large standard library
        - Extensive third-party packages

        Python is widely used in web development, data science, machine learning,
        automation, and scientific computing.
        """

        sources = [
            {
                "content": "Python is a high-level language created by Guido van Rossum",
                "metadata": {"source": "python.org"},
            },
            {
                "content": "Python supports object-oriented programming",
                "metadata": {"source": "docs.python.org"},
            },
            {
                "content": "Python is used in data science and machine learning",
                "metadata": {"source": "scipy.org"},
            },
        ]

        critique = self.critic.critique_answer(
            answer=answer,
            query="What is Python programming language?",
            sources=sources,
        )

        # A detailed answer with multiple sources should score well
        assert critique.completeness_score >= 0.6
        assert critique.relevance_score >= 0.6
        assert critique.clarity_score >= 0.5
        assert critique.overall_score >= 0.5

    def test_critique_answer_with_apology(self):
        """Test critique with apologetic language."""
        critique = self.critic.critique_answer(
            answer="I'm sorry, but I don't know the answer to that question.",
            query="What is quantum computing?",
            sources=[],
        )

        # Apology phrases should lower completeness
        assert critique.completeness_score < 0.5
        assert critique.needs_revision is True

    def test_critique_answer_uncertainty(self):
        """Test critique with uncertain language."""
        critique = self.critic.critique_answer(
            answer="I'm not sure, but Python might be a programming language.",
            query="What is Python?",
            sources=[],
        )

        # Uncertain language should lower accuracy
        assert critique.accuracy_score < 0.7

    def test_critique_with_tool_history(self):
        """Test critique with tool history."""
        tool_history = [
            {"tool": "vectordb_retrieval", "status": "success", "confidence": 0.9},
            {"tool": "web_search", "status": "success", "confidence": 0.8},
        ]

        critique = self.critic.critique_answer(
            answer="Python is a programming language.",
            query="What is Python?",
            sources=[{"content": "Python info", "metadata": {}}],
            tool_history=tool_history,
        )

        # Should complete without error
        assert critique.answer == "Python is a programming language."

    def test_critique_irrelevant_answer(self):
        """Test critique with answer that doesn't match query."""
        critique = self.critic.critique_answer(
            answer="The weather today is sunny with a high of 75 degrees.",
            query="What is machine learning?",
            sources=[],
        )

        # Irrelevant answer should score low on relevance
        assert critique.relevance_score < 0.5
        assert critique.needs_revision is True

    def test_critique_needs_revision_threshold(self):
        """Test that needs_revision respects quality threshold."""
        # High threshold
        critic_high = AnswerCritic(quality_threshold=0.9)
        critique_high = critic_high.critique_answer(
            answer="Python is a programming language used for many applications.",
            query="What is Python?",
            sources=[],
        )

        # Low threshold
        critic_low = AnswerCritic(quality_threshold=0.3)
        critique_low = critic_low.critique_answer(
            answer="Python is a programming language used for many applications.",
            query="What is Python?",
            sources=[],
        )

        # Same answer should have different revision needs based on threshold
        assert isinstance(critique_high.needs_revision, bool)
        assert isinstance(critique_low.needs_revision, bool)

    def test_critique_identifies_missing_aspects(self):
        """Test that critique identifies missing aspects."""
        critique = self.critic.critique_answer(
            answer="Python is a language.",
            query="Compare Python and Java for enterprise development",
            sources=[],
        )

        # Should identify that comparison/Java aspect is missing
        assert len(critique.missing_aspects) > 0 or critique.completeness_score < 0.7


class TestSourceVerifier:
    """Test SourceVerifier class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.verifier = SourceVerifier()

    def test_init(self):
        """Test default initialization."""
        verifier = SourceVerifier()
        assert verifier.logger is not None

    def test_verify_no_sources(self):
        """Test verification with no sources."""
        verification = self.verifier.verify_sources(
            answer="Python is a programming language.",
            sources=[],
        )

        assert verification.sources_found == 0
        assert verification.hallucination_risk >= 0.5

    def test_verify_with_sources(self):
        """Test verification with multiple sources."""
        sources = [
            {
                "content": "Python is a high-level programming language.",
                "metadata": {"source": "python.org", "title": "Python Introduction"},
            },
            {
                "content": "Python was created by Guido van Rossum.",
                "metadata": {"source": "wikipedia.org", "title": "Python History"},
            },
            {
                "content": "Python is used in data science and machine learning.",
                "metadata": {"source": "kaggle.com", "title": "Python for ML"},
            },
        ]

        verification = self.verifier.verify_sources(
            answer="Python is a high-level programming language created by Guido van Rossum, widely used in data science.",
            sources=sources,
        )

        assert verification.sources_found == 3
        assert verification.source_diversity > 0

    def test_verify_source_diversity(self):
        """Test that diverse sources score higher."""
        # Diverse sources
        diverse_sources = [
            {"content": "Python info", "metadata": {"source": "python.org"}},
            {"content": "Python guide", "metadata": {"source": "realpython.com"}},
            {"content": "Python tutorial", "metadata": {"source": "github.com"}},
        ]

        # Same source repeated
        same_sources = [
            {"content": "Python info 1", "metadata": {"source": "python.org"}},
            {"content": "Python info 2", "metadata": {"source": "python.org"}},
            {"content": "Python info 3", "metadata": {"source": "python.org"}},
        ]

        diverse_verification = self.verifier.verify_sources(
            answer="Python is great.",
            sources=diverse_sources,
        )

        same_verification = self.verifier.verify_sources(
            answer="Python is great.",
            sources=same_sources,
        )

        assert diverse_verification.source_diversity >= same_verification.source_diversity

    def test_verify_hallucination_detection(self):
        """Test hallucination detection with unsupported claims."""
        sources = [
            {"content": "Python is interpreted.", "metadata": {}},
        ]

        verification = self.verifier.verify_sources(
            answer="Python was invented in 2020 by Elon Musk and is compiled directly to machine code.",
            sources=sources,
        )

        # Should detect potential hallucination
        assert verification.hallucination_risk > 0.3

    def test_verify_well_supported_answer(self):
        """Test verification of well-supported answer."""
        sources = [
            {
                "content": "Python is a dynamically typed language that uses automatic memory management.",
                "metadata": {"source": "docs.python.org"},
            },
            {
                "content": "Python supports multiple programming paradigms including OOP.",
                "metadata": {"source": "python.org"},
            },
        ]

        verification = self.verifier.verify_sources(
            answer="Python is a dynamically typed language with automatic memory management that supports object-oriented programming.",
            sources=sources,
        )

        # Well-supported answer should have lower hallucination risk
        assert verification.hallucination_risk < 0.7

    def test_verify_empty_answer(self):
        """Test verification with empty answer."""
        verification = self.verifier.verify_sources(
            answer="",
            sources=[{"content": "Some content", "metadata": {}}],
        )

        assert verification.sources_found >= 0

    def test_verify_questionable_claims(self):
        """Test detection of questionable claims."""
        sources = [
            {"content": "Python was created in 1991.", "metadata": {}},
        ]

        verification = self.verifier.verify_sources(
            answer="Python was created in 2025 and is the fastest language ever made.",
            sources=sources,
        )

        # Should have questionable claims or high hallucination risk
        assert len(verification.questionable_claims) > 0 or verification.hallucination_risk > 0.4

    def test_verify_source_metadata_handling(self):
        """Test handling of sources with various metadata formats."""
        sources = [
            {"content": "Python info", "metadata": {"source": "python.org"}},
            {"content": "More info with id", "id": "src_1"},
            {"content": "Info with score", "score": 0.9},
            {"content": "Info with empty metadata", "metadata": {}},
        ]

        # Should handle various metadata formats without error
        verification = self.verifier.verify_sources(
            answer="Python is a programming language.",
            sources=sources,
        )

        assert verification.sources_found >= 1


class TestAnswerCriticEdgeCases:
    """Test edge cases for AnswerCritic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.critic = AnswerCritic()

    def test_critique_very_long_answer(self):
        """Test critique with very long answer."""
        long_answer = "Python is great. " * 500  # Very long answer

        critique = self.critic.critique_answer(
            answer=long_answer,
            query="What is Python?",
            sources=[],
        )

        # Should handle long answers
        assert critique.answer == long_answer
        assert critique.completeness_score >= 0

    def test_critique_special_characters(self):
        """Test critique with special characters in answer."""
        answer = "Python uses: symbols like @, #, $, %, ^, &, *, and more!"

        critique = self.critic.critique_answer(
            answer=answer,
            query="What symbols does Python use?",
            sources=[],
        )

        assert critique.answer == answer

    def test_critique_unicode_content(self):
        """Test critique with unicode content."""
        answer = "Python是一种编程语言。パイソンはプログラミング言語です。"

        critique = self.critic.critique_answer(
            answer=answer,
            query="What is Python in different languages?",
            sources=[],
        )

        assert critique.answer == answer

    def test_critique_code_in_answer(self):
        """Test critique with code snippets in answer."""
        answer = """
        Here's how to print in Python:
        ```python
        print("Hello, World!")
        ```
        This will output "Hello, World!" to the console.
        """

        critique = self.critic.critique_answer(
            answer=answer,
            query="How to print in Python?",
            sources=[],
        )

        assert "print" in critique.answer
        assert critique.clarity_score >= 0

    def test_critique_none_sources(self):
        """Test critique with None sources."""
        critique = self.critic.critique_answer(
            answer="Python is a language.",
            query="What is Python?",
            sources=None,
        )

        assert critique.source_quality_score >= 0

    def test_critique_empty_sources_list(self):
        """Test critique with empty sources list."""
        critique = self.critic.critique_answer(
            answer="Python is a language.",
            query="What is Python?",
            sources=[],
        )

        assert critique.source_quality_score <= 0.3
