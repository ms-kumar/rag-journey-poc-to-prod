"""Self-reflection and answer critique module for Agentic RAG."""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AnswerCritique:
    """Critique assessment of a generated answer.

    Attributes:
        answer: The answer being critiqued
        completeness_score: How complete the answer is (0.0 to 1.0)
        accuracy_score: Estimated accuracy based on sources (0.0 to 1.0)
        relevance_score: How relevant to the query (0.0 to 1.0)
        clarity_score: How clear and well-structured (0.0 to 1.0)
        source_quality_score: Quality of sources used (0.0 to 1.0)
        overall_score: Overall quality score (0.0 to 1.0)
        issues: List of identified issues
        suggestions: List of improvement suggestions
        needs_revision: Whether the answer needs to be revised
        missing_aspects: What's missing from the answer
        timestamp: When the critique was performed
    """

    answer: str
    completeness_score: float = 0.0
    accuracy_score: float = 0.0
    relevance_score: float = 0.0
    clarity_score: float = 0.0
    source_quality_score: float = 0.0
    overall_score: float = 0.0
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    needs_revision: bool = False
    missing_aspects: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SourceVerification:
    """Verification results for answer sources.

    Attributes:
        sources_found: Number of sources cited
        sources_verified: Number of sources verified as valid
        source_diversity: Diversity of source types (0.0 to 1.0)
        recency_score: How recent the sources are (0.0 to 1.0)
        authority_score: Authority/credibility of sources (0.0 to 1.0)
        citation_quality: Quality of citations (0.0 to 1.0)
        hallucination_risk: Estimated risk of hallucination (0.0 to 1.0)
        verified_sources: List of verified source references
        questionable_claims: Claims that lack source support
    """

    sources_found: int = 0
    sources_verified: int = 0
    source_diversity: float = 0.0
    recency_score: float = 0.0
    authority_score: float = 0.0
    citation_quality: float = 0.0
    hallucination_risk: float = 0.0
    verified_sources: list[dict[str, Any]] = field(default_factory=list)
    questionable_claims: list[str] = field(default_factory=list)


class AnswerCritic:
    """Critiques generated answers for quality and accuracy."""

    def __init__(self, llm: Any | None = None, quality_threshold: float = 0.7):
        """Initialize answer critic.

        Args:
            llm: Optional LLM for advanced critique
            quality_threshold: Minimum quality score to pass (0.0 to 1.0)
        """
        self.llm = llm
        self.quality_threshold = quality_threshold
        self.logger = logging.getLogger(__name__)

    def critique_answer(
        self,
        answer: str,
        query: str,
        sources: list[dict[str, Any]] | None = None,
        tool_history: list[dict[str, Any]] | None = None,
    ) -> AnswerCritique:
        """Critique an answer for quality.

        Args:
            answer: The answer to critique
            query: The original query
            sources: Optional list of sources used
            tool_history: Optional history of tool invocations

        Returns:
            AnswerCritique with scores and suggestions
        """
        self.logger.info("Critiquing answer...")

        critique = AnswerCritique(answer=answer)

        # 1. Completeness: Does it fully address the query?
        critique.completeness_score = self._assess_completeness(answer, query)

        # 2. Accuracy: Can we verify claims with sources?
        critique.accuracy_score = self._assess_accuracy(answer, sources)

        # 3. Relevance: Is it on-topic?
        critique.relevance_score = self._assess_relevance(answer, query)

        # 4. Clarity: Is it well-structured and clear?
        critique.clarity_score = self._assess_clarity(answer)

        # 5. Source quality: Are sources reliable?
        critique.source_quality_score = self._assess_source_quality(sources)

        # Calculate overall score (weighted average)
        critique.overall_score = (
            0.25 * critique.completeness_score
            + 0.25 * critique.accuracy_score
            + 0.20 * critique.relevance_score
            + 0.15 * critique.clarity_score
            + 0.15 * critique.source_quality_score
        )

        # Identify issues and suggestions
        critique.issues = self._identify_issues(critique, answer, query)
        critique.suggestions = self._generate_suggestions(critique, query)
        critique.missing_aspects = self._identify_missing_aspects(answer, query)

        # Decide if revision is needed
        critique.needs_revision = (
            critique.overall_score < self.quality_threshold or len(critique.issues) > 2
        )

        self.logger.info(
            f"Critique complete: overall={critique.overall_score:.2f}, "
            f"needs_revision={critique.needs_revision}"
        )

        return critique

    def _assess_completeness(self, answer: str, query: str) -> float:
        """Assess how complete the answer is.

        Args:
            answer: The answer text
            query: The query

        Returns:
            Completeness score (0.0 to 1.0)
        """
        if not answer or len(answer) < 50:
            return 0.2

        # Check for apology phrases
        apology_phrases = [
            "i apologize",
            "i'm sorry",
            "i couldn't",
            "i don't have",
            "no information",
        ]
        if any(phrase in answer.lower() for phrase in apology_phrases):
            return 0.3

        # Extract key terms from query
        query_terms = {word.lower() for word in re.findall(r"\b\w+\b", query) if len(word) > 3}

        # Check how many query terms appear in answer
        answer_terms = {word.lower() for word in re.findall(r"\b\w+\b", answer) if len(word) > 3}

        if not query_terms:
            return 0.7

        term_coverage = len(query_terms & answer_terms) / len(query_terms)

        # Length bonus (longer answers tend to be more complete)
        length_score = min(len(answer) / 500, 1.0)

        return 0.6 * term_coverage + 0.4 * length_score

    def _assess_accuracy(self, answer: str, sources: list[dict[str, Any]] | None) -> float:
        """Assess accuracy based on source support.

        Args:
            answer: The answer text
            sources: Sources used

        Returns:
            Accuracy score (0.0 to 1.0)
        """
        if not sources:
            # No sources = lower confidence in accuracy
            return 0.5

        # Check for specific numbers, dates, names that should be verifiable
        specific_claims = len(re.findall(r"\b\d+\b|\b\d{4}\b", answer))

        if specific_claims == 0:
            # General answer without specifics
            return 0.7 if sources else 0.5

        # Higher source count increases confidence
        source_count = len(sources)
        source_score = min(source_count / 5, 1.0)

        return 0.6 + 0.4 * source_score

    def _assess_relevance(self, answer: str, query: str) -> float:
        """Assess how relevant the answer is to the query.

        Args:
            answer: The answer text
            query: The query

        Returns:
            Relevance score (0.0 to 1.0)
        """
        # Extract key concepts from query
        query_lower = query.lower()
        answer_lower = answer.lower()

        # Check for direct query terms in answer
        query_words = {word for word in re.findall(r"\b\w+\b", query_lower) if len(word) > 3}

        if not query_words:
            return 0.7

        # Count matches
        matches = sum(1 for word in query_words if word in answer_lower)
        relevance = matches / len(query_words)

        # Check if answer starts with relevant content
        answer_start = answer_lower[:200]
        start_relevance = 0.2 if any(word in answer_start for word in query_words) else 0.0

        return min(relevance + start_relevance, 1.0)

    def _assess_clarity(self, answer: str) -> float:
        """Assess clarity and structure of answer.

        Args:
            answer: The answer text

        Returns:
            Clarity score (0.0 to 1.0)
        """
        if not answer:
            return 0.0

        score = 0.5  # Base score

        # Check for good structure (paragraphs, lists)
        paragraphs = answer.split("\n\n")
        if len(paragraphs) > 1:
            score += 0.1

        # Check for lists/bullets
        if re.search(r"^\s*[-*•]\s", answer, re.MULTILINE):
            score += 0.1

        # Check for numbered items
        if re.search(r"^\s*\d+[\.)]\s", answer, re.MULTILINE):
            score += 0.1

        # Penalize very long sentences
        sentences = re.split(r"[.!?]+", answer)
        avg_sentence_length = (
            sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        )
        if avg_sentence_length > 40:
            score -= 0.1

        # Reward moderate length
        if 200 <= len(answer) <= 1500:
            score += 0.1

        return min(max(score, 0.0), 1.0)

    def _assess_source_quality(self, sources: list[dict[str, Any]] | None) -> float:
        """Assess quality of sources.

        Args:
            sources: List of sources

        Returns:
            Source quality score (0.0 to 1.0)
        """
        if not sources:
            return 0.3

        # Source diversity (different source types)
        source_types = {source.get("type", "unknown") for source in sources}
        diversity = min(len(source_types) / 3, 1.0)

        # Source count (more sources = better)
        count_score = min(len(sources) / 5, 1.0)

        # Check for metadata (well-structured sources)
        with_metadata = sum(
            1 for source in sources if source.get("metadata") or source.get("score")
        )
        metadata_score = with_metadata / len(sources) if sources else 0.0

        return 0.3 * diversity + 0.3 * count_score + 0.4 * metadata_score

    def _identify_issues(self, critique: AnswerCritique, answer: str, query: str) -> list[str]:
        """Identify specific issues with the answer.

        Args:
            critique: The critique object
            answer: The answer text
            query: The query

        Returns:
            List of identified issues
        """
        issues = []

        if critique.completeness_score < 0.5:
            issues.append("Answer is incomplete or too brief")

        if critique.accuracy_score < 0.6:
            issues.append("Answer lacks sufficient source support")

        if critique.relevance_score < 0.6:
            issues.append("Answer may not fully address the query")

        if critique.clarity_score < 0.5:
            issues.append("Answer structure could be improved")

        if len(answer) < 100:
            issues.append("Answer is too short")

        if "i don't know" in answer.lower() or "i'm not sure" in answer.lower():
            issues.append("Answer expresses uncertainty")

        return issues

    def _generate_suggestions(self, critique: AnswerCritique, query: str) -> list[str]:
        """Generate improvement suggestions.

        Args:
            critique: The critique object
            query: The original query

        Returns:
            List of suggestions
        """
        suggestions = []

        if critique.completeness_score < 0.6:
            suggestions.append("Include more details and context")

        if critique.source_quality_score < 0.6:
            suggestions.append("Gather more diverse and reliable sources")

        if critique.clarity_score < 0.6:
            suggestions.append("Improve answer structure with paragraphs or lists")

        if critique.accuracy_score < 0.7:
            suggestions.append("Verify claims with additional sources")

        return suggestions

    def _identify_missing_aspects(self, answer: str, query: str) -> list[str]:
        """Identify aspects of the query not addressed in the answer.

        Args:
            answer: The answer text
            query: The query

        Returns:
            List of missing aspects
        """
        missing = []

        # Check for question words not addressed
        question_patterns = {
            "what": "definition or explanation",
            "why": "reasoning or cause",
            "how": "process or method",
            "when": "time or date",
            "where": "location",
            "who": "person or entity",
        }

        query_lower = query.lower()
        for word, aspect in question_patterns.items():
            if word in query_lower and word not in answer.lower()[:300]:
                missing.append(f"Missing {aspect}")

        return missing


class SourceVerifier:
    """Verifies sources and detects potential hallucinations."""

    def __init__(self):
        """Initialize source verifier."""
        self.logger = logging.getLogger(__name__)

    def verify_sources(
        self,
        answer: str,
        sources: list[dict[str, Any]] | None = None,
        tool_history: list[dict[str, Any]] | None = None,
    ) -> SourceVerification:
        """Verify sources used in answer.

        Args:
            answer: The generated answer
            sources: List of sources
            tool_history: Tool execution history

        Returns:
            SourceVerification result
        """
        self.logger.info("Verifying sources...")

        verification = SourceVerification()

        if not sources:
            verification.hallucination_risk = 0.7
            return verification

        verification.sources_found = len(sources)

        # Verify each source
        for source in sources:
            if self._is_source_valid(source):
                verification.sources_verified += 1
                verification.verified_sources.append(
                    {
                        "id": source.get("id", "unknown"),
                        "content": source.get("content", "")[:200],
                        "score": source.get("score", 0.0),
                    }
                )

        # Calculate metrics
        verification.source_diversity = self._calculate_diversity(sources)
        verification.recency_score = self._calculate_recency(sources)
        verification.authority_score = self._calculate_authority(sources)
        verification.citation_quality = (
            verification.sources_verified / verification.sources_found
            if verification.sources_found > 0
            else 0.0
        )

        # Estimate hallucination risk
        verification.hallucination_risk = self._estimate_hallucination_risk(
            answer, sources, verification
        )

        # Identify questionable claims
        verification.questionable_claims = self._find_questionable_claims(answer, sources)

        self.logger.info(
            f"Source verification: {verification.sources_verified}/"
            f"{verification.sources_found} verified, "
            f"hallucination_risk={verification.hallucination_risk:.2f}"
        )

        return verification

    def _is_source_valid(self, source: dict[str, Any]) -> bool:
        """Check if a source is valid.

        Args:
            source: Source dictionary

        Returns:
            True if valid
        """
        # Must have content
        if not source.get("content"):
            return False

        # Must have reasonable length
        if len(source.get("content", "")) < 20:
            return False

        # Check for metadata
        has_metadata = bool(
            source.get("metadata")
            or source.get("source")
            or source.get("id")
            or source.get("score")
        )

        return has_metadata

    def _calculate_diversity(self, sources: list[dict[str, Any]]) -> float:
        """Calculate source diversity.

        Args:
            sources: List of sources

        Returns:
            Diversity score (0.0 to 1.0)
        """
        if not sources:
            return 0.0

        # Check for different source types
        source_types = set()
        for source in sources:
            source_type = (
                source.get("type") or source.get("metadata", {}).get("source") or "document"
            )
            source_types.add(source_type)

        return min(len(source_types) / 3, 1.0)

    def _calculate_recency(self, sources: list[dict[str, Any]]) -> float:
        """Calculate recency score of sources.

        Args:
            sources: List of sources

        Returns:
            Recency score (0.0 to 1.0)
        """
        # For now, assume moderate recency
        # In production, check timestamps in metadata
        return 0.7

    def _calculate_authority(self, sources: list[dict[str, Any]]) -> float:
        """Calculate authority score of sources.

        Args:
            sources: List of sources

        Returns:
            Authority score (0.0 to 1.0)
        """
        if not sources:
            return 0.5

        # Check for high-quality indicators
        authority_indicators = 0
        for source in sources:
            metadata = source.get("metadata", {})

            # Has source attribution
            if metadata.get("source"):
                authority_indicators += 1

            # Has author
            if metadata.get("author"):
                authority_indicators += 1

            # Has high relevance score
            if source.get("score", 0) > 0.8:
                authority_indicators += 1

        max_possible = len(sources) * 3
        return authority_indicators / max_possible if max_possible > 0 else 0.5

    def _estimate_hallucination_risk(
        self,
        answer: str,
        sources: list[dict[str, Any]],
        verification: SourceVerification,
    ) -> float:
        """Estimate risk of hallucination.

        Args:
            answer: The answer
            sources: List of sources
            verification: Verification results

        Returns:
            Hallucination risk (0.0 to 1.0)
        """
        risk = 0.3  # Base risk

        # No sources = higher risk
        if not sources:
            return 0.8

        # Few verified sources = higher risk
        if verification.sources_verified < 2:
            risk += 0.2

        # Low citation quality = higher risk
        if verification.citation_quality < 0.5:
            risk += 0.2

        # Specific claims without source support = higher risk
        specific_claims = len(re.findall(r"\b\d+\b|\b\d{4}\b", answer))
        if specific_claims > 3 and len(sources) < 3:
            risk += 0.15

        return min(risk, 1.0)

    def _find_questionable_claims(self, answer: str, sources: list[dict[str, Any]]) -> list[str]:
        """Find claims that may lack source support.

        Args:
            answer: The answer
            sources: List of sources

        Returns:
            List of questionable claims
        """
        questionable = []

        # Extract sentences with specific claims
        sentences = re.split(r"[.!?]+", answer)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Look for specific numbers or dates
            has_specifics = bool(re.search(r"\b\d+\b|\b\d{4}\b", sentence))

            if has_specifics and len(sentence) > 20:
                # Check if this claim appears in any source
                found_support = False
                for source in sources:
                    content = source.get("content", "").lower()
                    if any(word in content for word in sentence.lower().split()[:5]):
                        found_support = True
                        break

                if not found_support:
                    questionable.append(sentence[:100])

        return questionable[:5]  # Limit to top 5
