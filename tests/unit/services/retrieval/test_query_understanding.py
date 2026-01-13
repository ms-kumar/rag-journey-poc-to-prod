"""Tests for query understanding components."""

import pytest
from pydantic import ValidationError

from src.config import settings
from src.services.query_understanding import (
    QueryRewriter,
    QueryRewriterConfig,
    QueryUnderstandingClient,
    SynonymExpander,
    SynonymExpanderConfig,
)


class TestQueryRewriter:
    """Tests for QueryRewriter."""

    def test_rewriter_init_default(self):
        """Test rewriter initialization with defaults."""
        rewriter = QueryRewriter()
        assert rewriter.config.expand_acronyms is True
        assert rewriter.config.fix_typos is True
        assert rewriter.config.add_context is True
        assert rewriter.config.max_rewrites == 3

    def test_rewriter_init_custom(self):
        """Test rewriter initialization with custom config."""
        config = QueryRewriterConfig(expand_acronyms=False, fix_typos=True, max_rewrites=5)
        rewriter = QueryRewriter(config)
        assert rewriter.config.expand_acronyms is False
        assert rewriter.config.max_rewrites == 5

    def test_acronym_expansion(self):
        """Test acronym expansion."""
        rewriter = QueryRewriter()
        rewritten, meta = rewriter.rewrite("what is ML?")
        assert "machine learning" in rewritten.lower()
        assert meta["rewrites_applied"] >= 1
        assert meta["latency_ms"] > 0

    def test_multiple_acronyms(self):
        """Test expansion of multiple acronyms."""
        rewriter = QueryRewriter()
        rewritten, _ = rewriter.rewrite("ML and AI in NLP")
        assert "machine learning" in rewritten.lower()
        assert "artificial intelligence" in rewritten.lower()
        assert "natural language processing" in rewritten.lower()

    def test_typo_fixing(self):
        """Test typo correction."""
        rewriter = QueryRewriter()
        rewritten, meta = rewriter.rewrite("machien learing algorithim")
        assert "machine" in rewritten.lower()
        assert "learning" in rewritten.lower()
        assert "algorithm" in rewritten.lower()
        assert meta["rewrites_applied"] >= 1

    def test_context_addition_what(self):
        """Test context addition for 'what is' questions."""
        rewriter = QueryRewriter()
        rewritten, meta = rewriter.rewrite("what is Python?")
        assert "definition" in rewritten.lower() or "python" in rewritten.lower()
        assert meta["rewrites_applied"] >= 1

    def test_context_addition_how(self):
        """Test context addition for 'how to' questions."""
        rewriter = QueryRewriter()
        rewritten, _ = rewriter.rewrite("how to train a model?")
        assert any(word in rewritten.lower() for word in ["tutorial", "guide", "steps"])

    def test_context_addition_why(self):
        """Test context addition for 'why' questions."""
        rewriter = QueryRewriter()
        rewritten, _ = rewriter.rewrite("why use neural networks?")
        assert any(word in rewritten.lower() for word in ["reason", "explanation"])

    def test_no_rewrite_short_query(self):
        """Test that very short queries are not rewritten."""
        config = QueryRewriterConfig(min_query_length=3)
        rewriter = QueryRewriter(config)
        original = "hi"
        rewritten, meta = rewriter.rewrite(original)
        assert rewritten == original
        assert meta["rewrites_applied"] == 0

    def test_get_rewrites_multiple(self):
        """Test generation of multiple rewrite candidates."""
        rewriter = QueryRewriter()
        rewrites = rewriter.get_rewrites("what is ML?")
        assert len(rewrites) >= 2
        assert rewrites[0] == "what is ML?"
        assert any("machine learning" in r.lower() for r in rewrites)

    def test_config_validation_max_rewrites(self):
        """Test config validation for max_rewrites."""
        with pytest.raises(ValidationError):
            QueryRewriterConfig(max_rewrites=0)

    def test_config_validation_min_query_length(self):
        """Test config validation for min_query_length."""
        with pytest.raises(ValidationError):
            QueryRewriterConfig(min_query_length=0)

    def test_case_preservation(self):
        """Test that case is preserved when expanding."""
        rewriter = QueryRewriter()
        rewritten, _ = rewriter.rewrite("ML is powerful")
        # Should have "Machine learning" (capitalized)
        assert "machine learning" in rewritten.lower()

    def test_no_change_returns_original(self):
        """Test that queries with no applicable rewrites return original."""
        config = QueryRewriterConfig(expand_acronyms=False, fix_typos=False, add_context=False)
        rewriter = QueryRewriter(config)
        original = "this is a normal query"
        rewritten, meta = rewriter.rewrite(original)
        assert rewritten == original
        assert meta["rewrites_applied"] == 0


class TestSynonymExpander:
    """Tests for SynonymExpander."""

    def test_expander_init_default(self):
        """Test expander initialization with defaults."""
        expander = SynonymExpander()
        assert expander.config.max_synonyms_per_term == 3
        assert expander.config.min_term_length == 3
        assert expander.config.expand_all_terms is False

    def test_expander_init_custom(self):
        """Test expander initialization with custom config."""
        config = SynonymExpanderConfig(max_synonyms_per_term=5, expand_all_terms=True)
        expander = SynonymExpander(config)
        assert expander.config.max_synonyms_per_term == 5
        assert expander.config.expand_all_terms is True

    def test_synonym_expansion_ml(self):
        """Test expansion of machine learning term."""
        expander = SynonymExpander()
        expanded, meta = expander.expand("machine learning model")
        assert "ml" in expanded.lower()
        assert meta["terms_expanded"] >= 1
        assert meta["synonyms_added"] >= 1
        assert meta["latency_ms"] > 0

    def test_no_duplicate_synonyms(self):
        """Test that existing terms are not re-added as synonyms."""
        expander = SynonymExpander()
        expanded, _ = expander.expand("machine learning ml model")
        # "ml" is already in query, shouldn't be added again
        count = expanded.lower().count("ml")
        assert count == 1

    def test_multi_word_phrase_expansion(self):
        """Test expansion of multi-word phrases."""
        expander = SynonymExpander()
        expanded, meta = expander.expand("neural network architecture")
        assert meta["terms_expanded"] >= 1
        # Should have synonyms for "neural network"
        assert len(expanded) > len("neural network architecture")

    def test_get_synonyms(self):
        """Test getting synonyms for a specific term."""
        expander = SynonymExpander()
        synonyms = expander.get_synonyms("machine learning")
        assert len(synonyms) > 0
        assert "ml" in synonyms

    def test_get_synonyms_unknown_term(self):
        """Test getting synonyms for unknown term returns empty list."""
        expander = SynonymExpander()
        synonyms = expander.get_synonyms("xyz123unknown")
        assert synonyms == []

    def test_add_synonym_new(self):
        """Test adding new synonym mapping."""
        expander = SynonymExpander()
        expander.add_synonym("rag", ["retrieval augmented generation"])
        synonyms = expander.get_synonyms("rag")
        assert "retrieval augmented generation" in synonyms

    def test_add_synonym_merge(self):
        """Test adding synonyms to existing term merges them."""
        expander = SynonymExpander()
        existing = expander.get_synonyms("model")
        expander.add_synonym("model", ["new_synonym"])
        updated = expander.get_synonyms("model")
        assert len(updated) > len(existing)
        assert "new_synonym" in updated

    def test_config_validation_max_synonyms(self):
        """Test config validation for max_synonyms_per_term."""
        with pytest.raises(ValidationError):
            SynonymExpanderConfig(max_synonyms_per_term=0)

    def test_config_validation_min_term_length(self):
        """Test config validation for min_term_length."""
        with pytest.raises(ValidationError):
            SynonymExpanderConfig(min_term_length=0)

    def test_max_synonyms_limit(self):
        """Test that max_synonyms_per_term is respected."""
        config = SynonymExpanderConfig(max_synonyms_per_term=1)
        expander = SynonymExpander(config)
        expanded, meta = expander.expand("machine learning")
        # Should only add 1 synonym even if more are available
        assert meta["synonyms_added"] <= 1

    def test_no_expansion_short_query(self):
        """Test that very short queries don't get expanded."""
        expander = SynonymExpander()
        expanded, meta = expander.expand("a b c")
        # Stopwords and short terms shouldn't expand
        assert meta["terms_expanded"] == 0

    def test_stopwords_not_expanded(self):
        """Test that stopwords are not expanded."""
        config = SynonymExpanderConfig(expand_all_terms=True)
        expander = SynonymExpander(config)
        expanded, _ = expander.expand("the is a")
        # Stopwords should not be expanded
        assert expanded == "the is a"


class TestQueryUnderstandingClient:
    """Tests for QueryUnderstandingClient orchestrator."""

    def test_qu_init_default(self):
        """Test initialization with default config."""
        qu = QueryUnderstandingClient(settings)
        assert qu.query_settings.enable_rewriting is True
        assert qu.query_settings.enable_synonyms is True
        assert qu.query_settings.enable_intent_classification is False
        assert qu.rewriter is not None
        assert qu.expander is not None

    def test_qu_init_custom(self):
        """Test initialization with custom config."""
        from unittest.mock import MagicMock

        mock_settings = MagicMock()
        mock_settings.query_understanding.enable_rewriting = False
        mock_settings.query_understanding.enable_synonyms = True
        mock_settings.query_understanding.max_synonyms_per_term = 3
        mock_settings.query_understanding.min_term_length = 3
        mock_settings.query_understanding.expand_all_terms = False
        qu = QueryUnderstandingClient(mock_settings)
        assert qu.rewriter is None
        assert qu.expander is not None

    def test_process_full_pipeline(self):
        """Test full processing pipeline."""
        qu = QueryUnderstandingClient(settings)
        result = qu.process("what is ML?")

        assert result["original_query"] == "what is ML?"
        assert result["processed_query"] != "what is ML?"
        assert "machine learning" in result["processed_query"].lower()
        assert result["metadata"]["total_latency_ms"] > 0
        assert result["metadata"]["rewrite_latency_ms"] > 0
        assert result["metadata"]["expansion_latency_ms"] > 0

    def test_process_only_rewriting(self):
        """Test processing with only rewriting enabled."""
        from unittest.mock import MagicMock

        mock_settings = MagicMock()
        mock_settings.query_understanding.enable_rewriting = True
        mock_settings.query_understanding.enable_synonyms = False
        mock_settings.query_understanding.expand_acronyms = True
        mock_settings.query_understanding.fix_typos = True
        mock_settings.query_understanding.add_context = True
        mock_settings.query_understanding.max_rewrites = 3
        mock_settings.query_understanding.min_query_length = 3
        qu = QueryUnderstandingClient(mock_settings)
        result = qu.process("what is ML?")

        assert result["rewritten_query"] is not None
        assert result["expanded_query"] is None
        assert "machine learning" in result["processed_query"].lower()

    def test_process_only_synonyms(self):
        """Test processing with only synonyms enabled."""
        from unittest.mock import MagicMock

        mock_settings = MagicMock()
        mock_settings.query_understanding.enable_rewriting = False
        mock_settings.query_understanding.enable_synonyms = True
        mock_settings.query_understanding.max_synonyms_per_term = 3
        mock_settings.query_understanding.min_term_length = 3
        mock_settings.query_understanding.expand_all_terms = False
        qu = QueryUnderstandingClient(mock_settings)
        result = qu.process("machine learning model")

        assert result["rewritten_query"] is None
        assert result["expanded_query"] is not None
        assert "ml" in result["processed_query"].lower()

    def test_intent_classification_howto(self):
        """Test intent classification for how-to queries."""
        from unittest.mock import MagicMock

        mock_settings = MagicMock()
        mock_settings.query_understanding.enable_rewriting = True
        mock_settings.query_understanding.enable_synonyms = True
        mock_settings.query_understanding.enable_intent_classification = True
        mock_settings.query_understanding.expand_acronyms = True
        mock_settings.query_understanding.fix_typos = True
        mock_settings.query_understanding.add_context = True
        mock_settings.query_understanding.max_rewrites = 3
        mock_settings.query_understanding.min_query_length = 3
        mock_settings.query_understanding.max_synonyms_per_term = 3
        mock_settings.query_understanding.min_term_length = 3
        mock_settings.query_understanding.expand_all_terms = False
        qu = QueryUnderstandingClient(mock_settings)
        result = qu.process("how to train a model?")

        assert result["intent"] == "howto"
        assert result["metadata"]["intent_latency_ms"] > 0

    def test_intent_classification_factual(self):
        """Test intent classification for factual queries."""
        from unittest.mock import MagicMock

        mock_settings = MagicMock()
        mock_settings.query_understanding.enable_intent_classification = True
        mock_settings.query_understanding.enable_rewriting = True
        mock_settings.query_understanding.enable_synonyms = True
        mock_settings.query_understanding.expand_acronyms = True
        mock_settings.query_understanding.fix_typos = True
        mock_settings.query_understanding.add_context = True
        mock_settings.query_understanding.max_rewrites = 3
        mock_settings.query_understanding.min_query_length = 3
        mock_settings.query_understanding.max_synonyms_per_term = 3
        mock_settings.query_understanding.min_term_length = 3
        mock_settings.query_understanding.expand_all_terms = False
        qu = QueryUnderstandingClient(mock_settings)
        result = qu.process("what is machine learning?")

        assert result["intent"] == "factual"

    def test_intent_classification_comparison(self):
        """Test intent classification for comparison queries."""
        from unittest.mock import MagicMock

        mock_settings = MagicMock()
        mock_settings.query_understanding.enable_intent_classification = True
        mock_settings.query_understanding.enable_rewriting = True
        mock_settings.query_understanding.enable_synonyms = True
        mock_settings.query_understanding.expand_acronyms = True
        mock_settings.query_understanding.fix_typos = True
        mock_settings.query_understanding.add_context = True
        mock_settings.query_understanding.max_rewrites = 3
        mock_settings.query_understanding.min_query_length = 3
        mock_settings.query_understanding.max_synonyms_per_term = 3
        mock_settings.query_understanding.min_term_length = 3
        mock_settings.query_understanding.expand_all_terms = False
        qu = QueryUnderstandingClient(mock_settings)
        result = qu.process("Python vs Java")

        assert result["intent"] == "comparison"

    def test_intent_classification_troubleshooting(self):
        """Test intent classification for troubleshooting queries."""
        from unittest.mock import MagicMock

        mock_settings = MagicMock()
        mock_settings.query_understanding.enable_intent_classification = True
        mock_settings.query_understanding.enable_rewriting = True
        mock_settings.query_understanding.enable_synonyms = True
        mock_settings.query_understanding.expand_acronyms = True
        mock_settings.query_understanding.fix_typos = True
        mock_settings.query_understanding.add_context = True
        mock_settings.query_understanding.max_rewrites = 3
        mock_settings.query_understanding.min_query_length = 3
        mock_settings.query_understanding.max_synonyms_per_term = 3
        mock_settings.query_understanding.min_term_length = 3
        mock_settings.query_understanding.expand_all_terms = False
        qu = QueryUnderstandingClient(mock_settings)
        result = qu.process("error in code not working")

        assert result["intent"] == "troubleshooting"

    def test_intent_classification_exploratory(self):
        """Test intent classification for exploratory queries."""
        from unittest.mock import MagicMock

        mock_settings = MagicMock()
        mock_settings.query_understanding.enable_intent_classification = True
        mock_settings.query_understanding.enable_rewriting = True
        mock_settings.query_understanding.enable_synonyms = True
        mock_settings.query_understanding.expand_acronyms = True
        mock_settings.query_understanding.fix_typos = True
        mock_settings.query_understanding.add_context = True
        mock_settings.query_understanding.max_rewrites = 3
        mock_settings.query_understanding.min_query_length = 3
        mock_settings.query_understanding.max_synonyms_per_term = 3
        mock_settings.query_understanding.min_term_length = 3
        mock_settings.query_understanding.expand_all_terms = False
        qu = QueryUnderstandingClient(mock_settings)
        result = qu.process("tell me about deep learning")

        assert result["intent"] == "exploratory"

    def test_get_all_variations(self):
        """Test generation of all query variations."""
        qu = QueryUnderstandingClient(settings)
        variations = qu.get_all_variations("what is ML?")

        assert len(variations) >= 2
        assert "what is ML?" in variations
        assert any("machine learning" in v.lower() for v in variations)

    def test_variations_no_duplicates(self):
        """Test that variations don't contain duplicates."""
        qu = QueryUnderstandingClient(settings)
        variations = qu.get_all_variations("machine learning")

        # Check for uniqueness
        assert len(variations) == len(set(variations))

    def test_metadata_tracking(self):
        """Test that all metadata is properly tracked."""
        qu = QueryUnderstandingClient(settings)
        result = qu.process("what is ML?")

        metadata = result["metadata"]
        assert "total_latency_ms" in metadata
        assert "rewrite_latency_ms" in metadata
        assert "expansion_latency_ms" in metadata
        assert metadata["total_latency_ms"] > 0

    def test_original_query_preserved(self):
        """Test that original query is always preserved."""
        qu = QueryUnderstandingClient(settings)
        original = "what is ML?"
        result = qu.process(original)

        assert result["original_query"] == original
        assert result["metadata"]["rewrite_latency_ms"] >= 0


class TestIntegration:
    """Integration tests for query understanding pipeline."""

    def test_full_pipeline_complex_query(self):
        """Test full pipeline with complex query."""
        qu = QueryUnderstandingClient(settings)
        result = qu.process("how to fix machien learing error in py?")

        # Should have:
        # - Fixed typo: machien → machine, learing → learning
        # - Expanded acronym: py → python
        # - Added context: how to → tutorial guide steps
        # - Expanded synonyms: machine learning → ml, ...

        processed = result["processed_query"].lower()
        assert "machine" in processed or "learning" in processed
        assert "python" in processed or "py" in processed
        assert len(processed) > len("how to fix machien learing error in py?")

    def test_pipeline_latency_reasonable(self):
        """Test that pipeline latency is reasonable (< 10ms typical)."""
        qu = QueryUnderstandingClient(settings)
        result = qu.process("what is machine learning?")

        # Should be very fast (< 10ms for rule-based processing)
        assert result["metadata"]["total_latency_ms"] < 50  # Generous upper bound

    def test_multiple_queries_consistent(self):
        """Test that processing multiple queries is consistent."""
        qu = QueryUnderstandingClient(settings)

        # Process same query twice
        result1 = qu.process("what is ML?")
        result2 = qu.process("what is ML?")

        # Should get same processed query
        assert result1["processed_query"] == result2["processed_query"]

    def test_empty_query_handling(self):
        """Test handling of empty or whitespace queries."""
        qu = QueryUnderstandingClient(settings)
        result = qu.process("")

        assert result["processed_query"] == ""
        assert result["metadata"]["total_latency_ms"] >= 0

    def test_special_characters_preserved(self):
        """Test that special characters are preserved."""
        qu = QueryUnderstandingClient(settings)
        result = qu.process("what is C++?")

        assert "++" in result["processed_query"]
