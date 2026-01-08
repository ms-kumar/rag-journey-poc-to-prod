"""
Synonym expansion for query enrichment.

Expands queries with synonyms to improve recall by matching
semantically similar terms.
"""

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SynonymExpanderConfig:
    """Configuration for synonym expander."""

    max_synonyms_per_term: int = 3
    min_term_length: int = 3
    expand_all_terms: bool = False  # If False, only expand key terms

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_synonyms_per_term < 1:
            raise ValueError("max_synonyms_per_term must be at least 1")
        if self.min_term_length < 1:
            raise ValueError("min_term_length must be at least 1")


class SynonymExpander:
    """
    Expand queries with synonyms for better recall.

    Uses a predefined synonym dictionary to expand key terms
    in the query, improving chances of matching relevant documents.
    """

    def __init__(self, config: SynonymExpanderConfig | None = None):
        """
        Initialize synonym expander.

        Args:
            config: Expander configuration
        """
        self.config = config or SynonymExpanderConfig()
        self._synonym_dict = self._build_synonym_dict()
        self._stopwords = self._build_stopwords()

    def _build_stopwords(self) -> set[str]:
        """Build set of stopwords to skip."""
        return {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "could",
            "may",
            "might",
            "can",
            "of",
            "to",
            "for",
            "in",
            "on",
            "at",
            "by",
            "with",
            "from",
            "as",
            "this",
            "that",
            "these",
            "those",
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "how",
        }

    def _build_synonym_dict(self) -> dict[str, list[str]]:
        """Build synonym dictionary for common terms."""
        return {
            # ML/AI terms
            "machine learning": ["ml", "statistical learning", "predictive modeling"],
            "deep learning": ["dl", "neural networks", "deep neural nets"],
            "artificial intelligence": ["ai", "machine intelligence", "cognitive computing"],
            "neural network": ["nn", "neural net", "artificial neural network"],
            "model": ["algorithm", "predictor", "classifier"],
            "training": ["learning", "fitting", "optimization"],
            "prediction": ["inference", "forecast", "estimation"],
            "accuracy": ["precision", "performance", "correctness"],
            "dataset": ["data", "corpus", "collection"],
            "features": ["attributes", "variables", "inputs"],
            "labels": ["targets", "outputs", "ground truth"],
            # NLP terms
            "embedding": ["vector", "representation", "encoding"],
            "token": ["word", "term", "symbol"],
            "tokenization": ["segmentation", "splitting", "parsing"],
            "semantic": ["meaning", "contextual", "conceptual"],
            "similarity": ["resemblance", "closeness", "relatedness"],
            "retrieval": ["search", "lookup", "fetching"],
            "ranking": ["ordering", "sorting", "scoring"],
            "relevance": ["pertinence", "applicability", "suitability"],
            # Programming terms
            "function": ["method", "procedure", "routine"],
            "variable": ["parameter", "attribute", "field"],
            "error": ["exception", "bug", "fault"],
            "debug": ["troubleshoot", "diagnose", "fix"],
            "optimize": ["improve", "enhance", "tune"],
            "implement": ["code", "develop", "create"],
            "deploy": ["release", "launch", "publish"],
            "test": ["verify", "validate", "check"],
            "performance": ["speed", "efficiency", "throughput"],
            "latency": ["delay", "response time", "lag"],
            # Database terms
            "database": ["db", "datastore", "data storage"],
            "query": ["search", "request", "lookup"],
            "index": ["indexing", "catalog", "directory"],
            "table": ["relation", "entity", "dataset"],
            "record": ["row", "entry", "item"],
            "field": ["column", "attribute", "property"],
            # General terms
            "fast": ["quick", "rapid", "speedy"],
            "slow": ["sluggish", "delayed", "lagging"],
            "large": ["big", "huge", "massive"],
            "small": ["tiny", "little", "compact"],
            "efficient": ["optimized", "streamlined", "effective"],
            "simple": ["easy", "straightforward", "basic"],
            "complex": ["complicated", "intricate", "sophisticated"],
            "improve": ["enhance", "boost", "upgrade"],
            "reduce": ["decrease", "minimize", "lower"],
            "increase": ["boost", "raise", "augment"],
        }

    def expand(self, query: str) -> tuple[str, dict[str, object]]:
        """
        Expand query with synonyms.

        Args:
            query: Original query string

        Returns:
            Tuple of (expanded_query, metadata with latency and expansion info)

        Example:
            >>> expander = SynonymExpander()
            >>> expanded, meta = expander.expand("machine learning model")
            >>> print(expanded)
            "machine learning ml statistical learning model algorithm predictor"
            >>> print(meta["terms_expanded"])
            2
        """
        start_time = time.perf_counter()

        query_lower = query.lower()
        words = query_lower.split()

        # Track what we've expanded
        expanded_terms: list[str] = []
        terms_expanded = 0

        # Try multi-word phrases first (longer matches)
        for phrase, synonyms in sorted(
            self._synonym_dict.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if phrase in query_lower:
                # Add synonyms for this phrase
                for syn in synonyms[: self.config.max_synonyms_per_term]:
                    if syn not in expanded_terms and syn not in query_lower:
                        expanded_terms.append(syn)
                terms_expanded += 1

        # Then try single words (if expand_all_terms is True)
        if self.config.expand_all_terms:
            for word in words:
                if len(word) < self.config.min_term_length:
                    continue
                if word in self._stopwords:
                    continue
                if word in self._synonym_dict:
                    for syn in self._synonym_dict[word][: self.config.max_synonyms_per_term]:
                        if syn not in expanded_terms and syn not in query_lower:
                            expanded_terms.append(syn)
                    terms_expanded += 1

        # Build expanded query
        if expanded_terms:
            expanded_query = f"{query} {' '.join(expanded_terms)}"
        else:
            expanded_query = query

        latency = (time.perf_counter() - start_time) * 1000

        metadata = {
            "latency_ms": latency,
            "terms_expanded": terms_expanded,
            "synonyms_added": len(expanded_terms),
            "original_query": query,
        }

        if expanded_terms:
            logger.info(
                f"Query expanded: '{query}' → added {len(expanded_terms)} synonyms "
                f"({latency:.2f}ms)"
            )

        return expanded_query, metadata

    def get_synonyms(self, term: str) -> list[str]:
        """
        Get synonyms for a specific term.

        Args:
            term: Term to get synonyms for

        Returns:
            List of synonyms (empty if no synonyms found)

        Example:
            >>> expander = SynonymExpander()
            >>> synonyms = expander.get_synonyms("machine learning")
            >>> print(synonyms)
            ['ml', 'statistical learning', 'predictive modeling']
        """
        term_lower = term.lower()
        return self._synonym_dict.get(term_lower, [])

    def add_synonym(self, term: str, synonyms: list[str]) -> None:
        """
        Add custom synonym mapping.

        Args:
            term: Base term
            synonyms: List of synonyms for the term

        Example:
            >>> expander = SynonymExpander()
            >>> expander.add_synonym("rag", ["retrieval augmented generation", "retrieval-based generation"])
        """
        term_lower = term.lower()
        if term_lower in self._synonym_dict:
            # Merge with existing
            existing = self._synonym_dict[term_lower]
            self._synonym_dict[term_lower] = list(set(existing + synonyms))
        else:
            self._synonym_dict[term_lower] = synonyms

        logger.debug(f"Added {len(synonyms)} synonyms for '{term}'")
