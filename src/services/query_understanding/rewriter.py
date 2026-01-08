"""
Query rewriting for improved retrieval.

Rewrites user queries to be more explicit, better formed, and optimized
for semantic search and keyword matching.
"""

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QueryRewriterConfig:
    """Configuration for query rewriter."""

    expand_acronyms: bool = True
    fix_typos: bool = True
    add_context: bool = True
    max_rewrites: int = 3
    min_query_length: int = 3

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_rewrites < 1:
            raise ValueError("max_rewrites must be at least 1")
        if self.min_query_length < 1:
            raise ValueError("min_query_length must be at least 1")


class QueryRewriter:
    """
    Rewrite queries to improve retrieval quality.

    Performs multiple rewriting strategies:
    - Acronym expansion (ML → machine learning)
    - Typo correction
    - Context addition (implicit → explicit)
    - Question reformulation
    """

    def __init__(self, config: QueryRewriterConfig | None = None):
        """
        Initialize query rewriter.

        Args:
            config: Rewriter configuration
        """
        self.config = config or QueryRewriterConfig()
        self._acronym_map = self._build_acronym_map()
        self._common_typos = self._build_typo_map()

    def _build_acronym_map(self) -> dict[str, str]:
        """Build map of common acronyms to expansions."""
        return {
            "ml": "machine learning",
            "ai": "artificial intelligence",
            "nlp": "natural language processing",
            "dl": "deep learning",
            "nn": "neural network",
            "rnn": "recurrent neural network",
            "cnn": "convolutional neural network",
            "lstm": "long short-term memory",
            "gru": "gated recurrent unit",
            "bert": "bidirectional encoder representations from transformers",
            "gpt": "generative pre-trained transformer",
            "api": "application programming interface",
            "rest": "representational state transfer",
            "crud": "create read update delete",
            "sql": "structured query language",
            "http": "hypertext transfer protocol",
            "https": "hypertext transfer protocol secure",
            "url": "uniform resource locator",
            "uri": "uniform resource identifier",
            "json": "javascript object notation",
            "xml": "extensible markup language",
            "html": "hypertext markup language",
            "css": "cascading style sheets",
            "js": "javascript",
            "ts": "typescript",
            "py": "python",
            "rag": "retrieval augmented generation",
            "llm": "large language model",
            "gpu": "graphics processing unit",
            "cpu": "central processing unit",
            "ram": "random access memory",
            "ssd": "solid state drive",
            "hdd": "hard disk drive",
            "os": "operating system",
            "db": "database",
            "ci": "continuous integration",
            "cd": "continuous deployment",
            "aws": "amazon web services",
            "gcp": "google cloud platform",
            "k8s": "kubernetes",
        }

    def _build_typo_map(self) -> dict[str, str]:
        """Build map of common typos to corrections."""
        return {
            "machien": "machine",
            "learing": "learning",
            "nueral": "neural",
            "netowrk": "network",
            "retrevial": "retrieval",
            "retreival": "retrieval",
            "retreval": "retrieval",
            "algorithim": "algorithm",
            "algorythm": "algorithm",
            "definately": "definitely",
            "seperate": "separate",
            "occured": "occurred",
            "recieve": "receive",
            "beleive": "believe",
            "acheive": "achieve",
            "thier": "their",
            "wierd": "weird",
            "freind": "friend",
            "calender": "calendar",
            "colum": "column",
            "databse": "database",
            "querey": "query",
            "functon": "function",
            "fucntion": "function",
            "implmentation": "implementation",
            "paramter": "parameter",
            "varible": "variable",
            "variabel": "variable",
        }

    def rewrite(self, query: str) -> tuple[str, dict[str, object]]:
        """
        Rewrite query for better retrieval.

        Args:
            query: Original query string

        Returns:
            Tuple of (rewritten_query, metadata with latency info)

        Example:
            >>> rewriter = QueryRewriter()
            >>> rewritten, meta = rewriter.rewrite("what is ML?")
            >>> print(rewritten)
            "what is machine learning?"
            >>> print(meta["latency_ms"])
            0.5
        """
        start_time = time.perf_counter()

        # Skip very short queries
        if len(query.strip()) < self.config.min_query_length:
            latency = (time.perf_counter() - start_time) * 1000
            return query, {"latency_ms": latency, "rewrites_applied": 0}

        original_query = query
        rewrites_applied = 0

        # Step 1: Fix typos
        if self.config.fix_typos:
            query_fixed, changed = self._fix_typos(query)
            if changed:
                query = query_fixed
                rewrites_applied += 1
                logger.debug(f"Typo fix: '{original_query}' → '{query}'")

        # Step 2: Expand acronyms
        if self.config.expand_acronyms:
            query_expanded, changed = self._expand_acronyms(query)
            if changed:
                query = query_expanded
                rewrites_applied += 1
                logger.debug(f"Acronym expansion: '{original_query}' → '{query}'")

        # Step 3: Add context (if needed)
        if self.config.add_context:
            query_contextualized, changed = self._add_context(query)
            if changed:
                query = query_contextualized
                rewrites_applied += 1
                logger.debug(f"Context addition: '{original_query}' → '{query}'")

        latency = (time.perf_counter() - start_time) * 1000

        metadata = {
            "latency_ms": latency,
            "rewrites_applied": rewrites_applied,
            "original_query": original_query,
        }

        if rewrites_applied > 0:
            logger.info(
                f"Query rewritten: '{original_query}' → '{query}' "
                f"({rewrites_applied} rewrites, {latency:.2f}ms)"
            )

        return query, metadata

    def _fix_typos(self, query: str) -> tuple[str, bool]:
        """Fix common typos in query."""
        words = query.split()
        fixed_words = []
        changed = False

        for word in words:
            word_lower = word.lower()
            # Strip punctuation for matching
            word_clean = word_lower.strip(".,!?;:")

            if word_clean in self._common_typos:
                fixed = self._common_typos[word_clean]
                # Preserve case
                if word[0].isupper():
                    fixed = fixed.capitalize()
                fixed_words.append(fixed)
                changed = True
            else:
                fixed_words.append(word)

        return " ".join(fixed_words), changed

    def _expand_acronyms(self, query: str) -> tuple[str, bool]:
        """Expand acronyms to full terms."""
        words = query.split()
        expanded_words = []
        changed = False

        for word in words:
            word_lower = word.lower()
            # Strip punctuation
            word_clean = word_lower.strip(".,!?;:")

            if word_clean in self._acronym_map:
                expansion = self._acronym_map[word_clean]
                # Preserve case
                if word[0].isupper():
                    expansion = expansion.capitalize()
                expanded_words.append(expansion)
                changed = True
            else:
                expanded_words.append(word)

        return " ".join(expanded_words), changed

    def _add_context(self, query: str) -> tuple[str, bool]:
        """Add implicit context to query."""
        # Convert questions to statements
        query_lower = query.lower().strip()

        # "what is X?" → "X definition explanation"
        if query_lower.startswith("what is "):
            topic = query[8:].rstrip("?").strip()
            return f"{topic} definition explanation", True

        # "how to X?" → "X tutorial guide steps"
        if query_lower.startswith("how to "):
            topic = query[7:].rstrip("?").strip()
            return f"{topic} tutorial guide steps", True

        # "why X?" → "X reason explanation"
        if query_lower.startswith("why "):
            topic = query[4:].rstrip("?").strip()
            return f"{topic} reason explanation", True

        # "when X?" → "X time timing schedule"
        if query_lower.startswith("when "):
            topic = query[5:].rstrip("?").strip()
            return f"{topic} time timing schedule", True

        return query, False

    def get_rewrites(self, query: str) -> list[str]:
        """
        Generate multiple rewrite candidates.

        Args:
            query: Original query

        Returns:
            List of rewritten queries (original + rewrites)

        Example:
            >>> rewriter = QueryRewriter()
            >>> rewrites = rewriter.get_rewrites("what is ML?")
            >>> for r in rewrites:
            ...     print(r)
            what is ML?
            what is machine learning?
            machine learning definition explanation
        """
        rewrites = [query]

        # Add progressive rewrites
        current = query

        # Typo fix
        if self.config.fix_typos:
            fixed, changed = self._fix_typos(current)
            if changed and fixed not in rewrites:
                rewrites.append(fixed)
                current = fixed

        # Acronym expansion
        if self.config.expand_acronyms:
            expanded, changed = self._expand_acronyms(current)
            if changed and expanded not in rewrites:
                rewrites.append(expanded)
                current = expanded

        # Context addition
        if self.config.add_context:
            contextualized, changed = self._add_context(current)
            if changed and contextualized not in rewrites:
                rewrites.append(contextualized)

        return rewrites[: self.config.max_rewrites]
