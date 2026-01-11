"""
Demo script for query understanding features.

Demonstrates query rewriting, synonym expansion, and intent classification.
"""

from src.services.query_understanding import (
    QueryRewriter,
    QueryRewriterConfig,
    QueryUnderstanding,
    QueryUnderstandingConfig,
    SynonymExpander,
    SynonymExpanderConfig,
)


def demo_query_rewriter():
    """Demonstrate query rewriting capabilities."""
    print("=" * 80)
    print("QUERY REWRITER DEMO")
    print("=" * 80)

    rewriter = QueryRewriter()

    test_cases = [
        "what is ML?",
        "how to use NLP in AI?",
        "machien learing algorithim",
        "explain RAG system with LLM",
        "why use nn for dl?",
    ]

    for query in test_cases:
        rewritten, meta = rewriter.rewrite(query)
        print(f"\nOriginal:  {query}")
        print(f"Rewritten: {rewritten}")
        print(f"  Rewrites applied: {meta['rewrites_applied']}")
        print(f"  Latency: {meta['latency_ms']:.2f}ms")


def demo_synonym_expander():
    """Demonstrate synonym expansion."""
    print("\n" + "=" * 80)
    print("SYNONYM EXPANDER DEMO")
    print("=" * 80)

    expander = SynonymExpander()

    test_cases = [
        "machine learning model",
        "neural network training",
        "database query optimization",
        "fast retrieval system",
    ]

    for query in test_cases:
        expanded, meta = expander.expand(query)
        print(f"\nOriginal: {query}")
        print(f"Expanded: {expanded}")
        print(f"  Terms expanded: {meta['terms_expanded']}")
        print(f"  Synonyms added: {meta['synonyms_added']}")
        print(f"  Latency: {meta['latency_ms']:.2f}ms")

    # Demo custom synonyms
    print("\n" + "-" * 80)
    print("Custom Synonyms Demo")
    print("-" * 80)

    expander.add_synonym("rag", ["retrieval augmented generation", "retrieval-based AI"])
    query = "explain rag system"
    expanded, meta = expander.expand(query)
    print("\nAfter adding custom synonyms for 'rag':")
    print(f"Original: {query}")
    print(f"Expanded: {expanded}")


def demo_full_pipeline():
    """Demonstrate full query understanding pipeline."""
    print("\n" + "=" * 80)
    print("FULL PIPELINE DEMO")
    print("=" * 80)

    config = QueryUnderstandingConfig(
        enable_rewriting=True,
        enable_synonyms=True,
        enable_intent_classification=True,
    )
    qu = QueryUnderstandingClient(config)

    test_cases = [
        "what is ML?",
        "how to train a nueral networ?",
        "Python vs Java performance",
        "error in database query not working",
        "explain deep learning",
    ]

    for query in test_cases:
        result = qu.process(query)
        print(f"\nOriginal:  {result['original_query']}")
        print(f"Processed: {result['processed_query']}")
        if result.get("intent"):
            print(f"Intent:    {result['intent']}")
        print("Metadata:")
        print(f"  Total latency: {result['metadata']['total_latency_ms']:.2f}ms")
        if result["metadata"].get("rewrites_applied"):
            print(f"  Rewrites: {result['metadata']['rewrites_applied']}")
        if result["metadata"].get("synonyms_added"):
            print(f"  Synonyms added: {result['metadata']['synonyms_added']}")


def demo_query_variations():
    """Demonstrate generation of query variations."""
    print("\n" + "=" * 80)
    print("QUERY VARIATIONS DEMO")
    print("=" * 80)

    qu = QueryUnderstanding()

    test_queries = [
        "what is ML?",
        "how to use NLP?",
    ]

    for query in test_queries:
        variations = qu.get_all_variations(query)
        print(f"\nOriginal: {query}")
        print(f"Variations ({len(variations)}):")
        for i, var in enumerate(variations, 1):
            print(f"  {i}. {var}")


def demo_configuration_options():
    """Demonstrate different configuration options."""
    print("\n" + "=" * 80)
    print("CONFIGURATION OPTIONS DEMO")
    print("=" * 80)

    query = "what is ML?"

    # Config 1: Only rewriting
    config1 = QueryUnderstandingConfig(
        enable_rewriting=True,
        enable_synonyms=False,
    )
    qu1 = QueryUnderstandingClient(config1)
    result1 = qu1.process(query)

    print("\nConfig 1: Rewriting only")
    print(f"  Input:  {query}")
    print(f"  Output: {result1['processed_query']}")

    # Config 2: Only synonyms
    config2 = QueryUnderstandingConfig(
        enable_rewriting=False,
        enable_synonyms=True,
    )
    qu2 = QueryUnderstandingClient(config2)
    result2 = qu2.process(query)

    print("\nConfig 2: Synonyms only")
    print(f"  Input:  {query}")
    print(f"  Output: {result2['processed_query']}")

    # Config 3: Custom rewriter config
    rewriter_config = QueryRewriterConfig(
        expand_acronyms=True,
        fix_typos=True,
        add_context=False,  # Disable context addition
        max_rewrites=2,
    )
    config3 = QueryUnderstandingConfig(
        enable_rewriting=True,
        enable_synonyms=True,
        rewriter_config=rewriter_config,
    )
    qu3 = QueryUnderstanding(config3)
    result3 = qu3.process(query)

    print("\nConfig 3: Custom rewriter (no context addition)")
    print(f"  Input:  {query}")
    print(f"  Output: {result3['processed_query']}")

    # Config 4: Custom expander config
    expander_config = SynonymExpanderConfig(
        max_synonyms_per_term=1,  # Limit to 1 synonym per term
        expand_all_terms=False,
    )
    config4 = QueryUnderstandingConfig(
        enable_rewriting=True,
        enable_synonyms=True,
        expander_config=expander_config,
    )
    qu4 = QueryUnderstanding(config4)
    result4 = qu4.process("machine learning model")

    print("\nConfig 4: Custom expander (max 1 synonym per term)")
    print("  Input:  machine learning model")
    print(f"  Output: {result4['processed_query']}")


def demo_latency_analysis():
    """Analyze latency characteristics."""
    print("\n" + "=" * 80)
    print("LATENCY ANALYSIS")
    print("=" * 80)

    qu = QueryUnderstanding()

    test_queries = [
        "ML",  # Very short
        "what is machine learning?",  # Medium
        "how to implement a neural network with backpropagation for deep learning?",  # Long
    ]

    for query in test_queries:
        result = qu.process(query)
        meta = result["metadata"]

        print(f"\nQuery: {query}")
        print(f"  Length: {len(query)} chars")
        print(f"  Total latency: {meta['total_latency_ms']:.3f}ms")
        print(f"    Rewriting: {meta['rewrite_latency_ms']:.3f}ms")
        print(f"    Expansion: {meta['expansion_latency_ms']:.3f}ms")
        if meta.get("intent_latency_ms"):
            print(f"    Intent: {meta['intent_latency_ms']:.3f}ms")


if __name__ == "__main__":
    print("\n🔍 QUERY UNDERSTANDING DEMO\n")

    demo_query_rewriter()
    demo_synonym_expander()
    demo_full_pipeline()
    demo_query_variations()
    demo_configuration_options()
    demo_latency_analysis()

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)
    print("\nKey takeaways:")
    print("✓ Query rewriting handles typos, acronyms, and adds context")
    print("✓ Synonym expansion improves recall with semantically similar terms")
    print("✓ Intent classification helps optimize retrieval strategy")
    print("✓ Latency is very low (< 5ms typical for rule-based processing)")
    print("✓ Configurable components allow fine-tuning for specific use cases")
    print()
