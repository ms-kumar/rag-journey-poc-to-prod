"""
Demonstration of metadata filtering capabilities.

This script shows practical examples of filtering documents by:
- Source (single/multiple files)
- Tags (categories, topics)
- Date ranges
- Authors
- Complex combinations

Run with: uv run python examples/metadata_filters_demo.py
"""

from datetime import datetime

from src.services.vectorstore.filters import FilterBuilder, build_filter_from_dict


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}\n")


def demo_source_filters() -> None:
    """Demonstrate source-based filtering."""
    print_section("Source Filtering")

    # Single source
    print("1. Single source filter:")
    filter1 = FilterBuilder().source("research_paper.pdf").build()
    print(f"   Filter: source = 'research_paper.pdf'")
    print(f"   Result: {filter1}")

    # Multiple sources
    print("\n2. Multiple sources (match any):")
    filter2 = FilterBuilder().sources(["paper1.pdf", "paper2.pdf", "paper3.pdf"]).build()
    print(f"   Filter: sources in ['paper1.pdf', 'paper2.pdf', 'paper3.pdf']")
    print(f"   Result: {filter2}")

    # Exclude source
    print("\n3. Exclude specific source:")
    filter3 = FilterBuilder().sources(["paper1.pdf", "paper2.pdf"]).exclude_source("draft.txt").build()
    print(f"   Filter: sources in ['paper1.pdf', 'paper2.pdf'] AND NOT 'draft.txt'")
    print(f"   Result: {filter3}")

    # Dictionary format
    print("\n4. Using dictionary format:")
    filter_dict = {"sources": ["doc1.txt", "doc2.txt"]}
    filter4 = build_filter_from_dict(filter_dict)
    print(f"   Filter dict: {filter_dict}")
    print(f"   Result: {filter4}")


def demo_tag_filters() -> None:
    """Demonstrate tag-based filtering."""
    print_section("Tag Filtering")

    # Single tag
    print("1. Single tag filter:")
    filter1 = FilterBuilder().tag("machine-learning").build()
    print(f"   Filter: tag = 'machine-learning'")
    print(f"   Result: {filter1}")

    # Multiple tags
    print("\n2. Multiple tags (match any):")
    filter2 = FilterBuilder().tags(["ai", "ml", "deep-learning", "nlp"]).build()
    print(f"   Filter: tags in ['ai', 'ml', 'deep-learning', 'nlp']")
    print(f"   Result: {filter2}")

    # Dictionary format
    print("\n3. Using dictionary format:")
    filter_dict = {"tags": ["python", "fastapi", "rag"]}
    filter3 = build_filter_from_dict(filter_dict)
    print(f"   Filter dict: {filter_dict}")
    print(f"   Result: {filter3}")


def demo_date_filters() -> None:
    """Demonstrate date range filtering."""
    print_section("Date Range Filtering")

    # Date range with ISO strings
    print("1. Date range with ISO strings:")
    filter1 = FilterBuilder().date_range(after="2024-01-01", before="2024-12-31").build()
    print(f"   Filter: date between '2024-01-01' and '2024-12-31'")
    print(f"   Result: {filter1}")

    # Date range with datetime objects
    print("\n2. Date range with datetime objects:")
    start = datetime(2024, 1, 1)
    end = datetime(2024, 12, 31)
    filter2 = FilterBuilder().date_range(after=start, before=end).build()
    print(f"   Filter: date between {start} and {end}")
    print(f"   Result: {filter2}")

    # After date only
    print("\n3. Documents created after specific date:")
    filter3 = FilterBuilder().created_after("2024-06-01").build()
    print(f"   Filter: created_at >= '2024-06-01'")
    print(f"   Result: {filter3}")

    # Before date only
    print("\n4. Documents created before specific date:")
    filter4 = FilterBuilder().created_before("2023-12-31").build()
    print(f"   Filter: created_at <= '2023-12-31'")
    print(f"   Result: {filter4}")

    # Dictionary format
    print("\n5. Using dictionary format:")
    filter_dict = {"date_after": "2024-01-01", "date_before": "2024-12-31"}
    filter5 = build_filter_from_dict(filter_dict)
    print(f"   Filter dict: {filter_dict}")
    print(f"   Result: {filter5}")

    # Custom date field
    print("\n6. Custom date field:")
    filter6 = FilterBuilder().date_range(after="2024-01-01", field="published_date").build()
    print(f"   Filter: published_date >= '2024-01-01'")
    print(f"   Result: {filter6}")


def demo_author_filters() -> None:
    """Demonstrate author-based filtering."""
    print_section("Author Filtering")

    # Single author
    print("1. Single author filter:")
    filter1 = FilterBuilder().author("John Smith").build()
    print(f"   Filter: author = 'John Smith'")
    print(f"   Result: {filter1}")

    # Multiple authors
    print("\n2. Multiple authors (match any):")
    filter2 = FilterBuilder().authors(["Smith", "Johnson", "Williams"]).build()
    print(f"   Filter: author in ['Smith', 'Johnson', 'Williams']")
    print(f"   Result: {filter2}")

    # Dictionary format
    print("\n3. Using dictionary format:")
    filter_dict = {"authors": ["LeCun", "Hinton", "Bengio"]}
    filter3 = build_filter_from_dict(filter_dict)
    print(f"   Filter dict: {filter_dict}")
    print(f"   Result: {filter3}")


def demo_complex_filters() -> None:
    """Demonstrate complex filter combinations."""
    print_section("Complex Filter Combinations")

    # Research paper scenario
    print("1. Research paper filtering:")
    print("   Scenario: Recent papers by specific authors on AI/ML topics")
    filter1 = (
        FilterBuilder()
        .authors(["Smith", "Johnson", "Lee"])
        .tags(["ai", "machine-learning"])
        .date_range(after="2024-01-01")
        .range("citations", gte=50)
        .must_not("venue", "workshop")
        .build()
    )
    print(f"   Filter: authors in [Smith, Johnson, Lee]")
    print(f"           AND tags in [ai, machine-learning]")
    print(f"           AND date >= 2024-01-01")
    print(f"           AND citations >= 50")
    print(f"           AND venue != workshop")
    print(f"   Result: {filter1}")

    # Document management scenario
    print("\n2. Document management filtering:")
    print("   Scenario: Q1 2024 financial reports (final versions only)")
    filter2 = (
        FilterBuilder()
        .sources(["quarterly_report_q1.pdf", "financial_analysis_q1.pdf"])
        .tag("quarterly-report")
        .date_range(after="2024-01-01", before="2024-03-31")
        .author("Finance Team")
        .must_not("status", "draft")
        .build()
    )
    print(f"   Filter: sources in [quarterly_report_q1.pdf, financial_analysis_q1.pdf]")
    print(f"           AND tag = quarterly-report")
    print(f"           AND date between 2024-01-01 and 2024-03-31")
    print(f"           AND author = Finance Team")
    print(f"           AND status != draft")
    print(f"   Result: {filter2}")

    # News/content scenario
    print("\n3. News content filtering:")
    print("   Scenario: Recent AI news from trusted sources")
    filter3 = (
        FilterBuilder()
        .sources(["techcrunch.com", "mit-news.com", "arxiv.org"])
        .tags(["artificial-intelligence", "technology"])
        .date_range(after="2024-11-01")
        .must_not("category", "opinion")
        .build()
    )
    print(f"   Filter: sources in [techcrunch.com, mit-news.com, arxiv.org]")
    print(f"           AND tags in [artificial-intelligence, technology]")
    print(f"           AND date >= 2024-11-01")
    print(f"           AND category != opinion")
    print(f"   Result: {filter3}")


def demo_dictionary_filters() -> None:
    """Demonstrate dictionary-based filter construction."""
    print_section("Dictionary-Based Filters")

    # Simple filter
    print("1. Simple filter:")
    filter_dict1 = {"source": "paper.pdf", "author": "Smith"}
    filter1 = build_filter_from_dict(filter_dict1)
    print(f"   Dict: {filter_dict1}")
    print(f"   Result: {filter1}")

    # With operators
    print("\n2. Filter with operators:")
    filter_dict2 = {
        "source": "research.pdf",
        "year$gte": 2020,
        "year$lte": 2024,
        "citations$gt": 100,
        "status$not": "retracted",
    }
    filter2 = build_filter_from_dict(filter_dict2)
    print(f"   Dict: {filter_dict2}")
    print(f"   Result: {filter2}")

    # Comprehensive filter
    print("\n3. Comprehensive filter:")
    filter_dict3 = {
        "sources": ["paper1.pdf", "paper2.pdf"],
        "tags": ["ai", "ml"],
        "authors": ["Smith", "Johnson"],
        "date_after": "2024-01-01",
        "citations$gte": 50,
        "venue$in": ["NeurIPS", "ICML", "ICLR"],
        "status$not": "draft",
    }
    filter3 = build_filter_from_dict(filter_dict3)
    print(f"   Dict: {filter_dict3}")
    print(f"   Result: {filter3}")


def demo_api_usage() -> None:
    """Demonstrate API usage patterns."""
    print_section("API Usage Examples")

    print("1. Simple source filter (curl):")
    print("""
    curl -X POST http://localhost:8000/api/v1/rag/generate \\
      -H "Content-Type: application/json" \\
      -d '{
        "prompt": "What are the key findings?",
        "top_k": 5,
        "metadata_filters": {"source": "research_paper.pdf"}
      }'
    """)

    print("\n2. Date range filter (curl):")
    print("""
    curl -X POST http://localhost:8000/api/v1/rag/generate \\
      -H "Content-Type: application/json" \\
      -d '{
        "prompt": "Recent developments?",
        "metadata_filters": {
          "date_after": "2024-01-01",
          "date_before": "2024-12-31"
        }
      }'
    """)

    print("\n3. Complex filter with reranking (curl):")
    print("""
    curl -X POST http://localhost:8000/api/v1/rag/generate \\
      -H "Content-Type: application/json" \\
      -d '{
        "prompt": "AI breakthroughs",
        "top_k": 10,
        "metadata_filters": {
          "authors": ["LeCun", "Hinton", "Bengio"],
          "tags": ["deep-learning", "neural-networks"],
          "date_after": "2023-01-01",
          "citations$gte": 50
        },
        "enable_reranking": true,
        "search_type": "hybrid"
      }'
    """)


def demo_practical_scenarios() -> None:
    """Demonstrate practical real-world scenarios."""
    print_section("Practical Real-World Scenarios")

    print("Scenario 1: Academic Literature Review")
    print("Task: Find recent deep learning papers by top researchers\n")
    scenario1 = {
        "authors": ["LeCun", "Hinton", "Bengio", "Schmidhuber"],
        "tags": ["deep-learning", "neural-networks"],
        "date_after": "2022-01-01",
        "citations$gte": 100,
        "venue$in": ["NeurIPS", "ICML", "ICLR", "CVPR"],
    }
    print(f"Filter: {scenario1}")
    filter1 = build_filter_from_dict(scenario1)
    print(f"Built filter: {filter1}\n")

    print("\nScenario 2: Corporate Knowledge Base")
    print("Task: Find Q4 2024 reports from engineering team\n")
    scenario2 = {
        "sources": ["engineering/"],
        "tag": "quarterly-report",
        "author": "Engineering Team",
        "date_after": "2024-10-01",
        "date_before": "2024-12-31",
        "status$not": "draft",
    }
    print(f"Filter: {scenario2}")
    filter2 = build_filter_from_dict(scenario2)
    print(f"Built filter: {filter2}\n")

    print("\nScenario 3: Legal Document Search")
    print("Task: Find case law from specific jurisdictions in last 5 years\n")
    scenario3 = {
        "sources": ["supreme-court", "circuit-court"],
        "tags": ["copyright", "patent-law"],
        "date_after": "2019-01-01",
        "jurisdiction$in": ["9th Circuit", "Federal Circuit"],
    }
    print(f"Filter: {scenario3}")
    filter3 = build_filter_from_dict(scenario3)
    print(f"Built filter: {filter3}\n")

    print("\nScenario 4: Medical Research")
    print("Task: Find recent clinical trials on specific conditions\n")
    scenario4 = {
        "tags": ["clinical-trial", "randomized-controlled"],
        "date_after": "2023-01-01",
        "sources": ["pubmed", "clinicaltrials.gov"],
        "phase$in": ["Phase 3", "Phase 4"],
        "status": "completed",
    }
    print(f"Filter: {scenario4}")
    filter4 = build_filter_from_dict(scenario4)
    print(f"Built filter: {filter4}\n")


def main() -> None:
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("METADATA FILTERING DEMONSTRATION")
    print("=" * 70)

    demo_source_filters()
    demo_tag_filters()
    demo_date_filters()
    demo_author_filters()
    demo_complex_filters()
    demo_dictionary_filters()
    demo_api_usage()
    demo_practical_scenarios()

    print_section("Summary")
    print("Key Takeaways:")
    print("1. Use FilterBuilder for programmatic filter construction")
    print("2. Use dictionaries for API-friendly filter definitions")
    print("3. Combine multiple criteria for precise filtering")
    print("4. Use convenience methods (source, tags, date_range) for clarity")
    print("5. Apply operators ($gte, $in, $not) for advanced filtering")
    print("\nFor more information, see docs/metadata-filters.md")


if __name__ == "__main__":
    main()
