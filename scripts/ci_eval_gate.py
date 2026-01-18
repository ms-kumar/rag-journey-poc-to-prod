"""
CI Evaluation Gate Script.

Runs evaluation suite and exits with error code if thresholds are not met.
Used in CI/CD pipelines to ensure quality standards.
"""

import json
import logging
import sys
from pathlib import Path

from src.config import Settings
from src.services.embeddings.adapter import LangChainEmbeddingsAdapter
from src.services.embeddings.factory import get_embed_client
from src.services.evaluation import EvalDataset, EvaluationHarness, ThresholdConfig
from src.services.vectorstore.factory import get_vectorstore_client

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config() -> ThresholdConfig:
    """
    Load threshold configuration from file or use defaults.

    Returns:
        ThresholdConfig with evaluation thresholds
    """
    config_path = Path("config/eval_thresholds.json")

    if config_path.exists():
        logger.info(f"Loading thresholds from {config_path}")
        with open(config_path) as f:
            config_data = json.load(f)

        return ThresholdConfig(
            min_precision_at_5=config_data.get("min_precision_at_5", 0.6),
            min_recall_at_10=config_data.get("min_recall_at_10", 0.7),
            min_mrr=config_data.get("min_mrr", 0.5),
            min_ndcg_at_10=config_data.get("min_ndcg_at_10", 0.65),
            min_map=config_data.get("min_map", 0.6),
            max_latency_p95=config_data.get("max_latency_p95", 2000.0),
        )
    else:
        logger.info("Using default thresholds")
        return ThresholdConfig()


def create_retrieval_function(config: Settings):
    """
    Create retrieval function for evaluation.

    Args:
        config: Application settings

    Returns:
        Retrieval function that takes query and returns documents
    """
    # Initialize clients
    embed_client = get_embed_client(config)
    embeddings_adapter = LangChainEmbeddingsAdapter(embed_client)
    vectorstore = get_vectorstore_client(
        embeddings=embeddings_adapter,
        settings=config.vectorstore,
    )

    def retrieval_fn(query: str, k: int = 20):
        """Retrieve documents for query."""
        return vectorstore.similarity_search(query, k=k)

    return retrieval_fn


def run_evaluation(
    dataset_path: Path | str,
    config: Settings,
    thresholds: ThresholdConfig,
    output_path: Path | str | None = None,
) -> bool:
    """
    Run evaluation on dataset.

    Args:
        dataset_path: Path to evaluation dataset
        config: Application settings
        thresholds: Threshold configuration
        output_path: Optional path to save results

    Returns:
        True if evaluation passed, False otherwise
    """
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = EvalDataset.load(dataset_path)
    logger.info(f"Loaded {len(dataset)} evaluation examples")

    # Create retrieval function
    retrieval_fn = create_retrieval_function(config)

    # Create harness
    harness = EvaluationHarness(
        retrieval_function=retrieval_fn,
        thresholds=thresholds,
    )

    # Run evaluation
    logger.info("Starting evaluation...")
    result = harness.evaluate(dataset, k_values=[1, 3, 5, 10, 20])

    # Print results
    print("\n" + "=" * 60)
    print(result.metrics.get_summary())
    print("=" * 60)

    if result.passed:
        print("\n✅ EVALUATION PASSED - All thresholds met")
    else:
        print("\n❌ EVALUATION FAILED - Thresholds not met:")
        for check in result.failed_checks:
            print(f"  • {check}")

    # Save results if output path provided
    if output_path:
        output_path = Path(output_path)
        harness.save_results(result, output_path)
        logger.info(f"Results saved to {output_path}")

    return result.passed


def main():
    """Main entry point for CI evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run CI evaluation gate")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/eval/rag_test_small.json",
        help="Path to evaluation dataset (default: data/eval/rag_test_small.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/eval_ci_result.json",
        help="Path to save evaluation results (default: results/eval_ci_result.json)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use stricter thresholds for evaluation",
    )

    args = parser.parse_args()

    # Load configuration
    config = Settings()
    thresholds = load_config()

    # Apply strict thresholds if requested
    if args.strict:
        logger.info("Using strict thresholds")
        thresholds.min_precision_at_5 = 0.7
        thresholds.min_recall_at_10 = 0.8
        thresholds.min_mrr = 0.6
        thresholds.min_ndcg_at_10 = 0.75

    try:
        # Run evaluation
        passed = run_evaluation(
            dataset_path=args.dataset,
            config=config,
            thresholds=thresholds,
            output_path=args.output,
        )

        # Exit with appropriate code
        sys.exit(0 if passed else 1)

    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {e}")
        logger.error("Please create evaluation datasets first:")
        logger.error("  python scripts/create_eval_datasets.py")
        sys.exit(2)
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
