#!/usr/bin/env python3
"""Run Phase 3: Reranker Scoring.

This script executes the third phase of the pipeline, running multiple
cross-encoder rerankers on the top-k1 documents from Phase 2 aggregation.

Usage
-----
    # Run specific rerankers
    python scripts/run_phase3.py --dataset scifact --rerankers minilm_l6,bge_reranker

    # Run all rerankers
    python scripts/run_phase3.py --dataset scifact --rerankers all

    # Custom top-k1 and aggregation method
    python scripts/run_phase3.py --dataset scifact --rerankers all --top-k1 500 --agg-method rrf

Examples
--------
    # Quick test with smallest reranker
    python scripts/run_phase3.py --dataset scifact --rerankers minilm_l6 -v

    # Full run with all rerankers
    python scripts/run_phase3.py --dataset scifact --rerankers all --top-k1 500

    # Force re-run even if output exists
    python scripts/run_phase3.py --dataset scifact --rerankers minilm_l6 --force
"""

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_dataset
from src.data.models import RankingEntry
from src.reranking import RERANKER_REGISTRY, get_reranker


def setup_logging(verbose: bool = False) -> None:
    """Configure logging.

    Parameters
    ----------
    verbose : bool
        If True, use DEBUG level. Otherwise INFO.
    """
    level: int = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run Phase 3: Reranker Scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run MiniLM L6 on SciFact
    python scripts/run_phase3.py --dataset scifact --rerankers minilm_l6

    # Run all rerankers
    python scripts/run_phase3.py --dataset scifact --rerankers all

    # Run with custom top-k1
    python scripts/run_phase3.py --dataset scifact --rerankers all --top-k1 500
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name (scifact, scidocs)",
    )

    parser.add_argument(
        "--rerankers",
        type=str,
        help=f"Comma-separated reranker names or 'all'. Available: {list(RERANKER_REGISTRY.keys())}",
    )

    parser.add_argument(
        "--top-k1",
        type=int,
        default=500,
        help="Number of top documents to rerank per query (default: 500)",
    )

    parser.add_argument(
        "--agg-method",
        type=str,
        default="rrf",
        choices=["rrf", "borda"],
        help="Aggregation method from Phase 2 to use (default: rrf)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for reranking (default: model-specific)",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Base data directory (default: ./data)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if output exists",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--list-rerankers",
        action="store_true",
        help="List available rerankers and exit",
    )

    return parser.parse_args()


def load_phase2_rankings(
    data_dir: Path,
    dataset: str,
    agg_method: str,
) -> pd.DataFrame:
    """Load aggregated rankings from Phase 2.

    Parameters
    ----------
    data_dir : Path
        Base data directory.
    dataset : str
        Dataset name.
    agg_method : str
        Aggregation method ('rrf' or 'borda').

    Returns
    -------
    pd.DataFrame
        Aggregated rankings with columns: query_id, doc_id, rank, agg_score.

    Raises
    ------
    FileNotFoundError
        If Phase 2 output does not exist.
    """
    phase2_dir: Path = data_dir / "phase2_ir_aggregation" / dataset
    rankings_path: Path = phase2_dir / f"aggregated_{agg_method}.parquet"

    if not rankings_path.exists():
        raise FileNotFoundError(
            f"Phase 2 rankings not found at {rankings_path}. "
            f"Run Phase 2 first with: python scripts/run_phase2.py --dataset {dataset}"
        )

    return pd.read_parquet(rankings_path)


def save_rankings(
    rankings: list[RankingEntry],
    output_path: Path,
) -> None:
    """Save rankings to parquet file.

    Parameters
    ----------
    rankings : list[RankingEntry]
        List of ranking entries to save.
    output_path : Path
        Path to output parquet file.
    """
    data: list[dict[str, Any]] = [r.model_dump() for r in rankings]
    df: pd.DataFrame = pd.DataFrame(data)

    # Ensure correct dtypes per schema
    df = df.astype(
        {
            "query_id": "string",
            "doc_id": "string",
            "rank": "int32",
            "score": "float64",
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def main() -> int:
    """Main entry point.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    args: argparse.Namespace = parse_args()
    setup_logging(args.verbose)
    logger: logging.Logger = logging.getLogger(__name__)

    # Handle --list-rerankers flag
    if args.list_rerankers:
        print("Available rerankers:")
        for name in sorted(RERANKER_REGISTRY.keys()):
            print(f"  - {name}")
        return 0

    # Validate required arguments
    if not args.dataset:
        logger.error("--dataset is required")
        return 1
    if not args.rerankers:
        logger.error("--rerankers is required")
        return 1

    # Determine rerankers to run
    reranker_names: list[str]
    if args.rerankers.lower() == "all":
        reranker_names = list(RERANKER_REGISTRY.keys())
    else:
        reranker_names = [r.strip() for r in args.rerankers.split(",")]

    # Validate reranker names
    invalid: list[str] = [r for r in reranker_names if r not in RERANKER_REGISTRY]
    if invalid:
        logger.error(
            f"Unknown rerankers: {invalid}. "
            f"Available: {list(RERANKER_REGISTRY.keys())}"
        )
        return 1

    # Setup paths
    data_dir: Path = args.data_dir.resolve()
    output_dir: Path = data_dir / "phase3_reranking" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Phase 3: Reranker Scoring")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Rerankers: {reranker_names}")
    logger.info(f"Top-k1: {args.top_k1}")
    logger.info(f"Aggregation method: {args.agg_method}")

    # Load Phase 2 rankings
    logger.info(f"Loading Phase 2 rankings ({args.agg_method})...")
    try:
        rankings_df: pd.DataFrame = load_phase2_rankings(
            data_dir, args.dataset, args.agg_method
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    num_queries: int = rankings_df["query_id"].nunique()
    logger.info(f"Loaded rankings: {len(rankings_df):,} rows, {num_queries} queries")

    # Load dataset (for corpus and queries)
    logger.info(f"Loading dataset: {args.dataset}")
    try:
        dataset = load_dataset(args.dataset, data_dir)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return 1

    logger.info(
        f"Dataset loaded: {len(dataset.corpus)} docs, {len(dataset.queries)} queries"
    )

    # Track metadata
    metadata: dict[str, Any] = {
        "phase": 3,
        "phase_name": "reranking",
        "dataset": args.dataset,
        "top_k1": args.top_k1,
        "agg_method": args.agg_method,
        "num_documents": len(dataset.corpus),
        "num_queries": num_queries,
        "rerankers": {},
        "timestamp": datetime.now(UTC).isoformat(),
    }

    # Run each reranker
    success_count: int = 0
    for reranker_name in reranker_names:
        output_path: Path = output_dir / f"{reranker_name}.parquet"

        if output_path.exists() and not args.force:
            logger.info(
                f"Skipping {reranker_name} (output exists, use --force to override)"
            )
            metadata["rerankers"][reranker_name] = {"status": "skipped"}
            continue

        logger.info(f"Running reranker: {reranker_name}")
        start_time: datetime = datetime.now(UTC)

        try:
            # Build kwargs for reranker
            reranker_kwargs: dict[str, Any] = {}
            if args.batch_size is not None:
                reranker_kwargs["batch_size"] = args.batch_size

            reranker = get_reranker(reranker_name, **reranker_kwargs)

            # Run reranking
            rankings: list[RankingEntry] = reranker.run(
                dataset=dataset,
                rankings_df=rankings_df,
                top_k=args.top_k1,
                show_progress=True,
            )

            save_rankings(rankings, output_path)

            elapsed: float = (datetime.now(UTC) - start_time).total_seconds()
            result_queries: int = len(set(r.query_id for r in rankings))
            logger.info(
                f"{reranker_name}: {len(rankings):,} rankings for "
                f"{result_queries} queries in {elapsed:.1f}s"
            )

            metadata["rerankers"][reranker_name] = {
                "status": "completed",
                "num_rankings": len(rankings),
                "num_queries": result_queries,
                "elapsed_seconds": elapsed,
                "output_file": str(output_path.name),
                "model_name": reranker.model_name if hasattr(reranker, "model_name") else None,
            }

            # Clear model to free memory before next reranker
            reranker.clear()
            success_count += 1

        except Exception as e:
            logger.exception(f"Failed to run {reranker_name}: {e}")
            metadata["rerankers"][reranker_name] = {
                "status": "failed",
                "error": str(e),
            }

    # Save metadata
    metadata_path: Path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Phase 3 complete. {success_count}/{len(reranker_names)} rerankers succeeded."
    )
    logger.info(f"Outputs in: {output_dir}")

    # Return success only if at least one reranker succeeded
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
