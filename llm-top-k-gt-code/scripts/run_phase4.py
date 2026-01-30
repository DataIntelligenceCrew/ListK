#!/usr/bin/env python
"""Phase 4: Reranker Rank Aggregation

This script aggregates rankings from multiple rerankers (Phase 3) into
unified rankings using various rank aggregation algorithms.

Usage:
    python scripts/run_phase4.py --dataset scifact --methods rrf,borda
    python scripts/run_phase4.py --dataset scifact --methods rrf --rrf-k 60

Input:
    data/phase3_reranking/{dataset}/*.parquet

Output:
    data/phase4_rerank_aggregation/{dataset}/aggregated_{method}.parquet
    data/phase4_rerank_aggregation/{dataset}/metadata.json
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

from src.aggregation import get_aggregator


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
        description="Phase 4: Reranker Rank Aggregation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run RRF aggregation
    python scripts/run_phase4.py --dataset scifact --methods rrf

    # Run multiple aggregation methods
    python scripts/run_phase4.py --dataset scifact --methods rrf,borda

    # Custom RRF k parameter
    python scripts/run_phase4.py --dataset scifact --methods rrf --rrf-k 40
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., scifact, scidocs)",
    )

    parser.add_argument(
        "--methods",
        type=str,
        default="rrf,borda",
        help="Comma-separated list of aggregation methods (default: rrf,borda)",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF smoothing constant k (default: 60)",
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

    return parser.parse_args()


def load_phase3_rankings(
    phase3_dir: Path,
    dataset: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Load all reranker rankings from Phase 3.

    Parameters
    ----------
    phase3_dir : Path
        Base directory for Phase 3 outputs.
    dataset : str
        Dataset name (e.g., "scifact").

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Combined DataFrame with all rankings (with 'reranker' column),
        and list of reranker names.

    Raises
    ------
    FileNotFoundError
        If no parquet files are found.
    """
    dataset_dir: Path = phase3_dir / dataset
    parquet_files: list[Path] = list(dataset_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {dataset_dir}. "
            f"Run Phase 3 first with: python scripts/run_phase3.py --dataset {dataset}"
        )

    dfs: list[pd.DataFrame] = []
    reranker_names: list[str] = []

    for pq_file in sorted(parquet_files):
        reranker_name: str = pq_file.stem
        reranker_names.append(reranker_name)

        df: pd.DataFrame = pd.read_parquet(pq_file)
        df["reranker"] = reranker_name
        dfs.append(df)

    combined_df: pd.DataFrame = pd.concat(dfs, ignore_index=True)
    return combined_df, reranker_names


def run_aggregation(
    rankings_df: pd.DataFrame,
    method: str,
    rrf_k: int = 60,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run rank aggregation using the specified method.

    Parameters
    ----------
    rankings_df : pd.DataFrame
        Combined rankings DataFrame with 'reranker' column.
    method : str
        Aggregation method: "rrf", "borda", etc.
    rrf_k : int, optional
        RRF smoothing constant. Default 60.
    show_progress : bool, optional
        Whether to show progress bar. Default True.

    Returns
    -------
    pd.DataFrame
        Aggregated rankings with columns: query_id, doc_id, rank, agg_score.
    """
    if method == "rrf":
        aggregator = get_aggregator(method, k=rrf_k)
    else:
        aggregator = get_aggregator(method)

    return aggregator.aggregate_all(
        rankings_df,
        ranker_column="reranker",
        show_progress=show_progress,
    )


def save_aggregated_rankings(
    aggregated_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save aggregated rankings to parquet.

    Parameters
    ----------
    aggregated_df : pd.DataFrame
        Aggregated rankings.
    output_path : Path
        Path to output parquet file.
    """
    # Ensure correct dtypes per schema
    aggregated_df = aggregated_df.astype(
        {
            "query_id": "string",
            "doc_id": "string",
            "rank": "int32",
            "agg_score": "float64",
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated_df.to_parquet(output_path, index=False)


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

    # Parse methods
    methods: list[str] = [m.strip() for m in args.methods.split(",")]

    # Setup paths
    data_dir: Path = args.data_dir.resolve()
    phase3_dir: Path = data_dir / "phase3_reranking"
    output_dir: Path = data_dir / "phase4_rerank_aggregation" / args.dataset

    logger.info("Phase 4: Reranker Rank Aggregation")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Methods: {methods}")
    logger.info(f"RRF k: {args.rrf_k}")

    # Load Phase 3 rankings
    logger.info("Loading Phase 3 rankings...")
    try:
        rankings_df, reranker_names = load_phase3_rankings(phase3_dir, args.dataset)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    num_queries: int = rankings_df["query_id"].nunique()
    logger.info(
        f"Loaded rankings: {len(rankings_df):,} rows, "
        f"{num_queries} queries, "
        f"{len(reranker_names)} rerankers"
    )
    logger.info(f"Rerankers: {reranker_names}")

    # Track metadata
    metadata: dict[str, Any] = {
        "phase": 4,
        "phase_name": "rerank_aggregation",
        "dataset": args.dataset,
        "rerankers": reranker_names,
        "num_queries": num_queries,
        "methods": {},
        "rrf_k": args.rrf_k,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    # Run aggregation for each method
    success_count: int = 0
    for method in methods:
        output_path: Path = output_dir / f"aggregated_{method}.parquet"

        if output_path.exists() and not args.force:
            logger.info(
                f"Skipping {method} (output exists, use --force to override)"
            )
            metadata["methods"][method] = {"status": "skipped"}
            continue

        logger.info(f"Running {method} aggregation...")
        start_time: datetime = datetime.now(UTC)

        try:
            aggregated_df: pd.DataFrame = run_aggregation(
                rankings_df,
                method=method,
                rrf_k=args.rrf_k,
                show_progress=True,
            )

            save_aggregated_rankings(aggregated_df, output_path)

            elapsed: float = (datetime.now(UTC) - start_time).total_seconds()
            logger.info(
                f"{method}: {len(aggregated_df):,} aggregated rankings in {elapsed:.1f}s"
            )

            metadata["methods"][method] = {
                "status": "completed",
                "num_rankings": len(aggregated_df),
                "elapsed_seconds": elapsed,
                "output_file": str(output_path.name),
            }
            success_count += 1

        except Exception as e:
            logger.exception(f"Failed to run {method}: {e}")
            metadata["methods"][method] = {
                "status": "failed",
                "error": str(e),
            }

    # Save metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path: Path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Phase 4 complete. {success_count}/{len(methods)} methods succeeded."
    )
    logger.info(f"Outputs in: {output_dir}")

    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
