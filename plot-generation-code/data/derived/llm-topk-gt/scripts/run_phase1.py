#!/usr/bin/env python3
"""Run Phase 1: IR Retrieval.

This script executes the first phase of the pipeline, running multiple
retrievers on BEIR datasets and saving rankings to parquet files.

Usage
-----
    # Run specific retrievers
    python scripts/run_phase1.py --dataset scifact --retrievers bm25,e5

    # Run all retrievers
    python scripts/run_phase1.py --dataset scifact --retrievers all

    # Custom top-N and data directory
    python scripts/run_phase1.py --dataset scifact --retrievers bm25 --top-n 500 --data-dir ./data

Examples
--------
    # Quick test with BM25 only
    python scripts/run_phase1.py --dataset scifact --retrievers bm25 -v

    # Full run with all retrievers
    python scripts/run_phase1.py --dataset scifact --retrievers all

    # Force re-run even if output exists
    python scripts/run_phase1.py --dataset scifact --retrievers bm25 --force
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
from src.retrieval import RETRIEVER_REGISTRY, get_retriever


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
        description="Run Phase 1: IR Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run BM25 on SciFact
    python scripts/run_phase1.py --dataset scifact --retrievers bm25

    # Run all retrievers
    python scripts/run_phase1.py --dataset scifact --retrievers all

    # Run dense retrievers only
    python scripts/run_phase1.py --dataset scifact --retrievers e5,bge
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (scifact, scidocs)",
    )

    parser.add_argument(
        "--retrievers",
        type=str,
        required=True,
        help=f"Comma-separated retriever names or 'all'. Available: {list(RETRIEVER_REGISTRY.keys())}",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=1000,
        help="Number of documents to retrieve per query (default: 1000)",
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
        "--list-retrievers",
        action="store_true",
        help="List available retrievers and exit",
    )

    return parser.parse_args()


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

    # Handle --list-retrievers flag
    if args.list_retrievers:
        print("Available retrievers:")
        for name in sorted(RETRIEVER_REGISTRY.keys()):
            print(f"  - {name}")
        return 0

    # Determine retrievers to run
    retriever_names: list[str]
    if args.retrievers.lower() == "all":
        retriever_names = list(RETRIEVER_REGISTRY.keys())
    else:
        retriever_names = [r.strip() for r in args.retrievers.split(",")]

    # Validate retriever names
    invalid: list[str] = [r for r in retriever_names if r not in RETRIEVER_REGISTRY]
    if invalid:
        logger.error(
            f"Unknown retrievers: {invalid}. "
            f"Available: {list(RETRIEVER_REGISTRY.keys())}"
        )
        return 1

    # Setup paths
    data_dir: Path = args.data_dir.resolve()
    output_dir: Path = data_dir / "phase1_retrieval" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Retrievers: {retriever_names}")
    logger.info(f"Top-N: {args.top_n}")

    # Load dataset
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
        "dataset": args.dataset,
        "top_n": args.top_n,
        "num_documents": len(dataset.corpus),
        "num_queries": len(dataset.queries),
        "retrievers": {},
        "timestamp": datetime.now(UTC).isoformat(),
    }

    # Run each retriever
    success_count: int = 0
    for retriever_name in retriever_names:
        output_path: Path = output_dir / f"{retriever_name}.parquet"

        if output_path.exists() and not args.force:
            logger.info(f"Skipping {retriever_name} (output exists, use --force to override)")
            metadata["retrievers"][retriever_name] = {"status": "skipped"}
            continue

        logger.info(f"Running retriever: {retriever_name}")
        start_time: datetime = datetime.now(UTC)

        try:
            retriever = get_retriever(retriever_name, top_n=args.top_n)
            rankings: list[RankingEntry] = retriever.run(dataset)

            save_rankings(rankings, output_path)

            elapsed: float = (datetime.now(UTC) - start_time).total_seconds()
            num_queries: int = len(set(r.query_id for r in rankings))
            logger.info(
                f"{retriever_name}: {len(rankings)} rankings for {num_queries} queries in {elapsed:.1f}s"
            )

            metadata["retrievers"][retriever_name] = {
                "status": "completed",
                "num_rankings": len(rankings),
                "num_queries": num_queries,
                "elapsed_seconds": elapsed,
                "output_file": str(output_path.name),
            }

            # Clear index to free memory before next retriever
            retriever.clear_index()
            success_count += 1

        except Exception as e:
            logger.exception(f"Failed to run {retriever_name}: {e}")
            metadata["retrievers"][retriever_name] = {
                "status": "failed",
                "error": str(e),
            }

    # Save metadata
    metadata_path: Path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Phase 1 complete. {success_count}/{len(retriever_names)} retrievers succeeded.")
    logger.info(f"Outputs in: {output_dir}")

    # Return success only if at least one retriever succeeded
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
