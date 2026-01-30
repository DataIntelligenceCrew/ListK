#!/usr/bin/env python
"""Phase 2: IR Rank Aggregation

This script aggregates rankings from multiple IR retrievers (Phase 1) into
unified rankings using various rank aggregation algorithms.

Usage:
    python scripts/run_phase2.py --dataset scifact --methods rrf,borda,copeland,schulze
    python scripts/run_phase2.py --dataset scifact --methods schulze --rrf_k 60

Input:
    data/phase1_retrieval/{dataset}/*.parquet

Output:
    data/phase2_ir_aggregation/{dataset}/aggregated_{method}.parquet
    data/phase2_ir_aggregation/{dataset}/metadata.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.aggregation import get_aggregator


def load_phase1_rankings(
    phase1_dir: Path,
    dataset: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Load all retriever rankings from Phase 1.

    Parameters
    ----------
    phase1_dir : Path
        Base directory for Phase 1 outputs.
    dataset : str
        Dataset name (e.g., "scifact").

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Combined DataFrame with all rankings (with 'ranker' column),
        and list of ranker names.
    """
    dataset_dir = phase1_dir / dataset
    parquet_files = list(dataset_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {dataset_dir}. "
            f"Run Phase 1 first."
        )

    dfs: list[pd.DataFrame] = []
    ranker_names: list[str] = []

    for pq_file in sorted(parquet_files):
        ranker_name = pq_file.stem
        ranker_names.append(ranker_name)

        df = pd.read_parquet(pq_file)
        df["ranker"] = ranker_name
        dfs.append(df)

        print(f"  Loaded {ranker_name}: {len(df):,} rows")

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df, ranker_names


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
        Combined rankings DataFrame with 'ranker' column.
    method : str
        Aggregation method: "rrf", "borda", "copeland", or "schulze".
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
        ranker_column="ranker",
        show_progress=show_progress,
    )


def save_aggregated_rankings(
    aggregated_df: pd.DataFrame,
    output_dir: Path,
    method: str,
) -> Path:
    """Save aggregated rankings to parquet.

    Parameters
    ----------
    aggregated_df : pd.DataFrame
        Aggregated rankings.
    output_dir : Path
        Output directory.
    method : str
        Aggregation method name.

    Returns
    -------
    Path
        Path to saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"aggregated_{method}.parquet"

    # Ensure correct dtypes
    aggregated_df = aggregated_df.astype({
        "query_id": "string",
        "doc_id": "string",
        "rank": "int32",
        "agg_score": "float64",
    })

    aggregated_df.to_parquet(output_path, index=False)
    return output_path


def save_metadata(
    output_dir: Path,
    dataset: str,
    rankers: list[str],
    methods: list[str],
    rrf_k: int,
) -> None:
    """Save Phase 2 metadata.

    Parameters
    ----------
    output_dir : Path
        Output directory.
    dataset : str
        Dataset name.
    rankers : list[str]
        List of retriever names used.
    methods : list[str]
        List of aggregation methods used.
    rrf_k : int
        RRF smoothing constant.
    """
    metadata = {
        "phase": 2,
        "phase_name": "ir_aggregation",
        "dataset": dataset,
        "rankers": rankers,
        "methods": methods,
        "rrf_k": rrf_k,
        "timestamp": datetime.utcnow().isoformat(),
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    """Main entry point for Phase 2."""
    parser = argparse.ArgumentParser(
        description="Phase 2: IR Rank Aggregation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., scifact)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="rrf,borda,copeland,schulze",
        help="Comma-separated list of aggregation methods (default: rrf,borda,copeland,schulze)",
    )
    parser.add_argument(
        "--rrf_k",
        type=int,
        default=60,
        help="RRF smoothing constant k (default: 60)",
    )
    parser.add_argument(
        "--phase1_dir",
        type=str,
        default="data/phase1_retrieval",
        help="Phase 1 output directory (default: data/phase1_retrieval)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/phase2_ir_aggregation",
        help="Output directory (default: data/phase2_ir_aggregation)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if output exists",
    )

    args = parser.parse_args()

    phase1_dir = Path(args.phase1_dir)
    output_dir = Path(args.output_dir) / args.dataset
    methods = [m.strip() for m in args.methods.split(",")]

    print(f"Phase 2: IR Rank Aggregation")
    print(f"  Dataset: {args.dataset}")
    print(f"  Methods: {methods}")
    print(f"  RRF k: {args.rrf_k}")
    print()

    # Load Phase 1 rankings
    print("Loading Phase 1 rankings...")
    rankings_df, ranker_names = load_phase1_rankings(phase1_dir, args.dataset)
    print(f"  Total rows: {len(rankings_df):,}")
    print(f"  Rankers: {ranker_names}")
    print()

    # Run aggregation for each method
    for method in methods:
        output_path = output_dir / f"aggregated_{method}.parquet"

        if output_path.exists() and not args.force:
            print(f"Skipping {method} (output exists, use --force to overwrite)")
            continue

        print(f"Running {method} aggregation...")
        aggregated_df = run_aggregation(
            rankings_df,
            method=method,
            rrf_k=args.rrf_k,
            show_progress=True,
        )

        # Save results
        saved_path = save_aggregated_rankings(aggregated_df, output_dir, method)
        print(f"  Saved to {saved_path}")
        print(f"  Aggregated rows: {len(aggregated_df):,}")
        print()

    # Save metadata
    save_metadata(output_dir, args.dataset, ranker_names, methods, args.rrf_k)
    print(f"Metadata saved to {output_dir / 'metadata.json'}")
    print("Phase 2 complete!")


if __name__ == "__main__":
    main()
