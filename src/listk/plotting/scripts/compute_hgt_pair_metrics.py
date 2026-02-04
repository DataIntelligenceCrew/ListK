#!/usr/bin/env python3
"""
Compute Recall@K and NDCG@K for HGT Pair data (pairwise quicksort final sort).

This script computes metrics for the hgt_pair data where pairwise quicksort
replaces LMPQsort for the final sorting step. Results are evaluated against
the LLM-as-a-judge ground truth rankings.

Data Sources:
    - HGT Pair data: data/raw/hgt_pair/{l1,l2,l5,l10,l15}/
    - Ground Truth: llm-topk-gt/data/phase7_combined_rankings/scifact/{qid}.parquet

Usage:
    python -m src.scripts.compute_hgt_pair_metrics [--output-dir DIR] [--k K]

Output:
    Saves metrics CSV files to output directory (default: data/derived/hgt_pair_llm_metrics/):
        - hgt_pair_l{L}_metrics_llm_{k}_{num_queries}.csv (per-query)
        - hgt_pair_l{L}_metrics_llm_agg_{k}_{num_queries}.csv (aggregated)
"""

import argparse
import ast
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# =============================================================================
# Configuration
# =============================================================================

# HGT Pair data paths (relative to project root)
HGT_PAIR_BASE = Path("data/raw/hgt_pair")
HGT_PAIR_PATHS = {
    "l1": HGT_PAIR_BASE / "l1",
    "l2": HGT_PAIR_BASE / "l2",
    "l5": HGT_PAIR_BASE / "l5",
    "l10": HGT_PAIR_BASE / "l10",
    "l15": HGT_PAIR_BASE / "l15",
}

# File name patterns for each L value
# Format: bier_sorted_10_{p}_{x}_{L}_{W}.csv where W=2 for pairwise
HGT_PAIR_FILES = {
    "l1": "bier_sorted_10_16_2_1_2.csv",
    "l2": "bier_sorted_10_16_2_2_2.csv",
    "l5": "bier_sorted_10_16_2_5_2.csv",
    "l10": "bier_sorted_10_16_2_10_2.csv",
    "l15": "bier_sorted_10_16_2_15_2.csv",
}

GROUND_TRUTH_DIR = Path("data/derived/llm-topk-gt/data/phase7_combined_rankings/scifact")

# Canonical output directory for computed metrics
DEFAULT_OUTPUT_DIR = Path("data/derived/hgt_pair_llm_metrics")

# Default metric parameters
DEFAULT_K = 10


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class QueryResult:
    """Result for a single query."""
    qid: int
    doc_ids: list[str]
    time: float


@dataclass
class QueryMetrics:
    """Computed metrics for a single query."""
    qid: int
    recall: float
    ndcg: float
    time: float


# =============================================================================
# Data Loading Functions
# =============================================================================

def parse_doc_id_list(raw: str) -> list[str]:
    """Parse a stringified list of document IDs."""
    try:
        parsed = ast.literal_eval(raw)
        return [str(doc_id) for doc_id in parsed]
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Failed to parse document ID list: {raw}") from e


def load_hgt_pair_results(path: Path) -> list[QueryResult]:
    """
    Load HGT pair sorted results from CSV.

    Expected CSV format:
        q,qid,did,time

    Args:
        path: Path to the result CSV file.

    Returns:
        List of QueryResult objects.
    """
    df = pd.read_csv(path)
    results = []

    for _, row in df.iterrows():
        qid = int(row["qid"])
        doc_ids = parse_doc_id_list(row["did"])
        time = float(row["time"])
        results.append(QueryResult(qid=qid, doc_ids=doc_ids, time=time))

    return results


def load_ground_truth(qid: int, gt_dir: Path) -> list[str]:
    """Load ground truth ranking for a query from parquet file."""
    gt_path = gt_dir / f"{qid}.parquet"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    table = pq.read_table(gt_path)
    df = table.to_pandas()

    # Sort by rank (1 = best) and extract doc_ids
    df = df.sort_values("rank")
    return [str(doc_id) for doc_id in df["doc_id"].tolist()]


# =============================================================================
# Metric Computation Functions
# =============================================================================

def compute_recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Compute Recall@K."""
    retrieved_top_k = set(retrieved[:k])
    relevant_top_k = set(relevant[:k])
    intersection = retrieved_top_k & relevant_top_k
    return len(intersection) / k


def build_relevance_map(
    relevant: list[str],
    k: int,
    k_rrf: float | None = None,
) -> dict[str, float]:
    """
    Build a document-to-relevance mapping from ranked ground truth.

    Args:
        relevant: Ground truth document IDs in ranked order (best first).
        k: Number of top documents to consider.
        k_rrf: If provided, uses RRF-style graded relevance:
               rel(r) = 1 / (k_rrf + r) where r is 1-indexed rank.
               If None, uses binary relevance (1.0 for all top-K docs).

    Returns:
        Dict mapping doc_id to relevance score.
    """
    top_k = relevant[:k]
    if k_rrf is None:
        return {doc_id: 1.0 for doc_id in top_k}
    return {doc_id: 1.0 / (k_rrf + r) for r, doc_id in enumerate(top_k, start=1)}


def compute_dcg(ranking: list[str], relevance: dict[str, float], k: int) -> float:
    """Compute Discounted Cumulative Gain at K."""
    dcg = 0.0
    for i, doc_id in enumerate(ranking[:k]):
        rel = relevance.get(doc_id, 0.0)
        if rel > 0:
            dcg += rel / math.log2(i + 2)  # +2 because i is 0-indexed
    return dcg


def compute_ndcg_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int,
    k_rrf: float | None = None,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at K."""
    rel_map = build_relevance_map(relevant, k, k_rrf)

    dcg = compute_dcg(retrieved, rel_map, k)
    ideal_scores = sorted(rel_map.values(), reverse=True)
    idcg = sum(s / math.log2(i + 2) for i, s in enumerate(ideal_scores))

    if idcg == 0:
        return 0.0
    return dcg / idcg


# =============================================================================
# Main Processing Functions
# =============================================================================

def compute_metrics_for_method(
    results: list[QueryResult],
    gt_dir: Path,
    k: int,
    method_name: str,
    k_rrf: float | None = None,
) -> tuple[list[QueryMetrics], pd.DataFrame]:
    """Compute metrics for all queries of a method."""
    metrics_list = []
    rows = []

    print(f"\n{'='*60}")
    print(f"Computing metrics for: {method_name}")
    print(f"{'='*60}")
    print(f"{'QID':>6} {'Recall@'+str(k):>12} {'NDCG@'+str(k):>12} {'Time (s)':>12}")
    print("-" * 48)

    for result in results:
        try:
            gt_ranking = load_ground_truth(result.qid, gt_dir)
        except FileNotFoundError as e:
            print(f"  [WARN] Skipping query {result.qid}: {e}")
            continue

        recall = compute_recall_at_k(result.doc_ids, gt_ranking, k)
        ndcg = compute_ndcg_at_k(result.doc_ids, gt_ranking, k, k_rrf=k_rrf)

        metrics = QueryMetrics(
            qid=result.qid,
            recall=recall,
            ndcg=ndcg,
            time=result.time,
        )
        metrics_list.append(metrics)

        print(f"{result.qid:>6} {recall:>12.4f} {ndcg:>12.4f} {result.time:>12.2f}")

        rows.append({
            "qid": result.qid,
            f"Recall@{k}": recall,
            f"NDCG@{k}": ndcg,
            "time": result.time,
        })

    # Compute aggregated statistics
    if metrics_list:
        mean_recall = np.mean([m.recall for m in metrics_list])
        mean_ndcg = np.mean([m.ndcg for m in metrics_list])
        mean_time = np.mean([m.time for m in metrics_list])

        print("-" * 48)
        print(f"{'MEAN':>6} {mean_recall:>12.4f} {mean_ndcg:>12.4f} {mean_time:>12.2f}")
        print(f"{'STD':>6} {np.std([m.recall for m in metrics_list]):>12.4f} "
              f"{np.std([m.ndcg for m in metrics_list]):>12.4f} "
              f"{np.std([m.time for m in metrics_list]):>12.2f}")

    df = pd.DataFrame(rows)
    return metrics_list, df


def save_metrics(
    df: pd.DataFrame,
    method_name: str,
    k: int,
    output_dir: Path,
) -> Path:
    """Save metrics DataFrame to CSV."""
    num_queries = len(df)
    filename = f"{method_name}_metrics_llm_{k}_{num_queries}.csv"
    output_path = output_dir / filename

    # Also save aggregated metrics (single row with means)
    agg_row = {
        f"Recall@{k}": df[f"Recall@{k}"].mean(),
        f"NDCG@{k}": df[f"NDCG@{k}"].mean(),
        "time": df["time"].mean(),
    }
    agg_df = pd.DataFrame([agg_row])
    agg_filename = f"{method_name}_metrics_llm_agg_{k}_{num_queries}.csv"
    agg_path = output_dir / agg_filename

    df.to_csv(output_path, index=False)
    agg_df.to_csv(agg_path, index=False)

    print(f"\nSaved: {output_path}")
    print(f"Saved: {agg_path}")

    return output_path


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compute Recall@K and NDCG@K for HGT Pair data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save output files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"K value for Recall@K and NDCG@K (default: {DEFAULT_K})",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=GROUND_TRUTH_DIR,
        help="Ground truth directory",
    )
    parser.add_argument(
        "--k-rrf",
        type=float,
        default=None,
        help="RRF constant for graded NDCG relevance (e.g. 60). Default: binary relevance.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    k = args.k
    gt_dir = args.gt_dir

    print(f"Configuration:")
    print(f"  K: {k}")
    print(f"  Ground Truth Dir: {gt_dir}")
    print(f"  Output Dir: {output_dir}")

    all_results = {}

    # Process each L value
    for l_key, l_path in HGT_PAIR_PATHS.items():
        data_file = l_path / HGT_PAIR_FILES[l_key]

        if not data_file.exists():
            print(f"\n[ERROR] Data file not found: {data_file}")
            continue

        results = load_hgt_pair_results(data_file)
        method_name = f"hgt_pair_{l_key}"

        metrics_list, df = compute_metrics_for_method(
            results, gt_dir, k, method_name, k_rrf=args.k_rrf
        )
        save_metrics(df, method_name, k, output_dir)

        # Store for summary
        if metrics_list:
            all_results[method_name] = {
                "recall": np.mean([m.recall for m in metrics_list]),
                "ndcg": np.mean([m.ndcg for m in metrics_list]),
                "time": np.mean([m.time for m in metrics_list]),
            }

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'Recall@'+str(k):>12} {'NDCG@'+str(k):>12} {'Time (s)':>12}")
    print("-" * 63)
    for method_name, stats in all_results.items():
        print(f"{method_name:<25} {stats['recall']:>12.4f} {stats['ndcg']:>12.4f} {stats['time']:>12.2f}")


if __name__ == "__main__":
    main()