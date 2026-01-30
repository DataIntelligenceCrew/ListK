#!/usr/bin/env python3
"""
Compute Recall@K and NDCG@K for LOTUS and Tournament baselines.

This script computes metrics for LOTUS (Zephyr-7B, QWEN3-8B) and Tournament Top-K
methods against the LLM-as-a-judge ground truth rankings.

Data Sources:
    - LOTUS Zephyr-7B: lotusk/combined_z7b/bier_lotus_semtopk_result.csv
    - LOTUS QWEN3-8B: lotusk/qwen_data/bier_lotus_semtopk_result.csv
    - Tournament Top-K: tourk5000/bier_result_unsorted_tour_10_25.csv
    - Ground Truth: llm-topk-gt/data/phase7_combined_rankings/scifact/{qid}.parquet

Ground Truth Format:
    Parquet files with columns: doc_id, rank, source, source_rank
    - doc_id: Document identifier
    - rank: Position in ground truth ranking (1 = best)

Result File Format (CSV):
    - qid: Query ID from LOTUS/Tournament results
    - did: Stringified list of document IDs (e.g., "['doc1', 'doc2', ...]")
    - time: Execution time in seconds

Metrics Computed:
    - Recall@K: Fraction of top-K ground truth documents retrieved in top-K results
    - NDCG@K: Normalized Discounted Cumulative Gain at K

Usage:
    python -m src.scripts.compute_baseline_metrics [--output-dir DIR] [--k K]

Output:
    - Prints per-query metrics and aggregated statistics
    - Saves metrics CSV files to output directory (default: llm_gt_metrics/):
        - lotus_zephyr_metrics_llm_{k}_{num_queries}.csv (per-query)
        - lotus_zephyr_metrics_llm_agg_{k}_{num_queries}.csv (aggregated)
        - lotus_qwen_metrics_llm_{k}_{num_queries}.csv
        - lotus_qwen_metrics_llm_agg_{k}_{num_queries}.csv
        - tournament_metrics_llm_{k}_{num_queries}.csv
        - tournament_metrics_llm_agg_{k}_{num_queries}.csv

Example:
    # Compute metrics with default settings (K=10)
    python -m src.scripts.compute_baseline_metrics

    # Compute metrics for K=20 and save to custom directory
    python -m src.scripts.compute_baseline_metrics --output-dir ./results --k 20

Author: Auto-generated
Date: 2025-01-23
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

# Default paths (relative to project root)
LOTUS_ZEPHYR_PATH = Path("data/raw/lotusk/combined_z7b/bier_lotus_semtopk_result.csv")
LOTUS_QWEN_PATH = Path("data/raw/lotusk/qwen_data/bier_lotus_semtopk_result.csv")
POINTWISE_ZEPHYR_PATH = Path("data/raw/bier_pointwise/zephyr/bier_lotus_map_result.csv")
POINTWISE_QWEN_PATH = Path("data/raw/bier_pointwise/qwen/bier_lotus_map_result_q.csv")
TOURNAMENT_PATH = Path("data/raw/tourk5000/bier_result_unsorted_tour_10_25.csv")
GROUND_TRUTH_DIR = Path("data/derived/llm-topk-gt/data/phase7_combined_rankings/scifact")

# Canonical output directory for computed LLM ground truth metrics
DEFAULT_OUTPUT_DIR = Path("data/derived/llm_gt_metrics")

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
    """
    Parse a stringified list of document IDs.

    Args:
        raw: String representation of a list, e.g., "['doc1', 'doc2']"

    Returns:
        List of document ID strings.
    """
    try:
        parsed = ast.literal_eval(raw)
        return [str(doc_id) for doc_id in parsed]
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Failed to parse document ID list: {raw}") from e


def load_lotus_results(path: Path) -> list[QueryResult]:
    """
    Load LOTUS semantic top-K results from CSV.

    Expected CSV format:
        q,qid,time,did,stats

    Args:
        path: Path to the LOTUS result CSV file.

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


def load_tournament_results(path: Path) -> list[QueryResult]:
    """
    Load Tournament Top-K results from CSV.

    Expected CSV format:
        q,qid,did,time

    Args:
        path: Path to the Tournament result CSV file.

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
    """
    Load ground truth ranking for a query from parquet file.

    Args:
        qid: Query ID.
        gt_dir: Directory containing ground truth parquet files.

    Returns:
        List of document IDs in ground truth order (best first).
    """
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
    """
    Compute Recall@K.

    Recall@K = |retrieved_top_k ∩ relevant_top_k| / k

    Args:
        retrieved: List of retrieved document IDs in ranked order.
        relevant: List of relevant document IDs in ground truth order.
        k: Number of top documents to consider.

    Returns:
        Recall@K score between 0 and 1.
    """
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
    """
    Compute Discounted Cumulative Gain at K.

    DCG@K = Σ(i=1 to k) rel_i / log2(i + 1)

    Args:
        ranking: List of document IDs in ranked order.
        relevance: Dict mapping doc_id to relevance score.
        k: Number of positions to consider.

    Returns:
        DCG@K score.
    """
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
    """
    Compute Normalized Discounted Cumulative Gain at K.

    NDCG@K = DCG@K / IDCG@K

    Args:
        retrieved: List of retrieved document IDs in ranked order.
        relevant: List of relevant document IDs in ground truth order.
        k: Number of top documents to consider.
        k_rrf: If provided, uses RRF-style graded relevance.

    Returns:
        NDCG@K score between 0 and 1.
    """
    rel_map = build_relevance_map(relevant, k, k_rrf)

    dcg = compute_dcg(retrieved, rel_map, k)
    # IDCG: sort relevance scores descending and place in ideal positions
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
    """
    Compute metrics for all queries of a method.

    Args:
        results: List of QueryResult objects.
        gt_dir: Directory containing ground truth parquet files.
        k: K value for Recall@K and NDCG@K.
        method_name: Name of the method (for logging).
        k_rrf: If provided, uses RRF-style graded relevance for NDCG.

    Returns:
        Tuple of (list of QueryMetrics, DataFrame with results).
    """
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
            "Recall@" + str(k): recall,
            "NDCG@" + str(k): ndcg,
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
    """
    Save metrics DataFrame to CSV.

    Args:
        df: DataFrame with per-query metrics.
        method_name: Name of the method.
        k: K value used for metrics.
        output_dir: Directory to save the file.

    Returns:
        Path to the saved file.
    """
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
        description="Compute Recall@K and NDCG@K for LOTUS and Tournament baselines."
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
        help="Ground truth directory (default: llm-topk-gt/data/phase7_combined_rankings/scifact)",
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

    # Process each method
    methods = [
        ("lotus_zephyr", LOTUS_ZEPHYR_PATH, load_lotus_results),
        ("lotus_qwen", LOTUS_QWEN_PATH, load_lotus_results),
        ("pointwise_zephyr", POINTWISE_ZEPHYR_PATH, load_lotus_results),
        ("pointwise_qwen", POINTWISE_QWEN_PATH, load_lotus_results),
        ("tournament", TOURNAMENT_PATH, load_tournament_results),
    ]

    all_results = {}

    for method_name, data_path, loader_func in methods:
        if not data_path.exists():
            print(f"\n[ERROR] Data file not found: {data_path}")
            continue

        results = loader_func(data_path)
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
    print(f"{'Method':<20} {'Recall@'+str(k):>12} {'NDCG@'+str(k):>12} {'Time (s)':>12}")
    print("-" * 58)
    for method_name, stats in all_results.items():
        print(f"{method_name:<20} {stats['recall']:>12.4f} {stats['ndcg']:>12.4f} {stats['time']:>12.2f}")


if __name__ == "__main__":
    main()
