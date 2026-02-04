"""
Metric calculation functions for retrieval evaluation.

This module provides pure functions for computing standard IR metrics
including Recall, DCG, and NDCG.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import QueryResult, load_ground_truth, load_results
from .config import PIVOT_VALUES, METRICS_FALLBACK_SUBDIRS


@dataclass
class AggregatedStats:
    """
    Aggregated statistics for an experiment configuration.

    Attributes:
        p: Pivot count parameter.
        x: Expansion factor parameter.
        mean_time: Mean execution time across queries.
        mean_recall: Mean recall across queries.
    """
    p: int
    x: int
    mean_time: float
    mean_recall: float


def calc_recall(
    retrieved: list[str],
    relevant: list[str],
    k: int | None = None,
) -> float:
    """
    Calculate Recall@K for a single query.

    Recall@K = |retrieved ∩ relevant| / |relevant|

    Args:
        retrieved: List of retrieved document IDs.
        relevant: List of relevant document IDs (ground truth).
        k: Number of top documents to consider. If None, uses len(relevant).

    Returns:
        Recall score between 0 and 1.
    """
    if not relevant:
        return 0.0

    k = k or len(relevant)
    relevant_set = set(relevant[:k])
    retrieved_set = set(retrieved)

    intersection = relevant_set & retrieved_set
    return len(intersection) / len(relevant_set)


def calc_recall_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int,
) -> float:
    """
    Calculate Recall@K using top-K of ground truth.

    This variant divides by K instead of the number of retrieved docs.

    Args:
        retrieved: List of retrieved document IDs.
        relevant: List of relevant document IDs (ground truth, ranked).
        k: Number of top documents to consider from ground truth.

    Returns:
        Recall score between 0 and 1.
    """
    if k == 0:
        return 0.0

    relevant_top_k = set(relevant[:k])
    retrieved_set = set(retrieved)

    intersection = relevant_top_k & retrieved_set
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


def calc_dcg(ranking: list[str], relevance: dict[str, float]) -> float:
    """
    Calculate Discounted Cumulative Gain.

    DCG = Σ rel_i / log2(i + 1) for i in 1..n

    Args:
        ranking: List of document IDs in ranked order.
        relevance: Dict mapping doc_id to relevance score.

    Returns:
        DCG score.
    """
    dcg = 0.0
    for i, doc_id in enumerate(ranking, start=1):
        rel = relevance.get(doc_id, 0.0)
        if rel > 0:
            dcg += rel / math.log2(1 + i)
    return dcg


def calc_idcg(relevance_scores: list[float]) -> float:
    """
    Calculate Ideal Discounted Cumulative Gain.

    IDCG assumes documents are ranked by decreasing relevance.

    Args:
        relevance_scores: List of relevance scores for the relevant documents.

    Returns:
        IDCG score.
    """
    scores = sorted(relevance_scores, reverse=True)
    idcg = 0.0
    for i, score in enumerate(scores, start=1):
        idcg += score / math.log2(1 + i)
    return idcg


def calc_ndcg(
    ranking: list[str],
    relevant: list[str],
    k: int | None = None,
    k_rrf: float | None = None,
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.

    NDCG@K = DCG@K / IDCG@K

    Args:
        ranking: List of document IDs in ranked order.
        relevant: List of relevant document IDs (ground truth, ranked).
        k: Number of top documents to consider. If None, uses len(relevant).
        k_rrf: If provided, uses RRF-style graded relevance.
               If None, uses binary relevance (default).

    Returns:
        NDCG score between 0 and 1.
    """
    if not relevant:
        return 0.0

    k = k or len(relevant)
    rel_map = build_relevance_map(relevant, k, k_rrf)

    dcg = calc_dcg(ranking[:k], rel_map)
    idcg = calc_idcg(list(rel_map.values()))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_recalls_for_results(
    results: list[QueryResult],
    gt_path: Path,
    k: int,
) -> list[float]:
    """
    Compute recall for each query in a result set.

    Args:
        results: List of query results.
        gt_path: Path to ground truth parquet files.
        k: Number of top documents for Recall@K.

    Returns:
        List of recall scores, one per query.
    """
    recalls = []
    for result in results:
        ground_truth = load_ground_truth(result.query_id, gt_path)[:k]
        recall = calc_recall(result.doc_ids, ground_truth)
        recalls.append(recall)
    return recalls


def compute_recalls_tour_for_results(
    results: list[QueryResult],
    gt_path: Path,
    k: int,
) -> list[float]:
    """
    Compute recall for tournament filter results.

    Uses K as the denominator instead of number of retrieved docs.

    Args:
        results: List of query results.
        gt_path: Path to ground truth parquet files.
        k: Number of top documents for Recall@K.

    Returns:
        List of recall scores, one per query.
    """
    recalls = []
    for result in results:
        ground_truth = load_ground_truth(result.query_id, gt_path)[:k]
        recall = calc_recall_at_k(result.doc_ids, ground_truth, k)
        recalls.append(recall)
    return recalls


def compute_ndcg_for_results(
    results: list[QueryResult],
    gt_path: Path,
    k: int,
    k_rrf: float | None = None,
) -> list[float]:
    """
    Compute NDCG for each query in a result set.

    Args:
        results: List of query results.
        gt_path: Path to ground truth parquet files.
        k: Number of top documents for NDCG@K.
        k_rrf: If provided, uses RRF-style graded relevance.

    Returns:
        List of NDCG scores, one per query.
    """
    ndcgs = []
    for result in results:
        ground_truth = load_ground_truth(result.query_id, gt_path)[:k]
        ndcg = calc_ndcg(result.doc_ids[:k], ground_truth, k, k_rrf=k_rrf)
        ndcgs.append(ndcg)
    return ndcgs


def calc_stats_for_px_grid(
    base_path: Path,
    gt_path: Path,
    pivot_values: list[int] | None = None,
    k: int = 10,
    use_precomputed_metrics: bool = False,
) -> pd.DataFrame:
    """
    Calculate statistics for all valid (p, x) parameter combinations.

    Args:
        base_path: Directory containing experiment results.
        gt_path: Path to ground truth parquet files.
        pivot_values: List of p and x values. Defaults to PIVOT_VALUES.
        k: K value for Recall@K.
        use_precomputed_metrics: If True, read recall from metrics files.

    Returns:
        DataFrame with columns: p, x, time, recall.
    """
    pivot_values = pivot_values or PIVOT_VALUES
    datapoints = []

    for p in pivot_values:
        for x in pivot_values:
            if p >= x:
                result_file = base_path / f"bier_result_unsorted_{p}_{x}_25.csv"
                results = load_results(result_file)
                mean_time = np.mean(results[0].times)

                if use_precomputed_metrics:
                    metrics_file = base_path / f"bier_metrics_{p}_{x}_25.csv"
                    metrics_df = pd.read_csv(metrics_file)
                    mean_recall = metrics_df["Recall@10"].iloc[0]
                else:
                    recalls = compute_recalls_for_results(results, gt_path, k)
                    mean_recall = np.mean(recalls)

                datapoints.append([p, x, mean_time, mean_recall])

    return pd.DataFrame(datapoints, columns=["p", "x", "time", "recall"])


def calc_stats_for_px_grid_split(
    no_early_path: Path,
    early_path: Path,
    gt_path: Path,
    pivot_values: list[int] | None = None,
    k: int = 10,
    use_precomputed_metrics: bool = False,
) -> pd.DataFrame:
    """
    Calculate statistics for (p, x) grid with split paths.

    Uses early_path for diagonal (p == x) and no_early_path for off-diagonal.

    Args:
        no_early_path: Path for experiments without early stopping.
        early_path: Path for experiments with early stopping.
        gt_path: Path to ground truth parquet files.
        pivot_values: List of p and x values.
        k: K value for Recall@K.
        use_precomputed_metrics: If True, read recall from metrics files.

    Returns:
        DataFrame with columns: p, x, time, recall.
    """
    pivot_values = pivot_values or PIVOT_VALUES
    datapoints = []

    for p in pivot_values:
        for x in pivot_values:
            if p > x:
                path = no_early_path
            elif p == x:
                path = early_path
            else:
                continue

            result_file = path / f"bier_result_unsorted_{p}_{x}_25.csv"

            # Skip if result file doesn't exist
            if not result_file.exists():
                continue

            results = load_results(result_file)
            mean_time = np.mean(results[0].times)

            if use_precomputed_metrics:
                mean_recall = None

                # Try main metrics file
                metrics_file = path / f"bier_metrics_{p}_{x}_25.csv"
                if metrics_file.exists():
                    metrics_df = pd.read_csv(metrics_file)
                    mean_recall = metrics_df["Recall@10"].iloc[0]
                else:
                    # Try fallback paths
                    for subdir in METRICS_FALLBACK_SUBDIRS:
                        fallback = path / subdir / f"bier_metrics_{p}_{x}_25.csv"
                        if fallback.exists():
                            metrics_df = pd.read_csv(fallback)
                            mean_recall = metrics_df["Recall@10"].iloc[0]
                            break

                # If no metrics file found, calculate from scratch
                if mean_recall is None:
                    recalls = compute_recalls_for_results(results, gt_path, k)
                    mean_recall = np.mean(recalls)
            else:
                recalls = compute_recalls_for_results(results, gt_path, k)
                mean_recall = np.mean(recalls)

            datapoints.append([p, x, mean_time, mean_recall])

    return pd.DataFrame(datapoints, columns=["p", "x", "time", "recall"])
