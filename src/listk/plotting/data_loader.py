"""
Data loading utilities for experiment results.

This module provides functions to load and parse experiment data from
CSV, JSON, and Parquet files with proper typing and error handling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd


@dataclass
class QueryResult:
    """
    Represents retrieval results for a single query.

    Attributes:
        query_id: Unique identifier for the query.
        doc_ids: List of retrieved document IDs.
        times: List of execution times (one per query in the batch).
    """
    query_id: int
    doc_ids: list[str]
    times: list[float]


@dataclass
class TFilterResult:
    """
    Represents tournament filter results for a single query.

    Attributes:
        query_index: Index of the query in the experiment.
        doc_count: Number of documents retained after filtering.
        time: Execution time in seconds.
        doc_ids: List of retained document IDs.
    """
    query_index: int
    doc_count: int
    time: float
    doc_ids: list[str]


def parse_string_list(raw: str) -> list[str]:
    """
    Parse a stringified list from CSV into a Python list.

    Handles format like: "['doc1', 'doc2', 'doc3']"

    Args:
        raw: String representation of a list.

    Returns:
        Parsed list of strings.

    Example:
        >>> parse_string_list("['a', 'b', 'c']")
        ['a', 'b', 'c']
    """
    cleaned = raw.replace("[", "").replace("]", "")
    cleaned = cleaned.replace(" ", "").replace("'", "")
    return cleaned.split(",") if cleaned else []


def load_results(path: Path | str) -> list[QueryResult]:
    """
    Load experiment results from a CSV file.

    Expected columns: qid, did, time

    Args:
        path: Path to the CSV file.

    Returns:
        List of QueryResult objects, one per query.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If required columns are missing.
    """
    path = Path(path)
    df = pd.read_csv(path)

    results = []
    times = df["time"].tolist()

    for idx, row in df.iterrows():
        results.append(QueryResult(
            query_id=row["qid"],
            doc_ids=parse_string_list(row["did"]),
            times=times,
        ))

    return results


def load_sort_results(path: Path | str) -> list[QueryResult]:
    """
    Load sorted experiment results from a CSV file.

    Expected columns: qid, ids, time (note: 'ids' instead of 'did')

    Args:
        path: Path to the CSV file.

    Returns:
        List of QueryResult objects.
    """
    path = Path(path)
    df = pd.read_csv(path)

    results = []
    times = df["time"].tolist()

    for idx, row in df.iterrows():
        results.append(QueryResult(
            query_id=row["qid"],
            doc_ids=parse_string_list(row["ids"]),
            times=times,
        ))

    return results


def load_tfilter_results(
    tfilter_path: Path | str,
    qid_path: Path | str,
) -> list[QueryResult]:
    """
    Load tournament filter results, combining with query IDs from another file.

    Args:
        tfilter_path: Path to tournament filter results CSV.
        qid_path: Path to file containing query IDs.

    Returns:
        List of QueryResult objects with combined data.
    """
    tfilter_df = pd.read_csv(Path(tfilter_path))
    qid_df = pd.read_csv(Path(qid_path))

    results = []
    times = qid_df["time"].tolist()

    for idx in range(len(tfilter_df)):
        results.append(QueryResult(
            query_id=qid_df["qid"].iloc[idx],
            doc_ids=parse_string_list(tfilter_df["ids"].iloc[idx]),
            times=times,
        ))

    return results


def load_ground_truth(query_id: int, gt_path: Path | str) -> list[str]:
    """
    Load ground truth document rankings for a query.

    Args:
        query_id: The query ID to load rankings for.
        gt_path: Directory containing parquet files.

    Returns:
        List of document IDs in ranked order.

    Raises:
        FileNotFoundError: If the parquet file does not exist.
    """
    gt_path = Path(gt_path)
    parquet_path = gt_path / f"{query_id}.parquet"
    df = pd.read_parquet(parquet_path)
    return df["doc_id"].tolist()


def load_metrics(path: Path | str) -> pd.DataFrame:
    """
    Load pre-computed metrics from a CSV file.

    Expected columns: NDCG@10, MAP@10, Recall@10, P@10, time

    Args:
        path: Path to the metrics CSV file.

    Returns:
        DataFrame with metrics.
    """
    return pd.read_csv(Path(path))


def load_metrics_with_fallback(
    base_path: Path,
    filename: str,
    fallback_subdirs: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load metrics with fallback to alternative subdirectories.

    Tries the base path first, then each fallback subdirectory in order.

    Args:
        base_path: Primary directory to search.
        filename: Name of the metrics file.
        fallback_subdirs: List of subdirectory names to try if not found.

    Returns:
        DataFrame with metrics from the first found file.

    Raises:
        FileNotFoundError: If no metrics file is found in any location.
    """
    fallback_subdirs = fallback_subdirs or []
    paths_to_try = [base_path / filename]
    paths_to_try.extend(base_path / subdir / filename for subdir in fallback_subdirs)

    for path in paths_to_try:
        if path.exists():
            return pd.read_csv(path)

    tried = ", ".join(str(p) for p in paths_to_try)
    raise FileNotFoundError(f"Metrics file not found. Tried: {tried}")


def get_metric_value(df: pd.DataFrame, column: str) -> float:
    """
    Extract a single metric value from a DataFrame.

    Args:
        df: DataFrame containing metrics.
        column: Name of the column to extract.

    Returns:
        The first value in the specified column.
    """
    return df[column].iloc[0]


def iter_px_combinations(
    pivot_values: list[int],
    require_p_ge_x: bool = True,
) -> Iterator[tuple[int, int]]:
    """
    Iterate over valid (p, x) parameter combinations.

    Args:
        pivot_values: List of values for both p and x.
        require_p_ge_x: If True, only yield combinations where p >= x.

    Yields:
        Tuples of (p, x) values.
    """
    for p in pivot_values:
        for x in pivot_values:
            if not require_p_ge_x or p >= x:
                yield (p, x)


def iter_px_combinations_split(
    pivot_values: list[int],
    early_stop_path: Path,
    no_early_stop_path: Path,
) -> Iterator[tuple[int, int, Path]]:
    """
    Iterate over (p, x) combinations with appropriate paths.

    For p == x, uses early_stop_path; for p > x, uses no_early_stop_path.

    Args:
        pivot_values: List of values for both p and x.
        early_stop_path: Path for diagonal (p == x) experiments.
        no_early_stop_path: Path for off-diagonal (p > x) experiments.

    Yields:
        Tuples of (p, x, appropriate_path).
    """
    for p in pivot_values:
        for x in pivot_values:
            if p > x:
                yield (p, x, no_early_stop_path)
            elif p == x:
                yield (p, x, early_stop_path)
