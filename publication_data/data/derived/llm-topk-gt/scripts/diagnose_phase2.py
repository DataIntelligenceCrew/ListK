#!/usr/bin/env python
"""Phase 2 Diagnostic Script

This script provides diagnostics for Phase 2 IR aggregation results:
1. Rank correlation between aggregated rankings and individual retrievers
2. Human-readable samples of ranked documents for specific "important" queries

Usage:
    python scripts/diagnose_phase2.py --dataset scifact
    python scripts/diagnose_phase2.py --dataset scifact --top_k 5 --sample_count 5
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# Important query IDs for diagnostic inspection (as specified)
IMPORTANT_QUERY_IDS: list[str] = [
    "1", "3", "5", "13", "36", "42", "48", "49", "50", "51",
    "53", "54", "56", "57", "70", "72", "75", "94", "99", "100",
    "113", "115", "118", "124", "127"
]

# Maximum character length for document text display
MAX_DOC_LENGTH: int = 500


def load_phase1_rankings(phase1_dir: Path, dataset: str) -> dict[str, pd.DataFrame]:
    """Load all retriever rankings from Phase 1.

    Parameters
    ----------
    phase1_dir : Path
        Base directory for Phase 1 outputs.
    dataset : str
        Dataset name.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from retriever name to its rankings DataFrame.
    """
    dataset_dir = phase1_dir / dataset
    rankings: dict[str, pd.DataFrame] = {}

    for pq_file in sorted(dataset_dir.glob("*.parquet")):
        retriever_name = pq_file.stem
        rankings[retriever_name] = pd.read_parquet(pq_file)
        print(f"  Loaded {retriever_name}: {len(rankings[retriever_name]):,} rows")

    return rankings


def load_phase2_rankings(phase2_dir: Path, dataset: str) -> dict[str, pd.DataFrame]:
    """Load aggregated rankings from Phase 2.

    Parameters
    ----------
    phase2_dir : Path
        Base directory for Phase 2 outputs.
    dataset : str
        Dataset name.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from aggregation method to its rankings DataFrame.
    """
    dataset_dir = phase2_dir / dataset
    rankings: dict[str, pd.DataFrame] = {}

    for pq_file in sorted(dataset_dir.glob("aggregated_*.parquet")):
        method_name = pq_file.stem.replace("aggregated_", "")
        rankings[method_name] = pd.read_parquet(pq_file)
        print(f"  Loaded {method_name}: {len(rankings[method_name]):,} rows")

    return rankings


def load_corpus(data_dir: Path, dataset: str) -> dict[str, dict]:
    """Load the corpus from raw BEIR data.

    Parameters
    ----------
    data_dir : Path
        Base data directory.
    dataset : str
        Dataset name.

    Returns
    -------
    dict[str, dict]
        Mapping from doc_id to document dict with title and text.
    """
    import json

    corpus_file = data_dir / "raw" / dataset / "corpus.jsonl"
    corpus: dict[str, dict] = {}

    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line.strip())
            doc_id = doc.pop("_id")
            corpus[doc_id] = doc

    return corpus


def load_queries(data_dir: Path, dataset: str) -> dict[str, str]:
    """Load queries from raw BEIR data.

    Parameters
    ----------
    data_dir : Path
        Base data directory.
    dataset : str
        Dataset name.

    Returns
    -------
    dict[str, str]
        Mapping from query_id to query text.
    """
    import json

    queries_file = data_dir / "raw" / dataset / "queries.jsonl"
    queries: dict[str, str] = {}

    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line.strip())
            queries[q["_id"]] = q["text"]

    return queries


def compute_rank_correlation(
    agg_df: pd.DataFrame,
    retriever_df: pd.DataFrame,
    query_id: str,
) -> tuple[float, float]:
    """Compute Spearman rank correlation between two rankings for a query.

    Parameters
    ----------
    agg_df : pd.DataFrame
        Aggregated rankings DataFrame.
    retriever_df : pd.DataFrame
        Individual retriever rankings DataFrame.
    query_id : str
        Query ID to compute correlation for.

    Returns
    -------
    tuple[float, float]
        Spearman correlation coefficient and p-value.
    """
    agg_query = agg_df[agg_df["query_id"] == query_id][["doc_id", "rank"]].copy()
    ret_query = retriever_df[retriever_df["query_id"] == query_id][["doc_id", "rank"]].copy()

    # Merge on doc_id to get paired ranks
    merged = agg_query.merge(ret_query, on="doc_id", suffixes=("_agg", "_ret"))

    if len(merged) < 2:
        return np.nan, np.nan

    corr, pval = stats.spearmanr(merged["rank_agg"], merged["rank_ret"])
    return corr, pval


def compute_all_correlations(
    agg_rankings: dict[str, pd.DataFrame],
    retriever_rankings: dict[str, pd.DataFrame],
    sample_queries: Optional[list[str]] = None,
    max_queries: int = 100,
) -> pd.DataFrame:
    """Compute rank correlations between all aggregation methods and retrievers.

    Parameters
    ----------
    agg_rankings : dict[str, pd.DataFrame]
        Aggregated rankings per method.
    retriever_rankings : dict[str, pd.DataFrame]
        Rankings per retriever.
    sample_queries : list[str], optional
        Subset of query IDs to use. If None, samples from all.
    max_queries : int
        Maximum number of queries to sample for correlation analysis.

    Returns
    -------
    pd.DataFrame
        Correlation matrix with aggregation methods as rows and retrievers as columns.
    """
    # Get all query IDs from first aggregation method
    first_agg = list(agg_rankings.values())[0]
    all_query_ids = first_agg["query_id"].unique().tolist()

    if sample_queries is not None:
        query_ids = [q for q in sample_queries if q in all_query_ids]
    else:
        # Sample queries if too many
        if len(all_query_ids) > max_queries:
            np.random.seed(42)
            query_ids = list(np.random.choice(all_query_ids, max_queries, replace=False))
        else:
            query_ids = all_query_ids

    print(f"  Computing correlations over {len(query_ids)} queries...")

    results: dict[str, dict[str, float]] = {}

    for agg_name, agg_df in agg_rankings.items():
        results[agg_name] = {}
        # Pre-filter to target queries for efficiency
        agg_filtered = agg_df[agg_df["query_id"].isin(query_ids)]

        for ret_name, ret_df in retriever_rankings.items():
            ret_filtered = ret_df[ret_df["query_id"].isin(query_ids)]

            # Merge all at once
            merged = agg_filtered.merge(
                ret_filtered,
                on=["query_id", "doc_id"],
                suffixes=("_agg", "_ret")
            )

            if len(merged) == 0:
                results[agg_name][ret_name] = np.nan
                continue

            # Compute correlation per query
            correlations: list[float] = []
            for qid in query_ids:
                qdf = merged[merged["query_id"] == qid]
                if len(qdf) >= 2:
                    corr, _ = stats.spearmanr(qdf["rank_agg"], qdf["rank_ret"])
                    if not np.isnan(corr):
                        correlations.append(corr)

            mean_corr = np.mean(correlations) if correlations else np.nan
            results[agg_name][ret_name] = mean_corr

    return pd.DataFrame(results).T


def compute_inter_aggregation_correlation(
    agg_rankings: dict[str, pd.DataFrame],
    sample_queries: Optional[list[str]] = None,
    max_queries: int = 100,
) -> pd.DataFrame:
    """Compute rank correlations between aggregation methods.

    Parameters
    ----------
    agg_rankings : dict[str, pd.DataFrame]
        Aggregated rankings per method.
    sample_queries : list[str], optional
        Subset of query IDs to use.
    max_queries : int
        Maximum number of queries to sample.

    Returns
    -------
    pd.DataFrame
        Correlation matrix between aggregation methods.
    """
    first_agg = list(agg_rankings.values())[0]
    all_query_ids = first_agg["query_id"].unique().tolist()

    if sample_queries is not None:
        query_ids = [q for q in sample_queries if q in all_query_ids]
    else:
        if len(all_query_ids) > max_queries:
            np.random.seed(42)
            query_ids = list(np.random.choice(all_query_ids, max_queries, replace=False))
        else:
            query_ids = all_query_ids

    agg_names = list(agg_rankings.keys())
    n = len(agg_names)
    corr_matrix = np.ones((n, n))

    # Pre-filter all dataframes
    filtered_rankings = {
        name: df[df["query_id"].isin(query_ids)]
        for name, df in agg_rankings.items()
    }

    for i, name_i in enumerate(agg_names):
        for j, name_j in enumerate(agg_names):
            if i < j:
                # Merge the two rankings
                merged = filtered_rankings[name_i].merge(
                    filtered_rankings[name_j],
                    on=["query_id", "doc_id"],
                    suffixes=("_i", "_j")
                )

                correlations: list[float] = []
                for qid in query_ids:
                    qdf = merged[merged["query_id"] == qid]
                    if len(qdf) >= 2:
                        corr, _ = stats.spearmanr(qdf["rank_i"], qdf["rank_j"])
                        if not np.isnan(corr):
                            correlations.append(corr)

                mean_corr = np.mean(correlations) if correlations else np.nan
                corr_matrix[i, j] = mean_corr
                corr_matrix[j, i] = mean_corr

    return pd.DataFrame(corr_matrix, index=agg_names, columns=agg_names)


def truncate_text(text: str, max_length: int = MAX_DOC_LENGTH) -> str:
    """Truncate text to maximum length with ellipsis.

    Parameters
    ----------
    text : str
        Text to truncate.
    max_length : int
        Maximum length.

    Returns
    -------
    str
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def get_uniform_samples(max_rank: int, sample_count: int) -> list[int]:
    """Get uniformly spaced rank positions for sampling.

    Parameters
    ----------
    max_rank : int
        Maximum rank in the ranking.
    sample_count : int
        Number of samples to take.

    Returns
    -------
    list[int]
        List of rank positions to sample.
    """
    if sample_count >= max_rank:
        return list(range(1, max_rank + 1))

    # Create evenly spaced positions
    positions = np.linspace(1, max_rank, sample_count + 2)[1:-1]
    return [int(round(p)) for p in positions]


def display_query_diagnostics(
    query_id: str,
    query_text: str,
    agg_rankings: dict[str, pd.DataFrame],
    retriever_rankings: dict[str, pd.DataFrame],
    corpus: dict[str, dict],
    top_k: int = 5,
    sample_count: int = 5,
) -> None:
    """Display diagnostic information for a single query.

    Parameters
    ----------
    query_id : str
        Query ID to display.
    query_text : str
        Query text.
    agg_rankings : dict[str, pd.DataFrame]
        Aggregated rankings.
    retriever_rankings : dict[str, pd.DataFrame]
        Per-retriever rankings.
    corpus : dict[str, dict]
        Document corpus.
    top_k : int
        Number of top documents to show.
    sample_count : int
        Number of uniform samples to show.
    """
    print("\n" + "=" * 100)
    print(f"QUERY ID: {query_id}")
    print("-" * 100)
    print(f"Query: {query_text}")
    print("=" * 100)

    # Use first aggregation method (typically RRF) for main display
    primary_agg = "rrf" if "rrf" in agg_rankings else list(agg_rankings.keys())[0]
    agg_df = agg_rankings[primary_agg]
    query_ranks = agg_df[agg_df["query_id"] == query_id].sort_values("rank")

    if query_ranks.empty:
        print(f"  [No rankings found for query {query_id}]")
        return

    max_rank = query_ranks["rank"].max()
    print(f"\nUsing {primary_agg.upper()} aggregation | Total docs ranked: {len(query_ranks)}")

    # Show rank comparison across methods
    print("\n--- Rank Comparison (Top-5 docs) ---")
    top_docs = query_ranks.head(5)["doc_id"].tolist()

    header = f"{'Doc ID':<12}"
    for agg_name in agg_rankings.keys():
        header += f" | {agg_name.upper():>6}"
    for ret_name in retriever_rankings.keys():
        header += f" | {ret_name:>8}"
    print(header)
    print("-" * len(header))

    for doc_id in top_docs:
        row = f"{doc_id:<12}"
        for agg_name, agg_df_inner in agg_rankings.items():
            rank = agg_df_inner[
                (agg_df_inner["query_id"] == query_id) & (agg_df_inner["doc_id"] == doc_id)
            ]["rank"].values
            row += f" | {rank[0]:>6}" if len(rank) > 0 else " |    N/A"
        for ret_name, ret_df in retriever_rankings.items():
            rank = ret_df[
                (ret_df["query_id"] == query_id) & (ret_df["doc_id"] == doc_id)
            ]["rank"].values
            row += f" | {rank[0]:>8}" if len(rank) > 0 else " |      N/A"
        print(row)

    # Show top-k documents
    print(f"\n--- Top-{top_k} Documents ({primary_agg.upper()}) ---")
    for _, row in query_ranks.head(top_k).iterrows():
        doc_id = row["doc_id"]
        rank = row["rank"]
        score = row["agg_score"]

        doc = corpus.get(doc_id, {})
        title = doc.get("title", "[No title]")
        text = doc.get("text", "[No text]")

        print(f"\n[Rank {rank}] Doc ID: {doc_id} | Score: {score:.6f}")
        print(f"  Title: {title}")
        print(f"  Text: {truncate_text(text)}")

    # Show uniform samples across rank distribution
    sample_positions = get_uniform_samples(max_rank, sample_count)
    print(f"\n--- Uniform Samples (ranks: {sample_positions}) ---")

    for target_rank in sample_positions:
        row = query_ranks[query_ranks["rank"] == target_rank]
        if row.empty:
            # Find closest rank
            row = query_ranks.iloc[(query_ranks["rank"] - target_rank).abs().argmin()]
            doc_id = row["doc_id"]
            rank = row["rank"]
            score = row["agg_score"]
        else:
            row = row.iloc[0]
            doc_id = row["doc_id"]
            rank = row["rank"]
            score = row["agg_score"]

        doc = corpus.get(doc_id, {})
        title = doc.get("title", "[No title]")
        text = doc.get("text", "[No text]")

        print(f"\n[Rank {rank}] Doc ID: {doc_id} | Score: {score:.6f}")
        print(f"  Title: {title}")
        print(f"  Text: {truncate_text(text)}")


def main() -> None:
    """Main entry point for Phase 2 diagnostics."""
    parser = argparse.ArgumentParser(
        description="Phase 2 Diagnostic Script"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="scifact",
        help="Dataset name (default: scifact)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base data directory (default: data)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top documents to show per query (default: 5)",
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=5,
        help="Number of uniform samples across rank distribution (default: 5)",
    )
    parser.add_argument(
        "--query_ids",
        type=str,
        default=None,
        help="Comma-separated list of query IDs (default: use important queries)",
    )
    parser.add_argument(
        "--correlation_only",
        action="store_true",
        help="Only show correlation analysis, skip document display",
    )
    parser.add_argument(
        "--max_corr_queries",
        type=int,
        default=100,
        help="Maximum number of queries to sample for correlation analysis (default: 100)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    phase1_dir = data_dir / "phase1_retrieval"
    phase2_dir = data_dir / "phase2_ir_aggregation"

    print("=" * 80)
    print("PHASE 2 DIAGNOSTICS")
    print(f"Dataset: {args.dataset}")
    print("=" * 80)

    # Load rankings
    print("\nLoading Phase 1 (retriever) rankings...")
    retriever_rankings = load_phase1_rankings(phase1_dir, args.dataset)

    print("\nLoading Phase 2 (aggregated) rankings...")
    agg_rankings = load_phase2_rankings(phase2_dir, args.dataset)

    # Load corpus and queries
    print("\nLoading corpus and queries...")
    corpus = load_corpus(data_dir, args.dataset)
    queries = load_queries(data_dir, args.dataset)
    print(f"  Loaded {len(corpus):,} documents and {len(queries):,} queries")

    # =====================================================================
    # PART 1: Correlation Analysis
    # =====================================================================
    print("\n" + "=" * 80)
    print("PART 1: RANK CORRELATION ANALYSIS")
    print("=" * 80)

    print("\n--- Aggregation vs Individual Retrievers (Mean Spearman ρ) ---")
    corr_df = compute_all_correlations(
        agg_rankings, retriever_rankings, max_queries=args.max_corr_queries
    )
    print(corr_df.round(4).to_string())

    print("\n--- Inter-Aggregation Correlation (Mean Spearman ρ) ---")
    inter_agg_corr = compute_inter_aggregation_correlation(
        agg_rankings, max_queries=args.max_corr_queries
    )
    print(inter_agg_corr.round(4).to_string())

    # Show per-retriever stats
    print("\n--- Summary Statistics ---")
    for agg_name in agg_rankings.keys():
        print(f"\n{agg_name.upper()} correlations with retrievers:")
        for ret_name, corr_val in corr_df.loc[agg_name].items():
            print(f"  {ret_name:<10}: {corr_val:.4f}")

    if args.correlation_only:
        print("\n[Skipping document display (--correlation_only flag)]")
        return

    # =====================================================================
    # PART 2: Document Display for Important Queries
    # =====================================================================
    print("\n" + "=" * 80)
    print("PART 2: DOCUMENT SAMPLES FOR IMPORTANT QUERIES")
    print("=" * 80)

    # Determine which queries to display
    if args.query_ids:
        target_query_ids = [q.strip() for q in args.query_ids.split(",")]
    else:
        target_query_ids = IMPORTANT_QUERY_IDS

    # Filter to queries that exist in the dataset
    available_query_ids = list(queries.keys())
    valid_query_ids = [q for q in target_query_ids if q in available_query_ids]

    print(f"\nDisplaying {len(valid_query_ids)} queries (of {len(target_query_ids)} requested)")
    missing = set(target_query_ids) - set(valid_query_ids)
    if missing:
        print(f"  [Missing query IDs: {sorted(missing)}]")

    for query_id in valid_query_ids:
        query_text = queries.get(query_id, "[Query not found]")
        display_query_diagnostics(
            query_id=query_id,
            query_text=query_text,
            agg_rankings=agg_rankings,
            retriever_rankings=retriever_rankings,
            corpus=corpus,
            top_k=args.top_k,
            sample_count=args.sample_count,
        )

    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
