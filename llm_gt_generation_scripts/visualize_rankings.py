#!/usr/bin/env python
"""
Visualization script for ranking data from any pipeline phase.

Produces a single image showing:
1. Score/rating distributions for each method
2. Rank correlations between methods (Spearman correlation heatmap)

Usage:
    python scripts/visualize_rankings.py --phase_dir data/phase1_retrieval/scifact --output viz_phase1.png
    python scripts/visualize_rankings.py --phase_dir data/phase3_reranking/scifact --output viz_phase3.png
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def load_phase_data(phase_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Load all parquet files from a phase directory.

    Parameters
    ----------
    phase_dir : Path
        Directory containing parquet files for each method.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping method names to their ranking DataFrames.
    """
    data: dict[str, pd.DataFrame] = {}
    parquet_files = sorted(phase_dir.glob("*.parquet"))

    for pq_file in parquet_files:
        method_name = pq_file.stem
        # Skip aggregated files if present (they follow different naming)
        if method_name.startswith("aggregated_"):
            method_name = method_name.replace("aggregated_", "") + " (agg)"
        data[method_name] = pd.read_parquet(pq_file)
        print(f"Loaded {method_name}: {len(data[method_name]):,} rows")

    return data


def compute_rank_correlations(
    data: dict[str, pd.DataFrame],
    sample_queries: Optional[int] = 50,
) -> pd.DataFrame:
    """
    Compute pairwise Spearman rank correlations between methods.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Dictionary mapping method names to their ranking DataFrames.
    sample_queries : Optional[int]
        Number of queries to sample for correlation computation.
        If None, use all queries.

    Returns
    -------
    pd.DataFrame
        Correlation matrix between methods.
    """
    methods = list(data.keys())
    n_methods = len(methods)

    # Get common query IDs across all methods
    query_sets = [set(df["query_id"].unique()) for df in data.values()]
    common_queries = set.intersection(*query_sets)
    common_queries = sorted(common_queries)

    if sample_queries and len(common_queries) > sample_queries:
        rng = np.random.default_rng(42)
        common_queries = list(rng.choice(common_queries, sample_queries, replace=False))

    print(f"Computing correlations over {len(common_queries)} queries...")

    # Build rank lookup for each method: (query_id, doc_id) -> rank
    rank_lookups: list[dict[tuple[str, str], int]] = []
    for method in methods:
        df = data[method]
        lookup = {
            (row["query_id"], row["doc_id"]): row["rank"]
            for _, row in df[df["query_id"].isin(common_queries)].iterrows()
        }
        rank_lookups.append(lookup)

    # Compute correlations
    corr_matrix = np.ones((n_methods, n_methods))

    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            # Get common (query_id, doc_id) pairs
            common_pairs = set(rank_lookups[i].keys()) & set(rank_lookups[j].keys())

            if len(common_pairs) < 10:
                corr_matrix[i, j] = np.nan
                corr_matrix[j, i] = np.nan
                continue

            ranks_i = [rank_lookups[i][pair] for pair in common_pairs]
            ranks_j = [rank_lookups[j][pair] for pair in common_pairs]

            corr, _ = stats.spearmanr(ranks_i, ranks_j)
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    return pd.DataFrame(corr_matrix, index=methods, columns=methods)


def plot_distributions_and_correlations(
    data: dict[str, pd.DataFrame],
    corr_matrix: pd.DataFrame,
    output_path: Path,
    phase_name: str = "Phase",
    score_column: str = "score",
) -> None:
    """
    Create a combined figure with score distributions and correlation heatmap.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Dictionary mapping method names to their ranking DataFrames.
    corr_matrix : pd.DataFrame
        Correlation matrix between methods.
    output_path : Path
        Path to save the output image.
    phase_name : str
        Name of the phase for the plot title.
    score_column : str
        Name of the score column to plot distributions for.
    """
    methods = list(data.keys())
    n_methods = len(methods)

    # Calculate grid layout
    # Top rows: score distributions (2 per row)
    # Bottom: correlation heatmap
    n_dist_rows = (n_methods + 1) // 2
    total_rows = n_dist_rows + 1

    fig = plt.figure(figsize=(14, 4 * total_rows))

    # Create GridSpec for flexible layout
    gs = fig.add_gridspec(
        total_rows,
        2,
        height_ratios=[1] * n_dist_rows + [1.2],
        hspace=0.35,
        wspace=0.25,
    )

    # Color palette for methods
    colors = sns.color_palette("husl", n_methods)

    # Plot score distributions
    for idx, method in enumerate(methods):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])

        df = data[method]

        # Detect score column (could be 'score', 'agg_score', 'elo_rating')
        if score_column in df.columns:
            scores = df[score_column]
        elif "agg_score" in df.columns:
            scores = df["agg_score"]
        elif "elo_rating" in df.columns:
            scores = df["elo_rating"]
        else:
            scores = df["score"]

        # Plot histogram with KDE
        sns.histplot(
            scores,
            bins=50,
            kde=True,
            color=colors[idx],
            ax=ax,
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )

        ax.set_title(f"{method}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Score", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)

        # Add statistics annotation
        stats_text = (
            f"n={len(scores):,}\n"
            f"mean={scores.mean():.2f}\n"
            f"std={scores.std():.2f}\n"
            f"min={scores.min():.2f}\n"
            f"max={scores.max():.2f}"
        )
        ax.text(
            0.97,
            0.97,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Hide empty subplot if odd number of methods
    if n_methods % 2 == 1:
        ax_empty = fig.add_subplot(gs[n_dist_rows - 1, 1])
        ax_empty.axis("off")

    # Plot correlation heatmap (spanning both columns)
    ax_corr = fig.add_subplot(gs[n_dist_rows, :])

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="RdYlBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax_corr,
        cbar_kws={"shrink": 0.6, "label": "Spearman ρ"},
        annot_kws={"size": 10},
    )

    ax_corr.set_title(
        "Rank Correlations (Spearman ρ)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    # Main title
    fig.suptitle(
        f"{phase_name} - Score Distributions & Rank Correlations",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved visualization to {output_path}")


def main() -> None:
    """Main entry point for the visualization script."""
    parser = argparse.ArgumentParser(
        description="Visualize ranking distributions and correlations"
    )
    parser.add_argument(
        "--phase_dir",
        type=str,
        required=True,
        help="Directory containing parquet files for the phase",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ranking_visualization.png",
        help="Output image path (default: ranking_visualization.png)",
    )
    parser.add_argument(
        "--sample_queries",
        type=int,
        default=50,
        help="Number of queries to sample for correlation computation (default: 50)",
    )
    parser.add_argument(
        "--phase_name",
        type=str,
        default=None,
        help="Name for the phase (auto-detected if not provided)",
    )

    args = parser.parse_args()

    phase_dir = Path(args.phase_dir)
    output_path = Path(args.output)

    if not phase_dir.exists():
        raise FileNotFoundError(f"Phase directory not found: {phase_dir}")

    # Auto-detect phase name from directory
    if args.phase_name:
        phase_name = args.phase_name
    else:
        # Extract from path like data/phase1_retrieval/scifact
        parts = phase_dir.parts
        phase_part = next((p for p in parts if p.startswith("phase")), None)
        dataset_part = parts[-1] if parts else "unknown"

        if phase_part:
            phase_name = f"{phase_part} ({dataset_part})"
        else:
            phase_name = str(phase_dir)

    print(f"Loading data from {phase_dir}...")
    data = load_phase_data(phase_dir)

    if not data:
        raise ValueError(f"No parquet files found in {phase_dir}")

    print(f"\nComputing rank correlations...")
    corr_matrix = compute_rank_correlations(data, sample_queries=args.sample_queries)

    print(f"\nGenerating visualization...")
    plot_distributions_and_correlations(
        data=data,
        corr_matrix=corr_matrix,
        output_path=output_path,
        phase_name=phase_name,
    )

    print("\nCorrelation matrix:")
    print(corr_matrix.round(3).to_string())


if __name__ == "__main__":
    main()
