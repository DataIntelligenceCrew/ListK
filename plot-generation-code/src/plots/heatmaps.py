"""
Heatmap plotting functions.

This module provides functions for creating P×X parameter heatmaps
showing time and recall metrics.
"""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..config import HEATMAP_CONFIGS, PIVOT_VALUES
from .base import save_figure, close_figure, FIGURE_SIZE_SMALL

import matplotlib as mpl
import matplotlib.font_manager as fm

_font_path = '/home/jwc/.fonts/LinLibertine_R.otf'
fm.fontManager.addfont(_font_path)
_font_prop = fm.FontProperties(fname=_font_path)
_font_name = _font_prop.get_name()


def _format_condition(label: str) -> str:
    """Convert label to filename-friendly condition string."""
    return label.lower().replace(" ", "_")


def plot_px_heatmap(
    df: pd.DataFrame,
    metric: Literal["time", "recall"],
    label: str,
    output_dir: Path | str = Path("."),
    suffix: str = "",
    pivot_values: list[int] | None = None,
) -> None:
    """
    Create a P×X lower-triangular heatmap.

    This unified function replaces both make_half_plot and make_half_plot_hgt
    from the original code. The only difference was the output filename suffix.

    Args:
        df: DataFrame with columns 'p', 'x', and the metric column.
        metric: Which metric to plot ('time' or 'recall').
        label: Label for the plot title (e.g., 'Early Stopping').
        output_dir: Directory to save the plot.
        suffix: Optional suffix for the output filename (e.g., '_hgt').
        pivot_values: List of pivot values for axes. Defaults to PIVOT_VALUES.

    Output filename format:
        select_PvX_{gt_type}_{metric}_{condition}.pdf
        e.g., select_PvX_hgt_recall_early_stopping.pdf
    """
    pivot_values = pivot_values or PIVOT_VALUES
    config = HEATMAP_CONFIGS[metric]

    # Build the data table for the heatmap
    n = len(pivot_values)
    table = np.zeros((n, n))

    for p_idx, p_val in enumerate(pivot_values):
        for x_idx, x_val in enumerate(pivot_values):
            if p_val >= x_val:
                mask = (df["p"] == p_val) & (df["x"] == x_val)
                if mask.any():
                    table[p_idx][x_idx] = df.loc[mask, metric].values[0]

    # Create upper triangular mask (we show lower triangular)
    mask = np.triu(np.ones_like(table, dtype=bool), k=1)

    # Create the heatmap
    plt.figure(figsize=config.figsize)
    sns.heatmap(
        table,
        mask=mask,
        cmap=config.cmap,
        vmin=config.vmin,
        vmax=config.vmax,
        center=config.center,
        square=True,
        linewidths=0.5,
        annot=True,
        fmt=config.fmt,
        cbar_kws={"shrink": 0.5},
        xticklabels=pivot_values,
        yticklabels=pivot_values,
    )

    plt.xlabel("P'")
    plt.ylabel("P")

    # Build filename: select_PvX_{gt_type}_{metric}_{condition}
    condition = _format_condition(label)
    if metric == "time":
        # Time doesn't have gt type distinction
        filename = f"select_PvX_{metric}_{condition}"
    else:
        # Recall has gt type: _hgt suffix means HGT, no suffix means LLM
        gt_type = "hgt" if suffix == "_hgt" else "llm"
        filename = f"select_PvX_{gt_type}_{metric}_{condition}"

    save_figure(filename, output_dir)
    close_figure()


def plot_px_diagonal(
    df: pd.DataFrame,
    metric: Literal["time", "recall"],
    label: str,
    output_dir: Path | str = Path("."),
    suffix: str = "",
    pivot_values: list[int] | None = None,
) -> None:
    """
    Create a line plot for P=X diagonal values.

    Shows how the metric changes as P increases (with X=P constraint).

    Args:
        df: DataFrame with columns 'p', 'x', and the metric column.
        metric: Which metric to plot ('time' or 'recall').
        label: Label for the plot title (e.g., 'Early Stopping').
        output_dir: Directory to save the plot.
        suffix: Optional suffix to indicate gt type (e.g., '_hgt').
        pivot_values: List of pivot values. Defaults to PIVOT_VALUES.

    Output filename format:
        select_PeqX_{gt_type}_{metric}_{condition}.pdf
        e.g., select_PeqX_hgt_recall_early_stopping.pdf
    """
    pivot_values = pivot_values or PIVOT_VALUES

    # Extract diagonal values (P=X)
    p_values = []
    metric_values = []

    for p_val in pivot_values:
        mask = (df["p"] == p_val) & (df["x"] == p_val)
        if mask.any():
            p_values.append(p_val)
            metric_values.append(df.loc[mask, metric].values[0])

    if not p_values:
        print(f"  Warning: No diagonal data found for {label}")
        return

    # Create the line plot (no markers)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SMALL)

    ax.plot(p_values, metric_values, linewidth=2, color='#0077BB')

    ax.set_xlabel("P (= P')")
    if metric == "time":
        ax.set_ylabel("Latency (s)")
        ax.set_ylim(0, None)  # Start y-axis at 0
    else:
        ax.set_ylabel("Recall@10")
        ax.set_ylim(0, 1)

    # Use the same x values as tick labels
    ax.set_xticks(p_values)

    # Build filename: select_PeqX_{gt_type}_{metric}_{condition}
    condition = _format_condition(label)
    if metric == "time":
        filename = f"select_PeqX_{metric}_{condition}"
    else:
        gt_type = "hgt" if suffix == "_hgt" else "llm"
        filename = f"select_PeqX_{gt_type}_{metric}_{condition}"

    save_figure(filename, output_dir)
    close_figure()


def plot_px_heatmap_pair(
    df: pd.DataFrame,
    label: str,
    output_dir: Path | str = Path("."),
    suffix: str = "",
) -> None:
    """
    Create both time and recall heatmaps for a dataset.

    Convenience function that generates both metric heatmaps
    for a given DataFrame.

    Args:
        df: DataFrame with columns 'p', 'x', 'time', 'recall'.
        label: Label for the plot titles.
        output_dir: Directory to save the plots.
        suffix: Optional suffix for output filenames.
    """
    plot_px_heatmap(df, "time", label, output_dir, suffix)
    plot_px_heatmap(df, "recall", label, output_dir, suffix)


def plot_px_diagonal_pair(
    df: pd.DataFrame,
    label: str,
    output_dir: Path | str = Path("."),
    suffix: str = "",
) -> None:
    """
    Create both time and recall diagonal (P=X) line plots for a dataset.

    Args:
        df: DataFrame with columns 'p', 'x', 'time', 'recall'.
        label: Label for the plot titles.
        output_dir: Directory to save the plots.
        suffix: Optional suffix for output filenames.
    """
    plot_px_diagonal(df, "time", label, output_dir, suffix)
    plot_px_diagonal(df, "recall", label, output_dir, suffix)


def plot_px_time_comparison(
    df_early: pd.DataFrame,
    df_no_early: pd.DataFrame,
    output_dir: Path | str = Path("."),
    pivot_values: list[int] | None = None,
) -> None:
    """
    Create a combined line plot comparing P=P' latencies vs optimal P' configurations.

    Shows three lines:
    - P=P' with early stopping (diagonal constraint)
    - Optimal P'≤P with early stopping (best configuration for each P)
    - Same optimal P',P configuration without early stopping

    Args:
        df_early: DataFrame for early stopping experiments.
        df_no_early: DataFrame for no early stopping experiments.
        output_dir: Directory to save the plot.
        pivot_values: List of pivot values. Defaults to PIVOT_VALUES.
    """
    pivot_values = pivot_values or PIVOT_VALUES

    # Extract P=P' diagonal for early stopping
    p_vals_diagonal = []
    time_diagonal = []
    for p_val in pivot_values:
        mask = (df_early["p"] == p_val) & (df_early["x"] == p_val)
        if mask.any():
            p_vals_diagonal.append(p_val)
            time_diagonal.append(df_early.loc[mask, "time"].values[0])

    # Extract optimal P'≤P for each P with early stopping
    p_vals_optimal = []
    time_optimal = []
    x_optimal = []
    for p_val in pivot_values:
        mask = (df_early["p"] == p_val) & (df_early["x"] <= p_val)
        if mask.any():
            subset = df_early.loc[mask]
            min_idx = subset["time"].idxmin()
            min_time = subset.loc[min_idx, "time"]
            opt_x = subset.loc[min_idx, "x"]
            p_vals_optimal.append(p_val)
            time_optimal.append(min_time)
            x_optimal.append(int(opt_x))

    # Find optimal P'≤P for each P WITHOUT early stopping
    p_vals_optimal_ne = []
    time_optimal_ne = []
    x_optimal_ne = []
    for p_val in pivot_values:
        mask = (df_no_early["p"] == p_val) & (df_no_early["x"] <= p_val)
        if mask.any():
            subset = df_no_early.loc[mask]
            min_idx = subset["time"].idxmin()
            min_time = subset.loc[min_idx, "time"]
            opt_x = subset.loc[min_idx, "x"]
            p_vals_optimal_ne.append(p_val)
            time_optimal_ne.append(min_time)
            x_optimal_ne.append(int(opt_x))

    _rc = {
        'font.family': _font_name,
        'axes.labelsize': 9,
        'legend.fontsize': 6,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
    }
    with mpl.rc_context(_rc):
        fig, ax = plt.subplots(figsize=(2.31, 1.7))
        fig.subplots_adjust(left=0.22, right=0.99, top=0.97, bottom=0.22)

        # Map P values to uniform integer positions
        p_to_idx = {p: i for i, p in enumerate(pivot_values)}

        # Plot P=P' line
        idx_diag = [p_to_idx[p] for p in p_vals_diagonal]
        ax.plot(idx_diag, time_diagonal, linewidth=1,
                color='#004488', label="P=P'")

        # Plot optimal P'≤P line WITHOUT early stopping (dotted) — plotted before early stopping for legend order
        if p_vals_optimal_ne:
            idx_opt_ne = [p_to_idx[p] for p in p_vals_optimal_ne]
            ax.plot(idx_opt_ne, time_optimal_ne, linewidth=1,
                    color='#BB5566', linestyle=':', label="P'$\\leq$P (no early stopping)")

        # Plot optimal P'≤P line with early stopping (dashed)
        if p_vals_optimal:
            idx_opt = [p_to_idx[p] for p in p_vals_optimal]
            ax.plot(idx_opt, time_optimal, linewidth=1,
                    color='#DDAA33', linestyle='--', label="P'$\\leq$P (early stopping)")

        ax.set_xlabel("P")
        ax.set_ylabel("Latency (s)")
        ax.set_ylim(0, None)
        ax.set_xticks(range(len(pivot_values)))
        ax.set_xticklabels([str(p) for p in pivot_values])
        ax.legend()

        save_figure("select_PvX_time_comparison", output_dir, bbox_inches=None)
        close_figure()
