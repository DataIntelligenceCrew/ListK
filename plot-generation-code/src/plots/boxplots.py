"""
Box plot functions.

This module provides functions for creating box plots comparing
distributions of metrics across different configurations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .base import (
    PlotMetadata,
    save_figure,
    close_figure,
    setup_axis,
    FIGURE_SIZE_SMALL,
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
)

import matplotlib as mpl
import matplotlib.font_manager as fm

_font_path = '/home/jwc/.fonts/LinLibertine_R.otf'
fm.fontManager.addfont(_font_path)
_font_prop = fm.FontProperties(fname=_font_path)
_font_name = _font_prop.get_name()


# Box plot styling
BOXPLOT_WIDTH: float = 0.5


def plot_embedding_comparison(
    embedding_times: list[float],
    no_embedding_times: list[float],
    output_dir: Path | str = Path("."),
) -> None:
    """
    Create box plot comparing latency with/without embedding-based pivot selection.

    Args:
        embedding_times: List of latencies with embedding pivot selection.
        no_embedding_times: List of latencies without embedding pivot selection.
        output_dir: Directory to save the plot.
    """
    _rc = {
        'font.family': _font_name,
        'axes.labelsize': 9,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
    }
    with mpl.rc_context(_rc):
        fig, ax = plt.subplots(figsize=(2.31, 1.7))
        fig.subplots_adjust(left=0.22, right=0.99, top=0.97, bottom=0.22)

        data = [embedding_times, no_embedding_times]
        labels = ["with heuristic", "without heuristic"]

        # Create box plot with wider boxes and black median
        medianprops = dict(color="black")

        ax.boxplot(
            data,
            tick_labels=labels,
            widths=0.7,
            medianprops=medianprops,
        )
        ax.tick_params(axis='x', labelsize=9)

        # Set y-axis to start at 0
        y_max = max(max(embedding_times), max(no_embedding_times))
        ax.set_ylabel("Latency (s)")
        ax.set_ylim(0, y_max * 1.1)

        # Create metadata
        metadata = PlotMetadata(
            title="Latency for Embedding vs Non-Embedding Pivot Selection",
            xlabel="Pivot Selection Method",
            ylabel="Latency (s)",
            description="Compares execution latency between pivot selection with and without embedding-based heuristics.",
            data_sources=[
                "5000e/bier_result_unsorted_16_2_25.csv",
                "no-em-p/bier_result_unsorted_16_2_25.csv",
            ],
            algorithm="Multi-pivot Quickselect",
            parameters={
                "p": 16,
                "x": 2,
                "num_queries": 25,
            },
        )

        save_figure("embedding_comparison", output_dir, metadata=metadata, bbox_inches=None)
        close_figure()


def plot_embedding_comparison_from_df(
    embedding_df: pd.DataFrame,
    no_embedding_df: pd.DataFrame,
    output_dir: Path | str = Path("."),
) -> None:
    """
    Create embedding comparison box plot from DataFrames.

    Args:
        embedding_df: DataFrame with 'time' column for embedding method.
        no_embedding_df: DataFrame with 'time' column for non-embedding method.
        output_dir: Directory to save the plot.
    """
    plot_embedding_comparison(
        embedding_df["time"].tolist(),
        no_embedding_df["time"].tolist(),
        output_dir,
    )


def plot_tournament_filter(
    recall_distributions: list[list[float]],
    labels: list[str],
    k: int,
    output_dir: Path | str = Path("."),
    show_mean_line: bool = True,
) -> None:
    """
    Create box plot for tournament filter recall at different L values.

    Args:
        recall_distributions: List of recall value lists, one per L value.
        labels: Labels for each L value.
        k: The K value for Recall@K.
        output_dir: Directory to save the plot.
        show_mean_line: If True, overlay a line connecting mean values.
    """
    fig, ax = plt.subplots()

    # Calculate means for the line plot
    if show_mean_line:
        means = [np.mean(recalls) for recalls in recall_distributions]
        x_positions = np.arange(1, len(means) + 1)
        ax.plot(x_positions, means, color="k")

    # Create box plot
    ax.boxplot(recall_distributions, tick_labels=labels)

    setup_axis(
        ax,
        xlabel="L",
        ylabel=f"Maximum Possible Recall@{k}",
    )

    save_figure(f"tfilter_{k}_plot", output_dir)
    close_figure()


def plot_tournament_filter_summary(
    l_values: list[int],
    k_values: list[int],
    recall_data: dict[int, tuple[list[float], list[float]]],
    output_dir: Path | str = Path("."),
) -> None:
    """
    Create summary line plot for tournament filter recall across L and K values.

    Args:
        l_values: List of L values for x-axis.
        k_values: List of K values (one line per K).
        recall_data: Dict mapping K -> (mean_recalls, std_recalls) for each L.
        output_dir: Directory to save the plot.
    """
    _rc = {
        'font.family': _font_name,
        'axes.labelsize': 9,
        'legend.fontsize': 9,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
    }
    with mpl.rc_context(_rc):
        fig, ax = plt.subplots(figsize=(2.24, 1.68))
        fig.subplots_adjust(left=0.19, right=0.97, top=0.97, bottom=0.21)

        # Different line styles for each K value
        linestyles = ["-", "--", "-.", ":"]

        for i, k in enumerate(k_values):
            means, stds = recall_data[k]
            linestyle = linestyles[i % len(linestyles)]

            # Plot line in black
            ax.plot(l_values, means, color="k", linestyle=linestyle, linewidth=1, label=f"K={k}")

        setup_axis(
            ax,
            xlabel="# Survivors (S)",
            ylabel="Recall",
            ylim=(0, 1),
            fontsize=9,
        )
        ax.tick_params(axis='both', labelsize=6)

        ax.legend(loc="lower right", fontsize=6, borderaxespad=0.3)

        save_figure("tfilter_summary", output_dir, bbox_inches=None)
        close_figure()


def plot_sort_ndcg_boxplot(
    ndcg_distributions: list[list[float]],
    labels: list[str],
    k: int,
    output_dir: Path | str = Path("."),
) -> None:
    """
    Create box plot for NDCG at different window sizes.

    Args:
        ndcg_distributions: List of NDCG value lists, one per window size.
        labels: Labels for each window size.
        k: The K value for NDCG@K.
        output_dir: Directory to save the plot.
    """
    fig, ax = plt.subplots()

    ax.boxplot(ndcg_distributions, tick_labels=labels)

    setup_axis(
        ax,
        xlabel="K",
        ylabel=f"NDCG@{k}",
        ylim=(0, 1),
    )

    save_figure(f"sort_ndcg@{k}", output_dir)
    close_figure()
