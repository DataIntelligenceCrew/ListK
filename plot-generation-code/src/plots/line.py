"""
Line plot functions.

This module provides functions for creating line plots showing
trends across parameter values.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .base import save_figure, close_figure, setup_axis

import matplotlib.font_manager as fm
_font_path = '/home/jwc/.fonts/LinLibertine_R.otf'
fm.fontManager.addfont(_font_path)
_font_prop = fm.FontProperties(fname=_font_path)
_font_name = _font_prop.get_name()


def plot_k_recall(
    k_values: list[int],
    human_recall_values: list[float],
    llm_recall_values: list[float],
    labels: list[str],
    output_dir: Path | str = Path("."),
    human_std: list[float] | None = None,
    llm_std: list[float] | None = None,
    human_bounds: tuple[list[float], list[float]] | None = None,
    llm_bounds: tuple[list[float], list[float]] | None = None,
    ci_label: str = "1std Confidence Interval",
) -> None:
    """
    Create line plot of Recall@K vs K with Human and LLM GT.

    Args:
        k_values: List of K values.
        human_recall_values: Recall values for human ground truth.
        llm_recall_values: Recall values for LLM ground truth.
        labels: Tick labels for x-axis.
        output_dir: Directory to save the plot.
        human_std: Optional standard deviations for human recall (used if human_bounds not provided).
        llm_std: Optional standard deviations for LLM recall (used if llm_bounds not provided).
        human_bounds: Optional tuple of (lower, upper) bounds for human CI.
        llm_bounds: Optional tuple of (lower, upper) bounds for LLM CI.
        ci_label: Label for the confidence interval in the legend.
    """
    from matplotlib.lines import Line2D

    plt.rcParams['font.family'] = _font_name
    fig, ax = plt.subplots(figsize=(1.715, 1.4))
    fig.subplots_adjust(left=0.20, right=0.98, top=0.98, bottom=0.19)

    x_positions = np.arange(len(k_values))

    # Fit linear models using numpy polyfit
    human_coeffs = np.polyfit(x_positions, human_recall_values, 1)
    llm_coeffs = np.polyfit(x_positions, llm_recall_values, 1)

    # Extended x for smooth fit line
    x_fit = np.linspace(x_positions[0], x_positions[-1], 100)
    human_fit = np.polyval(human_coeffs, x_fit)
    llm_fit = np.polyval(llm_coeffs, x_fit)

    # Plot human GT: dots + linear fit
    ax.scatter(x_positions, human_recall_values, color="#0077BB", s=5, zorder=3)
    ax.plot(x_fit, human_fit, color="#0077BB", linestyle="-", linewidth=0.5, zorder=2)

    # Plot LLM GT: dots + linear fit
    ax.scatter(x_positions, llm_recall_values, color="#EE3377", s=5, zorder=3)
    ax.plot(x_fit, llm_fit, color="#EE3377", linestyle="--", linewidth=0.5, zorder=2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)

    ax.set_xlabel("K", fontsize=6, labelpad=2)
    ax.set_ylabel("Recall@K", fontsize=6, labelpad=2)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis='both', labelsize=5)

    legend_elements = [
        Line2D([0], [0], color="#0077BB", linestyle="-", linewidth=0.5, marker="o", markersize=3, label="Human Labels (R=1)"),
        Line2D([0], [0], color="#EE3377", linestyle="--", linewidth=0.5, marker="o", markersize=3, label="LLM-as-Judge (R=10)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=6)

    save_figure("krecallplot", output_dir, bbox_inches=None)
    close_figure()


def plot_k_recall_comparison(
    k_values: list[int],
    human_recall_values: list[float],
    llm_recall_values: list[float],
    output_dir: Path | str = Path("."),
    filename: str = "k_recall_comparison",
    human_std: list[float] | None = None,
    llm_std: list[float] | None = None,
    human_bounds: tuple[list[float], list[float]] | None = None,
    llm_bounds: tuple[list[float], list[float]] | None = None,
    ci_label: str = "1std Confidence Interval",
) -> None:
    """
    Create line plot comparing Recall@K for human and LLM labels.

    Args:
        k_values: List of K values (P=X diagonal values).
        human_recall_values: Recall values computed against human ground truth.
        llm_recall_values: Recall values computed against LLM ground truth.
        output_dir: Directory to save the plot.
        filename: Base filename for the output.
        human_std: Standard deviations for human recall (used if human_bounds not provided).
        llm_std: Standard deviations for LLM recall (used if llm_bounds not provided).
        human_bounds: Optional tuple of (lower, upper) bounds for human CI.
        llm_bounds: Optional tuple of (lower, upper) bounds for LLM CI.
        ci_label: Label for the confidence interval in the legend.
    """
    from matplotlib.lines import Line2D

    plt.rcParams['font.family'] = _font_name
    fig, ax = plt.subplots(figsize=(1.715, 1.4))
    fig.subplots_adjust(left=0.20, right=0.98, top=0.98, bottom=0.19)

    labels = [str(k) for k in k_values]
    x_positions = np.arange(len(k_values))

    # Fit linear models using numpy polyfit
    human_coeffs = np.polyfit(x_positions, human_recall_values, 1)
    llm_coeffs = np.polyfit(x_positions, llm_recall_values, 1)

    # Extended x for smooth fit line
    x_fit = np.linspace(x_positions[0], x_positions[-1], 100)
    human_fit = np.polyval(human_coeffs, x_fit)
    llm_fit = np.polyval(llm_coeffs, x_fit)

    # Plot human GT: dots + linear fit
    ax.scatter(x_positions, human_recall_values, color="#0077BB", s=5, zorder=3)
    ax.plot(x_fit, human_fit, color="#0077BB", linestyle="-", linewidth=0.5, zorder=2)

    # Plot LLM GT: dots + linear fit
    ax.scatter(x_positions, llm_recall_values, color="#EE3377", s=5, zorder=3)
    ax.plot(x_fit, llm_fit, color="#EE3377", linestyle="--", linewidth=0.5, zorder=2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)

    ax.set_xlabel("Pivot Count (P)", fontsize=6, labelpad=2)
    ax.set_ylabel("Recall@10", fontsize=6, labelpad=2)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis='both', labelsize=5)

    legend_elements = [
        Line2D([0], [0], color="#0077BB", linestyle="-", linewidth=0.5, marker="o", markersize=3, label="Human Labels (R=1)"),
        Line2D([0], [0], color="#EE3377", linestyle="--", linewidth=0.5, marker="o", markersize=3, label="LLM-as-Judge (R=10)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=6)

    save_figure(filename, output_dir, bbox_inches=None)
    close_figure()


def plot_sort_k_ndcg_comparison(
    k_values: list[int],
    human_ndcg_values: list[float],
    llm_ndcg_values: list[float],
    output_dir: Path | str = Path("."),
    human_std: list[float] | None = None,
    llm_std: list[float] | None = None,
    human_bounds: tuple[list[float], list[float]] | None = None,
    llm_bounds: tuple[list[float], list[float]] | None = None,
    ci_label: str = "1std Confidence Interval",
) -> None:
    """
    Create line plot comparing NDCG@10 for human and LLM labels across sort window sizes.

    Args:
        k_values: List of K (sort window) values.
        human_ndcg_values: NDCG values computed against human ground truth.
        llm_ndcg_values: NDCG values computed against LLM ground truth.
        output_dir: Directory to save the plot.
        human_std: Standard deviations for human NDCG (used if human_bounds not provided).
        llm_std: Standard deviations for LLM NDCG (used if llm_bounds not provided).
        human_bounds: Optional tuple of (lower, upper) bounds for human CI.
        llm_bounds: Optional tuple of (lower, upper) bounds for LLM CI.
        ci_label: Label for the confidence interval in the legend.
    """
    from matplotlib.lines import Line2D

    plt.rcParams['font.family'] = _font_name
    fig, ax = plt.subplots(figsize=(1.715, 1.4))
    fig.subplots_adjust(left=0.20, right=0.98, top=0.98, bottom=0.19)

    labels = [str(k) for k in k_values]
    x_positions = np.arange(len(k_values))

    # Fit linear models using numpy polyfit
    human_coeffs = np.polyfit(x_positions, human_ndcg_values, 1)
    llm_coeffs = np.polyfit(x_positions, llm_ndcg_values, 1)

    # Extended x for smooth fit line
    x_fit = np.linspace(x_positions[0], x_positions[-1], 100)
    human_fit = np.polyval(human_coeffs, x_fit)
    llm_fit = np.polyval(llm_coeffs, x_fit)

    # Plot human GT: dots + linear fit
    ax.scatter(x_positions, human_ndcg_values, color="#0077BB", s=5, zorder=3)
    ax.plot(x_fit, human_fit, color="#0077BB", linestyle="-", linewidth=0.5, zorder=2)

    # Plot LLM GT: dots + linear fit
    ax.scatter(x_positions, llm_ndcg_values, color="#EE3377", s=5, zorder=3)
    ax.plot(x_fit, llm_fit, color="#EE3377", linestyle="--", linewidth=0.5, zorder=2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)

    ax.set_xlabel("N", fontsize=6, labelpad=2)
    ax.set_ylabel("NDCG@10", fontsize=6, labelpad=2)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis='both', labelsize=5)

    legend_elements = [
        Line2D([0], [0], color="#0077BB", linestyle="-", linewidth=0.5, marker="o", markersize=3, label="Human Labels (R=1)"),
        Line2D([0], [0], color="#EE3377", linestyle="--", linewidth=0.5, marker="o", markersize=3, label="LLM-as-Judge (R=10)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=6)

    save_figure("sort_K_ndcg_comparison", output_dir, bbox_inches=None)
    close_figure()


def plot_sort_time(
    window_labels: list[str],
    time_values: list[float],
    output_dir: Path | str = Path("."),
) -> None:
    """
    Create line plot of latency vs window size for sorting.

    Args:
        window_labels: Labels for window sizes.
        time_values: Corresponding latency values.
        output_dir: Directory to save the plot.
    """
    fig, ax = plt.subplots()

    ax.plot(window_labels, time_values, color="k")

    setup_axis(
        ax,
        xlabel="K",
        ylabel="Latency (s)",
    )

    save_figure("sort_time", output_dir)
    close_figure()


def plot_sort_ndcg_line(
    window_labels: list[str],
    ndcg_values: list[float],
    output_dir: Path | str = Path("."),
) -> None:
    """
    Create line plot of NDCG vs window size (HGT variant).

    Args:
        window_labels: Labels for window sizes.
        ndcg_values: Corresponding NDCG values.
        output_dir: Directory to save the plot.
    """
    fig, ax = plt.subplots()

    ax.plot(window_labels, ndcg_values, color="k")

    setup_axis(
        ax,
        xlabel="K",
        ylabel="NDCG@K",
        ylim=(0, 1),
    )

    save_figure("sort_ndcg_hgt", output_dir)
    close_figure()


def plot_sort_metrics(
    window_labels: list[str],
    ndcg_distributions: list[list[float]],
    time_values: list[float],
    k: int,
    output_dir: Path | str = Path("."),
) -> None:
    """
    Create both NDCG box plot and latency line plot for sorting.

    This combines the functionality of sort_plots from the original code.

    Args:
        window_labels: Labels for window sizes.
        ndcg_distributions: List of NDCG distributions per window.
        time_values: Mean latency per window.
        k: K value for NDCG@K.
        output_dir: Directory to save plots.
    """
    # Plot 1: NDCG box plot
    fig, ax = plt.subplots()
    ax.boxplot(ndcg_distributions, tick_labels=window_labels, medianprops=dict(color="black"))

    setup_axis(
        ax,
        xlabel="K",
        ylabel=f"NDCG@{k}",
        ylim=(0, 1),
    )

    save_figure(f"sort_ndcg@{k}", output_dir)
    close_figure()

    # Plot 2: Latency line plot
    plot_sort_time(window_labels, time_values, output_dir)
