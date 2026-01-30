"""
Scatter plot functions.

This module provides functions for creating scatter plots comparing
different methods on latency vs. metric axes.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .base import (
    PlotMetadata,
    save_figure,
    close_figure,
    setup_axis,
    add_trend_line,
    format_latency_label,
    FIGURE_SIZE_SMALL,
    LEGEND_FONTSIZE,
)

import matplotlib as mpl
import matplotlib.font_manager as fm

_font_path = '/home/jwc/.fonts/LinLibertine_R.otf'
fm.fontManager.addfont(_font_path)
_font_prop = fm.FontProperties(fname=_font_path)
_font_name = _font_prop.get_name()


def draw_pareto_frontier(
    ax,
    x_values: list[float],
    y_values: list[float],
    x_max: float | None = None,
    linewidth: float = 1.5,
) -> None:
    """Draw a Pareto frontier line on the given axes.

    The frontier connects Pareto-optimal points (minimize x, maximize y)
    and extends to (0, 0) on the left and (x_max, max_y) on the right.

    Args:
        ax: Matplotlib axes to draw on.
        x_values: X coordinates (latency).
        y_values: Y coordinates (metric value).
        x_max: Right extension limit. If None, uses the axes xlim.
    """
    if len(x_values) < 2:
        return

    # Compute Pareto frontier: sort by x ascending, keep points with
    # strictly increasing y (no other point is both faster and better)
    points = sorted(zip(x_values, y_values), key=lambda p: (p[0], -p[1]))
    frontier = []
    max_y = float("-inf")
    for x, y in points:
        if y > max_y:
            frontier.append((x, y))
            max_y = y

    if not frontier:
        return

    # Extend: (0, 0) -> frontier -> (x_max, max_y_on_frontier)
    # On a log x-axis, x=0 is undefined; use the left axis limit instead.
    if x_max is None:
        x_max = ax.get_xlim()[1]
    x_left = ax.get_xlim()[0] if ax.get_xscale() == "log" else 0

    fx = [x_left] + [p[0] for p in frontier] + [x_max]
    fy = [0] + [p[1] for p in frontier] + [frontier[-1][1]]

    ax.plot(fx, fy, color="red", linestyle=":", linewidth=linewidth, zorder=1)


@dataclass
class MethodResult:
    """
    Results for a single method in comparison plots.

    Attributes:
        label: Display name for the method.
        color: Color for plotting.
        time: Mean latency in seconds.
        metric_value: Mean metric value (recall, NDCG, etc.).
        marker_size: Size of the marker (default 36).
        line_group: If set, methods with same line_group will be connected.
        marker: Matplotlib marker style (default "o" for circle).
    """
    label: str
    color: str
    time: float
    metric_value: float
    marker_size: float = 36
    line_group: str | None = None
    marker: str = "o"


def plot_method_comparison_advanced(
    methods: list[MethodResult],
    metric_name: str,
    output_name: str,
    output_dir: Path | str = Path("."),
    show_trend_line: bool = False,
    xlim_padding: float = 500,
    metadata: PlotMetadata | None = None,
    line_group_label: str | None = None,
    log_x: bool = True,
    show_pareto: bool = False,
) -> None:
    """
    Create a scatter plot comparing multiple methods with advanced features.

    Supports connecting methods in the same line_group with a line,
    different marker sizes, and logarithmic x-axis.

    Args:
        methods: List of method results to plot.
        metric_name: Name of the metric for y-axis label.
        output_name: Base filename for output.
        output_dir: Directory to save the plot.
        show_trend_line: If True, add a linear trend line.
        xlim_padding: Padding to add to x-axis max (ignored if log_x=True).
        metadata: Optional metadata for the plot.
        line_group_label: Label to use for the line group in legend.
        log_x: If True, use logarithmic x-axis scale.
        show_pareto: If True, draw Pareto frontier line.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SMALL)

    x_values = [m.time for m in methods]
    y_values = [m.metric_value for m in methods]

    # Set log scale for x-axis
    if log_x:
        ax.set_xscale('log')

    # Separate methods into line groups and individual points
    line_groups: dict[str, list[MethodResult]] = {}
    individual_methods: list[MethodResult] = []

    for method in methods:
        if method.line_group:
            if method.line_group not in line_groups:
                line_groups[method.line_group] = []
            line_groups[method.line_group].append(method)
        else:
            individual_methods.append(method)

    # Plot line groups (behind individual points)
    for group_name, group_methods in line_groups.items():
        # Sort by time for proper line connection
        group_methods_sorted = sorted(group_methods, key=lambda m: m.time)

        group_x = [m.time for m in group_methods_sorted]
        group_y = [m.metric_value for m in group_methods_sorted]

        # Use the darkest color (last in sorted list) for the line
        line_color = group_methods_sorted[-1].color

        # Plot line connecting points (thick enough to be visible)
        ax.plot(group_x, group_y, color=line_color, linewidth=2.0, zorder=1)

        # Plot points with individual gradient colors
        # Build legend label that shows group name and S parameter info
        if line_group_label:
            legend_label = line_group_label
        else:
            # Extract S values from method labels if present
            s_values = []
            for m in group_methods_sorted:
                if "S=" in m.label:
                    s_val = m.label.split("S=")[1].split(")")[0]
                    s_values.append(s_val)
            if s_values:
                legend_label = f"{group_name}\n(S={','.join(s_values)})"
            else:
                legend_label = group_name

        for i, method in enumerate(group_methods_sorted):
            # Only add label for first point to avoid duplicate legend entries
            label = legend_label if i == 0 else None
            ax.scatter(
                method.time,
                method.metric_value,
                c=method.color,
                s=method.marker_size,
                marker=method.marker,
                label=label,
                zorder=2,
            )

    # Plot individual methods
    for method in individual_methods:
        ax.scatter(
            method.time,
            method.metric_value,
            c=method.color,
            s=method.marker_size,
            marker=method.marker,
            label=method.label,
            zorder=3,
        )

    ax.legend(fontsize=LEGEND_FONTSIZE, loc='best')

    if show_trend_line and len(methods) >= 2:
        add_trend_line(ax, x_values, y_values)

    # Set axis limits
    if log_x:
        # For log scale, set reasonable bounds
        x_min = min(x_values) * 0.8
        x_max = max(x_values) * 1.2
        ax.set_xlim(x_min, x_max)

        # Add more tick marks for readability
        import matplotlib.ticker as ticker
        # Use fixed ticks at readable values
        tick_values = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        # Filter to only include ticks within our range
        visible_ticks = [t for t in tick_values if x_min <= t <= x_max]
        ax.set_xticks(visible_ticks)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='x')
    else:
        ax.set_xlim(0, max(x_values) + xlim_padding)

    ax.set_ylim(0, 1)
    ax.set_xlabel(format_latency_label(), fontsize=LEGEND_FONTSIZE)
    ax.set_ylabel(metric_name, fontsize=LEGEND_FONTSIZE)

    if show_pareto:
        draw_pareto_frontier(ax, x_values, y_values)

    save_figure(output_name, output_dir, metadata=metadata)
    close_figure()


def plot_method_comparison(
    methods: list[MethodResult],
    metric_name: str,
    output_name: str,
    output_dir: Path | str = Path("."),
    show_trend_line: bool = False,
    xlim_padding: float = 500,
    metadata: PlotMetadata | None = None,
    log_x: bool = True,
    show_pareto: bool = False,
) -> None:
    """
    Create a scatter plot comparing multiple methods.

    This is the main comparison plot showing latency (x) vs a metric (y)
    for different algorithm configurations.

    Args:
        methods: List of method results to plot.
        metric_name: Name of the metric for y-axis label (e.g., 'Recall@10').
        output_name: Base filename for output.
        output_dir: Directory to save the plot.
        show_trend_line: If True, add a linear trend line.
        xlim_padding: Padding to add to x-axis max (ignored if log_x=True).
        metadata: Optional metadata for the plot.
        log_x: If True, use logarithmic x-axis scale.
        show_pareto: If True, draw Pareto frontier line.
    """
    # Check if any methods have line groups
    has_line_groups = any(m.line_group for m in methods)

    if has_line_groups:
        plot_method_comparison_advanced(
            methods, metric_name, output_name, output_dir,
            show_trend_line, xlim_padding, metadata, log_x=log_x,
            show_pareto=show_pareto,
        )
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SMALL)

    # Set log scale for x-axis
    if log_x:
        ax.set_xscale('log')

    x_values = [m.time for m in methods]
    y_values = [m.metric_value for m in methods]

    # Plot each method as a scatter point
    for method in methods:
        ax.scatter(
            method.time,
            method.metric_value,
            c=method.color,
            s=method.marker_size,
            marker=method.marker,
            label=method.label,
        )

    ax.legend(fontsize=LEGEND_FONTSIZE)

    if show_trend_line and len(methods) >= 2:
        add_trend_line(ax, x_values, y_values)

    # Set axis limits
    if log_x:
        x_min = min(x_values) * 0.8
        x_max = max(x_values) * 1.2
        ax.set_xlim(x_min, x_max)

        # Add more tick marks for readability
        import matplotlib.ticker as ticker
        tick_values = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        visible_ticks = [t for t in tick_values if x_min <= t <= x_max]
        ax.set_xticks(visible_ticks)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='x')
    else:
        ax.set_xlim(0, max(x_values) + xlim_padding)

    ax.set_ylim(0, 1)
    ax.set_xlabel(format_latency_label(), fontsize=LEGEND_FONTSIZE)
    ax.set_ylabel(metric_name, fontsize=LEGEND_FONTSIZE)

    if show_pareto:
        draw_pareto_frontier(ax, x_values, y_values)

    save_figure(output_name, output_dir, metadata=metadata)
    close_figure()


def plot_method_comparison_from_metrics(
    metrics_data: list[tuple[str, str, pd.DataFrame]],
    metric_column: str,
    metric_name: str,
    output_name: str,
    output_dir: Path | str = Path("."),
    show_trend_line: bool = False,
    metadata: PlotMetadata | None = None,
) -> None:
    """
    Create comparison plot from pre-loaded metrics DataFrames.

    Args:
        metrics_data: List of (label, color, DataFrame) tuples.
        metric_column: Column name for the metric (e.g., 'Recall@10').
        metric_name: Display name for y-axis.
        output_name: Base filename for output.
        output_dir: Directory to save the plot.
        show_trend_line: If True, add a linear trend line.
        metadata: Optional metadata for the plot.
    """
    methods = []
    for label, color, df in metrics_data:
        methods.append(MethodResult(
            label=label,
            color=color,
            time=df["time"].iloc[0],
            metric_value=df[metric_column].iloc[0],
        ))

    plot_method_comparison(
        methods,
        metric_name,
        output_name,
        output_dir,
        show_trend_line,
        metadata=metadata,
    )


def plot_window_size_analysis(
    l_values: list[int],
    human_recall_means: list[float],
    llm_recall_means: list[float],
    latency_means: list[float],
    output_dir: Path | str = Path("."),
) -> None:
    """
    Create plots for window size analysis.

    Generates two plots:
    1. List Size (L) vs Recall@10 with Human and LLM GT lines and quadratic fits
    2. List Size (L) vs Latency with mean values

    Args:
        l_values: List of L values.
        human_recall_means: Mean recall for human ground truth.
        llm_recall_means: Mean recall for LLM ground truth.
        latency_means: Mean latency values.
        output_dir: Directory to save plots.
    """
    import numpy as np
    from matplotlib.lines import Line2D

    _rc = {
        'font.family': _font_name,
        'axes.labelsize': 9,
        'legend.fontsize': 9,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
    }
    with mpl.rc_context(_rc):
        # Plot 1: L vs Recall with both GT dots and quadratic fits
        fig, ax = plt.subplots(figsize=(2.24, 1.68))
        fig.subplots_adjust(left=0.19, right=0.97, top=0.97, bottom=0.21)

        l_arr = np.array(l_values)
        x_line = np.linspace(min(l_values), max(l_values), 100)

        # Human GT - dots only
        ax.scatter(l_values, human_recall_means, color="#0077BB", s=10, marker="o", zorder=2)
        # Quadratic fit for Human GT
        human_coeffs = np.polyfit(l_arr, human_recall_means, 2)
        human_fit = np.polyval(human_coeffs, x_line)
        ax.plot(x_line, human_fit, color="#0077BB", linestyle="-", linewidth=1, zorder=1)

        # LLM GT - dots only
        ax.scatter(l_values, llm_recall_means, color="#EE3377", s=10, marker="s", zorder=2)
        # Quadratic fit for LLM GT
        llm_coeffs = np.polyfit(l_arr, llm_recall_means, 2)
        llm_fit = np.polyval(llm_coeffs, x_line)
        ax.plot(x_line, llm_fit, color="#EE3377", linestyle="--", linewidth=1, zorder=1)

        setup_axis(
            ax,
            xlabel="List Size (L)",
            ylabel="Recall@10",
            ylim=(0, 1),
            fontsize=9,
        )
        ax.tick_params(axis='both', labelsize=6)

        # Legend
        legend_elements = [
            Line2D([0], [0], color="#0077BB", linestyle="-", linewidth=1, marker="o", markersize=3, label="Human Labels (R=1)"),
            Line2D([0], [0], color="#EE3377", linestyle="--", linewidth=1, marker="s", markersize=3, label="LLM-as-Judge Labels (R=10)"),
        ]
        ax.legend(handles=legend_elements, loc="best", fontsize=6, borderaxespad=0.3)

        save_figure("w_recall@10", output_dir, bbox_inches=None)
        close_figure()

        # Plot 2: L vs Latency with dots and custom fit: alpha/L + beta + gamma*L
        fig, ax = plt.subplots(figsize=(2.24, 1.68))
        fig.subplots_adjust(left=0.19, right=0.97, top=0.97, bottom=0.21)

        # Dots only
        ax.scatter(l_values, latency_means, color="black", s=10, marker="o", zorder=2)

        # Custom fit: f(L) = alpha/L + beta + gamma*L
        # Rewrite as: y = alpha * (1/L) + beta * 1 + gamma * L
        # This is linear in the parameters, so use least squares
        l_arr = np.array(l_values, dtype=float)
        latency_arr = np.array(latency_means)

        # Design matrix: [1/L, 1, L]
        A = np.column_stack([1/l_arr, np.ones_like(l_arr), l_arr])
        # Solve least squares: A @ [alpha, beta, gamma] = latency
        coeffs, _, _, _ = np.linalg.lstsq(A, latency_arr, rcond=None)
        alpha, beta, gamma = coeffs

        x_fit = np.linspace(min(l_values), max(l_values), 100)
        y_fit = alpha / x_fit + beta + gamma * x_fit
        ax.plot(x_fit, y_fit, color="black", linewidth=1, zorder=1)

        setup_axis(
            ax,
            xlabel="List Size (L)",
            ylabel="Latency (s)",
            ylim=(0, max(latency_means) * 1.1),
            fontsize=9,
        )
        ax.tick_params(axis='both', labelsize=6)

        save_figure("w_latency", output_dir, bbox_inches=None)
        close_figure()


def plot_wsort_comparison(
    l_values: list[int],
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
    Create line plot for sort window comparison.

    Plots L vs NDCG@10 with confidence interval for both Human and LLM GT.

    Args:
        l_values: List of L values.
        human_ndcg_values: NDCG values computed against human ground truth.
        llm_ndcg_values: NDCG values computed against LLM ground truth.
        output_dir: Directory to save the plot.
        human_std: Standard deviations for human NDCG.
        llm_std: Standard deviations for LLM NDCG.
        human_bounds: Optional tuple of (lower, upper) bounds for human CI.
        llm_bounds: Optional tuple of (lower, upper) bounds for LLM CI.
        ci_label: Label for the confidence interval in the legend.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams['font.family'] = _font_name
    fig, ax = plt.subplots(figsize=(1.715, 1.4))
    fig.subplots_adjust(left=0.20, right=0.98, top=0.98, bottom=0.19)

    import numpy as np

    l_arr = np.array(l_values, dtype=float)
    x_fit = np.linspace(l_arr.min(), l_arr.max(), 100)

    # Plot human GT: dots + linear fit
    ax.scatter(l_values, human_ndcg_values, color="#0077BB", s=5, marker="o", zorder=3)
    human_coeffs = np.polyfit(l_arr, human_ndcg_values, 1)
    ax.plot(x_fit, np.polyval(human_coeffs, x_fit), color="#0077BB", linestyle="-", linewidth=0.5, zorder=2)

    # Plot LLM GT: dots + linear fit
    ax.scatter(l_values, llm_ndcg_values, color="#EE3377", s=5, marker="o", zorder=3)
    llm_coeffs = np.polyfit(l_arr, llm_ndcg_values, 1)
    ax.plot(x_fit, np.polyval(llm_coeffs, x_fit), color="#EE3377", linestyle="--", linewidth=0.5, zorder=2)

    ax.set_xlabel("List Size (L)", fontsize=6, labelpad=2)
    ax.set_ylabel("NDCG@10", fontsize=6, labelpad=2)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis='both', labelsize=5)

    # Legend
    legend_elements = [
        Line2D([0], [0], color="#0077BB", linestyle="-", linewidth=0.5, marker="o", markersize=3, label="Human Labels (R=1)"),
        Line2D([0], [0], color="#EE3377", linestyle="--", linewidth=0.5, marker="o", markersize=3, label="LLM-as-Judge (R=10)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=6)

    save_figure("sort_L_ndcg@10", output_dir, bbox_inches=None)
    close_figure()


