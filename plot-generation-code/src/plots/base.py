"""
Base plotting utilities and common functions.

This module provides shared functionality for all plot types including
figure saving, axis setup, metadata generation, and common styling.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# =============================================================================
# Default Style Constants
# =============================================================================

# Font sizes
AXIS_LABEL_FONTSIZE: int = 14
TICK_LABEL_FONTSIZE: int = 12
LEGEND_FONTSIZE: int = 11

# Default figure sizes (width, height) in inches
FIGURE_SIZE_DEFAULT: tuple[float, float] = (8, 6)
FIGURE_SIZE_SMALL: tuple[float, float] = (6, 5)
FIGURE_SIZE_LARGE: tuple[float, float] = (10, 8)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PlotPoint:
    """
    A single data point for scatter plots.

    Attributes:
        x: X-axis value (typically latency).
        y: Y-axis value (typically a metric like recall or NDCG).
        label: Display label for the legend.
        color: Color for the point.
    """
    x: float
    y: float
    label: str
    color: str


@dataclass
class PlotMetadata:
    """
    Metadata for a generated plot.

    Attributes:
        title: Descriptive title (not shown on plot, for reference).
        xlabel: X-axis label.
        ylabel: Y-axis label.
        description: Brief description of what the plot shows.
        data_sources: List of data files or paths used.
        algorithm: Algorithm or method being evaluated.
        parameters: Dict of relevant parameters.
    """
    title: str
    xlabel: str
    ylabel: str
    description: str = ""
    data_sources: list[str] = field(default_factory=list)
    algorithm: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Metadata Functions
# =============================================================================

def save_metadata(
    metadata: PlotMetadata,
    filename: str,
    output_dir: Path | str = Path("."),
) -> None:
    """
    Save plot metadata to a JSON file.

    Args:
        metadata: PlotMetadata object with plot information.
        filename: Base filename (without extension).
        output_dir: Directory to save the metadata file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f"{filename}.json"
    with open(filepath, "w") as f:
        json.dump(asdict(metadata), f, indent=2)


# =============================================================================
# Figure Functions
# =============================================================================

def save_figure(
    filename: str,
    output_dir: Path | str = Path("."),
    formats: list[str] | None = None,
    bbox_inches: str = "tight",
    metadata: PlotMetadata | None = None,
) -> None:
    """
    Save the current figure in multiple formats, optionally with metadata.

    Args:
        filename: Base filename without extension.
        output_dir: Directory to save figures to.
        formats: List of file formats (default: ["pdf", "png"]).
        bbox_inches: Bounding box setting for savefig.
        metadata: Optional PlotMetadata to save alongside the figure.
    """
    formats = formats or ["pdf", "png"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        filepath = output_dir / f"{filename}.{fmt}"
        # Use high DPI for PNG files
        dpi = 300 if fmt == "png" else None
        plt.savefig(filepath, bbox_inches=bbox_inches, dpi=dpi)

    if metadata:
        save_metadata(metadata, filename, output_dir)


def setup_axis(
    ax: Axes,
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    fontsize: int = AXIS_LABEL_FONTSIZE,
) -> None:
    """
    Configure axis labels and limits (no title - titles go in metadata).

    Args:
        ax: Matplotlib Axes object.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        xlim: Optional (min, max) tuple for x-axis.
        ylim: Optional (min, max) tuple for y-axis.
        fontsize: Font size for axis labels.
    """
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)


def create_figure(
    figsize: tuple[float, float] = FIGURE_SIZE_DEFAULT,
) -> tuple[Figure, Axes]:
    """
    Create a new figure with a single subplot.

    Args:
        figsize: Figure size as (width, height) in inches.

    Returns:
        Tuple of (Figure, Axes) objects.
    """
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def close_figure() -> None:
    """Close the current figure to free memory."""
    plt.close()


def add_trend_line(
    ax: Axes,
    x_values: list[float],
    y_values: list[float],
    linestyle: str = "dotted",
    color: str = "black",
) -> None:
    """
    Add a linear trend line to the plot.

    Args:
        ax: Matplotlib Axes object.
        x_values: X coordinates of data points.
        y_values: Y coordinates of data points.
        linestyle: Line style for the trend line.
        color: Color of the trend line.
    """
    import numpy as np

    if len(x_values) < 2:
        return

    x_arr = np.array(x_values)
    y_arr = np.array(y_values)

    slope, intercept = np.polyfit(x_arr, y_arr, 1)
    ax.plot(x_arr, slope * x_arr + intercept, linestyle=linestyle, color=color)


def format_metric_label(metric: Literal["recall", "ndcg"], k: int) -> str:
    """
    Format a metric label with K value.

    Args:
        metric: The metric type.
        k: The K value.

    Returns:
        Formatted label string like "Recall@10" or "NDCG@10".
    """
    metric_names = {
        "recall": "Recall",
        "ndcg": "NDCG",
    }
    return f"{metric_names[metric]}@{k}"


def format_latency_label() -> str:
    """Return the standard latency axis label."""
    return "Latency (s)"
