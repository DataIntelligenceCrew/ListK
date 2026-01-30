"""
Plot comparing LMPQSelect/Sort with and without pairwise rerank heuristic.

Shows the impact of applying pairwise quicksort to re-rank the top-10 results
after LMPQSort, with arrows indicating the improvement in NDCG.
"""

from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


@dataclass
class MethodMetrics:
    """Metrics for a single method configuration."""
    label: str
    time: float
    ndcg: float
    recall: float


def plot_pairwise_rerank_impact(
    output_dir: Path,
    filename: str = "pairwise_rerank_impact",
) -> None:
    """
    Create a plot showing the impact of pairwise rerank heuristic.

    Shows LMPQSelect/Sort with and without pairwise rerank, with arrows
    indicating the improvement in NDCG for both HGT and LLM ground truth.
    """
    # Data: Without pairwise rerank (L=20, W=20)
    old_hgt = MethodMetrics(
        label="LMPQSelect/Sort (W=20)",
        time=455.48,
        ndcg=0.46501,
        recall=0.88,
    )
    old_llm = MethodMetrics(
        label="LMPQSelect/Sort (W=20)",
        time=455.48,
        ndcg=0.4468,
        recall=0.42,
    )

    # Data: With pairwise rerank (L=20, W=2)
    new_hgt = MethodMetrics(
        label="LMPQSelect/Sort + Pairwise (W=2)",
        time=461.13,
        ndcg=0.84136,
        recall=0.88,
    )
    new_llm = MethodMetrics(
        label="LMPQSelect/Sort + Pairwise (W=2)",
        time=461.13,
        ndcg=0.5167,
        recall=0.42,
    )

    # Create single figure
    fig, ax = plt.subplots(figsize=(7, 5))

    # Canonical colors
    color_human = "#0077BB"  # Blue for Human GT
    color_llm = "#EE3377"    # Red/magenta for LLM GT

    # Marker settings
    marker_size = 100
    marker_old = 'o'  # Circle for old (without pairwise)
    marker_new = 's'  # Square for new (with pairwise)

    # Plot Human GT points
    ax.scatter(old_hgt.time, old_hgt.ndcg, s=marker_size, c=color_human,
               marker=marker_old, zorder=3, edgecolors='white', linewidths=1)
    ax.scatter(new_hgt.time, new_hgt.ndcg, s=marker_size, c=color_human,
               marker=marker_new, zorder=3, edgecolors='white', linewidths=1)

    # Plot LLM GT points
    ax.scatter(old_llm.time, old_llm.ndcg, s=marker_size, c=color_llm,
               marker=marker_old, zorder=3, edgecolors='white', linewidths=1)
    ax.scatter(new_llm.time, new_llm.ndcg, s=marker_size, c=color_llm,
               marker=marker_new, zorder=3, edgecolors='white', linewidths=1)

    # Draw arrows from old to new
    # Human GT arrow
    ax.annotate(
        '',
        xy=(new_hgt.time, new_hgt.ndcg),
        xytext=(old_hgt.time, old_hgt.ndcg),
        arrowprops=dict(
            arrowstyle='->',
            color=color_human,
            lw=2,
            connectionstyle='arc3,rad=0',
        ),
        zorder=2,
    )

    # LLM GT arrow
    ax.annotate(
        '',
        xy=(new_llm.time, new_llm.ndcg),
        xytext=(old_llm.time, old_llm.ndcg),
        arrowprops=dict(
            arrowstyle='->',
            color=color_llm,
            lw=2,
            connectionstyle='arc3,rad=0',
        ),
        zorder=2,
    )

    # Formatting
    ax.set_xlabel("Latency (s)", fontsize=12)
    ax.set_ylabel("NDCG@10", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Set axis limits
    ax.set_xlim(400, 500)
    ax.set_ylim(0, 1)

    # Create legend with custom handles
    legend_elements = [
        Line2D([0], [0], color=color_human, linestyle='-', linewidth=2,
               marker='o', markersize=8, label='Human Labels'),
        Line2D([0], [0], color=color_llm, linestyle='-', linewidth=2,
               marker='o', markersize=8, label='LLM-as-Judge Labels'),
        Line2D([0], [0], color='gray', linestyle='', marker='o', markersize=8,
               markerfacecolor='gray', label='Without top-few refinement heuristic'),
        Line2D([0], [0], color='gray', linestyle='', marker='s', markersize=8,
               markerfacecolor='gray', label='With top-few refinement heuristic'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()

    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)

    for ext in ['png', 'pdf']:
        output_path = output_dir / f"{filename}.{ext}"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close(fig)

    # Save metadata
    import json
    metadata = {
        "title": "Impact of Pairwise Rerank Heuristic",
        "description": "Compares LMPQSelect/Sort with and without pairwise quicksort reranking of top-10 results",
        "data": {
            "without_pairwise": {
                "config": "L=20, W=20",
                "hgt_ndcg": old_hgt.ndcg,
                "llm_ndcg": old_llm.ndcg,
                "time": old_hgt.time,
            },
            "with_pairwise": {
                "config": "L=20, W=2",
                "hgt_ndcg": new_hgt.ndcg,
                "llm_ndcg": new_llm.ndcg,
                "time": new_hgt.time,
            },
            "improvement": {
                "hgt_ndcg_delta": new_hgt.ndcg - old_hgt.ndcg,
                "llm_ndcg_delta": new_llm.ndcg - old_llm.ndcg,
                "time_delta": new_hgt.time - old_hgt.time,
            }
        }
    }

    json_path = output_dir / f"{filename}.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    plot_pairwise_rerank_impact(Path("plots/heuristics"))
