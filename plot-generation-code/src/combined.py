import sys
from pathlib import Path

# Allow running directly: python src/combined.py
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.config import METHOD_LABELS, METHOD_COLORS, MARKER_SIZE_NORMAL, METHOD_MARKERS, DATA_PATHS, HGT_DATA_PATHS, \
    DLLM_PATHS, LOTUS_PATHS, POINTWISE_PATHS, METHOD_ORDER, LTFILTER_COLORS, LTFILTER_LTTOPK_COLORS, \
    LTFILTER_LTTOPK_PATH, MARKER_SIZE_SMALL, GROUND_TRUTH_PATH, LLM_GT_METRICS_PATH
from src.data_loader import load_metrics, load_results
from src.metrics import compute_recalls_for_results, compute_ndcg_for_results
from src.plots import save_figure
from src.plots.base import PlotMetadata, LEGEND_FONTSIZE, close_figure
from src.plots.scatter import MethodResult, draw_pareto_frontier


import matplotlib.font_manager as fm
font_path = '/home/jwc/.fonts/LinLibertine_R.otf'
font_path_bold = '/home/jwc/.fonts/LinLibertine_RB.otf'
fm.fontManager.addfont(font_path)
fm.fontManager.addfont(font_path_bold)
font_prop = fm.FontProperties(fname=font_path, size=9)
font_name = font_prop.get_name()


def generate_combined_main_plot(output_dir: Path, k_rrf: float | None = None) -> None:
    """Generate combined 2x2 subplot with all 4 main comparison plots.

    Creates a single figure with:
    - Top-left: Human Labels, Selection (HGT Recall@10)
    - Top-right: Human Labels, Selection+Sort (HGT NDCG@10)
    - Bottom-left: LLM-as-Judge Labels, Selection (LLM Recall@10)
    - Bottom-right: LLM-as-Judge Labels, Selection+Sort (LLM NDCG@10)

    Uses a single shared legend outside the plot area.

    Args:
        output_dir: Directory to save plots.
        k_rrf: If provided, uses RRF-style graded relevance for NDCG.
    """
    print("Generating combined main plot...")

    # Build HGT method results
    def build_hgt_methods(metric_column: str) -> list[MethodResult]:
        method_data: dict[str, MethodResult] = {}

        def make_method(key: str, metrics_path: Path) -> None:
            df = load_metrics(metrics_path)
            method_data[key] = MethodResult(
                label=METHOD_LABELS[key],
                color=METHOD_COLORS[key],
                time=df["time"].iloc[0],
                metric_value=df[metric_column].iloc[0],
                marker_size=MARKER_SIZE_NORMAL,
                marker=METHOD_MARKERS[key],
            )

        make_method("tournament", DATA_PATHS["tourk5000"] / "bier_metrics_tour_10_25.csv")
        make_method("lmpqselect", HGT_DATA_PATHS["base"] / "bier_metrics_sorted_16_2_20_2_10_25.csv")
        make_method("lmpqselect_zephyr7b", DLLM_PATHS["zephyr7b"] / "bier_metrics_sorted_16_2_20_2_10_25.csv")
        make_method("lmpqselect_qwen3_8b", DLLM_PATHS["qwen3_8b"] / "bier_metrics_sorted_16_2_20_2_10_25.csv")
        make_method("pairwise", HGT_DATA_PATHS["pairwise"] / "bier_metrics_sorted_1_1_2_2_10_25.csv")
        make_method("lotus_qwen", LOTUS_PATHS["qwen3_8b"] / "bier_metrics_lotus_10_25.csv")
        make_method("lotus_zephyr", LOTUS_PATHS["zephyr_7b"] / "bier_metrics_lotus_10_25.csv")
        make_method("pointwise_qwen", POINTWISE_PATHS["qwen3_8b"] / "bier_metrics_lotus_10_25.csv")
        make_method("pointwise_zephyr", POINTWISE_PATHS["zephyr"] / "bier_metrics_lotus_10_25.csv")

        # Build ordered list
        methods = []
        for method_key in METHOD_ORDER:
            if method_key == "ltfilter":
                s_configs = [
                    (1, HGT_DATA_PATHS["l1"] / "bier_metrics_sorted_16_2_1_2_10_25.csv"),
                    (5, HGT_DATA_PATHS["l5"] / "bier_metrics_sorted_16_2_5_2_10_25.csv"),
                    (10, HGT_DATA_PATHS["l10"] / "bier_metrics_sorted_16_2_10_2_10_25.csv"),
                    (15, HGT_DATA_PATHS["l15"] / "bier_metrics_sorted_16_2_15_2_10_25.csv"),
                ]
                for s_val, metrics_path in s_configs:
                    df = load_metrics(metrics_path)
                    methods.append(MethodResult(
                        label=f"{METHOD_LABELS['ltfilter']} (S={s_val})",
                        color=LTFILTER_COLORS[s_val],
                        time=df["time"].iloc[0],
                        metric_value=df[metric_column].iloc[0],
                        marker_size=MARKER_SIZE_SMALL,
                        line_group=METHOD_LABELS["ltfilter"],
                    ))
            elif method_key == "ltfilter_lttopk":
                # LTFilter + Tournament Top-K (S=1, 5, 10, 15)
                s_configs_lttopk = [
                    (1, LTFILTER_LTTOPK_PATH / "bier_metrics_tour_10_1_25.csv"),
                    (5, LTFILTER_LTTOPK_PATH / "bier_metrics_tour_10_5_25.csv"),
                    (10, LTFILTER_LTTOPK_PATH / "bier_metrics_tour_10_10_25.csv"),
                    (15, LTFILTER_LTTOPK_PATH / "bier_metrics_tour_10_15_25.csv"),
                ]
                for s_val, metrics_path in s_configs_lttopk:
                    if not metrics_path.exists():
                        continue
                    df = load_metrics(metrics_path)
                    methods.append(MethodResult(
                        label=f"{METHOD_LABELS['ltfilter_lttopk']} (S={s_val})",
                        color=LTFILTER_LTTOPK_COLORS[s_val],
                        time=df["time"].iloc[0],
                        metric_value=df[metric_column].iloc[0],
                        marker_size=MARKER_SIZE_SMALL,
                        line_group=METHOD_LABELS["ltfilter_lttopk"],
                    ))
            elif method_key in method_data:
                methods.append(method_data[method_key])

        return methods

    # Build LLM GT method results
    def build_llm_methods() -> tuple[list[MethodResult], list[MethodResult]]:
        def compute_method_metrics(path: Path) -> tuple[float, float, float]:
            results = load_results(path)
            recalls = compute_recalls_for_results(results, GROUND_TRUTH_PATH, 10)
            ndcgs = compute_ndcg_for_results(results, GROUND_TRUTH_PATH, 10, k_rrf=k_rrf)
            mean_time = np.mean(results[0].times)
            return mean_time, np.mean(recalls), np.mean(ndcgs)

        def load_baseline_metrics(agg_path: Path) -> tuple[float, float, float] | None:
            if not agg_path.exists():
                return None
            df_agg = pd.read_csv(agg_path)
            return df_agg["time"].iloc[0], df_agg["Recall@10"].iloc[0], df_agg["NDCG@10"].iloc[0]

        method_data: dict[str, tuple[float, float, float]] = {}

        # Tournament Top-K
        tournament_metrics = load_baseline_metrics(LLM_GT_METRICS_PATH / "tournament_metrics_llm_agg_10_25.csv")
        if tournament_metrics:
            method_data["tournament"] = tournament_metrics

        # LMPQSelect/Sort (RankZephyr)
        path = HGT_DATA_PATHS["base"] / "bier_sorted_10_16_2_20_2.csv"
        if path.exists():
            method_data["lmpqselect"] = compute_method_metrics(path)

        # LMPQSelect/Sort (Zephyr7b)
        path = DLLM_PATHS["zephyr7b"] / "bier_sorted_10_16_2_20_2.csv"
        if path.exists():
            method_data["lmpqselect_zephyr7b"] = compute_method_metrics(path)

        # LMPQSelect/Sort (Qwen3-8b)
        path = DLLM_PATHS["qwen3_8b"] / "bier_sorted_10_16_2_20_2.csv"
        if path.exists():
            method_data["lmpqselect_qwen3_8b"] = compute_method_metrics(path)

        # Pairwise Quickselect/Sort
        path = HGT_DATA_PATHS["pairwise"] / "bier_sorted_10_1_1_2_2.csv"
        if path.exists():
            method_data["pairwise"] = compute_method_metrics(path)

        # LOTUS (QWEN3-8B)
        lotus_qwen_metrics = load_baseline_metrics(LLM_GT_METRICS_PATH / "lotus_qwen_metrics_llm_agg_10_20.csv")
        if lotus_qwen_metrics:
            method_data["lotus_qwen"] = lotus_qwen_metrics

        # LOTUS (Zephyr-7B)
        lotus_zephyr_metrics = load_baseline_metrics(LLM_GT_METRICS_PATH / "lotus_zephyr_metrics_llm_agg_10_24.csv")
        if lotus_zephyr_metrics:
            method_data["lotus_zephyr"] = lotus_zephyr_metrics

        # Pointwise (QWEN3-8B)
        pw_qwen_metrics = load_baseline_metrics(LLM_GT_METRICS_PATH / "pointwise_qwen_metrics_llm_agg_10_25.csv")
        if pw_qwen_metrics:
            method_data["pointwise_qwen"] = pw_qwen_metrics

        # Pointwise (Zephyr-7B)
        pw_zephyr_metrics = load_baseline_metrics(LLM_GT_METRICS_PATH / "pointwise_zephyr_metrics_llm_agg_10_25.csv")
        if pw_zephyr_metrics:
            method_data["pointwise_zephyr"] = pw_zephyr_metrics

        # Build ordered lists
        recall_methods = []
        ndcg_methods = []

        for method_key in METHOD_ORDER:
            if method_key == "ltfilter":
                s_configs = [
                    (1, HGT_DATA_PATHS["l1"] / "bier_sorted_10_16_2_1_2.csv"),
                    (5, HGT_DATA_PATHS["l5"] / "bier_sorted_10_16_2_5_2.csv"),
                    (10, HGT_DATA_PATHS["l10"] / "bier_sorted_10_16_2_10_2.csv"),
                    (15, HGT_DATA_PATHS["l15"] / "bier_sorted_10_16_2_15_2.csv"),
                ]
                for s_val, path in s_configs:
                    if not path.exists():
                        continue
                    mean_time, mean_recall, mean_ndcg = compute_method_metrics(path)
                    recall_methods.append(MethodResult(
                        label=f"{METHOD_LABELS['ltfilter']} (S={s_val})",
                        color=LTFILTER_COLORS[s_val],
                        time=mean_time,
                        metric_value=mean_recall,
                        marker_size=MARKER_SIZE_SMALL,
                        line_group=METHOD_LABELS["ltfilter"],
                    ))
                    ndcg_methods.append(MethodResult(
                        label=f"{METHOD_LABELS['ltfilter']} (S={s_val})",
                        color=LTFILTER_COLORS[s_val],
                        time=mean_time,
                        metric_value=mean_ndcg,
                        marker_size=MARKER_SIZE_SMALL,
                        line_group=METHOD_LABELS["ltfilter"],
                    ))
            elif method_key == "ltfilter_lttopk":
                # LTFilter + Tournament Top-K (S=1, 5, 10, 15)
                s_configs_lttopk = [
                    (1, LTFILTER_LTTOPK_PATH / "bier_result_unsorted_tour_10_1_25.csv"),
                    (5, LTFILTER_LTTOPK_PATH / "bier_result_unsorted_tour_10_5_25.csv"),
                    (10, LTFILTER_LTTOPK_PATH / "bier_result_unsorted_tour_10_10_25.csv"),
                    (15, LTFILTER_LTTOPK_PATH / "bier_result_unsorted_tour_10_15_25.csv"),
                ]
                for s_val, path in s_configs_lttopk:
                    if not path.exists():
                        continue
                    mean_time, mean_recall, mean_ndcg = compute_method_metrics(path)
                    recall_methods.append(MethodResult(
                        label=f"{METHOD_LABELS['ltfilter_lttopk']} (S={s_val})",
                        color=LTFILTER_LTTOPK_COLORS[s_val],
                        time=mean_time,
                        metric_value=mean_recall,
                        marker_size=MARKER_SIZE_SMALL,
                        line_group=METHOD_LABELS["ltfilter_lttopk"],
                    ))
                    ndcg_methods.append(MethodResult(
                        label=f"{METHOD_LABELS['ltfilter_lttopk']} (S={s_val})",
                        color=LTFILTER_LTTOPK_COLORS[s_val],
                        time=mean_time,
                        metric_value=mean_ndcg,
                        marker_size=MARKER_SIZE_SMALL,
                        line_group=METHOD_LABELS["ltfilter_lttopk"],
                    ))
            elif method_key in method_data:
                mean_time, mean_recall, mean_ndcg = method_data[method_key]
                recall_methods.append(MethodResult(
                    label=METHOD_LABELS[method_key],
                    color=METHOD_COLORS[method_key],
                    time=mean_time,
                    metric_value=mean_recall,
                    marker_size=MARKER_SIZE_NORMAL,
                    marker=METHOD_MARKERS[method_key],
                ))
                ndcg_methods.append(MethodResult(
                    label=METHOD_LABELS[method_key],
                    color=METHOD_COLORS[method_key],
                    time=mean_time,
                    metric_value=mean_ndcg,
                    marker_size=MARKER_SIZE_NORMAL,
                    marker=METHOD_MARKERS[method_key],
                ))

        return recall_methods, ndcg_methods

    # Build all method lists
    hgt_recall_methods = build_hgt_methods("Recall@10")
    hgt_ndcg_methods = build_hgt_methods("NDCG@10")
    llm_recall_methods, llm_ndcg_methods = build_llm_methods()

    # Generate combined plot (log scale)
    plot_combined_main_comparison(
        hgt_recall_methods,
        hgt_ndcg_methods,
        llm_recall_methods,
        llm_ndcg_methods,
        "main_combined",
        output_dir,
        show_pareto=True,
    )

    # Generate combined plot (linear scale)
    plot_combined_main_comparison(
        hgt_recall_methods,
        hgt_ndcg_methods,
        llm_recall_methods,
        llm_ndcg_methods,
        "main_combined_linear",
        output_dir,
        log_x=False,
        show_pareto=True,
    )


def plot_combined_main_comparison(
    hgt_recall_methods: list[MethodResult],
    hgt_ndcg_methods: list[MethodResult],
    llm_recall_methods: list[MethodResult],
    llm_ndcg_methods: list[MethodResult],
    output_name: str,
    output_dir: Path | str = Path("."),
    metadata: PlotMetadata | None = None,
    log_x: bool = True,
    show_pareto: bool = False,
) -> None:
    """
    Create a combined 2x2 subplot figure with all 4 main comparison plots.

    Layout:
        Top-left: Human Labels, Selection (Recall@10)
        Top-right: Human Labels, Selection+Sort (NDCG@10)
        Bottom-left: LLM-as-Judge Labels, Selection (Recall@10)
        Bottom-right: LLM-as-Judge Labels, Selection+Sort (NDCG@10)

    Args:
        hgt_recall_methods: Methods for HGT Recall plot.
        hgt_ndcg_methods: Methods for HGT NDCG plot.
        llm_recall_methods: Methods for LLM Recall plot.
        llm_ndcg_methods: Methods for LLM NDCG plot.
        output_name: Base filename for output.
        output_dir: Directory to save the plot.
        metadata: Optional metadata for the plot.
    """
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D

    plt.rcParams['font.family'] = font_name
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.145), dpi=300)

    LABEL_FONTSIZE = 9
    TICK_FONTSIZE = 6
    LEGEND_FS = 7

    # Subplot configurations: (ax, methods, title, ylabel)
    subplot_configs = [
        (axes[0, 0], hgt_recall_methods, "Human Labels, Selection", "Recall@10"),
        (axes[0, 1], hgt_ndcg_methods, "Human Labels, Selection+Sort", "NDCG@10"),
        (axes[1, 0], llm_recall_methods, "LLM-as-Judge Labels, Selection", "Recall@10"),
        (axes[1, 1], llm_ndcg_methods, "LLM-as-Judge Labels, Selection+Sort", "NDCG@10"),
    ]

    # Small vertical nudge for zero-valued points so markers are fully visible
    ZERO_NUDGE = 0.02

    def _round_nice(val):
        """Round to nearest nice number (1 significant figure)."""
        if val <= 0:
            return 0
        magnitude = 10 ** int(np.floor(np.log10(val)))
        return int(round(val / magnitude) * magnitude)

    for ax, methods, title, ylabel in subplot_configs:
        if log_x:
            ax.set_xscale('log')

        x_values = [m.time for m in methods]

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

        # Plot line groups
        for group_name, group_methods in line_groups.items():
            group_methods_sorted = sorted(group_methods, key=lambda m: m.time)
            group_x = [m.time for m in group_methods_sorted]
            group_y = [m.metric_value for m in group_methods_sorted]
            line_color = group_methods_sorted[-1].color

            ax.plot(group_x, group_y, color=line_color, linewidth=0.7, zorder=1)

            for method in group_methods_sorted:
                y = method.metric_value
                if y == 0.0:
                    y = ZERO_NUDGE
                ax.scatter(
                    method.time, y,
                    c=method.color,
                    s=method.marker_size / 2,
                    marker=method.marker,
                    zorder=2,
                )

        # Plot individual methods
        for method in individual_methods:
            y = method.metric_value
            if y == 0.0:
                y = ZERO_NUDGE
            ax.scatter(
                method.time, y,
                c=method.color,
                s=method.marker_size / 2,
                marker=method.marker,
                zorder=3,
            )

        # Set axis limits
        x_data_min = min(x_values)
        x_data_max = max(x_values)
        nice_min = _round_nice(x_data_min)
        nice_max = _round_nice(x_data_max)

        if log_x:
            x_min = x_data_min * 0.8
            x_max = x_data_max * 1.2
            ax.set_xlim(x_min, x_max)

            tick_values = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
            visible_ticks = [t for t in tick_values if x_min <= t <= x_max]
            # Ensure min/max range values appear as ticks
            for v in [nice_min, nice_max]:
                if v > 0 and x_min <= v <= x_max and v not in visible_ticks:
                    visible_ticks.append(v)
            visible_ticks.sort()
            ax.set_xticks(visible_ticks)
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
            ax.ticklabel_format(style='plain', axis='x')
        else:
            ax.set_xlim(0, max(x_values) * 1.05)

        ax.set_ylim(0, 1)
        ax.set_xlabel("Latency (s)", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
        ax.set_title(title, fontsize=LABEL_FONTSIZE, fontweight='bold')
        ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)

        # Grid lines: y-axis every 0.2, x-axis at every major tick
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.grid(True, axis='y', color='gray', linewidth=0.3, alpha=0.4)
        ax.grid(True, axis='x', which='major', color='gray', linewidth=0.3, alpha=0.4)

        if show_pareto:
            all_x = [m.time for m in methods]
            all_y = [m.metric_value for m in methods]
            draw_pareto_frontier(ax, all_x, all_y, linewidth=0.7)

    # Build legend with 5 columns:
    # Col 1: LTFilter+LTTopK, LTFilter+LMPQ
    # Col 2: LTTopK (RankZephyr), LMPQ (RankZephyr)
    # Col 3: Pairwise, LOTUS Qwen, LOTUS Zephyr
    # Col 4: LMPQ Qwen, LMPQ Zephyr
    # Col 5: Pointwise Qwen, Pointwise Zephyr

    def make_method_handle(method_key):
        marker = METHOD_MARKERS[method_key]
        marker_size = 8 if marker == "*" else 6
        return Line2D([0], [0], marker=marker, color='w',
                      markerfacecolor=METHOD_COLORS[method_key],
                      markersize=marker_size, label=METHOD_LABELS[method_key],
                      linestyle='None')

    def make_blank_handle():
        return Line2D([0], [0], marker='None', color='w',
                      markersize=0, label=' ', linestyle='None')

    # Column-major order (matplotlib fills top-to-bottom, then left-to-right)
    # Pad with blanks to force exact 3×5 grid layout
    # Col 1: LTFilter+LTTopK, LTFilter+LMPQ, (blank)
    # Col 2: LTTopK, LMPQ, (blank)
    # Col 3: Pairwise, LOTUS Qwen, LOTUS Zephyr
    # Col 4: LMPQ Qwen, LMPQ Zephyr, (blank)
    # Col 5: Pointwise Qwen, Pointwise Zephyr, (blank)
    legend_handles = [
        # Col 1
        Line2D([0], [1], marker='o', color=LTFILTER_LTTOPK_COLORS[10],
               markerfacecolor=LTFILTER_LTTOPK_COLORS[10],
               markersize=5, label="LTFilter+LTTopK (RankZephyr)",
               linestyle='-', linewidth=0.7),
        Line2D([0], [1], marker='o', color=LTFILTER_COLORS[5],
               markerfacecolor=LTFILTER_COLORS[5],
               markersize=5, label="LTFilter+LMPQ (RankZephyr)",
               linestyle='-', linewidth=0.7),
        make_blank_handle(),
        # Col 2
        make_method_handle("tournament"),
        make_method_handle("lmpqselect"),
        make_blank_handle(),
        # Col 3
        make_method_handle("pairwise"),
        make_method_handle("lotus_qwen"),
        make_method_handle("lotus_zephyr"),
        # Col 4
        make_method_handle("lmpqselect_qwen3_8b"),
        make_method_handle("lmpqselect_zephyr7b"),
        make_blank_handle(),
        # Col 5
        make_method_handle("pointwise_qwen"),
        make_method_handle("pointwise_zephyr"),
        make_blank_handle(),
    ]

    # Explicit layout for consistent thin margins.
    #
    # Target: uniform visual padding (~PAD fig-frac) from outermost ink
    # to figure edge on every side.
    #
    # Estimated label extents beyond the axis frame (fig-frac):
    #   left   – ylabel + ytick labels  ≈ 0.050
    #   bottom – xlabel + xtick labels  ≈ 0.050
    #   right  – rightmost tick overhang ≈ 0.005
    #   top    – subplot title above frame ≈ 0.030
    #
    # Legend: 3 rows @ 7 pt ≈ 0.085 fig-frac tall.
    #
    PAD = 0.015                       # uniform ink-to-edge padding
    LEFT_LABELS   = 0.050
    BOTTOM_LABELS = 0.050
    RIGHT_LABELS  = 0.005
    LEGEND_HEIGHT = 0.085
    TITLE_HEIGHT  = 0.045

    left   = PAD + LEFT_LABELS                                  # 0.065
    right  = 1 - PAD - RIGHT_LABELS                             # 0.980
    bottom = PAD + BOTTOM_LABELS                                # 0.065
    top    = 1 - PAD - LEGEND_HEIGHT - PAD - TITLE_HEIGHT       # 0.860
    legend_y = 1 - PAD                                          # 0.985

    fig.subplots_adjust(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=0.25,
        hspace=0.43,
    )

    leg = fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, legend_y),
        fontsize=LEGEND_FS,
        frameon=True,
        ncol=5,
        columnspacing=0.6,
        handletextpad=0.2,
    )

    # Bold our methods in the legend (excluding pairwise)
    bold_labels = {
        METHOD_LABELS["tournament"],
        METHOD_LABELS["lmpqselect"],
        "LTFilter+LMPQ (RankZephyr)",
        "LTFilter+LTTopK (RankZephyr)",
    }
    for text in leg.get_texts():
        if text.get_text() in bold_labels:
            text.set_fontweight('bold')

    save_figure(output_name, output_dir, metadata=metadata, bbox_inches=None)
    close_figure()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate combined main comparison plot.")
    parser.add_argument(
        "--k-rrf", type=float, default=None,
        help="RRF constant for graded NDCG relevance (e.g. 60). Default: binary relevance.",
    )
    args = parser.parse_args()
    generate_combined_main_plot(Path('plots/main'), k_rrf=args.k_rrf)
