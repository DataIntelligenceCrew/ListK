"""
Main entry point for generating all experiment plots.

This module orchestrates the generation of all plots from experiment data.
Run with: python -m src.main

Usage:
    python -m src.main                    # Generate all plots
    python -m src.main --output-dir ./out # Specify output directory
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.combined import generate_combined_main_plot
from .config import (
    CI_TYPE,
    DATA_PATHS,
    DATA_RAW_DIR,
    DLLM_PATHS,
    GROUND_TRUTH_PATH,
    HGT_DATA_PATHS,
    LLM_GT_METRICS_PATH,
    LOTUS_PATHS,
    LTFILTER_COLORS,
    MARKER_SIZE_NORMAL,
    MARKER_SIZE_SMALL,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_MARKERS,
    METHOD_ORDER,
    POINTWISE_PATHS,
    SORT_WINDOW_VALUES,
    TFILTER_DOC_NUMS,
    WSORT_VALUES,
)

import matplotlib.font_manager as fm
font_path = '/home/jwc/.fonts/LinLibertine_R.otf'
fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname=font_path, size=9)
font_name = font_prop.get_name()

def compute_ci_bounds(
    distributions: list[list[float]],
    means: list[float],
    ci_type: str = "std",
) -> tuple[list[float], list[float], str]:
    """
    Compute confidence interval bounds from distributions.

    Args:
        distributions: List of per-query value lists for each data point.
        means: Mean values for each data point.
        ci_type: "std" for ±1 std, "iqr" for 25th-75th percentile.

    Returns:
        Tuple of (lower_bounds, upper_bounds, ci_label).
    """
    if ci_type == "iqr":
        lower = [np.percentile(d, 25) for d in distributions]
        upper = [np.percentile(d, 75) for d in distributions]
        label = "25th-75th Percentile"
    else:  # std
        stds = [np.std(d, ddof=1) for d in distributions]
        lower = [m - s for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]
        label = "1std Confidence Interval"
    return lower, upper, label
from .data_loader import (
    load_metrics,
    load_results,
    load_sort_results,
    load_tfilter_results,
)
from .metrics import (
    calc_stats_for_px_grid,
    calc_stats_for_px_grid_split,
    compute_ndcg_for_results,
    compute_recalls_for_results,
    compute_recalls_tour_for_results,
)
from .plots.heatmaps import plot_px_heatmap, plot_px_diagonal, plot_px_time_comparison
from .plots.scatter import (
    MethodResult,
    plot_method_comparison,
    plot_window_size_analysis,
    plot_wsort_comparison,
)
from .plots.base import PlotMetadata
from .plots.boxplots import (
    plot_embedding_comparison_from_df,
    plot_tournament_filter,
    plot_tournament_filter_summary,
)
from .plots.line import plot_k_recall, plot_k_recall_comparison, plot_sort_metrics, plot_sort_ndcg_line, plot_sort_k_ndcg_comparison


def generate_px_heatmaps(output_dir: Path) -> None:
    """Generate P×X heatmaps and P=X diagonal plots for early stopping and non-early stopping experiments."""
    print("Generating P×X heatmaps...")

    # Early stopping experiments (LLM ground truth)
    e5000 = calc_stats_for_px_grid(
        DATA_PATHS["early_stopping"],
        GROUND_TRUTH_PATH,
    )
    plot_px_heatmap(e5000, "time", "Early Stopping", output_dir)
    plot_px_heatmap(e5000, "recall", "Early Stopping", output_dir)
    plot_px_diagonal(e5000, "time", "Early Stopping", output_dir)
    plot_px_diagonal(e5000, "recall", "Early Stopping", output_dir)

    # Non-early stopping experiments (LLM ground truth)
    ne5000 = calc_stats_for_px_grid_split(
        DATA_PATHS["no_early_stopping"],
        DATA_PATHS["early_stopping"],
        GROUND_TRUTH_PATH,
    )
    plot_px_heatmap(ne5000, "time", "no Early Stopping", output_dir)
    plot_px_heatmap(ne5000, "recall", "no Early Stopping", output_dir)
    plot_px_diagonal(ne5000, "time", "no Early Stopping", output_dir)
    plot_px_diagonal(ne5000, "recall", "no Early Stopping", output_dir)

    # HGT variants (using pre-computed metrics)
    e5000_hgt = calc_stats_for_px_grid(
        DATA_PATHS["early_stopping"],
        GROUND_TRUTH_PATH,
        use_precomputed_metrics=True,
    )
    plot_px_heatmap(e5000_hgt, "recall", "Early Stopping", output_dir, suffix="_hgt")
    plot_px_diagonal(e5000_hgt, "recall", "Early Stopping", output_dir, suffix="_hgt")

    ne5000_hgt = calc_stats_for_px_grid_split(
        DATA_PATHS["no_early_stopping"],
        DATA_PATHS["early_stopping"],
        GROUND_TRUTH_PATH,
        use_precomputed_metrics=True,
    )
    plot_px_heatmap(ne5000_hgt, "recall", "no Early Stopping", output_dir, suffix="_hgt")
    plot_px_diagonal(ne5000_hgt, "recall", "no Early Stopping", output_dir, suffix="_hgt")

    # Combined time comparison plot
    plot_px_time_comparison(e5000, ne5000, output_dir)


def generate_embedding_comparison(output_dir: Path) -> None:
    """Generate embedding vs no-embedding pivot selection comparison."""
    print("Generating embedding comparison box plot...")

    embedding_df = pd.read_csv(
        DATA_PATHS["early_stopping"] / "bier_result_unsorted_16_2_25.csv"
    )
    no_embedding_df = pd.read_csv(
        DATA_PATHS["no_embedding_pivot"] / "bier_result_unsorted_16_2_25.csv"
    )

    plot_embedding_comparison_from_df(embedding_df, no_embedding_df, output_dir)


def generate_tournament_filter_plots(output_dir: Path) -> None:
    """Generate tournament filter plots for different K values."""
    print("Generating tournament filter plots...")

    k_values = [5, 10, 20, 50, 100]
    l_values = [1, 5, 10, 15]
    labels = ["1", "5", "10", "15"]

    qid_path = DATA_PATHS["early_stopping"] / "bier_result_unsorted_16_2_25.csv"

    # Collect data for summary plot
    recall_data = {}

    for k in k_values:
        recall_distributions = []

        for l_val in l_values:
            doc_num = TFILTER_DOC_NUMS[l_val]
            tfilter_path = DATA_PATHS["tfilter"] / f"bier_tfilter_result_{doc_num}_25.csv"

            results = load_tfilter_results(tfilter_path, qid_path)
            recalls = compute_recalls_tour_for_results(results, GROUND_TRUTH_PATH, k)
            recall_distributions.append(recalls)

        plot_tournament_filter(recall_distributions, labels, k, output_dir)

        # Store mean and std for summary plot
        means = [np.mean(recalls) for recalls in recall_distributions]
        stds = [np.std(recalls, ddof=1) for recalls in recall_distributions]
        recall_data[k] = (means, stds)

    # Generate summary plot (exclude K=100)
    summary_k_values = [5, 10, 20, 50]
    summary_recall_data = {k: recall_data[k] for k in summary_k_values}
    plot_tournament_filter_summary(l_values, summary_k_values, summary_recall_data, output_dir)


def generate_k_recall_plot(output_dir: Path) -> None:
    """Generate Recall@K vs K line plot with Human GT and LLM GT."""
    print("Generating K-recall plot...")

    import ast

    # Use K values that have HGT metrics (K=100 doesn't have HGT metrics)
    k_values = [10, 20, 50]
    labels = [str(k) for k in k_values]

    human_recall_values = []
    llm_recall_values = []
    human_distributions = []
    llm_distributions = []

    for k in k_values:
        # LLM GT: compute per-query recall from results
        result_file = DATA_PATHS["k5000"] / f"bier_result_unsorted_16_2_{k}_25.csv"
        results = load_results(result_file)
        llm_recalls = compute_recalls_for_results(results, GROUND_TRUTH_PATH, k)
        llm_recall_values.append(np.mean(llm_recalls))
        llm_distributions.append(llm_recalls)

        # Human GT: parse per-query HGT metrics
        i_metrics_file = DATA_PATHS["k5000"] / f"bier_i_metrics_16_2_{k}_25.csv"
        df = pd.read_csv(i_metrics_file)
        human_recalls = []
        for _, row in df.iterrows():
            recall_list = ast.literal_eval(row["Recall@10"])
            if recall_list:
                human_recalls.append(recall_list[-1])
        human_recall_values.append(np.mean(human_recalls))
        human_distributions.append(human_recalls)

    # Compute CI bounds based on config
    human_lower, human_upper, ci_label = compute_ci_bounds(
        human_distributions, human_recall_values, CI_TYPE
    )
    llm_lower, llm_upper, _ = compute_ci_bounds(
        llm_distributions, llm_recall_values, CI_TYPE
    )

    plot_k_recall(
        k_values,
        human_recall_values,
        llm_recall_values,
        labels,
        output_dir,
        human_bounds=(human_lower, human_upper),
        llm_bounds=(llm_lower, llm_upper),
        ci_label=ci_label,
    )


def generate_k_recall_comparison_plot(output_dir: Path) -> None:
    """Generate Recall@K comparison plot for P=X configurations.

    Compares Human GT (HGT) vs LLM GT recall for the diagonal P=X values.
    Includes confidence interval bands based on CI_TYPE config.
    """
    print("Generating K-recall comparison plot (Human vs LLM)...")

    import ast
    from .config import PIVOT_VALUES

    def parse_hgt_per_query_recalls(metrics_file: Path) -> list[float]:
        """Parse per-query HGT metrics file and extract final recall values."""
        df = pd.read_csv(metrics_file)
        recalls = []
        for _, row in df.iterrows():
            recall_list = ast.literal_eval(row["Recall@10"])
            # Take the last (final) value from the progressive recall list
            if recall_list:
                recalls.append(recall_list[-1])
        return recalls

    # P=X diagonal values
    k_values = PIVOT_VALUES  # [1, 2, 4, 8, 16]

    human_recall_values = []
    llm_recall_values = []
    human_distributions = []
    llm_distributions = []

    for k in k_values:
        # Load result file for P=X=k
        result_file = DATA_PATHS["early_stopping"] / f"bier_result_unsorted_{k}_{k}_25.csv"
        results = load_results(result_file)

        # Human GT: from per-query HGT metrics
        i_metrics_file = DATA_PATHS["early_stopping"] / f"bier_i_metrics_{k}_{k}_25.csv"
        human_recalls = parse_hgt_per_query_recalls(i_metrics_file)
        human_recall_values.append(np.mean(human_recalls))
        human_distributions.append(human_recalls)

        # LLM GT: computed against LLM ground truth
        llm_recalls = compute_recalls_for_results(results, GROUND_TRUTH_PATH, 10)
        llm_recall_values.append(np.mean(llm_recalls))
        llm_distributions.append(llm_recalls)

    # Compute CI bounds based on config
    human_lower, human_upper, ci_label = compute_ci_bounds(
        human_distributions, human_recall_values, CI_TYPE
    )
    llm_lower, llm_upper, _ = compute_ci_bounds(
        llm_distributions, llm_recall_values, CI_TYPE
    )

    plot_k_recall_comparison(
        k_values,
        human_recall_values,
        llm_recall_values,
        output_dir,
        filename="P_recall",
        human_bounds=(human_lower, human_upper),
        llm_bounds=(llm_lower, llm_upper),
        ci_label=ci_label,
    )


def generate_sort_plots(output_dir: Path, k_rrf: float | None = None) -> None:
    """Generate semantic sort NDCG and latency plots."""
    print("Generating sort plots...")

    # Pure semantic sort experiments (K=10 not available in this format)
    sort_windows = [20, 50, 100, 250, 500, 750, 1000]
    window_labels = ["20", "50", "100", "250", "500", "750", "1000"]
    k_values = [5, 10, 20]

    for k in k_values:
        ndcg_distributions = []
        time_values = []

        for window in sort_windows:
            result_file = DATA_PATHS["sort"] / f"bier_sort_result_{window}_25.csv"
            results = load_sort_results(result_file)

            time_values.append(np.mean(results[0].times))
            ndcgs = compute_ndcg_for_results(results, GROUND_TRUTH_PATH, k, k_rrf=k_rrf)
            ndcg_distributions.append(ndcgs)

        plot_sort_metrics(window_labels, ndcg_distributions, time_values, k, output_dir)

    # HGT variant
    ndcg_values = []
    for window in sort_windows:
        metrics_file = DATA_PATHS["sort_hgt_metrics"] / f"bier_metrics_sorted_{window}_{window}_10_25.csv"
        df = load_metrics(metrics_file)
        ndcg_values.append(df["NDCG@10"].iloc[0])

    plot_sort_ndcg_line(window_labels, ndcg_values, output_dir)

    # Generate K vs NDCG comparison (Human GT vs LLM GT)
    generate_sort_k_ndcg_comparison(output_dir, k_rrf=k_rrf)


def generate_sort_k_ndcg_comparison(output_dir: Path, k_rrf: float | None = None) -> None:
    """Generate sort K vs NDCG comparison plot (Human GT vs LLM GT)."""
    print("Generating sort K vs NDCG comparison plot...")

    human_ndcg_values = []
    llm_ndcg_values = []
    llm_distributions = []

    for window in SORT_WINDOW_VALUES:
        # K=10 uses base experiments (L=10, W=20), others use sort experiments
        if window == 10:
            result_file = HGT_DATA_PATHS["l10"] / "bier_sorted_10_16_2_10_20.csv"
            metrics_file = HGT_DATA_PATHS["l10"] / "bier_metrics_sorted_16_2_10_20_10_25.csv"
            results = load_results(result_file)
        else:
            result_file = DATA_PATHS["sort"] / f"bier_sort_result_{window}_25.csv"
            metrics_file = DATA_PATHS["sort_hgt_metrics"] / f"bier_metrics_sorted_{window}_{window}_10_25.csv"
            results = load_sort_results(result_file)

        # LLM GT: compute per-query NDCG from results
        llm_ndcgs = compute_ndcg_for_results(results, GROUND_TRUTH_PATH, 10, k_rrf=k_rrf)
        llm_ndcg_values.append(np.mean(llm_ndcgs))
        llm_distributions.append(llm_ndcgs)

        # Human GT: use pre-computed HGT metrics (aggregate only)
        df = load_metrics(metrics_file)
        human_ndcg_values.append(df["NDCG@10"].iloc[0])

    # Compute CI bounds for LLM (we have per-query data)
    llm_lower, llm_upper, ci_label = compute_ci_bounds(
        llm_distributions, llm_ndcg_values, CI_TYPE
    )

    # For Human GT, estimate bounds by scaling LLM bounds by ratio of means
    human_lower = []
    human_upper = []
    for i, human_mean in enumerate(human_ndcg_values):
        llm_mean = llm_ndcg_values[i]
        if llm_mean > 0:
            ratio = human_mean / llm_mean
            # Scale the distance from mean proportionally
            human_lower.append(human_mean - (llm_mean - llm_lower[i]) * ratio)
            human_upper.append(human_mean + (llm_upper[i] - llm_mean) * ratio)
        else:
            human_lower.append(human_mean)
            human_upper.append(human_mean)

    plot_sort_k_ndcg_comparison(
        SORT_WINDOW_VALUES,
        human_ndcg_values,
        llm_ndcg_values,
        output_dir,
        human_bounds=(human_lower, human_upper),
        llm_bounds=(llm_lower, llm_upper),
        ci_label=ci_label,
    )


def generate_hgt_comparison_plots(output_dir: Path) -> None:
    """Generate HGT-based comparison plots for recall and NDCG.

    Uses pre-computed metrics from HGT (Hierarchical Ground Truth) evaluation.
    Uses canonical colors and ordering from config.py.

    For LTFilter methods:
    - S<=10: Uses pairwise quicksort (hgt_pair) data that replaces LMPQsort entirely
    - S>10: Uses pairwise quicksort on all L items (from hgt_pair)
    """
    print("Generating HGT comparison plots...")

    # Build method results for each metric using canonical config
    def build_methods(metric_column: str) -> tuple[list[MethodResult], list[str]]:
        # Temporary storage for methods before sorting
        method_data: dict[str, tuple[MethodResult, str]] = {}

        # Helper to build a MethodResult from a method key and metrics path
        def make_method(key: str, metrics_path: Path, ms: float = MARKER_SIZE_NORMAL) -> tuple[MethodResult, str]:
            df = load_metrics(metrics_path)
            return MethodResult(
                label=METHOD_LABELS[key],
                color=METHOD_COLORS[key],
                time=df["time"].iloc[0],
                metric_value=df[metric_column].iloc[0],
                marker_size=ms,
                marker=METHOD_MARKERS[key],
            ), str(metrics_path)

        method_data["tournament"] = make_method("tournament", DATA_PATHS["tourk5000"] / "bier_metrics_tour_10_25.csv")
        method_data["lmpqselect"] = make_method("lmpqselect", HGT_DATA_PATHS["base"] / "bier_metrics_sorted_16_2_20_2_10_25.csv")
        method_data["lmpqselect_zephyr7b"] = make_method("lmpqselect_zephyr7b", DLLM_PATHS["zephyr7b"] / "bier_metrics_sorted_16_2_20_2_10_25.csv")
        method_data["lmpqselect_qwen3_8b"] = make_method("lmpqselect_qwen3_8b", DLLM_PATHS["qwen3_8b"] / "bier_metrics_sorted_16_2_20_2_10_25.csv")
        method_data["pairwise"] = make_method("pairwise", HGT_DATA_PATHS["pairwise"] / "bier_metrics_sorted_1_1_2_2_10_25.csv")
        method_data["lotus_qwen"] = make_method("lotus_qwen", LOTUS_PATHS["qwen3_8b"] / "bier_metrics_lotus_10_25.csv")
        method_data["lotus_zephyr"] = make_method("lotus_zephyr", LOTUS_PATHS["zephyr_7b"] / "bier_metrics_lotus_10_25.csv")
        method_data["pointwise_qwen"] = make_method("pointwise_qwen", POINTWISE_PATHS["qwen3_8b"] / "bier_metrics_lotus_10_25.csv")
        method_data["pointwise_zephyr"] = make_method("pointwise_zephyr", POINTWISE_PATHS["zephyr"] / "bier_metrics_lotus_10_25.csv")

        # Build ordered list following METHOD_ORDER
        methods = []
        data_sources = []

        for method_key in METHOD_ORDER:
            if method_key == "ltfilter":
                s_configs = [
                    (1, HGT_DATA_PATHS["l1"] / "bier_metrics_sorted_16_2_1_2_10_25.csv"),
                    (2, HGT_DATA_PATHS["l2"] / "bier_metrics_sorted_16_2_2_2_10_25.csv"),
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
                    data_sources.append(str(metrics_path))
            elif method_key in method_data:
                method, source = method_data[method_key]
                methods.append(method)
                data_sources.append(source)

        return methods, data_sources

    # Generate Recall@10 plot
    recall_methods, data_sources = build_methods("Recall@10")
    recall_metadata = PlotMetadata(
        title="HGT Method Comparison: Recall@10 vs Latency",
        xlabel="Latency (s)",
        ylabel="Recall@10",
        description="Compares different algorithm configurations on Recall@10 using pre-computed HGT metrics.",
        data_sources=data_sources,
        algorithm="Multi-pivot Quickselect variants + baselines",
        parameters={"k": 10, "ground_truth": "HGT pre-computed metrics", "num_queries": 25},
    )
    plot_method_comparison(
        recall_methods,
        "Recall@10",
        "main_human_recall@10",
        output_dir,
        metadata=recall_metadata,
        show_pareto=True,
    )

    # Generate NDCG@10 plot
    ndcg_methods, data_sources = build_methods("NDCG@10")
    ndcg_metadata = PlotMetadata(
        title="HGT Method Comparison: NDCG@10 vs Latency",
        xlabel="Latency (s)",
        ylabel="NDCG@10",
        description="Compares different algorithm configurations on NDCG@10 using pre-computed HGT metrics.",
        data_sources=data_sources,
        algorithm="Multi-pivot Quickselect variants + baselines",
        parameters={"k": 10, "ground_truth": "HGT pre-computed metrics", "num_queries": 25},
    )
    plot_method_comparison(
        ndcg_methods,
        "NDCG@10",
        "main_human_ndcg@10",
        output_dir,
        metadata=ndcg_metadata,
        show_pareto=True,
    )


def generate_llm_gt_comparison_plots(output_dir: Path, k_rrf: float | None = None) -> None:
    """Generate comparison plots using LLM-as-a-judge ground truth.

    Computes Recall@10 and NDCG@10 from raw sorted results against
    the LLM ground truth rankings. Uses canonical colors and ordering from config.py.

    For LTFilter methods:
    - S<=10: Uses pairwise quicksort (hgt_pair) data that replaces LMPQsort entirely
    - S>10: Uses pairwise quicksort on all L items (from hgt_pair)
    """
    print("Generating LLM ground truth comparison plots...")

    # Helper function to compute metrics for a single method
    def compute_method_metrics(path: Path) -> tuple[float, float, float]:
        """Returns (mean_time, mean_recall, mean_ndcg)."""
        results = load_results(path)
        recalls = compute_recalls_for_results(results, GROUND_TRUTH_PATH, 10)
        ndcgs = compute_ndcg_for_results(results, GROUND_TRUTH_PATH, 10, k_rrf=k_rrf)
        mean_time = np.mean(results[0].times)
        return mean_time, np.mean(recalls), np.mean(ndcgs)

    # Helper to load pre-computed baseline metrics
    def load_baseline_metrics(agg_path: Path) -> tuple[float, float, float] | None:
        """Load pre-computed metrics. Returns (time, recall, ndcg)."""
        if not agg_path.exists():
            return None
        df_agg = pd.read_csv(agg_path)
        return df_agg["time"].iloc[0], df_agg["Recall@10"].iloc[0], df_agg["NDCG@10"].iloc[0]

    # Temporary storage for method data before ordering
    # key -> (time, recall, ndcg, source)
    method_data: dict[str, tuple[float, float, float, str]] = {}

    # Tournament Top-K (from pre-computed metrics in canonical location)
    tournament_metrics = load_baseline_metrics(LLM_GT_METRICS_PATH / "tournament_metrics_llm_agg_10_25.csv")
    if tournament_metrics:
        method_data["tournament"] = (*tournament_metrics, "tourk5000/bier_result_unsorted_tour_10_25.csv")

    # LMPQSelect/Sort (L=20) - using pairwise quicksort (W=2) for final sort - RankZephyr
    path = HGT_DATA_PATHS["base"] / "bier_sorted_10_16_2_20_2.csv"
    if path.exists():
        metrics = compute_method_metrics(path)
        method_data["lmpqselect"] = (*metrics, str(path))

    # LMPQSelect/Sort (L=20) - Zephyr7b
    path = DLLM_PATHS["zephyr7b"] / "bier_sorted_10_16_2_20_2.csv"
    if path.exists():
        metrics = compute_method_metrics(path)
        method_data["lmpqselect_zephyr7b"] = (*metrics, str(path))

    # LMPQSelect/Sort (L=20) - Qwen3-8b
    path = DLLM_PATHS["qwen3_8b"] / "bier_sorted_10_16_2_20_2.csv"
    if path.exists():
        metrics = compute_method_metrics(path)
        method_data["lmpqselect_qwen3_8b"] = (*metrics, str(path))

    # Pairwise Quickselect/Sort
    path = HGT_DATA_PATHS["pairwise"] / "bier_sorted_10_1_1_2_2.csv"
    if path.exists():
        metrics = compute_method_metrics(path)
        method_data["pairwise"] = (*metrics, str(path))

    # LOTUS (QWEN3-8B) from pre-computed metrics in canonical location
    lotus_qwen_metrics = load_baseline_metrics(LLM_GT_METRICS_PATH / "lotus_qwen_metrics_llm_agg_10_20.csv")
    if lotus_qwen_metrics:
        method_data["lotus_qwen"] = (*lotus_qwen_metrics, "lotusk/qwen_data/bier_lotus_semtopk_result.csv")

    # LOTUS (Zephyr-7B) from pre-computed metrics in canonical location
    lotus_zephyr_metrics = load_baseline_metrics(LLM_GT_METRICS_PATH / "lotus_zephyr_metrics_llm_agg_10_24.csv")
    if lotus_zephyr_metrics:
        method_data["lotus_zephyr"] = (*lotus_zephyr_metrics, "lotusk/combined_z7b/bier_lotus_semtopk_result.csv")

    # Pointwise (QWEN3-8B) from pre-computed metrics
    pw_qwen_metrics = load_baseline_metrics(LLM_GT_METRICS_PATH / "pointwise_qwen_metrics_llm_agg_10_25.csv")
    if pw_qwen_metrics:
        method_data["pointwise_qwen"] = (*pw_qwen_metrics, "bier_pointwise/qwen/bier_lotus_map_result_q.csv")

    # Pointwise (Zephyr-7B) from pre-computed metrics
    pw_zephyr_metrics = load_baseline_metrics(LLM_GT_METRICS_PATH / "pointwise_zephyr_metrics_llm_agg_10_25.csv")
    if pw_zephyr_metrics:
        method_data["pointwise_zephyr"] = (*pw_zephyr_metrics, "bier_pointwise/zephyr/bier_lotus_map_result.csv")

    # Build ordered method lists following METHOD_ORDER
    recall_methods = []
    ndcg_methods = []
    data_sources = []

    for method_key in METHOD_ORDER:
        if method_key == "ltfilter":
            # LTFilter+LMPQ (S=1,2,5,10,15) - use pairwise quicksort for final sort
            s_configs = [
                (1, HGT_DATA_PATHS["l1"] / "bier_sorted_10_16_2_1_2.csv"),
                (2, HGT_DATA_PATHS["l2"] / "bier_sorted_10_16_2_2_2.csv"),
                (5, HGT_DATA_PATHS["l5"] / "bier_sorted_10_16_2_5_2.csv"),
                (10, HGT_DATA_PATHS["l10"] / "bier_sorted_10_16_2_10_2.csv"),
                (15, HGT_DATA_PATHS["l15"] / "bier_sorted_10_16_2_15_2.csv"),
            ]
            for s_val, path in s_configs:
                if not path.exists():
                    print(f"  Skipping S={s_val}: file not found at {path}")
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
                data_sources.append(str(path))
        elif method_key in method_data:
            mean_time, mean_recall, mean_ndcg, source = method_data[method_key]
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
            data_sources.append(source)

    # Generate Recall@10 plot
    recall_metadata = PlotMetadata(
        title="Method Comparison: Recall@10 vs Latency",
        xlabel="Latency (s)",
        ylabel="Recall@10",
        description="Compares different algorithm configurations on Recall@10 computed against LLM-as-a-judge ground truth rankings.",
        data_sources=data_sources,
        algorithm="Multi-pivot Quickselect variants",
        parameters={
            "k": 10,
            "ground_truth": "llm-topk-gt/phase7_combined_rankings",
            "num_queries": 25,
        },
    )
    plot_method_comparison(
        recall_methods,
        "Recall@10",
        "main_llm_recall@10",
        output_dir,
        metadata=recall_metadata,
        show_pareto=True,
    )

    # Generate NDCG@10 plot
    ndcg_metadata = PlotMetadata(
        title="Method Comparison: NDCG@10 vs Latency",
        xlabel="Latency (s)",
        ylabel="NDCG@10",
        description="Compares different algorithm configurations on NDCG@10 computed against LLM-as-a-judge ground truth rankings.",
        data_sources=data_sources,
        algorithm="Multi-pivot Quickselect variants",
        parameters={
            "k": 10,
            "ground_truth": "llm-topk-gt/phase7_combined_rankings",
            "num_queries": 25,
        },
    )
    plot_method_comparison(
        ndcg_methods,
        "NDCG@10",
        "main_llm_ndcg@10",
        output_dir,
        metadata=ndcg_metadata,
        show_pareto=True,
    )

    # Return methods for combined plot
    return recall_methods, ndcg_methods


def generate_window_size_plot(output_dir: Path) -> None:
    """Generate window size analysis plots with Human GT and LLM GT."""
    print("Generating window size plots...")

    import ast

    def parse_hgt_per_query_recalls(metrics_file: Path) -> list[float]:
        """Parse per-query HGT metrics file and extract final recall values."""
        df = pd.read_csv(metrics_file)
        recalls = []
        for _, row in df.iterrows():
            recall_list = ast.literal_eval(row["Recall@10"])
            if recall_list:
                recalls.append(recall_list[-1])
        return recalls

    # L values that have HGT metrics (excluding L=32 which doesn't have them)
    l_values = [2, 4, 8, 16, 48, 64, 128]

    human_recall_means = []
    llm_recall_means = []
    latency_means = []

    for l_val in l_values:
        if l_val == 2:
            # L=2 uses comparison data
            result_file = DATA_PATHS["comparison"] / "bier_result_unsorted_1_1_25.csv"
            hgt_metrics_file = DATA_PATHS["comparison"] / "bier_i_metrics_1_1_25.csv"
        else:
            result_file = DATA_PATHS["window_sweep"] / f"bier_result_unsorted_E_{l_val}_25_A.csv"
            hgt_metrics_file = DATA_PATHS["window_sweep"] / f"bier_i_metrics_E_{l_val}_25_A.csv"

        # LLM GT: compute from results
        results = load_results(result_file)
        llm_recalls = compute_recalls_for_results(results, GROUND_TRUTH_PATH, 10)
        llm_recall_means.append(np.mean(llm_recalls))

        # Human GT: from HGT metrics
        human_recalls = parse_hgt_per_query_recalls(hgt_metrics_file)
        human_recall_means.append(np.mean(human_recalls))

        # Latency: mean of all times
        all_times = []
        for r in results:
            all_times.extend(r.times)
        latency_means.append(np.mean(all_times))

    plot_window_size_analysis(l_values, human_recall_means, llm_recall_means, latency_means, output_dir)


def generate_wsort_plot(output_dir: Path, k_rrf: float | None = None) -> None:
    """Generate sort window comparison plot with Human GT and LLM GT."""
    print("Generating wsort plot...")

    l_values = []
    human_ndcg_values = []
    llm_ndcg_values = []
    llm_distributions = []

    for w_val in WSORT_VALUES:
        l_values.append(w_val)

        # W=20 data is in base experiments, others in wsort
        if w_val == 20:
            result_file = HGT_DATA_PATHS["base"] / "bier_sorted_10_16_2_20_20.csv"
            metrics_file = HGT_DATA_PATHS["base"] / "bier_metrics_sorted_16_2_20_20_10_25.csv"
        else:
            result_file = DATA_PATHS["wsort"] / f"bier_sorted_10_16_2_20_{w_val}.csv"
            metrics_file = DATA_PATHS["wsort"] / f"bier_metrics_sorted_16_2_20_{w_val}_10_25.csv"

        # LLM GT: compute per-query NDCG from sorted results
        results = load_results(result_file)
        llm_ndcgs = compute_ndcg_for_results(results, GROUND_TRUTH_PATH, 10, k_rrf=k_rrf)
        llm_ndcg_values.append(np.mean(llm_ndcgs))
        llm_distributions.append(llm_ndcgs)

        # Human GT: use pre-computed HGT metrics (aggregate only)
        df = load_metrics(metrics_file)
        human_ndcg_values.append(df["NDCG@10"].iloc[0])

    # Compute CI bounds for LLM (we have per-query data)
    llm_lower, llm_upper, ci_label = compute_ci_bounds(
        llm_distributions, llm_ndcg_values, CI_TYPE
    )

    # For Human GT, estimate bounds by scaling LLM bounds by ratio of means
    human_lower = []
    human_upper = []
    for i, human_mean in enumerate(human_ndcg_values):
        llm_mean = llm_ndcg_values[i]
        if llm_mean > 0:
            ratio = human_mean / llm_mean
            # Scale the distance from mean proportionally
            human_lower.append(human_mean - (llm_mean - llm_lower[i]) * ratio)
            human_upper.append(human_mean + (llm_upper[i] - llm_mean) * ratio)
        else:
            human_lower.append(human_mean)
            human_upper.append(human_mean)

    plot_wsort_comparison(
        l_values,
        human_ndcg_values,
        llm_ndcg_values,
        output_dir,
        human_bounds=(human_lower, human_upper),
        llm_bounds=(llm_lower, llm_upper),
        ci_label=ci_label,
    )


def generate_sembench_main_plot(output_dir: Path) -> None:
    """Generate main comparison plots for SemBench dataset.

    Computes NDCG@10 against human ground truth (SemBench Q9) for multiple
    ranking methods. Color encodes method family, marker shape encodes LLM
    backend. Tournament Top-K and LMPQSort shown w/ top-few refinement.
    """
    print("Generating SemBench main plots...")

    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from .plots.base import save_figure, close_figure, LEGEND_FONTSIZE
    from .config import LLM_BACKEND_MARKERS

    sembench_dir = DATA_RAW_DIR / "sembench_data"

    # Human ground truth path (SemBench Q9: review scoring task)
    gt_path = Path("/home/jwc/semop-exprs/data/SemBench/files/movie/raw_results/ground_truth/Q9.csv")

    if not gt_path.exists():
        print(f"  Warning: Ground truth not found at {gt_path}, skipping plots")
        return

    # Load ground truth and build relevance dict
    gt_df = pd.read_csv(gt_path).drop_duplicates(subset=['reviewId'], keep='first')
    gt_dict = dict(zip(gt_df['reviewId'].astype(int), gt_df['reviewScore']))
    print(f"  Loaded ground truth: {len(gt_dict)} unique reviews")

    # Sort GT by score descending to establish ground truth ranking
    gt_sorted = sorted(gt_dict.items(), key=lambda x: x[1], reverse=True)
    gt_ranked_ids = [rid for rid, _ in gt_sorted]

    # NDCG computation helpers
    def dcg_at_k(scores, k):
        scores = np.array(scores[:k])
        discounts = np.log2(np.arange(2, len(scores) + 2))
        return np.sum(scores / discounts)

    def ndcg_at_k(predicted_ranking, gt_dict, k=10):
        rel_scores = [gt_dict.get(rid, 0) for rid in predicted_ranking]
        dcg = dcg_at_k(rel_scores, k)
        ideal_scores = sorted(gt_dict.values(), reverse=True)
        idcg = dcg_at_k(ideal_scores, k)
        return dcg / idcg if idcg > 0 else 0

    def load_ranking(filepath):
        df = pd.read_csv(filepath)
        df['reviewId'] = df['reviewId'].astype(int)
        seen = set()
        ranking = []
        for rid in df['reviewId']:
            if rid not in seen:
                ranking.append(rid)
                seen.add(rid)
        return ranking

    def recall_at_k_fn(predicted_ranking, k=10):
        """Recall@K: fraction of GT top-K docs found in predicted top-K."""
        predicted_top_k = set(predicted_ranking[:k])
        gt_top_k = set(gt_ranked_ids[:k])
        return len(predicted_top_k & gt_top_k) / k

    def spearman_at_k_fn(predicted_ranking, k=50):
        """Spearman@K: rank correlation for GT top-K items."""
        from scipy.stats import spearmanr
        gt_top_k = gt_ranked_ids[:k]
        pred_rank_map = {rid: rank for rank, rid in enumerate(predicted_ranking, 1)}
        gt_ranks = list(range(1, k + 1))
        pred_ranks = [pred_rank_map.get(rid, len(predicted_ranking) + 1)
                      for rid in gt_top_k]
        corr, _ = spearmanr(gt_ranks, pred_ranks)
        return corr

    def pearson_at_k_fn(predicted_ranking, k=50):
        """Pearson@K: linear correlation for GT top-K items."""
        from scipy.stats import pearsonr
        gt_top_k = gt_ranked_ids[:k]
        pred_rank_map = {rid: rank for rank, rid in enumerate(predicted_ranking, 1)}
        gt_ranks = list(range(1, k + 1))
        pred_ranks = [pred_rank_map.get(rid, len(predicted_ranking) + 1)
                      for rid in gt_top_k]
        corr, _ = pearsonr(gt_ranks, pred_ranks)
        return corr

    def ndcg_rrf_at_k_fn(predicted_ranking, k, k_rrf=60.0):
        """NDCG@K with RRF-graded relevance: rel(r) = 1/(k_rrf + r)."""
        gt_top_k = gt_ranked_ids[:k]
        rel_map = {rid: 1.0 / (k_rrf + rank)
                   for rank, rid in enumerate(gt_top_k, 1)}
        dcg = 0.0
        for i, rid in enumerate(predicted_ranking[:k], 1):
            rel = rel_map.get(rid, 0.0)
            if rel > 0:
                dcg += rel / np.log2(1 + i)
        ideal_scores = sorted(rel_map.values(), reverse=True)
        idcg = sum(s / np.log2(1 + i)
                   for i, s in enumerate(ideal_scores, 1))
        return dcg / idcg if idcg > 0 else 0.0

    # Colors from config (Paul Tol vibrant), shapes by LLM backend
    # Only refined variants kept for Tournament Top-K and LMPQSort
    # Order determines legend order (2 columns, left-to-right then top-to-bottom)
    METHODS = [
        # Row 1: LTTopK, LMPQ (bolded)
        {"family": "Tournament Top-K", "label": "LTTopK",
         "color": "#0077BB", "marker": "o",
         "ranking_pattern": "recomputed/tour/{}_recomputed.csv",
         "time_pattern": "recomputed/tour/{}_score_recomputed.csv",
         "runs": range(1, 11)},
        {"family": "LMPQSort", "label": "LMPQ",
         "color": "#33BBEE", "marker": "o",
         "ranking_pattern": "recomputed/rz20/{}_recomputed.csv",
         "time_pattern": "recomputed/rz20/{}_score_recomputed.csv",
         "runs": range(1, 11)},
        # Row 2: Pairwise (RankZephyr) (unbolded)
        {"family": "Pairwise", "label": "Pairwise (RankZephyr)",
         "color": "#009988", "marker": "o",
         "ranking_pattern": "sort/rz2_{}.csv",
         "time_pattern": "sort/q9_2_{}.csv",
         "runs": [1, 3, 4, 5, 6, 7, 8, 9, 10]},
        # Row 3: Pointwise variants (unbolded)
        {"family": "Pointwise", "label": "Pointwise (QWEN)",
         "color": "#CC3311", "marker": "*",
         "ranking_pattern": "pointwise/qw_{}.csv",
         "time_pattern": "pointwise/qw_r_{}.csv",
         "runs": range(1, 11)},
        {"family": "Pointwise", "label": "Pointwise (Zephyr)",
         "color": "#CC3311", "marker": "s",
         "ranking_pattern": "pointwise/z_{}.csv",
         "time_pattern": "pointwise/z_r_{}.csv",
         "runs": range(1, 11)},
    ]

    # Pre-load all rankings and times
    from .plots.scatter import draw_pareto_frontier

    method_data = []
    for method in METHODS:
        rankings = []
        time_vals = []
        for run in method["runs"]:
            rank_file = sembench_dir / method["ranking_pattern"].format(run)
            if rank_file.exists():
                rankings.append(load_ranking(rank_file))
            time_file = sembench_dir / method["time_pattern"].format(run)
            if time_file.exists():
                time_df = pd.read_csv(time_file)
                time_vals.append(time_df['time'].iloc[0])
        method_data.append((method, rankings, time_vals))

    if not any(rankings for _, rankings, _ in method_data):
        print("  Warning: No SemBench data found, skipping plots")
        return

    # Load w/o refinement data for rerank impact plots
    lmpq_base_rankings = []
    lmpq_base_times = []
    tour_base_rankings = []
    tour_base_times = []
    for run in range(1, 11):
        rank_file = sembench_dir / f"sort/rz20_{run}.csv"
        if rank_file.exists():
            lmpq_base_rankings.append(load_ranking(rank_file))
        time_file = sembench_dir / f"sort/q9_20_{run}.csv"
        if time_file.exists():
            time_df = pd.read_csv(time_file)
            lmpq_base_times.append(time_df['time'].iloc[0])
        rank_file = sembench_dir / f"tour/rzt20_{run}.csv"
        if rank_file.exists():
            tour_base_rankings.append(load_ranking(rank_file))
        time_file = sembench_dir / f"tour/q9t_20_{run}.csv"
        if time_file.exists():
            time_df = pd.read_csv(time_file)
            tour_base_times.append(time_df['time'].iloc[0])

    # Define all metric variants: (file_suffix, y_label, metric_fn)
    METRIC_VARIANTS = [
        ("ndcg@10", "NDCG@10",
         lambda r: ndcg_at_k(r, gt_dict, k=10)),
        ("recall@10", "Recall@10",
         lambda r: recall_at_k_fn(r, k=10)),
        ("spearman@128", "Spearman's Correlation",
         lambda r: spearman_at_k_fn(r, k=128)),
        ("pearson@50", "Pearson's r",
         lambda r: pearson_at_k_fn(r, k=50)),
        ("ndcg@10_rrf", "NDCG@10",
         lambda r: ndcg_rrf_at_k_fn(r, k=10, k_rrf=60.0)),
        ("ndcg@50_rrf", "NDCG@50",
         lambda r: ndcg_rrf_at_k_fn(r, k=50, k_rrf=60.0)),
    ]

    FAMILY_ORDER = ["Tournament Top-K", "LMPQSort", "Pairwise", "Pointwise"]

    for metric_key, metric_label, metric_fn in METRIC_VARIANTS:
        plot_points = []
        for method, rankings, time_vals in method_data:
            metric_vals = [metric_fn(r) for r in rankings]
            if metric_vals:
                mean_val = np.mean(metric_vals)
                std_val = np.std(metric_vals)
                print(f"  {method['family']} [{metric_key}]: "
                      f"{mean_val:.4f} ± {std_val:.4f}")
                plot_points.append({
                    "x": np.mean(time_vals) if time_vals else 100,
                    "y": mean_val,
                    "color": method["color"],
                    "marker": method["marker"],
                    "family": method["family"],
                    "label": method["label"],
                })

        if not plot_points:
            continue

        # Plot configuration — canonical plots use publication sizing
        is_canonical = metric_key in ("ndcg@50_rrf", "pearson@50", "spearman@128")
        if is_canonical:
            plt.rcParams['font.family'] = font_name
            fig_w, fig_h = 3.3335, 2.7  # Slightly taller for thicker legend
            fs_label, fs_legend, fs_tick = 9, 6, 6
            ms_star, ms_other = 10, 10
            lms_star, lms_other = 8, 5
            lw_pareto = 1.0
        else:
            fig_w, fig_h = 8, 6
            fs_label = fs_legend = LEGEND_FONTSIZE
            fs_tick = None
            ms_star, ms_other = 80, 40
            lms_star, lms_other = 12, 8
            lw_pareto = 1.5

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        if is_canonical:
            fig.subplots_adjust(left=0.13, right=0.98, top=0.98, bottom=0.15)

        for pt in plot_points:
            marker_size = ms_star if pt["marker"] == "*" else ms_other
            ax.scatter(
                pt["x"], pt["y"],
                c=pt["color"], s=marker_size, marker=pt["marker"], zorder=3,
            )

        x_values = [pt["x"] for pt in plot_points]
        y_values = [pt["y"] for pt in plot_points]
        x_pad = (max(x_values) - min(x_values)) * 0.1
        ax.set_xlim(0, max(x_values) + x_pad)

        if "pearson" in metric_key:
            ax.set_ylim(0, 0.55)
        elif "spearman" in metric_key:
            ax.set_ylim(0, 0.75)
        else:
            ax.set_ylim(0, 1)

        ax.set_xlabel("Latency (s)", fontsize=fs_label)
        ax.set_ylabel(metric_label, fontsize=fs_label)
        if fs_tick is not None:
            ax.tick_params(axis='both', labelsize=fs_tick)

        # Legend — one entry per method dot
        BOLD_FAMILIES = {"Tournament Top-K", "LMPQSort"}
        legend_handles = []
        for pt in plot_points:
            pt_ms = lms_star if pt["marker"] == "*" else lms_other
            legend_handles.append(mlines.Line2D(
                [0], [0], marker=pt["marker"], color="w",
                markerfacecolor=pt["color"],
                markersize=pt_ms, label=pt["label"],
                linestyle="None",
            ))
        if is_canonical:
            leg = ax.legend(handles=legend_handles, fontsize=fs_legend,
                            loc="lower right", ncol=2,
                            borderaxespad=0.3, handletextpad=0.3,
                            borderpad=0.3, columnspacing=0.8)
            for i, text in enumerate(leg.get_texts()):
                if plot_points[i]["family"] in BOLD_FAMILIES:
                    text.set_fontweight("bold")
        else:
            ax.legend(handles=legend_handles, fontsize=fs_legend, loc="best", ncol=2)

        draw_pareto_frontier(ax, x_values, y_values, linewidth=lw_pareto)

        if not is_canonical:
            plt.tight_layout()
        save_figure(f"sembench_human_{metric_key}", output_dir,
                    bbox_inches=None if is_canonical else "tight")
        close_figure()

        # --- Rerank impact variant (w/o → w/ refinement arrows) ---
        # Collect base data for each method that has refinement data
        rerank_methods = []
        if lmpq_base_rankings:
            lmpq_ref_pt = next(
                (pt for pt in plot_points if pt["family"] == "LMPQSort"), None)
            if lmpq_ref_pt:
                lmpq_base_mv = [metric_fn(r) for r in lmpq_base_rankings]
                lmpq_bx = np.mean(lmpq_base_times) if lmpq_base_times else 100
                lmpq_by = np.mean(lmpq_base_mv)
                rerank_methods.append(("LMPQSort", "#33BBEE",
                                       lmpq_bx, lmpq_by, lmpq_ref_pt))
        if tour_base_rankings:
            tour_ref_pt = next(
                (pt for pt in plot_points
                 if pt["family"] == "Tournament Top-K"), None)
            if tour_ref_pt:
                tour_base_mv = [metric_fn(r) for r in tour_base_rankings]
                tour_bx = np.mean(tour_base_times) if tour_base_times else 100
                tour_by = np.mean(tour_base_mv)
                rerank_methods.append(("Tournament Top-K", "#0077BB",
                                       tour_bx, tour_by, tour_ref_pt))

        if not rerank_methods:
            continue

        for name, color, bx, by, ref_pt in rerank_methods:
            print(f"  {name} rerank [{metric_key}]: "
                  f"{by:.4f} → {ref_pt['y']:.4f} "
                  f"(+{ref_pt['y'] - by:.4f})")

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot all methods (same as main plot)
        for pt in plot_points:
            ms = 80 if pt["marker"] == "*" else 40
            ax.scatter(
                pt["x"], pt["y"],
                c=pt["color"], s=ms, marker=pt["marker"], zorder=3,
            )

        extra_x, extra_y = [], []
        for name, color, bx, by, ref_pt in rerank_methods:
            # Hollow marker for w/o refinement
            ax.scatter(
                bx, by, c="none", s=40, marker="o", zorder=3,
                edgecolors=color, linewidths=1.5,
            )
            # Arrow from w/o → w/ refinement
            ax.annotate(
                "", xy=(ref_pt["x"], ref_pt["y"]),
                xytext=(bx, by),
                arrowprops=dict(arrowstyle="->", color=color, lw=2),
                zorder=2,
            )
            extra_x.append(bx)
            extra_y.append(by)

        all_x = x_values + extra_x
        all_y = y_values + extra_y
        x_pad_r = (max(all_x) - min(all_x)) * 0.1
        ax.set_xlim(0, max(all_x) + x_pad_r)

        if "pearson" in metric_key:
            ax.set_ylim(0, 0.55)
        elif "spearman" in metric_key:
            ax.set_ylim(0, 0.75)
        else:
            ax.set_ylim(0, 1)

        ax.set_xlabel("Latency (s)", fontsize=LEGEND_FONTSIZE)
        ax.set_ylabel(metric_label, fontsize=LEGEND_FONTSIZE)

        # Legend: one entry per method dot + refinement markers
        legend_handles_r = []
        for pt in plot_points:
            pt_ms = 12 if pt["marker"] == "*" else 8
            legend_handles_r.append(mlines.Line2D(
                [0], [0], marker=pt["marker"], color="w",
                markerfacecolor=pt["color"],
                markersize=pt_ms, label=pt["label"],
                linestyle="None",
            ))
        legend_handles_r.append(mlines.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor="none", markeredgecolor="gray",
            markeredgewidth=1.5, markersize=8,
            label="w/o refinement", linestyle="None",
        ))
        legend_handles_r.append(mlines.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor="gray",
            markersize=8, label="w/ refinement", linestyle="None",
        ))

        ax.legend(
            handles=legend_handles_r, fontsize=LEGEND_FONTSIZE, loc="best")

        draw_pareto_frontier(ax, x_values, y_values)

        plt.tight_layout()
        save_figure(f"sembench_human_{metric_key}_rerank", output_dir)
        close_figure()

    print("  SemBench plots generated successfully!")


def generate_sembench_rerank_impact_plot(output_dir: Path) -> None:
    """Generate a plot showing the impact of top-few refinement on SemBench.

    Analogous to the pairwise_rerank_impact plot for scifact: shows arrows
    from w/o to w/ refinement for Tournament Top-K and LMPQSort, evaluated
    against SemBench Q9 human ground truth.
    """
    print("Generating SemBench rerank impact plot...")

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from .plots.base import save_figure, close_figure

    sembench_dir = DATA_RAW_DIR / "sembench_data"

    # Human ground truth (SemBench Q9)
    gt_path = Path("/home/jwc/semop-exprs/data/SemBench/files/movie/raw_results/ground_truth/Q9.csv")
    if not gt_path.exists():
        print(f"  Warning: Ground truth not found at {gt_path}, skipping")
        return

    gt_df = pd.read_csv(gt_path).drop_duplicates(subset=['reviewId'], keep='first')
    gt_dict = dict(zip(gt_df['reviewId'].astype(int), gt_df['reviewScore']))

    # NDCG helpers
    def dcg_at_k(scores, k):
        scores = np.array(scores[:k])
        return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))

    def ndcg_at_k(predicted_ranking, gt_dict, k=10):
        rel_scores = [gt_dict.get(rid, 0) for rid in predicted_ranking]
        dcg = dcg_at_k(rel_scores, k)
        ideal_scores = sorted(gt_dict.values(), reverse=True)
        idcg = dcg_at_k(ideal_scores, k)
        return dcg / idcg if idcg > 0 else 0

    def load_ranking(filepath):
        df = pd.read_csv(filepath)
        df['reviewId'] = df['reviewId'].astype(int)
        seen = set()
        ranking = []
        for rid in df['reviewId']:
            if rid not in seen:
                ranking.append(rid)
                seen.add(rid)
        return ranking

    def compute_metrics(ranking_pattern, time_pattern, runs):
        ndcg_vals, time_vals = [], []
        for run in runs:
            rank_file = sembench_dir / ranking_pattern.format(run)
            if rank_file.exists():
                ranking = load_ranking(rank_file)
                ndcg_vals.append(ndcg_at_k(ranking, gt_dict, k=10))
            time_file = sembench_dir / time_pattern.format(run)
            if time_file.exists():
                time_df = pd.read_csv(time_file)
                time_vals.append(time_df['time'].iloc[0])
        return np.mean(ndcg_vals), np.mean(time_vals)

    runs = range(1, 11)

    # Tournament Top-K: w/o and w/ refinement
    tour_base_ndcg, tour_base_time = compute_metrics(
        "tour/rzt20_{}.csv", "tour/q9t_20_{}.csv", runs)
    tour_ref_ndcg, tour_ref_time = compute_metrics(
        "recomputed/tour/{}_recomputed.csv",
        "recomputed/tour/{}_score_recomputed.csv", runs)

    # LMPQSort: w/o and w/ refinement
    sort_base_ndcg, sort_base_time = compute_metrics(
        "sort/rz20_{}.csv", "sort/q9_20_{}.csv", runs)
    sort_ref_ndcg, sort_ref_time = compute_metrics(
        "recomputed/rz20/{}_recomputed.csv",
        "recomputed/rz20/{}_score_recomputed.csv", runs)

    print(f"  Tournament Top-K:  {tour_base_ndcg:.4f} → {tour_ref_ndcg:.4f} "
          f"(+{tour_ref_ndcg - tour_base_ndcg:.4f}), "
          f"{tour_base_time:.1f}s → {tour_ref_time:.1f}s")
    print(f"  LMPQSort:          {sort_base_ndcg:.4f} → {sort_ref_ndcg:.4f} "
          f"(+{sort_ref_ndcg - sort_base_ndcg:.4f}), "
          f"{sort_base_time:.1f}s → {sort_ref_time:.1f}s")

    # Colors matching config
    color_tour = "#0077BB"   # Blue — Tournament
    color_sort = "#33BBEE"   # Cyan — LMPQSort

    marker_size = 100
    marker_old = "o"   # circle: without refinement
    marker_new = "s"   # square: with refinement

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot points
    ax.scatter(tour_base_time, tour_base_ndcg, s=marker_size, c=color_tour,
               marker=marker_old, zorder=3, edgecolors="white", linewidths=1)
    ax.scatter(tour_ref_time, tour_ref_ndcg, s=marker_size, c=color_tour,
               marker=marker_new, zorder=3, edgecolors="white", linewidths=1)

    ax.scatter(sort_base_time, sort_base_ndcg, s=marker_size, c=color_sort,
               marker=marker_old, zorder=3, edgecolors="white", linewidths=1)
    ax.scatter(sort_ref_time, sort_ref_ndcg, s=marker_size, c=color_sort,
               marker=marker_new, zorder=3, edgecolors="white", linewidths=1)

    # Arrows from base → refined
    ax.annotate("", xy=(tour_ref_time, tour_ref_ndcg),
                xytext=(tour_base_time, tour_base_ndcg),
                arrowprops=dict(arrowstyle="->", color=color_tour, lw=2),
                zorder=2)
    ax.annotate("", xy=(sort_ref_time, sort_ref_ndcg),
                xytext=(sort_base_time, sort_base_ndcg),
                arrowprops=dict(arrowstyle="->", color=color_sort, lw=2),
                zorder=2)

    # Axis formatting
    all_times = [tour_base_time, tour_ref_time, sort_base_time, sort_ref_time]
    x_pad = (max(all_times) - min(all_times)) * 0.8
    ax.set_xlim(min(all_times) - x_pad, max(all_times) + x_pad)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Latency (s)", fontsize=12)
    ax.set_ylabel("NDCG@10", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=color_tour, linestyle="-", linewidth=2,
               marker="o", markersize=8, label="Tournament Top-K"),
        Line2D([0], [0], color=color_sort, linestyle="-", linewidth=2,
               marker="o", markersize=8, label="LMPQSort"),
        Line2D([0], [0], color="gray", linestyle="", marker="o", markersize=8,
               markerfacecolor="gray", label="Without top-few refinement"),
        Line2D([0], [0], color="gray", linestyle="", marker="s", markersize=8,
               markerfacecolor="gray", label="With top-few refinement"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

    plt.tight_layout()
    save_figure("sembench_rerank_impact", output_dir)
    close_figure()

    print("  SemBench rerank impact plot generated!")


def generate_combined_rerank_impact_plot(output_dir: Path) -> None:
    """Generate a combined figure showing rerank impact for SciFact and SemBench.

    Two side-by-side subplots sharing the y-axis (NDCG@10) with separate
    x-axes (Latency), labeled by dataset.
    """
    print("Generating combined rerank impact plot...")

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from .plots.base import save_figure, close_figure

    # ── SciFact data (hardcoded, matching pairwise_rerank.py) ──
    scifact = {
        "human_old": (455.48, 0.46501),
        "human_new": (461.13, 0.84136),
        "llm_old": (455.48, 0.4468),
        "llm_new": (461.13, 0.5167),
    }
    color_human = "#004488"
    color_llm = "#BB5566"

    # ── SemBench data (dynamically computed) ──
    sembench_dir = DATA_RAW_DIR / "sembench_data"
    gt_path = Path("/home/jwc/semop-exprs/data/SemBench/files/movie/raw_results/ground_truth/Q9.csv")
    if not gt_path.exists():
        print(f"  Warning: Ground truth not found at {gt_path}, skipping")
        return

    gt_df = pd.read_csv(gt_path).drop_duplicates(subset=['reviewId'], keep='first')
    gt_dict = dict(zip(gt_df['reviewId'].astype(int), gt_df['reviewScore']))

    def dcg_at_k(scores, k):
        scores = np.array(scores[:k])
        return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))

    def ndcg_at_k(predicted_ranking, gt_dict_local, k=10):
        rel_scores = [gt_dict_local.get(rid, 0) for rid in predicted_ranking]
        dcg = dcg_at_k(rel_scores, k)
        ideal_scores = sorted(gt_dict_local.values(), reverse=True)
        idcg = dcg_at_k(ideal_scores, k)
        return dcg / idcg if idcg > 0 else 0

    def load_ranking(filepath):
        df = pd.read_csv(filepath)
        df['reviewId'] = df['reviewId'].astype(int)
        seen = set()
        ranking = []
        for rid in df['reviewId']:
            if rid not in seen:
                ranking.append(rid)
                seen.add(rid)
        return ranking

    def compute_metrics(ranking_pattern, time_pattern, runs):
        ndcg_vals, time_vals = [], []
        for run in runs:
            rank_file = sembench_dir / ranking_pattern.format(run)
            if rank_file.exists():
                ranking = load_ranking(rank_file)
                ndcg_vals.append(ndcg_at_k(ranking, gt_dict, k=10))
            time_file = sembench_dir / time_pattern.format(run)
            if time_file.exists():
                time_df = pd.read_csv(time_file)
                time_vals.append(time_df['time'].iloc[0])
        return np.mean(ndcg_vals), np.mean(time_vals)

    runs = range(1, 11)
    sort_base_ndcg, sort_base_time = compute_metrics(
        "sort/rz20_{}.csv", "sort/q9_20_{}.csv", runs)
    sort_ref_ndcg, sort_ref_time = compute_metrics(
        "recomputed/rz20/{}_recomputed.csv",
        "recomputed/rz20/{}_score_recomputed.csv", runs)

    color_sort = "#DDAA33"

    # ── Create combined figure (single plot, relative overhead x-axis) ──
    import matplotlib as mpl
    _rc = {
        'font.family': font_name,
        'axes.labelsize': 9,
        'legend.fontsize': 6,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
    }
    with mpl.rc_context(_rc):
        fig, ax = plt.subplots(figsize=(2.31, 1.7))
        fig.subplots_adjust(left=0.22, right=0.99, top=0.97, bottom=0.22)

        marker_size = 20
        marker_old = "o"
        marker_new = "o"

        # Compute relative overhead (100% = cost before rerank)
        sf_base_time = scifact["human_old"][0]
        sf_new_x = (scifact["human_new"][0] / sf_base_time) * 100
        sb_sort_new_x = (sort_ref_time / sort_base_time) * 100

        # SciFact, Human Labels
        ax.scatter(100, scifact["human_old"][1],
                   s=marker_size, c=color_human, marker=marker_old,
                   zorder=3, edgecolors="white", linewidths=0.5)
        ax.scatter(sf_new_x, scifact["human_new"][1],
                   s=marker_size, c=color_human, marker=marker_new,
                   zorder=3, edgecolors="white", linewidths=0.5)
        ax.annotate("", xy=(sf_new_x, scifact["human_new"][1]),
                     xytext=(100, scifact["human_old"][1]),
                     arrowprops=dict(arrowstyle="->", color=color_human, lw=1),
                     zorder=2)

        # SciFact, LLM-as-Judge Labels
        ax.scatter(100, scifact["llm_old"][1],
                   s=marker_size, c=color_llm, marker=marker_old,
                   zorder=3, edgecolors="white", linewidths=0.5)
        ax.scatter(sf_new_x, scifact["llm_new"][1],
                   s=marker_size, c=color_llm, marker=marker_new,
                   zorder=3, edgecolors="white", linewidths=0.5)
        ax.annotate("", xy=(sf_new_x, scifact["llm_new"][1]),
                     xytext=(100, scifact["llm_old"][1]),
                     arrowprops=dict(arrowstyle="->", color=color_llm, lw=1),
                     zorder=2)

        # SemBench, LMPQSort
        ax.scatter(100, sort_base_ndcg,
                   s=marker_size, c=color_sort, marker=marker_old,
                   zorder=3, edgecolors="white", linewidths=0.5)
        ax.scatter(sb_sort_new_x, sort_ref_ndcg,
                   s=marker_size, c=color_sort, marker=marker_new,
                   zorder=3, edgecolors="white", linewidths=0.5)
        ax.annotate("", xy=(sb_sort_new_x, sort_ref_ndcg),
                     xytext=(100, sort_base_ndcg),
                     arrowprops=dict(arrowstyle="->", color=color_sort, lw=1),
                     zorder=2)

        ax.set_xlim(99.6, 104.4)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Relative overhead (%)")
        ax.set_ylabel("NDCG@10")
        ax.grid(True, alpha=0.3)

        legend_handles = [
            Line2D([0], [0], color=color_human, linestyle="", marker="o",
                   markersize=4, label="SciFact, Human"),
            Line2D([0], [0], color=color_llm, linestyle="", marker="o",
                   markersize=4, label="SciFact, LLM-as-Judge"),
            Line2D([0], [0], color=color_sort, linestyle="", marker="o",
                   markersize=4, label="Movies (SemBench)"),
        ]
        ax.legend(handles=legend_handles, loc="lower right")

        save_figure("combined_rerank_impact", output_dir, bbox_inches=None)
        close_figure()

    print("  Combined rerank impact plot generated!")


def main() -> None:
    """Generate all experiment plots."""
    parser = argparse.ArgumentParser(
        description="Generate plots for ListK experiment results."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to save output plots (default: current directory)",
    )
    parser.add_argument(
        "--k-rrf",
        type=float,
        default=None,
        help="RRF constant for graded NDCG relevance (e.g. 60). Default: binary relevance.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for organized output
    main_dir = output_dir / "main"
    heuristics_dir = output_dir / "heuristics"
    select_dir = output_dir / "select"
    tfilter_dir = output_dir / "tfilter"
    window_size_dir = output_dir / "window_size"
    sort_dir = output_dir / "sort"
    sembench_dir = output_dir / "sembench"

    for d in [main_dir, heuristics_dir, select_dir, tfilter_dir, window_size_dir, sort_dir, sembench_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 50)

    # Generate plots into organized subdirectories
    generate_px_heatmaps(select_dir)
    generate_embedding_comparison(heuristics_dir)
    generate_tournament_filter_plots(tfilter_dir)
    generate_k_recall_plot(select_dir)
    generate_k_recall_comparison_plot(select_dir)
    generate_sort_plots(sort_dir, k_rrf=args.k_rrf)
    generate_window_size_plot(window_size_dir)
    generate_hgt_comparison_plots(main_dir)
    generate_llm_gt_comparison_plots(main_dir, k_rrf=args.k_rrf)
    generate_combined_main_plot(main_dir, k_rrf=args.k_rrf)
    generate_wsort_plot(sort_dir, k_rrf=args.k_rrf)
    generate_sembench_main_plot(sembench_dir)
    generate_sembench_rerank_impact_plot(sembench_dir)
    generate_combined_rerank_impact_plot(heuristics_dir)

    print("=" * 50)
    print("All plots generated successfully!")


if __name__ == "__main__":
    main()
