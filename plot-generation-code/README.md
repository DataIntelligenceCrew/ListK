# ListK Data Repository

Data and plotting scripts for evaluating multi-pivot quickselect algorithms for top-K retrieval.

> **Note:** You must clone the LLM gt repository into the main folder of this repository for the python file to work.

---

## Overview

This repository contains experimental results comparing various top-K retrieval algorithms on the SciFact dataset. The experiments evaluate trade-offs between **latency** and **ranking quality** (Recall@K, NDCG@K) across different algorithm configurations.

**Total data files:** ~492 (CSV, JSON, Parquet, JSONL)

---

## Algorithm Parameters

| Parameter | Name | Values | Description |
|-----------|------|--------|-------------|
| `p` | Pivot count | 1, 2, 4, 8, 16 | Number of pivots in multi-pivot quickselect |
| `x` | Expansion factor | 1, 2, 4, 8, 16 | Candidates considered per pivot (constraint: `p >= x`) |
| `L` | List length | 1, 2, 5, 10, 15, 20 | Candidates retained per tournament stage |
| `W` | Window size | 2, 4, 8, 16, 20, 32, 48, 64, 128 | Comparison window for sorting |
| `K` | Top-K | 10, 20, 50, 100 | Number of documents to retrieve/evaluate |

---

## File Naming Conventions

### Result Files

| Pattern | Description |
|---------|-------------|
| `bier_result_unsorted_{p}_{x}_{queries}.csv` | Raw retrieval results |
| `bier_metrics_{p}_{x}_{queries}.csv` | Aggregate metrics (single row) |
| `bier_i_metrics_{p}_{x}_{queries}.csv` | Per-query metrics |
| `bier_formatted_{p}_{x}_{queries}.json` | Rankings as `{query_id: {doc_id: rank}}` |

### Sorted/HGT Files

| Pattern | Description |
|---------|-------------|
| `bier_metrics_sorted_{p}_{x}_{L}_{W}_{K}_{queries}.csv` | Metrics after sorting stage |
| `bier_sorted_{K}_{p}_{x}_{L}_{W}.csv` | Sorted results |
| `bier_formatted_sorted_{p}_{x}_{L}_{W}_{queries}.json` | Sorted rankings as JSON |

### Special Prefixes/Suffixes

| Token | Meaning |
|-------|---------|
| `bier_` | Algorithm name prefix (all files) |
| `E` in filename | Early stopping variant |
| `_A` suffix | Variant A configuration |
| `_b`, `_e`, `_t` suffixes | Algorithm variants (B, E, T) |
| `25` | Number of queries (constant across experiments) |

---

## Directory Structure

```
listk_data/
├── 5000/                    # Base experiments (no early stopping)
├── 5000e/                   # Early stopping experiments
├── 5000w/                   # Window size sweep experiments
├── Tour_1/, Tour_2/, ...    # Tournament method with L=1,2,5,10,15
├── tour5-b/, Tour_10-b/     # Tournament variant B
├── hgt_data/                # Pre-computed HGT metric results
│   ├── l1data/, l2data/, ...   # List length experiments (L=1,2,5,10,15)
│   ├── w128data/               # Window size W=128
│   ├── 16_2_data/              # Base config (p=16, x=2, L=20)
│   └── pairwise_data/          # Pairwise baseline (p=1, x=1)
├── hgt_pair/                # HGT pairwise comparison results
│   └── l1/, l2/, l5/, ...
├── tfilter/                 # Tournament filter results
├── sort/                    # Semantic sort experiments
│   └── hgt_metrics/
├── window_size_data/        # Window size experiments
│   └── {100,250,500,750,1000}/
├── k5000/                   # K-parameter experiments
├── tourk5000/               # Tournament sort on 5000 docs
├── lotusk/                  # LOTUS framework baselines
│   ├── combined_z7b/           # Zephyr-7B backend
│   └── qwen_data/              # QWEN3-8B backend
├── comparison_1_data/       # Pairwise comparison baseline
├── no-em-p/                 # No-embedding pivot selection
├── wsortdata/               # Sort window experiments
├── llm-topk-gt/             # LLM ground truth generation pipeline
│   └── data/
│       ├── phase2_ir_aggregation/
│       ├── phase3_reranking/
│       ├── phase4_rerank_aggregation/
│       ├── phase5_comparisons/
│       └── phase7_combined_rankings/
└── pxplot.py                # Plotting script
```

---

## Data Schemas

### `bier_result_unsorted_*.csv` - Raw Results

| Column | Type | Description |
|--------|------|-------------|
| `q` | int | Row index |
| `qid` | int | Query ID |
| `did` | list[str] | Retrieved document IDs |
| `n_call` | int | Number of LLM/model calls |
| `time` | float | Execution time (seconds) |
| `d_size` | list[int] | Document buffer sizes per stage |
| `m_size` | list[int] | Memory allocation per stage |
| `l_latency` | list[float] | Cumulative latency per stage |
| `c_count` | list[int] | Comparison counts per stage |
| `i_der` | list[list[str]] | Intermediate derived results per stage |

### `bier_metrics_*.csv` - Aggregate Metrics

| Column | Type | Description |
|--------|------|-------------|
| `NDCG@10` | float | Normalized DCG at 10 |
| `MAP@10` | float | Mean Average Precision at 10 |
| `Recall@10` | float | Recall at 10 |
| `P@10` | float | Precision at 10 |
| `time` | float | Total execution time (seconds) |

### `bier_i_metrics_*.csv` - Per-Query Metrics

| Column | Type | Description |
|--------|------|-------------|
| `q` | int | Query index |
| `NDCG@10` | list[float] | NDCG per query |
| `MAP@10` | list[float] | MAP per query |
| `Recall@10` | list[float] | Recall per query |
| `P@10` | list[float] | Precision per query |

### `bier_formatted_*.json` - Rankings

```json
{
  "query_id": {
    "doc_id": rank_position,
    ...
  },
  ...
}
```

### `tfilter` Results - Tournament Filter

| Column | Type | Description |
|--------|------|-------------|
| `q` | int | Query index |
| `doc_num` | int | Number of documents retained |
| `time` | float | Execution time |
| `ids` | list[str] | Retained document IDs |

**Tournament filter doc_num to L mapping:**

| doc_num | L value |
|---------|---------|
| 260 | 1 |
| 1295 | 5 |
| 2591 | 10 |
| 3887 | 15 |

---

## LLM Ground Truth Pipeline

The `llm-topk-gt/` directory contains a 7-phase pipeline for generating ground truth rankings using LLM pairwise comparisons.

### Phase Structure

| Phase | Directory | Output |
|-------|-----------|--------|
| 2 | `phase2_ir_aggregation/` | Aggregated IR results (RRF, Borda) |
| 3 | `phase3_reranking/` | Individual reranker outputs |
| 4 | `phase4_rerank_aggregation/` | Aggregated reranker results |
| 5 | `phase5_comparisons/` | Per-query pairwise LLM comparisons |
| 7 | `phase7_combined_rankings/` | Final combined rankings |

### Phase 2/4 Parquet Schema

| Column | Type | Description |
|--------|------|-------------|
| `query_id` | str | Query identifier |
| `doc_id` | str | Document identifier |
| `rank` | int | Rank position |
| `agg_score` | float | Aggregation score |

### Phase 5 Comparisons (`comparisons.jsonl`)

```json
{
  "comparison_id": "uuid",
  "query_id": "string",
  "doc_a_id": "string",
  "doc_b_id": "string",
  "presented_order": ["doc_a_id", "doc_b_id"],
  "winner_id": "string",
  "reasoning": "string",
  "model": "string",
  "timestamp": "ISO8601"
}
```

### Phase 5 Sorted Rankings (`sorted_ranking.json`)

```json
{
  "query_id": "string",
  "sorted_doc_ids": ["doc_id", ...],
  "top_k3_doc_ids": ["doc_id", ...],
  "stage1_comparisons": int,
  "stage2_comparisons": int
}
```

### Metadata Files (`metadata.json`)

```json
{
  "phase": 4,
  "phase_name": "rerank_aggregation",
  "dataset": "scifact",
  "rerankers": ["bge_reranker", "minilm_l12", "minilm_l6", "mmarco_minilm"],
  "num_queries": 1109,
  "methods": {"rrf": {...}, "borda": {...}},
  "rrf_k": 60,
  "timestamp": "ISO8601"
}
```

---

## Plotting Script (`pxplot.py`)

### Plot Types Generated

| Function | Output Files | Description |
|----------|--------------|-------------|
| `make_half_plot()` | `PvX_{var}_{label}.pdf/png` | P×X heatmap (lower triangular) |
| `make_box_plots()` | `em_and_nem_comparison_.pdf/png` | Embedding vs no-embedding latency |
| `make_tournament_plot()` | `tfilter_{k}_plot.pdf/png` | Max recall vs L (tournament filter) |
| `make_k_plot()` | `krecallplot.pdf/png` | Recall@K vs K |
| `sort_plots()` | `sort_ndcg@{k}.pdf/png`, `sort_time.pdf/png` | NDCG/latency vs window size |
| `a_recall_plot()` | `main_plot_recall@10.pdf/png` | Main comparison scatter (latency vs recall) |
| `a_plots_hgt()` | `hgt_main_plot_*.pdf/png` | HGT comparison scatter |
| `window_size_plot()` | `w_recall@10.pdf/png`, `w_latency.pdf/png` | Window size analysis |
| `wsort_plot()` | `sortw_ndcg@10.pdf/png` | Sort window NDCG |

### Key Comparisons

The main plots compare these methods:

| Label | Description | Config |
|-------|-------------|--------|
| `pairwise` | Pairwise comparison baseline | p=1, x=1 |
| `L=1,2,5,10,15,20` | Tournament with list length L | p=16, x=2, varying L |
| `W=128` | Large window size | W=128 |
| `LOTUS (Zephyr-7B)` | LOTUS framework baseline | Zephyr-7B LLM |
| `LOTUS (QWEN3-8B)` | LOTUS framework baseline | QWEN3-8B LLM |
| `Tournament Sort` | Tournament-style sorting | - |

### Ground Truth

All metrics are computed against the LLM-generated ground truth rankings at:
```
llm-topk-gt/data/phase7_combined_rankings/scifact/{query_id}.parquet
```

---

## Usage

Run all plots:
```bash
python pxplot.py
```

This generates PDF and PNG files for all configured experiments.

---

## Glossary

| Term | Full Name | Description |
|------|-----------|-------------|
| BIER | (Algorithm name) | Multi-pivot quickselect retrieval algorithm |
| HGT | Hierarchical Ground Truth | Pre-computed metrics using LLM ground truth |
| RRF | Reciprocal Rank Fusion | Rank aggregation method |
| Borda | Borda Count | Alternative rank aggregation method |
| LOTUS | - | External LLM-based retrieval framework for comparison |
| NDCG | Normalized Discounted Cumulative Gain | Ranking quality metric |
| MAP | Mean Average Precision | Ranking quality metric |
