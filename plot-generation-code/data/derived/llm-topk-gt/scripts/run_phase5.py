#!/usr/bin/env python3
"""Run Phase 5: LLM Pairwise Comparison Collection (Two-Stage).

This script executes the fifth phase of the pipeline using a two-stage approach:
1. Stage 1: Merge sort top-k2 documents using O(n log n) LLM comparisons
2. Stage 2: Full pairwise comparisons on top-k3 documents (quadratic)

Usage
-----
    # Run with default settings (top-500 merge sort, top-100 quadratic)
    python scripts/run_phase5.py --dataset scifact --model llama-3.1-8b

    # Quick test with small values
    python scripts/run_phase5.py --dataset scifact --model llama-3.1-8b --top-k2 20 --top-k3 5 --query-ids 1

    # Run for first 25 test queries
    python scripts/run_phase5.py --dataset scifact --model llama-3.1-8b --max-queries 25

Examples
--------
    # Test on first query with small k values
    python scripts/run_phase5.py --dataset scifact --model llama-3.1-8b --top-k2 10 --top-k3 5 --query-ids 1 -v

    # Full run for first 25 test queries
    python scripts/run_phase5.py --dataset scifact --model llama-3.1-8b --top-k2 500 --top-k3 100 --max-queries 25

    # Resume: skip queries with Phase 7 rankings, process remaining up to 25
    python scripts/run_phase5.py --dataset scifact --model gpt-4o-mini --max-queries 25 --skip-completed

Notes
-----
    - Only processes TEST split queries (not train queries)
    - Stage 1 (merge sort): ~n*log(n) comparisons (e.g., ~4500 for n=500)
    - Stage 2 (quadratic): n*(n-1)/2 comparisons (e.g., ~4950 for n=100)
    - Total for k2=500, k3=100: ~9500 comparisons per query (vs 124,750 for full quadratic)
"""

import argparse
import json
import logging
import math
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_dataset
from src.data.beir_loader import load_qrels
from src.data.models import PairwiseComparison
from src.llm import (
    DOC_MODE_SNIPPET,
    DOC_MODE_SUMMARY,
    DOC_MODE_TITLE,
    DOC_MODES,
    DocumentSummarizer,
    LLM_REGISTRY,
    LLMComparator,
    generate_all_pairs,
    get_completed_pairs,
    get_llm_backend,
    run_pairwise_comparison,
    sort_documents_with_llm,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging.

    Parameters
    ----------
    verbose : bool
        If True, use DEBUG level. Otherwise WARNING (quiet mode).
    """
    # Default to WARNING to keep output clean for progress bars
    # Use -v for verbose INFO/DEBUG output
    level: int = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run Phase 5: LLM Pairwise Comparison Collection (Two-Stage)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test with one query
    python scripts/run_phase5.py --dataset scifact --model llama-3.1-8b --top-k2 10 --top-k3 5 --query-ids 1

    # Run for first 25 test queries
    python scripts/run_phase5.py --dataset scifact --model llama-3.1-8b --max-queries 25

    # Use OpenAI model
    python scripts/run_phase5.py --dataset scifact --model gpt-4o --top-k2 500 --top-k3 100
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (scifact, scidocs)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"LLM model name. Available: {list(LLM_REGISTRY.keys())}",
    )

    parser.add_argument(
        "--top-k2",
        type=int,
        default=500,
        help="Number of top docs for Stage 1 merge sort (default: 500)",
    )

    parser.add_argument(
        "--top-k3",
        type=int,
        default=100,
        help="Number of top docs for Stage 2 quadratic comparison (default: 100)",
    )

    parser.add_argument(
        "--query-ids",
        type=str,
        default=None,
        help="Comma-separated list of specific query IDs to process",
    )

    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum number of test queries to process (default: all)",
    )

    parser.add_argument(
        "--agg-method",
        type=str,
        default="rrf",
        choices=["rrf", "borda"],
        help="Aggregation method from Phase 4 to use (default: rrf)",
    )

    parser.add_argument(
        "--sort-algorithm",
        type=str,
        default="quick_batched",
        choices=["merge", "quick", "quick_batched"],
        help="Sorting algorithm for Stage 1 (default: quick_batched for parallel)",
    )

    parser.add_argument(
        "--doc-mode",
        type=str,
        default="title",
        choices=["title", "summary", "snippet"],
        help="Document representation in prompts: 'title' (just title), 'summary' (LLM-generated), 'snippet' (text excerpt). Default: title",
    )

    parser.add_argument(
        "--summary-model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for generating summaries (only used with --doc-mode=summary). Default: gpt-4o-mini (fast/cheap)",
    )

    parser.add_argument(
        "--max-snippet-chars",
        type=int,
        default=1500,
        help="Maximum characters per document snippet in prompt (default: 1500, only for --doc-mode=snippet)",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Base data directory (default: ./data)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for presentation order randomization (default: 42)",
    )

    parser.add_argument(
        "--no-randomize",
        action="store_true",
        help="Disable randomization of document presentation order",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available LLM models and exit",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without running comparisons",
    )

    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip queries that already have final rankings in phase7_combined_rankings",
    )

    return parser.parse_args()


def get_test_query_ids(data_dir: Path, dataset: str) -> set[str]:
    """Get the set of test query IDs for a dataset.

    Parameters
    ----------
    data_dir : Path
        Base data directory.
    dataset : str
        Dataset name.

    Returns
    -------
    set[str]
        Set of query IDs in the test split.
    """
    dataset_path: Path = data_dir / "raw" / dataset
    test_qrels: dict[str, dict[str, int]] = load_qrels(dataset_path, split="test")
    return set(test_qrels.keys())


def get_completed_phase7_query_ids(data_dir: Path, dataset: str) -> set[str]:
    """Get query IDs that already have final rankings in Phase 7.

    Parameters
    ----------
    data_dir : Path
        Base data directory.
    dataset : str
        Dataset name.

    Returns
    -------
    set[str]
        Set of query IDs with completed Phase 7 rankings.
    """
    phase7_dir: Path = data_dir / "phase7_combined_rankings" / dataset
    if not phase7_dir.exists():
        return set()

    completed: set[str] = set()
    for parquet_file in phase7_dir.glob("*.parquet"):
        # Extract query ID from filename (e.g., "123.parquet" -> "123")
        query_id: str = parquet_file.stem
        # Skip metadata file if it exists as parquet
        if query_id != "metadata":
            completed.add(query_id)

    return completed


def load_phase4_rankings(
    data_dir: Path,
    dataset: str,
    agg_method: str,
) -> pd.DataFrame:
    """Load aggregated rankings from Phase 4.

    Parameters
    ----------
    data_dir : Path
        Base data directory.
    dataset : str
        Dataset name.
    agg_method : str
        Aggregation method ('rrf' or 'borda').

    Returns
    -------
    pd.DataFrame
        Aggregated rankings with columns: query_id, doc_id, rank, agg_score.

    Raises
    ------
    FileNotFoundError
        If Phase 4 output does not exist.
    """
    phase4_dir: Path = data_dir / "phase4_rerank_aggregation" / dataset
    rankings_path: Path = phase4_dir / f"aggregated_{agg_method}.parquet"

    if not rankings_path.exists():
        raise FileNotFoundError(
            f"Phase 4 rankings not found at {rankings_path}. "
            f"Run Phase 4 first with: python scripts/run_phase4.py --dataset {dataset}"
        )

    return pd.read_parquet(rankings_path)


def load_existing_comparisons(comparisons_file: Path) -> list[PairwiseComparison]:
    """Load existing comparisons from JSONL file.

    Parameters
    ----------
    comparisons_file : Path
        Path to comparisons.jsonl file.

    Returns
    -------
    list[PairwiseComparison]
        List of existing comparisons.
    """
    comparisons: list[PairwiseComparison] = []

    if not comparisons_file.exists():
        return comparisons

    with open(comparisons_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data: dict = json.loads(line)
                comparisons.append(PairwiseComparison(**data))

    return comparisons


def append_comparisons(
    comparisons_file: Path, comparisons: list[PairwiseComparison]
) -> None:
    """Append comparisons to the JSONL file.

    Parameters
    ----------
    comparisons_file : Path
        Path to comparisons.jsonl file.
    comparisons : list[PairwiseComparison]
        The comparisons to append.
    """
    comparisons_file.parent.mkdir(parents=True, exist_ok=True)

    with open(comparisons_file, "a") as f:
        for comp in comparisons:
            f.write(comp.model_dump_json() + "\n")


def estimate_comparisons(k2: int, k3: int) -> tuple[int, int, int]:
    """Estimate number of comparisons for two-stage approach.

    Parameters
    ----------
    k2 : int
        Number of documents for Stage 1 (merge sort).
    k3 : int
        Number of documents for Stage 2 (quadratic).

    Returns
    -------
    tuple[int, int, int]
        (stage1_comparisons, stage2_comparisons, total)
    """
    # Merge sort: O(n log n)
    stage1: int = int(k2 * math.log2(k2)) if k2 > 1 else 0
    # Quadratic: n*(n-1)/2
    stage2: int = k3 * (k3 - 1) // 2
    return stage1, stage2, stage1 + stage2


def main() -> int:
    """Main entry point.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    args: argparse.Namespace = parse_args()
    setup_logging(args.verbose)
    logger: logging.Logger = logging.getLogger(__name__)

    # Handle --list-models flag
    if args.list_models:
        print("Available LLM models:")
        print("\n  Local models (Ollama - easy setup, requires ollama serve):")
        for name in sorted(LLM_REGISTRY.keys()):
            if name.startswith("ollama-"):
                print(f"    - {name}")
        print("\n  Local models (vLLM - requires GPU + vLLM setup):")
        for name in sorted(LLM_REGISTRY.keys()):
            if name.startswith("llama-"):
                print(f"    - {name}")
        print("\n  OpenAI API models:")
        for name in sorted(LLM_REGISTRY.keys()):
            if not name.startswith(("ollama-", "llama-")):
                print(f"    - {name}")
        return 0

    # Validate required arguments
    if not args.dataset:
        print("Error: --dataset is required")
        return 1
    if not args.model:
        print("Error: --model is required")
        return 1

    # Set random seed
    random.seed(args.seed)

    # Validate model
    if args.model not in LLM_REGISTRY:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available: {', '.join(sorted(LLM_REGISTRY.keys()))}")
        return 1

    # Validate k3 <= k2
    if args.top_k3 > args.top_k2:
        print(f"Error: top-k3 ({args.top_k3}) cannot be larger than top-k2 ({args.top_k2})")
        return 1

    # Setup paths
    data_dir: Path = args.data_dir.resolve()
    output_dir: Path = data_dir / "phase5_comparisons" / args.dataset

    # Print config summary
    print("=" * 60)
    print("Phase 5: LLM Pairwise Comparison (Two-Stage)")
    print("=" * 60)
    print(f"Dataset:    {args.dataset}")
    print(f"Model:      {args.model}")
    print(f"Stage 1:    {args.sort_algorithm} sort top-{args.top_k2} docs")
    print(f"Stage 2:    Quadratic on top-{args.top_k3} docs")
    print(f"Doc mode:   {args.doc_mode}")

    # Estimate comparisons
    s1_est, s2_est, total_est = estimate_comparisons(args.top_k2, args.top_k3)
    print(f"Est. comparisons/query: ~{s1_est:,} + {s2_est:,} = ~{total_est:,}")
    print("-" * 60)

    # Get test query IDs (only process test queries)
    print("Loading test queries...", end=" ", flush=True)
    test_query_ids: set[str] = get_test_query_ids(data_dir, args.dataset)
    print(f"{len(test_query_ids)} found")

    # Determine which queries to process
    if args.query_ids:
        requested_ids: list[str] = [q.strip() for q in args.query_ids.split(",")]
        # Filter to only test queries
        query_ids: list[str] = [q for q in requested_ids if q in test_query_ids]
        skipped: list[str] = [q for q in requested_ids if q not in test_query_ids]
        if skipped:
            print(f"Warning: Skipping non-test query IDs: {skipped}")
    else:
        query_ids = sorted(test_query_ids, key=lambda x: int(x) if x.isdigit() else x)

    # Apply max-queries limit
    if args.max_queries and len(query_ids) > args.max_queries:
        query_ids = query_ids[: args.max_queries]

    # Skip queries with completed Phase 7 rankings
    skipped_count: int = 0
    if args.skip_completed:
        completed_phase7: set[str] = get_completed_phase7_query_ids(
            data_dir, args.dataset
        )
        if completed_phase7:
            original_count: int = len(query_ids)
            query_ids = [q for q in query_ids if q not in completed_phase7]
            skipped_count = original_count - len(query_ids)
            if skipped_count > 0:
                print(
                    f"Skipping {skipped_count} queries with Phase 7 rankings "
                    f"({len(query_ids)} remaining)"
                )

    print(f"Processing: {len(query_ids)} queries")

    if not query_ids:
        if skipped_count > 0:
            print("All requested queries already have Phase 7 rankings. Nothing to do.")
            return 0
        print("Error: No valid test queries to process")
        return 1

    # Load Phase 4 rankings
    print("Loading Phase 4 rankings...", end=" ", flush=True)
    try:
        rankings_df: pd.DataFrame = load_phase4_rankings(
            data_dir, args.dataset, args.agg_method
        )
        print(f"{len(rankings_df):,} rows")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1

    # Load dataset (for corpus and queries)
    print("Loading dataset...", end=" ", flush=True)
    try:
        dataset = load_dataset(args.dataset, data_dir)
        print(f"{len(dataset.corpus):,} docs, {len(dataset.queries):,} queries")
    except Exception as e:
        print(f"\nError: Failed to load dataset: {e}")
        return 1

    if args.dry_run:
        total_pairs: int = total_est * len(query_ids)
        print("-" * 60)
        print(f"Dry run: would run ~{total_pairs:,} comparisons")
        print("Exiting without running comparisons")
        return 0

    # Initialize LLM backend
    print(f"Initializing LLM backend...", end=" ", flush=True)
    try:
        llm = get_llm_backend(args.model)
        print("ready")
    except Exception as e:
        print(f"\nError: Failed to initialize LLM: {e}")
        return 1

    print("=" * 60)
    print()

    # Track metadata
    metadata: dict[str, Any] = {
        "phase": 5,
        "phase_name": "llm_comparisons_two_stage",
        "dataset": args.dataset,
        "model": args.model,
        "top_k2": args.top_k2,
        "top_k3": args.top_k3,
        "sort_algorithm": args.sort_algorithm,
        "agg_method": args.agg_method,
        "doc_mode": args.doc_mode,
        "max_snippet_chars": args.max_snippet_chars,
        "randomize_order": not args.no_randomize,
        "seed": args.seed,
        "total_queries": len(query_ids),
        "estimated_comparisons_per_query": total_est,
        "queries": {},
        "timestamp": datetime.now(UTC).isoformat(),
    }

    # Initialize document summarizer if needed
    summaries: dict[str, str] = {}
    if args.doc_mode == DOC_MODE_SUMMARY:
        print("Generating document summaries...")
        # Use a separate (typically cheaper/faster) model for summarization
        if args.summary_model != args.model:
            print(f"  Using {args.summary_model} for summarization...")
            summary_llm = get_llm_backend(args.summary_model)
        else:
            summary_llm = llm
        summarizer = DocumentSummarizer(summary_llm, max_concurrent=10)

        # Collect all unique doc IDs across all queries
        all_doc_ids: set[str] = set()
        for query_id in query_ids:
            query_rankings = rankings_df[rankings_df["query_id"] == query_id].nsmallest(
                args.top_k2, "rank"
            )
            all_doc_ids.update(query_rankings["doc_id"].tolist())

        # Get documents and summarize
        docs_to_summarize: list = [
            dataset.corpus[doc_id] for doc_id in all_doc_ids if doc_id in dataset.corpus
        ]
        print(f"  Summarizing {len(docs_to_summarize)} unique documents...")

        pbar_summary = tqdm(
            total=len(docs_to_summarize),
            desc="  Summarizing",
            unit="doc",
            position=0,
        )

        def summary_progress(count: int) -> None:
            pbar_summary.n = count
            pbar_summary.refresh()

        summaries = summarizer.summarize_batch(docs_to_summarize, summary_progress)
        pbar_summary.close()
        print(f"  Generated {len(summaries)} summaries")
        metadata["summaries_generated"] = len(summaries)
        metadata["summary_model"] = args.summary_model

        # Clean up summary model if it was separate
        if args.summary_model != args.model:
            summary_llm.clear()

    # Process each query
    success_count: int = 0
    total_comparisons: int = 0

    # Main progress bar for queries
    query_pbar = tqdm(query_ids, desc="Queries", unit="query", position=0)

    for query_id in query_pbar:
        query_start: datetime = datetime.now(UTC)

        # Get top-k2 documents for this query
        query_rankings: pd.DataFrame = rankings_df[
            rankings_df["query_id"] == query_id
        ].nsmallest(args.top_k2, "rank")

        doc_ids: list[str] = query_rankings["doc_id"].tolist()

        if len(doc_ids) < 2:
            logger.warning(f"Query {query_id} has fewer than 2 documents, skipping")
            metadata["queries"][query_id] = {"status": "skipped", "reason": "too_few_docs"}
            continue

        # Verify query exists
        if query_id not in dataset.queries:
            logger.error(f"Query {query_id} not found in dataset")
            metadata["queries"][query_id] = {"status": "failed", "reason": "query_not_found"}
            continue

        query = dataset.queries[query_id]

        # Setup output directory for this query
        query_output_dir: Path = output_dir / query_id
        query_output_dir.mkdir(parents=True, exist_ok=True)

        all_comparisons: list[PairwiseComparison] = []

        try:
            # ========== STAGE 1: Merge Sort ==========
            # Progress bar for stage 1
            pbar_s1 = tqdm(
                total=s1_est,
                desc=f"  Q{query_id} Sort",
                unit="cmp",
                position=1,
                leave=False,
            )

            def stage1_progress(count: int) -> None:
                pbar_s1.n = min(count, s1_est)
                pbar_s1.refresh()

            sorted_doc_ids, stage1_comparisons = sort_documents_with_llm(
                doc_ids=doc_ids,
                llm=llm,
                query=query,
                corpus=dataset.corpus,
                dataset_name=args.dataset,
                algorithm=args.sort_algorithm,
                max_snippet_chars=args.max_snippet_chars,
                randomize_order=not args.no_randomize,
                doc_mode=args.doc_mode,
                summaries=summaries,
                progress_callback=stage1_progress,
            )

            pbar_s1.close()
            all_comparisons.extend(stage1_comparisons)

            # ========== STAGE 2: Quadratic Comparisons on Top-k3 ==========
            top_k3_doc_ids: list[str] = sorted_doc_ids[: args.top_k3]

            # Get pairs that need comparison (excluding those from stage 1)
            completed_pairs: set[tuple[str, str]] = get_completed_pairs(stage1_comparisons)
            all_pairs: list[tuple[str, str]] = generate_all_pairs(top_k3_doc_ids)
            remaining_pairs: list[tuple[str, str]] = [
                p for p in all_pairs if p not in completed_pairs
            ]
            # Shuffle pairs for fairness (avoid position bias in processing order)
            random.shuffle(remaining_pairs)

            stage2_count: int = 0
            if remaining_pairs:
                # Use batched comparator for parallel execution
                stage2_comparator = LLMComparator(
                    llm=llm,
                    query=query,
                    corpus=dataset.corpus,
                    dataset_name=args.dataset,
                    randomize_order=not args.no_randomize,
                    max_snippet_chars=args.max_snippet_chars,
                    minimal=True,
                    doc_mode=args.doc_mode,
                    summaries=summaries,
                )
                # Pre-populate cache from stage 1 comparisons
                for comp in stage1_comparisons:
                    cache_key = (min(comp.doc_a_id, comp.doc_b_id), max(comp.doc_a_id, comp.doc_b_id))
                    stage2_comparator._cache[cache_key] = comp.winner_id

                # Show progress bar for stage 2
                pbar_s2 = tqdm(
                    total=len(remaining_pairs),
                    desc=f"  Q{query_id} Pairs",
                    unit="cmp",
                    position=1,
                    leave=False,
                )

                # Batch all pairs at once (async parallel)
                stage2_comparator.compare_batch(remaining_pairs)
                pbar_s2.n = len(remaining_pairs)
                pbar_s2.refresh()
                pbar_s2.close()

                all_comparisons.extend(stage2_comparator.comparisons)
                stage2_count = len(stage2_comparator.comparisons)

            # ========== Save Results ==========
            comparisons_file: Path = query_output_dir / "comparisons.jsonl"
            append_comparisons(comparisons_file, all_comparisons)

            # Save sorted ranking
            ranking_file: Path = query_output_dir / "sorted_ranking.json"
            with open(ranking_file, "w") as f:
                json.dump(
                    {
                        "query_id": query_id,
                        "sorted_doc_ids": sorted_doc_ids,
                        "top_k3_doc_ids": top_k3_doc_ids,
                        "stage1_comparisons": len(stage1_comparisons),
                        "stage2_comparisons": stage2_count,
                    },
                    f,
                    indent=2,
                )

            elapsed: float = (datetime.now(UTC) - query_start).total_seconds()
            total_comparisons += len(all_comparisons)

            metadata["queries"][query_id] = {
                "status": "completed",
                "num_docs": len(doc_ids),
                "stage1_comparisons": len(stage1_comparisons),
                "stage2_comparisons": stage2_count,
                "total_comparisons": len(all_comparisons),
                "elapsed_seconds": elapsed,
            }

            success_count += 1

            # Update main progress bar description with stats
            query_pbar.set_postfix(
                {"done": success_count, "comps": total_comparisons},
                refresh=True
            )

        except Exception as e:
            logger.exception(f"Failed to process query {query_id}: {e}")
            metadata["queries"][query_id] = {"status": "failed", "error": str(e)}

    query_pbar.close()

    # Clean up LLM
    llm.clear()

    # Save metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata["completed_queries"] = success_count
    metadata["total_comparisons"] = total_comparisons
    metadata["end_timestamp"] = datetime.now(UTC).isoformat()

    metadata_path: Path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print()
    print("=" * 60)
    print("Complete!")
    print("=" * 60)
    print(f"Queries:     {success_count}/{len(query_ids)} processed")
    print(f"Comparisons: {total_comparisons:,} total")
    print(f"Output:      {output_dir}")
    print("=" * 60)

    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
