#!/usr/bin/env python3
"""Benchmark retrieval methods on SciFact dataset.

This script measures the time required for each retriever to:
1. Index the corpus
2. Retrieve for 1 query
3. Estimate total time for all queries
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_dataset
from src.retrieval import RETRIEVER_REGISTRY, get_retriever


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def benchmark_retriever(
    retriever_name: str,
    dataset,
    top_n: int = 100,
    num_test_queries: int = 5,
) -> dict:
    """Benchmark a single retriever.

    Returns dict with timing info.
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {retriever_name.upper()}")
    print(f"{'='*60}")

    results = {
        "name": retriever_name,
        "status": "pending",
    }

    try:
        # Initialize retriever
        retriever = get_retriever(retriever_name, top_n=top_n)

        # Time indexing
        print(f"  Indexing {len(dataset.corpus)} documents...")
        start = time.time()
        retriever.index(dataset.corpus, show_progress=True)
        index_time = time.time() - start
        print(f"  Index time: {format_time(index_time)}")
        results["index_time"] = index_time

        # Get test queries
        query_ids = dataset.get_query_ids()[:num_test_queries]
        queries = [dataset.queries[qid] for qid in query_ids]

        # Time single query retrieval
        single_query = queries[0]
        start = time.time()
        _ = retriever.retrieve(single_query)
        single_query_time = time.time() - start
        print(f"  Single query time: {format_time(single_query_time)}")
        results["single_query_time"] = single_query_time

        # Time batch retrieval for test queries
        start = time.time()
        _ = retriever.retrieve_batch(queries, show_progress=False)
        batch_time = time.time() - start
        avg_per_query = batch_time / len(queries)
        print(f"  Batch ({len(queries)} queries) time: {format_time(batch_time)}")
        print(f"  Avg per query (batch): {format_time(avg_per_query)}")
        results["batch_time"] = batch_time
        results["avg_per_query"] = avg_per_query

        # Estimate total time for all queries
        total_queries = len(dataset.queries)
        estimated_total = index_time + (avg_per_query * total_queries)
        print(f"  Estimated total ({total_queries} queries): {format_time(estimated_total)}")
        results["estimated_total"] = estimated_total
        results["total_queries"] = total_queries
        results["status"] = "success"

        # Cleanup
        retriever.clear_index()

    except ImportError as e:
        print(f"  SKIPPED: Missing dependency - {e}")
        results["status"] = "skipped"
        results["error"] = str(e)
    except Exception as e:
        print(f"  FAILED: {e}")
        results["status"] = "failed"
        results["error"] = str(e)

    return results


def main():
    print("Loading SciFact dataset...")
    data_dir = Path("./tmp")
    dataset = load_dataset("scifact", data_dir)
    print(f"  Corpus: {len(dataset.corpus)} documents")
    print(f"  Queries: {len(dataset.queries)} queries")

    # Benchmark each retriever
    retrievers_to_test = ["bm25", "e5", "bge", "splade", "colbert"]
    results = []

    for name in retrievers_to_test:
        result = benchmark_retriever(name, dataset, top_n=100, num_test_queries=5)
        results.append(result)

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Retriever':<12} {'Status':<10} {'Index':<10} {'1 Query':<10} {'Est. Total':<12}")
    print("-"*80)

    for r in results:
        name = r["name"]
        status = r["status"]
        if status == "success":
            index = format_time(r["index_time"])
            single = format_time(r["single_query_time"])
            total = format_time(r["estimated_total"])
        else:
            index = single = total = "N/A"
        print(f"{name:<12} {status:<10} {index:<10} {single:<10} {total:<12}")

    print("-"*80)
    print(f"Total queries: {len(dataset.queries)}")
    print(f"Corpus size: {len(dataset.corpus)} documents")


if __name__ == "__main__":
    main()
