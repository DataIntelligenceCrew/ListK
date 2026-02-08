#!/usr/bin/env python
"""Phase 7: Combined Full-Corpus Rankings

This script creates full-corpus rankings for queries with completed Phase 5
(LLM-based) comparisons. For each document in the corpus, it uses the most
accurate ranking available:

Ranking hierarchy (highest accuracy first):
1. LLM ranking (Phase 5) - for top-k2 documents ranked via LLM comparisons
2. Reranker ensemble (Phase 4) - for documents in reranker results but not in LLM
3. IR ensemble (Phase 2) - for documents in retriever results but not in reranker
4. Unranked - documents not retrieved by any method (placed at the end)

Usage:
    python scripts/run_phase7.py --dataset scifact
    python scripts/run_phase7.py --dataset scifact --query-ids 1,3,5

Input:
    data/phase5_comparisons/{dataset}/{query_id}/sorted_ranking.json
    data/phase4_rerank_aggregation/{dataset}/aggregated_rrf.parquet
    data/phase2_ir_aggregation/{dataset}/aggregated_rrf.parquet
    data/raw/{dataset}/corpus.jsonl

Output:
    data/phase7_combined_rankings/{dataset}/{query_id}.parquet
    data/phase7_combined_rankings/{dataset}/metadata.json
"""

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(verbose: bool = False) -> None:
    """Configure logging.

    Parameters
    ----------
    verbose : bool
        If True, use DEBUG level. Otherwise INFO.
    """
    level: int = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Phase 7: Combined Full-Corpus Rankings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all completed Phase 5 queries
    python scripts/run_phase7.py --dataset scifact

    # Process specific queries
    python scripts/run_phase7.py --dataset scifact --query-ids 1,3,5

    # Force reprocessing
    python scripts/run_phase7.py --dataset scifact --force
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., scifact, scidocs)",
    )

    parser.add_argument(
        "--query-ids",
        type=str,
        default=None,
        help="Comma-separated list of query IDs to process (default: all completed)",
    )

    parser.add_argument(
        "--agg-method",
        type=str,
        default="rrf",
        help="Aggregation method to use from Phase 2/4 (default: rrf)",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Base data directory (default: ./data)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if output exists",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def load_corpus_doc_ids(raw_dir: Path, dataset: str) -> set[str]:
    """Load all document IDs from the corpus.

    Parameters
    ----------
    raw_dir : Path
        Path to raw data directory.
    dataset : str
        Dataset name.

    Returns
    -------
    set[str]
        Set of all document IDs in the corpus.
    """
    corpus_path: Path = raw_dir / dataset / "corpus.jsonl"
    doc_ids: set[str] = set()

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            doc: dict = json.loads(line)
            doc_ids.add(doc["_id"])

    return doc_ids


def get_completed_query_ids(phase5_dir: Path, dataset: str) -> list[str]:
    """Get list of query IDs with completed Phase 5 comparisons.

    Parameters
    ----------
    phase5_dir : Path
        Path to Phase 5 output directory.
    dataset : str
        Dataset name.

    Returns
    -------
    list[str]
        List of query IDs with completed comparisons.
    """
    metadata_path: Path = phase5_dir / dataset / "metadata.json"

    if not metadata_path.exists():
        return []

    with open(metadata_path, "r") as f:
        metadata: dict = json.load(f)

    completed_ids: list[str] = []
    queries: dict = metadata.get("queries", {})

    for query_id, info in queries.items():
        if info.get("status") == "completed":
            completed_ids.append(query_id)

    return sorted(completed_ids, key=lambda x: int(x) if x.isdigit() else x)


def load_phase5_ranking(phase5_dir: Path, dataset: str, query_id: str) -> list[str]:
    """Load LLM-based sorted ranking from Phase 5.

    Parameters
    ----------
    phase5_dir : Path
        Path to Phase 5 output directory.
    dataset : str
        Dataset name.
    query_id : str
        Query ID.

    Returns
    -------
    list[str]
        Ordered list of document IDs (best to worst).
    """
    ranking_path: Path = phase5_dir / dataset / query_id / "sorted_ranking.json"

    with open(ranking_path, "r") as f:
        data: dict = json.load(f)

    return data["sorted_doc_ids"]


def load_phase4_ranking(
    phase4_dir: Path,
    dataset: str,
    agg_method: str,
    query_id: str,
) -> dict[str, int]:
    """Load reranker-aggregated ranking from Phase 4.

    Parameters
    ----------
    phase4_dir : Path
        Path to Phase 4 output directory.
    dataset : str
        Dataset name.
    agg_method : str
        Aggregation method (e.g., "rrf").
    query_id : str
        Query ID.

    Returns
    -------
    dict[str, int]
        Mapping from doc_id to rank (1-indexed).
    """
    parquet_path: Path = phase4_dir / dataset / f"aggregated_{agg_method}.parquet"
    df: pd.DataFrame = pd.read_parquet(parquet_path)

    query_df: pd.DataFrame = df[df["query_id"] == query_id]
    return dict(zip(query_df["doc_id"], query_df["rank"]))


def load_phase2_ranking(
    phase2_dir: Path,
    dataset: str,
    agg_method: str,
    query_id: str,
) -> dict[str, int]:
    """Load IR-aggregated ranking from Phase 2.

    Parameters
    ----------
    phase2_dir : Path
        Path to Phase 2 output directory.
    dataset : str
        Dataset name.
    agg_method : str
        Aggregation method (e.g., "rrf").
    query_id : str
        Query ID.

    Returns
    -------
    dict[str, int]
        Mapping from doc_id to rank (1-indexed).
    """
    parquet_path: Path = phase2_dir / dataset / f"aggregated_{agg_method}.parquet"
    df: pd.DataFrame = pd.read_parquet(parquet_path)

    query_df: pd.DataFrame = df[df["query_id"] == query_id]
    return dict(zip(query_df["doc_id"], query_df["rank"]))


def combine_rankings(
    corpus_doc_ids: set[str],
    phase5_ranking: list[str],
    phase4_ranking: dict[str, int],
    phase2_ranking: dict[str, int],
) -> pd.DataFrame:
    """Combine rankings from all phases into a full-corpus ranking.

    Uses the most accurate ranking available for each document:
    1. LLM ranking (Phase 5) - highest accuracy
    2. Reranker ensemble (Phase 4)
    3. IR ensemble (Phase 2)
    4. Unranked (documents not in any retriever)

    Parameters
    ----------
    corpus_doc_ids : set[str]
        Set of all document IDs in the corpus.
    phase5_ranking : list[str]
        Ordered list of doc_ids from Phase 5 (LLM ranking).
    phase4_ranking : dict[str, int]
        Mapping from doc_id to rank from Phase 4.
    phase2_ranking : dict[str, int]
        Mapping from doc_id to rank from Phase 2.

    Returns
    -------
    pd.DataFrame
        Combined ranking with columns:
        - doc_id: Document ID
        - rank: Final rank (1-indexed)
        - source: Which phase provided the ranking ("llm", "reranker", "ir", "unranked")
        - source_rank: Original rank within that source
    """
    # Track which docs we've ranked
    ranked_docs: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    current_rank: int = 1

    # Tier 1: LLM rankings (Phase 5)
    phase5_set: set[str] = set(phase5_ranking)
    for source_rank, doc_id in enumerate(phase5_ranking, start=1):
        ranked_docs.append({
            "doc_id": doc_id,
            "rank": current_rank,
            "source": "llm",
            "source_rank": source_rank,
        })
        seen_doc_ids.add(doc_id)
        current_rank += 1

    # Tier 2: Reranker rankings (Phase 4) - docs not in Phase 5
    # Note: In practice, Phase 4 and Phase 5 have the same doc set,
    # so this tier will likely be empty
    phase4_only: list[tuple[str, int]] = [
        (doc_id, rank)
        for doc_id, rank in phase4_ranking.items()
        if doc_id not in seen_doc_ids
    ]
    phase4_only.sort(key=lambda x: x[1])  # Sort by original rank

    for doc_id, source_rank in phase4_only:
        ranked_docs.append({
            "doc_id": doc_id,
            "rank": current_rank,
            "source": "reranker",
            "source_rank": source_rank,
        })
        seen_doc_ids.add(doc_id)
        current_rank += 1

    # Tier 3: IR rankings (Phase 2) - docs not in Phase 4 or 5
    phase2_only: list[tuple[str, int]] = [
        (doc_id, rank)
        for doc_id, rank in phase2_ranking.items()
        if doc_id not in seen_doc_ids
    ]
    phase2_only.sort(key=lambda x: x[1])  # Sort by original rank

    for doc_id, source_rank in phase2_only:
        ranked_docs.append({
            "doc_id": doc_id,
            "rank": current_rank,
            "source": "ir",
            "source_rank": source_rank,
        })
        seen_doc_ids.add(doc_id)
        current_rank += 1

    # Tier 4: Unranked documents (not in any retriever results)
    unranked_docs: list[str] = sorted(corpus_doc_ids - seen_doc_ids)

    for doc_id in unranked_docs:
        ranked_docs.append({
            "doc_id": doc_id,
            "rank": current_rank,
            "source": "unranked",
            "source_rank": 0,  # No meaningful source rank
        })
        current_rank += 1

    df: pd.DataFrame = pd.DataFrame(ranked_docs)

    # Ensure correct dtypes
    df = df.astype({
        "doc_id": "string",
        "rank": "int32",
        "source": "string",
        "source_rank": "int32",
    })

    return df


def save_combined_ranking(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save combined ranking to parquet.

    Parameters
    ----------
    df : pd.DataFrame
        Combined ranking DataFrame.
    output_path : Path
        Output file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


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

    # Setup paths
    data_dir: Path = args.data_dir.resolve()
    raw_dir: Path = data_dir / "raw"
    phase2_dir: Path = data_dir / "phase2_ir_aggregation"
    phase4_dir: Path = data_dir / "phase4_rerank_aggregation"
    phase5_dir: Path = data_dir / "phase5_comparisons"
    output_dir: Path = data_dir / "phase7_combined_rankings" / args.dataset

    logger.info("Phase 7: Combined Full-Corpus Rankings")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Aggregation method: {args.agg_method}")

    # Load corpus document IDs
    logger.info("Loading corpus document IDs...")
    corpus_doc_ids: set[str] = load_corpus_doc_ids(raw_dir, args.dataset)
    logger.info(f"Corpus size: {len(corpus_doc_ids):,} documents")

    # Determine which queries to process
    if args.query_ids:
        query_ids: list[str] = [q.strip() for q in args.query_ids.split(",")]
        logger.info(f"Processing specified queries: {query_ids}")
    else:
        query_ids = get_completed_query_ids(phase5_dir, args.dataset)
        logger.info(f"Found {len(query_ids)} completed Phase 5 queries")

    if not query_ids:
        logger.warning("No queries to process. Run Phase 5 first.")
        return 1

    # Track metadata
    metadata: dict[str, Any] = {
        "phase": 7,
        "phase_name": "combined_corpus_rankings",
        "dataset": args.dataset,
        "agg_method": args.agg_method,
        "corpus_size": len(corpus_doc_ids),
        "queries": {},
        "timestamp": datetime.now(UTC).isoformat(),
    }

    # Process each query
    success_count: int = 0
    for query_id in query_ids:
        output_path: Path = output_dir / f"{query_id}.parquet"

        if output_path.exists() and not args.force:
            logger.info(f"Skipping query {query_id} (output exists, use --force)")
            metadata["queries"][query_id] = {"status": "skipped"}
            continue

        logger.info(f"Processing query {query_id}...")
        start_time: datetime = datetime.now(UTC)

        try:
            # Load rankings from each phase
            phase5_ranking: list[str] = load_phase5_ranking(
                phase5_dir, args.dataset, query_id
            )
            phase4_ranking: dict[str, int] = load_phase4_ranking(
                phase4_dir, args.dataset, args.agg_method, query_id
            )
            phase2_ranking: dict[str, int] = load_phase2_ranking(
                phase2_dir, args.dataset, args.agg_method, query_id
            )

            # Combine rankings
            combined_df: pd.DataFrame = combine_rankings(
                corpus_doc_ids,
                phase5_ranking,
                phase4_ranking,
                phase2_ranking,
            )

            # Save output
            save_combined_ranking(combined_df, output_path)

            elapsed: float = (datetime.now(UTC) - start_time).total_seconds()

            # Count docs per tier
            tier_counts: dict[str, int] = combined_df["source"].value_counts().to_dict()

            logger.info(
                f"Query {query_id}: {len(combined_df):,} docs ranked in {elapsed:.2f}s"
            )
            logger.info(
                f"  Tier breakdown: LLM={tier_counts.get('llm', 0)}, "
                f"Reranker={tier_counts.get('reranker', 0)}, "
                f"IR={tier_counts.get('ir', 0)}, "
                f"Unranked={tier_counts.get('unranked', 0)}"
            )

            metadata["queries"][query_id] = {
                "status": "completed",
                "num_docs": len(combined_df),
                "tier_counts": tier_counts,
                "elapsed_seconds": elapsed,
            }
            success_count += 1

        except FileNotFoundError as e:
            logger.error(f"Query {query_id}: Missing file - {e}")
            metadata["queries"][query_id] = {
                "status": "failed",
                "error": str(e),
            }
        except Exception as e:
            logger.exception(f"Query {query_id}: Failed - {e}")
            metadata["queries"][query_id] = {
                "status": "failed",
                "error": str(e),
            }

    # Compute summary statistics
    if success_count > 0:
        completed_queries: list[dict] = [
            q for q in metadata["queries"].values()
            if q.get("status") == "completed"
        ]

        total_llm: int = sum(q["tier_counts"].get("llm", 0) for q in completed_queries)
        total_reranker: int = sum(
            q["tier_counts"].get("reranker", 0) for q in completed_queries
        )
        total_ir: int = sum(q["tier_counts"].get("ir", 0) for q in completed_queries)
        total_unranked: int = sum(
            q["tier_counts"].get("unranked", 0) for q in completed_queries
        )

        metadata["summary"] = {
            "completed_queries": success_count,
            "total_queries": len(query_ids),
            "avg_llm_docs": total_llm / success_count if success_count > 0 else 0,
            "avg_reranker_docs": (
                total_reranker / success_count if success_count > 0 else 0
            ),
            "avg_ir_docs": total_ir / success_count if success_count > 0 else 0,
            "avg_unranked_docs": (
                total_unranked / success_count if success_count > 0 else 0
            ),
        }

    metadata["end_timestamp"] = datetime.now(UTC).isoformat()

    # Save metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path: Path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Phase 7 complete. {success_count}/{len(query_ids)} queries processed."
    )
    logger.info(f"Outputs in: {output_dir}")

    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
