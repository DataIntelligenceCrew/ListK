"""Abstract base class for reranking models.

This module defines the Reranker interface that all cross-encoder
reranking models must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.data.beir_loader import BeirDataset
from src.data.models import Document, Query, RankingEntry


class Reranker(ABC):
    """Abstract base class for document reranking models.

    Rerankers score query-document pairs directly using cross-encoder
    architectures that jointly encode the query and document together.

    Attributes
    ----------
    name : str
        Unique identifier for this reranker (used in output filenames).
    """

    def __init__(self, name: str) -> None:
        """Initialize the reranker.

        Parameters
        ----------
        name : str
            Unique identifier for this reranker.
        """
        self.name: str = name

    @abstractmethod
    def score(
        self,
        query: Query,
        documents: list[Document],
        show_progress: bool = False,
    ) -> list[float]:
        """Score a list of documents for a single query.

        Parameters
        ----------
        query : Query
            The query to score documents against.
        documents : list[Document]
            List of documents to score.
        show_progress : bool, optional
            Whether to show a progress bar. Default False.

        Returns
        -------
        list[float]
            Relevance scores for each document (same order as input).
        """
        pass

    @abstractmethod
    def score_pairs(
        self,
        pairs: list[tuple[str, str]],
        show_progress: bool = True,
    ) -> list[float]:
        """Score a list of (query_text, document_text) pairs.

        Parameters
        ----------
        pairs : list[tuple[str, str]]
            List of (query_text, document_text) tuples.
        show_progress : bool, optional
            Whether to show a progress bar. Default True.

        Returns
        -------
        list[float]
            Relevance scores for each pair.
        """
        pass

    def rerank(
        self,
        query: Query,
        documents: list[Document],
        doc_ids: list[str],
        show_progress: bool = False,
    ) -> list[RankingEntry]:
        """Rerank documents for a single query.

        Parameters
        ----------
        query : Query
            The query.
        documents : list[Document]
            List of documents to rerank.
        doc_ids : list[str]
            Document IDs corresponding to documents list.
        show_progress : bool, optional
            Whether to show progress. Default False.

        Returns
        -------
        list[RankingEntry]
            Reranked documents sorted by score (descending).
        """
        scores: list[float] = self.score(query, documents, show_progress=show_progress)

        # Create (doc_id, score) pairs and sort by score descending
        doc_scores: list[tuple[str, float]] = list(zip(doc_ids, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Build ranking entries
        results: list[RankingEntry] = []
        for rank, (doc_id, score) in enumerate(doc_scores, start=1):
            results.append(
                RankingEntry(
                    query_id=query.query_id,
                    doc_id=doc_id,
                    rank=rank,
                    score=float(score),
                )
            )

        return results

    def rerank_from_rankings(
        self,
        query: Query,
        corpus: dict[str, Document],
        ranking_df: pd.DataFrame,
        top_k: int,
        show_progress: bool = False,
    ) -> list[RankingEntry]:
        """Rerank top-k documents from an existing ranking.

        Parameters
        ----------
        query : Query
            The query to rerank for.
        corpus : dict[str, Document]
            Full document corpus.
        ranking_df : pd.DataFrame
            DataFrame with columns: query_id, doc_id, rank, agg_score.
            Should be filtered to the specific query.
        top_k : int
            Number of top documents to rerank.
        show_progress : bool, optional
            Whether to show progress. Default False.

        Returns
        -------
        list[RankingEntry]
            Reranked documents.
        """
        # Get top-k doc_ids from ranking
        query_ranking: pd.DataFrame = ranking_df[
            ranking_df["query_id"] == query.query_id
        ].nsmallest(top_k, "rank")

        doc_ids: list[str] = query_ranking["doc_id"].tolist()
        documents: list[Document] = [corpus[doc_id] for doc_id in doc_ids]

        return self.rerank(query, documents, doc_ids, show_progress=show_progress)

    def run(
        self,
        dataset: BeirDataset,
        rankings_df: pd.DataFrame,
        top_k: int,
        query_ids: Optional[list[str]] = None,
        show_progress: bool = True,
    ) -> list[RankingEntry]:
        """Run reranking on top-k documents for multiple queries.

        Parameters
        ----------
        dataset : BeirDataset
            The dataset containing corpus and queries.
        rankings_df : pd.DataFrame
            Aggregated rankings from Phase 2 with columns:
            query_id, doc_id, rank, agg_score.
        top_k : int
            Number of top documents to rerank per query.
        query_ids : list[str], optional
            Specific query IDs to process. If None, process all queries
            present in rankings_df.
        show_progress : bool, optional
            Whether to show progress bars. Default True.

        Returns
        -------
        list[RankingEntry]
            All reranked entries for the specified queries.
        """
        # Determine queries to process
        if query_ids is None:
            query_ids = rankings_df["query_id"].unique().tolist()

        results: list[RankingEntry] = []

        iterator = tqdm(
            query_ids,
            desc=f"Reranking [{self.name}]",
            disable=not show_progress,
        )

        for query_id in iterator:
            if query_id not in dataset.queries:
                continue

            query: Query = dataset.queries[query_id]
            query_rankings: list[RankingEntry] = self.rerank_from_rankings(
                query=query,
                corpus=dataset.corpus,
                ranking_df=rankings_df,
                top_k=top_k,
                show_progress=False,
            )
            results.extend(query_rankings)

        return results

    @abstractmethod
    def clear(self) -> None:
        """Clear the model and free memory."""
        pass
