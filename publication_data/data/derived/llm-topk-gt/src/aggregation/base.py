"""Abstract base class for rank aggregation methods.

This module defines the Aggregator interface that all rank aggregation
algorithms must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AggregatedRanking:
    """Result of aggregating multiple rankings for a single query.

    Attributes
    ----------
    query_id : str
        The query identifier.
    doc_ids : np.ndarray
        Document IDs in aggregated rank order (index 0 = rank 1).
    scores : np.ndarray
        Aggregation scores corresponding to each document.
    ranks : np.ndarray
        Rank positions (1-indexed).
    """

    query_id: str
    doc_ids: np.ndarray
    scores: np.ndarray
    ranks: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame format.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: query_id, doc_id, rank, agg_score.
        """
        return pd.DataFrame({
            "query_id": self.query_id,
            "doc_id": self.doc_ids,
            "rank": self.ranks,
            "agg_score": self.scores,
        })


class Aggregator(ABC):
    """Abstract base class for rank aggregation methods.

    Rank aggregation methods combine multiple ranked lists into a single
    consensus ranking. Different methods have different properties
    regarding Condorcet consistency, computational complexity, and
    handling of ties.

    Attributes
    ----------
    name : str
        Unique identifier for this aggregation method.
    """

    def __init__(self, name: str) -> None:
        """Initialize the aggregator.

        Parameters
        ----------
        name : str
            Unique identifier for this aggregation method.
        """
        self.name: str = name

    @abstractmethod
    def aggregate_query(
        self,
        query_id: str,
        rankings: dict[str, dict[str, int]],
    ) -> AggregatedRanking:
        """Aggregate rankings from multiple rankers for a single query.

        Parameters
        ----------
        query_id : str
            The query identifier.
        rankings : dict[str, dict[str, int]]
            Mapping from ranker name to {doc_id: rank} dictionaries.
            Ranks are 1-indexed (1 = best).

        Returns
        -------
        AggregatedRanking
            The aggregated ranking result.

        Notes
        -----
        Documents that appear in some but not all rankings are handled
        according to each method's specific strategy.
        """
        pass

    def aggregate_all(
        self,
        rankings_df: pd.DataFrame,
        ranker_column: str = "ranker",
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Aggregate rankings for all queries in a DataFrame.

        Parameters
        ----------
        rankings_df : pd.DataFrame
            DataFrame with columns: query_id, doc_id, rank, and a column
            identifying the ranker (specified by ranker_column).
        ranker_column : str, optional
            Name of the column containing ranker identifiers. Default "ranker".
        show_progress : bool, optional
            Whether to show a progress bar. Default True.

        Returns
        -------
        pd.DataFrame
            Aggregated rankings with columns: query_id, doc_id, rank, agg_score.
        """
        from tqdm import tqdm

        query_ids = rankings_df["query_id"].unique()
        rankers = rankings_df[ranker_column].unique()

        results: list[pd.DataFrame] = []

        iterator = tqdm(
            query_ids,
            desc=f"Aggregating [{self.name}]",
            disable=not show_progress,
        )

        for query_id in iterator:
            # Build rankings dict for this query
            query_df = rankings_df[rankings_df["query_id"] == query_id]

            rankings: dict[str, dict[str, int]] = {}
            for ranker in rankers:
                ranker_df = query_df[query_df[ranker_column] == ranker]
                rankings[ranker] = dict(
                    zip(ranker_df["doc_id"], ranker_df["rank"])
                )

            # Aggregate and collect result
            agg_result = self.aggregate_query(query_id, rankings)
            results.append(agg_result.to_dataframe())

        return pd.concat(results, ignore_index=True)

    def _get_all_documents(
        self,
        rankings: dict[str, dict[str, int]],
    ) -> set[str]:
        """Get the union of all documents across all rankers.

        Parameters
        ----------
        rankings : dict[str, dict[str, int]]
            Mapping from ranker name to {doc_id: rank} dictionaries.

        Returns
        -------
        set[str]
            Set of all document IDs appearing in any ranking.
        """
        all_docs: set[str] = set()
        for ranker_rankings in rankings.values():
            all_docs.update(ranker_rankings.keys())
        return all_docs

    def _build_preference_matrix(
        self,
        rankings: dict[str, dict[str, int]],
        doc_ids: list[str],
    ) -> np.ndarray:
        """Build a pairwise preference matrix from rankings.

        Parameters
        ----------
        rankings : dict[str, dict[str, int]]
            Mapping from ranker name to {doc_id: rank} dictionaries.
        doc_ids : list[str]
            List of document IDs (defines matrix indices).

        Returns
        -------
        np.ndarray
            Square matrix where entry [i, j] is the number of rankers
            that prefer doc_ids[i] over doc_ids[j].
        """
        n_docs = len(doc_ids)
        doc_to_idx = {doc: idx for idx, doc in enumerate(doc_ids)}
        n_rankers = len(rankings)

        # preference_matrix[i, j] = number of rankers preferring i over j
        preference_matrix = np.zeros((n_docs, n_docs), dtype=np.int32)

        for ranker_rankings in rankings.values():
            # Assign default rank for missing documents (worst possible)
            max_rank = max(ranker_rankings.values()) if ranker_rankings else 0
            default_rank = max_rank + 1

            for i, doc_i in enumerate(doc_ids):
                rank_i = ranker_rankings.get(doc_i, default_rank)
                for j, doc_j in enumerate(doc_ids):
                    if i == j:
                        continue
                    rank_j = ranker_rankings.get(doc_j, default_rank)
                    # Lower rank number = better = preferred
                    if rank_i < rank_j:
                        preference_matrix[i, j] += 1

        return preference_matrix
