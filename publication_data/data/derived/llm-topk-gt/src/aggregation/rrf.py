"""Reciprocal Rank Fusion (RRF) aggregation method.

RRF is a simple but effective rank aggregation method that combines
rankings by summing reciprocal ranks. It is robust to outliers and
does not require score normalization.

Reference:
    Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
    Reciprocal rank fusion outperforms condorcet and individual
    rank learning methods. SIGIR '09.
"""

import numpy as np

from src.aggregation.base import AggregatedRanking, Aggregator


class RRFAggregator(Aggregator):
    """Reciprocal Rank Fusion aggregator.

    RRF computes the aggregated score for each document as:
        score(d) = sum_{r in rankers} 1 / (k + rank_r(d))

    where k is a smoothing constant (default 60).

    Attributes
    ----------
    k : int
        Smoothing constant to prevent division by very small numbers.
        Higher k reduces the impact of top ranks.

    Notes
    -----
    - Documents missing from a ranker are assigned a default rank of
      (max_rank + 1) for that ranker.
    - RRF is not Condorcet-consistent but is computationally efficient
      and performs well in practice.
    """

    def __init__(self, k: int = 60) -> None:
        """Initialize RRF aggregator.

        Parameters
        ----------
        k : int, optional
            Smoothing constant. Default 60 (standard value from the paper).
        """
        super().__init__(name=f"rrf_k{k}")
        self.k: int = k

    def aggregate_query(
        self,
        query_id: str,
        rankings: dict[str, dict[str, int]],
    ) -> AggregatedRanking:
        """Aggregate rankings using Reciprocal Rank Fusion.

        Parameters
        ----------
        query_id : str
            The query identifier.
        rankings : dict[str, dict[str, int]]
            Mapping from ranker name to {doc_id: rank} dictionaries.

        Returns
        -------
        AggregatedRanking
            The aggregated ranking result.
        """
        all_docs = self._get_all_documents(rankings)
        doc_scores: dict[str, float] = {doc: 0.0 for doc in all_docs}

        for ranker_rankings in rankings.values():
            # Default rank for missing documents
            max_rank = max(ranker_rankings.values()) if ranker_rankings else 0
            default_rank = max_rank + 1

            for doc in all_docs:
                rank = ranker_rankings.get(doc, default_rank)
                doc_scores[doc] += 1.0 / (self.k + rank)

        # Sort by score (descending) and assign ranks
        sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])
        doc_ids = np.array([doc for doc, _ in sorted_docs])
        scores = np.array([score for _, score in sorted_docs])
        ranks = np.arange(1, len(doc_ids) + 1, dtype=np.int32)

        return AggregatedRanking(
            query_id=query_id,
            doc_ids=doc_ids,
            scores=scores,
            ranks=ranks,
        )
