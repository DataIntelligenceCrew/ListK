"""Borda count rank aggregation method.

Borda count is a positional voting method where each position in a ranking
contributes points. The document with the most total points wins.

Reference:
    Borda, J. C. (1781). Mémoire sur les élections au scrutin.
    Histoire de l'Académie royale des sciences.
"""

import numpy as np

from src.aggregation.base import AggregatedRanking, Aggregator


class BordaAggregator(Aggregator):
    """Borda count aggregator.

    Each ranker assigns points based on rank position:
        points(d, r) = (n - rank_r(d) + 1)

    where n is the number of documents ranked by that ranker.
    The aggregated score is the sum of points across all rankers.

    Notes
    -----
    - Documents missing from a ranker receive 0 points from that ranker.
    - This implementation uses the "modified Borda" where missing items
      get 0 points rather than being assigned worst rank.
    """

    def __init__(self) -> None:
        """Initialize Borda count aggregator."""
        super().__init__(name="borda")

    def aggregate_query(
        self,
        query_id: str,
        rankings: dict[str, dict[str, int]],
    ) -> AggregatedRanking:
        """Aggregate rankings using Borda count.

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
            n = len(ranker_rankings)
            for doc_id, rank in ranker_rankings.items():
                # Points = (n - rank + 1), so rank 1 gets n points
                doc_scores[doc_id] += n - rank + 1

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
