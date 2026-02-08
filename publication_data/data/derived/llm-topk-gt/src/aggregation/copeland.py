"""Copeland's method for rank aggregation.

Copeland's method is a Condorcet method that scores each candidate based
on the number of pairwise victories minus the number of defeats. It
selects the Condorcet winner when one exists.

Reference:
    Copeland, A. H. (1951). A reasonable social welfare function.
    Seminar on Mathematics in Social Sciences, University of Michigan.
"""

import numpy as np

from src.aggregation.base import AggregatedRanking, Aggregator


class CopelandAggregator(Aggregator):
    """Copeland's method aggregator.

    For each pair of documents (A, B), we count how many rankers prefer
    A over B (A has lower rank number). A beats B if more rankers prefer
    A than prefer B. The Copeland score is:

        copeland_score(d) = wins(d) - losses(d)

    Ties in pairwise comparisons contribute to neither wins nor losses.

    Attributes
    ----------
    tie_breaker : str
        Method for breaking ties: "borda" (use Borda score) or "none".

    Notes
    -----
    - This is a Condorcet method: if a document beats all others head-to-head,
      it will be ranked first.
    - Ties are common with Copeland's method; the tie_breaker parameter
      controls how these are resolved.
    """

    def __init__(self, tie_breaker: str = "borda") -> None:
        """Initialize Copeland aggregator.

        Parameters
        ----------
        tie_breaker : str, optional
            Method for breaking ties. Options:
            - "borda": Use Borda count as secondary sort key.
            - "none": Do not break ties (may result in arbitrary order).
            Default "borda".
        """
        super().__init__(name="copeland")
        self.tie_breaker: str = tie_breaker

    def aggregate_query(
        self,
        query_id: str,
        rankings: dict[str, dict[str, int]],
    ) -> AggregatedRanking:
        """Aggregate rankings using Copeland's method.

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
        all_docs = list(self._get_all_documents(rankings))
        n_docs = len(all_docs)
        n_rankers = len(rankings)

        # Build preference matrix
        pref_matrix = self._build_preference_matrix(rankings, all_docs)

        # Compute Copeland scores
        # wins[i] = number of j where pref_matrix[i,j] > pref_matrix[j,i]
        # losses[i] = number of j where pref_matrix[i,j] < pref_matrix[j,i]
        wins = np.zeros(n_docs, dtype=np.int32)
        losses = np.zeros(n_docs, dtype=np.int32)

        for i in range(n_docs):
            for j in range(n_docs):
                if i == j:
                    continue
                if pref_matrix[i, j] > pref_matrix[j, i]:
                    wins[i] += 1
                elif pref_matrix[i, j] < pref_matrix[j, i]:
                    losses[i] += 1
                # Ties contribute to neither

        copeland_scores = wins - losses

        # Prepare for sorting
        if self.tie_breaker == "borda":
            # Compute Borda scores for tie-breaking
            borda_scores = self._compute_borda_scores(rankings, all_docs)
            # Sort by Copeland (desc), then Borda (desc)
            sort_keys = [(-copeland_scores[i], -borda_scores[i], all_docs[i])
                         for i in range(n_docs)]
        else:
            # Sort by Copeland only
            sort_keys = [(-copeland_scores[i], all_docs[i])
                         for i in range(n_docs)]

        sorted_indices = sorted(range(n_docs), key=lambda i: sort_keys[i])

        doc_ids = np.array([all_docs[i] for i in sorted_indices])
        scores = np.array([float(copeland_scores[i]) for i in sorted_indices])
        ranks = np.arange(1, n_docs + 1, dtype=np.int32)

        return AggregatedRanking(
            query_id=query_id,
            doc_ids=doc_ids,
            scores=scores,
            ranks=ranks,
        )

    def _compute_borda_scores(
        self,
        rankings: dict[str, dict[str, int]],
        doc_ids: list[str],
    ) -> np.ndarray:
        """Compute Borda scores for tie-breaking.

        Parameters
        ----------
        rankings : dict[str, dict[str, int]]
            Mapping from ranker name to {doc_id: rank} dictionaries.
        doc_ids : list[str]
            List of document IDs.

        Returns
        -------
        np.ndarray
            Borda scores for each document.
        """
        scores = np.zeros(len(doc_ids))
        doc_to_idx = {doc: idx for idx, doc in enumerate(doc_ids)}

        for ranker_rankings in rankings.values():
            n = len(ranker_rankings)
            for doc_id, rank in ranker_rankings.items():
                idx = doc_to_idx.get(doc_id)
                if idx is not None:
                    scores[idx] += n - rank + 1

        return scores
