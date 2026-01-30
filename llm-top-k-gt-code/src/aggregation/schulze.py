"""Schulze method for rank aggregation.

The Schulze method (also known as Beatpath or Path Winner) is a Condorcet
voting method that computes the strongest paths between all pairs of
candidates. It is one of the most robust rank aggregation methods and
satisfies many desirable voting criteria.

Reference:
    Schulze, M. (2011). A new monotonic, clone-independent, reversal
    symmetric, and condorcet-consistent single-winner election method.
    Social Choice and Welfare, 36(2), 267-303.
"""

import numpy as np

from src.aggregation.base import AggregatedRanking, Aggregator


class SchulzeAggregator(Aggregator):
    """Schulze method aggregator (Beatpath / Path Winner).

    The Schulze method works by:
    1. Building a pairwise preference matrix from all rankings
    2. Computing the strength of the strongest path between all pairs
       using a modified Floyd-Warshall algorithm
    3. Ranking candidates by the number of other candidates they beat
       via the strongest path criterion

    Key Properties
    --------------
    - Condorcet-consistent: If a Condorcet winner exists, it is selected.
    - Clone-independent: Adding similar alternatives doesn't change the
      outcome for dissimilar alternatives.
    - Reversal symmetric: Reversing all preferences reverses the outcome.
    - Monotonic: Improving a candidate's position cannot hurt them.

    Notes
    -----
    - The strongest path strength from A to B is the maximum over all
      paths from A to B of the minimum edge weight along the path.
    - Edge weight d[i,j] is the number of rankers preferring i over j.
    - This implementation uses the "winning votes" margin variant.
    """

    def __init__(self) -> None:
        """Initialize Schulze aggregator."""
        super().__init__(name="schulze")

    def aggregate_query(
        self,
        query_id: str,
        rankings: dict[str, dict[str, int]],
    ) -> AggregatedRanking:
        """Aggregate rankings using the Schulze method.

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

        # Build preference matrix: pref[i,j] = # rankers preferring i over j
        pref_matrix = self._build_preference_matrix(rankings, all_docs)

        # Compute strongest paths using Floyd-Warshall variant
        strength = self._compute_strongest_paths(pref_matrix)

        # Compute Schulze ranking scores
        # score[i] = number of j where strength[i,j] > strength[j,i]
        scores = np.zeros(n_docs, dtype=np.float64)
        for i in range(n_docs):
            for j in range(n_docs):
                if i != j and strength[i, j] > strength[j, i]:
                    scores[i] += 1

        # Sort by score (descending), with Borda tie-breaker
        borda_scores = self._compute_borda_scores(rankings, all_docs)
        sort_keys = [(-scores[i], -borda_scores[i], all_docs[i])
                     for i in range(n_docs)]
        sorted_indices = sorted(range(n_docs), key=lambda i: sort_keys[i])

        doc_ids = np.array([all_docs[i] for i in sorted_indices])
        final_scores = np.array([scores[i] for i in sorted_indices])
        ranks = np.arange(1, n_docs + 1, dtype=np.int32)

        return AggregatedRanking(
            query_id=query_id,
            doc_ids=doc_ids,
            scores=final_scores,
            ranks=ranks,
        )

    def _compute_strongest_paths(
        self,
        pref_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute strongest path strengths between all pairs.

        Uses a modified Floyd-Warshall algorithm where:
        - Path strength = minimum edge weight along the path
        - Strongest path = maximum strength over all paths

        Parameters
        ----------
        pref_matrix : np.ndarray
            Pairwise preference matrix where pref_matrix[i,j] is the
            number of rankers preferring i over j.

        Returns
        -------
        np.ndarray
            Matrix where strength[i,j] is the strength of the strongest
            path from i to j.
        """
        n = pref_matrix.shape[0]

        # Initialize strength matrix
        # strength[i,j] = pref[i,j] if pref[i,j] > pref[j,i], else 0
        strength = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(n):
                if i != j:
                    if pref_matrix[i, j] > pref_matrix[j, i]:
                        strength[i, j] = pref_matrix[i, j]

        # Floyd-Warshall variant for strongest paths
        # strength[i,j] = max over all k of min(strength[i,k], strength[k,j])
        for k in range(n):
            for i in range(n):
                if i == k:
                    continue
                for j in range(n):
                    if j == i or j == k:
                        continue
                    # Path through k: min of (i->k, k->j)
                    path_through_k = min(strength[i, k], strength[k, j])
                    if path_through_k > strength[i, j]:
                        strength[i, j] = path_through_k

        return strength

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
