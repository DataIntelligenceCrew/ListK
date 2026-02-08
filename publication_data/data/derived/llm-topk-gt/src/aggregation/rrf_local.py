"""RRF with Local Search for pairwise concordance optimization.

This method starts with RRF rankings and then performs local search
to improve pairwise concordance with the original rankers. It uses
adjacent swaps to find a local optimum.

The concordance score counts how many pairwise orderings in the
aggregated ranking agree with the majority of rankers.
"""

import numpy as np
from typing import Optional

from src.aggregation.base import AggregatedRanking, Aggregator


class RRFLocalSearchAggregator(Aggregator):
    """RRF with local search optimization for pairwise concordance.

    This aggregator:
    1. Computes initial ranking using RRF
    2. Performs local search with adjacent swaps to maximize concordance
    3. Stops when no improving swap exists (local optimum)

    Concordance measures agreement with the majority of rankers:
    For each pair (d_i, d_j) where d_i is ranked above d_j, we count
    how many rankers agree vs disagree with this ordering.

    Attributes
    ----------
    k : int
        RRF smoothing constant.
    max_iterations : int
        Maximum number of local search iterations to prevent infinite loops.
    verbose : bool
        Whether to print progress information.

    Notes
    -----
    - This is more expensive than RRF: O(n^2 * R) per query for building
      the preference matrix, plus O(n * max_iterations) for local search.
    - Guarantees a local optimum in pairwise concordance.
    """

    def __init__(
        self,
        k: int = 60,
        max_iterations: int = 1000,
        verbose: bool = False,
    ) -> None:
        """Initialize RRF+LocalSearch aggregator.

        Parameters
        ----------
        k : int, optional
            RRF smoothing constant. Default 60.
        max_iterations : int, optional
            Maximum local search iterations. Default 1000.
        verbose : bool, optional
            Whether to print progress. Default False.
        """
        super().__init__(name="rrf_local")
        self.k: int = k
        self.max_iterations: int = max_iterations
        self.verbose: bool = verbose

    def _compute_rrf_ranking(
        self,
        rankings: dict[str, dict[str, int]],
        all_docs: list[str],
    ) -> list[str]:
        """Compute initial RRF ranking.

        Parameters
        ----------
        rankings : dict[str, dict[str, int]]
            Mapping from ranker name to {doc_id: rank} dictionaries.
        all_docs : list[str]
            List of all document IDs.

        Returns
        -------
        list[str]
            Document IDs sorted by RRF score (best first).
        """
        doc_scores: dict[str, float] = {doc: 0.0 for doc in all_docs}

        for ranker_rankings in rankings.values():
            max_rank = max(ranker_rankings.values()) if ranker_rankings else 0
            default_rank = max_rank + 1

            for doc in all_docs:
                rank = ranker_rankings.get(doc, default_rank)
                doc_scores[doc] += 1.0 / (self.k + rank)

        sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])
        return [doc for doc, _ in sorted_docs]

    def _build_pairwise_margin_matrix(
        self,
        rankings: dict[str, dict[str, int]],
        doc_ids: list[str],
    ) -> np.ndarray:
        """Build matrix of pairwise preference margins.

        Parameters
        ----------
        rankings : dict[str, dict[str, int]]
            Mapping from ranker name to {doc_id: rank} dictionaries.
        doc_ids : list[str]
            List of document IDs (defines matrix indices).

        Returns
        -------
        np.ndarray
            Matrix where entry [i, j] is the number of rankers preferring
            doc_ids[i] over doc_ids[j] minus those preferring j over i.
            Positive values mean i is preferred over j by majority.
        """
        n_docs = len(doc_ids)
        doc_to_idx = {doc: idx for idx, doc in enumerate(doc_ids)}

        # preference[i, j] = # rankers preferring i over j
        preference = np.zeros((n_docs, n_docs), dtype=np.int32)

        for ranker_rankings in rankings.values():
            max_rank = max(ranker_rankings.values()) if ranker_rankings else 0
            default_rank = max_rank + 1

            for i, doc_i in enumerate(doc_ids):
                rank_i = ranker_rankings.get(doc_i, default_rank)
                for j, doc_j in enumerate(doc_ids):
                    if i != j:
                        rank_j = ranker_rankings.get(doc_j, default_rank)
                        if rank_i < rank_j:  # i is ranked better
                            preference[i, j] += 1

        # Margin matrix: margin[i,j] = preference[i,j] - preference[j,i]
        margin = preference - preference.T
        return margin

    def _compute_concordance(
        self,
        ranking_indices: np.ndarray,
        margin_matrix: np.ndarray,
    ) -> int:
        """Compute total concordance score for a ranking.

        Parameters
        ----------
        ranking_indices : np.ndarray
            Array of document indices in rank order (index 0 = rank 1).
        margin_matrix : np.ndarray
            Pairwise margin matrix.

        Returns
        -------
        int
            Total concordance: sum of margin[i,j] for all pairs where
            i is ranked above j in the given ranking.
        """
        n = len(ranking_indices)
        concordance = 0

        for pos_i in range(n):
            idx_i = ranking_indices[pos_i]
            for pos_j in range(pos_i + 1, n):
                idx_j = ranking_indices[pos_j]
                # i is ranked above j, add margin[i,j]
                concordance += margin_matrix[idx_i, idx_j]

        return concordance

    def _compute_swap_delta(
        self,
        ranking_indices: np.ndarray,
        pos: int,
        margin_matrix: np.ndarray,
    ) -> int:
        """Compute concordance change from swapping adjacent elements.

        Parameters
        ----------
        ranking_indices : np.ndarray
            Current ranking as array of document indices.
        pos : int
            Position to swap (swaps pos and pos+1).
        margin_matrix : np.ndarray
            Pairwise margin matrix.

        Returns
        -------
        int
            Change in concordance if swap is performed.
            Positive means improvement.
        """
        idx_i = ranking_indices[pos]
        idx_j = ranking_indices[pos + 1]

        # Currently i is above j, contributing margin[i,j]
        # After swap, j is above i, contributing margin[j,i] = -margin[i,j]
        # Delta = margin[j,i] - margin[i,j] = -2 * margin[i,j]
        return -2 * margin_matrix[idx_i, idx_j]

    def _local_search(
        self,
        initial_ranking: list[str],
        margin_matrix: np.ndarray,
        doc_to_idx: dict[str, int],
    ) -> list[str]:
        """Perform local search to optimize concordance.

        Parameters
        ----------
        initial_ranking : list[str]
            Initial ranking from RRF (doc_ids in rank order).
        margin_matrix : np.ndarray
            Pairwise margin matrix.
        doc_to_idx : dict[str, int]
            Mapping from doc_id to matrix index.

        Returns
        -------
        list[str]
            Optimized ranking (doc_ids in rank order).
        """
        n = len(initial_ranking)
        if n <= 1:
            return initial_ranking

        # Convert to indices for efficient operations
        ranking_indices = np.array([doc_to_idx[doc] for doc in initial_ranking])
        idx_to_doc = {idx: doc for doc, idx in doc_to_idx.items()}

        iteration = 0
        improved = True

        while improved and iteration < self.max_iterations:
            improved = False
            iteration += 1

            # Try all adjacent swaps, find best improvement
            best_delta = 0
            best_pos = -1

            for pos in range(n - 1):
                delta = self._compute_swap_delta(ranking_indices, pos, margin_matrix)
                if delta > best_delta:
                    best_delta = delta
                    best_pos = pos

            # Make the best swap if it improves concordance
            if best_pos >= 0:
                ranking_indices[best_pos], ranking_indices[best_pos + 1] = (
                    ranking_indices[best_pos + 1],
                    ranking_indices[best_pos],
                )
                improved = True

        if self.verbose and iteration > 1:
            print(f"    Local search: {iteration} iterations")

        # Convert back to doc_ids
        return [idx_to_doc[idx] for idx in ranking_indices]

    def aggregate_query(
        self,
        query_id: str,
        rankings: dict[str, dict[str, int]],
    ) -> AggregatedRanking:
        """Aggregate rankings using RRF + local search.

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

        if len(all_docs) == 0:
            return AggregatedRanking(
                query_id=query_id,
                doc_ids=np.array([], dtype=object),
                scores=np.array([], dtype=np.float64),
                ranks=np.array([], dtype=np.int32),
            )

        # Step 1: Get initial RRF ranking
        rrf_ranking = self._compute_rrf_ranking(rankings, all_docs)

        # Step 2: Build pairwise margin matrix
        doc_to_idx = {doc: idx for idx, doc in enumerate(all_docs)}
        margin_matrix = self._build_pairwise_margin_matrix(rankings, all_docs)

        # Step 3: Local search optimization
        optimized_ranking = self._local_search(rrf_ranking, margin_matrix, doc_to_idx)

        # Compute final scores (use negative rank as score for consistency)
        n = len(optimized_ranking)
        doc_ids = np.array(optimized_ranking)
        ranks = np.arange(1, n + 1, dtype=np.int32)
        # Score based on position (higher is better)
        scores = np.array([1.0 / (self.k + r) for r in ranks], dtype=np.float64)

        return AggregatedRanking(
            query_id=query_id,
            doc_ids=doc_ids,
            scores=scores,
            ranks=ranks,
        )
