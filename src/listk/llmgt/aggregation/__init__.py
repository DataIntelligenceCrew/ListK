"""Rank aggregation algorithms for combining multiple rankings.

This module provides various rank aggregation methods:
- RRF: Reciprocal Rank Fusion (fast, practical)
- Borda: Borda count (positional voting)
- Copeland: Copeland's method (Condorcet method)
- Schulze: Schulze/Beatpath method (sophisticated Condorcet method)
- RRF+Local: RRF with local search for concordance optimization
"""

from listk.llmgt.aggregation.base import AggregatedRanking, Aggregator
from listk.llmgt.aggregation.borda import BordaAggregator
from listk.llmgt.aggregation.copeland import CopelandAggregator
from listk.llmgt.aggregation.rrf import RRFAggregator
from listk.llmgt.aggregation.rrf_local import RRFLocalSearchAggregator
from listk.llmgt.aggregation.schulze import SchulzeAggregator

__all__ = [
    "Aggregator",
    "AggregatedRanking",
    "RRFAggregator",
    "RRFLocalSearchAggregator",
    "BordaAggregator",
    "CopelandAggregator",
    "SchulzeAggregator",
]


def get_aggregator(method: str, **kwargs) -> Aggregator:
    """Factory function to get an aggregator by name.

    Parameters
    ----------
    method : str
        Aggregation method name: "rrf", "borda", "copeland", or "schulze".
    **kwargs
        Additional arguments passed to the aggregator constructor.

    Returns
    -------
    Aggregator
        The requested aggregator instance.

    Raises
    ------
    ValueError
        If the method name is not recognized.
    """
    method_lower = method.lower()

    if method_lower == "rrf":
        return RRFAggregator(**kwargs)
    elif method_lower == "borda":
        return BordaAggregator()
    elif method_lower == "copeland":
        return CopelandAggregator(**kwargs)
    elif method_lower == "schulze":
        return SchulzeAggregator()
    elif method_lower in ("rrf_local", "rrf-local", "rrflocal"):
        return RRFLocalSearchAggregator(**kwargs)
    else:
        raise ValueError(
            f"Unknown aggregation method: {method}. "
            f"Available: rrf, borda, copeland, schulze, rrf_local"
        )
