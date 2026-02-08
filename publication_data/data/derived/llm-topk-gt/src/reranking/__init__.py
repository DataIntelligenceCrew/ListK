"""Reranking module for cross-encoder based document reranking.

This module provides implementations of various cross-encoder reranking models
for Phase 3 of the LLM Top-K Ground Truth pipeline.

Rerankers
---------
MiniLML6Reranker : Fast cross-encoder using MiniLM L6
MiniLML12Reranker : Medium cross-encoder using MiniLM L12
BGEReranker : Large cross-encoder using BGE-reranker-large
MMarcoMiniLMReranker : Multilingual cross-encoder using mMARCO MiniLM

Base Classes
------------
Reranker : Abstract base class for all rerankers
CrossEncoderReranker : Generic cross-encoder wrapper

Examples
--------
>>> from src.reranking import get_reranker, MiniLML6Reranker
>>> from src.data import load_dataset
>>> import pandas as pd
>>>
>>> # Load dataset and Phase 2 rankings
>>> dataset = load_dataset("scifact", Path("./data"))
>>> rankings = pd.read_parquet("data/phase2_ir_aggregation/scifact/aggregated_rrf.parquet")
>>>
>>> # Use factory function
>>> reranker = get_reranker("minilm_l6", batch_size=64)
>>> results = reranker.run(dataset, rankings, top_k=500)
>>>
>>> # Or instantiate directly
>>> reranker = MiniLML6Reranker(batch_size=64)
>>> results = reranker.run(dataset, rankings, top_k=500)
"""

from src.reranking.base import Reranker
from src.reranking.cross_encoder import (
    BGEReranker,
    CrossEncoderReranker,
    MiniLML6Reranker,
    MiniLML12Reranker,
    MMarcoMiniLMReranker,
)

__all__ = [
    # Base classes
    "Reranker",
    "CrossEncoderReranker",
    # Implementations
    "MiniLML6Reranker",
    "MiniLML12Reranker",
    "BGEReranker",
    "MMarcoMiniLMReranker",
    # Factory
    "get_reranker",
    "RERANKER_REGISTRY",
]


# Reranker registry for dynamic instantiation
RERANKER_REGISTRY: dict[str, type[Reranker]] = {
    "minilm_l6": MiniLML6Reranker,
    "minilm_l12": MiniLML12Reranker,
    "bge_reranker": BGEReranker,
    "mmarco_minilm": MMarcoMiniLMReranker,
}


def get_reranker(name: str, **kwargs) -> Reranker:
    """Get a reranker instance by name.

    Parameters
    ----------
    name : str
        Reranker name ('minilm_l6', 'minilm_l12', 'bge_reranker', 'mmarco_minilm').
    **kwargs
        Additional arguments passed to the reranker constructor.

    Returns
    -------
    Reranker
        Configured reranker instance.

    Raises
    ------
    ValueError
        If reranker name is not recognized.

    Examples
    --------
    >>> reranker = get_reranker("minilm_l6", batch_size=64)
    >>> reranker = get_reranker("bge_reranker", device="cuda")
    """
    if name not in RERANKER_REGISTRY:
        raise ValueError(
            f"Unknown reranker: {name}. "
            f"Available: {list(RERANKER_REGISTRY.keys())}"
        )
    return RERANKER_REGISTRY[name](**kwargs)
