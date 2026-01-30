"""Retrieval module for IR-based document ranking.

This module provides implementations of various retrieval models
for the first phase of the LLM Top-K Ground Truth pipeline.

Retrievers
----------
BM25Retriever : Sparse lexical retrieval using BM25
SpladeRetriever : Learned sparse retrieval using SPLADE++
E5Retriever : Dense retrieval using E5-large-v2
BGERetriever : Dense retrieval using BGE-large-en-v1.5
ColBERTRetriever : Late-interaction retrieval using ColBERTv2

Base Classes
------------
Retriever : Abstract base class for all retrievers

Examples
--------
>>> from src.retrieval import get_retriever, BM25Retriever
>>> from src.data import load_dataset
>>>
>>> # Load dataset
>>> dataset = load_dataset("scifact", Path("./data"))
>>>
>>> # Use factory function
>>> retriever = get_retriever("bm25", top_n=100)
>>> rankings = retriever.run(dataset)
>>>
>>> # Or instantiate directly
>>> retriever = BM25Retriever(top_n=100)
>>> rankings = retriever.run(dataset)
"""

from src.retrieval.base import Retriever
from src.retrieval.bge import BGERetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.colbert import ColBERTRetriever
from src.retrieval.e5 import E5Retriever
from src.retrieval.splade import SpladeRetriever
from src.retrieval.utils import get_device, get_gpu_memory_gb

__all__ = [
    # Base class
    "Retriever",
    # Implementations
    "BM25Retriever",
    "SpladeRetriever",
    "E5Retriever",
    "BGERetriever",
    "ColBERTRetriever",
    # Utilities
    "get_device",
    "get_gpu_memory_gb",
    # Factory
    "get_retriever",
    "RETRIEVER_REGISTRY",
]


# Retriever registry for dynamic instantiation
RETRIEVER_REGISTRY: dict[str, type[Retriever]] = {
    "bm25": BM25Retriever,
    "splade": SpladeRetriever,
    "e5": E5Retriever,
    "bge": BGERetriever,
    "colbert": ColBERTRetriever,
}


def get_retriever(name: str, **kwargs) -> Retriever:
    """Get a retriever instance by name.

    Parameters
    ----------
    name : str
        Retriever name ('bm25', 'splade', 'e5', 'bge', 'colbert').
    **kwargs
        Additional arguments passed to the retriever constructor.

    Returns
    -------
    Retriever
        Configured retriever instance.

    Raises
    ------
    ValueError
        If retriever name is not recognized.

    Examples
    --------
    >>> retriever = get_retriever("bm25", top_n=100)
    >>> retriever = get_retriever("e5", batch_size=64, device="cuda")
    """
    if name not in RETRIEVER_REGISTRY:
        raise ValueError(
            f"Unknown retriever: {name}. "
            f"Available: {list(RETRIEVER_REGISTRY.keys())}"
        )
    return RETRIEVER_REGISTRY[name](**kwargs)
