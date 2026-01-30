"""E5 dense retriever implementation using sentence-transformers.

This module implements dense retrieval using the E5-large-v2 model.
"""

import logging
from typing import Optional

import numpy as np

from src.data.models import Document, Query, RankingEntry
from src.retrieval.base import Retriever
from src.retrieval.utils import get_device

logger: logging.Logger = logging.getLogger(__name__)


class E5Retriever(Retriever):
    """E5-large-v2 dense retriever.

    Attributes
    ----------
    model_name : str
        HuggingFace model identifier.
    batch_size : int
        Batch size for encoding.
    device : str
        Device to run model on ('cuda' or 'cpu').
    _model : SentenceTransformer or None
        The embedding model.
    _doc_embeddings : np.ndarray or None
        Cached document embeddings (num_docs, embedding_dim).
    _doc_ids : list[str]
        Ordered list of document IDs matching embedding rows.
    """

    def __init__(
        self,
        top_n: int = 1000,
        model_name: str = "intfloat/e5-large-v2",
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> None:
        """Initialize E5 retriever.

        Parameters
        ----------
        top_n : int, optional
            Number of documents to retrieve. Default 1000.
        model_name : str, optional
            HuggingFace model name. Default 'intfloat/e5-large-v2'.
        batch_size : int, optional
            Encoding batch size. Default 32.
        device : str, optional
            Device ('cuda' or 'cpu'). Auto-detected if None.
        """
        super().__init__(name="e5", top_n=top_n)
        self.model_name: str = model_name
        self.batch_size: int = batch_size
        self.device: str = device or get_device()

        self._model = None  # Type: SentenceTransformer (imported lazily)
        self._doc_embeddings: Optional[np.ndarray] = None
        self._doc_ids: list[str] = []

    def _load_model(self):
        """Lazy load the model.

        Returns
        -------
        SentenceTransformer
            The loaded model.
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for E5Retriever. "
                    "Install with: pip install sentence-transformers"
                ) from e

            logger.info(f"Loading E5 model: {self.model_name} on {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _prepare_passage(self, doc: Document) -> str:
        """Prepare passage text with E5 prefix.

        Parameters
        ----------
        doc : Document
            The document.

        Returns
        -------
        str
            Prefixed passage text.
        """
        text: str = f"{doc.title} {doc.text}".strip()
        return f"passage: {text}"

    def _prepare_query(self, query: Query) -> str:
        """Prepare query text with E5 prefix.

        Parameters
        ----------
        query : Query
            The query.

        Returns
        -------
        str
            Prefixed query text.
        """
        return f"query: {query.text}"

    def index(self, corpus: dict[str, Document], show_progress: bool = True) -> None:
        """Build embedding index from corpus.

        Parameters
        ----------
        corpus : dict[str, Document]
            Document corpus.
        show_progress : bool, optional
            Whether to show progress bar. Default True.
        """
        model = self._load_model()

        logger.info(f"Encoding {len(corpus)} documents with E5")

        self._doc_ids = list(corpus.keys())
        passages: list[str] = [
            self._prepare_passage(corpus[doc_id]) for doc_id in self._doc_ids
        ]

        # Encode in batches with progress bar
        self._doc_embeddings = model.encode(
            passages,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # For cosine similarity
            convert_to_numpy=True,
        )

        self._is_indexed = True
        logger.info(f"E5 index built: {self._doc_embeddings.shape}")

    def retrieve(self, query: Query) -> list[RankingEntry]:
        """Retrieve top-N documents for a query.

        Parameters
        ----------
        query : Query
            The query.

        Returns
        -------
        list[RankingEntry]
            Ranked documents.

        Raises
        ------
        RuntimeError
            If index not built.
        """
        if not self._is_indexed or self._doc_embeddings is None:
            raise RuntimeError("Must call index() before retrieve()")

        model = self._load_model()

        # Encode query
        query_text: str = self._prepare_query(query)
        query_embedding: np.ndarray = model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        # Compute cosine similarities (dot product since normalized)
        scores: np.ndarray = self._doc_embeddings @ query_embedding

        # Get top-N indices
        top_indices: np.ndarray = np.argsort(scores)[::-1][: self.top_n]

        # Build results
        results: list[RankingEntry] = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                RankingEntry(
                    query_id=query.query_id,
                    doc_id=self._doc_ids[idx],
                    rank=rank,
                    score=float(scores[idx]),
                )
            )

        return results

    def retrieve_batch(
        self,
        queries: list[Query],
        show_progress: bool = True,
    ) -> list[RankingEntry]:
        """Batch retrieve for multiple queries (optimized).

        Parameters
        ----------
        queries : list[Query]
            List of queries.
        show_progress : bool, optional
            Whether to show progress. Default True.

        Returns
        -------
        list[RankingEntry]
            All ranking entries.
        """
        if not self._is_indexed or self._doc_embeddings is None:
            raise RuntimeError("Must call index() before retrieve()")

        model = self._load_model()

        # Encode all queries at once
        query_texts: list[str] = [self._prepare_query(q) for q in queries]
        query_embeddings: np.ndarray = model.encode(
            query_texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Compute all similarities at once: (num_queries, num_docs)
        all_scores: np.ndarray = query_embeddings @ self._doc_embeddings.T

        # Build results for each query
        results: list[RankingEntry] = []
        for i, query in enumerate(queries):
            scores: np.ndarray = all_scores[i]
            top_indices: np.ndarray = np.argsort(scores)[::-1][: self.top_n]

            for rank, idx in enumerate(top_indices, start=1):
                results.append(
                    RankingEntry(
                        query_id=query.query_id,
                        doc_id=self._doc_ids[idx],
                        rank=rank,
                        score=float(scores[idx]),
                    )
                )

        return results

    def clear_index(self) -> None:
        """Clear embeddings and free memory."""
        self._doc_embeddings = None
        self._doc_ids = []
        self._is_indexed = False

        # Clear model from GPU if applicable
        if self._model is not None and self.device == "cuda":
            try:
                import torch

                del self._model
                self._model = None
                torch.cuda.empty_cache()
            except ImportError:
                self._model = None
