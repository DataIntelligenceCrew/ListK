"""BGE dense retriever implementation using sentence-transformers.

This module implements dense retrieval using the BGE-large-en-v1.5 model.
"""

import logging
from typing import Optional

import numpy as np

from src.data.models import Document, Query, RankingEntry
from src.retrieval.base import Retriever
from src.retrieval.utils import get_device

logger: logging.Logger = logging.getLogger(__name__)

# BGE query instruction for retrieval tasks
BGE_QUERY_INSTRUCTION: str = "Represent this sentence for searching relevant passages: "


class BGERetriever(Retriever):
    """BGE-large-en-v1.5 dense retriever.

    Attributes
    ----------
    model_name : str
        HuggingFace model identifier.
    batch_size : int
        Batch size for encoding.
    device : str
        Device to run model on.
    use_instruction : bool
        Whether to use query instruction prefix.
    _model : SentenceTransformer or None
        The embedding model.
    _doc_embeddings : np.ndarray or None
        Cached document embeddings.
    _doc_ids : list[str]
        Ordered document IDs.
    """

    def __init__(
        self,
        top_n: int = 1000,
        model_name: str = "BAAI/bge-large-en-v1.5",
        batch_size: int = 32,
        device: Optional[str] = None,
        use_instruction: bool = True,
    ) -> None:
        """Initialize BGE retriever.

        Parameters
        ----------
        top_n : int, optional
            Number of documents to retrieve. Default 1000.
        model_name : str, optional
            HuggingFace model name. Default 'BAAI/bge-large-en-v1.5'.
        batch_size : int, optional
            Encoding batch size. Default 32.
        device : str, optional
            Device. Auto-detected if None.
        use_instruction : bool, optional
            Whether to use query instruction. Default True.
        """
        super().__init__(name="bge", top_n=top_n)
        self.model_name: str = model_name
        self.batch_size: int = batch_size
        self.device: str = device or get_device()
        self.use_instruction: bool = use_instruction

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
                    "sentence-transformers is required for BGERetriever. "
                    "Install with: pip install sentence-transformers"
                ) from e

            logger.info(f"Loading BGE model: {self.model_name} on {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _prepare_passage(self, doc: Document) -> str:
        """Prepare passage text (no prefix for BGE passages).

        Parameters
        ----------
        doc : Document
            The document.

        Returns
        -------
        str
            Passage text.
        """
        return f"{doc.title} {doc.text}".strip()

    def _prepare_query(self, query: Query) -> str:
        """Prepare query text with optional instruction.

        Parameters
        ----------
        query : Query
            The query.

        Returns
        -------
        str
            Query text with optional instruction prefix.
        """
        if self.use_instruction:
            return f"{BGE_QUERY_INSTRUCTION}{query.text}"
        return query.text

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

        logger.info(f"Encoding {len(corpus)} documents with BGE")

        self._doc_ids = list(corpus.keys())
        passages: list[str] = [
            self._prepare_passage(corpus[doc_id]) for doc_id in self._doc_ids
        ]

        self._doc_embeddings = model.encode(
            passages,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        self._is_indexed = True
        logger.info(f"BGE index built: {self._doc_embeddings.shape}")

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

        query_text: str = self._prepare_query(query)
        query_embedding: np.ndarray = model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        scores: np.ndarray = self._doc_embeddings @ query_embedding
        top_indices: np.ndarray = np.argsort(scores)[::-1][: self.top_n]

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

        query_texts: list[str] = [self._prepare_query(q) for q in queries]
        query_embeddings: np.ndarray = model.encode(
            query_texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        all_scores: np.ndarray = query_embeddings @ self._doc_embeddings.T

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

        if self._model is not None and self.device == "cuda":
            try:
                import torch

                del self._model
                self._model = None
                torch.cuda.empty_cache()
            except ImportError:
                self._model = None
