"""SPLADE retriever implementation using transformers.

This module implements learned sparse retrieval using SPLADE++.
"""

import logging
from typing import Optional

import numpy as np

from src.data.models import Document, Query, RankingEntry
from src.retrieval.base import Retriever
from src.retrieval.utils import get_device

logger: logging.Logger = logging.getLogger(__name__)


class SpladeRetriever(Retriever):
    """SPLADE++ learned sparse retriever.

    Attributes
    ----------
    model_name : str
        HuggingFace model identifier.
    batch_size : int
        Batch size for encoding.
    max_length : int
        Maximum token length.
    device : str
        Device to run model on.
    _model : AutoModelForMaskedLM or None
        The SPLADE model.
    _tokenizer : AutoTokenizer or None
        The tokenizer.
    _doc_sparse : scipy.sparse.csr_matrix or None
        Sparse document representations.
    _doc_ids : list[str]
        Ordered document IDs.
    """

    def __init__(
        self,
        top_n: int = 1000,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        batch_size: int = 16,
        max_length: int = 256,
        device: Optional[str] = None,
    ) -> None:
        """Initialize SPLADE retriever.

        Parameters
        ----------
        top_n : int, optional
            Number of documents to retrieve. Default 1000.
        model_name : str, optional
            HuggingFace model name.
        batch_size : int, optional
            Encoding batch size. Default 16 (SPLADE is memory-intensive).
        max_length : int, optional
            Maximum token length. Default 256.
        device : str, optional
            Device. Auto-detected if None.
        """
        super().__init__(name="splade", top_n=top_n)
        self.model_name: str = model_name
        self.batch_size: int = batch_size
        self.max_length: int = max_length
        self.device: str = device or get_device()

        self._model = None  # Type: AutoModelForMaskedLM (imported lazily)
        self._tokenizer = None  # Type: AutoTokenizer (imported lazily)
        self._doc_sparse = None  # Type: scipy.sparse.csr_matrix
        self._doc_ids: list[str] = []

    def _load_model(self):
        """Lazy load model and tokenizer.

        Returns
        -------
        tuple
            (model, tokenizer) tuple.
        """
        if self._model is None or self._tokenizer is None:
            try:
                import torch
                from transformers import AutoModelForMaskedLM, AutoTokenizer
            except ImportError as e:
                raise ImportError(
                    "transformers and torch are required for SpladeRetriever. "
                    "Install with: pip install transformers torch"
                ) from e

            logger.info(f"Loading SPLADE model: {self.model_name} on {self.device}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()

        return self._model, self._tokenizer

    def _encode_sparse(self, texts: list[str], show_progress: bool = True):
        """Encode texts to sparse SPLADE vectors.

        Parameters
        ----------
        texts : list[str]
            Texts to encode.
        show_progress : bool
            Whether to show progress bar.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse matrix of shape (num_texts, vocab_size).
        """
        import scipy.sparse as sp
        import torch
        from tqdm import tqdm

        model, tokenizer = self._load_model()
        vocab_size: int = tokenizer.vocab_size

        all_sparse: list[np.ndarray] = []

        # Process in batches
        num_batches: int = (len(texts) + self.batch_size - 1) // self.batch_size
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="SPLADE encoding")

        with torch.no_grad():
            for start_idx in iterator:
                batch_texts: list[str] = texts[start_idx : start_idx + self.batch_size]

                # Tokenize
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Forward pass
                outputs = model(**inputs)
                logits = outputs.logits  # (batch, seq_len, vocab_size)

                # SPLADE aggregation: log(1 + ReLU(max_over_tokens))
                # Apply ReLU and take max over sequence dimension
                relu_log = torch.log1p(torch.relu(logits))
                # Mask padding tokens
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                relu_log = relu_log * attention_mask
                # Max pooling over sequence
                sparse_vecs = torch.max(relu_log, dim=1).values  # (batch, vocab)

                all_sparse.append(sparse_vecs.cpu().numpy())

        # Stack all batches
        dense_matrix: np.ndarray = np.vstack(all_sparse)

        # Convert to sparse (most values are zero or near-zero)
        return sp.csr_matrix(dense_matrix)

    def index(self, corpus: dict[str, Document], show_progress: bool = True) -> None:
        """Build SPLADE index from corpus.

        Parameters
        ----------
        corpus : dict[str, Document]
            Document corpus.
        show_progress : bool, optional
            Whether to show progress bar. Default True.
        """
        logger.info(f"Building SPLADE index for {len(corpus)} documents")

        self._doc_ids = list(corpus.keys())
        texts: list[str] = [
            f"{corpus[doc_id].title} {corpus[doc_id].text}".strip()
            for doc_id in self._doc_ids
        ]

        self._doc_sparse = self._encode_sparse(texts, show_progress=show_progress)
        self._is_indexed = True
        logger.info(f"SPLADE index built: {self._doc_sparse.shape}")

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
        if not self._is_indexed or self._doc_sparse is None:
            raise RuntimeError("Must call index() before retrieve()")

        # Encode query
        query_sparse = self._encode_sparse([query.text], show_progress=False)

        # Compute dot product scores
        scores: np.ndarray = (self._doc_sparse @ query_sparse.T).toarray().flatten()

        # Get top-N
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
        """Batch retrieve for multiple queries.

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
        if not self._is_indexed or self._doc_sparse is None:
            raise RuntimeError("Must call index() before retrieve()")

        # Encode all queries
        query_texts: list[str] = [q.text for q in queries]
        query_sparse = self._encode_sparse(query_texts, show_progress=show_progress)

        # Compute all scores: (num_queries, num_docs)
        all_scores: np.ndarray = (query_sparse @ self._doc_sparse.T).toarray()

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
        """Clear index and free memory."""
        self._doc_sparse = None
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
