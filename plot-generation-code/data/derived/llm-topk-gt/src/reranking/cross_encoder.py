"""Cross-encoder reranker implementations using sentence-transformers.

This module implements reranking using cross-encoder models that jointly
encode query-document pairs to produce relevance scores.
"""

import logging
from typing import Optional

from src.data.models import Document, Query
from src.reranking.base import Reranker

logger: logging.Logger = logging.getLogger(__name__)


def get_device(preferred: Optional[str] = None) -> str:
    """Get the compute device to use.

    Parameters
    ----------
    preferred : str, optional
        Preferred device ('cuda', 'cpu', or 'mps'). Auto-detect if None.

    Returns
    -------
    str
        Device string for PyTorch.
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed, using CPU")
        return "cpu"

    if preferred is not None:
        return preferred

    if torch.cuda.is_available():
        device: str = "cuda"
        device_name: str = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA device: {device_name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple MPS device")
    else:
        device = "cpu"
        logger.info("Using CPU device")

    return device


class CrossEncoderReranker(Reranker):
    """Generic cross-encoder reranker using sentence-transformers.

    This class wraps sentence-transformers' CrossEncoder to provide
    a unified interface for various cross-encoder models.

    Attributes
    ----------
    model_name : str
        HuggingFace model identifier.
    batch_size : int
        Batch size for scoring.
    device : str
        Device to run model on ('cuda', 'cpu', or 'mps').
    max_length : int
        Maximum input sequence length.
    _model : CrossEncoder or None
        The cross-encoder model (loaded lazily).
    """

    def __init__(
        self,
        name: str,
        model_name: str,
        batch_size: int = 32,
        device: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        """Initialize cross-encoder reranker.

        Parameters
        ----------
        name : str
            Unique identifier for this reranker.
        model_name : str
            HuggingFace model name for the cross-encoder.
        batch_size : int, optional
            Batch size for scoring. Default 32.
        device : str, optional
            Device ('cuda', 'cpu', 'mps'). Auto-detected if None.
        max_length : int, optional
            Maximum input sequence length. Default 512.
        """
        super().__init__(name=name)
        self.model_name: str = model_name
        self.batch_size: int = batch_size
        self.device: str = device or get_device()
        self.max_length: int = max_length
        self._model = None  # Type: CrossEncoder (imported lazily)

    def _load_model(self):
        """Lazy load the cross-encoder model.

        Returns
        -------
        CrossEncoder
            The loaded model.
        """
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for CrossEncoderReranker. "
                    "Install with: pip install sentence-transformers"
                ) from e

            logger.info(f"Loading cross-encoder: {self.model_name} on {self.device}")
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
            )

        return self._model

    def _prepare_text(self, doc: Document) -> str:
        """Prepare document text for scoring.

        Parameters
        ----------
        doc : Document
            The document.

        Returns
        -------
        str
            Combined title and text.
        """
        if doc.title:
            return f"{doc.title} {doc.text}".strip()
        return doc.text

    def score(
        self,
        query: Query,
        documents: list[Document],
        show_progress: bool = False,
    ) -> list[float]:
        """Score documents for a query using the cross-encoder.

        Parameters
        ----------
        query : Query
            The query.
        documents : list[Document]
            Documents to score.
        show_progress : bool, optional
            Whether to show progress. Default False.

        Returns
        -------
        list[float]
            Relevance scores for each document.
        """
        model = self._load_model()

        # Create query-document pairs
        pairs: list[list[str]] = [
            [query.text, self._prepare_text(doc)] for doc in documents
        ]

        # Score pairs
        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
        )

        # Convert to list if numpy array
        if hasattr(scores, "tolist"):
            return scores.tolist()
        return list(scores)

    def score_pairs(
        self,
        pairs: list[tuple[str, str]],
        show_progress: bool = True,
    ) -> list[float]:
        """Score (query_text, document_text) pairs.

        Parameters
        ----------
        pairs : list[tuple[str, str]]
            List of (query_text, document_text) tuples.
        show_progress : bool, optional
            Whether to show progress. Default True.

        Returns
        -------
        list[float]
            Relevance scores for each pair.
        """
        model = self._load_model()

        # Convert tuples to lists for CrossEncoder
        pair_lists: list[list[str]] = [list(p) for p in pairs]

        scores = model.predict(
            pair_lists,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
        )

        if hasattr(scores, "tolist"):
            return scores.tolist()
        return list(scores)

    def clear(self) -> None:
        """Clear the model and free memory."""
        if self._model is not None and self.device == "cuda":
            try:
                import torch

                del self._model
                self._model = None
                torch.cuda.empty_cache()
            except ImportError:
                self._model = None
        else:
            self._model = None


class MiniLML6Reranker(CrossEncoderReranker):
    """MS MARCO MiniLM L6 cross-encoder reranker.

    A small but effective cross-encoder model trained on MS MARCO.
    Fast inference with good quality.
    """

    def __init__(
        self,
        batch_size: int = 64,
        device: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        """Initialize MiniLM L6 reranker.

        Parameters
        ----------
        batch_size : int, optional
            Batch size for scoring. Default 64 (small model allows larger batches).
        device : str, optional
            Device. Auto-detected if None.
        max_length : int, optional
            Maximum sequence length. Default 512.
        """
        super().__init__(
            name="minilm_l6",
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            batch_size=batch_size,
            device=device,
            max_length=max_length,
        )


class MiniLML12Reranker(CrossEncoderReranker):
    """MS MARCO MiniLM L12 cross-encoder reranker.

    A medium-sized cross-encoder model trained on MS MARCO.
    Better quality than L6, slightly slower.
    """

    def __init__(
        self,
        batch_size: int = 32,
        device: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        """Initialize MiniLM L12 reranker.

        Parameters
        ----------
        batch_size : int, optional
            Batch size for scoring. Default 32.
        device : str, optional
            Device. Auto-detected if None.
        max_length : int, optional
            Maximum sequence length. Default 512.
        """
        super().__init__(
            name="minilm_l12",
            model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
            batch_size=batch_size,
            device=device,
            max_length=max_length,
        )


class BGEReranker(CrossEncoderReranker):
    """BGE Reranker Large cross-encoder.

    A large cross-encoder from BAAI with excellent reranking performance.
    Slower but more accurate than MiniLM variants.
    """

    def __init__(
        self,
        batch_size: int = 16,
        device: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        """Initialize BGE Reranker.

        Parameters
        ----------
        batch_size : int, optional
            Batch size for scoring. Default 16 (larger model).
        device : str, optional
            Device. Auto-detected if None.
        max_length : int, optional
            Maximum sequence length. Default 512.
        """
        super().__init__(
            name="bge_reranker",
            model_name="BAAI/bge-reranker-large",
            batch_size=batch_size,
            device=device,
            max_length=max_length,
        )


class MMarcoMiniLMReranker(CrossEncoderReranker):
    """Multilingual mMARCO MiniLM cross-encoder reranker.

    A multilingual cross-encoder trained on mMARCO dataset.
    Supports multiple languages while maintaining good English performance.
    """

    def __init__(
        self,
        batch_size: int = 32,
        device: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        """Initialize mMARCO MiniLM reranker.

        Parameters
        ----------
        batch_size : int, optional
            Batch size for scoring. Default 32.
        device : str, optional
            Device. Auto-detected if None.
        max_length : int, optional
            Maximum sequence length. Default 512.
        """
        super().__init__(
            name="mmarco_minilm",
            model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            batch_size=batch_size,
            device=device,
            max_length=max_length,
        )
