"""BM25 retriever implementation using rank_bm25.

This module implements sparse lexical retrieval using the BM25 algorithm.
"""

import logging
from typing import Optional

from tqdm import tqdm

from src.data.models import Document, Query, RankingEntry
from src.retrieval.base import Retriever

logger: logging.Logger = logging.getLogger(__name__)


class BM25Retriever(Retriever):
    """BM25 sparse retriever using rank_bm25 library.

    Attributes
    ----------
    _bm25 : BM25Okapi or None
        The BM25 index.
    _doc_ids : list[str]
        Ordered list of document IDs (matches BM25 index order).
    """

    def __init__(self, top_n: int = 1000) -> None:
        """Initialize BM25 retriever.

        Parameters
        ----------
        top_n : int, optional
            Number of documents to retrieve per query. Default 1000.
        """
        super().__init__(name="bm25", top_n=top_n)
        self._bm25 = None  # Type: BM25Okapi (imported lazily)
        self._doc_ids: list[str] = []

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25.

        Parameters
        ----------
        text : str
            Text to tokenize.

        Returns
        -------
        list[str]
            List of lowercase tokens.
        """
        return text.lower().split()

    def index(self, corpus: dict[str, Document], show_progress: bool = True) -> None:
        """Build BM25 index from corpus.

        Parameters
        ----------
        corpus : dict[str, Document]
            Document corpus.
        show_progress : bool, optional
            Whether to show progress bar. Default True.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError(
                "rank_bm25 is required for BM25Retriever. "
                "Install with: pip install rank-bm25"
            ) from e

        logger.info(f"Building BM25 index for {len(corpus)} documents")

        self._doc_ids = list(corpus.keys())

        # Tokenize all documents
        tokenized_corpus: list[list[str]] = []
        iterator = tqdm(
            self._doc_ids,
            desc="Tokenizing corpus",
            disable=not show_progress,
        )

        for doc_id in iterator:
            doc: Document = corpus[doc_id]
            # Combine title and text for indexing
            full_text: str = f"{doc.title} {doc.text}".strip()
            tokenized_corpus.append(self._tokenize(full_text))

        # Build BM25 index
        logger.info("Building BM25 index...")
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._is_indexed = True
        logger.info("BM25 index built successfully")

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
        if not self._is_indexed or self._bm25 is None:
            raise RuntimeError("Must call index() before retrieve()")

        tokenized_query: list[str] = self._tokenize(query.text)
        scores = self._bm25.get_scores(tokenized_query)

        # Convert to list if numpy array
        if hasattr(scores, "tolist"):
            scores_list: list[float] = scores.tolist()
        else:
            scores_list = list(scores)

        # Create (doc_id, score) pairs and sort by score descending
        doc_scores: list[tuple[str, float]] = list(zip(self._doc_ids, scores_list))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Take top_n and create RankingEntry objects
        results: list[RankingEntry] = []
        for rank, (doc_id, score) in enumerate(doc_scores[: self.top_n], start=1):
            results.append(
                RankingEntry(
                    query_id=query.query_id,
                    doc_id=doc_id,
                    rank=rank,
                    score=float(score),
                )
            )

        return results

    def clear_index(self) -> None:
        """Clear the BM25 index."""
        self._bm25 = None
        self._doc_ids = []
        self._is_indexed = False
