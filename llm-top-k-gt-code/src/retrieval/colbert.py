"""ColBERT retriever implementation using RAGatouille.

This module implements late-interaction retrieval using ColBERTv2.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.data.models import Document, Query, RankingEntry
from src.retrieval.base import Retriever

logger: logging.Logger = logging.getLogger(__name__)


class ColBERTRetriever(Retriever):
    """ColBERTv2 late-interaction retriever using RAGatouille.

    Attributes
    ----------
    model_name : str
        ColBERT model identifier.
    index_name : str
        Name for the index (used in path).
    index_root : Path
        Root directory for storing indexes.
    _rag : RAGPretrainedModel or None
        The RAGatouille model.
    _doc_ids : list[str]
        Ordered document IDs.
    _doc_id_to_idx : dict[str, int]
        Mapping from doc_id to index position.
    """

    def __init__(
        self,
        top_n: int = 1000,
        model_name: str = "colbert-ir/colbertv2.0",
        index_name: str = "corpus_index",
        index_root: Optional[Path] = None,
    ) -> None:
        """Initialize ColBERT retriever.

        Parameters
        ----------
        top_n : int, optional
            Number of documents to retrieve. Default 1000.
        model_name : str, optional
            ColBERT model name. Default 'colbert-ir/colbertv2.0'.
        index_name : str, optional
            Name for the index. Default 'corpus_index'.
        index_root : Path, optional
            Root directory for indexes. Default '.ragatouille/colbert/indexes'.
        """
        super().__init__(name="colbert", top_n=top_n)
        self.model_name: str = model_name
        self.index_name: str = index_name
        self.index_root: Path = index_root or Path(".ragatouille/colbert/indexes")

        self._rag = None  # Type: RAGPretrainedModel (imported lazily)
        self._doc_ids: list[str] = []
        self._doc_id_to_idx: dict[str, int] = {}

    def _load_model(self):
        """Lazy load the RAGatouille model.

        Returns
        -------
        RAGPretrainedModel
            The loaded model.
        """
        if self._rag is None:
            try:
                from ragatouille import RAGPretrainedModel
            except ImportError as e:
                raise ImportError(
                    "RAGatouille is required for ColBERTRetriever. "
                    "Install with: pip install ragatouille"
                ) from e

            logger.info(f"Loading ColBERT model: {self.model_name}")
            self._rag = RAGPretrainedModel.from_pretrained(self.model_name)
        return self._rag

    def index(self, corpus: dict[str, Document], show_progress: bool = True) -> None:
        """Build ColBERT index from corpus.

        Parameters
        ----------
        corpus : dict[str, Document]
            Document corpus.
        show_progress : bool, optional
            Whether to show progress (ColBERT has its own progress display).
        """
        rag = self._load_model()

        logger.info(f"Building ColBERT index for {len(corpus)} documents")

        self._doc_ids = list(corpus.keys())
        self._doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self._doc_ids)}

        # Prepare documents for indexing
        # ColBERT expects a list of document texts
        documents: list[str] = [
            f"{corpus[doc_id].title} {corpus[doc_id].text}".strip()
            for doc_id in self._doc_ids
        ]

        # Create document metadata with IDs
        document_ids: list[str] = self._doc_ids.copy()

        # Build the index
        # RAGatouille will create the index at index_root/index_name/
        index_path: str = rag.index(
            collection=documents,
            document_ids=document_ids,
            index_name=self.index_name,
            max_document_length=256,
            split_documents=False,  # Don't split documents
        )

        logger.info(f"ColBERT index built at: {index_path}")
        self._is_indexed = True

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
        if not self._is_indexed:
            raise RuntimeError("Must call index() before retrieve()")

        rag = self._load_model()

        # Search returns list of dicts with 'content', 'document_id', 'score', 'rank'
        results_raw = rag.search(
            query=query.text,
            k=self.top_n,
        )

        results: list[RankingEntry] = []
        for rank_idx, item in enumerate(results_raw, start=1):
            doc_id: str = item.get("document_id", "")
            score: float = float(item.get("score", 0.0))
            # Use our own rank counter to ensure 1-indexed ranking
            rank: int = rank_idx

            results.append(
                RankingEntry(
                    query_id=query.query_id,
                    doc_id=doc_id,
                    rank=rank,
                    score=score,
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
        if not self._is_indexed:
            raise RuntimeError("Must call index() before retrieve_batch()")

        all_results: list[RankingEntry] = []

        iterator = tqdm(queries, desc="ColBERT retrieval") if show_progress else queries
        for query in iterator:
            results = self.retrieve(query)
            all_results.extend(results)

        return all_results

    def clear_index(self) -> None:
        """Clear the ColBERT index."""
        self._doc_ids = []
        self._doc_id_to_idx = {}
        self._is_indexed = False

        # Optionally remove index files
        index_path: Path = self.index_root / self.index_name
        if index_path.exists():
            logger.info(f"Removing ColBERT index at: {index_path}")
            shutil.rmtree(index_path)

        self._rag = None
