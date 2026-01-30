"""Abstract base class for retrieval models.

This module defines the Retriever interface that all IR models must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional

from tqdm import tqdm

from src.data.beir_loader import BeirDataset
from src.data.models import Document, Query, RankingEntry


class Retriever(ABC):
    """Abstract base class for document retrieval models.

    All retriever implementations must inherit from this class and implement
    the abstract methods for indexing and retrieval.

    Attributes
    ----------
    name : str
        Unique identifier for this retriever (used in output filenames).
    top_n : int
        Number of documents to retrieve per query.
    _is_indexed : bool
        Whether the corpus has been indexed.
    """

    def __init__(self, name: str, top_n: int = 1000) -> None:
        """Initialize the retriever.

        Parameters
        ----------
        name : str
            Unique identifier for this retriever.
        top_n : int, optional
            Number of documents to retrieve per query. Default 1000.
        """
        self.name: str = name
        self.top_n: int = top_n
        self._is_indexed: bool = False

    @abstractmethod
    def index(self, corpus: dict[str, Document], show_progress: bool = True) -> None:
        """Build an index from the document corpus.

        Parameters
        ----------
        corpus : dict[str, Document]
            Mapping from doc_id to Document objects.
        show_progress : bool, optional
            Whether to show a progress bar. Default True.

        Notes
        -----
        After calling this method, `_is_indexed` should be set to True.
        """
        pass

    @abstractmethod
    def retrieve(self, query: Query) -> list[RankingEntry]:
        """Retrieve top-N documents for a single query.

        Parameters
        ----------
        query : Query
            The query to retrieve documents for.

        Returns
        -------
        list[RankingEntry]
            List of RankingEntry objects sorted by rank (1-indexed).

        Raises
        ------
        RuntimeError
            If called before indexing.
        """
        pass

    def retrieve_batch(
        self,
        queries: list[Query],
        show_progress: bool = True,
    ) -> list[RankingEntry]:
        """Retrieve documents for multiple queries.

        Parameters
        ----------
        queries : list[Query]
            List of queries to retrieve documents for.
        show_progress : bool, optional
            Whether to show a progress bar. Default True.

        Returns
        -------
        list[RankingEntry]
            List of RankingEntry objects for all queries.

        Notes
        -----
        Default implementation calls retrieve() for each query.
        Subclasses may override for more efficient batch processing.
        """
        results: list[RankingEntry] = []
        iterator = tqdm(
            queries,
            desc=f"Retrieving [{self.name}]",
            disable=not show_progress,
        )

        for query in iterator:
            results.extend(self.retrieve(query))

        return results

    def run(
        self,
        dataset: BeirDataset,
        query_ids: Optional[list[str]] = None,
        show_progress: bool = True,
    ) -> list[RankingEntry]:
        """Run full retrieval pipeline on a dataset.

        Parameters
        ----------
        dataset : BeirDataset
            The dataset to run retrieval on.
        query_ids : list[str], optional
            Specific query IDs to process. If None, process all queries.
        show_progress : bool, optional
            Whether to show progress bars. Default True.

        Returns
        -------
        list[RankingEntry]
            All ranking entries for the specified queries.
        """
        # Index corpus if not already indexed
        if not self._is_indexed:
            self.index(dataset.corpus, show_progress=show_progress)

        # Select queries
        if query_ids is None:
            query_ids = dataset.get_query_ids()

        queries: list[Query] = [dataset.queries[qid] for qid in query_ids]

        # Run retrieval
        return self.retrieve_batch(queries, show_progress=show_progress)

    @abstractmethod
    def clear_index(self) -> None:
        """Clear the index and free memory.

        Notes
        -----
        After calling this method, `_is_indexed` should be set to False.
        """
        pass

    @property
    def is_indexed(self) -> bool:
        """Check if the corpus has been indexed.

        Returns
        -------
        bool
            True if the corpus has been indexed, False otherwise.
        """
        return self._is_indexed
