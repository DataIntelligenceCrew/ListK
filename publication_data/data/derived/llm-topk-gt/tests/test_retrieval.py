"""Unit tests for the retrieval module.

This module tests the retrieval components including the base class,
individual retrievers, and the registry/factory functions.
"""

import pytest

from src.data.models import Document, Query, RankingEntry
from src.retrieval import (
    RETRIEVER_REGISTRY,
    Retriever,
    get_retriever,
)
from src.retrieval.base import Retriever as BaseRetriever
from src.retrieval.bm25 import BM25Retriever


# Fixtures for test data


@pytest.fixture
def sample_corpus() -> dict[str, Document]:
    """Create a small sample corpus for testing.

    Returns
    -------
    dict[str, Document]
        A corpus with 5 documents about different topics.
    """
    return {
        "doc1": Document(
            doc_id="doc1",
            title="Machine Learning Basics",
            text="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        ),
        "doc2": Document(
            doc_id="doc2",
            title="Deep Learning Neural Networks",
            text="Deep learning uses neural networks with multiple layers to model complex patterns in data.",
        ),
        "doc3": Document(
            doc_id="doc3",
            title="Natural Language Processing",
            text="NLP focuses on the interaction between computers and human language, enabling text analysis.",
        ),
        "doc4": Document(
            doc_id="doc4",
            title="Computer Vision",
            text="Computer vision enables machines to interpret and understand visual information from the world.",
        ),
        "doc5": Document(
            doc_id="doc5",
            title="Reinforcement Learning",
            text="Reinforcement learning trains agents to make decisions by maximizing cumulative rewards.",
        ),
    }


@pytest.fixture
def sample_queries() -> list[Query]:
    """Create sample queries for testing.

    Returns
    -------
    list[Query]
        A list of test queries.
    """
    return [
        Query(query_id="q1", text="What is machine learning?"),
        Query(query_id="q2", text="How do neural networks work?"),
        Query(query_id="q3", text="Natural language processing applications"),
    ]


# Test Base Class


class TestRetrieverBaseClass:
    """Tests for the abstract Retriever base class."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that Retriever cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseRetriever(name="test", top_n=10)  # type: ignore

    def test_subclass_must_implement_abstract_methods(self) -> None:
        """Test that subclasses must implement abstract methods."""

        # This should fail because abstract methods are not implemented
        class IncompleteRetriever(BaseRetriever):
            pass

        with pytest.raises(TypeError):
            IncompleteRetriever(name="incomplete", top_n=10)  # type: ignore


# Test BM25 Retriever


class TestBM25Retriever:
    """Tests for the BM25 retriever."""

    def test_initialization(self) -> None:
        """Test BM25 retriever initialization."""
        retriever = BM25Retriever(top_n=100)

        assert retriever.name == "bm25"
        assert retriever.top_n == 100
        assert not retriever.is_indexed

    def test_index_builds_successfully(
        self, sample_corpus: dict[str, Document]
    ) -> None:
        """Test that indexing works correctly."""
        retriever = BM25Retriever(top_n=10)
        retriever.index(sample_corpus, show_progress=False)

        assert retriever.is_indexed
        assert len(retriever._doc_ids) == len(sample_corpus)

    def test_retrieve_before_index_raises_error(self) -> None:
        """Test that retrieve() raises error if called before index()."""
        retriever = BM25Retriever(top_n=10)
        query = Query(query_id="q1", text="test query")

        with pytest.raises(RuntimeError, match="Must call index"):
            retriever.retrieve(query)

    def test_retrieve_returns_ranking_entries(
        self, sample_corpus: dict[str, Document]
    ) -> None:
        """Test that retrieve() returns properly formatted RankingEntry objects."""
        retriever = BM25Retriever(top_n=10)
        retriever.index(sample_corpus, show_progress=False)

        query = Query(query_id="q1", text="machine learning")
        results = retriever.retrieve(query)

        assert len(results) == len(sample_corpus)  # top_n >= corpus size
        assert all(isinstance(r, RankingEntry) for r in results)
        assert all(r.query_id == "q1" for r in results)

    def test_retrieve_ranks_are_one_indexed(
        self, sample_corpus: dict[str, Document]
    ) -> None:
        """Test that ranks start at 1, not 0."""
        retriever = BM25Retriever(top_n=10)
        retriever.index(sample_corpus, show_progress=False)

        query = Query(query_id="q1", text="machine learning")
        results = retriever.retrieve(query)

        ranks = [r.rank for r in results]
        assert min(ranks) == 1
        assert max(ranks) == len(sample_corpus)
        assert sorted(ranks) == list(range(1, len(sample_corpus) + 1))

    def test_retrieve_scores_are_descending(
        self, sample_corpus: dict[str, Document]
    ) -> None:
        """Test that results are sorted by score descending."""
        retriever = BM25Retriever(top_n=10)
        retriever.index(sample_corpus, show_progress=False)

        query = Query(query_id="q1", text="machine learning")
        results = retriever.retrieve(query)

        # Sort by rank to ensure order
        results_by_rank = sorted(results, key=lambda r: r.rank)
        scores = [r.score for r in results_by_rank]

        # Scores should be in descending order
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_top_n_limits_results(
        self, sample_corpus: dict[str, Document]
    ) -> None:
        """Test that top_n limits the number of results."""
        retriever = BM25Retriever(top_n=2)
        retriever.index(sample_corpus, show_progress=False)

        query = Query(query_id="q1", text="machine learning")
        results = retriever.retrieve(query)

        assert len(results) == 2

    def test_retrieve_batch(
        self, sample_corpus: dict[str, Document], sample_queries: list[Query]
    ) -> None:
        """Test batch retrieval."""
        retriever = BM25Retriever(top_n=3)
        retriever.index(sample_corpus, show_progress=False)

        results = retriever.retrieve_batch(sample_queries, show_progress=False)

        # Should have 3 results per query
        assert len(results) == 3 * len(sample_queries)

        # Check all query IDs are present
        query_ids = set(r.query_id for r in results)
        expected_ids = set(q.query_id for q in sample_queries)
        assert query_ids == expected_ids

    def test_clear_index(self, sample_corpus: dict[str, Document]) -> None:
        """Test that clear_index() resets state."""
        retriever = BM25Retriever(top_n=10)
        retriever.index(sample_corpus, show_progress=False)

        assert retriever.is_indexed

        retriever.clear_index()

        assert not retriever.is_indexed
        assert len(retriever._doc_ids) == 0

    def test_tokenize_lowercases(self) -> None:
        """Test that tokenization lowercases text."""
        retriever = BM25Retriever(top_n=10)

        tokens = retriever._tokenize("Hello WORLD Test")
        assert tokens == ["hello", "world", "test"]

    def test_relevant_document_ranked_higher(
        self, sample_corpus: dict[str, Document]
    ) -> None:
        """Test that relevant documents are ranked higher than irrelevant ones."""
        retriever = BM25Retriever(top_n=10)
        retriever.index(sample_corpus, show_progress=False)

        # Query about machine learning should rank doc1 highly
        query = Query(query_id="q1", text="machine learning artificial intelligence")
        results = retriever.retrieve(query)

        # Find doc1's rank
        doc1_result = next(r for r in results if r.doc_id == "doc1")

        # doc1 should be in top 2 since it's about ML
        assert doc1_result.rank <= 2


# Test Registry and Factory


class TestRetrieverRegistry:
    """Tests for the retriever registry and factory function."""

    def test_registry_contains_all_retrievers(self) -> None:
        """Test that registry contains expected retrievers."""
        expected_retrievers = {"bm25", "e5", "bge", "splade", "colbert"}
        assert set(RETRIEVER_REGISTRY.keys()) == expected_retrievers

    def test_get_retriever_returns_correct_type(self) -> None:
        """Test that get_retriever returns the correct retriever type."""
        retriever = get_retriever("bm25", top_n=50)

        assert isinstance(retriever, BM25Retriever)
        assert retriever.top_n == 50

    def test_get_retriever_with_invalid_name_raises_error(self) -> None:
        """Test that get_retriever raises error for unknown retriever."""
        with pytest.raises(ValueError, match="Unknown retriever"):
            get_retriever("invalid_retriever")

    def test_get_retriever_passes_kwargs(self) -> None:
        """Test that kwargs are passed to retriever constructor."""
        retriever = get_retriever("bm25", top_n=123)
        assert retriever.top_n == 123


# Test Utilities


class TestRetrieverUtils:
    """Tests for retrieval utility functions."""

    def test_get_device_returns_string(self) -> None:
        """Test that get_device returns a valid device string."""
        from src.retrieval.utils import get_device

        device = get_device()
        assert device in {"cuda", "cpu", "mps"}

    def test_get_device_with_preferred(self) -> None:
        """Test that get_device respects preferred device."""
        from src.retrieval.utils import get_device

        device = get_device(preferred="cpu")
        assert device == "cpu"

    def test_get_gpu_memory_returns_float(self) -> None:
        """Test that get_gpu_memory_gb returns a float."""
        from src.retrieval.utils import get_gpu_memory_gb

        memory = get_gpu_memory_gb()
        assert isinstance(memory, float)
        assert memory >= 0


# Integration-style tests (still uses mock data but tests full flow)


class TestRetrieverIntegration:
    """Integration tests for the retrieval workflow."""

    def test_full_retrieval_workflow(
        self, sample_corpus: dict[str, Document], sample_queries: list[Query]
    ) -> None:
        """Test the complete retrieval workflow."""
        retriever = BM25Retriever(top_n=5)

        # Index
        assert not retriever.is_indexed
        retriever.index(sample_corpus, show_progress=False)
        assert retriever.is_indexed

        # Retrieve for all queries
        all_results: list[RankingEntry] = []
        for query in sample_queries:
            results = retriever.retrieve(query)
            all_results.extend(results)

        # Verify results structure
        assert len(all_results) == len(sample_queries) * 5

        # All doc_ids should be from corpus
        result_doc_ids = set(r.doc_id for r in all_results)
        corpus_doc_ids = set(sample_corpus.keys())
        assert result_doc_ids.issubset(corpus_doc_ids)

        # Cleanup
        retriever.clear_index()
        assert not retriever.is_indexed

    def test_multiple_retrievers_same_corpus(
        self, sample_corpus: dict[str, Document]
    ) -> None:
        """Test running multiple retrievers on same corpus."""
        query = Query(query_id="q1", text="machine learning")

        # Run BM25
        bm25 = get_retriever("bm25", top_n=3)
        bm25.index(sample_corpus, show_progress=False)
        bm25_results = bm25.retrieve(query)
        bm25.clear_index()

        # Verify BM25 results
        assert len(bm25_results) == 3
        assert all(r.query_id == "q1" for r in bm25_results)

        # Note: We don't test E5/BGE/SPLADE/ColBERT here because they require
        # downloading large models. Those should be tested in integration tests.
