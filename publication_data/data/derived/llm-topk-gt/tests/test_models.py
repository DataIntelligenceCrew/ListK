"""Unit tests for data models."""

import pytest
from datetime import datetime

from src.data.models import (
    Document,
    Query,
    Qrel,
    DatasetInfo,
    RankingEntry,
    PairwiseComparison,
    EloRating,
    EloState,
)


class TestDocument:
    """Tests for Document model."""

    def test_document_creation_minimal(self) -> None:
        """Test creating a document with minimal required fields."""
        doc = Document(doc_id="doc1", text="Sample text content")

        assert doc.doc_id == "doc1"
        assert doc.text == "Sample text content"
        assert doc.title == ""
        assert doc.metadata == {}

    def test_document_creation_full(self) -> None:
        """Test creating a document with all fields."""
        doc = Document(
            doc_id="doc1",
            title="Sample Title",
            text="Sample text content",
            metadata={"source": "test", "year": 2024},
        )

        assert doc.doc_id == "doc1"
        assert doc.title == "Sample Title"
        assert doc.text == "Sample text content"
        assert doc.metadata == {"source": "test", "year": 2024}

    def test_document_serialization(self) -> None:
        """Test document JSON serialization."""
        doc = Document(doc_id="doc1", title="Title", text="Text")
        data: dict = doc.model_dump()

        assert data["doc_id"] == "doc1"
        assert data["title"] == "Title"
        assert data["text"] == "Text"


class TestQuery:
    """Tests for Query model."""

    def test_query_creation(self) -> None:
        """Test creating a query."""
        query = Query(query_id="q1", text="What is machine learning?")

        assert query.query_id == "q1"
        assert query.text == "What is machine learning?"
        assert query.metadata == {}

    def test_query_with_metadata(self) -> None:
        """Test query with metadata."""
        query = Query(
            query_id="q1",
            text="Sample query",
            metadata={"category": "science"},
        )

        assert query.metadata == {"category": "science"}


class TestQrel:
    """Tests for Qrel model."""

    def test_qrel_creation(self) -> None:
        """Test creating a relevance judgment."""
        qrel = Qrel(query_id="q1", doc_id="doc1", relevance=2)

        assert qrel.query_id == "q1"
        assert qrel.doc_id == "doc1"
        assert qrel.relevance == 2


class TestDatasetInfo:
    """Tests for DatasetInfo model."""

    def test_dataset_info_creation(self) -> None:
        """Test creating dataset info."""
        info = DatasetInfo(
            name="scifact",
            num_documents=5000,
            num_queries=300,
            num_qrels=500,
        )

        assert info.name == "scifact"
        assert info.num_documents == 5000
        assert info.num_queries == 300
        assert info.num_qrels == 500
        assert isinstance(info.download_timestamp, datetime)
        assert info.beir_version is None
        assert info.source_url is None

    def test_dataset_info_full(self) -> None:
        """Test dataset info with all fields."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        info = DatasetInfo(
            name="scifact",
            num_documents=5000,
            num_queries=300,
            num_qrels=500,
            download_timestamp=timestamp,
            beir_version="2.0.0",
            source_url="https://example.com/scifact.zip",
        )

        assert info.download_timestamp == timestamp
        assert info.beir_version == "2.0.0"
        assert info.source_url == "https://example.com/scifact.zip"


class TestRankingEntry:
    """Tests for RankingEntry model."""

    def test_ranking_entry_creation(self) -> None:
        """Test creating a ranking entry."""
        entry = RankingEntry(
            query_id="q1",
            doc_id="doc1",
            rank=1,
            score=0.95,
        )

        assert entry.query_id == "q1"
        assert entry.doc_id == "doc1"
        assert entry.rank == 1
        assert entry.score == 0.95

    def test_ranking_entry_rank_validation(self) -> None:
        """Test that rank must be >= 1."""
        with pytest.raises(ValueError):
            RankingEntry(query_id="q1", doc_id="doc1", rank=0, score=0.5)


class TestPairwiseComparison:
    """Tests for PairwiseComparison model."""

    def test_comparison_creation(self) -> None:
        """Test creating a pairwise comparison."""
        comparison = PairwiseComparison(
            comparison_id="cmp1",
            query_id="q1",
            doc_a_id="doc1",
            doc_b_id="doc2",
            presented_order=["doc2", "doc1"],
            winner_id="doc1",
            reasoning="Document 1 is more relevant because...",
            model="llama-3.1-8b",
        )

        assert comparison.comparison_id == "cmp1"
        assert comparison.query_id == "q1"
        assert comparison.doc_a_id == "doc1"
        assert comparison.doc_b_id == "doc2"
        assert comparison.presented_order == ["doc2", "doc1"]
        assert comparison.winner_id == "doc1"
        assert comparison.model == "llama-3.1-8b"
        assert isinstance(comparison.timestamp, datetime)

    def test_comparison_presented_order_validation(self) -> None:
        """Test that presented_order must have exactly 2 elements."""
        with pytest.raises(ValueError):
            PairwiseComparison(
                comparison_id="cmp1",
                query_id="q1",
                doc_a_id="doc1",
                doc_b_id="doc2",
                presented_order=["doc1"],  # Should have 2 elements
                winner_id="doc1",
                reasoning="...",
                model="test",
            )


class TestEloRating:
    """Tests for EloRating model."""

    def test_elo_rating_defaults(self) -> None:
        """Test EloRating with default values."""
        rating = EloRating(doc_id="doc1")

        assert rating.doc_id == "doc1"
        assert rating.rating == 1500.0
        assert rating.comparison_count == 0

    def test_elo_rating_custom(self) -> None:
        """Test EloRating with custom values."""
        rating = EloRating(doc_id="doc1", rating=1650.5, comparison_count=10)

        assert rating.rating == 1650.5
        assert rating.comparison_count == 10


class TestEloState:
    """Tests for EloState model."""

    def test_elo_state_creation(self) -> None:
        """Test creating EloState."""
        state = EloState(
            query_id="q1",
            top_k=100,
            initial_ratings={"doc1": 1500.0, "doc2": 1500.0},
            final_ratings={"doc1": 1550.0, "doc2": 1450.0},
        )

        assert state.query_id == "q1"
        assert state.top_k == 100
        assert state.initial_ratings == {"doc1": 1500.0, "doc2": 1500.0}
        assert state.final_ratings == {"doc1": 1550.0, "doc2": 1450.0}
        assert state.k_factor == 32.0
        assert state.iterations == 1
        assert state.convergence_threshold == 1e-6
        assert state.history == []