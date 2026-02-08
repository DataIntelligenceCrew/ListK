"""Pydantic data models for BEIR datasets and pipeline artifacts.

This module defines the core data structures used throughout the pipeline,
including documents, queries, relevance judgments, and ranking results.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A document from a BEIR corpus.

    Attributes
    ----------
    doc_id : str
        Unique identifier for the document within the dataset.
    title : str
        Document title (may be empty for some datasets).
    text : str
        Main text content of the document.
    metadata : dict, optional
        Additional metadata fields specific to the dataset.
    """

    doc_id: str = Field(..., description="Unique document identifier")
    title: str = Field(default="", description="Document title")
    text: str = Field(..., description="Document text content")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class Query(BaseModel):
    """A query from a BEIR dataset.

    Attributes
    ----------
    query_id : str
        Unique identifier for the query within the dataset.
    text : str
        The query text.
    metadata : dict, optional
        Additional metadata fields specific to the dataset.
    """

    query_id: str = Field(..., description="Unique query identifier")
    text: str = Field(..., description="Query text")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class Qrel(BaseModel):
    """A relevance judgment (qrel) from BEIR ground truth.

    Attributes
    ----------
    query_id : str
        The query identifier.
    doc_id : str
        The document identifier.
    relevance : int
        Relevance score (typically 0, 1, 2, or 3 depending on dataset).
    """

    query_id: str = Field(..., description="Query identifier")
    doc_id: str = Field(..., description="Document identifier")
    relevance: int = Field(..., description="Relevance score")


class DatasetInfo(BaseModel):
    """Metadata about a downloaded BEIR dataset.

    Attributes
    ----------
    name : str
        Dataset name (e.g., 'scifact', 'scidocs').
    num_documents : int
        Total number of documents in the corpus.
    num_queries : int
        Total number of queries.
    num_qrels : int
        Total number of relevance judgments.
    download_timestamp : datetime
        When the dataset was downloaded.
    beir_version : str, optional
        Version of the BEIR library used for download.
    source_url : str, optional
        URL from which the dataset was downloaded.
    """

    name: str = Field(..., description="Dataset name")
    num_documents: int = Field(..., description="Number of documents")
    num_queries: int = Field(..., description="Number of queries")
    num_qrels: int = Field(..., description="Number of relevance judgments")
    download_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Download timestamp"
    )
    beir_version: Optional[str] = Field(default=None, description="BEIR version")
    source_url: Optional[str] = Field(default=None, description="Source URL")


class RankingEntry(BaseModel):
    """A single entry in a ranking result.

    Attributes
    ----------
    query_id : str
        The query identifier.
    doc_id : str
        The document identifier.
    rank : int
        Position in the ranking (1-indexed).
    score : float
        Retrieval/relevance score from the model.
    """

    query_id: str = Field(..., description="Query identifier")
    doc_id: str = Field(..., description="Document identifier")
    rank: int = Field(..., ge=1, description="Rank position (1-indexed)")
    score: float = Field(..., description="Retrieval score")


class PairwiseComparison(BaseModel):
    """Result of a pairwise LLM comparison between two documents.

    Attributes
    ----------
    comparison_id : str
        Unique identifier for this comparison.
    query_id : str
        The query identifier.
    doc_a_id : str
        First document identifier.
    doc_b_id : str
        Second document identifier.
    presented_order : list[str]
        Order in which documents were presented to the LLM.
    winner_id : str
        The document ID that was judged more relevant.
    reasoning : str
        The LLM's reasoning for its decision.
    model : str
        The model used for comparison.
    timestamp : datetime
        When the comparison was made.
    """

    comparison_id: str = Field(..., description="Unique comparison identifier")
    query_id: str = Field(..., description="Query identifier")
    doc_a_id: str = Field(..., description="First document ID")
    doc_b_id: str = Field(..., description="Second document ID")
    presented_order: list[str] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Order of document presentation"
    )
    winner_id: str = Field(..., description="Winning document ID")
    reasoning: str = Field(..., description="LLM reasoning")
    model: str = Field(..., description="Model used")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Comparison timestamp"
    )


class EloRating(BaseModel):
    """ELO rating for a document within a query context.

    Attributes
    ----------
    doc_id : str
        The document identifier.
    rating : float
        Current ELO rating.
    comparison_count : int
        Number of comparisons this document participated in.
    """

    doc_id: str = Field(..., description="Document identifier")
    rating: float = Field(default=1500.0, description="ELO rating")
    comparison_count: int = Field(default=0, description="Number of comparisons")


class EloState(BaseModel):
    """Complete state of an ELO tournament for a query.

    Attributes
    ----------
    query_id : str
        The query identifier.
    top_k : int
        Number of top documents being ranked.
    initial_ratings : dict[str, float]
        Starting ELO ratings per document.
    final_ratings : dict[str, float]
        Final ELO ratings after tournament.
    k_factor : float
        ELO K-factor used.
    iterations : int
        Number of tournament iterations run.
    convergence_threshold : float
        Threshold used for convergence check.
    history : list[dict]
        History of rating updates per iteration.
    """

    query_id: str = Field(..., description="Query identifier")
    top_k: int = Field(..., description="Number of top documents")
    initial_ratings: dict[str, float] = Field(
        ...,
        description="Initial ELO ratings"
    )
    final_ratings: dict[str, float] = Field(
        ...,
        description="Final ELO ratings"
    )
    k_factor: float = Field(default=32.0, description="ELO K-factor")
    iterations: int = Field(default=1, description="Tournament iterations")
    convergence_threshold: float = Field(
        default=1e-6,
        description="Convergence threshold"
    )
    history: list[dict] = Field(
        default_factory=list,
        description="Rating history per iteration"
    )