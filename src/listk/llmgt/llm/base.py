"""Abstract base class for LLM backends.

This module defines the interface that all LLM backends must implement
for pairwise document comparison in Phase 5 of the pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ..data.models import Document, Query


@dataclass
class ComparisonResult:
    """Result of a single pairwise comparison.

    Attributes
    ----------
    winner : str
        Which document won: 'A' or 'B'.
    reasoning : str
        The LLM's reasoning for the decision.
    raw_response : str
        The full raw response from the LLM.
    """

    winner: str
    reasoning: str
    raw_response: str


class LLMBackend(ABC):
    """Abstract base class for LLM backends.

    All LLM backends must implement this interface to be used in the
    pairwise comparison pipeline. Backends handle loading models,
    generating responses, and cleaning up resources.

    Attributes
    ----------
    name : str
        Unique identifier for this backend (e.g., 'llama-3.1-8b').
    model_id : str
        Full model identifier (e.g., 'meta-llama/Llama-3.1-8B-Instruct').
    """

    def __init__(self, name: str, model_id: str) -> None:
        """Initialize the LLM backend.

        Parameters
        ----------
        name : str
            Short name for this backend.
        model_id : str
            Full model identifier for loading.
        """
        self.name: str = name
        self.model_id: str = model_id

    @abstractmethod
    def compare_documents(
        self,
        query: Query,
        doc_a: Document,
        doc_b: Document,
        dataset_description: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
    ) -> ComparisonResult:
        """Compare two documents for relevance to a query.

        Parameters
        ----------
        query : Query
            The query to judge relevance against.
        doc_a : Document
            First document (will be labeled 'A' in prompt).
        doc_b : Document
            Second document (will be labeled 'B' in prompt).
        dataset_description : str
            Description of the dataset/task for context.
        system_prompt : str
            System prompt to set LLM behavior.
        user_prompt : str
            Formatted user prompt with query and documents.
        max_tokens : int, optional
            Maximum tokens to generate. Default 1024.

        Returns
        -------
        ComparisonResult
            The comparison result with winner, reasoning, and raw response.
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Generate a response from the LLM.

        Parameters
        ----------
        prompt : str
            The user prompt.
        system_prompt : str, optional
            System prompt to prepend.
        max_tokens : int, optional
            Maximum tokens to generate. Default 1024.
        temperature : float, optional
            Sampling temperature. Default 0.0 (deterministic).

        Returns
        -------
        str
            The generated response.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Release model resources and free memory.

        Should be called after processing is complete to free GPU memory.
        """
        pass

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(name='{self.name}', model_id='{self.model_id}')"
