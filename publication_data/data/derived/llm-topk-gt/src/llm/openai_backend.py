"""OpenAI API backend for LLM inference.

This module provides an LLM backend using the OpenAI API for
inference with GPT-4.5 and other OpenAI models.

Optimized for efficiency:
- Single-step generation (no constrained decoding overhead)
- Async concurrent requests for batching
- Minimal prompts
"""

import asyncio
import logging
import os
from typing import Optional

from ..data.models import Document, Query
from .base import ComparisonResult, LLMBackend
from .pairwise import parse_winner

logger = logging.getLogger(__name__)


class OpenAIBackend(LLMBackend):
    """LLM backend using OpenAI API with async support.

    This backend uses the OpenAI API for inference with GPT models.
    Supports concurrent async requests for efficient batching.

    Attributes
    ----------
    name : str
        Short name for this backend.
    model_id : str
        OpenAI model identifier (e.g., 'gpt-4o').
    api_key : str, optional
        OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
    max_concurrent : int
        Maximum concurrent requests. Default 10.
    """

    # Models that require max_completion_tokens instead of max_tokens
    COMPLETION_TOKEN_MODELS = {"o1", "o1-mini", "o3-mini", "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.2"}

    def __init__(
        self,
        name: str,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_concurrent: int = 10,
    ) -> None:
        """Initialize the OpenAI backend.

        Parameters
        ----------
        name : str
            Short name for this backend.
        model_id : str
            OpenAI model identifier.
        api_key : str, optional
            OpenAI API key. Uses env var if not provided.
        base_url : str, optional
            Custom base URL for API (for compatible endpoints).
        max_concurrent : int, optional
            Maximum concurrent requests. Default 10.
        """
        super().__init__(name, model_id)
        self.api_key: Optional[str] = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url: Optional[str] = base_url
        self.max_concurrent: int = max_concurrent

        self._client = None
        self._async_client = None

    def _get_client(self):
        """Lazily initialize the sync OpenAI client."""
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            ) from e

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment "
                "variable or pass api_key to constructor."
            )

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        logger.info(f"OpenAI client initialized for model: {self.model_id}")
        return self._client

    def _get_async_client(self):
        """Lazily initialize the async OpenAI client."""
        if self._async_client is not None:
            return self._async_client

        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            ) from e

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment "
                "variable or pass api_key to constructor."
            )

        self._async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        logger.info(f"Async OpenAI client initialized for model: {self.model_id}")
        return self._async_client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 150,
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
            Maximum tokens to generate. Default 150 (succinct).
        temperature : float, optional
            Sampling temperature. Default 0.0.

        Returns
        -------
        str
            The generated response.
        """
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Use max_completion_tokens for newer models (o1, gpt-5, etc.)
        if self.model_id in self.COMPLETION_TOKEN_MODELS:
            response = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_completion_tokens=max_tokens,
            )
        else:
            response = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        return response.choices[0].message.content or ""

    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 150,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> str:
        """Generate a response asynchronously.

        Parameters
        ----------
        prompt : str
            The user prompt.
        system_prompt : str, optional
            System prompt to prepend.
        max_tokens : int, optional
            Maximum tokens to generate. Default 150.
        temperature : float, optional
            Sampling temperature. Default 0.0.
        max_retries : int, optional
            Maximum retry attempts for transient errors. Default 3.

        Returns
        -------
        str
            The generated response.
        """
        client = self._get_async_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(max_retries):
            try:
                # Use max_completion_tokens for newer models (o1, gpt-5, etc.)
                if self.model_id in self.COMPLETION_TOKEN_MODELS:
                    response = await client.chat.completions.create(
                        model=self.model_id,
                        messages=messages,
                        max_completion_tokens=max_tokens,
                    )
                else:
                    response = await client.chat.completions.create(
                        model=self.model_id,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    await asyncio.sleep(2 ** attempt)
                    logger.warning(f"Retry {attempt + 1}/{max_retries} after error: {e}")

        raise last_error

    async def generate_batch_async(
        self,
        prompts: list[tuple[str, Optional[str]]],  # (user_prompt, system_prompt)
        max_tokens: int = 150,
        temperature: float = 0.0,
    ) -> list[str]:
        """Generate responses for multiple prompts concurrently.

        Parameters
        ----------
        prompts : list[tuple[str, Optional[str]]]
            List of (user_prompt, system_prompt) tuples.
        max_tokens : int, optional
            Maximum tokens per response. Default 150.
        temperature : float, optional
            Sampling temperature. Default 0.0.

        Returns
        -------
        list[str]
            List of responses in same order as prompts.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def limited_generate(prompt: str, system: Optional[str]) -> str:
            async with semaphore:
                return await self.generate_async(prompt, system, max_tokens, temperature)

        tasks = [limited_generate(p, s) for p, s in prompts]
        return await asyncio.gather(*tasks)

    def compare_documents(
        self,
        query: Query,
        doc_a: Document,
        doc_b: Document,
        dataset_description: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 150,
    ) -> ComparisonResult:
        """Compare two documents (single-step, succinct).

        Parameters
        ----------
        query : Query
            The query to judge relevance against.
        doc_a : Document
            First document.
        doc_b : Document
            Second document.
        dataset_description : str
            Description of the dataset.
        system_prompt : str
            System prompt.
        user_prompt : str
            Formatted user prompt.
        max_tokens : int, optional
            Maximum tokens. Default 150 (succinct).

        Returns
        -------
        ComparisonResult
            The comparison result.
        """
        response: str = self.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )

        winner: Optional[str] = parse_winner(response)

        return ComparisonResult(
            winner=winner or "",
            reasoning=response,
            raw_response=response,
        )

    async def compare_documents_async(
        self,
        query: Query,
        doc_a: Document,
        doc_b: Document,
        dataset_description: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 150,
    ) -> ComparisonResult:
        """Compare two documents asynchronously.

        Parameters
        ----------
        query : Query
            The query to judge relevance against.
        doc_a : Document
            First document.
        doc_b : Document
            Second document.
        dataset_description : str
            Description of the dataset.
        system_prompt : str
            System prompt.
        user_prompt : str
            Formatted user prompt.
        max_tokens : int, optional
            Maximum tokens. Default 150.

        Returns
        -------
        ComparisonResult
            The comparison result.
        """
        response: str = await self.generate_async(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )

        winner: Optional[str] = parse_winner(response)

        return ComparisonResult(
            winner=winner or "",
            reasoning=response,
            raw_response=response,
        )

    def clear(self) -> None:
        """Release client resources."""
        if self._client is not None:
            logger.info("Clearing OpenAI client")
            self._client = None
        if self._async_client is not None:
            self._async_client = None


# Convenience factory functions
def create_gpt45(**kwargs) -> OpenAIBackend:
    """Create an OpenAI backend for GPT-4.5 Turbo."""
    return OpenAIBackend(
        name="gpt-4.5",
        model_id="gpt-4.5-turbo",
        **kwargs,
    )


def create_gpt4o(**kwargs) -> OpenAIBackend:
    """Create an OpenAI backend for GPT-4o."""
    return OpenAIBackend(
        name="gpt-4o",
        model_id="gpt-4o",
        **kwargs,
    )


def create_gpt4o_mini(**kwargs) -> OpenAIBackend:
    """Create an OpenAI backend for GPT-4o-mini (cheapest)."""
    return OpenAIBackend(
        name="gpt-4o-mini",
        model_id="gpt-4o-mini",
        **kwargs,
    )


def create_o1(**kwargs) -> OpenAIBackend:
    """Create an OpenAI backend for o1 (full reasoning model)."""
    return OpenAIBackend(
        name="o1",
        model_id="o1",
        **kwargs,
    )


def create_o1_mini(**kwargs) -> OpenAIBackend:
    """Create an OpenAI backend for o1-mini (smaller reasoning model)."""
    return OpenAIBackend(
        name="o1-mini",
        model_id="o1-mini",
        **kwargs,
    )


def create_o3_mini(**kwargs) -> OpenAIBackend:
    """Create an OpenAI backend for o3-mini (latest reasoning model)."""
    return OpenAIBackend(
        name="o3-mini",
        model_id="o3-mini",
        **kwargs,
    )


# GPT-5 class models
def create_gpt5(**kwargs) -> OpenAIBackend:
    """Create an OpenAI backend for GPT-5."""
    return OpenAIBackend(
        name="gpt-5",
        model_id="gpt-5",
        **kwargs,
    )


def create_gpt5_mini(**kwargs) -> OpenAIBackend:
    """Create an OpenAI backend for GPT-5 Mini (smaller, cheaper)."""
    return OpenAIBackend(
        name="gpt-5-mini",
        model_id="gpt-5-mini",
        **kwargs,
    )


def create_gpt5_nano(**kwargs) -> OpenAIBackend:
    """Create an OpenAI backend for GPT-5 Nano (smallest, cheapest)."""
    return OpenAIBackend(
        name="gpt-5-nano",
        model_id="gpt-5-nano",
        **kwargs,
    )


def create_gpt52(**kwargs) -> OpenAIBackend:
    """Create an OpenAI backend for GPT-5.2 (latest)."""
    return OpenAIBackend(
        name="gpt-5.2",
        model_id="gpt-5.2",
        **kwargs,
    )
