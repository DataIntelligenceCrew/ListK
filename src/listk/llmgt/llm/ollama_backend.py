"""Ollama backend for local LLM inference.

This module provides an LLM backend using Ollama for easy local inference.
Ollama runs a local server and provides a simple HTTP API.

Prerequisites
-------------
1. Install Ollama: https://ollama.ai/download
2. Pull a model: `ollama pull llama3.1:8b` or `ollama pull qwen2.5:7b`
3. Ollama server runs automatically at http://localhost:11434
"""

import logging
from typing import Optional

import requests

from ..data.models import Document, Query
from .base import ComparisonResult, LLMBackend
from .pairwise import parse_winner

logger = logging.getLogger(__name__)

# Default Ollama server URL
DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaBackend(LLMBackend):
    """LLM backend using Ollama for local inference.

    Ollama provides a simple way to run local LLMs via HTTP API.
    Much easier to set up than vLLM - just install Ollama and pull a model.

    Attributes
    ----------
    name : str
        Short name for this backend.
    model_id : str
        Ollama model name (e.g., 'llama3.1:8b', 'qwen2.5:7b').
    base_url : str
        Ollama server URL.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: int = 300,
    ) -> None:
        """Initialize the Ollama backend.

        Parameters
        ----------
        name : str
            Short name for this backend.
        model_id : str
            Ollama model name (e.g., 'llama3.1:8b').
        base_url : str, optional
            Ollama server URL. Default: http://localhost:11434
        timeout : int, optional
            Request timeout in seconds. Default: 300 (5 min).
        """
        super().__init__(name, model_id)
        self.base_url: str = base_url.rstrip("/")
        self.timeout: int = timeout
        self._verified: bool = False

    def _verify_connection(self) -> None:
        """Verify Ollama server is running and model is available."""
        if self._verified:
            return

        # Check server is running
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama server at {self.base_url}. "
                "Make sure Ollama is installed and running:\n"
                "  1. Install: https://ollama.ai/download\n"
                "  2. Start: ollama serve (usually auto-starts)"
            )
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Error connecting to Ollama: {e}")

        # Check model is available
        models_data = response.json()
        available_models = [m["name"] for m in models_data.get("models", [])]

        # Normalize model name for comparison (remove :latest if present)
        model_base = self.model_id.split(":")[0]
        model_found = any(
            m.startswith(model_base) or m.split(":")[0] == model_base
            for m in available_models
        )

        if not model_found:
            available_str = ", ".join(available_models) if available_models else "none"
            raise ValueError(
                f"Model '{self.model_id}' not found in Ollama. "
                f"Available models: {available_str}\n"
                f"Pull the model with: ollama pull {self.model_id}"
            )

        logger.info(f"Ollama connection verified: {self.model_id}")
        self._verified = True

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Generate a response using Ollama chat API.

        Parameters
        ----------
        prompt : str
            The user prompt.
        system_prompt : str, optional
            System prompt to prepend.
        max_tokens : int, optional
            Maximum tokens to generate. Default 1024.
        temperature : float, optional
            Sampling temperature. Default 0.0.

        Returns
        -------
        str
            The generated response.
        """
        self._verify_connection()

        # Build messages for chat API
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Ollama chat API request
        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Ollama request timed out after {self.timeout}s. "
                "Try increasing timeout or using a smaller model."
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}")

        result = response.json()
        return result.get("message", {}).get("content", "")

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
        """Compare two documents for relevance.

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

    def clear(self) -> None:
        """Release resources (no-op for Ollama - model stays loaded in server)."""
        logger.info(f"Ollama backend cleared: {self.model_id}")
        # Ollama manages model memory independently
        # Could call /api/delete or just let Ollama handle it
        pass


# Factory functions for common Ollama models


def create_ollama_llama31_8b(**kwargs) -> OllamaBackend:
    """Create Ollama backend for Llama 3.1 8B.

    Parameters
    ----------
    **kwargs
        Additional arguments passed to OllamaBackend.

    Returns
    -------
    OllamaBackend
        Configured backend.
    """
    return OllamaBackend(
        name="ollama-llama3.1-8b",
        model_id="llama3.1:8b",
        **kwargs,
    )


def create_ollama_llama32_3b(**kwargs) -> OllamaBackend:
    """Create Ollama backend for Llama 3.2 3B (lightweight).

    Parameters
    ----------
    **kwargs
        Additional arguments passed to OllamaBackend.

    Returns
    -------
    OllamaBackend
        Configured backend.
    """
    return OllamaBackend(
        name="ollama-llama3.2-3b",
        model_id="llama3.2:3b",
        **kwargs,
    )


def create_ollama_qwen25_7b(**kwargs) -> OllamaBackend:
    """Create Ollama backend for Qwen 2.5 7B.

    Parameters
    ----------
    **kwargs
        Additional arguments passed to OllamaBackend.

    Returns
    -------
    OllamaBackend
        Configured backend.
    """
    return OllamaBackend(
        name="ollama-qwen2.5-7b",
        model_id="qwen2.5:7b",
        **kwargs,
    )


def create_ollama_qwen25_14b(**kwargs) -> OllamaBackend:
    """Create Ollama backend for Qwen 2.5 14B.

    Parameters
    ----------
    **kwargs
        Additional arguments passed to OllamaBackend.

    Returns
    -------
    OllamaBackend
        Configured backend.
    """
    return OllamaBackend(
        name="ollama-qwen2.5-14b",
        model_id="qwen2.5:14b",
        **kwargs,
    )


def create_ollama_mistral_7b(**kwargs) -> OllamaBackend:
    """Create Ollama backend for Mistral 7B.

    Parameters
    ----------
    **kwargs
        Additional arguments passed to OllamaBackend.

    Returns
    -------
    OllamaBackend
        Configured backend.
    """
    return OllamaBackend(
        name="ollama-mistral-7b",
        model_id="mistral:7b",
        **kwargs,
    )


def create_ollama_gemma2_9b(**kwargs) -> OllamaBackend:
    """Create Ollama backend for Gemma 2 9B.

    Parameters
    ----------
    **kwargs
        Additional arguments passed to OllamaBackend.

    Returns
    -------
    OllamaBackend
        Configured backend.
    """
    return OllamaBackend(
        name="ollama-gemma2-9b",
        model_id="gemma2:9b",
        **kwargs,
    )


def create_ollama_phi3_medium(**kwargs) -> OllamaBackend:
    """Create Ollama backend for Phi-3 Medium (14B).

    Parameters
    ----------
    **kwargs
        Additional arguments passed to OllamaBackend.

    Returns
    -------
    OllamaBackend
        Configured backend.
    """
    return OllamaBackend(
        name="ollama-phi3-medium",
        model_id="phi3:medium",
        **kwargs,
    )


def create_ollama_custom(model_name: str, **kwargs) -> OllamaBackend:
    """Create Ollama backend for any model.

    Parameters
    ----------
    model_name : str
        Ollama model name (e.g., 'llama3.1:70b', 'codellama:13b').
    **kwargs
        Additional arguments passed to OllamaBackend.

    Returns
    -------
    OllamaBackend
        Configured backend.
    """
    # Sanitize name for registry key
    safe_name = f"ollama-{model_name.replace(':', '-').replace('/', '-')}"
    return OllamaBackend(
        name=safe_name,
        model_id=model_name,
        **kwargs,
    )
