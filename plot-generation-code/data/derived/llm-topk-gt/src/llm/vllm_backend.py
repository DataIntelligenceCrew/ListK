"""vLLM backend for local LLM inference.

This module provides an LLM backend using vLLM for efficient
local inference with Llama and other models.

Optimized for efficiency:
- Single-step generation (no constrained decoding overhead)
- Minimal prompts
"""

import logging
from typing import Optional

from ..data.models import Document, Query
from .base import ComparisonResult, LLMBackend
from .pairwise import parse_winner

logger = logging.getLogger(__name__)


class VLLMBackend(LLMBackend):
    """LLM backend using vLLM for local inference.

    This backend loads models locally using vLLM for efficient
    batched inference with GPU acceleration.

    Optimized for efficiency with single-step generation.

    Attributes
    ----------
    name : str
        Short name for this backend.
    model_id : str
        HuggingFace model identifier.
    tensor_parallel_size : int
        Number of GPUs for tensor parallelism.
    max_model_len : int
        Maximum sequence length.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        """Initialize the vLLM backend.

        Parameters
        ----------
        name : str
            Short name for this backend.
        model_id : str
            HuggingFace model identifier.
        tensor_parallel_size : int, optional
            Number of GPUs for tensor parallelism. Default 1.
        max_model_len : int, optional
            Maximum sequence length. Default 8192.
        gpu_memory_utilization : float, optional
            Fraction of GPU memory to use. Default 0.9.
        """
        super().__init__(name, model_id)
        self.tensor_parallel_size: int = tensor_parallel_size
        self.max_model_len: int = max_model_len
        self.gpu_memory_utilization: float = gpu_memory_utilization

        self._llm = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Lazily load the vLLM model."""
        if self._llm is not None:
            return

        logger.info(f"Loading vLLM model: {self.model_id}")

        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError(
                "vLLM is required for this backend. Install with: pip install vllm"
            ) from e

        self._llm = LLM(
            model=self.model_id,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
        )

        logger.info(f"vLLM model loaded: {self.model_id}")

    def _format_chat_prompt(
        self, system_prompt: Optional[str], user_prompt: str
    ) -> str:
        """Format prompts using the model's chat template.

        Parameters
        ----------
        system_prompt : str, optional
            System message.
        user_prompt : str
            User message.

        Returns
        -------
        str
            Formatted prompt string.
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})

        # Use the tokenizer's chat template
        tokenizer = self._llm.get_tokenizer()
        formatted: str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return formatted

    def _format_multi_turn_prompt(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        assistant_response: str,
        followup_prompt: str,
    ) -> str:
        """Format a multi-turn conversation prompt.

        Parameters
        ----------
        system_prompt : str, optional
            System message.
        user_prompt : str
            Initial user message.
        assistant_response : str
            Assistant's first response.
        followup_prompt : str
            User's followup message.

        Returns
        -------
        str
            Formatted prompt string.
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "assistant", "content": assistant_response})
        messages.append({"role": "user", "content": followup_prompt})

        tokenizer = self._llm.get_tokenizer()
        formatted: str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return formatted

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
            Sampling temperature. Default 0.0.

        Returns
        -------
        str
            The generated response.
        """
        self._load_model()

        from vllm import SamplingParams

        # Format the prompt
        formatted_prompt: str = self._format_chat_prompt(system_prompt, prompt)

        # Set up sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0 if temperature == 0 else 0.95,
        )

        # Generate
        outputs = self._llm.generate([formatted_prompt], sampling_params)

        return outputs[0].outputs[0].text

    def generate_constrained(
        self,
        prompt: str,
        choices: list[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate with constrained decoding (choice from list).

        Parameters
        ----------
        prompt : str
            The user prompt.
        choices : list[str]
            Allowed output choices (e.g., ["A", "B"]).
        system_prompt : str, optional
            System prompt to prepend.
        temperature : float, optional
            Sampling temperature. Default 0.0.

        Returns
        -------
        str
            One of the allowed choices.
        """
        self._load_model()

        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams

        # Format the prompt
        formatted_prompt: str = self._format_chat_prompt(system_prompt, prompt)

        # Set up guided decoding to constrain to choices
        guided_params = GuidedDecodingParams(choice=choices)

        sampling_params = SamplingParams(
            max_tokens=1,  # Only need single token for A or B
            temperature=temperature,
            guided_decoding=guided_params,
        )

        # Generate
        outputs = self._llm.generate([formatted_prompt], sampling_params)

        return outputs[0].outputs[0].text.strip()

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

    def clear(self) -> None:
        """Release model resources."""
        if self._llm is not None:
            logger.info(f"Clearing vLLM model: {self.model_id}")
            # vLLM doesn't have a built-in cleanup, but we can delete references
            del self._llm
            self._llm = None

            # Try to free GPU memory
            try:
                import gc

                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("vLLM model cleared")


# Convenience factory functions for common models
def create_llama_31_8b(**kwargs) -> VLLMBackend:
    """Create a vLLM backend for Llama 3.1 8B Instruct.

    Parameters
    ----------
    **kwargs
        Additional arguments passed to VLLMBackend.

    Returns
    -------
    VLLMBackend
        Configured backend.
    """
    return VLLMBackend(
        name="llama-3.1-8b",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        **kwargs,
    )


def create_llama_32_11b(**kwargs) -> VLLMBackend:
    """Create a vLLM backend for Llama 3.2 11B Vision Instruct.

    Parameters
    ----------
    **kwargs
        Additional arguments passed to VLLMBackend.

    Returns
    -------
    VLLMBackend
        Configured backend.
    """
    return VLLMBackend(
        name="llama-3.2-11b",
        model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        **kwargs,
    )
