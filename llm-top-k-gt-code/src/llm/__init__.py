"""LLM backends for pairwise document comparison.

This module provides LLM backends and utilities for Phase 5 of the pipeline,
where documents are compared pairwise using large language models.
"""

from typing import Callable

from .base import ComparisonResult, LLMBackend
from .openai_backend import (
    OpenAIBackend,
    create_gpt45,
    create_gpt4o,
    create_gpt4o_mini,
    create_gpt5,
    create_gpt5_mini,
    create_gpt5_nano,
    create_gpt52,
    create_o1,
    create_o1_mini,
    create_o3_mini,
)
from .pairwise import (
    DATASET_DESCRIPTIONS,
    DOC_MODE_SNIPPET,
    DOC_MODE_SUMMARY,
    DOC_MODE_TITLE,
    DOC_MODES,
    DocumentSummarizer,
    LLMComparator,
    build_system_prompt,
    build_user_prompt,
    build_user_prompt_with_mode,
    format_document_for_prompt,
    format_document_snippet,
    generate_all_pairs,
    get_completed_pairs,
    get_dataset_description,
    merge_sort_with_llm,
    parse_winner,
    quicksort_with_llm,
    run_pairwise_comparison,
    sort_documents_with_llm,
)
from .vllm_backend import VLLMBackend, create_llama_31_8b, create_llama_32_11b
from .ollama_backend import (
    OllamaBackend,
    create_ollama_llama31_8b,
    create_ollama_llama32_3b,
    create_ollama_qwen25_7b,
    create_ollama_qwen25_14b,
    create_ollama_mistral_7b,
    create_ollama_gemma2_9b,
    create_ollama_phi3_medium,
    create_ollama_custom,
)

# Registry of LLM backend factory functions
# Maps short name -> factory function that creates the backend
LLM_REGISTRY: dict[str, Callable[..., LLMBackend]] = {
    # Local models (vLLM - requires GPU setup)
    "llama-3.1-8b": create_llama_31_8b,
    "llama-3.2-11b": create_llama_32_11b,
    # Local models (Ollama - easy setup)
    "ollama-llama3.1-8b": create_ollama_llama31_8b,
    "ollama-llama3.2-3b": create_ollama_llama32_3b,
    "ollama-qwen2.5-7b": create_ollama_qwen25_7b,
    "ollama-qwen2.5-14b": create_ollama_qwen25_14b,
    "ollama-qwen2.5-32b": lambda **kwargs: create_ollama_custom("qwen2.5:32b", **kwargs),
    "ollama-mistral-7b": create_ollama_mistral_7b,
    "ollama-gemma2-9b": create_ollama_gemma2_9b,
    "ollama-phi3-medium": create_ollama_phi3_medium,
    # OpenAI GPT-4 class
    "gpt-4.5": create_gpt45,
    "gpt-4o": create_gpt4o,
    "gpt-4o-mini": create_gpt4o_mini,
    # OpenAI GPT-5 class
    "gpt-5": create_gpt5,
    "gpt-5-mini": create_gpt5_mini,
    "gpt-5-nano": create_gpt5_nano,
    "gpt-5.2": create_gpt52,
    # OpenAI reasoning models (o1/o3 class)
    "o1": create_o1,
    "o1-mini": create_o1_mini,
    "o3-mini": create_o3_mini,
}


def get_llm_backend(name: str, **kwargs) -> LLMBackend:
    """Get an LLM backend by name.

    Parameters
    ----------
    name : str
        Name of the backend (e.g., 'llama-3.1-8b', 'gpt-4.5').
    **kwargs
        Additional arguments passed to the backend constructor.

    Returns
    -------
    LLMBackend
        Configured LLM backend.

    Raises
    ------
    ValueError
        If the backend name is not recognized.
    """
    if name not in LLM_REGISTRY:
        available: str = ", ".join(sorted(LLM_REGISTRY.keys()))
        raise ValueError(
            f"Unknown LLM backend: '{name}'. Available backends: {available}"
        )

    return LLM_REGISTRY[name](**kwargs)


def list_available_backends() -> list[str]:
    """List available LLM backend names.

    Returns
    -------
    list[str]
        Sorted list of available backend names.
    """
    return sorted(LLM_REGISTRY.keys())


__all__ = [
    # Base classes
    "LLMBackend",
    "ComparisonResult",
    # Backends
    "VLLMBackend",
    "OpenAIBackend",
    "OllamaBackend",
    # Factory functions - vLLM
    "create_llama_31_8b",
    "create_llama_32_11b",
    # Factory functions - Ollama
    "create_ollama_llama31_8b",
    "create_ollama_llama32_3b",
    "create_ollama_qwen25_7b",
    "create_ollama_qwen25_14b",
    "create_ollama_mistral_7b",
    "create_ollama_gemma2_9b",
    "create_ollama_phi3_medium",
    "create_ollama_custom",
    "create_gpt45",
    "create_gpt4o",
    "create_gpt4o_mini",
    "create_gpt5",
    "create_gpt5_mini",
    "create_gpt5_nano",
    "create_gpt52",
    "create_o1",
    "create_o1_mini",
    "create_o3_mini",
    # Registry
    "LLM_REGISTRY",
    "get_llm_backend",
    "list_available_backends",
    # Pairwise utilities
    "run_pairwise_comparison",
    "generate_all_pairs",
    "get_completed_pairs",
    "build_system_prompt",
    "build_user_prompt",
    "build_user_prompt_with_mode",
    "format_document_snippet",
    "format_document_for_prompt",
    "parse_winner",
    "get_dataset_description",
    "DATASET_DESCRIPTIONS",
    # Document modes
    "DOC_MODE_TITLE",
    "DOC_MODE_SUMMARY",
    "DOC_MODE_SNIPPET",
    "DOC_MODES",
    "DocumentSummarizer",
    # Sorting with LLM
    "LLMComparator",
    "sort_documents_with_llm",
    "merge_sort_with_llm",
    "quicksort_with_llm",
]
