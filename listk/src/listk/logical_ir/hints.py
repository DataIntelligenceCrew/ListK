"""
Typed hint classes for the SoliceDB logical IR.

This module defines hint structures that allow users to influence physical
execution strategies without changing the logical semantics. Hints are optional
and the optimizer/executor can choose to ignore them.

Two levels of hints are supported:
- Per-operator hints: Attached to specific logical nodes
- Global hints: Apply to the entire query plan
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ============================================================================
# Per-Operator Hints
# ============================================================================


@dataclass(frozen=True)
class SemanticFilterHints:
    """
    Hints for SemanticFilter operator execution.

    Parameters
    ----------
    batch_size : int | None
        Number of rows to process in each LLM batch call.
    use_cot : bool | None
        Whether to use chain-of-thought prompting.
    model : str | None
        Specific model to use for this operator.
    temperature : float | None
        LLM temperature setting.
    """

    batch_size: int | None = None
    use_cot: bool | None = None
    model: str | None = None
    temperature: float | None = None


@dataclass(frozen=True)
class SemanticProjectHints:
    """
    Hints for SemanticProject operator execution.

    Parameters
    ----------
    batch_size : int | None
        Number of rows to process in each LLM batch call.
    use_cot : bool | None
        Whether to use chain-of-thought prompting.
    model : str | None
        Specific model to use for this operator.
    temperature : float | None
        LLM temperature setting.
    max_tokens : int | None
        Maximum tokens for LLM response.
    """

    batch_size: int | None = None
    use_cot: bool | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class SemanticTopKHints:
    """
    Hints for SemanticTopK operator execution.

    Parameters
    ----------
    method : Literal["multipivot", "heapsort", "tournament", "pairwise"] | None
        The physical algorithm to use for top-k selection.
    num_pivots : int | None
        Number of pivots for multipivot method.
    window_size : int | None
        Window size for listwise ranking methods.
    batch_size : int | None
        Batch size for LLM calls.
    model : str | None
        Specific model to use for this operator.
    use_listwise : bool | None
        Whether to use listwise ranking (e.g., RankZephyr) vs pairwise.
    """

    method: Literal["multipivot", "heapsort", "tournament", "pairwise"] | None = None
    num_pivots: int | None = None
    window_size: int | None = None
    batch_size: int | None = None
    model: str | None = None
    use_listwise: bool | None = None


@dataclass(frozen=True)
class SemanticJoinHints:
    """
    Hints for SemanticJoin operator execution.

    Parameters
    ----------
    blocking_strategy : Literal["none", "embedding", "key"] | None
        Strategy to reduce candidate pairs before semantic comparison.
        - "none": Full cross product (expensive)
        - "embedding": Use embedding similarity for blocking
        - "key": Use a key column for blocking
    blocking_key : str | None
        Column name to use for key-based blocking.
    similarity_threshold : float | None
        Threshold for embedding-based blocking (0.0 to 1.0).
    batch_size : int | None
        Number of pairs to process in each LLM batch call.
    model : str | None
        Specific model to use for this operator.
    """

    blocking_strategy: Literal["none", "embedding", "key"] | None = None
    blocking_key: str | None = None
    similarity_threshold: float | None = None
    batch_size: int | None = None
    model: str | None = None


@dataclass(frozen=True)
class SemanticGroupByHints:
    """
    Hints for SemanticGroupBy operator execution.

    Parameters
    ----------
    batch_size : int | None
        Number of rows to process in each LLM batch call.
    model : str | None
        Specific model to use for this operator.
    num_groups : int | None
        Expected number of groups (helps with prompting).
    """

    batch_size: int | None = None
    model: str | None = None
    num_groups: int | None = None


@dataclass(frozen=True)
class SemanticSummarizeHints:
    """
    Hints for SemanticSummarize operator execution.

    Parameters
    ----------
    model : str | None
        Specific model to use for this operator.
    max_tokens : int | None
        Maximum tokens for summary output.
    temperature : float | None
        LLM temperature setting.
    max_input_rows : int | None
        Maximum rows to include in summarization context.
        If group has more rows, sampling or truncation occurs.
    """

    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    max_input_rows: int | None = None


@dataclass(frozen=True)
class SemanticFillHints:
    """
    Hints for SemanticFill operator execution.

    Parameters
    ----------
    batch_size : int | None
        Number of rows to process in each LLM batch call.
    model : str | None
        Specific model to use for this operator.
    use_cot : bool | None
        Whether to use chain-of-thought prompting.
    """

    batch_size: int | None = None
    model: str | None = None
    use_cot: bool | None = None


# ============================================================================
# Global Hints
# ============================================================================


@dataclass(frozen=True)
class GlobalHints:
    """
    Global hints that apply to the entire query plan.

    These hints influence optimizer decisions and default execution behavior
    across all operators in the plan.

    Parameters
    ----------
    prefer_listwise : bool
        Prefer listwise ranking methods over pairwise when applicable.
    max_batch_size : int | None
        Default maximum batch size for all LLM calls.
    default_model : str | None
        Default model to use when not specified per-operator.
    parallelism : int | None
        Number of parallel workers for LLM inference.
    enable_caching : bool
        Whether to cache LLM call results.
    debug_mode : bool
        Enable additional logging and diagnostics.
    extra : dict[str, Any]
        Additional untyped hints for experimental features.
    """

    prefer_listwise: bool = False
    max_batch_size: int | None = None
    default_model: str | None = None
    parallelism: int | None = None
    enable_caching: bool = True
    debug_mode: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def with_extra(self, key: str, value: Any) -> GlobalHints:
        """
        Create a new GlobalHints with an additional extra hint.

        Parameters
        ----------
        key : str
            The hint key.
        value : Any
            The hint value.

        Returns
        -------
        GlobalHints
            A new GlobalHints with the extra hint added.
        """
        new_extra = dict(self.extra)
        new_extra[key] = value
        return GlobalHints(
            prefer_listwise=self.prefer_listwise,
            max_batch_size=self.max_batch_size,
            default_model=self.default_model,
            parallelism=self.parallelism,
            enable_caching=self.enable_caching,
            debug_mode=self.debug_mode,
            extra=new_extra,
        )


# ============================================================================
# Type alias for any hint type
# ============================================================================

OperatorHints = (
    SemanticFilterHints
    | SemanticProjectHints
    | SemanticTopKHints
    | SemanticJoinHints
    | SemanticGroupByHints
    | SemanticSummarizeHints
    | SemanticFillHints
    | None
)
