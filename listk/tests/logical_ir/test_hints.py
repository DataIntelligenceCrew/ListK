"""
Tests for solicedb.logical_ir.hints module.

Covers:
- Per-operator hint classes
- GlobalHints class
- OperatorHints type alias
"""

import pytest

from solicedb.logical_ir.hints import (
    GlobalHints,
    OperatorHints,
    SemanticFilterHints,
    SemanticFillHints,
    SemanticGroupByHints,
    SemanticJoinHints,
    SemanticProjectHints,
    SemanticSummarizeHints,
    SemanticTopKHints,
)


class TestSemanticFilterHints:
    """Tests for SemanticFilterHints."""

    def test_default_values(self) -> None:
        """Test that default values are None."""
        hints = SemanticFilterHints()
        assert hints.batch_size is None
        assert hints.use_cot is None
        assert hints.model is None
        assert hints.temperature is None

    def test_with_values(self) -> None:
        """Test creating hints with specific values."""
        hints = SemanticFilterHints(
            batch_size=32,
            use_cot=True,
            model="llama-7b",
            temperature=0.7,
        )
        assert hints.batch_size == 32
        assert hints.use_cot is True
        assert hints.model == "llama-7b"
        assert hints.temperature == 0.7

    def test_immutability(self) -> None:
        """Test that hints are immutable."""
        hints = SemanticFilterHints(batch_size=32)
        with pytest.raises(AttributeError):
            hints.batch_size = 64  # type: ignore


class TestSemanticProjectHints:
    """Tests for SemanticProjectHints."""

    def test_default_values(self) -> None:
        """Test that default values are None."""
        hints = SemanticProjectHints()
        assert hints.batch_size is None
        assert hints.use_cot is None
        assert hints.model is None
        assert hints.temperature is None
        assert hints.max_tokens is None

    def test_with_values(self) -> None:
        """Test creating hints with specific values."""
        hints = SemanticProjectHints(
            batch_size=16,
            use_cot=False,
            model="llama-7b",
            temperature=0.5,
            max_tokens=256,
        )
        assert hints.batch_size == 16
        assert hints.use_cot is False
        assert hints.max_tokens == 256


class TestSemanticTopKHints:
    """Tests for SemanticTopKHints."""

    def test_default_values(self) -> None:
        """Test that default values are None."""
        hints = SemanticTopKHints()
        assert hints.method is None
        assert hints.num_pivots is None
        assert hints.window_size is None
        assert hints.batch_size is None
        assert hints.model is None
        assert hints.use_listwise is None

    def test_with_multipivot_method(self) -> None:
        """Test creating hints with multipivot method."""
        hints = SemanticTopKHints(
            method="multipivot",
            num_pivots=3,
            window_size=20,
        )
        assert hints.method == "multipivot"
        assert hints.num_pivots == 3
        assert hints.window_size == 20

    def test_with_heapsort_method(self) -> None:
        """Test creating hints with heapsort method."""
        hints = SemanticTopKHints(method="heapsort")
        assert hints.method == "heapsort"

    def test_with_tournament_method(self) -> None:
        """Test creating hints with tournament method."""
        hints = SemanticTopKHints(method="tournament")
        assert hints.method == "tournament"

    def test_with_pairwise_method(self) -> None:
        """Test creating hints with pairwise method."""
        hints = SemanticTopKHints(method="pairwise")
        assert hints.method == "pairwise"

    def test_use_listwise_flag(self) -> None:
        """Test use_listwise flag."""
        hints = SemanticTopKHints(use_listwise=True)
        assert hints.use_listwise is True


class TestSemanticJoinHints:
    """Tests for SemanticJoinHints."""

    def test_default_values(self) -> None:
        """Test that default values are None."""
        hints = SemanticJoinHints()
        assert hints.blocking_strategy is None
        assert hints.blocking_key is None
        assert hints.similarity_threshold is None
        assert hints.batch_size is None
        assert hints.model is None

    def test_with_embedding_blocking(self) -> None:
        """Test creating hints with embedding blocking strategy."""
        hints = SemanticJoinHints(
            blocking_strategy="embedding",
            similarity_threshold=0.8,
        )
        assert hints.blocking_strategy == "embedding"
        assert hints.similarity_threshold == 0.8

    def test_with_key_blocking(self) -> None:
        """Test creating hints with key blocking strategy."""
        hints = SemanticJoinHints(
            blocking_strategy="key",
            blocking_key="category",
        )
        assert hints.blocking_strategy == "key"
        assert hints.blocking_key == "category"

    def test_with_no_blocking(self) -> None:
        """Test creating hints with no blocking."""
        hints = SemanticJoinHints(blocking_strategy="none")
        assert hints.blocking_strategy == "none"


class TestSemanticGroupByHints:
    """Tests for SemanticGroupByHints."""

    def test_default_values(self) -> None:
        """Test that default values are None."""
        hints = SemanticGroupByHints()
        assert hints.batch_size is None
        assert hints.model is None
        assert hints.num_groups is None

    def test_with_values(self) -> None:
        """Test creating hints with specific values."""
        hints = SemanticGroupByHints(
            batch_size=64,
            model="llama-7b",
            num_groups=5,
        )
        assert hints.batch_size == 64
        assert hints.num_groups == 5


class TestSemanticSummarizeHints:
    """Tests for SemanticSummarizeHints."""

    def test_default_values(self) -> None:
        """Test that default values are None."""
        hints = SemanticSummarizeHints()
        assert hints.model is None
        assert hints.max_tokens is None
        assert hints.temperature is None
        assert hints.max_input_rows is None

    def test_with_values(self) -> None:
        """Test creating hints with specific values."""
        hints = SemanticSummarizeHints(
            model="llama-7b",
            max_tokens=500,
            temperature=0.3,
            max_input_rows=100,
        )
        assert hints.max_tokens == 500
        assert hints.max_input_rows == 100


class TestSemanticFillHints:
    """Tests for SemanticFillHints."""

    def test_default_values(self) -> None:
        """Test that default values are None."""
        hints = SemanticFillHints()
        assert hints.batch_size is None
        assert hints.model is None
        assert hints.use_cot is None

    def test_with_values(self) -> None:
        """Test creating hints with specific values."""
        hints = SemanticFillHints(
            batch_size=32,
            model="llama-7b",
            use_cot=True,
        )
        assert hints.batch_size == 32
        assert hints.use_cot is True


class TestGlobalHints:
    """Tests for GlobalHints."""

    def test_default_values(self) -> None:
        """Test default values."""
        hints = GlobalHints()
        assert hints.prefer_listwise is False
        assert hints.max_batch_size is None
        assert hints.default_model is None
        assert hints.parallelism is None
        assert hints.enable_caching is True
        assert hints.debug_mode is False
        assert hints.extra == {}

    def test_with_values(self) -> None:
        """Test creating hints with specific values."""
        hints = GlobalHints(
            prefer_listwise=True,
            max_batch_size=64,
            default_model="llama-7b",
            parallelism=4,
            enable_caching=False,
            debug_mode=True,
        )
        assert hints.prefer_listwise is True
        assert hints.max_batch_size == 64
        assert hints.default_model == "llama-7b"
        assert hints.parallelism == 4
        assert hints.enable_caching is False
        assert hints.debug_mode is True

    def test_with_extra_dict(self) -> None:
        """Test creating hints with extra dict."""
        hints = GlobalHints(extra={"custom_key": "custom_value"})
        assert hints.extra["custom_key"] == "custom_value"

    def test_with_extra_method(self) -> None:
        """Test with_extra method creates new hints."""
        hints1 = GlobalHints()
        hints2 = hints1.with_extra("key1", "value1")

        # Original unchanged
        assert "key1" not in hints1.extra
        # New has the value
        assert hints2.extra["key1"] == "value1"

    def test_with_extra_preserves_existing(self) -> None:
        """Test with_extra preserves existing values."""
        hints1 = GlobalHints(
            prefer_listwise=True,
            max_batch_size=64,
            extra={"existing": "value"},
        )
        hints2 = hints1.with_extra("new_key", "new_value")

        assert hints2.prefer_listwise is True
        assert hints2.max_batch_size == 64
        assert hints2.extra["existing"] == "value"
        assert hints2.extra["new_key"] == "new_value"

    def test_with_extra_multiple_calls(self) -> None:
        """Test chaining multiple with_extra calls."""
        hints = (
            GlobalHints()
            .with_extra("key1", "value1")
            .with_extra("key2", "value2")
            .with_extra("key3", "value3")
        )
        assert hints.extra == {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
        }

    def test_immutability(self) -> None:
        """Test that GlobalHints is immutable."""
        hints = GlobalHints()
        with pytest.raises(AttributeError):
            hints.prefer_listwise = True  # type: ignore


class TestOperatorHintsTypeAlias:
    """Tests for OperatorHints type alias."""

    def test_accepts_semantic_filter_hints(self) -> None:
        """Test OperatorHints accepts SemanticFilterHints."""
        hints: OperatorHints = SemanticFilterHints(batch_size=32)
        assert hints is not None

    def test_accepts_semantic_project_hints(self) -> None:
        """Test OperatorHints accepts SemanticProjectHints."""
        hints: OperatorHints = SemanticProjectHints(max_tokens=256)
        assert hints is not None

    def test_accepts_semantic_topk_hints(self) -> None:
        """Test OperatorHints accepts SemanticTopKHints."""
        hints: OperatorHints = SemanticTopKHints(method="multipivot")
        assert hints is not None

    def test_accepts_semantic_join_hints(self) -> None:
        """Test OperatorHints accepts SemanticJoinHints."""
        hints: OperatorHints = SemanticJoinHints(blocking_strategy="embedding")
        assert hints is not None

    def test_accepts_semantic_groupby_hints(self) -> None:
        """Test OperatorHints accepts SemanticGroupByHints."""
        hints: OperatorHints = SemanticGroupByHints(num_groups=5)
        assert hints is not None

    def test_accepts_semantic_summarize_hints(self) -> None:
        """Test OperatorHints accepts SemanticSummarizeHints."""
        hints: OperatorHints = SemanticSummarizeHints(max_tokens=500)
        assert hints is not None

    def test_accepts_semantic_fill_hints(self) -> None:
        """Test OperatorHints accepts SemanticFillHints."""
        hints: OperatorHints = SemanticFillHints(use_cot=True)
        assert hints is not None

    def test_accepts_none(self) -> None:
        """Test OperatorHints accepts None."""
        hints: OperatorHints = None
        assert hints is None