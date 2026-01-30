"""
Annotations for physical IR nodes.

This module defines metadata attached to physical nodes:
- CostEstimate: Predicted resource usage (tokens, time, memory)
- Provenance: Tracks the logical origin and optimization rule that created a node

All annotations are immutable (frozen dataclasses).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from solicedb.logical_ir.nodes import NodeId


@dataclass(frozen=True)
class CostEstimate:
    """
    Estimated cost of executing a physical operator.

    Costs are estimates made during planning. Actual costs are tracked
    separately in ExecutionState during runtime.

    Parameters
    ----------
    token_count : int | None
        Estimated number of LLM tokens (input + output).
        None for classical operators that don't use LLMs.
    llm_calls : int | None
        Estimated number of LLM API calls.
        None for classical operators.
    row_count : int | None
        Estimated output cardinality.
    cpu_cost : float | None
        Relative CPU cost (unitless, for comparison).
    memory_bytes : int | None
        Estimated peak memory usage in bytes.
    latency_ms : float | None
        Estimated wall-clock time in milliseconds.

    Notes
    -----
    Not all fields need to be populated. The optimizer uses whichever
    fields are available for cost-based decisions.
    """

    token_count: int | None = None
    llm_calls: int | None = None
    row_count: int | None = None
    cpu_cost: float | None = None
    memory_bytes: int | None = None
    latency_ms: float | None = None

    def __add__(self, other: CostEstimate) -> CostEstimate:
        """
        Combine two cost estimates (for summing costs along a path).

        Parameters
        ----------
        other : CostEstimate
            Another cost estimate to add.

        Returns
        -------
        CostEstimate
            Combined cost estimate. None values are treated as 0
            if the other value is present.
        """

        def add_optional(a: float | int | None, b: float | int | None) -> float | int | None:
            if a is None and b is None:
                return None
            return (a or 0) + (b or 0)

        return CostEstimate(
            token_count=add_optional(self.token_count, other.token_count),
            llm_calls=add_optional(self.llm_calls, other.llm_calls),
            row_count=other.row_count,  # Take the output cardinality of the later op
            cpu_cost=add_optional(self.cpu_cost, other.cpu_cost),
            memory_bytes=max(self.memory_bytes or 0, other.memory_bytes or 0),
            latency_ms=add_optional(self.latency_ms, other.latency_ms),
        )

    @classmethod
    def zero(cls) -> CostEstimate:
        """
        Create a zero-cost estimate.

        Returns
        -------
        CostEstimate
            Cost estimate with all values set to 0.
        """
        return cls(
            token_count=0,
            llm_calls=0,
            row_count=0,
            cpu_cost=0.0,
            memory_bytes=0,
            latency_ms=0.0,
        )

    @classmethod
    def unknown(cls) -> CostEstimate:
        """
        Create an unknown cost estimate (all None).

        Returns
        -------
        CostEstimate
            Cost estimate with all values set to None.
        """
        return cls()


@dataclass(frozen=True)
class Provenance:
    """
    Tracks the origin of a physical node.

    This enables tracing from physical plan back to logical plan,
    and understanding which optimization rules produced the plan.

    Parameters
    ----------
    logical_node_ids : tuple[NodeId, ...]
        IDs of logical nodes that this physical node implements.
        Usually a single ID, but can be multiple for fused operators.
        Empty tuple for nodes with no logical counterpart (e.g., added
        by physical optimizations like adding sorts for merge join).
    rule_name : str | None
        Name of the optimization rule that created this node.
        None if created by direct translation (no optimization).
    rule_metadata : dict[str, Any]
        Additional metadata from the rule (e.g., why this choice was made).
        Useful for debugging and research analysis.

    Examples
    --------
    >>> # Direct translation of logical Select
    >>> prov = Provenance(logical_node_ids=(3,), rule_name=None)

    >>> # Created by SummarizeThenJoin optimization rule
    >>> prov = Provenance(
    ...     logical_node_ids=(5,),  # The original SemanticJoin node
    ...     rule_name="SummarizeThenJoinRule",
    ...     rule_metadata={"reason": "tables exceed token threshold"}
    ... )

    >>> # Sort added to satisfy merge join requirement (no logical origin)
    >>> prov = Provenance(
    ...     logical_node_ids=(),
    ...     rule_name="EnforceSortForMergeJoin",
    ... )
    """

    logical_node_ids: tuple[NodeId, ...] = field(default_factory=tuple)
    rule_name: str | None = None
    rule_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def direct(cls, logical_node_id: NodeId) -> Provenance:
        """
        Create provenance for direct translation (no optimization rule).

        Parameters
        ----------
        logical_node_id : NodeId
            The logical node ID being translated.

        Returns
        -------
        Provenance
            Provenance with single logical origin and no rule.
        """
        return cls(logical_node_ids=(logical_node_id,))

    @classmethod
    def from_rule(
        cls,
        logical_node_ids: tuple[NodeId, ...] | NodeId,
        rule_name: str,
        **metadata: Any,
    ) -> Provenance:
        """
        Create provenance for a node created by an optimization rule.

        Parameters
        ----------
        logical_node_ids : tuple[NodeId, ...] | NodeId
            The logical node ID(s) this physical node implements.
        rule_name : str
            Name of the optimization rule.
        **metadata : Any
            Additional metadata to attach.

        Returns
        -------
        Provenance
            Provenance with rule information.
        """
        if isinstance(logical_node_ids, int):
            logical_node_ids = (logical_node_ids,)
        return cls(
            logical_node_ids=logical_node_ids,
            rule_name=rule_name,
            rule_metadata=dict(metadata) if metadata else {},
        )

    @classmethod
    def synthetic(cls, rule_name: str, **metadata: Any) -> Provenance:
        """
        Create provenance for a synthetic node (no logical origin).

        Parameters
        ----------
        rule_name : str
            Name of the rule that created this node.
        **metadata : Any
            Additional metadata.

        Returns
        -------
        Provenance
            Provenance with no logical origin.
        """
        return cls(
            logical_node_ids=(),
            rule_name=rule_name,
            rule_metadata=dict(metadata) if metadata else {},
        )