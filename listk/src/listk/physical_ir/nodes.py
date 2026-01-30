"""
Physical operator nodes for the SoliceDB physical IR.

This module defines physical operators that describe HOW to execute a query.
These specify concrete algorithms and execution strategies.

All nodes are immutable and reference inputs by ID.

Node Categories
---------------
1. Source: TableScan
2. Classical operators: Filter, Project, Join variants, Aggregate variants, etc.
3. Utility: Materialize, Sort, Limit

Semantic operators are defined in the physical_ir/semantic/ submodule.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum, auto

from solicedb.logical_ir.expressions import Aggregation, Expr
from solicedb.logical_ir.nodes import JoinType
from solicedb.logical_ir.types import Schema
from solicedb.physical_ir.annotations import CostEstimate, Provenance
from solicedb.physical_ir.properties import Ordering, PhysicalProperties


# Type alias for physical node identifiers
PhysicalNodeId = int


# ============================================================================
# Base Node Class
# ============================================================================


@dataclass(frozen=True, kw_only=True)
class PhysicalNode(ABC):
    """
    Abstract base class for all physical operator nodes.

    Every physical node has:
    - A unique ID (assigned by PhysicalPlan)
    - References to input nodes by their IDs
    - Optional cost estimate (set during planning)
    - Optional physical properties of output (set during planning)
    - Provenance tracking (where this node came from)

    Parameters
    ----------
    node_id : PhysicalNodeId
        Unique identifier for this node within the physical plan.
    input_ids : tuple[PhysicalNodeId, ...]
        IDs of input nodes. Empty for source nodes.
    cost : CostEstimate | None
        Estimated execution cost.
    output_properties : PhysicalProperties | None
        Physical properties of this operator's output.
    provenance : Provenance | None
        Tracks logical origin and optimization rule.

    Notes
    -----
    All physical nodes use keyword-only arguments (kw_only=True) to avoid
    issues with dataclass inheritance when parent has optional fields.
    """

    node_id: PhysicalNodeId
    input_ids: tuple[PhysicalNodeId, ...]
    cost: CostEstimate | None = None
    output_properties: PhysicalProperties | None = None
    provenance: Provenance | None = None

    def is_source(self) -> bool:
        """Check if this is a source node (no inputs)."""
        return len(self.input_ids) == 0

    def is_unary(self) -> bool:
        """Check if this is a unary operator (one input)."""
        return len(self.input_ids) == 1

    def is_binary(self) -> bool:
        """Check if this is a binary operator (two inputs)."""
        return len(self.input_ids) == 2


# ============================================================================
# Source Node
# ============================================================================


@dataclass(frozen=True)
class TableScan(PhysicalNode):
    """
    Scan a table/DataFrame by reference name.

    This is the physical counterpart to logical Source.

    Parameters
    ----------
    ref : str
        Reference name to resolve the DataFrame at execution time.
    schema : Schema | None
        Optional schema for validation.
    columns : tuple[str, ...] | None
        If specified, only these columns are read (projection pushdown).

    Examples
    --------
    >>> scan = TableScan(
    ...     node_id=0,
    ...     input_ids=(),
    ...     ref="movies",
    ...     columns=("title", "year"),  # Only read these columns
    ... )
    """

    ref: str
    schema: Schema | None = None
    columns: tuple[str, ...] | None = None


# ============================================================================
# Classical Filter and Project
# ============================================================================


@dataclass(frozen=True)
class Filter(PhysicalNode):
    """
    Filter rows based on a classical predicate.

    Physical counterpart to logical Select.

    Parameters
    ----------
    predicate : Expr
        Boolean expression to filter rows.

    Examples
    --------
    >>> filt = Filter(
    ...     node_id=1,
    ...     input_ids=(0,),
    ...     predicate=gt(col("year"), lit(2000)),
    ... )
    """

    predicate: Expr


@dataclass(frozen=True)
class Project(PhysicalNode):
    """
    Project columns and/or compute derived columns.

    Physical counterpart to logical Project.

    Parameters
    ----------
    columns : tuple[str | tuple[str, Expr], ...]
        Columns to include. Each element is either:
        - A string (column name to pass through)
        - A tuple (new_name, expression) for computed columns

    Examples
    --------
    >>> proj = Project(
    ...     node_id=1,
    ...     input_ids=(0,),
    ...     columns=("title", ("decade", div(col("year"), lit(10)))),
    ... )
    """

    columns: tuple[str | tuple[str, Expr], ...]


# ============================================================================
# Join Operators
# ============================================================================


@dataclass(frozen=True)
class HashJoin(PhysicalNode):
    """
    Hash join implementation.

    Builds a hash table on the right input, probes with the left input.
    Best for equi-joins when one side fits in memory.

    Parameters
    ----------
    join_type : JoinType
        Type of join (INNER, LEFT, etc.).
    left_keys : tuple[str, ...]
        Join key columns from left input.
    right_keys : tuple[str, ...]
        Join key columns from right input.
    condition : Expr | None
        Additional non-equi join condition (applied after hash match).
    build_side : str
        Which side to build hash table from: "left" or "right".

    Notes
    -----
    The first input_id is left, second is right.
    """

    join_type: JoinType
    left_keys: tuple[str, ...]
    right_keys: tuple[str, ...]
    condition: Expr | None = None
    build_side: str = "right"


@dataclass(frozen=True)
class SortMergeJoin(PhysicalNode):
    """
    Sort-merge join implementation.

    Requires both inputs to be sorted on join keys.
    Good for large datasets that are already sorted or when result needs ordering.

    Parameters
    ----------
    join_type : JoinType
        Type of join (INNER, LEFT, etc.).
    left_keys : tuple[str, ...]
        Join key columns from left input.
    right_keys : tuple[str, ...]
        Join key columns from right input.
    condition : Expr | None
        Additional non-equi join condition.

    Notes
    -----
    The optimizer should ensure inputs are sorted appropriately,
    inserting Sort operators if needed.
    """

    join_type: JoinType
    left_keys: tuple[str, ...]
    right_keys: tuple[str, ...]
    condition: Expr | None = None


@dataclass(frozen=True)
class NestedLoopJoin(PhysicalNode):
    """
    Nested loop join implementation.

    Iterates over all pairs of rows. Only practical for small inputs
    or when no better join strategy is available (e.g., non-equi joins).

    Parameters
    ----------
    join_type : JoinType
        Type of join (INNER, LEFT, etc.).
    condition : Expr | None
        Join condition. None for CROSS join.
    """

    join_type: JoinType
    condition: Expr | None = None


# ============================================================================
# Aggregation Operators
# ============================================================================


@dataclass(frozen=True)
class HashAggregate(PhysicalNode):
    """
    Hash-based aggregation.

    Uses a hash table to group rows and compute aggregates.
    Good when number of groups is manageable.

    Parameters
    ----------
    group_columns : tuple[str, ...]
        Columns to group by. Empty for global aggregation.
    aggregations : tuple[tuple[str, Aggregation], ...]
        Aggregations as (output_name, aggregation) pairs.
    """

    group_columns: tuple[str, ...]
    aggregations: tuple[tuple[str, Aggregation], ...]


@dataclass(frozen=True)
class SortAggregate(PhysicalNode):
    """
    Sort-based aggregation.

    Requires input to be sorted by group columns.
    Good when input is already sorted or for streaming aggregation.

    Parameters
    ----------
    group_columns : tuple[str, ...]
        Columns to group by (must match input sort order).
    aggregations : tuple[tuple[str, Aggregation], ...]
        Aggregations as (output_name, aggregation) pairs.
    """

    group_columns: tuple[str, ...]
    aggregations: tuple[tuple[str, Aggregation], ...]


# ============================================================================
# Sort and Limit
# ============================================================================


@dataclass(frozen=True)
class Sort(PhysicalNode):
    """
    Sort rows by specified columns.

    Parameters
    ----------
    ordering : Ordering
        Sort specification (columns and directions).
    limit : int | None
        If specified, only keep top N rows after sorting (top-k optimization).

    Examples
    --------
    >>> sort = Sort(
    ...     node_id=1,
    ...     input_ids=(0,),
    ...     ordering=Ordering.by(("year", SortOrder.DESC), "title"),
    ... )
    """

    ordering: Ordering
    limit: int | None = None


@dataclass(frozen=True)
class Limit(PhysicalNode):
    """
    Limit output to first N rows.

    Parameters
    ----------
    count : int
        Maximum number of rows to return.
    offset : int
        Number of rows to skip before returning.

    Examples
    --------
    >>> # LIMIT 10 OFFSET 20
    >>> limit = Limit(node_id=1, input_ids=(0,), count=10, offset=20)
    """

    count: int
    offset: int = 0


# ============================================================================
# Distinct Operators
# ============================================================================


@dataclass(frozen=True)
class HashDistinct(PhysicalNode):
    """
    Hash-based duplicate removal.

    Uses a hash set to track seen rows.

    Parameters
    ----------
    columns : tuple[str, ...] | None
        Columns to consider for uniqueness. None means all columns.
    """

    columns: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SortDistinct(PhysicalNode):
    """
    Sort-based duplicate removal.

    Requires input to be sorted by distinct columns.
    Removes consecutive duplicates.

    Parameters
    ----------
    columns : tuple[str, ...] | None
        Columns to consider for uniqueness. None means all columns.
    """

    columns: tuple[str, ...] | None = None


# ============================================================================
# Utility Operators
# ============================================================================


@dataclass(frozen=True)
class Materialize(PhysicalNode):
    """
    Materialize intermediate results.

    Forces computation and caching of the input. Useful for:
    - Breaking pipeline for adaptive re-optimization
    - Reusing results that are consumed multiple times
    - Collecting statistics for cardinality estimation

    Parameters
    ----------
    cache_key : str | None
        Optional key for caching the result across queries.

    Examples
    --------
    >>> # Materialize before expensive semantic operation
    >>> mat = Materialize(node_id=1, input_ids=(0,), cache_key="filtered_products")
    """

    cache_key: str | None = None