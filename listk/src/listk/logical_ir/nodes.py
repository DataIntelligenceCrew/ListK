"""
Logical operator nodes for the SoliceDB logical IR.

This module defines all logical operators in the query plan DAG:
- Source: Data source (table/DataFrame reference)
- Classical operators: Select, Project, Join, GroupBy, OrderBy, Distinct
- Semantic operators: SemanticFilter, SemanticProject, SemanticJoin,
                      SemanticGroupBy, SemanticSummarize, SemanticTopK, SemanticFill

All nodes are immutable (frozen dataclasses) and reference inputs by NodeId.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto

from solicedb.logical_ir import and_, gt, col, lit, ge, sub, count, avg, eq
from solicedb.logical_ir.expressions import Aggregation, Expr
from solicedb.logical_ir.hints import (
    SemanticFilterHints,
    SemanticFillHints,
    SemanticGroupByHints,
    SemanticJoinHints,
    SemanticProjectHints,
    SemanticSummarizeHints,
    SemanticTopKHints,
)
from solicedb.logical_ir.specs import SemanticSpec
from solicedb.logical_ir.types import DType, Schema



# Type alias for node identifiers
NodeId = int


class JoinType(Enum):
    """
    Types of join operations.
    """

    INNER = auto()
    LEFT = auto()
    RIGHT = auto()
    FULL = auto()
    CROSS = auto()


class SortDirection(Enum):
    """
    Sort direction for ORDER BY operations.
    """

    ASC = auto()
    DESC = auto()


# ============================================================================
# Base Node Class
# ============================================================================


@dataclass(frozen=True)
class LogicalNode(ABC):
    """
    Abstract base class for all logical operator nodes.

    Every node has a unique ID (assigned by QueryPlan) and references
    to its input nodes by their IDs.

    Parameters
    ----------
    node_id : NodeId
        Unique identifier for this node within the query plan.
    input_ids : tuple[NodeId, ...]
        IDs of input nodes. Empty for Source, one for unary operators,
        two for binary operators (e.g., Join).
    """

    node_id: NodeId
    input_ids: tuple[NodeId, ...]

    def is_source(self) -> bool:
        """
        Check if this is a source node (no inputs).

        Returns
        -------
        bool
            True if this node has no inputs.
        """
        return len(self.input_ids) == 0

    def is_unary(self) -> bool:
        """
        Check if this is a unary operator (one input).

        Returns
        -------
        bool
            True if this node has exactly one input.
        """
        return len(self.input_ids) == 1

    def is_binary(self) -> bool:
        """
        Check if this is a binary operator (two inputs).

        Returns
        -------
        bool
            True if this node has exactly two inputs.
        """
        return len(self.input_ids) == 2


# ============================================================================
# Source Node
# ============================================================================


@dataclass(frozen=True)
class Source(LogicalNode):
    """
    Data source node representing input data.

    The source can have an optional schema for validation. If not provided,
    the schema is inferred from the DataFrame at execution time.

    Parameters
    ----------
    node_id : NodeId
        Unique identifier for this node.
    input_ids : tuple[NodeId, ...]
        Always empty for source nodes.
    ref : str
        Reference name to resolve the DataFrame at execution time.
    schema : Schema | None
        Optional schema for validation. If provided, the actual DataFrame
        must be compatible with this schema.

    Examples
    --------
    >>> source = Source(
    ...     node_id=0,
    ...     input_ids=(),
    ...     ref="movies",
    ...     schema=Schema.from_dict({"title": DType.STRING, "year": DType.INT})
    ... )
    """

    ref: str
    schema: Schema | None = None


# ============================================================================
# Classical Operators
# ============================================================================


@dataclass(frozen=True)
class Select(LogicalNode):
    """
    Classical row filtering (WHERE clause).

    Filters rows based on a classical predicate expression.

    Parameters
    ----------
    node_id : NodeId
        Unique identifier for this node.
    input_ids : tuple[NodeId, ...]
        Single input node ID.
    predicate : Expr
        Boolean expression to filter rows.

    Examples
    --------
    >>> # WHERE year > 2000 AND rating >= 4.0
    >>> select = Select(
    ...     node_id=1,
    ...     input_ids=(0,),
    ...     predicate=and_(gt(col("year"), lit(2000)), ge(col("rating"), lit(4.0)))
    ... )
    """

    predicate: Expr


@dataclass(frozen=True)
class Project(LogicalNode):
    """
    Classical column projection and derivation.

    Selects a subset of columns and/or creates new computed columns.

    Parameters
    ----------
    node_id : NodeId
        Unique identifier for this node.
    input_ids : tuple[NodeId, ...]
        Single input node ID.
    columns : tuple[str | tuple[str, Expr], ...]
        Columns to include. Each element is either:
        - A string (column name to pass through)
        - A tuple (new_name, expression) for computed columns

    Examples
    --------
    >>> # SELECT title, year, revenue - budget AS profit
    >>> project = Project(
    ...     node_id=1,
    ...     input_ids=(0,),
    ...     columns=("title", "year", ("profit", sub(col("revenue"), col("budget"))))
    ... )
    """

    columns: tuple[str | tuple[str, Expr], ...]


@dataclass(frozen=True)
class Join(LogicalNode):
    """
    Classical join operation.

    Joins two relations based on a classical predicate.

    Parameters
    ----------
    node_id : NodeId
        Unique identifier for this node.
    input_ids : tuple[NodeId, ...]
        Two input node IDs (left, right).
    join_type : JoinType
        Type of join (INNER, LEFT, RIGHT, FULL, CROSS).
    condition : Expr | None
        Join condition expression. None for CROSS join.

    Examples
    --------
    >>> # INNER JOIN ON movies.id = ratings.movie_id
    >>> join = Join(
    ...     node_id=2,
    ...     input_ids=(0, 1),
    ...     join_type=JoinType.INNER,
    ...     condition=eq(col("id", "movies"), col("movie_id", "ratings"))
    ... )
    """

    join_type: JoinType
    condition: Expr | None = None


@dataclass(frozen=True)
class GroupBy(LogicalNode):
    """
    Classical grouping and aggregation.

    Groups rows by specified columns and applies aggregation functions.

    Parameters
    ----------
    node_id : NodeId
        Unique identifier for this node.
    input_ids : tuple[NodeId, ...]
        Single input node ID.
    group_columns : tuple[str, ...]
        Columns to group by.
    aggregations : tuple[tuple[str, Aggregation], ...]
        Aggregations to compute as (output_name, aggregation) pairs.

    Examples
    --------
    >>> # GROUP BY genre, year WITH COUNT(*), AVG(rating)
    >>> groupby = GroupBy(
    ...     node_id=1,
    ...     input_ids=(0,),
    ...     group_columns=("genre", "year"),
    ...     aggregations=(("count", count()), ("avg_rating", avg("rating")))
    ... )
    """

    group_columns: tuple[str, ...]
    aggregations: tuple[tuple[str, Aggregation], ...]


@dataclass(frozen=True)
class OrderBy(LogicalNode):
    """
    Classical ordering with optional limit.

    Sorts rows by specified columns with optional row limit.

    Parameters
    ----------
    node_id : NodeId
        Unique identifier for this node.
    input_ids : tuple[NodeId, ...]
        Single input node ID.
    keys : tuple[tuple[str, SortDirection], ...]
        Sort keys as (column_name, direction) pairs.
    limit : int | None
        Optional maximum number of rows to return.

    Examples
    --------
    >>> # ORDER BY year DESC, title ASC LIMIT 10
    >>> orderby = OrderBy(
    ...     node_id=1,
    ...     input_ids=(0,),
    ...     keys=(("year", SortDirection.DESC), ("title", SortDirection.ASC)),
    ...     limit=10
    ... )
    """

    keys: tuple[tuple[str, SortDirection], ...]
    limit: int | None = None


@dataclass(frozen=True)
class Distinct(LogicalNode):
    """
    Remove duplicate rows.

    Removes duplicate rows based on specified columns or all columns.

    Parameters
    ----------
    node_id : NodeId
        Unique identifier for this node.
    input_ids : tuple[NodeId, ...]
        Single input node ID.
    columns : tuple[str, ...] | None
        Columns to consider for uniqueness. None means all columns.

    Examples
    --------
    >>> # SELECT DISTINCT genre, year
    >>> distinct = Distinct(
    ...     node_id=1,
    ...     input_ids=(0,),
    ...     columns=("genre", "year")
    ... )
    """

    columns: tuple[str, ...] | None = None


# ============================================================================
# Semantic Operators
# ============================================================================


@dataclass(frozen=True)
class SemanticFilter(LogicalNode):
    """
    Semantic row filtering using natural language predicate.

    Filters rows based on an LLM-evaluated natural language condition.

    Parameters
    ----------
    node_id : NodeId
        Unique identifier for this node.
    input_ids : tuple[NodeId, ...]
        Single input node ID.
    spec : SemanticSpec
        Natural language specification for the filter condition.
    hints : SemanticFilterHints | None
        Optional execution hints.

    Examples
    --------
    >>> # Filter for "movies that are considered classics"
    >>> sem_filter = SemanticFilter(
    ...     node_id=1,
    ...     input_ids=(0,),
    ...     spec=SemanticSpec.parse("Is {title} from {year} considered a classic?"),
    ...     hints=SemanticFilterHints(batch_size=32)
    ... )
    """

    spec: SemanticSpec
    hints: SemanticFilterHints | None = None


@dataclass(frozen=True)
class SemanticProject(LogicalNode):
    """
    Semantic column derivation using natural language.

    Derives new columns by applying an LLM to each row.

    Parameters
    ----------
    node_id : NodeId
        Unique identifier for this node.
    input_ids : tuple[NodeId, ...]
        Single input node ID.
    spec : SemanticSpec
        Natural language specification for deriving the output.
    output_schema : Schema
        Schema of the output columns to be derived.
    hints : SemanticProjectHints | None
        Optional execution hints.

    Examples
    --------
    >>> # Derive genre and mood from title and synopsis
    >>> sem_project = SemanticProject(
    ...     node_id=1,
    ...     input_ids=(0,),
    ...     spec=SemanticSpec.parse("Based on {title} and {synopsis}, classify:"),
    ...     output_schema=Schema.from_dict({"genre": DType.STRING, "mood": DType.STRING}),
    ...     hints=SemanticProjectHints(batch_size=16)
    ... )
    """

    spec: SemanticSpec
    output_schema: Schema
    hints: SemanticProjectHints | None = None


@dataclass(frozen=True)
class SemanticJoin(LogicalNode):
    """
    Semantic join using natural language condition.

    Joins two relations based on an LLM-evaluated semantic condition.

    Parameters
    ----------
    node_id : NodeId
        Unique identifier for this node.
    input_ids : tuple[NodeId, ...]
        Two input node IDs (left, right).
    spec : SemanticSpec
        Natural language specification for the join condition.
    join_type : JoinType
        Type of join (typically INNER or LEFT for semantic joins).
    output_schema : Schema | None
        Optional additional output columns derived during join.
        If None, output is the concatenation of left and right schemas.
    hints : SemanticJoinHints | None
        Optional execution hints.

    Examples
    --------
    >>> # Join products to reviews where review describes the product
    >>> sem_join = SemanticJoin(
    ...     node_id=2,
    ...     input_ids=(0, 1),
    ...     spec=SemanticSpec.parse("{review_text} describes {product_name}"),
    ...     join_type=JoinType.INNER,
    ...     hints=SemanticJoinHints(blocking_strategy="embedding")
    ... )
    """

    spec: SemanticSpec
    join_type: JoinType
    output_schema: Schema | None = None
    hints: SemanticJoinHints | None = None


@dataclass(frozen=True)
class SemanticGroupBy(LogicalNode):
    """
    Semantic grouping using natural language criterion.

    Groups rows using an LLM-determined grouping criterion.
    This assigns each row to a group without aggregation.

    Parameters
    ----------
    node_id : NodeId
        Unique identifier for this node.
    input_ids : tuple[NodeId, ...]
        Single input node ID.
    spec : SemanticSpec
        Natural language specification for the grouping criterion.
    output_group_column : str
        Name of the output column containing group assignments.
    hints : SemanticGroupByHints | None
        Optional execution hints.

    Examples
    --------
    >>> # Group reviews by sentiment
    >>> sem_groupby = SemanticGroupBy(
    ...     node_id=1,
    ...     input_ids=(0,),
    ...     spec=SemanticSpec.parse("Classify {review_text} as positive, negative, or neutral"),
    ...     output_group_column="sentiment",
    ...     hints=SemanticGroupByHints(num_groups=3)
    ... )
    """

    spec: SemanticSpec
    output_group_column: str
    hints: SemanticGroupByHints | None = None


@dataclass(frozen=True)
class SemanticSummarize(LogicalNode):
    """
    Semantic summarization of grouped rows.

    Summarizes rows within each group using an LLM.

    Parameters
    ----------
    node_id : NodeId
        Unique identifier for this node.
    input_ids : tuple[NodeId, ...]
        Single input node ID.
    group_columns : tuple[str, ...]
        Columns to group by (classical grouping).
    spec : SemanticSpec
        Natural language specification for summarization.
    output_schema : Schema
        Schema of the summary output columns.
    hints : SemanticSummarizeHints | None
        Optional execution hints.

    Examples
    --------
    >>> # Summarize all reviews per product
    >>> sem_summarize = SemanticSummarize(
    ...     node_id=1,
    ...     input_ids=(0,),
    ...     group_columns=("product_id",),
    ...     spec=SemanticSpec.parse("Summarize these reviews: {ALL_COLS}"),
    ...     output_schema=Schema.from_dict({"summary": DType.STRING}),
    ...     hints=SemanticSummarizeHints(max_tokens=500)
    ... )
    """

    group_columns: tuple[str, ...]
    spec: SemanticSpec
    output_schema: Schema
    hints: SemanticSummarizeHints | None = None


@dataclass(frozen=True)
class SemanticTopK(LogicalNode):
    """
    Semantic top-k selection using natural language ordering.

    Selects the top k rows based on an LLM-evaluated ordering criterion.

    Parameters
    ----------
    node_id : NodeId
        Unique identifier for this node.
    input_ids : tuple[NodeId, ...]
        Single input node ID.
    spec : SemanticSpec
        Natural language specification for the ordering criterion.
    k : int
        Number of top rows to return.
    hints : SemanticTopKHints | None
        Optional execution hints.

    Examples
    --------
    >>> # Get top 10 most influential papers
    >>> sem_topk = SemanticTopK(
    ...     node_id=1,
    ...     input_ids=(0,),
    ...     spec=SemanticSpec.parse("Rank by influence: {title}, {abstract}"),
    ...     k=10,
    ...     hints=SemanticTopKHints(method="multipivot", num_pivots=3)
    ... )
    """

    spec: SemanticSpec
    k: int
    hints: SemanticTopKHints | None = None


@dataclass(frozen=True)
class SemanticFill(LogicalNode):
    """
    Semantic NULL value filling using natural language.

    Fills NULL values in a column using LLM-generated values based on
    other columns in the row.

    Parameters
    ----------
    node_id : NodeId
        Unique identifier for this node.
    input_ids : tuple[NodeId, ...]
        Single input node ID.
    target_column : str
        Column to fill (can be existing or new).
    spec : SemanticSpec
        Natural language specification for generating fill values.
    output_dtype : DType
        Data type of the filled/new column.
    hints : SemanticFillHints | None
        Optional execution hints.

    Examples
    --------
    >>> # Fill missing genres based on title and synopsis
    >>> sem_fill = SemanticFill(
    ...     node_id=1,
    ...     input_ids=(0,),
    ...     target_column="genre",
    ...     spec=SemanticSpec.parse("Infer genre from {title} and {synopsis}"),
    ...     output_dtype=DType.STRING,
    ...     hints=SemanticFillHints(batch_size=32)
    ... )
    """

    target_column: str
    spec: SemanticSpec
    output_dtype: DType
    hints: SemanticFillHints | None = None
