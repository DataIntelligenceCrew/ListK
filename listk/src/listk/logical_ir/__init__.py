"""
SoliceDB Logical Intermediate Representation (IR).

This module provides the logical IR for representing query plans as DAGs.
It includes types, expressions, semantic specifications, hints, operator nodes,
and the query plan container.

Example Usage
-------------
>>> from solicedb.logical_ir import QueryPlan, DType, col, lit, gt, and_
>>>
>>> # Build a query plan
>>> plan = QueryPlan()
>>> movies = plan.source("movies", {"title": DType.STRING, "year": DType.INT})
>>> filtered = plan.select(movies, gt(col("year"), lit(2000)))
>>> top_movies = plan.semantic_top_k(
...     filtered,
...     template="Rank by cultural significance: {title}",
...     k=10
... )
>>> plan.set_root(top_movies)
"""

# Types
from solicedb.logical_ir.types import DType, Schema

# Expressions
from solicedb.logical_ir.expressions import (
    # Enums
    AggFunc,
    BinaryOp,
    UnaryOp,
    # AST nodes
    Aggregation,
    BinaryExpr,
    Call,
    Col,
    Expr,
    Literal,
    UnaryExpr,
    # Convenience constructors
    add,
    and_,
    avg,
    col,
    count,
    div,
    eq,
    first,
    ge,
    gt,
    last,
    le,
    lit,
    lt,
    max_,
    min_,
    mod,
    mul,
    ne,
    neg,
    not_,
    or_,
    sub,
    sum_,
)

# Semantic specifications
from solicedb.logical_ir.specs import ALL_COLUMNS_MARKER, SemanticSpec

# Hints
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

# Nodes
from solicedb.logical_ir.nodes import (
    # Type aliases and enums
    JoinType,
    NodeId,
    SortDirection,
    # Base class
    LogicalNode,
    # Source
    Source,
    # Classical operators
    Distinct,
    GroupBy,
    Join,
    OrderBy,
    Project,
    Select,
    # Semantic operators
    SemanticFilter,
    SemanticFill,
    SemanticGroupBy,
    SemanticJoin,
    SemanticProject,
    SemanticSummarize,
    SemanticTopK,
)

# Query plan
from solicedb.logical_ir.plan import QueryPlan

__all__ = [
    # Types
    "DType",
    "Schema",
    # Expression enums
    "AggFunc",
    "BinaryOp",
    "UnaryOp",
    # Expression AST
    "Aggregation",
    "BinaryExpr",
    "Call",
    "Col",
    "Expr",
    "Literal",
    "UnaryExpr",
    # Expression constructors
    "add",
    "and_",
    "avg",
    "col",
    "count",
    "div",
    "eq",
    "first",
    "ge",
    "gt",
    "last",
    "le",
    "lit",
    "lt",
    "max_",
    "min_",
    "mod",
    "mul",
    "ne",
    "neg",
    "not_",
    "or_",
    "sub",
    "sum_",
    # Semantic specs
    "ALL_COLUMNS_MARKER",
    "SemanticSpec",
    # Hints
    "GlobalHints",
    "OperatorHints",
    "SemanticFilterHints",
    "SemanticFillHints",
    "SemanticGroupByHints",
    "SemanticJoinHints",
    "SemanticProjectHints",
    "SemanticSummarizeHints",
    "SemanticTopKHints",
    # Node types
    "JoinType",
    "NodeId",
    "SortDirection",
    "LogicalNode",
    "Source",
    "Distinct",
    "GroupBy",
    "Join",
    "OrderBy",
    "Project",
    "Select",
    "SemanticFilter",
    "SemanticFill",
    "SemanticGroupBy",
    "SemanticJoin",
    "SemanticProject",
    "SemanticSummarize",
    "SemanticTopK",
    # Query plan
    "QueryPlan",
]