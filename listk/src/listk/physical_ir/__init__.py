"""
SoliceDB Physical Intermediate Representation (IR).

This module provides the physical IR for representing executable query plans.
Unlike the logical IR (which describes WHAT to compute), the physical IR
describes HOW to compute it, with specific algorithm choices and execution
strategies.

Key design principles:
1. Immutability: Physical nodes and plans are frozen, enabling safe optimization.
2. Separation from runtime state: Execution statistics are held separately.
3. Provenance tracking: Each physical node tracks its logical origin and
   the optimization rule that created it.
4. Composability: Complex strategies (e.g., SummarizeThenJoin) are subgraphs
   of simpler physical nodes.

Module Structure
----------------
- nodes.py: Physical operator node definitions
- plan.py: PhysicalPlan container
- properties.py: Physical properties (ordering, partitioning)
- annotations.py: Cost estimates and provenance tracking
"""

# Annotations and properties
from solicedb.physical_ir.annotations import (
    CostEstimate,
    Provenance,
)
from solicedb.physical_ir.properties import (
    Ordering,
    Partitioning,
    PhysicalProperties,
    SortOrder,
)

# Base node and classical operators
from solicedb.physical_ir.nodes import (
    # Type alias
    PhysicalNodeId,
    # Base class
    PhysicalNode,
    # Source
    TableScan,
    # Classical operators
    Filter,
    Project,
    HashJoin,
    SortMergeJoin,
    NestedLoopJoin,
    HashAggregate,
    SortAggregate,
    Sort,
    Limit,
    HashDistinct,
    SortDistinct,
    # Materialization
    Materialize,
)

# Physical plan container
from solicedb.physical_ir.plan import PhysicalPlan

__all__ = [
    # Annotations
    "CostEstimate",
    "Provenance",
    # Properties
    "Ordering",
    "Partitioning",
    "PhysicalProperties",
    "SortOrder",
    # Node types
    "PhysicalNodeId",
    "PhysicalNode",
    "TableScan",
    "Filter",
    "Project",
    "HashJoin",
    "SortMergeJoin",
    "NestedLoopJoin",
    "HashAggregate",
    "SortAggregate",
    "Sort",
    "Limit",
    "HashDistinct",
    "SortDistinct",
    "Materialize",
    # Plan
    "PhysicalPlan",
]