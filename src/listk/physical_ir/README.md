# SoliceDB Physical Intermediate Representation (IR)

This module defines the physical IR for SoliceDB query plans. While the logical IR describes **WHAT** to compute, the physical IR describes **HOW** to compute it with specific algorithms and execution strategies.

## Design Principles

1. **Immutability**: Physical nodes and plans are frozen dataclasses, enabling safe sharing during optimization passes.

2. **Separation from Runtime State**: The physical plan describes execution strategy. Runtime statistics (actual row counts, timing, etc.) are tracked separately in `ExecutionState` by the engine.

3. **Provenance Tracking**: Each physical node tracks:
   - Which logical node(s) it implements
   - Which optimization rule created it
   - Optional metadata explaining the decision

4. **Composability**: Complex strategies (e.g., `SummarizeThenJoin`) are implemented as subgraphs of simpler physical nodes, not monolithic operators.

## Module Structure

```
physical_ir/
├── __init__.py          # Public exports
├── nodes.py             # Physical node definitions (classical operators)
├── plan.py              # PhysicalPlan container
├── properties.py        # Physical properties (ordering, partitioning)
├── annotations.py       # Cost estimates, provenance
├── README.md            # This file
└── semantic/            # Semantic physical operators (to be added)
```

## Quick Start

```python
from solicedb.physical_ir import (
    PhysicalPlan,
    TableScan, Filter, HashJoin, Sort, Limit,
    Provenance, CostEstimate,
    Ordering, SortOrder,
)
from solicedb.logical_ir import col, lit, gt, JoinType

# Create a physical plan
plan = PhysicalPlan(name="example_query")

# Add a table scan
movies = plan.add(TableScan(
    node_id=-1,
    input_ids=(),
    ref="movies",
    columns=("id", "title", "year"),  # projection pushdown
    provenance=Provenance.direct(logical_node_id=0),
))

# Add a filter
filtered = plan.add(Filter(
    node_id=-1,
    input_ids=(movies,),
    predicate=gt(col("year"), lit(2000)),
    cost=CostEstimate(row_count=500, cpu_cost=0.1),
    provenance=Provenance.direct(logical_node_id=1),
))

# Set root and visualize
plan.set_root(filtered)
print(plan.to_tree_str())
```

## Physical vs Logical: Key Differences

| Aspect | Logical IR | Physical IR |
|--------|-----------|-------------|
| Purpose | WHAT to compute | HOW to compute |
| Join | Single `Join` node | `HashJoin`, `SortMergeJoin`, `NestedLoopJoin` |
| Aggregate | Single `GroupBy` node | `HashAggregate`, `SortAggregate` |
| SemanticTopK | Single node | Subgraph (e.g., pivot selection → comparisons → merge) |
| Ordering | Implicit | Explicit `Ordering` property + `Sort` nodes |
| Cost | None | `CostEstimate` on each node |

## Provenance Tracking

Every physical node can track where it came from:

```python
# Direct translation (no optimization)
prov = Provenance.direct(logical_node_id=3)

# Created by an optimization rule
prov = Provenance.from_rule(
    logical_node_ids=(5,),
    rule_name="SummarizeThenJoinRule",
    reason="tables exceed 10K tokens",
)

# Synthetic node (no logical origin)
prov = Provenance.synthetic(
    rule_name="EnforceSortForMergeJoin",
    target_join_id=7,
)
```

## Physical Properties

Physical properties describe characteristics of operator output:

```python
from solicedb.physical_ir import Ordering, SortOrder, PhysicalProperties

# Output is sorted by year DESC, then title ASC
props = PhysicalProperties(
    ordering=Ordering.by(("year", SortOrder.DESC), "title"),
    row_count_estimate=1000,
)

# Check if ordering satisfies a requirement
required = Ordering.by("year")
props.ordering.satisfies(required)  # True (prefix match)
```

## Visualization

### ASCII Tree

```python
print(plan.to_tree_str(show_cost=True))
```

Output:
```
`-- Limit (n=10) [~10 rows]
    `-- Sort (by [year])
        `-- HashJoin (INNER on ('id',)=('movie_id',)) [~5000 rows]
            |-- Filter (predicate=...) [~500 rows]
            |   `-- TableScan (movies)
            `-- TableScan (ratings)
```

### Graphviz DOT

```python
dot_str = plan.to_dot()
# Render with: dot -Tpng -o plan.png
```

## Cost Model

Cost estimates are optional annotations on each node:

```python
cost = CostEstimate(
    token_count=5000,      # LLM tokens (semantic ops)
    llm_calls=10,          # Number of LLM API calls
    row_count=100,         # Output cardinality
    cpu_cost=0.5,          # Relative CPU cost
    memory_bytes=1024*1024, # Peak memory
    latency_ms=500,        # Wall-clock time
)
```

The optimizer uses these for cost-based plan selection.

## Semantic Operators

Semantic physical operators will be defined in `physical_ir/semantic/`. These implement various strategies for semantic operations:

- **SemanticFilter**: Pointwise, batched, with/without chain-of-thought
- **SemanticTopK**: MultiPivot, tournament, heap-based
- **SemanticJoin**: Brute-force, blocking, summarize-then-join
- **SemanticProject**: Pointwise, batched
- etc.

Each logical semantic operator can map to multiple physical strategies, selected by the optimizer based on data characteristics and cost model.

## Integration with Execution Engine

The physical plan is executed by the engine, which:

1. Traverses in topological order
2. Maintains `ExecutionState` (separate from the plan)
3. Collects runtime statistics
4. Supports checkpointing for adaptive re-optimization

```python
# Conceptual execution flow
engine = Executor(plan, context)
result = engine.execute()

# Access runtime stats
stats = engine.state.get_stats(node_id)
print(f"Actual rows: {stats.actual_row_count}")
```

## Future Work

- [ ] Semantic physical operators (`physical_ir/semantic/`)
- [ ] Plan deserialization (`from_json()`)
- [ ] Plan diff for comparing optimization results
- [ ] Subgraph extraction and templates for composed strategies