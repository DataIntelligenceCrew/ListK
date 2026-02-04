# Physical IR Design Notes (Work in Progress)

This document captures design decisions and next steps for the physical IR implementation.

## Completed

### Core Infrastructure
- `annotations.py` - CostEstimate, Provenance (tracks logical origin + optimization rule)
- `properties.py` - Ordering, Partitioning, PhysicalProperties
- `nodes.py` - Classical physical operators (TableScan, Filter, Project, HashJoin, SortMergeJoin, NestedLoopJoin, HashAggregate, SortAggregate, Sort, Limit, HashDistinct, SortDistinct, Materialize)
- `plan.py` - PhysicalPlan container with visualization (ASCII tree, DOT export)
- `README.md` - Documentation

### Key Design Decisions Made

1. **Immutable plans, mutable runtime state**: Physical nodes and plans are frozen dataclasses. Runtime statistics are held separately in `ExecutionState` by the engine (not yet implemented).

2. **Provenance tracking**: Each physical node tracks:
   - `logical_node_ids`: Which logical node(s) it implements
   - `rule_name`: Which optimization rule created it
   - `rule_metadata`: Additional context (e.g., why this choice was made)

3. **LLM module boundary**: The `llm/` module handles:
   - Individual LLM calls
   - Table-level primitives (pointwise/listwise batch calls)

   The `physical_ir/` operates above that, composing LLM primitives to implement operators.

4. **SemanticSpec handling**: User's NL template is preserved as-is. System/compiler can inject additional instructions on top, but the original template remains unchanged for intent preservation.

5. **Physical node inheritance**: Uses `kw_only=True` in base dataclass to allow child classes to have required fields after parent's optional fields.

---

## Next Steps: Semantic Physical Operators

### Directory Structure (both by-pattern AND by-logical-operator)
```
physical_ir/semantic/
├── __init__.py
├── base.py              # SemanticPhysicalNode base class
│
├── # By pattern (building blocks)
├── pointwise.py         # PointwiseFilter, PointwiseProject, PointwiseFill
├── listwise.py          # ListwiseRank, ListwiseCompare
├── pairwise.py          # PairwiseCompare, PairwiseJoinPredicate
├── summarize.py         # RowSummarize, GroupSummarize, TableSummarize
├── cluster.py           # LLMCluster
│
├── # Composed strategies (subgraphs)
├── composed/
│   ├── __init__.py
│   ├── topk.py          # MultiPivotTopK, TournamentTopK, HeapTopK
│   ├── join.py          # SummarizeThenJoin, BlockingJoin
│   └── ...
```

### Proposed Operators (NEEDS REVIEW)

#### Pointwise Operators
| Operator | Implements | Description |
|----------|-----------|-------------|
| `PointwiseFilter` | SemanticFilter | Evaluate predicate on each row independently |
| `PointwiseProject` | SemanticProject | Derive columns for each row independently |
| `PointwiseFill` | SemanticFill | Fill NULL values row by row |

#### Batched Pointwise
| Operator | Implements | Description |
|----------|-----------|-------------|
| `BatchedFilter` | SemanticFilter | Multiple rows in one prompt, independent decisions |
| `BatchedProject` | SemanticProject | Multiple rows in one prompt |

#### Listwise Operators
| Operator | Implements | Description |
|----------|-----------|-------------|
| `ListwiseRank` | SemanticTopK, SemanticOrderBy | Rank a window of rows in one call |
| `ListwiseCompare` | (building block) | Compare items against reference(s) |

#### Pairwise Operators
| Operator | Implements | Description |
|----------|-----------|-------------|
| `PairwiseJoinPredicate` | SemanticJoin | Evaluate join condition on row pairs |

#### Summarization Operators
| Operator | Implements | Description |
|----------|-----------|-------------|
| `RowSummarize` | (building block) | Summarize a single row into shorter text |
| `GroupSummarize` | SemanticSummarize | Summarize rows within a group |
| `TableSummarize` | (building block) | Summarize entire table/partition |

#### Clustering Operators
| Operator | Implements | Description |
|----------|-----------|-------------|
| `LLMCluster` | SemanticGroupBy | LLM assigns rows to groups |

#### Composed Strategies
| Operator | Implements | Description |
|----------|-----------|-------------|
| `MultiPivotTopK` | SemanticTopK | Pivot selection → partition → recurse |
| `TournamentTopK` | SemanticTopK | Tournament-style elimination |
| `SummarizeThenJoin` | SemanticJoin | Summarize → LLM predicate → classical join |
| `BlockingJoin` | SemanticJoin | Blocking → LLM verification |

### Open Questions (for next session)

1. **Are there other SemanticTopK strategies?** (beyond MultiPivot, Tournament)

2. **Are there other SemanticJoin patterns?** (beyond SummarizeThenJoin, Blocking)

3. **Should embedding-based operators be included?** (e.g., `EmbeddingSimilarityFilter`)
   - Could be useful as blocking/pre-filtering step
   - Might belong in a separate module

4. **What about SemanticOrderBy?**
   - Currently not in logical IR
   - Would use same physical operators as SemanticTopK but without the k limit

5. **Chain-of-thought variants?**
   - Should CoT be a flag on operators, or separate operator types?
   - e.g., `PointwiseFilterCoT` vs `PointwiseFilter(use_cot=True)`

---

## Architecture Reminders

### Data Flow
```
User API
    ↓
Logical IR (QueryPlan) ← declarative WHAT
    ↓
Optimizer (rules, cost model)
    ↓
Physical IR (PhysicalPlan) ← imperative HOW
    ↓
Execution Engine
    ↓
LLM Module (pointwise/listwise batch calls)
    ↓
vLLM
```

### Key Insight: Physical Subgraphs
One logical operator can expand into multiple physical nodes. Example:

```
Logical: SemanticJoin(left, right, "{left.desc} matches {right.desc}")

Physical (SummarizeThenJoin):
├── TableSummarize(left)
├── TableSummarize(right)
├── PairwiseJoinPredicate(summarized_left, summarized_right)
└── ClassicalSemiJoin(left, right, matching_pairs)
```

### Reuse Patterns
- Summarization is useful for: SemanticJoin, SemanticAggregate, SemanticOrderBy
- ListwiseCompare is useful for: SemanticTopK, SemanticOrderBy
- Blocking is useful for: SemanticJoin, SemanticGroupBy

---

## Files Modified/Created This Session

```
src/solicedb/physical_ir/
├── __init__.py      (new)
├── annotations.py   (new)
├── properties.py    (new)
├── nodes.py         (new)
├── plan.py          (new)
├── README.md        (new)
└── DESIGN_NOTES.md  (this file)
```

## Testing

Basic import and usage test passed:
```python
from solicedb.physical_ir import (
    PhysicalPlan, TableScan, Filter, HashJoin,
    Provenance, CostEstimate, Ordering, SortOrder,
)
# Creates plan, adds nodes, visualizes as tree - all working
```
