# SoliceDB Logical Intermediate Representation (IR)

This module defines the logical IR for SoliceDB query plans. It provides a DAG-based representation of queries that combine classical relational operators with semantic (LLM-powered) operators.

## Overview

The IR is designed with the following principles:

1. **Immutability**: All nodes and specifications are frozen dataclasses, enabling safe sharing and transformation during optimization.
2. **Container-based DAG**: A `QueryPlan` container manages nodes by ID, enabling clean serialization and complex optimization algorithms.
3. **Unified IR**: Both classical and semantic operators live in the same IR, allowing cross-boundary optimizations.
4. **Typed hints**: Optional execution hints provide control over physical execution without changing logical semantics.
5. **User-friendly API**: High-level builder methods for regular users, low-level node API for power users and optimizers.

## Module Structure

```
solicedb/ir/
├── __init__.py      # Public exports
├── types.py         # DType enum, Schema class
├── expressions.py   # Classical expression AST
├── specs.py         # SemanticSpec for NL templates
├── hints.py         # Typed execution hints
├── nodes.py         # Logical operator nodes
├── plan.py          # QueryPlan container
└── README.md        # This file
```

## Quick Start

```python
from solicedb.logical_ir import (
    QueryPlan, DType, Schema,
    col, lit, gt, and_, eq,
    JoinType, SortDirection,
    SemanticTopKHints,
)

# Create a query plan
plan = QueryPlan()

# Add a source (references a DataFrame by name)
movies = plan.source("movies", {
    "title": DType.STRING,
    "year": DType.INT,
    "synopsis": DType.STRING,
})

# Classical filter: year > 2000
recent = plan.select(movies, gt(col("year"), lit(2000)))

# Semantic top-k: rank by cultural significance
top_movies = plan.semantic_top_k(
    recent,
    template="Rank by cultural significance based on {title} and {synopsis}",
    k=10,
    hints=SemanticTopKHints(method="multipivot", num_pivots=3),
)

# Set the root (final output)
plan.set_root(top_movies)

# Validate the plan
errors = plan.validate()
if errors:
    print("Validation errors:", errors)

# Serialize to JSON
json_str = plan.to_json()
```

## Core Components

### Types (`types.py`)

#### `DType`
Enumeration of supported data types:
- `DType.STRING` - Text data
- `DType.INT` - Integer numbers
- `DType.FLOAT` - Floating-point numbers
- `DType.BOOL` - Boolean values

```python
from solicedb.logical_ir import DType

dtype = DType.STRING
dtype = DType.from_pandas_dtype(df["column"].dtype)  # Infer from DataFrame
python_type = dtype.to_python_type()  # Returns `str`
```

#### `Schema`
Ordered collection of (column_name, dtype) pairs:

```python
from solicedb.logical_ir import Schema, DType

# Create from dict
schema = Schema.from_dict({"title": DType.STRING, "year": DType.INT})

# Create from DataFrame
schema = Schema.from_dataframe(df)

# Query
schema.column_names()  # ['title', 'year']
schema.dtype_of("title")  # DType.STRING
schema.has_column("title")  # True

# Transform (returns new Schema)
schema.select(["title"])  # Project to subset
schema.add("genre", DType.STRING)  # Add column
schema.remove("year")  # Remove column
schema.rename("title", "name")  # Rename column
schema.merge(other_schema)  # For joins
```

### Expressions (`expressions.py`)

Classical expressions for predicates and computations.

#### Expression AST

| Class | Description | Example |
|-------|-------------|---------|
| `Literal` | Constant value | `Literal(42, DType.INT)` |
| `Col` | Column reference | `Col("age")`, `Col("age", table="users")` |
| `BinaryExpr` | Binary operation | `BinaryExpr(BinaryOp.GT, col, lit)` |
| `UnaryExpr` | Unary operation | `UnaryExpr(UnaryOp.NOT, expr)` |
| `Call` | Function call | `Call("ABS", (col,))` |
| `Aggregation` | Aggregation function | `Aggregation(AggFunc.SUM, "amount")` |

#### Fluent Constructors

For convenient expression building:

```python
from solicedb.logical_ir import col, lit, gt, lt, eq, and_, or_, not_, add, sub

# Column references and literals
age = col("age")
threshold = lit(18)

# Comparisons: eq, ne, lt, le, gt, ge
expr = gt(age, threshold)  # age > 18

# Boolean logic: and_, or_, not_
expr = and_(gt(col("age"), lit(18)), lt(col("age"), lit(65)))

# Arithmetic: add, sub, mul, div, mod, neg
profit = sub(col("revenue"), col("cost"))

# Aggregations: count, sum_, avg, min_, max_, first, last
total = sum_("amount")
row_count = count()  # COUNT(*)
distinct_count = count("category", distinct=True)
```

### Semantic Specifications (`specs.py`)

#### `SemanticSpec`
Represents a natural language template with column dependencies:

```python
from solicedb.logical_ir import SemanticSpec

# Parse from template string
spec = SemanticSpec.parse("Is {title} from {year} a classic film?")
spec.template  # "Is {title} from {year} a classic film?"
spec.input_columns  # frozenset({'title', 'year'})
spec.all_columns  # False

# Use {ALL_COLS} to reference all columns
spec = SemanticSpec.parse("Summarize this record: {ALL_COLS}")
spec.all_columns  # True

# Format with actual values
spec.format({"title": "The Matrix", "year": "1999"})
# "Is The Matrix from 1999 a classic film?"

# Validate column references
missing = spec.validate_columns({"title", "year", "rating"})  # []
missing = spec.validate_columns({"title"})  # ['year']
```

### Hints (`hints.py`)

Optional execution hints for controlling physical execution.

#### Per-Operator Hints

| Hint Class | For Operator | Key Fields |
|------------|--------------|------------|
| `SemanticFilterHints` | `SemanticFilter` | `batch_size`, `use_cot`, `model`, `temperature` |
| `SemanticProjectHints` | `SemanticProject` | `batch_size`, `use_cot`, `model`, `max_tokens` |
| `SemanticTopKHints` | `SemanticTopK` | `method`, `num_pivots`, `window_size`, `use_listwise` |
| `SemanticJoinHints` | `SemanticJoin` | `blocking_strategy`, `blocking_key`, `similarity_threshold` |
| `SemanticGroupByHints` | `SemanticGroupBy` | `batch_size`, `num_groups` |
| `SemanticSummarizeHints` | `SemanticSummarize` | `max_tokens`, `max_input_rows` |
| `SemanticFillHints` | `SemanticFill` | `batch_size`, `use_cot` |

```python
from solicedb.logical_ir import SemanticTopKHints

hints = SemanticTopKHints(
    method="multipivot",
    num_pivots=3,
    window_size=20,
)
```

#### Global Hints

Apply to the entire query plan:

```python
from solicedb.logical_ir import GlobalHints

plan = QueryPlan()
plan.global_hints = GlobalHints(
    prefer_listwise=True,
    max_batch_size=64,
    default_model="llama-7b",
    parallelism=4,
    enable_caching=True,
    debug_mode=False,
)
```

### Nodes (`nodes.py`)

All logical operators inherit from `LogicalNode` and are immutable.

#### Source Node

```python
Source(node_id, input_ids=(), ref="table_name", schema=optional_schema)
```

#### Classical Operators

| Node | Description | Key Fields |
|------|-------------|------------|
| `Select` | Row filtering (WHERE) | `predicate: Expr` |
| `Project` | Column selection/derivation | `columns: tuple[str \| tuple[str, Expr], ...]` |
| `Join` | Join two relations | `join_type: JoinType`, `condition: Expr` |
| `GroupBy` | Grouping + aggregation | `group_columns`, `aggregations` |
| `OrderBy` | Sorting with optional limit | `keys: tuple[tuple[str, SortDirection], ...]`, `limit` |
| `Distinct` | Remove duplicates | `columns: tuple[str, ...] \| None` |

#### Semantic Operators

| Node | Description | Key Fields |
|------|-------------|------------|
| `SemanticFilter` | NL-based row filtering | `spec: SemanticSpec` |
| `SemanticProject` | NL-based column derivation | `spec`, `output_schema: Schema` |
| `SemanticJoin` | NL-based join condition | `spec`, `join_type`, `output_schema` |
| `SemanticGroupBy` | NL-based grouping | `spec`, `output_group_column: str` |
| `SemanticSummarize` | NL summarization per group | `group_columns`, `spec`, `output_schema` |
| `SemanticTopK` | NL-based top-k selection | `spec`, `k: int` |
| `SemanticFill` | NL-based NULL filling | `target_column`, `spec`, `output_dtype` |

### Query Plan (`plan.py`)

The `QueryPlan` class manages the DAG of nodes.

#### Low-Level API

For optimizers and power users:

```python
plan = QueryPlan()

# Add nodes directly (node_id is auto-assigned)
source_id = plan.add(Source(node_id=-1, input_ids=(), ref="movies"))
filter_id = plan.add(Select(node_id=-1, input_ids=(source_id,), predicate=expr))

# Access nodes
node = plan.get(filter_id)
all_nodes = plan.nodes()

# Set root
plan.set_root(filter_id)
```

#### High-Level Builder API

For regular users:

```python
plan = QueryPlan()

# Source
movies = plan.source("movies", {"title": DType.STRING, "year": DType.INT})

# Classical operators
filtered = plan.select(movies, gt(col("year"), lit(2000)))
projected = plan.project(filtered, ["title", "year", ("decade", div(col("year"), lit(10)))])
joined = plan.join(left_id, right_id, JoinType.INNER, eq(col("id"), col("movie_id")))
grouped = plan.group_by(movies, ["genre"], [("count", count()), ("avg_year", avg("year"))])
ordered = plan.order_by(movies, [("year", SortDirection.DESC)], limit=100)
unique = plan.distinct(movies, ["title"])

# Semantic operators
sem_filtered = plan.semantic_filter(movies, "Is {title} a classic?")
sem_projected = plan.semantic_project(movies, "Classify {title}: ", {"genre": DType.STRING})
sem_joined = plan.semantic_join(left, right, "{left.desc} matches {right.summary}")
sem_grouped = plan.semantic_group_by(reviews, "Sentiment of {text}", "sentiment")
sem_summarized = plan.semantic_summarize(reviews, ["product_id"], "Summarize: {ALL_COLS}", {"summary": DType.STRING})
sem_topk = plan.semantic_top_k(papers, "Rank by influence: {title}", k=10)
sem_filled = plan.semantic_fill(movies, "genre", "Infer from {title}", DType.STRING)

plan.set_root(sem_topk)
```

#### DAG Traversal

```python
# Get inputs/outputs
children = plan.children(node_id)  # Input node IDs
parents = plan.parents(node_id)    # Nodes that consume this node

# Topological order (for execution)
order = plan.topological_order()        # Inputs before outputs
order = plan.reverse_topological_order()  # Outputs before inputs
```

#### Validation

```python
errors = plan.validate()
# Checks: root set, no cycles, all nodes reachable

if plan.is_valid():
    print("Plan is valid")
```

#### Serialization

```python
# To JSON
json_str = plan.to_json(indent=2)

# To dict
plan_dict = plan.to_dict()
```

## Design Decisions

### Why Container-Based DAG?

Nodes reference inputs by ID rather than direct object references. This enables:
- Clean JSON serialization/deserialization
- Complex optimization algorithms that rewrite the DAG
- Clear ownership semantics

### Why Unified Classical + Semantic IR?

Having both operator types in one IR allows the optimizer to:
- Push classical filters before expensive semantic operators
- Fuse adjacent semantic operators
- Make global optimization decisions

### Why Typed Hints?

Hints are typed (not just dicts) to provide:
- IDE autocomplete
- Validation of hint names and types
- Clear documentation of available options

### Why Immutable Nodes?

Frozen dataclasses ensure:
- Safe sharing during optimization
- Easy debugging (nodes don't change after creation)
- Thread safety

## Extending the IR

### Adding a New Operator

1. Add hint class in `hints.py` (if needed)
2. Add node class in `nodes.py` inheriting from `LogicalNode`
3. Add builder method in `plan.py`
4. Export in `__init__.py`

### Adding a New Hint Field

Simply add the field to the appropriate hint dataclass with a default of `None`.

### Adding a New DType

Add to the `DType` enum in `types.py` and update `from_pandas_dtype()`.

## Future Work

- [ ] `from_json()` deserialization
- [ ] Schema inference through the DAG
- [ ] Cost model integration
- [ ] Physical plan IR (separate module)
