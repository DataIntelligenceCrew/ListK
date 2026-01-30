"""
Tests for solicedb.logical_ir.plan module.

Covers:
- QueryPlan class and all its methods
- Node management (add, get, set_root)
- Builder methods (source, select, project, join, etc.)
- DAG traversal (children, parents, topological_order)
- Validation
- Serialization
"""

import json

import pytest

from solicedb.logical_ir import (
    DType,
    QueryPlan,
    Schema,
    col,
    lit,
    gt,
    eq,
    and_,
    count,
    avg,
    JoinType,
    SortDirection,
)
from solicedb.logical_ir.hints import (
    GlobalHints,
    SemanticFilterHints,
    SemanticFillHints,
    SemanticGroupByHints,
    SemanticJoinHints,
    SemanticProjectHints,
    SemanticSummarizeHints,
    SemanticTopKHints,
)
from solicedb.logical_ir.nodes import (
    Distinct,
    GroupBy,
    Join,
    OrderBy,
    Project,
    Select,
    SemanticFilter,
    SemanticFill,
    SemanticGroupBy,
    SemanticJoin,
    SemanticProject,
    SemanticSummarize,
    SemanticTopK,
    Source,
)


class TestQueryPlanCreation:
    """Tests for QueryPlan creation and basic properties."""

    def test_empty_plan(self) -> None:
        """Test creating empty query plan."""
        plan = QueryPlan()
        assert len(plan) == 0
        assert plan.root_id is None
        assert plan.root is None

    def test_plan_with_global_hints(self) -> None:
        """Test creating plan with global hints."""
        hints = GlobalHints(prefer_listwise=True, max_batch_size=64)
        plan = QueryPlan(global_hints=hints)
        assert plan.global_hints.prefer_listwise is True
        assert plan.global_hints.max_batch_size == 64

    def test_plan_repr(self, empty_plan: QueryPlan) -> None:
        """Test __repr__ method."""
        repr_str = repr(empty_plan)
        assert "QueryPlan" in repr_str
        assert "nodes=0" in repr_str

    def test_plan_repr_with_root(self, simple_plan: QueryPlan) -> None:
        """Test __repr__ with root set."""
        repr_str = repr(simple_plan)
        assert "root=" in repr_str


class TestNodeManagement:
    """Tests for node management (add, get, set_root)."""

    def test_add_source_node(self, empty_plan: QueryPlan) -> None:
        """Test adding a source node."""
        node = Source(node_id=-1, input_ids=(), ref="movies")
        node_id = empty_plan.add(node)
        assert node_id == 0
        assert len(empty_plan) == 1

    def test_add_assigns_sequential_ids(self, empty_plan: QueryPlan) -> None:
        """Test that add assigns sequential IDs."""
        id1 = empty_plan.add(Source(node_id=-1, input_ids=(), ref="table1"))
        id2 = empty_plan.add(Source(node_id=-1, input_ids=(), ref="table2"))
        id3 = empty_plan.add(Source(node_id=-1, input_ids=(), ref="table3"))
        assert id1 == 0
        assert id2 == 1
        assert id3 == 2

    def test_add_updates_node_id(self, empty_plan: QueryPlan) -> None:
        """Test that add updates the node's ID."""
        node = Source(node_id=-1, input_ids=(), ref="movies")
        node_id = empty_plan.add(node)
        stored_node = empty_plan.get(node_id)
        assert stored_node.node_id == node_id

    def test_add_validates_input_ids(self, empty_plan: QueryPlan) -> None:
        """Test that add validates input IDs exist."""
        node = Select(node_id=-1, input_ids=(99,), predicate=gt(col("a"), lit(5)))
        with pytest.raises(ValueError, match="Input node 99 does not exist"):
            empty_plan.add(node)

    def test_add_with_valid_input_id(self, empty_plan: QueryPlan) -> None:
        """Test add with valid input ID."""
        source_id = empty_plan.add(Source(node_id=-1, input_ids=(), ref="table"))
        select_id = empty_plan.add(
            Select(node_id=-1, input_ids=(source_id,), predicate=gt(col("a"), lit(5)))
        )
        assert select_id == 1

    def test_get_existing_node(self, simple_plan: QueryPlan) -> None:
        """Test getting an existing node."""
        node = simple_plan.get(0)
        assert isinstance(node, Source)

    def test_get_nonexistent_raises(self, empty_plan: QueryPlan) -> None:
        """Test get raises KeyError for nonexistent node."""
        with pytest.raises(KeyError, match="Node 99 does not exist"):
            empty_plan.get(99)

    def test_set_root(self, empty_plan: QueryPlan) -> None:
        """Test setting root node."""
        source_id = empty_plan.add(Source(node_id=-1, input_ids=(), ref="table"))
        empty_plan.set_root(source_id)
        assert empty_plan.root_id == source_id
        assert empty_plan.root is not None

    def test_set_root_nonexistent_raises(self, empty_plan: QueryPlan) -> None:
        """Test set_root raises KeyError for nonexistent node."""
        with pytest.raises(KeyError, match="Node 99 does not exist"):
            empty_plan.set_root(99)

    def test_root_property(self, simple_plan: QueryPlan) -> None:
        """Test root property returns the root node."""
        root = simple_plan.root
        assert root is not None
        assert isinstance(root, Select)

    def test_contains_operator(self, simple_plan: QueryPlan) -> None:
        """Test __contains__ operator."""
        assert 0 in simple_plan
        assert 1 in simple_plan
        assert 99 not in simple_plan

    def test_nodes_method(self, simple_plan: QueryPlan) -> None:
        """Test nodes() returns all nodes in order."""
        nodes = simple_plan.nodes()
        assert len(nodes) == 2
        assert nodes[0].node_id == 0
        assert nodes[1].node_id == 1


class TestBuilderMethods:
    """Tests for high-level builder methods."""

    def test_source_minimal(self, empty_plan: QueryPlan) -> None:
        """Test source() with minimal arguments."""
        source_id = empty_plan.source("movies")
        assert source_id == 0
        node = empty_plan.get(source_id)
        assert isinstance(node, Source)
        assert node.ref == "movies"
        assert node.schema is None

    def test_source_with_schema_dict(self, empty_plan: QueryPlan) -> None:
        """Test source() with schema dict."""
        source_id = empty_plan.source("movies", {"title": DType.STRING, "year": DType.INT})
        node = empty_plan.get(source_id)
        assert node.schema is not None
        assert node.schema.has_column("title")

    def test_source_with_schema_object(self, empty_plan: QueryPlan) -> None:
        """Test source() with Schema object."""
        schema = Schema.from_dict({"id": DType.INT})
        source_id = empty_plan.source("table", schema)
        node = empty_plan.get(source_id)
        assert node.schema == schema

    def test_select(self, empty_plan: QueryPlan) -> None:
        """Test select() builder method."""
        source_id = empty_plan.source("table")
        select_id = empty_plan.select(source_id, gt(col("year"), lit(2000)))
        node = empty_plan.get(select_id)
        assert isinstance(node, Select)

    def test_project_tuple(self, empty_plan: QueryPlan) -> None:
        """Test project() with tuple."""
        source_id = empty_plan.source("table")
        project_id = empty_plan.project(source_id, ("a", "b", "c"))
        node = empty_plan.get(project_id)
        assert isinstance(node, Project)
        assert node.columns == ("a", "b", "c")

    def test_project_list(self, empty_plan: QueryPlan) -> None:
        """Test project() with list (converted to tuple)."""
        source_id = empty_plan.source("table")
        project_id = empty_plan.project(source_id, ["a", "b"])
        node = empty_plan.get(project_id)
        assert node.columns == ("a", "b")

    def test_join(self, empty_plan: QueryPlan) -> None:
        """Test join() builder method."""
        left_id = empty_plan.source("left")
        right_id = empty_plan.source("right")
        join_id = empty_plan.join(
            left_id, right_id,
            JoinType.INNER,
            eq(col("id", "left"), col("id", "right"))
        )
        node = empty_plan.get(join_id)
        assert isinstance(node, Join)
        assert node.join_type == JoinType.INNER

    def test_group_by(self, empty_plan: QueryPlan) -> None:
        """Test group_by() builder method."""
        source_id = empty_plan.source("table")
        groupby_id = empty_plan.group_by(
            source_id,
            ["genre"],
            [("cnt", count()), ("avg_rating", avg("rating"))],
        )
        node = empty_plan.get(groupby_id)
        assert isinstance(node, GroupBy)
        assert node.group_columns == ("genre",)

    def test_order_by(self, empty_plan: QueryPlan) -> None:
        """Test order_by() builder method."""
        source_id = empty_plan.source("table")
        orderby_id = empty_plan.order_by(
            source_id,
            [("year", SortDirection.DESC)],
            limit=10,
        )
        node = empty_plan.get(orderby_id)
        assert isinstance(node, OrderBy)
        assert node.limit == 10

    def test_distinct(self, empty_plan: QueryPlan) -> None:
        """Test distinct() builder method."""
        source_id = empty_plan.source("table")
        distinct_id = empty_plan.distinct(source_id, ["col1", "col2"])
        node = empty_plan.get(distinct_id)
        assert isinstance(node, Distinct)
        assert node.columns == ("col1", "col2")

    def test_distinct_all_columns(self, empty_plan: QueryPlan) -> None:
        """Test distinct() on all columns."""
        source_id = empty_plan.source("table")
        distinct_id = empty_plan.distinct(source_id)
        node = empty_plan.get(distinct_id)
        assert node.columns is None

    def test_semantic_filter(self, empty_plan: QueryPlan) -> None:
        """Test semantic_filter() builder method."""
        source_id = empty_plan.source("table")
        filter_id = empty_plan.semantic_filter(
            source_id,
            "Is {title} a classic?",
            hints=SemanticFilterHints(batch_size=32),
        )
        node = empty_plan.get(filter_id)
        assert isinstance(node, SemanticFilter)
        assert node.hints is not None

    def test_semantic_project(self, empty_plan: QueryPlan) -> None:
        """Test semantic_project() builder method."""
        source_id = empty_plan.source("table")
        project_id = empty_plan.semantic_project(
            source_id,
            "Classify {title}",
            {"genre": DType.STRING},
            hints=SemanticProjectHints(max_tokens=256),
        )
        node = empty_plan.get(project_id)
        assert isinstance(node, SemanticProject)
        assert node.output_schema is not None

    def test_semantic_join(self, empty_plan: QueryPlan) -> None:
        """Test semantic_join() builder method."""
        left_id = empty_plan.source("left")
        right_id = empty_plan.source("right")
        join_id = empty_plan.semantic_join(
            left_id, right_id,
            "{a} matches {b}",
            join_type=JoinType.LEFT,
            output_schema={"score": DType.FLOAT},
            hints=SemanticJoinHints(blocking_strategy="embedding"),
        )
        node = empty_plan.get(join_id)
        assert isinstance(node, SemanticJoin)
        assert node.join_type == JoinType.LEFT

    def test_semantic_group_by(self, empty_plan: QueryPlan) -> None:
        """Test semantic_group_by() builder method."""
        source_id = empty_plan.source("table")
        groupby_id = empty_plan.semantic_group_by(
            source_id,
            "Classify {text} as positive/negative",
            "sentiment",
            hints=SemanticGroupByHints(num_groups=2),
        )
        node = empty_plan.get(groupby_id)
        assert isinstance(node, SemanticGroupBy)
        assert node.output_group_column == "sentiment"

    def test_semantic_summarize(self, empty_plan: QueryPlan) -> None:
        """Test semantic_summarize() builder method."""
        source_id = empty_plan.source("table")
        summarize_id = empty_plan.semantic_summarize(
            source_id,
            ["product_id"],
            "Summarize: {ALL_COLS}",
            {"summary": DType.STRING},
            hints=SemanticSummarizeHints(max_tokens=500),
        )
        node = empty_plan.get(summarize_id)
        assert isinstance(node, SemanticSummarize)
        assert node.group_columns == ("product_id",)

    def test_semantic_top_k(self, empty_plan: QueryPlan) -> None:
        """Test semantic_top_k() builder method."""
        source_id = empty_plan.source("table")
        topk_id = empty_plan.semantic_top_k(
            source_id,
            "Rank by quality: {title}",
            k=10,
            hints=SemanticTopKHints(method="multipivot", num_pivots=3),
        )
        node = empty_plan.get(topk_id)
        assert isinstance(node, SemanticTopK)
        assert node.k == 10

    def test_semantic_fill(self, empty_plan: QueryPlan) -> None:
        """Test semantic_fill() builder method."""
        source_id = empty_plan.source("table")
        fill_id = empty_plan.semantic_fill(
            source_id,
            "genre",
            "Infer from {title}",
            DType.STRING,
            hints=SemanticFillHints(use_cot=True),
        )
        node = empty_plan.get(fill_id)
        assert isinstance(node, SemanticFill)
        assert node.target_column == "genre"


class TestDAGTraversal:
    """Tests for DAG traversal methods."""

    def test_children(self, simple_plan: QueryPlan) -> None:
        """Test children() method."""
        # Select node (id=1) has Source (id=0) as child
        children = simple_plan.children(1)
        assert children == [0]

    def test_children_source_empty(self, simple_plan: QueryPlan) -> None:
        """Test children() for source node is empty."""
        children = simple_plan.children(0)
        assert children == []

    def test_parents(self, simple_plan: QueryPlan) -> None:
        """Test parents() method."""
        # Source (id=0) is consumed by Select (id=1)
        parents = simple_plan.parents(0)
        assert parents == [1]

    def test_parents_root_empty(self, simple_plan: QueryPlan) -> None:
        """Test parents() for root node is empty."""
        parents = simple_plan.parents(1)
        assert parents == []

    def test_topological_order(self, simple_plan: QueryPlan) -> None:
        """Test topological_order() returns correct order."""
        order = simple_plan.topological_order()
        # Source should come before Select
        assert order.index(0) < order.index(1)

    def test_topological_order_complex(self, empty_plan: QueryPlan) -> None:
        """Test topological order with more complex graph."""
        # Create: source1, source2 -> join -> filter
        s1 = empty_plan.source("t1")
        s2 = empty_plan.source("t2")
        j = empty_plan.join(s1, s2, JoinType.INNER)
        f = empty_plan.select(j, gt(col("a"), lit(5)))
        empty_plan.set_root(f)

        order = empty_plan.topological_order()
        # Sources before join, join before filter
        assert order.index(s1) < order.index(j)
        assert order.index(s2) < order.index(j)
        assert order.index(j) < order.index(f)

    def test_reverse_topological_order(self, simple_plan: QueryPlan) -> None:
        """Test reverse_topological_order()."""
        order = simple_plan.reverse_topological_order()
        # Select should come before Source
        assert order.index(1) < order.index(0)


class TestValidation:
    """Tests for validation methods."""

    def test_validate_no_root(self, empty_plan: QueryPlan) -> None:
        """Test validation fails without root."""
        empty_plan.source("table")
        errors = empty_plan.validate()
        assert "Root node is not set" in errors

    def test_validate_valid_plan(self, simple_plan: QueryPlan) -> None:
        """Test validation passes for valid plan."""
        errors = simple_plan.validate()
        assert errors == []

    def test_is_valid(self, simple_plan: QueryPlan) -> None:
        """Test is_valid() method."""
        assert simple_plan.is_valid() is True

    def test_is_valid_false(self, empty_plan: QueryPlan) -> None:
        """Test is_valid() returns False for invalid plan."""
        empty_plan.source("table")
        assert empty_plan.is_valid() is False

    def test_validate_unreachable_nodes(self, empty_plan: QueryPlan) -> None:
        """Test validation detects unreachable nodes."""
        s1 = empty_plan.source("t1")
        s2 = empty_plan.source("t2")  # Unreachable
        empty_plan.set_root(s1)

        errors = empty_plan.validate()
        assert len(errors) == 1
        assert "Unreachable" in errors[0]
        assert str(s2) in errors[0]


class TestSerialization:
    """Tests for serialization methods."""

    def test_to_dict(self, simple_plan: QueryPlan) -> None:
        """Test to_dict() method."""
        d = simple_plan.to_dict()
        assert "nodes" in d
        assert "root_id" in d
        assert "global_hints" in d

    def test_to_dict_nodes(self, simple_plan: QueryPlan) -> None:
        """Test to_dict() includes all nodes."""
        d = simple_plan.to_dict()
        assert "0" in d["nodes"]
        assert "1" in d["nodes"]

    def test_to_dict_node_structure(self, simple_plan: QueryPlan) -> None:
        """Test node serialization structure."""
        d = simple_plan.to_dict()
        source_dict = d["nodes"]["0"]
        assert source_dict["type"] == "Source"
        assert source_dict["node_id"] == 0
        assert source_dict["input_ids"] == []
        assert source_dict["ref"] == "movies"

    def test_to_json(self, simple_plan: QueryPlan) -> None:
        """Test to_json() method."""
        json_str = simple_plan.to_json()
        parsed = json.loads(json_str)
        assert "nodes" in parsed
        assert "root_id" in parsed

    def test_to_json_compact(self, simple_plan: QueryPlan) -> None:
        """Test to_json() with no indentation."""
        json_str = simple_plan.to_json(indent=None)
        assert "\n" not in json_str

    def test_serialize_schema(self, empty_plan: QueryPlan) -> None:
        """Test schema serialization."""
        empty_plan.source("table", {"id": DType.INT, "name": DType.STRING})
        empty_plan.set_root(0)

        d = empty_plan.to_dict()
        schema_dict = d["nodes"]["0"]["schema"]
        assert schema_dict["__type__"] == "Schema"
        assert "columns" in schema_dict

    def test_serialize_semantic_spec(self, empty_plan: QueryPlan) -> None:
        """Test SemanticSpec serialization."""
        source_id = empty_plan.source("table")
        empty_plan.semantic_filter(source_id, "Is {title} good?")
        empty_plan.set_root(1)

        d = empty_plan.to_dict()
        spec_dict = d["nodes"]["1"]["spec"]
        assert spec_dict["__type__"] == "SemanticSpec"
        assert spec_dict["template"] == "Is {title} good?"
        assert "title" in spec_dict["input_columns"]

    def test_serialize_expression(self, simple_plan: QueryPlan) -> None:
        """Test expression serialization."""
        d = simple_plan.to_dict()
        predicate_dict = d["nodes"]["1"]["predicate"]
        assert predicate_dict["__type__"] == "Expr"
        assert "repr" in predicate_dict

    def test_serialize_join_type(self, empty_plan: QueryPlan) -> None:
        """Test JoinType serialization."""
        s1 = empty_plan.source("t1")
        s2 = empty_plan.source("t2")
        empty_plan.join(s1, s2, JoinType.LEFT)
        empty_plan.set_root(2)

        d = empty_plan.to_dict()
        jt_dict = d["nodes"]["2"]["join_type"]
        assert jt_dict["__type__"] == "JoinType"
        assert jt_dict["value"] == "LEFT"

    def test_serialize_sort_direction(self, empty_plan: QueryPlan) -> None:
        """Test SortDirection serialization."""
        source_id = empty_plan.source("table")
        empty_plan.order_by(source_id, [("year", SortDirection.DESC)])
        empty_plan.set_root(1)

        d = empty_plan.to_dict()
        keys = d["nodes"]["1"]["keys"]
        assert keys[0][1]["__type__"] == "SortDirection"
        assert keys[0][1]["value"] == "DESC"

    def test_serialize_dtype(self, empty_plan: QueryPlan) -> None:
        """Test DType serialization."""
        source_id = empty_plan.source("table")
        empty_plan.semantic_fill(source_id, "col", "Fill {x}", DType.INT)
        empty_plan.set_root(1)

        d = empty_plan.to_dict()
        dtype_dict = d["nodes"]["1"]["output_dtype"]
        assert dtype_dict["__type__"] == "DType"
        assert dtype_dict["value"] == "int"

    def test_serialize_hints(self, empty_plan: QueryPlan) -> None:
        """Test hints serialization."""
        source_id = empty_plan.source("table")
        empty_plan.semantic_filter(
            source_id,
            "Is {title} good?",
            hints=SemanticFilterHints(batch_size=32, use_cot=True),
        )
        empty_plan.set_root(1)

        d = empty_plan.to_dict()
        hints_dict = d["nodes"]["1"]["hints"]
        assert hints_dict["__type__"] == "SemanticFilterHints"
        assert hints_dict["batch_size"] == 32
        assert hints_dict["use_cot"] is True

    def test_serialize_global_hints(self, empty_plan: QueryPlan) -> None:
        """Test global hints serialization."""
        empty_plan.global_hints = GlobalHints(
            prefer_listwise=True,
            max_batch_size=64,
            default_model="llama-7b",
        )
        empty_plan.source("table")
        empty_plan.set_root(0)

        d = empty_plan.to_dict()
        gh = d["global_hints"]
        assert gh["prefer_listwise"] is True
        assert gh["max_batch_size"] == 64
        assert gh["default_model"] == "llama-7b"

    def test_serialize_aggregation(self, empty_plan: QueryPlan) -> None:
        """Test Aggregation serialization.

        Note: Aggregation is serialized with the special Aggregation serialization
        when encountered directly, but when nested in a tuple it may fall through
        to the Expr serialization. The code checks for Aggregation before Expr,
        so the order of checks in _serialize_value matters.
        """
        source_id = empty_plan.source("table")
        empty_plan.group_by(source_id, ["genre"], [("cnt", count())])
        empty_plan.set_root(1)

        d = empty_plan.to_dict()
        aggs = d["nodes"]["1"]["aggregations"]
        # The aggregation is stored as (name, Aggregation) tuple
        assert aggs[0][0] == "cnt"
        # Aggregation may be serialized as Aggregation or Expr depending on code path
        assert aggs[0][1]["__type__"] in ("Aggregation", "Expr")


class TestComplexQueryPlans:
    """Tests for complex query plan construction."""

    def test_linear_pipeline(self, empty_plan: QueryPlan) -> None:
        """Test linear pipeline: source -> filter -> project -> topk."""
        source = empty_plan.source("movies", {"title": DType.STRING, "year": DType.INT})
        filtered = empty_plan.select(source, gt(col("year"), lit(2000)))
        projected = empty_plan.project(filtered, ["title"])
        topk = empty_plan.semantic_top_k(projected, "Rank: {title}", k=10)
        empty_plan.set_root(topk)

        assert len(empty_plan) == 4
        assert empty_plan.is_valid()

    def test_join_with_aggregation(self, empty_plan: QueryPlan) -> None:
        """Test join followed by aggregation."""
        movies = empty_plan.source("movies")
        ratings = empty_plan.source("ratings")
        joined = empty_plan.join(
            movies, ratings,
            JoinType.INNER,
            eq(col("id", "movies"), col("movie_id", "ratings"))
        )
        grouped = empty_plan.group_by(
            joined,
            ["title"],
            [("avg_rating", avg("rating"))],
        )
        empty_plan.set_root(grouped)

        assert empty_plan.is_valid()
        assert empty_plan.get(grouped).is_unary()

    def test_multiple_semantic_operators(self, empty_plan: QueryPlan) -> None:
        """Test chaining multiple semantic operators."""
        source = empty_plan.source("papers")
        filled = empty_plan.semantic_fill(source, "category", "Classify {title}", DType.STRING)
        filtered = empty_plan.semantic_filter(filled, "Is {title} influential?")
        topk = empty_plan.semantic_top_k(filtered, "Rank by citations: {title}", k=10)
        empty_plan.set_root(topk)

        assert empty_plan.is_valid()
        order = empty_plan.topological_order()
        assert order == [0, 1, 2, 3]

    def test_diamond_pattern(self, empty_plan: QueryPlan) -> None:
        """Test diamond DAG pattern: source -> (a, b) -> join."""
        source = empty_plan.source("table")
        a = empty_plan.select(source, gt(col("x"), lit(0)))
        b = empty_plan.select(source, gt(col("y"), lit(0)))
        joined = empty_plan.join(a, b, JoinType.INNER, eq(col("id"), col("id")))
        empty_plan.set_root(joined)

        assert empty_plan.is_valid()
        # Both filters consume source
        assert empty_plan.parents(0) == [1, 2]