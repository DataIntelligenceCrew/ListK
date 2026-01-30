"""
Tests for solicedb.logical_ir.nodes module.

Covers:
- JoinType and SortDirection enums
- LogicalNode base class
- Source node
- Classical operators (Select, Project, Join, GroupBy, OrderBy, Distinct)
- Semantic operators (SemanticFilter, SemanticProject, SemanticJoin,
                      SemanticGroupBy, SemanticSummarize, SemanticTopK, SemanticFill)
"""

import pytest

from solicedb.logical_ir import DType, Schema
from solicedb.logical_ir.expressions import col, lit, gt, eq, and_, ge, sub, count, avg
from solicedb.logical_ir.hints import (
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
    JoinType,
    LogicalNode,
    NodeId,
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
    SortDirection,
    Source,
)
from solicedb.logical_ir.specs import SemanticSpec


class TestJoinType:
    """Tests for the JoinType enum."""

    def test_join_types_exist(self) -> None:
        """Test that all join types exist."""
        assert JoinType.INNER is not None
        assert JoinType.LEFT is not None
        assert JoinType.RIGHT is not None
        assert JoinType.FULL is not None
        assert JoinType.CROSS is not None

    def test_join_types_unique(self) -> None:
        """Test that join types have unique values."""
        values = [jt.value for jt in JoinType]
        assert len(values) == len(set(values))


class TestSortDirection:
    """Tests for the SortDirection enum."""

    def test_sort_directions_exist(self) -> None:
        """Test that sort directions exist."""
        assert SortDirection.ASC is not None
        assert SortDirection.DESC is not None

    def test_sort_directions_unique(self) -> None:
        """Test that sort directions have unique values."""
        assert SortDirection.ASC.value != SortDirection.DESC.value


class TestLogicalNodeBase:
    """Tests for LogicalNode base class methods."""

    def test_is_source_true_for_source_node(self) -> None:
        """Test is_source returns True for source nodes."""
        source = Source(node_id=0, input_ids=(), ref="table")
        assert source.is_source() is True

    def test_is_source_false_for_unary_node(self) -> None:
        """Test is_source returns False for unary nodes."""
        select = Select(node_id=1, input_ids=(0,), predicate=gt(col("a"), lit(5)))
        assert select.is_source() is False

    def test_is_unary_true(self) -> None:
        """Test is_unary returns True for single-input nodes."""
        select = Select(node_id=1, input_ids=(0,), predicate=gt(col("a"), lit(5)))
        assert select.is_unary() is True

    def test_is_unary_false(self) -> None:
        """Test is_unary returns False for source and binary nodes."""
        source = Source(node_id=0, input_ids=(), ref="table")
        join = Join(node_id=2, input_ids=(0, 1), join_type=JoinType.INNER)
        assert source.is_unary() is False
        assert join.is_unary() is False

    def test_is_binary_true(self) -> None:
        """Test is_binary returns True for two-input nodes."""
        join = Join(node_id=2, input_ids=(0, 1), join_type=JoinType.INNER)
        assert join.is_binary() is True

    def test_is_binary_false(self) -> None:
        """Test is_binary returns False for source and unary nodes."""
        source = Source(node_id=0, input_ids=(), ref="table")
        select = Select(node_id=1, input_ids=(0,), predicate=gt(col("a"), lit(5)))
        assert source.is_binary() is False
        assert select.is_binary() is False


class TestSource:
    """Tests for the Source node."""

    def test_source_creation_minimal(self) -> None:
        """Test minimal source creation."""
        source = Source(node_id=0, input_ids=(), ref="movies")
        assert source.node_id == 0
        assert source.input_ids == ()
        assert source.ref == "movies"
        assert source.schema is None

    def test_source_creation_with_schema(self) -> None:
        """Test source creation with schema."""
        schema = Schema.from_dict({"id": DType.INT, "title": DType.STRING})
        source = Source(node_id=0, input_ids=(), ref="movies", schema=schema)
        assert source.schema == schema

    def test_source_is_source_node(self) -> None:
        """Test source is identified as source node."""
        source = Source(node_id=0, input_ids=(), ref="table")
        assert source.is_source() is True
        assert source.is_unary() is False
        assert source.is_binary() is False

    def test_source_immutability(self) -> None:
        """Test source is immutable."""
        source = Source(node_id=0, input_ids=(), ref="table")
        with pytest.raises(AttributeError):
            source.ref = "other"  # type: ignore


class TestSelect:
    """Tests for the Select node."""

    def test_select_creation(self) -> None:
        """Test Select node creation."""
        predicate = gt(col("year"), lit(2000))
        select = Select(node_id=1, input_ids=(0,), predicate=predicate)
        assert select.node_id == 1
        assert select.input_ids == (0,)
        assert select.predicate == predicate

    def test_select_is_unary(self) -> None:
        """Test Select is unary operator."""
        select = Select(node_id=1, input_ids=(0,), predicate=gt(col("a"), lit(5)))
        assert select.is_unary() is True

    def test_select_complex_predicate(self) -> None:
        """Test Select with complex predicate."""
        predicate = and_(
            gt(col("year"), lit(2000)),
            ge(col("rating"), lit(4.0))
        )
        select = Select(node_id=1, input_ids=(0,), predicate=predicate)
        assert select.predicate == predicate


class TestProject:
    """Tests for the Project node."""

    def test_project_simple_columns(self) -> None:
        """Test Project with simple column list."""
        project = Project(
            node_id=1,
            input_ids=(0,),
            columns=("title", "year"),
        )
        assert project.columns == ("title", "year")

    def test_project_with_computed_column(self) -> None:
        """Test Project with computed column."""
        project = Project(
            node_id=1,
            input_ids=(0,),
            columns=("title", "year", ("profit", sub(col("revenue"), col("budget")))),
        )
        assert len(project.columns) == 3
        assert project.columns[2][0] == "profit"

    def test_project_is_unary(self) -> None:
        """Test Project is unary operator."""
        project = Project(node_id=1, input_ids=(0,), columns=("a", "b"))
        assert project.is_unary() is True


class TestJoin:
    """Tests for the Join node."""

    def test_join_creation_inner(self) -> None:
        """Test Join node creation with INNER join."""
        condition = eq(col("id", "movies"), col("movie_id", "ratings"))
        join = Join(
            node_id=2,
            input_ids=(0, 1),
            join_type=JoinType.INNER,
            condition=condition,
        )
        assert join.join_type == JoinType.INNER
        assert join.condition == condition

    def test_join_cross_no_condition(self) -> None:
        """Test CROSS join with no condition."""
        join = Join(
            node_id=2,
            input_ids=(0, 1),
            join_type=JoinType.CROSS,
            condition=None,
        )
        assert join.join_type == JoinType.CROSS
        assert join.condition is None

    def test_join_is_binary(self) -> None:
        """Test Join is binary operator."""
        join = Join(node_id=2, input_ids=(0, 1), join_type=JoinType.INNER)
        assert join.is_binary() is True

    def test_join_types(self) -> None:
        """Test all join types can be used."""
        for jt in JoinType:
            join = Join(node_id=2, input_ids=(0, 1), join_type=jt)
            assert join.join_type == jt


class TestGroupBy:
    """Tests for the GroupBy node."""

    def test_groupby_creation(self) -> None:
        """Test GroupBy node creation."""
        groupby = GroupBy(
            node_id=1,
            input_ids=(0,),
            group_columns=("genre", "year"),
            aggregations=(("count", count()), ("avg_rating", avg("rating"))),
        )
        assert groupby.group_columns == ("genre", "year")
        assert len(groupby.aggregations) == 2

    def test_groupby_is_unary(self) -> None:
        """Test GroupBy is unary operator."""
        groupby = GroupBy(
            node_id=1,
            input_ids=(0,),
            group_columns=("a",),
            aggregations=(("count", count()),),
        )
        assert groupby.is_unary() is True


class TestOrderBy:
    """Tests for the OrderBy node."""

    def test_orderby_creation(self) -> None:
        """Test OrderBy node creation."""
        orderby = OrderBy(
            node_id=1,
            input_ids=(0,),
            keys=(("year", SortDirection.DESC), ("title", SortDirection.ASC)),
            limit=10,
        )
        assert orderby.keys == (("year", SortDirection.DESC), ("title", SortDirection.ASC))
        assert orderby.limit == 10

    def test_orderby_no_limit(self) -> None:
        """Test OrderBy with no limit."""
        orderby = OrderBy(
            node_id=1,
            input_ids=(0,),
            keys=(("year", SortDirection.DESC),),
        )
        assert orderby.limit is None

    def test_orderby_is_unary(self) -> None:
        """Test OrderBy is unary operator."""
        orderby = OrderBy(
            node_id=1,
            input_ids=(0,),
            keys=(("a", SortDirection.ASC),),
        )
        assert orderby.is_unary() is True


class TestDistinct:
    """Tests for the Distinct node."""

    def test_distinct_all_columns(self) -> None:
        """Test Distinct on all columns."""
        distinct = Distinct(node_id=1, input_ids=(0,), columns=None)
        assert distinct.columns is None

    def test_distinct_specific_columns(self) -> None:
        """Test Distinct on specific columns."""
        distinct = Distinct(node_id=1, input_ids=(0,), columns=("genre", "year"))
        assert distinct.columns == ("genre", "year")

    def test_distinct_is_unary(self) -> None:
        """Test Distinct is unary operator."""
        distinct = Distinct(node_id=1, input_ids=(0,))
        assert distinct.is_unary() is True


class TestSemanticFilter:
    """Tests for the SemanticFilter node."""

    def test_semantic_filter_creation(self) -> None:
        """Test SemanticFilter node creation."""
        spec = SemanticSpec.parse("Is {title} from {year} a classic?")
        sem_filter = SemanticFilter(
            node_id=1,
            input_ids=(0,),
            spec=spec,
        )
        assert sem_filter.spec == spec
        assert sem_filter.hints is None

    def test_semantic_filter_with_hints(self) -> None:
        """Test SemanticFilter with hints."""
        spec = SemanticSpec.parse("Is {title} a classic?")
        hints = SemanticFilterHints(batch_size=32, use_cot=True)
        sem_filter = SemanticFilter(
            node_id=1,
            input_ids=(0,),
            spec=spec,
            hints=hints,
        )
        assert sem_filter.hints == hints

    def test_semantic_filter_is_unary(self) -> None:
        """Test SemanticFilter is unary operator."""
        spec = SemanticSpec.parse("Is {title} a classic?")
        sem_filter = SemanticFilter(node_id=1, input_ids=(0,), spec=spec)
        assert sem_filter.is_unary() is True


class TestSemanticProject:
    """Tests for the SemanticProject node."""

    def test_semantic_project_creation(self) -> None:
        """Test SemanticProject node creation."""
        spec = SemanticSpec.parse("Based on {title} and {synopsis}, classify:")
        output_schema = Schema.from_dict({"genre": DType.STRING, "mood": DType.STRING})
        sem_project = SemanticProject(
            node_id=1,
            input_ids=(0,),
            spec=spec,
            output_schema=output_schema,
        )
        assert sem_project.spec == spec
        assert sem_project.output_schema == output_schema
        assert sem_project.hints is None

    def test_semantic_project_with_hints(self) -> None:
        """Test SemanticProject with hints."""
        spec = SemanticSpec.parse("Classify {title}")
        output_schema = Schema.from_dict({"genre": DType.STRING})
        hints = SemanticProjectHints(batch_size=16, max_tokens=256)
        sem_project = SemanticProject(
            node_id=1,
            input_ids=(0,),
            spec=spec,
            output_schema=output_schema,
            hints=hints,
        )
        assert sem_project.hints == hints


class TestSemanticJoin:
    """Tests for the SemanticJoin node."""

    def test_semantic_join_creation(self) -> None:
        """Test SemanticJoin node creation."""
        spec = SemanticSpec.parse("{review_text} describes {product_name}")
        sem_join = SemanticJoin(
            node_id=2,
            input_ids=(0, 1),
            spec=spec,
            join_type=JoinType.INNER,
        )
        assert sem_join.spec == spec
        assert sem_join.join_type == JoinType.INNER
        assert sem_join.output_schema is None
        assert sem_join.hints is None

    def test_semantic_join_with_output_schema(self) -> None:
        """Test SemanticJoin with output schema."""
        spec = SemanticSpec.parse("{a} matches {b}")
        output_schema = Schema.from_dict({"match_score": DType.FLOAT})
        sem_join = SemanticJoin(
            node_id=2,
            input_ids=(0, 1),
            spec=spec,
            join_type=JoinType.INNER,
            output_schema=output_schema,
        )
        assert sem_join.output_schema == output_schema

    def test_semantic_join_with_hints(self) -> None:
        """Test SemanticJoin with hints."""
        spec = SemanticSpec.parse("{a} matches {b}")
        hints = SemanticJoinHints(blocking_strategy="embedding", similarity_threshold=0.8)
        sem_join = SemanticJoin(
            node_id=2,
            input_ids=(0, 1),
            spec=spec,
            join_type=JoinType.INNER,
            hints=hints,
        )
        assert sem_join.hints == hints

    def test_semantic_join_is_binary(self) -> None:
        """Test SemanticJoin is binary operator."""
        spec = SemanticSpec.parse("{a} matches {b}")
        sem_join = SemanticJoin(
            node_id=2,
            input_ids=(0, 1),
            spec=spec,
            join_type=JoinType.INNER,
        )
        assert sem_join.is_binary() is True


class TestSemanticGroupBy:
    """Tests for the SemanticGroupBy node."""

    def test_semantic_groupby_creation(self) -> None:
        """Test SemanticGroupBy node creation."""
        spec = SemanticSpec.parse("Classify {review_text} as positive, negative, or neutral")
        sem_groupby = SemanticGroupBy(
            node_id=1,
            input_ids=(0,),
            spec=spec,
            output_group_column="sentiment",
        )
        assert sem_groupby.spec == spec
        assert sem_groupby.output_group_column == "sentiment"
        assert sem_groupby.hints is None

    def test_semantic_groupby_with_hints(self) -> None:
        """Test SemanticGroupBy with hints."""
        spec = SemanticSpec.parse("Classify {text}")
        hints = SemanticGroupByHints(num_groups=3, batch_size=32)
        sem_groupby = SemanticGroupBy(
            node_id=1,
            input_ids=(0,),
            spec=spec,
            output_group_column="category",
            hints=hints,
        )
        assert sem_groupby.hints == hints


class TestSemanticSummarize:
    """Tests for the SemanticSummarize node."""

    def test_semantic_summarize_creation(self) -> None:
        """Test SemanticSummarize node creation."""
        spec = SemanticSpec.parse("Summarize these reviews: {ALL_COLS}")
        output_schema = Schema.from_dict({"summary": DType.STRING})
        sem_summarize = SemanticSummarize(
            node_id=1,
            input_ids=(0,),
            group_columns=("product_id",),
            spec=spec,
            output_schema=output_schema,
        )
        assert sem_summarize.group_columns == ("product_id",)
        assert sem_summarize.spec == spec
        assert sem_summarize.output_schema == output_schema
        assert sem_summarize.hints is None

    def test_semantic_summarize_with_hints(self) -> None:
        """Test SemanticSummarize with hints."""
        spec = SemanticSpec.parse("Summarize: {ALL_COLS}")
        output_schema = Schema.from_dict({"summary": DType.STRING})
        hints = SemanticSummarizeHints(max_tokens=500, max_input_rows=100)
        sem_summarize = SemanticSummarize(
            node_id=1,
            input_ids=(0,),
            group_columns=("group",),
            spec=spec,
            output_schema=output_schema,
            hints=hints,
        )
        assert sem_summarize.hints == hints


class TestSemanticTopK:
    """Tests for the SemanticTopK node."""

    def test_semantic_topk_creation(self) -> None:
        """Test SemanticTopK node creation."""
        spec = SemanticSpec.parse("Rank by influence: {title}, {abstract}")
        sem_topk = SemanticTopK(
            node_id=1,
            input_ids=(0,),
            spec=spec,
            k=10,
        )
        assert sem_topk.spec == spec
        assert sem_topk.k == 10
        assert sem_topk.hints is None

    def test_semantic_topk_with_hints(self) -> None:
        """Test SemanticTopK with hints."""
        spec = SemanticSpec.parse("Rank by quality: {title}")
        hints = SemanticTopKHints(method="multipivot", num_pivots=3)
        sem_topk = SemanticTopK(
            node_id=1,
            input_ids=(0,),
            spec=spec,
            k=5,
            hints=hints,
        )
        assert sem_topk.hints == hints

    def test_semantic_topk_is_unary(self) -> None:
        """Test SemanticTopK is unary operator."""
        spec = SemanticSpec.parse("Rank: {title}")
        sem_topk = SemanticTopK(node_id=1, input_ids=(0,), spec=spec, k=10)
        assert sem_topk.is_unary() is True


class TestSemanticFill:
    """Tests for the SemanticFill node."""

    def test_semantic_fill_creation(self) -> None:
        """Test SemanticFill node creation."""
        spec = SemanticSpec.parse("Infer genre from {title} and {synopsis}")
        sem_fill = SemanticFill(
            node_id=1,
            input_ids=(0,),
            target_column="genre",
            spec=spec,
            output_dtype=DType.STRING,
        )
        assert sem_fill.target_column == "genre"
        assert sem_fill.spec == spec
        assert sem_fill.output_dtype == DType.STRING
        assert sem_fill.hints is None

    def test_semantic_fill_with_hints(self) -> None:
        """Test SemanticFill with hints."""
        spec = SemanticSpec.parse("Infer from {title}")
        hints = SemanticFillHints(batch_size=32, use_cot=True)
        sem_fill = SemanticFill(
            node_id=1,
            input_ids=(0,),
            target_column="category",
            spec=spec,
            output_dtype=DType.STRING,
            hints=hints,
        )
        assert sem_fill.hints == hints

    def test_semantic_fill_is_unary(self) -> None:
        """Test SemanticFill is unary operator."""
        spec = SemanticSpec.parse("Fill {col}")
        sem_fill = SemanticFill(
            node_id=1,
            input_ids=(0,),
            target_column="col",
            spec=spec,
            output_dtype=DType.STRING,
        )
        assert sem_fill.is_unary() is True


class TestNodeImmutability:
    """Tests for node immutability across all node types."""

    def test_source_immutable(self) -> None:
        """Test Source is immutable."""
        node = Source(node_id=0, input_ids=(), ref="table")
        with pytest.raises(AttributeError):
            node.ref = "other"  # type: ignore

    def test_select_immutable(self) -> None:
        """Test Select is immutable."""
        node = Select(node_id=1, input_ids=(0,), predicate=gt(col("a"), lit(5)))
        with pytest.raises(AttributeError):
            node.predicate = gt(col("b"), lit(10))  # type: ignore

    def test_project_immutable(self) -> None:
        """Test Project is immutable."""
        node = Project(node_id=1, input_ids=(0,), columns=("a", "b"))
        with pytest.raises(AttributeError):
            node.columns = ("c", "d")  # type: ignore

    def test_join_immutable(self) -> None:
        """Test Join is immutable."""
        node = Join(node_id=2, input_ids=(0, 1), join_type=JoinType.INNER)
        with pytest.raises(AttributeError):
            node.join_type = JoinType.LEFT  # type: ignore

    def test_semantic_topk_immutable(self) -> None:
        """Test SemanticTopK is immutable."""
        spec = SemanticSpec.parse("Rank: {title}")
        node = SemanticTopK(node_id=1, input_ids=(0,), spec=spec, k=10)
        with pytest.raises(AttributeError):
            node.k = 20  # type: ignore