"""
Tests for solicedb.logical_ir.expressions module.

Covers:
- BinaryOp, UnaryOp, AggFunc enums
- Expr AST nodes (Literal, Col, BinaryExpr, UnaryExpr, Call, Aggregation)
- Convenience constructor functions
"""

import pytest

from solicedb.logical_ir import DType
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


class TestBinaryOp:
    """Tests for the BinaryOp enum."""

    def test_comparison_operators(self) -> None:
        """Test classification of comparison operators."""
        assert BinaryOp.EQ.is_comparison()
        assert BinaryOp.NE.is_comparison()
        assert BinaryOp.LT.is_comparison()
        assert BinaryOp.LE.is_comparison()
        assert BinaryOp.GT.is_comparison()
        assert BinaryOp.GE.is_comparison()

    def test_boolean_operators(self) -> None:
        """Test classification of boolean operators."""
        assert BinaryOp.AND.is_boolean()
        assert BinaryOp.OR.is_boolean()
        assert not BinaryOp.EQ.is_boolean()

    def test_arithmetic_operators(self) -> None:
        """Test classification of arithmetic operators."""
        assert BinaryOp.ADD.is_arithmetic()
        assert BinaryOp.SUB.is_arithmetic()
        assert BinaryOp.MUL.is_arithmetic()
        assert BinaryOp.DIV.is_arithmetic()
        assert BinaryOp.MOD.is_arithmetic()
        assert not BinaryOp.AND.is_arithmetic()

    def test_symbols(self) -> None:
        """Test symbol() method returns correct symbols."""
        assert BinaryOp.EQ.symbol() == "=="
        assert BinaryOp.NE.symbol() == "!="
        assert BinaryOp.LT.symbol() == "<"
        assert BinaryOp.LE.symbol() == "<="
        assert BinaryOp.GT.symbol() == ">"
        assert BinaryOp.GE.symbol() == ">="
        assert BinaryOp.AND.symbol() == "AND"
        assert BinaryOp.OR.symbol() == "OR"
        assert BinaryOp.ADD.symbol() == "+"
        assert BinaryOp.SUB.symbol() == "-"
        assert BinaryOp.MUL.symbol() == "*"
        assert BinaryOp.DIV.symbol() == "/"
        assert BinaryOp.MOD.symbol() == "%"


class TestUnaryOp:
    """Tests for the UnaryOp enum."""

    def test_symbol(self) -> None:
        """Test symbol() method."""
        assert UnaryOp.NOT.symbol() == "NOT"
        assert UnaryOp.NEG.symbol() == "-"


class TestAggFunc:
    """Tests for the AggFunc enum."""

    def test_name_lower(self) -> None:
        """Test name_lower() method."""
        assert AggFunc.COUNT.name_lower() == "count"
        assert AggFunc.SUM.name_lower() == "sum"
        assert AggFunc.AVG.name_lower() == "avg"
        assert AggFunc.MIN.name_lower() == "min"
        assert AggFunc.MAX.name_lower() == "max"
        assert AggFunc.FIRST.name_lower() == "first"
        assert AggFunc.LAST.name_lower() == "last"


class TestLiteral:
    """Tests for the Literal expression node."""

    def test_literal_creation(self) -> None:
        """Test basic Literal creation."""
        lit_ = Literal(42, DType.INT)
        assert lit_.value == 42
        assert lit_.dtype == DType.INT

    def test_literal_children_empty(self) -> None:
        """Test that Literal has no children."""
        lit_ = Literal(42, DType.INT)
        assert lit_.children() == ()

    def test_literal_to_string_int(self) -> None:
        """Test string representation of int literal."""
        lit_ = Literal(42, DType.INT)
        assert lit_.to_string() == "42"

    def test_literal_to_string_float(self) -> None:
        """Test string representation of float literal."""
        lit_ = Literal(3.14, DType.FLOAT)
        assert lit_.to_string() == "3.14"

    def test_literal_to_string_string(self) -> None:
        """Test string representation of string literal (with quotes)."""
        lit_ = Literal("hello", DType.STRING)
        assert lit_.to_string() == "'hello'"

    def test_literal_to_string_bool(self) -> None:
        """Test string representation of bool literal."""
        lit_ = Literal(True, DType.BOOL)
        assert lit_.to_string() == "True"

    def test_literal_to_string_null(self) -> None:
        """Test string representation of NULL literal."""
        # Note: When dtype is STRING, None is quoted as 'None'
        # NULL representation is used for non-STRING types with None value
        lit_str = Literal(None, DType.STRING)
        assert lit_str.to_string() == "'None'"

        # For non-STRING dtypes, None displays as NULL
        lit_int = Literal(None, DType.INT)
        assert lit_int.to_string() == "NULL"

    def test_literal_of_int(self) -> None:
        """Test Literal.of() with int."""
        lit_ = Literal.of(42)
        assert lit_.value == 42
        assert lit_.dtype == DType.INT

    def test_literal_of_float(self) -> None:
        """Test Literal.of() with float."""
        lit_ = Literal.of(3.14)
        assert lit_.value == 3.14
        assert lit_.dtype == DType.FLOAT

    def test_literal_of_string(self) -> None:
        """Test Literal.of() with string."""
        lit_ = Literal.of("hello")
        assert lit_.value == "hello"
        assert lit_.dtype == DType.STRING

    def test_literal_of_bool(self) -> None:
        """Test Literal.of() with bool (must be checked before int)."""
        lit_ = Literal.of(True)
        assert lit_.value is True
        assert lit_.dtype == DType.BOOL

    def test_literal_of_none(self) -> None:
        """Test Literal.of() with None."""
        lit_ = Literal.of(None)
        assert lit_.value is None
        assert lit_.dtype == DType.STRING  # Default for NULL

    def test_literal_of_unsupported_type(self) -> None:
        """Test Literal.of() raises TypeError for unsupported types."""
        with pytest.raises(TypeError, match="Cannot create Literal"):
            Literal.of([1, 2, 3])

    def test_literal_str_method(self) -> None:
        """Test __str__ method delegates to to_string."""
        lit_ = Literal(42, DType.INT)
        assert str(lit_) == "42"


class TestCol:
    """Tests for the Col (column reference) expression node."""

    def test_col_creation(self) -> None:
        """Test basic Col creation."""
        c = Col("age")
        assert c.name == "age"
        assert c.table is None

    def test_col_with_table(self) -> None:
        """Test Col creation with table qualifier."""
        c = Col("age", table="users")
        assert c.name == "age"
        assert c.table == "users"

    def test_col_children_empty(self) -> None:
        """Test that Col has no children."""
        c = Col("age")
        assert c.children() == ()

    def test_col_to_string_simple(self) -> None:
        """Test string representation without table."""
        c = Col("age")
        assert c.to_string() == "age"

    def test_col_to_string_qualified(self) -> None:
        """Test string representation with table qualifier."""
        c = Col("age", table="users")
        assert c.to_string() == "users.age"

    def test_col_qualified_name(self) -> None:
        """Test qualified_name() method."""
        c1 = Col("age")
        c2 = Col("age", table="users")
        assert c1.qualified_name() == "age"
        assert c2.qualified_name() == "users.age"


class TestBinaryExpr:
    """Tests for the BinaryExpr expression node."""

    def test_binary_expr_creation(self) -> None:
        """Test basic BinaryExpr creation."""
        left = Col("age")
        right = Literal(18, DType.INT)
        expr = BinaryExpr(BinaryOp.GT, left, right)
        assert expr.op == BinaryOp.GT
        assert expr.left == left
        assert expr.right == right

    def test_binary_expr_children(self) -> None:
        """Test children() returns both operands."""
        left = Col("age")
        right = Literal(18, DType.INT)
        expr = BinaryExpr(BinaryOp.GT, left, right)
        assert expr.children() == (left, right)

    def test_binary_expr_to_string(self) -> None:
        """Test string representation."""
        expr = BinaryExpr(BinaryOp.GT, Col("age"), Literal(18, DType.INT))
        assert expr.to_string() == "(age > 18)"

    def test_nested_binary_expr(self) -> None:
        """Test nested binary expressions."""
        inner = BinaryExpr(BinaryOp.GT, Col("age"), Literal(18, DType.INT))
        outer = BinaryExpr(BinaryOp.AND, inner, BinaryExpr(
            BinaryOp.LT, Col("age"), Literal(65, DType.INT)
        ))
        assert "AND" in outer.to_string()
        assert "(age > 18)" in outer.to_string()


class TestUnaryExpr:
    """Tests for the UnaryExpr expression node."""

    def test_unary_expr_not(self) -> None:
        """Test NOT unary expression."""
        operand = Col("is_active")
        expr = UnaryExpr(UnaryOp.NOT, operand)
        assert expr.op == UnaryOp.NOT
        assert expr.operand == operand

    def test_unary_expr_neg(self) -> None:
        """Test NEG unary expression."""
        operand = Col("value")
        expr = UnaryExpr(UnaryOp.NEG, operand)
        assert expr.op == UnaryOp.NEG

    def test_unary_expr_children(self) -> None:
        """Test children() returns single operand."""
        operand = Col("is_active")
        expr = UnaryExpr(UnaryOp.NOT, operand)
        assert expr.children() == (operand,)

    def test_unary_expr_to_string_not(self) -> None:
        """Test string representation of NOT."""
        expr = UnaryExpr(UnaryOp.NOT, Col("is_active"))
        assert expr.to_string() == "(NOT is_active)"

    def test_unary_expr_to_string_neg(self) -> None:
        """Test string representation of NEG."""
        expr = UnaryExpr(UnaryOp.NEG, Col("value"))
        assert expr.to_string() == "(-value)"


class TestCall:
    """Tests for the Call (function call) expression node."""

    def test_call_creation(self) -> None:
        """Test basic Call creation."""
        call = Call("ABS", (Col("value"),))
        assert call.func == "ABS"
        assert len(call.args) == 1

    def test_call_children(self) -> None:
        """Test children() returns args."""
        arg = Col("value")
        call = Call("ABS", (arg,))
        assert call.children() == (arg,)

    def test_call_to_string_single_arg(self) -> None:
        """Test string representation with single argument."""
        call = Call("ABS", (Col("value"),))
        assert call.to_string() == "ABS(value)"

    def test_call_to_string_multiple_args(self) -> None:
        """Test string representation with multiple arguments."""
        call = Call("CONCAT", (Col("first"), Literal(" ", DType.STRING), Col("last")))
        assert call.to_string() == "CONCAT(first, ' ', last)"


class TestAggregation:
    """Tests for the Aggregation expression node."""

    def test_aggregation_creation(self) -> None:
        """Test basic Aggregation creation."""
        agg = Aggregation(AggFunc.SUM, "amount")
        assert agg.func == AggFunc.SUM
        assert agg.column == "amount"
        assert agg.distinct is False

    def test_aggregation_count_star(self) -> None:
        """Test COUNT(*) aggregation."""
        agg = Aggregation(AggFunc.COUNT, None)
        assert agg.column is None
        assert agg.to_string() == "COUNT(*)"

    def test_aggregation_with_distinct(self) -> None:
        """Test aggregation with DISTINCT."""
        agg = Aggregation(AggFunc.COUNT, "category", distinct=True)
        assert agg.distinct is True
        assert agg.to_string() == "COUNT(DISTINCT category)"

    def test_aggregation_children_empty(self) -> None:
        """Test that Aggregation has no children (it's a leaf)."""
        agg = Aggregation(AggFunc.SUM, "amount")
        assert agg.children() == ()

    def test_aggregation_to_string(self) -> None:
        """Test string representation for various aggregations."""
        assert Aggregation(AggFunc.SUM, "amount").to_string() == "SUM(amount)"
        assert Aggregation(AggFunc.AVG, "rating").to_string() == "AVG(rating)"
        assert Aggregation(AggFunc.MIN, "price").to_string() == "MIN(price)"
        assert Aggregation(AggFunc.MAX, "score").to_string() == "MAX(score)"


class TestCollectColumns:
    """Tests for the collect_columns method on Expr."""

    def test_literal_no_columns(self) -> None:
        """Test that literals have no column references."""
        lit_ = Literal(42, DType.INT)
        assert lit_.collect_columns() == frozenset()

    def test_col_single_column(self) -> None:
        """Test column collection from Col."""
        c = Col("age")
        assert c.collect_columns() == frozenset({"age"})

    def test_binary_expr_multiple_columns(self) -> None:
        """Test column collection from binary expression."""
        expr = BinaryExpr(BinaryOp.ADD, Col("a"), Col("b"))
        assert expr.collect_columns() == frozenset({"a", "b"})

    def test_nested_expr_collects_all(self) -> None:
        """Test column collection from deeply nested expression."""
        expr = BinaryExpr(
            BinaryOp.AND,
            BinaryExpr(BinaryOp.GT, Col("age"), Literal(18, DType.INT)),
            BinaryExpr(BinaryOp.EQ, Col("status"), Literal("active", DType.STRING)),
        )
        assert expr.collect_columns() == frozenset({"age", "status"})


class TestConvenienceConstructors:
    """Tests for the convenience constructor functions."""

    def test_col_function(self) -> None:
        """Test col() convenience function."""
        c1 = col("age")
        assert isinstance(c1, Col)
        assert c1.name == "age"

        c2 = col("age", "users")
        assert c2.table == "users"

    def test_lit_function(self) -> None:
        """Test lit() convenience function."""
        l1 = lit(42)
        assert isinstance(l1, Literal)
        assert l1.value == 42

    def test_comparison_functions(self) -> None:
        """Test comparison convenience functions."""
        left, right = col("a"), lit(5)

        assert isinstance(eq(left, right), BinaryExpr)
        assert eq(left, right).op == BinaryOp.EQ

        assert ne(left, right).op == BinaryOp.NE
        assert lt(left, right).op == BinaryOp.LT
        assert le(left, right).op == BinaryOp.LE
        assert gt(left, right).op == BinaryOp.GT
        assert ge(left, right).op == BinaryOp.GE

    def test_boolean_functions(self) -> None:
        """Test boolean convenience functions."""
        left = gt(col("a"), lit(5))
        right = lt(col("a"), lit(10))

        assert and_(left, right).op == BinaryOp.AND
        assert or_(left, right).op == BinaryOp.OR
        assert isinstance(not_(left), UnaryExpr)
        assert not_(left).op == UnaryOp.NOT

    def test_arithmetic_functions(self) -> None:
        """Test arithmetic convenience functions."""
        left, right = col("a"), col("b")

        assert add(left, right).op == BinaryOp.ADD
        assert sub(left, right).op == BinaryOp.SUB
        assert mul(left, right).op == BinaryOp.MUL
        assert div(left, right).op == BinaryOp.DIV
        assert mod(left, right).op == BinaryOp.MOD

        assert isinstance(neg(left), UnaryExpr)
        assert neg(left).op == UnaryOp.NEG

    def test_aggregation_functions(self) -> None:
        """Test aggregation convenience functions."""
        assert isinstance(count(), Aggregation)
        assert count().func == AggFunc.COUNT
        assert count().column is None

        assert count("id").column == "id"
        assert count("id", distinct=True).distinct is True

        assert sum_("amount").func == AggFunc.SUM
        assert sum_("amount").column == "amount"
        assert sum_("amount", distinct=True).distinct is True

        assert avg("rating").func == AggFunc.AVG
        assert avg("rating", distinct=True).distinct is True

        assert min_("price").func == AggFunc.MIN
        assert max_("score").func == AggFunc.MAX
        assert first("value").func == AggFunc.FIRST
        assert last("value").func == AggFunc.LAST


class TestExprImmutability:
    """Tests for expression immutability."""

    def test_literal_immutable(self) -> None:
        """Test that Literal is frozen."""
        lit_ = Literal(42, DType.INT)
        with pytest.raises(AttributeError):
            lit_.value = 100  # type: ignore

    def test_col_immutable(self) -> None:
        """Test that Col is frozen."""
        c = Col("age")
        with pytest.raises(AttributeError):
            c.name = "other"  # type: ignore

    def test_binary_expr_immutable(self) -> None:
        """Test that BinaryExpr is frozen."""
        expr = BinaryExpr(BinaryOp.GT, Col("a"), Literal(5, DType.INT))
        with pytest.raises(AttributeError):
            expr.op = BinaryOp.LT  # type: ignore