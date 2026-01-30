"""
Classical expression AST for the SoliceDB logical IR.

This module defines the expression tree used for classical (non-semantic)
predicates and computations in relational operators like Select, Project, and Join.

The AST supports:
- Literals (constants)
- Column references
- Binary operations (comparison, boolean, arithmetic)
- Unary operations (NOT, negation)
- Function calls (ABS, LOWER, etc.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from solicedb.logical_ir.types import DType


class BinaryOp(Enum):
    """
    Binary operators for expressions.

    Includes comparison, boolean logic, and arithmetic operators.
    """

    # Comparison operators
    EQ = auto()   # ==
    NE = auto()   # !=
    LT = auto()   # <
    LE = auto()   # <=
    GT = auto()   # >
    GE = auto()   # >=

    # Boolean operators
    AND = auto()
    OR = auto()

    # Arithmetic operators
    ADD = auto()  # +
    SUB = auto()  # -
    MUL = auto()  # *
    DIV = auto()  # /
    MOD = auto()  # %

    def is_comparison(self) -> bool:
        """
        Check if this is a comparison operator.

        Returns
        -------
        bool
            True if this is a comparison operator (EQ, NE, LT, LE, GT, GE).
        """
        return self in (BinaryOp.EQ, BinaryOp.NE, BinaryOp.LT, BinaryOp.LE, BinaryOp.GT, BinaryOp.GE)

    def is_boolean(self) -> bool:
        """
        Check if this is a boolean operator.

        Returns
        -------
        bool
            True if this is a boolean operator (AND, OR).
        """
        return self in (BinaryOp.AND, BinaryOp.OR)

    def is_arithmetic(self) -> bool:
        """
        Check if this is an arithmetic operator.

        Returns
        -------
        bool
            True if this is an arithmetic operator (ADD, SUB, MUL, DIV, MOD).
        """
        return self in (BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL, BinaryOp.DIV, BinaryOp.MOD)

    def symbol(self) -> str:
        """
        Get the symbolic representation of this operator.

        Returns
        -------
        str
            The symbol (e.g., "==", "+", "AND").
        """
        symbols = {
            BinaryOp.EQ: "==",
            BinaryOp.NE: "!=",
            BinaryOp.LT: "<",
            BinaryOp.LE: "<=",
            BinaryOp.GT: ">",
            BinaryOp.GE: ">=",
            BinaryOp.AND: "AND",
            BinaryOp.OR: "OR",
            BinaryOp.ADD: "+",
            BinaryOp.SUB: "-",
            BinaryOp.MUL: "*",
            BinaryOp.DIV: "/",
            BinaryOp.MOD: "%",
        }
        return symbols[self]


class UnaryOp(Enum):
    """
    Unary operators for expressions.
    """

    NOT = auto()  # Boolean negation
    NEG = auto()  # Arithmetic negation (-)

    def symbol(self) -> str:
        """
        Get the symbolic representation of this operator.

        Returns
        -------
        str
            The symbol ("NOT" or "-").
        """
        symbols = {
            UnaryOp.NOT: "NOT",
            UnaryOp.NEG: "-",
        }
        return symbols[self]


class AggFunc(Enum):
    """
    Aggregation functions for GroupBy operations.
    """

    COUNT = auto()
    SUM = auto()
    AVG = auto()
    MIN = auto()
    MAX = auto()
    FIRST = auto()
    LAST = auto()

    def name_lower(self) -> str:
        """
        Get the lowercase name of the function.

        Returns
        -------
        str
            The function name in lowercase (e.g., "count", "sum").
        """
        return self.name.lower()


@dataclass(frozen=True)
class Expr(ABC):
    """
    Abstract base class for all expressions in the AST.

    Expressions are immutable and form a tree structure representing
    computations over column values and literals.
    """

    @abstractmethod
    def children(self) -> tuple[Expr, ...]:
        """
        Return child expressions.

        Returns
        -------
        tuple[Expr, ...]
            Child expressions of this node. Empty for leaf nodes.
        """
        pass

    @abstractmethod
    def to_string(self) -> str:
        """
        Return a human-readable string representation.

        Returns
        -------
        str
            String representation of this expression.
        """
        pass

    def __str__(self) -> str:
        return self.to_string()

    def collect_columns(self) -> frozenset[str]:
        """
        Collect all column references in this expression.

        Returns
        -------
        frozenset[str]
            Set of column names referenced in this expression.
        """
        columns: set[str] = set()
        self._collect_columns_recursive(columns)
        return frozenset(columns)

    def _collect_columns_recursive(self, columns: set[str]) -> None:
        """Recursively collect column names into the provided set."""
        if isinstance(self, Col):
            columns.add(self.name)
        for child in self.children():
            child._collect_columns_recursive(columns)


@dataclass(frozen=True)
class Literal(Expr):
    """
    A constant literal value.

    Parameters
    ----------
    value : Any
        The literal value (int, float, str, bool, or None).
    dtype : DType
        The data type of the literal.

    Examples
    --------
    >>> lit_ = Literal(42, DType.INT)
    >>> lit_.to_string()
    '42'
    >>> lit_ = Literal("hello", DType.STRING)
    >>> lit_.to_string()
    "'hello'"
    """

    value: Any
    dtype: DType

    def children(self) -> tuple[Expr, ...]:
        return ()

    def to_string(self) -> str:
        if self.dtype == DType.STRING:
            return f"'{self.value}'"
        elif self.value is None:
            return "NULL"
        else:
            return str(self.value)

    @classmethod
    def of(cls, value: Any) -> Literal:
        """
        Create a Literal with inferred dtype.

        Parameters
        ----------
        value : Any
            The value to wrap.

        Returns
        -------
        Literal
            A Literal with automatically inferred dtype.

        Raises
        ------
        TypeError
            If the value type is not supported.
        """
        if isinstance(value, bool):
            return cls(value, DType.BOOL)
        elif isinstance(value, int):
            return cls(value, DType.INT)
        elif isinstance(value, float):
            return cls(value, DType.FLOAT)
        elif isinstance(value, str):
            return cls(value, DType.STRING)
        elif value is None:
            # Default to STRING for NULL; may need context-dependent handling
            return cls(None, DType.STRING)
        else:
            raise TypeError(f"Cannot create Literal from type: {type(value)}")


@dataclass(frozen=True)
class Col(Expr):
    """
    A reference to a column.

    Parameters
    ----------
    name : str
        The column name.
    table : str | None
        Optional table qualifier (for joins).

    Examples
    --------
    >>> col_ = Col("age")
    >>> col_.to_string()
    'age'
    >>> col_ = Col("age", table="users")
    >>> col_.to_string()
    'users.age'
    """

    name: str
    table: str | None = None

    def children(self) -> tuple[Expr, ...]:
        return ()

    def to_string(self) -> str:
        if self.table:
            return f"{self.table}.{self.name}"
        return self.name

    def qualified_name(self) -> str:
        """
        Get the fully qualified column name.

        Returns
        -------
        str
            The column name, optionally prefixed with table name.
        """
        return self.to_string()


@dataclass(frozen=True)
class BinaryExpr(Expr):
    """
    A binary operation between two expressions.

    Parameters
    ----------
    op : BinaryOp
        The binary operator.
    left : Expr
        The left operand.
    right : Expr
        The right operand.

    Examples
    --------
    >>> expr = BinaryExpr(BinaryOp.GT, Col("age"), Literal(18, DType.INT))
    >>> expr.to_string()
    '(age > 18)'
    """

    op: BinaryOp
    left: Expr
    right: Expr

    def children(self) -> tuple[Expr, ...]:
        return self.left, self.right

    def to_string(self) -> str:
        return f"({self.left.to_string()} {self.op.symbol()} {self.right.to_string()})"


@dataclass(frozen=True)
class UnaryExpr(Expr):
    """
    A unary operation on an expression.

    Parameters
    ----------
    op : UnaryOp
        The unary operator.
    operand : Expr
        The operand expression.

    Examples
    --------
    >>> expr = UnaryExpr(UnaryOp.NOT, Col("is_active"))
    >>> expr.to_string()
    '(NOT is_active)'
    """

    op: UnaryOp
    operand: Expr

    def children(self) -> tuple[Expr, ...]:
        return (self.operand,)

    def to_string(self) -> str:
        if self.op == UnaryOp.NEG:
            return f"(-{self.operand.to_string()})"
        return f"({self.op.symbol()} {self.operand.to_string()})"


@dataclass(frozen=True)
class Call(Expr):
    """
    A function call expression.

    Parameters
    ----------
    func : str
        The function name (e.g., "ABS", "LOWER", "UPPER").
    args : tuple[Expr, ...]
        The function arguments.

    Examples
    --------
    >>> expr = Call("ABS", (Col("value"),))
    >>> expr.to_string()
    'ABS(value)'
    >>> expr = Call("CONCAT", (Col("first"), Literal(" ", DType.STRING), Col("last")))
    >>> expr.to_string()
    "CONCAT(first, ' ', last)"
    """

    func: str
    args: tuple[Expr, ...]

    def children(self) -> tuple[Expr, ...]:
        return self.args

    def to_string(self) -> str:
        args_str = ", ".join(arg.to_string() for arg in self.args)
        return f"{self.func}({args_str})"


@dataclass(frozen=True)
class Aggregation(Expr):
    """
    An aggregation expression for use in GroupBy.

    Parameters
    ----------
    func : AggFunc
        The aggregation function.
    column : str | None
        The column to aggregate (None for COUNT(*)).
    distinct : bool
        Whether to aggregate distinct values only.

    Examples
    --------
    >>> agg = Aggregation(AggFunc.SUM, "amount")
    >>> agg.to_string()
    'SUM(amount)'
    >>> agg = Aggregation(AggFunc.COUNT, None)
    >>> agg.to_string()
    'COUNT(*)'
    """

    func: AggFunc
    column: str | None = None
    distinct: bool = False

    def children(self) -> tuple[Expr, ...]:
        return ()

    def to_string(self) -> str:
        if self.column is None:
            inner = "*"
        elif self.distinct:
            inner = f"DISTINCT {self.column}"
        else:
            inner = self.column
        return f"{self.func.name}({inner})"


# ============================================================================
# Convenience constructors for building expressions fluently
# ============================================================================


def col(name: str, table: str | None = None) -> Col:
    """
    Create a column reference.

    Parameters
    ----------
    name : str
        Column name.
    table : str | None
        Optional table qualifier.

    Returns
    -------
    Col
        A column reference expression.
    """
    return Col(name, table)


def lit(value: Any) -> Literal:
    """
    Create a literal with inferred type.

    Parameters
    ----------
    value : Any
        The literal value.

    Returns
    -------
    Literal
        A literal expression.
    """
    return Literal.of(value)


# Comparison operators
def eq(left: Expr, right: Expr) -> BinaryExpr:
    """
    Create an equality comparison (==).

    Parameters
    ----------
    left : Expr
        Left operand.
    right : Expr
        Right operand.

    Returns
    -------
    BinaryExpr
        A binary expression representing left == right.
    """
    return BinaryExpr(BinaryOp.EQ, left, right)


def ne(left: Expr, right: Expr) -> BinaryExpr:
    """
    Create a not-equal comparison (!=).

    Parameters
    ----------
    left : Expr
        Left operand.
    right : Expr
        Right operand.

    Returns
    -------
    BinaryExpr
        A binary expression representing left != right.
    """
    return BinaryExpr(BinaryOp.NE, left, right)


def lt(left: Expr, right: Expr) -> BinaryExpr:
    """
    Create a less-than comparison (<).

    Parameters
    ----------
    left : Expr
        Left operand.
    right : Expr
        Right operand.

    Returns
    -------
    BinaryExpr
        A binary expression representing left < right.
    """
    return BinaryExpr(BinaryOp.LT, left, right)


def le(left: Expr, right: Expr) -> BinaryExpr:
    """
    Create a less-than-or-equal comparison (<=).

    Parameters
    ----------
    left : Expr
        Left operand.
    right : Expr
        Right operand.

    Returns
    -------
    BinaryExpr
        A binary expression representing left <= right.
    """
    return BinaryExpr(BinaryOp.LE, left, right)


def gt(left: Expr, right: Expr) -> BinaryExpr:
    """
    Create a greater-than comparison (>).

    Parameters
    ----------
    left : Expr
        Left operand.
    right : Expr
        Right operand.

    Returns
    -------
    BinaryExpr
        A binary expression representing left > right.
    """
    return BinaryExpr(BinaryOp.GT, left, right)


def ge(left: Expr, right: Expr) -> BinaryExpr:
    """
    Create a greater-than-or-equal comparison (>=).

    Parameters
    ----------
    left : Expr
        Left operand.
    right : Expr
        Right operand.

    Returns
    -------
    BinaryExpr
        A binary expression representing left >= right.
    """
    return BinaryExpr(BinaryOp.GE, left, right)


# Boolean operators
def and_(left: Expr, right: Expr) -> BinaryExpr:
    """
    Create a boolean AND.

    Parameters
    ----------
    left : Expr
        Left operand.
    right : Expr
        Right operand.

    Returns
    -------
    BinaryExpr
        A binary expression representing left AND right.
    """
    return BinaryExpr(BinaryOp.AND, left, right)


def or_(left: Expr, right: Expr) -> BinaryExpr:
    """
    Create a boolean OR.

    Parameters
    ----------
    left : Expr
        Left operand.
    right : Expr
        Right operand.

    Returns
    -------
    BinaryExpr
        A binary expression representing left OR right.
    """
    return BinaryExpr(BinaryOp.OR, left, right)


def not_(operand: Expr) -> UnaryExpr:
    """
    Create a boolean NOT.

    Parameters
    ----------
    operand : Expr
        The expression to negate.

    Returns
    -------
    UnaryExpr
        A unary expression representing NOT operand.
    """
    return UnaryExpr(UnaryOp.NOT, operand)


# Arithmetic operators
def add(left: Expr, right: Expr) -> BinaryExpr:
    """
    Create an addition (+).

    Parameters
    ----------
    left : Expr
        Left operand.
    right : Expr
        Right operand.

    Returns
    -------
    BinaryExpr
        A binary expression representing left + right.
    """
    return BinaryExpr(BinaryOp.ADD, left, right)


def sub(left: Expr, right: Expr) -> BinaryExpr:
    """
    Create a subtraction (-).

    Parameters
    ----------
    left : Expr
        Left operand.
    right : Expr
        Right operand.

    Returns
    -------
    BinaryExpr
        A binary expression representing left - right.
    """
    return BinaryExpr(BinaryOp.SUB, left, right)


def mul(left: Expr, right: Expr) -> BinaryExpr:
    """
    Create a multiplication (*).

    Parameters
    ----------
    left : Expr
        Left operand.
    right : Expr
        Right operand.

    Returns
    -------
    BinaryExpr
        A binary expression representing left * right.
    """
    return BinaryExpr(BinaryOp.MUL, left, right)


def div(left: Expr, right: Expr) -> BinaryExpr:
    """
    Create a division (/).

    Parameters
    ----------
    left : Expr
        Left operand.
    right : Expr
        Right operand.

    Returns
    -------
    BinaryExpr
        A binary expression representing left / right.
    """
    return BinaryExpr(BinaryOp.DIV, left, right)


def mod(left: Expr, right: Expr) -> BinaryExpr:
    """
    Create a modulo (%).

    Parameters
    ----------
    left : Expr
        Left operand.
    right : Expr
        Right operand.

    Returns
    -------
    BinaryExpr
        A binary expression representing left % right.
    """
    return BinaryExpr(BinaryOp.MOD, left, right)


def neg(operand: Expr) -> UnaryExpr:
    """
    Create an arithmetic negation (-).

    Parameters
    ----------
    operand : Expr
        The expression to negate.

    Returns
    -------
    UnaryExpr
        A unary expression representing -operand.
    """
    return UnaryExpr(UnaryOp.NEG, operand)


# Aggregations
def count(column: str | None = None, distinct: bool = False) -> Aggregation:
    """
    Create a COUNT aggregation.

    Parameters
    ----------
    column : str | None
        Column to count, or None for COUNT(*).
    distinct : bool
        Whether to count distinct values only.

    Returns
    -------
    Aggregation
        A COUNT aggregation expression.
    """
    return Aggregation(AggFunc.COUNT, column, distinct)


def sum_(column: str, distinct: bool = False) -> Aggregation:
    """
    Create a SUM aggregation.

    Parameters
    ----------
    column : str
        Column to sum.
    distinct : bool
        Whether to sum distinct values only.

    Returns
    -------
    Aggregation
        A SUM aggregation expression.
    """
    return Aggregation(AggFunc.SUM, column, distinct)


def avg(column: str, distinct: bool = False) -> Aggregation:
    """
    Create an AVG aggregation.

    Parameters
    ----------
    column : str
        Column to average.
    distinct : bool
        Whether to average distinct values only.

    Returns
    -------
    Aggregation
        An AVG aggregation expression.
    """
    return Aggregation(AggFunc.AVG, column, distinct)


def min_(column: str) -> Aggregation:
    """
    Create a MIN aggregation.

    Parameters
    ----------
    column : str
        Column to find minimum of.

    Returns
    -------
    Aggregation
        A MIN aggregation expression.
    """
    return Aggregation(AggFunc.MIN, column)


def max_(column: str) -> Aggregation:
    """
    Create a MAX aggregation.

    Parameters
    ----------
    column : str
        Column to find maximum of.

    Returns
    -------
    Aggregation
        A MAX aggregation expression.
    """
    return Aggregation(AggFunc.MAX, column)


def first(column: str) -> Aggregation:
    """
    Create a FIRST aggregation.

    Parameters
    ----------
    column : str
        Column to get first value of.

    Returns
    -------
    Aggregation
        A FIRST aggregation expression.
    """
    return Aggregation(AggFunc.FIRST, column)


def last(column: str) -> Aggregation:
    """
    Create a LAST aggregation.

    Parameters
    ----------
    column : str
        Column to get last value of.

    Returns
    -------
    Aggregation
        A LAST aggregation expression.
    """
    return Aggregation(AggFunc.LAST, column)
