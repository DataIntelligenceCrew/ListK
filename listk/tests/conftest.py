"""
Pytest configuration and shared fixtures for SoliceDB tests.
"""

import pandas as pd
import pytest

from solicedb.logical_ir import (
    DType,
    Schema,
    QueryPlan,
    col,
    lit,
    gt,
    eq,
    and_,
    count,
    avg,
    SemanticSpec,
    JoinType,
    SortDirection,
)


@pytest.fixture
def sample_schema() -> Schema:
    """Create a sample schema for testing."""
    return Schema.from_dict({
        "id": DType.INT,
        "title": DType.STRING,
        "year": DType.INT,
        "rating": DType.FLOAT,
        "is_classic": DType.BOOL,
    })


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "title": ["The Matrix", "Inception", "Interstellar"],
        "year": [1999, 2010, 2014],
        "rating": [8.7, 8.8, 8.6],
        "is_classic": [True, False, False],
    })


@pytest.fixture
def empty_plan() -> QueryPlan:
    """Create an empty QueryPlan for testing."""
    return QueryPlan()


@pytest.fixture
def simple_plan() -> QueryPlan:
    """Create a simple QueryPlan with source and filter."""
    plan = QueryPlan()
    source_id = plan.source("movies", {"title": DType.STRING, "year": DType.INT})
    filter_id = plan.select(source_id, gt(col("year"), lit(2000)))
    plan.set_root(filter_id)
    return plan