"""
Tests for solicedb.logical_ir.types module.

Covers:
- DType enum and its methods
- Schema class and all its operations
"""

import numpy as np
import pandas as pd
import pytest

from solicedb.logical_ir import DType, Schema


class TestDType:
    """Tests for the DType enum."""

    def test_dtype_values(self) -> None:
        """Test that DType has expected values."""
        assert DType.STRING.value == "string"
        assert DType.INT.value == "int"
        assert DType.FLOAT.value == "float"
        assert DType.BOOL.value == "bool"

    def test_from_pandas_dtype_int(self) -> None:
        """Test DType inference from integer pandas dtypes."""
        assert DType.from_pandas_dtype(np.dtype("int32")) == DType.INT
        assert DType.from_pandas_dtype(np.dtype("int64")) == DType.INT
        assert DType.from_pandas_dtype(np.dtype("uint8")) == DType.INT

    def test_from_pandas_dtype_float(self) -> None:
        """Test DType inference from float pandas dtypes."""
        assert DType.from_pandas_dtype(np.dtype("float32")) == DType.FLOAT
        assert DType.from_pandas_dtype(np.dtype("float64")) == DType.FLOAT

    def test_from_pandas_dtype_bool(self) -> None:
        """Test DType inference from boolean pandas dtype."""
        assert DType.from_pandas_dtype(np.dtype("bool")) == DType.BOOL

    def test_from_pandas_dtype_string(self) -> None:
        """Test DType inference from object/string pandas dtypes."""
        assert DType.from_pandas_dtype(np.dtype("object")) == DType.STRING

    def test_from_pandas_dtype_unsupported(self) -> None:
        """Test that unsupported dtypes raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported pandas dtype"):
            DType.from_pandas_dtype(np.dtype("datetime64[ns]"))

    def test_to_python_type(self) -> None:
        """Test conversion to Python types."""
        assert DType.STRING.to_python_type() == str
        assert DType.INT.to_python_type() == int
        assert DType.FLOAT.to_python_type() == float
        assert DType.BOOL.to_python_type() == bool


class TestSchema:
    """Tests for the Schema class."""

    def test_schema_creation(self) -> None:
        """Test basic Schema creation."""
        schema = Schema((("id", DType.INT), ("name", DType.STRING)))
        assert len(schema) == 2
        assert schema.column_names() == ["id", "name"]

    def test_schema_duplicate_columns_raises(self) -> None:
        """Test that duplicate column names raise ValueError."""
        with pytest.raises(ValueError, match="Duplicate column names"):
            Schema((("id", DType.INT), ("id", DType.STRING)))

    def test_from_dict(self) -> None:
        """Test Schema creation from dictionary."""
        schema = Schema.from_dict({"id": DType.INT, "name": DType.STRING})
        assert schema.column_names() == ["id", "name"]
        assert schema.dtype_of("id") == DType.INT
        assert schema.dtype_of("name") == DType.STRING

    def test_from_dict_with_string_types(self) -> None:
        """Test Schema creation from dict with string type values."""
        schema = Schema.from_dict({"id": "int", "name": "string"})
        assert schema.dtype_of("id") == DType.INT
        assert schema.dtype_of("name") == DType.STRING

    def test_from_dataframe(self, sample_dataframe: pd.DataFrame) -> None:
        """Test Schema inference from DataFrame."""
        schema = Schema.from_dataframe(sample_dataframe)
        assert "id" in schema.column_names()
        assert "title" in schema.column_names()
        assert schema.dtype_of("id") == DType.INT
        assert schema.dtype_of("title") == DType.STRING

    def test_column_names(self, sample_schema: Schema) -> None:
        """Test column_names returns ordered list."""
        names = sample_schema.column_names()
        assert isinstance(names, list)
        assert "id" in names
        assert "title" in names

    def test_dtypes(self, sample_schema: Schema) -> None:
        """Test dtypes returns ordered list of DTypes."""
        dtypes = sample_schema.dtypes()
        assert all(isinstance(dt, DType) for dt in dtypes)

    def test_dtype_of_existing(self, sample_schema: Schema) -> None:
        """Test dtype_of for existing column."""
        assert sample_schema.dtype_of("id") == DType.INT
        assert sample_schema.dtype_of("rating") == DType.FLOAT

    def test_dtype_of_nonexistent_raises(self, sample_schema: Schema) -> None:
        """Test dtype_of raises KeyError for nonexistent column."""
        with pytest.raises(KeyError, match="Column not found"):
            sample_schema.dtype_of("nonexistent")

    def test_has_column(self, sample_schema: Schema) -> None:
        """Test has_column method."""
        assert sample_schema.has_column("id") is True
        assert sample_schema.has_column("title") is True
        assert sample_schema.has_column("nonexistent") is False

    def test_contains_operator(self, sample_schema: Schema) -> None:
        """Test __contains__ operator."""
        assert "id" in sample_schema
        assert "nonexistent" not in sample_schema

    def test_select(self, sample_schema: Schema) -> None:
        """Test schema projection to subset of columns."""
        projected = sample_schema.select(["title", "year"])
        assert projected.column_names() == ["title", "year"]
        assert len(projected) == 2

    def test_select_preserves_order(self, sample_schema: Schema) -> None:
        """Test that select preserves requested column order."""
        projected = sample_schema.select(["year", "title"])
        assert projected.column_names() == ["year", "title"]

    def test_select_nonexistent_raises(self, sample_schema: Schema) -> None:
        """Test select raises KeyError for nonexistent column."""
        with pytest.raises(KeyError, match="Column not found"):
            sample_schema.select(["title", "nonexistent"])

    def test_add(self, sample_schema: Schema) -> None:
        """Test adding a new column."""
        new_schema = sample_schema.add("genre", DType.STRING)
        assert new_schema.has_column("genre")
        assert new_schema.dtype_of("genre") == DType.STRING
        assert len(new_schema) == len(sample_schema) + 1

    def test_add_existing_raises(self, sample_schema: Schema) -> None:
        """Test add raises ValueError for existing column."""
        with pytest.raises(ValueError, match="Column already exists"):
            sample_schema.add("id", DType.INT)

    def test_remove(self, sample_schema: Schema) -> None:
        """Test removing a column."""
        new_schema = sample_schema.remove("rating")
        assert not new_schema.has_column("rating")
        assert len(new_schema) == len(sample_schema) - 1

    def test_remove_nonexistent_raises(self, sample_schema: Schema) -> None:
        """Test remove raises KeyError for nonexistent column."""
        with pytest.raises(KeyError, match="Column not found"):
            sample_schema.remove("nonexistent")

    def test_rename(self, sample_schema: Schema) -> None:
        """Test renaming a column."""
        new_schema = sample_schema.rename("title", "movie_title")
        assert not new_schema.has_column("title")
        assert new_schema.has_column("movie_title")
        assert new_schema.dtype_of("movie_title") == DType.STRING

    def test_rename_nonexistent_raises(self, sample_schema: Schema) -> None:
        """Test rename raises KeyError for nonexistent source column."""
        with pytest.raises(KeyError, match="Column not found"):
            sample_schema.rename("nonexistent", "new_name")

    def test_rename_to_existing_raises(self, sample_schema: Schema) -> None:
        """Test rename raises ValueError when target name exists."""
        with pytest.raises(ValueError, match="Column already exists"):
            sample_schema.rename("title", "id")

    def test_merge_no_conflicts(self) -> None:
        """Test merging schemas with no column conflicts."""
        schema1 = Schema.from_dict({"id": DType.INT, "name": DType.STRING})
        schema2 = Schema.from_dict({"age": DType.INT, "email": DType.STRING})
        merged = schema1.merge(schema2)
        assert len(merged) == 4
        assert merged.has_column("id")
        assert merged.has_column("age")

    def test_merge_with_conflicts_and_prefixes(self) -> None:
        """Test merging schemas with conflicts using prefixes."""
        schema1 = Schema.from_dict({"id": DType.INT, "name": DType.STRING})
        schema2 = Schema.from_dict({"id": DType.INT, "email": DType.STRING})
        merged = schema1.merge(schema2, prefix_left="left_", prefix_right="right_")
        assert merged.has_column("left_id")
        assert merged.has_column("right_id")
        assert merged.has_column("name")
        assert merged.has_column("email")

    def test_merge_with_conflicts_no_prefix(self) -> None:
        """Test merging schemas with conflicts but no prefix raises error.

        When there are column name conflicts and no prefixes are provided,
        the merge will result in duplicate column names, which Schema
        rejects with a ValueError.
        """
        schema1 = Schema.from_dict({"id": DType.INT, "name": DType.STRING})
        schema2 = Schema.from_dict({"id": DType.INT, "email": DType.STRING})
        with pytest.raises(ValueError, match="Duplicate column names"):
            schema1.merge(schema2)

    def test_is_compatible_with(self) -> None:
        """Test schema compatibility checking."""
        full_schema = Schema.from_dict({
            "id": DType.INT,
            "name": DType.STRING,
            "age": DType.INT,
        })
        subset_schema = Schema.from_dict({
            "id": DType.INT,
            "name": DType.STRING,
        })
        assert full_schema.is_compatible_with(subset_schema)

    def test_is_compatible_with_missing_column(self) -> None:
        """Test compatibility fails for missing columns."""
        schema1 = Schema.from_dict({"id": DType.INT})
        schema2 = Schema.from_dict({"id": DType.INT, "name": DType.STRING})
        assert not schema1.is_compatible_with(schema2)

    def test_is_compatible_with_wrong_type(self) -> None:
        """Test compatibility fails for mismatched types."""
        schema1 = Schema.from_dict({"id": DType.INT})
        schema2 = Schema.from_dict({"id": DType.STRING})
        assert not schema1.is_compatible_with(schema2)

    def test_to_dict(self, sample_schema: Schema) -> None:
        """Test conversion to dictionary."""
        d = sample_schema.to_dict()
        assert isinstance(d, dict)
        assert d["id"] == "int"
        assert d["title"] == "string"

    def test_len(self, sample_schema: Schema) -> None:
        """Test __len__ returns column count."""
        assert len(sample_schema) == 5

    def test_iter(self, sample_schema: Schema) -> None:
        """Test __iter__ yields (name, dtype) pairs."""
        columns = list(sample_schema)
        assert all(isinstance(c, tuple) and len(c) == 2 for c in columns)
        assert columns[0][0] == "id"
        assert columns[0][1] == DType.INT

    def test_schema_immutability(self, sample_schema: Schema) -> None:
        """Test that Schema is immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            sample_schema.columns = ()  # type: ignore