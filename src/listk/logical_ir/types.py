"""
Core type definitions for the SoliceDB logical IR.

This module defines the fundamental types used throughout the IR:
- DType: Enumeration of supported data types
- Schema: Ordered collection of column names and their types
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class DType(Enum):
    """
    All supported data types for table attributes/columns in SoliceDB.

    These types map to pandas/numpy types during execution and are used
    for schema validation and type inference.
    """

    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    # TODO: DATETIME, LIST, JSON

    @classmethod
    def from_pandas_dtype(cls, dtype: np.dtype) -> DType:
        """
        Infer DType from a pandas/numpy dtype.

        Parameters
        ----------
        dtype : np.dtype
            A pandas or numpy dtype object.

        Returns
        -------
        DType
            The corresponding SoliceDB DType.

        Raises
        ------
        ValueError
            If the dtype cannot be mapped to a supported DType.
        """
        dtype_str = str(dtype).lower()

        if "int" in dtype_str:
            return cls.INT
        elif "float" in dtype_str:
            return cls.FLOAT
        elif "bool" in dtype_str:
            return cls.BOOL
        elif "object" in dtype_str or "string" in dtype_str:
            return cls.STRING
        else:
            raise ValueError(f"Unsupported pandas dtype: {dtype}")

    def to_python_type(self) -> type:
        """
        Return the corresponding Python type.

        Returns
        -------
        type
            The Python type corresponding to this DType.
        """
        mapping = {
            DType.STRING: str,
            DType.INT: int,
            DType.FLOAT: float,
            DType.BOOL: bool,
        }
        return mapping[self]


@dataclass(frozen=True)
class Schema:
    """
    An ordered collection of column names and their data types.

    Schema is immutable and represents the structure of a relation (table).
    Column order is preserved, which matters for operations like SELECT *.

    Parameters
    ----------
    columns : tuple[tuple[str, DType], ...]
        Ordered sequence of (column_name, dtype) pairs.

    Examples
    --------
    >>> schema = Schema((("id", DType.INT), ("name", DType.STRING)))
    >>> schema.column_names()
    ['id', 'name']
    >>> schema.dtype_of("name")
    <DType.STRING: 'string'>

    Raises
    ------
    ValueError
        If the column names are not unique.
    """

    columns: tuple[tuple[str, DType], ...]

    def __post_init__(self) -> None:
        """
        Validates that column names are unique.
        """
        names = [name for name, _ in self.columns]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate column names: {set(duplicates)}")

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Schema:
        """
        Infer schema from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to extract schema from.

        Returns
        -------
        Schema
            A Schema object representing the DataFrame's structure.

        Examples
        --------
        >>> import pandas
        >>> dataframe = pandas.DataFrame({"id": [1, 2], "name": ["a", "b"]})
        >>> schema = Schema.from_dataframe(dataframe)
        >>> schema.column_names()
        ['id', 'name']
        """
        return cls(tuple((col, DType.from_pandas_dtype(df[col].dtype)) for col in df.columns))

    @classmethod
    def from_dict(cls, d: dict[str, DType | str]) -> Schema:
        """
        Create schema from a dictionary.

        Parameters
        ----------
        d : dict[str, DType | str]
            Mapping of column names to types. Types can be DType enums
            or string names like "int", "string", etc.

        Returns
        -------
        Schema
            A Schema object.

        Examples
        --------
        >>> schema = Schema.from_dict({"id": DType.INT, "name": "string"})
        """
        columns = []
        for name, dtype in d.items():
            if isinstance(dtype, str):
                dtype = DType(dtype)
            columns.append((name, dtype))
        return cls(tuple(columns))

    def column_names(self) -> list[str]:
        """
        Return list of column names in order.

        Returns
        -------
        list[str]
            Ordered list of column names.
        """
        return [name for name, _ in self.columns]

    def dtypes(self) -> list[DType]:
        """
        Return list of column dtypes in order.

        Returns
        -------
        list[DType]
            Ordered list of column dtypes.
        """
        return [dtype for _, dtype in self.columns]

    def dtype_of(self, col: str) -> DType:
        """
        Get the dtype of a specific column.

        Parameters
        ----------
        col : str
            Column name.

        Returns
        -------
        DType
            The dtype of the column.

        Raises
        ------
        KeyError
            If the column does not exist.
        """
        for name, dtype in self.columns:
            if name == col:
                return dtype
        raise KeyError(f"Column not found: {col}")

    def has_column(self, col: str) -> bool:
        """
        Check if a column exists in the schema.

        Parameters
        ----------
        col : str
            Column name to check.

        Returns
        -------
        bool
            True if column exists, False otherwise.
        """
        return col in self.column_names()

    def select(self, cols: list[str]) -> Schema:
        """
        Project schema to a subset of columns.

        Parameters
        ----------
        cols : list[str]
            Column names to keep, in the desired order.

        Returns
        -------
        Schema
            A new Schema with only the specified columns.

        Raises
        ------
        KeyError
            If any column does not exist.
        """
        col_map = {name: dtype for name, dtype in self.columns}
        new_columns = []
        for col in cols:
            if col not in col_map:
                raise KeyError(f"Column not found: {col}")
            new_columns.append((col, col_map[col]))
        return Schema(tuple(new_columns))

    def add(self, name: str, dtype: DType) -> Schema:
        """
        Add a new column to the schema.

        Parameters
        ----------
        name : str
            Name of the new column.
        dtype : DType
            Type of the new column.

        Returns
        -------
        Schema
            A new Schema with the additional column appended.

        Raises
        ------
        ValueError
            If a column with the same name already exists.
        """
        if self.has_column(name):
            raise ValueError(f"Column already exists: {name}")
        return Schema(self.columns + ((name, dtype),))

    def remove(self, col: str) -> Schema:
        """
        Remove a column from the schema.

        Parameters
        ----------
        col : str
            Name of the column to remove.

        Returns
        -------
        Schema
            A new Schema without the specified column.

        Raises
        ------
        KeyError
            If the column does not exist.
        """
        if not self.has_column(col):
            raise KeyError(f"Column not found: {col}")
        return Schema(tuple((n, d) for n, d in self.columns if n != col))

    def rename(self, old_name: str, new_name: str) -> Schema:
        """
        Rename a column in the schema.

        Parameters
        ----------
        old_name : str
            Current column name.
        new_name : str
            New column name.

        Returns
        -------
        Schema
            A new Schema with the column renamed.

        Raises
        ------
        KeyError
            If the old column does not exist.
        ValueError
            If the new name already exists.
        """
        if not self.has_column(old_name):
            raise KeyError(f"Column not found: {old_name}")
        if self.has_column(new_name):
            raise ValueError(f"Column already exists: {new_name}")
        return Schema(
            tuple(
                (new_name if n == old_name else n, d) for n, d in self.columns
            )
        )

    def merge(self, other: Schema, prefix_left: str = "", prefix_right: str = "") -> Schema:
        """
        Merge two schemas (for join operations).

        Parameters
        ----------
        other : Schema
            The other schema to merge with.
        prefix_left : str, optional
            Prefix to add to this schema's column names if there are conflicts.
        prefix_right : str, optional
            Prefix to add to the other schema's column names if there are conflicts.

        Returns
        -------
        Schema
            A new Schema containing columns from both schemas.
        """
        left_names = set(self.column_names())
        right_names = set(other.column_names())
        conflicts = left_names & right_names

        new_columns = []

        for name, dtype in self.columns:
            if name in conflicts and prefix_left:
                new_columns.append((f"{prefix_left}{name}", dtype))
            else:
                new_columns.append((name, dtype))

        for name, dtype in other.columns:
            if name in conflicts and prefix_right:
                new_columns.append((f"{prefix_right}{name}", dtype))
            elif name not in conflicts:
                new_columns.append((name, dtype))
            else:
                # Conflict with no prefix - for now, include with original name
                # TODO: Verify how to handle join name conflict with no left/right prefix
                new_columns.append((name, dtype))

        return Schema(tuple(new_columns))

    def is_compatible_with(self, other: Schema) -> bool:
        """
        Check if another schema is compatible (subset) with this schema.

        A schema is compatible if all its columns exist in this schema
        with matching types. This is used to validate that a DataFrame
        matches an expected schema.

        Parameters
        ----------
        other : Schema
            The schema to check compatibility against.

        Returns
        -------
        bool
            True if other is compatible with self.
        """
        for name, dtype in other.columns:
            if not self.has_column(name):
                return False
            if self.dtype_of(name) != dtype:
                return False
        return True

    def to_dict(self) -> dict[str, str]:
        """
        Convert schema to a dictionary (for serialization).

        Returns
        -------
        dict[str, str]
            Mapping of column names to dtype string values.
        """
        return {name: dtype.value for name, dtype in self.columns}

    def __len__(self) -> int:
        """Return the number of columns."""
        return len(self.columns)

    def __iter__(self):
        """Iterate over (name, dtype) pairs."""
        return iter(self.columns)

    def __contains__(self, col: str) -> bool:
        """Check if column exists."""
        return self.has_column(col)
