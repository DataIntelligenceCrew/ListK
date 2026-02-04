"""
Semantic specification types for the SoliceDB logical IR.

This module defines structured representations for semantic operator specifications:
- SemanticSpec: Parsed NL template with explicit column dependencies
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Special marker for referencing all columns
ALL_COLUMNS_MARKER = "ALL_COLS"


@dataclass(frozen=True)
class SemanticSpec:
    """
    A parsed semantic specification with explicit column dependencies.

    SemanticSpec represents a natural language template that references
    input columns using `{column_name}` syntax. The special marker
    `{ALL_COLS}` references all columns from the input.

    The class parses templates to extract column dependencies, enabling
    the optimizer to reason about which columns are needed.

    Parameters
    ----------
    template : str
        The original template string with {col} references.
    input_columns : frozenset[str]
        Set of column names explicitly referenced in the template.
    all_columns : bool
        True if {ALL_COLS} was used, meaning all input columns are needed.

    Examples
    --------
    >>> # Single column reference
    >>> spec = SemanticSpec.parse("Is {title} a classic film?")
    >>> spec.input_columns
    frozenset({'title'})
    >>> spec.all_columns
    False

    >>> # Multiple column references
    >>> spec = SemanticSpec.parse("Based on {title} and {year}, classify the genre")
    >>> spec.input_columns
    frozenset({'title', 'year'})

    >>> # All columns reference
    >>> spec = SemanticSpec.parse("Summarize this record: {ALL_COLS}")
    >>> spec.all_columns
    True
    """

    template: str
    input_columns: frozenset[str]
    all_columns: bool

    # Regex pattern to match {column_name} references
    _COLUMN_PATTERN: re.Pattern[str] = re.compile(r"\{([^}]+)")

    @classmethod
    def parse(cls, template: str) -> SemanticSpec:
        """
        Parse a template string to extract column references.

        Extracts all `{column_name}` references from the template.
        If `{ALL_COLS}` is present, sets `all_columns=True`.

        Parameters
        ----------
        template : str
            The template string with {col} references.

        Returns
        -------
        SemanticSpec
            A parsed SemanticSpec with extracted column dependencies.

        Examples
        --------
        >>> spec = SemanticSpec.parse("Is {title} from {year} a good movie?")
        >>> sorted(spec.input_columns)
        ['title', 'year']

        >>> spec = SemanticSpec.parse("Analyze: {ALL_COLS}")
        >>> spec.all_columns
        True
        """
        matches = cls._COLUMN_PATTERN.findall(template)

        all_columns = ALL_COLUMNS_MARKER in matches
        input_columns = frozenset(m for m in matches if m != ALL_COLUMNS_MARKER)

        return cls(
            template=template,
            input_columns=input_columns,
            all_columns=all_columns,
        )

    @classmethod
    def create(
        cls,
        template: str,
        input_columns: frozenset[str] | set[str] | list[str] | None = None,
        all_columns: bool = False,
    ) -> SemanticSpec:
        """
        Create a SemanticSpec with explicit column dependencies.

        Use this when you want to specify column dependencies explicitly
        rather than parsing them from the template.

        Parameters
        ----------
        template : str
            The template string.
        input_columns : frozenset[str] | set[str] | list[str] | None
            Explicit set of input column names. If None, parsed from template.
        all_columns : bool
            Whether all input columns are needed.

        Returns
        -------
        SemanticSpec
            A SemanticSpec with the specified dependencies.

        Examples
        --------
        >>> spec = SemanticSpec.create(
        ...     "Classify the genre",
        ...     input_columns=["title", "synopsis"],
        ...     all_columns=False
        ... )
        """
        if input_columns is None:
            # Parse from template
            return cls.parse(template)

        if isinstance(input_columns, (set, list)):
            input_columns = frozenset(input_columns)

        return cls(
            template=template,
            input_columns=input_columns,
            all_columns=all_columns,
        )

    def format(self, row: dict[str, str]) -> str:
        """
        Format the template with actual values from a row.

        Replaces `{column_name}` placeholders with values from the row dict.
        If `all_columns` is True, `{ALL_COLS}` is replaced with a formatted
        representation of all columns.

        Parameters
        ----------
        row : dict[str, str]
            Mapping of column names to their string values.

        Returns
        -------
        str
            The formatted template with placeholders replaced.

        Examples
        --------
        >>> spec = SemanticSpec.parse("Is {title} from {year} good?")
        >>> spec.format({"title": "The Matrix", "year": "1999"})
        'Is The Matrix from 1999 good?'
        """
        result = self.template

        # Replace individual column references
        for col in self.input_columns:
            if col in row:
                result = result.replace(f"{{{col}}}", str(row[col]))

        # Replace ALL_COLS if present
        if self.all_columns and ALL_COLUMNS_MARKER in result:
            all_cols_str = ", ".join(f"{k}: {v}" for k, v in row.items())
            result = result.replace(f"{{{ALL_COLUMNS_MARKER}}}", all_cols_str)

        return result

    def get_required_columns(self, available_columns: list[str] | None = None) -> frozenset[str]:
        """
        Get the set of columns required by this spec.

        If `all_columns` is True and `available_columns` is provided,
        returns all available columns. Otherwise returns `input_columns`.

        Parameters
        ----------
        available_columns : list[str] | None
            List of available columns (needed when all_columns=True).

        Returns
        -------
        frozenset[str]
            Set of required column names.

        Raises
        ------
        ValueError
            If all_columns=True but available_columns is not provided.
        """
        if self.all_columns:
            if available_columns is None:
                raise ValueError(
                    "available_columns must be provided when all_columns=True"
                )
            return frozenset(available_columns)
        return self.input_columns

    def validate_columns(self, available_columns: set[str] | frozenset[str]) -> list[str]:
        """
        Validate that all referenced columns are available.

        Parameters
        ----------
        available_columns : set[str] | frozenset[str]
            Set of available column names.

        Returns
        -------
        list[str]
            List of missing column names (empty if all present).
        """
        if self.all_columns:
            return []  # ALL_COLS is always valid
        missing = self.input_columns - available_columns
        return sorted(missing)

    def to_string(self) -> str:
        """
        Get a human-readable string representation.

        Returns
        -------
        str
            String representation showing template and dependencies.
        """
        if self.all_columns:
            return f"SemanticSpec({self.template!r}, all_columns=True)"
        return f"SemanticSpec({self.template!r}, columns={sorted(self.input_columns)})"

    def __str__(self) -> str:
        return self.to_string()
