"""
Physical properties for physical IR nodes.

Physical properties describe characteristics of a physical operator's output
that are relevant for optimization and execution:
- Ordering: Whether output is sorted and by which columns
- Partitioning: How data is distributed (for future parallel execution)

These properties enable the optimizer to make informed decisions about
operator placement (e.g., avoiding unnecessary sorts for merge join).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class SortOrder(Enum):
    """
    Sort direction for ordered data.
    """

    ASC = auto()
    DESC = auto()
    ANY = auto()  # Either direction is acceptable (for optimization flexibility)


@dataclass(frozen=True)
class Ordering:
    """
    Describes the sort order of a relation.

    Parameters
    ----------
    columns : tuple[tuple[str, SortOrder], ...]
        Ordered list of (column_name, direction) pairs.
        Earlier columns are more significant (leftmost = primary sort key).
        Empty tuple means unordered.

    Examples
    --------
    >>> # Sorted by year DESC, then title ASC
    >>> ordering = Ordering(columns=(("year", SortOrder.DESC), ("title", SortOrder.ASC)))

    >>> # Unordered
    >>> ordering = Ordering.unordered()
    """

    columns: tuple[tuple[str, SortOrder], ...] = field(default_factory=tuple)

    @classmethod
    def unordered(cls) -> Ordering:
        """
        Create an unordered property.

        Returns
        -------
        Ordering
            Ordering with no sort columns.
        """
        return cls(columns=())

    @classmethod
    def by(cls, *columns: tuple[str, SortOrder] | str) -> Ordering:
        """
        Create an ordering from column specifications.

        Parameters
        ----------
        *columns : tuple[str, SortOrder] | str
            Column specifications. Strings default to ASC order.

        Returns
        -------
        Ordering
            The specified ordering.

        Examples
        --------
        >>> Ordering.by("year", ("title", SortOrder.DESC))
        Ordering(columns=(('year', SortOrder.ASC), ('title', SortOrder.DESC)))
        """
        normalized: list[tuple[str, SortOrder]] = []
        for col in columns:
            if isinstance(col, str):
                normalized.append((col, SortOrder.ASC))
            else:
                normalized.append(col)
        return cls(columns=tuple(normalized))

    def is_ordered(self) -> bool:
        """
        Check if this represents any ordering.

        Returns
        -------
        bool
            True if there is at least one sort column.
        """
        return len(self.columns) > 0

    def satisfies(self, required: Ordering) -> bool:
        """
        Check if this ordering satisfies a required ordering.

        An ordering satisfies a requirement if it has at least the required
        columns in the same order with compatible directions.

        Parameters
        ----------
        required : Ordering
            The required ordering.

        Returns
        -------
        bool
            True if this ordering satisfies the requirement.

        Examples
        --------
        >>> actual = Ordering.by(("a", SortOrder.ASC), ("b", SortOrder.DESC))
        >>> actual.satisfies(Ordering.by("a"))  # True - prefix match
        >>> actual.satisfies(Ordering.by(("a", SortOrder.DESC)))  # False - wrong direction
        """
        if len(required.columns) == 0:
            return True
        if len(self.columns) < len(required.columns):
            return False

        for (req_col, req_order), (act_col, act_order) in zip(
            required.columns, self.columns
        ):
            if req_col != act_col:
                return False
            if req_order != SortOrder.ANY and act_order != req_order:
                return False

        return True

    def prefix(self, n: int) -> Ordering:
        """
        Get the first n columns of this ordering.

        Parameters
        ----------
        n : int
            Number of columns to keep.

        Returns
        -------
        Ordering
            Ordering with only the first n columns.
        """
        return Ordering(columns=self.columns[:n])


@dataclass(frozen=True)
class Partitioning:
    """
    Describes how data is partitioned (for parallel/distributed execution).

    Parameters
    ----------
    columns : tuple[str, ...] | None
        Columns used for partitioning. None means single partition (not distributed).
        Empty tuple means random/round-robin partitioning.
    num_partitions : int | None
        Number of partitions if known, None otherwise.

    Notes
    -----
    This is a placeholder for future distributed execution support.
    Currently, SoliceDB assumes single-node execution.
    """

    columns: tuple[str, ...] | None = None
    num_partitions: int | None = None

    @classmethod
    def single(cls) -> Partitioning:
        """
        Create a single-partition (non-distributed) property.

        Returns
        -------
        Partitioning
            Single partition property.
        """
        return cls(columns=None, num_partitions=1)

    @classmethod
    def hash_partitioned(cls, *columns: str, num_partitions: int | None = None) -> Partitioning:
        """
        Create a hash-partitioned property.

        Parameters
        ----------
        *columns : str
            Columns to partition by.
        num_partitions : int | None
            Number of partitions if known.

        Returns
        -------
        Partitioning
            Hash partitioning property.
        """
        return cls(columns=tuple(columns), num_partitions=num_partitions)

    def is_partitioned(self) -> bool:
        """
        Check if data is partitioned across multiple partitions.

        Returns
        -------
        bool
            True if partitioned (more than one partition).
        """
        return self.num_partitions is not None and self.num_partitions > 1


@dataclass(frozen=True)
class PhysicalProperties:
    """
    Combined physical properties of a relation.

    Parameters
    ----------
    ordering : Ordering
        Sort order of the data.
    partitioning : Partitioning
        Partitioning scheme of the data.
    row_count_estimate : int | None
        Estimated number of rows (cardinality).

    Examples
    --------
    >>> props = PhysicalProperties(
    ...     ordering=Ordering.by("id"),
    ...     partitioning=Partitioning.single(),
    ...     row_count_estimate=1000
    ... )
    """

    ordering: Ordering = field(default_factory=Ordering.unordered)
    partitioning: Partitioning = field(default_factory=Partitioning.single)
    row_count_estimate: int | None = None

    @classmethod
    def default(cls) -> PhysicalProperties:
        """
        Create default physical properties (unordered, single partition).

        Returns
        -------
        PhysicalProperties
            Default properties.
        """
        return cls()

    def with_ordering(self, ordering: Ordering) -> PhysicalProperties:
        """
        Create a copy with different ordering.

        Parameters
        ----------
        ordering : Ordering
            New ordering.

        Returns
        -------
        PhysicalProperties
            New properties with updated ordering.
        """
        return PhysicalProperties(
            ordering=ordering,
            partitioning=self.partitioning,
            row_count_estimate=self.row_count_estimate,
        )

    def with_row_count(self, row_count: int | None) -> PhysicalProperties:
        """
        Create a copy with different row count estimate.

        Parameters
        ----------
        row_count : int | None
            New row count estimate.

        Returns
        -------
        PhysicalProperties
            New properties with updated row count.
        """
        return PhysicalProperties(
            ordering=self.ordering,
            partitioning=self.partitioning,
            row_count_estimate=row_count,
        )