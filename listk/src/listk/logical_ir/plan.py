"""
Query plan container for the SoliceDB logical IR.

This module defines the QueryPlan class that manages the DAG of logical nodes.
It provides:
- Node registration with auto-incremented IDs
- Builder methods for easy query construction
- DAG traversal and validation
- JSON serialization/deserialization
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from solicedb.logical_ir.expressions import Aggregation, Expr
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
from solicedb.logical_ir.types import DType, Schema


@dataclass
class QueryPlan:
    """
    Container for a logical query plan DAG.

    QueryPlan manages the collection of logical nodes, assigns unique IDs,
    and provides both low-level node registration and high-level builder
    methods for query construction.

    Parameters
    ----------
    global_hints : GlobalHints
        Global execution hints for the entire plan.

    Examples
    --------
    >>> # Low-level API: register nodes directly
    >>> plan = QueryPlan()
    >>> source_id = plan.add(Source(node_id=-1, input_ids=(), ref="movies"))
    >>> filter_id = plan.add(Select(node_id=-1, input_ids=(source_id,), predicate=...))
    >>> plan.set_root(filter_id)

    >>> # High-level API: use builder methods
    >>> plan = QueryPlan()
    >>> source_id = plan.source("movies")
    >>> filter_id = plan.select(source_id, predicate=...)
    >>> plan.set_root(filter_id)
    """

    _nodes: dict[NodeId, LogicalNode] = field(default_factory=dict)
    _next_id: NodeId = field(default=0)
    _root_id: NodeId | None = field(default=None)
    global_hints: GlobalHints = field(default_factory=GlobalHints)

    # ========================================================================
    # Core Node Management
    # ========================================================================

    def add(self, node: LogicalNode) -> NodeId:
        """
        Add a node to the plan with an auto-assigned ID.

        The node's `node_id` field is ignored; a new unique ID is assigned.
        The node is recreated with the correct ID.

        Parameters
        ----------
        node : LogicalNode
            The node to add. Its node_id will be replaced.

        Returns
        -------
        NodeId
            The assigned node ID.

        Raises
        ------
        ValueError
            If any input_id references a non-existent node.
        """
        # Validate input references
        for input_id in node.input_ids:
            if input_id not in self._nodes:
                raise ValueError(f"Input node {input_id} does not exist")

        # Assign new ID
        new_id = self._alloc_id()

        # Recreate node with correct ID (nodes are frozen, so we rebuild)
        new_node = self._rebuild_node_with_id(node, new_id)
        self._nodes[new_id] = new_node

        return new_id

    def _alloc_id(self) -> NodeId:
        """
        Allocate the next available node ID.

        Returns
        -------
        NodeId
            A unique node ID.
        """
        node_id = self._next_id
        self._next_id += 1
        return node_id

    @staticmethod
    def _rebuild_node_with_id(node: LogicalNode, new_id: NodeId) -> LogicalNode:
        """
        Rebuild a node with a new ID.

        Parameters
        ----------
        node : LogicalNode
            The original node.
        new_id : NodeId
            The new ID to assign.

        Returns
        -------
        LogicalNode
            A new node instance with the updated ID.
        """
        # Get all fields except node_id
        node_dict = {
            k: v for k, v in node.__dict__.items() if k != "node_id"
        }
        node_dict["node_id"] = new_id

        # Reconstruct the node
        return type(node)(**node_dict)

    def get(self, node_id: NodeId) -> LogicalNode:
        """
        Get a node by its ID.

        Parameters
        ----------
        node_id : NodeId
            The node ID.

        Returns
        -------
        LogicalNode
            The node with the given ID.

        Raises
        ------
        KeyError
            If the node ID does not exist.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id} does not exist")
        return self._nodes[node_id]

    def set_root(self, node_id: NodeId) -> None:
        """
        Set the root node of the query plan.

        The root node is the final output of the query.

        Parameters
        ----------
        node_id : NodeId
            The ID of the root node.

        Raises
        ------
        KeyError
            If the node ID does not exist.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id} does not exist")
        self._root_id = node_id

    @property
    def root_id(self) -> NodeId | None:
        """
        Get the root node ID.

        Returns
        -------
        NodeId | None
            The root node ID, or None if not set.
        """
        return self._root_id

    @property
    def root(self) -> LogicalNode | None:
        """
        Get the root node.

        Returns
        -------
        LogicalNode | None
            The root node, or None if not set.
        """
        if self._root_id is None:
            return None
        return self._nodes[self._root_id]

    def __len__(self) -> int:
        """Return the number of nodes in the plan."""
        return len(self._nodes)

    def __contains__(self, node_id: NodeId) -> bool:
        """Check if a node ID exists in the plan."""
        return node_id in self._nodes

    def nodes(self) -> list[LogicalNode]:
        """
        Get all nodes in the plan.

        Returns
        -------
        list[LogicalNode]
            List of all nodes in ID order.
        """
        return [self._nodes[i] for i in sorted(self._nodes.keys())]

    # ========================================================================
    # DAG Traversal
    # ========================================================================

    def children(self, node_id: NodeId) -> list[NodeId]:
        """
        Get the input (child) node IDs of a node.

        Parameters
        ----------
        node_id : NodeId
            The node ID.

        Returns
        -------
        list[NodeId]
            List of input node IDs.
        """
        return list(self.get(node_id).input_ids)

    def parents(self, node_id: NodeId) -> list[NodeId]:
        """
        Get the nodes that consume (depend on) this node.

        Parameters
        ----------
        node_id : NodeId
            The node ID.

        Returns
        -------
        list[NodeId]
            List of node IDs that have this node as input.
        """
        return [
            nid for nid, node in self._nodes.items()
            if node_id in node.input_ids
        ]

    def topological_order(self) -> list[NodeId]:
        """
        Get nodes in topological order (inputs before outputs).

        Returns
        -------
        list[NodeId]
            Node IDs in topological order.

        Raises
        ------
        ValueError
            If the graph contains a cycle.
        """
        visited: set[NodeId] = set()
        result: list[NodeId] = []
        temp_marks: set[NodeId] = set()

        def visit(node_id_: NodeId) -> None:
            if node_id_ in temp_marks:
                raise ValueError(f"Cycle detected involving node {node_id_}")
            if node_id_ in visited:
                return

            temp_marks.add(node_id_)
            for child_id in self.children(node_id_):
                visit(child_id)
            temp_marks.remove(node_id_)
            visited.add(node_id_)
            result.append(node_id_)

        for node_id in self._nodes:
            if node_id not in visited:
                visit(node_id)

        return result

    def reverse_topological_order(self) -> list[NodeId]:
        """
        Get nodes in reverse topological order (outputs before inputs).

        Returns
        -------
        list[NodeId]
            Node IDs in reverse topological order.
        """
        return list(reversed(self.topological_order()))

    # ========================================================================
    # Validation
    # ========================================================================

    def validate(self) -> list[str]:
        """
        Validate the query plan.

        Checks for:
        - Root node is set
        - No dangling references
        - No cycles
        - All nodes are reachable from root

        Returns
        -------
        list[str]
            List of validation error messages. Empty if valid.
        """
        errors: list[str] = []

        # Check root is set
        if self._root_id is None:
            errors.append("Root node is not set")
            return errors

        # Check for cycles (topological_order raises on cycle)
        try:
            self.topological_order()
        except ValueError as e:
            errors.append(str(e))
            return errors

        # Check all nodes are reachable from root
        reachable: set[NodeId] = set()

        def mark_reachable(node_id: NodeId) -> None:
            if node_id in reachable:
                return
            reachable.add(node_id)
            for child_id in self.children(node_id):
                mark_reachable(child_id)

        mark_reachable(self._root_id)

        unreachable = set(self._nodes.keys()) - reachable
        if unreachable:
            errors.append(f"Unreachable nodes: {sorted(unreachable)}")

        return errors

    def is_valid(self) -> bool:
        """
        Check if the query plan is valid.

        Returns
        -------
        bool
            True if the plan has no validation errors.
        """
        return len(self.validate()) == 0

    # ========================================================================
    # Builder Methods (High-Level API)
    # ========================================================================

    def source(
        self,
        ref: str,
        schema: Schema | dict[str, DType | str] | None = None,
    ) -> NodeId:
        """
        Add a source node.

        Parameters
        ----------
        ref : str
            Reference name to resolve the DataFrame.
        schema : Schema | dict[str, DType | str] | None
            Optional schema. Can be a Schema object or a dict.

        Returns
        -------
        NodeId
            The source node ID.
        """
        if isinstance(schema, dict):
            schema = Schema.from_dict(schema)

        return self.add(Source(
            node_id=-1,
            input_ids=(),
            ref=ref,
            schema=schema,
        ))

    def select(self, input_id: NodeId, predicate: Expr) -> NodeId:
        """
        Add a classical Select (filter) node.

        Parameters
        ----------
        input_id : NodeId
            Input node ID.
        predicate : Expr
            Boolean filter expression.

        Returns
        -------
        NodeId
            The select node ID.
        """
        return self.add(Select(
            node_id=-1,
            input_ids=(input_id,),
            predicate=predicate,
        ))

    def project(
        self,
        input_id: NodeId,
        columns: tuple[str | tuple[str, Expr], ...] | list[str | tuple[str, Expr]],
    ) -> NodeId:
        """
        Add a classical Project node.

        Parameters
        ----------
        input_id : NodeId
            Input node ID.
        columns : tuple or list
            Columns to project. Each element is a string (pass-through)
            or tuple (new_name, expression).

        Returns
        -------
        NodeId
            The project node ID.
        """
        if isinstance(columns, list):
            columns = tuple(columns)

        return self.add(Project(
            node_id=-1,
            input_ids=(input_id,),
            columns=columns,
        ))

    def join(
        self,
        left_id: NodeId,
        right_id: NodeId,
        join_type: JoinType,
        condition: Expr | None = None,
    ) -> NodeId:
        """
        Add a classical Join node.

        Parameters
        ----------
        left_id : NodeId
            Left input node ID.
        right_id : NodeId
            Right input node ID.
        join_type : JoinType
            Type of join.
        condition : Expr | None
            Join condition (None for CROSS join).

        Returns
        -------
        NodeId
            The join node ID.
        """
        return self.add(Join(
            node_id=-1,
            input_ids=(left_id, right_id),
            join_type=join_type,
            condition=condition,
        ))

    def group_by(
        self,
        input_id: NodeId,
        group_columns: tuple[str, ...] | list[str],
        aggregations: tuple[tuple[str, Aggregation], ...] | list[tuple[str, Aggregation]],
    ) -> NodeId:
        """
        Add a classical GroupBy node.

        Parameters
        ----------
        input_id : NodeId
            Input node ID.
        group_columns : tuple or list
            Columns to group by.
        aggregations : tuple or list
            Aggregations as (output_name, Aggregation) pairs.

        Returns
        -------
        NodeId
            The group by node ID.
        """
        if isinstance(group_columns, list):
            group_columns = tuple(group_columns)
        if isinstance(aggregations, list):
            aggregations = tuple(aggregations)

        return self.add(GroupBy(
            node_id=-1,
            input_ids=(input_id,),
            group_columns=group_columns,
            aggregations=aggregations,
        ))

    def order_by(
        self,
        input_id: NodeId,
        keys: tuple[tuple[str, SortDirection], ...] | list[tuple[str, SortDirection]],
        limit: int | None = None,
    ) -> NodeId:
        """
        Add a classical OrderBy node.

        Parameters
        ----------
        input_id : NodeId
            Input node ID.
        keys : tuple or list
            Sort keys as (column_name, direction) pairs.
        limit : int | None
            Optional row limit.

        Returns
        -------
        NodeId
            The order by node ID.
        """
        if isinstance(keys, list):
            keys = tuple(keys)

        return self.add(OrderBy(
            node_id=-1,
            input_ids=(input_id,),
            keys=keys,
            limit=limit,
        ))

    def distinct(
        self,
        input_id: NodeId,
        columns: tuple[str, ...] | list[str] | None = None,
    ) -> NodeId:
        """
        Add a Distinct node.

        Parameters
        ----------
        input_id : NodeId
            Input node ID.
        columns : tuple, list, or None
            Columns for uniqueness, or None for all columns.

        Returns
        -------
        NodeId
            The distinct node ID.
        """
        if isinstance(columns, list):
            columns = tuple(columns)

        return self.add(Distinct(
            node_id=-1,
            input_ids=(input_id,),
            columns=columns,
        ))

    def semantic_filter(
        self,
        input_id: NodeId,
        template: str,
        hints: SemanticFilterHints | None = None,
    ) -> NodeId:
        """
        Add a SemanticFilter node.

        Parameters
        ----------
        input_id : NodeId
            Input node ID.
        template : str
            Natural language template with {col} references.
        hints : SemanticFilterHints | None
            Optional execution hints.

        Returns
        -------
        NodeId
            The semantic filter node ID.
        """
        return self.add(SemanticFilter(
            node_id=-1,
            input_ids=(input_id,),
            spec=SemanticSpec.parse(template),
            hints=hints,
        ))

    def semantic_project(
        self,
        input_id: NodeId,
        template: str,
        output_schema: Schema | dict[str, DType | str],
        hints: SemanticProjectHints | None = None,
    ) -> NodeId:
        """
        Add a SemanticProject node.

        Parameters
        ----------
        input_id : NodeId
            Input node ID.
        template : str
            Natural language template with {col} references.
        output_schema : Schema | dict
            Schema of output columns.
        hints : SemanticProjectHints | None
            Optional execution hints.

        Returns
        -------
        NodeId
            The semantic project node ID.
        """
        if isinstance(output_schema, dict):
            output_schema = Schema.from_dict(output_schema)

        return self.add(SemanticProject(
            node_id=-1,
            input_ids=(input_id,),
            spec=SemanticSpec.parse(template),
            output_schema=output_schema,
            hints=hints,
        ))

    def semantic_join(
        self,
        left_id: NodeId,
        right_id: NodeId,
        template: str,
        join_type: JoinType = JoinType.INNER,
        output_schema: Schema | dict[str, DType | str] | None = None,
        hints: SemanticJoinHints | None = None,
    ) -> NodeId:
        """
        Add a SemanticJoin node.

        Parameters
        ----------
        left_id : NodeId
            Left input node ID.
        right_id : NodeId
            Right input node ID.
        template : str
            Natural language template for join condition.
        join_type : JoinType
            Type of join.
        output_schema : Schema | dict | None
            Optional additional output columns.
        hints : SemanticJoinHints | None
            Optional execution hints.

        Returns
        -------
        NodeId
            The semantic join node ID.
        """
        if isinstance(output_schema, dict):
            output_schema = Schema.from_dict(output_schema)

        return self.add(SemanticJoin(
            node_id=-1,
            input_ids=(left_id, right_id),
            spec=SemanticSpec.parse(template),
            join_type=join_type,
            output_schema=output_schema,
            hints=hints,
        ))

    def semantic_group_by(
        self,
        input_id: NodeId,
        template: str,
        output_group_column: str,
        hints: SemanticGroupByHints | None = None,
    ) -> NodeId:
        """
        Add a SemanticGroupBy node.

        Parameters
        ----------
        input_id : NodeId
            Input node ID.
        template : str
            Natural language template for grouping criterion.
        output_group_column : str
            Name of the output column for group assignments.
        hints : SemanticGroupByHints | None
            Optional execution hints.

        Returns
        -------
        NodeId
            The semantic group by node ID.
        """
        return self.add(SemanticGroupBy(
            node_id=-1,
            input_ids=(input_id,),
            spec=SemanticSpec.parse(template),
            output_group_column=output_group_column,
            hints=hints,
        ))

    def semantic_summarize(
        self,
        input_id: NodeId,
        group_columns: tuple[str, ...] | list[str],
        template: str,
        output_schema: Schema | dict[str, DType | str],
        hints: SemanticSummarizeHints | None = None,
    ) -> NodeId:
        """
        Add a SemanticSummarize node.

        Parameters
        ----------
        input_id : NodeId
            Input node ID.
        group_columns : tuple or list
            Columns to group by.
        template : str
            Natural language template for summarization.
        output_schema : Schema | dict
            Schema of summary output columns.
        hints : SemanticSummarizeHints | None
            Optional execution hints.

        Returns
        -------
        NodeId
            The semantic summarize node ID.
        """
        if isinstance(group_columns, list):
            group_columns = tuple(group_columns)
        if isinstance(output_schema, dict):
            output_schema = Schema.from_dict(output_schema)

        return self.add(SemanticSummarize(
            node_id=-1,
            input_ids=(input_id,),
            group_columns=group_columns,
            spec=SemanticSpec.parse(template),
            output_schema=output_schema,
            hints=hints,
        ))

    def semantic_top_k(
        self,
        input_id: NodeId,
        template: str,
        k: int,
        hints: SemanticTopKHints | None = None,
    ) -> NodeId:
        """
        Add a SemanticTopK node.

        Parameters
        ----------
        input_id : NodeId
            Input node ID.
        template : str
            Natural language template for ordering criterion.
        k : int
            Number of top rows to return.
        hints : SemanticTopKHints | None
            Optional execution hints.

        Returns
        -------
        NodeId
            The semantic top-k node ID.
        """
        return self.add(SemanticTopK(
            node_id=-1,
            input_ids=(input_id,),
            spec=SemanticSpec.parse(template),
            k=k,
            hints=hints,
        ))

    def semantic_fill(
        self,
        input_id: NodeId,
        target_column: str,
        template: str,
        output_dtype: DType,
        hints: SemanticFillHints | None = None,
    ) -> NodeId:
        """
        Add a SemanticFill node.

        Parameters
        ----------
        input_id : NodeId
            Input node ID.
        target_column : str
            Column to fill (existing or new).
        template : str
            Natural language template for fill logic.
        output_dtype : DType
            Data type of the filled column.
        hints : SemanticFillHints | None
            Optional execution hints.

        Returns
        -------
        NodeId
            The semantic fill node ID.
        """
        return self.add(SemanticFill(
            node_id=-1,
            input_ids=(input_id,),
            target_column=target_column,
            spec=SemanticSpec.parse(template),
            output_dtype=output_dtype,
            hints=hints,
        ))

    # ========================================================================
    # Serialization
    # ========================================================================

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the query plan to a dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the plan.
        """
        return {
            "nodes": {
                str(node_id): self._node_to_dict(node)
                for node_id, node in self._nodes.items()
            },
            "root_id": self._root_id,
            "global_hints": self._hints_to_dict(self.global_hints),
        }

    def _node_to_dict(self, node: LogicalNode) -> dict[str, Any]:
        """
        Convert a node to a dictionary.

        Parameters
        ----------
        node : LogicalNode
            The node to convert.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the node.
        """
        result: dict[str, Any] = {
            "type": type(node).__name__,
            "node_id": node.node_id,
            "input_ids": list(node.input_ids),
        }

        # Add type-specific fields
        for key, value in node.__dict__.items():
            if key in ("node_id", "input_ids"):
                continue
            result[key] = self._serialize_value(value)

        return result

    def _serialize_value(self, value: Any) -> Any:
        """
        Serialize a value for JSON output.

        Parameters
        ----------
        value : Any
            The value to serialize.

        Returns
        -------
        Any
            JSON-serializable representation.
        """
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if isinstance(value, DType):
            return {"__type__": "DType", "value": value.value}
        if isinstance(value, (JoinType, SortDirection)):
            return {"__type__": type(value).__name__, "value": value.name}
        if isinstance(value, Schema):
            return {
                "__type__": "Schema",
                "columns": [(name, dtype.value) for name, dtype in value.columns],
            }
        if isinstance(value, SemanticSpec):
            return {
                "__type__": "SemanticSpec",
                "template": value.template,
                "input_columns": sorted(value.input_columns),
                "all_columns": value.all_columns,
            }
        if isinstance(value, Expr):
            return {"__type__": "Expr", "repr": repr(value)}
        if isinstance(value, Aggregation):
            return {
                "__type__": "Aggregation",
                "func": value.func.name,
                "column": value.column,
                "distinct": value.distinct,
            }
        if hasattr(value, "__dataclass_fields__"):
            # Generic dataclass serialization (for hints)
            return {
                "__type__": type(value).__name__,
                **{k: self._serialize_value(v) for k, v in value.__dict__.items()},
            }

        return str(value)

    @staticmethod
    def _hints_to_dict(hints: GlobalHints) -> dict[str, Any]:
        """
        Convert global hints to a dictionary.

        Parameters
        ----------
        hints : GlobalHints
            The hints to convert.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "prefer_listwise": hints.prefer_listwise,
            "max_batch_size": hints.max_batch_size,
            "default_model": hints.default_model,
            "parallelism": hints.parallelism,
            "enable_caching": hints.enable_caching,
            "debug_mode": hints.debug_mode,
            "extra": hints.extra,
        }

    def to_json(self, indent: int | None = 2) -> str:
        """
        Serialize the query plan to JSON.

        Parameters
        ----------
        indent : int | None
            JSON indentation level. None for compact output.

        Returns
        -------
        str
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def __repr__(self) -> str:
        """Return a string representation of the plan."""
        node_count = len(self._nodes)
        root_info = f", root={self._root_id}" if self._root_id is not None else ""
        return f"QueryPlan(nodes={node_count}{root_info})"
