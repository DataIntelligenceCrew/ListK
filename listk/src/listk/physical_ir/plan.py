"""
Physical plan container for the SoliceDB physical IR.

This module defines the PhysicalPlan class that manages the DAG of physical nodes.
Similar to QueryPlan for logical IR, but with additional support for:
- Visualization (DOT export, ASCII tree)
- Plan comparison (diff for debugging optimizations)
- Subgraph extraction and manipulation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from solicedb.physical_ir.annotations import CostEstimate, Provenance
from solicedb.physical_ir.nodes import PhysicalNode, PhysicalNodeId
from solicedb.physical_ir.properties import PhysicalProperties


@dataclass
class PhysicalPlan:
    """
    Container for a physical query plan DAG.

    PhysicalPlan manages the collection of physical nodes, assigns unique IDs,
    and provides traversal, visualization, and manipulation utilities.

    The plan itself is mutable (nodes can be added), but individual nodes
    are immutable (frozen dataclasses).

    Parameters
    ----------
    name : str | None
        Optional name for the plan (useful for debugging).
    metadata : dict[str, Any]
        Arbitrary metadata (e.g., optimization history).

    Examples
    --------
    >>> plan = PhysicalPlan(name="optimized_query_1")
    >>> scan_id = plan.add(TableScan(node_id=-1, input_ids=(), ref="movies"))
    >>> filter_id = plan.add(Filter(node_id=-1, input_ids=(scan_id,), predicate=...))
    >>> plan.set_root(filter_id)
    """

    _nodes: dict[PhysicalNodeId, PhysicalNode] = field(default_factory=dict)
    _next_id: PhysicalNodeId = field(default=0)
    _root_id: PhysicalNodeId | None = field(default=None)
    name: str | None = field(default=None)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ========================================================================
    # Core Node Management
    # ========================================================================

    def add(self, node: PhysicalNode) -> PhysicalNodeId:
        """
        Add a node to the plan with an auto-assigned ID.

        The node's `node_id` field is ignored; a new unique ID is assigned.

        Parameters
        ----------
        node : PhysicalNode
            The node to add. Its node_id will be replaced.

        Returns
        -------
        PhysicalNodeId
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

        # Rebuild node with correct ID
        new_node = self._rebuild_node_with_id(node, new_id)
        self._nodes[new_id] = new_node

        return new_id

    def _alloc_id(self) -> PhysicalNodeId:
        """Allocate the next available node ID."""
        node_id = self._next_id
        self._next_id += 1
        return node_id

    @staticmethod
    def _rebuild_node_with_id(node: PhysicalNode, new_id: PhysicalNodeId) -> PhysicalNode:
        """Rebuild a node with a new ID."""
        node_dict = {k: v for k, v in node.__dict__.items() if k != "node_id"}
        node_dict["node_id"] = new_id
        return type(node)(**node_dict)

    def get(self, node_id: PhysicalNodeId) -> PhysicalNode:
        """
        Get a node by its ID.

        Parameters
        ----------
        node_id : PhysicalNodeId
            The node ID.

        Returns
        -------
        PhysicalNode
            The node with the given ID.

        Raises
        ------
        KeyError
            If the node ID does not exist.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id} does not exist")
        return self._nodes[node_id]

    def set_root(self, node_id: PhysicalNodeId) -> None:
        """
        Set the root node of the physical plan.

        Parameters
        ----------
        node_id : PhysicalNodeId
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
    def root_id(self) -> PhysicalNodeId | None:
        """Get the root node ID."""
        return self._root_id

    @property
    def root(self) -> PhysicalNode | None:
        """Get the root node."""
        if self._root_id is None:
            return None
        return self._nodes[self._root_id]

    def __len__(self) -> int:
        """Return the number of nodes in the plan."""
        return len(self._nodes)

    def __contains__(self, node_id: PhysicalNodeId) -> bool:
        """Check if a node ID exists in the plan."""
        return node_id in self._nodes

    def __iter__(self) -> Iterator[PhysicalNode]:
        """Iterate over nodes in ID order."""
        for node_id in sorted(self._nodes.keys()):
            yield self._nodes[node_id]

    def nodes(self) -> list[PhysicalNode]:
        """Get all nodes in ID order."""
        return [self._nodes[i] for i in sorted(self._nodes.keys())]

    def node_ids(self) -> list[PhysicalNodeId]:
        """Get all node IDs in order."""
        return sorted(self._nodes.keys())

    # ========================================================================
    # DAG Traversal
    # ========================================================================

    def children(self, node_id: PhysicalNodeId) -> list[PhysicalNodeId]:
        """Get the input (child) node IDs of a node."""
        return list(self.get(node_id).input_ids)

    def parents(self, node_id: PhysicalNodeId) -> list[PhysicalNodeId]:
        """Get the nodes that consume (depend on) this node."""
        return [
            nid for nid, node in self._nodes.items()
            if node_id in node.input_ids
        ]

    def topological_order(self) -> list[PhysicalNodeId]:
        """
        Get nodes in topological order (inputs before outputs).

        Returns
        -------
        list[PhysicalNodeId]
            Node IDs in topological order (sources first).

        Raises
        ------
        ValueError
            If the graph contains a cycle.
        """
        visited: set[PhysicalNodeId] = set()
        result: list[PhysicalNodeId] = []
        temp_marks: set[PhysicalNodeId] = set()

        def visit(node_id: PhysicalNodeId) -> None:
            if node_id in temp_marks:
                raise ValueError(f"Cycle detected involving node {node_id}")
            if node_id in visited:
                return

            temp_marks.add(node_id)
            for child_id in self.children(node_id):
                visit(child_id)
            temp_marks.remove(node_id)
            visited.add(node_id)
            result.append(node_id)

        for node_id in self._nodes:
            if node_id not in visited:
                visit(node_id)

        return result

    def reverse_topological_order(self) -> list[PhysicalNodeId]:
        """Get nodes in reverse topological order (outputs before inputs)."""
        return list(reversed(self.topological_order()))

    def sources(self) -> list[PhysicalNodeId]:
        """Get all source node IDs (nodes with no inputs)."""
        return [nid for nid, node in self._nodes.items() if node.is_source()]

    # ========================================================================
    # Cost Aggregation
    # ========================================================================

    def total_cost(self) -> CostEstimate:
        """
        Compute total estimated cost of the plan.

        Returns
        -------
        CostEstimate
            Sum of all node costs.
        """
        total = CostEstimate.zero()
        for node in self._nodes.values():
            if node.cost is not None:
                total = total + node.cost
        return total

    # ========================================================================
    # Validation
    # ========================================================================

    def validate(self) -> list[str]:
        """
        Validate the physical plan.

        Returns
        -------
        list[str]
            List of validation error messages. Empty if valid.
        """
        errors: list[str] = []

        if self._root_id is None:
            errors.append("Root node is not set")
            return errors

        # Check for cycles
        try:
            self.topological_order()
        except ValueError as e:
            errors.append(str(e))
            return errors

        # Check all nodes are reachable from root
        reachable: set[PhysicalNodeId] = set()

        def mark_reachable(node_id: PhysicalNodeId) -> None:
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
        """Check if the physical plan is valid."""
        return len(self.validate()) == 0

    # ========================================================================
    # Visualization
    # ========================================================================

    def to_tree_str(self, show_cost: bool = True, show_properties: bool = False) -> str:
        """
        Pretty-print the plan as an ASCII tree.

        Parameters
        ----------
        show_cost : bool
            Whether to show cost estimates.
        show_properties : bool
            Whether to show physical properties.

        Returns
        -------
        str
            ASCII tree representation.
        """
        if self._root_id is None:
            return "(empty plan)"

        lines: list[str] = []
        self._format_subtree(self._root_id, "", True, lines, show_cost, show_properties)
        return "\n".join(lines)

    def _format_subtree(
        self,
        node_id: PhysicalNodeId,
        prefix: str,
        is_last: bool,
        lines: list[str],
        show_cost: bool,
        show_properties: bool,
    ) -> None:
        """Recursively format a subtree."""
        node = self.get(node_id)
        connector = "`-- " if is_last else "|-- "

        # Format node name and key attributes
        node_str = self._format_node(node, show_cost, show_properties)
        lines.append(f"{prefix}{connector}{node_str}")

        # Format children
        child_prefix = prefix + ("    " if is_last else "|   ")
        children = self.children(node_id)
        for i, child_id in enumerate(children):
            is_last_child = i == len(children) - 1
            self._format_subtree(child_id, child_prefix, is_last_child, lines, show_cost, show_properties)

    def _format_node(self, node: PhysicalNode, show_cost: bool, show_properties: bool) -> str:
        """Format a single node for display."""
        parts = [type(node).__name__]

        # Add key attributes based on node type
        attrs = self._get_display_attrs(node)
        if attrs:
            parts.append(f"({attrs})")

        # Add cost if requested
        if show_cost and node.cost is not None:
            cost_parts = []
            if node.cost.token_count is not None:
                cost_parts.append(f"{node.cost.token_count} tokens")
            if node.cost.llm_calls is not None:
                cost_parts.append(f"{node.cost.llm_calls} calls")
            if node.cost.row_count is not None:
                cost_parts.append(f"~{node.cost.row_count} rows")
            if cost_parts:
                parts.append(f"[{', '.join(cost_parts)}]")

        # Add properties if requested
        if show_properties and node.output_properties is not None:
            props = node.output_properties
            if props.ordering.is_ordered():
                cols = ", ".join(c for c, _ in props.ordering.columns)
                parts.append(f"{{ordered by {cols}}}")

        return " ".join(parts)

    def _get_display_attrs(self, node: PhysicalNode) -> str:
        """Get key attributes for display based on node type."""
        from solicedb.physical_ir.nodes import (
            TableScan, Filter, Project, HashJoin, SortMergeJoin,
            NestedLoopJoin, HashAggregate, Sort, Limit,
        )

        if isinstance(node, TableScan):
            return node.ref
        elif isinstance(node, Filter):
            return f"predicate={node.predicate!r}"[:50]
        elif isinstance(node, Project):
            cols = [c if isinstance(c, str) else c[0] for c in node.columns]
            return f"cols=[{', '.join(cols[:3])}{'...' if len(cols) > 3 else ''}]"
        elif isinstance(node, (HashJoin, SortMergeJoin)):
            return f"{node.join_type.name} on {node.left_keys}={node.right_keys}"
        elif isinstance(node, NestedLoopJoin):
            return node.join_type.name
        elif isinstance(node, HashAggregate):
            return f"by {node.group_columns}" if node.group_columns else "global"
        elif isinstance(node, Sort):
            cols = ", ".join(c for c, _ in node.ordering.columns)
            return f"by [{cols}]"
        elif isinstance(node, Limit):
            return f"n={node.count}"
        return ""

    def to_dot(self, show_cost: bool = True) -> str:
        """
        Export the plan to Graphviz DOT format.

        Parameters
        ----------
        show_cost : bool
            Whether to include cost estimates in labels.

        Returns
        -------
        str
            DOT format string.
        """
        lines = ["digraph PhysicalPlan {", "  rankdir=BT;", "  node [shape=box];"]

        for node_id, node in self._nodes.items():
            label = self._format_node(node, show_cost, show_properties=False)
            label = label.replace('"', '\\"')
            style = ""
            if node_id == self._root_id:
                style = ', style="bold"'
            lines.append(f'  n{node_id} [label="{label}"{style}];')

        for node_id, node in self._nodes.items():
            for input_id in node.input_ids:
                lines.append(f"  n{input_id} -> n{node_id};")

        lines.append("}")
        return "\n".join(lines)

    # ========================================================================
    # Serialization
    # ========================================================================

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the physical plan to a dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the plan.
        """
        return {
            "name": self.name,
            "metadata": self.metadata,
            "root_id": self._root_id,
            "nodes": {
                str(node_id): self._node_to_dict(node)
                for node_id, node in self._nodes.items()
            },
        }

    def _node_to_dict(self, node: PhysicalNode) -> dict[str, Any]:
        """Convert a node to a dictionary."""
        result: dict[str, Any] = {
            "type": type(node).__name__,
            "node_id": node.node_id,
            "input_ids": list(node.input_ids),
        }

        # Add type-specific fields (excluding base fields)
        base_fields = {"node_id", "input_ids", "cost", "output_properties", "provenance"}
        for key, value in node.__dict__.items():
            if key in base_fields:
                continue
            result[key] = self._serialize_value(value)

        # Add optional fields if present
        if node.cost is not None:
            result["cost"] = {
                "token_count": node.cost.token_count,
                "llm_calls": node.cost.llm_calls,
                "row_count": node.cost.row_count,
                "cpu_cost": node.cost.cpu_cost,
                "memory_bytes": node.cost.memory_bytes,
                "latency_ms": node.cost.latency_ms,
            }
        if node.provenance is not None:
            result["provenance"] = {
                "logical_node_ids": list(node.provenance.logical_node_ids),
                "rule_name": node.provenance.rule_name,
                "rule_metadata": node.provenance.rule_metadata,
            }

        return result

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for JSON output."""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if hasattr(value, "name"):  # Enum
            return value.name
        if hasattr(value, "__dataclass_fields__"):
            return {k: self._serialize_value(v) for k, v in value.__dict__.items()}
        return str(value)

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize the physical plan to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    def __repr__(self) -> str:
        """Return a string representation of the plan."""
        node_count = len(self._nodes)
        name_info = f'"{self.name}", ' if self.name else ""
        root_info = f", root={self._root_id}" if self._root_id is not None else ""
        return f"PhysicalPlan({name_info}nodes={node_count}{root_info})"