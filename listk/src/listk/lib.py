"""
The main entry point for SoliceDB when using it as a library.
Users write SoliceDB programs as a DAG using the provided API.
SoliceDB lazily collects the DAG as an immutable IR, then lazily performs query optimization and execution when the user
runs the eval() command.
"""