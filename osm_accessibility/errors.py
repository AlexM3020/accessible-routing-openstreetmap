"""Exceptions for graph research and invalid source data."""


class ResearchError(Exception):
    """Base exception for a failed research operation."""


class DataError(ResearchError, ValueError):
    """Source data or numerical parameters violate an invariant."""


class NoRouteFound(ResearchError):
    """No route satisfies the requested constraints."""


class DatabaseError(ResearchError):
    """A graph data store could not complete an operation."""
