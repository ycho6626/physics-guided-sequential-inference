"""Custom exceptions for Module 06 policies."""


class PoliciesError(Exception):
    """Base class for policy module errors."""


class ConfigValidationError(PoliciesError):
    """Raised when policy config/schema validation fails."""


class InputValidationError(PoliciesError):
    """Raised when stability input artifacts violate contracts."""


class PolicyValidationError(PoliciesError):
    """Raised when policy logic/state transitions are invalid."""
