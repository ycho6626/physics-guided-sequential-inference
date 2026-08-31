"""Custom exceptions for Module 07 reports."""


class ReportsError(Exception):
    """Base class for Module 07 errors."""


class ConfigValidationError(ReportsError):
    """Raised when config or schema validation fails."""


class InputValidationError(ReportsError):
    """Raised when input artifact contracts are violated."""


class RenderingError(ReportsError):
    """Raised when report rendering fails."""


class ValidationError(ReportsError):
    """Raised when post-render grounding validation fails."""
