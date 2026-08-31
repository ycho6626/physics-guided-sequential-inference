"""Custom errors for indicators module."""


class ConfigValidationError(ValueError):
    """Raised when indicators config fails validation."""


class InputValidationError(ValueError):
    """Raised when input spectra artifact fails schema or invariant checks."""
