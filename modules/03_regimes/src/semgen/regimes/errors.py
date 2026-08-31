"""Custom errors for Module 03 risk regimes."""


class ConfigValidationError(ValueError):
    """Raised when regimes configuration fails validation."""


class InputValidationError(ValueError):
    """Raised when indicators input artifact fails schema or invariants."""


class OTNumericalError(ValueError):
    """Raised when OT computation encounters non-finite or unstable numerics."""


class OTConvergenceError(ValueError):
    """Raised when iterative OT computation fails to converge."""
