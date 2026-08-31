"""Custom errors for Module 05 stability."""


class StabilityError(Exception):
    """Base class for stability module errors."""


class ConfigValidationError(StabilityError):
    """Raised when config/schema validation fails."""


class InputValidationError(StabilityError):
    """Raised when input artifacts violate contracts."""


class ModelValidationError(StabilityError):
    """Raised when HMM parameters/outputs are invalid."""


class TrainingError(StabilityError):
    """Raised when deterministic model fitting cannot proceed."""
