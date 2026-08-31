"""Custom exceptions for experiments layer."""


class ExperimentError(Exception):
    """Base experiments error."""


class ConfigValidationError(ExperimentError):
    """Raised on invalid experiment config."""


class SplitError(ExperimentError):
    """Raised when deterministic split integrity fails."""


class EventExtractionError(ExperimentError):
    """Raised when event extraction cannot proceed."""


class MetricComputationError(ExperimentError):
    """Raised when metrics cannot be computed."""


class PipelineExecutionError(ExperimentError):
    """Raised when subprocess module orchestration fails."""


class UnsupportedVariantError(ExperimentError):
    """Raised for explicitly unsupported baseline/ablation variants."""
