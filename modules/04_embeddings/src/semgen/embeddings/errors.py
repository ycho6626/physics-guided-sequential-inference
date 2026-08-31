"""Custom error types for Module 04 embeddings."""

from __future__ import annotations


class EmbeddingsError(Exception):
    """Base class for embeddings module failures."""


class ConfigValidationError(EmbeddingsError):
    """Raised when YAML/JSON-schema/semantic config validation fails."""


class InputValidationError(EmbeddingsError):
    """Raised when input artifacts violate the module contract."""


class TrainingError(EmbeddingsError):
    """Raised when deterministic model training cannot proceed safely."""
