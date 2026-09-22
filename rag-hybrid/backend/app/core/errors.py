"""Domain-specific exceptions, mapped to HTTP responses in `api/errors.py`."""

from __future__ import annotations


class RAGError(Exception):
    """Base class for expected application errors."""


class LLMConfigError(RAGError):
    """The LLM provider's API key is missing or invalid."""


class PdfError(RAGError):
    """The uploaded PDF could not be read or contains no extractable text."""


class IndexMismatchError(RAGError):
    """The persisted vector index was built with a different embedding model."""


class NotFoundError(RAGError):
    """A requested resource does not exist."""


class LLMError(RAGError):
    """A call to the LLM provider failed (bad key, no credit, unknown model, network...). Message is user-readable."""
