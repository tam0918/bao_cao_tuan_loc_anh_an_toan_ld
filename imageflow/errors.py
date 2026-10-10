"""Error taxonomy shared by providers, retries and the pipeline (spec section 26)."""

from __future__ import annotations


class ImageFlowError(Exception):
    """Base class for errors that ImageFlow handles on purpose."""


class TransientError(ImageFlowError):
    """Retryable: timeout, 429, 5xx, temporary network failure.

    ``retry_after`` is the server-provided delay in seconds, when one was sent.
    """

    def __init__(self, code: str, message: str = "", retry_after: float | None = None) -> None:
        super().__init__(message or code)
        self.code = code
        self.retry_after = retry_after


class PermanentError(ImageFlowError):
    """Fail fast: unsupported/corrupted image, auth failure, invalid schema."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code
