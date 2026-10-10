"""Retry policy (spec section 26): transient errors only, bounded, exponential backoff with jitter."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from ..errors import TransientError

T = TypeVar("T")

MAX_RETRIES = 2


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = MAX_RETRIES
    base_delay: float = 0.5
    max_delay: float = 8.0


def backoff_delay(policy: RetryPolicy, attempt: int, retry_after: float | None, rand: Callable[[], float]) -> float:
    """Delay before retry number ``attempt`` (0-based).

    A server-provided ``Retry-After`` wins over our own schedule. Otherwise use equal-jitter
    exponential backoff: half the capped exponential delay plus a random share of the other half.
    """
    if retry_after is not None:
        return min(policy.max_delay, max(0.0, retry_after))
    capped = min(policy.max_delay, policy.base_delay * (2**attempt))
    return capped / 2 + rand() * capped / 2


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy = RetryPolicy(),
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rand: Callable[[], float] = random.random,
) -> tuple[T, int]:
    """Run ``fn`` and return ``(value, retries_used)``.

    ``PermanentError`` and any non-ImageFlow exception propagate immediately (fail fast).
    """
    retries = 0
    while True:
        try:
            return await fn(), retries
        except TransientError as exc:
            if retries >= policy.max_retries:
                raise
            await sleep(backoff_delay(policy, retries, exc.retry_after, rand))
            retries += 1
