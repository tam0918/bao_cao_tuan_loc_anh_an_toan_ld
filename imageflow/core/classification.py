"""Classification engine: primary model, fallback model, then review (spec sections 14, 15).

Auto-route requires ``top1 >= ROUTE_CONFIDENCE`` AND ``top1 - top2 >= MIN_MARGIN``.
The engine never guesses. A response that fails validation counts as "no decision".
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Literal

from ..classifiers.base import Choice, ClassifierProvider, validate_probabilities
from ..errors import PermanentError, TransientError
from ..infrastructure.retries import RetryPolicy, retry_async
from ..preprocessing.image_loader import EncodedImage

ROUTE_CONFIDENCE = 0.85
MIN_MARGIN = 0.15


class ClassifierUnavailable(Exception):
    """Every model failed transiently. The image stays NORMALIZED and is retried on the next run."""


@dataclass
class Attempt:
    provider: str
    model: str
    outcome: Literal["accepted", "rejected", "no_decision", "error"]
    latency_ms: float = 0.0
    retries: int = 0
    error_code: str | None = None
    top_choice: str | None = None
    top_probability: float | None = None
    second_choice: str | None = None
    second_probability: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    kind: Literal["ROUTE", "REVIEW"]
    choice_id: str | None
    confidence: float
    second_choice: str | None
    second_probability: float
    model: str | None
    provider: str | None
    attempts: list[Attempt] = field(default_factory=list)
    candidates: list[tuple[str, float]] = field(default_factory=list)  # top-3, shown in the review UI


def rank(probs: dict[str, float]) -> list[tuple[str, float]]:
    """Sort by probability, then by id so ties resolve the same way every run."""
    return sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))


def passes_thresholds(ranked: list[tuple[str, float]], route_confidence: float, min_margin: float) -> bool:
    top1 = ranked[0][1]
    top2 = ranked[1][1] if len(ranked) > 1 else 0.0
    return top1 >= route_confidence and top1 - top2 >= min_margin


class ClassificationEngine:
    def __init__(
        self,
        primary: ClassifierProvider,
        fallback: ClassifierProvider | None = None,
        *,
        route_confidence: float = ROUTE_CONFIDENCE,
        min_margin: float = MIN_MARGIN,
        retry_policy: RetryPolicy = RetryPolicy(),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._providers = [p for p in (primary, fallback) if p is not None]
        self._route_confidence = route_confidence
        self._min_margin = min_margin
        self._retry_policy = retry_policy
        self._sleep = sleep

    async def decide(self, image: EncodedImage, choices: list[Choice]) -> Decision:
        attempts: list[Attempt] = []
        best_probs: dict[str, float] | None = None

        for provider in self._providers:
            attempt, probs = await self._attempt(provider, image, choices)
            attempts.append(attempt)
            if attempt.outcome == "error":
                if attempt.error_code == "NETWORK":
                    break  # the fallback model sits behind the same gateway, so it cannot do better
                continue
            if probs is None:
                continue
            ranked = rank(probs)
            best_probs = probs
            if passes_thresholds(ranked, self._route_confidence, self._min_margin):
                attempt.outcome = "accepted"
                return _route_decision(provider, ranked, attempts)
            attempt.outcome = "rejected"

        if best_probs is None and all(a.outcome == "error" for a in attempts):
            raise ClassifierUnavailable("all classifier models failed transiently")
        ranked = rank(best_probs) if best_probs else []
        return Decision(
            kind="REVIEW",
            choice_id=None,
            confidence=ranked[0][1] if ranked else 0.0,
            second_choice=ranked[1][0] if len(ranked) > 1 else None,
            second_probability=ranked[1][1] if len(ranked) > 1 else 0.0,
            model=None,
            provider=None,
            attempts=attempts,
            candidates=ranked[:3],
        )

    async def _attempt(
        self, provider: ClassifierProvider, image: EncodedImage, choices: list[Choice]
    ) -> tuple[Attempt, dict[str, float] | None]:
        started = time.perf_counter()
        try:
            result, retries = await retry_async(
                lambda: provider.classify(image, choices), self._retry_policy, sleep=self._sleep
            )
        except TransientError as exc:
            return (
                Attempt(provider.name, provider.model, "error", _elapsed_ms(started), retries=self._retry_policy.max_retries, error_code=exc.code),
                None,
            )
        except PermanentError as exc:
            if exc.code == "INVALID_SCHEMA":
                return Attempt(provider.name, provider.model, "no_decision", _elapsed_ms(started), error_code=exc.code), None
            raise  # auth failure, unsupported image, ... are handled by the pipeline

        latency = _elapsed_ms(started)
        probs = validate_probabilities(result, choices)
        if probs is None:
            return Attempt(provider.name, result.model, "no_decision", latency, retries=retries, error_code="INVALID_SCHEMA"), None
        ranked = rank(probs)
        attempt = Attempt(
            provider.name,
            result.model,
            "rejected",
            latency,
            retries=retries,
            top_choice=ranked[0][0],
            top_probability=ranked[0][1],
            second_choice=ranked[1][0] if len(ranked) > 1 else None,
            second_probability=ranked[1][1] if len(ranked) > 1 else 0.0,
        )
        return attempt, probs


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 2)


def _route_decision(provider: ClassifierProvider, ranked: list[tuple[str, float]], attempts: list[Attempt]) -> Decision:
    second_id, second_p = ranked[1] if len(ranked) > 1 else (None, 0.0)
    return Decision(
        kind="ROUTE",
        choice_id=ranked[0][0],
        confidence=ranked[0][1],
        second_choice=second_id,
        second_probability=second_p,
        model=provider.model,
        provider=provider.name,
        attempts=attempts,
        candidates=ranked[:3],
    )
