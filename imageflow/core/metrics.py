"""Job metrics (spec section 41). Computed from observed events, never estimated."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .classification import Decision


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile. None when there is no data."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return round(ordered[rank - 1], 2)


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


@dataclass
class JobMetrics:
    started: float = field(default_factory=time.perf_counter)
    decision_latencies: list[float] = field(default_factory=list)
    primary_attempts: int = 0
    primary_accepted: int = 0
    fallback_attempts: int = 0
    decisions: int = 0
    reviews: int = 0
    images_routed: int = 0
    images_actioned: int = 0
    edit_attempts: int = 0
    edit_successes: int = 0
    edit_latencies: list[float] = field(default_factory=list)

    def observe_decision(self, decision: Decision) -> None:
        self.decisions += 1
        if decision.kind == "REVIEW":
            self.reviews += 1
        for idx, attempt in enumerate(decision.attempts):
            if idx == 0:
                self.primary_attempts += 1
                if attempt.outcome == "accepted":
                    self.primary_accepted += 1
            else:
                self.fallback_attempts += 1
            self.decision_latencies.append(attempt.latency_ms)

    def observe_route(self) -> None:
        self.images_routed += 1

    def observe_action(self, status: str, latency_ms: float) -> None:
        self.edit_attempts += 1
        self.edit_latencies.append(latency_ms)
        if status == "SUCCESS":
            self.edit_successes += 1
            self.images_actioned += 1

    def summary(self) -> dict[str, object]:
        return {
            "classification_count": self.decisions,
            "classification_latency_p50_ms": percentile(self.decision_latencies, 50),
            "classification_latency_p95_ms": percentile(self.decision_latencies, 95),
            "clef_flash_accept_rate": _ratio(self.primary_accepted, self.primary_attempts),
            "clef_fallback_rate": _ratio(self.fallback_attempts, self.decisions),
            "manual_review_rate": _ratio(self.reviews, self.decisions),
            "images_routed": self.images_routed,
            "images_actioned": self.images_actioned,
            "edit_success_rate": _ratio(self.edit_successes, self.edit_attempts),
            "edit_latency_p50_ms": percentile(self.edit_latencies, 50),
            "edit_latency_p95_ms": percentile(self.edit_latencies, 95),
            "total_job_duration_s": round(time.perf_counter() - self.started, 3),
        }
