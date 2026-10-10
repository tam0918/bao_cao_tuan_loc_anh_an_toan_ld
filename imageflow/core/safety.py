"""Independent PPE safety screening after business-folder routing.

This is deliberately NOT part of the user's folder taxonomy. The first classifier chooses
one user-created folder. This second decision only decides whether the routed original
needs a derived PPE illustration.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..classifiers.base import Choice
from ..preprocessing.image_loader import EncodedImage
from .classification import ClassificationEngine

PPE_OK = "ppe_ok"
PPE_NEEDS_EDIT = "ppe_needs_edit"
PPE_UNCLEAR = "ppe_unclear"

PPE_CHOICES = [
    Choice(
        choice_id=PPE_OK,
        label="PPE compliant",
        description="Clearly visible workers already wear the required safety helmet and high-visibility reflective vest.",
    ),
    Choice(
        choice_id=PPE_NEEDS_EDIT,
        label="PPE missing",
        description="At least one clearly visible worker is missing a required safety helmet or high-visibility reflective vest.",
    ),
    Choice(
        choice_id=PPE_UNCLEAR,
        label="PPE unclear",
        description="The image does not provide enough visual evidence to decide PPE compliance reliably.",
    ),
]


@dataclass(frozen=True)
class SafetyDecision:
    status: str  # SAFE | NEEDS_EDIT | REVIEW
    confidence: float
    model: str | None
    provider: str | None
    attempts: list[dict]


class SafetyScreenEngine:
    def __init__(self, engine: ClassificationEngine) -> None:
        self._engine = engine

    async def decide(self, image: EncodedImage) -> SafetyDecision:
        decision = await self._engine.decide(image, PPE_CHOICES)
        attempts = [a.as_dict() for a in decision.attempts]
        if decision.kind != "ROUTE" or decision.choice_id == PPE_UNCLEAR:
            return SafetyDecision("REVIEW", decision.confidence, decision.model, decision.provider, attempts)
        if decision.choice_id == PPE_NEEDS_EDIT:
            return SafetyDecision("NEEDS_EDIT", decision.confidence, decision.model, decision.provider, attempts)
        return SafetyDecision("SAFE", decision.confidence, decision.model, decision.provider, attempts)
