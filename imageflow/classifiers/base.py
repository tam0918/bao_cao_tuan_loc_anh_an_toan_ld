"""Classifier provider interface (spec sections 11, 36).

Core depends only on ``ClassifierProvider``. Vendor names (Cloudflare, Clef, ...) live in
the concrete provider modules, never in the pipeline.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..preprocessing.image_loader import EncodedImage


@dataclass(frozen=True)
class Choice:
    """One classification option. ``choice_id`` is the stable folder ID, never the folder name."""

    choice_id: str
    label: str  # folder name
    description: str  # folder name + optional .rule.txt text


@dataclass(frozen=True)
class ClassificationResult:
    choice_id: str
    confidence: float
    probabilities: dict[str, float]
    model: str


class ClassifierProvider(ABC):
    """Anything that can pick one choice for one image. Exactly one image per call (spec 13)."""

    name: str  # provider family, recorded in the manifest, e.g. "clef"
    model: str  # model identifier, e.g. "clef-flash"

    @abstractmethod
    async def classify(self, image: EncodedImage, choices: list[Choice]) -> ClassificationResult:
        """Return the best choice. Raise ``TransientError`` or ``PermanentError`` on failure."""


def validate_probabilities(result: ClassificationResult, choices: list[Choice]) -> dict[str, float] | None:
    """Return the probabilities restricted to known choices, or None if the output is unusable.

    An unusable response is never turned into a guess (spec section 14).
    """
    known = {c.choice_id for c in choices}
    if result.choice_id not in known:
        return None
    probs: dict[str, float] = {}
    for cid in known:
        value = result.probabilities.get(cid, 0.0)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            return None
        probs[cid] = float(value)
    if max(probs, key=probs.__getitem__) != result.choice_id:
        return None  # the model's own pick disagrees with its probabilities
    return probs
