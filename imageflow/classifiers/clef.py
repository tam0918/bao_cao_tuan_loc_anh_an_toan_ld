"""Clef classifier provider (spec section 12).

Talks to the gateway's classify route. The model name (``clef-omni`` by default, ``clef-flash``
for fallback) is configuration. The gateway maps it to the upstream model.
"""

from __future__ import annotations

import base64
from typing import Any

from ..errors import PermanentError
from ..infrastructure.gateway import GatewayClient
from ..preprocessing.image_loader import EncodedImage
from .base import Choice, ClassificationResult, ClassifierProvider

CLASSIFY_PATH = "/v1/classify"
DEFAULT_MODEL = "clef-omni"
FALLBACK_MODEL = "clef-flash"  # asked when DEFAULT_MODEL is not confident enough


class ClefClassifierProvider(ClassifierProvider):
    name = "clef"

    def __init__(self, gateway: GatewayClient, model: str = DEFAULT_MODEL) -> None:
        self._gateway = gateway
        self.model = model

    async def classify(self, image: EncodedImage, choices: list[Choice]) -> ClassificationResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "image": {"mime": image.mime, "data_b64": base64.b64encode(image.data).decode("ascii")},
            # Criteria are built at runtime from the folder list. Nothing is hard-coded.
            "choices": [{"id": c.choice_id, "label": c.label, "description": c.description} for c in choices],
            "question": "Trong các thư mục được cung cấp, ảnh này phù hợp nhất với thư mục nào?",
        }
        data = await self._gateway.post_json(CLASSIFY_PATH, payload)
        return _parse(data, self.model)


def _parse(data: dict[str, Any], requested_model: str) -> ClassificationResult:
    choice_id = data.get("choice_id")
    probabilities = data.get("probabilities")
    confidence = data.get("confidence")
    if not isinstance(choice_id, str) or not isinstance(probabilities, dict):
        raise PermanentError("INVALID_SCHEMA", "classify response missing choice_id or probabilities")
    if not isinstance(confidence, (int, float)):
        confidence = probabilities.get(choice_id, 0.0)
    return ClassificationResult(
        choice_id=choice_id,
        confidence=float(confidence),
        probabilities={str(k): float(v) for k, v in probabilities.items() if isinstance(v, (int, float))},
        model=str(data.get("model", requested_model)),
    )
