"""Seedream 5.0 Flash editor (spec sections 22, 33).

Calls the gateway's edit route. The upstream model ID is configuration, so it can change
without touching the action plugins.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any

from ..errors import PermanentError
from ..infrastructure.gateway import GatewayClient
from ..preprocessing.image_loader import EncodedImage
from .base import ImageEditProvider, ImageEditResult

EDIT_PATH = "/v1/edit"
DEFAULT_MODEL = "seedream-5.0-flash"


class Seedream5FlashProvider(ImageEditProvider):
    name = "seedream"

    def __init__(self, gateway: GatewayClient, model: str = DEFAULT_MODEL) -> None:
        self._gateway = gateway
        self.model = model

    async def edit(self, image: EncodedImage, instruction: str, metadata: dict[str, Any]) -> ImageEditResult:
        payload = {
            "model": self.model,
            "image": {"mime": image.mime, "data_b64": base64.b64encode(image.data).decode("ascii")},
            "instruction": instruction,
            "metadata": metadata,
        }
        data = await self._gateway.post_json(EDIT_PATH, payload)
        out = data.get("image")
        if not isinstance(out, dict) or not isinstance(out.get("data_b64"), str):
            raise PermanentError("INVALID_SCHEMA", "edit response missing image.data_b64")
        try:
            raw = base64.b64decode(out["data_b64"], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise PermanentError("INVALID_SCHEMA", "edit response is not valid base64") from exc
        return ImageEditResult(
            image=EncodedImage(data=raw, mime=str(out.get("mime", "image/png"))),
            model=str(data.get("model", self.model)),
        )
