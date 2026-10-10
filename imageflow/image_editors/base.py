"""Image edit provider interface (spec section 22). Actions depend on this, never on a vendor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..preprocessing.image_loader import EncodedImage


@dataclass(frozen=True)
class ImageEditResult:
    image: EncodedImage
    model: str


class ImageEditProvider(ABC):
    name: str
    model: str

    @abstractmethod
    async def edit(self, image: EncodedImage, instruction: str, metadata: dict[str, Any]) -> ImageEditResult:
        """Return an edited copy. The input is never modified."""
