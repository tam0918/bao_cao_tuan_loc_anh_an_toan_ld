"""Resize helper for normalized inference images (spec section 10)."""

from __future__ import annotations

from PIL import Image

RESAMPLE = Image.Resampling.LANCZOS


def fit_within(img: Image.Image, max_side: int) -> Image.Image:
    """Return ``img`` scaled so its longest side is at most ``max_side``. Never upscales."""
    if max(img.size) <= max_side:
        return img
    out = img.copy()
    out.thumbnail((max_side, max_side), RESAMPLE)
    return out
