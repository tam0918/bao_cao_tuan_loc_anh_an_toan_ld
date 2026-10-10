"""Decode, validate and normalise an image for AI inference (spec sections 10, 39).

The original file is only read here. Routing always works on the original, so the
normalized copy is a cache that can be deleted at any time.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

from ..core.manifest import atomic_write_bytes
from ..errors import PermanentError
from .resize import fit_within

ACCEPTED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
MAX_PIXELS = 80_000_000  # guards against decompression bombs
NORMALIZED_MAX_SIDE = 1024
NORMALIZED_JPEG_QUALITY = 85


@dataclass(frozen=True)
class EncodedImage:
    data: bytes
    mime: str


def _flatten_alpha(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return img.convert("RGB")


def normalize_file(src: Path, cache_dir: Path, sha256: str) -> EncodedImage:
    """Return the inference copy of ``src``. Reuses the cached copy when present."""
    cached = cache_dir / f"{sha256}.jpg"
    if cached.exists():
        return EncodedImage(data=cached.read_bytes(), mime="image/jpeg")

    raw = src.read_bytes()
    try:
        probe = Image.open(BytesIO(raw))
        probe.verify()  # detects truncated/corrupt files without decoding pixels
        img = Image.open(BytesIO(raw))
        fmt = img.format
        width, height = img.size
    except Exception as exc:  # Pillow raises several unrelated types for bad input
        raise PermanentError("CORRUPTED_IMAGE", f"cannot decode {src.name}") from exc

    if fmt not in ACCEPTED_FORMATS:
        raise PermanentError("UNSUPPORTED_IMAGE", f"format {fmt} is not supported")
    if width * height > MAX_PIXELS:
        raise PermanentError("UNSUPPORTED_IMAGE", f"{width}x{height} exceeds pixel limit")

    img = ImageOps.exif_transpose(img)  # auto-rotate from EXIF orientation
    img = _flatten_alpha(img)
    img = fit_within(img, NORMALIZED_MAX_SIDE)

    out = BytesIO()
    img.save(out, format="JPEG", quality=NORMALIZED_JPEG_QUALITY, optimize=True)
    data = out.getvalue()
    atomic_write_bytes(cached, data)
    return EncodedImage(data=data, mime="image/jpeg")


def is_decodable(path: Path) -> bool:
    """True when ``path`` is an image Pillow can fully open. Used to validate AI output."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False
