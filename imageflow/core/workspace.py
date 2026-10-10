"""Workspace scanner (spec sections 4, 5, 9, 38, 39).

Contract:
* files directly under JOB_FOLDER are unclassified images
* immediate subdirectories are category folders (their names are the only taxonomy)
* nested folders are never categories
* ``.imageflow``, hidden and system folders are skipped
* formats outside the supported set are reported, never converted
"""

from __future__ import annotations

import os
import stat
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

SYSTEM_DIR_NAME = ".imageflow"
RULE_FILE_NAME = ".rule.txt"
RULE_MAX_CHARS = 2000

SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
# Recognised image formats that V1 does not process. Reported to the user, never converted.
UNSUPPORTED_IMAGE_EXTENSIONS = frozenset({".heic", ".heif", ".gif", ".bmp", ".tif", ".tiff"})

SYSTEM_FOLDER_NAMES = frozenset({"System Volume Information", "$RECYCLE.BIN", "$Recycle.Bin"})
IGNORED_FILE_NAMES = frozenset({"Thumbs.db", "desktop.ini"})


def nfc(name: str) -> str:
    """Normalise a name to NFC so Vietnamese diacritics compare equal across OSes."""
    return unicodedata.normalize("NFC", name)


@dataclass(frozen=True)
class ImageFile:
    name: str  # NFC file name
    path: Path  # absolute path on disk
    rel: str  # POSIX path relative to the job root (stored in the manifest)


@dataclass(frozen=True)
class FolderInfo:
    name: str  # NFC folder name, the choice label shown to the classifier
    path: Path
    dev: int
    ino: int
    rule: str | None  # content of the optional .rule.txt


@dataclass
class ScanResult:
    images: list[ImageFile] = field(default_factory=list)
    folders: list[FolderInfo] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)


def is_hidden(path: Path, st: os.stat_result) -> bool:
    if path.name.startswith("."):
        return True
    # Windows hidden attribute. Attribute is absent on POSIX, so this is a no-op there.
    attrs = getattr(st, "st_file_attributes", 0)
    return bool(attrs & getattr(stat, "FILE_ATTRIBUTE_HIDDEN", 0))


def read_rule(folder: Path) -> str | None:
    rule_path = folder / RULE_FILE_NAME
    if not rule_path.is_file():
        return None
    text = rule_path.read_text(encoding="utf-8-sig").strip()
    return text[:RULE_MAX_CHARS] or None


def scan_workspace(root: Path, artifact_suffixes: frozenset[str] = frozenset()) -> ScanResult:
    """``artifact_suffixes`` (e.g. ``_AI_PPE``) marks action outputs, which are never reclassified."""
    result = ScanResult()
    with os.scandir(root) as it:
        entries = sorted(it, key=lambda e: e.name)
    for entry in entries:
        path = Path(entry.path)
        st = entry.stat(follow_symlinks=False)
        if entry.name == SYSTEM_DIR_NAME or is_hidden(path, st):
            continue
        if entry.is_dir(follow_symlinks=False):
            if entry.name in SYSTEM_FOLDER_NAMES:
                continue
            result.folders.append(
                FolderInfo(
                    name=nfc(entry.name),
                    path=path,
                    dev=st.st_dev,
                    ino=st.st_ino,
                    rule=read_rule(path),
                )
            )
        elif entry.is_file(follow_symlinks=False):
            if entry.name in IGNORED_FILE_NAMES:
                continue
            suffix = Path(entry.name).suffix.lower()
            stem = Path(entry.name).stem
            if any(stem.endswith(art) for art in artifact_suffixes):
                continue
            if suffix in SUPPORTED_EXTENSIONS:
                result.images.append(ImageFile(name=nfc(entry.name), path=path, rel=entry.name))
            elif suffix in UNSUPPORTED_IMAGE_EXTENSIONS:
                result.unsupported.append(nfc(entry.name))
    return result


def choice_description(folder: FolderInfo) -> str:
    """Folder name, plus the optional rule text when the user wrote one (spec section 37)."""
    if folder.rule:
        return f"{folder.name}: {folder.rule}"
    return folder.name
