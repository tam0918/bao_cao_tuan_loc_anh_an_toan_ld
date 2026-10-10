"""Routing engine (spec section 16).

The routing engine does not understand images. It moves or copies one file into one
destination folder, with these guarantees:

* the destination is chosen deterministically (``IMG001.jpg`` -> ``IMG001__2.jpg``)
* an existing file is never overwritten (hard link creates the target atomically)
* a move that crashed halfway can be finished from the on-disk state alone
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .hashing import sha256_file

MAX_SUFFIX_SEARCH = 10_000


class DestinationMissing(RuntimeError):
    """The category folder no longer exists. Routing never creates it silently."""


@dataclass(frozen=True)
class RecoveryPlan:
    action: str  # "DONE" | "FINISH_MOVE" | "REROUTE" | "LOST_SOURCE"
    destination: Path | None


def deterministic_name(folder: Path, filename: str, taken: set[str] | None = None) -> Path:
    """Return the first free name: ``stem.ext``, then ``stem__2.ext``, ``stem__3.ext``, ..."""
    candidate = folder / filename
    if not _exists(folder, candidate.name, taken):
        return candidate
    stem, suffix = Path(filename).stem, Path(filename).suffix
    for n in range(2, MAX_SUFFIX_SEARCH):
        candidate = folder / f"{stem}__{n}{suffix}"
        if not _exists(folder, candidate.name, taken):
            return candidate
    raise RuntimeError(f"no free file name for {filename} in {folder}")


def _exists(folder: Path, name: str, taken: set[str] | None) -> bool:
    if taken and name in taken:
        return True
    return (folder / name).exists()


def route_file(
    src: Path,
    folder: Path,
    filename: str,
    mode: str,
    *,
    dest: Path | None = None,
    temp_dir: Path | None = None,
) -> Path:
    """Move or copy ``src`` into ``folder``. Returns the final destination path.

    ``dest`` pins the planned name (written to the manifest before this call). If that name
    has since been taken, a new deterministic name is chosen and returned. ``temp_dir`` holds
    the staging file for copies, so a crash never leaves a stray file inside a user's folder.
    """
    if mode not in ("move", "copy"):
        raise ValueError(f"unknown routing mode {mode!r}")
    if not folder.is_dir():
        raise DestinationMissing(str(folder))
    target = dest if dest is not None else deterministic_name(folder, filename)
    while True:
        try:
            _link_or_copy(src, target, mode, temp_dir or folder)
            return target
        except FileExistsError:
            # Another file took the planned name after the manifest was written. Pick the next one.
            target = deterministic_name(folder, filename, taken={target.name})


def _link_or_copy(src: Path, dst: Path, mode: str, temp_dir: Path) -> None:
    """Create ``dst`` only if absent.

    * move: hard-link ``src`` to ``dst`` (atomic create-if-absent), then remove ``src``.
    * copy: write an independent file first, then hard-link it into place. Hard-linking the
      source directly would make the copy share bytes with the root original, so editing one
      would silently edit the other.
    """
    if mode == "copy":
        temp_dir.mkdir(parents=True, exist_ok=True)
        staged = temp_dir / f".{dst.name}.part"
        shutil.copy2(src, staged)
        try:
            os.link(staged, dst)
        except FileExistsError:
            staged.unlink(missing_ok=True)
            raise
        except OSError:
            # Filesystem without hard links. Check first, then rename.
            if dst.exists():
                staged.unlink(missing_ok=True)
                raise FileExistsError(str(dst)) from None
            os.replace(staged, dst)
            return
        staged.unlink(missing_ok=True)
        return

    try:
        os.link(src, dst)  # atomic create-if-absent on the same volume
    except FileExistsError:
        raise
    except OSError:
        # Filesystem without hard links (some FAT/SMB setups). Check first, then rename.
        if dst.exists():
            raise FileExistsError(str(dst)) from None
        os.replace(src, dst)
        return
    os.unlink(src)


def plan_recovery(src: Path, dst: Path | None, expected_sha: str, mode: str) -> RecoveryPlan:
    """Decide how to finish a routing step that was interrupted (spec Test F).

    * destination holds the right bytes and the source is the same inode -> remove the source
    * destination holds the right bytes and the source is gone -> nothing left to do
    * destination missing or holds other bytes, source present -> route again
    * neither present -> the original is lost; report it, never guess
    """
    src_exists = src.exists()
    if dst is not None and dst.exists() and sha256_file(dst) == expected_sha:
        if mode == "move" and src_exists and os.path.samefile(src, dst):
            return RecoveryPlan("FINISH_MOVE", dst)
        return RecoveryPlan("DONE", dst)
    if src_exists:
        return RecoveryPlan("REROUTE", dst)
    return RecoveryPlan("LOST_SOURCE", dst)
