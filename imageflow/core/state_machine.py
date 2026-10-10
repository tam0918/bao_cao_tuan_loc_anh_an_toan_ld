"""Image state machine (spec section 28).

Every transition is checked here, so an illegal jump fails loudly instead of silently
corrupting the manifest.
"""

from __future__ import annotations

from enum import Enum


class ImageState(str, Enum):
    DISCOVERED = "DISCOVERED"
    NORMALIZED = "NORMALIZED"
    CLASSIFYING = "CLASSIFYING"
    CLASSIFIED = "CLASSIFIED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    ROUTING = "ROUTING"
    ROUTED = "ROUTED"
    ACTION_PENDING = "ACTION_PENDING"
    ACTION_RUNNING = "ACTION_RUNNING"
    ACTION_FAILED = "ACTION_FAILED"
    COMPLETE = "COMPLETE"
    # Permanent error (unsupported/corrupted file, auth-independent rejection). Terminal.
    FAILED = "FAILED"


TERMINAL_STATES = frozenset({ImageState.COMPLETE, ImageState.FAILED})

_TRANSITIONS: dict[ImageState, set[ImageState]] = {
    ImageState.DISCOVERED: {ImageState.NORMALIZED},
    # A transient classifier failure returns CLASSIFYING -> NORMALIZED, so the next run retries.
    ImageState.NORMALIZED: {ImageState.CLASSIFYING},
    ImageState.CLASSIFYING: {ImageState.CLASSIFIED, ImageState.REVIEW_REQUIRED, ImageState.NORMALIZED},
    ImageState.CLASSIFIED: {ImageState.ROUTING, ImageState.REVIEW_REQUIRED},
    ImageState.REVIEW_REQUIRED: {ImageState.CLASSIFIED},  # the user assigns a folder by hand
    ImageState.ROUTING: {ImageState.ROUTED, ImageState.CLASSIFIED},
    ImageState.ROUTED: {ImageState.ACTION_PENDING, ImageState.COMPLETE},
    ImageState.ACTION_PENDING: {ImageState.ACTION_RUNNING},
    ImageState.ACTION_RUNNING: {ImageState.COMPLETE, ImageState.ACTION_FAILED},
    # ACTION_FAILED -> ACTION_RUNNING retries on resume. -> COMPLETE when the user removed the action.
    ImageState.ACTION_FAILED: {ImageState.ACTION_RUNNING, ImageState.COMPLETE},
    ImageState.COMPLETE: set(),
    ImageState.FAILED: set(),
}

# A permanent error can surface at any non-terminal stage, so FAILED is reachable from all of them.
for _state in ImageState:
    if _state not in TERMINAL_STATES:
        _TRANSITIONS[_state].add(ImageState.FAILED)

# States a resumed job must pick up again. COMPLETE and FAILED are terminal.
# REVIEW_REQUIRED is not resumable: it waits for the user.
RESUMABLE_STATES = frozenset(
    {
        ImageState.DISCOVERED,
        ImageState.NORMALIZED,
        ImageState.CLASSIFYING,
        ImageState.CLASSIFIED,
        ImageState.ROUTING,
        ImageState.ROUTED,
        ImageState.ACTION_PENDING,
        ImageState.ACTION_RUNNING,
        ImageState.ACTION_FAILED,
    }
)


class IllegalTransition(RuntimeError):
    pass


def check_transition(current: ImageState, target: ImageState) -> None:
    if target not in _TRANSITIONS[current]:
        raise IllegalTransition(f"{current.value} -> {target.value} is not allowed")
