"""Action plugin interface (spec sections 18, 19).

Core dispatches by ``action_type`` and never imports business logic. Plugins register
themselves in the composition root (``app.py``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..preprocessing.image_loader import EncodedImage


@dataclass(frozen=True)
class ActionContext:
    job_id: str
    image_sha256: str
    folder_id: str
    original_name: str  # for logs and metadata only


@dataclass(frozen=True)
class ActionResult:
    output: EncodedImage | None
    status: str  # "SUCCESS" | "FAILED"
    provider: str | None = None
    model: str | None = None
    latency_ms: float = 0.0
    retries: int = 0
    error_code: str | None = None


class ActionPlugin(ABC):
    action_type: str  # e.g. "PPE_SIMULATION"
    output_suffix: str  # appended to the original stem, e.g. "_AI_PPE"

    @abstractmethod
    async def execute(self, source_image: EncodedImage, context: ActionContext) -> ActionResult:
        """Produce a derived output. The source file is never modified or deleted."""


class ActionRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, ActionPlugin] = {}

    def register(self, plugin: ActionPlugin) -> None:
        if plugin.action_type in self._plugins:
            raise ValueError(f"duplicate action type {plugin.action_type}")
        self._plugins[plugin.action_type] = plugin

    def get(self, action_type: str) -> ActionPlugin:
        return self._plugins[action_type]

    def known_types(self) -> frozenset[str]:
        return frozenset(self._plugins)

    def output_suffixes(self) -> frozenset[str]:
        return frozenset(p.output_suffix for p in self._plugins.values())
