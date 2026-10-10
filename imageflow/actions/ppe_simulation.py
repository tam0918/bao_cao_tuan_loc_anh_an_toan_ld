"""PPE simulation action (spec sections 20, 21).

Runs only on images routed into a folder the user explicitly assigned this action to.
The prompt lives in the package. The user never types one.
"""

from __future__ import annotations

import time

from ..errors import PermanentError, TransientError
from ..image_editors.base import ImageEditProvider
from ..infrastructure.retries import RetryPolicy, retry_async
from ..preprocessing.image_loader import EncodedImage
from .base import ActionContext, ActionPlugin, ActionResult

ACTION_TYPE = "PPE_SIMULATION"
OUTPUT_SUFFIX = "_AI_PPE"

PPE_PROMPT = """Chỉ tạo phiên bản minh họa PPE.

Nếu người nhìn rõ trong ảnh đang thiếu:
- mũ bảo hộ
- áo phản quang

thì bổ sung PPE tương ứng.

Không thay đổi:
- khuôn mặt;
- nhận dạng người;
- tư thế;
- công cụ;
- máy móc;
- kết cấu;
- bố cục;
- môi trường xung quanh.

Không thay đổi người đã có đầy đủ PPE.

Giữ ánh sáng và phối cảnh nhất quán."""


class PpeSimulationAction(ActionPlugin):
    action_type = ACTION_TYPE
    output_suffix = OUTPUT_SUFFIX

    def __init__(self, editor: ImageEditProvider, retry_policy: RetryPolicy = RetryPolicy(), sleep=None) -> None:
        self._editor = editor
        self._retry_policy = retry_policy
        self._sleep = sleep

    async def execute(self, source_image: EncodedImage, context: ActionContext) -> ActionResult:
        started = time.perf_counter()
        metadata = {
            "action": ACTION_TYPE,
            "job_id": context.job_id,
            "image_sha256": context.image_sha256,
            "folder_id": context.folder_id,
            # Marks the output as derived. It is not evidence that PPE existed on site.
            "derived": True,
        }
        kwargs = {} if self._sleep is None else {"sleep": self._sleep}
        try:
            result, retries = await retry_async(
                lambda: self._editor.edit(source_image, PPE_PROMPT, metadata),
                self._retry_policy,
                **kwargs,
            )
        except TransientError as exc:
            return ActionResult(
                output=None,
                status="FAILED",
                provider=self._editor.name,
                latency_ms=_ms(started),
                retries=self._retry_policy.max_retries,
                error_code=exc.code,
            )
        except PermanentError as exc:
            if exc.code == "AUTH_FAILED":
                raise  # job-level problem, the pipeline aborts
            return ActionResult(output=None, status="FAILED", provider=self._editor.name, latency_ms=_ms(started), error_code=exc.code)
        return ActionResult(
            output=result.image,
            status="SUCCESS",
            provider=self._editor.name,
            model=result.model,
            latency_ms=_ms(started),
            retries=retries,
        )


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 2)
