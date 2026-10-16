"""Client for the ImageFlow API gateway (spec sections 33, 34).

The desktop app never holds vendor keys. It authenticates with a revocable device token
and talks only to the gateway. The gateway owns Cloudflare/Seedream credentials and the
upstream request formats.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..errors import PermanentError, TransientError

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
PERMANENT_STATUS_CODES = frozenset({400, 404, 409, 413, 415, 422})


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """urllib copies the Authorization header onto the redirected request, so following a redirect
    could send the device token to another host or over plain http. The gateway never redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


class GatewayClient:
    def __init__(self, base_url: str, device_token: str, timeout: float = 60.0) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme != "https" and parsed.hostname not in LOCAL_HOSTS:
            raise ValueError("gateway URL must use https (plain http is allowed for localhost only)")
        if not device_token:
            raise ValueError("device token is required")
        self._base = base_url.rstrip("/")
        self._token = device_token
        self._timeout = timeout

    async def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._post_sync, path, payload)

    def _post_sync(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._base + path,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with _OPENER.open(req, timeout=self._timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            raise _map_http_error(exc) from exc
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            # Timeouts, resets, DNS and TLS failures, and a reply cut off mid-body (IncompleteRead is an
            # HTTPException, not an OSError) are all flaky-network symptoms. Retry them.
            raise TransientError("NETWORK", str(exc)) from exc
        try:
            data = json.loads(raw)
        except ValueError as exc:  # JSONDecodeError, and UnicodeDecodeError for a body that is not UTF-8
            raise PermanentError("INVALID_SCHEMA", "gateway returned non-JSON") from exc
        if not isinstance(data, dict):
            raise PermanentError("INVALID_SCHEMA", "gateway returned a non-object")
        return data


def _map_http_error(exc: urllib.error.HTTPError) -> Exception:
    status = exc.code
    upstream_code = _error_code(exc)
    if status in (401, 403):
        return PermanentError("AUTH_FAILED", f"gateway refused credentials ({status})")
    if status == 408:
        return TransientError("TIMEOUT", "gateway request timed out", retry_after=_retry_after(exc))
    if status == 429:
        return TransientError("RATE_LIMITED", "gateway rate limit", retry_after=_retry_after(exc))
    if status >= 500:
        return TransientError("UPSTREAM_UNAVAILABLE", f"gateway returned {status}", retry_after=_retry_after(exc))
    if status in PERMANENT_STATUS_CODES:
        return PermanentError(upstream_code or "INVALID_REQUEST", f"gateway rejected request ({status})")
    return PermanentError("INVALID_REQUEST", f"unexpected status {status}")


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    value = exc.headers.get("Retry-After") if exc.headers else None
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None  # HTTP-date form is not used by the gateway; fall back to our own backoff


def _error_code(exc: urllib.error.HTTPError) -> str | None:
    try:
        payload = json.loads(exc.read() or b"{}")
    except (ValueError, OSError):
        return None
    err = payload.get("error") if isinstance(payload, dict) else None
    code = err.get("code") if isinstance(err, dict) else None
    allowed = {"UNSUPPORTED_IMAGE", "CORRUPTED_IMAGE", "INVALID_REQUEST"}
    return code if code in allowed else None
