"""Request-body limits for the SocialGraph-FM Governance multipart upload boundary."""

from __future__ import annotations

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

GOVERNANCE_MULTIPART_OVERHEAD_MAX_BYTES = 64 * 1024
GOVERNANCE_UPLOAD_CHUNK_BYTES = 1024 * 1024
GOVERNANCE_UPLOAD_PATHS = frozenset(
    {
        "/api/v2/gfm/governance/artifacts",
        "/api/v2/gfm/governance/artifacts/compatibility",
        "/api/v2/gfm/governance/target-tasks",
    }
)


class GovernanceUploadLimitMiddleware:
    """Reject oversized Governance multipart bodies before parsing finishes."""

    def __init__(self, app: ASGIApp, *, max_bundle_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = (
            max_bundle_bytes + GOVERNANCE_MULTIPART_OVERHEAD_MAX_BYTES
        )

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            status_code=413,
            content={"detail": {"code": "GOVERNANCE_BUNDLE_TOO_LARGE"}},
        )
        await response(scope, receive, send)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in GOVERNANCE_UPLOAD_PATHS
        ):
            await self.app(scope, receive, send)
            return

        declared_lengths: list[int] = []
        for name, raw_value in scope.get("headers", []):
            if name.lower() != b"content-length":
                continue
            try:
                declared_lengths.append(int(raw_value))
            except ValueError:
                # A missing or malformed declaration cannot be trusted; the streaming
                # counter below remains the authoritative request-body boundary.
                continue
        if any(length > self.max_body_bytes for length in declared_lengths):
            await self._reject(scope, receive, send)
            return

        received_bytes = 0
        body_too_large = False

        async def limited_receive() -> Message:
            nonlocal body_too_large, received_bytes
            if body_too_large:
                return {"type": "http.request", "body": b"", "more_body": False}
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    body_too_large = True
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def limited_send(message: Message) -> None:
            if not body_too_large:
                await send(message)

        await self.app(scope, limited_receive, limited_send)
        if body_too_large:
            await self._reject(scope, receive, send)


__all__ = [
    "GOVERNANCE_MULTIPART_OVERHEAD_MAX_BYTES",
    "GOVERNANCE_UPLOAD_CHUNK_BYTES",
    "GovernanceUploadLimitMiddleware",
]
