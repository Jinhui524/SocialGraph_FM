"""Loopback OpenAI-compatible provider used by clean-clone acceptance."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Literal

ApiMode = Literal["chat_completions", "responses", "anthropic_messages"]

_CONNECTION_RESULT = {"socialgraph_fm_connection_check": "ok"}
_HEADING_PATTERN = re.compile(
    r"Begin exactly with the Markdown heading ## (?P<heading>[^.]+)\."
)


def _prompt(request: Mapping[str, Any], api_mode: ApiMode, role: str) -> str | None:
    if api_mode == "responses":
        value = request.get("instructions" if role == "system" else "input")
        return value if isinstance(value, str) else None
    if api_mode == "anthropic_messages" and role == "system":
        value = request.get("system")
        return value if isinstance(value, str) else None
    messages = request.get("messages")
    if not isinstance(messages, list):
        return None
    for item in messages:
        if (
            isinstance(item, Mapping)
            and item.get("role") == role
            and isinstance(item.get("content"), str)
        ):
            return str(item["content"])
    return None


def _strict_result(request: Mapping[str, Any], api_mode: ApiMode) -> dict[str, Any]:
    """Return only the bounded JSON contracts exercised by public acceptance."""

    system_prompt = _prompt(request, api_mode, "system") or ""
    user_prompt = _prompt(request, api_mode, "user") or ""

    if (
        "connection verifier" in system_prompt
        and "socialgraph_fm_connection_check" in user_prompt
    ):
        return _CONNECTION_RESULT
    if '"toolCalls"' in system_prompt:
        return {"toolCalls": []}
    if 'Return JSON exactly as {"narrative":"..."}' in system_prompt:
        return {
            "narrative": "该草稿仅依据已绑定事实与引用生成，仍需由人工核对后使用。"
        }
    if 'Return JSON exactly as {"answer":"..."}' in system_prompt:
        heading_match = _HEADING_PATTERN.search(system_prompt)
        prefix = f"## {heading_match.group('heading')}\n\n" if heading_match else ""
        return {
            "answer": (
                prefix
                + "请分别核对已登记的直接关系、上下文关联与潜在线索。"
                "潜在线索不属于已确认事实；发布时间、原始内容与采集来源仍需人工核验，"
                "最终判断应由分析人员记录。"
            )
        }
    raise ValueError("unsupported mock LLM request contract")


def _envelope(result: Mapping[str, Any], api_mode: ApiMode, model: object) -> dict[str, Any]:
    content = json.dumps(dict(result), ensure_ascii=False, separators=(",", ":"))
    model_name = model if isinstance(model, str) and model else "socialgraph-fm-mock"
    if api_mode == "responses":
        return {
            "id": "resp_socialgraph_fm_mock",
            "object": "response",
            "status": "completed",
            "model": model_name,
            "output": [
                {
                    "id": "msg_socialgraph_fm_mock",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": content,
                            "annotations": [],
                        }
                    ],
                }
            ],
            "output_text": content,
        }
    if api_mode == "anthropic_messages":
        return {
            "id": "msg_socialgraph_fm_mock",
            "type": "message",
            "role": "assistant",
            "model": model_name,
            "content": [{"type": "text", "text": content}],
            "stop_reason": "end_turn",
        }
    return {
        "id": "chatcmpl_socialgraph_fm_mock",
        "object": "chat.completion",
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


class _Handler(BaseHTTPRequestHandler):
    server_version = "SocialGraphFMMock/1.0"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _json(self, status: int, payload: object) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_048_576:
            self._json(400, {"error": "invalid_body"})
            return
        try:
            request = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json(400, {"error": "invalid_json"})
            return
        if self.path == "/v1/responses":
            api_mode: ApiMode = "responses"
        elif self.path == "/v1/chat/completions":
            api_mode = "chat_completions"
        elif self.path == "/v1/messages":
            api_mode = "anthropic_messages"
        else:
            self._json(404, {"error": "not_found"})
            return
        bearer = self.headers.get("Authorization", "").startswith("Bearer ")
        x_api_key = bool(self.headers.get("x-api-key", ""))
        anthropic_version = self.headers.get("anthropic-version")
        authorized = (
            bearer
            if api_mode != "anthropic_messages"
            else (bearer or x_api_key) and anthropic_version == "2023-06-01"
        )
        if not isinstance(request, dict) or not authorized:
            self._json(401, {"error": "unauthorized"})
            return
        try:
            result = _strict_result(request, api_mode)
        except ValueError:
            self._json(
                422,
                {
                    "error": {
                        "type": "unsupported_mock_contract",
                        "message": "request does not match a public acceptance contract",
                    }
                },
            )
            return
        self._json(200, _envelope(result, api_mode, request.get("model")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    arguments = parser.parse_args()
    ThreadingHTTPServer(("127.0.0.1", arguments.port), _Handler).serve_forever()


if __name__ == "__main__":
    main()
