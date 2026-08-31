"""Loopback Chat Completions provider used only by repository acceptance tests."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_CONNECTION_RESULT = {"socialgraph_fm_connection_check": "ok"}
_HEADING_PATTERN = re.compile(
    r"Begin exactly with the Markdown heading ## (?P<heading>[^.]+)\."
)
_CHINESE_HEADING_PATTERN = re.compile(
    r"必须以 Markdown 标题 ## (?P<heading>[^。]+) 开头。"
)


def _prompt(request: Mapping[str, Any], role: str) -> str | None:
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


def _strict_result(request: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the bounded JSON contracts exercised by public acceptance."""

    system_prompt = _prompt(request, "system") or ""
    user_prompt = _prompt(request, "user") or ""
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
    if (
        'Return JSON exactly as {"answer":"..."}' in system_prompt
        or '仅返回 {"answer":"..."}' in system_prompt
    ):
        heading_match = _HEADING_PATTERN.search(system_prompt) or _CHINESE_HEADING_PATTERN.search(
            system_prompt
        )
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


def _envelope(result: Mapping[str, Any], model: object) -> dict[str, Any]:
    content = json.dumps(dict(result), ensure_ascii=False, separators=(",", ":"))
    model_name = model if isinstance(model, str) and model else "socialgraph-fm-mock"
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
    server_version = "SocialGraphFMMock/2.0"

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
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not_found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_048_576:
            self._json(400, {"error": "invalid_body"})
            return
        try:
            request = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json(400, {"error": "invalid_json"})
            return
        if (
            not isinstance(request, dict)
            or not self.headers.get("Authorization", "").startswith("Bearer ")
        ):
            self._json(401, {"error": "unauthorized"})
            return
        try:
            result = _strict_result(request)
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
        self._json(200, _envelope(result, request.get("model")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    arguments = parser.parse_args()
    ThreadingHTTPServer(("127.0.0.1", arguments.port), _Handler).serve_forever()


if __name__ == "__main__":
    main()
