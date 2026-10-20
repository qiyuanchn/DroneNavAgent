"""Tiny JSON-lines worker used by the managed-worker tests (goal.md M3).

The protocol is one JSON object per line on stdin/stdout:

request  ``{"id": 1, "method": "echo", "params": {"value": 7}}``
reply    ``{"id": 1, "ok": true, "result": {"value": 7}}``

Supported methods: ``ping``, ``echo``, ``sleep`` and ``shutdown``. Anything
else is reported as an error reply; the worker never writes non-protocol text
to stdout (the stderr banner below is opt-in and exists so tests can prove that
stderr ends up in the managed worker's log file).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

BANNER_ENV = "VLM_NAV_WORKER_ECHO_BANNER"


def _reply(request_id: Any, *, result: Any = None, error: str = "") -> None:
    message: dict[str, Any] = {"id": request_id, "ok": not error}
    if error:
        message["error"] = error
    else:
        message["result"] = result
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def dispatch(method: str, params: dict) -> Any:
    if method == "ping":
        return {"pong": True, "pid": os.getpid(), "time": time.time()}
    if method == "echo":
        return dict(params)
    if method == "sleep":
        seconds = float(params.get("seconds", 0.0))
        if seconds < 0:
            raise ValueError("sleep seconds must be >= 0")
        time.sleep(seconds)
        return {"slept": seconds}
    raise ValueError(f"unknown method: {method}")


def main() -> int:
    banner = os.environ.get(BANNER_ENV, "")
    if banner:
        sys.stderr.write(banner + "\n")
        sys.stderr.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _reply(None, error=f"invalid json: {exc}")
            continue
        if not isinstance(request, dict):
            _reply(None, error="request must be a JSON object")
            continue
        request_id = request.get("id")
        method = str(request.get("method", "") or "")
        params = request.get("params") or {}
        if not isinstance(params, dict):
            _reply(request_id, error="params must be an object")
            continue
        if method == "shutdown":
            _reply(request_id, result={"shutdown": True})
            return 0
        try:
            result = dispatch(method, params)
        except Exception as exc:  # noqa: BLE001 - reported over the protocol
            _reply(request_id, error=f"{type(exc).__name__}: {exc}")
            continue
        _reply(request_id, result=result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
