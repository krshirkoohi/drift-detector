"""driftd proxy: an OpenAI-compatible chat proxy with inline drift detection.

Point ANY OpenAI-compatible client (Gemini CLI via compatibility endpoint,
LibreChat, Open WebUI, LangChain, the openai SDK, Ollama frontends) at this
proxy instead of the upstream, and every assistant response is scored for
drift transparently.

  client --> driftd proxy (/v1/chat/completions) --> upstream model API
                     |
                     +-- scores each assistant response
                     +-- x-drift-* response headers
                     +-- optional inline [drift warning] appended to content

Baseline policy (per V1 design insights): built from the first N assistant
responses of the session, then frozen; subsequent turns are scored against it.
Sessions are keyed by the X-Drift-Session header, or a hash of the first user
message when absent.

Run:  python -m drift_detector.proxy --port 8899 \
          --upstream https://api.openai.com --provider local --inline-warnings
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .baseline import BaselineStore
from .detector import DriftDetector
from .embedding import get_provider

CONFIG = {
    "upstream": "http://127.0.0.1:9100",
    "baseline_n": 5,
    "inline_warnings": False,
}
PROVIDER = None
SESSIONS: dict[str, dict] = {}  # sid -> {"responses": [...], "detector": DriftDetector | None}


def _session_id(headers, payload: dict) -> str:
    sid = headers.get("X-Drift-Session")
    if sid:
        return sid
    for msg in payload.get("messages", []):
        if msg.get("role") == "user":
            return hashlib.sha256(str(msg.get("content", "")).encode()).hexdigest()[:12]
    return "default"


def _score_turn(sid: str, text: str) -> dict:
    state = SESSIONS.setdefault(sid, {"responses": [], "detector": None})
    state["responses"].append(text)
    n = CONFIG["baseline_n"]
    if state["detector"] is None:
        if len(state["responses"]) < n:
            return {"phase": "collecting-baseline", "collected": len(state["responses"]), "needed": n}
        baseline = BaselineStore(PROVIDER).build(state["responses"][:n])
        state["detector"] = DriftDetector(baseline, PROVIDER)
        return {"phase": "baseline-ready", "collected": n, "needed": n}
    score = state["detector"].score(text)
    return {"phase": "scoring", **score.to_dict()}


class Handler(BaseHTTPRequestHandler):
    server_version = "driftd-proxy/0.2"

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._send(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return self._send(400, {"error": "invalid JSON"})

        # Forward to upstream unchanged (streaming disabled for v0).
        payload["stream"] = False
        req = urllib.request.Request(
            f"{CONFIG['upstream'].rstrip('/')}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": self.headers.get("Authorization", ""),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read())
        except urllib.error.URLError as e:
            return self._send(502, {"error": f"upstream unreachable: {e}"})

        sid = _session_id(self.headers, payload)
        drift = {}
        try:
            text = body["choices"][0]["message"]["content"]
            drift = _score_turn(sid, text)
            if CONFIG["inline_warnings"] and drift.get("drifted"):
                body["choices"][0]["message"]["content"] = (
                    text
                    + "\n\n[driftd] Sustained semantic drift detected "
                    + f"(cosine {drift['cosine_distance']}). Consider compacting or resetting context."
                )
        except (KeyError, IndexError, TypeError):
            drift = {"phase": "skipped", "reason": "unexpected upstream response shape"}
        body["drift"] = drift  # also embedded for clients that read JSON only

        headers = {
            "x-drift-session": sid,
            "x-drift-phase": str(drift.get("phase", "")),
            "x-drift-cosine": str(drift.get("cosine_distance", "")),
            "x-drift-alarm": str(drift.get("drifted", "")).lower(),
        }
        self._send(200, body, headers)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            return self._send(200, {"status": "ok", "sessions": len(SESSIONS)})
        self._send(404, {"error": "not found"})

    def _send(self, code: int, payload: dict, extra_headers: dict | None = None) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    global PROVIDER
    p = argparse.ArgumentParser(description="driftd OpenAI-compatible drift proxy")
    p.add_argument("--port", type=int, default=8899)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--upstream", required=True, help="Upstream base URL, e.g. https://api.openai.com")
    p.add_argument("--provider", default="local", help="local | gemini | openai")
    p.add_argument("--baseline-n", type=int, default=5)
    p.add_argument("--inline-warnings", action="store_true")
    args = p.parse_args()
    PROVIDER = get_provider(args.provider)
    CONFIG.update(upstream=args.upstream, baseline_n=args.baseline_n, inline_warnings=args.inline_warnings)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"driftd proxy on {args.host}:{args.port} -> {args.upstream}")
    server.serve_forever()


if __name__ == "__main__":
    main()
