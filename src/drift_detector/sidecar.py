"""driftd sidecar: a zero-dependency (stdlib + numpy) HTTP drift-scoring service.

Any chat technology (web app, Discord/Telegram bot, agent framework, CLI,
Slack app) POSTs each assistant turn and gets a drift verdict back. The chat
stack never changes how it talks to its model; it just reports turns.

Endpoints:
  GET  /healthz                     liveness probe
  POST /sessions                    {"baseline": [texts], "metric"?, "use_trend"?}
  POST /sessions/<id>/turns         {"text": "assistant response"}
  GET  /sessions/<id>               summary + per-turn history

Run:  python -m drift_detector.sidecar --port 8787 --provider local
"""
from __future__ import annotations

import argparse
import json
import re
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .baseline import BaselineStore
from .detector import DriftDetector
from .embedding import get_provider

SESSIONS: dict[str, DriftDetector] = {}
PROVIDER = None  # set in main()


class Handler(BaseHTTPRequestHandler):
    server_version = "driftd/0.2"

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_OPTIONS(self) -> None:  # CORS preflight for browser dashboards
        self._send(204, {})

    def do_GET(self) -> None:
        if self.path == "/healthz":
            return self._send(200, {"status": "ok", "sessions": len(SESSIONS)})
        m = re.fullmatch(r"/sessions/([\w-]+)", self.path)
        if m:
            det = SESSIONS.get(m.group(1))
            if not det:
                return self._send(404, {"error": "unknown session"})
            return self._send(200, {
                "summary": det.summary(),
                "history": [t.to_dict() for t in det.history],
                "thresholds": {
                    "cosine": det.baseline.cosine_threshold,
                    "euclidean": det.baseline.euclidean_threshold,
                },
            })
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            data = self._read_json()
        except json.JSONDecodeError:
            return self._send(400, {"error": "invalid JSON"})
        if self.path == "/sessions":
            texts = data.get("baseline", [])
            if len(texts) < 3:
                return self._send(400, {"error": "baseline needs at least 3 samples"})
            baseline = BaselineStore(PROVIDER).build(texts)
            det = DriftDetector(
                baseline,
                PROVIDER,
                metric=data.get("metric", "cosine"),
                use_trend=bool(data.get("use_trend", True)),
            )
            sid = uuid.uuid4().hex[:12]
            SESSIONS[sid] = det
            return self._send(201, {
                "session_id": sid,
                "n_samples": baseline.n_samples,
                "cosine_threshold": round(baseline.cosine_threshold, 4),
                "euclidean_threshold": round(baseline.euclidean_threshold, 4),
            })
        m = re.fullmatch(r"/sessions/([\w-]+)/turns", self.path)
        if m:
            det = SESSIONS.get(m.group(1))
            if not det:
                return self._send(404, {"error": "unknown session"})
            text = data.get("text", "")
            if not text:
                return self._send(400, {"error": "missing text"})
            return self._send(200, det.score(text).to_dict())
        self._send(404, {"error": "not found"})

    def log_message(self, *args) -> None:  # keep stdout clean
        pass


def main() -> None:
    global PROVIDER
    p = argparse.ArgumentParser(description="driftd sidecar service")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--provider", default="local", help="local | gemini | openai")
    args = p.parse_args()
    PROVIDER = get_provider(args.provider)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"driftd sidecar listening on {args.host}:{args.port} (provider={args.provider})")
    server.serve_forever()


if __name__ == "__main__":
    main()
