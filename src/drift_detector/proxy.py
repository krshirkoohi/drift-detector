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


def _score_turn(
    sid: str,
    text: str,
    history_len: int | None = None,
    prompt_tokens: int | None = None,
    is_compacted: bool = False,
    compacted_summary: str | None = None,
) -> dict:
    state = SESSIONS.setdefault(sid, {
        "responses": [],
        "detector": None,
        "prev_history_len": None,
        "prev_prompt_tokens": None,
    })
    
    # Check for compaction triggers in proxy
    compaction_detected = is_compacted
    if history_len is not None and state.get("prev_history_len") is not None and history_len < state["prev_history_len"]:
        compaction_detected = True
    if (
        prompt_tokens is not None
        and state.get("prev_prompt_tokens") is not None
        and state["prev_prompt_tokens"] > 200
        and prompt_tokens < int(state["prev_prompt_tokens"] * 0.6)
    ):
        compaction_detected = True

    if history_len is not None:
        state["prev_history_len"] = history_len
    if prompt_tokens is not None:
        state["prev_prompt_tokens"] = prompt_tokens

    if compaction_detected and state["detector"] is not None:
        state["detector"].handle_compaction(compacted_summary=compacted_summary)

    state["responses"].append(text)
    n = CONFIG["baseline_n"]
    if state["detector"] is None:
        if len(state["responses"]) < n:
            return {
                "phase": "collecting-baseline",
                "collected": len(state["responses"]),
                "needed": n,
                "compacted_reset": compaction_detected,
            }
        baseline = BaselineStore(PROVIDER).build(state["responses"][:n])
        state["detector"] = DriftDetector(baseline, PROVIDER)
        return {
            "phase": "baseline-ready",
            "collected": n,
            "needed": n,
            "compacted_reset": compaction_detected,
        }
    score = state["detector"].score(
        text,
        history_len=history_len,
        prompt_tokens=prompt_tokens,
        is_compacted=compaction_detected,
        compacted_summary=compacted_summary,
    )
    res = {"phase": "scoring", **score.to_dict()}
    if compaction_detected:
        res["compacted_reset"] = True
    return res


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
            messages = payload.get("messages", [])
            history_len = len(messages)
            prompt_tokens = body.get("usage", {}).get("prompt_tokens") or sum(len(str(m.get("content", ""))) // 4 for m in messages)
            
            # Check for /compact command in recent user prompt
            is_compacted = False
            compacted_summary = None
            for m in reversed(messages):
                if m.get("role") == "user":
                    u_content = str(m.get("content", "")).strip()
                    if u_content.startswith("/compact"):
                        is_compacted = True
                        parts = u_content.split(" ", 1)
                        if len(parts) > 1:
                            compacted_summary = parts[1].strip()
                    break

            drift = _score_turn(
                sid,
                text,
                history_len=history_len,
                prompt_tokens=prompt_tokens,
                is_compacted=is_compacted,
                compacted_summary=compacted_summary,
            )
            if CONFIG["inline_warnings"] and drift.get("drifted"):
                body["choices"][0]["message"]["content"] = (
                    text
                    + "\n\n[driftd] Sustained semantic drift detected "
                    + f"(cosine {drift['cosine_distance']}). Consider compacting or resetting context."
                )
            elif CONFIG["inline_warnings"] and drift.get("compacted_reset"):
                body["choices"][0]["message"]["content"] = (
                    text + "\n\n[driftd] Chat compacted: detector reset."
                )
        except (KeyError, IndexError, TypeError):
            drift = {"phase": "skipped", "reason": "unexpected upstream response shape"}
        body["drift"] = drift  # also embedded for clients that read JSON only

        headers = {
            "x-drift-session": sid,
            "x-drift-phase": str(drift.get("phase", "")),
            "x-drift-cosine": str(drift.get("cosine_distance", "")),
            "x-drift-alarm": str(drift.get("drifted", "")).lower(),
            "x-drift-compaction-reset": str(bool(drift.get("compacted_reset", False))).lower(),
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
