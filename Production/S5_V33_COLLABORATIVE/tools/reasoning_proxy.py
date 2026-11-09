#!/usr/bin/env python3
"""Rewriting proxy that lets the frozen S5 package run with qwen3 reasoning enabled.

`stronger_ai.py` builds its Ollama request internally: `think` is hardcoded False and the
`leading_rules` object carries neither a `required` list nor `minItems`, so `{}` satisfies the
grammar. The package must not be edited, so the request is rewritten in flight instead and
`run_s5_benchmark.py --ollama-url` is pointed here.

Two rewrites, both necessary and both logged:

* `think` -> True when --think is given. This is the experiment.
* `leading_rules` gains `required` for all six rule families and `minItems: 1` on each. Without
  it a reasoning model returns `leading_rules: {}` -- measured on 8 of 8 sampled turns -- which
  makes `_parse_rule_candidate` return None, so the arbiter can never choose the LLM and the
  collaborative condition silently degenerates into plain QAHS.

Every call is appended to a JSONL log so the intervention is auditable rather than invisible:
what was rewritten, and what came back (families, thinking length, tokens, latency).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FAMILIES = ("P", "G", "M", "T", "I", "S")
LOG_LOCK = threading.Lock()
ARGS = None


def rewrite(body: dict) -> tuple[dict, list[str]]:
    """Apply the two rewrites, returning the new body and what changed."""
    changed: list[str] = []

    if ARGS.think and body.get("think") is not True:
        body["think"] = True
        changed.append("think->True")

    fmt = body.get("format")
    if not isinstance(fmt, dict):
        # Only the structured-output path is expected here. Anything else means the package
        # changed shape; say so rather than quietly forwarding an unrewritten request.
        raise ValueError("request has no JSON-schema 'format'; refusing to forward silently")
    lr = (fmt.get("properties") or {}).get("leading_rules")
    if not isinstance(lr, dict) or not (lr.get("properties") or {}):
        raise ValueError("'leading_rules' missing from the schema; refusing to forward silently")
    if lr.get("required") != list(FAMILIES):
        lr["required"] = list(FAMILIES)
        changed.append("leading_rules.required")
    for fam in FAMILIES:
        prop = lr["properties"].get(fam)
        if isinstance(prop, dict) and prop.get("minItems") != 1:
            prop["minItems"] = 1
            changed.append(f"minItems[{fam}]")

    # Only inject a cap when one is asked for. A cap is tempting as a safety valve but it bounds
    # thinking and content *together*: a 6000-token cap once produced a 26,797-character think with
    # zero budget left for the JSON, so content came back empty and the package's unguarded
    # json.loads aborted the whole suite. The package itself sets no num_predict; default to that.
    if ARGS.num_predict and ARGS.num_predict > 0:
        opts = body.setdefault("options", {})
        if opts.get("num_predict") != ARGS.num_predict:
            opts["num_predict"] = ARGS.num_predict
            changed.append(f"num_predict->{ARGS.num_predict}")

    return body, changed


def log(record: dict) -> None:
    with LOG_LOCK:
        with open(ARGS.log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args) -> None:  # keep stderr for real problems only
        pass

    def _send(self, code: int, payload: bytes, kind: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # /api/tags and friends pass straight through
        try:
            with urllib.request.urlopen(ARGS.upstream + self.path, timeout=30) as response:
                self._send(200, response.read())
        except Exception as error:
            self._send(502, json.dumps({"error": str(error)}).encode())

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path != "/api/chat":
            self._send(404, json.dumps({"error": f"unproxied path {self.path}"}).encode())
            return
        started = time.time()
        try:
            body, changed = rewrite(json.loads(raw.decode("utf-8")))
        except Exception as error:
            # A rewrite failure must stop the run, not degrade it to an unmodified request.
            log({"ts": time.time(), "fatal": f"{type(error).__name__}: {error}"})
            print(f"reasoning_proxy: {error}", file=sys.stderr, flush=True)
            self._send(400, json.dumps({"error": str(error)}).encode())
            return

        data = json.dumps(body).encode("utf-8")

        def ask() -> bytes:
            request = urllib.request.Request(
                ARGS.upstream + "/api/chat", data=data,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=ARGS.timeout) as response:
                return response.read()

        # stronger_ai.py has no retry and does a bare json.loads on message.content, so one empty
        # or truncated response kills the suite. Retry the identical request once -- a model call is
        # idempotent, and generation is not deterministic here, so a shorter think is likely. Every
        # retry is logged; a second failure is passed through rather than papered over.
        # Greedy decoding with thinking occasionally sends qwen3 into a repetition loop: twice in
        # ~140 calls a request produced nothing at all before the socket timeout. The package has no
        # retry and does a bare json.loads on message.content, so either that stall or a truncated
        # body kills the suite. Retry here instead: a model call is idempotent and generation is not
        # deterministic, so a retry usually lands a normal-length think. Every attempt is logged.
        retried = 0
        envelope = None
        message = {}
        last_error = None
        while retried <= ARGS.retries:
            try:
                out = ask()
                envelope = json.loads(out.decode("utf-8"))
                message = envelope.get("message") or {}
                reason = None
                if envelope.get("done_reason") == "length":
                    reason = "length"
                elif not (message.get("content") or "").strip():
                    reason = "empty_content"
                if reason is None:
                    last_error = None
                    break
                last_error = reason
            except Exception as error:          # stalled or unreachable upstream
                last_error = f"{type(error).__name__}: {error}"
                envelope = None
            if retried == ARGS.retries:
                break
            retried += 1
            log({"ts": time.time(), "retry": retried, "retry_reason": last_error,
                 "eval_count": (envelope or {}).get("eval_count"),
                 "thinking_chars": len(message.get("thinking") or "")})

        if envelope is None or last_error:
            log({"ts": time.time(), "elapsed": round(time.time() - started, 2),
                 "rewrote": changed, "retried": retried,
                 "upstream_error": last_error or "unknown"})
            self._send(502, json.dumps({"error": str(last_error)}).encode())
            return
        families: list[str] = []
        parsed_ok = True
        try:
            rules = (json.loads(message.get("content") or "") or {}).get("leading_rules") or {}
            families = [f for f in FAMILIES if rules.get(f)]
        except Exception:
            parsed_ok = False
        log({"ts": time.time(), "elapsed": round(time.time() - started, 2),
             "rewrote": changed, "think": body.get("think"), "retried": retried,
             "prompt_eval_count": envelope.get("prompt_eval_count"),
             "eval_count": envelope.get("eval_count"),
             "done_reason": envelope.get("done_reason"),
             "thinking_chars": len(message.get("thinking") or ""),
             "content_chars": len(message.get("content") or ""),
             "content_is_json": parsed_ok,
             "families": len(families), "complete_six": len(families) == 6})
        self._send(200, out)


def main() -> None:
    global ARGS
    parser = argparse.ArgumentParser(description="Rewriting proxy for the frozen S5 Ollama path")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--upstream", default="http://127.0.0.1:11434")
    parser.add_argument("--think", action="store_true", help="rewrite think:false to think:true")
    parser.add_argument("--num-predict", type=int, default=12000,
                        help="token cap; <=0 means none, matching the package. A cap bounds thinking "
                             "and content together and starved the JSON once, so it is off by default")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--retries", type=int, default=2,
                        help="extra attempts on a stalled, truncated or empty response")
    parser.add_argument("--log", default="reasoning_proxy.jsonl")
    ARGS = parser.parse_args()
    ARGS.upstream = ARGS.upstream.rstrip("/")

    server = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)

    def shutdown(_signum: int, _frame: object) -> None:
        raise SystemExit(0)

    for name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), shutdown)

    print(f"reasoning_proxy on 127.0.0.1:{ARGS.port} -> {ARGS.upstream} "
          f"(think={'rewritten to True' if ARGS.think else 'left as sent'}, "
          f"required families always, num_predict={ARGS.num_predict}) log={ARGS.log}",
          flush=True)
    try:
        server.serve_forever()
    except SystemExit:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
