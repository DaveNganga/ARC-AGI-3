from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import warnings
from copy import deepcopy
from typing import Any

from .anthropic_provider import strip_unsupported, truncate_arrays, violations

DEFAULT_MODEL = "qwen3:8b"
DEFAULT_HOST = "http://127.0.0.1:11434"
# Ollama's own default is VRAM-dependent (4k/32k/256k), so it is pinned here instead: an
# overflowing prompt is shifted, not rejected, and the ARC demonstrations sit at the front.
DEFAULT_NUM_CTX = 32768
# A hosted model answers in seconds; a local 8B emitting a large constrained grid catalog
# does not. OPENAI_REQUEST_TIMEOUT's 180 would abort healthy calls.
DEFAULT_TIMEOUT = 900.0
# Ollama unloads after 5m by default, which would reload 5.2 GB between conditions.
DEFAULT_KEEP_ALIVE = "30m"
MAX_LOGGED_ERRORS = 20
# Attempts per call when the response breaks a constraint the grammar did not enforce.
CONSTRAINT_ATTEMPTS = 3
# arc_reasoner._call_json already wraps every call in call_with_retries(attempts=3); a second
# 3-attempt transport loop would stall a dead server for ~25 minutes at this timeout.
TRANSPORT_ATTEMPTS = 2
THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")
TRUTHY = {"1", "true", "yes", "on"}


def resolve_host(value: str | None) -> str:
    """Accept both ``127.0.0.1:11434`` (Ollama's own convention) and a full URL."""
    host = (value or DEFAULT_HOST).strip().rstrip("/")
    return host if "://" in host else f"http://{host}"


def extract_json(content: str) -> dict[str, Any]:
    """Parse the assistant message, tolerating a stray think block or Markdown fence."""
    text = FENCE.sub("", THINK_BLOCK.sub("", content)).strip()
    if not text:
        raise RuntimeError("The model returned no structured output")
    return json.loads(text)


class OllamaJSONCaller:
    """Local counterpart of ``OpenAIJSONCaller`` using Ollama's native /api/chat.

    The native endpoint is used rather than Ollama's OpenAI-compatible routes because
    ``options.num_ctx`` and ``options.seed`` cannot be expressed there, and both matter:
    a wrong context length silently shifts the oldest part of a long ARC grid prompt out
    of the window, and a fixed seed is what makes temperature 0 reproducible.

    Thinking is disabled and temperature is pinned to 0, so ``reasoning_effort`` has no
    meaning here. The requested value is kept in ``settings`` for the record, but the
    effort reported into the trace is always "none" rather than a figure never applied.
    """

    def __init__(self, model: str | None = None, reasoning_effort: str | None = None):
        self.model = model or os.environ.get("OLLAMA_MODEL") or DEFAULT_MODEL
        self.host = resolve_host(os.environ.get("OLLAMA_HOST"))
        self.requested_effort = reasoning_effort or os.environ.get("OLLAMA_EFFORT") or "none"
        # Never claim an effort that was not applied: arc_agent.py records this into the trace.
        self.reasoning_effort = "none"
        if self.requested_effort != "none":
            warnings.warn(
                f"Ollama ignores reasoning effort {self.requested_effort!r}; thinking is disabled "
                'and the trace will record effort "none".',
                RuntimeWarning, stacklevel=2,
            )
        self.think = os.environ.get("OLLAMA_THINK", "").strip().lower() in TRUTHY
        self.num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", DEFAULT_NUM_CTX))
        # A capped prediction length truncates mid-JSON; prefer to hit the context wall and say so.
        self.num_predict = int(os.environ.get("OLLAMA_NUM_PREDICT", "-1"))
        self.seed = int(os.environ.get("OLLAMA_SEED", "7"))
        self.timeout = float(os.environ.get("OLLAMA_REQUEST_TIMEOUT", DEFAULT_TIMEOUT))
        self.keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE", DEFAULT_KEEP_ALIVE)
        self.on_images = os.environ.get("OLLAMA_ON_IMAGES", "warn").strip().lower()
        self.request_count = 0
        self.constraint_retries = 0
        self.constraint_truncations = 0
        self.schema_relaxations = 0
        self.images_dropped = 0
        self.eval_counts: dict[str, int] = {"prompt_eval": 0, "eval": 0}
        self.call_errors: list[str] = []
        self._warned_images = False

    @property
    def effort(self) -> str:
        return self.reasoning_effort

    @property
    def settings(self) -> dict[str, Any]:
        return {
            "provider": "ollama", "model": self.model, "host": self.host,
            "effort": self.reasoning_effort, "requested_effort": self.requested_effort,
            "thinking_enabled": self.think, "temperature": 0,
            "num_ctx": self.num_ctx, "num_predict": self.num_predict, "seed": self.seed,
            "request_count": self.request_count,
            "constraint_retries": self.constraint_retries,
            "constraint_truncations": self.constraint_truncations,
            "schema_relaxations": self.schema_relaxations,
            # A text-only local model cannot see the rendered frames the hosted providers
            # receive; this count keeps the modality gap auditable in the saved trace.
            "images_dropped": self.images_dropped,
            "eval_counts": dict(self.eval_counts),
            "call_errors": list(self.call_errors[-MAX_LOGGED_ERRORS:]),
        }

    def __call__(self, prompt: str) -> str:
        message = {"role": "user", "content": (
            "Return only valid JSON matching the requested schema. "
            "Do not include Markdown fences.\n\n" + prompt)}
        return self._chat([message], schema=None)["message"]["content"]

    def call_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        name: str = "structured_result",
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return schema-constrained JSON. ``name`` is accepted for interface parity only."""
        if images:
            self._handle_images(len(images))
        sent_schema = deepcopy(schema)
        payload: dict[str, Any] = {}
        for attempt in range(CONSTRAINT_ATTEMPTS):
            response, sent_schema = self._request(prompt, sent_schema)
            payload = extract_json(response["message"]["content"])
            if not violations(schema, payload):
                return payload
            if attempt < CONSTRAINT_ATTEMPTS - 1:
                self.constraint_retries += 1
        self.constraint_truncations += 1
        return truncate_arrays(schema, payload)

    def _handle_images(self, count: int) -> None:
        if self.on_images == "error":
            raise RuntimeError(
                f"{self.model} is text-only; {count} rendered frame(s) were supplied. "
                "Set OLLAMA_ON_IMAGES=warn to drop them and accept the reduced input fidelity.")
        self.images_dropped += count
        if not self._warned_images:
            self._warned_images = True
            warnings.warn(
                f"{self.model} is text-only: dropping rendered frames. The grid is still supplied "
                "losslessly as text, but this run sees strictly less than an OpenAI or Anthropic "
                "run of the same harness, so comparisons are not modality-matched.",
                RuntimeWarning, stacklevel=3,
            )

    def _request(self, prompt: str, schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        # Ollama's structured-output guidance is to restate the schema in the prompt as well,
        # so the grammar is not the only thing steering the shape.
        grounded = (
            "Return only valid JSON matching this schema. Do not include Markdown fences.\n"
            f"SCHEMA: {json.dumps(schema, separators=(',', ':'))}\n\n" + prompt)
        messages = [{"role": "user", "content": grounded}]
        try:
            return self._chat(messages, schema), schema
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:300]
            self.call_errors.append(f"HTTPError {error.code}: {detail}")
            lowered = detail.lower()
            if error.code == 400 and "think" in lowered:
                # An older build, or a model without a thinking toggle: retry without the field.
                self.think = None
                return self._chat(messages, schema), schema
            if error.code == 400 and any(word in lowered for word in ("schema", "grammar", "format")):
                # Mirror the Anthropic path: restate the constraints in prose, then re-validate
                # locally with violations()/truncate_arrays().
                self.schema_relaxations += 1
                relaxed = strip_unsupported(schema)
                return self._chat(messages, relaxed), relaxed
            raise

    def _chat(self, messages: list[dict[str, Any]], schema: dict[str, Any] | None) -> dict[str, Any]:
        from ..retry import call_with_retries

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "num_ctx": self.num_ctx,
                        "num_predict": self.num_predict, "seed": self.seed,
                        "top_p": 1, "top_k": 0, "repeat_penalty": 1.0},
        }
        if self.think is not None:
            body["think"] = self.think
        if schema is not None:
            body["format"] = schema
        data = json.dumps(body).encode("utf-8")

        def once() -> dict[str, Any]:
            self.request_count += 1
            request = urllib.request.Request(
                f"{self.host}/api/chat", data=data,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))

        def transport() -> dict[str, Any]:
            try:
                return once()
            except urllib.error.HTTPError:
                raise  # Classified by _request; a 400 must not be retried blindly.
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                self.call_errors.append(f"{type(error).__name__}: {str(error)[:300]}")
                raise RuntimeError(
                    f"Ollama at {self.host} did not respond. Start it with `ollama serve` "
                    f"and confirm `ollama list` shows {self.model}.") from error

        response, _ = call_with_retries(transport, attempts=TRANSPORT_ATTEMPTS)
        self.eval_counts["prompt_eval"] += int(response.get("prompt_eval_count", 0) or 0)
        self.eval_counts["eval"] += int(response.get("eval_count", 0) or 0)
        if response.get("done_reason") == "length":
            raise RuntimeError(
                f"The model hit the {self.num_ctx}-token context before finishing the JSON; "
                "raise OLLAMA_NUM_CTX.")
        if not (response.get("message") or {}).get("content"):
            raise RuntimeError("The model returned no structured output")
        return response
