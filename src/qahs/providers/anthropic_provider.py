from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
EFFORTS = {"low", "medium", "high", "xhigh", "max"}
# Claude thinking counts toward max_tokens; 16000 cut off 2 of 47 calls in a 40-action lp85 run.
MAX_TOKENS = 64000
MAX_LOGGED_ERRORS = 20
# Attempts per call when the response breaks a constraint the API cannot enforce (see strip_unsupported).
CONSTRAINT_ATTEMPTS = 3

# JSON-schema keywords that OpenAI strict mode enforces but Anthropic structured outputs rejects with a 400.
UNSUPPORTED_KEYWORDS = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
                        "minLength", "maxLength", "maxItems")


def strip_unsupported(schema: Any) -> Any:
    """Copy ``schema`` without constraints Anthropic cannot compile; each one is restated in the description."""
    if isinstance(schema, list):
        return [strip_unsupported(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    result = {key: strip_unsupported(value) for key, value in schema.items()}
    notes = [f"{key}={schema[key]}" for key in UNSUPPORTED_KEYWORDS if key in schema]
    for key in UNSUPPORTED_KEYWORDS:
        result.pop(key, None)
    if schema.get("minItems", 0) > 1:
        notes.append(f"minItems={schema['minItems']}")
        result["minItems"] = 1
    if notes:
        description = schema.get("description", "")
        result["description"] = (description + " " if description else "") + "Constraints: " + ", ".join(notes) + "."
    return result


def violations(schema: Any, value: Any, path: str = "$") -> list[str]:
    """Constraint violations of ``value`` against the original (unstripped) ``schema``."""
    if not isinstance(schema, dict):
        return []
    problems = []
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            problems.append(f"{path}: {len(value)} items < minItems {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            problems.append(f"{path}: {len(value)} items > maxItems {schema['maxItems']}")
        for index, item in enumerate(value):
            problems += violations(schema.get("items"), item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            problems += violations(schema.get("properties", {}).get(key), item, f"{path}.{key}")
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", len(value)):
            problems.append(f"{path}: string length {len(value)} out of bounds")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < schema.get("minimum", value) or value > schema.get("maximum", value):
            problems.append(f"{path}: {value} out of bounds")
    return problems


def truncate_arrays(schema: Any, value: Any) -> Any:
    """Cut arrays to ``maxItems``, which OpenAI strict mode would have enforced while decoding."""
    if not isinstance(schema, dict):
        return value
    if isinstance(value, list):
        value = value[:schema["maxItems"]] if "maxItems" in schema else value
        return [truncate_arrays(schema.get("items"), item) for item in value]
    if isinstance(value, dict):
        return {key: truncate_arrays(schema.get("properties", {}).get(key), item) for key, item in value.items()}
    return value


def image_block(data_url: str) -> dict[str, Any]:
    header, _, data = data_url.partition(",")
    if not header.startswith("data:") or ";base64" not in header or not data:
        raise ValueError("Images must be base64 data URLs")
    return {"type": "image", "source": {"type": "base64", "media_type": header[5:].split(";")[0], "data": data}}


class AnthropicJSONCaller:
    """Claude counterpart of ``OpenAIJSONCaller`` using Messages API structured outputs.

    Server-side refusal fallbacks are deliberately not enabled: a silent switch to another
    model would contaminate a model comparison, so a refusal is raised and retried instead.
    """

    def __init__(self, model: str | None = None, reasoning_effort: str | None = None, client: Any = None):
        self.model = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL
        self.reasoning_effort = reasoning_effort or os.environ.get("ANTHROPIC_EFFORT") or DEFAULT_EFFORT
        if self.reasoning_effort not in EFFORTS:
            raise RuntimeError("ANTHROPIC_EFFORT must be low, medium, high, xhigh, or max")
        if client is None:
            try:
                import anthropic
            except ImportError as error:
                raise RuntimeError("Install the optional Claude dependency with: pip install -e '.[anthropic]'") from error
            # Resolves ANTHROPIC_API_KEY or an `ant auth login` profile.
            client = anthropic.Anthropic()
        self.client = client
        self.last_usage: dict[str, int] = {}
        self.constraint_retries = 0
        self.constraint_truncations = 0
        # Run totals, saved into each trace's model_settings so cost and failures are visible
        # even where a harness records no per-turn usage or omits error details.
        self.request_count = 0
        self.usage_totals: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
        self.call_errors: list[str] = []

    @property
    def effort(self) -> str:
        return self.reasoning_effort

    @property
    def settings(self) -> dict[str, Any]:
        return {"provider": "anthropic", "model": self.model, "effort": self.reasoning_effort,
                "max_tokens": MAX_TOKENS, "constraint_retries": self.constraint_retries,
                "constraint_truncations": self.constraint_truncations,
                "request_count": self.request_count,
                # Keys avoid the word "token", which the dashboard exporter redacts as a secret.
                "usage_totals": {key.removesuffix("_tokens"): value for key, value in self.usage_totals.items()},
                "call_errors": list(self.call_errors[-MAX_LOGGED_ERRORS:])}

    def call_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        name: str = "structured_result",
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return schema-constrained JSON. ``name`` is accepted for interface parity only."""
        sent_schema = strip_unsupported(deepcopy(schema))
        content: Any = prompt
        if images:
            content = [image_block(url) for url in images] + [{"type": "text", "text": prompt}]
        for attempt in range(CONSTRAINT_ATTEMPTS):
            payload = self._request(content, sent_schema)
            if not violations(schema, payload):
                return payload
            if attempt < CONSTRAINT_ATTEMPTS - 1:
                self.constraint_retries += 1
        self.constraint_truncations += 1
        return truncate_arrays(schema, payload)

    def _request(self, content: Any, schema: dict[str, Any]) -> dict[str, Any]:
        self.request_count += 1
        try:
            return self._request_once(content, schema)
        except Exception as error:
            # The error class and the API's or this module's own message; never request content.
            status = getattr(error, "status_code", None)
            self.call_errors.append(f"{type(error).__name__}{f' {status}' if status else ''}: {str(error)[:300]}")
            raise

    def _request_once(self, content: Any, schema: dict[str, Any]) -> dict[str, Any]:
        # Streamed: the SDK refuses a non-streaming request with a max_tokens this large,
        # and adaptive thinking at medium effort often needs well over 16k output tokens.
        with self.client.messages.stream(
            model=self.model,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": content}],
            output_config={
                "format": {"type": "json_schema", "schema": schema},
                "effort": self.reasoning_effort,
            },
        ) as stream:
            response = stream.get_final_message()
        usage = getattr(response, "usage", None)
        self.last_usage = {
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        }
        for key, value in self.last_usage.items():
            self.usage_totals[key] = self.usage_totals.get(key, 0) + value
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise RuntimeError(f"The model refused (category: {getattr(details, 'category', None)})")
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"The model hit max_tokens={MAX_TOKENS} before finishing the JSON")
        text = next((block.text for block in response.content if block.type == "text"), "")
        if not text:
            raise RuntimeError("The model returned no structured output")
        return json.loads(text)
