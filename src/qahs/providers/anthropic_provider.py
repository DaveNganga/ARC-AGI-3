from __future__ import annotations

import json
import os
from typing import Any

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
MAX_TOKENS = 16000


class AnthropicJSONCaller:
    """Claude counterpart of ``OpenAIJSONCaller`` using Messages API structured outputs.

    Server-side refusal fallbacks are deliberately not enabled: a silent switch to another
    model would contaminate a model comparison, so a refusal is raised and retried instead.
    """

    def __init__(self, model: str | None = None, effort: str | None = None, client: Any = None):
        self.model = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL
        self.effort = effort or os.environ.get("ANTHROPIC_EFFORT") or DEFAULT_EFFORT
        if client is None:
            try:
                import anthropic
            except ImportError as error:
                raise RuntimeError("Install the optional Claude dependency with: pip install -e '.[anthropic]'") from error
            # Resolves ANTHROPIC_API_KEY or an `ant auth login` profile.
            client = anthropic.Anthropic()
        self.client = client
        self.last_usage: dict[str, int] = {}

    @property
    def settings(self) -> dict[str, Any]:
        return {"provider": "anthropic", "model": self.model, "effort": self.effort, "max_tokens": MAX_TOKENS}

    def call_json(self, prompt: str, schema: dict[str, Any], name: str = "structured_result") -> dict[str, Any]:
        """Return schema-constrained JSON. ``name`` is accepted for interface parity only."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "format": {"type": "json_schema", "schema": schema},
                "effort": self.effort,
            },
        )
        usage = getattr(response, "usage", None)
        self.last_usage = {
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        }
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise RuntimeError(f"The model refused (category: {getattr(details, 'category', None)})")
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"The model hit max_tokens={MAX_TOKENS} before finishing the JSON")
        text = next((block.text for block in response.content if block.type == "text"), "")
        if not text:
            raise RuntimeError("The model returned no structured output")
        return json.loads(text)
