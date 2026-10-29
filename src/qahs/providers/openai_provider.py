from __future__ import annotations

import json
import os
from typing import Any


class OpenAIJSONCaller:
    """Callable adapter for ``StructuredLLMInterpreter`` using the Responses API."""

    def __init__(self, model: str | None = None):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in the environment before enabling the OpenAI interpreter")
        self.model = model or os.environ.get("OPENAI_MODEL")
        if not self.model:
            raise RuntimeError("Set OPENAI_MODEL to a model available to your API project")
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("Install the optional LLM dependency with: pip install -e '.[llm]'") from error
        self.client = OpenAI(api_key=api_key)
        # Only sent when set: some reasoning models reject an explicit temperature.
        temperature = os.environ.get("OPENAI_TEMPERATURE", "").strip()
        self.temperature = float(temperature) if temperature else None
        self.last_usage: dict[str, int] = {}

    @property
    def settings(self) -> dict[str, Any]:
        return {"model": self.model, "temperature": self.temperature}

    def _sampling(self) -> dict[str, Any]:
        return {} if self.temperature is None else {"temperature": self.temperature}

    def __call__(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=(
                "Return only valid JSON matching the requested schema. "
                "Do not include Markdown fences.\n\n" + prompt
            ),
            **self._sampling(),
        )
        text = response.output_text
        if not text:
            raise RuntimeError("The model returned no text")
        return text

    def call_json(self, prompt: str, schema: dict[str, Any], name: str = "structured_result") -> dict[str, Any]:
        """Return schema-constrained JSON through the Responses API."""
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": name,
                    "strict": True,
                    "schema": schema,
                }
            },
            **self._sampling(),
        )
        usage = getattr(response, "usage", None)
        self.last_usage = {
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        }
        text = response.output_text
        if not text:
            raise RuntimeError("The model returned no structured output")
        return json.loads(text)

