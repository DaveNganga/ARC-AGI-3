from __future__ import annotations

import json
import os
from typing import Any


REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}


class OpenAIJSONCaller:
    """Callable adapter for ``StructuredLLMInterpreter`` using the Responses API."""

    def __init__(self, model: str | None = None, reasoning_effort: str | None = None):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in the environment before enabling the OpenAI interpreter")
        self.model = model or os.environ.get("OPENAI_MODEL")
        if not self.model:
            raise RuntimeError("Set OPENAI_MODEL to a model available to your API project")
        self.reasoning_effort = reasoning_effort or os.environ.get("OPENAI_REASONING_EFFORT", "medium")
        if self.reasoning_effort not in REASONING_EFFORTS:
            raise RuntimeError("OPENAI_REASONING_EFFORT must be none, low, medium, high, xhigh, or max")
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("Install the optional LLM dependency with: pip install -e '.[llm]'") from error
        timeout = float(os.environ.get("OPENAI_REQUEST_TIMEOUT", "180"))
        retries = int(os.environ.get("OPENAI_MAX_RETRIES", "3"))
        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=retries)

    def __call__(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            reasoning={"effort": self.reasoning_effort},
            input=(
                "Return only valid JSON matching the requested schema. "
                "Do not include Markdown fences.\n\n" + prompt
            ),
        )
        text = response.output_text
        if not text:
            raise RuntimeError("The model returned no text")
        return text

    def call_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        name: str = "structured_result",
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return schema-constrained JSON through the Responses API."""
        model_input: Any = prompt
        if images:
            content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
            content.extend({"type": "input_image", "image_url": image, "detail": "high"} for image in images)
            model_input = [{"role": "user", "content": content}]
        response = self.client.responses.create(
            model=self.model,
            reasoning={"effort": self.reasoning_effort},
            input=model_input,
            text={
                "format": {
                    "type": "json_schema",
                    "name": name,
                    "strict": True,
                    "schema": schema,
                }
            },
        )
        text = response.output_text
        if not text:
            raise RuntimeError("The model returned no structured output")
        return json.loads(text)

