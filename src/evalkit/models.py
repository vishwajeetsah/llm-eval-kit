"""Model clients.

A model client wraps "give me a prompt, get back text" behind a tiny
interface so the runner doesn't care whether you're calling a real API or
a deterministic mock (great for testing the harness itself).
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field


@dataclass
class ModelResponse:
    """A single model generation."""

    text: str
    latency_s: float
    usage: dict = field(default_factory=dict)


class BaseModel:
    """Interface every model client must implement."""

    name: str = "base"

    def generate(self, prompt: str) -> ModelResponse:
        raise NotImplementedError


class MockModel(BaseModel):
    """Deterministic mock model — no API key, no network, no cost.

    ``mode="echo"`` returns the prompt back; ``mode="oracle"`` returns the
    reference answer for a fixed lookup table (perfect scores, useful for
    verifying the harness); any other mode returns a canned wrong answer.
    """

    name = "mock"

    def __init__(self, mode: str = "echo", answers: dict[str, str] | None = None):
        self.mode = mode
        self.answers = answers or {}

    def generate(self, prompt: str) -> ModelResponse:
        start = time.perf_counter()
        if self.mode == "oracle":
            text = self.answers.get(prompt, "I don't know.")
        elif self.mode == "echo":
            text = prompt
        else:
            text = "This is a mock response."
        return ModelResponse(text=text, latency_s=time.perf_counter() - start)


class OpenAICompatibleModel(BaseModel):
    """Client for any OpenAI-compatible chat-completions endpoint.

    Works with OpenAI, Azure OpenAI, Ollama, vLLM, LM Studio, Together, etc.
    Just point ``base_url`` at the server and set ``model`` accordingly.
    Only the standard library is used — no extra dependencies.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: int = 120,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.name = f"openai-compatible:{model}"

    def generate(self, prompt: str) -> ModelResponse:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"model request failed: {exc}") from exc
        latency = time.perf_counter() - start
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected API response shape: {payload!r}") from exc
        return ModelResponse(
            text=text or "",
            latency_s=latency,
            usage=payload.get("usage", {}),
        )
