"""The provider seam.

This is the ONLY file that knows which LLM vendor/model we use. Switching
Groq <-> Gemini <-> Ollama is one env var (AI_PROVIDER). Everything else in the
app calls `get_provider().chat(messages)` and gets back a normalized
`ProviderResponse`, so no other file mentions a vendor.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import requests


class RetryableError(Exception):
    """429 / 5xx / network timeout — safe to retry."""


class NonRetryableError(Exception):
    """4xx that is the caller's fault (e.g. 400) — never retry."""


@dataclass
class ProviderResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str


def _raise_for_status(status: int, body: str) -> None:
    if status == 429 or 500 <= status < 600:
        raise RetryableError(f"provider returned {status}: {body[:200]}")
    if 400 <= status < 500:
        raise NonRetryableError(f"provider returned {status}: {body[:200]}")


class Provider:
    name = "base"

    def chat(self, messages: list[dict], *, timeout: float) -> ProviderResponse:
        raise NotImplementedError


class GroqProvider(Provider):
    name = "groq"
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self) -> None:
        self.model = os.environ.get("AI_MODEL", "llama-3.1-8b-instant")
        self.key = os.environ["GROQ_API_KEY"]  # from .env, never hardcoded

    def chat(self, messages, *, timeout):
        try:
            r = requests.post(
                self.URL,
                headers={"Authorization": f"Bearer {self.key}"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"},
                },
                timeout=timeout,
            )
        except requests.Timeout as e:
            raise RetryableError(f"timeout after {timeout}s") from e
        if r.status_code != 200:
            _raise_for_status(r.status_code, r.text)
        data = r.json()
        usage = data.get("usage", {})
        return ProviderResponse(
            text=data["choices"][0]["message"]["content"],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            model=self.model,
        )


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self) -> None:
        self.model = os.environ.get("AI_MODEL", "gemini-1.5-flash")
        self.key = os.environ["GEMINI_API_KEY"]

    def chat(self, messages, *, timeout):
        # Gemini has one system + user; fold messages into its shape.
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n".join(m["content"] for m in messages if m["role"] == "user")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.key}"
        )
        try:
            r = requests.post(
                url,
                json={
                    "system_instruction": {"parts": [{"text": system}]},
                    "contents": [{"parts": [{"text": user}]}],
                    "generationConfig": {"response_mime_type": "application/json"},
                },
                timeout=timeout,
            )
        except requests.Timeout as e:
            raise RetryableError(f"timeout after {timeout}s") from e
        if r.status_code != 200:
            _raise_for_status(r.status_code, r.text)
        data = r.json()
        usage = data.get("usageMetadata", {})
        return ProviderResponse(
            text=data["candidates"][0]["content"]["parts"][0]["text"],
            prompt_tokens=usage.get("promptTokenCount", 0),
            completion_tokens=usage.get("candidatesTokenCount", 0),
            model=self.model,
        )


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self) -> None:
        self.model = os.environ.get("AI_MODEL", "llama3.1")
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def chat(self, messages, *, timeout):
        try:
            r = requests.post(
                f"{self.host}/api/chat",
                json={"model": self.model, "messages": messages,
                      "format": "json", "stream": False},
                timeout=timeout,
            )
        except requests.Timeout as e:
            raise RetryableError(f"timeout after {timeout}s") from e
        if r.status_code != 200:
            _raise_for_status(r.status_code, r.text)
        data = r.json()
        return ProviderResponse(
            text=data["message"]["content"],
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            model=self.model,
        )


class MockProvider(Provider):
    """Scriptable provider for tests and offline demos — no key, no network.

    Feed it a list of behaviours; each call pops the next one. A behaviour is
    either a status int (to raise the matching error) or a response string.
    """

    name = "mock"

    def __init__(self, script=None):
        self.model = "mock-1"
        self.script = list(script or [])
        self.calls = 0

    def chat(self, messages, *, timeout):
        self.calls += 1
        if not self.script:
            behaviour = '{"bullets": ["a", "b", "c"]}'
        else:
            behaviour = self.script.pop(0)
        if isinstance(behaviour, int):
            _raise_for_status(behaviour, "mock error")
        return ProviderResponse(
            text=behaviour, prompt_tokens=42, completion_tokens=18, model=self.model
        )


_PROVIDERS = {
    "groq": GroqProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
}


def get_provider() -> Provider:
    name = os.environ.get("AI_PROVIDER", "groq").lower()
    if name == "mock":
        return MockProvider()
    if name not in _PROVIDERS:
        raise NonRetryableError(f"unknown AI_PROVIDER: {name}")
    return _PROVIDERS[name]()
