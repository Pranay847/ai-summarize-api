"""Prove every requirement WITHOUT a real API key, using MockProvider.

Each test scripts the provider's behaviour and asserts the wrapper does the
right thing: schema-retry, transport-retry on 429/5xx, no-retry on 400, and
cost logging.
"""

import logging

import pytest

from app import ai_client
from app.providers import MockProvider, NonRetryableError, RetryableError
from app.pricing import estimate_cost_usd
from app.schemas import Summary


def use_mock(monkeypatch, script):
    mock = MockProvider(script=script)
    monkeypatch.setattr(ai_client, "get_provider", lambda: mock)
    monkeypatch.setattr(ai_client, "BACKOFF_BASE_S", 0)  # no real sleeping
    return mock


def test_happy_path_returns_validated_summary(monkeypatch):
    mock = use_mock(monkeypatch, ['{"bullets": ["one", "two", "three"]}'])
    result = ai_client.complete_summary("some text")
    assert isinstance(result.summary, Summary)
    assert result.summary.bullets == ["one", "two", "three"]
    assert mock.calls == 1


def test_malformed_output_retries_once_then_succeeds(monkeypatch):
    # first reply is junk, second is valid -> wrapper retries once and wins
    mock = use_mock(monkeypatch, [
        "not json at all",
        '{"bullets": ["a", "b", "c"]}',
    ])
    result = ai_client.complete_summary("text")
    assert result.summary.bullets == ["a", "b", "c"]
    assert mock.calls == 2


def test_persistently_malformed_returns_clean_error_not_crash(monkeypatch):
    mock = use_mock(monkeypatch, ["nope", "still nope"])
    with pytest.raises(ValueError):          # clean error, not an unhandled crash
        ai_client.complete_summary("text")
    assert mock.calls == 2                   # tried once, retried once, gave up


def test_wrong_shape_json_is_rejected(monkeypatch):
    # valid JSON but only 2 bullets -> schema catches it, retry, then succeed
    mock = use_mock(monkeypatch, [
        '{"bullets": ["only", "two"]}',
        '{"bullets": ["a", "b", "c"]}',
    ])
    result = ai_client.complete_summary("text")
    assert len(result.summary.bullets) == 3


def test_429_is_retried_then_succeeds(monkeypatch):
    mock = use_mock(monkeypatch, [429, 429, '{"bullets": ["a", "b", "c"]}'])
    result = ai_client.complete_summary("text")
    assert result.summary.bullets == ["a", "b", "c"]
    assert mock.calls == 3


def test_500_is_retried(monkeypatch):
    mock = use_mock(monkeypatch, [503, '{"bullets": ["a", "b", "c"]}'])
    result = ai_client.complete_summary("text")
    assert mock.calls == 2


def test_400_is_never_retried(monkeypatch):
    mock = use_mock(monkeypatch, [400, '{"bullets": ["a", "b", "c"]}'])
    with pytest.raises(NonRetryableError):
        ai_client.complete_summary("text")
    assert mock.calls == 1                   # gave up immediately, did NOT retry


def test_cost_and_tokens_are_logged(monkeypatch, caplog):
    use_mock(monkeypatch, ['{"bullets": ["a", "b", "c"]}'])
    with caplog.at_level(logging.INFO, logger="ai"):
        result = ai_client.complete_summary("text", feature="summarize")
    log = "\n".join(r.getMessage() for r in caplog.records)
    assert "feature=summarize" in log
    assert "prompt_tokens=42" in log
    assert "completion_tokens=18" in log
    assert "est_cost_usd=" in log
    # tokens surfaced on the result too
    assert result.prompt_tokens == 42
    assert result.completion_tokens == 18


def test_pricing_math():
    # 1,000,000 in-tokens at $0.05/M + 1,000,000 out at $0.08/M = $0.13
    cost = estimate_cost_usd("groq", "llama-3.1-8b-instant", 1_000_000, 1_000_000)
    assert round(cost, 4) == 0.13


def test_empty_text_rejected(monkeypatch):
    use_mock(monkeypatch, [])
    with pytest.raises(ValueError):
        ai_client.complete_summary("   ")


def test_pricing_wildcard_and_unknown_fallbacks():
    # any ollama model falls back to the ("ollama", "*") free rate
    assert estimate_cost_usd("ollama", "llama3", 500_000, 500_000) == 0.0
    # unknown model of a known paid provider => no guessed price
    assert estimate_cost_usd("groq", "some-new-model", 1_000_000, 1_000_000) == 0.0
    # unknown provider entirely => 0.0, not a KeyError
    assert estimate_cost_usd("nope", "x", 1_000_000, 1_000_000) == 0.0


def test_summary_strips_whitespace_and_rejects_blank_bullets():
    s = Summary(bullets=["  one ", "two\n", "\tthree"])
    assert s.bullets == ["one", "two", "three"]
    with pytest.raises(ValueError):
        Summary(bullets=["one", "   ", "three"])
