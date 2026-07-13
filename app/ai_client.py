"""The wrapper — the actual skill of this assignment.

`complete_summary(text)` is what the rest of the app calls. It:
  * enforces a timeout on every provider call,
  * retries 429/5xx with short exponential backoff, but NEVER a 400,
  * asks for JSON and validates it against the Summary schema; on malformed
    output it retries ONCE, then returns a clean error instead of crashing,
  * logs tokens + estimated cost per call, tagged with the feature name.

No vendor names appear here — only the provider seam (providers.get_provider).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from pydantic import ValidationError

from .pricing import estimate_cost_usd
from .prompts import build_messages
from .providers import (NonRetryableError, Provider, ProviderResponse,
                        RetryableError, get_provider)
from .schemas import Summary

logger = logging.getLogger("ai")

TIMEOUT_S = 15.0
MAX_TRANSPORT_RETRIES = 3   # for 429/5xx
BACKOFF_BASE_S = 0.5


@dataclass
class Result:
    summary: Summary
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    model: str
    provider: str


def _call_with_transport_retries(provider: Provider, messages) -> ProviderResponse:
    """Handle 429/5xx with backoff; surface 400 immediately."""
    last: Exception | None = None
    for attempt in range(MAX_TRANSPORT_RETRIES):
        try:
            return provider.chat(messages, timeout=TIMEOUT_S)
        except NonRetryableError:
            raise  # 400 etc. — retrying would be pointless and wrong
        except RetryableError as e:
            last = e
            sleep = BACKOFF_BASE_S * (2 ** attempt)
            logger.warning("retryable error (attempt %d): %s; backing off %.1fs",
                           attempt + 1, e, sleep)
            time.sleep(sleep)
    raise RetryableError(f"gave up after {MAX_TRANSPORT_RETRIES} retries: {last}")


def _log_cost(feature: str, provider: str, resp: ProviderResponse,
              cost: float) -> None:
    logger.info(
        "feature=%s provider=%s model=%s prompt_tokens=%d completion_tokens=%d "
        "est_cost_usd=%.6f",
        feature, provider, resp.model, resp.prompt_tokens,
        resp.completion_tokens, cost,
    )


def complete_summary(text: str, feature: str = "summarize") -> Result:
    if not text or not text.strip():
        raise ValueError("text is required")

    provider = get_provider()
    messages = build_messages(text)

    # Up to 2 attempts to get SCHEMA-VALID output (retry once on malformed).
    last_validation_err: Exception | None = None
    for schema_attempt in range(2):
        resp = _call_with_transport_retries(provider, messages)

        cost = estimate_cost_usd(
            provider.name, resp.model, resp.prompt_tokens, resp.completion_tokens
        )
        _log_cost(feature, provider.name, resp, cost)  # log EVERY call's cost

        try:
            data = json.loads(resp.text)
            summary = Summary.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_validation_err = e
            logger.warning("malformed model output (schema attempt %d): %s",
                           schema_attempt + 1, e)
            messages = messages + [{
                "role": "user",
                "content": "Your last reply was not valid. Reply with ONLY "
                           '{"bullets": ["...", "...", "..."]} and nothing else.',
            }]
            continue

        return Result(
            summary=summary,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            cost_usd=cost,
            model=resp.model,
            provider=provider.name,
        )

    raise ValueError(f"model did not return valid JSON: {last_validation_err}")
