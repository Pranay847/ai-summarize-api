"""Per-call cost estimate from token usage, using each provider's public price
list (USD per 1M tokens). Prices are easy to update in one place."""

from __future__ import annotations

# (input_per_million, output_per_million) USD. Public list prices; update freely.
PRICES = {
    ("groq", "llama-3.1-8b-instant"): (0.05, 0.08),
    ("gemini", "gemini-1.5-flash"): (0.075, 0.30),
    # Ollama runs locally => no per-token charge, but we still log token counts.
    ("ollama", "*"): (0.0, 0.0),
    ("mock", "*"): (0.0, 0.0),
}


def estimate_cost_usd(provider: str, model: str, prompt_tokens: int,
                      completion_tokens: int) -> float:
    rate = PRICES.get((provider, model)) or PRICES.get((provider, "*"))
    if rate is None:
        return 0.0  # unknown model => don't guess a price, just log tokens
    in_rate, out_rate = rate
    return (prompt_tokens / 1_000_000) * in_rate + (
        completion_tokens / 1_000_000
    ) * out_rate
