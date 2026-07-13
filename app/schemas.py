"""The schema the model's output MUST satisfy. If it doesn't, we retry once,
then fail cleanly — we never trust raw model text."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Summary(BaseModel):
    bullets: list[str] = Field(..., min_length=3, max_length=3)

    @field_validator("bullets")
    @classmethod
    def no_empty_bullets(cls, v: list[str]) -> list[str]:
        cleaned = [b.strip() for b in v]
        if any(not b for b in cleaned):
            raise ValueError("bullets must be non-empty")
        return cleaned
