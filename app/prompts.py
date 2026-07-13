"""Templated prompt (system + user parts) for the summarize feature."""

SYSTEM = (
    "You summarize text into EXACTLY three short bullet points. "
    "Reply ONLY with JSON of the form {\"bullets\": [\"...\", \"...\", \"...\"]}. "
    "No prose, no markdown, no extra keys."
)


def build_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Summarize this into 3 bullets:\n\n{text}"},
    ]
