"""Fire ONE real call at whatever provider AI_PROVIDER points to.

    cp .env.example .env      # set AI_PROVIDER=groq and GROQ_API_KEY
    python run_live.py "Paste some text to summarize here."

Prints the validated bullets plus the real token usage and estimated cost.
Reads the key from .env — nothing is hardcoded and nothing is printed.
"""

import sys

from dotenv import load_dotenv

load_dotenv()

from app.ai_client import complete_summary  # noqa: E402  (after load_dotenv)

text = " ".join(sys.argv[1:]) or (
    "The FlyRank internship is a self-paced, remote program where you build a "
    "real backend AI feature: call a free LLM, validate its JSON, retry on "
    "failure, and log what each call costs."
)

result = complete_summary(text)
print("bullets:")
for b in result.summary.bullets:
    print("  -", b)
print(
    f"\nprovider={result.provider} model={result.model} "
    f"prompt_tokens={result.prompt_tokens} "
    f"completion_tokens={result.completion_tokens} "
    f"est_cost_usd={result.cost_usd:.6f}"
)
