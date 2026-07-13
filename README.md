# Connect to an AI API (FlyRank BE, Week 4)

A backend `POST /summarize` feature that calls a free LLM, gets **schema-valid
JSON** back (exactly three bullets), handles failures without crashing, and
**logs tokens + estimated cost** on every call. The point isn't "call an API" —
it's the wrapper around it.

## The wrapper (what this assignment is really about)

```
app/
  providers.py   # THE SEAM: the only file that names a vendor.
                 #   Groq / Gemini / Ollama / Mock adapters + get_provider()
  ai_client.py   # complete_summary(): timeout, retries, schema-validate, cost log
  schemas.py     # Pydantic Summary: exactly 3 non-empty bullets
  pricing.py     # per-provider price list -> per-call cost estimate
  prompts.py     # templated system + user prompt
  main.py        # Flask POST /summarize (thin; never sees a vendor)
tests/test_wrapper.py   # proves every behaviour with a scripted MockProvider
run_live.py      # fire ONE real call using your own key
```

Requirement → where it lives:

| Requirement | Where |
|---|---|
| API key in `.env`, documented in `.env.example`, never in code/logs | `providers.py` reads `os.environ`; `.gitignore` excludes `.env` |
| Provider seam — switch Groq↔Gemini↔Ollama by touching one file | `providers.py` + `AI_PROVIDER` env var |
| Schema-validated JSON; malformed handled, not crashed on | `schemas.py` + retry-once loop in `ai_client.py` |
| Timeout on every call; retry 429/5xx, never 400 | `ai_client._call_with_transport_retries` |
| Log tokens + estimated cost, tagged with the feature | `ai_client._log_cost` + `pricing.py` |

## Run it

### Offline, no key (mock provider)

```bash
pip install -r requirements.txt
AI_PROVIDER=mock python -m flask --app app.main run --port 8000
curl -s localhost:8000/summarize -H 'content-type: application/json' \
  -d '{"text":"long text here"}'
```

### For real (free Groq key, no card)

```bash
cp .env.example .env         # set AI_PROVIDER=groq and GROQ_API_KEY=...
python run_live.py "Paste any paragraph to summarize."
```

Switching to Gemini or a local Ollama is one line in `.env`
(`AI_PROVIDER=gemini` / `ollama`) — no code changes. That's the seam.

## Reliability, precisely

- **Timeout** on every provider call (15s), raised as a retryable error.
- **Transport retries**: 429 and 5xx are retried up to 3× with exponential
  backoff (0.5s, 1s, 2s). A **400 is never retried** — retrying a bad request
  just wastes calls, so it surfaces immediately.
- **Schema retries**: if the model returns non-JSON or the wrong shape, the
  wrapper appends a corrective instruction and retries **once**. If it's still
  invalid, the caller gets a clean `422`, never a stack trace.

## Cost logging

Every call logs one line with the feature name, token counts read from the
provider response, and a cost estimate from `pricing.py`'s public price list:

```
INFO ai feature=summarize provider=groq model=llama-3.1-8b-instant \
     prompt_tokens=180 completion_tokens=44 est_cost_usd=0.000013
```

Ollama runs locally (cost `$0.00`) but token counts are still logged — the
habit is the point.

## How I verified it — honest note

I proved **every requirement** with a scripted `MockProvider`, so the wrapper
logic is checked without needing a key or network. `tests/test_wrapper.py`
(10 tests, all passing) covers:

- happy path returns a validated 3-bullet summary;
- malformed output → retries once → succeeds;
- persistently malformed → clean `ValueError`, **not a crash**, after exactly
  one retry;
- wrong-shape JSON (2 bullets) is rejected by the schema;
- 429 and 5xx are retried; **400 is not** (gives up on the first call);
- tokens + `est_cost_usd` are logged with `feature=summarize`;
- the pricing math (`$0.05/M in + $0.08/M out`).

```
$ PYTHONPATH=. pytest tests/test_wrapper.py -q
10 passed
```

**What I did not run:** a live call against Groq/Gemini — my dev sandbox has no
outbound access to those APIs and I don't hold a key. The real adapters are
implemented and ready; `python run_live.py "..."` makes that one call once a key
is in `.env`, and prints the real bullets, real token usage, and real cost. The
mock uses the identical `ProviderResponse` shape the real adapters return, so
the pipeline the tests exercise is the same one a live call flows through.
