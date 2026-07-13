"""Flask app exposing the AI feature. The route is thin: it calls the wrapper
and turns wrapper errors into clean HTTP responses. It never sees a vendor."""

from __future__ import annotations

import logging

from flask import Flask, jsonify, request

from .ai_client import complete_summary
from .providers import NonRetryableError, RetryableError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.post("/summarize")
    def summarize():
        body = request.get_json(silent=True) or {}
        text = body.get("text", "")
        try:
            result = complete_summary(text)
        except ValueError as e:
            # bad input or model gave un-parseable output after a retry
            return jsonify(error=str(e)), 422
        except NonRetryableError as e:
            return jsonify(error=f"provider rejected request: {e}"), 400
        except RetryableError as e:
            return jsonify(error=f"provider unavailable: {e}"), 503

        return jsonify(
            bullets=result.summary.bullets,
            usage={
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "est_cost_usd": round(result.cost_usd, 6),
                "provider": result.provider,
                "model": result.model,
            },
        )

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
