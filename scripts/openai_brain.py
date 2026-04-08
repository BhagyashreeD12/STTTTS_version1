"""
openai_brain.py — Modular OpenAI LLM brain for Crumbs & Cream Caffè voice assistant.

Model is controlled by the LLM_MODEL env var (or .env file).
Pricing is looked up from MODEL_CONFIGS; edit as OpenAI updates its pricing page.

Usage:
    from openai_brain import get_brain
    reply, usage = get_brain().chat(messages)
"""

import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError
from openai.types.chat import ChatCompletionMessageParam

# ── .env loading ─────────────────────────────────────────────────────────────
# Try scripts/.env first, then repo root .env
for _env_candidate in [
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / ".env",
]:
    if _env_candidate.exists():
        load_dotenv(dotenv_path=_env_candidate, override=False)

logger = logging.getLogger(__name__)

# ── Model pricing registry ────────────────────────────────────────────────────
# USD per 1 million tokens.  Verify current rates at: https://openai.com/api/pricing
# Last checked: 2026-04
MODEL_CONFIGS: dict[str, dict] = {
    # ── GPT-5.4 family ────────────────────────────────────────
    "gpt-5.4-nano": {
        "input_per_1m":   0.30,
        "output_per_1m":  1.20,
        "tier": "nano",
        "note": "Fastest & cheapest — great for simple intents",
    },
    "gpt-5.4-mini": {
        "input_per_1m":   0.75,
        "output_per_1m":  4.50,
        "tier": "mini",
        "note": "Best price/quality balance for café Q&A (current default)",
    },
    "gpt-5.4-pro": {
        "input_per_1m":   7.50,
        "output_per_1m": 30.00,
        "tier": "pro",
        "note": "Premium — overkill for menu ordering, but flawless NLU",
    },
    # ── GPT-4.1 family (fallback / comparison) ─────────────────
    "gpt-4.1-nano": {
        "input_per_1m":   0.10,
        "output_per_1m":  0.40,
        "tier": "nano",
        "note": "Ultra-cheap older gen, good baseline",
    },
    "gpt-4.1-mini": {
        "input_per_1m":   0.40,
        "output_per_1m":  1.60,
        "tier": "mini",
        "note": "Solid older-gen budget model",
    },
    "gpt-4o-mini": {
        "input_per_1m":   0.15,
        "output_per_1m":  0.60,
        "tier": "mini",
        "note": "Proven, very low cost — good fallback option",
    },
    "gpt-4o": {
        "input_per_1m":   2.50,
        "output_per_1m": 10.00,
        "tier": "full",
        "note": "High quality, higher cost",
    },
}

DEFAULT_MODEL = "gpt-5.4-mini"
INR_ENV_KEY = "USD_TO_INR"


def get_model_name() -> str:
    """Read the active model from env, falling back to DEFAULT_MODEL."""
    return os.getenv("LLM_MODEL", DEFAULT_MODEL).strip()


def list_models() -> list[str]:
    """Return all model IDs in the pricing registry."""
    return list(MODEL_CONFIGS.keys())


# ── Brain class ───────────────────────────────────────────────────────────────

class OpenAIBrain:
    """Thin wrapper around the OpenAI chat completions endpoint.

    - Loads API key from OPENAI_API_KEY env var (never accepts it as a parameter).
    - Reads the active model from LLM_MODEL env var at instantiation.
    - Logs token usage, latency, and estimated cost after every call.
    """

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Add it to scripts/.env or the repo root .env file."
            )

        self._client = OpenAI(api_key=api_key)
        self.model = get_model_name()

        cfg = MODEL_CONFIGS.get(self.model)
        if cfg is None:
            logger.warning(
                "[Brain] '%s' not in MODEL_CONFIGS; cost tracking disabled.", self.model
            )
            cfg = {"input_per_1m": 0.0, "output_per_1m": 0.0, "tier": "unknown", "note": ""}

        self._input_rate  = cfg["input_per_1m"]
        self._output_rate = cfg["output_per_1m"]
        self._usd_to_inr  = float(os.getenv(INR_ENV_KEY, "84"))

        logger.info("[Brain] Initialized — model=%s  (%s)", self.model, cfg["note"])

    # ── Public API ─────────────────────────────────────────────────────────

    @staticmethod
    def _token_limit_kwarg(model: str, limit: int) -> dict:
        """GPT-5.x requires max_completion_tokens; older models use max_tokens."""
        if re.match(r"gpt-5", model):
            return {"max_completion_tokens": limit}
        return {"max_tokens": limit}

    def chat(
        self,
        messages: list[ChatCompletionMessageParam],
        temperature: float = 0.5,
        max_tokens: int = 80,
    ) -> tuple[str, dict]:
        """Send messages to the OpenAI API and return (reply_text, usage_info).

        usage_info keys:
            model, prompt_tokens, completion_tokens, total_tokens,
            cost_usd, cost_inr, latency_s
        """
        t0 = time.perf_counter()

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                **self._token_limit_kwarg(self.model, max_tokens),
            )
        except AuthenticationError as exc:
            logger.error("[Brain] Authentication failed — check OPENAI_API_KEY: %s", exc)
            raise
        except RateLimitError as exc:
            logger.warning("[Brain] Rate limit hit: %s", exc)
            raise
        except APIConnectionError as exc:
            logger.error("[Brain] Connection error: %s", exc)
            raise

        latency = time.perf_counter() - t0
        usage   = resp.usage

        cost_usd = (
            (usage.prompt_tokens / 1_000_000) * self._input_rate
            + (usage.completion_tokens / 1_000_000) * self._output_rate
        )
        cost_inr = cost_usd * self._usd_to_inr

        usage_info = {
            "model":             self.model,
            "prompt_tokens":     usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens":      usage.total_tokens,
            "cost_usd":          cost_usd,
            "cost_inr":          cost_inr,
            "latency_s":         round(latency, 3),
        }

        logger.info(
            "[LLM] %-18s | latency=%.2fs | tokens=%d/%d | cost=$%.6f (INR %.4f)",
            self.model,
            latency,
            usage.prompt_tokens,
            usage.completion_tokens,
            cost_usd,
            cost_inr,
        )

        # Console-friendly block (mirrors old inline print, visible when DEBUG=True)
        print(
            f"\n--- OPENAI USAGE ---\n"
            f"Model:             {self.model}\n"
            f"Prompt tokens:     {usage.prompt_tokens}\n"
            f"Completion tokens: {usage.completion_tokens}\n"
            f"Total tokens:      {usage.total_tokens}\n"
            f"Latency:           {latency:.2f}s\n"
            f"Cost USD:          ${cost_usd:.6f}\n"
            f"Cost INR:          INR {cost_inr:.4f}\n"
            f"--------------------"
        )

        reply = (resp.choices[0].message.content or "").strip()
        return reply, usage_info


# ── Module-level singleton ────────────────────────────────────────────────────

_brain: Optional[OpenAIBrain] = None


def get_brain() -> OpenAIBrain:
    """Return the shared OpenAIBrain instance, creating it on first call."""
    global _brain
    if _brain is None:
        _brain = OpenAIBrain()
    return _brain


def reset_brain() -> None:
    """Force recreation of the brain on the next get_brain() call.

    Useful in tests or when LLM_MODEL env var changes at runtime.
    """
    global _brain
    _brain = None
