"""
model_benchmark.py — Compare OpenAI models for the Insurance Voice Agent use case.

Tests each model against realistic insurance intake conversation scenarios and
scores them on natural language quality, latency, and token cost.

Run:
    .venv\\Scripts\\python.exe scripts/model_benchmark.py
    .venv\\Scripts\\python.exe scripts/model_benchmark.py --models gpt-5.4-nano gpt-5.4-mini
    .venv\\Scripts\\python.exe scripts/model_benchmark.py --save results_2026-04.json

Results are printed as a comparison table and optionally saved to JSON.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

for _p in [Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"]:
    if _p.exists():
        load_dotenv(dotenv_path=_p, override=False)

sys.path.insert(0, str(Path(__file__).parent))
from openai_brain import MODEL_CONFIGS, OpenAIBrain  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,  # suppress INFO noise during benchmark
    format="%(levelname)s %(name)s — %(message)s",
)

# ── System prompt (matches voice_assistant.py → insurance_prompt.py) ──────────
SYSTEM_PROMPT = """
You are Alex, a professional auto insurance intake agent on a phone call.
You collect information to help prepare an insurance quote.

Personality:
- Calm, friendly, and professional
- Concise — this is a voice call, not a text form
- Empathetic and patient

Rules:
- Ask ONE question at a time only
- Keep every response to 1–3 short sentences
- Never use: "Certainly", "Absolutely", "I'd be happy to help",
  "Is there anything else I can assist", "Great question"
- Sound like a real insurance agent on the phone
- Never mention being an AI
""".strip()

# ── Benchmark scenarios (based on Excel flow: VoiceChat_QuestionSequenceAndProb.xlsx) ─────
# Each scenario is a short conversation followed by the final user message.
# The model must respond ONLY to the final user message.

SCENARIOS: list[dict] = [
    {
        "id": "consent_yes",
        "description": "Caller agrees to proceed — agent asks for WhatsApp number (step 1.1→1.2)",
        "history": [],
        "user": "Yes, that's fine.",
        "good_signals": ["phone", "whatsapp", "number", "best", "reach"],
        "bad_signals": ["certainly", "i'd be happy", "how may i assist"],
    },
    {
        "id": "consent_no",
        "description": "Caller declines consent — agent ends politely (step 1.1 No branch)",
        "history": [],
        "user": "No, I don't want to proceed.",
        "good_signals": ["note", "thank", "call back", "anytime", "day", "goodbye"],
        "bad_signals": ["certainly", "i'd be happy", "menu", "waffle"],
    },
    {
        "id": "phone_number",
        "description": "Caller provides WhatsApp number (step 1.2)",
        "history": [
            {
                "role": "assistant",
                "content": "What is the best phone number to reach you on WhatsApp?",
            },
        ],
        "user": "Sure, it's 416-555-0100.",
        "good_signals": ["policy", "date", "effective", "when", "insurance"],
        "bad_signals": ["certainly", "i'd be happy", "how may i assist"],
    },
    {
        "id": "driver_count",
        "description": "Caller states number of drivers in household (step 2.1)",
        "history": [
            {
                "role": "assistant",
                "content": "Now let's talk about the drivers. How many licensed drivers are in your household?",
            },
        ],
        "user": "There are two of us.",
        "good_signals": ["registered", "owner", "operator", "primary", "vehicle"],
        "bad_signals": ["certainly", "of course", "i'd be happy", "please let me know"],
    },
    {
        "id": "marital_status",
        "description": "Caller states marital status (step 2.3)",
        "history": [
            {
                "role": "assistant",
                "content": "Are you the registered owner and primary operator of the vehicle?",
            },
            {"role": "user", "content": "Yes, I am."},
            {"role": "assistant", "content": "What is your marital status?"},
        ],
        "user": "I'm married.",
        "good_signals": ["applicant", "policy", "will you", "driver", "next"],
        "bad_signals": ["certainly", "congratulations", "i'd be happy"],
    },
    {
        "id": "vehicle_ownership",
        "description": "Caller states vehicle ownership type (step 3.4)",
        "history": [
            {
                "role": "assistant",
                "content": "What is the ownership type — owned, financed, or leased?",
            },
        ],
        "user": "It's financed.",
        "good_signals": ["financing", "leasing", "company", "name", "lender"],
        "bad_signals": ["certainly", "i'd be happy", "please let me know"],
    },
    {
        "id": "hesitation",
        "description": "Caller is unsure about licence issue date (step 2.6)",
        "history": [
            {
                "role": "assistant",
                "content": "When was your licence first issued in Canada? I just need the month and year.",
            },
        ],
        "user": "Hmm, I'm not totally sure... maybe 2015?",
        "good_signals": [
            "approximate",
            "okay",
            "noted",
            "2015",
            "training",
            "certificate",
        ],
        "bad_signals": ["certainly", "i understand your concern", "i'd be happy"],
    },
    {
        "id": "off_topic",
        "description": "Caller asks unrelated question mid-intake",
        "history": [
            {
                "role": "assistant",
                "content": "What is the vehicle primarily used for — pleasure, commute, or business?",
            },
        ],
        "user": "What's the weather like today?",
        "good_signals": ["insurance", "question", "back", "vehicle", "used", "focus"],
        "bad_signals": ["certainly", "weather", "temperature", "i'd be happy to"],
    },
]


# ── Scoring helpers ───────────────────────────────────────────────────────────


def _score_reply(reply: str, scenario: dict) -> dict:
    """Return a quality score dict for a (reply, scenario) pair."""
    r = reply.lower()
    words = r.split()
    word_count = len(words)

    good_hits = sum(1 for sig in scenario.get("good_signals", []) if sig in r)
    bad_hits = sum(1 for sig in scenario.get("bad_signals", []) if sig in r)

    # 0–10 score
    length_ok = 5 <= word_count <= 40
    score = (
        (good_hits * 2)  # +2 per good signal
        - (bad_hits * 3)  # -3 per bad/robotic phrase
        + (2 if length_ok else 0)  # +2 if concise
    )
    score = max(0, min(10, score))

    return {
        "score": score,
        "word_count": word_count,
        "good_hits": good_hits,
        "bad_hits": bad_hits,
    }


# ── Benchmark runner ──────────────────────────────────────────────────────────


def run_benchmark(
    models: list[str], scenarios: list[dict] | None = None, repeat: int = 1
) -> list[dict]:
    """Run scenarios across all models.  Returns list of result dicts."""
    active_scenarios = scenarios if scenarios is not None else SCENARIOS
    results = []

    for model_id in models:
        cfg = MODEL_CONFIGS.get(model_id, {})
        print(f"\n{'=' * 65}")
        print(f"  MODEL: {model_id}  ({cfg.get('note', 'no description')})")
        print(f"{'=' * 65}")

        # Temporarily override LLM_MODEL so OpenAIBrain picks it up
        os.environ["LLM_MODEL"] = model_id
        try:
            brain = OpenAIBrain()
        except Exception as exc:
            print(f"  [SKIP] Could not init brain for {model_id}: {exc}")
            continue

        model_totals = {
            "latencies": [],
            "prompt_tokens": [],
            "completion_tokens": [],
            "costs_usd": [],
            "scores": [],
        }

        for scenario in active_scenarios:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages += scenario.get("history", [])
            messages.append({"role": "user", "content": scenario["user"]})

            run_latencies, run_scores = [], []
            run_ptok, run_ctok, run_cost = [], [], []

            for _ in range(repeat):
                try:
                    reply, usage = brain.chat(
                        messages,  # type: ignore[arg-type]
                        temperature=0.5,
                        max_tokens=80,
                    )
                except Exception as exc:
                    print(f"  [ERR] {scenario['id']}: {exc}")
                    run_latencies.append(None)
                    continue

                quality = _score_reply(reply, scenario)
                run_latencies.append(usage["latency_s"])
                run_scores.append(quality["score"])
                run_ptok.append(usage["prompt_tokens"])
                run_ctok.append(usage["completion_tokens"])
                run_cost.append(usage["cost_usd"])

                print(
                    f"  [{scenario['id']:15s}] {usage['latency_s']:.2f}s | "
                    f"tokens={usage['prompt_tokens']}/{usage['completion_tokens']} | "
                    f"score={quality['score']}/10 | "
                    f"reply: {reply[:70]!r}"
                )

            if run_latencies and any(v is not None for v in run_latencies):
                valid_lat = [v for v in run_latencies if v is not None]
                avg_lat = sum(valid_lat) / len(valid_lat)
                avg_sc = sum(run_scores) / len(run_scores) if run_scores else 0
                avg_ptok = sum(run_ptok) / len(run_ptok) if run_ptok else 0
                avg_ctok = sum(run_ctok) / len(run_ctok) if run_ctok else 0
                avg_cost = sum(run_cost) / len(run_cost) if run_cost else 0

                model_totals["latencies"].append(avg_lat)
                model_totals["scores"].append(avg_sc)
                model_totals["prompt_tokens"].append(avg_ptok)
                model_totals["completion_tokens"].append(avg_ctok)
                model_totals["costs_usd"].append(avg_cost)

                results.append(
                    {
                        "model": model_id,
                        "scenario": scenario["id"],
                        "description": scenario["description"],
                        "latency_s": round(avg_lat, 3),
                        "prompt_tokens": round(avg_ptok, 1),
                        "completion_tokens": round(avg_ctok, 1),
                        "cost_usd": round(avg_cost, 7),
                        "quality_score": round(avg_sc, 2),
                    }
                )

        # Per-model summary
        if model_totals["latencies"]:
            n = len(model_totals["latencies"])
            print(f"\n  --- {model_id} SUMMARY ({n} scenarios) ---")
            avg = lambda lst: sum(lst) / len(lst) if lst else 0
            total_cost = sum(model_totals["costs_usd"])
            print(f"  Avg latency:        {avg(model_totals['latencies']):.3f}s")
            print(f"  Avg quality score:  {avg(model_totals['scores']):.2f}/10")
            print(f"  Avg prompt tokens:  {avg(model_totals['prompt_tokens']):.0f}")
            print(f"  Avg completion tok: {avg(model_totals['completion_tokens']):.0f}")
            print(
                f"  Total cost (bench): ${total_cost:.6f}  (INR {total_cost * float(os.getenv('USD_TO_INR', '84')):.4f})"
            )

    return results


# ── Comparison table ──────────────────────────────────────────────────────────


def print_comparison(results: list[dict]) -> None:
    """Print a summary comparison table grouped by model."""
    from collections import defaultdict

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    col_w = [20, 8, 8, 10, 8, 10]
    headers = ["Model", "Lat(s)", "Score", "P-tok", "C-tok", "Cost($)"]
    sep = "  ".join("-" * w for w in col_w)

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPARISON SUMMARY")
    print("=" * 70)
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_w)))
    print("  " + sep)

    ranked = []
    for model_id, rows in by_model.items():
        avg_lat = sum(r["latency_s"] for r in rows) / len(rows)
        avg_score = sum(r["quality_score"] for r in rows) / len(rows)
        avg_ptok = sum(r["prompt_tokens"] for r in rows) / len(rows)
        avg_ctok = sum(r["completion_tokens"] for r in rows) / len(rows)
        total_cost = sum(r["cost_usd"] for r in rows)

        cfg_note = MODEL_CONFIGS.get(model_id, {}).get("note", "")
        ranked.append(
            (model_id, avg_lat, avg_score, avg_ptok, avg_ctok, total_cost, cfg_note)
        )
        print(
            "  "
            + model_id.ljust(col_w[0])
            + "  "
            + f"{avg_lat:.3f}".ljust(col_w[1])
            + "  "
            + f"{avg_score:.2f}".ljust(col_w[2])
            + "  "
            + f"{avg_ptok:.0f}".ljust(col_w[3])
            + "  "
            + f"{avg_ctok:.0f}".ljust(col_w[4])
            + "  "
            + f"{total_cost:.6f}".ljust(col_w[5])
        )

    # Recommendation: best composite = score_rank + latency_rank - cost_rank
    print("\n" + "=" * 70)
    print("  RECOMMENDATION")
    print("=" * 70)

    # Sort by composite: higher score better, lower latency better, lower cost better
    if ranked:
        scored = []
        for m, lat, sc, ptok, ctok, cost, note in ranked:
            composite = sc * 2 - lat * 1.5 - cost * 500
            scored.append((composite, m, lat, sc, cost, note))
        scored.sort(key=lambda x: -x[0])

        best = scored[0]
        print(f"\n  Best overall for production:  {best[1]}")
        print(
            f"  Quality score: {best[3]:.2f}/10  |  Latency: {best[2]:.3f}s  |  Bench cost: ${best[4]:.6f}"
        )
        print(f"  Note: {best[5]}")

        print("\n  Full ranking:")
        for rank, (comp, m, lat, sc, cost, note) in enumerate(scored, 1):
            print(
                f"  {rank}. {m:<22} composite={comp:.2f}  score={sc:.2f}  lat={lat:.2f}s  cost=${cost:.6f}"
            )


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark OpenAI models for Crumbs & Cream"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        help="Model IDs to benchmark (default: all from openai_brain.MODEL_CONFIGS)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat each scenario N times and average results (default: 1)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional path to save raw JSON results (e.g. bench_results.json)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Only run the first 3 scenarios (quick mode)",
    )
    args = parser.parse_args()

    scenarios_to_run = SCENARIOS[:3] if args.fast else SCENARIOS

    print(
        f"\nRunning benchmark: {len(scenarios_to_run)} scenarios \u00d7 {len(args.models)} models \u00d7 {args.repeat} repeat(s)"
    )
    print(f"Models: {args.models}\n")

    t_start = time.perf_counter()
    results = run_benchmark(args.models, scenarios=scenarios_to_run, repeat=args.repeat)
    elapsed = time.perf_counter() - t_start

    print_comparison(results)

    print(f"\n  Total benchmark time: {elapsed:.1f}s")

    if args.save:
        out_path = Path(args.save)
        out_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  Results saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
