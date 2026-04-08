"""
verify_api_key.py — Quick sanity-check for the OpenAI API key.

Run before launching the voice assistant:
    .venv\\Scripts\\python.exe scripts/verify_api_key.py

Checks:
  1. OPENAI_API_KEY is present in the environment / .env
  2. The key is accepted by the OpenAI API
  3. Required models (from openai_brain.MODEL_CONFIGS) are accessible
  4. Does a tiny test chat completion to confirm billing is enabled
"""

import os
import sys
import time
from pathlib import Path

# ── .env loading (same priority as the rest of the app) ──────────────────────
from dotenv import load_dotenv
for _p in [Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"]:
    if _p.exists():
        load_dotenv(dotenv_path=_p, override=False)
        print(f"[.env] Loaded from {_p}")

# ── 1. Key presence check ─────────────────────────────────────────────────────
key = os.getenv("OPENAI_API_KEY", "").strip()
if not key:
    print("[FAIL] OPENAI_API_KEY is not set in .env.")
    print("       Add a line like:  OPENAI_API_KEY=sk-proj-...")
    sys.exit(1)

masked = f"{key[:12]}...{key[-4:]}"
print(f"[OK]   Key found: {masked}  ({len(key)} chars)")

# ── 2. Import & initialise client ────────────────────────────────────────────
try:
    from openai import OpenAI, AuthenticationError
except ImportError:
    print("[FAIL] 'openai' package not installed.  Run:  pip install openai")
    sys.exit(1)

client = OpenAI(api_key=key)

# ── 3. List accessible models ─────────────────────────────────────────────────
print("\n[>] Querying available models ...")
try:
    all_models = client.models.list()
except AuthenticationError:
    print("[FAIL] Authentication rejected.  The key is invalid or revoked.")
    sys.exit(1)
except Exception as exc:
    print(f"[FAIL] Could not reach OpenAI API: {exc}")
    sys.exit(1)

available_ids: set[str] = {m.id for m in all_models.data}
gpt_ids = sorted(i for i in available_ids if i.startswith("gpt-"))
print(f"[OK]   {len(gpt_ids)} GPT models accessible:")
for mid in gpt_ids:
    print(f"       {mid}")

# ── 4. Check required models from openai_brain ────────────────────────────────
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from openai_brain import MODEL_CONFIGS, get_model_name
    active_model = get_model_name()
    print(f"\n[>] Active model (LLM_MODEL env): {active_model}")
    print("[>] Checking all registered models from openai_brain.MODEL_CONFIGS ...")
    missing = []
    for mid in MODEL_CONFIGS:
        status = "[OK]  " if mid in available_ids else "[MISS]"
        print(f"  {status} {mid}")
        if mid not in available_ids:
            missing.append(mid)

    if missing:
        print(f"\n[WARN] {len(missing)} model(s) not accessible for this key: {missing}")
    else:
        print("[OK]   All registered models are accessible.")

    if active_model not in available_ids:
        print(f"[FAIL] Active model '{active_model}' is NOT accessible!")
        print("       Change LLM_MODEL in .env to an available model.")
        sys.exit(1)
except Exception as exc:
    print(f"[WARN] Could not import openai_brain: {exc}")

# ── 5. Test chat completion (billing check) ────────────────────────────────────
test_model = active_model if "active_model" in dir() else "gpt-4.1-nano"
print(f"\n[>] Test chat completion with '{test_model}' ...")
t0 = time.perf_counter()
try:
    import re
    token_kwarg = {"max_completion_tokens": 10} if re.match(r"gpt-5", test_model) else {"max_tokens": 10}
    resp = client.chat.completions.create(
        model=test_model,
        messages=[
            {"role": "system", "content": "You are a café assistant. Reply in 5 words max."},
            {"role": "user",   "content": "Say hi."},
        ],
        temperature=0.3,
        **token_kwarg,
    )
    latency = time.perf_counter() - t0
    reply = (resp.choices[0].message.content or "").strip()
    tokens = resp.usage.total_tokens
    print(f"[OK]   Reply: '{reply}' | tokens={tokens} | latency={latency:.2f}s")
except Exception as exc:
    print(f"[FAIL] Chat completion failed: {exc}")
    sys.exit(1)

print("\n" + "=" * 55)
print(" API key verified — ready to run the voice assistant.")
print("=" * 55)
