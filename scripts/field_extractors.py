"""
field_extractors.py — Step-aware field extraction, normalisation, and
validation for the insurance voice agent.

Design goals
------------
* Robust to speech-to-text noise, filler words, casual phrasing,
  partial utterances, and common STT homophones.
* Still strict enough to reject random off-topic answers.
* Every extractor returns (is_valid, normalised_value, error_reason).
* Call extract_structured_answer(step_id, raw_text, options) from the engine.

Field types
-----------
  yes_no, phone_number, date, month_year, number, currency,
  marital_status, ownership_type, vehicle_condition, enum, free_text
"""

from __future__ import annotations

import re
from datetime import date as _date, timedelta as _timedelta
from typing import Any, Optional

from insurance_prompt import (
    validate_phone_number,
    validate_date,
    validate_month_year,
    validate_option,
    format_options_for_voice,
    is_affirmative_intent,
    is_negative_intent,
)


# ---------------------------------------------------------------------------
# Universal pre-normaliser — runs before every extractor
# ---------------------------------------------------------------------------

# Filler words / STT noise that add no semantic content.
_FILLER_RE = re.compile(
    r"\b(um+|uh+|er+|ah+|eh+|hmm+|mhm+|uhh+|like|you know|so|well|"
    r"i mean|i think|i guess|i believe|i suppose|basically|actually|"
    r"literally|honestly|right\s*so|okay\s*so|yeah\s*so|let me see|"
    r"let me think|hold on|just a sec|sorry|pardon|excuse me)\b",
    re.IGNORECASE,
)

# Leading sentence starters the caller says before the real answer.
_LEAD_IN_RE = re.compile(
    r"^(?:it(?:'s| is)|that(?:'s| is)|there(?:'s| are)|"
    r"my|mine|the|its|so|well|uh|um|yeah|yes|no|"
    r"i(?:'d| would| am| have| think| guess| suppose| believe)|"
    r"we(?:'re| are)|they(?:'re| are))\s+",
    re.IGNORECASE,
)


def _denoise(text: str) -> str:
    """Strip filler words and collapse whitespace. Preserves actual content."""
    t = _FILLER_RE.sub(" ", text)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


def _strip_lead_in(text: str) -> str:
    """Remove one leading conversational starter, once."""
    return _LEAD_IN_RE.sub("", text, count=1).strip()


# ---------------------------------------------------------------------------
# Step → field definitions
# ---------------------------------------------------------------------------

STEP_FIELD_DEFS: dict[str, dict] = {
    # ── START ──────────────────────────────────────────────────────────────
    # Note: step 1.1 (consent) is handled entirely by _apply_branch in the
    # engine; no extraction needed here — it falls through to free_text.
    "1.2":     {"field_name": "whatsapp_number",           "field_type": "phone_number"},
    "1.3":     {"field_name": "policy_start_date",         "field_type": "date"},
    "1.4":     {"field_name": "has_modifications",         "field_type": "yes_no"},
    "1.4_desc":{"field_name": "modification_description",  "field_type": "free_text"},

    # ── DRIVER ────────────────────────────────────────────────────────────
    "2.1":  {"field_name": "driver_count",                 "field_type": "number",   "min_val": 1, "max_val": 10},
    "2.2":  {"field_name": "is_registered_owner",          "field_type": "yes_no"},
    "2.2a": {"field_name": "registered_owner_name",        "field_type": "free_text"},
    "2.3":  {"field_name": "marital_status",               "field_type": "marital_status"},
    "2.4":  {"field_name": "additional_driver_listed",     "field_type": "yes_no"},
    "2.4a": {"field_name": "relationship_to_owner",        "field_type": "free_text"},
    "2.5":  {"field_name": "is_primary_operator",          "field_type": "yes_no"},
    "2.6":  {"field_name": "licence_issue_date",           "field_type": "month_year"},
    "2.7":  {"field_name": "has_driver_training",          "field_type": "yes_no"},
    "2.8":  {"field_name": "qualifies_for_retiree",        "field_type": "yes_no"},
    "2.9":  {"field_name": "retiree_discount_confirmed",   "field_type": "yes_no"},

    # ── VEHICLE ───────────────────────────────────────────────────────────
    "3.1":  {"field_name": "vehicle_count",                "field_type": "number",   "min_val": 1, "max_val": 10},
    "3.3":  {"field_name": "principal_driver",             "field_type": "free_text"},
    "3.4":  {"field_name": "ownership_type",               "field_type": "ownership_type"},
    "3.5":  {"field_name": "finance_company",              "field_type": "free_text"},
    "3.6":  {"field_name": "vehicle_year_make_model",      "field_type": "free_text"},
    "3.7":  {"field_name": "vehicle_condition",            "field_type": "vehicle_condition"},
    "3.8":  {"field_name": "purchase_price",               "field_type": "currency"},
    "3.9":  {"field_name": "has_winter_tires",             "field_type": "yes_no"},
    "3.10": {"field_name": "vehicle_loop_notes",           "field_type": "free_text"},

    # ── USAGE ─────────────────────────────────────────────────────────────
    "4.1":  {
        "field_name":    "vehicle_usage",
        "field_type":    "enum",
        "allowed_values": ["pleasure", "commuting", "business", "farm", "commercial"],
    },
    "4.2":  {"field_name": "commute_days_per_week", "field_type": "number", "min_val": 0, "max_val": 7},
    "4.3":  {"field_name": "one_way_km",            "field_type": "number", "min_val": 0, "max_val": 500},
    "4.4":  {"field_name": "annual_mileage_km",     "field_type": "number", "min_val": 0, "max_val": 500_000},
    "4.5":  {
        "field_name":    "overnight_parking",
        "field_type":    "enum",
        "allowed_values": ["private driveway", "private garage", "condo or apartment garage", "underground parking", "street"],
    },

    # ── RISK ──────────────────────────────────────────────────────────────
    "5.1":  {"field_name": "has_cancellation_history",  "field_type": "yes_no"},
    "5.2":  {"field_name": "has_claims_history",        "field_type": "yes_no"},
    "5.3":  {"field_name": "has_licence_suspension",    "field_type": "yes_no"},
    "5.4":  {"field_name": "risk_additional_notes",     "field_type": "free_text"},

    # ── CONTACT ───────────────────────────────────────────────────────────
    "6.1":  {"field_name": "contact_info",              "field_type": "free_text"},

    # ── SUMMARY ───────────────────────────────────────────────────────────
    "7.1":  {"field_name": "summary_confirmed",         "field_type": "yes_no"},
}


# ---------------------------------------------------------------------------
# Shared constants used by multiple extractors
# ---------------------------------------------------------------------------

# Qualifier words that modify amounts but don't change the core value.
_QUALIFIERS_RE = re.compile(
    r"\b(around|about|approximately|roughly|nearly|almost|"
    r"just over|just under|over|under|more than|less than|at least|at most|"
    r"maybe|probably|close to|somewhere around|something like)\b",
    re.IGNORECASE,
)

# Single spoken digit words → numeral character (used by phone extractor).
_SPOKEN_DIGIT_MAP: dict[str, str] = {
    "zero":  "0", "oh": "0", "o": "0",
    "one":   "1",
    "two":   "2", "to": "2", "too": "2",
    "three": "3",
    "four":  "4", "for": "4",
    "five":  "5",
    "six":   "6",
    "seven": "7",
    "eight": "8", "ate": "8",
    "nine":  "9",
}

# Filler words to strip before reassembling a phone number from spoken digits.
_PHONE_FILLER_RE = re.compile(
    r"\b(my|the|number|is|it|that|call|me|at|on|phone|cell|"
    r"mobile|whatsapp|contact|reach|its)\b",
    re.IGNORECASE,
)

# Word-to-number mapping — covers enough spoken forms for voice calls.
_WORD_NUMS_EXT: dict[str, int] = {
    "zero": 0,    "one": 1,      "two": 2,       "three": 3,    "four": 4,
    "five": 5,    "six": 6,      "seven": 7,     "eight": 8,    "nine": 9,
    "ten": 10,    "eleven": 11,  "twelve": 12,   "thirteen": 13,"fourteen": 14,
    "fifteen": 15,"sixteen": 16, "seventeen": 17,"eighteen": 18,"nineteen": 19,
    "twenty": 20, "thirty": 30,  "forty": 40,    "fifty": 50,
    "sixty": 60,  "seventy": 70, "eighty": 80,   "ninety": 90,
    # STT homophones that sometimes appear
    "won": 1, "to": 2, "too": 2, "for": 4, "ate": 8,
}

# Shorthand expressions that map directly to a count.
_COUNT_SHORTHANDS: dict[str, int] = {
    "just me":       1, "only me":       1, "just myself":   1, "only myself":  1,
    "just one":      1, "only one":      1, "a single":      1,
    "a couple":      2, "a pair":        2, "both":          2,
    "a few":         3, "several":       4,
}

# Filler phrases to strip from phone input.
_PHONE_FILLER_RE = re.compile(
    r"\b(my|the|number|is|it|that|call|me|at|on|"
    r"phone|cell|mobile|whatsapp|contact|reach|its|"
    r"um+|uh+|er+|so|you|can)\b",
    re.IGNORECASE,
)

# "double <digit_word>" → repeated digit: "double five" → "55"
_DOUBLE_RE = re.compile(r"\bdouble\s+(\w+)\b", re.IGNORECASE)
# "triple <digit_word>"
_TRIPLE_RE = re.compile(r"\btriple\s+(\w+)\b", re.IGNORECASE)


def _preprocess_phone_text(text: str) -> str:
    """Convert spoken digit words and patterns to a raw digit string.

    Handles:
    - Spoken digits: "nine zero two one seven zero seven zero two" → "9021707..."
    - Double/triple: "double four" → "44", "triple one" → "111"
    - STT homophones: "for" → "4", "to" → "2", "oh" → "0"
    - Filler phrases: "my number is ..." → just the digits
    - Mixed formats: "416 five five five 0199"
    """
    t = text.lower()

    # Expand double/triple before anything else.
    def _expand_double(m: re.Match) -> str:
        d = _SPOKEN_DIGIT_MAP.get(m.group(1).strip(), "")
        return d * 2 if d else m.group(0)

    def _expand_triple(m: re.Match) -> str:
        d = _SPOKEN_DIGIT_MAP.get(m.group(1).strip(), "")
        return d * 3 if d else m.group(0)

    t = _DOUBLE_RE.sub(_expand_double, t)
    t = _TRIPLE_RE.sub(_expand_triple, t)

    # Strip filler.
    t = _PHONE_FILLER_RE.sub(" ", t)

    # Split on whitespace, dashes, dots, commas.
    parts = re.split(r"[\s\-\.\,]+", t.strip())
    out: list[str] = []
    for part in parts:
        p = part.strip()
        if not p:
            continue
        if p in _SPOKEN_DIGIT_MAP:
            out.append(_SPOKEN_DIGIT_MAP[p])
        elif p.isdigit():
            out.append(p)
        # Unknown tokens silently ignored — they're likely filler.
    return "".join(out)


# ---------------------------------------------------------------------------
# Field extractors  (each returns (is_valid, normalised_value, error_reason))
# ---------------------------------------------------------------------------

# Semantic yes/no helpers — applied AFTER the consent intent-helpers so that
# phrases like "no problem" (affirmative) are already resolved before these fire.

# Guard: utterances that are genuinely ambiguous — do NOT push them to yes/no.
_YESNO_AMBIGUOUS_RE = re.compile(
    r"\bnot?\s+sure\b"           # "not sure", "no sure"
    r"|\bmaybe\b"
    r"|\bperhaps\b"
    r"|\buncertain\b"
    r"|\bunsure\b"
    r"|\bi\s+(?:don.?t|do\s+not)\s+know\b",
    re.IGNORECASE,
)

# Semantic negative markers: absence/negation of the subject matter.
_YESNO_NEG_SEMANTIC_RE = re.compile(
    r"\bnone\b"                  # "none"
    r"|\bnothing\b"              # "nothing"
    r"|\bnot\s+\w"               # "not modified", "not applicable"
    r"|\bno\s+\w"                # "no modifications", "no changes"
    r"|\bdon.?t\s+have\b"        # "don't have any"
    r"|\bdoesn.?t\s+have\b"      # "doesn't have"
    r"|\bdo\s+not\s+have\b"
    r"|\bdid\s+not\s+have\b"
    r"|\bnever\s+had\b"
    r"|\bhasn.?t\b"
    r"|\bhaven.?t\b"
    r"|\bclean\s+(?:record|history|slate)\b",  # "clean record"
    re.IGNORECASE,
)

# Semantic affirmative markers: presence/confirmation of subject matter.
_YESNO_AFF_SEMANTIC_RE = re.compile(
    r"\bi\s+(?:do\s+)?have\b"           # "I have", "I do have"
    r"|\bthere\s+(?:is|are|was|were)\b"  # "there is / there are"
    r"|\bsome\b"                        # "some modifications"
    r"|\bit\s+(?:is|was|has)\b",        # "it is modified"
    re.IGNORECASE,
)


def extract_yes_no(text: str) -> tuple[bool, str, str]:
    """Normalise a spoken yes/no answer.

    Strips filler first, then uses the full intent-detection pattern set so
    casual phrasing, hedges, and STT fragments all resolve correctly.
    """
    denoised = _denoise(text)
    if is_affirmative_intent(denoised):
        return True, "yes", ""
    if is_negative_intent(denoised):
        return True, "no", ""
    # Single-word fallbacks that intent helpers might miss.
    t = denoised.strip().lower()
    if t in {"y", "yea", "ye", "ya", "k", "kk", "aye", "correct", "true",
             "affirmative", "indeed", "roger"}:
        return True, "yes", ""
    if t in {"n", "nay", "negative", "false", "incorrect", "never",
             "not at all", "absolutely not"}:
        return True, "no", ""
    # Guard: don't force ambiguous utterances into yes/no.
    if _YESNO_AMBIGUOUS_RE.search(denoised):
        return False, "", "unclear_yes_no"
    # Semantic presence/absence patterns — catches "none", "nothing",
    # "not modified", "don't have X", "I have modifications", etc.
    if _YESNO_NEG_SEMANTIC_RE.search(denoised):
        return True, "no", ""
    if _YESNO_AFF_SEMANTIC_RE.search(denoised):
        return True, "yes", ""
    return False, "", "unclear_yes_no"


def extract_phone_number(text: str) -> tuple[bool, str, str]:
    """Extract a phone number from spoken or typed input.

    Strategy (applied in order, stops at first success):
    1. Direct validation — typed formats, formatted strings.
    2. Spoken-word conversion — handles digit words, "double X", homophones.
    3. Brute-force digits-only — strip everything non-digit.
    """
    denoised = _denoise(text)

    # 1. Direct
    ok, val, err = validate_phone_number(denoised)
    if ok:
        return ok, val, err

    # 2. Spoken-word conversion
    preprocessed = _preprocess_phone_text(denoised)
    if preprocessed:
        ok, val, err2 = validate_phone_number(preprocessed)
        if ok:
            return ok, val, err2

    # 3. Brute-force digits
    digits_only = re.sub(r"\D", "", denoised)
    if digits_only:
        ok, val, err3 = validate_phone_number(digits_only)
        if ok:
            return ok, val, err3

    return False, "", err or "invalid_length"


# Month names and abbreviations for date parsing.
_MONTH_FULL: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}
_MONTH_ABBR: dict[str, int] = {k[:3]: v for k, v in _MONTH_FULL.items()}
# STT commonly mishears month names:
_MONTH_STT_FIXES: dict[str, str] = {
    "jan": "january", "feb": "february", "mar": "march", "apr": "april",
    "jun": "june", "jul": "july", "aug": "august", "sep": "september",
    "sept": "september", "oct": "october", "nov": "november", "dec": "december",
    "marchand": "march", "march the": "march", "april the": "april",
    "2nd": "2", "3rd": "3", "4th": "4", "5th": "5", "6th": "6",
    "7th": "7", "8th": "8", "9th": "9",
}
# Ordinal suffix stripper: "26th" → "26", "1st" → "1"
_ORDINAL_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)

# Ordinal word forms for day numbers 1–31 used in spoken dates.
# Compound forms must appear before their simple components (longest match first).
_ORDINAL_WORD_SUBS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\btwenty[-\s]+first\b",   re.IGNORECASE), "21"),
    (re.compile(r"\btwenty[-\s]+second\b",  re.IGNORECASE), "22"),
    (re.compile(r"\btwenty[-\s]+third\b",   re.IGNORECASE), "23"),
    (re.compile(r"\btwenty[-\s]+fourth\b",  re.IGNORECASE), "24"),
    (re.compile(r"\btwenty[-\s]+fifth\b",   re.IGNORECASE), "25"),
    (re.compile(r"\btwenty[-\s]+sixth\b",   re.IGNORECASE), "26"),
    (re.compile(r"\btwenty[-\s]+seventh\b", re.IGNORECASE), "27"),
    (re.compile(r"\btwenty[-\s]+eighth\b",  re.IGNORECASE), "28"),
    (re.compile(r"\btwenty[-\s]+ninth\b",   re.IGNORECASE), "29"),
    (re.compile(r"\bthirty[-\s]+first\b",   re.IGNORECASE), "31"),
    (re.compile(r"\bthirtieth\b",           re.IGNORECASE), "30"),
    (re.compile(r"\btwentieth\b",           re.IGNORECASE), "20"),
    (re.compile(r"\bnineteenth\b",          re.IGNORECASE), "19"),
    (re.compile(r"\beighteenth\b",          re.IGNORECASE), "18"),
    (re.compile(r"\bseventeenth\b",         re.IGNORECASE), "17"),
    (re.compile(r"\bsixteenth\b",           re.IGNORECASE), "16"),
    (re.compile(r"\bfifteenth\b",           re.IGNORECASE), "15"),
    (re.compile(r"\bfourteenth\b",          re.IGNORECASE), "14"),
    (re.compile(r"\bthirteenth\b",          re.IGNORECASE), "13"),
    (re.compile(r"\btwelfth\b",             re.IGNORECASE), "12"),
    (re.compile(r"\beleventh\b",            re.IGNORECASE), "11"),
    (re.compile(r"\btenth\b",               re.IGNORECASE), "10"),
    (re.compile(r"\bninth\b",               re.IGNORECASE),  "9"),
    (re.compile(r"\beighth\b",              re.IGNORECASE),  "8"),
    (re.compile(r"\bseventh\b",             re.IGNORECASE),  "7"),
    (re.compile(r"\bsixth\b",               re.IGNORECASE),  "6"),
    (re.compile(r"\bfifth\b",               re.IGNORECASE),  "5"),
    (re.compile(r"\bfourth\b",              re.IGNORECASE),  "4"),
    (re.compile(r"\bthird\b",               re.IGNORECASE),  "3"),
    (re.compile(r"\bsecond\b",              re.IGNORECASE),  "2"),
    (re.compile(r"\bfirst\b",               re.IGNORECASE),  "1"),
]


def _normalise_date_text(text: str) -> str:
    """Pre-process spoken date text for parsing: strip ordinals, normalise separators."""
    t = text.lower().strip()
    t = _ORDINAL_RE.sub(r"\1", t)                     # "26th" → "26"
    for pat, sub in _ORDINAL_WORD_SUBS:               # "twenty-sixth" → "26"
        t = pat.sub(sub, t)
    t = re.sub(r"\bof\b", " ", t)                     # "26th of March" → "26 March"
    t = re.sub(r",", " ", t)                          # "March 26, 2026" → "March 26  2026"
    t = re.sub(r"(?<=\d)\.(?=\d)", " ", t)            # "26.3.2027" → "26 3 2027"
    t = re.sub(r"\s*/\s*", " ", t)                    # "26 / 3 / 2027" → "26 3 2027"
    t = re.sub(r"(?<=\d)-(?=\d)", " ", t)             # "26-03-2027" → "26 03 2027"
    for wrong, fixed in _MONTH_STT_FIXES.items():
        t = re.sub(rf"\b{wrong}\b", fixed, t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[.,;!?]+$", "", t).strip()              # drop trailing STT punctuation artefacts
    return t


def _reconstruct_numeric_date(tokens: list[str]) -> Optional[str]:
    """Build an ISO date string from a list of purely-numeric tokens.

    Handles two common spoken/STT forms:
    - [DD, MM, YYYY]             e.g. ["26", "3", "2027"]       -> 2027-03-26
    - [DD, MM, D, D, D, D]       e.g. ["26", "3", "2", "0", "2", "7"] -> 2027-03-26

    Returns an ISO string or None if the tokens do not match a plausible date.
    """
    if not tokens or not all(t.isdigit() for t in tokens):
        return None
    nums = [int(t) for t in tokens]

    # Form: DD MM YYYY  (3 tokens) or YYYY MM DD (ISO-ish, first token 4 digits)
    if len(tokens) == 3:
        # Try DD MM YYYY first
        day, month, year = nums
        if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
            try:
                return _date(year, month, day).isoformat()
            except ValueError:
                pass
        # Try YYYY MM DD ("2027 03 26")
        if len(tokens[0]) == 4:
            year, month, day = nums
            if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
                try:
                    return _date(year, month, day).isoformat()
                except ValueError:
                    pass

    # Form: DD MM D D D D  (6 tokens — year spoken digit by digit)
    if len(tokens) == 6 and all(len(t) == 1 for t in tokens[2:]):
        day, month = nums[0], nums[1]
        year = nums[2] * 1000 + nums[3] * 100 + nums[4] * 10 + nums[5]
        if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
            try:
                return _date(year, month, day).isoformat()
            except ValueError:
                pass

    return None


def _relative_to_date(text: str) -> Optional[str]:
    """Resolve simple relative date expressions to ISO strings.

    Returns None if the text does not contain a recognisable relative reference.
    """
    today = _date.today()
    t = _denoise(text).lower().strip()

    if t in ("today",):
        return today.isoformat()
    if t in ("tomorrow",):
        return (today + _timedelta(days=1)).isoformat()
    if re.search(r"\bnext\s+month\b", t):
        # First day of next month
        yr = today.year + (today.month // 12)
        mo = (today.month % 12) + 1
        return _date(yr, mo, 1).isoformat()
    # "in X days/weeks/months"
    m = re.search(r"in\s+(\w+)\s+(days?|weeks?|months?)", t)
    if m:
        n_str, unit = m.group(1), m.group(2)
        n = _WORD_NUMS_EXT.get(n_str) or (int(n_str) if n_str.isdigit() else None)
        if n is not None:
            if "day" in unit:
                return (today + _timedelta(days=n)).isoformat()
            if "week" in unit:
                return (today + _timedelta(weeks=n)).isoformat()
            if "month" in unit:
                yr = today.year + ((today.month - 1 + n) // 12)
                mo = (today.month - 1 + n) % 12 + 1
                return _date(yr, mo, 1).isoformat()
    return None


def _expand_compound_word_nums(text: str) -> str:
    """Expand compound spoken number forms that appear in dates.

    Run BEFORE individual word-number substitution so that:
    - "twenty six"        → "26"    (day numbers)
    - "twenty twenty seven" → "2027"  (years)
    Patterns are matched from longest (most-specific) to shortest.
    """
    t = text
    _ones: dict[str, int] = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9,
    }
    # "twenty twenty <ones>" → 2021-2029  (longest pattern, checked first)
    for w, v in _ones.items():
        t = re.sub(
            rf"\btwenty[-\s]+twenty[-\s]+{w}\b",
            str(2020 + v), t, flags=re.IGNORECASE,
        )
    # "twenty twenty" → 2020
    t = re.sub(r"\btwenty[-\s]+twenty\b", "2020", t, flags=re.IGNORECASE)
    # "twenty thirty" → 2030
    t = re.sub(r"\btwenty[-\s]+thirty\b", "2030", t, flags=re.IGNORECASE)
    # "twenty <ones>" → 21-29
    for w, v in _ones.items():
        t = re.sub(
            rf"\btwenty[-\s]+{w}\b",
            str(20 + v), t, flags=re.IGNORECASE,
        )
    # "thirty one" → 31
    t = re.sub(r"\bthirty[-\s]+one\b", "31", t, flags=re.IGNORECASE)
    # "two thousand N" (N already numeric after rules above) → 2000+N
    # e.g. "march 26 two thousand 26" → "march 26 2026"
    t = re.sub(
        r"\btwo\s+thousand\s+(?:and\s+)?(\d{1,2})\b",
        lambda m: str(2000 + int(m.group(1))),
        t, flags=re.IGNORECASE,
    )
    # "two thousand" alone → 2000
    t = re.sub(r"\btwo\s+thousand\b", "2000", t, flags=re.IGNORECASE)
    return t


def extract_date(text: str) -> tuple[bool, str, str]:
    """Extract a future policy start date from spoken text.

    Handles: relative expressions, ordinals, abbreviated months, reordered
    date components, dot/slash/dash separators, compound word numbers, and
    many STT variants.
    """
    denoised = _denoise(text)

    # 1. Relative dates
    rel = _relative_to_date(denoised)
    if rel:
        # Policy start can be today — allow past=False only blocks genuine past dates
        today = _date.today()
        parsed = _date.fromisoformat(rel)
        if parsed >= today:
            return True, rel, ""
        return False, "", "date_in_past"

    # 2. Standard validator first (handles many typed formats)
    ok, val, err = validate_date(denoised, allow_past=False)
    if ok:
        return ok, val, err

    # 3. Normalise spoken text and retry
    normalised = _normalise_date_text(denoised)
    ok, val, err2 = validate_date(normalised, allow_past=False)
    if ok:
        return ok, val, err2
    if err2 == "date_in_past":
        return False, "", "date_in_past"

    # 4. All-numeric forms: "26 3 2027", "26.3.2027", "26 3 2 0 2 7", etc.
    #    _normalise_date_text already converted commas and digit-dots to spaces.
    tokens = normalised.split()
    if all(t.isdigit() for t in tokens):
        iso = _reconstruct_numeric_date(tokens)
        if iso:
            ok, val, err4 = validate_date(iso, allow_past=False)
            if ok:
                return ok, val, err4
            if err4 == "date_in_past":
                return False, "", "date_in_past"

    # 5. Compound word-number expansion — BEFORE individual word substitution.
    #    "twenty six march twenty twenty seven" → "26 march 2027"
    expanded = _expand_compound_word_nums(normalised)
    ok, val, err5 = validate_date(expanded, allow_past=False)
    if ok:
        return ok, val, err5
    if err5 == "date_in_past":
        return False, "", "date_in_past"
    # Also try all-numeric path after compound expansion.
    tokens5 = expanded.split()
    if all(t.isdigit() for t in tokens5):
        iso5 = _reconstruct_numeric_date(tokens5)
        if iso5:
            ok, val, err5b = validate_date(iso5, allow_past=False)
            if ok:
                return ok, val, err5b
            if err5b == "date_in_past":
                return False, "", "date_in_past"

    # 6. Individual word-number substitution on the compound-expanded text.
    reconstructed = expanded
    for word, num in sorted(_WORD_NUMS_EXT.items(), key=lambda x: -len(x[0])):
        reconstructed = re.sub(rf"\b{word}\b", str(num), reconstructed)
    ok, val, err3 = validate_date(reconstructed, allow_past=False)
    if ok:
        return ok, val, err3
    if err3 == "date_in_past":
        return False, "", "date_in_past"

    return False, "", err or "unrecognised_format"


# STT fixes for common month-name mishearings in licence date context.
_LICENCE_DATE_STT: dict[str, str] = {
    "to": "2", "too": "2", "for": "4",
    "won": "1", "ate": "8",
}


def extract_month_year(text: str) -> tuple[bool, str, str]:
    """Extract a month + year (licence issue date). Returns "YYYY-MM".

    Robust to spoken forms like: "around January 2010", "jan twenty ten",
    "two thousand and ten", and common STT errors.
    """
    denoised = _denoise(text)

    # 1. Try standard validator directly
    ok, val, err = validate_month_year(denoised)
    if ok:
        return ok, val, err

    # 2. Normalise ordinals + abbreviations then retry
    normalised = _normalise_date_text(denoised)
    ok, val, err2 = validate_month_year(normalised)
    if ok:
        return ok, val, err2
    if err2 == "year_out_of_range":
        return False, "", "year_out_of_range"

    # 3. Expand spoken year forms: "twenty ten" → "2010", "two thousand ten"
    t = normalised
    # "two thousand and X" → "200X"
    t = re.sub(
        r"\btwo\s*thousand\s*(?:and\s*)?(\d{1,2})\b",
        lambda m: str(2000 + int(m.group(1))),
        t,
    )
    # "twenty <digit_word>" → "20XX"
    t = re.sub(
        r"\btwenty[-\s]+(\w+)\b",
        lambda m: str(2020 + (_WORD_NUMS_EXT.get(m.group(1).lower(), 0) - 20))
        if _WORD_NUMS_EXT.get(m.group(1).lower(), 100) < 10
        else m.group(0),
        t,
    )
    # Compound year forms not caught above: "twenty-twenty-seven" → 2027, etc.
    t = _expand_compound_word_nums(t)
    ok, val, err3 = validate_month_year(t)
    if ok:
        return ok, val, err3
    if err3 == "year_out_of_range":
        return False, "", "year_out_of_range"

    # 4. Extract any 4-digit year as fallback (year-only is valid for this field)
    m = re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", t)
    if m:
        yr = int(m.group(1))
        today_year = _date.today().year
        if yr <= today_year:
            # Try to find a month
            for mon_str, mon_num in {**_MONTH_FULL, **_MONTH_ABBR}.items():
                if mon_str in t:
                    return True, f"{yr:04d}-{mon_num:02d}", ""
            # Year only
            return True, f"{yr:04d}-01", ""
        return False, "", "year_out_of_range"

    return False, "", err or "unrecognised_format"


def extract_number(
    text: str,
    min_val: int = 0,
    max_val: int = 100,
) -> tuple[bool, int, str]:
    """Extract a numeric count or quantity from spoken text.

    Handles:
    - Digit strings with qualifiers: "around 4", "about twelve"
    - Word numbers: "three drivers", "just the two of us"
    - Shorthand: "just me" → 1, "a couple" → 2, "a few" → 3
    - STT noise: strips filler before looking for numbers
    """
    denoised = _denoise(text).lower()

    # 1. Shorthand expressions (checked before anything else)
    for phrase, val in sorted(_COUNT_SHORTHANDS.items(), key=lambda x: -len(x[0])):
        if phrase in denoised:
            if min_val <= val <= max_val:
                return True, val, ""

    # 2. Strip qualifiers
    cleaned = _QUALIFIERS_RE.sub("", denoised).strip()

    # 3. Explicit digit sequence (possibly with commas: "20,000")
    m = re.search(r"\b(\d[\d,]*)\b", cleaned)
    if m:
        val = int(m.group(1).replace(",", ""))
        if val < min_val:
            return False, val, "below_minimum"
        if val > max_val:
            return False, val, "above_maximum"
        return True, val, ""

    # 4. Word numbers — check all words, pick the first match.
    #    Sort by word length descending to avoid "nine" matching inside "nineteen".
    words = re.sub(r"[^\w\s]", " ", cleaned).split()
    for word in words:
        if word in _WORD_NUMS_EXT:
            val = _WORD_NUMS_EXT[word]
            if val < min_val:
                return False, val, "below_minimum"
            if val > max_val:
                return False, val, "above_maximum"
            return True, val, ""

    return False, 0, "not_a_number"


def extract_currency(text: str) -> tuple[bool, int, str]:
    """Extract a dollar / rupee amount from spoken text.

    Supports:
    - Lakh notation:   "1 lakh 60 thousand", "1.6 lakh", "one lakh sixty thousand"
    - Thousand suffix: "160 thousand", "a hundred sixty thousand"
    - Plain numbers:   "160000", "1,60,000", "160,000"
    - Qualifier words: "around 1.6 lakh", "about 160 thousand"
    - "hundred" alone: "one hundred sixty thousand" → 160000
    """
    t = _denoise(text).lower().strip()

    # Strip qualifiers and currency symbols.
    t = _QUALIFIERS_RE.sub("", t).strip()
    t = re.sub(r"[₹$€£]", "", t)

    # Expand word-numbers (longest first to avoid partial matches).
    for word, num in sorted(_WORD_NUMS_EXT.items(), key=lambda x: -len(x[0])):
        t = re.sub(rf"\b{word}\b", str(num), t)

    # Also handle "a hundred" → "100"
    t = re.sub(r"\ba\b\s*(hundred|thousand|lakh|lac)\b",
               lambda m: {"hundred": "100", "thousand": "1000",
                          "lakh": "100000", "lac": "100000"}[m.group(1)], t)

    # Remove all commas.
    t = t.replace(",", "")

    # Pattern: N lakh [N thousand/hundred] [N]
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:lakh|lac)\b"
        r"(?:\s+(\d+(?:\.\d+)?)\s*(?:thousand|hundreds?))?"
        r"(?:\s+(\d+(?:\.\d+)?))?",
        t,
    )
    if m:
        lakhs     = float(m.group(1)) * 100_000
        thousands = float(m.group(2)) * 1_000 if m.group(2) else 0.0
        units     = float(m.group(3)) if m.group(3) else 0.0
        total = int(lakhs + thousands + units)
        if total > 0:
            return True, total, ""

    # Pattern: N million
    m = re.search(r"(\d+(?:\.\d+)?)\s*million\b", t)
    if m:
        total = int(float(m.group(1)) * 1_000_000)
        if total > 0:
            return True, total, ""

    # Pattern: N hundred thousand (e.g. "160 hundred thousand" is wrong
    #   but "one hundred sixty thousand" → after word expansion = "100 60 thousand")
    #   Handle "100 60 thousand" edge case → 160000
    m = re.search(r"(\d+)\s+(\d+)\s*thousand\b", t)
    if m:
        total = (int(m.group(1)) + int(m.group(2))) * 1_000
        if 0 < total < 10_000_000:
            return True, total, ""

    # Pattern: N thousand
    m = re.search(r"(\d+(?:\.\d+)?)\s*thousand\b", t)
    if m:
        total = int(float(m.group(1)) * 1_000)
        if total > 0:
            return True, total, ""

    # Pattern: N hundred
    m = re.search(r"(\d+(?:\.\d+)?)\s*hundred\b", t)
    if m:
        total = int(float(m.group(1)) * 100)
        if total > 0:
            return True, total, ""

    # Plain integer or float in the whole string.
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", t)
    if m:
        val = float(m.group(1))
        if val > 0:
            return True, int(val), ""

    # Any digit sequence as last resort.
    m = re.search(r"(\d+(?:\.\d+)?)", t)
    if m:
        val = float(m.group(1))
        if val > 0:
            return True, int(val), ""

    return False, 0, "not_a_currency"


# ── Categorical field types ──────────────────────────────────────────────────
# Each map is a list of (keyword_tuple, canonical_value) pairs.
# Keywords are matched as substrings of the lowercased input.
# More specific phrases are checked before shorter ones where needed.

_MARITAL_MAP: list[tuple[tuple[str, ...], str]] = [
    (
        (
            "single", "unmarried", "not married", "never married",
            "on my own", "by myself", "no partner", "unattached",
            "bachelor", "bachelorette", "solo",
        ),
        "single",
    ),
    (
        (
            "common law", "common-law", "commonlaw",
            "living together", "living with", "live together",
            "live with my", "cohabit", "co-habit",
            "common-law partner", "common law partner",
            "together but not married", "not married but together",
            "like married", "practically married",
        ),
        "common_law",
    ),
    (
        (
            "divorced", "separated", "separation",
            "ex-wife", "ex-husband", "ex wife", "ex husband",
            "split up", "split from", "no longer married",
            "used to be married", "was married",
        ),
        "divorced",
    ),
    (
        (
            "widowed", "widow", "widower",
            "my spouse passed", "my husband passed", "my wife passed",
            "lost my spouse", "lost my husband", "lost my wife",
        ),
        "widowed",
    ),
    (
        # Checked last — "married" is a substring of several divorced/common-law phrases.
        (
            "married", "spouse", "husband", "wife", "wed",
            "legally married", "got married", "been married",
            "my wife", "my husband",
        ),
        "married",
    ),
]


def extract_marital_status(text: str) -> tuple[bool, str, str]:
    """Normalise spoken marital status.

    Returns one of: single | married | common_law | divorced | widowed.
    """
    t = _denoise(text).lower().strip()
    t = _strip_lead_in(t)
    for keywords, canonical in _MARITAL_MAP:
        for kw in keywords:
            if kw in t:
                return True, canonical, ""
    return False, "", "no_marital_match"


_OWNERSHIP_MAP: list[tuple[tuple[str, ...], str]] = [
    (
        (
            "lease", "leasing", "renting", "on a lease", "under a lease",
            "lease agreement", "leased vehicle", "i lease",
        ),
        "leased",
    ),
    (
        (
            "financ", "loan", "paying it off", "still paying", "making payments",
            "monthly payments", "bank loan", "credit union", "auto loan",
            "car loan", "i owe", "owe money", "not paid off",
            "not fully paid", "got a loan", "have a loan",
        ),
        "financed",
    ),
    (
        (
            "own it", "own the", "owned", "outright", "paid off", "paid it off",
            "fully paid", "no loan", "no payments", "free and clear",
            "it's mine", "its mine", "bought it cash", "cash purchase",
            "no finance", "no financing", "cleared", "lien free", "lien-free",
        ),
        "owned",
    ),
]


def extract_ownership_type(text: str) -> tuple[bool, str, str]:
    """Normalise vehicle ownership. Returns leased | financed | owned.

    Order matters: leased and financed are checked before owned to avoid
    "I own it but I'm still financing it" → owned incorrectly.
    """
    t = _denoise(text).lower().strip()
    t = _strip_lead_in(t)
    for keywords, canonical in _OWNERSHIP_MAP:
        for kw in keywords:
            if kw in t:
                return True, canonical, ""
    return False, "", "no_ownership_match"


_CONDITION_MAP: list[tuple[tuple[str, ...], str]] = [
    (
        (
            "demo", "demonstration", "demo vehicle", "demo model",
            "floor model", "dealer demo", "show model", "display model",
        ),
        "demo",
    ),
    (
        (
            "used", "second hand", "secondhand", "second-hand",
            "pre-owned", "preowned", "pre owned", "pre-owned",
            "older", "previously owned", "not new", "old car",
            "a few years old", "few years", "couple years old",
            "high mileage", "it's old", "pretty old",
        ),
        "used",
    ),
    (
        (
            "brand new", "brand-new", "factory new",
            "never driven", "0 km", "zero km", "zero kilometers", "zero kilometres",
            "just bought new", "just got it new", "fresh off the lot",
            "off the lot", "straight from the dealer", "new from the dealer",
        ),
        "new",
    ),
]

# Standalone "new" as last-resort fallback for condition.
_NEW_WORD_RE = re.compile(r"\bnew\b")


def extract_vehicle_condition(text: str) -> tuple[bool, str, str]:
    """Normalise spoken vehicle condition. Returns demo | used | new.

    Order: demo and used before new (avoids "brand new" matching "new" early).
    """
    t = _denoise(text).lower().strip()
    t = _strip_lead_in(t)
    for keywords, canonical in _CONDITION_MAP:
        for kw in keywords:
            if kw in t:
                return True, canonical, ""
    if _NEW_WORD_RE.search(t):
        return True, "new", ""
    return False, "", "no_condition_match"


# Common spoken usage aliases for enum steps (vehicle_usage, etc.).
# Checked longest-phrase-first inside extract_enum.
_ENUM_ALIASES: dict[str, str] = {
    # ── business
    "for work":               "business",
    "work use":               "business",
    "work purposes":          "business",
    "business use":           "business",
    "for business":           "business",
    "business purposes":      "business",
    "company car":            "business",
    "work car":               "business",
    "going to work":          "business",

    # ── commercial
    "commercial purpose":     "commercial",
    "for hire":               "commercial",
    "taxi":                   "commercial",
    "delivery":               "commercial",
    "rideshare":              "commercial",
    "uber":                   "commercial",
    "lyft":                   "commercial",
    "drive for hire":         "commercial",
    "transport people":       "commercial",

    # ── commuting
    "commute":                "commuting",
    "daily driver":           "commuting",
    "back and forth":         "commuting",
    "to and from work":       "commuting",
    "to work and back":       "commuting",
    "every day to work":      "commuting",
    "drive to work":          "commuting",
    "daily commute":          "commuting",

    # ── pleasure
    "personal use":           "pleasure",
    "personal":               "pleasure",
    "leisure":                "pleasure",
    "weekend":                "pleasure",
    "recreational":           "pleasure",
    "just driving around":    "pleasure",
    "for fun":                "pleasure",
    "pleasure driving":       "pleasure",
    "joy rides":              "pleasure",
    "casual":                 "pleasure",
    "occasional use":         "pleasure",
    "here and there":         "pleasure",

    # ── farm
    "farm use":               "farm",
    "farming":                "farm",
    "agricultural":           "farm",
    "on the farm":            "farm",
    "farm work":              "farm",
}


def extract_enum(text: str, allowed_values: list[str]) -> tuple[bool, str, str]:
    """Validate an answer against a fixed set of allowed values.

    Checks (in order):
    1. _ENUM_ALIASES mapping — longer phrases checked first.
    2. Partial-match against allowed values (case-insensitive substring match).

    Returns (is_valid, matched_canonical_value, error_reason).
    """
    t = _denoise(text).lower().strip()
    allowed_lower = [v.lower() for v in allowed_values]

    # Check aliases — longest phrases first.
    for phrase, canonical in sorted(_ENUM_ALIASES.items(), key=lambda x: -len(x[0])):
        if phrase in t and canonical in allowed_lower:
            return True, canonical, ""

    # Partial-match against allowed values directly.
    ok, val, err = validate_option(text, allowed_values)
    if ok:
        return True, val.lower(), ""

    return False, "", "no_option_match"


# Patterns that flag STT noise or non-answer speech rather than a genuine reply.
_NOISE_FREE_TEXT_RE = re.compile(
    r"\b(good[\-\s]?bye|farewell|see\s+you|talk\s+(?:to\s+you\s+)?later|"
    r"i\s+already\s+reached|i\s+need\s+to\s+go|i\s+have\s+to\s+go|"
    r"i\s*'?m\s+hanging\s+up|let\s+me\s+go)\b",
    re.IGNORECASE,
)


def extract_free_text(text: str) -> tuple[bool, str, str]:
    """Accept any non-trivial, non-pure-filler answer.

    Rejects: empty, single chars, answers that are *only* filler words
    (e.g. "um", "uh", "like you know"), and obvious STT noise / non-answers.
    """
    stripped = _denoise(text).strip()
    if not stripped or len(stripped) < 2:
        return False, "", "empty_answer"
    # Reject if the denoised result is just one character.
    if len(stripped.replace(" ", "")) < 2:
        return False, "", "empty_answer"
    # Reject obvious non-answer phrases (leave-taking, background speech artefacts).
    if _NOISE_FREE_TEXT_RE.search(stripped):
        return False, "", "unrecognised_format"
    return True, text.strip(), ""  # return original (not denoised) for free-text


# ---------------------------------------------------------------------------
# Intent classifier — distinguish answer vs. clarification/objection/off-topic
# ---------------------------------------------------------------------------

# Phrases that signal the caller is asking for help or clarification, NOT answering.
_CLARIFICATION_RE = re.compile(
    r"what do you mean"
    r"|what does that mean"
    r"|what did you mean"
    r"|can you repeat"
    r"|could you repeat"
    r"|say that again"
    r"|repeat that"
    r"|i don.?t understand"
    r"|i do not understand"
    r"|not sure what you mean"
    r"|what format"
    r"|which format"
    r"|\bhow do i\b"
    r"|\bhow should i\b"
    r"|why do you need"
    r"|why are you asking"
    r"|can you explain"
    r"|could you explain"
    r"|\bconfused\b"
    r"|what was the question"
    r"|what are the options"
    r"|\bpardon\b"
    r"|\bhuh\b"
    r"|come again",
    re.IGNORECASE,
)

# Phrases that signal the caller is pushing back or refusing this question.
_OBJECTION_RE = re.compile(
    r"prefer not to"
    r"|rather not"
    r"|\bcan i skip\b"
    r"|\bcould i skip\b"
    r"|skip that\b"
    r"|skip this\b"
    r"|pass on that"
    r"|pass on this"
    r"|\bdo i have to\b"
    r"|is that required"
    r"|is that necessary"
    r"|is that mandatory"
    r"|\bnot comfortable\b"
    r"|that.s private"
    r"|that is private"
    r"|don.t want to (?:share|answer|give|say)"
    r"|do not want to (?:share|answer|give|say)",
    re.IGNORECASE,
)


def classify_intent(text: str) -> str:
    """Classify the user's intent in a form-filling voice conversation.

    Called by the flow engine BEFORE extraction and field storage so the engine
    can respond appropriately without accidentally treating a clarification
    question as a field value.

    Returns one of:
    - ``"clarification"`` — the user asked a help / repeat / format question
    - ``"objection"``     — the user is resisting or refusing this question
    - ``"off_topic"``     — the utterance has no meaningful content
    - ``"answer"``        — the user is attempting to answer (default)
    """
    denoised = _denoise(text).strip()

    # Genuinely empty / all-filler input.
    if not denoised or len(denoised.replace(" ", "")) < 3:
        return "off_topic"

    if _CLARIFICATION_RE.search(denoised):
        return "clarification"

    if _OBJECTION_RE.search(denoised):
        return "objection"

    return "answer"


# ---------------------------------------------------------------------------
# Main router — called by the flow engine
# ---------------------------------------------------------------------------

def extract_structured_answer(
    step_id: str,
    user_text: str,
    options: Optional[list[str]] = None,
) -> tuple[bool, Any, str]:
    """Entry point for the flow engine.

    Given a step ID and the raw transcribed answer, extracts, normalises, and
    validates the value. Returns (is_valid, normalised_value, error_reason).

    If the step is not in STEP_FIELD_DEFS and no options list is provided,
    accepts any non-empty text (free-text fallback — never blocks flow).
    """
    field_def = STEP_FIELD_DEFS.get(step_id)

    if field_def is None:
        if options:
            return extract_enum(user_text, options)
        return extract_free_text(user_text)

    ft = field_def["field_type"]

    if ft == "yes_no":
        return extract_yes_no(user_text)
    if ft == "phone_number":
        return extract_phone_number(user_text)
    if ft == "date":
        return extract_date(user_text)
    if ft == "month_year":
        return extract_month_year(user_text)
    if ft == "number":
        return extract_number(
            user_text,
            min_val=field_def.get("min_val", 0),
            max_val=field_def.get("max_val", 1_000_000),
        )
    if ft == "currency":
        return extract_currency(user_text)
    if ft == "marital_status":
        return extract_marital_status(user_text)
    if ft == "ownership_type":
        return extract_ownership_type(user_text)
    if ft == "vehicle_condition":
        return extract_vehicle_condition(user_text)
    if ft == "enum":
        allowed = field_def.get("allowed_values") or options or []
        return extract_enum(user_text, allowed)

    return extract_free_text(user_text)


# ---------------------------------------------------------------------------
# Retry prompt builder
# ---------------------------------------------------------------------------

# Each entry is a list of variant hints that are cycled through on subsequent
# retries to avoid the agent sounding like a broken recording.
_RETRY_VARIANTS: dict[str, list[str]] = {
    # phone
    "invalid_length": [
        "I need the full ten-digit number — could you try again?",
        "That number's a bit short — can you read it out digit by digit?",
    ],
    "invalid_area_code": [
        "That area code doesn't look right — can you give me the full number again?",
        "Could you repeat the number? Just the digits, including area code.",
    ],
    # date
    "unrecognised_format": [
        "Could you give me that date — something like March 26th 2026?",
        "What date did you have in mind? Month, day, and year works.",
    ],
    "date_in_past": [
        "We'd need a future date for the policy start — what were you thinking?",
        "That date's already passed — when would you like the coverage to begin?",
    ],
    # month_year (licence date)
    "year_out_of_range": [
        "That year doesn't look right — roughly when did you get your licence?",
        "Could you try that again? Just the year, or month and year.",
    ],
    "month_or_year_out_of_range": [
        "Something seems off with that date — could you try again?",
        "Just month and year is fine — like January 2010.",
    ],
    # numbers
    "not_a_number": [
        "I just need a number there — how many would that be?",
        "Could you give me a number? Even an approximate one is fine.",
    ],
    "below_minimum": [
        "That seems a bit low — could you double-check?",
        "Just to confirm — did you mean a higher number?",
    ],
    "above_maximum": [
        "That seems quite high — could you double-check?",
        "That's more than I'd expect — could you confirm that figure?",
    ],
    # currency
    "not_a_currency": [
        "Can you give me a dollar amount — like a hundred and sixty thousand?",
        "What was the purchase price? An approximate dollar figure is fine.",
    ],
    # yes / no
    "unclear_yes_no": [
        "Just a yes or no is fine.",
        "Sorry — was that a yes or a no?",
    ],
    # enum
    "no_option_match": [
        "That didn't quite match — could you pick one of the options?",
        "Could you choose one of the listed options?",
    ],
    # categorical
    "no_marital_match": [
        "You can say single, married, common law, divorced, or widowed.",
        "Just one of these: single, married, common law, divorced, or widowed.",
    ],
    "no_ownership_match": [
        "Is it owned outright, financed, or leased?",
        "Just checking — owned, financed, or on a lease?",
    ],
    "no_condition_match": [
        "Is it new, used, or a demo vehicle?",
        "New, used, or demo — which one fits?",
    ],
    # generic
    "empty_answer": [
        "Sorry — I didn't catch anything. Could you try again?",
        "Didn't get that — could you repeat?",
    ],
}

# Index tracking for variant rotation per (step_id, error_reason) pair.
_retry_counters: dict[tuple[str, str], int] = {}


def build_retry_prompt(
    step_id: str,
    error_reason: str,
    options: Optional[list[str]] = None,
) -> str:
    """Return a short, voice-friendly retry hint.

    Rotates through variant phrasings on repeated retries for the same step
    so the agent doesn't sound like a broken record.

    For option-mismatch errors the hint is enriched with the actual choices.
    """
    variants = _RETRY_VARIANTS.get(
        error_reason,
        ["Sorry — I didn't quite get that. Could you try again?",
         "Could you try that again?"],
    )

    key = (step_id, error_reason)
    idx = _retry_counters.get(key, 0)
    hint = variants[idx % len(variants)]
    _retry_counters[key] = idx + 1

    # Enrich option-list hints.
    if error_reason == "no_option_match":
        choices = options
        if not choices:
            field_def = STEP_FIELD_DEFS.get(step_id, {})
            choices = field_def.get("allowed_values")
        if choices:
            hint = f"Just checking — is it {format_options_for_voice(choices)}?"

    return hint
