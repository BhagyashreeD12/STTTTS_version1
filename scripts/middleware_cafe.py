import json
import os
import re
from difflib import SequenceMatcher


# =========================================================
# MENU LOADING
# =========================================================

_MENU_PATH = os.path.join(os.path.dirname(__file__), "menu.json")

with open(_MENU_PATH, encoding="utf-8") as f:
    _raw = json.load(f)

MENU = {}
for row in _raw:
    cat = str(row.get("Category", "")).strip()
    sub = str(row.get("Sub-Category", "")).strip()
    item = str(row.get("Item Name", "")).strip()
    size = str(row.get("Options", "Regular")).strip()
    price = float(row.get("Price", 0.0))

    if not cat or not sub or not item:
        continue

    MENU.setdefault(cat, {}).setdefault(sub, {}).setdefault(item, []).append(
        {"size": size, "price": price}
    )


# =========================================================
# GLOBAL STATE
# =========================================================


def reset_state():
    return {
        "step": "greeting",
        "category": None,
        "subcategory": None,
        "pending_item": None,
        "pending_sizes": [],
        "cart": [],
        "name": None,
        "phone": None,
        "order_type": None,
        "people_count": None,
        "datetime": None,
        "confirmed": False,
        "greeted_once": False,
        "last_user_intent": None,
        "last_recommendation_context": None,
        "turn_count": 0,
    }


state = reset_state()

greeting = "Hello, this is Alex at Crumbs and Cream. How can I help you today?"


# =========================================================
# SYSTEM PROMPT EXPORT
# =========================================================


def build_system_prompt():
    return _build_system_prompt()


# =========================================================
# NORMALIZATION / HELPERS
# =========================================================


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _strip_correction_prefixes(text: str) -> str:
    prefixes = [
        "no ",
        "no, ",
        "actually ",
        "actually, ",
        "sorry ",
        "sorry, ",
        "i mean ",
        "i mean, ",
        "wait ",
        "wait, ",
        "not that ",
        "not this ",
        "instead ",
    ]
    cleaned = text.strip().lower()
    for p in prefixes:
        if cleaned.startswith(p):
            return cleaned[len(p) :].strip()
    return cleaned


def _semantic_guard_replacements(text: str) -> str:
    t = _normalize(text)

    protected_pairs = {
        "ice cream": "ice cream",
        "iced coffee": "iced coffee",
        "ice coffee": "iced coffee",
        "cold coffee": "iced coffee",
    }

    for wrongish, canonical in protected_pairs.items():
        if wrongish in t:
            return canonical

    return t


def _spoken_join(items):
    return " ... ".join(items)


def _contains_greeting(text: str) -> bool:
    t = _normalize(text)
    words = set(t.split())
    single = {"hello", "hi", "hey", "salam"}
    multi = ["good morning", "good afternoon", "good evening", "assalamualaikum"]
    return bool(words & single) or any(g in t for g in multi)


def _contains_correction(text: str) -> bool:
    t = _normalize(text)
    correction_markers = [
        "no",
        "not",
        "i said",
        "i mean",
        "not that",
        "not this",
        "instead",
        "actually",
        "wrong",
        "not ice cream",
        "not from",
    ]
    return any(marker in t for marker in correction_markers)


def _contains_recommendation_request(text: str) -> bool:
    t = _normalize(text)
    triggers = [
        "recommend",
        "suggest",
        "what's good",
        "what is good",
        "popular",
        "best",
        "what do people get",
        "what should i order",
        "something nice",
        "what do you suggest",
        "what do you recommend",
    ]
    return any(x in t for x in triggers)


def _contains_browse_style_request(text: str) -> bool:
    t = _normalize(text)
    triggers = [
        "show me",
        "i want to see",
        "what do you have",
        "explore",
        "see options",
        "what options",
        "what all",
        "available",
    ]
    return any(x in t for x in triggers)


def _contains_chitchat(text: str) -> bool:
    t = _normalize(text)
    phrases = {
        "oh nice",
        "nice",
        "okay",
        "ok",
        "hmm",
        "hmmm",
        "cool",
        "sounds good",
        "great",
        "oh",
        "i like it",
        "good",
        "alright",
    }
    return t in phrases or (len(t.split()) <= 3 and t in phrases)


def _contains_favorite_question(text: str) -> bool:
    t = _normalize(text)
    triggers = [
        "your favorite",
        "what's your favorite",
        "what is your favorite",
        "what do you like",
        "what do people like",
        "popular one",
        "most popular",
        "best seller",
        "customer favorite",
        "favourite",
    ]
    return any(x in t for x in triggers)


def _extract_order_type(text: str):
    t = _normalize(text)

    pickup_signals = [
        "pickup",
        "pick up",
        "takeaway",
        "take away",
        "carry out",
        "to go",
    ]
    dinein_signals = [
        "dine in",
        "dine-in",
        "dining",
        "eat here",
        "for here",
        "sit here",
    ]

    if any(x in t for x in pickup_signals):
        return "pickup"

    if any(x in t for x in dinein_signals):
        return "dine-in"

    return None


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _match(text, options, cutoff=0.70):
    if not text or not options:
        return None

    text_norm = _normalize(text)
    options_list = list(options)

    # 1) exact full match
    for opt in options_list:
        opt_norm = _normalize(opt)
        if opt_norm == text_norm:
            return opt

    # 2) exact phrase inside text
    for opt in options_list:
        opt_norm = _normalize(opt)
        if len(opt_norm) >= 4 and re.search(rf"\b{re.escape(opt_norm)}\b", text_norm):
            return opt

    # 3) strong token match (all meaningful words must be present)
    text_words = set(re.findall(r"[a-zA-Z]+", text_norm))
    for opt in options_list:
        opt_norm = _normalize(opt)
        opt_words = [w for w in re.findall(r"[a-zA-Z]+", opt_norm) if len(w) > 2]

        if not opt_words:
            continue

        # Require all important words to exist
        if all(w in text_words for w in opt_words):
            return opt

    # 4) safer fuzzy fallback
    scored = []
    for opt in options_list:
        score = _similar(text_norm, opt)
        scored.append((score, opt))

    scored.sort(key=lambda x: -x[0])

    if scored and scored[0][0] >= cutoff:
        # Avoid bad near-collisions like "ice cream" -> "iced coffee"
        top_score, top_opt = scored[0]
        top_words = set(re.findall(r"[a-zA-Z]+", _normalize(top_opt)))
        overlap = len(text_words & top_words)

        # Require at least one strong word overlap, preferably 2 for multiword items
        if len(top_words) == 1 and overlap >= 1:
            return top_opt
        if len(top_words) >= 2 and overlap >= 2:
            return top_opt

    return None


# =========================================================
# MENU HELPERS
# =========================================================


def _get_categories():
    return list(MENU.keys())


def _get_subcategories(category):
    return list(MENU.get(category, {}).keys())


def _get_items(category, subcategory):
    return list(MENU.get(category, {}).get(subcategory, {}).keys())


def _get_item_options(category, subcategory, item):
    return MENU.get(category, {}).get(subcategory, {}).get(item, [])


def _all_subcategories():
    subs = []
    for cat in MENU:
        subs.extend(MENU[cat].keys())
    return list(set(subs))


def _all_items():
    items = []
    for cat in MENU:
        for sub in MENU[cat]:
            items.extend(MENU[cat][sub].keys())
    return list(set(items))


def _find_category_from_anywhere(text):
    return _match(text, _get_categories(), cutoff=0.52)


def _find_subcategory_from_anywhere(text):
    return _match(text, _all_subcategories(), cutoff=0.52)


def _find_item_from_anywhere(text):
    return _match(text, _all_items(), cutoff=0.70)


def _find_category_for_subcategory(subcategory):
    for cat in MENU:
        if subcategory in MENU[cat]:
            return cat
    return None


def _find_category_and_subcategory_for_item(item):
    for cat in MENU:
        for sub in MENU[cat]:
            if item in MENU[cat][sub]:
                return cat, sub
    return None, None


# =========================================================
# INTENT / FLOW HELPERS
# =========================================================


def _is_go_back(text):
    text = _normalize(text)
    return any(
        x in text
        for x in [
            "go back",
            "back",
            "previous",
            "start again",
            "restart",
            "change category",
            "show categories",
            "main menu",
        ]
    )


def _is_menu_request(text):
    text = _normalize(text)
    triggers = [
        "menu",
        "show menu",
        "what do you have",
        "what is available",
        "what can i order",
        "categories",
        "options",
        "menu please",
        "what are the menus",
        "menus present",
    ]
    return any(t in text for t in triggers)


def _is_more_order(text):
    text = _normalize(text)
    return any(
        x in text
        for x in [
            "yes",
            "yeah",
            "yep",
            "more",
            "another",
            "something else",
            "add more",
            "one more",
            "continue",
        ]
    )


def _is_done(text):
    text = _normalize(text)
    return any(
        x in text
        for x in ["no", "done", "nothing", "that's all", "thats all", "finish", "nope"]
    )


def _is_exit(text):
    text = _normalize(text)
    return any(x in text for x in ["bye", "goodbye", "end call", "exit"])


def _format_price(p):
    return f"{p:.2f}".rstrip("0").rstrip(".")


def _cart_total():
    return sum(item["price"] for item in state["cart"])


def _cart_summary():
    parts = []
    for item in state["cart"]:
        if item.get("size"):
            parts.append(
                f"{item['item']} {item['size']} - {_format_price(item['price'])} AED"
            )
        else:
            parts.append(f"{item['item']} - {_format_price(item['price'])} AED")
    return ", ".join(parts)


def _add_to_cart(item_name, price, size=None):
    state["cart"].append({"item": item_name, "size": size, "price": price})


# =========================================================
# FLOW PROMPTS
# =========================================================


def _ask_categories():
    cats = _get_categories()
    return f"We have {_spoken_join(cats)}. Which category would you like to explore?"


def _ask_subcategories(category):
    subs = _get_subcategories(category)
    return f"For {category} we have {_spoken_join(subs)}. Which one would you like?"


def _ask_items(category, subcategory):
    items = _get_items(category, subcategory)
    return (
        f"In {subcategory} we have {_spoken_join(items)}. What would you like to order?"
    )


def _ask_sizes_for_pending():
    sizes = [opt["size"] for opt in state["pending_sizes"]]
    return f"Would you like {_spoken_join(sizes)}?"


# =========================================================
# VALIDATION / NAME MEMORY
# =========================================================


def _extract_name(text: str) -> str | None:
    if not text:
        return None

    lowered = text.lower().strip()
    cleaned = re.sub(r"[^\w\s'\-]", " ", lowered)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    intro_patterns = [
        "hello my name is",
        "my name is",
        "name is",
        "hello i am",
        "i am",
        "i'm",
        "hello this is",
        "this is",
        "call me",
        "order for",
    ]

    candidate = None
    for intro in intro_patterns:
        if cleaned.startswith(intro):
            candidate = cleaned[len(intro) :].strip()
            break

    if not candidate:
        return None

    candidate = re.sub(r"[^a-zA-Z\s'\-]", "", candidate).strip()
    candidate = re.sub(r"\s+", " ", candidate)

    if len(candidate) < 2:
        return None

    words = candidate.split()
    if len(words) > 4:
        return None

    banned_words = {
        "interested",
        "coffee",
        "cream",
        "menu",
        "pickup",
        "dine",
        "cold",
        "drink",
        "drinks",
        "ice",
        "iced",
        "waffle",
        "dessert",
        "hot",
        "show",
        "want",
        "today",
        "tomorrow",
        "now",
        "later",
        "yes",
        "no",
    }

    if any(w in banned_words for w in words):
        return None

    return " ".join(w.capitalize() for w in words)


def _is_valid_phone(phone: str) -> bool:
    phone = re.sub(r"[^\d+]", "", phone.strip())
    digits_only = phone[1:] if phone.startswith("+") else phone
    return digits_only.isdigit() and 8 <= len(digits_only) <= 15


def _is_valid_people_count(text: str) -> bool:
    return text.strip().isdigit() and 1 <= int(text.strip()) <= 50


def _is_valid_datetime(text: str) -> bool:
    text = text.strip()
    return len(text) >= 3 and _normalize(text) not in {
        "asdf",
        "test",
        "nothing",
        "idk",
        "don't know",
        "dont know",
    }


# =========================================================
# EARLY WARMUP / INCOMPLETE INTENT GUARDS
# =========================================================


def _looks_like_early_warmup_input(text: str) -> bool:
    t = _normalize(text)
    warmup_signals = [
        "hello",
        "hi",
        "hey",
        "yeah",
        "yes",
        "hii",
        "can you hear me",
        "myself",
        "this side",
        "i am",
        "my name",
        "name is",
        "here",
        "speaking",
    ]
    if any(x in t for x in warmup_signals):
        return True
    if len(t.split()) <= 4:
        return True
    return False


def _is_incomplete_order_leadin(text: str) -> bool:
    t = _normalize(text)
    bad_phrases = [
        "i would like",
        "i want",
        "can i have",
        "let me see",
        "show me",
        "maybe",
        "something",
        "that one",
        "this one",
        "one please",
        "uh",
        "um",
        "hmm",
        "hmmm",
        "wait",
        "hold on",
    ]
    return t in bad_phrases


# =========================================================
# NATURAL FALLBACK / RECOMMENDATION
# =========================================================


def _build_system_prompt():
    return """
You are Alex, a friendly female voice assistant for Crumbs and Cream cafe.

Your role:
- Help users browse the menu naturally
- Understand corrections and category changes
- Stay conversational and human
- Never invent menu items or prices
- Never override business logic

Rules:
- Keep replies short and natural for voice
- If user is unclear, ask one short clarification question
- If user asks for recommendation, recommend only from menu context
- If user changes their mind, adapt naturally
- Never sound robotic or overly formal
- No markdown, no bullets, no emojis
""".strip()


def _recommend_with_context():
    if state["category"] and state["subcategory"]:
        items = _get_items(state["category"], state["subcategory"])
        picks = items[:3]
        if picks:
            return f"If you'd like, I can suggest {', '.join(picks[:-1]) + ' or ' + picks[-1] if len(picks) > 1 else picks[0]}."
    elif state["category"]:
        subs = _get_subcategories(state["category"])
        picks = subs[:3]
        if picks:
            return f"In {state['category']}, you could try {', '.join(picks[:-1]) + ' or ' + picks[-1] if len(picks) > 1 else picks[0]}."
    else:
        cats = _get_categories()[:4]
        if cats:
            return f"If you'd like, I can show you {', '.join(cats[:-1]) + ' or ' + cats[-1] if len(cats) > 1 else cats[0]}."
    return "I can help you explore the menu."


def _favorite_reply():
    if state["category"] and state["subcategory"]:
        items = _get_items(state["category"], state["subcategory"])
        if items:
            top = items[:2]
            if len(top) == 1:
                return f"A popular choice here is {top[0]}. Would you like to try it?"
            return f"A lot of people like {top[0]} and {top[1]}. Would you like one of those?"

    if state["category"]:
        subs = _get_subcategories(state["category"])
        if subs:
            return f"In {state['category']}, a good place to start is {subs[0]}. Would you like to explore that?"

    cats = _get_categories()
    if cats:
        return f"A lot of people enjoy {cats[0]} and {cats[1] if len(cats) > 1 else cats[0]}. What sounds good to you?"

    return "I can suggest something popular if you'd like."


def _natural_clarify(text):
    if state["step"] == "size" and state["pending_sizes"]:
        return _ask_sizes_for_pending()
    if state["step"] == "item" and state["category"] and state["subcategory"]:
        return f"Sure — which item would you like from {state['subcategory']}?"
    if state["step"] in ["category", "subcategory"]:
        return _ask_categories()
    return "Sorry, I didn’t catch that clearly. What would you like today?"


def _llm_natural_interpretation(user_text: str, llm_fn=None) -> str | None:
    if not llm_fn:
        return None
    try:
        return llm_fn(_build_system_prompt(), user_text, state)
    except Exception:
        return None


# =========================================================
# FLEXIBLE NAVIGATION ROUTER
# =========================================================


def _resolve_user_navigation(text: str):
    raw = text.strip()
    norm = _normalize(raw)
    stripped = _semantic_guard_replacements(_strip_correction_prefixes(norm))

    if _is_incomplete_order_leadin(stripped):
        return {"type": "unknown", "confidence": 0.0}

    current_step = state["step"]

    if current_step == "size" and state["pending_sizes"]:
        sizes = [opt["size"] for opt in state["pending_sizes"]]
        size = _match(stripped, sizes)
        if size:
            return {"type": "select_size", "size": size, "confidence": 0.95}

    cat = _find_category_from_anywhere(stripped)
    sub = _find_subcategory_from_anywhere(stripped)
    item = _find_item_from_anywhere(stripped)

    if _contains_correction(norm):
        if item:
            return {"type": "correction_item", "item": item, "confidence": 0.95}
        if sub:
            return {
                "type": "correction_subcategory",
                "subcategory": sub,
                "confidence": 0.92,
            }
        if cat:
            return {"type": "correction_category", "category": cat, "confidence": 0.90}

    if _contains_recommendation_request(norm):
        return {
            "type": "recommendation",
            "category": cat,
            "subcategory": sub,
            "item": item,
            "confidence": 0.85,
        }

    if _contains_browse_style_request(norm):
        # IMPORTANT:
        # For browse/explore language, prefer category/subcategory navigation
        # before direct item selection.
        if sub:
            return {
                "type": "switch_subcategory",
                "subcategory": sub,
                "confidence": 0.90,
            }
        if cat:
            return {"type": "switch_category", "category": cat, "confidence": 0.88}
        if item:
            return {"type": "select_item", "item": item, "confidence": 0.82}

    # If user sounds like ordering, allow item first
    orderish_phrases = [
        "order",
        "i want",
        "i would like",
        "can i have",
        "give me",
        "add",
    ]

    if any(p in norm for p in orderish_phrases):
        if item:
            return {"type": "select_item", "item": item, "confidence": 0.90}
    if sub:
        return {"type": "switch_subcategory", "subcategory": sub, "confidence": 0.88}
    if cat:
        return {"type": "switch_category", "category": cat, "confidence": 0.86}

    # Otherwise prefer navigation first
    if sub:
        return {"type": "switch_subcategory", "subcategory": sub, "confidence": 0.88}
    if cat:
        return {"type": "switch_category", "category": cat, "confidence": 0.86}
    if item:
        return {"type": "select_item", "item": item, "confidence": 0.80}

    return {"type": "unknown", "confidence": 0.0}


# =========================================================
# APPLY FLEXIBLE NAVIGATION
# =========================================================


def _apply_navigation(intent: dict, raw_text: str):
    t = intent.get("type")

    if t in ["switch_category", "correction_category"]:
        category = intent.get("category")
        if not category:
            return None
        state["category"] = category
        state["subcategory"] = None
        state["pending_item"] = None
        state["pending_sizes"] = []

        subs = _get_subcategories(category)
        if len(subs) == 1:
            state["subcategory"] = subs[0]
            state["step"] = "item"
            return _ask_items(category, subs[0])

        state["step"] = "subcategory"
        return f"In {category} we have {_spoken_join(subs)}. Which one would you like to explore?"

    if t in ["switch_subcategory", "correction_subcategory"]:
        sub = intent.get("subcategory")
        if not sub:
            return None
        cat_for_sub = _find_category_for_subcategory(sub)
        if not cat_for_sub:
            return None

        state["category"] = cat_for_sub
        state["subcategory"] = sub
        state["pending_item"] = None
        state["pending_sizes"] = []
        state["step"] = "item"
        return f"In {sub} we have {_spoken_join(_get_items(cat_for_sub, sub))}. What would you like to order?"

    if t in ["select_item", "correction_item"]:
        item = intent.get("item")
        if not item:
            return None
        cat_for_item, sub_for_item = _find_category_and_subcategory_for_item(item)
        if not cat_for_item or not sub_for_item:
            return None

        state["category"] = cat_for_item
        state["subcategory"] = sub_for_item

        options = _get_item_options(cat_for_item, sub_for_item, item)

        if len(options) == 1:
            price = options[0]["price"]
            size = options[0]["size"]
            spoken_size = (
                size
                if size and size.lower() not in ["regular", "default", "standard"]
                else None
            )
            _add_to_cart(item, price, spoken_size)
            state["step"] = "more"
            return f"The {item} is {_format_price(price)} AED. Added to your order. Would you like anything else?"

        state["pending_item"] = item
        state["pending_sizes"] = options
        state["step"] = "size"
        return _ask_sizes_for_pending()

    if t == "select_size":
        chosen_size = intent.get("size")
        if not chosen_size or not state["pending_item"]:
            return None

        chosen = next(
            (opt for opt in state["pending_sizes"] if opt["size"] == chosen_size), None
        )
        if not chosen:
            return None

        _add_to_cart(state["pending_item"], chosen["price"], chosen["size"])
        item_name = state["pending_item"]
        price = chosen["price"]

        state["pending_item"] = None
        state["pending_sizes"] = []
        state["step"] = "more"

        return f"The {item_name} {chosen_size} is {_format_price(price)} AED. Added to your order. Would you like anything else?"

    if t == "recommendation":
        if intent.get("subcategory"):
            sub = intent["subcategory"]
            cat = _find_category_for_subcategory(sub)
            if cat:
                state["category"] = cat
                state["subcategory"] = sub
                items = _get_items(cat, sub)[:4]
                if items:
                    return f"In {sub}, some good options are {_spoken_join(items)}. Would you like one of those?"
        elif intent.get("category"):
            cat = intent["category"]
            subs = _get_subcategories(cat)[:4]
            if subs:
                state["category"] = cat
                state["subcategory"] = None
                state["step"] = "subcategory"
                return f"In {cat}, we have {_spoken_join(subs)}. Which one sounds good to you?"
        return _recommend_with_context()

    if t == "smalltalk":
        if state["step"] == "more":
            return "Would you like anything else?"
        if state["step"] == "size" and state["pending_sizes"]:
            return _ask_sizes_for_pending()
        if state["step"] == "item" and state["category"] and state["subcategory"]:
            return _ask_items(state["category"], state["subcategory"])
        return "Sure — what would you like to explore?"

    return None


# =========================================================
# MAIN HANDLER
# =========================================================


def handleUserInput(user_text: str, llm_fn=None) -> str:
    global state

    if not user_text or not user_text.strip():
        return "Sorry, I didn't catch that. Could you please repeat?"

    text = user_text.strip()
    state["turn_count"] = state.get("turn_count", 0) + 1

    raw_norm = _normalize(text)
    norm = _strip_correction_prefixes(raw_norm)

    # EXIT
    if _is_exit(norm):
        return "Thank you for calling Crumbs and Cream. Have a lovely day!"

    # SOFT NAME MEMORY
    if state["step"] not in [
        "details_name",
        "details_phone",
        "details_people",
        "details_type",
        "details_datetime",
        "confirm",
    ]:
        extracted_name = _extract_name(text)
        if extracted_name:
            if not state.get("name"):
                state["name"] = extracted_name
                state["greeted_once"] = True
                if state["step"] == "greeting":
                    state["step"] = "category"
                return f"Nice to meet you, {extracted_name}! What would you like today?"
            else:
                state["name"] = extracted_name

    # GREETING / SMALL-TALK
    if _contains_greeting(raw_norm) and len(raw_norm.split()) <= 6:
        state["greeted_once"] = True
        if state["step"] == "greeting":
            state["step"] = "category"
            return "Hi there! What would you like today?"
        if state["step"] == "category":
            return "Hi there! " + _ask_categories()
        return "Hello! What would you like today?"

    # EARLY WARM-UP / FIRST USER TURN
    if not state.get("greeted_once"):
        state["greeted_once"] = True

        if _is_menu_request(norm):
            state["step"] = "category"
            return _ask_categories()

        if state["step"] == "greeting":
            state["step"] = "category"

        if _looks_like_early_warmup_input(text):
            if state.get("name"):
                return f"Hi {state['name']}! What would you like today?"
            return "Hi there! What would you like today?"

        return "Hi there! What would you like today?"

    # FAVORITE / POPULAR / NATURAL REACTION
    if _contains_favorite_question(norm):
        return _favorite_reply()

    if _contains_chitchat(norm):
        if state["step"] == "item" and state["category"] and state["subcategory"]:
            return f"Sure. In {state['subcategory']} we have {_spoken_join(_get_items(state['category'], state['subcategory']))}. What would you like to try?"
        if state["step"] == "subcategory" and state["category"]:
            return f"Sure. In {state['category']} we have {_spoken_join(_get_subcategories(state['category']))}. Which one would you like to explore?"

    # GLOBAL ORDER TYPE CORRECTION
    if state["step"] in [
        "details_type",
        "details_people",
        "details_name",
        "details_phone",
        "details_datetime",
        "confirm",
    ]:
        detected_order_type = _extract_order_type(norm)

        if detected_order_type == "pickup":
            state["order_type"] = "pickup"
            state["people_count"] = None

            if state["step"] == "details_type" or state["step"] == "details_people":
                if state.get("name"):
                    state["step"] = "details_phone"
                    return f"Got it — this will be for pickup. I’ll place that under {state['name']}. Please provide your contact number."
                else:
                    state["step"] = "details_name"
                    return "Got it — this will be for pickup. May I have your name for the order?"

        if detected_order_type == "dine-in":
            state["order_type"] = "dine-in"

            if state["step"] in [
                "details_type",
                "details_name",
                "details_phone",
                "details_datetime",
            ]:
                state["step"] = "details_people"
                return "Got it — this will be for dine-in. For how many people?"

    # Global menu / restart / go-back
    if _is_menu_request(norm):
        state["step"] = "category"
        state["category"] = None
        state["subcategory"] = None
        state["pending_item"] = None
        state["pending_sizes"] = []
        return _ask_categories()

    if _is_go_back(norm):
        state["step"] = "category"
        state["category"] = None
        state["subcategory"] = None
        state["pending_item"] = None
        state["pending_sizes"] = []
        return _ask_categories()

    # FLEXIBLE NAVIGATION LAYER
    if state["step"] not in [
        "details_name",
        "details_phone",
        "details_people",
        "details_type",
        "details_datetime",
        "confirm",
    ]:
        nav_intent = _resolve_user_navigation(text)
        state["last_user_intent"] = nav_intent
        nav_reply = _apply_navigation(nav_intent, text)
        if nav_reply:
            return nav_reply

    # STRUCTURED DETERMINISTIC FLOW
    if state["step"] == "category":
        category = _match(norm, _get_categories())
        if not category:
            if _contains_recommendation_request(norm):
                return _recommend_with_context()
            return "Sure — which category would you like to explore?"

        state["category"] = category
        subs = _get_subcategories(category)

        if len(subs) == 1:
            state["subcategory"] = subs[0]
            state["step"] = "item"
            return _ask_items(category, subs[0])

        state["step"] = "subcategory"
        return _ask_subcategories(category)

    if state["step"] == "subcategory":
        subs = _get_subcategories(state["category"])
        sub = _match(norm, subs)
        if not sub:
            if _contains_recommendation_request(norm):
                return _recommend_with_context()
            return f"Sure — which one would you like from {state['category']}?"

        state["subcategory"] = sub
        state["step"] = "item"
        return _ask_items(state["category"], sub)

    if state["step"] == "item":
        if _is_incomplete_order_leadin(norm):
            return f"Sure — which item would you like from {state['subcategory']}?"

        items = _get_items(state["category"], state["subcategory"])
        item = _match(norm, items, cutoff=0.72)

        if not item:
            if _contains_recommendation_request(norm):
                return _recommend_with_context()
            return f"Sorry, I didn’t catch the item. Which one would you like from {state['subcategory']}?"

        options = _get_item_options(state["category"], state["subcategory"], item)

        if len(options) == 1:
            price = options[0]["price"]
            size = options[0]["size"]
            spoken_size = (
                size
                if size and size.lower() not in ["regular", "default", "standard"]
                else None
            )
            _add_to_cart(item, price, spoken_size)
            state["step"] = "more"
            return f"The {item} is {_format_price(price)} AED. Added to your order. Would you like anything else?"

        state["pending_item"] = item
        state["pending_sizes"] = options
        state["step"] = "size"
        return _ask_sizes_for_pending()

    if state["step"] == "size":
        sizes = [opt["size"] for opt in state["pending_sizes"]]
        chosen_size = _match(norm, sizes)
        if not chosen_size:
            return _ask_sizes_for_pending()

        chosen = next(
            (opt for opt in state["pending_sizes"] if opt["size"] == chosen_size), None
        )
        if not chosen:
            return "Sorry, I didn't catch the size. Could you repeat?"

        _add_to_cart(state["pending_item"], chosen["price"], chosen["size"])
        item_name = state["pending_item"]
        price = chosen["price"]

        state["pending_item"] = None
        state["pending_sizes"] = []
        state["step"] = "more"

        return f"The {item_name} {chosen_size} is {_format_price(price)} AED. Added to your order. Would you like anything else?"

    if state["step"] == "more":
        if _is_more_order(raw_norm):
            state["step"] = "category"
            return _ask_categories()

        if _is_done(raw_norm):
            state["step"] = "details_type"
            return "Will that be pickup or dine-in?"

        possible_category = _match(norm, _get_categories())
        if possible_category:
            state["step"] = "category"
            return handleUserInput(text, llm_fn)

        if _contains_recommendation_request(norm):
            return _recommend_with_context()

        return "Would you like anything else?"

    if state["step"] == "details_type":
        detected_order_type = _extract_order_type(norm)

        if detected_order_type == "pickup":
            state["order_type"] = "pickup"
            if state.get("name"):
                state["step"] = "details_phone"
                return f"I’ll place that under {state['name']}. Please provide your contact number."
            state["step"] = "details_name"
            return "May I have your name for the order?"

        if detected_order_type == "dine-in":
            state["order_type"] = "dine-in"
            state["step"] = "details_people"
            return "For how many people?"

        return "Please tell me if this order is for pickup or dine-in."

    if state["step"] == "details_people":
        if not _is_valid_people_count(text):
            detected_order_type = _extract_order_type(norm)
            if detected_order_type == "pickup":
                state["order_type"] = "pickup"
                state["people_count"] = None
                if state.get("name"):
                    state["step"] = "details_phone"
                    return f"Got it — changing this to pickup. I’ll place that under {state['name']}. Please provide your contact number."
                state["step"] = "details_name"
                return "Got it — changing this to pickup. May I have your name for the order?"

            return "Please tell me the number of people, like 2 or 4."

        state["people_count"] = int(text)

        if state.get("name"):
            state["step"] = "details_phone"
            return f"I’ll place that under {state['name']}. Please provide your contact number."

        state["step"] = "details_name"
        return "May I have your name for the order?"

    if state["step"] == "details_name":
        extracted_name = _extract_name(text)
        if not extracted_name:
            if len(text.split()) <= 4 and re.search(r"[A-Za-z]", text):
                extracted_name = " ".join(
                    w.capitalize()
                    for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", text)[:4]
                )

        if not extracted_name:
            return "Please tell me a valid name for the order."

        state["name"] = extracted_name
        state["step"] = "details_phone"
        return "Please provide your contact number."

    if state["step"] == "details_phone":
        if not _is_valid_phone(text):
            return "Please provide a valid contact number."

        state["phone"] = re.sub(r"[^\d+]", "", text.strip())
        state["step"] = "details_datetime"
        return "What date and time would you prefer?"

    if state["step"] == "details_datetime":
        if not _is_valid_datetime(text):
            return "Please tell me your preferred date and time."

        state["datetime"] = text.strip()
        state["step"] = "confirm"

        summary = _cart_summary()
        total = _format_price(_cart_total())

        extra = ""
        if state["order_type"] == "dine-in" and state["people_count"]:
            extra = f" for {state['people_count']} people"

        return (
            f"Here is your order. {summary}. Total is {total} AED. "
            f"It is for {state['order_type']}{extra} on {state['datetime']} "
            f"under the name {state['name']}. Shall I confirm your order?"
        )

    if state["step"] == "confirm":
        if any(x in norm for x in ["yes", "confirm", "confirmed", "okay", "ok"]):
            final_response = "Your order has been confirmed. Thank you for ordering with Crumbs and Cream."
            state = reset_state()
            return final_response

        if any(x in norm for x in ["no", "change", "edit", "modify"]):
            state["step"] = "more"
            return "Sure. What would you like to change?"

        return "Shall I confirm your order?"

    if _contains_recommendation_request(norm):
        return _recommend_with_context()

    llm_reply = _llm_natural_interpretation(user_text, llm_fn)
    if llm_reply:
        return llm_reply

    return _natural_clarify(user_text)


def get_menu_snapshot():
    return MENU
