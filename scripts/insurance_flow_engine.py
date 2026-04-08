"""
insurance_flow_engine.py — Insurance voice agent conversation engine.

Architecture (hybrid flow + LLM):
  - Flow  controls WHAT to ask  (structured sequence + branching rules)
  - OpenAI controls HOW to say it (natural, voice-friendly phrasing)

Usage:
    from insurance_flow_engine import InsuranceSession

    session = InsuranceSession.from_json()
    greeting = session.get_greeting()           # step 1.1 text

    # Per voice turn:
    reply = session.process_turn_naturalized(user_text, brain=get_brain(), history=[...])
    if session.ended:
        # call is over
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional

from openai_brain import OpenAIBrain
from insurance_prompt import (
    INSURANCE_SYSTEM_PROMPT,
    NATURALIZE_PROMPT,
    AGENT_NAME,
    build_naturalize_prompt,
    format_options_for_voice,
    is_negative_intent,
    is_affirmative_intent,
    is_ambiguous_intent,
    should_exit_flow,
    REFUSAL_FAREWELL,
    AMBIGUOUS_CONSENT_CLARIFICATION,
)
from field_extractors import extract_structured_answer, classify_intent, STEP_FIELD_DEFS
from session_store import SessionStore

logger = logging.getLogger(__name__)

# ── Default flow path ─────────────────────────────────────────────────────────
_DEFAULT_FLOW_PATH = (
    Path(__file__).parent.parent.parent / "data" / "insurance_flow.json"
)

# Step sequence and loop ranges are derived from insurance_flow.json at session
# init (see InsuranceSession._sequence) — no hardcoded lists here.

# Maximum validation-failure retries per step before the agent stores null
# and advances. Users get this many re-asks after the first invalid answer.
_MAX_FIELD_RETRIES: int = 3

# ── Virtual steps not present in Excel (derived from conditions column) ────────
_VIRTUAL_STEPS: dict[str, dict] = {
    "2.2a": {
        "id": "2.2a",
        "block": "DRIVER",
        "question": "Could you tell me who is the registered owner and primary operator of the vehicle?",
        "options": [],
        "expected": None,
        "conditions": "Follow-up if 2.2 = No",
        "branch_logic": None,
        "in_driver_loop": True,
        "in_vehicle_loop": False,
    },
    "1.4_desc": {
        "id": "1.4_desc",
        "block": "START",
        "question": "Could you briefly describe those modifications for me?",
        "options": [],
        "expected": None,
        "conditions": "Follow-up if 1.4 = Yes",
        "branch_logic": None,
        "in_driver_loop": False,
        "in_vehicle_loop": False,
    },
}


# ── Flow loader ───────────────────────────────────────────────────────────────


def load_flow(path: Optional[Path] = None) -> dict:
    """Load insurance_flow.json. Falls back to default path."""
    p = path or _DEFAULT_FLOW_PATH
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


# ── Session ───────────────────────────────────────────────────────────────────


@dataclass
class InsuranceSession:
    """Manages state for a single insurance intake voice call.

    Attributes (public, useful for logging):
        current_step_id   — ID of the step currently being asked
        answers           — dict mapping answer_key → user answer string
        driver_count      — parsed from 2.1 answer
        vehicle_count     — parsed from 3.1 answer
        ended             — True once the call flow is complete/terminated
        end_reason        — "completed" | "no_consent" | "error"
    """

    flow: dict = field(repr=False)

    def __post_init__(self) -> None:
        self._step_map: dict[str, dict] = {s["id"]: s for s in self.flow["steps"]}
        self._step_map.update(_VIRTUAL_STEPS)
        # Canonical step order from JSON — single source of truth for sequencing.
        self._sequence: list[str] = [s["id"] for s in self.flow["steps"]]

        self.current_step_id: str = "1.1"
        self.answers: dict[str, str] = {}
        self._clarification_counts: dict[str, int] = {}
        self._best_extracted: dict[str, Any] = {}  # best valid value seen per step

        self.driver_count: int = 0
        self.current_driver_idx: int = 0  # 0-based
        self.vehicle_count: int = 0
        self.current_vehicle_idx: int = 0  # 0-based
        self.in_driver_loop: bool = False
        self.in_vehicle_loop: bool = False

        self._skip_steps: set[str] = set()

        self.collected_data: dict[str, Any] = {}
        self.session_id: str = uuid.uuid4().hex[:8]
        self._store: SessionStore = SessionStore(self.session_id)

        self.ended: bool = False
        self.end_reason: str = ""

    @classmethod
    def from_json(cls, path: Optional[Path] = None) -> "InsuranceSession":
        """Create a session from the parsed flow JSON."""
        return cls(flow=load_flow(path))

    # ── Public API ─────────────────────────────────────────────────────────

    def get_greeting(self) -> str:
        """Return the opening statement (step 1.1) with agent name substituted."""
        step = self._step_map.get("1.1", {})
        raw = step.get(
            "question",
            f"Hi, I'm {AGENT_NAME}. Is it okay if I collect a few details for your insurance quote?",
        )
        return raw.replace("<Voice_Agent_Name>", AGENT_NAME)

    def get_current_step(self) -> Optional[dict]:
        return self._step_map.get(self.current_step_id)

    def process_turn_naturalized(
        self,
        user_text: str,
        brain: OpenAIBrain,
        history: Optional[list] = None,
    ) -> str:
        """Process user input, advance the flow, return the next response as
        naturally-phrased text ready for TTS.

        Returns a farewell string when the call ends (session.ended will be True).
        """
        if self.ended:
            return "Thank you — our conversation has concluded. Goodbye!"

        history = history or []

        # 0. Empty / whitespace STT output — retry the same step without advancing.
        if not user_text or not user_text.strip():
            logger.info(
                "[Flow] step=%-6s | empty STT input — retrying", self.current_step_id
            )
            return self._empty_input_retry(self.get_current_step())

        # 1. Hard-stop check — any turn can trigger an immediate exit
        if should_exit_flow(user_text):
            self.ended = True
            self.end_reason = "user_exit"
            return self._naturalize_farewell(REFUSAL_FAREWELL, brain, history)

        # 2. Classify and record the answer against the current step
        step = self.get_current_step()
        if step:
            # 1a. Detect clarification/objection/off-topic BEFORE storing anything.
            intent = classify_intent(user_text)
            if intent == "clarification":
                return self._handle_clarification_question(
                    user_text, step, brain, history
                )
            if intent == "objection":
                return self._handle_objection(step, brain, history)
            if intent == "off_topic":
                return self._handle_off_topic(step, brain, history)

            # intent == "answer" — record and validate.
            key = self._answer_key(self.current_step_id)
            self.answers[key] = user_text.strip()
            logger.info(
                "[Flow] step=%-6s | answer=%r", self.current_step_id, user_text[:60]
            )

            # 1b. Extract, normalise, and validate
            options = step.get("options") or []
            is_valid, extracted_value, err_reason = extract_structured_answer(
                self.current_step_id, user_text, options=options
            )
            attempt = self._clarification_counts.get(self.current_step_id, 0)
            # Track the best validated extraction across retries (fallback reference).
            if is_valid:
                self._best_extracted[self.current_step_id] = extracted_value

            if not is_valid and attempt < _MAX_FIELD_RETRIES:
                del self.answers[key]  # discard invalid answer — do not advance
                self._clarification_counts[self.current_step_id] = attempt + 1
                return self._naturalize_clarification(
                    step, err_reason, brain, history, attempt=attempt
                )

            # Resolve the value to persist:
            #   valid answer  → store the normalised extraction
            #   retries exhausted → use best previously-validated value, else None
            field_def = STEP_FIELD_DEFS.get(self.current_step_id, {})
            field_name = field_def.get("field_name")
            if field_name:
                if self.in_driver_loop:
                    data_key = f"driver_{self.current_driver_idx + 1}_{field_name}"
                elif self.in_vehicle_loop:
                    data_key = f"vehicle_{self.current_vehicle_idx + 1}_{field_name}"
                else:
                    data_key = field_name
                if is_valid:
                    store_val = extracted_value
                else:
                    # Retries exhausted — fall back to best prior valid value, else null.
                    store_val = self._best_extracted.get(self.current_step_id)
                    logger.warning(
                        "[Flow] step=%-6s | retries exhausted — storing %s",
                        self.current_step_id,
                        repr(store_val),
                    )
                    self.answers.pop(key, None)  # drop raw garbage from answers
                self.collected_data[data_key] = store_val
                self._store.update(data_key, store_val)

            # Clear retry state for this step.
            self._clarification_counts.pop(self.current_step_id, None)
            self._best_extracted.pop(self.current_step_id, None)

        # 3. Apply branching logic → determine next action
        action = self._apply_branch(self.current_step_id, user_text)
        logger.debug("[Flow] branch result: %s", action)

        # 4. Act on the action
        if action["action"] == "clarify_consent":
            # Do NOT advance the step — re-ask with soft clarification
            del self.answers[
                self._answer_key(self.current_step_id)
            ]  # discard ambiguous answer
            return self._naturalize_farewell(
                AMBIGUOUS_CONSENT_CLARIFICATION, brain, history
            )

        if action["action"] == "end_call":
            self.ended = True
            self.end_reason = action.get("reason", "call_ended")
            farewell = action.get(
                "message",
                "Thank you for your time. Goodbye!",
            )
            return self._naturalize_farewell(farewell, brain, history)

        if action["action"] == "done":
            self.ended = True
            self.end_reason = "completed"
            farewell_step = self._step_map.get("7.2", {})
            farewell_q = farewell_step.get(
                "question", "Thank you so much! We'll be in touch."
            )
            return self._naturalize_farewell(
                farewell_q.replace("<Voice_Agent_Name>", AGENT_NAME), brain, history
            )

        if action["action"] == "continue":
            next_id = action["next_step_id"]
            # Explicit loop-iteration labels take priority; otherwise auto-detect
            # a block transition from the JSON block metadata.
            block_transition = action.get(
                "block_transition"
            ) or self._block_transition_tag(self.current_step_id, next_id)
            self.current_step_id = next_id
            # Sync loop membership from JSON step metadata.
            _ns = self._step_map.get(next_id, {})
            self.in_driver_loop = _ns.get("in_driver_loop", False)
            self.in_vehicle_loop = _ns.get("in_vehicle_loop", False)

            next_step = self._step_map.get(next_id)
            if not next_step:
                logger.error(
                    "[Flow] Step '%s' not in step_map — ending session.", next_id
                )
                self.ended = True
                self.end_reason = "error"
                return "I'm sorry, something went wrong on my end. Thank you for your time. Goodbye!"

            return self._naturalize_question(
                next_step,
                brain,
                history,
                block_transition=block_transition,
            )

        # Fallback
        logger.error("[Flow] Unknown action '%s' — terminating.", action.get("action"))
        self.ended = True
        return "Thank you for your time. Goodbye!"

    def get_state_summary(self) -> dict:
        """Snapshot of current state — useful for debug logging."""
        return {
            "current_step": self.current_step_id,
            "driver_count": self.driver_count,
            "current_driver": self.current_driver_idx + 1
            if self.in_driver_loop
            else None,
            "vehicle_count": self.vehicle_count,
            "current_vehicle": self.current_vehicle_idx + 1
            if self.in_vehicle_loop
            else None,
            "answers_collected": len(self.answers),
            "ended": self.ended,
            "end_reason": self.end_reason,
        }

    # ── Branching logic ────────────────────────────────────────────────────

    def _apply_branch(self, step_id: str, answer: str) -> dict:
        """Return an action dict based on the answer to the given step.

        Dispatches via the step's ``branch_logic`` JSON field so that
        branching rules are driven by the flow definition, not hardcoded
        step-ID literals.  Steps with null branch_logic that need custom
        handling are caught by step-ID fallbacks at the end.

        Action dicts:
          {"action": "continue", "next_step_id": "...", "block_transition": "..."}
          {"action": "end_call",  "reason": "...", "message": "..."}
          {"action": "clarify_consent"}
          {"action": "done"}
        """
        a = answer.strip().lower()
        branch_logic = (self._step_map.get(step_id) or {}).get("branch_logic")

        # ── branch_logic dispatch ─────────────────────────────────────────
        if branch_logic == "consent_check":
            if is_negative_intent(a):
                return {
                    "action": "end_call",
                    "reason": "no_consent",
                    "message": REFUSAL_FAREWELL,
                }
            if is_ambiguous_intent(a) and not is_affirmative_intent(a):
                return {"action": "clarify_consent"}
            return {"action": "continue", "next_step_id": "1.2"}

        if branch_logic == "modifications_check":
            if self._is_yes(a):
                return {"action": "continue", "next_step_id": "1.4_desc"}
            return {"action": "continue", "next_step_id": "2.1"}

        if branch_logic == "driver_loop_start":
            count = self._extract_number(a, default=1)
            self.driver_count = count
            self.in_driver_loop = True
            self.current_driver_idx = 0
            return {"action": "continue", "next_step_id": "2.2"}

        if branch_logic == "registered_owner_check":
            if self._is_no(a):
                return {"action": "continue", "next_step_id": "2.2a"}
            return {"action": "continue", "next_step_id": "2.3"}

        if branch_logic == "applicant_check":
            if self._is_no(a):
                return {"action": "continue", "next_step_id": "2.4a"}
            return {"action": "continue", "next_step_id": "2.5"}

        if branch_logic == "driver_loop_end":
            # Advance to next driver, or leave the driver loop.
            next_idx = self.current_driver_idx + 1
            if next_idx < self.driver_count:
                self.current_driver_idx = next_idx
                return {
                    "action": "continue",
                    "next_step_id": "2.2",
                    "block_transition": f"next_driver_{next_idx + 1}",
                }
            self.in_driver_loop = False
            return {"action": "continue", "next_step_id": "3.1"}

        if branch_logic == "vehicle_loop_start":
            count = self._extract_number(a, default=1)
            self.vehicle_count = count
            self.in_vehicle_loop = True
            self.current_vehicle_idx = 0
            if count == 1:
                self._skip_steps.add(
                    "3.3"
                )  # skip principal-driver Q for single vehicle
                return {"action": "continue", "next_step_id": "3.4"}
            return {"action": "continue", "next_step_id": "3.3"}

        if branch_logic == "principal_driver_check":
            return {"action": "continue", "next_step_id": "3.4"}

        if branch_logic == "ownership_type_check":
            # Prefer the normalised extracted value so that casual phrases like
            # "I'm still paying it off" correctly branch to "financed" rather
            # than accidentally matching "own" in the raw text.
            veh_key = f"vehicle_{self.current_vehicle_idx + 1}_ownership_type"
            ownership = self.collected_data.get(veh_key) or self.collected_data.get(
                "ownership_type"
            )
            if ownership is None:
                # Graceful fallback when extraction failed (e.g. 3rd retry).
                ownership = (
                    "owned"
                    if ("own" in a and "financ" not in a and "leas" not in a)
                    else "other"
                )
            if ownership == "owned":
                self._skip_steps.add("3.5")  # outright owners skip financing Q
                return {"action": "continue", "next_step_id": "3.6"}
            return {"action": "continue", "next_step_id": "3.5"}

        if branch_logic == "vehicle_loop_end":
            # Advance to next vehicle, or leave the vehicle loop.
            next_idx = self.current_vehicle_idx + 1
            if next_idx < self.vehicle_count:
                self.current_vehicle_idx = next_idx
                next_start = "3.3" if self.vehicle_count > 1 else "3.4"
                return {
                    "action": "continue",
                    "next_step_id": next_start,
                    "block_transition": f"next_vehicle_{next_idx + 1}",
                }
            self.in_vehicle_loop = False
            return {"action": "continue", "next_step_id": "4.1"}

        if branch_logic == "confirmation_check":
            if self._is_no(a):
                logger.info(
                    "[Flow] User declined final confirmation — ending politely."
                )
            return {"action": "continue", "next_step_id": "7.2"}

        if branch_logic == "farewell":
            return {"action": "done"}

        # ── Steps with null branch_logic requiring custom handling ─────────
        if step_id == "2.3":
            # Skip 2.4 / 2.4a when there is only one driver
            if self.driver_count <= 1:
                return {"action": "continue", "next_step_id": "2.5"}
            return {"action": "continue", "next_step_id": "2.4"}

        if step_id == "2.2a":
            # Virtual step — always advance to 2.5 (primary driver question)
            return {"action": "continue", "next_step_id": "2.5"}

        if step_id == "1.4_desc":
            # Modification description received — move to driver section
            return {"action": "continue", "next_step_id": "2.1"}

        # ── Default: linear advance through JSON step sequence ────────────
        return {"action": "continue", "next_step_id": self._next_in_sequence(step_id)}

    def _block_transition_tag(self, from_id: str, to_id: str) -> Optional[str]:
        """Return a section-transition label when crossing a JSON block boundary, else None.

        Driven entirely by the ``block`` field in the flow JSON so no strings
        need to be hardcoded here.  Loop-internal transitions (same block,
        different iteration) are NOT detected here — callers supply those
        explicitly via ``block_transition`` in the action dict.
        """
        from_block = self._step_map.get(from_id, {}).get("block", "")
        to_block = self._step_map.get(to_id, {}).get("block", "")
        if from_block and to_block and from_block != to_block:
            return to_block.lower() + "_section"
        return None

    def _next_in_sequence(self, current_id: str) -> str:
        """Return the next step ID in the JSON-defined sequence, skipping marked steps."""
        try:
            idx = self._sequence.index(current_id)
        except ValueError:
            logger.warning(
                "[Flow] '%s' not in JSON sequence — falling back to 7.1", current_id
            )
            return "7.1"

        for i in range(idx + 1, len(self._sequence)):
            nid = self._sequence[i]
            if nid not in self._skip_steps:
                return nid

        return "7.2"

    # ── Naturalization ─────────────────────────────────────────────────────

    def _naturalize_question(
        self,
        step: dict,
        brain: OpenAIBrain,
        history: list,
        block_transition: Optional[str] = None,
    ) -> str:
        """Use OpenAI to rephrase the raw question naturally for voice."""
        raw_q = step["question"].replace("<Voice_Agent_Name>", AGENT_NAME)
        options = step.get("options", [])

        # Build loop context string when inside a loop
        loop_ctx: Optional[str] = None
        if self.in_driver_loop and self.driver_count > 1:
            loop_ctx = (
                f"You are currently collecting information for driver "
                f"{self.current_driver_idx + 1} of {self.driver_count}."
            )
        elif self.in_vehicle_loop and self.vehicle_count > 1:
            loop_ctx = (
                f"You are currently collecting information for vehicle "
                f"{self.current_vehicle_idx + 1} of {self.vehicle_count}."
            )

        # Handle block transition tag for next_driver_ / next_vehicle_
        if block_transition and block_transition.startswith("next_driver_"):
            idx_str = block_transition.split("_")[-1]
            loop_ctx = f"You're now collecting information for driver {idx_str}."
            block_transition = None
        elif block_transition and block_transition.startswith("next_vehicle_"):
            idx_str = block_transition.split("_")[-1]
            loop_ctx = f"You're now collecting information for vehicle {idx_str}."
            block_transition = None

        user_prompt = build_naturalize_prompt(
            raw_q, options, block_transition, loop_ctx
        )

        messages = [
            {
                "role": "system",
                "content": INSURANCE_SYSTEM_PROMPT + "\n\n" + NATURALIZE_PROMPT,
            },
            *history,
            {"role": "user", "content": user_prompt},
        ]

        try:
            reply, usage = brain.chat(messages, temperature=0.4, max_tokens=60)
            logger.info(
                "[Flow] Naturalized step %-6s | lat=%.2fs | tokens=%d/%d | %r",
                step["id"],
                usage["latency_s"],
                usage["prompt_tokens"],
                usage["completion_tokens"],
                (reply or "")[:80],
            )
            return reply.strip() if reply else raw_q
        except Exception as exc:
            logger.error(
                "[Flow] Naturalization failed for step %s: %s", step["id"], exc
            )
            return raw_q  # fall back to raw question text

    def _naturalize_farewell(
        self, farewell_text: str, brain: OpenAIBrain, history: list
    ) -> str:
        """Rephrase the farewell text naturally."""
        messages = [
            {"role": "system", "content": INSURANCE_SYSTEM_PROMPT},
            *history,
            {
                "role": "user",
                "content": (
                    f"Deliver this closing message naturally on a voice call — "
                    f"warm and brief: {farewell_text}"
                ),
            },
        ]
        try:
            reply, _ = brain.chat(messages, temperature=0.4, max_tokens=60)
            return reply.strip() if reply else farewell_text
        except Exception:
            return farewell_text

    # ── Intent-aware response handlers ────────────────────────────────────

    def _handle_clarification_question(
        self,
        user_text: str,
        step: dict,
        brain: OpenAIBrain,
        history: list,
    ) -> str:
        """Answer the caller's clarification/help question then re-ask the same step.

        Flow position is NOT advanced and nothing is stored.
        """
        raw_q = step["question"].replace("<Voice_Agent_Name>", AGENT_NAME)
        options = step.get("options") or []
        opts_str = (
            f" The options are: {format_options_for_voice(options)}." if options else ""
        )
        field_def = STEP_FIELD_DEFS.get(self.current_step_id, {})
        field_type = field_def.get("field_type", "free_text")
        _fmt_hints: dict[str, str] = {
            "phone_number": "We need a ten-digit phone number, for example 416-555-0199.",
            "date": "A date like March 26th 2026 works perfectly.",
            "month_year": "Just month and year is fine, like January 2010.",
            "number": "Just a number, like 1 or 2.",
            "currency": "A dollar amount, like one hundred and sixty thousand.",
            "yes_no": "Just yes or no.",
        }
        fmt_hint = _fmt_hints.get(field_type, "")
        prompt = (
            f"The caller asked: '{user_text}'. "
            f"Answer their question very briefly (one sentence) then re-ask: '{raw_q}'{opts_str} "
            + (f"If it helps, mention: {fmt_hint} " if fmt_hint else "")
            + "Keep it under two short sentences. Sound warm and natural."
        )
        messages = [
            {"role": "system", "content": INSURANCE_SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": prompt},
        ]
        try:
            reply, _ = brain.chat(messages, temperature=0.4, max_tokens=80)
            return reply.strip() if reply else raw_q
        except Exception:
            return raw_q

    def _handle_objection(
        self,
        step: dict,
        brain: OpenAIBrain,
        history: list,
    ) -> str:
        """Acknowledge the caller's resistance, explain why the field matters, re-ask.

        Flow position is NOT advanced and nothing is stored.
        """
        raw_q = step["question"].replace("<Voice_Agent_Name>", AGENT_NAME)
        options = step.get("options") or []
        opts_str = (
            f" The options are: {format_options_for_voice(options)}." if options else ""
        )
        prompt = (
            f"The caller is hesitant about this question. "
            f"Briefly acknowledge their concern, explain this detail helps us get an accurate quote, "
            f"and re-ask gently: '{raw_q}'{opts_str} "
            f"Be warm, empathetic, and keep it to two sentences."
        )
        messages = [
            {"role": "system", "content": INSURANCE_SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": prompt},
        ]
        try:
            reply, _ = brain.chat(messages, temperature=0.4, max_tokens=80)
            return reply.strip() if reply else raw_q
        except Exception:
            return raw_q

    def _handle_off_topic(
        self,
        step: dict,
        brain: OpenAIBrain,
        history: list,
    ) -> str:
        """Gently redirect the caller back to the current question.

        Flow position is NOT advanced and nothing is stored.
        """
        raw_q = step["question"].replace("<Voice_Agent_Name>", AGENT_NAME)
        options = step.get("options") or []
        opts_str = (
            f" The options are: {format_options_for_voice(options)}." if options else ""
        )
        prompt = (
            f"The caller said something that doesn't answer the current question. "
            f"Gently steer them back and re-ask: '{raw_q}'{opts_str} "
            f"One or two short sentences, warm and natural."
        )
        messages = [
            {"role": "system", "content": INSURANCE_SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": prompt},
        ]
        try:
            reply, _ = brain.chat(messages, temperature=0.4, max_tokens=60)
            return reply.strip() if reply else raw_q
        except Exception:
            return raw_q

    _CLARIFICATION_HINTS: ClassVar[dict[str, str]] = {
        "invalid_length": "The number wasn't long enough to be a valid phone number — we need ten digits.",
        "invalid_area_code": "That area code doesn't look right for a Canadian number.",
        "unrecognised_format": "The format wasn't clear.",
        "date_in_past": "That date has already passed — we need a policy start date from today or later.",
        "year_out_of_range": "That year doesn't look right for a Canadian licence date. It should be a past year, not a future one.",
        "not_a_number": "We need a number there.",
        "below_minimum": "That number seems too low.",
        "above_maximum": "That number seems unrealistically high.",
        "no_option_match": "That didn't match the available choices.",
        "unclear_yes_no": "It wasn't clear if that was a yes or a no.",
        "empty_answer": "No answer was detected.",
        # Categorical fields
        "no_marital_match": "You can say single, married, common law, divorced, or widowed.",
        "no_ownership_match": "Is it owned outright, financed, or leased?",
        "no_condition_match": "Is it new, used, or a demo vehicle?",
        "not_a_currency": "Could you give me a dollar amount — like 'a hundred and sixty thousand'?",
    }

    def _naturalize_clarification(
        self,
        step: dict,
        error_reason: str,
        brain: OpenAIBrain,
        history: list,
        attempt: int = 0,
    ) -> str:
        """Re-ask a step naturally when the answer failed validation.

        ``attempt`` (0-based) drives the escalation:
          0 → gentle first correction
          1 → simpler phrasing + concrete example
          2+ → final-attempt warning, very explicit guidance
        """
        raw_q = step["question"].replace("<Voice_Agent_Name>", AGENT_NAME)
        options = step.get("options") or []
        hint = self._CLARIFICATION_HINTS.get(error_reason, "The answer wasn't clear.")
        opts_str = (
            f" Options are: {format_options_for_voice(options)}." if options else ""
        )
        is_last = attempt >= _MAX_FIELD_RETRIES - 1
        if is_last:
            urgency = (
                "This is the final attempt — if still unable to get a valid answer, "
                "this field will be marked as missing and we'll move on. "
            )
        elif attempt == 1:
            urgency = (
                "Simplify the question as much as possible and include "
                "a short concrete example to guide the caller. "
            )
        else:
            urgency = ""
        # Frame the problem correctly: range errors were understood but out of range;
        # format errors were genuinely unclear.
        _RANGE_REASONS = {
            "date_in_past",
            "year_out_of_range",
            "below_minimum",
            "above_maximum",
        }
        if error_reason in _RANGE_REASONS:
            framing = "the previous answer was understood but needs correcting"
        else:
            framing = "the previous answer was unclear"
        prompt = (
            f"{urgency}"
            f"Re-ask this question naturally — {framing}. "
            f"Reason: {hint} "
            f"Question: {raw_q}{opts_str} "
            f"One or two sentences max. Sound human, not robotic."
        )
        messages = [
            {"role": "system", "content": INSURANCE_SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": prompt},
        ]
        try:
            reply, _ = brain.chat(messages, temperature=0.4, max_tokens=80)
            return reply.strip() if reply else raw_q
        except Exception:
            return raw_q

    # ── Helpers ────────────────────────────────────────────────────────────

    def _empty_input_retry(self, step: Optional[dict]) -> str:
        """Return a step-aware retry prompt when STT produces empty/whitespace input.

        Does NOT advance the flow or store anything.
        """
        if step is None:
            return "Sorry, I didn't catch that — could you repeat?"

        field_def = STEP_FIELD_DEFS.get(self.current_step_id, {})
        field_type = field_def.get("field_type", "free_text")
        options = step.get("options") or []

        _HINTS: dict[str, str] = {
            "phone_number": "Sorry, I didn't catch that — you can say the number digit by digit.",
            "date": "Sorry, I didn't catch that — you can say it like 26 March 2026.",
            "month_year": "Sorry, I didn't catch that — just the month and year, like January 2010.",
            "currency": "Sorry, I didn't catch that — a dollar amount works, like one sixty thousand.",
            "yes_no": "Sorry, I didn't catch that — just a yes or no is fine.",
        }

        if field_type == "number":
            min_val = field_def.get("min")
            max_val = field_def.get("max")
            if min_val is not None and max_val is not None:
                return f"Sorry, I didn't catch that — just say a number between {int(min_val)} and {int(max_val)}."
            return "Sorry, I didn't catch that — just say a number."

        if field_type in _HINTS:
            return _HINTS[field_type]

        if options:
            opts_voice = format_options_for_voice(options)
            return f"Sorry, I didn't catch that — you can say {opts_voice}."

        return "Sorry, I didn't catch that — could you repeat?"

    def _answer_key(self, step_id: str) -> str:
        """Unique key for an answer accounting for loop iteration."""
        if self.in_driver_loop:
            return f"driver_{self.current_driver_idx + 1}.{step_id}"
        if self.in_vehicle_loop:
            return f"vehicle_{self.current_vehicle_idx + 1}.{step_id}"
        return step_id

    @staticmethod
    def _is_yes(text: str) -> bool:
        return is_affirmative_intent(text)

    @staticmethod
    def _is_no(text: str) -> bool:
        return is_negative_intent(text)

    @staticmethod
    def _extract_number(text: str, default: int = 1) -> int:
        """Extract the first integer from a spoken answer."""
        # Handle words
        word_map = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
        }
        t = text.lower().strip()
        for word, val in word_map.items():
            if re.search(rf"\b{word}\b", t):
                return val
        m = re.search(r"\d+", t)
        return int(m.group()) if m else default
