"""
prompt.py
=========
Production-ready prompt module for a voice AI insurance agent.

Architecture: Hybrid flow + LLM
  - The Excel-parsed flow engine decides WHAT question/step comes next.
  - This module decides HOW the LLM says it — naturally, warmly, voice-friendly.

Usage:
    from prompt import build_system_prompt, build_agent_prompt, PromptConfig

The module is designed to be dropped into any OpenAI-compatible backend.
All functions return plain strings suitable for the `messages` array.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import json


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FlowStep:
    """
    Represents a single parsed step from the Excel flow sheet.

    Attributes
    ----------
    step_id         : Unique step ID from the Excel (e.g. "2.3", "3.4")
    block           : Section label (START | DRIVER | VEHICLE | USAGE | RISK + HISTORY)
    question_text   : The raw question as authored in the Excel
    options         : Allowed spoken options, if any (e.g. ["Yes", "No"])
    conditions      : Any branching/conditional notes from the Excel
    expected_answer : Hint about the expected answer type/format
    voice_eligible  : Whether this step can be handled entirely by voice
    loop_context    : If inside a driver/vehicle loop, which iteration (e.g. "Driver 2 of 3")
    """

    step_id: str
    block: str
    question_text: str
    options: list[str] = field(default_factory=list)
    conditions: str = ""
    expected_answer: str = ""
    voice_eligible: bool = True
    loop_context: str = ""  # e.g. "Driver 2 of 3" or "Vehicle 1 of 2"


@dataclass
class SessionState:
    """
    Live state of the conversation for the current call.

    Attributes
    ----------
    call_sid            : Unique call identifier
    agent_name          : Display name for the voice agent
    collected_answers   : Dict of step_id -> raw user answer
    current_step_id     : The step currently being asked
    previous_step_id    : The step just completed
    last_agent_reply    : Text of the agent's last spoken turn
    last_user_reply     : Transcribed text of the user's last turn
    objection_flag      : True if user expressed hesitation/objection
    clarification_count : How many times we've re-asked the current step
    lead_qualified      : Overall qualification state (None | "qualified" | "disqualified" | "broker_referral")
    modification_flag   : True if vehicle modification was declared (triggers broker referral)
    driver_count        : Number of drivers collected from Q2.1
    vehicle_count       : Number of vehicles collected from Q3.1
    current_driver_idx  : Which driver loop iteration we're in (1-based)
    current_vehicle_idx : Which vehicle loop iteration we're in (1-based)
    whatsapp_number     : Phone number collected for WhatsApp document upload
    extra               : Catch-all dict for any ad-hoc metadata
    """

    call_sid: str = ""
    agent_name: str = "Sarah"
    collected_answers: dict[str, Any] = field(default_factory=dict)
    current_step_id: str = "1.1"
    previous_step_id: str = ""
    last_agent_reply: str = ""
    last_user_reply: str = ""
    objection_flag: bool = False
    clarification_count: int = 0
    lead_qualified: Optional[str] = None
    modification_flag: bool = False
    driver_count: int = 1
    vehicle_count: int = 1
    current_driver_idx: int = 1
    current_vehicle_idx: int = 1
    whatsapp_number: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptConfig:
    """
    Top-level configuration that controls tone and behaviour knobs.

    Attributes
    ----------
    max_response_sentences  : Hard ceiling on how many sentences the LLM may produce
    allow_clarification     : Whether the agent may ask for clarification
    max_clarification_tries : Before escalating to broker or ending
    use_few_shot            : Include few-shot examples in every call (set False to save tokens)
    temperature_hint        : Informational only — set on the API call, not in the prompt
    province                : Canadian province for any region-specific phrasing
    """

    max_response_sentences: int = 3
    allow_clarification: bool = True
    max_clarification_tries: int = 2
    use_few_shot: bool = True
    temperature_hint: float = 0.4
    province: str = "Ontario"


# ---------------------------------------------------------------------------
# Core system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are {agent_name}, calling on behalf of an insurance brokerage to collect details for an auto insurance quote.

## What you're doing
You're going through a short set of questions with the caller — one at a time — so a broker can put together their quote.
You don't give prices, coverage advice, or recommendations. That's the broker's job. Yours is just to get the details down accurately.

## How you talk
You sound like a real person — not a script, not a robot, not a corporate hotline.
Think of yourself as someone who does this every day: calm, friendly, slightly casual, efficient without being rushed.

Speak in short natural sentences. One thought at a time.
Never string three sentences together when one will do.
Never ask more than one question per turn — ever.

Keep acknowledgments short. "Got it." "Yep." "Okay." "Right." — then move straight to the next question.
Don't over-acknowledge. Don't thank the caller for every answer.
Don't narrate what you're doing: never say "I'm now going to ask you about..." — just ask.

When options exist, fold them into the question naturally.
Say: "Is it owned, financed, or leased?" — not "Please select from the following: Owned, Financed, or Leased."

Never sound like an IVR. Never sound like a chatbot. Never sound like a form being read aloud.

## Phrases to avoid entirely
"Certainly!" / "Absolutely!" / "Of course!" / "Great question!" / "How may I assist you?"
"Thank you for your response." / "Please provide the requested information." / "Kindly confirm."
"As an AI" / "I'm an AI assistant" / "I've noted that down for you."
Starting every reply with "Great!" or "Perfect!" — vary it or drop it.

## Natural phrases that work
"Got it." / "Okay." / "Yep." / "Right." / "Sure."
"Just to double-check..." / "Quick one —" / "And one more —"
"That's fine." / "No worries." / "We can work with that."
"Your broker'll sort that out." / "They'll follow up on that bit."

## When you don't know or can't say
Don't guess. Don't improvise policy details. Don't give pricing or coverage opinions.
If the caller asks something outside your scope, just say: "That's a question for your broker — they'll go over all that with you."

## Voice and TTS formatting
Short sentences. Natural contractions — "you're", "we'll", "it's", "they'll".
Spell out numbers: "two drivers" not "2 drivers".
No symbols that read oddly aloud: say "month and year" not "MM/YYYY", "the dollar amount" not "$", "percent" not "%".
Use a comma or dash where you'd naturally pause — not "..." which sounds like a glitch in TTS.

## If the caller asks why you need something
Give one honest sentence — no legal disclaimers, no over-explaining.
Then gently carry on with the question. See examples below.

## If the caller pushes back or seems hesitant
Acknowledge it in one breath. Don't argue, don't pile on reassurances.
Offer a simple out if needed: let them know the broker can handle it.
Then move on — or offer to end the call gracefully if they don't want to continue.

## If the caller goes off-topic
One short redirect: "Your broker can cover that — let me just grab a couple more details first."
Don't answer the off-topic question. Don't explain why you can't.

## Hard rules
One question per turn. Always.
Do not skip steps in the flow.
Do not invent questions outside the flow.
Do not reveal internal step IDs or flow logic to the caller.
Province context: {province}.
"""


def build_system_prompt(
    session: SessionState,
    config: PromptConfig,
) -> str:
    """
    Return the system prompt string for the current session.

    Parameters
    ----------
    session : SessionState
        Live session object containing agent name etc.
    config  : PromptConfig
        Behaviour configuration.

    Returns
    -------
    str
        Fully rendered system prompt ready to pass as the `system` message.
    """
    return _SYSTEM_PROMPT_TEMPLATE.format(
        agent_name=session.agent_name,
        province=config.province,
    ).strip()


# ---------------------------------------------------------------------------
# Flow context builder
# ---------------------------------------------------------------------------


def build_flow_context(
    current_step: FlowStep,
    allowed_next_steps: list[str],
    session: SessionState,
) -> str:
    """
    Build the structured flow context block that tells the LLM exactly where
    it is in the conversation and what it must ask next.

    This is injected as a system-level or assistant-priming message — NOT
    shown to the user.

    Parameters
    ----------
    current_step       : FlowStep
        The step the LLM must handle right now.
    allowed_next_steps : list[str]
        Step IDs the flow engine will accept as next destinations.
    session            : SessionState
        Current call state.

    Returns
    -------
    str
        A compact structured context block.
    """
    lines = [
        "## CURRENT FLOW STEP",
        f"Step ID      : {current_step.step_id}",
        f"Section      : {current_step.block}",
        f"Question     : {current_step.question_text}",
    ]

    if current_step.options:
        lines.append(f"Options      : {' / '.join(current_step.options)}")

    if current_step.loop_context:
        lines.append(f"Loop context : {current_step.loop_context}")

    if current_step.conditions:
        lines.append(f"Conditions   : {current_step.conditions}")

    if current_step.expected_answer:
        lines.append(f"Expected ans : {current_step.expected_answer}")

    lines.append(
        f"Voice OK     : {'Yes' if current_step.voice_eligible else 'No — prompt user to text/WhatsApp'}"
    )

    if allowed_next_steps:
        lines.append(f"Next steps   : {', '.join(allowed_next_steps)}")

    lines += [
        "",
        "## YOUR TASK FOR THIS TURN",
        "Say the question above as you would on a real phone call — short, natural, relaxed.",
        "If the caller just answered something, acknowledge it in one short word or phrase, then ask.",
        "Do NOT ask anything else. Do NOT explain to the caller what you're doing or why.",
    ]

    if session.clarification_count > 0:
        lines += [
            "",
            f"NOTE: You've already asked this {session.clarification_count} time(s).",
            "Change the wording. Don't repeat the same line again — it sounds like a broken recording.",
        ]

    if session.objection_flag:
        lines += [
            "",
            "NOTE: The caller pushed back or seemed reluctant.",
            "One brief acknowledgment, then either re-ask lightly or offer to have the broker handle it.",
            "Don't over-reassure. Don't lecture. Keep it short.",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session context builder
# ---------------------------------------------------------------------------


def build_session_context(session: SessionState) -> str:
    """
    Serialize the current session state as a readable context block.
    Gives the LLM memory of what has already been collected.

    Parameters
    ----------
    session : SessionState

    Returns
    -------
    str
        Human-readable session summary injected into the prompt.
    """
    lines = ["## SESSION STATE"]

    if session.agent_name:
        lines.append(f"Agent name          : {session.agent_name}")

    if session.driver_count:
        lines.append(f"Drivers in household: {session.driver_count}")
    if session.current_driver_idx > 1:
        lines.append(
            f"Currently on driver : {session.current_driver_idx} of {session.driver_count}"
        )

    if session.vehicle_count:
        lines.append(f"Vehicles            : {session.vehicle_count}")
    if session.current_vehicle_idx > 1:
        lines.append(
            f"Currently on vehicle: {session.current_vehicle_idx} of {session.vehicle_count}"
        )

    if session.modification_flag:
        lines.append("Vehicle modification: YES — broker referral already noted")

    if session.lead_qualified:
        lines.append(f"Lead status         : {session.lead_qualified}")

    if session.whatsapp_number:
        lines.append(f"WhatsApp number     : {session.whatsapp_number}")

    if session.collected_answers:
        lines.append("\n## COLLECTED ANSWERS (step_id → answer)")
        for step_id, answer in session.collected_answers.items():
            lines.append(f"  {step_id}: {answer}")

    if session.last_agent_reply:
        lines.append(f"\nLast agent said     : {session.last_agent_reply}")

    if session.last_user_reply:
        lines.append(f"Caller just said    : {session.last_user_reply}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES = """
## CALL EXAMPLES — TONE AND RHYTHM REFERENCE

Read these to calibrate your style. This is how you sound on a real call.

---

### Opening / consent (Step 1.1)
Agent: "Hey, this is Sarah calling from the brokerage — I just need to grab a few details to get your auto quote started. Should only take a few minutes. That okay with you?"

---

### Consent given — collecting WhatsApp number (Step 1.2)
Caller: "Yeah sure, go ahead."
Agent: "Great. What's a good number to reach you on WhatsApp? I'll send a link over so you can upload your licence after the call."

---

### Effective date (Step 1.3)
Agent: "And when would you want the policy to start — what date are you thinking?"

---

### Vehicle modifications (Step 1.4)
Agent: "Does the vehicle have any modifications? Anything non-standard from the factory?"

Caller: "I put a lift kit on it last year."
Agent: "Okay, I'll flag that for your broker — they'll follow up with you on that. Let me keep going with the rest."

---

### Number of drivers (Step 2.1)
Agent: "How many licensed drivers are in the household?"

---

### Caller answers indirectly
Caller: "It's just me and my daughter, she just got her licence."
Agent: "Got it — two drivers. And are you the registered owner and main driver of the vehicle?"

---

### Marital status (Step 2.3)
Agent: "What's your marital status — single, married, common law, divorced, or widowed?"

---

### Registered owner (Step 2.5)
Agent: "Are you the registered owner and primary driver on this one?"

---

### Licence issue date (Step 2.6)
Agent: "When was your licence first issued in Canada — roughly month and year is fine."

---

### Caller asks "why do you need that?"
Caller: "Why do you need to know when my licence was issued?"
Agent: "It's just to see how long you've been licensed — longer history can actually help with the rate. So roughly when was it first issued?"

---

### Caller asks "do I have to answer that?"
Caller: "Do I have to give you that?"
Agent: "No, you don't have to — but if we skip it your broker may need to follow up before the quote's ready. Up to you."

---

### Caller asks "what is this for?"
Caller: "What's this information even for?"
Agent: "It all goes to your broker so they can put the quote together — they won't do anything with it beyond that. So, how many vehicles are we looking at?"

---

### Retiree discount (Step 2.8 / 2.9)
Agent: "One quick thing — do you qualify for a retiree discount? Yes, no, or not sure is fine."

---

### Number of vehicles (Step 3.1)
Agent: "And how many vehicles are we insuring?"

---

### VIN — voice-ineligible step
Agent: "For the VIN — that's the seventeen-character ID — it's easier to send that over in the chat rather than spell it out. Can you pop it in there when you get a chance?"

---

### Vehicle ownership type (Step 3.4)
Agent: "Is the vehicle owned outright, financed, or leased?"

Caller: "I'm still paying it off."
Agent: "So financed — got it. And what's the name of the finance company?"

---

### Vehicle condition (Step 3.7)
Agent: "What's the condition of the vehicle — new, used, or demo?"

---

### Winter tires (Step 3.9)
Agent: "Do you run winter tires from November through to April?"

---

### Vehicle usage (Step 4.1)
Agent: "What's the vehicle mainly used for — pleasure, commuting, business, farm, or commercial?"

---

### Annual mileage (Step 4.4)
Agent: "Roughly how many kilometres do you put on it in a year?"

---

### Incomplete / vague answer — re-ask
Caller: "I don't drive that much."
Agent: "Fair enough — do you have a rough number? Even a ballpark is fine."

---

### Overnight parking (Step 4.5)
Agent: "Where does the vehicle usually get parked overnight?"

---

### Policy cancellation history (Step 5.1)
Agent: "Has any auto policy been cancelled for non-payment in the last three years?"

---

### Accidents or claims (Step 5.2)
Agent: "Any accidents or at-fault claims in the past ten years?"

---

### Licence suspension (Step 5.3)
Agent: "Has your licence ever been suspended?"

---

### Caller hesitates before a risk question
Caller: "Um... why are you asking about that?"
Agent: "It's standard on all auto applications — the broker just needs it on file. Has your licence ever been suspended?"

---

### Transition between sections (DRIVER → VEHICLE)
Agent: "Okay, that's the driver info done. Now just a few questions about the vehicle itself — how many are we insuring?"

---

### Caller goes off-topic
Caller: "Can you tell me what kind of coverage I should get?"
Agent: "Your broker'll go over all of that with you — they're better placed to advise on coverage. For now, let me just grab the last few details."

---

### Caller declines to continue
Caller: "Actually I don't want to do this right now."
Agent: "No problem at all. If you want to pick it up later, just give us a call. Take care."

---

### Wrap-up
Agent: "That's everything I need. I'll send a WhatsApp message to the number you gave me — there'll be a link to upload your licence securely. Your broker will be in touch from there. Thanks for your time."
"""


def get_few_shot_examples() -> str:
    """Return the few-shot examples block as a string."""
    return _FEW_SHOT_EXAMPLES.strip()


# ---------------------------------------------------------------------------
# Master agent prompt builder
# ---------------------------------------------------------------------------


def build_agent_prompt(
    current_step: FlowStep,
    allowed_next_steps: list[str],
    session: SessionState,
    config: PromptConfig,
) -> list[dict[str, str]]:
    """
    Build the complete messages list for an OpenAI-compatible API call.

    This is the main entry point. It assembles:
      1. System prompt   — who the agent is and tone rules
      2. Few-shot block  — example exchanges (optional)
      3. Session context — what has been collected so far
      4. Flow context    — what to ask right now
      5. Trigger         — a short instruction to generate the next spoken turn

    Parameters
    ----------
    current_step       : FlowStep
        The step the LLM must handle.
    allowed_next_steps : list[str]
        Valid next step IDs from the flow engine.
    session            : SessionState
        Live call state.
    config             : PromptConfig
        Tone / behaviour config.

    Returns
    -------
    list[dict]
        Ready-to-pass `messages` list for the completions API.

    Example
    -------
    ::

        messages = build_agent_prompt(step, next_steps, session, config)
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=config.temperature_hint,
            max_tokens=120,
        )
        agent_reply = response.choices[0].message.content
    """
    system_text = build_system_prompt(session, config)

    # Optionally append few-shot examples to the system prompt
    if config.use_few_shot:
        system_text += "\n\n" + get_few_shot_examples()

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_text},
    ]

    # Session context as a "hidden" user message (assistant won't see it as a turn)
    session_block = build_session_context(session)
    flow_block = build_flow_context(current_step, allowed_next_steps, session)

    context_payload = f"{session_block}\n\n{flow_block}"
    messages.append({"role": "user", "content": context_payload})

    # Trigger the agent's spoken response
    trigger = (
        "Say your next line — as you would on a real phone call. "
        "Short and natural. One to three sentences at most. "
        "No lists. No bullet points. No over-explaining. "
        "If acknowledging the last answer, one word or phrase is enough — then ask."
    )
    messages.append({"role": "user", "content": trigger})

    return messages


# ---------------------------------------------------------------------------
# Specialised prompt builders for edge-case turns
# ---------------------------------------------------------------------------


def build_clarification_prompt(
    current_step: FlowStep,
    session: SessionState,
    config: PromptConfig,
    reason: str = "unclear",
) -> list[dict[str, str]]:
    """
    Build a prompt for when the user's answer was unclear or invalid and we
    need to re-ask without sounding robotic.

    Parameters
    ----------
    current_step : FlowStep
    session      : SessionState
    config       : PromptConfig
    reason       : str
        Short tag for why clarification is needed:
        "unclear" | "invalid_format" | "out_of_range" | "no_answer"

    Returns
    -------
    list[dict]
        Messages list ready for the completions API.
    """
    reason_hints = {
        "unclear": "The caller's answer wasn't clear enough — couldn't parse an intent or value from it.",
        "invalid_format": "The caller gave something in the wrong format — for example, a date said incorrectly.",
        "out_of_range": "The answer doesn't fit what's expected — outside the valid range or options.",
        "no_answer": "The caller didn't really answer — went off on a tangent or stayed silent.",
    }
    hint = reason_hints.get(reason, reason_hints["unclear"])

    system_text = build_system_prompt(session, config)
    session_block = build_session_context(session)

    clarification_instruction = (
        f"## NEED TO RE-ASK\n"
        f"Why: {hint}\n"
        f"Step: {current_step.step_id} — {current_step.question_text}\n\n"
        "Re-ask naturally. Change the wording — don't repeat yourself. "
        "If the format was wrong, mention it casually in one breath. "
        "One or two sentences. Sound like you're on a call, not reading a form."
    )

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": f"{session_block}\n\n{clarification_instruction}"},
        {
            "role": "user",
            "content": "Re-ask now — natural, brief, different wording than before.",
        },
    ]
    return messages


def build_objection_prompt(
    current_step: FlowStep,
    session: SessionState,
    config: PromptConfig,
    objection_text: str = "",
) -> list[dict[str, str]]:
    """
    Build a prompt to handle a caller objection or hesitation mid-flow.

    Parameters
    ----------
    current_step   : FlowStep
    session        : SessionState
    config         : PromptConfig
    objection_text : str
        The raw transcription of what the caller said.

    Returns
    -------
    list[dict]
    """
    system_text = build_system_prompt(session, config)
    session_block = build_session_context(session)

    objection_instruction = (
        f"## CALLER PUSHED BACK\n"
        f'Caller said: "{objection_text}"\n'
        f"Current step: {current_step.step_id} — {current_step.question_text}\n\n"
        "Acknowledge it briefly — one short sentence. Don't over-explain or over-reassure.\n"
        "Then either re-ask lightly, or offer to let the broker handle it if they're really not comfortable.\n"
        "Two sentences maximum. Keep it calm and easy."
    )

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": f"{session_block}\n\n{objection_instruction}"},
        {"role": "user", "content": "Say your response now — keep it short and calm."},
    ]
    return messages


def build_redirect_prompt(
    off_topic_text: str,
    current_step: FlowStep,
    session: SessionState,
    config: PromptConfig,
) -> list[dict[str, str]]:
    """
    Build a prompt to handle off-topic questions or tangents and redirect
    back to the current flow step.

    Parameters
    ----------
    off_topic_text : str
        What the caller said that was off-topic.
    current_step   : FlowStep
    session        : SessionState
    config         : PromptConfig

    Returns
    -------
    list[dict]
    """
    system_text = build_system_prompt(session, config)
    session_block = build_session_context(session)

    redirect_instruction = (
        f"## CALLER WENT OFF-TOPIC\n"
        f'Caller said: "{off_topic_text}"\n'
        f"Current step: {current_step.step_id} — {current_step.question_text}\n\n"
        "One short redirect — something like 'Your broker can cover that.' "
        "Do not answer the off-topic question. Do not explain why you can't. "
        "Then carry straight on with the current question. Two sentences maximum."
    )

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": f"{session_block}\n\n{redirect_instruction}"},
        {
            "role": "user",
            "content": "Redirect now, then ask the question — short and natural.",
        },
    ]
    return messages


def build_broker_referral_prompt(
    reason: str,
    session: SessionState,
    config: PromptConfig,
) -> list[dict[str, str]]:
    """
    Build a prompt for gracefully ending the call with a broker referral
    (e.g. vehicle modifications declared, complex scenario).

    Parameters
    ----------
    reason  : str  Short reason for referral (e.g. "vehicle modification", "complex risk").
    session : SessionState
    config  : PromptConfig

    Returns
    -------
    list[dict]
    """
    system_text = build_system_prompt(session, config)

    referral_instruction = (
        f"## BROKER REFERRAL — WRAPPING UP\n"
        f"Reason: {reason}\n\n"
        "Let the caller know a broker will be in touch to handle this. "
        "Don't make it sound like a rejection or a problem. "
        "Keep it light and easy — like it's just the normal next step. "
        "Two sentences maximum."
    )

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": referral_instruction},
        {"role": "user", "content": "Say your closing line — short, warm, natural."},
    ]
    return messages


def build_wrap_up_prompt(
    session: SessionState,
    config: PromptConfig,
) -> list[dict[str, str]]:
    """
    Build a prompt for the call wrap-up turn once all steps are complete.

    Parameters
    ----------
    session : SessionState
    config  : PromptConfig

    Returns
    -------
    list[dict]
    """
    system_text = build_system_prompt(session, config)
    session_block = build_session_context(session)

    wrap_instruction = (
        "## ALL DONE — WRAP UP THE CALL\n"
        "All questions are answered. Close out the call naturally — like a real person ending a phone conversation.\n"
        "  - A brief thanks (one short phrase, not effusive).\n"
        "  - Mention the WhatsApp message for document upload if a number was collected.\n"
        "  - Let them know the broker will be in touch.\n"
        "  - Sign off warmly but briefly.\n"
        "Two to three sentences. Sound like someone who does this every day — not a scripted goodbye."
    )

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": f"{session_block}\n\n{wrap_instruction}"},
        {"role": "user", "content": "Close the call now — natural, brief, human."},
    ]
    return messages


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def format_options_for_voice(options: list[str]) -> str:
    """
    Convert a list of options into a natural spoken string.

    Examples
    --------
    >>> format_options_for_voice(["Yes", "No"])
    'yes or no'
    >>> format_options_for_voice(["Owned", "Financed", "Leased"])
    'owned, financed, or leased'
    """
    if not options:
        return ""
    options = [o.lower() for o in options]
    if len(options) == 1:
        return options[0]
    if len(options) == 2:
        return f"{options[0]} or {options[1]}"
    return ", ".join(options[:-1]) + f", or {options[-1]}"


def session_to_json(session: SessionState) -> str:
    """Serialize a SessionState to a JSON string for logging / persistence."""
    import dataclasses

    return json.dumps(dataclasses.asdict(session), indent=2, default=str)


def session_from_dict(data: dict) -> SessionState:
    """Reconstruct a SessionState from a plain dict (e.g. loaded from JSON)."""
    return SessionState(
        **{k: v for k, v in data.items() if k in SessionState.__dataclass_fields__}
    )


def flow_step_from_dict(data: dict) -> FlowStep:
    """Construct a FlowStep from a plain dict (e.g. from the parsed Excel row)."""
    return FlowStep(
        **{k: v for k, v in data.items() if k in FlowStep.__dataclass_fields__}
    )


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

import re as _re
from datetime import date as _date, datetime as _datetime

_PHONE_STRIP_RE = _re.compile(r"\D")

_MONTH_NAMES: dict[str, int] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_MONTH_ABBR: dict[str, int] = {k[:3]: v for k, v in _MONTH_NAMES.items()}

_WORD_NUMS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

# Map step IDs to their validator type
_STEP_VALIDATORS: dict[str, str] = {
    "1.2": "phone",  # WhatsApp number
    "1.3": "date",  # Policy effective date
    "1.4": "yes_no",  # Vehicle modifications?
    "2.1": "drivers",  # Number of drivers
    "2.2": "yes_no",  # Registered owner / main driver?
    "2.4": "yes_no",  # Additional driver also listed?
    "2.6": "month_year",  # Licence issue date
    "2.7": "yes_no",  # Driver training certificate?
    "2.8": "yes_no",  # Retiree discount?
    "2.9": "yes_no",  # Retiree discount confirmation
    "3.1": "vehicles",  # Number of vehicles
    "3.9": "yes_no",  # Winter tires?
    "4.4": "mileage",  # Annual kilometres
    "5.1": "yes_no",  # Policy cancellation history?
    "5.2": "yes_no",  # Accidents / claims?
    "5.3": "yes_no",  # Licence suspension?
}


def validate_phone_number(text: str) -> tuple[bool, str, str]:
    """
    Validate a North American phone number from spoken/typed input.

    Accepts formats like: 4165550199, (416) 555-0199, 416-555-0199,
    +1 416 555 0199, 1-800-555-0199.

    Returns
    -------
    (is_valid, normalised_e164, error_reason)
    error_reason is one of: "" | "invalid_length" | "invalid_area_code"
    """
    digits = _PHONE_STRIP_RE.sub("", text.strip())
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return False, "", "invalid_length"
    if digits[0] in ("0", "1"):
        return False, "", "invalid_area_code"
    return True, f"+1{digits}", ""


def validate_date(text: str, allow_past: bool = False) -> tuple[bool, str, str]:
    """
    Validate a full date for policy effective date.

    Tries multiple spoken and written formats.

    Returns
    -------
    (is_valid, ISO-date "YYYY-MM-DD", error_reason)
    error_reason is one of: "" | "unrecognised_format" | "date_in_past"
    """
    text = text.strip()
    _formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%B %d %Y",
        "%b %d %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %d %y",
        "%b %d %y",
        "%d %B %y",
        "%d %b %y",  # 2-digit year forms
    ]
    parsed: Optional[_date] = None
    for fmt in _formats:
        try:
            parsed = _datetime.strptime(text, fmt).date()
            break
        except ValueError:
            continue
    if parsed is None:
        return False, "", "unrecognised_format"
    if not allow_past and parsed < _date.today():
        return False, "", "date_in_past"
    return True, parsed.isoformat(), ""


def validate_month_year(text: str) -> tuple[bool, str, str]:
    """
    Validate a month + year for licence issue date.

    Accepts: "January 2015", "Jan 2015", "01/2015", "2015-01", "2015".

    Returns
    -------
    (is_valid, "YYYY-MM", error_reason)
    error_reason is one of: "" | "year_out_of_range" |
    "month_or_year_out_of_range" | "unrecognised_format"
    """
    text = text.strip().lower()
    today_year = _date.today().year

    # Year only — e.g. "2015"
    m = _re.fullmatch(r"(\d{4})", text)
    if m:
        yr = int(m.group(1))
        if 1950 <= yr <= today_year:
            return True, f"{yr:04d}-01", ""
        return False, "", "year_out_of_range"

    # MM/YYYY or MM-YYYY
    m = _re.fullmatch(r"(\d{1,2})[/\-](\d{4})", text)
    if m:
        mo, yr = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12 and 1950 <= yr <= today_year:
            return True, f"{yr:04d}-{mo:02d}", ""
        return False, "", "month_or_year_out_of_range"

    # YYYY-MM or YYYY/MM
    m = _re.fullmatch(r"(\d{4})[/\-](\d{1,2})", text)
    if m:
        yr, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12 and 1950 <= yr <= today_year:
            return True, f"{yr:04d}-{mo:02d}", ""
        return False, "", "month_or_year_out_of_range"

    # "January 2015" or "Jan 2015"
    m = _re.fullmatch(r"([a-z]+)\s+(\d{4})", text)
    if m:
        mon_str, yr = m.group(1), int(m.group(2))
        mo = _MONTH_NAMES.get(mon_str) or _MONTH_ABBR.get(mon_str[:3])
        if mo and 1950 <= yr <= today_year:
            return True, f"{yr:04d}-{mo:02d}", ""

    # "2015 January" or "2015 Jan"
    m = _re.fullmatch(r"(\d{4})\s+([a-z]+)", text)
    if m:
        yr, mon_str = int(m.group(1)), m.group(2)
        mo = _MONTH_NAMES.get(mon_str) or _MONTH_ABBR.get(mon_str[:3])
        if mo and 1950 <= yr <= today_year:
            return True, f"{yr:04d}-{mo:02d}", ""

    # "Month DD YYYY" — day included in STT output; extract month + year, ignore day
    m = _re.fullmatch(r"([a-z]+)\s+\d{1,2}\s+(\d{4})", text)
    if m:
        mon_str, yr = m.group(1), int(m.group(2))
        mo = _MONTH_NAMES.get(mon_str) or _MONTH_ABBR.get(mon_str[:3])
        if mo and 1950 <= yr <= today_year:
            return True, f"{yr:04d}-{mo:02d}", ""
        if mo:
            return False, "", "year_out_of_range"

    # "Month [DD] YY" — 2-digit year shorthand (e.g. "April 26" or "April 25 26")
    m = _re.fullmatch(r"([a-z]+)\s+(?:\d{1,2}\s+)?(\d{2})", text)
    if m:
        mon_str, yr2 = m.group(1), int(m.group(2))
        mo = _MONTH_NAMES.get(mon_str) or _MONTH_ABBR.get(mon_str[:3])
        yr = 2000 + yr2 if yr2 < 69 else 1900 + yr2  # Python %y convention
        if mo and 1950 <= yr <= today_year:
            return True, f"{yr:04d}-{mo:02d}", ""
        if mo:
            return False, "", "year_out_of_range"

    return False, "", "unrecognised_format"


def validate_integer(
    text: str, min_val: int = 1, max_val: int = 100
) -> tuple[bool, int, str]:
    """
    Validate a spoken/typed integer (drivers, vehicles, mileage).

    Returns
    -------
    (is_valid, int_value, error_reason)
    error_reason is one of: "" | "not_a_number" | "below_minimum" | "above_maximum"
    """
    text = text.strip().lower()
    if text in _WORD_NUMS:
        val = _WORD_NUMS[text]
    else:
        cleaned = _re.sub(r"[\s,]", "", text)  # handle "20,000" or "20 000"
        if not cleaned.lstrip("-").isdigit():
            return False, 0, "not_a_number"
        val = int(cleaned)
    if val < min_val:
        return False, val, "below_minimum"
    if val > max_val:
        return False, val, "above_maximum"
    return True, val, ""


def validate_option(text: str, options: list[str]) -> tuple[bool, str, str]:
    """
    Validate that the answer matches one of the allowed spoken options.
    Case-insensitive, partial-match friendly.

    Returns
    -------
    (is_valid, matched_option, error_reason)
    error_reason is one of: "" | "no_option_match"
    """
    text_clean = _re.sub(r"[^\w\s]", "", text).strip().lower()
    for opt in options:
        if opt.lower() in text_clean or text_clean in opt.lower():
            return True, opt, ""
    return False, "", "no_option_match"


def validate_yes_no(text: str) -> tuple[bool, str, str]:
    """
    Validate a yes/no answer from speech.

    Returns
    -------
    (is_valid, "yes" | "no", error_reason)
    error_reason is one of: "" | "unclear_yes_no"
    """
    text_lower = text.strip().lower()
    _YES = {
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "correct",
        "right",
        "definitely",
        "absolutely",
        "of course",
        "that's right",
    }
    _NO = {"no", "nope", "nah", "not really", "never", "no i haven't", "no i don't"}
    for w in _YES:
        if w in text_lower:
            return True, "yes", ""
    for w in _NO:
        if w in text_lower:
            return True, "no", ""
    return False, "", "unclear_yes_no"


def validate_step_answer(
    step_id: str,
    answer: str,
    options: Optional[list[str]] = None,
) -> tuple[bool, Any, str]:
    """
    Route-based validator — the main entry point for the flow engine.

    Given a step ID and the raw transcribed answer, returns
    ``(is_valid, normalised_value, error_reason)``.

    ``error_reason`` is ``""`` on success, otherwise one of:
    ``"invalid_length"`` | ``"invalid_area_code"`` | ``"unrecognised_format"`` |
    ``"date_in_past"`` | ``"year_out_of_range"`` | ``"not_a_number"`` |
    ``"below_minimum"`` | ``"above_maximum"`` | ``"no_option_match"`` |
    ``"unclear_yes_no"`` | ``"empty_answer"``

    Parameters
    ----------
    step_id : str
        The flow step ID (e.g. ``"1.2"``).
    answer  : str
        Raw transcribed text from the caller.
    options : list[str] | None
        Allowed answer options for this step (from the flow engine).
    """
    vtype = _STEP_VALIDATORS.get(step_id)

    if vtype == "phone":
        return validate_phone_number(answer)
    if vtype == "date":
        return validate_date(answer, allow_past=False)
    if vtype == "month_year":
        return validate_month_year(answer)
    if vtype == "drivers":
        return validate_integer(answer, min_val=1, max_val=10)
    if vtype == "vehicles":
        return validate_integer(answer, min_val=1, max_val=10)
    if vtype == "mileage":
        return validate_integer(answer, min_val=0, max_val=500_000)
    if vtype == "yes_no":
        return validate_yes_no(answer)

    # Option-list steps
    if options:
        return validate_option(answer, options)

    # Fallback: accept any non-empty answer
    stripped = answer.strip()
    if stripped:
        return True, stripped, ""
    return False, "", "empty_answer"


# ---------------------------------------------------------------------------
# Quick self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Run: python prompt.py
    Prints a sample rendered prompt to stdout so you can eyeball it.
    """

    config = PromptConfig(province="Ontario", use_few_shot=True)

    session = SessionState(
        call_sid="DEMO-001",
        agent_name="Sarah",
        collected_answers={
            "1.1": "yes",
            "1.2": "4165550199",
            "1.3": "2025-06-01",
            "1.4": "no modifications",
            "2.1": "2",
        },
        current_step_id="2.3",
        previous_step_id="2.2",
        last_agent_reply="Are you the registered owner and primary operator of the vehicle?",
        last_user_reply="Yes, that's me.",
        driver_count=2,
        current_driver_idx=1,
    )

    step = FlowStep(
        step_id="2.3",
        block="DRIVER",
        question_text="What is your marital status?",
        options=["Single", "Married", "Common Law", "Divorced", "Widowed"],
        conditions="",
        expected_answer="One of the listed options",
        voice_eligible=True,
        loop_context="Driver 1 of 2",
    )

    messages = build_agent_prompt(
        current_step=step,
        allowed_next_steps=["2.4", "2.5"],
        session=session,
        config=config,
    )

    print("=" * 60)
    print("SAMPLE RENDERED PROMPT (build_agent_prompt)")
    print("=" * 60)
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        print(f"\n[{role}]\n{content}\n")
        print("-" * 60)

    # ---------------------------------------------------------------------------
# Backward compatibility exports
# ---------------------------------------------------------------------------

DEFAULT_PROMPT_CONFIG = PromptConfig()
DEFAULT_SESSION_STATE = SessionState(agent_name="Alex")
AGENT_NAME: str = DEFAULT_SESSION_STATE.agent_name

INSURANCE_SYSTEM_PROMPT = build_system_prompt(
    session=DEFAULT_SESSION_STATE,
    config=DEFAULT_PROMPT_CONFIG,
)

# ---------------------------------------------------------------------------
# Naturalisation helpers  (used by insurance_flow_engine)
# ---------------------------------------------------------------------------

NATURALIZE_PROMPT = """\
You are rephrasing a scripted insurance question into warm, natural spoken English.
Rules:
- Keep it concise — one or two short sentences max.
- Sound like a friendly human agent, not a form.
- Do NOT add new questions or information not in the original.
- Do NOT use filler words like "certainly" or "absolutely".
- Preserve any answer options exactly as given.
- If a "Format hint" is provided, weave it naturally into the question as a brief aside (e.g. "You can say it digit by digit.").
- Output ONLY the rephrased question, no commentary."""


def build_naturalize_prompt(
    raw_question: str,
    options: list[str] | None = None,
    block_transition: str | None = None,
    loop_context: str | None = None,
    format_hint: str | None = None,
) -> str:
    """Return the user-turn prompt for the naturalisation LLM call."""
    parts: list[str] = []
    if loop_context:
        parts.append(f"Context: {loop_context}")
    if block_transition:
        parts.append(f"Transition note: {block_transition}")
    parts.append(f"Original question: {raw_question}")
    if options:
        parts.append("Answer options: " + ", ".join(options))
    if format_hint:
        parts.append(f"Format hint: {format_hint}")
    parts.append("Rephrase this question naturally for a voice call.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Intent detection helpers  (consent, refusal, affirmation, ambiguity)
# ---------------------------------------------------------------------------

# ── Negative / refusal patterns ──────────────────────────────────────────────
_NEG_EXACT = frozenset(
    {
        "no",
        "nope",
        "nah",
        "nah thanks",
        "no thanks",
        "stop",
        "cancel",
        "quit",
        "abort",
        "decline",
        "refuse",
        "rejected",
        "negative",
        "never",
    }
)

_NEG_FRAGMENTS: tuple[str, ...] = (
    # Direct negatives
    "not okay",
    "not ok",
    "not alright",
    "not fine",
    "thats not okay",
    "that is not okay",
    "im not okay",
    "i am not okay",
    # Comfort / willingness
    "not comfortable",
    "not happy with",
    "not interested",
    "dont want",
    "do not want",
    "i dont want",
    "i do not want",
    "dont wish",
    "do not wish",
    "id rather not",
    "i would rather not",
    "id prefer not",
    "i would prefer not",
    # Consent-specific
    "not consent",
    "dont consent",
    "do not consent",
    "withhold consent",
    "not agree",
    "dont agree",
    "do not agree",
    "not accept",
    "dont accept",
    "do not accept",
    # Continuation
    "dont continue",
    "do not continue",
    "dont proceed",
    "do not proceed",
    # Disinterest
    "not at this time",
    "not right now",
    "not today",
    "not for me",
    "this isnt for me",
    "this is not for me",
    "not looking",
    "im good thanks",
)

# ── Affirmative patterns ──────────────────────────────────────────────────────
# _AFF_EXACT: single words / short complete answers that are ONLY affirmative
_AFF_EXACT = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "ok",
        "okay",
        "fine",
        "alright",
        "agreed",
    }
)

_AFF_FRAGMENTS: tuple[str, ...] = (
    "thats okay",
    "that is okay",
    "thats fine",
    "that is fine",
    "thats alright",
    "that is alright",
    "i agree",
    "i accept",
    "i consent",
    "im okay",
    "i am okay",
    "im fine",
    "i am fine",
    "im happy",
    "i am happy",
    "go right ahead",
    "please continue",
    "carry on",
    "go ahead",
    "absolutely",
    "definitely",
    "of course",
    "no problem",
    "sounds good",
    "works for me",
    "that works",
    "sure thing",
    "alright then",
)

# ── Ambiguous / soft-negative patterns ───────────────────────────────────────
_AMB_FRAGMENTS: tuple[str, ...] = (
    "i dont think so",
    "i do not think so",
    "maybe not",
    "probably not",
    "not sure",
    "im not sure",
    "i am not sure",
    "not certain",
    "i suppose not",
    "i guess not",
    "ill have to think",
    "i need to think",
    "let me think",
    "im unsure",
    "i am unsure",
    "not really sure",
    "kind of not",
)


def _normalise(text: str) -> str:
    """Lowercase, remove apostrophes/curly-quotes entirely (so "don't"→"dont"),
    then replace all remaining punctuation with spaces, collapse whitespace.
    """
    t = text.lower()
    t = _re.sub(r"['\u2018\u2019`]", "", t)  # remove apostrophes — "don't" → "dont"
    t = _re.sub(r"[^\w\s]", " ", t)  # replace other punctuation with space
    return _re.sub(r"\s+", " ", t).strip()


def is_negative_intent(text: str) -> bool:
    """
    Return True if the spoken text expresses refusal, rejection, or
    any form of 'no'.

    Covers:
    - Direct negatives ("no", "nope", "stop")
    - Comfort/willingness phrases ("not comfortable", "I don't want to")
    - Consent-specific declines ("I don't consent")
    - Disinterest phrases ("not for me", "not right now")
    """
    t = _normalise(text)
    # Exact full match
    if t in _NEG_EXACT:
        return True
    # Starts with a known negative word (catches "No, it's fine" etc.)
    first_word = t.split()[0] if t.split() else ""
    if first_word in _NEG_EXACT:
        return True
    # Substring fragment match (patterns are already normalised — no apostrophes)
    for frag in _NEG_FRAGMENTS:
        if frag in t:
            return True
    return False


def is_affirmative_intent(text: str) -> bool:
    """
    Return True if the spoken text expresses agreement, consent, or
    any form of 'yes'.
    """
    t = _normalise(text)
    # Exact full match
    if t in _AFF_EXACT:
        return True
    # Starts with a known affirmative word (catches "Yeah sure", "Sure, go ahead")
    first_word = t.split()[0] if t.split() else ""
    if first_word in _AFF_EXACT:
        return True
    # Substring fragment match
    for frag in _AFF_FRAGMENTS:
        if frag in t:
            return True
    return False


def is_ambiguous_intent(text: str) -> bool:
    """
    Return True if the spoken text is ambiguous — neither clearly yes nor no.

    Examples: "I don't think so", "maybe not", "not sure".
    Use this to trigger a soft clarification question rather than assuming
    consent or refusal.
    """
    t = _normalise(text)
    for frag in _AMB_FRAGMENTS:
        if frag in t:
            return True
    return False


def should_exit_flow(text: str) -> bool:
    """
    Return True when the caller wants to stop the entire call immediately.

    More decisive than is_negative_intent — reserved for hard stops.
    Examples: "stop", "cancel", "I want to quit", "end the call".
    """
    t = _normalise(text)
    hard_stops = frozenset(
        {
            "stop",
            "cancel",
            "quit",
            "abort",
            "end",
            "hang up",
            "bye",
            "goodbye",
            "end the call",
            "i want to stop",
            "i want to quit",
            "id like to stop",
        }
    )
    if t in hard_stops:
        return True
    for phrase in (
        "want to stop",
        "want to quit",
        "want to end",
        "end this call",
        "hang up",
        "dont want to continue",
        "do not want to continue",
    ):
        if phrase in t:
            return True
    return False


# ── Graceful exit messages ────────────────────────────────────────────────────
REFUSAL_FAREWELL = (
    "Got it — no worries at all. "
    "If you change your mind, just give us a call and we'll be happy to help. "
    "Have a great day!"
)

AMBIGUOUS_CONSENT_CLARIFICATION = (
    "Just to confirm — you'd prefer not to continue right now? Totally fine either way."
)
