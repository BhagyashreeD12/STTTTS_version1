from openai import OpenAI
from insurance_prompt import (
    FlowStep,
    SessionState,
    PromptConfig,
    build_agent_prompt,
)
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

config = PromptConfig(
    province="Ontario",
    use_few_shot=False,
)

session = SessionState(
    call_sid="TEST-001",
    agent_name="Sarah",
    collected_answers={
        "1.1": "yes",
        "1.2": "4165550199",
        "1.3": "2025-06-01",
        "2.1": "2",
    },
    current_step_id="2.3",
    previous_step_id="2.2",
    last_agent_reply="Are you the registered owner and primary driver of the vehicle?",
    last_user_reply="Yes, that's me.",
    driver_count=2,
    current_driver_idx=1,
)

step = FlowStep(
    step_id="2.3",
    block="DRIVER",
    question_text="What is your marital status?",
    options=["Single", "Married", "Common Law", "Divorced", "Widowed"],
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

response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=messages,
    temperature=0.4,
    max_tokens=80,
)

print("\n=== MODEL REPLY ===")
print(response.choices[0].message.content)
print("\n=== TOKEN USAGE ===")
print(response.usage)