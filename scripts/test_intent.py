"""Quick smoke-test for intent helpers."""
import sys
sys.path.insert(0, ".")

from insurance_prompt import (
    is_negative_intent, is_affirmative_intent,
    is_ambiguous_intent, should_exit_flow,
)

tests = [
    # (text,                            exp_neg, exp_aff, exp_amb, exp_stop)
    ("No",                              True,  False, False, False),
    ("Nope",                            True,  False, False, False),
    ("It's not okay",                   True,  False, False, False),
    ("Not okay",                        True,  False, False, False),
    ("I'm not comfortable with that",   True,  False, False, False),
    ("I'm not interested",              True,  False, False, False),
    ("I don't want to continue",        True,  False, False, True),   # also exit
    ("No thanks",                       True,  False, False, False),
    ("Stop",                            True,  False, False, True),   # also exit
    ("Cancel",                          True,  False, False, True),   # also exit
    ("Yes",                             False, True,  False, False),
    ("Yeah sure",                       False, True,  False, False),
    ("Sure, go ahead",                  False, True,  False, False),
    ("That's fine",                     False, True,  False, False),
    ("I don't think so",                False, False, True,  False),
    ("Maybe not",                       False, False, True,  False),
    ("I'm not sure",                    False, False, True,  False),
]

all_pass = True
for text, exp_neg, exp_aff, exp_amb, exp_stop in tests:
    neg  = is_negative_intent(text)
    aff  = is_affirmative_intent(text)
    amb  = is_ambiguous_intent(text)
    stop = should_exit_flow(text)
    ok = (neg == exp_neg) and (aff == exp_aff) and (amb == exp_amb) and (stop == exp_stop)
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False
    print(f"{status}  neg={neg} aff={aff} amb={amb} stop={stop}  |  {text!r}")

print()
print("All passed!" if all_pass else "SOME TESTS FAILED")
sys.exit(0 if all_pass else 1)
