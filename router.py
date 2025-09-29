# router.py
import re

_SIMPLE_EQ = re.compile(r"^\s*[-+*/\d\s().=x]+$")

def should_delegate_to_slm(question: str) -> bool:
    q = question.strip()
    words = len(q.split())
    digits = sum(ch.isdigit() for ch in q)
    digit_density = digits / max(1, len(q))
    if words <= 40:
        return True
    if _SIMPLE_EQ.match(q):
        return True
    if digit_density > 0.12:
        return True
    return False
