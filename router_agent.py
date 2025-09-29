# router_agent.py
import os, re, time, asyncio, torch
from dotenv import load_dotenv

# Your Agents SDK (as in your snippet)
from agents import Agent, function_tool, Runner  # , ItemHelpers, Run, ContextWrapper  # if you use them

from transformers import AutoModelForCausalLM, AutoTokenizer

from router.py import should_delegate_to_slm

# ---------------------------
# Env
# ---------------------------
load_dotenv()  # expects OPENAI_API_KEY for the Agents SDK

# ---------------------------
# SLM (Qwen) lazy loader
# ---------------------------
_SLM, _TOK = None, None
_SLM_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"

def _device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    elif torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32

def _lazy_load_slm():
    global _SLM, _TOK
    if _SLM is None or _TOK is None:
        device, dtype = _device_dtype()
        _SLM = AutoModelForCausalLM.from_pretrained(
            _SLM_ID,
            device_map="auto" if device != "cpu" else None,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        _TOK = AutoTokenizer.from_pretrained(_SLM_ID)
        _TOK.padding_side = "left"  # safer for batched autoregressive decoding
    return _SLM, _TOK

# ---------------------------
# Utils
# ---------------------------
_BOXED = re.compile(r"\\boxed\{([^}]+)\}")
_LAST_NUM = re.compile(r"(?<!\d)(-?\d+(?:/\d+)?)(?!\d)")

def extract_boxed_or_lastnum(text: str) -> str:
    m = _BOXED.findall(text)
    if m:
        return m[-1].strip()
    m = _LAST_NUM.findall(text)
    return m[-1] if m else ""

# ---------------------------
# Tool: call SLM for help
# ---------------------------
@function_tool
def slm_help(question: str, mode: str = "cot", max_new_tokens: int = 256) -> str:
    """
    Solve a high-school math problem with the local Small Language Model (SLM).

    Args:
      question (str): The problem text.
      mode (str): "cot" for step-by-step; "direct" for concise final answer.
      max_new_tokens (int): Maximum generated tokens.

    Returns:
      str: JSON-like string with {"answer": "<value>", "reasoning": "<model_output>"}
    """
    model, tok = _lazy_load_slm()

    sys = (
        "Please reason step by step and put your final answer within \\boxed{}."
        if mode == "cot"
        else "Answer concisely; only provide the final answer within \\boxed{}."
    )
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": question},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([prompt], return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tok.eos_token_id,
        )
    gen = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    ans = extract_boxed_or_lastnum(gen)
    _lat = time.time() - t0

    # Keep it string-y for the Agents SDK tool schema
    return f'{{"answer":"{ans}","latency_sec":{_lat:.3f},"reasoning":{gen!r}}}'

# ---------------------------
# Agent (LLM controller)
# ---------------------------
INSTRUCTIONS = (
    "You are an expert at solving high-school competition math problems. "
    "Use the tool `slm_help` when the task is arithmetic/algebraic or short deterministic math; "
    "otherwise reason yourself. Always provide a final numeric answer."
)

agent = Agent(
    name="Math Expert Agent",
    model="gpt-4o-mini",        # <- ensure your Agents SDK accepts this param; otherwise set globally
    instructions=INSTRUCTIONS,
    tools=[slm_help],
)

# ---------------------------
# Optional: small heuristic router (pre-filter)
# ---------------------------
_SIMPLE_EQ = re.compile(r"^\s*[-+*/\d\s().=x]+$")

def should_delegate_to_slm(question: str) -> bool:
    """
    Toy router: delegate if (a) short prompt, or (b) looks like a simple algebra/arithmetic form,
    or (c) has high digit density.
    """
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

async def route_and_run(question: str, force: str | None = None):
    """
    Route to SLM (tool) or LLM agent. `force` can be "slm" or "llm" for debugging.
    """
    if force == "slm" or (force is None and should_delegate_to_slm(question)):
        # Call the tool directly (fast path)
        return slm_help(question=question, mode="cot", max_new_tokens=256)
    else:
        # Let the LLM decide (it can still call slm_help if it wants)
        runner = Runner(agent)
        # Your SDK might support streaming; here we ask for a simple final text
        result = await runner.run(question)
        # Normalize to string
        return getattr(result, "output_text", str(result))

# ---------------------------
# Test
# ---------------------------
async def main():
    qs = [
        "If 3x + 5 = 20, what is x?",
        "A triangle has sides 5, 12, and 13. What is its area?",
        "The sum of the first 50 positive integers divisible by 3 equals what?"
    ]

    for q in qs:
        out = await route_and_run(q)
        print("\nQ:", q)
        print("Ans:", out)

if __name__ == "__main__":
    asyncio.run(main())
