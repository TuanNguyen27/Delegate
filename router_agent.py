# toy_router_agent.py
import os, re, time, asyncio, torch, json
from dotenv import load_dotenv

# OpenAI Agents SDK
from agents import Agent, function_tool, Runner

from transformers import AutoModelForCausalLM, AutoTokenizer

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
def _slm_help_impl(question: str, mode: str = "cot", max_new_tokens: int = 256) -> str:
    """
    Solve a high-school math problem with the local Small Language Model (SLM).

    Args:
      question (str): The problem text.
      mode (str): "cot" for step-by-step; "direct" for concise final answer.
      max_new_tokens (int): Maximum generated tokens.

    Returns:
      str: JSON string with {"answer": "<value>", "reasoning": "<model_output>"}
    """
    try:
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

        # Return properly formatted JSON string
        result_json = json.dumps({
            "answer": ans,
            "latency_sec": round(_lat, 3),
            "reasoning": gen
        })
        
        # Log to tracker if it exists (for evaluation)
        try:
            from math500_evaluation import tracker
            tracker.log_tool_call(question, result_json, _lat)
        except ImportError:
            pass  # Tracker not imported, skip logging
        
        return result_json
        
    except Exception as e:
        error_json = json.dumps({
            "error": str(e),
            "answer": "",
            "reasoning": ""
        })
        
        # Log error to tracker if it exists
        try:
            from math500_evaluation import tracker
            tracker.log_tool_call(question, error_json, 0.0)
        except ImportError:
            pass
        
        return error_json

# Wrap the implementation for the Agent SDK
@function_tool
def slm_help(question: str, mode: str = "cot", max_new_tokens: int = 256) -> str:
    """
    Solve a high-school math problem with the local Small Language Model (SLM).

    Args:
      question (str): The problem text.
      mode (str): "cot" for step-by-step; "direct" for concise final answer.
      max_new_tokens (int): Maximum generated tokens.

    Returns:
      str: JSON string with {"answer": "<value>", "reasoning": "<model_output>"}
    """
    print(f"[TOOL CALLED] LLM invoked slm_help with question: {question[:60]}...")
    return _slm_help_impl(question, mode, max_new_tokens)

# ---------------------------
# Agent (LLM controller)
# ---------------------------
INSTRUCTIONS = (
    "You are an expert at solving high school competition math problems (MATH dataset level). "
    "Break down complex problems into steps. When you encounter:\n"
    "- Arithmetic computations (e.g., 127 × 89, 7^2023 mod 1000)\n"
    "- Algebraic simplifications (e.g., expand (x+2)(x-3), factor x^2-5x+6)\n"
    "- Solving equations (e.g., 3x^2 + 5x - 2 = 0)\n"
    "- Number theory calculations (e.g., gcd, lcm, modular arithmetic)\n"
    "Use the `slm_help` tool to compute these efficiently and accurately.\n\n"
    "For conceptual reasoning, geometric proofs, counting arguments, or strategic problem-solving, "
    "work through them yourself step-by-step.\n\n"
    "Always show your reasoning and provide the final answer in \\boxed{} format."
)

agent = Agent(
    name="Math Expert Agent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    model_settings=ModelSettings(max_tokens=2048),
    tools=[slm_help],
)

# ---------------------------
# Run Agent
# ---------------------------
async def run_agent(question: str):
    """
    Run the LLM agent, which can delegate to SLM when needed.
    """
    print(f"[AGENT] Processing question with LLM...")
    result = await Runner.run(agent, question)
    return result.final_output

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
        print(f"\nQ: {q}")
        out = await run_agent(q)
        print(f"Ans: {out}")

if __name__ == "__main__":
    # Check if we're in a notebook environment
    try:
        # If in notebook, use await directly instead of asyncio.run()
        get_ipython()  # This will raise NameError if not in IPython/Jupyter
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.run(main())
    except NameError:
        # Not in notebook, use asyncio.run() normally
        asyncio.run(main())