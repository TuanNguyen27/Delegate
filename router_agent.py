# router_agent.py
import os, re, time, asyncio, torch, json
from dotenv import load_dotenv

# OpenAI Agents SDK
from agents import Agent, function_tool, Runner, ModelSettings

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
            dtype=dtype,
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

def extract_answer_from_slm(text: str) -> str:
    """Extract answer from SLM output, prioritizing \boxed{} format"""
    # First try \boxed{} format
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        answer = boxed[-1].strip()
        # Extract numbers from the boxed content
        nums = re.findall(r"-?\d+(?:\.\d+)?", answer)
        return nums[0] if nums else answer
    
    # Then try "The answer is: X" pattern
    m = re.search(r"[Tt]he answer is:?\s*([^\n.]+)", text)
    if m:
        answer = m.group(1).strip()
        nums = re.findall(r"-?\d+(?:\.\d+)?", answer)
        return nums[-1] if nums else answer
    
    # Fallback: last number in entire text
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""

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
            "You are a math calculation assistant. "
            "Solve the problem step by step. "
            "At the very end, you MUST write 'The answer is: X' where X is the final numerical answer. "
            "Do not write anything after 'The answer is: X'."
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
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        ans = extract_answer_from_slm(gen)
        _lat = time.time() - t0

        # Debug logging - BEFORE the return
        print(f"[SLM DEBUG] Output: {gen[:100]}...")
        print(f"[SLM DEBUG] Extracted answer: {ans}")

        # Return properly formatted JSON string
        result_json = json.dumps({
            "answer": ans,
            "latency_sec": round(_lat, 3),
            "reasoning": gen
        })
        
        # Log to tracker if it exists (for evaluation)
        try:
            from router_experiment import tracker
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
            from router_experiment import tracker
            tracker.log_tool_call(question, error_json, 0.0)
        except ImportError:
            pass
        
        return error_json

# Wrap the implementation for the Agent SDK
@function_tool
def slm_help(question: str, mode: str = "cot", max_new_tokens: int = 512) -> str:
    """
    Solve a high-school math problem with the local Small Language Model (SLM).
    Use this tool for ANY computational or numerical calculation including:
    - Arithmetic operations (addition, subtraction, multiplication, division)
    - Solving equations (linear, quadratic, polynomial)
    - Algebraic simplifications and expansions
    - Modular arithmetic and number theory calculations
    - Any calculation involving numbers
    Returns the complete step-by-step solution.

    Args:
      question (str): The math problem or calculation to solve.
      mode (str): "cot" for step-by-step reasoning; "direct" for concise answer.
      max_new_tokens (int): Maximum tokens to generate.

    Returns:
      str: Complete solution with reasoning and final answer in \boxed{} format
    """
    print(f"[TOOL CALLED] LLM invoked slm_help with question: {question[:60]}...")
    
    try:
        model, tok = _lazy_load_slm()

        sys = (
            "You are a math calculation assistant. "
            "Solve the problem step by step. "
            "Put your final answer in \\boxed{} format at the end."
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
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        _lat = time.time() - t0

        # Log to tracker if it exists
        try:
            from router_experiment import tracker
            tracker.log_tool_call(question, gen, _lat)
        except ImportError:
            pass
        
        # Return the complete SLM output directly
        return gen
        
    except Exception as e:
        return f"Error: {str(e)}"

# ---------------------------
# Agent (LLM controller)
# ---------------------------
INSTRUCTIONS = (
    "You are an expert at solving high school competition math problems (MATH dataset level). "
    "IMPORTANT: You MUST use the `slm_help` tool for ANY computational step including:\n"
    "- ALL arithmetic (addition, subtraction, multiplication, division)\n"
    "- Solving equations (linear, quadratic, polynomial)\n"
    "- Algebraic simplifications and expansions\n"
    "- Modular arithmetic and number theory calculations\n"
    "- ANY calculation with numbers\n\n"
    "Do NOT try to calculate these yourself. Always call slm_help for computational steps.\n\n"
    "Your workflow:\n"
    "1. Read and understand the problem\n"
    "2. Plan your solution approach\n"
    "3. For ANY calculation, call slm_help with the specific computation\n"
    "4. Interpret the tool's result and continue reasoning\n"
    "5. Provide your final answer in \\boxed{} format\n\n"
    "Example: If you need to compute 15 × 7, call slm_help('What is 15 times 7?')\n"
    "Example: If you need to solve x^2 + 3x - 10 = 0, call slm_help('Solve x^2 + 3x - 10 = 0')\n\n"
    "Always show your reasoning and provide the final answer in \\boxed{} format."
    "\n\nIMPORTANT: Trust the slm_help tool's calculations. "
    "Do NOT call the same calculation multiple times. "
    "If you get an answer from slm_help, use it and move forward with your solution.\n"
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