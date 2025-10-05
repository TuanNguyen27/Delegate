# router_agent_demo.py
import os, re, time, asyncio, torch
from dotenv import load_dotenv

from agents import Agent, function_tool, Runner, ModelSettings
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

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
        _TOK.padding_side = "left"
    return _SLM, _TOK

# ---------------------------
# Tool: call SLM for help
# ---------------------------
@function_tool
def slm_help(question: str) -> str:
    """
    Solve a mathematical calculation using specialized math model.
    Returns definitive answer that should be trusted immediately.
    
    Args:
        question: The calculation to perform
    
    Returns:
        Definitive answer in format "CALCULATION COMPLETE: answer"
    """
    print(f"[TOOL] slm_help: {question[:60]}...")
    
    try:
        model, tok = _lazy_load_slm()

        sys = "You are a math calculator. Put the final answer in \\boxed{} at the end."
        
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
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        
        gen = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        latency = time.time() - t0

        print(f"[SLM OUTPUT] Time: {latency:.2f}s")
        print(f"[SLM OUTPUT] Full response:\n{gen}\n")

        # Extract the boxed answer
        match = re.search(r'\\boxed\{([^}]+)\}', gen)

        if match:
            answer = match.group(1)
            result = f"CALCULATION COMPLETE: The answer to '{question}' is {answer}."
            print(f"[SLM RESULT] Extracted answer: {answer}")
            return result
        else:
            result = f"CALCULATION COMPLETE: {gen}"
            print(f"[SLM RESULT] No boxed answer found, returning full output")
            return result
        
    except Exception as e:
        return f"CALCULATION ERROR: {str(e)}. Please solve this calculation yourself."

# ---------------------------
# Agent
# ---------------------------
INSTRUCTIONS = (
    "You are a math expert designed to solve problems step-by-step.\n\n"

    "## CORE DIRECTIVE\n"
    "You must delegate ALL arithmetic, algebraic, or symbolic computations "
    "to the provided tool `slm_help(question)` — every time numbers or equations appear.\n\n"

    "## WORKFLOW\n"
    "1️. Read and understand the user's math problem.\n"
    "2️. For **any** calculation (addition, subtraction, multiplication, division, simplification, equation solving, etc.), "
    "immediately call `slm_help()` with the expression or subproblem.\n"
    "3️. When `slm_help` returns a message starting with 'CALCULATION COMPLETE:', "
    "trust that answer immediately and use it in your reasoning.\n"
    "4️. Never verify, repeat, or re-check the calculation yourself.\n"
    "5️. Continue reasoning using only those verified results until the final solution.\n"
    "6️. Present your final answer clearly in the form: \\boxed{final_answer}\n\n"

    "## STRICT RULES\n"
    "• ALWAYS use `slm_help()` for *every* numeric or algebraic computation.\n"
    "• NEVER compute or approximate results manually.\n"
    "• NEVER call `slm_help` twice for the same subproblem.\n"
    "• NEVER rewrite or simplify numeric results yourself.\n"
    "• If the result from `slm_help` seems odd, still trust it — do not re-check.\n\n"

    "## EXAMPLES\n"
    "- Example 1:\n"
    "  User: What is (23 + 19) × 4?\n"
    "  → Call: slm_help('(23 + 19) × 4')\n\n"
    "- Example 2:\n"
    "  User: Solve 3x + 5 = 14\n"
    "  → Call: slm_help('Solve 3x + 5 = 14 for x')\n\n"

    "Final note: Your role is *reasoning and delegation*. "
    "All computation belongs to the SLM."
)


agent = Agent(
    name="Math Expert Agent",
    instructions=INSTRUCTIONS,
    model="gpt-4o",
    model_settings=ModelSettings(
        max_tokens=512,
        parallel_tool_calls=False
    ),
    tools=[slm_help],
)

async def run_agent(question: str):
    result = await Runner.run(agent, question, max_turns=15)
    return result.final_output