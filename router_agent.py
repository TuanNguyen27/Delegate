# router_agent.py
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
    Solve a mathematical calculation using a specialized math model.
    Returns a definitive answer that should be trusted immediately.
    
    Args:
        question: The specific calculation to perform (e.g., "What is 520 + 650?")
    
    Returns:
        Definitive answer in format "CALCULATION COMPLETE: The answer is X"
    """
    print(f"[TOOL CALLED] slm_help: {question[:60]}...")
    
    try:
        model, tok = _lazy_load_slm()

        sys = "You are a math calculator. Solve step-by-step. Put the final answer in \\boxed{} at the end."
        
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

        # Print full SLM output for debugging
        print(f"[SLM OUTPUT] Time: {latency:.2f}s")
        print(f"[SLM OUTPUT] Full response:\n{gen}\n")

        # Extract the boxed answer
        match = re.search(r'\\boxed\{([^}]+)\}', gen)
        
        # Log to tracker - import at call time to avoid circular import
        try:
            import sys
            if 'router_experiment' in sys.modules:
                from router_experiment import tracker
                tracker.log_tool_call(question, gen, latency)
                print(f"[TRACKER] Logged tool call (latency: {latency:.2f}s)")
        except Exception as e:
            print(f"[TRACKER] Failed to log: {e}")

        if match:
            answer = match.group(1)
            result = f"CALCULATION COMPLETE: The answer to '{question}' is {answer}. Use this value directly in your solution."
            print(f"[SLM RESULT] Extracted answer: {answer}")
            return result
        else:
            result = f"CALCULATION COMPLETE: {gen}\n\nUse the final result from above in your solution."
            print(f"[SLM RESULT] No boxed answer found, returning full output")
            return result
        
    except Exception as e:
        return f"CALCULATION ERROR: {str(e)}. Please solve this calculation yourself."

# ---------------------------
# Agent
# ---------------------------
INSTRUCTIONS = (
    "You solve high school math competition problems.\n\n"
    
    "WORKFLOW:\n"
    "1. Understand the problem\n"
    "2. For ANY calculation, call slm_help ONCE with that exact calculation\n"
    "3. When you receive 'CALCULATION COMPLETE:', the tool has finished\n"
    "4. Extract the answer provided and USE IT immediately\n"
    "5. Continue with your solution using that result\n"
    "6. Provide your final answer in \\boxed{}\n\n"
    
    "CRITICAL RULES:\n"
    "• Call slm_help for ALL arithmetic, algebra, combinatorics\n"
    "• When you see 'CALCULATION COMPLETE:', the calculation is DONE\n"
    "• NEVER call slm_help again for the same calculation\n"
    "• Use the provided answer immediately - do NOT verify or retry\n"
    "• After you have your final answer, write \\boxed{answer} and STOP\n\n"
    
    "Example:\n"
    "You: slm_help('What is 26 times 10?')\n"
    "Tool: CALCULATION COMPLETE: The answer to 'What is 26 times 10?' is 260.\n"
    "You: Use 260 in the next step. Do NOT call slm_help again for 26*10."
)

agent = Agent(
    name="Math Expert Agent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    model_settings=ModelSettings(
        max_tokens=2048,
        parallel_tool_calls=False
    ),
    tools=[slm_help],
)

async def run_agent(question: str):
    result = await Runner.run(agent, question, max_turns=15)
    return result.final_output

# Test
async def main():
    qs = [
        "If 3x + 5 = 20, what is x?",
        "What is 520 + 650?",
        "Calculate 26 choose 2"
    ]

    for q in qs:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        out = await run_agent(q)
        print(f"Answer: {out}")
        print(f"{'='*60}")

if __name__ == "__main__":
    try:
        get_ipython()
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.run(main())
    except NameError:
        asyncio.run(main())