# router_agent_demo.py
import os, re, time, asyncio, torch, json
from dotenv import load_dotenv

import google.generativeai as genai
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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
def slm_help_impl(question: str) -> str:
    """
    Implementation of SLM help tool - called when Gemini invokes the tool.
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
            result = f"CALCULATION COMPLETE: The answer is {answer}"
            print(f"[SLM RESULT] Extracted answer: {answer}")
            return result
        else:
            result = f"CALCULATION COMPLETE: {gen}"
            print(f"[SLM RESULT] No boxed answer found, returning full output")
            return result
        
    except Exception as e:
        return f"CALCULATION ERROR: {str(e)}. Please try a different approach."

# Define tool for Gemini function calling
slm_help_tool = genai.protos.Tool(
    function_declarations=[
        genai.protos.FunctionDeclaration(
            name="slm_help",
            description=(
                "Use specialized math model to solve ANY calculation or equation. "
                "IMPORTANT: You MUST use this tool for ALL arithmetic operations, "
                "no matter how simple (addition, subtraction, multiplication, division, etc.)"
            ),
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "question": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="The mathematical question or calculation to solve"
                    )
                },
                required=["question"]
            )
        )
    ]
)

# ---------------------------
# Agent with STRONGER routing
# ---------------------------
INSTRUCTIONS = """You are a math problem solver with access to a specialized calculation tool. You reason and think in steps.

You MUST use the `slm_help` tool for EVERY calculation, including:
- Basic arithmetic (addition, subtraction, multiplication, division)
- Percentages and fractions
- Equations and algebra
- Any operation involving numbers

## Your Workflow:

1. Read the problem carefully
2. Identify what calculations are needed
3. For EACH calculation (no matter how simple):
   - Call slm_help(question) with the specific calculation
   - Wait for the result
   - Integrate that result in your reasoning
4. After all calculations are complete, provide the final answer in \\boxed{} format

## Examples:

Problem: "What is 156 + 243?"
→ CORRECT: Call slm_help("156 + 243")
→ WRONG: Answering "399" directly

Problem: "If I buy 3 shirts at $15 each, what's the total?"
→ CORRECT: Call slm_help("3 × 15")
→ WRONG: Saying "3 times 15 is 45"

Problem: "Natalia sold 48 clips in April and half as many in May. Total?"
→ Step 1: Call slm_help("48 ÷ 2") to find May's amount
→ Step 2: Call slm_help("48 + [May's amount]") to find total

## Remember:
- ALWAYS use the tool for calculations
- NEVER compute anything yourself
- Present final answer as \\boxed{answer}
"""

# ---------------------------
# Agent Runner with Gemini
# ---------------------------
async def run_agent(question: str, max_turns: int = 15):
    """
    Run the agent with Gemini 2.5 Flash and function calling.
    """
    # Initialize Gemini model with function calling
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        tools=[slm_help_tool],
        system_instruction=INSTRUCTIONS
    )
    
    # Start chat
    chat = model.start_chat(enable_automatic_function_calling=False)
    
    # Send initial question
    response = await asyncio.to_thread(chat.send_message, question)
    
    # Handle function calls in a loop (max_turns to prevent infinite loops)
    for turn in range(max_turns):
        # Check if we have function calls
        if not response.candidates or not response.candidates[0].content.parts:
            break
            
        parts = response.candidates[0].content.parts
        function_calls = [part.function_call for part in parts if part.function_call]
        
        if not function_calls:
            # No more function calls, we're done
            break
        
        # Execute each function call
        function_responses = []
        for fc in function_calls:
            if fc.name == "slm_help":
                # Extract the question parameter
                question_param = fc.args.get("question", "")
                
                # Call the SLM
                result = slm_help_impl(question_param)
                
                # Create function response
                function_responses.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name="slm_help",
                            response={"result": result}
                        )
                    )
                )
        
        # Send function responses back to Gemini
        if function_responses:
            response = await asyncio.to_thread(
                chat.send_message,
                function_responses
            )
        else:
            break
    
    # Extract final answer
    if response.candidates and response.candidates[0].content.parts:
        final_text = ""
        for part in response.candidates[0].content.parts:
            if part.text:
                final_text += part.text
        return final_text
    
    return "No response generated"