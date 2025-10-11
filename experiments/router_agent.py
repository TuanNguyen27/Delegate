# experiments/router_agent.py
"""
Router Agent with token tracking - Using Gemini 2.5 Flash
"""
import os, re, time, torch, asyncio
from dotenv import load_dotenv
import google.generativeai as genai
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

# Import API key manager  
from tools.api_key_manager import create_key_manager

# SLM lazy loader
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
        _TOK.padding_side = "left"
    return _SLM, _TOK


def slm_help_impl(question: str) -> str:
    """
    Implementation of SLM help tool with tracking.
    """
    print(f"[TOOL] slm_help: {question[:60]}...")
    
    try:
        model, tok = _lazy_load_slm()

        sys = "You are a math calculator. Solve step-by-step. Put final answer in \\boxed{} at the end."
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": question},
        ]
        
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([prompt], return_tensors="pt").to(model.device)
        
        # Count input tokens
        input_tokens = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        latency = time.time() - t0
        
        # Count output tokens (only new tokens)
        output_tokens = out.shape[1] - input_tokens
        
        gen = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

        # Extract boxed answer
        match = re.search(r'\\boxed\{([^}]+)\}', gen)
        
        # Log to tracker
        try:
            import sys as _sys
            if 'router_experiment' in _sys.modules:
                from experiments.router_experiment import tracker
                tracker.log_tool_call(question, gen, latency, input_tokens, output_tokens)
                print(f"[TRACKER] Logged: {latency:.2f}s, {input_tokens}→{output_tokens} tokens")
        except Exception as e:
            print(f"[TRACKER] Failed: {e}")

        if match:
            answer = match.group(1)
            result = f"CALCULATION COMPLETE: The answer is {answer}. Use this directly."
            print(f"[SLM] Answer: {answer} ({latency:.2f}s)")
            return result
        else:
            result = f"CALCULATION COMPLETE: {gen}"
            print(f"[SLM] No boxed answer ({latency:.2f}s)")
            return result
        
    except Exception as e:
        return f"CALCULATION ERROR: {str(e)}. Solve yourself."


# Define tool for Gemini function calling
slm_help_tool = genai.protos.Tool(
    function_declarations=[
        genai.protos.FunctionDeclaration(
            name="slm_help",
            description=(
                "Solve a mathematical calculation using specialized math model. "
                "Returns definitive answer that should be trusted immediately."
            ),
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "question": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="The calculation to perform"
                    )
                },
                required=["question"]
            )
        )
    ]
)

# Agent instructions
INSTRUCTIONS = (
    "You solve math problems step by step.\n\n"
    
    "WORKFLOW:\n"
    "1. Understand the problem\n"
    "2. For ANY calculation, call slm_help ONCE\n"
    "3. When you receive 'CALCULATION COMPLETE:', use that answer\n"
    "4. NEVER call slm_help again for the same calculation\n"
    "5. Provide final answer in \\boxed{}\n\n"
    
    "CRITICAL:\n"
    "• Call slm_help for ALL arithmetic/algebra\n"
    "• Trust 'CALCULATION COMPLETE' results immediately\n"
    "• Do NOT verify or retry calculations\n"
    "• After final answer, write \\boxed{answer} and STOP"
)

# Agent runner function (replaces Agent/Runner classes)
async def run_agent(question: str, max_turns: int = 15, key_manager=None):
    """
    Run the agent with Gemini 2.5 Flash and function calling.
    Returns final output and usage metadata.
    
    Args:
        question: The math problem to solve
        max_turns: Maximum function calling turns
        key_manager: APIKeyManager instance (created if None)
    """
    # Initialize key manager if not provided
    if key_manager is None:
        key_manager = create_key_manager(cooldown_seconds=60)
    
    # Get model with next available API key
    model = key_manager.get_model(
        model_name="gemini-2.5-flash",
        tools=[slm_help_tool],
        system_instruction=INSTRUCTIONS
    )
    
    # Start chat
    chat = model.start_chat(enable_automatic_function_calling=False)
    
    # Send initial question
    response = await asyncio.to_thread(chat.send_message, question)
    
    # Handle function calls in a loop
    for turn in range(max_turns):
        if not response.candidates or not response.candidates[0].content.parts:
            break
            
        parts = response.candidates[0].content.parts
        function_calls = [part.function_call for part in parts if hasattr(part, 'function_call') and part.function_call]
        
        if not function_calls:
            break
        
        # Execute function calls
        function_responses = []
        for fc in function_calls:
            if fc.name == "slm_help":
                question_param = fc.args.get("question", "")
                result = slm_help_impl(question_param)
                
                function_responses.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name="slm_help",
                            response={"result": result}
                        )
                    )
                )
        
        if function_responses:
            response = await asyncio.to_thread(chat.send_message, function_responses)
        else:
            break
    
    # Extract final answer
    final_text = ""
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text:
                final_text += part.text
    
    # Create result object with final_output attribute (to match old Agent API)
    class Result:
        def __init__(self, text):
            self.final_output = text
    
    return Result(final_text if final_text else "No response generated")