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
    print(f"[TOOL] slm_help: {question}...")
    
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
        
        # Log to tracker (always attempt, this is only used in experiments)
        try:
            from experiments.router_experiment import tracker
            tracker.log_tool_call(question, gen, latency, input_tokens, output_tokens)
            print(f"[TRACKER] Logged: {latency:.2f}s, {input_tokens}→{output_tokens} tokens")
        except Exception as e:
            # Tracker not available (shouldn't happen in experiments, but fail gracefully)
            print(f"[TRACKER] Warning: Could not log tool call: {e}")

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
# Import centralized prompts (must be before tool definition)
import sys as _sys_prompt
from pathlib import Path as _Path_prompt
_sys_prompt.path.insert(0, str(_Path_prompt(__file__).parent.parent))
from prompts import ROUTER_INSTRUCTIONS_EXPERIMENT, ROUTER_TOOL_DESCRIPTION, ROUTER_TOOL_PARAMETER_DESCRIPTION

slm_help_tool = genai.protos.Tool(
    function_declarations=[
        genai.protos.FunctionDeclaration(
            name="slm_help",
            description=ROUTER_TOOL_DESCRIPTION,
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "question": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description=ROUTER_TOOL_PARAMETER_DESCRIPTION
                    )
                },
                required=["question"]
            )
        )
    ]
)

# Agent instructions (imported from centralized prompts above)
INSTRUCTIONS = ROUTER_INSTRUCTIONS_EXPERIMENT

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
        key_manager = create_key_manager(cooldown_seconds=1)
    
    # Get model with next available API key
    model = key_manager.get_model(
        model_name="gemini-2.5-flash-lite",
        tools=[slm_help_tool],
        system_instruction=INSTRUCTIONS
    )
    
    # Start chat
    chat = model.start_chat(enable_automatic_function_calling=False)
    
    # Track token usage across all turns
    total_input_tokens = 0
    total_output_tokens = 0
    
    # Send initial question
    response = await asyncio.to_thread(chat.send_message, question)
    
    # Track tokens from initial response
    if hasattr(response, 'usage_metadata'):
        total_input_tokens += response.usage_metadata.prompt_token_count
        total_output_tokens += response.usage_metadata.candidates_token_count
    
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
            # Track tokens from function response
            if hasattr(response, 'usage_metadata'):
                total_input_tokens += response.usage_metadata.prompt_token_count
                total_output_tokens += response.usage_metadata.candidates_token_count
        else:
            break
    
    # Extract final answer with finish_reason handling
    final_text = ""
    finish_reason = None
    
    if response.candidates:
        finish_reason = response.candidates[0].finish_reason
        
        # Handle problematic finish reasons
        if finish_reason == 2:  # MAX_TOKENS
            final_text = "[INCOMPLETE - Hit max tokens]"
        elif finish_reason == 3:  # SAFETY
            final_text = "[BLOCKED - Safety filter]"
        elif finish_reason == 4:  # RECITATION
            final_text = "[BLOCKED - Recitation filter]"
        elif response.candidates[0].content.parts:
            # Extract text normally for STOP (1) or other valid reasons
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    final_text += part.text
    
    # Create result object with final_output attribute (to match old Agent API)
    class Result:
        def __init__(self, text, input_tokens=0, output_tokens=0):
            self.final_output = text
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens
    
    return Result(
        final_text if final_text else "No response generated",
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens
    )