"""
Modal wrapper for SWE-bench with Router (Gemini + Qwen2.5-Coder).

Router system:
- Gemini analyzes GitHub issue and plans approach
- Delegates code generation to Qwen2.5-Coder-7B
- Integrates and verifies the result

Usage:
    modal run modal_swebench_router.py --samples 10 --seed 42
    modal run modal_swebench_router.py --samples 10 --model 7b  # or 32b
"""
import modal
import json
from pathlib import Path
import os
import re

# Create Modal app
app = modal.App("swebench-router")

# Model configurations
CODE_MODELS = {
    "7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
}

MODEL_CACHE_DIR = "/cache"
RESULTS_DIR = "/results"

# Create persistent volumes
model_volume = modal.Volume.from_name("model-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("swebench-results", create_if_missing=True)

# Create Modal image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch>=2.0.0",
        "transformers>=4.37.0",
        "accelerate>=0.27.0",
        "datasets>=2.18.0",
        "pandas>=2.0.0",
        "tqdm>=4.66.0",
        "google-generativeai>=0.3.0",
    )
)


# Router prompt for Gemini
ROUTER_INSTRUCTIONS = """You are an expert software engineering architect working with a specialized code generation model.

Your role:
1. Analyze the GitHub issue carefully
2. Identify what needs to be fixed
3. Plan the approach (which files, what changes)
4. Delegate code generation to the specialist model using the generate_code tool
5. Return ONLY the generated patch in unified diff format

CRITICAL: Your final response must be ONLY the patch in unified diff format, with no additional text, explanations, or markdown formatting.

When you receive a GitHub issue:
- Understand the problem thoroughly
- Identify the root cause and files to modify
- Call the generate_code tool with clear, specific instructions
- After receiving the generated code, extract and return ONLY the unified diff patch
- Do NOT add explanations, markdown code blocks, or any wrapper text

Your final output should look exactly like:
diff --git a/path/to/file.py b/path/to/file.py
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context
-old line
+new line
 context
"""


def extract_patch_from_response(text: str) -> str:
    """
    Extract unified diff patch from LLM response.
    Handles cases where patch is wrapped in markdown or has explanatory text.
    """
    if not text:
        return ""
    
    # Try to extract from markdown code block first
    markdown_pattern = r'```(?:diff)?\n(diff --git.*?)```'
    match = re.search(markdown_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try to find patch starting with "diff --git"
    diff_pattern = r'(diff --git.*?)(?:\n\n[A-Z]|\Z)'
    match = re.search(diff_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If text starts with diff, assume it's already a clean patch
    if text.strip().startswith('diff --git'):
        return text.strip()
    
    # Last resort: return as-is
    return text.strip()


@app.cls(
    image=image,
    gpu="A100-40GB",  # For 7B code model
    timeout=3600,
    volumes={MODEL_CACHE_DIR: model_volume},
    scaledown_window=300,
    secrets=[modal.Secret.from_name("google-api-key")],
)
class RouterWithQwen7B:
    """Router system: Gemini (planning) + Qwen2.5-Coder-7B (execution)."""
    
    model_size = "7b"
    code_model_name = CODE_MODELS["7b"]
    
    @modal.enter()
    def load_models(self):
        """Load code generation model and initialize Gemini."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import google.generativeai as genai
        
        print(f"Loading code model: {self.code_model_name}...")
        print(f"GPU: A100-40GB")
        
        # Load Qwen for code generation
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.code_model_name,
            cache_dir=MODEL_CACHE_DIR,
            trust_remote_code=True,
        )
        
        self.code_model = AutoModelForCausalLM.from_pretrained(
            self.code_model_name,
            cache_dir=MODEL_CACHE_DIR,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        print(f"✅ Code model loaded")
        
        # Initialize Gemini
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in secrets")
        
        genai.configure(api_key=api_key)
        
        # Define code generation tool for Gemini
        self.code_gen_tool = genai.protos.Tool(
            function_declarations=[
                genai.protos.FunctionDeclaration(
                    name="generate_code",
                    description="Generate code to fix the GitHub issue. Provide clear instructions about what code to generate.",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "instructions": genai.protos.Schema(
                                type=genai.protos.Type.STRING,
                                description="Detailed instructions for what code to generate, including context about the issue and desired fix"
                            ),
                            "context": genai.protos.Schema(
                                type=genai.protos.Type.STRING,
                                description="Additional context: file paths, function names, relevant code snippets"
                            ),
                        },
                        required=["instructions"]
                    )
                )
            ]
        )
        
        # Create Gemini model with tool
        self.llm = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            tools=[self.code_gen_tool],
            system_instruction=ROUTER_INSTRUCTIONS
        )
        
        print(f"✅ Router initialized (Gemini + Qwen-7B)")
    
    def _generate_code_with_specialist(self, instructions: str, context: str = "") -> str:
        """Use Qwen to generate code based on instructions."""
        import torch
        
        # Format prompt for code model
        prompt = f"""Generate code to fix this issue:

Instructions:
{instructions}

Context:
{context}

Provide the complete code fix."""
        
        messages = [
            {"role": "system", "content": "You are an expert software engineer. Generate precise code fixes."},
            {"role": "user", "content": prompt}
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.code_model.device)
        
        with torch.no_grad():
            generated_ids = self.code_model.generate(
                **model_inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_ids = generated_ids[:, model_inputs['input_ids'].shape[1]:]
        code = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return code
    
    @modal.method()
    def generate_patch_with_router(self, problem_statement: str, repo: str, 
                                   hints: str = "") -> dict:
        """
        Generate patch using router system.
        
        Args:
            problem_statement: GitHub issue description
            repo: Repository name
            hints: Optional hints
            
        Returns:
            Dictionary with completion, routing info, and metadata
        """
        import time
        import google.generativeai as genai
        
        # Format issue for Gemini
        issue_prompt = f"""Repository: {repo}

GitHub Issue:
{problem_statement}
"""
        if hints:
            issue_prompt += f"""

Hints about relevant code:
{hints}
"""
        
        issue_prompt += """

Analyze this issue and generate a patch to fix it. Use the code_generation tool to delegate the actual code writing."""
        
        print(f"  [Router] Analyzing issue...")
        t_start = time.time()
        
        # Start chat with Gemini
        chat = self.llm.start_chat()
        response = chat.send_message(issue_prompt)
        
        llm_latency = time.time() - t_start
        tool_calls = 0
        slm_latency = 0.0
        final_response = ""
        
        # Track conversation for debugging
        conversation_log = []
        conversation_log.append({
            "role": "user",
            "content": issue_prompt
        })
        
        # Handle function calling loop
        max_turns = 3
        for turn in range(max_turns):
            # Check for function calls
            if not response.candidates:
                break
            
            parts = response.candidates[0].content.parts
            function_calls = [part for part in parts if hasattr(part, 'function_call')]
            
            # Log Gemini's response (text or function call)
            gemini_response = {
                "role": "assistant",
                "turn": turn,
                "has_function_calls": len(function_calls) > 0,
                "function_calls": []
            }
            
            # Try to get text response
            try:
                gemini_response["text"] = response.text
            except:
                gemini_response["text"] = ""
            
            if not function_calls:
                # No more function calls, get final response
                try:
                    final_response = response.text
                except:
                    final_response = ""
                conversation_log.append(gemini_response)
                break
            
            # Execute function calls
            function_responses = []
            for fc in function_calls:
                if fc.function_call.name == "generate_code":
                    tool_calls += 1
                    args = dict(fc.function_call.args)
                    instructions = args.get("instructions", "")
                    context = args.get("context", "")
                    
                    print(f"  [Router] Delegating to Qwen-{self.model_size.upper()}...")
                    print(f"    Instructions: {instructions[:100]}...")
                    
                    # Log the function call
                    gemini_response["function_calls"].append({
                        "name": "generate_code",
                        "instructions": instructions,
                        "context": context
                    })
                    
                    # Call specialist model
                    t_slm_start = time.time()
                    generated_code = self._generate_code_with_specialist(instructions, context)
                    slm_latency += time.time() - t_slm_start
                    
                    print(f"  [Router] Code generated ({len(generated_code)} chars)")
                    
                    # Log SLM response
                    conversation_log.append({
                        "role": "tool",
                        "tool_name": "generate_code",
                        "slm_model": f"Qwen-{self.model_size}",
                        "output": generated_code,
                        "output_length": len(generated_code)
                    })
                    
                    # Return to Gemini
                    function_responses.append(
                        genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name="generate_code",
                                response={"code": generated_code}
                            )
                        )
                    )
            
            conversation_log.append(gemini_response)
            
            # Send function responses back to Gemini (only if we have responses)
            if function_responses:
                t_llm_start = time.time()
                response = chat.send_message(function_responses)
                llm_latency += time.time() - t_llm_start
            else:
                # No function responses to send, break the loop
                break
        
        total_latency = time.time() - t_start
        
        # Extract final patch
        try:
            if not final_response and response.text:
                final_response = response.text
        except:
            pass
        
        # Clean up the response to extract just the patch
        clean_patch = extract_patch_from_response(final_response)
        
        print(f"  [Router] Complete: {tool_calls} delegations, {total_latency:.2f}s total")
        if clean_patch != final_response:
            print(f"  [Router] Extracted clean patch ({len(clean_patch)} chars from {len(final_response)} chars)")
        
        return {
            "completion": clean_patch,
            "latency": total_latency,
            "llm_latency": llm_latency,
            "slm_latency": slm_latency,
            "tool_calls": tool_calls,
            "length": len(clean_patch),
            "routing_used": tool_calls > 0,
            "conversation": conversation_log,
            "raw_response": final_response
        }


@app.cls(
    image=image,
    gpu="A100-80GB",  # For 32B code model
    timeout=3600,
    volumes={MODEL_CACHE_DIR: model_volume},
    scaledown_window=300,
    secrets=[modal.Secret.from_name("google-api-key")],
)
class RouterWithQwen32B:
    """Router system: Gemini (planning) + Qwen2.5-Coder-32B (execution)."""
    
    model_size = "32b"
    code_model_name = CODE_MODELS["32b"]
    
    @modal.enter()
    def load_models(self):
        """Load code generation model and initialize Gemini."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import google.generativeai as genai
        
        print(f"Loading code model: {self.code_model_name}...")
        print(f"GPU: A100-80GB")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.code_model_name,
            cache_dir=MODEL_CACHE_DIR,
            trust_remote_code=True,
        )
        
        self.code_model = AutoModelForCausalLM.from_pretrained(
            self.code_model_name,
            cache_dir=MODEL_CACHE_DIR,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        print(f"✅ Code model loaded")
        
        # Initialize Gemini (same as 7B version)
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in secrets")
        
        genai.configure(api_key=api_key)
        
        self.code_gen_tool = genai.protos.Tool(
            function_declarations=[
                genai.protos.FunctionDeclaration(
                    name="generate_code",
                    description="Generate code to fix the GitHub issue.",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "instructions": genai.protos.Schema(
                                type=genai.protos.Type.STRING,
                                description="Detailed instructions for code generation"
                            ),
                            "context": genai.protos.Schema(
                                type=genai.protos.Type.STRING,
                                description="Additional context"
                            ),
                        },
                        required=["instructions"]
                    )
                )
            ]
        )
        
        self.llm = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            tools=[self.code_gen_tool],
            system_instruction=ROUTER_INSTRUCTIONS
        )
        
        print(f"✅ Router initialized (Gemini + Qwen-32B)")
    
    def _generate_code_with_specialist(self, instructions: str, context: str = "") -> str:
        """Use Qwen-32B to generate code."""
        import torch
        
        prompt = f"""Generate code to fix this issue:

Instructions:
{instructions}

Context:
{context}

Provide the complete code fix."""
        
        messages = [
            {"role": "system", "content": "You are an expert software engineer. Generate precise code fixes."},
            {"role": "user", "content": prompt}
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.code_model.device)
        
        with torch.no_grad():
            generated_ids = self.code_model.generate(
                **model_inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_ids = generated_ids[:, model_inputs['input_ids'].shape[1]:]
        code = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return code
    
    @modal.method()
    def generate_patch_with_router(self, problem_statement: str, repo: str, 
                                   hints: str = "") -> dict:
        """Generate patch using router with 32B model."""
        import time
        import google.generativeai as genai
        
        issue_prompt = f"""Repository: {repo}

GitHub Issue:
{problem_statement}
"""
        if hints:
            issue_prompt += f"""

Hints:
{hints}
"""
        
        issue_prompt += """

Analyze and fix this issue using the code_generation tool."""
        
        print(f"  [Router] Analyzing issue...")
        t_start = time.time()
        
        chat = self.llm.start_chat()
        response = chat.send_message(issue_prompt)
        
        llm_latency = time.time() - t_start
        tool_calls = 0
        slm_latency = 0.0
        final_response = ""
        
        # Track conversation for debugging
        conversation_log = []
        conversation_log.append({
            "role": "user",
            "content": issue_prompt
        })
        
        max_turns = 3
        for turn in range(max_turns):
            if not response.candidates:
                break
            
            parts = response.candidates[0].content.parts
            function_calls = [part for part in parts if hasattr(part, 'function_call')]
            
            # Log Gemini's response
            gemini_response = {
                "role": "assistant",
                "turn": turn,
                "has_function_calls": len(function_calls) > 0,
                "function_calls": []
            }
            
            try:
                gemini_response["text"] = response.text
            except:
                gemini_response["text"] = ""
            
            if not function_calls:
                try:
                    final_response = response.text
                except:
                    final_response = ""
                conversation_log.append(gemini_response)
                break
            
            function_responses = []
            for fc in function_calls:
                if fc.function_call.name == "generate_code":
                    tool_calls += 1
                    args = dict(fc.function_call.args)
                    
                    print(f"  [Router] Delegating to Qwen-{self.model_size.upper()}...")
                    
                    # Log the function call
                    gemini_response["function_calls"].append({
                        "name": "generate_code",
                        "instructions": args.get("instructions", ""),
                        "context": args.get("context", "")
                    })
                    
                    t_slm_start = time.time()
                    generated_code = self._generate_code_with_specialist(
                        args.get("instructions", ""),
                        args.get("context", "")
                    )
                    slm_latency += time.time() - t_slm_start
                    
                    print(f"  [Router] Code generated ({len(generated_code)} chars)")
                    
                    # Log SLM response
                    conversation_log.append({
                        "role": "tool",
                        "tool_name": "generate_code",
                        "slm_model": f"Qwen-{self.model_size}",
                        "output": generated_code,
                        "output_length": len(generated_code)
                    })
                    
                    function_responses.append(
                        genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name="generate_code",
                                response={"code": generated_code}
                            )
                        )
                    )
            
            conversation_log.append(gemini_response)
            
            # Send function responses back to Gemini (only if we have responses)
            if function_responses:
                t_llm_start = time.time()
                response = chat.send_message(function_responses)
                llm_latency += time.time() - t_llm_start
            else:
                # No function responses to send, break the loop
                break
        
        total_latency = time.time() - t_start
        
        try:
            if not final_response and response.text:
                final_response = response.text
        except:
            pass
        
        # Clean up the response to extract just the patch
        clean_patch = extract_patch_from_response(final_response)
        
        print(f"  [Router] Complete: {tool_calls} delegations, {total_latency:.2f}s")
        if clean_patch != final_response:
            print(f"  [Router] Extracted clean patch ({len(clean_patch)} chars from {len(final_response)} chars)")
        
        return {
            "completion": clean_patch,
            "latency": total_latency,
            "llm_latency": llm_latency,
            "slm_latency": slm_latency,
            "tool_calls": tool_calls,
            "length": len(clean_patch),
            "routing_used": tool_calls > 0,
            "conversation": conversation_log,
            "raw_response": final_response
        }


@app.function(
    image=image,
    timeout=600,
    volumes={RESULTS_DIR: results_volume},
)
def save_to_volume(results: dict, filename: str):
    """Save results to Modal volume for later evaluation."""
    import json
    from pathlib import Path
    
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    filepath = Path(RESULTS_DIR) / filename
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    results_volume.commit()
    print(f"✅ Saved to Modal volume: {filepath}")


@app.function(image=image, timeout=3600)
def load_swebench_dataset(n_samples: int = 10, random_seed: int = 42):
    """Load SWE-bench Lite dataset."""
    from datasets import load_dataset
    import pandas as pd
    
    print(f"Loading SWE-bench Lite (n={n_samples}, seed={random_seed})...")
    
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    df = pd.DataFrame({
        'instance_id': ds['instance_id'],
        'repo': ds['repo'],
        'problem_statement': ds['problem_statement'],
        'patch': ds['patch'],
        'hints_text': ds['hints_text'],
    })
    
    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)
    
    print(f"✅ Loaded {len(df)} problems")
    return df.to_dict('records')


@app.function(image=image, timeout=3600)
def evaluate_router(
    n_samples: int = 10,
    random_seed: int = 42,
    model_size: str = "7b",
):
    """Run SWE-bench evaluation with router system."""
    import time
    
    code_model_name = CODE_MODELS[model_size]
    
    print(f"\n{'='*60}")
    print(f"SWE-BENCH ROUTER EVALUATION")
    print(f"{'='*60}")
    print(f"Router: Gemini 2.0 Flash")
    print(f"Code Model: {code_model_name}")
    print(f"Samples: {n_samples}")
    print(f"{'='*60}\n")
    
    problems = load_swebench_dataset.remote(n_samples, random_seed)
    
    if model_size == "7b":
        router = RouterWithQwen7B()
    elif model_size == "32b":
        router = RouterWithQwen32B()
    else:
        raise ValueError(f"Invalid model size: {model_size}")
    
    results = []
    total_latency = 0.0
    
    for i, problem in enumerate(problems):
        print(f"[{i+1}/{len(problems)}] {problem['instance_id']}...")
        
        try:
            result = router.generate_patch_with_router.remote(
                problem_statement=problem['problem_statement'],
                repo=problem['repo'],
                hints=problem.get('hints_text', '')
            )
            
            has_output = len(result['completion'].strip()) > 0
            
        except Exception as e:
            print(f"  ERROR: {e}")
            result = {
                "completion": "",
                "latency": 0.0,
                "llm_latency": 0.0,
                "slm_latency": 0.0,
                "tool_calls": 0,
                "length": 0,
                "routing_used": False
            }
            has_output = False
        
        total_latency += result['latency']
        
        results.append({
            'instance_id': problem['instance_id'],
            'repo': problem['repo'],
            'problem_statement': problem['problem_statement'],
            'ground_truth_patch': problem['patch'],
            'prediction': result['completion'],
            'has_output': has_output,
            'latency_total': result['latency'],
            'latency_llm': result['llm_latency'],
            'latency_slm': result['slm_latency'],
            'tool_calls': result['tool_calls'],
            'routing_used': result['routing_used'],
            'prediction_length': result['length'],
            'conversation': result.get('conversation', []),
            'raw_response': result.get('raw_response', '')
        })
        
        status = "✓" if has_output else "✗"
        print(f"  {status} | {result['latency']:.2f}s | {result['tool_calls']} calls\n")
    
    # Summary
    n_with_output = sum(r['has_output'] for r in results)
    n_with_routing = sum(r['routing_used'] for r in results)
    avg_tool_calls = sum(r['tool_calls'] for r in results) / len(results)
    
    summary = {
        'total': len(results),
        'with_output': n_with_output,
        'output_rate': n_with_output / len(results),
        'routing_rate': n_with_routing / len(results),
        'avg_tool_calls': avg_tool_calls,
        'avg_latency_total': total_latency / len(results),
        'avg_latency_llm': sum(r['latency_llm'] for r in results) / len(results),
        'avg_latency_slm': sum(r['latency_slm'] for r in results) / len(results),
        'total_latency': total_latency,
        'model': f"Router (Gemini + {code_model_name})",
        'code_model_size': model_size,
    }
    
    print(f"\n{'='*60}")
    print(f"ROUTER RESULTS")
    print(f"{'='*60}")
    print(f"Output rate: {summary['output_rate']:.1%}")
    print(f"Routing used: {summary['routing_rate']:.1%}")
    print(f"Avg tool calls: {summary['avg_tool_calls']:.2f}")
    print(f"Avg latency: {summary['avg_latency_total']:.2f}s")
    print(f"  LLM: {summary['avg_latency_llm']:.2f}s")
    print(f"  SLM: {summary['avg_latency_slm']:.2f}s")
    print(f"{'='*60}\n")
    
    return {'summary': summary, 'results': results}


@app.local_entrypoint()
def main(
    samples: int = 10,
    seed: int = 42,
    model: str = "7b",
    output: str = None,
):
    """Run router evaluation."""
    import json
    from datetime import datetime
    
    if model not in CODE_MODELS:
        print(f"❌ Invalid model: {model}")
        return
    
    if output is None:
        output = f"results_swebench_router_{model}.json"
    
    print(f"\n🚀 Starting Router Evaluation...")
    print(f"   Router: Gemini 2.0 Flash")
    print(f"   Code Model: {CODE_MODELS[model]}")
    print(f"   Samples: {samples}\n")
    
    results = evaluate_router.remote(
        n_samples=samples,
        random_seed=seed,
        model_size=model,
    )
    
    results['timestamp'] = datetime.now().isoformat()
    
    # Save to local file
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved locally to {output}")
    
    # Save to Modal volume
    save_to_volume.remote(results, output)
    print(f"✅ Results also saved to Modal volume: {output}")
    
    return results

