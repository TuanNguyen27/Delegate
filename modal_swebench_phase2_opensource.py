"""
SWE-bench Phase 2: Open-Source Version
Uses Qwen 32B (8-bit) as router + Qwen 7B as worker on single A100-80GB

This avoids Gemini API rate limits by using open-source models.
"""

import modal
import os
import re
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests

# Modal setup
app = modal.App("swebench-phase2-opensource")

# Create volume for storing results
volume = modal.Volume.from_name("swebench-results", create_if_missing=True)

# Docker image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets>=2.14.0",
        "requests>=2.31.0",
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",  # For 8-bit quantization
    )
)

# Models
ROUTER_MODEL = "Qwen/Qwen2.5-32B-Instruct"  # Router (8-bit)
WORKER_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"  # Worker (bfloat16)


# ============================================================================
# ROUTER SYSTEM PROMPT (with localization examples)
# ============================================================================

ROUTER_SYSTEM_PROMPT = """You are an expert software engineering assistant. Your role is to:
1. Analyze bug reports
2. Explore repositories to localize bugs
3. Delegate patch generation to a specialized code generation tool

You have access to these tools:
- fetch_file_metadata: Get file info (size, functions, classes)
- search_codebase: Search for function/class definitions
- fetch_code_section: Get specific lines from a file
- generate_patch_with_qwen: Delegate patch generation to Qwen SLM

WORKFLOW:
1. Analyze problem statement and hints
2. Use hints to identify relevant files (or search if no hints)
3. Fetch file metadata to understand structure
4. Search for relevant functions/classes
5. Fetch focused code section (50-200 lines max)
6. Delegate to generate_patch_with_qwen tool

CRITICAL RULES:
- Make decisions quickly (max 5-7 tool calls)
- Don't repeat the same tool call
- Fetch file metadata before fetching code
- ALWAYS delegate patch generation to generate_patch_with_qwen (your specialized worker)
- Your job is localization, not patch generation
- Never generate patches yourself - use the tool!

---
LOCALIZATION EXAMPLES:

EXAMPLE 1: Simple Bug with Hint
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROBLEM: The calculate_total() function returns incorrect values when discount is applied.
HINTS: ecommerce/cart/cart.py

ACTIONS:
1. fetch_file_metadata("ecommerce/cart/cart.py") → ShoppingCart class found
2. fetch_code_section("ecommerce/cart/cart.py", function="calculate_total") → Got code
3. generate_patch_with_qwen(problem, code, file) → Patch generated

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLE 2: Large File Requiring Search
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROBLEM: simplify(cos(x)**I) hangs with complex exponents.
HINTS: sympy/simplify/fu.py

ACTIONS:
1. fetch_file_metadata("sympy/simplify/fu.py") → 691 functions, 66KB
2. search_codebase("sympy/simplify/fu.py", "power trig") → Found _TR56
3. fetch_code_section("sympy/simplify/fu.py", function="_TR56") → Got code
4. generate_patch_with_qwen(problem, code, file) → Patch generated

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLE 3: No Hints - Must Search
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROBLEM: Email validation is inconsistent between model and serializer.
HINTS: (none)

ACTIONS:
1. search_codebase(".", "email validation") → Found users/models.py, users/serializers.py
2. fetch_file_metadata("users/models.py") → User model found
3. fetch_code_section("users/models.py", class_name="User") → Got code
4. fetch_code_section("users/serializers.py", class_name="UserSerializer") → Got code
5. generate_patch_with_qwen(problem, combined_code, files) → Patch generated

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Now, localize and fix this REAL problem. Be efficient - max 5-7 tool calls!
"""


# ============================================================================
# WORKER PROMPT (for patch generation)
# ============================================================================

WORKER_SYSTEM_PROMPT = """You are an expert code generator specializing in bug fixes. Generate a unified diff patch to fix bugs.

CRITICAL RULES:
1. Output ONLY the patch in unified diff format
2. Start directly with "--- a/" (no markdown, no explanation)
3. Make MINIMAL changes (only fix the bug)
4. Preserve all formatting, indentation, and whitespace
5. Include 3 lines of context before and after changes

Generate the patch:
"""


# ============================================================================
# HELPER FUNCTIONS (same as Phase 2)
# ============================================================================

def fetch_file_metadata_impl(repo: str, commit: str, file_path: str) -> Dict:
    """Fetch file metadata: size, functions, classes."""
    try:
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{file_path}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        content = response.text
        lines = content.splitlines()
        
        functions = []
        classes = []
        
        if file_path.endswith('.py'):
            for i, line in enumerate(lines):
                if line.strip().startswith('def '):
                    func_name = line.strip().split('(')[0].replace('def ', '')
                    functions.append({'name': func_name, 'line': i + 1})
                elif line.strip().startswith('class '):
                    class_name = line.strip().split('(')[0].split(':')[0].replace('class ', '')
                    classes.append({'name': class_name, 'line': i + 1})
        
        return {
            'file_path': file_path,
            'size': len(content),
            'lines': len(lines),
            'functions': functions[:20],  # Limit to first 20
            'classes': classes[:10],  # Limit to first 10
            'language': 'python' if file_path.endswith('.py') else 'unknown'
        }
    except Exception as e:
        return {'error': str(e), 'file_path': file_path}


def search_codebase_impl(repo: str, commit: str, file_path: str, query: str) -> List[Dict]:
    """Search for functions/classes matching query."""
    try:
        metadata = fetch_file_metadata_impl(repo, commit, file_path)
        if 'error' in metadata:
            return []
        
        results = []
        query_lower = query.lower()
        
        for func in metadata.get('functions', []):
            if query_lower in func['name'].lower():
                results.append({
                    'type': 'function',
                    'name': func['name'],
                    'line': func['line'],
                    'file_path': file_path
                })
        
        for cls in metadata.get('classes', []):
            if query_lower in cls['name'].lower():
                results.append({
                    'type': 'class',
                    'name': cls['name'],
                    'line': cls['line'],
                    'file_path': file_path
                })
        
        return results[:10]  # Limit to top 10
    except Exception as e:
        return []


def fetch_code_section_impl(
    repo: str,
    commit: str,
    file_path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    function_name: Optional[str] = None,
    context_lines: int = 10
) -> str:
    """Fetch specific code section from file."""
    try:
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{file_path}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        lines = response.text.splitlines()
        
        if function_name:
            target = f"def {function_name}"
            for i, line in enumerate(lines):
                if target in line:
                    start_line = max(1, i + 1 - context_lines)
                    end_line = min(len(lines), i + 50)
                    break
        
        if start_line is None:
            start_line = 1
        if end_line is None:
            end_line = min(len(lines), start_line + 100)
        
        section_lines = lines[start_line - 1:end_line]
        numbered_lines = [
            f"{i + start_line:4d} | {line}"
            for i, line in enumerate(section_lines)
        ]
        
        return "\n".join(numbered_lines)
    except Exception as e:
        return f"Error: {e}"


def extract_patch_from_response(response: str) -> str:
    """Extract patch from LLM response."""
    response = re.sub(r'```diff\n?', '', response)
    response = re.sub(r'```\n?', '', response)
    
    lines = response.split('\n')
    patch_lines = []
    in_patch = False
    
    for line in lines:
        if line.startswith('--- a/') or line.startswith('diff --git'):
            in_patch = True
        if in_patch:
            patch_lines.append(line)
    
    return '\n'.join(patch_lines).strip()


# ============================================================================
# DUAL QWEN SYSTEM (Router + Worker on same GPU)
# ============================================================================

@app.cls(
    image=image,
    gpu="A100-80GB",
    timeout=1800,  # 30 minutes for model loading + inference
    scaledown_window=600,  # Keep warm for 10 minutes
)
class DualQwenSystem:
    """
    Runs both Qwen 32B (router) and Qwen 7B (worker) on single A100-80GB.
    
    Memory usage:
    - Qwen 32B (8-bit): ~39 GB
    - Qwen 7B (bfloat16): ~18 GB
    - Total: ~57 GB (fits in 80GB!)
    """
    
    @modal.enter()
    def load_models(self):
        """Load both models on container startup."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print("="*80)
        print("Loading Dual Qwen System...")
        print("="*80)
        
        # Load Router (Qwen 32B in 8-bit)
        print(f"\n1. Loading Router: {ROUTER_MODEL} (8-bit)...")
        router_start = time.time()
        
        self.router_tokenizer = AutoTokenizer.from_pretrained(
            ROUTER_MODEL,
            trust_remote_code=True
        )
        
        self.router_model = AutoModelForCausalLM.from_pretrained(
            ROUTER_MODEL,
            load_in_8bit=True,  # 8-bit quantization
            device_map="auto",
            trust_remote_code=True
        )
        
        router_time = time.time() - router_start
        print(f"   ✅ Router loaded in {router_time:.1f}s")
        print(f"   Memory: ~39 GB (8-bit)")
        
        # Load Worker (Qwen 7B in bfloat16)
        print(f"\n2. Loading Worker: {WORKER_MODEL} (bfloat16)...")
        worker_start = time.time()
        
        self.worker_tokenizer = AutoTokenizer.from_pretrained(
            WORKER_MODEL,
            trust_remote_code=True
        )
        
        self.worker_model = AutoModelForCausalLM.from_pretrained(
            WORKER_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        worker_time = time.time() - worker_start
        print(f"   ✅ Worker loaded in {worker_time:.1f}s")
        print(f"   Memory: ~18 GB (bfloat16)")
        
        print(f"\n{'='*80}")
        print(f"✅ Both models loaded! Total time: {router_time + worker_time:.1f}s")
        print(f"   Estimated memory: ~57 GB / 80 GB")
        print(f"{'='*80}\n")
    
    @modal.method()
    def generate_with_router(
        self,
        messages: List[Dict],
        tools: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> Dict:
        """Generate response with router using native function calling."""
        import torch
        import json
        
        # Convert tools to Qwen's native function calling format
        qwen_tools = []
        for tool in tools:
            # Build parameters schema
            properties = {}
            required = []
            
            for param in tool['parameters']:
                properties[param] = {
                    "type": "string",
                    "description": f"The {param} parameter"
                }
                required.append(param)
            
            qwen_tools.append({
                "type": "function",
                "function": {
                    "name": tool['name'],
                    "description": tool['description'],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            })
        
        # Use Qwen's native function calling via chat template
        try:
            text = self.router_tokenizer.apply_chat_template(
                messages,
                tools=qwen_tools if qwen_tools else None,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            # Fallback if tools parameter not supported
            print(f"  ⚠️  Native function calling not supported, using manual format: {e}")
            text = self.router_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Add tools manually
            if tools:
                tools_desc = "\n\nAvailable tools:\n"
                for tool in tools:
                    params = ", ".join(tool['parameters'])
                    tools_desc += f"- {tool['name']}({params}): {tool['description']}\n"
                tools_desc += "\nTo use a tool, output: TOOL_CALL: {\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n"
                text = text.replace("</s>", tools_desc + "</s>")
        
        # Generate
        inputs = self.router_tokenizer([text], return_tensors="pt").to(self.router_model.device)
        
        with torch.no_grad():
            outputs = self.router_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.router_tokenizer.eos_token_id
            )
        
        response_text = self.router_tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse tool calls from response
        tool_calls = []
        try:
            # Check for Qwen's native function call format
            if "✿FUNCTION✿" in response_text:
                # Extract function call
                function_match = re.search(r'✿FUNCTION✿:\s*(\{.*?\})', response_text, re.DOTALL)
                if function_match:
                    tool_call_data = json.loads(function_match.group(1))
                    tool_calls.append(tool_call_data)
            # Check for manual TOOL_CALL format
            elif "TOOL_CALL:" in response_text:
                tool_match = re.search(r'TOOL_CALL:\s*(\{.*?\})', response_text, re.DOTALL)
                if tool_match:
                    tool_call_data = json.loads(tool_match.group(1))
                    tool_calls.append(tool_call_data)
            # Check if entire response is JSON tool call
            elif response_text.strip().startswith('{') and '"name"' in response_text:
                tool_call_data = json.loads(response_text.strip())
                if 'name' in tool_call_data and 'arguments' in tool_call_data:
                    tool_calls.append(tool_call_data)
        except (json.JSONDecodeError, AttributeError) as e:
            # No structured tool calls found, will fall back to regex parsing
            pass
        
        return {
            'response': response_text,
            'tool_calls': tool_calls,  # Structured tool calls if available
            'model': ROUTER_MODEL
        }
    
    @modal.method()
    def generate_patch_with_worker(
        self,
        problem: str,
        code_section: str,
        file_path: str
    ) -> Dict:
        """Generate patch using worker (Qwen 7B)."""
        import torch
        
        start_time = time.time()
        
        user_prompt = f"""
PROBLEM:
{problem}

FILE: {file_path}
CODE SECTION:
```python
{code_section}
```

Generate the patch:
"""
        
        messages = [
            {"role": "system", "content": WORKER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.worker_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.worker_tokenizer([text], return_tensors="pt").to(self.worker_model.device)
        
        with torch.no_grad():
            outputs = self.worker_model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.2,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.worker_tokenizer.eos_token_id
            )
        
        generated_text = self.worker_tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        patch = extract_patch_from_response(generated_text)
        latency = time.time() - start_time
        
        return {
            'patch': patch,
            'latency': latency,
            'model': WORKER_MODEL
        }


# ============================================================================
# HELPER FUNCTIONS FOR TOOL EXECUTION
# ============================================================================

def execute_tool_call(
    tool_name: str,
    tool_args: Dict,
    problem: Dict,
    tool_calls: List,
    messages: List,
    response_text: str,
    system
) -> bool:
    """Execute a structured tool call and return True if successful."""
    
    if tool_name == 'fetch_file_metadata':
        file_path = tool_args.get('file_path')
        if file_path:
            result = fetch_file_metadata_impl(
                problem['repo'],
                problem['base_commit'],
                file_path
            )
            
            tool_calls.append({
                'tool': 'fetch_file_metadata',
                'file_path': file_path,
                'result': result
            })
            
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": f"Tool result: {json.dumps(result, indent=2)}"})
            return True
    
    elif tool_name == 'search_codebase':
        file_path = tool_args.get('file_path')
        query = tool_args.get('query')
        if file_path and query:
            result = search_codebase_impl(
                problem['repo'],
                problem['base_commit'],
                file_path,
                query
            )
            
            tool_calls.append({
                'tool': 'search_codebase',
                'file_path': file_path,
                'query': query,
                'result': result
            })
            
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": f"Tool result: {json.dumps(result, indent=2)}"})
            return True
    
    elif tool_name == 'fetch_code_section':
        file_path = tool_args.get('file_path')
        function_name = tool_args.get('function_name')
        start_line = tool_args.get('start_line')
        end_line = tool_args.get('end_line')
        
        if file_path:
            result = fetch_code_section_impl(
                problem['repo'],
                problem['base_commit'],
                file_path,
                start_line=int(start_line) if start_line else None,
                end_line=int(end_line) if end_line else None,
                function=function_name
            )
            
            tool_calls.append({
                'tool': 'fetch_code_section',
                'file_path': file_path,
                'function': function_name,
                'result': result
            })
            
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": f"Code section:\n```python\n{result}\n```"})
            return True
    
    elif tool_name == 'generate_patch_with_qwen':
        # This is handled separately in the main loop
        return False
    
    return False


# ============================================================================
# PHASE 2 WORKFLOW (with open-source models)
# ============================================================================

@app.function(
    image=image,
    timeout=1800,  # 30 minutes
)
def generate_patch_phase2_opensource(problem: Dict) -> Dict:
    """
    Phase 2 with open-source models (no API rate limits!).
    """
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"Processing: {problem['instance_id']}")
    print(f"{'='*80}\n")
    
    # Initialize dual system
    system = DualQwenSystem()
    
    # Track metrics
    router_latency = 0
    worker_latency = 0
    tool_calls = []
    
    # Track token usage
    router_tokens_input = 0
    router_tokens_output = 0
    router_tokens_total = 0
    worker_tokens_input = 0
    worker_tokens_output = 0
    worker_tokens_total = 0
    total_tokens = 0
    routing_decision = "unknown"
    efficiency_score = 0.0
    
    # Define tools
    tools = [
        {
            'name': 'fetch_file_metadata',
            'description': 'Get file metadata: size, functions, classes',
            'parameters': ['file_path']
        },
        {
            'name': 'search_codebase',
            'description': 'Search for functions/classes in file',
            'parameters': ['file_path', 'query']
        },
        {
            'name': 'fetch_code_section',
            'description': 'Fetch specific code section from file',
            'parameters': ['file_path', 'start_line', 'end_line', 'function_name']
        },
        {
            'name': 'generate_patch_with_qwen',
            'description': 'Generate patch using Qwen worker',
            'parameters': ['problem', 'code_section', 'file_path']
        }
    ]
    
    # Start conversation
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""
PROBLEM: {problem['problem_statement']}

REPOSITORY: {problem['repo']}

HINTS: {problem.get('hints_text', 'No hints provided')}

Localize the bug and generate a patch. Be efficient - max 5-7 tool calls!
"""
        }
    ]
    
    print("🧠 Router analyzing problem...")
    
    # Track full conversation for analysis
    conversation_log = []
    
    # Simple tool calling loop (manual parsing)
    final_patch = None
    max_turns = 7  # Limit turns to avoid excessive calls
    
    for turn in range(max_turns):
        print(f"\n  Turn {turn + 1}:")
        
        # Get router response
        router_start = time.time()
        response = system.generate_with_router.remote(messages, tools, max_tokens=512)
        router_latency += time.time() - router_start
        
        response_text = response['response']
        structured_tool_calls = response.get('tool_calls', [])
        
        # Track router tokens
        turn_input_tokens = len(' '.join([m['content'] for m in messages])) // 4
        turn_output_tokens = len(response_text) // 4
        router_tokens_input += turn_input_tokens
        router_tokens_output += turn_output_tokens
        router_tokens_total += turn_input_tokens + turn_output_tokens
        
        # Log router turn
        conversation_log.append({
            'turn': turn + 1,
            'role': 'router',
            'model': ROUTER_MODEL,
            'content': response_text,
            'structured_tool_calls': structured_tool_calls,
            'timestamp': time.time(),
            'tokens_input': turn_input_tokens,
            'tokens_output': turn_output_tokens
        })
        
        # Try structured tool calls first, then fall back to regex parsing
        tool_executed = False
        
        # PRIORITY 1: Use structured tool calls if available
        if structured_tool_calls:
            print(f"  ✅ Using native function calling (structured)")
            for tool_call in structured_tool_calls:
                tool_name = tool_call.get('name') or tool_call.get('function', {}).get('name')
                tool_args = tool_call.get('arguments') or tool_call.get('function', {}).get('arguments', {})
                
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        continue
                
                print(f"  🔧 {tool_name}({json.dumps(tool_args)})")
                
                # Special handling for generate_patch_with_qwen
                if tool_name == 'generate_patch_with_qwen':
                    print(f"  🤖 Router delegating to worker (structured call)...")
                    
                    # Extract the last code section we fetched
                    if tool_calls:
                        last_code_call = [tc for tc in tool_calls if tc['tool'] == 'fetch_code_section']
                        if last_code_call:
                            last_code = last_code_call[-1]['result']
                            
                            worker_result = system.generate_patch_with_worker.remote(
                                problem=problem['problem_statement'],
                                code_section=last_code,
                                file_path=tool_calls[-1].get('file_path', 'unknown')
                            )
                            
                            final_patch = worker_result['patch']
                            worker_latency = worker_result['latency']
                            
                            # Track worker tokens
                            worker_input_tokens = (len(problem['problem_statement']) + len(last_code)) // 4
                            worker_output_tokens = len(final_patch) // 4
                            worker_tokens_input += worker_input_tokens
                            worker_tokens_output += worker_output_tokens
                            worker_tokens_total += worker_input_tokens + worker_output_tokens
                            
                            # Set routing decision
                            routing_decision = "delegated"
                            
                            # Log worker interaction
                            conversation_log.append({
                                'turn': turn + 1,
                                'role': 'worker',
                                'model': WORKER_MODEL,
                                'input': {
                                    'problem': problem['problem_statement'][:200] + '...',
                                    'code_section_length': len(last_code),
                                    'file_path': tool_calls[-1].get('file_path', 'unknown')
                                },
                                'output': final_patch,
                                'timestamp': time.time(),
                                'latency': worker_latency,
                                'tokens_input': worker_input_tokens,
                                'tokens_output': worker_output_tokens
                            })
                            
                            tool_calls.append({
                                'tool': 'generate_patch_with_qwen',
                                'latency': worker_latency
                            })
                            
                            print(f"  ✅ Patch generated!")
                            tool_executed = True
                            break
                else:
                    tool_executed = execute_tool_call(
                        tool_name, tool_args, problem, tool_calls, messages, response_text, system
                    )
                    
                    if tool_executed:
                        break
        
        # If we got a final patch from structured call, break
        if final_patch:
            break
        
        # PRIORITY 2: Fall back to regex parsing if no structured calls
        if not tool_executed:
            print(f"  ⚠️  No structured tool calls, falling back to regex parsing")
            tool_pattern = r'(fetch_file_metadata|search_codebase|fetch_code_section|generate_patch_with_qwen)\s*\('
        else:
            # Structured tool executed successfully, continue to next turn
            continue
        
        # Check if router generated patch directly (starts with --- a/)
        if response_text.strip().startswith('---') or '--- a/' in response_text:
            print(f"  🎯 Router generated patch directly!")
            final_patch = extract_patch_from_response(response_text)
            router_latency += time.time() - router_start
            
            tool_calls.append({
                'tool': 'direct_patch_generation',
                'latency': 0
            })
            
            print(f"  ✅ Patch generated by router!")
            break
        
        elif 'generate_patch_with_qwen' in response_text:
            # Final step - delegate to worker
            print(f"  🤖 Router delegating to worker...")
            
            # Extract the last code section we fetched
            if tool_calls:
                last_code_call = [tc for tc in tool_calls if tc['tool'] == 'fetch_code_section']
                if last_code_call:
                    last_code = last_code_call[-1]['result']
                    
                    worker_result = system.generate_patch_with_worker.remote(
                        problem=problem['problem_statement'],
                        code_section=last_code,
                        file_path=tool_calls[-1].get('file_path', 'unknown')
                    )
                    
                    final_patch = worker_result['patch']
                    worker_latency = worker_result['latency']
                    
                    # Log worker interaction
                    conversation_log.append({
                        'turn': turn + 1,
                        'role': 'worker',
                        'model': WORKER_MODEL,
                        'input': {
                            'problem': problem['problem_statement'][:200] + '...',
                            'code_section_length': len(last_code),
                            'file_path': tool_calls[-1].get('file_path', 'unknown')
                        },
                        'output': final_patch,
                        'timestamp': time.time(),
                        'latency': worker_latency,
                        'tokens_input': (len(problem['problem_statement']) + len(last_code)) // 4,
                        'tokens_output': len(final_patch) // 4
                    })
                    
                    tool_calls.append({
                        'tool': 'generate_patch_with_qwen',
                        'latency': worker_latency
                    })
                    
                    print(f"  ✅ Patch generated!")
                    break
        
        elif re.search(tool_pattern, response_text):
            # Execute tool call
            tool_executed = False
            if 'fetch_file_metadata' in response_text:
                # Extract file path (simple parsing)
                match = re.search(r'fetch_file_metadata\(["\'](.+?)["\']\)', response_text)
                if match:
                    file_path = match.group(1)
                    print(f"  🔧 fetch_file_metadata({file_path})")
                    
                    result = fetch_file_metadata_impl(
                        problem['repo'],
                        problem['base_commit'],
                        file_path
                    )
                    
                    tool_calls.append({
                        'tool': 'fetch_file_metadata',
                        'file_path': file_path,
                        'result': result
                    })
                    
                    messages.append({
                        "role": "assistant",
                        "content": response_text
                    })
                    messages.append({
                        "role": "user",
                        "content": f"Tool result: {json.dumps(result, indent=2)}"
                    })
                    tool_executed = True
            
            elif 'search_codebase' in response_text:
                match = re.search(r'search_codebase\(["\'](.+?)["\']\s*,\s*["\'](.+?)["\']\)', response_text)
                if match:
                    file_path, query = match.groups()
                    print(f"  🔧 search_codebase({file_path}, '{query}')")
                    
                    result = search_codebase_impl(
                        problem['repo'],
                        problem['base_commit'],
                        file_path,
                        query
                    )
                    
                    tool_calls.append({
                        'tool': 'search_codebase',
                        'file_path': file_path,
                        'query': query,
                        'result': result
                    })
                    
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": f"Tool result: {json.dumps(result, indent=2)}"})
                    tool_executed = True
            
            elif 'fetch_code_section' in response_text:
                # Try to extract function name
                match = re.search(r'fetch_code_section\(["\'](.+?)["\']\s*,\s*function_name=["\'](.+?)["\']\)', response_text)
                if match:
                    file_path, function_name = match.groups()
                    print(f"  🔧 fetch_code_section({file_path}, function='{function_name}')")
                    
                    result = fetch_code_section_impl(
                        problem['repo'],
                        problem['base_commit'],
                        file_path,
                        function_name=function_name
                    )
                    
                    tool_calls.append({
                        'tool': 'fetch_code_section',
                        'file_path': file_path,
                        'function_name': function_name,
                        'result': result
                    })
                    
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": f"Tool result:\n{result}"})
                    tool_executed = True
            
            # If no tool was executed despite pattern match, break
            if not tool_executed:
                print(f"  ⚠️  Tool pattern matched but no tool executed, breaking...")
                break
        else:
            # No tool calls, might be final response
            print(f"  💬 Router: {response_text[:200]}...")
            break
    
    total_latency = time.time() - start_time
    
    # Calculate final metrics
    total_tokens = router_tokens_total + worker_tokens_total
    efficiency_score = worker_tokens_total / total_tokens if total_tokens > 0 else 0.0
    
    # Set routing decision if not already set
    if routing_decision == "unknown":
        if worker_latency > 0:
            routing_decision = "delegated"
        else:
            routing_decision = "direct"
    
    print(f"\n{'='*80}")
    print(f"✅ Completed in {total_latency:.2f}s")
    print(f"   Router (Qwen 32B): {router_latency:.2f}s")
    print(f"   Worker (Qwen 7B): {worker_latency:.2f}s")
    print(f"   Tool calls: {len(tool_calls)}")
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Routing: {routing_decision}")
    print(f"   Efficiency: {efficiency_score:.2%}")
    print(f"{'='*80}\n")
    
    # Calculate additional metrics
    avg_tokens_per_turn = router_tokens_total / len(tool_calls) if tool_calls else 0
    tokens_per_second = total_tokens / total_latency if total_latency > 0 else 0
    
    return {
        "instance_id": problem['instance_id'],
        "repo": problem['repo'],
        "base_commit": problem['base_commit'],
        "success": bool(final_patch),
        "completion": final_patch or "",
        "ground_truth_patch": problem.get('patch', ''),
        
        # Localization metrics
        "localization": {
            "method": "hint_based_opensource",
            "hints": problem.get('hints_text', ''),
            "tool_calls": len(tool_calls),
            "tools_used": [tc['tool'] for tc in tool_calls],
            "localization_turns": len([tc for tc in tool_calls if tc['tool'] != 'generate_patch_with_qwen' and tc['tool'] != 'direct_patch_generation'])
        },
        
        # Latency metrics
        "latency": {
            "total": total_latency,
            "router": router_latency,
            "worker": worker_latency,
            "inference_only": router_latency + worker_latency
        },
        
        # Token efficiency metrics
        "token_efficiency": {
            "router_tokens_input": router_tokens_input,
            "router_tokens_output": router_tokens_output,
            "router_tokens_total": router_tokens_total,
            "worker_tokens_input": worker_tokens_input,
            "worker_tokens_output": worker_tokens_output,
            "worker_tokens_total": worker_tokens_total,
            "total_tokens": total_tokens,
            "avg_tokens_per_turn": avg_tokens_per_turn,
            "tokens_per_second": tokens_per_second,
            "routing_decision": routing_decision,
            "efficiency_score": efficiency_score
        },
        
        # Tool usage metrics
        "tool_usage": {
            "total_tool_calls": len(tool_calls),
            "tool_call_details": tool_calls,
            "tool_frequency": {
                tool: sum(1 for tc in tool_calls if tc['tool'] == tool)
                for tool in set(tc['tool'] for tc in tool_calls)
            }
        },
        
        # Model info
        "models": {
            "router": ROUTER_MODEL + " (8-bit)",
            "worker": WORKER_MODEL + " (bfloat16)"
        },
        
        # Cost metrics
        "cost_estimate": {
            "gpu_time_minutes": total_latency / 60,
            "gpu_type": "A100-80GB",
            "estimated_cost_usd": (total_latency / 3600) * 2.5,
            "cost_per_token": ((total_latency / 3600) * 2.5) / total_tokens if total_tokens > 0 else 0
        },
        
        # Strategy info
        "strategy": "phase2_opensource_qwen32b_qwen7b",
        "generation_only": True,  # Flag to indicate this is generation-only
        "verification_required": True,  # Flag to indicate verification needed
        
        # Full conversation for analysis
        "conversation_log": conversation_log,
        "conversation_summary": {
            "total_turns": len(conversation_log),
            "router_turns": sum(1 for log in conversation_log if log['role'] == 'router'),
            "worker_turns": sum(1 for log in conversation_log if log['role'] == 'worker'),
            "total_conversation_tokens": sum(log.get('tokens_input', 0) + log.get('tokens_output', 0) for log in conversation_log)
        }
    }


@app.function(image=image, timeout=600)
def load_swebench_problem(instance_id: Optional[str] = None, index: int = 0):
    """Load a single problem from SWE-bench Lite."""
    from datasets import load_dataset
    
    print("Loading SWE-bench Lite dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    
    if instance_id:
        for problem in dataset:
            if problem['instance_id'] == instance_id:
                return problem
        raise ValueError(f"Instance {instance_id} not found")
    else:
        return dataset[index]


@app.local_entrypoint()
def main(
    instance_id: Optional[str] = None,
    index: int = 0,
    save_local: bool = True
):
    """
    Run Phase 2 with open-source models (no API rate limits!).
    
    Args:
        instance_id: Specific instance ID to test
        index: Dataset index if instance_id not provided
        save_local: Save results locally
    """
    print("\n" + "="*80)
    print("SWE-BENCH PHASE 2: OPEN-SOURCE VERSION")
    print("Router: Qwen 32B (8-bit) + Worker: Qwen 7B (bfloat16)")
    print("Single A100-80GB - No API Rate Limits!")
    print("="*80 + "\n")
    
    # Load problem
    if instance_id:
        print(f"Loading problem: {instance_id}")
    else:
        print(f"Loading problem at index: {index}")
    
    problem = load_swebench_problem.remote(instance_id=instance_id, index=index)
    
    print(f"\nProblem loaded:")
    print(f"  Instance ID: {problem['instance_id']}")
    print(f"  Repository: {problem['repo']}")
    print(f"  Hints: {problem.get('hints_text', 'No hints')}")
    print(f"  Problem: {problem['problem_statement'][:200]}...")
    print()
    
    # Generate patch
    result = generate_patch_phase2_opensource.remote(problem)
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")
    
    if result['success']:
        print(f"✅ Success!")
        print(f"   Total Latency: {result['latency']:.2f}s")
        print(f"   Router (Qwen 32B 8-bit): {result['router_latency']:.2f}s")
        print(f"   Worker (Qwen 7B bf16): {result['worker_latency']:.2f}s")
        print(f"   Tool Calls: {result['localization']['tool_calls']}")
        print(f"   Estimated Cost: ${result['cost_estimate']['estimated_cost']:.4f}")
        print()
        
        print("Generated Patch:")
        print("-" * 80)
        print(result['completion'])
        print("-" * 80)
        print()
        
        print("Ground Truth Patch:")
        print("-" * 80)
        print(result['ground_truth_patch'])
        print("-" * 80)
        print()
        
        # Compare sizes
        gen_size = len(result['completion'])
        gt_size = len(result['ground_truth_patch'])
        ratio = gen_size / gt_size if gt_size > 0 else 0
        
        print(f"Patch Comparison:")
        print(f"  Generated: {gen_size:,} chars")
        print(f"  Ground Truth: {gt_size:,} chars")
        print(f"  Ratio: {ratio:.2f}x")
        
        if ratio < 3:
            print(f"  ✅ Good! Patch size is reasonable")
        else:
            print(f"  ⚠️  Warning: Patch is {ratio:.1f}x larger than ground truth")
    else:
        print(f"❌ Failed: No patch generated")
    
    # Save results
    if save_local:
        # Save main results
        filename = f"results_phase2_opensource_{problem['instance_id']}.json"
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "strategy": "phase2_opensource_qwen32b_qwen7b",
                "result": result
            }, f, indent=2)
        print(f"\n✅ Results saved to: {filename}")
        
        # Save conversation log separately for analysis
        conversation_filename = f"conversation_{problem['instance_id']}.json"
        with open(conversation_filename, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "instance_id": problem['instance_id'],
                "conversation_log": result['conversation_log'],
                "conversation_summary": result['conversation_summary'],
                "routing_decision": result['token_efficiency']['routing_decision'],
                "efficiency_score": result['token_efficiency']['efficiency_score']
            }, f, indent=2)
        print(f"📝 Conversation log saved to: {conversation_filename}")
    
    print("\n" + "="*80)
    print("DONE! No rate limits - run as many as you want!")
    print("="*80 + "\n")

