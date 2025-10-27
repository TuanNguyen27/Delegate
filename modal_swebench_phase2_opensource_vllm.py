"""
SWE-bench Phase 2: Open-Source Version with vLLM
Uses Qwen 32B (8-bit) as router + Qwen 7B as worker on single A100-80GB

This version uses vLLM for MUCH faster inference!
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
app = modal.App("swebench-phase2-opensource-vllm")

# Create volume for storing results
volume = modal.Volume.from_name("swebench-results", create_if_missing=True)

# Docker image with vLLM
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",
        "datasets>=2.14.0",
        "requests>=2.31.0",
    )
)

# Models - 32B router for smart routing, 1.5B worker for fast execution
ROUTER_MODEL = "Qwen/Qwen2.5-32B-Instruct"  # Router (smart localization)
WORKER_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # Worker (fast patch generation)


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
# HELPER FUNCTIONS (same as before)
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
            'functions': functions[:20],
            'classes': classes[:10],
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
        
        return results[:10]
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
# VLLM DUAL QWEN SYSTEM (Router + Worker on same GPU)
# ============================================================================

@app.cls(
    image=image,
    gpu="A100-80GB",
    timeout=1800,  # 30 minutes for model loading + inference
    scaledown_window=600,  # Keep warm for 10 minutes
)
class DualQwenSystemVLLM:
    """
    Runs both Qwen 32B (router) and Qwen 7B (worker) on single A100-80GB using vLLM.
    
    vLLM Benefits:
    - 10-20x faster inference than transformers
    - PagedAttention for efficient KV cache
    - Continuous batching
    - Lower memory overhead
    
    Memory usage:
    - Qwen 32B (quantized): ~35-40 GB
    - Qwen 7B (bfloat16): ~15 GB
    - Total: ~50-55 GB (fits in 80GB!)
    """
    
    @modal.enter()
    def load_models(self):
        """Load both models with vLLM on container startup."""
        from vllm import LLM
        
        print("="*80)
        print("Loading Dual Qwen System with vLLM...")
        print("="*80)
        
        # Load Router (Qwen 32B)
        print(f"\n1. Loading Router: {ROUTER_MODEL}...")
        router_start = time.time()
        
        self.router_llm = LLM(
            model=ROUTER_MODEL,
            dtype="bfloat16",
            max_model_len=4096,  # Enough for localization
            gpu_memory_utilization=0.65,  # Use 65% for 32B router
            tensor_parallel_size=1,
            trust_remote_code=True
        )
        
        router_time = time.time() - router_start
        print(f"   ✅ Router loaded in {router_time:.1f}s")
        print(f"   Memory: ~64 GB (bfloat16)")
        
        # Load Worker (Qwen 1.5B)
        print(f"\n2. Loading Worker: {WORKER_MODEL}...")
        worker_start = time.time()
        
        self.worker_llm = LLM(
            model=WORKER_MODEL,
            dtype="bfloat16",
            max_model_len=4096,  # Enough for patch generation
            gpu_memory_utilization=0.15,  # Use 15% for 1.5B worker
            tensor_parallel_size=1,
            trust_remote_code=True
        )
        
        worker_time = time.time() - worker_start
        print(f"   ✅ Worker loaded in {worker_time:.1f}s")
        print(f"   Memory: ~3 GB (bfloat16)")
        
        print(f"\n{'='*80}")
        print(f"✅ Both models loaded! Total time: {router_time + worker_time:.1f}s")
        print(f"   Estimated memory: ~67 GB / 80 GB")
        print(f"   Router: 32B (smart localization)")
        print(f"   Worker: 1.5B (fast patch generation)")
        print(f"   vLLM acceleration: 10-20x faster inference!")
        print(f"{'='*80}\n")
    
    @modal.method()
    def generate_with_router(
        self,
        messages: List[Dict],
        tools: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> Dict:
        """Generate response with router using vLLM."""
        from vllm import SamplingParams
        
        # Format messages for Qwen
        # Combine system + user messages
        system_msg = next((m['content'] for m in messages if m['role'] == 'system'), '')
        user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
        
        # Add tools description
        if tools:
            tools_desc = "\n\nAvailable tools:\n"
            for tool in tools:
                tools_desc += f"- {tool['name']}: {tool['description']}\n"
            user_msg = user_msg + tools_desc
        
        # Create prompt
        prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        
        # vLLM sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        
        # Generate
        outputs = self.router_llm.generate([prompt], sampling_params)
        response_text = outputs[0].outputs[0].text
        
        return {
            'response': response_text,
            'model': ROUTER_MODEL
        }
    
    @modal.method()
    def generate_patch_with_worker(
        self,
        problem: str,
        code_section: str,
        file_path: str
    ) -> Dict:
        """Generate patch using worker with vLLM."""
        from vllm import SamplingParams
        
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
        
        prompt = f"<|im_start|>system\n{WORKER_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.95,
            max_tokens=2048,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        
        outputs = self.worker_llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        patch = extract_patch_from_response(generated_text)
        latency = time.time() - start_time
        
        return {
            'patch': patch,
            'latency': latency,
            'model': WORKER_MODEL
        }


# ============================================================================
# PHASE 2 WORKFLOW (with vLLM)
# ============================================================================

@app.function(
    image=image,
    timeout=1800,  # 30 minutes
)
def generate_patch_phase2_opensource_vllm(problem: Dict) -> Dict:
    """
    Phase 2 with open-source models + vLLM (FAST!).
    """
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"Processing: {problem['instance_id']}")
    print(f"{'='*80}\n")
    
    # Initialize dual system
    system = DualQwenSystemVLLM()
    
    # Track metrics
    router_latency = 0
    worker_latency = 0
    tool_calls = []
    
    # Track token efficiency
    router_tokens_input = 0
    router_tokens_output = 0
    worker_tokens_input = 0
    worker_tokens_output = 0
    
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
    
    # Simple tool calling loop
    final_patch = None
    max_turns = 7
    
    for turn in range(max_turns):
        print(f"\n  Turn {turn + 1}:")
        
        # Get router response
        router_start = time.time()
        response = system.generate_with_router.remote(messages, tools, max_tokens=512)
        router_latency += time.time() - router_start
        
        response_text = response['response']
        
        # Track tokens (approximate)
        router_tokens_input += len(' '.join([m['content'] for m in messages])) // 4  # ~4 chars per token
        router_tokens_output += len(response_text) // 4
        
        # Parse for tool calls or direct patch generation
        tool_pattern = r'(fetch_file_metadata|search_codebase|fetch_code_section|generate_patch_with_qwen)\s*\('
        
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
                    worker_tokens_input += (len(problem['problem_statement']) + len(last_code)) // 4
                    worker_tokens_output += len(final_patch) // 4
                    
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
                    
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": f"Tool result: {json.dumps(result, indent=2)}"})
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
            print(f"  💬 Router: {response_text[:200]}...")
            break
    
    total_latency = time.time() - start_time
    
    # Calculate token efficiency metrics
    total_tokens = router_tokens_input + router_tokens_output + worker_tokens_input + worker_tokens_output
    router_tokens_total = router_tokens_input + router_tokens_output
    worker_tokens_total = worker_tokens_input + worker_tokens_output
    
    # Routing decision
    routing_decision = "direct" if any(tc['tool'] == 'direct_patch_generation' for tc in tool_calls) else "delegated"
    
    # Efficiency score (lower is better)
    # Penalize: many turns, many tokens, delegation when not needed
    efficiency_score = (
        len(tool_calls) * 10 +  # Each tool call costs
        total_tokens / 100 +  # Token cost
        (50 if routing_decision == "delegated" and worker_tokens_total < 500 else 0)  # Unnecessary delegation penalty
    )
    
    print(f"\n{'='*80}")
    print(f"✅ Completed in {total_latency:.2f}s")
    print(f"   Router (Qwen 32B vLLM): {router_latency:.2f}s")
    print(f"   Worker (Qwen 7B vLLM): {worker_latency:.2f}s")
    print(f"   Tool calls: {len(tool_calls)}")
    print(f"   🚀 vLLM speedup: ~10-20x faster!")
    print(f"\n📊 Token Efficiency:")
    print(f"   Router tokens: {router_tokens_input:,} in + {router_tokens_output:,} out = {router_tokens_total:,} total")
    print(f"   Worker tokens: {worker_tokens_input:,} in + {worker_tokens_output:,} out = {worker_tokens_total:,} total")
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Routing decision: {routing_decision.upper()}")
    print(f"   Efficiency score: {efficiency_score:.1f} (lower is better)")
    print(f"{'='*80}\n")
    
    return {
        "instance_id": problem['instance_id'],
        "repo": problem['repo'],
        "success": bool(final_patch),
        "completion": final_patch or "",
        "ground_truth_patch": problem.get('patch', ''),
        "localization": {
            "method": "hint_based_opensource_vllm",
            "hints": problem.get('hints_text', ''),
            "tool_calls": len(tool_calls),
            "tools_used": [tc['tool'] for tc in tool_calls]
        },
        "latency": total_latency,
        "router_latency": router_latency,
        "worker_latency": worker_latency,
        "tool_call_details": tool_calls,
        "token_efficiency": {
            "router_tokens_input": router_tokens_input,
            "router_tokens_output": router_tokens_output,
            "router_tokens_total": router_tokens_total,
            "worker_tokens_input": worker_tokens_input,
            "worker_tokens_output": worker_tokens_output,
            "worker_tokens_total": worker_tokens_total,
            "total_tokens": total_tokens,
            "routing_decision": routing_decision,
            "efficiency_score": efficiency_score,
            "tokens_per_second": total_tokens / total_latency if total_latency > 0 else 0
        },
        "strategy": "phase2_opensource_qwen32b_qwen7b_vllm",
        "models": {
            "router": ROUTER_MODEL + " (quantized, vLLM)",
            "worker": WORKER_MODEL + " (bfloat16, vLLM)"
        },
        "cost_estimate": {
            "gpu_time_minutes": total_latency / 60,
            "gpu_type": "A100-80GB",
            "estimated_cost": (total_latency / 3600) * 2.5
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
    Run Phase 2 with open-source models + vLLM (FAST!).
    
    Args:
        instance_id: Specific instance ID to test
        index: Dataset index if instance_id not provided
        save_local: Save results locally
    """
    print("\n" + "="*80)
    print("SWE-BENCH PHASE 2: OPEN-SOURCE VERSION with vLLM")
    print("Router: Qwen 32B (quantized) + Worker: Qwen 7B (bfloat16)")
    print("Single A100-80GB - 10-20x Faster with vLLM!")
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
    result = generate_patch_phase2_opensource_vllm.remote(problem)
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")
    
    if result['success']:
        print(f"✅ Success!")
        print(f"   Total Latency: {result['latency']:.2f}s")
        print(f"   Router (Qwen 32B vLLM): {result['router_latency']:.2f}s")
        print(f"   Worker (Qwen 7B vLLM): {result['worker_latency']:.2f}s")
        print(f"   Tool Calls: {result['localization']['tool_calls']}")
        print(f"   Estimated Cost: ${result['cost_estimate']['estimated_cost']:.4f}")
        print(f"   🚀 vLLM acceleration: 10-20x faster!")
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
        filename = f"results_phase2_opensource_vllm_{problem['instance_id']}.json"
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "strategy": "phase2_opensource_qwen32b_qwen7b_vllm",
                "result": result
            }, f, indent=2)
        print(f"\n✅ Results saved to: {filename}")
    
    print("\n" + "="*80)
    print("DONE! vLLM makes it 10-20x faster!")
    print("="*80 + "\n")

