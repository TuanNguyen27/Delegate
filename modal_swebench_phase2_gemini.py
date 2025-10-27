"""
SWE-bench Phase 2: Gemini Router with Full Metrics & Conversation Logging

This version uses:
- Gemini 2.0 Flash-Lite as router (30 RPM free, 4,000 RPM paid - still free!)
- Qwen 2.5-Coder-7B-Instruct as worker
- Full token tracking and efficiency metrics
- Complete conversation logging to Modal volume

Usage:
    modal run modal_swebench_phase2_gemini.py --index 0
    
Rate Limits:
    Free tier: 30 RPM, 1M TPM, 200 RPD
    Tier 1 (paid billing enabled): 4,000 RPM, 4M TPM, unlimited RPD
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
app = modal.App("swebench-phase2-gemini")

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
        "google-generativeai>=0.3.0",
    )
)

# Worker model
WORKER_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


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

Remember: Your goal is efficient localization, then delegation. Don't overthink it!
"""


# ============================================================================
# TOOL DEFINITIONS for Gemini
# ============================================================================

def get_tool_definitions():
    """Return tool definitions in Gemini format."""
    import google.generativeai as genai
    
    return [
        genai.protos.FunctionDeclaration(
            name="fetch_file_metadata",
            description="Get metadata about a file (size, functions, classes). Use this before fetching code.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "file_path": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Path to the file (e.g., 'src/utils.py')"
                    )
                },
                required=["file_path"]
            )
        ),
        genai.protos.FunctionDeclaration(
            name="search_codebase",
            description="Search for function/class definitions in a file. Returns line numbers.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "file_path": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Path to the file"
                    ),
                    "query": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Search query (function/class name or keyword)"
                    )
                },
                required=["file_path", "query"]
            )
        ),
        genai.protos.FunctionDeclaration(
            name="fetch_code_section",
            description="Fetch specific lines or a function/class from a file.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "file_path": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Path to the file"
                    ),
                    "start_line": genai.protos.Schema(
                        type=genai.protos.Type.INTEGER,
                        description="Start line number (optional if function is specified)"
                    ),
                    "end_line": genai.protos.Schema(
                        type=genai.protos.Type.INTEGER,
                        description="End line number (optional if function is specified)"
                    ),
                    "function": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Function/class name to fetch (optional if lines are specified)"
                    )
                },
                required=["file_path"]
            )
        ),
        genai.protos.FunctionDeclaration(
            name="generate_patch_with_qwen",
            description="Delegate patch generation to Qwen SLM. Use this after localizing the bug.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "file_path": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Path to the file to patch"
                    ),
                    "code_section": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="The code section containing the bug"
                    ),
                    "problem_description": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Description of what needs to be fixed"
                    ),
                    "start_line": genai.protos.Schema(
                        type=genai.protos.Type.INTEGER,
                        description="Start line number of the code section"
                    )
                },
                required=["file_path", "code_section", "problem_description"]
            )
        )
    ]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fetch_file_from_github(repo: str, commit: str, file_path: str) -> str:
    """Fetch file content from GitHub."""
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{file_path}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error fetching file: {e}"


def get_file_metadata(content: str) -> Dict:
    """Extract metadata from file content."""
    lines = content.split('\n')
    
    # Find functions and classes
    functions = []
    classes = []
    
    for i, line in enumerate(lines, 1):
        if line.strip().startswith('def '):
            func_name = line.strip().split('(')[0].replace('def ', '')
            functions.append({"name": func_name, "line": i})
        elif line.strip().startswith('class '):
            class_name = line.strip().split('(')[0].split(':')[0].replace('class ', '')
            classes.append({"name": class_name, "line": i})
    
    return {
        "size_bytes": len(content),
        "size_kb": len(content) / 1024,
        "num_lines": len(lines),
        "num_functions": len(functions),
        "num_classes": len(classes),
        "functions": functions[:10],  # First 10
        "classes": classes[:10]
    }


def search_in_file(content: str, query: str) -> List[Dict]:
    """Search for query in file content."""
    lines = content.split('\n')
    results = []
    
    query_lower = query.lower()
    
    for i, line in enumerate(lines, 1):
        if query_lower in line.lower():
            results.append({
                "line": i,
                "content": line.strip()
            })
    
    return results[:20]  # Top 20 matches


def extract_code_section(content: str, start_line: Optional[int] = None, 
                        end_line: Optional[int] = None, 
                        function: Optional[str] = None) -> Tuple[str, int]:
    """Extract a section of code."""
    lines = content.split('\n')
    
    # Convert to int if strings (Gemini sometimes returns strings)
    if start_line is not None and not isinstance(start_line, int):
        try:
            start_line = int(start_line)
        except (ValueError, TypeError):
            start_line = None
    
    if end_line is not None and not isinstance(end_line, int):
        try:
            end_line = int(end_line)
        except (ValueError, TypeError):
            end_line = None
    
    if function:
        # Find function definition
        for i, line in enumerate(lines):
            if f'def {function}(' in line or f'class {function}' in line:
                start_line = i + 1
                # Find end of function (next def/class or end of file)
                end_line = len(lines)
                indent = len(line) - len(line.lstrip())
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() and not lines[j].startswith(' ' * (indent + 1)):
                        if lines[j].strip().startswith('def ') or lines[j].strip().startswith('class '):
                            end_line = j
                            break
                break
    
    if start_line is None:
        start_line = 1
    if end_line is None:
        end_line = len(lines)
    
    # Extract section
    section_lines = lines[start_line-1:end_line]
    return '\n'.join(section_lines), start_line


# ============================================================================
# WORKER MODEL (Qwen)
# ============================================================================

@app.cls(
    image=image,
    gpu="A100-40GB",
    timeout=1800,
    scaledown_window=600,
)
class QwenWorker:
    """Qwen 7B worker for patch generation."""
    
    @modal.enter()
    def load_model(self):
        """Load Qwen model."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("Loading Qwen 7B worker...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            WORKER_MODEL,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            WORKER_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ Qwen 7B loaded")
    
    @modal.method()
    def extract_function_batch(
        self,
        requests: List[Tuple[str, str]]
    ) -> List[str]:
        """Extract multiple functions in batch using SLM.
        
        Args:
            requests: List of (file_content, function_name) tuples
            
        Returns:
            List of extracted function code strings
        """
        import torch
        
        if not requests:
            return []
        
        # Build prompts for all requests
        prompts = []
        for file_content, function_name in requests:
            prompt = f"""Extract ONLY the function '{function_name}' from the code below.

CRITICAL RULES:
1. Return ONLY the function definition and its body
2. Do NOT include other functions (even if they're helpers)
3. Do NOT include module-level variables, config, or imports
4. Do NOT include comments outside the function
5. Include the function's docstring if it has one
6. Stop at the end of the function body (before next function/class/variable)

EXAMPLES:

Example 1: Extract single function
═══════════════════════════════════════════════════════════════
CODE:
```python
def foo():
    return 1

def bar():
    return 2

config = {{'key': 'value'}}
```

EXTRACT 'foo':
```python
def foo():
    return 1
```
═══════════════════════════════════════════════════════════════

Example 2: Extract function with docstring
═══════════════════════════════════════════════════════════════
CODE:
```python
def process(data):
    \"\"\"Process the data.\"\"\"
    return data.strip()

def helper():
    pass

SETTINGS = {{}}
```

EXTRACT 'process':
```python
def process(data):
    \"\"\"Process the data.\"\"\"
    return data.strip()
```
═══════════════════════════════════════════════════════════════

Example 3: Stop before module-level variable
═══════════════════════════════════════════════════════════════
CODE:
```python
def compute(x):
    if x > 0:
        return x * 2
    return 0

# Configuration
_config = {{'a': 1}}
```

EXTRACT 'compute':
```python
def compute(x):
    if x > 0:
        return x * 2
    return 0
```
═══════════════════════════════════════════════════════════════

NOW EXTRACT THIS FUNCTION:

CODE:
```python
{file_content}
```

EXTRACT '{function_name}':
```python
"""
            prompts.append(prompt)
        
        # Tokenize all prompts
        messages_batch = [[{"role": "user", "content": p}] for p in prompts]
        
        texts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            for messages in messages_batch
        ]
        
        # Batch tokenization
        model_inputs = self.tokenizer(
            texts, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.model.device)
        
        # Batch generation
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode all outputs
        generated_ids = generated_ids[:, model_inputs.input_ids.shape[-1]:]
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Extract code from each response
        results = []
        for response in responses:
            extracted = response.strip()
            
            # Remove markdown code blocks if present
            if '```python' in extracted:
                extracted = extracted.split('```python')[1].split('```')[0].strip()
            elif '```' in extracted:
                extracted = extracted.split('```')[1].split('```')[0].strip()
            
            results.append(extracted)
        
        return results
    
    @modal.method()
    def extract_function(
        self,
        file_content: str,
        function_name: str
    ) -> str:
        """Extract a single function from file content using SLM (calls batch method internally)."""
        # Note: extract_function_batch is also a @modal.method, so we call it directly
        # within the same class (not via .remote())
        import torch
        
        # Inline the extraction logic to avoid method-calling-method issues
        prompt = f"""Extract ONLY the function '{function_name}' from the code below.

CRITICAL RULES:
1. Return ONLY the function definition and its body
2. Do NOT include other functions (even if they're helpers)
3. Do NOT include module-level variables, config, or imports
4. Do NOT include comments outside the function
5. Include the function's docstring if it has one
6. Stop at the end of the function body (before next function/class/variable)

NOW EXTRACT THIS FUNCTION:

CODE:
```python
{file_content}
```

EXTRACT '{function_name}':
```python
"""
        
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_ids = generated_ids[:, model_inputs.input_ids.shape[-1]:]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract code from response
        extracted = response.strip()
        
        # Remove markdown code blocks if present
        if '```python' in extracted:
            extracted = extracted.split('```python')[1].split('```')[0].strip()
        elif '```' in extracted:
            extracted = extracted.split('```')[1].split('```')[0].strip()
        
        return extracted
    
    @modal.method()
    def _extract_function_single(
        self,
        file_content: str,
        function_name: str
    ) -> str:
        """Extract a single function from file content using SLM (original non-batch version)."""
        import torch
        
        prompt = f"""Extract ONLY the function '{function_name}' from the code below.

CRITICAL RULES:
1. Return ONLY the function definition and its body
2. Do NOT include other functions (even if they're helpers)
3. Do NOT include module-level variables, config, or imports
4. Do NOT include comments outside the function
5. Include the function's docstring if it has one
6. Stop at the end of the function body (before next function/class/variable)

EXAMPLES:

Example 1: Extract single function
═══════════════════════════════════════════════════════════════
CODE:
```python
def foo():
    return 1

def bar():
    return 2

config = {{'key': 'value'}}
```

EXTRACT 'foo':
```python
def foo():
    return 1
```
═══════════════════════════════════════════════════════════════

Example 2: Extract function with docstring
═══════════════════════════════════════════════════════════════
CODE:
```python
def process(data):
    \"\"\"Process the data.\"\"\"
    return data.strip()

def helper():
    pass

SETTINGS = {{}}
```

EXTRACT 'process':
```python
def process(data):
    \"\"\"Process the data.\"\"\"
    return data.strip()
```
═══════════════════════════════════════════════════════════════

Example 3: Stop before module-level variable
═══════════════════════════════════════════════════════════════
CODE:
```python
def compute(x):
    if x > 0:
        return x * 2
    return 0

# Configuration
_config = {{'a': 1}}
```

EXTRACT 'compute':
```python
def compute(x):
    if x > 0:
        return x * 2
    return 0
```
═══════════════════════════════════════════════════════════════

NOW EXTRACT THIS FUNCTION:

CODE:
```python
{file_content}
```

EXTRACT '{function_name}':
```python
"""
        
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_ids = generated_ids[:, model_inputs.input_ids.shape[-1]:]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract code from response
        extracted = response.strip()
        
        # Remove markdown code blocks if present
        if '```python' in extracted:
            extracted = extracted.split('```python')[1].split('```')[0].strip()
        elif '```' in extracted:
            extracted = extracted.split('```')[1].split('```')[0].strip()
        
        return extracted
    
    @modal.method()
    def generate_patch_batch(
        self,
        requests: List[Tuple[str, str, str, int]]
    ) -> List[str]:
        """Generate patches for multiple code sections in batch.
        
        Args:
            requests: List of (file_path, code_section, problem_description, start_line) tuples
            
        Returns:
            List of patch strings
        """
        import torch
        import difflib
        
        if not requests:
            return []
        
        # Build prompts for all requests
        prompts = []
        for file_path, code_section, problem_description, start_line in requests:
            prompt = f"""You are an expert software engineer. Fix the bug by generating the COMPLETE FIXED VERSION of the code.

EXAMPLES:

Example 1: Adding a missing check
═══════════════════════════════════════════════════════════════
PROBLEM: Function crashes when input is None

BUGGY CODE:
```python
def process_data(data):
    \"\"\"Process input data.\"\"\"
    return data.strip()
```

FIXED CODE:
```python
def process_data(data):
    \"\"\"Process input data.\"\"\"
    if data is None:
        return ""
    return data.strip()
```
═══════════════════════════════════════════════════════════════

Example 2: Fixing conditional logic
═══════════════════════════════════════════════════════════════
PROBLEM: Function doesn't handle nested models correctly

BUGGY CODE:
```python
def compute_matrix(model):
    \"\"\"Compute matrix for model.\"\"\"
    if model.is_simple:
        return simple_matrix(model)
    return complex_matrix(model)
```

FIXED CODE:
```python
def compute_matrix(model):
    \"\"\"Compute matrix for model.\"\"\"
    if model.is_simple:
        return simple_matrix(model)
    if model.is_nested:
        return nested_matrix(model)
    return complex_matrix(model)
```
═══════════════════════════════════════════════════════════════

Example 3: Replacing incorrect operator
═══════════════════════════════════════════════════════════════
PROBLEM: Wrong comparison operator causes incorrect results

BUGGY CODE:
```python
def find_item(items, target):
    \"\"\"Find target in items.\"\"\"
    for i in range(len(items)):
        if items[i] > target:
            return i
    return -1
```

FIXED CODE:
```python
def find_item(items, target):
    \"\"\"Find target in items.\"\"\"
    for i in range(len(items)):
        if items[i] == target:
            return i
    return -1
```
═══════════════════════════════════════════════════════════════

CRITICAL RULES:
1. Output the COMPLETE fixed function/code
2. Keep ALL docstrings, comments, and formatting EXACTLY as-is
3. Only change the lines that fix the bug
4. Preserve indentation and style
5. Do NOT add explanations - just output the fixed code
6. Start with ```python and end with ```

NOW FIX THIS CODE:

PROBLEM:
{problem_description}

BUGGY CODE:
```python
{code_section}
```

FIXED CODE:
"""
            prompts.append(prompt)
        
        # Tokenize all prompts
        messages_batch = [[{"role": "user", "content": p}] for p in prompts]
        
        texts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            for messages in messages_batch
        ]
        
        # Batch tokenization
        model_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.model.device)
        
        # Batch generation
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode all outputs
        generated_ids = generated_ids[:, model_inputs.input_ids.shape[-1]:]
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Generate patches from responses
        patches = []
        for i, response in enumerate(responses):
            file_path, code_section, problem_description, start_line = requests[i]
            
            # Extract fixed code from response
            fixed_code = response.strip()
            if '```python' in fixed_code:
                fixed_code = fixed_code.split('```python')[1].split('```')[0].strip()
            elif '```' in fixed_code:
                fixed_code = fixed_code.split('```')[1].split('```')[0].strip()
            
            # Generate diff
            original_lines = code_section.splitlines(keepends=True)
            fixed_lines = fixed_code.splitlines(keepends=True)
            diff = difflib.unified_diff(
                original_lines, fixed_lines,
                fromfile=f'a/{file_path}',
                tofile=f'b/{file_path}',
                lineterm=''
            )
            patch = '\n'.join(diff)
            
            if not patch:
                patch = "No changes detected - model may not have fixed the code"
            
            patches.append(patch)
        
        return patches
    
    @modal.method()
    def generate_patch(
        self,
        file_path: str,
        code_section: str,
        problem_description: str,
        start_line: int = 1
    ) -> str:
        """Generate a fixed version of the code, then create a patch using difflib."""
        import torch
        import difflib
        
        # Inline the logic to avoid Modal method-calling-method issues
        prompt = f"""You are an expert software engineer. Fix the bug by generating the COMPLETE FIXED VERSION of the code.

EXAMPLES:

Example 1: Adding a missing check
═══════════════════════════════════════════════════════════════
PROBLEM: Function crashes when input is None

BUGGY CODE:
```python
def process_data(data):
    \"\"\"Process input data.\"\"\"
    return data.strip()
```

FIXED CODE:
```python
def process_data(data):
    \"\"\"Process input data.\"\"\"
    if data is None:
        return ""
    return data.strip()
```
═══════════════════════════════════════════════════════════════

Example 2: Fixing conditional logic
═══════════════════════════════════════════════════════════════
PROBLEM: Function doesn't handle nested models correctly

BUGGY CODE:
```python
def compute_matrix(model):
    \"\"\"Compute matrix for model.\"\"\"
    if model.is_simple:
        return simple_matrix(model)
    return complex_matrix(model)
```

FIXED CODE:
```python
def compute_matrix(model):
    \"\"\"Compute matrix for model.\"\"\"
    if model.is_simple:
        return simple_matrix(model)
    if model.is_nested:
        return nested_matrix(model)
    return complex_matrix(model)
```
═══════════════════════════════════════════════════════════════

Example 3: Replacing incorrect operator
═══════════════════════════════════════════════════════════════
PROBLEM: Wrong comparison operator causes incorrect results

BUGGY CODE:
```python
def find_item(items, target):
    \"\"\"Find target in items.\"\"\"
    for i in range(len(items)):
        if items[i] > target:
            return i
    return -1
```

FIXED CODE:
```python
def find_item(items, target):
    \"\"\"Find target in items.\"\"\"
    for i in range(len(items)):
        if items[i] == target:
            return i
    return -1
```
═══════════════════════════════════════════════════════════════

CRITICAL RULES:
1. Output the COMPLETE fixed function/code
2. Keep ALL docstrings, comments, and formatting EXACTLY as-is
3. Only change the lines that fix the bug
4. Preserve indentation and style
5. Do NOT add explanations - just output the fixed code
6. Start with ```python and end with ```

NOW FIX THIS CODE:

PROBLEM:
{problem_description}

BUGGY CODE:
```python
{code_section}
```

FIXED CODE:
"""
        
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_ids = generated_ids[:, model_inputs.input_ids.shape[-1]:]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract fixed code from response
        fixed_code = response.strip()
        if '```python' in fixed_code:
            fixed_code = fixed_code.split('```python')[1].split('```')[0].strip()
        elif '```' in fixed_code:
            fixed_code = fixed_code.split('```')[1].split('```')[0].strip()
        
        # Generate diff
        original_lines = code_section.splitlines(keepends=True)
        fixed_lines = fixed_code.splitlines(keepends=True)
        diff = difflib.unified_diff(
            original_lines, fixed_lines,
            fromfile=f'a/{file_path}',
            tofile=f'b/{file_path}',
            lineterm=''
        )
        patch = '\n'.join(diff)
        
        if not patch:
            patch = "No changes detected - model may not have fixed the code"
        
        return patch
    
    @modal.method()
    def _generate_patch_single(
        self,
        file_path: str,
        code_section: str,
        problem_description: str,
        start_line: int = 1
    ) -> str:
        """Generate a fixed version of the code, then create a patch using difflib (original non-batch version)."""
        import torch
        import difflib
        
        prompt = f"""You are an expert software engineer. Fix the bug by generating the COMPLETE FIXED VERSION of the code.

EXAMPLES:

Example 1: Adding a missing check
═══════════════════════════════════════════════════════════════
PROBLEM: Function crashes when input is None

BUGGY CODE:
```python
def process_data(data):
    \"\"\"Process input data.\"\"\"
    return data.strip()
```

FIXED CODE:
```python
def process_data(data):
    \"\"\"Process input data.\"\"\"
    if data is None:
        return ""
    return data.strip()
```
═══════════════════════════════════════════════════════════════

Example 2: Fixing conditional logic
═══════════════════════════════════════════════════════════════
PROBLEM: Function doesn't handle nested models correctly

BUGGY CODE:
```python
def compute_matrix(model):
    \"\"\"Compute matrix for model.\"\"\"
    if model.is_simple:
        return simple_matrix(model)
    return complex_matrix(model)
```

FIXED CODE:
```python
def compute_matrix(model):
    \"\"\"Compute matrix for model.\"\"\"
    if model.is_simple:
        return simple_matrix(model)
    if model.is_nested:
        return nested_matrix(model)
    return complex_matrix(model)
```
═══════════════════════════════════════════════════════════════

Example 3: Replacing incorrect operator
═══════════════════════════════════════════════════════════════
PROBLEM: Wrong comparison operator causes incorrect results

BUGGY CODE:
```python
def find_item(items, target):
    \"\"\"Find target in items.\"\"\"
    for i in range(len(items)):
        if items[i] > target:
            return i
    return -1
```

FIXED CODE:
```python
def find_item(items, target):
    \"\"\"Find target in items.\"\"\"
    for i in range(len(items)):
        if items[i] == target:
            return i
    return -1
```
═══════════════════════════════════════════════════════════════

CRITICAL RULES:
1. Output the COMPLETE fixed function/code
2. Keep ALL docstrings, comments, and formatting EXACTLY as-is
3. Only change the lines that fix the bug
4. Preserve indentation and style
5. Do NOT add explanations - just output the fixed code
6. Start with ```python and end with ```

NOW FIX THIS CODE:

PROBLEM:
{problem_description}

BUGGY CODE:
```python
{code_section}
```

FIXED CODE:
"""
        
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_ids = generated_ids[:, model_inputs.input_ids.shape[-1]:]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract fixed code from response
        fixed_code = response.strip()
        
        # Remove markdown code blocks if present
        if '```python' in fixed_code:
            fixed_code = fixed_code.split('```python')[1].split('```')[0].strip()
        elif '```' in fixed_code:
            fixed_code = fixed_code.split('```')[1].split('```')[0].strip()
        
        # Create unified diff using difflib
        original_lines = code_section.splitlines(keepends=True)
        fixed_lines = fixed_code.splitlines(keepends=True)
        
        # Generate unified diff
        diff = difflib.unified_diff(
            original_lines,
            fixed_lines,
            fromfile=f'a/{file_path}',
            tofile=f'b/{file_path}',
            lineterm=''
        )
        
        # Convert to string
        patch = '\n'.join(diff)
        
        # If no diff (no changes), return original response
        if not patch or patch.strip() == '':
            return f"No changes detected. Model output:\n{response}"
        
        return patch


# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("google-api-key")],
    volumes={"/results": volume},
    timeout=1800,
)
def generate_patch_phase2_gemini(problem: Dict) -> Dict:
    """Generate patch using Gemini router + Qwen worker with full metrics."""
    import google.generativeai as genai
    
    # Configure Gemini
    api_key = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    
    # Initialize Gemini with tools
    # Using gemini-2.0-flash-lite for better rate limits:
    # Free tier: 30 RPM, Tier 1: 4,000 RPM (still free!)
    tools = get_tool_definitions()
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-lite",
        tools=tools,
        system_instruction=ROUTER_SYSTEM_PROMPT
    )
    
    # Initialize worker
    worker = QwenWorker()
    
    # Extract problem info
    instance_id = problem['instance_id']
    problem_statement = problem['problem_statement']
    hints_text = problem.get('hints_text', '')
    repo = problem['repo']
    base_commit = problem['base_commit']
    
    print(f"\n{'='*80}")
    print(f"Processing: {instance_id}")
    print(f"Repository: {repo}")
    print(f"{'='*80}\n")
    
    # Build initial prompt
    initial_prompt = f"""Repository: {repo}
Base Commit: {base_commit}

PROBLEM STATEMENT:
{problem_statement}
"""
    
    if hints_text:
        initial_prompt += f"""
HINTS (files to check):
{hints_text}
"""
    
    initial_prompt += """
Your task: Localize the bug and delegate patch generation to generate_patch_with_qwen.
"""
    
    # Track metrics
    start_time = time.time()
    router_tokens_input = 0
    router_tokens_output = 0
    worker_tokens_input = 0
    worker_tokens_output = 0
    tool_calls_count = 0
    worker_calls = 0
    
    # Track conversation
    conversation_log = []
    conversation_log.append({
        "role": "user",
        "model": "human",
        "content": initial_prompt,
        "tokens": len(initial_prompt) // 4,
        "timestamp": datetime.now().isoformat()
    })
    
    # File cache
    file_cache = {}
    
    # Start chat
    print("🧠 Router analyzing problem...")
    chat = model.start_chat()
    
    # Track tokens for initial prompt
    router_tokens_input += len(initial_prompt) // 4
    
    response = chat.send_message(initial_prompt)
    
    # Safely extract text (may be empty if only function calls)
    try:
        response_text = response.text if response.text else ""
    except ValueError:
        # Response only contains function calls, no text
        response_text = ""
    
    router_tokens_output += len(response_text) // 4
    
    conversation_log.append({
        "role": "assistant",
        "model": "gemini-2.0-flash-exp",
        "content": response_text,
        "tokens_in": len(initial_prompt) // 4,
        "tokens_out": len(response_text) // 4,
        "timestamp": datetime.now().isoformat()
    })
    
    # Tool calling loop
    final_patch = None
    max_turns = 7
    
    for turn in range(max_turns):
        print(f"\n--- Turn {turn + 1}/{max_turns} ---")
        
        # Check if router generated patch directly (shouldn't happen, but handle it)
        if response_text.strip().startswith('---') or '--- a/' in response_text:
            print("⚠️  Router generated patch directly (should use tool!)")
            final_patch = response_text
            conversation_log.append({
                "role": "note",
                "content": "Router generated patch directly instead of using tool",
                "timestamp": datetime.now().isoformat()
            })
            break
        
        # Check for function calls
        if not response.candidates or not response.candidates[0].content.parts:
            print("No more function calls")
            break
        
        parts = response.candidates[0].content.parts
        function_calls = [p for p in parts if hasattr(p, 'function_call') and p.function_call]
        
        if not function_calls:
            print("No function calls found")
            break
        
        # Track if we executed any tool
        tool_executed = False
        function_responses = []
        
        for fc in function_calls:
            if not fc.function_call or not fc.function_call.name:
                print("⚠️  Empty function call, skipping")
                continue
            
            tool_executed = True
            tool_calls_count += 1
            
            func_name = fc.function_call.name
            args = dict(fc.function_call.args) if fc.function_call.args else {}
            
            print(f"🔧 Tool: {func_name}")
            print(f"   Args: {args}")
            
            tool_start = time.time()
            tool_response = {}
            
            if func_name == "fetch_file_metadata":
                file_path = args.get("file_path", "")
                
                # Fetch file if not cached
                if file_path not in file_cache:
                    content = fetch_file_from_github(repo, base_commit, file_path)
                    file_cache[file_path] = content
                else:
                    content = file_cache[file_path]
                
                if content.startswith("Error"):
                    tool_response = {"error": content}
                else:
                    metadata = get_file_metadata(content)
                    tool_response = metadata
            
            elif func_name == "search_codebase":
                file_path = args.get("file_path", "")
                query = args.get("query", "")
                
                # Fetch file if not cached
                if file_path not in file_cache:
                    content = fetch_file_from_github(repo, base_commit, file_path)
                    file_cache[file_path] = content
                else:
                    content = file_cache[file_path]
                
                if content.startswith("Error"):
                    tool_response = {"error": content}
                else:
                    results = search_in_file(content, query)
                    tool_response = {"results": results, "num_matches": len(results)}
            
            elif func_name == "fetch_code_section":
                file_path = args.get("file_path", "")
                start_line = args.get("start_line")
                end_line = args.get("end_line")
                function = args.get("function")
                
                # Fetch file if not cached
                if file_path not in file_cache:
                    content = fetch_file_from_github(repo, base_commit, file_path)
                    file_cache[file_path] = content
                else:
                    content = file_cache[file_path]
                
                if content.startswith("Error"):
                    tool_response = {"error": content}
                else:
                    # Use SLM for function extraction (cleaner, more reliable)
                    if function:
                        print(f"   🤖 Using SLM to extract function '{function}'...")
                        extraction_start = time.time()
                        code_section = worker.extract_function.remote(content, function)
                        extraction_time = time.time() - extraction_start
                        print(f"   ✅ Extracted in {extraction_time:.1f}s")
                        
                        # Track extraction tokens
                        extraction_tokens = (len(content) + len(function) + 500) // 4  # Prompt overhead
                        worker_tokens_input += extraction_tokens
                        worker_tokens_output += len(code_section) // 4
                        
                        actual_start = 1  # SLM doesn't provide line numbers
                    else:
                        # Fall back to Python extraction for line-based requests
                        code_section, actual_start = extract_code_section(
                            content, start_line, end_line, function
                        )
                    
                    tool_response = {
                        "code": code_section,
                        "start_line": actual_start,
                        "num_lines": len(code_section.split('\n')),
                        "extraction_method": "slm" if function else "python"
                    }
            
            elif func_name == "generate_patch_with_qwen":
                file_path = args.get("file_path", "")
                code_section = args.get("code_section", "")
                problem_desc = args.get("problem_description", "")
                start_line = args.get("start_line", 1)
                
                print(f"🤖 Delegating to Qwen worker...")
                worker_calls += 1
                
                # Track worker tokens (approximate)
                worker_input = f"{file_path}\n{problem_desc}\n{code_section}"
                worker_tokens_input += len(worker_input) // 4
                
                # Call worker
                worker_start = time.time()
                patch = worker.generate_patch.remote(
                    file_path, code_section, problem_desc, start_line
                )
                worker_latency = time.time() - worker_start
                
                worker_tokens_output += len(patch) // 4
                
                print(f"   ✅ Worker completed in {worker_latency:.1f}s")
                print(f"   Generated {len(patch)} chars")
                
                final_patch = patch
                tool_response = {
                    "status": "success",
                    "patch_generated": True,
                    "patch_length": len(patch),
                    "latency_seconds": worker_latency
                }
                
                print(f"   ✅ Patch generated successfully!")
                
                # Log worker interaction
                conversation_log.append({
                    "role": "worker",
                    "model": "Qwen2.5-Coder-7B",
                    "input": worker_input[:500] + "..." if len(worker_input) > 500 else worker_input,
                    "output": patch[:500] + "..." if len(patch) > 500 else patch,
                    "tokens_in": len(worker_input) // 4,
                    "tokens_out": len(patch) // 4,
                    "latency": worker_latency,
                    "timestamp": datetime.now().isoformat()
                })
            
            tool_latency = time.time() - tool_start
            
            # Log tool call
            conversation_log.append({
                "role": "tool",
                "tool_name": func_name,
                "args": args,
                "response": str(tool_response)[:200] + "..." if len(str(tool_response)) > 200 else str(tool_response),
                "latency": tool_latency,
                "timestamp": datetime.now().isoformat()
            })
            
            # Prepare response for Gemini
            import google.generativeai as genai
            function_responses.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=func_name,
                        response=tool_response
                    )
                )
            )
        
        # If we got a patch, we're done
        if final_patch:
            break
        
        # If no tool was executed, break to avoid infinite loop
        if not tool_executed:
            print("⚠️  No tools executed, breaking")
            break
        
        # Send function responses back to Gemini
        if function_responses:
            # Track tokens for function responses
            response_str = str(function_responses)
            router_tokens_input += len(response_str) // 4
            
            try:
                response = chat.send_message(function_responses)
                
                # Safely extract text (may be empty if only function calls)
                try:
                    response_text = response.text if response.text else ""
                except ValueError:
                    # Response only contains function calls, no text
                    response_text = ""
                
                router_tokens_output += len(response_text) // 4
                
                conversation_log.append({
                    "role": "assistant",
                    "model": "gemini-2.0-flash-exp",
                    "content": response_text,
                    "tokens_in": len(response_str) // 4,
                    "tokens_out": len(response_text) // 4,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                print(f"⚠️  Gemini error: {e}")
                print(f"   Stopping conversation loop")
                conversation_log.append({
                    "role": "error",
                    "content": f"Gemini API error: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
                break
    
    # Calculate metrics
    total_time = time.time() - start_time
    total_tokens = router_tokens_input + router_tokens_output + worker_tokens_input + worker_tokens_output
    router_tokens_total = router_tokens_input + router_tokens_output
    worker_tokens_total = worker_tokens_input + worker_tokens_output
    
    routing_decision = "delegated" if worker_calls > 0 else "direct"
    efficiency_score = worker_tokens_total / total_tokens if total_tokens > 0 else 0
    avg_tokens_per_turn = total_tokens / max(1, tool_calls_count)
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    
    # Print metrics
    print(f"\n{'='*80}")
    print(f"✅ COMPLETED: {instance_id}")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Total tokens: {total_tokens:,}")
    print(f"  Router: {router_tokens_total:,} ({router_tokens_input:,} in, {router_tokens_output:,} out)")
    print(f"  Worker: {worker_tokens_total:,} ({worker_tokens_input:,} in, {worker_tokens_output:,} out)")
    print(f"Tool calls: {tool_calls_count}")
    print(f"Worker calls: {worker_calls}")
    print(f"Routing decision: {routing_decision}")
    print(f"Efficiency score: {efficiency_score:.2%} (worker tokens / total tokens)")
    print(f"Avg tokens/turn: {avg_tokens_per_turn:.0f}")
    print(f"Tokens/second: {tokens_per_second:.1f}")
    print(f"{'='*80}\n")
    
    # Prepare result
    result = {
        "instance_id": instance_id,
        "model_name_or_path": "gemini-2.0-flash-exp + Qwen2.5-Coder-7B",
        "prediction": final_patch or "No patch generated",
        "metadata": {
            "total_time_seconds": total_time,
            "total_tokens": total_tokens,
            "router_tokens": {
                "total": router_tokens_total,
                "input": router_tokens_input,
                "output": router_tokens_output
            },
            "worker_tokens": {
                "total": worker_tokens_total,
                "input": worker_tokens_input,
                "output": worker_tokens_output
            },
            "tool_calls": tool_calls_count,
            "worker_calls": worker_calls,
            "routing_decision": routing_decision,
            "efficiency_score": efficiency_score,
            "avg_tokens_per_turn": avg_tokens_per_turn,
            "tokens_per_second": tokens_per_second,
            "timestamp": datetime.now().isoformat()
        },
        "conversation_log": conversation_log,
        "conversation_summary": {
            "total_turns": len([c for c in conversation_log if c["role"] == "assistant"]),
            "tool_calls": tool_calls_count,
            "worker_calls": worker_calls,
            "total_messages": len(conversation_log)
        }
    }
    
    return result


# ============================================================================
# LOAD PROBLEM
# ============================================================================

@app.function(image=image)
def load_swebench_problem(index: int) -> Dict:
    """Load a single problem from SWE-bench Lite."""
    from datasets import load_dataset
    
    print("Loading SWE-bench Lite dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    
    if index >= len(dataset):
        raise ValueError(f"Index {index} out of range (dataset has {len(dataset)} problems)")
    
    problem = dataset[index]
    print(f"Loaded problem: {problem['instance_id']}")
    
    return dict(problem)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

@app.local_entrypoint()
def main(start_index: int = 0, num_problems: int = 1, parallel: bool = True):
    """
    Run Phase 2 Gemini router on SWE-bench problems.
    
    Args:
        start_index: Starting index of problems to process (default: 0)
        num_problems: Number of problems to process (default: 1)
        parallel: Process problems in parallel (default: True)
    """
    print(f"\n{'='*80}")
    print(f"SWE-BENCH PHASE 2: GEMINI ROUTER with FULL METRICS")
    print(f"Router: Gemini 2.0 Flash + Worker: Qwen 7B")
    print(f"{'='*80}\n")
    
    mode = "PARALLEL" if parallel else "SEQUENTIAL"
    print(f"Processing {num_problems} problem(s) starting at index {start_index} ({mode})")
    
    if parallel:
        # Load all problems first
        print(f"\n📥 Loading {num_problems} problems...")
        problem_calls = [
            load_swebench_problem.remote(start_index + i) 
            for i in range(num_problems)
        ]
        problems = [call for call in problem_calls]
        print(f"✅ Loaded {len(problems)} problems")
        
        # Process all in parallel
        print(f"\n🚀 Starting {num_problems} parallel jobs...")
        result_calls = [
            generate_patch_phase2_gemini.remote(problem)
            for problem in problems
        ]
        
        # Wait for all to complete
        print(f"⏳ Waiting for all jobs to complete...")
        all_results = []
        for i, result_call in enumerate(result_calls):
            try:
                result = result_call
                all_results.append(result)
                print(f"✅ Problem {i+1}/{num_problems} completed: {result['instance_id']}")
                
                # Save individual conversation
                conversation_file = f"conversation_{result['instance_id']}.json"
                with open(conversation_file, "w") as f:
                    json.dump(result["conversation_log"], f, indent=2)
                    
            except Exception as e:
                print(f"❌ Problem {i+1}/{num_problems} (index {start_index+i}) failed: {e}")
                all_results.append({
                    "instance_id": f"error_index_{start_index+i}",
                    "error": str(e),
                    "index": start_index + i
                })
    else:
        # Sequential processing (original behavior)
        all_results = []
        
        for i in range(num_problems):
            index = start_index + i
            print(f"\n{'='*80}")
            print(f"PROBLEM {i+1}/{num_problems} (Index: {index})")
            print(f"{'='*80}\n")
            
            try:
                # Load problem
                problem = load_swebench_problem.remote(index)
                
                # Generate patch
                result = generate_patch_phase2_gemini.remote(problem)
                
                all_results.append(result)
                
                # Save individual conversation
                conversation_file = f"conversation_{result['instance_id']}.json"
                with open(conversation_file, "w") as f:
                    json.dump(result["conversation_log"], f, indent=2)
                
                print(f"\n✅ Problem {i+1} completed: {result['instance_id']}")
                
            except Exception as e:
                print(f"\n❌ Problem {i+1} (index {index}) failed: {e}")
                all_results.append({
                    "instance_id": f"error_index_{index}",
                    "error": str(e),
                    "index": index
                })
    
    # Save all results
    output_file = f"results_swebench_phase2_gemini_batch_{start_index}_{start_index+num_problems-1}.json"
    
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"BATCH SUMMARY")
    print(f"{'='*80}")
    print(f"Total problems: {num_problems}")
    print(f"Successful: {sum(1 for r in all_results if 'error' not in r)}")
    print(f"Failed: {sum(1 for r in all_results if 'error' in r)}")
    print(f"\n✅ Results saved to: {output_file}")
    
    # Print aggregate metrics
    successful_results = [r for r in all_results if 'error' not in r and 'metadata' in r]
    if successful_results:
        print(f"\n{'='*80}")
        print(f"AGGREGATE METRICS")
        print(f"{'='*80}")
        
        total_time = sum(r['metadata']['total_time_seconds'] for r in successful_results)
        total_tokens = sum(r['metadata']['total_tokens'] for r in successful_results)
        avg_efficiency = sum(r['metadata']['efficiency_score'] for r in successful_results) / len(successful_results)
        
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Avg time per problem: {total_time/len(successful_results):.1f}s")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Avg tokens per problem: {total_tokens//len(successful_results):,}")
        print(f"Avg efficiency score: {avg_efficiency:.2%}")
        print(f"\nRouting decisions:")
        delegated = sum(1 for r in successful_results if r['metadata']['routing_decision'] == 'delegated')
        print(f"  Delegated: {delegated}/{len(successful_results)}")
        print(f"  Direct: {len(successful_results)-delegated}/{len(successful_results)}")

