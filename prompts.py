# prompts.py
"""
Centralized prompt management for all experiments and demos.
All prompts used across the project are defined here for easy tracking and modification.
"""

# ============================================================================
# LLM BASELINE EXPERIMENT
# ============================================================================

def get_llm_baseline_prompt(problem: str) -> str:
    """
    Prompt for LLM-only baseline (Gemini 2.5 Flash alone).
    Used in: experiments/llm_experiment.py
    
    Args:
        problem: The math problem to solve
        
    Returns:
        Complete prompt string
    """
    return f"""You are an expert at solving math problems.
Solve this problem step by step.
Provide your final answer in \\boxed{{answer}} format.

Problem: {problem}"""


# ============================================================================
# SLM BASELINE EXPERIMENT
# ============================================================================

def get_slm_baseline_prompt(problem: str) -> str:
    """
    Prompt for SLM-only baseline (Qwen 2.5 Math alone).
    Used in: experiments/slm_experiment.py
    
    Args:
        problem: The math problem to solve
        
    Returns:
        Complete prompt string
    """
    return f"""You are an expert at solving math problems.
Solve this problem step by step.
Provide your final answer in \\boxed{{answer}} format.

Problem: {problem}"""


# ============================================================================
# ROUTER SYSTEM (EXPERIMENT VERSION)
# ============================================================================

# Tool description for Gemini function calling
ROUTER_TOOL_DESCRIPTION = (
    "A specialized math calculator that solves mathematical expressions and equations. "
    "Pass it a mathematical expression (e.g., '(490 - 150) * 194' or '21 * (3/8) * 24'). "
    "It returns the numerical answer that you should use directly in your final response."
)

ROUTER_TOOL_PARAMETER_DESCRIPTION = "A mathematical expression to calculate"

# System instructions for the router agent (experiment version)
ROUTER_INSTRUCTIONS_EXPERIMENT = """You are an expert at solving math problems by effectively breaking down word problems into mathematical expressions.
For ANY calculation, call slm_help ONCE.
Provide your final answer in \\boxed{{answer}} format.
"""

# Alternative version: Multi-step approach (if single-step doesn't work well)
ROUTER_INSTRUCTIONS_EXPERIMENT_MULTISTEP = """You are a math problem solver with access to a calculator tool.

## Your Task:

For each problem:
1. Break down the problem into clear steps
2. For each step that requires calculation, call slm_help with a mathematical expression
3. Use the results to provide the final answer in \\boxed{} format

## Important Rules:

- **Always formulate MATHEMATICAL EXPRESSIONS** (e.g., "(100 - 20 - 44)", "3 * 7 * (3/8) * 24")
- **Never pass word problems to the tool** - convert them to expressions first
- **Always respond after receiving tool results** - don't leave your response empty
- **Use tool results directly** - they are correct

## Example:

Problem: "A team of 4 painters worked 3/8 of a day for 3 weeks. How many hours per painter? (7 days/week, 24 hours/day)"

Your response:
"I need to calculate: (weeks × days/week × fraction of day × hours/day) / painters

Let me calculate: 3 * 7 * (3/8) * 24 / 4"

[Call slm_help("3 * 7 * (3/8) * 24 / 4")]
[Tool returns: "47.25"]

Your response:
"Each painter worked \\boxed{47.25} hours"
"""

# Backup versions (kept for reference)
ROUTER_INSTRUCTIONS_EXPERIMENT_BU2 = f"""You are an expert at decomposing math problems into mathematical expression.
Convert the math problem into mathematical expressions and call slm_help.
Provide your final answer in \\boxed{{answer}} format."""

ROUTER_INSTRUCTIONS_EXPERIMENT_BU = (
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


# ============================================================================
# HELPER FUNCTIONS FOR ROUTER PROMPTS
# ============================================================================

def get_router_instructions(version: str = "default") -> str:
    """
    Get router instructions by version name.
    
    Args:
        version: One of:
            - "default" or "experiment" - Main experimental prompt (recommended)
            - "multistep" - Alternative multi-step approach
            - "backup" or "bu" - Original simple version
            
    Returns:
        System instruction string
        
    Examples:
        >>> instructions = get_router_instructions("default")
        >>> instructions = get_router_instructions("multistep")
    """
    versions = {
        "default": ROUTER_INSTRUCTIONS_EXPERIMENT,
        "experiment": ROUTER_INSTRUCTIONS_EXPERIMENT,
        "multistep": ROUTER_INSTRUCTIONS_EXPERIMENT_MULTISTEP,
        "backup": ROUTER_INSTRUCTIONS_EXPERIMENT_BU,
        "bu": ROUTER_INSTRUCTIONS_EXPERIMENT_BU,
    }
    
    return versions.get(version, ROUTER_INSTRUCTIONS_EXPERIMENT)


# ============================================================================
# PROMPT VERSIONING & TRACKING
# ============================================================================

PROMPT_VERSION = "2.0.0"
LAST_UPDATED = "2024-10-12"

PROMPT_CHANGELOG = """
Version 2.0.0 (2024-10-12):
- MAJOR: Comprehensive router prompt rewrite to fix "No Response Generated" issue
- Added detailed workflow with 5 clear steps
- Added 3 concrete examples showing word problem → expression conversion
- Emphasized: "ALWAYS respond after receiving tool result"
- Emphasized: "Formulate MATHEMATICAL EXPRESSIONS, not word problems"
- Improved tool description to be clearer about expected input format
- Changed SLM response format to: "The calculation result is: X"
- Added explicit instruction: "Never leave the response empty"
- Added alternative ROUTER_INSTRUCTIONS_EXPERIMENT_MULTISTEP for complex problems
- Added get_router_instructions() helper function

Version 1.0.0 (2024-10-11):
- Initial centralized prompt system
- Extracted prompts from experiments/llm_experiment.py
- Extracted prompts from experiments/slm_experiment.py
- Extracted prompts from experiments/router_agent.py
- Extracted prompts from router_agent_demo.py
- Added helper functions for easy access
- Added template for custom domains
"""
