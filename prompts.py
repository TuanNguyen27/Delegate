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
    "Solve a mathematical calculation using specialized math model. "
    "Returns definitive answer that should be trusted immediately."
)

ROUTER_TOOL_PARAMETER_DESCRIPTION = "The calculation to perform"

# System instructions for the router agent (experiment version)
ROUTER_INSTRUCTIONS_EXPERIMENT = (
    "You solve math problems step by step.\n\n"
    
    "WORKFLOW:\n"
    "1. Break down the problem\n"
    "2. For ANY calculation, call slm_help\n"
    "3. When you receive 'CALCULATION COMPLETE:', use that answer\n"
    "4. Provide final answer in \\boxed{}\n\n"
)

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
