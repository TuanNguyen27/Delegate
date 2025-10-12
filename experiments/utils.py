# experiments/utils.py
"""
Utility functions for MATH500 evaluation experiments
"""
import re
import json
from typing import Dict, List
from dataclasses import dataclass, asdict

# ---------------------------
# Data Structures
# ---------------------------
@dataclass
class ProblemResult:
    problem_id: str
    subject: str
    difficulty: str
    question: str
    ground_truth: str
    prediction: str
    is_correct: bool
    latency_total: float
    latency_llm: float = 0.0
    latency_slm: float = 0.0
    tool_calls: int = 0
    tool_call_details: List[Dict] = None
    # Debug fields for router (stores full conversation)
    llm_conversation: List[Dict] = None  # All LLM turns: [{"turn": 1, "input": "...", "output": "...", "function_calls": [...]}]
    slm_calls: List[Dict] = None  # All SLM calls: [{"input": "...", "output": "...", "latency": 0.5}]
    
    def __post_init__(self):
        if self.tool_call_details is None:
            self.tool_call_details = []
        if self.llm_conversation is None:
            self.llm_conversation = []
        if self.slm_calls is None:
            self.slm_calls = []
    
    def to_dict(self):
        return asdict(self)

# ---------------------------
# Answer Extraction (improved to handle boxed answers and decimals)
# ---------------------------
def normalize_number(num_str: str) -> str:
    """
    Normalize a number string for comparison.
    Handles decimals, commas, etc.
    
    Args:
        num_str: Number string like "42", "42.00", "1,000"
        
    Returns:
        Normalized string (integer if no decimal part, else float)
    """
    if not num_str:
        return ""
    
    # Remove commas and dollar signs
    cleaned = num_str.replace(",", "").replace("$", "").strip()
    
    try:
        # Try to parse as float
        num = float(cleaned)
        # If it's effectively an integer, return as int string
        if num == int(num):
            return str(int(num))
        else:
            return str(num)
    except ValueError:
        # Not a valid number, return as-is
        return cleaned

def extract_answer(text: str) -> str:
    """
    Extract numerical answer from model output.
    Priority order:
    1. Last \\boxed{...} expression (LaTeX format)
    2. "the answer is X" pattern
    3. Last number in text
    
    Args:
        text: Model output text
        
    Returns:
        Extracted answer as normalized string
    """
    # Priority 1: Look for \boxed{...} (most reliable for math problems)
    # Match both \\boxed{X} and \boxed{X}, and handle nested expressions
    boxed_matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed_matches:
        # Take the LAST boxed expression (final answer)
        last_boxed = boxed_matches[-1]
        # Extract number from inside boxed (might have $ or other formatting)
        # Support numbers with commas like 1,000
        numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', last_boxed)
        if numbers:
            return normalize_number(numbers[-1])
    
    # Priority 2: "the answer is X" pattern
    # Now handles decimals too
    m = re.findall(r"(?:the answer is|answer is)\s*\$?(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return normalize_number(m[-1])
    
    # Priority 3: Fallback - last number in text (handles decimals)
    m = re.findall(r'-?\d+(?:\.\d+)?', text)
    if m:
        return normalize_number(m[-1])
    
    return ""

def extract_ground_truth(answer: str) -> str:
    """
    Extract ground truth answer from answer column.
    Now handles decimals and normalizes the output.
    
    Args:
        answer: Content from 'answer' column
        
    Returns:
        Extracted answer as normalized string
    """
    # Handle decimals in ground truth too
    gold_num = re.findall(r'-?\d+(?:\.\d+)?', str(answer))
    if gold_num:
        return normalize_number(gold_num[-1])
    return ""

def check_answer(prediction: str, ground_truth: str) -> bool:
    """
    Check if predicted answer matches ground truth.
    
    Args:
        prediction: Model's full output text
        ground_truth: Content from 'answer' column
        
    Returns:
        True if answers match, False otherwise
    """
    pred = extract_answer(prediction)
    gold = extract_ground_truth(ground_truth)
    return pred == gold

# ---------------------------
# Metrics Tracker
# ---------------------------
class MetricsTracker:
    """Track tool usage metrics and debug info during evaluation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracking for new experiment"""
        self.current_tool_calls = []
        self.current_slm_time = 0.0
        # Debug fields
        self.current_llm_conversation = []
        self.current_slm_calls = []
    
    def start_problem(self):
        """Reset counters for new problem"""
        self.current_tool_calls = []
        self.current_slm_time = 0.0
        self.current_llm_conversation = []
        self.current_slm_calls = []
    
    def log_tool_call(self, question: str, result: str, latency: float, input_tokens: int = 0, output_tokens: int = 0):
        """
        Log a single SLM tool call.
        
        Args:
            question: Question sent to SLM
            result: JSON result from SLM
            latency: Time taken by SLM
            input_tokens: SLM input tokens
            output_tokens: SLM output tokens
        """
        self.current_tool_calls.append({
            "question": question,
            "result": result,
            "latency": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        })
        self.current_slm_time += latency
    
    def log_slm_call(self, question: str, full_output: str, latency: float, input_tokens: int = 0, output_tokens: int = 0):
        """
        Log full SLM input/output for debugging.
        
        Args:
            question: Question sent to SLM
            full_output: Full SLM response (before extraction)
            latency: Time taken
            input_tokens: Input token count
            output_tokens: Output token count
        """
        self.current_slm_calls.append({
            "input": question,
            "output": full_output,
            "latency": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        })
    
    def log_llm_turn(self, turn: int, user_input: str, llm_output: str, function_calls: List[Dict] = None):
        """
        Log a single LLM conversation turn for debugging.
        
        Args:
            turn: Turn number (0-indexed)
            user_input: Input prompt/message to LLM
            llm_output: LLM text response
            function_calls: List of function calls made (if any)
        """
        self.current_llm_conversation.append({
            "turn": turn,
            "input": user_input,
            "output": llm_output,
            "function_calls": function_calls or []
        })

# ---------------------------
# Results Saving
# ---------------------------
def save_results(results: List[ProblemResult], filepath: str, summary: Dict = None):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: List of ProblemResult objects
        filepath: Output file path
        summary: Optional summary statistics dict
    """
    output_data = {
        "detailed_results": [r.to_dict() for r in results]
    }
    
    if summary:
        output_data["summary"] = summary
    
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {filepath}")

# ---------------------------
# Summary Statistics
# ---------------------------
def calculate_summary(results: List[ProblemResult]) -> Dict:
    """
    Calculate summary statistics from results.
    
    Args:
        results: List of ProblemResult objects
        
    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {}
    
    correct_count = sum(1 for r in results if r.is_correct)
    total = len(results)
    
    summary = {
        "total_problems": total,
        "correct": correct_count,
        "accuracy": correct_count / total if total > 0 else 0,
        "avg_latency": sum(r.latency_total for r in results) / total if total > 0 else 0,
    }
    
    # Add router-specific metrics if available
    if any(r.tool_calls > 0 for r in results):
        summary["avg_llm_latency"] = sum(r.latency_llm for r in results) / total if total > 0 else 0
        summary["avg_slm_latency"] = sum(r.latency_slm for r in results) / total if total > 0 else 0
        summary["avg_tool_calls"] = sum(r.tool_calls for r in results) / total if total > 0 else 0
        summary["tool_usage_rate"] = sum(1 for r in results if r.tool_calls > 0) / total if total > 0 else 0
    
    return summary

def print_summary(summary: Dict, experiment_name: str = "Experiment"):
    """
    Print formatted summary statistics.
    
    Args:
        summary: Summary dictionary from calculate_summary()
        experiment_name: Name to display in header
    """
    print("\n" + "="*60)
    print(f"{experiment_name.upper()} RESULTS")
    print("="*60)
    print(f"Total problems: {summary['total_problems']}")
    print(f"Correct: {summary['correct']}")
    print(f"Accuracy: {summary['accuracy']*100:.2f}%")
    print(f"Average latency: {summary['avg_latency']:.2f}s")
    
    if "avg_tool_calls" in summary:
        print(f"Average LLM latency: {summary['avg_llm_latency']:.2f}s")
        print(f"Average SLM latency: {summary['avg_slm_latency']:.2f}s")
        print(f"Tool usage rate: {summary['tool_usage_rate']*100:.1f}%")
        print(f"Average tool calls: {summary['avg_tool_calls']:.2f}")