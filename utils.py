"""
Utility functions and classes for experiment tracking
"""
import re
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class ProblemResult:
    """Results for a single problem"""
    problem_id: str
    subject: str
    difficulty: str
    question: str
    ground_truth: str
    prediction: str
    is_correct: bool
    latency_total: float
    latency_llm: float
    latency_slm: float
    tool_calls: int
    tool_call_details: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return asdict(self)

class MetricsTracker:
    """Track metrics across tool calls and problems"""
    
    def __init__(self):
        self.current_tool_calls = []
        self.current_slm_time = 0.0
        self.all_problems = []
        
    def start_problem(self):
        """Reset tracking for a new problem"""
        if self.current_tool_calls:  # Save previous problem if exists
            self.all_problems.append({
                'tool_calls': self.current_tool_calls.copy(),
                'slm_time': self.current_slm_time
            })
        self.current_tool_calls = []
        self.current_slm_time = 0.0
    
    def add_tool_call(self, tool_name: str, input_text: str, output_text: str, 
                      latency: float, is_duplicate: bool = False):
        """
        Track a tool call
        
        Args:
            tool_name: Name of the tool called
            input_text: Input to the tool
            output_text: Output from the tool
            latency: Time taken for the tool call
            is_duplicate: Whether this is a duplicate call
        """
        call_info = {
            'tool': tool_name,
            'input_text': input_text,
            'output_text': output_text,
            'latency': latency,
            'is_duplicate': is_duplicate,
            'timestamp': time.time()
        }
        self.current_tool_calls.append(call_info)
        
        # Add to SLM time if it's not a duplicate
        if not is_duplicate and tool_name == "slm_help":
            self.current_slm_time += latency
    
    def get_summary(self):
        """Get summary statistics"""
        total_calls = sum(len(p['tool_calls']) for p in self.all_problems)
        duplicate_calls = sum(
            len([c for c in p['tool_calls'] if c.get('is_duplicate', False)])
            for p in self.all_problems
        )
        
        return {
            'total_problems': len(self.all_problems),
            'total_tool_calls': total_calls,
            'duplicate_calls': duplicate_calls,
            'duplicate_rate': duplicate_calls / total_calls if total_calls > 0 else 0
        }

def extract_answer(text: str) -> str:
    """
    Extract numerical answer from text
    Handles various formats including boxed answers
    
    Args:
        text: Text containing the answer
    
    Returns:
        Extracted answer as string
    """
    if not text:
        return ""
    
    # First try to find boxed answer (most reliable)
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # Try to find "answer is" patterns
    patterns = [
        r"final answer is[:\s]*([+-]?\d+\.?\d*)",
        r"answer is[:\s]*([+-]?\d+\.?\d*)",
        r"equals?[:\s]*([+-]?\d+\.?\d*)",
        r"result is[:\s]*([+-]?\d+\.?\d*)",
        r"therefore[,\s]+([+-]?\d+\.?\d*)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Look for a number at the end of the text
    end_number = re.search(r'([+-]?\d+\.?\d*)\s*[.!]?\s*$', text)
    if end_number:
        return end_number.group(1).strip()
    
    return ""

def check_answer(prediction: str, ground_truth: str, tolerance: float = 1e-9) -> bool:
    """
    Check if prediction matches ground truth
    
    Args:
        prediction: The predicted answer (full text)
        ground_truth: The correct answer
        tolerance: Numerical tolerance for float comparison
    
    Returns:
        Boolean indicating if the answer is correct
    """
    # Extract answer from prediction if needed
    pred_answer = extract_answer(prediction)
    
    if not pred_answer:
        return False
    
    # Clean up ground truth
    ground_truth = str(ground_truth).strip()
    
    # Try exact string match first
    if pred_answer == ground_truth:
        return True
    
    # Try numerical comparison
    try:
        pred_num = float(pred_answer)
        truth_num = float(ground_truth)
        return abs(pred_num - truth_num) < tolerance
    except (ValueError, TypeError):
        # If not numbers, fall back to string comparison
        return pred_answer.lower() == ground_truth.lower()

def calculate_summary(results: List[ProblemResult]) -> Dict[str, Any]:
    """
    Calculate summary statistics from results
    
    Args:
        results: List of problem results
    
    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {
            'total_problems': 0,
            'correct': 0,
            'accuracy': 0.0,
            'avg_latency_total': 0.0,
            'avg_latency_llm': 0.0,
            'avg_latency_slm': 0.0,
            'avg_tool_calls': 0.0,
            'total_tool_calls': 0
        }
    
    correct_count = sum(1 for r in results if r.is_correct)
    total_tool_calls = sum(r.tool_calls for r in results)
    
    # Calculate averages
    avg_latency_total = sum(r.latency_total for r in results) / len(results)
    avg_latency_llm = sum(r.latency_llm for r in results) / len(results)
    avg_latency_slm = sum(r.latency_slm for r in results) / len(results)
    avg_tool_calls = total_tool_calls / len(results)
    
    # Calculate duplicate statistics
    duplicate_calls_total = sum(
        len([c for c in r.tool_call_details if c.get('is_duplicate', False)])
        for r in results
    )
    
    return {
        'total_problems': len(results),
        'correct': correct_count,
        'accuracy': correct_count / len(results),
        'avg_latency_total': avg_latency_total,
        'avg_latency_llm': avg_latency_llm,
        'avg_latency_slm': avg_latency_slm,
        'avg_tool_calls': avg_tool_calls,
        'total_tool_calls': total_tool_calls,
        'duplicate_calls': duplicate_calls_total,
        'duplicate_rate': duplicate_calls_total / total_tool_calls if total_tool_calls > 0 else 0
    }

def print_summary(summary: Dict[str, Any], experiment_name: str = "Experiment"):
    """
    Print formatted summary
    
    Args:
        summary: Summary dictionary from calculate_summary
        experiment_name: Name of the experiment for the header
    """
    print("\n" + "="*60)
    print(f"{experiment_name} Results Summary")
    print("="*60)
    print(f"Total Problems: {summary['total_problems']}")
    print(f"Correct: {summary['correct']}/{summary['total_problems']}")
    print(f"Accuracy: {summary['accuracy']:.2%}")
    print(f"\nLatency Statistics:")
    print(f"  Average Total: {summary['avg_latency_total']:.3f}s")
    print(f"  Average LLM: {summary['avg_latency_llm']:.3f}s")
    print(f"  Average SLM: {summary['avg_latency_slm']:.3f}s")
    print(f"\nTool Call Statistics:")
    print(f"  Average per problem: {summary['avg_tool_calls']:.2f}")
    print(f"  Total calls: {summary['total_tool_calls']}")
    if 'duplicate_rate' in summary:
        print(f"  Duplicate calls: {summary.get('duplicate_calls', 0)} ({summary['duplicate_rate']:.1%})")

def save_results(results: List[ProblemResult], filename: str, summary: Dict[str, Any] = None):
    """
    Save results to JSON file
    
    Args:
        results: List of problem results
        filename: Output filename
        summary: Optional summary to include
    """
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'results': [r.to_dict() for r in results],
    }
    
    if summary:
        output_data['summary'] = summary
    
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {filename}")

def load_results(filename: str) -> tuple[List[ProblemResult], Dict[str, Any]]:
    """
    Load results from JSON file
    
    Args:
        filename: Input filename
    
    Returns:
        Tuple of (results list, summary dict)
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    results = [
        ProblemResult(**r) for r in data.get('results', [])
    ]
    
    summary = data.get('summary', {})
    
    return results, summary

# Test functions
def test_extraction():
    """Test answer extraction"""
    test_cases = [
        ("The answer is 42", "42"),
        ("Therefore, \\boxed{123}", "123"),
        ("equals 3.14", "3.14"),
        ("The result is -7.", "-7"),
        ("After calculation: 999", "999"),
    ]
    
    print("Testing answer extraction:")
    for text, expected in test_cases:
        result = extract_answer(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{text}' -> '{result}' (expected '{expected}')")

if __name__ == "__main__":
    # Run tests
    test_extraction()