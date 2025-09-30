# math500_evaluation.py
"""
Comprehensive evaluation system for MATH500 with routing experiments
Tracks accuracy and latency metrics only.
"""
import json
import time
import re
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio

# ---------------------------
# Data Structures
# ---------------------------
@dataclass
class ProblemResult:
    problem_id: str
    subject: str
    difficulty: str  # Level 1-5 in MATH dataset
    question: str
    ground_truth: str
    
    # Prediction
    prediction: str
    is_correct: bool
    
    # Performance metrics
    latency_total: float
    latency_llm: float  # Time spent in LLM calls
    latency_slm: float  # Time spent in SLM calls
    
    # Tool usage
    tool_calls: int
    tool_call_details: List[Dict]  # Each tool call's question and result
    
    def to_dict(self):
        return asdict(self)

# ---------------------------
# Global Metrics Tracker
# ---------------------------
class MetricsTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.results: List[ProblemResult] = []
        self.current_tool_calls = []
        self.current_llm_time = 0.0
        self.current_slm_time = 0.0
    
    def start_problem(self):
        """Reset counters for new problem"""
        self.current_tool_calls = []
        self.current_llm_time = 0.0
        self.current_slm_time = 0.0
    
    def log_tool_call(self, question: str, result: str, latency: float):
        """Log a single tool call"""
        self.current_tool_calls.append({
            "question": question,
            "result": result,
            "latency": latency
        })
        self.current_slm_time += latency
    
    def add_result(self, result: ProblemResult):
        """Add completed problem result"""
        self.results.append(result)
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.results:
            return {}
        
        df = pd.DataFrame([r.to_dict() for r in self.results])
        
        return {
            "total_problems": len(self.results),
            "accuracy": df["is_correct"].mean(),
            "avg_latency": df["latency_total"].mean(),
            "avg_llm_latency": df["latency_llm"].mean(),
            "avg_slm_latency": df["latency_slm"].mean(),
            "avg_tool_calls": df["tool_calls"].mean(),
            "tool_usage_rate": (df["tool_calls"] > 0).mean(),
            "by_subject": self._summarize_by_subject(df),
            "by_difficulty": self._summarize_by_difficulty(df)
        }
    
    def _summarize_by_subject(self, df: pd.DataFrame) -> Dict:
        """Subject-specific metrics"""
        subjects = df.groupby("subject").agg({
            "is_correct": "mean",
            "latency_total": "mean",
            "tool_calls": ["mean", "sum"],
        }).round(3)
        return subjects.to_dict()
    
    def _summarize_by_difficulty(self, df: pd.DataFrame) -> Dict:
        """Difficulty-specific metrics"""
        difficulty = df.groupby("difficulty").agg({
            "is_correct": "mean",
            "latency_total": "mean",
            "tool_calls": "mean",
        }).round(3)
        return difficulty.to_dict()
    
    def save_results(self, filepath: str):
        """Save detailed results to JSON"""
        data = {
            "summary": self.get_summary(),
            "detailed_results": [r.to_dict() for r in self.results]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {filepath}")

# Global tracker instance
tracker = MetricsTracker()

# ---------------------------
# Answer Extraction & Checking
# ---------------------------
def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{} format"""
    pattern = r"\\boxed\{([^}]+)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    
    # Fallback: look for numbers at the end
    numbers = re.findall(r"(?<!\d)(-?\d+(?:\.\d+)?(?:/\d+)?)(?!\d)", text)
    return numbers[-1] if numbers else ""

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison"""
    answer = answer.strip().lower()
    # Remove common latex commands
    answer = answer.replace("\\text", "").replace("{", "").replace("}", "")
    answer = answer.replace("\\$", "").replace("$", "")
    # Remove spaces
    answer = "".join(answer.split())
    return answer

def check_answer(prediction: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth"""
    pred = normalize_answer(extract_boxed_answer(prediction))
    truth = normalize_answer(ground_truth)
    
    # Direct match
    if pred == truth:
        return True
    
    # Try numerical comparison for decimals/fractions
    try:
        pred_val = eval(pred) if '/' in pred else float(pred)
        truth_val = eval(truth) if '/' in truth else float(truth)
        return abs(pred_val - truth_val) < 1e-6
    except:
        pass
    
    return False

# ---------------------------
# Modified Agent Functions with Tracking
# ---------------------------
async def run_agent_with_tracking(question: str, agent, approach: str = "main"):
    """
    Run agent with full tracking
    
    Args:
        question: Math problem
        agent: Agent instance
        approach: "baseline" (no tools) or "main" (with tools)
    """
    from agents import Runner
    
    tracker.start_problem()
    
    t_start = time.time()
    result = await Runner.run(agent, question)
    t_end = time.time()
    
    latency_total = t_end - t_start
    tracker.current_llm_time = latency_total - tracker.current_slm_time
    
    return result.final_output

# ---------------------------
# Experiment Runners
# ---------------------------
async def run_baseline_experiment(test_df: pd.DataFrame, agent_no_tools):
    """Experiment 1: Baseline (LLM only, no tools)"""
    print("\n" + "="*60)
    print("EXPERIMENT 1: BASELINE (GPT-4o-mini alone)")
    print("="*60)
    
    tracker.reset()
    
    for idx, row in test_df.iterrows():
        print(f"\n[{idx+1}/{len(test_df)}] Processing {row['subject']}...")
        
        t_start = time.time()
        prediction = await run_agent_with_tracking(
            row["problem"], 
            agent_no_tools, 
            approach="baseline"
        )
        t_end = time.time()
        
        is_correct = check_answer(prediction, row["solution"])
        
        result = ProblemResult(
            problem_id=row.get("problem_id", f"prob_{idx}"),
            subject=row["subject"],
            difficulty=row.get("level", "unknown"),
            question=row["problem"],
            ground_truth=row["solution"],
            prediction=prediction,
            is_correct=is_correct,
            latency_total=t_end - t_start,
            latency_llm=tracker.current_llm_time,
            latency_slm=0.0,
            tool_calls=0,
            tool_call_details=[]
        )
        
        tracker.add_result(result)
        print(f"   {'Correct' if is_correct else 'Incorrect'}: {is_correct} | Latency: {result.latency_total:.2f}s")
    
    tracker.save_results("results_baseline.json")
    return tracker.get_summary()

async def run_main_experiment(test_df: pd.DataFrame, agent_with_tools):
    """Experiment 2: Main System (LLM + SLM tool)"""
    print("\n" + "="*60)
    print("EXPERIMENT 2: MAIN SYSTEM (GPT-4o-mini + Qwen SLM)")
    print("="*60)
    
    tracker.reset()
    
    for idx, row in test_df.iterrows():
        print(f"\n[{idx+1}/{len(test_df)}] Processing {row['subject']}...")
        
        t_start = time.time()
        prediction = await run_agent_with_tracking(
            row["problem"], 
            agent_with_tools, 
            approach="main"
        )
        t_end = time.time()
        
        is_correct = check_answer(prediction, row["solution"])
        
        result = ProblemResult(
            problem_id=row.get("problem_id", f"prob_{idx}"),
            subject=row["subject"],
            difficulty=row.get("level", "unknown"),
            question=row["problem"],
            ground_truth=row["solution"],
            prediction=prediction,
            is_correct=is_correct,
            latency_total=t_end - t_start,
            latency_llm=tracker.current_llm_time,
            latency_slm=tracker.current_slm_time,
            tool_calls=len(tracker.current_tool_calls),
            tool_call_details=tracker.current_tool_calls
        )
        
        tracker.add_result(result)
        print(f"   {'Correct' if is_correct else 'Incorrect'}: {is_correct} | Tools: {len(tracker.current_tool_calls)} | Latency: {result.latency_total:.2f}s")
    
    tracker.save_results("results_main.json")
    return tracker.get_summary()

# ---------------------------
# Analysis Functions
# ---------------------------
def compare_experiments(baseline_summary: Dict, main_summary: Dict):
    """Compare baseline vs main system"""
    print("\n" + "="*60)
    print("COMPARISON: Baseline vs Main System")
    print("="*60)
    
    metrics = ["accuracy", "avg_latency"]
    
    print(f"\n{'Metric':<20} {'Baseline':>12} {'Main System':>12} {'Change':>12}")
    print("-" * 60)
    
    for metric in metrics:
        baseline_val = baseline_summary.get(metric, 0)
        main_val = main_summary.get(metric, 0)
        
        if metric == "accuracy":
            change = f"{(main_val - baseline_val)*100:+.1f}pp"
        elif "latency" in metric:
            pct_change = ((main_val - baseline_val) / baseline_val * 100) if baseline_val else 0
            change = f"{pct_change:+.1f}%"
        else:
            pct_change = ((main_val - baseline_val) / baseline_val * 100) if baseline_val else 0
            change = f"{pct_change:+.1f}%"
        
        print(f"{metric:<20} {baseline_val:>12.3f} {main_val:>12.3f} {change:>12}")
    
    print(f"\n{'Tool Usage Rate':<20} {'N/A':>12} {main_summary.get('tool_usage_rate', 0):>12.1%} {'—':>12}")
    print(f"{'Avg Tool Calls':<20} {'N/A':>12} {main_summary.get('avg_tool_calls', 0):>12.2f} {'—':>12}")

def analyze_by_subject(results_file: str):
    """Detailed subject-level analysis"""
    with open(results_file) as f:
        data = json.load(f)
    
    df = pd.DataFrame(data["detailed_results"])
    
    print("\n" + "="*60)
    print("SUBJECT-LEVEL ANALYSIS")
    print("="*60)
    
    subject_stats = df.groupby("subject").agg({
        "is_correct": ["mean", "count"],
        "latency_total": "mean",
        "tool_calls": ["mean", "sum"]
    }).round(3)
    
    print("\n", subject_stats)
    
    # Identify best subjects for tool usage
    if "tool_calls" in df.columns:
        df["has_tools"] = df["tool_calls"] > 0
        tool_benefit = df.groupby(["subject", "has_tools"])["is_correct"].mean().unstack(fill_value=0)
        
        if False in tool_benefit.columns and True in tool_benefit.columns:
            tool_benefit["benefit"] = tool_benefit[True] - tool_benefit[False]
            print("\nTool Usage Benefit by Subject:")
            print(tool_benefit.sort_values("benefit", ascending=False))

# ---------------------------
# Main Execution
# ---------------------------
async def main():
    """Run all experiments"""
    
    # Load test data
    test_df = pd.read_csv("math500/test.csv")
    print(f"Loaded {len(test_df)} problems from math500/test.csv")
    
    # Ensure required columns exist
    required_cols = ["problem", "solution", "subject"]
    if not all(col in test_df.columns for col in required_cols):
        print(f"CSV must contain columns: {required_cols}")
        return
    
    # Import your agents (adjust based on your actual implementation)
    from router_agent import agent as agent_with_tools
    
    # Create baseline agent (same but no tools)
    from agents import Agent
    agent_no_tools = Agent(
        name="Math Expert Agent (Baseline)",
        instructions=agent_with_tools.instructions,
        model="gpt-4o-mini",
        tools=[]  # No tools
    )
    
    # Run experiments
    print("\nStarting evaluation pipeline...")
    
    baseline_summary = await run_baseline_experiment(test_df, agent_no_tools)
    main_summary = await run_main_experiment(test_df, agent_with_tools)
    
    # Compare results
    compare_experiments(baseline_summary, main_summary)
    
    # Detailed analysis
    analyze_by_subject("results_main.json")
    
    print("\nAll experiments complete!")
    print("Results saved to: results_baseline.json, results_main.json")

if __name__ == "__main__":
    asyncio.run(main())