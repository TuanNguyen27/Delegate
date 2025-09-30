# baseline_experiment.py
"""
Baseline experiment: GPT-4o-mini alone (no tools)
Uses the same answer extraction as your working llm_test.py
"""
import json
import time
import re
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass, asdict
import asyncio
from agents import Agent, Runner

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
    
    def to_dict(self):
        return asdict(self)

# ---------------------------
# Answer Extraction (matching your working code)
# ---------------------------
def extract_answer(text: str) -> str:
    """Extract answer using multiple strategies"""
    # Strategy 1: "the answer is X" pattern
    m = re.findall(r"(?:the answer is|answer is)\s*[:\-]?\s*(.+?)(?:\.|$)", text, re.IGNORECASE)
    if m:
        answer = m[-1].strip()
        # Clean up common patterns
        answer = re.sub(r'^[\$\\]*boxed\{([^}]+)\}', r'\1', answer)
        return answer
    
    # Strategy 2: \boxed{} format
    m = re.findall(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m[-1].strip()
    
    # Strategy 3: Last number in text
    m = re.findall(r"(?<!\d)(-?\d+(?:\.\d+)?(?:/\d+)?)(?!\d)", text)
    if m:
        return m[-1]
    
    return ""

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison"""
    answer = answer.strip().lower()
    # Remove common latex/formatting
    answer = answer.replace("\\text", "").replace("{", "").replace("}", "")
    answer = answer.replace("\\$", "").replace("$", "")
    answer = answer.replace("\\", "")
    # Remove spaces
    answer = "".join(answer.split())
    return answer

def check_answer(prediction: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth"""
    pred = normalize_answer(extract_answer(prediction))
    truth = normalize_answer(str(ground_truth))
    
    if not pred:
        return False
    
    # Direct match
    if pred == truth:
        return True
    
    # Try numerical comparison
    try:
        pred_val = eval(pred) if '/' in pred else float(pred)
        truth_val = eval(truth) if '/' in truth else float(truth)
        return abs(pred_val - truth_val) < 1e-6
    except:
        pass
    
    return False

# ---------------------------
# Experiment Runner
# ---------------------------
async def run_baseline_experiment(test_df: pd.DataFrame):
    """Run baseline experiment with GPT-4o-mini (no tools)"""
    
    # Create baseline agent
    agent = Agent(
        name="Math Expert Agent (Baseline)",
        instructions=(
            "You are an expert at solving high school competition math problems. "
            "Solve the problem step by step, showing your reasoning. "
            "At the end, provide your final answer in the format: 'The answer is <value>' "
            "or use \\boxed{<value>} format."
        ),
        model="gpt-4o-mini",
        tools=[]
    )
    
    print("\n" + "="*60)
    print("BASELINE EXPERIMENT: GPT-4o-mini alone (no tools)")
    print("="*60)
    
    results = []
    correct_count = 0
    
    for idx, row in test_df.iterrows():
        print(f"\n[{idx+1}/{len(test_df)}] Processing {row['subject']}...")
        
        t_start = time.time()
        result = await Runner.run(agent, row["problem"])
        t_end = time.time()
        
        prediction = result.final_output
        is_correct = check_answer(prediction, row["solution"])
        
        if is_correct:
            correct_count += 1
        
        problem_result = ProblemResult(
            problem_id=row.get("problem_id", f"prob_{idx}"),
            subject=row["subject"],
            difficulty=row.get("level", "unknown"),
            question=row["problem"],
            ground_truth=str(row["solution"]),
            prediction=prediction,
            is_correct=is_correct,
            latency_total=t_end - t_start
        )
        
        results.append(problem_result)
        
        # Debug output
        extracted = extract_answer(prediction)
        print(f"   Result: {'CORRECT' if is_correct else 'WRONG'}")
        print(f"   Extracted: {extracted}")
        print(f"   Ground truth: {row['solution']}")
        print(f"   Latency: {problem_result.latency_total:.2f}s")
    
    # Calculate summary
    accuracy = correct_count / len(test_df) if len(test_df) > 0 else 0
    avg_latency = sum(r.latency_total for r in results) / len(results) if results else 0
    
    summary = {
        "total_problems": len(results),
        "correct": correct_count,
        "accuracy": accuracy,
        "avg_latency": avg_latency,
        "by_subject": {}
    }
    
    # Subject breakdown
    df_results = pd.DataFrame([r.to_dict() for r in results])
    subject_stats = df_results.groupby("subject").agg({
        "is_correct": ["mean", "count"],
        "latency_total": "mean"
    }).round(3)
    summary["by_subject"] = subject_stats.to_dict()
    
    # Save results
    output_data = {
        "summary": summary,
        "detailed_results": [r.to_dict() for r in results]
    }
    
    with open("results_baseline.json", 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\n" + "="*60)
    print("BASELINE RESULTS")
    print("="*60)
    print(f"Total problems: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Average latency: {avg_latency:.2f}s")
    print("\nSubject breakdown:")
    print(subject_stats)
    print("\nResults saved to: results_baseline.json")
    
    return summary

# ---------------------------
# Main
# ---------------------------
async def main():
    # Load test data
    test_df = pd.read_csv("math500/test.csv")
    print(f"Loaded {len(test_df)} problems from math500/test.csv")
    
    # Check required columns
    required_cols = ["problem", "answer", "subject"]
    if not all(col in test_df.columns for col in required_cols):
        print(f"CSV must contain columns: {required_cols}")
        print(f"Available columns: {list(test_df.columns)}")
        return
    
    # Run baseline experiment
    await run_baseline_experiment(test_df)

if __name__ == "__main__":
    asyncio.run(main())