# llm_experiment.py
"""
Baseline experiment: GPT-4o-mini alone (no tools)
"""
import time
import pandas as pd
import asyncio
from agents import Agent, Runner, ModelSettings

from utils import (
    ProblemResult,
    check_answer,
    extract_answer,
    save_results,
    calculate_summary,
    print_summary
)

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
        model_settings=ModelSettings(max_tokens=2048),
        tools=[]
    )
    
    print("\n" + "="*60)
    print("BASELINE EXPERIMENT: GPT-4o-mini alone (no tools)")
    print("="*60)
    
    results = []
    
    for idx, row in test_df.iterrows():
        print(f"\n[{idx+1}/{len(test_df)}] Processing {row['subject']}...")
        
        t_start = time.time()
        result = await Runner.run(agent, row["problem"])
        t_end = time.time()
        
        prediction = result.final_output
        is_correct = check_answer(prediction, row["answer"])
        
        problem_result = ProblemResult(
            problem_id=row.get("problem_id", f"prob_{idx}"),
            subject=row["subject"],
            difficulty=row.get("level", "unknown"),
            question=row["problem"],
            ground_truth=str(row["answer"]),
            prediction=prediction,
            is_correct=is_correct,
            latency_total=t_end - t_start
        )
        
        results.append(problem_result)
        
        # Debug output
        extracted = extract_answer(prediction)
        print(f"   Result: {'CORRECT' if is_correct else 'WRONG'}")
        print(f"   Extracted: {extracted}")
        print(f"   Ground truth: {row['answer']}")
        print(f"   Latency: {problem_result.latency_total:.2f}s")
    
    # Calculate and print summary
    summary = calculate_summary(results)
    
    # Add subject breakdown
    df_results = pd.DataFrame([r.to_dict() for r in results])
    subject_stats = df_results.groupby("subject").agg({
        "is_correct": ["mean", "count"],
        "latency_total": "mean"
    }).round(3)
    
    print_summary(summary, "Baseline")
    print("\nSubject breakdown:")
    print(subject_stats)
    
    # Save results
    save_results(results, "results_baseline.json", summary)
    
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