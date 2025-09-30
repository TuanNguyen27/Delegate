# router_experiment.py
"""
Router experiment: GPT-4o-mini with SLM tool routing
Usage: python router_experiment.py [--sample N]
"""
import time
import pandas as pd
import asyncio
import argparse
from agents import Runner

from utils import (
    ProblemResult,
    MetricsTracker,
    check_answer,
    extract_answer,
    save_results,
    calculate_summary,
    print_summary
)

# Create global tracker
tracker = MetricsTracker()

# ---------------------------
# Experiment Runner
# ---------------------------
async def run_router_experiment(test_df: pd.DataFrame):
    """Run router experiment with GPT-4o-mini + SLM tool"""
    
    # Import agent with tools
    from router_agent import agent
    
    print("\n" + "="*60)
    print("ROUTER EXPERIMENT: GPT-4o-mini + Qwen SLM Tool")
    print("="*60)
    print(f"Running on {len(test_df)} problems")
    
    results = []
    failed_problems = []
    
    for idx, row in test_df.iterrows():
        print(f"\n[{idx+1}/{len(test_df)}] Processing {row['subject']}...")
        
        tracker.start_problem()
        
        try:
            t_start = time.time()
            result = await Runner.run(agent, row["problem"], max_turns=15)
            t_end = time.time()
            
            prediction = result.final_output
            is_correct = check_answer(prediction, row["answer"])
            
            latency_total = t_end - t_start
            latency_llm = latency_total - tracker.current_slm_time
            
            problem_result = ProblemResult(
                problem_id=row.get("problem_id", f"prob_{idx}"),
                subject=row["subject"],
                difficulty=row.get("level", "unknown"),
                question=row["problem"],
                ground_truth=str(row["answer"]),
                prediction=prediction,
                is_correct=is_correct,
                latency_total=latency_total,
                latency_llm=latency_llm,
                latency_slm=tracker.current_slm_time,
                tool_calls=len(tracker.current_tool_calls),
                tool_call_details=tracker.current_tool_calls
            )
            
            results.append(problem_result)
            
            # Debug output
            extracted = extract_answer(prediction)
            print(f"   Result: {'CORRECT' if is_correct else 'WRONG'}")
            print(f"   Tool calls: {len(tracker.current_tool_calls)}")
            print(f"   Extracted: {extracted}")
            print(f"   Ground truth: {row['answer']}")
            print(f"   Latency: {latency_total:.2f}s (LLM: {latency_llm:.2f}s, SLM: {tracker.current_slm_time:.2f}s)")
            
        except Exception as e:
            print(f"   ERROR: {str(e)}")
            failed_problems.append({
                "index": idx,
                "subject": row["subject"],
                "error": str(e)
            })
            # Still add a result with error
            problem_result = ProblemResult(
                problem_id=row.get("problem_id", f"prob_{idx}"),
                subject=row["subject"],
                difficulty=row.get("level", "unknown"),
                question=row["problem"],
                ground_truth=str(row["answer"]),
                prediction=f"ERROR: {str(e)}",
                is_correct=False,
                latency_total=0.0,
                latency_llm=0.0,
                latency_slm=0.0,
                tool_calls=0,
                tool_call_details=[]
            )
            results.append(problem_result)
    
    # Calculate and print summary
    summary = calculate_summary(results)
    
    # Add subject breakdown
    df_results = pd.DataFrame([r.to_dict() for r in results])
    subject_stats = df_results.groupby("subject").agg({
        "is_correct": ["mean", "count"],
        "latency_total": "mean",
        "tool_calls": ["mean", "sum"]
    }).round(3)
    
    print_summary(summary, "Router")
    print("\nSubject breakdown:")
    print(subject_stats)
    
    if failed_problems:
        print(f"\n⚠️  {len(failed_problems)} problems failed:")
        for fail in failed_problems:
            print(f"   [{fail['index']}] {fail['subject']}: {fail['error'][:100]}")
    
    # Save results
    save_results(results, "results_router.json", summary)
    
    return summary

# ---------------------------
# Main
# ---------------------------
async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run router experiment on MATH500')
    parser.add_argument('--sample', type=int, default=None,
                       help='Number of problems to sample (default: all)')
    parser.add_argument('--random', action='store_true',
                       help='Random sample instead of first N')
    args = parser.parse_args()
    
    # Load test data
    test_df = pd.read_csv("math500/test.csv")
    print(f"Loaded {len(test_df)} problems from math500/test.csv")
    
    # Check required columns
    required_cols = ["problem", "answer", "subject"]
    if not all(col in test_df.columns for col in required_cols):
        print(f"CSV must contain columns: {required_cols}")
        print(f"Available columns: {list(test_df.columns)}")
        return
    
    # Sample if requested
    if args.sample:
        if args.random:
            test_df = test_df.sample(n=args.sample, random_state=42)
        else:
            test_df = test_df.head(args.sample)
        print(f"Sampled {len(test_df)} problems")
    
    # Run router experiment
    await run_router_experiment(test_df)

if __name__ == "__main__":
    asyncio.run(main())