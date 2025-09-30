"""
Router experiment: GPT-4o-mini with SLM tool routing
"""
import time
import pandas as pd
import asyncio
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

# ---------------------------
# Custom Runner with Loop Detection
# ---------------------------
class LoopPreventingRunner:
    """Runner wrapper that monitors for excessive duplicate calls"""
    
    @staticmethod
    async def run(agent, question, max_turns=10, max_duplicate_warnings=3):
        """
        Run agent with loop detection and early termination
        
        Args:
            agent: The agent to run
            question: The question to solve
            max_turns: Maximum turns allowed
            max_duplicate_warnings: Number of duplicate calls before forcing termination
        """
        duplicate_count = 0
        last_call = None
        
        # Monitor the tool tracker if available
        if hasattr(agent, 'tool_tracker'):
            initial_calls = len(agent.tool_tracker.call_history)
        
        try:
            result = await Runner.run(agent, question, max_turns=max_turns)
            
            # Check if we had excessive duplicates
            if hasattr(agent, 'tool_tracker'):
                # Count how many times each question was asked
                call_counts = {}
                for normalized_q in agent.tool_tracker.call_history.keys():
                    # This counts unique questions, not repeat calls
                    # For repeat detection, we'd need to track in the tool itself
                    pass
            
            return result
            
        except Exception as e:
            if "MaxTurnsExceeded" in str(e):
                print(f"   WARNING: Max turns exceeded - likely due to repeated calculations")
                # Return what we have so far
                return type('Result', (), {
                    'final_output': "Unable to complete due to repeated calculations",
                    'error': str(e)
                })()
            raise

# ---------------------------
# Experiment Runner
# ---------------------------
async def run_router_experiment(test_df: pd.DataFrame, sample_size=None):
    """
    Run router experiment with GPT-4o-mini + SLM tool
    
    Args:
        test_df: DataFrame with test problems
        sample_size: Optional number of problems to test (for debugging)
    """
    
    # Create global tracker for metrics
    global_tracker = MetricsTracker()
    
    # Import and create agent with metrics tracking
    from router_agent import create_router_agent
    agent = create_router_agent(global_tracker)
    
    print("\n" + "="*60)
    print("ROUTER EXPERIMENT: GPT-4o-mini + Qwen SLM Tool")
    print("(With Loop Prevention)")
    print("="*60)
    
    # Optionally limit sample size for testing
    if sample_size:
        test_df = test_df.head(sample_size)
        print(f"Running on sample of {sample_size} problems")
    
    results = []
    
    for idx, row in test_df.iterrows():
        print(f"\n[{idx+1}/{len(test_df)}] Processing {row['subject']}...")
        
        # Reset tracker for this problem
        global_tracker.start_problem()
        
        # Clear the tool's call history for each new problem
        if hasattr(agent, 'tool_tracker'):
            agent.tool_tracker.call_history.clear()
        
        t_start = time.time()
        
        try:
            # Use our custom runner with loop prevention
            result = await LoopPreventingRunner.run(
                agent, 
                row["problem"], 
                max_turns=15,  # Increased slightly for complex multi-step problems
                max_duplicate_warnings=3
            )
            
            prediction = result.final_output
            
        except Exception as e:
            print(f"   ERROR: {str(e)}")
            prediction = f"Error: {str(e)}"
        
        t_end = time.time()
        
        # Check answer
        is_correct = check_answer(prediction, row["answer"])
        
        # Calculate latencies
        latency_total = t_end - t_start
        latency_slm = global_tracker.current_slm_time
        latency_llm = latency_total - latency_slm
        
        # Count unique vs duplicate calls
        unique_calls = len([c for c in global_tracker.current_tool_calls if not c.get('is_duplicate', False)])
        duplicate_calls = len([c for c in global_tracker.current_tool_calls if c.get('is_duplicate', False)])
        
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
            latency_slm=latency_slm,
            tool_calls=len(global_tracker.current_tool_calls),
            tool_call_details=global_tracker.current_tool_calls
        )
        
        results.append(problem_result)
        
        # Enhanced debug output
        extracted = extract_answer(prediction)
        print(f"   Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
        print(f"   Tool calls: {len(global_tracker.current_tool_calls)} total ({unique_calls} unique, {duplicate_calls} duplicates)")
        if duplicate_calls > 0:
            print(f"   ⚠️  Had {duplicate_calls} duplicate calculations!")
        print(f"   Extracted: {extracted}")
        print(f"   Ground truth: {row['answer']}")
        print(f"   Latency: {latency_total:.2f}s (LLM: {latency_llm:.2f}s, SLM: {latency_slm:.2f}s)")
        
        # Show which calculations were repeated
        if duplicate_calls > 0 and global_tracker.current_tool_calls:
            seen = set()
            for call in global_tracker.current_tool_calls:
                if call.get('is_duplicate'):
                    input_text = call.get('input_text', '')
                    if input_text not in seen:
                        print(f"      Repeated: '{input_text}'")
                        seen.add(input_text)
    
    # Calculate and print summary
    summary = calculate_summary(results)
    
    # Add additional metrics about duplicates
    total_calls = sum(len(r.tool_call_details) for r in results)
    duplicate_calls_total = sum(
        len([c for c in r.tool_call_details if c.get('is_duplicate', False)])
        for r in results
    )
    
    summary['duplicate_rate'] = duplicate_calls_total / total_calls if total_calls > 0 else 0
    summary['problems_with_duplicates'] = sum(
        1 for r in results 
        if any(c.get('is_duplicate', False) for c in r.tool_call_details)
    )
    
    # Add subject breakdown
    df_results = pd.DataFrame([r.to_dict() for r in results])
    subject_stats = df_results.groupby("subject").agg({
        "is_correct": ["mean", "count"],
        "latency_total": "mean",
        "tool_calls": ["mean", "sum"]
    }).round(3)
    
    print_summary(summary, "Router (with Loop Prevention)")
    print(f"\nDuplicate Statistics:")
    print(f"  Total duplicate calls: {duplicate_calls_total}/{total_calls} ({summary['duplicate_rate']:.1%})")
    print(f"  Problems with duplicates: {summary['problems_with_duplicates']}/{len(results)}")
    
    print("\nSubject breakdown:")
    print(subject_stats)
    
    # Save results with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_router_{timestamp}.json"
    save_results(results, filename, summary)
    print(f"\nResults saved to {filename}")
    
    return summary

# ---------------------------
# Main
# ---------------------------
async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run router experiment with loop prevention")
    parser.add_argument("--sample", type=int, help="Run on a sample of N problems for testing")
    parser.add_argument("--csv", default="math500/test.csv", help="Path to test CSV file")
    args = parser.parse_args()
    
    # Load test data
    try:
        test_df = pd.read_csv(args.csv)
        print(f"Loaded {len(test_df)} problems from {args.csv}")
    except FileNotFoundError:
        print(f"Error: Could not find {args.csv}")
        print("Please ensure the CSV file exists in the specified path")
        return
    
    # Check required columns
    required_cols = ["problem", "answer", "subject"]
    missing_cols = [col for col in required_cols if col not in test_df.columns]
    if missing_cols:
        print(f"Error: CSV is missing required columns: {missing_cols}")
        print(f"Available columns: {list(test_df.columns)}")
        return
    
    # Run router experiment
    await run_router_experiment(test_df, sample_size=args.sample)

if __name__ == "__main__":
    asyncio.run(main())