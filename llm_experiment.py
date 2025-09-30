# llm_experiment.py
"""
Baseline experiment: GPT-4o-mini alone (no tools)
Works with both MATH500 and GSM8K datasets

Usage: 
    python llm_experiment.py --dataset gsm8k --sample 30
    python llm_experiment.py --dataset math500 --sample 10
"""
import time
import pandas as pd
import asyncio
import argparse
from agents import Agent, Runner  # assumes your custom agent system

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
async def run_llm_experiment(test_df: pd.DataFrame, output_file: str):
    """Run baseline experiment with GPT-4o-mini (no tools)"""

    agent = Agent(
        name="Math Expert Agent (Baseline)",
        instructions=(
            "You are an expert at solving high school competition math problems. "
            "Solve the problem step by step, showing your reasoning. "
            "At the end, provide your final answer in the format: 'The answer is <value>' "
            "or use \\boxed{<value>} format."
        ),
        model="gpt-4o-mini",
        max_tokens=2048,   # <-- no ModelSettings
        tools=[]
    )

    print("\n" + "="*60)
    print("BASELINE EXPERIMENT: GPT-4o-mini alone (no tools)")
    print("="*60)
    print(f"Running on {len(test_df)} problems")

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

    df_results = pd.DataFrame([r.to_dict() for r in results])
    subject_stats = df_results.groupby("subject").agg({
        "is_correct": ["mean", "count"],
        "latency_total": "mean"
    }).round(3)

    print_summary(summary, "Baseline")
    print("\nSubject breakdown:")
    print(subject_stats)

    save_results(results, output_file, summary)
    print(f"\nResults saved to: {output_file}")

    return summary


# ---------------------------
# Main
# ---------------------------
async def main():
    parser = argparse.ArgumentParser(description='Run baseline LLM experiment')
    parser.add_argument('--dataset', type=str, default='math500',
                       choices=['math500', 'gsm8k'])
    parser.add_argument('--sample', type=int, default=None,
                       help='Number of problems to sample')
    parser.add_argument('--random', action='store_true',
                       help='Random sample instead of first N')
    args = parser.parse_args()

    # Load dataset
    if args.dataset == 'gsm8k':
        from gsm8k_loader import load_gsm8k_as_df
        test_df = load_gsm8k_as_df(
            split='test',
            n_samples=args.sample,
            random_seed=42 if args.random else None
        )
        output_file = f"results_llm_gsm8k{'_sample' + str(args.sample) if args.sample else ''}.json"
    else:
        test_df = pd.read_csv("math500/test.csv")
        print(f"Loaded {len(test_df)} problems from math500/test.csv")

        if args.sample:
            if args.random:
                test_df = test_df.sample(n=args.sample, random_state=42)
            else:
                test_df = test_df.head(args.sample)
            print(f"Sampled {len(test_df)} problems")

        output_file = f"results_llm_math500{'_sample' + str(args.sample) if args.sample else ''}.json"

    # Run experiment
    await run_llm_experiment(test_df, output_file)


if __name__ == "__main__":
    asyncio.run(main())
