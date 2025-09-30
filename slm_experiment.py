# slm_experiment.py
"""
SLM experiment: Qwen 2.5 alone (no tools)
Runs locally on GPU (T4 / A100 / etc.)
Works with both MATH500 and GSM8K datasets.

Usage: 
    python slm_experiment.py --dataset gsm8k --sample 30
    python slm_experiment.py --dataset math500 --sample 10
"""

import time
import pandas as pd
import asyncio
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    ProblemResult,
    check_answer,
    extract_answer,
    save_results,
    calculate_summary,
    print_summary
)

# ---------------------------
# Qwen Agent (local only)
# ---------------------------
class QwenAgent:
    def __init__(self, model_id="Qwen/Qwen2.5-Math-1.5B-Instruct", max_new_tokens=128):
        self.max_new_tokens = max_new_tokens
        print(f"🔥 Loading {model_id} locally...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            dtype=torch.float16,
            trust_remote_code=True
        )
        print("✅ Model ready (local inference).")

    async def run(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# ---------------------------
# Experiment Runner
# ---------------------------
async def run_slm_experiment(test_df: pd.DataFrame, output_file: str, max_new_tokens: int):
    agent = QwenAgent(max_new_tokens=max_new_tokens)

    print("\n" + "="*60)
    print("SLM EXPERIMENT: Qwen 2.5 alone (no tools)")
    print("="*60)
    print(f"Running on {len(test_df)} problems")

    results = []

    for idx, row in test_df.iterrows():
        print(f"\n[{idx+1}/{len(test_df)}] Processing {row['subject']}...")

        t_start = time.time()
        prediction = await agent.run(row["problem"])
        t_end = time.time()

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

        extracted = extract_answer(prediction)
        print(f"   Result: {'CORRECT' if is_correct else 'WRONG'}")
        print(f"   Extracted: {extracted}")
        print(f"   Ground truth: {row['answer']}")
        print(f"   Latency: {problem_result.latency_total:.2f}s")

    # Summary
    summary = calculate_summary(results)
    df_results = pd.DataFrame([r.to_dict() for r in results])
    subject_stats = df_results.groupby("subject").agg({
        "is_correct": ["mean", "count"],
        "latency_total": "mean"
    }).round(3)

    print_summary(summary, "SLM Qwen 2.5 (local)")
    print("\nSubject breakdown:")
    print(subject_stats)

    save_results(results, output_file, summary)
    print(f"\n💾 Results saved to: {output_file}")

    return summary


# ---------------------------
# Main
# ---------------------------
async def main():
    parser = argparse.ArgumentParser(description='Run SLM experiment (local Qwen)')
    parser.add_argument('--dataset', type=str, default='math500',
                       choices=['math500', 'gsm8k'])
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--max_tokens', type=int, default=512,
                       help='Max new tokens for Qwen output')
    args = parser.parse_args()

    # Load dataset
    if args.dataset == 'gsm8k':
        from gsm8k_loader import load_gsm8k_as_df
        test_df = load_gsm8k_as_df(
            split='test',
            n_samples=args.sample,
            random_seed=42 if args.random else None
        )
        output_file = f"results_slm_gsm8k{'_sample' + str(args.sample) if args.sample else ''}.json"
    else:
        test_df = pd.read_csv("math500/test.csv")
        print(f"Loaded {len(test_df)} problems from math500/test.csv")

        if args.sample:
            if args.random:
                test_df = test_df.sample(n=args.sample, random_state=42)
            else:
                test_df = test_df.head(args.sample)
            print(f"Sampled {len(test_df)} problems")

        output_file = f"results_slm_math500{'_sample' + str(args.sample) if args.sample else ''}.json"

    # Run experiment
    await run_slm_experiment(test_df, output_file, args.max_tokens)


if __name__ == "__main__":
    asyncio.run(main())
