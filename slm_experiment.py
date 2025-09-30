# slm_experiment.py
"""
SLM experiment: Qwen 2.5-Math-1.5B-Instruct (no tools)
"""
import time
import pandas as pd
import asyncio
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
# Qwen Wrapper Agent
# ---------------------------
class QwenAgent:
    def __init__(self, model_id="Qwen/Qwen2.5-Math-1.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16
        )

    async def run(self, prompt: str) -> str:
        """Generate an answer using Qwen 2.5"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0  # deterministic
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# ---------------------------
# Experiment Runner
# ---------------------------
async def run_slm_experiment(test_df: pd.DataFrame):
    """Run experiment with Qwen 2.5"""
    agent = QwenAgent()

    print("\n" + "="*60)
    print("SLM EXPERIMENT: Qwen 2.5 alone (no tools)")
    print("="*60)

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

    print_summary(summary, "SLM Qwen 2.5")
    print("\nSubject breakdown:")
    print(subject_stats)

    # Save results
    save_results(results, "results_slm_qwen2.5.json", summary)

    return summary


# ---------------------------
# Main
# ---------------------------
async def main(n_samples=None):
    # Load test data
    test_df = pd.read_csv("math500/test.csv")
    print(f"Loaded {len(test_df)} problems from math500/test.csv")

    # Optionally reduce sample size
    if n_samples is not None:
        test_df = test_df.sample(n=min(n_samples, len(test_df)), random_state=42)

    # Check required columns
    required_cols = ["problem", "answer", "subject"]
    if not all(col in test_df.columns for col in required_cols):
        print(f"CSV must contain columns: {required_cols}")
        print(f"Available columns: {list(test_df.columns)}")
        return

    # Run experiment
    await run_slm_experiment(test_df)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples to run")
    args = parser.parse_args()

    asyncio.run(main(n_samples=args.n_samples))
