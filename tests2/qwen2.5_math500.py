# qwen2_5_math500.py
import os, time, re, random, numpy as np, pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------
# Answer extraction
# ---------------------------
def extract_answer(text: str) -> str:
    """Extracts the numeric answer from model output."""
    m = re.findall(r"(?:the answer is|answer is)\s*(\d+)", text, re.IGNORECASE)
    if m:
        return m[-1]
    m = re.findall(r"\d+", text)
    return m[-1] if m else ""

# ---------------------------
# Model loading
# ---------------------------
def load_qwen(model_id="Qwen/Qwen2.5-Math-1.5B-Instruct"):
    """Load Qwen 2.5-Math SLM and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    return model, tokenizer

# ---------------------------
# Single problem solver
# ---------------------------
def solve_problem(model, tokenizer, problem: str, max_new_tokens=200) -> tuple[str, str, float]:
    """
    Solve one math problem using the model.

    Returns:
        pred (str): extracted numeric prediction
        output_text (str): raw model output
        latency (float): seconds
    """
    prompt = f"""
You are a highschool math expert. You think in steps.
Give the answer in the format: 'The answer is <number>'.

Problem:
{problem}
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy decoding
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    latency = time.time() - start

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = extract_answer(output_text)
    return pred, output_text, latency

# ---------------------------
# Evaluation loop
# ---------------------------
def evaluate_model(model, tokenizer, df: pd.DataFrame, n_samples=20, seed=42):
    """Evaluate Qwen on a random subset of MATH-500 problems."""
    sample_df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    results, latencies = [], []
    correct = 0

    for i, row in sample_df.iterrows():
        problem, answer = row["problem"], row["answer"]

        pred, raw_output, latency = solve_problem(model, tokenizer, problem)
        latencies.append(latency)

        gold_num = re.findall(r"\d+", str(answer))
        gold_num = gold_num[-1] if gold_num else ""
        is_correct = (pred == gold_num)
        if is_correct:
            correct += 1

        results.append({
            "id": row.get("unique_id", i),
            "problem": problem,
            "gold": gold_num,
            "pred": pred,
            "correct": is_correct,
            "latency": latency,
            "raw": raw_output[:200] + "..." if len(raw_output) > 200 else raw_output,
        })

        mark = "✓" if is_correct else "✗"
        print(f"{mark} {row.get('unique_id', i)} | Pred={pred or '∅'} | Gold={gold_num} | {latency:.2f}s")

    acc = correct / len(sample_df) * 100
    avg_latency = np.mean(latencies)

    print("\n📊 Qwen2.5-Math-1.5B-Instruct on MATH-500")
    print(f"Samples: {len(sample_df)}")
    print(f"Accuracy: {acc:.2f}% ({correct}/{len(sample_df)})")
    print(f"Avg latency: {avg_latency:.2f}s")

    return pd.DataFrame(results)

# ---------------------------
# Main
# ---------------------------
def main():
    set_seed(42)

    # Load model
    model, tokenizer = load_qwen()

    # Load dataset
    df = pd.read_csv("math500/test.csv")
    print(f"✓ Loaded {len(df)} test problems")

    # Evaluate on a subset
    results_df = evaluate_model(model, tokenizer, df, n_samples=20, seed=42)

    # Save results
    os.makedirs("outputs", exist_ok=True)
    out_file = "outputs/qwen2_5_math500_eval.csv"
    results_df.to_csv(out_file, index=False)
    print(f"💾 Saved results to {out_file}")

if __name__ == "__main__":
    main()
