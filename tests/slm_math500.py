# gemma_baseline_math500.py
import os, time, re, random, numpy as np, pandas as pd
from dotenv import load_dotenv
from google import genai
from omegaconf import OmegaConf
from datasets import load_dataset

# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

# ---------------------------
# Answer extraction
# ---------------------------
def extract_answer(text):
    # Look for "The answer is 42"
    m = re.findall(r"(?:the answer is|answer is)\s*(\d+)", text, re.IGNORECASE)
    if m:
        return m[-1]

    # Otherwise last number
    m = re.findall(r"\d+", text)
    if m:
        return m[-1]

    return ""

# ---------------------------
# Main evaluation
# ---------------------------
def main():
    load_dotenv()
    api_key = os.getenv("GEMMA_API_KEY")
    if not api_key:
        raise ValueError("❌ GEMMA_API_KEY not found in .env")

    client = genai.Client(api_key=api_key)

    # Load dataset
    cfg = OmegaConf.load("configs/math500.yaml")
    set_seed(cfg.seed)

    ds = load_dataset(cfg.dataset, split=cfg.split)
    if cfg.get("n_samples"):
        ds = ds.shuffle(seed=cfg.seed).select(range(cfg.n_samples))

    print(f"✓ Loaded {len(ds)} MATH-500 samples")

    results, latencies = [], []

    for i, row in enumerate(ds):
        problem, solution, answer = row["problem"], row["solution"], row["answer"]
        subject, level, uid = row["subject"], row["level"], row["unique_id"]

        prompt = f"""
You are a math tutor.
Solve the following problem step by step.
At the very end, give the answer in the format: 'The answer is <number>'.

Problem:
{problem}
"""

        start = time.time()
        response = client.models.generate_content(
            model="gemma-3-4b-it",  
            contents=prompt,
            config={"max_output_tokens": 1024} 
        )
        latency = time.time() - start
        latencies.append(latency)

        output = response.text
        pred = extract_answer(output)

        gold_num = re.findall(r"\d+", str(answer))
        gold_num = gold_num[-1] if gold_num else ""

        is_correct = (pred == gold_num)

        results.append({
            "id": i,
            "problem": problem,
            "solution": solution,
            "answer": gold_num,
            "subject": subject,
            "level": level,
            "unique_id": uid,
            "pred": pred,
            "correct": is_correct,
            "latency": latency,
            "raw": output[:200] + "..." if len(output) > 200 else output,
        })

        mark = "✓" if is_correct else "✗"
        print(f"{mark} Q{i} | Pred={pred or '∅'} | Gold={gold_num} | {latency:.2f}s")

    # Metrics
    acc = sum(r["correct"] for r in results) / len(results) * 100 if results else 0
    avg_latency = np.mean(latencies) if latencies else 0

    print("\n📊 Gemma-3-4B-it on MATH-500")
    print(f"Samples: {len(results)}")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Avg latency: {avg_latency:.2f}s")

    # Save
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_file = os.path.join(cfg.output_dir, "math500_gemma_baseline.csv")
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"💾 Saved results to {out_file}")


if __name__ == "__main__":
    main()
