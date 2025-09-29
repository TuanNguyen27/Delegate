# slm_test.py

import os, time, re, random, numpy as np, pandas as pd
from dotenv import load_dotenv
from google import genai

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
    m = re.findall(r"(?:the answer is|answer is)\s*(\d+)", text, re.IGNORECASE)
    if m: return m[-1]
    m = re.findall(r"\d+", text)
    if m: return m[-1]
    return ""

# ---------------------------
# Main
# ---------------------------
def main():
    load_dotenv()
    api_key = os.getenv("GEMMA_API_KEY")
    if not api_key:
        raise ValueError("❌ GEMMA_API_KEY not found in .env")

    client = genai.Client(api_key=api_key)
    set_seed(42)

    # Load test split
    df = pd.read_csv("math500/test.csv")
    print(f"✓ Loaded {len(df)} test problems")

    results, latencies = [], []
    correct = 0

    for i, row in df.iterrows():
        problem, answer = row["problem"], row["answer"]

        prompt = f"""
You are a math tutor.
Solve the following problem step by step.
At the very end, give the answer in the format: 'The answer is <number>'.

Problem:
{problem}
"""

        start = time.time()
        response = client.models.generate_content(
            model="gemma-3-4b-it",   # your SLM
            contents=prompt,
            config={"max_output_tokens": 800}
        )
        latency = time.time() - start
        latencies.append(latency)

        output = response.text
        pred = extract_answer(output)

        gold_num = re.findall(r"\d+", str(answer))
        gold_num = gold_num[-1] if gold_num else ""
        is_correct = (pred == gold_num)
        if is_correct: correct += 1

        results.append({
            "id": row["unique_id"],
            "problem": problem,
            "gold": gold_num,
            "pred": pred,
            "correct": is_correct,
            "latency": latency,
            "raw": output[:200] + "..." if len(output) > 200 else output,
        })

        mark = "✓" if is_correct else "✗"
        print(f"{mark} {row['unique_id']} | Pred={pred or '∅'} | Gold={gold_num} | {latency:.2f}s")

    acc = correct / len(df) * 100 if len(df) else 0
    avg_latency = np.mean(latencies) if latencies else 0

    print("\n📊 Gemma-3-4B-it on MATH-500 (test)")
    print(f"Samples: {len(df)}")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Avg latency: {avg_latency:.2f}s")

    os.makedirs("Outputs/math500", exist_ok=True)
    out_file = "Outputs/math500/math500_slm_test.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"💾 Saved results to {out_file}")

if __name__ == "__main__":
    main()
