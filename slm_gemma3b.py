# gemma_baseline_gsm8k.py
import os, time, re, random, numpy as np, pandas as pd
from dotenv import load_dotenv
from google import genai
from omegaconf import OmegaConf
from src.data.gsm8k_loader import load_gsm8k

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

    cfg = OmegaConf.load("configs/gsm8k.yaml")
    set_seed(cfg.seed)

    # Load dataset
    data = list(load_gsm8k(cfg))
    print(f"✓ Loaded {len(data)} GSM8k samples")

    results, latencies = [], []
    correct, total = 0, 0

    for ex in data:
        q, gold = ex["question"], ex["answer"]

        prompt = f"""
You are a math tutor.
Solve the problem step by step. 
At the very end, give the answer in the format: 'The answer is <number>'.

Question:
{q}
"""

        start = time.time()
        response = client.models.generate_content(
            model="gemma-3-4b-it",   # 🔥 switch to 27B later if you want
            contents=prompt
        )
        latency = time.time() - start
        latencies.append(latency)

        output = response.text
        pred = extract_answer(output)

        # Normalize gold
        gold_num = re.findall(r"\d+", gold)
        gold_num = gold_num[-1] if gold_num else ""

        is_correct = (pred == gold_num)
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "id": ex["id"],
            "question": q,
            "gold": gold_num,
            "pred": pred,
            "correct": is_correct,
            "latency": latency,
            "raw": output[:200] + "..." if len(output) > 200 else output,
        })

        mark = "✓" if is_correct else "✗"
        print(f"{mark} Q{ex['id']} | Pred={pred or '∅'} | Gold={gold_num} | {latency:.2f}s")

    # Metrics
    acc = correct / total * 100 if total else 0
    avg_latency = np.mean(latencies) if latencies else 0

    print("\n📊 Gemma-3-4B-it RESULTS")
    print(f"Samples: {total}")
    print(f"Accuracy: {acc:.2f}% ({correct}/{total})")
    print(f"Avg latency: {avg_latency:.2f}s")

    # Save
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_file = os.path.join(cfg.output_dir, "gsm8k_gemma_baseline.csv")
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"💾 Saved results to {out_file}")


if __name__ == "__main__":
    main()
