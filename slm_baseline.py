import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import time
import re
import os
import random
import numpy as np

# ---------------------------
# 1) Reproducibility
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ---------------------------
# 2) Device setup
# ---------------------------
def setup_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    elif torch.backends.mps.is_available():
        return "mps", torch.float32  # safer for MPS
    else:
        return "cpu", torch.float32

# ---------------------------
# 3) Model loading with fallback
# ---------------------------
def load_model_safe(model_id, device, dtype):
    try:
        print(f"→ Loading model on {device} ({dtype})")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device if device != "cpu" else None,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"✓ Model loaded on {device}")
        return model, tokenizer, device
    except Exception as e:
        print(f"✗ Failed on {device}: {e}")
        print("→ Falling back to CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=None,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return model, tokenizer, "cpu"

# ---------------------------
# 4) AIME answer normalization & extraction
# ---------------------------
def normalize_aime(ans: str) -> str:
    digits = re.findall(r"\d+", ans)
    if not digits:
        return ""
    return digits[-1].zfill(3)  # AIME expects 3-digit integer

def extract_answer(text: str, question: str = "") -> str:
    # Drop echoed question, if present
    if question and question in text:
        text = text.split(question)[-1]

    print(f"🔧 Extracting from: '{text[-200:]}'")  # Debug: show last 200 chars
    
    # Look for "The answer is XXX" format first
    m = re.findall(r"(?:the answer is|answer is)\s*(\d+)", text, re.IGNORECASE)
    if m:
        result = normalize_aime(m[-1])
        print(f"🎯 Found 'answer is' pattern: {result}")
        return result

    # Prefer LaTeX boxed answer
    m = re.findall(r"\\boxed{(\d+)}", text)
    if m:
        result = normalize_aime(m[-1])
        print(f"🎯 Found boxed pattern: {result}")
        return result

    # Common phrasings
    m = re.findall(r"(?:answer:|equals?|=)\s*(\d+)", text, re.IGNORECASE)
    if m:
        result = normalize_aime(m[-1])
        print(f"🎯 Found common pattern: {result}")
        return result

    # Look for numbers near the end
    words = text.split()
    for i in range(len(words)-1, max(len(words)-10, 0), -1):
        if re.match(r'^\d{1,3}

# ---------------------------
# 5) Main evaluation loop
# ---------------------------
def main():
    model_id = "microsoft/phi-4-mini-reasoning"
    device, dtype = setup_device()
    model, tokenizer, device = load_model_safe(model_id, device, dtype)

    # Load dataset
    csv_path = "AIME/AIME_test.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Dataset not found: {csv_path}")
        return

    test_df = pd.read_csv(csv_path)
    print(f"✓ Loaded dataset with {len(test_df)} samples")

    required_cols = ["Question", "Answer", "Problem Number"]
    if not all(col in test_df.columns for col in required_cols):
        print(f"❌ Missing columns. Found: {list(test_df.columns)}")
        return

    latencies, results = [], []
    correct, total = 0, 0

    print(f"\n🚀 Starting evaluation on {device}")
    print("=" * 60)

    for _, row in test_df.iterrows():
        question = str(row["Question"]).strip()
        gold_raw = str(row["Answer"]).strip()
        gold = normalize_aime(gold_raw)
        prob_num = row["Problem Number"]

        # FIXED: Proper indentation for messages creation
        messages = [
            {"role": "system", "content": "You are an expert competition mathematician."},
            {"role": "user", "content": f"Solve the following AIME problem:\n\n{question}\n\n"
                                        "Think carefully step by step. Give your final answer in the end "
                                        "in the format: The answer is XXX (3-digit integer)."}
        ]

        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            start = time.time()
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,  # Increased for better reasoning
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            end = time.time()

            latency = end - start
            latencies.append(latency)

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # DEBUG: Show what the model actually generated
            generated_part = response[len(prompt):].strip() if len(response) > len(prompt) else response
            print(f"\n🔍 Q{prob_num} Generated text:")
            print(f"'{generated_part[:300]}{'...' if len(generated_part) > 300 else ''}'")
            
            pred = extract_answer(response, question)

            is_correct = (pred == gold)
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "Problem_Number": prob_num,
                "Gold": gold,
                "Pred": pred,
                "Correct": is_correct,
                "Latency": latency,
                "Raw_Response": response[:200] + "..." if len(response) > 200 else response
            })

            mark = "✓" if is_correct else "✗"
            print(f"{mark} Q{prob_num} | Pred={pred or '∅'} | Gold={gold} | {latency:.2f}s")
            print("-" * 40)

        except Exception as e:
            print(f"✗ Q{prob_num} | Error: {e}")
            results.append({
                "Problem_Number": prob_num,
                "Gold": gold,
                "Pred": "ERROR",
                "Correct": False,
                "Latency": 0.0,
                "Raw_Response": f"Error: {e}"
            })
            total += 1
            latencies.append(0.0)

    # ---------------------------
    # 6) Metrics
    # ---------------------------
    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
    total_time = sum(latencies)
    acc = (correct / total * 100.0) if total else 0.0

    print("\n" + "=" * 60)
    print("📊 BASELINE EVALUATION")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Device: {device} ({dtype})")
    print(f"Samples: {total}")
    print(f"Accuracy: {acc:.2f}% ({correct}/{total})")
    print(f"Avg latency: {avg_latency:.3f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {total/total_time:.2f} samples/s" if total_time > 0 else "N/A")

    # ---------------------------
    # 7) Save results
    # ---------------------------
    out_file = "baseline_results.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"\n💾 Results saved to {out_file}")

    wrong = [r for r in results if not r["Correct"]]
    if wrong:
        print("\n❌ Sample incorrect predictions:")
        for r in wrong[:3]:
            print(f"  Q{r['Problem_Number']}: Gold={r['Gold']} | Pred={r['Pred']}")

if __name__ == "__main__":
    main(), words[i]):
            result = normalize_aime(words[i])
            print(f"🎯 Found end number: {result}")
            return result

    # Fallback: any number
    digits = re.findall(r"\d+", text)
    if digits:
        result = normalize_aime(digits[-1])
        print(f"🎯 Fallback number: {result}")
        return result

    print(f"❌ No number found")
    return ""

# ---------------------------
# 5) Main evaluation loop
# ---------------------------
def main():
    model_id = "microsoft/phi-4-mini-reasoning"
    device, dtype = setup_device()
    model, tokenizer, device = load_model_safe(model_id, device, dtype)

    # Load dataset
    csv_path = "AIME/AIME_test.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Dataset not found: {csv_path}")
        return

    test_df = pd.read_csv(csv_path)
    print(f"✓ Loaded dataset with {len(test_df)} samples")

    required_cols = ["Question", "Answer", "Problem Number"]
    if not all(col in test_df.columns for col in required_cols):
        print(f"❌ Missing columns. Found: {list(test_df.columns)}")
        return

    latencies, results = [], []
    correct, total = 0, 0

    print(f"\n🚀 Starting evaluation on {device}")
    print("=" * 60)

    for _, row in test_df.iterrows():
        question = str(row["Question"]).strip()
        gold_raw = str(row["Answer"]).strip()
        gold = normalize_aime(gold_raw)
        prob_num = row["Problem Number"]

        # FIXED: Proper indentation for messages creation
        messages = [
            {"role": "system", "content": "You are an expert competition mathematician."},
            {"role": "user", "content": f"Solve the following AIME problem:\n\n{question}\n\n"
                                        "Think carefully step by step. Give your final answer in the end "
                                        "in the format: The answer is XXX (3-digit integer)."}
        ]

        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            start = time.time()
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,  # Increased for better reasoning
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            end = time.time()

            latency = end - start
            latencies.append(latency)

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # DEBUG: Show what the model actually generated
            generated_part = response[len(prompt):].strip() if len(response) > len(prompt) else response
            print(f"\n🔍 Q{prob_num} Generated text:")
            print(f"'{generated_part[:300]}{'...' if len(generated_part) > 300 else ''}'")
            
            pred = extract_answer(response, question)

            is_correct = (pred == gold)
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "Problem_Number": prob_num,
                "Gold": gold,
                "Pred": pred,
                "Correct": is_correct,
                "Latency": latency,
                "Raw_Response": response[:200] + "..." if len(response) > 200 else response
            })

            mark = "✓" if is_correct else "✗"
            print(f"{mark} Q{prob_num} | Pred={pred or '∅'} | Gold={gold} | {latency:.2f}s")
            print("-" * 40)

        except Exception as e:
            print(f"✗ Q{prob_num} | Error: {e}")
            results.append({
                "Problem_Number": prob_num,
                "Gold": gold,
                "Pred": "ERROR",
                "Correct": False,
                "Latency": 0.0,
                "Raw_Response": f"Error: {e}"
            })
            total += 1
            latencies.append(0.0)

    # ---------------------------
    # 6) Metrics
    # ---------------------------
    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
    total_time = sum(latencies)
    acc = (correct / total * 100.0) if total else 0.0

    print("\n" + "=" * 60)
    print("📊 BASELINE EVALUATION")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Device: {device} ({dtype})")
    print(f"Samples: {total}")
    print(f"Accuracy: {acc:.2f}% ({correct}/{total})")
    print(f"Avg latency: {avg_latency:.3f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {total/total_time:.2f} samples/s" if total_time > 0 else "N/A")

    # ---------------------------
    # 7) Save results
    # ---------------------------
    out_file = "baseline_results.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"\n💾 Results saved to {out_file}")

    wrong = [r for r in results if not r["Correct"]]
    if wrong:
        print("\n❌ Sample incorrect predictions:")
        for r in wrong[:3]:
            print(f"  Q{r['Problem_Number']}: Gold={r['Gold']} | Pred={r['Pred']}")

if __name__ == "__main__":
    main()