# slm_baseline.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time, re, random, numpy as np
import pandas as pd
from omegaconf import OmegaConf
from src.data.gsm8k_loader import load_gsm8k
import os

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
# Device setup
# ---------------------------
def setup_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    elif torch.backends.mps.is_available():
        return "mps", torch.float32
    else:
        return "cpu", torch.float32

# ---------------------------
# Model loading
# ---------------------------
def load_model(model_id, device, dtype):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device if device != "cpu" else None,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

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
# Main evaluation
# ---------------------------
def main():
    cfg = OmegaConf.load("configs/gsm8k.yaml")
    set_seed(cfg.seed)

    device, dtype = setup_device()
    model_id = "microsoft/phi-4-mini-instruct"
    print(f"→ Loading {model_id} on {device} ({dtype})")
    model, tokenizer = load_model(model_id, device, dtype)
    
    print("🧠 Model ID:", model_id)
    print("Model class:", model.__class__.__name__)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()) // 1_000_000, "M")
    print("HF Config:", model.config)

    # Load dataset
    data = list(load_gsm8k(cfg))
    print(f"✓ Loaded {len(data)} GSM8k samples")

    results, latencies = [], []
    correct, total = 0, 0
    BATCH_SIZE = 8   # 🔥 adjust batch size if GPU memory allows

    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i : i + BATCH_SIZE]

        # Prepare prompts
        batch_prompts = []
        for ex in batch:
            q = ex["question"]
            messages = [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": f"{q}\n\nPlease think step by step and finish with 'The answer is <number>'."}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_prompts.append(prompt)

        # Tokenize batch
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)

        # Generate
        start = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,    # reduced token limit
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency = time.time() - start
        latencies.append(latency / len(batch))  # avg per sample

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Process results
        for ex, resp in zip(batch, decoded):
            gold_num = re.findall(r"\d+", ex["answer"])
            gold_num = gold_num[-1] if gold_num else ""
            pred = extract_answer(resp)

            is_correct = (pred == gold_num)
            if is_correct: correct += 1
            total += 1

            results.append({
                "id": ex["id"],
                "question": ex["question"],
                "gold": gold_num,
                "pred": pred,
                "correct": is_correct,
                "latency": latency / len(batch),
                "raw": resp[:200] + "..." if len(resp) > 200 else resp,
            })

            mark = "✓" if is_correct else "✗"
            print(f"{mark} Q{ex['id']} | Pred={pred or '∅'} | Gold={gold_num} | {latency/len(batch):.2f}s")

    # Metrics
    acc = correct / total * 100 if total else 0
    avg_latency = np.mean(latencies) if latencies else 0

    print("\n📊 RESULTS")
    print(f"Samples: {total}")
    print(f"Accuracy: {acc:.2f}% ({correct}/{total})")
    print(f"Avg latency: {avg_latency:.2f}s")

    # Save
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_file = os.path.join(cfg.output_dir, "gsm8k_SLM_baseline.csv")
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"💾 Saved results to {out_file}")

if __name__ == "__main__":
    main()
