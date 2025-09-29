# make_router_dataset.py
import os, re, time, random, numpy as np, pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import OmegaConf
from src.data.gsm8k_loader import load_gsm8k

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def setup_device():
    if torch.cuda.is_available(): return "cuda", torch.float16
    elif torch.backends.mps.is_available(): return "mps", torch.float32
    else: return "cpu", torch.float32

def load_model(model_id, device, dtype):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device if device != "cpu" else None,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    return model, tok

def extract_answer(text):
    m = re.findall(r"(?:the answer is|answer is)\s*(\d+)", text, re.IGNORECASE)
    if m: return m[-1]
    m = re.findall(r"\d+", text)
    if m: return m[-1]
    return ""

def main():
    cfg = OmegaConf.load("configs/gsm8k.yaml")
    set_seed(cfg.seed)

    device, dtype = setup_device()
    model_id = "microsoft/phi-4-mini-instruct"
    print(f"→ Loading {model_id} on {device} ({dtype})")
    model, tok = load_model(model_id, device, dtype)

    # Load dataset (1700 samples)
    data = list(load_gsm8k(cfg))
    print(f"✓ Loaded {len(data)} samples")

    rows = []
    for ex in data:
        q, gold = ex["question"], ex["answer"]

        prompt = tok.apply_chat_template(
            [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": f"{q}\n\nGive the answer in format: 'The answer is <number>'."},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )

        response = tok.decode(out[0], skip_special_tokens=True)
        pred = extract_answer(response)

        # Normalize gold
        gold_num = re.findall(r"\d+", gold)
        gold_num = gold_num[-1] if gold_num else ""

        correct = (pred == gold_num)
        label = 1 if correct else 0

        rows.append({
            "id": ex["id"],
            "question": q,
            "gold": gold_num,
            "pred": pred,
            "label": label,   # 1 if SLM solved, 0 otherwise
            "raw": response,
        })

    # Save router dataset
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_file = os.path.join(cfg.output_dir, "router_train.csv")
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print(f"💾 Router dataset saved to {out_file}")

if __name__ == "__main__":
    main()
