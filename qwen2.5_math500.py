# qwen2.5_math500.py
import torch, re, time
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"

_BOXED = re.compile(r"\\boxed\{([^}]+)\}")
_LAST_NUM = re.compile(r"(?<!\d)(-?\d+(?:/\d+)?)(?!\d)")

def extract_answer(text: str) -> str:
    m = _BOXED.findall(text)
    if m:
        return m[-1].strip()
    m = _LAST_NUM.findall(text)
    return m[-1] if m else ""

def device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    elif torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32

def load_qwen():
    device, dtype = device_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto" if device != "cpu" else None,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.padding_side = "left"
    return model, tok

def solve_with_qwen(model, tok, prompt: str, mode="cot", max_new_tokens=256):
    sys = (
        "Please reason step by step, and put your final answer within \\boxed{}."
        if mode == "cot"
        else "Answer concisely and only put the final answer within \\boxed{}."
    )
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": prompt}
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tok.eos_token_id,
        )
    latency = time.time() - t0

    # strip the prompt portion
    gen = tok.batch_decode(out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    ans = extract_answer(gen)
    return {"answer": ans, "reasoning": gen, "latency_sec": latency}

if __name__ == "__main__":
    model, tok = load_qwen()
    prompt = "Find the value of x that satisfies 4x + 5 = 6x + 7."
    out = solve_with_qwen(model, tok, prompt, mode="cot", max_new_tokens=256)
    print(out)
