# src/models/slm.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

_tokenizer = None
_model = None

def load_slm(model_name: str):
    """Load a Hugging Face model lazily (only once)."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    return _tokenizer, _model

def slm_answer(question: str, cfg) -> str:
    """Generate an answer using the chosen SLM."""
    tokenizer, model = load_slm(cfg.slm.model_name)
    inputs = tokenizer(question, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.slm.max_new_tokens,
            do_sample=False,  # deterministic
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
