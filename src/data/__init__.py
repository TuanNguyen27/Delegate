# src/data/__init__.py

from .gsm8k_loader import load_gsm8k
from .simpleqa_loader import load_simpleqa

def load_data(cfg):
    if cfg.dataset == "openai/gsm8k": # Gsm8k
        return load_gsm8k(cfg)
    elif cfg.dataset == "simpleqa": # SimpleQA
        return load_simpleqa(cfg)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")
