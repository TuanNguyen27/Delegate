# src/data/gsm8k_loader.py

from datasets import load_dataset
import random
from omegaconf import OmegaConf

def load_gsm8k(cfg):
    random.seed(cfg.seed) # Reproducibility
    subset = cfg.get("subset", None)
    n_samples = cfg.get("n_samples", None)

    # Load dataset
    if subset:
        ds = load_dataset(cfg.dataset, subset, split=cfg.split)
    else:
        ds = load_dataset(cfg.dataset, split=cfg.split)

    # Sampling
    if n_samples:
        num_samples = min(n_samples, len(ds)) # Ensures n_samples does not exceed number of rows
        indices = random.sample(range(len(ds)), num_samples) # Pick num_samples random rows (by index) from the dataset
        ds = ds.select(indices)

    for i, row in enumerate(ds):
        yield {
            "id": i,
            "question": row["question"],
            "answer": row["answer"],
        }