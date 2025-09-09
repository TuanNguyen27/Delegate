from datasets import load_dataset
import random

def load_gsm8k(cfg):
    # Reproducibility
    seed = getattr(cfg, "seed", 123)
    random.seed(seed)

    # Load HuggingFace dataset
    subset = getattr(cfg, "subset", None)
    if subset:
        ds = load_dataset(cfg.dataset, subset, split=cfg.split)
    else:
        ds = load_dataset(cfg.dataset, split=cfg.split)

    for i, row in enumerate(ds):
        yield {
            "id": i,
            "question": row["question"],
            "answer": row["answer"],
        }
